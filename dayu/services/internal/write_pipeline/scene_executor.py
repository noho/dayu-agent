"""Scene Prompt 执行协调模块。

该模块封装写作流水线中所有 Prompt/Agent 调用的执行、重试与结果解析逻辑。
契约准备由 SceneContractPreparer 完成；本模块不持有 Host 层引用，通过
注入的 ``contract_executor`` 协议执行已构建的 ExecutionContract。

helper 层 ``_run_scene_prompt_with_retry`` 显式区分两类失败路径：

1. 业务错误（``AppResult.errors`` 非空）：保留旧的「无历史重发」语义，
   重新提交相同 ``prompt_text``，因为上一次没产生有效产出；
2. 脏数据（``success_parser`` 抛 ``ValueError``）：当 scene 显式开启
   ``replay_on_parse_failure`` / ``replay_on_empty_output`` 时，
   helper 调用 ``ScenePromptContractExecutorProtocol.replay`` 在 Host 端
   带历史回放（在原对话末尾追加一条 user 消息），避免上下文 zigzag。

每个 attempt 内最多触发 1 次 replay；replay 仍失败视为本 attempt 用尽，
进下一个 attempt 时 helper 仍以 replay 起手（只要句柄仍在），保证
「一旦切到 replay 路径就不回退到无历史重发」。

.. note::
    SceneAgentCreationError 已迁移至 scene_contract_preparer 模块，
    此处仅 re-export 以保持向后兼容。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable
from types import SimpleNamespace
from typing import Any, Callable, Protocol, TypeVar

from dayu.contracts.agent_execution import ExecutionContract, ReplayHandle
from dayu.contracts.cancellation import CancelledError
from dayu.contracts.events import AppEvent, AppEventType, AppResult, extract_cancel_reason
from dayu.execution.runtime_config import OpenAIRunnerRuntimeConfig
from dayu.log import Log
from dayu.services.internal.write_pipeline.audit_formatting import (
    _extract_markdown_content,
    parse_markdown_scene_output,
)
from dayu.services.internal.write_pipeline.audit_formatting import (
    _extract_json_text,
    _extract_markdown_content,
)
from dayu.services.internal.write_pipeline.audit_rules import (
    ConfirmOutputError,
    EmptyOutputError,
    RepairOutputError,
    _parse_audit_decision,
    _parse_evidence_confirmation_result,
)
from dayu.services.internal.write_pipeline.company_facets import parse_company_facets
from dayu.services.internal.write_pipeline.models import (
    AuditDecision,
    CompanyFacetProfile,
    EvidenceConfirmationResult,
    WriteRunConfig,
)
from dayu.services.internal.write_pipeline.repair_executor import _parse_repair_plan
from dayu.services.internal.write_pipeline.scene_contract_preparer import (
    SceneAgentCreationError,
    SceneContractPreparer,
)
from dayu.services.scene_execution_acceptance import AcceptedSceneExecution


class PromptAgentProtocol(Protocol):
    """测试缝使用的最小 Prompt Agent 协议。"""

    def stream(
        self,
        prepared_scene: AcceptedSceneExecution,
        turn_input: SimpleNamespace,
    ) -> AsyncIterator[AppEvent]:
        """按序输出应用层事件流。"""

        ...


class ScenePromptContractExecutorProtocol(Protocol):
    """ScenePromptRunner 注入的 Host 执行入口协议。

    该协议刻意只暴露 helper 真正会调用的两条路径，对应
    ``Host.run_agent_and_wait_replayable`` / ``Host.replay_agent_and_wait``，
    不再回退到无历史的 ``run_agent_and_wait``——helper 在每次首发时也通过
    replayable 入口拿到 ReplayHandle，由本协议保持 Host 与 Service 边界稳定。
    """

    async def run_replayable(
        self,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        """首发执行：返回结果与可用于带历史回放的不透明句柄。"""

        ...

    async def replay(
        self,
        handle: ReplayHandle,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        """带历史回放：消费上一次 ``run_replayable`` 颁发的句柄再跑一次。"""

        ...

    def discard(self, handle: ReplayHandle) -> None:
        """释放未消费的 replay 句柄对应的 Host 内存状态。

        Args:
            handle: 待释放的句柄；若已不存在则静默跳过。
        """

        ...


MODULE = "APP.WRITE_PIPELINE"

_LLM_RETRY_LIMIT = 1
_LLM_RETRY_DELAY_SECONDS = 3
_DEFAULT_PARSE_FAILURE_REPLAY_USER_MESSAGE = (
    "上一轮输出无法解析，请直接基于已有取证与历史，按要求格式输出最终结果，不要再调用任何工具。"
)
_DEFAULT_EMPTY_OUTPUT_REPLAY_USER_MESSAGE = (
    "上一轮没有输出有效的章节正文，请直接基于已有取证与历史，输出完整的 ```markdown 代码块，不要再调用任何工具。"
)

_RetryResult = TypeVar("_RetryResult")


def _resolve_scene_tool_timeout_seconds(prepared_scene: AcceptedSceneExecution) -> float | None:
    """解析 scene 生效的工具超时秒数。

    Args:
        prepared_scene: 已解析的 scene 执行信息。

    Returns:
        工具超时秒数；当前 runner 不支持时返回 ``None``。

    Raises:
        无。
    """

    runner_running_config = prepared_scene.resolved_execution_options.runner_running_config
    if isinstance(runner_running_config, OpenAIRunnerRuntimeConfig):
        return runner_running_config.tool_timeout_seconds
    return None


def _build_scene_dispatch_debug_message(*, prepared_scene: AcceptedSceneExecution) -> str:
    """构造 scene 调用开始调试日志。

    Args:
        prepared_scene: 已解析的 scene 执行信息。

    Returns:
        统一格式的调试日志文本。

    Raises:
        无。
    """

    return (
        "开始执行 scene contract: "
        f"scene_name={prepared_scene.scene_name}, "
        f"model_name={prepared_scene.accepted_execution_spec.model.model_name}, "
        f"temperature={prepared_scene.resolved_temperature}, "
        f"max_iterations={prepared_scene.resolved_execution_options.agent_running_config.max_iterations}, "
        f"tool_timeout_seconds={_resolve_scene_tool_timeout_seconds(prepared_scene)}, "
        f"resumable={prepared_scene.default_resumable}"
    )


def _build_scene_result_debug_message(
    *,
    prepared_scene: AcceptedSceneExecution,
    result: AppResult,
) -> str:
    """构造 scene 返回结果调试日志。

    Args:
        prepared_scene: 已解析的 scene 执行信息。
        result: 当前调用返回结果。

    Returns:
        统一格式的调试日志文本。

    Raises:
        无。
    """

    return (
        "scene contract 执行完成: "
        f"scene_name={prepared_scene.scene_name}, "
        f"errors_count={len(result.errors)}, warnings_count={len(result.warnings)}, "
        f"degraded={result.degraded}, filtered={result.filtered}"
    )


class ScenePromptRunner:
    """Scene Prompt 执行协调器。

    职责：
    - 通过 SceneContractPreparer 获取 scene 执行信息和构建契约。
    - 通过注入的 ``contract_executor`` 协议执行契约（不直接引用 Host 类型）。
    - 管理重试策略和结果解析，区分业务错误与脏数据两条路径。
    """

    def __init__(
        self,
        *,
        preparer: SceneContractPreparer,
        contract_executor: ScenePromptContractExecutorProtocol,
        write_config: WriteRunConfig,
        prompt_agent: PromptAgentProtocol | None = None,
    ) -> None:
        """初始化 Scene Prompt 执行协调器。

        Args:
            preparer: Scene 契约准备器（纯 Service 层）。
            contract_executor: 契约执行协议，暴露 ``run_replayable`` 与
                ``replay`` 两条入口；由上层在组装时绑定到 Host 执行器方法。
            write_config: 写作运行配置。
            prompt_agent: 测试模式下的替代 Agent。
        """

        self._preparer = preparer
        self._contract_executor = contract_executor
        self._write_config = write_config
        self._prompt_agent = prompt_agent

    # ------------------------------------------------------------------
    # 内部执行基础设施
    # ------------------------------------------------------------------

    async def _collect_prompt_result(
        self,
        *,
        prepared_scene: AcceptedSceneExecution,
        prompt_text: str,
    ) -> tuple[AppResult, ReplayHandle | None]:
        """执行一次单轮 Agent 子执行，并返回可选回放句柄。

        Args:
            prepared_scene: 已解析的 scene 执行信息。
            prompt_text: 当前轮用户输入。

        Returns:
            ``(AppResult, ReplayHandle | None)`` 二元组；测试桩路径不颁发句柄。
        """

        execution_contract = self._preparer.build_execution_contract(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
        )
        if self._prompt_agent is not None:
            Log.debug(
                _build_scene_dispatch_debug_message(prepared_scene=prepared_scene),
                module=MODULE,
            )
            result = await self._collect_prompt_result_via_test_seam(
                prepared_scene=prepared_scene,
                prompt_text=prompt_text,
            )
            Log.debug(
                _build_scene_result_debug_message(prepared_scene=prepared_scene, result=result),
                module=MODULE,
            )
            return result, None
        Log.debug(
            _build_scene_dispatch_debug_message(prepared_scene=prepared_scene),
            module=MODULE,
        )
        result, handle = await self._contract_executor.run_replayable(execution_contract)
        Log.debug(
            _build_scene_result_debug_message(prepared_scene=prepared_scene, result=result),
            module=MODULE,
        )
        return result, handle

    async def _collect_prompt_result_via_replay(
        self,
        *,
        prepared_scene: AcceptedSceneExecution,
        replay_user_message: str,
        handle: ReplayHandle,
    ) -> tuple[AppResult, ReplayHandle]:
        """通过 ``Host.replay_agent_and_wait`` 带历史回放执行一次 scene。

        Args:
            prepared_scene: 已解析的 scene 执行信息。
            replay_user_message: 追加在原对话末尾的 user 消息。
            handle: 上一次首发返回的回放句柄。

        Returns:
            ``(AppResult, ReplayHandle)`` 二元组；新句柄可继续追加 replay。
        """

        execution_contract = self._preparer.build_execution_contract(
            prepared_scene=prepared_scene,
            prompt_text=replay_user_message,
            replay_from=handle,
            replay_disable_tools=True,
        )
        Log.debug(
            _build_scene_dispatch_debug_message(prepared_scene=prepared_scene),
            module=MODULE,
        )
        result, new_handle = await self._contract_executor.replay(handle, execution_contract)
        Log.debug(
            _build_scene_result_debug_message(prepared_scene=prepared_scene, result=result),
            module=MODULE,
        )
        return result, new_handle

    def run_prepared_scene_prompt(self, *, prepared_scene: AcceptedSceneExecution, prompt_text: str) -> AppResult:
        """同步执行一次单轮 Agent 子执行。

        若当前已处于事件循环中，则复用已有循环；否则创建新循环执行。
        本入口仅返回 ``AppResult``，丢弃 replay 句柄；replay 路径必须使用
        ``_run_scene_prompt_with_retry``。

        Args:
            prepared_scene: 已解析的 scene 执行信息。
            prompt_text: 当前轮用户输入。

        Returns:
            聚合后的应用层结果。

        Raises:
            RuntimeError: Agent 执行异常时抛出。
        """

        coro = self._collect_prompt_result(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            raise RuntimeError(
                "run_prepared_scene_prompt 不支持在已有事件循环中同步调用，"
                "请使用 _collect_prompt_result 的 async 版本"
            )
        result, _handle = asyncio.run(coro)
        # 同步入口不再需要 replay 状态，立即释放 Host stash 避免内存堆积。
        self._discard_replay_handle_if_present(_handle)
        return result

    async def _collect_prompt_result_via_test_seam(
        self,
        *,
        prepared_scene: AcceptedSceneExecution,
        prompt_text: str,
    ) -> AppResult:
        """通过测试桩 Prompt Agent 汇总结果。

        Args:
            prepared_scene: 已解析的 scene 执行信息。
            prompt_text: 当前轮用户输入。

        Returns:
            聚合后的应用层结果。

        Raises:
            CancelledError: 当执行被取消时抛出。
            ValueError: 当 prompt 文本为空时抛出。
        """

        user_message = str(prompt_text or "").strip()
        if not user_message:
            raise ValueError("写作流水线 prompt 不能为空")
        prompt_agent = self._prompt_agent
        if prompt_agent is None:
            raise RuntimeError("测试 Prompt Agent 未设置")
        content = ""
        degraded = False
        filtered = False
        warnings: list[str] = []
        errors: list[str] = []
        async for event in prompt_agent.stream(
            prepared_scene,
            SimpleNamespace(user_text=user_message, scene_request=self._write_config),
        ):
            if event.type == AppEventType.WARNING:
                warnings.append(str(event.payload))
                continue
            if event.type == AppEventType.ERROR:
                errors.append(str(event.payload))
                continue
            if event.type == AppEventType.CANCELLED:
                raise CancelledError(_build_cancelled_prompt_message(event.payload))
            if event.type == AppEventType.FINAL_ANSWER:
                payload = event.payload if isinstance(event.payload, dict) else {"content": str(event.payload)}
                content = str(payload.get("content") or "")
                degraded = bool(payload.get("degraded", False))
                filtered = bool(payload.get("filtered", False))
        return AppResult(content=content, errors=errors, warnings=warnings, degraded=degraded, filtered=filtered)

    # ------------------------------------------------------------------
    # 高层 prompt 调用方法
    # ------------------------------------------------------------------

    def _run_scene_prompt_with_retry(
        self,
        *,
        prepared_scene: AcceptedSceneExecution,
        prompt_text: str,
        execution_error_message: str,
        execution_retry_message: str,
        success_parser: Callable[[str], _RetryResult],
        parse_retry_message: str | None = None,
        parse_error_builder: Callable[[str, ValueError], RuntimeError] | None = None,
        replay_on_parse_failure: bool = False,
        replay_on_empty_output: bool = False,
        parse_replay_user_message: str | None = None,
    ) -> _RetryResult:
        """按统一重试策略执行 scene prompt 并解析结果。

        helper 显式收口三类失败路径：

        - 业务错误（``AppResult.errors`` 非空）→ 保留旧的「无历史重发」；
        - 解析失败（``success_parser`` 抛 ``ValueError``）→
          若开启 replay，在 Host 端带历史回放（每 attempt 最多 1 次 replay）；
          replay 仍失败视为本 attempt 用尽，下一 attempt 仍以 replay 起手；
          未开启 replay 则保留旧的「无历史重发 + parse_error_builder」语义；
        - 重试预算耗尽 → 抛出 ``parse_error_builder`` 构造的异常或
          ``execution_error_message`` 包装的运行时异常。

        Args:
            prepared_scene: 已解析的 scene 执行信息。
            prompt_text: 当前轮用户输入。
            execution_error_message: 执行失败时最终抛错前缀。
            execution_retry_message: 执行失败时的重试日志前缀。
            success_parser: 成功返回文本后的解析函数。
            parse_retry_message: 解析失败时的重试日志前缀；为 ``None`` 表示不处理解析重试。
            parse_error_builder: 解析失败时的异常构造器；为 ``None`` 表示直接透传解析异常。
            replay_on_parse_failure: 解析失败时是否触发带历史 replay 兜底。
            replay_on_empty_output: 输出空白时是否触发带历史 replay 兜底；
                语义上与 ``replay_on_parse_failure`` 等价（``parse_markdown_scene_output``
                通过 ``ValueError`` 抛出空输出），单独保留是为了让 Markdown
                类 scene 在装配处可读性更高。
            parse_replay_user_message: 触发 replay 时追加到对话末尾的 user 消息；
                ``None`` 时使用模块默认文案（见
                ``_DEFAULT_PARSE_FAILURE_REPLAY_USER_MESSAGE`` /
                ``_DEFAULT_EMPTY_OUTPUT_REPLAY_USER_MESSAGE``）。

        Returns:
            解析后的结构化结果。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当执行失败或解析失败且重试耗尽时抛出。
            ValueError: 当未配置解析错误构造器且解析函数本身抛出时透传。
        """

        replay_enabled = bool(replay_on_parse_failure or replay_on_empty_output)
        if parse_replay_user_message is not None:
            resolved_replay_user_message = parse_replay_user_message
        elif replay_on_empty_output:
            resolved_replay_user_message = _DEFAULT_EMPTY_OUTPUT_REPLAY_USER_MESSAGE
        else:
            resolved_replay_user_message = _DEFAULT_PARSE_FAILURE_REPLAY_USER_MESSAGE

        last_error: RuntimeError | None = None
        # replay 起手机制：上一 attempt 的 replay 仍失败时，下一 attempt
        # 也以 replay 起手而不是回退「无历史重发」，避免上下文 zigzag。
        replay_startup_handle: ReplayHandle | None = None

        for attempt in range(_LLM_RETRY_LIMIT + 1):
            try:
                if replay_enabled and replay_startup_handle is not None:
                    result, handle = asyncio.run(
                        self._collect_prompt_result_via_replay(
                            prepared_scene=prepared_scene,
                            replay_user_message=resolved_replay_user_message,
                            handle=replay_startup_handle,
                        )
                    )
                else:
                    result, handle = asyncio.run(
                        self._collect_prompt_result(
                            prepared_scene=prepared_scene,
                            prompt_text=prompt_text,
                        )
                    )
            except CancelledError:
                raise
            replay_startup_handle = None

            if result.errors:
                last_error = RuntimeError(f"{execution_error_message}: {result.errors}")
                # 失败 attempt 的 handle 不会被任何路径消费，立即释放避免 stash 堆积。
                self._discard_replay_handle_if_present(handle)
                if attempt < _LLM_RETRY_LIMIT:
                    self._sleep_before_prompt_retry(
                        retry_message=execution_retry_message,
                        attempt=attempt,
                        detail_text=str(result.errors),
                    )
                    continue
                raise last_error

            raw_text = str(result.content)
            try:
                parsed = success_parser(raw_text)
            except ValueError as exc:
                if not replay_enabled:
                    # 不走 replay：handle 不再使用，释放避免 stash 堆积。
                    self._discard_replay_handle_if_present(handle)
                    if parse_error_builder is None or parse_retry_message is None:
                        raise
                    last_error = parse_error_builder(raw_text, exc)
                    if attempt < _LLM_RETRY_LIMIT:
                        self._sleep_before_prompt_retry(
                            retry_message=parse_retry_message,
                            attempt=attempt,
                            detail_text=str(exc),
                        )
                        continue
                    raise last_error from exc

                # replay 路径：每 attempt 最多 1 次 replay
                if handle is None:
                    if parse_error_builder is None or parse_retry_message is None:
                        raise
                    last_error = parse_error_builder(raw_text, exc)
                    if attempt < _LLM_RETRY_LIMIT:
                        self._sleep_before_prompt_retry(
                            retry_message=parse_retry_message,
                            attempt=attempt,
                            detail_text=str(exc),
                        )
                        continue
                    raise last_error from exc

                Log.warning(
                    f"{parse_retry_message or '场景输出解析失败'}，触发 Host replay 兜底: {exc}",
                    module=MODULE,
                )
                try:
                    replay_result, new_handle = asyncio.run(
                        self._collect_prompt_result_via_replay(
                            prepared_scene=prepared_scene,
                            replay_user_message=resolved_replay_user_message,
                            handle=handle,
                        )
                    )
                except CancelledError:
                    raise
                # 旧 handle 已被 replay() 消费出 stash，无需再 discard。

                if replay_result.errors:
                    last_error = RuntimeError(f"{execution_error_message}（replay 路径）: {replay_result.errors}")
                    if attempt < _LLM_RETRY_LIMIT:
                        # 下一 attempt 仍以 replay 起手，保留 new_handle。
                        replay_startup_handle = new_handle
                        self._sleep_before_prompt_retry(
                            retry_message=execution_retry_message,
                            attempt=attempt,
                            detail_text=str(replay_result.errors),
                        )
                        continue
                    # 已是最后一次 attempt，new_handle 不会再被消费，立即释放。
                    self._discard_replay_handle_if_present(new_handle)
                    raise last_error

                replay_raw_text = str(replay_result.content)
                try:
                    replay_parsed = success_parser(replay_raw_text)
                except ValueError as replay_exc:
                    builder = parse_error_builder
                    retry_msg = parse_retry_message or "场景输出 replay 后仍解析失败"
                    last_error = (
                        builder(replay_raw_text, replay_exc)
                        if builder is not None
                        else RuntimeError(f"{retry_msg}: {replay_exc}")
                    )
                    if attempt < _LLM_RETRY_LIMIT:
                        # 下一 attempt 仍以 replay 起手，保留 new_handle。
                        replay_startup_handle = new_handle
                        self._sleep_before_prompt_retry(
                            retry_message=retry_msg,
                            attempt=attempt,
                            detail_text=str(replay_exc),
                        )
                        continue
                    # 已是最后一次 attempt，new_handle 不会再被消费，立即释放。
                    self._discard_replay_handle_if_present(new_handle)
                    if builder is not None:
                        raise last_error from replay_exc
                    raise last_error
                else:
                    # replay 成功：释放新句柄对应的 stash，避免内存堆积。
                    self._discard_replay_handle_if_present(new_handle)
                    return replay_parsed
            else:
                # 首发即解析成功：释放未消费的 stash 句柄，避免内存堆积。
                self._discard_replay_handle_if_present(handle)
                return parsed

        raise self._build_unexpected_retry_error(last_error=last_error, operation=execution_retry_message)

    def _discard_replay_handle_if_present(self, handle: ReplayHandle | None) -> None:
        """释放未消费的 replay 句柄，避免 Host stash 内存泄漏。

        Args:
            handle: 待释放的句柄；为 ``None`` 时（测试桩路径）静默跳过。

        Returns:
            无。

        Raises:
            无。
        """

        if handle is None:
            return
        self._contract_executor.discard(handle)

    def _sleep_before_prompt_retry(self, *, retry_message: str, attempt: int, detail_text: str) -> None:
        """记录统一格式的 prompt 重试日志并等待固定退避时间。

        Args:
            retry_message: 当前失败场景的日志前缀。
            attempt: 当前已失败次数（从 0 开始）。
            detail_text: 失败细节文本。

        Returns:
            无。

        Raises:
            无。
        """

        Log.warning(
            f"{retry_message}，{_LLM_RETRY_DELAY_SECONDS}s 后重试 ({attempt + 1}/{_LLM_RETRY_LIMIT}): {detail_text}",
            module=MODULE,
        )
        time.sleep(_LLM_RETRY_DELAY_SECONDS)

    def _build_unexpected_retry_error(self, *, last_error: RuntimeError | None, operation: str) -> RuntimeError:
        """为理论上不可达的重试尾部分支构造稳定异常。

        Args:
            last_error: 循环内记录的最后一次错误。
            operation: 当前操作描述，用于兜底消息。

        Returns:
            最终应抛出的运行时异常。

        Raises:
            无。
        """

        if last_error is not None:
            return last_error
        return RuntimeError(f"{operation} 结束时未得到任何有效结果")

    def run_write_prompt(self, prompt_text: str) -> str:
        """调用写作 Agent 并提取 Markdown 正文。

        Args:
            prompt_text: 用户提示词。

        Returns:
            提取后的章节正文。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当 Agent 返回错误或输出仍为空时抛出。
        """

        prepared_scene = self._preparer.get_or_create_write_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="写作 Agent 执行失败",
            execution_retry_message="写作 Agent 调用失败",
            success_parser=parse_markdown_scene_output,
            parse_retry_message="写作 Agent 输出解析失败",
            parse_error_builder=_build_empty_output_error,
            replay_on_empty_output=True,
        )

    def run_infer_prompt(self, prompt_text: str) -> CompanyFacetProfile:
        """调用公司级 facet 归因 Agent 并解析结构化结果。

        Args:
            prompt_text: 用户提示词。

        Returns:
            公司级 facet 归因结果。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当 Agent 返回错误或输出解析失败时抛出。
        """

        prepared_scene = self._preparer.get_or_create_infer_scene()
        facet_catalog = self._preparer.get_company_facet_catalog()

        def _parse_infer_output(raw_text: str) -> CompanyFacetProfile:
            return parse_company_facets(raw_text, facet_catalog=facet_catalog)

        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="公司级 Facet 归因 Agent 执行失败",
            execution_retry_message="公司级 Facet 归因 Agent 调用失败",
            success_parser=_parse_infer_output,
            parse_retry_message="公司级 Facet 归因输出解析失败",
            parse_error_builder=_build_infer_output_error,
            replay_on_parse_failure=True,
        )

    def run_audit_prompt(self, prompt_text: str) -> AuditDecision:
        """调用审计 Agent 并解析结构化审计决策。

        Args:
            prompt_text: 审计 prompt 文本。

        Returns:
            结构化审计决策。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当 Agent 返回错误或重试耗尽时抛出。
        """

        prepared_scene = self._preparer.get_or_create_audit_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="审计 Agent 执行失败",
            execution_retry_message="审计 Agent 调用失败",
            success_parser=_parse_audit_output,
            parse_retry_message="审计 Agent 输出解析失败",
            parse_error_builder=_build_audit_output_error,
            replay_on_parse_failure=True,
        )

    def run_initial_chapter_prompt(self, *, prompt_name: str, prompt_text: str) -> str:
        """根据初始任务类型调用对应 Agent。

        Args:
            prompt_name: 初始 task prompt 名称。
            prompt_text: 已渲染的用户提示词。

        Returns:
            提取后的章节正文。

        Raises:
            RuntimeError: 当对应 Agent 调用失败时抛出。
            ValueError: 当 ``prompt_name`` 不受支持时抛出。
        """

        if prompt_name == "write_chapter":
            return self.run_write_prompt(prompt_text)
        if prompt_name == "fill_overview":
            return self.run_overview_prompt(prompt_text)
        if prompt_name == "write_research_decision":
            return self.run_decision_prompt(prompt_text)
        raise ValueError(f"不支持的初始章节 prompt: {prompt_name}")

    def run_decision_prompt(self, prompt_text: str) -> str:
        """调用研究决策综合 Agent 并提取 Markdown 正文。

        Args:
            prompt_text: 用户提示词。

        Returns:
            提取后的章节正文。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当 Agent 返回错误或输出仍为空时抛出。
        """

        prepared_scene = self._preparer.get_or_create_decision_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="研究决策综合 Agent 执行失败",
            execution_retry_message="研究决策综合 Agent 调用失败",
            success_parser=parse_markdown_scene_output,
            parse_retry_message="研究决策综合 Agent 输出解析失败",
            parse_error_builder=_build_empty_output_error,
            replay_on_empty_output=True,
        )

    def run_overview_prompt(self, prompt_text: str) -> str:
        """调用第0章封面页 Agent 并提取 Markdown 正文。

        Args:
            prompt_text: 用户提示词。

        Returns:
            提取后的章节正文。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当 Agent 返回错误或输出仍为空时抛出。
        """

        prepared_scene = self._preparer.get_or_create_overview_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="第0章概览 Agent 执行失败",
            execution_retry_message="第0章概览 Agent 调用失败",
            success_parser=parse_markdown_scene_output,
            parse_retry_message="第0章概览 Agent 输出解析失败",
            parse_error_builder=_build_empty_output_error,
            replay_on_empty_output=True,
        )

    def run_fix_prompt(self, prompt_text: str) -> str:
        """调用占位符补强 Agent 并提取 Markdown 正文。

        Args:
            prompt_text: 用户提示词。

        Returns:
            提取后的章节正文。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当 Agent 返回错误或输出仍为空时抛出。
        """

        prepared_scene = self._preparer.get_or_create_fix_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="占位符补强 Agent 执行失败",
            execution_retry_message="占位符补强 Agent 调用失败",
            success_parser=parse_markdown_scene_output,
            parse_retry_message="占位符补强 Agent 输出解析失败",
            parse_error_builder=_build_empty_output_error,
            replay_on_empty_output=True,
        )

    def run_regenerate_prompt(self, prompt_text: str) -> str:
        """调用整章重建 Agent 并提取 Markdown 正文。

        Args:
            prompt_text: 用户提示词。

        Returns:
            提取后的章节正文。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当 Agent 返回错误或输出仍为空时抛出。
        """

        prepared_scene = self._preparer.get_or_create_regenerate_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="整章重建 Agent 执行失败",
            execution_retry_message="整章重建 Agent 调用失败",
            success_parser=parse_markdown_scene_output,
            parse_retry_message="整章重建 Agent 输出解析失败",
            parse_error_builder=_build_empty_output_error,
            replay_on_empty_output=True,
        )

    def run_repair_prompt(self, prompt_text: str) -> tuple[dict[str, Any], str]:
        """调用 repair prompt 并解析局部补丁计划。

        LLM 返回异常或 JSON 解析失败时自动重试一次（延迟 _LLM_RETRY_DELAY_SECONDS 秒）。

        Args:
            prompt_text: repair prompt 文本。

        Returns:
            `(repair_plan, raw_text)` 二元组。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当重试耗尽仍失败时抛出。
        """

        prepared_scene = self._preparer.get_or_create_repair_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="repair Agent 执行失败",
            execution_retry_message="repair Agent 调用失败",
            success_parser=_parse_repair_result,
            parse_retry_message="repair 输出解析失败",
            parse_error_builder=_build_repair_output_error,
            replay_on_parse_failure=True,
        )

    def run_confirm_prompt(self, prompt_text: str) -> EvidenceConfirmationResult:
        """调用证据复核 Agent 并解析结构化确认结果。

        Args:
            prompt_text: 证据复核 prompt 文本。

        Returns:
            结构化证据确认结果。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当重试耗尽仍失败时抛出。
        """

        prepared_scene = self._preparer.get_or_create_confirm_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="证据复核 Agent 执行失败",
            execution_retry_message="证据复核 Agent 调用失败",
            success_parser=_parse_evidence_confirmation_result,
            parse_retry_message="证据复核输出解析失败",
            parse_error_builder=_build_confirm_output_error,
            replay_on_parse_failure=True,
        )


def _parse_audit_output(raw_text: str) -> AuditDecision:
    """审计 success_parser：脏数据时抛 ``ValueError``，触发 helper 的
    ``replay_on_parse_failure`` 兜底。

    设计意图：
    - JSON 不可解析 / 缺失（典型如 LLM 退化为自然语言解释）属于"脏数据"，
      项目既有 Host.replay 兜底机制要求 success_parser 抛 ``ValueError``，
      由 ``_run_scene_prompt_with_retry`` 进入 replay 路径让 Agent 自纠。
    - JSON 形态合法但内部字段不规范（缺 violations / pass 等）属于"业务退化"，
      由 ``_parse_audit_decision`` 内部规范化处理，仍返回有效 ``AuditDecision``，
      不应触发 replay；replay 在已是合法 JSON 上重试无意义。

    Args:
        raw_text: 审计 Agent 原始输出。

    Returns:
        规范化后的审计决策。

    Raises:
        ValueError: 当 ``raw_text`` 取不到 JSON 块或 JSON 解析失败时抛出，
            交由 helper 触发 Host.replay 兜底。
    """

    normalized = _extract_markdown_content(raw_text)
    json_text = _extract_json_text(normalized)
    if not json_text:
        raise ValueError("审计输出不是合法 JSON：未提取到 JSON 块")
    try:
        json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"审计输出不是合法 JSON：{exc}") from exc
    return _parse_audit_decision(raw_text)


def _build_audit_output_error(raw_text: str, error: ValueError) -> RuntimeError:
    """构造审计输出解析失败异常，用于 replay 仍失败后的最终上抛。

    Args:
        raw_text: 审计原始输出。
        error: 原始解析异常。

    Returns:
        带原始输出片段的 ``RuntimeError``，便于上层日志定位。

    Raises:
        无。
    """

    snippet = raw_text.strip().replace("\n", " ")
    if len(snippet) > 200:
        snippet = snippet[:200] + "..."
    return RuntimeError(f"审计输出解析失败: {error}; raw={snippet!r}")


def _parse_repair_result(raw_text: str) -> tuple[dict[str, Any], str]:
    """解析 repair 原始输出，并保留原始文本。

    Args:
        raw_text: repair scene 原始输出。

    Returns:
        `(repair_plan, raw_text)` 二元组。

    Raises:
        ValueError: 当 repair JSON 非法时抛出。
    """

    return _parse_repair_plan(raw_text), raw_text


def _build_repair_output_error(raw_text: str, error: ValueError) -> RuntimeError:
    """构造 repair 输出解析失败异常。

    Args:
        raw_text: repair 原始输出。
        error: 原始解析异常。

    Returns:
        带原始输出的 `RepairOutputError`。

    Raises:
        无。
    """

    return RepairOutputError(f"repair 输出非法: {error}", raw_output=raw_text)


def _build_confirm_output_error(raw_text: str, error: ValueError) -> RuntimeError:
    """构造证据复核输出解析失败异常。

    Args:
        raw_text: confirm 原始输出。
        error: 原始解析异常。

    Returns:
        带原始输出和解析信息的 `ConfirmOutputError`。

    Raises:
        无。
    """

    return ConfirmOutputError(
        f"证据复核输出非法: {error}",
        raw_output=raw_text,
        parse_error=str(error),
    )


def _build_empty_output_error(raw_text: str, error: ValueError) -> RuntimeError:
    """构造 markdown 类 scene 空输出异常。

    Args:
        raw_text: scene 原始输出。
        error: 原始 ``ValueError``（来自 ``parse_markdown_scene_output``）。

    Returns:
        统一的 ``EmptyOutputError``，便于上层落盘留痕与监控统计。

    Raises:
        无。
    """

    return EmptyOutputError(f"场景输出为空或脏数据: {error}", raw_output=raw_text)


def _build_infer_output_error(raw_text: str, error: ValueError) -> RuntimeError:
    """构造公司级 facet 归因输出解析失败异常。

    Args:
        raw_text: infer scene 原始输出。
        error: 原始解析异常。

    Returns:
        带原始输出的 ``RuntimeError``，沿用 infer 现有错误形状（无独立错误类型）。

    Raises:
        无。
    """

    del raw_text
    return RuntimeError(f"公司级 Facet 归因输出非法: {error}")


def _build_cancelled_prompt_message(payload: Any) -> str:
    """将取消事件负载归一为写作流水线错误信息。

    Args:
        payload: 取消事件负载。

    Returns:
        适合日志与异常的取消描述。

    Raises:
        无。
    """

    reason = extract_cancel_reason(payload)
    if reason:
        return f"写作 Agent 执行被取消: {reason}"
    return "写作 Agent 执行被取消"


# 兼容旧 import：``Awaitable`` 在 Protocol 引用 contract_executor 时仍可能被
# 类型注解使用；保留 import 以避免下游测试模块隐式依赖该符号时报错。
_ = Awaitable  # type: ignore[assignment]
_ = _extract_markdown_content  # 显式标记 import 仍被审计相关模块以同名符号复用
_ = SceneAgentCreationError  # 兼容旧 import 路径
