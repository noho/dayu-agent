"""Scene Prompt 执行协调模块。

该模块封装写作流水线中所有 Prompt/Agent 调用的执行、重试与结果解析逻辑。
契约准备由 SceneContractPreparer 完成；本模块不持有 Host 层引用，通过
注入的 ``contract_executor`` 回调执行已构建的 ExecutionContract。

.. note::
    SceneAgentCreationError 已迁移至 scene_contract_preparer 模块，
    此处仅 re-export 以保持向后兼容。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable
from types import SimpleNamespace
from typing import Any, Callable, Protocol, TypeVar

from dayu.contracts.agent_execution import ExecutionContract
from dayu.contracts.cancellation import CancelledError
from dayu.contracts.events import AppEvent, AppEventType, AppResult
from dayu.log import Log
from dayu.services.internal.write_pipeline.audit_formatting import (
    _extract_markdown_content,
)
from dayu.services.internal.write_pipeline.audit_rules import (
    ConfirmOutputError,
    RepairOutputError,
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


MODULE = "APP.WRITE_PIPELINE"

_LLM_RETRY_LIMIT = 1
_LLM_RETRY_DELAY_SECONDS = 3

_RetryResult = TypeVar("_RetryResult")


class ScenePromptRunner:
    """Scene Prompt 执行协调器。

    职责：
    - 通过 SceneContractPreparer 获取 scene 执行信息和构建契约。
    - 通过注入的 ``contract_executor`` 回调执行契约（不直接引用 Host 类型）。
    - 管理重试策略和结果解析。
    """

    def __init__(
        self,
        *,
        preparer: SceneContractPreparer,
        contract_executor: Callable[[ExecutionContract], Awaitable[AppResult]],
        write_config: WriteRunConfig,
        prompt_agent: PromptAgentProtocol | None = None,
    ) -> None:
        """初始化 Scene Prompt 执行协调器。

        Args:
            preparer: Scene 契约准备器（纯 Service 层）。
            contract_executor: 契约执行回调，签名为
                ``(ExecutionContract) -> Awaitable[AppResult]``。
                由上层在组装时绑定到 Host 执行器方法。
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
    ) -> AppResult:
        """执行一次单轮 Agent 子执行。

        Args:
            prepared_scene: 已解析的 scene 执行信息。
            prompt_text: 当前轮用户输入。

        Returns:
            聚合后的应用层结果。
        """

        if self._prompt_agent is not None:
            return await self._collect_prompt_result_via_test_seam(
                prepared_scene=prepared_scene,
                prompt_text=prompt_text,
            )
        execution_contract = self._preparer.build_execution_contract(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
        )
        return await self._contract_executor(execution_contract)

    def run_prepared_scene_prompt(self, *, prepared_scene: AcceptedSceneExecution, prompt_text: str) -> AppResult:
        """同步执行一次单轮 Agent 子执行。

        若当前已处于事件循环中，则复用已有循环；否则创建新循环执行。

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
            # 已在异步上下文中，不能调用 asyncio.run()；
            # 创建 task 并同步等待会死锁，此处属于设计缺陷，暂抛明确异常。
            raise RuntimeError(
                "run_prepared_scene_prompt 不支持在已有事件循环中同步调用，"
                "请使用 _collect_prompt_result 的 async 版本"
            )
        return asyncio.run(coro)

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
    ) -> _RetryResult:
        """按统一重试策略执行 scene prompt 并解析结果。

        该 helper 统一收口两类失败路径：
        1. LLM/Agent 执行失败（`AppResult.errors` 非空）；
        2. 成功返回后解析失败（如 repair/confirm JSON 不合法）。

        Args:
            prepared_scene: 已解析的 scene 执行信息。
            prompt_text: 当前轮用户输入。
            execution_error_message: 执行失败时最终抛错前缀。
            execution_retry_message: 执行失败时的重试日志前缀。
            success_parser: 成功返回文本后的解析函数。
            parse_retry_message: 解析失败时的重试日志前缀；为 ``None`` 表示不处理解析重试。
            parse_error_builder: 解析失败时的异常构造器；为 ``None`` 表示直接透传解析异常。

        Returns:
            解析后的结构化结果。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当执行失败或解析失败且重试耗尽时抛出。
            ValueError: 当未配置解析错误构造器且解析函数本身抛出时透传。
        """

        last_error: RuntimeError | None = None
        for attempt in range(_LLM_RETRY_LIMIT + 1):
            try:
                result = self.run_prepared_scene_prompt(prepared_scene=prepared_scene, prompt_text=prompt_text)
            except CancelledError:
                raise
            if result.errors:
                last_error = RuntimeError(f"{execution_error_message}: {result.errors}")
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
                return success_parser(raw_text)
            except ValueError as exc:
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

        raise self._build_unexpected_retry_error(last_error=last_error, operation=execution_retry_message)

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
            RuntimeError: 当 Agent 返回错误时抛出。
        """

        return _extract_markdown_content(self._run_agent_prompt_raw(prompt_text))

    def run_infer_prompt(self, prompt_text: str) -> CompanyFacetProfile:
        """调用公司级 facet 归因 Agent 并解析结构化结果。

        Args:
            prompt_text: 用户提示词。

        Returns:
            公司级 facet 归因结果。

        Raises:
            RuntimeError: 当 Agent 返回错误或输出解析失败时抛出。
        """

        raw_text = self._run_infer_agent_prompt_raw(prompt_text)
        return parse_company_facets(raw_text, facet_catalog=self._preparer.get_company_facet_catalog())

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
            RuntimeError: 当 Agent 返回错误时抛出。
        """

        return _extract_markdown_content(self._run_decision_agent_prompt_raw(prompt_text))

    def run_overview_prompt(self, prompt_text: str) -> str:
        """调用第0章封面页 Agent 并提取 Markdown 正文。

        Args:
            prompt_text: 用户提示词。

        Returns:
            提取后的章节正文。

        Raises:
            RuntimeError: 当 Agent 返回错误时抛出。
        """

        return _extract_markdown_content(self._run_overview_agent_prompt_raw(prompt_text))

    def run_fix_prompt(self, prompt_text: str) -> str:
        """调用占位符补强 Agent 并提取 Markdown 正文。

        Args:
            prompt_text: 用户提示词。

        Returns:
            提取后的章节正文。

        Raises:
            RuntimeError: 当 Agent 返回错误时抛出。
        """

        return _extract_markdown_content(self._run_fix_agent_prompt_raw(prompt_text))

    def run_regenerate_prompt(self, prompt_text: str) -> str:
        """调用整章重建 Agent 并提取 Markdown 正文。

        Args:
            prompt_text: 用户提示词。

        Returns:
            提取后的章节正文。

        Raises:
            RuntimeError: 当 Agent 返回错误时抛出。
        """

        return _extract_markdown_content(self._run_regenerate_agent_prompt_raw(prompt_text))

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
        )

    # ------------------------------------------------------------------
    # 内部 raw prompt 方法（带重试）
    # ------------------------------------------------------------------

    def _run_agent_prompt_raw(self, prompt_text: str) -> str:
        """调用写作 Agent 并返回原始输出。

        LLM 返回异常时自动重试一次（延迟 _LLM_RETRY_DELAY_SECONDS 秒）。

        Args:
            prompt_text: 用户提示词。

        Returns:
            模型原始输出文本。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当重试耗尽仍失败时抛出。
        """

        prepared_scene = self._preparer.get_or_create_write_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="写作 Agent 执行失败",
            execution_retry_message="写作 Agent 调用失败",
            success_parser=_return_raw_prompt_text,
        )

    def _run_overview_agent_prompt_raw(self, prompt_text: str) -> str:
        """执行第0章封面页 scene，并返回原始文本。

        Args:
            prompt_text: 用户提示词。

        Returns:
            Agent 返回的原始文本。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当 Agent 返回错误时抛出。
        """

        prepared_scene = self._preparer.get_or_create_overview_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="第0章概览 Agent 执行失败",
            execution_retry_message="第0章概览 Agent 调用失败",
            success_parser=_return_raw_prompt_text,
        )

    def _run_infer_agent_prompt_raw(self, prompt_text: str) -> str:
        """执行公司级 facet 归因 scene，并返回原始文本。

        Args:
            prompt_text: 用户提示词。

        Returns:
            Agent 返回的原始文本。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当 Agent 返回错误时抛出。
        """

        prepared_scene = self._preparer.get_or_create_infer_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="公司级 Facet 归因 Agent 执行失败",
            execution_retry_message="公司级 Facet 归因 Agent 调用失败",
            success_parser=_return_raw_prompt_text,
        )

    def _run_decision_agent_prompt_raw(self, prompt_text: str) -> str:
        """调用研究决策综合 Agent 并返回原始输出。

        LLM 返回异常时自动重试一次（延迟 _LLM_RETRY_DELAY_SECONDS 秒）。

        Args:
            prompt_text: 用户提示词。

        Returns:
            模型原始输出文本。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当重试耗尽仍失败时抛出。
        """

        prepared_scene = self._preparer.get_or_create_decision_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="研究决策综合 Agent 执行失败",
            execution_retry_message="研究决策综合 Agent 调用失败",
            success_parser=_return_raw_prompt_text,
        )

    def _run_fix_agent_prompt_raw(self, prompt_text: str) -> str:
        """调用占位符补强 Agent 并返回原始输出。

        LLM 返回异常时自动重试一次（延迟 _LLM_RETRY_DELAY_SECONDS 秒）。

        Args:
            prompt_text: 用户提示词。

        Returns:
            模型原始输出文本。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当重试耗尽仍失败时抛出。
        """

        prepared_scene = self._preparer.get_or_create_fix_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="占位符补强 Agent 执行失败",
            execution_retry_message="占位符补强 Agent 调用失败",
            success_parser=_return_raw_prompt_text,
        )

    def _run_regenerate_agent_prompt_raw(self, prompt_text: str) -> str:
        """调用整章重建 Agent 并返回原始输出。

        LLM 返回异常时自动重试一次（延迟 _LLM_RETRY_DELAY_SECONDS 秒）。

        Args:
            prompt_text: 用户提示词。

        Returns:
            模型原始输出文本。

        Raises:
            CancelledError: 当执行被取消时透传。
            RuntimeError: 当重试耗尽仍失败时抛出。
        """

        prepared_scene = self._preparer.get_or_create_regenerate_scene()
        return self._run_scene_prompt_with_retry(
            prepared_scene=prepared_scene,
            prompt_text=prompt_text,
            execution_error_message="整章重建 Agent 执行失败",
            execution_retry_message="整章重建 Agent 调用失败",
            success_parser=_return_raw_prompt_text,
        )


def _return_raw_prompt_text(raw_text: str) -> str:
    """原样返回 prompt 文本结果。

    Args:
        raw_text: 模型原始输出。

    Returns:
        原始文本本身。

    Raises:
        无。
    """

    return raw_text


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


def _build_cancelled_prompt_message(payload: Any) -> str:
    """将取消事件负载归一为写作流水线错误信息。

    Args:
        payload: 取消事件负载。

    Returns:
        适合日志与异常的取消描述。

    Raises:
        无。
    """

    if isinstance(payload, dict):
        cancel_reason = str(payload.get("cancel_reason") or "").strip()
        if cancel_reason:
            return f"写作 Agent 执行被取消: {cancel_reason}"
    return "写作 Agent 执行被取消"
