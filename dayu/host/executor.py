"""默认宿主执行器实现。

统一管理 Service 侧重复出现的 run 生命周期治理：
- register/start/complete/fail/cancel run
- 创建 CancellationToken 并桥接跨进程取消
- 获取/释放并发许可
- 向 event bus 双写流式事件
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import threading
import uuid
from contextlib import suppress
from typing import Any, AsyncIterator, Callable, Literal, TypeVar

from dayu.host.cancellation_bridge import CancellationBridge
from dayu.host.events import build_app_event_from_stream_event
from dayu.host.host_execution import HostExecutorProtocol, HostedRunContext, HostedRunSpec
from dayu.host.agent_builder import build_async_agent
from dayu.host.concurrency import HOST_AGENT_LANE
from dayu.host.pending_turn_store import PendingConversationTurnState
from dayu.host.prepared_turn import (
    AcceptedAgentTurnSnapshot,
    PreparedAgentTurnSnapshot,
    serialize_accepted_agent_turn_snapshot,
    serialize_prepared_agent_turn_snapshot,
)
from dayu.host.scene_preparer import ScenePreparationProtocol
from dayu.host.protocols import (
    ConcurrencyGovernorProtocol,
    ConcurrencyPermit,
    PendingConversationTurnStoreProtocol,
    RunEventBusProtocol,
    RunRegistryProtocol,
    SessionWriteBlockedError,
)
from dayu.contracts.agent_execution import (
    AgentInput,
    ExecutionContract,
    ReplayHandle,
)
from dayu.contracts.agent_types import AgentMessage, build_user_chat_message
from dayu.contracts.cancellation import CancelledError, CancellationToken
from dayu.contracts.execution_metadata import ExecutionDeliveryContext, normalize_execution_delivery_context
from dayu.contracts.events import AppEvent, AppEventType, AppResult, PublishedRunEventProtocol
from dayu.contracts.run import RunCancelReason, RunRecord, RunState
from dayu.engine.events import EventType
from dayu.engine.tool_result import project_for_llm
from dayu.host.conversation_store import ConversationToolUseSummary
from dayu.log import Log

TStreamEvent = TypeVar("TStreamEvent", bound=PublishedRunEventProtocol)
TSyncResult = TypeVar("TSyncResult")
MODULE = "HOST.EXECUTOR"
_DEFAULT_CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS = 300.0
_MAX_TOOL_RESULT_SUMMARY_CHARS = 2000
_CONCURRENCY_POLICY_MODE_HOST_DEFAULT: Literal["host_default"] = "host_default"
_CONCURRENCY_POLICY_MODE_TIMEOUT: Literal["timeout"] = "timeout"
_CONCURRENCY_POLICY_MODE_UNBOUNDED: Literal["unbounded"] = "unbounded"


def should_delete_pending_turn_after_terminal_run(
    *,
    run: RunRecord | None,
    resumable: bool,
) -> bool:
    """给定终态 run，判断是否应当删除关联 pending turn。

    Args:
        run: 对应的 run 记录；``None`` 表示 run 记录缺失。
        resumable: pending turn 所属 scene 是否允许 resume。

    Returns:
        ``True`` 表示应当删除 pending turn；``False`` 表示应当保留以供恢复。

    Raises:
        无。
    """

    if run is None:
        return True
    if run.state in (RunState.FAILED, RunState.UNSETTLED):
        return not resumable
    if run.state == RunState.CANCELLED:
        if resumable and run.cancel_reason == RunCancelReason.TIMEOUT:
            return False
        return True
    return True


def _format_run_identity(
    *,
    run_id: str,
    session_id: str | None,
    scene_name: str | None,
    service_type: str,
) -> str:
    """格式化 run 基本身份信息。

    Args:
        run_id: Host run ID。
        session_id: 宿主会话 ID。
        scene_name: scene 名称。
        service_type: 服务类型。

    Returns:
        统一格式的 run 身份摘要。

    Raises:
        无。
    """

    return (
        f"run_id={run_id}, session_id={session_id or ''}, "
        f"scene_name={scene_name or ''}, service_type={service_type}"
    )


def _build_agent_run_start_debug_message(
    *,
    run_id: str,
    session_id: str | None,
    scene_name: str | None,
    service_type: str,
    timeout_ms: int | None,
    business_lane: str | None,
    resumable: bool,
) -> str:
    """构造 Agent 子执行启动调试日志。

    Args:
        run_id: Host run ID。
        session_id: 宿主会话 ID。
        scene_name: scene 名称。
        service_type: 服务类型。
        timeout_ms: run 超时毫秒数。
        business_lane: 业务 lane。
        resumable: 当前 scene 是否允许恢复。

    Returns:
        启动调试日志文本。

    Raises:
        无。
    """

    return (
        "Host 启动 agent run: "
        f"{_format_run_identity(run_id=run_id, session_id=session_id, scene_name=scene_name, service_type=service_type)}, "
        f"timeout_ms={timeout_ms}, business_lane={business_lane or ''}, resumable={resumable}"
    )


def _build_agent_settle_debug_message(
    *,
    phase: str,
    run_id: str,
    pending_turn_id: str | None,
    resumable: bool,
) -> str:
    """构造 Agent 收敛路径调试日志。

    Args:
        phase: 当前收敛路径名称。
        run_id: Host run ID。
        pending_turn_id: 当前 pending turn ID。
        resumable: 当前 scene 是否允许恢复。

    Returns:
        收敛路径调试日志文本。

    Raises:
        无。
    """

    return (
        "Host 收敛 agent run: "
        f"phase={phase}, run_id={run_id}, pending_turn_id={pending_turn_id or ''}, resumable={resumable}"
    )


def _resolve_concurrency_acquire_timeout_seconds(spec: HostedRunSpec) -> float | None:
    """解析并发 permit 获取超时。

    Args:
        spec: 宿主 run 规格。

    Returns:
        permit 获取最大等待秒数；返回 ``None`` 表示无限等待。

    Raises:
        无。
    """

    policy = spec.concurrency_acquire_policy
    if policy.mode == _CONCURRENCY_POLICY_MODE_UNBOUNDED:
        return None
    if policy.mode == _CONCURRENCY_POLICY_MODE_TIMEOUT:
        return policy.timeout_seconds
    if spec.timeout_ms is None:
        return _DEFAULT_CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS
    return max(0.001, spec.timeout_ms / 1000.0)


def _required_lanes_for_spec(spec: HostedRunSpec, *, include_agent_lane: bool) -> list[str]:
    """计算一次 run 需要叠加持有的 lane 名，按字母序返回以防死锁。

    Args:
        spec: 宿主 run 规格。
        include_agent_lane: 是否由 Host 独立裁决叠加 ``llm_api`` lane。

    Returns:
        需要 acquire 的 lane 名列表，按字母序。

    Raises:
        ValueError: Service 在 ``business_concurrency_lane`` 写入 Host 自治
            lane 名时抛出。
    """

    lanes: list[str] = []
    if include_agent_lane:
        lanes.append(HOST_AGENT_LANE)
    business_lane = spec.business_concurrency_lane
    if business_lane:
        if business_lane == HOST_AGENT_LANE:
            raise ValueError(
                "business_concurrency_lane 不允许等于 Host 自治 lane 名；"
                "llm_api 由 Host 根据调用路径自动管理"
            )
        lanes.append(business_lane)
    return sorted(lanes)


class RunDeadlineWatcher:
    """run 级 deadline watcher。

    使用 `threading.Timer` 在独立 OS 线程计时，确保墙钟超时在 asyncio 事件循环
    饥饿（被阻塞调用、大块 CPU 计算）时仍可靠触发——这是 deadline watcher 的
    硬语义要求，不可让位于"节省线程"的微观优化。回调触发的 `request_cancel`
    与 `token.cancel` 均为同步、幂等、线程安全操作。
    """

    def __init__(
        self,
        run_registry: RunRegistryProtocol,
        run_id: str,
        token: CancellationToken,
        timeout_ms: int | None,
    ) -> None:
        """初始化 deadline watcher。

        Args:
            run_registry: run 注册表。
            run_id: 目标 run ID。
            token: 取消令牌。
            timeout_ms: 超时毫秒数；`None` 表示不启用。

        Raises:
            ValueError: `timeout_ms` 非正整数时抛出。
        """

        if timeout_ms is not None and timeout_ms <= 0:
            raise ValueError("timeout_ms 必须为正整数或 None")
        self._run_registry = run_registry
        self._run_id = run_id
        self._token = token
        self._timeout_ms = timeout_ms
        self._timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()

    def start(self) -> None:
        """启动 watcher。幂等。"""

        with self._timer_lock:
            if self._timeout_ms is None or self._timer is not None:
                return
            timer = threading.Timer(self._timeout_ms / 1000.0, self._on_timeout)
            timer.daemon = True
            timer.name = f"run-deadline-{self._run_id}"
            self._timer = timer
        timer.start()

    def stop(self) -> None:
        """停止 watcher。幂等。"""

        with self._timer_lock:
            timer = self._timer
            self._timer = None
        if timer is not None:
            timer.cancel()

    def _on_timeout(self) -> None:
        """处理 deadline 到时。在 Timer 线程执行。"""

        with self._timer_lock:
            self._timer = None
        self._run_registry.request_cancel(
            self._run_id,
            cancel_reason=RunCancelReason.TIMEOUT,
        )
        self._token.cancel()


@dataclass
class _ReplayCapture:
    """`_run_prepared_agent_stream` 把执行结束时的 messages 与原始 ``AgentInput``
    填回该容器，供 ``run_agent_and_wait_replayable`` 在 stream 收敛后登记到
    replay stash。

    replay 路径**不会**复用上一次的 ``AsyncAgent`` 实例：上一次构造的 runner
    与 agent 都绑死在第一次 run 的 ``CancellationToken`` 上，复用后新一次
    ``_start_run`` 创建的取消桥 / deadline watcher 无法穿透到 runner 与
    工具调用层。replay 时会用同一份 ``AgentInput`` 透过 ``build_async_agent``
    重新构造 ``AsyncAgent`` 并绑定新 token。

    Args:
        messages: 执行结束时的完整对话历史（含 assistant + tool 消息）。
        agent_input: 上一次执行使用的 ``AgentInput``，replay 时仅复用其
            ``agent_create_args`` / ``tools`` / ``tool_trace_recorder_factory``
            / ``trace_identity``，``cancellation_handle`` 在 replay 时被替换。

    Returns:
        无。

    Raises:
        无。
    """

    messages: list[AgentMessage] = field(default_factory=list)
    agent_input: AgentInput | None = None


@dataclass
class _ReplayState:
    """单次执行结束后登记到 Host 的 replay 状态。

    Args:
        messages: 上一次执行结束时的完整对话历史。
        agent_input: 上一次执行的 ``AgentInput``，replay 路径基于其字段重新
            ``build_async_agent`` 出绑定新 token 的 ``AsyncAgent``。
        session_id: 关联的 session_id，跨 session 使用 handle 时直接拒绝。
        run_id: 颁发该 handle 时的 Host run_id，仅作日志辅助。

    Returns:
        无。

    Raises:
        无。
    """

    messages: list[AgentMessage]
    agent_input: AgentInput
    session_id: str | None
    run_id: str


@dataclass(frozen=True)
class _RunResources:
    """绑定到单个 run 的宿主资源三元组。

    由 ``DefaultHostExecutor._start_run`` 创建并登记入资源注册表，
    由 ``_finish_run``（异步路径终态）或 ``release_resources_for_run``
    （SIGINT/SIGTERM 同步路径）通过 atomic-pop 二选一释放。

    所有字段均为不可变引用，资源对象自身的 ``stop`` / governor.release 已经
    幂等，不需要在此再加状态标记。
    """

    bridge: CancellationBridge
    deadline_watcher: RunDeadlineWatcher
    permits: list[ConcurrencyPermit]


@dataclass
class DefaultHostExecutor(HostExecutorProtocol):
    """默认宿主执行器。"""

    run_registry: RunRegistryProtocol
    pending_turn_store: PendingConversationTurnStoreProtocol | None = None
    concurrency_governor: ConcurrencyGovernorProtocol | None = None
    event_bus: RunEventBusProtocol | None = None
    scene_preparation: ScenePreparationProtocol | None = None
    # CancellationBridge 配置：默认值与 ``CancellationBridge.__init__`` 默认值
    # 保持一致；调用方需要更精细的失败降级策略时由 Host 装配链路透传。
    cancellation_bridge_poll_interval_seconds: float = 0.5
    cancellation_bridge_failure_grace_period_seconds: float = 5.0
    _replay_stash: dict[str, _ReplayState] = field(default_factory=dict, init=False, repr=False)
    _replay_stash_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _run_resources: dict[str, _RunResources] = field(default_factory=dict, init=False, repr=False)
    _run_resources_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    async def run_operation_stream(
        self,
        *,
        spec: HostedRunSpec,
        event_stream_factory: Callable[[HostedRunContext], AsyncIterator[TStreamEvent]],
    ) -> AsyncIterator[TStreamEvent]:
        """托管一次流式执行。"""

        run = self.run_registry.register_run(
            session_id=spec.session_id,
            service_type=spec.operation_name,
            scene_name=spec.scene_name,
            metadata=spec.metadata,
        )
        try:
            context, bridge, deadline_watcher, permits = self._start_run(
                spec=spec, run_id=run.run_id, include_agent_lane=False,
            )
        except CancelledError:
            self._finalize_cancelled(run.run_id)
            return
        try:
            async for event in event_stream_factory(context):
                if spec.publish_events:
                    self._publish_event(run.run_id, event)
                yield event
            if self._is_cancelled(run_id=run.run_id, token=context.cancellation_token):
                self._finalize_cancelled(run.run_id)
                return
            settled_run = self._complete_run_preserving_terminal_state(run_id=run.run_id)
            if settled_run.state == RunState.CANCELLED:
                return
            if settled_run.state == RunState.FAILED:
                raise RuntimeError(self._describe_external_terminal_failure(settled_run))
        except CancelledError:
            self._finalize_cancelled(run.run_id)
        except Exception as exc:
            if self._is_cancelled(run_id=run.run_id, token=context.cancellation_token):
                self._finalize_cancelled(run.run_id)
                return
            self._fail_run_preserving_original_exception(
                run_id=run.run_id,
                error_summary=self._summarize_error(exc, spec.error_summary_limit),
            )
            raise
        finally:
            self._finish_run(bridge=bridge, deadline_watcher=deadline_watcher, permits=permits, run_id=run.run_id)

    def run_operation_sync(
        self,
        *,
        spec: HostedRunSpec,
        operation: Callable[[HostedRunContext], TSyncResult],
        on_cancel: Callable[[], TSyncResult] | None = None,
    ) -> TSyncResult:
        """托管一次同步执行。"""

        run = self.run_registry.register_run(
            session_id=spec.session_id,
            service_type=spec.operation_name,
            scene_name=spec.scene_name,
            metadata=spec.metadata,
        )
        try:
            context, bridge, deadline_watcher, permits = self._start_run(
                spec=spec, run_id=run.run_id, include_agent_lane=False,
            )
        except CancelledError:
            return self._finish_sync_cancelled_run(run_id=run.run_id, on_cancel=on_cancel)
        try:
            result = operation(context)
            if self._is_cancelled(run_id=run.run_id, token=context.cancellation_token):
                return self._finish_sync_cancelled_run(run_id=run.run_id, on_cancel=on_cancel)
            settled_run = self._complete_run_preserving_terminal_state(run_id=run.run_id)
            if settled_run.state == RunState.CANCELLED:
                return self._finish_sync_cancelled_run(run_id=run.run_id, on_cancel=on_cancel)
            if settled_run.state == RunState.FAILED:
                raise RuntimeError(self._describe_external_terminal_failure(settled_run))
            return result
        except CancelledError:
            return self._finish_sync_cancelled_run(run_id=run.run_id, on_cancel=on_cancel)
        except Exception as exc:
            if self._is_cancelled(run_id=run.run_id, token=context.cancellation_token):
                return self._finish_sync_cancelled_run(run_id=run.run_id, on_cancel=on_cancel)
            self._fail_run_preserving_original_exception(
                run_id=run.run_id,
                error_summary=self._summarize_error(exc, spec.error_summary_limit),
            )
            raise
        finally:
            self._finish_run(bridge=bridge, deadline_watcher=deadline_watcher, permits=permits, run_id=run.run_id)

    async def run_agent_stream(
        self,
        execution_contract: ExecutionContract,
        *,
        resumed_pending_turn_id: str | None = None,
    ) -> AsyncIterator[Any]:
        """托管一次 Agent 子执行并返回应用层事件流。

        Args:
            execution_contract: Service 输出的执行契约。
            resumed_pending_turn_id: 仅在 Host 层发起 pending turn resume 流程时传入，
                表示本次执行已由 Host 端持有 RESUMING lease；executor 必须跳过
                ``_register_accepted_pending_turn`` 的 upsert，以免覆盖调用方持有的
                lease。非 resume 路径保持 ``None``。

        Returns:
            应用层事件流。

        Raises:
            RuntimeError: 未配置 scene preparation 时抛出。
        """

        async for event in self._run_agent_stream_internal(
            execution_contract,
            resumed_pending_turn_id=resumed_pending_turn_id,
            replay_capture=None,
        ):
            yield event

    async def _run_agent_stream_internal(
        self,
        execution_contract: ExecutionContract,
        *,
        resumed_pending_turn_id: str | None = None,
        replay_capture: _ReplayCapture | None = None,
    ) -> AsyncIterator[Any]:
        """``run_agent_stream`` 的内部实现，附带 replay 捕获能力。

        Args:
            execution_contract: Service 输出的执行契约。
            resumed_pending_turn_id: 见 ``run_agent_stream``。
            replay_capture: 可选的 replay 捕获容器；非 ``None`` 时 stream 结束
                后底层会把最终 messages / AsyncAgent 实例填入容器，供
                ``run_agent_and_wait_replayable`` 登记到 stash。该参数仅供 Host
                内部的 replay 实现使用，不在 ``HostExecutorProtocol`` 上暴露。

        Returns:
            应用层事件流。

        Raises:
            RuntimeError: 未配置 scene preparation 时抛出。
        """

        if self.scene_preparation is None:
            raise RuntimeError("当前 HostExecutor 未配置 scene preparation")

        spec = HostedRunSpec(
            operation_name=execution_contract.service_name,
            session_id=execution_contract.host_policy.session_key,
            scene_name=execution_contract.scene_name,
            metadata=execution_contract.metadata,
            business_concurrency_lane=execution_contract.host_policy.business_concurrency_lane,
            concurrency_acquire_policy=execution_contract.host_policy.concurrency_acquire_policy,
            timeout_ms=execution_contract.host_policy.timeout_ms,
        )
        run = self.run_registry.register_run(
            session_id=spec.session_id,
            service_type=spec.operation_name,
            scene_name=spec.scene_name,
            metadata=spec.metadata,
        )
        resumable = bool(execution_contract.host_policy.resumable)
        try:
            context, bridge, deadline_watcher, permits = self._start_run(
                spec=spec, run_id=run.run_id, include_agent_lane=True,
            )
        except CancelledError:
            yield self._settle_agent_cancelled_before_pending_turn_binding(
                run_id=run.run_id,
                resumed_pending_turn_id=resumed_pending_turn_id,
                resumable=resumable,
            )
            return
        Log.debug(
            _build_agent_run_start_debug_message(
                run_id=run.run_id,
                session_id=spec.session_id,
                scene_name=spec.scene_name,
                service_type=spec.operation_name,
                timeout_ms=spec.timeout_ms,
                business_lane=spec.business_concurrency_lane,
                resumable=resumable,
            ),
            module=MODULE,
        )
        pending_turn_id: str | None = None
        try:
            # pending turn 登记 / resume lease 重绑必须在 try/finally 收敛范围内：
            # 任一抛异常都必须把当前新 run 推进到终态并释放 permit / deadline
            # watcher / cancellation bridge，否则会泄漏活跃 run。
            pending_turn_id = (
                resumed_pending_turn_id
                if resumed_pending_turn_id is not None
                else self._register_accepted_pending_turn(
                    execution_contract=execution_contract,
                    run_id=run.run_id,
                )
            )
            if resumed_pending_turn_id is not None and self.pending_turn_store is not None:
                # resume 路径下跳过了 _register_*_pending_turn 的 upsert，必须显式把
                # pending turn 的 source_run_id 重绑到当前 resumed run；否则 "成功执行
                # 后 delete 瞬时失败" 会留下 source_run_id 指向旧 timeout-cancelled run
                # 的残留，后续 resume gate 仍会放行，触发重复恢复。
                self.pending_turn_store.rebind_source_run_id_for_resume(
                    resumed_pending_turn_id,
                    new_source_run_id=run.run_id,
                )
            prepared_execution = await self.scene_preparation.prepare(execution_contract, context)
            if resumable and prepared_execution.resume_snapshot is None:
                raise RuntimeError("resumable scene preparation 缺少 prepared snapshot")
            if prepared_execution.resume_snapshot is not None and resumed_pending_turn_id is None:
                # resume 路径已由 Host 端持有 RESUMING lease，不能再 upsert 覆盖；
                # 非 resume 路径在此刻把 pending turn 切换到 prepared 真源。
                pending_turn_id = self._register_prepared_pending_turn(
                    prepared_turn=prepared_execution.resume_snapshot,
                    run_id=run.run_id,
                )
            async for event in self._run_prepared_agent_stream(
                run_id=run.run_id,
                session_id=spec.session_id,
                pending_turn_id=pending_turn_id,
                agent_input=prepared_execution.agent_input,
                replay_capture=replay_capture,
            ):
                yield event
            terminal_event = self._settle_agent_stream_completion(
                run_id=run.run_id, token=context.cancellation_token,
                pending_turn_id=pending_turn_id, resumable=resumable,
            )
            if terminal_event is not None:
                yield terminal_event
        except CancelledError:
            yield self._settle_agent_cancelled(
                run_id=run.run_id, pending_turn_id=pending_turn_id, resumable=resumable,
            )
        except Exception as exc:
            terminal_event = self._settle_agent_exception(
                run_id=run.run_id, token=context.cancellation_token,
                pending_turn_id=pending_turn_id, resumable=resumable,
                exc=exc, error_summary_limit=spec.error_summary_limit,
            )
            if terminal_event is not None:
                yield terminal_event
                return
            raise
        finally:
            self._finish_run(bridge=bridge, deadline_watcher=deadline_watcher, permits=permits, run_id=run.run_id)

    async def run_prepared_turn_stream(
        self,
        prepared_turn: PreparedAgentTurnSnapshot,
        *,
        resumed_pending_turn_id: str | None = None,
    ) -> AsyncIterator[Any]:
        """基于 prepared turn 快照恢复一次 Agent 子执行。

        Args:
            prepared_turn: Host prepared snapshot。
            resumed_pending_turn_id: 仅在 Host 层发起 pending turn resume 流程时传入，
                表示本次执行已由 Host 端持有 RESUMING lease；executor 必须跳过
                ``_register_prepared_pending_turn`` 的 upsert，以免覆盖调用方持有
                的 lease。非 resume 路径保持 ``None``。
        """

        if self.scene_preparation is None:
            raise RuntimeError("当前 HostExecutor 未配置 scene preparation")

        spec = _build_run_spec_from_prepared_turn(prepared_turn)
        run = self.run_registry.register_run(
            session_id=spec.session_id,
            service_type=spec.operation_name,
            scene_name=spec.scene_name,
            metadata=spec.metadata,
        )
        resumable = bool(prepared_turn.resumable)
        try:
            context, bridge, deadline_watcher, permits = self._start_run(
                spec=spec, run_id=run.run_id, include_agent_lane=True,
            )
        except CancelledError:
            yield self._settle_agent_cancelled_before_pending_turn_binding(
                run_id=run.run_id,
                resumed_pending_turn_id=resumed_pending_turn_id,
                resumable=resumable,
            )
            return
        Log.debug(
            _build_agent_run_start_debug_message(
                run_id=run.run_id,
                session_id=spec.session_id,
                scene_name=spec.scene_name,
                service_type=spec.operation_name,
                timeout_ms=spec.timeout_ms,
                business_lane=spec.business_concurrency_lane,
                resumable=resumable,
            ),
            module=MODULE,
        )
        pending_turn_id: str | None = None
        try:
            # pending turn 登记 / resume lease 重绑必须在 try/finally 收敛范围内：
            # 任一抛异常都必须把当前新 run 推进到终态并释放 permit / deadline
            # watcher / cancellation bridge，否则会泄漏活跃 run。
            pending_turn_id = (
                resumed_pending_turn_id
                if resumed_pending_turn_id is not None
                else self._register_prepared_pending_turn(
                    prepared_turn=prepared_turn, run_id=run.run_id,
                )
            )
            if resumed_pending_turn_id is not None and self.pending_turn_store is not None:
                # 同 run_agent_stream 的 resume 分支：显式把 source_run_id 切到当前
                # resumed run，避免 delete 失败后旧 source_run 被再次放行恢复。
                self.pending_turn_store.rebind_source_run_id_for_resume(
                    resumed_pending_turn_id,
                    new_source_run_id=run.run_id,
                )
            agent_input = await self.scene_preparation.restore_prepared_execution(prepared_turn, context)
            async for event in self._run_prepared_agent_stream(
                run_id=run.run_id,
                session_id=spec.session_id,
                pending_turn_id=pending_turn_id,
                agent_input=agent_input,
            ):
                yield event
            terminal_event = self._settle_agent_stream_completion(
                run_id=run.run_id, token=context.cancellation_token,
                pending_turn_id=pending_turn_id, resumable=resumable,
            )
            if terminal_event is not None:
                yield terminal_event
        except CancelledError:
            yield self._settle_agent_cancelled(
                run_id=run.run_id, pending_turn_id=pending_turn_id, resumable=resumable,
            )
        except Exception as exc:
            terminal_event = self._settle_agent_exception(
                run_id=run.run_id, token=context.cancellation_token,
                pending_turn_id=pending_turn_id, resumable=resumable,
                exc=exc, error_summary_limit=spec.error_summary_limit,
            )
            if terminal_event is not None:
                yield terminal_event
                return
            raise
        finally:
            self._finish_run(bridge=bridge, deadline_watcher=deadline_watcher, permits=permits, run_id=run.run_id)

    def _register_accepted_pending_turn(
        self,
        *,
        execution_contract: ExecutionContract,
        run_id: str,
    ) -> str | None:
        """在 Host 接单后登记 accepted pending turn 真源。"""

        if self.pending_turn_store is None:
            return None
        session_id = str(execution_contract.host_policy.session_key or "").strip()
        scene_name = str(execution_contract.scene_name or "").strip()
        user_text = str(execution_contract.message_inputs.user_message or "").strip()
        if not bool(execution_contract.host_policy.resumable) or not session_id or not scene_name or not user_text:
            return None
        accepted_turn = _build_accepted_turn_snapshot(execution_contract)
        accepted_turn_json = json.dumps(
            serialize_accepted_agent_turn_snapshot(accepted_turn),
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            record = self.pending_turn_store.upsert_pending_turn(
                session_id=session_id,
                scene_name=scene_name,
                user_text=user_text,
                source_run_id=run_id,
                resumable=True,
                state=PendingConversationTurnState.ACCEPTED_BY_HOST,
                resume_source_json=accepted_turn_json,
                metadata=_extract_pending_turn_metadata(execution_contract.metadata),
            )
        except SessionWriteBlockedError as exc:
            # cancel_session / clear_session_history 已在仓储层屏障处使 session
            # 进入非 ACTIVE 状态，executor 必须放弃登记以避免生成孤儿 pending
            # turn。三个具体子类（CLOSED / CLEARING / CLEARING_FAILED）共享同一
            # 吸收语义，统一捕获基类。
            Log.verbose(
                f"[{run_id}] session 写入屏障已立，跳过 accepted pending turn 登记: "
                f"session_id={session_id}, scene_name={scene_name}, "
                f"barrier={type(exc).__name__}",
                module=MODULE,
            )
            return None
        Log.verbose(
            f"[{run_id}] pending turn 已登记 accepted 真源: "
            f"pending_turn_id={record.pending_turn_id}, session_id={session_id}, "
            f"scene_name={scene_name}, state={record.state.value}",
            module=MODULE,
        )
        return record.pending_turn_id

    def _register_prepared_pending_turn(
        self,
        *,
        prepared_turn: PreparedAgentTurnSnapshot,
        run_id: str,
    ) -> str | None:
        """在 Host 完成 scene preparation 后登记 prepared pending turn 真源。

        Args:
            prepared_turn: Host 已完成 scene preparation 的快照。
            run_id: 当前 Host run ID。

        Returns:
            新建或复用的 pending turn ID；无需登记时返回 ``None``。

        Raises:
            无。
        """

        if self.pending_turn_store is None:
            return None
        session_snapshot = prepared_turn.conversation_session
        session_id = str(session_snapshot.session_id if session_snapshot is not None else "").strip()
        scene_name = str(prepared_turn.scene_name or "").strip()
        user_text = str(session_snapshot.user_message if session_snapshot is not None else "").strip()
        if not session_id or not scene_name or not user_text:
            return None
        metadata = _extract_pending_turn_metadata(prepared_turn.metadata)
        prepared_turn_json = json.dumps(
            serialize_prepared_agent_turn_snapshot(prepared_turn),
            ensure_ascii=False,
            sort_keys=True,
        )
        try:
            record = self.pending_turn_store.upsert_pending_turn(
                session_id=session_id,
                scene_name=scene_name,
                user_text=user_text,
                source_run_id=run_id,
                resumable=bool(prepared_turn.resumable),
                state=PendingConversationTurnState.PREPARED_BY_HOST,
                resume_source_json=prepared_turn_json,
                metadata=metadata,
            )
        except SessionWriteBlockedError as exc:
            # cancel_session / clear_session_history 已在仓储层屏障处使 session
            # 进入非 ACTIVE 状态，executor 必须放弃登记以避免生成孤儿 pending
            # turn。三个具体子类共享同一吸收语义，统一捕获基类。
            Log.verbose(
                f"[{run_id}] session 写入屏障已立，跳过 prepared pending turn 登记: "
                f"session_id={session_id}, scene_name={scene_name}, "
                f"barrier={type(exc).__name__}",
                module=MODULE,
            )
            return None
        Log.verbose(
            f"[{run_id}] pending turn 已登记 prepared 真源: "
            f"pending_turn_id={record.pending_turn_id}, session_id={session_id}, "
            f"scene_name={scene_name}, state={record.state.value}",
            module=MODULE,
        )
        return record.pending_turn_id

    async def run_agent_and_wait(
        self,
        execution_contract: ExecutionContract,
    ) -> AppResult:
        """托管一次 Agent 子执行并等待完整结果。

        Args:
            execution_contract: Service 输出的执行契约。

        Returns:
            应用层聚合结果。

        Raises:
            CancelledError: 执行被取消时抛出。
            RuntimeError: 未配置 scene preparation 时抛出。
        """

        result, _ = await self._run_agent_and_collect(execution_contract, capture=None)
        return result

    async def run_agent_and_wait_replayable(
        self,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        """托管一次 Agent 子执行并颁发用于带历史回放的句柄。

        相比 ``run_agent_and_wait``，本方法会在执行结束时把完整对话历史登记
        到 Host 内部 stash，返回一个对上层不透明的 ``ReplayHandle``。Service
        若发现本次输出脏数据（解析失败、空输出等），可把句柄塞回新的执行
        契约调用 ``replay_agent_and_wait`` 兜底。

        Args:
            execution_contract: Service 输出的执行契约。

        Returns:
            ``(AppResult, ReplayHandle)`` 二元组。``ReplayHandle`` 仅在颁发
            它的 Host 实例生命周期内有效；session 关闭或 Host 重启后即失效。

        Raises:
            CancelledError: 执行被取消时抛出。
            RuntimeError: 未配置 scene preparation 时抛出；或 stream 结束后
                未捕获到完整 messages（理论上不可达）。
        """

        capture = _ReplayCapture()
        result, captured = await self._run_agent_and_collect(execution_contract, capture=capture)
        if captured.agent_input is None:
            raise RuntimeError("Host run 未捕获到 AgentInput，无法颁发 replay handle")
        handle = self._register_replay_state(
            messages=captured.messages,
            agent_input=captured.agent_input,
            session_id=execution_contract.host_policy.session_key,
        )
        return result, handle

    async def replay_agent_and_wait(
        self,
        handle: ReplayHandle,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        """基于已颁发的 ``ReplayHandle`` 带历史回放一次 Agent 执行。

        Host 取出 handle 关联的对话历史，把
        ``execution_contract.message_inputs.user_message`` 追加在历史末尾，
        然后复用同一个 ``AsyncAgent`` 实例调用 ``run_messages``；当
        ``message_inputs.replay_disable_tools=True`` 时透传到 engine，强制
        本次执行禁用工具调用，逼迫模型输出文本。

        Args:
            handle: 上一次 ``run_agent_and_wait_replayable`` 颁发的句柄。
            execution_contract: 用于本次回放的执行契约；只有
                ``message_inputs.user_message`` / ``replay_disable_tools`` /
                ``host_policy.session_key`` 被使用，其余字段保留供 trace。

        Returns:
            ``(AppResult, ReplayHandle)`` 二元组。返回的句柄是新颁发的，
            可用于继续追加 replay 的下一轮（service 当前不会触发二次 replay）。

        Raises:
            RuntimeError: handle 已失效；或 session 不匹配。
            CancelledError: 执行被取消时抛出。
        """

        state = self._consume_replay_state(handle)
        target_session = execution_contract.host_policy.session_key
        if state.session_id != target_session:
            raise RuntimeError(
                f"replay handle 与当前 session 不匹配: handle_session={state.session_id}, "
                f"target_session={target_session}"
            )
        retry_text = execution_contract.message_inputs.user_message or ""
        if not retry_text.strip():
            raise RuntimeError("replay 调用必须在 message_inputs.user_message 提供追加文本")
        # replay 路径与普通 agent run 共用同一套 Host run 生命周期：先注册 run、
        # 再 ``_start_run`` 挂取消桥 / deadline watcher / 并发 lane（含 ``llm_api``
        # 自治 lane），最后通过 ``_finish_run`` 释放。绕过这一步会让 SQLite
        # run_registry 在 ``complete_run`` 时报 CREATED -> SUCCEEDED 非法转换，
        # 同时也无法响应取消与 timeout。
        spec = HostedRunSpec(
            operation_name=execution_contract.service_name,
            session_id=target_session,
            scene_name=execution_contract.scene_name,
            metadata=execution_contract.metadata,
            business_concurrency_lane=execution_contract.host_policy.business_concurrency_lane,
            concurrency_acquire_policy=execution_contract.host_policy.concurrency_acquire_policy,
            timeout_ms=execution_contract.host_policy.timeout_ms,
        )
        run = self.run_registry.register_run(
            session_id=spec.session_id,
            service_type=spec.operation_name,
            scene_name=spec.scene_name,
            metadata=spec.metadata,
        )
        context, bridge, deadline_watcher, permits = self._start_run(
            spec=spec, run_id=run.run_id, include_agent_lane=True,
        )
        Log.verbose(
            f"[{run.run_id}] replay agent run 启动: parent_run_id={state.run_id}, "
            f"session_id={target_session or ''}, scene_name={execution_contract.scene_name}, "
            f"disable_tools={execution_contract.message_inputs.replay_disable_tools}",
            module=MODULE,
        )
        replay_messages = list(state.messages)
        replay_messages.append(build_user_chat_message(retry_text))
        # 复用第一次 run 捕获的 AsyncAgent 会让本次 replay 期间无法响应取消：
        # 该 agent 的 runner / cancellation_token 都绑死在第一次 run。这里基于
        # 上一次的 ``AgentInput`` 重新 ``build_async_agent``，把 runner 绑到
        # 本次 ``_start_run`` 颁发的新 token 上。
        previous_input = state.agent_input
        agent = build_async_agent(
            agent_create_args=previous_input.agent_create_args,
            tool_executor=previous_input.tools,
            tool_trace_recorder_factory=previous_input.tool_trace_recorder_factory,
            trace_identity=previous_input.trace_identity,
            cancellation_token=context.cancellation_token,
        )
        replay_agent_input = AgentInput(
            system_prompt=previous_input.system_prompt,
            messages=replay_messages,
            tools=previous_input.tools,
            agent_create_args=previous_input.agent_create_args,
            session_state=previous_input.session_state,
            runtime_limits=previous_input.runtime_limits,
            cancellation_handle=context.cancellation_token,
            tool_trace_recorder_factory=previous_input.tool_trace_recorder_factory,
            trace_identity=previous_input.trace_identity,
        )
        normalized_session_id = str(target_session or "").strip() or None
        content = ""
        warnings: list[str] = []
        errors: list[str] = []
        degraded = False
        filtered = False
        cancelled_payload: dict[str, str] | None = None
        try:
            try:
                async for stream_event in agent.run_messages(
                    replay_messages,
                    session_id=normalized_session_id,
                    run_id=run.run_id,
                    stream=True,
                    disable_tools=execution_contract.message_inputs.replay_disable_tools,
                ):
                    mapped = build_app_event_from_stream_event(stream_event)
                    if mapped is not None:
                        self._publish_event(run.run_id, mapped)
                    if stream_event.type == EventType.WARNING:
                        warnings.append(_extract_event_message(stream_event.data))
                    elif stream_event.type == EventType.ERROR:
                        errors.append(_extract_event_message(stream_event.data))
                    elif (
                        stream_event.type == EventType.FINAL_ANSWER
                        and isinstance(stream_event.data, dict)
                    ):
                        content = str(stream_event.data.get("content") or "")
                        degraded = bool(stream_event.data.get("degraded", False))
                        filtered = bool(stream_event.data.get("filtered", False))
                if self._is_cancelled(run_id=run.run_id, token=context.cancellation_token):
                    # stream 自然结束但期间已被请求取消 / timeout：必须像
                    # ``run_agent_and_wait`` 在收到 CANCELLED 事件那样抛
                    # ``CancelledError``，否则上层会拿到误导性的“成功 AppResult”。
                    cancelled_run = self._finalize_cancelled(run.run_id)
                    cancel_event = self._publish_cancelled_app_event(
                        run_id=run.run_id, run=cancelled_run,
                    )
                    payload = cancel_event.payload
                    cancelled_payload = (
                        dict(payload) if isinstance(payload, dict) else None
                    )
                else:
                    self._complete_run_preserving_terminal_state(run_id=run.run_id)
            except CancelledError:
                # 与上方"自然结束但被取消"分支保持一致：把 run 收敛到取消终态
                # 后立刻发布 CANCELLED AppEvent，避免订阅 replay run 的
                # UI / Service 缺少终态事件、与 run registry 状态漂移。
                cancelled_run = self._finalize_cancelled(run.run_id)
                self._publish_cancelled_app_event(run_id=run.run_id, run=cancelled_run)
                raise
            except Exception as exc:
                if self._is_cancelled(run_id=run.run_id, token=context.cancellation_token):
                    # 取消请求已落库，runner / tool 抛的业务异常本质属于取消余波；
                    # 必须把它收敛成与 run / event 终态一致的 CancelledError，避免
                    # 上层把"取消"误读成"业务失败"。
                    cancelled_run = self._finalize_cancelled(run.run_id)
                    cancel_event = self._publish_cancelled_app_event(
                        run_id=run.run_id, run=cancelled_run,
                    )
                    raise _build_cancelled_error(cancel_event.payload) from exc
                self._fail_run_preserving_original_exception(
                    run_id=run.run_id,
                    error_summary=self._summarize_error(exc, 200),
                )
                raise
        finally:
            self._finish_run(bridge=bridge, deadline_watcher=deadline_watcher, permits=permits, run_id=run.run_id)
        if cancelled_payload is not None:
            raise _build_cancelled_error(cancelled_payload)
        result = AppResult(
            content=content,
            warnings=warnings,
            errors=errors,
            degraded=degraded,
            filtered=filtered,
        )
        new_handle = self._register_replay_state(
            messages=replay_messages,
            agent_input=replay_agent_input,
            session_id=target_session,
            parent_run_id=run.run_id,
        )
        return result, new_handle

    async def _run_agent_and_collect(
        self,
        execution_contract: ExecutionContract,
        *,
        capture: _ReplayCapture | None,
    ) -> tuple[AppResult, _ReplayCapture]:
        """统一收敛 ``run_agent_stream`` 的事件流为 AppResult。

        Args:
            execution_contract: 执行契约。
            capture: 可选 replay 捕获容器；为 ``None`` 时本方法仍会构造一个
                空容器返回，调用方据此判断是否采纳。

        Returns:
            ``(AppResult, _ReplayCapture)`` 二元组。

        Raises:
            CancelledError: 执行被取消时抛出。
            RuntimeError: 未配置 scene preparation 时抛出。
        """

        effective_capture = capture if capture is not None else _ReplayCapture()
        content = ""
        warnings: list[str] = []
        errors: list[str] = []
        degraded = False
        filtered = False
        # 无 replay 需求时仍走公开的 ``run_agent_stream``，以兼容测试对该方法的
        # monkeypatch；带 replay 捕获时才下沉到 ``_run_agent_stream_internal``。
        if capture is None:
            stream = self.run_agent_stream(execution_contract)
        else:
            stream = self._run_agent_stream_internal(
                execution_contract,
                replay_capture=effective_capture,
            )
        async for event in stream:
            if event.type == AppEventType.FINAL_ANSWER:
                payload = event.payload if isinstance(event.payload, dict) else {}
                content = str(payload.get("content") or "")
                degraded = bool(payload.get("degraded", False))
                filtered = bool(payload.get("filtered", False))
            elif event.type == AppEventType.CANCELLED:
                raise _build_cancelled_error(event.payload)
            elif event.type == AppEventType.WARNING:
                warnings.append(_extract_event_message(event.payload))
            elif event.type == AppEventType.ERROR:
                errors.append(_extract_event_message(event.payload))
        result = AppResult(
            content=content,
            warnings=warnings,
            errors=errors,
            degraded=degraded,
            filtered=filtered,
        )
        return result, effective_capture

    def _register_replay_state(
        self,
        *,
        messages: list[AgentMessage],
        agent_input: AgentInput,
        session_id: str | None,
        parent_run_id: str | None = None,
    ) -> ReplayHandle:
        """登记一次执行结束时的 replay 状态并颁发不透明句柄。

        Args:
            messages: 执行结束时的完整对话历史。
            agent_input: 上一次执行使用的 ``AgentInput``，replay 时基于其
                重新构造绑定新 ``CancellationToken`` 的 ``AsyncAgent``。
            session_id: 关联的 session_id；可选。
            parent_run_id: 颁发该 handle 时的来源 run_id，仅作日志辅助。

        Returns:
            可被 ``replay_agent_and_wait`` 消费的句柄。

        Raises:
            无。
        """

        handle_id = f"replay_{uuid.uuid4().hex[:16]}"
        state = _ReplayState(
            messages=list(messages),
            agent_input=agent_input,
            session_id=session_id,
            run_id=parent_run_id or "",
        )
        with self._replay_stash_lock:
            self._replay_stash[handle_id] = state
        return ReplayHandle(handle_id=handle_id)

    def _consume_replay_state(self, handle: ReplayHandle) -> _ReplayState:
        """取出并删除 handle 对应的 replay 状态（一次性消费）。

        Args:
            handle: 上层持有的不透明句柄。

        Returns:
            对应的 replay 状态。

        Raises:
            RuntimeError: handle 在本 Host 实例上不存在。
        """

        with self._replay_stash_lock:
            state = self._replay_stash.pop(handle.handle_id, None)
        if state is None:
            raise RuntimeError(f"无效的 replay handle: {handle.handle_id}")
        return state

    def discard_replay_state_for_session(self, session_id: str) -> None:
        """清理 session 关联的所有 replay 状态。

        Args:
            session_id: 需要清理的 session_id。

        Returns:
            无。

        Raises:
            无。
        """

        with self._replay_stash_lock:
            stale = [
                handle_id
                for handle_id, state in self._replay_stash.items()
                if state.session_id == session_id
            ]
            for handle_id in stale:
                self._replay_stash.pop(handle_id, None)

    def discard_replay_state(self, handle: ReplayHandle) -> None:
        """释放单个未消费的 replay 句柄对应的内存状态。

        Args:
            handle: 待释放的句柄；若 stash 中已不存在则静默跳过。

        Returns:
            无。

        Raises:
            无。
        """

        with self._replay_stash_lock:
            self._replay_stash.pop(handle.handle_id, None)

    async def _run_prepared_agent_stream(
        self,
        *,
        run_id: str,
        session_id: str | None,
        pending_turn_id: str | None,
        agent_input: AgentInput,
        replay_capture: _ReplayCapture | None = None,
    ) -> AsyncIterator[Any]:
        """执行已经准备好的 Agent 输入。

        Args:
            run_id: 当前 Host run ID。
            session_id: 当前执行归属的 Host session_id。
            pending_turn_id: 已登记的 pending turn ID。
            agent_input: 立即可执行的 Agent 输入。
            replay_capture: 可选的 replay 捕获容器；非 ``None`` 时本方法在
                stream 结束时把最终 messages 与所用 ``AsyncAgent`` 实例填入
                容器，供 replay 路径登记到 stash。
        """

        final_content = ""
        degraded = False
        filtered = False
        warnings: list[str] = []
        errors: list[str] = []
        tool_uses: list[object] = []
        normalized_session_id = str(session_id or "").strip() or None
        agent = build_async_agent(
            agent_create_args=agent_input.agent_create_args,
            tool_executor=agent_input.tools,
            tool_trace_recorder_factory=agent_input.tool_trace_recorder_factory,
            trace_identity=agent_input.trace_identity,
            cancellation_token=agent_input.cancellation_handle,
        )
        # ``run_messages`` 会原地追加消息；保留同一份引用以便 replay capture 在
        # stream 结束后取到完整历史。
        running_messages: list[AgentMessage] = list(agent_input.messages)
        async for stream_event in agent.run_messages(
            running_messages,
            session_id=normalized_session_id,
            run_id=run_id,
            stream=True,
        ):
            mapped = build_app_event_from_stream_event(stream_event)
            if mapped is not None:
                self._publish_event(run_id, mapped)
                yield mapped
            if stream_event.type == EventType.TOOL_CALL_RESULT and isinstance(stream_event.data, dict):
                tool_uses.append(_build_conversation_tool_use(stream_event.data))
            elif stream_event.type == EventType.WARNING:
                warnings.append(_extract_event_message(stream_event.data))
            elif stream_event.type == EventType.ERROR:
                errors.append(_extract_event_message(stream_event.data))
            elif stream_event.type == EventType.FINAL_ANSWER and isinstance(stream_event.data, dict):
                final_content = str(stream_event.data.get("content") or "")
                degraded = bool(stream_event.data.get("degraded", False))
                filtered = bool(stream_event.data.get("filtered", False))
            elif stream_event.type == EventType.REASONING_DELTA:
                # reasoning 仅作为历史展示字段 buffer，不进入运行态 transcript / persist。
                # session_state 缺失时静默丢弃；reasoning 是展示侧能力，不应阻塞 run。
                if agent_input.session_state is not None:
                    agent_input.session_state.record_reasoning_delta(
                        str(stream_event.data or "")
                    )
        if replay_capture is not None:
            # 即使被取消或失败也把当前历史快照交给 caller，由调用方决定是否登记。
            replay_capture.messages = list(running_messages)
            replay_capture.agent_input = agent_input
        token = agent_input.cancellation_handle
        if token is not None and self._is_cancelled(run_id=run_id, token=token):
            Log.verbose(f"[{run_id}] 检测到取消，跳过 transcript persist", module=MODULE)
            return
        if token is None and self.run_registry.is_cancel_requested(run_id):
            Log.verbose(f"[{run_id}] 检测到取消请求，跳过 transcript persist", module=MODULE)
            return
        if agent_input.session_state is not None:
            agent_input.session_state.persist_turn(
                final_content=final_content,
                degraded=degraded or filtered,
                tool_uses=tuple(tool_uses),
                warnings=tuple(filter(None, warnings)),
                errors=tuple(filter(None, errors)),
            )
        self._complete_run_preserving_terminal_state(run_id=run_id)
        self._mark_pending_turn_sent_to_llm_best_effort(
            run_id=run_id,
            pending_turn_id=pending_turn_id,
        )

    def _start_run(
        self,
        *,
        spec: HostedRunSpec,
        run_id: str,
        include_agent_lane: bool,
    ) -> tuple[HostedRunContext, CancellationBridge, RunDeadlineWatcher, list[ConcurrencyPermit]]:
        """启动宿主级 run 生命周期。

        Args:
            spec: 宿主 run 规格。
            run_id: 当前 run ID。
            include_agent_lane: 是否需要为本次 run 额外持有 Host 自治 ``llm_api``
                permit；Agent 执行路径由 Host 独立裁决置为 ``True``，通用宿主
                路径为 ``False``。

        Returns:
            run 上下文、取消桥、deadline watcher、已持有的 permit 列表（按字母序）。

        Raises:
            ValueError: ``business_concurrency_lane`` 写入 Host 自治 lane 名时抛出。
        """

        token = CancellationToken()
        if self.run_registry.is_cancel_requested(run_id):
            token.cancel()
            self._finalize_cancelled(run_id)
            raise CancelledError("run 已在启动前收到取消请求")
        bridge = CancellationBridge(
            self.run_registry,
            run_id,
            token,
            poll_interval=self.cancellation_bridge_poll_interval_seconds,
            failure_grace_period_seconds=self.cancellation_bridge_failure_grace_period_seconds,
        )
        deadline_watcher = RunDeadlineWatcher(
            self.run_registry,
            run_id,
            token,
            spec.timeout_ms,
        )
        # bridge / deadline_watcher 已经启动后台线程或 Timer，若后续步骤抛
        # 异常而调用方拿不到句柄，将造成守护线程持续轮询 SQLite 与 Timer
        # 持续到进程退出。这里在剩余初始化阶段统一兜底，确保异常时立刻
        # 释放已启动的资源。此外，若 ``start_run`` 已成功把 run 置为
        # RUNNING，但随后的 ``acquire_many`` 抛异常，run 将永久卡在
        # RUNNING 态（外层 try/finally 要等 ``_start_run`` 返回后才生效），
        # 必须在此显式把 run 收敛为 FAILED，否则 run_registry 状态与真实
        # 执行结果将长期漂移。
        run_started = False
        try:
            self.run_registry.start_run(run_id)
            run_started = True
            bridge.start()
            deadline_watcher.start()
            permits: list[ConcurrencyPermit] = []
            if self.concurrency_governor is not None:
                lanes = _required_lanes_for_spec(spec, include_agent_lane=include_agent_lane)
                if lanes:
                    acquire_timeout = _resolve_concurrency_acquire_timeout_seconds(spec)
                    # 走 acquire_many：单事务原子拿齐全部 lane；
                    # 进程若在两步 acquire 之间被 SIGKILL，SQLite 事务未 COMMIT 等于没写，
                    # 不会残留半截 permit，无需 executor 层 try/except 回滚。
                    permits = self.concurrency_governor.acquire_many(
                        lanes,
                        timeout=acquire_timeout,
                        cancellation_token=token,
                    )
        except BaseException as exc:
            cancelled_during_start = isinstance(exc, CancelledError) or self._is_cancelled(
                run_id=run_id,
                token=token,
            )
            if run_started:
                if cancelled_during_start:
                    with suppress(Exception):
                        self._finalize_cancelled(run_id)
                else:
                    summary: str | None = None
                    if isinstance(exc, Exception):
                        summary = self._summarize_error(exc, spec.error_summary_limit)
                    else:
                        summary = str(exc)[: spec.error_summary_limit] if spec.error_summary_limit else str(exc)
                    with suppress(Exception):
                        self._fail_run_preserving_original_exception(
                            run_id=run_id,
                            error_summary=summary,
                        )
            with suppress(Exception):
                deadline_watcher.stop()
            with suppress(Exception):
                bridge.stop()
            raise
        with self._run_resources_lock:
            self._run_resources[run_id] = _RunResources(
                bridge=bridge,
                deadline_watcher=deadline_watcher,
                permits=list(permits),
            )
        return HostedRunContext(run_id=run_id, cancellation_token=token), bridge, deadline_watcher, permits

    def _release_resources(self, *, resources: _RunResources, log_origin: str, run_id: str) -> None:
        """释放单组 run 资源（permits / deadline_watcher / bridge）。

        资源对象自身的 ``stop`` / governor.release 已经幂等，本方法只负责
        遍历释放并记录失败日志，不做"已停止"标记。

        Args:
            resources: 待释放的资源三元组。
            log_origin: 失败日志中的调用源标识（``"finish_run"`` /
                ``"sigint_release"``），便于事后区分两条释放路径。
            run_id: 当前 run_id，仅用于日志输出。

        Raises:
            无。permit 释放失败仅记日志。
        """

        if self.concurrency_governor is not None:
            for acquired in reversed(resources.permits):
                try:
                    self.concurrency_governor.release(acquired)
                except Exception:  # noqa: BLE001
                    Log.warn(
                        f"permit 释放失败: origin={log_origin}, run_id={run_id}, lane={acquired.lane}",
                        module=MODULE,
                    )
        resources.deadline_watcher.stop()
        resources.bridge.stop()

    def _finish_run(
        self,
        *,
        bridge: CancellationBridge,
        deadline_watcher: RunDeadlineWatcher,
        permits: list[ConcurrencyPermit],
        run_id: str,
    ) -> None:
        """释放宿主级资源（atomic-pop，多入口幂等）。

        与 ``release_resources_for_run`` 共用资源注册表：注册表
        ``pop`` 谁谁释放——SIGINT/SIGTERM 同步路径若已抢先 pop 释放，
        本方法变成 no-op，避免双重释放。

        Args:
            bridge: 当前栈上的取消桥句柄；仅在注册表 pop 命中时使用。
            deadline_watcher: 当前栈上的 deadline watcher 句柄；同上。
            permits: 当前栈上的 permit 列表；同上。
            run_id: 当前 run_id，定位注册表条目。

        Raises:
            无。
        """

        del bridge, deadline_watcher, permits  # 实际释放对象以注册表中 pop 出的为准
        with self._run_resources_lock:
            resources = self._run_resources.pop(run_id, None)
        if resources is None:
            return
        self._release_resources(resources=resources, log_origin="finish_run", run_id=run_id)

    def release_resources_for_run(self, run_id: str) -> None:
        """SIGINT/SIGTERM 同步路径上释放指定 run 的宿主资源（atomic-pop，幂等）。

        ``KeyboardInterrupt`` 同步打断 ``asyncio.run()`` 时，``_finish_run``
        所在的 ``finally`` 不会执行；Host 在 ``cancel_run_and_settle``
        路径上调用本方法，主动从资源注册表 pop 出 ``(bridge, watcher,
        permits)`` 并释放，避免 permit 泄漏与守护线程残留。

        与 ``_finish_run`` 共用注册表：atomic-pop 保证至多一次真实释放。

        Args:
            run_id: 目标 run_id；未在注册表中静默 no-op。

        Raises:
            无。permit 释放失败仅记日志。
        """

        with self._run_resources_lock:
            resources = self._run_resources.pop(run_id, None)
        if resources is None:
            return
        self._release_resources(resources=resources, log_origin="sigint_release", run_id=run_id)

    def _publish_event(self, run_id: str, event: PublishedRunEventProtocol) -> None:
        """向事件总线发布事件。

        Host 只依赖稳定事件包络，不在执行器层绑定具体上层事件实现。
        """

        if self.event_bus is None:
            return
        self.event_bus.publish(run_id, event)

    def _complete_run_preserving_terminal_state(self, *, run_id: str) -> RunRecord:
        """将运行收敛到成功态，并容忍外部已写入的终态。

        Args:
            run_id: 目标 run ID。

        Returns:
            当前最终 run 记录。

        Raises:
            ValueError: run 状态既非成功可修复，也不是已知外部终态时抛出。
        """

        try:
            return self.run_registry.complete_run(run_id)
        except ValueError:
            run = self.run_registry.get_run(run_id)
            if run is None or not run.is_terminal():
                raise
            Log.warn(
                f"[{run_id}] run 已被外部收敛到终态，跳过重复 complete: state={run.state.value}",
                module=MODULE,
            )
            return run

    def _fail_run_preserving_original_exception(
        self,
        *,
        run_id: str,
        error_summary: str | None,
    ) -> RunRecord | None:
        """将运行标记为失败，但不让重复终态写入掩盖原始异常。

        Args:
            run_id: 目标 run ID。
            error_summary: 失败摘要。

        Returns:
            更新后的 run；若 run 已不存在则返回 `None`。

        Raises:
            ValueError: run 仍处于未知非法状态时抛出。
        """

        try:
            return self.run_registry.fail_run(run_id, error_summary=error_summary)
        except ValueError:
            run = self.run_registry.get_run(run_id)
            if run is None or not run.is_terminal():
                raise
            Log.warn(
                f"[{run_id}] run 已被外部收敛到终态，保留原始异常: state={run.state.value}",
                module=MODULE,
            )
            return run

    def _describe_external_terminal_failure(self, run: RunRecord) -> str:
        """格式化外部已落失败终态的错误描述。

        Args:
            run: 已收敛为失败态的 run。

        Returns:
            用户可读错误描述。

        Raises:
            无。
        """

        if run.state == RunState.UNSETTLED and run.owner_pid > 0:
            return (
                "run 已在外部被标记为 orphan/unsettled；"
                f"run_id={run.run_id} owner_pid={run.owner_pid}"
            )
        if run.error_summary:
            return f"run 已在外部失败收口: run_id={run.run_id} error={run.error_summary}"
        return f"run 已在外部失败收口: run_id={run.run_id}"

    def _publish_cancelled_app_event(self, *, run_id: str, run: RunRecord | None) -> AppEvent:
        """构造并发布应用层取消事件。

        Args:
            run_id: 当前 Host run ID。
            run: 已收敛到终态后的 run 记录；缺失时回退到默认取消原因。

        Returns:
            已发布的应用层取消事件。

        Raises:
            无。
        """

        cancel_reason = RunCancelReason.USER_CANCELLED
        if run is not None:
            cancel_reason = run.cancel_reason or run.cancel_requested_reason or RunCancelReason.USER_CANCELLED
        event = AppEvent(
            type=AppEventType.CANCELLED,
            payload={"cancel_reason": cancel_reason.value},
            meta={"run_id": run_id},
        )
        self._publish_event(run_id, event)
        return event

    def _is_cancelled(self, *, run_id: str, token: CancellationToken) -> bool:
        """判断指定 run 是否已被请求取消。

        Args:
            run_id: 目标 run ID。
            token: 当前执行上下文的取消令牌。

        Returns:
            `True` 表示 run 已被取消。
        """

        return token.is_cancelled() or self.run_registry.is_cancel_requested(run_id)

    def _finalize_cancelled(
        self,
        run_id: str,
        *,
        cancel_reason: RunCancelReason = RunCancelReason.USER_CANCELLED,
    ) -> RunRecord | None:
        """把 run 收敛到取消终态。

        Args:
            run_id: 目标 run ID。

        Returns:
            更新后的 RunRecord；run 不存在时返回 `None`。
        """

        run = self.run_registry.get_run(run_id)
        if run is None:
            return None
        if run.state == RunState.CANCELLED or run.is_terminal():
            return run
        effective_cancel_reason = run.cancel_requested_reason or cancel_reason
        return self.run_registry.mark_cancelled(run_id, cancel_reason=effective_cancel_reason)

    def _delete_pending_turn(self, pending_turn_id: str | None) -> None:
        """删除指定 pending turn。

        Args:
            pending_turn_id: 待删除的 pending turn ID。

        Returns:
            无。

        Raises:
            无。
        """

        if pending_turn_id is None or self.pending_turn_store is None:
            return
        self.pending_turn_store.delete_pending_turn(pending_turn_id)
        Log.verbose(f"清理 pending turn: pending_turn_id={pending_turn_id}", module=MODULE)

    def _delete_pending_turn_best_effort(self, *, run_id: str, pending_turn_id: str | None) -> None:
        """在成功态下尽力清理 pending turn，不让清理失败污染 run 终态。

        Args:
            run_id: 当前 Host run ID。
            pending_turn_id: 待清理的 pending turn ID。

        Returns:
            无。

        Raises:
            无。
        """

        if pending_turn_id is None:
            return
        try:
            self._delete_pending_turn(pending_turn_id)
        except Exception as exc:
            Log.warn(
                f"[{run_id}] pending turn 清理失败，但 run 已成功收口，恢复 gate 将拒绝重放: "
                f"pending_turn_id={pending_turn_id}, error={exc}",
                module=MODULE,
            )

    def _mark_pending_turn_sent_to_llm_best_effort(
        self,
        *,
        run_id: str,
        pending_turn_id: str | None,
    ) -> None:
        """在成功态下尽力把 pending turn 推进到 sent_to_llm。

        Args:
            run_id: 当前 Host run ID。
            pending_turn_id: 当前 pending turn ID。

        Returns:
            无。

        Raises:
            无。
        """

        if pending_turn_id is None or self.pending_turn_store is None:
            return
        try:
            self.pending_turn_store.update_state(
                pending_turn_id,
                state=PendingConversationTurnState.SENT_TO_LLM,
            )
            Log.verbose(
                f"[{run_id}] pending turn 已推进到 sent_to_llm: pending_turn_id={pending_turn_id}",
                module=MODULE,
            )
        except Exception as exc:
            Log.warn(
                f"[{run_id}] pending turn sent_to_llm 写入失败，但 run 已成功收口: "
                f"pending_turn_id={pending_turn_id}, error={exc}",
                module=MODULE,
            )

    def _is_run_succeeded(self, run_id: str) -> bool:
        """检查当前 run 是否已经成功收口。"""

        run = self.run_registry.get_run(run_id)
        return bool(run is not None and run.state == RunState.SUCCEEDED)

    def _reconcile_pending_turn_after_terminal_run(
        self,
        *,
        pending_turn_id: str | None,
        run: RunRecord | None,
        resumable: bool,
    ) -> None:
        """根据终态 run 收口 pending turn 生命周期。

        Args:
            pending_turn_id: 当前 pending turn ID。
            run: 已经落终态的 run 记录。
            resumable: 当前 scene 是否允许 resume。

        Returns:
            无。

        Raises:
            无。
        """

        if pending_turn_id is None:
            return
        if run is None:
            Log.verbose(
                f"pending turn 对应 run 记录缺失，执行清理: pending_turn_id={pending_turn_id}",
                module=MODULE,
            )
            self._delete_pending_turn(pending_turn_id)
            return
        if should_delete_pending_turn_after_terminal_run(run=run, resumable=resumable):
            self._delete_pending_turn(pending_turn_id)
            return
        Log.verbose(
            f"[{run.run_id}] run 终态={run.state.value}，保留 pending turn 供恢复: "
            f"pending_turn_id={pending_turn_id}",
            module=MODULE,
        )

    def _settle_agent_cancelled(
        self,
        *,
        run_id: str,
        pending_turn_id: str | None,
        resumable: bool,
    ) -> AppEvent:
        """Agent 取消三步收敛：标记 run 取消 → 调和 pending turn → 发布取消事件。

        Args:
            run_id: 当前 Host run ID。
            pending_turn_id: 已登记的 pending turn ID。
            resumable: 当前 scene 是否允许 resume。

        Returns:
            需要 yield 到事件流的取消事件。
        """

        Log.debug(
            _build_agent_settle_debug_message(
                phase="cancel",
                run_id=run_id,
                pending_turn_id=pending_turn_id,
                resumable=resumable,
            ),
            module=MODULE,
        )
        cancelled_run = self._finalize_cancelled(run_id)
        self._reconcile_pending_turn_after_terminal_run(
            pending_turn_id=pending_turn_id,
            run=cancelled_run,
            resumable=resumable,
        )
        return self._publish_cancelled_app_event(run_id=run_id, run=cancelled_run)

    def _settle_agent_cancelled_before_pending_turn_binding(
        self,
        *,
        run_id: str,
        resumed_pending_turn_id: str | None,
        resumable: bool,
    ) -> AppEvent:
        """收敛发生在 pending turn 绑定前的 Agent 取消。

        该分支只会出现在 `_start_run()` 期间：run 已注册并可能已持有取消意图，
        但 resumed pending turn 还没进入 executor 的常规调和路径。对普通
        agent run，此时只需把 run 收敛为 `CANCELLED`；对 resume 路径，
        必须显式释放 Host 外层已经持有的 `RESUMING` lease，避免记录长期卡在
        `RESUMING` 中间态。

        Args:
            run_id: 当前 Host run ID。
            resumed_pending_turn_id: Host 外层已持有的 resume lease 对应 pending turn ID；
                非 resume 路径为 `None`。
            resumable: 当前 scene 是否允许恢复。

        Returns:
            需要 yield 给调用方的取消事件。

        Raises:
            无。
        """

        Log.debug(
            _build_agent_settle_debug_message(
                phase="cancel-before-bind",
                run_id=run_id,
                pending_turn_id=resumed_pending_turn_id,
                resumable=resumable,
            ),
            module=MODULE,
        )
        cancelled_run = self._finalize_cancelled(run_id)
        self._release_resumed_pending_turn_lease_best_effort(
            run_id=run_id,
            pending_turn_id=resumed_pending_turn_id,
        )
        return self._publish_cancelled_app_event(run_id=run_id, run=cancelled_run)

    def _release_resumed_pending_turn_lease_best_effort(
        self,
        *,
        run_id: str,
        pending_turn_id: str | None,
    ) -> None:
        """best-effort 释放 resume 路径已持有的 RESUMING lease。

        Args:
            run_id: 当前 Host run ID，仅用于日志。
            pending_turn_id: 待释放 lease 的 pending turn ID；为 `None` 时直接返回。

        Returns:
            无。

        Raises:
            无。释放失败只记录告警，最终由 stale-RESUMING cleanup 兜底。
        """

        if pending_turn_id is None or self.pending_turn_store is None:
            return
        try:
            self.pending_turn_store.release_resume_lease(pending_turn_id)
        except Exception as exc:  # noqa: BLE001
            Log.warn(
                "resume 取消前释放 RESUMING lease 失败，依赖 cleanup 兜底: "
                f"run_id={run_id}, pending_turn_id={pending_turn_id}, error={exc}",
                module=MODULE,
            )

    def _settle_agent_stream_completion(
        self,
        *,
        run_id: str,
        token: CancellationToken,
        pending_turn_id: str | None,
        resumable: bool,
    ) -> AppEvent | None:
        """Agent 流正常结束后收敛 run 终态。

        三种终态路径：
        1. ``_run_prepared_agent_stream`` 已内部完成 run → 清理 pending turn。
        2. 检测到取消 → 走取消三步收敛。
        3. 正常完成 → 标记 run 成功并清理 pending turn。

        Args:
            run_id: 当前 Host run ID。
            token: 取消令牌。
            pending_turn_id: 已登记的 pending turn ID。
            resumable: 当前 scene 是否允许 resume。

        Returns:
            取消路径返回需要 yield 的取消事件；其它路径返回 ``None``。
        """

        Log.debug(
            _build_agent_settle_debug_message(
                phase="complete",
                run_id=run_id,
                pending_turn_id=pending_turn_id,
                resumable=resumable,
            ),
            module=MODULE,
        )
        if self._is_run_succeeded(run_id):
            self._delete_pending_turn_best_effort(run_id=run_id, pending_turn_id=pending_turn_id)
            return None
        if self._is_cancelled(run_id=run_id, token=token):
            return self._settle_agent_cancelled(
                run_id=run_id,
                pending_turn_id=pending_turn_id,
                resumable=resumable,
            )
        self._complete_run_preserving_terminal_state(run_id=run_id)
        # 若 run 已被外部（如 cleanup_orphan_runs）收敛为非 SUCCEEDED 终态，
        # 需要保留 pending turn 以便后续恢复路径使用；仅在 run 最终为
        # SUCCEEDED 时才视为本 executor 正常收口，清理 pending turn。
        if self._is_run_succeeded(run_id):
            self._delete_pending_turn_best_effort(run_id=run_id, pending_turn_id=pending_turn_id)
        return None

    def _settle_agent_exception(
        self,
        *,
        run_id: str,
        token: CancellationToken,
        pending_turn_id: str | None,
        resumable: bool,
        exc: Exception,
        error_summary_limit: int,
    ) -> AppEvent | None:
        """处理 Agent 执行异常。

        若异常实际源于取消，走取消三步收敛；否则标记 run 失败并调和 pending turn。

        Args:
            run_id: 当前 Host run ID。
            token: 取消令牌。
            pending_turn_id: 已登记的 pending turn ID。
            resumable: 当前 scene 是否允许 resume。
            exc: 捕获到的异常。
            error_summary_limit: 错误摘要长度限制。

        Returns:
            取消路径返回需要 yield 的取消事件，调用方应 yield 后 return；
            失败路径返回 ``None``，调用方应 raise。
        """

        Log.debug(
            _build_agent_settle_debug_message(
                phase="fail",
                run_id=run_id,
                pending_turn_id=pending_turn_id,
                resumable=resumable,
            ),
            module=MODULE,
        )
        if self._is_cancelled(run_id=run_id, token=token):
            return self._settle_agent_cancelled(
                run_id=run_id,
                pending_turn_id=pending_turn_id,
                resumable=resumable,
            )
        failed_run = self._fail_run_preserving_original_exception(
            run_id=run_id,
            error_summary=self._summarize_error(exc, error_summary_limit),
        )
        self._reconcile_pending_turn_after_terminal_run(
            pending_turn_id=pending_turn_id,
            run=failed_run,
            resumable=resumable,
        )
        return None

    def _finish_sync_cancelled_run(
        self,
        *,
        run_id: str,
        on_cancel: Callable[[], TSyncResult] | None,
    ) -> TSyncResult:
        """收口同步执行的取消结果。

        Args:
            run_id: 目标 run ID。
            on_cancel: 可选取消回调。

        Returns:
            取消回调返回值。

        Raises:
            CancelledError: 未提供取消回调时抛出。
        """

        self._finalize_cancelled(run_id)
        if on_cancel is not None:
            return on_cancel()
        raise CancelledError("操作已被取消")

    @staticmethod
    def _summarize_error(exc: Exception, limit: int) -> str:
        """裁剪异常摘要。"""

        normalized_limit = max(1, int(limit))
        return str(exc)[:normalized_limit]


__all__ = ["DefaultHostExecutor"]


def _extract_event_message(payload: Any) -> str:
    """提取 warning/error 事件的文本。

    Args:
        payload: 事件负载。

    Returns:
        文本消息。

    Raises:
        无。
    """

    if isinstance(payload, dict):
        return str(payload.get("message") or payload.get("error") or "")
    return str(payload or "")


def _build_cancelled_error(payload: Any) -> CancelledError:
    """基于取消事件负载构造统一取消异常。

    Args:
        payload: 取消事件负载。

    Returns:
        统一的取消异常对象。

    Raises:
        无。
    """

    if isinstance(payload, dict):
        cancel_reason = str(payload.get("cancel_reason") or "").strip()
        if cancel_reason:
            return CancelledError(f"操作已被取消: {cancel_reason}")
    return CancelledError("操作已被取消")


def _build_conversation_tool_use(payload: dict[str, Any]) -> ConversationToolUseSummary:
    """把 tool_call_result 事件转成 transcript 摘要对象。

    Args:
        payload: tool_call_result 事件负载。

    Returns:
        ``ConversationToolUseSummary`` 对象。

    Raises:
        无。
    """

    return ConversationToolUseSummary(
        name=str(payload.get("name") or ""),
        arguments=dict(payload.get("arguments") or {}),
        result_summary=_summarize_tool_result(payload.get("result")),
    )


def _summarize_tool_result(result: Any) -> str:
    """将工具结果压缩为 transcript 摘要。

    Args:
        result: 原始工具结果。

    Returns:
        可写入 transcript 的字符串摘要。

    Raises:
        无。
    """

    projected = project_for_llm(result)
    serialized = json.dumps(projected, ensure_ascii=False, sort_keys=True)
    if len(serialized) <= _MAX_TOOL_RESULT_SUMMARY_CHARS:
        return serialized
    suffix = f"...<truncated {len(serialized) - _MAX_TOOL_RESULT_SUMMARY_CHARS} chars>"
    keep = max(0, _MAX_TOOL_RESULT_SUMMARY_CHARS - len(suffix))
    return serialized[:keep] + suffix


def _build_accepted_turn_snapshot(
    execution_contract: ExecutionContract,
) -> AcceptedAgentTurnSnapshot:
    """从 ExecutionContract 投影 Host-owned accepted snapshot。

    Args:
        execution_contract: Service 输出的执行契约。

    Returns:
        Host 内部 accepted snapshot。

    Raises:
        无。
    """

    return AcceptedAgentTurnSnapshot(
        service_name=execution_contract.service_name,
        scene_name=execution_contract.scene_name,
        host_policy=execution_contract.host_policy,
        preparation_spec=execution_contract.preparation_spec,
        message_inputs=execution_contract.message_inputs,
        accepted_execution_spec=execution_contract.accepted_execution_spec,
        execution_options=execution_contract.execution_options,
        metadata=_extract_pending_turn_metadata(execution_contract.metadata),
    )


def _build_run_spec_from_prepared_turn(prepared_turn: PreparedAgentTurnSnapshot) -> HostedRunSpec:
    """从 prepared turn 快照恢复 HostedRunSpec。

    Args:
        prepared_turn: Host 已准备好的恢复快照。

    Returns:
        对应的 HostedRunSpec。

    Raises:
        无。
    """

    session_snapshot = prepared_turn.conversation_session
    session_id = session_snapshot.session_id if session_snapshot is not None else None
    return HostedRunSpec(
        operation_name=prepared_turn.service_name,
        session_id=session_id,
        scene_name=prepared_turn.scene_name,
        metadata=prepared_turn.metadata,
        business_concurrency_lane=prepared_turn.business_concurrency_lane,
        concurrency_acquire_policy=prepared_turn.concurrency_acquire_policy,
        timeout_ms=prepared_turn.timeout_ms,
    )


def _extract_pending_turn_metadata(metadata: ExecutionDeliveryContext) -> ExecutionDeliveryContext:
    """从执行元数据中提取需要随 pending turn 持久化的交付上下文。

    Args:
        metadata: 执行契约交付上下文。

    Returns:
        规范化后的交付上下文。

    Raises:
        无。
    """
    return normalize_execution_delivery_context(metadata)
