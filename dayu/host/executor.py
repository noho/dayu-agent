"""默认宿主执行器实现。

统一管理 Service 侧重复出现的 run 生命周期治理：
- register/start/complete/fail/cancel run
- 创建 CancellationToken 并桥接跨进程取消
- 获取/释放并发许可
- 向 event bus 双写流式事件
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import threading
from typing import Any, AsyncIterator, Callable, TypeVar

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
)
from dayu.contracts.agent_execution import AgentInput, ExecutionContract
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


def _resolve_concurrency_acquire_timeout_seconds(timeout_ms: int | None) -> float:
    """解析并发 permit 获取超时。

    Args:
        timeout_ms: run 级超时毫秒数。

    Returns:
        permit 获取最大等待秒数；未配置 run 超时时返回默认值。

    Raises:
        无。
    """

    if timeout_ms is None:
        return _DEFAULT_CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS
    return max(0.001, timeout_ms / 1000.0)


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
class DefaultHostExecutor(HostExecutorProtocol):
    """默认宿主执行器。"""

    run_registry: RunRegistryProtocol
    pending_turn_store: PendingConversationTurnStoreProtocol | None = None
    concurrency_governor: ConcurrencyGovernorProtocol | None = None
    event_bus: RunEventBusProtocol | None = None
    scene_preparation: ScenePreparationProtocol | None = None

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
            metadata=dict(spec.metadata),
        )
        context, bridge, deadline_watcher, permits = self._start_run(
            spec=spec, run_id=run.run_id, include_agent_lane=False,
        )
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
            self._finish_run(bridge=bridge, deadline_watcher=deadline_watcher, permits=permits)

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
            metadata=dict(spec.metadata),
        )
        context, bridge, deadline_watcher, permits = self._start_run(
            spec=spec, run_id=run.run_id, include_agent_lane=False,
        )
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
            self._finish_run(bridge=bridge, deadline_watcher=deadline_watcher, permits=permits)

    async def run_agent_stream(
        self,
        execution_contract: ExecutionContract,
    ) -> AsyncIterator[Any]:
        """托管一次 Agent 子执行并返回应用层事件流。

        Args:
            execution_contract: Service 输出的执行契约。

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
            timeout_ms=execution_contract.host_policy.timeout_ms,
        )
        run = self.run_registry.register_run(
            session_id=spec.session_id,
            service_type=spec.operation_name,
            scene_name=spec.scene_name,
            metadata=dict(spec.metadata),
        )
        context, bridge, deadline_watcher, permits = self._start_run(
            spec=spec, run_id=run.run_id, include_agent_lane=True,
        )
        pending_turn_id = self._register_accepted_pending_turn(
            execution_contract=execution_contract,
            run_id=run.run_id,
        )
        resumable = bool(execution_contract.host_policy.resumable)
        try:
            prepared_execution = await self.scene_preparation.prepare(execution_contract, context)
            if resumable and prepared_execution.resume_snapshot is None:
                raise RuntimeError("resumable scene preparation 缺少 prepared snapshot")
            if prepared_execution.resume_snapshot is not None:
                pending_turn_id = self._register_prepared_pending_turn(
                    prepared_turn=prepared_execution.resume_snapshot,
                    run_id=run.run_id,
                )
            async for event in self._run_prepared_agent_stream(
                run_id=run.run_id,
                session_id=spec.session_id,
                pending_turn_id=pending_turn_id,
                agent_input=prepared_execution.agent_input,
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
            self._finish_run(bridge=bridge, deadline_watcher=deadline_watcher, permits=permits)

    async def run_prepared_turn_stream(
        self,
        prepared_turn: PreparedAgentTurnSnapshot,
    ) -> AsyncIterator[Any]:
        """基于 prepared turn 快照恢复一次 Agent 子执行。"""

        if self.scene_preparation is None:
            raise RuntimeError("当前 HostExecutor 未配置 scene preparation")

        spec = _build_run_spec_from_prepared_turn(prepared_turn)
        run = self.run_registry.register_run(
            session_id=spec.session_id,
            service_type=spec.operation_name,
            scene_name=spec.scene_name,
            metadata=dict(spec.metadata),
        )
        context, bridge, deadline_watcher, permits = self._start_run(
            spec=spec, run_id=run.run_id, include_agent_lane=True,
        )
        pending_turn_id = self._register_prepared_pending_turn(prepared_turn=prepared_turn, run_id=run.run_id)
        resumable = bool(prepared_turn.resumable)
        try:
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
            self._finish_run(bridge=bridge, deadline_watcher=deadline_watcher, permits=permits)

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

        content = ""
        warnings: list[str] = []
        errors: list[str] = []
        degraded = False
        filtered = False
        async for event in self.run_agent_stream(execution_contract):
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
        return AppResult(
            content=content,
            warnings=warnings,
            errors=errors,
            degraded=degraded,
            filtered=filtered,
        )

    async def _run_prepared_agent_stream(
        self,
        *,
        run_id: str,
        session_id: str | None,
        pending_turn_id: str | None,
        agent_input: AgentInput,
    ) -> AsyncIterator[Any]:
        """执行已经准备好的 Agent 输入。

        Args:
            run_id: 当前 Host run ID。
            session_id: 当前执行归属的 Host session_id。
            pending_turn_id: 已登记的 pending turn ID。
            agent_input: 立即可执行的 Agent 输入。
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
        async for stream_event in agent.run_messages(
            list(agent_input.messages),
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
        bridge = CancellationBridge(self.run_registry, run_id, token)
        bridge.start()
        deadline_watcher = RunDeadlineWatcher(
            self.run_registry,
            run_id,
            token,
            spec.timeout_ms,
        )
        deadline_watcher.start()
        self.run_registry.start_run(run_id)
        permits: list[ConcurrencyPermit] = []
        if self.concurrency_governor is not None:
            lanes = _required_lanes_for_spec(spec, include_agent_lane=include_agent_lane)
            if lanes:
                acquire_timeout = _resolve_concurrency_acquire_timeout_seconds(spec.timeout_ms)
                # 走 acquire_many：单事务原子拿齐全部 lane；
                # 进程若在两步 acquire 之间被 SIGKILL，SQLite 事务未 COMMIT 等于没写，
                # 不会残留半截 permit，无需 executor 层 try/except 回滚。
                permits = self.concurrency_governor.acquire_many(
                    lanes, timeout=acquire_timeout
                )
        return HostedRunContext(run_id=run_id, cancellation_token=token), bridge, deadline_watcher, permits

    def _finish_run(
        self,
        *,
        bridge: CancellationBridge,
        deadline_watcher: RunDeadlineWatcher,
        permits: list[ConcurrencyPermit],
    ) -> None:
        """释放宿主级资源。"""

        if self.concurrency_governor is not None:
            for acquired in reversed(permits):
                try:
                    self.concurrency_governor.release(acquired)
                except Exception:  # noqa: BLE001
                    Log.warn(
                        f"permit 释放失败: lane={acquired.lane}",
                        module=MODULE,
                    )
        deadline_watcher.stop()
        bridge.stop()

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
        if run.state in (RunState.FAILED, RunState.UNSETTLED):
            if resumable:
                Log.verbose(
                    f"[{run.run_id}] run 终态={run.state.value}，保留 pending turn 供恢复: pending_turn_id={pending_turn_id}",
                    module=MODULE,
                )
                return
            self._delete_pending_turn(pending_turn_id)
            return
        if run.state == RunState.CANCELLED:
            if resumable and run.cancel_reason == RunCancelReason.TIMEOUT:
                Log.verbose(
                    f"[{run.run_id}] run timeout 取消，保留 pending turn 供恢复: "
                    f"pending_turn_id={pending_turn_id}",
                    module=MODULE,
                )
                return
            self._delete_pending_turn(pending_turn_id)
            return
        self._delete_pending_turn(pending_turn_id)

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

        cancelled_run = self._finalize_cancelled(run_id)
        self._reconcile_pending_turn_after_terminal_run(
            pending_turn_id=pending_turn_id,
            run=cancelled_run,
            resumable=resumable,
        )
        return self._publish_cancelled_app_event(run_id=run_id, run=cancelled_run)

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
    if len(serialized) <= 2000:
        return serialized
    suffix = f"...<truncated {len(serialized) - 2000} chars>"
    keep = max(0, 2000 - len(suffix))
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
