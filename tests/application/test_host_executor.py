"""DefaultHostExecutor 测试。"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

import dayu.host.executor as executor_module
from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedModelSpec,
    AgentCreateArgs,
    AgentInput,
    ExecutionContract,
    ExecutionDocPermissions,
    ExecutionHostPolicy,
    ExecutionMessageInputs,
    ExecutionPermissions,
    ExecutionWebPermissions,
    ScenePreparationSpec,
)
from dayu.contracts.agent_types import AgentMessage
from dayu.contracts.host_execution import ConcurrencyAcquirePolicy
from dayu.contracts.run import RunCancelReason, RunState
from dayu.execution.options import ConversationMemorySettings, ExecutionOptions
from dayu.host.host import Host
from dayu.host.conversation_store import ConversationTranscript
from dayu.host.host_execution import HostedRunContext, HostedRunSpec
from dayu.host.executor import DefaultHostExecutor
from dayu.host.prepared_turn import (
    AcceptedAgentTurnSnapshot,
    PreparedAgentTurnSnapshot,
    PreparedConversationSessionSnapshot,
    deserialize_accepted_agent_turn_snapshot,
    deserialize_prepared_agent_turn_snapshot,
    serialize_accepted_agent_turn_snapshot,
    serialize_prepared_agent_turn_snapshot,
)
from dayu.host.scene_preparer import PreparedAgentExecution
from dayu.contracts.events import AppEvent, AppEventType
from dayu.contracts.cancellation import CancelledError
from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.engine.events import EventType, StreamEvent
from dayu.host.host_store import HostStore
from dayu.host.run_registry import SQLiteRunRegistry
from dayu.host.pending_turn_store import PendingConversationTurn, PendingConversationTurnState
from dayu.host.protocols import SessionClosedError
from dayu.log import Log
from dayu.contracts.run import ORPHAN_RUN_ERROR_SUMMARY


def _minimal_accepted_execution_spec() -> AcceptedExecutionSpec:
    """构造仅包含模型信息的最小 accepted execution spec。"""

    return AcceptedExecutionSpec(model=AcceptedModelSpec(model_name="test-model"))


@dataclass(frozen=True)
class _Permit:
    """测试用 permit。"""

    permit_id: str
    lane: str
    acquired_at: object = object()


class _StubGovernor:
    """测试用并发治理器。"""

    def __init__(self) -> None:
        self.acquired: list[str] = []
        self.acquire_timeouts: list[float | None] = []
        self.acquire_cancellation_tokens: list[executor_module.CancellationToken | None] = []
        self.released: list[str] = []

    def acquire(
        self,
        lane: str,
        *,
        timeout: float | None = None,
        cancellation_token: executor_module.CancellationToken | None = None,
    ) -> _Permit:
        self.acquired.append(lane)
        self.acquire_timeouts.append(timeout)
        self.acquire_cancellation_tokens.append(cancellation_token)
        return _Permit(permit_id=f"permit-{lane}", lane=lane)

    def acquire_many(
        self,
        lanes: list[str],
        *,
        timeout: float | None = None,
        cancellation_token: executor_module.CancellationToken | None = None,
    ) -> list[_Permit]:
        """一次性拿齐多 lane 的测试实现：转发给 acquire 逐个累积。"""

        return [
            self.acquire(
                lane_name,
                timeout=timeout,
                cancellation_token=cancellation_token,
            )
            for lane_name in lanes
        ]

    def try_acquire(self, lane: str):
        del lane
        return None

    def release(self, permit: _Permit) -> None:
        self.released.append(permit.lane)

    def get_lane_status(self, lane: str):
        raise NotImplementedError()

    def get_all_status(self):
        raise NotImplementedError()

    def cleanup_stale_permits(self):
        return []


class _StubEventBus:
    """测试用事件总线。"""

    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    def publish(self, run_id: str, event: object) -> None:
        self.published.append((run_id, event))

    def subscribe(self, *, run_id: str | None = None, session_id: str | None = None):
        raise NotImplementedError()


def _build_prepared_execution(
    *,
    execution_contract: ExecutionContract,
    system_prompt: str = "sys",
    messages: list[AgentMessage] | None = None,
) -> PreparedAgentExecution:
    """构造测试用 prepared execution。"""

    normalized_messages = messages or [{"role": "user", "content": str(execution_contract.message_inputs.user_message or "")}]
    agent_input = AgentInput(
        system_prompt=system_prompt,
        messages=normalized_messages,
        agent_create_args=AgentCreateArgs(runner_type="openai", model_name="test-model"),
    )
    session_id = str(execution_contract.host_policy.session_key or "").strip()
    resume_snapshot = PreparedAgentTurnSnapshot(
        service_name=execution_contract.service_name,
        scene_name=execution_contract.scene_name,
        metadata=execution_contract.metadata,
        business_concurrency_lane=execution_contract.host_policy.business_concurrency_lane,
        timeout_ms=execution_contract.host_policy.timeout_ms,
        resumable=bool(execution_contract.host_policy.resumable),
        system_prompt=system_prompt,
        messages=normalized_messages,
        agent_create_args=agent_input.agent_create_args,
        selected_toolsets=execution_contract.preparation_spec.selected_toolsets,
        execution_permissions=execution_contract.preparation_spec.execution_permissions,
        toolset_configs=execution_contract.accepted_execution_spec.tools.toolset_configs,
        trace_settings=execution_contract.accepted_execution_spec.infrastructure.trace_settings,
        conversation_memory_settings=ConversationMemorySettings(),
        conversation_session=(
            None
            if not session_id
            else PreparedConversationSessionSnapshot(
                session_id=session_id,
                user_message=str(execution_contract.message_inputs.user_message or ""),
                transcript=ConversationTranscript.create_empty(session_id),
            )
        ),
    )
    return PreparedAgentExecution(agent_input=agent_input, resume_snapshot=resume_snapshot)


@pytest.mark.unit
def test_run_stream_manages_run_lifecycle_and_event_publish() -> None:
    """流式执行应统一管理 run 生命周期。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    governor = _StubGovernor()
    event_bus = _StubEventBus()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        concurrency_governor=governor,  # type: ignore[arg-type]
        event_bus=event_bus,  # type: ignore[arg-type]
    )
    spec = HostedRunSpec(operation_name="prompt", session_id="s1", business_concurrency_lane="sec_download")

    async def _stream(context: HostedRunContext):
        assert context.run_id
        assert context.cancellation_token.is_cancelled() is False
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="hello", meta={})

    async def _collect() -> list[AppEvent]:
        events: list[AppEvent] = []
        async for event in executor.run_operation_stream(spec=spec, event_stream_factory=_stream):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert len(events) == 1
    assert governor.acquired == ["sec_download"]
    assert governor.acquire_timeouts == [executor_module._DEFAULT_CONCURRENCY_ACQUIRE_TIMEOUT_SECONDS]
    assert governor.released == ["sec_download"]
    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "succeeded"
    assert event_bus.published == [(run.run_id, events[0])]


@pytest.mark.unit
def test_run_stream_uses_explicit_concurrency_timeout_policy() -> None:
    """显式有限等待策略应覆盖 Host 默认 acquire timeout。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    governor = _StubGovernor()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        concurrency_governor=governor,  # type: ignore[arg-type]
    )
    spec = HostedRunSpec(
        operation_name="prompt",
        session_id="s1",
        business_concurrency_lane="sec_download",
        concurrency_acquire_policy=ConcurrencyAcquirePolicy.with_timeout(7.5),
    )

    async def _stream(_context: HostedRunContext):
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="hello", meta={})

    async def _collect() -> None:
        async for _event in executor.run_operation_stream(spec=spec, event_stream_factory=_stream):
            pass

    asyncio.run(_collect())

    assert governor.acquire_timeouts == [7.5]


@pytest.mark.unit
def test_run_stream_marks_failed_when_acquire_many_raises_after_start_run() -> None:
    """start_run 成功后 acquire_many 抛异常时，run 必须被显式收敛为 FAILED。

    回归点：
    - `_start_run` 先调用 `run_registry.start_run`，随后调用 `concurrency_governor.acquire_many`。
    - 若 `acquire_many` 抛 TimeoutError/RuntimeError，run 已是 RUNNING，外层
      `run_operation_stream` 还没进入自己的 try/finally，需要 `_start_run` 自身兜底
      把 run 写成 FAILED，否则 run 永远卡 RUNNING 态。
    """

    from tests.application.conftest import StubRunRegistry

    class _FailingGovernor(_StubGovernor):
        """acquire_many 会抛 TimeoutError 的治理器。"""

        def acquire_many(
            self,
            lanes: list[str],
            *,
            timeout: float | None = None,
            cancellation_token: executor_module.CancellationToken | None = None,
        ) -> list[_Permit]:
            del lanes, timeout, cancellation_token
            raise TimeoutError("acquire timed out")

    run_registry = StubRunRegistry()
    governor = _FailingGovernor()
    event_bus = _StubEventBus()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        concurrency_governor=governor,  # type: ignore[arg-type]
        event_bus=event_bus,  # type: ignore[arg-type]
    )
    spec = HostedRunSpec(operation_name="prompt", session_id="s1", business_concurrency_lane="sec_download")

    async def _stream(_context: HostedRunContext):
        # 不会进入——acquire_many 先抛
        if False:
            yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="", meta={})

    async def _collect() -> None:
        async for _event in executor.run_operation_stream(spec=spec, event_stream_factory=_stream):
            pass

    with pytest.raises(TimeoutError, match="acquire timed out"):
        asyncio.run(_collect())

    runs = list(run_registry._runs.values())
    assert len(runs) == 1, "register_run 应当只登记一条 run"
    run = runs[0]
    assert run.state == RunState.FAILED, f"acquire 失败后 run 应被收敛为 FAILED，实际为 {run.state}"
    assert run.completed_at is not None, "FAILED 终态必须带 completed_at"
    # 守护资源也应被清理：governor 在 acquire_many 抛出前未把 permit 放进 acquired
    assert governor.acquired == [], "acquire_many 抛异常时不应残留已获取的 permit 记录"


@pytest.mark.unit
def test_start_run_marks_failed_when_deadline_watcher_start_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bridge 已启动但 deadline watcher 启动失败时，run 必须收敛为 FAILED 且资源被停止。"""

    from tests.application.conftest import StubRunRegistry

    class _TrackingBridge:
        """记录 start/stop 调用的取消桥桩。"""

        instances: list["_TrackingBridge"] = []

        def __init__(
            self,
            run_registry: object,
            run_id: str,
            token: executor_module.CancellationToken,
            poll_interval: float = 0.5,
            failure_grace_period_seconds: float = 5.0,
        ) -> None:
            del run_registry, run_id, token, poll_interval, failure_grace_period_seconds
            self.started = False
            self.stopped = False
            _TrackingBridge.instances.append(self)

        def start(self) -> None:
            """记录已启动。"""

            self.started = True

        def stop(self) -> None:
            """记录已停止。"""

            self.stopped = True

    class _FailingDeadlineWatcher:
        """启动时抛异常的 deadline watcher 桩。"""

        instances: list["_FailingDeadlineWatcher"] = []

        def __init__(
            self,
            run_registry: object,
            run_id: str,
            token: executor_module.CancellationToken,
            timeout_ms: int | None,
        ) -> None:
            del run_registry, run_id, token, timeout_ms
            self.stop_calls = 0
            _FailingDeadlineWatcher.instances.append(self)

        def start(self) -> None:
            """模拟 watcher 启动失败。"""

            raise RuntimeError("deadline watcher boom")

        def stop(self) -> None:
            """记录 stop 调用。"""

            self.stop_calls += 1

    run_registry = StubRunRegistry()
    run = run_registry.register_run(session_id="s1", service_type="prompt")
    executor = DefaultHostExecutor(run_registry=run_registry)
    spec = HostedRunSpec(operation_name="prompt", session_id="s1")

    monkeypatch.setattr(executor_module, "CancellationBridge", _TrackingBridge)
    monkeypatch.setattr(executor_module, "RunDeadlineWatcher", _FailingDeadlineWatcher)

    with pytest.raises(RuntimeError, match="deadline watcher boom"):
        executor._start_run(spec=spec, run_id=run.run_id, include_agent_lane=False)

    final_run = run_registry.get_run(run.run_id)
    assert final_run is not None
    assert final_run.state == RunState.FAILED
    assert final_run.completed_at is not None
    assert run_registry.list_active_runs() == []
    assert len(_TrackingBridge.instances) == 1
    assert _TrackingBridge.instances[0].started is True
    assert _TrackingBridge.instances[0].stopped is True
    assert len(_FailingDeadlineWatcher.instances) == 1
    assert _FailingDeadlineWatcher.instances[0].stop_calls == 1


@pytest.mark.unit
def test_start_run_short_circuits_when_cancel_already_requested() -> None:
    """启动前已存在取消意图时，不应推 RUNNING，也不应申请 permit。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    governor = _StubGovernor()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        concurrency_governor=governor,  # type: ignore[arg-type]
        event_bus=_StubEventBus(),  # type: ignore[arg-type]
    )
    spec = HostedRunSpec(
        operation_name="write_pipeline",
        session_id="s1",
        business_concurrency_lane="write_chapter",
    )
    run = run_registry.register_run(
        session_id="s1",
        service_type="write_pipeline",
        scene_name=spec.scene_name,
    )
    requested = run_registry.request_cancel(run.run_id, cancel_reason=RunCancelReason.USER_CANCELLED)
    assert requested is True

    with pytest.raises(CancelledError, match="启动前收到取消请求"):
        executor._start_run(spec=spec, run_id=run.run_id, include_agent_lane=True)

    final_run = run_registry.get_run(run.run_id)
    assert final_run is not None
    assert final_run.state == RunState.CANCELLED
    assert final_run.cancel_reason == RunCancelReason.USER_CANCELLED
    assert final_run.started_at is None
    assert governor.acquired == []
    assert governor.released == []


@pytest.mark.unit
def test_run_sync_marks_cancelled_and_uses_on_cancel() -> None:
    """同步执行取消时应标记 run 并走 on_cancel。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    executor = DefaultHostExecutor(run_registry=run_registry)
    spec = HostedRunSpec(operation_name="write_pipeline", session_id="s1")

    def _operation(_context: HostedRunContext) -> int:
        raise CancelledError()

    result = executor.run_operation_sync(spec=spec, operation=_operation, on_cancel=lambda: 1)

    run = next(iter(run_registry._runs.values()))
    assert result == 1
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.USER_CANCELLED


@pytest.mark.unit
def test_run_sync_treats_external_cancel_as_cancelled() -> None:
    """外部先请求取消、业务后返回时，执行器应保持取消终态。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    executor = DefaultHostExecutor(run_registry=run_registry)
    spec = HostedRunSpec(operation_name="write_pipeline", session_id="s1")

    def _operation(context: HostedRunContext) -> int:
        run_registry.request_cancel(context.run_id)
        return 42

    result = executor.run_operation_sync(spec=spec, operation=_operation, on_cancel=lambda: 1)

    run = next(iter(run_registry._runs.values()))
    assert result == 1
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.USER_CANCELLED


@pytest.mark.unit
def test_run_sync_treats_external_cancelled_failure_as_cancelled() -> None:
    """外部取消后即使业务抛异常，也不应再尝试写失败终态。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    executor = DefaultHostExecutor(run_registry=run_registry)
    spec = HostedRunSpec(operation_name="write_pipeline", session_id="s1")

    def _operation(context: HostedRunContext) -> int:
        run_registry.request_cancel(context.run_id)
        raise RuntimeError("cancelled late")

    result = executor.run_operation_sync(spec=spec, operation=_operation, on_cancel=lambda: 1)

    run = next(iter(run_registry._runs.values()))
    assert result == 1
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.USER_CANCELLED


@pytest.mark.unit
def test_run_sync_recovers_owned_orphan_failure_before_success(tmp_path: Path) -> None:
    """同步执行若被误判为 UNSETTLED orphan，当前 owner 仍可成功收口。"""

    host_store = HostStore(tmp_path / "host.db")
    host_store.initialize_schema()
    run_registry = SQLiteRunRegistry(host_store)
    executor = DefaultHostExecutor(run_registry=run_registry)
    spec = HostedRunSpec(operation_name="write_pipeline", session_id="s1")

    def _operation(context: HostedRunContext) -> int:
        run_registry.mark_unsettled(context.run_id, error_summary=ORPHAN_RUN_ERROR_SUMMARY)
        return 42

    result = executor.run_operation_sync(spec=spec, operation=_operation)
    run = next(iter(run_registry.list_runs()))

    assert result == 42
    assert run.state.value == "succeeded"
    assert run.error_summary is None


@pytest.mark.unit
def test_run_sync_preserves_original_exception_when_run_already_unsettled_externally(tmp_path: Path) -> None:
    """外部写入 UNSETTLED 终态时，执行器不应再把异常掩盖成状态机错误。"""

    host_store = HostStore(tmp_path / "host.db")
    host_store.initialize_schema()
    run_registry = SQLiteRunRegistry(host_store)
    executor = DefaultHostExecutor(run_registry=run_registry)
    spec = HostedRunSpec(operation_name="write_pipeline", session_id="s1")

    def _operation(context: HostedRunContext) -> int:
        run_registry.mark_unsettled(context.run_id, error_summary=ORPHAN_RUN_ERROR_SUMMARY)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        executor.run_operation_sync(spec=spec, operation=_operation)

    run = next(iter(run_registry.list_runs()))
    assert run.state.value == "unsettled"
    assert run.error_summary == ORPHAN_RUN_ERROR_SUMMARY


@pytest.mark.unit
def test_run_stream_treats_external_cancel_as_cancelled() -> None:
    """流式执行在外部取消后结束时应保持取消终态。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    executor = DefaultHostExecutor(run_registry=run_registry)
    spec = HostedRunSpec(operation_name="prompt", session_id="s1")

    async def _stream(context: HostedRunContext):
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="hello", meta={})
        run_registry.request_cancel(context.run_id)

    async def _collect() -> list[AppEvent]:
        events: list[AppEvent] = []
        async for event in executor.run_operation_stream(spec=spec, event_stream_factory=_stream):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert len(events) == 1
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.USER_CANCELLED


@pytest.mark.unit
def test_run_stream_marks_timeout_as_timeout_cancel_reason() -> None:
    """deadline watcher 超时后应收敛到 timeout cancel。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    executor = DefaultHostExecutor(run_registry=run_registry)
    spec = HostedRunSpec(operation_name="prompt", session_id="s1", timeout_ms=30)

    async def _stream(context: HostedRunContext):
        if False:
            yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="never", meta={})
        while True:
            await asyncio.sleep(0.01)
            context.cancellation_token.raise_if_cancelled()

    async def _collect() -> list[AppEvent]:
        events: list[AppEvent] = []
        async for event in executor.run_operation_stream(spec=spec, event_stream_factory=_stream):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert events == []
    assert run.cancel_requested_at is not None
    assert run.cancel_requested_reason == RunCancelReason.TIMEOUT
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.TIMEOUT


@pytest.mark.unit
def test_run_agent_stream_cleans_pending_turn_before_marking_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent 成功执行后应先清理 pending turn，再把 run 标记为成功。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            return _build_prepared_execution(execution_contract=execution_contract)

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del run_context
            return AgentInput(
                system_prompt=prepared_turn.system_prompt,
                messages=list(prepared_turn.messages),
                agent_create_args=prepared_turn.agent_create_args,
            )

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive", "interactive_key": "cli-default"},
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    asyncio.run(_collect())

    pending_turns = pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive")
    assert pending_turns == []
    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "succeeded"


@pytest.mark.unit
def test_run_agent_stream_keeps_session_id_for_non_resumable_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 resumable 场景也必须把当前 Host session_id 传给 Agent。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
            return PreparedAgentExecution(
                agent_input=prepared_execution.agent_input,
                resume_snapshot=None,
            )

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    captured: dict[str, object] = {}

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, run_id, stream
            captured["session_id"] = session_id
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s-non-resumable", resumable=False),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    asyncio.run(_collect())

    assert captured["session_id"] == "s-non-resumable"
    pending_turns = pending_turn_store.list_pending_turns(session_id="s-non-resumable", scene_name="interactive")
    assert pending_turns == []


@pytest.mark.unit
def test_run_agent_stream_keeps_accepted_pending_turn_when_prepare_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """prepare 阶段 timeout 时，resumable turn 仍应保留 accepted 真源。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del execution_contract
            run_registry.request_cancel(run_context.run_id, cancel_reason=RunCancelReason.TIMEOUT)
            run_context.cancellation_token.cancel()
            raise CancelledError("prepare timeout")

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(
        "dayu.host.executor.build_async_agent",
        lambda **_: (_ for _ in ()).throw(AssertionError("prepare timeout 不应进入 agent 构造")),
    )
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.TIMEOUT
    pending_turns = pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive")
    assert len(pending_turns) == 1
    assert pending_turns[0].state == PendingConversationTurnState.ACCEPTED_BY_HOST
    snapshot_payload = json.loads(pending_turns[0].resume_source_json)
    assert snapshot_payload["message_inputs"]["user_message"] == "问题"


@pytest.mark.unit
def test_run_agent_stream_keeps_prepared_pending_turn_when_timeout_occurs_before_first_agent_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首个 Agent 事件前 timeout 时，pending turn 应保持 prepared 真源。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            run_registry.request_cancel(run_context.run_id, cancel_reason=RunCancelReason.TIMEOUT)
            run_context.cancellation_token.cancel()
            return _build_prepared_execution(execution_contract=execution_contract)

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, stream
            raise CancelledError(f"cancelled before first event: {run_id}")
            yield  # pragma: no cover

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    log_calls: list[str] = []
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    monkeypatch.setattr(
        Log,
        "verbose",
        lambda message, *, module: log_calls.append(f"{module}:{message}"),
    )
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive", "interactive_key": "cli-default"},
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.TIMEOUT
    pending_turns = pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive")
    assert len(pending_turns) == 1
    assert pending_turns[0].state == PendingConversationTurnState.PREPARED_BY_HOST
    assert pending_turns[0].metadata == {
        "delivery_channel": "interactive",
        "interactive_key": "cli-default",
    }
    snapshot_payload = json.loads(pending_turns[0].resume_source_json)
    assert snapshot_payload["conversation_session"]["user_message"] == "问题"
    assert snapshot_payload["metadata"] == {
        "delivery_channel": "interactive",
        "interactive_key": "cli-default",
    }
    assert not any("sent_to_llm" in item for item in log_calls)


@pytest.mark.unit
def test_run_agent_stream_keeps_prepared_pending_turn_when_timeout_occurs_after_first_agent_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """首个 Agent 事件后 timeout 时，pending turn 仍应保持 prepared 真源。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            run_registry.request_cancel(run_context.run_id, cancel_reason=RunCancelReason.TIMEOUT)
            run_context.cancellation_token.cancel()
            return _build_prepared_execution(execution_contract=execution_contract)

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})
            raise CancelledError(f"cancelled: {run_id}")

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive", "interactive_key": "cli-default"},
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.TIMEOUT
    pending_turns = pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive")
    assert len(pending_turns) == 1
    assert pending_turns[0].state == PendingConversationTurnState.PREPARED_BY_HOST
    assert pending_turns[0].metadata == {
        "delivery_channel": "interactive",
        "interactive_key": "cli-default",
    }
    snapshot_payload = json.loads(pending_turns[0].resume_source_json)
    assert snapshot_payload["conversation_session"]["user_message"] == "问题"
    assert snapshot_payload["metadata"] == {
        "delivery_channel": "interactive",
        "interactive_key": "cli-default",
    }
    assert pending_turns[0].resume_source_json == json.dumps(snapshot_payload, ensure_ascii=False, sort_keys=True)

@pytest.mark.unit
def test_run_agent_stream_skips_persist_turn_after_external_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent 已给出回答但随后被取消时，不应再写入 transcript。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _RecordingSessionState:
        def __init__(self) -> None:
            self.persist_calls: list[dict[str, object]] = []

        def record_reasoning_delta(self, chunk: str) -> None:
            del chunk

        def persist_turn(self, **kwargs) -> None:
            self.persist_calls.append(kwargs)

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
            return PreparedAgentExecution(
                agent_input=AgentInput(
                    system_prompt=prepared_execution.agent_input.system_prompt,
                    messages=list(prepared_execution.agent_input.messages),
                    agent_create_args=prepared_execution.agent_input.agent_create_args,
                    session_state=session_state,
                ),
                resume_snapshot=prepared_execution.resume_snapshot,
            )

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})
            run_registry.request_cancel(run_id, cancel_reason=RunCancelReason.USER_CANCELLED)

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    session_state = _RecordingSessionState()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )

    async def _collect() -> list[AppEvent]:
        events: list[AppEvent] = []
        async for event in executor.run_agent_stream(execution_contract):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.USER_CANCELLED
    assert session_state.persist_calls == []
    assert [event.type for event in events].count(AppEventType.CANCELLED) == 1
    assert events[-1].type == AppEventType.CANCELLED
    assert events[-1].payload == {"cancel_reason": RunCancelReason.USER_CANCELLED.value}


@pytest.mark.unit
def test_run_agent_stream_emits_verbose_logs_for_pending_turn_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """pending turn 生命周期应输出 accepted/prepared/sent/cleanup verbose 日志。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _RecordingSessionState:
        def __init__(self) -> None:
            self.persist_calls = 0
            self.pending_states_during_persist: list[PendingConversationTurnState] = []

        def record_reasoning_delta(self, chunk: str) -> None:
            del chunk

        def persist_turn(self, **_kwargs) -> None:
            self.persist_calls += 1
            self.pending_states_during_persist = [
                record.state
                for record in pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive")
            ]

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
            return PreparedAgentExecution(
                agent_input=AgentInput(
                    system_prompt=prepared_execution.agent_input.system_prompt,
                    messages=list(prepared_execution.agent_input.messages),
                    agent_create_args=prepared_execution.agent_input.agent_create_args,
                    session_state=session_state,
                ),
                resume_snapshot=prepared_execution.resume_snapshot,
            )

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})

    verbose_mock = pytest.MonkeyPatch()
    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    session_state = _RecordingSessionState()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    log_calls: list[str] = []
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    monkeypatch.setattr(
        Log,
        "verbose",
        lambda message, *, module: log_calls.append(f"{module}:{message}"),
    )
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    asyncio.run(_collect())

    assert session_state.persist_calls == 1
    assert session_state.pending_states_during_persist == [PendingConversationTurnState.PREPARED_BY_HOST]
    assert pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive") == []
    assert any("HOST.EXECUTOR:" in item and "accepted 真源" in item for item in log_calls)
    assert any("HOST.EXECUTOR:" in item and "prepared 真源" in item for item in log_calls)
    assert any("HOST.EXECUTOR:" in item and "sent_to_llm" in item for item in log_calls)
    assert any("HOST.EXECUTOR:" in item and "清理 pending turn" in item for item in log_calls)


@pytest.mark.unit
def test_run_agent_stream_emits_debug_logs_for_start_and_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """正常完成路径会输出启动、收敛与临时探针日志。"""

    from tests.application.conftest import StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            return _build_prepared_execution(execution_contract=execution_contract)

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})

    debug_logs: list[str] = []
    executor = DefaultHostExecutor(
        run_registry=StubRunRegistry(),
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    monkeypatch.setattr(Log, "debug", lambda message, *, module: debug_logs.append(f"{module}:{message}"))
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True, timeout_ms=1234),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    asyncio.run(_collect())

    assert any("Host 启动 agent run" in item and "scene_name=interactive" in item and "service_type=chat_turn" in item for item in debug_logs)
    assert any("Host 收敛 agent run: phase=complete" in item for item in debug_logs)


@pytest.mark.unit
def test_run_agent_stream_emits_debug_logs_for_fail_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """异常失败路径会输出 fail 收敛日志。"""

    from tests.application.conftest import StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            return _build_prepared_execution(execution_contract=execution_contract)

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FailingAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            raise RuntimeError("boom")
            yield

    debug_logs: list[str] = []
    executor = DefaultHostExecutor(
        run_registry=StubRunRegistry(),
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FailingAgent())
    monkeypatch.setattr(Log, "debug", lambda message, *, module: debug_logs.append(f"{module}:{message}"))
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=False),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(_collect())

    assert any("Host 收敛 agent run: phase=fail" in item for item in debug_logs)


@pytest.mark.unit
def test_run_agent_stream_emits_debug_logs_for_cancel_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """取消路径会输出 cancel 收敛日志。"""

    from tests.application.conftest import StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            run_registry.request_cancel(run_context.run_id, cancel_reason=RunCancelReason.USER_CANCELLED)
            run_context.cancellation_token.cancel()
            return _build_prepared_execution(execution_contract=execution_contract)

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})

    debug_logs: list[str] = []
    run_registry = StubRunRegistry()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    monkeypatch.setattr(Log, "debug", lambda message, *, module: debug_logs.append(f"{module}:{message}"))
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    asyncio.run(_collect())

    assert any("Host 收敛 agent run: phase=cancel" in item for item in debug_logs)


@pytest.mark.unit
def test_run_agent_stream_deletes_pending_turn_for_user_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    """用户取消后不应保留 pending turn。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            run_registry.request_cancel(run_context.run_id, cancel_reason=RunCancelReason.USER_CANCELLED)
            run_context.cancellation_token.cancel()
            return _build_prepared_execution(execution_contract=execution_contract)

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})
            raise CancelledError(f"cancelled: {run_id}")

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        metadata={"delivery_channel": "interactive"},
    )

    async def _collect() -> list[AppEvent]:
        events: list[AppEvent] = []
        async for event in executor.run_agent_stream(execution_contract):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.USER_CANCELLED
    assert pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive") == []
    assert [event.type for event in events].count(AppEventType.CANCELLED) == 1
    assert events[-1].type == AppEventType.CANCELLED
    assert events[-1].payload == {"cancel_reason": RunCancelReason.USER_CANCELLED.value}


@pytest.mark.unit
def test_run_prepared_turn_stream_yields_cancelled_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """恢复执行被取消时，事件流也必须显式产出 CANCELLED。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del execution_contract, run_context
            raise AssertionError("该测试不应走 prepare 路径")

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            run_registry.request_cancel(run_context.run_id, cancel_reason=RunCancelReason.USER_CANCELLED)
            run_context.cancellation_token.cancel()
            return AgentInput(
                system_prompt=prepared_turn.system_prompt,
                messages=list(prepared_turn.messages),
                agent_create_args=prepared_turn.agent_create_args,
            )

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, stream
            raise CancelledError(f"cancelled prepared turn: {run_id}")
            yield  # pragma: no cover

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )
    prepared_turn = _build_prepared_execution(execution_contract=execution_contract).resume_snapshot
    assert prepared_turn is not None

    async def _collect() -> list[AppEvent]:
        events: list[AppEvent] = []
        async for event in executor.run_prepared_turn_stream(prepared_turn):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "cancelled"
    assert run.cancel_reason == RunCancelReason.USER_CANCELLED
    assert [event.type for event in events] == [AppEventType.CANCELLED]
    assert events[0].payload == {"cancel_reason": RunCancelReason.USER_CANCELLED.value}


@pytest.mark.unit
def test_run_agent_stream_keeps_prepared_pending_turn_when_agent_build_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """build_async_agent 异常时，resumable turn 应保留 prepared 真源。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            return _build_prepared_execution(execution_contract=execution_contract)

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )

    def _raise_build_error(**_kwargs):
        raise RuntimeError("build failed")

    monkeypatch.setattr("dayu.host.executor.build_async_agent", _raise_build_error)
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    with pytest.raises(RuntimeError, match="build failed"):
        asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "failed"
    pending_turns = pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive")
    assert len(pending_turns) == 1
    assert pending_turns[0].state == PendingConversationTurnState.PREPARED_BY_HOST


@pytest.mark.unit
def test_run_agent_stream_keeps_prepared_pending_turn_when_persist_turn_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """persist_turn 异常时，resumable turn 仍应保留 prepared 真源。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _BrokenSessionState:
        def record_reasoning_delta(self, chunk: str) -> None:
            del chunk

        def persist_turn(self, **_kwargs) -> None:
            raise RuntimeError("persist failed")

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
            return PreparedAgentExecution(
                agent_input=AgentInput(
                    system_prompt=prepared_execution.agent_input.system_prompt,
                    messages=list(prepared_execution.agent_input.messages),
                    agent_create_args=prepared_execution.agent_input.agent_create_args,
                    session_state=_BrokenSessionState(),
                ),
                resume_snapshot=prepared_execution.resume_snapshot,
            )

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    with pytest.raises(RuntimeError, match="persist failed"):
        asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "failed"
    pending_turns = pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive")
    assert len(pending_turns) == 1
    assert pending_turns[0].state == PendingConversationTurnState.PREPARED_BY_HOST


@pytest.mark.unit
def test_run_agent_stream_keeps_success_when_sent_to_llm_update_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """transcript 已成功持久化后，sent_to_llm 写入失败不应把 run 降级为 failed。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    class _RecordingSessionState:
        def __init__(self) -> None:
            self.persist_calls = 0

        def record_reasoning_delta(self, chunk: str) -> None:
            del chunk

        def persist_turn(self, **_kwargs) -> None:
            self.persist_calls += 1

    class _BrokenUpdatePendingTurnStore(StubPendingTurnStore):
        def update_state(self, pending_turn_id: str, *, state: PendingConversationTurnState):
            del pending_turn_id, state
            raise RuntimeError("update_state failed")

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
            return PreparedAgentExecution(
                agent_input=AgentInput(
                    system_prompt=prepared_execution.agent_input.system_prompt,
                    messages=list(prepared_execution.agent_input.messages),
                    agent_create_args=prepared_execution.agent_input.agent_create_args,
                    session_state=session_state,
                ),
                resume_snapshot=prepared_execution.resume_snapshot,
            )

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})

    run_registry = StubRunRegistry()
    pending_turn_store = _BrokenUpdatePendingTurnStore()
    session_state = _RecordingSessionState()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )

    async def _collect() -> list[AppEvent]:
        events: list[AppEvent] = []
        async for event in executor.run_agent_stream(execution_contract):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "succeeded"
    assert session_state.persist_calls == 1
    assert pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive") == []
    assert [event.type for event in events] == [AppEventType.FINAL_ANSWER]


@pytest.mark.unit
def test_run_agent_stream_delete_pending_failure_keeps_succeeded_run_and_blocks_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """成功写入 transcript 后清理 pending turn 失败时，run 仍应保持成功且恢复 gate 必须拒绝重放。"""

    from tests.application.conftest import StubHostExecutor, StubPendingTurnStore, StubRunRegistry, StubSessionRegistry

    class _RecordingSessionState:
        def record_reasoning_delta(self, chunk: str) -> None:
            del chunk

        def persist_turn(self, **_kwargs) -> None:
            return None

    class _DeleteFailingPendingTurnStore(StubPendingTurnStore):
        def delete_pending_turn(self, pending_turn_id: str) -> None:
            del pending_turn_id
            raise RuntimeError("delete failed")

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
            return PreparedAgentExecution(
                agent_input=AgentInput(
                    system_prompt=prepared_execution.agent_input.system_prompt,
                    messages=list(prepared_execution.agent_input.messages),
                    agent_create_args=prepared_execution.agent_input.agent_create_args,
                    session_state=_RecordingSessionState(),
                ),
                resume_snapshot=prepared_execution.resume_snapshot,
            )

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("该测试不应走恢复路径")

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})

    run_registry = StubRunRegistry()
    pending_turn_store = _DeleteFailingPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )

    async def _collect() -> None:
        async for _event in executor.run_agent_stream(execution_contract):
            pass

    asyncio.run(_collect())

    run = next(iter(run_registry._runs.values()))
    assert run.state.value == "succeeded"
    pending_turns = pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive")
    assert len(pending_turns) == 1
    host = Host(
        executor=StubHostExecutor(),
        session_registry=StubSessionRegistry(),
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,
    )

    async def _resume() -> None:
        async for _event in host.resume_pending_turn_stream(
            pending_turn_id=pending_turns[0].pending_turn_id,
            session_id="s1",
        ):
            pass

    with pytest.raises(ValueError, match="已成功完成"):
        asyncio.run(_resume())


@pytest.mark.unit
def test_run_agent_sync_returns_filtered_app_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """同步聚合路径应保留 final_answer 的 filtered 状态。"""

    from tests.application.conftest import StubRunRegistry

    class _StubScenePreparation:
        async def prepare(self, execution_contract: ExecutionContract, run_context: HostedRunContext) -> PreparedAgentExecution:
            del run_context
            return _build_prepared_execution(execution_contract=execution_contract)

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del run_context
            return AgentInput(
                system_prompt=prepared_turn.system_prompt,
                messages=list(prepared_turn.messages),
                agent_create_args=prepared_turn.agent_create_args,
            )

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            yield StreamEvent(
                EventType.FINAL_ANSWER,
                {"content": "partial", "degraded": True, "filtered": True, "finish_reason": "content_filter"},
                {},
            )

    executor = DefaultHostExecutor(
        run_registry=StubRunRegistry(),
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=False),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
    )

    result = asyncio.run(executor.run_agent_and_wait(execution_contract))

    assert result.content == "partial"
    assert result.degraded is True
    assert result.filtered is True


@pytest.mark.unit
def test_run_agent_and_wait_uses_app_event_enum_instead_of_value_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """同步聚合路径应直接比较 AppEventType，而不是依赖其字符串 value。"""

    from tests.application.conftest import StubRunRegistry

    async def _fake_run_agent_stream(_execution_contract: ExecutionContract):
        yield AppEvent(type=AppEventType.WARNING, payload="warn", meta={})
        yield AppEvent(type=AppEventType.ERROR, payload="err", meta={})
        yield AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload={"content": "done", "degraded": False, "filtered": False},
            meta={},
        )

    monkeypatch.setattr(AppEventType.WARNING, "_value_", "warn-renamed")
    monkeypatch.setattr(AppEventType.ERROR, "_value_", "err-renamed")
    monkeypatch.setattr(AppEventType.FINAL_ANSWER, "_value_", "final-renamed")

    executor = DefaultHostExecutor(run_registry=StubRunRegistry())
    monkeypatch.setattr(executor, "run_agent_stream", _fake_run_agent_stream)
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=False),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
    )

    result = asyncio.run(executor.run_agent_and_wait(execution_contract))

    assert result.content == "done"
    assert result.warnings == ["warn"]
    assert result.errors == ["err"]


@pytest.mark.unit
def test_run_agent_and_wait_raises_cancelled_error_on_cancelled_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """同步聚合路径收到 CANCELLED 终态时应抛出取消异常。"""

    from tests.application.conftest import StubRunRegistry

    async def _fake_run_agent_stream(_execution_contract: ExecutionContract):
        yield AppEvent(type=AppEventType.WARNING, payload="warn", meta={})
        yield AppEvent(
            type=AppEventType.CANCELLED,
            payload={"cancel_reason": RunCancelReason.TIMEOUT.value},
            meta={"run_id": "run-timeout"},
        )

    executor = DefaultHostExecutor(run_registry=StubRunRegistry())
    monkeypatch.setattr(executor, "run_agent_stream", _fake_run_agent_stream)
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s1", resumable=False),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
    )

    with pytest.raises(CancelledError, match="timeout"):
        asyncio.run(executor.run_agent_and_wait(execution_contract))


@pytest.mark.unit
def test_host_executor_helper_functions_cover_deadline_and_summary_edges() -> None:
    """验证 host executor helper 的剩余分支。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    token = executor_module.CancellationToken()

    with pytest.raises(ValueError, match="timeout_ms"):
        executor_module.RunDeadlineWatcher(run_registry, "run-1", token, 0)

    watcher = executor_module.RunDeadlineWatcher(run_registry, "run-1", token, None)
    watcher.start()
    watcher.stop()

    run = run_registry.register_run(session_id="s1", service_type="prompt", scene_name="interactive")
    timeout_token = executor_module.CancellationToken()
    timeout_watcher = executor_module.RunDeadlineWatcher(run_registry, run.run_id, timeout_token, 10)
    timeout_watcher._on_timeout()
    current_run = run_registry.get_run(run.run_id)
    assert current_run is not None
    assert timeout_token.is_cancelled() is True
    assert current_run.cancel_requested_reason == RunCancelReason.TIMEOUT

    prepared_turn = PreparedAgentTurnSnapshot(
        service_name="chat_turn",
        scene_name="interactive",
        metadata={"delivery_channel": "interactive"},
        business_concurrency_lane=None,
        timeout_ms=None,
        resumable=False,
        system_prompt="sys",
        messages=[{"role": "user", "content": "hello"}],
        agent_create_args=AgentCreateArgs(runner_type="openai", model_name="test-model"),
        selected_toolsets=(),
        execution_permissions=ExecutionPermissions(
            web=ExecutionWebPermissions(allow_private_network_url=False),
            doc=ExecutionDocPermissions(),
        ),
        toolset_configs=(),
        trace_settings=None,
        conversation_memory_settings=ConversationMemorySettings(),
    )
    run_spec = executor_module._build_run_spec_from_prepared_turn(prepared_turn)

    assert run_spec.session_id is None
    assert executor_module._extract_event_message({"message": "warn"}) == "warn"
    assert executor_module._extract_event_message("plain") == "plain"
    assert executor_module.DefaultHostExecutor._summarize_error(RuntimeError("abcdef"), 3) == "abc"
    assert executor_module.DefaultHostExecutor._summarize_error(RuntimeError("abcdef"), 0) == "a"
    assert run_spec.operation_name == "chat_turn"
    assert run_spec.scene_name == "interactive"
    long_summary = executor_module._summarize_tool_result({"ok": True, "value": {"body": "x" * 5000}})
    assert "<truncated" in long_summary
    assert len(long_summary) <= executor_module._MAX_TOOL_RESULT_SUMMARY_CHARS


@pytest.mark.unit
def test_run_deadline_watcher_start_is_idempotent_and_timeout_after_stop_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 deadline watcher 的 timer 状态切换保持幂等。"""

    from tests.application.conftest import StubRunRegistry

    class _FakeTimer:
        """测试用 timer。"""

        def __init__(self, interval: float, callback: object) -> None:
            self.interval = interval
            self.callback = callback
            self.daemon = False
            self.name = ""
            self.start_count = 0
            self.cancel_count = 0

        def start(self) -> None:
            """记录启动次数。"""

            self.start_count += 1

        def cancel(self) -> None:
            """记录取消次数。"""

            self.cancel_count += 1

    timers: list[_FakeTimer] = []

    def _build_fake_timer(interval: float, callback: object) -> _FakeTimer:
        """构建测试用假 timer。"""

        timer = _FakeTimer(interval, callback)
        timers.append(timer)
        return timer

    monkeypatch.setattr(executor_module.threading, "Timer", _build_fake_timer)

    run_registry = StubRunRegistry()
    token = executor_module.CancellationToken()
    watcher = executor_module.RunDeadlineWatcher(run_registry, "run-1", token, 10)

    watcher.start()
    watcher.start()
    watcher._on_timeout()
    watcher.stop()

    assert len(timers) == 1
    assert timers[0].start_count == 1
    assert timers[0].cancel_count == 0


@pytest.mark.unit
def test_run_operation_uses_run_timeout_as_concurrency_acquire_budget() -> None:
    """验证宿主执行器会把 run timeout 传给 permit 获取流程。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    governor = _StubGovernor()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        concurrency_governor=governor,  # type: ignore[arg-type]
    )
    spec = HostedRunSpec(
        operation_name="prompt",
        session_id="s1",
        business_concurrency_lane="sec_download",
        timeout_ms=1500,
    )

    def _operation(_context: HostedRunContext) -> int:
        """测试桩：直接返回。"""

        return 1

    result = executor.run_operation_sync(spec=spec, operation=_operation)

    assert result == 1
    assert governor.acquire_timeouts == [1.5]


@pytest.mark.unit
def test_run_operation_uses_unbounded_concurrency_acquire_policy() -> None:
    """显式无限等待策略应把 acquire timeout 透传为 ``None``。"""

    from tests.application.conftest import StubRunRegistry

    run_registry = StubRunRegistry()
    governor = _StubGovernor()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        concurrency_governor=governor,  # type: ignore[arg-type]
    )
    spec = HostedRunSpec(
        operation_name="write_pipeline",
        session_id="s1",
        business_concurrency_lane="sec_download",
        concurrency_acquire_policy=ConcurrencyAcquirePolicy.unbounded(),
    )

    def _operation(_context: HostedRunContext) -> int:
        """测试桩：直接返回。"""

        return 1

    result = executor.run_operation_sync(spec=spec, operation=_operation)

    assert result == 1
    assert governor.acquire_timeouts == [None]


@pytest.mark.unit
def test_run_operation_cancelled_during_concurrency_wait_returns_on_cancel() -> None:
    """等待 permit 阶段收到取消请求时，同步 run 应收敛为 CANCELLED 并返回取消退出码。"""

    from tests.application.conftest import StubRunRegistry

    wait_started = threading.Event()

    class _BlockingGovernor(_StubGovernor):
        """在 acquire_many 阶段阻塞，直到取消令牌被触发。"""

        def acquire_many(
            self,
            lanes: list[str],
            *,
            timeout: float | None = None,
            cancellation_token: executor_module.CancellationToken | None = None,
        ) -> list[_Permit]:
            del lanes, timeout
            if cancellation_token is None:
                raise AssertionError("测试前提失败：executor 未传入 cancellation_token")
            self.acquire_cancellation_tokens.append(cancellation_token)
            wait_started.set()
            cancellation_token.wait(5.0)
            cancellation_token.raise_if_cancelled()
            raise AssertionError("测试前提失败：等待未被取消")

    run_registry = StubRunRegistry()
    governor = _BlockingGovernor()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        concurrency_governor=governor,  # type: ignore[arg-type]
    )
    spec = HostedRunSpec(
        operation_name="write_pipeline",
        session_id="s1",
        business_concurrency_lane="sec_download",
        concurrency_acquire_policy=ConcurrencyAcquirePolicy.unbounded(),
    )
    result_holder: dict[str, int] = {}

    def _run_sync_operation() -> None:
        """在线程中执行同步 run，等待主线程触发取消。"""

        result_holder["result"] = executor.run_operation_sync(
            spec=spec,
            operation=lambda _context: pytest.fail("取消发生在 acquire 阶段，不应进入业务回调"),
            on_cancel=lambda: 130,
        )

    worker = threading.Thread(target=_run_sync_operation, name="test-host-cancel-during-acquire")
    worker.start()
    assert wait_started.wait(1.0) is True
    [current_run] = list(run_registry._runs.values())
    assert run_registry.request_cancel(
        current_run.run_id,
        cancel_reason=RunCancelReason.USER_CANCELLED,
    ) is True
    worker.join(timeout=3.0)

    assert worker.is_alive() is False
    assert result_holder["result"] == 130
    assert governor.acquire_cancellation_tokens[0] is not None
    final_run = run_registry.get_run(current_run.run_id)
    assert final_run is not None
    assert final_run.state == RunState.CANCELLED
    assert final_run.cancel_reason == RunCancelReason.USER_CANCELLED


@pytest.mark.unit
def test_host_executor_finish_cancel_and_pending_turn_reconcile_helpers() -> None:
    """验证取消收口与 pending turn reconcile 的辅助分支。"""

    from tests.application.conftest import StubPendingTurnStore, StubRunRegistry

    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
    )

    with pytest.raises(CancelledError, match="操作已被取消"):
        executor._finish_sync_cancelled_run(run_id="missing-run", on_cancel=None)

    finished = executor._finish_sync_cancelled_run(run_id="missing-run", on_cancel=lambda: 7)
    assert finished == 7

    record = pending_turn_store.upsert_pending_turn(
        session_id="s1",
        scene_name="interactive",
        user_text="问题",
        source_run_id="run-source",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json="{}",
    )
    executor._reconcile_pending_turn_after_terminal_run(
        pending_turn_id=record.pending_turn_id,
        run=None,
        resumable=True,
    )
    assert pending_turn_store.list_pending_turns(session_id="s1", scene_name="interactive") == []


@pytest.mark.unit
def test_hosted_run_spec_normalizes_execution_delivery_context() -> None:
    """HostedRunSpec 应只保留稳定交付上下文字段。"""

    spec = HostedRunSpec(
        operation_name="prompt",
        metadata=cast(
            ExecutionDeliveryContext,
            {
                "delivery_channel": " wechat ",
                "delivery_target": " user-1 ",
                "filtered": True,
                "unexpected": "ignored",
            },
        ),
    )

    assert spec.metadata == {
        "delivery_channel": "wechat",
        "delivery_target": "user-1",
        "filtered": True,
    }


@pytest.mark.unit
def test_resume_pending_turn_stream_keeps_lease_through_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resume 路径下 DefaultHostExecutor 不得 upsert 覆盖 Host 端持有的 RESUMING lease。

    用例构造真实 ``DefaultHostExecutor`` + ``StubPendingTurnStore``，预置一条
    PREPARED_BY_HOST 状态的 pending turn，并让 source run 处于 TIMEOUT 取消终态；
    调用 ``Host.resume_pending_turn_stream`` 后：

    - ``_register_accepted_pending_turn`` / ``_register_prepared_pending_turn`` 均不应被调用，
      即 executor 读取到的 `resumed_pending_turn_id` 非 None，不触发 upsert 覆盖；
    - pending turn 最终按"成功执行 → delete"路径被清理，lease 没有被迫回到 PREPARED_BY_HOST。
    """

    from tests.application.conftest import (
        StubPendingTurnStore,
        StubRunRegistry,
        StubSessionRegistry,
    )

    class _RecordingSessionState:
        def record_reasoning_delta(self, chunk: str) -> None:
            del chunk

        def persist_turn(self, **_kwargs) -> None:
            return None

    class _StubScenePreparation:
        async def prepare(
            self,
            execution_contract: ExecutionContract,
            run_context: HostedRunContext,
        ) -> PreparedAgentExecution:
            del run_context
            raise AssertionError("resume 路径不应走 prepare")

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del run_context
            return AgentInput(
                system_prompt=prepared_turn.system_prompt,
                messages=list(prepared_turn.messages),
                agent_create_args=prepared_turn.agent_create_args,
                session_state=_RecordingSessionState(),
            )

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})

    session_registry = StubSessionRegistry()
    session_registry.create_session(source=cast("object", "interactive"), session_id="s-resume-ok")  # type: ignore[arg-type]
    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()

    # 构造 prepared snapshot 及其真源 JSON
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s-resume-ok", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题-resume"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )
    prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
    prepared_turn = prepared_execution.resume_snapshot
    assert prepared_turn is not None
    prepared_turn_json = json.dumps(
        serialize_prepared_agent_turn_snapshot(prepared_turn),
        ensure_ascii=False,
        sort_keys=True,
    )

    # 登记并标记源 run 为 TIMEOUT 取消，使其满足 resume 的 source run 合法性校验
    source_run = run_registry.register_run(
        session_id="s-resume-ok",
        service_type="chat_turn",
        scene_name="interactive",
    )
    run_registry.start_run(source_run.run_id)
    run_registry.mark_cancelled(source_run.run_id, cancel_reason=RunCancelReason.TIMEOUT)

    seeded = pending_turn_store.seed_pending_turn(
        session_id="s-resume-ok",
        scene_name="interactive",
        user_text="问题-resume",
        source_run_id=source_run.run_id,
        resumable=True,
        resume_source_json=prepared_turn_json,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )

    executor = DefaultHostExecutor(
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())

    # 侦测 _register_*_pending_turn 是否被调用；resume 路径下严禁触发
    accepted_register_calls: list[str] = []
    prepared_register_calls: list[str] = []

    original_register_accepted = executor._register_accepted_pending_turn
    original_register_prepared = executor._register_prepared_pending_turn

    def _spy_register_accepted(**kwargs) -> str | None:  # type: ignore[no-untyped-def]
        accepted_register_calls.append(kwargs.get("run_id", ""))
        return original_register_accepted(**kwargs)

    def _spy_register_prepared(**kwargs) -> str | None:  # type: ignore[no-untyped-def]
        prepared_register_calls.append(kwargs.get("run_id", ""))
        return original_register_prepared(**kwargs)

    monkeypatch.setattr(executor, "_register_accepted_pending_turn", _spy_register_accepted)
    monkeypatch.setattr(executor, "_register_prepared_pending_turn", _spy_register_prepared)

    host = Host(
        executor=executor,
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
    )

    async def _collect() -> list[AppEvent]:
        events: list[AppEvent] = []
        async for event in host.resume_pending_turn_stream(
            pending_turn_id=seeded.pending_turn_id,
            session_id="s-resume-ok",
        ):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert accepted_register_calls == []
    assert prepared_register_calls == []
    # 执行成功后 pending turn 被 executor 正常清理
    assert pending_turn_store.list_pending_turns(session_id="s-resume-ok", scene_name="interactive") == []
    assert [event.type for event in events] == [AppEventType.FINAL_ANSWER]


@pytest.mark.unit
def test_resume_pending_turn_stream_second_resumer_rejected_without_deleting_record() -> None:
    """第二位并发 resumer 在发现 RESUMING lease 时应被 ``ValueError`` 拒绝，且不得删除记录。

    Finding 070 的老集成漏洞：在 acquire 之前按 state 解析真源，会把 RESUMING +
    pre_resume_state=ACCEPTED_BY_HOST 的记录按 prepared snapshot 路径反序列化，
    抛 "messages 必须是 JSON array"、被判定"永久损坏"后误删合法持有者的 lease。
    修复后 Host 必须在 acquire 之后再解析，第二位申请直接在 acquire 阶段被拒，
    不会进入反序列化分支，pending turn 记录保持存在。
    """

    from tests.application.conftest import (
        StubHostExecutor,
        StubPendingTurnStore,
        StubRunRegistry,
        StubSessionRegistry,
    )

    session_registry = StubSessionRegistry()
    session_registry.create_session(source=cast("object", "interactive"), session_id="s-resume-conflict")  # type: ignore[arg-type]
    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()

    # 源 run 置为 TIMEOUT 取消，满足 source run 合法性校验
    source_run = run_registry.register_run(
        session_id="s-resume-conflict",
        service_type="chat_turn",
        scene_name="interactive",
    )
    run_registry.start_run(source_run.run_id)
    run_registry.mark_cancelled(source_run.run_id, cancel_reason=RunCancelReason.TIMEOUT)

    # 构造 accepted 真源 JSON：有效 payload，保证即便被错误路由到 prepared 反序列化
    # 仍然能被检测出来——更重要的是验证"第二位根本不会进入解析"。
    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s-resume-conflict", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题-冲突"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )
    prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
    prepared_turn = prepared_execution.resume_snapshot
    assert prepared_turn is not None
    prepared_turn_json = json.dumps(
        serialize_prepared_agent_turn_snapshot(prepared_turn),
        ensure_ascii=False,
        sort_keys=True,
    )

    seeded = pending_turn_store.seed_pending_turn(
        session_id="s-resume-conflict",
        scene_name="interactive",
        user_text="问题-冲突",
        source_run_id=source_run.run_id,
        resumable=True,
        resume_source_json=prepared_turn_json,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )

    # 第一位 resumer：直接在仓储层 acquire，模拟已持有 RESUMING lease
    first_record = pending_turn_store.record_resume_attempt(
        seeded.pending_turn_id,
        max_attempts=5,
    )
    assert first_record.state is PendingConversationTurnState.RESUMING
    assert first_record.pre_resume_state is PendingConversationTurnState.PREPARED_BY_HOST

    # 第二位 resumer：走 Host 入口，应在 acquire 阶段被拒绝
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
    )

    async def _second_resume() -> None:
        async for _event in host.resume_pending_turn_stream(
            pending_turn_id=seeded.pending_turn_id,
            session_id="s-resume-conflict",
        ):
            pass

    with pytest.raises(ValueError, match="正被其他 resumer 持有"):
        asyncio.run(_second_resume())

    # 第二位请求不得触碰 lease 或删除记录；留给第一位安全收尾
    surviving = pending_turn_store.get_pending_turn_record(seeded.pending_turn_id)
    assert surviving is not None
    assert surviving.state is PendingConversationTurnState.RESUMING
    assert surviving.pre_resume_state is PendingConversationTurnState.PREPARED_BY_HOST
    # 第一位刚 acquire，attempt_count 应为 1；第二位被拒后不得再前进
    assert surviving.resume_attempt_count == 1


@pytest.mark.unit
def test_resume_pending_turn_stream_rebinds_source_run_id_and_blocks_reresume_when_delete_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """resume 成功执行后即便 delete_pending_turn 失败，后续 resumer 也必须被拒。

    修复 Finding 070 三次加固引入的回归：resume 路径为避免覆盖 RESUMING lease
    跳过了 ``_register_*_pending_turn``，但若不重绑 ``source_run_id`` 到当前
    resumed run，则"delete 瞬时失败"场景下旧 timeout-cancelled run 仍会被
    ``_validate_source_run_for_resume`` 认作"可 resume 的 cancelled run"，触发
    重复恢复窗口。本测试模拟 delete 失败 → 断言 pending turn 的 source_run_id
    已指向当前 resumed run；下次 resume 尝试因新 run 处于 SUCCEEDED 终态被拒。
    """

    from tests.application.conftest import (
        StubPendingTurnStore,
        StubRunRegistry,
        StubSessionRegistry,
    )

    class _RecordingSessionState:
        def record_reasoning_delta(self, chunk: str) -> None:
            del chunk

        def persist_turn(self, **_kwargs) -> None:
            return None

    class _DeleteFailingPendingTurnStore(StubPendingTurnStore):
        def __init__(self) -> None:
            super().__init__()
            self.delete_attempts = 0

        def delete_pending_turn(self, pending_turn_id: str) -> None:
            self.delete_attempts += 1
            # 只模拟首次 delete 瞬时失败
            if self.delete_attempts <= 1:
                raise RuntimeError("delete failed")
            super().delete_pending_turn(pending_turn_id)

    class _StubScenePreparation:
        async def prepare(
            self,
            execution_contract: ExecutionContract,
            run_context: HostedRunContext,
        ) -> PreparedAgentExecution:
            del run_context
            raise AssertionError("resume 路径不应走 prepare")

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del run_context
            return AgentInput(
                system_prompt=prepared_turn.system_prompt,
                messages=list(prepared_turn.messages),
                agent_create_args=prepared_turn.agent_create_args,
                session_state=_RecordingSessionState(),
            )

    class _FakeAgent:
        async def run_messages(self, messages, *, session_id, run_id, stream):
            del messages, session_id, run_id, stream
            yield StreamEvent(EventType.FINAL_ANSWER, {"content": "done", "degraded": False}, {})

    session_registry = StubSessionRegistry()
    session_registry.create_session(source=cast("object", "interactive"), session_id="s-rebind")  # type: ignore[arg-type]
    run_registry = StubRunRegistry()
    pending_turn_store = _DeleteFailingPendingTurnStore()

    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s-rebind", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题-rebind"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )
    prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
    prepared_turn = prepared_execution.resume_snapshot
    assert prepared_turn is not None
    prepared_turn_json = json.dumps(
        serialize_prepared_agent_turn_snapshot(prepared_turn),
        ensure_ascii=False,
        sort_keys=True,
    )

    source_run = run_registry.register_run(
        session_id="s-rebind",
        service_type="chat_turn",
        scene_name="interactive",
    )
    run_registry.start_run(source_run.run_id)
    run_registry.mark_cancelled(source_run.run_id, cancel_reason=RunCancelReason.TIMEOUT)

    seeded = pending_turn_store.seed_pending_turn(
        session_id="s-rebind",
        scene_name="interactive",
        user_text="问题-rebind",
        source_run_id=source_run.run_id,
        resumable=True,
        resume_source_json=prepared_turn_json,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )

    executor = DefaultHostExecutor(
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: _FakeAgent())

    host = Host(
        executor=executor,
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
    )

    async def _resume_once() -> None:
        async for _event in host.resume_pending_turn_stream(
            pending_turn_id=seeded.pending_turn_id,
            session_id="s-rebind",
        ):
            pass

    # 首次 resume：执行本体成功但 delete 瞬时失败；executor 会吞掉 delete 异常
    # 并把 run 按成功路径收口（顺带依赖后续 resume gate 拒绝重放作为兜底）。
    asyncio.run(_resume_once())
    assert pending_turn_store.delete_attempts >= 1

    # pending turn 仍存在（因首次 delete 失败），但 source_run_id 必须已被重绑到当前 resumed run
    surviving = pending_turn_store.get_pending_turn_record(seeded.pending_turn_id)
    assert surviving is not None
    assert surviving.source_run_id != source_run.run_id
    rebound_run = run_registry.get_run(surviving.source_run_id)
    assert rebound_run is not None
    # 当前 resumed run 已成功执行 → SUCCEEDED；后续 resume gate 应据此拒绝
    assert rebound_run.state == RunState.SUCCEEDED

    # 第二次 resume 尝试：_validate_source_run_for_resume 看到新 run 已 SUCCEEDED，
    # 按"V1 不支持补投递恢复"路径拒绝；老的 timeout-cancelled run 不再被认作 gate。
    with pytest.raises(ValueError, match="已成功完成"):
        asyncio.run(_resume_once())


@pytest.mark.unit
def test_resume_pending_turn_stream_rebind_failure_finishes_run_without_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rebind_source_run_id_for_resume 抛异常时当前 resumed run 必须进入 FAILED 终态。

    Finding 070 四次加固回归测试：此前 ``rebind_source_run_id_for_resume`` 放在 executor
    主体 try/finally 之外——若它因 ``SessionClosedError`` 等仓储异常抛出，则当前已经
    ``register_run`` + ``_start_run`` 创建的 run 不会进入 except/finally 收敛路径，
    ``_finish_run`` 不执行，permit / deadline watcher / cancellation bridge 全部泄漏，
    run 永久卡在 RUNNING。

    本用例通过 wrapper pending turn store 让 rebind 抛 ``SessionClosedError``，断言：
    - ``Host.resume_pending_turn_stream`` 把异常按 Host 层约定转成 ``ValueError``
      并回退 lease；
    - 新建的 resumed run 不再停留在 RUNNING——由 executor 的 try/except 收敛为 FAILED；
    - 不遗留活跃 run，``list_active_runs`` 为空。
    """

    from tests.application.conftest import (
        StubPendingTurnStore,
        StubRunRegistry,
        StubSessionRegistry,
    )

    class _RebindFailingPendingTurnStore(StubPendingTurnStore):
        """rebind 抛 SessionClosedError 的 wrapper，用于复现异常收敛缺口。"""

        def rebind_source_run_id_for_resume(
            self,
            pending_turn_id: str,
            *,
            new_source_run_id: str,
            lease_id: str,
        ) -> PendingConversationTurn:
            del pending_turn_id, new_source_run_id, lease_id
            raise SessionClosedError("session closed during rebind")

    class _StubScenePreparation:
        async def prepare(
            self,
            execution_contract: ExecutionContract,
            run_context: HostedRunContext,
        ) -> PreparedAgentExecution:
            del execution_contract, run_context
            raise AssertionError("rebind 失败路径不应走 prepare")

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("rebind 失败路径不应走 restore")

    session_registry = StubSessionRegistry()
    session_registry.create_session(source=cast("object", "interactive"), session_id="s-rebind-fail")  # type: ignore[arg-type]
    run_registry = StubRunRegistry()
    pending_turn_store = _RebindFailingPendingTurnStore()

    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s-rebind-fail", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题-rebind-fail"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )
    prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
    prepared_turn = prepared_execution.resume_snapshot
    assert prepared_turn is not None
    prepared_turn_json = json.dumps(
        serialize_prepared_agent_turn_snapshot(prepared_turn),
        ensure_ascii=False,
        sort_keys=True,
    )

    source_run = run_registry.register_run(
        session_id="s-rebind-fail",
        service_type="chat_turn",
        scene_name="interactive",
    )
    run_registry.start_run(source_run.run_id)
    run_registry.mark_cancelled(source_run.run_id, cancel_reason=RunCancelReason.TIMEOUT)

    seeded = pending_turn_store.seed_pending_turn(
        session_id="s-rebind-fail",
        scene_name="interactive",
        user_text="问题-rebind-fail",
        source_run_id=source_run.run_id,
        resumable=True,
        resume_source_json=prepared_turn_json,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )

    executor = DefaultHostExecutor(
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )

    host = Host(
        executor=executor,
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
    )

    async def _resume() -> None:
        async for _event in host.resume_pending_turn_stream(
            pending_turn_id=seeded.pending_turn_id,
            session_id="s-rebind-fail",
        ):
            pass

    # Host 必须把内部仓储屏障异常 (SessionWriteBlockedError 子类) 收敛为业务语义
    # ValueError，避免把 RuntimeError 子类泄漏到 Service / UI。
    with pytest.raises(ValueError, match="session 已不再接受写入"):
        asyncio.run(_resume())

    # 关键断言 1：不泄漏活跃 run —— rebind 抛异常后 executor 的 try/except/finally
    # 应把新创建的 resumed run 推进到终态（FAILED），permit / bridge / watcher 释放。
    assert run_registry.list_active_runs() == []

    # 关键断言 2：新 resumed run 必须处于失败终态，而非 RUNNING / CREATED。
    resumed_runs = [
        run for run in run_registry.list_runs(session_id="s-rebind-fail")
        if run.run_id != source_run.run_id
    ]
    assert len(resumed_runs) == 1
    assert resumed_runs[0].state == RunState.FAILED

    # 关键断言 3：pending turn lease 应被 Host 层回退至可 resume 状态，
    # 而非卡在 RESUMING（由 Host 外层的 release_resume_lease 保证）。
    surviving = pending_turn_store.get_pending_turn_record(seeded.pending_turn_id)
    assert surviving is not None
    assert surviving.state in {
        PendingConversationTurnState.PREPARED_BY_HOST,
        PendingConversationTurnState.ACCEPTED_BY_HOST,
    }
    del monkeypatch  # 本用例未使用 monkeypatch，保留签名与现有 fixture 约定一致


@pytest.mark.unit
def test_resume_pending_turn_stream_translates_session_closed_error_at_acquire() -> None:
    """acquire resume lease 阶段仓储抛 SessionClosedError 时 Host 必须转为 ValueError。

    Finding 070 五次加固的剩余缺口：``record_resume_attempt`` 自身在仓储层入口会跑
    ``ensure_session_active`` 屏障；若 session 恰在 acquire 瞬间被关闭，``SessionClosedError``
    会在 lease 尚未落地时抛出。此前 Host 只在 ``except Exception`` 收尾分支做了
    ``SessionClosedError -> ValueError`` 转译，acquire 阶段未被覆盖，RuntimeError 子类仍
    会泄漏给 Service / UI，与 docstring 声明的 ``KeyError | ValueError`` 契约不一致。

    本用例让 ``record_resume_attempt`` 直接抛 ``SessionClosedError``，断言：
    - ``Host.resume_pending_turn_stream`` 对外只抛 ``ValueError("...session 已关闭...")``；
    - 异常链 ``__cause__`` 仍指向原始 ``SessionClosedError``，便于诊断；
    - 失败发生在 acquire 阶段，executor 不应被触发——``list_active_runs`` 为空，
      pending turn 记录保持可 resume 态（未被推入 RESUMING）。
    """

    from tests.application.conftest import (
        StubHostExecutor,
        StubPendingTurnStore,
        StubRunRegistry,
        StubSessionRegistry,
    )

    class _AcquireBarrierPendingTurnStore(StubPendingTurnStore):
        """acquire 阶段仓储屏障抛 SessionClosedError 的 wrapper。"""

        def record_resume_attempt(
            self,
            pending_turn_id: str,
            *,
            max_attempts: int,
        ) -> PendingConversationTurn:
            del pending_turn_id, max_attempts
            raise SessionClosedError("s-acquire-barrier")

    session_registry = StubSessionRegistry()
    session_registry.create_session(source=cast("object", "interactive"), session_id="s-acquire-barrier")  # type: ignore[arg-type]
    run_registry = StubRunRegistry()
    pending_turn_store = _AcquireBarrierPendingTurnStore()

    source_run = run_registry.register_run(
        session_id="s-acquire-barrier",
        service_type="chat_turn",
        scene_name="interactive",
    )
    run_registry.start_run(source_run.run_id)
    run_registry.mark_cancelled(source_run.run_id, cancel_reason=RunCancelReason.TIMEOUT)

    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s-acquire-barrier", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题-acquire-barrier"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )
    prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
    prepared_turn = prepared_execution.resume_snapshot
    assert prepared_turn is not None
    prepared_turn_json = json.dumps(
        serialize_prepared_agent_turn_snapshot(prepared_turn),
        ensure_ascii=False,
        sort_keys=True,
    )
    seeded = pending_turn_store.seed_pending_turn(
        session_id="s-acquire-barrier",
        scene_name="interactive",
        user_text="问题-acquire-barrier",
        source_run_id=source_run.run_id,
        resumable=True,
        resume_source_json=prepared_turn_json,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )

    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
    )

    async def _resume() -> None:
        async for _event in host.resume_pending_turn_stream(
            pending_turn_id=seeded.pending_turn_id,
            session_id="s-acquire-barrier",
        ):
            pass

    with pytest.raises(ValueError, match="session 已不再接受写入") as exc_info:
        asyncio.run(_resume())
    # 保留诊断链路：原始仓储屏障异常应挂在 __cause__ 上。
    assert isinstance(exc_info.value.__cause__, SessionClosedError)

    # acquire 阶段失败，executor 不应被触发；无活跃 run、lease 未进 RESUMING。
    assert run_registry.list_active_runs() == []
    surviving = pending_turn_store.get_pending_turn_record(seeded.pending_turn_id)
    assert surviving is not None
    assert surviving.state is PendingConversationTurnState.PREPARED_BY_HOST


async def _collect_resume_events(
    *,
    host: Host,
    pending_turn_id: str,
    session_id: str,
) -> list[AppEvent]:
    """收集一次 resume_pending_turn_stream 的全部事件。"""

    events: list[AppEvent] = []
    async for event in host.resume_pending_turn_stream(
        pending_turn_id=pending_turn_id,
        session_id=session_id,
    ):
        events.append(event)
    return events


@pytest.mark.unit
def test_resume_pending_turn_stream_releases_resuming_lease_when_cancelled_during_acquire() -> None:
    """resume 若在等待 permit 时被取消，必须释放已持有的 RESUMING lease。"""

    from tests.application.conftest import (
        StubPendingTurnStore,
        StubRunRegistry,
        StubSessionRegistry,
    )

    class _BlockingGovernor(_StubGovernor):
        """在 acquire_many 阶段阻塞，直到取消令牌被触发。"""

        def __init__(self) -> None:
            super().__init__()
            self.wait_started = threading.Event()

        def acquire_many(
            self,
            lanes: list[str],
            *,
            timeout: float | None = None,
            cancellation_token: executor_module.CancellationToken | None = None,
        ) -> list[_Permit]:
            del lanes, timeout
            if cancellation_token is None:
                raise AssertionError("测试前提失败：executor 未传入 cancellation_token")
            self.acquire_cancellation_tokens.append(cancellation_token)
            self.wait_started.set()
            cancellation_token.wait(5.0)
            cancellation_token.raise_if_cancelled()
            raise AssertionError("测试前提失败：等待未被取消")

    class _StubScenePreparation:
        async def prepare(
            self,
            execution_contract: ExecutionContract,
            run_context: HostedRunContext,
        ) -> PreparedAgentExecution:
            del execution_contract, run_context
            raise AssertionError("resume 路径不应走 prepare")

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            raise AssertionError("取消发生在 acquire 阶段，不应进入 restore_prepared_execution")

    session_registry = StubSessionRegistry()
    session_registry.create_session(source=cast("object", "interactive"), session_id="s-resume-cancel")  # type: ignore[arg-type]
    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    governor = _BlockingGovernor()

    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s-resume-cancel", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题-resume-cancel"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )
    prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
    prepared_turn = prepared_execution.resume_snapshot
    assert prepared_turn is not None
    prepared_turn_json = json.dumps(
        serialize_prepared_agent_turn_snapshot(prepared_turn),
        ensure_ascii=False,
        sort_keys=True,
    )

    source_run = run_registry.register_run(
        session_id="s-resume-cancel",
        service_type="chat_turn",
        scene_name="interactive",
    )
    run_registry.start_run(source_run.run_id)
    run_registry.mark_cancelled(source_run.run_id, cancel_reason=RunCancelReason.TIMEOUT)

    seeded = pending_turn_store.seed_pending_turn(
        session_id="s-resume-cancel",
        scene_name="interactive",
        user_text="问题-resume-cancel",
        source_run_id=source_run.run_id,
        resumable=True,
        resume_source_json=prepared_turn_json,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )

    executor = DefaultHostExecutor(
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        concurrency_governor=governor,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    host = Host(
        executor=executor,
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
    )

    event_holder: dict[str, list[AppEvent]] = {}
    error_holder: dict[str, BaseException] = {}

    def _run_resume_in_thread() -> None:
        """在线程中跑 resume 流，主线程负责注入取消请求。"""

        try:
            event_holder["events"] = asyncio.run(
                _collect_resume_events(
                    host=host,
                    pending_turn_id=seeded.pending_turn_id,
                    session_id="s-resume-cancel",
                )
            )
        except BaseException as exc:  # noqa: BLE001
            error_holder["error"] = exc

    worker = threading.Thread(
        target=_run_resume_in_thread,
        name="test-resume-cancel-during-acquire",
    )
    worker.start()
    assert governor.wait_started.wait(1.0) is True
    resumed_runs = [
        run
        for run in run_registry.list_runs(session_id="s-resume-cancel")
        if run.run_id != source_run.run_id
    ]
    assert len(resumed_runs) == 1
    assert run_registry.request_cancel(
        resumed_runs[0].run_id,
        cancel_reason=RunCancelReason.USER_CANCELLED,
    ) is True
    worker.join(timeout=3.0)

    assert worker.is_alive() is False
    assert "error" not in error_holder
    events = event_holder["events"]

    assert [event.type for event in events] == [AppEventType.CANCELLED]
    surviving = pending_turn_store.get_pending_turn_record(seeded.pending_turn_id)
    assert surviving is not None
    assert surviving.state is PendingConversationTurnState.PREPARED_BY_HOST
    assert surviving.pre_resume_state is None


@pytest.mark.unit
def test_prepared_turn_snapshot_rejects_missing_concurrency_policy() -> None:
    """prepared snapshot 缺失并发等待策略时必须显式失败。"""

    prepared_turn = PreparedAgentTurnSnapshot(
        service_name="chat_turn",
        scene_name="interactive",
        metadata={"delivery_channel": "interactive"},
        business_concurrency_lane="write_chapter",
        concurrency_acquire_policy=ConcurrencyAcquirePolicy.unbounded(),
        timeout_ms=None,
        resumable=True,
        system_prompt="sys",
        messages=[{"role": "user", "content": "hi"}],
        agent_create_args=AgentCreateArgs(runner_type="openai", model_name="test-model"),
        selected_toolsets=(),
        execution_permissions=ExecutionPermissions(),
        toolset_configs=(),
        trace_settings=None,
        conversation_memory_settings=ConversationMemorySettings(),
    )
    payload = serialize_prepared_agent_turn_snapshot(prepared_turn)
    payload.pop("concurrency_acquire_policy", None)

    with pytest.raises(ValueError, match="concurrency_acquire_policy"):
        deserialize_prepared_agent_turn_snapshot(payload)


@pytest.mark.unit
def test_accepted_turn_snapshot_rejects_missing_concurrency_policy() -> None:
    """accepted snapshot 的 host_policy 缺失并发等待策略时必须显式失败。"""

    snapshot = AcceptedAgentTurnSnapshot(
        service_name="write_pipeline",
        scene_name="write",
        host_policy=ExecutionHostPolicy(
            session_key="session-1",
            business_concurrency_lane="write_chapter",
            concurrency_acquire_policy=ConcurrencyAcquirePolicy.unbounded(),
            timeout_ms=None,
            resumable=True,
        ),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="请继续"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(),
        metadata={"delivery_channel": "interactive"},
    )
    payload = serialize_accepted_agent_turn_snapshot(snapshot)
    host_policy_payload = cast(dict[str, object], payload["host_policy"])
    host_policy_payload.pop("concurrency_acquire_policy", None)

    with pytest.raises(ValueError, match="host_policy.concurrency_acquire_policy"):
        deserialize_accepted_agent_turn_snapshot(payload)


# ---------------------------------------------------------------------------
# Review 缺口回归：CLEARING / CLEARING_FAILED 屏障下 executor 必须吸收抛错，
# 不能升级为未预期失败。
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_accepted_pending_turn_absorbs_clearing_error() -> None:
    """``CLEARING`` 屏障期间 ``_register_accepted_pending_turn`` 抛 ``SessionClearingError``
    必须被吸收为 ``None`` 返回（与 ``SessionClosedError`` 同侧降级）。"""

    from dayu.host.protocols import SessionClearingError

    class _ClearingPendingTurnStore:
        """无条件抛 SessionClearingError 的最小 stub。"""

        def upsert_pending_turn(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            raise SessionClearingError("s-x")

    executor = DefaultHostExecutor(
        run_registry=None,  # type: ignore[arg-type]
        pending_turn_store=_ClearingPendingTurnStore(),  # type: ignore[arg-type]
    )
    contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s-x", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="hi"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(),
        metadata={"delivery_channel": "interactive"},
    )
    result = executor._register_accepted_pending_turn(  # type: ignore[attr-defined]
        execution_contract=contract,
        run_id="run_test",
    )
    assert result is None  # 异常被吸收，返回 None


@pytest.mark.unit
def test_register_accepted_pending_turn_absorbs_clearing_failed_error() -> None:
    """``CLEARING_FAILED`` 持久锁定下迟到 accepted 登记同样降级为 no-op。"""

    from dayu.host.protocols import SessionClearingFailedError

    class _ClearingFailedPendingTurnStore:
        def upsert_pending_turn(self, **kwargs):  # type: ignore[no-untyped-def]
            del kwargs
            raise SessionClearingFailedError("s-y")

    executor = DefaultHostExecutor(
        run_registry=None,  # type: ignore[arg-type]
        pending_turn_store=_ClearingFailedPendingTurnStore(),  # type: ignore[arg-type]
    )
    contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s-y", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="hi"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(),
        metadata={"delivery_channel": "interactive"},
    )
    result = executor._register_accepted_pending_turn(  # type: ignore[attr-defined]
        execution_contract=contract,
        run_id="run_test",
    )
    assert result is None


# ---------------------------------------------------------------------------
# PR-2 review #4 回归：执行期间被 cleanup_stale_pending_turns 抢占时，
# 失败补写抛出的 LeaseExpiredError 不得覆盖原始执行异常。
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resume_pending_turn_stream_lease_expired_during_failure_record_does_not_mask_original_exc() -> None:
    """executor 抛业务异常 + 并发 cleanup 抢占 lease 场景下原始异常必须穿透到上层。

    场景：``Host.resume_pending_turn_stream`` 进入 try/except 后，executor 抛出
    业务异常 ``RuntimeError("executor 内部失败")``；此时若 ``cleanup_stale_pending_turns``
    在运行时（``host_admin_service`` / ``interactive_ui`` 都可主动触发）已把记录
    回退至非 RESUMING，``record_resume_failure`` 会抛 ``LeaseExpiredError``。Host 必须
    捕获并降级为告警日志，让原始 ``RuntimeError`` 继续向上抛，否则上层看到的会
    是 ``LeaseExpiredError`` 这种内部 fence token 异常，原因诊断被掩盖。
    """

    from dayu.host.lease import LeaseExpiredError
    from tests.application.conftest import (
        StubPendingTurnStore,
        StubRunRegistry,
        StubSessionRegistry,
    )

    class _LeaseExpiredOnFailurePendingTurnStore(StubPendingTurnStore):
        """``record_resume_failure`` 强制抛 ``LeaseExpiredError``，模拟并发 cleanup。"""

        def record_resume_failure(
            self,
            pending_turn_id: str,
            *,
            error_message: str,
            lease_id: str,
        ) -> PendingConversationTurn:
            del pending_turn_id, error_message
            raise LeaseExpiredError(
                "lease 已被 cleanup 抢占",
                record_id="pt-mocked",
                lease_id=lease_id,
            )

    class _StubScenePreparation:
        async def prepare(
            self,
            execution_contract: ExecutionContract,
            run_context: HostedRunContext,
        ) -> PreparedAgentExecution:
            del execution_contract, run_context
            raise AssertionError("prepared snapshot 路径不应走 prepare")

        async def restore_prepared_execution(
            self,
            prepared_turn: PreparedAgentTurnSnapshot,
            run_context: HostedRunContext,
        ) -> AgentInput:
            del prepared_turn, run_context
            # executor 在 restore 阶段抛业务异常，模拟"恢复执行失败"。
            raise RuntimeError("executor 内部失败")

    session_registry = StubSessionRegistry()
    session_registry.create_session(source=cast("object", "interactive"), session_id="s-lease-overrides")  # type: ignore[arg-type]
    run_registry = StubRunRegistry()
    pending_turn_store = _LeaseExpiredOnFailurePendingTurnStore()

    execution_contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key="s-lease-overrides", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题-lease-overrides"),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="resume-model", max_iterations=6),
        metadata={"delivery_channel": "interactive"},
    )
    prepared_execution = _build_prepared_execution(execution_contract=execution_contract)
    prepared_turn = prepared_execution.resume_snapshot
    assert prepared_turn is not None
    prepared_turn_json = json.dumps(
        serialize_prepared_agent_turn_snapshot(prepared_turn),
        ensure_ascii=False,
        sort_keys=True,
    )

    source_run = run_registry.register_run(
        session_id="s-lease-overrides",
        service_type="chat_turn",
        scene_name="interactive",
    )
    run_registry.start_run(source_run.run_id)
    run_registry.mark_cancelled(source_run.run_id, cancel_reason=RunCancelReason.TIMEOUT)

    seeded = pending_turn_store.seed_pending_turn(
        session_id="s-lease-overrides",
        scene_name="interactive",
        user_text="问题-lease-overrides",
        source_run_id=source_run.run_id,
        resumable=True,
        resume_source_json=prepared_turn_json,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )

    executor = DefaultHostExecutor(
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    host = Host(
        executor=executor,
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
    )

    async def _resume() -> None:
        async for _event in host.resume_pending_turn_stream(
            pending_turn_id=seeded.pending_turn_id,
            session_id="s-lease-overrides",
        ):
            pass

    # 关键断言：上抛的必须是 executor 抛出的原始 RuntimeError，
    # 而不是 record_resume_failure 抛出的 LeaseExpiredError。
    with pytest.raises(RuntimeError, match="executor 内部失败") as exc_info:
        asyncio.run(_resume())
    assert not isinstance(exc_info.value, LeaseExpiredError)

