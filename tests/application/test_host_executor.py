"""DefaultHostExecutor 测试。"""

from __future__ import annotations

import asyncio
import json
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
from dayu.contracts.run import RunCancelReason
from dayu.execution.options import ConversationMemorySettings, ExecutionOptions
from dayu.host.host import Host
from dayu.host.conversation_store import ConversationTranscript
from dayu.host.host_execution import HostedRunContext, HostedRunSpec
from dayu.host.executor import DefaultHostExecutor
from dayu.host.prepared_turn import PreparedAgentTurnSnapshot, PreparedConversationSessionSnapshot
from dayu.host.scene_preparer import PreparedAgentExecution
from dayu.contracts.events import AppEvent, AppEventType
from dayu.contracts.cancellation import CancelledError
from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.engine.events import EventType, StreamEvent
from dayu.host.host_store import HostStore
from dayu.host.run_registry import SQLiteRunRegistry
from dayu.host.pending_turn_store import PendingConversationTurnState
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
        self.released: list[str] = []

    def acquire(self, lane: str, *, timeout: float | None = None) -> _Permit:
        self.acquired.append(lane)
        self.acquire_timeouts.append(timeout)
        return _Permit(permit_id=f"permit-{lane}", lane=lane)

    def acquire_many(
        self, lanes: list[str], *, timeout: float | None = None
    ) -> list[_Permit]:
        """一次性拿齐多 lane 的测试实现：转发给 acquire 逐个累积。"""

        return [self.acquire(lane_name, timeout=timeout) for lane_name in lanes]

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
