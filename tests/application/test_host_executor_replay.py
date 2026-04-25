"""Host replay 能力测试：``run_agent_and_wait_replayable`` / ``replay_agent_and_wait``。

覆盖目标：
1. ``run_agent_and_wait_replayable`` 返回 ``(AppResult, ReplayHandle)``，且把
   完整对话历史登记到 stash。
2. ``replay_agent_and_wait`` 在 handle 上正确拼接历史、复用同一 AsyncAgent，
   并把 ``replay_disable_tools`` 透传到 engine。
3. session 关闭后 handle 失效。
4. 跨 session 使用 handle 抛 RuntimeError。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedModelSpec,
    AgentCreateArgs,
    AgentInput,
    ExecutionContract,
    ExecutionHostPolicy,
    ExecutionMessageInputs,
    ReplayHandle,
    ScenePreparationSpec,
)
from dayu.contracts.agent_types import AgentMessage
from dayu.engine.events import EventType, StreamEvent
from dayu.execution.options import ExecutionOptions
from dayu.host.executor import DefaultHostExecutor
from dayu.host.host_execution import HostedRunContext
from dayu.host.host_store import HostStore
from dayu.host.run_registry import SQLiteRunRegistry
from dayu.host.scene_preparer import PreparedAgentExecution


def _minimal_accepted_execution_spec() -> AcceptedExecutionSpec:
    return AcceptedExecutionSpec(model=AcceptedModelSpec(model_name="test-model"))


def _build_contract(
    *,
    user_message: str = "你好",
    session_key: str | None = "s1",
    replay_from: ReplayHandle | None = None,
    replay_disable_tools: bool = False,
) -> ExecutionContract:
    return ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(session_key=session_key, resumable=False),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(
            user_message=user_message,
            replay_from=replay_from,
            replay_disable_tools=replay_disable_tools,
        ),
        accepted_execution_spec=_minimal_accepted_execution_spec(),
        execution_options=ExecutionOptions(model_name="test-model", max_iterations=4),
    )


class _StubScenePreparation:
    """最小 scene preparation 桩：直接返回 AgentInput，不走真实 toolset。"""

    async def prepare(
        self,
        execution_contract: ExecutionContract,
        run_context: HostedRunContext,
    ) -> PreparedAgentExecution:
        del run_context
        messages: list[AgentMessage] = [
            {"role": "user", "content": str(execution_contract.message_inputs.user_message or "")}
        ]
        agent_input = AgentInput(
            system_prompt="sys",
            messages=messages,
            agent_create_args=AgentCreateArgs(runner_type="openai", model_name="test-model"),
        )
        return PreparedAgentExecution(agent_input=agent_input, resume_snapshot=None)

    async def restore_prepared_execution(self, prepared_turn, run_context):  # pragma: no cover - 不走 resume
        del prepared_turn, run_context
        raise NotImplementedError


class _FakeAgent:
    """记录历史并在 run_messages 末尾追加 assistant 消息的 Agent 桩。"""

    def __init__(self, *, final_text: str = "answer-1") -> None:
        self.run_calls: list[dict[str, Any]] = []
        self._final_text = final_text

    async def run_messages(
        self,
        messages,
        *,
        session_id,
        run_id,
        stream,
        disable_tools: bool = False,
    ):
        self.run_calls.append(
            {
                "messages_snapshot": list(messages),
                "session_id": session_id,
                "run_id": run_id,
                "stream": stream,
                "disable_tools": disable_tools,
            }
        )
        # 模拟 engine 行为：原地追加 assistant 消息。
        messages.append({"role": "assistant", "content": self._final_text})
        yield StreamEvent(
            EventType.FINAL_ANSWER,
            {"content": self._final_text, "degraded": False, "filtered": False},
            {},
        )


def _make_executor(monkeypatch: pytest.MonkeyPatch, agent: _FakeAgent) -> DefaultHostExecutor:
    from tests.application.conftest import StubRunRegistry

    executor = DefaultHostExecutor(
        run_registry=StubRunRegistry(),
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: agent)
    return executor


@pytest.mark.unit
def test_run_agent_and_wait_replayable_returns_handle_and_stashes_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_agent_and_wait_replayable`` 应在 stash 留下完整历史与 agent 引用。"""

    agent = _FakeAgent(final_text="第一次")
    executor = _make_executor(monkeypatch, agent)
    contract = _build_contract(user_message="问题1")

    result, handle = asyncio.run(executor.run_agent_and_wait_replayable(contract))

    assert result.content == "第一次"
    assert isinstance(handle, ReplayHandle)
    assert handle.handle_id in executor._replay_stash
    state = executor._replay_stash[handle.handle_id]
    assert state.session_id == "s1"
    # 第一次执行后，历史里应有 user + assistant 两条。
    roles = [msg.get("role") for msg in state.messages]
    assert roles == ["user", "assistant"]
    # stash 不再持有 AsyncAgent 实例本身（避免锁死第一次的 cancellation_token），
    # 改为持有原 AgentInput 以便 replay 时重建绑定新 token 的 AsyncAgent。
    assert state.agent_input is not None
    assert state.agent_input.agent_create_args.model_name == "test-model"


@pytest.mark.unit
def test_replay_agent_and_wait_appends_user_message_and_passes_disable_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """replay 路径应在原历史末尾追加 user 消息，并透传 disable_tools。"""

    agent = _FakeAgent(final_text="第一次")
    executor = _make_executor(monkeypatch, agent)

    initial = _build_contract(user_message="问题1")
    _, handle = asyncio.run(executor.run_agent_and_wait_replayable(initial))

    # 切换 final_text 模拟第二次返回
    agent._final_text = "第二次"
    replay_contract = _build_contract(
        user_message="再来一次",
        replay_from=handle,
        replay_disable_tools=True,
    )
    result, new_handle = asyncio.run(
        executor.replay_agent_and_wait(handle, replay_contract)
    )

    assert result.content == "第二次"
    # replay 调用 agent.run_messages 时，messages_snapshot 末尾必须是新的 user。
    second_call = agent.run_calls[-1]
    snapshot = second_call["messages_snapshot"]
    assert snapshot[-1].get("role") == "user"
    assert snapshot[-1].get("content") == "再来一次"
    # 历史应当包含上一次的 user / assistant + 本次 user，共三条。
    assert [m.get("role") for m in snapshot] == ["user", "assistant", "user"]
    # disable_tools 必须透传。
    assert second_call["disable_tools"] is True
    # 旧 handle 已被消费。
    assert handle.handle_id not in executor._replay_stash
    # 新 handle 已登记，可继续 replay。
    assert new_handle.handle_id in executor._replay_stash


@pytest.mark.unit
def test_replay_agent_and_wait_rejects_invalid_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """无效 handle 应抛 RuntimeError。"""

    agent = _FakeAgent()
    executor = _make_executor(monkeypatch, agent)
    fake_handle = ReplayHandle(handle_id="replay_does_not_exist")
    contract = _build_contract(user_message="x", replay_from=fake_handle)

    with pytest.raises(RuntimeError, match="无效的 replay handle"):
        asyncio.run(executor.replay_agent_and_wait(fake_handle, contract))


@pytest.mark.unit
def test_replay_agent_and_wait_rejects_cross_session_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """handle 与目标 session 不匹配时抛 RuntimeError。"""

    agent = _FakeAgent()
    executor = _make_executor(monkeypatch, agent)
    initial = _build_contract(user_message="问题", session_key="session-A")
    _, handle = asyncio.run(executor.run_agent_and_wait_replayable(initial))

    cross_contract = _build_contract(
        user_message="再来", session_key="session-B", replay_from=handle
    )
    with pytest.raises(RuntimeError, match="replay handle 与当前 session 不匹配"):
        asyncio.run(executor.replay_agent_and_wait(handle, cross_contract))


@pytest.mark.unit
def test_discard_replay_state_for_session_clears_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``discard_replay_state_for_session`` 应清掉对应 session 的所有句柄。"""

    agent = _FakeAgent()
    executor = _make_executor(monkeypatch, agent)

    contract_a = _build_contract(user_message="问题", session_key="session-A")
    _, handle_a = asyncio.run(executor.run_agent_and_wait_replayable(contract_a))
    contract_b = _build_contract(user_message="问题", session_key="session-B")
    _, handle_b = asyncio.run(executor.run_agent_and_wait_replayable(contract_b))

    executor.discard_replay_state_for_session("session-A")

    assert handle_a.handle_id not in executor._replay_stash
    assert handle_b.handle_id in executor._replay_stash

    # 对已清理 session 的 handle 再 replay 必须失败。
    replay_contract = _build_contract(
        user_message="再来", session_key="session-A", replay_from=handle_a
    )
    with pytest.raises(RuntimeError, match="无效的 replay handle"):
        asyncio.run(executor.replay_agent_and_wait(handle_a, replay_contract))


@pytest.mark.unit
def test_replay_agent_and_wait_requires_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """replay 必须在 message_inputs.user_message 提供文本，否则抛 RuntimeError。"""

    agent = _FakeAgent()
    executor = _make_executor(monkeypatch, agent)
    initial = _build_contract(user_message="问题")
    _, handle = asyncio.run(executor.run_agent_and_wait_replayable(initial))

    bad_contract = _build_contract(user_message="   ", replay_from=handle)
    with pytest.raises(RuntimeError, match="必须在 message_inputs.user_message"):
        asyncio.run(executor.replay_agent_and_wait(handle, bad_contract))


@pytest.mark.unit
def test_replay_agent_and_wait_drives_full_run_lifecycle_on_real_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """回归：replay 必须像普通 agent run 一样走 ``CREATED -> RUNNING -> SUCCEEDED``。

    若 replay 路径绕过 ``_start_run``（不调 ``start_run``），SQLiteRunRegistry
    会在 ``complete_run`` 时拒绝从 CREATED 直接转到 SUCCEEDED，导致成功路径
    在收口阶段抛状态转换错误。同时这里也保证取消桥 / deadline watcher / 并发
    lane 都通过 ``_finish_run`` 正确释放。
    """

    host_store = HostStore(tmp_path / "host.db")
    host_store.initialize_schema()
    run_registry = SQLiteRunRegistry(host_store)
    agent = _FakeAgent(final_text="第一次")
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", lambda **_: agent)

    initial = _build_contract(user_message="问题1")
    _, handle = asyncio.run(executor.run_agent_and_wait_replayable(initial))

    agent._final_text = "第二次"
    replay_contract = _build_contract(
        user_message="再来", replay_from=handle, replay_disable_tools=True
    )
    result, new_handle = asyncio.run(
        executor.replay_agent_and_wait(handle, replay_contract)
    )

    assert result.content == "第二次"
    assert isinstance(new_handle, ReplayHandle)
    runs = list(run_registry.list_runs())
    # 至少有 2 个 run（初次 + replay），且全部走到 SUCCEEDED 终态。
    assert len(runs) >= 2
    states = {run.state.value for run in runs}
    assert states == {"succeeded"}


@pytest.mark.unit
def test_replay_agent_and_wait_rebuilds_async_agent_with_new_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """replay 不能复用第一次的 AsyncAgent，必须基于 AgentInput 重建并绑定新 token。

    第一次的 ``AsyncAgent`` 绑死在第一次 run 的 ``CancellationToken``，复用
    会让 replay 的取消桥 / deadline watcher 形同虚设。这里通过 ``build_async_agent``
    的调用次数和 ``cancellation_token`` 参数来回归该行为：每次 run 都必须
    至少触发一次 ``build_async_agent`` 且 token 是各自 run 的新 token。
    """

    initial_agent = _FakeAgent(final_text="第一次")
    replay_agent = _FakeAgent(final_text="第二次")
    builds: list[dict[str, Any]] = []

    def _record_build(**kwargs: Any) -> _FakeAgent:
        builds.append(kwargs)
        return initial_agent if len(builds) == 1 else replay_agent

    from tests.application.conftest import StubRunRegistry

    executor = DefaultHostExecutor(
        run_registry=StubRunRegistry(),
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", _record_build)

    initial = _build_contract(user_message="问题1")
    _, handle = asyncio.run(executor.run_agent_and_wait_replayable(initial))

    replay_contract = _build_contract(user_message="再来", replay_from=handle)
    result, _ = asyncio.run(executor.replay_agent_and_wait(handle, replay_contract))

    assert result.content == "第二次"
    # 必须发生第二次 build_async_agent；replay agent 是新构造的，不是第一次复用。
    assert len(builds) == 2
    second_token = builds[1].get("cancellation_token")
    # replay 必须把本次 _start_run 颁发的新 token 透给底层 runner，否则
    # 取消桥 / deadline watcher 都无法穿透到模型与工具调用。
    assert second_token is not None, "replay 必须用 _start_run 颁发的新 cancellation_token 构造 AsyncAgent"
    first_token = builds[0].get("cancellation_token")
    if first_token is not None:
        assert first_token is not second_token, "replay token 必须独立于第一次"
    # replay 实际跑的应该是第二个 fake agent，不是第一个。
    assert replay_agent.run_calls and not initial_agent.run_calls[1:]


@pytest.mark.unit
def test_replay_agent_and_wait_publishes_cancelled_event_when_runner_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """runner 在 replay 中直接抛 CancelledError 时，必须发布 CANCELLED AppEvent。

    若只把 run 标到 CANCELLED 而不发事件，订阅 replay run 的 UI / Service
    会丢失终态事件，与 run_registry 出现状态漂移。
    """

    from dayu.contracts.cancellation import CancelledError as DayuCancelledError
    from dayu.contracts.events import AppEvent, AppEventType
    from tests.application.conftest import StubRunRegistry

    class _RecordingEventBus:
        def __init__(self) -> None:
            self.events: list[tuple[str, AppEvent]] = []

        def publish(self, run_id: str, event: AppEvent) -> None:
            self.events.append((run_id, event))

    initial_agent = _FakeAgent(final_text="第一次")

    class _RaiseCancelledAgent:
        async def run_messages(
            self,
            messages,
            *,
            session_id,
            run_id,
            stream,
            disable_tools: bool = False,
        ):
            del messages, session_id, run_id, stream, disable_tools
            if False:  # 让函数保持 async generator 形态。
                yield None
            raise DayuCancelledError("runner 中途取消")

    builds = [initial_agent, _RaiseCancelledAgent()]
    cursor: list[int] = []

    def _record_build(**_kwargs: Any):
        idx = len(cursor)
        cursor.append(idx)
        return builds[idx]

    monkeypatch.setattr("dayu.host.executor.build_async_agent", _record_build)

    bus = _RecordingEventBus()
    executor = DefaultHostExecutor(
        run_registry=StubRunRegistry(),
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
        event_bus=bus,  # type: ignore[arg-type]
    )

    initial = _build_contract(user_message="问题1")
    _, handle = asyncio.run(executor.run_agent_and_wait_replayable(initial))

    replay_contract = _build_contract(user_message="再来", replay_from=handle)
    with pytest.raises(DayuCancelledError):
        asyncio.run(executor.replay_agent_and_wait(handle, replay_contract))

    cancelled = [e for _, e in bus.events if e.type == AppEventType.CANCELLED]
    assert cancelled, "replay 取消时必须发布 CANCELLED AppEvent"


@pytest.mark.unit
def test_replay_agent_and_wait_converts_cancellation_origin_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """取消请求已落库后 runner 抛普通业务异常时，必须收敛成统一 CancelledError。

    若直接把业务异常往上 raise，调用方看到的就不是与 run / event 终态一致的
    CancelledError，而是与 CANCELLED AppEvent 矛盾的业务异常。
    """

    from dayu.contracts.cancellation import CancelledError as DayuCancelledError
    from tests.application.conftest import StubRunRegistry

    initial_agent = _FakeAgent(final_text="第一次")
    registry = StubRunRegistry()

    class _CancelThenRaiseAgent:
        async def run_messages(
            self,
            messages,
            *,
            session_id,
            run_id,
            stream,
            disable_tools: bool = False,
        ):
            del messages, session_id, stream, disable_tools
            if False:  # async generator 形态。
                yield None
            # 模拟 deadline / 用户取消已落库后，runner 抛业务异常作为取消余波。
            registry.request_cancel(run_id)
            raise RuntimeError("HTTP timeout from upstream while cancelling")

    builds = [initial_agent, _CancelThenRaiseAgent()]
    cursor: list[int] = []

    def _record_build(**_kwargs: Any):
        idx = len(cursor)
        cursor.append(idx)
        return builds[idx]

    monkeypatch.setattr("dayu.host.executor.build_async_agent", _record_build)

    executor = DefaultHostExecutor(
        run_registry=registry,
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )

    initial = _build_contract(user_message="问题1")
    _, handle = asyncio.run(executor.run_agent_and_wait_replayable(initial))

    replay_contract = _build_contract(user_message="再来", replay_from=handle)
    with pytest.raises(DayuCancelledError):
        asyncio.run(executor.replay_agent_and_wait(handle, replay_contract))


@pytest.mark.unit
def test_replay_agent_and_wait_raises_cancelled_when_cancel_requested_mid_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """replay 期间已被请求取消但 stream 自然结束时，必须抛 CancelledError。

    若只把 run 标记为 CANCELLED 却仍返回 AppResult，上层 service 会以为是
    成功结果继续处理脏数据，掩盖取消语义。
    """

    from dayu.contracts.cancellation import CancelledError as DayuCancelledError
    from tests.application.conftest import StubRunRegistry

    initial_agent = _FakeAgent(final_text="第一次")
    registry = StubRunRegistry()
    executor = DefaultHostExecutor(
        run_registry=registry,
        scene_preparation=_StubScenePreparation(),  # type: ignore[arg-type]
    )

    class _CancelMidStreamAgent:
        """在 stream 第一帧后请求取消，模拟 deadline / 用户取消落点。"""

        def __init__(self) -> None:
            self.run_calls: list[dict[str, Any]] = []

        async def run_messages(
            self,
            messages,
            *,
            session_id,
            run_id,
            stream,
            disable_tools: bool = False,
        ):
            self.run_calls.append({"run_id": run_id})
            # 模拟下游收到 cancel 请求；stream 仍然自然结束（无 CancelledError）。
            registry.request_cancel(run_id)
            messages.append({"role": "assistant", "content": "中途被取消"})
            yield StreamEvent(
                EventType.FINAL_ANSWER,
                {"content": "中途被取消", "degraded": False, "filtered": False},
                {},
            )

    builds = [initial_agent, _CancelMidStreamAgent()]
    build_calls: list[int] = []

    def _record_build(**_kwargs: Any):
        idx = len(build_calls)
        build_calls.append(idx)
        return builds[idx]

    monkeypatch.setattr("dayu.host.executor.build_async_agent", _record_build)

    initial = _build_contract(user_message="问题1")
    _, handle = asyncio.run(executor.run_agent_and_wait_replayable(initial))

    replay_contract = _build_contract(user_message="再来", replay_from=handle)
    with pytest.raises(DayuCancelledError):
        asyncio.run(executor.replay_agent_and_wait(handle, replay_contract))
