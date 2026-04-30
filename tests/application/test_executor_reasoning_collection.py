"""Executor REASONING_DELTA → session_state.record_reasoning_delta 接线测试。

守住 `#118` 接线契约：

1. 当 ``agent_input.session_state`` 提供时，executor 必须把 REASONING_DELTA 流式
   增量原文转交给 ``session_state.record_reasoning_delta``。
2. ``session_state`` 缺失时 REASONING_DELTA 被静默丢弃，不抛异常、不污染流。
3. reasoning 永不进入运行态 transcript（持久化后仍守住该约束）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import dayu.host.executor as executor_module
from dayu.contracts.agent_execution import AgentCreateArgs, AgentInput
from dayu.engine.events import EventType, StreamEvent
from dayu.host import scene_preparer as scene_preparer_module
from dayu.host.conversation_session_archive import ConversationSessionArchive
from dayu.host.conversation_store import FileConversationSessionArchiveStore
from dayu.host.executor import DefaultHostExecutor
from tests.application.conftest import StubPendingTurnStore, StubRunRegistry


def _build_stub_executor() -> DefaultHostExecutor:
    """构造仅满足 ``_run_prepared_agent_stream`` 直接调用的 executor。"""

    return DefaultHostExecutor(
        run_registry=cast("executor_module.RunRegistryProtocol", StubRunRegistry()),
        pending_turn_store=cast(
            "executor_module.PendingConversationTurnStoreProtocol",
            StubPendingTurnStore(),
        ),
        scene_preparation=cast(
            "executor_module.ScenePreparationProtocol", SimpleNamespace()
        ),
    )


class _StreamScriptedAgent:
    """按脚本顺序产出 StreamEvent 的最小 stub agent。"""

    def __init__(self, script: list[StreamEvent]) -> None:
        self._script = script

    async def run_messages(self, messages, *, session_id, run_id, stream):
        del messages, session_id, run_id, stream
        for event in self._script:
            yield event


def _drain_executor_stream(
    executor: DefaultHostExecutor,
    *,
    agent_input: AgentInput,
    session_id: str | None = None,
) -> list[object]:
    """收集 ``_run_prepared_agent_stream`` 输出。

    需要先在 run_registry 注册 + start，否则 stream 结束时
    ``complete_run`` 会因为找不到 run_id 报 KeyError。
    """

    record = executor.run_registry.register_run(
        session_id=session_id, service_type="chat_turn"
    )
    executor.run_registry.start_run(record.run_id)
    collected: list[object] = []

    async def _run() -> None:
        async for event in executor._run_prepared_agent_stream(
            run_id=record.run_id,
            session_id=session_id,
            pending_turn_id=None,
            agent_input=agent_input,
        ):
            collected.append(event)

    asyncio.run(_run())
    return collected


@pytest.mark.unit
def test_executor_forwards_reasoning_delta_to_session_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REASONING_DELTA → session_state.record_reasoning_delta，FINAL_ANSWER 正常处理。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    archive = store.save(
        ConversationSessionArchive.create_empty("sess_exec1"),
        expected_revision=None,
    )
    captured: list[object] = []

    def _schedule(*, session_id, prepared_scene, transcript, system_prompt) -> None:
        del session_id, prepared_scene, transcript, system_prompt
        captured.append("scheduled")

    state = scene_preparer_module.ConversationSessionState(
        session_id="sess_exec1",
        scene_name="interactive",
        current_archive=archive,
        archive_store=store,
        memory_manager=cast(
            scene_preparer_module.DefaultConversationMemoryManager,
            SimpleNamespace(schedule_compaction=_schedule),
        ),
        prepared_scene=cast(
            scene_preparer_module.PreparedSceneState, SimpleNamespace()
        ),
        user_message="问题",
        system_prompt="SYS",
    )

    script = [
        StreamEvent(EventType.REASONING_DELTA, "想-", {}),
        StreamEvent(EventType.REASONING_DELTA, "一下", {}),
        StreamEvent(EventType.FINAL_ANSWER, {"content": "答", "degraded": False}, {}),
    ]
    monkeypatch.setattr(
        executor_module,
        "build_async_agent",
        lambda **_: _StreamScriptedAgent(script),
    )

    executor = _build_stub_executor()
    agent_input = AgentInput(
        system_prompt="SYS",
        messages=[],
        agent_create_args=AgentCreateArgs(runner_type="openai_compatible", model_name="m"),
        session_state=state,
    )
    _drain_executor_stream(executor, agent_input=agent_input, session_id="sess_exec1")

    # executor 在 FINAL_ANSWER 后已经调用 persist_turn，buffer 被清空，
    # reasoning 已经投影到 history_archive。
    assert state._reasoning_buffer == []
    persisted = store.load("sess_exec1")
    assert persisted is not None
    assert persisted.history_archive.turns[-1].assistant_reasoning == "想-一下"
    runtime_turn = persisted.runtime_transcript.turns[-1]
    assert not hasattr(runtime_turn, "assistant_reasoning")
    assert runtime_turn.assistant_final == "答"


@pytest.mark.unit
def test_executor_silently_drops_reasoning_delta_when_session_state_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """session_state 为 None 时 REASONING_DELTA 静默丢弃，run 不抛错。"""

    script = [
        StreamEvent(EventType.REASONING_DELTA, "ignored", {}),
        StreamEvent(EventType.FINAL_ANSWER, {"content": "ok", "degraded": False}, {}),
    ]
    monkeypatch.setattr(
        executor_module,
        "build_async_agent",
        lambda **_: _StreamScriptedAgent(script),
    )
    executor = _build_stub_executor()
    agent_input = AgentInput(
        system_prompt="SYS",
        messages=[],
        agent_create_args=AgentCreateArgs(runner_type="openai_compatible", model_name="m"),
        session_state=None,
    )
    # 不应抛异常
    events = _drain_executor_stream(executor, agent_input=agent_input)
    # 至少 FINAL_ANSWER 被映射出去（具体事件类型留给 build_app_event_from_stream_event 决定）
    assert events  # 非空即可，不依赖具体 mapping 数量
