"""resume 链路不携带 reasoning + reconcile-on-write 契约测试。

守住 `#118` 关键契约：

1. ``serialize_prepared_agent_turn_snapshot`` 序列化结果不出现
   ``assistant_reasoning`` key/字符串。
2. resume 输出 reconcile（live 存在）：``persist_turn`` 在 live 之上推进，
   不会用 prepared snapshot 的旧 transcript 覆盖 live。
3. resume 输出 reconcile（live 缺失）：``persist_turn`` 抛
   ``ConversationArchiveMissingError``，不静默 ``create_empty``。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from dayu.contracts.agent_execution import AgentCreateArgs
from dayu.execution.options import ConversationMemorySettings
from dayu.host import scene_preparer as scene_preparer_module
from dayu.host.conversation_session_archive import (
    ConversationArchiveMissingError,
    ConversationHistoryArchive,
    ConversationSessionArchive,
)
from dayu.host.conversation_store import (
    ConversationTranscript,
    ConversationTurnRecord,
    FileConversationSessionArchiveStore,
)
from dayu.host.prepared_turn import (
    PreparedAgentTurnSnapshot,
    PreparedConversationSessionSnapshot,
    serialize_prepared_agent_turn_snapshot,
)
from dayu.contracts.host_execution import ConcurrencyAcquirePolicy
from dayu.contracts.execution_metadata import empty_execution_delivery_context
from dayu.contracts.agent_execution import ExecutionPermissions


def _make_prepared_snapshot(transcript: ConversationTranscript) -> PreparedAgentTurnSnapshot:
    """构造最小 PreparedAgentTurnSnapshot，绑定 conversation_session。"""

    return PreparedAgentTurnSnapshot(
        service_name="chat_turn",
        scene_name="interactive",
        metadata=empty_execution_delivery_context(),
        business_concurrency_lane=None,
        timeout_ms=None,
        resumable=True,
        system_prompt="SYS",
        messages=[],
        agent_create_args=AgentCreateArgs(runner_type="openai_compatible", model_name="m"),
        selected_toolsets=(),
        execution_permissions=ExecutionPermissions(),
        toolset_configs=(),
        trace_settings=None,
        conversation_memory_settings=ConversationMemorySettings(),
        concurrency_acquire_policy=ConcurrencyAcquirePolicy.use_host_default(),
        trace_identity=None,
        conversation_session=PreparedConversationSessionSnapshot(
            session_id=transcript.session_id,
            user_message="Q-resume",
            transcript=transcript,
        ),
    )


@pytest.mark.unit
def test_prepared_snapshot_serialization_does_not_carry_reasoning() -> None:
    """prepared snapshot 序列化中既无 ``assistant_reasoning`` key 也无对应字符串。"""

    transcript = ConversationTranscript.create_empty("sess_resume_serial")
    transcript = transcript.append_turn(
        ConversationTurnRecord(
            turn_id="turn_1",
            scene_name="interactive",
            user_text="Q1",
            assistant_final="A1",
        )
    )
    snapshot = _make_prepared_snapshot(transcript)

    payload = serialize_prepared_agent_turn_snapshot(snapshot)
    serialized_text = json.dumps(payload, ensure_ascii=False)

    assert "assistant_reasoning" not in serialized_text
    assert "ConversationHistoryArchive" not in serialized_text
    assert "ConversationHistoryTurnRecord" not in serialized_text


def _build_session_state_from_resume_path(
    *,
    archive_store: scene_preparer_module.ConversationSessionArchiveStore,
    session_id: str,
    transcript: ConversationTranscript,
    user_message: str = "Q-resume",
) -> scene_preparer_module.ConversationSessionState:
    """直接复用 restore 路径的 placeholder archive 构造逻辑。"""

    placeholder = ConversationSessionArchive(
        session_id=session_id,
        revision="",
        created_at=transcript.created_at,
        updated_at=transcript.updated_at,
        runtime_transcript=transcript,
        history_archive=ConversationHistoryArchive.create_empty(session_id),
    )

    def _schedule(*, session_id, prepared_scene, transcript, system_prompt) -> None:
        del session_id, prepared_scene, transcript, system_prompt

    return scene_preparer_module.ConversationSessionState(
        session_id=session_id,
        scene_name="interactive",
        current_archive=placeholder,
        archive_store=archive_store,
        memory_manager=cast(
            scene_preparer_module.DefaultConversationMemoryManager,
            SimpleNamespace(schedule_compaction=_schedule),
        ),
        prepared_scene=cast(
            scene_preparer_module.PreparedSceneState, SimpleNamespace()
        ),
        user_message=user_message,
        system_prompt="SYS",
    )


@pytest.mark.unit
def test_resume_persist_turn_reconciles_with_live_archive(tmp_path: Path) -> None:
    """live 存在时 ``persist_turn`` 把新轮次合并到 live archive 之上，不覆盖。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    # live archive 含一条历史 turn
    base = ConversationSessionArchive.create_empty("sess_resume_live")
    seeded_turn = ConversationTurnRecord(
        turn_id="turn_live_1",
        scene_name="interactive",
        user_text="先前",
        assistant_final="先前回答",
    )
    from dayu.host.conversation_session_archive import ConversationHistoryTurnRecord

    seeded_history = ConversationHistoryTurnRecord(
        turn_id=seeded_turn.turn_id,
        scene_name=seeded_turn.scene_name,
        user_text=seeded_turn.user_text,
        assistant_text=seeded_turn.assistant_final,
        assistant_reasoning="historical-reasoning",
        created_at=seeded_turn.created_at,
    )
    live = base.with_next_turn(seeded_turn, seeded_history)
    store.save(live, expected_revision=None)

    # prepared snapshot 中的 transcript 故意只包含旧空 transcript（模拟 prepare 时刻），
    # 不应把 live 已有的轮次抹掉。
    prepared_transcript = base.runtime_transcript
    state = _build_session_state_from_resume_path(
        archive_store=store,
        session_id="sess_resume_live",
        transcript=prepared_transcript,
    )
    state.record_reasoning_delta("resume-reasoning")

    state.persist_turn(
        final_content="resume 回答",
        degraded=False,
        tool_uses=(),
        warnings=(),
        errors=(),
    )

    persisted = store.load("sess_resume_live")
    assert persisted is not None
    # live 的旧 turn + resume 新 turn 都在
    assert [t.turn_id for t in persisted.runtime_transcript.turns] == [
        "turn_live_1",
        persisted.runtime_transcript.turns[-1].turn_id,
    ]
    assert len(persisted.runtime_transcript.turns) == 2
    # history 同步推进，旧历史 reasoning 保留，新历史 reasoning 落盘
    assert persisted.history_archive.turns[0].assistant_reasoning == "historical-reasoning"
    assert persisted.history_archive.turns[-1].assistant_reasoning == "resume-reasoning"


@pytest.mark.unit
def test_resume_persist_turn_raises_when_live_archive_missing(tmp_path: Path) -> None:
    """live archive 不存在时 ``persist_turn`` 抛 ``ConversationArchiveMissingError``。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    transcript = ConversationTranscript.create_empty("sess_resume_missing")

    state = _build_session_state_from_resume_path(
        archive_store=store,
        session_id="sess_resume_missing",
        transcript=transcript,
    )

    with pytest.raises(ConversationArchiveMissingError):
        state.persist_turn(
            final_content="x",
            degraded=False,
            tool_uses=(),
            warnings=(),
            errors=(),
        )

    # 显式契约：不静默 create_empty，磁盘仍无 archive
    assert store.load("sess_resume_missing") is None
