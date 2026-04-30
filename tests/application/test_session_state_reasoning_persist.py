"""ConversationSessionState `record_reasoning_delta` + `persist_turn` 集成测试。

守住 `#118` 关键契约：

1. reasoning 仅以 buffer 形式存活，到 ``persist_turn`` 时才投影进
   ``history_archive``，**永不**进入 ``runtime_transcript`` / ``ConversationTurnRecord``。
2. ``persist_turn`` 失败时 buffer 不清空，磁盘旧 archive 完整。
3. ``persist_turn`` 成功后清空 buffer，下一轮不串味。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from dayu.host import scene_preparer as scene_preparer_module
from dayu.host.conversation_session_archive import (
    ConversationSessionArchive,
)
from dayu.host.conversation_store import (
    ConversationToolUseSummary,
    FileConversationSessionArchiveStore,
)


def _make_session_state(
    archive_store: scene_preparer_module.ConversationSessionArchiveStore,
    *,
    archive: ConversationSessionArchive,
    user_message: str = "Q1",
) -> scene_preparer_module.ConversationSessionState:
    """构造最小 ConversationSessionState 用于单测。"""

    scheduled: list[object] = []

    def _schedule(*, session_id, prepared_scene, transcript, system_prompt) -> None:
        del prepared_scene, system_prompt
        scheduled.append((session_id, transcript))

    state = scene_preparer_module.ConversationSessionState(
        session_id=archive.session_id,
        scene_name="interactive",
        current_archive=archive,
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
    return state


@pytest.mark.unit
def test_record_reasoning_delta_persists_to_history_archive_only(tmp_path: Path) -> None:
    """多次 reasoning_delta 拼接结果只写入 history_archive，runtime 不含。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    archive = store.save(
        ConversationSessionArchive.create_empty("sess_r1"), expected_revision=None
    )
    state = _make_session_state(store, archive=archive, user_message="苹果营收?")

    state.record_reasoning_delta("先")
    state.record_reasoning_delta("分析")
    state.record_reasoning_delta("营收")

    state.persist_turn(
        final_content="100 亿",
        degraded=False,
        tool_uses=(ConversationToolUseSummary(name="search"),),
        warnings=(),
        errors=(),
    )

    persisted = store.load("sess_r1")
    assert persisted is not None
    runtime_turn = persisted.runtime_transcript.turns[-1]
    history_turn = persisted.history_archive.turns[-1]

    assert history_turn.assistant_reasoning == "先分析营收"
    assert history_turn.turn_id == runtime_turn.turn_id
    assert history_turn.assistant_text == "100 亿"
    assert not hasattr(runtime_turn, "assistant_reasoning")


@pytest.mark.unit
def test_persist_turn_without_reasoning_yields_empty_string(tmp_path: Path) -> None:
    """未调用 reasoning_delta 时 history_archive 中 reasoning 为空字符串。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    archive = store.save(
        ConversationSessionArchive.create_empty("sess_r2"), expected_revision=None
    )
    state = _make_session_state(store, archive=archive)

    state.persist_turn(
        final_content="ok",
        degraded=False,
        tool_uses=(),
        warnings=(),
        errors=(),
    )
    persisted = store.load("sess_r2")
    assert persisted is not None
    assert persisted.history_archive.turns[-1].assistant_reasoning == ""


@pytest.mark.unit
def test_persist_turn_failure_keeps_reasoning_buffer_and_old_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """archive_store.save 抛错时 buffer 不清空，磁盘旧 archive 完整。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    archive = store.save(
        ConversationSessionArchive.create_empty("sess_r3"), expected_revision=None
    )
    state = _make_session_state(store, archive=archive)
    state.record_reasoning_delta("buffered-thought")

    from dayu.host import conversation_store as cs

    def _raise(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(cs, "_atomic_write_text", _raise)

    with pytest.raises(OSError):
        state.persist_turn(
            final_content="x",
            degraded=False,
            tool_uses=(),
            warnings=(),
            errors=(),
        )

    # buffer 未清空（重试不丢内容）
    assert state._reasoning_buffer == ["buffered-thought"]

    # 磁盘 archive 仍是初始空 archive
    persisted = store.load("sess_r3")
    assert persisted is not None
    assert len(persisted.runtime_transcript.turns) == 0
    assert len(persisted.history_archive.turns) == 0


@pytest.mark.unit
def test_persist_turn_success_clears_buffer_for_next_turn(tmp_path: Path) -> None:
    """成功落盘后 buffer 已清空，下一轮 reasoning 不串味。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    archive = store.save(
        ConversationSessionArchive.create_empty("sess_r4"), expected_revision=None
    )
    state = _make_session_state(store, archive=archive)

    state.record_reasoning_delta("first-turn")
    state.persist_turn(
        final_content="A1",
        degraded=False,
        tool_uses=(),
        warnings=(),
        errors=(),
    )
    assert state._reasoning_buffer == []

    state.user_message = "Q2"
    state.record_reasoning_delta("second-turn-only")
    state.persist_turn(
        final_content="A2",
        degraded=False,
        tool_uses=(),
        warnings=(),
        errors=(),
    )

    persisted = store.load("sess_r4")
    assert persisted is not None
    assert [t.assistant_reasoning for t in persisted.history_archive.turns] == [
        "first-turn",
        "second-turn-only",
    ]


@pytest.mark.unit
def test_record_reasoning_delta_ignores_blank_chunks(tmp_path: Path) -> None:
    """空白/None chunk 不会污染 buffer。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    archive = store.save(
        ConversationSessionArchive.create_empty("sess_r5"), expected_revision=None
    )
    state = _make_session_state(store, archive=archive)

    state.record_reasoning_delta("")
    state.record_reasoning_delta("real")
    assert state._reasoning_buffer == ["real"]
