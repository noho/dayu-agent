"""ConversationStore 测试。"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from dayu.host._coercion import _coerce_string_tuple
from dayu.host.conversation_store import (
    ConversationEpisodeSummary,
    ConversationPinnedState,
    ConversationToolUseSummary,
    ConversationTranscript,
    ConversationTurnRecord,
    FileConversationStore,
)
from dayu.log import Log


@pytest.mark.unit
def test_coerce_string_tuple_filters_blank_values() -> None:
    """共享字符串元组规范化 helper 应过滤空白并忽略非列表输入。"""

    assert _coerce_string_tuple([" AAPL ", "", None, 0, "  ", "MSFT"]) == ("AAPL", "MSFT")
    assert _coerce_string_tuple("AAPL") == ()


@pytest.mark.unit
def test_file_conversation_store_roundtrip_keeps_derived_memory(tmp_path: Path) -> None:
    """验证文件存储可正确保存并读取 transcript 的分层记忆字段。"""

    store = FileConversationStore(tmp_path / "conversations")
    transcript = ConversationTranscript.create_empty("sess_1").append_turn(
        ConversationTurnRecord(
            turn_id="turn_1",
            scene_name="interactive",
            user_text="苹果营收是多少",
            assistant_final="营收是 100。",
            tool_uses=(
                ConversationToolUseSummary(
                    name="list_documents",
                    arguments={"ticker": "AAPL"},
                    result_summary='{"documents": 3}',
                ),
            ),
        )
    ).replace_memory(
        pinned_state=ConversationPinnedState(
            current_goal="跟踪苹果最新经营变化",
            confirmed_subjects=("AAPL",),
            user_constraints=("只看最近财报",),
            open_questions=("Q1 毛利率变化",),
        ),
        episodes=(
            ConversationEpisodeSummary(
                episode_id="ep_1",
                start_turn_id="turn_1",
                end_turn_id="turn_1",
                title="确认分析对象",
                goal="确定当前会话主题",
                confirmed_facts=("分析对象是苹果",),
            ),
        ),
        compacted_turn_count=1,
    )

    store.save(transcript)
    loaded = store.load("sess_1")

    assert loaded is not None
    assert loaded.session_id == "sess_1"
    assert loaded.compacted_turn_count == 1
    assert loaded.pinned_state.current_goal == "跟踪苹果最新经营变化"
    assert loaded.pinned_state.confirmed_subjects == ("AAPL",)
    assert loaded.episodes[0].title == "确认分析对象"
    assert loaded.turns[0].tool_uses[0].name == "list_documents"


@pytest.mark.unit
def test_file_conversation_store_rejects_revision_conflict(tmp_path: Path) -> None:
    """验证文件存储会拒绝 revision 冲突。"""

    store = FileConversationStore(tmp_path / "conversations")
    transcript = ConversationTranscript.create_empty("sess_1")
    store.save(transcript)
    next_transcript = transcript.append_turn(
        ConversationTurnRecord(
            turn_id="turn_1",
            scene_name="interactive",
            user_text="Q1",
            assistant_final="A1",
        )
    )

    with pytest.raises(RuntimeError, match="revision 冲突"):
        store.save(next_transcript, expected_revision="stale_revision")


@pytest.mark.unit
def test_file_conversation_store_first_save_with_expected_revision_succeeds(tmp_path: Path) -> None:
    """验证首次保存（文件不存在）时 expected_revision 不触发冲突。

    首轮交互 transcript 仅存在于内存，persist_turn 带着内存 revision 调用 save，
    此时文件不存在不应被视为冲突。
    """

    store = FileConversationStore(tmp_path / "conversations")
    transcript = ConversationTranscript.create_empty("sess_first")
    next_transcript = transcript.append_turn(
        ConversationTurnRecord(
            turn_id="turn_1",
            scene_name="interactive",
            user_text="Q1",
            assistant_final="A1",
        )
    )

    # 文件不存在时带 expected_revision 应正常保存
    result = store.save(next_transcript, expected_revision=transcript.revision)
    assert result.session_id == "sess_first"

    loaded = store.load("sess_first")
    assert loaded is not None
    assert len(loaded.turns) == 1


@pytest.mark.unit
def test_file_conversation_store_rejects_unsafe_session_id(tmp_path: Path) -> None:
    """验证文件存储会拒绝包含路径语义的 session_id。"""

    store = FileConversationStore(tmp_path / "conversations")

    with pytest.raises(ValueError, match="session_id 只能包含"):
        store.load("../sess_1")


@pytest.mark.unit
def test_file_conversation_store_emits_load_save_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ConversationStore 读写 transcript 时应输出调试日志。"""

    store = FileConversationStore(tmp_path / "conversations")
    debug_mock = Mock()
    monkeypatch.setattr(Log, "debug", debug_mock)

    transcript = ConversationTranscript.create_empty("sess_log").append_turn(
        ConversationTurnRecord(
            turn_id="turn_1",
            scene_name="interactive",
            user_text="Q1",
            assistant_final="A1",
        )
    )

    assert store.load("sess_log") is None
    store.save(transcript)
    loaded = store.load("sess_log")

    assert loaded is not None
    debug_messages = [call.args[0] for call in debug_mock.call_args_list]
    assert any("transcript 不存在" in message for message in debug_messages)
    assert any("保存 transcript" in message for message in debug_messages)
    assert any("加载 transcript" in message for message in debug_messages)


@pytest.mark.unit
def test_file_conversation_store_serializes_concurrent_save_and_surfaces_revision_conflict(
    tmp_path: Path,
) -> None:
    """并发保存同一 session 时，后写线程必须在锁内看到 revision 冲突。"""

    store = FileConversationStore(tmp_path / "conversations")
    base = ConversationTranscript.create_empty("sess_race")
    store.save(base)
    first_update = base.append_turn(
        ConversationTurnRecord(
            turn_id="turn_1",
            scene_name="interactive",
            user_text="Q1",
            assistant_final="A1",
        )
    )
    second_update = base.append_turn(
        ConversationTurnRecord(
            turn_id="turn_2",
            scene_name="interactive",
            user_text="Q2",
            assistant_final="A2",
        )
    )

    barrier = threading.Barrier(3)
    results: list[str] = []
    errors: list[str] = []
    result_lock = threading.Lock()

    def _save_worker(transcript: ConversationTranscript) -> None:
        """并发执行保存。"""

        barrier.wait()
        try:
            saved = store.save(transcript, expected_revision=base.revision)
        except RuntimeError as exc:
            with result_lock:
                errors.append(str(exc))
            return
        with result_lock:
            results.append(saved.revision)

    first_thread = threading.Thread(target=_save_worker, args=(first_update,))
    second_thread = threading.Thread(target=_save_worker, args=(second_update,))
    first_thread.start()
    second_thread.start()
    barrier.wait()
    first_thread.join()
    second_thread.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert "revision 冲突" in errors[0]

    loaded = store.load("sess_race")
    assert loaded is not None
    assert loaded.revision == results[0]
    assert len(loaded.turns) == 1
