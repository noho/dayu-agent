"""ConversationSessionArchiveStore 测试。"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from dayu.host._coercion import _coerce_string_tuple
from dayu.host.conversation_session_archive import (
    ConversationHistoryArchive,
    ConversationHistoryTurnRecord,
    ConversationSessionArchive,
)
from dayu.host.conversation_store import (
    ConversationEpisodeSummary,
    ConversationPinnedState,
    ConversationToolUseSummary,
    ConversationTranscript,
    ConversationTurnRecord,
    FileConversationSessionArchiveStore,
)
from dayu.log import Log


def _make_archive_with_one_turn(session_id: str) -> ConversationSessionArchive:
    """构造含一条 runtime turn + 同步 history 的 archive。"""

    archive = ConversationSessionArchive.create_empty(session_id)
    turn = ConversationTurnRecord(
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
    history = ConversationHistoryTurnRecord(
        turn_id=turn.turn_id,
        scene_name=turn.scene_name,
        user_text=turn.user_text,
        assistant_text=turn.assistant_final,
        assistant_reasoning="",
        created_at=turn.created_at,
    )
    return archive.with_next_turn(turn, history)


@pytest.mark.unit
def test_coerce_string_tuple_filters_blank_values() -> None:
    """共享字符串元组规范化 helper 应过滤空白并忽略非列表输入。"""

    assert _coerce_string_tuple([" AAPL ", "", None, 0, "  ", "MSFT"]) == ("AAPL", "MSFT")
    assert _coerce_string_tuple("AAPL") == ()


@pytest.mark.unit
def test_archive_store_roundtrip_keeps_runtime_and_history(tmp_path: Path) -> None:
    """archive 存储读写 roundtrip 后两个子视图原样保留。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    archive = _make_archive_with_one_turn("sess_1")
    next_runtime = archive.runtime_transcript.replace_memory(
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
    next_archive = archive.with_runtime_transcript(next_runtime)
    store.save(next_archive, expected_revision=archive.revision)

    loaded = store.load("sess_1")
    assert loaded is not None
    assert loaded.session_id == "sess_1"
    assert loaded.runtime_transcript.compacted_turn_count == 1
    assert loaded.runtime_transcript.pinned_state.current_goal == "跟踪苹果最新经营变化"
    assert loaded.runtime_transcript.episodes[0].title == "确认分析对象"
    assert loaded.runtime_transcript.turns[0].tool_uses[0].name == "list_documents"
    assert loaded.history_archive.turns[0].user_text == "苹果营收是多少"
    assert loaded.history_archive.turns[0].assistant_text == "营收是 100。"
    assert loaded.history_archive.turns[0].assistant_reasoning == ""


@pytest.mark.unit
def test_archive_store_rejects_revision_conflict(tmp_path: Path) -> None:
    """archive 存储在 expected_revision 不匹配时拒绝写入。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    archive = ConversationSessionArchive.create_empty("sess_1")
    store.save(archive, expected_revision=None)
    next_archive = _make_archive_with_one_turn("sess_1")

    with pytest.raises(RuntimeError, match="revision 冲突"):
        store.save(next_archive, expected_revision="stale_revision")


@pytest.mark.unit
def test_archive_store_first_save_with_expected_revision_succeeds(tmp_path: Path) -> None:
    """文件不存在时带 expected_revision 不应被视为冲突。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    archive = _make_archive_with_one_turn("sess_first")

    saved = store.save(archive, expected_revision="any_initial_revision")
    assert saved.session_id == "sess_first"

    loaded = store.load("sess_first")
    assert loaded is not None
    assert len(loaded.runtime_transcript.turns) == 1


@pytest.mark.unit
def test_archive_store_rejects_unsafe_session_id(tmp_path: Path) -> None:
    """archive 存储拒绝包含路径语义的 session_id。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    with pytest.raises(ValueError, match="session_id 只能包含"):
        store.load("../sess_1")


@pytest.mark.unit
def test_archive_store_load_legacy_schema_raises(tmp_path: Path) -> None:
    """加载旧 transcript schema 时显式报错，提示运行迁移。"""

    target_dir = tmp_path / "conversations"
    target_dir.mkdir(parents=True)
    legacy_path = target_dir / "sess_legacy.json"
    legacy_path.write_text(
        '{"session_id": "sess_legacy", "revision": "rev_x", "turns": []}',
        encoding="utf-8",
    )

    store = FileConversationSessionArchiveStore(target_dir)
    with pytest.raises(RuntimeError, match="dayu-cli init"):
        store.load("sess_legacy")


@pytest.mark.unit
def test_archive_store_emits_load_save_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """archive 存储读写 archive 时输出调试日志。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    debug_mock = Mock()
    monkeypatch.setattr(Log, "debug", debug_mock)

    archive = _make_archive_with_one_turn("sess_log")
    assert store.load("sess_log") is None
    store.save(archive, expected_revision=None)
    loaded = store.load("sess_log")

    assert loaded is not None
    debug_messages = [call.args[0] for call in debug_mock.call_args_list]
    assert any(
        "transcript" in message or "archive" in message for message in debug_messages
    )


@pytest.mark.unit
def test_archive_store_serializes_concurrent_save_and_surfaces_revision_conflict(
    tmp_path: Path,
) -> None:
    """并发保存同一 session 时后写线程在锁内看到 revision 冲突。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    base = ConversationSessionArchive.create_empty("sess_race")
    store.save(base, expected_revision=None)

    def _build_update(turn_id: str, text: str) -> ConversationSessionArchive:
        turn = ConversationTurnRecord(
            turn_id=turn_id,
            scene_name="interactive",
            user_text=text,
            assistant_final=text + "_a",
        )
        history = ConversationHistoryTurnRecord(
            turn_id=turn.turn_id,
            scene_name=turn.scene_name,
            user_text=turn.user_text,
            assistant_text=turn.assistant_final,
            assistant_reasoning="",
            created_at=turn.created_at,
        )
        return base.with_next_turn(turn, history)

    first_update = _build_update("turn_1", "Q1")
    second_update = _build_update("turn_2", "Q2")

    barrier = threading.Barrier(3)
    results: list[str] = []
    errors: list[str] = []
    result_lock = threading.Lock()

    def _save_worker(target: ConversationSessionArchive) -> None:
        """并发执行保存。"""

        barrier.wait()
        try:
            saved = store.save(target, expected_revision=base.revision)
        except RuntimeError as exc:
            with result_lock:
                errors.append(str(exc))
            return
        with result_lock:
            results.append(saved.revision)

    threads = [
        threading.Thread(target=_save_worker, args=(first_update,)),
        threading.Thread(target=_save_worker, args=(second_update,)),
    ]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert "revision 冲突" in errors[0]

    loaded = store.load("sess_race")
    assert loaded is not None
    assert loaded.revision == results[0]
    assert len(loaded.runtime_transcript.turns) == 1


@pytest.mark.unit
def test_archive_store_append_turn_reconciles_with_live(tmp_path: Path) -> None:
    """append_turn 在 live archive 之上 reconcile 推进。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    base = _make_archive_with_one_turn("sess_reconcile")
    store.save(base, expected_revision=None)

    new_turn = ConversationTurnRecord(
        turn_id="turn_2",
        scene_name="interactive",
        user_text="第二轮问题",
        assistant_final="第二轮回答",
    )
    new_history = ConversationHistoryTurnRecord(
        turn_id=new_turn.turn_id,
        scene_name=new_turn.scene_name,
        user_text=new_turn.user_text,
        assistant_text=new_turn.assistant_final,
        assistant_reasoning="思考片段",
        created_at=new_turn.created_at,
    )
    persisted = store.append_turn(
        "sess_reconcile",
        turn_record=new_turn,
        history_record=new_history,
    )

    assert len(persisted.runtime_transcript.turns) == 2
    assert persisted.history_archive.turns[-1].assistant_reasoning == "思考片段"


@pytest.mark.unit
def test_archive_store_append_turn_raises_when_live_missing(tmp_path: Path) -> None:
    """live archive 缺失时 append_turn 显式抛错而非自动 create_empty。"""

    from dayu.host.conversation_session_archive import ConversationArchiveMissingError

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    new_turn = ConversationTurnRecord(
        turn_id="turn_x",
        scene_name="interactive",
        user_text="Q",
        assistant_final="A",
    )
    new_history = ConversationHistoryTurnRecord(
        turn_id=new_turn.turn_id,
        scene_name=new_turn.scene_name,
        user_text=new_turn.user_text,
        assistant_text=new_turn.assistant_final,
        assistant_reasoning="",
        created_at=new_turn.created_at,
    )
    with pytest.raises(ConversationArchiveMissingError):
        store.append_turn(
            "sess_missing", turn_record=new_turn, history_record=new_history
        )


@pytest.mark.unit
def test_archive_store_delete_returns_correct_status(tmp_path: Path) -> None:
    """delete 在文件存在/不存在两种情况下返回正确状态。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    assert store.delete("sess_absent") is False

    archive = ConversationSessionArchive.create_empty("sess_delete")
    store.save(archive, expected_revision=None)
    assert store.delete("sess_delete") is True
    assert store.load("sess_delete") is None


@pytest.mark.unit
def test_archive_save_failure_keeps_old_file_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """模拟原子写失败时旧文件保持完整。"""

    from dayu.host import conversation_store as cs

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    base = _make_archive_with_one_turn("sess_keep")
    store.save(base, expected_revision=None)

    def _raise_on_atomic_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(cs, "_atomic_write_text", _raise_on_atomic_write)

    next_turn = ConversationTurnRecord(
        turn_id="turn_2",
        scene_name="interactive",
        user_text="Q2",
        assistant_final="A2",
    )
    next_history = ConversationHistoryTurnRecord(
        turn_id=next_turn.turn_id,
        scene_name=next_turn.scene_name,
        user_text=next_turn.user_text,
        assistant_text=next_turn.assistant_final,
        assistant_reasoning="",
        created_at=next_turn.created_at,
    )
    next_archive = base.with_next_turn(next_turn, next_history)
    with pytest.raises(OSError):
        store.save(next_archive, expected_revision=base.revision)

    loaded = store.load("sess_keep")
    assert loaded is not None
    assert loaded.revision == base.revision
    assert len(loaded.runtime_transcript.turns) == 1


def test_history_archive_create_empty_rejects_blank_session_id() -> None:
    """history archive 拒绝空 session_id。"""

    with pytest.raises(ValueError):
        ConversationHistoryArchive.create_empty("   ")


def test_archive_to_dict_from_dict_roundtrip() -> None:
    """archive to_dict / from_dict 等价 roundtrip。"""

    archive = _make_archive_with_one_turn("sess_rt")
    payload = archive.to_dict()
    restored = ConversationSessionArchive.from_dict(payload)
    assert restored.session_id == archive.session_id
    assert restored.revision == archive.revision
    assert len(restored.runtime_transcript.turns) == 1
    assert len(restored.history_archive.turns) == 1


def test_transcript_legacy_alias_keeps_imports_working() -> None:
    """ConversationTranscript 仍可以从 conversation_store 导入。"""

    transcript = ConversationTranscript.create_empty("sess_alias")
    assert transcript.session_id == "sess_alias"


def test_archive_from_dict_rejects_missing_history_archive() -> None:
    """``history_archive`` 缺失时 ``from_dict`` fail-closed，不静默降级为空历史。

    守住硬约束：``history_archive`` 是聚合根的一部分，缺失视为数据损坏，
    必须显式报错，避免 ``persist_turn`` / compaction 后续写回时把已存的
    ``assistant_reasoning`` 永久抹掉。
    """

    archive = _make_archive_with_one_turn("sess_no_history")
    payload = archive.to_dict()
    payload.pop("history_archive")
    with pytest.raises(ValueError, match="history_archive"):
        ConversationSessionArchive.from_dict(payload)


def test_archive_from_dict_rejects_non_object_history_archive() -> None:
    """``history_archive`` 非对象时同样 fail-closed。"""

    archive = _make_archive_with_one_turn("sess_bad_history")
    payload = archive.to_dict()
    payload["history_archive"] = "broken"
    with pytest.raises(ValueError, match="history_archive"):
        ConversationSessionArchive.from_dict(payload)


def test_load_or_create_returns_existing_archive_without_overwrite(tmp_path: Path) -> None:
    """磁盘已有 archive 时 ``load_or_create`` 必须返回 live，不写空版本覆盖。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    seeded = _make_archive_with_one_turn("sess_loc1")
    store.save(seeded, expected_revision=None)
    before_text = (tmp_path / "conversations" / "sess_loc1.json").read_text("utf-8")

    loaded = store.load_or_create("sess_loc1")
    assert loaded.revision == seeded.revision
    assert len(loaded.runtime_transcript.turns) == 1
    after_text = (tmp_path / "conversations" / "sess_loc1.json").read_text("utf-8")
    assert before_text == after_text


def test_load_or_create_atomically_creates_when_missing(tmp_path: Path) -> None:
    """磁盘缺失时 ``load_or_create`` 在锁内创建空 archive，文件落盘可读。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    created = store.load_or_create("sess_loc2")
    assert created.session_id == "sess_loc2"
    assert created.runtime_transcript.turns == ()
    assert created.history_archive.turns == ()

    persisted = store.load("sess_loc2")
    assert persisted is not None
    assert persisted.revision == created.revision


def test_load_or_create_does_not_overwrite_concurrently_written_archive(tmp_path: Path) -> None:
    """模拟并发：两个进程都先 ``load`` -> ``None``，但只有先到的能在锁内胜出，
    另一个调用必须读到 live archive，不能用空版本覆盖。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    # 第一个进程已经 ``load_or_create`` 完成（live archive 落盘）
    first = store.load_or_create("sess_loc3")
    # 第二个进程紧随其后再调一次 —— 必须读到 live，不写新空版本
    second = store.load_or_create("sess_loc3")
    assert second.revision == first.revision
    assert second.runtime_transcript.turns == ()
