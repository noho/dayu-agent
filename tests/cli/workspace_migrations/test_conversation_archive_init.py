"""conversation_archive_init 迁移测试。

覆盖 plan 用例 24-28：

- 旧 transcript 全量投影到 history_archive，原地写回原文件路径
- 单文件每条 ``assistant_reasoning == ""``、其余字段全量保留
- 二次执行幂等
- 旧目录不存在 → no-op
- 单文件 JSON 损坏 → warning 跳过、其余文件继续
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dayu.cli.workspace_migrations.conversation_archive_init import (
    migrate_conversation_archive_init,
)
from dayu.host.conversation_session_archive import ConversationSessionArchive
from dayu.host.conversation_store import (
    ConversationTranscript,
    ConversationTurnRecord,
    _serialize_transcript,
)
from dayu.workspace_paths import CONVERSATION_STORE_RELATIVE_DIR


def _make_legacy_transcript(session_id: str) -> ConversationTranscript:
    """构造含 2 个 turn 的旧 transcript（assistant_reasoning 历史不存在）。"""

    base = ConversationTranscript.create_empty(session_id)
    t1 = ConversationTurnRecord(
        turn_id="turn_1",
        scene_name="interactive",
        user_text="Q1",
        assistant_final="A1",
    )
    t2 = ConversationTurnRecord(
        turn_id="turn_2",
        scene_name="interactive",
        user_text="Q2",
        assistant_final="A2",
    )
    return base.append_turn(t1).append_turn(t2)


def _seed_legacy_file(workspace_root: Path, session_id: str) -> Path:
    """把 legacy transcript 落到目标路径。"""

    target_dir = workspace_root / CONVERSATION_STORE_RELATIVE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    transcript = _make_legacy_transcript(session_id)
    file_path = target_dir / f"{session_id}.json"
    file_path.write_text(
        json.dumps(_serialize_transcript(transcript), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return file_path


@pytest.mark.unit
def test_migration_rewrites_legacy_transcript_into_archive(tmp_path: Path) -> None:
    """旧 transcript 被原地包装成 archive，history_archive 全量投影。"""

    file_path = _seed_legacy_file(tmp_path, "sess_legacy_a")

    rewritten = migrate_conversation_archive_init(tmp_path)
    assert rewritten == 1

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    assert "runtime_transcript" in payload
    assert "history_archive" in payload

    archive = ConversationSessionArchive.from_dict(payload)
    assert archive.session_id == "sess_legacy_a"
    assert len(archive.runtime_transcript.turns) == 2
    assert len(archive.history_archive.turns) == 2

    for runtime_turn, history_turn in zip(
        archive.runtime_transcript.turns, archive.history_archive.turns
    ):
        assert history_turn.turn_id == runtime_turn.turn_id
        assert history_turn.scene_name == runtime_turn.scene_name
        assert history_turn.user_text == runtime_turn.user_text
        assert history_turn.assistant_text == runtime_turn.assistant_final
        assert history_turn.created_at == runtime_turn.created_at
        # 老会话当时没有 reasoning，迁移产物必须是空字符串
        assert history_turn.assistant_reasoning == ""


@pytest.mark.unit
def test_migration_keeps_file_path_unchanged(tmp_path: Path) -> None:
    """迁移在原路径覆盖，不产生新文件、不更换目录。"""

    file_path = _seed_legacy_file(tmp_path, "sess_legacy_b")
    target_dir = file_path.parent
    other_files_before = sorted(p.name for p in target_dir.iterdir())

    migrate_conversation_archive_init(tmp_path)

    other_files_after = sorted(p.name for p in target_dir.iterdir())
    assert other_files_before == other_files_after
    assert file_path.exists()


@pytest.mark.unit
def test_migration_is_idempotent_on_new_schema(tmp_path: Path) -> None:
    """二次运行识别到新 schema → no-op，不动文件。"""

    file_path = _seed_legacy_file(tmp_path, "sess_legacy_c")
    assert migrate_conversation_archive_init(tmp_path) == 1
    text_after_first = file_path.read_text(encoding="utf-8")
    assert migrate_conversation_archive_init(tmp_path) == 0
    assert file_path.read_text(encoding="utf-8") == text_after_first


@pytest.mark.unit
def test_migration_returns_zero_when_target_dir_missing(tmp_path: Path) -> None:
    """旧目录不存在直接返回 0。"""

    assert migrate_conversation_archive_init(tmp_path) == 0


@pytest.mark.unit
def test_migration_skips_corrupted_file_and_continues(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """单文件 JSON 损坏只 warning 跳过，不阻塞同目录其他文件。"""

    good_file = _seed_legacy_file(tmp_path, "sess_legacy_good")
    bad_file = good_file.parent / "sess_legacy_bad.json"
    bad_file.write_text("{ this is not valid json", encoding="utf-8")

    rewritten = migrate_conversation_archive_init(tmp_path)
    assert rewritten == 1

    captured = capsys.readouterr()
    assert "sess_legacy_bad" in captured.err

    # 好文件成功升级
    good_payload = json.loads(good_file.read_text(encoding="utf-8"))
    assert "runtime_transcript" in good_payload
    # 损坏文件保持原样
    assert bad_file.read_text(encoding="utf-8") == "{ this is not valid json"


@pytest.mark.unit
def test_migration_skips_files_without_legacy_turns_key(tmp_path: Path) -> None:
    """既无 ``runtime_transcript`` 也无 ``turns`` 的对象不算旧 schema，跳过。"""

    target_dir = tmp_path / CONVERSATION_STORE_RELATIVE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    odd_file = target_dir / "weird.json"
    odd_file.write_text(
        json.dumps({"session_id": "x", "revision": "y"}, ensure_ascii=False),
        encoding="utf-8",
    )

    rewritten = migrate_conversation_archive_init(tmp_path)
    assert rewritten == 0
    payload = json.loads(odd_file.read_text(encoding="utf-8"))
    assert "runtime_transcript" not in payload
