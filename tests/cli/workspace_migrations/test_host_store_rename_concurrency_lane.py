"""Host SQLite ``concurrency_lane`` 改名迁移单测。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from dayu.cli.workspace_migrations.host_store_rename_concurrency_lane import (
    migrate_host_store_rename_concurrency_lane,
)


def _build_pending_turns_db(db_path: Path, rows: list[tuple[str, str]]) -> None:
    """构造一个最小的 ``pending_conversation_turns`` 表并插入指定行。"""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE pending_conversation_turns (
                pending_turn_id TEXT PRIMARY KEY,
                resume_source_json TEXT NOT NULL DEFAULT ''
            )
            """
        )
        conn.executemany(
            "INSERT INTO pending_conversation_turns (pending_turn_id, resume_source_json) VALUES (?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _read_row(db_path: Path, pending_turn_id: str) -> str:
    """读取指定行的 ``resume_source_json`` 内容。"""

    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(
            "SELECT resume_source_json FROM pending_conversation_turns WHERE pending_turn_id = ?",
            (pending_turn_id,),
        )
        row = cur.fetchone()
        assert row is not None
        return str(row[0])
    finally:
        conn.close()


@pytest.mark.unit
def test_migration_renames_top_level_concurrency_lane(tmp_path: Path) -> None:
    """顶层 ``host_policy.concurrency_lane`` 应被改名。"""

    db_path = tmp_path / "dayu_host.db"
    payload = {
        "host_policy": {
            "concurrency_lane": "write_chapter",
            "timeout_ms": 1000,
        }
    }
    _build_pending_turns_db(db_path, [("p1", json.dumps(payload))])

    rewritten = migrate_host_store_rename_concurrency_lane(db_path)
    assert rewritten == 1

    new_raw = _read_row(db_path, "p1")
    new_payload = json.loads(new_raw)
    assert "concurrency_lane" not in new_payload["host_policy"]
    assert new_payload["host_policy"]["business_concurrency_lane"] == "write_chapter"
    assert new_payload["host_policy"]["timeout_ms"] == 1000


@pytest.mark.unit
def test_migration_renames_nested_prepared_snapshot(tmp_path: Path) -> None:
    """嵌套 ``PreparedAgentTurnSnapshot.concurrency_lane`` 也要被改名。"""

    db_path = tmp_path / "dayu_host.db"
    payload = {
        "prepared_turn": {
            "scene_name": "x",
            "concurrency_lane": "sec_download",
            "messages": [{"role": "user", "content": "hi"}],
        }
    }
    _build_pending_turns_db(db_path, [("p1", json.dumps(payload))])

    rewritten = migrate_host_store_rename_concurrency_lane(db_path)
    assert rewritten == 1

    new_payload = json.loads(_read_row(db_path, "p1"))
    assert "concurrency_lane" not in new_payload["prepared_turn"]
    assert new_payload["prepared_turn"]["business_concurrency_lane"] == "sec_download"


@pytest.mark.unit
def test_migration_is_idempotent(tmp_path: Path) -> None:
    """二次执行必须是 no-op。"""

    db_path = tmp_path / "dayu_host.db"
    payload = {"host_policy": {"concurrency_lane": "write_chapter"}}
    _build_pending_turns_db(db_path, [("p1", json.dumps(payload))])

    first = migrate_host_store_rename_concurrency_lane(db_path)
    second = migrate_host_store_rename_concurrency_lane(db_path)
    assert first == 1
    assert second == 0


@pytest.mark.unit
def test_migration_skips_rows_without_old_key(tmp_path: Path) -> None:
    """没有旧 key 的行不应被动。"""

    db_path = tmp_path / "dayu_host.db"
    payload = {"host_policy": {"business_concurrency_lane": "write_chapter"}}
    _build_pending_turns_db(db_path, [("p1", json.dumps(payload))])

    rewritten = migrate_host_store_rename_concurrency_lane(db_path)
    assert rewritten == 0


@pytest.mark.unit
def test_migration_missing_db_returns_zero(tmp_path: Path) -> None:
    """数据库文件不存在时返回 0。"""

    assert migrate_host_store_rename_concurrency_lane(tmp_path / "missing.db") == 0


@pytest.mark.unit
def test_migration_missing_table_returns_zero(tmp_path: Path) -> None:
    """目标表不存在时返回 0 不抛异常。"""

    db_path = tmp_path / "dayu_host.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("CREATE TABLE other (id INTEGER)")
        conn.commit()
    finally:
        conn.close()

    assert migrate_host_store_rename_concurrency_lane(db_path) == 0


@pytest.mark.unit
def test_migration_invalid_json_is_skipped(tmp_path: Path) -> None:
    """resume_source_json 非法 JSON 的行应被跳过。"""

    db_path = tmp_path / "dayu_host.db"
    good = {"host_policy": {"concurrency_lane": "write_chapter"}}
    _build_pending_turns_db(
        db_path,
        [
            ("p_bad", "{not json but contains concurrency_lane"),
            ("p_good", json.dumps(good)),
        ],
    )

    rewritten = migrate_host_store_rename_concurrency_lane(db_path)
    assert rewritten == 1

    bad_raw = _read_row(db_path, "p_bad")
    assert bad_raw.startswith("{not json")

    good_payload = json.loads(_read_row(db_path, "p_good"))
    assert good_payload["host_policy"]["business_concurrency_lane"] == "write_chapter"


@pytest.mark.unit
def test_migration_drops_old_key_when_new_key_already_exists(tmp_path: Path) -> None:
    """新旧 key 同时存在时丢弃旧 key，保留新 key。"""

    db_path = tmp_path / "dayu_host.db"
    payload = {
        "host_policy": {
            "concurrency_lane": "OLD",
            "business_concurrency_lane": "NEW",
        }
    }
    _build_pending_turns_db(db_path, [("p1", json.dumps(payload))])

    rewritten = migrate_host_store_rename_concurrency_lane(db_path)
    assert rewritten == 1

    new_payload = json.loads(_read_row(db_path, "p1"))
    assert new_payload["host_policy"].get("concurrency_lane") is None
    assert new_payload["host_policy"]["business_concurrency_lane"] == "NEW"
