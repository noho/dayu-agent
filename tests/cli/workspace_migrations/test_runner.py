"""runner 集成层面测试。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from dayu.cli.workspace_migrations import apply_all_workspace_migrations
from dayu.workspace_paths import build_host_store_default_path


@pytest.mark.unit
def test_apply_all_migrates_both_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """runner 同时完成 run.json 与 Host SQLite 两项迁移。"""

    base_dir = tmp_path
    config_dir = base_dir / "config"
    config_dir.mkdir()
    (config_dir / "run.json").write_text(
        json.dumps({"host_config": {"lane": {"llm_api": 8}}}),
        encoding="utf-8",
    )

    db_path = build_host_store_default_path(base_dir)
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
        conn.execute(
            "INSERT INTO pending_conversation_turns VALUES (?, ?)",
            (
                "p1",
                json.dumps(
                    {
                        "host_policy": {"concurrency_lane": "write_chapter"},
                        "agent_create_args": {"max_output_tokens": 1024},
                    }
                ),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    apply_all_workspace_migrations(base_dir=base_dir, config_dir=config_dir)

    # run.json
    payload = json.loads((config_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["host_config"]["lane"]["write_chapter"] == 5
    assert payload["host_config"]["lane"]["cn_download"] == 1
    assert payload["host_config"]["lane"]["hk_download"] == 1

    # db
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT resume_source_json FROM pending_conversation_turns WHERE pending_turn_id=?",
            ("p1",),
        ).fetchone()
    finally:
        conn.close()
    migrated = json.loads(str(row[0]))
    assert migrated["host_policy"]["business_concurrency_lane"] == "write_chapter"
    assert "max_output_tokens" not in migrated["agent_create_args"]

    captured = capsys.readouterr()
    assert "run.json" in captured.out
    assert "cn_download" in captured.out
    assert "business_concurrency_lane" in captured.out
    assert "max_output_tokens" in captured.out


@pytest.mark.unit
def test_apply_all_is_silent_when_nothing_to_do(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """无旧结构时 runner 不应打印迁移消息。"""

    base_dir = tmp_path
    config_dir = base_dir / "config"
    config_dir.mkdir()
    (config_dir / "run.json").write_text(
        json.dumps(
            {
                "host_config": {
                    "lane": {
                        "llm_api": 8,
                        "write_chapter": 5,
                        "cn_download": 1,
                        "hk_download": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    apply_all_workspace_migrations(base_dir=base_dir, config_dir=config_dir)
    out = capsys.readouterr().out
    assert "工作区迁移" not in out
