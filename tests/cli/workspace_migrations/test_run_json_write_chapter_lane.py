"""run.json 迁移单测。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dayu.cli.workspace_migrations.run_json_write_chapter_lane import (
    migrate_run_json_add_write_chapter_lane,
)


@pytest.mark.unit
def test_migrate_run_json_adds_write_chapter_when_missing(tmp_path: Path) -> None:
    """缺少 ``write_chapter`` 时应补 5 并返回 True。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run_json = config_dir / "run.json"
    run_json.write_text(
        json.dumps({"host_config": {"lane": {"llm_api": 8, "sec_download": 1}}}),
        encoding="utf-8",
    )

    changed = migrate_run_json_add_write_chapter_lane(config_dir)
    assert changed is True

    payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert payload["host_config"]["lane"]["write_chapter"] == 5
    # 原有 key 不应被动。
    assert payload["host_config"]["lane"]["llm_api"] == 8
    assert payload["host_config"]["lane"]["sec_download"] == 1


@pytest.mark.unit
def test_migrate_run_json_is_idempotent_when_key_present(tmp_path: Path) -> None:
    """已存在 ``write_chapter`` 时必须保留用户取值且不改写文件。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run_json = config_dir / "run.json"
    original = {"host_config": {"lane": {"llm_api": 8, "write_chapter": 2}}}
    run_json.write_text(json.dumps(original), encoding="utf-8")

    changed = migrate_run_json_add_write_chapter_lane(config_dir)
    assert changed is False

    payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert payload["host_config"]["lane"]["write_chapter"] == 2


@pytest.mark.unit
def test_migrate_run_json_missing_file_returns_false(tmp_path: Path) -> None:
    """配置目录下没有 run.json 时返回 False 不抛异常。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    assert migrate_run_json_add_write_chapter_lane(config_dir) is False


@pytest.mark.unit
def test_migrate_run_json_invalid_json_returns_false(tmp_path: Path) -> None:
    """解析失败时返回 False 不抛异常。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "run.json").write_text("{not json", encoding="utf-8")
    assert migrate_run_json_add_write_chapter_lane(config_dir) is False


@pytest.mark.unit
def test_migrate_run_json_without_host_config_returns_false(tmp_path: Path) -> None:
    """缺少 host_config 时不擅自创建，安静返回 False。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "run.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
    assert migrate_run_json_add_write_chapter_lane(config_dir) is False
