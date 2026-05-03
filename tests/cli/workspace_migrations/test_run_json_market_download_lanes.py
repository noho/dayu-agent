"""run.json CN/HK 下载 lane 迁移单测。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dayu.cli.workspace_migrations.run_json_market_download_lanes import (
    migrate_run_json_add_market_download_lanes,
)


@pytest.mark.unit
def test_migrate_run_json_adds_market_download_lanes_when_missing(tmp_path: Path) -> None:
    """缺少 ``cn_download`` 与 ``hk_download`` 时应补 1 并保留旧 key。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run_json = config_dir / "run.json"
    run_json.write_text(
        json.dumps({"host_config": {"lane": {"llm_api": 8, "sec_download": 1}}}),
        encoding="utf-8",
    )

    changed = migrate_run_json_add_market_download_lanes(config_dir)
    assert changed is True

    payload = json.loads(run_json.read_text(encoding="utf-8"))
    lane = payload["host_config"]["lane"]
    assert lane["cn_download"] == 1
    assert lane["hk_download"] == 1
    assert lane["llm_api"] == 8
    assert lane["sec_download"] == 1


@pytest.mark.unit
def test_migrate_run_json_preserves_existing_market_lane_values(tmp_path: Path) -> None:
    """已存在的市场下载 lane 取值必须保留，只补缺失项。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run_json = config_dir / "run.json"
    run_json.write_text(
        json.dumps({"host_config": {"lane": {"llm_api": 8, "cn_download": 2}}}),
        encoding="utf-8",
    )

    changed = migrate_run_json_add_market_download_lanes(config_dir)
    assert changed is True

    payload = json.loads(run_json.read_text(encoding="utf-8"))
    lane = payload["host_config"]["lane"]
    assert lane["cn_download"] == 2
    assert lane["hk_download"] == 1


@pytest.mark.unit
def test_migrate_run_json_is_idempotent_when_market_lanes_present(tmp_path: Path) -> None:
    """两项市场下载 lane 都存在时不改写文件。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    run_json = config_dir / "run.json"
    original = {
        "host_config": {
            "lane": {
                "llm_api": 8,
                "cn_download": 2,
                "hk_download": 3,
            }
        }
    }
    run_json.write_text(json.dumps(original), encoding="utf-8")

    changed = migrate_run_json_add_market_download_lanes(config_dir)
    assert changed is False

    payload = json.loads(run_json.read_text(encoding="utf-8"))
    assert payload["host_config"]["lane"]["cn_download"] == 2
    assert payload["host_config"]["lane"]["hk_download"] == 3


@pytest.mark.unit
def test_migrate_run_json_missing_file_returns_false(tmp_path: Path) -> None:
    """配置目录下没有 run.json 时返回 False 不抛异常。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    assert migrate_run_json_add_market_download_lanes(config_dir) is False


@pytest.mark.unit
def test_migrate_run_json_invalid_json_raises(tmp_path: Path) -> None:
    """解析失败时显式抛 ``json.JSONDecodeError``，不吞错返回 False。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "run.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        migrate_run_json_add_market_download_lanes(config_dir)


@pytest.mark.unit
def test_migrate_run_json_without_lane_returns_false(tmp_path: Path) -> None:
    """缺少 ``host_config.lane`` 时不擅自创建，安静返回 False。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "run.json").write_text(
        json.dumps({"host_config": {"other": 1}}),
        encoding="utf-8",
    )
    assert migrate_run_json_add_market_download_lanes(config_dir) is False
