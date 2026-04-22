"""startup 工作区初始化公共能力测试。"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from dayu.startup import workspace_initializer


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """写入 JSON 文件。

    Args:
        path: 目标文件路径。
        payload: 需要写入的 JSON 对象。

    Returns:
        无。

    Raises:
        OSError: 文件写入失败时抛出。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_initialize_workspace_configuration_copies_and_migrates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初始化应复制 config/assets 并执行迁移。"""

    package_config_dir = tmp_path / "pkg_config"
    package_assets_dir = tmp_path / "pkg_assets"
    _write_json(
        package_config_dir / "run.json",
        {"host_config": {"lane": {"llm_api": 1}}},
    )
    _write_json(package_config_dir / "llm_models.json", {"m1": {"runner_type": "openai_compatible"}})
    (package_assets_dir / "readme.txt").parent.mkdir(parents=True, exist_ok=True)
    (package_assets_dir / "readme.txt").write_text("asset", encoding="utf-8")

    monkeypatch.setattr(workspace_initializer, "resolve_package_config_path", lambda: package_config_dir)
    monkeypatch.setattr(workspace_initializer, "resolve_package_assets_path", lambda: package_assets_dir)

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    host_store_path = workspace_initializer.build_host_store_default_path(workspace_root)
    host_store_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(host_store_path)) as connection:
        connection.execute(
            """
            CREATE TABLE pending_conversation_turns (
                pending_turn_id TEXT PRIMARY KEY,
                resume_source_json TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO pending_conversation_turns (pending_turn_id, resume_source_json)
            VALUES (?, ?)
            """,
            ("turn-1", '{"concurrency_lane": "llm_api"}'),
        )
        connection.commit()

    result = workspace_initializer.initialize_workspace_configuration(
        workspace_root,
        overwrite=False,
    )

    assert result.config_dir == (workspace_root / "config").resolve()
    assert result.assets_dir == (workspace_root / "assets").resolve()
    assert result.migration_result.run_json_write_chapter_added is True
    assert result.migration_result.host_store_lane_rows_rewritten == 1

    run_json = json.loads((workspace_root / "config" / "run.json").read_text(encoding="utf-8"))
    assert run_json["host_config"]["lane"]["write_chapter"] == 5


def test_update_manifest_default_models_uses_stored_role_and_package_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """更新默认模型应优先角色标记，缺失时回退包内 manifest。"""

    package_config_dir = tmp_path / "pkg_config"
    _write_json(
        package_config_dir / "prompts" / "manifests" / "interactive.json",
        {"model": {"default_name": "mimo-v2-pro-thinking-plan"}},
    )
    monkeypatch.setattr(workspace_initializer, "resolve_package_config_path", lambda: package_config_dir)

    config_dir = tmp_path / "workspace" / "config"
    _write_json(
        config_dir / "prompts" / "manifests" / "write.json",
        {"model": {"default_name": "old-a", "_init_model_role": "non_thinking", "allowed_names": ["old-a"]}},
    )
    _write_json(
        config_dir / "prompts" / "manifests" / "interactive.json",
        {"model": {"default_name": "old-b", "allowed_names": ["old-b"]}},
    )

    updated_count = workspace_initializer.update_manifest_default_models(
        config_dir,
        non_thinking_model="model-non-thinking",
        thinking_model="model-thinking",
    )

    assert updated_count == 2
    write_manifest = json.loads((config_dir / "prompts" / "manifests" / "write.json").read_text(encoding="utf-8"))
    interactive_manifest = json.loads(
        (config_dir / "prompts" / "manifests" / "interactive.json").read_text(encoding="utf-8")
    )
    assert write_manifest["model"]["default_name"] == "model-non-thinking"
    assert write_manifest["model"]["_init_model_role"] == "non_thinking"
    assert "model-non-thinking" in write_manifest["model"]["allowed_names"]
    assert interactive_manifest["model"]["default_name"] == "model-thinking"
    assert interactive_manifest["model"]["_init_model_role"] == "thinking"
    assert "model-thinking" in interactive_manifest["model"]["allowed_names"]


def test_load_available_model_names_returns_sorted_names(tmp_path: Path) -> None:
    """模型列表读取应过滤内部键并稳定排序。"""

    config_dir = tmp_path / "workspace" / "config"
    _write_json(
        config_dir / "llm_models.json",
        {
            "zzz": {"runner_type": "openai_compatible"},
            "aaa": {"runner_type": "openai_compatible"},
            "_meta": {"note": "ignored"},
        },
    )

    model_names = workspace_initializer.load_available_model_names(config_dir)

    assert model_names == ("aaa", "zzz")
