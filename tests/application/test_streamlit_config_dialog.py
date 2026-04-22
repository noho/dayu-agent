"""Streamlit 配置对话框测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dayu.web.streamlit.components import config_dialog


def test_load_scene_manifests_reads_manifest_data(tmp_path: Path) -> None:
    """应正确读取场景 manifest 信息。"""

    manifests_dir = tmp_path / "config" / "prompts" / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # 创建测试 manifest
    (manifests_dir / "write.json").write_text(
        json.dumps(
            {
                "name": "财报撰写",
                "model": {
                    "default_name": "gpt-4o",
                    "_init_model_role": "non_thinking",
                    "allowed_names": ["gpt-4o", "claude-3-5-sonnet"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (manifests_dir / "chat.json").write_text(
        json.dumps(
            {
                "name": "对话分析",
                "model": {
                    "default_name": "claude-3-5-sonnet-thinking",
                    "_init_model_role": "thinking",
                    "allowed_names": ["claude-3-5-sonnet-thinking", "o1-preview"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    scenes = config_dialog._load_scene_manifests(tmp_path)

    assert len(scenes) == 2

    scene_ids = {s["scene_id"] for s in scenes}
    assert scene_ids == {"write", "chat"}

    # 验证场景详情
    write_scene = next(s for s in scenes if s["scene_id"] == "write")
    assert write_scene["display_name"] == "财报撰写"
    assert write_scene["current_model"] == "gpt-4o"
    assert write_scene["role"] == "non_thinking"
    assert write_scene["allowed_models"] == ["gpt-4o", "claude-3-5-sonnet"]

    chat_scene = next(s for s in scenes if s["scene_id"] == "chat")
    assert chat_scene["display_name"] == "对话分析"
    assert chat_scene["current_model"] == "claude-3-5-sonnet-thinking"
    assert chat_scene["role"] == "thinking"


def test_load_scene_manifests_returns_empty_when_dir_missing(tmp_path: Path) -> None:
    """manifest 目录不存在时应返回空列表。"""

    scenes = config_dialog._load_scene_manifests(tmp_path)
    assert scenes == []


def test_load_available_models_for_scene_uses_allowed_names(tmp_path: Path) -> None:
    """应优先使用 manifest 定义的 allowed_names。"""

    scene = {
        "allowed_models": ["model-a", "model-b"],
    }
    models = config_dialog._load_available_models_for_scene(tmp_path, scene)
    assert models == ("model-a", "model-b")


def test_load_available_models_for_scene_fallback_to_all_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """allowed_names 为空时应回退到全部可用模型。"""

    # 创建配置目录结构
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    scene = {"allowed_models": []}

    monkeypatch.setattr(
        config_dialog,
        "load_available_model_names",
        lambda cfg: ("model-x", "model-y"),
    )

    models = config_dialog._load_available_models_for_scene(tmp_path, scene)
    assert models == ("model-x", "model-y")


def test_resolve_manifest_role_uses_stored_role() -> None:
    """应优先使用 manifest 中存储的角色标记。"""

    assert config_dialog._resolve_manifest_role("test.json", "non_thinking") == "non_thinking"
    assert config_dialog._resolve_manifest_role("test.json", "thinking") == "thinking"


def test_is_workspace_initialized_returns_false_when_missing_init_dirs(tmp_path: Path) -> None:
    """缺少 config/assets 任一目录时，应判定为未初始化。"""

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    assert config_dialog._is_workspace_initialized(tmp_path) is False


def test_is_workspace_initialized_returns_true_when_init_dirs_ready(tmp_path: Path) -> None:
    """config 与 assets 都存在时，应判定为已初始化。"""

    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "assets").mkdir(parents=True, exist_ok=True)
    assert config_dialog._is_workspace_initialized(tmp_path) is True


def test_resolve_simplified_default_models_prefers_role_matched_current_model() -> None:
    """简化模式默认值应优先使用角色匹配的当前模型。"""

    scenes = [
        {"scene_id": "a", "role": "thinking", "current_model": "model-thinking"},
        {"scene_id": "b", "role": "non_thinking", "current_model": "model-normal"},
    ]
    thinking, non_thinking = config_dialog._resolve_simplified_default_models(
        scenes,
        ("model-thinking", "model-normal", "model-other"),
    )
    assert thinking == "model-thinking"
    assert non_thinking == "model-normal"


def test_build_simplified_updated_models_maps_all_scenes_by_role() -> None:
    """简化模式应按角色映射并仅返回有变更的场景。"""

    scenes = [
        {"scene_id": "s1", "role": "thinking", "current_model": "old-thinking"},
        {"scene_id": "s2", "role": "non_thinking", "current_model": "old-normal"},
        {"scene_id": "s3", "role": None, "current_model": "old-normal"},
    ]
    updated = config_dialog._build_simplified_updated_models(
        scenes,
        thinking_model="new-thinking",
        non_thinking_model="old-normal",
    )
    assert updated == {"s1": "new-thinking"}


def test_run_config_initialization_with_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """开启 reset 时应先清理后初始化。"""

    reset_calls: list[Path] = []
    init_calls: list[tuple[Path, bool]] = []
    removed_targets = (tmp_path / "config",)

    from dayu.startup.workspace_initializer import WorkspaceInitializationResult, WorkspaceMigrationResult

    expected_result = WorkspaceInitializationResult(
        config_dir=(tmp_path / "config").resolve(),
        assets_dir=(tmp_path / "assets").resolve(),
        migration_result=WorkspaceMigrationResult(
            run_json_write_chapter_added=False,
            host_store_lane_rows_rewritten=0,
        ),
    )

    monkeypatch.setattr(
        config_dialog,
        "reset_workspace_init_targets",
        lambda workspace_root: reset_calls.append(workspace_root) or removed_targets,
    )
    monkeypatch.setattr(
        config_dialog,
        "initialize_workspace_configuration",
        lambda workspace_root, *, overwrite: init_calls.append((workspace_root, overwrite)) or expected_result,
    )

    result, removed = config_dialog.run_config_initialization(
        tmp_path,
        overwrite=True,
        reset=True,
    )

    assert result == expected_result
    assert removed == removed_targets
    assert reset_calls == [tmp_path]
    assert init_calls == [(tmp_path, True)]


def test_run_config_initialization_without_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不开启 reset 时直接初始化。"""

    init_calls: list[tuple[Path, bool]] = []

    from dayu.startup.workspace_initializer import WorkspaceInitializationResult, WorkspaceMigrationResult

    expected_result = WorkspaceInitializationResult(
        config_dir=(tmp_path / "config").resolve(),
        assets_dir=(tmp_path / "assets").resolve(),
        migration_result=WorkspaceMigrationResult(
            run_json_write_chapter_added=False,
            host_store_lane_rows_rewritten=0,
        ),
    )

    monkeypatch.setattr(
        config_dialog,
        "reset_workspace_init_targets",
        lambda workspace_root: (),  # 返回空元组
    )
    monkeypatch.setattr(
        config_dialog,
        "initialize_workspace_configuration",
        lambda workspace_root, *, overwrite: init_calls.append((workspace_root, overwrite)) or expected_result,
    )

    result, removed = config_dialog.run_config_initialization(
        tmp_path,
        overwrite=False,
        reset=False,
    )

    assert result == expected_result
    assert removed == ()
    assert init_calls == [(tmp_path, False)]
