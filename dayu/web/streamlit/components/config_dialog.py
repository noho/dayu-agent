"""Streamlit 配置对话框。

提供统一的工作区初始化配置与模型切换功能。
"""

from __future__ import annotations

import json
from pathlib import Path
import streamlit as st

from dayu.startup.workspace_initializer import load_available_model_names

_INIT_ROLE_KEY = "_init_model_role"
_ROLE_NON_THINKING = "non_thinking"
_ROLE_THINKING = "thinking"
_SIDEBAR_REFRESH_REQUEST_KEY = "sidebar_needs_refresh"


def _load_scene_manifests(workspace_root: Path) -> list[dict]:
    """加载所有场景 manifest 信息。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        manifest 信息列表，每项包含场景标识、显示名称、当前模型、角色。
    """

    manifests_dir = workspace_root / "config" / "prompts" / "manifests"
    if not manifests_dir.exists():
        return []

    scenes: list[dict] = []
    for manifest_path in sorted(manifests_dir.glob("*.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue

        model_section = data.get("model")
        if not isinstance(model_section, dict):
            continue

        default_name = model_section.get("default_name")
        if not isinstance(default_name, str):
            continue

        # 从文件名推断场景显示名
        scene_name = manifest_path.stem
        display_name = data.get("name", scene_name)

        # 识别角色
        stored_role = model_section.get(_INIT_ROLE_KEY, "")
        role = _resolve_manifest_role(manifest_path.name, stored_role if isinstance(stored_role, str) else "")

        # 读取可用模型列表
        allowed_models = model_section.get("allowed_names", [])
        if not isinstance(allowed_models, list):
            allowed_models = []

        scenes.append({
            "scene_id": scene_name,
            "display_name": display_name,
            "current_model": default_name,
            "role": role,
            "manifest_path": manifest_path,
            "allowed_models": allowed_models,
        })

    return scenes


def _resolve_manifest_role(manifest_filename: str, stored_role: str) -> str | None:
    """解析 manifest 的模型角色。

    Args:
        manifest_filename: manifest 文件名。
        stored_role: 当前 manifest 中记录的 `_init_model_role`。

    Returns:
        `non_thinking` / `thinking`；无法判断时返回 `None`。
    """

    if stored_role in (_ROLE_NON_THINKING, _ROLE_THINKING):
        return stored_role
    return _resolve_role_from_package_manifest(manifest_filename)


def _resolve_role_from_package_manifest(manifest_filename: str) -> str | None:
    """从包内原始 manifest 推断角色。

    Args:
        manifest_filename: manifest 文件名。

    Returns:
        `non_thinking` / `thinking`；无法判断时返回 `None`。
    """

    from dayu.startup.config_file_resolver import resolve_package_config_path

    package_manifest_path = resolve_package_config_path() / "prompts" / "manifests" / manifest_filename
    if not package_manifest_path.exists():
        return None
    try:
        payload = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    model_section = payload.get("model")
    if not isinstance(model_section, dict):
        return None
    default_name = model_section.get("default_name")
    if not isinstance(default_name, str):
        return None
    if "thinking" in default_name:
        return _ROLE_THINKING
    return _ROLE_NON_THINKING


def _load_available_models_for_scene(workspace_root: Path, scene: dict) -> tuple[str, ...]:
    """加载场景可用的模型列表。

    优先使用 manifest 中定义的 allowed_names，否则加载全部可用模型。

    Args:
        workspace_root: 工作区根目录。
        scene: 场景信息字典。

    Returns:
        可用模型名称元组。
    """

    allowed = scene.get("allowed_models", [])
    if isinstance(allowed, list) and len(allowed) > 0:
        return tuple(str(m) for m in allowed if isinstance(m, str))

    # 回退到全部可用模型
    try:
        config_dir = (workspace_root / "config").resolve()
        return load_available_model_names(config_dir)
    except Exception:
        return ()


def _resolve_simplified_default_models(
    scenes: list[dict],
    all_models: tuple[str, ...],
) -> tuple[str, str]:
    """解析简化配置模式下的默认模型。

    Args:
        scenes: 场景配置列表。
        all_models: 全局可用模型列表。

    Returns:
        `(thinking_model, non_thinking_model)`。

    Raises:
        ValueError: `all_models` 为空时抛出。
    """

    if not all_models:
        raise ValueError("可用模型列表不能为空")

    default_thinking: str | None = None
    default_non_thinking: str | None = None
    for scene in scenes:
        role = scene.get("role")
        current_model = scene.get("current_model")
        if not isinstance(current_model, str):
            continue
        if role == _ROLE_THINKING and default_thinking is None and current_model in all_models:
            default_thinking = current_model
        if role == _ROLE_NON_THINKING and default_non_thinking is None and current_model in all_models:
            default_non_thinking = current_model

    if default_thinking is None:
        default_thinking = all_models[0]
    if default_non_thinking is None:
        default_non_thinking = all_models[0]
    return default_thinking, default_non_thinking


def _build_simplified_updated_models(
    scenes: list[dict],
    *,
    thinking_model: str,
    non_thinking_model: str,
) -> dict[str, str]:
    """根据简化配置选择生成按场景更新映射。

    Args:
        scenes: 场景配置列表。
        thinking_model: thinking 统一模型。
        non_thinking_model: 普通统一模型。

    Returns:
        仅包含发生变化场景的 `{scene_id: new_model}` 映射。

    Raises:
        无。
    """

    updated_models: dict[str, str] = {}
    for scene in scenes:
        scene_id = scene.get("scene_id")
        current_model = scene.get("current_model")
        role = scene.get("role")
        if not isinstance(scene_id, str) or not isinstance(current_model, str):
            continue
        target_model = thinking_model if role == _ROLE_THINKING else non_thinking_model
        if target_model != current_model:
            updated_models[scene_id] = target_model
    return updated_models


@st.dialog("模型配置", width="large")
def render_config_dialog(workspace_root: Path) -> None:
    """渲染配置对话框，提供初始化与模型切换功能。

    Args:
        workspace_root: 工作区根目录。
    """

    mode = st.radio(
        "配置模式",
        options=("简化配置", "按场景配置"),
        key="config_dialog_mode",
        horizontal=True,
    )

    scenes = _load_scene_manifests(workspace_root)
    if not scenes:
        st.info("未读取到场景配置，请先运行初始化配置。")
        return

    # 检查是否有全局模型列表
    try:
        config_dir = (workspace_root / "config").resolve()
        all_models = load_available_model_names(config_dir)
    except Exception:
        all_models = ()

    if not all_models:
        st.info("未读取到可用模型列表，请先完成初始化配置。")
        return

    updated_models: dict[str, str] = {}
    if mode == "按场景配置":
        st.markdown("为每个场景选择默认模型，仅显示该场景可用的模型。")
        col1, col2 = st.columns([1, 1], vertical_alignment="top")
        columns = [col1, col2]
        for idx, scene in enumerate(scenes):
            scene_id = scene["scene_id"]
            display_name = scene["display_name"]
            current_model = scene["current_model"]
            role = scene["role"] or "未指定"

            # 获取该场景可用模型
            available_models = _load_available_models_for_scene(workspace_root, scene)
            if not available_models:
                available_models = all_models

            # 确保当前模型在可用列表中
            if current_model not in available_models:
                available_models = (current_model,) + available_models

            role_label = "🧠 Thinking" if role == _ROLE_THINKING else "⚡ 标准"

            # 按两列分发
            with columns[idx % 2]:
                selected_model = st.selectbox(
                    f"{display_name} ({scene_id}) - {role_label}",
                    options=available_models,
                    index=available_models.index(current_model) if current_model in available_models else 0,
                    key=f"config_dialog_model_{scene_id}",
                )

            if selected_model != current_model:
                updated_models[scene_id] = selected_model
    else:
        st.markdown("简化模式：只配置 thinking 与普通模型，并自动映射到全部场景。")
        default_thinking, default_non_thinking = _resolve_simplified_default_models(scenes, all_models)
        selected_thinking = st.selectbox(
            "thinking 模型",
            options=all_models,
            index=all_models.index(default_thinking),
            key="config_dialog_thinking_model",
        )
        selected_non_thinking = st.selectbox(
            "普通模型",
            options=all_models,
            index=all_models.index(default_non_thinking),
            key="config_dialog_non_thinking_model",
        )
        updated_models = _build_simplified_updated_models(
            scenes,
            thinking_model=selected_thinking,
            non_thinking_model=selected_non_thinking,
        )

    # 保存按钮
    if updated_models:
        if st.button(
            f"保存模型配置 ({len(updated_models)} 个场景)",
            key="config_dialog_save_models",
            type="primary",
            use_container_width=True,
        ):
            try:
                updated_count = 0
                for scene in scenes:
                    scene_id = scene["scene_id"]
                    if scene_id not in updated_models:
                        continue

                    new_model = updated_models[scene_id]
                    manifest_path = scene["manifest_path"]

                    # 读取并更新 manifest
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and isinstance(data.get("model"), dict):
                        model_section = data["model"]
                        model_section["default_name"] = new_model

                        # 确保新模型在 allowed_names 中
                        allowed = model_section.get("allowed_names", [])
                        if isinstance(allowed, list) and new_model not in allowed:
                            allowed.append(new_model)
                            model_section["allowed_names"] = allowed

                        manifest_path.write_text(
                            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        updated_count += 1

                st.success(f"已更新 {updated_count} 个场景")
            except Exception as exception:
                st.error(f"保存模型配置失败: {exception}")
    else:
        st.button(
            "保存模型配置",
            key="config_dialog_save_models_disabled",
            type="secondary",
            use_container_width=True,
            disabled=True,
        )
