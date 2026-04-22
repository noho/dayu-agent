"""Streamlit 侧边栏组件。

提供自选股列表展示和选择功能，并展示当前工作区目录。
使用本地JSON文件存储（workspace/.dayu/streamlit/watchlist.json），刷新不丢失。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import streamlit as st
from dayu.startup.workspace_initializer import (
    WorkspaceInitializationResult,
    initialize_workspace_configuration,
    load_available_model_names,
    reset_workspace_init_targets,
    update_manifest_default_models,
)

_WORKSPACE_SETUP_FEEDBACK_KEY = "workspace_setup_feedback"
_RUNTIME_RELOAD_FLAG_KEY = "streamlit_needs_reinitialize"
_INIT_ROLE_KEY = "_init_model_role"
_ROLE_NON_THINKING = "non_thinking"
_ROLE_THINKING = "thinking"


@dataclass(frozen=True)
class WatchlistItem:
    """自选股条目（Streamlit内部使用）。

    Attributes:
        ticker: 股票代码，如 AAPL。
        company_name: 公司名称，如 苹果。
        created_at: 创建时间 ISO8601 格式。
        updated_at: 更新时间 ISO8601 格式。
    """

    ticker: str
    company_name: str
    created_at: str
    updated_at: str


def _watchlist_storage_path(workspace_root: Path) -> Path:
    """返回自选股持久化文件路径。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        自选股 JSON 文件绝对路径。

    Raises:
        无。
    """

    return workspace_root / ".dayu" / "streamlit" / "watchlist.json"


def load_watchlist_items(workspace_root: Path) -> list[WatchlistItem]:
    """从本地 JSON 文件加载自选股列表。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        自选股条目列表；文件不存在或解析失败时返回空列表。

    Raises:
        无：解析失败时返回空列表，不向调用方抛出。
    """

    storage_path = _watchlist_storage_path(workspace_root)
    if not storage_path.exists():
        return []

    try:
        with open(storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items: list[WatchlistItem] = []
        for item_data in data.get("items", []):
            items.append(
                WatchlistItem(
                    ticker=str(item_data["ticker"]),
                    company_name=str(item_data["company_name"]),
                    created_at=str(item_data["created_at"]),
                    updated_at=str(item_data["updated_at"]),
                )
            )
        return items
    except Exception:
        return []


def save_watchlist_items(workspace_root: Path, items: list[WatchlistItem]) -> None:
    """将自选股列表写入本地 JSON 文件。

    Args:
        workspace_root: 工作区根目录。
        items: 要持久化的条目列表。

    Raises:
        OSError: 无法创建目录或写入文件时抛出。
    """

    storage_path = _watchlist_storage_path(workspace_root)
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "items": [
            {
                "ticker": item.ticker,
                "company_name": item.company_name,
                "created_at": item.created_at,
                "updated_at": item.updated_at,
            }
            for item in items
        ],
        "version": "1.0",
    }
    with open(storage_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_sidebar_workspace_initialization(
    workspace_root: Path,
    *,
    overwrite: bool,
    reset: bool,
) -> tuple[WorkspaceInitializationResult, tuple[Path, ...]]:
    """执行侧栏触发的工作区初始化。

    Args:
        workspace_root: 工作区根目录。
        overwrite: 是否覆盖已有 config/assets。
        reset: 是否先清理 `.dayu` / `config` / `assets`。

    Returns:
        二元组 `(初始化结果, 实际被删除的路径元组)`。

    Raises:
        OSError: 删除或复制目录失败时抛出。
    """

    removed_targets: tuple[Path, ...] = ()
    if reset:
        removed_targets = reset_workspace_init_targets(workspace_root)
    initialization_result = initialize_workspace_configuration(
        workspace_root,
        overwrite=overwrite,
    )
    return initialization_result, removed_targets


def persist_sidebar_default_models(
    workspace_root: Path,
    *,
    non_thinking_model: str,
    thinking_model: str,
) -> int:
    """持久化侧栏选择的默认模型。

    Args:
        workspace_root: 工作区根目录。
        non_thinking_model: non-thinking 默认模型名。
        thinking_model: thinking 默认模型名。

    Returns:
        被更新的 manifest 文件数量。

    Raises:
        FileNotFoundError: 配置目录不存在时抛出。
        OSError: manifest 读写失败时抛出。
        json.JSONDecodeError: manifest JSON 非法时抛出。
    """

    config_dir = (workspace_root / "config").resolve()
    if not config_dir.exists():
        raise FileNotFoundError(f"配置目录不存在: {config_dir}")
    return update_manifest_default_models(
        config_dir,
        non_thinking_model=non_thinking_model,
        thinking_model=thinking_model,
    )


def load_sidebar_model_options(workspace_root: Path) -> tuple[str, ...]:
    """读取侧栏模型切换可选项。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        可选模型名元组。

    Raises:
        FileNotFoundError: 配置目录不存在时抛出。
        OSError: 模型配置文件读取失败时抛出。
    """

    config_dir = (workspace_root / "config").resolve()
    if not config_dir.exists():
        raise FileNotFoundError(f"配置目录不存在: {config_dir}")
    return load_available_model_names(config_dir)


def load_sidebar_selected_models(workspace_root: Path) -> tuple[str | None, str | None]:
    """读取当前已生效的默认模型选择。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        `(non_thinking_model, thinking_model)`；未识别时对应项为 `None`。

    Raises:
        无：读取失败时返回 `(None, None)`。
    """

    manifests_dir = (workspace_root / "config" / "prompts" / "manifests").resolve()
    if not manifests_dir.exists():
        return None, None
    non_thinking_model: str | None = None
    thinking_model: str | None = None
    for manifest_path in sorted(manifests_dir.glob("*.json")):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        model_section = payload.get("model")
        if not isinstance(model_section, dict):
            continue
        default_name = model_section.get("default_name")
        if not isinstance(default_name, str) or not default_name.strip():
            continue
        role = model_section.get(_INIT_ROLE_KEY, "")
        if role == _ROLE_NON_THINKING and non_thinking_model is None:
            non_thinking_model = default_name
            continue
        if role == _ROLE_THINKING and thinking_model is None:
            thinking_model = default_name
            continue
        if "thinking" in default_name and thinking_model is None:
            thinking_model = default_name
            continue
        if non_thinking_model is None:
            non_thinking_model = default_name
    return non_thinking_model, thinking_model


def _set_workspace_setup_feedback(message: str, *, is_error: bool) -> None:
    """写入侧栏操作反馈消息。

    Args:
        message: 要展示的消息内容。
        is_error: 是否为错误消息。

    Returns:
        无。

    Raises:
        无。
    """

    st.session_state[_WORKSPACE_SETUP_FEEDBACK_KEY] = {
        "message": message,
        "is_error": is_error,
    }


def set_sidebar_workspace_setup_feedback(message: str, *, is_error: bool) -> None:
    """对外暴露侧栏反馈写入能力。

    Args:
        message: 反馈消息。
        is_error: 是否为错误消息。

    Returns:
        无。

    Raises:
        无。
    """

    _set_workspace_setup_feedback(message, is_error=is_error)


def _render_workspace_setup_section(workspace_root: Path) -> None:
    """渲染侧栏工作区配置入口，仅显示配置按钮。

    点击配置按钮弹出对话框，在对话框内完成初始化配置和模型切换。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        无。

    Raises:
        无。
    """

    feedback = st.session_state.pop(_WORKSPACE_SETUP_FEEDBACK_KEY, None)
    if isinstance(feedback, dict):
        message = feedback.get("message")
        if isinstance(message, str) and message:
            if bool(feedback.get("is_error", False)):
                st.sidebar.error(message)
            else:
                st.sidebar.success(message)

    if st.sidebar.button(
        "⚙️ 模型配置",
        key="sidebar_config_btn",
        type="secondary",
        use_container_width=True,
        help="配置初始化和模型设置",
    ):
        from dayu.web.streamlit.components.config_dialog import render_config_dialog

        render_config_dialog(workspace_root)


def render_sidebar(
    workspace_root: Path,
    on_select_callback: Callable[[WatchlistItem], None] | None = None,
) -> WatchlistItem | None:
    """渲染侧边栏，展示自选股列表并处理选择。

    Args:
        workspace_root: 工作区根目录，用于读取自选股存储文件。
        on_select_callback: 选中自选股后的回调函数，接收 WatchlistItem 参数。

    Returns:
        当前选中的自选股条目，未选中时返回 None。
    """

    st.sidebar.title("大禹 Agent")

    # 工作区信息展示（优化样式）
    workspace_resolved = workspace_root.resolve()

    # 使用 caption 样式展示工作区路径
    st.sidebar.markdown(f"**📁 工作区**  \n`{workspace_resolved}`")

    _render_workspace_setup_section(workspace_root=workspace_root)
    st.sidebar.markdown("---")

    # 选中按钮使用 primary 类型，通过 CSS 改为仅边框高亮
    st.sidebar.markdown(
        """
        <style>
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
            background-color: transparent;
            color: inherit;
            border: 1px solid #ff4b4b;
            box-shadow: 0 0 0 1px rgba(255, 75, 75, 0.25);
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover {
            background-color: rgba(255, 75, 75, 0.06);
            color: inherit;
            border: 1px solid #ff4b4b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 检查是否需要刷新数据（对话框保存后设置）
    refresh_key = "watchlist_needs_refresh"
    needs_refresh = st.session_state.pop(refresh_key, False)

    # 获取自选股列表（从文件读取）
    try:
        watchlist = load_watchlist_items(workspace_root)
    except Exception as e:
        st.sidebar.error(f"加载自选股失败: {e}")
        watchlist = []

    # 初始化选中状态
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = None

    # 如果刷新了数据，检查当前选中的股票是否还在列表中，不在则清除选中状态
    selected_ticker = st.session_state.get("selected_ticker")
    if needs_refresh and isinstance(selected_ticker, str) and selected_ticker:
        current_ticker = selected_ticker
        if not any(item.ticker == current_ticker for item in watchlist):
            st.session_state["selected_ticker"] = None

    # 自选股标题行：左侧标题，右侧管理按钮（icon 按钮，更小）
    col1, col2 = st.sidebar.columns([5, 1], vertical_alignment="center")
    with col1:
        st.markdown("**❤️ 自选股**")
   
    with col2:
        if st.button("", key="manage_watchlist_btn", icon=":material/list_alt_add:", type="tertiary", help="管理自选股"):
            # 调用对话框函数（装饰器会自动处理）
            from dayu.web.streamlit.components.watchlist_dialog import render_watchlist_manager
            render_watchlist_manager(workspace_root)

    # 展示自选股列表
    selected_item = None
    for item in watchlist:
        display_name = f"{item.company_name} ({item.ticker})"
        current_selected_ticker = st.session_state.get("selected_ticker")
        is_selected = isinstance(current_selected_ticker, str) and current_selected_ticker == item.ticker

        # 使用按钮展示每个自选股
        button_type = "primary" if is_selected else "secondary"
        if st.sidebar.button(display_name, key=f"stock_{item.ticker}", type=button_type, width="stretch"):
            st.session_state["selected_ticker"] = item.ticker
            selected_item = item
            if on_select_callback:
                on_select_callback(item)
            st.rerun()

    if not watchlist:
        st.sidebar.info("暂无自选股，请点击管理按钮添加")

    st.sidebar.markdown("---")

    # 返回当前选中的条目
    current_selected_ticker = st.session_state.get("selected_ticker")
    if isinstance(current_selected_ticker, str) and current_selected_ticker and not selected_item:
        # 从列表中找到选中的条目
        for item in watchlist:
            if item.ticker == current_selected_ticker:
                selected_item = item
                break

    return selected_item
