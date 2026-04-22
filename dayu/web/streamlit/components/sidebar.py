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
        st.session_state.selected_ticker = None

    # 如果刷新了数据，检查当前选中的股票是否还在列表中，不在则清除选中状态
    if needs_refresh and st.session_state.selected_ticker:
        current_ticker = st.session_state.selected_ticker
        if not any(item.ticker == current_ticker for item in watchlist):
            st.session_state.selected_ticker = None

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
        is_selected = st.session_state.selected_ticker == item.ticker

        # 使用按钮展示每个自选股
        button_type = "primary" if is_selected else "secondary"
        if st.sidebar.button(display_name, key=f"stock_{item.ticker}", type=button_type, width="stretch"):
            st.session_state.selected_ticker = item.ticker
            selected_item = item
            if on_select_callback:
                on_select_callback(item)
            st.rerun()

    if not watchlist:
        st.sidebar.info("暂无自选股，请点击管理按钮添加")

    st.sidebar.markdown("---")

    # 返回当前选中的条目
    if st.session_state.selected_ticker and not selected_item:
        # 从列表中找到选中的条目
        for item in watchlist:
            if item.ticker == st.session_state.selected_ticker:
                selected_item = item
                break

    return selected_item
