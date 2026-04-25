"""Streamlit 侧边栏组件。

提供自选股列表展示和选择功能，并展示当前工作区目录。
使用本地JSON文件存储（workspace/.dayu/streamlit/watchlist.json），刷新不丢失。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st

from dayu.web.streamlit.components.watchlist import WatchlistItem, load_watchlist_items, render_watchlist_manager

_INIT_ROLE_KEY = "_init_model_role"
_ROLE_NON_THINKING = "non_thinking"
_ROLE_THINKING = "thinking"
_CONFIG_NEEDS_REFRESH_KEY = "config_needs_refresh"

def render_sidebar(
    workspace_root: Path,
    on_select_callback: Callable[[WatchlistItem], None] | None = None,
) -> WatchlistItem | None:
    """渲染侧边栏，展示自选股列表并处理选择。

    参数:
        workspace_root: 工作区根目录，用于读取自选股存储文件。
        on_select_callback: 选中自选股后的回调函数，接收 WatchlistItem 参数。

    返回值:
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
        st.session_state["selected_ticker"] = None

    # 如果刷新了数据，检查当前选中的股票是否还在列表中，不在则清除选中状态
    selected_ticker = st.session_state.get("selected_ticker")
    if needs_refresh:
        if selected_ticker is not None and not any(item.ticker == selected_ticker for item in watchlist):
            st.session_state["selected_ticker"] = None

    # 自选股标题行：左侧标题，右侧管理按钮（icon 按钮，更小）
    col1, col2 = st.sidebar.columns([5, 1], vertical_alignment="center")
    with col1:
        st.markdown("**❤️ 自选股**")
   
    with col2:
        if st.button("", key="manage_watchlist_btn", icon=":material/list_alt_add:", type="tertiary", help="管理自选股"):
            # 调用对话框函数（装饰器会自动处理）
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
