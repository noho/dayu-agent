"""Streamlit 交互式分析 Tab。

当前为占位实现，展示选中自选股的基本信息。
后续将接入 ChatServiceProtocol 提供多轮对话能力。
"""

from __future__ import annotations

import streamlit as st

from dayu.services.protocols import ChatServiceProtocol
from dayu.web.streamlit.components.watchlist import WatchlistItem


def render_chat_tab(
    *,
    selected_stock: WatchlistItem,
    chat_service: ChatServiceProtocol | None,
) -> None:
    """渲染交互式分析 Tab。

    参数:
        selected_stock: 当前选中的自选股。
        chat_service: 聊天服务协议实例；为 None 时交互式分析不可用。

    返回值:
        无。

    异常:
        无。
    """

    if chat_service is None:
        st.warning("交互式分析功能不可用（聊天服务未初始化）。")
        return

    st.write(f"交互式分析: {selected_stock.ticker}")
