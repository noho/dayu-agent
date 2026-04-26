"""Streamlit 财报管理 Tab。

当前为占位实现，展示选中自选股的基本信息。
后续将接入 FinsServiceProtocol 提供财报下载与管理能力。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dayu.services.protocols import FinsServiceProtocol
from dayu.web.streamlit.components.watchlist import WatchlistItem


def render_filing_tab(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    fins_service: FinsServiceProtocol | None,
) -> None:
    """渲染财报管理 Tab。

    参数:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        fins_service: 财报服务协议实例；为 None 时部分功能不可用。

    返回值:
        无。

    异常:
        无。
    """

    if fins_service is None:
        st.warning("财报服务不可用（服务未初始化）。")
        return

    st.write(f"财报管理: {selected_stock.ticker}")
    st.write(f"工作区根目录: {workspace_root}")
