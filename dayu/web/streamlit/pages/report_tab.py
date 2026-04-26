"""Streamlit 分析报告 Tab。

当前为占位实现，展示选中自选股的基本信息。
后续将接入 WriteServiceProtocol 提供分析报告生成与展示能力。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dayu.services.protocols import WriteServiceProtocol
from dayu.web.streamlit.components.watchlist import WatchlistItem


def render_report_tab(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    write_service: WriteServiceProtocol | None,
) -> None:
    """渲染分析报告 Tab。

    参数:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        write_service: 写作服务协议实例；为 None 时分析报告功能不可用。

    返回值:
        无。

    异常:
        无。
    """

    if write_service is None:
        st.warning("分析报告功能不可用（写作服务未初始化）。")
        return

    st.write(f"分析报告: {selected_stock.ticker}")
    st.write(f"工作区根目录: {workspace_root}")
