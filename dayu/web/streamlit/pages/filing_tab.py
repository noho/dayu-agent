import streamlit as st
from dayu.web.streamlit.components.watchlist import WatchlistItem
from pathlib import Path
from dayu.services.protocols import FinsServiceProtocol


def render_filing_tab(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    fins_service: FinsServiceProtocol | None,
) -> None:
    """渲染财报管理 Tab。

    参数:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        fins_service: 财报服务实例；为 None 时部分功能不可用。
    """

    st.write(f"财报管理: {selected_stock.ticker}")
    st.write(f"工作区根目录: {workspace_root}")