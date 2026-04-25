import streamlit as st
from dayu.web.streamlit.components.watchlist import WatchlistItem
from pathlib import Path
from dayu.services.protocols import WriteServiceProtocol

def render_report_tab(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    write_service: WriteServiceProtocol | None,
) -> None:
    """渲染分析报告 Tab。

    根据报告存在性和任务运行状态，展示三种不同UI：
    1. 无报告：引导用户启动分析任务
    2. 有报告：展示详细分析报告

    参数:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        write_service: 写作服务实例；为 None 时部分功能不可用。
    """

    st.write(f"分析报告: {selected_stock.ticker}")
    st.write(f"工作区根目录: {workspace_root}")