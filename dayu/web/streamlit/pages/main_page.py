"""Streamlit 主功能区页面。

根据是否选中自选股，展示系统介绍或功能 Tab 页面。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dayu.services.protocols import FinsServiceProtocol, WriteServiceProtocol
from dayu.web.streamlit.components.watchlist import WatchlistItem
from dayu.web.streamlit.pages.chat_tab import ChatServiceClient, render_chat_tab
from dayu.web.streamlit.pages.filing_tab import render_filing_tab
from dayu.web.streamlit.pages.report_tab import render_report_tab


def render_welcome_page() -> None:
    """渲染欢迎页面（未选中自选股时展示）。"""

    # 系统标题
    st.title("大禹 Agent")
    st.markdown("### 每个投资者的助理分析师")
    st.markdown("---")

    # 系统介绍
    st.markdown("""
    #### 系统介绍

    **大愚 Agent** 是一个面向买方财报分析场景的 Agent 系统，它让AI读财报的方式从丢给它整份财报“大海捞针”变成“按图索骥”，让数据有**置信度**，让投资结论、投资报告可审计、可追踪。  
    """)
    

    # 操作指引
    st.markdown("---")
    st.markdown("""
    ### 环境配置：
    1. **模型配置**：点击左侧「配置」按钮，完成工作区初始化与模型配置。
    2. **管理自选股**：点击左侧「管理自选股」按钮，添加您关注的公司
    3. **选择模型**：在左侧模型列表中选择您想要使用的模型
    """)
    st.caption("ℹ️ 提示：请先配置模型 API Key 的环境变量，否则无法使用交互式分析和分析报告功能")

    st.markdown("---")
    st.markdown("""
    ### 操作指引：
    1. **选择股票**：在左侧自选股列表中点击任意股票
    2. **下载财报**：在「财报管理」页面点击「下载财报」按钮获取最新财报，从 SEC EDGAR 获取上市公司财报文件（10-K、10-Q、8-K 等）
    3. **交互式分析**：财报下载完成后，可以进行交互式分析，通过自然语言对话，深度分析公司财务状况与业务趋势
    4. **生成报告**：财报下载完成后，可以根据内置的定性分析模板生成完整的分析报告，报告分析时间约为 10-30 分钟
    """)
    st.info("注意：交互式分析或报告生成过程中不要关闭或刷新页面，否则应答会中断，需要重新开始对话或报告生成", icon="⚠️")


def render_stock_detail_page(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    fins_service: FinsServiceProtocol,
    write_service: WriteServiceProtocol | None = None,
    chat_service_client: ChatServiceClient | None = None,
) -> None:
    """渲染股票详情页面（选中自选股后展示）。

    参数:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        fins_service: 财报服务实例。
        write_service: 写作服务实例；为 None 时分析报告功能受限。
        chat_service_client: 聊天Service客户端；为 None 时交互式分析功能不可用。
    """
    # 功能 Tab（带图标）
    tabs = st.tabs(["🧾 财报管理", "🧠 交互式分析", "📊 分析报告"])

    # Tab 1: 财报管理
    with tabs[0]:
        render_filing_tab(
            selected_stock=selected_stock,
            workspace_root=workspace_root,
            fins_service=fins_service,
        )

    # Tab 2: 交互式分析
    with tabs[1]:
        render_chat_tab(
            selected_stock=selected_stock,
            chat_service_client=chat_service_client,
        )

    # Tab 3: 分析报告
    with tabs[2]:
        render_report_tab(
            selected_stock=selected_stock,
            workspace_root=workspace_root,
            write_service=write_service,
        )
