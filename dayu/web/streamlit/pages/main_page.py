"""Streamlit 主功能区页面。

根据是否选中自选股，展示系统介绍或功能 Tab 页面。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dayu.services.protocols import FinsServiceProtocol, WriteServiceProtocol
from dayu.web.streamlit.components.sidebar import WatchlistItem
from dayu.web.streamlit.pages.chat.chat_client import ChatServiceClient
from dayu.web.streamlit.pages.chat_tab import render_chat_tab
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

    - **财报下载**：自动从 SEC EDGAR 获取上市公司财报文件（10-K、10-Q、8-K 等）
    - **财报解析**：智能提取财报中的关键财务数据、风险因素、管理层讨论等
    - **交互式分析**：通过自然语言对话，深度分析公司财务状况与业务趋势
    - **分析报告生成**：基于模板自动生成标准化的投资分析报告
    """)

    st.markdown("---")

    # 操作指引
    st.markdown("""
    #### 操作指引
    0. **模型配置**：点击左侧「配置」按钮，完成工作区初始化与模型配置。
    1. **管理自选股**：点击左侧「管理自选股」按钮，添加您关注的公司
    2. **选择股票**：在左侧自选股列表中点击任意股票
    3. **下载财报**：在「财报管理」页面点击「下载财报」按钮获取最新财报
    4. **开始分析**：财报下载完成后，可进行交互式分析或生成报告

    开始您的财务分析之旅吧!
    """)

    st.markdown("---")
    st.caption("提示：本系统需要配置工作区目录以存储财报文件。首次使用请确保 workspace 目录已正确设置。")


def render_stock_detail_page(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    fins_service: FinsServiceProtocol,
    write_service: WriteServiceProtocol | None = None,
    chat_service_client: ChatServiceClient | None = None,
) -> None:
    """渲染股票详情页面（选中自选股后展示）。

    Args:
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
        render_filing_tab(selected_stock, workspace_root, fins_service)

    # Tab 2: 交互式分析
    with tabs[1]:
        render_chat_tab(
            selected_stock=selected_stock,
            service_client=chat_service_client,
        )

    # Tab 3: 分析报告
    with tabs[2]:
        render_report_tab(selected_stock, workspace_root, write_service)
