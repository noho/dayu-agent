"""Streamlit 主功能区页面。

根据是否选中自选股，展示系统介绍或功能 Tab 页面。
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dayu.contracts.session import SessionSource
from dayu.services.chat_service import ChatService
from dayu.services.fins_service import FinsService
from dayu.services.protocols import (
    ChatServiceProtocol,
    FinsServiceProtocol,
    WriteServiceProtocol,
)
from dayu.services.startup_preparation import PreparedHostRuntimeDependencies
from dayu.services.write_service import WriteService
from dayu.web.streamlit.components.watchlist import WatchlistItem
from dayu.web.streamlit.pages.chat_tab import render_chat_tab
from dayu.web.streamlit.pages.filing_tab import render_filing_tab
from dayu.web.streamlit.pages.report_tab import render_report_tab


def render_welcome_page() -> None:
    """渲染欢迎页面（未选中自选股时展示）。"""

    st.title("大禹 Agent")
    st.markdown("### 每个投资者的助理分析师")
    st.markdown("---")

    st.markdown("""
    #### 系统介绍

    **大愚 Agent** 是一个面向买方财报分析场景的 Agent 系统，它让AI读财报的方式从丢给它整份财报"大海捞针"变成"按图索骥"，让数据有**置信度**，让投资结论、投资报告可审计、可追踪。
    """)

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


def _resolve_fins_service(prepared_deps: PreparedHostRuntimeDependencies) -> FinsServiceProtocol:
    """从 prepared_deps 获取或构造财报服务，缓存于 session_state。"""
    if "fins_service" not in st.session_state:
        st.session_state["fins_service"] = FinsService(
            host=prepared_deps.host,
            fins_runtime=prepared_deps.fins_runtime,
            session_source=SessionSource.WEB,
        )
    return st.session_state["fins_service"]


def _resolve_chat_service(prepared_deps: PreparedHostRuntimeDependencies) -> ChatServiceProtocol:
    """从 prepared_deps 获取或构造聊天服务，缓存于 session_state。"""
    if "chat_service" not in st.session_state:
        st.session_state["chat_service"] = ChatService(
            host=prepared_deps.host,
            scene_execution_acceptance_preparer=prepared_deps.scene_execution_acceptance_preparer,
            company_name_resolver=prepared_deps.fins_runtime.get_company_name,
            session_source=SessionSource.WEB,
        )
    return st.session_state["chat_service"]


def _resolve_write_service(prepared_deps: PreparedHostRuntimeDependencies) -> WriteServiceProtocol:
    """从 prepared_deps 获取或构造写作服务，缓存于 session_state。"""
    if "write_service" not in st.session_state:
        st.session_state["write_service"] = WriteService(
            host=prepared_deps.host,
            host_governance=prepared_deps.host,
            workspace=prepared_deps.workspace,
            scene_execution_acceptance_preparer=prepared_deps.scene_execution_acceptance_preparer,
            company_name_resolver=prepared_deps.fins_runtime.get_company_name,
            company_meta_summary_resolver=prepared_deps.fins_runtime.get_company_meta_summary,
        )
    return st.session_state["write_service"]


def render_stock_detail_page(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    prepared_deps: PreparedHostRuntimeDependencies,
) -> None:
    """渲染股票详情页面（选中自选股后展示）。

    参数:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        prepared_deps: 已装配的 Host 运行时依赖，本函数负责构造各 Service 并注入 Tab。

    返回值:
        无。

    异常:
        无。
    """

    fins_service: FinsServiceProtocol = _resolve_fins_service(prepared_deps)
    chat_service: ChatServiceProtocol = _resolve_chat_service(prepared_deps)
    write_service: WriteServiceProtocol = _resolve_write_service(prepared_deps)

    tabs = st.tabs(["🧾 财报管理", "🧠 交互式分析", "📊 分析报告"])

    with tabs[0]:
        render_filing_tab(
            selected_stock=selected_stock,
            workspace_root=workspace_root,
            fins_service=fins_service,
        )

    with tabs[1]:
        render_chat_tab(
            selected_stock=selected_stock,
            chat_service=chat_service,
        )

    with tabs[2]:
        render_report_tab(
            selected_stock=selected_stock,
            workspace_root=workspace_root,
            write_service=write_service,
        )
