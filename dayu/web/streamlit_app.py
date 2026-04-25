"""Streamlit Web 应用入口。

大禹 Agent 的 Web UI 实现，基于 Streamlit 框架。
提供自选股管理、财报下载、财务分析等功能。

自选股数据存储于 workspace/.dayu/streamlit/watchlist.json，刷新不丢失。

页面布局：
- 左侧边栏：自选股列表 + 管理按钮
- 主功能区：
  - 未选中自选股时：系统介绍 + 操作指引
  - 选中自选股后：财报管理 / 交互式分析 / 分析报告 Tab

Usage:
    streamlit run dayu/web/streamlit_app.py
    # 或使用 CLI：
    dayu-web
    # 或使用模块入口：
    python -m dayu.web
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import streamlit as st

if TYPE_CHECKING:
    from dayu.services.protocols import FinsServiceProtocol, WriteServiceProtocol
    from dayu.web.streamlit.components.watchlist import WatchlistItem
    from dayu.web.streamlit.pages.chat_tab import ChatServiceClient
from dayu.web.streamlit.bootstrap import initialize_services



def _configure_streamlit_page() -> None:
    """配置 Streamlit 页面元信息。"""

    st.set_page_config(
        page_title="大禹 Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def _ensure_session_state_initialized() -> None:
    """初始化 Streamlit 会话状态默认键。"""

    if "initialized" not in st.session_state:
        st.session_state["initialized"] = False
        st.session_state["workspace_root"] = None
        st.session_state["fins_service"] = None
        st.session_state["write_service"] = None
        st.session_state["chat_service_client"] = None
        st.session_state["host_admin_service"] = None


def main() -> None:
    """Streamlit 应用主入口。"""

    from dayu.web.streamlit.components.sidebar import render_sidebar
    from dayu.web.streamlit.pages.main_page import render_welcome_page, render_stock_detail_page

    _configure_streamlit_page()
    _ensure_session_state_initialized()

    # 初始化服务（只执行一次）
    if not bool(st.session_state["initialized"]):
        try:
            workspace_root, fins_service, write_service, chat_service_client = initialize_services()
            st.session_state["workspace_root"] = workspace_root
            st.session_state["fins_service"] = fins_service
            st.session_state["write_service"] = write_service
            st.session_state["chat_service_client"] = chat_service_client
            st.session_state["initialized"] = True
        except Exception as e:
            st.error(f"服务初始化失败: {e}")
            st.stop()

    # 从会话状态获取工作区路径和服务
    workspace_root = st.session_state["workspace_root"]
    fins_service = st.session_state["fins_service"]
    write_service = st.session_state["write_service"]
    chat_service_client = st.session_state["chat_service_client"]


    # 左侧边栏：渲染自选股列表（直接从文件读取）
    selected_stock = render_sidebar(
        workspace_root=workspace_root,
        on_select_callback=_on_stock_selected,
    )

    # 主功能区（自选股管理对话框由侧边栏「管理」按钮触发）
    if selected_stock is None:
        # 未选中自选股：展示欢迎页面
        render_welcome_page()
    elif fins_service is not None:
        # 选中自选股且服务可用：展示详情页面
        render_stock_detail_page(
            selected_stock=selected_stock,
            workspace_root=workspace_root,
            fins_service=fins_service,
            write_service=write_service,
            chat_service_client=chat_service_client,
        )
    else:
        # 选中自选股但服务不可用：提示错误
        st.error("财报服务不可用，无法展示股票详情。请检查配置后刷新页面。")


def _on_stock_selected(stock: WatchlistItem) -> None:
    """自选股选中回调。

    参数:
        stock: 被选中的自选股。
    """

    # 可以在这里添加选中后的逻辑
    pass


def run_streamlit() -> int:
    """启动 Streamlit 服务并返回进程退出码。

    参数:
        无。

    返回值:
        Streamlit 子进程退出码；正常退出通常为 `0`。

    异常:
        无。子进程启动失败时由 `subprocess.run` 抛出异常。
    """

    import subprocess

    # 获取当前文件路径
    app_path = Path(__file__).resolve()

    # 构建 streamlit 命令
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        *sys.argv[1:],  # 传递所有原始参数
    ]

    # 执行命令
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    main()
