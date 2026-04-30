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

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from dayu.services.protocols import (
        ChatServiceProtocol,
        FinsServiceProtocol,
        WriteServiceProtocol,
    )
    from dayu.services.startup_preparation import PreparedHostRuntimeDependencies

MODULE = "WEB.STREAMLIT.APP"


def _resolve_workspace_root() -> Path:
    """解析 Streamlit 运行工作区目录。

    优先级：
    1. 环境变量 ``DAYU_WORKSPACE``
    2. 当前目录下的 ``workspace``

    使用环境变量而非命令行参数，避免与 Streamlit 自身的参数解析冲突。

    参数:
        无。

    返回值:
        解析后的绝对路径。

    异常:
        无。
    """

    env_workspace = os.environ.get("DAYU_WORKSPACE")
    if env_workspace:
        return Path(env_workspace).resolve()

    return (Path.cwd() / "workspace").resolve()


def _prepare_host_runtime() -> PreparedHostRuntimeDependencies | None:
    """准备 Streamlit 页面所需的 Host 运行时依赖。

    调用 Service 层 ``prepare_host_runtime_dependencies()`` 完成 Host 运行时装配，
    由入口 ``main()`` 统一构造各 Service 并注入页面。

    参数:
        无。

    返回值:
        ``PreparedHostRuntimeDependencies`` 实例。

    异常:
        内部异常直接向上抛出，由调用方（``main()``）统一捕获并转为 UI warning。
    """

    workspace_root = _resolve_workspace_root()
    if not workspace_root.exists():
        workspace_root.mkdir(parents=True, exist_ok=True)

    from dayu.log import Log
    Log.info(f"工作区根目录: {workspace_root}", module=MODULE)

    from dayu.services.startup_preparation import prepare_host_runtime_dependencies

    return prepare_host_runtime_dependencies(
        workspace_root=workspace_root,
        config_root=None,
        execution_options=None,
        runtime_label="Web Host runtime",
        log_module=MODULE,
    )


def _resolve_services(
    prepared_deps: PreparedHostRuntimeDependencies,
) -> tuple[FinsServiceProtocol, ChatServiceProtocol, WriteServiceProtocol]:
    """从 prepared_deps 构造三个 Service 并缓存在 session_state。

    参数:
        prepared_deps: 已装配的 Host 运行时依赖。

    返回值:
        ``(fins_service, chat_service, write_service)`` 三元组。
    """

    from dayu.contracts.session import SessionSource
    from dayu.services.chat_service import ChatService
    from dayu.services.fins_service import FinsService
    from dayu.services.write_service import WriteService

    if "fins_service" not in st.session_state:
        st.session_state["fins_service"] = FinsService(
            host=prepared_deps.host,
            fins_runtime=prepared_deps.fins_runtime,
            session_source=SessionSource.WEB,
        )

    if "chat_service" not in st.session_state:
        st.session_state["chat_service"] = ChatService(
            host=prepared_deps.host,
            scene_execution_acceptance_preparer=prepared_deps.scene_execution_acceptance_preparer,
            default_scene_name="interactive",
            company_name_resolver=prepared_deps.fins_runtime.get_company_name,
            session_source=SessionSource.WEB,
        )

    if "write_service" not in st.session_state:
        st.session_state["write_service"] = WriteService(
            host=prepared_deps.host,
            host_governance=prepared_deps.host,
            workspace=prepared_deps.workspace,
            scene_execution_acceptance_preparer=prepared_deps.scene_execution_acceptance_preparer,
            company_name_resolver=prepared_deps.fins_runtime.get_company_name,
            company_meta_summary_resolver=prepared_deps.fins_runtime.get_company_meta_summary,
        )

    return (
        st.session_state["fins_service"],
        st.session_state["chat_service"],
        st.session_state["write_service"],
    )


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
        st.session_state["prepared_deps"] = None


def main() -> None:
    """Streamlit 应用主入口。"""

    from dayu.web.streamlit.components.sidebar import render_sidebar
    from dayu.web.streamlit.pages.main_page import render_welcome_page, render_stock_detail_page

    _configure_streamlit_page()
    _ensure_session_state_initialized()

    prepared_deps: PreparedHostRuntimeDependencies | None = st.session_state["prepared_deps"]

    # 准备 Host 运行时依赖（只执行一次）
    if not bool(st.session_state["initialized"]):
        try:
            prepared_deps = _prepare_host_runtime()
            st.session_state["prepared_deps"] = prepared_deps
            st.session_state["initialized"] = True
        except Exception as exc:
            st.error(f"服务初始化失败: {exc}")
            st.stop()

    workspace_root = _resolve_workspace_root()

    # 左侧边栏：渲染自选股列表
    selected_stock = render_sidebar(workspace_root=workspace_root)

    # 主功能区
    if selected_stock is None:
        render_welcome_page()
    elif prepared_deps is not None:
        fins_service, chat_service, write_service = _resolve_services(prepared_deps)
        render_stock_detail_page(
            selected_stock=selected_stock,
            workspace_root=workspace_root,
            fins_service=fins_service,
            chat_service=chat_service,
            write_service=write_service,
        )
    else:
        st.error("财报服务不可用，无法展示股票详情。请检查配置后刷新页面。")


def run_streamlit() -> int:
    """启动 Streamlit 服务并返回进程退出码。

    参数:
        无。

    返回值:
        Streamlit 子进程退出码；正常退出通常为 ``0``。

    异常:
        无。子进程启动失败时由 ``subprocess.run`` 抛出异常。
    """

    import subprocess

    app_path = Path(__file__).resolve()

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        *sys.argv[1:],
    ]

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


if __name__ == "__main__":
    main()
