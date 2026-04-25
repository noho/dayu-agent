"""Streamlit 入口依赖装配模块。

负责工作区路径解析、Service 依赖初始化，以及本地文件服务安全启动。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from dayu.log import Log

if TYPE_CHECKING:
    from dayu.services.startup_preparation import PreparedHostRuntimeDependencies
    from dayu.services.protocols import ChatServiceProtocol, FinsServiceProtocol, WriteServiceProtocol
    from dayu.web.streamlit.pages.chat_tab import ChatServiceClient

MODULE = "WEB.STREAMLIT.BOOTSTRAP"

def initialize_services() -> tuple[Path, FinsServiceProtocol | None, WriteServiceProtocol | None, ChatServiceClient | None]:
    """初始化 Streamlit 页面所需 Service 依赖。

    参数:
        无。

    返回值:
        四元组：`(workspace_root, fins_service, write_service, chat_service_client)`。

    异常:
        无。内部异常会转换为 UI warning 并降级返回空服务。
    """

    workspace_root = resolve_workspace_root()
    Log.info(f"工作区根目录: {workspace_root}", module=MODULE)

    prepared_runtime_dependencies: PreparedHostRuntimeDependencies | None = None
    try:
        prepared_runtime_dependencies = _prepare_host_runtime_dependencies(workspace_root)
    except Exception as exception:  # noqa: BLE001
        st.warning(f"Host 运行时依赖初始化失败（功能不可用）: {exception}")

    fins_service = None
    host_admin_service = None
    if prepared_runtime_dependencies is not None:
        try:
            fins_service = _create_fins_service(prepared_runtime_dependencies)
        except Exception as exception:  # noqa: BLE001
            st.warning(f"财报服务初始化失败（部分功能不可用）: {exception}")

        from dayu.services.host_admin_service import HostAdminService

        host_admin_service = HostAdminService(host=prepared_runtime_dependencies.host)

    write_service = None
    chat_service_client = None
    if prepared_runtime_dependencies is not None:
        try:
            write_service, chat_service = _create_write_service_and_chat_service(prepared_runtime_dependencies)
            from dayu.services.reply_delivery_service import ReplyDeliveryService

            if host_admin_service is None:
                raise RuntimeError("Host 管理服务未初始化")
            reply_delivery_service = ReplyDeliveryService(host=prepared_runtime_dependencies.host)

            from dayu.web.streamlit.pages.chat_tab import create_chat_service_client
            chat_service_client = create_chat_service_client(
                chat_service=chat_service,
                host_admin_service=host_admin_service,
                reply_delivery_service=reply_delivery_service,
            )
        except Exception as exception:  # noqa: BLE001
            st.warning(f"分析报告和交互式分析功能不可用: {exception}")

    st.session_state["host_admin_service"] = host_admin_service
    return workspace_root, fins_service, write_service, chat_service_client


def resolve_workspace_root() -> Path:
    """解析 Streamlit 运行工作区目录。

    优先级：
    1. 命令行参数 `--workspace` / `-w`
    2. 当前目录下的 `workspace`

    参数:
        无。

    返回值:
        解析后的绝对路径。

    异常:
        无。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-w", type=str, help="工作区目录路径")
    parsed_args, _ = parser.parse_known_args()
    if parsed_args.workspace:
        return Path(parsed_args.workspace).resolve()

    return (Path.cwd() / "workspace").resolve()


def _prepare_host_runtime_dependencies(workspace_root: Path) -> PreparedHostRuntimeDependencies:
    """准备 Streamlit 共享 Host 运行时依赖。

    参数:
        workspace_root: 工作区根目录。

    返回值:
        Streamlit 各 Service 共享的 Host 运行时依赖。

    异常:
        Exception: 启动依赖准备失败时抛出。
    """

    from dayu.services import prepare_host_runtime_dependencies

    return prepare_host_runtime_dependencies(
        workspace_root=workspace_root,
        config_root=None,
        execution_options=None,
        runtime_label="Streamlit Host runtime",
        log_module=MODULE,
    )


def _create_fins_service(prepared_runtime_dependencies: PreparedHostRuntimeDependencies) -> FinsServiceProtocol:
    """创建财报服务实例。

    参数:
        prepared_runtime_dependencies: 共享 Host 运行时依赖。

    返回值:
        财报服务实例。

    异常:
        Exception: 初始化依赖失败时抛出。
    """

    from dayu.services import FinsService

    return FinsService(
        host=prepared_runtime_dependencies.host,
        fins_runtime=prepared_runtime_dependencies.fins_runtime,
    )


def _create_write_service_and_chat_service(
    prepared_runtime_dependencies: PreparedHostRuntimeDependencies,
) -> tuple[WriteServiceProtocol, ChatServiceProtocol]:
    """创建写作服务与聊天服务。

    参数:
        prepared_runtime_dependencies: 共享 Host 运行时依赖。

    返回值:
        二元组：`(write_service, chat_service)`。

    异常:
        Exception: 依赖准备或实例化失败时抛出。
    """

    from dayu.contracts.session import SessionSource
    from dayu.services import ChatService, WriteService

    write_service = WriteService(
        host=prepared_runtime_dependencies.host,
        host_governance=prepared_runtime_dependencies.host,
        workspace=prepared_runtime_dependencies.workspace,
        scene_execution_acceptance_preparer=prepared_runtime_dependencies.scene_execution_acceptance_preparer,
        company_name_resolver=prepared_runtime_dependencies.fins_runtime.get_company_name,
        company_meta_summary_resolver=prepared_runtime_dependencies.fins_runtime.get_company_meta_summary,
    )
    chat_service = ChatService(
        host=prepared_runtime_dependencies.host,
        scene_execution_acceptance_preparer=prepared_runtime_dependencies.scene_execution_acceptance_preparer,
        company_name_resolver=prepared_runtime_dependencies.fins_runtime.get_company_name,
        session_source=SessionSource.WEB,
    )
    return write_service, chat_service
