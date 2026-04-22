"""Streamlit 入口依赖装配模块。

负责工作区路径解析、Service 依赖初始化，以及本地文件服务安全启动。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as streamlit_module
from typing import Protocol, cast

if TYPE_CHECKING:
    from dayu.host.host import Host
    from dayu.services.protocols import ChatServiceProtocol, FinsServiceProtocol, WriteServiceProtocol
    from dayu.web.streamlit.file_server import FileServerHandle
    from dayu.web.streamlit.pages.chat.chat_client import ChatServiceClient


class _StreamlitBootstrapProtocol(Protocol):
    """bootstrap 所需的最小 Streamlit 协议。"""

    session_state: dict[str, object]

    def warning(self, body: str) -> None:
        """展示 warning。"""
        ...


st = cast(_StreamlitBootstrapProtocol, streamlit_module)


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
    fins_service = None
    try:
        fins_service = _create_fins_service(workspace_root)
    except Exception as exception:  # noqa: BLE001
        st.warning(f"财报服务初始化失败（部分功能不可用）: {exception}")

    write_service = None
    chat_service_client = None
    host_admin_service = None
    try:
        write_service, chat_service, host = _create_write_service_and_chat_service(workspace_root)

        from dayu.services.host_admin_service import HostAdminService
        from dayu.services.reply_delivery_service import ReplyDeliveryService

        host_admin_service = HostAdminService(host=host)
        reply_delivery_service = ReplyDeliveryService(host=host)

        from dayu.web.streamlit.pages.chat.chat_client import create_chat_service_client

        chat_service_client = create_chat_service_client(
            chat_service=chat_service,
            host_admin_service=host_admin_service,
            reply_delivery_service=reply_delivery_service,
        )
    except Exception as exception:  # noqa: BLE001
        st.warning(f"写作服务和聊天服务初始化失败（分析报告和交互式分析功能不可用）: {exception}")

    st.session_state["host_admin_service"] = host_admin_service
    return workspace_root, fins_service, write_service, chat_service_client


def resolve_workspace_root() -> Path:
    """解析 Streamlit 运行工作区目录。

    优先级：
    1. 命令行参数 `--workspace` / `-w`
    2. 环境变量 `DAYU_WORKSPACE`
    3. 当前目录下的 `workspace`

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

    env_workspace = os.environ.get("DAYU_WORKSPACE")
    if env_workspace:
        return Path(env_workspace).resolve()

    return (Path.cwd() / "workspace").resolve()


def start_file_server_safely(workspace_root: Path) -> FileServerHandle | None:
    """安全启动本地文件服务，失败时降级为 `None`。

    参数:
        workspace_root: 工作区根目录。

    返回值:
        启动成功返回文件服务句柄；失败返回 `None`。

    异常:
        无。所有异常都转为 warning。
    """

    from dayu.web.streamlit.file_server import start_file_server

    try:
        return start_file_server(workspace_root)
    except Exception as exception:  # noqa: BLE001
        st.warning(f"本地文件服务启动失败，将无法在新标签打开财报文件: {exception}")
        return None


def _create_fins_service(workspace_root: Path) -> FinsServiceProtocol:
    """创建财报服务实例。

    参数:
        workspace_root: 工作区根目录。

    返回值:
        财报服务实例。

    异常:
        Exception: 初始化依赖失败时抛出。
    """

    from dayu.host.host import Host
    from dayu.services import FinsService
    from dayu.startup.dependencies import prepare_fins_runtime

    fins_runtime = prepare_fins_runtime(workspace_root=workspace_root)
    host = Host(
        host_store_path=workspace_root / ".host",
        lane_config={
            "sec_download": 2,
        },
    )
    return FinsService(host=host, fins_runtime=fins_runtime)


def _create_write_service_and_chat_service(workspace_root: Path) -> tuple[WriteServiceProtocol, ChatServiceProtocol, Host]:
    """创建写作服务、聊天服务及共享 Host。

    参数:
        workspace_root: 工作区根目录。

    返回值:
        三元组：`(write_service, chat_service, host)`。

    异常:
        Exception: 依赖准备或实例化失败时抛出。
    """

    from dayu.contracts.session import SessionSource
    from dayu.host.host import Host
    from dayu.services import ChatService, WriteService, prepare_scene_execution_acceptance_preparer
    from dayu.startup.dependencies import (
        prepare_config_file_resolver,
        prepare_config_loader,
        prepare_default_execution_options,
        prepare_fins_runtime,
        prepare_model_catalog,
        prepare_prompt_asset_store,
        prepare_startup_paths,
        prepare_workspace_resources,
    )

    startup_paths = prepare_startup_paths(workspace_root=workspace_root, config_root=None)
    resolver = prepare_config_file_resolver(config_root=startup_paths.config_root)
    config_loader = prepare_config_loader(resolver=resolver)
    prompt_asset_store = prepare_prompt_asset_store(resolver=resolver)
    workspace = prepare_workspace_resources(
        paths=startup_paths,
        config_loader=config_loader,
        prompt_asset_store=prompt_asset_store,
    )
    model_catalog = prepare_model_catalog(config_loader=config_loader)
    default_execution_options = prepare_default_execution_options(
        workspace_root=startup_paths.workspace_root,
        config_loader=config_loader,
        execution_options=None,
    )
    scene_execution_acceptance_preparer = prepare_scene_execution_acceptance_preparer(
        workspace_root=startup_paths.workspace_root,
        default_execution_options=default_execution_options,
        model_catalog=model_catalog,
        prompt_asset_store=prompt_asset_store,
    )
    fins_runtime = prepare_fins_runtime(workspace_root=workspace_root)

    host = Host(
        workspace=workspace,
        model_catalog=model_catalog,
        default_execution_options=default_execution_options,
        host_store_path=workspace_root / ".host",
        lane_config={
            "llm_api": 3,
        },
    )

    write_service = WriteService(
        host=host,
        workspace=workspace,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        company_name_resolver=fins_runtime.get_company_name,
        company_meta_summary_resolver=fins_runtime.get_company_meta_summary,
    )
    chat_service = ChatService(
        host=host,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        company_name_resolver=fins_runtime.get_company_name,
        session_source=SessionSource.WEB,
    )
    return write_service, chat_service, host
