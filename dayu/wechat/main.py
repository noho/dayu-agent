"""WeChat UI 启动入口。"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, Protocol, TYPE_CHECKING

from dayu.log import Log, LogLevel
from dayu.contracts.session import SessionSource
from dayu.wechat.service_manager import (
    InstalledServiceDefinition,
    ServiceBackend,
    ServiceStatus,
    build_service_log_lines,
    build_service_label,
    build_service_spec,
    describe_service_backend,
    detect_service_backend,
    is_service_running,
    install_service,
    query_service_status,
    restart_service,
    resolve_service_definition_path,
    list_installed_service_definitions,
    start_service,
    stop_service,
    uninstall_service,
)
from dayu.wechat.state_store import FileWeChatStateStore, load_tracked_session_ids
from dayu.workspace_paths import DEFAULT_WECHAT_INSTANCE_LABEL, build_host_store_default_path, build_wechat_state_dir

if TYPE_CHECKING:
    from dayu.host import resolve_host_config
    from dayu.services import prepare_scene_execution_acceptance_preparer, recover_host_startup_state
    from dayu.services.contracts import (
        ChatPendingTurnView,
        ChatResumeRequest,
        ChatTurnRequest,
        ChatTurnSubmission,
        ReplyDeliveryFailureRequest,
        ReplyDeliverySubmitRequest,
        ReplyDeliveryView,
    )
    from dayu.startup.config_file_resolver import ConfigFileResolver
    from dayu.startup.config_loader import ConfigLoader
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
    from dayu.contracts.toolset_config import ToolsetConfigSnapshot
    from dayu.execution.options import ExecutionOptions, ExecutionOptionsOverridePayload, ResolvedExecutionOptions
    from dayu.fins.service_runtime import DefaultFinsRuntime
    from dayu.host.host import Host
    from dayu.services.chat_service import ChatService
    from dayu.services.host_admin_service import HostAdminService
    from dayu.services.reply_delivery_service import ReplyDeliveryService
    from dayu.services.scene_execution_acceptance import SceneExecutionAcceptancePreparer
    from dayu.startup.workspace import WorkspaceResources
    from dayu.wechat.daemon import WeChatDaemonConfig
    from dayu.wechat.state_store import WeChatDaemonState
else:

    def ConfigFileResolver(config_root: Path | None) -> object:
        """延迟导入并构建配置文件解析器。

        Args:
            config_root: 配置目录。

        Returns:
            配置文件解析器实例。

        Raises:
            无。
        """

        from dayu.startup.config_file_resolver import ConfigFileResolver as _RuntimeConfigFileResolver

        return _RuntimeConfigFileResolver(config_root)


    def ConfigLoader(resolver: object) -> object:
        """延迟导入并构建配置加载器。

        Args:
            resolver: 配置文件解析器。

        Returns:
            配置加载器实例。

        Raises:
            无。
        """

        from dayu.startup.config_loader import ConfigLoader as _RuntimeConfigLoader

        return _RuntimeConfigLoader(resolver)


    def prepare_startup_paths(*, workspace_root: Path, config_root: Path | None) -> object:
        """延迟导入并准备启动路径集合。

        Args:
            workspace_root: 工作区根目录。
            config_root: 配置目录。

        Returns:
            启动路径集合对象。

        Raises:
            无。
        """

        from dayu.startup.dependencies import prepare_startup_paths as _runtime_prepare_startup_paths

        return _runtime_prepare_startup_paths(workspace_root=workspace_root, config_root=config_root)


    def prepare_config_file_resolver(*, config_root: Path | None) -> object:
        """延迟导入并准备启动用配置解析器。

        Args:
            config_root: 配置目录。

        Returns:
            配置解析器实例。

        Raises:
            无。
        """

        from dayu.startup.dependencies import prepare_config_file_resolver as _runtime_prepare_config_file_resolver

        return _runtime_prepare_config_file_resolver(config_root=config_root)


    def prepare_config_loader(*, resolver: object) -> object:
        """延迟导入并准备启动用配置加载器。

        Args:
            resolver: 配置解析器。

        Returns:
            配置加载器实例。

        Raises:
            无。
        """

        from dayu.startup.dependencies import prepare_config_loader as _runtime_prepare_config_loader

        return _runtime_prepare_config_loader(resolver=resolver)


    def prepare_prompt_asset_store(*, resolver: object) -> object:
        """延迟导入并准备 prompt 资产仓。

        Args:
            resolver: 配置解析器。

        Returns:
            Prompt 资产仓实例。

        Raises:
            无。
        """

        from dayu.startup.dependencies import prepare_prompt_asset_store as _runtime_prepare_prompt_asset_store

        return _runtime_prepare_prompt_asset_store(resolver=resolver)


    def prepare_workspace_resources(*, paths: object, config_loader: object, prompt_asset_store: object) -> object:
        """延迟导入并准备工作区资源。

        Args:
            paths: 启动路径集合。
            config_loader: 配置加载器。
            prompt_asset_store: Prompt 资产仓。

        Returns:
            工作区资源对象。

        Raises:
            无。
        """

        from dayu.startup.dependencies import prepare_workspace_resources as _runtime_prepare_workspace_resources

        return _runtime_prepare_workspace_resources(
            paths=paths,
            config_loader=config_loader,
            prompt_asset_store=prompt_asset_store,
        )


    def prepare_model_catalog(*, config_loader: object) -> object:
        """延迟导入并准备模型目录。

        Args:
            config_loader: 配置加载器。

        Returns:
            模型目录对象。

        Raises:
            无。
        """

        from dayu.startup.dependencies import prepare_model_catalog as _runtime_prepare_model_catalog

        return _runtime_prepare_model_catalog(config_loader=config_loader)


    def prepare_default_execution_options(
        *,
        workspace_root: Path,
        config_loader: object,
        execution_options: ExecutionOptions,
    ) -> object:
        """延迟导入并准备默认执行选项。

        Args:
            workspace_root: 工作区根目录。
            config_loader: 配置加载器。
            execution_options: 请求级执行选项。

        Returns:
            解析后的默认执行选项。

        Raises:
            无。
        """

        from dayu.startup.dependencies import prepare_default_execution_options as _runtime_prepare_default_execution_options

        return _runtime_prepare_default_execution_options(
            workspace_root=workspace_root,
            config_loader=config_loader,
            execution_options=execution_options,
        )


    def prepare_scene_execution_acceptance_preparer(
        *,
        workspace_root: Path,
        default_execution_options: ResolvedExecutionOptions,
        model_catalog: object,
        prompt_asset_store: object,
    ) -> object:
        """延迟导入并准备 scene 执行接受预处理器。

        Args:
            workspace_root: 工作区根目录。
            default_execution_options: 默认执行选项。
            model_catalog: 模型目录。
            prompt_asset_store: Prompt 资产仓。

        Returns:
            Scene 执行接受预处理器。

        Raises:
            无。
        """

        from dayu.services import prepare_scene_execution_acceptance_preparer as _runtime_prepare_scene_execution_acceptance_preparer

        return _runtime_prepare_scene_execution_acceptance_preparer(
            workspace_root=workspace_root,
            default_execution_options=default_execution_options,
            model_catalog=model_catalog,
            prompt_asset_store=prompt_asset_store,
        )


    def prepare_fins_runtime(*, workspace_root: Path) -> object:
        """延迟导入并准备财报运行时。

        Args:
            workspace_root: 工作区根目录。

        Returns:
            财报运行时对象。

        Raises:
            无。
        """

        from dayu.startup.dependencies import prepare_fins_runtime as _runtime_prepare_fins_runtime

        return _runtime_prepare_fins_runtime(workspace_root=workspace_root)


    def resolve_host_config(*, workspace_root: Path, run_config: object, explicit_lane_config: object | None) -> object:
        """延迟导入并解析 Host 配置。

        Args:
            workspace_root: 工作区根目录。
            run_config: 运行配置。
            explicit_lane_config: 可选显式 lane 配置。

        Returns:
            Host 配置对象。

        Raises:
            无。
        """

        from dayu.host import resolve_host_config as _runtime_resolve_host_config

        return _runtime_resolve_host_config(
            workspace_root=workspace_root,
            run_config=run_config,
            explicit_lane_config=explicit_lane_config,
        )


    def Host(**kwargs: object) -> object:
        """延迟导入并构建 Host。

        Args:
            **kwargs: Host 构造参数。

        Returns:
            Host 实例。

        Raises:
            无。
        """

        from dayu.host.host import Host as _RuntimeHost

        return _RuntimeHost(**kwargs)


    def HostAdminService(*, host: object) -> object:
        """延迟导入并构建 Host 管理服务。

        Args:
            host: Host 实例。

        Returns:
            Host 管理服务实例。

        Raises:
            无。
        """

        from dayu.services.host_admin_service import HostAdminService as _RuntimeHostAdminService

        return _RuntimeHostAdminService(host=host)


    def recover_host_startup_state(host_admin_service: object, *, runtime_label: str, log_module: str) -> object:
        """延迟导入并执行统一 startup recovery。

        Args:
            host_admin_service: Host 管理服务。
            runtime_label: 运行时标签。
            log_module: 日志模块名。

        Returns:
            Startup recovery 结果对象。

        Raises:
            无。
        """

        from dayu.services import recover_host_startup_state as _runtime_recover_host_startup_state

        return _runtime_recover_host_startup_state(
            host_admin_service,
            runtime_label=runtime_label,
            log_module=log_module,
        )


    def ChatService(**kwargs: object) -> object:
        """延迟导入并构建聊天服务。

        Args:
            **kwargs: ChatService 构造参数。

        Returns:
            ChatService 实例。

        Raises:
            无。
        """

        from dayu.services.chat_service import ChatService as _RuntimeChatService

        return _RuntimeChatService(**kwargs)


    def ReplyDeliveryService(*, host: object) -> object:
        """延迟导入并构建回复投递服务。

        Args:
            host: Host 实例。

        Returns:
            回复投递服务实例。

        Raises:
            无。
        """

        from dayu.services.reply_delivery_service import ReplyDeliveryService as _RuntimeReplyDeliveryService

        return _RuntimeReplyDeliveryService(host=host)

MODULE = "APP.WECHAT.MAIN"
DEFAULT_TYPING_INTERVAL_SEC = 8.0
DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS = 3
_WECHAT_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
# 由各 service/tool 模块作为真源声明，daemon 运行时需要显式捕获转发。
# 如需新增/改名，必须在真源模块改，而不是在这里硬编码。


def _direct_service_env_var_names() -> tuple[str, ...]:
    """延迟导入后台 service 直接依赖的环境变量名集合。

    Args:
        无。

    Returns:
        后台 service 必须直传的环境变量名元组。

    Raises:
        无。
    """

    from dayu.engine.processors.perf_utils import PROFILE_ENV_NAME
    from dayu.engine.tools.web_search_providers import SERPER_API_KEY_ENV, TAVILY_API_KEY_ENV
    from dayu.engine.tools.web_tools import SEC_USER_AGENT_ENV
    from dayu.fins.resolver.fmp_company_alias_resolver import FMP_API_KEY_ENV

    return (
        FMP_API_KEY_ENV,
        SEC_USER_AGENT_ENV,
        SERPER_API_KEY_ENV,
        TAVILY_API_KEY_ENV,
        PROFILE_ENV_NAME,
    )


class _WeChatDaemonLike(Protocol):
    """WeChat daemon 的最小可调用协议。"""

    async def ensure_authenticated(self, *, force_relogin: bool = False) -> WeChatDaemonState:
        """确保 daemon 完成登录认证。"""

        ...

    async def aclose(self) -> None:
        """关闭 daemon 持有的外部资源。"""

        ...

    async def run_forever(self, *, require_existing_auth: bool) -> None:
        """以前台方式持续运行 daemon。"""

        ...


def WeChatDaemon(
    *,
    chat_service: ChatService | _NoOpChatService,
    reply_delivery_service: ReplyDeliveryService | _NoOpReplyDeliveryService,
    state_store: FileWeChatStateStore,
    config: WeChatDaemonConfig,
) -> _WeChatDaemonLike:
    """延迟导入并构建 WeChat daemon。

    Args:
        chat_service: 聊天服务。
        reply_delivery_service: 回复投递服务。
        state_store: WeChat 状态仓储。
        config: daemon 配置。

    Returns:
        WeChat daemon 实例。

    Raises:
        无。
    """

    from dayu.wechat.daemon import WeChatDaemon as _RuntimeWeChatDaemon

    return _RuntimeWeChatDaemon(
        chat_service=chat_service,
        reply_delivery_service=reply_delivery_service,
        state_store=state_store,
        config=config,
    )


@dataclass(frozen=True)
class _ResolvedWechatContext:
    """WeChat 命令解析后的共享上下文。"""

    workspace_root: Path
    config_root: Path | None
    state_dir: Path
    execution_options: ExecutionOptions
    delivery_max_attempts: int = DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS
    instance_label: str = DEFAULT_WECHAT_INSTANCE_LABEL


@dataclass(frozen=True)
class _ResolvedWechatServiceIdentity:
    """WeChat service 的稳定身份信息。"""

    backend: ServiceBackend
    label: str
    definition_path: Path
    state_dir: Path
    instance_label: str = DEFAULT_WECHAT_INSTANCE_LABEL


@dataclass(frozen=True)
class _InstalledWechatServiceView:
    """已安装 WeChat service 的展示视图。"""

    instance_label: str
    service_label: str
    backend: ServiceBackend
    definition_path: Path
    state_dir: Path
    running: bool
    logged_in: bool


@dataclass
class _DaemonShutdownState:
    """WeChat daemon 的关停状态。"""

    signal_name: str | None = None
    exit_code: int = 0


class _NoOpChatService:
    """仅供 login 命令使用的占位 ChatService。"""

    async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
        """login 命令不应调用聊天逻辑。

        Args:
            request: 忽略。

        Returns:
            无。

        Raises:
            RuntimeError: 一旦误调用即抛错。
        """

        del request
        raise RuntimeError("login 模式不应调用 ChatService")

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        """login 命令不应调用恢复逻辑。"""

        del request
        raise RuntimeError("login 模式不应调用 ChatService")

    def list_resumable_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
    ) -> list[ChatPendingTurnView]:
        """login 命令不应调用恢复列表逻辑。"""

        del session_id
        del scene_name
        raise RuntimeError("login 模式不应调用 ChatService")


class _NoOpReplyDeliveryService:
    """仅供 login 命令使用的占位 ReplyDeliveryService。"""

    def submit_reply_for_delivery(self, request: ReplyDeliverySubmitRequest) -> ReplyDeliveryView:
        """login 模式不应调用交付逻辑。"""

        del request
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")

    def get_delivery(self, delivery_id: str) -> ReplyDeliveryView | None:
        """login 模式不应调用交付逻辑。"""

        del delivery_id
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")

    def list_deliveries(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: str | None = None,
    ) -> list[ReplyDeliveryView]:
        """login 模式不应调用交付逻辑。"""

        del session_id
        del scene_name
        del state
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")

    def claim_delivery(self, delivery_id: str) -> ReplyDeliveryView:
        """login 模式不应调用交付逻辑。"""

        del delivery_id
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")

    def mark_delivery_delivered(self, delivery_id: str) -> ReplyDeliveryView:
        """login 模式不应调用交付逻辑。"""

        del delivery_id
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")

    def mark_delivery_failed(self, request: ReplyDeliveryFailureRequest) -> ReplyDeliveryView:
        """login 模式不应调用交付逻辑。"""

        del request
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")


class DayuWechatArgumentParser(argparse.ArgumentParser):
    """`dayu.wechat` 顶层参数解析器。

    设计意图：
    - 统一固定 `python -m dayu.wechat` 作为程序名，避免暴露 `__main__.py`。
    - 在缺少顶层子命令时输出完整帮助，而不是仅输出 argparse 默认错误。
    """

    def error(self, message: str) -> NoReturn:
        """输出更适合人读的参数错误信息。

        Args:
            message: argparse 生成的错误文案。

        Returns:
            无。

        Raises:
            SystemExit: 参数解析失败时退出。
        """

        if "required: command" in message:
            self.print_help(sys.stderr)
            self.exit(2, "\n错误: 缺少命令。请先选择一个命令，再用 `--help` 查看该命令的具体参数。\n")
        super().error(message)


def _add_log_level_args(parser: argparse.ArgumentParser) -> None:
    """为 parser 添加日志参数。

    Args:
        parser: argparse parser。

    Returns:
        无。

    Raises:
        无。
    """

    level_group = parser.add_mutually_exclusive_group()
    parser.add_argument("--log-level", choices=["debug", "verbose", "info", "warn", "error"], default=None)
    level_group.add_argument("--debug", action="store_true", default=False)
    level_group.add_argument("--verbose", action="store_true", default=False)
    level_group.add_argument("--info", action="store_true", default=False)
    level_group.add_argument("--quiet", action="store_true", default=False)


def _add_agent_args(parser: argparse.ArgumentParser) -> None:
    """添加与 interactive 对齐的 Agent 覆盖参数。

    Args:
        parser: argparse parser。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--model-name", default=None, help="覆盖 wechat scene 的模型配置名称")
    parser.add_argument("--temperature", default=None, help="覆盖模型 temperature")
    parser.add_argument("--web-provider", default=None, help="覆盖联网 provider")
    parser.add_argument("--tool-timeout-seconds", type=float, default=None, help="覆盖 tool timeout（秒）")
    parser.add_argument("--max-iterations", type=int, default=None, help="覆盖 Agent 最大迭代次数")
    parser.add_argument(
        "--max-consecutive-failed-tool-batches",
        type=int,
        default=None,
        help="覆盖连续失败工具批次上限",
    )
    parser.add_argument("--enable-tool-trace", action="store_true", default=False, help="显式开启 tool trace")
    parser.add_argument("--tool-trace-dir", default=None, help="覆盖 tool trace 输出目录")
    parser.add_argument("--doc-limits-json", default=None, help="文档工具 limits 的 JSON 覆盖")
    parser.add_argument("--fins-limits-json", default=None, help="财报工具 limits 的 JSON 覆盖")


def _parse_wechat_label_argument(raw_label: str) -> str:
    """解析并校验 WeChat 实例标签。

    Args:
        raw_label: 原始命令行标签。

    Returns:
        去除首尾空白后的实例标签。

    Raises:
        argparse.ArgumentTypeError: 当标签为空或包含非法字符时抛出。
    """

    normalized_label = raw_label.strip()
    if not normalized_label:
        raise argparse.ArgumentTypeError("`--label` 不能为空")
    if _WECHAT_LABEL_PATTERN.fullmatch(normalized_label) is None:
        raise argparse.ArgumentTypeError("`--label` 只允许字母、数字、下划线和连字符，且必须以字母或数字开头")
    return normalized_label


def _add_base_args(parser: argparse.ArgumentParser) -> None:
    """添加工作区与 WeChat 实例参数。

    Args:
        parser: argparse parser。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--base", default="./workspace", help="工作区根目录，默认 ./workspace")
    parser.add_argument("--config", default=None, help="配置目录，默认 <base>/config")
    parser.add_argument(
        "--label",
        default=None,
        type=_parse_wechat_label_argument,
        help="WeChat 实例标签，默认 default；状态目录映射到 <base>/.dayu/wechat-<label>",
    )


def _add_login_args(parser: argparse.ArgumentParser) -> None:
    """添加登录相关参数。

    Args:
        parser: argparse parser。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--relogin", action="store_true", default=False, help="忽略缓存 token，强制重新扫码登录")
    parser.add_argument("--qrcode-timeout-sec", type=float, default=None, help="扫码登录超时秒数")


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """添加运行相关参数。

    Args:
        parser: argparse parser。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--typing-interval-sec", type=float, default=DEFAULT_TYPING_INTERVAL_SEC, help="发送 typing 的间隔秒数")
    parser.add_argument(
        "--delivery-max-attempts",
        type=int,
        default=DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS,
        help="微信 reply delivery 的最大发送次数",
    )


def _add_service_identity_args(parser: argparse.ArgumentParser) -> None:
    """添加 service 控制命令所需的实例参数。

    Args:
        parser: argparse parser。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--base", default="./workspace", help="工作区根目录，默认 ./workspace")
    parser.add_argument(
        "--label",
        default=None,
        type=_parse_wechat_label_argument,
        help="WeChat 实例标签，默认 default；状态目录映射到 <base>/.dayu/wechat-<label>",
    )


def _add_service_list_args(parser: argparse.ArgumentParser) -> None:
    """添加 `service list` 所需的工作区参数。

    Args:
        parser: argparse parser。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--base", default="./workspace", help="工作区根目录，默认 ./workspace")


def _create_parser() -> argparse.ArgumentParser:
    """创建 WeChat CLI 参数解析器。

    Args:
        无。

    Returns:
        argparse parser。

    Raises:
        无。
    """

    parser = DayuWechatArgumentParser(
        prog="python -m dayu.wechat",
        description="Dayu WeChat CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser("login", help="扫码登录 WeChat ClawBot")
    _add_base_args(login_parser)
    _add_login_args(login_parser)
    _add_log_level_args(login_parser)

    run_parser = subparsers.add_parser("run", help="使用现有登录态运行 WeChat daemon")
    _add_base_args(run_parser)
    _add_run_args(run_parser)
    _add_log_level_args(run_parser)
    _add_agent_args(run_parser)

    service_parser = subparsers.add_parser("service", help="管理当前平台的用户级系统 service")
    service_subparsers = service_parser.add_subparsers(dest="service_command", required=True)

    service_install_parser = service_subparsers.add_parser("install", help="安装当前平台的 service 定义")
    _add_base_args(service_install_parser)
    _add_run_args(service_install_parser)
    _add_log_level_args(service_install_parser)
    _add_agent_args(service_install_parser)

    service_start_parser = service_subparsers.add_parser("start", help="启动已安装的系统 service")
    _add_service_identity_args(service_start_parser)
    _add_log_level_args(service_start_parser)

    service_restart_parser = service_subparsers.add_parser("restart", help="重启已安装的系统 service")
    _add_service_identity_args(service_restart_parser)
    _add_log_level_args(service_restart_parser)

    service_stop_parser = service_subparsers.add_parser("stop", help="停止已安装的系统 service")
    _add_service_identity_args(service_stop_parser)
    _add_log_level_args(service_stop_parser)

    service_status_parser = service_subparsers.add_parser("status", help="查看系统 service 状态")
    _add_service_identity_args(service_status_parser)
    _add_log_level_args(service_status_parser)

    service_list_parser = service_subparsers.add_parser("list", help="列出当前 workspace 下已安装的系统 service")
    _add_service_list_args(service_list_parser)
    _add_log_level_args(service_list_parser)

    service_uninstall_parser = service_subparsers.add_parser("uninstall", help="卸载系统 service 定义")
    _add_service_identity_args(service_uninstall_parser)
    _add_log_level_args(service_uninstall_parser)

    return parser


def _parse_limits_override(
    raw_json: str | None,
    *,
    field_name: str,
) -> ExecutionOptionsOverridePayload | None:
    """解析工具 limits JSON 覆盖。

    Args:
        raw_json: JSON 字符串。
        field_name: 参数名。

    Returns:
        字典对象；未传时返回 `None`。

    Raises:
        SystemExit: 当 JSON 非法时退出。
    """

    if raw_json is None:
        return None
    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        Log.error(f"{field_name} 不是合法 JSON: {exc}", module=MODULE)
        raise SystemExit(2) from exc
    if not isinstance(parsed, dict):
        Log.error(f"{field_name} 必须是 JSON 对象", module=MODULE)
        raise SystemExit(2)
    normalized: ExecutionOptionsOverridePayload = {}
    for key, value in parsed.items():
        if value is None or isinstance(value, str | int | float | bool):
            normalized[str(key)] = value
            continue
        Log.error(f"{field_name} 只允许 JSON 标量值，字段 {key!r} 非法", module=MODULE)
        raise SystemExit(2)
    return normalized


def _parse_temperature_argument(raw_value: Any, *, field_name: str) -> float | None:
    """解析 temperature 参数。

    Args:
        raw_value: 原始值。
        field_name: 参数名。

    Returns:
        标准化后的 float；未传时返回 `None`。

    Raises:
        SystemExit: 当参数非法时退出。
    """

    from dayu.execution.options import normalize_temperature

    try:
        return normalize_temperature(raw_value, field_name=field_name)
    except ValueError as exc:
        Log.error(str(exc), module=MODULE)
        raise SystemExit(2) from exc


def _build_execution_options(args: argparse.Namespace) -> ExecutionOptions:
    """构建 WeChat 命令的请求级执行选项。

    Args:
        args: argparse 解析结果。

    Returns:
        ExecutionOptions。

    Raises:
        SystemExit: 当参数非法时退出。
    """

    from dayu.contracts.toolset_config import build_toolset_config_snapshot
    from dayu.execution.options import ExecutionOptions

    doc_limits = _parse_limits_override(getattr(args, "doc_limits_json", None), field_name="--doc-limits-json")
    fins_limits = _parse_limits_override(getattr(args, "fins_limits_json", None), field_name="--fins-limits-json")
    toolset_config_overrides: list[ToolsetConfigSnapshot] = []
    for snapshot in (
        build_toolset_config_snapshot("doc", doc_limits),
        build_toolset_config_snapshot("fins", fins_limits),
    ):
        if snapshot is not None:
            toolset_config_overrides.append(snapshot)
    return ExecutionOptions(
        model_name=(raw_model_name if (raw_model_name := str(getattr(args, "model_name", "") or "").strip()) else None),
        temperature=_parse_temperature_argument(getattr(args, "temperature", None), field_name="--temperature"),
        tool_timeout_seconds=getattr(args, "tool_timeout_seconds", None),
        max_iterations=getattr(args, "max_iterations", None),
        max_consecutive_failed_tool_batches=getattr(args, "max_consecutive_failed_tool_batches", None),
        web_provider=getattr(args, "web_provider", None),
        trace_enabled=(True if bool(getattr(args, "enable_tool_trace", False)) else None),
        trace_output_dir=Path(getattr(args, "tool_trace_dir")).expanduser().resolve()
        if getattr(args, "tool_trace_dir", None)
        else None,
        toolset_config_overrides=tuple(toolset_config_overrides),
    )


def _resolve_workspace_root(raw_base: str) -> Path:
    """解析工作区根目录。

    Args:
        raw_base: `--base` 原始值。

    Returns:
        绝对路径。

    Raises:
        SystemExit: 当目录不存在或不是目录时退出。
    """

    workspace_root = Path(raw_base).expanduser().resolve()
    if not workspace_root.exists():
        Log.error(f"工作区目录不存在: {workspace_root}", module=MODULE)
        raise SystemExit(1)
    if not workspace_root.is_dir():
        Log.error(f"工作区路径不是目录: {workspace_root}", module=MODULE)
        raise SystemExit(1)
    return workspace_root


def _resolve_config_root(workspace_root: Path, raw_config_root: str | None) -> Path | None:
    """解析配置目录。

    Args:
        workspace_root: 工作区目录。
        raw_config_root: `--config` 原始值。

    Returns:
        配置目录绝对路径；未传时返回 `None`。

    Raises:
        无。
    """

    if raw_config_root:
        return Path(raw_config_root).expanduser().resolve()
    return (workspace_root / "config").resolve()


def _resolve_instance_label(raw_label: str | None) -> str:
    """解析 WeChat 实例标签。

    Args:
        raw_label: `--label` 原始值。

    Returns:
        合法的实例标签。

    Raises:
        SystemExit: 当标签非法时抛出。
    """

    if raw_label is None:
        return DEFAULT_WECHAT_INSTANCE_LABEL
    try:
        return _parse_wechat_label_argument(raw_label)
    except argparse.ArgumentTypeError as error:
        Log.error(str(error), module=MODULE)
        raise SystemExit(2) from error


def _resolve_state_dir(workspace_root: Path, instance_label: str) -> Path:
    """根据实例标签解析 WeChat 状态目录。

    Args:
        workspace_root: 工作区目录。
        instance_label: 已校验的 WeChat 实例标签。

    Returns:
        状态目录绝对路径。

    Raises:
        无。
    """

    return build_wechat_state_dir(workspace_root, label=instance_label).resolve()


def _resolve_command_context(args: argparse.Namespace) -> _ResolvedWechatContext:
    """解析 WeChat 命令的共享上下文。

    Args:
        args: argparse 解析结果。

    Returns:
        解析后的上下文。

    Raises:
        SystemExit: 当工作区路径非法时抛出。
    """

    workspace_root = _resolve_workspace_root(args.base)
    instance_label = _resolve_instance_label(getattr(args, "label", None))
    return _ResolvedWechatContext(
        workspace_root=workspace_root,
        config_root=_resolve_config_root(workspace_root, getattr(args, "config", None)),
        state_dir=_resolve_state_dir(workspace_root, instance_label),
        execution_options=_build_execution_options(args),
        delivery_max_attempts=int(
            getattr(args, "delivery_max_attempts", DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS)
        ),
        instance_label=instance_label,
    )


def _find_installed_service_definition_for_instance(
    workspace_root: Path,
    instance_label: str,
    backend: ServiceBackend,
) -> InstalledServiceDefinition | None:
    """按 workspace 与实例标签查找已安装的 WeChat service definition。

    Args:
        workspace_root: 工作区根目录。
        instance_label: WeChat 实例标签。
        backend: 当前 service backend。

    Returns:
        命中的已安装 definition；未命中时返回 `None`。

    Raises:
        RuntimeError: 当平台不支持 service 管理时抛出。
    """

    resolved_workspace_root = workspace_root.resolve()
    for definition in list_installed_service_definitions(backend):
        runtime_identity = _parse_installed_service_runtime_identity(definition)
        if runtime_identity is None:
            continue
        definition_workspace_root, definition_instance_label = runtime_identity
        if definition_workspace_root != resolved_workspace_root:
            continue
        if definition_instance_label != instance_label:
            continue
        return definition
    return None


def _resolve_service_identity(args: argparse.Namespace) -> _ResolvedWechatServiceIdentity:
    """解析 WeChat service 的稳定身份。

    Args:
        args: argparse 解析结果。

    Returns:
        service 身份。

    Raises:
        SystemExit: 当工作区路径非法时抛出。
    """

    workspace_root = _resolve_workspace_root(args.base)
    instance_label = _resolve_instance_label(getattr(args, "label", None))
    state_dir = _resolve_state_dir(workspace_root, instance_label)
    backend = detect_service_backend()
    installed_definition = _find_installed_service_definition_for_instance(workspace_root, instance_label, backend)
    if installed_definition is not None:
        return _ResolvedWechatServiceIdentity(
            backend=backend,
            label=installed_definition.label,
            definition_path=installed_definition.definition_path,
            state_dir=state_dir,
            instance_label=instance_label,
        )
    label = build_service_label(state_dir)
    return _ResolvedWechatServiceIdentity(
        backend=backend,
        label=label,
        definition_path=resolve_service_definition_path(label, backend=backend),
        state_dir=state_dir,
        instance_label=instance_label,
    )


def _get_service_backend_display_name(backend: ServiceBackend) -> str:
    """返回当前 service backend 的展示名。

    Args:
        backend: service backend。

    Returns:
        适合 CLI 输出的展示名。

    Raises:
        ValueError: 当 backend 非法时抛出。
    """

    return describe_service_backend(backend)


def _resolve_repo_root() -> Path:
    """解析仓库根目录。

    Args:
        无。

    Returns:
        仓库根目录绝对路径。

    Raises:
        无。
    """

    return Path(__file__).resolve().parents[2]


def _build_daemon_config(
    args: argparse.Namespace,
    context: _ResolvedWechatContext,
    *,
    allow_interactive_relogin: bool,
) -> WeChatDaemonConfig:
    """构建 WeChat daemon 配置。

    Args:
        args: argparse 解析结果。
        context: 共享上下文。
        allow_interactive_relogin: 是否允许运行中重新扫码登录。

    Returns:
        daemon 配置。

    Raises:
        无。
    """

    from dayu.wechat.daemon import WeChatDaemonConfig as _RuntimeWeChatDaemonConfig

    return _RuntimeWeChatDaemonConfig(
        scene_name="wechat",
        allow_interactive_relogin=allow_interactive_relogin,
        execution_options=context.execution_options,
        qrcode_timeout_sec=getattr(args, "qrcode_timeout_sec", None),
        typing_interval_sec=float(getattr(args, "typing_interval_sec", DEFAULT_TYPING_INTERVAL_SEC)),
        delivery_max_attempts=context.delivery_max_attempts,
    )


def _build_run_cli_arguments(args: argparse.Namespace, context: _ResolvedWechatContext) -> list[str]:
    """构建 launchd service 运行时的命令行参数。

    Args:
        args: argparse 解析结果。
        context: 已解析的共享上下文。

    Returns:
        `python -m dayu.wechat` 后续参数列表。

    Raises:
        无。
    """

    cli_arguments = [
        "run",
        "--base",
        str(context.workspace_root),
        "--config",
        str(context.config_root),
        "--label",
        context.instance_label,
    ]
    typing_interval_sec = float(getattr(args, "typing_interval_sec", DEFAULT_TYPING_INTERVAL_SEC))
    if typing_interval_sec != DEFAULT_TYPING_INTERVAL_SEC:
        cli_arguments.extend(["--typing-interval-sec", str(typing_interval_sec)])
    delivery_max_attempts = int(getattr(args, "delivery_max_attempts", context.delivery_max_attempts))
    # argparse 不保留“用户显式传了默认值”这一信息，因此这里退化为：
    # 1. context 已经携带非默认值时，必须显式写回 service run 参数；
    # 2. 当前解析值本身是非默认值时，也必须显式写回。
    context_uses_non_default_delivery_max_attempts = (
        context.delivery_max_attempts != DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS
    )
    delivery_max_attempts_overrides_default = (
        delivery_max_attempts != DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS
    )
    if context_uses_non_default_delivery_max_attempts or delivery_max_attempts_overrides_default:
        cli_arguments.extend(["--delivery-max-attempts", str(delivery_max_attempts)])
    _append_log_level_arguments(args, cli_arguments)
    _append_agent_override_arguments(args, cli_arguments)
    return cli_arguments


def _append_log_level_arguments(args: argparse.Namespace, cli_arguments: list[str]) -> None:
    """把日志参数追加到命令行参数列表。

    Args:
        args: argparse 解析结果。
        cli_arguments: 待追加的参数列表。

    Returns:
        无。

    Raises:
        无。
    """

    if getattr(args, "log_level", None):
        cli_arguments.extend(["--log-level", str(args.log_level)])
    elif bool(getattr(args, "debug", False)):
        cli_arguments.append("--debug")
    elif bool(getattr(args, "verbose", False)):
        cli_arguments.append("--verbose")
    elif bool(getattr(args, "info", False)):
        cli_arguments.append("--info")
    elif bool(getattr(args, "quiet", False)):
        cli_arguments.append("--quiet")


def _append_agent_override_arguments(args: argparse.Namespace, cli_arguments: list[str]) -> None:
    """把 Agent 覆盖参数追加到命令行参数列表。

    Args:
        args: argparse 解析结果。
        cli_arguments: 待追加的参数列表。

    Returns:
        无。

    Raises:
        无。
    """

    _append_optional_argument(cli_arguments, "--model-name", getattr(args, "model_name", None))
    _append_optional_argument(cli_arguments, "--temperature", getattr(args, "temperature", None))
    _append_optional_argument(cli_arguments, "--web-provider", getattr(args, "web_provider", None))
    _append_optional_argument(cli_arguments, "--tool-timeout-seconds", getattr(args, "tool_timeout_seconds", None))
    _append_optional_argument(cli_arguments, "--max-iterations", getattr(args, "max_iterations", None))
    _append_optional_argument(
        cli_arguments,
        "--max-consecutive-failed-tool-batches",
        getattr(args, "max_consecutive_failed_tool_batches", None),
    )
    if bool(getattr(args, "enable_tool_trace", False)):
        cli_arguments.append("--enable-tool-trace")
    _append_optional_argument(cli_arguments, "--tool-trace-dir", getattr(args, "tool_trace_dir", None))
    _append_optional_argument(cli_arguments, "--doc-limits-json", getattr(args, "doc_limits_json", None))
    _append_optional_argument(cli_arguments, "--fins-limits-json", getattr(args, "fins_limits_json", None))


def _collect_service_environment_variables(context: _ResolvedWechatContext) -> dict[str, str]:
    """收集后台 service 需要显式注入的环境变量。

    策略分两类：
    - 配置占位符引用到的环境变量：由 ``ConfigLoader`` 按当前生效配置自动发现；
    - 代码中直接读取、无法从配置占位符反推的环境变量：走显式清单。

    Args:
        context: WeChat 共享上下文。

    Returns:
        仅包含当前进程里已配置值的环境变量映射。

    Raises:
        OSError: 当配置文件读取失败时抛出。
    """

    config_loader = ConfigLoader(ConfigFileResolver(context.config_root))
    required_names = set(config_loader.collect_referenced_env_vars())
    required_names.update(_direct_service_env_var_names())
    captured_environment: dict[str, str] = {}
    for name in sorted(required_names):
        value = os.environ.get(name)
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized:
            continue
        captured_environment[name] = normalized
    return captured_environment


def _append_optional_argument(cli_arguments: list[str], flag: str, value: Any) -> None:
    """把可选参数追加到命令行列表。

    Args:
        cli_arguments: 待追加列表。
        flag: 参数名。
        value: 参数值。

    Returns:
        无。

    Raises:
        无。
    """

    if value is None:
        return
    normalized = str(value).strip()
    if not normalized:
        return
    cli_arguments.extend([flag, normalized])


def _create_login_daemon(args: argparse.Namespace, context: _ResolvedWechatContext) -> _WeChatDaemonLike:
    """构建仅用于登录的 WeChat daemon。

    Args:
        args: argparse 解析结果。
        context: 共享上下文。

    Returns:
        login 专用 daemon。

    Raises:
        无。
    """

    return WeChatDaemon(
        chat_service=_NoOpChatService(),
        reply_delivery_service=_NoOpReplyDeliveryService(),
        state_store=FileWeChatStateStore(context.state_dir),
        config=_build_daemon_config(args, context, allow_interactive_relogin=False),
    )


def _prepare_wechat_host_dependencies(
    context: _ResolvedWechatContext,
) -> tuple[
    WorkspaceResources,
    ResolvedExecutionOptions,
    SceneExecutionAcceptancePreparer,
    Host,
    DefaultFinsRuntime,
]:
    """准备 WeChat 的 Host 级稳定依赖。

    Args:
        context: 共享上下文。

    Returns:
        ``(workspace, default_execution_options,
        scene_execution_acceptance_preparer, host, fins_runtime)``。

    Raises:
        无。
    """

    paths = prepare_startup_paths(
        workspace_root=context.workspace_root,
        config_root=context.config_root,
    )
    resolver = prepare_config_file_resolver(config_root=paths.config_root)
    config_loader = prepare_config_loader(resolver=resolver)
    prompt_asset_store = prepare_prompt_asset_store(resolver=resolver)
    workspace = prepare_workspace_resources(
        paths=paths,
        config_loader=config_loader,
        prompt_asset_store=prompt_asset_store,
    )
    model_catalog = prepare_model_catalog(config_loader=config_loader)
    default_execution_options = prepare_default_execution_options(
        workspace_root=paths.workspace_root,
        config_loader=config_loader,
        execution_options=context.execution_options,
    )
    scene_execution_acceptance_preparer = prepare_scene_execution_acceptance_preparer(
        workspace_root=paths.workspace_root,
        default_execution_options=default_execution_options,
        model_catalog=model_catalog,
        prompt_asset_store=prompt_asset_store,
    )
    fins_runtime = prepare_fins_runtime(workspace_root=paths.workspace_root)
    run_config = config_loader.load_run_config()
    host_config = resolve_host_config(
        workspace_root=paths.workspace_root,
        run_config=run_config,
        explicit_lane_config=None,
    )
    host = Host(
        workspace=workspace,
        model_catalog=model_catalog,
        default_execution_options=default_execution_options,
        host_store_path=host_config.store_path,
        lane_config=host_config.lane_config,
        pending_turn_resume_max_attempts=host_config.pending_turn_resume_max_attempts,
        event_bus=None,
    )
    recover_host_startup_state(
        HostAdminService(host=host),
        runtime_label="WeChat Host runtime",
        log_module=MODULE,
    )
    return (
        workspace,
        default_execution_options,
        scene_execution_acceptance_preparer,
        host,
        fins_runtime,
    )


def _create_run_daemon(args: argparse.Namespace, context: _ResolvedWechatContext) -> _WeChatDaemonLike:
    """构建运行命令使用的 WeChat daemon。

    Args:
        args: argparse 解析结果。
        context: 共享上下文。

    Returns:
        run 专用 daemon。

    Raises:
        无。
    """

    (
        _workspace,
        _default_execution_options,
        scene_execution_acceptance_preparer,
        host,
        fins_runtime,
    ) = _prepare_wechat_host_dependencies(context)

    # 记录 scene 模型信息用于运维诊断
    scene_model = scene_execution_acceptance_preparer.resolve_scene_model("wechat", context.execution_options)
    Log.info(f"工作目录: {context.workspace_root}", module=MODULE)
    Log.info(
        "wechat scene 模型: " + json.dumps(scene_model, ensure_ascii=False, sort_keys=True),
        module=MODULE,
    )
    chat_service = ChatService(
        host=host,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        company_name_resolver=fins_runtime.get_company_name,
        session_source=SessionSource.WECHAT,
    )
    reply_delivery_service = ReplyDeliveryService(host=host)
    return WeChatDaemon(
        chat_service=chat_service,
        reply_delivery_service=reply_delivery_service,
        state_store=FileWeChatStateStore(context.state_dir),
        config=_build_daemon_config(args, context, allow_interactive_relogin=False),
    )


def _request_daemon_shutdown(
    run_task: asyncio.Task[None],
    shutdown_state: _DaemonShutdownState,
    signal_name: str,
    exit_code: int,
) -> None:
    """请求 daemon 进入优雅退出流程。

    Args:
        run_task: daemon 主任务。
        shutdown_state: 关停状态。
        signal_name: 触发退出的信号名。
        exit_code: 退出码。

    Returns:
        无。

    Raises:
        无。
    """

    if shutdown_state.signal_name is not None:
        return
    shutdown_state.signal_name = signal_name
    shutdown_state.exit_code = exit_code
    Log.info(f"收到 {signal_name}，WeChat daemon 正在优雅退出", module=MODULE)
    run_task.cancel()


def _install_daemon_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    run_task: asyncio.Task[None],
    shutdown_state: _DaemonShutdownState,
) -> list[signal.Signals]:
    """为 daemon 主任务安装退出信号处理器。

    Args:
        loop: 当前事件循环。
        run_task: daemon 主任务。
        shutdown_state: 关停状态。

    Returns:
        成功安装的信号列表。

    Raises:
        无。
    """

    installed_signals: list[signal.Signals] = []
    for os_signal, signal_name, exit_code in (
        (signal.SIGINT, "SIGINT", 130),
        (signal.SIGTERM, "SIGTERM", 0),
    ):
        try:
            loop.add_signal_handler(
                os_signal,
                _request_daemon_shutdown,
                run_task,
                shutdown_state,
                signal_name,
                exit_code,
            )
        except (NotImplementedError, RuntimeError, ValueError):
            continue
        installed_signals.append(os_signal)
    return installed_signals


def _remove_daemon_signal_handlers(loop: asyncio.AbstractEventLoop, installed_signals: list[signal.Signals]) -> None:
    """移除之前安装的 daemon 信号处理器。

    Args:
        loop: 当前事件循环。
        installed_signals: 已安装的信号列表。

    Returns:
        无。

    Raises:
        无。
    """

    for os_signal in installed_signals:
        with contextlib.suppress(RuntimeError, ValueError):
            loop.remove_signal_handler(os_signal)


async def _run_daemon_with_graceful_shutdown(
    daemon: _WeChatDaemonLike,
    *,
    require_existing_auth: bool,
) -> int:
    """以前台方式运行 daemon，并统一处理 SIGINT/SIGTERM。

    Args:
        daemon: 待运行的 WeChat daemon。
        require_existing_auth: 是否要求使用已有登录态启动。

    Returns:
        退出码。

    Raises:
        asyncio.CancelledError: 当任务被非预期取消时抛出。
    """

    loop = asyncio.get_running_loop()
    shutdown_state = _DaemonShutdownState()
    run_task = asyncio.create_task(daemon.run_forever(require_existing_auth=require_existing_auth))
    installed_signals = _install_daemon_signal_handlers(loop, run_task, shutdown_state)
    try:
        await run_task
        return 0
    except asyncio.CancelledError:
        if shutdown_state.signal_name is None:
            raise
        return shutdown_state.exit_code
    finally:
        _remove_daemon_signal_handlers(loop, installed_signals)
        await daemon.aclose()


async def _run_login_command(args: argparse.Namespace) -> int:
    """执行 `login` 子命令。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        无。
    """

    context = _resolve_command_context(args)
    daemon = _create_login_daemon(args, context)
    try:
        await daemon.ensure_authenticated(force_relogin=bool(getattr(args, "relogin", False)))
    finally:
        await daemon.aclose()
    Log.info("WeChat 登录态已就绪", module=MODULE)
    return 0


async def _run_run_command(args: argparse.Namespace) -> int:
    """执行 `run` 子命令。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        无。
    """

    context = _resolve_command_context(args)
    state_store = FileWeChatStateStore(context.state_dir)
    if not state_store.load().bot_token:
        Log.error(
            f"未检测到实例 {context.instance_label} 的 iLink 登录态，请先执行 `python -m dayu.wechat login --label {context.instance_label}`",
            module=MODULE,
        )
        return 1
    daemon = _create_run_daemon(args, context)
    return await _run_daemon_with_graceful_shutdown(daemon, require_existing_auth=True)


def _run_service_install_command(args: argparse.Namespace) -> int:
    """执行 `service install` 子命令。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        RuntimeError: 当平台不支持 service 管理时抛出。
    """

    context = _resolve_command_context(args)
    backend = detect_service_backend()
    spec = build_service_spec(
        state_dir=context.state_dir,
        working_directory=_resolve_repo_root(),
        python_executable=sys.executable,
        run_arguments=_build_run_cli_arguments(args, context),
        environment_variables=_collect_service_environment_variables(context),
        backend=backend,
    )
    install_service(spec)
    backend_name = _get_service_backend_display_name(spec.backend)
    print(f"已安装 {backend_name} 服务实例: {context.instance_label}")
    print(f"state_dir: {context.state_dir}")
    print(f"service_label: {spec.label}")
    print(f"定义文件路径: {spec.definition_path}")
    print(f"下一步可执行: python -m dayu.wechat service start --label {context.instance_label}")
    return 0


def _run_service_start_command(args: argparse.Namespace) -> int:
    """执行 `service start` 子命令。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        RuntimeError: 当平台不支持 service 管理时抛出。
    """

    identity = _resolve_service_identity(args)
    status = _query_installed_service_status(identity)
    if status is None:
        return 1
    if is_service_running(status):
        Log.info(
            f"{_get_service_backend_display_name(identity.backend)} 服务实例已在运行: {identity.instance_label}",
            module=MODULE,
        )
        return 0
    if not _has_persisted_wechat_login(identity.state_dir):
        Log.error(
            f"未检测到实例 {identity.instance_label} 的 iLink 登录态，请先执行 `python -m dayu.wechat login --label {identity.instance_label}`",
            module=MODULE,
        )
        return 1
    start_service(
        label=identity.label,
        definition_path=identity.definition_path,
        backend=identity.backend,
    )
    print(f"已启动 {_get_service_backend_display_name(identity.backend)} 服务实例: {identity.instance_label}")
    return 0


def _query_installed_service_status(identity: _ResolvedWechatServiceIdentity) -> ServiceStatus | None:
    """查询 WeChat service 状态并校验已安装。

    Args:
        identity: WeChat service 稳定身份。

    Returns:
        已安装时返回 service 状态；未安装时返回 `None`。

    Raises:
        RuntimeError: 当平台不支持 service 管理时抛出。
    """

    status = query_service_status(
        label=identity.label,
        definition_path=identity.definition_path,
        backend=identity.backend,
    )
    if status.installed:
        return status
    Log.error(
        f"未安装 WeChat service 实例: {identity.instance_label}，请先执行 `python -m dayu.wechat service install --label {identity.instance_label}`",
        module=MODULE,
    )
    return None


def _has_persisted_wechat_login(state_dir: Path) -> bool:
    """检查状态目录中是否存在可复用的 WeChat 登录态。

    Args:
        state_dir: WeChat 状态目录。

    Returns:
        `True` 表示存在 bot token；否则返回 `False`。

    Raises:
        无。
    """

    return bool(FileWeChatStateStore(state_dir).load().bot_token)


def _extract_cli_option_value(arguments: tuple[str, ...], option_name: str) -> str | None:
    """从命令行参数元组中提取一个选项值。

    Args:
        arguments: 参数元组。
        option_name: 选项名，例如 `--base`。

    Returns:
        选项值；未找到时返回 `None`。

    Raises:
        无。
    """

    for index, argument in enumerate(arguments):
        if argument == option_name:
            if index + 1 >= len(arguments):
                return None
            return arguments[index + 1]
        prefix = f"{option_name}="
        if argument.startswith(prefix):
            return argument.removeprefix(prefix)
    return None


def _parse_installed_service_runtime_identity(
    definition: InstalledServiceDefinition,
) -> tuple[Path, str] | None:
    """从已安装 service definition 中解析 WeChat run 的 workspace 与实例标签。

    Args:
        definition: 已安装 service definition。

    Returns:
        ``(workspace_root, instance_label)``；definition 不属于当前 WeChat run 命令时返回 `None`。

    Raises:
        无。
    """

    arguments = definition.program_arguments
    if len(arguments) < 4:
        return None
    if arguments[1:4] != ("-m", "dayu.wechat", "run"):
        return None
    run_arguments = arguments[4:]
    raw_base = _extract_cli_option_value(run_arguments, "--base")
    if raw_base is None:
        return None
    workspace_root = Path(raw_base).expanduser().resolve()
    raw_instance_label = _extract_cli_option_value(run_arguments, "--label")
    if raw_instance_label is None:
        instance_label = DEFAULT_WECHAT_INSTANCE_LABEL
    else:
        try:
            instance_label = _parse_wechat_label_argument(raw_instance_label)
        except argparse.ArgumentTypeError:
            return None
    return workspace_root, instance_label


def _list_installed_wechat_services(workspace_root: Path) -> tuple[_InstalledWechatServiceView, ...]:
    """列出当前工作区下已安装的 WeChat service 实例。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        已安装 service 视图元组。

    Raises:
        RuntimeError: 当平台不支持 service 管理时抛出。
    """

    backend = detect_service_backend()
    installed_services: list[_InstalledWechatServiceView] = []
    resolved_workspace_root = workspace_root.resolve()
    for definition in list_installed_service_definitions(backend):
        runtime_identity = _parse_installed_service_runtime_identity(definition)
        if runtime_identity is None:
            continue
        definition_workspace_root, instance_label = runtime_identity
        if definition_workspace_root != resolved_workspace_root:
            continue
        state_dir = _resolve_state_dir(resolved_workspace_root, instance_label)
        status = query_service_status(
            label=definition.label,
            definition_path=definition.definition_path,
            backend=backend,
        )
        if not status.installed:
            continue
        installed_services.append(
            _InstalledWechatServiceView(
                instance_label=instance_label,
                service_label=definition.label,
                backend=backend,
                definition_path=definition.definition_path,
                state_dir=state_dir,
                running=is_service_running(status),
                logged_in=_has_persisted_wechat_login(state_dir),
            )
        )
    installed_services.sort(key=lambda item: item.instance_label)
    return tuple(installed_services)


def _run_service_list_command(args: argparse.Namespace) -> int:
    """执行 `service list` 子命令。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        RuntimeError: 当平台不支持 service 管理时抛出。
    """

    workspace_root = _resolve_workspace_root(args.base)
    installed_services = _list_installed_wechat_services(workspace_root)
    if not installed_services:
        print("当前 workspace 未发现已安装的 WeChat service")
        return 0
    for index, service_view in enumerate(installed_services):
        print(f"instance_label: {service_view.instance_label}")
        print(f"service_label: {service_view.service_label}")
        print(f"backend: {_get_service_backend_display_name(service_view.backend)}")
        print(f"state_dir: {service_view.state_dir}")
        print(f"definition: {service_view.definition_path}")
        print(f"service: {'运行中' if service_view.running else '已安装但未运行'}")
        print(f"logged_in: {'yes' if service_view.logged_in else 'no'}")
        if index != len(installed_services) - 1:
            print("")
    return 0


def _run_service_restart_command(args: argparse.Namespace) -> int:
    """执行 `service restart` 子命令。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        RuntimeError: 当平台不支持 service 管理时抛出。
    """

    identity = _resolve_service_identity(args)
    status = _query_installed_service_status(identity)
    if status is None:
        return 1
    if not _has_persisted_wechat_login(identity.state_dir):
        Log.error(
            f"未检测到实例 {identity.instance_label} 的 iLink 登录态，请先执行 `python -m dayu.wechat login --label {identity.instance_label}`",
            module=MODULE,
        )
        return 1
    restart_service(
        label=identity.label,
        definition_path=identity.definition_path,
        backend=identity.backend,
    )
    if is_service_running(status):
        print(f"已重启 {_get_service_backend_display_name(identity.backend)} 服务实例: {identity.instance_label}")
    else:
        print(f"{_get_service_backend_display_name(identity.backend)} 服务实例未运行，已启动: {identity.instance_label}")
    return 0


def _run_service_stop_command(args: argparse.Namespace) -> int:
    """执行 `service stop` 子命令。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        RuntimeError: 当平台不支持 service 管理时抛出。
    """

    identity = _resolve_service_identity(args)
    stopped = stop_service(
        label=identity.label,
        definition_path=identity.definition_path,
        backend=identity.backend,
    )
    if stopped:
        print(f"已停止 {_get_service_backend_display_name(identity.backend)} 服务实例: {identity.instance_label}")
    else:
        print(f"{_get_service_backend_display_name(identity.backend)} 服务实例未运行: {identity.instance_label}")
    return 0


def _run_service_status_command(args: argparse.Namespace) -> int:
    """执行 `service status` 子命令。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        RuntimeError: 当平台不支持 service 管理时抛出。
    """

    identity = _resolve_service_identity(args)
    status = query_service_status(
        label=identity.label,
        definition_path=identity.definition_path,
        backend=identity.backend,
    )
    logged_in = bool(FileWeChatStateStore(identity.state_dir).load().bot_token)
    print(f"backend: {_get_service_backend_display_name(identity.backend)}")
    print(f"instance_label: {identity.instance_label}")
    print(f"service_label: {identity.label}")
    print(f"state_dir: {identity.state_dir}")
    print(f"definition: {identity.definition_path}")
    if not status.installed:
        print("service: 未安装")
    elif status.loaded:
        pid_text = str(status.pid) if status.pid is not None else "unknown"
        print(f"service: 运行中 (pid={pid_text})")
    else:
        print("service: 已安装但未运行")
    print(f"logged_in: {'yes' if logged_in else 'no'}")
    for line in build_service_log_lines(
        label=identity.label,
        state_dir=identity.state_dir,
        backend=identity.backend,
    ):
        print(line)
    return 0


def _purge_tracked_session_data(*, workspace_root: Path, state_dir: Path) -> None:
    """清理 Host DB 中与 state_dir 关联的 pending turns 和 reply outbox。

    从 state_dir 下的 tracked_sessions.json 读取 session_id 列表，
    委托 Host 层按 session_id 删除对应的 pending turns 和 reply outbox 记录。

    Args:
        workspace_root: 工作区根目录，用于定位 Host DB。
        state_dir: 当前实例的状态目录。

    Returns:
        无。

    Raises:
        无。失败时仅记录日志，不中断 uninstall 流程。
    """

    session_ids = load_tracked_session_ids(state_dir)
    if not session_ids:
        return
    host_db_path = build_host_store_default_path(workspace_root)
    if not host_db_path.exists():
        return

    from dayu.host.host_cleanup import purge_sessions_from_host_db

    total_pending, total_outbox = purge_sessions_from_host_db(
        host_db_path=host_db_path,
        session_ids=session_ids,
    )
    if total_pending or total_outbox:
        Log.info(
            f"已清理 Host DB 数据: pending_turns={total_pending}, reply_outbox={total_outbox}",
            module=MODULE,
        )


def _run_service_uninstall_command(args: argparse.Namespace) -> int:
    """执行 `service uninstall` 子命令。

    仅当系统服务定义成功卸载后，才清理 Host DB 中关联的
    pending turns / reply outbox 并删除 state_dir。
    若服务尚未安装，不执行任何清理。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        RuntimeError: 当平台不支持 service 管理时抛出。
    """

    identity = _resolve_service_identity(args)
    removed = uninstall_service(
        label=identity.label,
        definition_path=identity.definition_path,
        backend=identity.backend,
    )
    if removed:
        print(f"已卸载 {_get_service_backend_display_name(identity.backend)} 服务实例: {identity.instance_label}")

        # 清理 Host DB 中该实例关联的 pending turns 和 reply outbox
        _purge_tracked_session_data(
            workspace_root=_resolve_workspace_root(args.base),
            state_dir=identity.state_dir,
        )

        # 删除 state_dir
        if identity.state_dir.exists():
            import shutil
            shutil.rmtree(identity.state_dir, ignore_errors=True)
            Log.info(f"已删除状态目录: {identity.state_dir}", module=MODULE)
    else:
        print(f"{_get_service_backend_display_name(identity.backend)} 服务实例尚未安装: {identity.instance_label}")

    return 0


def _run_service_command(args: argparse.Namespace) -> int:
    """执行 `service` 命令分发。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        ValueError: 当子命令非法时抛出。
    """

    command = str(getattr(args, "service_command", "") or "").strip()
    if command == "install":
        return _run_service_install_command(args)
    if command == "start":
        return _run_service_start_command(args)
    if command == "restart":
        return _run_service_restart_command(args)
    if command == "stop":
        return _run_service_stop_command(args)
    if command == "status":
        return _run_service_status_command(args)
    if command == "list":
        return _run_service_list_command(args)
    if command == "uninstall":
        return _run_service_uninstall_command(args)
    raise ValueError(f"未知 service 子命令: {command}")


def _dispatch_command(args: argparse.Namespace) -> int:
    """分发 WeChat CLI 子命令。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        ValueError: 当主命令非法时抛出。
    """

    command = str(getattr(args, "command", "") or "").strip()
    if command == "login":
        return asyncio.run(_run_login_command(args))
    if command == "run":
        return asyncio.run(_run_run_command(args))
    if command == "service":
        return _run_service_command(args)
    raise ValueError(f"未知命令: {command}")


def setup_loglevel(args: argparse.Namespace) -> None:
    """设置日志级别。

    Args:
        args: argparse 解析结果。

    Returns:
        无。

    Raises:
        KeyError: 当日志级别名称非法时抛出。
    """

    if args.log_level:
        Log.set_level(LogLevel[args.log_level.upper()])
    elif args.debug:
        Log.set_level(LogLevel.DEBUG)
    elif args.verbose:
        Log.set_level(LogLevel.VERBOSE)
    elif args.info:
        Log.set_level(LogLevel.INFO)
    elif args.quiet:
        Log.set_level(LogLevel.ERROR)
    else:
        Log.set_level(LogLevel.INFO)


def main(argv: list[str] | None = None) -> int:
    """WeChat CLI 主入口。

    Args:
        argv: 可选参数列表；为空时读取命令行。

    Returns:
        退出码。

    Raises:
        无。
    """

    parser = _create_parser()
    args = parser.parse_args(argv)
    setup_loglevel(args)
    try:
        return _dispatch_command(args)
    except KeyboardInterrupt:
        Log.info("收到中断信号，WeChat daemon 正在退出", module=MODULE)
        return 130
    except Exception as exc:
        Log.error(f"WeChat 命令失败: {exc}", module=MODULE)
        return 1


__all__ = ["main"]
