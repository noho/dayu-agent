"""WeChat CLI 运行时装配与共享 helper。"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from dayu.contracts.session import SessionSource
from dayu.host.host import Host
from dayu.log import Log
from dayu.services import prepare_host_runtime_dependencies
from dayu.services.chat_service import ChatService
from dayu.services.contracts import (
    ChatPendingTurnView,
    ChatResumeRequest,
    ChatTurnRequest,
    ChatTurnSubmission,
    ReplyDeliveryFailureRequest,
    ReplyDeliverySubmitRequest,
    ReplyDeliveryView,
)
from dayu.services.reply_delivery_service import ReplyDeliveryService
from dayu.services.startup_preparation import PreparedHostRuntimeDependencies
from dayu.startup.config_file_resolver import ConfigFileResolver
from dayu.startup.config_loader import ConfigLoader
from dayu.wechat.arg_parsing import (
    DEFAULT_TYPING_INTERVAL_SEC,
    DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS,
    MODULE,
    ResolvedWechatContext,
    _parse_wechat_label_argument,
    _resolve_instance_label,
    _resolve_state_dir,
    _resolve_workspace_root,
)
from dayu.wechat.daemon import WeChatDaemon, WeChatDaemonConfig
from dayu.wechat.service_manager import (
    InstalledServiceDefinition,
    ServiceBackend,
    ServiceStatus,
    build_service_label,
    describe_service_backend,
    detect_service_backend,
    is_service_running,
    list_installed_service_definitions,
    query_service_status,
    resolve_service_definition_path,
)
from dayu.wechat.state_store import FileWeChatStateStore, WeChatDaemonState, load_tracked_session_ids
from dayu.workspace_paths import DEFAULT_WECHAT_INSTANCE_LABEL, build_host_store_default_path

def _direct_service_env_var_names() -> tuple[str, ...]:
    """返回后台 service 直接依赖的环境变量名集合。

    Args:
        无。

    Returns:
        环境变量名元组。

    Raises:
        无。
    """

    from dayu.contracts.env_keys import (
        FMP_API_KEY_ENV,
        FINS_PROCESSOR_PROFILE_ENV,
        SEC_USER_AGENT_ENV,
        SERPER_API_KEY_ENV,
        TAVILY_API_KEY_ENV,
    )

    return (
        FMP_API_KEY_ENV,
        SEC_USER_AGENT_ENV,
        SERPER_API_KEY_ENV,
        TAVILY_API_KEY_ENV,
        FINS_PROCESSOR_PROFILE_ENV,
    )


class WeChatDaemonLike(Protocol):
    """WeChat daemon 的最小可调用协议。"""

    async def ensure_authenticated(self, *, force_relogin: bool = False) -> WeChatDaemonState:
        """确保 daemon 完成登录认证。

        Args:
            force_relogin: 是否忽略现有登录态并强制重新扫码。

        Returns:
            最新的 WeChat 登录态快照。

        Raises:
            RuntimeError: 登录认证失败时抛出。
        """
        ...

    async def aclose(self) -> None:
        """关闭 daemon 持有的外部资源。

        Args:
            无。

        Returns:
            无。

        Raises:
            RuntimeError: 资源关闭失败时抛出。
        """
        ...

    async def run_forever(self, *, require_existing_auth: bool) -> None:
        """以前台方式持续运行 daemon。

        Args:
            require_existing_auth: 是否要求启动前已存在登录态。

        Returns:
            无。

        Raises:
            RuntimeError: daemon 运行失败时抛出。
        """
        ...


@dataclass(frozen=True)
class ResolvedWechatServiceIdentity:
    """WeChat service 的稳定身份信息。"""

    backend: ServiceBackend
    label: str
    definition_path: Path
    state_dir: Path
    instance_label: str = DEFAULT_WECHAT_INSTANCE_LABEL


@dataclass(frozen=True)
class InstalledWechatServiceView:
    """已安装 WeChat service 的展示视图。"""

    instance_label: str
    service_label: str
    backend: ServiceBackend
    definition_path: Path
    state_dir: Path
    running: bool
    logged_in: bool


class NoOpChatService:
    """仅供 login 命令使用的占位 ChatService。"""

    async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
        """login 命令不应调用聊天逻辑。

        Args:
            request: 聊天请求。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出，提示 login 路径不应消费聊天能力。
        """

        del request
        raise RuntimeError("login 模式不应调用 ChatService")

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        """login 命令不应调用恢复逻辑。

        Args:
            request: 恢复请求。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出，提示 login 路径不应消费恢复能力。
        """

        del request
        raise RuntimeError("login 模式不应调用 ChatService")

    def list_resumable_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
    ) -> list[ChatPendingTurnView]:
        """login 命令不应调用恢复列表逻辑。

        Args:
            session_id: 会话过滤条件。
            scene_name: scene 过滤条件。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出，提示 login 路径不应消费恢复列表能力。
        """

        del session_id
        del scene_name
        raise RuntimeError("login 模式不应调用 ChatService")

    def cleanup_stale_pending_turns(
        self,
        *,
        session_id: str | None = None,
    ) -> list[str]:
        """login 命令不需要清理 pending turn。

        Args:
            session_id: 未使用。

        Returns:
            空列表。

        Raises:
            无。
        """

        del session_id
        return []


class NoOpReplyDeliveryService:
    """仅供 login 命令使用的占位 ReplyDeliveryService。"""

    def submit_reply_for_delivery(self, request: ReplyDeliverySubmitRequest) -> ReplyDeliveryView:
        """login 模式不应调用交付逻辑。

        Args:
            request: reply delivery 提交请求。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出，提示 login 路径不应消费交付能力。
        """

        del request
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")

    def get_delivery(self, delivery_id: str) -> ReplyDeliveryView | None:
        """login 模式不应调用交付逻辑。

        Args:
            delivery_id: delivery 标识。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出，提示 login 路径不应消费交付查询能力。
        """

        del delivery_id
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")

    def list_deliveries(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: str | None = None,
    ) -> list[ReplyDeliveryView]:
        """login 模式不应调用交付逻辑。

        Args:
            session_id: 会话过滤条件。
            scene_name: scene 过滤条件。
            state: 状态过滤条件。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出，提示 login 路径不应消费交付列表能力。
        """

        del session_id
        del scene_name
        del state
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")

    def claim_delivery(self, delivery_id: str) -> ReplyDeliveryView:
        """login 模式不应调用交付逻辑。

        Args:
            delivery_id: delivery 标识。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出，提示 login 路径不应消费 claim 能力。
        """

        del delivery_id
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")

    def mark_delivery_delivered(self, delivery_id: str) -> ReplyDeliveryView:
        """login 模式不应调用交付逻辑。

        Args:
            delivery_id: delivery 标识。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出，提示 login 路径不应消费 delivered 标记能力。
        """

        del delivery_id
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")

    def mark_delivery_failed(self, request: ReplyDeliveryFailureRequest) -> ReplyDeliveryView:
        """login 模式不应调用交付逻辑。

        Args:
            request: 失败标记请求。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出，提示 login 路径不应消费 failed 标记能力。
        """

        del request
        raise RuntimeError("login 模式不应调用 ReplyDeliveryService")


def _find_installed_service_definition_for_instance(
    workspace_root: Path,
    instance_label: str,
    backend: ServiceBackend,
) -> InstalledServiceDefinition | None:
    """按 workspace 与实例标签查找已安装的 WeChat service definition。

    Args:
        workspace_root: 当前工作区根目录。
        instance_label: WeChat 实例标签。
        backend: 当前平台 service backend。

    Returns:
        匹配的 service definition；不存在时返回 ``None``。

    Raises:
        无。
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


def _resolve_service_identity(args: argparse.Namespace) -> ResolvedWechatServiceIdentity:
    """解析 WeChat service 的稳定身份。

    Args:
        args: 命令行参数对象。

    Returns:
        归一化后的 WeChat service 身份。

    Raises:
        SystemExit: 当实例标签非法时抛出。
    """

    workspace_root = _resolve_workspace_root(args.base)
    instance_label = _resolve_instance_label(getattr(args, "label", None))
    state_dir = _resolve_state_dir(workspace_root, instance_label)
    backend = detect_service_backend()
    installed_definition = _find_installed_service_definition_for_instance(workspace_root, instance_label, backend)
    if installed_definition is not None:
        return ResolvedWechatServiceIdentity(
            backend=backend,
            label=installed_definition.label,
            definition_path=installed_definition.definition_path,
            state_dir=state_dir,
            instance_label=instance_label,
        )
    label = build_service_label(state_dir)
    return ResolvedWechatServiceIdentity(
        backend=backend,
        label=label,
        definition_path=resolve_service_definition_path(label, backend=backend),
        state_dir=state_dir,
        instance_label=instance_label,
    )


def _get_service_backend_display_name(backend: ServiceBackend) -> str:
    """返回当前 service backend 的展示名。

    Args:
        backend: service backend 枚举值。

    Returns:
        面向用户的 backend 展示名。

    Raises:
        无。
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
    context: ResolvedWechatContext,
    *,
    allow_interactive_relogin: bool,
) -> WeChatDaemonConfig:
    """构建 WeChat daemon 配置。

    Args:
        args: 命令行参数对象。
        context: 已解析的共享上下文。
        allow_interactive_relogin: 是否允许 daemon 进入交互式重新登录。

    Returns:
        WeChat daemon 配置对象。

    Raises:
        无。
    """

    return WeChatDaemonConfig(
        scene_name="wechat",
        allow_interactive_relogin=allow_interactive_relogin,
        execution_options=context.execution_options,
        qrcode_timeout_sec=getattr(args, "qrcode_timeout_sec", None),
        typing_interval_sec=float(getattr(args, "typing_interval_sec", DEFAULT_TYPING_INTERVAL_SEC)),
        delivery_max_attempts=context.delivery_max_attempts,
    )


def _build_run_cli_arguments(args: argparse.Namespace, context: ResolvedWechatContext) -> list[str]:
    """构建 service 运行时的命令行参数。

    Args:
        args: 命令行参数对象。
        context: 已解析的共享上下文。

    Returns:
        用于后台 service 的命令行参数列表。

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
    if (
        context.delivery_max_attempts != DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS
        or delivery_max_attempts != DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS
    ):
        cli_arguments.extend(["--delivery-max-attempts", str(delivery_max_attempts)])
    _append_log_level_arguments(args, cli_arguments)
    _append_agent_override_arguments(args, cli_arguments)
    return cli_arguments


def _append_log_level_arguments(args: argparse.Namespace, cli_arguments: list[str]) -> None:
    """把日志参数追加到命令行参数列表。

    Args:
        args: 命令行参数对象。
        cli_arguments: 待追加的命令行参数列表。

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
        args: 命令行参数对象。
        cli_arguments: 待追加的命令行参数列表。

    Returns:
        无。

    Raises:
        无。
    """

    _append_optional_argument(cli_arguments, "--model-name", getattr(args, "model_name", None))
    _append_optional_argument(cli_arguments, "--temperature", getattr(args, "temperature", None))
    _append_optional_argument(cli_arguments, "--web-provider", getattr(args, "web_provider", None))
    if bool(getattr(args, "debug_sse", False)):
        cli_arguments.append("--debug-sse")
    if bool(getattr(args, "debug_tool_delta", False)):
        cli_arguments.append("--debug-tool-delta")
    _append_optional_argument(cli_arguments, "--debug-sse-sample-rate", getattr(args, "debug_sse_sample_rate", None))
    _append_optional_argument(cli_arguments, "--debug-sse-throttle-sec", getattr(args, "debug_sse_throttle_sec", None))
    _append_optional_argument(cli_arguments, "--tool-timeout-seconds", getattr(args, "tool_timeout_seconds", None))
    _append_optional_argument(cli_arguments, "--max-iterations", getattr(args, "max_iterations", None))
    _append_optional_argument(cli_arguments, "--fallback-mode", getattr(args, "fallback_mode", None))
    _append_optional_argument(cli_arguments, "--fallback-prompt", getattr(args, "fallback_prompt", None))
    _append_optional_argument(
        cli_arguments,
        "--max-consecutive-failed-tool-batches",
        getattr(args, "max_consecutive_failed_tool_batches", None),
    )
    _append_optional_argument(
        cli_arguments,
        "--max-duplicate-tool-calls",
        getattr(args, "max_duplicate_tool_calls", None),
    )
    _append_optional_argument(
        cli_arguments,
        "--duplicate-tool-hint-prompt",
        getattr(args, "duplicate_tool_hint_prompt", None),
    )
    if bool(getattr(args, "enable_tool_trace", False)):
        cli_arguments.append("--enable-tool-trace")
    _append_optional_argument(cli_arguments, "--tool-trace-dir", getattr(args, "tool_trace_dir", None))
    _append_optional_argument(cli_arguments, "--doc-limits-json", getattr(args, "doc_limits_json", None))
    _append_optional_argument(cli_arguments, "--fins-limits-json", getattr(args, "fins_limits_json", None))


def _collect_service_environment_variables(context: ResolvedWechatContext) -> dict[str, str]:
    """收集后台 service 需要显式注入的环境变量。

    Args:
        context: 已解析的共享上下文。

    Returns:
        需要透传给后台 service 的环境变量字典。

    Raises:
        无。
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


def _append_optional_argument(
    cli_arguments: list[str],
    flag: str,
    value: str | int | float | None,
) -> None:
    """把可选参数追加到命令行列表。

    Args:
        cli_arguments: 待追加的命令行参数列表。
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


def _create_login_daemon(args: argparse.Namespace, context: ResolvedWechatContext) -> WeChatDaemonLike:
    """构建仅用于登录的 WeChat daemon。

    Args:
        args: 命令行参数对象。
        context: 已解析的共享上下文。

    Returns:
        仅负责登录的 daemon 协议对象。

    Raises:
        无。
    """

    return WeChatDaemon(
        chat_service=NoOpChatService(),
        reply_delivery_service=NoOpReplyDeliveryService(),
        state_store=FileWeChatStateStore(context.state_dir),
        config=_build_daemon_config(args, context, allow_interactive_relogin=False),
    )


def _prepare_wechat_host_dependencies(
    context: ResolvedWechatContext,
) -> PreparedHostRuntimeDependencies:
    """准备 WeChat 的 Host 级稳定依赖。

    Args:
        context: 已解析的共享上下文。

    Returns:
        Service 层封装的共享依赖集合（包含 workspace、默认执行选项、
        scene preparation、Host 网关与 FinsRuntime）。

    Raises:
        无。
    """

    return prepare_host_runtime_dependencies(
        workspace_root=context.workspace_root,
        config_root=context.config_root,
        execution_options=context.execution_options,
        runtime_label="WeChat Host runtime",
        log_module=MODULE,
    )


def _create_run_daemon(
    args: argparse.Namespace, context: ResolvedWechatContext
) -> tuple[WeChatDaemonLike, Host]:
    """构建运行命令使用的 WeChat daemon。

    Args:
        args: 命令行参数对象。
        context: 已解析的共享上下文。

    Returns:
        ``(WeChat daemon, host)``。``host`` 用于让上层接入进程级
        ``ProcessShutdownCoordinator``，将 SIGINT/SIGTERM 收敛到统一的
        优雅退出契约。

    Raises:
        无。
    """

    prepared = _prepare_wechat_host_dependencies(context)
    scene_execution_acceptance_preparer = prepared.scene_execution_acceptance_preparer
    host = prepared.host
    fins_runtime = prepared.fins_runtime
    scene_model = scene_execution_acceptance_preparer.resolve_scene_model("wechat", context.execution_options)
    Log.info(f"工作目录: {context.workspace_root}", module=MODULE)
    Log.info(
        "wechat scene 模型: " + json.dumps(asdict(scene_model), ensure_ascii=False, sort_keys=True),
        module=MODULE,
    )
    chat_service = ChatService(
        host=host,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        company_name_resolver=fins_runtime.get_company_name,
        session_source=SessionSource.WECHAT,
    )
    reply_delivery_service = ReplyDeliveryService(host=host)
    daemon = WeChatDaemon(
        chat_service=chat_service,
        reply_delivery_service=reply_delivery_service,
        state_store=FileWeChatStateStore(context.state_dir),
        config=_build_daemon_config(args, context, allow_interactive_relogin=False),
    )
    return daemon, host


def _query_installed_service_status(identity: ResolvedWechatServiceIdentity) -> ServiceStatus | None:
    """查询 WeChat service 状态并校验已安装。

    Args:
        identity: 已解析的 service 身份。

    Returns:
        已安装时返回状态对象；未安装时返回 ``None``。

    Raises:
        无。
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
        是否存在可复用登录态。

    Raises:
        无。
    """

    return bool(FileWeChatStateStore(state_dir).load().bot_token)


def _extract_cli_option_value(arguments: tuple[str, ...], option_name: str) -> str | None:
    """从命令行参数元组中提取一个选项值。

    Args:
        arguments: 命令行参数元组。
        option_name: 待提取的选项名。

    Returns:
        对应的选项值；不存在时返回 ``None``。

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
        definition: 已安装的 service definition。

    Returns:
        ``(workspace_root, instance_label)``；不符合 WeChat run 约定时返回 ``None``。

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


def _list_installed_wechat_services(workspace_root: Path) -> tuple[InstalledWechatServiceView, ...]:
    """列出当前工作区下已安装的 WeChat service 实例。

    Args:
        workspace_root: 当前工作区根目录。

    Returns:
        当前工作区下的已安装服务视图元组。

    Raises:
        无。
    """

    backend = detect_service_backend()
    installed_services: list[InstalledWechatServiceView] = []
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
            InstalledWechatServiceView(
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


def _purge_tracked_session_data(*, workspace_root: Path, state_dir: Path) -> None:
    """清理 Host DB 中与 state_dir 关联的 pending turns 和 reply outbox。

    Args:
        workspace_root: 当前工作区根目录。
        state_dir: WeChat 状态目录。

    Returns:
        无。

    Raises:
        无。
    """

    session_ids = load_tracked_session_ids(state_dir)
    if not session_ids:
        return
    host_db_path = build_host_store_default_path(workspace_root)
    if not host_db_path.exists():
        return

    # host cleanup 仅在卸载 service 时需要，延迟导入避免影响冷启动路径。
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


__all__ = [
    "InstalledWechatServiceView",
    "NoOpChatService",
    "NoOpReplyDeliveryService",
    "ResolvedWechatServiceIdentity",
    "WeChatDaemonLike",
    "_build_daemon_config",
    "_build_run_cli_arguments",
    "_collect_service_environment_variables",
    "_create_login_daemon",
    "_create_run_daemon",
    "_direct_service_env_var_names",
    "_extract_cli_option_value",
    "_find_installed_service_definition_for_instance",
    "_get_service_backend_display_name",
    "_has_persisted_wechat_login",
    "_list_installed_wechat_services",
    "_parse_installed_service_runtime_identity",
    "_prepare_wechat_host_dependencies",
    "_purge_tracked_session_data",
    "_query_installed_service_status",
    "_resolve_repo_root",
    "_resolve_service_identity",
]
