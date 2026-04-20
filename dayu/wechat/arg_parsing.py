"""WeChat CLI 参数定义与轻量上下文解析。"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from dayu.execution.cli_execution_options import (
    add_execution_option_arguments,
    build_execution_options_from_args,
)
from dayu.log import Log, LogLevel
from dayu.workspace_paths import DEFAULT_WECHAT_INSTANCE_LABEL, build_wechat_state_dir

if TYPE_CHECKING:
    from dayu.contracts.toolset_config import ToolsetConfigSnapshot
    from dayu.execution.options import ExecutionOptions

MODULE = "APP.WECHAT.MAIN"
DEFAULT_TYPING_INTERVAL_SEC = 8.0
DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS = 3
_WECHAT_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ResolvedWechatContext:
    """WeChat 命令解析后的共享上下文。"""

    workspace_root: Path
    config_root: Path | None
    state_dir: Path
    execution_options: "ExecutionOptions"
    delivery_max_attempts: int = DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS
    instance_label: str = DEFAULT_WECHAT_INSTANCE_LABEL


class DayuWechatArgumentParser(argparse.ArgumentParser):
    """`dayu.wechat` 顶层参数解析器。"""

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
        parser: 待扩展的参数解析器。

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
        parser: 待扩展的参数解析器。

    Returns:
        无。

    Raises:
        无。
    """

    parser.add_argument("--model-name", default=None, help="覆盖 wechat scene 的模型配置名称")
    add_execution_option_arguments(parser)


def _parse_wechat_label_argument(raw_label: str) -> str:
    """解析并校验 WeChat 实例标签。

    Args:
        raw_label: 原始标签字符串。

    Returns:
        归一化后的实例标签。

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
        parser: 待扩展的参数解析器。

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
        parser: 待扩展的参数解析器。

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
        parser: 待扩展的参数解析器。

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
        parser: 待扩展的参数解析器。

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
        parser: 待扩展的参数解析器。

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
        顶层 WeChat CLI 参数解析器。

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



def _build_execution_options(args: argparse.Namespace) -> "ExecutionOptions":
    """构建 WeChat 命令的请求级执行选项。

    Args:
        args: 命令行参数对象。

    Returns:
        请求级执行选项。

    Raises:
        SystemExit: 当工具 limits 或 temperature 参数非法时抛出。
    """

    return build_execution_options_from_args(args)


def _resolve_workspace_root(raw_base: str) -> Path:
    """解析工作区根目录。

    Args:
        raw_base: 原始工作区路径。

    Returns:
        归一化后的工作区根目录。

    Raises:
        SystemExit: 当目录不存在或不是目录时抛出。
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
        workspace_root: 已解析的工作区根目录。
        raw_config_root: 原始配置目录覆盖值。

    Returns:
        归一化后的配置目录；未显式提供时回退到 ``<workspace>/config``。

    Raises:
        无。
    """

    if raw_config_root:
        return Path(raw_config_root).expanduser().resolve()
    return (workspace_root / "config").resolve()


def _resolve_instance_label(raw_label: str | None) -> str:
    """解析 WeChat 实例标签。

    Args:
        raw_label: 原始实例标签。

    Returns:
        归一化后的实例标签。

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
        workspace_root: 工作区根目录。
        instance_label: 已解析的实例标签。

    Returns:
        当前实例对应的状态目录路径。

    Raises:
        无。
    """

    return build_wechat_state_dir(workspace_root, label=instance_label).resolve()


def _resolve_command_context(args: argparse.Namespace) -> ResolvedWechatContext:
    """解析 WeChat 命令的共享上下文。

    Args:
        args: 命令行参数对象。

    Returns:
        共享上下文对象。

    Raises:
        SystemExit: 当工作区、标签或执行选项非法时抛出。
    """

    workspace_root = _resolve_workspace_root(args.base)
    instance_label = _resolve_instance_label(getattr(args, "label", None))
    return ResolvedWechatContext(
        workspace_root=workspace_root,
        config_root=_resolve_config_root(workspace_root, getattr(args, "config", None)),
        state_dir=_resolve_state_dir(workspace_root, instance_label),
        execution_options=_build_execution_options(args),
        delivery_max_attempts=int(
            getattr(args, "delivery_max_attempts", DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS)
        ),
        instance_label=instance_label,
    )


def setup_loglevel(args: argparse.Namespace) -> None:
    """设置日志级别。

    Args:
        args: 命令行参数对象。

    Returns:
        无。

    Raises:
        无。
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


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 WeChat CLI 参数。

    Args:
        argv: 可选参数列表；为空时读取进程命令行。

    Returns:
        解析后的命令行参数对象。

    Raises:
        SystemExit: 当参数非法时抛出。
    """

    return _create_parser().parse_args(argv)


__all__ = [
    "DEFAULT_TYPING_INTERVAL_SEC",
    "DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS",
    "DayuWechatArgumentParser",
    "MODULE",
    "ResolvedWechatContext",
    "parse_arguments",
    "setup_loglevel",
]
