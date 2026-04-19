"""WeChat `service` 子命令实现。"""

from __future__ import annotations

import argparse
import shutil
import sys

from dayu.log import Log
from dayu.wechat.arg_parsing import MODULE, _resolve_command_context, _resolve_workspace_root
from dayu.wechat.runtime import (
    _build_run_cli_arguments,
    _collect_service_environment_variables,
    _get_service_backend_display_name,
    _has_persisted_wechat_login,
    _list_installed_wechat_services,
    _purge_tracked_session_data,
    _query_installed_service_status,
    _resolve_repo_root,
    _resolve_service_identity,
)
from dayu.wechat.service_manager import (
    build_service_log_lines,
    build_service_spec,
    detect_service_backend,
    install_service,
    is_service_running,
    query_service_status,
    restart_service,
    start_service,
    stop_service,
    uninstall_service,
)
from dayu.wechat.state_store import FileWeChatStateStore


def _run_service_install_command(args: argparse.Namespace) -> int:
    """执行 `service install` 子命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        退出码，成功时返回 ``0``。

    Raises:
        RuntimeError: 当前平台不支持用户级 service，或安装 service 失败时抛出。
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
    """执行 `service start` 子命令。"""

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


def _run_service_list_command(args: argparse.Namespace) -> int:
    """执行 `service list` 子命令。"""

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
    """执行 `service restart` 子命令。"""

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
    """执行 `service stop` 子命令。"""

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
    """执行 `service status` 子命令。"""

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


def _run_service_uninstall_command(args: argparse.Namespace) -> int:
    """执行 `service uninstall` 子命令。"""

    identity = _resolve_service_identity(args)
    removed = uninstall_service(
        label=identity.label,
        definition_path=identity.definition_path,
        backend=identity.backend,
    )
    if removed:
        print(f"已卸载 {_get_service_backend_display_name(identity.backend)} 服务实例: {identity.instance_label}")
        _purge_tracked_session_data(
            workspace_root=_resolve_workspace_root(args.base),
            state_dir=identity.state_dir,
        )
        if identity.state_dir.exists():
            shutil.rmtree(identity.state_dir, ignore_errors=True)
            Log.info(f"已删除状态目录: {identity.state_dir}", module=MODULE)
    else:
        print(f"{_get_service_backend_display_name(identity.backend)} 服务实例尚未安装: {identity.instance_label}")
    return 0


def run_service_command(args: argparse.Namespace) -> int:
    """执行 `service` 命令分发。"""

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


__all__ = [
    "_run_service_install_command",
    "_run_service_list_command",
    "_run_service_restart_command",
    "_run_service_start_command",
    "_run_service_status_command",
    "_run_service_stop_command",
    "_run_service_uninstall_command",
    "run_service_command",
]
