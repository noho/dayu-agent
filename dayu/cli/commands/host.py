"""宿主管理 CLI 子命令。

提供 sessions / runs / cancel / host 四组管理子命令，
通过 `HostAdminService` 展示和管理宿主层状态。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dayu.host import resolve_host_config
from dayu.host.host import Host
from dayu.services.host_admin_service import HostAdminService
from dayu.services.protocols import HostAdminServiceProtocol
from dayu.startup.dependencies import prepare_config_file_resolver, prepare_config_loader, prepare_startup_paths
from dayu.startup.paths import StartupPaths


@dataclass(frozen=True)
class HostCliRuntime:
    """宿主管理命令使用的最小运行时。"""

    paths: StartupPaths
    host_admin_service: HostAdminServiceProtocol


def register_host_subcommands(
    subparsers: Any,
    *,
    add_global_args: Callable[[argparse.ArgumentParser], None] | None = None,
) -> None:
    """注册宿主管理子命令到 CLI 解析器。

    Args:
        subparsers: argparse 的子命令注册器。
        add_global_args: 可选全局参数注册函数。

    Returns:
        无。

    Raises:
        无。
    """

    sessions_parser = subparsers.add_parser("sessions", help="管理会话")
    if add_global_args is not None:
        add_global_args(sessions_parser)
    sessions_parser.add_argument("--all", action="store_true", dest="show_all", help="列出全部会话（含已关闭）")
    sessions_sub = sessions_parser.add_subparsers(dest="sessions_action")
    close_parser = sessions_sub.add_parser("close", help="关闭会话")
    close_parser.add_argument("session_id", help="要关闭的 session ID")

    runs_parser = subparsers.add_parser("runs", help="管理运行记录")
    if add_global_args is not None:
        add_global_args(runs_parser)
    runs_parser.add_argument("--all", action="store_true", dest="show_all", help="列出全部 run（含已完成）")
    runs_parser.add_argument("--session", dest="session_id", help="按 session 过滤")

    cancel_parser = subparsers.add_parser("cancel", help="取消运行")
    if add_global_args is not None:
        add_global_args(cancel_parser)
    cancel_parser.add_argument("run_id", nargs="?", help="要取消的 run ID")
    cancel_parser.add_argument("--session", dest="session_id", help="取消 session 下所有活跃 run")

    host_parser = subparsers.add_parser("host", help="宿主维护")
    if add_global_args is not None:
        add_global_args(host_parser)
    host_sub = host_parser.add_subparsers(dest="host_action")
    host_sub.add_parser("cleanup", help="清理孤儿 run 和过期 permit")
    host_sub.add_parser("status", help="显示宿主状态")


def run_host_command(args: argparse.Namespace) -> int:
    """分发宿主管理子命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        退出码。

    Raises:
        无。
    """

    command = args.command
    if command == "sessions":
        return _run_sessions_command(args)
    if command == "runs":
        return _run_runs_command(args)
    if command == "cancel":
        return _run_cancel_command(args)
    if command == "host":
        return _run_host_command(args)
    return 1


def _build_host_runtime(args: argparse.Namespace) -> HostCliRuntime:
    """构建宿主管理命令使用的最小运行时对象。

    Args:
        args: 命令行参数。

    Returns:
        宿主管理命令运行时。

    Raises:
        无。
    """

    workspace_root = Path(str(getattr(args, "base", "./workspace"))).expanduser().resolve()
    raw_config_root = getattr(args, "config", None)
    config_root = Path(str(raw_config_root)).expanduser().resolve() if raw_config_root else None
    paths = prepare_startup_paths(
        workspace_root=workspace_root,
        config_root=config_root,
    )
    resolver = prepare_config_file_resolver(config_root=paths.config_root)
    config_loader = prepare_config_loader(resolver=resolver)
    run_config = config_loader.load_run_config()
    host_config = resolve_host_config(
        workspace_root=paths.workspace_root,
        run_config=run_config,
        explicit_lane_config=None,
    )
    host = Host(
        host_store_path=host_config.store_path,
        lane_config=host_config.lane_config,
        pending_turn_resume_max_attempts=host_config.pending_turn_resume_max_attempts,
        event_bus=None,
    )
    return HostCliRuntime(
        paths=paths,
        host_admin_service=HostAdminService(host=host),
    )


def _resolve_host_admin_service(runtime: Any) -> HostAdminServiceProtocol:
    """从运行时对象中解析宿主管理服务。

    Args:
        runtime: 运行时对象。

    Returns:
        宿主管理服务实例。

    Raises:
        AttributeError: 运行时不包含 `host_admin_service` 时抛出。
    """

    service = getattr(runtime, "host_admin_service", None)
    if service is None:
        raise AttributeError("运行时缺少 host_admin_service")
    return service


def _run_sessions_command(args: argparse.Namespace) -> int:
    """执行 sessions 子命令。

    Args:
        args: 命令行参数。

    Returns:
        退出码。

    Raises:
        无。
    """

    runtime = _build_host_runtime(args)
    service = _resolve_host_admin_service(runtime)

    if getattr(args, "sessions_action", None) == "close":
        session_id = args.session_id
        try:
            _record, cancelled_run_ids = service.close_session(session_id)
        except KeyError:
            print(f"session 不存在: {session_id}", file=sys.stderr)
            return 1
        print(f"已关闭 session {session_id}，取消了 {len(cancelled_run_ids)} 个活跃 run")
        return 0

    if args.show_all:
        sessions = service.list_sessions()
    else:
        sessions = service.list_sessions(state="active")

    if not sessions:
        print("无会话记录")
        return 0

    header = f"{'SESSION_ID':<36} {'SOURCE':<10} {'STATE':<10} {'CREATED':<20} {'LAST_ACTIVITY':<20}"
    print(header)
    print("-" * len(header))
    for session in sessions:
        print(
            f"{session.session_id:<36} {session.source:<10} {session.state:<10} "
            f"{_format_datetime_iso(session.created_at):<20} {_format_datetime_iso(session.last_activity_at):<20}"
        )
    return 0


def _run_runs_command(args: argparse.Namespace) -> int:
    """执行 runs 子命令。

    Args:
        args: 命令行参数。

    Returns:
        退出码。

    Raises:
        无。
    """

    runtime = _build_host_runtime(args)
    service = _resolve_host_admin_service(runtime)

    if args.show_all:
        runs = service.list_runs(session_id=getattr(args, "session_id", None))
    elif getattr(args, "session_id", None):
        runs = service.list_runs(session_id=args.session_id)
    else:
        runs = service.list_runs(active_only=True)

    if not runs:
        print("无运行记录")
        return 0

    header = f"{'RUN_ID':<20} {'SERVICE':<20} {'STATE':<12} {'DURATION':<12} {'STARTED':<20}"
    print(header)
    print("-" * len(header))
    for run in runs:
        duration = _format_duration_iso(run.started_at, run.finished_at)
        print(
            f"{run.run_id:<20} {run.service_type:<20} {run.state:<12} "
            f"{duration:<12} {_format_datetime_iso(run.started_at):<20}"
        )
    return 0


def _run_cancel_command(args: argparse.Namespace) -> int:
    """执行 cancel 子命令。

    Args:
        args: 命令行参数。

    Returns:
        退出码。

    Raises:
        无。
    """

    runtime = _build_host_runtime(args)
    service = _resolve_host_admin_service(runtime)

    if getattr(args, "session_id", None):
        cancelled_run_ids = service.cancel_session_runs(args.session_id)
        print(f"已请求取消 session {args.session_id} 下 {len(cancelled_run_ids)} 个 run")
        return 0

    run_id = getattr(args, "run_id", None)
    if not run_id:
        print("请指定 run_id 或 --session", file=sys.stderr)
        return 1

    try:
        record = service.cancel_run(run_id)
    except KeyError:
        print(f"run {run_id} 不存在", file=sys.stderr)
        return 1
    if record.state != "cancelled":
        print(f"run {run_id} 不在活跃状态，无法取消", file=sys.stderr)
        return 1
    print(f"已请求取消 run {run_id}")
    return 0


def _run_host_command(args: argparse.Namespace) -> int:
    """执行 host 子命令（cleanup / status）。

    Args:
        args: 命令行参数。

    Returns:
        退出码。

    Raises:
        无。
    """

    action = getattr(args, "host_action", None)
    if not action:
        print("用法: dayu host {cleanup|status}", file=sys.stderr)
        return 1

    runtime = _build_host_runtime(args)
    service = _resolve_host_admin_service(runtime)

    if action == "cleanup":
        result = service.cleanup()
        print(f"清理完成: {len(result.orphan_run_ids)} 个孤儿 run, {len(result.stale_permit_ids)} 个过期 permit")
        return 0

    if action == "status":
        status = service.get_status()
        print(f"活跃会话: {status.active_session_count} / 总计: {status.total_session_count}")
        print(f"活跃运行: {status.active_run_count}")
        if status.active_runs_by_type:
            print("  按类型:")
            for service_type, count in status.active_runs_by_type.items():
                print(f"    {service_type}: {count}")
        if status.lane_statuses:
            print("并发通道:")
            for lane_name, snapshot in status.lane_statuses.items():
                print(f"  {lane_name}: {snapshot.active}/{snapshot.max_concurrent}")
        return 0

    return 1


def _format_datetime_iso(value: str | None) -> str:
    """把 ISO 时间文本格式化为可读字符串。

    Args:
        value: ISO 时间文本。

    Returns:
        格式化后的时间字符串。

    Raises:
        无。
    """

    if value is None:
        return "-"
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def _format_duration_iso(started_at: str | None, finished_at: str | None) -> str:
    """根据 ISO 时间文本计算可读时长。

    Args:
        started_at: 开始时间。
        finished_at: 结束时间。

    Returns:
        时长字符串或 `-`。

    Raises:
        无。
    """

    if started_at is None:
        return "-"
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(finished_at) if finished_at is not None else datetime.now(timezone.utc)
    delta = end - start
    total_seconds = int(delta.total_seconds())
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}m{seconds}s"


__all__ = ["register_host_subcommands", "run_host_command"]
