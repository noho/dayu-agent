"""WeChat `run` 子命令实现。"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass

from dayu.host.host import Host
from dayu.log import Log
from dayu.process_lifecycle import (
    ProcessShutdownCoordinator,
    install_async_signal_handlers,
)
from dayu.wechat.arg_parsing import MODULE, _resolve_command_context
from dayu.wechat.runtime import WeChatDaemonLike, _create_run_daemon
from dayu.wechat.state_store import FileWeChatStateStore


@dataclass
class _DaemonShutdownState:
    """WeChat daemon 的关停状态。

    仅供 ``_run_daemon_with_graceful_shutdown`` 内部使用，记录第一次收到的
    退出信号名与对应退出码，避免重复触发关停或丢失信号语义。
    """

    signal_name: str | None = None
    exit_code: int = 0


async def _run_daemon_with_graceful_shutdown(
    daemon: WeChatDaemonLike,
    *,
    host: Host,
    require_existing_auth: bool,
) -> int:
    """以前台方式运行 daemon，并通过统一的进程级协调器处理退出信号。

    Args:
        daemon: WeChat daemon。
        host: WeChat daemon 关联的 Host，用于让协调器执行 owner-run 收敛。
        require_existing_auth: 是否要求已有登录态。

    Returns:
        daemon 退出码；信号触发时返回信号对应的统一退出码。

    Raises:
        asyncio.CancelledError: 当主任务被外部取消且无信号语义可归因时继续向上抛出。
    """

    loop = asyncio.get_running_loop()
    shutdown_state = _DaemonShutdownState()
    coordinator = ProcessShutdownCoordinator(host=host)
    run_task = asyncio.create_task(daemon.run_forever(require_existing_auth=require_existing_auth))

    def _on_signal(signal_name: str, exit_code: int) -> None:
        """signal handler 回调：记录退出码并打断主任务。"""

        if shutdown_state.signal_name is not None:
            return
        shutdown_state.signal_name = signal_name
        shutdown_state.exit_code = exit_code
        Log.info(f"收到 {signal_name}，WeChat daemon 正在优雅退出", module=MODULE)
        run_task.cancel()

    with install_async_signal_handlers(loop, coordinator, on_signal=_on_signal):
        try:
            await run_task
            return 0
        except asyncio.CancelledError:
            if shutdown_state.signal_name is None:
                raise
            return shutdown_state.exit_code
        finally:
            await daemon.aclose()


async def _run_run_command(args: argparse.Namespace) -> int:
    """执行 `run` 子命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        命令退出码：未登录时返回 ``1``；信号触发时返回信号对应的统一退出码。

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
    daemon, host = _create_run_daemon(args, context)
    return await _run_daemon_with_graceful_shutdown(
        daemon,
        host=host,
        require_existing_auth=True,
    )


def run_run_command(args: argparse.Namespace) -> int:
    """以同步入口执行 `run` 子命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        命令退出码。

    Raises:
        无。
    """

    return asyncio.run(_run_run_command(args))


__all__ = [
    "_run_daemon_with_graceful_shutdown",
    "_run_run_command",
    "run_run_command",
]
