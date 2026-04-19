"""WeChat `run` 子命令实现。"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
from dataclasses import dataclass

from dayu.log import Log
from dayu.wechat.arg_parsing import MODULE, _resolve_command_context
from dayu.wechat.runtime import WeChatDaemonLike, _create_run_daemon
from dayu.wechat.state_store import FileWeChatStateStore


@dataclass
class DaemonShutdownState:
    """WeChat daemon 的关停状态。"""

    signal_name: str | None = None
    exit_code: int = 0


def _request_daemon_shutdown(
    run_task: asyncio.Task[None],
    shutdown_state: DaemonShutdownState,
    signal_name: str,
    exit_code: int,
) -> None:
    """请求 daemon 进入优雅退出流程。"""

    if shutdown_state.signal_name is not None:
        return
    shutdown_state.signal_name = signal_name
    shutdown_state.exit_code = exit_code
    Log.info(f"收到 {signal_name}，WeChat daemon 正在优雅退出", module=MODULE)
    run_task.cancel()


def _install_daemon_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    run_task: asyncio.Task[None],
    shutdown_state: DaemonShutdownState,
) -> list[signal.Signals]:
    """为 daemon 主任务安装退出信号处理器。"""

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
    """移除之前安装的 daemon 信号处理器。"""

    for os_signal in installed_signals:
        with contextlib.suppress(RuntimeError, ValueError):
            loop.remove_signal_handler(os_signal)


async def _run_daemon_with_graceful_shutdown(
    daemon: WeChatDaemonLike,
    *,
    require_existing_auth: bool,
) -> int:
    """以前台方式运行 daemon，并统一处理 SIGINT/SIGTERM。"""

    loop = asyncio.get_running_loop()
    shutdown_state = DaemonShutdownState()
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


async def _run_run_command(args: argparse.Namespace) -> int:
    """执行 `run` 子命令。"""

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


def run_run_command(args: argparse.Namespace) -> int:
    """以同步入口执行 `run` 子命令。"""

    return asyncio.run(_run_run_command(args))


__all__ = [
    "DaemonShutdownState",
    "_install_daemon_signal_handlers",
    "_remove_daemon_signal_handlers",
    "_request_daemon_shutdown",
    "_run_daemon_with_graceful_shutdown",
    "_run_run_command",
    "run_run_command",
]
