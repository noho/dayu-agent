"""async daemon 的优雅退出 signal handler。

使用 ``loop.add_signal_handler`` 在 asyncio 事件循环里注册 SIGINT/SIGTERM 处理器：
- 收到信号后调 ``coordinator.settle_active_runs`` 同步收敛已登记的 run
  （cancel + 推终态 + 清理 pending turn），最后回调 ``on_signal`` 让调用方决定
  如何打断主任务（通常是 ``run_task.cancel()``）。
- 上下文退出时移除 handler，保证多次进入循环互不影响。
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
from typing import Callable, Iterator

from dayu.log import Log
from dayu.process_lifecycle.coordinator import ProcessShutdownCoordinator
from dayu.process_lifecycle.exit_codes import map_signal_to_exit_code


MODULE = "PROCESS.LIFECYCLE.ASYNC"


_SignalCallback = Callable[[str, int], None]


_TARGET_SIGNALS: tuple[str, ...] = ("SIGINT", "SIGTERM")


def _resolve_target_signals() -> list[signal.Signals]:
    """解析当前平台 daemon 需要监听的信号。"""

    resolved: list[signal.Signals] = []
    for name in _TARGET_SIGNALS:
        sig = getattr(signal, name, None)
        if isinstance(sig, signal.Signals):
            resolved.append(sig)
    return resolved


@contextlib.contextmanager
def install_async_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    coordinator: ProcessShutdownCoordinator,
    *,
    on_signal: _SignalCallback,
) -> Iterator[list[signal.Signals]]:
    """为 asyncio daemon 注册信号处理器。

    Args:
        loop: 当前 daemon 运行的 event loop。
        coordinator: 进程级协调器。
        on_signal: 回调；参数为 ``(signal_name, exit_code)``，由调用方决定如何
            打断主任务。回调内部异常会被吞掉只记录日志，避免在信号上下文里
            抛异常导致 daemon 无法退出。

    Yields:
        实际注册成功的信号列表。

    Raises:
        无。某个信号注册失败时静默降级。
    """

    installed: list[signal.Signals] = []

    def _handler(os_signal: signal.Signals) -> None:
        """signal handler：settle 当前活跃 run + 通知 daemon 退出。"""

        name = os_signal.name
        trigger = f"signal:{name}"
        coordinator.settle_active_runs(trigger=trigger)
        try:
            on_signal(name, map_signal_to_exit_code(name))
        except Exception as exc:
            Log.warn(
                f"async 优雅退出回调失败: trigger={trigger}, error={exc}",
                module=MODULE,
            )

    for os_signal in _resolve_target_signals():
        try:
            loop.add_signal_handler(os_signal, _handler, os_signal)
        except (NotImplementedError, RuntimeError, ValueError) as exc:
            Log.warn(
                f"async 优雅退出 signal handler 注册失败: signal={os_signal.name}, error={exc}",
                module=MODULE,
            )
            continue
        installed.append(os_signal)

    try:
        yield installed
    finally:
        for os_signal in installed:
            with contextlib.suppress(RuntimeError, ValueError):
                loop.remove_signal_handler(os_signal)


__all__ = ["install_async_signal_handlers"]
