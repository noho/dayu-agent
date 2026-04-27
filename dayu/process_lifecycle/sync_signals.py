"""sync CLI 的优雅退出 signal handler。

提供两条 API：
- ``install_sync_signal_handlers``：上下文管理器，进入时安装 SIGINT/SIGTERM/SIGHUP
  + atexit 钩子，退出时还原。适合单元测试或 ``cli interactive`` 这类长生命周期。
- ``register_process_shutdown_hook``：进程内一次性注册，退出时不还原。适合短命
  CLI 的装配阶段挂上协调器。

两条 API 共用同一个 ``ProcessShutdownCoordinator``：handler 触发时先调
``cancel_active_runs`` 协作式取消正在执行的 run，再调 ``shutdown_owner_runs``
兜底强收敛，最后抛 ``KeyboardInterrupt`` 维持 sync CLI 的退出语义。
"""

from __future__ import annotations

import atexit
import contextlib
import signal
import threading
from typing import Iterator

from dayu.log import Log
from dayu.process_lifecycle.coordinator import ProcessShutdownCoordinator
from dayu.process_lifecycle.exit_codes import map_signal_to_exit_code


MODULE = "PROCESS.LIFECYCLE.SYNC"

_SHUTDOWN_SIGNAL_NAMES: tuple[str, ...] = ("SIGINT", "SIGTERM", "SIGHUP")


def _resolve_signals() -> list[signal.Signals]:
    """解析当前平台可用的退出信号。

    Args:
        无。

    Returns:
        平台上真正存在的信号对象列表。

    Raises:
        无。
    """

    resolved: list[signal.Signals] = []
    for name in _SHUTDOWN_SIGNAL_NAMES:
        sig = getattr(signal, name, None)
        if isinstance(sig, signal.Signals):
            resolved.append(sig)
    return resolved


def _resolve_signal_name(signum: int) -> str:
    """把信号编号映射回名称。"""

    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


@contextlib.contextmanager
def install_sync_signal_handlers(
    coordinator: ProcessShutdownCoordinator,
) -> Iterator[None]:
    """为 sync CLI 安装信号处理器与 atexit 兜底。

    Args:
        coordinator: 进程级协调器。

    Yields:
        上下文期间 SIGINT/SIGTERM/SIGHUP 与 atexit 都会触发协调器的完整退出流程。

    Raises:
        无。无法注册某个信号（如非主线程）时静默降级，atexit 仍会兜底。
    """

    def _handler(signum: int, _frame: object) -> None:
        """signal handler：先协作式取消，再强收敛，最后按信号语义退出进程。

        SIGINT 抛 ``KeyboardInterrupt`` 维持 sync REPL 退出语义；SIGTERM/SIGHUP
        视为外部明确停止信号，直接 ``SystemExit`` 让进程立即退出，避免回到
        prompt_toolkit REPL 循环继续等待输入。
        """

        name = _resolve_signal_name(signum)
        trigger = f"signal:{name}"
        coordinator.run_full_shutdown_sequence(trigger=trigger)
        with contextlib.suppress(Exception):
            signal.signal(signum, signal.SIG_DFL)
        if name == "SIGINT":
            raise KeyboardInterrupt()
        raise SystemExit(map_signal_to_exit_code(name))

    installed: list[tuple[signal.Signals, object]] = []
    for sig in _resolve_signals():
        try:
            previous = signal.getsignal(sig)
            signal.signal(sig, _handler)
            installed.append((sig, previous))
        except (OSError, ValueError):
            continue

    def _atexit_hook() -> None:
        """atexit 兜底：只跑强收敛，cancel_active 此时已无意义。"""

        coordinator.shutdown_owner_runs(trigger="atexit")

    atexit.register(_atexit_hook)

    try:
        yield
    finally:
        for sig, previous in installed:
            with contextlib.suppress(OSError, ValueError):
                if previous is None:
                    signal.signal(sig, signal.SIG_DFL)
                else:
                    signal.signal(sig, previous)  # type: ignore[arg-type]
        with contextlib.suppress(Exception):
            atexit.unregister(_atexit_hook)
        # 上下文退出再跑一次强收敛，覆盖 CLI 正常返回路径。
        coordinator.shutdown_owner_runs(trigger="context-exit")


_REGISTRATION_LOCK = threading.Lock()
_REGISTERED = False


def register_process_shutdown_hook(coordinator: ProcessShutdownCoordinator) -> None:
    """为短命 sync CLI 一次性挂上优雅退出钩子。

    与 ``install_sync_signal_handlers`` 的差异：
    - 不还原 signal handler、不 unregister atexit。
    - 进程内重复调用只会生效一次。

    Args:
        coordinator: 进程级协调器。

    Returns:
        无。

    Raises:
        无。注册某个信号失败仅 ``Log.warn``。
    """

    global _REGISTERED
    with _REGISTRATION_LOCK:
        if _REGISTERED:
            return
        _REGISTERED = True

    def _handler(signum: int, _frame: object) -> None:
        """signal handler：协作式取消 + 强收敛 + 按信号语义退出进程。"""

        name = _resolve_signal_name(signum)
        trigger = f"signal:{name}"
        coordinator.run_full_shutdown_sequence(trigger=trigger)
        with contextlib.suppress(Exception):
            signal.signal(signum, signal.SIG_DFL)
        if name == "SIGINT":
            raise KeyboardInterrupt()
        raise SystemExit(map_signal_to_exit_code(name))

    for sig in _resolve_signals():
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError) as exc:
            Log.warn(
                f"sync 优雅退出 signal handler 注册失败: signal={sig.name}, error={exc}",
                module=MODULE,
            )

    atexit.register(coordinator.shutdown_owner_runs, trigger="atexit")


def _reset_registration_for_testing() -> None:
    """仅供测试重置一次性注册标记。"""

    global _REGISTERED
    with _REGISTRATION_LOCK:
        _REGISTERED = False


__all__ = [
    "install_sync_signal_handlers",
    "register_process_shutdown_hook",
]
