"""sync CLI 的优雅退出 signal handler。

提供单一 API ``register_process_shutdown_hook(coordinator, *, interactive)``：
- 进程内一次性注册 SIGINT/SIGTERM/SIGHUP + atexit 钩子；重复调用 no-op。
- SIGINT 行为按 ``interactive`` 分派：
    * ``interactive=True``  → ``settle_active_runs`` + ``raise KeyboardInterrupt``
      （不还原 SIG_DFL，REPL 多次 Ctrl+C 都命中自定义 handler）；
    * ``interactive=False`` → ``settle_active_runs`` + ``raise SystemExit(EXIT_CODE_SIGINT)``
      让短命命令直接退出，避免业务层 try/except 形态吞掉 KeyboardInterrupt。
- SIGTERM/SIGHUP 一律 ``settle_active_runs`` + 还原 SIG_DFL +
  ``raise SystemExit(map_signal_to_exit_code(name))``。
- atexit 钩子调 ``settle_active_runs(trigger="atexit")`` 兜底。
"""

from __future__ import annotations

import atexit
import contextlib
import signal
import threading

from dayu.log import Log
from dayu.process_lifecycle.coordinator import ProcessShutdownCoordinator
from dayu.process_lifecycle.exit_codes import EXIT_CODE_SIGINT, map_signal_to_exit_code


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
    """把信号编号映射回名称。

    Args:
        signum: 数字形式的信号编号。

    Returns:
        ``signal.Signals(signum).name``；若无法映射回名称则返回数字字符串。

    Raises:
        无。
    """

    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


_REGISTRATION_LOCK = threading.Lock()
_REGISTERED = False


def register_process_shutdown_hook(
    coordinator: ProcessShutdownCoordinator,
    *,
    interactive: bool,
) -> None:
    """sync CLI 进程级一次性注册 SIGINT/SIGTERM/SIGHUP + atexit 钩子。

    Args:
        coordinator: 进程级协调器。
        interactive: 是否 interactive 命令。
            ``True`` 时 SIGINT 抛 ``KeyboardInterrupt`` 让 REPL 循环继续，
            ``False`` 时 SIGINT 抛 ``SystemExit(EXIT_CODE_SIGINT)`` 直接退出。

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
        """signal handler：先 settle 所有活跃 run，再按信号语义退出进程。

        SIGINT 的 KeyboardInterrupt / SystemExit 由 ``interactive`` 决定；
        SIGTERM/SIGHUP 一律还原 SIG_DFL 后 ``SystemExit``。
        """

        name = _resolve_signal_name(signum)
        trigger = f"signal:{name}"
        coordinator.settle_active_runs(trigger=trigger)
        if name == "SIGINT":
            if interactive:
                # 不还原 SIG_DFL：interactive REPL 期望第二次 Ctrl+C 仍命中自定义 handler。
                raise KeyboardInterrupt()
            raise SystemExit(EXIT_CODE_SIGINT)
        with contextlib.suppress(Exception):
            signal.signal(signum, signal.SIG_DFL)
        raise SystemExit(map_signal_to_exit_code(name))

    for sig in _resolve_signals():
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError) as exc:
            Log.warn(
                f"sync 优雅退出 signal handler 注册失败: signal={sig.name}, error={exc}",
                module=MODULE,
            )

    def _atexit_hook() -> None:
        """atexit 兜底：进程退出前再 settle 一次。"""

        coordinator.settle_active_runs(trigger="atexit")

    atexit.register(_atexit_hook)


def _reset_registration_for_testing() -> None:
    """仅供测试重置一次性注册标记。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """

    global _REGISTERED
    with _REGISTRATION_LOCK:
        _REGISTERED = False


__all__ = [
    "register_process_shutdown_hook",
]
