"""CLI 优雅退出钩子。

模块职责：
- 在 CLI 同步命令入口注册 SIGINT / SIGTERM / SIGHUP 信号处理器，
  让收到终止信号的进程能把同 owner_pid 的活跃 run 主动收敛为 CANCELLED，
  而不是留给下一次启动的 cleanup 误判为 UNSETTLED orphan。
- 通过 `atexit.register` 追加兜底，覆盖普通异常退出路径。
- 所有动作幂等；任何失败只写 Log.warn，不扩散。

不覆盖 SIGKILL / 断电等不可捕获场景，这类场景仍由
`RunRegistry.cleanup_orphan_runs` + grace period 在下次启动时收敛。
"""

from __future__ import annotations

import atexit
import contextlib
import signal
import threading
from typing import Iterator, Protocol

from dayu.log import Log


MODULE = "CLI.SHUTDOWN"


class _HostShutdownHook(Protocol):
    """Host 侧收敛协议。"""

    def shutdown_active_runs_for_owner(self) -> list[str]:
        """把本进程的活跃 run 主动收敛为 CANCELLED。"""

        ...


class _ShutdownCoordinator:
    """协调 signal / atexit 两条路径，保证只运行一次且线程安全。"""

    def __init__(self, host: _HostShutdownHook) -> None:
        """初始化协调器。

        Args:
            host: Host 聚合根。

        Returns:
            无。

        Raises:
            无。
        """

        self._host = host
        self._invoked = False
        self._lock = threading.Lock()

    def run_once(self, *, trigger: str) -> list[str]:
        """幂等地触发一次收敛。

        Args:
            trigger: 触发源标识，仅用于日志。

        Returns:
            被收敛的 run_id 列表；若已被其他路径先触发则返回空列表。

        Raises:
            无。
        """

        with self._lock:
            if self._invoked:
                return []
            self._invoked = True

        try:
            cancelled = self._host.shutdown_active_runs_for_owner()
        except Exception as exc:  # pragma: no cover - 防御 Host 内部异常
            Log.warn(
                f"CLI 优雅退出钩子触发失败: trigger={trigger}, error={exc}",
                module=MODULE,
            )
            return []
        if cancelled:
            Log.info(
                f"CLI 优雅退出收敛活跃 runs: trigger={trigger}, count={len(cancelled)}",
                module=MODULE,
            )
        return cancelled


_SHUTDOWN_SIGNALS: tuple[str, ...] = ("SIGINT", "SIGTERM", "SIGHUP")


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
    for name in _SHUTDOWN_SIGNALS:
        sig = getattr(signal, name, None)
        if isinstance(sig, signal.Signals):
            resolved.append(sig)
    return resolved


@contextlib.contextmanager
def install_cli_signal_handlers(host: _HostShutdownHook) -> Iterator[None]:
    """为 CLI 同步命令安装 SIGINT / SIGTERM / SIGHUP + atexit 优雅退出钩子。

    Args:
        host: Host 聚合根。

    Yields:
        上下文期间 host 的活跃 run 会在信号 / 正常退出时被收敛。

    Raises:
        无。无法注册 signal handler（如非主线程）时降级为 atexit only。
    """

    coordinator = _ShutdownCoordinator(host)

    def _handler(signum: int, _frame: object) -> None:
        """signal handler：收敛活跃 run 后重抛默认行为。"""

        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        coordinator.run_once(trigger=f"signal:{name}")
        # 恢复默认 handler 并再次抛给自己，保留 CLI 退出语义
        with contextlib.suppress(Exception):
            signal.signal(signum, signal.SIG_DFL)
        raise KeyboardInterrupt()

    installed: list[tuple[signal.Signals, object]] = []
    for sig in _resolve_signals():
        try:
            previous = signal.getsignal(sig)
            signal.signal(sig, _handler)
            installed.append((sig, previous))
        except (OSError, ValueError):
            continue

    def _atexit_hook() -> None:
        """atexit 兜底。"""

        coordinator.run_once(trigger="atexit")

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
        # 退出上下文时主动跑一次，覆盖 CLI 正常返回路径
        coordinator.run_once(trigger="context-exit")


_REGISTERED_HOOK_INVOKED_FLAG = "_dayu_cli_shutdown_registered"


def register_cli_shutdown_hook(host: _HostShutdownHook) -> None:
    """为 CLI 短命进程一次性挂上优雅退出钩子。

    与 `install_cli_signal_handlers` 的区别：
    - 本函数不还原 signal handler、不 unregister atexit，适合 CLI 进程
      装配完 host 后整个生命周期使用。
    - 进程内多次调用只会生效一次，避免 host 多次装配时重复注册。

    Args:
        host: Host 聚合根。

    Returns:
        无。

    Raises:
        无。注册失败仅 Log.warn。
    """

    if getattr(register_cli_shutdown_hook, _REGISTERED_HOOK_INVOKED_FLAG, False):
        return
    setattr(register_cli_shutdown_hook, _REGISTERED_HOOK_INVOKED_FLAG, True)

    coordinator = _ShutdownCoordinator(host)

    def _handler(signum: int, _frame: object) -> None:
        """signal handler：收敛活跃 run 后重抛默认行为。"""

        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        coordinator.run_once(trigger=f"signal:{name}")
        with contextlib.suppress(Exception):
            signal.signal(signum, signal.SIG_DFL)
        raise KeyboardInterrupt()

    for sig in _resolve_signals():
        try:
            signal.signal(sig, _handler)
        except (OSError, ValueError) as exc:
            Log.warn(f"CLI 优雅退出钩子注册失败: signal={sig.name}, error={exc}", module=MODULE)

    atexit.register(coordinator.run_once, trigger="atexit")


def _reset_registration_for_testing() -> None:
    """仅供测试重置一次性注册标记。"""

    if hasattr(register_cli_shutdown_hook, _REGISTERED_HOOK_INVOKED_FLAG):
        delattr(register_cli_shutdown_hook, _REGISTERED_HOOK_INVOKED_FLAG)
