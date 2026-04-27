"""进程优雅退出协调器实现。

负责把"取消当前活跃 run""强制收敛同 owner 剩余 run"两类动作做成
幂等、可被信号 / atexit / 上下文退出多入口共享的对象。
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from dayu.log import Log


MODULE = "PROCESS.LIFECYCLE"


@runtime_checkable
class HostCancelRunHook(Protocol):
    """Host 协作式取消单个 run 的协议。"""

    def cancel_run(self, run_id: str) -> object:
        """请求取消指定 run。

        Args:
            run_id: 目标 run_id。

        Returns:
            实现方返回值由调用方忽略，仅用于满足 Host 真实签名。

        Raises:
            KeyError: run 不存在时由实现方抛出，协调器会吞掉只记录日志。
        """

        ...


@runtime_checkable
class HostShutdownHook(Protocol):
    """Host 收敛同 owner 剩余 active run 的协议。"""

    def shutdown_active_runs_for_owner(self) -> list[str]:
        """把本进程的活跃 run 主动收敛为 CANCELLED。

        Args:
            无。

        Returns:
            被收敛的 run_id 列表。

        Raises:
            实现方异常会被协调器吞掉只记录日志。
        """

        ...


@runtime_checkable
class RunLifecycleObserver(Protocol):
    """业务侧观察当前持有 run_id 的协议。

    用于让 Service 层（如 ``WriteService``）在拿到 ``HostedRunContext.run_id``
    之后立即把当前 run 登记到协调器，使 Ctrl-C 路径能精确取消正在执行
    的 run，而不是等强收敛兜底。
    """

    def register_active_run(self, run_id: str) -> None:
        """登记当前 run。"""

        ...

    def clear_active_run(self, run_id: str) -> None:
        """清除当前 run 登记。"""

        ...


class ProcessShutdownCoordinator:
    """进程级优雅退出协调器。

    使用约定：
    - 业务侧（interactive / write / wechat run）在拿到当前 run_id 时调用
      ``register_active_run``；该方法可重入，登记多次同一个 run 视为一次。
    - 信号入口先调 ``cancel_active_runs`` 触发协作式取消，再调
      ``shutdown_owner_runs`` 兜底把进程内剩余 active run 强收敛。
    - 强收敛是幂等的，多个入口（signal handler、context exit、atexit）
      只会真正生效一次。
    """

    def __init__(
        self,
        host: HostShutdownHook,
        *,
        cancel_hook: HostCancelRunHook | None = None,
    ) -> None:
        """初始化协调器。

        Args:
            host: Host 聚合根，用于强收敛 owner 剩余 active run。
            cancel_hook: 可选的协作式取消入口；缺省时尝试复用 ``host``。
                Host 真实实现 ``cancel_run`` 与 ``shutdown_active_runs_for_owner``
                同源，但保留独立参数便于测试注入桩。

        Returns:
            无。

        Raises:
            无。
        """

        self._host = host
        self._cancel_hook: HostCancelRunHook | None
        if cancel_hook is not None:
            self._cancel_hook = cancel_hook
        elif isinstance(host, HostCancelRunHook):
            self._cancel_hook = host
        else:
            self._cancel_hook = None
        self._active_runs: list[str] = []
        self._shutdown_invoked = False
        self._lock = threading.Lock()

    def register_active_run(self, run_id: str) -> None:
        """登记当前进程持有的活跃 run。

        Args:
            run_id: Host 颁发的 run_id；空字符串视为无效，直接忽略。

        Returns:
            无。

        Raises:
            无。
        """

        if not run_id:
            return
        with self._lock:
            if run_id in self._active_runs:
                return
            self._active_runs.append(run_id)

    def clear_active_run(self, run_id: str) -> None:
        """清除指定活跃 run 登记。

        Args:
            run_id: 要清除的 run_id；不存在时静默忽略。

        Returns:
            无。

        Raises:
            无。
        """

        if not run_id:
            return
        with self._lock:
            if run_id in self._active_runs:
                self._active_runs.remove(run_id)

    def snapshot_active_runs(self) -> list[str]:
        """返回当前已登记的 run_id 副本，仅用于测试与日志。"""

        with self._lock:
            return list(self._active_runs)

    def cancel_active_runs(self, *, trigger: str) -> list[str]:
        """协作式取消所有已登记的 run。

        Args:
            trigger: 触发源标识，仅用于日志。

        Returns:
            已请求取消的 run_id 列表。``cancel_run`` 内部异常被吞掉只记录日志。

        Raises:
            无。
        """

        with self._lock:
            run_ids = list(self._active_runs)
        if not run_ids or self._cancel_hook is None:
            return []
        cancelled: list[str] = []
        for run_id in run_ids:
            try:
                self._cancel_hook.cancel_run(run_id)
            except Exception as exc:
                Log.warn(
                    f"协调器协作式取消失败: trigger={trigger}, run_id={run_id}, error={exc}",
                    module=MODULE,
                )
                continue
            cancelled.append(run_id)
        if cancelled:
            Log.debug(
                f"协调器协作式取消 active runs: trigger={trigger}, count={len(cancelled)}",
                module=MODULE,
            )
        return cancelled

    def shutdown_owner_runs(self, *, trigger: str) -> list[str]:
        """幂等地触发一次强收敛。

        Args:
            trigger: 触发源标识，仅用于日志。

        Returns:
            被收敛的 run_id 列表；若已被其他路径先触发则返回空列表。

        Raises:
            无。
        """

        with self._lock:
            if self._shutdown_invoked:
                return []
            self._shutdown_invoked = True

        try:
            cancelled = self._host.shutdown_active_runs_for_owner()
        except Exception as exc:
            Log.warn(
                f"进程优雅退出强收敛失败: trigger={trigger}, error={exc}",
                module=MODULE,
            )
            return []
        if cancelled:
            Log.debug(
                f"进程优雅退出强收敛 active runs: trigger={trigger}, count={len(cancelled)}",
                module=MODULE,
            )
        return cancelled

    def run_full_shutdown_sequence(self, *, trigger: str) -> tuple[list[str], list[str]]:
        """按"协作式取消 → 强收敛"顺序执行完整退出流程。

        Args:
            trigger: 触发源标识，仅用于日志。

        Returns:
            ``(协作式取消的 run_id 列表, 强收敛的 run_id 列表)``。

        Raises:
            无。
        """

        cancelled = self.cancel_active_runs(trigger=trigger)
        owner_cancelled = self.shutdown_owner_runs(trigger=trigger)
        return cancelled, owner_cancelled


__all__ = [
    "HostCancelRunHook",
    "HostShutdownHook",
    "ProcessShutdownCoordinator",
    "RunLifecycleObserver",
]
