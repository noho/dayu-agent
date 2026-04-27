"""进程优雅退出协调器。

把"收敛活跃 run"动作做成一个原子、幂等、可被信号 / atexit 多入口共享的对象。
设计要点：
- 业务侧（interactive / write / wechat run）拿到当前 run_id 后调
  ``register_active_run`` 登记；执行结束 finally 调 ``clear_active_run`` 取消登记。
- 信号 / atexit 入口调 ``settle_active_runs``：先取 observer 已登记 run_id，再合并
  ``Host.list_active_run_ids_for_current_owner`` 兜底覆盖未走 observer 路径的 run
  （如 fins 直接同步执行、interactive/prompt 在事件流首帧前的窗口期），对每个
  同步执行 ``Host.cancel_run_and_settle``。
- 因为底层 ``mark_cancelled`` 与 ``cleanup_stale_pending_turns`` 已幂等，所以
  ``settle_active_runs`` 不需要额外的"已触发"标记，多次调用安全。
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

from dayu.log import Log


MODULE = "PROCESS.LIFECYCLE"


@runtime_checkable
class HostSettleHook(Protocol):
    """Host 同步取消并收敛单个 run 的协议。"""

    def cancel_run_and_settle(self, run_id: str) -> object:
        """同步取消 run、推到 CANCELLED 终态并清理关联 pending turn。

        Args:
            run_id: 目标 run_id。

        Returns:
            实现方返回值由调用方忽略，仅用于满足 Host 真实签名。

        Raises:
            KeyError: run 不存在时由实现方抛出，协调器会吞掉只记录日志。
        """

        ...

    def list_active_run_ids_for_current_owner(self) -> list[str]:
        """列出当前进程持有的全部活跃 run_id，供协调器兜底扫描使用。

        Args:
            无。

        Returns:
            owner_pid 等于当前进程 PID 的活跃 run_id 列表。

        Raises:
            无。
        """

        ...


@runtime_checkable
class RunLifecycleObserver(Protocol):
    """业务侧观察当前持有 run_id 的协议。

    用于让 Service 层（如 ``WriteService``）在拿到 ``HostedRunContext.run_id``
    之后立即把当前 run 登记到协调器，使 Ctrl-C 路径能精确取消正在执行的 run。
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
    - 业务侧拿到当前 run_id 后调 ``register_active_run``；执行完毕 finally
      调 ``clear_active_run``。``register_active_run`` 可重入，重复登记同一
      run 视为一次。
    - 信号 / atexit 入口调 ``settle_active_runs``，对当前已登记 run 调
      ``Host.cancel_run_and_settle``。
    - ``settle_active_runs`` 同步、可重入、无幂等标记 — 因为 run 终态及
      pending turn 清理本身已幂等。
    """

    def __init__(self, host: HostSettleHook) -> None:
        """初始化协调器。

        Args:
            host: Host 聚合根，需实现 ``cancel_run_and_settle``。

        Returns:
            无。

        Raises:
            无。
        """

        self._host = host
        self._active_runs: list[str] = []
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
        """返回当前已登记 run_id 副本，仅用于测试与日志。

        Args:
            无。

        Returns:
            登记的 run_id 列表副本。

        Raises:
            无。
        """

        with self._lock:
            return list(self._active_runs)

    def settle_active_runs(self, *, trigger: str) -> list[str]:
        """同步收敛所有活跃 run（cancel + 推终态 + 清理 pending turn）。

        合并两路来源：
        1. observer 通过 ``register_active_run`` 登记的 run_id（精确路径）；
        2. ``Host.list_active_run_ids_for_current_owner()`` 兜底覆盖未登记的
           run（fins 直接同步执行、interactive/prompt 在事件流首帧前的窗口期）。

        Args:
            trigger: 触发源标识，仅用于日志。

        Returns:
            成功 settle 的 run_id 列表（保留顺序、去重，observer 路径优先）。
            某个 run 失败仅记录日志，不影响其他 run。

        Raises:
            无。
        """

        with self._lock:
            registered = list(self._active_runs)
        owner_runs: list[str] = []
        try:
            owner_runs = list(self._host.list_active_run_ids_for_current_owner())
        except Exception as exc:
            Log.warn(
                f"协调器扫描 owner 活跃 run 失败: trigger={trigger}, error={exc}",
                module=MODULE,
            )
        merged: list[str] = []
        seen: set[str] = set()
        for run_id in (*registered, *owner_runs):
            if not run_id or run_id in seen:
                continue
            seen.add(run_id)
            merged.append(run_id)
        if not merged:
            return []
        settled: list[str] = []
        for run_id in merged:
            try:
                self._host.cancel_run_and_settle(run_id)
            except Exception as exc:
                Log.warn(
                    f"协调器 settle 失败: trigger={trigger}, run_id={run_id}, error={exc}",
                    module=MODULE,
                )
                continue
            settled.append(run_id)
        if settled:
            Log.debug(
                f"协调器 settle active runs: trigger={trigger}, count={len(settled)}",
                module=MODULE,
            )
        return settled


__all__ = [
    "HostSettleHook",
    "ProcessShutdownCoordinator",
    "RunLifecycleObserver",
]
