"""ConcurrencyGovernor 的 SQLite 实现。

基于 HostStore permits 表实现跨进程信号量语义。
"""

from __future__ import annotations

import time
import uuid

from dayu.contracts.cancellation import CancelledError, CancellationToken
from dayu.host.host_store import HostStore, write_transaction
from dayu.host.protocols import ConcurrencyGovernorProtocol, ConcurrencyPermit, LaneStatus
from dayu.process_liveness import OwnerIdentity, current_owner_identity, is_owner_identity_alive

# Host 自治 lane 名称：所有 Agent 执行路径都会自动叠加该 lane。
# Service 层禁止使用该字面量，也不允许在 business_concurrency_lane 中写入该值。
HOST_AGENT_LANE: str = "llm_api"

# 默认 lane 配置：仅保留 Host 自治 lane；业务 lane 默认值由 Service 启动期注入。
DEFAULT_LANE_CONFIG: dict[str, int] = {
    HOST_AGENT_LANE: 8,
}

# 轮询间隔（秒）
_POLL_INTERVAL = 0.1

# 在阻塞等待 permit 时，最多每隔这么久做一次 dead-PID stale permit 回收。
_STALE_PERMIT_REAP_INTERVAL_SECONDS = 5.0


from dayu.host._datetime_utils import now_utc as _now_utc


class SQLiteConcurrencyGovernor(ConcurrencyGovernorProtocol):
    """基于 SQLite 的跨进程并发治理实现。

    使用 BEGIN IMMEDIATE 事务保证跨进程互斥，
    轮询等待直到获得许可或超时。
    """

    def __init__(
        self,
        host_store: HostStore,
        lane_config: dict[str, int] | None = None,
    ) -> None:
        """初始化 ConcurrencyGovernor。

        Args:
            host_store: 共享 SQLite 存储。
            lane_config: lane 名到最大并发数的映射，默认使用 DEFAULT_LANE_CONFIG。
        """

        self._host_store = host_store
        self._lane_config = lane_config or dict(DEFAULT_LANE_CONFIG)

    def acquire(
        self,
        lane: str,
        *,
        timeout: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ConcurrencyPermit:
        """获取并发许可，超时前轮询等待。

        Args:
            lane: 目标并发通道名。
            timeout: 最大等待秒数；`None` 表示无限等待。
            cancellation_token: 可选取消令牌；等待期间若被触发，则立即结束等待。

        Returns:
            成功获取到的并发许可。

        Raises:
            TimeoutError: 达到等待超时仍未拿到 permit。
            CancelledError: 等待期间收到取消请求。
            ValueError: lane 未配置时由 `try_acquire()` 透传。
        """

        deadline = _build_deadline(timeout)
        last_reap_started_at = time.monotonic()
        while True:
            _raise_if_cancelled(cancellation_token)
            permit = self.try_acquire(lane)
            if permit is not None:
                return permit
            if _is_deadline_reached(deadline):
                raise TimeoutError(
                    f"获取并发许可超时: lane={lane}, timeout={timeout}s"
                )
            last_reap_started_at = self._maybe_reap_stale_permits(last_reap_started_at)
            _wait_for_retry(deadline=deadline, cancellation_token=cancellation_token)

    def acquire_many(
        self,
        lanes: list[str],
        *,
        timeout: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[ConcurrencyPermit]:
        """原子获取多 lane 许可：单事务内全部检查+全部写入，要么全拿要么全不拿。

        单 lane 场景退化为单 INSERT，与 :meth:`try_acquire` 语义一致。

        Args:
            lanes: 待申请的 lane 名列表。
            timeout: 最大等待秒数；`None` 表示无限等待。
            cancellation_token: 可选取消令牌；等待期间若被触发，则立即结束等待。

        Returns:
            与 `lanes` 同序的 permit 列表。

        Raises:
            TimeoutError: 达到等待超时仍未拿到 permit。
            CancelledError: 等待期间收到取消请求。
            ValueError: 任一 lane 未配置时抛出。
        """

        if not lanes:
            return []
        for lane_name in lanes:
            if lane_name not in self._lane_config:
                raise ValueError(f"未配置的并发通道: {lane_name}")

        deadline = _build_deadline(timeout)
        last_reap_started_at = time.monotonic()
        while True:
            _raise_if_cancelled(cancellation_token)
            permits = self._try_acquire_many(lanes)
            if permits is not None:
                return permits
            if _is_deadline_reached(deadline):
                raise TimeoutError(
                    f"获取多 lane 并发许可超时: lanes={lanes}, timeout={timeout}s"
                )
            last_reap_started_at = self._maybe_reap_stale_permits(last_reap_started_at)
            _wait_for_retry(deadline=deadline, cancellation_token=cancellation_token)

    def _try_acquire_many(self, lanes: list[str]) -> list[ConcurrencyPermit] | None:
        """在单个 BEGIN IMMEDIATE 事务内尝试一次性拿齐全部 lane。

        Args:
            lanes: 已校验为合法 lane 名的列表。

        Returns:
            全部 lane 都有额度时返回 permit 列表（与 ``lanes`` 同序）；
            任一 lane 额度不足时返回 ``None`` 并回滚事务。

        Raises:
            sqlite3.Error: SQLite 层异常会原样抛出；事务已回滚。
        """

        conn = self._host_store.get_connection()
        permits: list[ConcurrencyPermit] = []
        over_capacity = False
        with write_transaction(conn):
            # 先统一点名：任意一个不够就全部放弃。
            # 不同 lane 的 COUNT 查询在同一事务内读到的是一致快照，
            # 避免"先拿到 A、检查 B 时 A 的名额被抢"的逻辑漏洞。
            for lane_name in lanes:
                max_concurrent = self._lane_config[lane_name]
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM permits WHERE lane = ?",
                    (lane_name,),
                ).fetchone()
                if row["cnt"] >= max_concurrent:
                    over_capacity = True
                    break

            if not over_capacity:
                now = _now_utc()
                now_iso = now.isoformat()
                owner = current_owner_identity()
                for lane_name in lanes:
                    permit_id = f"permit_{uuid.uuid4().hex[:12]}"
                    conn.execute(
                        """
                        INSERT INTO permits (permit_id, lane, owner_pid,
                                             owner_process_start_time, owner_boot_id,
                                             acquired_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            permit_id,
                            lane_name,
                            owner.pid,
                            owner.process_start_time,
                            owner.boot_id,
                            now_iso,
                        ),
                    )
                    permits.append(
                        ConcurrencyPermit(
                            permit_id=permit_id,
                            lane=lane_name,
                            acquired_at=now,
                        )
                    )
        if over_capacity:
            return None
        return permits

    def try_acquire(self, lane: str) -> ConcurrencyPermit | None:
        """尝试立即获取并发许可（非阻塞）。"""

        max_concurrent = self._lane_config.get(lane)
        if max_concurrent is None:
            raise ValueError(f"未配置的并发通道: {lane}")

        conn = self._host_store.get_connection()
        permit: ConcurrencyPermit | None = None
        with write_transaction(conn):
            # write_transaction(BEGIN IMMEDIATE) 保证跨进程写互斥
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM permits WHERE lane = ?",
                (lane,),
            ).fetchone()
            current_count = row["cnt"]

            if current_count < max_concurrent:
                permit_id = f"permit_{uuid.uuid4().hex[:12]}"
                now = _now_utc()
                owner = current_owner_identity()
                conn.execute(
                    """
                    INSERT INTO permits (permit_id, lane, owner_pid,
                                         owner_process_start_time, owner_boot_id,
                                         acquired_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        permit_id,
                        lane,
                        owner.pid,
                        owner.process_start_time,
                        owner.boot_id,
                        now.isoformat(),
                    ),
                )
                permit = ConcurrencyPermit(
                    permit_id=permit_id,
                    lane=lane,
                    acquired_at=now,
                )
        return permit

    def release(self, permit: ConcurrencyPermit) -> None:
        """释放并发许可。"""

        conn = self._host_store.get_connection()
        with write_transaction(conn):
            conn.execute(
                "DELETE FROM permits WHERE permit_id = ?",
                (permit.permit_id,),
            )

    def get_lane_status(self, lane: str) -> LaneStatus:
        """查询指定 lane 的当前状态。"""

        max_concurrent = self._lane_config.get(lane, 0)
        conn = self._host_store.get_connection()
        row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM permits WHERE lane = ?",
            (lane,),
        ).fetchone()
        return LaneStatus(
            lane=lane,
            max_concurrent=max_concurrent,
            active=row["cnt"],
        )

    def get_all_status(self) -> dict[str, LaneStatus]:
        """查询所有 lane 的当前状态。"""

        result: dict[str, LaneStatus] = {}
        for lane_name in self._lane_config:
            result[lane_name] = self.get_lane_status(lane_name)
        return result

    def cleanup_stale_permits(self) -> list[str]:
        """清理 owner identity 已死亡的 permit。

        以 ``OwnerIdentity`` 三元组判活，闭合 PID 复用窗口；任一字段
        采集失败为 ``NULL`` 时退化为剩余字段比对，最差等价于仅 PID 判活。
        """

        conn = self._host_store.get_connection()
        rows = conn.execute(
            "SELECT permit_id, owner_pid, owner_process_start_time, owner_boot_id FROM permits"
        ).fetchall()

        stale_ids: list[str] = []
        for row in rows:
            stored = OwnerIdentity(
                pid=int(row["owner_pid"]),
                process_start_time=row["owner_process_start_time"],
                boot_id=row["owner_boot_id"],
            )
            if not is_owner_identity_alive(stored):
                stale_ids.append(row["permit_id"])

        if stale_ids:
            with write_transaction(conn):
                conn.executemany(
                    "DELETE FROM permits WHERE permit_id = ?",
                    ((permit_id,) for permit_id in stale_ids),
                )

        return stale_ids

    def _maybe_reap_stale_permits(self, last_reap_started_at: float) -> float:
        """在阻塞等待期间按节流频率回收 dead-PID stale permit。

        Args:
            last_reap_started_at: 上一次启动回收时的 monotonic 时间。

        Returns:
            最新一次回收启动时间；若本轮未触发回收，则原样返回入参。

        Raises:
            无。回收失败时静默降级到下一轮重试，避免把治理噪声放大成业务失败。
        """

        now = time.monotonic()
        if now - last_reap_started_at < _STALE_PERMIT_REAP_INTERVAL_SECONDS:
            return last_reap_started_at
        try:
            self.cleanup_stale_permits()
        except Exception:
            return now
        return now


def _build_deadline(timeout: float | None) -> float | None:
    """根据超时秒数构造 monotonic deadline。

    Args:
        timeout: 最大等待秒数；`None` 表示无限等待。

    Returns:
        绝对 monotonic deadline；无限等待时返回 `None`。

    Raises:
        无。
    """

    if timeout is None:
        return None
    return time.monotonic() + timeout


def _is_deadline_reached(deadline: float | None) -> bool:
    """判断当前等待是否已经到达 deadline。

    Args:
        deadline: 绝对 monotonic deadline；`None` 表示无限等待。

    Returns:
        已到达或超过 deadline 时返回 `True`，否则返回 `False`。

    Raises:
        无。
    """

    return deadline is not None and time.monotonic() >= deadline


def _raise_if_cancelled(cancellation_token: CancellationToken | None) -> None:
    """在取消令牌已触发时抛出协作式取消异常。

    Args:
        cancellation_token: 可选取消令牌。

    Returns:
        无。

    Raises:
        CancelledError: 取消令牌已触发时抛出。
    """

    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()


def _wait_for_retry(
    *,
    deadline: float | None,
    cancellation_token: CancellationToken | None,
) -> None:
    """在下一轮 permit 尝试前等待一小段时间，并对取消保持敏感。

    Args:
        deadline: 当前等待的绝对超时边界；`None` 表示无限等待。
        cancellation_token: 可选取消令牌。

    Returns:
        无。

    Raises:
        CancelledError: 等待期间收到取消请求。
    """

    wait_seconds = _POLL_INTERVAL
    if deadline is not None:
        remaining_seconds = max(0.0, deadline - time.monotonic())
        wait_seconds = min(wait_seconds, remaining_seconds)
    if wait_seconds <= 0:
        return
    if cancellation_token is None:
        time.sleep(wait_seconds)
        return
    if cancellation_token.wait(wait_seconds):
        raise CancelledError("操作已被取消")


__all__ = ["DEFAULT_LANE_CONFIG", "HOST_AGENT_LANE", "SQLiteConcurrencyGovernor"]
