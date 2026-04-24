"""ConcurrencyGovernor 的 SQLite 实现。

基于 HostStore permits 表实现跨进程信号量语义。
"""

from __future__ import annotations

import os
import time
import uuid

from dayu.host.host_store import HostStore, write_transaction
from dayu.host.protocols import ConcurrencyGovernorProtocol, ConcurrencyPermit, LaneStatus
from dayu.process_liveness import is_pid_alive

# Host 自治 lane 名称：所有 Agent 执行路径都会自动叠加该 lane。
# Service 层禁止使用该字面量，也不允许在 business_concurrency_lane 中写入该值。
HOST_AGENT_LANE: str = "llm_api"

# 默认 lane 配置：仅保留 Host 自治 lane；业务 lane 默认值由 Service 启动期注入。
DEFAULT_LANE_CONFIG: dict[str, int] = {
    HOST_AGENT_LANE: 8,
}

# 轮询间隔（秒）
_POLL_INTERVAL = 0.1


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

    def acquire(self, lane: str, *, timeout: float | None = None) -> ConcurrencyPermit:
        """获取并发许可，超时前轮询等待。"""

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            permit = self.try_acquire(lane)
            if permit is not None:
                return permit
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"获取并发许可超时: lane={lane}, timeout={timeout}s"
                )
            time.sleep(_POLL_INTERVAL)

    def acquire_many(
        self,
        lanes: list[str],
        *,
        timeout: float | None = None,
    ) -> list[ConcurrencyPermit]:
        """原子获取多 lane 许可：单事务内全部检查+全部写入，要么全拿要么全不拿。

        单 lane 场景退化为单 INSERT，与 :meth:`try_acquire` 语义一致。
        """

        if not lanes:
            return []
        for lane_name in lanes:
            if lane_name not in self._lane_config:
                raise ValueError(f"未配置的并发通道: {lane_name}")

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            permits = self._try_acquire_many(lanes)
            if permits is not None:
                return permits
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"获取多 lane 并发许可超时: lanes={lanes}, timeout={timeout}s"
                )
            time.sleep(_POLL_INTERVAL)

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
                pid = os.getpid()
                for lane_name in lanes:
                    permit_id = f"permit_{uuid.uuid4().hex[:12]}"
                    conn.execute(
                        """
                        INSERT INTO permits (permit_id, lane, owner_pid, acquired_at)
                        VALUES (?, ?, ?, ?)
                        """,
                        (permit_id, lane_name, pid, now_iso),
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
                conn.execute(
                    """
                    INSERT INTO permits (permit_id, lane, owner_pid, acquired_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (permit_id, lane, os.getpid(), now.isoformat()),
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
        """清理 owner_pid 已死亡的 permit。"""

        conn = self._host_store.get_connection()
        rows = conn.execute("SELECT permit_id, owner_pid FROM permits").fetchall()

        stale_ids: list[str] = []
        for row in rows:
            if not is_pid_alive(row["owner_pid"]):
                stale_ids.append(row["permit_id"])

        if stale_ids:
            with write_transaction(conn):
                conn.executemany(
                    "DELETE FROM permits WHERE permit_id = ?",
                    ((permit_id,) for permit_id in stale_ids),
                )

        return stale_ids


__all__ = ["DEFAULT_LANE_CONFIG", "HOST_AGENT_LANE", "SQLiteConcurrencyGovernor"]
