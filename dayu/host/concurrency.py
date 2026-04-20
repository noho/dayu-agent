"""ConcurrencyGovernor 的 SQLite 实现。

基于 HostStore permits 表实现跨进程信号量语义。
"""

from __future__ import annotations

import os
import time
import uuid

from typing import Any

from dayu.host.host_store import HostStore
from dayu.host.protocols import ConcurrencyGovernorProtocol, ConcurrencyPermit, LaneStatus
from dayu.process_liveness import is_pid_alive

# 默认 lane 配置
DEFAULT_LANE_CONFIG: dict[str, int] = {
    "llm_api": 8,
    "sec_download": 1,
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

    def try_acquire(self, lane: str) -> ConcurrencyPermit | None:
        """尝试立即获取并发许可（非阻塞）。"""

        max_concurrent = self._lane_config.get(lane)
        if max_concurrent is None:
            raise ValueError(f"未配置的并发通道: {lane}")

        conn = self._host_store.get_connection()
        try:
            # BEGIN IMMEDIATE 保证跨进程写互斥
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM permits WHERE lane = ?",
                (lane,),
            ).fetchone()
            current_count = row["cnt"]

            if current_count >= max_concurrent:
                conn.execute("ROLLBACK")
                return None

            permit_id = f"permit_{uuid.uuid4().hex[:12]}"
            now = _now_utc()
            conn.execute(
                """
                INSERT INTO permits (permit_id, lane, owner_pid, acquired_at)
                VALUES (?, ?, ?, ?)
                """,
                (permit_id, lane, os.getpid(), now.isoformat()),
            )
            conn.execute("COMMIT")
            return ConcurrencyPermit(
                permit_id=permit_id,
                lane=lane,
                acquired_at=now,
            )
        except Exception:
            # 确保异常时回滚
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001, S110
                pass
            raise

    def release(self, permit: ConcurrencyPermit) -> None:
        """释放并发许可。"""

        conn = self._host_store.get_connection()
        conn.execute(
            "DELETE FROM permits WHERE permit_id = ?",
            (permit.permit_id,),
        )
        conn.commit()

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
            conn.executemany(
                "DELETE FROM permits WHERE permit_id = ?",
                ((permit_id,) for permit_id in stale_ids),
            )
            conn.commit()

        return stale_ids


__all__ = ["DEFAULT_LANE_CONFIG", "SQLiteConcurrencyGovernor"]
