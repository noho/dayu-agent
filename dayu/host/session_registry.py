"""SessionRegistry 的 SQLite 实现。

基于 HostStore 提供跨进程可见的 session 生命周期管理。
"""

from __future__ import annotations

import json
import uuid
from datetime import timedelta
from typing import Any

from dayu.contracts.execution_metadata import ExecutionDeliveryContext, normalize_execution_delivery_context
from dayu.contracts.session import SessionRecord, SessionSource, SessionState
from dayu.host.host_store import HostStore, write_transaction
from dayu.host.protocols import SessionRegistryProtocol, SessionStateTransitionError
from dayu.log import Log

MODULE = "HOST.SESSION_REGISTRY"


from dayu.host._datetime_utils import now_utc as _now_utc, parse_dt as _parse_dt, serialize_dt as _serialize_dt


def _row_to_record(row: dict[str, Any]) -> SessionRecord:
    """将 SQLite 行记录转换为 SessionRecord。

    Args:
        row: SQLite 行（dict 模式）。

    Returns:
        SessionRecord 实例。
    """

    raw_metadata = row["metadata_json"]
    metadata = normalize_execution_delivery_context(json.loads(raw_metadata) if raw_metadata else {})
    return SessionRecord(
        session_id=row["session_id"],
        source=SessionSource(row["source"]),
        state=SessionState(row["state"]),
        scene_name=row["scene_name"],
        created_at=_parse_dt(row["created_at"]),
        last_activity_at=_parse_dt(row["last_activity_at"]),
        metadata=metadata,
    )


class SQLiteSessionRegistry(SessionRegistryProtocol):
    """基于 SQLite 的 SessionRegistry 实现。

    所有操作通过 HostStore.get_connection() 执行 SQL，
    支持跨进程可见性（SQLite WAL 模式）。
    """

    def __init__(self, host_store: HostStore) -> None:
        """初始化 SessionRegistry。

        Args:
            host_store: 共享 SQLite 存储。
        """

        self._host_store = host_store

    def create_session(
        self,
        source: SessionSource,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """创建新 session。"""

        sid = session_id or uuid.uuid4().hex
        now = _now_utc()
        now_str = _serialize_dt(now)
        normalized_metadata = normalize_execution_delivery_context(metadata)
        metadata_json = json.dumps(normalized_metadata, ensure_ascii=False)

        conn = self._host_store.get_connection()
        with write_transaction(conn):
            conn.execute(
                """
                INSERT INTO sessions (session_id, source, state, scene_name,
                                      created_at, last_activity_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sid, source.value, SessionState.ACTIVE.value, scene_name, now_str, now_str, metadata_json),
            )

        Log.debug(
            f"创建 session: session_id={sid}, source={source.value}, scene_name={scene_name or ''}",
            module=MODULE,
        )

        return SessionRecord(
            session_id=sid,
            source=source,
            state=SessionState.ACTIVE,
            scene_name=scene_name,
            created_at=now,
            last_activity_at=now,
            metadata=normalized_metadata,
        )

    def ensure_session(
        self,
        session_id: str,
        source: SessionSource,
        *,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """幂等获取或创建 session。"""

        now = _now_utc()
        now_str = _serialize_dt(now)
        normalized_metadata = normalize_execution_delivery_context(metadata)
        metadata_json = json.dumps(normalized_metadata, ensure_ascii=False)

        conn = self._host_store.get_connection()
        # INSERT OR IGNORE：存在则忽略，不存在则插入
        existing = self.get_session(session_id)
        with write_transaction(conn):
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions
                    (session_id, source, state, scene_name,
                     created_at, last_activity_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, source.value, SessionState.ACTIVE.value, scene_name, now_str, now_str, metadata_json),
            )
            # 无论是否新插入，都 touch last_activity_at
            conn.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE session_id = ?",
                (now_str, session_id),
            )

        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        Log.debug(
            f"ensure session: session_id={session_id}, source={source.value}, existed={existing is not None}",
            module=MODULE,
        )
        return _row_to_record(dict(row))

    def get_session(self, session_id: str) -> SessionRecord | None:
        """查询单个 session。"""

        conn = self._host_store.get_connection()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_record(dict(row))

    def list_sessions(
        self,
        *,
        state: SessionState | None = None,
        source: SessionSource | None = None,
        scene_name: str | None = None,
    ) -> list[SessionRecord]:
        """列出 sessions，可选按状态/来源/scene 过滤。"""

        conn = self._host_store.get_connection()
        where_clauses: list[str] = []
        query_params: list[str] = []
        if state is not None:
            where_clauses.append("state = ?")
            query_params.append(state.value)
        if source is not None:
            where_clauses.append("source = ?")
            query_params.append(source.value)
        if scene_name is not None:
            where_clauses.append("scene_name = ?")
            query_params.append(scene_name)

        sql = "SELECT * FROM sessions"
        if where_clauses:
            sql += f" WHERE {' AND '.join(where_clauses)}"
        sql += " ORDER BY created_at DESC"
        rows = conn.execute(sql, tuple(query_params)).fetchall()
        return [_row_to_record(dict(row)) for row in rows]

    def touch_session(self, session_id: str) -> None:
        """更新 session 最后活跃时间。"""

        conn = self._host_store.get_connection()
        with write_transaction(conn):
            cursor = conn.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE session_id = ?",
                (_serialize_dt(_now_utc()), session_id),
            )
            rowcount = cursor.rowcount
        if rowcount == 0:
            raise KeyError(f"session 不存在: {session_id}")
        Log.debug(f"刷新 session 活跃时间: session_id={session_id}", module=MODULE)

    def close_session(self, session_id: str) -> None:
        """关闭 session。"""

        conn = self._host_store.get_connection()
        with write_transaction(conn):
            cursor = conn.execute(
                "UPDATE sessions SET state = ? WHERE session_id = ?",
                (SessionState.CLOSED.value, session_id),
            )
            rowcount = cursor.rowcount
        if rowcount == 0:
            raise KeyError(f"session 不存在: {session_id}")
        Log.debug(f"关闭 session: session_id={session_id}", module=MODULE)

    def is_session_active(self, session_id: str) -> bool:
        """查询 session 是否处于 ``ACTIVE`` 状态。

        Args:
            session_id: 目标 session ID。

        Returns:
            session 存在且状态为 ``ACTIVE`` 时返回 ``True``；不存在或处于
            ``CLEARING`` / ``CLEARING_FAILED`` / ``CLOSED`` 任一非 ACTIVE
            状态时返回 ``False``。

        Raises:
            无。
        """

        normalized = str(session_id or "").strip()
        if not normalized:
            return False
        conn = self._host_store.get_connection()
        row = conn.execute(
            "SELECT state FROM sessions WHERE session_id = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            return False
        return str(row["state"]) == SessionState.ACTIVE.value

    def get_session_state(self, session_id: str) -> SessionState | None:
        """查询 session 当前状态。

        Args:
            session_id: 目标 session ID。

        Returns:
            session 当前 ``SessionState``；不存在返回 ``None``。

        Raises:
            无。
        """

        normalized = str(session_id or "").strip()
        if not normalized:
            return None
        conn = self._host_store.get_connection()
        row = conn.execute(
            "SELECT state FROM sessions WHERE session_id = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            return None
        return SessionState(str(row["state"]))

    def _transition_state(
        self,
        session_id: str,
        *,
        expected_states: tuple[SessionState, ...],
        target_state: SessionState,
        operation: str,
    ) -> None:
        """通用状态机迁移：仅当当前状态在 ``expected_states`` 集合内时切换。

        SQL 直接用 ``WHERE state IN (...)`` 做条件 update，rowcount=0 表示
        前置条件不满足。session 不存在时通过额外查询区分 ``KeyError`` 与
        ``RuntimeError``。
        """

        normalized = str(session_id or "").strip()
        if not normalized:
            raise KeyError(f"session 不存在: {session_id}")

        expected_values = tuple(state.value for state in expected_states)
        placeholders = ",".join("?" for _ in expected_values)
        conn = self._host_store.get_connection()
        with write_transaction(conn):
            cursor = conn.execute(
                f"""
                UPDATE sessions SET state = ?
                WHERE session_id = ? AND state IN ({placeholders})
                """,  # noqa: S608
                (target_state.value, normalized, *expected_values),
            )
            rowcount = cursor.rowcount
        if rowcount > 0:
            Log.debug(
                f"{operation}: session_id={normalized}, target_state={target_state.value}",
                module=MODULE,
            )
            return

        existing_state = self.get_session_state(normalized)
        if existing_state is None:
            raise KeyError(f"session 不存在: {normalized}")
        raise SessionStateTransitionError(
            normalized,
            operation=operation,
            current_state=existing_state,
            expected_states=expected_states,
        )

    def begin_clearing(self, session_id: str) -> None:
        """从 ``ACTIVE`` 推进到 ``CLEARING``。"""

        self._transition_state(
            session_id,
            expected_states=(SessionState.ACTIVE,),
            target_state=SessionState.CLEARING,
            operation="进入 CLEARING 屏障",
        )

    def end_clearing(self, session_id: str) -> None:
        """从 ``CLEARING`` 退出回 ``ACTIVE``。"""

        self._transition_state(
            session_id,
            expected_states=(SessionState.CLEARING,),
            target_state=SessionState.ACTIVE,
            operation="退出 CLEARING 屏障",
        )

    def mark_clearing_failed(self, session_id: str) -> None:
        """从 ``CLEARING`` 升级为 ``CLEARING_FAILED`` 持久锁定。"""

        self._transition_state(
            session_id,
            expected_states=(SessionState.CLEARING,),
            target_state=SessionState.CLEARING_FAILED,
            operation="升级 CLEARING_FAILED 锁定",
        )

    def close_idle_sessions(self, idle_threshold: timedelta) -> list[str]:
        """关闭超过空闲阈值的活跃 session。"""

        cutoff = _now_utc() - idle_threshold
        cutoff_str = _serialize_dt(cutoff)

        conn = self._host_store.get_connection()
        rows = conn.execute(
            """
            SELECT session_id FROM sessions
            WHERE state = ? AND last_activity_at < ?
            """,
            (SessionState.ACTIVE.value, cutoff_str),
        ).fetchall()

        closed_ids = [row["session_id"] for row in rows]
        if closed_ids:
            placeholders = ",".join("?" for _ in closed_ids)
            with write_transaction(conn):
                conn.execute(
                    f"UPDATE sessions SET state = ? WHERE session_id IN ({placeholders})",  # noqa: S608
                    [SessionState.CLOSED.value, *closed_ids],
                )
            Log.info(
                f"关闭空闲 session: count={len(closed_ids)}, session_ids={','.join(closed_ids)}",
                module=MODULE,
            )

        return closed_ids


__all__ = ["SQLiteSessionRegistry"]
