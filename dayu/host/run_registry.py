"""RunRegistry 的 SQLite 实现。

基于 HostStore 提供跨进程可见的 run 生命周期管理和状态机校验。
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Any

from dayu.host.host_store import HostStore
from dayu.host.protocols import RunRegistryProtocol
from dayu.contracts.run import (
    ACTIVE_STATES,
    ORPHAN_RUN_ERROR_SUMMARY,
    RunCancelReason,
    RunRecord,
    RunState,
    is_valid_transition,
)
from dayu.log import Log
from dayu.process_liveness import is_pid_alive

MODULE = "HOST.RUN_REGISTRY"
_ORPHAN_CLEANUP_MIN_RUN_AGE = timedelta(minutes=10)


from dayu.host._datetime_utils import now_utc as _now_utc, parse_dt_optional as _parse_dt_optional, serialize_dt as _serialize_dt


def _row_to_record(row: dict[str, Any]) -> RunRecord:
    """将 SQLite 行记录转换为 RunRecord。

    Args:
        row: SQLite 行（dict 模式）。

    Returns:
        RunRecord 实例。
    """

    raw_metadata = row["metadata_json"]
    metadata = json.loads(raw_metadata) if raw_metadata else {}
    return RunRecord(
        run_id=row["run_id"],
        session_id=row["session_id"],
        service_type=row["service_type"],
        scene_name=row["scene_name"],
        state=RunState(row["state"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=_parse_dt_optional(row["started_at"]),
        completed_at=_parse_dt_optional(row["completed_at"]),
        error_summary=row["error_summary"],
        cancel_requested_at=_parse_dt_optional(row.get("cancel_requested_at")),
        cancel_requested_reason=(
            RunCancelReason(row["cancel_requested_reason"])
            if row.get("cancel_requested_reason")
            else None
        ),
        cancel_reason=RunCancelReason(row["cancel_reason"]) if row.get("cancel_reason") else None,
        owner_pid=row["owner_pid"],
        metadata=metadata,
    )


class SQLiteRunRegistry(RunRegistryProtocol):
    """基于 SQLite 的 RunRegistry 实现。

    所有操作通过 HostStore.get_connection() 执行 SQL，
    状态机校验使用 contracts.run 中的转换表。
    """

    def __init__(self, host_store: HostStore) -> None:
        """初始化 RunRegistry。

        Args:
            host_store: 共享 SQLite 存储。
        """

        self._host_store = host_store

    def register_run(
        self,
        *,
        session_id: str | None = None,
        service_type: str,
        scene_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RunRecord:
        """注册一个新 run。"""

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = _now_utc()
        now_str = _serialize_dt(now)
        pid = os.getpid()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False)

        conn = self._host_store.get_connection()
        conn.execute(
            """
            INSERT INTO runs (run_id, session_id, service_type, scene_name,
                              state, created_at, owner_pid, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, session_id, service_type, scene_name, RunState.CREATED.value, now_str, pid, metadata_json),
        )
        conn.commit()

        Log.debug(
            f"注册 run: run_id={run_id}, service_type={service_type}, session_id={session_id or ''}, scene_name={scene_name or ''}",
            module=MODULE,
        )

        return RunRecord(
            run_id=run_id,
            session_id=session_id,
            service_type=service_type,
            scene_name=scene_name,
            state=RunState.CREATED,
            created_at=now,
            cancel_requested_at=None,
            cancel_requested_reason=None,
            cancel_reason=None,
            owner_pid=pid,
            metadata=metadata or {},
        )

    def start_run(self, run_id: str) -> RunRecord:
        """将 run 标记为 RUNNING。"""

        return self._transition(run_id, RunState.RUNNING, started_at=_now_utc())

    def complete_run(self, run_id: str, *, error_summary: str | None = None) -> RunRecord:
        """标记 run 成功完成。"""

        return self._transition(
            run_id,
            RunState.SUCCEEDED,
            completed_at=_now_utc(),
            error_summary=error_summary,
        )

    def fail_run(self, run_id: str, *, error_summary: str | None = None) -> RunRecord:
        """标记 run 失败。"""

        return self._transition(
            run_id,
            RunState.FAILED,
            completed_at=_now_utc(),
            error_summary=error_summary,
        )

    def mark_cancelled(
        self,
        run_id: str,
        *,
        cancel_reason: RunCancelReason = RunCancelReason.USER_CANCELLED,
    ) -> RunRecord:
        """标记 run 已取消。"""

        return self._transition(
            run_id,
            RunState.CANCELLED,
            completed_at=_now_utc(),
            cancel_reason=cancel_reason,
        )

    def request_cancel(
        self,
        run_id: str,
        *,
        cancel_reason: RunCancelReason = RunCancelReason.USER_CANCELLED,
    ) -> bool:
        """请求取消 run（跨进程可见）。"""

        conn = self._host_store.get_connection()
        active_values = tuple(s.value for s in ACTIVE_STATES)
        placeholders = ",".join("?" for _ in active_values)
        cursor = conn.execute(
            f"""
            UPDATE runs
            SET cancel_requested_at = ?, cancel_requested_reason = ?
            WHERE run_id = ?
              AND state IN ({placeholders})
              AND cancel_requested_at IS NULL
            """,  # noqa: S608
            [
                _serialize_dt(_now_utc()),
                cancel_reason.value,
                run_id,
                *active_values,
            ],
        )
        conn.commit()
        if cursor.rowcount > 0:
            Log.info(
                f"登记 run 取消请求: run_id={run_id}, cancel_reason={cancel_reason.value}",
                module=MODULE,
            )
        return cursor.rowcount > 0

    def is_cancel_requested(self, run_id: str) -> bool:
        """查询 run 是否已记录取消意图。"""

        conn = self._host_store.get_connection()
        row = conn.execute(
            "SELECT cancel_requested_at FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return False
        return row["cancel_requested_at"] is not None

    def get_run(self, run_id: str) -> RunRecord | None:
        """查询单个 run。"""

        conn = self._host_store.get_connection()
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_record(dict(row))

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        state: RunState | None = None,
        service_type: str | None = None,
    ) -> list[RunRecord]:
        """列出 runs，支持多维过滤。"""

        conditions: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if state is not None:
            conditions.append("state = ?")
            params.append(state.value)
        if service_type is not None:
            conditions.append("service_type = ?")
            params.append(service_type)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        conn = self._host_store.get_connection()
        rows = conn.execute(
            f"SELECT * FROM runs WHERE {where_clause} ORDER BY created_at DESC",  # noqa: S608
            params,
        ).fetchall()
        return [_row_to_record(dict(row)) for row in rows]

    def list_active_runs(self) -> list[RunRecord]:
        """列出所有活跃 run。"""

        active_values = tuple(s.value for s in ACTIVE_STATES)
        placeholders = ",".join("?" for _ in active_values)
        conn = self._host_store.get_connection()
        rows = conn.execute(
            f"SELECT * FROM runs WHERE state IN ({placeholders}) ORDER BY created_at DESC",  # noqa: S608
            active_values,
        ).fetchall()
        return [_row_to_record(dict(row)) for row in rows]

    def cleanup_orphan_runs(self) -> list[str]:
        """清理 owner_pid 已死亡的活跃 run，标记为 FAILED。"""

        active_runs = self.list_active_runs()
        candidate_orphan_ids: list[str] = []
        now = _now_utc()
        for run in active_runs:
            reference_time = run.started_at or run.created_at
            if now - reference_time < _ORPHAN_CLEANUP_MIN_RUN_AGE:
                continue
            if not is_pid_alive(run.owner_pid):
                candidate_orphan_ids.append(run.run_id)

        if candidate_orphan_ids:
            now_str = _serialize_dt(now)
            conn = self._host_store.get_connection()
            active_values = tuple(state.value for state in ACTIVE_STATES)
            placeholders = ",".join("?" for _ in active_values)
            orphan_ids: list[str] = []
            for oid in candidate_orphan_ids:
                # 这里写入的 ORPHAN_RUN_ERROR_SUMMARY 必须与 _transition() 中的修复判定保持同源。
                cursor = conn.execute(
                    """
                    UPDATE runs SET state = ?, completed_at = ?,
                           error_summary = ?
                    WHERE run_id = ?
                      AND state IN ({placeholders})
                    """.format(placeholders=placeholders),
                    (
                        RunState.FAILED.value,
                        now_str,
                        ORPHAN_RUN_ERROR_SUMMARY,
                        oid,
                        *active_values,
                    ),
                )
                if cursor.rowcount > 0:
                    orphan_ids.append(oid)
            conn.commit()
            if orphan_ids:
                Log.info(
                    f"清理 orphan runs: count={len(orphan_ids)}, run_ids={','.join(orphan_ids)}",
                    module=MODULE,
                )
            return orphan_ids

        return []

    def _transition(
        self,
        run_id: str,
        target_state: RunState,
        *,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error_summary: str | None = None,
        cancel_requested_at: datetime | None = None,
        cancel_requested_reason: RunCancelReason | None = None,
        cancel_reason: RunCancelReason | None = None,
    ) -> RunRecord:
        """执行状态转换并返回更新后的记录。

        Args:
            run_id: 目标 run ID。
            target_state: 目标状态。
            started_at: 启动时间（RUNNING 转换时设置）。
            completed_at: 完成时间（终态转换时设置）。
            error_summary: 失败摘要。
            cancel_requested_at: 请求取消时间。
            cancel_requested_reason: 请求取消原因。
            cancel_reason: 取消原因。

        Returns:
            更新后的 RunRecord。

        Raises:
            KeyError: run 不存在。
            ValueError: 非法状态转换。
        """

        conn = self._host_store.get_connection()
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"run 不存在: {run_id}")

        current_state = RunState(row["state"])
        if not is_valid_transition(current_state, target_state):
            if (
                current_state == RunState.FAILED
                and target_state == RunState.SUCCEEDED
                and int(row["owner_pid"]) == os.getpid()
                and str(row["error_summary"] or "").strip() == ORPHAN_RUN_ERROR_SUMMARY
            ):
                Log.warn(
                    (
                        "检测到当前 owner 修复被误判为 orphan 的 run，"
                        f"允许恢复成功终态: run_id={run_id}"
                    ),
                    module=MODULE,
                )
            else:
                raise ValueError(
                    f"非法状态转换: {current_state.value} -> {target_state.value} (run_id={run_id})"
                )

        # 构造 SET 子句
        set_parts = ["state = ?"]
        params: list[Any] = [target_state.value]

        if started_at is not None:
            set_parts.append("started_at = ?")
            params.append(_serialize_dt(started_at))
        if completed_at is not None:
            set_parts.append("completed_at = ?")
            params.append(_serialize_dt(completed_at))
        if error_summary is not None:
            set_parts.append("error_summary = ?")
            params.append(error_summary)
        elif target_state != RunState.FAILED:
            # 终态离开 FAILED 时统一清空 error_summary，包括 owner 修复 orphan failure 的成功收口。
            set_parts.append("error_summary = NULL")
        if cancel_requested_at is not None:
            set_parts.append("cancel_requested_at = ?")
            params.append(_serialize_dt(cancel_requested_at))
        if cancel_requested_reason is not None:
            set_parts.append("cancel_requested_reason = ?")
            params.append(cancel_requested_reason.value)
        if cancel_reason is not None:
            set_parts.append("cancel_reason = ?")
            params.append(cancel_reason.value)

        set_clause = ", ".join(set_parts)
        params.append(run_id)
        conn.execute(
            f"UPDATE runs SET {set_clause} WHERE run_id = ?",  # noqa: S608
            params,
        )
        conn.commit()

        # 重新读取最终状态
        updated_row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        Log.debug(
            f"run 状态迁移: run_id={run_id}, from_state={current_state.value}, to_state={target_state.value}",
            module=MODULE,
        )
        return _row_to_record(dict(updated_row))


__all__ = ["SQLiteRunRegistry"]
