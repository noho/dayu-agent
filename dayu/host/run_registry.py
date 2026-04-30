"""RunRegistry 的 SQLite 实现。

基于 HostStore 提供跨进程可见的 run 生命周期管理和状态机校验。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from typing import Any

from dayu.host._session_barrier import ensure_session_active
from dayu.host.host_store import HostStore, write_transaction
from dayu.host.protocols import RunRegistryProtocol, SessionActivityQueryProtocol
from dayu.contracts.execution_metadata import (
    ExecutionDeliveryContext,
    empty_execution_delivery_context,
    normalize_execution_delivery_context,
)
from dayu.contracts.run import (
    ACTIVE_STATES,
    ORPHAN_RUN_ERROR_SUMMARY,
    RunCancelReason,
    RunRecord,
    RunState,
    is_valid_transition,
)
from dayu.log import Log
from dayu.process_liveness import OwnerIdentity, current_owner_identity, is_owner_identity_alive

MODULE = "HOST.RUN_REGISTRY"
_ORPHAN_CLEANUP_MIN_RUN_AGE = timedelta(minutes=10)


from dayu.host._datetime_utils import now_utc as _now_utc, parse_dt_optional as _parse_dt_optional, serialize_dt as _serialize_dt


def _build_run_registration_debug_message(
    *,
    run_id: str,
    service_type: str,
    session_id: str | None,
    scene_name: str | None,
    owner_pid: int,
    db_path: str,
) -> str:
    """构造 run 注册调试日志。

    Args:
        run_id: 新注册的 run ID。
        service_type: 服务类型。
        session_id: 宿主会话 ID。
        scene_name: scene 名称。
        owner_pid: 当前 owner PID。
        db_path: Host SQLite 路径。

    Returns:
        统一格式的调试日志文本。

    Raises:
        无。
    """

    return (
        "注册 run: "
        f"run_id={run_id}, service_type={service_type}, "
        f"session_id={session_id or ''}, scene_name={scene_name or ''}, "
        f"owner_pid={owner_pid}, db_path={db_path}"
    )


def _build_run_transition_debug_message(
    *,
    run_id: str,
    current_state: RunState,
    target_state: RunState,
    db_path: str,
) -> str:
    """构造 run 状态迁移调试日志。

    Args:
        run_id: 目标 run ID。
        current_state: 当前状态。
        target_state: 目标状态。
        db_path: Host SQLite 路径。

    Returns:
        统一格式的调试日志文本。

    Raises:
        无。
    """

    return (
        "run 状态迁移: "
        f"run_id={run_id}, from_state={current_state.value}, "
        f"to_state={target_state.value}, db_path={db_path}"
    )


def _row_to_record(row: dict[str, Any]) -> RunRecord:
    """将 SQLite 行记录转换为 RunRecord。

    Args:
        row: SQLite 行（dict 模式）。

    Returns:
        RunRecord 实例。
    """

    raw_metadata = row["metadata_json"]
    raw_parsed: object = json.loads(raw_metadata) if raw_metadata else {}
    metadata: ExecutionDeliveryContext = (
        normalize_execution_delivery_context(raw_parsed)
        if isinstance(raw_parsed, dict)
        else empty_execution_delivery_context()
    )
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
        owner_process_start_time=row.get("owner_process_start_time"),
        owner_boot_id=row.get("owner_boot_id"),
        metadata=metadata,
    )


class SQLiteRunRegistry(RunRegistryProtocol):
    """基于 SQLite 的 RunRegistry 实现。

    所有操作通过 HostStore.get_connection() 执行 SQL，
    状态机校验使用 contracts.run 中的转换表。
    """

    def __init__(
        self,
        host_store: HostStore,
        *,
        session_activity: SessionActivityQueryProtocol | None = None,
    ) -> None:
        """初始化 RunRegistry。

        Args:
            host_store: 共享 SQLite 存储。
            session_activity: 可选的 session 活性查询源；装配后 ``register_run``
                会在 session 不再 ``ACTIVE`` 时拒绝新 run 写入，与 #117
                ``CLEARING`` / ``CLEARING_FAILED`` 屏障语义一致。装配为
                ``None`` 时退化为不做屏障的旧行为，仅用于独立 store 单元测试。
        """

        self._host_store = host_store
        self._session_activity: SessionActivityQueryProtocol | None = session_activity

    def register_run(
        self,
        *,
        session_id: str | None = None,
        service_type: str,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> RunRecord:
        """注册一个新 run。"""

        if session_id is not None:
            ensure_session_active(
                self._session_activity,
                session_id=session_id,
                operation="register_run",
                module=MODULE,
                target_name="run",
            )

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = _now_utc()
        now_str = _serialize_dt(now)
        owner = current_owner_identity()
        pid = owner.pid
        metadata_typed: ExecutionDeliveryContext = (
            normalize_execution_delivery_context(metadata)
            if metadata is not None
            else empty_execution_delivery_context()
        )
        metadata_json = json.dumps(metadata_typed, ensure_ascii=False)

        conn = self._host_store.get_connection()
        with write_transaction(conn):
            conn.execute(
                """
                INSERT INTO runs (run_id, session_id, service_type, scene_name,
                                  state, created_at, owner_pid,
                                  owner_process_start_time, owner_boot_id,
                                  metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    service_type,
                    scene_name,
                    RunState.CREATED.value,
                    now_str,
                    pid,
                    owner.process_start_time,
                    owner.boot_id,
                    metadata_json,
                ),
            )

        Log.debug(
            _build_run_registration_debug_message(
                run_id=run_id,
                service_type=service_type,
                session_id=session_id,
                scene_name=scene_name,
                owner_pid=pid,
                db_path=str(self._host_store.db_path),
            ),
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
            owner_process_start_time=owner.process_start_time,
            owner_boot_id=owner.boot_id,
            metadata=metadata_typed,
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
        """标记 run 已取消。

        幂等：如果 run 已处于 CANCELLED 状态，直接返回当前记录，
        避免 executor 取消路径与 signal handler shutdown 路径竞态时
        触发「cancelled -> cancelled」非法状态转换。
        """

        conn = self._host_store.get_connection()
        with write_transaction(conn):
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"run 不存在: {run_id}")
            current_state = RunState(row["state"])
            if current_state == RunState.CANCELLED:
                Log.debug(
                    f"mark_cancelled 幂等跳过: run 已处于 CANCELLED 状态, run_id={run_id}",
                    module=MODULE,
                )
                return _row_to_record(dict(row))

        return self._transition(
            run_id,
            RunState.CANCELLED,
            completed_at=_now_utc(),
            cancel_reason=cancel_reason,
        )

    def mark_unsettled(
        self,
        run_id: str,
        *,
        error_summary: str | None = None,
    ) -> RunRecord:
        """将 run 标记为 UNSETTLED（orphan cleanup / 无法判定的残留）。

        Args:
            run_id: 目标 run ID。
            error_summary: 错误摘要（通常为 orphan 描述）。

        Returns:
            更新后的 RunRecord。

        Raises:
            KeyError: run 不存在。
            ValueError: 非法状态转换。
        """

        return self._transition(
            run_id,
            RunState.UNSETTLED,
            completed_at=_now_utc(),
            error_summary=error_summary,
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
        with write_transaction(conn):
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
            rowcount = cursor.rowcount
        if rowcount > 0:
            Log.debug(
                f"登记 run 取消请求: run_id={run_id}, cancel_reason={cancel_reason.value}",
                module=MODULE,
            )
        return rowcount > 0

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

    def list_active_runs_for_owner(self, owner: OwnerIdentity) -> list[RunRecord]:
        """列出指定 ``OwnerIdentity`` 拥有的所有活跃 run。

        匹配规则分两步：

        1. **SQL 粗筛**：以 ``owner_pid`` 为索引列做等值过滤，命中所有
           PID 相同的活跃 run（包含 PID 已被复用为新 owner 的旧 record）。
        2. **Python 精筛**：在内存里调用 :meth:`OwnerIdentity.matches`
           做完整三元组匹配，``process_start_time`` / ``owner_boot_id``
           任一字段双方均非 ``None`` 时必须等值，否则视为不参与比对，
           最差等价于仅 PID 比对。

        粗筛 + 精筛的拆分原因：``owner_pid`` 是 SQL 索引列；
        三元组里另两列含 NULL 语义，做 SQL 等值需要 ``IS`` / NULL 友好
        谓词，复杂度收益不及 Python 侧直接调用 ``matches``。
        ACTIVE 状态下被 PID 复用的 orphan record 由 :meth:`cleanup_orphan_runs`
        在启动期回收，残留窗口极窄。

        Args:
            owner: 当前进程身份。

        Returns:
            指定 owner 仍活跃的 run 列表。

        Raises:
            无。
        """

        active_values = tuple(s.value for s in ACTIVE_STATES)
        placeholders = ",".join("?" for _ in active_values)
        conn = self._host_store.get_connection()
        rows = conn.execute(
            f"""
            SELECT * FROM runs
            WHERE state IN ({placeholders})
              AND owner_pid = ?
            ORDER BY created_at DESC
            """,  # noqa: S608
            (*active_values, int(owner.pid)),
        ).fetchall()
        records = [_row_to_record(dict(row)) for row in rows]
        return [
            record
            for record in records
            if record.owner_identity.matches(owner)
        ]

    def cleanup_orphan_runs(self) -> list[str]:
        """清理 owner_pid 已死亡的活跃 run，标记为 UNSETTLED。

        UNSETTLED 与 FAILED 语义分离：
            - FAILED：业务/Agent 明确失败；
            - UNSETTLED：Host 无法判定的残留（owner 进程异常终止），
              admin / 自愈逻辑以 state 为 discriminator，不再依赖 error_summary。
        """

        active_runs = self.list_active_runs()
        candidate_orphan_ids: list[str] = []
        now = _now_utc()
        for run in active_runs:
            reference_time = run.started_at or run.created_at
            if now - reference_time < _ORPHAN_CLEANUP_MIN_RUN_AGE:
                continue
            if not is_owner_identity_alive(run.owner_identity):
                candidate_orphan_ids.append(run.run_id)

        if candidate_orphan_ids:
            now_str = _serialize_dt(now)
            conn = self._host_store.get_connection()
            active_values = tuple(state.value for state in ACTIVE_STATES)
            placeholders = ",".join("?" for _ in active_values)
            orphan_ids: list[str] = []
            with write_transaction(conn):
                for oid in candidate_orphan_ids:
                    # 写入 UNSETTLED + ORPHAN_RUN_ERROR_SUMMARY；
                    # ORPHAN_RUN_ERROR_SUMMARY 仅作为描述性填充，判定一律走 state == UNSETTLED。
                    cursor = conn.execute(
                        """
                        UPDATE runs SET state = ?, completed_at = ?,
                               error_summary = ?
                        WHERE run_id = ?
                          AND state IN ({placeholders})
                        """.format(placeholders=placeholders),
                        (
                            RunState.UNSETTLED.value,
                            now_str,
                            ORPHAN_RUN_ERROR_SUMMARY,
                            oid,
                            *active_values,
                        ),
                    )
                    if cursor.rowcount > 0:
                        orphan_ids.append(oid)
            if orphan_ids:
                Log.debug(
                    f"清理 orphan runs -> UNSETTLED: count={len(orphan_ids)}, run_ids={','.join(orphan_ids)}",
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
        with write_transaction(conn):
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"run 不存在: {run_id}")

            current_state = RunState(row["state"])
            is_owner_unsettled_recovery = False
            if not is_valid_transition(current_state, target_state):
                stored_owner = OwnerIdentity(
                    pid=int(row["owner_pid"]),
                    process_start_time=row["owner_process_start_time"],
                    boot_id=row["owner_boot_id"],
                )
                if (
                    current_state == RunState.UNSETTLED
                    and target_state == RunState.SUCCEEDED
                    and stored_owner.matches(current_owner_identity())
                ):
                    is_owner_unsettled_recovery = True
                    Log.warn(
                        (
                            "检测到当前 owner 修复被误判为 orphan 的 run，"
                            f"允许从 UNSETTLED 恢复成功终态: run_id={run_id}"
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
            elif is_owner_unsettled_recovery:
                # owner 把 UNSETTLED 修复回 SUCCEEDED 时清空 orphan 填充的 error_summary。
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

            # 事务内重新读取最终状态，保证与 UPDATE 原子一致。
            updated_row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        Log.debug(
            _build_run_transition_debug_message(
                run_id=run_id,
                current_state=current_state,
                target_state=target_state,
                db_path=str(self._host_store.db_path),
            ),
            module=MODULE,
        )
        return _row_to_record(dict(updated_row))


__all__ = ["SQLiteRunRegistry"]
