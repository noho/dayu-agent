"""pending conversation turn 仓储与默认实现。

该模块定义 Host 内部用于恢复 V1 的真源对象：pending conversation turn。
transcript 只记录已经成功提交完成的 conversation turn，
尚未完成的当前 turn 必须通过该仓储持久化。
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from dayu.contracts.execution_metadata import (
    ExecutionDeliveryContext,
    empty_execution_delivery_context,
    normalize_execution_delivery_context,
)
from dayu.host._session_barrier import ensure_session_active
from dayu.host.host_store import HostStore, write_transaction

if TYPE_CHECKING:
    from dayu.host.protocols import SessionActivityQueryProtocol


from dayu.host._datetime_utils import now_utc as _now_utc, parse_dt as _parse_dt, serialize_dt as _serialize_dt
from dayu.log import Log


MODULE = "HOST.PENDING_TURN_STORE"


def _normalize_metadata(metadata: ExecutionDeliveryContext | None) -> ExecutionDeliveryContext:
    """规范化 pending turn 元数据。

    Args:
        metadata: 原始元数据。

    Returns:
        过滤空 key 后的字符串字典。

    Raises:
        无。
    """

    return normalize_execution_delivery_context(metadata)


def _normalize_resume_source_json(value: str | None) -> str:
    """规范化 pending turn 恢复真源 JSON 文本。"""

    normalized = str(value or "").strip()
    if not normalized:
        return ""
    payload = json.loads(normalized)
    if not isinstance(payload, dict):
        raise ValueError("resume_source_json 必须是 JSON object")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _normalize_text(value: str, *, field_name: str) -> str:
    """规范化必填文本字段。

    Args:
        value: 原始文本。
        field_name: 字段名。

    Returns:
        去除首尾空白后的文本。

    Raises:
        ValueError: 文本为空时抛出。
    """

    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


class PendingConversationTurnState(str, Enum):
    """pending conversation turn 的 Host 内部执行状态。

    RESUMING 表示某个 resumer 正在消费该 pending turn，用于跨进程互斥；
    acquire 成功时由仓储原子地把来源状态（ACCEPTED_BY_HOST / PREPARED_BY_HOST）
    写入 ``pre_resume_state`` 字段，失败路径据此回退。
    """

    ACCEPTED_BY_HOST = "accepted_by_host"
    PREPARED_BY_HOST = "prepared_by_host"
    SENT_TO_LLM = "sent_to_llm"
    RESUMING = "resuming"


# 允许 acquire resume lease 的来源状态集合；除 RESUMING 自身（已被他人持有）
# 外，任何可恢复态都允许进入 resume 流程：ACCEPTED_BY_HOST / PREPARED_BY_HOST
# 属于 Host 内部就绪态；SENT_TO_LLM 常见于 LLM 阶段启动后 source run crash 留下
# 的残留，也需允许接管恢复。
_RESUME_ACQUIRABLE_STATES: frozenset[PendingConversationTurnState] = frozenset(
    {
        PendingConversationTurnState.ACCEPTED_BY_HOST,
        PendingConversationTurnState.PREPARED_BY_HOST,
        PendingConversationTurnState.SENT_TO_LLM,
    }
)


class PendingTurnResumeConflictError(RuntimeError):
    """并发 resume 互斥失败：pending turn 当前正被其他 resumer 持有。

    用于 acquire resume lease 时区分 "已被占用" 与 "达上限 / 不存在 / 不可恢复"
    等其他失败语义。上层 Host 负责把该异常转译为调用方可理解的错误消息。
    """

    def __init__(self, pending_turn_id: str) -> None:
        """构造冲突异常。

        Args:
            pending_turn_id: 冲突发生的 pending turn ID。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(
            f"pending conversation turn 正被其他 resumer 持有: pending_turn_id={pending_turn_id}"
        )
        self.pending_turn_id = pending_turn_id


@dataclass(frozen=True)
class PendingConversationTurn:
    """Host 内部的 pending conversation turn 记录。"""

    pending_turn_id: str
    session_id: str
    scene_name: str
    user_text: str
    source_run_id: str
    created_at: datetime
    updated_at: datetime
    resumable: bool
    state: PendingConversationTurnState
    resume_source_json: str = ""
    resume_attempt_count: int = 0
    last_resume_error_message: str | None = None
    pre_resume_state: PendingConversationTurnState | None = None
    metadata: ExecutionDeliveryContext = field(default_factory=empty_execution_delivery_context)


@dataclass(frozen=True)
class _PendingTurnSlot:
    """锁内冲突判断所需的最小 pending turn 槽位信息。"""

    pending_turn_id: str
    user_text: str


class InMemoryPendingConversationTurnStore:
    """最小化内存版 pending turn 仓储。

    仅用于单元测试或显式注入 Host 内部组件时的默认兜底。
    """

    def __init__(
        self,
        *,
        session_activity: "SessionActivityQueryProtocol | None" = None,
    ) -> None:
        """初始化内存仓储。

        Args:
            session_activity: 可选的 session 活性查询。装配后所有写入入口会先
                校验 session 是否仍处于非 CLOSED 状态；装配为 ``None`` 时退化为
                不做屏障的旧行为，仅用于独立 store 单元测试。

        Returns:
            无。

        Raises:
            无。
        """

        self._records: dict[str, PendingConversationTurn] = {}
        self._session_activity: "SessionActivityQueryProtocol | None" = session_activity

    def upsert_pending_turn(
        self,
        *,
        session_id: str,
        scene_name: str,
        user_text: str,
        source_run_id: str,
        resumable: bool,
        state: PendingConversationTurnState,
        resume_source_json: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> PendingConversationTurn:
        """创建或更新当前 session/scene 的活跃 pending turn。"""

        normalized_session_id = _normalize_text(session_id, field_name="session_id")
        normalized_scene_name = _normalize_text(scene_name, field_name="scene_name")
        normalized_user_text = _normalize_text(user_text, field_name="user_text")
        normalized_source_run_id = _normalize_text(source_run_id, field_name="source_run_id")
        normalized_resume_source_json = _normalize_resume_source_json(resume_source_json)
        normalized_metadata = _normalize_metadata(metadata)
        ensure_session_active(
            self._session_activity,
            session_id=normalized_session_id,
            operation="upsert_pending_turn",
            module=MODULE,
            target_name="pending turn",
        )
        existing = self.get_session_pending_turn(
            session_id=normalized_session_id,
            scene_name=normalized_scene_name,
        )
        now = _now_utc()
        if existing is None:
            record = PendingConversationTurn(
                pending_turn_id=f"pending_{uuid.uuid4().hex[:12]}",
                session_id=normalized_session_id,
                scene_name=normalized_scene_name,
                user_text=normalized_user_text,
                source_run_id=normalized_source_run_id,
                created_at=now,
                updated_at=now,
                resumable=bool(resumable),
                state=state,
                resume_source_json=normalized_resume_source_json,
                metadata=normalized_metadata,
            )
            self._records[record.pending_turn_id] = record
            return record
        if existing.user_text != normalized_user_text:
            raise ValueError(
                "当前 session/scene 已存在活跃 pending turn，不能写入不同 user_text: "
                f"session_id={normalized_session_id}, scene_name={normalized_scene_name}"
            )
        updated = PendingConversationTurn(
            pending_turn_id=existing.pending_turn_id,
            session_id=existing.session_id,
            scene_name=existing.scene_name,
            user_text=existing.user_text,
            source_run_id=normalized_source_run_id,
            created_at=existing.created_at,
            updated_at=now,
            resumable=bool(resumable),
            state=state,
            resume_source_json=normalized_resume_source_json,
            resume_attempt_count=existing.resume_attempt_count,
            last_resume_error_message=existing.last_resume_error_message,
            pre_resume_state=existing.pre_resume_state,
            metadata=normalized_metadata,
        )
        self._records[updated.pending_turn_id] = updated
        return updated

    def get_pending_turn(self, pending_turn_id: str) -> PendingConversationTurn | None:
        """按 ID 查询 pending turn。"""

        return self._records.get(str(pending_turn_id or "").strip())

    def get_session_pending_turn(
        self,
        *,
        session_id: str,
        scene_name: str,
    ) -> PendingConversationTurn | None:
        """按 session/scene 查询当前 pending turn。"""

        normalized_session_id = _normalize_text(session_id, field_name="session_id")
        normalized_scene_name = _normalize_text(scene_name, field_name="scene_name")
        candidates = [
            record
            for record in self._records.values()
            if record.session_id == normalized_session_id and record.scene_name == normalized_scene_name
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item.updated_at, reverse=True)[0]

    def list_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: PendingConversationTurnState | None = None,
        resumable_only: bool = False,
    ) -> list[PendingConversationTurn]:
        """列出 pending turn。"""

        records = list(self._records.values())
        if session_id is not None:
            normalized_session_id = _normalize_text(session_id, field_name="session_id")
            records = [record for record in records if record.session_id == normalized_session_id]
        if scene_name is not None:
            normalized_scene_name = _normalize_text(scene_name, field_name="scene_name")
            records = [record for record in records if record.scene_name == normalized_scene_name]
        if state is not None:
            records = [record for record in records if record.state == state]
        if resumable_only:
            records = [record for record in records if record.resumable]
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def update_state(
        self,
        pending_turn_id: str,
        *,
        state: PendingConversationTurnState,
    ) -> PendingConversationTurn:
        """更新 pending turn 的 Host 内部状态。"""

        existing = self.get_pending_turn(pending_turn_id)
        if existing is None:
            raise KeyError(f"pending turn 不存在: {pending_turn_id}")
        ensure_session_active(
            self._session_activity,
            session_id=existing.session_id,
            operation="update_state",
            module=MODULE,
            target_name="pending turn",
        )
        updated = PendingConversationTurn(
            pending_turn_id=existing.pending_turn_id,
            session_id=existing.session_id,
            scene_name=existing.scene_name,
            user_text=existing.user_text,
            source_run_id=existing.source_run_id,
            created_at=existing.created_at,
            updated_at=_now_utc(),
            resumable=existing.resumable,
            state=state,
            resume_source_json=existing.resume_source_json,
            resume_attempt_count=existing.resume_attempt_count,
            last_resume_error_message=existing.last_resume_error_message,
            pre_resume_state=existing.pre_resume_state,
            metadata=_normalize_metadata(existing.metadata),
        )
        self._records[updated.pending_turn_id] = updated
        return updated

    def record_resume_attempt(
        self,
        pending_turn_id: str,
        *,
        max_attempts: int,
    ) -> PendingConversationTurn:
        """在未达到上限时记录一次 pending turn 恢复尝试。"""

        existing = self.get_pending_turn(pending_turn_id)
        if existing is None:
            raise KeyError(f"pending turn 不存在: {pending_turn_id}")
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须是正整数")
        ensure_session_active(
            self._session_activity,
            session_id=existing.session_id,
            operation="record_resume_attempt",
            module=MODULE,
            target_name="pending turn",
        )
        if existing.resume_attempt_count >= max_attempts:
            self._records.pop(existing.pending_turn_id, None)
            Log.warn(
                "pending turn 恢复次数已达到上限，已原子删除: "
                f"pending_turn_id={existing.pending_turn_id}, "
                f"session_id={existing.session_id}, "
                f"max_attempts={max_attempts}",
                module=MODULE,
            )
            raise ValueError(
                "pending turn 恢复次数已达到上限，已删除: "
                f"pending_turn_id={pending_turn_id}, max_attempts={max_attempts}"
            )
        if existing.state not in _RESUME_ACQUIRABLE_STATES:
            # 当前状态已是 RESUMING / SENT_TO_LLM 等；视为被其他 resumer 占用。
            raise PendingTurnResumeConflictError(existing.pending_turn_id)
        updated = PendingConversationTurn(
            pending_turn_id=existing.pending_turn_id,
            session_id=existing.session_id,
            scene_name=existing.scene_name,
            user_text=existing.user_text,
            source_run_id=existing.source_run_id,
            created_at=existing.created_at,
            updated_at=_now_utc(),
            resumable=existing.resumable,
            state=PendingConversationTurnState.RESUMING,
            resume_source_json=existing.resume_source_json,
            resume_attempt_count=existing.resume_attempt_count + 1,
            last_resume_error_message=None,
            pre_resume_state=existing.state,
            metadata=_normalize_metadata(existing.metadata),
        )
        self._records[updated.pending_turn_id] = updated
        return updated

    def release_resume_lease(self, pending_turn_id: str) -> PendingConversationTurn | None:
        """把 RESUMING 的 pending turn 回退到 ``pre_resume_state``。

        Args:
            pending_turn_id: 目标 pending turn ID。

        Returns:
            回退后的 pending turn；若记录不存在或当前 state 非 RESUMING 则返回
            ``None``（幂等 no-op）。

        Raises:
            无：记录缺失、状态不符合回退条件时视为 no-op。
        """

        existing = self.get_pending_turn(pending_turn_id)
        if existing is None:
            return None
        if existing.state is not PendingConversationTurnState.RESUMING:
            return existing
        restored_state = existing.pre_resume_state or PendingConversationTurnState.ACCEPTED_BY_HOST
        updated = PendingConversationTurn(
            pending_turn_id=existing.pending_turn_id,
            session_id=existing.session_id,
            scene_name=existing.scene_name,
            user_text=existing.user_text,
            source_run_id=existing.source_run_id,
            created_at=existing.created_at,
            updated_at=_now_utc(),
            resumable=existing.resumable,
            state=restored_state,
            resume_source_json=existing.resume_source_json,
            resume_attempt_count=existing.resume_attempt_count,
            last_resume_error_message=existing.last_resume_error_message,
            pre_resume_state=None,
            metadata=_normalize_metadata(existing.metadata),
        )
        self._records[updated.pending_turn_id] = updated
        return updated

    def rebind_source_run_id_for_resume(
        self,
        pending_turn_id: str,
        *,
        new_source_run_id: str,
    ) -> PendingConversationTurn:
        """把持有 RESUMING lease 的 pending turn 的 ``source_run_id`` 重绑到当前 resumed run。

        resume 路径下 executor 不再 upsert pending turn（否则会覆盖 Host 端持有的
        RESUMING lease），因此必须在 executor 拿到新 run_id 后显式调用本方法，把
        pending turn 的 ``source_run_id`` 切到当前 resumed run。否则"成功执行后
        delete_pending_turn 失败"会退化为可重复恢复的错误残留——旧 source_run
        仍处于 timeout-cancelled 终态，后续 resume gate 与 cleanup 都会放行。

        Args:
            pending_turn_id: 目标 pending turn ID。
            new_source_run_id: 当前 resumed run 的 run_id。

        Returns:
            重绑后的 pending turn 记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: ``new_source_run_id`` 为空字符串时抛出。
            PendingTurnResumeConflictError: 当前 state 非 ``RESUMING``（本方法仅
                允许 lease 在手的 resumer 调用）时抛出。
        """

        normalized_new_source_run_id = _normalize_text(new_source_run_id, field_name="new_source_run_id")
        existing = self.get_pending_turn(pending_turn_id)
        if existing is None:
            raise KeyError(f"pending turn 不存在: {pending_turn_id}")
        ensure_session_active(
            self._session_activity,
            session_id=existing.session_id,
            operation="rebind_source_run_id_for_resume",
            module=MODULE,
            target_name="pending turn",
        )
        if existing.state is not PendingConversationTurnState.RESUMING:
            raise PendingTurnResumeConflictError(existing.pending_turn_id)
        updated = PendingConversationTurn(
            pending_turn_id=existing.pending_turn_id,
            session_id=existing.session_id,
            scene_name=existing.scene_name,
            user_text=existing.user_text,
            source_run_id=normalized_new_source_run_id,
            created_at=existing.created_at,
            updated_at=_now_utc(),
            resumable=existing.resumable,
            state=existing.state,
            resume_source_json=existing.resume_source_json,
            resume_attempt_count=existing.resume_attempt_count,
            last_resume_error_message=existing.last_resume_error_message,
            pre_resume_state=existing.pre_resume_state,
            metadata=_normalize_metadata(existing.metadata),
        )
        self._records[updated.pending_turn_id] = updated
        return updated

    def record_resume_failure(
        self,
        pending_turn_id: str,
        *,
        error_message: str,
    ) -> PendingConversationTurn:
        """记录一次 pending turn 恢复失败，并把 RESUMING 状态回退到来源态。"""

        existing = self.get_pending_turn(pending_turn_id)
        if existing is None:
            raise KeyError(f"pending turn 不存在: {pending_turn_id}")
        ensure_session_active(
            self._session_activity,
            session_id=existing.session_id,
            operation="record_resume_failure",
            module=MODULE,
            target_name="pending turn",
        )
        # 失败路径一并回退 RESUMING lease；非 RESUMING 则仅写错误消息。
        if existing.state is PendingConversationTurnState.RESUMING:
            restored_state = existing.pre_resume_state or PendingConversationTurnState.ACCEPTED_BY_HOST
            new_pre_resume_state: PendingConversationTurnState | None = None
        else:
            restored_state = existing.state
            new_pre_resume_state = existing.pre_resume_state
        updated = PendingConversationTurn(
            pending_turn_id=existing.pending_turn_id,
            session_id=existing.session_id,
            scene_name=existing.scene_name,
            user_text=existing.user_text,
            source_run_id=existing.source_run_id,
            created_at=existing.created_at,
            updated_at=_now_utc(),
            resumable=existing.resumable,
            state=restored_state,
            resume_source_json=existing.resume_source_json,
            resume_attempt_count=existing.resume_attempt_count,
            last_resume_error_message=str(error_message).strip() or None,
            pre_resume_state=new_pre_resume_state,
            metadata=_normalize_metadata(existing.metadata),
        )
        self._records[updated.pending_turn_id] = updated
        return updated

    def delete_pending_turn(self, pending_turn_id: str) -> None:
        """删除指定 pending turn。"""

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        self._records.pop(normalized_pending_turn_id, None)

    def delete_by_session_id(self, session_id: str) -> int:
        """删除指定 session 的所有 pending turn。

        Args:
            session_id: 目标 session ID。

        Returns:
            被删除的记录数。

        Raises:
            无。
        """

        normalized = _normalize_text(session_id, field_name="session_id")
        to_delete = [
            tid for tid, record in self._records.items()
            if record.session_id == normalized
        ]
        for tid in to_delete:
            del self._records[tid]
        return len(to_delete)


class SQLitePendingConversationTurnStore:
    """基于 SQLite 的 pending turn 仓储。"""

    def __init__(
        self,
        host_store: HostStore,
        *,
        session_activity: "SessionActivityQueryProtocol | None" = None,
    ) -> None:
        """初始化 SQLite pending turn 仓储。

        Args:
            host_store: Host 共享 SQLite 存储。
            session_activity: 可选的 session 活性查询。装配后所有写入入口会在
                事务内先校验 session 是否仍处于非 CLOSED 状态；装配为 ``None``
                时退化为不做屏障的旧行为，仅用于独立 store 单元测试。

        Returns:
            无。

        Raises:
            无。
        """

        self._host_store = host_store
        self._session_activity: "SessionActivityQueryProtocol | None" = session_activity

    def upsert_pending_turn(
        self,
        *,
        session_id: str,
        scene_name: str,
        user_text: str,
        source_run_id: str,
        resumable: bool,
        state: PendingConversationTurnState,
        resume_source_json: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> PendingConversationTurn:
        """创建或更新当前 session/scene 的活跃 pending turn。

        Args:
            session_id: 当前会话 ID。
            scene_name: 当前场景名。
            user_text: 当前用户输入文本。
            source_run_id: 当前 Host run ID。
            resumable: 当前 pending turn 是否允许恢复。
            state: 当前 pending turn 的 Host 内部状态。
            resume_source_json: Host 持久化的 accepted/prepared 恢复快照。
            metadata: 交付上下文元数据。

        Returns:
            新建或更新后的 pending turn 记录。

        Raises:
            ValueError: 输入为空、恢复快照非法，或同一 session/scene 已存在不同 user_text 时抛出。
            RuntimeError: 写入成功后重新读取记录失败时抛出。
        """

        normalized_session_id = _normalize_text(session_id, field_name="session_id")
        normalized_scene_name = _normalize_text(scene_name, field_name="scene_name")
        normalized_user_text = _normalize_text(user_text, field_name="user_text")
        normalized_source_run_id = _normalize_text(source_run_id, field_name="source_run_id")
        normalized_resume_source_json = _normalize_resume_source_json(resume_source_json)
        normalized_metadata = _normalize_metadata(metadata)
        ensure_session_active(
            self._session_activity,
            session_id=normalized_session_id,
            operation="upsert_pending_turn",
            module=MODULE,
            target_name="pending turn",
        )
        now = _now_utc()
        conn = self._host_store.get_connection()
        metadata_json = json.dumps(normalized_metadata, ensure_ascii=False, sort_keys=True)
        pending_turn_id: str = ""
        # 先获取写锁再读取当前槽位，避免 check-then-insert 竞争窗口。
        with write_transaction(conn):
            existing = _get_session_pending_turn_slot_in_connection(
                conn,
                session_id=normalized_session_id,
                scene_name=normalized_scene_name,
            )
            if existing is None:
                pending_turn_id = f"pending_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO pending_conversation_turns (
                        pending_turn_id, session_id, scene_name, user_text,
                        source_run_id, created_at, updated_at, resumable,
                        state, resume_source_json, resume_attempt_count,
                        last_resume_error_message, pre_resume_state, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pending_turn_id,
                        normalized_session_id,
                        normalized_scene_name,
                        normalized_user_text,
                        normalized_source_run_id,
                        _serialize_dt(now),
                        _serialize_dt(now),
                        1 if resumable else 0,
                        state.value,
                        normalized_resume_source_json,
                        0,
                        None,
                        None,
                        metadata_json,
                    ),
                )
            else:
                if existing.user_text != normalized_user_text:
                    raise ValueError(
                        "当前 session/scene 已存在活跃 pending turn，不能写入不同 user_text: "
                        f"session_id={normalized_session_id}, scene_name={normalized_scene_name}"
                    )
                pending_turn_id = existing.pending_turn_id
                conn.execute(
                    """
                    UPDATE pending_conversation_turns
                    SET updated_at = ?,
                        source_run_id = ?,
                        resumable = ?,
                        state = ?,
                        resume_source_json = ?,
                        metadata_json = ?
                    WHERE pending_turn_id = ?
                    """,
                    (
                        _serialize_dt(now),
                        normalized_source_run_id,
                        1 if resumable else 0,
                        state.value,
                        normalized_resume_source_json,
                        metadata_json,
                        existing.pending_turn_id,
                    ),
                )
        updated = self.get_pending_turn(pending_turn_id)
        if updated is None:
            raise RuntimeError(f"pending turn 写入后读取失败: {pending_turn_id}")
        return updated

    def get_pending_turn(self, pending_turn_id: str) -> PendingConversationTurn | None:
        """按 ID 查询 pending turn。"""

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        conn = self._host_store.get_connection()
        row = conn.execute(
            "SELECT * FROM pending_conversation_turns WHERE pending_turn_id = ?",
            (normalized_pending_turn_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_pending_turn(dict(row))

    def get_session_pending_turn(
        self,
        *,
        session_id: str,
        scene_name: str,
    ) -> PendingConversationTurn | None:
        """按 session/scene 查询当前 pending turn。"""

        normalized_session_id = _normalize_text(session_id, field_name="session_id")
        normalized_scene_name = _normalize_text(scene_name, field_name="scene_name")
        conn = self._host_store.get_connection()
        return _get_session_pending_turn_in_connection(
            conn,
            session_id=normalized_session_id,
            scene_name=normalized_scene_name,
        )

    def list_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: PendingConversationTurnState | None = None,
        resumable_only: bool = False,
    ) -> list[PendingConversationTurn]:
        """列出 pending turn。"""

        conditions: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(_normalize_text(session_id, field_name="session_id"))
        if scene_name is not None:
            conditions.append("scene_name = ?")
            params.append(_normalize_text(scene_name, field_name="scene_name"))
        if state is not None:
            conditions.append("state = ?")
            params.append(state.value)
        if resumable_only:
            conditions.append("resumable = 1")
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        conn = self._host_store.get_connection()
        rows = conn.execute(
            f"SELECT * FROM pending_conversation_turns WHERE {where_clause} ORDER BY updated_at DESC",  # noqa: S608
            params,
        ).fetchall()
        return [_row_to_pending_turn(dict(row)) for row in rows]

    def update_state(
        self,
        pending_turn_id: str,
        *,
        state: PendingConversationTurnState,
    ) -> PendingConversationTurn:
        """更新 pending turn 的 Host 内部状态。"""

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        existing = self.get_pending_turn(normalized_pending_turn_id)
        if existing is None:
            raise KeyError(f"pending turn 不存在: {normalized_pending_turn_id}")
        ensure_session_active(
            self._session_activity,
            session_id=existing.session_id,
            operation="update_state",
            module=MODULE,
            target_name="pending turn",
        )
        conn = self._host_store.get_connection()
        with write_transaction(conn):
            cursor = conn.execute(
                """
                UPDATE pending_conversation_turns
                SET state = ?, updated_at = ?
                WHERE pending_turn_id = ?
                """,
                (state.value, _serialize_dt(_now_utc()), normalized_pending_turn_id),
            )
            rowcount = cursor.rowcount
        if rowcount == 0:
            raise KeyError(f"pending turn 不存在: {normalized_pending_turn_id}")
        updated = self.get_pending_turn(normalized_pending_turn_id)
        if updated is None:
            raise RuntimeError(f"pending turn 更新后读取失败: {normalized_pending_turn_id}")
        return updated

    def record_resume_attempt(
        self,
        pending_turn_id: str,
        *,
        max_attempts: int,
    ) -> PendingConversationTurn:
        """原子地 acquire 一次 resume lease。

        成功条件：当前 ``state`` 属于可 acquire 集合（ACCEPTED_BY_HOST /
        PREPARED_BY_HOST）且 ``resume_attempt_count`` 未达上限。此时把
        state 置为 ``RESUMING``、记录 ``pre_resume_state``、递增 attempt
        count，返回更新后的记录。

        Args:
            pending_turn_id: 目标 pending turn ID。
            max_attempts: 允许的最大恢复尝试次数。

        Returns:
            acquire 成功后的 pending turn 记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: ``max_attempts`` 非正、或恢复次数已达上限（此时
                记录原子删除）。
            PendingTurnResumeConflictError: 当前 state 不在可 acquire
                集合内（已被他人持有）。
            RuntimeError: acquire 成功后重新读取失败时抛出。
        """

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须是正整数")
        existing = self.get_pending_turn(normalized_pending_turn_id)
        if existing is None:
            raise KeyError(f"pending turn 不存在: {normalized_pending_turn_id}")
        ensure_session_active(
            self._session_activity,
            session_id=existing.session_id,
            operation="record_resume_attempt",
            module=MODULE,
            target_name="pending turn",
        )
        conn = self._host_store.get_connection()
        acquirable_state_values = tuple(state.value for state in _RESUME_ACQUIRABLE_STATES)
        acquirable_placeholders = ",".join("?" for _ in acquirable_state_values)
        # CAS UPDATE、观测、以及"达上限原子删除"全部收在同一 write_transaction 内，
        # 使 observed_row 的语义与 DELETE 共享事务快照：DELETE 以
        # resume_attempt_count >= max_attempts 作为与观测一致的谓词，
        # 关闭"观测后、删除前"被其他 resumer 推进/回退的 TOCTOU 窗口。
        acquired_row: sqlite3.Row | None = None
        observed_row: sqlite3.Row | None = None
        over_limit_deleted = False
        over_limit_session_id = ""
        with write_transaction(conn):
            cursor = conn.execute(
                f"""
                UPDATE pending_conversation_turns
                SET state = ?,
                    pre_resume_state = state,
                    resume_attempt_count = resume_attempt_count + 1,
                    last_resume_error_message = NULL,
                    updated_at = ?
                WHERE pending_turn_id = ?
                  AND state IN ({acquirable_placeholders})
                  AND resume_attempt_count < ?
                """,
                (
                    PendingConversationTurnState.RESUMING.value,
                    _serialize_dt(_now_utc()),
                    normalized_pending_turn_id,
                    *acquirable_state_values,
                    max_attempts,
                ),
            )
            if cursor.rowcount == 0:
                observed_row = conn.execute(
                    "SELECT * FROM pending_conversation_turns WHERE pending_turn_id = ?",
                    (normalized_pending_turn_id,),
                ).fetchone()
                if (
                    observed_row is not None
                    and int(observed_row["resume_attempt_count"] or 0) >= max_attempts
                ):
                    # 谓词与观测共识：只有仍然处于"达上限"语义的记录会被删除。
                    delete_cursor = conn.execute(
                        """
                        DELETE FROM pending_conversation_turns
                        WHERE pending_turn_id = ?
                          AND resume_attempt_count >= ?
                        """,
                        (normalized_pending_turn_id, max_attempts),
                    )
                    if delete_cursor.rowcount > 0:
                        over_limit_deleted = True
                        over_limit_session_id = str(observed_row["session_id"])
            else:
                acquired_row = conn.execute(
                    "SELECT * FROM pending_conversation_turns WHERE pending_turn_id = ?",
                    (normalized_pending_turn_id,),
                ).fetchone()

        if acquired_row is not None:
            return _row_to_pending_turn(dict(acquired_row))

        if over_limit_deleted:
            Log.warn(
                "pending turn 恢复次数已达到上限，已原子删除: "
                f"pending_turn_id={normalized_pending_turn_id}, "
                f"session_id={over_limit_session_id}, "
                f"max_attempts={max_attempts}",
                module=MODULE,
            )
            raise ValueError(
                "pending turn 恢复次数已达到上限，已删除: "
                f"pending_turn_id={normalized_pending_turn_id}, max_attempts={max_attempts}"
            )

        # CAS 失败分支：按事务内观测的最新状态分类处理。
        if observed_row is None:
            raise KeyError(f"pending turn 不存在: {normalized_pending_turn_id}")
        current_state = str(observed_row["state"])
        current_session_id = str(observed_row["session_id"])
        # 未达上限但 state 不在可 acquire 集合：被其他 resumer 占用。
        Log.warn(
            "pending turn 已被其他 resumer 持有，当前 acquire 失败: "
            f"pending_turn_id={normalized_pending_turn_id}, "
            f"session_id={current_session_id}, current_state={current_state}",
            module=MODULE,
        )
        raise PendingTurnResumeConflictError(normalized_pending_turn_id)

    def release_resume_lease(self, pending_turn_id: str) -> PendingConversationTurn | None:
        """把 RESUMING 的 pending turn 原子回退到 ``pre_resume_state``。

        Args:
            pending_turn_id: 目标 pending turn ID。

        Returns:
            回退后的 pending turn；若记录缺失或当前 state 非 RESUMING
            则返回 ``None`` / 原记录（幂等 no-op）。

        Raises:
            RuntimeError: 回退成功后重新读取失败时抛出。
        """

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        conn = self._host_store.get_connection()
        # write_transaction(BEGIN IMMEDIATE) 序列化 SELECT + UPDATE，避免并发下
        # "释放胜出方读到 state 已被再次 acquire" 的竞态。
        early_row: sqlite3.Row | None = None
        refreshed: sqlite3.Row | None = None
        with write_transaction(conn):
            row = conn.execute(
                "SELECT * FROM pending_conversation_turns WHERE pending_turn_id = ?",
                (normalized_pending_turn_id,),
            ).fetchone()
            if row is None:
                early_row = None
            elif str(row["state"]) != PendingConversationTurnState.RESUMING.value:
                early_row = row
            else:
                pre_resume_state_raw = row["pre_resume_state"] if "pre_resume_state" in row.keys() else None
                restored_state_value = (
                    str(pre_resume_state_raw)
                    if pre_resume_state_raw
                    else PendingConversationTurnState.ACCEPTED_BY_HOST.value
                )
                conn.execute(
                    """
                    UPDATE pending_conversation_turns
                    SET state = ?,
                        pre_resume_state = NULL,
                        updated_at = ?
                    WHERE pending_turn_id = ?
                    """,
                    (
                        restored_state_value,
                        _serialize_dt(_now_utc()),
                        normalized_pending_turn_id,
                    ),
                )
                refreshed = conn.execute(
                    "SELECT * FROM pending_conversation_turns WHERE pending_turn_id = ?",
                    (normalized_pending_turn_id,),
                ).fetchone()
        if refreshed is not None:
            return _row_to_pending_turn(dict(refreshed))
        if early_row is None:
            # 既未读取到记录（缺失），也未进入 UPDATE 分支。
            # 事务内 row is None 时 early_row 保持 None，此处直接返回 None。
            return None
        return _row_to_pending_turn(dict(early_row))

    def rebind_source_run_id_for_resume(
        self,
        pending_turn_id: str,
        *,
        new_source_run_id: str,
    ) -> PendingConversationTurn:
        """在持有 RESUMING lease 的前提下，把 ``source_run_id`` 原子重绑到当前 resumed run。

        CAS 条件：``state = 'resuming'``。lease 不在手时直接返回
        ``PendingTurnResumeConflictError``，杜绝"调用方误把别人的 pending turn
        绑到自己 run"的错接风险。

        Args:
            pending_turn_id: 目标 pending turn ID。
            new_source_run_id: 当前 resumed run 的 run_id。

        Returns:
            重绑后的 pending turn 记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: ``new_source_run_id`` 为空字符串时抛出。
            PendingTurnResumeConflictError: 当前 state 非 ``RESUMING`` 时抛出。
            RuntimeError: 重绑成功后重新读取失败时抛出。
        """

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        normalized_new_source_run_id = _normalize_text(new_source_run_id, field_name="new_source_run_id")
        existing = self.get_pending_turn(normalized_pending_turn_id)
        if existing is None:
            raise KeyError(f"pending turn 不存在: {normalized_pending_turn_id}")
        ensure_session_active(
            self._session_activity,
            session_id=existing.session_id,
            operation="rebind_source_run_id_for_resume",
            module=MODULE,
            target_name="pending turn",
        )
        conn = self._host_store.get_connection()
        refreshed: sqlite3.Row | None = None
        cas_failed_row_missing = False
        cas_failed_conflict = False
        with write_transaction(conn):
            cursor = conn.execute(
                """
                UPDATE pending_conversation_turns
                SET source_run_id = ?,
                    updated_at = ?
                WHERE pending_turn_id = ?
                  AND state = ?
                """,
                (
                    normalized_new_source_run_id,
                    _serialize_dt(_now_utc()),
                    normalized_pending_turn_id,
                    PendingConversationTurnState.RESUMING.value,
                ),
            )
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT * FROM pending_conversation_turns WHERE pending_turn_id = ?",
                    (normalized_pending_turn_id,),
                ).fetchone()
                if row is None:
                    cas_failed_row_missing = True
                else:
                    cas_failed_conflict = True
            else:
                refreshed = conn.execute(
                    "SELECT * FROM pending_conversation_turns WHERE pending_turn_id = ?",
                    (normalized_pending_turn_id,),
                ).fetchone()
        if cas_failed_row_missing:
            raise KeyError(f"pending turn 不存在: {normalized_pending_turn_id}")
        if cas_failed_conflict:
            raise PendingTurnResumeConflictError(normalized_pending_turn_id)
        if refreshed is None:
            raise RuntimeError(f"pending turn 重绑 source_run_id 后读取失败: {normalized_pending_turn_id}")
        return _row_to_pending_turn(dict(refreshed))

    def record_resume_failure(
        self,
        pending_turn_id: str,
        *,
        error_message: str,
    ) -> PendingConversationTurn:
        """记录一次 pending turn 恢复失败，并在 RESUMING 时回退到来源态。"""

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        normalized_error_message = str(error_message).strip() or None
        existing = self.get_pending_turn(normalized_pending_turn_id)
        if existing is None:
            raise KeyError(f"pending turn 不存在: {normalized_pending_turn_id}")
        ensure_session_active(
            self._session_activity,
            session_id=existing.session_id,
            operation="record_resume_failure",
            module=MODULE,
            target_name="pending turn",
        )
        conn = self._host_store.get_connection()
        refreshed: sqlite3.Row | None = None
        missing = False
        with write_transaction(conn):
            row = conn.execute(
                "SELECT * FROM pending_conversation_turns WHERE pending_turn_id = ?",
                (normalized_pending_turn_id,),
            ).fetchone()
            if row is None:
                missing = True
            else:
                current_state = str(row["state"])
                if current_state == PendingConversationTurnState.RESUMING.value:
                    pre_resume_state_raw = row["pre_resume_state"] if "pre_resume_state" in row.keys() else None
                    restored_state_value = (
                        str(pre_resume_state_raw)
                        if pre_resume_state_raw
                        else PendingConversationTurnState.ACCEPTED_BY_HOST.value
                    )
                    conn.execute(
                        """
                        UPDATE pending_conversation_turns
                        SET state = ?,
                            pre_resume_state = NULL,
                            last_resume_error_message = ?,
                            updated_at = ?
                        WHERE pending_turn_id = ?
                        """,
                        (
                            restored_state_value,
                            normalized_error_message,
                            _serialize_dt(_now_utc()),
                            normalized_pending_turn_id,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE pending_conversation_turns
                        SET last_resume_error_message = ?,
                            updated_at = ?
                        WHERE pending_turn_id = ?
                        """,
                        (
                            normalized_error_message,
                            _serialize_dt(_now_utc()),
                            normalized_pending_turn_id,
                        ),
                    )
                refreshed = conn.execute(
                    "SELECT * FROM pending_conversation_turns WHERE pending_turn_id = ?",
                    (normalized_pending_turn_id,),
                ).fetchone()
        if missing:
            raise KeyError(f"pending turn 不存在: {normalized_pending_turn_id}")
        if refreshed is None:
            raise RuntimeError(f"pending turn 更新后读取失败: {normalized_pending_turn_id}")
        return _row_to_pending_turn(dict(refreshed))

    def delete_pending_turn(self, pending_turn_id: str) -> None:
        """删除指定 pending turn。"""

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        conn = self._host_store.get_connection()
        with write_transaction(conn):
            conn.execute(
                "DELETE FROM pending_conversation_turns WHERE pending_turn_id = ?",
                (normalized_pending_turn_id,),
            )

    def delete_by_session_id(self, session_id: str) -> int:
        """删除指定 session 的所有 pending turn。

        Args:
            session_id: 目标 session ID。

        Returns:
            被删除的记录数。

        Raises:
            无。
        """

        normalized = _normalize_text(session_id, field_name="session_id")
        conn = self._host_store.get_connection()
        with write_transaction(conn):
            cursor = conn.execute(
                "DELETE FROM pending_conversation_turns WHERE session_id = ?",
                (normalized,),
            )
            rowcount = cursor.rowcount
        return rowcount


def _row_to_pending_turn(row: dict[str, Any]) -> PendingConversationTurn:
    """把 SQLite 行转换为 pending turn 记录。

    Args:
        row: SQLite 行数据。

    Returns:
        结构化 pending turn 记录。

    Raises:
        ValueError: 状态字段非法时抛出。
    """

    raw_metadata = row.get("metadata_json")
    metadata_object = json.loads(raw_metadata) if raw_metadata else {}
    metadata = normalize_execution_delivery_context(metadata_object if isinstance(metadata_object, dict) else None)
    raw_pre_resume_state = row.get("pre_resume_state")
    pre_resume_state_value = str(raw_pre_resume_state).strip() if raw_pre_resume_state else ""
    pre_resume_state = (
        PendingConversationTurnState(pre_resume_state_value) if pre_resume_state_value else None
    )
    return PendingConversationTurn(
        pending_turn_id=str(row["pending_turn_id"]),
        session_id=str(row["session_id"]),
        scene_name=str(row["scene_name"]),
        user_text=str(row["user_text"]),
        source_run_id=str(row["source_run_id"]),
        created_at=_parse_dt(str(row["created_at"])),
        updated_at=_parse_dt(str(row["updated_at"])),
        resumable=bool(int(row["resumable"])),
        state=PendingConversationTurnState(str(row["state"])),
        resume_source_json=_normalize_resume_source_json(str(row.get("resume_source_json") or "")),
        resume_attempt_count=int(row.get("resume_attempt_count") or 0),
        last_resume_error_message=str(row["last_resume_error_message"]).strip()
        if row.get("last_resume_error_message") is not None
        else None,
        pre_resume_state=pre_resume_state,
        metadata=metadata,
    )


def _get_session_pending_turn_in_connection(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    scene_name: str,
) -> PendingConversationTurn | None:
    """在指定连接上按 session/scene 查询当前 pending turn。

    Args:
        conn: 调用方持有的 SQLite 连接。
        session_id: 已规范化的会话 ID。
        scene_name: 已规范化的场景名。

    Returns:
        匹配记录；不存在时返回 ``None``。

    Raises:
        无。
    """

    row = conn.execute(
        """
        SELECT * FROM pending_conversation_turns
        WHERE session_id = ? AND scene_name = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (session_id, scene_name),
    ).fetchone()
    if row is None:
        return None
    return _row_to_pending_turn(dict(row))


def _get_session_pending_turn_slot_in_connection(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    scene_name: str,
) -> _PendingTurnSlot | None:
    """在指定连接上读取锁内冲突判断所需的最小槽位信息。

    Args:
        conn: 调用方持有的 SQLite 连接。
        session_id: 已规范化的会话 ID。
        scene_name: 已规范化的场景名。

    Returns:
        仅包含 pending_turn_id 与 user_text 的槽位信息；不存在时返回 ``None``。

    Raises:
        无。
    """

    row = conn.execute(
        """
        SELECT pending_turn_id, user_text
        FROM pending_conversation_turns
        WHERE session_id = ? AND scene_name = ?
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (session_id, scene_name),
    ).fetchone()
    if row is None:
        return None
    return _PendingTurnSlot(
        pending_turn_id=str(row["pending_turn_id"]),
        user_text=str(row["user_text"]),
    )


__all__ = [
    "InMemoryPendingConversationTurnStore",
    "PendingConversationTurn",
    "PendingConversationTurnState",
    "PendingTurnResumeConflictError",
    "SQLitePendingConversationTurnStore",
]
