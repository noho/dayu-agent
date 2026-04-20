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
from typing import Any

from dayu.contracts.execution_metadata import (
    ExecutionDeliveryContext,
    empty_execution_delivery_context,
    normalize_execution_delivery_context,
)
from dayu.host.host_store import HostStore


from dayu.host._datetime_utils import now_utc as _now_utc, parse_dt as _parse_dt, serialize_dt as _serialize_dt


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
    """pending conversation turn 的 Host 内部执行状态。"""

    ACCEPTED_BY_HOST = "accepted_by_host"
    PREPARED_BY_HOST = "prepared_by_host"
    SENT_TO_LLM = "sent_to_llm"


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

    def __init__(self) -> None:
        """初始化内存仓储。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self._records: dict[str, PendingConversationTurn] = {}

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
        if existing.resume_attempt_count >= max_attempts:
            raise ValueError(
                "pending turn 恢复次数已达到上限: "
                f"pending_turn_id={pending_turn_id}, max_attempts={max_attempts}"
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
            state=existing.state,
            resume_source_json=existing.resume_source_json,
            resume_attempt_count=existing.resume_attempt_count + 1,
            last_resume_error_message=None,
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
        """记录一次 pending turn 恢复失败。"""

        existing = self.get_pending_turn(pending_turn_id)
        if existing is None:
            raise KeyError(f"pending turn 不存在: {pending_turn_id}")
        updated = PendingConversationTurn(
            pending_turn_id=existing.pending_turn_id,
            session_id=existing.session_id,
            scene_name=existing.scene_name,
            user_text=existing.user_text,
            source_run_id=existing.source_run_id,
            created_at=existing.created_at,
            updated_at=_now_utc(),
            resumable=existing.resumable,
            state=existing.state,
            resume_source_json=existing.resume_source_json,
            resume_attempt_count=existing.resume_attempt_count,
            last_resume_error_message=str(error_message).strip() or None,
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

    def __init__(self, host_store: HostStore) -> None:
        """初始化 SQLite pending turn 仓储。

        Args:
            host_store: Host 共享 SQLite 存储。

        Returns:
            无。

        Raises:
            无。
        """

        self._host_store = host_store

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
        now = _now_utc()
        conn = self._host_store.get_connection()
        metadata_json = json.dumps(normalized_metadata, ensure_ascii=False, sort_keys=True)
        pending_turn_id: str | None = None
        try:
            # 先获取写锁再读取当前槽位，避免 check-then-insert 竞争窗口。
            conn.execute("BEGIN IMMEDIATE")
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
                        last_resume_error_message, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            conn.commit()
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        if pending_turn_id is None:
            raise RuntimeError("pending turn 写入后缺少 pending_turn_id")
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
        conn = self._host_store.get_connection()
        cursor = conn.execute(
            """
            UPDATE pending_conversation_turns
            SET state = ?, updated_at = ?
            WHERE pending_turn_id = ?
            """,
            (state.value, _serialize_dt(_now_utc()), normalized_pending_turn_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
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
        """在未达到上限时原子记录一次 pending turn 恢复尝试。"""

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        if max_attempts <= 0:
            raise ValueError("max_attempts 必须是正整数")
        conn = self._host_store.get_connection()
        cursor = conn.execute(
            """
            UPDATE pending_conversation_turns
            SET resume_attempt_count = resume_attempt_count + 1,
                last_resume_error_message = NULL,
                updated_at = ?
            WHERE pending_turn_id = ?
              AND resume_attempt_count < ?
            """,
            (_serialize_dt(_now_utc()), normalized_pending_turn_id, max_attempts),
        )
        conn.commit()
        if cursor.rowcount == 0:
            existing = self.get_pending_turn(normalized_pending_turn_id)
            if existing is None:
                raise KeyError(f"pending turn 不存在: {normalized_pending_turn_id}")
            raise ValueError(
                "pending turn 恢复次数已达到上限: "
                f"pending_turn_id={normalized_pending_turn_id}, max_attempts={max_attempts}"
            )
        updated = self.get_pending_turn(normalized_pending_turn_id)
        if updated is None:
            raise RuntimeError(f"pending turn 更新后读取失败: {normalized_pending_turn_id}")
        return updated

    def record_resume_failure(
        self,
        pending_turn_id: str,
        *,
        error_message: str,
    ) -> PendingConversationTurn:
        """记录一次 pending turn 恢复失败。"""

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        normalized_error_message = str(error_message).strip() or None
        conn = self._host_store.get_connection()
        cursor = conn.execute(
            """
            UPDATE pending_conversation_turns
            SET last_resume_error_message = ?,
                updated_at = ?
            WHERE pending_turn_id = ?
            """,
            (normalized_error_message, _serialize_dt(_now_utc()), normalized_pending_turn_id),
        )
        conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"pending turn 不存在: {normalized_pending_turn_id}")
        updated = self.get_pending_turn(normalized_pending_turn_id)
        if updated is None:
            raise RuntimeError(f"pending turn 更新后读取失败: {normalized_pending_turn_id}")
        return updated

    def delete_pending_turn(self, pending_turn_id: str) -> None:
        """删除指定 pending turn。"""

        normalized_pending_turn_id = _normalize_text(pending_turn_id, field_name="pending_turn_id")
        conn = self._host_store.get_connection()
        conn.execute(
            "DELETE FROM pending_conversation_turns WHERE pending_turn_id = ?",
            (normalized_pending_turn_id,),
        )
        conn.commit()

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
        cursor = conn.execute(
            "DELETE FROM pending_conversation_turns WHERE session_id = ?",
            (normalized,),
        )
        conn.commit()
        return cursor.rowcount


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
    "SQLitePendingConversationTurnStore",
]
