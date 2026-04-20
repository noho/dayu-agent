"""reply outbox 仓储与默认实现。

该模块定义 Host 内用于托管可选 reply outbox 真源的仓储实现。
reply outbox 与 pending conversation turn 完全独立：

- pending conversation turn 表示 Host 内部执行是否仍可恢复
- reply outbox 表示某条最终回复是否已被显式提交为待交付记录

本模块只负责真源持久化与状态流转，不负责具体渠道发送。
"""

from __future__ import annotations

import json
import sqlite3
import uuid


from dayu.contracts.execution_metadata import ExecutionDeliveryContext, normalize_execution_delivery_context
from dayu.contracts.reply_outbox import ReplyOutboxRecord, ReplyOutboxState, ReplyOutboxSubmitRequest
from dayu.host.host_store import HostStore


from dayu.host._datetime_utils import now_utc as _now_utc, parse_dt as _parse_dt, serialize_dt as _serialize_dt


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


def _normalize_error_message(value: str | None) -> str | None:
    """规范化失败消息。"""

    normalized = str(value or "").strip()
    return normalized or None


def _normalize_metadata(metadata: ExecutionDeliveryContext | None) -> ExecutionDeliveryContext:
    """规范化交付上下文。"""

    return normalize_execution_delivery_context(metadata)


def _serialize_metadata(metadata: ExecutionDeliveryContext) -> str:
    """序列化交付上下文 JSON。"""

    return json.dumps(metadata, ensure_ascii=False, sort_keys=True)


def _normalize_submit_request(request: ReplyOutboxSubmitRequest) -> ReplyOutboxSubmitRequest:
    """规范化提交请求。"""

    return ReplyOutboxSubmitRequest(
        delivery_key=_normalize_text(request.delivery_key, field_name="delivery_key"),
        session_id=_normalize_text(request.session_id, field_name="session_id"),
        scene_name=_normalize_text(request.scene_name, field_name="scene_name"),
        source_run_id=_normalize_text(request.source_run_id, field_name="source_run_id"),
        reply_content=_normalize_text(request.reply_content, field_name="reply_content"),
        metadata=_normalize_metadata(request.metadata),
    )


def _ensure_submit_request_matches(existing: ReplyOutboxRecord, request: ReplyOutboxSubmitRequest) -> None:
    """校验相同 delivery_key 的提交负载一致。"""

    if (
        existing.delivery_key != request.delivery_key
        or existing.session_id != request.session_id
        or existing.scene_name != request.scene_name
        or existing.source_run_id != request.source_run_id
        or existing.reply_content != request.reply_content
        or existing.metadata != request.metadata
    ):
        raise ValueError(
            "delivery_key 已存在且负载不一致: "
            f"delivery_key={request.delivery_key}"
        )


class InMemoryReplyOutboxStore:
    """最小化内存版 reply outbox 仓储。

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

        self._records: dict[str, ReplyOutboxRecord] = {}
        self._delivery_key_index: dict[str, str] = {}

    def submit_reply(self, request: ReplyOutboxSubmitRequest) -> ReplyOutboxRecord:
        """显式提交待交付回复。

        Args:
            request: 提交请求。

        Returns:
            创建或幂等返回的交付记录。

        Raises:
            ValueError: 请求非法或同幂等键负载不一致时抛出。
        """

        normalized_request = _normalize_submit_request(request)
        existing = self.get_by_delivery_key(normalized_request.delivery_key)
        if existing is not None:
            _ensure_submit_request_matches(existing, normalized_request)
            return existing

        now = _now_utc()
        record = ReplyOutboxRecord(
            delivery_id=f"delivery_{uuid.uuid4().hex[:12]}",
            delivery_key=normalized_request.delivery_key,
            session_id=normalized_request.session_id,
            scene_name=normalized_request.scene_name,
            source_run_id=normalized_request.source_run_id,
            reply_content=normalized_request.reply_content,
            metadata=normalized_request.metadata,
            state=ReplyOutboxState.PENDING_DELIVERY,
            created_at=now,
            updated_at=now,
        )
        self._records[record.delivery_id] = record
        self._delivery_key_index[record.delivery_key] = record.delivery_id
        return record

    def get_reply(self, delivery_id: str) -> ReplyOutboxRecord | None:
        """按 ID 查询交付记录。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            匹配记录；不存在时返回 ``None``。

        Raises:
            无。
        """

        normalized_delivery_id = str(delivery_id or "").strip()
        if not normalized_delivery_id:
            return None
        return self._records.get(normalized_delivery_id)

    def get_by_delivery_key(self, delivery_key: str) -> ReplyOutboxRecord | None:
        """按幂等键查询交付记录。

        Args:
            delivery_key: 业务侧幂等键。

        Returns:
            匹配记录；不存在时返回 ``None``。

        Raises:
            无。
        """

        normalized_key = str(delivery_key or "").strip()
        if not normalized_key:
            return None
        delivery_id = self._delivery_key_index.get(normalized_key)
        if delivery_id is None:
            return None
        return self._records.get(delivery_id)

    def list_replies(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: ReplyOutboxState | None = None,
    ) -> list[ReplyOutboxRecord]:
        """列出交付记录。

        Args:
            session_id: 可选 session 过滤。
            scene_name: 可选 scene 过滤。
            state: 可选状态过滤。

        Returns:
            匹配记录列表。

        Raises:
            无。
        """

        records = list(self._records.values())
        if session_id is not None:
            normalized_session_id = str(session_id or "").strip()
            records = [record for record in records if record.session_id == normalized_session_id]
        if scene_name is not None:
            normalized_scene_name = str(scene_name or "").strip()
            records = [record for record in records if record.scene_name == normalized_scene_name]
        if state is not None:
            records = [record for record in records if record.state == state]
        return sorted(records, key=lambda record: (record.updated_at, record.created_at), reverse=True)

    def claim_reply(self, delivery_id: str) -> ReplyOutboxRecord:
        """把记录推进到发送中状态。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 当前状态不允许 claim 时抛出。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        existing = self.get_reply(normalized_delivery_id)
        if existing is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if existing.state not in {ReplyOutboxState.PENDING_DELIVERY, ReplyOutboxState.FAILED_RETRYABLE}:
            raise ValueError(
                "reply delivery 当前状态不允许 claim: "
                f"delivery_id={delivery_id}, state={existing.state.value}"
            )
        updated = ReplyOutboxRecord(
            delivery_id=existing.delivery_id,
            delivery_key=existing.delivery_key,
            session_id=existing.session_id,
            scene_name=existing.scene_name,
            source_run_id=existing.source_run_id,
            reply_content=existing.reply_content,
            metadata=existing.metadata,
            state=ReplyOutboxState.DELIVERY_IN_PROGRESS,
            created_at=existing.created_at,
            updated_at=_now_utc(),
            delivery_attempt_count=existing.delivery_attempt_count + 1,
            last_error_message=None,
        )
        self._records[updated.delivery_id] = updated
        return updated

    def mark_delivered(self, delivery_id: str) -> ReplyOutboxRecord:
        """标记记录已完成交付。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 当前状态不允许 delivered 时抛出。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        existing = self.get_reply(normalized_delivery_id)
        if existing is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if existing.state == ReplyOutboxState.DELIVERED:
            return existing
        if existing.state != ReplyOutboxState.DELIVERY_IN_PROGRESS:
            raise ValueError(
                "reply delivery 当前状态不允许 delivered: "
                f"delivery_id={delivery_id}, state={existing.state.value}"
            )
        updated = ReplyOutboxRecord(
            delivery_id=existing.delivery_id,
            delivery_key=existing.delivery_key,
            session_id=existing.session_id,
            scene_name=existing.scene_name,
            source_run_id=existing.source_run_id,
            reply_content=existing.reply_content,
            metadata=existing.metadata,
            state=ReplyOutboxState.DELIVERED,
            created_at=existing.created_at,
            updated_at=_now_utc(),
            delivery_attempt_count=existing.delivery_attempt_count,
            last_error_message=None,
        )
        self._records[updated.delivery_id] = updated
        return updated

    def mark_failed(
        self,
        delivery_id: str,
        *,
        retryable: bool,
        error_message: str,
    ) -> ReplyOutboxRecord:
        """标记记录交付失败。

        Args:
            delivery_id: 交付记录 ID。
            retryable: 是否允许后续再次 claim。
            error_message: 失败消息。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 已完成交付的记录重复标记失败时抛出。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        existing = self.get_reply(normalized_delivery_id)
        if existing is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if existing.state == ReplyOutboxState.DELIVERED:
            raise ValueError(
                "已完成交付的 reply delivery 不能再标记失败: "
                f"delivery_id={delivery_id}"
            )
        normalized_error_message = _normalize_text(error_message, field_name="error_message")
        updated = ReplyOutboxRecord(
            delivery_id=existing.delivery_id,
            delivery_key=existing.delivery_key,
            session_id=existing.session_id,
            scene_name=existing.scene_name,
            source_run_id=existing.source_run_id,
            reply_content=existing.reply_content,
            metadata=existing.metadata,
            state=(
                ReplyOutboxState.FAILED_RETRYABLE
                if retryable else ReplyOutboxState.FAILED_TERMINAL
            ),
            created_at=existing.created_at,
            updated_at=_now_utc(),
            delivery_attempt_count=existing.delivery_attempt_count,
            last_error_message=normalized_error_message,
        )
        self._records[updated.delivery_id] = updated
        return updated

    def delete_by_session_id(self, session_id: str) -> int:
        """删除指定 session 的所有交付记录。

        Args:
            session_id: 目标 session ID。

        Returns:
            被删除的记录数。

        Raises:
            无。
        """

        normalized = _normalize_text(session_id, field_name="session_id")
        to_delete = [
            did for did, record in self._records.items()
            if record.session_id == normalized
        ]
        for did in to_delete:
            del self._records[did]
        return len(to_delete)


class SQLiteReplyOutboxStore:
    """SQLite 版 reply outbox 仓储。"""

    def __init__(self, host_store: HostStore) -> None:
        """初始化 SQLite 仓储。

        Args:
            host_store: 宿主层 SQLite 存储。

        Returns:
            无。

        Raises:
            无。
        """

        self._host_store = host_store

    def submit_reply(self, request: ReplyOutboxSubmitRequest) -> ReplyOutboxRecord:
        """显式提交待交付回复。

        Args:
            request: 提交请求。

        Returns:
            创建或幂等返回的交付记录。

        Raises:
            ValueError: 请求非法或同幂等键负载不一致时抛出。
            RuntimeError: 创建后或回读既有记录失败时抛出。
        """

        normalized_request = _normalize_submit_request(request)
        now = _now_utc()
        delivery_id = f"delivery_{uuid.uuid4().hex[:12]}"
        conn = self._host_store.get_connection()
        # INSERT OR IGNORE 用于在数据库层原子收敛相同 delivery_key 的并发首写。
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO reply_outbox (
                delivery_id, delivery_key, session_id, scene_name, source_run_id,
                reply_content, state, delivery_attempt_count, last_error_message,
                created_at, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                delivery_id,
                normalized_request.delivery_key,
                normalized_request.session_id,
                normalized_request.scene_name,
                normalized_request.source_run_id,
                normalized_request.reply_content,
                ReplyOutboxState.PENDING_DELIVERY.value,
                0,
                None,
                _serialize_dt(now),
                _serialize_dt(now),
                _serialize_metadata(normalized_request.metadata),
            ),
        )
        conn.commit()
        if cursor.rowcount == 0:
            existing = self.get_by_delivery_key(normalized_request.delivery_key)
            if existing is None:
                raise RuntimeError(
                    "reply delivery 幂等回读失败: "
                    f"delivery_key={normalized_request.delivery_key}"
                )
            _ensure_submit_request_matches(existing, normalized_request)
            return existing
        created = self.get_reply(delivery_id)
        if created is None:
            raise RuntimeError(f"reply delivery 创建后读取失败: {delivery_id}")
        return created

    def get_reply(self, delivery_id: str) -> ReplyOutboxRecord | None:
        """按 ID 查询交付记录。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            匹配记录；不存在时返回 ``None``。

        Raises:
            ValueError: delivery_id 为空时抛出。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        conn = self._host_store.get_connection()
        row = conn.execute(
            "SELECT * FROM reply_outbox WHERE delivery_id = ?",
            (normalized_delivery_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_reply_outbox_record(row)

    def get_by_delivery_key(self, delivery_key: str) -> ReplyOutboxRecord | None:
        """按幂等键查询交付记录。

        Args:
            delivery_key: 业务侧幂等键。

        Returns:
            匹配记录；不存在时返回 ``None``。

        Raises:
            ValueError: delivery_key 为空时抛出。
        """

        normalized_delivery_key = _normalize_text(delivery_key, field_name="delivery_key")
        conn = self._host_store.get_connection()
        row = conn.execute(
            "SELECT * FROM reply_outbox WHERE delivery_key = ?",
            (normalized_delivery_key,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_reply_outbox_record(row)

    def list_replies(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: ReplyOutboxState | None = None,
    ) -> list[ReplyOutboxRecord]:
        """列出交付记录。

        Args:
            session_id: 可选 session 过滤。
            scene_name: 可选 scene 过滤。
            state: 可选状态过滤。

        Returns:
            匹配记录列表。

        Raises:
            ValueError: 过滤字段为空字符串时抛出。
        """

        clauses: list[str] = []
        params: list[str] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(_normalize_text(session_id, field_name="session_id"))
        if scene_name is not None:
            clauses.append("scene_name = ?")
            params.append(_normalize_text(scene_name, field_name="scene_name"))
        if state is not None:
            clauses.append("state = ?")
            params.append(state.value)

        sql = "SELECT * FROM reply_outbox"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC, created_at DESC"
        conn = self._host_store.get_connection()
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [_row_to_reply_outbox_record(row) for row in rows]

    def claim_reply(self, delivery_id: str) -> ReplyOutboxRecord:
        """把记录推进到发送中状态。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 当前状态不允许 claim 时抛出。
            RuntimeError: 更新后读取失败时抛出。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        conn = self._host_store.get_connection()
        cursor = conn.execute(
            """
            UPDATE reply_outbox
            SET state = ?,
                delivery_attempt_count = delivery_attempt_count + 1,
                last_error_message = ?,
                updated_at = ?
            WHERE delivery_id = ?
              AND state IN (?, ?)
            """,
            (
                ReplyOutboxState.DELIVERY_IN_PROGRESS.value,
                None,
                _serialize_dt(_now_utc()),
                normalized_delivery_id,
                ReplyOutboxState.PENDING_DELIVERY.value,
                ReplyOutboxState.FAILED_RETRYABLE.value,
            ),
        )
        conn.commit()
        updated = self.get_reply(normalized_delivery_id)
        if cursor.rowcount == 0 and updated is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if updated is None:
            raise RuntimeError(f"reply delivery 更新后读取失败: {normalized_delivery_id}")
        if cursor.rowcount == 0:
            raise ValueError(
                "reply delivery 当前状态不允许 claim: "
                f"delivery_id={delivery_id}, state={updated.state.value}"
            )
        return updated

    def mark_delivered(self, delivery_id: str) -> ReplyOutboxRecord:
        """标记记录已完成交付。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 当前状态不允许 delivered 时抛出。
            RuntimeError: 更新后读取失败时抛出。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        existing = self.get_reply(normalized_delivery_id)
        if existing is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if existing.state == ReplyOutboxState.DELIVERED:
            return existing
        conn = self._host_store.get_connection()
        cursor = conn.execute(
            """
            UPDATE reply_outbox
            SET state = ?,
                last_error_message = ?,
                updated_at = ?
            WHERE delivery_id = ?
              AND state = ?
            """,
            (
                ReplyOutboxState.DELIVERED.value,
                None,
                _serialize_dt(_now_utc()),
                existing.delivery_id,
                ReplyOutboxState.DELIVERY_IN_PROGRESS.value,
            ),
        )
        conn.commit()
        updated = self.get_reply(existing.delivery_id)
        if updated is None:
            raise RuntimeError(f"reply delivery 更新后读取失败: {existing.delivery_id}")
        if cursor.rowcount == 0:
            if updated.state == ReplyOutboxState.DELIVERED:
                return updated
            raise ValueError(
                "reply delivery 当前状态不允许 delivered: "
                f"delivery_id={delivery_id}, state={updated.state.value}"
            )
        return updated

    def mark_failed(
        self,
        delivery_id: str,
        *,
        retryable: bool,
        error_message: str,
    ) -> ReplyOutboxRecord:
        """标记记录交付失败。

        Args:
            delivery_id: 交付记录 ID。
            retryable: 是否允许后续再次 claim。
            error_message: 失败消息。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 已完成交付的记录重复标记失败时抛出。
            RuntimeError: 更新后读取失败时抛出。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        existing = self.get_reply(normalized_delivery_id)
        if existing is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if existing.state == ReplyOutboxState.DELIVERED:
            raise ValueError(
                "已完成交付的 reply delivery 不能再标记失败: "
                f"delivery_id={delivery_id}"
            )
        normalized_error_message = _normalize_text(error_message, field_name="error_message")
        conn = self._host_store.get_connection()
        conn.execute(
            """
            UPDATE reply_outbox
            SET state = ?,
                last_error_message = ?,
                updated_at = ?
            WHERE delivery_id = ?
            """,
            (
                ReplyOutboxState.FAILED_RETRYABLE.value if retryable else ReplyOutboxState.FAILED_TERMINAL.value,
                normalized_error_message,
                _serialize_dt(_now_utc()),
                existing.delivery_id,
            ),
        )
        conn.commit()
        updated = self.get_reply(existing.delivery_id)
        if updated is None:
            raise RuntimeError(f"reply delivery 更新后读取失败: {existing.delivery_id}")
        return updated

    def delete_by_session_id(self, session_id: str) -> int:
        """删除指定 session 的所有交付记录。

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
            "DELETE FROM reply_outbox WHERE session_id = ?",
            (normalized,),
        )
        conn.commit()
        return cursor.rowcount


def _row_to_reply_outbox_record(row: sqlite3.Row) -> ReplyOutboxRecord:
    """将 SQLite 行转换为 reply outbox 记录。

    Args:
        row: SQLite 查询结果行。

        Returns:
            解析后的 reply outbox 记录。

        Raises:
            ValueError: metadata_json 结构非法时抛出。
    """

    metadata_raw = str(row["metadata_json"] or "{}")
    metadata_payload = json.loads(metadata_raw)
    if not isinstance(metadata_payload, dict):
        raise ValueError("reply outbox metadata_json 必须是 JSON object")
    metadata = normalize_execution_delivery_context(metadata_payload)
    return ReplyOutboxRecord(
        delivery_id=str(row["delivery_id"]),
        delivery_key=str(row["delivery_key"]),
        session_id=str(row["session_id"]),
        scene_name=str(row["scene_name"]),
        source_run_id=str(row["source_run_id"]),
        reply_content=str(row["reply_content"]),
        metadata=metadata,
        state=ReplyOutboxState(str(row["state"])),
        created_at=_parse_dt(str(row["created_at"])),
        updated_at=_parse_dt(str(row["updated_at"])),
        delivery_attempt_count=int(row["delivery_attempt_count"]),
        last_error_message=_normalize_error_message(str(row["last_error_message"] or "")),
    )


__all__ = [
    "InMemoryReplyOutboxStore",
    "SQLiteReplyOutboxStore",
]
