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
from datetime import timedelta
from typing import TYPE_CHECKING


from dayu.contracts.execution_metadata import ExecutionDeliveryContext, normalize_execution_delivery_context
from dayu.contracts.reply_outbox import ReplyOutboxRecord, ReplyOutboxState, ReplyOutboxSubmitRequest
from dayu.host._session_barrier import ensure_session_active
from dayu.host.host_store import HostStore, write_transaction
from dayu.host.lease import LeaseExpiredError, generate_lease_id
from dayu.log import Log

if TYPE_CHECKING:
    from dayu.host.protocols import SessionActivityQueryProtocol


MODULE = "HOST.REPLY_OUTBOX_STORE"


STALE_IN_PROGRESS_ERROR_MESSAGE = "stale in_progress recovery"

# stale cleanup 的 max_age 下界。低于该值时正在执行交付的 record（刚 claim、
# 尚未 mark_delivered）会与 cleanup 的 CAS（``state='delivery_in_progress'``）
# 在同一时间窗内竞争，导致已成功交付被错误地回退为 FAILED_RETRYABLE，引发
# 重复投递或交付丢失。下界由 store 自身强制，避免调用方因配置错误踩到 race
# window。如确需更小的窗口，需要先把 mark_delivered 与 cleanup 之间的隔离
# 机制（例如 fence token / leased ownership）补齐再放开下界。
MIN_STALE_AGE = timedelta(minutes=5)


from dayu.host._datetime_utils import now_utc as _now_utc, parse_dt as _parse_dt, serialize_dt as _serialize_dt


def _validate_stale_max_age(max_age: timedelta) -> None:
    """校验 stale cleanup 的 max_age 不低于 ``MIN_STALE_AGE``。

    Args:
        max_age: 调用方传入的 stale 阈值。

    Returns:
        无。

    Raises:
        ValueError: 当 ``max_age`` 小于 ``MIN_STALE_AGE`` 时抛出，强制调用方
            为正在执行交付的 record 留出足够的非竞态窗口。
    """

    if max_age < MIN_STALE_AGE:
        raise ValueError(
            "cleanup_stale_in_progress_deliveries 的 max_age 不能小于 "
            f"{MIN_STALE_AGE}; 当前传入={max_age}; "
            "下界用于隔离 mark_delivered CAS 与 stale cleanup CAS 的竞态窗口"
        )


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

    def __init__(
        self,
        *,
        session_activity: "SessionActivityQueryProtocol | None" = None,
    ) -> None:
        """初始化内存仓储。

        Args:
            session_activity: 可选的 session 活性查询；装配后 ``submit_reply``
                在 session 已 CLOSED 时抛 ``SessionClosedError``；传 ``None``
                时退化为不做屏障的旧行为，仅用于独立 store 单元测试。

        Returns:
            无。

        Raises:
            无。
        """

        self._records: dict[str, ReplyOutboxRecord] = {}
        self._delivery_key_index: dict[str, str] = {}
        self._session_activity: "SessionActivityQueryProtocol | None" = session_activity

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
        ensure_session_active(
            self._session_activity,
            session_id=normalized_request.session_id,
            operation="submit_reply",
            module=MODULE,
            target_name="reply outbox",
        )
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
        """把记录推进到发送中状态，并分配新的 ``lease_id``。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            更新后的交付记录，``lease_id`` 为本次 acquire 分配的新 fence token。

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
        new_lease_id = generate_lease_id()
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
            lease_id=new_lease_id,
        )
        self._records[updated.delivery_id] = updated
        return updated

    def mark_delivered(self, delivery_id: str, *, lease_id: str) -> ReplyOutboxRecord:
        """标记记录已完成交付。

        Args:
            delivery_id: 交付记录 ID。
            lease_id: claim 时返回的 fence token；必须与当前持有 lease 完全匹配。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 当前状态不允许 delivered 时抛出。
            LeaseExpiredError: 持有的 lease 已被抢占（cleanup 抢占改写）。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        normalized_lease_id = _normalize_text(lease_id, field_name="lease_id")
        existing = self.get_reply(normalized_delivery_id)
        if existing is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if existing.state == ReplyOutboxState.DELIVERED:
            # DELIVERED 是吸收态：仅当持有者用同一 lease 重试时才幂等返回；
            # 任何 lease mismatch（含旧 holder 在 cleanup 抢占后迟到）必须暴露
            # 为 LeaseExpiredError，避免把"写入未生效、ownership 已变化"伪装成成功。
            if existing.lease_id == normalized_lease_id:
                return existing
            raise LeaseExpiredError(
                "reply delivery lease 已失效: "
                f"delivery_id={delivery_id}",
                record_id=existing.delivery_id,
                lease_id=normalized_lease_id,
            )
        if existing.state != ReplyOutboxState.DELIVERY_IN_PROGRESS:
            raise ValueError(
                "reply delivery 当前状态不允许 delivered: "
                f"delivery_id={delivery_id}, state={existing.state.value}"
            )
        if existing.lease_id != normalized_lease_id:
            raise LeaseExpiredError(
                "reply delivery lease 已失效: "
                f"delivery_id={delivery_id}",
                record_id=existing.delivery_id,
                lease_id=normalized_lease_id,
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
            # 终态保留 lease_id，使后续合法 holder 自重试可幂等返回；
            # 旧 holder 拿不同 lease 重试时则在上面的吸收态分支被识别为 mismatch。
            lease_id=existing.lease_id,
        )
        self._records[updated.delivery_id] = updated
        return updated

    def mark_failed(
        self,
        delivery_id: str,
        *,
        retryable: bool,
        error_message: str,
        lease_id: str,
    ) -> ReplyOutboxRecord:
        """标记记录交付失败。

        Args:
            delivery_id: 交付记录 ID。
            retryable: 是否允许后续再次 claim。
            error_message: 失败消息。
            lease_id: claim 时返回的 fence token；必须与当前持有 lease 完全匹配。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 已完成交付的记录重复标记失败时抛出。
            LeaseExpiredError: 持有的 lease 已被抢占（cleanup 抢占改写）。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        normalized_lease_id = _normalize_text(lease_id, field_name="lease_id")
        existing = self.get_reply(normalized_delivery_id)
        if existing is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if existing.state == ReplyOutboxState.DELIVERED:
            raise ValueError(
                "已完成交付的 reply delivery 不能再标记失败: "
                f"delivery_id={delivery_id}"
            )
        if existing.state == ReplyOutboxState.FAILED_TERMINAL:
            # FAILED_TERMINAL 为吸收态：仅当持有者用同一 lease 重试时才幂等返回；
            # lease mismatch（旧 holder 已被 cleanup 抢占失去 lease 之后又看到他人收口为 terminal）
            # 必须抛 LeaseExpiredError，避免把"写入未生效、ownership 已变化"伪装成失败回写成功。
            if existing.lease_id == normalized_lease_id:
                return existing
            raise LeaseExpiredError(
                "reply delivery lease 已失效: "
                f"delivery_id={delivery_id}",
                record_id=existing.delivery_id,
                lease_id=normalized_lease_id,
            )
        if existing.lease_id != normalized_lease_id:
            raise LeaseExpiredError(
                "reply delivery lease 已失效: "
                f"delivery_id={delivery_id}",
                record_id=existing.delivery_id,
                lease_id=normalized_lease_id,
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
            # FAILED_RETRYABLE 释放 lease（置 NULL）：本次 attempt 已失败，ownership 必须
            # 在下一次 claim 重新分配前释放，旧 holder 不得再用旧 lease 改写记录；
            # FAILED_TERMINAL 保留 lease 作为吸收态幂等校验依据（同 holder 自重试可幂等返回）。
            lease_id=(None if retryable else existing.lease_id),
        )
        self._records[updated.delivery_id] = updated
        return updated

    def cleanup_stale_in_progress_deliveries(
        self,
        *,
        max_age: timedelta,
    ) -> list[str]:
        """把超过 max_age 的 DELIVERY_IN_PROGRESS 回退为 FAILED_RETRYABLE。

        回退时 ``lease_id`` 一并置 NULL，与 ``mark_failed(retryable=True)`` 路径
        保持一致："FAILED_RETRYABLE 不持 lease"是 record 级 fence token 的契约。
        旧持有者后续 ack/nack 用旧 lease 撞 ``state='delivery_in_progress' AND
        lease_id = ?`` 双条件 CAS 仍然必然 mismatch，``LeaseExpiredError`` 语义不变。

        Args:
            max_age: 超过多久未收到终态的 IN_PROGRESS 视为 stale；
                必须 ``>= MIN_STALE_AGE``，避免与 ``mark_delivered`` CAS
                竞态导致已成功交付被回退。

        Returns:
            被回退的 delivery_id 列表。

        Raises:
            ValueError: ``max_age < MIN_STALE_AGE`` 时抛出。
        """

        _validate_stale_max_age(max_age)
        cutoff = _now_utc() - max_age
        stale_ids: list[str] = []
        for delivery_id, record in list(self._records.items()):
            if record.state != ReplyOutboxState.DELIVERY_IN_PROGRESS:
                continue
            if record.updated_at > cutoff:
                continue
            self._records[delivery_id] = ReplyOutboxRecord(
                delivery_id=record.delivery_id,
                delivery_key=record.delivery_key,
                session_id=record.session_id,
                scene_name=record.scene_name,
                source_run_id=record.source_run_id,
                reply_content=record.reply_content,
                metadata=record.metadata,
                state=ReplyOutboxState.FAILED_RETRYABLE,
                created_at=record.created_at,
                updated_at=_now_utc(),
                delivery_attempt_count=record.delivery_attempt_count,
                last_error_message=STALE_IN_PROGRESS_ERROR_MESSAGE,
                lease_id=None,
            )
            stale_ids.append(delivery_id)
        if stale_ids:
            Log.warn(
                f"reply outbox 清理 stale in_progress: count={len(stale_ids)}, ids={','.join(stale_ids)}",
                module=MODULE,
            )
        return stale_ids

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

    def __init__(
        self,
        host_store: HostStore,
        *,
        session_activity: "SessionActivityQueryProtocol | None" = None,
    ) -> None:
        """初始化 SQLite 仓储。

        Args:
            host_store: 宿主层 SQLite 存储。
            session_activity: 可选的 session 活性查询；装配后 ``submit_reply``
                在 session 已 CLOSED 时抛 ``SessionClosedError``，防止
                ``cancel_session`` 窗口期内产生孤儿 outbox 记录。传 ``None``
                时退化为不做屏障的旧行为，仅用于独立 store 单元测试。

        Returns:
            无。

        Raises:
            无。
        """

        self._host_store = host_store
        self._session_activity: "SessionActivityQueryProtocol | None" = session_activity

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
        ensure_session_active(
            self._session_activity,
            session_id=normalized_request.session_id,
            operation="submit_reply",
            module=MODULE,
            target_name="reply outbox",
        )
        now = _now_utc()
        delivery_id = f"delivery_{uuid.uuid4().hex[:12]}"
        conn = self._host_store.get_connection()
        # INSERT OR IGNORE 用于在数据库层原子收敛相同 delivery_key 的并发首写。
        with write_transaction(conn):
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO reply_outbox (
                    delivery_id, delivery_key, session_id, scene_name, source_run_id,
                    reply_content, state, delivery_attempt_count, last_error_message,
                    created_at, updated_at, metadata_json, lease_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    None,
                ),
            )
            rowcount = cursor.rowcount
        if rowcount == 0:
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
        """把记录推进到发送中状态，并分配新的 ``lease_id``。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            更新后的交付记录，``lease_id`` 为本次 acquire 分配的新 fence token。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 当前状态不允许 claim 时抛出。
            RuntimeError: 更新后读取失败时抛出。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        new_lease_id = generate_lease_id()
        conn = self._host_store.get_connection()
        with write_transaction(conn):
            cursor = conn.execute(
                """
                UPDATE reply_outbox
                SET state = ?,
                    delivery_attempt_count = delivery_attempt_count + 1,
                    last_error_message = ?,
                    updated_at = ?,
                    lease_id = ?
                WHERE delivery_id = ?
                  AND state IN (?, ?)
                """,
                (
                    ReplyOutboxState.DELIVERY_IN_PROGRESS.value,
                    None,
                    _serialize_dt(_now_utc()),
                    new_lease_id,
                    normalized_delivery_id,
                    ReplyOutboxState.PENDING_DELIVERY.value,
                    ReplyOutboxState.FAILED_RETRYABLE.value,
                ),
            )
            rowcount = cursor.rowcount
        updated = self.get_reply(normalized_delivery_id)
        if rowcount == 0 and updated is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if updated is None:
            raise RuntimeError(f"reply delivery 更新后读取失败: {normalized_delivery_id}")
        if rowcount == 0:
            raise ValueError(
                "reply delivery 当前状态不允许 claim: "
                f"delivery_id={delivery_id}, state={updated.state.value}"
            )
        return updated

    def mark_delivered(self, delivery_id: str, *, lease_id: str) -> ReplyOutboxRecord:
        """标记记录已完成交付。

        采用 ``state + lease_id`` 双条件 CAS：当 lease_id mismatch 时（典型场景：
        cleanup 抢占已分配新 lease），rowcount=0 且当前状态非 DELIVERED，抛
        ``LeaseExpiredError``，让旧持有者明确感知交付权已被抢占。

        Args:
            delivery_id: 交付记录 ID。
            lease_id: claim 时返回的 fence token；必须与当前持有 lease 完全匹配。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 当前状态不允许 delivered 时抛出。
            LeaseExpiredError: 持有的 lease 已被抢占。
            RuntimeError: 更新后读取失败时抛出。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        normalized_lease_id = _normalize_text(lease_id, field_name="lease_id")
        existing = self.get_reply(normalized_delivery_id)
        if existing is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if existing.state == ReplyOutboxState.DELIVERED:
            # DELIVERED 是吸收态：仅当持有者用同一 lease 重试时才幂等返回；
            # lease mismatch 必须抛 LeaseExpiredError，避免把旧 holder 在
            # cleanup 抢占之后又看到他人收口的场景伪装成本次写入成功。
            if existing.lease_id == normalized_lease_id:
                return existing
            raise LeaseExpiredError(
                "reply delivery lease 已失效: "
                f"delivery_id={delivery_id}",
                record_id=existing.delivery_id,
                lease_id=normalized_lease_id,
            )
        conn = self._host_store.get_connection()
        with write_transaction(conn):
            cursor = conn.execute(
                """
                UPDATE reply_outbox
                SET state = ?,
                    last_error_message = ?,
                    updated_at = ?
                WHERE delivery_id = ?
                  AND state = ?
                  AND lease_id = ?
                """,
                (
                    ReplyOutboxState.DELIVERED.value,
                    None,
                    _serialize_dt(_now_utc()),
                    existing.delivery_id,
                    ReplyOutboxState.DELIVERY_IN_PROGRESS.value,
                    normalized_lease_id,
                ),
            )
            rowcount = cursor.rowcount
        # 事务外的 pre-read（self.get_reply 在函数顶部）只用于"快速路径 + 错误信息"；
        # 双条件 CAS 在事务内执行后，必须再做一次 post-read 复核 rowcount=0 的真因——
        # 因为 pre-read 与 UPDATE 之间存在并发窗口，可能被 cleanup 抢占改写 lease 或被
        # 他人收口为 DELIVERED。复核读取本身轻量，且只在 CAS 失败这一条罕见路径上发生。
        updated = self.get_reply(existing.delivery_id)
        if updated is None:
            raise RuntimeError(f"reply delivery 更新后读取失败: {existing.delivery_id}")
        if rowcount == 0:
            if updated.state == ReplyOutboxState.DELIVERED:
                # 进入吸收态：再次按 lease 等值校验，旧 holder 撞上他人收口时也要 fail-loud。
                if updated.lease_id == normalized_lease_id:
                    return updated
                raise LeaseExpiredError(
                    "reply delivery lease 已失效: "
                    f"delivery_id={delivery_id}",
                    record_id=existing.delivery_id,
                    lease_id=normalized_lease_id,
                )
            if updated.state != ReplyOutboxState.DELIVERY_IN_PROGRESS:
                raise ValueError(
                    "reply delivery 当前状态不允许 delivered: "
                    f"delivery_id={delivery_id}, state={updated.state.value}"
                )
            # state 仍是 IN_PROGRESS 但 CAS 失败 → lease_id mismatch
            raise LeaseExpiredError(
                "reply delivery lease 已失效: "
                f"delivery_id={delivery_id}",
                record_id=existing.delivery_id,
                lease_id=normalized_lease_id,
            )
        return updated

    def mark_failed(
        self,
        delivery_id: str,
        *,
        retryable: bool,
        error_message: str,
        lease_id: str,
    ) -> ReplyOutboxRecord:
        """标记记录交付失败。

        采用 ``state + lease_id`` 双条件 CAS。lease_id mismatch 时抛
        ``LeaseExpiredError``；已 DELIVERED 的记录仍按现有 ``ValueError`` 报错。

        Args:
            delivery_id: 交付记录 ID。
            retryable: 是否允许后续再次 claim。
            error_message: 失败消息。
            lease_id: claim 时返回的 fence token；必须与当前持有 lease 完全匹配。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 已完成交付的记录重复标记失败时抛出。
            LeaseExpiredError: 持有的 lease 已被抢占。
            RuntimeError: 更新后读取失败时抛出。
        """

        normalized_delivery_id = _normalize_text(delivery_id, field_name="delivery_id")
        normalized_lease_id = _normalize_text(lease_id, field_name="lease_id")
        existing = self.get_reply(normalized_delivery_id)
        if existing is None:
            raise KeyError(f"reply delivery 不存在: {delivery_id}")
        if existing.state == ReplyOutboxState.DELIVERED:
            raise ValueError(
                "已完成交付的 reply delivery 不能再标记失败: "
                f"delivery_id={delivery_id}"
            )
        if existing.state == ReplyOutboxState.FAILED_TERMINAL:
            # FAILED_TERMINAL 是吸收态：仅当 lease 与当前持有者一致时幂等返回；
            # lease mismatch（旧 holder 在 cleanup 抢占之后又看到他人收口为 terminal）
            # 必须抛 LeaseExpiredError，避免把"写入未生效、ownership 已变化"伪装成成功。
            if existing.lease_id == normalized_lease_id:
                return existing
            raise LeaseExpiredError(
                "reply delivery lease 已失效: "
                f"delivery_id={delivery_id}",
                record_id=existing.delivery_id,
                lease_id=normalized_lease_id,
            )
        normalized_error_message = _normalize_text(error_message, field_name="error_message")
        conn = self._host_store.get_connection()
        with write_transaction(conn):
            cursor = conn.execute(
                """
                UPDATE reply_outbox
                SET state = ?,
                    last_error_message = ?,
                    updated_at = ?,
                    lease_id = CASE WHEN ? = 1 THEN NULL ELSE lease_id END
                WHERE delivery_id = ?
                  AND state != ?
                  AND lease_id = ?
                """,
                (
                    ReplyOutboxState.FAILED_RETRYABLE.value if retryable else ReplyOutboxState.FAILED_TERMINAL.value,
                    normalized_error_message,
                    _serialize_dt(_now_utc()),
                    1 if retryable else 0,
                    existing.delivery_id,
                    ReplyOutboxState.DELIVERED.value,
                    normalized_lease_id,
                ),
            )
            rowcount = cursor.rowcount
        if rowcount == 0:
            current = self.get_reply(existing.delivery_id)
            if current is not None and current.state == ReplyOutboxState.DELIVERED:
                raise ValueError(
                    "已完成交付的 reply delivery 不能再标记失败: "
                    f"delivery_id={delivery_id}"
                )
            if current is not None and current.state == ReplyOutboxState.FAILED_TERMINAL:
                # 并发情形下，本地预读还是非 terminal，但 SQL 锁排队后已被他人收口为 terminal；
                # 此时必须按"lease 等值"复核：与持有者一致幂等返回，否则抛 LeaseExpiredError。
                if current.lease_id == normalized_lease_id:
                    return current
                raise LeaseExpiredError(
                    "reply delivery lease 已失效: "
                    f"delivery_id={delivery_id}",
                    record_id=existing.delivery_id,
                    lease_id=normalized_lease_id,
                )
            # 排除 DELIVERED / FAILED_TERMINAL 后仍 mismatch → lease_id 失效
            raise LeaseExpiredError(
                "reply delivery lease 已失效: "
                f"delivery_id={delivery_id}",
                record_id=existing.delivery_id,
                lease_id=normalized_lease_id,
            )
        updated = self.get_reply(existing.delivery_id)
        if updated is None:
            raise RuntimeError(f"reply delivery 更新后读取失败: {existing.delivery_id}")
        return updated

    def cleanup_stale_in_progress_deliveries(
        self,
        *,
        max_age: timedelta,
    ) -> list[str]:
        """把超过 max_age 的 DELIVERY_IN_PROGRESS 回退为 FAILED_RETRYABLE。

        回退时 ``lease_id`` 一并置 NULL，与 ``mark_failed(retryable=True)`` 路径
        保持一致："FAILED_RETRYABLE 不持 lease"是 record 级 fence token 的契约。
        旧持有者后续 ack/nack 用旧 lease 撞 ``state='delivery_in_progress' AND
        lease_id = ?`` 双条件 CAS 仍然必然 mismatch，``LeaseExpiredError`` 语义不变。

        Args:
            max_age: 超过多久未收到终态的 IN_PROGRESS 视为 stale；
                必须 ``>= MIN_STALE_AGE``，避免与 ``mark_delivered`` CAS
                竞态导致已成功交付被回退。

        Returns:
            被回退的 delivery_id 列表。

        Raises:
            ValueError: ``max_age < MIN_STALE_AGE`` 时抛出。
        """

        _validate_stale_max_age(max_age)
        cutoff = _serialize_dt(_now_utc() - max_age)
        now_ts = _serialize_dt(_now_utc())
        conn = self._host_store.get_connection()
        rows = conn.execute(
            """
            SELECT delivery_id FROM reply_outbox
            WHERE state = ? AND updated_at <= ?
            """,
            (ReplyOutboxState.DELIVERY_IN_PROGRESS.value, cutoff),
        ).fetchall()
        candidate_ids = [str(row["delivery_id"]) for row in rows]
        if not candidate_ids:
            return []
        # 仅返回事务内 rowcount=1 的 delivery_id：候选 id 在事务外被读出，与 UPDATE
        # 之间存在窗口，期间记录可能被并发推进到 DELIVERED / FAILED_*；那种 id 不应
        # 计入"本次实际回退"列表，以免日志与回收统计出现假阳性。
        rotated_ids: list[str] = []
        with write_transaction(conn):
            for stale_id in candidate_ids:
                cursor = conn.execute(
                    """
                    UPDATE reply_outbox
                    SET state = ?,
                        last_error_message = ?,
                        updated_at = ?,
                        lease_id = NULL
                    WHERE delivery_id = ?
                      AND state = ?
                    """,
                    (
                        ReplyOutboxState.FAILED_RETRYABLE.value,
                        STALE_IN_PROGRESS_ERROR_MESSAGE,
                        now_ts,
                        stale_id,
                        ReplyOutboxState.DELIVERY_IN_PROGRESS.value,
                    ),
                )
                if cursor.rowcount == 1:
                    rotated_ids.append(stale_id)
        if not rotated_ids:
            return []
        Log.warn(
            f"reply outbox 清理 stale in_progress: count={len(rotated_ids)}, ids={','.join(rotated_ids)}",
            module=MODULE,
        )
        return rotated_ids

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
        with write_transaction(conn):
            cursor = conn.execute(
                "DELETE FROM reply_outbox WHERE session_id = ?",
                (normalized,),
            )
            rowcount = cursor.rowcount
        return rowcount


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
    raw_lease_id = row["lease_id"]
    lease_id = str(raw_lease_id) if raw_lease_id is not None else None
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
        lease_id=lease_id,
    )


__all__ = [
    "InMemoryReplyOutboxStore",
    "SQLiteReplyOutboxStore",
]
