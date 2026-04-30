"""reply outbox 公共契约。

该模块定义 Host 可选托管的 reply outbox 真源数据对象。
reply outbox 与 pending conversation turn 严格分离：

- pending conversation turn 只表示 Host 内尚未完成、可 resume 的执行真源
- reply outbox 表示 Host 已被显式提交的出站交付真源

reply outbox 记录由上层显式提交，Host internal success 不会自动创建记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dayu.contracts.execution_metadata import ExecutionDeliveryContext, empty_execution_delivery_context


class ReplyOutboxState(str, Enum):
    """reply outbox 记录状态。"""

    PENDING_DELIVERY = "pending_delivery"
    DELIVERY_IN_PROGRESS = "delivery_in_progress"
    DELIVERED = "delivered"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"


@dataclass(frozen=True)
class ReplyOutboxSubmitRequest:
    """创建 reply outbox 记录所需的最小提交请求。

    Args:
        delivery_key: 业务侧显式提供的幂等键。
        session_id: 关联 Host session ID。
        scene_name: 关联 scene 名。
        source_run_id: 关联 source run ID。
        reply_content: 待交付的最终回复内容。
        metadata: 交付上下文。

    Returns:
        无。

    Raises:
        无。
    """

    delivery_key: str
    session_id: str
    scene_name: str
    source_run_id: str
    reply_content: str
    metadata: ExecutionDeliveryContext = field(default_factory=empty_execution_delivery_context)


@dataclass(frozen=True)
class ReplyOutboxRecord:
    """reply outbox 真源记录。

    Args:
        delivery_id: Host 分配的交付记录 ID。
        delivery_key: 业务侧提供的稳定幂等键。
        session_id: 关联 Host session ID。
        scene_name: 关联 scene 名。
        source_run_id: 关联 source run ID。
        reply_content: 待交付的最终回复内容。
        metadata: 交付上下文。
        state: 当前交付状态。
        created_at: 创建时间。
        updated_at: 最近更新时间。
        delivery_attempt_count: 已进入发送中的次数。
        last_error_message: 最近一次失败消息。
        lease_id: 当前持有者的 fence token。仅在 ``DELIVERY_IN_PROGRESS`` 状态下有
            意义，由 ``claim_reply`` 在成功 acquire 时分配；mark_delivered/mark_failed
            必须携带本字段做双条件 CAS，cleanup 抢占时分配新 lease_id 让旧持有者
            迟到回写必失败；其他状态下为 ``None``。

    Returns:
        无。

    Raises:
        无。
    """

    delivery_id: str
    delivery_key: str
    session_id: str
    scene_name: str
    source_run_id: str
    reply_content: str
    metadata: ExecutionDeliveryContext
    state: ReplyOutboxState
    created_at: datetime
    updated_at: datetime
    delivery_attempt_count: int = 0
    last_error_message: str | None = None
    lease_id: str | None = None


__all__ = [
    "ReplyOutboxRecord",
    "ReplyOutboxState",
    "ReplyOutboxSubmitRequest",
]