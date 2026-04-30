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
        lease_id: 当前持有者的 fence token，由 ``claim_reply`` 在成功 acquire 时
            分配。``DELIVERY_IN_PROGRESS`` 与 ``DELIVERED`` / ``FAILED_TERMINAL``
            （吸收态）下保留 lease：mark_delivered / mark_failed 必须携带本字段做
            双条件 CAS，吸收态的幂等返回也用 lease 等值校验，过滤"旧持有者写入
            已被接管的记录"。``PENDING_DELIVERY`` / ``FAILED_RETRYABLE`` 下为
            ``None``——这两个状态意味着"等待下一次 claim 重新分配 ownership"，
            ``mark_failed(retryable=True)`` 与 ``cleanup_stale_in_progress_deliveries``
            在落库时把 lease 显式置 ``None``，旧持有者持有的旧 lease 因此立即失效。

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


class ReplyOutboxDeliveryKeyConflictError(ValueError):
    """同 ``delivery_key`` 二次 submit 携带的 payload 与已落库记录不一致。

    该异常是 reply outbox 的稳定契约面：上游通道（Web / WeChat）允许从
    Service / Host 链路接收并 fail-closed 处理；放在 ``dayu.contracts.reply_outbox``
    可以让 UI 层只依赖契约模块，不被 Host 内部仓储实现的模块结构变更绑死。

    继承自 ``ValueError`` 以保持向上层 service / web / wechat 的兼容契约
    （Host 文档与 Service docstring 长期声明 submit 失败抛 ``ValueError``），
    同时通过精确类型让调用方能够 fail-closed 显式 catch、写结构化告警，
    避免与"参数非法 / 字段为空"这类普通 ValueError 混在一起静默吞掉。

    携带 ``existing_payload`` 与 ``attempted_payload`` 字段，便于调用方在
    日志中输出 payload diff，定位"同一 run 产生分叉回复"的业务事故源头。
    """

    def __init__(
        self,
        *,
        delivery_key: str,
        existing_payload: dict[str, object],
        attempted_payload: dict[str, object],
    ) -> None:
        """构造冲突异常。

        Args:
            delivery_key: 触发冲突的幂等键。
            existing_payload: 已落库记录的 payload 快照。
            attempted_payload: 本次 submit 提交的 payload 快照。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(
            "delivery_key 已存在且负载不一致: "
            f"delivery_key={delivery_key}"
        )
        self.delivery_key = delivery_key
        self.existing_payload = existing_payload
        self.attempted_payload = attempted_payload


__all__ = [
    "ReplyOutboxDeliveryKeyConflictError",
    "ReplyOutboxRecord",
    "ReplyOutboxState",
    "ReplyOutboxSubmitRequest",
]
