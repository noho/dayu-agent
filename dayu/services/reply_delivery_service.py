"""reply delivery 服务实现。

该服务把 Host 暴露的 reply outbox 能力收口为 Service-owned DTO，
供 Web / WeChat 等渠道层使用，避免 UI 请求期直接触碰 Host。
"""

from __future__ import annotations

from dataclasses import dataclass

from dayu.contracts.execution_metadata import normalize_execution_delivery_context
from dayu.contracts.reply_outbox import ReplyOutboxRecord, ReplyOutboxState, ReplyOutboxSubmitRequest
from dayu.host.protocols import ReplyDeliveryGatewayProtocol
from dayu.services.contracts import (
    ReplyDeliveryFailureRequest,
    ReplyDeliverySubmitRequest,
    ReplyDeliveryView,
)
from dayu.services.protocols import ReplyDeliveryServiceProtocol


def _parse_reply_outbox_state(state: str | None) -> ReplyOutboxState | None:
    """解析交付状态字符串。

    Args:
        state: 原始状态字符串。

    Returns:
        解析后的交付状态；未传时返回 ``None``。

    Raises:
        ValueError: 状态值非法时抛出。
    """

    if state is None:
        return None
    return ReplyOutboxState(str(state).strip().lower())


def _to_reply_delivery_view(record: ReplyOutboxRecord) -> ReplyDeliveryView:
    """把 reply outbox 记录转换为 Service 视图。

    Args:
        record: Host reply outbox 记录。

    Returns:
        Service-owned 交付视图。

    Raises:
        无。
    """

    return ReplyDeliveryView(
        delivery_id=record.delivery_id,
        delivery_key=record.delivery_key,
        session_id=record.session_id,
        scene_name=record.scene_name,
        source_run_id=record.source_run_id,
        reply_content=record.reply_content,
        metadata=normalize_execution_delivery_context(record.metadata),
        state=record.state,
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
        delivery_attempt_count=record.delivery_attempt_count,
        last_error_message=record.last_error_message,
        lease_id=record.lease_id,
    )


@dataclass
class ReplyDeliveryService(ReplyDeliveryServiceProtocol):
    """reply delivery 服务。"""

    host: ReplyDeliveryGatewayProtocol

    def submit_reply_for_delivery(self, request: ReplyDeliverySubmitRequest) -> ReplyDeliveryView:
        """显式提交待交付回复。

        Args:
            request: 提交请求。

        Returns:
            交付视图。

        Raises:
            ValueError: 提交参数非法或幂等键负载冲突时抛出。
        """

        record = self.host.submit_reply_for_delivery(
            ReplyOutboxSubmitRequest(
                delivery_key=request.delivery_key,
                session_id=request.session_id,
                scene_name=request.scene_name,
                source_run_id=request.source_run_id,
                reply_content=request.reply_content,
                metadata=normalize_execution_delivery_context(request.metadata),
            )
        )
        return _to_reply_delivery_view(record)

    def get_delivery(self, delivery_id: str) -> ReplyDeliveryView | None:
        """按 ID 查询交付记录。"""

        record = self.host.get_reply_outbox(delivery_id)
        if record is None:
            return None
        return _to_reply_delivery_view(record)

    def list_deliveries(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: str | None = None,
    ) -> list[ReplyDeliveryView]:
        """列出交付记录。"""

        parsed_state = _parse_reply_outbox_state(state)
        records = self.host.list_reply_outbox(
            session_id=session_id,
            scene_name=scene_name,
            state=parsed_state,
        )
        return [_to_reply_delivery_view(record) for record in records]

    def claim_delivery(self, delivery_id: str) -> ReplyDeliveryView:
        """把交付记录推进到发送中状态。"""

        return _to_reply_delivery_view(self.host.claim_reply_delivery(delivery_id))

    def mark_delivery_delivered(self, delivery_id: str, *, lease_id: str) -> ReplyDeliveryView:
        """标记交付完成。

        Args:
            delivery_id: 交付记录 ID。
            lease_id: ``claim_delivery`` 返回视图中的 fence token。

        Returns:
            更新后的交付视图。

        Raises:
            KeyError: 记录不存在时抛出。
            LeaseExpiredError: 持有的 lease 已被 cleanup 抢占。
        """

        return _to_reply_delivery_view(
            self.host.mark_reply_delivered(delivery_id, lease_id=lease_id)
        )

    def mark_delivery_failed(self, request: ReplyDeliveryFailureRequest) -> ReplyDeliveryView:
        """标记交付失败。

        Args:
            request: 失败回写请求；其 ``lease_id`` 必须与 claim 返回 lease 一致。

        Returns:
            更新后的交付视图。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 已完成交付时抛出。
            LeaseExpiredError: 持有的 lease 已被 cleanup 抢占。
        """

        return _to_reply_delivery_view(
            self.host.mark_reply_delivery_failed(
                request.delivery_id,
                retryable=request.retryable,
                error_message=request.error_message,
                lease_id=request.lease_id,
            )
        )


__all__ = ["ReplyDeliveryService"]