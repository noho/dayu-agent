"""reply outbox REST 端点。"""

from __future__ import annotations

from typing import Any

from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.contracts.reply_outbox import ReplyOutboxState
from dayu.host.lease import LeaseExpiredError
from dayu.services.contracts import ReplyDeliveryFailureRequest, ReplyDeliveryView
from dayu.services.protocols import ReplyDeliveryServiceProtocol


def _parse_reply_delivery_state(state: str | None) -> str | None:
    """把查询参数里的交付状态标准化为小写字符串。

    Args:
        state: 原始状态字符串。

    Returns:
        规范化后的状态字符串；未传时返回 ``None``。

    Raises:
        ValueError: 状态值不合法时抛出。
    """

    if state is None:
        return None
    return ReplyOutboxState(str(state).strip().lower()).value


def _build_reply_delivery_payload(record: ReplyDeliveryView) -> dict[str, Any]:
    """把交付视图转为可序列化响应载荷。"""

    return {
        "delivery_id": record.delivery_id,
        "delivery_key": record.delivery_key,
        "session_id": record.session_id,
        "scene_name": record.scene_name,
        "source_run_id": record.source_run_id,
        "reply_content": record.reply_content,
        "metadata": dict(record.metadata),
        "state": record.state.value,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "delivery_attempt_count": record.delivery_attempt_count,
        "last_error_message": record.last_error_message,
        "lease_id": record.lease_id,
    }


def create_reply_outbox_router(reply_delivery_service: ReplyDeliveryServiceProtocol):
    """创建 reply outbox 路由。

    Args:
        reply_delivery_service: reply delivery 服务。

    Returns:
        reply outbox 路由对象。

    Raises:
        无。
    """

    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    router = APIRouter(prefix="/api/reply-outbox", tags=["reply-outbox"])

    class ReplyDeliveryResponse(BaseModel):
        """reply delivery 响应。

        ``lease_id`` 在 ``DELIVERY_IN_PROGRESS`` 与 ``DELIVERED`` /
        ``FAILED_TERMINAL`` 三态下都会原样透传到响应体；渠道侧必须把它原样
        回传给后续 ack/nack 才能通过 fence token 双条件 CAS。
        ``PENDING_DELIVERY`` / ``FAILED_RETRYABLE`` 下为 ``None``，等待下次
        ``claim`` 重新分配 ownership。
        """

        delivery_id: str
        delivery_key: str
        session_id: str
        scene_name: str
        source_run_id: str
        reply_content: str
        metadata: ExecutionDeliveryContext
        state: str
        created_at: str
        updated_at: str
        delivery_attempt_count: int
        last_error_message: str | None = None
        lease_id: str | None = None

    class DeliveryAckRequest(BaseModel):
        """delivery ack 请求。

        path 中已携带 ``delivery_id``，body 仅承载 ack 必需的 fence token，
        避免 path/body 双真源。
        """

        lease_id: str

    class DeliveryNackRequest(BaseModel):
        """delivery nack 请求。"""

        retryable: bool = True
        error_message: str
        lease_id: str

    def _to_response(record: ReplyDeliveryView) -> ReplyDeliveryResponse:
        """把交付视图转换为响应。"""

        return ReplyDeliveryResponse(**_build_reply_delivery_payload(record))

    @router.get("", response_model=list[ReplyDeliveryResponse])
    async def list_reply_outbox(
        session_id: str | None = None,
        scene_name: str | None = None,
        state: str | None = None,
    ) -> list[ReplyDeliveryResponse]:
        """列出 reply outbox 记录。"""

        try:
            parsed_state = _parse_reply_delivery_state(state)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid reply delivery state: {state}") from exc
        records = reply_delivery_service.list_deliveries(
            session_id=session_id,
            scene_name=scene_name,
            state=parsed_state,
        )
        return [_to_response(record) for record in records]

    @router.get("/{delivery_id}", response_model=ReplyDeliveryResponse)
    async def get_reply_outbox(delivery_id: str) -> ReplyDeliveryResponse:
        """获取单个 reply outbox 记录。"""

        record = reply_delivery_service.get_delivery(delivery_id)
        if record is None:
            raise HTTPException(status_code=404, detail="reply delivery not found")
        return _to_response(record)

    @router.post("/{delivery_id}/claim", response_model=ReplyDeliveryResponse)
    async def claim_reply_outbox(delivery_id: str) -> ReplyDeliveryResponse:
        """独占领取一条待投递 reply outbox 记录。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            已切换到发送中状态的交付记录。

        Raises:
            HTTPException: 记录不存在或当前状态不可 claim 时抛出。
        """

        try:
            record = reply_delivery_service.claim_delivery(delivery_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="reply delivery not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _to_response(record)

    @router.post("/{delivery_id}/ack", response_model=ReplyDeliveryResponse)
    async def ack_reply_outbox(delivery_id: str, body: DeliveryAckRequest) -> ReplyDeliveryResponse:
        """标记 reply outbox 已送达。

        Args:
            delivery_id: path 参数；交付记录 ID。
            body: 必填 ``lease_id``，必须与 claim 时返回的 fence token 完全匹配。

        Returns:
            已 DELIVERED 的交付记录。

        Raises:
            HTTPException: 记录不存在 (404)、状态不允许 (409) 或 lease 已被抢占 (409) 时抛出。
        """

        try:
            record = reply_delivery_service.mark_delivery_delivered(
                delivery_id, lease_id=body.lease_id
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="reply delivery not found") from exc
        except LeaseExpiredError as exc:
            raise HTTPException(status_code=409, detail=f"reply delivery lease expired: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _to_response(record)

    @router.post("/{delivery_id}/nack", response_model=ReplyDeliveryResponse)
    async def nack_reply_outbox(delivery_id: str, body: DeliveryNackRequest) -> ReplyDeliveryResponse:
        """标记 reply outbox 发送失败。

        Args:
            delivery_id: path 参数；交付记录 ID。
            body: 必填 ``lease_id`` / ``retryable`` / ``error_message``；
                ``lease_id`` 必须与 claim 时返回的 fence token 完全匹配。

        Returns:
            已转入 FAILED_RETRYABLE / FAILED_TERMINAL 的交付记录。

        Raises:
            HTTPException: 记录不存在 (404)、参数非法 (400) 或 lease 已被抢占 (409) 时抛出。
        """

        try:
            record = reply_delivery_service.mark_delivery_failed(
                ReplyDeliveryFailureRequest(
                    delivery_id=delivery_id,
                    retryable=body.retryable,
                    error_message=body.error_message,
                    lease_id=body.lease_id,
                )
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="reply delivery not found") from exc
        except LeaseExpiredError as exc:
            raise HTTPException(status_code=409, detail=f"reply delivery lease expired: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_response(record)

    return router


__all__ = ["create_reply_outbox_router"]
