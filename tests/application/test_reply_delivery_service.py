"""ReplyDeliveryService 测试。"""

from __future__ import annotations

import pytest

from dayu.contracts.reply_outbox import ReplyOutboxState
from dayu.host.host import Host
from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore
from dayu.services.contracts import ReplyDeliveryFailureRequest, ReplyDeliverySubmitRequest
from dayu.services.protocols import ReplyDeliveryServiceProtocol
from dayu.services.reply_delivery_service import ReplyDeliveryService
from tests.application.conftest import StubHostExecutor, StubRunRegistry, StubSessionRegistry


def _build_service() -> ReplyDeliveryService:
    """构造测试用 ReplyDeliveryService。"""

    host = Host(
        executor=StubHostExecutor(),
        session_registry=StubSessionRegistry(),
        run_registry=StubRunRegistry(),
        reply_outbox_store=InMemoryReplyOutboxStore(),
    )
    return ReplyDeliveryService(host=host)


@pytest.mark.unit
def test_reply_delivery_service_submit_and_get() -> None:
    """ReplyDeliveryService 应返回 Service-owned 交付视图。"""

    service = _build_service()

    created = service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="web:run_1",
            session_id="session_1",
            scene_name="web_chat",
            source_run_id="run_1",
            reply_content="最终结论",
            metadata={"delivery_channel": "web", "delivery_target": "session_1"},
        )
    )
    fetched = service.get_delivery(created.delivery_id)

    assert created.state == ReplyOutboxState.PENDING_DELIVERY
    assert fetched is not None
    assert fetched.delivery_id == created.delivery_id
    assert fetched.reply_content == "最终结论"


@pytest.mark.unit
def test_reply_delivery_service_claim_fail_and_deliver() -> None:
    """ReplyDeliveryService 应暴露完整状态流转。"""

    service = _build_service()
    created = service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="wechat:run_1",
            session_id="session_1",
            scene_name="wechat",
            source_run_id="run_1",
            reply_content="结论",
            metadata={"delivery_channel": "wechat", "delivery_target": "user_1"},
        )
    )

    claimed = service.claim_delivery(created.delivery_id)
    assert claimed.lease_id is not None
    failed = service.mark_delivery_failed(
        ReplyDeliveryFailureRequest(
            delivery_id=claimed.delivery_id,
            retryable=True,
            error_message="网络抖动",
            lease_id=claimed.lease_id,
        )
    )
    reclaimed = service.claim_delivery(failed.delivery_id)
    assert reclaimed.lease_id is not None
    delivered = service.mark_delivery_delivered(reclaimed.delivery_id, lease_id=reclaimed.lease_id)

    assert claimed.state == ReplyOutboxState.DELIVERY_IN_PROGRESS
    assert failed.state == ReplyOutboxState.FAILED_RETRYABLE
    assert delivered.state == ReplyOutboxState.DELIVERED
    assert delivered.delivery_attempt_count == 2


@pytest.mark.unit
def test_reply_delivery_service_implements_runtime_protocol() -> None:
    """ReplyDeliveryService 应满足 runtime protocol。"""

    assert isinstance(_build_service(), ReplyDeliveryServiceProtocol)