"""Web reply outbox 集成测试。"""

from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from typing import Any, Callable, Coroutine, cast

import pytest

from dayu.contracts.events import AppEvent, AppEventType
from dayu.contracts.reply_outbox import ReplyOutboxState
from dayu.host.host import Host
from dayu.host.host_execution import HostExecutorProtocol
from dayu.host.protocols import RunRegistryProtocol, SessionRegistryProtocol
from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore
from dayu.services.contracts import ReplyDeliverySubmitRequest, ReplyDeliveryView
from dayu.services.reply_delivery_service import ReplyDeliveryService
from dayu.web.routes.chat import _consume_stream
from dayu.web.routes.reply_outbox import (
    _build_reply_delivery_payload,
    _parse_reply_delivery_state,
    create_reply_outbox_router,
)
from tests.application.conftest import StubHostExecutor, StubRunRegistry, StubSessionRegistry


class _FakeHTTPException(Exception):
    """最小 HTTPException 测试桩。"""

    def __init__(self, *, status_code: int, detail: str) -> None:
        """记录异常状态码与错误详情。

        Args:
            status_code: HTTP 状态码。
            detail: 错误详情。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _FakeBaseModel:
    """最小 BaseModel 测试桩。"""

    def __init__(self, **kwargs: object) -> None:
        """把传入字段挂到实例属性上。

        Args:
            **kwargs: 模型字段。

        Returns:
            无。

        Raises:
            无。
        """

        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeRouter:
    """最小 APIRouter 测试桩。"""

    def __init__(self, *, prefix: str, tags: list[str]) -> None:
        """记录路由前缀、标签与注册处理器。

        Args:
            prefix: 路由前缀。
            tags: 标签列表。

        Returns:
            无。

        Raises:
            无。
        """

        self.prefix = prefix
        self.tags = tags
        self.handlers: dict[tuple[str, str], object] = {}

    def get(self, path: str, **_kwargs: object):
        """记录 GET 处理器。

        Args:
            path: 路由路径。
            **_kwargs: 其余参数。

        Returns:
            装饰器。

        Raises:
            无。
        """

        def _decorator(func: object) -> object:
            self.handlers[("GET", path)] = func
            return func

        return _decorator

    def post(self, path: str, **_kwargs: object):
        """记录 POST 处理器。

        Args:
            path: 路由路径。
            **_kwargs: 其余参数。

        Returns:
            装饰器。

        Raises:
            无。
        """

        def _decorator(func: object) -> object:
            self.handlers[("POST", path)] = func
            return func

        return _decorator


def _install_fake_web_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """为 reply outbox router 安装最小 FastAPI/Pydantic 测试桩。

    Args:
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        无。
    """

    fake_fastapi = ModuleType("fastapi")
    cast(Any, fake_fastapi).APIRouter = _FakeRouter
    cast(Any, fake_fastapi).HTTPException = _FakeHTTPException
    fake_pydantic = ModuleType("pydantic")
    cast(Any, fake_pydantic).BaseModel = _FakeBaseModel
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
    monkeypatch.setitem(sys.modules, "pydantic", fake_pydantic)


def _build_reply_outbox_router_for_test(
    monkeypatch: pytest.MonkeyPatch,
    service: ReplyDeliveryService,
) -> _FakeRouter:
    """构造使用测试桩依赖的 reply outbox router。

    Args:
        monkeypatch: pytest monkeypatch。
        service: reply delivery 服务。

    Returns:
        绑定好处理器的假 router。

    Raises:
        无。
    """

    _install_fake_web_modules(monkeypatch)
    return cast(_FakeRouter, create_reply_outbox_router(service))


def _delivery_channel(metadata: object) -> object:
    """读取 reply metadata 中的可选 delivery_channel。"""

    return cast(Any, metadata).get("delivery_channel")


def _filtered_flag(metadata: object) -> object:
    """读取 reply metadata 中的可选 filtered。"""

    return cast(Any, metadata).get("filtered")


def _route_endpoint(route: object) -> object:
    """读取 FastAPI route 的 endpoint。"""

    return cast(Any, route).endpoint


def _build_reply_delivery_service() -> ReplyDeliveryService:
    """构造测试用 ReplyDeliveryService。"""

    host = Host(
        executor=cast(HostExecutorProtocol, StubHostExecutor()),
        session_registry=cast(SessionRegistryProtocol, StubSessionRegistry()),
        run_registry=cast(RunRegistryProtocol, StubRunRegistry()),
        reply_outbox_store=InMemoryReplyOutboxStore(),
    )
    return ReplyDeliveryService(host=host)


@pytest.mark.unit
def test_chat_stream_consumer_submits_web_reply_outbox() -> None:
    """Web chat 后台消费完成后应显式写入 reply outbox。"""

    service = _build_reply_delivery_service()

    async def _stream():
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="前半段", meta={"run_id": "run_web_1"})
        yield AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload={"content": "最终答案", "degraded": False},
            meta={"run_id": "run_web_1"},
        )

    asyncio.run(
        _consume_stream(
            _stream(),
            reply_delivery_service=service,
            session_id="session_web_1",
            scene_name="web_chat",
        )
    )

    records = service.list_deliveries(session_id="session_web_1")

    assert len(records) == 1
    assert records[0].delivery_key == "web:run_web_1"
    assert records[0].reply_content == "最终答案"
    assert _delivery_channel(records[0].metadata) == "web"


@pytest.mark.unit
def test_chat_stream_consumer_warns_when_reply_ready_but_run_id_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """有 reply 但事件 meta 缺 run_id 时应发出 warning，且不写 reply outbox。"""

    service = _build_reply_delivery_service()

    async def _stream():
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="前半段", meta={})
        yield AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload={"content": "最终答案", "degraded": False},
            meta={},
        )

    warnings: list[str] = []
    from dayu.log import Log as _Log

    monkeypatch.setattr(_Log, "warning", lambda message, *, module=None: warnings.append(str(message)))

    asyncio.run(
        _consume_stream(
            _stream(),
            reply_delivery_service=service,
            session_id="session_web_missing_run",
            scene_name="web_chat",
        )
    )

    assert service.list_deliveries(session_id="session_web_missing_run") == []
    assert any("缺少 run_id" in entry for entry in warnings)


@pytest.mark.unit
def test_chat_stream_consumer_preserves_filtered_metadata() -> None:
    """Web chat 后台消费应把 filtered 状态写入 reply metadata。"""

    service = _build_reply_delivery_service()

    async def _stream():
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="前半段", meta={"run_id": "run_web_filtered"})
        yield AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload={"content": "部分答案", "degraded": True, "filtered": True},
            meta={"run_id": "run_web_filtered"},
        )

    asyncio.run(
        _consume_stream(
            _stream(),
            reply_delivery_service=service,
            session_id="session_web_filtered",
            scene_name="web_chat",
        )
    )

    records = service.list_deliveries(session_id="session_web_filtered")
    assert len(records) == 1
    assert records[0].reply_content == "部分答案"
    assert _filtered_flag(records[0].metadata) is True


@pytest.mark.unit
def test_chat_stream_consumer_skips_partial_reply_when_cancelled() -> None:
    """Web chat 后台消费者在取消时不应提交 partial reply。"""

    service = _build_reply_delivery_service()

    async def _stream():
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="前半段", meta={"run_id": "run_web_cancelled"})
        yield AppEvent(
            type=AppEventType.CANCELLED,
            payload={"cancel_reason": "user_cancelled"},
            meta={"run_id": "run_web_cancelled"},
        )

    asyncio.run(
        _consume_stream(
            _stream(),
            reply_delivery_service=service,
            session_id="session_web_cancelled",
            scene_name="web_chat",
        )
    )

    assert service.list_deliveries(session_id="session_web_cancelled") == []


@pytest.mark.unit
def test_reply_outbox_route_helpers_parse_state_and_build_payload() -> None:
    """reply outbox route helper 应正确解析状态与构造响应载荷。"""

    service = _build_reply_delivery_service()
    created = service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="web:run_helper",
            session_id="session_helper",
            scene_name="web_chat",
            source_run_id="run_helper",
            reply_content="payload",
            metadata={"delivery_channel": "web", "delivery_target": "session_helper"},
        ),
    )

    payload = _build_reply_delivery_payload(created)

    assert _parse_reply_delivery_state("pending_delivery") == "pending_delivery"
    assert payload["delivery_id"] == created.delivery_id
    assert payload["state"] == "pending_delivery"


@pytest.mark.unit
def test_reply_outbox_list_route_accepts_boolean_metadata_with_real_model() -> None:
    """reply outbox 路由响应模型应允许 metadata 中携带布尔字段。"""

    pytest.importorskip("fastapi")
    pytest.importorskip("pydantic")

    service = _build_reply_delivery_service()
    service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="web:run_bool_meta",
            session_id="session_bool_meta",
            scene_name="web_chat",
            source_run_id="run_bool_meta",
            reply_content="payload",
            metadata={"delivery_channel": "web", "delivery_target": "session_bool_meta", "filtered": True},
        ),
    )

    router = create_reply_outbox_router(service)
    list_handler = cast(
        Callable[[], Coroutine[Any, Any, list[object]]],
        next(
            _route_endpoint(route)
            for route in cast(Any, router).routes
            if getattr(_route_endpoint(route), "__name__", "") == "list_reply_outbox"
        ),
    )

    records = cast(list[ReplyDeliveryView], asyncio.run(list_handler()))

    assert len(records) == 1
    assert _filtered_flag(records[0].metadata) is True


@pytest.mark.unit
def test_reply_outbox_claim_route_claims_delivery(monkeypatch: pytest.MonkeyPatch) -> None:
    """reply outbox route 应暴露独占 claim 端点。"""

    service = _build_reply_delivery_service()
    created = service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="web:run_claim_1",
            session_id="session_claim_1",
            scene_name="web_chat",
            source_run_id="run_claim_1",
            reply_content="payload",
            metadata={"delivery_channel": "web", "delivery_target": "session_claim_1"},
        ),
    )
    router = _build_reply_outbox_router_for_test(monkeypatch, service)
    claim_handler = cast(
        Callable[[str], Coroutine[Any, Any, object]],
        router.handlers[("POST", "/{delivery_id}/claim")],
    )

    claimed = cast(ReplyDeliveryView, asyncio.run(claim_handler(created.delivery_id)))

    assert claimed.delivery_id == created.delivery_id
    assert claimed.state == ReplyOutboxState.DELIVERY_IN_PROGRESS.value
    assert claimed.delivery_attempt_count == 1


@pytest.mark.unit
def test_reply_outbox_claim_route_returns_conflict_for_non_claimable_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重复 claim 同一条记录时，route 应返回 409。"""

    service = _build_reply_delivery_service()
    created = service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="web:run_claim_2",
            session_id="session_claim_2",
            scene_name="web_chat",
            source_run_id="run_claim_2",
            reply_content="payload",
            metadata={"delivery_channel": "web", "delivery_target": "session_claim_2"},
        ),
    )
    router = _build_reply_outbox_router_for_test(monkeypatch, service)
    claim_handler = cast(
        Callable[[str], Coroutine[Any, Any, object]],
        router.handlers[("POST", "/{delivery_id}/claim")],
    )
    asyncio.run(claim_handler(created.delivery_id))

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(claim_handler(created.delivery_id))

    assert exc_info.value.status_code == 409


@pytest.mark.unit
def test_reply_outbox_ack_route_marks_claimed_delivery_as_delivered(monkeypatch: pytest.MonkeyPatch) -> None:
    """reply outbox route 只应允许已 claim 的记录 ack 成功。"""

    service = _build_reply_delivery_service()
    created = service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="web:run_ack_1",
            session_id="session_ack_1",
            scene_name="web_chat",
            source_run_id="run_ack_1",
            reply_content="payload",
            metadata={"delivery_channel": "web", "delivery_target": "session_ack_1"},
        ),
    )
    claimed = service.claim_delivery(created.delivery_id)
    assert claimed.lease_id is not None
    router = _build_reply_outbox_router_for_test(monkeypatch, service)
    ack_handler = cast(
        Callable[[str, object], Coroutine[Any, Any, object]],
        router.handlers[("POST", "/{delivery_id}/ack")],
    )

    class _AckBody:
        """ack 请求体测试桩。"""

        def __init__(self, lease_id: str) -> None:
            self.lease_id = lease_id

    acknowledged = cast(
        ReplyDeliveryView,
        asyncio.run(ack_handler(claimed.delivery_id, _AckBody(claimed.lease_id))),
    )

    assert acknowledged.delivery_id == created.delivery_id
    assert acknowledged.state == ReplyOutboxState.DELIVERED.value
    assert acknowledged.delivery_attempt_count == 1


@pytest.mark.unit
def test_reply_outbox_ack_route_returns_conflict_without_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未经 claim 直接 ack 时，route 应返回 409。"""

    service = _build_reply_delivery_service()
    created = service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="web:run_ack_2",
            session_id="session_ack_2",
            scene_name="web_chat",
            source_run_id="run_ack_2",
            reply_content="payload",
            metadata={"delivery_channel": "web", "delivery_target": "session_ack_2"},
        ),
    )
    router = _build_reply_outbox_router_for_test(monkeypatch, service)
    ack_handler = cast(
        Callable[[str, object], Coroutine[Any, Any, object]],
        router.handlers[("POST", "/{delivery_id}/ack")],
    )

    class _AckBody:
        """ack 请求体测试桩。"""

        def __init__(self, lease_id: str) -> None:
            self.lease_id = lease_id

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(ack_handler(created.delivery_id, _AckBody("any-lease")))

    assert exc_info.value.status_code == 409


@pytest.mark.unit
def test_reply_outbox_get_route_returns_record_and_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """reply outbox get 路由应返回记录，并在缺失时返回 404。"""

    service = _build_reply_delivery_service()
    created = service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="web:run_get_1",
            session_id="session_get_1",
            scene_name="web_chat",
            source_run_id="run_get_1",
            reply_content="payload",
            metadata={"delivery_channel": "web", "delivery_target": "session_get_1"},
        ),
    )
    router = _build_reply_outbox_router_for_test(monkeypatch, service)
    get_handler = cast(
        Callable[[str], Coroutine[Any, Any, object]],
        router.handlers[("GET", "/{delivery_id}")],
    )

    loaded = cast(ReplyDeliveryView, asyncio.run(get_handler(created.delivery_id)))

    assert loaded.delivery_id == created.delivery_id

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(get_handler("missing_delivery"))

    assert exc_info.value.status_code == 404


@pytest.mark.unit
def test_reply_outbox_list_route_rejects_invalid_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """reply outbox list 路由应把非法状态映射为 400。"""

    service = _build_reply_delivery_service()
    router = _build_reply_outbox_router_for_test(monkeypatch, service)
    list_handler = cast(
        Callable[..., Coroutine[Any, Any, list[object]]],
        router.handlers[("GET", "")],
    )

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(list_handler(state="bad-state"))

    assert exc_info.value.status_code == 400


@pytest.mark.unit
def test_reply_outbox_nack_route_marks_failed_and_validates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """reply outbox nack 路由应支持成功失败回写，并映射 404/400。"""

    service = _build_reply_delivery_service()
    created = service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="web:run_nack_1",
            session_id="session_nack_1",
            scene_name="web_chat",
            source_run_id="run_nack_1",
            reply_content="payload",
            metadata={"delivery_channel": "web", "delivery_target": "session_nack_1"},
        ),
    )
    claimed = service.claim_delivery(created.delivery_id)
    assert claimed.lease_id is not None
    router = _build_reply_outbox_router_for_test(monkeypatch, service)
    nack_handler = cast(
        Callable[[str, object], Coroutine[Any, Any, object]],
        router.handlers[("POST", "/{delivery_id}/nack")],
    )

    class _Body:
        """nack 请求体测试桩。"""

        retryable = True
        error_message = "network failed"

        def __init__(self, lease_id: str) -> None:
            self.lease_id = lease_id

    failed = cast(
        ReplyDeliveryView,
        asyncio.run(nack_handler(created.delivery_id, _Body(claimed.lease_id))),
    )

    assert failed.state == ReplyOutboxState.FAILED_RETRYABLE.value

    with pytest.raises(_FakeHTTPException) as missing_exc:
        asyncio.run(nack_handler("missing_delivery", _Body("any-lease")))

    assert missing_exc.value.status_code == 404
