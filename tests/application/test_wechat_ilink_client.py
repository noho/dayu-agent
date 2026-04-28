"""WeChat iLink 客户端测试。"""

from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from dayu.wechat import ilink_client as ilink_client_module
from dayu.wechat.ilink_client import IlinkApiClient, IlinkApiError, build_ilink_auth_headers, build_x_wechat_uin_header


@pytest.mark.unit
def test_build_ilink_auth_headers_requires_token_when_auth_required() -> None:
    """验证鉴权请求缺少 token 时会抛错。"""

    with pytest.raises(ValueError):
        build_ilink_auth_headers(bot_token=None, auth_required=True)


@pytest.mark.unit
def test_build_ilink_auth_headers_include_bearer_and_random_uin() -> None:
    """验证请求头包含 Bearer 和随机 UIN。"""

    first = build_ilink_auth_headers(bot_token="token-1", auth_required=True)
    second = build_ilink_auth_headers(bot_token="token-1", auth_required=True)

    assert first["Authorization"] == "Bearer token-1"
    assert first["AuthorizationType"] == "ilink_bot_token"
    assert first["X-WECHAT-UIN"]
    assert second["X-WECHAT-UIN"]
    assert first["X-WECHAT-UIN"] != second["X-WECHAT-UIN"]


@pytest.mark.unit
def test_build_x_wechat_uin_header_returns_base64_uint32() -> None:
    """验证 X-WECHAT-UIN 请求头是 base64 编码的整数。"""

    header = build_x_wechat_uin_header()
    decoded = base64.b64decode(header).decode("utf-8")

    assert decoded.isdigit()


@pytest.mark.unit
def test_ilink_private_helpers_cover_url_and_nested_lookup() -> None:
    """验证 iLink 私有 helper 的基础分支。"""

    assert ilink_client_module._normalize_base_url(" https://example.com/ ") == "https://example.com"
    assert ilink_client_module._normalize_base_url(None) == ilink_client_module.DEFAULT_ILINK_BASE_URL
    assert ilink_client_module._looks_like_http_url("http://example.com") is True
    assert ilink_client_module._looks_like_http_url("mailto:test@example.com") is False
    assert ilink_client_module._find_first_string_value(
        {"outer": [{"message": ""}, {"nested": {"message": " hello "}}]},
        key="message",
    ) == "hello"
    assert ilink_client_module._find_first_string_value({"outer": [1, 2]}, key="missing") is None


@pytest.mark.unit
def test_update_auth_normalizes_token_and_url() -> None:
    """验证 update_auth 会裁剪 token 并标准化 base_url。"""

    client = IlinkApiClient(base_url="https://old.example.com/", bot_token=" old ")

    client.update_auth(base_url=" https://new.example.com/path/ ", bot_token=" token-1 ")

    assert client.base_url == "https://new.example.com/path"
    assert client.bot_token == "token-1"


@pytest.mark.unit
def test_get_updates_sends_required_cursor_and_channel_version() -> None:
    """验证长轮询请求体带上游标与 channel_version。"""

    captured: dict[str, object] = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"ret": 0, "msgs": [], "get_updates_buf": "cursor-next"})

    async def _run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        api = IlinkApiClient(client=client, bot_token="token-1")
        try:
            payload = await api.get_updates(get_updates_buf="cursor-1")
        finally:
            await client.aclose()
        assert payload["get_updates_buf"] == "cursor-next"

    asyncio.run(_run())

    assert captured["body"] == {
        "get_updates_buf": "cursor-1",
        "base_info": {"channel_version": "1.0.2"},
    }
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["authorization"] == "Bearer token-1"
    assert headers["authorizationtype"] == "ilink_bot_token"
    assert headers["x-wechat-uin"]


@pytest.mark.unit
def test_send_text_message_preserves_context_token() -> None:
    """验证发送消息时原样透传 context_token。"""

    captured: dict[str, object] = {}

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"ret": 0})

    async def _run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        api = IlinkApiClient(client=client, bot_token="token-1")
        try:
            await api.send_text_message(
                to_user_id="user@im.wechat",
                context_token="ctx-001",
                text="hello",
                group_id="group-1",
            )
        finally:
            await client.aclose()

    asyncio.run(_run())

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["base_info"] == {"channel_version": "1.0.2"}
    assert body["msg"] == {
        "from_user_id": "",
        "to_user_id": "user@im.wechat",
        "client_id": body["msg"]["client_id"],
        "message_type": 2,
        "message_state": 2,
        "context_token": "ctx-001",
        "group_id": "group-1",
        "item_list": [{"type": 1, "text_item": {"text": "hello"}}],
    }
    assert isinstance(body["msg"]["client_id"], str)
    assert body["msg"]["client_id"].startswith("dayu-wechat-")


@pytest.mark.unit
def test_get_qrcode_status_returns_wait_on_client_timeout() -> None:
    """验证二维码状态长轮询超时时按 wait 处理。"""

    async def _handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    async def _run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        api = IlinkApiClient(client=client)
        try:
            status = await api.get_qrcode_status("qr-1")
        finally:
            await client.aclose()
        assert status.status == "wait"

    asyncio.run(_run())


@pytest.mark.unit
def test_get_bot_qrcode_requires_qrcode_field() -> None:
    """验证二维码接口缺少 qrcode 字段时抛出业务错误。"""

    async def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ret": 0, "url": "https://example.com/q"})

    async def _run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        api = IlinkApiClient(client=client)
        try:
            with pytest.raises(IlinkApiError, match="qrcode"):
                await api.get_bot_qrcode()
        finally:
            await client.aclose()

    asyncio.run(_run())


@pytest.mark.unit
def test_get_typing_ticket_and_send_typing_match_reference_payload() -> None:
    """验证 typing 相关请求体与参考实现一致。"""

    captured: list[dict[str, object]] = []

    async def _handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        if request.url.path.endswith("/getconfig"):
            return httpx.Response(200, json={"ret": 0, "typing_ticket": "ticket-1"})
        return httpx.Response(200, json={"ret": 0})

    async def _run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        api = IlinkApiClient(client=client, bot_token="token-1")
        try:
            ticket = await api.get_typing_ticket(ilink_user_id="user@im.wechat", context_token="ctx-1")
            await api.send_typing(ilink_user_id="user@im.wechat", typing_ticket=ticket or "", status=1)
        finally:
            await client.aclose()

    asyncio.run(_run())

    assert captured[0] == {
        "ilink_user_id": "user@im.wechat",
        "context_token": "ctx-1",
        "base_info": {"channel_version": "1.0.2"},
    }
    assert captured[1] == {
        "ilink_user_id": "user@im.wechat",
        "typing_ticket": "ticket-1",
        "status": 1,
        "base_info": {"channel_version": "1.0.2"},
    }


@pytest.mark.unit
def test_request_json_raises_ilink_error_on_transport_failure() -> None:
    """验证底层 HTTPError 会包装成 IlinkApiError。"""

    async def _handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connect failed")

    async def _run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        api = IlinkApiClient(client=client)
        try:
            with pytest.raises(IlinkApiError, match="请求失败"):
                await api._request_json(method="GET", path="/test", auth_required=False)
        finally:
            await client.aclose()

    asyncio.run(_run())


@pytest.mark.unit
def test_request_json_raises_ilink_error_on_http_status() -> None:
    """验证 HTTP 非 2xx 会抛出包含状态码的 IlinkApiError。"""

    async def _handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="bad gateway")

    async def _run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        api = IlinkApiClient(client=client)
        try:
            with pytest.raises(IlinkApiError) as exc_info:
                await api._request_json(method="GET", path="/test", auth_required=False)
        finally:
            await client.aclose()
        assert exc_info.value.status_code == 502
        assert exc_info.value.payload == "bad gateway"

    asyncio.run(_run())


@pytest.mark.unit
def test_request_json_rejects_non_json_and_non_object_payload() -> None:
    """验证非 JSON 响应和非对象 JSON 都会抛出业务异常。"""

    async def _non_json_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="plain-text")

    async def _list_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2, 3])

    async def _run() -> None:
        non_json_client = httpx.AsyncClient(transport=httpx.MockTransport(_non_json_handler))
        list_client = httpx.AsyncClient(transport=httpx.MockTransport(_list_handler))
        try:
            with pytest.raises(IlinkApiError, match="非 JSON"):
                await IlinkApiClient(client=non_json_client)._request_json(method="GET", path="/test", auth_required=False)
            with pytest.raises(IlinkApiError, match="顶层不是对象"):
                await IlinkApiClient(client=list_client)._request_json(method="GET", path="/test", auth_required=False)
        finally:
            await non_json_client.aclose()
            await list_client.aclose()

    asyncio.run(_run())


@pytest.mark.unit
def test_request_json_raises_business_error_using_errmsg_or_message() -> None:
    """验证 ret 非零时优先提取 errmsg/message。"""

    async def _errmsg_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ret": 500, "nested": {"errmsg": "boom"}})

    async def _message_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ret": 400, "message": "bad request"})

    async def _run() -> None:
        errmsg_client = httpx.AsyncClient(transport=httpx.MockTransport(_errmsg_handler))
        message_client = httpx.AsyncClient(transport=httpx.MockTransport(_message_handler))
        try:
            with pytest.raises(IlinkApiError, match="boom"):
                await IlinkApiClient(client=errmsg_client)._request_json(method="GET", path="/test", auth_required=False)
            with pytest.raises(IlinkApiError, match="bad request"):
                await IlinkApiClient(client=message_client)._request_json(method="GET", path="/test", auth_required=False)
        finally:
            await errmsg_client.aclose()
            await message_client.aclose()

    asyncio.run(_run())


@pytest.mark.unit
def test_request_json_rethrows_timeout_when_no_fallback() -> None:
    """验证未提供 timeout_fallback 时会把超时继续抛出。"""

    async def _handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    async def _run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        api = IlinkApiClient(client=client)
        try:
            with pytest.raises(httpx.TimeoutException):
                await api._request_json(method="GET", path="/test", auth_required=False)
        finally:
            await client.aclose()

    asyncio.run(_run())


@pytest.mark.unit
def test_request_json_logs_debug_when_timeout_fallback_hits(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证 timeout_fallback 命中时会留下可观测的 debug 日志，便于排查长轮询超时。"""

    async def _handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    async def _run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
        api = IlinkApiClient(client=client)
        try:
            with caplog.at_level("DEBUG", logger="WECHAT.ILINK_CLIENT"):
                payload = await api._request_json(
                    method="POST",
                    path="/ilink/bot/getupdates",
                    auth_required=False,
                    timeout_fallback={"ret": 0, "msgs": [], "get_updates_buf": ""},
                )
            assert payload == {"ret": 0, "msgs": [], "get_updates_buf": ""}
            messages = [record.getMessage() for record in caplog.records]
            assert any(
                "iLink 请求超时命中 fallback" in msg
                and "method=POST" in msg
                and "path=/ilink/bot/getupdates" in msg
                for msg in messages
            ), messages
        finally:
            await client.aclose()

    asyncio.run(_run())