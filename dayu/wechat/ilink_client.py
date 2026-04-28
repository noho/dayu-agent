"""iLink HTTP/JSON 客户端。

该模块只负责微信 iLink 协议的 HTTP 通信细节：
- 二维码登录
- 长轮询收消息
- 文本回复
- typing best-effort 调用

不负责：
- Dayu UI / Service / Host 装配
- ChatService 编排
- 会话状态持久化
"""

from __future__ import annotations

import base64
import random
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from dayu.log import Log

DEFAULT_ILINK_BASE_URL = "https://ilinkai.weixin.qq.com"
DEFAULT_BOT_TYPE = 3
DEFAULT_CHANNEL_VERSION = "1.0.2"
TYPING_STATUS_TYPING = 1
TYPING_STATUS_CANCEL = 2

MODULE = "WECHAT.ILINK_CLIENT"


def build_x_wechat_uin_header() -> str:
    """生成 iLink 要求的随机 `X-WECHAT-UIN` 请求头值。

    Args:
        无。

    Returns:
        base64 编码后的十进制随机 uint32 字符串。

    Raises:
        无。
    """

    random_uint32 = random.getrandbits(32)
    return base64.b64encode(str(random_uint32).encode("utf-8")).decode("utf-8")


def build_ilink_auth_headers(
    *,
    bot_token: str | None,
    auth_required: bool,
) -> dict[str, str]:
    """构建 iLink 请求头。

    Args:
        bot_token: Bearer token；未登录时可为空。
        auth_required: 当前请求是否必须带鉴权头。

    Returns:
        HTTP 请求头字典。

    Raises:
        ValueError: 当请求要求鉴权但未提供 `bot_token` 时抛出。
    """

    if auth_required and not bot_token:
        raise ValueError("当前 iLink 请求要求已登录，但 bot_token 缺失")
    headers = {
        "Content-Type": "application/json",
        "X-WECHAT-UIN": build_x_wechat_uin_header(),
    }
    if bot_token:
        headers["AuthorizationType"] = "ilink_bot_token"
        headers["Authorization"] = f"Bearer {bot_token}"
    return headers


def _normalize_base_url(base_url: str | None) -> str:
    """标准化 iLink base URL。

    Args:
        base_url: 原始 base URL。

    Returns:
        去掉末尾 `/` 的 URL；为空时回退到默认地址。

    Raises:
        无。
    """

    normalized = str(base_url or DEFAULT_ILINK_BASE_URL).strip()
    return normalized.rstrip("/")


def _looks_like_http_url(value: str | None) -> bool:
    """判断字符串是否看起来是 HTTP/HTTPS URL。

    Args:
        value: 待判断字符串。

    Returns:
        `True` 表示看起来是 URL，否则返回 `False`。

    Raises:
        无。
    """

    normalized = str(value or "").strip().lower()
    return normalized.startswith("http://") or normalized.startswith("https://")


def _find_first_string_value(payload: Any, *, key: str) -> str | None:
    """在任意嵌套 JSON 中递归查找首个字符串字段。

    Args:
        payload: JSON 负载。
        key: 目标字段名。

    Returns:
        首个匹配到的非空字符串；未找到时返回 `None`。

    Raises:
        无。
    """

    if isinstance(payload, dict):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        for nested in payload.values():
            found = _find_first_string_value(nested, key=key)
            if found:
                return found
        return None
    if isinstance(payload, list):
        for item in payload:
            found = _find_first_string_value(item, key=key)
            if found:
                return found
    return None


def _build_client_id() -> str:
    """生成 outbound sendmessage 使用的 client_id。

    Args:
        无。

    Returns:
        稳定前缀 + UUID 的 client_id。

    Raises:
        无。
    """

    return f"dayu-wechat-{uuid.uuid4()}"


class IlinkApiError(RuntimeError):
    """iLink API 调用失败。"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        business_ret_code: int | None = None,
        payload: Any = None,
    ) -> None:
        """初始化异常对象。

        Args:
            message: 错误信息。
            status_code: HTTP 状态码。
            business_ret_code: iLink 业务返回码。
            payload: 服务端返回 payload。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(message)
        self.status_code = status_code
        self.business_ret_code = business_ret_code
        self.payload = payload


@dataclass(frozen=True)
class QRCodeLoginTicket:
    """二维码登录票据。"""

    qrcode: str
    url: str | None = None
    qrcode_img_content: str | None = None


@dataclass(frozen=True)
class QRCodeLoginStatus:
    """二维码登录状态。"""

    status: str
    bot_token: str | None = None
    base_url: str | None = None


class IlinkApiClient:
    """iLink API 异步客户端。"""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        bot_token: str | None = None,
        channel_version: str = DEFAULT_CHANNEL_VERSION,
        client: httpx.AsyncClient | None = None,
        long_poll_timeout_sec: float = 40.0,
    ) -> None:
        """初始化 iLink 客户端。

        Args:
            base_url: iLink 基础 URL。
            bot_token: Bearer token。
            channel_version: 通道版本号。
            client: 外部注入的异步 HTTP client。
            long_poll_timeout_sec: 长轮询读取超时时间。

        Returns:
            无。

        Raises:
            无。
        """

        self._base_url = _normalize_base_url(base_url)
        self._bot_token = str(bot_token or "").strip() or None
        self._channel_version = str(channel_version).strip() or DEFAULT_CHANNEL_VERSION
        self._long_poll_timeout_sec = float(long_poll_timeout_sec)
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient()

    @property
    def base_url(self) -> str:
        """返回当前基础 URL。"""

        return self._base_url

    @property
    def bot_token(self) -> str | None:
        """返回当前 bot token。"""

        return self._bot_token

    def update_auth(self, *, base_url: str | None, bot_token: str | None) -> None:
        """更新客户端登录态。

        Args:
            base_url: 新的 base URL。
            bot_token: 新的 Bearer token。

        Returns:
            无。

        Raises:
            无。
        """

        self._base_url = _normalize_base_url(base_url)
        self._bot_token = str(bot_token or "").strip() or None

    async def aclose(self) -> None:
        """关闭底层 HTTP client。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        if self._owns_client:
            await self._client.aclose()

    async def get_bot_qrcode(self, *, bot_type: int = DEFAULT_BOT_TYPE) -> QRCodeLoginTicket:
        """申请登录二维码。

        Args:
            bot_type: iLink bot 类型。

        Returns:
            二维码票据。

        Raises:
            IlinkApiError: 当远端返回错误时抛出。
        """

        payload = await self._request_json(
            method="GET",
            path="/ilink/bot/get_bot_qrcode",
            params={"bot_type": bot_type},
            auth_required=False,
        )
        qrcode = str(payload.get("qrcode") or "").strip()
        if not qrcode:
            raise IlinkApiError("get_bot_qrcode 未返回 qrcode", payload=payload)
        return QRCodeLoginTicket(
            qrcode=qrcode,
            url=(
                str(payload.get("url") or payload.get("qrcode_url") or payload.get("qrcode_img_content") or "").strip()
                or None
            ),
            qrcode_img_content=str(payload.get("qrcode_img_content") or "").strip() or None,
        )

    async def get_qrcode_status(self, qrcode: str) -> QRCodeLoginStatus:
        """查询二维码登录状态。

        Args:
            qrcode: 二维码票据值。

        Returns:
            登录状态对象。

        Raises:
            IlinkApiError: 当远端返回错误时抛出。
        """

        payload = await self._request_json(
            method="GET",
            path="/ilink/bot/get_qrcode_status",
            params={"qrcode": qrcode},
            auth_required=False,
            read_timeout_sec=self._long_poll_timeout_sec,
            extra_headers={"iLink-App-ClientVersion": "1"},
            timeout_fallback={"status": "wait"},
        )
        return QRCodeLoginStatus(
            status=str(payload.get("status") or "unknown").strip() or "unknown",
            bot_token=str(payload.get("bot_token") or "").strip() or None,
            base_url=str(payload.get("baseurl") or payload.get("base_url") or "").strip() or None,
        )

    async def get_updates(self, *, get_updates_buf: str) -> dict[str, Any]:
        """执行一次长轮询获取消息。

        Args:
            get_updates_buf: 上次返回的游标。

        Returns:
            服务端 JSON 结果。

        Raises:
            IlinkApiError: 当远端返回错误时抛出。
        """

        return await self._request_json(
            method="POST",
            path="/ilink/bot/getupdates",
            json_body={
                "get_updates_buf": get_updates_buf,
                "base_info": {"channel_version": self._channel_version},
            },
            auth_required=True,
            read_timeout_sec=self._long_poll_timeout_sec,
            timeout_fallback={"ret": 0, "msgs": [], "get_updates_buf": get_updates_buf},
        )

    async def send_text_message(
        self,
        *,
        to_user_id: str,
        context_token: str,
        text: str,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        """发送文本消息。

        Args:
            to_user_id: 接收方用户 ID。
            context_token: 上游消息携带的上下文 token。
            text: 回复文本。
            group_id: 群聊 ID；直聊时为空。

        Returns:
            服务端 JSON 结果。

        Raises:
            IlinkApiError: 当远端返回错误时抛出。
        """

        msg: dict[str, Any] = {
            "from_user_id": "",
            "to_user_id": to_user_id,
            "client_id": _build_client_id(),
            "message_type": 2,
            "message_state": 2,
            "context_token": context_token,
            "item_list": [{"type": 1, "text_item": {"text": text}}],
        }
        if group_id:
            msg["group_id"] = group_id
        return await self._request_json(
            method="POST",
            path="/ilink/bot/sendmessage",
            json_body={
                "msg": msg,
                "base_info": {"channel_version": self._channel_version},
            },
            auth_required=True,
        )

    async def get_typing_ticket(self, *, ilink_user_id: str, context_token: str | None = None) -> str | None:
        """获取 typing ticket。

        Args:
            ilink_user_id: 微信用户 ID。
            context_token: 当前上下文 token。

        Returns:
            typing_ticket；服务端未返回时为 `None`。

        Raises:
            IlinkApiError: 当远端返回错误时抛出。
        """

        payload = await self._request_json(
            method="POST",
            path="/ilink/bot/getconfig",
            json_body={
                "ilink_user_id": ilink_user_id,
                "context_token": context_token,
                "base_info": {"channel_version": self._channel_version},
            },
            auth_required=True,
        )
        return _find_first_string_value(payload, key="typing_ticket")

    async def send_typing(
        self,
        *,
        ilink_user_id: str,
        typing_ticket: str,
        status: int = TYPING_STATUS_TYPING,
    ) -> dict[str, Any]:
        """发送“正在输入”状态。

        Args:
            ilink_user_id: 微信用户 ID。
            typing_ticket: getconfig 返回的 ticket。
            status: typing 状态，`1` 表示输入中，`2` 表示取消。

        Returns:
            服务端 JSON 结果。

        Raises:
            IlinkApiError: 当远端返回错误时抛出。
        """

        return await self._request_json(
            method="POST",
            path="/ilink/bot/sendtyping",
            json_body={
                "ilink_user_id": ilink_user_id,
                "typing_ticket": typing_ticket,
                "status": status,
                "base_info": {"channel_version": self._channel_version},
            },
            auth_required=True,
        )

    async def _request_json(
        self,
        *,
        method: str,
        path: str,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        auth_required: bool,
        read_timeout_sec: float = 20.0,
        extra_headers: Mapping[str, str] | None = None,
        timeout_fallback: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """发送请求并返回 JSON。

        Args:
            method: HTTP 方法。
            path: 接口路径。
            params: 查询参数。
            json_body: JSON 请求体。
            auth_required: 是否要求 Bearer token。
            read_timeout_sec: 响应读取超时。
            extra_headers: 额外请求头。
            timeout_fallback: 当请求因客户端超时而失败时返回的兜底结果。

        Returns:
            JSON 字典。

        Raises:
            IlinkApiError: 当 HTTP 或业务 ret 失败时抛出。
        """

        url = f"{self._base_url}/{path.lstrip('/')}"
        headers = build_ilink_auth_headers(bot_token=self._bot_token, auth_required=auth_required)
        if extra_headers:
            headers.update(extra_headers)
        timeout = httpx.Timeout(connect=10.0, read=read_timeout_sec, write=10.0, pool=10.0)
        try:
            response = await self._client.request(
                method=method,
                url=url,
                params=params,
                json=json_body,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException:
            if timeout_fallback is not None:
                # 长轮询的"读超时"是预期路径（服务端无新消息时主动断开），
                # 但短请求超时可能是异常事件。这里用 debug 级日志记录，
                # 既能在排查时看到 fallback 命中，也不污染正常运行的日志流。
                Log.debug(
                    f"iLink 请求超时命中 fallback: method={method} path={path} "
                    f"read_timeout_sec={read_timeout_sec}",
                    module=MODULE,
                )
                return dict(timeout_fallback)
            raise
        except httpx.HTTPError as exc:
            raise IlinkApiError(f"iLink 请求失败: {exc}") from exc

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise IlinkApiError(
                f"iLink HTTP 错误: {response.status_code}",
                status_code=response.status_code,
                payload=response.text,
            ) from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise IlinkApiError("iLink 返回了非 JSON 响应", status_code=response.status_code, payload=response.text) from exc
        if not isinstance(payload, dict):
            raise IlinkApiError("iLink 返回的 JSON 顶层不是对象", status_code=response.status_code, payload=payload)
        ret = payload.get("ret")
        if ret not in (None, 0):
            message = _find_first_string_value(payload, key="errmsg") or _find_first_string_value(payload, key="message")
            raise IlinkApiError(
                message or f"iLink 业务错误 ret={ret}",
                status_code=response.status_code,
                business_ret_code=ret if isinstance(ret, int) else None,
                payload=payload,
            )
        return payload


__all__ = [
    "DEFAULT_BOT_TYPE",
    "DEFAULT_CHANNEL_VERSION",
    "DEFAULT_ILINK_BASE_URL",
    "IlinkApiClient",
    "IlinkApiError",
    "QRCodeLoginStatus",
    "QRCodeLoginTicket",
    "TYPING_STATUS_CANCEL",
    "TYPING_STATUS_TYPING",
    "build_ilink_auth_headers",
    "build_x_wechat_uin_header",
]
