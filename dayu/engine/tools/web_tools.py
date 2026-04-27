"""联网检索工具模块。

该模块提供写作流水线使用的联网检索能力，包含：
- `search_web`：按关键词检索公开网页。
- `fetch_web_page`：抓取网页正文文本。

设计约束：
- 仅允许访问 `http/https` 地址。
- 拒绝内网、回环地址与本地地址。
- Provider 选择遵循：`tavily` -> `serper` -> `duckduckgo`。

维护说明(不拆分本模块):
    本模块约 2000 行, 核心是 search_web 和 fetch_web_page 两个工厂
    函数. fetch 路径是一条线性 pipeline(requests -> content-type 探测
    -> docling 转换 -> playwright fallback -> URL 安全检查), 各工具
    函数互相依赖. Playwright 子系统虽有 330 行, 但深度嵌入 fetch 的
    重试/fallback 逻辑, 拆分会引入大量参数传递. 外部仅消费
    register_web_tools 一个符号.
"""

from __future__ import annotations

import json
import ipaddress
import os
import re
import socket
import ssl
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol
from urllib.parse import quote, urlparse

import requests
from requests.utils import requote_uri
from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError

from dayu.contracts.cancellation import CancellationToken
from dayu.contracts.env_keys import SEC_USER_AGENT_ENV
from dayu.contracts.protocols import ToolExecutionContext
from dayu.log import Log
from dayu.engine.processors.html_pipeline import HtmlPipelineStageError, convert_html_to_llm_markdown
from dayu.engine.tool_contracts import ToolSchema, ToolTruncateSpec
from dayu.engine.tool_errors import ToolBusinessError
from dayu.engine.tool_registry import ToolRegistry
from dayu.engine.tools.base import tool
from dayu.engine.tools import web_fetch_orchestrator as _web_fetch_orchestrator
from dayu.engine.tools import web_playwright_backend as _web_playwright_backend
from dayu.engine.tools.web_challenge_detection import (
    BotChallengeDetectionResult,
    detect_bot_challenge as _detect_bot_challenge,
)
from dayu.engine.tools.web_http_encoding import (
    _build_accept_encoding_value,
    _decode_response_text,
    _extract_charset_from_content_type,
    _extract_charset_from_html_bytes,
    _extract_content_encoding_tokens,
    _find_unsupported_content_encodings,
    _is_optional_module_available,
    _normalize_charset_name,
    _resolve_response_text_encoding,
    _resolve_supported_accept_encodings,
)
from dayu.engine.tools.web_http_session import (
    _compute_deadline_monotonic,
    _create_no_retry_session,
    _create_retry_session,
    _get_no_retry_web_session,
    _get_web_session,
    _normalize_timeout_budget,
    _prepare_call_session,
    _resolve_timeout_budget,
    _safe_timeout,
)
from dayu.engine.tools.web_recovery import (
    RECOVERY_CONTRACT_VERSION,
    NEXT_ACTION_CHANGE_SOURCE,
    NEXT_ACTION_CONTINUE_WITHOUT_WEB,
    NEXT_ACTION_RETRY,
    REASON_BLOCKED_BY_SITE_POLICY,
    REASON_CONTENT_CONVERSION_FAILED,
    REASON_EMPTY_CONTENT,
    REASON_HTTP_ERROR,
    REASON_REDIRECT_CHAIN_TOO_LONG,
    REASON_REQUEST_TIMEOUT,
    build_hint,
    normalize_next_action,
    normalize_reason,
)
from dayu.engine.tools.web_search_providers import SearchWebOutput, search_public_web

MODULE = "ENGINE.WEB_TOOLS"

_ALLOWED_SCHEMES = {"http", "https"}
_PRIVATE_HOST_PATTERNS = (
    "localhost",
    "127.",
    "0.0.0.0",
    "::1",
)
_FAKE_IP_NETWORKS = (
    ipaddress.ip_network("198.18.0.0/15"),
)

_DEFAULT_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)
_DEFAULT_SEC_USER_AGENT = "Codex Web Fetcher support@example.com"
_DEFAULT_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
_DEFAULT_ACCEPT_LANGUAGE = "zh-CN,zh;q=0.9,en;q=0.8"

# --- Client Hints（现代 Chrome 必带，缺失是典型爬虫特征）---
_DEFAULT_SEC_CH_UA = '"Chromium";v="131", "Google Chrome";v="131", "Not_A Brand";v="24"'
_DEFAULT_SEC_CH_UA_MOBILE = "?0"
_DEFAULT_SEC_CH_UA_PLATFORM = '"macOS"'

_RESPONSE_SNIPPET_MAX_CHARS = 500
_EMPTY_CONTENT_MIN_CHARS = 5


_FetchContentRuntimeContext = _web_fetch_orchestrator._FetchContentRuntimeContext


class _RegisteredToolCallable(Protocol):
    """带工具装饰器元数据的可调用对象协议。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """

    __tool_name__: str
    __tool_schema__: ToolSchema


def _build_registered_tool_parts(
    tool_func: _RegisteredToolCallable,
) -> tuple[str, _RegisteredToolCallable, ToolSchema]:
    """提取带 decorator 元数据的工具三元组。

    Args:
        tool_func: 已经过 ``@tool`` 装饰的工具函数。

    Returns:
        ``(tool_name, tool_callable, tool_schema)`` 三元组。

    Raises:
        无。
    """

    return tool_func.__tool_name__, tool_func, tool_func.__tool_schema__


_FetchContentConversionError = _web_fetch_orchestrator._FetchContentConversionError


def _load_storage_state_cookies(storage_state_path: str) -> list[dict[str, Any]]:
    """从 Playwright storage state 文件读取 cookie 列表。

    Args:
        storage_state_path: storage state 文件路径。

    Returns:
        storage state 中声明的 cookie 列表；文件不可用或结构非法时返回空列表。

    Raises:
        无。
    """

    normalized_path = str(storage_state_path or "").strip()
    if not normalized_path or not os.path.isfile(normalized_path):
        return []

    try:
        with open(normalized_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []

    raw_cookies = payload.get("cookies") if isinstance(payload, dict) else None
    if not isinstance(raw_cookies, list):
        return []

    normalized_cookies: list[dict[str, Any]] = []
    for item in raw_cookies:
        if isinstance(item, dict):
            normalized_cookies.append(item)
    return normalized_cookies


def _apply_storage_state_cookies_to_session(
    session: requests.Session,
    *,
    storage_state_path: str,
) -> int:
    """把 Playwright storage state 里的 cookie 注入 requests Session。

    诊断流程里人工浏览器验证后拿到的 storage state，往往包含能让
    `requests` 主路径直接恢复正文访问的 cookie。这里把该状态机械复制到
    当前 `Session`，避免只把 storage state 用在 Playwright fallback。

    Args:
        session: 当前调用使用的 requests Session。
        storage_state_path: storage state 文件路径。

    Returns:
        成功注入的 cookie 数量。

    Raises:
        无。
    """

    if not isinstance(session, requests.Session):
        return 0

    applied_count = 0
    for cookie in _load_storage_state_cookies(storage_state_path):
        name = str(cookie.get("name", "") or "").strip()
        value = str(cookie.get("value", "") or "")
        domain = str(cookie.get("domain", "") or "").strip() or None
        path = str(cookie.get("path", "") or "").strip() or "/"
        if not name:
            continue
        # HTTP header 值使用 latin-1 编码；含非 latin-1 字符的 cookie
        # （如本地化日期时间戳）会导致 requests 发送时 UnicodeEncodeError。
        # 这类 cookie 通常不含认证信息，直接跳过。
        try:
            name.encode("latin-1")
            value.encode("latin-1")
        except UnicodeEncodeError:
            continue
        session.cookies.set(
            name,
            value,
            domain=domain,
            path=path,
            secure=bool(cookie.get("secure", False)),
        )
        applied_count += 1
    return applied_count


def _iter_exception_chain(error: BaseException) -> list[BaseException]:
    """展开异常对象的因果链与嵌套 reason。

    Args:
        error: 起始异常。

    Returns:
        按遍历顺序展开后的异常对象列表。

    Raises:
        无。
    """

    pending: list[BaseException] = [error]
    visited: set[int] = set()
    collected: list[BaseException] = []
    while pending:
        current = pending.pop()
        marker = id(current)
        if marker in visited:
            continue
        visited.add(marker)
        collected.append(current)
        nested_candidates = [
            getattr(current, "__cause__", None),
            getattr(current, "__context__", None),
            getattr(current, "reason", None),
        ]
        first_arg = current.args[0] if getattr(current, "args", ()) else None
        if isinstance(first_arg, BaseException):
            nested_candidates.append(first_arg)
        for nested in nested_candidates:
            if isinstance(nested, BaseException):
                pending.append(nested)
    return collected


def _is_timeout_like_request_exception(error: BaseException) -> bool:
    """判断 requests 异常是否本质上由读超时引起。

    真实链路里，`requests` 可能直接抛 `requests.Timeout`，也可能把
    `urllib3.ReadTimeoutError` 包成 `requests.ConnectionError(MaxRetryError(...))`。

    Args:
        error: 待判断的异常对象。

    Returns:
        `True` 表示属于超时类异常，否则返回 `False`。

    Raises:
        无。
    """

    for current in _iter_exception_chain(error):
        if isinstance(current, (requests.Timeout, Urllib3ReadTimeoutError)):
            return True
    return False


def _is_timeout_like_exception(error: BaseException) -> bool:
    """判断任意异常是否本质上属于超时类异常。

    Args:
        error: 待判断的异常对象。

    Returns:
        `True` 表示异常链中存在超时语义，否则返回 `False`。

    Raises:
        无。
    """

    return _is_timeout_like_request_exception(error)


def _is_ssl_like_request_exception(error: BaseException) -> bool:
    """判断 requests 异常是否本质上属于 SSL/TLS 握手失败。

    Args:
        error: 待判断的异常对象。

    Returns:
        `True` 表示异常链中存在 SSL/TLS 失败语义，否则返回 `False`。

    Raises:
        无。
    """

    for current in _iter_exception_chain(error):
        if isinstance(current, (requests.exceptions.SSLError, ssl.SSLError)):
            return True
        current_name = type(current).__name__.lower()
        current_message = str(current).lower()
        if "ssl" in current_name or "tls" in current_name:
            return True
        if any(
            marker in current_message
            for marker in (
                "ssl",
                "tls",
                "certificate",
                "unexpected eof while reading",
                "wrong version number",
                "handshake failure",
            )
        ):
            return True
    return False


def _close_response_safely(response: Any) -> None:
    """尽力关闭响应对象，兼容测试桩。

    Args:
        response: 任意响应对象。

    Returns:
        无。

    Raises:
        无。
    """

    _web_fetch_orchestrator._close_response_safely(response)


_build_fetch_content_runtime_context = _web_fetch_orchestrator._build_fetch_content_runtime_context
_extract_response_snippet = _web_fetch_orchestrator._extract_response_snippet
_sanitize_response_headers = _web_fetch_orchestrator._sanitize_response_headers
_should_escalate_conversion_failure_to_browser = _web_fetch_orchestrator._should_escalate_conversion_failure_to_browser
_should_escalate_http_status_to_browser = _web_fetch_orchestrator._should_escalate_http_status_to_browser
_should_escalate_pipeline_failure_to_browser = _web_fetch_orchestrator._should_escalate_pipeline_failure_to_browser
_should_escalate_stage_result_to_browser = _web_fetch_orchestrator._should_escalate_stage_result_to_browser


def _build_playwright_success_payload(url: str, pw_result: dict[str, Any]) -> dict[str, Any]:
    """将 Playwright 回退成功结果规整为 fetch_web_page 输出。"""

    return {
        "url": url,
        "final_url": pw_result.get("final_url", url),
        "title": pw_result.get("title", ""),
        "content": pw_result.get("content", ""),
        "fetch_backend": "playwright",
    }


def _build_text_excerpt(text: str) -> str:
    """规整任意文本并截取诊断前缀。

    Args:
        text: 原始文本。

    Returns:
        规整后的限长文本前缀。

    Raises:
        无。
    """

    normalized = _normalize_whitespace(text or "")
    return normalized[:_RESPONSE_SNIPPET_MAX_CHARS]


def _resolve_execution_cancellation_token(
    execution_context: ToolExecutionContext | None,
) -> CancellationToken | None:
    """从 execution context 中提取取消令牌。"""

    if execution_context is None:
        return None
    return execution_context.cancellation_token


def _raise_if_tool_cancelled(cancellation_token: CancellationToken | None) -> None:
    """在进入新的联网阶段前执行协作式取消检查。"""

    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()


def _try_playwright_fallback(
    *,
    url: str,
    timeout_seconds: float,
    headers: dict[str, str],
    timeout_budget: float | None,
    deadline_monotonic: float | None,
    playwright_channel: str | None = None,
    playwright_storage_state_path: str = "",
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any] | None:
    """尝试使用 Playwright 浏览器回退抓取页面。

    Args:
        url: 原始网页链接。
        timeout_seconds: 浏览器抓取基础超时秒数。
        headers: 请求头。
        timeout_budget: Runner 注入的工具总预算。
        deadline_monotonic: 当前工具调用 deadline。
        playwright_channel: 浏览器回退使用的 Chromium channel。
        playwright_storage_state_path: 浏览器回退可选 storage state 文件路径。

    Returns:
        成功时返回标准化后的抓取结果；失败时返回 `None`。

    Raises:
        无。
    """

    playwright_kwargs: dict[str, Any] = {
        "url": url,
        "timeout_seconds": timeout_seconds,
        "headers": headers,
        "timeout_budget": timeout_budget,
        "deadline_monotonic": deadline_monotonic,
        "playwright_channel": playwright_channel,
        "playwright_storage_state_path": playwright_storage_state_path,
    }
    if cancellation_token is not None:
        playwright_kwargs["cancellation_token"] = cancellation_token
    pw_result = _fetch_and_convert_with_playwright(**playwright_kwargs)
    if not pw_result.get("ok"):
        Log.debug(
            "Playwright 浏览器回退未成功: "
            f"availability={pw_result.get('availability')} "
            f"reason={pw_result.get('reason')}",
            module=MODULE,
        )
        return None
    return _build_playwright_success_payload(url, pw_result)


def _status_class(status_code: Optional[int]) -> str:
    """将状态码归类为状态段。

    Args:
        status_code: HTTP 状态码。

    Returns:
        形如 ``2xx``/``4xx`` 的分类，未知时返回 ``unknown``。

    Raises:
        无。
    """

    if status_code is None:
        return "unknown"
    hundred = status_code // 100
    if 1 <= hundred <= 5:
        return f"{hundred}xx"
    return "unknown"


def _build_domain_home_url(url: str) -> str:
    """构建同域首页 URL。

    Args:
        url: 目标链接。

    Returns:
        同域首页 URL。

    Raises:
        ValueError: 当 URL 无法解析时抛出。
    """

    parsed = urlparse(_normalize_url_for_http(url))
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无效 URL: {url}")
    return f"{parsed.scheme}://{parsed.netloc}/"


def _normalize_url_for_http(url: str) -> str:
    """将 URL 规整为适合 HTTP 请求与 Header 传输的 ASCII 形式。

    设计意图：
    - `requests` 可以处理部分 Unicode URL，但 HTTP header 值最终会走
      `latin-1` 编码；若直接把含中文路径的 URL 放进 `Referer`，会在
      `http.client.putheader()` 阶段抛出 `UnicodeEncodeError`。
    - 因此这里统一做两件事：域名转为 IDNA，路径/查询/片段转为百分号编码。

    Args:
        url: 原始 URL。

    Returns:
        适合 HTTP 传输的 ASCII URL。

    Raises:
        ValueError: 当 URL 缺少 scheme 或 netloc 时抛出。
    """

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无效 URL: {url}")

    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError(f"无效 URL: {url}")

    username = parsed.username
    password = parsed.password
    auth_parts: list[str] = []
    if username is not None:
        auth_parts.append(quote(username, safe=""))
    if password is not None:
        auth_parts.append(quote(password, safe=""))

    host_ascii = hostname.encode("idna").decode("ascii")
    auth_prefix = ""
    if auth_parts:
        auth_prefix = ":".join(auth_parts) + "@"
    netloc = f"{auth_prefix}{host_ascii}"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    normalized = parsed._replace(netloc=netloc).geturl()
    return requote_uri(normalized)


def _build_referer(url: str) -> str:
    """构建请求 Referer。

    Args:
        url: 目标链接。

    Returns:
        Referer 链接。

    Raises:
        ValueError: 当 URL 无法解析时抛出。
    """

    parsed = urlparse(_normalize_url_for_http(url))
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"无效 URL: {url}")
    path = parsed.path or "/"
    if path == "/":
        return f"{parsed.scheme}://{parsed.netloc}/"
    parent = path.rsplit("/", 1)[0]
    if not parent:
        parent = "/"
    if not parent.endswith("/"):
        parent = f"{parent}/"
    return f"{parsed.scheme}://{parsed.netloc}{parent}"


def _warmup_domain(
    session: requests.Session,
    *,
    url: str,
    timeout_seconds: float,
    headers: dict[str, str],
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """对目标域做一次预热请求以建立 Cookie。"""

    return _web_fetch_orchestrator._warmup_domain(
        session,
        url=url,
        timeout_seconds=timeout_seconds,
        headers=headers,
        resolve_timeout_budget=_resolve_timeout_budget,
        build_domain_home_url=_build_domain_home_url,
        is_timeout_like_exception=_is_timeout_like_exception,
        timeout_budget=timeout_budget,
        deadline_monotonic=deadline_monotonic,
        cancellation_token=cancellation_token,
    )


def _probe_content_type(
    session: requests.Session,
    *,
    url: str,
    timeout_seconds: float,
    headers: dict[str, str],
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """探测目标资源类型（HEAD 优先，失败降级到 GET）。"""

    return _web_fetch_orchestrator._probe_content_type(
        session,
        url=url,
        timeout_seconds=timeout_seconds,
        headers=headers,
        resolve_timeout_budget=_resolve_timeout_budget,
        is_timeout_like_exception=_is_timeout_like_exception,
        timeout_budget=timeout_budget,
        deadline_monotonic=deadline_monotonic,
        cancellation_token=cancellation_token,
    )


def _raise_fetch_failure(
    *,
    url: str,
    error_code: str,
    message: str,
    hint: str,
    next_action: str,
    http_status: Optional[int] = None,
    internal_diagnostics: Optional[dict[str, Any]] = None,
) -> None:
    """记录诊断日志并抛出 ToolBusinessError。

    将失败信息写入诊断日志后，以 ToolBusinessError 的形式抛出，
    由 ToolRegistry 统一捕获并转换为标准错误信封。

    Args:
        url: 原始请求 URL。
        error_code: 错误码（对齐 ErrorCode 枚举值）。
        message: 错误说明。
        hint: LLM 可执行提示（来自 web_recovery.build_hint）。
        next_action: 下一步动作（retry/change_source/continue_without_web）。
        http_status: HTTP 状态码（可选）。
        internal_diagnostics: 内部诊断信息（可选，仅写日志）。

    Returns:
        无（始终抛出异常）。

    Raises:
        ToolBusinessError: 始终抛出。
    """
    normalized_action = normalize_next_action(next_action)
    # 构建诊断日志
    diagnostics: dict[str, Any] = {
        "url": url,
        "error_code": error_code,
        "message": message,
        "next_action": normalized_action,
    }
    if http_status is not None:
        diagnostics["http_status"] = http_status
    if internal_diagnostics:
        diagnostics["internal_diagnostics"] = internal_diagnostics
    _log_fetch_diagnostics(diagnostics)
    # hint 中嵌入 next_action 标签，供 LLM 解析
    hint_text = f"[{normalized_action}] {hint}"
    raise ToolBusinessError(
        code=error_code,
        message=message,
        hint=hint_text,
        url=url,
        next_action=normalized_action,
        http_status=http_status,
        internal_diagnostics=internal_diagnostics or {},
    )


def _parse_retry_after_seconds(response_headers: Optional[dict[str, str]]) -> Optional[int]:
    """解析 Retry-After 头。

    Args:
        response_headers: 响应头。

    Returns:
        可解析时返回秒数，否则返回 ``None``。

    Raises:
        无。
    """

    if not response_headers:
        return None
    raw = response_headers.get("retry-after") or response_headers.get("Retry-After")
    if raw is None:
        return None
    try:
        value = int(str(raw).strip())
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _log_fetch_diagnostics(payload: dict[str, Any]) -> None:
    """输出网页抓取诊断日志。

    Args:
        payload: 诊断信息。

    Returns:
        无。

    Raises:
        无。
    """

    Log.debug(f"fetch_web_page diagnostics={payload}", module=MODULE)


@dataclass(frozen=True)
class WebProviderConfig:
    """联网检索 Provider 配置。

    Args:
        provider: provider 名称。

    Returns:
        无。

    Raises:
        无。
    """

    provider: str


def register_web_tools(
    registry: ToolRegistry,
    *,
    provider: str = "auto",
    request_timeout_seconds: float = 12.0,
    max_search_results: int = 20,
    fetch_truncate_chars: int = 80000,
    allow_private_network_url: bool = False,
    playwright_channel: str | None = "chrome",
    playwright_storage_state_dir: str = "",
    timeout_budget: float | None = None,
) -> None:
    """向工具注册表注册联网检索工具。

    Args:
        registry: 工具注册表实例。
        provider: Provider 选择策略，支持 ``auto``、``tavily``、``serper``、``duckduckgo``。
        request_timeout_seconds: HTTP 请求超时秒数（搜索与网页下载共用）。
        max_search_results: search_web 返回结果数量上限。
        fetch_truncate_chars: fetch_web_page 内容截断字符上限。
        allow_private_network_url: 是否允许访问内网/本地网络 URL。
        playwright_channel: 浏览器回退使用的 Chromium channel；空字符串表示不指定。
        playwright_storage_state_dir: 浏览器回退可选 storage state 目录；目录内按 host 自动查找 `<host>.json`。
        timeout_budget: Runner 为单次 tool call 提供的预算秒数。

    Returns:
        无。

    Raises:
        ValueError: 当 provider 非法时抛出。
    """
    name_search, func_search, schema_search = _create_search_web_tool(
        registry,
        provider=provider,
        request_timeout_seconds=request_timeout_seconds,
        max_search_results=max_search_results,
        allow_private_network_url=allow_private_network_url,
        timeout_budget=timeout_budget,
    )
    registry.register(name_search, func_search, schema_search)
    name_fetch, func_fetch, schema_fetch = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=request_timeout_seconds,
        fetch_truncate_chars=fetch_truncate_chars,
        allow_private_network_url=allow_private_network_url,
        playwright_channel=playwright_channel,
        playwright_storage_state_dir=playwright_storage_state_dir,
        timeout_budget=timeout_budget,
    )
    registry.register(name_fetch, func_fetch, schema_fetch)
    Log.verbose(f"已注册 2 个联网工具 provider={provider}", module=MODULE)


def _create_search_web_tool(
    registry: ToolRegistry,
    *,
    provider: str,
    request_timeout_seconds: float,
    max_search_results: int,
    allow_private_network_url: bool = False,
    timeout_budget: float | None = None,
) -> tuple[str, Any, Any]:
    """创建 `search_web` 工具。

    Args:
        registry: 工具注册表实例。
        provider: Provider 策略。
        request_timeout_seconds: HTTP 请求超时秒数（闭包传入）。
        max_search_results: 结果数量上限（闭包传入，同时影响 schema 约束）。
        allow_private_network_url: 是否允许访问内网/本地网络 URL。
        timeout_budget: Runner 为单次 tool call 提供的预算秒数（闭包传入）。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: 当 provider 非法时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索关键词。直接写你最自然的查询。",
            },
            "domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选域名限制。只在你明确要收窄来源时填写。",
            },
            "recency_days": {
                "type": "integer",
                "minimum": 0,
                "description": "可选最近天数限制。只在你明确要限制时效时填写。",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_search_results,
                "description": f"返回结果上限。只在你明确需要更少结果时调整；最大值 {max_search_results}。",
            },
        },
        "required": ["query"],
    }

    @tool(
        registry,
        name="search_web",
        description=(
            "搜索公开网页来源。"
        ),
        parameters=parameters,
        tags={"web"},
        display_name="联网搜索",
        summary_params=["query"],
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="list_items",
            limits={"max_items": 10},
            target_field="results",
        ),
    )
    def search_web(
        query: str,
        domains: Optional[list[str]] = None,
        recency_days: Optional[int] = None,
        max_results: int = 8,
    ) -> SearchWebOutput:
        """联网检索公开网页。

        Args:
            query: 检索关键词。
            domains: 可选域名过滤。
            recency_days: 可选最近天数过滤。
            max_results: 返回结果上限。

        Returns:
            检索结果字典。

        Raises:
            ValueError: 当参数非法时抛出。
            RuntimeError: 当所有 provider 均失败时抛出。
        """

        return search_public_web(
            query=query,
            domains=domains,
            recency_days=recency_days,
            max_results=max_results,
            max_search_results=max_search_results,
            provider=provider,
            request_timeout_seconds=request_timeout_seconds,
            timeout_budget=timeout_budget,
            deadline_monotonic=_compute_deadline_monotonic(timeout_budget),
            allow_private_network_url=allow_private_network_url,
            is_safe_public_url=_is_safe_public_url,
            normalize_whitespace=_normalize_whitespace,
            resolve_timeout_budget=_resolve_timeout_budget,
        )

    return _build_registered_tool_parts(search_web)


def _create_fetch_web_page_tool(
    registry: ToolRegistry,
    *,
    request_timeout_seconds: float,
    fetch_truncate_chars: int,
    allow_private_network_url: bool = False,
    playwright_channel: str | None = "chrome",
    playwright_storage_state_dir: str = "",
    timeout_budget: float | None = None,
) -> tuple[str, Any, Any]:
    """创建 `fetch_web_page` 工具。

    Args:
        registry: 工具注册表实例。
        request_timeout_seconds: HTTP 下载超时秒数（闭包传入）。
        fetch_truncate_chars: 内容截断字符上限（闭包传入）。
        allow_private_network_url: 是否允许访问内网/本地网络 URL。
        playwright_channel: 浏览器回退使用的 Chromium channel（闭包传入）。
        playwright_storage_state_dir: 浏览器回退可选 storage state 目录（闭包传入，目录内按 host 自动查找 `<host>.json`）。
        timeout_budget: Runner 为单次 tool call 提供的预算秒数（闭包传入）。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        无。
    """

    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string"
            },
        },
        "required": ["url"],
    }

    @tool(
        registry,
        name="fetch_web_page",
        description=(
            "抓取网页正文并转成 Markdown。失败时先看 hint 和 next_action，再决定重试、换来源或忽略当前网页。"
        ),
        parameters=parameters,
        tags={"web"},
        execution_context_param_name="execution_context",
        display_name="抓取网页",
        summary_params=["url"],
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="text_chars",
            limits={"max_chars": fetch_truncate_chars},
            target_field="content",
        ),
    )
    def fetch_web_page(
        url: str,
        execution_context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """抓取网页正文并转换为低噪音 Markdown。

        Args:
            url: 网页链接。
            execution_context: 当前工具调用执行上下文。

        Returns:
            抓取结果字典（不含 ``ok`` 字段，由 ToolRegistry 包装信封）。

            成功时包含 ``url/final_url/title/content/fetch_backend``。

        Raises:
            ToolBusinessError: 当抓取失败时抛出（含 error_code/message/hint）。
        """

        cancellation_token = _resolve_execution_cancellation_token(execution_context)
        _raise_if_tool_cancelled(cancellation_token)

        if not _is_safe_public_url(url, allow_private_network_url=allow_private_network_url):
            _raise_fetch_failure(
                url=url,
                error_code="permission_denied",
                message=f"URL is blocked by fetch safety policy: {url}",
                hint=build_hint(REASON_BLOCKED_BY_SITE_POLICY),
                next_action=NEXT_ACTION_CHANGE_SOURCE,
                internal_diagnostics={
                    "blocked_by_safety_policy": True,
                    "input_url": url,
                },
            )

        normalized_url = _normalize_url_for_http(url)
        deadline_monotonic = _compute_deadline_monotonic(timeout_budget)
        playwright_storage_state_path = _resolve_playwright_storage_state_path(
            url=normalized_url,
            playwright_storage_state_dir=playwright_storage_state_dir,
        )
        base_session = _get_web_session()
        session, should_close_session = _prepare_call_session(
            base_session,
            timeout_budget=timeout_budget,
        )
        applied_storage_state_cookie_count = _apply_storage_state_cookies_to_session(
            session,
            storage_state_path=playwright_storage_state_path,
        )
        headers = _build_fetch_headers(normalized_url)
        headers["Referer"] = _build_referer(normalized_url)
        warmup: dict[str, Any] = {"attempted": False}
        content_type_probe: dict[str, Any] = {"attempted": False, "ok": False}
        fetch_result: dict[str, Any] | None = None
        playwright_fallback_kwargs: dict[str, Any] = {
            "timeout_seconds": request_timeout_seconds,
            "headers": headers,
            "timeout_budget": timeout_budget,
            "deadline_monotonic": deadline_monotonic,
            "playwright_channel": playwright_channel,
            "playwright_storage_state_path": playwright_storage_state_path,
        }
        if cancellation_token is not None:
            playwright_fallback_kwargs["cancellation_token"] = cancellation_token
        try:
            _raise_if_tool_cancelled(cancellation_token)
            warmup_kwargs: dict[str, Any] = {
                "url": normalized_url,
                "timeout_seconds": request_timeout_seconds,
                "headers": headers,
                "timeout_budget": timeout_budget,
                "deadline_monotonic": deadline_monotonic,
            }
            if cancellation_token is not None:
                warmup_kwargs["cancellation_token"] = cancellation_token
            warmup = _warmup_domain(
                session,
                **warmup_kwargs,
            )
            if _should_escalate_stage_result_to_browser(warmup):
                browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                if browser_result is not None:
                    return browser_result
            _raise_if_tool_cancelled(cancellation_token)
            probe_kwargs: dict[str, Any] = {
                "url": normalized_url,
                "timeout_seconds": request_timeout_seconds,
                "headers": headers,
                "timeout_budget": timeout_budget,
                "deadline_monotonic": deadline_monotonic,
            }
            if cancellation_token is not None:
                probe_kwargs["cancellation_token"] = cancellation_token
            content_type_probe = _probe_content_type(
                session,
                **probe_kwargs,
            )
            if _should_escalate_stage_result_to_browser(content_type_probe):
                browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                if browser_result is not None:
                    return browser_result
            _raise_if_tool_cancelled(cancellation_token)
            fetch_kwargs: dict[str, Any] = {
                "timeout_seconds": request_timeout_seconds,
                "session": session,
                "headers": headers,
                "content_type_probe": content_type_probe,
                "timeout_budget": timeout_budget,
                "deadline_monotonic": deadline_monotonic,
            }
            if cancellation_token is not None:
                fetch_kwargs["cancellation_token"] = cancellation_token
            fetch_result = _fetch_and_convert_content(
                normalized_url,
                **fetch_kwargs,
            )
        except requests.TooManyRedirects as exc:
            response = getattr(exc, "response", None)
            _raise_fetch_failure(
                url=url,
                error_code="too_many_redirects",
                message="Redirect chain too long; cannot reliably fetch this page",
                http_status=response.status_code if response is not None else None,
                hint=build_hint(REASON_REDIRECT_CHAIN_TOO_LONG),
                next_action=NEXT_ACTION_CHANGE_SOURCE,
                internal_diagnostics={
                    "final_url": response.url if response is not None else url,
                    "response_headers": _sanitize_response_headers(response.headers if response is not None else {}),
                    "response_excerpt": _extract_response_snippet(response),
                },
            )
        except requests.Timeout as exc:
            browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
            if browser_result is not None:
                return browser_result
            _raise_fetch_failure(
                url=url,
                error_code="request_timeout",
                message=f"Request timed out: {exc}",
                hint=build_hint(REASON_REQUEST_TIMEOUT),
                next_action=NEXT_ACTION_RETRY,
                internal_diagnostics={
                    "final_url": url,
                    "warmup": warmup,
                    "content_type_probe": content_type_probe,
                    "applied_storage_state_cookie_count": applied_storage_state_cookie_count,
                },
            )
        except requests.RequestException as exc:
            response = getattr(exc, "response", None)
            http_status = response.status_code if response is not None else None
            challenge_hint = ""
            error_code = "http_error"
            next_action = (
                NEXT_ACTION_RETRY
                if http_status in {429, 500, 502, 503, 504} or http_status is None
                else NEXT_ACTION_CHANGE_SOURCE
            )
            if _is_timeout_like_request_exception(exc):
                browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                if browser_result is not None:
                    return browser_result
                _raise_fetch_failure(
                    url=url,
                    error_code="request_timeout",
                    message=str(exc),
                    hint=build_hint(REASON_REQUEST_TIMEOUT),
                    next_action=NEXT_ACTION_RETRY,
                    internal_diagnostics={
                        "final_url": response.url if response is not None else url,
                        "warmup": warmup,
                        "content_type_probe": content_type_probe,
                        "applied_storage_state_cookie_count": applied_storage_state_cookie_count,
                        "response_headers": _sanitize_response_headers(response.headers if response is not None else {}),
                        "response_excerpt": _extract_response_snippet(response),
                    },
                )
            if _is_ssl_like_request_exception(exc):
                browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                if browser_result is not None:
                    return browser_result
                _raise_fetch_failure(
                    url=url,
                    error_code="ssl_error",
                    message=f"SSL/TLS 握手失败: {exc}",
                    hint=build_hint(REASON_HTTP_ERROR),
                    next_action=NEXT_ACTION_CHANGE_SOURCE,
                    internal_diagnostics={
                        "final_url": response.url if response is not None else url,
                        "warmup": warmup,
                        "content_type_probe": content_type_probe,
                        "applied_storage_state_cookie_count": applied_storage_state_cookie_count,
                        "response_headers": _sanitize_response_headers(response.headers if response is not None else {}),
                        "response_excerpt": _extract_response_snippet(response),
                        "exception_types": [type(item).__name__ for item in _iter_exception_chain(exc)],
                    },
                )
            challenge = _detect_bot_challenge(
                response=response,
                content_text=_extract_response_snippet(response),
            )
            if challenge.challenge_detected and http_status in {401, 403, 429, 503}:
                browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                if browser_result is not None:
                    return browser_result
                _raise_fetch_failure(
                    url=url,
                    error_code="blocked",
                    message="Page appears to be a bot challenge page or access gate; fetched content is unusable.",
                    http_status=http_status,
                    hint=build_hint(REASON_BLOCKED_BY_SITE_POLICY),
                    next_action=NEXT_ACTION_CHANGE_SOURCE,
                    internal_diagnostics={
                        "final_url": response.url if response is not None else url,
                        "warmup": warmup,
                        "content_type_probe": content_type_probe,
                        "applied_storage_state_cookie_count": applied_storage_state_cookie_count,
                        "challenge_signals": list(challenge.challenge_signals),
                        "response_headers": _sanitize_response_headers(response.headers if response is not None else {}),
                        "response_excerpt": _extract_response_snippet(response),
                    },
                )
            if _should_escalate_http_status_to_browser(http_status):
                browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                if browser_result is not None:
                    return browser_result
            if http_status == 403:
                browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                if browser_result is not None:
                    return browser_result
                challenge_hint = "Target site may have anti-bot or access policies; try a different source."
                next_action = NEXT_ACTION_CHANGE_SOURCE
                error_code = "blocked"
            _raise_fetch_failure(
                url=url,
                error_code=error_code,
                message=str(exc),
                http_status=http_status,
                hint=challenge_hint or build_hint(REASON_HTTP_ERROR),
                next_action=next_action,
                internal_diagnostics={
                    "final_url": response.url if response is not None else url,
                    "warmup": warmup,
                    "content_type_probe": content_type_probe,
                    "applied_storage_state_cookie_count": applied_storage_state_cookie_count,
                    "response_headers": _sanitize_response_headers(response.headers if response is not None else {}),
                    "response_excerpt": _extract_response_snippet(response),
                },
            )
        except RuntimeError as exc:
            internal_diagnostics: dict[str, Any] | None = None
            challenge_context: _FetchContentRuntimeContext | None = None
            challenge: BotChallengeDetectionResult | None = None
            pipeline_error: HtmlPipelineStageError | None = None
            conversion_failure_reason = ""
            if isinstance(exc, _FetchContentConversionError):
                challenge_context = exc.response_context
                conversion_failure_reason = exc.failure_reason
                if isinstance(exc.original_error, HtmlPipelineStageError):
                    pipeline_error = exc.original_error
            elif isinstance(exc, HtmlPipelineStageError):
                pipeline_error = exc

            if conversion_failure_reason in {"unsupported_content_encoding", "meta_refresh_requires_browser"}:
                browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                if browser_result is not None:
                    return browser_result

            if challenge_context is not None:
                challenge = _detect_bot_challenge(
                    response=None,
                    response_headers=challenge_context.response_headers,
                    http_status=challenge_context.http_status,
                    content_text=challenge_context.raw_content_text or challenge_context.response_excerpt,
                )
                if challenge.challenge_detected:
                    browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                    if browser_result is not None:
                        return browser_result

            if _should_escalate_pipeline_failure_to_browser(
                pipeline_error=pipeline_error,
                response_context=challenge_context,
            ):
                browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                if browser_result is not None:
                    return browser_result

            if _should_escalate_conversion_failure_to_browser(
                error_message=str(exc),
                response_context=challenge_context,
            ):
                browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
                if browser_result is not None:
                    return browser_result

            if pipeline_error is not None or challenge_context is not None:
                internal_diagnostics = {}
                if pipeline_error is not None:
                    internal_diagnostics.update(
                        {
                            "pipeline_stage": pipeline_error.stage,
                            "extractor_source": pipeline_error.extractor_source,
                            "quality_flags": list(pipeline_error.quality_flags),
                            "content_stats": pipeline_error.content_stats,
                        }
                    )
                if challenge_context is not None:
                    internal_diagnostics.update(
                        {
                            "final_url": challenge_context.final_url or url,
                            "http_status": challenge_context.http_status,
                            "applied_storage_state_cookie_count": applied_storage_state_cookie_count,
                            "response_headers": _sanitize_response_headers(challenge_context.response_headers),
                            "response_excerpt": challenge_context.response_excerpt,
                        }
                    )
                if conversion_failure_reason:
                    internal_diagnostics["conversion_failure_reason"] = conversion_failure_reason
                if challenge is not None and challenge.challenge_signals:
                    internal_diagnostics["challenge_signals"] = list(challenge.challenge_signals)
            _raise_fetch_failure(
                url=url,
                error_code="blocked" if challenge is not None and challenge.challenge_detected else "content_conversion_failed",
                message=str(exc),
                hint=(
                    build_hint(REASON_BLOCKED_BY_SITE_POLICY)
                    if challenge is not None and challenge.challenge_detected
                    else build_hint(REASON_CONTENT_CONVERSION_FAILED)
                ),
                next_action=NEXT_ACTION_CHANGE_SOURCE,
                http_status=challenge_context.http_status if challenge_context is not None else None,
                internal_diagnostics=internal_diagnostics,
            )
        finally:
            if should_close_session:
                session.close()

        if fetch_result is None:
            raise RuntimeError("网页抓取流程异常结束，未获得抓取结果")

        challenge = _detect_bot_challenge(
            response=fetch_result.get("response"),
            content_text=fetch_result.get("content", ""),
        )
        if challenge.challenge_detected:
            browser_result = _try_playwright_fallback(url=url, **playwright_fallback_kwargs)
            if browser_result is not None:
                return browser_result
            _raise_fetch_failure(
                url=url,
                error_code="blocked",
                message="Page appears to be a bot challenge page; fetched content is unusable.",
                http_status=fetch_result.get("http_status"),
                hint=build_hint(REASON_BLOCKED_BY_SITE_POLICY),
                next_action=NEXT_ACTION_CHANGE_SOURCE,
                internal_diagnostics={
                    "final_url": fetch_result.get("final_url", url),
                    "redirect_hops": fetch_result.get("redirect_hops"),
                    "challenge_signals": list(challenge.challenge_signals),
                    "response_headers": _sanitize_response_headers(fetch_result.get("response_headers", {})),
                    "response_excerpt": fetch_result.get("response_excerpt", ""),
                },
            )

        content = fetch_result.get("content", "")
        if len(content.strip()) < _EMPTY_CONTENT_MIN_CHARS:
            _raise_fetch_failure(
                url=url,
                error_code="empty_content",
                message="Page body is empty or too short to be useful.",
                http_status=fetch_result.get("http_status"),
                hint=build_hint(REASON_EMPTY_CONTENT),
                next_action=NEXT_ACTION_CONTINUE_WITHOUT_WEB,
                internal_diagnostics={
                    "final_url": fetch_result.get("final_url", url),
                    "redirect_hops": fetch_result.get("redirect_hops"),
                    "response_headers": _sanitize_response_headers(fetch_result.get("response_headers", {})),
                    "response_excerpt": fetch_result.get("response_excerpt", ""),
                },
            )

        success = {
            "url": url,
            "final_url": fetch_result.get("final_url", url),
            "title": fetch_result.get("title", ""),
            "content": content,
            "fetch_backend": "requests",
        }
        _log_fetch_diagnostics(
            {
                **success,
                "extraction_source": fetch_result.get("extraction_source", ""),
                "renderer_source": fetch_result.get("renderer_source", ""),
                "normalization_applied": fetch_result.get("normalization_applied", False),
                "internal_diagnostics": {
                    "final_url": fetch_result.get("final_url", url),
                    "http_status": fetch_result.get("http_status"),
                    "redirect_hops": fetch_result.get("redirect_hops"),
                    "fetch_backend": "requests",
                    "extraction_source": fetch_result.get("extraction_source", ""),
                    "renderer_source": fetch_result.get("renderer_source", ""),
                    "normalization_applied": fetch_result.get("normalization_applied", False),
                    "quality_flags": fetch_result.get("quality_flags", []),
                    "content_stats": fetch_result.get("content_stats", {}),
                    "content_type_probe": content_type_probe,
                    "warmup": warmup,
                    "response_headers": _sanitize_response_headers(fetch_result.get("response_headers", {})),
                    "response_excerpt": fetch_result.get("response_excerpt", ""),
                },
            }
        )
        return success

    return _build_registered_tool_parts(fetch_web_page)



def _docling_convert_to_markdown(raw_bytes: bytes, stream_name: str) -> tuple[str, str, str]:
    """使用 Docling 将非 HTML 原始字节转换为 Markdown。

    Args:
        raw_bytes: 页面原始内容字节。
        stream_name: 流名称，决定 Docling 解析模式（如 ``page.pdf``）。

    Returns:
        ``(title, markdown, extraction_source)`` 三元组。

    Raises:
        RuntimeError: Docling 未安装或转换失败时抛出。
    """
    title, markdown, extraction_source = _web_fetch_orchestrator._docling_convert_to_markdown(raw_bytes, stream_name)
    if not title:
        title = _extract_first_markdown_heading(markdown)
    return title, markdown, extraction_source


def _should_route_response_to_html_pipeline(
    *,
    url: str,
    content_type: str,
    response_text: str,
    response_content: bytes,
) -> bool:
    """判断响应是否应进入 HTML 四段式流水线。"""

    return _web_fetch_orchestrator._should_route_response_to_html_pipeline(
        url=url,
        content_type=content_type,
        response_text=response_text,
        response_content=response_content,
    )


def _infer_docling_stream_name(*, url: str, content_type: str) -> str:
    """为 Docling 推断更稳定的输入流名称。

    Args:
        url: 当前响应 URL。
        content_type: 已归一化的小写 Content-Type。

    Returns:
        供 Docling 使用的伪文件名。

    Raises:
        无。
    """

    return _web_fetch_orchestrator._infer_docling_stream_name(url=url, content_type=content_type)


def _fetch_and_convert_content(
    url: str,
    *,
    timeout_seconds: float,
    session: Optional[requests.Session] = None,
    headers: Optional[dict[str, str]] = None,
    content_type_probe: Optional[dict[str, Any]] = None,
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """先下载页面内容，再按内容类型转换为低噪音 Markdown。"""

    return _web_fetch_orchestrator._fetch_and_convert_content(
        url,
        timeout_seconds=timeout_seconds,
        resolve_timeout_budget=_resolve_timeout_budget,
        normalize_url_for_http=_normalize_url_for_http,
        build_referer=_build_referer,
        convert_html=convert_html_to_llm_markdown,
        convert_non_html=_docling_convert_to_markdown,
        session=session,
        get_web_session=_get_web_session,
        headers=headers,
        build_fetch_headers=_build_fetch_headers,
        content_type_probe=content_type_probe,
        timeout_budget=timeout_budget,
        deadline_monotonic=deadline_monotonic,
        cancellation_token=cancellation_token,
    )


def _close_playwright_browser() -> None:
    """关闭 Playwright Browser 和 Playwright 运行时单例（atexit 注册）。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """

    _web_playwright_backend._close_playwright_browser()


def _normalize_playwright_channel(playwright_channel: str | None) -> str | None:
    """标准化 Playwright channel 配置。

    Args:
        playwright_channel: 原始 channel 配置。

    Returns:
        规整后的 channel；空字符串时返回 `None`。

    Raises:
        无。
    """

    return _web_playwright_backend._normalize_playwright_channel(playwright_channel)


def _normalize_playwright_storage_state_dir(path_value: str | None) -> str | None:
    """标准化 Playwright storage state 目录路径。

    Args:
        path_value: 原始路径配置。

    Returns:
        目录路径字符串；未配置时返回 `None`。

    Raises:
        无。
    """

    return _web_playwright_backend._normalize_playwright_storage_state_dir(path_value)


def _resolve_playwright_storage_state_path(
    *,
    url: str,
    playwright_storage_state_dir: str | None,
) -> str:
    """按 host 解析 Playwright storage state 文件路径。

    Args:
        url: 当前抓取 URL。
        playwright_storage_state_dir: storage state 目录配置。

    Returns:
        命中的 storage state 文件绝对路径；未命中时返回空字符串。

    Raises:
        无。
    """

    return _web_playwright_backend._resolve_playwright_storage_state_path(
        url=url,
        playwright_storage_state_dir=playwright_storage_state_dir,
    )


def _get_playwright_browser(
    *,
    playwright_channel: str | None = None,
    headless: bool = True,
) -> Optional[Any]:
    """获取（或懒初始化）全局 Playwright Browser 单例。

    使用 double-checked locking 保证线程安全。若 playwright 未安装或启动失败，
    返回 None 而不抛异常。

    Args:
        无。

    Returns:
        playwright.sync_api.Browser 单例，或 None（不可用时）。

    Raises:
        无。
    """

    return _web_playwright_backend._get_playwright_browser(
        playwright_channel=playwright_channel,
        headless=headless,
    )


def _route_handler_abort_resources(route: Any) -> None:
    """Playwright 路由拦截器：中止图片/字体/媒体请求，放行其余资源。

    降低页面渲染流量，加快加载速度。

    Args:
        route: playwright.sync_api.Route 对象。

    Returns:
        无。

    Raises:
        无。
    """

    _web_playwright_backend._route_handler_abort_resources(route)


def _maybe_warmup_playwright_page(
    *,
    page: Any,
    url: str,
    deadline_monotonic: float,
) -> None:
    """在浏览器回退前先做一次同域首页预热。

    部分站点会在首页下发 Cookie、地域态或轻量挑战票据。requests 路径
    已有 warmup，这里为 Playwright 路径补齐同样的机械预热，减少
    “浏览器上下文过于冷启动”带来的差异。

    Args:
        page: Playwright Page。
        url: 目标 URL。
        deadline_monotonic: 本次浏览器抓取总预算 deadline。

    Returns:
        无。

    Raises:
        无。
    """

    _web_playwright_backend._maybe_warmup_playwright_page(
        page=page,
        url=url,
        deadline_monotonic=deadline_monotonic,
        build_domain_home_url=_build_domain_home_url,
        normalize_url_for_http=_normalize_url_for_http,
        time_monotonic=time.monotonic,
    )


def _settle_playwright_page(
    *,
    page: Any,
    deadline_monotonic: float,
) -> None:
    """在浏览器导航后做有上限的页面稳定化等待。

    不直接无限等待 `networkidle`，而是用一组有上限的小等待，兼顾 SPA
    首屏渲染与长连接页面，尽量逼近人工浏览器“打开后停留片刻”的效果。

    Args:
        page: Playwright Page。
        deadline_monotonic: 本次浏览器抓取总预算 deadline。

    Returns:
        无。

    Raises:
        无。
    """

    _web_playwright_backend._settle_playwright_page(
        page=page,
        deadline_monotonic=deadline_monotonic,
        time_monotonic=time.monotonic,
    )


def _get_remaining_playwright_timeout_ms(deadline_monotonic: float) -> int:
    """计算 Playwright 当前阶段还可使用的剩余超时。

    Args:
        deadline_monotonic: 本次浏览器抓取总预算 deadline。

    Returns:
        剩余可用毫秒数；预算已耗尽时返回 0。

    Raises:
        无。
    """

    return _web_playwright_backend._get_remaining_playwright_timeout_ms(
        deadline_monotonic,
        time_monotonic=time.monotonic,
    )


def _require_playwright_timeout_ms(deadline_monotonic: float) -> int:
    """为必需的 Playwright 阶段解析剩余超时。

    Args:
        deadline_monotonic: 本次浏览器抓取总预算 deadline。

    Returns:
        当前阶段可用的毫秒超时。

    Raises:
        RuntimeError: 当浏览器总预算已耗尽时抛出。
    """

    return _web_playwright_backend._require_playwright_timeout_ms(
        deadline_monotonic,
        time_monotonic=time.monotonic,
    )


def _playwright_sync_worker(
    *,
    url: str,
    timeout_seconds: float,
    headers: Optional[dict[str, str]] = None,
    playwright_channel: str | None = None,
    playwright_storage_state_path: str = "",
) -> dict[str, Any]:
    """在独立线程中执行完整的 Playwright 同步抓取流程。

    不得在 asyncio event loop 所在线程直接调用；须通过 ThreadPoolExecutor 提交。

    流程：获取 Browser 单例 → 创建隔离 BrowserContext → stealth_sync → 路由拦截
    → page.goto → 检查 content-type → page.content() → Docling 转 Markdown。

    Args:
        url: 已通过安全校验的网页链接。
        timeout_seconds: 本次浏览器回退总预算秒数。
        headers: 可选额外请求头（当前仍以浏览器默认导航画像为准，不直接覆写 Context headers）。
        playwright_channel: 浏览器回退使用的 Chromium channel。
        playwright_storage_state_path: 浏览器回退可选 storage state 文件路径。

    Returns:
        成功时返回含 ``ok=True`` 的结果字典；失败时抛出异常由调用方处理。

    Raises:
        RuntimeError: playwright 未安装、Browser 不可用、页面加载失败、内容转换失败等。
    """

    return _web_playwright_backend._playwright_sync_worker(
        url=url,
        timeout_seconds=timeout_seconds,
        headers=headers,
        playwright_channel=playwright_channel,
        playwright_storage_state_path=playwright_storage_state_path,
        get_playwright_browser=_get_playwright_browser,
        build_domain_home_url=_build_domain_home_url,
        normalize_url_for_http=_normalize_url_for_http,
        sanitize_response_headers=_sanitize_response_headers,
        build_text_excerpt=_build_text_excerpt,
        convert_html_to_markdown=convert_html_to_llm_markdown,
        time_monotonic=time.monotonic,
    )


def _fetch_and_convert_with_playwright(
    *,
    url: str,
    timeout_seconds: float,
    headers: Optional[dict[str, str]] = None,
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    playwright_channel: str | None = None,
    playwright_storage_state_path: str = "",
    cancellation_token: CancellationToken | None = None,
) -> dict[str, Any]:
    """使用 Playwright 执行浏览器抓取并转换为 Markdown。

    架构：优先在独立子进程中执行同步 worker，以便在超时或取消时硬终止。
    Browser 为 worker 进程内单例，Context 为每次请求独立创建并在完成后关闭。

    Args:
        url: 已通过安全校验的网页链接。
        timeout_seconds: 浏览器回退总预算秒数。
        headers: 可选请求头（当前透传给 worker，供将来扩展使用）。
        timeout_budget: Runner 注入的单次 tool call 总预算。
        deadline_monotonic: 当前工具调用的单调时钟 deadline。
        playwright_channel: 浏览器回退使用的 Chromium channel。
        playwright_storage_state_path: 浏览器回退可选 storage state 文件路径。
        cancellation_token: 当前工具调用的取消令牌。

    Returns:
        成功时：``{ok: True, title, content, final_url}``，结构与 docling 路径一致。
        失败时：``{ok: False, availability, reason}`` 或超时字典。

    Raises:
        无（所有异常在函数内捕获并转换为失败字典）。
    """

    return _web_playwright_backend._fetch_and_convert_with_playwright(
        url=url,
        timeout_seconds=timeout_seconds,
        headers=headers,
        timeout_budget=timeout_budget,
        deadline_monotonic=deadline_monotonic,
        playwright_channel=playwright_channel,
        playwright_storage_state_path=playwright_storage_state_path,
        cancellation_token=cancellation_token,
        resolve_timeout_budget=_resolve_timeout_budget,
        playwright_sync_worker=_playwright_sync_worker,
        detect_bot_challenge=_detect_bot_challenge,
    )


def _extract_first_markdown_heading(markdown: str) -> str:
    """从 Markdown 文本中提取第一个标题行的文本。

    Args:
        markdown: Markdown 格式文本。

    Returns:
        标题文本；无标题时返回空字符串。

    Raises:
        无。
    """

    for line in markdown.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def _build_fetch_headers(url: str) -> dict[str, str]:
    """构建网页抓取请求头。

    Args:
        url: 目标网页 URL。

    Returns:
        请求头字典。

    Raises:
        无。
    """

    if _is_sec_host(url):
        sec_user_agent = (os.environ.get(SEC_USER_AGENT_ENV) or _DEFAULT_SEC_USER_AGENT).strip()
        return {
            "User-Agent": sec_user_agent or _DEFAULT_SEC_USER_AGENT,
            "Accept": _DEFAULT_ACCEPT,
            "Accept-Language": _DEFAULT_ACCEPT_LANGUAGE,
            "Accept-Encoding": _build_accept_encoding_value(),
            "Connection": "keep-alive",
        }

    return {
        "User-Agent": _DEFAULT_BROWSER_USER_AGENT,
        "Accept": _DEFAULT_ACCEPT,
        "Accept-Language": _DEFAULT_ACCEPT_LANGUAGE,
        "Accept-Encoding": _build_accept_encoding_value(),
        "Connection": "keep-alive",
        # --- 现代 Chrome 标准 headers，缺失是典型爬虫特征 ---
        "Sec-Ch-Ua": _DEFAULT_SEC_CH_UA,
        "Sec-Ch-Ua-Mobile": _DEFAULT_SEC_CH_UA_MOBILE,
        "Sec-Ch-Ua-Platform": _DEFAULT_SEC_CH_UA_PLATFORM,
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
    }


def _is_sec_host(url: str) -> bool:
    """判断 URL 是否指向 SEC 域名。

    Args:
        url: 目标 URL。

    Returns:
        是 SEC 域名返回 ``True``，否则返回 ``False``。

    Raises:
        无。
    """

    host = (urlparse(url).hostname or "").lower().strip()
    return bool(host) and (host == "sec.gov" or host.endswith(".sec.gov") or host == "data.sec.gov")


def _normalize_whitespace(text: str) -> str:
    """规整文本中的空白字符。

    Args:
        text: 原始文本。

    Returns:
        规整后的文本。

    Raises:
        无。
    """

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join([line for line in lines if line])


def _is_public_ip(ip_text: str) -> bool:
    """判断 IP 是否属于可访问公网地址。

    Args:
        ip_text: 待判断 IP 字符串。

    Returns:
        公网地址返回 ``True``，内网/保留地址返回 ``False``。

    Raises:
        ValueError: 当 IP 文本非法时抛出。
    """

    ip_value = ipaddress.ip_address(ip_text)
    return not (
        ip_value.is_private
        or ip_value.is_loopback
        or ip_value.is_link_local
        or ip_value.is_reserved
        or ip_value.is_multicast
        or ip_value.is_unspecified
    )


def _is_fake_ip(ip_text: str) -> bool:
    """判断 IP 是否落在常见 fake-ip 保留网段。

    Args:
        ip_text: 待判断 IP 字符串。

    Returns:
        命中 fake-ip 网段返回 ``True``，否则返回 ``False``。

    Raises:
        ValueError: 当 IP 文本非法时抛出。
    """

    ip_value = ipaddress.ip_address(ip_text)
    return any(ip_value in network for network in _FAKE_IP_NETWORKS)


def _looks_like_public_hostname(hostname: str) -> bool:
    """判断主机名是否形似公开互联网域名。

    设计意图：
    - fake-ip 场景下，公开域名会被本地 DNS 虚拟化到保留地址；
    - 这里仅在主机名本身明显不是本地域名时，才允许用 fake-ip 结果放行；
    - 避免把 ``localhost``、单标签主机名、``*.local`` 一类本地地址误判成公网地址。

    Args:
        hostname: 已归一化的小写主机名。

    Returns:
        形似公开域名返回 ``True``，否则返回 ``False``。

    Raises:
        无。
    """

    if not hostname or "." not in hostname:
        return False
    if hostname.endswith(".local") or hostname.endswith(".localhost") or hostname.endswith(".localdomain"):
        return False
    for pattern in _PRIVATE_HOST_PATTERNS:
        if hostname == pattern or hostname.startswith(pattern):
            return False
    return True


def _resolve_hostname_ips(hostname: str) -> set[str]:
    """解析域名并返回去重后的 IP 集合。

    Args:
        hostname: 域名（不含 scheme/path）。

    Returns:
        解析到的 IP 字符串集合；解析失败返回空集合。

    Raises:
        无。
    """

    try:
        infos = socket.getaddrinfo(
            hostname,
            None,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except OSError:
        return set()

    resolved: set[str] = set()
    for item in infos:
        sockaddr = item[4]
        if not sockaddr:
            continue
        ip_text = str(sockaddr[0]).strip()
        if ip_text:
            resolved.add(ip_text)
    return resolved


def _is_safe_public_url(url: str, *, allow_private_network_url: bool = False) -> bool:
    """校验 URL 是否为可访问目标地址。

    Args:
        url: 待校验链接。
        allow_private_network_url: 是否允许访问内网/本地网络 URL。

    Returns:
        安全可访问返回 ``True``，否则返回 ``False``。

    Raises:
        无。
    """

    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return False
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        return False

    if allow_private_network_url:
        return True

    for pattern in _PRIVATE_HOST_PATTERNS:
        if hostname == pattern or hostname.startswith(pattern):
            return False

    try:
        return _is_public_ip(hostname)
    except ValueError:
        if hostname.endswith(".local") or hostname.endswith(".localhost") or hostname.endswith(".localdomain"):
            return False
        resolved_ips = _resolve_hostname_ips(hostname)
        if not resolved_ips:
            return False
        if all(_is_public_ip(ip_text) for ip_text in resolved_ips):
            return True
        # fake-ip（例如 OpenClash）会把公开域名虚拟解析到 198.18.0.0/15。
        # 这里仅对“看起来像公开域名”的主机名放行，字面量 IP 与本地域名仍严格拒绝。
        if _looks_like_public_hostname(hostname) and all(_is_fake_ip(ip_text) for ip_text in resolved_ips):
            return True
        return False
