"""网页抓取编排辅助模块。

本模块聚合 requests 主路径中的 warmup、content-type probe、
HTML/Docling 路由与浏览器升级判定，避免这些编排细节继续膨胀在
``web_tools.py`` 中。
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection
from dataclasses import dataclass
from threading import Lock
from urllib.parse import urljoin, urlparse

import requests
from requests.structures import CaseInsensitiveDict
from dayu.contracts.cancellation import CancellationToken
from dayu.docling_runtime import (
    DoclingRuntimeInitializationError,
    run_docling_pdf_conversion,
)
from bs4 import BeautifulSoup

from dayu.engine.processors.html_pipeline import HtmlPipelineResult, HtmlPipelineStageError
from dayu.engine.processors.text_utils import infer_suffix_from_uri
from dayu.engine.tools.web_challenge_detection import detect_bot_challenge
from dayu.engine.tools.web_http_encoding import _decode_response_text, _find_unsupported_content_encodings

_WARMUP_TIMEOUT_SECONDS = 6.0
_RESPONSE_SNIPPET_MAX_CHARS = 500
_EMPTY_CONTENT_MIN_CHARS = 5
_MAX_META_REFRESH_HOPS = 3
_META_REFRESH_IMMEDIATE_MAX_SECONDS = 1.0
_PLAYWRIGHT_HTTP_ESCALATION_STATUSES = frozenset(
    {
        412,
        421,
        422,
        423,
        425,
        426,
        428,
        431,
        440,
        444,
        449,
        450,
        451,
        495,
        496,
        497,
        498,
        499,
        520,
        521,
        522,
        523,
        524,
        525,
        526,
        530,
    }
)
_WARMED_HOSTS_LOCK = Lock()


@dataclass(frozen=True)
class _FetchContentRuntimeContext:
    """抓取转换失败时保留的原始响应上下文。"""

    http_status: int | None
    final_url: str
    response_headers: dict[str, str]
    response_excerpt: str
    raw_content_text: str


@dataclass(frozen=True)
class _MetaRefreshDirective:
    """HTML meta refresh 指令。"""

    target_url: str
    raw_target: str
    delay_seconds: float | None
    raw_content: str


class _FetchContentConversionError(RuntimeError):
    """抓取转换阶段的包装异常。"""

    def __init__(
        self,
        message: str,
        *,
        response_context: _FetchContentRuntimeContext,
        original_error: RuntimeError,
        failure_reason: str = "",
    ) -> None:
        """初始化包装异常。

        Args:
            message: 对外暴露的错误信息。
            response_context: 原始响应上下文。
            original_error: 原始运行时异常。
            failure_reason: 供上层判定是否升级浏览器的失败类型。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(message)
        self.response_context = response_context
        self.original_error = original_error
        self.failure_reason = str(failure_reason or "").strip()

def _raise_if_cancelled(cancellation_token: CancellationToken | None) -> None:
    """在进入下一网络阶段前执行协作式取消检查。

    Args:
        cancellation_token: 当前工具调用的取消令牌。

    Returns:
        无。

    Raises:
        CancelledError: 调用已被取消时抛出。
    """

    if cancellation_token is not None:
        cancellation_token.raise_if_cancelled()


def _close_response_safely(response: requests.Response) -> None:
    """尽力关闭响应对象，兼容测试桩。

    Args:
        response: 任意响应对象。

    Returns:
        无。

    Raises:
        无。
    """

    close = getattr(response, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            return


def _extract_response_snippet(response: requests.Response | None) -> str:
    """提取响应文本前缀用于诊断。

    Args:
        response: HTTP 响应对象。

    Returns:
        限长后的文本前缀。

    Raises:
        无。
    """

    if response is None:
        return ""
    try:
        text = _decode_response_text(response).strip()
    except Exception:
        return ""
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", text)
    return compact[:_RESPONSE_SNIPPET_MAX_CHARS]


def _sanitize_response_headers(headers: CaseInsensitiveDict[str] | dict[str, str] | None) -> dict[str, str]:
    """筛选可用于分析的关键响应头。

    Args:
        headers: 原始响应头映射。

    Returns:
        过滤后的响应头字典。

    Raises:
        无。
    """

    if not headers:
        return {}
    selected_keys = (
        "content-type",
        "content-length",
        "server",
        "cf-ray",
        "x-datadome",
        "x-datadome-cid",
        "x-dd-b",
        "retry-after",
        "location",
        "set-cookie",
    )
    normalized: dict[str, str] = {}
    for key in selected_keys:
        value = headers.get(key) or headers.get(key.title())
        if value is None:
            continue
        text = str(value)
        if key == "set-cookie":
            cookie_names = [chunk.split("=", 1)[0].strip() for chunk in text.split(";") if "=" in chunk]
            text = ",".join(sorted(set(filter(None, cookie_names))))
        normalized[key] = text[:200]
    return normalized


def _build_fetch_content_runtime_context(response: requests.Response) -> _FetchContentRuntimeContext:
    """从响应对象构造抓取转换失败上下文。

    Args:
        response: 原始 HTTP 响应。

    Returns:
        供上层异常处理使用的响应上下文。

    Raises:
        无。
    """

    try:
        raw_content_text = _decode_response_text(response)
    except Exception:
        raw_content_text = ""
    return _FetchContentRuntimeContext(
        http_status=getattr(response, "status_code", None),
        final_url=str(getattr(response, "url", "") or ""),
        response_headers={str(key): str(value) for key, value in dict(getattr(response, "headers", {}) or {}).items()},
        response_excerpt=_extract_response_snippet(response),
        raw_content_text=raw_content_text,
    )


def _extract_html_response_text(response: requests.Response) -> str:
    """提取已确认走 HTML 管线的响应文本。

    Args:
        response: 原始 HTTP 响应。

    Returns:
        可供 HTML 抽取器消费的文本。

    Raises:
        _FetchContentConversionError: 当响应使用了当前运行时不支持的内容编码时抛出。
    """

    unsupported_encodings = _find_unsupported_content_encodings(getattr(response, "headers", {}))
    if unsupported_encodings:
        raise _FetchContentConversionError(
            f"HTML 响应使用当前运行时不支持的内容编码: {', '.join(unsupported_encodings)}",
            response_context=_build_fetch_content_runtime_context(response),
            original_error=RuntimeError("unsupported_content_encoding"),
            failure_reason="unsupported_content_encoding",
        )
    return _decode_response_text(response)


def _extract_meta_refresh_directive(
    html_text: str,
    *,
    base_url: str,
    normalize_url_for_http: Callable[[str], str],
) -> _MetaRefreshDirective | None:
    """从 HTML 中解析 meta refresh 指令。

    Args:
        html_text: HTML 文本。
        base_url: 当前页面 URL，用于解析相对跳转目标。
        normalize_url_for_http: URL 规范化函数。

    Returns:
        若存在 meta refresh 则返回解析结果，否则返回 `None`。

    Raises:
        无。
    """

    if not html_text.strip():
        return None

    soup = BeautifulSoup(html_text, "lxml")
    for meta_tag in soup.find_all("meta"):
        http_equiv = str(meta_tag.get("http-equiv", "") or "").strip().lower()
        if http_equiv != "refresh":
            continue

        raw_content = str(meta_tag.get("content", "") or "").strip()
        if not raw_content:
            return _MetaRefreshDirective(target_url="", raw_target="", delay_seconds=None, raw_content="")

        match = re.match(
            r"^\s*(?P<delay>\d+(?:\.\d+)?)\s*(?:;\s*url\s*=\s*(?P<target>.+?)\s*)?$",
            raw_content,
            flags=re.IGNORECASE,
        )
        if match is None:
            return _MetaRefreshDirective(target_url="", raw_target="", delay_seconds=None, raw_content=raw_content)

        delay_text = str(match.group("delay") or "").strip()
        raw_target = str(match.group("target") or "").strip().strip("\"'")
        delay_seconds: float | None = None
        if delay_text:
            try:
                delay_seconds = float(delay_text)
            except ValueError:
                delay_seconds = None

        target_url = ""
        if raw_target:
            try:
                target_url = normalize_url_for_http(urljoin(base_url, raw_target))
            except ValueError:
                target_url = ""
        return _MetaRefreshDirective(
            target_url=target_url,
            raw_target=raw_target,
            delay_seconds=delay_seconds,
            raw_content=raw_content,
        )
    return None


def _resolve_meta_refresh_follow_target(
    *,
    response: requests.Response,
    html_text: str,
    visited_urls: Collection[str],
    meta_refresh_hops: int,
    normalize_url_for_http: Callable[[str], str],
) -> str | None:
    """判断当前 HTML 是否需要按 meta refresh 继续抓取。

    Args:
        response: 当前响应对象。
        html_text: 当前 HTML 文本。
        visited_urls: 已访问 URL 集合，用于防环。
        meta_refresh_hops: 已发生的 meta refresh 跳数。
        normalize_url_for_http: URL 规范化函数。

    Returns:
        若需要继续抓取则返回下一跳 URL；否则返回 `None`。

    Raises:
        _FetchContentConversionError: 当 meta refresh 需要浏览器执行或出现循环时抛出。
    """

    directive = _extract_meta_refresh_directive(
        html_text,
        base_url=str(getattr(response, "url", "") or ""),
        normalize_url_for_http=normalize_url_for_http,
    )
    if directive is None:
        return None

    if directive.delay_seconds is None or directive.delay_seconds > _META_REFRESH_IMMEDIATE_MAX_SECONDS:
        raise _FetchContentConversionError(
            "HTML 页面包含需要浏览器执行的 meta refresh 跳转。",
            response_context=_build_fetch_content_runtime_context(response),
            original_error=RuntimeError("meta_refresh_requires_browser"),
            failure_reason="meta_refresh_requires_browser",
        )

    if not directive.target_url:
        raise _FetchContentConversionError(
            "HTML 页面包含无法解析目标的 meta refresh 跳转。",
            response_context=_build_fetch_content_runtime_context(response),
            original_error=RuntimeError("meta_refresh_requires_browser"),
            failure_reason="meta_refresh_requires_browser",
        )

    if meta_refresh_hops >= _MAX_META_REFRESH_HOPS or directive.target_url in visited_urls:
        raise _FetchContentConversionError(
            "HTML 页面 meta refresh 跳转出现循环或超过上限。",
            response_context=_build_fetch_content_runtime_context(response),
            original_error=RuntimeError("meta_refresh_requires_browser"),
            failure_reason="meta_refresh_requires_browser",
        )
    return directive.target_url


def _html_text_has_client_rendering_markers(raw_text: str) -> bool:
    """判断 HTML 文本是否更像需要真实浏览器渲染的前端壳页。

    Args:
        raw_text: 原始 HTML 或其文本摘录。

    Returns:
        命中典型客户端渲染壳页特征时返回 `True`。

    Raises:
        无。
    """

    normalized_text = str(raw_text or "").lower()
    if not normalized_text:
        return False
    return any(
        marker in normalized_text
        for marker in (
            "<script",
            'id="app"',
            "id='app'",
            'id="root"',
            "id='root'",
            "__next",
            "chunk-vendors",
            "webpack",
            "hydrate",
            "render(",
            "#/",
        )
    )


def _should_escalate_http_status_to_browser(http_status: int | None) -> bool:
    """判断 HTTP 错误状态是否值得优先升级到浏览器回退。"""

    return http_status in _PLAYWRIGHT_HTTP_ESCALATION_STATUSES


def _should_escalate_conversion_failure_to_browser(
    *,
    error_message: str,
    response_context: _FetchContentRuntimeContext | None,
) -> bool:
    """判断未类型化的 HTML 转换失败是否应升级到浏览器。"""

    if response_context is None:
        return False

    http_status = response_context.http_status
    if http_status is not None and not 200 <= http_status < 300:
        return False

    normalized_message = str(error_message or "").lower()
    if not any(
        token in normalized_message
        for token in (
            "主体抽取失败",
            "正文为空",
            "未产出结果",
            "empty",
            "no content",
        )
    ):
        return False

    raw_text = str(response_context.raw_content_text or response_context.response_excerpt or "")
    return _html_text_has_client_rendering_markers(raw_text)


def _should_escalate_stage_result_to_browser(stage_result: dict[str, str | bool | int | float | None] | None) -> bool:
    """判断阶段性 requests 结果是否应立即升级到浏览器回退。"""

    if not isinstance(stage_result, dict):
        return False
    if not bool(stage_result.get("attempted", True)):
        return False
    return bool(stage_result.get("timeout_like"))


def _should_escalate_pipeline_failure_to_browser(
    *,
    pipeline_error: HtmlPipelineStageError | None,
    response_context: _FetchContentRuntimeContext | None,
) -> bool:
    """判断 HTML 抽取失败是否值得升级到浏览器回退。"""

    if pipeline_error is None or response_context is None:
        return False
    if pipeline_error.stage != "extract":
        return False

    http_status = response_context.http_status
    if http_status is not None and not 200 <= http_status < 300:
        return False

    content_stats = pipeline_error.content_stats if isinstance(pipeline_error.content_stats, dict) else {}
    try:
        text_length = int(content_stats.get("text_length", 0) or 0)
    except (TypeError, ValueError):
        text_length = 0
    try:
        paragraph_count = int(content_stats.get("paragraph_count", 0) or 0)
    except (TypeError, ValueError):
        paragraph_count = 0

    quality_flags = {str(flag).strip().lower() for flag in pipeline_error.quality_flags}
    raw_text = str(response_context.raw_content_text or response_context.response_excerpt or "").lower()

    extractor_found_no_body = text_length <= _EMPTY_CONTENT_MIN_CHARS or paragraph_count <= 0
    quality_indicates_empty_shell = bool({"too_short", "too_few_blocks"} & quality_flags)
    has_client_rendering_markers = _html_text_has_client_rendering_markers(raw_text)
    return (extractor_found_no_body or quality_indicates_empty_shell) and has_client_rendering_markers


def _get_session_warmed_hosts(session: requests.Session) -> set[str]:
    """返回与当前 Session 同源的 warmup host 集合。"""

    warmed_hosts = getattr(session, "__dayu_warmed_hosts__", None)
    if isinstance(warmed_hosts, set):
        return warmed_hosts
    warmed_hosts = set()
    setattr(session, "__dayu_warmed_hosts__", warmed_hosts)
    return warmed_hosts


def _warmup_domain(
    session: requests.Session,
    *,
    url: str,
    timeout_seconds: float,
    headers: dict[str, str],
    resolve_timeout_budget: Callable[..., float],
    build_domain_home_url: Callable[[str], str],
    is_timeout_like_exception: Callable[[BaseException], bool],
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, str | bool | int | float | None]:
    """对目标域做一次预热请求以建立 Cookie。"""

    host = (urlparse(url).hostname or "").lower().strip()
    if not host:
        return {"attempted": False, "success": False, "reason": "invalid_host"}

    with _WARMED_HOSTS_LOCK:
        warmed_hosts = _get_session_warmed_hosts(session)
        if host in warmed_hosts:
            return {"attempted": False, "success": True, "reason": "already_warmed"}

    warmup_url = build_domain_home_url(url)
    _raise_if_cancelled(cancellation_token)
    warmup_timeout = min(
        resolve_timeout_budget(
            timeout_seconds,
            timeout_budget=timeout_budget,
            deadline_monotonic=deadline_monotonic,
        ),
        _WARMUP_TIMEOUT_SECONDS,
    )
    try:
        _raise_if_cancelled(cancellation_token)
        response = session.get(
            warmup_url,
            timeout=warmup_timeout,
            headers=headers,
            allow_redirects=True,
        )
        _raise_if_cancelled(cancellation_token)
        with _WARMED_HOSTS_LOCK:
            _get_session_warmed_hosts(session).add(host)
        return {
            "attempted": True,
            "success": True,
            "http_status": response.status_code,
            "final_url": response.url,
        }
    except Exception as exc:
        _raise_if_cancelled(cancellation_token)
        return {
            "attempted": True,
            "success": False,
            "reason": type(exc).__name__,
            "detail": str(exc),
            "timeout_like": is_timeout_like_exception(exc),
        }


def _probe_content_type(
    session: requests.Session,
    *,
    url: str,
    timeout_seconds: float,
    headers: dict[str, str],
    resolve_timeout_budget: Callable[..., float],
    is_timeout_like_exception: Callable[[BaseException], bool],
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, str | bool | int | None]:
    """探测目标资源类型（HEAD 优先，失败降级到 GET）。"""

    timeout = min(
        resolve_timeout_budget(
            timeout_seconds,
            timeout_budget=timeout_budget,
            deadline_monotonic=deadline_monotonic,
        ),
        _WARMUP_TIMEOUT_SECONDS,
    )
    try:
        _raise_if_cancelled(cancellation_token)
        response = session.head(url, timeout=timeout, headers=headers, allow_redirects=True)
        _raise_if_cancelled(cancellation_token)
        content_type = str(response.headers.get("Content-Type", "")).lower()
        return {
            "method": "HEAD",
            "content_type": content_type,
            "http_status": response.status_code,
            "final_url": response.url,
            "ok": True,
        }
    except Exception as head_exc:
        try:
            _raise_if_cancelled(cancellation_token)
            response = session.get(url, timeout=timeout, headers=headers, stream=True, allow_redirects=True)
            _raise_if_cancelled(cancellation_token)
            content_type = str(response.headers.get("Content-Type", "")).lower()
            response.close()
            return {
                "method": "GET",
                "content_type": content_type,
                "http_status": response.status_code,
                "final_url": response.url,
                "ok": True,
                "head_error": type(head_exc).__name__,
            }
        except Exception as get_exc:
            _raise_if_cancelled(cancellation_token)
            return {
                "method": "UNKNOWN",
                "content_type": "",
                "ok": False,
                "head_error": type(head_exc).__name__,
                "get_error": type(get_exc).__name__,
                "head_timeout_like": is_timeout_like_exception(head_exc),
                "get_timeout_like": is_timeout_like_exception(get_exc),
                "timeout_like": is_timeout_like_exception(head_exc) or is_timeout_like_exception(get_exc),
            }


def _should_route_response_to_html_pipeline(
    *,
    url: str,
    content_type: str,
    response_text: str,
    response_content: bytes,
) -> bool:
    """判断响应是否应进入 HTML 四段式流水线。"""

    normalized_content_type = str(content_type or "").lower()
    if "html" in normalized_content_type:
        return True

    uri_suffix = infer_suffix_from_uri(url)
    if uri_suffix in {".html", ".htm", ".xhtml"}:
        return True

    candidate_text = str(response_text or "").lstrip()
    if not candidate_text and response_content:
        candidate_text = response_content.decode("utf-8", errors="replace").lstrip()

    lowered_prefix = candidate_text[:256].lower()
    return lowered_prefix.startswith("<!doctype html") or "<html" in lowered_prefix


def _infer_docling_stream_name(*, url: str, content_type: str) -> str:
    """为 Docling 推断更稳定的输入流名称。"""

    normalized_content_type = str(content_type or "").lower()
    uri_suffix = infer_suffix_from_uri(url)

    if "pdf" in normalized_content_type or uri_suffix == ".pdf":
        return "page.pdf"
    if "xml" in normalized_content_type or uri_suffix in {".xml", ".xbrl"}:
        return "page.xml"
    if "json" in normalized_content_type or uri_suffix == ".json":
        return "page.json"
    if uri_suffix:
        return f"page{uri_suffix}"
    if normalized_content_type.startswith("text/"):
        return "page.txt"
    return "page.bin"


def _docling_convert_to_markdown(raw_bytes: bytes, stream_name: str) -> tuple[str, str, str]:
    """使用 Docling 将非 HTML 原始字节转换为 Markdown。

    Args:
        raw_bytes: 页面原始内容字节。
        stream_name: 流名称，决定 Docling 解析模式。

    Returns:
        ``(title, markdown, extraction_source)`` 三元组。

    Raises:
        RuntimeError: Docling 未安装或转换失败时抛出。
    """

    try:
        from io import BytesIO

        from docling.datamodel.base_models import DocumentStream
    except ImportError as exc:
        raise RuntimeError("Docling 未安装，无法转换非 HTML 内容") from exc

    stream = DocumentStream(name=stream_name, stream=BytesIO(raw_bytes))
    try:
        result = run_docling_pdf_conversion(
            lambda converter: converter.convert(stream),
            do_ocr=True,
            do_table_structure=True,
            table_mode="accurate",
            do_cell_matching=True,
        )
    except DoclingRuntimeInitializationError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Docling 转换失败: {exc}") from exc

    markdown = result.document.export_to_markdown().strip()
    title = ""
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            break
    return title, markdown, "docling"


def _fetch_and_convert_content(
    url: str,
    *,
    timeout_seconds: float,
    resolve_timeout_budget: Callable[..., float],
    normalize_url_for_http: Callable[[str], str],
    build_referer: Callable[[str], str],
    convert_html: Callable[..., HtmlPipelineResult],
    convert_non_html: Callable[[bytes, str], tuple[str, str, str]],
    session: requests.Session | None = None,
    get_web_session: Callable[[], requests.Session] | None = None,
    headers: dict[str, str] | None = None,
    build_fetch_headers: Callable[[str], dict[str, str]] | None = None,
    content_type_probe: dict[str, str | bool | int | None] | None = None,
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    cancellation_token: CancellationToken | None = None,
) -> dict[str, str | int | bool | list[str] | dict[str, int] | requests.Response | dict[str, str]]:
    """先下载页面内容，再按内容类型转换为低噪音 Markdown。

    Args:
        url: 已通过安全校验的网页链接。
        timeout_seconds: HTTP 请求超时秒数。
        resolve_timeout_budget: timeout 预算解析函数。
        normalize_url_for_http: URL 规范化函数。
        build_referer: Referer 构造函数。
        convert_html: HTML 四段式转换器。
        convert_non_html: 非 HTML 内容转换器。
        session: 可选复用 Session。
        get_web_session: 默认 Session 提供器。
        headers: 可选请求头。
        build_fetch_headers: 默认请求头构造器。
        content_type_probe: 可选内容类型探测结果。
        timeout_budget: Runner 注入的单次 tool call 总预算。
        deadline_monotonic: 当前工具调用的单调时钟 deadline。

    Returns:
        抓取和转换结果，包含 ``title/content/http_status/final_url`` 等字段。

    Raises:
        RuntimeError: HTTP 请求失败或内容转换失败时抛出。
        ValueError: 当缺少默认 Session 或请求头构造器时抛出。
    """

    resolved_session = session
    if resolved_session is None:
        if get_web_session is None:
            raise ValueError("缺少默认 requests Session 提供器")
        resolved_session = get_web_session()

    resolved_headers = headers
    if resolved_headers is None:
        if build_fetch_headers is None:
            raise ValueError("缺少默认请求头构造器")
        resolved_headers = build_fetch_headers(url)

    current_url = url
    current_headers = dict(resolved_headers)
    visited_urls = {url}
    meta_refresh_hops = 0

    while True:
        _raise_if_cancelled(cancellation_token)
        timeout = resolve_timeout_budget(
            timeout_seconds,
            timeout_budget=timeout_budget,
            deadline_monotonic=deadline_monotonic,
        )
        _raise_if_cancelled(cancellation_token)
        response = resolved_session.get(
            current_url,
            timeout=timeout,
            headers=current_headers,
            allow_redirects=True,
        )
        response.raise_for_status()
        _raise_if_cancelled(cancellation_token)

        probe = (
            content_type_probe or {"ok": False, "content_type": ""}
            if meta_refresh_hops == 0
            else {"ok": False, "content_type": ""}
        )
        content_type = str(probe.get("content_type", "") or response.headers.get("Content-Type", "")).lower()
        response_text = _decode_response_text(response)
        if _should_route_response_to_html_pipeline(
            url=getattr(response, "url", current_url),
            content_type=content_type,
            response_text=response_text,
            response_content=response.content,
        ):
            html_text = _extract_html_response_text(response)
            next_meta_refresh_url = _resolve_meta_refresh_follow_target(
                response=response,
                html_text=html_text,
                visited_urls=visited_urls,
                meta_refresh_hops=meta_refresh_hops,
                normalize_url_for_http=normalize_url_for_http,
            )
            if next_meta_refresh_url is not None:
                _raise_if_cancelled(cancellation_token)
                current_headers = dict(resolved_headers)
                current_headers["Referer"] = build_referer(str(getattr(response, "url", current_url) or current_url))
                current_url = next_meta_refresh_url
                visited_urls.add(next_meta_refresh_url)
                meta_refresh_hops += 1
                _close_response_safely(response)
                continue

            raw_challenge = detect_bot_challenge(
                response=response,
                content_text=html_text,
            )
            if raw_challenge.challenge_detected:
                raise _FetchContentConversionError(
                    "HTML 原始响应疑似反爬挑战页或访问门禁。",
                    response_context=_build_fetch_content_runtime_context(response),
                    original_error=RuntimeError("raw_html_bot_challenge"),
                )
            try:
                _raise_if_cancelled(cancellation_token)
                pipeline_result = convert_html(
                    html_text,
                    url=getattr(response, "url", current_url),
                )
            except RuntimeError as exc:
                raise _FetchContentConversionError(
                    str(exc),
                    response_context=_build_fetch_content_runtime_context(response),
                    original_error=exc,
                ) from exc
            title = pipeline_result.title
            markdown = pipeline_result.markdown
            extraction_source = pipeline_result.extractor_source
            renderer_source = pipeline_result.renderer_source
            normalization_applied = pipeline_result.normalization_applied
            quality_flags = list(pipeline_result.quality_flags)
            content_stats = dict(pipeline_result.content_stats)
        else:
            _raise_if_cancelled(cancellation_token)
            title, markdown, extraction_source = convert_non_html(
                response.content,
                _infer_docling_stream_name(
                    url=getattr(response, "url", current_url),
                    content_type=content_type,
                ),
            )
            renderer_source = "docling"
            normalization_applied = False
            quality_flags = []
            content_stats = {
                "text_length": len(markdown),
                "markdown_length": len(markdown),
            }
        return {
            "title": title,
            "content": markdown,
            "extraction_source": extraction_source,
            "renderer_source": renderer_source,
            "normalization_applied": normalization_applied,
            "quality_flags": quality_flags,
            "content_stats": content_stats,
            "http_status": response.status_code,
            "final_url": response.url,
            "redirect_hops": len(response.history) + meta_refresh_hops,
            "response": response,
            "response_headers": dict(response.headers),
            "response_excerpt": _extract_response_snippet(response),
        }
