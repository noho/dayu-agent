"""联网工具模块测试。"""

from __future__ import annotations

import builtins
from collections.abc import Callable
from pathlib import Path
from queue import Empty
import socket
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Optional, cast

import pytest
import requests
from urllib3.exceptions import MaxRetryError, ReadTimeoutError as Urllib3ReadTimeoutError

from dayu.contracts.protocols import ToolExecutionContext
from dayu.contracts.cancellation import CancelledError, CancellationToken
from dayu.engine.processors.html_pipeline import HtmlPipelineStageError
from dayu.engine.tool_errors import ToolBusinessError
from dayu.engine.tool_registry import ToolRegistry
from dayu.engine.tools.web_search_providers import (
    _candidate_providers,
    _normalize_domains,
    _resolve_duckduckgo_result_url,
    _resolve_provider,
    _search_with_duckduckgo,
    _search_with_serper,
    _search_with_tavily,
)
from dayu.engine.tools.web_tools import (
    _DEFAULT_BROWSER_USER_AGENT,
    _DEFAULT_SEC_USER_AGENT,
    _build_fetch_headers,
    _build_referer,
    _compute_deadline_monotonic,
    _create_fetch_web_page_tool,
    _create_search_web_tool,
    _extract_first_markdown_heading,
    _fetch_and_convert_content,
    _is_safe_public_url,
    _is_sec_host,
    _normalize_whitespace,
    _prepare_call_session,
    _resolve_timeout_budget,
    _warmup_domain,
    register_web_tools,
)


@pytest.fixture(autouse=True)
def _mock_public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """为 web_tools URL 安全校验提供稳定 DNS 结果。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        无。
    """

    def _fake_getaddrinfo(
        host: str,
        port: object,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        _ = (port, family, type, proto, flags)
        normalized = str(host).strip().lower()
        ip = "93.184.216.34"  # example.com 公网地址
        if normalized in {"sec.gov", "www.sec.gov", "data.sec.gov"}:
            ip = "151.101.2.132"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, 0))]

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.socket.getaddrinfo",
        _fake_getaddrinfo,
    )


class _FakeHttpResponse:
    """用于模拟 HTTP 响应对象。"""

    def __init__(
        self,
        *,
        text: str = "",
        content: bytes = b"",
        status_code: int = 200,
        headers: Optional[dict[str, str]] = None,
        url: str = "https://example.com",
        history: Optional[list[object]] = None,
        json_payload: Optional[dict[str, Any]] = None,
        http_error: Optional[requests.HTTPError] = None,
        encoding: str = "utf-8",
        apparent_encoding: str = "utf-8",
    ) -> None:
        """初始化模拟响应。

        Args:
            text: 响应文本。
            content: 响应二进制内容。
            json_payload: `json()` 返回值。
            http_error: `raise_for_status()` 时抛出的异常。
            encoding: requests 视角下的响应编码。
            apparent_encoding: requests 猜测得到的编码。

        Returns:
            无。

        Raises:
            无。
        """

        self.text = text
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}
        self.url = url
        self.history = history or []
        self._json_payload = json_payload or {}
        self._http_error = http_error
        self.encoding = encoding
        self.apparent_encoding = apparent_encoding

    def raise_for_status(self) -> None:
        """模拟 `requests.Response.raise_for_status`。

        Args:
            无。

        Returns:
            无。

        Raises:
            requests.HTTPError: 当配置了 `_http_error` 时抛出。
        """

        if self._http_error is not None:
            if getattr(self._http_error, "response", None) is None:
                self._http_error.response = self
            raise self._http_error

    def json(self) -> dict[str, Any]:
        """返回预设 JSON 内容。

        Args:
            无。

        Returns:
            预设 JSON 字典。

        Raises:
            无。
        """

        return self._json_payload

    def close(self) -> None:
        """模拟关闭连接。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """


class _FakeSession(requests.Session):
    """用于模拟 requests.Session。"""

    def __init__(
        self,
        *,
        get_handler: Optional[Any] = None,
        head_handler: Optional[Any] = None,
    ) -> None:
        """初始化会话。

        Args:
            get_handler: get 的处理函数。
            head_handler: head 的处理函数。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__()
        self._get_handler = get_handler
        self._head_handler = head_handler

    def get(self, url: str | bytes, **kwargs: Any) -> Any:
        """模拟 GET 请求。

        Args:
            url: 请求 URL。
            **kwargs: 额外参数。

        Returns:
            模拟响应对象。

        Raises:
            无。
        """

        normalized_url = url.decode("utf-8") if isinstance(url, bytes) else url
        if self._get_handler is None:
            return _FakeHttpResponse(url=normalized_url)
        return self._get_handler(normalized_url, **kwargs)

    def head(self, url: str | bytes, **kwargs: Any) -> Any:
        """模拟 HEAD 请求。

        Args:
            url: 请求 URL。
            **kwargs: 额外参数。

        Returns:
            模拟响应对象。

        Raises:
            无。
        """

        normalized_url = url.decode("utf-8") if isinstance(url, bytes) else url
        if self._head_handler is None:
            return _FakeHttpResponse(url=normalized_url)
        return self._head_handler(normalized_url, **kwargs)


class _FakeDocument:
    """模拟 Docling 的文档对象。"""

    def __init__(self, markdown: str) -> None:
        """初始化文档对象。

        Args:
            markdown: 导出的 Markdown 文本。

        Returns:
            无。

        Raises:
            无。
        """

        self._markdown = markdown

    def export_to_markdown(self) -> str:
        """导出 Markdown 文本。

        Args:
            无。

        Returns:
            预设 Markdown。

        Raises:
            无。
        """

        return self._markdown


class _FakeConvertResult:
    """模拟 Docling 转换结果容器。"""

    def __init__(self, markdown: str) -> None:
        """初始化转换结果。

        Args:
            markdown: 导出 Markdown 文本。

        Returns:
            无。

        Raises:
            无。
        """

        self.document = _FakeDocument(markdown)


class _FakeConverter:
    """模拟 Docling 转换器。"""

    def __init__(self, markdown: str) -> None:
        """初始化转换器。

        Args:
            markdown: 转换结果 Markdown。

        Returns:
            无。

        Raises:
            无。
        """

        self._markdown = markdown

    def convert(self, source: object) -> _FakeConvertResult:
        """执行转换。

        Args:
            source: 输入对象（测试中无需解析）。

        Returns:
            模拟转换结果。

        Raises:
            无。
        """

        _ = source
        return _FakeConvertResult(self._markdown)


def _build_wrapped_read_timeout_connection_error() -> requests.ConnectionError:
    """构造 `requests.ConnectionError(MaxRetryError(ReadTimeoutError))` 异常。"""

    wrapped_error = MaxRetryError(
        pool=cast(Any, None),
        url="/demo",
        reason=Urllib3ReadTimeoutError(cast(Any, None), "/demo", "timed out"),
    )
    return requests.ConnectionError(wrapped_error)


@pytest.mark.unit
def test_resolve_provider_and_candidates(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 provider 解析与候选顺序。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-demo")
    monkeypatch.setenv("SERPER_API_KEY", "serper-demo")

    assert _resolve_provider(preferred="auto") == "auto"
    assert _candidate_providers("auto") == ["tavily", "serper", "duckduckgo"]
    assert _candidate_providers("tavily") == ["tavily", "duckduckgo"]

    with pytest.raises(ValueError):
        _resolve_provider(preferred="invalid")


@pytest.mark.unit
def test_candidate_providers_auto_skips_unconfigured_api_key_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 auto 模式会跳过未配置 API key 的 provider。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    assert _candidate_providers("auto") == ["duckduckgo"]

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-demo")
    assert _candidate_providers("auto") == ["tavily", "duckduckgo"]

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SERPER_API_KEY", "serper-demo")
    assert _candidate_providers("auto") == ["serper", "duckduckgo"]


@pytest.mark.unit
def test_register_web_tools_registers_search_and_fetch_schema() -> None:
    """验证注册函数会写入两个 web tool 且 schema 参数生效。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    register_web_tools(
        registry,
        provider="duckduckgo",
        max_search_results=7,
        fetch_truncate_chars=2345,
    )

    assert "search_web" in registry.tools
    assert "fetch_web_page" in registry.tools

    search_parameters = registry.schemas["search_web"]["function"]["parameters"]["properties"]
    assert search_parameters["max_results"]["maximum"] == 7

    search_truncate = cast(Any, registry.tools["search_web"]).__tool_extra__.__truncate__
    fetch_truncate = cast(Any, registry.tools["fetch_web_page"]).__tool_extra__.__truncate__
    assert search_truncate.limits == {"max_items": 10}
    assert fetch_truncate.limits == {"max_chars": 2345}


@pytest.mark.unit
def test_web_tool_schema_descriptions_follow_workflow() -> None:
    """验证联网工具 schema 文案与当前实现保持一致。"""

    registry = ToolRegistry()
    register_web_tools(
        registry,
        provider="duckduckgo",
        max_search_results=7,
        fetch_truncate_chars=2345,
    )

    search_schema = registry.schemas["search_web"]["function"]
    fetch_schema = registry.schemas["fetch_web_page"]["function"]

    assert "直接写你最自然的查询" in search_schema["parameters"]["properties"]["query"]["description"]
    assert "只在你明确要收窄来源时填写" in search_schema["parameters"]["properties"]["domains"]["description"]
    assert search_schema["description"] == "搜索公开网页来源。"
    assert "description" not in fetch_schema["parameters"]["properties"]["url"]
    assert "先看 hint 和 next_action" in fetch_schema["description"]


@pytest.mark.unit
def test_resolve_timeout_budget_clamps_to_remaining_tool_deadline() -> None:
    """验证 web_tools 会按显式 tool budget 与剩余时间裁剪 timeout。"""

    deadline_monotonic = _compute_deadline_monotonic(2.0)
    assert deadline_monotonic is not None
    timeout = _resolve_timeout_budget(
        12.0,
        timeout_budget=2.0,
        deadline_monotonic=deadline_monotonic,
    )

    assert 0.0 < timeout <= 2.0


@pytest.mark.unit
def test_resolve_timeout_budget_keeps_subsecond_budget() -> None:
    """验证子秒级 tool budget 不会被隐式抬升到 1 秒。"""

    deadline_monotonic = _compute_deadline_monotonic(0.01)
    assert deadline_monotonic is not None

    with pytest.raises(requests.Timeout):
        _resolve_timeout_budget(
            12.0,
            timeout_budget=0.01,
            deadline_monotonic=deadline_monotonic,
        )


@pytest.mark.unit
def test_prepare_call_session_disables_retries_when_tool_deadline_exists() -> None:
    """验证存在显式 tool budget 时，会话会切换为无自动重试版本。"""

    original_session = requests.Session()
    prepared_session, should_close = _prepare_call_session(
        original_session,
        timeout_budget=5.0,
    )
    prepared_session_again, should_close_again = _prepare_call_session(
        original_session,
        timeout_budget=5.0,
    )

    assert should_close is False
    assert should_close_again is False
    assert prepared_session is not original_session
    assert prepared_session_again is prepared_session
    https_adapter = cast(Any, prepared_session.adapters["https://"])
    assert https_adapter.max_retries.total == 0


@pytest.mark.unit
def test_warmup_hosts_are_scoped_to_session() -> None:
    """验证 warmup 缓存只对当前 session 生效。"""

    session_a = _FakeSession(
        get_handler=lambda url, **_kwargs: _FakeHttpResponse(
            status_code=200,
            url=url,
        ),
    )
    session_b = _FakeSession(
        get_handler=lambda url, **_kwargs: _FakeHttpResponse(
            status_code=200,
            url=url,
        ),
    )
    headers = {"User-Agent": "pytest"}

    warmup_a_first = _warmup_domain(
        session_a,
        url="https://example.com/article",
        timeout_seconds=3.0,
        headers=headers,
    )
    warmup_a_second = _warmup_domain(
        session_a,
        url="https://example.com/article",
        timeout_seconds=3.0,
        headers=headers,
    )
    warmup_b_first = _warmup_domain(
        session_b,
        url="https://example.com/article",
        timeout_seconds=3.0,
        headers=headers,
    )

    assert warmup_a_first["attempted"] is True
    assert warmup_a_second["reason"] == "already_warmed"
    assert warmup_b_first["attempted"] is True


@pytest.mark.unit
def test_normalize_domains_handles_none_blank_and_invalid() -> None:
    """验证域名归一化覆盖空值、空白和非法元素分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
        ValueError: 传入非法元素时抛出。
    """

    assert _normalize_domains(None) == []
    assert _normalize_domains([" SEC.GOV ", "", "   ", "Example.com"]) == ["sec.gov", "example.com"]

    with pytest.raises(ValueError, match="domains 元素必须是字符串"):
        _normalize_domains(cast(Any, ["sec.gov", 123]))


@pytest.mark.unit
def test_is_safe_public_url_blocks_private_targets() -> None:
    """验证 URL 安全校验规则。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _is_safe_public_url("https://www.sec.gov/ixviewer") is True
    assert _is_safe_public_url("file:///tmp/a.txt") is False
    assert _is_safe_public_url("http://localhost:8080") is False
    assert _is_safe_public_url("http://127.0.0.1/test") is False
    assert _is_safe_public_url("https:///no-host") is False
    assert _is_safe_public_url("http://169.254.1.2/path") is False
    assert _is_safe_public_url("http://240.0.0.1/path") is False
    assert _is_safe_public_url("https://internal.local/path") is False


@pytest.mark.unit
def test_is_safe_public_url_blocks_private_dns_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证域名解析到私网地址时会被拒绝。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    def _private_dns(
        host: str,
        port: object,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        _ = (host, port, family, type, proto, flags)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("10.8.0.12", 0))]

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.socket.getaddrinfo",
        _private_dns,
    )

    assert _is_safe_public_url("https://safe.example.com/path") is False


@pytest.mark.unit
def test_is_safe_public_url_allows_fake_ip_for_public_hostname(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证公开域名在 fake-ip 环境下不会被误判为不安全。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    def _fake_ip_dns(
        host: str,
        port: object,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[int, int, int, str, tuple[str, int]]]:
        _ = (host, port, family, type, proto, flags)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("198.18.12.34", 0))]

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.socket.getaddrinfo",
        _fake_ip_dns,
    )

    assert _is_safe_public_url("https://safe.example.com/path") is True


@pytest.mark.unit
def test_is_safe_public_url_rejects_literal_fake_ip() -> None:
    """验证字面量 fake-ip 地址仍然会被拒绝。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _is_safe_public_url("https://198.18.12.34/path") is False


@pytest.mark.unit
def test_is_safe_public_url_allows_private_network_when_enabled() -> None:
    """验证开启内网放行后，私网目标会被允许。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _is_safe_public_url("http://localhost:8080", allow_private_network_url=True) is True
    assert _is_safe_public_url("http://127.0.0.1/test", allow_private_network_url=True) is True


@pytest.mark.unit
def test_is_safe_public_url_still_rejects_file_scheme_when_private_network_enabled() -> None:
    """验证开启内网放行后仍然拒绝非 HTTP/HTTPS scheme。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _is_safe_public_url("file:///tmp/a.txt", allow_private_network_url=True) is False


@pytest.mark.unit
def test_search_web_falls_back_to_next_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证检索在首选 provider 失败后会回退。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, search_web, _ = _create_search_web_tool(
        registry,
        provider="auto",
        request_timeout_seconds=12.0,
        max_search_results=20,
    )
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-demo")
    monkeypatch.setenv("SERPER_API_KEY", "serper-demo")

    def _raise_error(**_kwargs: Any) -> list[dict[str, str]]:
        raise RuntimeError("boom")

    monkeypatch.setattr("dayu.engine.tools.web_search_providers._search_with_tavily", _raise_error)
    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers._search_with_serper",
        lambda **_kwargs: [
            {
                "title": "SEC",
                "url": "https://www.sec.gov/Archives/edgar/data/1",
                "snippet": "ok",
                "published_date": "2025-01-01",
            }
        ],
    )

    output = search_web(query="AAPL 10-K")

    assert output["total"] == 1
    assert output["preferred_result"]["url"].startswith("https://www.sec.gov")
    assert "SEC" in output["preferred_result_summary"]
    assert output["next_action"] == "fetch_web_page"
    assert output["next_action_args"] == {"url": "https://www.sec.gov/Archives/edgar/data/1"}
    assert "fetch_web_page" in output["hint"]
    assert output["results"][0]["url"].startswith("https://www.sec.gov")


@pytest.mark.unit
def test_search_web_auto_missing_optional_api_keys_skips_provider_without_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 auto 模式下未配置 API key 的 provider 会被静默跳过并回退到 DuckDuckGo。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, search_web, _ = _create_search_web_tool(
        registry,
        provider="auto",
        request_timeout_seconds=12.0,
        max_search_results=20,
    )
    captured_infos: list[str] = []
    captured_warns: list[str] = []
    tavily_calls = 0
    serper_calls = 0

    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers.Log.info",
        lambda message, *, module="APP": captured_infos.append(f"{module}|{message}"),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers.Log.warn",
        lambda message, *, module="APP": captured_warns.append(f"{module}|{message}"),
    )
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SERPER_API_KEY", raising=False)

    def _fake_tavily(**_kwargs: Any) -> list[dict[str, str]]:
        """标记 Tavily 被错误调用。

        Args:
            **_kwargs: 透传的 provider 调用参数。

        Returns:
            无。

        Raises:
            AssertionError: 当 auto 模式错误调用 Tavily 时抛出。
        """

        nonlocal tavily_calls
        tavily_calls += 1
        raise AssertionError("auto 模式缺少 TAVILY_API_KEY 时不应调用 Tavily")

    def _fake_serper(**_kwargs: Any) -> list[dict[str, str]]:
        """标记 Serper 被错误调用。

        Args:
            **_kwargs: 透传的 provider 调用参数。

        Returns:
            无。

        Raises:
            AssertionError: 当 auto 模式错误调用 Serper 时抛出。
        """

        nonlocal serper_calls
        serper_calls += 1
        raise AssertionError("auto 模式缺少 SERPER_API_KEY 时不应调用 Serper")

    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers._search_with_tavily",
        _fake_tavily,
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers._search_with_serper",
        _fake_serper,
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers._search_with_duckduckgo",
        lambda **_kwargs: [
            {
                "title": "Example",
                "url": "https://example.com/news",
                "snippet": "ok",
                "published_date": "2026-03-31",
            }
        ],
    )

    output = search_web(query="example")

    assert output["total"] == 1
    assert tavily_calls == 0
    assert serper_calls == 0
    assert not captured_infos
    assert not captured_warns


@pytest.mark.unit
def test_search_web_explicit_provider_missing_api_key_still_logs_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证显式指定 provider 时，缺失 API key 仍按 warning 记录并继续回退。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, search_web, _ = _create_search_web_tool(
        registry,
        provider="tavily",
        request_timeout_seconds=12.0,
        max_search_results=20,
    )
    captured_infos: list[str] = []
    captured_warns: list[str] = []

    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers.Log.info",
        lambda message, *, module="APP": captured_infos.append(f"{module}|{message}"),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers.Log.warn",
        lambda message, *, module="APP": captured_warns.append(f"{module}|{message}"),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers._search_with_tavily",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("TAVILY_API_KEY 未配置")),
    )

    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers._search_with_duckduckgo",
        lambda **_kwargs: [
            {
                "title": "Example",
                "url": "https://example.com/news",
                "snippet": "ok",
                "published_date": "2026-03-31",
            }
        ],
    )

    output = search_web(query="example")

    assert output["total"] == 1
    assert not captured_infos
    assert len(captured_warns) == 1
    assert "provider=tavily 检索失败: TAVILY_API_KEY 未配置" in captured_warns[0]


@pytest.mark.unit
def test_search_web_uses_duckduckgo_branch_and_filters_unsafe(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 `search_web` 在 duckduckgo 分支会过滤不安全 URL。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, search_web, _ = _create_search_web_tool(
        registry,
        provider="duckduckgo",
        request_timeout_seconds=12.0,
        max_search_results=5,
    )

    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers._search_with_duckduckgo",
        lambda **_kwargs: [
            {"title": "local", "url": "http://localhost:8080", "snippet": "", "published_date": ""},
            {"title": "ok", "url": "https://example.com/news", "snippet": "", "published_date": ""},
        ],
    )

    output = search_web(query="example")

    assert output["total"] == 1
    assert output["preferred_result"]["url"] == "https://example.com/news"
    assert output["next_action"] == "fetch_web_page"
    assert output["results"][0]["url"] == "https://example.com/news"


@pytest.mark.unit
def test_search_web_allows_private_network_results_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证开启内网放行后，search_web 不再过滤私网 URL。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, search_web, _ = _create_search_web_tool(
        registry,
        provider="duckduckgo",
        request_timeout_seconds=12.0,
        max_search_results=5,
        allow_private_network_url=True,
    )

    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers._search_with_duckduckgo",
        lambda **_kwargs: [
            {"title": "local", "url": "http://localhost:8080", "snippet": "", "published_date": ""},
        ],
    )

    output = search_web(query="example")

    assert output["total"] == 1
    assert output["preferred_result"]["url"] == "http://localhost:8080"
    assert output["next_action_args"] == {"url": "http://localhost:8080"}
    assert output["results"][0]["url"] == "http://localhost:8080"


@pytest.mark.unit
def test_search_web_returns_refine_query_hint_when_no_safe_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `search_web` 在无可用结果时返回显式改写查询提示。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, search_web, _ = _create_search_web_tool(
        registry,
        provider="duckduckgo",
        request_timeout_seconds=12.0,
        max_search_results=5,
    )

    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers._search_with_duckduckgo",
        lambda **_kwargs: [
            {"title": "local", "url": "http://localhost:8080", "snippet": "blocked", "published_date": ""},
        ],
    )

    output = search_web(query="example")

    assert output["total"] == 0
    assert output["preferred_result"] is None
    assert output["preferred_result_summary"] == "未找到可直接抓取正文的公开网页结果。"
    assert output["next_action"] == "refine_query"
    assert output["next_action_args"] == {}
    assert "不要对空结果调用 fetch_web_page" in output["hint"]


@pytest.mark.unit
def test_search_web_empty_query_and_all_provider_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证空查询与全 provider 失败分支。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, search_web, _ = _create_search_web_tool(
        registry,
        provider="auto",
        request_timeout_seconds=12.0,
        max_search_results=20,
    )

    with pytest.raises(ValueError, match="query 不能为空"):
        search_web(query="   ")

    def _raise_error(**_kwargs: Any) -> list[dict[str, str]]:
        raise RuntimeError("down")

    monkeypatch.setattr("dayu.engine.tools.web_search_providers._search_with_tavily", _raise_error)
    monkeypatch.setattr("dayu.engine.tools.web_search_providers._search_with_serper", _raise_error)
    monkeypatch.setattr("dayu.engine.tools.web_search_providers._search_with_duckduckgo", _raise_error)

    with pytest.raises(RuntimeError, match="所有 provider 均不可用"):
        search_web(query="AAPL")


@pytest.mark.unit
def test_fetch_web_page_reads_html(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证网页抓取工具先用 requests 下载 HTML，再用 Docling 本地转换为 Markdown。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )

    fake_html = b"<html><body><h1>Demo</h1><p>hello world</p></body></html>"

    def _fake_get(_url: str, **_kwargs: Any) -> _FakeHttpResponse:
        return _FakeHttpResponse(
            content=fake_html,
            text="<html><body><h1>Demo</h1><p>hello world</p></body></html>",
            status_code=200,
            headers={"Content-Type": "text/html"},
            url="https://example.com/demo",
        )

    fake_session = _FakeSession(get_handler=_fake_get, head_handler=lambda _url, **_kwargs: _FakeHttpResponse(
        status_code=200,
        headers={"Content-Type": "text/html"},
        url="https://example.com/demo",
    ))
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="Demo",
            markdown="# Demo\n\nhello world",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 11, "markdown_length": 20},
        ),
    )

    output = fetch_web_page(url="https://example.com/demo")

    assert output["title"] == "Demo"
    assert "hello world" in output["content"]
    assert "url" in output


@pytest.mark.unit
def test_fetch_web_page_rejects_unsafe_url() -> None:
    """验证 `fetch_web_page` 拒绝不安全 URL。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )

    with pytest.raises(ToolBusinessError) as exc_info:
        fetch_web_page(url="file:///tmp/a.txt")

    assert exc_info.value.code == "permission_denied"
    assert "fetch safety policy" in exc_info.value.message
    assert "change_source" in exc_info.value.hint


@pytest.mark.unit
def test_fetch_web_page_uses_sec_user_agent_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证抓取 SEC 页面时使用 `SEC_USER_AGENT` 作为请求头。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )

    captured_headers: dict[str, str] = {}

    def _fake_get(_url: str, **kwargs: Any) -> _FakeHttpResponse:
        headers = kwargs.get("headers", {})
        if isinstance(headers, dict):
            captured_headers.update(headers)
        return _FakeHttpResponse(
            content=b"<html><body><h1>SEC Doc</h1></body></html>",
            text="<html><body><h1>SEC Doc</h1></body></html>",
            status_code=200,
            headers={"Content-Type": "text/html"},
            url="https://www.sec.gov/Archives/edgar/data/1/index.html",
        )

    def _fake_head(_url: str, **kwargs: Any) -> _FakeHttpResponse:
        headers = kwargs.get("headers", {})
        if isinstance(headers, dict):
            captured_headers.update(headers)
        return _FakeHttpResponse(
            status_code=200,
            headers={"Content-Type": "text/html"},
            url="https://www.sec.gov/Archives/edgar/data/1/index.html",
        )

    monkeypatch.setenv("SEC_USER_AGENT", "MyApp test@example.com")
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._get_web_session",
        lambda: _FakeSession(get_handler=_fake_get, head_handler=_fake_head),
    )

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="SEC Doc",
            markdown="# SEC Doc",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 7, "markdown_length": 9},
        ),
    )

    output = fetch_web_page(url="https://www.sec.gov/Archives/edgar/data/1/index.html")

    assert output["title"] == "SEC Doc"
    assert captured_headers.get("User-Agent") == "MyApp test@example.com"


@pytest.mark.unit
def test_fetch_web_page_percent_encodes_non_ascii_referer(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 `fetch_web_page` 会把中文 URL 的 Referer 规整为 ASCII。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )

    url = "https://www.msn.cn/zh-cn/money/经济/美团大涨-财报将靴子落地/ar-AA1ZlQ9I"
    captured_headers: dict[str, str] = {}
    captured_urls: list[str] = []

    def _fake_get(request_url: str, **kwargs: Any) -> _FakeHttpResponse:
        headers = kwargs.get("headers", {})
        if isinstance(headers, dict):
            captured_headers.update(headers)
        captured_urls.append(request_url)
        return _FakeHttpResponse(
            content=b"<html><body><h1>Demo</h1></body></html>",
            text="<html><body><h1>Demo</h1></body></html>",
            status_code=200,
            headers={"Content-Type": "text/html"},
            url=request_url,
        )

    def _fake_head(request_url: str, **kwargs: Any) -> _FakeHttpResponse:
        headers = kwargs.get("headers", {})
        if isinstance(headers, dict):
            captured_headers.update(headers)
        captured_urls.append(request_url)
        return _FakeHttpResponse(
            status_code=200,
            headers={"Content-Type": "text/html"},
            url=request_url,
        )

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._get_web_session",
        lambda: _FakeSession(get_handler=_fake_get, head_handler=_fake_head),
    )

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="Demo",
            markdown="# Demo",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 4, "markdown_length": 6},
        ),
    )

    output = fetch_web_page(url=url)

    assert output["title"] == "Demo"
    assert captured_urls
    assert any("%E7%BB%8F%E6%B5%8E" in item for item in captured_urls)
    assert captured_headers["Referer"] == _build_referer(url)
    assert "%E7%BB%8F%E6%B5%8E" in captured_headers["Referer"]
    assert "经济" not in captured_headers["Referer"]


@pytest.mark.unit
def test_search_with_tavily_requires_api_key() -> None:
    """验证 Tavily 缺失 API key 时抛错。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    with pytest.MonkeyPatch().context() as mp:
        mp.delenv("TAVILY_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="TAVILY_API_KEY 未配置"):
            _search_with_tavily(
                query="AAPL",
                domains=[],
                recency_days=None,
                max_results=3,
                timeout_seconds=12.0,
            )


@pytest.mark.unit
def test_search_with_tavily_maps_payload_and_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Tavily 请求 payload 组装和结果映射。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    captured: dict[str, Any] = {}

    def _fake_post(url: str, *, json: dict[str, Any], timeout: float) -> _FakeHttpResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeHttpResponse(
            json_payload={
                "results": [
                    {
                        "title": "Doc",
                        "url": "https://example.com/doc",
                        "content": "snippet",
                        "published_date": "2026-01-01",
                    }
                ]
            }
        )

    monkeypatch.setenv("TAVILY_API_KEY", "t-key")
    monkeypatch.setattr("dayu.engine.tools.web_search_providers.requests.post", _fake_post)

    rows = _search_with_tavily(
        query="AAPL",
        domains=["sec.gov"],
        recency_days=7,
        max_results=4,
        timeout_seconds=8.0,
    )

    assert captured["url"] == "https://api.tavily.com/search"
    assert captured["json"]["include_domains"] == ["sec.gov"]
    assert captured["json"]["days"] == 7
    assert rows[0]["snippet"] == "snippet"


@pytest.mark.unit
def test_search_with_serper_requires_api_key() -> None:
    """验证 Serper 缺失 API key 时抛错。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    with pytest.raises(RuntimeError, match="SERPER_API_KEY 未配置"):
        _search_with_serper(
            query="AAPL",
            domains=[],
            recency_days=None,
            max_results=3,
            timeout_seconds=12.0,
        )


@pytest.mark.unit
def test_search_with_serper_maps_payload_and_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Serper 请求 payload 组装和结果映射。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    captured: dict[str, Any] = {}

    def _fake_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> _FakeHttpResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeHttpResponse(
            json_payload={
                "organic": [
                    {
                        "title": "Apple",
                        "link": "https://example.com/apple",
                        "snippet": "intro",
                    }
                ]
            }
        )

    monkeypatch.setenv("SERPER_API_KEY", "s-key")
    monkeypatch.setattr("dayu.engine.tools.web_search_providers.requests.post", _fake_post)

    rows = _search_with_serper(
        query="AAPL",
        domains=["sec.gov", "apple.com"],
        recency_days=2,
        max_results=6,
        timeout_seconds=9.0,
    )

    assert captured["url"] == "https://google.serper.dev/search"
    assert "site:sec.gov" in captured["json"]["q"]
    assert captured["json"]["tbs"] == "qdr:d2"
    assert captured["headers"]["X-API-KEY"] == "s-key"
    assert rows[0]["url"] == "https://example.com/apple"


@pytest.mark.unit
def test_duckduckgo_extracts_redirect_target_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 DuckDuckGo /l/?uddg= 跳转链接可解析为真实目标 URL。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake_html = """
    <html><body>
      <div class="result">
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fnews">Example News</a>
        <div class="result__snippet">snippet</div>
      </div>
    </body></html>
    """

    monkeypatch.setattr(
        "dayu.engine.tools.web_search_providers.requests.get",
        lambda *_args, **_kwargs: _FakeHttpResponse(text=fake_html),
    )

    rows = _search_with_duckduckgo(
        query="example",
        domains=[],
        max_results=5,
        timeout_seconds=12.0,
    )

    assert len(rows) == 1
    assert rows[0]["url"] == "https://example.com/news"


@pytest.mark.unit
def test_search_with_duckduckgo_skips_invalid_rows_and_breaks(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 DuckDuckGo 结果解析的跳过与 `max_results` 截断分支。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake_html = """
    <html><body>
      <div class="result"><div class="result__snippet">no anchor</div></div>
      <div class="result"><a class="result__a" href="">empty</a></div>
      <div class="result"><a class="result__a" href="https://ok.example/a">First</a></div>
      <div class="result"><a class="result__a" href="https://ok.example/b">Second</a></div>
    </body></html>
    """

    captured_params: dict[str, Any] = {}

    def _fake_get(url: str, *, params: dict[str, str], timeout: float, headers: dict[str, str]) -> _FakeHttpResponse:
        captured_params["url"] = url
        captured_params["params"] = params
        captured_params["timeout"] = timeout
        captured_params["headers"] = headers
        return _FakeHttpResponse(text=fake_html)

    monkeypatch.setattr("dayu.engine.tools.web_search_providers.requests.get", _fake_get)

    rows = _search_with_duckduckgo(
        query="visa",
        domains=["sec.gov", "example.com"],
        max_results=1,
        timeout_seconds=5.0,
    )

    assert captured_params["params"]["q"] == "visa site:sec.gov site:example.com"
    assert rows == [
        {
            "title": "First",
            "url": "https://ok.example/a",
            "snippet": "",
            "published_date": "",
        }
    ]


@pytest.mark.unit
def test_resolve_duckduckgo_result_url_branches() -> None:
    """验证 DuckDuckGo URL 解析函数的各分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _resolve_duckduckgo_result_url("") == ""
    assert (
        _resolve_duckduckgo_result_url("//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa")
        == "https://example.com/a"
    )
    assert _resolve_duckduckgo_result_url("/x/?uddg=https%3A%2F%2Fexample.com") == ""
    assert (
        _resolve_duckduckgo_result_url("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fb")
        == "https://example.com/b"
    )
    assert _resolve_duckduckgo_result_url("https://example.com/c") == "https://example.com/c"


@pytest.mark.unit
def test_extract_heading_and_normalize_whitespace() -> None:
    """验证 Markdown 标题提取与空白规整函数。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _extract_first_markdown_heading("# Title\n\ncontent") == "Title"
    assert _extract_first_markdown_heading("no heading") == ""
    assert _normalize_whitespace(" a   b  \n\n c\t\t d ") == "a b\nc d"


@pytest.mark.unit
def test_build_fetch_headers_and_is_sec_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 SEC/非 SEC 请求头与域名判断逻辑。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setenv("SEC_USER_AGENT", "   ")

    sec_headers = _build_fetch_headers("https://www.sec.gov/Archives/edgar/data/1")
    normal_headers = _build_fetch_headers("https://example.com/page")

    assert _is_sec_host("https://sec.gov") is True
    assert _is_sec_host("https://data.sec.gov/sub") is True
    assert _is_sec_host("https://www.sec.gov/Archives") is True
    assert _is_sec_host("https://example.com") is False

    assert sec_headers["User-Agent"] == _DEFAULT_SEC_USER_AGENT
    assert "Accept-Encoding" in sec_headers
    assert normal_headers["User-Agent"] == _DEFAULT_BROWSER_USER_AGENT
    assert "Accept-Language" in normal_headers


@pytest.mark.unit
def test_build_fetch_headers_adapts_accept_encoding_to_runtime_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 Accept-Encoding 会按当前运行时解码能力自适应收敛。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(
        "dayu.engine.tools.web_http_encoding._is_optional_module_available",
        lambda module_names: any(name in {"brotli", "zstandard"} for name in module_names),
    )
    headers_with_optional_codecs = _build_fetch_headers("https://example.com/article")
    assert headers_with_optional_codecs["Accept-Encoding"] == "gzip, deflate, br, zstd"

    monkeypatch.setattr(
        "dayu.engine.tools.web_http_encoding._is_optional_module_available",
        lambda module_names: False,
    )
    headers_without_optional_codecs = _build_fetch_headers("https://example.com/article")
    assert headers_without_optional_codecs["Accept-Encoding"] == "gzip, deflate"


@pytest.mark.unit
def test_fetch_and_convert_content_pdf_docling_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 PDF 内容在 Docling 不可用时抛出友好错误。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    original_import = builtins.__import__

    def _fake_import(name: str, globals_: Any, locals_: Any, fromlist: tuple[str, ...], level: int) -> Any:
        if name.startswith("docling"):
            raise ImportError("docling missing")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=b"%PDF-1.4",
            text="",
            headers={"Content-Type": "application/pdf"},
        ),
    )

    with pytest.raises(RuntimeError, match="Docling 未安装"):
        _fetch_and_convert_content("https://example.com/report.pdf", timeout_seconds=3.0, session=fake_session)


@pytest.mark.unit
def test_fetch_and_convert_content_sec_403_reraises_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 SEC 403 在底层函数会抛出 HTTPError。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    http_error = requests.HTTPError("403")
    http_error.response = SimpleNamespace(status_code=403)

    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(http_error=http_error),
    )

    with pytest.raises(requests.HTTPError):
        _fetch_and_convert_content(
            "https://www.sec.gov/Archives/edgar/data/1",
            timeout_seconds=3.0,
            session=fake_session,
        )


@pytest.mark.unit
def test_fetch_and_convert_content_non_403_http_error_reraises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证非 SEC 403 的 HTTPError 会原样上抛。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    http_error = requests.HTTPError("500")
    http_error.response = SimpleNamespace(status_code=500)

    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(http_error=http_error),
    )

    with pytest.raises(requests.HTTPError):
        _fetch_and_convert_content("https://example.com", timeout_seconds=3.0, session=fake_session)


@pytest.mark.unit
def test_fetch_and_convert_content_pdf_convert_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 PDF Docling 转换异常会被包装为 RuntimeError。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=b"%PDF-1.4",
            text="",
            headers={"Content-Type": "application/pdf"},
        ),
    )

    def _raise_convert_failure(
        convert_operation: Callable[..., _FakeConvertResult],
        *,
        do_ocr: bool,
        do_table_structure: bool,
        table_mode: str,
        do_cell_matching: bool,
    ) -> _FakeConvertResult:
        """模拟统一 Docling 运行时抛出转换失败。

        Args:
            convert_operation: 转换回调。
            do_ocr: OCR 开关。
            do_table_structure: 表格结构开关。
            table_mode: 表格模式。
            do_cell_matching: 单元格匹配开关。

        Returns:
            无。

        Raises:
            ValueError: 固定抛出。
        """

        _ = (convert_operation, do_ocr, do_table_structure, table_mode, do_cell_matching)
        raise ValueError("convert failed")

    monkeypatch.setattr(
        "dayu.engine.tools.web_fetch_orchestrator.run_docling_pdf_conversion",
        _raise_convert_failure,
    )

    with pytest.raises(RuntimeError, match="Docling 转换失败"):
        _fetch_and_convert_content("https://example.com/report.pdf", timeout_seconds=3.0, session=fake_session)


@pytest.mark.unit
def test_fetch_and_convert_content_pdf_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 PDF 仍走 Docling 成功路径。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=b"%PDF-1.4",
            text="",
            status_code=200,
            headers={"Content-Type": "application/pdf"},
            url="https://example.com/report.pdf",
        ),
    )

    def _return_success(
        convert_operation: Callable[..., _FakeConvertResult],
        *,
        do_ocr: bool,
        do_table_structure: bool,
        table_mode: str,
        do_cell_matching: bool,
    ) -> _FakeConvertResult:
        """模拟统一 Docling 运行时返回成功结果。

        Args:
            convert_operation: 转换回调。
            do_ocr: OCR 开关。
            do_table_structure: 表格结构开关。
            table_mode: 表格模式。
            do_cell_matching: 单元格匹配开关。

        Returns:
            模拟转换结果。

        Raises:
            无。
        """

        _ = (convert_operation, do_ocr, do_table_structure, table_mode, do_cell_matching)
        return _FakeConvertResult("# My Title\n\nBody")

    monkeypatch.setattr(
        "dayu.engine.tools.web_fetch_orchestrator.run_docling_pdf_conversion",
        _return_success,
    )

    result = _fetch_and_convert_content("https://example.com/report.pdf", timeout_seconds=3.0, session=fake_session)

    assert result["title"] == "My Title"
    assert result["content"] == "# My Title\n\nBody"
    assert result["extraction_source"] == "docling"
    assert result["renderer_source"] == "docling"


@pytest.mark.unit
def test_fetch_and_convert_content_plain_text_uses_docling(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证非 HTML 文本响应仍走 Docling 而不是 HTML 流水线。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=b"line1\nline2\nplain text body",
            text="line1\nline2\nplain text body",
            status_code=200,
            headers={"Content-Type": "text/plain; charset=utf-8"},
            url="https://example.com/report.txt",
        ),
    )

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": (_ for _ in ()).throw(AssertionError("plain text 不应走 HTML pipeline")),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._docling_convert_to_markdown",
        lambda raw_bytes, stream_name: ("Plain Title", "# Plain Title\n\nPlain body", "docling"),
    )

    result = _fetch_and_convert_content(
        "https://example.com/report.txt",
        timeout_seconds=3.0,
        session=fake_session,
    )

    assert result["title"] == "Plain Title"
    assert result["content"] == "# Plain Title\n\nPlain body"
    assert result["extraction_source"] == "docling"
    assert result["renderer_source"] == "docling"


@pytest.mark.unit
def test_fetch_and_convert_content_html_uses_four_stage_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 HTML 内容会走四段式流水线而不是 Docling。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html = """
    <html>
      <head>
        <title>示例新闻标题 | 站点</title>
      </head>
      <body>
        <main>
          <div class="g-articl-text">
            <p>第一段：公司发布业绩后，市场对全年增速指引低于预期作出负面反应。</p>
            <p>第二段：管理层同时提示短期毛利率承压，进一步放大了资金抛售情绪。</p>
            <p>第三段：截至午后，公司股价一度跌超 15%，成交额显著放大。</p>
          </div>
        </main>
      </body>
    </html>
    """.encode("utf-8")

    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=html,
            text=html.decode("utf-8"),
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            url="https://example.com/article",
        ),
    )

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="示例新闻标题",
            markdown="# 示例新闻标题\n\n第一段：公司发布业绩后，市场对全年增速指引低于预期作出负面反应。",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 80},
        ),
    )

    result = _fetch_and_convert_content(
        "https://example.com/article",
        timeout_seconds=3.0,
        session=fake_session,
    )

    assert result["title"] == "示例新闻标题"
    assert "全年增速指引低于预期" in result["content"]
    assert result["extraction_source"] == "trafilatura"
    assert result["renderer_source"] == "markdownify"
    assert result["normalization_applied"] is True


@pytest.mark.unit
def test_fetch_and_convert_content_html_without_content_type_uses_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证缺失 Content-Type 但 URL 为 HTML 时仍走四段式流水线。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html = "<html><head><title>隐式 HTML</title></head><body><article><p>正文一</p><p>正文二</p></article></body></html>"
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=html.encode("utf-8"),
            text=html,
            status_code=200,
            headers={},
            url="https://example.com/article.html",
        ),
    )

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="隐式 HTML",
            markdown="# 隐式 HTML\n\n正文一",
            extractor_source="bs_fallback",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 10, "markdown_length": 16},
        ),
    )

    result = _fetch_and_convert_content(
        "https://example.com/article.html",
        timeout_seconds=3.0,
        session=fake_session,
    )

    assert result["title"] == "隐式 HTML"
    assert result["extraction_source"] == "bs_fallback"


@pytest.mark.unit
def test_fetch_and_convert_content_corrects_html_charset_before_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 HTML header 缺 charset 时，会按 meta charset 纠正解码后再进入抽取。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html = (
        "<html><head><title>中国新闻网</title>"
        "<meta http-equiv='Content-Type' content='text/html; charset=gb2312'></head>"
        "<body><article><p>财经新闻滚动新闻</p><p>梳理天下新闻</p></article></body></html>"
    )
    encoded_html = html.encode("gb18030")
    wrongly_decoded_text = encoded_html.decode("latin-1")
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=encoded_html,
            text=wrongly_decoded_text,
            status_code=200,
            headers={"Content-Type": "text/html"},
            url="http://www.chinanews.com.cn/economic.html",
            encoding="ISO-8859-1",
            apparent_encoding="GB18030",
        ),
    )

    def _fake_convert(html_text: str, url: str = "") -> SimpleNamespace:
        assert "中国新闻网" in html_text
        assert "财经新闻滚动新闻" in html_text
        assert "ÖÐ¹ú" not in html_text
        _ = url
        return SimpleNamespace(
            title="中国新闻网",
            markdown="# 中国新闻网\n\n财经新闻滚动新闻",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 20, "markdown_length": 18},
        )

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        _fake_convert,
    )

    result = _fetch_and_convert_content(
        "http://www.chinanews.com.cn/economic.html",
        timeout_seconds=3.0,
        session=fake_session,
        content_type_probe={"ok": True, "content_type": "text/html"},
    )

    assert result["title"] == "中国新闻网"
    assert "财经新闻滚动新闻" in result["content"]


@pytest.mark.unit
def test_fetch_and_convert_content_follows_immediate_meta_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证立即执行的 meta refresh 会在抓取层继续跟随而不是直接抽壳页。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    requested_urls: list[str] = []
    shell_html = "<html><head><meta http-equiv='Refresh' content='0;URL=/economic.shtml'></head><body></body></html>"
    article_html = "<html><head><title>经济频道</title></head><body><article><p>正文一</p><p>正文二</p></article></body></html>"

    def _get_handler(url: str, **_kwargs: Any) -> _FakeHttpResponse:
        requested_urls.append(url)
        if url.endswith("economic.html"):
            return _FakeHttpResponse(
                content=shell_html.encode("utf-8"),
                text=shell_html,
                status_code=200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                url="http://www.chinanews.com.cn/economic.html",
            )
        return _FakeHttpResponse(
            content=article_html.encode("utf-8"),
            text=article_html,
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            url="http://www.chinanews.com.cn/economic.shtml",
        )

    fake_session = _FakeSession(get_handler=_get_handler)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="经济频道",
            markdown="# 经济频道\n\n正文一\n\n正文二",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 20, "markdown_length": 18},
        ),
    )

    result = _fetch_and_convert_content(
        "http://www.chinanews.com.cn/economic.html",
        timeout_seconds=3.0,
        session=fake_session,
        headers={"User-Agent": "UA", "Referer": "http://www.chinanews.com.cn/"},
        content_type_probe={"ok": True, "content_type": "text/html; charset=utf-8"},
    )

    assert requested_urls == [
        "http://www.chinanews.com.cn/economic.html",
        "http://www.chinanews.com.cn/economic.shtml",
    ]
    assert result["final_url"] == "http://www.chinanews.com.cn/economic.shtml"
    assert result["redirect_hops"] == 1
    assert result["title"] == "经济频道"


@pytest.mark.unit
def test_fetch_and_convert_content_recomputes_timeout_for_each_meta_refresh_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 meta refresh 每一跳都会按剩余预算重新计算 timeout。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    requested_timeouts: list[float] = []
    shell_html = "<html><head><meta http-equiv='refresh' content='0;url=/next'></head><body></body></html>"
    article_html = "<html><head><title>Next</title></head><body><article><p>正文</p></article></body></html>"

    class _TimeoutTrackingSession(requests.Session):
        """记录每次 GET timeout 的最小测试桩。"""

        def __init__(self) -> None:
            """初始化 requests 会话基类。"""

            super().__init__()

        def get(self, url: str | bytes, **kwargs: Any) -> Any:
            """返回带 meta refresh 的响应，并记录 timeout。"""

            requested_timeouts.append(float(kwargs["timeout"]))
            normalized_url = url.decode("utf-8") if isinstance(url, bytes) else url
            if normalized_url.endswith("/start"):
                return _FakeHttpResponse(
                    content=shell_html.encode("utf-8"),
                    text=shell_html,
                    status_code=200,
                    headers={"Content-Type": "text/html; charset=utf-8"},
                    url="https://example.com/start",
                )
            return _FakeHttpResponse(
                content=article_html.encode("utf-8"),
                text=article_html,
                status_code=200,
                headers={"Content-Type": "text/html; charset=utf-8"},
                url="https://example.com/next",
            )

    monotonic_values = iter([100.0, 101.6])
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.time.monotonic",
        lambda: next(monotonic_values),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="Next",
            markdown="# Next\n\n正文",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 10, "markdown_length": 8},
        ),
    )

    result = _fetch_and_convert_content(
        "https://example.com/start",
        timeout_seconds=2.0,
        session=_TimeoutTrackingSession(),
        content_type_probe={"ok": True, "content_type": "text/html; charset=utf-8"},
        timeout_budget=2.0,
        deadline_monotonic=102.0,
    )

    assert result["final_url"] == "https://example.com/next"
    assert requested_timeouts == [2.0, pytest.approx(0.4, abs=0.01)]


@pytest.mark.unit
def test_fetch_and_convert_content_raises_cancelled_before_network_stage() -> None:
    """验证 requests 抓取在进入下一联网阶段前会执行协作式取消检查。"""

    cancellation_token = CancellationToken()
    cancellation_token.cancel()

    with pytest.raises(CancelledError):
        _fetch_and_convert_content(
            "https://example.com",
            timeout_seconds=3.0,
            session=_FakeSession(),
            cancellation_token=cancellation_token,
        )


@pytest.mark.unit
def test_fetch_web_page_uses_playwright_for_unsupported_content_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 HTML 响应使用不受支持编码时会优先升级到浏览器回退。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=b"compressed-bytes",
            text="\x00\x01compressed",
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8", "Content-Encoding": "br"},
            url="https://www.zaobao.com.sg/finance/world",
        ),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
        ),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr("dayu.engine.tools.web_tools._warmup_domain", lambda *args, **kwargs: {"attempted": False})
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {"attempted": True, "ok": True, "content_type": "text/html; charset=utf-8"},
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "联合早报财经",
            "content": "# 联合早报财经\n\n浏览器成功拿到正文。",
            "final_url": url,
        },
    )

    output = fetch_web_page(url="https://www.zaobao.com.sg/finance/world")

    assert output["fetch_backend"] == "playwright"
    assert output["title"] == "联合早报财经"


@pytest.mark.unit
def test_fetch_web_page_uses_playwright_for_delayed_meta_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证延时 meta refresh 不会继续抽壳页，而是升级到浏览器回退。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    delayed_refresh_html = "<html><head><meta http-equiv='refresh' content='3;url=/finance/world'></head><body></body></html>"
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=delayed_refresh_html.encode("utf-8"),
            text=delayed_refresh_html,
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            url="https://example.com/shell",
        ),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
        ),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr("dayu.engine.tools.web_tools._warmup_domain", lambda *args, **kwargs: {"attempted": False})
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {"attempted": True, "ok": True, "content_type": "text/html; charset=utf-8"},
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "Delayed Refresh",
            "content": "# Delayed Refresh\n\nBrowser fallback content.",
            "final_url": "https://example.com/finance/world",
        },
    )

    output = fetch_web_page(url="https://example.com/shell")

    assert output["fetch_backend"] == "playwright"
    assert output["final_url"] == "https://example.com/finance/world"


@pytest.mark.unit
def test_fetch_web_page_uses_playwright_fallback_on_ssl_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 requests SSL/TLS 失败时会优先切到浏览器回退。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    fake_session = _FakeSession()
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr("dayu.engine.tools.web_tools._warmup_domain", lambda *args, **kwargs: {"attempted": False})
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {"attempted": True, "ok": True, "content_type": "text/html; charset=utf-8"},
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_content",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.SSLError("[SSL: UNEXPECTED_EOF_WHILE_READING]")),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "TLS Browser Recovery",
            "content": "# TLS Browser Recovery\n\nRecovered by browser.",
            "final_url": url,
        },
    )

    output = fetch_web_page(url="https://www.capitalfutures.com.tw/article")

    assert output["fetch_backend"] == "playwright"
    assert output["title"] == "TLS Browser Recovery"


@pytest.mark.unit
def test_fetch_web_page_reports_ssl_error_when_browser_recovery_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 SSL/TLS 失败且浏览器回退也不可用时，工具返回明确的 ssl_error。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    fake_session = _FakeSession()
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr("dayu.engine.tools.web_tools._warmup_domain", lambda *args, **kwargs: {"attempted": False})
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {"attempted": True, "ok": True, "content_type": "text/html; charset=utf-8"},
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_content",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.SSLError("certificate verify failed")),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": False,
            "availability": "unprocessable",
            "reason": "playwright_error",
        },
    )

    with pytest.raises(ToolBusinessError) as exc_info:
        fetch_web_page(url="https://www.capitalfutures.com.tw/article")

    assert exc_info.value.code == "ssl_error"


@pytest.mark.unit
def test_fetch_web_page_passes_execution_cancellation_token_to_playwright_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 fetch_web_page 会把 execution_context 中的取消令牌透传给浏览器回退。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    cancellation_token = CancellationToken()
    captured: dict[str, object] = {}
    fake_session = _FakeSession()
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr("dayu.engine.tools.web_tools._warmup_domain", lambda *args, **kwargs: {"attempted": False})
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {"attempted": True, "ok": True, "content_type": "text/html; charset=utf-8"},
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_content",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.exceptions.SSLError("certificate verify failed")),
    )

    def _fake_playwright_fallback(
        *,
        url: str,
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
        timeout_budget: float | None = None,
        deadline_monotonic: float | None = None,
        playwright_channel: str | None = None,
        playwright_storage_state_path: str = "",
        cancellation_token: CancellationToken | None = None,
    ) -> dict[str, Any]:
        captured["url"] = url
        captured["timeout_seconds"] = timeout_seconds
        captured["cancellation_token"] = cancellation_token
        _ = (headers, timeout_budget, deadline_monotonic, playwright_channel, playwright_storage_state_path)
        return {
            "ok": True,
            "title": "TLS Browser Recovery",
            "content": "# TLS Browser Recovery\n\nRecovered by browser.",
            "final_url": url,
        }

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        _fake_playwright_fallback,
    )

    output = fetch_web_page(
        url="https://www.capitalfutures.com.tw/article",
        execution_context=ToolExecutionContext(cancellation_token=cancellation_token),
    )

    assert output["fetch_backend"] == "playwright"
    assert captured["cancellation_token"] is cancellation_token


@pytest.mark.unit
def test_fetch_web_page_logs_pipeline_diagnostics_for_html(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `fetch_web_page` 会记录四段式诊断字段。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )

    html = """
    <html>
      <head><title>通用回退测试</title></head>
      <body>
        <article>
          <p>公司公布业绩后，市场担忧增速放缓，股价随即走弱。</p>
          <p>管理层对下一季度利润率的表述偏谨慎，成为短线情绪催化。</p>
          <p>盘中成交量明显放大，说明抛售来自主动资金而非单笔噪声成交。</p>
        </article>
      </body>
    </html>
    """.encode("utf-8")

    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=html,
            text=html.decode("utf-8"),
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            url="https://example.com/fallback",
        ),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            url="https://example.com/fallback",
        ),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    diagnostics_payloads: list[dict[str, Any]] = []
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._log_fetch_diagnostics",
        lambda payload: diagnostics_payloads.append(payload),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="通用回退测试",
            markdown="# 通用回退测试\n\n公司公布业绩后，市场担忧增速放缓。",
            extractor_source="bs_fallback",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=("too_short",),
            content_stats={"text_length": 64, "markdown_length": 70},
        ),
    )

    output = fetch_web_page(url="https://example.com/fallback")

    assert output["title"] == "通用回退测试"
    assert output["fetch_backend"] == "requests"
    assert "市场担忧增速放缓" in output["content"]
    assert diagnostics_payloads[0]["extraction_source"] == "bs_fallback"
    assert diagnostics_payloads[0]["renderer_source"] == "markdownify"
    assert diagnostics_payloads[0]["normalization_applied"] is True
    assert diagnostics_payloads[0]["internal_diagnostics"]["fetch_backend"] == "requests"
    assert diagnostics_payloads[0]["internal_diagnostics"]["quality_flags"] == ["too_short"]


@pytest.mark.unit
def test_fetch_web_page_returns_structured_failure_on_http_403(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 fetch_web_page 在 403 时返回结构化失败信息。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    http_error = requests.HTTPError("403")
    http_error.response = _FakeHttpResponse(
        status_code=403,
        headers={"Content-Type": "text/html", "cf-ray": "abc"},
        url="https://example.com/protected",
        text="forbidden",
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(http_error=http_error),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(status_code=403, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    # 确保 Playwright 回退也失败，以验证结构化失败信息的格式
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": False,
            "availability": "unprocessable",
            "reason": "playwright_error",
        },
    )

    with pytest.raises(ToolBusinessError) as exc_info:
        fetch_web_page(url="https://example.com/protected")

    assert exc_info.value.code == "blocked"
    assert "change_source" in exc_info.value.hint


@pytest.mark.unit
def test_fetch_web_page_calls_playwright_placeholder_on_http_403(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 403 分支会触发 Playwright 回退调用；Playwright 失败时工具返回 ok=False。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    http_error = requests.HTTPError("403")
    http_error.response = _FakeHttpResponse(
        status_code=403,
        headers={"Content-Type": "text/html"},
        url="https://example.com/protected",
        text="forbidden",
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(http_error=http_error),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(status_code=403, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)

    called = {"value": 0}

    def _fake_playwright_fallback(
        *,
        url: str,
        timeout_seconds: float,
        headers: Optional[dict[str, str]] = None,
        timeout_budget: float | None = None,
        deadline_monotonic: float | None = None,
        playwright_channel: str | None = None,
        playwright_storage_state_path: str = "",
    ) -> dict[str, Any]:
        _ = (
            url,
            timeout_seconds,
            headers,
            timeout_budget,
            deadline_monotonic,
            playwright_channel,
            playwright_storage_state_path,
        )
        called["value"] += 1
        return {"ok": False, "availability": "unprocessable", "reason": "playwright_error"}

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        _fake_playwright_fallback,
    )

    with pytest.raises(ToolBusinessError):
        fetch_web_page(url="https://example.com/protected")

    assert called["value"] == 1


@pytest.mark.unit
def test_fetch_web_page_normalizes_connection_wrapped_read_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证读超时被 `requests.ConnectionError` 包裹时仍归类为 timeout。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: _FakeSession())
    monkeypatch.setattr("dayu.engine.tools.web_tools._warmup_domain", lambda *args, **kwargs: {"attempted": False})
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {"ok": True, "content_type": "text/html"},
    )

    wrapped_timeout = _build_wrapped_read_timeout_connection_error()

    def _raise_wrapped_timeout(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = (args, kwargs)
        raise wrapped_timeout

    monkeypatch.setattr("dayu.engine.tools.web_tools._fetch_and_convert_content", _raise_wrapped_timeout)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": False,
            "availability": "timeout",
            "reason": "playwright_timeout",
        },
    )

    with pytest.raises(ToolBusinessError) as exc_info:
        fetch_web_page(url="https://example.com/timeout")

    assert exc_info.value.code == "request_timeout"


@pytest.mark.unit
def test_fetch_web_page_uses_playwright_fallback_on_connection_wrapped_read_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证包裹式读超时会触发浏览器回退，并在成功时返回正文。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: _FakeSession())
    monkeypatch.setattr("dayu.engine.tools.web_tools._warmup_domain", lambda *args, **kwargs: {"attempted": False})
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {"ok": True, "content_type": "text/html"},
    )

    wrapped_timeout = _build_wrapped_read_timeout_connection_error()

    def _raise_wrapped_timeout(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = (args, kwargs)
        raise wrapped_timeout

    monkeypatch.setattr("dayu.engine.tools.web_tools._fetch_and_convert_content", _raise_wrapped_timeout)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "Recovered By Browser",
            "content": "# Recovered\n\nBrowser fallback content.",
            "final_url": "https://example.com/timeout?browser=1",
        },
    )

    output = fetch_web_page(url="https://example.com/timeout")

    assert output["title"] == "Recovered By Browser"
    assert output["fetch_backend"] == "playwright"
    assert "Browser fallback content." in output["content"]
    assert output["final_url"] == "https://example.com/timeout?browser=1"


@pytest.mark.unit
def test_fetch_web_page_escalates_to_browser_after_warmup_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 warmup 暴露 timeout-like 信号时会优先尝试浏览器回退。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
        playwright_channel="chrome",
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: _FakeSession())
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._warmup_domain",
        lambda *args, **kwargs: {
            "attempted": True,
            "success": False,
            "timeout_like": True,
            "reason": "ConnectionError",
        },
    )

    probe_calls = {"value": 0}
    content_fetch_calls = {"value": 0}
    captured: dict[str, Any] = {}

    def _unexpected_probe(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = (args, kwargs)
        probe_calls["value"] += 1
        return {"ok": True, "content_type": "text/html"}

    def _unexpected_content_fetch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = (args, kwargs)
        content_fetch_calls["value"] += 1
        return {"content": "should not happen"}

    def _fake_playwright(
        *,
        url: str,
        timeout_seconds: float,
        headers: Optional[dict[str, str]] = None,
        timeout_budget: float | None = None,
        deadline_monotonic: float | None = None,
        playwright_channel: str | None = None,
        playwright_storage_state_path: str = "",
    ) -> dict[str, Any]:
        _ = (headers, timeout_budget, deadline_monotonic, playwright_storage_state_path)
        captured["url"] = url
        captured["timeout_seconds"] = timeout_seconds
        captured["playwright_channel"] = playwright_channel
        return {
            "ok": True,
            "title": "Warmup Browser",
            "content": "# Warmup Browser\n\nRecovered after warmup timeout.",
            "final_url": "https://example.com/warmup?browser=1",
        }

    monkeypatch.setattr("dayu.engine.tools.web_tools._probe_content_type", _unexpected_probe)
    monkeypatch.setattr("dayu.engine.tools.web_tools._fetch_and_convert_content", _unexpected_content_fetch)
    monkeypatch.setattr("dayu.engine.tools.web_tools._fetch_and_convert_with_playwright", _fake_playwright)

    output = fetch_web_page(url="https://example.com/warmup")

    assert output["title"] == "Warmup Browser"
    assert output["fetch_backend"] == "playwright"
    assert probe_calls["value"] == 0
    assert content_fetch_calls["value"] == 0
    assert captured["playwright_channel"] == "chrome"


@pytest.mark.unit
def test_fetch_web_page_escalates_to_browser_after_probe_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 probe 暴露 timeout-like 信号时会在正文下载前尝试浏览器回退。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
        playwright_channel="chrome",
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: _FakeSession())
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._warmup_domain",
        lambda *args, **kwargs: {"attempted": True, "success": True, "http_status": 200},
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {
            "attempted": True,
            "ok": False,
            "timeout_like": True,
            "head_error": "ReadTimeout",
            "get_error": "ReadTimeout",
        },
    )

    content_fetch_calls = {"value": 0}

    def _unexpected_content_fetch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = (args, kwargs)
        content_fetch_calls["value"] += 1
        return {"content": "should not happen"}

    monkeypatch.setattr("dayu.engine.tools.web_tools._fetch_and_convert_content", _unexpected_content_fetch)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "Probe Browser",
            "content": "# Probe Browser\n\nRecovered after probe timeout.",
            "final_url": "https://example.com/probe?browser=1",
        },
    )

    output = fetch_web_page(url="https://example.com/probe")

    assert output["title"] == "Probe Browser"
    assert output["fetch_backend"] == "playwright"
    assert content_fetch_calls["value"] == 0


@pytest.mark.unit
def test_fetch_web_page_returns_browser_fallback_for_challenge(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证命中挑战页信号时若 Playwright 也失败，工具返回 ok=False。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=b"<html><title>Just a moment...</title></html>",
            text="<html><title>Just a moment...</title></html>",
            status_code=200,
            headers={"Content-Type": "text/html", "cf-ray": "xyz", "set-cookie": "__cf_bm=token;"},
            url="https://example.com/challenge",
        ),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(status_code=200, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="Just a moment",
            markdown="# Just a moment\n\nChecking your browser before accessing",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=("challenge_like",),
            content_stats={"text_length": 44, "markdown_length": 56},
        ),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": False,
            "availability": "unprocessable",
            "reason": "playwright_error",
        },
    )

    with pytest.raises(ToolBusinessError) as exc_info:
        fetch_web_page(url="https://example.com/challenge")

    assert exc_info.value.code == "blocked"
    assert "change_source" in exc_info.value.hint


@pytest.mark.unit
def test_fetch_web_page_detects_raw_html_challenge_before_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证原始 HTML 命中挑战页特征时，会在主体抽取前直接升级到 Playwright。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=b"<html><head><meta name='aliyun_waf_action' content='captcha'></head></html>",
            text=(
                "<html><head><meta name='aliyun_waf_action' content='captcha'>"
                "<script src='/u21pn7x6/index.js'></script></head><body></body></html>"
            ),
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            url="https://xueqiu.com/demo",
        ),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
        ),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("原始挑战页不应进入主体抽取")),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "雪球正文",
            "content": "# 雪球正文\n\n浏览器成功拿到正文。",
            "final_url": "https://xueqiu.com/demo?browser=1",
        },
    )

    output = fetch_web_page(url="https://xueqiu.com/demo")

    assert output["title"] == "雪球正文"
    assert output["fetch_backend"] == "playwright"
    assert "浏览器成功拿到正文" in output["content"]
    assert output["final_url"] == "https://xueqiu.com/demo?browser=1"


@pytest.mark.unit
def test_fetch_web_page_retries_playwright_after_html_pipeline_error_with_raw_challenge_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 HTML 抽取失败后仍会基于原始响应上下文判定 challenge 并升级到 Playwright。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    registry = ToolRegistry()
    _, fetch_web_page, _ = web_tools_mod._create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: _FakeSession())
    monkeypatch.setattr("dayu.engine.tools.web_tools._warmup_domain", lambda *args, **kwargs: {"attempted": False})
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {
            "attempted": True,
            "ok": True,
            "content_type": "text/html; charset=utf-8",
            "http_status": 200,
            "final_url": "https://xueqiu.com/demo",
        },
    )

    def _raise_conversion_error(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = (args, kwargs)
        pipeline_error = HtmlPipelineStageError(
            "extract",
            "HTML 主体抽取失败：所有抽取器均未产出结果",
        )
        raise web_tools_mod._FetchContentConversionError(
            str(pipeline_error),
            response_context=web_tools_mod._FetchContentRuntimeContext(
                http_status=200,
                final_url="https://xueqiu.com/demo",
                response_headers={"content-type": "text/html; charset=utf-8"},
                response_excerpt="aliyun_waf_action captcha /u21pn7x6/",
                raw_content_text=(
                    "<html><head><meta name='aliyun_waf_action' content='captcha'>"
                    "<script src='/u21pn7x6/index.js'></script></head></html>"
                ),
            ),
            original_error=pipeline_error,
        )

    monkeypatch.setattr("dayu.engine.tools.web_tools._fetch_and_convert_content", _raise_conversion_error)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "恢复后的正文",
            "content": "# 恢复后的正文\n\n浏览器回退成功。",
            "final_url": "https://xueqiu.com/demo?browser=1",
        },
    )

    output = fetch_web_page(url="https://xueqiu.com/demo")

    assert output["title"] == "恢复后的正文"
    assert output["fetch_backend"] == "playwright"
    assert "浏览器回退成功" in output["content"]
    assert output["final_url"] == "https://xueqiu.com/demo?browser=1"


@pytest.mark.unit
def test_fetch_web_page_escalates_empty_html_shell_to_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 2xx HTML 壳页抽取失败时会升级到浏览器回退。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    registry = ToolRegistry()
    _, fetch_web_page, _ = web_tools_mod._create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: _FakeSession())
    monkeypatch.setattr("dayu.engine.tools.web_tools._warmup_domain", lambda *args, **kwargs: {"attempted": False})
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {
            "attempted": True,
            "ok": True,
            "content_type": "text/html; charset=utf-8",
            "http_status": 200,
            "final_url": "https://m.example.com/#/",
        },
    )

    def _raise_shell_extract_failure(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = (args, kwargs)
        pipeline_error = HtmlPipelineStageError(
            "extract",
            "HTML 主体抽取失败：正文为空",
            extractor_source="bs_fallback",
            quality_flags=("too_short", "too_few_blocks"),
            content_stats={"text_length": 0, "paragraph_count": 0},
        )
        raise web_tools_mod._FetchContentConversionError(
            str(pipeline_error),
            response_context=web_tools_mod._FetchContentRuntimeContext(
                http_status=200,
                final_url="https://m.example.com/#/",
                response_headers={"content-type": "text/html; charset=utf-8"},
                response_excerpt=(
                    "<!DOCTYPE html><html><head><title>示例站点</title></head>"
                    "<body><div id='app'></div><script src='/js/app.js'></script></body></html>"
                ),
                raw_content_text=(
                    "<!DOCTYPE html><html><head><title>示例站点</title></head>"
                    "<body><div id='app'></div><script src='/js/app.js'></script></body></html>"
                ),
            ),
            original_error=pipeline_error,
        )

    monkeypatch.setattr("dayu.engine.tools.web_tools._fetch_and_convert_content", _raise_shell_extract_failure)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "浏览器渲染后的正文",
            "content": "# 浏览器渲染后的正文\n\n这里是浏览器回退拿到的正文。",
            "final_url": "https://m.example.com/#/rendered",
        },
    )

    output = fetch_web_page(url="https://m.example.com/")

    assert output["title"] == "浏览器渲染后的正文"
    assert output["fetch_backend"] == "playwright"
    assert "浏览器回退拿到的正文" in output["content"]
    assert output["final_url"] == "https://m.example.com/#/rendered"


@pytest.mark.unit
def test_fetch_web_page_escalates_untyped_html_extract_failure_to_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证未类型化但明显是前端壳页的抽取失败也会升级到浏览器。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    registry = ToolRegistry()
    _, fetch_web_page, _ = web_tools_mod._create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: _FakeSession())
    monkeypatch.setattr("dayu.engine.tools.web_tools._warmup_domain", lambda *args, **kwargs: {"attempted": False})
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {
            "attempted": True,
            "ok": True,
            "content_type": "text/html; charset=utf-8",
            "http_status": 200,
            "final_url": "https://open.example.com/app",
        },
    )

    def _raise_untyped_conversion_error(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = (args, kwargs)
        raise web_tools_mod._FetchContentConversionError(
            "HTML 主体抽取失败：正文为空",
            response_context=web_tools_mod._FetchContentRuntimeContext(
                http_status=200,
                final_url="https://open.example.com/app",
                response_headers={"content-type": "text/html; charset=utf-8"},
                response_excerpt="<html><body><div id='root'></div></body></html>",
                raw_content_text=(
                    "<!DOCTYPE html><html><head><script src='/assets/app.js'></script></head>"
                    "<body><div id='root'></div></body></html>"
                ),
            ),
            original_error=RuntimeError("extract failed"),
        )

    monkeypatch.setattr("dayu.engine.tools.web_tools._fetch_and_convert_content", _raise_untyped_conversion_error)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "Rendered App",
            "content": "# Rendered App\n\nBrowser rendered content.",
            "final_url": "https://open.example.com/app?browser=1",
        },
    )

    output = fetch_web_page(url="https://open.example.com/app")

    assert output["title"] == "Rendered App"
    assert output["fetch_backend"] == "playwright"
    assert output["final_url"] == "https://open.example.com/app?browser=1"


@pytest.mark.unit
def test_detect_bot_challenge_ignores_cloudflare_infra_signals_on_article() -> None:
    """验证正常正文页不会仅因 Cloudflare 基础设施头被误判为挑战页。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    result = web_tools_mod._detect_bot_challenge(
        response=None,
        response_headers={
            "content-type": "text/html; charset=utf-8",
            "cf-ray": "abcd",
            "server": "cloudflare",
        },
        http_status=200,
        content_text="# Roblox\n\nRoblox bookings increased year over year and management updated guidance.",
    )

    assert result.challenge_detected is False
    assert result.challenge_signals == ("header:cf-ray", "server:cloudflare")


@pytest.mark.unit
def test_fetch_web_page_accepts_cloudflare_served_article(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 `fetch_web_page` 不会把 Cloudflare 正常文章页误判为 blocked。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=b"<html><title>Roblox</title></html>",
            text="<html><title>Roblox</title><body><article><p>Roblox bookings increased.</p></article></body></html>",
            status_code=200,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "cf-ray": "xyz",
                "server": "cloudflare",
            },
            url="https://example.com/roblox",
        ),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
        ),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="Roblox",
            markdown="# Roblox\n\nRoblox bookings increased year over year and margins improved.",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 72, "markdown_length": 79},
        ),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("不应升级到 Playwright 回退")),
    )

    output = fetch_web_page(url="https://example.com/roblox")

    assert output["title"] == "Roblox"
    assert output["fetch_backend"] == "requests"
    assert "bookings increased year over year" in output["content"]


@pytest.mark.unit
def test_detect_bot_challenge_ignores_datadome_cookie_without_other_signals() -> None:
    """验证单独的 DataDome cookie 信号不会把正常页面误判为 challenge。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    result = web_tools_mod._detect_bot_challenge(
        response=None,
        response_headers={
            "content-type": "text/html; charset=utf-8",
            "set-cookie": "datadome=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/; Secure",
        },
        http_status=200,
        content_text=(
            "# The New York Times\n\n"
            "Live news, investigations, opinion, photos and video from more than 150 countries."
        ),
    )

    assert result.challenge_detected is False
    assert result.challenge_signals == ("cookie:datadome",)


@pytest.mark.unit
def test_fetch_web_page_accepts_article_when_datadome_cookie_is_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证正常 HTML 页面不会因清理型 DataDome cookie 被误判为 blocked。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    html = (
        "<html><head><title>The New York Times</title></head>"
        "<body><article><p>Live news, investigations, opinion and analysis.</p></article></body></html>"
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=html.encode("utf-8"),
            text=html,
            status_code=200,
            headers={
                "Content-Type": "text/html; charset=utf-8",
                "Set-Cookie": "datadome=0; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/; Secure",
            },
            url="https://www.nytimes.com/",
        ),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            status_code=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            url="https://www.nytimes.com/",
        ),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="The New York Times",
            markdown="# The New York Times\n\nLive news, investigations, opinion and analysis.",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 52, "markdown_length": 71},
        ),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("不应升级到 Playwright 回退")),
    )

    output = fetch_web_page(url="https://www.nytimes.com/")

    assert output["title"] == "The New York Times"
    assert output["fetch_backend"] == "requests"
    assert "Live news, investigations" in output["content"]


@pytest.mark.unit
def test_fetch_web_page_treats_datadome_401_as_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 `401 + DataDome` 挑战页会被归类为 blocked，而不是 http_error。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    http_error = requests.HTTPError("401")
    http_error.response = _FakeHttpResponse(
        status_code=401,
        headers={
            "Content-Type": "text/html; charset=utf-8",
            "x-datadome": "protected",
            "x-dd-b": "3",
            "set-cookie": "datadome=token; Path=/; Secure",
        },
        url="https://www.reuters.com/article",
        text="<html><body>Please enable JS and disable any ad blocker</body></html>",
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(http_error=http_error),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(status_code=401, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": False,
            "availability": "blocked",
            "reason": "bot_challenge",
            "http_status": 401,
            "response_headers": {"x-datadome": "protected"},
            "response_excerpt": "Please enable JS and disable any ad blocker",
            "challenge_signals": ["header:x-datadome", "content:please enable js and disable any ad blocker"],
        },
    )

    with pytest.raises(ToolBusinessError) as exc_info:
        fetch_web_page(url="https://www.reuters.com/article")

    assert exc_info.value.code == "blocked"
    assert exc_info.value.extra["http_status"] == 401
    diagnostics = exc_info.value.extra["internal_diagnostics"]
    assert "header:x-datadome" in diagnostics["challenge_signals"]
    assert diagnostics["response_headers"]["x-datadome"] == "protected"


@pytest.mark.unit
def test_playwright_fallback_success_on_403(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 403 分支中 Playwright 回退成功时，工具返回 ok=True 的成功结果。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    http_error = requests.HTTPError("403")
    http_error.response = _FakeHttpResponse(
        status_code=403,
        headers={"Content-Type": "text/html"},
        url="https://example.com/protected",
        text="forbidden",
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(http_error=http_error),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(status_code=403, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "Protected Page",
            "content": "# Protected Page\n\nContent fetched via browser.",
            "final_url": "https://example.com/protected?browser=1",
        },
    )

    output = fetch_web_page(url="https://example.com/protected")

    assert output["title"] == "Protected Page"
    assert output["fetch_backend"] == "playwright"
    assert "Content fetched via browser" in output["content"]
    assert output["final_url"] == "https://example.com/protected?browser=1"


@pytest.mark.unit
def test_fetch_web_page_escalates_http_412_to_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 412 这类浏览器可恢复状态也会升级到 Playwright。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    http_error = requests.HTTPError("412")
    http_error.response = _FakeHttpResponse(
        status_code=412,
        headers={"Content-Type": "text/html", "server": "nginx"},
        url="https://example.com/protected",
        text="precondition failed",
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(http_error=http_error),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(status_code=412, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "Recovered 412",
            "content": "# Recovered 412\n\nBrowser fallback content.",
            "final_url": "https://example.com/protected?browser=1",
        },
    )

    output = fetch_web_page(url="https://example.com/protected")

    assert output["title"] == "Recovered 412"
    assert output["fetch_backend"] == "playwright"
    assert output["final_url"] == "https://example.com/protected?browser=1"


@pytest.mark.unit
def test_fetch_web_page_does_not_escalate_plain_401_to_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证普通 401 不会被新的泛化状态升级规则送进 Playwright。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    http_error = requests.HTTPError("401")
    http_error.response = _FakeHttpResponse(
        status_code=401,
        headers={"Content-Type": "text/html"},
        url="https://example.com/protected",
        text="unauthorized",
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(http_error=http_error),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(status_code=401, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("普通 401 不应触发 Playwright 回退")),
    )

    with pytest.raises(ToolBusinessError) as exc_info:
        fetch_web_page(url="https://example.com/protected")

    assert exc_info.value.code == "http_error"
    assert "change_source" in exc_info.value.hint


@pytest.mark.unit
def test_playwright_fallback_success_on_bot_challenge(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证抓取结果命中反爬挫战斶， Playwright 回退成功后返回 ok=True。"""

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(
            content=b"<html><title>Just a moment...</title></html>",
            text="<html><title>Just a moment...</title></html>",
            status_code=200,
            headers={"Content-Type": "text/html", "cf-ray": "xyz", "set-cookie": "__cf_bm=token;"},
            url="https://example.com/challenge",
        ),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(status_code=200, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="Just a moment",
            markdown="# Just a moment\n\nChecking your browser before accessing",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=("challenge_like",),
            content_stats={"text_length": 44, "markdown_length": 56},
        ),
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._fetch_and_convert_with_playwright",
        lambda *, url, timeout_seconds, headers=None, timeout_budget=None, deadline_monotonic=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "Real Page",
            "content": "# Real Content\n\nActual article text.",
            "final_url": "https://example.com/challenge?browser=1",
        },
    )

    output = fetch_web_page(url="https://example.com/challenge")

    assert output["title"] == "Real Page"
    assert output["fetch_backend"] == "playwright"
    assert "Actual article text" in output["content"]
    assert output["final_url"] == "https://example.com/challenge?browser=1"


@pytest.mark.unit
def test_fetch_and_convert_with_playwright_accepts_cloudflare_served_article(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证浏览器回退拿到正常正文时，不会仅因 Cloudflare 头部被判为 challenge。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._playwright_sync_worker",
        lambda *, url, timeout_seconds, headers=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "Roblox",
            "content": "# Roblox\n\nRoblox bookings increased year over year and margins improved.",
            "final_url": url,
            "http_status": 200,
            "response_headers": {
                "content-type": "text/html; charset=utf-8",
                "cf-ray": "xyz",
                "server": "cloudflare",
            },
            "response_excerpt": "Roblox bookings increased year over year and margins improved.",
        },
    )

    result = web_tools_mod._fetch_and_convert_with_playwright(
        url="https://example.com/roblox",
        timeout_seconds=10.0,
    )

    assert result["ok"] is True
    assert result["title"] == "Roblox"
    assert "margins improved" in result["content"]


@pytest.mark.unit
def test_playwright_fallback_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 playwright 未安装时，_fetch_and_convert_with_playwright 返回 availability=unprocessable。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    # 模拟 playwright 未安装：覆盖 __import__ 使 playwright 导入失败
    original_import = builtins.__import__

    def _mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "playwright":
            raise ImportError("No module named 'playwright'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)

    result = web_tools_mod._fetch_and_convert_with_playwright(
        url="https://example.com",
        timeout_seconds=10.0,
    )

    assert result["ok"] is False
    assert result["availability"] == "unprocessable"
    assert result["reason"] == "playwright_not_installed"


@pytest.mark.unit
def test_playwright_fallback_non_html_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Playwright 返回非 text/html 内容类型时，返回 availability=unprocessable。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._playwright_sync_worker",
        lambda *, url, timeout_seconds, headers=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": False,
            "availability": "unprocessable",
            "reason": "non_html_content_type",
            "content_type": "application/pdf",
        },
    )

    result = web_tools_mod._fetch_and_convert_with_playwright(
        url="https://example.com/report.pdf",
        timeout_seconds=10.0,
    )

    assert result["ok"] is False
    assert result["availability"] == "unprocessable"
    assert result["reason"] == "non_html_content_type"


@pytest.mark.unit
def test_playwright_sync_worker_returns_pipeline_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Playwright worker 会返回四段式 pipeline 元数据。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    fake_playwright_module = ModuleType("playwright")
    fake_sync_api_module = ModuleType("playwright.sync_api")

    class _FakePlaywrightTimeoutError(Exception):
        """模拟 Playwright TimeoutError。"""

    cast(Any, fake_sync_api_module).TimeoutError = _FakePlaywrightTimeoutError
    monkeypatch.setitem(sys.modules, "playwright", fake_playwright_module)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync_api_module)

    fake_stealth_module = ModuleType("playwright_stealth")

    class _FakeStealth:
        """模拟 stealth 注入器。"""

        def apply_stealth_sync(self, page: object) -> None:
            _ = page

    cast(Any, fake_stealth_module).Stealth = _FakeStealth
    monkeypatch.setitem(sys.modules, "playwright_stealth", fake_stealth_module)

    class _FakeResponse:
        """模拟 Playwright Response。"""

        headers = {"content-type": "text/html; charset=UTF-8"}

    class _FakePage:
        """模拟 Playwright Page。"""

        def __init__(self) -> None:
            self.url = "https://example.com/browser"
            self.goto_calls: list[str] = []

        def route(self, pattern: str, handler: Any) -> None:
            _ = (pattern, handler)

        def goto(self, url: str, wait_until: str, timeout: int) -> _FakeResponse:
            _ = (url, wait_until, timeout)
            self.goto_calls.append(url)
            return _FakeResponse()

        def wait_for_load_state(self, state: str, timeout: int) -> None:
            _ = (state, timeout)

        def wait_for_timeout(self, timeout_ms: int) -> None:
            _ = timeout_ms

        def content(self) -> str:
            return "<html><body><article>browser content</article></body></html>"

    class _FakeContext:
        """模拟 Playwright BrowserContext。"""

        def __init__(self) -> None:
            self.page = _FakePage()

        def new_page(self) -> _FakePage:
            return self.page

        def close(self) -> None:
            return None

    class _FakeBrowser:
        """模拟 Playwright Browser。"""

        def new_context(self, **kwargs: Any) -> _FakeContext:
            _ = kwargs
            return _FakeContext()

    monkeypatch.setattr("dayu.engine.tools.web_tools._get_playwright_browser", lambda **kwargs: _FakeBrowser())
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="Browser Title",
            markdown="# Browser Title\n\nBrowser content.",
            extractor_source="readability",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 15, "markdown_length": 33},
        ),
    )

    result = web_tools_mod._playwright_sync_worker(
        url="https://example.com/browser",
        timeout_seconds=5.0,
        playwright_channel="chrome",
    )

    assert result["ok"] is True
    assert result["title"] == "Browser Title"
    assert result["extraction_source"] == "readability"
    assert result["renderer_source"] == "markdownify"


@pytest.mark.unit
def test_playwright_sync_worker_uses_storage_state_when_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 Playwright worker 会把 storage state 文件传给 BrowserContext。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    storage_state_path = tmp_path / "storage-state.json"
    storage_state_path.write_text("{}", encoding="utf-8")

    fake_stealth_module = ModuleType("playwright_stealth")

    class _FakeStealth:
        """模拟 stealth 注入器。"""

        def apply_stealth_sync(self, page: object) -> None:
            _ = page

    cast(Any, fake_stealth_module).Stealth = _FakeStealth
    monkeypatch.setitem(sys.modules, "playwright_stealth", fake_stealth_module)

    class _FakeResponse:
        """模拟 Playwright Response。"""

        status = 200
        headers = {"content-type": "text/html; charset=UTF-8"}

    class _FakePage:
        """模拟 Playwright Page。"""

        def __init__(self) -> None:
            self.url = "https://example.com/browser"

        def route(self, pattern: str, handler: Any) -> None:
            _ = (pattern, handler)

        def goto(self, url: str, wait_until: str, timeout: int) -> _FakeResponse:
            _ = (url, wait_until, timeout)
            return _FakeResponse()

        def wait_for_timeout(self, timeout_ms: int) -> None:
            _ = timeout_ms

        def content(self) -> str:
            return "<html><body><article>browser content</article></body></html>"

        def evaluate(self, script: str) -> str:
            _ = script
            return "browser content"

    class _FakeContext:
        """模拟 Playwright BrowserContext。"""

        def __init__(self) -> None:
            self.page = _FakePage()

        def new_page(self) -> _FakePage:
            return self.page

        def close(self) -> None:
            return None

    captured: dict[str, Any] = {}

    class _FakeBrowser:
        """模拟 Playwright Browser。"""

        def new_context(self, **kwargs: Any) -> _FakeContext:
            captured.update(kwargs)
            return _FakeContext()

    monkeypatch.setattr("dayu.engine.tools.web_tools._get_playwright_browser", lambda **kwargs: _FakeBrowser())
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="Browser Title",
            markdown="# Browser Title\n\nBrowser content.",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 15, "markdown_length": 33},
        ),
    )

    result = web_tools_mod._playwright_sync_worker(
        url="https://example.com/browser",
        timeout_seconds=5.0,
        playwright_storage_state_path=str(storage_state_path),
    )

    assert result["ok"] is True
    assert captured["storage_state"] == str(storage_state_path)
    assert captured["ignore_https_errors"] is True
    assert captured["extra_http_headers"]["Upgrade-Insecure-Requests"] == "1"
    assert captured["extra_http_headers"]["Accept"]


@pytest.mark.unit
def test_playwright_sync_worker_uses_single_deadline_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Playwright worker 的 warmup/goto/settle 共享同一个总 deadline。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    fake_stealth_module = ModuleType("playwright_stealth")

    class _FakeStealth:
        """模拟 stealth 注入器。"""

        def apply_stealth_sync(self, page: object) -> None:
            _ = page

    cast(Any, fake_stealth_module).Stealth = _FakeStealth
    monkeypatch.setitem(sys.modules, "playwright_stealth", fake_stealth_module)

    class _FakeResponse:
        """模拟 Playwright Response。"""

        status = 200
        headers = {"content-type": "text/html; charset=UTF-8"}

    current_time = {"value": 100.0}
    recorded: dict[str, list[int]] = {"goto": [], "load_state": [], "wait": []}

    class _FakePage:
        """模拟带耗时推进的 Playwright Page。"""

        def __init__(self) -> None:
            self.url = "https://example.com/browser"

        def route(self, pattern: str, handler: Any) -> None:
            _ = (pattern, handler)

        def goto(self, url: str, wait_until: str, timeout: int) -> _FakeResponse:
            _ = wait_until
            recorded["goto"].append(timeout)
            if url == "https://example.com/":
                current_time["value"] += 0.4
            else:
                current_time["value"] += 0.35
            return _FakeResponse()

        def wait_for_load_state(self, state: str, timeout: int) -> None:
            recorded["load_state"].append(timeout)
            if state == "load":
                current_time["value"] += 0.1
            else:
                current_time["value"] += 0.05

        def wait_for_timeout(self, timeout_ms: int) -> None:
            recorded["wait"].append(timeout_ms)
            current_time["value"] += timeout_ms / 1000.0

        def content(self) -> str:
            return "<html><body><article>browser content</article></body></html>"

        def evaluate(self, script: str) -> str:
            _ = script
            return "browser content"

    class _FakeContext:
        """模拟 Playwright BrowserContext。"""

        def __init__(self) -> None:
            self.page = _FakePage()

        def new_page(self) -> _FakePage:
            return self.page

        def close(self) -> None:
            return None

    class _FakeBrowser:
        """模拟 Playwright Browser。"""

        def new_context(self, **kwargs: Any) -> _FakeContext:
            _ = kwargs
            return _FakeContext()

    monkeypatch.setattr("dayu.engine.tools.web_tools.time.monotonic", lambda: current_time["value"])
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_playwright_browser", lambda **kwargs: _FakeBrowser())
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools.convert_html_to_llm_markdown",
        lambda html, url="": SimpleNamespace(
            title="Browser Title",
            markdown="# Browser Title\n\nBrowser content.",
            extractor_source="trafilatura",
            renderer_source="markdownify",
            normalization_applied=True,
            quality_flags=(),
            content_stats={"text_length": 15, "markdown_length": 33},
        ),
    )

    result = web_tools_mod._playwright_sync_worker(
        url="https://example.com/browser",
        timeout_seconds=1.0,
    )

    assert result["ok"] is True
    assert recorded["goto"] == [1000, 600]
    assert recorded["load_state"][0] == 250
    assert recorded["load_state"][1] in {150, 151}
    assert recorded["wait"][0] in {100, 101}


@pytest.mark.unit
def test_fetch_web_page_resolves_storage_state_by_host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 fetch_web_page 会按 host 及 www 变体解析 storage state 文件。"""

    storage_dir = tmp_path / "storage_states"
    storage_dir.mkdir(parents=True, exist_ok=True)
    host_state = storage_dir / "www.reuters.com.json"
    host_state.write_text("{}", encoding="utf-8")

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
        playwright_storage_state_dir=str(storage_dir),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: _FakeSession())
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._warmup_domain",
        lambda *args, **kwargs: {"attempted": True, "success": False, "timeout_like": True},
    )

    captured: dict[str, Any] = {}

    def _fake_playwright(
        *,
        url: str,
        timeout_seconds: float,
        headers: Optional[dict[str, str]] = None,
        timeout_budget: float | None = None,
        deadline_monotonic: float | None = None,
        playwright_channel: str | None = None,
        playwright_storage_state_path: str = "",
    ) -> dict[str, Any]:
        _ = (url, timeout_seconds, headers, timeout_budget, deadline_monotonic, playwright_channel)
        captured["playwright_storage_state_path"] = playwright_storage_state_path
        return {
            "ok": True,
            "title": "Recovered",
            "content": "# Recovered",
            "final_url": "https://www.reuters.com/article",
        }

    monkeypatch.setattr("dayu.engine.tools.web_tools._fetch_and_convert_with_playwright", _fake_playwright)

    fetch_web_page(url="https://reuters.com/article")

    assert captured["playwright_storage_state_path"] == str(host_state)


@pytest.mark.unit
def test_fetch_web_page_applies_storage_state_cookies_to_requests_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 fetch_web_page 会把 storage state cookie 注入 requests 主路径。"""

    storage_dir = tmp_path / "storage_states"
    storage_dir.mkdir(parents=True, exist_ok=True)
    state_path = storage_dir / "www.bloomberg.com.json"
    state_path.write_text(
        '{"cookies": [{"name": "session_key", "value": "abc123", "domain": ".bloomberg.com", "path": "/", "secure": true}], "origins": []}',
        encoding="utf-8",
    )

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
        playwright_storage_state_dir=str(storage_dir),
        timeout_budget=10.0,
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", requests.Session)
    no_retry_session = requests.Session()
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_no_retry_web_session", lambda: no_retry_session)
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._warmup_domain",
        lambda *args, **kwargs: {"attempted": True, "success": True, "http_status": 200},
    )
    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._probe_content_type",
        lambda *args, **kwargs: {"attempted": True, "ok": True, "content_type": "text/html; charset=utf-8"},
    )

    captured: dict[str, Any] = {}

    def _fake_fetch_and_convert_content(
        url: str,
        *,
        timeout_seconds: float,
        session: Optional[requests.Session] = None,
        headers: Optional[dict[str, str]] = None,
        content_type_probe: Optional[dict[str, Any]] = None,
        timeout_budget: float | None = None,
        deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        _ = (headers, content_type_probe, timeout_budget, deadline_monotonic)
        assert isinstance(session, requests.Session)
        captured["url"] = url
        captured["timeout_seconds"] = timeout_seconds
        captured["session_key"] = session.cookies.get("session_key", domain=".bloomberg.com", path="/")
        return {
            "title": "Markets - Bloomberg",
            "content": "# Markets - Bloomberg\n\nRecovered via requests cookie state.",
            "extraction_source": "trafilatura",
            "renderer_source": "markdownify",
            "normalization_applied": True,
            "quality_flags": [],
            "content_stats": {"text_length": 36, "markdown_length": 58},
            "http_status": 200,
            "final_url": url,
            "redirect_hops": 0,
            "response": _FakeHttpResponse(status_code=200, url=url),
            "response_headers": {"content-type": "text/html; charset=utf-8"},
            "response_excerpt": "Markets - Bloomberg",
        }

    monkeypatch.setattr("dayu.engine.tools.web_tools._fetch_and_convert_content", _fake_fetch_and_convert_content)

    output = fetch_web_page(url="https://www.bloomberg.com/markets")

    assert output["title"] == "Markets - Bloomberg"
    assert output["fetch_backend"] == "requests"
    assert captured["session_key"] == "abc123"


@pytest.mark.unit
def test_fetch_and_convert_with_playwright_rejects_challenge_page(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证浏览器回退若拿到挑战页，不会被误判为正文成功。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    monkeypatch.setattr(
        "dayu.engine.tools.web_tools._playwright_sync_worker",
        lambda *, url, timeout_seconds, headers=None, playwright_channel=None, playwright_storage_state_path="": {
            "ok": True,
            "title": "reuters.com",
            "content": "# reuters.com\n\nPlease enable JS and disable any ad blocker",
            "final_url": "https://www.reuters.com/article",
            "http_status": 401,
            "response_headers": {
                "x-datadome": "protected",
                "x-dd-b": "3",
                "set-cookie": "datadome",
            },
            "response_excerpt": "Please enable JS and disable any ad blocker",
        },
    )

    result = web_tools_mod._fetch_and_convert_with_playwright(
        url="https://www.reuters.com/article",
        timeout_seconds=10.0,
    )

    assert result["ok"] is False
    assert result["availability"] == "blocked"
    assert result["reason"] == "bot_challenge"
    assert "header:x-datadome" in result["challenge_signals"]


@pytest.mark.unit
def test_run_playwright_worker_process_consumes_queue_before_worker_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证父进程会先消费结果队列，不要求 worker 先完全退出。"""

    import dayu.engine.tools.web_playwright_backend as backend_mod

    state: dict[str, bool] = {"alive": True}

    class _FakeQueue:
        def __init__(self) -> None:
            self._consumed = False

        def get(self, timeout: float) -> dict[str, Any]:
            _ = timeout
            self._consumed = True
            state["alive"] = False
            return {
                "kind": "result",
                "payload": {"ok": True, "title": "Recovered", "content": "# Recovered"},
            }

        def get_nowait(self) -> dict[str, Any]:
            raise AssertionError("该场景不应走 get_nowait")

        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

    class _FakeProcess:
        def __init__(self) -> None:
            self.daemon = False

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return state["alive"]

        def join(self, timeout: float = 0) -> None:
            _ = timeout
            return None

        def terminate(self) -> None:
            state["alive"] = False

        def kill(self) -> None:
            state["alive"] = False

    class _FakeContext:
        def __init__(self) -> None:
            self.queue = _FakeQueue()
            self.process = _FakeProcess()

        def Queue(self, maxsize: int = 1) -> _FakeQueue:
            _ = maxsize
            return self.queue

        def Process(self, target: Any, args: tuple[Any, ...]) -> _FakeProcess:
            _ = (target, args)
            return self.process

    fake_context = _FakeContext()
    monkeypatch.setattr(backend_mod.multiprocessing, "get_context", lambda method: fake_context)

    result = backend_mod._run_playwright_worker_process(
        playwright_sync_worker=lambda **kwargs: {"ok": True, **kwargs},
        worker_kwargs={"url": "https://example.com"},
        total_timeout=1.0,
        cancellation_token=None,
    )

    assert result["ok"] is True
    assert fake_context.queue._consumed is True


@pytest.mark.unit
def test_run_playwright_worker_process_allows_queue_drain_after_worker_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 worker 已退出后，父进程仍会给结果队列一个短暂 drain 窗口。"""

    import dayu.engine.tools.web_playwright_backend as backend_mod

    state: dict[str, int] = {"poll_count": 0}

    class _FakeQueue:
        def get(self, timeout: float) -> dict[str, Any]:
            _ = timeout
            state["poll_count"] += 1
            if state["poll_count"] == 1:
                raise Empty()
            return {
                "kind": "result",
                "payload": {"ok": True, "title": "Delayed", "content": "# Delayed"},
            }

        def get_nowait(self) -> dict[str, Any]:
            raise AssertionError("修复后不应依赖 get_nowait 抢最后一条结果")

        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

    class _FakeProcess:
        def __init__(self) -> None:
            self.daemon = False

        def start(self) -> None:
            return None

        def is_alive(self) -> bool:
            return state["poll_count"] == 0

        def join(self, timeout: float = 0) -> None:
            _ = timeout
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    class _FakeContext:
        def __init__(self) -> None:
            self.queue = _FakeQueue()
            self.process = _FakeProcess()

        def Queue(self, maxsize: int = 1) -> _FakeQueue:
            _ = maxsize
            return self.queue

        def Process(self, target: Any, args: tuple[Any, ...]) -> _FakeProcess:
            _ = (target, args)
            return self.process

    current_time = {"value": 100.0}

    def _fake_monotonic() -> float:
        current_time["value"] += 0.05
        return current_time["value"]

    fake_context = _FakeContext()
    monkeypatch.setattr(backend_mod.multiprocessing, "get_context", lambda method: fake_context)
    monkeypatch.setattr(backend_mod.time, "monotonic", _fake_monotonic)

    result = backend_mod._run_playwright_worker_process(
        playwright_sync_worker=lambda **kwargs: {"ok": True, **kwargs},
        worker_kwargs={"url": "https://example.com"},
        total_timeout=1.0,
        cancellation_token=None,
    )

    assert result["ok"] is True
    assert state["poll_count"] == 2


@pytest.mark.unit
def test_fetch_web_page_failure_fields_use_stable_enums(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证失败返回中的 next_action/reason 始终落在稳定枚举集合。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    http_error = requests.HTTPError("500")
    http_error.response = _FakeHttpResponse(
        status_code=500,
        headers={"Content-Type": "text/html"},
        url="https://example.com/error",
        text="server error",
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(http_error=http_error),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(status_code=500, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)

    with pytest.raises(ToolBusinessError) as exc_info:
        fetch_web_page(url="https://example.com/error")

    # hint 中嵌入的 next_action 标签属于稳定枚举
    err = exc_info.value
    assert err.code in {
        "request_timeout",
        "http_error",
        "blocked",
        "redirect_chain_too_long",
        "content_conversion_failed",
        "empty_content",
    }
    # hint 格式为 "[next_action] 提示文案"
    assert any(action in err.hint for action in {"retry", "change_source", "continue_without_web"})
    assert "目标：" in err.hint
    assert "允许动作：" in err.hint
    assert "下一步：" in err.hint


@pytest.mark.unit
def test_fetch_web_page_retry_after_seconds_from_response_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证可重试错误会返回 retry_after_seconds。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    registry = ToolRegistry()
    _, fetch_web_page, _ = _create_fetch_web_page_tool(
        registry,
        request_timeout_seconds=12.0,
        fetch_truncate_chars=80000,
    )
    http_error = requests.HTTPError("429")
    http_error.response = _FakeHttpResponse(
        status_code=429,
        headers={"Content-Type": "text/html", "Retry-After": "15"},
        url="https://example.com/rate-limit",
        text="rate limited",
    )
    fake_session = _FakeSession(
        get_handler=lambda _url, **_kwargs: _FakeHttpResponse(http_error=http_error),
        head_handler=lambda _url, **_kwargs: _FakeHttpResponse(status_code=429, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("dayu.engine.tools.web_tools._get_web_session", lambda: fake_session)

    with pytest.raises(ToolBusinessError) as exc_info:
        fetch_web_page(url="https://example.com/rate-limit")

    assert exc_info.value.code == "http_error"
    assert "retry" in exc_info.value.hint


# ---------------------------------------------------------------------------
# Challenge detection 增强覆盖
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_detect_bot_challenge_bloomberg_are_you_a_robot() -> None:
    """验证 Bloomberg 'Are you a robot?' challenge 页面被正确检测。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    result = web_tools_mod._detect_bot_challenge(
        response=None,
        response_headers={"Content-Type": "text/html"},
        http_status=403,
        content_text='<title>Bloomberg - Are you a robot?</title><p>click the box below to let us know you\'re not a robot</p>',
    )

    assert result.challenge_detected is True
    assert any("are you a robot" in s for s in result.challenge_signals)


@pytest.mark.unit
def test_detect_bot_challenge_cloudflare_chinese_variant() -> None:
    """验证 Cloudflare 中文验证页被正确检测。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    result = web_tools_mod._detect_bot_challenge(
        response=None,
        response_headers={"cf-ray": "abc123", "server": "cloudflare"},
        http_status=403,
        content_text="# 请稍候…\n\n## 执行安全验证\n\n此网站使用安全服务来防范恶意自动程序。",
    )

    assert result.challenge_detected is True
    content_signals = [s for s in result.challenge_signals if s.startswith("content:")]
    assert len(content_signals) >= 1


@pytest.mark.unit
def test_detect_bot_challenge_not_a_robot_pattern() -> None:
    """验证 'not a robot' 模式被检测为 challenge。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    result = web_tools_mod._detect_bot_challenge(
        response=None,
        response_headers={},
        http_status=403,
        content_text="Please confirm you're not a robot by clicking below.",
    )

    assert result.challenge_detected is True


@pytest.mark.unit
def test_detect_bot_challenge_unusual_activity_pattern() -> None:
    """验证 'unusual activity' 模式被检测为 challenge。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    result = web_tools_mod._detect_bot_challenge(
        response=None,
        response_headers={},
        http_status=403,
        content_text="We've detected unusual activity from your computer network.",
    )

    assert result.challenge_detected is True


@pytest.mark.unit
def test_detect_bot_challenge_challenges_cloudflare_domain() -> None:
    """验证包含 challenges.cloudflare.com 的页面被检测为 challenge。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    result = web_tools_mod._detect_bot_challenge(
        response=None,
        response_headers={"cf-ray": "abc", "server": "cloudflare"},
        http_status=403,
        content_text='<iframe src="https://challenges.cloudflare.com/cdn-cgi/challenge-platform/turnstile">',
    )

    assert result.challenge_detected is True


@pytest.mark.unit
def test_normal_article_not_false_positive_on_cloudflare() -> None:
    """验证 Cloudflare 正常文章页面不会因 infra signals 被误判为 challenge。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    result = web_tools_mod._detect_bot_challenge(
        response=None,
        response_headers={"cf-ray": "abc", "server": "cloudflare"},
        http_status=200,
        content_text=(
            "# Apple Q3 2025 Earnings Report\n\n"
            "Revenue reached $94.8 billion, a 5% increase year over year. "
            "iPhone revenue was $46.2 billion."
        ),
    )

    assert result.challenge_detected is False


@pytest.mark.unit
def test_build_fetch_headers_includes_client_hints() -> None:
    """验证非 SEC 网站的请求头包含现代 Chrome 指纹 headers。"""

    headers = _build_fetch_headers("https://finance.yahoo.com/quote/AAPL/")

    assert "Sec-Ch-Ua" in headers
    assert "Sec-Ch-Ua-Mobile" in headers
    assert "Sec-Ch-Ua-Platform" in headers
    assert "Sec-Fetch-Dest" in headers
    assert headers["Sec-Fetch-Dest"] == "document"
    assert "Sec-Fetch-Mode" in headers
    assert headers["Sec-Fetch-Mode"] == "navigate"
    assert "Upgrade-Insecure-Requests" in headers
    assert headers["Upgrade-Insecure-Requests"] == "1"


@pytest.mark.unit
def test_build_fetch_headers_sec_host_no_client_hints() -> None:
    """验证 SEC 网站的请求头不包含浏览器指纹 headers。"""

    headers = _build_fetch_headers("https://www.sec.gov/cgi-bin/browse-edgar")

    assert "Sec-Ch-Ua" not in headers
    assert "Sec-Fetch-Dest" not in headers
    assert "Upgrade-Insecure-Requests" not in headers


# ---------- 中国 WAF / 反爬 pattern 测试 ----------


@pytest.mark.unit
def test_detect_bot_challenge_catches_eo_bot() -> None:
    """验证 EO_Bot 类中国 WAF 挑战页被检测为 challenge。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    html = (
        "<script>function a(a){var t='EO_Bot_Ssid=';t+=2703425536;"
        "document.cookie=t;location.reload();}</script>"
    )
    result = web_tools_mod._detect_bot_challenge(
        response=None, content_text=html, http_status=200,
    )

    assert result.challenge_detected is True
    assert any("eo_bot" in s for s in result.challenge_signals)


@pytest.mark.unit
def test_detect_bot_challenge_catches_antibot() -> None:
    """验证带 antibot 关键词的页面被检测为 challenge。"""

    import dayu.engine.tools.web_tools as web_tools_mod

    html = '<script src="/antibot/check.js"></script><div>验证中</div>'
    result = web_tools_mod._detect_bot_challenge(
        response=None, content_text=html, http_status=200,
    )

    assert result.challenge_detected is True
    assert any("antibot" in s for s in result.challenge_signals)
