"""联网检索 provider 与结果组装。

本模块只负责 `search_web` 的 provider 选择、请求发送、结果解析与
返回结果组装，不承载网页抓取、HTML 转换或浏览器回退逻辑。
"""

from __future__ import annotations

import os
from typing import Callable, NotRequired, Optional, Protocol, TypedDict
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from dayu.contracts.env_keys import SERPER_API_KEY_ENV, TAVILY_API_KEY_ENV
from dayu.log import Log

MODULE = "ENGINE.WEB_SEARCH"

_SEARCH_WEB_SNIPPET_PREVIEW_CHARS = 240
_SEARCH_WEB_NEXT_ACTION_FETCH_PAGE = "fetch_web_page"
_SEARCH_WEB_NEXT_ACTION_REFINE_QUERY = "refine_query"


class SearchResultRow(TypedDict):
    """联网检索单条结果。"""

    title: str
    url: str
    snippet: str
    published_date: str


class SearchWebOutput(TypedDict):
    """`search_web` 对外返回结构。"""

    query: str
    domains: list[str]
    total: int
    preferred_result: SearchResultRow | None
    preferred_result_summary: str
    next_action: str
    next_action_args: dict[str, str]
    hint: str
    results: list[SearchResultRow]


class TavilyResultItem(TypedDict):
    """Tavily 响应结果项。"""

    title: NotRequired[str]
    url: NotRequired[str]
    content: NotRequired[str]
    published_date: NotRequired[str]


class TavilyResponsePayload(TypedDict):
    """Tavily 响应载荷。"""

    results: NotRequired[list[TavilyResultItem]]


class SerperOrganicItem(TypedDict):
    """Serper organic 结果项。"""

    title: NotRequired[str]
    link: NotRequired[str]
    snippet: NotRequired[str]


class SerperResponsePayload(TypedDict):
    """Serper 响应载荷。"""

    organic: NotRequired[list[SerperOrganicItem]]


class _TimeoutBudgetResolver(Protocol):
    """搜索请求 timeout 解析协议。"""

    def __call__(
        self,
        timeout_seconds: float,
        *,
        timeout_budget: float | None = None,
        deadline_monotonic: float | None = None,
    ) -> float:
        """解析当前请求可用 timeout。"""

        ...


class _PublicUrlSafetyChecker(Protocol):
    """公网 URL 安全校验协议。"""

    def __call__(self, url: str, *, allow_private_network_url: bool = False) -> bool:
        """判断 URL 是否允许暴露给上层。"""

        ...


def search_public_web(
    *,
    query: str,
    domains: Optional[list[str]],
    recency_days: Optional[int],
    max_results: int,
    max_search_results: int,
    provider: str,
    request_timeout_seconds: float,
    timeout_budget: float | None,
    deadline_monotonic: float | None,
    allow_private_network_url: bool,
    is_safe_public_url: _PublicUrlSafetyChecker,
    normalize_whitespace: Callable[[str], str],
    resolve_timeout_budget: _TimeoutBudgetResolver,
) -> SearchWebOutput:
    """执行公开网页检索并组装 tool 输出。

    Args:
        query: 原始查询文本。
        domains: 原始域名限制列表。
        recency_days: 最近天数限制。
        max_results: 当前调用声明的结果上限。
        max_search_results: tool 注册时声明的最大结果上限。
        provider: provider 选择策略。
        request_timeout_seconds: provider 请求超时秒数。
        timeout_budget: 单次 tool call 总预算。
        deadline_monotonic: 当前调用 deadline。
        allow_private_network_url: 是否允许保留私网 URL 结果。
        is_safe_public_url: 公网 URL 安全校验函数。
        normalize_whitespace: 文本空白规整函数。
        resolve_timeout_budget: timeout 预算解析函数。

    Returns:
        `search_web` 对外返回字典。

    Raises:
        ValueError: 当 query 或 domains 非法时抛出。
        RuntimeError: 当所有 provider 都失败时抛出。
    """

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query 不能为空")

    normalized_domains = _normalize_domains(domains)
    limited_results = max(1, min(int(max_results), max_search_results))
    resolved_provider = _resolve_provider(preferred=provider)

    for candidate_provider in _candidate_providers(resolved_provider):
        try:
            if candidate_provider == "tavily":
                rows = _search_with_tavily(
                    query=normalized_query,
                    domains=normalized_domains,
                    recency_days=recency_days,
                    max_results=limited_results,
                    timeout_seconds=request_timeout_seconds,
                    timeout_budget=timeout_budget,
                    deadline_monotonic=deadline_monotonic,
                    resolve_timeout_budget=resolve_timeout_budget,
                )
            elif candidate_provider == "serper":
                rows = _search_with_serper(
                    query=normalized_query,
                    domains=normalized_domains,
                    recency_days=recency_days,
                    max_results=limited_results,
                    timeout_seconds=request_timeout_seconds,
                    timeout_budget=timeout_budget,
                    deadline_monotonic=deadline_monotonic,
                    resolve_timeout_budget=resolve_timeout_budget,
                )
            else:
                rows = _search_with_duckduckgo(
                    query=normalized_query,
                    domains=normalized_domains,
                    max_results=limited_results,
                    timeout_seconds=request_timeout_seconds,
                    timeout_budget=timeout_budget,
                    deadline_monotonic=deadline_monotonic,
                    normalize_whitespace=normalize_whitespace,
                    resolve_timeout_budget=resolve_timeout_budget,
                )
        except Exception as exc:  # pragma: no cover - 失败路径由单测通过 monkeypatch 覆盖
            _log_search_provider_failure(
                candidate_provider=candidate_provider,
                error=exc,
            )
            continue

        visible_results = _filter_visible_results(
            rows=rows,
            allow_private_network_url=allow_private_network_url,
            is_safe_public_url=is_safe_public_url,
        )[:limited_results]
        preferred_result = _build_search_web_preferred_result(visible_results)
        return {
            "query": normalized_query,
            "domains": normalized_domains,
            "total": len(visible_results),
            "preferred_result": preferred_result,
            "preferred_result_summary": _build_search_web_preferred_summary(
                preferred_result=preferred_result,
                normalize_whitespace=normalize_whitespace,
            ),
            "next_action": _build_search_web_next_action(preferred_result=preferred_result),
            "next_action_args": _build_search_web_next_action_args(preferred_result=preferred_result),
            "hint": _build_search_web_hint(
                preferred_result=preferred_result,
                normalize_whitespace=normalize_whitespace,
            ),
            "results": visible_results,
        }

    raise RuntimeError("联网检索失败：所有 provider 均不可用")


def _filter_visible_results(
    *,
    rows: list[SearchResultRow],
    allow_private_network_url: bool,
    is_safe_public_url: _PublicUrlSafetyChecker,
) -> list[SearchResultRow]:
    """按 URL 安全策略过滤 provider 结果。

    Args:
        rows: provider 返回的原始结果列表。
        allow_private_network_url: 是否允许保留私网 URL。
        is_safe_public_url: 公网 URL 安全校验函数。

    Returns:
        通过安全过滤的结果列表。

    Raises:
        无。
    """

    return [
        row
        for row in rows
        if is_safe_public_url(
            row["url"],
            allow_private_network_url=allow_private_network_url,
        )
    ]


def _default_resolve_timeout_budget(
    timeout_seconds: float,
    *,
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
) -> float:
    """默认 timeout 解析函数。

    Args:
        timeout_seconds: 调用方声明的基础 timeout。
        timeout_budget: 未使用的总预算参数。
        deadline_monotonic: 未使用的 deadline 参数。

    Returns:
        至少为 1 秒的 timeout。

    Raises:
        无。
    """

    _ = (timeout_budget, deadline_monotonic)
    return max(1.0, float(timeout_seconds))


def _get_search_provider_api_key_env_name(provider: str) -> str | None:
    """返回联网检索 provider 对应的 API key 环境变量名。

    Args:
        provider: 已归一化的联网检索 provider。

    Returns:
        若该 provider 依赖 API key，则返回环境变量名；否则返回 `None`。

    Raises:
        无。
    """

    if provider == "tavily":
        return TAVILY_API_KEY_ENV
    if provider == "serper":
        return SERPER_API_KEY_ENV
    return None


def _has_configured_search_provider_api_key(provider: str) -> bool:
    """判断联网检索 provider 是否已配置可用 API key。

    Args:
        provider: 已归一化的联网检索 provider。

    Returns:
        若 provider 不依赖 API key，或已配置非空 API key，则返回 `True`；否则返回 `False`。

    Raises:
        无。
    """

    env_name = _get_search_provider_api_key_env_name(provider)
    if not env_name:
        return True
    return bool(os.environ.get(env_name, "").strip())


def _log_search_provider_failure(
    *,
    candidate_provider: str,
    error: Exception,
) -> None:
    """记录联网检索 provider 的真实失败日志。

    Args:
        candidate_provider: 当前尝试的 provider。
        error: provider 抛出的异常对象。

    Returns:
        无。

    Raises:
        无。
    """

    Log.warn(f"provider={candidate_provider} 检索失败: {error}", module=MODULE)


def _normalize_domains(domains: Optional[list[str]]) -> list[str]:
    """归一化域名过滤列表。

    Args:
        domains: 原始域名列表。

    Returns:
        归一化后的域名列表。

    Raises:
        ValueError: 当域名元素非法时抛出。
    """

    if domains is None:
        return []
    normalized: list[str] = []
    for item in domains:
        if not isinstance(item, str):
            raise ValueError("domains 元素必须是字符串")
        value = item.strip().lower()
        if not value:
            continue
        normalized.append(value)
    return normalized


def _resolve_provider(*, preferred: str) -> str:
    """解析 provider 策略。

    Args:
        preferred: 首选 provider。

    Returns:
        规范化 provider 名称。

    Raises:
        ValueError: 当 provider 非法时抛出。
    """

    normalized = preferred.strip().lower() if isinstance(preferred, str) else "auto"
    allowed = {"auto", "tavily", "serper", "duckduckgo"}
    if normalized not in allowed:
        raise ValueError(f"不支持的 web provider: {preferred}")
    return normalized


def _candidate_providers(provider: str) -> list[str]:
    """获取 provider 候选顺序。

    Args:
        provider: 已归一化 provider。

    Returns:
        候选 provider 列表。

    Raises:
        无。
    """

    if provider == "auto":
        candidates: list[str] = []
        if _has_configured_search_provider_api_key("tavily"):
            candidates.append("tavily")
        if _has_configured_search_provider_api_key("serper"):
            candidates.append("serper")
        candidates.append("duckduckgo")
        return candidates
    return [provider, "duckduckgo"] if provider in {"tavily", "serper"} else ["duckduckgo"]


def _search_with_tavily(
    *,
    query: str,
    domains: list[str],
    recency_days: Optional[int],
    max_results: int,
    timeout_seconds: float,
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    resolve_timeout_budget: _TimeoutBudgetResolver = _default_resolve_timeout_budget,
) -> list[SearchResultRow]:
    """使用 Tavily API 搜索。

    Args:
        query: 检索关键词。
        domains: 域名过滤。
        recency_days: 最近天数。
        max_results: 返回数量。
        timeout_seconds: HTTP 请求超时秒数。
        timeout_budget: Runner 注入的单次 tool call 总预算。
        deadline_monotonic: 当前工具调用的单调时钟 deadline。
        resolve_timeout_budget: timeout 预算解析函数。

    Returns:
        结果列表。

    Raises:
        RuntimeError: 当 key 缺失或请求失败时抛出。
    """

    api_key = os.environ.get(TAVILY_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY 未配置")

    payload: dict[str, str | int | list[str]] = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
    }
    if domains:
        payload["include_domains"] = domains
    if recency_days is not None and recency_days >= 0:
        payload["days"] = int(recency_days)

    response = requests.post(
        "https://api.tavily.com/search",
        json=payload,
        timeout=resolve_timeout_budget(
            timeout_seconds,
            timeout_budget=timeout_budget,
            deadline_monotonic=deadline_monotonic,
        ),
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        return []

    rows: list[SearchResultRow] = []
    for item in data.get("results", []) or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "snippet": str(item.get("content", "")).strip(),
                "published_date": str(item.get("published_date", "")).strip(),
            }
        )
    return rows


def _search_with_serper(
    *,
    query: str,
    domains: list[str],
    recency_days: Optional[int],
    max_results: int,
    timeout_seconds: float,
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    resolve_timeout_budget: _TimeoutBudgetResolver = _default_resolve_timeout_budget,
) -> list[SearchResultRow]:
    """使用 Serper API 搜索。

    Args:
        query: 检索关键词。
        domains: 域名过滤。
        recency_days: 最近天数。
        max_results: 返回数量。
        timeout_seconds: HTTP 请求超时秒数。
        timeout_budget: Runner 注入的单次 tool call 总预算。
        deadline_monotonic: 当前工具调用的单调时钟 deadline。
        resolve_timeout_budget: timeout 预算解析函数。

    Returns:
        结果列表。

    Raises:
        RuntimeError: 当 key 缺失或请求失败时抛出。
    """

    api_key = os.environ.get(SERPER_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError("SERPER_API_KEY 未配置")

    query_with_domain = query
    if domains:
        domain_expr = " OR ".join(f"site:{domain}" for domain in domains)
        query_with_domain = f"({query}) ({domain_expr})"

    payload: dict[str, str | int] = {
        "q": query_with_domain,
        "num": max_results,
    }
    if recency_days is not None and recency_days >= 0:
        payload["tbs"] = f"qdr:d{int(recency_days)}"

    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
        json=payload,
        timeout=resolve_timeout_budget(
            timeout_seconds,
            timeout_budget=timeout_budget,
            deadline_monotonic=deadline_monotonic,
        ),
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        return []

    rows: list[SearchResultRow] = []
    for item in data.get("organic", []) or []:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("link", "")).strip(),
                "snippet": str(item.get("snippet", "")).strip(),
                "published_date": "",
            }
        )
    return rows


def _search_with_duckduckgo(
    *,
    query: str,
    domains: list[str],
    max_results: int,
    timeout_seconds: float,
    timeout_budget: float | None = None,
    deadline_monotonic: float | None = None,
    normalize_whitespace: Callable[[str], str] = lambda value: " ".join(value.split()),
    resolve_timeout_budget: _TimeoutBudgetResolver = _default_resolve_timeout_budget,
) -> list[SearchResultRow]:
    """使用 DuckDuckGo HTML 页面搜索。

    Args:
        query: 检索关键词。
        domains: 域名过滤。
        max_results: 返回数量。
        timeout_seconds: HTTP 请求超时秒数。
        timeout_budget: Runner 注入的单次 tool call 总预算。
        deadline_monotonic: 当前工具调用的单调时钟 deadline。
        normalize_whitespace: 文本空白规整函数。
        resolve_timeout_budget: timeout 预算解析函数。

    Returns:
        结果列表。

    Raises:
        RuntimeError: 当请求失败时抛出。
    """

    query_with_domain = query
    if domains:
        query_with_domain = f"{query} " + " ".join(f"site:{domain}" for domain in domains)

    response = requests.get(
        "https://duckduckgo.com/html/",
        params={"q": query_with_domain},
        timeout=resolve_timeout_budget(
            timeout_seconds,
            timeout_budget=timeout_budget,
            deadline_monotonic=deadline_monotonic,
        ),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    results: list[SearchResultRow] = []
    for node in soup.select("div.result"):
        anchor = node.select_one("a.result__a")
        if anchor is None:
            continue
        snippet_node = node.select_one("a.result__snippet") or node.select_one("div.result__snippet")
        title = normalize_whitespace(anchor.get_text(" ", strip=True))
        raw_href = anchor.get("href")
        if not isinstance(raw_href, str):
            continue
        url = _resolve_duckduckgo_result_url(raw_href)
        if not url:
            continue
        snippet = normalize_whitespace(snippet_node.get_text(" ", strip=True) if snippet_node else "")
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "published_date": "",
            }
        )
        if len(results) >= max_results:
            break
    return results


def _resolve_duckduckgo_result_url(raw_url: str) -> str:
    """解析 DuckDuckGo 搜索结果链接为可访问目标 URL。

    Args:
        raw_url: 结果项中的原始 href。

    Returns:
        解析后的目标 URL；无法解析时返回原值或空字符串。

    Raises:
        无。
    """

    candidate = str(raw_url or "").strip()
    if not candidate:
        return ""

    if candidate.startswith("//"):
        candidate = f"https:{candidate}"

    if candidate.startswith("/"):
        parsed_relative = urlparse(candidate)
        if parsed_relative.path.startswith("/l"):
            uddg_values = parse_qs(parsed_relative.query).get("uddg", [])
            if uddg_values:
                return unquote(uddg_values[0]).strip()
        return ""

    parsed = urlparse(candidate)
    if parsed.hostname and parsed.hostname.lower().endswith("duckduckgo.com") and parsed.path.startswith("/l"):
        uddg_values = parse_qs(parsed.query).get("uddg", [])
        if uddg_values:
            return unquote(uddg_values[0]).strip()
    return candidate


def _build_search_web_preferred_result(
    results: list[SearchResultRow],
) -> SearchResultRow | None:
    """提取 `search_web` 的首选结果。

    Args:
        results: 已完成安全过滤与数量裁剪的结果列表。

    Returns:
        首条结果存在时返回其浅拷贝；否则返回 `None`。

    Raises:
        无。
    """

    if not results:
        return None
    first_result = results[0]
    return {
        "title": first_result["title"],
        "url": first_result["url"],
        "snippet": first_result["snippet"],
        "published_date": first_result["published_date"],
    }


def _build_search_web_preferred_summary(
    *,
    preferred_result: SearchResultRow | None,
    normalize_whitespace: Callable[[str], str],
) -> str:
    """构建 `search_web` 的首选结果摘要。

    Args:
        preferred_result: 首选结果；无结果时为 `None`。
        normalize_whitespace: 文本空白规整函数。

    Returns:
        面向 LLM 的单行摘要。

    Raises:
        无。
    """

    if preferred_result is None:
        return "未找到可直接抓取正文的公开网页结果。"

    title = normalize_whitespace(preferred_result["title"].strip())
    url = preferred_result["url"].strip()
    published_date = preferred_result["published_date"].strip()
    snippet = normalize_whitespace(preferred_result["snippet"].strip())
    snippet_preview = snippet[:_SEARCH_WEB_SNIPPET_PREVIEW_CHARS]
    if len(snippet) > _SEARCH_WEB_SNIPPET_PREVIEW_CHARS:
        snippet_preview = f"{snippet_preview}..."

    summary_parts = ["首选结果"]
    if title:
        summary_parts.append(f"标题：{title}")
    if published_date:
        summary_parts.append(f"日期：{published_date}")
    if url:
        summary_parts.append(f"URL：{url}")
    if snippet_preview:
        summary_parts.append(f"摘要：{snippet_preview}")
    return "；".join(summary_parts)


def _build_search_web_next_action(*, preferred_result: SearchResultRow | None) -> str:
    """构建 `search_web` 的下一步动作。

    Args:
        preferred_result: 首选结果；无结果时为 `None`。

    Returns:
        下一步动作名称。

    Raises:
        无。
    """

    if preferred_result is None:
        return _SEARCH_WEB_NEXT_ACTION_REFINE_QUERY
    return _SEARCH_WEB_NEXT_ACTION_FETCH_PAGE


def _build_search_web_next_action_args(
    *,
    preferred_result: SearchResultRow | None,
) -> dict[str, str]:
    """构建 `search_web` 的下一步动作参数。

    Args:
        preferred_result: 首选结果；无结果时为 `None`。

    Returns:
        下一步动作参数字典。

    Raises:
        无。
    """

    if preferred_result is None:
        return {}
    return {"url": preferred_result["url"].strip()}


def _build_search_web_hint(
    *,
    preferred_result: SearchResultRow | None,
    normalize_whitespace: Callable[[str], str],
) -> str:
    """构建 `search_web` 成功返回的下一步提示。

    Args:
        preferred_result: 首选结果；无结果时为 `None`。
        normalize_whitespace: 文本空白规整函数。

    Returns:
        直接指向下一步动作的提示文案。

    Raises:
        无。
    """

    if preferred_result is None:
        return (
            "当前没有可直接抓取的网页正文。下一步应改写 query，或放宽 domains/recency_days 后重新调用 "
            "search_web；不要对空结果调用 fetch_web_page。"
        )

    url = preferred_result["url"].strip()
    title = normalize_whitespace(preferred_result["title"].strip())
    target_label = f"《{title}》" if title else url
    return (
        f"优先抓取首选结果正文：下一步直接调用 fetch_web_page(url='{url}') 读取 {target_label}。"
        "只有当首选结果抓取失败或正文不相关时，再回看 results 中其他候选。"
    )
