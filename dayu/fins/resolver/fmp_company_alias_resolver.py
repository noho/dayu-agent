"""FMP 公司 alias 推断器。

该模块负责基于 Financial Modeling Prep 的公司搜索接口推断：
- 规范公司名称
- 同公司的跨市场 ticker alias

设计约束：
- 只接受 canonical ticker 作为输入。
- 采用“两跳式”推断：先 `search-symbol`，再 `search-name`。
- 使用严格同名规则过滤候选，避免把模糊匹配误当成同公司 alias。
"""

from __future__ import annotations

import json
import os
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from dayu.contracts.env_keys import FMP_API_KEY_ENV


@dataclass(frozen=True)
class FmpAliasInferenceResult:
    """FMP alias 推断结果。

    Attributes:
        canonical_ticker: 规范 ticker。
        company_name: FMP 返回的规范公司名称。
        ticker_aliases: 去重后的同公司 ticker 列表，首项恒为规范 ticker。
    """

    canonical_ticker: str
    company_name: str
    ticker_aliases: list[str]


class FmpAliasInferenceError(RuntimeError):
    """FMP alias 推断失败异常。"""


def infer_company_aliases_from_fmp(canonical_ticker: str) -> FmpAliasInferenceResult:
    """根据 canonical ticker 推断 FMP 公司名称与 alias。

    算法：
    1. `search-symbol` 先定位规范公司名。
    2. 再用公司名走 `search-name` 搜索所有严格同名证券。
    3. 聚合同名 symbol，并保证 canonical ticker 始终排在首位。

    Args:
        canonical_ticker: 规范 ticker。

    Returns:
        FMP alias 推断结果。

    Raises:
        FmpAliasInferenceError: 当环境变量缺失、接口无结果或返回格式非法时抛出。
    """

    normalized_ticker = _normalize_ticker_token(canonical_ticker)
    if not normalized_ticker:
        raise FmpAliasInferenceError("canonical ticker 不能为空")

    api_key = str(os.environ.get(FMP_API_KEY_ENV) or "").strip()
    if not api_key:
        raise FmpAliasInferenceError(f"环境变量 {FMP_API_KEY_ENV} 未配置")

    symbol_results = _fetch_fmp_search_results(
        endpoint="search-symbol",
        query=normalized_ticker,
        limit=10,
        api_key=api_key,
    )
    selected_company = _select_symbol_result(
        results=symbol_results,
        canonical_ticker=normalized_ticker,
    )
    company_name = _extract_required_company_name(selected_company)
    normalized_company_name = _normalize_company_name(company_name)

    same_name_symbol_results = _filter_same_name_results(
        results=symbol_results,
        normalized_company_name=normalized_company_name,
    )
    name_results = _fetch_fmp_search_results(
        endpoint="search-name",
        query=company_name,
        limit=50,
        api_key=api_key,
    )
    same_name_name_results = _filter_same_name_results(
        results=name_results,
        normalized_company_name=normalized_company_name,
    )
    ticker_aliases = _dedupe_ticker_aliases(
        canonical_ticker=normalized_ticker,
        raw_aliases=[
            *[str(item.get("symbol", "")) for item in same_name_symbol_results],
            *[str(item.get("symbol", "")) for item in same_name_name_results],
        ],
    )
    return FmpAliasInferenceResult(
        canonical_ticker=normalized_ticker,
        company_name=company_name,
        ticker_aliases=ticker_aliases,
    )


def _fetch_fmp_search_results(
    *,
    endpoint: str,
    query: str,
    limit: int,
    api_key: str,
) -> list[dict[str, Any]]:
    """请求 FMP 搜索接口并返回 JSON 数组。

    Args:
        endpoint: FMP 稳定搜索端点名称。
        query: 查询词。
        limit: 结果上限。
        api_key: FMP API Key。

    Returns:
        结果对象数组。

    Raises:
        FmpAliasInferenceError: 请求失败或返回格式非法时抛出。
    """

    url = (
        "https://financialmodelingprep.com/stable/"
        f"{endpoint}?query={urllib.parse.quote(query)}&limit={limit}"
        f"&apikey={urllib.parse.quote(api_key)}"
    )
    try:
        with urllib.request.urlopen(url, timeout=20) as response:
            raw_body = response.read().decode("utf-8")
    except Exception as exc:  # pragma: no cover - 由测试桩覆盖错误路径
        raise FmpAliasInferenceError(f"请求 FMP {endpoint} 失败: {exc}") from exc
    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise FmpAliasInferenceError(f"FMP {endpoint} 返回非 JSON 内容") from exc
    if not isinstance(payload, list):
        raise FmpAliasInferenceError(f"FMP {endpoint} 返回格式非法，期望数组")
    normalized_results: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            normalized_results.append(item)
    return normalized_results


def _select_symbol_result(
    *,
    results: list[dict[str, Any]],
    canonical_ticker: str,
) -> dict[str, Any]:
    """从 `search-symbol` 结果中选择规范公司条目。

    规则：
    - 优先选择 symbol 与 canonical ticker 完全相等的结果。
    - 若不存在完全相等项，则回退到第一条结果。

    Args:
        results: `search-symbol` 返回结果。
        canonical_ticker: 规范 ticker。

    Returns:
        选中的公司结果对象。

    Raises:
        FmpAliasInferenceError: 结果为空时抛出。
    """

    if not results:
        raise FmpAliasInferenceError(f"FMP search-symbol 未返回结果: ticker={canonical_ticker}")
    for item in results:
        normalized_symbol = _normalize_ticker_token(str(item.get("symbol", "")))
        if normalized_symbol == canonical_ticker:
            return item
    return results[0]


def _extract_required_company_name(result: dict[str, Any]) -> str:
    """从搜索结果中提取必填公司名。

    Args:
        result: 单条搜索结果。

    Returns:
        非空公司名称。

    Raises:
        FmpAliasInferenceError: 名称缺失时抛出。
    """

    company_name = str(result.get("name", "")).strip()
    if not company_name:
        raise FmpAliasInferenceError("FMP 搜索结果缺少公司名称")
    return company_name


def _filter_same_name_results(
    *,
    results: list[dict[str, Any]],
    normalized_company_name: str,
) -> list[dict[str, Any]]:
    """过滤出与目标公司名严格同名的结果。

    Args:
        results: 原始搜索结果。
        normalized_company_name: 已规范化的目标公司名。

    Returns:
        严格同名结果列表。

    Raises:
        无。
    """

    filtered_results: list[dict[str, Any]] = []
    for item in results:
        item_name = _normalize_company_name(str(item.get("name", "")))
        if item_name != normalized_company_name:
            continue
        filtered_results.append(item)
    return filtered_results


def _normalize_company_name(company_name: str) -> str:
    """规范化公司名，供严格同名比较使用。

    Args:
        company_name: 原始公司名称。

    Returns:
        规范化后的公司名称。

    Raises:
        无。
    """

    normalized = unicodedata.normalize("NFKC", company_name)
    normalized = " ".join(normalized.strip().split())
    return normalized.upper()


def _normalize_ticker_token(raw_token: str) -> str:
    """规范化 ticker token。

    Args:
        raw_token: 原始 ticker token。

    Returns:
        去空白并大写后的 ticker；空输入返回空字符串。

    Raises:
        无。
    """

    compact_token = "".join(str(raw_token).strip().split())
    return compact_token.upper()


def _dedupe_ticker_aliases(
    *,
    canonical_ticker: str,
    raw_aliases: list[str],
) -> list[str]:
    """对 alias 列表做规范化与去重。

    Args:
        canonical_ticker: 规范 ticker。
        raw_aliases: 原始 alias 列表。

    Returns:
        首项恒为 canonical ticker 的去重 alias 列表。

    Raises:
        无。
    """

    deduped_aliases: list[str] = []
    seen_aliases: set[str] = set()
    for raw_alias in [canonical_ticker, *raw_aliases]:
        normalized_alias = _normalize_ticker_token(raw_alias)
        if not normalized_alias or normalized_alias in seen_aliases:
            continue
        seen_aliases.add(normalized_alias)
        deduped_aliases.append(normalized_alias)
    return deduped_aliases
