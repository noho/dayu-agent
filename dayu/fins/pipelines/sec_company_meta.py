"""SEC company meta 与 ticker alias 真源。"""

from __future__ import annotations

from typing import Any, Optional

from dayu.fins.domain.document_models import CompanyMeta, now_iso8601
from dayu.fins.storage import CompanyMetaRepositoryProtocol
from dayu.fins.ticker_normalization import try_normalize_ticker


def _canonicalize_alias_token(raw_token: Any) -> str:
    """把 SEC / pipeline 侧的 ticker alias token 归一到 canonical。

    业务动机：写入 meta 的 alias 必须与真源 canonical 一致，让工具查询时
    无论传 ``9988``/``9988.HK``/``HK.00700`` 都能命中同一公司。真源识别失败
    时（例如 SEC 将来返回非标准字符串）回退到 ``strip().upper()``，保留
    尽力而为的写入行为。

    Args:
        raw_token: 原始 alias token。

    Returns:
        归一化后的字符串；空字符串表示应被丢弃。

    Raises:
        无。
    """

    text = str(raw_token or "").strip()
    if not text:
        return ""
    normalized = try_normalize_ticker(text)
    if normalized is not None:
        return normalized.canonical
    return text.upper()


def normalize_sec_ticker_aliases(
    *,
    primary_ticker: str,
    raw_aliases: Optional[list[Any]],
) -> list[str]:
    """标准化 SEC 返回的 ticker alias 列表。

    每个 alias 都走真源归一化后整体去重；primary_ticker 始终位列第一。

    Args:
        primary_ticker: 主 ticker。
        raw_aliases: SEC 返回的 alias 原始列表。

    Returns:
        去重后的规范 alias 列表，首项为主 ticker 的 canonical。

    Raises:
        ValueError: ``primary_ticker`` 归一化后仍为空时抛出。
    """

    canonical_primary = _canonicalize_alias_token(primary_ticker)
    if not canonical_primary:
        raise ValueError("primary_ticker 不能为空")
    normalized_aliases: list[str] = [canonical_primary]
    for raw_alias in raw_aliases or []:
        canonical_alias = _canonicalize_alias_token(raw_alias)
        if not canonical_alias or canonical_alias in normalized_aliases:
            continue
        normalized_aliases.append(canonical_alias)
    return normalized_aliases


def extract_sec_ticker_aliases(
    *,
    submissions: dict[str, Any],
    primary_ticker: str,
) -> list[str]:
    """从 SEC submissions 中提取 ticker alias。

    Args:
        submissions: SEC submissions JSON。
        primary_ticker: 主 ticker。

    Returns:
        去重后的规范 alias 列表。

    Raises:
        ValueError: ``primary_ticker`` 归一化后仍为空时抛出。
    """

    raw_aliases = submissions.get("tickers")
    alias_list = raw_aliases if isinstance(raw_aliases, list) else None
    return normalize_sec_ticker_aliases(
        primary_ticker=primary_ticker,
        raw_aliases=alias_list,
    )


def merge_ticker_aliases(
    *,
    primary_ticker: str,
    alias_groups: list[Optional[list[str]]],
) -> list[str]:
    """按顺序合并多组 ticker alias。

    每个 alias 都走真源归一化后整体去重；primary_ticker 始终位列第一。

    Args:
        primary_ticker: 主 ticker。
        alias_groups: 待合并的 alias 组；顺序决定进入 meta 的先后。

    Returns:
        去重后的规范 alias 列表，首项为主 ticker 的 canonical。

    Raises:
        ValueError: ``primary_ticker`` 归一化后仍为空时抛出。
    """

    canonical_primary = _canonicalize_alias_token(primary_ticker)
    if not canonical_primary:
        raise ValueError("primary_ticker 不能为空")
    merged_aliases: list[str] = [canonical_primary]
    for alias_group in alias_groups:
        for alias in alias_group or []:
            canonical_alias = _canonicalize_alias_token(alias)
            if not canonical_alias or canonical_alias in merged_aliases:
                continue
            merged_aliases.append(canonical_alias)
    return merged_aliases


def upsert_company_meta(
    *,
    repository: CompanyMetaRepositoryProtocol,
    ticker: str,
    company_id: str,
    company_name: str,
    ticker_aliases: Optional[list[str]] = None,
) -> None:
    """写入 SEC 公司级元数据。"""

    repository.upsert_company_meta(
        CompanyMeta(
            company_id=company_id,
            company_name=company_name or ticker,
            ticker=ticker,
            market="US",
            resolver_version="market_resolver_v1",
            updated_at=now_iso8601(),
            ticker_aliases=normalize_sec_ticker_aliases(
                primary_ticker=ticker,
                raw_aliases=ticker_aliases,
            ),
        )
    )


__all__ = [
    "extract_sec_ticker_aliases",
    "merge_ticker_aliases",
    "normalize_sec_ticker_aliases",
    "upsert_company_meta",
]