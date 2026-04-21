"""`sec_company_meta` 真源边界测试。"""

from __future__ import annotations

from dataclasses import dataclass

from dayu.fins.domain.document_models import CompanyMeta
from dayu.fins.pipelines.sec_company_meta import (
    extract_sec_ticker_aliases,
    merge_ticker_aliases,
    normalize_sec_ticker_aliases,
    upsert_company_meta,
)
from dayu.fins.storage import CompanyMetaRepositoryProtocol


@dataclass
class _CompanyRepoSpy:
    """公司元数据仓储测试桩。"""

    captured: CompanyMeta | None = None

    def get_company_meta(self, ticker: str) -> CompanyMeta:
        """读取公司元数据。"""

        raise FileNotFoundError(ticker)

    def upsert_company_meta(self, meta: CompanyMeta) -> None:
        """写入公司元数据。"""

        self.captured = meta

    def scan_company_meta_inventory(self) -> list[dict[str, str]]:
        """扫描公司元数据。"""

        return []

    def resolve_existing_ticker(self, ticker: str) -> str | None:
        """解析已存在 ticker。"""

        return None


def test_normalize_sec_ticker_aliases_keeps_primary_and_deduplicates() -> None:
    """alias 标准化应保持主 ticker 在首位并统一大写去重。"""

    result = normalize_sec_ticker_aliases(
        primary_ticker="aapl",
        raw_aliases=["AAPL", "apc", "", "ApC", "AAPL.SW"],
    )

    assert result == ["AAPL", "APC", "AAPL.SW"]


def test_extract_sec_ticker_aliases_ignores_non_list_payload() -> None:
    """SEC submissions 的 tickers 字段不是列表时应只保留主 ticker。"""

    result = extract_sec_ticker_aliases(
        submissions={"tickers": "AAPL"},
        primary_ticker="AAPL",
    )

    assert result == ["AAPL"]


def test_merge_ticker_aliases_preserves_group_priority() -> None:
    """多组 alias 合并应按组顺序保留优先级。"""

    result = merge_ticker_aliases(
        primary_ticker="BABA",
        alias_groups=[["BABA", "9988"], ["9988.HK", "9988", "BABAF"]],
    )

    # 每个 alias 都走真源归一化后整体去重：`9988.HK` canonical=`9988`，与前者重复。
    assert result == ["BABA", "9988", "BABAF"]


def test_upsert_company_meta_falls_back_to_ticker_name() -> None:
    """公司名为空时应回退为 ticker，且 alias 规则一致。"""

    spy = _CompanyRepoSpy()
    upsert_company_meta(
        repository=spy,  # type: ignore[arg-type]
        ticker="AAPL",
        company_id="320193",
        company_name="",
        ticker_aliases=["AAPL", "APC"],
    )

    assert spy.captured is not None
    assert spy.captured.company_name == "AAPL"
    assert spy.captured.ticker_aliases == ["AAPL", "APC"]
