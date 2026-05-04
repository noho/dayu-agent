"""披露易 HK 财报下载器。

本模块实现 ``CnReportDiscoveryClientProtocol``：从披露易 stock list 解析
``stockId``，通过 ``titleSearchServlet.do`` 发现年报、半年报与季度公告，并
下载 PDF。模块只依赖 HTTP 客户端和 CN/HK typed model，不依赖 pipeline、
docling 或 storage，也不生成 ``document_id``。
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Optional, TypeAlias, cast

import httpx

from dayu.fins.pipelines.cn_download_models import (
    CnCompanyProfile,
    CnFiscalPeriod,
    CnLanguage,
    CnReportCandidate,
    CnReportQuery,
    DownloadedReportAsset,
)
from dayu.log import Log

_MODULE: Final[str] = "FINS.HKEXNEWS_DOWNLOADER"

JsonScalar: TypeAlias = str | int | float | bool | None
"""JSON 标量值。"""

JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
"""披露易接口 JSON 响应值。"""

HKEXNEWS_BASE_URL: Final[str] = "https://www1.hkexnews.hk"
HKEXNEWS_ACTIVE_STOCK_ZH_URL: Final[str] = (
    f"{HKEXNEWS_BASE_URL}/ncms/script/eds/activestock_sehk_c.json"
)
HKEXNEWS_INACTIVE_STOCK_ZH_URL: Final[str] = (
    f"{HKEXNEWS_BASE_URL}/ncms/script/eds/inactivestock_sehk_c.json"
)
HKEXNEWS_TITLE_SEARCH_URL: Final[str] = (
    f"{HKEXNEWS_BASE_URL}/search/titleSearchServlet.do"
)

DEFAULT_USER_AGENT: Final[str] = "DayuAgent/1.0 (+hk-download)"
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_SLEEP_SECONDS: Final[float] = 0.3
DEFAULT_MAX_RETRIES: Final[int] = 3
RETRY_BACKOFF_BASE_SECONDS: Final[float] = 0.8
DEFAULT_LANGUAGES: Final[tuple[CnLanguage, ...]] = ("zh",)

_PERIOD_SORT_KEY: Final[dict[CnFiscalPeriod, int]] = {
    "FY": 0,
    "H1": 1,
    "Q1": 2,
    "Q2": 3,
    "Q3": 4,
    "Q4": 5,
}

_PDF_MAGIC_BYTES: Final[bytes] = b"%PDF-"
_PDF_MIN_BYTES: Final[int] = 1024
_TITLE_AMENDED_TOKENS: Final[tuple[str, ...]] = (
    "更正",
    "修訂",
    "修订",
    "補充",
    "补充",
    "REVISED",
    "SUPPLEMENTAL",
)
_ENGLISH_REPORT_TITLE_TOKENS: Final[tuple[str, ...]] = (
    "ANNUAL REPORT",
    "INTERIM REPORT",
    "QUARTERLY REPORT",
    "QUARTERLY RESULTS",
    "FIRST QUARTER",
    "SECOND QUARTER",
    "THIRD QUARTER",
    "FOURTH QUARTER",
)
_HKEXNEWS_CATEGORY_MARKET: Final[str] = "SEHK"
_HKEXNEWS_CATEGORY_ZERO: Final[str] = "0"
_HKEXNEWS_SEARCH_TYPE_BY_STOCK: Final[str] = "1"
_HKEXNEWS_DOCUMENT_TYPE_ALL: Final[str] = "-1"
_HKEXNEWS_T1_FINANCIAL_STATEMENTS: Final[str] = "40000"
_HKEXNEWS_T1_ANNOUNCEMENTS: Final[str] = "10000"
_HKEXNEWS_T2_GROUP_ALL: Final[str] = "-2"
_HKEXNEWS_T2_GROUP_RESULTS: Final[str] = "3"
_HKEXNEWS_T2_ANNUAL_REPORT: Final[str] = "40100"
_HKEXNEWS_T2_INTERIM_REPORT: Final[str] = "40200"
_HKEXNEWS_T2_QUARTERLY_RESULTS: Final[str] = "13600"
_HKEXNEWS_ROW_RANGE: Final[str] = "100"
_HKEXNEWS_MB_DATE_RANGE: Final[str] = "0"
_HKEXNEWS_SORT_BY_DATETIME: Final[str] = "DateTime"
_HKEXNEWS_SORT_DIR_DESC: Final[str] = "0"
_HKEXNEWS_FILE_TYPE_PDF: Final[str] = "PDF"
_PERIOD_INFERENCE_TOKENS: Final[dict[CnFiscalPeriod, tuple[str, ...]]] = {
    "FY": ("ANNUAL REPORT", "年報", "年报", "年度報告", "年度报告"),
    "H1": ("INTERIM REPORT", "HALF-YEAR", "HALF YEAR", "中期報告", "中期报告", "半年報", "半年度報告"),
    "Q1": ("FIRST QUARTER", "FIRST QUARTERLY", "THREE MONTHS", "3 MONTHS", "第一季度", "第一季", "一季度", "一季", "三個月", "三个月"),
    "Q2": ("SECOND QUARTER", "SECOND QUARTERLY", "SIX MONTHS", "6 MONTHS", "HALF YEAR", "Q2", "第二季度", "第二季", "二季度", "二季", "六個月", "六个月", "半年"),
    "Q3": ("THIRD QUARTER", "THIRD QUARTERLY", "NINE MONTHS", "9 MONTHS", "第三季度", "第三季", "三季度", "三季", "九個月", "九个月"),
    "Q4": ("FOURTH QUARTER", "FOURTH QUARTERLY", "TWELVE MONTHS", "12 MONTHS", "FULL YEAR", "Q4", "第四季度", "第四季", "四季度", "四季", "十二個月", "十二个月", "全年"),
}
_TITLE_YEAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"(20\d{2}|19\d{2})")
_TITLE_CHINESE_YEAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"([零〇一二三四五六七八九]{4})年")
_CHINESE_DIGIT_TO_INT: Final[dict[str, int]] = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_DATE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})"
)
_BR_PATTERN: Final[re.Pattern[str]] = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class _HkCategorySpec:
    """披露易 title search 标题分类参数。"""

    t1code: str
    t2_group_code: str
    t2code: str


@dataclass(frozen=True)
class _HkStockMappingEntry:
    """披露易 stock list 单只股票记录。"""

    stock_code: str
    stock_id: str
    company_name: str


@dataclass(frozen=True)
class _RawHkAnnouncement:
    """披露易 title search 单条公告强类型抽象。"""

    document_id: str
    title: str
    file_link: str
    stock_code_payload: str
    category_text: str
    filing_date: str
    language: CnLanguage


@dataclass(frozen=True)
class _HeadMeta:
    """HEAD 响应中的候选 fingerprint 字段。"""

    content_length: Optional[int]
    etag: Optional[str]
    last_modified: Optional[str]


_PERIOD_TO_CATEGORY_SPEC: Final[dict[CnFiscalPeriod, _HkCategorySpec]] = {
    "FY": _HkCategorySpec(
        t1code=_HKEXNEWS_T1_FINANCIAL_STATEMENTS,
        t2_group_code=_HKEXNEWS_T2_GROUP_ALL,
        t2code=_HKEXNEWS_T2_ANNUAL_REPORT,
    ),
    "H1": _HkCategorySpec(
        t1code=_HKEXNEWS_T1_FINANCIAL_STATEMENTS,
        t2_group_code=_HKEXNEWS_T2_GROUP_ALL,
        t2code=_HKEXNEWS_T2_INTERIM_REPORT,
    ),
    "Q2": _HkCategorySpec(
        t1code=_HKEXNEWS_T1_ANNOUNCEMENTS,
        t2_group_code=_HKEXNEWS_T2_GROUP_RESULTS,
        t2code=_HKEXNEWS_T2_QUARTERLY_RESULTS,
    ),
    "Q1": _HkCategorySpec(
        t1code=_HKEXNEWS_T1_ANNOUNCEMENTS,
        t2_group_code=_HKEXNEWS_T2_GROUP_RESULTS,
        t2code=_HKEXNEWS_T2_QUARTERLY_RESULTS,
    ),
    "Q3": _HkCategorySpec(
        t1code=_HKEXNEWS_T1_ANNOUNCEMENTS,
        t2_group_code=_HKEXNEWS_T2_GROUP_RESULTS,
        t2code=_HKEXNEWS_T2_QUARTERLY_RESULTS,
    ),
    "Q4": _HkCategorySpec(
        t1code=_HKEXNEWS_T1_ANNOUNCEMENTS,
        t2_group_code=_HKEXNEWS_T2_GROUP_RESULTS,
        t2code=_HKEXNEWS_T2_QUARTERLY_RESULTS,
    ),
}


class HkexnewsDiscoveryClient:
    """披露易 HK discovery / 下载客户端。"""

    def __init__(
        self,
        *,
        client: Optional[httpx.Client] = None,
        user_agent: Optional[str] = None,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        languages: tuple[CnLanguage, ...] = DEFAULT_LANGUAGES,
        sleep_func: Optional[Callable[[float], None]] = None,
    ) -> None:
        """初始化披露易下载器。

        Args:
            client: 可选 ``httpx.Client``；测试可注入 ``MockTransport`` 客户端。
            user_agent: HTTP User-Agent。
            sleep_seconds: 连续请求间隔秒数。
            max_retries: 单次 HTTP 请求最大重试次数。
            request_timeout_seconds: 单次请求超时秒数。
            languages: 查询语言顺序；默认只查中文。
            sleep_func: 可注入 sleep 函数；测试可传 ``lambda _: None``。

        Raises:
            ValueError: ``max_retries`` 非正、``sleep_seconds`` 为负或语言为空时抛出。
        """

        if max_retries <= 0:
            raise ValueError("max_retries 必须大于 0")
        if sleep_seconds < 0:
            raise ValueError("sleep_seconds 不能为负数")
        if not languages:
            raise ValueError("languages 不能为空")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=request_timeout_seconds,
            headers={"User-Agent": user_agent or DEFAULT_USER_AGENT},
        )
        self._sleep_seconds = sleep_seconds
        self._max_retries = max_retries
        self._languages = languages
        self._sleep_func: Callable[[float], None] = (
            sleep_func if sleep_func is not None else time.sleep
        )
        self._last_request_finished_at: float | None = None
        self._stock_mapping_cache: dict[str, _HkStockMappingEntry] | None = None

    def close(self) -> None:
        """关闭底层 HTTP 客户端。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        if self._owns_client:
            self._client.close()

    def resolve_company(self, query: CnReportQuery) -> CnCompanyProfile:
        """解析 HK ticker 对应的披露易公司元数据。

        Args:
            query: 单次 download 查询参数；``query.market`` 必须为 ``"HK"``。

        Returns:
            ``CnCompanyProfile``，``company_id`` 形如 ``"HKEX:{stockId}"``。

        Raises:
            ValueError: market 非 HK，或 stock list 未命中 ticker 时抛出。
            RuntimeError: stock list 请求失败时抛出。
        """

        if query.market != "HK":
            raise ValueError(f"HkexnewsDiscoveryClient 仅支持 HK，收到 market={query.market!r}")
        stock_code = _to_hkex_stock_code(query.normalized_ticker)
        mapping = self._fetch_stock_mapping()
        entry = mapping.get(stock_code)
        if entry is None:
            raise ValueError(f"披露易 stock list 未命中 ticker={query.normalized_ticker!r}")
        return CnCompanyProfile(
            provider="hkexnews",
            company_id=f"HKEX:{entry.stock_id}",
            company_name=entry.company_name,
            ticker=query.normalized_ticker,
        )

    def list_report_candidates(
        self,
        query: CnReportQuery,
        profile: CnCompanyProfile,
    ) -> tuple[CnReportCandidate, ...]:
        """列出符合窗口和财期的 HK 报告候选。

        Args:
            query: 单次 download 查询参数。
            profile: ``resolve_company`` 返回的公司元数据。

        Returns:
            候选报告 tuple。HK 季度报告查无返回空 tuple，不抛异常。

        Raises:
            ValueError: market/provider/company_id 非法时抛出。
            RuntimeError: 任一有效财期分类的底层请求或 JSON 解析失败时抛出。
        """

        if query.market != "HK":
            raise ValueError(f"HkexnewsDiscoveryClient 仅支持 HK，收到 market={query.market!r}")
        if profile.provider != "hkexnews":
            raise ValueError(f"profile.provider 必须为 hkexnews，收到 {profile.provider!r}")
        stock_id = profile.company_id.removeprefix("HKEX:")
        if not stock_id:
            raise ValueError(f"profile.company_id 缺少 HKEX: 前缀: {profile.company_id!r}")
        stock_code = _to_hkex_stock_code(query.normalized_ticker)

        grouped: dict[tuple[CnFiscalPeriod, int], list[_RawHkAnnouncement]] = {}
        periods_by_category: dict[_HkCategorySpec, list[CnFiscalPeriod]] = {}
        for period in query.target_periods:
            category_spec = _PERIOD_TO_CATEGORY_SPEC.get(period)
            if category_spec is None:
                Log.warn(f"未知 fiscal_period={period!r}，已跳过", module=_MODULE)
                continue
            periods = periods_by_category.setdefault(category_spec, [])
            if period not in periods:
                periods.append(period)

        for category_spec, requested_periods in periods_by_category.items():
            try:
                announcements = self._query_period_announcements(
                    stock_id=stock_id,
                    stock_code=stock_code,
                    category_spec=category_spec,
                    start_date=query.start_date,
                    end_date=query.end_date,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"披露易公告分类查询失败: stock_code={stock_code} periods={','.join(requested_periods)} error={exc}"
                ) from exc
            for item in announcements:
                inferred_period = _infer_fiscal_period_from_text(
                    title=item.title,
                    category_text=item.category_text,
                )
                if inferred_period not in requested_periods:
                    continue
                fiscal_year = _infer_fiscal_year(
                    title=item.title,
                    filing_date=item.filing_date,
                )
                if fiscal_year is None:
                    continue
                grouped.setdefault((inferred_period, fiscal_year), []).append(item)

        candidates: list[CnReportCandidate] = []
        for (period, fiscal_year), items in grouped.items():
            best = _pick_best_announcement(items)
            if best is None:
                continue
            candidates.append(
                self._build_candidate(
                    announcement=best,
                    period=period,
                    fiscal_year=fiscal_year,
                )
            )
        candidates.sort(key=lambda c: (-c.fiscal_year, _PERIOD_SORT_KEY[c.fiscal_period]))
        return tuple(candidates)

    def download_report_pdf(self, candidate: CnReportCandidate) -> DownloadedReportAsset:
        """下载单份 HK PDF 并返回资产对象。

        Args:
            candidate: 远端候选报告。

        Returns:
            已下载 PDF 资产。

        Raises:
            RuntimeError: provider 非 hkexnews、下载失败或 PDF 校验失败时抛出。
        """

        if candidate.provider != "hkexnews":
            raise RuntimeError(
                f"HkexnewsDiscoveryClient 不支持 provider={candidate.provider!r}"
            )
        payload = self._http_download_bytes(candidate.source_url)
        if len(payload) < _PDF_MIN_BYTES:
            raise RuntimeError(
                f"PDF 字节数过小 ({len(payload)} bytes)，url={candidate.source_url}"
            )
        if not payload.startswith(_PDF_MAGIC_BYTES):
            raise RuntimeError(f"PDF magic bytes 校验失败，url={candidate.source_url}")
        sha256 = hashlib.sha256(payload).hexdigest()
        tmp_dir = Path(tempfile.gettempdir()) / "dayu_hk_downloads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="hkexnews_",
            suffix=".pdf",
            dir=tmp_dir,
            delete=False,
        ) as fp:
            fp.write(payload)
            pdf_path = Path(fp.name)
        return DownloadedReportAsset(
            candidate=candidate,
            pdf_path=pdf_path,
            sha256=sha256,
            content_length=len(payload),
            downloaded_at=_utc_now_isoformat(),
        )

    def _fetch_stock_mapping(self) -> dict[str, _HkStockMappingEntry]:
        """拉取并缓存披露易 active/inactive stock list。

        Args:
            无。

        Returns:
            ``STOCK_CODE -> _HkStockMappingEntry`` 映射。

        Raises:
            RuntimeError: HTTP 或 JSON 解析失败时抛出。
        """

        if self._stock_mapping_cache is not None:
            return self._stock_mapping_cache
        mapping: dict[str, _HkStockMappingEntry] = {}
        for url in (HKEXNEWS_ACTIVE_STOCK_ZH_URL, HKEXNEWS_INACTIVE_STOCK_ZH_URL):
            payload = self._http_get_json(url)
            for raw in _extract_json_rows(payload):
                entry = _parse_stock_mapping_entry(raw)
                if entry is not None:
                    mapping.setdefault(entry.stock_code, entry)
        self._stock_mapping_cache = mapping
        return mapping

    def _query_period_announcements(
        self,
        *,
        stock_id: str,
        stock_code: str,
        category_spec: _HkCategorySpec,
        start_date: str,
        end_date: str,
    ) -> list[_RawHkAnnouncement]:
        """查询单个披露易二级分类的公告列表。

        Args:
            stock_id: 披露易 stockId。
            stock_code: 5 位股票代码。
            category_spec: 披露易标题分类参数。
            start_date: 起始日期 ``YYYY-MM-DD``。
            end_date: 结束日期 ``YYYY-MM-DD``。

        Returns:
            匹配目标股票且非英文的公告列表。

        Raises:
            RuntimeError: HTTP 或 JSON 解析失败时抛出。
        """

        primary: list[_RawHkAnnouncement] = []
        for language in self._languages:
            payload = self._http_get_json(
                HKEXNEWS_TITLE_SEARCH_URL,
                params={
                    "lang": _language_param(language),
                    "category": _HKEXNEWS_CATEGORY_ZERO,
                    "market": _HKEXNEWS_CATEGORY_MARKET,
                    "stockId": stock_id,
                    "searchType": _HKEXNEWS_SEARCH_TYPE_BY_STOCK,
                    "documentType": _HKEXNEWS_DOCUMENT_TYPE_ALL,
                    "t1code": category_spec.t1code,
                    "t2Gcode": category_spec.t2_group_code,
                    "t2code": category_spec.t2code,
                    "fromDate": start_date.replace("-", ""),
                    "toDate": end_date.replace("-", ""),
                    "MB-Daterange": _HKEXNEWS_MB_DATE_RANGE,
                    "rowRange": _HKEXNEWS_ROW_RANGE,
                    "sortByOptions": _HKEXNEWS_SORT_BY_DATETIME,
                    "sortDir": _HKEXNEWS_SORT_DIR_DESC,
                },
            )
            rows = _extract_json_rows(payload)
            parsed_rows = [
                item
                for item in (_parse_announcement(row, language=language) for row in rows)
                if item is not None
                and _announcement_matches_stock(item.stock_code_payload, stock_code)
                and not _is_english_announcement(item)
            ]
            primary.extend(parsed_rows)
        return primary

    def _build_candidate(
        self,
        *,
        announcement: _RawHkAnnouncement,
        period: CnFiscalPeriod,
        fiscal_year: int,
    ) -> CnReportCandidate:
        """把披露易公告转换为 ``CnReportCandidate``。

        Args:
            announcement: 原始公告。
            period: 财期。
            fiscal_year: 财年。

        Returns:
            下载候选。

        Raises:
            无。HEAD 失败会软降级为空 fingerprint 字段。
        """

        source_url = _build_absolute_file_url(announcement.file_link)
        head_meta = self._http_head_meta(source_url)
        return CnReportCandidate(
            provider="hkexnews",
            source_id=announcement.document_id,
            source_url=source_url,
            title=announcement.title,
            language=announcement.language,
            filing_date=announcement.filing_date,
            fiscal_year=fiscal_year,
            fiscal_period=period,
            amended=_is_amended_title(announcement.title),
            content_length=head_meta.content_length,
            etag=head_meta.etag,
            last_modified=head_meta.last_modified,
        )

    def _http_get_json(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> JsonValue:
        """GET JSON。

        Args:
            url: 请求 URL。
            params: 可选 query 参数。

        Returns:
            JSON 响应。

        Raises:
            RuntimeError: 重试后仍失败时抛出。
        """

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                self._throttle_before_request()
                try:
                    response = self._client.get(url, params=params)
                    response.raise_for_status()
                    return cast(JsonValue, response.json())
                finally:
                    self._mark_request_finished()
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                self._retry_backoff(attempt)
        raise RuntimeError(f"GET JSON 失败: url={url} error={last_exc}")

    def _http_head_meta(self, url: str) -> _HeadMeta:
        """HEAD 拉取 content-length / etag / last-modified。

        Args:
            url: PDF URL。

        Returns:
            HEAD 元数据；请求失败时返回空字段。

        Raises:
            无。
        """

        try:
            self._throttle_before_request()
            try:
                response = self._client.head(url, follow_redirects=True)
                response.raise_for_status()
            finally:
                self._mark_request_finished()
        except httpx.HTTPError as exc:
            Log.warn(f"HEAD 失败: url={url} error={exc}", module=_MODULE)
            return _HeadMeta(content_length=None, etag=None, last_modified=None)
        raw_length = response.headers.get("Content-Length")
        try:
            content_length = int(raw_length) if raw_length is not None else None
        except ValueError:
            content_length = None
        return _HeadMeta(
            content_length=content_length,
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )

    def _http_download_bytes(self, url: str) -> bytes:
        """带重试下载文件字节。

        Args:
            url: PDF URL。

        Returns:
            响应字节。

        Raises:
            RuntimeError: 重试后仍失败时抛出。
        """

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                self._throttle_before_request()
                try:
                    response = self._client.get(url, follow_redirects=True)
                    response.raise_for_status()
                    return response.content
                finally:
                    self._mark_request_finished()
            except httpx.HTTPError as exc:
                last_exc = exc
                self._retry_backoff(attempt)
        raise RuntimeError(f"PDF 下载失败: url={url} error={last_exc}")

    def _throttle_before_request(self) -> None:
        """按连续请求间隔限制发起 HTTP 请求。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        now = time.monotonic()
        if self._sleep_seconds > 0 and self._last_request_finished_at is not None:
            elapsed = now - self._last_request_finished_at
            remaining = self._sleep_seconds - elapsed
            if remaining > 0:
                self._sleep_func(remaining)

    def _mark_request_finished(self) -> None:
        """记录最近一次 HTTP 请求结束时间。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self._last_request_finished_at = time.monotonic()

    def _retry_backoff(self, attempt_index: int) -> None:
        """指数退避。

        Args:
            attempt_index: 当前重试序号。

        Returns:
            无。

        Raises:
            无。
        """

        if attempt_index >= self._max_retries - 1:
            return
        self._sleep_func(RETRY_BACKOFF_BASE_SECONDS * (2**attempt_index))


def _extract_json_rows(payload: JsonValue) -> list[JsonValue]:
    """从披露易 JSON 响应中提取列表行。

    Args:
        payload: JSON 响应。

    Returns:
        列表行；无法识别时返回空列表。

    Raises:
        无。
    """

    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in (
        "stockInfo",
        "stockList",
        "stocks",
        "data",
        "result",
        "records",
        "rows",
        "announcements",
    ):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            parsed = _parse_embedded_json_list(value)
            if parsed is not None:
                return parsed
    return []


def _parse_embedded_json_list(raw: str) -> list[JsonValue] | None:
    """解析披露易 ``result`` 字符串 JSON。

    Args:
        raw: 字符串 JSON，常见值为 ``"[]"`` 或 ``"[{...}]"``。

    Returns:
        列表 JSON；不是列表或无法解析时返回 ``None``。

    Raises:
        无。
    """

    text = raw.strip()
    if text in {"", "null"}:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        return cast(list[JsonValue], parsed)
    return None


def _parse_stock_mapping_entry(raw: JsonValue) -> _HkStockMappingEntry | None:
    """解析 stock list 单行。

    Args:
        raw: JSON 行。

    Returns:
        股票映射；关键字段缺失时返回 ``None``。

    Raises:
        ValueError: 股票代码字段存在但格式非法时抛出。
    """

    if not isinstance(raw, dict):
        return None
    code = _first_text(raw, ("stockCode", "STOCK_CODE", "stock_code", "code", "CODE", "c"))
    stock_id = _first_text(raw, ("stockId", "STOCK_ID", "stock_id", "id", "ID", "i"))
    name = _first_text(raw, ("stockName", "STOCK_NAME", "name", "NAME", "longName", "n"))
    if code is None or stock_id is None:
        return None
    normalized_code = _to_hkex_stock_code(code)
    return _HkStockMappingEntry(
        stock_code=normalized_code,
        stock_id=stock_id,
        company_name=_strip_html(name or normalized_code),
    )


def _parse_announcement(
    raw: JsonValue,
    *,
    language: CnLanguage,
) -> _RawHkAnnouncement | None:
    """解析 title search 单行公告。

    Args:
        raw: JSON 行。
        language: 当前查询语言。

    Returns:
        公告对象；关键字段缺失时返回 ``None``。

    Raises:
        无。
    """

    if not isinstance(raw, dict):
        return None
    file_type = _first_text(raw, ("FILE_TYPE", "fileType", "file_type"))
    if file_type is not None and file_type.upper() != _HKEXNEWS_FILE_TYPE_PDF:
        return None
    document_id = _first_text(
        raw,
        ("NEWS_ID", "newsId", "DOC_ID", "docID", "documentId", "id", "SEQUENCE"),
    )
    title = _first_text(raw, ("TITLE", "title", "LONG_TEXT", "longText"))
    file_link = _first_text(raw, ("FILE_LINK", "fileLink", "url"))
    stock_code_payload = _first_text(raw, ("STOCK_CODE", "stockCode", "stock_code"))
    category_text = _first_text(raw, ("LONG_TEXT", "longText", "SHORT_TEXT", "shortText"))
    raw_date = _first_text(raw, ("DATE_TIME", "RELEASE_TIME", "dateTime", "releaseTime"))
    filing_date = _parse_filing_date(raw_date)
    if document_id is None and file_link is not None:
        document_id = _stable_id_from_url(file_link)
    if (
        document_id is None
        or title is None
        or file_link is None
        or stock_code_payload is None
        or filing_date is None
    ):
        return None
    return _RawHkAnnouncement(
        document_id=document_id,
        title=_strip_html(title),
        file_link=file_link,
        stock_code_payload=stock_code_payload,
        category_text=_strip_html(category_text or ""),
        filing_date=filing_date,
        language=language,
    )


def _first_text(data: dict[str, JsonValue], keys: tuple[str, ...]) -> str | None:
    """按 key 顺序读取首个非空文本。

    Args:
        data: JSON dict。
        keys: 备选 key。

    Returns:
        非空文本；不存在时返回 ``None``。

    Raises:
        无。
    """

    for key in keys:
        value = data.get(key)
        if isinstance(value, (str, int, float)):
            text = str(value).strip()
            if text:
                return text
    return None


def _to_hkex_stock_code(raw: str) -> str:
    """把 HK canonical ticker 转成披露易 5 位 STOCK_CODE。

    Args:
        raw: 原始 ticker 或 stock code，如 ``0700``、``00700``、``700.HK``。

    Returns:
        5 位股票代码。

    Raises:
        ValueError: 输入缺少数字或位数非法时抛出。
    """

    digits = re.sub(r"\D", "", raw.strip())
    if not digits:
        raise ValueError(f"HK ticker 缺少数字: {raw!r}")
    if raw.strip().upper().endswith(".HK") and len(digits) > 5:
        digits = digits[:-2]
    if len(digits) <= 4:
        return digits.zfill(5)
    if len(digits) == 5:
        return digits
    raise ValueError(f"HK ticker 位数非法: {raw!r}")


def _announcement_matches_stock(stock_code_payload: str, target_stock_code: str) -> bool:
    """判断 ``STOCK_CODE`` 多代码字段是否包含目标股票。

    Args:
        stock_code_payload: 披露易 ``STOCK_CODE`` 字段。
        target_stock_code: 目标 5 位股票代码。

    Returns:
        命中返回 ``True``。

    Raises:
        无。
    """

    tokens = _split_stock_code_tokens(stock_code_payload)
    return target_stock_code in tokens


def _split_stock_code_tokens(stock_code_payload: str) -> set[str]:
    """拆分披露易 ``STOCK_CODE`` 多代码字段。

    Args:
        stock_code_payload: 原始多代码字段，可含 ``<br/>``。

    Returns:
        5 位股票代码集合。

    Raises:
        无。
    """

    text = _BR_PATTERN.sub(",", stock_code_payload)
    text = _TAG_PATTERN.sub("", text)
    tokens: set[str] = set()
    for raw in re.split(r"[,;，\s]+", text):
        digits = re.sub(r"\D", "", raw)
        if not digits:
            continue
        if len(digits) <= 4:
            tokens.add(digits.zfill(5))
        elif len(digits) == 5:
            tokens.add(digits)
    return tokens


def _build_absolute_file_url(file_link: str) -> str:
    """把 ``FILE_LINK`` 拼成绝对 URL。

    Args:
        file_link: 相对或绝对文件链接。

    Returns:
        绝对 URL。

    Raises:
        无。
    """

    text = file_link.strip()
    if text.startswith("http://") or text.startswith("https://"):
        return text
    if text.startswith("/"):
        return f"{HKEXNEWS_BASE_URL}{text}"
    return f"{HKEXNEWS_BASE_URL}/{text}"


def _parse_filing_date(raw_date: str | None) -> str | None:
    """解析披露日期为 ``YYYY-MM-DD``。

    Args:
        raw_date: 原始披露日期。

    Returns:
        规范日期；无法解析返回 ``None``。

    Raises:
        无。
    """

    if raw_date is None:
        return None
    matched = _DATE_PATTERN.search(raw_date)
    if matched is not None:
        year = int(matched.group("year"))
        month = int(matched.group("month"))
        day = int(matched.group("day"))
        return f"{year:04d}-{month:02d}-{day:02d}"
    slash_parts = raw_date.strip().split("/")
    if len(slash_parts) >= 3 and all(part.isdigit() for part in slash_parts[:3]):
        day = int(slash_parts[0])
        month = int(slash_parts[1])
        year = int(slash_parts[2])
        if year >= 1900:
            return f"{year:04d}-{month:02d}-{day:02d}"
    slash_time_parts = raw_date.strip().split()
    if slash_time_parts:
        slash_parts = slash_time_parts[0].split("/")
        if len(slash_parts) == 3 and all(part.isdigit() for part in slash_parts):
            day = int(slash_parts[0])
            month = int(slash_parts[1])
            year = int(slash_parts[2])
            if year >= 1900:
                return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def _infer_fiscal_year(title: str, filing_date: str) -> int | None:
    """从标题和披露日期推断财年。

    Args:
        title: 公告标题。
        filing_date: 披露日期 ``YYYY-MM-DD``。

    Returns:
        推断财年；解析失败返回 ``None``。

    Raises:
        无。
    """

    matched = _TITLE_YEAR_PATTERN.search(title)
    if matched is not None:
        return int(matched.group(1))
    chinese_matched = _TITLE_CHINESE_YEAR_PATTERN.search(title)
    if chinese_matched is not None:
        chinese_year = _parse_chinese_digit_year(chinese_matched.group(1))
        if chinese_year is not None:
            return chinese_year
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", filing_date):
        return int(filing_date[:4])
    return None


def _parse_chinese_digit_year(value: str) -> int | None:
    """解析 ``二零二五`` 这类逐位中文数字年份。

    Args:
        value: 四位中文数字年份。

    Returns:
        解析出的公历年份；格式或范围异常返回 ``None``。

    Raises:
        无。
    """

    if len(value) != 4:
        return None
    digits: list[str] = []
    for char in value:
        digit = _CHINESE_DIGIT_TO_INT.get(char)
        if digit is None:
            return None
        digits.append(str(digit))
    year = int("".join(digits))
    if 1900 <= year <= 2099:
        return year
    return None


def _infer_fiscal_period_from_text(
    *,
    title: str,
    category_text: str,
) -> CnFiscalPeriod | None:
    """从标题和分类文本推断 HK 财期。

    Args:
        title: 公告标题。
        category_text: 披露易分类文本。

    Returns:
        推断财期；无法判定返回 ``None``。

    Raises:
        无。
    """

    combined = f"{title} {category_text}".upper()
    normalized_category = category_text.upper()
    if "季度" in category_text or "QUARTER" in normalized_category:
        order: tuple[CnFiscalPeriod, ...] = ("Q4", "Q3", "Q2", "Q1", "H1", "FY")
    else:
        order = ("H1", "FY", "Q4", "Q3", "Q2", "Q1")
    for period in order:
        tokens = _PERIOD_INFERENCE_TOKENS[period]
        if any(token.upper() in combined for token in tokens):
            return period
    return None


def _pick_best_announcement(items: list[_RawHkAnnouncement]) -> _RawHkAnnouncement | None:
    """从同一 fiscal year + period 的公告中挑最佳版本。

    Args:
        items: 同组公告。

    Returns:
        最佳公告；空列表返回 ``None``。

    Raises:
        无。
    """

    if not items:
        return None

    def sort_key(item: _RawHkAnnouncement) -> tuple[int, str]:
        return (1 if _is_amended_title(item.title) else 0, item.filing_date)

    return max(items, key=sort_key)


def _is_amended_title(title: str) -> bool:
    """判断标题是否为更正/修订版本。

    Args:
        title: 公告标题。

    Returns:
        是更正版本返回 ``True``。

    Raises:
        无。
    """

    upper = title.upper()
    return any(token.upper() in upper for token in _TITLE_AMENDED_TOKENS)


def _is_english_announcement(announcement: _RawHkAnnouncement) -> bool:
    """判断披露易公告是否属于英文候选。

    Args:
        announcement: 披露易原始公告对象。

    Returns:
        英文语言入口或标题/分类明显为英文时返回 ``True``。

    Raises:
        无。
    """

    if announcement.language == "en":
        return True
    if _looks_like_english_report_text(announcement.title):
        return True
    if not _contains_cjk(announcement.title) and _looks_like_english_report_text(
        announcement.category_text
    ):
        return True
    return False


def _looks_like_english_report_text(text: str) -> bool:
    """判断文本是否明显是英文财报标题或分类。

    Args:
        text: 标题或分类文本。

    Returns:
        命中英文财报关键词且缺少中文/繁中文字符时返回 ``True``。

    Raises:
        无。
    """

    if _contains_cjk(text):
        return False
    upper = text.upper()
    return any(token in upper for token in _ENGLISH_REPORT_TITLE_TOKENS)


def _contains_cjk(text: str) -> bool:
    """判断文本是否包含中日韩统一表意文字。

    Args:
        text: 待检测文本。

    Returns:
        包含中文/繁中文字符返回 ``True``。

    Raises:
        无。
    """

    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _language_param(language: CnLanguage) -> str:
    """把候选语言转换为披露易 lang 参数。

    Args:
        language: 语言字面量。

    Returns:
        披露易参数值。

    Raises:
        无。
    """

    return "zh" if language == "zh" else "E"


def _strip_html(raw: str) -> str:
    """清洗 HTML 标签与多余空白。

    Args:
        raw: 原始文本。

    Returns:
        清洗后的文本。

    Raises:
        无。
    """

    unescaped = html.unescape(raw)
    without_br = _BR_PATTERN.sub(" ", unescaped)
    without_tags = _TAG_PATTERN.sub("", without_br)
    return " ".join(without_tags.split())


def _stable_id_from_url(file_link: str) -> str:
    """从 URL 派生稳定 source id。

    Args:
        file_link: 文件链接。

    Returns:
        ``sha256`` 前 16 位。

    Raises:
        无。
    """

    return hashlib.sha256(file_link.encode("utf-8")).hexdigest()[:16]


def _utc_now_isoformat() -> str:
    """生成 ISO-8601 UTC 时间戳。

    Args:
        无。

    Returns:
        UTC 时间戳。

    Raises:
        无。
    """

    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


__all__ = [
    "HKEXNEWS_ACTIVE_STOCK_ZH_URL",
    "HKEXNEWS_BASE_URL",
    "HKEXNEWS_INACTIVE_STOCK_ZH_URL",
    "HKEXNEWS_TITLE_SEARCH_URL",
    "HkexnewsDiscoveryClient",
]
