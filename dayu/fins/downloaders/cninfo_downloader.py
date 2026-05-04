"""巨潮 CN 财报下载器。

实现 :class:`CnReportDiscoveryClientProtocol`，覆盖 A 股年报 / 半年报 /
一季报 / 三季报的 discovery 与 PDF 下载。巨潮当前没有稳定的独立 Q2/Q4
分类证据，直接请求 Q2/Q4 时返回空候选，由 workflow 统一标记 skipped。

设计要点：

- 仅依赖 ``httpx`` 与 typed model；不依赖 ``CnPipeline``、不写 workspace、
  不调用 docling，也不生成 ``document_id``（``document_id`` 由 pipeline 层
  通过 ``build_cn_filing_ids`` 统一生成）。
- 主源接口：

  - ``GET http://www.cninfo.com.cn/new/data/szse_stock.json``：
    全市场 A 股公司基础映射（``code`` -> ``orgId``）。
  - ``POST http://www.cninfo.com.cn/new/hisAnnouncement/query``：按
    ``stock={code},{orgId}`` + ``category_*_szsh;`` 分类拉公告列表。
  - ``GET http://static.cninfo.com.cn/{adjunctUrl}``：PDF 实体下载。

- 候选筛选：白名单按 category（与请求参数一致），黑名单按标题
  关键词（摘要 / 英文版 / ``（英文）`` / 英文简版 / 已取消 / 募集说明书 /
  ESG / 可持续 / 审计 等），并额外排除关于财报正本的公告类文件。
- 同 ``fiscal_period`` 多版本：``amended=True`` 优先，再按
  ``announcementTime`` 取最新；无 amended 时取最新一条全文。
- HEAD 失败、PDF magic bytes 校验失败仅影响该 candidate，不让整个
  ticker 流程崩。公告分类查询失败属于 discovery 阶段远端错误，必须抛
  ``RuntimeError``，避免被 workflow 误报成缺报告 skipped。
- 接口契约 / 公告字段非正式开放，参数随时变化；本模块**只**消费稳定字段，
  其余字段忽略，避免实现绑死巨潮 schema。
"""

from __future__ import annotations

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
    CnReportCandidate,
    CnReportQuery,
    DownloadedReportAsset,
)
from dayu.log import Log

_MODULE: Final[str] = "FINS.CNINFO_DOWNLOADER"

JsonScalar: TypeAlias = str | int | float | bool | None
"""JSON 标量值。"""

JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
"""巨潮接口 JSON 响应值。"""

CNINFO_BASE_URL: Final[str] = "http://www.cninfo.com.cn"
CNINFO_STATIC_BASE_URL: Final[str] = "http://static.cninfo.com.cn/"
CNINFO_STOCK_JSON_URL: Final[str] = f"{CNINFO_BASE_URL}/new/data/szse_stock.json"
CNINFO_QUERY_URL: Final[str] = f"{CNINFO_BASE_URL}/new/hisAnnouncement/query"

DEFAULT_USER_AGENT: Final[str] = "DayuAgent/1.0 (+cn-download)"
DEFAULT_REQUEST_TIMEOUT_SECONDS: Final[float] = 30.0
DEFAULT_SLEEP_SECONDS: Final[float] = 0.3
DEFAULT_MAX_RETRIES: Final[int] = 3
RETRY_BACKOFF_BASE_SECONDS: Final[float] = 0.8
# CN form -> 巨潮 category 映射；HK 不走此 downloader。
_PERIOD_TO_CATEGORY: Final[dict[CnFiscalPeriod, str]] = {
    "FY": "category_ndbg_szsh;",
    "H1": "category_bndbg_szsh;",
    "Q1": "category_yjdbg_szsh;",
    "Q3": "category_sjdbg_szsh;",
}
_CNINFO_UNSUPPORTED_INDEPENDENT_PERIODS: Final[frozenset[CnFiscalPeriod]] = frozenset(
    {"Q2", "Q4"}
)

# 标题黑名单关键词：命中即排除（大小写不敏感）。
_TITLE_BLOCKLIST: Final[tuple[str, ...]] = (
    "摘要",
    "已取消",
    "已撤销",
    "撤回",
    "取消",
    "更正前",
    "募集说明书",
    "ESG",
    "可持续发展",
    "审计报告",
    "财务报表",
    "意见",
    "（英文）",
    "(英文)",
    "英文)",
    "英文）",
    "英文版",
    "英文简版",
    "英文简本",
    "english",
    "港股公告",
    "h股公告",
    "h股",
)

# 标题含财报关键词但语义是公告时，不应作为财报正本候选。
_REPORT_NOTICE_TITLE_TOKENS: Final[tuple[str, ...]] = (
    "公告",
    "提示性公告",
    "自愿性披露公告",
)
_REPORT_TITLE_TOKENS: Final[tuple[str, ...]] = (
    "年度报告",
    "年报",
    "半年度报告",
    "一季度报告",
    "第一季度报告",
    "三季度报告",
    "第三季度报告",
)

# 标题中 "amended" 标记关键词。
_TITLE_AMENDED_TOKENS: Final[tuple[str, ...]] = ("更正", "更正后", "修订", "补充", "修正")

_PDF_MAGIC_BYTES: Final[bytes] = b"%PDF-"
_PDF_MIN_BYTES: Final[int] = 1024  # 1 KiB；正常财报 PDF 至少几百 KB。
_CNINFO_ADJUNCT_TYPE_PDF: Final[str] = "PDF"

# A 股 ticker 前缀 -> 巨潮 column / plate 映射。
_TICKER_PREFIX_TO_MARKET_PARAMS: Final[
    dict[str, tuple[str, str]]
] = {
    # 深市主板 / 中小板 / 创业板
    "000": ("szse", "sz"),
    "001": ("szse", "sz"),
    "002": ("szse", "sz"),
    "003": ("szse", "sz"),
    "300": ("szse", "sz"),
    "301": ("szse", "sz"),
    # 沪市主板 / 科创板
    "600": ("sse", "sh"),
    "601": ("sse", "sh"),
    "603": ("sse", "sh"),
    "605": ("sse", "sh"),
    "688": ("sse", "sh"),
}


@dataclass(frozen=True)
class _CninfoExchangeContext:
    """巨潮单只 ticker 的市场上下文。"""

    column: str  # "szse" | "sse"
    plate: str  # "sz" | "sh"


@dataclass(frozen=True)
class _CninfoCompanyLookupEntry:
    """巨潮 stockList 中的公司基础信息。"""

    code: str
    org_id: str
    company_name: str


class CninfoDiscoveryClient:
    """巨潮 CN discovery / 下载客户端。

    实现 :class:`CnReportDiscoveryClientProtocol`：``resolve_company`` /
    ``list_report_candidates`` / ``download_report_pdf``。

    构造参数允许测试注入 ``client`` 与回调，避免真实网络访问。
    """

    def __init__(
        self,
        *,
        client: Optional[httpx.Client] = None,
        user_agent: Optional[str] = None,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        request_timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        page_size: int = 30,
        sleep_func: Optional[Callable[[float], None]] = None,
    ) -> None:
        """初始化巨潮下载器。

        Args:
            client: 可选 ``httpx.Client``；测试可注入伪客户端。生产 ``None``
                时由本类管理生命周期。
            user_agent: HTTP UA；缺省走 :data:`DEFAULT_USER_AGENT`。
            sleep_seconds: 连续请求间隔，避免触发巨潮风控。
            max_retries: 单次 HTTP 请求最大重试次数（指数退避）。
            request_timeout_seconds: 单次请求超时。
            page_size: ``hisAnnouncement/query`` 单页大小。
            sleep_func: 用于注入的 sleep 实现，签名 ``Callable[[float], None]``；
                测试可注入 ``lambda _: None`` 跳过实际等待。

        Raises:
            ValueError: ``max_retries`` ≤ 0 / ``sleep_seconds`` < 0 时抛出。
        """

        if max_retries <= 0:
            raise ValueError("max_retries 必须大于 0")
        if sleep_seconds < 0:
            raise ValueError("sleep_seconds 不能为负数")
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=request_timeout_seconds,
            headers={"User-Agent": user_agent or DEFAULT_USER_AGENT},
        )
        self._user_agent = user_agent or DEFAULT_USER_AGENT
        self._sleep_seconds = sleep_seconds
        self._max_retries = max_retries
        self._request_timeout_seconds = request_timeout_seconds
        self._page_size = page_size
        if sleep_func is not None and not callable(sleep_func):
            raise ValueError("sleep_func 必须是可调用对象")
        self._sleep_func: Callable[[float], None] = (
            sleep_func if sleep_func is not None else time.sleep
        )
        self._last_request_finished_at: float | None = None
        self._stock_mapping_cache: dict[str, _CninfoCompanyLookupEntry] | None = None

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

    # ---------- Protocol 实现 ----------

    def resolve_company(self, query: CnReportQuery) -> CnCompanyProfile:
        """解析 ``query.normalized_ticker`` 对应的公司元数据。

        Args:
            query: 单次 download 查询参数；``query.market`` 必须为 ``"CN"``。

        Returns:
            :class:`CnCompanyProfile`，``company_id`` 形如 ``"CNINFO:{orgId}"``。

        Raises:
            ValueError: ``market`` 非 CN，或巨潮主源未命中此 ticker 时抛出。
            RuntimeError: 主源接口请求失败时抛出。
        """

        if query.market != "CN":
            raise ValueError(f"CninfoDiscoveryClient 仅支持 CN，收到 market={query.market!r}")
        ticker = query.normalized_ticker.strip()
        context = self._resolve_exchange_context(ticker)
        entry = self._resolve_company_lookup(ticker=ticker, context=context)
        return CnCompanyProfile(
            provider="cninfo",
            company_id=f"CNINFO:{entry.org_id}",
            company_name=entry.company_name,
            ticker=ticker,
        )

    def list_report_candidates(
        self,
        query: CnReportQuery,
        profile: CnCompanyProfile,
    ) -> tuple[CnReportCandidate, ...]:
        """列出符合 ``query.target_periods`` 与窗口约束的候选报告。

        实现细节：

        - 按 ``target_periods`` 分批调用 ``hisAnnouncement/query``，每个 fiscal
          period 单独使用对应 ``category_*_szsh;``，避免巨潮把多个分类混淆。
        - 同 fiscal period 多版本仅保留最新有效全文，amended 优先。
        - HEAD 拉取 ``content-length`` / ``etag`` / ``last-modified``；HEAD
          失败仅记录 ``Log.warn`` 不阻塞流程。

        Args:
            query: 单次 download 查询参数。
            profile: ``resolve_company`` 返回的公司元数据。

        Returns:
            候选报告 tuple；按 ``fiscal_year`` 降序、再按 ``fiscal_period``
            稳定顺序排序。

        Raises:
            ValueError: market/provider/company_id 非法时抛出。
            RuntimeError: 任一有效财期分类的底层请求或 JSON 解析失败时抛出。
        """

        if query.market != "CN":
            raise ValueError(f"CninfoDiscoveryClient 仅支持 CN，收到 market={query.market!r}")
        if profile.provider != "cninfo":
            raise ValueError(f"profile.provider 必须为 cninfo，收到 {profile.provider!r}")
        org_id = profile.company_id.removeprefix("CNINFO:")
        if not org_id:
            raise ValueError(f"profile.company_id 缺少 CNINFO: 前缀: {profile.company_id!r}")
        ticker = query.normalized_ticker.strip()
        context = self._resolve_exchange_context(ticker)

        per_period_year: dict[tuple[CnFiscalPeriod, int], list[_RawAnnouncement]] = {}
        for period in query.target_periods:
            category = _PERIOD_TO_CATEGORY.get(period)
            if category is None:
                if period in _CNINFO_UNSUPPORTED_INDEPENDENT_PERIODS:
                    Log.warn(
                        f"巨潮暂无独立 fiscal_period={period!r} 分类，已按无候选跳过",
                        module=_MODULE,
                    )
                    continue
                Log.warn(f"未知 fiscal_period={period!r}，已跳过", module=_MODULE)
                continue
            try:
                announcements = self._query_announcements(
                    column=context.column,
                    plate=context.plate,
                    stock=ticker,
                    org_id=org_id,
                    category=category,
                    start_date=query.start_date,
                    end_date=query.end_date,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"巨潮公告分类查询失败: ticker={ticker} period={period} category={category} error={exc}"
                ) from exc
            for item in announcements:
                if _is_title_blocked(item.title):
                    continue
                fiscal_year = _infer_fiscal_year(item.title, item.announcement_date)
                if fiscal_year is None:
                    continue
                per_period_year.setdefault((period, fiscal_year), []).append(item)

        candidates: list[CnReportCandidate] = []
        for (period, fiscal_year), items in per_period_year.items():
            best = _pick_best_announcement(items)
            if best is None:
                continue
            candidate = self._build_candidate_from_announcement(
                announcement=best,
                period=period,
                fiscal_year=fiscal_year,
            )
            if candidate is not None:
                candidates.append(candidate)
        candidates.sort(key=lambda c: (-c.fiscal_year, _PERIOD_SORT_KEY[c.fiscal_period]))
        return tuple(candidates)

    def download_report_pdf(self, candidate: CnReportCandidate) -> DownloadedReportAsset:
        """下载单份候选 PDF 并返回强类型资产对象。

        实现细节：

        - 校验 PDF magic bytes（``%PDF-``）与最小字节数；非 PDF 抛 RuntimeError。
        - 暂存路径为 ``tempfile.NamedTemporaryFile``，调用方取出字节后会
          ``unlink``。
        - HTTP 失败按 ``max_retries`` 重试（指数退避）。

        Args:
            candidate: 远端候选元数据。

        Returns:
            :class:`DownloadedReportAsset`。

        Raises:
            RuntimeError: 下载失败、PDF 校验失败、HTTP 状态码异常时抛出。
        """

        if candidate.provider != "cninfo":
            raise RuntimeError(
                f"CninfoDiscoveryClient 不支持 provider={candidate.provider!r}"
            )
        payload = self._http_download_bytes(candidate.source_url)
        if len(payload) < _PDF_MIN_BYTES:
            raise RuntimeError(
                f"PDF 字节数过小 ({len(payload)} bytes)，url={candidate.source_url}"
            )
        if not payload.startswith(_PDF_MAGIC_BYTES):
            raise RuntimeError(
                f"PDF magic bytes 校验失败，url={candidate.source_url}"
            )
        sha256 = _sha256_hex(payload)
        downloaded_at = _utc_now_isoformat()
        tmp_dir = Path(tempfile.gettempdir()) / "dayu_cn_downloads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="cninfo_",
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
            downloaded_at=downloaded_at,
        )

    # ---------- 内部辅助 ----------

    def _resolve_exchange_context(self, ticker: str) -> _CninfoExchangeContext:
        """根据 ticker 前缀决定巨潮 column / plate。

        Args:
            ticker: 已归一化的 6 位 A 股代码。

        Returns:
            :class:`_CninfoExchangeContext`。

        Raises:
            ValueError: ticker 不属于已知 A 股前缀时抛出（如北交所 / 新三板）。
        """

        if len(ticker) != 6 or not ticker.isdigit():
            raise ValueError(f"巨潮仅支持 6 位 A 股代码，收到 ticker={ticker!r}")
        prefix = ticker[:3]
        params = _TICKER_PREFIX_TO_MARKET_PARAMS.get(prefix)
        if params is None:
            raise ValueError(f"巨潮未支持的 A 股前缀 {prefix!r}（ticker={ticker!r}）")
        column, plate = params
        return _CninfoExchangeContext(column=column, plate=plate)

    def _resolve_company_lookup(
        self,
        *,
        ticker: str,
        context: _CninfoExchangeContext,
    ) -> _CninfoCompanyLookupEntry:
        """通过全市场 stockList 解析并缓存公司基础信息。

        Args:
            ticker: 已归一化的 6 位 A 股代码。
            context: ticker 对应的巨潮市场上下文。

        Returns:
            :class:`_CninfoCompanyLookupEntry`。

        Raises:
            ValueError: 公告搜索结果未命中 ticker 时抛出。
            RuntimeError: 主源接口请求失败或响应字段缺失时抛出。
        """

        del context
        mapping = self._fetch_stock_mapping()
        entry = mapping.get(ticker)
        if entry is None:
            raise ValueError(f"巨潮 stockList 未命中 ticker={ticker!r}")
        return entry

    def _fetch_stock_mapping(self) -> dict[str, _CninfoCompanyLookupEntry]:
        """拉取并缓存巨潮全市场 A 股 stockList。

        Args:
            无。

        Returns:
            ``code -> _CninfoCompanyLookupEntry`` 映射。

        Raises:
            RuntimeError: 主源接口请求失败或响应字段缺失时抛出。
        """

        if self._stock_mapping_cache is not None:
            return self._stock_mapping_cache
        payload = self._http_get_json(CNINFO_STOCK_JSON_URL)
        items = payload.get("stockList") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise RuntimeError(f"巨潮 stockList schema 异常: url={CNINFO_STOCK_JSON_URL}")
        mapping: dict[str, _CninfoCompanyLookupEntry] = {}
        for raw in items:
            if not isinstance(raw, dict):
                continue
            code = str(raw.get("code", "")).strip()
            org_id = str(raw.get("orgId", "")).strip()
            company_name = str(raw.get("zwjc", "")).strip()
            if not code or not org_id:
                continue
            mapping[code] = _CninfoCompanyLookupEntry(
                code=code,
                org_id=org_id,
                company_name=company_name or code,
            )
        self._stock_mapping_cache = mapping
        return mapping

    def _query_announcements(
        self,
        *,
        column: str,
        plate: str,
        stock: str,
        org_id: str,
        category: str,
        start_date: str,
        end_date: str,
    ) -> list[_RawAnnouncement]:
        """调 ``hisAnnouncement/query`` 拉取一类公告（自动翻页）。

        Args:
            column: ``szse`` / ``sse``。
            plate: ``sz`` / ``sh``。
            stock: 6 位股票代码。
            org_id: 巨潮 orgId。
            category: ``category_*_szsh;``。
            start_date: 窗口起点 ``YYYY-MM-DD``。
            end_date: 窗口终点 ``YYYY-MM-DD``。

        Returns:
            原始公告对象列表。

        Raises:
            RuntimeError: 主源响应不可解析时抛出。
        """

        announcements: list[_RawAnnouncement] = []
        page_num = 1
        while True:
            payload = self._query_announcement_page(
                column=column,
                plate=plate,
                stock=f"{stock},{org_id}",
                search_key="",
                category=category,
                start_date=start_date,
                end_date=end_date,
                page_num=page_num,
                page_size=self._page_size,
            )
            items = payload.get("announcements") if isinstance(payload, dict) else None
            if not isinstance(items, list) or not items:
                break
            for raw in items:
                parsed = _parse_raw_announcement(raw)
                if parsed is not None and parsed.sec_code == stock:
                    announcements.append(parsed)
            has_more = bool(payload.get("hasMore")) if isinstance(payload, dict) else False
            if not has_more:
                break
            page_num += 1
            if page_num > 50:
                Log.warn(
                    f"hisAnnouncement 翻页超过 50 页保护: stock={stock} category={category}",
                    module=_MODULE,
                )
                break
        return announcements

    def _query_announcement_page(
        self,
        *,
        column: str,
        plate: str,
        stock: str,
        search_key: str,
        category: str,
        start_date: str,
        end_date: str,
        page_num: int,
        page_size: int,
    ) -> JsonValue:
        """请求单页 ``hisAnnouncement/query``。

        Args:
            column: ``szse`` / ``sse``。
            plate: ``sz`` / ``sh``。
            stock: 巨潮 ``stock`` 字段；可为空，或 ``{code},{orgId}``。
            search_key: 巨潮全文搜索关键词；公司解析时传 ticker。
            category: 巨潮公告分类；为空表示不限制分类。
            start_date: 窗口起点 ``YYYY-MM-DD``。
            end_date: 窗口终点 ``YYYY-MM-DD``。
            page_num: 页码，从 1 开始。
            page_size: 单页大小。

        Returns:
            JSON 解析后的响应对象。

        Raises:
            RuntimeError: 请求 / 解析失败时抛出。
        """

        data = {
            "pageNum": str(page_num),
            "pageSize": str(page_size),
            "column": column,
            "tabName": "fulltext",
            "plate": plate,
            "stock": stock,
            "searchkey": search_key,
            "secid": "",
            "category": category,
            "trade": "",
            "seDate": f"{start_date}~{end_date}",
            "sortName": "time",
            "sortType": "desc",
            "isHLtitle": "true",
        }
        return self._http_post_form(CNINFO_QUERY_URL, data=data)

    def _build_candidate_from_announcement(
        self,
        *,
        announcement: _RawAnnouncement,
        period: CnFiscalPeriod,
        fiscal_year: int,
    ) -> Optional[CnReportCandidate]:
        """把 ``_RawAnnouncement`` 转为 :class:`CnReportCandidate`。

        Args:
            announcement: 已筛选公告。
            period: 当前 fiscal period（来自请求分类）。
            fiscal_year: 已由调用方解析的财年；为避免重复推断而由调用方注入。

        Returns:
            候选；HEAD 失败但其他字段齐全时仍返回 candidate（content_length /
            etag / last_modified 字段为 ``None``）。返回 ``None`` 表示
            adjunctUrl 缺失等致命缺陷。

        Raises:
            无（HEAD 失败软降级，不抛）。
        """

        if not announcement.adjunct_url:
            return None
        source_url = CNINFO_STATIC_BASE_URL + announcement.adjunct_url.lstrip("/")
        head_meta = self._http_head_meta(source_url)
        title_amended = any(token in announcement.title for token in _TITLE_AMENDED_TOKENS)
        return CnReportCandidate(
            provider="cninfo",
            source_id=announcement.announcement_id,
            source_url=source_url,
            title=announcement.title,
            language="zh",
            filing_date=announcement.announcement_date,
            fiscal_year=fiscal_year,
            fiscal_period=period,
            amended=title_amended,
            content_length=head_meta.content_length,
            etag=head_meta.etag,
            last_modified=head_meta.last_modified,
        )

    # ---------- HTTP 辅助 ----------

    def _http_get_json(self, url: str) -> JsonValue:
        """GET JSON。

        Args:
            url: 请求地址。

        Returns:
            JSON 解析后的对象（dict / list / 标量）。

        Raises:
            RuntimeError: 请求 / 解析失败时抛出。
        """

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                self._throttle_before_request()
                try:
                    response = self._client.get(url)
                    response.raise_for_status()
                    return cast(JsonValue, response.json())
                finally:
                    self._mark_request_finished()
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                Log.debug(
                    f"GET JSON 失败: url={url} attempt={attempt + 1} error={exc}",
                    module=_MODULE,
                )
                self._retry_backoff(attempt)
        raise RuntimeError(f"GET JSON 失败: url={url} error={last_exc}")

    def _http_post_form(self, url: str, data: dict[str, str]) -> JsonValue:
        """POST form-urlencoded JSON。

        Args:
            url: 请求地址。
            data: form 字段。

        Returns:
            JSON 解析后的对象。

        Raises:
            RuntimeError: 请求 / 解析失败时抛出。
        """

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                self._throttle_before_request()
                try:
                    response = self._client.post(url, data=data)
                    response.raise_for_status()
                    return cast(JsonValue, response.json())
                finally:
                    self._mark_request_finished()
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_exc = exc
                Log.debug(
                    f"POST form 失败: url={url} attempt={attempt + 1} error={exc}",
                    module=_MODULE,
                )
                self._retry_backoff(attempt)
        raise RuntimeError(f"POST form 失败: url={url} error={last_exc}")

    def _http_head_meta(self, url: str) -> "_HeadMeta":
        """HEAD 拉取 ``content-length`` / ``etag`` / ``last-modified``。

        Args:
            url: 文件 URL。

        Returns:
            :class:`_HeadMeta`；失败时所有字段均为 ``None``。

        Raises:
            无（软降级；失败仅打 warn）。
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
        content_length_header = response.headers.get("Content-Length")
        try:
            content_length = (
                int(content_length_header) if content_length_header is not None else None
            )
        except (TypeError, ValueError):
            content_length = None
        return _HeadMeta(
            content_length=content_length,
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )

    def _http_download_bytes(self, url: str) -> bytes:
        """带重试地下载文件字节。

        Args:
            url: 文件 URL。

        Returns:
            响应体字节。

        Raises:
            RuntimeError: 重试耗尽仍失败时抛出。
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
                Log.debug(
                    f"PDF 下载失败: url={url} attempt={attempt + 1} error={exc}",
                    module=_MODULE,
                )
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
        """指数退避：``RETRY_BACKOFF_BASE_SECONDS * 2**attempt_index``。"""

        if attempt_index >= self._max_retries - 1:
            return
        delay = RETRY_BACKOFF_BASE_SECONDS * (2**attempt_index)
        self._sleep_func(delay)


# ---------- 模块级私有辅助 ----------


@dataclass(frozen=True)
class _HeadMeta:
    """HEAD 响应中的 fingerprint 输入字段。"""

    content_length: Optional[int]
    etag: Optional[str]
    last_modified: Optional[str]


@dataclass(frozen=True)
class _RawAnnouncement:
    """巨潮 ``hisAnnouncement/query`` 单条公告的强类型抽象。"""

    sec_code: str
    announcement_id: str
    title: str
    announcement_date: str
    adjunct_url: str


_PERIOD_SORT_KEY: Final[dict[CnFiscalPeriod, int]] = {
    "FY": 0,
    "H1": 1,
    "Q1": 2,
    "Q2": 3,
    "Q3": 4,
    "Q4": 5,
}

_TITLE_FY_PATTERN: Final[re.Pattern[str]] = re.compile(r"(\d{4})\s*年[年度]?\s*(年度报告|年报)")
_TITLE_FISCAL_YEAR_FALLBACK: Final[re.Pattern[str]] = re.compile(r"(\d{4})\s*年")
_CNINFO_HTML_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")


def _is_title_blocked(title: str) -> bool:
    """判断标题是否命中黑名单。

    Args:
        title: 公告标题。

    Returns:
        命中黑名单返回 ``True``；否则 ``False``。

    Raises:
        无。
    """

    lowered = title.lower()
    if any(token.lower() in lowered for token in _TITLE_BLOCKLIST):
        return True
    if _has_report_language_marker(title):
        return True
    has_report_title = any(token in title for token in _REPORT_TITLE_TOKENS)
    has_notice_title = any(token in title for token in _REPORT_NOTICE_TITLE_TOKENS)
    return has_report_title and has_notice_title


def _has_report_language_marker(title: str) -> bool:
    """判断财报标题是否带英文语言标记。

    Args:
        title: 公告标题。

    Returns:
        财报标题带 ``英文`` 语言标记时返回 ``True``。

    Raises:
        无。
    """

    if "英文" not in title:
        return False
    return any(token in title for token in _REPORT_TITLE_TOKENS)


def _pick_best_announcement(items: list[_RawAnnouncement]) -> Optional[_RawAnnouncement]:
    """从同一 fiscal period 的多条公告中挑最新有效全文。

    挑选规则：
    - 标题命中 :data:`_TITLE_AMENDED_TOKENS`（更正 / 修订 等）优先级最高；
    - 同优先级下取 ``announcement_date`` 字典序最大（即最新披露日期）。

    Args:
        items: 同 fiscal period 的候选列表。

    Returns:
        最佳候选；输入空列表返回 ``None``。

    Raises:
        无。
    """

    if not items:
        return None

    def sort_key(announcement: _RawAnnouncement) -> tuple[int, str]:
        is_amended = any(token in announcement.title for token in _TITLE_AMENDED_TOKENS)
        return (1 if is_amended else 0, announcement.announcement_date)

    return max(items, key=sort_key)


def _parse_raw_announcement(raw: JsonValue) -> Optional[_RawAnnouncement]:
    """把巨潮原始公告 dict 转为 :class:`_RawAnnouncement`。

    Args:
        raw: ``announcements`` 列表中的单条 dict。

    Returns:
        强类型公告；缺失关键字段返回 ``None``。

    Raises:
        无。
    """

    if not isinstance(raw, dict):
        return None
    sec_code = str(raw.get("secCode", "")).strip()
    adjunct_type = str(raw.get("adjunctType", "")).strip().upper()
    announcement_id = str(raw.get("announcementId", "")).strip()
    title = _clean_cninfo_text(str(raw.get("announcementTitle", "")).strip())
    adjunct_url = str(raw.get("adjunctUrl", "")).strip()
    raw_time = raw.get("announcementTime")
    announcement_date = _format_announcement_date(raw_time)
    if adjunct_type != _CNINFO_ADJUNCT_TYPE_PDF:
        return None
    if not sec_code or not announcement_id or not title or not adjunct_url or not announcement_date:
        return None
    return _RawAnnouncement(
        sec_code=sec_code,
        announcement_id=announcement_id,
        title=title,
        announcement_date=announcement_date,
        adjunct_url=adjunct_url,
    )


def _clean_cninfo_text(text: str) -> str:
    """清洗巨潮返回的高亮 HTML 文本。

    Args:
        text: 巨潮文本字段，可能包含 ``<em>`` 高亮标签。

    Returns:
        去掉 HTML 标签并压缩首尾空白后的文本。

    Raises:
        无。
    """

    without_tags = _CNINFO_HTML_TAG_PATTERN.sub("", text)
    return without_tags.strip()


def _format_announcement_date(raw_time: JsonValue) -> Optional[str]:
    """把巨潮 ``announcementTime`` 规范为 ``YYYY-MM-DD``。

    巨潮返回的 ``announcementTime`` 既可能是毫秒级时间戳整数，也可能是
    ``YYYY-MM-DD`` 字符串；本函数兼容这两种形态。

    Args:
        raw_time: 原始字段。

    Returns:
        ``YYYY-MM-DD`` 字符串；无法解析返回 ``None``。

    Raises:
        无。
    """

    if isinstance(raw_time, (int, float)):
        try:
            timestamp = float(raw_time) / 1000.0
            local = time.gmtime(timestamp)
            return time.strftime("%Y-%m-%d", local)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(raw_time, str):
        text = raw_time.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return text
        if text.isdigit():
            try:
                timestamp = float(text) / 1000.0
                local = time.gmtime(timestamp)
                return time.strftime("%Y-%m-%d", local)
            except (OverflowError, OSError, ValueError):
                return None
    return None


def _infer_fiscal_year(title: str, announcement_date: str) -> Optional[int]:
    """从标题与披露日期推断 fiscal year。

    优先匹配标题 ``YYYY 年年度报告`` 形态；失败则按 ``YYYY 年``；再失败按
    ``announcement_date`` 推断（H1 / 三季度 等中报通常披露当年；FY 通常披露次年）。

    Args:
        title: 公告标题。
        announcement_date: 披露日期 ``YYYY-MM-DD``。

    Returns:
        推断 fiscal year；解析失败返回 ``None``。

    Raises:
        无。
    """

    matched = _TITLE_FY_PATTERN.search(title)
    if matched is not None:
        return int(matched.group(1))
    fallback = _TITLE_FISCAL_YEAR_FALLBACK.search(title)
    if fallback is not None:
        return int(fallback.group(1))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", announcement_date):
        return int(announcement_date[:4])
    return None


def _sha256_hex(payload: bytes) -> str:
    """计算 PDF 字节内容 SHA-256。"""

    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _utc_now_isoformat() -> str:
    """生成 ISO-8601 UTC 时间戳，秒级精度。"""

    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "CNINFO_BASE_URL",
    "CNINFO_QUERY_URL",
    "CNINFO_STOCK_JSON_URL",
    "CNINFO_STATIC_BASE_URL",
    "CninfoDiscoveryClient",
]
