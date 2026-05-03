"""``dayu/fins/downloaders/hkexnews_downloader.py`` 单元测试。

覆盖披露易 stock list 解析、title search 参数、语言策略、多代码匹配、
季度空结果与 PDF 校验。所有测试都通过 ``httpx.MockTransport`` 注入 fixture，
禁止访问真实披露易网络。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeAlias
from urllib.parse import parse_qs

import httpx
import pytest

from dayu.fins.downloaders.hkexnews_downloader import (
    HKEXNEWS_ACTIVE_STOCK_ZH_URL,
    HKEXNEWS_BASE_URL,
    HKEXNEWS_INACTIVE_STOCK_ZH_URL,
    HKEXNEWS_TITLE_SEARCH_URL,
    HkexnewsDiscoveryClient,
)
from dayu.fins.pipelines.cn_download_models import (
    CnCompanyProfile,
    CnFiscalPeriod,
    CnReportCandidate,
    CnReportQuery,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_PDF_URL = f"{HKEXNEWS_BASE_URL}/listedco/listconews/sehk/2025/0401/2025040100001.pdf"


def _build_pdf_payload(size: int = 4096, marker: bytes = b"%PDF-1.7\n") -> bytes:
    """构造测试 PDF 字节。

    Args:
        size: 字节数。
        marker: 文件前缀。

    Returns:
        PDF 字节。

    Raises:
        无。
    """

    return marker + b"0" * (size - len(marker))


def _stock_mapping_payload() -> dict[str, list[dict[str, str]]]:
    """构造披露易 stock list fixture。

    Args:
        无。

    Returns:
        stock list JSON。

    Raises:
        无。
    """

    return {
        "stockInfo": [
            {"c": "00700", "i": "7609", "n": "腾讯控股"},
            {"c": "00005", "i": "5", "n": "汇丰控股"},
        ]
    }


def _empty_stock_mapping_payload() -> dict[str, list[dict[str, str]]]:
    """构造空 stock list fixture。

    Args:
        无。

    Returns:
        空 stock list JSON。

    Raises:
        无。
    """

    return {"stockInfo": []}


def _inactive_duplicate_stock_mapping_payload() -> dict[str, list[dict[str, str]]]:
    """构造含历史重复代码的 inactive stock list fixture。

    Args:
        无。

    Returns:
        inactive stock list JSON。

    Raises:
        无。
    """

    return {"stockInfo": [{"c": "00700", "i": "1639", "n": "八佰伴國際"}]}


def _announcement(
    *,
    document_id: str,
    title: str,
    file_link: str = "/listedco/listconews/sehk/2025/0401/2025040100001.pdf",
    stock_code: str = "00700<br/>80700",
    date_time: str = "01/04/2025 16:30",
    category_text: str = "Financial Statements/ESG Information - [Annual Report]",
) -> dict[str, str]:
    """构造 title search 单条公告 fixture。

    Args:
        document_id: 披露易文档 ID。
        title: 公告标题。
        file_link: PDF 链接。
        stock_code: ``STOCK_CODE`` 字段。
        date_time: 披露时间。
        category_text: 披露易分类文本。

    Returns:
        公告 JSON dict。

    Raises:
        无。
    """

    return {
        "NEWS_ID": document_id,
        "TITLE": title,
        "FILE_LINK": file_link,
        "STOCK_CODE": stock_code,
        "DATE_TIME": date_time,
        "FILE_TYPE": "PDF",
        "LONG_TEXT": category_text,
    }


def _query_from_request(request: httpx.Request) -> dict[str, tuple[str, ...]]:
    """解析 GET query 参数。

    Args:
        request: HTTP 请求。

    Returns:
        字段名到值 tuple 的映射。

    Raises:
        无。
    """

    parsed = parse_qs(str(request.url).split("?", 1)[1] if "?" in str(request.url) else "")
    return {key: tuple(values) for key, values in parsed.items()}


def _title_search_payload(rows: list[dict[str, str]]) -> dict[str, str]:
    """构造披露易 title search 响应。

    Args:
        rows: 结果行。

    Returns:
        ``result`` 为字符串 JSON 的响应 dict。

    Raises:
        无。
    """

    return {"result": json.dumps(rows, ensure_ascii=False)}


def _build_http_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    """构造 MockTransport HTTP client。

    Args:
        handler: 请求处理函数。

    Returns:
        HTTP client。

    Raises:
        无。
    """

    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


def _build_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> HkexnewsDiscoveryClient:
    """构造测试用披露易客户端。

    Args:
        handler: 请求处理函数。

    Returns:
        披露易 downloader。

    Raises:
        无。
    """

    return HkexnewsDiscoveryClient(
        client=_build_http_client(handler),
        sleep_seconds=0.0,
        max_retries=2,
        sleep_func=lambda _delay: None,
    )


def _query(
    *,
    ticker: str = "0700",
    periods: tuple[CnFiscalPeriod, ...] = ("FY",),
) -> CnReportQuery:
    """构造 HK 查询对象。

    Args:
        ticker: ticker。
        periods: 财期 tuple。

    Returns:
        查询对象。

    Raises:
        无。
    """

    return CnReportQuery(
        market="HK",
        normalized_ticker=ticker,
        start_date="2024-01-01",
        end_date="2026-12-31",
        target_periods=periods,
    )


def _profile(ticker: str = "0700") -> CnCompanyProfile:
    """构造 HK 公司元数据。

    Args:
        ticker: ticker。

    Returns:
        公司元数据。

    Raises:
        无。
    """

    return CnCompanyProfile(
        provider="hkexnews",
        company_id="HKEX:7609",
        company_name="腾讯控股",
        ticker=ticker,
    )


def test_resolve_company_parses_active_stock_list_and_normalizes_ticker() -> None:
    """验证 ``0700/00700/700.HK`` 都能命中 5 位 stock code。"""

    requested_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if str(request.url) == HKEXNEWS_ACTIVE_STOCK_ZH_URL:
            return httpx.Response(200, json=_stock_mapping_payload())
        if str(request.url) == HKEXNEWS_INACTIVE_STOCK_ZH_URL:
            return httpx.Response(200, json=_inactive_duplicate_stock_mapping_payload())
        raise AssertionError(f"unexpected url {request.url}")

    client = _build_client(handler)

    for ticker in ("0700", "00700", "700.HK"):
        profile = client.resolve_company(_query(ticker=ticker))
        assert profile == CnCompanyProfile(
            provider="hkexnews",
            company_id="HKEX:7609",
            company_name="腾讯控股",
            ticker=ticker,
        )

    assert requested_urls == [
        HKEXNEWS_ACTIVE_STOCK_ZH_URL,
        HKEXNEWS_INACTIVE_STOCK_ZH_URL,
    ]


def test_resolve_company_rejects_non_hk_market() -> None:
    """``market != HK`` 时立即拒绝。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_empty_stock_mapping_payload())

    client = _build_client(handler)
    with pytest.raises(ValueError):
        client.resolve_company(
            CnReportQuery(
                market="CN",
                normalized_ticker="0700",
                start_date="2024-01-01",
                end_date="2026-12-31",
                target_periods=("FY",),
            )
        )


def test_list_report_candidates_gets_title_search_and_builds_absolute_url() -> None:
    """验证 title search GET 参数、``FILE_LINK`` 绝对 URL 与多代码过滤。"""

    posted_forms: list[dict[str, tuple[str, ...]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(HKEXNEWS_TITLE_SEARCH_URL) and request.method == "GET":
            form = _query_from_request(request)
            posted_forms.append(form)
            if form["lang"] == ("zh",):
                return httpx.Response(
                    200,
                    json=_title_search_payload(
                        [
                            _announcement(
                                document_id="DOC1",
                                title="腾讯控股有限公司：2024年年度报告",
                            ),
                            _announcement(
                                document_id="DOC2",
                                title="汇丰控股有限公司：2024年年度报告",
                                stock_code="00005",
                            ),
                        ]
                    ),
                )
            return httpx.Response(
                200,
                json=_title_search_payload(
                    [
                        _announcement(
                            document_id="DOC1_EN",
                            title="Tencent Holdings Limited: 2024 Annual Report",
                        )
                    ]
                ),
            )
        if str(request.url) == _PDF_URL and request.method == "HEAD":
            return httpx.Response(
                200,
                headers={
                    "Content-Length": "4096",
                    "ETag": '"hk-v1"',
                    "Last-Modified": "Tue, 01 Apr 2025 00:00:00 GMT",
                },
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _build_client(handler)
    candidates = client.list_report_candidates(_query(), _profile())

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.provider == "hkexnews"
    assert candidate.source_id == "DOC1"
    assert candidate.source_url == _PDF_URL
    assert candidate.language == "zh"
    assert candidate.fiscal_year == 2024
    assert candidate.fiscal_period == "FY"
    assert candidate.content_length == 4096
    assert candidate.etag == '"hk-v1"'
    assert candidate.last_modified == "Tue, 01 Apr 2025 00:00:00 GMT"
    assert posted_forms[0]["stockId"] == ("7609",)
    assert posted_forms[0]["searchType"] == ("1",)
    assert posted_forms[0]["t1code"] == ("40000",)
    assert posted_forms[0]["t2Gcode"] == ("-2",)
    assert posted_forms[0]["t2code"] == ("40100",)
    assert posted_forms[0]["fromDate"] == ("20240101",)
    assert posted_forms[0]["toDate"] == ("20261231",)


def test_list_report_candidates_uses_secondary_language_only_when_primary_empty() -> None:
    """主语言为空时允许英文补位，并把 language 标为 ``en``。"""

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(HKEXNEWS_TITLE_SEARCH_URL) and request.method == "GET":
            form = _query_from_request(request)
            if form["lang"] == ("zh",):
                return httpx.Response(200, json={"result": "[]"})
            return httpx.Response(
                200,
                json=_title_search_payload(
                    [
                        _announcement(
                            document_id="DOC_EN",
                            title="Tencent Holdings Limited: 2024 Annual Report",
                        )
                    ]
                ),
            )
        if str(request.url) == _PDF_URL and request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _build_client(handler)
    candidates = client.list_report_candidates(_query(), _profile())

    assert len(candidates) == 1
    assert candidates[0].source_id == "DOC_EN"
    assert candidates[0].language == "en"


def test_list_report_candidates_maps_hk_period_codes_and_allows_empty_quarters() -> None:
    """验证 FY/H1/Q1/Q3 标题分类映射；季度查无不抛异常。"""

    category_params: list[tuple[str, str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(HKEXNEWS_TITLE_SEARCH_URL) and request.method == "GET":
            form = _query_from_request(request)
            category_params.append(
                (
                    form["t1code"][0],
                    form["t2Gcode"][0],
                    form["t2code"][0],
                )
            )
            return httpx.Response(200, json={"result": "[]"})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _build_client(handler)
    candidates = client.list_report_candidates(
        _query(periods=("FY", "H1", "Q1", "Q3")),
        _profile(),
    )

    assert candidates == ()
    assert category_params == [
        ("40000", "-2", "40100"),
        ("40000", "-2", "40100"),
        ("40000", "-2", "40200"),
        ("40000", "-2", "40200"),
        ("10000", "3", "13600"),
        ("10000", "3", "13600"),
        ("10000", "3", "13600"),
        ("10000", "3", "13600"),
    ]


def test_list_report_candidates_treats_traditional_half_year_as_h1() -> None:
    """真实繁体 ``中期/半年度報告`` 分类必须归入 H1 而非 FY。"""

    h1_url = f"{HKEXNEWS_BASE_URL}/listedco/listconews/sehk/2025/0826/h1.pdf"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(HKEXNEWS_TITLE_SEARCH_URL) and request.method == "GET":
            form = _query_from_request(request)
            if form["lang"] == ("E",):
                return httpx.Response(200, json={"result": "[]"})
            return httpx.Response(
                200,
                json=_title_search_payload(
                    [
                        _announcement(
                            document_id="H1_2025",
                            title="中期報告 2025",
                            file_link="/listedco/listconews/sehk/2025/0826/h1.pdf",
                            date_time="26/08/2025 16:30",
                            category_text="財務報表/環境、社會及管治資料 - [中期/半年度報告]",
                        ),
                    ]
                ),
            )
        if str(request.url) == h1_url and request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _build_client(handler)
    candidates = client.list_report_candidates(
        _query(periods=("FY", "H1")),
        _profile(),
    )

    assert [(candidate.source_id, candidate.fiscal_period) for candidate in candidates] == [
        ("H1_2025", "H1"),
    ]


def test_list_report_candidates_filters_q1_q3_by_title_period() -> None:
    """同一季度业绩分类结果必须按标题区分 Q1/Q3，不能互相误标。"""

    first_quarter_url = f"{HKEXNEWS_BASE_URL}/listedco/listconews/sehk/2025/0420/q1.pdf"
    third_quarter_url = f"{HKEXNEWS_BASE_URL}/listedco/listconews/sehk/2025/1020/q3.pdf"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(HKEXNEWS_TITLE_SEARCH_URL) and request.method == "GET":
            form = _query_from_request(request)
            assert form["t1code"] == ("10000",)
            assert form["t2Gcode"] == ("3",)
            assert form["t2code"] == ("13600",)
            if form["lang"] == ("E",):
                return httpx.Response(200, json={"result": "[]"})
            return httpx.Response(
                200,
                json=_title_search_payload(
                    [
                        _announcement(
                            document_id="Q1_2024",
                            title="FIRST QUARTERLY REPORT 2024",
                            file_link="/listedco/listconews/sehk/2025/0420/q1.pdf",
                            date_time="2025-04-20 16:30",
                            category_text="Financial Statements/ESG Information - [Quarterly Report]",
                        ),
                        _announcement(
                            document_id="Q3_2024",
                            title="THIRD QUARTERLY REPORT 2024",
                            file_link="/listedco/listconews/sehk/2025/1020/q3.pdf",
                            date_time="2025-10-20 16:30",
                            category_text="Financial Statements/ESG Information - [Quarterly Report]",
                        ),
                    ]
                ),
            )
        if str(request.url) in {first_quarter_url, third_quarter_url} and request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _build_client(handler)
    candidates = client.list_report_candidates(
        _query(periods=("Q1", "Q3")),
        _profile(),
    )

    assert [(candidate.source_id, candidate.fiscal_period) for candidate in candidates] == [
        ("Q1_2024", "Q1"),
        ("Q3_2024", "Q3"),
    ]


def test_list_report_candidates_reads_hk_quarterly_results_announcements() -> None:
    """真实腾讯式 ``公告及通告 - [季度業績]`` 应归入 Q1/Q3。"""

    q1_url = f"{HKEXNEWS_BASE_URL}/listedco/listconews/sehk/2025/0514/q1.pdf"
    q3_url = f"{HKEXNEWS_BASE_URL}/listedco/listconews/sehk/2025/1113/q3.pdf"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(HKEXNEWS_TITLE_SEARCH_URL) and request.method == "GET":
            form = _query_from_request(request)
            assert form["t1code"] == ("10000",)
            assert form["t2Gcode"] == ("3",)
            assert form["t2code"] == ("13600",)
            if form["lang"] == ("E",):
                return httpx.Response(200, json={"result": "[]"})
            return httpx.Response(
                200,
                json=_title_search_payload(
                    [
                        _announcement(
                            document_id="Q3_2025",
                            title="截至二零二五年九月三十日止三個月及九個月業績公佈",
                            file_link="/listedco/listconews/sehk/2025/1113/q3.pdf",
                            date_time="13/11/2025 16:30",
                            category_text="公告及通告 - [季度業績]",
                        ),
                        _announcement(
                            document_id="Q1_2025",
                            title="截至二零二五年三月三十一日止三個月業績公佈",
                            file_link="/listedco/listconews/sehk/2025/0514/q1.pdf",
                            date_time="14/05/2025 16:31",
                            category_text="公告及通告 - [季度業績]",
                        ),
                    ]
                ),
            )
        if str(request.url) in {q1_url, q3_url} and request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _build_client(handler)
    candidates = client.list_report_candidates(
        _query(periods=("Q1", "Q3")),
        _profile(),
    )

    assert [(candidate.source_id, candidate.fiscal_period) for candidate in candidates] == [
        ("Q1_2025", "Q1"),
        ("Q3_2025", "Q3"),
    ]


def test_list_report_candidates_groups_by_year_and_prefers_amended() -> None:
    """同一 period 多年度都保留；同年同 period 更正版优先。"""

    pdf_urls = {
        f"{HKEXNEWS_BASE_URL}/listedco/listconews/sehk/2025/0401/2024_regular.pdf",
        f"{HKEXNEWS_BASE_URL}/listedco/listconews/sehk/2025/0402/2024_amended.pdf",
        f"{HKEXNEWS_BASE_URL}/listedco/listconews/sehk/2024/0401/2023_regular.pdf",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(HKEXNEWS_TITLE_SEARCH_URL) and request.method == "GET":
            form = _query_from_request(request)
            if form["lang"] == ("E",):
                return httpx.Response(200, json={"result": "[]"})
            return httpx.Response(
                200,
                json=_title_search_payload(
                    [
                        _announcement(
                            document_id="FY2024_REGULAR",
                            title="腾讯控股有限公司：2024年年度报告",
                            file_link="/listedco/listconews/sehk/2025/0401/2024_regular.pdf",
                            date_time="2025-04-03 16:30",
                        ),
                        _announcement(
                            document_id="FY2024_AMENDED",
                            title="腾讯控股有限公司：2024年年度报告（修訂）",
                            file_link="/listedco/listconews/sehk/2025/0402/2024_amended.pdf",
                            date_time="2025-04-02 16:30",
                        ),
                        _announcement(
                            document_id="FY2023_REGULAR",
                            title="腾讯控股有限公司：2023年年度报告",
                            file_link="/listedco/listconews/sehk/2024/0401/2023_regular.pdf",
                            date_time="2024-04-01 16:30",
                        ),
                    ]
                ),
            )
        if str(request.url) in pdf_urls and request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _build_client(handler)
    candidates = client.list_report_candidates(_query(), _profile())

    assert [candidate.fiscal_year for candidate in candidates] == [2024, 2023]
    assert [candidate.source_id for candidate in candidates] == [
        "FY2024_AMENDED",
        "FY2023_REGULAR",
    ]
    assert candidates[0].amended is True


def test_download_report_pdf_returns_asset_for_valid_pdf() -> None:
    """合法 PDF 下载返回资产对象。"""

    pdf_payload = _build_pdf_payload()

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == _PDF_URL
        return httpx.Response(200, content=pdf_payload)

    candidate = CnReportCandidate(
        provider="hkexnews",
        source_id="DOC1",
        source_url=_PDF_URL,
        title="Tencent Holdings Limited: 2024 Annual Report",
        language="en",
        filing_date="2025-04-01",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
        content_length=len(pdf_payload),
        etag=None,
        last_modified=None,
    )

    client = _build_client(handler)
    asset = client.download_report_pdf(candidate)

    assert asset.candidate == candidate
    assert asset.content_length == len(pdf_payload)
    assert asset.pdf_path.read_bytes() == pdf_payload
    asset.pdf_path.unlink()


def test_download_report_pdf_rejects_short_or_non_pdf_payload() -> None:
    """短文件或非 PDF magic bytes 必须被拒绝。"""

    payloads = [b"%PDF-", b"not-a-pdf" + b"0" * 2048]

    for payload in payloads:
        def handler(request: httpx.Request, payload: bytes = payload) -> httpx.Response:
            assert str(request.url) == _PDF_URL
            return httpx.Response(200, content=payload)

        candidate = CnReportCandidate(
            provider="hkexnews",
            source_id="DOC_BAD",
            source_url=_PDF_URL,
            title="Tencent Holdings Limited: 2024 Annual Report",
            language="en",
            filing_date="2025-04-01",
            fiscal_year=2024,
            fiscal_period="FY",
            amended=False,
            content_length=len(payload),
            etag=None,
            last_modified=None,
        )

        client = _build_client(handler)
        with pytest.raises(RuntimeError):
            client.download_report_pdf(candidate)
