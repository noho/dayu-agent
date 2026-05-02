"""``dayu/fins/downloaders/cninfo_downloader.py`` 单元测试。

覆盖：

- ticker 前缀路由（深市 / 沪市 / 北交所拒绝）；
- 全市场 ``szse_stock.json`` 公司解析与缓存；
- ``hisAnnouncement/query`` 翻页、标题黑名单、amended 优先；
- HEAD 软失败软降级；
- PDF magic bytes 校验、最小字节数校验；
- 重试与下载失败路径。

策略：使用 ``httpx.MockTransport`` 注入 fixture 响应，避免真实网络。
"""

from __future__ import annotations

import json
from typing import Callable, TypeAlias
from urllib.parse import parse_qs

import httpx
import pytest

from dayu.fins.downloaders.cninfo_downloader import (
    CNINFO_QUERY_URL,
    CNINFO_STOCK_JSON_URL,
    CNINFO_STATIC_BASE_URL,
    CninfoDiscoveryClient,
)
from dayu.fins.pipelines.cn_download_models import (
    CnCompanyProfile,
    CnReportCandidate,
    CnReportQuery,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


# ---------- helpers ----------


def _build_pdf_payload(size: int = 4096, marker: bytes = b"%PDF-1.7\n") -> bytes:
    """构造合法 PDF 字节，前缀 ``%PDF-`` + 填充。"""

    body = marker + b"\x00" * (size - len(marker))
    return body


def _stock_mapping_payload() -> dict[str, list[dict[str, str]]]:
    """构造巨潮全市场 ``szse_stock.json`` fixture。"""

    return {
        "stockList": [
            {"code": "000001", "orgId": "gssz0000001", "zwjc": "平安银行"},
            {"code": "002594", "orgId": "gssz0002594", "zwjc": "比亚迪"},
            {"code": "600519", "orgId": "gssh0600519", "zwjc": "贵州茅台"},
            {"code": "688981", "orgId": "gshk0000981", "zwjc": "中芯国际"},
        ]
    }


def _build_announcement(
    *,
    announcement_id: str,
    title: str,
    announcement_date: str,
    adjunct_url: str,
    sec_code: str = "002594",
    sec_name: str = "比亚迪",
    org_id: str = "gssz0002594",
) -> dict[str, JsonValue]:
    """构造 ``hisAnnouncement/query`` 单条公告 fixture。

    巨潮真实 ``announcementTime`` 为毫秒时间戳整数；测试统一用字符串
    ``YYYY-MM-DD`` 来简化断言。
    """

    if "贵州茅台" in title and sec_code == "002594":
        sec_code = "600519"
        sec_name = "贵州茅台"
        org_id = "gssh0600519"
    return {
        "announcementId": announcement_id,
        "announcementTitle": title,
        "announcementTime": announcement_date,
        "adjunctUrl": adjunct_url,
        "adjunctType": "PDF",
        "secCode": sec_code,
        "secName": sec_name,
        "tileSecName": sec_name,
        "orgId": org_id,
    }


def _build_transport(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    """构造一个使用 ``MockTransport`` 的 ``httpx.Client``。"""

    transport = httpx.MockTransport(handler)
    return httpx.Client(transport=transport, timeout=5.0)


def _build_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> CninfoDiscoveryClient:
    """构造测试专用 ``CninfoDiscoveryClient``，关闭 sleep。"""

    http_client = _build_transport(handler)
    return CninfoDiscoveryClient(
        client=http_client,
        sleep_seconds=0.0,
        max_retries=2,
        sleep_func=lambda _delay: None,
    )


def _read_form(request: httpx.Request) -> dict[str, str]:
    """读取测试请求中的 form-urlencoded 字段。"""

    parsed = parse_qs(request.content.decode(), keep_blank_values=True)
    return {key: values[0] if values else "" for key, values in parsed.items()}


def _stock_mapping_response() -> httpx.Response:
    """返回巨潮全市场 stockList fixture 响应。"""

    return httpx.Response(200, json=_stock_mapping_payload())


# ---------- resolve_company ----------


def test_resolve_company_szse_ticker_returns_cninfo_prefix() -> None:
    """深市 ticker -> 调全市场 stockList -> ``CNINFO:{orgId}``。"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CNINFO_STOCK_JSON_URL
        return _stock_mapping_response()

    client = _build_client(handler)
    profile = client.resolve_company(
        CnReportQuery(
            market="CN",
            normalized_ticker="002594",
            start_date="2024-01-01",
            end_date="2025-12-31",
            target_periods=("FY",),
        )
    )

    assert profile == CnCompanyProfile(
        provider="cninfo",
        company_id="CNINFO:gssz0002594",
        company_name="比亚迪",
        ticker="002594",
    )


def test_resolve_company_sse_ticker_uses_all_market_stock_list() -> None:
    """沪市 ticker 也通过同一个全市场 stockList 解析 orgId。"""

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == CNINFO_STOCK_JSON_URL
        return _stock_mapping_response()

    client = _build_client(handler)
    profile = client.resolve_company(
        CnReportQuery(
            market="CN",
            normalized_ticker="600519",
            start_date="2024-01-01",
            end_date="2025-12-31",
            target_periods=("FY",),
        )
    )

    assert profile.company_id == "CNINFO:gssh0600519"
    assert profile.company_name == "贵州茅台"


def test_resolve_company_unknown_ticker_raises_value_error() -> None:
    """主源 stockList 未命中 -> ``ValueError``，调用方据此升级 failed。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"stockList": []})

    client = _build_client(handler)
    with pytest.raises(ValueError):
        client.resolve_company(
            CnReportQuery(
                market="CN",
                normalized_ticker="002594",
                start_date="2024-01-01",
                end_date="2025-12-31",
                target_periods=("FY",),
            )
        )


def test_resolve_company_rejects_non_cn_market() -> None:
    """``market != CN`` 立即报错，避免 HK 流量误流入巨潮。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"announcements": [], "hasMore": False})

    client = _build_client(handler)
    with pytest.raises(ValueError):
        client.resolve_company(
            CnReportQuery(
                market="HK",
                normalized_ticker="0700",
                start_date="2024-01-01",
                end_date="2025-12-31",
                target_periods=("FY",),
            )
        )


def test_resolve_company_rejects_non_a_share_prefix() -> None:
    """非 A 股前缀（如 8 起首北交所）必须报错，不静默走默认分支。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"announcements": [], "hasMore": False})

    client = _build_client(handler)
    with pytest.raises(ValueError):
        client.resolve_company(
            CnReportQuery(
                market="CN",
                normalized_ticker="830001",
                start_date="2024-01-01",
                end_date="2025-12-31",
                target_periods=("FY",),
            )
        )


def test_resolve_company_caches_stock_mapping_response() -> None:
    """全市场 stockList 仅请求一次：第二次 ``resolve_company`` 命中缓存。"""

    call_counter = {"stock_list": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == CNINFO_STOCK_JSON_URL:
            call_counter["stock_list"] += 1
            return _stock_mapping_response()
        raise AssertionError(f"unexpected url {request.url}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="002594",
        start_date="2024-01-01",
        end_date="2025-12-31",
        target_periods=("FY",),
    )
    client.resolve_company(query)
    client.resolve_company(query)

    assert call_counter["stock_list"] == 1


def test_resolve_company_does_not_depend_on_noisy_search_first_page() -> None:
    """000001 公司解析不得依赖 searchkey 公告第一页精确命中。"""

    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        if str(request.url) == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        raise AssertionError(f"unexpected url {request.url}")

    client = _build_client(handler)
    profile = client.resolve_company(
        CnReportQuery(
            market="CN",
            normalized_ticker="000001",
            start_date="2024-01-01",
            end_date="2025-12-31",
            target_periods=("FY",),
        )
    )

    assert profile.company_id == "CNINFO:gssz0000001"
    assert profile.company_name == "平安银行"
    assert seen_urls == [CNINFO_STOCK_JSON_URL]


# ---------- list_report_candidates ----------


def test_list_report_candidates_filters_blocklisted_titles() -> None:
    """标题命中黑名单（摘要 / 英文版 / 港股公告 等）必须被排除。"""

    announcement_payload = {
        "announcements": [
            _build_announcement(
                announcement_id="A1",
                title="比亚迪：2024年年度报告",
                announcement_date="2025-04-03",
                adjunct_url="finalpage/2025-04-03/full.PDF",
            ),
            _build_announcement(
                announcement_id="A2",
                title="比亚迪：2024年年度报告摘要",
                announcement_date="2025-04-03",
                adjunct_url="finalpage/2025-04-03/summary.PDF",
            ),
            _build_announcement(
                announcement_id="A3",
                title="比亚迪：2024 Annual Report (English)",
                announcement_date="2025-04-03",
                adjunct_url="finalpage/2025-04-03/english.PDF",
            ),
            _build_announcement(
                announcement_id="A4",
                title="比亚迪：港股公告：2024年年报",
                announcement_date="2025-04-10",
                adjunct_url="finalpage/2025-04-10/hk.PDF",
            ),
        ],
        "hasMore": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            return httpx.Response(200, json=announcement_payload)
        if request.method == "HEAD":
            return httpx.Response(
                200,
                headers={
                    "Content-Length": "12345",
                    "ETag": '"abc"',
                    "Last-Modified": "Wed, 03 Apr 2025 02:00:00 GMT",
                },
            )
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _build_client(handler)
    profile = client.resolve_company(
        CnReportQuery(
            market="CN",
            normalized_ticker="002594",
            start_date="2024-01-01",
            end_date="2025-12-31",
            target_periods=("FY",),
        )
    )
    candidates = client.list_report_candidates(
        CnReportQuery(
            market="CN",
            normalized_ticker="002594",
            start_date="2024-01-01",
            end_date="2025-12-31",
            target_periods=("FY",),
        ),
        profile,
    )

    assert len(candidates) == 1
    only = candidates[0]
    assert only.source_id == "A1"
    assert only.fiscal_period == "FY"
    assert only.fiscal_year == 2024
    assert only.source_url == CNINFO_STATIC_BASE_URL + "finalpage/2025-04-03/full.PDF"
    assert only.content_length == 12345
    assert only.etag == '"abc"'


def test_list_report_candidates_prefers_a_share_fy_over_later_h_share_notice() -> None:
    """688981 年报候选应排除更晚披露的港股公告，选择 A 股年度报告。"""

    announcement_payload = {
        "announcements": [
            _build_announcement(
                announcement_id="HK1",
                title="港股公告：2025年年报",
                announcement_date="2026-04-09",
                adjunct_url="finalpage/2026-04-09/hk.PDF",
                sec_code="688981",
                sec_name="中芯国际",
                org_id="gshk0000981",
            ),
            _build_announcement(
                announcement_id="A1",
                title="中芯国际2025年年度报告",
                announcement_date="2026-03-27",
                adjunct_url="finalpage/2026-03-27/a.PDF",
                sec_code="688981",
                sec_name="中芯国际",
                org_id="gshk0000981",
            ),
            _build_announcement(
                announcement_id="S1",
                title="中芯国际2025年年度报告摘要",
                announcement_date="2026-03-27",
                adjunct_url="finalpage/2026-03-27/summary.PDF",
                sec_code="688981",
                sec_name="中芯国际",
                org_id="gshk0000981",
            ),
        ],
        "hasMore": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            return httpx.Response(200, json=announcement_payload)
        if request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="688981",
        start_date="2026-01-01",
        end_date="2026-05-02",
        target_periods=("FY",),
    )
    profile = client.resolve_company(query)
    candidates = client.list_report_candidates(query, profile)

    assert [candidate.source_id for candidate in candidates] == ["A1"]


def test_list_report_candidates_filters_non_pdf_and_other_sec_code() -> None:
    """非 PDF 或非目标证券公告不得进入候选。"""

    announcement_payload = {
        "announcements": [
            _build_announcement(
                announcement_id="A1",
                title="比亚迪：2024年年度报告",
                announcement_date="2025-04-03",
                adjunct_url="finalpage/2025-04-03/full.PDF",
            ),
            {
                **_build_announcement(
                    announcement_id="A2",
                    title="比亚迪：2024年年度报告",
                    announcement_date="2025-04-04",
                    adjunct_url="finalpage/2025-04-04/not-pdf.txt",
                ),
                "adjunctType": "TXT",
            },
            _build_announcement(
                announcement_id="A3",
                title="其他公司：2024年年度报告",
                announcement_date="2025-04-05",
                adjunct_url="finalpage/2025-04-05/other.PDF",
                sec_code="000001",
                sec_name="平安银行",
                org_id="gssz0000001",
            ),
        ],
        "hasMore": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            return httpx.Response(200, json=announcement_payload)
        if request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="002594",
        start_date="2024-01-01",
        end_date="2025-12-31",
        target_periods=("FY",),
    )
    profile = client.resolve_company(query)
    candidates = client.list_report_candidates(query, profile)

    assert [candidate.source_id for candidate in candidates] == ["A1"]


def test_list_report_candidates_amended_takes_priority() -> None:
    """同 fiscal period 多版本：``更正`` 优先于普通版本。"""

    announcements_payload = {
        "announcements": [
            _build_announcement(
                announcement_id="A1",
                title="贵州茅台：2024年年度报告",
                announcement_date="2025-04-01",
                adjunct_url="finalpage/2025-04-01/v1.PDF",
            ),
            _build_announcement(
                announcement_id="A2",
                title="贵州茅台：2024年年度报告（更正后）",
                announcement_date="2025-04-15",
                adjunct_url="finalpage/2025-04-15/v2.PDF",
            ),
        ],
        "hasMore": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            return httpx.Response(200, json=announcements_payload)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"Content-Length": "9999"})
        raise AssertionError(f"unexpected {request}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="600519",
        start_date="2024-01-01",
        end_date="2025-12-31",
        target_periods=("FY",),
    )
    profile = client.resolve_company(query)
    candidates = client.list_report_candidates(query, profile)

    assert len(candidates) == 1
    assert candidates[0].source_id == "A2"
    assert candidates[0].amended is True


def test_list_report_candidates_keeps_one_per_year_for_fy() -> None:
    """同一 fiscal_period=FY、不同 fiscal_year 必须各保留一份（窗口内全量）。"""

    announcements_payload = {
        "announcements": [
            _build_announcement(
                announcement_id="A1",
                title="贵州茅台：2024年年度报告",
                announcement_date="2025-04-01",
                adjunct_url="finalpage/2025-04-01/v1.PDF",
            ),
            _build_announcement(
                announcement_id="A2",
                title="贵州茅台：2023年年度报告",
                announcement_date="2024-04-01",
                adjunct_url="finalpage/2024-04-01/v1.PDF",
            ),
            _build_announcement(
                announcement_id="A3",
                title="贵州茅台：2022年年度报告",
                announcement_date="2023-04-01",
                adjunct_url="finalpage/2023-04-01/v1.PDF",
            ),
        ],
        "hasMore": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            return httpx.Response(200, json=announcements_payload)
        if request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected {request}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="600519",
        start_date="2022-01-01",
        end_date="2025-12-31",
        target_periods=("FY",),
    )
    profile = client.resolve_company(query)
    candidates = client.list_report_candidates(query, profile)

    assert [c.fiscal_year for c in candidates] == [2024, 2023, 2022]
    assert [c.source_id for c in candidates] == ["A1", "A2", "A3"]


def test_list_report_candidates_picks_amended_per_year_without_dropping_other_years() -> None:
    """同一年同 period 更正优先，但不同 fiscal_year 必须都保留。"""

    announcements_payload = {
        "announcements": [
            _build_announcement(
                announcement_id="A1",
                title="贵州茅台：2024年年度报告",
                announcement_date="2025-04-01",
                adjunct_url="finalpage/2025-04-01/2024-v1.PDF",
            ),
            _build_announcement(
                announcement_id="A2",
                title="贵州茅台：2024年年度报告（更正后）",
                announcement_date="2025-04-10",
                adjunct_url="finalpage/2025-04-10/2024-v2.PDF",
            ),
            _build_announcement(
                announcement_id="A3",
                title="贵州茅台：2023年年度报告",
                announcement_date="2024-04-01",
                adjunct_url="finalpage/2024-04-01/2023-v1.PDF",
            ),
        ],
        "hasMore": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            return httpx.Response(200, json=announcements_payload)
        if request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected {request}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="600519",
        start_date="2023-01-01",
        end_date="2025-12-31",
        target_periods=("FY",),
    )
    profile = client.resolve_company(query)
    candidates = client.list_report_candidates(query, profile)

    assert [(c.fiscal_year, c.source_id, c.amended) for c in candidates] == [
        (2024, "A2", True),
        (2023, "A3", False),
    ]


def test_list_report_candidates_handles_pagination() -> None:
    """``hasMore=True`` 时按 ``pageNum`` 翻页直到 ``hasMore=False``。"""

    pages = {
        "1": {
            "announcements": [
                _build_announcement(
                    announcement_id="A1",
                    title="比亚迪：2024年年度报告",
                    announcement_date="2025-04-01",
                    adjunct_url="finalpage/p1/A1.PDF",
                )
            ],
            "hasMore": True,
        },
        "2": {
            "announcements": [
                _build_announcement(
                    announcement_id="A2",
                    title="比亚迪：2023年年度报告",
                    announcement_date="2024-04-01",
                    adjunct_url="finalpage/p2/A2.PDF",
                )
            ],
            "hasMore": False,
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            payload_bytes = request.content
            form = dict(item.split("=", 1) for item in payload_bytes.decode().split("&"))
            return httpx.Response(200, json=pages[form["pageNum"]])
        if request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected {request}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="002594",
        start_date="2023-01-01",
        end_date="2025-12-31",
        target_periods=("FY",),
    )
    profile = client.resolve_company(query)
    candidates = client.list_report_candidates(query, profile)

    # 两条 announcement 同 fiscal_period=FY，但属于不同 fiscal_year；
    # 按 (period, fiscal_year) 分组去重 -> 两条都保留，按年份降序排。
    assert len(candidates) == 2
    assert [c.source_id for c in candidates] == ["A1", "A2"]
    assert [c.fiscal_year for c in candidates] == [2024, 2023]


def test_list_report_candidates_head_failure_softly_degrades() -> None:
    """HEAD 失败 -> candidate 仍生成，content_length / etag / last_modified=None。"""

    payload = {
        "announcements": [
            _build_announcement(
                announcement_id="A1",
                title="比亚迪：2024年年度报告",
                announcement_date="2025-04-01",
                adjunct_url="finalpage/2025-04-01/full.PDF",
            )
        ],
        "hasMore": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            return httpx.Response(200, json=payload)
        if request.method == "HEAD":
            return httpx.Response(500)
        raise AssertionError(f"unexpected {request}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="002594",
        start_date="2024-01-01",
        end_date="2025-12-31",
        target_periods=("FY",),
    )
    profile = client.resolve_company(query)
    candidates = client.list_report_candidates(query, profile)

    assert len(candidates) == 1
    only = candidates[0]
    assert only.content_length is None
    assert only.etag is None
    assert only.last_modified is None


def test_list_report_candidates_empty_when_no_announcements() -> None:
    """巨潮返回空 ``announcements`` -> 返回空 tuple，不抛。"""

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            return httpx.Response(200, json={"announcements": [], "hasMore": False})
        raise AssertionError(f"unexpected {request}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="002594",
        start_date="2024-01-01",
        end_date="2025-12-31",
        target_periods=("FY", "H1"),
    )
    profile = client.resolve_company(query)
    candidates = client.list_report_candidates(query, profile)

    assert candidates == ()


def test_list_report_candidates_invalid_profile_provider_raises() -> None:
    """``profile.provider`` 非 cninfo 必须报错，避免与 HK 链路串线。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"announcements": [], "hasMore": False})

    client = _build_client(handler)
    bogus_profile = CnCompanyProfile(
        provider="hkexnews",
        company_id="HKEX:7609",
        company_name="腾讯控股",
        ticker="0700",
    )
    with pytest.raises(ValueError):
        client.list_report_candidates(
            CnReportQuery(
                market="CN",
                normalized_ticker="002594",
                start_date="2024-01-01",
                end_date="2025-12-31",
                target_periods=("FY",),
            ),
            bogus_profile,
        )


def test_list_report_candidates_uses_per_period_category() -> None:
    """每个 fiscal_period 单独发请求，category 各不相同。"""

    seen_categories: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            form = dict(item.split("=", 1) for item in request.content.decode().split("&"))
            seen_categories.append(form["category"])
            return httpx.Response(200, json={"announcements": [], "hasMore": False})
        raise AssertionError(f"unexpected {request}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="002594",
        start_date="2024-01-01",
        end_date="2025-12-31",
        target_periods=("FY", "H1", "Q1", "Q3"),
    )
    profile = client.resolve_company(query)
    client.list_report_candidates(query, profile)

    # URL-encoded 分号 -> %3B
    assert sorted(seen_categories) == sorted(
        [
            "category_ndbg_szsh%3B",
            "category_bndbg_szsh%3B",
            "category_yjdbg_szsh%3B",
            "category_sjdbg_szsh%3B",
        ]
    )


# ---------- download_report_pdf ----------


def _make_candidate(*, source_url: str = "https://example.com/test.pdf") -> CnReportCandidate:
    return CnReportCandidate(
        provider="cninfo",
        source_id="A1",
        source_url=source_url,
        title="比亚迪：2024年年度报告",
        language="zh",
        filing_date="2025-04-01",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
        content_length=4096,
        etag='"abc"',
        last_modified="Wed, 03 Apr 2025 02:00:00 GMT",
    )


def test_download_report_pdf_returns_asset_with_sha256() -> None:
    """成功下载 -> 返回 :class:`DownloadedReportAsset`，sha256 正确。"""

    payload = _build_pdf_payload()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=payload)

    client = _build_client(handler)
    asset = client.download_report_pdf(_make_candidate())

    assert asset.candidate.source_id == "A1"
    assert asset.content_length == len(payload)
    # sha256 已知可验证
    import hashlib

    assert asset.sha256 == hashlib.sha256(payload).hexdigest()
    assert asset.pdf_path.exists()
    assert asset.pdf_path.read_bytes().startswith(b"%PDF-")
    asset.pdf_path.unlink()


def test_download_report_pdf_rejects_short_content() -> None:
    """字节数 < 1 KiB 视为破损，抛 ``RuntimeError``。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"%PDF-tiny")

    client = _build_client(handler)
    with pytest.raises(RuntimeError):
        client.download_report_pdf(_make_candidate())


def test_download_report_pdf_rejects_non_pdf_magic() -> None:
    """非 ``%PDF-`` 起始字节 -> 抛 ``RuntimeError``。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>" + b"\x00" * 4096)

    client = _build_client(handler)
    with pytest.raises(RuntimeError):
        client.download_report_pdf(_make_candidate())


def test_download_report_pdf_retries_then_raises() -> None:
    """连续失败耗尽重试 -> 抛 ``RuntimeError``。"""

    call_counter = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        call_counter["n"] += 1
        return httpx.Response(503)

    client = _build_client(handler)
    with pytest.raises(RuntimeError):
        client.download_report_pdf(_make_candidate())

    # max_retries=2 -> 至少 2 次实际请求。
    assert call_counter["n"] == 2


def test_download_report_pdf_rejects_non_cninfo_provider() -> None:
    """``candidate.provider != cninfo`` -> 抛 ``RuntimeError``。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_build_pdf_payload())

    client = _build_client(handler)
    bogus = CnReportCandidate(
        provider="hkexnews",
        source_id="X",
        source_url="https://example.com/x.pdf",
        title="t",
        language="zh",
        filing_date="2025-01-01",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
        content_length=None,
        etag=None,
        last_modified=None,
    )
    with pytest.raises(RuntimeError):
        client.download_report_pdf(bogus)


# ---------- 其他 ----------


def test_announcement_time_milliseconds_is_normalized() -> None:
    """巨潮 ``announcementTime`` 为毫秒整数时也能解析为 ``YYYY-MM-DD``。"""

    # 2025-04-03 00:00:00 UTC = 1743638400000 ms
    payload = {
        "announcements": [
            {
                "announcementId": "A1",
                "announcementTitle": "比亚迪：2024年年度报告",
                "announcementTime": 1743638400000,
                "adjunctUrl": "finalpage/2025-04-03/full.PDF",
                "adjunctType": "PDF",
                "secCode": "002594",
                "secName": "比亚迪",
                "tileSecName": "比亚迪",
                "orgId": "gssz0002594",
            }
        ],
        "hasMore": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            return httpx.Response(200, json=payload)
        if request.method == "HEAD":
            return httpx.Response(200, headers={})
        raise AssertionError(f"unexpected {request}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="002594",
        start_date="2024-01-01",
        end_date="2025-12-31",
        target_periods=("FY",),
    )
    profile = client.resolve_company(query)
    candidates = client.list_report_candidates(query, profile)

    assert candidates[0].filing_date == "2025-04-03"


def test_close_releases_owned_http_client() -> None:
    """``close`` 必须释放自管 HTTP client 资源。"""

    client = CninfoDiscoveryClient(sleep_seconds=0.0, sleep_func=lambda _delay: None)
    client.close()


def test_constructor_rejects_invalid_max_retries() -> None:
    """``max_retries <= 0`` 立即报错。"""

    with pytest.raises(ValueError):
        CninfoDiscoveryClient(max_retries=0)


def test_constructor_rejects_negative_sleep_seconds() -> None:
    """``sleep_seconds < 0`` 立即报错。"""

    with pytest.raises(ValueError):
        CninfoDiscoveryClient(sleep_seconds=-0.1)


def test_invalid_json_response_raises_runtime_error() -> None:
    """主源响应非 JSON -> 抛 ``RuntimeError``，不静默降级。"""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    client = _build_client(handler)
    with pytest.raises(RuntimeError):
        client.resolve_company(
            CnReportQuery(
                market="CN",
                normalized_ticker="002594",
                start_date="2024-01-01",
                end_date="2025-12-31",
                target_periods=("FY",),
            )
        )


def test_serialize_query_payload_structure_assertion() -> None:
    """直接断言 ``hisAnnouncement/query`` 收到的关键字段，固化对外契约。"""

    seen_payload: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str == CNINFO_STOCK_JSON_URL:
            return _stock_mapping_response()
        if url_str == CNINFO_QUERY_URL:
            seen_payload.update(_read_form(request))
            return httpx.Response(200, json={"announcements": [], "hasMore": False})
        raise AssertionError(f"unexpected {request}")

    client = _build_client(handler)
    query = CnReportQuery(
        market="CN",
        normalized_ticker="002594",
        start_date="2024-01-01",
        end_date="2025-12-31",
        target_periods=("FY",),
    )
    profile = client.resolve_company(query)
    client.list_report_candidates(query, profile)

    assert seen_payload["stock"] == "002594,gssz0002594"
    assert seen_payload["column"] == "szse"
    assert seen_payload["plate"] == "sz"
    assert seen_payload["seDate"] == "2024-01-01~2025-12-31"


def test_stock_mapping_filters_incomplete_rows() -> None:
    """``code`` / ``orgId`` 缺失的 stockList 行必须被过滤。"""

    incomplete_payload = {
        "stockList": [
            {"code": "", "orgId": "gssz0000001", "zwjc": "X"},
            {"code": "002594", "orgId": "", "zwjc": "Y"},
            {"code": "002594", "orgId": "gssz0002594", "zwjc": "比亚迪"},
        ]
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=incomplete_payload)

    client = _build_client(handler)
    profile = client.resolve_company(
        CnReportQuery(
            market="CN",
            normalized_ticker="002594",
            start_date="2024-01-01",
            end_date="2025-12-31",
            target_periods=("FY",),
        )
    )

    assert profile.company_id == "CNINFO:gssz0002594"
    assert profile.company_name == "比亚迪"


def test_form_payload_serialization_round_trip_via_json() -> None:
    """使用 ``json.dumps`` 反序列化 fixture payload，确保测试构造与生产一致。"""

    sample = json.dumps(_stock_mapping_payload(), ensure_ascii=False)
    payload = json.loads(sample)
    assert payload["stockList"][1]["code"] == "002594"
    assert payload["stockList"][1]["orgId"] == "gssz0002594"
