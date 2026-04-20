"""SecDownloader 异步实现测试。"""

from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
from itertools import repeat
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Optional

import httpx
import pytest

from dayu.fins.downloaders.sec_downloader import (
    BrowseEdgarFiling,
    DownloaderEvent,
    RemoteFileDescriptor,
    Sc13PartyRoles,
    SecDownloader,
    _await_if_needed,
    _parse_browse_edgar_atom,
    _parse_browse_edgar_href,
    _parse_index_header_document_entries,
    _parse_sc13_party_roles_from_index_headers,
    _parse_retry_after,
    _load_sec_throttle_state,
    _resolve_sec_throttle_delay,
    _select_primary_from_index_items,
    extract_same_filing_linked_html_files,
    _format_accession_with_dash,
    accession_to_no_dash,
    build_source_fingerprint,
    pick_exhibit_files,
    pick_extracted_instance_xml,
    pick_form_document_files,
    pick_taxonomy_files,
)
from dayu.fins.domain.document_models import FileObjectMeta
from dayu.fins._converters import optional_int


class StoreStub:
    """用于捕获 `download_files` 存储回调。"""

    def __init__(self) -> None:
        """初始化存储桩。"""

        self.calls: list[tuple[str, bytes]] = []

    def __call__(self, filename: str, stream: BinaryIO) -> FileObjectMeta:
        """存储回调实现。"""

        payload = stream.read()
        self.calls.append((filename, payload))
        return FileObjectMeta(uri=f"mem://{filename}", size=len(payload))


def _run(coro: Any) -> Any:
    """在同步测试中执行协程。"""

    return asyncio.run(coro)


def _create_downloader(tmp_path: Path) -> SecDownloader:
    """创建下载器实例并设置最小重试参数。"""

    downloader = SecDownloader(workspace_root=tmp_path)
    downloader.configure(user_agent="UA", sleep_seconds=0.0, max_retries=1)
    return downloader


def test_basic_helpers_cover_edge_cases(tmp_path: Path) -> None:
    """验证基础辅助函数边界行为。"""

    downloader = _create_downloader(tmp_path)
    assert downloader.normalize_ticker(" aapl ") == "AAPL"
    with pytest.raises(ValueError, match="ticker 不能为空"):
        downloader.normalize_ticker("  ")

    assert accession_to_no_dash(" 0001-0002 ") == "00010002"
    with pytest.raises(ValueError, match="accession_number 不能为空"):
        accession_to_no_dash(" ")

    items = [
        {"name": "sample-htm.xml"},
        {"name": "a_pre.xml"},
        {"name": "a_cal.xml"},
        {"name": "a_def.xml"},
        {"name": "a_lab.xml"},
        {"name": "sample.xsd"},
        {"name": "d123dex991.htm"},
        {"name": "x_other.xml"},
    ]
    assert pick_extracted_instance_xml(items) == "sample-htm.xml"
    assert pick_taxonomy_files(items) == ["a_cal.xml", "a_def.xml", "a_lab.xml", "a_pre.xml", "sample.xsd"]
    assert pick_exhibit_files(items) == ["d123dex991.htm"]
    # Edgar Filing Services 格式: ex99-1, ex99_1
    items_ex99 = [
        {"name": "tm257183d1_6k.htm"},
        {"name": "tm257183d1_ex99-1.htm"},
        {"name": "tm257183d1_ex99-2.htm"},
        {"name": "tm257183d1_ex99-1img01.jpg"},
        {"name": "index.html"},
    ]
    assert pick_exhibit_files(items_ex99) == ["tm257183d1_ex99-1.htm", "tm257183d1_ex99-2.htm"]
    # 混合格式
    items_mixed = [
        {"name": "d123dex992.htm"},
        {"name": "tm_ex99_1.htm"},
    ]
    assert pick_exhibit_files(items_mixed) == ["d123dex992.htm", "tm_ex99_1.htm"]
    # index-headers 类型识别：文件名不包含 ex99 但 type=EX-99.1/EX-99.2
    items_by_type = [
        {"name": "q12025pressrelease.htm", "type": "EX-99.1"},
        {"name": "q12025interimfinancialrepo.htm", "type": "EX-99.2"},
        {"name": "form6-k.htm", "type": "6-K"},
    ]
    assert pick_exhibit_files(items_by_type) == [
        "q12025interimfinancialrepo.htm",
        "q12025pressrelease.htm",
    ]
    assert pick_form_document_files(items_by_type, "6-K") == ["form6-k.htm"]

    assert optional_int(None) is None
    assert optional_int("") is None
    assert optional_int("abc") is None
    assert optional_int("12") == 12
    assert _format_accession_with_dash("000116737925000017") == "0001167379-25-000017"
    assert _format_accession_with_dash("bad-accession") == "bad-accession"


def test_await_if_needed_ignores_non_awaitable_dunder_await_marker() -> None:
    """验证 `_await_if_needed` 不会把伪 `__await__` 属性误判为可等待对象。"""

    class FakeAwaitMarker:
        """暴露伪 `__await__` 属性但并非 awaitable 的对象。"""

        __await__ = None

    marker = FakeAwaitMarker()

    assert _run(_await_if_needed(marker)) is marker


def test_extract_same_filing_linked_html_files_filters_to_same_archive_html() -> None:
    """验证主文档补链只保留同归档相对 HTML 文件。"""

    payload = b"""
<html>
  <body>
    <a href="d940644dex1.htm">linked exhibit</a>
    <a href="./d940644dex2.html?ref=cover">linked exhibit 2</a>
    <a href="#section-1">anchor</a>
    <a href="javascript:void(0)">script</a>
    <a href="https://www.sec.gov/Archives/edgar/data/1/2/external.htm">external</a>
    <a href="../outside.htm">outside</a>
    <a href="nested/other.htm">nested</a>
    <a href="chart.jpg">image</a>
    <a href="sample-6k.htm">self</a>
  </body>
</html>
"""

    assert extract_same_filing_linked_html_files(payload=payload, primary_document="sample-6k.htm") == [
        "d940644dex1.htm",
        "d940644dex2.html",
    ]


def test_build_source_fingerprint_is_order_independent() -> None:
    """验证 source_fingerprint 对输入顺序不敏感。"""

    descriptors = [
        RemoteFileDescriptor(
            name="b.htm",
            source_url="https://e/b.htm",
            http_etag="e2",
            http_last_modified="t2",
            remote_size=2,
        ),
        RemoteFileDescriptor(
            name="a.htm",
            source_url="https://e/a.htm",
            http_etag="e1",
            http_last_modified="t1",
            remote_size=1,
        ),
    ]
    fp1 = build_source_fingerprint(descriptors)
    fp2 = build_source_fingerprint(list(reversed(descriptors)))
    assert fp1 == fp2


def test_build_source_fingerprint_ignores_transport_level_etag_variants() -> None:
    """验证 source_fingerprint 忽略传输层 ETag/长度抖动。"""

    base = [
        RemoteFileDescriptor(
            name="a.xml",
            source_url="https://e/a.xml",
            http_etag='"abc123"',
            http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
            remote_size=7_110_372,
        ),
        RemoteFileDescriptor(
            name="b.xml",
            source_url="https://e/b.xml",
            http_etag="W/\"def456\"",
            http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
            remote_size=264_243,
        ),
    ]
    transport_variant = [
        RemoteFileDescriptor(
            name="a.xml",
            source_url="https://e/a.xml",
            http_etag='"abc123-gzip"',
            http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
            remote_size=20,
        ),
        RemoteFileDescriptor(
            name="b.xml",
            source_url="https://e/b.xml",
            http_etag='"def456"',
            http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
            remote_size=20,
        ),
    ]

    assert build_source_fingerprint(base) == build_source_fingerprint(transport_variant)


def test_parse_browse_edgar_helpers() -> None:
    """验证 browse-edgar 解析相关辅助函数。"""

    payload = b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<feed xmlns=\"http://www.w3.org/2005/Atom\">
  <entry>
    <title>SC 13G/A [Amend]  - Statement of Beneficial Ownership</title>
    <updated>2025-08-10T12:00:00-05:00</updated>
    <link href=\"https://www.sec.gov/Archives/edgar/data/1000/000000000025000777/0000000000-25-000777-index.htm\"/>
  </entry>
</feed>
"""
    results = _parse_browse_edgar_atom(payload)
    assert results == [
        BrowseEdgarFiling(
            form_type="SC 13G/A",
            filing_date="2025-08-10",
            accession_number="0000000000-25-000777",
            cik="1000",
            index_url="https://www.sec.gov/Archives/edgar/data/1000/000000000025000777/0000000000-25-000777-index.htm",
        )
    ]

    accession, cik = _parse_browse_edgar_href(
        "https://www.sec.gov/Archives/edgar/data/1000/000000000025000777/0000000000-25-000777-index.htm"
    )
    assert accession == "0000000000-25-000777"
    assert cik == "1000"


def test_select_primary_from_index_items_fallback() -> None:
    """验证 index 条目选择主文件的兜底逻辑。"""

    items: list[dict[str, Any]] = [
        {"name": "", "type": "10-K"},
        {"name": "doc.bin", "type": "EX-99"},
        {"name": "primary.htm", "type": "10-K"},
    ]
    assert _select_primary_from_index_items(items, "10-K") == "primary.htm"


def test_resolve_company_success_and_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 resolve_company 成功与失败分支。"""

    downloader = _create_downloader(tmp_path)
    monkeypatch.setattr(
        downloader,
        "_http_get_json",
        lambda url: {
            "0": {"ticker": "AAPL", "cik_str": "320193", "title": "Apple"},
            "1": {"ticker": "MSFT", "cik_str": "789019", "title": "Microsoft"},
        },
    )
    assert _run(downloader.resolve_company("aapl")) == ("320193", "Apple", "0000320193")

    monkeypatch.setattr(downloader, "_http_get_json", lambda url: {"0": {"ticker": "MSFT"}})
    monkeypatch.setattr(
        downloader,
        "_http_get_bytes",
        lambda url: b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<feed xmlns=\"http://www.w3.org/2005/Atom\"></feed>""",
    )
    with pytest.raises(RuntimeError, match="无法在 SEC ticker map 中找到"):
        _run(downloader.resolve_company("AAPL"))


def test_resolve_company_fallback_via_browse_edgar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 resolve_company 在 ticker map miss 时可回退 browse-edgar。"""

    downloader = _create_downloader(tmp_path)

    def _fake_get_json(url: str) -> dict[str, Any]:
        if url.endswith("company_tickers.json"):
            return {"0": {"ticker": "MSFT", "cik_str": "789019", "title": "Microsoft"}}
        if "CIK0000814052.json" in url:
            return {
                "name": "TELEFONICA S A",
                "tickers": ["TEFOF", "TEF", "TELFY"],
            }
        raise AssertionError(f"未预期的 JSON URL: {url}")

    monkeypatch.setattr(downloader, "_http_get_json", _fake_get_json)
    monkeypatch.setattr(
        downloader,
        "_http_get_bytes",
        lambda url: b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<feed xmlns=\"http://www.w3.org/2005/Atom\">
  <entry>
    <title>20-F - Annual report</title>
    <updated>2025-03-01T12:00:00-05:00</updated>
    <link href=\"https://www.sec.gov/Archives/edgar/data/814052/000119312525048701/0001193125-25-048701-index.htm\"/>
  </entry>
</feed>""",
    )

    assert _run(downloader.resolve_company("tef")) == ("814052", "TELEFONICA S A", "0000814052")


def test_fetch_wrappers_and_blank_filenum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 fetch 包装方法与空 filenum 分支。"""

    downloader = _create_downloader(tmp_path)
    captured_urls: list[str] = []

    def _fake_get_json(url: str) -> dict[str, Any]:
        captured_urls.append(url)
        return {"ok": True}

    monkeypatch.setattr(downloader, "_http_get_json", _fake_get_json)
    assert _run(downloader.fetch_submissions("0000320193")) == {"ok": True}
    assert _run(downloader.fetch_json("https://example.com/a.json")) == {"ok": True}
    assert any("CIK0000320193.json" in url for url in captured_urls)
    assert _run(downloader.fetch_browse_edgar_filenum("   ")) == []


def test_fetch_sc13_party_roles_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 SC13 方向角色解析成功路径。"""

    downloader = _create_downloader(tmp_path)
    captured_urls: list[str] = []
    payload = b"""
<SEC-HEADER>
FILED BY:
  COMPANY DATA:
    CENTRAL INDEX KEY: 0000886982
SUBJECT COMPANY:
  COMPANY DATA:
    CENTRAL INDEX KEY: 0000320193
</SEC-HEADER>
"""

    async def _fake_get_bytes(url: str) -> bytes:
        captured_urls.append(url)
        return payload

    monkeypatch.setattr(downloader, "_http_get_bytes", _fake_get_bytes)
    roles = _run(
        downloader.fetch_sc13_party_roles(
            archive_cik="0000886982",
            accession_number="0001193125-24-036431",
        )
    )
    assert roles == Sc13PartyRoles(filed_by_cik="886982", subject_cik="320193")
    assert captured_urls
    assert captured_urls[0].endswith(
        "/886982/000119312524036431/0001193125-24-036431-index-headers.html"
    )


def test_parse_sc13_party_roles_missing_field_returns_none() -> None:
    """验证 SC13 方向角色在字段缺失时返回 None。"""

    payload = b"""
<SEC-HEADER>
FILED BY:
  COMPANY DATA:
    CENTRAL INDEX KEY: 0000886982
</SEC-HEADER>
"""
    assert _parse_sc13_party_roles_from_index_headers(payload) is None


def test_fetch_sc13_party_roles_network_failure_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 SC13 方向角色在网络失败时返回 None。"""

    downloader = _create_downloader(tmp_path)

    async def _raise_error(url: str) -> bytes:
        del url
        raise RuntimeError("boom")

    monkeypatch.setattr(downloader, "_http_get_bytes", _raise_error)
    roles = _run(
        downloader.fetch_sc13_party_roles(
            archive_cik="886982",
            accession_number="0001193125-24-036431",
        )
    )
    assert roles is None


def test_list_filing_files_includes_xbrl_and_exhibits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 list_filing_files 会包含 XBRL、6-K cover 与 exhibits。"""

    downloader = _create_downloader(tmp_path)

    monkeypatch.setattr(
        downloader,
        "_try_fetch_index_items",
        lambda cik, accession_no_dash: [
            {"name": "sample-6k.htm"},
            {"name": "sample-6k_htm.xml"},
            {"name": "sample-6k.xsd"},
            {"name": "d123dex991.htm"},
        ],
    )
    monkeypatch.setattr(
        downloader,
        "_try_fetch_index_header_documents",
        lambda cik, accession_no_dash: [
            {"name": "form6kcover.htm", "type": "6-K", "description": "FORM 6-K"},
            {"name": "q12025pressrelease.htm", "type": "EX-99.1", "description": "EX-99.1"},
        ],
    )
    monkeypatch.setattr(
        downloader,
        "_http_head",
        lambda url, allow_redirects: type(
            "_Resp",
            (),
            {
                "headers": {
                    "ETag": '"etag"',
                    "Last-Modified": "Sat, 01 Feb 2025 12:00:00 GMT",
                    "Content-Length": "100",
                },
                "status_code": 200,
            },
        )(),
    )

    descriptors = _run(
        downloader.list_filing_files(
            cik="320193",
            accession_no_dash="000000000025000001",
            primary_document="sample-6k.htm",
            form_type="6-K",
            include_xbrl=True,
            include_exhibits=True,
        )
    )

    names = sorted([item.name for item in descriptors])
    assert names == sorted(
        [
            "sample-6k.htm",
            "sample-6k_htm.xml",
            "sample-6k.xsd",
            "d123dex991.htm",
            "form6kcover.htm",
            "q12025pressrelease.htm",
        ]
    )
    cover_descriptor = next(item for item in descriptors if item.name == "form6kcover.htm")
    assert cover_descriptor.sec_document_type == "6-K"
    assert cover_descriptor.sec_description == "FORM 6-K"
    exhibit_descriptor = next(item for item in descriptors if item.name == "q12025pressrelease.htm")
    assert exhibit_descriptor.sec_document_type == "EX-99.1"
    assert exhibit_descriptor.sec_description == "EX-99.1"


def test_list_filing_files_without_http_metadata_skips_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证首次下载可跳过文件级 HEAD 请求。"""

    downloader = _create_downloader(tmp_path)
    monkeypatch.setattr(
        downloader,
        "_try_fetch_index_items",
        lambda cik, accession_no_dash: [{"name": "sample-10k.htm"}, {"name": "sample-10k_htm.xml"}],
    )

    async def _unexpected_head(url: str, allow_redirects: bool) -> None:
        del url, allow_redirects
        raise AssertionError("include_http_metadata=False 时不应触发 HEAD")

    monkeypatch.setattr(downloader, "_http_head", _unexpected_head)

    descriptors = _run(
        downloader.list_filing_files(
            cik="320193",
            accession_no_dash="000000000025000001",
            primary_document="sample-10k.htm",
            form_type="10-K",
            include_http_metadata=False,
        )
    )
    assert [item.name for item in descriptors] == ["sample-10k.htm", "sample-10k_htm.xml"]
    assert all(item.http_etag is None for item in descriptors)
    assert all(item.http_last_modified is None for item in descriptors)
    assert all(item.http_status is None for item in descriptors)


def test_list_filing_files_includes_primary_linked_html_exhibits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 6-K 主文档中的同 filing 相对 HTML 链接会补入下载列表。"""

    downloader = _create_downloader(tmp_path)
    monkeypatch.setattr(downloader, "_try_fetch_index_items", lambda cik, accession_no_dash: [])
    monkeypatch.setattr(downloader, "_try_fetch_index_header_documents", lambda cik, accession_no_dash: [])

    async def _fake_get_bytes(url: str) -> bytes:
        """模拟返回带相对 exhibit 链接的 6-K cover。"""

        assert url.endswith("/sample-6k.htm")
        return b"""
<html>
  <body>
    <a href="d940644dex1.htm">Exhibit 1</a>
    <a href="./d940644dex2.htm?src=cover">Exhibit 2</a>
    <a href="nested/other.htm">Ignore nested</a>
    <a href="https://www.sec.gov/external.htm">Ignore external</a>
  </body>
</html>
"""

    monkeypatch.setattr(downloader, "_http_get_bytes", _fake_get_bytes)
    monkeypatch.setattr(
        downloader,
        "_http_head",
        lambda url, allow_redirects: type(
            "_Resp",
            (),
            {
                "headers": {
                    "ETag": '"etag"',
                    "Last-Modified": "Sat, 01 Feb 2025 12:00:00 GMT",
                    "Content-Length": "100",
                },
                "status_code": 200,
            },
        )(),
    )

    descriptors = _run(
        downloader.list_filing_files(
            cik="1762506",
            accession_no_dash="000119312525091353",
            primary_document="sample-6k.htm",
            form_type="6-K",
            include_xbrl=False,
            include_exhibits=True,
        )
    )

    assert sorted(item.name for item in descriptors) == [
        "d940644dex1.htm",
        "d940644dex2.htm",
        "sample-6k.htm",
    ]


def test_download_files_stream_304_downloaded_and_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 download_files_stream 的 skipped/downloaded/failed 分支。"""

    downloader = _create_downloader(tmp_path)
    store_stub = StoreStub()
    descriptors = [
        RemoteFileDescriptor(
            name="a.htm",
            source_url="https://example.com/a.htm",
            http_etag='"etag-a"',
            http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
            remote_size=1,
            http_status=200,
        ),
        RemoteFileDescriptor(
            name="b.htm",
            source_url="https://example.com/b.htm",
            http_etag='"etag-b"',
            http_last_modified="Mon, 01 Jan 2025 00:00:01 GMT",
            remote_size=2,
            http_status=200,
        ),
        RemoteFileDescriptor(
            name="c.htm",
            source_url="https://example.com/c.htm",
            http_etag='"etag-c"',
            http_last_modified="Mon, 01 Jan 2025 00:00:02 GMT",
            remote_size=3,
            http_status=200,
        ),
    ]

    async def _fake_conditional(url: str, etag: Optional[str], last_modified: Optional[str]) -> tuple[int, Optional[bytes]]:
        del etag, last_modified
        if url.endswith("a.htm"):
            return 304, None
        if url.endswith("b.htm"):
            return 200, b"payload-b"
        return 200, None

    monkeypatch.setattr(downloader, "_http_download_if_modified", _fake_conditional)

    async def _collect() -> list[DownloaderEvent]:
        events: list[DownloaderEvent] = []
        async for event in downloader.download_files_stream(
            remote_files=descriptors,
            overwrite=False,
            store_file=store_stub,
            existing_files={
                "a.htm": {"http_etag": '"etag-a"', "http_last_modified": "Mon, 01 Jan 2025 00:00:00 GMT"}
            },
        ):
            events.append(event)
        return events

    events = _run(_collect())
    assert [event.event_type for event in events] == ["file_skipped", "file_downloaded", "file_failed"]
    assert store_stub.calls == [("b.htm", b"payload-b")]
    assert events[0].reason_code == "not_modified"
    assert "未修改" in str(events[0].reason_message)
    assert events[2].reason_code == "empty_response"
    assert events[2].reason_message == "下载失败，未返回内容"


def test_download_files_stream_http_error_with_overwrite_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 overwrite=False 时 HTTP 异常（如503）被正确捕获并转换为 file_failed 事件。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = _create_downloader(tmp_path)
    store_stub = StoreStub()
    descriptors = [
        RemoteFileDescriptor(
            name="normal.htm",
            source_url="https://example.com/normal.htm",
            http_etag='"etag-normal"',
            http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
            remote_size=1,
            http_status=200,
        ),
        RemoteFileDescriptor(
            name="failed.xml",
            source_url="https://example.com/failed.xml",
            http_etag='"etag-failed"',
            http_last_modified="Mon, 01 Jan 2025 00:00:01 GMT",
            remote_size=2,
            http_status=200,
        ),
    ]

    async def _fake_conditional(
        url: str,
        etag: Optional[str],
        last_modified: Optional[str],
    ) -> tuple[int, Optional[bytes]]:
        """模拟下载，对 failed.xml 抛出 503 错误。"""

        del etag, last_modified
        if url.endswith("failed.xml"):
            raise RuntimeError(
                "下载失败: url=https://example.com/failed.xml "
                "error=Server error '503 Service Unavailable'"
            )
        return 200, b"payload-normal"

    monkeypatch.setattr(downloader, "_http_download_if_modified", _fake_conditional)

    async def _collect() -> list[DownloaderEvent]:
        events: list[DownloaderEvent] = []
        async for event in downloader.download_files_stream(
            remote_files=descriptors,
            overwrite=False,
            store_file=store_stub,
            existing_files=None,
        ):
            events.append(event)
        return events

    events = _run(_collect())
    # 验证第一个文件下载成功，第二个文件因HTTP错误而失败
    assert [event.event_type for event in events] == ["file_downloaded", "file_failed"]
    assert store_stub.calls == [("normal.htm", b"payload-normal")]
    # 验证错误信息被正确记录
    assert "503 Service Unavailable" in events[1].error
    assert events[1].reason_code == "download_error"
    assert "503 Service Unavailable" in str(events[1].reason_message)


def test_download_files_aggregates_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 download_files 会聚合流式事件为历史结构。"""

    downloader = _create_downloader(tmp_path)
    store_stub = StoreStub()
    descriptor = RemoteFileDescriptor(
        name="sample.htm",
        source_url="https://example.com/sample.htm",
        http_etag='"etag"',
        http_last_modified=None,
        remote_size=None,
        http_status=200,
    )

    monkeypatch.setattr(
        downloader,
        "_http_download_if_modified",
        lambda url, etag, last_modified: (200, b"dummy"),
    )
    results = _run(
        downloader.download_files(
            remote_files=[descriptor],
            overwrite=False,
            store_file=store_stub,
            existing_files={"sample.htm": {"etag": '"old"'}},
        )
    )

    assert results[0]["status"] == "downloaded"
    assert results[0]["file_meta"] is not None


def test_resolve_primary_document_failure_and_fetch_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 resolve_primary_document 失败分支与 fetch_file_bytes 包装。"""

    downloader = _create_downloader(tmp_path)
    monkeypatch.setattr(
        downloader,
        "_http_get_json",
        lambda url: {"directory": {"item": [{"name": ""}, {"type": "10-K"}, {"other": "x"}]}},
    )
    with pytest.raises(RuntimeError, match="无法解析 primary_document"):
        _run(
            downloader.resolve_primary_document(
                cik="320193",
                accession_no_dash="000032019325000001",
                form_type="10-K",
            )
        )

    monkeypatch.setattr(downloader, "_http_download", lambda url: b"payload")
    assert _run(downloader.fetch_file_bytes("https://example.com/file.bin")) == b"payload"


def test_close_owned_client(tmp_path: Path) -> None:
    """验证 close 可正常关闭内部 client。"""

    downloader = _create_downloader(tmp_path)
    _run(downloader.close())


def test_hash_file_sha256_and_pick_instance_fallback(tmp_path: Path) -> None:
    """验证文件哈希与 instance XML 兜底选择分支。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    payload = b"abc123"
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(payload)

    from dayu.fins.downloaders.sec_downloader import hash_file_sha256

    assert hash_file_sha256(file_path) == hashlib.sha256(payload).hexdigest()

    items = [
        {"name": "a_pre.xml"},
        {"name": "a_cal.xml"},
        {"name": "filingsummary.xml"},
        {"name": "report-20250131.xml"},
        {"name": "report.xml"},
    ]
    assert pick_extracted_instance_xml(items) == "report-20250131.xml"


def test_configure_validation_errors(tmp_path: Path) -> None:
    """验证 configure 的参数校验失败分支。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = SecDownloader(workspace_root=tmp_path)
    with pytest.raises(ValueError, match="max_retries 必须大于 0"):
        downloader.configure(user_agent="UA", sleep_seconds=0.0, max_retries=0)
    with pytest.raises(ValueError, match="sleep_seconds 不能为负数"):
        downloader.configure(user_agent="UA", sleep_seconds=-0.1, max_retries=1)
    _run(downloader.close())


def test_fetch_browse_edgar_filenum_non_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 browse-edgar 非空 filenum 分支会走 HTTP 并解析结果。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = _create_downloader(tmp_path)
    captured_urls: list[str] = []
    payload = b"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<feed xmlns=\"http://www.w3.org/2005/Atom\">
  <entry>
    <title>SC 13G - Demo</title>
    <updated>2025-08-10T12:00:00-05:00</updated>
    <link href=\"https://www.sec.gov/Archives/edgar/data/1000/000000000025000777/0000000000-25-000777-index.htm\"/>
  </entry>
</feed>
"""

    async def _fake_get_bytes(url: str) -> bytes:
        captured_urls.append(url)
        return payload

    monkeypatch.setattr(downloader, "_http_get_bytes", _fake_get_bytes)
    filings = _run(downloader.fetch_browse_edgar_filenum("005-79495", count=20))
    assert len(filings) == 1
    assert filings[0].form_type == "SC 13G"
    assert "filenum=005-79495" in captured_urls[0]
    assert "count=20" in captured_urls[0]


def test_download_files_stream_overwrite_with_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 overwrite=True 分支下的下载成功与失败事件。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = _create_downloader(tmp_path)
    descriptors = [
        RemoteFileDescriptor(
            name="ok.htm",
            source_url="https://example.com/ok.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=1,
            http_status=200,
        ),
        RemoteFileDescriptor(
            name="bad.htm",
            source_url="https://example.com/bad.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=2,
            http_status=500,
        ),
    ]
    store_stub = StoreStub()

    async def _fake_download(url: str) -> bytes:
        if url.endswith("bad.htm"):
            raise RuntimeError("network down")
        return b"ok"

    monkeypatch.setattr(downloader, "_http_download", _fake_download)

    async def _collect() -> list[DownloaderEvent]:
        events: list[DownloaderEvent] = []
        async for event in downloader.download_files_stream(
            remote_files=descriptors,
            overwrite=True,
            store_file=store_stub,
            existing_files=None,
        ):
            events.append(event)
        return events

    events = _run(_collect())
    assert [item.event_type for item in events] == ["file_downloaded", "file_failed"]
    assert store_stub.calls == [("ok.htm", b"ok")]
    assert events[1].error == "network down"
    assert events[1].reason_code == "download_error"
    assert events[1].reason_message == "network down"


@pytest.mark.unit
def test_download_files_stream_zero_byte_overwrite_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 overwrite=False 时，服务器返回 0 字节内容不落盘，转为 file_failed 事件。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = _create_downloader(tmp_path)
    store_stub = StoreStub()
    descriptors = [
        RemoteFileDescriptor(
            name="empty.htm",
            source_url="https://example.com/empty.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=0,
            http_status=200,
        ),
        RemoteFileDescriptor(
            name="normal.htm",
            source_url="https://example.com/normal.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=10,
            http_status=200,
        ),
    ]

    async def _fake_conditional(
        url: str,
        etag: Optional[str],
        last_modified: Optional[str],
    ) -> tuple[int, Optional[bytes]]:
        """模拟下载，对 empty.htm 返回 0 字节内容。"""

        del etag, last_modified
        if url.endswith("empty.htm"):
            return 200, b""
        return 200, b"content"

    monkeypatch.setattr(downloader, "_http_download_if_modified", _fake_conditional)

    async def _collect() -> list[DownloaderEvent]:
        events: list[DownloaderEvent] = []
        async for event in downloader.download_files_stream(
            remote_files=descriptors,
            overwrite=False,
            store_file=store_stub,
            existing_files=None,
        ):
            events.append(event)
        return events

    events = _run(_collect())
    assert [e.event_type for e in events] == ["file_failed", "file_downloaded"]
    # 0 字节文件不应落盘
    assert store_stub.calls == [("normal.htm", b"content")]
    assert "0 字节" in events[0].error
    assert events[0].reason_code == "empty_content"
    assert "0 字节" in str(events[0].reason_message)


@pytest.mark.unit
def test_download_files_stream_zero_byte_overwrite_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 overwrite=True 时，服务器返回 0 字节内容不落盘，转为 file_failed 事件。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = _create_downloader(tmp_path)
    store_stub = StoreStub()
    descriptors = [
        RemoteFileDescriptor(
            name="empty.htm",
            source_url="https://example.com/empty.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=0,
            http_status=200,
        ),
        RemoteFileDescriptor(
            name="normal.htm",
            source_url="https://example.com/normal.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=10,
            http_status=200,
        ),
    ]

    async def _fake_download(url: str) -> bytes:
        """模拟下载，对 empty.htm 返回 0 字节内容。"""

        if url.endswith("empty.htm"):
            return b""
        return b"content"

    monkeypatch.setattr(downloader, "_http_download", _fake_download)

    async def _collect() -> list[DownloaderEvent]:
        events: list[DownloaderEvent] = []
        async for event in downloader.download_files_stream(
            remote_files=descriptors,
            overwrite=True,
            store_file=store_stub,
            existing_files=None,
        ):
            events.append(event)
        return events

    events = _run(_collect())
    assert [e.event_type for e in events] == ["file_failed", "file_downloaded"]
    # 0 字节文件不应落盘
    assert store_stub.calls == [("normal.htm", b"content")]
    assert "0 字节" in events[0].error
    assert events[0].reason_code == "empty_content"
    assert "0 字节" in str(events[0].reason_message)


@pytest.mark.unit
def test_download_files_stream_zero_byte_primary_aborts_remaining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 primary_document 0 字节时整个 filing 中止，后续文件不被下载/落盘。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = _create_downloader(tmp_path)
    store_stub = StoreStub()
    # 模拟 FUTU/SNOW 场景：primary .htm 排在前，后跟 XBRL 文件
    descriptors = [
        RemoteFileDescriptor(
            name="futu-20201231x20f.htm",
            source_url="https://example.com/futu-20201231x20f.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=0,
            http_status=200,
        ),
        RemoteFileDescriptor(
            name="futu-20201231x20f.xsd",
            source_url="https://example.com/futu-20201231x20f.xsd",
            http_etag=None,
            http_last_modified=None,
            remote_size=100,
            http_status=200,
        ),
        RemoteFileDescriptor(
            name="futu-20201231x20f_htm.xml",
            source_url="https://example.com/futu-20201231x20f_htm.xml",
            http_etag=None,
            http_last_modified=None,
            remote_size=200,
            http_status=200,
        ),
    ]

    download_calls: list[str] = []

    async def _fake_download(url: str) -> bytes:
        """记录调用，primary 返回 0 字节，其余返回正常内容。"""

        download_calls.append(url)
        if url.endswith(".htm") and "x20f.htm" in url:
            return b""
        return b"xbrl-content"

    monkeypatch.setattr(downloader, "_http_download", _fake_download)

    async def _collect() -> list[DownloaderEvent]:
        events: list[DownloaderEvent] = []
        async for event in downloader.download_files_stream(
            remote_files=descriptors,
            overwrite=True,
            store_file=store_stub,
            existing_files=None,
            primary_document="futu-20201231x20f.htm",
        ):
            events.append(event)
        return events

    events = _run(_collect())
    # 只有一个 file_failed 事件（primary），后续文件全部中止，不产生任何事件
    assert [e.event_type for e in events] == ["file_failed"]
    assert events[0].name == "futu-20201231x20f.htm"
    assert events[0].reason_code == "empty_content"
    # 后续文件不应被下载（_http_download 只被调用一次）
    assert len(download_calls) == 1
    # 没有任何文件落盘
    assert store_stub.calls == []


@pytest.mark.unit
def test_download_files_aggregates_skipped_and_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 download_files 聚合 skipped/failed 事件分支。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = _create_downloader(tmp_path)

    async def _fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[DownloaderEvent]:
        del args, kwargs
        yield DownloaderEvent(
            event_type="file_skipped",
            name="a.htm",
            source_url="https://example.com/a.htm",
            http_etag='"etag-a"',
            http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
            http_status=304,
            reason_code="not_modified",
            reason_message="远端文件未修改，跳过重新下载",
        )
        yield DownloaderEvent(
            event_type="file_failed",
            name="b.htm",
            source_url="https://example.com/b.htm",
            http_etag='"etag-b"',
            http_last_modified=None,
            http_status=500,
            reason_code="download_error",
            reason_message="boom",
            error="boom",
        )

    monkeypatch.setattr(downloader, "download_files_stream", _fake_stream)
    results = _run(
        downloader.download_files(
            remote_files=[],
            overwrite=False,
            store_file=StoreStub(),
            existing_files=None,
        )
    )
    assert [item["status"] for item in results] == ["skipped", "failed"]
    assert results[0]["reason_code"] == "not_modified"
    assert "未修改" in str(results[0]["reason_message"])
    assert results[1]["reason_code"] == "download_error"
    assert results[1]["reason_message"] == "boom"
    assert results[1]["error"] == "boom"


def test_http_download_if_modified_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证条件下载的无条件分支与 304 分支。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = _create_downloader(tmp_path)

    async def _fake_download(url: str) -> bytes:
        return f"payload:{url}".encode("utf-8")

    monkeypatch.setattr(downloader, "_http_download", _fake_download)
    status_code, payload = _run(downloader._http_download_if_modified("https://example.com/a", None, None))
    assert status_code == 200
    assert payload == b"payload:https://example.com/a"

    class _Resp:
        """测试用响应对象。"""

        status_code = 304
        content = b""

        def raise_for_status(self) -> None:
            return None

    class _Client:
        """测试用 HTTP 客户端。"""

        async def get(self, **kwargs: Any) -> _Resp:
            del kwargs
            return _Resp()

    downloader._client = _Client()  # type: ignore[assignment]
    status_code_304, payload_304 = _run(
        downloader._http_download_if_modified("https://example.com/b", '"etag"', "Mon")
    )
    assert status_code_304 == 304
    assert payload_304 is None


def test_http_private_methods_retry_and_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证私有 HTTP 方法的重试与失败路径。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = SecDownloader(workspace_root=tmp_path)
    downloader.configure(user_agent="UA", sleep_seconds=0.0, max_retries=2)

    class _ErrClient:
        """始终抛出连接错误的客户端。"""

        async def get(self, **kwargs: Any) -> Any:
            del kwargs
            raise httpx.ConnectError("boom")

        async def head(self, **kwargs: Any) -> Any:
            del kwargs
            raise httpx.ConnectError("boom")

        async def aclose(self) -> None:
            """关闭客户端。"""

            return None

    downloader._client = _ErrClient()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="GET JSON 失败"):
        _run(downloader._http_get_json("https://example.com/a.json"))
    with pytest.raises(RuntimeError, match="下载失败"):
        _run(downloader._http_download("https://example.com/a.bin"))
    with pytest.raises(RuntimeError, match="GET bytes 失败"):
        _run(downloader._http_get_bytes("https://example.com/a.xml"))
    with pytest.raises(RuntimeError, match="条件下载失败"):
        _run(downloader._http_download_if_modified("https://example.com/a.bin", '"etag"', "Mon"))
    assert _run(downloader._http_head("https://example.com/a.bin", allow_redirects=True)) is None

    async def _fake_sleep(seconds: float) -> None:
        _sleep_calls.append(seconds)

    _sleep_calls: list[float] = []
    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    downloader._sleep_seconds = 0.2
    monkeypatch.setattr(downloader, "_reserve_global_request_slot", lambda min_interval: 0.0)
    # 设置为当前时间，使 rate_limit 认为刚刚发过请求需要等待
    import time
    downloader._last_request_time = time.monotonic()
    _run(downloader._rate_limit())
    _run(downloader._retry_backoff(0))
    _run(downloader._retry_backoff(1))
    # _rate_limit 至少等待 max(0.12, 0.2) = 0.2 秒
    assert _sleep_calls[0] == pytest.approx(0.2, abs=0.02)
    assert _sleep_calls[1] == 0.8
    _run(downloader.close())


def test_try_fetch_index_items_and_helper_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 index 拉取失败分支与若干工具函数边界。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    downloader = _create_downloader(tmp_path)

    async def _raise_runtime(url: str) -> dict[str, Any]:
        del url
        raise RuntimeError("bad index")

    monkeypatch.setattr(downloader, "_http_get_json", _raise_runtime)
    assert _run(downloader._try_fetch_index_items("320193", "000032019325000001")) == []
    monkeypatch.setattr(downloader, "_http_get_bytes", lambda url: b"")
    assert _run(downloader._try_fetch_index_header_documents("320193", "000032019325000001")) == []

    assert downloader._build_headers()["User-Agent"] == "UA"
    assert downloader._build_headers()["Accept-Encoding"] == "gzip, deflate"
    assert optional_int("12") == 12
    assert optional_int("abc") is None
    from dayu.fins.downloaders.sec_downloader import _safe_header

    assert _safe_header(None, "ETag") is None
    assert _safe_header(httpx.Response(200, headers={"ETag": '"etag"'}), "ETag") == '"etag"'


def test_parse_index_header_document_entries_from_escaped_payload() -> None:
    """验证 index-headers 解析支持 SEC 转义文档块。"""

    payload = b"""
<html><body><pre>
&lt;DOCUMENT&gt;
&lt;TYPE&gt;EX-99.1
&lt;SEQUENCE&gt;2
&lt;FILENAME&gt;q12025pressrelease.htm
&lt;DESCRIPTION&gt;EX-99.1
&lt;/DOCUMENT&gt;
&lt;DOCUMENT&gt;
&lt;TYPE&gt;EX-99.2
&lt;FILENAME&gt;q12025interimfinancialrepo.htm
&lt;DESCRIPTION&gt;EX-99.2
&lt;/DOCUMENT&gt;
</pre></body></html>
"""
    entries = _parse_index_header_document_entries(payload)
    assert entries == [
        {
            "name": "q12025pressrelease.htm",
            "type": "EX-99.1",
            "description": "EX-99.1",
        },
        {
            "name": "q12025interimfinancialrepo.htm",
            "type": "EX-99.2",
            "description": "EX-99.2",
        },
    ]


def test_parse_browse_edgar_atom_error() -> None:
    """验证 browse-edgar XML 解析异常。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    with pytest.raises(RuntimeError, match="browse-edgar XML 解析失败"):
        _parse_browse_edgar_atom(b"<feed>")


def test_parse_href_and_primary_selector_fallbacks() -> None:
    """验证 href 解析与主文件选择的回退路径。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    accession, cik = _parse_browse_edgar_href(
        "https://www.sec.gov/Archives/edgar/data/1000/000000000025000777/0000000000-25-000777-index.html"
    )
    assert accession == "0000000000-25-000777"
    assert cik == "1000"

    assert _select_primary_from_index_items(
        items=[{"name": "a.xml"}, {"name": "main.htm"}],
        form_type="10-K",
    ) == "main.htm"
    assert _select_primary_from_index_items(
        items=[{"name": ""}, {"name": "fallback.bin"}],
        form_type="10-K",
    ) == "fallback.bin"


def test_parse_retry_after() -> None:
    """验证 Retry-After 解析逻辑。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 有效的 Retry-After 头
    resp_with_header = httpx.Response(503, headers={"Retry-After": "10"})
    assert _parse_retry_after(resp_with_header) == 10.0

    # Retry-After 小于 1 时取下限 1.0
    resp_small = httpx.Response(429, headers={"Retry-After": "0.3"})
    assert _parse_retry_after(resp_small) == 1.0

    # 无 Retry-After 头，使用默认值
    resp_no_header = httpx.Response(503)
    assert _parse_retry_after(resp_no_header) == 5.0

    # 无效 Retry-After 值，使用默认值
    resp_invalid = httpx.Response(503, headers={"Retry-After": "not-a-number"})
    assert _parse_retry_after(resp_invalid) == 5.0


def test_resolve_sec_throttle_delay_uses_recovery_floor() -> None:
    """验证 SEC 限流等待时间至少为 10 分钟。"""

    resp_small = httpx.Response(503, headers={"Retry-After": "3"})
    assert _resolve_sec_throttle_delay(resp_small) == 600.0

    resp_large = httpx.Response(429, headers={"Retry-After": "1200"})
    assert _resolve_sec_throttle_delay(resp_large) == 1200.0


def test_throttle_retry_on_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 503 限流时额外重试不消耗正常重试预算。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    call_count = 0
    _dummy_request = httpx.Request("GET", "https://example.com/api.json")

    async def _mock_get(**kwargs: Any) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        # 前 2 次返回 503，第 3 次成功
        if call_count <= 2:
            return httpx.Response(503, headers={"Retry-After": "0.01"}, request=_dummy_request)
        return httpx.Response(200, json={"ok": True}, request=_dummy_request)

    # 跳过实际等待
    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    downloader = SecDownloader(workspace_root=tmp_path)
    downloader.configure(user_agent="UA", sleep_seconds=0.0, max_retries=1)
    downloader._client.get = _mock_get  # type: ignore[assignment]

    result = _run(downloader._http_get_json("https://example.com/api.json"))
    assert result == {"ok": True}
    # 总共调用 3 次（2 次 503 + 1 次成功），max_retries=1 未被消耗
    assert call_count == 3
    # 确保 503 触发了 10 分钟恢复窗口
    assert any(s >= 600.0 for s in sleep_calls)
    state = _load_sec_throttle_state(downloader._throttle_state_path)
    assert state.cooldown_until > 0


def test_conditional_download_reuses_shared_throttle_retry_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证条件下载同样复用 SEC 限流额外重试策略。"""

    call_count = 0
    request = httpx.Request("GET", "https://example.com/archive.htm")

    async def _mock_get(**kwargs: Any) -> httpx.Response:
        nonlocal call_count
        del kwargs
        call_count += 1
        if call_count <= 2:
            return httpx.Response(503, headers={"Retry-After": "0.01"}, request=request)
        return httpx.Response(200, content=b"payload", request=request)

    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    downloader = SecDownloader(workspace_root=tmp_path)
    downloader.configure(user_agent="UA", sleep_seconds=0.0, max_retries=1)
    downloader._client.get = _mock_get  # type: ignore[assignment]

    status_code, payload = _run(
        downloader._http_download_if_modified(
            "https://example.com/archive.htm",
            '"etag"',
            "Mon, 01 Jan 2025 00:00:00 GMT",
        )
    )

    assert status_code == 200
    assert payload == b"payload"
    assert call_count == 3
    assert any(seconds >= 600.0 for seconds in sleep_calls)


def test_rate_limit_uses_shared_state_across_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证多个下载器实例共享同一限流状态。"""

    sleep_calls: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    wall_time_values = repeat(1000.0)
    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr("dayu.fins.downloaders.sec_downloader.time.time", lambda: next(wall_time_values))

    downloader_a = _create_downloader(tmp_path)
    downloader_b = _create_downloader(tmp_path)
    _run(downloader_a._rate_limit())
    _run(downloader_b._rate_limit())

    assert any(abs(value - 0.12) < 1e-9 for value in sleep_calls)
    state = _load_sec_throttle_state(downloader_a._throttle_state_path)
    assert state.next_request_at == pytest.approx(1000.24)
    assert state.cooldown_until == pytest.approx(0.0)
