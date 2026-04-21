"""Fins 中型辅助模块额外覆盖测试。"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import pytest
from bs4 import BeautifulSoup, Comment

from dayu.fins.downloaders.sec_downloader import BrowseEdgarFiling, RemoteFileDescriptor
from dayu.fins.pipelines import sec_download_state as download_state
from dayu.fins.pipelines import sec_rebuild_workflow as rebuild_workflow
from dayu.fins.processors import sec_dom_helpers
from dayu.fins.resolver import fmp_company_alias_resolver as alias_module


@pytest.mark.unit
def test_fmp_alias_helper_functions_cover_normalization_and_selection() -> None:
    """FMP alias helper 应覆盖规范化、过滤、回退选择和去重。"""

    results = [
        {"symbol": "baba", "name": " Alibaba   Group "},
        {"symbol": "9988.HK", "name": "Alibaba Group"},
        {"symbol": "BABA.BO", "name": "Baba Arts"},
    ]

    assert alias_module._normalize_company_name(" Alibaba　Group ") == "ALIBABA GROUP"
    assert alias_module._normalize_ticker_token(" ba ba ") == "BABA"
    assert alias_module._filter_same_name_results(
        results=results,
        normalized_company_name="ALIBABA GROUP",
    ) == [results[0], results[1]]
    assert alias_module._select_symbol_result(results=results, canonical_ticker="BABA") == results[0]
    assert alias_module._dedupe_ticker_aliases(
        canonical_ticker="BABA",
        raw_aliases=["9988.hk", " BABA ", "9988.HK", ""],
    ) == ["BABA", "9988"]

    with pytest.raises(alias_module.FmpAliasInferenceError):
        alias_module._select_symbol_result(results=[], canonical_ticker="BABA")
    with pytest.raises(alias_module.FmpAliasInferenceError):
        alias_module._extract_required_company_name({})


@pytest.mark.unit
def test_fmp_fetch_results_rejects_invalid_json_and_non_array(monkeypatch: pytest.MonkeyPatch) -> None:
    """FMP HTTP 读取应拒绝非 JSON 和非数组响应。"""

    class _Response:
        def __init__(self, body: str) -> None:
            self._body = body.encode("utf-8")

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

        def read(self) -> bytes:
            return self._body

    monkeypatch.setattr(alias_module.urllib.request, "urlopen", lambda *args, **kwargs: _Response("not-json"))
    with pytest.raises(alias_module.FmpAliasInferenceError, match="非 JSON"):
        alias_module._fetch_fmp_search_results(endpoint="search-name", query="demo", limit=1, api_key="key")

    monkeypatch.setattr(alias_module.urllib.request, "urlopen", lambda *args, **kwargs: _Response(json.dumps({"bad": 1})))
    with pytest.raises(alias_module.FmpAliasInferenceError, match="期望数组"):
        alias_module._fetch_fmp_search_results(endpoint="search-name", query="demo", limit=1, api_key="key")


@pytest.mark.unit
def test_sec_dom_helpers_extract_context_and_node_guards() -> None:
    """SEC DOM helper 应覆盖表格前文、噪声节点、隐藏节点与表格判定。"""

    html = """
    <html><body>
      <h2>Overview</h2>
      <p>Context before table</p>
      <table><tr><td>Value</td></tr></table>
      <div style="display: none">Hidden</div>
    </body></html>
    """
    contexts = sec_dom_helpers._extract_dom_table_contexts(html, max_chars=80)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    assert table is not None
    hidden = soup.find("div")
    assert hidden is not None

    assert contexts == ["Overview Context before table"]
    assert sec_dom_helpers._extract_dom_context_before(table, max_chars=80) == "Overview Context before table"
    assert sec_dom_helpers._is_hidden_tag(hidden) is True
    assert sec_dom_helpers._is_noise_context_node(Comment("ignore")) is True
    assert sec_dom_helpers._is_within_table(table, table_tag_ids=frozenset({id(table)})) is True
    assert sec_dom_helpers._extract_text_from_raw_html("") == ""


@pytest.mark.unit
def test_sec_rebuild_workflow_failure_paths() -> None:
    """重建工作流应覆盖日期非法与单文档失败分支。"""

    with pytest.raises(ValueError, match="start_date"):
        rebuild_workflow.build_rebuild_filter_spec(
            form_type=None,
            start_date="2025-01-02",
            end_date="2025-01-01",
            expand_form_aliases=lambda forms: forms,
            split_form_input=lambda raw: [raw],
            parse_date=lambda raw, end: dt.date.fromisoformat(raw),
        )

    assert rebuild_workflow.passes_rebuild_filters(
        meta={"filing_date": "2024-01-01"},
        target_forms={"10-K"},
        start_bound=None,
        end_bound=None,
        normalize_form=lambda raw: raw,
        parse_date=lambda raw, end: dt.date.fromisoformat(raw),
    ) is False
    assert rebuild_workflow.passes_rebuild_filters(
        meta={"form_type": "10-K", "filing_date": "bad-date"},
        target_forms={"10-K"},
        start_bound=dt.date(2024, 1, 1),
        end_bound=dt.date(2024, 12, 31),
        normalize_form=lambda raw: raw,
        parse_date=lambda raw, end: (_ for _ in ()).throw(ValueError("bad")),
    ) is False

    failure_missing_form = rebuild_workflow.rebuild_single_local_filing(
        source_repository=cast(Any, object()),
        ticker="AAPL",
        document_id="fil_1",
        previous_meta={},
        company_meta=None,
        pipeline_download_version="v1",
        overwrite_rebuilt_meta=lambda *args, **kwargs: None,
    )
    failure_missing_files = rebuild_workflow.rebuild_single_local_filing(
        source_repository=cast(Any, object()),
        ticker="AAPL",
        document_id="fil_2",
        previous_meta={"form_type": "10-K", "filing_date": "2024-01-01"},
        company_meta=None,
        pipeline_download_version="v1",
        overwrite_rebuilt_meta=lambda *args, **kwargs: None,
    )

    assert failure_missing_form["reason_code"] == "missing_form_type"
    assert failure_missing_files["reason_code"] == "missing_files"


@pytest.mark.unit
def test_sec_download_state_helpers_cover_cache_roundtrip_and_equivalence() -> None:
    """SEC 下载状态 helper 应覆盖缓存、序列化和等价性判断。"""

    filing = BrowseEdgarFiling(
        form_type="10-K",
        filing_date="2024-01-01",
        accession_number="0001",
        cik="1234",
        index_url="https://example.com/index",
    )
    dictionaries = download_state._browse_edgar_filings_to_dicts([filing])
    restored = download_state._dicts_to_browse_edgar_filings(dictionaries)

    assert restored == [filing]
    assert download_state._index_file_entries(None) == {}
    assert download_state._index_file_entries({"files": "bad"}) == {}

    meta = {
        "files": [
            {
                "uri": "https://example.com/sample.htm",
                "etag": "etag-1",
                "size": 10,
                "last_modified": "Mon, 01 Jan 2024 00:00:00 GMT",
                "source_url": "https://example.com/sample.htm",
            }
        ]
    }
    normalized = download_state._normalize_rebuild_file_entries(meta)
    fingerprint = download_state._resolve_rebuild_source_fingerprint(previous_meta={}, file_entries=normalized)
    descriptor = RemoteFileDescriptor(
        name="sample.htm",
        source_url="https://example.com/sample.htm",
        http_etag="etag-1",
        http_last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
        remote_size=10,
    )

    assert normalized[0]["name"] == "sample.htm"
    assert fingerprint
    assert download_state._has_same_file_name_set(
        remote_files=[descriptor],
        existing_files={"sample.htm": normalized[0]},
    ) is True
    assert download_state._remote_files_equivalent_to_previous_meta(
        previous_meta={"files": normalized},
        remote_files=[descriptor],
    ) is True

    with TemporaryDirectory() as tmp_dir:
        workspace_root = Path(tmp_dir)
        download_state._write_sec_cache(workspace_root, "demo", "key", {"value": 1})
        assert download_state._read_sec_cache(workspace_root, "demo", "key") == {"value": 1}
