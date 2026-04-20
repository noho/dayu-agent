"""tool_snapshot_export 辅助函数覆盖测试。"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Optional, cast

import pytest

from dayu.fins.domain.document_models import CompanyMeta, CompanyMetaInventoryEntry, FileObjectMeta, ProcessedHandle, SourceHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.storage import CompanyMetaRepositoryProtocol, DocumentBlobRepositoryProtocol
from dayu.fins.pipelines import tool_snapshot_export as module


class _RepositoryStub:
    """仓储桩。"""

    def __init__(self) -> None:
        self.stored: dict[str, bytes] = {}

    def scan_company_meta_inventory(self) -> list[CompanyMetaInventoryEntry]:
        """返回空的公司盘点结果。"""

        return []

    def get_company_meta(self, ticker: str) -> CompanyMeta:
        """返回固定公司元数据。"""

        return CompanyMeta(
            company_id=ticker,
            company_name="Apple",
            ticker=ticker,
            market="US",
            resolver_version="test",
            updated_at="2024-01-01T00:00:00+00:00",
        )

    def upsert_company_meta(self, meta: CompanyMeta) -> None:
        """测试桩不实际写入公司元数据。"""

        del meta

    def resolve_existing_ticker(self, ticker_candidates: list[str]) -> str | None:
        """返回首个候选 ticker。"""

        return ticker_candidates[0] if ticker_candidates else None

    def list_entries(self, handle: SourceHandle | ProcessedHandle) -> list[object]:
        """测试桩不需要目录条目。"""

        del handle
        return []

    def read_file_bytes(self, handle: SourceHandle | ProcessedHandle, name: str) -> bytes:
        """按写入键读取文件内容。"""

        return self.stored[f"{handle.ticker}:{handle.document_id}:{name}"]

    def delete_entry(self, handle: SourceHandle | ProcessedHandle, name: str) -> None:
        """删除已写入文件。"""

        self.stored.pop(f"{handle.ticker}:{handle.document_id}:{name}", None)

    def store_file(
        self,
        handle: SourceHandle | ProcessedHandle,
        filename: str,
        data: BinaryIO,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> FileObjectMeta:
        del content_type, metadata
        self.stored[f"{handle.ticker}:{handle.document_id}:{filename}"] = data.read()
        return FileObjectMeta(uri=f"memory://{handle.ticker}/{handle.document_id}/{filename}")

    def list_files(self, handle: SourceHandle | ProcessedHandle) -> list[FileObjectMeta]:
        """返回当前目录下的文件对象元数据。"""

        prefix = f"{handle.ticker}:{handle.document_id}:"
        result: list[FileObjectMeta] = []
        for key in sorted(self.stored):
            if not key.startswith(prefix):
                continue
            filename = key.removeprefix(prefix)
            result.append(FileObjectMeta(uri=f"memory://{handle.ticker}/{handle.document_id}/{filename}"))
        return result


class _ServiceStub:
    """工具服务桩。"""

    def query_xbrl_facts(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "request": kwargs}


@pytest.mark.unit
def test_snapshot_file_name_and_query_builders() -> None:
    """覆盖快照文件名判断与查询词构建分支。"""

    assert module.is_snapshot_file_name(" ") is False
    assert module.is_snapshot_file_name("tool_snapshot_a.json") is True

    assert module._build_search_queries("US", form_type="SC 13G") == list(module.US_OWNERSHIP_PACK_QUERIES)
    assert module._build_search_queries("US", form_type="10-K") == list(module.US_ANNUAL_QUARTER_CORE40_QUERIES)
    assert module._build_search_queries("HK", form_type="10-K") == list(module.HK_ANNUAL_QUARTER_CORE40_QUERIES)
    assert module._build_search_queries("CN", form_type="10-K") == list(module.CN_ANNUAL_QUARTER_CORE40_QUERIES)
    assert module._build_search_queries("US", form_type="8-K") == list(module.US_EVENT_PACK_QUERIES)
    assert module._build_search_queries("US", form_type="DEF 14A") == list(module.US_GOVERNANCE_PACK_QUERIES)
    assert module._build_search_queries("unknown", form_type="10-K") == list(module.CN_ANNUAL_QUARTER_CORE40_QUERIES)

    search_pack = module._build_search_query_pack("US", form_type="10-K")
    assert search_pack["name"] == module.SEARCH_QUERY_PACK_ANNUAL_QUARTER_CORE40
    assert search_pack["version"] == module.TOOL_SNAPSHOT_SEARCH_QUERY_PACK_VERSION
    assert len(search_pack["queries"]) == 40
    search_specs = module._build_search_query_specs(
        pack_name=search_pack["name"],
        queries=list(search_pack["queries"][:2]),
    )
    assert search_specs[0]["query_id"] == "annual_quarter_core40.q001"
    assert search_specs[0]["query_text"] == search_pack["queries"][0]
    assert search_specs[0]["query_intent"]
    assert search_specs[0]["query_weight"] == 1.0

    assert module._normalize_form_type("SCHEDULE 13G") == "SC 13G"
    assert module._normalize_market("  hk ") == "HK"
    assert module._normalize_market("unknown") == "CN"
    assert module._build_xbrl_concepts() == list(module.XBRL_DEFAULT_CONCEPTS)


@pytest.mark.unit
def test_collect_helpers_and_page_candidates() -> None:
    """覆盖章节/表格引用提取与页码候选边界分支。"""

    assert module._collect_section_refs(sections_response={"sections": "bad"}) == []
    assert module._collect_table_refs(list_tables_response={"tables": "bad"}) == []

    section_refs = module._collect_section_refs(
        sections_response={"sections": [None, {"ref": " sec_1 "}, {"ref": ""}]}
    )
    assert section_refs == ["sec_1"]

    table_refs = module._collect_table_refs(
        list_tables_response={"tables": [None, {"table_ref": " tbl_1 "}, {"table_ref": ""}]}
    )
    assert table_refs == ["tbl_1"]

    pages = module._collect_page_candidates(
        sections_response={"sections": [{"page_range": [3, 2]}, {"page_range": [1, 2]}, "x"]},
        list_tables_response={"tables": [{"page_no": 3}, {"page_no": 0}, "x"]},
    )
    assert pages == [1, 2, 3]

    assert module._collect_page_candidates(sections_response={}, list_tables_response={}) == [1]


@pytest.mark.unit
def test_analyze_financial_statement_capability() -> None:
    """覆盖财务能力分析中的映射与 xbrl 判断分支。"""

    (
        has_statement,
        has_xbrl,
        availability,
        has_structured_financial_statements,
        has_financial_statement_sections,
        has_financial_data,
    ) = module._analyze_financial_statement_capability(
        financial_statement_calls=cast(
            list[dict[str, Any]],
            [
                "bad",
                {"response": "bad"},
                {"response": {"rows": [{"a": 1}], "periods": [], "data_quality": "xbrl"}},
            ],
        )
    )
    assert has_statement is True
    assert has_xbrl is True
    assert availability == "structured_data_available"
    assert has_structured_financial_statements is True
    assert has_financial_statement_sections is True
    assert has_financial_data is True


@pytest.mark.unit
def test_validate_handle_and_text_normalizers() -> None:
    """覆盖句柄校验与文本标准化分支。"""

    module._validate_processed_handle(
        processed_handle=ProcessedHandle(ticker="AAPL", document_id="fil_1"),
        ticker="AAPL",
        document_id="fil_1",
    )

    with pytest.raises(ValueError, match="ticker"):
        module._validate_processed_handle(
            processed_handle=ProcessedHandle(ticker="MSFT", document_id="fil_1"),
            ticker="AAPL",
            document_id="fil_1",
        )

    with pytest.raises(ValueError, match="document_id"):
        module._validate_processed_handle(
            processed_handle=ProcessedHandle(ticker="AAPL", document_id="fil_2"),
            ticker="AAPL",
            document_id="fil_1",
        )

    assert module._extract_page_range({"page_range": "bad"}) is None
    assert module._extract_page_range({"page_range": [1, "2"]}) is None
    assert module._extract_page_range({"page_range": [1, 2]}) == [1, 2]

    from dayu.fins._converters import normalize_optional_text, require_non_empty_text
    assert normalize_optional_text("  ") is None
    assert normalize_optional_text(" x ") == "x"
    with pytest.raises(ValueError, match="不能为空"):
        require_non_empty_text("  ", empty_error=ValueError("必填文本不能为空"))


@pytest.mark.unit
def test_query_call_builder_read_company_and_write_snapshot() -> None:
    """覆盖 query_xbrl 请求组装、公司信息回退和快照写入。"""

    service = _ServiceStub()
    request = module._build_query_xbrl_facts_call(
        service=cast(module.FinsToolService, service),
        ticker="AAPL",
        document_id="fil_1",
        source_meta={"fiscal_year": 2024, "fiscal_period": " FY ", "report_date": "2024-09-28"},
        concepts=["Revenue"],
    )
    assert request["request"]["fiscal_year"] == 2024
    assert request["request"]["fiscal_period"] == "FY"
    assert request["request"]["period_end"] == "2024-09-28"

    repo = _RepositoryStub()
    assert module._read_company_info(
        repository=cast(CompanyMetaRepositoryProtocol, repo),
        ticker="AAPL",
    ) == ("Apple", "US")

    class _MissingRepo(_RepositoryStub):
        def get_company_meta(self, ticker: str) -> Any:
            raise FileNotFoundError(ticker)

    assert module._read_company_info(
        repository=cast(CompanyMetaRepositoryProtocol, _MissingRepo()),
        ticker="AAPL",
    ) == ("AAPL", "unknown")

    payload = module._build_tool_snapshot_payload(
        tool="list_documents",
        mode="offline",
        ticker="AAPL",
        document_id="fil_1",
        source_kind=SourceKind.FILING,
        market="US",
        calls=[],
    )
    meta = module._build_tool_snapshot_meta(
        ticker="AAPL",
        document_id="fil_1",
        source_kind=SourceKind.FILING,
        market="US",
        mode="offline",
        parser_signature="sig",
        expected_parser_signature="expected_sig",
        source_document_version="v1",
        source_fingerprint="fp",
        form_type="10-K",
        document_type="annual_report",
        has_financial_statement=True,
        has_xbrl=True,
        financial_statement_availability="structured_data_available",
        has_structured_financial_statements=True,
        has_financial_statement_sections=True,
        has_financial_data=True,
        search_queries=[],
        search_query_pack_name=module.OFFLINE_SEARCH_QUERY_PACK_NAME,
        search_query_pack_version=module.TOOL_SNAPSHOT_SEARCH_QUERY_PACK_VERSION,
        search_query_count=0,
        xbrl_concepts=[],
        statement_types=["income"],
        tools=["list_documents"],
        written_files=["tool_snapshot_list_documents.json"],
    )
    assert payload["tool"] == "list_documents"
    assert meta["parser_signature"] == "sig"
    assert meta["expected_parser_signature"] == "expected_sig"
    assert meta["search_query_pack_name"] == module.OFFLINE_SEARCH_QUERY_PACK_NAME
    assert meta["search_query_count"] == 0

    handle = ProcessedHandle(ticker="AAPL", document_id="fil_1")
    module._write_tool_snapshot_file(
        repository=cast(DocumentBlobRepositoryProtocol, repo),
        processed_handle=handle,
        file_name="tool_snapshot_meta.json",
        payload={"a": 1},
    )
    raw = repo.stored["AAPL:fil_1:tool_snapshot_meta.json"]
    assert json.loads(raw.decode("utf-8"))["a"] == 1
