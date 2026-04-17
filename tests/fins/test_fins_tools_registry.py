"""fins 财报工具注册测试。"""

from __future__ import annotations

from typing import Any, Callable, Optional

import pytest

from dayu.engine.processors.source import Source
from dayu.engine.tool_registry import ToolRegistry
from dayu.fins.domain.document_models import CompanyMeta, SourceHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.tools import (
    FinsToolLimits,
    register_fins_ingestion_tools,
)
from tests.fins.legacy_repository_adapters import (
    register_fins_read_tools_with_legacy_repository as register_fins_read_tools,
)


class DummySource:
    """测试用 Source。"""

    def __init__(self, uri: str, media_type: Optional[str] = "text/html") -> None:
        """初始化 Source。"""

        self.uri = uri
        self.media_type = media_type
        self.content_length = None
        self.etag = None

    def open(self) -> Any:
        """测试桩。"""

        raise OSError("not implemented")

    def materialize(self, suffix: Optional[str] = None) -> Any:
        """测试桩。"""

        del suffix
        raise OSError("not implemented")


class FakeRepository:
    """仓储桩。"""

    def list_document_ids(self, ticker: str, source_kind: Optional[SourceKind] = None) -> list[str]:
        """返回文档 ID。"""

        del ticker
        if source_kind == SourceKind.FILING:
            return ["fil_1"]
        if source_kind == SourceKind.MATERIAL:
            return []
        return ["fil_1"]

    def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
        """按候选顺序解析已存在 ticker。"""

        if "AAPL" in candidates:
            return "AAPL"
        return None

    def get_document_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
        """返回文档元数据。"""

        del ticker
        if document_id != "fil_1":
            raise FileNotFoundError(document_id)
        return {
            "document_id": "fil_1",
            "form_type": "10-K",
            "fiscal_year": 2024,
            "fiscal_period": "FY",
            "report_date": "2024-09-28",
            "filing_date": "2024-11-01",
            "is_deleted": False,
            "ingest_complete": True,
        }

    def get_source_handle(self, ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
        """返回 source handle。"""

        del ticker
        if source_kind == SourceKind.FILING and document_id == "fil_1":
            return SourceHandle(ticker="AAPL", document_id="fil_1", source_kind="filing")
        raise FileNotFoundError(document_id)

    def get_primary_source(self, ticker: str, document_id: str, source_kind: SourceKind) -> DummySource:
        """返回 source。"""

        del ticker
        return DummySource(uri=f"local://{source_kind.value}/{document_id}.html", media_type="text/html")

    def get_company_meta(self, ticker: str) -> CompanyMeta:
        """返回公司信息。"""

        return CompanyMeta(
            company_id="320193",
            company_name="Apple Inc.",
            ticker=ticker,
            market="US",
            resolver_version="test",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    def get_processed_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
        """返回 processed meta。"""

        del ticker
        if document_id.startswith("fil_"):
            return {"has_financial_data": True}
        return {"has_financial_data": False}


class FakeProcessor:
    """处理器桩。"""

    def list_sections(self) -> list[dict[str, Any]]:
        """章节列表。"""

        return [{"ref": "s_0001", "title": "章节", "level": 1, "parent_ref": None, "preview": "x"}]

    def read_section(self, ref: str) -> dict[str, Any]:
        """读取章节。"""

        return {
            "ref": ref,
            "title": "章节",
            "content": "X" * 120,
            "tables": [],
            "word_count": 1,
            "contains_full_text": False,
        }

    def search(self, query: str, within_ref: Optional[str] = None) -> list[dict[str, Any]]:
        """搜索。"""

        del within_ref
        return [{"section_ref": "s_0001", "section_title": "章节", "snippet": query}]

    def list_tables(self) -> list[dict[str, Any]]:
        """列表表格。"""

        return []

    def read_table(self, table_ref: str) -> dict[str, Any]:
        """读取表格。"""

        return {
            "table_ref": table_ref,
            "data_format": "markdown",
            "data": "|A|",
            "columns": None,
            "row_count": 1,
            "col_count": 1,
            "is_financial": False,
        }


class FakeProcessorRegistry:
    """处理器注册表桩。"""

    def create(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> FakeProcessor:
        """返回处理器。"""

        del source, form_type, media_type
        return FakeProcessor()

    def create_with_fallback(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
        on_fallback: Optional[Callable[[type[object], Exception, int, int], None]] = None,
    ) -> FakeProcessor:
        """兼容统一回退接口并复用 create。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。
            on_fallback: 回退回调（本桩不触发）。

        Returns:
            处理器实例。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        del on_fallback
        return self.create(source, form_type=form_type, media_type=media_type)


@pytest.mark.unit
def test_register_fins_read_tools_registers_all_read_tool_names() -> None:
    """验证读取工具注册入口会注册全部财报读取工具。"""

    registry = ToolRegistry()
    register_fins_read_tools(
        registry,
        repository=FakeRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    names = set(registry.list_tools())
    assert "fetch_more" in names
    assert {
        "list_documents",
        "get_document_sections",
        "read_section",
        "search_document",
        "list_tables",
        "get_table",
        "get_page_content",
        "get_financial_statement",
        "query_xbrl_facts",
    }.issubset(names)
    assert "start_financial_filing_download_job" not in names


@pytest.mark.unit
def test_search_document_schema_declares_queries_max_items_20() -> None:
    """验证 search_document schema 对 queries 声明 maxItems=20。"""

    registry = ToolRegistry()
    register_fins_read_tools(
        registry,
        repository=FakeRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    schemas = registry.get_schemas()
    search_schema = next(
        schema for schema in schemas if schema["function"]["name"] == "search_document"
    )
    queries_schema = search_schema["function"]["parameters"]["properties"]["queries"]
    assert queries_schema["maxItems"] == 20


@pytest.mark.unit
def test_list_documents_schema_document_types_includes_quarterly_report() -> None:
    """验证 list_documents 的 document_types 枚举包含 quarterly_report。"""

    registry = ToolRegistry()
    register_fins_read_tools(
        registry,
        repository=FakeRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    schemas = registry.get_schemas()
    list_schema = next(
        schema for schema in schemas if schema["function"]["name"] == "list_documents"
    )
    document_types_schema = list_schema["function"]["parameters"]["properties"]["document_types"]
    assert document_types_schema["type"] == "array"
    assert "quarterly_report" in document_types_schema["items"]["enum"]
    assert "annual_report" in document_types_schema["items"]["enum"]


@pytest.mark.unit
def test_read_section_rejects_within_section_ref_as_unknown_param() -> None:
    """验证 read_section 不再接受 within_section_ref 参数（已移除）。"""

    registry = ToolRegistry()
    register_fins_read_tools(
        registry,
        repository=FakeRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    result = registry.execute(
        "read_section",
        {
            "ticker": "AAPL",
            "document_id": "fil_1",
            "ref": "s_0001",
            "within_section_ref": "s_0002",
        },
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_argument"


@pytest.mark.unit
def test_list_documents_contract_removed_old_processed_fields() -> None:
    """验证 list_documents 不再返回 processed 汇总字段。"""

    registry = ToolRegistry()
    register_fins_read_tools(
        registry,
        repository=FakeRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    result = registry.execute("list_documents", {"ticker": "AAPL"})
    assert result["ok"] is True
    payload = result["value"]
    doc = payload["documents"][0]

    assert "section_count" not in doc
    assert "table_count" not in doc
    assert "quality" not in doc
    assert "has_financials" not in doc
    # 新契约字段：document_type 和 has_financial_data
    assert "document_type" in doc
    assert "has_financial_data" in doc
    assert "has_financial_statements" not in doc
    assert "has_xbrl" not in doc
    assert "financial_statement_availability" not in doc


@pytest.mark.unit
def test_list_documents_returns_not_found_when_ticker_is_not_ingested() -> None:
    """验证工具层会将未收录 ticker 映射为标准 not_found 错误。"""

    class MissingTickerRepository(FakeRepository):
        """始终模拟 ticker 未收录。"""

        def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
            """返回未命中。"""

            del candidates
            return None

    registry = ToolRegistry()
    register_fins_read_tools(
        registry,
        repository=MissingTickerRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    result = registry.execute("list_documents", {"ticker": "09992.HK"})

    assert result["ok"] is False
    assert result["error"] == "not_found"
    assert "Financial Document Tools do not have this company" in result["message"]
    assert "不允许：继续穷举 ticker 变体" in result["hint"]


@pytest.mark.unit
def test_register_fins_ingestion_tools_registers_all_job_tool_names() -> None:
    """验证 ingestion 注册入口会注册全部长事务工具。"""

    registry = ToolRegistry()
    register_fins_ingestion_tools(
        registry,
        service_factory=lambda _ticker: None,
        manager_key="test-key",
    )

    names = set(registry.list_tools())
    assert "fetch_more" in names
    assert {
        "start_financial_filing_download_job",
        "get_financial_filing_download_job_status",
        "cancel_financial_filing_download_job",
    }.issubset(names)
    assert "start_financial_document_preprocess_job" not in names
    assert "get_financial_document_preprocess_job_status" not in names
    assert "cancel_financial_document_preprocess_job" not in names


@pytest.mark.unit
def test_ingestion_tool_schema_descriptions_follow_workflow() -> None:
    """验证长事务工具 schema 文案按工作流解释参数与下一步动作。"""

    registry = ToolRegistry()
    register_fins_ingestion_tools(
        registry,
        service_factory=lambda _ticker: None,
        manager_key="test-key",
    )

    schemas = registry.get_schemas()
    start_download = next(
        item for item in schemas if item.get("function", {}).get("name") == "start_financial_filing_download_job"
    )
    status_download = next(
        item for item in schemas if item.get("function", {}).get("name") == "get_financial_filing_download_job_status"
    )
    cancel_download = next(
        item for item in schemas if item.get("function", {}).get("name") == "cancel_financial_filing_download_job"
    )
    fetch_web = None

    ticker_desc = start_download["function"]["parameters"]["properties"]["ticker"]["description"]
    form_types_desc = start_download["function"]["parameters"]["properties"]["form_types"]["description"]
    job_id_desc = status_download["function"]["parameters"]["properties"]["job_id"]["description"]
    start_desc = start_download["function"]["description"]
    status_desc = status_download["function"]["description"]
    cancel_desc = cancel_download["function"]["description"]

    assert "最自然的写法" in ticker_desc
    assert "只在你明确要缩小下载范围时填写" in form_types_desc
    assert "直接使用启动工具返回的 job.job_id" in job_id_desc
    assert "下一步只用状态工具轮询" in start_desc
    assert "优先按 next_step.action 决定" in status_desc
    assert "取消不是立即完成的" in cancel_desc
    assert "Ticker symbol" not in ticker_desc


@pytest.mark.unit
def test_read_section_uses_registry_truncation() -> None:
    """验证 read_section 由 ToolRegistry 自动截断。"""

    registry = ToolRegistry()
    register_fins_read_tools(
        registry,
        repository=FakeRepository(),
        processor_registry=FakeProcessorRegistry(),
        limits=FinsToolLimits(read_section_max_chars=40),
    )

    result = registry.execute(
        "read_section",
        {"ticker": "AAPL", "document_id": "fil_1", "ref": "s_0001"},
    )

    assert result["ok"] is True
    assert result["truncation"] is not None
    assert result["truncation"]["has_more"] is True
    assert result["truncation"]["fetch_more_args"]["cursor"] == result["truncation"]["cursor"]
    assert result["truncation"]["fetch_more_args"]["scope_token"]
    assert result["truncation"]["continuation_required"] is True
    assert result["truncation"]["next_action"] == "fetch_more"


@pytest.mark.unit
def test_query_xbrl_facts_schema_makes_concepts_optional() -> None:
    """验证 `query_xbrl_facts` 的 concepts 参数为可选。"""

    registry = ToolRegistry()
    register_fins_read_tools(
        registry,
        repository=FakeRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    schemas = registry.get_schemas()
    query_schema = next(item for item in schemas if item.get("function", {}).get("name") == "query_xbrl_facts")
    schema = query_schema["function"]["parameters"]
    required = schema.get("required", [])
    assert "ticker" in required
    assert "document_id" in required
    assert "concepts" not in required


@pytest.mark.unit
def test_fins_tool_schema_descriptions_follow_workflow() -> None:
    """验证财报工具 schema 文案与当前实现保持一致。"""

    registry = ToolRegistry()
    register_fins_read_tools(
        registry,
        repository=FakeRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    schemas = registry.get_schemas()
    list_schema = next(item for item in schemas if item.get("function", {}).get("name") == "list_documents")
    sections_schema = next(item for item in schemas if item.get("function", {}).get("name") == "get_document_sections")
    read_schema = next(item for item in schemas if item.get("function", {}).get("name") == "read_section")

    list_ticker_desc = list_schema["function"]["parameters"]["properties"]["ticker"]["description"]
    read_ref_desc = read_schema["function"]["parameters"]["properties"]["ref"]["description"]
    sections_document_id_schema = sections_schema["function"]["parameters"]["properties"]["document_id"]
    sections_desc = sections_schema["function"]["description"]

    assert "最自然的写法" in list_ticker_desc
    assert "company.ticker" not in list_ticker_desc
    assert "description" not in sections_document_id_schema
    assert "章节 ref 列表" in sections_desc
    assert "sections[].ref" in read_ref_desc
    assert "search_document" in read_ref_desc
    assert "Ticker。" not in list_ticker_desc
