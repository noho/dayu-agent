"""SecPipeline process_filing Source 测试。"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Optional, cast

import pytest

from dayu.contracts.cancellation import CancelledError
from dayu.engine.processors.base import (
    DocumentProcessor,
    SearchHit,
    SectionContent,
    SectionSummary,
    TableContent,
    TableSummary,
)
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.engine.processors.source import Source
from dayu.fins.domain.document_models import DocumentQuery, FilingCreateRequest
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.sec_pipeline import SecPipeline
from dayu.fins.processors.registry import build_fins_processor_registry
from dayu.fins.score_sec_ci import find_form_dirs
from dayu.fins.storage import FsProcessedDocumentRepository
from dayu.fins.storage._fs_storage_core import FsStorageCore
from tests.fins.storage_testkit import build_storage_core
from dayu.fins.storage.local_file_store import LocalFileStore


def _prepare_filing(
    repository: FsStorageCore,
    ticker: str,
    document_id: str,
    filename: str,
    content: bytes,
    form_type: str = "DEF 14A",
    ingest_complete: bool = True,
    is_deleted: bool = False,
) -> None:
    """准备 filings 文档与本地文件。

    Args:
        repository: 仓储实例。
        ticker: 股票代码。
        document_id: 文档 ID。
        filename: 主文件名。
        content: 文件内容。
        ingest_complete: 是否完成入库。
        is_deleted: 是否标记删除。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    store = LocalFileStore(root=repository.portfolio_root, scheme="local")
    key = f"{ticker}/filings/{document_id}/{filename}"
    file_meta = store.put_object(key, BytesIO(content))
    repository.create_filing(
        FilingCreateRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=document_id,
            form_type=form_type,
            primary_document=filename,
            files=[file_meta],
            meta={
                "ingest_complete": ingest_complete,
                "is_deleted": is_deleted,
                "document_version": "v1",
                "source_fingerprint": "hash_v1",
                "filing_date": "2026-01-01",
                "report_date": "2026-01-01",
                "amended": False,
            },
        )
    )


class FakeSecProcessor:
    """测试用 SEC 处理器。"""

    PARSER_VERSION = "sec_processor_test_v1.0.0"

    @classmethod
    def get_parser_version(cls) -> str:
        """返回测试处理器 parser version。"""

        return str(cls.PARSER_VERSION)

    def __init__(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> None:
        """初始化测试处理器。

        Args:
            source: 文档来源对象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            无。

        Raises:
            RuntimeError: 初始化失败时抛出。
        """

        del source
        del form_type
        del media_type

    @classmethod
    def supports(
        cls,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> bool:
        """判断是否支持。

        Args:
            source: 文档来源对象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            是否支持。

        Raises:
            RuntimeError: 判定失败时抛出。
        """

        del source
        del media_type
        return form_type == "10-K"

    def list_sections(self) -> list[SectionSummary]:
        """返回章节列表。

        Args:
            无。

        Returns:
            章节列表。

        Raises:
            RuntimeError: 生成失败时抛出。
        """

        return [
            {
                "ref": "s_0001",
                "title": "Part I Item 1",
                "level": 1,
                "parent_ref": None,
                "preview": "business overview",
            }
        ]

    def get_section_title(self, ref: str) -> Optional[str]:
        """根据 section ref 获取章节标题。

        Args:
            ref: 章节引用。

        Returns:
            章节标题字符串；ref 不存在时返回 None。
        """
        for sec in self.list_sections():
            if sec.get("ref") == ref:
                return sec.get("title")
        return None

    def list_tables(self) -> list[TableSummary]:
        """返回表格列表。

        Args:
            无。

        Returns:
            表格列表。

        Raises:
            RuntimeError: 生成失败时抛出。
        """

        return [
            {
                "table_ref": "t_0001",
                "caption": "Main table",
                "context_before": "",
                "row_count": 1,
                "col_count": 1,
                "is_financial": False,
                "headers": ["A"],
                "section_ref": "s_0001",
                "table_type": "data",
            }
        ]

    def read_section(self, ref: str) -> SectionContent:
        """读取章节内容。

        Args:
            ref: 章节引用。

        Returns:
            章节内容。

        Raises:
            KeyError: 章节不存在时抛出。
        """

        return {
            "ref": ref,
            "title": "Part I Item 1",
            "content": "business overview",
            "tables": [],
            "word_count": 2,
            "contains_full_text": True,
        }

    def read_table(self, table_ref: str) -> TableContent:
        """读取表格内容。

        Args:
            table_ref: 表格引用。

        Returns:
            表格内容。

        Raises:
            KeyError: 表格不存在时抛出。
        """

        return {
            "table_ref": table_ref,
            "caption": "Main table",
            "data_format": "records",
            "data": [],
            "columns": ["A"],
            "row_count": 0,
            "col_count": 1,
            "is_financial": False,
            "section_ref": "s_0001",
            "table_type": "data",
        }

    def search(
        self,
        query: str,
        within_ref: Optional[str] = None,
    ) -> list[SearchHit]:
        """搜索内容。

        Args:
            query: 搜索词。
            within_ref: 可选范围引用。

        Returns:
            命中结果列表。

        Raises:
            RuntimeError: 搜索失败时抛出。
        """

        del query
        del within_ref
        return []

    def get_full_text(self) -> str:
        """返回测试用全文文本。"""

        return "Part I Item 1 business overview"

    def get_full_text_with_table_markers(self) -> str:
        """返回带表格占位的测试全文文本。"""

        return self.get_full_text()


class BrokenSecProcessor(FakeSecProcessor):
    """测试用异常处理器。"""

    PARSER_VERSION = "sec_processor_broken_v1.0.0"

    def __init__(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> None:
        """初始化即抛异常。

        Args:
            source: 文档来源对象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出。
        """

        del source
        del form_type
        del media_type
        raise RuntimeError("forced_create_error")


class FakeSecProcessorWithXbrl(FakeSecProcessor):
    """具备 XBRL 财务能力的测试处理器。"""

    PARSER_VERSION = "sec_processor_xbrl_v1.0.0"

    def get_financial_statement(
        self,
        statement_type: str,
        financials: Optional[dict[str, object]] = None,
        *,
        meta: Optional[dict[str, object]] = None,
    ) -> dict[str, object]:
        """返回带 XBRL 标记的财务报表结果。

        Args:
            statement_type: 报表类型。
            financials: 预留参数。
            meta: 预留参数。

        Returns:
            财务报表结果字典。

        Raises:
            RuntimeError: 生成失败时抛出。
        """

        del financials
        del meta
        return {
            "statement_type": statement_type,
            "periods": [{"period_end": "2024-09-28", "fiscal_year": 2024, "fiscal_period": "FY"}],
            "rows": [{"concept": "us-gaap:Assets", "values": {"2024-09-28": 1.0}}],
            "currency": "USD",
            "units": ["USD"],
            "data_quality": "xbrl",
        }


class FakeSecProcessorWithXbrl10Q(FakeSecProcessorWithXbrl):
    """支持 10-Q 的 XBRL 测试处理器。"""

    PARSER_VERSION = "sec_processor_xbrl_10q_v1.0.0"

    @classmethod
    def supports(
        cls,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> bool:
        """判断是否支持。

        Args:
            source: 文档来源对象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            是否支持。

        Raises:
            RuntimeError: 判定失败时抛出。
        """

        del source
        del media_type
        return form_type in {"10-K", "10-Q"}


class FakeSecProcessorWithoutXbrl(FakeSecProcessor):
    """无 XBRL 财务能力的测试处理器。"""

    PARSER_VERSION = "sec_processor_partial_v1.0.0"

    def get_financial_statement(
        self,
        statement_type: str,
        financials: Optional[dict[str, object]] = None,
        *,
        meta: Optional[dict[str, object]] = None,
    ) -> dict[str, object]:
        """返回无 XBRL 的财务报表结果。

        Args:
            statement_type: 报表类型。
            financials: 预留参数。
            meta: 预留参数。

        Returns:
            财务报表结果字典。

        Raises:
            RuntimeError: 生成失败时抛出。
        """

        del financials
        del meta
        return {
            "statement_type": statement_type,
            "periods": [],
            "rows": [],
            "currency": None,
            "units": None,
            "data_quality": "partial",
            "reason": "xbrl_not_available",
        }


class FakeSecProcessorWithoutXbrl6K(FakeSecProcessorWithoutXbrl):
    """无 XBRL 且支持 6-K 的测试处理器。"""

    PARSER_VERSION = "sec_processor_partial_6k_v1.0.0"

    @classmethod
    def supports(
        cls,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> bool:
        """判断是否支持 6-K。

        Args:
            source: 文档来源对象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            仅当 form_type 为 `6-K` 时返回 `True`。

        Raises:
            RuntimeError: 判定失败时抛出。
        """

        del source
        del media_type
        return form_type == "6-K"


class FakeSecProcessorSkipFinancial(FakeSecProcessorWithXbrl10Q):
    """用于验证“按 form 跳过财务提取”的测试处理器。"""

    PARSER_VERSION = "sec_processor_skip_financial_v1.0.0"
    statement_call_count = 0
    query_call_count = 0

    @classmethod
    def supports(
        cls,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> bool:
        """判断是否支持被跳过财务提取的 form。

        Args:
            source: 文档来源对象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            是否支持。

        Raises:
            RuntimeError: 判定失败时抛出。
        """

        del source
        del media_type
        return form_type in {"8-K", "DEF 14A", "SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"}

    def get_financial_statement(
        self,
        statement_type: str,
        financials: Optional[dict[str, object]] = None,
        *,
        meta: Optional[dict[str, object]] = None,
    ) -> dict[str, object]:
        """返回财务报表结果并记录调用次数。

        Args:
            statement_type: 报表类型。
            financials: 预留参数。
            meta: 预留参数。

        Returns:
            财务报表结果字典。

        Raises:
            RuntimeError: 生成失败时抛出。
        """

        del financials
        del meta
        type(self).statement_call_count += 1
        return super().get_financial_statement(statement_type=statement_type)

    def query_xbrl_facts(
        self,
        concepts: list[str],
        statement_type: Optional[str] = None,
        period_end: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> dict[str, object]:
        """返回 facts 结果并记录调用次数。

        Args:
            concepts: 概念列表。
            statement_type: 可选报表类型。
            period_end: 可选期末日期。
            fiscal_year: 可选财年。
            fiscal_period: 可选财季。
            min_value: 可选最小值。
            max_value: 可选最大值。

        Returns:
            facts 结果字典。

        Raises:
            RuntimeError: 查询失败时抛出。
        """

        del concepts
        del statement_type
        del period_end
        del fiscal_year
        del fiscal_period
        del min_value
        del max_value
        type(self).query_call_count += 1
        return {
            "query_params": {},
            "facts": [
                {
                    "period_end": "2024-09-28",
                    "fiscal_year": 2024,
                    "fiscal_period": "FY",
                }
            ],
            "total": 1,
        }


class NoVersionProcessor:
    """未声明 PARSER_VERSION 的测试处理器。"""

    def __init__(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> None:
        """初始化测试处理器。

        Args:
            source: 文档来源对象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            无。

        Raises:
            RuntimeError: 初始化失败时抛出。
        """

        del source
        del form_type
        del media_type

    @classmethod
    def supports(
        cls,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> bool:
        """判断是否支持。

        Args:
            source: 文档来源对象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            是否支持。

        Raises:
            RuntimeError: 判定失败时抛出。
        """

        del source
        del media_type
        return form_type == "10-K"

    def list_sections(self) -> list[SectionSummary]:
        """返回章节列表。

        Args:
            无。

        Returns:
            章节列表。

        Raises:
            RuntimeError: 生成失败时抛出。
        """

        return []

    def list_tables(self) -> list[TableSummary]:
        """返回表格列表。

        Args:
            无。

        Returns:
            表格列表。

        Raises:
            RuntimeError: 生成失败时抛出。
        """

        return []


@pytest.mark.unit
def test_sec_pipeline_requires_explicit_processor_registry(tmp_path: Path) -> None:
    """验证 SecPipeline 构造时必须显式注入处理器注册表。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    with pytest.raises(ValueError, match="processor_registry 必须由调用方显式传入"):
        SecPipeline(
            workspace_root=tmp_path,
            processor_registry=cast(ProcessorRegistry, None),
        )


@pytest.mark.unit
def test_process_filing_creates_processed(tmp_path: Path) -> None:
    """验证 process_filing 生成的 snapshot 能被 processed 仓储发现。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_0001",
        filename="sample.html",
        content=b"<html><h1>Section</h1><p>Text</p></html>",
    )

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    result = pipeline.process_filing("AAPL", "fil_0001")
    assert result["status"] == "processed"

    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / "fil_0001"
    assert (processed_dir / "tool_snapshot_get_document_sections.json").exists()
    assert (processed_dir / "tool_snapshot_list_tables.json").exists()
    meta = json.loads((processed_dir / "tool_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"]
    assert meta["parser_signature"]
    processed_repository = FsProcessedDocumentRepository(tmp_path)
    summaries = processed_repository.list_processed_documents(
        "AAPL",
        DocumentQuery(source_kind=SourceKind.FILING.value),
    )
    assert [item.document_id for item in summaries] == ["fil_0001"]
    discovered = find_form_dirs(str(tmp_path), ["AAPL"], "DEF 14A")
    assert [(item.ticker, item.document_id) for item in discovered] == [("AAPL", "fil_0001")]


@pytest.mark.unit
def test_process_filing_honors_cancel_checker_before_export(tmp_path: Path) -> None:
    """验证 SEC 单文档处理会在同步阶段边界响应取消请求。"""

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_cancelled",
        filename="sample.html",
        content=b"<html><h1>Section</h1><p>Text</p></html>",
    )

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )

    with pytest.raises(CancelledError, match="操作已被取消"):
        pipeline.process_filing(
            "AAPL",
            "fil_cancelled",
            cancel_checker=lambda: True,
        )

    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / "fil_cancelled"
    assert not processed_dir.exists()


@pytest.mark.unit
def test_process_filing_skips_incomplete(tmp_path: Path) -> None:
    """验证 ingest_complete=false 时跳过处理。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_0002",
        filename="sample.html",
        content=b"<html><h1>Section</h1><p>Text</p></html>",
        ingest_complete=False,
    )

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    result = pipeline.process_filing("AAPL", "fil_0002")
    assert result["status"] == "skipped"
    assert result["reason"] == "ingest_incomplete"


@pytest.mark.unit
def test_process_filing_uses_processor_parser_version(tmp_path: Path) -> None:
    """验证 process_filing 会写入处理器 parser_version。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_0003",
        filename="sample.html",
        content=b"<html><body>sec processor route</body></html>",
        form_type="10-K",
    )

    registry = ProcessorRegistry()
    registry.register(FakeSecProcessor, name="sec_processor", priority=10)

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=registry,
    )
    result = pipeline.process_filing("AAPL", "fil_0003")
    assert result["status"] == "processed"

    meta_path = tmp_path / "portfolio" / "AAPL" / "processed" / "fil_0003" / "tool_snapshot_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["parser_signature"] == "sec_processor_test_v1.0.0"


@pytest.mark.unit
def test_process_filing_fallback_to_next_candidate_when_primary_fails(tmp_path: Path) -> None:
    """验证主处理器失败时回退到下一候选处理器。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_0004",
        filename="sample.html",
        content=b"<html><h1>Fallback</h1><p>content</p></html>",
        form_type="10-K",
    )

    registry = ProcessorRegistry()
    registry.register(BrokenSecProcessor, name="broken_sec", priority=10)
    registry.register(FakeSecProcessor, name="sec_processor", priority=5)

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=registry,
    )
    result = pipeline.process_filing("AAPL", "fil_0004")
    assert result["status"] == "processed"

    meta_path = tmp_path / "portfolio" / "AAPL" / "processed" / "fil_0004" / "tool_snapshot_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["parser_signature"] == "sec_processor_test_v1.0.0"
    assert meta["expected_parser_signature"] == "sec_processor_broken_v1.0.0"


@pytest.mark.unit
def test_process_filing_raises_when_no_processor_available(tmp_path: Path) -> None:
    """验证未注册可用处理器时会抛错。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_0005",
        filename="sample.html",
        content=b"<html><h1>No Processor</h1></html>",
        form_type="10-K",
    )

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=ProcessorRegistry(),
    )
    with pytest.raises(RuntimeError, match="未找到可用处理器"):
        pipeline.process_filing("AAPL", "fil_0005")


@pytest.mark.unit
def test_process_filing_raises_when_processor_missing_parser_version(tmp_path: Path) -> None:
    """验证处理器未声明 PARSER_VERSION 时会抛错。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_0006",
        filename="sample.html",
        content=b"<html><h1>No Version</h1></html>",
        form_type="10-K",
    )

    registry = ProcessorRegistry()
    registry.register(cast(type[DocumentProcessor], NoVersionProcessor), name="no_version", priority=10)
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=registry,
    )
    with pytest.raises(RuntimeError, match="PARSER_VERSION"):
        pipeline.process_filing("AAPL", "fil_0006")


@pytest.mark.unit
def test_process_filing_marks_full_quality_with_xbrl(tmp_path: Path) -> None:
    """验证有 XBRL 时 quality=full 且 has_xbrl=true。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_0007",
        filename="sample.html",
        content=b"<html><body>XBRL</body></html>",
        form_type="10-K",
    )

    registry = ProcessorRegistry()
    registry.register(FakeSecProcessorWithXbrl10Q, name="sec_processor", priority=10)
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=registry,
    )
    result = pipeline.process_filing("AAPL", "fil_0007")
    assert result["status"] == "processed"

    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / "fil_0007"
    assert (processed_dir / "tool_snapshot_get_financial_statement.json").exists()
    meta = json.loads((processed_dir / "tool_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["has_financial_statement"] is True
    assert meta["has_xbrl"] is True


@pytest.mark.unit
def test_process_filing_marks_partial_quality_without_xbrl(tmp_path: Path) -> None:
    """验证无 XBRL 但有财务能力时 quality=partial 且 has_xbrl=false。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_0008",
        filename="sample.html",
        content=b"<html><body>No XBRL</body></html>",
        form_type="10-K",
    )

    registry = ProcessorRegistry()
    registry.register(FakeSecProcessorWithoutXbrl, name="sec_processor", priority=10)
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=registry,
    )
    result = pipeline.process_filing("AAPL", "fil_0008")
    assert result["status"] == "processed"

    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / "fil_0008"
    assert (processed_dir / "tool_snapshot_get_financial_statement.json").exists()
    meta = json.loads((processed_dir / "tool_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["has_financial_statement"] is False
    assert meta["has_xbrl"] is False


@pytest.mark.unit
def test_process_filing_handles_6k_without_xbrl_fiscal_signals(tmp_path: Path) -> None:
    """验证 6-K 在缺少 XBRL fiscal 信号时仍能正常完成处理。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="TCOM",
        document_id="fil_6k_0001",
        filename="sample.html",
        content=b"<html><body>6K without xbrl fiscal period</body></html>",
        form_type="6-K",
    )

    registry = ProcessorRegistry()
    registry.register(FakeSecProcessorWithoutXbrl6K, name="sec_processor", priority=10)
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=registry,
    )
    result = pipeline.process_filing("TCOM", "fil_6k_0001")
    assert result["status"] == "processed"

    processed_dir = tmp_path / "portfolio" / "TCOM" / "processed" / "fil_6k_0001"
    meta = json.loads((processed_dir / "tool_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["has_financial_statement"] is False
    assert meta["has_xbrl"] is False


@pytest.mark.unit
def test_process_filing_skips_financial_extraction_for_8k_sc13_and_marks_partial(tmp_path: Path) -> None:
    """验证 8-K/SC 13* 默认跳过财务提取，但质量标记为 partial。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    FakeSecProcessorSkipFinancial.statement_call_count = 0
    FakeSecProcessorSkipFinancial.query_call_count = 0
    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_0010",
        filename="sample.html",
        content=b"<html><body>8K</body></html>",
        form_type="8-K",
    )

    registry = ProcessorRegistry()
    registry.register(FakeSecProcessorSkipFinancial, name="sec_processor", priority=10)
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=registry,
    )
    result = pipeline.process_filing("AAPL", "fil_0010")
    assert result["status"] == "processed"
    # 快照导出始终为每种报表类型调用 get_financial_statement（共 5 种）
    assert FakeSecProcessorSkipFinancial.statement_call_count == 5
    # query_xbrl_facts 仅在 ci=True 时导出
    assert FakeSecProcessorSkipFinancial.query_call_count == 0

    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / "fil_0010"
    assert (processed_dir / "tool_snapshot_get_financial_statement.json").exists()
    meta = json.loads((processed_dir / "tool_snapshot_meta.json").read_text(encoding="utf-8"))
    # FakeSecProcessorSkipFinancial 继承自 WithXbrl，返回 XBRL 数据
    assert meta["has_financial_statement"] is True
    assert meta["has_xbrl"] is True


@pytest.mark.unit
def test_process_filing_def14a_routes_to_special_processor_and_marks_partial(tmp_path: Path) -> None:
    """验证 DEF 14A 走专项处理器且质量标记为 partial。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_def14a",
        filename="sample.html",
        content=b"<html><body><h1>DEF14A</h1><p>proxy text</p></body></html>",
        form_type="DEF 14A",
    )

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    result = pipeline.process_filing("AAPL", "fil_def14a")
    assert result["status"] == "processed"

    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / "fil_def14a"
    meta = json.loads((processed_dir / "tool_snapshot_meta.json").read_text(encoding="utf-8"))
    assert "def14a" in meta["parser_signature"]
    assert meta["has_financial_statement"] is False
    assert meta["has_xbrl"] is False


@pytest.mark.unit
def test_process_filing_sanitizes_fiscal_period_for_10q(tmp_path: Path) -> None:
    """验证 10-Q 不会写入无效 fiscal_period（如 FY）。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository=repository,
        ticker="AAPL",
        document_id="fil_0009",
        filename="sample.html",
        content=b"<html><body>10Q</body></html>",
        form_type="10-Q",
    )

    registry = ProcessorRegistry()
    registry.register(FakeSecProcessorWithXbrl10Q, name="sec_processor", priority=10)
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=registry,
    )
    result = pipeline.process_filing("AAPL", "fil_0009")
    assert result["status"] == "processed"

    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / "fil_0009"
    meta = json.loads((processed_dir / "tool_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["has_financial_statement"] is True
    assert meta["has_xbrl"] is True
