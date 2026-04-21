"""SecPipeline process_material 测试。"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Protocol

import pytest

from dayu.engine.processors.base import (
    SearchHit,
    SectionContent,
    SectionSummary,
    TableContent,
    TableSummary,
    build_search_hit,
    build_section_content,
    build_section_summary,
    build_table_content,
    build_table_summary,
)
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.engine.processors.source import Source
from dayu.fins.domain.document_models import MaterialCreateRequest
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.sec_pipeline import SecPipeline
from dayu.fins.storage.local_file_store import LocalFileStore
from tests.fins.storage_testkit import build_storage_core


class _MaterialRepositoryLike(Protocol):
    """process_material 测试所需的最小仓储边界。"""

    portfolio_root: Path

    def create_material(self, req: MaterialCreateRequest) -> object:
        """创建材料文档。"""


def _repository(repository: _MaterialRepositoryLike) -> _MaterialRepositoryLike:
    """显式收窄 build_storage_core 返回的仓储类型。"""

    return repository


class FakeJsonProcessor:
    """测试用 JSON 处理器。"""

    PARSER_VERSION = "fake_sec_material_processor_v1.0.0"

    @classmethod
    def get_parser_version(cls) -> str:
        """返回处理器版本。"""

        return str(cls.PARSER_VERSION)

    def __init__(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> None:
        """初始化处理器。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        del form_type
        del media_type
        self._source = source

    @classmethod
    def supports(
        cls,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> bool:
        """判断是否支持处理。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            是否支持。

        Raises:
            OSError: 访问失败时可能抛出。
        """

        del form_type
        resolved_media = str(media_type or source.media_type or "").lower()
        return "json" in resolved_media

    def list_sections(self) -> list[SectionSummary]:
        """返回章节列表。

        Args:
            无。

        Returns:
            章节列表。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        return [
            build_section_summary(
                ref="s_0001",
                title="M",
                level=1,
                parent_ref=None,
                preview="p",
            )
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
            RuntimeError: 读取失败时抛出。
        """

        return [
            build_table_summary(
                table_ref="t_0001",
                caption=None,
                context_before="ctx",
                row_count=1,
                col_count=1,
                table_type="data",
                headers=["A"],
                section_ref="s_0001",
                is_financial=False,
            )
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

        return build_section_content(
            ref=ref,
            title="M",
            content="",
            tables=[],
            word_count=0,
            contains_full_text=True,
        )

    def read_table(self, table_ref: str) -> TableContent:
        """读取表格内容。

        Args:
            table_ref: 表格引用。

        Returns:
            表格内容。

        Raises:
            KeyError: 表格不存在时抛出。
        """

        return build_table_content(
            table_ref=table_ref,
            caption=None,
            data_format="records",
            data=[],
            columns=["A"],
            row_count=0,
            col_count=1,
            section_ref="s_0001",
            table_type="data",
            is_financial=False,
        )

    def search(self, query: str, within_ref: Optional[str] = None) -> list[SearchHit]:
        """搜索内容。

        Args:
            query: 搜索词。
            within_ref: 可选范围。

        Returns:
            命中列表。

        Raises:
            RuntimeError: 搜索失败时抛出。
        """

        del query
        del within_ref
        return []

    def get_full_text(self) -> str:
        """返回文档全文。"""

        return ""

    def get_full_text_with_table_markers(self) -> str:
        """返回带表格占位符的全文。"""

        return ""


def _build_registry() -> ProcessorRegistry:
    """构建测试注册表。

    Args:
        无。

    Returns:
        注册表实例。

    Raises:
        RuntimeError: 注册失败时抛出。
    """

    registry = ProcessorRegistry()
    registry.register(FakeJsonProcessor, name="fake_json_processor", priority=100, overwrite=True)
    return registry


def _prepare_material(
    repository: _MaterialRepositoryLike,
    *,
    ticker: str,
    document_id: str,
    filename: str,
    with_file: bool = True,
    ingest_complete: bool = True,
) -> None:
    """准备材料文档。

    Args:
        repository: 仓储实例。
        ticker: 股票代码。
        document_id: 文档 ID。
        filename: 文件名。
        with_file: 是否写入实际文件条目。
        ingest_complete: 是否完成入库。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    file_entries = []
    if with_file:
        store = LocalFileStore(root=repository.portfolio_root, scheme="local")
        key = f"{ticker}/materials/{document_id}/{filename}"
        file_meta = store.put_object(key, BytesIO(b"{}"))
        file_entries = [file_meta]

    repository.create_material(
        MaterialCreateRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=document_id,
            form_type="MATERIAL_OTHER",
            primary_document=filename,
            files=file_entries,
            meta={
                "ingest_complete": ingest_complete,
                "document_version": "v1",
                "source_fingerprint": "fp_v1",
            },
        )
    )


@pytest.mark.unit
def test_sec_pipeline_process_material_processed_and_version_skip(tmp_path: Path) -> None:
    """验证 process_material 的处理与版本命中跳过。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = _repository(build_storage_core(tmp_path))
    _prepare_material(
        repository,
        ticker="AAPL",
        document_id="mat_001",
        filename="material_docling.json",
        with_file=True,
    )

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=_build_registry(),
    )

    first = pipeline.process_material("AAPL", "mat_001", overwrite=False)
    second = pipeline.process_material("AAPL", "mat_001", overwrite=False)
    third = pipeline.process_material("AAPL", "mat_001", overwrite=True)

    assert first["status"] == "processed"
    assert second["status"] == "skipped"
    assert second["reason"] == "version_matched"
    assert third["status"] == "processed"

    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / "mat_001"
    meta = json.loads((processed_dir / "tool_snapshot_meta.json").read_text(encoding="utf-8"))
    assert meta["parser_signature"]


@pytest.mark.unit
def test_sec_pipeline_process_marks_material_failed_when_source_missing(tmp_path: Path) -> None:
    """验证 process 在材料主文件缺失时会标记 failed。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = _repository(build_storage_core(tmp_path))
    _prepare_material(
        repository,
        ticker="AAPL",
        document_id="mat_missing",
        filename="missing_docling.json",
        with_file=False,
    )

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=_build_registry(),
    )

    result = pipeline.process("AAPL")

    assert result["material_summary"]["total"] == 1
    assert result["material_summary"]["failed"] == 1
    assert result["materials"][0]["status"] == "failed"
