"""DoclingUploadService 真实 PDF 集成测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.fins.domain.document_models import SourceHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.docling_upload_service import DoclingUploadService, _convert_file_with_docling
from tests.fins.storage_testkit import FsStorageTestContext, build_fs_storage_test_context

pytestmark = pytest.mark.integration


def _fixture_pdf_path() -> Path:
    """返回真实 Docling PDF fixture 路径。

    Args:
        无。

    Returns:
        真实 PDF fixture 文件路径。

    Raises:
        FileNotFoundError: fixture 缺失时抛出。
    """

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "docling"
        / "dayu_docling_integration_fixture.pdf"
    )
    if not fixture_path.exists():
        raise FileNotFoundError(f"Docling PDF fixture 不存在: {fixture_path}")
    return fixture_path


def test_convert_file_with_docling_reads_real_pdf_table() -> None:
    """验证真实 PDF 经 Docling 转换后包含稳定表格数据。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    result = _convert_file_with_docling(_fixture_pdf_path())
    assert result["name"] == "dayu_docling_integration_fixture"

    tables = result.get("tables")
    assert isinstance(tables, list)
    assert len(tables) == 1

    first_table = tables[0]
    assert isinstance(first_table, dict)

    table_data = first_table.get("data")
    assert isinstance(table_data, dict)
    assert table_data.get("num_rows") == 4
    assert table_data.get("num_cols") == 3

    table_cells = table_data.get("table_cells")
    assert isinstance(table_cells, list)
    assert table_cells

    first_cell = table_cells[0]
    assert isinstance(first_cell, dict)
    assert first_cell.get("text") == "Metric"
    assert first_cell.get("column_header") is True


def test_docling_upload_service_execute_upload_stores_real_outputs(tmp_path: Path) -> None:
    """验证真实上传链路会入库原始 PDF 与 `_docling.json`。

    Args:
        tmp_path: 临时工作区目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    context: FsStorageTestContext = build_fs_storage_test_context(tmp_path)
    service = DoclingUploadService(
        source_repository=context.source_repository,
        blob_repository=context.blob_repository,
    )

    result = service.execute_upload(
        ticker="AAPL",
        source_kind=SourceKind.MATERIAL,
        action="create",
        document_id="docling_fixture_material",
        internal_document_id="docling_fixture_material",
        form_type="MATERIAL_OTHER",
        files=[_fixture_pdf_path()],
        overwrite=False,
        meta={"material_name": "Docling Fixture", "ingest_method": "upload"},
    )

    assert result.status == "uploaded"
    event_names = {event.name for event in result.file_events}
    assert "dayu_docling_integration_fixture.pdf" in event_names
    assert "dayu_docling_integration_fixture_docling.json" in event_names

    stored_meta = context.source_repository.get_source_meta(
        "AAPL",
        "docling_fixture_material",
        SourceKind.MATERIAL,
    )
    assert stored_meta["primary_document"] == "dayu_docling_integration_fixture_docling.json"
    assert stored_meta["material_name"] == "Docling Fixture"

    file_entries = stored_meta["files"]
    assert isinstance(file_entries, list)
    stored_names = {entry["name"] for entry in file_entries}
    assert stored_names == {
        "dayu_docling_integration_fixture.pdf",
        "dayu_docling_integration_fixture_docling.json",
    }
    assert stored_meta["source_fingerprint"]

    handle = SourceHandle(
        ticker="AAPL",
        document_id="docling_fixture_material",
        source_kind=SourceKind.MATERIAL.value,
    )
    original_pdf_bytes = context.blob_repository.read_file_bytes(
        handle,
        "dayu_docling_integration_fixture.pdf",
    )
    generated_docling_bytes = context.blob_repository.read_file_bytes(
        handle,
        "dayu_docling_integration_fixture_docling.json",
    )
    assert original_pdf_bytes.startswith(b"%PDF")
    assert b"tables" in generated_docling_bytes
