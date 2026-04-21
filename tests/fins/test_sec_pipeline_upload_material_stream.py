"""SecPipeline upload_material_stream 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.upload_material_events import UploadMaterialEventType
from dayu.fins.pipelines.sec_pipeline import SecPipeline
from dayu.fins.processors.registry import build_fins_processor_registry


@pytest.mark.asyncio
async def test_upload_material_stream_uploads_docling_files(tmp_path: Path) -> None:
    """验证 `upload_material_stream` 可完成上传并生成 docling 主文件。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    pipeline._upload_service._convert_with_docling = lambda path: {  # type: ignore[attr-defined]
        "name": path.name,
        "format": "docling",
    }
    material_file = tmp_path / "material.pdf"
    material_file.write_text("demo material", encoding="utf-8")

    events = [
        event
        async for event in pipeline.upload_material_stream(
            ticker="AAPL",
            action="create",
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=[material_file],
            filing_date="2025-05-01",
            report_date="2025-03-31",
            company_id="320193",
            company_name="Apple Inc.",
            ticker_aliases=["AAPL", "APC"],
            overwrite=False,
        )
    ]

    assert len(events) == 5
    assert events[0].event_type == UploadMaterialEventType.UPLOAD_STARTED
    assert events[1].event_type == UploadMaterialEventType.CONVERSION_STARTED
    assert events[1].payload["name"] == "material.pdf"
    assert events[1].payload["message"] == "正在 convert"
    assert events[2].event_type == UploadMaterialEventType.FILE_UPLOADED
    assert events[2].payload["name"] == "material.pdf"
    assert events[2].payload["source"] == "original"
    assert events[3].event_type == UploadMaterialEventType.FILE_UPLOADED
    assert events[3].payload["name"] == "material_docling.json"
    assert events[3].payload["source"] == "docling"
    assert events[4].event_type == UploadMaterialEventType.UPLOAD_COMPLETED
    result = events[4].payload["result"]
    assert result["action"] == "upload_material"
    assert result["ticker"] == "AAPL"
    assert result["status"] == "ok"
    assert str(result["document_id"]).startswith("mat_")
    company_meta = pipeline._company_repository.get_company_meta("AAPL")  # type: ignore[attr-defined]
    assert company_meta.company_id == "320193"
    assert company_meta.company_name == "Apple Inc."
    assert company_meta.ticker_aliases == ["AAPL", "APC"]
    meta = pipeline._source_repository.get_source_meta("AAPL", result["document_id"], SourceKind.MATERIAL)  # type: ignore[attr-defined]
    assert str(meta["primary_document"]).endswith("_docling.json")


@pytest.mark.asyncio
async def test_upload_material_stream_auto_action_and_overwrite_reset(tmp_path: Path) -> None:
    """验证 SecPipeline material 上传会自动解析动作并在 overwrite 时重置单文档。"""

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    pipeline._upload_service._convert_with_docling = lambda path: {  # type: ignore[attr-defined]
        "name": path.name,
        "format": "docling",
    }
    old_file = tmp_path / "deck_old.pdf"
    new_file = tmp_path / "deck_new.pdf"
    old_file.write_text("old material", encoding="utf-8")
    new_file.write_text("new material", encoding="utf-8")

    create_events = [
        event
        async for event in pipeline.upload_material_stream(
            ticker="AAPL",
            action=None,
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=[old_file],
            company_id="320193",
            company_name="Apple Inc.",
            overwrite=False,
        )
    ]
    create_result = create_events[-1].payload["result"]
    assert create_result["material_action"] == "create"

    overwrite_events = [
        event
        async for event in pipeline.upload_material_stream(
            ticker="AAPL",
            action=None,
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=[new_file],
            company_id="320193",
            company_name="Apple Inc.",
            overwrite=True,
        )
    ]
    overwrite_result = overwrite_events[-1].payload["result"]
    assert overwrite_result["status"] == "ok"
    assert overwrite_result["material_action"] == "update"
    assert overwrite_result["document_id"] == create_result["document_id"]

    handle = pipeline._source_repository.get_source_handle("AAPL", overwrite_result["document_id"], SourceKind.MATERIAL)  # type: ignore[attr-defined]
    file_names = sorted(meta.uri.split("/")[-1] for meta in pipeline._blob_repository.list_files(handle))  # type: ignore[attr-defined]
    assert file_names == ["deck_new.pdf", "deck_new_docling.json"]
