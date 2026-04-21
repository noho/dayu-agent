"""CnPipeline 占位行为测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.download_events import DownloadEventType
from dayu.fins.pipelines.upload_filing_events import UploadFilingEventType
from dayu.fins.pipelines.upload_material_events import UploadMaterialEventType
from dayu.fins.pipelines.cn_pipeline import CnPipeline
from dayu.fins.processors.registry import build_fins_processor_registry


def test_download_returns_not_implemented_status(tmp_path: Path) -> None:
    """验证 `download` 返回未实现状态。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )

    result = pipeline.download(
        ticker="000001",
        form_type="ANNUAL",
        start_date="2025-01-01",
        end_date="2025-12-31",
        overwrite=True,
    )

    assert result["pipeline"] == "cn"
    assert result["action"] == "download"
    assert result["status"] == "not_implemented"
    assert result["message"] == "CnPipeline.download 尚未实现"
    assert result["ticker"] == "000001"


@pytest.mark.asyncio
async def test_download_stream_emits_not_implemented_result(tmp_path: Path) -> None:
    """验证 `download_stream` 结束事件返回未实现结果。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )

    events = [
        event
        async for event in pipeline.download_stream(
            ticker="000001",
            form_type="ANNUAL",
            start_date="2025-01-01",
            end_date="2025-12-31",
            overwrite=False,
        )
    ]

    assert len(events) == 2
    assert events[0].event_type == DownloadEventType.PIPELINE_STARTED
    assert events[1].event_type == DownloadEventType.PIPELINE_COMPLETED
    assert events[1].payload["result"]["status"] == "not_implemented"
    assert events[1].payload["result"]["message"] == "CnPipeline.download_stream 尚未实现"


@pytest.mark.asyncio
async def test_upload_filing_stream_uploads_files_with_docling(tmp_path: Path) -> None:
    """验证 `upload_filing_stream` 可完成上传并生成 docling 主文件。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    pipeline._upload_service._convert_with_docling = lambda path: {  # type: ignore[attr-defined]
        "name": path.name,
        "format": "docling",
    }
    sample_file = tmp_path / "sample.pdf"
    sample_file.write_text("demo", encoding="utf-8")

    events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="000001",
            action="create",
            files=[sample_file],
            fiscal_year=2025,
            fiscal_period="FY",
            amended=False,
            filing_date="2025-01-01",
            report_date="2024-12-31",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]

    assert len(events) == 5
    assert events[0].event_type == UploadFilingEventType.UPLOAD_STARTED
    assert events[1].event_type == UploadFilingEventType.CONVERSION_STARTED
    assert events[1].payload["name"] == "sample.pdf"
    assert events[1].payload["message"] == "正在 convert"
    assert events[2].event_type == UploadFilingEventType.FILE_UPLOADED
    assert events[2].payload["name"] == "sample.pdf"
    assert events[2].payload["source"] == "original"
    assert events[3].event_type == UploadFilingEventType.FILE_UPLOADED
    assert events[3].payload["name"] == "sample_docling.json"
    assert events[3].payload["source"] == "docling"
    assert events[4].event_type == UploadFilingEventType.UPLOAD_COMPLETED
    result = events[4].payload["result"]
    assert result["status"] == "ok"
    assert str(result["document_id"]).startswith("fil_cn_")
    assert str(result["internal_document_id"]).startswith("cn_")
    company_meta = pipeline._company_repository.get_company_meta("000001")  # type: ignore[attr-defined]
    assert company_meta.company_id == "000001"
    assert company_meta.company_name == "平安银行"
    meta = pipeline._source_repository.get_source_meta("000001", result["document_id"], SourceKind.FILING)  # type: ignore[attr-defined]
    assert str(meta["primary_document"]).endswith("_docling.json")


@pytest.mark.asyncio
async def test_upload_material_stream_uploads_files_with_docling(tmp_path: Path) -> None:
    """验证 `upload_material_stream` 可完成上传并生成 docling 主文件。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    pipeline._upload_service._convert_with_docling = lambda path: {  # type: ignore[attr-defined]
        "name": path.name,
        "format": "docling",
    }
    sample_file = tmp_path / "material.pdf"
    sample_file.write_text("demo", encoding="utf-8")

    events = [
        event
        async for event in pipeline.upload_material_stream(
            ticker="000001",
            action="create",
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=[sample_file],
            filing_date="2025-01-01",
            report_date="2024-12-31",
            company_id="000001",
            company_name="平安银行",
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
    assert result["status"] == "ok"
    assert str(result["document_id"]).startswith("mat_")
    company_meta = pipeline._company_repository.get_company_meta("000001")  # type: ignore[attr-defined]
    assert company_meta.company_id == "000001"
    assert company_meta.company_name == "平安银行"
    meta = pipeline._source_repository.get_source_meta("000001", result["document_id"], SourceKind.MATERIAL)  # type: ignore[attr-defined]
    assert str(meta["primary_document"]).endswith("_docling.json")


@pytest.mark.asyncio
async def test_upload_filing_stream_auto_resolves_create_update_skip(tmp_path: Path) -> None:
    """验证 upload_filing_stream 在未显式传 action 时会自动 create/update/skip。"""

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    pipeline._upload_service._convert_with_docling = lambda path: {  # type: ignore[attr-defined]
        "name": path.name,
        "format": "docling",
    }
    sample_file = tmp_path / "sample.pdf"
    sample_file.write_text("demo-v1", encoding="utf-8")

    create_events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="000001",
            action=None,
            files=[sample_file],
            fiscal_year=2025,
            fiscal_period="FY",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]
    create_result = create_events[-1].payload["result"]
    assert create_result["status"] == "ok"
    assert create_result["filing_action"] == "create"

    skip_events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="000001",
            action=None,
            files=[sample_file],
            fiscal_year=2025,
            fiscal_period="FY",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]
    skip_result = skip_events[-1].payload["result"]
    assert skip_result["status"] == "skipped"
    assert skip_result["filing_action"] == "update"
    assert skip_result["skip_reason"] == "already_uploaded"
    assert [event.event_type.value for event in skip_events] == [
        "upload_started",
        "file_skipped",
        "upload_completed",
    ]

    sample_file.write_text("demo-v2", encoding="utf-8")
    update_events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="000001",
            action=None,
            files=[sample_file],
            fiscal_year=2025,
            fiscal_period="FY",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]
    update_result = update_events[-1].payload["result"]
    assert update_result["status"] == "ok"
    assert update_result["filing_action"] == "update"
    assert update_result["document_version"] == "v2"


@pytest.mark.asyncio
async def test_upload_material_stream_overwrite_resets_single_document(tmp_path: Path) -> None:
    """验证 upload_material_stream 的 overwrite 会重置当前 material 文档而非保留旧文件。"""

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    pipeline._upload_service._convert_with_docling = lambda path: {  # type: ignore[attr-defined]
        "name": path.name,
        "format": "docling",
    }
    old_file = tmp_path / "deck_old.pdf"
    new_file = tmp_path / "deck_new.pdf"
    old_file.write_text("old", encoding="utf-8")
    new_file.write_text("new", encoding="utf-8")

    first_events = [
        event
        async for event in pipeline.upload_material_stream(
            ticker="000001",
            action=None,
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=[old_file],
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]
    document_id = str(first_events[-1].payload["result"]["document_id"])

    second_events = [
        event
        async for event in pipeline.upload_material_stream(
            ticker="000001",
            action=None,
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=[new_file],
            company_id="000001",
            company_name="平安银行",
            overwrite=True,
        )
    ]
    second_result = second_events[-1].payload["result"]
    assert second_result["status"] == "ok"
    assert second_result["material_action"] == "update"
    assert second_result["document_id"] == document_id

    handle = pipeline._source_repository.get_source_handle("000001", document_id, SourceKind.MATERIAL)  # type: ignore[attr-defined]
    file_names = sorted(meta.uri.split("/")[-1] for meta in pipeline._blob_repository.list_files(handle))  # type: ignore[attr-defined]
    assert file_names == ["deck_new.pdf", "deck_new_docling.json"]
