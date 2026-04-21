"""SecPipeline upload_filing_stream 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.sec_pipeline import SecPipeline
from dayu.fins.pipelines.upload_filing_events import UploadFilingEventType
from dayu.fins.processors.registry import build_fins_processor_registry


@pytest.mark.asyncio
async def test_upload_filing_stream_uploads_docling_files(tmp_path: Path) -> None:
    """验证 `SecPipeline.upload_filing_stream` 可完成上传并生成 docling 主文件。

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
    filing_file = tmp_path / "filing.pdf"
    filing_file.write_text("demo filing", encoding="utf-8")

    events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="AAPL",
            action="create",
            files=[filing_file],
            fiscal_year=2025,
            fiscal_period="Q1",
            amended=False,
            filing_date="2025-05-01",
            report_date="2025-03-31",
            company_id="320193",
            company_name="Apple Inc.",
            ticker_aliases=["AAPL", "APC"],
            overwrite=False,
        )
    ]

    assert len(events) == 5
    assert events[0].event_type == UploadFilingEventType.UPLOAD_STARTED
    assert events[1].event_type == UploadFilingEventType.CONVERSION_STARTED
    assert events[1].payload["name"] == "filing.pdf"
    assert events[2].event_type == UploadFilingEventType.FILE_UPLOADED
    assert events[2].payload["name"] == "filing.pdf"
    assert events[2].payload["source"] == "original"
    assert events[3].event_type == UploadFilingEventType.FILE_UPLOADED
    assert events[3].payload["name"] == "filing_docling.json"
    assert events[4].event_type == UploadFilingEventType.UPLOAD_COMPLETED
    result = events[4].payload["result"]
    assert result["action"] == "upload_filing"
    assert result["ticker"] == "AAPL"
    assert result["status"] == "ok"
    assert str(result["document_id"]).startswith("fil_sec_")
    assert result["filing_action"] == "create"
    company_meta = pipeline._company_repository.get_company_meta("AAPL")  # type: ignore[attr-defined]
    assert company_meta.ticker_aliases == ["AAPL", "APC"]
    meta = pipeline._source_repository.get_source_meta("AAPL", result["document_id"], SourceKind.FILING)  # type: ignore[attr-defined]
    assert str(meta["primary_document"]).endswith("_docling.json")
    assert str(meta["form_type"]) == "Q1"


@pytest.mark.asyncio
async def test_upload_filing_stream_auto_action_and_overwrite_reset(tmp_path: Path) -> None:
    """验证 `SecPipeline.upload_filing_stream` 会自动解析动作并在 overwrite 时重置单文档。

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
    old_file = tmp_path / "q1_old.pdf"
    new_file = tmp_path / "q1_new.pdf"
    old_file.write_text("old filing", encoding="utf-8")
    new_file.write_text("new filing", encoding="utf-8")

    create_events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="AAPL",
            action=None,
            files=[old_file],
            fiscal_year=2025,
            fiscal_period="Q1",
            company_id="320193",
            company_name="Apple Inc.",
            overwrite=False,
        )
    ]
    create_result = create_events[-1].payload["result"]
    assert create_result["filing_action"] == "create"

    skip_events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="AAPL",
            action=None,
            files=[old_file],
            fiscal_year=2025,
            fiscal_period="Q1",
            company_id="320193",
            company_name="Apple Inc.",
            overwrite=False,
        )
    ]
    skip_result = skip_events[-1].payload["result"]
    assert skip_result["status"] == "skipped"
    assert skip_result["filing_action"] == "update"
    assert [event.event_type.value for event in skip_events] == [
        "upload_started",
        "file_skipped",
        "upload_completed",
    ]

    overwrite_events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="AAPL",
            action=None,
            files=[new_file],
            fiscal_year=2025,
            fiscal_period="Q1",
            company_id="320193",
            company_name="Apple Inc.",
            overwrite=True,
        )
    ]
    overwrite_result = overwrite_events[-1].payload["result"]
    assert overwrite_result["status"] == "ok"
    assert overwrite_result["filing_action"] == "update"
    assert overwrite_result["document_id"] == create_result["document_id"]

    handle = pipeline._source_repository.get_source_handle("AAPL", overwrite_result["document_id"], SourceKind.FILING)  # type: ignore[attr-defined]
    file_names = sorted(meta.uri.split("/")[-1] for meta in pipeline._blob_repository.list_files(handle))  # type: ignore[attr-defined]
    assert file_names == ["q1_new.pdf", "q1_new_docling.json"]
