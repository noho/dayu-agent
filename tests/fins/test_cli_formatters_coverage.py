"""cli_formatters 补充覆盖测试。"""

from __future__ import annotations

from typing import cast

from dayu.contracts.fins import (
    DownloadFailedFile,
    DownloadFilingResultItem,
    DownloadFilingResultStatus,
    DownloadResultData,
    DownloadSummary,
    FinsCommandName,
    ProcessDocumentResultItem,
    ProcessResultData,
    ProcessSingleResultData,
    ProcessSummary,
    UploadFilingResultData,
)
from dayu.fins import cli_formatters as module
from dayu.fins.ingestion.process_events import ProcessEvent, ProcessEventType
from dayu.fins.pipelines.download_events import DownloadEvent, DownloadEventType
from dayu.fins.pipelines.upload_filing_events import UploadFilingEvent, UploadFilingEventType


def test_format_upload_stream_event_line_with_extra_fields() -> None:
    """覆盖上传事件行格式化的 reason/message 分支。"""

    event = UploadFilingEvent(
        event_type=UploadFilingEventType.FILE_FAILED,
        ticker="AAPL",
        payload={"name": "a.txt", "size": 10, "reason": "network", "message": "timeout"},
    )
    line = module.format_upload_stream_event_line(event)
    assert line is not None
    assert "reason=network" in line
    assert "message=timeout" in line

    converting_event = UploadFilingEvent(
        event_type=UploadFilingEventType.CONVERSION_STARTED,
        ticker="AAPL",
        document_id="fil_1",
        payload={"name": "a.pdf", "message": "正在 convert"},
    )
    converting_line = module.format_upload_stream_event_line(converting_event)
    assert converting_line is not None
    assert "converting" in converting_line
    assert "name=a.pdf" in converting_line

    failed_event = UploadFilingEvent(
        event_type=UploadFilingEventType.UPLOAD_FAILED,
        ticker="AAPL",
        payload={"error": "boom"},
    )
    assert "error=boom" in (module.format_upload_stream_event_line(failed_event) or "")

    unknown = UploadFilingEvent(
        event_type=cast(UploadFilingEventType, "unknown"),
        ticker="AAPL",
        payload={},
    )
    assert module.format_upload_stream_event_line(unknown) is None

    download_line = module.format_download_stream_event_line(
        DownloadEvent(
            event_type=DownloadEventType.FILE_FAILED,
            ticker="AAPL",
            document_id="fil_1",
            payload={"name": "x.htm", "reason_code": "download_error", "reason_message": "timeout"},
        )
    )
    assert download_line is not None
    assert "reason=download_error" in download_line
    assert "message=timeout" in download_line

    filing_line = module.format_download_stream_event_line(
        DownloadEvent(
            event_type=DownloadEventType.FILING_FAILED,
            ticker="AAPL",
            document_id="fil_1",
            payload={
                "filing_result": {
                    "status": "failed",
                    "failed_files": [{"reason_message": "network down"}],
                    "reason_code": "file_download_failed",
                }
            },
        )
    )
    assert filing_line is not None
    assert "reason=file_download_failed" in filing_line
    assert "message=network down" in filing_line

    process_line = module.format_process_stream_event_line(
        ProcessEvent(
            event_type=ProcessEventType.DOCUMENT_FAILED,
            ticker="AAPL",
            document_id="fil_1",
            payload={"source_kind": "filing", "reason": "parse_error"},
        )
    )
    assert process_line is not None
    assert "parse_error" in process_line


def test_format_download_company_resolved_prefers_sec_issuer_type() -> None:
    """覆盖 download company_resolved 对统一 `issuer_type` 字段的兼容读取。"""

    line = module.format_download_stream_event_line(
        DownloadEvent(
            event_type=DownloadEventType.COMPANY_RESOLVED,
            ticker="AAPL",
            payload={"company_name": "Apple Inc.", "sec_issuer_type": "domestic"},
        )
    )
    assert line is not None
    assert "company_name=Apple Inc." in line
    assert "issuer_type=domestic" in line

    cn_line = module.format_download_stream_event_line(
        DownloadEvent(
            event_type=DownloadEventType.COMPANY_RESOLVED,
            ticker="0700",
            payload={"company_name": "Tencent", "issuer_type": "cn"},
        )
    )
    assert cn_line is not None
    assert "issuer_type=cn" in cn_line


def test_format_cli_result_generic_path() -> None:
    """覆盖 format_cli_result 的 generic 分支。"""

    generic = module.format_cli_result(
        "unknown_cmd",
        ProcessSingleResultData(
            pipeline="sec",
            action="process",
            status="ok",
            ticker="AAPL",
            document_id="doc_1",
        ),
    )
    assert "unknown_cmd 结果" in generic


def test_coerce_cli_result_and_format_known_command() -> None:
    """覆盖已知命令结果的 coerce 与 formatter dispatch。"""

    result = module.coerce_cli_result(
        FinsCommandName.UPLOAD_FILING,
        {
            "pipeline": "sec",
            "status": "ok",
            "ticker": "AAPL",
            "filing_action": "create",
            "company_name": "Apple Inc.",
            "uploaded_files": "2",
            "files": ["report.htm", "report.xbrl"],
        },
    )

    assert isinstance(result, UploadFilingResultData)
    assert result.uploaded_files == 2

    rendered = module.format_cli_result(FinsCommandName.UPLOAD_FILING, result)
    assert "上传财报结果" in rendered
    assert "Apple Inc." in rendered


def test_download_and_upload_filings_from_formatters() -> None:
    """覆盖下载/批量上传格式化中的 warning 与空列表分支。"""

    download_text = module._format_download_result(
        DownloadResultData(
            pipeline="sec",
            status="ok",
            ticker="AAPL",
            warnings=("w1", "w2"),
            filings=(
                DownloadFilingResultItem(
                    document_id="x",
                    status=DownloadFilingResultStatus.FAILED,
                    failed_files=(DownloadFailedFile(reason_message="network down"),),
                    reason_code="file_download_failed",
                ),
            ),
            summary=DownloadSummary(total=1, downloaded=0, skipped=0, failed=1, elapsed_ms=9),
        )
    )
    assert "- warnings:" in download_text
    assert "w1" in download_text
    assert "reason=file_download_failed" in download_text
    assert "message=network down" in download_text

    assert module._format_upload_filings_from_items([]) == ["  - （无）"]
    assert module._format_upload_filings_from_skipped_items([]) == ["  - （无）"]


def test_process_and_scalar_helpers() -> None:
    """覆盖 process/materials 汇总与标量格式化分支。"""

    text = module._format_process_result(
        ProcessResultData(
            pipeline="sec",
            ticker="AAPL",
            status="ok",
            filing_summary=ProcessSummary(total=0, processed=0, skipped=0, failed=0),
            material_summary=ProcessSummary(total=1, processed=1, skipped=0, failed=0),
            filings=(),
            materials=(ProcessDocumentResultItem(document_id="m1", status="processed"),),
        )
    )
    assert "materials 汇总" in text

    assert module._format_scalar_list([]) == ["  - （无）"]


def test_extract_extra_and_value_inline_branches() -> None:
    """覆盖 extra 字段提取与内联值渲染分支。"""

    extras = module._extract_extra_fields({"a": 1, "b": 2}, excluded={"a"})
    assert extras == [("b", 2)]

    rendered_json = module._format_value_inline({"a": 1})
    assert rendered_json.startswith('{"a"')

    long_text = "x" * 300
    rendered_long = module._format_value_inline(long_text, max_len=20)
    assert rendered_long.endswith("...")


def test_formatters_accept_enum_event_types() -> None:
    """CLI formatter 应直接接受事件枚举值而非依赖裸字符串。"""

    download_line = module.format_download_stream_event_line(
        DownloadEvent(
            event_type=DownloadEventType.PIPELINE_STARTED,
            ticker="AAPL",
            payload={"form_type": "10-K", "overwrite": False, "rebuild": False},
        )
    )
    process_line = module.format_process_stream_event_line(
        ProcessEvent(
            event_type=ProcessEventType.DOCUMENT_FAILED,
            ticker="AAPL",
            document_id="fil_1",
            payload={"source_kind": "filing", "reason": "parse_error"},
        )
    )
    upload_line = module.format_upload_stream_event_line(
        UploadFilingEvent(
            event_type=UploadFilingEventType.UPLOAD_FAILED,
            ticker="AAPL",
            payload={"error": "boom"},
        )
    )

    assert download_line is not None and "started" in download_line
    assert process_line is not None and "parse_error" in process_line
    assert upload_line is not None and "error=boom" in upload_line
