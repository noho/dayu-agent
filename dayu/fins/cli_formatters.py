"""CLI 结果格式化函数集。

职责：
- 将 Pipeline 操作结果字典格式化为面向终端阅读的多行文本。
- 纯展示逻辑，不包含业务控制流。
- 从 `cli.py` 提取以降低单文件复杂度。

使用方式：
    from .cli_formatters import (
        format_cli_result,
        format_download_stream_event_line,
        format_process_stream_event_line,
        format_upload_stream_event_line,
    )
"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Optional, TYPE_CHECKING

from dayu.fins._converters import int_or_zero, optional_int
from dayu.contracts.fins import (
    DownloadFailedFile,
    DownloadFilingResultItem,
    DownloadFilingResultStatus,
    DownloadResultData,
    DownloadSummary,
    FinsCommandName,
    FinsResultData,
    ProcessDocumentResultItem,
    ProcessResultData,
    ProcessSingleResultData,
    ProcessSummary,
    UploadFileResultItem,
    UploadFilingResultData,
    UploadFilingsFromMaterialItem,
    UploadFilingsFromRecognizedItem,
    UploadFilingsFromResultData,
    UploadFilingsFromSkippedItem,
    UploadMaterialResultData,
)

from .ingestion.process_events import ProcessEventType
from .pipelines.download_events import DownloadEventType
from .pipelines.upload_filing_events import UploadFilingEventType
from .pipelines.upload_material_events import UploadMaterialEventType

if TYPE_CHECKING:
    from .ingestion.process_events import ProcessEvent
    from .pipelines.download_events import DownloadEvent
    from .pipelines.upload_filing_events import UploadFilingEvent
    from .pipelines.upload_material_events import UploadMaterialEvent


def format_download_stream_event_line(event: "DownloadEvent") -> Optional[str]:
    """将下载事件格式化为单行回显文本。

    Args:
        event: 下载事件。

    Returns:
        单行文本；不需要输出时返回 ``None``。
    """

    payload = event.payload if isinstance(event.payload, dict) else {}
    if event.event_type == DownloadEventType.PIPELINE_STARTED:
        return (
            f"[download] started ticker={event.ticker} "
            f"form_type={payload.get('form_type') or '-'} "
            f"overwrite={bool(payload.get('overwrite', False))} "
            f"rebuild={bool(payload.get('rebuild', False))}"
        )
    if event.event_type == DownloadEventType.COMPANY_RESOLVED:
        company_name = str(payload.get("company_name", "")).strip() or "-"
        issuer_type = (
            str(payload.get("issuer_type", "")).strip()
            or str(payload.get("sec_issuer_type", "")).strip()
            or "-"
        )
        return f"[download] company_resolved ticker={event.ticker} company_name={company_name} issuer_type={issuer_type}"
    if event.event_type == DownloadEventType.FILING_STARTED:
        form_type = str(payload.get("form_type", "")).strip() or "-"
        filing_date = str(payload.get("filing_date", "")).strip() or "-"
        total_filings = payload.get("total_filings")
        total_text = total_filings if total_filings is not None else "?"
        return (
            f"[download] filing_started document_id={event.document_id or '-'} "
            f"form_type={form_type} filing_date={filing_date} total_filings={total_text}"
        )
    if event.event_type in {
        DownloadEventType.FILE_DOWNLOADED,
        DownloadEventType.FILE_SKIPPED,
        DownloadEventType.FILE_FAILED,
    }:
        name = str(payload.get("name", "")).strip() or "-"
        reason, message = _resolve_download_reason_fields(payload)
        extra_parts: list[str] = []
        if reason:
            extra_parts.append(f"reason={reason}")
        if message:
            extra_parts.append(f"message={message}")
        suffix = f" {' '.join(extra_parts)}" if extra_parts else ""
        return f"[download] {event.event_type} document_id={event.document_id or '-'} name={name}{suffix}"
    if event.event_type in {
        DownloadEventType.FILING_COMPLETED,
        DownloadEventType.FILING_FAILED,
    }:
        filing_payload = _extract_download_filing_payload(payload)
        status = str(filing_payload.get("status", "")).strip() or event.event_type
        downloaded_files = filing_payload.get("downloaded_files")
        skipped_files = filing_payload.get("skipped_files")
        failed_files = filing_payload.get("failed_files")
        reason, message = _resolve_download_reason_fields(filing_payload)
        extra_parts: list[str] = [f"status={status}"]
        if downloaded_files is not None:
            extra_parts.append(f"downloaded_files={downloaded_files}")
        if skipped_files is not None:
            extra_parts.append(f"skipped_files={skipped_files}")
        if failed_files is not None:
            failed_count = len(failed_files) if isinstance(failed_files, list) else failed_files
            extra_parts.append(f"failed_files={failed_count}")
        if reason:
            extra_parts.append(f"reason={reason}")
        if message:
            extra_parts.append(f"message={message}")
        return f"[download] {event.event_type} document_id={event.document_id or '-'} {' '.join(extra_parts)}"
    if event.event_type == DownloadEventType.PIPELINE_COMPLETED:
        return f"[download] completed ticker={event.ticker}"
    return None


def format_process_stream_event_line(event: "ProcessEvent") -> Optional[str]:
    """将预处理事件格式化为单行回显文本。

    Args:
        event: 预处理事件。

    Returns:
        单行文本；不需要输出时返回 ``None``。
    """

    payload = event.payload if isinstance(event.payload, dict) else {}
    if event.event_type == ProcessEventType.PIPELINE_STARTED:
        total_documents = payload.get("total_documents")
        return (
            f"[process] started ticker={event.ticker} "
            f"total_documents={total_documents if total_documents is not None else '?'} "
            f"overwrite={bool(payload.get('overwrite', False))} ci={bool(payload.get('ci', False))}"
        )
    if event.event_type == ProcessEventType.DOCUMENT_STARTED:
        source_kind = str(payload.get("source_kind", "")).strip() or "-"
        return f"[process] document_started source_kind={source_kind} document_id={event.document_id or '-'}"
    if event.event_type in {
        ProcessEventType.DOCUMENT_COMPLETED,
        ProcessEventType.DOCUMENT_SKIPPED,
        ProcessEventType.DOCUMENT_FAILED,
    }:
        source_kind = str(payload.get("source_kind", "")).strip() or "-"
        reason = str(payload.get("reason", "")).strip()
        suffix = f" reason={reason}" if reason else ""
        return (
            f"[process] {event.event_type} source_kind={source_kind} "
            f"document_id={event.document_id or '-'}{suffix}"
        )
    if event.event_type == ProcessEventType.PIPELINE_COMPLETED:
        return f"[process] completed ticker={event.ticker}"
    return None


def format_upload_stream_event_line(
    event: "UploadFilingEvent | UploadMaterialEvent",
) -> Optional[str]:
    """将上传事件格式化为单行回显文本。

    Args:
        event: 上传事件。

    Returns:
        单行文本；不需要输出时返回 ``None``。
    """
    payload = event.payload if isinstance(event.payload, dict) else {}
    if event.event_type == UploadFilingEventType.UPLOAD_STARTED:
        action = str(payload.get("action", "")).strip() or "-"
        file_count = int_or_zero(payload.get("file_count"))
        return (
            f"[upload] started ticker={event.ticker} document_id={event.document_id or '-'} "
            f"action={action} file_count={file_count}"
        )
    if event.event_type == UploadFilingEventType.CONVERSION_STARTED:
        name = str(payload.get("name", "")).strip() or "-"
        message = str(payload.get("message", "")).strip() or "正在 convert"
        return (
            f"[upload] converting ticker={event.ticker} document_id={event.document_id or '-'} "
            f"name={name} message={message}"
        )
    if event.event_type in {
        UploadFilingEventType.FILE_UPLOADED,
        UploadFilingEventType.FILE_SKIPPED,
        UploadFilingEventType.FILE_FAILED,
        UploadMaterialEventType.FILE_UPLOADED,
        UploadMaterialEventType.FILE_SKIPPED,
        UploadMaterialEventType.FILE_FAILED,
    }:
        name = str(payload.get("name", "")).strip() or "-"
        size = payload.get("size")
        reason = payload.get("reason")
        message = payload.get("message")
        extra_parts: list[str] = []
        if size is not None:
            extra_parts.append(f"size={size}")
        if reason:
            extra_parts.append(f"reason={reason}")
        if message:
            extra_parts.append(f"message={message}")
        suffix = f" {' '.join(extra_parts)}" if extra_parts else ""
        return f"[upload] {event.event_type} name={name}{suffix}"
    if event.event_type in {
        UploadFilingEventType.UPLOAD_FAILED,
        UploadMaterialEventType.UPLOAD_FAILED,
    }:
        error_text = str(payload.get("error", "")).strip() or "-"
        return f"[upload] failed ticker={event.ticker} document_id={event.document_id or '-'} error={error_text}"
    if event.event_type in {
        UploadFilingEventType.UPLOAD_COMPLETED,
        UploadMaterialEventType.UPLOAD_COMPLETED,
    }:
        return f"[upload] completed ticker={event.ticker} document_id={event.document_id or '-'}"
    return None


def coerce_cli_result(command: FinsCommandName | str, result: dict[str, Any]) -> FinsResultData:
    """按命令名把原始 CLI 结果收敛为强类型结果。

    Args:
        command: 子命令名称。
        result: 原始 pipeline 结果字典。

    Returns:
        对应命令的强类型结果对象。
    """

    command_name = command
    if isinstance(command, str) and command in FinsCommandName._value2member_map_:
        command_name = FinsCommandName(command)
    if command_name == FinsCommandName.DOWNLOAD:
        return _coerce_download_result(result)
    if command_name == FinsCommandName.UPLOAD_FILINGS_FROM:
        return _coerce_upload_filings_from_result(result)
    if command_name == FinsCommandName.UPLOAD_FILING:
        return _coerce_upload_filing_result(result)
    if command_name == FinsCommandName.UPLOAD_MATERIAL:
        return _coerce_upload_material_result(result)
    if command_name == FinsCommandName.PROCESS:
        return _coerce_process_result(result)
    return _coerce_process_single_result(result)


def format_cli_result(command: FinsCommandName | str, result: FinsResultData) -> str:
    """根据子命令格式化结果为可读文本。

    Args:
        command: 子命令名称。
        result: 子命令执行结果。

    Returns:
        面向终端阅读的多行文本。
    """
    command_name = command
    if isinstance(command, str) and command in FinsCommandName._value2member_map_:
        command_name = FinsCommandName(command)
    if command_name == FinsCommandName.DOWNLOAD:
        if not isinstance(result, DownloadResultData):
            raise TypeError("download 结果必须使用 DownloadResultData")
        return _format_download_result(result)
    if command_name == FinsCommandName.UPLOAD_FILINGS_FROM:
        if not isinstance(result, UploadFilingsFromResultData):
            raise TypeError("upload_filings_from 结果必须使用 UploadFilingsFromResultData")
        return _format_upload_filings_from_result(result)
    if command_name == FinsCommandName.UPLOAD_FILING:
        if not isinstance(result, UploadFilingResultData):
            raise TypeError("upload_filing 结果必须使用 UploadFilingResultData")
        return _format_upload_filing_result(result)
    if command_name == FinsCommandName.UPLOAD_MATERIAL:
        if not isinstance(result, UploadMaterialResultData):
            raise TypeError("upload_material 结果必须使用 UploadMaterialResultData")
        return _format_upload_material_result(result)
    if command_name == FinsCommandName.PROCESS:
        if not isinstance(result, ProcessResultData):
            raise TypeError("process 结果必须使用 ProcessResultData")
        return _format_process_result(result)
    if command_name in {FinsCommandName.PROCESS_FILING, FinsCommandName.PROCESS_MATERIAL}:
        if not isinstance(result, ProcessSingleResultData):
            raise TypeError("process_filing/process_material 结果必须使用 ProcessSingleResultData")
        return _format_process_single_result(result)
    return _format_generic_result(result, title=f"{command} 结果")


# ---------------------------------------------------------------------------
# 以下为各子命令专用格式化函数
# ---------------------------------------------------------------------------


def _coerce_download_result(result: DownloadResultData | dict[str, Any]) -> DownloadResultData:
    """把 download 结果规范化为强类型对象。"""

    if isinstance(result, DownloadResultData):
        return result
    filings_raw = result.get("filings")
    filings_list = filings_raw if isinstance(filings_raw, list) else []
    filing_items: list[DownloadFilingResultItem] = []
    for item in filings_list:
        if not isinstance(item, dict):
            continue
        failed_files_raw = item.get("failed_files")
        failed_files_list = failed_files_raw if isinstance(failed_files_raw, list) else []
        failed_files = tuple(
            DownloadFailedFile(
                file_name=_first_non_empty_text(failed.get("file_name")) if isinstance(failed, dict) else None,
                source=_first_non_empty_text(failed.get("source")) if isinstance(failed, dict) else None,
                reason_code=_first_non_empty_text(failed.get("reason_code")) if isinstance(failed, dict) else None,
                reason_message=(
                    _first_non_empty_text(failed.get("reason_message"), failed.get("message"), failed.get("error"))
                    if isinstance(failed, dict)
                    else None
                ),
            )
            for failed in failed_files_list
            if isinstance(failed, dict)
        )
        filing_items.append(
            DownloadFilingResultItem(
                document_id=str(item.get("document_id", "")).strip(),
                status=DownloadFilingResultStatus.from_raw(str(item.get("status", ""))),
                form_type=_first_non_empty_text(item.get("form_type")),
                filing_date=_first_non_empty_text(item.get("filing_date")),
                report_date=_first_non_empty_text(item.get("report_date")),
                downloaded_files=int_or_zero(item.get("downloaded_files")),
                skipped_files=int_or_zero(item.get("skipped_files")),
                failed_files=failed_files,
                has_xbrl=item.get("has_xbrl") if isinstance(item.get("has_xbrl"), bool) else None,
                reason_code=_first_non_empty_text(item.get("reason_code"), item.get("skip_reason"), item.get("reason")),
                reason_message=_first_non_empty_text(item.get("reason_message"), item.get("message"), item.get("error")),
                skip_reason=_first_non_empty_text(item.get("skip_reason")),
                filter_category=_first_non_empty_text(item.get("filter_category")),
            )
        )
    summary_raw = result.get("summary")
    summary = summary_raw if isinstance(summary_raw, dict) else {}
    warnings_raw = result.get("warnings")
    warnings = tuple(str(item) for item in warnings_raw) if isinstance(warnings_raw, list) else ()
    return DownloadResultData(
        pipeline=str(result.get("pipeline", "")).strip(),
        status=str(result.get("status", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        warnings=warnings,
        filings=tuple(filing_items),
        summary=DownloadSummary(
            total=int_or_zero(summary.get("total")),
            downloaded=int_or_zero(summary.get("downloaded")),
            skipped=int_or_zero(summary.get("skipped")),
            failed=int_or_zero(summary.get("failed")),
            elapsed_ms=int_or_zero(summary.get("elapsed_ms")),
            reused_downloads=int_or_zero(summary.get("reused_downloads")),
            converted=int_or_zero(summary.get("converted")),
        ),
    )


def _format_download_result(result: DownloadResultData) -> str:
    """格式化 download 结果为可读文本。

    Args:
        result: ``pipeline.download`` 返回结果。

    Returns:
        面向终端阅读的多行文本。
    """
    downloaded_items = [item for item in result.filings if item.status == DownloadFilingResultStatus.DOWNLOADED]
    skipped_items = [item for item in result.filings if item.status == DownloadFilingResultStatus.SKIPPED]
    failed_items = [item for item in result.filings if item.status == DownloadFilingResultStatus.FAILED]
    lines = [
        "下载结果",
        f"- ticker: {result.ticker}",
        (
            f"- 汇总: total={result.summary.total}, downloaded={result.summary.downloaded}, "
            f"skipped={result.summary.skipped}, failed={result.summary.failed}, "
            f"elapsed_ms={result.summary.elapsed_ms}, reused_downloads={result.summary.reused_downloads}, "
            f"converted={result.summary.converted}"
        ),
    ]
    if result.warnings:
        lines.append("- warnings:")
        for warning in result.warnings:
            lines.append(f"  - {warning}")
    lines.append("成功下载的 filings:")
    lines.extend(_format_filing_items(downloaded_items))
    lines.append("跳过的 filings:")
    lines.extend(_format_filing_items(skipped_items))
    lines.append("失败的 filings:")
    lines.extend(_format_filing_items(failed_items))
    return "\n".join(lines)


def _format_filing_items(items: list[DownloadFilingResultItem]) -> list[str]:
    """格式化 filing 列表。

    Args:
        items: filing 结果列表。

    Returns:
        单行文本列表；无数据时返回 ``["  - （无）"]``。
    """
    if not items:
        return ["  - （无）"]
    lines: list[str] = []
    for item in items:
        failed_count = len(item.failed_files)
        reason, message = _resolve_download_reason_fields(item)
        lines.append(
            (
                "  - "
                f"{item.document_id} | form={item.form_type or ''} | filing_date={item.filing_date or ''} | "
                f"report_date={item.report_date or ''} | status={item.status} | "
                f"downloaded_files={item.downloaded_files} | skipped_files={item.skipped_files} | "
                f"failed_files={failed_count} | reason={reason or '-'} | message={message or '-'}"
            )
        )
    return lines


def _extract_download_filing_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """提取 download filing 级事件的标准负载。

    Args:
        payload: 原始事件负载。

    Returns:
        `filing_result` 存在时返回其字典，否则返回顶层负载。

    Raises:
        无。
    """

    filing_payload = payload.get("filing_result")
    if isinstance(filing_payload, dict):
        return filing_payload
    return payload


def _resolve_download_reason_fields(payload: DownloadFilingResultItem | dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """解析 download 事件中的原因码与说明文本。

    Args:
        payload: 文件级或 filing 级负载。

    Returns:
        `(reason_code, reason_message)` 二元组。

    Raises:
        无。
    """

    if isinstance(payload, dict):
        reason = _first_non_empty_text(
            payload.get("reason_code"),
            payload.get("skip_reason"),
            payload.get("reason"),
        )
        message = _first_non_empty_text(
            payload.get("reason_message"),
            payload.get("message"),
            payload.get("error"),
        )
        failed_files_raw = payload.get("failed_files")
        failed_files = failed_files_raw if isinstance(failed_files_raw, list) else []
    else:
        reason = _first_non_empty_text(payload.reason_code, payload.skip_reason)
        message = _first_non_empty_text(payload.reason_message)
        failed_files = list(payload.failed_files)
    if message is None:
        if failed_files:
            messages: list[str] = []
            for item in failed_files:
                if isinstance(item, dict):
                    item_message = _first_non_empty_text(
                        item.get("reason_message"),
                        item.get("message"),
                        item.get("error"),
                    )
                else:
                    item_message = _first_non_empty_text(item.reason_message)
                if item_message is None or item_message in messages:
                    continue
                messages.append(item_message)
            if messages:
                message = "；".join(messages[:2])
                if len(messages) > 2:
                    message = f"{message} 等{len(messages)}项"
    return reason, message


def _first_non_empty_text(*values: Any) -> Optional[str]:
    """返回第一个非空白文本值。

    Args:
        *values: 待筛选的候选值。

    Returns:
        第一个非空白字符串；若均为空则返回 `None`。

    Raises:
        无。
    """

    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _coerce_upload_filings_from_result(result: UploadFilingsFromResultData | dict[str, Any]) -> UploadFilingsFromResultData:
    """把 upload_filings_from 结果规范化为强类型对象。"""

    if isinstance(result, UploadFilingsFromResultData):
        return result
    recognized_raw = result.get("recognized")
    material_raw = result.get("material")
    skipped_raw = result.get("skipped")
    return UploadFilingsFromResultData(
        script_path=str(result.get("script_path", "")).strip(),
        script_platform=str(result.get("script_platform", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        source_dir=str(result.get("source_dir", "")).strip(),
        total_files=int_or_zero(result.get("total_files")),
        recognized_count=int_or_zero(result.get("recognized_count")),
        material_count=int_or_zero(result.get("material_count")),
        skipped_count=int_or_zero(result.get("skipped_count")),
        recognized=tuple(
            UploadFilingsFromRecognizedItem(
                file=str(item.get("file", "")).strip(),
                fiscal_year=optional_int(item.get("fiscal_year")),
                fiscal_period=_first_non_empty_text(item.get("fiscal_period")),
            )
            for item in (recognized_raw if isinstance(recognized_raw, list) else [])
            if isinstance(item, dict)
        ),
        material=tuple(
            UploadFilingsFromMaterialItem(
                file=str(item.get("file", "")).strip(),
                material_name=_first_non_empty_text(item.get("material_name")),
            )
            for item in (material_raw if isinstance(material_raw, list) else [])
            if isinstance(item, dict)
        ),
        skipped=tuple(
            UploadFilingsFromSkippedItem(
                file=str(item.get("file", "")).strip(),
                reason=_first_non_empty_text(item.get("reason")),
            )
            for item in (skipped_raw if isinstance(skipped_raw, list) else [])
            if isinstance(item, dict)
        ),
    )


def _format_upload_filings_from_result(result: UploadFilingsFromResultData) -> str:
    """格式化 ``upload_filings_from`` 结果为可读文本。

    Args:
        result: ``upload_filings_from`` 返回结果。

    Returns:
        面向终端阅读的多行文本。
    """
    lines = [
        "批量上传脚本生成结果",
        f"- ticker: {result.ticker}",
        f"- source_dir: {result.source_dir}",
        f"- script_path: {result.script_path}",
        f"- script_platform: {result.script_platform}",
        (
            f"- 汇总: total_files={result.total_files}, "
            f"recognized={result.recognized_count}, material={result.material_count}, skipped={result.skipped_count}"
        ),
        "识别成功的文件 (upload_filing):",
    ]
    lines.extend(_format_upload_filings_from_items(list(result.recognized)))
    lines.append("材料文件 (upload_material):")
    lines.extend(_format_upload_filings_from_material_items(list(result.material)))
    lines.append("跳过的文件:")
    lines.extend(_format_upload_filings_from_skipped_items(list(result.skipped)))
    return "\n".join(lines)


def _format_upload_filings_from_material_items(items: list[UploadFilingsFromMaterialItem]) -> list[str]:
    """格式化 ``upload_filings_from`` 材料文件（upload_material）条目。

    Args:
        items: 材料条目列表，每条包含 ``file`` 和 ``material_name`` 字段。

    Returns:
        单行文本列表；无数据时返回 ``["  - （无）"]``。
    """
    if not items:
        return ["  - （无）"]
    lines: list[str] = []
    for item in items:
        lines.append(f"  - {item.file} | material_name={item.material_name or ''}")
    return lines


def _format_upload_filings_from_items(items: list[UploadFilingsFromRecognizedItem]) -> list[str]:
    """格式化 ``upload_filings_from`` 识别成功条目。

    Args:
        items: 成功识别条目列表。

    Returns:
        单行文本列表；无数据时返回 ``["  - （无）"]``。
    """
    if not items:
        return ["  - （无）"]
    lines: list[str] = []
    for item in items:
        lines.append(f"  - {item.file} | fiscal_year={item.fiscal_year or ''} | fiscal_period={item.fiscal_period or ''}")
    return lines


def _format_upload_filings_from_skipped_items(items: list[UploadFilingsFromSkippedItem]) -> list[str]:
    """格式化 ``upload_filings_from`` 跳过条目。

    Args:
        items: 跳过条目列表。

    Returns:
        单行文本列表；无数据时返回 ``["  - （无）"]``。
    """
    if not items:
        return ["  - （无）"]
    lines: list[str] = []
    for item in items:
        lines.append(f"  - {item.file} | reason={item.reason or ''}")
    return lines


def _coerce_upload_filing_result(result: UploadFilingResultData | dict[str, Any]) -> UploadFilingResultData:
    """把 upload_filing 结果规范化为强类型对象。"""

    if isinstance(result, UploadFilingResultData):
        return result
    files_raw = result.get("files")
    file_items = tuple(
        UploadFileResultItem(path=str(item).strip())
        for item in (files_raw if isinstance(files_raw, list) else [])
        if str(item).strip()
    )
    return UploadFilingResultData(
        pipeline=str(result.get("pipeline", "")).strip(),
        status=str(result.get("status", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        filing_action=str(result.get("filing_action", result.get("action", ""))).strip(),
        files=file_items,
        form_type=_first_non_empty_text(result.get("form_type")),
        fiscal_year=optional_int(result.get("fiscal_year")),
        fiscal_period=_first_non_empty_text(result.get("fiscal_period")),
        amended=result.get("amended") if isinstance(result.get("amended"), bool) else None,
        company_id=_first_non_empty_text(result.get("company_id")),
        company_name=_first_non_empty_text(result.get("company_name")),
        document_id=_first_non_empty_text(result.get("document_id")),
        primary_document=_first_non_empty_text(result.get("primary_document")),
        uploaded_files=optional_int(result.get("uploaded_files")),
        document_version=_first_non_empty_text(result.get("document_version")),
        source_fingerprint=_first_non_empty_text(result.get("source_fingerprint")),
        filing_date=_first_non_empty_text(result.get("filing_date")),
        report_date=_first_non_empty_text(result.get("report_date")),
        overwrite=result.get("overwrite") if isinstance(result.get("overwrite"), bool) else None,
        skip_reason=_first_non_empty_text(result.get("skip_reason")),
        message=_first_non_empty_text(result.get("message")),
    )


def _format_upload_filing_result(result: UploadFilingResultData) -> str:
    """格式化上传类命令结果。

    Args:
        result: 上传命令结果字典。
        title: 标题文本。
        action_label: 动作字段名称（``filing_action`` 或 ``material_action``）。

    Returns:
        可读多行文本。
    """
    lines = [
        "上传财报结果",
        f"- pipeline: {result.pipeline}",
        f"- ticker: {result.ticker}",
        f"- status: {result.status}",
        f"- filing_action: {result.filing_action}",
    ]
    _append_optional_field_lines(
        lines,
        [
            ("form_type", result.form_type),
            ("fiscal_year", result.fiscal_year),
            ("fiscal_period", result.fiscal_period),
            ("amended", result.amended),
            ("company_id", result.company_id),
            ("company_name", result.company_name),
            ("document_id", result.document_id),
            ("primary_document", result.primary_document),
            ("uploaded_files", result.uploaded_files),
            ("document_version", result.document_version),
            ("source_fingerprint", result.source_fingerprint),
            ("filing_date", result.filing_date),
            ("report_date", result.report_date),
            ("overwrite", result.overwrite),
            ("skip_reason", result.skip_reason),
            ("message", result.message),
        ],
    )
    lines.append("files:")
    lines.extend(_format_scalar_list([item.path for item in result.files]))
    return "\n".join(lines)


def _coerce_upload_material_result(result: UploadMaterialResultData | dict[str, Any]) -> UploadMaterialResultData:
    """把 upload_material 结果规范化为强类型对象。"""

    if isinstance(result, UploadMaterialResultData):
        return result
    files_raw = result.get("files")
    file_items = tuple(
        UploadFileResultItem(path=str(item).strip())
        for item in (files_raw if isinstance(files_raw, list) else [])
        if str(item).strip()
    )
    return UploadMaterialResultData(
        pipeline=str(result.get("pipeline", "")).strip(),
        status=str(result.get("status", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        material_action=str(result.get("material_action", result.get("action", ""))).strip(),
        files=file_items,
        form_type=_first_non_empty_text(result.get("form_type")),
        material_name=_first_non_empty_text(result.get("material_name")),
        company_id=_first_non_empty_text(result.get("company_id")),
        company_name=_first_non_empty_text(result.get("company_name")),
        document_id=_first_non_empty_text(result.get("document_id")),
        internal_document_id=_first_non_empty_text(result.get("internal_document_id")),
        primary_document=_first_non_empty_text(result.get("primary_document")),
        uploaded_files=optional_int(result.get("uploaded_files")),
        document_version=_first_non_empty_text(result.get("document_version")),
        source_fingerprint=_first_non_empty_text(result.get("source_fingerprint")),
        filing_date=_first_non_empty_text(result.get("filing_date")),
        report_date=_first_non_empty_text(result.get("report_date")),
        overwrite=result.get("overwrite") if isinstance(result.get("overwrite"), bool) else None,
        skip_reason=_first_non_empty_text(result.get("skip_reason")),
        message=_first_non_empty_text(result.get("message")),
    )


def _format_upload_material_result(result: UploadMaterialResultData) -> str:
    """格式化上传材料命令结果。"""

    lines = [
        "上传材料结果",
        f"- pipeline: {result.pipeline}",
        f"- ticker: {result.ticker}",
        f"- status: {result.status}",
        f"- material_action: {result.material_action}",
    ]
    _append_optional_field_lines(
        lines,
        [
            ("form_type", result.form_type),
            ("material_name", result.material_name),
            ("company_id", result.company_id),
            ("company_name", result.company_name),
            ("document_id", result.document_id),
            ("internal_document_id", result.internal_document_id),
            ("primary_document", result.primary_document),
            ("uploaded_files", result.uploaded_files),
            ("document_version", result.document_version),
            ("source_fingerprint", result.source_fingerprint),
            ("filing_date", result.filing_date),
            ("report_date", result.report_date),
            ("overwrite", result.overwrite),
            ("skip_reason", result.skip_reason),
            ("message", result.message),
        ],
    )
    lines.append("files:")
    lines.extend(_format_scalar_list([item.path for item in result.files]))
    return "\n".join(lines)


def _coerce_process_document_items(values: object) -> tuple[ProcessDocumentResultItem, ...]:
    """把 process 明细列表规范化为强类型对象。"""

    if not isinstance(values, list):
        return ()
    items: list[ProcessDocumentResultItem] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        items.append(
            ProcessDocumentResultItem(
                document_id=str(value.get("document_id", "")).strip(),
                status=str(value.get("status", "")).strip(),
                reason=_first_non_empty_text(value.get("reason")),
                form_type=_first_non_empty_text(value.get("form_type")),
                fiscal_year=optional_int(value.get("fiscal_year")),
                quality=_first_non_empty_text(value.get("quality")),
                has_xbrl=value.get("has_xbrl") if isinstance(value.get("has_xbrl"), bool) else None,
                section_count=optional_int(value.get("section_count")),
                table_count=optional_int(value.get("table_count")),
                skip_reason=_first_non_empty_text(value.get("skip_reason")),
                source_kind=_first_non_empty_text(value.get("source_kind")),
            )
        )
    return tuple(items)


def _coerce_process_summary(value: object) -> ProcessSummary:
    """把 process 汇总规范化为强类型对象。"""

    if isinstance(value, ProcessSummary):
        return value
    if not isinstance(value, dict):
        return ProcessSummary(total=0, processed=0, skipped=0, failed=0)
    return ProcessSummary(
        total=int_or_zero(value.get("total")),
        processed=int_or_zero(value.get("processed")),
        skipped=int_or_zero(value.get("skipped")),
        failed=int_or_zero(value.get("failed")),
        todo=bool(value.get("todo", False)),
    )


def _coerce_process_result(result: ProcessResultData | dict[str, Any]) -> ProcessResultData:
    """把 process 结果规范化为强类型对象。"""

    if isinstance(result, ProcessResultData):
        return result
    return ProcessResultData(
        pipeline=str(result.get("pipeline", "")).strip(),
        status=str(result.get("status", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        overwrite=bool(result.get("overwrite", False)),
        ci=bool(result.get("ci", False)),
        filings=_coerce_process_document_items(result.get("filings")),
        filing_summary=_coerce_process_summary(result.get("filing_summary")),
        materials=_coerce_process_document_items(result.get("materials")),
        material_summary=_coerce_process_summary(result.get("material_summary")),
    )


def _format_process_result(result: ProcessResultData) -> str:
    """格式化 ``process`` 命令结果。

    Args:
        result: ``pipeline.process`` 返回结果。

    Returns:
        可读多行文本。
    """
    processed_filings = _filter_items_by_status(list(result.filings), "processed")
    skipped_filings = _filter_items_by_status(list(result.filings), "skipped")
    failed_filings = _filter_items_by_status(list(result.filings), "failed")
    processed_materials = _filter_items_by_status(list(result.materials), "processed")
    skipped_materials = _filter_items_by_status(list(result.materials), "skipped")
    failed_materials = _filter_items_by_status(list(result.materials), "failed")
    lines = [
        "全量处理结果",
        f"- ticker: {result.ticker}",
        f"- status: {result.status}",
        _format_process_summary_line("filings", result.filing_summary),
    ]
    if result.material_summary.todo is True:
        lines.append("- materials 处理: 未实现（TODO）")
    else:
        lines.append(_format_process_summary_line("materials", result.material_summary))
    lines.append("成功处理的 filings:")
    lines.extend(_format_process_items(processed_filings))
    lines.append("跳过的 filings:")
    lines.extend(_format_process_items(skipped_filings))
    lines.append("失败的 filings:")
    lines.extend(_format_process_items(failed_filings))
    lines.append("成功处理的 materials:")
    lines.extend(_format_process_items(processed_materials))
    lines.append("跳过的 materials:")
    lines.extend(_format_process_items(skipped_materials))
    lines.append("失败的 materials:")
    lines.extend(_format_process_items(failed_materials))
    return "\n".join(lines)


def _coerce_process_single_result(result: ProcessSingleResultData | dict[str, Any]) -> ProcessSingleResultData:
    """把 process_filing/process_material 结果规范化为强类型对象。"""

    if isinstance(result, ProcessSingleResultData):
        return result
    return ProcessSingleResultData(
        pipeline=str(result.get("pipeline", "")).strip(),
        action=str(result.get("action", "")).strip(),
        status=str(result.get("status", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        document_id=str(result.get("document_id", "")).strip(),
        overwrite=bool(result.get("overwrite", False)),
        ci=bool(result.get("ci", False)),
        reason=_first_non_empty_text(result.get("reason")),
        form_type=_first_non_empty_text(result.get("form_type")),
        fiscal_year=optional_int(result.get("fiscal_year")),
        quality=_first_non_empty_text(result.get("quality")),
        has_xbrl=result.get("has_xbrl") if isinstance(result.get("has_xbrl"), bool) else None,
        section_count=optional_int(result.get("section_count")),
        table_count=optional_int(result.get("table_count")),
        skip_reason=_first_non_empty_text(result.get("skip_reason")),
        message=_first_non_empty_text(result.get("message")),
    )


def _format_process_single_result(result: ProcessSingleResultData) -> str:
    """格式化 ``process_filing/process_material`` 结果。

    Args:
        result: 单文档处理结果。

    Returns:
        可读多行文本。
    """
    title = "处理单文档结果" if not result.action else f"{result.action} 结果"
    lines = [
        title,
        f"- pipeline: {result.pipeline}",
        f"- ticker: {result.ticker}",
        f"- document_id: {result.document_id}",
        f"- status: {result.status}",
    ]
    _append_optional_field_lines(
        lines,
        [
            ("reason", result.reason),
            ("form_type", result.form_type),
            ("fiscal_year", result.fiscal_year),
            ("section_count", result.section_count),
            ("table_count", result.table_count),
            ("quality", result.quality),
            ("has_xbrl", result.has_xbrl),
            ("overwrite", result.overwrite),
            ("message", result.message),
        ],
    )
    return "\n".join(lines)


def _format_generic_result(result: FinsResultData, *, title: str) -> str:
    """格式化通用结果。

    Args:
        result: 命令结果字典。
        title: 输出标题。

    Returns:
        可读多行文本。
    """
    result_dict = asdict(result)
    lines = [
        title,
        f"- pipeline: {result_dict.get('pipeline', '')}",
        f"- action: {result_dict.get('action', '')}",
        f"- status: {result_dict.get('status', '')}",
        f"- ticker: {result_dict.get('ticker', '')}",
    ]
    extras = _extract_extra_fields(
        result=result_dict,
        excluded={"pipeline", "action", "status", "ticker"},
    )
    if extras:
        lines.append("返回字段:")
        for key, value in extras:
            lines.append(f"  - {key}: {_format_value_inline(value)}")
    return "\n".join(lines)


def _format_process_items(items: list[ProcessDocumentResultItem]) -> list[str]:
    """格式化 process 明细条目。

    Args:
        items: 条目列表。

    Returns:
        可读条目列表；为空时返回默认占位文本。
    """
    if not items:
        return ["  - （无）"]
    lines: list[str] = []
    for item in items:
        line = f"  - {item.document_id} | status={item.status}"
        optional_parts = _collect_optional_parts(
            [
                ("reason", item.reason),
                ("form_type", item.form_type),
                ("fiscal_year", item.fiscal_year),
                ("quality", item.quality),
                ("has_xbrl", item.has_xbrl),
                ("section_count", item.section_count),
                ("table_count", item.table_count),
                ("skip_reason", item.skip_reason),
            ]
        )
        if optional_parts:
            line = f"{line} | {' | '.join(optional_parts)}"
        lines.append(line)
    return lines


def _filter_items_by_status(items: list[ProcessDocumentResultItem], status: str) -> list[ProcessDocumentResultItem]:
    """按状态过滤条目。

    Args:
        items: 原始条目列表。
        status: 目标状态。

    Returns:
        过滤后的条目列表。
    """
    return [item for item in items if item.status == status]


def _append_optional_field_lines(lines: list[str], fields: list[tuple[str, object | None]]) -> None:
    """将存在值的字段追加为单行文本。

    Args:
        lines: 输出文本列表（就地修改）。
        fields: 待追加字段集合。
    """
    for key, value in fields:
        if value is None or value == "":
            continue
        lines.append(f"- {key}: {_format_value_inline(value)}")


def _format_scalar_list(values: list[Any]) -> list[str]:
    """格式化标量列表。

    Args:
        values: 标量列表。

    Returns:
        可读条目行列表。
    """
    if not values:
        return ["  - （无）"]
    return [f"  - {str(item)}" for item in values]


def _collect_optional_parts(fields: list[tuple[str, object | None]]) -> list[str]:
    """收集条目中的可选字段片段。

    Args:
        fields: 字段集合。

    Returns:
        ``key=value`` 文本片段列表。
    """
    parts: list[str] = []
    for key, value in fields:
        if value is None or value == "":
            continue
        parts.append(f"{key}={_format_value_inline(value)}")
    return parts


def _format_process_summary_line(label: str, summary: ProcessSummary) -> str:
    """格式化 process 汇总单行。"""

    return (
        f"- {label} 汇总: total={summary.total}, processed={summary.processed}, "
        f"skipped={summary.skipped}, failed={summary.failed}"
    )


def _extract_extra_fields(result: dict[str, Any], excluded: set[str]) -> list[tuple[str, Any]]:
    """提取额外字段列表。

    Args:
        result: 源结果字典。
        excluded: 需排除的字段键集合。

    Returns:
        ``(key, value)`` 列表。
    """
    extras: list[tuple[str, Any]] = []
    for key, value in result.items():
        if key in excluded:
            continue
        extras.append((key, value))
    return extras


def _format_value_inline(value: Any, max_len: int = 180) -> str:
    """将任意值格式化为单行文本。

    Args:
        value: 任意值。
        max_len: 最大长度，超出时截断。

    Returns:
        单行可读文本。
    """
    if isinstance(value, (dict, list)):
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        rendered = str(value)
    compact = " ".join(rendered.split())
    if len(compact) <= max_len:
        return compact
    return f"{compact[: max_len - 3]}..."
