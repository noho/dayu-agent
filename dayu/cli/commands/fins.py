"""CLI 财报命令构建与执行模块。

模块职责：
- 将 argparse 参数转换为 ``FinsCommand`` 对象。
- 格式化流式进度日志。
- 消费 ``FinsService`` 流式结果并输出最终结果。
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import AsyncIterator, cast

from dayu.contracts.fins import (
    DownloadCommandPayload,
    DownloadProgressPayload,
    DownloadResultData,
    FinsCommand,
    FinsCommandName,
    FinsEvent,
    FinsEventType,
    FinsProgressPayload,
    FinsResult,
    FinsResultData,
    ProcessCommandPayload,
    ProcessFilingCommandPayload,
    ProcessMaterialCommandPayload,
    ProcessResultData,
    ProcessSingleResultData,
    UploadFilingCommandPayload,
    UploadFilingResultData,
    UploadFilingsFromCommandPayload,
    UploadFilingsFromResultData,
    UploadMaterialCommandPayload,
    UploadMaterialResultData,
)
from dayu.fins.cli_support import (
    _coerce_document_ids_input as coerce_document_ids_input,
    _prepare_cli_args as prepare_cli_args,
)
from dayu.log import Log
from dayu.presenters import format_fins_cli_result
from dayu.services.contracts import FinsSubmitRequest

from dayu.cli.command_names import FINS_COMMANDS
from dayu.cli.dependency_setup import MODULE, _build_fins_ops_service, setup_loglevel

_STREAMING_COMMANDS = frozenset(
    {
        FinsCommandName.DOWNLOAD,
        FinsCommandName.PROCESS,
        FinsCommandName.UPLOAD_FILING,
        FinsCommandName.UPLOAD_MATERIAL,
    }
)


def _build_fins_command(args: argparse.Namespace) -> FinsCommand:
    """将 argparse 参数转换为 `FinsCommand`。

    Args:
        args: 命令行参数对象。

    Returns:
        统一财报命令对象。

    Raises:
        ValueError: 命令不支持时抛出。
    """

    raw_command_name = str(args.command)
    if raw_command_name not in FINS_COMMANDS:
        raise ValueError(f"不是财报命令: {raw_command_name}")
    command_name = FinsCommandName(raw_command_name)
    prepare_cli_args(args)
    if command_name == FinsCommandName.DOWNLOAD:
        payload = DownloadCommandPayload(
            ticker=str(args.ticker),
            form_type=tuple(args.form_type or ()),
            start_date=args.start_date,
            end_date=args.end_date,
            overwrite=bool(args.overwrite),
            rebuild=bool(getattr(args, "rebuild", False)),
            infer=bool(getattr(args, "infer", False)),
            ticker_aliases=tuple(getattr(args, "ticker_aliases", ()) or ()),
        )
    elif command_name == FinsCommandName.UPLOAD_FILING:
        payload = UploadFilingCommandPayload(
            ticker=str(args.ticker),
            files=tuple(Path(file_path) for file_path in (args.files or ())),
            fiscal_year=int(args.fiscal_year),
            action=str(args.action) if args.action is not None else None,
            fiscal_period=str(args.fiscal_period),
            amended=bool(args.amended),
            filing_date=args.filing_date,
            report_date=args.report_date,
            company_id=args.company_id,
            company_name=args.company_name,
            infer=bool(getattr(args, "infer", False)),
            ticker_aliases=tuple(getattr(args, "ticker_aliases", ()) or ()),
            overwrite=bool(args.overwrite),
        )
    elif command_name == FinsCommandName.UPLOAD_FILINGS_FROM:
        payload = UploadFilingsFromCommandPayload(
            ticker=str(args.ticker),
            source_dir=Path(args.source_dir),
            action=str(args.action) if args.action is not None else None,
            output_script=Path(args.output_script) if args.output_script else None,
            recursive=bool(args.recursive),
            amended=bool(args.amended),
            filing_date=args.filing_date,
            report_date=args.report_date,
            company_id=args.company_id,
            company_name=args.company_name,
            infer=bool(args.infer),
            overwrite=bool(args.overwrite),
            material_forms=tuple(getattr(args, "material_forms", ()) or ()),
            verbose=bool(args.verbose),
            debug=bool(args.debug),
            info=bool(args.info),
            quiet=bool(args.quiet),
            log_level=args.log_level,
        )
    elif command_name == FinsCommandName.UPLOAD_MATERIAL:
        payload = UploadMaterialCommandPayload(
            ticker=str(args.ticker),
            files=tuple(Path(file_path) for file_path in (args.files or ())),
            action=str(args.action) if args.action is not None else None,
            form_type=str(args.form_type),
            material_name=str(args.material_name),
            document_id=args.document_id,
            internal_document_id=args.internal_document_id,
            fiscal_year=args.fiscal_year,
            fiscal_period=args.fiscal_period,
            filing_date=args.filing_date,
            report_date=args.report_date,
            company_id=args.company_id,
            company_name=args.company_name,
            infer=bool(getattr(args, "infer", False)),
            ticker_aliases=tuple(getattr(args, "ticker_aliases", ()) or ()),
            overwrite=bool(args.overwrite),
        )
    elif command_name == FinsCommandName.PROCESS:
        payload = ProcessCommandPayload(
            ticker=str(args.ticker),
            document_ids=tuple(coerce_document_ids_input(getattr(args, "document_ids", None)) or ()),
            overwrite=bool(args.overwrite),
            ci=bool(args.ci),
        )
    elif command_name == FinsCommandName.PROCESS_FILING:
        payload = ProcessFilingCommandPayload(
            ticker=str(args.ticker),
            document_id=str(args.document_id),
            overwrite=bool(args.overwrite),
            ci=bool(args.ci),
        )
    elif command_name == FinsCommandName.PROCESS_MATERIAL:
        payload = ProcessMaterialCommandPayload(
            ticker=str(args.ticker),
            document_id=str(args.document_id),
            overwrite=bool(args.overwrite),
            ci=bool(args.ci),
        )
    else:
        raise ValueError(f"不是财报命令: {command_name}")
    stream_enabled = command_name in _STREAMING_COMMANDS
    return FinsCommand(
        name=command_name,
        payload=payload,
        stream=stream_enabled,
    )


def _format_fins_progress_line(command_name: FinsCommandName, payload: FinsProgressPayload) -> str:
    """格式化财报流式进度日志。

    Args:
        command_name: 命令名。
        payload: 进度事件负载。

    Returns:
        可读单行文本。

    Raises:
        无。
    """

    event_type = payload.event_type.value
    ticker = payload.ticker
    document_id = payload.document_id or ""
    message = ""
    extra_parts: list[str] = []
    action = getattr(payload, "action", None)
    if action:
        extra_parts.append(f"action={action}")
    name = getattr(payload, "name", None)
    if name:
        extra_parts.append(f"name={name}")
    if isinstance(payload, DownloadProgressPayload) and payload.form_type:
        extra_parts.append(f"form_type={payload.form_type}")
    file_count = getattr(payload, "file_count", None)
    if file_count is not None:
        extra_parts.append(f"file_count={file_count}")
    size = getattr(payload, "size", None)
    if size is not None:
        extra_parts.append(f"size={size}")
    message = getattr(payload, "message", None) or getattr(payload, "reason", None) or getattr(payload, "error", None) or ""
    line = f"[{command_name}] {event_type} ticker={ticker}"
    if document_id:
        line += f" document_id={document_id}"
    if extra_parts:
        line += f" {' '.join(extra_parts)}"
    if message:
        line += f" message={message}"
    return line


def _should_log_fins_progress_as_info(command_name: str) -> bool:
    """判断财报进度事件是否应按 INFO 输出。

    Args:
        command_name: 命令名。

    Returns:
        `upload_filing` / `upload_material` 返回 `True`，其余返回 `False`。

    Raises:
        无。
    """

    return command_name in {"upload_filing", "upload_material"}


_FINS_DOWNLOAD_SUCCESS_STATUSES: frozenset[str] = frozenset({"ok", "downloaded", "skipped", "cancelled"})
_FINS_UPLOAD_SUCCESS_STATUSES: frozenset[str] = frozenset({"ok", "uploaded", "deleted", "skipped"})
_FINS_PROCESS_SUCCESS_STATUSES: frozenset[str] = frozenset({"ok", "cancelled"})
_FINS_PROCESS_SINGLE_SUCCESS_STATUSES: frozenset[str] = frozenset({"processed", "skipped"})
_FINS_FAILURE_STATUS: str = "failed"


def _classify_fins_status(status: str, success_set: frozenset[str]) -> bool:
    """按 per-result-type 白名单判定单条结果是否需要以失败码上报。

    Args:
        status: 结果对象的 status 字段（已 strip）。
        success_set: 该结果类型已枚举的成功 status 白名单。

    Returns:
        当 status 命中 ``"failed"`` 时返回 ``True``；命中成功白名单时返回 ``False``。

    Raises:
        ValueError: status 既不在成功白名单也不在 ``"failed"`` 时抛出，
            强制未来新增 status 必须显式归类，避免被静默归"成功"。
    """

    if status == _FINS_FAILURE_STATUS:
        return True
    if status in success_set:
        return False
    raise ValueError(
        f"未知财报命令 status，未显式归类成功/失败: status={status!r}; "
        f"已知成功集合={sorted(success_set)}; 失败={_FINS_FAILURE_STATUS!r}"
    )


def _is_fins_result_failure(result: FinsResultData) -> bool:
    """判断财报命令结果是否需要以非零退出码上报。

    Args:
        result: 单次命令的最终结果。

    Returns:
        当结果显式标注 ``failed`` 状态时返回 ``True``；
        ``UploadFilingsFromResultData`` 没有 status 字段，本身代表
        脚本生成步骤已经成功完成，统一返回 ``False``；
        其它结果按 per-result-type 白名单分支显式归类。

    Raises:
        ValueError: 结果 status 既不在成功白名单也不是 ``"failed"`` 时抛出，
            通过类型 ``match`` 的穷尽性 + 白名单 ``case _`` 双重防线，
            强迫未来新增结果类型 / 新增 status 必须显式归类。
    """

    match result:
        case UploadFilingsFromResultData():
            return False
        case DownloadResultData():
            return _classify_fins_status(result.status, _FINS_DOWNLOAD_SUCCESS_STATUSES)
        case UploadFilingResultData() | UploadMaterialResultData():
            return _classify_fins_status(result.status, _FINS_UPLOAD_SUCCESS_STATUSES)
        case ProcessResultData():
            return _classify_fins_status(result.status, _FINS_PROCESS_SUCCESS_STATUSES)
        case ProcessSingleResultData():
            return _classify_fins_status(result.status, _FINS_PROCESS_SINGLE_SUCCESS_STATUSES)


def _describe_fins_status(result: FinsResultData) -> str:
    """提取财报命令结果的状态描述用于日志。

    Args:
        result: 单次命令的最终结果。

    Returns:
        若结果带 ``status`` 字段则返回其取值，否则返回 ``"<no-status>"``。

    Raises:
        无。
    """

    if isinstance(result, UploadFilingsFromResultData):
        return "<no-status>"
    return result.status


async def _consume_fins_stream(
    result_stream: AsyncIterator[FinsEvent],
    command_name: FinsCommandName,
) -> FinsResultData:
    """消费 `FinsService` 流式结果并返回最终结果。

    Args:
        result_stream: 流式事件迭代器。
        command_name: 命令名。

    Returns:
        最终结果字典。

    Raises:
        RuntimeError: 未收到最终结果时抛出。
    """

    final_result: FinsResultData | None = None
    async for event in result_stream:
        if not isinstance(event, FinsEvent):
            continue
        if event.type == FinsEventType.PROGRESS:
            line = _format_fins_progress_line(command_name, cast(FinsProgressPayload, event.payload))
            if _should_log_fins_progress_as_info(command_name):
                Log.info(line, module=MODULE)
            else:
                Log.verbose(line, module=MODULE)
            continue
        if event.type == FinsEventType.RESULT:
            final_result = cast(FinsResultData, event.payload)
    if final_result is None:
        raise RuntimeError(f"{command_name} 流式执行未返回最终结果")
    return final_result


def run_fins_command(args: argparse.Namespace) -> int:
    """执行财报命令并输出结果。

    Args:
        args: 命令行参数对象。

    Returns:
        退出码，0 表示成功，1 表示失败。

    Raises:
        无。
    """

    try:
        setup_loglevel(args)
        service = _build_fins_ops_service(args)
        command = _build_fins_command(args)
        submission = service.submit(FinsSubmitRequest(command=command))
        execution = submission.execution
        if command.stream:
            if isinstance(execution, FinsResult):
                raise RuntimeError("流式命令收到同步结果，预期异步事件流")
            result = asyncio.run(_consume_fins_stream(execution, command.name))
        else:
            if not isinstance(execution, FinsResult):
                raise RuntimeError("财报命令返回类型异常")
            result = execution.data
    except Exception as exc:
        Log.error(f"财报命令执行失败: {exc}", module=MODULE)
        return 1
    print(format_fins_cli_result(command.name, result))
    # FinsResultData 各 variant（除 UploadFilingsFromResultData 之外）携带
    # status 字段，按 per-result-type 白名单显式归类成功/失败；未在白名单
    # 也未命中 "failed" 的 status 视为契约违规，按非零退出码上报，避免被
    # 静默归"成功"导致 CI 失真。
    try:
        is_failure = _is_fins_result_failure(result)
    except ValueError as exc:
        Log.error(f"财报命令结果 status 未显式归类: {exc}", module=MODULE)
        return 1
    if is_failure:
        Log.warn(
            f"财报命令以非成功状态结束: command={command.name.value}, status={_describe_fins_status(result)}",
            module=MODULE,
        )
        return 1
    return 0
