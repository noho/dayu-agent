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
    UploadFilingCommandPayload,
    UploadFilingsFromCommandPayload,
    UploadMaterialCommandPayload,
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


def _build_fins_command(args: argparse.Namespace) -> FinsCommand:
    """将 argparse 参数转换为 `FinsCommand`。

    Args:
        args: 命令行参数对象。

    Returns:
        统一财报命令对象。

    Raises:
        ValueError: 命令不支持时抛出。
    """

    command_name = str(args.command)
    if command_name not in FINS_COMMANDS:
        raise ValueError(f"不是财报命令: {command_name}")
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
    stream_enabled = command_name in {"download", "process", "upload_filing", "upload_material"}
    return FinsCommand(
        name=FinsCommandName(command_name),
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
    return 0
