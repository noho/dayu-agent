"""下载进度管理。

本模块只负责进度领域逻辑：任务状态模型、进度事件应用、下载流事件消费。
不负责页面渲染与 ``st.session_state`` 管理。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from queue import Queue
from typing import Literal, TypedDict

from dayu.contracts.fins import (
    DownloadFilingResultItem,
    DownloadFilingResultStatus,
    DownloadProgressPayload,
    FinsEvent,
    FinsEventType,
    FinsProgressEventName,
)
from dayu.services.contracts import FinsSubmission


class DownloadStatus(StrEnum):
    """下载任务状态枚举。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class DownloadQueueEvent:
    """下载后台线程向主线程投递的队列事件。"""

    kind: Literal["progress", "result", "error", "done"]
    payload: DownloadProgressPayload | None = None
    message: str = ""


class LogEntry(TypedDict):
    """日志条目结构。"""

    timestamp: str
    message: str
    level: str


@dataclass
class DownloadTaskState:
    """下载任务状态。"""

    session_id: str
    ticker: str
    status: DownloadStatus = DownloadStatus.PENDING
    progress: float = 0.0
    current_form_type: str | None = None
    current_document_id: str | None = None
    message: str = "等待开始..."
    downloaded_count: int = 0
    downloaded_filing_count: int = 0
    total_count: int | None = None
    errors: list[str] = field(default_factory=list)
    logs: list[LogEntry] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None


def create_download_task(session_id: str, ticker: str) -> DownloadTaskState:
    """创建新的下载任务初始状态。"""

    task = DownloadTaskState(
        session_id=session_id,
        ticker=ticker,
        status=DownloadStatus.RUNNING,
        started_at=datetime.now().isoformat(),
    )
    _add_log_entry(task, "下载任务已创建，等待事件流")
    return task


def apply_download_progress(task: DownloadTaskState, payload: DownloadProgressPayload) -> None:
    """按进度事件更新下载任务状态。"""

    task.current_form_type = payload.form_type
    task.current_document_id = payload.document_id

    match payload.event_type:
        case FinsProgressEventName.PIPELINE_STARTED:
            task.message = "开始下载任务..."
            task.status = DownloadStatus.RUNNING
            _add_log_entry(task, f"开始下载任务: {payload.ticker}")
        case FinsProgressEventName.COMPANY_RESOLVED:
            task.message = f"已解析公司信息: {payload.ticker}"
            _add_log_entry(task, task.message)
        case FinsProgressEventName.FILING_STARTED:
            form_label = payload.form_type or "文件"
            task.message = f"开始下载 {form_label}..."
            task.current_document_id = payload.document_id
            _add_log_entry(task, f"开始下载: {form_label}")
        case FinsProgressEventName.FILE_DOWNLOADED:
            task.downloaded_count += 1
            task.message = _build_file_downloaded_message(payload.name or "文件", payload.size)
            _add_log_entry(task, task.message)
            _update_download_progress_ratio(task)
        case FinsProgressEventName.FILE_SKIPPED:
            task.message = f"跳过已存在文件: {payload.name or '未知'}"
            _add_log_entry(task, task.message, level="warning")
        case FinsProgressEventName.FILE_FAILED:
            error_msg = f"下载失败: {payload.name or '未知'} - {payload.reason or '未知错误'}"
            _record_download_error(task, error_msg, mark_failed=False)
        case FinsProgressEventName.FILING_COMPLETED:
            task.downloaded_filing_count += 1
            task.message, log_level = _build_filing_completed_message(
                payload.form_type,
                payload.filing_result,
                payload.reason,
            )
            _add_log_entry(task, task.message, level=log_level)
            if payload.file_count is not None:
                task.total_count = payload.file_count
                _update_download_progress_ratio(task)
        case FinsProgressEventName.FILING_FAILED:
            error_msg = f"下载失败: {payload.form_type or '文件'} - {payload.reason or '未知错误'}"
            _record_download_error(task, error_msg, mark_failed=True)
        case FinsProgressEventName.PIPELINE_COMPLETED:
            task.message = "下载任务完成"
            task.status = DownloadStatus.COMPLETED
            task.progress = 100.0
            task.completed_at = datetime.now().isoformat()
            _add_log_entry(task, f"下载任务完成，共下载 {task.downloaded_count} 个文件")


def apply_download_completion(task: DownloadTaskState, success: bool = True, message: str = "") -> None:
    """应用下载任务终态。"""

    task.status = DownloadStatus.COMPLETED if success else DownloadStatus.FAILED
    task.progress = 100.0 if success else task.progress
    task.completed_at = datetime.now().isoformat()
    if message:
        task.message = message
    _add_log_entry(task, task.message, level="info" if success else "error")


async def consume_download_stream_events_to_queue(
    submission: FinsSubmission,
    event_queue: Queue[DownloadQueueEvent],
) -> None:
    """消费下载事件流并写入线程队列。"""

    execution = submission.execution
    if not isinstance(execution, AsyncIterator):
        event_queue.put(DownloadQueueEvent(kind="result", message="下载完成（同步模式）"))
        return

    async for event in execution:
        if not isinstance(event, FinsEvent):
            continue
        if event.type == FinsEventType.PROGRESS and isinstance(event.payload, DownloadProgressPayload):
            event_queue.put(DownloadQueueEvent(kind="progress", payload=event.payload))
        elif event.type == FinsEventType.RESULT:
            event_queue.put(DownloadQueueEvent(kind="result", message="下载完成"))
            break


def run_download_stream_worker(
    submission: FinsSubmission,
    event_queue: Queue[DownloadQueueEvent],
) -> None:
    """后台线程入口：运行异步下载流消费。"""

    try:
        asyncio.run(consume_download_stream_events_to_queue(submission, event_queue))
    except Exception as exception:  # noqa: BLE001
        event_queue.put(DownloadQueueEvent(kind="error", message=f"下载任务执行异常: {exception}"))
    finally:
        event_queue.put(DownloadQueueEvent(kind="done"))


def _update_download_progress_ratio(task: DownloadTaskState) -> None:
    """按总数和已下载数量刷新下载进度百分比。"""

    if task.total_count is None or task.total_count <= 0:
        return
    task.progress = min(100.0, (task.downloaded_count / task.total_count) * 100)


def _record_download_error(task: DownloadTaskState, error_msg: str, mark_failed: bool) -> None:
    """记录下载错误并按需标记任务失败。"""

    task.errors.append(error_msg)
    task.message = error_msg
    if mark_failed:
        task.status = DownloadStatus.FAILED
    _add_log_entry(task, error_msg, level="error")


def _format_download_size(size_in_bytes: int | None) -> str | None:
    """格式化下载文件大小。"""

    if size_in_bytes is None:
        return None
    return f"{size_in_bytes} 字节"


def _build_file_downloaded_message(filename: str, size_in_bytes: int | None) -> str:
    """构建文件下载完成消息。"""

    size_text = _format_download_size(size_in_bytes)
    if size_text is None:
        return f"已下载 {filename}"
    return f"已下载 {filename} ({size_text})"


def _build_filing_completed_message(
    form_type: str | None,
    filing_result: DownloadFilingResultItem | None,
    reason: str | None,
) -> tuple[str, str]:
    """构建 filing 完成日志消息。"""

    resolved_form_type = form_type or (filing_result.form_type if filing_result is not None else None) or "文件"
    if filing_result is None:
        return f"完成下载 {resolved_form_type}", "info"

    status = filing_result.status
    reason_text = filing_result.reason_message or filing_result.skip_reason or filing_result.reason_code or reason
    if status == DownloadFilingResultStatus.SKIPPED:
        if reason_text:
            return f"跳过下载 {resolved_form_type}: {reason_text}", "warning"
        return f"跳过下载 {resolved_form_type}", "warning"
    if status == DownloadFilingResultStatus.FAILED:
        if reason_text:
            return f"下载失败 {resolved_form_type}: {reason_text}", "error"
        return f"下载失败 {resolved_form_type}", "error"
    if filing_result.downloaded_files > 0:
        return f"完成下载 {resolved_form_type}（{filing_result.downloaded_files} 个文件）", "info"
    return f"完成下载 {resolved_form_type}", "info"


def _add_log_entry(task: DownloadTaskState, message: str, level: str = "info") -> None:
    """向下载任务追加日志。"""

    entry: LogEntry = {
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "level": level,
    }
    task.logs.append(entry)
