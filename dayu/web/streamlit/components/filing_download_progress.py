"""Streamlit 下载状态组件。

负责下载任务状态模型、事件驱动状态更新与进度区渲染。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from typing import TypedDict

import streamlit as st

from dayu.contracts.fins import (
    DownloadFilingResultItem,
    DownloadProgressPayload,
    FinsProgressEventName,
)

_DOWNLOAD_STATUS_PENDING = "pending"
_DOWNLOAD_STATUS_RUNNING = "running"
_DOWNLOAD_STATUS_COMPLETED = "completed"
_DOWNLOAD_STATUS_FAILED = "failed"

# 公开导出，供 UI 层比较任务终态
DOWNLOAD_STATUS_COMPLETED = _DOWNLOAD_STATUS_COMPLETED
DOWNLOAD_STATUS_FAILED = _DOWNLOAD_STATUS_FAILED
_STATUS_CONTAINER_LOG_HEIGHT_PX = 220
_STATUS_CONTAINER_MAX_LOG_ITEMS = 120

_DOWNLOAD_STATUS_LABELS: dict[str, str] = {
    _DOWNLOAD_STATUS_PENDING: "等待中",
    _DOWNLOAD_STATUS_RUNNING: "运行中",
    _DOWNLOAD_STATUS_COMPLETED: "已完成",
    _DOWNLOAD_STATUS_FAILED: "失败",
}

_CSS_LOG_CONTAINER = (
    "border:1px solid #E5E7EB; border-radius:6px; padding:8px; "
    f"height:{_STATUS_CONTAINER_LOG_HEIGHT_PX}px; overflow-y:auto; background:#FAFAFA;"
)
_CSS_LOG_LINE = (
    "font-family:monospace; font-size:12px; line-height:1.5; margin-bottom:4px;"
)
_CSS_LOG_TIMESTAMP_COLOR = "#6B7280"
_CSS_LOG_MESSAGE_COLOR = "#111827"
_CSS_LOG_LEVEL_COLORS: dict[str, str] = {
    "error": "#DC2626",
    "warning": "#B45309",
    "info": "#2563EB",
}
_CSS_LOG_LEVEL_LABELS: dict[str, str] = {
    "error": "ERROR",
    "warning": "WARN",
    "info": "INFO",
}


class LogEntry(TypedDict):
    """日志条目结构。"""

    timestamp: str
    message: str
    level: str


class DownloadTaskStateSerialized(TypedDict, total=False):
    """下载任务状态序列化结构。"""

    session_id: str
    ticker: str
    status: str
    progress: float
    current_form_type: str | None
    current_document_id: str | None
    message: str
    downloaded_count: int
    downloaded_filing_count: int
    total_count: int | None
    errors: list[str]
    logs: list[LogEntry]
    started_at: str | None
    completed_at: str | None


@dataclass
class DownloadTaskState:
    """下载任务状态。"""

    session_id: str
    ticker: str
    status: str = _DOWNLOAD_STATUS_PENDING
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

    def to_dict(self) -> DownloadTaskStateSerialized:
        """转换为字典格式以便存储在 session_state 中。

        参数:
            无。

        返回值:
            可序列化状态字典。

        异常:
            无。
        """

        return {
            "session_id": self.session_id,
            "ticker": self.ticker,
            "status": self.status,
            "progress": self.progress,
            "current_form_type": self.current_form_type,
            "current_document_id": self.current_document_id,
            "message": self.message,
            "downloaded_count": self.downloaded_count,
            "downloaded_filing_count": self.downloaded_filing_count,
            "total_count": self.total_count,
            "errors": self.errors,
            "logs": self.logs,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: DownloadTaskStateSerialized) -> "DownloadTaskState":
        """从字典恢复对象。

        参数:
            data: 序列化状态字典。

        返回值:
            下载任务状态对象。

        异常:
            无。
        """

        return cls(
            session_id=data.get("session_id", ""),
            ticker=data.get("ticker", ""),
            status=data.get("status", _DOWNLOAD_STATUS_PENDING),
            progress=data.get("progress", 0.0),
            current_form_type=data.get("current_form_type"),
            current_document_id=data.get("current_document_id"),
            message=data.get("message", "等待开始..."),
            downloaded_count=data.get("downloaded_count", 0),
            downloaded_filing_count=data.get("downloaded_filing_count", 0),
            total_count=data.get("total_count"),
            errors=data.get("errors", []),
            logs=data.get("logs", []),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


def init_download_state() -> None:
    """初始化下载任务会话状态。
    """

    if "active_downloads" not in st.session_state:
        st.session_state.active_downloads = {}


def add_active_download(session_id: str, ticker: str) -> DownloadTaskState:
    """添加新的活跃下载任务。

    参数:
        session_id: 会话 ID。
        ticker: 股票代码。

    返回值:
        创建后的任务状态对象。

    异常:
        无。
    """

    init_download_state()
    task = DownloadTaskState(
        session_id=session_id,
        ticker=ticker,
        status=_DOWNLOAD_STATUS_RUNNING,
        started_at=datetime.now().isoformat(),
    )
    _add_log_entry(task, "下载任务已创建，等待事件流")
    st.session_state.active_downloads[session_id] = task.to_dict()
    return task


def build_download_logs_html(logs: list[LogEntry]) -> str:
    """构建可滚动下载日志 HTML。

    参数:
        logs: 下载任务日志列表。

    返回值:
        可直接渲染的 HTML 字符串。

    异常:
        无。
    """

    recent_logs = logs[-_STATUS_CONTAINER_MAX_LOG_ITEMS:]
    log_lines: list[str] = []
    for log in recent_logs:
        time_text = escape(_format_log_time(log.get("timestamp", "")))
        message_text = escape(log.get("message", ""))
        level = log.get("level", "info")
        level_text = _CSS_LOG_LEVEL_LABELS.get(level, _CSS_LOG_LEVEL_LABELS["info"])
        level_color = _CSS_LOG_LEVEL_COLORS.get(level, _CSS_LOG_LEVEL_COLORS["info"])

        line_html = (
            f'<div style="{_CSS_LOG_LINE}">'
            f'<span style="color:{_CSS_LOG_TIMESTAMP_COLOR};">[{time_text}]</span> '
            f'<span style="color:{level_color}; font-weight:600;">{level_text}</span> '
            f'<span style="color:{_CSS_LOG_MESSAGE_COLOR};">{message_text}</span>'
            "</div>"
        )
        log_lines.append(line_html)

    content_html = "".join(log_lines) if log_lines else "<div>暂无日志</div>"
    return (
        f'<div style="{_CSS_LOG_CONTAINER}">'
        f"{content_html}"
        "</div>"
    )


def update_download_progress(session_id: str, payload: DownloadProgressPayload) -> None:
    """按进度事件更新下载任务状态。

    参数:
        session_id: 会话 ID。
        payload: 进度事件负载。

    返回值:
        无。

    异常:
        无。
    """

    init_download_state()
    if session_id not in st.session_state.active_downloads:
        return

    task_data = st.session_state.active_downloads[session_id]
    task = DownloadTaskState.from_dict(task_data)
    task.current_form_type = payload.form_type
    task.current_document_id = payload.document_id

    event_type = payload.event_type
    if event_type == FinsProgressEventName.PIPELINE_STARTED:
        task.message = "开始下载任务..."
        task.status = _DOWNLOAD_STATUS_RUNNING
        _add_log_entry(task, f"开始下载任务: {payload.ticker}")
    elif event_type == FinsProgressEventName.COMPANY_RESOLVED:
        task.message = f"已解析公司信息: {payload.ticker}"
        _add_log_entry(task, task.message)
    elif event_type == FinsProgressEventName.FILING_STARTED:
        form_label = payload.form_type or "文件"
        task.message = f"开始下载 {form_label}..."
        task.current_document_id = payload.document_id
        _add_log_entry(task, f"开始下载: {form_label}")
    elif event_type == FinsProgressEventName.FILE_DOWNLOADED:
        task.downloaded_count += 1
        task.message = _build_file_downloaded_message(payload.name or "文件", payload.size)
        _add_log_entry(task, task.message)
        if task.total_count and task.total_count > 0:
            task.progress = min(100.0, (task.downloaded_count / task.total_count) * 100)
    elif event_type == FinsProgressEventName.FILE_SKIPPED:
        task.message = f"跳过已存在文件: {payload.name or '未知'}"
        _add_log_entry(task, task.message, level="warning")
    elif event_type == FinsProgressEventName.FILE_FAILED:
        error_msg = f"下载失败: {payload.name or '未知'} - {payload.reason or '未知错误'}"
        task.errors.append(error_msg)
        task.message = error_msg
        _add_log_entry(task, error_msg, level="error")
    elif event_type == FinsProgressEventName.FILING_COMPLETED:
        task.downloaded_filing_count += 1
        task.message, log_level = _build_filing_completed_message(
            payload.form_type,
            payload.filing_result,
            payload.reason,
        )
        _add_log_entry(task, task.message, level=log_level)
        if payload.file_count is not None:
            task.total_count = payload.file_count
    elif event_type == FinsProgressEventName.FILING_FAILED:
        error_msg = f"下载失败: {payload.form_type or '文件'} - {payload.reason or '未知错误'}"
        task.errors.append(error_msg)
        task.message = error_msg
        task.status = _DOWNLOAD_STATUS_FAILED
        _add_log_entry(task, error_msg, level="error")
    elif event_type == FinsProgressEventName.PIPELINE_COMPLETED:
        task.message = "下载任务完成"
        task.status = _DOWNLOAD_STATUS_COMPLETED
        task.progress = 100.0
        task.completed_at = datetime.now().isoformat()
        _add_log_entry(task, f"下载任务完成，共下载 {task.downloaded_count} 个文件")

    st.session_state.active_downloads[session_id] = task.to_dict()


def mark_download_completed(session_id: str, success: bool = True, message: str = "") -> None:
    """标记下载任务终态。

    参数:
        session_id: 会话 ID。
        success: 是否成功完成。
        message: 可选终态消息。

    返回值:
        无。

    异常:
        无。
    """

    init_download_state()
    if session_id not in st.session_state.active_downloads:
        return

    task_data = st.session_state.active_downloads[session_id]
    task = DownloadTaskState.from_dict(task_data)
    task.status = _DOWNLOAD_STATUS_COMPLETED if success else _DOWNLOAD_STATUS_FAILED
    task.progress = 100.0 if success else task.progress
    task.completed_at = datetime.now().isoformat()
    if message:
        task.message = message

    _add_log_entry(task, task.message, level="info" if success else "error")
    st.session_state.active_downloads[session_id] = task.to_dict()


def remove_active_download(session_id: str) -> None:
    """删除活跃下载任务。

    参数:
        session_id: 会话 ID。

    返回值:
        无。

    异常:
        无。
    """

    init_download_state()
    if session_id in st.session_state.active_downloads:
        del st.session_state.active_downloads[session_id]


def get_ticker_active_download(ticker: str) -> DownloadTaskState | None:
    """获取指定 ticker 的活跃任务。

    参数:
        ticker: 股票代码。

    返回值:
        活跃任务；不存在时返回 ``None``。

    异常:
        无。
    """

    init_download_state()
    for task_data in st.session_state.active_downloads.values():
        task = DownloadTaskState.from_dict(task_data)
        if task.ticker == ticker and task.status in (_DOWNLOAD_STATUS_PENDING, _DOWNLOAD_STATUS_RUNNING):
            return task
    return None


def render_download_progress_area(ticker: str) -> None:
    """渲染下载进度区域。

    参数:
        ticker: 当前页面股票代码。

    返回值:
        无。

    异常:
        无。
    """

    task = get_ticker_active_download(ticker)
    if task is None:
        return

    with st.container():
        st.markdown("---")
        st.markdown("**📥 正在下载财报**")
        progress_text = f"{task.progress:.1f}% - {task.message}"
        st.progress(task.progress / 100.0, text=progress_text)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"股票代码: {task.ticker}")
        with col2:
            st.caption(f"已下载 {task.downloaded_filing_count} 份财报、{task.downloaded_count} 个文件")
        with col3:
            status_text = _DOWNLOAD_STATUS_LABELS.get(task.status, task.status)
            st.caption(f"状态: {status_text}")

        if task.errors:
            with st.expander(f"⚠️ 错误信息 ({len(task.errors)} 条)"):
                for error in task.errors:
                    st.error(error)

        st.markdown("---")
        if task.status == _DOWNLOAD_STATUS_COMPLETED:
            st.success("✅ 下载完成！财报列表已更新。")
            if st.button("关闭任务", key=f"close_completed_download_{task.session_id}"):
                remove_active_download(task.session_id)
                st.rerun()
        elif task.status == _DOWNLOAD_STATUS_FAILED:
            st.error("❌ 下载失败，请检查错误信息。")
            if st.button("清除任务", key=f"clear_failed_download_{task.session_id}"):
                remove_active_download(task.session_id)
                st.rerun()


def _format_log_time(timestamp: str) -> str:
    """格式化日志时间为时分秒。

    参数:
        timestamp: ISO 格式时间戳字符串。

    返回值:
        ``HH:MM:SS`` 格式字符串；输入为空或无效时返回原值。

    异常:
        无。
    """

    if not timestamp:
        return ""
    try:
        return datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
    except ValueError:
        return timestamp


def _format_download_size(size_in_bytes: int | None) -> str | None:
    """格式化下载文件大小。

    参数:
        size_in_bytes: 文件字节数，为 ``None`` 时返回 ``None``。

    返回值:
        ``"{n} 字节"`` 格式字符串；输入为 ``None`` 时返回 ``None``。

    异常:
        无。
    """

    if size_in_bytes is None:
        return None
    return f"{size_in_bytes} 字节"


def _build_file_downloaded_message(filename: str, size_in_bytes: int | None) -> str:
    """构建文件下载完成消息。

    参数:
        filename: 文件名。
        size_in_bytes: 文件字节数，为 ``None`` 时省略大小信息。

    返回值:
        格式化后的下载完成消息字符串。

    异常:
        无。
    """

    size_text = _format_download_size(size_in_bytes)
    if size_text is None:
        return f"已下载 {filename}"
    return f"已下载 {filename} ({size_text})"


def _build_filing_completed_message(
    form_type: str | None,
    filing_result: DownloadFilingResultItem | None,
    reason: str | None,
) -> tuple[str, str]:
    """构建 filing 完成日志消息。

    参数:
        form_type: 表单类型；为 ``None`` 时回退到 ``filing_result.form_type``。
        filing_result: filing 下载结果项；为 ``None`` 时返回默认完成消息。
        reason: 失败/跳过原因，优先使用 ``filing_result`` 中的原因字段。

    返回值:
        二元组 ``(消息文本, 日志级别)``。级别为 ``"info"`` / ``"warning"`` / ``"error"``。

    异常:
        无。
    """

    resolved_form_type = form_type or (filing_result.form_type if filing_result is not None else None) or "文件"
    if filing_result is None:
        return f"完成下载 {resolved_form_type}", "info"

    status = filing_result.status.strip().lower()
    reason_text = filing_result.reason_message or filing_result.skip_reason or filing_result.reason_code or reason
    if status == "skipped":
        if reason_text:
            return f"跳过下载 {resolved_form_type}: {reason_text}", "warning"
        return f"跳过下载 {resolved_form_type}", "warning"
    if status == "failed":
        if reason_text:
            return f"下载失败 {resolved_form_type}: {reason_text}", "error"
        return f"下载失败 {resolved_form_type}", "error"
    if filing_result.downloaded_files > 0:
        return f"完成下载 {resolved_form_type}（{filing_result.downloaded_files} 个文件）", "info"
    return f"完成下载 {resolved_form_type}", "info"


def _add_log_entry(task: DownloadTaskState, message: str, level: str = "info") -> None:
    """向下载任务追加日志。

    参数:
        task: 目标下载任务状态对象。
        message: 日志消息文本。
        level: 日志级别（``"info"`` / ``"warning"`` / ``"error"``），默认 ``"info"``。

    返回值:
        无。

    异常:
        无。
    """

    entry: LogEntry = {
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "level": level,
    }
    task.logs.append(entry)
