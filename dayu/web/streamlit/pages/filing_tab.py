"""财报管理 Tab 页面。

展示已下载财报列表，提供下载新财报功能。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import streamlit as st

from dayu.contracts.fins import (
    DownloadFilingResultItem,
    DownloadProgressPayload,
    FinsEvent,
    FinsEventType,
    FinsProgressEventName,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins.storage import FsSourceDocumentRepository
from dayu.services.protocols import FinsServiceProtocol
from dayu.web.streamlit.components.sidebar import WatchlistItem

_DATAFRAME_ROW_HEIGHT_PX = 35
_DATAFRAME_HEADER_HEIGHT_PX = 38

# 下载任务状态常量
_DOWNLOAD_STATUS_PENDING = "pending"
_DOWNLOAD_STATUS_RUNNING = "running"
_DOWNLOAD_STATUS_COMPLETED = "completed"
_DOWNLOAD_STATUS_FAILED = "failed"
_STATUS_CONTAINER_LOG_HEIGHT_PX = 220
_STATUS_CONTAINER_MAX_LOG_ITEMS = 120


@dataclass
class DownloadTaskState:
    """下载任务状态。

    Attributes:
        session_id: 会话 ID。
        ticker: 股票代码。
        status: 任务状态（pending/running/completed/failed）。
        progress: 进度百分比（0-100）。
        current_form_type: 当前处理的表单类型。
        current_document_id: 当前处理的文档 ID。
        message: 状态描述信息。
        downloaded_count: 已下载文件数。
        downloaded_filing_count: 已下载财报数。
        total_count: 预计总文件数。
        errors: 错误信息列表。
        started_at: 开始时间。
        completed_at: 完成时间。
    """

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
    logs: list[dict[str, str]] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式以便存储在 session_state 中。"""
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
    def from_dict(cls, data: dict[str, Any]) -> "DownloadTaskState":
        """从字典恢复对象。"""
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


def _init_download_state() -> None:
    """初始化下载任务会话状态。

    在 session_state 中创建 active_downloads 字典存储活跃下载任务。
    """
    if "active_downloads" not in st.session_state:
        st.session_state.active_downloads = {}


def _add_active_download(session_id: str, ticker: str) -> DownloadTaskState:
    """添加新的活跃下载任务。

    Args:
        session_id: 会话 ID。
        ticker: 股票代码。

    Returns:
        创建的下载任务状态对象。
    """
    _init_download_state()
    task = DownloadTaskState(
        session_id=session_id,
        ticker=ticker,
        status=_DOWNLOAD_STATUS_RUNNING,
        started_at=datetime.now().isoformat(),
    )
    _add_log_entry(task, "下载任务已创建，等待事件流")
    st.session_state.active_downloads[session_id] = task.to_dict()
    return task


def _format_log_time(timestamp: str) -> str:
    """格式化日志时间为时分秒。

    Args:
        timestamp: ISO 格式时间字符串。

    Returns:
        时分秒字符串；无法解析时返回原始值。
    """
    if not timestamp:
        return ""
    try:
        return datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
    except ValueError:
        return timestamp


def _format_download_size(size_in_bytes: int | None) -> str | None:
    """格式化下载文件大小展示文本。

    Args:
        size_in_bytes: 文件大小（字节）；未知时为 None。

    Returns:
        用于界面展示的文件大小文本。未知大小时返回 None。

    Raises:
        无。
    """
    if size_in_bytes is None:
        return None
    return f"{size_in_bytes} 字节"


def _build_file_downloaded_message(filename: str, size_in_bytes: int | None) -> str:
    """构建文件下载完成提示文案。

    Args:
        filename: 文件名展示文本。
        size_in_bytes: 文件大小（字节）；未知时为 None。

    Returns:
        下载完成提示文案；大小未知时不展示大小信息。

    Raises:
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
    """构建 filing 完成事件的日志文案与日志级别。

    Args:
        form_type: 事件中的表单类型。
        filing_result: 事件中的 filing 结果详情。
        reason: 事件中的原因码或原因文本。

    Returns:
        二元组 `(message, level)`，level 为 info/warning/error。

    Raises:
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


def _build_scrollable_log_html(logs: list[dict[str, str]]) -> str:
    """构建可滚动日志区域的 HTML 内容。

    Args:
        logs: 下载任务日志列表。

    Returns:
        可直接用于 Streamlit Markdown 渲染的 HTML 字符串。
    """
    recent_logs = logs[-_STATUS_CONTAINER_MAX_LOG_ITEMS:]
    log_lines: list[str] = []
    for log in recent_logs:
        time_text = escape(_format_log_time(log.get("timestamp", "")))
        message_text = escape(log.get("message", ""))
        level = log.get("level", "info")
        if level == "error":
            level_text = "ERROR"
            level_color = "#DC2626"
        elif level == "warning":
            level_text = "WARN"
            level_color = "#B45309"
        else:
            level_text = "INFO"
            level_color = "#2563EB"

        line_html = (
            '<div style="font-family:monospace; font-size:12px; line-height:1.5; margin-bottom:4px;">'
            f'<span style="color:#6B7280;">[{time_text}]</span> '
            f'<span style="color:{level_color}; font-weight:600;">{level_text}</span> '
            f'<span style="color:#111827;">{message_text}</span>'
            "</div>"
        )
        log_lines.append(line_html)

    content_html = "".join(log_lines) if log_lines else "<div>暂无日志</div>"
    return (
        '<div style="border:1px solid #E5E7EB; border-radius:6px; padding:8px; '
        f'height:{_STATUS_CONTAINER_LOG_HEIGHT_PX}px; overflow-y:auto; background:#FAFAFA;">'
        f"{content_html}"
        "</div>"
    )


def _add_log_entry(task: DownloadTaskState, message: str, level: str = "info") -> None:
    """向下载任务追加一条日志。

    Args:
        task: 下载任务状态对象。
        message: 日志消息内容。
        level: 日志级别，可选 info/warning/error。

    Returns:
        无。

    Raises:
        无。
    """
    task.logs.append({
        "timestamp": datetime.now().isoformat(),
        "message": message,
        "level": level,
    })


def _update_download_progress(session_id: str, payload: DownloadProgressPayload) -> None:
    """更新下载任务进度。

    Args:
        session_id: 会话 ID。
        payload: 进度事件负载。
    """
    _init_download_state()
    if session_id not in st.session_state.active_downloads:
        return

    task_data = st.session_state.active_downloads[session_id]
    task = DownloadTaskState.from_dict(task_data)

    # 更新基本信息
    task.current_form_type = payload.form_type
    task.current_document_id = payload.document_id

    # 根据事件类型更新状态和进度
    event_type = payload.event_type
    if event_type == FinsProgressEventName.PIPELINE_STARTED:
        task.message = "开始下载任务..."
        task.status = _DOWNLOAD_STATUS_RUNNING
        _add_log_entry(task, f"开始下载任务: {payload.ticker}")
    elif event_type == FinsProgressEventName.COMPANY_RESOLVED:
        task.message = f"已解析公司信息: {payload.ticker}"
        _add_log_entry(task, task.message)
    elif event_type == FinsProgressEventName.FILING_STARTED:
        task.message = f"开始下载 {payload.form_type or '文件'}..."
        task.current_document_id = payload.document_id
        _add_log_entry(task, f"开始下载: {payload.form_type or '文件'}")
    elif event_type == FinsProgressEventName.FILE_DOWNLOADED:
        task.downloaded_count += 1
        task.message = _build_file_downloaded_message(payload.name or "文件", payload.size)
        _add_log_entry(task, task.message)
        # 估算进度（如果已知总数）
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
        # 如果有 file_count 信息，更新总数
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

    # 保存更新后的状态
    st.session_state.active_downloads[session_id] = task.to_dict()


def _mark_download_completed(session_id: str, success: bool = True, message: str = "") -> None:
    """标记下载任务为完成状态。

    Args:
        session_id: 会话 ID。
        success: 是否成功完成。
        message: 完成消息。
    """
    _init_download_state()
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


def _remove_active_download(session_id: str) -> None:
    """移除活跃下载任务。

    Args:
        session_id: 会话 ID。
    """
    _init_download_state()
    if session_id in st.session_state.active_downloads:
        del st.session_state.active_downloads[session_id]


def _get_ticker_active_download(ticker: str) -> DownloadTaskState | None:
    """获取指定股票的活跃下载任务。

    Args:
        ticker: 股票代码。

    Returns:
        活跃下载任务状态；没有活跃任务时返回 None。
    """
    _init_download_state()
    for session_id, task_data in st.session_state.active_downloads.items():
        task = DownloadTaskState.from_dict(task_data)
        if task.ticker == ticker and task.status in (_DOWNLOAD_STATUS_PENDING, _DOWNLOAD_STATUS_RUNNING):
            return task
    return None


def _render_download_progress_area(ticker: str) -> None:
    """渲染下载进度展示区域。

    Args:
        ticker: 当前股票代码。
    """
    task = _get_ticker_active_download(ticker)
    if not task:
        return

    # 创建进度展示容器
    with st.container():
        st.markdown("---")
        st.markdown("**📥 正在下载财报**")

        # 显示进度条
        progress_text = f"{task.progress:.1f}% - {task.message}"
        st.progress(task.progress / 100.0, text=progress_text)

        # 显示详细信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption(f"股票代码: {task.ticker}")
        with col2:
            st.caption(f"已下载: {task.downloaded_count}")
        with col3:
            status_text = "运行中" if task.status == _DOWNLOAD_STATUS_RUNNING else task.status
            st.caption(f"状态: {status_text}")

        # 显示错误信息（如果有）
        if task.errors:
            with st.expander(f"⚠️ 错误信息 ({len(task.errors)} 条)"):
                for error in task.errors:
                    st.error(error)

        

        st.markdown("---")

        # 如果任务已完成，显示成功提示并提供关闭任务按钮
        if task.status == _DOWNLOAD_STATUS_COMPLETED:
            st.success("✅ 下载完成！财报列表已更新。")
            if st.button("关闭任务", key=f"close_completed_download_{task.session_id}"):
                _remove_active_download(task.session_id)
                st.rerun()
        elif task.status == _DOWNLOAD_STATUS_FAILED:
            st.error("❌ 下载失败，请检查错误信息。")
            if st.button("清除任务", key=f"clear_failed_download_{task.session_id}"):
                _remove_active_download(task.session_id)
                st.rerun()


def render_filing_tab(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    fins_service: FinsServiceProtocol | None,
) -> None:
    """渲染财报管理 Tab。

    Args:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        fins_service: 财报服务实例；为 None 时部分功能不可用。
    """

    # 初始化下载状态
    _init_download_state()
    _init_download_settings_state(selected_stock)

    title_column, actions_column = st.columns([4, 1], gap="small", vertical_alignment="center")
    with title_column:
        st.subheader(f"{selected_stock.company_name} ({selected_stock.ticker}) - 财报管理")
    with actions_column:
        _render_filing_header_actions(selected_stock)

    # 展示活跃下载任务进度（如果有）
    _render_download_progress_area(selected_stock.ticker)

    # 下载设置区域（展开/收起）
    if _should_show_download_settings_for_ticker(selected_stock.ticker):
        st.markdown("---")
        _render_download_settings(selected_stock, fins_service)
        st.markdown("---")

    # 获取已下载财报列表
    filings = _get_filing_list(workspace_root, selected_stock.ticker)

    st.markdown("---")

    # 展示财报列表
    if filings:
        _render_filing_table(filings)
    else:
        st.info("暂无财报，请点击「下载财报」按钮获取")


def _get_filing_list(workspace_root: Path, ticker: str) -> list[dict]:
    """获取指定股票的已下载财报列表。

    Args:
        workspace_root: 工作区根目录。
        ticker: 股票代码。

    Returns:
        财报文件信息列表。
    """

    try:
        # 使用 FsSourceDocumentRepository 获取文档列表
        source_repo = FsSourceDocumentRepository(
            workspace_root,
            create_directories=False,
        )

        # 获取 filing 文档 ID 列表
        document_ids = source_repo.list_source_document_ids(ticker, SourceKind.FILING)

        filings = []
        for doc_id in document_ids:
            try:
                # 读取文档元数据
                meta = source_repo.get_source_meta(ticker, doc_id, SourceKind.FILING)
                file_name, file_path = _resolve_primary_file_display(
                    source_repo=source_repo,
                    workspace_root=workspace_root,
                    ticker=ticker,
                    document_id=doc_id,
                )

                # 提取关键信息
                filing_info = {
                    "document_id": doc_id,
                    "file_name": file_name,
                    "file_path": file_path,
                    "form_type": meta.get("form_type", "未知"),
                    "filing_date": meta.get("filing_date", "未知"),
                    "report_date": meta.get("report_date", "未知"),
                    "fiscal_year": meta.get("fiscal_year", "未知"),
                    "fiscal_period": meta.get("fiscal_period", "未知"),
                    "status": "可用" if not meta.get("is_deleted", False) else "已删除",
                }
                filings.append(filing_info)
            except Exception:
                # 如果读取某个文档失败，跳过
                continue

        # 按申报日期排序（最新的在前）
        filings.sort(key=lambda x: x.get("filing_date", ""), reverse=True)
        return filings

    except Exception as e:
        st.error(f"读取财报列表失败: {e}")
        return []


def _resolve_primary_file_display(
    source_repo: FsSourceDocumentRepository,
    workspace_root: Path,
    ticker: str,
    document_id: str,
) -> tuple[str, str]:
    """解析源文档主文件的展示名称、路径。

    Args:
        source_repo: 源文档仓储实例。
        workspace_root: 工作区根目录。
        ticker: 股票代码。
        document_id: 文档 ID。

    Returns:
        二元组 `(文件名, 文件路径展示值)`。

    Raises:
        无：解析失败时返回“未知”占位，不向调用方抛出异常。
    """

    try:
        primary_source = source_repo.get_primary_source(ticker, document_id, SourceKind.FILING)
        materialized_path = primary_source.materialize().resolve()
        filename = materialized_path.name or "未知"
        try:
            relative_path = materialized_path.relative_to(workspace_root.resolve())
            file_path = str(relative_path)
        except ValueError:
            file_path = str(materialized_path)
        return filename, file_path
    except Exception:
        return "未知", "未知"


def _render_filing_table(filings: list[dict]) -> None:
    """渲染财报列表表格。

    Args:
        filings: 财报文件信息列表。
    """

    # 准备表格数据
    df_data = []
    for f in filings:
        df_data.append({
            "文件名称": f.get("file_name", "未知"),
            "文件路径": f.get("file_path", "未知"),
            "表单类型": f.get("form_type", "未知"),
            "申报日期": f.get("filing_date", "未知"),
            "报告日期": f.get("report_date", "未知"),
            "财年": f.get("fiscal_year", "未知"),
            "财期": f.get("fiscal_period", "未知"),
            "状态": f.get("status", "未知"),
        })

    if df_data:
        df = pd.DataFrame(df_data)
        table_height = _calculate_dataframe_height(len(df_data))
        st.dataframe(
            df,
            width="stretch",
            hide_index=True,
            height=table_height,
            on_select="ignore",
            selection_mode="single-row",
            key="filing_table",
            column_config={
                "文件名称": st.column_config.TextColumn("文件名称", width="medium"),
                "文件路径": st.column_config.TextColumn("文件路径", width="large"),
                "表单类型": st.column_config.TextColumn("表单类型", width="small"),
                "申报日期": st.column_config.TextColumn("申报日期", width="small"),
                "报告日期": st.column_config.TextColumn("报告日期", width="small"),
                "财年": st.column_config.TextColumn("财年", width="small"),
                "财期": st.column_config.TextColumn("财期", width="small"),
                "状态": st.column_config.TextColumn("状态", width="small"),
            },
        )
        
    else:
        st.info("暂无有效财报数据")

def _calculate_dataframe_height(visible_rows: int) -> int:
    """按可见行数计算 DataFrame 组件高度（像素）。

    Args:
        visible_rows: 目标可见数据行数。

    Returns:
        DataFrame 组件高度（像素）。

    Raises:
        ValueError: 当 visible_rows 小于 1 时抛出。
    """

    if visible_rows < 1:
        raise ValueError("visible_rows 必须大于等于 1")
    return _DATAFRAME_HEADER_HEIGHT_PX + visible_rows * _DATAFRAME_ROW_HEIGHT_PX


def _init_download_settings_state(selected_stock: WatchlistItem) -> None:
    """初始化下载设置会话状态。

    Args:
        selected_stock: 当前选中的自选股。
    """
    if "show_download_settings" not in st.session_state:
        st.session_state.show_download_settings = False
    if "download_settings_ticker" not in st.session_state:
        st.session_state.download_settings_ticker = selected_stock.ticker


def _render_filing_header_actions(selected_stock: WatchlistItem) -> None:
    """渲染财报页头部操作按钮。

    Args:
        selected_stock: 当前选中的自选股。

    Returns:
        无。

    Raises:
        无。
    """

    toggle_column, spacer_column = st.columns([1, 1])
    with spacer_column:
        button_text = _get_download_header_button_text(selected_stock.ticker)
        if st.button(button_text, width="stretch", type="secondary", key=f"toggle_download_settings_{selected_stock.ticker}"):
            _toggle_download_settings(selected_stock)
            st.rerun()


def _should_show_download_settings_for_ticker(ticker: str) -> bool:
    """判断当前股票是否应展示下载设置区域。

    Args:
        ticker: 股票代码。

    Returns:
        `True` 表示当前页面应展示该股票的下载设置区域，否则返回 `False`。

    Raises:
        无。
    """

    return st.session_state.get("show_download_settings", False) and st.session_state.get("download_settings_ticker") == ticker


def _get_download_header_button_text(ticker: str) -> str:
    """返回财报页头部下载按钮文案。

    Args:
        ticker: 股票代码。

    Returns:
        下载设置未展开时返回“下载财报”；当前股票的下载设置已展开时返回“取消下载”。

    Raises:
        无。
    """

    if _should_show_download_settings_for_ticker(ticker):
        return "❌ 关闭下载"
    return "📥 下载财报"


def _toggle_download_settings(selected_stock: WatchlistItem) -> None:
    """切换下载设置区域的显示/隐藏。

    Args:
        selected_stock: 当前选中的自选股。
    """
    _init_download_settings_state(selected_stock)
    # 如果当前是隐藏状态，或者切换到不同股票时，显示设置区域
    if (
        not st.session_state.show_download_settings
        or st.session_state.download_settings_ticker != selected_stock.ticker
    ):
        st.session_state.show_download_settings = True
        st.session_state.download_settings_ticker = selected_stock.ticker
    else:
        st.session_state.show_download_settings = False


def _render_download_settings(
    selected_stock: WatchlistItem,
    fins_service: FinsServiceProtocol | None,
) -> None:
    """在当前页面渲染下载任务设置区域。

    使用 expander 展开/收起下载设置表单，提交后实时展示下载进度。

    Args:
        selected_stock: 当前选中的自选股。
        fins_service: 财报服务实例；为 None 时仅显示提示信息。
    """
    _init_download_settings_state(selected_stock)
    ticker = selected_stock.ticker

    # 检查是否有正在进行的下载任务
    existing_task = _get_ticker_active_download(ticker)
    if existing_task:
        # 如果有正在进行的任务，显示提示，不提供新的下载设置
        st.info(f"📥 已有正在进行的下载任务（会话 ID: {existing_task.session_id}），请等待完成")
        return

    if fins_service is None:
        st.warning("财报服务不可用，无法进行下载操作")
        return

    with st.container():
        st.markdown("**📥 下载财报设置**")

        form_types = st.multiselect(
            "选择要下载的财报表单类型",
            options=["10-K", "10-Q", "8-K", "DEF 14A", "其他"],
            default=["10-K", "10-Q"],
            help="选择需要下载的 SEC 表单类型",
            key=f"download_form_types_{ticker}",
        )

        import datetime
        today = datetime.date.today()
        three_years_ago = today.replace(year=today.year - 3)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=three_years_ago,
                help="可选，默认三年前，留空表示不限制开始日期",
                key=f"download_start_date_{ticker}",
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=today,
                help="可选，默认今天，留空表示不限制结束日期",
                key=f"download_end_date_{ticker}",
            )
       

        overwrite = st.checkbox(
            "覆盖已有文件",
            value=False,
            help="如果文件已存在，是否重新下载",
            key=f"download_overwrite_{ticker}",
        )

        if st.button("开始下载", width="stretch", type="primary", key=f"download_start_btn_{ticker}"):
                if not form_types:
                    st.error("请至少选择一种表单类型")
                else:
                    start_date_str = start_date.isoformat() if start_date else None
                    end_date_str = end_date.isoformat() if end_date else None

                    submission = None
                    try:
                        from dayu.contracts.fins import (
                            DownloadCommandPayload,
                            FinsCommand,
                            FinsCommandName,
                            FinsEventType,
                        )
                        from dayu.services.contracts import FinsSubmitRequest

                        submission = fins_service.submit(
                            FinsSubmitRequest(
                                command=FinsCommand(
                                    name=FinsCommandName.DOWNLOAD,
                                    payload=DownloadCommandPayload(
                                        ticker=ticker,
                                        form_type=tuple(form_types),
                                        start_date=start_date_str,
                                        end_date=end_date_str,
                                        overwrite=overwrite,
                                    ),
                                    stream=True,
                                ),
                            ),
                        )

                        # 添加到活跃下载任务
                        task = _add_active_download(submission.session_id, ticker)

                        # 创建状态容器用于实时更新
                        status_container = st.status("开始下载财报...", expanded=True)
                        progress_bar = status_container.progress(0.0)
                        status_logs_placeholder = status_container.empty()

                        task_data = st.session_state.active_downloads.get(submission.session_id, {})
                        current_task = DownloadTaskState.from_dict(task_data)
                        status_logs_placeholder.markdown(
                            _build_scrollable_log_html(current_task.logs),
                            unsafe_allow_html=True,
                        )

                        # 消费流式事件
                        async def consume_stream():
                            from collections.abc import AsyncIterator

                            execution = submission.execution
                            if not isinstance(execution, AsyncIterator):
                                # 同步结果，直接标记完成
                                _mark_download_completed(
                                    submission.session_id,
                                    success=True,
                                    message="下载完成（同步模式）",
                                )
                                return

                            async for event in execution:
                                if not isinstance(event, FinsEvent):
                                    continue

                                if event.type == FinsEventType.PROGRESS:
                                    payload = event.payload
                                    if isinstance(payload, DownloadProgressPayload):
                                        # 更新任务状态
                                        _update_download_progress(submission.session_id, payload)
                                        # 获取最新状态
                                        task_data = st.session_state.active_downloads.get(
                                            submission.session_id, {}
                                        )
                                        current_task = DownloadTaskState.from_dict(task_data)

                                        # 更新 UI
                                        progress_bar.progress(current_task.progress / 100.0)
                                        status_logs_placeholder.markdown(
                                            _build_scrollable_log_html(current_task.logs),
                                            unsafe_allow_html=True,
                                        )

                                elif event.type == FinsEventType.RESULT:
                                    # 最终结果，任务完成
                                    _mark_download_completed(
                                        submission.session_id, success=True, message="下载完成"
                                    )
                                    break

                        # 运行异步消费
                        asyncio.run(consume_stream())

                        # 获取最终状态
                        final_task_data = st.session_state.active_downloads.get(
                            submission.session_id, {}
                        )
                        final_task = DownloadTaskState.from_dict(final_task_data)
                        status_logs_placeholder.markdown(
                            _build_scrollable_log_html(final_task.logs),
                            unsafe_allow_html=True,
                        )

                        # 更新最终状态
                        if final_task.status == _DOWNLOAD_STATUS_COMPLETED:
                            status_container.update(
                                label="✅ 下载完成！", state="complete", expanded=True
                            )
                            st.success(f"已成功下载 {final_task.downloaded_filing_count} 个财报，{final_task.downloaded_count} 个文件")

                            # 显示错误汇总（如果有）
                            if final_task.errors:
                                st.warning(f"下载过程中出现 {len(final_task.errors)} 个错误")

                        else:
                            status_container.update(label="❌ 下载失败", state="error", expanded=True)
                            if final_task.errors:
                                for error in final_task.errors:
                                    st.error(error)

                    except Exception as e:
                        st.error(f"下载任务失败: {e}")
                        if submission is not None:
                            _mark_download_completed(
                                submission.session_id, success=False, message=str(e)
                            )
