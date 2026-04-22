"""分析报告 Tab 页面。

展示已生成的分析报告，支持启动新的分析任务，
根据报告存在性和任务运行状态展示三种不同UI。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, cast

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx  # type: ignore

from dayu.services.protocols import HostAdminServiceProtocol, WriteServiceProtocol
from dayu.web.streamlit.components.sidebar import WatchlistItem
from dayu.web.streamlit.pages.report import report_export, report_host_sync, report_manifest, report_markdown_view

# 任务状态常量
_TASK_STATUS_PENDING = "pending"
_TASK_STATUS_RUNNING = "running"
_TASK_STATUS_COMPLETED = "completed"
_TASK_STATUS_FAILED = "failed"
_TASK_STATUS_CANCELLED = "cancelled"

# 默认模板路径
_DEFAULT_TEMPLATE_PATH = "定性分析模板.md"

# 报告文件名格式
_REPORT_FILE_NAME_FORMAT = "{ticker}_qual_report.md"
_SUMMARY_FILE_NAME = "run_summary.json"
_WRITE_MANIFEST_FILE_NAME = "manifest.json"

# 轮询间隔（秒）
_TASK_POLL_INTERVAL_SECONDS = 2.0
_RUN_DURATION_PRECISION = 2
_TASK_PROGRESS_INITIAL_PERCENT = 10.0
_TASK_PROGRESS_STREAMING_MAX_PERCENT = 95.0
_MANIFEST_CURSOR_READY_MARKER = "__manifest_ready__"
_MANIFEST_CURSOR_PARSE_ERROR_MARKER = "__manifest_parse_error__"
_MANIFEST_CURSOR_ARTIFACT_PREFIX = "__chapter_artifact__::"
_MANIFEST_CURSOR_FINAL_PREFIX = "__chapter_final__::"
_WRITE_PIPELINE_SERVICE_TYPE = "write_pipeline"
_CHAPTERS_DIR_NAME = "chapters"
_FINAL_CHAPTER_STATUSES = frozenset({"passed", "failed"})
_CHAPTER_FILE_PREFIX_PATTERN = re.compile(r"^(?P<index>\d+)_(?P<title>.+)$")
_GENERATION_MODE_FAST = "fast"
_GENERATION_MODE_DEEP = "deep"

# Markdown 标题解析
_MARKDOWN_HEADING_PATTERN = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_MARKDOWN_FENCE_PREFIXES = ("```", "~~~")

# 报告展示布局
_REPORT_LAYOUT_COLUMN_WIDTHS = [1, 3]
_TOC_INDENT_REM = 1.0
_REPORT_PANEL_MIN_HEIGHT_PX = 1200
_REPORT_PANEL_MAX_HEIGHT_PX = 6000
_REPORT_PANEL_BASE_HEIGHT_PX = 1200
_REPORT_PANEL_HEIGHT_PER_CONTENT_LINE_PX = 2
_REPORT_PANEL_HEIGHT_PER_HEADING_PX = 24
_REPORT_PANEL_HEIGHT_CALIBRATION_FACTOR = 0.6
_GUIDE_PRIMARY_HEADING_LEVEL = 1
_GUIDE_SECONDARY_HEADING_LEVEL = 2
_GUIDE_SECONDARY_BULLET_INDENT = "  "


@dataclass(frozen=True)
class MarkdownHeading:
    """Markdown 标题目录项。

    Attributes:
        line_index: 标题所在的原始行号（从 0 开始）。
        level: 标题层级，范围为 1 到 6。
        title: 标题展示文本。
        anchor: 目录与正文共享的锚点标识。
    """

    line_index: int
    level: int
    title: str
    anchor: str


@dataclass
class ReportState:
    """分析报告状态。

    Attributes:
        exists: 报告是否存在。
        summary: run_summary.json 解析内容。
        report_path: 报告文件路径。
        report_content: 报告内容缓存。
        modified_time: 报告文件修改时间ISO字符串。
    """

    exists: bool = False
    summary: dict[str, Any] | None = None
    report_path: Path | None = None
    report_content: str | None = None
    modified_time: str | None = None


@dataclass
class WriteTaskState:
    """Write 任务状态。

    Attributes:
        status: 任务状态（pending/running/completed/failed/cancelled）。
        session_id: 关联的Host会话ID。
        run_id: 关联的Host运行ID。
        started_at: 任务开始时间ISO字符串。
        completed_at: 任务完成时间ISO字符串。
        exit_code: 任务退出码；None表示未完成。
        message: 状态描述信息。
        progress: 进度百分比（0-100）。
        current_chapter: 当前正在处理的章节。
        output_dir: 任务输出目录，用于读取流式状态。
        expected_chapter_count: 预估章节总数。
        status_cursor: 章节状态游标，避免重复写日志。
        logs: 任务日志列表。
    """

    status: str = _TASK_STATUS_PENDING
    session_id: str | None = None
    run_id: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    exit_code: int | None = None
    message: str = "等待开始..."
    progress: float = 0.0
    current_chapter: str | None = None
    output_dir: str | None = None
    expected_chapter_count: int | None = None
    status_cursor: dict[str, str] = field(default_factory=dict)
    logs: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式以便存储在 session_state 中。"""
        return {
            "status": self.status,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "exit_code": self.exit_code,
            "message": self.message,
            "progress": self.progress,
            "current_chapter": self.current_chapter,
            "output_dir": self.output_dir,
            "expected_chapter_count": self.expected_chapter_count,
            "status_cursor": self.status_cursor,
            "logs": self.logs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WriteTaskState":
        """从字典恢复对象。"""
        return cls(
            status=data.get("status", _TASK_STATUS_PENDING),
            session_id=data.get("session_id"),
            run_id=data.get("run_id"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            exit_code=data.get("exit_code"),
            message=data.get("message", "等待开始..."),
            progress=data.get("progress", 0.0),
            current_chapter=data.get("current_chapter"),
            output_dir=data.get("output_dir"),
            expected_chapter_count=data.get("expected_chapter_count"),
            status_cursor=data.get("status_cursor", {}),
            logs=data.get("logs", []),
        )


@dataclass(frozen=True)
class ManifestChapterSnapshot:
    """manifest 中的单章节状态快照。"""

    title: str
    index: int
    status: str
    failure_reason: str


def _get_draft_dir(workspace_root: Path, ticker: str) -> Path:
    """获取指定股票的草稿目录路径。

    Args:
        workspace_root: 工作区根目录。
        ticker: 股票代码。

    Returns:
        草稿目录路径。
    """
    return workspace_root / "draft" / ticker


def _load_report_state(workspace_root: Path, ticker: str) -> ReportState:
    """加载指定股票的报告状态。

    Args:
        workspace_root: 工作区根目录。
        ticker: 股票代码。

    Returns:
        报告状态对象。
    """
    draft_dir = _get_draft_dir(workspace_root, ticker)
    summary_path = draft_dir / _SUMMARY_FILE_NAME
    report_path = draft_dir / _REPORT_FILE_NAME_FORMAT.format(ticker=ticker)

    state = ReportState()

    # 检查摘要文件
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                state.summary = json.load(f)
        except (json.JSONDecodeError, OSError):
            state.summary = None

    # 检查报告文件
    if report_path.exists():
        state.exists = True
        state.report_path = report_path
        try:
            mtime = report_path.stat().st_mtime
            state.modified_time = datetime.fromtimestamp(mtime).isoformat()
        except OSError:
            state.modified_time = None

    return state


def _load_report_content(report_path: Path) -> str | None:
    """加载报告内容。

    Args:
        report_path: 报告文件路径。

    Returns:
        报告内容字符串；读取失败时返回 None。
    """
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def _extract_markdown_headings(markdown_content: str) -> list[MarkdownHeading]:
    """从 Markdown 文本中提取标题目录。

    Args:
        markdown_content: 原始 Markdown 文本。

    Returns:
        目录项列表；未检测到标题时返回空列表。

    Raises:
        无。
    """

    headings = report_markdown_view.extract_markdown_headings(markdown_content)
    return [
        MarkdownHeading(
            line_index=heading.line_index,
            level=heading.level,
            title=heading.title,
            anchor=heading.anchor,
        )
        for heading in headings
    ]


def _is_markdown_fence_line(stripped_line: str) -> bool:
    """判断当前行是否为 Markdown 围栏代码块分隔符。

    Args:
        stripped_line: 去除首尾空白后的单行文本。

    Returns:
        `True` 表示当前行为围栏代码块起止符，否则返回 `False`。

    Raises:
        无。
    """

    return report_markdown_view.is_markdown_fence_line(stripped_line)


def _normalize_markdown_heading_title(raw_title: str) -> str:
    """规范化 Markdown 标题文本。

    Args:
        raw_title: 正则提取出的标题原文。

    Returns:
        去除尾随 `#` 与多余空白后的标题文本。

    Raises:
        无。
    """

    return report_markdown_view.normalize_markdown_heading_title(raw_title)


def _slugify_markdown_heading(title: str) -> str:
    """将标题文本转换为稳定锚点。

    Args:
        title: 标题展示文本。

    Returns:
        可用于 HTML `id` 与锚点链接的稳定字符串。

    Raises:
        无。
    """

    return report_markdown_view.slugify_markdown_heading(title)


def _build_report_toc_html(headings: list[MarkdownHeading]) -> str:
    """构建报告目录 HTML。

    Args:
        headings: Markdown 标题目录项列表。

    Returns:
        可直接传给 `st.markdown(..., unsafe_allow_html=True)` 的 HTML 字符串。

    Raises:
        无。
    """

    normalized_headings = [
        report_markdown_view.MarkdownHeading(
            line_index=heading.line_index,
            level=heading.level,
            title=heading.title,
            anchor=heading.anchor,
        )
        for heading in headings
    ]
    return report_markdown_view.build_report_toc_html(normalized_headings)


def _inject_heading_anchors(markdown_content: str, headings: list[MarkdownHeading]) -> str:
    """为 Markdown 标题注入锚点。

    Args:
        markdown_content: 原始 Markdown 文本。
        headings: 预先解析出的标题目录项列表。

    Returns:
        注入锚点后的 Markdown 文本。

    Raises:
        无。
    """

    normalized_headings = [
        report_markdown_view.MarkdownHeading(
            line_index=heading.line_index,
            level=heading.level,
            title=heading.title,
            anchor=heading.anchor,
        )
        for heading in headings
    ]
    return report_markdown_view.inject_heading_anchors(markdown_content, normalized_headings)


def _render_markdown_report(markdown_content: str) -> None:
    """按“目录 + 正文”布局渲染报告 Markdown。

    Args:
        markdown_content: 原始 Markdown 报告内容。

    Returns:
        无。

    Raises:
        无。
    """

    report_markdown_view.render_markdown_report(markdown_content)


def _get_report_panel_container_height_px(
    markdown_content: str,
    headings: list[MarkdownHeading],
) -> int:
    """动态计算报告双栏容器高度。

    Args:
        markdown_content: 原始 Markdown 报告内容。
        headings: 报告标题目录项列表。

    Returns:
        报告目录区与正文区共享的容器高度（像素）。

    Raises:
        无。
    """

    normalized_headings = [
        report_markdown_view.MarkdownHeading(
            line_index=heading.line_index,
            level=heading.level,
            title=heading.title,
            anchor=heading.anchor,
        )
        for heading in headings
    ]
    return report_markdown_view.get_report_panel_container_height_px(markdown_content, normalized_headings)


def _clamp_report_panel_height_px(estimated_height_px: int) -> int:
    """将动态估算高度约束在可读区间内。

    Args:
        estimated_height_px: 根据正文与标题估算得到的容器高度。

    Returns:
        限制在最小高度与最大高度之间的像素值。

    Raises:
        无。
    """

    return report_markdown_view.clamp_report_panel_height_px(estimated_height_px)


def _init_write_task_state() -> None:
    """初始化 Write 任务会话状态。"""
    if "active_write_tasks" not in st.session_state:
        st.session_state.active_write_tasks = {}
    if "write_task_settings" not in st.session_state:
        st.session_state.write_task_settings = {}


def _get_ticker_active_write_task(ticker: str) -> WriteTaskState | None:
    """获取指定股票的活跃 Write 任务。

    Args:
        ticker: 股票代码。

    Returns:
        活跃任务状态；没有活跃任务时返回 None。
    """
    _init_write_task_state()
    key = f"write_task_{ticker}"
    if key not in st.session_state.active_write_tasks:
        return None

    task_data = st.session_state.active_write_tasks[key]
    task = WriteTaskState.from_dict(task_data)

    return task


def _add_active_write_task(ticker: str) -> WriteTaskState:
    """添加新的活跃 Write 任务。

    Args:
        ticker: 股票代码。

    Returns:
        创建的任务状态对象。
    """
    _init_write_task_state()
    key = f"write_task_{ticker}"
    task = WriteTaskState(
        status=_TASK_STATUS_PENDING,
        started_at=datetime.now().isoformat(),
    )
    st.session_state.active_write_tasks[key] = task.to_dict()
    return task


def _update_write_task(ticker: str, **kwargs: Any) -> None:
    """更新 Write 任务状态。

    Args:
        ticker: 股票代码。
        **kwargs: 要更新的字段。
    """
    _init_write_task_state()
    key = f"write_task_{ticker}"
    if key in st.session_state.active_write_tasks:
        task_data = st.session_state.active_write_tasks[key]
        task_data.update(kwargs)
        st.session_state.active_write_tasks[key] = task_data


def _add_task_log(ticker: str, message: str, level: str = "info") -> None:
    """向任务日志追加一条记录。

    Args:
        ticker: 股票代码。
        message: 日志消息。
        level: 日志级别（info/warning/error）。
    """
    _init_write_task_state()
    key = f"write_task_{ticker}"
    if key in st.session_state.active_write_tasks:
        task_data = st.session_state.active_write_tasks[key]
        logs = task_data.get("logs", [])
        logs.append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "level": level,
        })
        # 限制日志数量
        if len(logs) > 200:
            logs = logs[-200:]
        task_data["logs"] = logs
        st.session_state.active_write_tasks[key] = task_data


def _estimate_template_chapter_count(template_path: Path) -> int | None:
    """估算模板中的章节数量。

    Args:
        template_path: 模板文件绝对路径。

    Returns:
        一级标题数量；无法读取或未识别到一级标题时返回 ``None``。

    Raises:
        无。
    """

    content = _load_report_content(template_path)
    if content is None:
        return None
    headings = _extract_markdown_headings(content)
    count = sum(1 for heading in headings if heading.level == _GUIDE_PRIMARY_HEADING_LEVEL)
    return count if count > 0 else None


def _parse_manifest_chapter_snapshots(manifest_path: Path) -> list[ManifestChapterSnapshot]:
    """解析 manifest 中的章节状态快照。

    Args:
        manifest_path: manifest 文件路径。

    Returns:
        已按章节序号排序的章节状态快照列表。

    Raises:
        OSError: 读取文件失败时抛出。
        json.JSONDecodeError: manifest 非法 JSON 时抛出。
    """

    snapshots = report_manifest.parse_manifest_chapter_snapshots(manifest_path)
    return [
        ManifestChapterSnapshot(
            title=snapshot.title,
            index=snapshot.index,
            status=snapshot.status,
            failure_reason=snapshot.failure_reason,
        )
        for snapshot in snapshots
    ]


def _refresh_task_status_from_manifest(ticker: str, task: WriteTaskState) -> None:
    """根据 manifest 增量刷新任务日志和进度。

    Args:
        ticker: 股票代码。
        task: 当前任务状态。

    Returns:
        无。

    Raises:
        无。
    """

    if task.status != _TASK_STATUS_RUNNING:
        return
    if task.output_dir is None:
        return

    manifest_path = Path(task.output_dir) / _WRITE_MANIFEST_FILE_NAME
    if not manifest_path.exists():
        return

    cursor = dict(task.status_cursor)
    try:
        snapshots = _parse_manifest_chapter_snapshots(manifest_path)
    except (OSError, json.JSONDecodeError) as exc:
        if cursor.get(_MANIFEST_CURSOR_PARSE_ERROR_MARKER) != "1":
            _add_task_log(ticker, f"读取任务状态清单失败: {exc}", level="warning")
            cursor[_MANIFEST_CURSOR_PARSE_ERROR_MARKER] = "1"
            _update_write_task(ticker, status_cursor=cursor)
        return

    if cursor.get(_MANIFEST_CURSOR_READY_MARKER) != "1":
        _add_task_log(ticker, "检测到任务状态清单，开始流式跟踪章节执行状态")
        cursor[_MANIFEST_CURSOR_READY_MARKER] = "1"

    completed_count = 0
    for snapshot in snapshots:
        if snapshot.status in _FINAL_CHAPTER_STATUSES:
            completed_count += 1
            final_cursor_key = f"{_MANIFEST_CURSOR_FINAL_PREFIX}{snapshot.title}"
            previous_status = cursor.get(final_cursor_key)
            if previous_status == snapshot.status:
                continue
            if snapshot.status == "passed":
                _add_task_log(ticker, f"章节完成: [{snapshot.index:02d}] {snapshot.title}")
            elif snapshot.status == "failed":
                reason = snapshot.failure_reason.strip()
                if reason:
                    _add_task_log(
                        ticker,
                        f"章节失败: [{snapshot.index:02d}] {snapshot.title}，原因: {reason}",
                        level="error",
                    )
                else:
                    _add_task_log(ticker, f"章节失败: [{snapshot.index:02d}] {snapshot.title}", level="error")
            else:
                _add_task_log(
                    ticker,
                    f"章节状态更新: [{snapshot.index:02d}] {snapshot.title} -> {snapshot.status}",
                )
            cursor[final_cursor_key] = snapshot.status
            continue

        artifact_cursor_key = f"{_MANIFEST_CURSOR_ARTIFACT_PREFIX}{snapshot.status}"
        if cursor.get(artifact_cursor_key) == "1":
            continue
        _add_task_log(
            ticker,
            f"章节产物生成: [{snapshot.index:02d}] {snapshot.title} -> {snapshot.status}",
        )
        cursor[artifact_cursor_key] = "1"

    if completed_count > 0:
        estimated_total = task.expected_chapter_count if task.expected_chapter_count is not None else len(snapshots)
        safe_total = max(estimated_total, completed_count, 1)
        streamed_progress = _TASK_PROGRESS_INITIAL_PERCENT + (
            completed_count / safe_total
        ) * (_TASK_PROGRESS_STREAMING_MAX_PERCENT - _TASK_PROGRESS_INITIAL_PERCENT)
        progress_value = min(_TASK_PROGRESS_STREAMING_MAX_PERCENT, streamed_progress)
        progress_value = max(task.progress, progress_value)
        _update_write_task(
            ticker,
            progress=progress_value,
            message=f"已完成章节 {completed_count}/{safe_total}，正在持续生成中...",
            status_cursor=cursor,
        )
        return

    _update_write_task(ticker, status_cursor=cursor)


def _collect_report_artifact_overview(workspace_root: Path, ticker: str) -> str:
    """收集报告产物概览信息并格式化为日志文本。

    Args:
        workspace_root: 工作区根目录。
        ticker: 股票代码。

    Returns:
        产物概览日志文本。

    Raises:
        无。
    """

    draft_dir = _get_draft_dir(workspace_root, ticker)
    report_path = draft_dir / _REPORT_FILE_NAME_FORMAT.format(ticker=ticker)
    summary_path = draft_dir / _SUMMARY_FILE_NAME

    report_exists = report_path.exists()
    report_size_bytes = report_path.stat().st_size if report_exists else 0
    summary_exists = summary_path.exists()

    chapter_count_text = "未知"
    failed_count_text = "未知"
    if summary_exists:
        try:
            with open(summary_path, "r", encoding="utf-8") as file:
                summary_data = json.load(file)
            chapter_count_value = summary_data.get("chapter_count")
            failed_count_value = summary_data.get("failed_count")
            chapter_count_text = str(chapter_count_value) if chapter_count_value is not None else "未知"
            failed_count_text = str(failed_count_value) if failed_count_value is not None else "未知"
        except (json.JSONDecodeError, OSError):
            chapter_count_text = "摘要读取失败"
            failed_count_text = "摘要读取失败"

    return (
        f"产物概览: draft_dir={draft_dir}, "
        f"report_exists={report_exists}, report_size_bytes={report_size_bytes}, "
        f"summary_exists={summary_exists}, chapter_count={chapter_count_text}, failed_count={failed_count_text}"
    )


def _remove_active_write_task(ticker: str) -> None:
    """移除活跃 Write 任务。

    Args:
        ticker: 股票代码。
    """
    _init_write_task_state()
    key = f"write_task_{ticker}"
    if key in st.session_state.active_write_tasks:
        del st.session_state.active_write_tasks[key]


def _format_time(time_str: str | None) -> str:
    """格式化时间字符串为可读格式。

    Args:
        time_str: ISO格式时间字符串。

    Returns:
        格式化后的时间字符串。
    """
    if not time_str:
        return "未知"
    try:
        dt = datetime.fromisoformat(time_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return time_str


def _get_host_admin_service_from_session() -> HostAdminServiceProtocol | None:
    """从 Streamlit 会话状态读取 Host 管理服务。

    Args:
        无。

    Returns:
        Host 管理服务实例；未就绪时返回 `None`。

    Raises:
        无。
    """


    candidate = st.session_state.get("host_admin_service")
    if candidate is None:
        return None
    return cast(HostAdminServiceProtocol, candidate)


def _bind_write_task_run_id_from_host(ticker: str, task: WriteTaskState) -> WriteTaskState:
    """尝试把运行中任务绑定到 Host run_id。

    Args:
        ticker: 股票代码。
        task: 当前任务状态。

    Returns:
        最新任务状态；若绑定失败返回原状态。

    Raises:
        无。
    """

    host_admin_service = _get_host_admin_service_from_session()
    if host_admin_service is None:
        return task
    active_write_tasks = st.session_state.get("active_write_tasks")
    mapping_tasks: dict[str, report_host_sync.ActiveWriteTaskRecord] | None = None
    if isinstance(active_write_tasks, dict):
        mapping_tasks = {
            key: value
            for key, value in active_write_tasks.items()
            if isinstance(key, str) and isinstance(value, dict)
        }
    bind_outcome = report_host_sync.bind_write_task_run_id_from_host(
        ticker=ticker,
        task=task,
        host_admin_service=host_admin_service,
        active_write_tasks=mapping_tasks,
        running_status=_TASK_STATUS_RUNNING,
        service_type=_WRITE_PIPELINE_SERVICE_TYPE,
    )
    if bind_outcome is None:
        return task
    _update_write_task(
        ticker,
        run_id=bind_outcome.run_id,
        session_id=bind_outcome.session_id,
    )
    _add_task_log(
        ticker,
        bind_outcome.log_message,
    )
    refreshed = _get_ticker_active_write_task(ticker)
    return refreshed if refreshed is not None else task


def _cancel_write_task_via_host(ticker: str, task: WriteTaskState) -> tuple[bool, str]:
    """通过 Host 管理服务请求取消写作任务。

    Args:
        ticker: 股票代码。
        task: 当前任务状态。

    Returns:
        二元组 `(是否成功发起取消请求, 反馈文案)`。

    Raises:
        无。
    """

    host_admin_service = _get_host_admin_service_from_session()
    if host_admin_service is None:
        return False, "Host 管理服务不可用，无法取消任务。"

    bound_task = _bind_write_task_run_id_from_host(ticker, task)
    cancel_outcome = report_host_sync.request_cancel_via_host(
        task=bound_task,
        host_admin_service=host_admin_service,
    )
    if not cancel_outcome.success:
        return False, cancel_outcome.message
    assert cancel_outcome.run_id is not None
    _add_task_log(ticker, f"已向 Host 发送取消请求: run_id={cancel_outcome.run_id}", level="warning")
    _update_write_task(
        ticker,
        message="已请求取消，等待任务收敛...",
    )
    return True, cancel_outcome.message


def _sync_write_task_status_from_host(ticker: str, task: WriteTaskState) -> WriteTaskState:
    """从 Host 拉取 run 状态并同步到任务状态。

    Args:
        ticker: 股票代码。
        task: 当前任务状态。

    Returns:
        同步后的任务状态。

    Raises:
        无。
    """

    host_admin_service = _get_host_admin_service_from_session()
    if host_admin_service is None:
        return task
    sync_outcome = report_host_sync.sync_write_task_status_from_host(
        task=task,
        host_admin_service=host_admin_service,
        running_states={"created", "queued", "running"},
        cancelled_status=_TASK_STATUS_CANCELLED,
        failed_status=_TASK_STATUS_FAILED,
        completed_status=_TASK_STATUS_COMPLETED,
    )
    if sync_outcome is None:
        return task
    _update_write_task(ticker, status_cursor=sync_outcome.status_cursor)
    if sync_outcome.message_override is not None:
        _update_write_task(ticker, message=sync_outcome.message_override)
    if sync_outcome.log_message is not None:
        _add_task_log(ticker, sync_outcome.log_message, level=sync_outcome.log_level)
    if sync_outcome.final_status is not None:
        _update_write_task(
            ticker,
            status=sync_outcome.final_status,
            exit_code=sync_outcome.final_exit_code,
            message=sync_outcome.final_message if sync_outcome.final_message is not None else task.message,
            completed_at=sync_outcome.completed_at,
            progress=sync_outcome.progress_override if sync_outcome.progress_override is not None else task.progress,
        )
    refreshed = _get_ticker_active_write_task(ticker)
    return refreshed if refreshed is not None else task


def _get_task_settings(ticker: str) -> dict[str, Any]:
    """获取指定股票的任务设置。

    Args:
        ticker: 股票代码。

    Returns:
        任务设置字典。
    """
    _init_write_task_state()
    key = f"settings_{ticker}"
    if key not in st.session_state.write_task_settings:
        # 默认设置
        st.session_state.write_task_settings[key] = {
            "template_path": _DEFAULT_TEMPLATE_PATH,
            "write_max_retries": 2,
            "resume": True,
            "fast": True,
            "force": False,
        }
    return st.session_state.write_task_settings[key]


def _generation_mode_label(mode: str) -> str:
    """返回生成模式展示文案。

    Args:
        mode: 生成模式标识。

    Returns:
        生成模式对应的中文标签。

    Raises:
        无。
    """

    if mode == _GENERATION_MODE_DEEP:
        return "深度生成（含审计/确认/修复）"
    return "快速生成（仅写作，速度更快）"


def _fast_to_generation_mode(fast: bool) -> str:
    """将 fast 标记映射为生成模式标识。

    Args:
        fast: `True` 为快速生成，`False` 为深度生成。

    Returns:
        生成模式标识。

    Raises:
        无。
    """

    if fast:
        return _GENERATION_MODE_FAST
    return _GENERATION_MODE_DEEP


def _generation_mode_to_fast(mode: str) -> bool:
    """将生成模式标识映射为 fast 标记。

    Args:
        mode: 生成模式标识。

    Returns:
        `True` 表示快速生成；`False` 表示深度生成。

    Raises:
        无。
    """

    return mode != _GENERATION_MODE_DEEP


def _update_task_settings(ticker: str, **kwargs: Any) -> None:
    """更新任务设置。

    Args:
        ticker: 股票代码。
        **kwargs: 要更新的设置字段。
    """
    _init_write_task_state()
    key = f"settings_{ticker}"
    settings = st.session_state.write_task_settings.get(key, {})
    settings.update(kwargs)
    st.session_state.write_task_settings[key] = settings


def _start_write_task(
    ticker: str,
    company_name: str,
    workspace_root: Path,
    write_service: WriteServiceProtocol | None,
    template_path: str,
    write_max_retries: int,
    resume: bool,
    fast: bool,
    force: bool,
) -> None:
    """在后台线程启动 Write 任务。

    Args:
        ticker: 股票代码。
        company_name: 公司名称。
        workspace_root: 工作区根目录。
        write_service: 写作服务实例。
        template_path: 模板文件路径。
        write_max_retries: 最大重试次数。
        resume: 是否启用断点恢复。
        fast: 是否仅执行写作（不进入audit/confirm/repair）。
        force: 是否强制放宽audit前置门禁。
    """
    if write_service is None:
        _add_task_log(ticker, "写作服务不可用", level="error")
        _update_write_task(ticker, status=_TASK_STATUS_FAILED, message="写作服务不可用")
        return

    # 解析模板路径
    template = Path(template_path).expanduser()
    if not template.is_absolute():
        template = (Path.cwd() / template).resolve()

    if not template.exists():
        _add_task_log(ticker, f"模板文件不存在: {template}", level="error")
        _update_write_task(ticker, status=_TASK_STATUS_FAILED, message=f"模板文件不存在: {template}")
        return

    # 设置输出目录
    output_dir = _get_draft_dir(workspace_root, ticker)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_chapter_count = _estimate_template_chapter_count(template)

    # 创建 WriteRunConfig
    from dayu.services.contracts import WriteRunConfig

    write_config = WriteRunConfig(
        ticker=ticker,
        company=company_name,
        template_path=str(template),
        output_dir=str(output_dir),
        write_max_retries=write_max_retries,
        web_provider="auto",
        resume=resume,
        fast=fast,
        force=force,
    )

    # 标记任务为运行中
    _update_write_task(
        ticker,
        status=_TASK_STATUS_RUNNING,
        message="正在启动分析任务...",
        output_dir=str(output_dir),
        expected_chapter_count=expected_chapter_count,
        status_cursor={},
    )
    _add_task_log(ticker, f"启动分析任务: ticker={ticker}, company={company_name}")
    _add_task_log(
        ticker,
        f"任务配置: template={template}, output_dir={output_dir}, write_max_retries={write_max_retries}, "
        f"resume={resume}, fast={fast}, force={force}",
    )

    def run_task() -> None:
        """在后台线程中运行任务。"""
        started_perf = time.perf_counter()
        try:
            from dayu.services.contracts import WriteRequest

            _add_task_log(ticker, "任务开始执行: 正在初始化写作流水线")
            _update_write_task(ticker, progress=_TASK_PROGRESS_INITIAL_PERCENT, message="初始化写作流水线...")

            request = WriteRequest(write_config=write_config)
            _add_task_log(ticker, "调用 write_service.run ...")
            exit_code = write_service.run(request)
            elapsed_seconds = round(time.perf_counter() - started_perf, _RUN_DURATION_PRECISION)
            _add_task_log(ticker, f"写作服务返回: exit_code={exit_code}, elapsed_seconds={elapsed_seconds}")

            if exit_code == 0:
                _add_task_log(ticker, "任务执行成功", level="info")
                _add_task_log(ticker, _collect_report_artifact_overview(workspace_root, ticker))
                _update_write_task(
                    ticker,
                    status=_TASK_STATUS_COMPLETED,
                    exit_code=exit_code,
                    progress=100.0,
                    message="分析报告生成完成",
                    completed_at=datetime.now().isoformat(),
                )
            elif exit_code == 130:  # 取消退出码
                _add_task_log(ticker, "任务已取消", level="warning")
                _update_write_task(
                    ticker,
                    status=_TASK_STATUS_CANCELLED,
                    exit_code=exit_code,
                    message="任务已取消",
                    completed_at=datetime.now().isoformat(),
                )
            else:
                _add_task_log(ticker, f"任务执行失败，退出码: {exit_code}", level="error")
                _update_write_task(
                    ticker,
                    status=_TASK_STATUS_FAILED,
                    exit_code=exit_code,
                    message=f"任务执行失败，退出码: {exit_code}",
                    completed_at=datetime.now().isoformat(),
                )
        except Exception as e:
            elapsed_seconds = round(time.perf_counter() - started_perf, _RUN_DURATION_PRECISION)
            _add_task_log(ticker, f"任务异常: {e}, elapsed_seconds={elapsed_seconds}", level="error")
            _update_write_task(
                ticker,
                status=_TASK_STATUS_FAILED,
                message=f"任务异常: {e}",
                completed_at=datetime.now().isoformat(),
            )

    # 启动后台线程
    thread = threading.Thread(target=run_task, daemon=True)
    add_script_run_ctx(thread)
    thread.start()


def _extract_template_guide_headings(template_path: str) -> list[MarkdownHeading]:
    """提取模板中的一级与二级目录标题。

    Args:
        template_path: 模板文件路径，支持相对路径和绝对路径。

    Returns:
        一级与二级标题列表；模板不存在或读取失败时返回空列表。

    Raises:
        无。
    """

    resolved_template_path = _resolve_template_path(template_path)
    template_content = _load_report_content(resolved_template_path)
    if template_content is None:
        return []

    headings = _extract_markdown_headings(template_content)
    guide_headings: list[MarkdownHeading] = []
    for heading in headings:
        if heading.level in (_GUIDE_PRIMARY_HEADING_LEVEL, _GUIDE_SECONDARY_HEADING_LEVEL):
            guide_headings.append(heading)
    return guide_headings


def _build_state1_guide_card_content(guide_headings: list[MarkdownHeading]) -> str:
    """构建状态1引导卡片内容。

    Args:
        guide_headings: 模板一级与二级目录标题列表。

    Returns:
        可直接渲染为 markdown 的卡片内容。

    Raises:
        无。
    """

    if not guide_headings:
        return """
**分析报告说明**

大禹Agent将基于《定性分析模板》生成报告，但当前未读取到模板目录。
请检查模板文件路径或模板内容后重试。
"""

    outlines: list[str] = []
    for heading in guide_headings:
        if heading.level == _GUIDE_PRIMARY_HEADING_LEVEL:
            outlines.append(f"- {heading.title}")
        elif heading.level == _GUIDE_SECONDARY_HEADING_LEVEL:
            outlines.append(f"{_GUIDE_SECONDARY_BULLET_INDENT}- {heading.title}")
    outline_text = "\n".join(outlines)
    return f"""
**分析报告说明**

大禹Agent将基于《定性分析模板》生成报告，内容包括：
{outline_text}

生成报告需要一定时间（约10-30分钟），请耐心等待。
"""


def _render_state1_no_report_no_task(selected_stock: WatchlistItem) -> None:
    """渲染状态1：无报告 + 无任务运行。

    引导用户从页头按钮启动分析任务。

    Args:
        selected_stock: 当前选中的自选股。
    """
    ticker = selected_stock.ticker

    st.markdown("---")
    st.warning("当前股票尚未生成分析报告，请点击“生成”按钮启动任务。")

    settings = _get_task_settings(ticker)
    template_path = st.text_input(
        "模板路径",
        value=settings.get("template_path", _DEFAULT_TEMPLATE_PATH),
        key=f"template_path_{ticker}",
    )
    if str(settings.get("template_path", _DEFAULT_TEMPLATE_PATH)) != template_path:
        _update_task_settings(ticker, template_path=template_path)


    default_mode = _fast_to_generation_mode(bool(settings.get("fast", True)))
    selected_mode = st.radio(
        "生成模式",
        options=[_GENERATION_MODE_FAST, _GENERATION_MODE_DEEP],
        index=0 if default_mode == _GENERATION_MODE_FAST else 1,
        format_func=_generation_mode_label,
        horizontal=True,
        key=f"state1_generation_mode_{ticker}",
    )
    selected_fast = _generation_mode_to_fast(selected_mode)
    if bool(settings.get("fast", True)) != selected_fast:
        _update_task_settings(ticker, fast=selected_fast)
    st.caption("快速生成仅写作，深度生成包含审计、确认、修复流程。")

    # 引导卡片
    guide_headings = _extract_template_guide_headings(template_path or _DEFAULT_TEMPLATE_PATH)
    st.info(_build_state1_guide_card_content(guide_headings))

    st.caption("任务参数使用已保存设置或默认配置。")


def _render_task_logs(logs: list[dict[str, str]]) -> str:
    """渲染任务日志为HTML。

    Args:
        logs: 日志列表。

    Returns:
        HTML字符串。
    """
    if not logs:
        return "<div style='color:#6B7280;'>暂无日志</div>"

    # 只显示最近100条
    recent_logs = logs[-100:]
    lines: list[str] = []

    for log in recent_logs:
        timestamp = log.get("timestamp", "")
        message = log.get("message", "")
        level = log.get("level", "info")

        # 格式化时间
        try:
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%H:%M:%S")
        except ValueError:
            time_str = timestamp[:8] if timestamp else "--:--:--"

        # 根据级别设置颜色
        if level == "error":
            color = "#DC2626"
            level_text = "ERROR"
        elif level == "warning":
            color = "#B45309"
            level_text = "WARN"
        else:
            color = "#2563EB"
            level_text = "INFO"

        line = (
            f'<div style="font-family:monospace; font-size:12px; line-height:1.5; margin-bottom:3px;">'
            f'<span style="color:#6B7280;">[{time_str}]</span> '
            f'<span style="color:{color}; font-weight:600;">{level_text}</span> '
            f'<span style="color:#111827;">{message}</span>'
            f'</div>'
        )
        lines.append(line)

    return "".join(lines)


def _render_state2_no_report_with_task(
    selected_stock: WatchlistItem,
) -> None:
    """渲染状态2：无报告 + 有任务运行。

    展示详细的任务状态。

    Args:
        selected_stock: 当前选中的自选股。
    """
    ticker = selected_stock.ticker
    task = _get_ticker_active_write_task(ticker)

    if task is None:
        # 任务可能刚结束，重新加载
        st.rerun()
        return

    _refresh_task_status_from_manifest(ticker, task)
    task = _get_ticker_active_write_task(ticker)
    if task is None:
        st.rerun()
        return
    task = _bind_write_task_run_id_from_host(ticker, task)
    task = _sync_write_task_status_from_host(ticker, task)

    st.markdown("---")
    st.markdown("### 📊 分析任务进行中")

    # 进度展示
    progress_text = f"{task.progress:.1f}% - {task.message}"
    st.progress(task.progress / 100.0, text=progress_text)

    # 任务详情
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"股票: {ticker}")
    with col2:
        st.caption(f"启动时间: {_format_time(task.started_at)}")
    with col3:
        status_map = {
            _TASK_STATUS_PENDING: "⏳ 等待中",
            _TASK_STATUS_RUNNING: "🔄 运行中",
            _TASK_STATUS_COMPLETED: "✅ 已完成",
            _TASK_STATUS_FAILED: "❌ 失败",
            _TASK_STATUS_CANCELLED: "🚫 已取消",
        }
        st.caption(f"状态: {status_map.get(task.status, task.status)}")

    # 日志区域
    st.markdown("#### 任务日志")
    logs_html = _render_task_logs(task.logs)
    st.markdown(
        f'<div style="border:1px solid #E5E7EB; border-radius:6px; padding:10px; '
        f'height:300px; overflow-y:auto; background:#FAFAFA;">'
        f'{logs_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 操作按钮
    st.markdown("---")
    if task.status == _TASK_STATUS_RUNNING:
        if st.button("🛑 取消任务", type="secondary", key=f"cancel_task_{ticker}"):
            cancelled, feedback = _cancel_write_task_via_host(ticker, task)
            if cancelled:
                st.info(feedback)
            else:
                _add_task_log(ticker, feedback, level="warning")
                st.warning(feedback)
    else:
        # 任务已完成或失败，提供关闭按钮
        if st.button("关闭任务状态", type="primary", key=f"close_task_{ticker}"):
            _remove_active_write_task(ticker)
            st.rerun()

    # 自动刷新（如果任务仍在运行）
    if task.status == _TASK_STATUS_RUNNING:
        time.sleep(_TASK_POLL_INTERVAL_SECONDS)
        st.rerun()


def _render_state3_has_report_no_task(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    write_service: WriteServiceProtocol | None,
) -> None:
    """渲染状态3：有报告 + 无任务运行。

    展示详细的分析报告。

    Args:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        write_service: 写作服务实例（用于重新生成）。
    """
    ticker = selected_stock.ticker
    company_name = selected_stock.company_name

    # 加载报告状态
    report_state = _load_report_state(workspace_root, ticker)

    # 重新生成设置区域（置于基本信息之前）
    if st.session_state.get(f"show_regenerate_settings_{ticker}", False):
        st.markdown("---")
        st.markdown("#### 生成设置")

        settings = _get_task_settings(ticker)

        template_path = st.text_input(
            "分析模板路径",
            value=settings.get("template_path", _DEFAULT_TEMPLATE_PATH),
            key=f"regen_template_{ticker}",
        )

        col3, col4 = st.columns(2)
        with col3:
            default_fast = bool(settings.get("fast", True))
            default_mode = _fast_to_generation_mode(default_fast)
            generation_mode = st.radio(
                "生成模式",
                options=[_GENERATION_MODE_FAST, _GENERATION_MODE_DEEP],
                index=0 if default_mode == _GENERATION_MODE_FAST else 1,
                format_func=_generation_mode_label,
                horizontal=True,
                key=f"regen_generation_mode_{ticker}",
            )
            fast = _generation_mode_to_fast(generation_mode)
            st.caption("默认使用快速生成；如需包含审计、确认、修复流程，请切换到深度生成。")
        with col4:
            default_resume = bool(settings.get("resume", True))
            overwrite_mode = st.checkbox(
                "覆盖模式",
                value=not default_resume,
                key=f"regen_resume_{ticker}",
            )
            resume = not overwrite_mode
            st.caption("覆盖模式默认关闭（不覆盖）；开启后将重跑并覆盖已完成章节。")
            

        # 确保template_path是字符串
        template_path_str = str(template_path) if template_path else _DEFAULT_TEMPLATE_PATH

        col5, col6 = st.columns(2)
        with col5:
            if st.button("🔄 重新生成", type="primary", use_container_width=True, key=f"confirm_regen_{ticker}"):
                _update_task_settings(
                    ticker,
                    template_path=template_path_str,
                    write_max_retries=int(settings.get("write_max_retries", 2)),
                    resume=resume,
                    fast=fast,
                )
                _add_active_write_task(ticker)
                _start_write_task(
                    ticker=ticker,
                    company_name=company_name,
                    workspace_root=workspace_root,
                    write_service=write_service,
                    template_path=template_path_str,
                    write_max_retries=int(settings.get("write_max_retries", 2)),
                    resume=resume,
                    fast=fast,
                    force=settings.get("force", False),
                )
                st.session_state[f"show_regenerate_settings_{ticker}"] = False
                st.rerun()
        with col6:
            if st.button("❌ 取消", type="secondary", use_container_width=True, key=f"cancel_regen_{ticker}"):
                st.session_state[f"show_regenerate_settings_{ticker}"] = False
                st.rerun()
        
    st.markdown("---")

    st.markdown("#### 基本信息")
    # 报告概览卡片
    summary = report_state.summary or {}
    chapter_count = summary.get("chapter_count", "未知")
    failed_count = summary.get("failed_count", 0)
    gen_time = _format_time(report_state.modified_time).split()[0] if report_state.modified_time else "未知"
    file_name = report_state.report_path.name if report_state.report_path else "未知"

    st.markdown(
        f"""
        | 章节数 | 失败章节 | 生成时间 | 文件名 |
        | :--- | :--- | :--- | :--- |
        | {chapter_count} | {failed_count if failed_count == 0 else f'**{failed_count}**'} | {gen_time} | `{file_name}` |
        """
    )

    # 如果有失败章节，显示警告
    failed_chapters = summary.get("failed_chapters", [])
    if failed_chapters:
        with st.expander(f"⚠️ 失败章节详情 ({len(failed_chapters)}个)", expanded=True):
            for fc in failed_chapters:
                title = fc.get("title", "未知章节")
                reason = fc.get("reason", "未知原因")
                retries = fc.get("retry_count", 0)
                st.warning(f"**{title}**: {reason} (重试{retries}次)")

    st.markdown("#### 报告内容")

    # 加载并显示报告内容
    if report_state.report_path and report_state.report_path.exists():
        content = _load_report_content(report_state.report_path)
        if content:
            _render_markdown_report(content)
        else:
            st.error("报告内容加载失败")
    else:
        st.error("报告文件不存在")

def _generate_report_file(report_path: Path, output_format: str) -> tuple[bytes, str, str]:
    """根据格式生成报告文件内容。

    Args:
        report_path: 原始 Markdown 报告文件路径。
        output_format: 输出格式，支持 "markdown", "html", "pdf"。

    Returns:
        三元组：(文件字节内容, 下载文件名, MIME 类型)。

    Raises:
        RuntimeError: 格式转换失败时抛出。
    """
    return report_export.generate_report_file(report_path, output_format)


def _ensure_pandoc_v3_or_newer() -> None:
    """校验 pandoc 版本是否满足 3.0+ 要求。

    Args:
        无。

    Returns:
        无。

    Raises:
        RuntimeError: 未安装 pandoc、版本解析失败或版本低于 3.0 时抛出。
    """
    report_export.ensure_pandoc_v3_or_newer()


def _convert_to_html(report_path: Path, ticker: str) -> tuple[bytes, str, str]:
    """将 Markdown 报告转换为 HTML 格式。

    Args:
        report_path: 原始 Markdown 报告文件路径。
        ticker: 股票代码，用于生成文件名。

    Returns:
        三元组：(HTML 文件字节内容, 下载文件名, MIME 类型)。

    Raises:
        RuntimeError: HTML 转换失败时抛出。
    """
    return report_export.convert_to_html(report_path, ticker)


def _convert_to_pdf(report_path: Path, ticker: str) -> tuple[bytes, str, str]:
    """将 Markdown 报告转换为 PDF 格式。

    Args:
        report_path: 原始 Markdown 报告文件路径。
        ticker: 股票代码，用于生成文件名。

    Returns:
        三元组：(PDF 文件字节内容, 下载文件名, MIME 类型)。

    Raises:
        RuntimeError: PDF 转换失败时抛出。
    """
    return report_export.convert_to_pdf(report_path, ticker)


def _get_report_primary_action_label(report_exists: bool) -> str:
    """获取报告主操作按钮文案。

    Args:
        report_exists: 当前股票是否已有报告文件。

    Returns:
        头部主操作按钮文案。

    Raises:
        无。
    """

    if report_exists:
        return "🔄 重新生成"
    return "🚀 生成"


def _resolve_template_path(template_path: str) -> Path:
    """解析模板文件路径。

    Args:
        template_path: 原始模板路径。

    Returns:
        绝对模板路径。

    Raises:
        无。
    """

    template = Path(template_path).expanduser()
    if not template.is_absolute():
        return (Path.cwd() / template).resolve()
    return template


def _start_report_generation_from_saved_settings(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    write_service: WriteServiceProtocol | None,
) -> None:
    """按已保存配置直接启动报告生成任务。

    Args:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        write_service: 写作服务实例。

    Returns:
        无。

    Raises:
        无。
    """

    ticker = selected_stock.ticker
    company_name = selected_stock.company_name
    if write_service is None:
        st.error("写作服务不可用，无法启动分析任务")
        return

    settings = _get_task_settings(ticker)
    template_path = str(settings.get("template_path", _DEFAULT_TEMPLATE_PATH))
    write_max_retries = int(settings.get("write_max_retries", 2))
    resume = bool(settings.get("resume", True))
    fast = bool(settings.get("fast", True))
    force = bool(settings.get("force", False))

    template = _resolve_template_path(template_path)
    if not template.exists():
        st.error(f"模板文件不存在，无法启动任务: {template}")
        return

    _add_active_write_task(ticker)
    _start_write_task(
        ticker=ticker,
        company_name=company_name,
        workspace_root=workspace_root,
        write_service=write_service,
        template_path=str(template),
        write_max_retries=write_max_retries,
        resume=resume,
        fast=fast,
        force=force,
    )
    st.rerun()


def _render_report_header_actions(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    write_service: WriteServiceProtocol | None,
    ticker: str,
    report_exists: bool,
    report_path: Path | None,
    has_active_task: bool,
) -> None:
    """渲染报告页头部操作按钮。

    Args:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        write_service: 写作服务实例。
        ticker: 股票代码。
        report_exists: 报告是否存在。
        report_path: 当前报告文件路径；不存在时下载按钮降级为禁用。
        has_active_task: 当前股票是否存在活跃任务。

    Returns:
        无。

    Raises:
        无。
    """

    regenerate_column, download_column = st.columns(2)
    primary_action_label = _get_report_primary_action_label(report_exists)
    primary_action_disabled = has_active_task
    download_action_disabled = has_active_task or not _is_report_download_available(report_path)

    with regenerate_column:
        if st.button(
            primary_action_label,
            type="secondary",
            use_container_width=True,
            disabled=primary_action_disabled,
            key=f"regenerate_report_{ticker}",
        ):
            if report_exists:
                st.session_state[f"show_regenerate_settings_{ticker}"] = True
                st.rerun()
            else:
                _start_report_generation_from_saved_settings(
                    selected_stock=selected_stock,
                    workspace_root=workspace_root,
                    write_service=write_service,
                )

    with download_column:
        if not download_action_disabled:
            assert report_path is not None
            # 抽屉式导出菜单
            with st.popover("💾 导出报告", use_container_width=True):
                # Markdown 格式
                try:
                    file_bytes, file_name, mime_type = _generate_report_file(report_path, "markdown")
                    st.download_button(
                        label="📄 Markdown",
                        data=file_bytes,
                        file_name=file_name,
                        mime=mime_type,
                        use_container_width=True,
                        key=f"download_report_{ticker}_markdown",
                    )
                except (OSError, RuntimeError) as e:
                    st.button(
                        "📄 Markdown",
                        disabled=True,
                        use_container_width=True,
                        key=f"download_markdown_disabled_{ticker}",
                        help=str(e),
                    )

                # HTML 格式
                try:
                    file_bytes, file_name, mime_type = _generate_report_file(report_path, "html")
                    st.download_button(
                        label="🌐 HTML",
                        data=file_bytes,
                        file_name=file_name,
                        mime=mime_type,
                        use_container_width=True,
                        key=f"download_report_{ticker}_html",
                    )
                except (OSError, RuntimeError) as e:
                    st.button(
                        "🌐 HTML",
                        disabled=True,
                        use_container_width=True,
                        key=f"download_html_disabled_{ticker}",
                        help=str(e),
                    )

                # PDF 格式
                try:
                    file_bytes, file_name, mime_type = _generate_report_file(report_path, "pdf")
                    st.download_button(
                        label="📑 PDF",
                        data=file_bytes,
                        file_name=file_name,
                        mime=mime_type,
                        use_container_width=True,
                        key=f"download_report_{ticker}_pdf",
                    )
                except (OSError, RuntimeError) as e:
                    st.button(
                        "📑 PDF",
                        disabled=True,
                        use_container_width=True,
                        key=f"download_pdf_disabled_{ticker}",
                        help=str(e),
                    )
        else:
            st.button(
                "💾 导出报告",
                disabled=True,
                use_container_width=True,
                key=f"download_report_disabled_{ticker}",
            )


def _is_report_download_available(report_path: Path | None) -> bool:
    """判断报告下载按钮是否可用。

    Args:
        report_path: 报告文件路径；允许为 `None`。

    Returns:
        `True` 表示报告文件存在且可尝试下载，否则返回 `False`。

    Raises:
        无。
    """

    return report_path is not None and report_path.exists()


def render_report_tab(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    write_service: WriteServiceProtocol | None,
) -> None:
    """渲染分析报告 Tab。

    根据报告存在性和任务运行状态，展示三种不同UI：
    1. 无报告 + 无任务：引导用户启动分析任务
    2. 无报告 + 有任务：展示详细任务状态
    3. 有报告 + 无任务：展示详细分析报告

    Args:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        write_service: 写作服务实例；为 None 时部分功能不可用。
    """
    ticker = selected_stock.ticker

    # 初始化状态
    _init_write_task_state()

    # 检查报告是否存在
    report_state = _load_report_state(workspace_root, ticker)

    # 检查是否有活跃任务
    active_task = _get_ticker_active_write_task(ticker)

    title_column, actions_column = st.columns([4, 1], gap="small", vertical_alignment="center")
    with title_column:
        st.subheader(f"{selected_stock.company_name} ({ticker}) 分析报告")
    with actions_column:
        _render_report_header_actions(
            selected_stock=selected_stock,
            workspace_root=workspace_root,
            write_service=write_service,
            ticker=ticker,
            report_exists=report_state.exists,
            report_path=report_state.report_path,
            has_active_task=active_task is not None,
        )

    # 状态路由
    if active_task is not None:
        # 有任务正在运行（无论报告是否存在，都优先显示任务状态）
        _render_state2_no_report_with_task(selected_stock)
    elif report_state.exists:
        # 有报告，无任务
        _render_state3_has_report_no_task(selected_stock, workspace_root, write_service)
    else:
        # 无报告，无任务
        _render_state1_no_report_no_task(selected_stock)
