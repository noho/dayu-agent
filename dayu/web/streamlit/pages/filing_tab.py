"""Streamlit 财报管理 Tab 页面。

展示已下载财报列表，提供下载新财报功能。
"""

from __future__ import annotations

import datetime
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import TypedDict, cast

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from dayu.contracts.fins import (
    DownloadCommandPayload,
    DownloadProgressPayload,
    FinsCommand,
    FinsCommandName,
)
from dayu.fins.score_sec_ci import FORM_PROFILES
from dayu.services.contracts import FinsSubmission, FinsSubmitRequest
from dayu.services.protocols import FinsServiceProtocol
from dayu.web.streamlit.pages.filing.download_progress import (
    DownloadQueueEvent,
    DownloadStatus,
    DownloadTaskState,
    LogEntry,
    run_download_stream_worker,
    apply_download_completion,
    apply_download_progress,
    create_download_task,
)
from dayu.web.streamlit.components.watchlist import WatchlistItem

_DATAFRAME_ROW_HEIGHT_PX = 35
_DATAFRAME_HEADER_HEIGHT_PX = 38
_DOWNLOAD_DEFAULT_FORM_TYPES: tuple[str, ...] = ("10-K", "10-Q")
_DOWNLOAD_DEFAULT_LOOKBACK_YEARS = 3
_DOWNLOAD_RUNTIME_STATE_KEY = "download_runtime_handles"
_DOWNLOAD_EVENT_BATCH_LIMIT = 128
_DOWNLOAD_POLL_INTERVAL_SECONDS = 1.0
_STATUS_CONTAINER_MAX_LOG_ITEMS = 120
_FragmentDecoratorFactory = Callable[
    ...,
    Callable[[Callable[[], None]], Callable[[], None]],
]
_DOWNLOAD_LOG_LEVEL_LABELS: dict[str, str] = {
    "error": "ERROR",
    "warning": "WARN",
    "info": "INFO",
}


class _FilingInfo(TypedDict):
    """单条财报文件展示信息。"""

    document_id: str
    file_name: str
    file_path: str
    form_type: str
    filing_date: str
    report_date: str
    fiscal_year: str
    fiscal_period: str
    status: str


@dataclass(frozen=True)
class _DownloadFormValues:
    """下载表单值对象。"""

    form_types: tuple[str, ...]
    start_date: datetime.date | None
    end_date: datetime.date | None
    overwrite: bool


class _DownloadRuntimeState(TypedDict):
    """下载任务运行时句柄。"""

    worker: threading.Thread
    event_queue: Queue[DownloadQueueEvent]
    done: bool


def init_download_state() -> None:
    """初始化下载任务会话状态。"""

    if "active_downloads" not in st.session_state:
        st.session_state.active_downloads = {}


def _clear_ticker_download_history(ticker: str) -> None:
    """清理指定 ticker 的历史下载任务状态。"""

    init_download_state()
    active_downloads = cast(dict[str, DownloadTaskState], st.session_state.active_downloads)
    stale_session_ids: list[str] = []
    for existing_session_id, task in active_downloads.items():
        if task.ticker == ticker:
            stale_session_ids.append(existing_session_id)

    for stale_session_id in stale_session_ids:
        del st.session_state.active_downloads[stale_session_id]


def add_active_download(session_id: str, ticker: str) -> DownloadTaskState:
    """添加新的活跃下载任务。"""

    init_download_state()
    _clear_ticker_download_history(ticker)
    task = create_download_task(session_id=session_id, ticker=ticker)
    st.session_state.active_downloads[session_id] = task
    return task


def _load_download_task(session_id: str) -> DownloadTaskState | None:
    """按会话 ID 读取下载任务状态。"""

    init_download_state()
    task = st.session_state.active_downloads.get(session_id)
    if not isinstance(task, DownloadTaskState):
        return None
    return task


def _save_download_task(task: DownloadTaskState) -> None:
    """保存下载任务状态到会话存储。"""

    init_download_state()
    st.session_state.active_downloads[task.session_id] = task


def update_download_progress(session_id: str, payload: DownloadProgressPayload) -> None:
    """按进度事件更新下载任务状态。"""

    task = _load_download_task(session_id)
    if task is None:
        return
    apply_download_progress(task, payload)
    _save_download_task(task)


def mark_download_completed(session_id: str, success: bool = True, message: str = "") -> None:
    """标记下载任务终态。"""

    task = _load_download_task(session_id)
    if task is None:
        return
    apply_download_completion(task, success=success, message=message)
    _save_download_task(task)


def _format_log_time(timestamp: str) -> str:
    """格式化日志时间为时分秒。"""

    if not timestamp:
        return ""
    try:
        return datetime.datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
    except ValueError:
        return timestamp


def _build_download_log_lines(logs: list[LogEntry]) -> list[str]:
    """构建下载日志的文本行。

    参数:
        logs: 下载任务日志列表。

    返回值:
        仅包含最近日志的文本行列表，每一行均已按 ``[时间] 级别 消息`` 格式化。

    异常:
        无。
    """

    recent_logs = logs[-_STATUS_CONTAINER_MAX_LOG_ITEMS:]
    formatted_lines: list[str] = []
    for log in recent_logs:
        time_text = _format_log_time(log.get("timestamp", ""))
        message_text = log.get("message", "")
        level = log.get("level", "info")
        level_text = _DOWNLOAD_LOG_LEVEL_LABELS.get(level, _DOWNLOAD_LOG_LEVEL_LABELS["info"])
        if time_text:
            formatted_lines.append(f"[{time_text}] {level_text} {message_text}")
        else:
            formatted_lines.append(f"{level_text} {message_text}")
    return formatted_lines


def _render_download_logs(status_container: DeltaGenerator, logs: list[LogEntry]) -> None:
    """使用 Streamlit 原生组件渲染下载日志。

    参数:
        status_container: 下载状态卡片容器。
        logs: 下载任务日志列表。

    返回值:
        无。

    异常:
        无。
    """

    status_container.caption("最近日志")
    log_lines = _build_download_log_lines(logs)
    if not log_lines:
        status_container.caption("暂无日志")
        return
    status_container.code("\n".join(log_lines))


def _get_download_runtime_state() -> dict[str, _DownloadRuntimeState]:
    """获取下载运行时状态字典。"""

    runtime_state = st.session_state.get(_DOWNLOAD_RUNTIME_STATE_KEY)
    if isinstance(runtime_state, dict):
        return cast(dict[str, _DownloadRuntimeState], runtime_state)
    initialized_state: dict[str, _DownloadRuntimeState] = {}
    st.session_state[_DOWNLOAD_RUNTIME_STATE_KEY] = initialized_state
    return initialized_state


def start_download_worker(submission: FinsSubmission) -> None:
    """启动下载后台线程并登记运行时句柄。"""

    runtime_state = _get_download_runtime_state()
    event_queue: Queue[DownloadQueueEvent] = Queue()
    worker = threading.Thread(
        target=run_download_stream_worker,
        args=(submission, event_queue),
        daemon=True,
    )
    runtime_state[submission.session_id] = {
        "worker": worker,
        "event_queue": event_queue,
        "done": False,
    }
    worker.start()


def _dispatch_download_runtime_event(session_id: str, event: DownloadQueueEvent) -> None:
    """把后台队列事件映射为前端会话状态更新。"""

    if event.kind == "progress" and event.payload is not None:
        update_download_progress(session_id, event.payload)
        return
    if event.kind == "result":
        mark_download_completed(session_id, success=True, message=event.message or "下载完成")
        return
    if event.kind == "error":
        mark_download_completed(session_id, success=False, message=event.message or "下载任务执行异常")


def _finalize_download_runtime_entry(session_id: str, runtime: _DownloadRuntimeState) -> None:
    """清理已结束下载任务的运行时句柄。"""

    worker = runtime["worker"]
    if worker.is_alive():
        worker.join(timeout=0.1)
    else:
        worker.join()

    task_state = st.session_state.active_downloads.get(session_id)
    if not isinstance(task_state, DownloadTaskState):
        return
    if task_state.status not in (DownloadStatus.COMPLETED, DownloadStatus.FAILED):
        mark_download_completed(
            session_id,
            success=False,
            message="下载任务提前结束，请稍后重试",
        )


def poll_download_runtime_events() -> None:
    """轮询后台队列并将事件落入会话状态。"""

    runtime_state = _get_download_runtime_state()
    for session_id in list(runtime_state.keys()):
        runtime = runtime_state[session_id]
        processed_count = 0
        while processed_count < _DOWNLOAD_EVENT_BATCH_LIMIT:
            try:
                event = runtime["event_queue"].get_nowait()
            except Empty:
                break
            processed_count += 1
            if event.kind == "done":
                runtime["done"] = True
                continue
            _dispatch_download_runtime_event(session_id, event)

        worker = runtime["worker"]
        should_cleanup = (runtime["done"] or (not worker.is_alive())) and runtime["event_queue"].empty()
        if should_cleanup:
            _finalize_download_runtime_entry(session_id, runtime)
            del runtime_state[session_id]


def _collect_ticker_download_tasks(ticker: str) -> list[DownloadTaskState]:
    """收集当前股票的下载任务状态。"""

    active_downloads = cast(dict[str, DownloadTaskState], st.session_state.active_downloads)
    task_states: list[DownloadTaskState] = []
    for task in active_downloads.values():
        if task.ticker == ticker:
            task_states.append(task)
    task_states.sort(key=lambda item: item.started_at or "", reverse=True)
    return task_states


def _has_running_download_task(ticker: str) -> bool:
    """判断当前股票是否存在运行中的下载任务。"""

    task_states = _collect_ticker_download_tasks(ticker)
    for task_state in task_states:
        if task_state.status not in (DownloadStatus.COMPLETED, DownloadStatus.FAILED):
            return True
    return False


def _get_latest_ticker_download_task(ticker: str) -> DownloadTaskState | None:
    """获取当前股票最新的一条下载任务状态。"""

    task_states = _collect_ticker_download_tasks(ticker)
    if not task_states:
        return None
    return task_states[0]


def _render_single_download_task(task_state: DownloadTaskState) -> None:
    """渲染单个下载任务状态卡片。"""

    label = "下载进行中..."
    state = "running"
    if task_state.status == DownloadStatus.COMPLETED:
        label = "✅ 下载完成"
        state = "complete"
    elif task_state.status == DownloadStatus.FAILED:
        label = "❌ 下载失败"
        state = "error"

    status_container = st.status(label, state=state, expanded=True)
    status_container.progress(task_state.progress / 100.0)
    _render_download_logs(status_container, task_state.logs)
    if task_state.status == DownloadStatus.COMPLETED:
        st.success(
            f"已成功下载 {task_state.downloaded_filing_count} 个财报，"
            f"{task_state.downloaded_count} 个文件"
        )
    if task_state.errors:
        for error in task_state.errors:
            st.warning(error)


def _render_download_tasks_for_ticker(ticker: str) -> None:
    """渲染当前股票的下载任务区域。"""

    latest_task = _get_latest_ticker_download_task(ticker)
    if latest_task is None:
        return
    st.markdown("### 下载进度")
    _render_single_download_task(latest_task)


def render_download_status_with_optional_polling(ticker: str) -> None:
    """渲染下载状态并在支持 fragment 时启用自动轮询。"""

    if not _has_running_download_task(ticker):
        _render_download_tasks_for_ticker(ticker)
        return

    typed_fragment_factory = _resolve_fragment_factory()
    if typed_fragment_factory is not None:

        @typed_fragment_factory(run_every=_DOWNLOAD_POLL_INTERVAL_SECONDS)
        def _download_status_fragment() -> None:
            poll_download_runtime_events()
            _render_download_tasks_for_ticker(ticker)

        _download_status_fragment()
        return

    poll_download_runtime_events()
    _render_download_tasks_for_ticker(ticker)


def _resolve_fragment_factory() -> _FragmentDecoratorFactory | None:
    """解析并返回 Streamlit fragment 装饰器工厂。"""

    try:
        fragment_factory = st.fragment
    except AttributeError:
        return None
    if callable(fragment_factory):
        return cast(_FragmentDecoratorFactory, fragment_factory)
    return None


def _get_filing_list(
    workspace_root: Path,
    ticker: str,
    fins_service: FinsServiceProtocol | None,
) -> list[_FilingInfo]:
    """获取指定股票的已下载财报列表。

    参数:
        workspace_root: 工作区根目录。
        ticker: 股票代码。
        fins_service: 财报服务协议实例；为 None 时返回空列表。

    返回值:
        财报文件信息列表。
    """

    if fins_service is None:
        return []

    try:
        summaries = fins_service.list_filings(ticker)
    except (OSError, ValueError):
        st.error("读取财报列表失败，请确认工作区路径是否正确")
        return []

    resolved_root = workspace_root.resolve()
    filings: list[_FilingInfo] = []
    for s in summaries:
        # 计算主文件展示名称与相对路径
        file_name = s.primary_file_name or "未知"
        if s.primary_file_path:
            try:
                relative_path = Path(s.primary_file_path).relative_to(resolved_root)
                file_path = str(relative_path)
            except ValueError:
                file_path = s.primary_file_path
        else:
            file_path = "未知"

        filing_info: _FilingInfo = {
            "document_id": s.document_id,
            "file_name": file_name,
            "file_path": file_path,
            "form_type": s.form_type or "未知",
            "filing_date": s.filing_date or "未知",
            "report_date": s.report_date or "未知",
            "fiscal_year": str(s.fiscal_year) if s.fiscal_year is not None else "未知",
            "fiscal_period": s.fiscal_period or "未知",
            "status": "可用" if not s.is_deleted else "已删除",
        }
        filings.append(filing_info)

    return filings


def _render_filing_table(filings: list[_FilingInfo]) -> None:
    """渲染财报列表表格。

    参数:
        filings: 财报文件信息列表。

    返回值:
        无。

    异常:
        无。
    """

    # 准备表格数据
    df_data = []
    for f in filings:
        df_data.append({
            "文件名称": f["file_name"],
            "文件路径": f["file_path"],
            "表单类型": f["form_type"],
            "申报日期": f["filing_date"],
            "报告日期": f["report_date"],
            "财年": f["fiscal_year"],
            "财期": f["fiscal_period"],
            "状态": f["status"],
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

    调用方保证 `visible_rows >= 1`；不满足时抛出 ``ValueError``。

    参数:
        visible_rows: 目标可见数据行数，必须 >= 1。

    返回值:
        DataFrame 组件高度（像素）。

    异常:
        ValueError: 当 visible_rows 小于 1 时抛出。
    """

    if visible_rows < 1:
        raise ValueError("visible_rows 必须大于等于 1")
    return _DATAFRAME_HEADER_HEIGHT_PX + visible_rows * _DATAFRAME_ROW_HEIGHT_PX


def _init_download_settings_state(selected_stock: WatchlistItem) -> None:
    """初始化下载设置会话状态。

    参数:
        selected_stock: 当前选中的自选股。

    返回值:
        无。

    异常:
        无。
    """
    if "show_download_settings" not in st.session_state:
        st.session_state.show_download_settings = False
    if "download_settings_ticker" not in st.session_state:
        st.session_state.download_settings_ticker = selected_stock.ticker


def _render_filing_header_actions(selected_stock: WatchlistItem) -> None:
    """渲染财报页头部操作按钮。

    参数:
        selected_stock: 当前选中的自选股。

    返回值:
        无。

    异常:
        无。
    """

    _, spacer_column = st.columns([1, 1])
    button_text = "❌ 关闭下载" if _should_show_download_settings_for_ticker(selected_stock.ticker) else "📥 下载财报"
    with spacer_column:
        if st.button(button_text, width="stretch", type="secondary", key=f"toggle_download_settings_{selected_stock.ticker}"):
            _toggle_download_settings(selected_stock)
            st.rerun()


def _should_show_download_settings_for_ticker(ticker: str) -> bool:
    """判断当前股票是否应展示下载设置区域。

    参数:
        ticker: 股票代码。

    返回值:
        `True` 表示当前页面应展示该股票的下载设置区域，否则返回 `False`。

    异常:
        无。
    """

    return st.session_state.get("show_download_settings", False) and st.session_state.get("download_settings_ticker") == ticker


def _toggle_download_settings(selected_stock: WatchlistItem) -> None:
    """切换下载设置区域的显示/隐藏。

    参数:
        selected_stock: 当前选中的自选股。

    返回值:
        无。

    异常:
        无。
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

    参数:
        selected_stock: 当前选中的自选股。
        fins_service: 财报服务实例；为 None 时仅显示提示信息。

    返回值:
        无。

    异常:
        无。
    """
    _init_download_settings_state(selected_stock)
    ticker = selected_stock.ticker

    if fins_service is None:
        st.warning("财报服务不可用，无法进行下载操作")
        return

    with st.container():
        st.markdown("**📥 下载财报设置**")
        form_values = _render_download_form_fields(ticker)
        _render_download_submit_button(
            ticker=ticker,
            form_values=form_values,
            fins_service=fins_service,
        )


def _render_download_form_fields(ticker: str) -> _DownloadFormValues:
    """渲染下载设置表单字段并返回用户输入。

    参数:
        ticker: 当前股票代码。

    返回值:
        下载表单值对象。

    异常:
        无。
    """

    selected_form_types = st.multiselect(
        "选择要下载的财报表单类型",
        options=FORM_PROFILES.keys(),
        default=list(_DOWNLOAD_DEFAULT_FORM_TYPES),
        help="选择需要下载的 SEC 表单类型",
        key=f"download_form_types_{ticker}",
    )

    today = datetime.date.today()
    default_start_date = today.replace(year=today.year - _DOWNLOAD_DEFAULT_LOOKBACK_YEARS)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "开始日期",
            value=default_start_date,
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
    return _DownloadFormValues(
        form_types=tuple(selected_form_types),
        start_date=start_date,
        end_date=end_date,
        overwrite=overwrite,
    )


def _render_download_submit_button(
    ticker: str,
    form_values: _DownloadFormValues,
    fins_service: FinsServiceProtocol,
) -> None:
    """渲染下载提交按钮并在点击后执行提交。

    参数:
        ticker: 当前股票代码。
        form_values: 下载表单值对象。
        fins_service: 财报服务实例。

    返回值:
        无。

    异常:
        无。
    """

    if not st.button("开始下载", width="stretch", type="primary", key=f"download_start_btn_{ticker}"):
        return
    if not form_values.form_types:
        st.error("请至少选择一种表单类型")
        return
    _submit_download_task(ticker=ticker, form_values=form_values, fins_service=fins_service)


def _submit_download_task(
    ticker: str,
    form_values: _DownloadFormValues,
    fins_service: FinsServiceProtocol,
) -> None:
    """提交下载任务并启动后台事件流 worker。

    参数:
        ticker: 当前股票代码。
        form_values: 下载表单值对象。
        fins_service: 财报服务实例。

    返回值:
        无。

    异常:
        无。
    """

    submission: FinsSubmission | None = None
    try:
        submission = fins_service.submit(_build_download_submit_request(ticker, form_values))
        add_active_download(submission.session_id, ticker)
        start_download_worker(submission)
        st.success("下载任务已提交，后台执行中。")
        st.rerun()
    except Exception:
        st.error("下载任务失败，请稍后重试")
        if submission is not None:
            mark_download_completed(
                submission.session_id,
                success=False,
                message="下载任务执行异常",
            )


def _build_download_submit_request(ticker: str, form_values: _DownloadFormValues) -> FinsSubmitRequest:
    """构建下载命令提交请求对象。

    参数:
        ticker: 当前股票代码。
        form_values: 下载表单值对象。

    返回值:
        可直接提交的下载请求对象。

    异常:
        无。
    """

    start_date_str = form_values.start_date.isoformat() if form_values.start_date else None
    end_date_str = form_values.end_date.isoformat() if form_values.end_date else None
    return FinsSubmitRequest(
        command=FinsCommand(
            name=FinsCommandName.DOWNLOAD,
            payload=DownloadCommandPayload(
                ticker=ticker,
                form_type=form_values.form_types,
                start_date=start_date_str,
                end_date=end_date_str,
                overwrite=form_values.overwrite,
            ),
            stream=True,
        ),
    )



def render_filing_tab(
    selected_stock: WatchlistItem,
    workspace_root: Path,
    fins_service: FinsServiceProtocol | None,
) -> None:
    """渲染财报管理 Tab。

    参数:
        selected_stock: 当前选中的自选股。
        workspace_root: 工作区根目录。
        fins_service: 财报服务协议实例；为 None 时下载功能不可用。

    返回值:
        无。

    异常:
        无。
    """

    # 初始化下载状态
    init_download_state()
    _init_download_settings_state(selected_stock)
    poll_download_runtime_events()

    title_column, actions_column = st.columns([4, 1], gap="small", vertical_alignment="center")
    with title_column:
        st.subheader(f"{selected_stock.company_name} ({selected_stock.ticker}) - 财报管理")
    with actions_column:
        if fins_service is not None:
            _render_filing_header_actions(selected_stock)

    # 下载设置区域（展开/收起）
    if _should_show_download_settings_for_ticker(selected_stock.ticker):
        _render_download_settings(selected_stock, fins_service)
    render_download_status_with_optional_polling(selected_stock.ticker)

    # 获取已下载财报列表
    filings = _get_filing_list(workspace_root, selected_stock.ticker, fins_service)

    st.markdown("---")

    # 展示财报列表
    if filings:
        _render_filing_table(filings)
    else:
        if not _should_show_download_settings_for_ticker(selected_stock.ticker):
            st.info("暂无财报，请点击「下载财报」按钮获取")

