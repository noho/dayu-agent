"""下载面板：状态、后台运行时、表单与状态展示。"""

from __future__ import annotations

import datetime
import threading
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty, Queue
from typing import TypedDict, cast

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
from dayu.web.streamlit.components.watchlist import WatchlistItem
from dayu.web.streamlit.pages.filing.download_progress import (
    DownloadQueueEvent,
    DownloadStatus,
    DownloadTaskState,
    LogEntry,
    apply_download_completion,
    apply_download_progress,
    create_download_task,
    run_download_stream_worker,
)

_DOWNLOAD_DEFAULT_FORM_TYPES: tuple[str, ...] = ("10-K", "10-Q")
_DOWNLOAD_DEFAULT_LOOKBACK_YEARS = 3
_DOWNLOAD_RUNTIME_STATE_KEY = "download_runtime_handles"
_DOWNLOAD_EVENT_BATCH_LIMIT = 128
_DOWNLOAD_POLL_INTERVAL_SECONDS = 1.0
_DOWNLOAD_WORKER_JOIN_TIMEOUT_SECONDS = 0.1
_STATUS_CONTAINER_MAX_LOG_ITEMS = 120
_DOWNLOAD_LOG_LEVEL_LABELS: dict[str, str] = {
    "error": "ERROR",
    "warning": "WARN",
    "info": "INFO",
}
_FragmentDecoratorFactory = Callable[
    ...,
    Callable[[Callable[[], None]], Callable[[], None]],
]


@dataclass(frozen=True)
class _DownloadFormValues:
    """下载表单值对象。"""

    form_types: tuple[str, ...]
    start_date: datetime.date | None
    end_date: datetime.date | None
    overwrite: bool


class DownloadRuntimeState(TypedDict):
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


def _get_download_runtime_state() -> dict[str, DownloadRuntimeState]:
    """获取下载运行时状态字典。"""

    runtime_state = st.session_state.get(_DOWNLOAD_RUNTIME_STATE_KEY)
    if isinstance(runtime_state, dict):
        return cast(dict[str, DownloadRuntimeState], runtime_state)
    initialized_state: dict[str, DownloadRuntimeState] = {}
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

    match event.kind:
        case "progress":
            if event.payload is not None:
                update_download_progress(session_id, event.payload)
        case "result":
            mark_download_completed(session_id, success=True, message=event.message or "下载完成")
        case "error":
            mark_download_completed(session_id, success=False, message=event.message or "下载任务执行异常")
        case "done":
            # done 事件由 poll_download_runtime_events 调用方消费并驱动清理。
            return
        case _:
            raise ValueError(f"未知下载运行时事件类型: {event.kind}")
    

def _finalize_download_runtime_entry(session_id: str, runtime: DownloadRuntimeState) -> bool:
    """清理已结束下载任务的运行时句柄。

    参数:
        session_id: 下载任务会话 ID。
        runtime: 下载运行时状态句柄。

    返回值:
        bool: 清理完成返回 True；worker 仍未结束返回 False，调用方应在下一轮轮询中重试。

    约束:
        必须在 Streamlit 脚本线程调用（直接读写 ``st.session_state``），
        当前唯一调用方 ``poll_download_runtime_events`` 始终满足此条件。
    """

    worker = runtime["worker"]
    if worker.is_alive():
        worker.join(timeout=_DOWNLOAD_WORKER_JOIN_TIMEOUT_SECONDS)
        if worker.is_alive():
            return False
    else:
        worker.join()

    task_state = st.session_state.active_downloads.get(session_id)
    if not isinstance(task_state, DownloadTaskState):
        return True
    if task_state.status not in (DownloadStatus.COMPLETED, DownloadStatus.FAILED):
        mark_download_completed(
            session_id,
            success=False,
            message="下载任务提前结束，请稍后重试",
        )
    return True


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

        should_cleanup = runtime["done"] and runtime["event_queue"].empty()
        if should_cleanup:
            finalized = _finalize_download_runtime_entry(session_id, runtime)
            if finalized:
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


def _format_log_time(timestamp: str) -> str:
    """格式化日志时间为时分秒。"""

    if not timestamp:
        return ""
    try:
        return datetime.datetime.fromisoformat(timestamp).strftime("%H:%M:%S")
    except ValueError:
        return timestamp


def _build_download_log_lines(logs: list[LogEntry]) -> list[str]:
    """构建下载日志的文本行。"""

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
    """使用 Streamlit 原生组件渲染下载日志。"""

    status_container.caption("最近日志")
    log_lines = _build_download_log_lines(logs)
    if not log_lines:
        status_container.caption("暂无日志")
        return
    status_container.code("\n".join(log_lines))


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


def _init_download_settings_state(selected_stock: WatchlistItem) -> None:
    """初始化下载设置会话状态。"""

    if "show_download_settings" not in st.session_state:
        st.session_state.show_download_settings = False
    if "download_settings_ticker" not in st.session_state:
        st.session_state.download_settings_ticker = selected_stock.ticker


def _render_filing_header_actions(selected_stock: WatchlistItem) -> None:
    """渲染财报页头部操作按钮。"""

    _, spacer_column = st.columns([1, 1])
    button_text = "❌ 关闭下载" if _should_show_download_settings_for_ticker(selected_stock.ticker) else "📥 下载财报"
    with spacer_column:
        if st.button(button_text, width="stretch", type="secondary", key=f"toggle_download_settings_{selected_stock.ticker}"):
            _toggle_download_settings(selected_stock)
            st.rerun()


def _should_show_download_settings_for_ticker(ticker: str) -> bool:
    """判断当前股票是否应展示下载设置区域。"""

    return st.session_state.get("show_download_settings", False) and st.session_state.get("download_settings_ticker") == ticker


def _toggle_download_settings(selected_stock: WatchlistItem) -> None:
    """切换下载设置区域的显示/隐藏。"""

    _init_download_settings_state(selected_stock)
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
    """在当前页面渲染下载任务设置区域。"""

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
    """渲染下载设置表单字段并返回用户输入。"""

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
    """渲染下载提交按钮并在点击后执行提交。"""

    has_running = _has_running_download_task(ticker)
    button_label = "正在下载..." if has_running else "开始下载"
    if not st.button(button_label, width="stretch", type="primary", disabled=has_running, key=f"download_start_btn_{ticker}"):
        return
    if not form_values.form_types:
        st.error("请至少选择一种表单类型")
        return
    _submit_download_task(ticker=ticker, form_values=form_values, fins_service=fins_service)


def render_download_section(
    selected_stock: WatchlistItem,
    fins_service: FinsServiceProtocol | None,
) -> bool:
    """渲染下载区并返回下载设置是否可见。

    参数:
        selected_stock: 当前选中的自选股。
        fins_service: 财报服务协议实例；为 None 时仅展示不可用提示。

    返回值:
        bool: 当前 ticker 的下载设置区是否处于展开状态。

    异常:
        无。
    """

    init_download_state()
    _init_download_settings_state(selected_stock)
    poll_download_runtime_events()

    title_column, actions_column = st.columns([4, 1], gap="small", vertical_alignment="center")
    with title_column:
        st.subheader(f"{selected_stock.company_name} ({selected_stock.ticker}) - 财报管理")
    with actions_column:
        if fins_service is not None:
            _render_filing_header_actions(selected_stock)

    show_download_settings = _should_show_download_settings_for_ticker(selected_stock.ticker)
    if show_download_settings:
        _render_download_settings(selected_stock, fins_service)
    render_download_status_with_optional_polling(selected_stock.ticker)
    return show_download_settings


def _submit_download_task(
    ticker: str,
    form_values: _DownloadFormValues,
    fins_service: FinsServiceProtocol,
) -> None:
    """提交下载任务并启动后台事件流 worker。"""

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
    """构建下载命令提交请求对象。"""

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
