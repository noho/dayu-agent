"""分析报告任务与 Host 状态同步辅助模块。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Protocol, cast

from dayu.services.contracts import RunAdminView
from dayu.services.protocols import HostAdminServiceProtocol

_HOST_STATE_CURSOR_KEY = "__host_state__"


class HostSyncTaskProtocol(Protocol):
    """Host 同步逻辑依赖的任务视图协议。"""

    status: str
    run_id: str | None
    session_id: str | None
    started_at: str | None
    status_cursor: dict[str, str]


ActiveWriteTaskRecord = dict[str, str | int | float | bool | None | dict[str, str] | list[dict[str, str]]]


@dataclass(frozen=True)
class BindRunOutcome:
    """绑定 Host run_id 的输出结果。"""

    run_id: str
    session_id: str | None
    log_message: str


@dataclass(frozen=True)
class HostCancelOutcome:
    """Host 取消请求结果。"""

    success: bool
    message: str
    run_id: str | None = None


@dataclass(frozen=True)
class HostStatusSyncOutcome:
    """Host 状态同步结果。"""

    status_cursor: dict[str, str]
    message_override: str | None = None
    final_status: str | None = None
    final_exit_code: int | None = None
    final_message: str | None = None
    completed_at: str | None = None
    progress_override: float | None = None
    log_message: str | None = None
    log_level: str = "info"


def parse_iso_datetime(time_str: str | None) -> datetime | None:
    """解析 ISO 时间文本。

    参数:
        time_str: ISO 8601 时间字符串，允许为空。

    返回值:
        解析成功返回 ``datetime``；为空或格式不合法返回 ``None``。

    异常:
        无。
    """

    if time_str is None:
        return None
    normalized = time_str.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def to_unix_timestamp(value: datetime | None) -> float | None:
    """把时间对象转换为可比较的 Unix 时间戳。

    参数:
        value: 时间对象，允许为空。

    返回值:
        输入为空返回 ``None``；否则返回 Unix 时间戳（秒）。

    异常:
        无。
    """

    if value is None:
        return None
    return value.timestamp()


def select_best_run(
    *,
    started_at: str | None,
    runs: Iterable[RunAdminView],
) -> RunAdminView | None:
    """按与任务启动时间的接近程度选择最匹配的运行实例。

    参数:
        started_at: 任务启动时间。
        runs: 候选 Host 运行列表。

    返回值:
        匹配到返回单个 ``RunAdminView``；无候选时返回 ``None``。

    异常:
        无。
    """

    started_at_ts = to_unix_timestamp(parse_iso_datetime(started_at))
    candidate_runs = list(runs)
    if not candidate_runs:
        return None

    def _sort_key(run: RunAdminView) -> tuple[float, float]:
        created_ts = to_unix_timestamp(parse_iso_datetime(run.created_at))
        if created_ts is None:
            return (1.0, 0.0)
        if started_at_ts is None:
            return (1.0, -created_ts)
        return (abs(created_ts - started_at_ts), -created_ts)

    return sorted(candidate_runs, key=_sort_key)[0]



def collect_occupied_run_ids(
    *,
    ticker: str,
    active_write_tasks: Mapping[str, ActiveWriteTaskRecord] | None,
) -> set[str]:
    """收集其它标的已占用的 run_id。

    参数:
        ticker: 当前标的代码。
        active_write_tasks: 会话中活跃任务数据映射。

    返回值:
        非当前标的已绑定的 run_id 集合。

    异常:
        无。
    """

    occupied_run_ids: set[str] = set()
    if active_write_tasks is None:
        return occupied_run_ids
    for key, task_data in active_write_tasks.items():
        if not key.startswith("write_task_") or key == f"write_task_{ticker}":
            continue
        raw_run_id = task_data.get("run_id")
        if isinstance(raw_run_id, str) and raw_run_id:
            occupied_run_ids.add(raw_run_id)
    return occupied_run_ids


def bind_write_task_run_id_from_host(
    *,
    ticker: str,
    task: HostSyncTaskProtocol,
    host_admin_service: HostAdminServiceProtocol,
    active_write_tasks: Mapping[str, ActiveWriteTaskRecord] | None,
    running_status: str,
    service_type: str,
) -> BindRunOutcome | None:
    """尝试根据 Host 活跃任务绑定 run_id。

    参数:
        ticker: 当前标的代码。
        task: 当前任务状态视图。
        host_admin_service: Host 管理服务。
        active_write_tasks: 会话中活跃任务数据映射。
        running_status: 运行中状态字符串。
        service_type: Host 查询的服务类型。

    返回值:
        绑定成功返回 ``BindRunOutcome``；无需绑定或绑定失败返回 ``None``。

    异常:
        无。
    """

    if task.status != running_status or task.run_id is not None:
        return None
    running_runs = host_admin_service.list_runs(
        active_only=True,
        service_type=service_type,
    )
    if not running_runs:
        return None

    occupied_run_ids = collect_occupied_run_ids(
        ticker=ticker,
        active_write_tasks=active_write_tasks,
    )
    candidate_runs = [run for run in running_runs if run.run_id not in occupied_run_ids]
    selected_run = select_best_run(started_at=task.started_at, runs=candidate_runs)
    if selected_run is None:
        return None
    return BindRunOutcome(
        run_id=selected_run.run_id,
        session_id=selected_run.session_id,
        log_message=f"已绑定 Host 任务: run_id={selected_run.run_id}",
    )


def request_cancel_via_host(
    *,
    task: HostSyncTaskProtocol,
    host_admin_service: HostAdminServiceProtocol,
) -> HostCancelOutcome:
    """向 Host 发起取消请求。

    参数:
        task: 当前任务状态视图，必须含 ``run_id``。
        host_admin_service: Host 管理服务。

    返回值:
        取消请求结果对象。

    异常:
        无。异常转换为失败结果。
    """

    if task.run_id is None:
        return HostCancelOutcome(success=False, message="当前任务尚未绑定 Host run_id，请稍后重试。")
    try:
        host_admin_service.cancel_run(task.run_id)
    except KeyError:
        return HostCancelOutcome(success=False, message=f"Host 中未找到运行任务: {task.run_id}")
    return HostCancelOutcome(success=True, message="已发送取消请求，请稍候。", run_id=task.run_id)


def sync_write_task_status_from_host(
    *,
    task: HostSyncTaskProtocol,
    host_admin_service: HostAdminServiceProtocol,
    running_states: set[str],
    cancelled_status: str,
    failed_status: str,
    completed_status: str,
) -> HostStatusSyncOutcome | None:
    """从 Host 同步任务状态并返回增量更新结果。

    参数:
        task: 当前任务状态视图。
        host_admin_service: Host 管理服务。
        running_states: Host 运行中状态集合。
        cancelled_status: 取消终态文案。
        failed_status: 失败终态文案。
        completed_status: 成功终态文案。

    返回值:
        有状态变化时返回 ``HostStatusSyncOutcome``；无变化返回 ``None``。

    异常:
        无。
    """

    if task.run_id is None:
        return None
    run_view = host_admin_service.get_run(task.run_id)
    if run_view is None:
        return None

    host_state = run_view.state.strip().lower()
    cursor = dict(task.status_cursor)
    previous_state = cursor.get(_HOST_STATE_CURSOR_KEY)
    if previous_state == host_state:
        return None
    cursor[_HOST_STATE_CURSOR_KEY] = host_state

    if host_state in running_states:
        if run_view.cancel_requested_at is not None:
            return HostStatusSyncOutcome(
                status_cursor=cursor,
                message_override="Host 已确认取消请求，等待任务收敛...",
            )
        return HostStatusSyncOutcome(status_cursor=cursor)

    if host_state == "cancelled":
        return HostStatusSyncOutcome(
            status_cursor=cursor,
            final_status=cancelled_status,
            final_exit_code=130,
            final_message="任务已取消（Host 已收敛）",
            completed_at=run_view.finished_at,
            log_message=f"Host 任务已取消: run_id={task.run_id}, cancel_reason={run_view.cancel_reason or 'unknown'}",
            log_level="warning",
        )

    if host_state == "failed":
        error_summary = run_view.error_summary or "未知错误"
        return HostStatusSyncOutcome(
            status_cursor=cursor,
            final_status=failed_status,
            final_message=f"任务执行失败（Host）: {error_summary}",
            completed_at=run_view.finished_at,
            log_message=f"Host 任务失败: run_id={task.run_id}, error={error_summary}",
            log_level="error",
        )

    if host_state == "succeeded":
        return HostStatusSyncOutcome(
            status_cursor=cursor,
            final_status=completed_status,
            final_message="分析报告生成完成",
            completed_at=run_view.finished_at,
            progress_override=100.0,
            log_message=f"Host 任务已完成: run_id={task.run_id}",
        )

    return HostStatusSyncOutcome(status_cursor=cursor)


__all__ = [
    "ActiveWriteTaskRecord",
    "BindRunOutcome",
    "HostCancelOutcome",
    "HostStatusSyncOutcome",
    "HostSyncTaskProtocol",
    "bind_write_task_run_id_from_host",
    "collect_occupied_run_ids",
    "parse_iso_datetime",
    "request_cancel_via_host",
    "select_best_run",
    "sync_write_task_status_from_host",
    "to_unix_timestamp",
]
