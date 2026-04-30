"""Run 数据模型。

定义 Agent 执行运行（Run）的状态机和记录结构。
Run 是宿主层对一次 Agent 调用的追踪单元，与 session 关联但生命周期独立。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dayu.contracts.execution_metadata import (
    ExecutionDeliveryContext,
    empty_execution_delivery_context,
)
from dayu.process_liveness import OwnerIdentity


ORPHAN_RUN_ERROR_SUMMARY = "orphan: owner process terminated"


class RunState(str, Enum):
    """Run 状态枚举。

    状态机合法转换:
        CREATED → QUEUED → RUNNING → SUCCEEDED / FAILED / CANCELLED
        CREATED → RUNNING → SUCCEEDED / FAILED / CANCELLED
        CREATED / QUEUED / RUNNING → CANCELLED
        CREATED / QUEUED / RUNNING → UNSETTLED（orphan cleanup 专用吸收态）

    UNSETTLED 与 FAILED 语义分离：
        - FAILED 表示业务/Agent 明确失败；
        - UNSETTLED 表示 Host 无法判定的残留（owner 进程异常终止），
          由 startup cleanup 统一收敛，方便 admin / 自愈逻辑基于 state
          做 discriminator，不再依赖 error_summary 字符串匹配。
    """

    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNSETTLED = "unsettled"


class RunCancelReason(str, Enum):
    """Run 取消原因枚举。"""

    USER_CANCELLED = "user_cancelled"
    TIMEOUT = "timeout"


# 合法的状态转换表：key 为当前状态，value 为允许的目标状态集合
_VALID_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset(
        {RunState.QUEUED, RunState.RUNNING, RunState.CANCELLED, RunState.UNSETTLED}
    ),
    RunState.QUEUED: frozenset(
        {RunState.RUNNING, RunState.CANCELLED, RunState.UNSETTLED}
    ),
    RunState.RUNNING: frozenset(
        {RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED, RunState.UNSETTLED}
    ),
    RunState.SUCCEEDED: frozenset(),
    RunState.FAILED: frozenset(),
    RunState.CANCELLED: frozenset(),
    RunState.UNSETTLED: frozenset(),
}

# 终态集合
TERMINAL_STATES: frozenset[RunState] = frozenset({
    RunState.SUCCEEDED,
    RunState.FAILED,
    RunState.CANCELLED,
    RunState.UNSETTLED,
})

# 活跃状态集合（用于查询过滤）
ACTIVE_STATES: frozenset[RunState] = frozenset({
    RunState.CREATED,
    RunState.QUEUED,
    RunState.RUNNING,
})


def is_valid_transition(from_state: RunState, to_state: RunState) -> bool:
    """检查状态转换是否合法。

    Args:
        from_state: 当前状态。
        to_state: 目标状态。

    Returns:
        True 表示转换合法。
    """
    return to_state in _VALID_TRANSITIONS.get(from_state, frozenset())


@dataclass
class RunRecord:
    """一次 Agent 运行的记录。

    Attributes:
        run_id: 运行唯一标识。
        session_id: 所属 session 标识，可为 None（无 session 关联的独立运行）。
        service_type: 服务类型（prompt / chat_turn / write_chapter / write_pipeline / download / process）。
        scene_name: 场景名称（fins_analysis / write / audit 等）。
        state: 当前运行状态。
        created_at: 创建时间。
        started_at: 开始执行时间。
        completed_at: 完成时间（成功/失败/取消）。
        error_summary: 错误摘要（失败时填充）。
        cancel_requested_at: 请求取消时间（写入取消意图时填充）。
        cancel_requested_reason: 请求取消原因（写入取消意图时填充）。
        cancel_reason: 取消原因（取消时填充）。
        owner_pid: 创建该 run 的进程 PID，用于死进程检测。
        owner_process_start_time: 创建该 run 的进程的进程创建时间（epoch 秒），
            与 ``owner_boot_id`` 共同构成 ``OwnerIdentity``，封堵 PID 复用窗口；
            采集失败时为 ``None``，等值校验自动退化。
        owner_boot_id: 创建该 run 时的系统 boot 标识，重启后必然变化；
            采集失败时为 ``None``。
        metadata: 宿主侧交付上下文，仅承载稳定键值（与 ExecutionContract.metadata
            同型 ExecutionDeliveryContext），不作为自由 dict 承载业务字段。
    """

    run_id: str
    session_id: str | None
    service_type: str
    scene_name: str | None
    state: RunState
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_summary: str | None = None
    cancel_requested_at: datetime | None = None
    cancel_requested_reason: RunCancelReason | None = None
    cancel_reason: RunCancelReason | None = None
    owner_pid: int = 0
    owner_process_start_time: float | None = None
    owner_boot_id: str | None = None
    metadata: ExecutionDeliveryContext = field(default_factory=empty_execution_delivery_context)

    @property
    def owner_identity(self) -> OwnerIdentity:
        """以 ``OwnerIdentity`` 形式暴露 owner 三元组。

        Returns:
            由 ``owner_pid`` / ``owner_process_start_time`` / ``owner_boot_id``
            构成的 ``OwnerIdentity``，供 cleanup / 判活逻辑使用。
        """

        return OwnerIdentity(
            pid=self.owner_pid,
            process_start_time=self.owner_process_start_time,
            boot_id=self.owner_boot_id,
        )

    def is_terminal(self) -> bool:
        """判断是否处于终态。

        Returns:
            True 表示 run 已结束（成功/失败/取消）。
        """
        return self.state in TERMINAL_STATES

    def is_active(self) -> bool:
        """判断是否处于活跃状态。

        Returns:
            True 表示 run 仍在进行中。
        """
        return self.state in ACTIVE_STATES
