"""Host 层协议定义。

定义 Host 层暴露给 Service / UI 的稳定协议。
所有 registry / governor / event bus 的公共契约都在此声明，默认实现放在各自模块。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from dayu.contracts.cancellation import CancellationToken
from dayu.contracts.events import AppEvent, PublishedRunEventProtocol
from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.contracts.reply_outbox import ReplyOutboxRecord, ReplyOutboxState, ReplyOutboxSubmitRequest
from dayu.contracts.run import ACTIVE_STATES, TERMINAL_STATES, RunCancelReason, RunRecord, RunState
from dayu.contracts.session import SessionRecord, SessionSource, SessionState
from dayu.host.host_execution import HostExecutorProtocol
from dayu.host.pending_turn_store import PendingConversationTurn, PendingConversationTurnState


# ---------------------------------------------------------------------------
# SessionRegistry
# ---------------------------------------------------------------------------


class SessionClosedError(RuntimeError):
    """尝试对不存在或已 CLOSED 的 session 执行写入时抛出。

    该异常由 Host 内部写入屏障（pending turn / reply outbox 仓储）在检测到
    session 不存在或已进入 ``CLOSED`` 终态时抛出，用于阻断
    ``cancel_session`` 窗口期内的并发写入，防止产生孤儿数据。
    """

    def __init__(self, session_id: str) -> None:
        """初始化异常。

        Args:
            session_id: 触发屏障的 session ID。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(f"session 不存在或已关闭，拒绝写入: session_id={session_id}")
        self.session_id = session_id


class SessionActivityQueryProtocol(Protocol):
    """面向仓储层的 session 活性查询协议。

    仅暴露仓储写入屏障所需的最小查询能力，避免把 ``SessionRegistryProtocol``
    的完整生命周期接口泄漏给 pending turn / reply outbox 仓储。

    查询语义：``True`` 表示 session 存在且当前不是 ``CLOSED``；
    ``False`` 表示 session 不存在或已 ``CLOSED``。
    """

    def is_session_active(self, session_id: str) -> bool:
        """查询指定 session 是否仍处于非 CLOSED 状态。

        Args:
            session_id: 目标 session ID。

        Returns:
            session 存在且非 CLOSED 时返回 ``True``，否则返回 ``False``。

        Raises:
            无。
        """
        ...


@runtime_checkable
class SessionRegistryProtocol(Protocol):
    """宿主级会话注册表协议。

    管理 SessionRecord 的生命周期，所有操作跨进程可见（底层 SQLite）。
    """

    def create_session(
        self,
        source: SessionSource,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """创建新 session。

        Args:
            source: 会话来源。
            session_id: 显式指定 ID，不传则自动生成。
            scene_name: 首次使用的 scene。
            metadata: 会话级交付上下文元数据。

        Returns:
            新创建的 SessionRecord。
        """
        ...

    def ensure_session(
        self,
        session_id: str,
        source: SessionSource,
        *,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """幂等获取或创建 session。

        session_id 已存在则返回现有记录（同时 touch），不存在则创建。
        适用于 WeChat daemon 重启等需要确定性 session_id 的场景。

        Args:
            session_id: 确定性 session ID。
            source: 会话来源。
            scene_name: 首次使用的 scene。
            metadata: 会话级交付上下文元数据。

        Returns:
            已有或新创建的 SessionRecord。
        """
        ...

    def get_session(self, session_id: str) -> SessionRecord | None:
        """查询单个 session。

        Args:
            session_id: 目标 session ID。

        Returns:
            SessionRecord 或 None（不存在时）。
        """
        ...

    def list_sessions(
        self,
        *,
        state: SessionState | None = None,
        source: SessionSource | None = None,
        scene_name: str | None = None,
    ) -> list[SessionRecord]:
        """列出 sessions。

        Args:
            state: 可选状态过滤。
            source: 可选来源过滤。
            scene_name: 可选 scene 名称过滤。

        Returns:
            匹配的 SessionRecord 列表。
        """
        ...

    def touch_session(self, session_id: str) -> None:
        """更新 session 最后活跃时间。

        Args:
            session_id: 目标 session ID。

        Raises:
            KeyError: session 不存在时抛出。
        """
        ...

    def close_session(self, session_id: str) -> None:
        """关闭 session。

        Args:
            session_id: 目标 session ID。

        Raises:
            KeyError: session 不存在时抛出。
        """
        ...

    def close_idle_sessions(self, idle_threshold: timedelta) -> list[str]:
        """关闭超过空闲阈值的活跃 session。

        Args:
            idle_threshold: 空闲判定阈值。

        Returns:
            被关闭的 session_id 列表。
        """
        ...

    def is_session_active(self, session_id: str) -> bool:
        """查询指定 session 是否仍处于非 CLOSED 状态。

        仓储写入屏障会在每次写入前调用本方法，若返回 ``False`` 则拒绝写入，
        确保 ``cancel_session`` 关闭 session 后不会再产生孤儿 pending turn /
        reply outbox 记录。

        Args:
            session_id: 目标 session ID。

        Returns:
            session 存在且状态非 ``CLOSED`` 时返回 ``True``；否则返回 ``False``。
        """
        ...


# ---------------------------------------------------------------------------
# RunRegistry
# ---------------------------------------------------------------------------


@runtime_checkable
class RunRegistryProtocol(Protocol):
    """宿主级运行注册表协议。

    管理 RunRecord 的生命周期和状态机，所有操作跨进程可见（底层 SQLite）。
    """

    def register_run(
        self,
        *,
        session_id: str | None = None,
        service_type: str,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> RunRecord:
        """注册一个新 run。

        Args:
            session_id: 关联 session（可选）。
            service_type: 服务类型标识。
            scene_name: 场景名。
            metadata: 宿主侧交付上下文，仅承载稳定键值（与 ExecutionContract.metadata
                同型），不允许随意塞入业务非结构化字段。

        Returns:
            状态为 CREATED 的 RunRecord。
        """
        ...

    def start_run(self, run_id: str) -> RunRecord:
        """将 run 状态从 CREATED/QUEUED 转为 RUNNING。"""
        ...

    def complete_run(self, run_id: str, *, error_summary: str | None = None) -> RunRecord:
        """标记 run 成功完成。"""
        ...

    def fail_run(self, run_id: str, *, error_summary: str | None = None) -> RunRecord:
        """标记 run 失败。"""
        ...

    def mark_cancelled(
        self,
        run_id: str,
        *,
        cancel_reason: RunCancelReason = RunCancelReason.USER_CANCELLED,
    ) -> RunRecord:
        """标记 run 已取消。"""
        ...

    def mark_unsettled(
        self,
        run_id: str,
        *,
        error_summary: str | None = None,
    ) -> RunRecord:
        """将 run 标记为 UNSETTLED（orphan cleanup / 无法判定的残留）。"""
        ...

    def request_cancel(
        self,
        run_id: str,
        *,
        cancel_reason: RunCancelReason = RunCancelReason.USER_CANCELLED,
    ) -> bool:
        """请求取消 run（跨进程可见）。"""
        ...

    def is_cancel_requested(self, run_id: str) -> bool:
        """查询 run 是否已记录取消意图。"""
        ...

    def get_run(self, run_id: str) -> RunRecord | None:
        """查询单个 run。"""
        ...

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        state: RunState | None = None,
        service_type: str | None = None,
    ) -> list[RunRecord]:
        """列出 runs，支持多维过滤。"""
        ...

    def list_active_runs(self) -> list[RunRecord]:
        """列出所有活跃 run。"""
        ...

    def list_active_runs_for_owner(self, owner_pid: int) -> list[RunRecord]:
        """列出指定 owner_pid 拥有的所有活跃 run。"""
        ...

    def cleanup_orphan_runs(self) -> list[str]:
        """清理 owner_pid 已死亡的活跃 run。"""
        ...


# ---------------------------------------------------------------------------
# PendingConversationTurnStore
# ---------------------------------------------------------------------------


@runtime_checkable
class PendingConversationTurnStoreProtocol(Protocol):
    """pending conversation turn 仓储协议。

    该仓储是 resume V1 的真源，只记录当前尚未完成的 conversation turn。
    """

    def upsert_pending_turn(
        self,
        *,
        session_id: str,
        scene_name: str,
        user_text: str,
        source_run_id: str,
        resumable: bool,
        state: PendingConversationTurnState,
        resume_source_json: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> PendingConversationTurn:
        """创建或更新当前 session/scene 的活跃 pending turn。

        `resume_source_json` 必须承载 Host 自己的 accepted/prepared 恢复快照。
        """
        ...

    def get_pending_turn(self, pending_turn_id: str) -> PendingConversationTurn | None:
        """按 ID 查询 pending turn。"""
        ...

    def get_session_pending_turn(
        self,
        *,
        session_id: str,
        scene_name: str,
    ) -> PendingConversationTurn | None:
        """按 session/scene 查询当前 pending turn。"""
        ...

    def list_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: PendingConversationTurnState | None = None,
        resumable_only: bool = False,
    ) -> list[PendingConversationTurn]:
        """列出 pending turn。"""
        ...

    def update_state(
        self,
        pending_turn_id: str,
        *,
        state: PendingConversationTurnState,
    ) -> PendingConversationTurn:
        """更新 pending turn 的 Host 内部状态。"""
        ...

    def record_resume_attempt(
        self,
        pending_turn_id: str,
        *,
        max_attempts: int,
    ) -> PendingConversationTurn:
        """在未达到上限时原子记录一次 pending turn 恢复尝试。"""
        ...

    def record_resume_failure(
        self,
        pending_turn_id: str,
        *,
        error_message: str,
    ) -> PendingConversationTurn:
        """记录一次 pending turn 恢复失败。"""
        ...

    def release_resume_lease(self, pending_turn_id: str) -> PendingConversationTurn | None:
        """把 RESUMING 的 pending turn 原子回退到 ``pre_resume_state``。

        Args:
            pending_turn_id: 目标 pending turn ID。

        Returns:
            回退后的 pending turn；若记录缺失或当前 state 非 RESUMING 则
            返回 ``None`` 或原记录（幂等 no-op）。

        Raises:
            无：状态不符合回退条件时视为 no-op。
        """
        ...

    def rebind_source_run_id_for_resume(
        self,
        pending_turn_id: str,
        *,
        new_source_run_id: str,
    ) -> PendingConversationTurn:
        """在持有 RESUMING lease 的前提下把 ``source_run_id`` 原子重绑到当前 resumed run。

        Args:
            pending_turn_id: 目标 pending turn ID。
            new_source_run_id: 当前 resumed run 的 run_id。

        Returns:
            重绑后的 pending turn 记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: ``new_source_run_id`` 为空字符串时抛出。
            PendingTurnResumeConflictError: 当前 state 非 ``RESUMING`` 时抛出。
        """
        ...

    def delete_pending_turn(self, pending_turn_id: str) -> None:
        """删除指定 pending turn。"""
        ...

    def delete_by_session_id(self, session_id: str) -> int:
        """删除指定 session 的所有 pending turn。

        Args:
            session_id: 目标 session ID。

        Returns:
            被删除的记录数。
        """
        ...


@runtime_checkable
class ReplyOutboxStoreProtocol(Protocol):
    """reply outbox 仓储协议。

    该仓储是可选出站交付真源，只记录已被显式提交的待交付回复。
    """

    def submit_reply(self, request: ReplyOutboxSubmitRequest) -> ReplyOutboxRecord:
        """显式提交待交付回复。"""
        ...

    def get_reply(self, delivery_id: str) -> ReplyOutboxRecord | None:
        """按 ID 查询交付记录。"""
        ...

    def get_by_delivery_key(self, delivery_key: str) -> ReplyOutboxRecord | None:
        """按幂等键查询交付记录。"""
        ...

    def list_replies(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: ReplyOutboxState | None = None,
    ) -> list[ReplyOutboxRecord]:
        """列出交付记录。"""
        ...

    def claim_reply(self, delivery_id: str) -> ReplyOutboxRecord:
        """把记录推进到发送中状态。"""
        ...

    def mark_delivered(self, delivery_id: str) -> ReplyOutboxRecord:
        """标记记录已完成交付。"""
        ...

    def mark_failed(
        self,
        delivery_id: str,
        *,
        retryable: bool,
        error_message: str,
    ) -> ReplyOutboxRecord:
        """标记记录交付失败。"""
        ...

    def delete_by_session_id(self, session_id: str) -> int:
        """删除指定 session 的所有交付记录。

        Args:
            session_id: 目标 session ID。

        Returns:
            被删除的记录数。
        """
        ...

    def cleanup_stale_in_progress_deliveries(
        self,
        *,
        max_age: timedelta,
    ) -> list[str]:
        """把超过 max_age 的 DELIVERY_IN_PROGRESS 回退为 FAILED_RETRYABLE。"""
        ...


# ---------------------------------------------------------------------------
# ConcurrencyGovernor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcurrencyPermit:
    """并发许可凭证。"""

    permit_id: str
    lane: str
    acquired_at: datetime


@dataclass(frozen=True)
class LaneStatus:
    """并发 lane 状态快照。"""

    lane: str
    max_concurrent: int
    active: int


@runtime_checkable
class ConcurrencyGovernorProtocol(Protocol):
    """跨进程并发治理协议。

    基于 SQLite permits 表实现跨进程信号量语义。
    """

    def acquire(
        self,
        lane: str,
        *,
        timeout: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ConcurrencyPermit:
        """获取并发许可，超时前阻塞等待。

        Args:
            lane: 并发通道名。
            timeout: 最大等待秒数，None 表示无限等待。
            cancellation_token: 可选取消令牌；若等待期间被触发，必须尽快结束等待。

        Returns:
            并发许可凭证。

        Raises:
            TimeoutError: 等待超时。
            CancelledError: 等待期间收到取消请求。
        """
        ...

    def acquire_many(
        self,
        lanes: list[str],
        *,
        timeout: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[ConcurrencyPermit]:
        """原子获取多个 lane 的并发许可，要么全拿要么全不拿。

        实现必须在单个持久化事务内完成"所有 lane 额度检查 + 所有 permit 写入"，
        保证进程被 SIGKILL / OOM 杀死于两步 acquire 之间时不会残留 permit。

        Args:
            lanes: 需要 acquire 的 lane 名列表；调用方应保证已去重并按字母序排序，
                以在跨 run 之间使用一致的顺序，规避潜在死锁。
            timeout: 最大等待秒数，None 表示无限等待。
            cancellation_token: 可选取消令牌；若等待期间被触发，必须尽快结束等待。

        Returns:
            与 ``lanes`` 对应顺序的许可列表。

        Raises:
            TimeoutError: 等待超时；此时不持有任何 permit。
            ValueError: 出现未配置的 lane 名时抛出；此时不持有任何 permit。
            CancelledError: 等待期间收到取消请求；此时不持有任何 permit。
        """
        ...

    def try_acquire(self, lane: str) -> ConcurrencyPermit | None:
        """尝试立即获取并发许可（非阻塞）。

        Args:
            lane: 并发通道名。

        Returns:
            ConcurrencyPermit 或 None（无可用额度）。
        """
        ...

    def release(self, permit: ConcurrencyPermit) -> None:
        """释放并发许可。

        Args:
            permit: 之前获取的许可凭证。
        """
        ...

    def get_lane_status(self, lane: str) -> LaneStatus:
        """查询指定 lane 的当前状态。

        Args:
            lane: 并发通道名。

        Returns:
            LaneStatus 快照。
        """
        ...

    def get_all_status(self) -> dict[str, LaneStatus]:
        """查询所有 lane 的当前状态。

        Returns:
            lane 名到 LaneStatus 的映射。
        """
        ...

    def cleanup_stale_permits(self) -> list[str]:
        """清理 owner_pid 已死亡的 permit。

        Returns:
            被清理的 permit_id 列表。
        """
        ...


# ---------------------------------------------------------------------------
# RunEventBus
# ---------------------------------------------------------------------------


@runtime_checkable
class EventSubscription(Protocol):
    """事件订阅句柄。"""

    def __aiter__(self) -> AsyncIterator[PublishedRunEventProtocol]:
        """异步迭代订阅事件。"""
        ...

    def close(self) -> None:
        """关闭订阅。"""
        ...

    @property
    def is_closed(self) -> bool:
        """返回订阅是否已关闭。"""
        ...


@runtime_checkable
class RunEventBusProtocol(Protocol):
    """进程内多消费者事件总线协议。

    适用于 Web / WeChat / GUI 等长驻进程，CLI 单命令进程不需要。
    """

    def publish(self, run_id: str, event: PublishedRunEventProtocol) -> None:
        """发布事件到指定 run 的所有订阅者。

        Args:
            run_id: 关联 run ID。
            event: 稳定运行事件包络。
        """
        ...

    def subscribe(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> EventSubscription:
        """订阅事件流。

        Args:
            run_id: 按 run 过滤（精确匹配）。
            session_id: 按 session 过滤（该 session 下所有 run 的事件）。

        Returns:
            EventSubscription 句柄。
        """
        ...


# ---------------------------------------------------------------------------
# Service-facing Host protocols
# ---------------------------------------------------------------------------


def _empty_execution_delivery_context() -> ExecutionDeliveryContext:
    """返回空的执行交付上下文。

    Returns:
        空交付上下文。

    Raises:
        无。
    """

    return {}


@dataclass(frozen=True)
class PendingTurnSummary:
    """Host 暴露给上层的 pending turn 摘要。

    该对象是 Service / UI 可依赖的稳定公开契约，
    不暴露 Host 内部仓储记录类型。
    """

    pending_turn_id: str
    session_id: str
    scene_name: str
    user_text: str
    source_run_id: str
    resumable: bool
    state: str
    metadata: ExecutionDeliveryContext = field(default_factory=_empty_execution_delivery_context)


@dataclass(frozen=True)
class ConversationSessionDigest:
    """Host 暴露给管理面的 conversation 摘要。

    该对象只承载可安全展示的 transcript 派生信息，不暴露完整
    ``ConversationTranscript``，避免 UI 或 Service 直接依赖 Host 内部存储结构。
    """

    turn_count: int
    first_question_preview: str
    last_question_preview: str


@dataclass(frozen=True)
class ConversationSessionTurnExcerpt:
    """Host 暴露给管理面的 conversation 单轮摘录。

    该对象只承载可安全展示的单轮对话文本，不暴露完整
    ``ConversationTranscript`` 与 derived memory 结构。
    """

    user_text: str
    assistant_text: str
    created_at: str


@runtime_checkable
class SessionOperationsProtocol(Protocol):
    """Service 可见的 Host session 能力协议。"""

    def create_session(
        self,
        source: SessionSource,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """创建新的 Host session。"""
        ...

    def ensure_session(
        self,
        session_id: str,
        source: SessionSource,
        *,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """按确定性 session_id 幂等获取或创建 Host session。"""
        ...

    def get_session(self, session_id: str) -> SessionRecord | None:
        """查询单个 Host session。"""
        ...

    def list_sessions(
        self,
        *,
        state: SessionState | None = None,
        source: SessionSource | None = None,
        scene_name: str | None = None,
    ) -> list[SessionRecord]:
        """列出 Host session。"""
        ...

    def touch_session(self, session_id: str) -> None:
        """刷新 Host session 最后活跃时间。"""
        ...


@runtime_checkable
class PendingTurnOperationsProtocol(Protocol):
    """Service 可见的 pending conversation turn 能力协议。"""

    def get_pending_turn(self, pending_turn_id: str) -> PendingTurnSummary | None:
        """按 ID 查询 pending turn。"""
        ...

    def resume_pending_turn_stream(
        self,
        pending_turn_id: str,
        *,
        session_id: str,
    ) -> AsyncIterator[AppEvent]:
        """校验 pending turn 是否允许恢复，并返回恢复后的事件流。"""
        ...

    def list_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        resumable_only: bool = False,
    ) -> list[PendingTurnSummary]:
        """列出 Host 侧 pending turn。"""
        ...

    def cleanup_stale_pending_turns(
        self,
        *,
        session_id: str | None = None,
    ) -> list[str]:
        """清理关联 run 已终态、且按调和规则应删除的 pending turn。

        Args:
            session_id: 若提供，仅扫描该 session 下的 pending turn；
                为 ``None`` 时全量扫描。

        Returns:
            被清理的 pending_turn_id 列表。
        """
        ...


@runtime_checkable
class ReplyOutboxOperationsProtocol(Protocol):
    """Service 可见的 reply outbox 能力协议。"""

    def submit_reply_for_delivery(self, request: ReplyOutboxSubmitRequest) -> ReplyOutboxRecord:
        """显式提交待交付回复。"""
        ...

    def get_reply_outbox(self, delivery_id: str) -> ReplyOutboxRecord | None:
        """按 ID 查询交付记录。"""
        ...

    def list_reply_outbox(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: ReplyOutboxState | None = None,
    ) -> list[ReplyOutboxRecord]:
        """列出交付记录。"""
        ...

    def claim_reply_delivery(self, delivery_id: str) -> ReplyOutboxRecord:
        """把记录推进到发送中状态。"""
        ...

    def mark_reply_delivered(self, delivery_id: str) -> ReplyOutboxRecord:
        """标记交付完成。"""
        ...

    def mark_reply_delivery_failed(
        self,
        delivery_id: str,
        *,
        retryable: bool,
        error_message: str,
    ) -> ReplyOutboxRecord:
        """标记交付失败。"""
        ...


@runtime_checkable
class RunAdministrationProtocol(Protocol):
    """Service 可见的 Host run 管理能力协议。"""

    def cancel_run(self, run_id: str) -> RunRecord:
        """请求取消指定 run。"""
        ...

    def cancel_session_runs(self, session_id: str) -> list[str]:
        """取消指定 session 下的全部活跃 run。"""
        ...

    def get_run(self, run_id: str) -> RunRecord | None:
        """查询单个 run。"""
        ...

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        state: RunState | None = None,
        service_type: str | None = None,
    ) -> list[RunRecord]:
        """列出 run。"""
        ...

    def list_active_runs(self) -> list[RunRecord]:
        """列出全部活跃 run。"""
        ...

    def cleanup_orphan_runs(self) -> list[str]:
        """清理孤儿 run。"""
        ...


@runtime_checkable
class HostGovernanceProtocol(Protocol):
    """Service 可见的 Host 治理查询能力协议。"""

    def cleanup_stale_permits(self) -> list[str]:
        """清理过期并发 permit。"""
        ...

    def cleanup_stale_reply_outbox_deliveries(
        self,
        *,
        max_age: timedelta = ...,
    ) -> list[str]:
        """回退超过 max_age 的 reply outbox DELIVERY_IN_PROGRESS 记录。

        Args:
            max_age: 认定 in_progress 陈旧的最大存活时间。

        Returns:
            被回退的 delivery_id 列表。

        Raises:
            无。
        """
        ...

    def cleanup_stale_pending_turns(
        self,
        *,
        session_id: str | None = None,
    ) -> list[str]:
        """清理关联 run 已终态、且按调和规则应删除的 pending turn。

        Args:
            session_id: 若提供，仅扫描该 session 下的 pending turn；
                为 ``None`` 时全量扫描。

        Returns:
            被清理的 pending_turn_id 列表。
        """
        ...

    def get_all_lane_statuses(self) -> dict[str, LaneStatus]:
        """获取全部并发 lane 状态快照。"""
        ...


@runtime_checkable
class EventSubscriptionOperationsProtocol(Protocol):
    """Service 可见的 Host 事件订阅能力协议。"""

    def subscribe_run_events(self, run_id: str) -> EventSubscription:
        """订阅指定 run 的事件流。"""
        ...

    def subscribe_session_events(self, session_id: str) -> EventSubscription:
        """订阅指定 session 下全部 run 的事件流。"""
        ...


@runtime_checkable
class HostedExecutionGatewayProtocol(SessionOperationsProtocol, HostExecutorProtocol, Protocol):
    """Service 可见的通用宿主执行网关协议。"""


@runtime_checkable
class ConversationalExecutionGatewayProtocol(
    HostedExecutionGatewayProtocol,
    PendingTurnOperationsProtocol,
    Protocol,
):
    """聊天类 Service 使用的宿主执行网关协议。"""


@runtime_checkable
class ReplyDeliveryGatewayProtocol(
    HostedExecutionGatewayProtocol,
    ReplyOutboxOperationsProtocol,
    Protocol,
):
    """需要显式写入 reply outbox 的 Service 宿主网关协议。"""


@runtime_checkable
class HostAdminOperationsProtocol(
    SessionOperationsProtocol,
    RunAdministrationProtocol,
    HostGovernanceProtocol,
    EventSubscriptionOperationsProtocol,
    Protocol,
):
    """宿主管理面使用的 Host 能力协议。"""

    def cancel_session(self, session_id: str) -> tuple[SessionRecord, list[str]]:
        """关闭 session 并取消其下所有活跃 run。"""
        ...

    def get_conversation_session_digest(self, session_id: str) -> ConversationSessionDigest:
        """读取指定 session 的 conversation 摘要。"""
        ...

    def list_conversation_session_turn_excerpts(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> list[ConversationSessionTurnExcerpt]:
        """读取指定 session 的最近 conversation 单轮摘录。"""
        ...


__all__ = [
    "ConversationSessionDigest",
    "ConversationSessionTurnExcerpt",
    "ConversationalExecutionGatewayProtocol",
    "ConcurrencyGovernorProtocol",
    "ConcurrencyPermit",
    "EventSubscription",
    "EventSubscriptionOperationsProtocol",
    "HostAdminOperationsProtocol",
    "HostGovernanceProtocol",
    "HostedExecutionGatewayProtocol",
    "LaneStatus",
    "PendingConversationTurnStoreProtocol",
    "PendingTurnSummary",
    "PendingTurnOperationsProtocol",
    "ReplyDeliveryGatewayProtocol",
    "ReplyOutboxOperationsProtocol",
    "ReplyOutboxStoreProtocol",
    "RunEventBusProtocol",
    "RunAdministrationProtocol",
    "RunRegistryProtocol",
    "SessionActivityQueryProtocol",
    "SessionClosedError",
    "SessionOperationsProtocol",
    "SessionRegistryProtocol",
]
