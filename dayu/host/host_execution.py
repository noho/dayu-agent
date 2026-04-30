"""Host 层宿主执行协议。

该模块定义 Host 内部使用的宿主执行协议，
让 Service 只表达 session 语义和业务 handler，不再直接操作 run registry、
并发治理器、事件总线或取消桥接实现。

``HostedRunSpec`` 和 ``HostedRunContext`` 的稳定定义位于
``dayu.contracts.host_execution``，本模块从该处导入并在 Host 内部复用。
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Protocol, TypeVar, runtime_checkable

from dayu.contracts.agent_execution import ExecutionContract, ReplayHandle
from dayu.contracts.events import AppEvent, AppResult, PublishedRunEventProtocol
from dayu.contracts.host_execution import HostedRunContext, HostedRunSpec
from dayu.host.prepared_turn import PreparedAgentTurnSnapshot

TStreamEvent = TypeVar("TStreamEvent", bound=PublishedRunEventProtocol)
TSyncResult = TypeVar("TSyncResult")


@runtime_checkable
class HostExecutorProtocol(Protocol):
    """统一的宿主执行协议。"""

    def run_operation_stream(
        self,
        *,
        spec: HostedRunSpec,
        event_stream_factory: Callable[[HostedRunContext], AsyncIterator[TStreamEvent]],
    ) -> AsyncIterator[TStreamEvent]:
        """托管一次流式执行。"""
        ...

    def run_operation_sync(
        self,
        *,
        spec: HostedRunSpec,
        operation: Callable[[HostedRunContext], TSyncResult],
        on_cancel: Callable[[], TSyncResult] | None = None,
    ) -> TSyncResult:
        """托管一次同步执行。"""
        ...

    def run_agent_stream(
        self,
        execution_contract: ExecutionContract,
        *,
        resumed_pending_turn_id: str | None = None,
        resumed_pending_turn_lease_id: str | None = None,
    ) -> AsyncIterator[AppEvent]:
        """托管一次 Agent 子执行并返回应用层事件流。"""
        ...

    def run_prepared_turn_stream(
        self,
        prepared_turn: PreparedAgentTurnSnapshot,
        *,
        resumed_pending_turn_id: str | None = None,
        resumed_pending_turn_lease_id: str | None = None,
    ) -> AsyncIterator[AppEvent]:
        """基于 Host prepared turn 快照恢复一次 Agent 子执行。"""
        ...

    async def run_agent_and_wait(
        self,
        execution_contract: ExecutionContract,
    ) -> AppResult:
        """托管一次 Agent 子执行并等待完整结果。

        Raises:
            CancelledError: 执行被取消时抛出。
        """
        ...

    async def run_agent_and_wait_replayable(
        self,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        """托管一次 Agent 子执行并颁发用于带历史回放的句柄。

        Raises:
            CancelledError: 执行被取消时抛出。
        """
        ...

    async def replay_agent_and_wait(
        self,
        handle: ReplayHandle,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        """基于上一次颁发的 ``ReplayHandle`` 带历史回放一次 Agent 执行。

        Raises:
            RuntimeError: handle 已失效或与目标 session 不匹配。
            CancelledError: 执行被取消时抛出。
        """
        ...

    def discard_replay_state_for_session(self, session_id: str) -> None:
        """清理 session 关联的所有 replay 状态。"""
        ...

    def discard_replay_state(self, handle: ReplayHandle) -> None:
        """释放单个未消费的 replay 句柄对应的内存状态。

        Args:
            handle: 待释放的句柄；若 stash 中已不存在则静默跳过。
        """
        ...


__all__ = [
    "HostExecutorProtocol",
    "HostedRunContext",
    "HostedRunSpec",
]
