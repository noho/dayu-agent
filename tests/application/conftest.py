"""Host / Service 测试共享 fixtures。"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace as dataclass_replace
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Callable, TypeVar

from dayu.contracts.agent_execution import ExecutionContract, ReplayHandle
from dayu.contracts.execution_metadata import (
    ExecutionDeliveryContext,
    empty_execution_delivery_context,
)
from dayu.contracts.events import AppEvent, AppEventType, AppResult
from dayu.contracts.run import RunCancelReason, RunRecord, RunState
from dayu.contracts.session import SessionRecord, SessionSource, SessionState
from dayu.host.host_execution import HostedRunContext, HostedRunSpec
from dayu.host.pending_turn_store import (
    InMemoryPendingConversationTurnStore,
    PendingConversationTurn,
    PendingConversationTurnState,
)
from dayu.host.prepared_turn import PreparedAgentTurnSnapshot
from dayu.contracts.cancellation import CancellationToken


TStreamEvent = TypeVar("TStreamEvent")
TSyncResult = TypeVar("TSyncResult")


class StubSessionRegistry:
    """最小化 SessionRegistry stub，满足 Service 构造需求。

    不走 SQLite，全内存实现，仅用于单元测试。
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}

    def create_session(
        self,
        source: SessionSource,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """创建 session。"""
        sid = session_id or uuid.uuid4().hex
        now = datetime.now(timezone.utc)
        record = SessionRecord(
            session_id=sid,
            source=source,
            state=SessionState.ACTIVE,
            scene_name=scene_name,
            created_at=now,
            last_activity_at=now,
            metadata=metadata or {},
        )
        self._sessions[sid] = record
        return record

    def ensure_session(
        self,
        session_id: str,
        source: SessionSource,
        *,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """幂等创建或获取 session。"""
        if session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session(source, session_id=session_id, scene_name=scene_name, metadata=metadata)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """查询 session。"""
        return self._sessions.get(session_id)

    def list_sessions(
        self,
        *,
        state: SessionState | None = None,
        source: SessionSource | None = None,
        scene_name: str | None = None,
    ) -> list[SessionRecord]:
        """列出 sessions。"""

        return [
            session
            for session in self._sessions.values()
            if (state is None or session.state == state)
            and (source is None or session.source == source)
            and (scene_name is None or session.scene_name == scene_name)
        ]

    def touch_session(self, session_id: str) -> None:
        """更新最后活跃时间。"""
        if session_id not in self._sessions:
            raise KeyError(f"session 不存在: {session_id}")

    def close_session(self, session_id: str) -> None:
        """关闭 session，并将状态推进到 CLOSED。"""
        existing = self._sessions.get(session_id)
        if existing is None:
            raise KeyError(f"session 不存在: {session_id}")
        self._sessions[session_id] = SessionRecord(
            session_id=existing.session_id,
            source=existing.source,
            state=SessionState.CLOSED,
            scene_name=existing.scene_name,
            created_at=existing.created_at,
            last_activity_at=existing.last_activity_at,
            metadata=existing.metadata,
        )

    def close_idle_sessions(self, idle_threshold: timedelta) -> list[str]:
        """测试桩不做 idle session 回收。"""

        del idle_threshold
        return []

    def is_session_active(self, session_id: str) -> bool:
        """查询 session 是否处于 ACTIVE 状态。"""

        existing = self._sessions.get(str(session_id or "").strip())
        if existing is None:
            return False
        return existing.state == SessionState.ACTIVE

    def get_session_state(self, session_id: str) -> SessionState | None:
        """查询 session 当前状态。"""

        existing = self._sessions.get(str(session_id or "").strip())
        if existing is None:
            return None
        return existing.state

    def _replace_state(self, session_id: str, target: SessionState) -> SessionRecord:
        existing = self._sessions.get(session_id)
        if existing is None:
            raise KeyError(f"session 不存在: {session_id}")
        replaced = dataclass_replace(existing, state=target)
        self._sessions[session_id] = replaced
        return replaced

    def begin_clearing(self, session_id: str) -> None:
        """从 ACTIVE 进入 CLEARING。"""
        existing = self._sessions.get(session_id)
        if existing is None:
            raise KeyError(f"session 不存在: {session_id}")
        if existing.state != SessionState.ACTIVE:
            from dayu.host.protocols import SessionStateTransitionError

            raise SessionStateTransitionError(
                session_id,
                operation="begin_clearing",
                current_state=existing.state,
                expected_states=(SessionState.ACTIVE,),
            )
        self._replace_state(session_id, SessionState.CLEARING)

    def end_clearing(self, session_id: str) -> None:
        """从 CLEARING 退出回 ACTIVE。"""
        existing = self._sessions.get(session_id)
        if existing is None:
            raise KeyError(f"session 不存在: {session_id}")
        if existing.state != SessionState.CLEARING:
            from dayu.host.protocols import SessionStateTransitionError

            raise SessionStateTransitionError(
                session_id,
                operation="end_clearing",
                current_state=existing.state,
                expected_states=(SessionState.CLEARING,),
            )
        self._replace_state(session_id, SessionState.ACTIVE)

    def mark_clearing_failed(self, session_id: str) -> None:
        """从 CLEARING 升级到 CLEARING_FAILED。"""
        existing = self._sessions.get(session_id)
        if existing is None:
            raise KeyError(f"session 不存在: {session_id}")
        if existing.state != SessionState.CLEARING:
            from dayu.host.protocols import SessionStateTransitionError

            raise SessionStateTransitionError(
                session_id,
                operation="mark_clearing_failed",
                current_state=existing.state,
                expected_states=(SessionState.CLEARING,),
            )
        self._replace_state(session_id, SessionState.CLEARING_FAILED)


class StubRunRegistry:
    """最小化 RunRegistry stub，满足 Service 构造需求。

    不走 SQLite，全内存实现，仅用于单元测试。
    """

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}

    def register_run(
        self,
        *,
        session_id: str | None = None,
        service_type: str,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> RunRecord:
        """注册 run。"""
        import os
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)
        record = RunRecord(
            run_id=run_id,
            session_id=session_id,
            service_type=service_type,
            scene_name=scene_name,
            state=RunState.CREATED,
            created_at=now,
            cancel_requested_at=None,
            cancel_requested_reason=None,
            owner_pid=os.getpid(),
            metadata=metadata if metadata is not None else empty_execution_delivery_context(),
        )
        self._runs[run_id] = record
        return record

    def start_run(self, run_id: str) -> RunRecord:
        """标记 RUNNING。"""
        record = self._runs[run_id]
        self._runs[run_id] = RunRecord(
            run_id=record.run_id,
            session_id=record.session_id,
            service_type=record.service_type,
            scene_name=record.scene_name,
            state=RunState.RUNNING,
            created_at=record.created_at,
            started_at=datetime.now(timezone.utc),
            cancel_requested_at=record.cancel_requested_at,
            cancel_requested_reason=record.cancel_requested_reason,
            cancel_reason=None,
            owner_pid=record.owner_pid,
            metadata=record.metadata,
        )
        return self._runs[run_id]

    def complete_run(self, run_id: str, *, error_summary: str | None = None) -> RunRecord:
        """标记成功。"""
        record = self._runs[run_id]
        self._runs[run_id] = RunRecord(
            run_id=record.run_id,
            session_id=record.session_id,
            service_type=record.service_type,
            scene_name=record.scene_name,
            state=RunState.SUCCEEDED,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=datetime.now(timezone.utc),
            error_summary=error_summary,
            cancel_requested_at=record.cancel_requested_at,
            cancel_requested_reason=record.cancel_requested_reason,
            cancel_reason=None,
            owner_pid=record.owner_pid,
            metadata=record.metadata,
        )
        return self._runs[run_id]

    def fail_run(self, run_id: str, *, error_summary: str | None = None) -> RunRecord:
        """标记失败。"""
        record = self._runs[run_id]
        self._runs[run_id] = RunRecord(
            run_id=record.run_id,
            session_id=record.session_id,
            service_type=record.service_type,
            scene_name=record.scene_name,
            state=RunState.FAILED,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=datetime.now(timezone.utc),
            error_summary=error_summary,
            cancel_requested_at=record.cancel_requested_at,
            cancel_requested_reason=record.cancel_requested_reason,
            cancel_reason=None,
            owner_pid=record.owner_pid,
            metadata=record.metadata,
        )
        return self._runs[run_id]

    def mark_unsettled(self, run_id: str, *, error_summary: str | None = None) -> RunRecord:
        """标记 UNSETTLED（测试桩同 fail_run 模型）。"""
        record = self._runs[run_id]
        self._runs[run_id] = RunRecord(
            run_id=record.run_id,
            session_id=record.session_id,
            service_type=record.service_type,
            scene_name=record.scene_name,
            state=RunState.UNSETTLED,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=datetime.now(timezone.utc),
            error_summary=error_summary,
            cancel_requested_at=record.cancel_requested_at,
            cancel_requested_reason=record.cancel_requested_reason,
            cancel_reason=None,
            owner_pid=record.owner_pid,
            metadata=record.metadata,
        )
        return self._runs[run_id]

    def mark_cancelled(
        self,
        run_id: str,
        *,
        cancel_reason: RunCancelReason = RunCancelReason.USER_CANCELLED,
    ) -> RunRecord:
        """标记取消。"""
        record = self._runs[run_id]
        self._runs[run_id] = RunRecord(
            run_id=record.run_id,
            session_id=record.session_id,
            service_type=record.service_type,
            scene_name=record.scene_name,
            state=RunState.CANCELLED,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=datetime.now(timezone.utc),
            cancel_requested_at=record.cancel_requested_at,
            cancel_requested_reason=record.cancel_requested_reason,
            cancel_reason=cancel_reason,
            owner_pid=record.owner_pid,
            metadata=record.metadata,
        )
        return self._runs[run_id]

    def get_run(self, run_id: str) -> RunRecord | None:
        """查询 run。"""
        return self._runs.get(run_id)

    def is_cancel_requested(self, run_id: str) -> bool:
        """查询取消状态。"""
        record = self._runs.get(run_id)
        if record is None:
            return False
        return record.cancel_requested_at is not None

    def request_cancel(
        self,
        run_id: str,
        *,
        cancel_reason: RunCancelReason = RunCancelReason.USER_CANCELLED,
    ) -> bool:
        """请求取消。"""
        record = self._runs.get(run_id)
        if record is None:
            return False
        if record.state not in {RunState.CREATED, RunState.QUEUED, RunState.RUNNING}:
            return False
        if record.cancel_requested_at is not None:
            return False
        self._runs[run_id] = RunRecord(
            run_id=record.run_id,
            session_id=record.session_id,
            service_type=record.service_type,
            scene_name=record.scene_name,
            state=record.state,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            error_summary=record.error_summary,
            cancel_requested_at=datetime.now(timezone.utc),
            cancel_requested_reason=cancel_reason,
            cancel_reason=record.cancel_reason,
            owner_pid=record.owner_pid,
            metadata=record.metadata,
        )
        return True

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        state: RunState | None = None,
        service_type: str | None = None,
    ) -> list[RunRecord]:
        """列出 run。"""

        runs = list(self._runs.values())
        if session_id is not None:
            runs = [run for run in runs if run.session_id == session_id]
        if state is not None:
            runs = [run for run in runs if run.state == state]
        if service_type is not None:
            runs = [run for run in runs if run.service_type == service_type]
        return runs

    def list_active_runs(self) -> list[RunRecord]:
        """列出活跃 run。"""

        return [run for run in self._runs.values() if run.state in {RunState.CREATED, RunState.QUEUED, RunState.RUNNING}]

    def list_active_runs_for_owner(self, owner_pid: int) -> list[RunRecord]:
        """按 owner_pid 过滤活跃 run。"""

        return [
            run for run in self._runs.values()
            if run.state in {RunState.CREATED, RunState.QUEUED, RunState.RUNNING}
            and run.owner_pid == int(owner_pid)
        ]

    def cleanup_orphan_runs(self) -> list[str]:
        """测试桩不做 orphan 清理。"""

        return []


class StubPendingTurnStore(InMemoryPendingConversationTurnStore):
    """测试用 pending turn 仓储。"""

    def get_pending_turn_record(self, pending_turn_id: str) -> PendingConversationTurn | None:
        """读取测试桩内部完整 pending turn 记录。"""

        return self._records.get(str(pending_turn_id or "").strip())

    def seed_pending_turn(
        self,
        *,
        session_id: str,
        scene_name: str,
        user_text: str,
        source_run_id: str,
        resumable: bool = True,
        resume_source_json: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
        state: PendingConversationTurnState = PendingConversationTurnState.PREPARED_BY_HOST,
    ) -> PendingConversationTurn:
        """预置一条 pending turn。"""

        record = self.upsert_pending_turn(
            session_id=session_id,
            scene_name=scene_name,
            user_text=user_text,
            source_run_id=source_run_id,
            resumable=resumable,
            state=state,
            resume_source_json=resume_source_json,
            metadata=metadata,
        )
        return record


class StubHostExecutor:
    """最小化 HostExecutor stub。"""

    def __init__(self) -> None:
        self.last_spec: HostedRunSpec | None = None
        self.last_execution_contract: ExecutionContract | None = None
        self.last_prepared_turn: PreparedAgentTurnSnapshot | None = None
        self.last_resumed_pending_turn_id: str | None = None
        self.stream_call_count = 0
        self.sync_call_count = 0

    async def run_operation_stream(
        self,
        *,
        spec: HostedRunSpec,
        event_stream_factory: Callable[[HostedRunContext], AsyncIterator[TStreamEvent]],
    ) -> AsyncIterator[TStreamEvent]:
        """执行流式 stub。"""

        self.last_spec = spec
        self.stream_call_count += 1
        context = HostedRunContext(run_id="run_test", cancellation_token=CancellationToken())
        async for event in event_stream_factory(context):
            yield event

    def run_operation_sync(
        self,
        *,
        spec: HostedRunSpec,
        operation: Callable[[HostedRunContext], TSyncResult],
        on_cancel: Callable[[], TSyncResult] | None = None,
    ) -> TSyncResult:
        """执行同步 stub。"""

        del on_cancel
        self.last_spec = spec
        self.sync_call_count += 1
        context = HostedRunContext(run_id="run_test", cancellation_token=CancellationToken())
        return operation(context)

    async def run_agent_stream(
        self,
        execution_contract: ExecutionContract,
        *,
        resumed_pending_turn_id: str | None = None,
        resumed_pending_turn_lease_id: str | None = None,
    ) -> AsyncIterator[AppEvent]:
        """执行 Agent 路径 stub。"""

        self.last_execution_contract = execution_contract
        self.last_resumed_pending_turn_id = resumed_pending_turn_id
        self.last_resumed_pending_turn_lease_id = resumed_pending_turn_lease_id
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="hello", meta={})
        yield AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload={"content": "done", "degraded": False},
            meta={},
        )

    async def run_prepared_turn_stream(
        self,
        prepared_turn: PreparedAgentTurnSnapshot,
        *,
        resumed_pending_turn_id: str | None = None,
        resumed_pending_turn_lease_id: str | None = None,
    ) -> AsyncIterator[AppEvent]:
        """执行 prepared turn 恢复路径 stub。"""

        self.last_prepared_turn = prepared_turn
        self.last_resumed_pending_turn_id = resumed_pending_turn_id
        self.last_resumed_pending_turn_lease_id = resumed_pending_turn_lease_id
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="hello", meta={})
        yield AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload={"content": "done", "degraded": False},
            meta={},
        )

    async def run_agent_and_wait(
        self,
        execution_contract: ExecutionContract,
    ) -> AppResult:
        """执行 Agent 路径并直接返回结果。"""

        self.last_execution_contract = execution_contract
        return AppResult(content="done", errors=[], warnings=[], degraded=False)

    async def run_agent_and_wait_replayable(
        self,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        """Stub：测试不验证 replay 路径，颁发空 handle 满足协议即可。"""

        result = await self.run_agent_and_wait(execution_contract)
        return result, ReplayHandle(handle_id="stub_replay")

    async def replay_agent_and_wait(
        self,
        handle: ReplayHandle,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        """Stub：测试不验证 replay 路径，原样返回新 handle。"""

        del handle
        result = await self.run_agent_and_wait(execution_contract)
        return result, ReplayHandle(handle_id="stub_replay")

    def discard_replay_state_for_session(self, session_id: str) -> None:
        """Stub：测试不验证 replay stash 清理路径，无操作满足协议即可。"""

        del session_id

    def discard_replay_state(self, handle: ReplayHandle) -> None:
        """Stub：测试不验证 replay stash 清理路径，无操作满足协议即可。"""

        del handle
