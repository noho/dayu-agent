"""HostAdminService 测试。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from dayu.contracts.events import AppEvent, AppEventType, PublishedRunEventProtocol
from dayu.contracts.run import RunCancelReason, RunRecord, RunState
from dayu.contracts.session import SessionRecord, SessionSource, SessionState
from dayu.host.event_bus import AsyncQueueEventBus
from dayu.host.host import Host
from dayu.host.protocols import LaneStatus
from dayu.services.contracts import HostCleanupResult
from dayu.services.host_admin_service import HostAdminService
from dayu.services.startup_recovery import recover_host_startup_state


class _FakeSessionRegistry:
    """测试用 session registry。"""

    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.records: dict[str, SessionRecord] = {
            "session_1": SessionRecord(
                session_id="session_1",
                source=SessionSource.WEB,
                state=SessionState.ACTIVE,
                scene_name="prompt",
                created_at=now,
                last_activity_at=now,
            )
        }

    def create_session(
        self,
        source: SessionSource,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SessionRecord:
        """创建会话。"""

        del metadata
        now = datetime.now(timezone.utc)
        record = SessionRecord(
            session_id=session_id or "session_created",
            source=source,
            state=SessionState.ACTIVE,
            scene_name=scene_name,
            created_at=now,
            last_activity_at=now,
        )
        self.records[record.session_id] = record
        return record

    def get_session(self, session_id: str) -> SessionRecord | None:
        """查询会话。"""

        return self.records.get(session_id)

    def list_sessions(self, *, state: SessionState | None = None) -> list[SessionRecord]:
        """列出会话。"""

        return [record for record in self.records.values() if state is None or record.state == state]

    def touch_session(self, session_id: str) -> None:
        """刷新活跃时间。"""

        record = self.records[session_id]
        self.records[session_id] = SessionRecord(
            session_id=record.session_id,
            source=record.source,
            state=record.state,
            scene_name=record.scene_name,
            created_at=record.created_at,
            last_activity_at=datetime.now(timezone.utc),
            metadata=record.metadata,
        )

    def close_session(self, session_id: str) -> None:
        """关闭会话。"""

        record = self.records[session_id]
        self.records[session_id] = SessionRecord(
            session_id=record.session_id,
            source=record.source,
            state=SessionState.CLOSED,
            scene_name=record.scene_name,
            created_at=record.created_at,
            last_activity_at=record.last_activity_at,
            metadata=record.metadata,
        )


class _FakeRunRegistry:
    """测试用 run registry。"""

    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.records: dict[str, RunRecord] = {
            "run_1": RunRecord(
                run_id="run_1",
                session_id="session_1",
                service_type="prompt",
                scene_name="prompt",
                state=RunState.RUNNING,
                created_at=now,
                started_at=now,
            ),
            "run_2": RunRecord(
                run_id="run_2",
                session_id="session_2",
                service_type="chat_turn",
                scene_name="interactive",
                state=RunState.SUCCEEDED,
                created_at=now,
                started_at=now,
                completed_at=now,
            ),
        }
        self.cleaned_orphans: list[str] = ["run_orphan_1"]

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        state: RunState | None = None,
        service_type: str | None = None,
    ) -> list[RunRecord]:
        """列出运行。"""

        result = list(self.records.values())
        if session_id is not None:
            result = [record for record in result if record.session_id == session_id]
        if state is not None:
            result = [record for record in result if record.state == state]
        if service_type is not None:
            result = [record for record in result if record.service_type == service_type]
        return result

    def list_active_runs(self) -> list[RunRecord]:
        """列出活跃运行。"""

        return [record for record in self.records.values() if record.is_active()]

    def get_run(self, run_id: str) -> RunRecord | None:
        """查询单个运行。"""

        return self.records.get(run_id)

    def request_cancel(
        self,
        run_id: str,
        *,
        cancel_reason: RunCancelReason = RunCancelReason.USER_CANCELLED,
    ) -> bool:
        """请求取消运行。"""

        record = self.records.get(run_id)
        if record is None or record.is_terminal():
            return False
        self.records[run_id] = RunRecord(
            run_id=record.run_id,
            session_id=record.session_id,
            service_type=record.service_type,
            scene_name=record.scene_name,
            state=record.state,
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            cancel_requested_at=datetime.now(timezone.utc),
            cancel_requested_reason=cancel_reason,
            cancel_reason=record.cancel_reason,
            metadata=record.metadata,
        )
        return True

    def cleanup_orphan_runs(self) -> list[str]:
        """清理孤儿运行。"""

        return list(self.cleaned_orphans)


@dataclass(frozen=True)
class _FakeGovernor:
    """测试用并发治理器。"""

    stale_permits: tuple[str, ...] = ("permit_1",)

    def cleanup_stale_permits(self) -> list[str]:
        """清理过期 permit。"""

        return list(self.stale_permits)

    def get_all_status(self) -> dict[str, LaneStatus]:
        """返回通道状态。"""

        return {
            "llm_api": LaneStatus(lane="llm_api", max_concurrent=4, active=1),
        }


def _build_service() -> tuple[HostAdminService, AsyncQueueEventBus]:
    """构建测试服务。"""

    session_registry = _FakeSessionRegistry()
    run_registry = _FakeRunRegistry()
    event_bus = AsyncQueueEventBus(run_registry=run_registry)  # type: ignore[arg-type]
    host = Host(
        executor=SimpleNamespace(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        concurrency_governor=_FakeGovernor(),  # type: ignore[arg-type]
        event_bus=event_bus,
    )
    return HostAdminService(host=host), event_bus


@pytest.mark.unit
def test_host_admin_service_lists_and_closes_sessions() -> None:
    """管理服务应能列出并关闭会话。"""

    service, _event_bus = _build_service()

    sessions = service.list_sessions(state="active")
    closed_session, cancelled_run_ids = service.close_session("session_1")

    assert [session.session_id for session in sessions] == ["session_1"]
    assert closed_session.state == "closed"
    assert cancelled_run_ids == ["run_1"]

    run = service.get_run("run_1")
    assert run is not None
    assert run.cancel_requested_at is not None
    assert run.cancel_requested_reason == "user_cancelled"
    assert run.cancel_reason is None


@pytest.mark.unit
def test_host_admin_service_lists_runs_and_builds_status() -> None:
    """管理服务应能列出运行并汇总宿主状态。"""

    service, _event_bus = _build_service()

    runs = service.list_runs(active_only=True)
    status = service.get_status()
    cleanup = service.cleanup()

    assert [run.run_id for run in runs] == ["run_1"]
    assert runs[0].cancel_reason is None
    assert status.active_session_count == 1
    assert status.active_run_count == 1
    assert status.active_runs_by_type == {"prompt": 1}
    assert status.lane_statuses["llm_api"].max_concurrent == 4
    assert cleanup.orphan_run_ids == ("run_orphan_1",)
    assert cleanup.stale_permit_ids == ("permit_1",)


@pytest.mark.unit
def test_recover_host_startup_state_logs_cleanup_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """统一 startup recovery helper 应记录清理结果。"""

    service, _event_bus = _build_service()
    info_logs: list[str] = []
    monkeypatch.setattr("dayu.services.startup_recovery.Log.info", lambda message, *, module="APP": info_logs.append(message))

    result = recover_host_startup_state(
        service,
        runtime_label="CLI Host runtime",
        log_module="APP.MAIN",
    )

    assert result.orphan_run_ids == ("run_orphan_1",)
    assert result.stale_permit_ids == ("permit_1",)
    assert any("CLI Host runtime 启动恢复完成 orphan_runs=1 stale_permits=1" in message for message in info_logs)


@pytest.mark.unit
def test_recover_host_startup_state_warns_and_continues_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """统一 startup recovery helper 失败时应只告警并返回空结果。"""

    warnings: list[str] = []
    monkeypatch.setattr(
        "dayu.services.startup_recovery.Log.warning",
        lambda message, *, module="APP": warnings.append(message),
    )

    class _FailingHostAdminService:
        def cleanup(self) -> HostCleanupResult:
            raise RuntimeError("cleanup failed")

    result = recover_host_startup_state(
        _FailingHostAdminService(),  # type: ignore[arg-type]
        runtime_label="WeChat Host runtime",
        log_module="APP.WECHAT.MAIN",
    )

    assert result.orphan_run_ids == ()
    assert result.stale_permit_ids == ()
    assert any("WeChat Host runtime 启动恢复失败，将继续启动: cleanup failed" in message for message in warnings)


@pytest.mark.unit
def test_host_admin_service_rejects_invalid_session_source() -> None:
    """管理服务创建 session 时不应把非法来源静默降级成 web。"""

    service, _event_bus = _build_service()

    with pytest.raises(ValueError):
        service.create_session(source="bad-source")


@pytest.mark.unit
def test_host_admin_service_wraps_event_bus_subscription() -> None:
    """管理服务应把 Host event bus 包装成事件流。"""

    service, event_bus = _build_service()

    async def _collect() -> PublishedRunEventProtocol:
        stream = service.subscribe_run_events("run_1")
        publish_task = asyncio.create_task(_publish_later(event_bus))
        try:
            async for event in stream:
                return event
        finally:
            await publish_task
        raise AssertionError("未收到事件")

    event = asyncio.run(_collect())

    assert event.type == AppEventType.DONE
    assert event.payload == {"ok": True}


async def _publish_later(event_bus: AsyncQueueEventBus) -> None:
    """异步发布一条测试事件。"""

    await asyncio.sleep(0)
    event_bus.publish(
        "run_1",
        AppEvent(type=AppEventType.DONE, payload={"ok": True}, meta={}),
    )


@pytest.mark.unit
def test_create_session_with_valid_source() -> None:
    """create_session 使用合法来源时应返回新会话视图。"""

    service, _event_bus = _build_service()

    view = service.create_session(source="web", scene_name="test_scene")

    assert view.session_id == "session_created"
    assert view.source == "web"
    assert view.state == "active"
    assert view.scene_name == "test_scene"


@pytest.mark.unit
def test_list_sessions_with_state_none() -> None:
    """list_sessions 不传 state 时应返回全部会话。"""

    service, _event_bus = _build_service()

    sessions = service.list_sessions(state=None)

    assert len(sessions) == 1
    assert sessions[0].session_id == "session_1"


@pytest.mark.unit
def test_list_runs_with_state_filter() -> None:
    """list_runs 传 state 参数时应使用 _parse_run_state 解析。"""

    service, _event_bus = _build_service()

    running_runs = service.list_runs(state="running")
    assert all(run.state == "running" for run in running_runs)

    succeeded_runs = service.list_runs(state="succeeded")
    assert all(run.state == "succeeded" for run in succeeded_runs)


@pytest.mark.unit
def test_list_runs_without_state_returns_all() -> None:
    """list_runs 不传 state 时应跳过状态过滤。"""

    service, _event_bus = _build_service()

    all_runs = service.list_runs()
    assert len(all_runs) == 2


@pytest.mark.unit
def test_list_runs_active_only_with_session_id_and_service_type() -> None:
    """list_runs active_only=True 时按 session_id 和 service_type 过滤。"""

    service, _event_bus = _build_service()

    # active_only + session_id 过滤
    runs_by_session = service.list_runs(active_only=True, session_id="session_1")
    assert [run.run_id for run in runs_by_session] == ["run_1"]

    # active_only + session_id 匹配不到
    runs_empty = service.list_runs(active_only=True, session_id="nonexistent")
    assert runs_empty == []

    # active_only + service_type 过滤
    runs_by_type = service.list_runs(active_only=True, service_type="prompt")
    assert [run.run_id for run in runs_by_type] == ["run_1"]

    # active_only + service_type 匹配不到
    runs_empty2 = service.list_runs(active_only=True, service_type="other")
    assert runs_empty2 == []


@pytest.mark.unit
def test_get_session_returns_none_for_missing() -> None:
    """get_session 找不到会话时应返回 None。"""

    service, _event_bus = _build_service()

    result = service.get_session("nonexistent_session")

    assert result is None


@pytest.mark.unit
def test_get_session_returns_view_for_existing() -> None:
    """get_session 找到会话时应返回视图。"""

    service, _event_bus = _build_service()

    result = service.get_session("session_1")

    assert result is not None
    assert result.session_id == "session_1"


@pytest.mark.unit
def test_get_run_returns_none_for_missing() -> None:
    """get_run 找不到运行时应返回 None。"""

    service, _event_bus = _build_service()

    result = service.get_run("nonexistent_run")

    assert result is None


@pytest.mark.unit
def test_cancel_run_returns_run_admin_view() -> None:
    """cancel_run 应返回更新后的 RunAdminView。"""

    service, _event_bus = _build_service()

    result = service.cancel_run("run_1")

    assert result.run_id == "run_1"
    assert result.cancel_requested_at is not None
    assert result.cancel_requested_reason == "user_cancelled"


@pytest.mark.unit
def test_cancel_session_runs_returns_cancelled_ids() -> None:
    """cancel_session_runs 应返回被取消的 run_id 列表。"""

    service, _event_bus = _build_service()

    cancelled = service.cancel_session_runs("session_1")

    assert cancelled == ["run_1"]


@pytest.mark.unit
def test_close_session_cleans_outbox_and_pending_turns() -> None:
    """close_session 应清理该 session 的 reply outbox 和 pending turns。"""

    from dayu.contracts.reply_outbox import ReplyOutboxSubmitRequest
    from dayu.host.pending_turn_store import (
        InMemoryPendingConversationTurnStore,
        PendingConversationTurnState,
    )
    from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore

    pending_store = InMemoryPendingConversationTurnStore()
    outbox_store = InMemoryReplyOutboxStore()

    session_registry = _FakeSessionRegistry()
    run_registry = _FakeRunRegistry()
    event_bus = AsyncQueueEventBus(run_registry=run_registry)  # type: ignore[arg-type]
    host = Host(
        executor=SimpleNamespace(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        concurrency_governor=_FakeGovernor(),  # type: ignore[arg-type]
        event_bus=event_bus,
        pending_turn_store=pending_store,
        reply_outbox_store=outbox_store,
    )

    # 模拟 session_1 有一条 outbox 记录
    outbox_store.submit_reply(ReplyOutboxSubmitRequest(
        delivery_key="dk_1",
        session_id="session_1",
        scene_name="test_scene",
        source_run_id="run_1",
        reply_content="hello",
    ))
    assert len(outbox_store.list_replies(session_id="session_1")) == 1

    # 模拟 session_1 有一条 pending turn
    pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="test_scene",
        user_text="test question",
        source_run_id="run_1",
        resumable=False,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    assert pending_store.get_session_pending_turn(session_id="session_1", scene_name="test_scene") is not None

    service = HostAdminService(host=host)
    service.close_session("session_1")

    # 验证 outbox 和 pending turns 已清理
    assert len(outbox_store.list_replies(session_id="session_1")) == 0
    assert pending_store.get_session_pending_turn(session_id="session_1", scene_name="test_scene") is None
    """subscribe_session_events 应把 Host event bus 包装成事件流。"""

    service, event_bus = _build_service()

    async def _collect() -> PublishedRunEventProtocol:
        stream = service.subscribe_session_events("session_1")
        publish_task = asyncio.create_task(_publish_session_event(event_bus))
        try:
            async for event in stream:
                return event
        finally:
            await publish_task
        raise AssertionError("未收到事件")

    event = asyncio.run(_collect())

    assert event.type == AppEventType.DONE
    assert event.payload == {"session_ok": True}


async def _publish_session_event(event_bus: AsyncQueueEventBus) -> None:
    """异步发布一条 session 级别测试事件。"""

    await asyncio.sleep(0)
    event_bus.publish(
        "run_1",
        AppEvent(type=AppEventType.DONE, payload={"session_ok": True}, meta={}),
    )


@pytest.mark.unit
def test_stream_subscription_events_closes_on_completion() -> None:
    """_stream_subscription_events 在迭代结束时应调用 subscription.close。"""

    class _EmptySubscription:
        """立即结束的空订阅，用于验证 finally 中 close 调用。"""

        def __init__(self) -> None:
            self.close_called = False

        @property
        def is_closed(self) -> bool:
            return self.close_called

        def close(self) -> None:
            self.close_called = True

        def __aiter__(self) -> AsyncIterator[PublishedRunEventProtocol]:
            """返回自身作为迭代器。"""
            return self

        async def __anext__(self) -> PublishedRunEventProtocol:
            """立即抛出 StopAsyncIteration。"""
            raise StopAsyncIteration

    from dayu.services.host_admin_service import _stream_subscription_events

    sub = _EmptySubscription()

    async def _drain() -> None:
        stream = _stream_subscription_events(sub)  # type: ignore[arg-type]
        async for _event in stream:
            pass  # pragma: no cover

    asyncio.run(_drain())

    assert sub.close_called is True
