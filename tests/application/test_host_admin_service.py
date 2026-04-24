"""HostAdminService 测试。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast

import pytest

from dayu.contracts.events import AppEvent, AppEventType, PublishedRunEventProtocol
from dayu.contracts.run import RunCancelReason, RunRecord, RunState
from dayu.contracts.session import SessionRecord, SessionSource, SessionState
from dayu.host.event_bus import AsyncQueueEventBus
from dayu.host.host import Host
from dayu.host.protocols import (
    ConversationSessionDigest,
    ConversationSessionTurnExcerpt,
    HostAdminOperationsProtocol,
    LaneStatus,
)
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

    def list_sessions(
        self,
        *,
        state: SessionState | None = None,
        source: SessionSource | None = None,
        scene_name: str | None = None,
    ) -> list[SessionRecord]:
        """列出会话。"""

        del source
        del scene_name
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

    def is_session_active(self, session_id: str) -> bool:
        """查询 session 是否仍处于非 CLOSED 状态。"""

        record = self.records.get(session_id)
        if record is None:
            return False
        return record.state != SessionState.CLOSED


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
    assert sessions[0].turn_count == 0
    assert sessions[0].first_question_preview == ""
    assert closed_session.state == "closed"
    assert closed_session.turn_count == 0
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
def test_host_admin_service_status_uses_single_session_listing() -> None:
    """状态汇总应基于一次 session 列表结果同时计算总数和活跃数。"""

    list_sessions_calls: list[tuple[SessionState | None, SessionSource | None, str | None]] = []

    class _Host:
        """记录 `list_sessions()` 调用次数的最小 Host。"""

        def list_sessions(
            self,
            *,
            state: SessionState | None = None,
            source: SessionSource | None = None,
            scene_name: str | None = None,
        ) -> list[SessionRecord]:
            """返回固定 session 列表。"""

            list_sessions_calls.append((state, source, scene_name))
            now = datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc)
            return [
                SessionRecord(
                    session_id="active_1",
                    source=SessionSource.WEB,
                    state=SessionState.ACTIVE,
                    scene_name="prompt",
                    created_at=now,
                    last_activity_at=now,
                ),
                SessionRecord(
                    session_id="closed_1",
                    source=SessionSource.WEB,
                    state=SessionState.CLOSED,
                    scene_name="prompt",
                    created_at=now,
                    last_activity_at=now,
                ),
            ]

        def list_active_runs(self) -> list[RunRecord]:
            """返回空运行列表。"""

            return []

        def get_all_lane_statuses(self) -> dict[str, LaneStatus]:
            """返回空 lane 状态。"""

            return {}

    service = HostAdminService(host=cast(HostAdminOperationsProtocol, _Host()))

    status = service.get_status()

    assert status.active_session_count == 1
    assert status.total_session_count == 2
    assert list_sessions_calls == [(None, None, None)]


@pytest.mark.unit
def test_host_admin_service_lists_sessions_with_state_source_scene_filters() -> None:
    """管理服务应支持 state/source/scene 组合过滤并返回 digest 视图。"""

    now = datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc)
    digest_calls: list[str] = []

    class _Host:
        """测试用最小 Host 管理面。"""

        def list_sessions(
            self,
            *,
            state: SessionState | None = None,
            source: SessionSource | None = None,
            scene_name: str | None = None,
        ) -> list[SessionRecord]:
            """返回多种来源的 session。"""

            del source
            del scene_name
            records = [
                SessionRecord(
                    session_id="interactive_1",
                    source=SessionSource.CLI,
                    state=SessionState.ACTIVE,
                    scene_name="interactive",
                    created_at=now,
                    last_activity_at=now,
                ),
                SessionRecord(
                    session_id="prompt_1",
                    source=SessionSource.CLI,
                    state=SessionState.ACTIVE,
                    scene_name="prompt_mt",
                    created_at=now,
                    last_activity_at=now,
                ),
                SessionRecord(
                    session_id="wechat_1",
                    source=SessionSource.WECHAT,
                    state=SessionState.ACTIVE,
                    scene_name="interactive",
                    created_at=now,
                    last_activity_at=now,
                ),
                SessionRecord(
                    session_id="interactive_closed",
                    source=SessionSource.CLI,
                    state=SessionState.CLOSED,
                    scene_name="interactive",
                    created_at=now,
                    last_activity_at=now,
                ),
            ]
            if state is None:
                return records
            return [record for record in records if record.state == state]

        def get_conversation_session_digest(self, session_id: str) -> ConversationSessionDigest:
            """返回指定 session 的 conversation 摘要。"""

            digest_calls.append(session_id)
            return ConversationSessionDigest(
                turn_count=2,
                first_question_preview=f"{session_id}-first",
                last_question_preview=f"{session_id}-last",
            )

    service = HostAdminService(host=cast(HostAdminOperationsProtocol, _Host()))

    sessions = service.list_sessions(
        state="active",
        source="cli",
        scene="interactive",
    )
    source_only_sessions = service.list_sessions(source="cli")
    scene_only_sessions = service.list_sessions(scene="interactive")
    closed_sessions = service.list_sessions(state="closed", scene="interactive")

    assert [session.session_id for session in sessions] == ["interactive_1"]
    assert [session.session_id for session in source_only_sessions] == [
        "interactive_1",
        "prompt_1",
        "interactive_closed",
    ]
    assert [session.session_id for session in scene_only_sessions] == [
        "interactive_1",
        "wechat_1",
        "interactive_closed",
    ]
    assert [session.session_id for session in closed_sessions] == ["interactive_closed"]
    assert sessions[0].turn_count == 2
    assert sessions[0].source == "cli"
    assert sessions[0].scene_name == "interactive"
    assert sessions[0].first_question_preview == "interactive_1-first"
    assert sessions[0].last_question_preview == "interactive_1-last"
    assert digest_calls == [
        "interactive_1",
        "interactive_1",
        "prompt_1",
        "interactive_closed",
        "interactive_1",
        "wechat_1",
        "interactive_closed",
        "interactive_closed",
    ]


@pytest.mark.unit
def test_host_admin_service_returns_digest_for_sessions_without_transcript() -> None:
    """管理服务应在 transcript 缺失时稳定返回空 digest 字段。"""

    service, _event_bus = _build_service()

    sessions = service.list_sessions(state="active")
    session = service.get_session("session_1")

    assert len(sessions) == 1
    assert sessions[0].turn_count == 0
    assert sessions[0].first_question_preview == ""
    assert sessions[0].last_question_preview == ""
    assert session is not None
    assert session.turn_count == 0
    assert session.first_question_preview == ""


@pytest.mark.unit
def test_host_admin_service_formats_missing_timestamps_as_empty_string() -> None:
    """管理视图中的空时间字段应输出为空字符串而不是字符串 `None`。"""

    class _Host:
        """返回缺失时间戳 session 的最小 Host。"""

        def list_sessions(
            self,
            *,
            state: SessionState | None = None,
            source: SessionSource | None = None,
            scene_name: str | None = None,
        ) -> list[SessionRecord]:
            """返回单条缺失时间戳的 session。"""

            del state
            del source
            del scene_name
            return [
                SessionRecord(
                    session_id="session_without_timestamps",
                    source=SessionSource.WEB,
                    state=SessionState.ACTIVE,
                    scene_name="prompt",
                    created_at=cast(datetime, None),
                    last_activity_at=cast(datetime, None),
                )
            ]

        def get_conversation_session_digest(self, _session_id: str) -> ConversationSessionDigest:
            """返回空 conversation digest。"""

            return ConversationSessionDigest(
                turn_count=0,
                first_question_preview="",
                last_question_preview="",
            )

    service = HostAdminService(host=cast(HostAdminOperationsProtocol, _Host()))

    sessions = service.list_sessions()

    assert len(sessions) == 1
    assert sessions[0].created_at == ""
    assert sessions[0].last_activity_at == ""


@pytest.mark.unit
def test_host_admin_service_lists_session_recent_turns() -> None:
    """管理服务应返回任意已存在会话的最近对话轮次。"""

    class _Host:
        """测试用最小 Host 管理面。"""

        def get_session(self, session_id: str) -> SessionRecord | None:
            """返回指定 session。"""

            assert session_id == "prompt_mt_1"
            now = datetime(2026, 4, 21, 8, 0, tzinfo=timezone.utc)
            return SessionRecord(
                session_id="prompt_mt_1",
                source=SessionSource.CLI,
                state=SessionState.CLOSED,
                scene_name="prompt_mt",
                created_at=now,
                last_activity_at=now,
            )

        def list_conversation_session_turn_excerpts(
            self,
            session_id: str,
            *,
            limit: int,
        ) -> list[ConversationSessionTurnExcerpt]:
            """返回指定 session 的最近 turn 摘录。"""

            assert session_id == "prompt_mt_1"
            assert limit == 1
            return [
                ConversationSessionTurnExcerpt(
                    user_text="上一轮问题",
                    assistant_text="上一轮回答",
                    created_at="2026-04-21T00:01:00+00:00",
                )
            ]

    service = HostAdminService(host=cast(HostAdminOperationsProtocol, _Host()))

    turns = service.list_session_recent_turns("prompt_mt_1", limit=1)

    assert len(turns) == 1
    assert turns[0].user_text == "上一轮问题"
    assert turns[0].assistant_text == "上一轮回答"


@pytest.mark.unit
def test_host_admin_service_returns_empty_recent_turns_for_missing_session() -> None:
    """管理服务在会话不存在时应稳定返回空摘录。"""

    class _Host:
        """测试用最小 Host 管理面。"""

        def get_session(self, session_id: str) -> SessionRecord | None:
            """返回缺失会话。"""

            assert session_id == "missing_1"
            return None

        def list_conversation_session_turn_excerpts(
            self,
            session_id: str,
            *,
            limit: int,
        ) -> list[ConversationSessionTurnExcerpt]:
            """缺失会话不应触达 transcript 摘录。"""

            raise AssertionError("缺失会话不应读取 conversation 摘录")

    service = HostAdminService(host=cast(HostAdminOperationsProtocol, _Host()))

    assert service.list_session_recent_turns("missing_1", limit=1) == []


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
    assert view.turn_count == 0
    assert view.first_question_preview == ""


@pytest.mark.unit
def test_list_sessions_with_state_none() -> None:
    """list_sessions 不传 state 时应返回全部会话 digest。"""

    service, _event_bus = _build_service()

    sessions = service.list_sessions(state=None)

    assert len(sessions) == 1
    assert sessions[0].session_id == "session_1"
    assert sessions[0].turn_count == 0


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


@pytest.mark.unit
def test_cleanup_stale_pending_turns_removes_turns_with_terminal_runs() -> None:
    """关联 run 已终态、且按调和规则应删除的 pending turn 应被清理。

    回归 review 条目 S-5.10-4：Host 进程崩溃或启动调和路径异常时可能漏调
    `_reconcile_pending_turn_after_terminal_run`，需要独立的启动清理兜底。
    """

    from dayu.host.pending_turn_store import (
        InMemoryPendingConversationTurnStore,
        PendingConversationTurnState,
    )
    from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore

    pending_store = InMemoryPendingConversationTurnStore()
    outbox_store = InMemoryReplyOutboxStore()
    session_registry = _FakeSessionRegistry()
    run_registry = _FakeRunRegistry()

    now = datetime.now(timezone.utc)
    # run_succ：成功终态 → 关联 pending turn 应清理
    run_registry.records["run_succ"] = RunRecord(
        run_id="run_succ",
        session_id="session_1",
        service_type="chat_turn",
        scene_name="chat",
        state=RunState.SUCCEEDED,
        created_at=now,
        started_at=now,
        completed_at=now,
    )
    # run_failed_resumable：失败且 resumable → 保留
    run_registry.records["run_failed_resumable"] = RunRecord(
        run_id="run_failed_resumable",
        session_id="session_1",
        service_type="chat_turn",
        scene_name="chat",
        state=RunState.FAILED,
        created_at=now,
        started_at=now,
        completed_at=now,
    )
    # run_1 已在 fixture 中为 RUNNING → 保留

    pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="chat",
        user_text="succ",
        source_run_id="run_succ",
        resumable=False,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )
    pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="chat_a",
        user_text="failed-resumable",
        source_run_id="run_failed_resumable",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )
    pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="chat_b",
        user_text="still-running",
        source_run_id="run_1",
        resumable=False,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )
    pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="chat_c",
        user_text="missing-run",
        source_run_id="run_missing",
        resumable=False,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )

    host = Host(
        executor=SimpleNamespace(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        concurrency_governor=_FakeGovernor(),  # type: ignore[arg-type]
        event_bus=AsyncQueueEventBus(run_registry=run_registry),  # type: ignore[arg-type]
        pending_turn_store=pending_store,
        reply_outbox_store=outbox_store,
    )

    cleaned = host.cleanup_stale_pending_turns()

    remaining_scenes = {
        record.scene_name for record in pending_store.list_pending_turns()
    }
    # chat (succ) + chat_c (missing run) 被清理；chat_a + chat_b 保留
    assert remaining_scenes == {"chat_a", "chat_b"}
    assert len(cleaned) == 2


@pytest.mark.unit
def test_cleanup_stale_pending_turns_retention_keeps_active_source_run() -> None:
    """source_run 仍活跃时，即使 updated_at 超过保留期也严格保留 pending turn。

    回归 Finding 071 review：分支 C 不得误删长任务 / 外部阻塞 / 人工调试场景下
    仍在执行链路上的恢复真源。一旦该 run 随后失败或超时，本应可 resume 的
    pending turn 必须仍然存在。
    """

    from dataclasses import replace
    from datetime import timedelta

    from dayu.host.pending_turn_store import (
        InMemoryPendingConversationTurnStore,
        PendingConversationTurnState,
    )
    from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore

    pending_store = InMemoryPendingConversationTurnStore()
    outbox_store = InMemoryReplyOutboxStore()
    session_registry = _FakeSessionRegistry()
    run_registry = _FakeRunRegistry()

    now = datetime.now(timezone.utc)
    # _FakeRunRegistry fixture 里 run_1 已是 RUNNING（活跃态）。
    record = pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="chat_active_long",
        user_text="long-running",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )
    # 手工把 updated_at 设成 30 天前，远超 168 小时保留期。
    pending_store._records[record.pending_turn_id] = replace(
        pending_store._records[record.pending_turn_id],
        updated_at=now - timedelta(days=30),
    )

    host = Host(
        executor=SimpleNamespace(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        concurrency_governor=_FakeGovernor(),  # type: ignore[arg-type]
        event_bus=AsyncQueueEventBus(run_registry=run_registry),  # type: ignore[arg-type]
        pending_turn_store=pending_store,
        reply_outbox_store=outbox_store,
        pending_turn_retention_hours=168,
    )

    cleaned = host.cleanup_stale_pending_turns()

    # 活跃 run 对应的 pending turn 必须完整保留，不得被 retention cleanup 吞掉。
    assert cleaned == []
    remaining = pending_store._records.get(record.pending_turn_id)
    assert remaining is not None
    assert remaining.state is PendingConversationTurnState.PREPARED_BY_HOST


@pytest.mark.unit
def test_cleanup_stale_pending_turns_expires_after_retention() -> None:
    """pending turn 在 ACCEPTED/PREPARED 状态下超过保留期应被兜底删除。

    回归 Finding 071：source_run 终态为 FAILED+resumable / CANCELLED+TIMEOUT+resumable
    时分支 B 判为保留；若用户始终不触发 resume，需要 Host 层按保留期兜底清理，
    避免库无限累积。
    """

    from dataclasses import replace
    from datetime import timedelta

    from dayu.host.pending_turn_store import (
        InMemoryPendingConversationTurnStore,
        PendingConversationTurnState,
    )
    from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore

    pending_store = InMemoryPendingConversationTurnStore()
    outbox_store = InMemoryReplyOutboxStore()
    session_registry = _FakeSessionRegistry()
    run_registry = _FakeRunRegistry()

    now = datetime.now(timezone.utc)
    run_registry.records["run_failed_resumable"] = RunRecord(
        run_id="run_failed_resumable",
        session_id="session_1",
        service_type="chat_turn",
        scene_name="chat",
        state=RunState.FAILED,
        created_at=now,
        started_at=now,
        completed_at=now,
    )
    run_registry.records["run_timeout_resumable"] = RunRecord(
        run_id="run_timeout_resumable",
        session_id="session_1",
        service_type="chat_turn",
        scene_name="chat",
        state=RunState.CANCELLED,
        cancel_reason=RunCancelReason.TIMEOUT,
        created_at=now,
        started_at=now,
        completed_at=now,
    )

    # 构造两条 PREPARED_BY_HOST 记录：一条 8 天前更新，一条 1 天前更新。
    pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="chat_stale",
        user_text="stale-failed",
        source_run_id="run_failed_resumable",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )
    pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="chat_stale_timeout",
        user_text="stale-timeout",
        source_run_id="run_timeout_resumable",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )
    pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="chat_fresh",
        user_text="fresh",
        source_run_id="run_failed_resumable",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )

    # 手工回调 updated_at：frozen dataclass 用 replace。
    stale_before = now - timedelta(days=8)
    fresh_before = now - timedelta(days=1)
    for record in list(pending_store._records.values()):
        if record.scene_name == "chat_stale":
            pending_store._records[record.pending_turn_id] = replace(
                record, updated_at=stale_before
            )
        elif record.scene_name == "chat_stale_timeout":
            pending_store._records[record.pending_turn_id] = replace(
                record, updated_at=stale_before
            )
        elif record.scene_name == "chat_fresh":
            pending_store._records[record.pending_turn_id] = replace(
                record, updated_at=fresh_before
            )

    host = Host(
        executor=SimpleNamespace(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        concurrency_governor=_FakeGovernor(),  # type: ignore[arg-type]
        event_bus=AsyncQueueEventBus(run_registry=run_registry),  # type: ignore[arg-type]
        pending_turn_store=pending_store,
        reply_outbox_store=outbox_store,
        pending_turn_retention_hours=168,
    )

    cleaned = host.cleanup_stale_pending_turns()

    remaining_scenes = {
        record.scene_name for record in pending_store.list_pending_turns()
    }
    # 两条 8 天前的应被兜底删除；1 天前的保留等待 UI 询问。
    assert remaining_scenes == {"chat_fresh"}
    assert len(cleaned) == 2


@pytest.mark.unit
def test_cleanup_stale_pending_turns_retention_respects_resuming_first() -> None:
    """RESUMING 状态即使 updated_at 超过保留期，也应走分支 A 回退 lease 而非直接删除。

    回归 Finding 071 分支顺序约束：分支 A（RESUMING lease 回退）严格优先于
    分支 C（超保留期删除），避免把 070 lease 机制保护的记录直接丢弃。
    """

    from dataclasses import replace
    from datetime import timedelta

    from dayu.host.pending_turn_store import (
        InMemoryPendingConversationTurnStore,
        PendingConversationTurnState,
    )
    from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore

    pending_store = InMemoryPendingConversationTurnStore()
    outbox_store = InMemoryReplyOutboxStore()
    session_registry = _FakeSessionRegistry()
    run_registry = _FakeRunRegistry()

    now = datetime.now(timezone.utc)
    record = pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="chat_resuming",
        user_text="resuming",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )
    # 先 acquire lease 进入 RESUMING，再手工把 updated_at 设置成 30 天前。
    pending_store.record_resume_attempt(record.pending_turn_id, max_attempts=3)
    after_acquire = pending_store._records[record.pending_turn_id]
    assert after_acquire.state is PendingConversationTurnState.RESUMING
    pending_store._records[record.pending_turn_id] = replace(
        after_acquire, updated_at=now - timedelta(days=30)
    )

    host = Host(
        executor=SimpleNamespace(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        concurrency_governor=_FakeGovernor(),  # type: ignore[arg-type]
        event_bus=AsyncQueueEventBus(run_registry=run_registry),  # type: ignore[arg-type]
        pending_turn_store=pending_store,
        reply_outbox_store=outbox_store,
        pending_turn_retention_hours=168,
    )

    cleaned = host.cleanup_stale_pending_turns()

    # 记录应仍然存在，且 state 已回退到 PREPARED_BY_HOST（lease 释放），不是直接删除。
    remaining = pending_store._records.get(record.pending_turn_id)
    assert remaining is not None
    assert remaining.state is PendingConversationTurnState.PREPARED_BY_HOST
    assert record.pending_turn_id not in cleaned


@pytest.mark.unit
def test_cleanup_stale_pending_turns_retention_keeps_recent() -> None:
    """保留期内的 pending turn 不得被分支 C 清理。

    回归 Finding 071：默认保留期 168 小时，1 天前更新的记录应完整保留给 UI 询问。
    """

    from dataclasses import replace
    from datetime import timedelta

    from dayu.host.pending_turn_store import (
        InMemoryPendingConversationTurnStore,
        PendingConversationTurnState,
    )
    from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore

    pending_store = InMemoryPendingConversationTurnStore()
    outbox_store = InMemoryReplyOutboxStore()
    session_registry = _FakeSessionRegistry()
    run_registry = _FakeRunRegistry()

    now = datetime.now(timezone.utc)
    run_registry.records["run_failed_resumable"] = RunRecord(
        run_id="run_failed_resumable",
        session_id="session_1",
        service_type="chat_turn",
        scene_name="chat",
        state=RunState.FAILED,
        created_at=now,
        started_at=now,
        completed_at=now,
    )
    record = pending_store.upsert_pending_turn(
        session_id="session_1",
        scene_name="chat_fresh",
        user_text="fresh",
        source_run_id="run_failed_resumable",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )
    pending_store._records[record.pending_turn_id] = replace(
        pending_store._records[record.pending_turn_id],
        updated_at=now - timedelta(hours=24),
    )

    host = Host(
        executor=SimpleNamespace(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        concurrency_governor=_FakeGovernor(),  # type: ignore[arg-type]
        event_bus=AsyncQueueEventBus(run_registry=run_registry),  # type: ignore[arg-type]
        pending_turn_store=pending_store,
        reply_outbox_store=outbox_store,
        pending_turn_retention_hours=168,
    )

    cleaned = host.cleanup_stale_pending_turns()

    assert cleaned == []
    assert pending_store._records.get(record.pending_turn_id) is not None
