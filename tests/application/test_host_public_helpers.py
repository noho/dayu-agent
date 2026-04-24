"""Host 公共辅助模块覆盖测试。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import pytest

from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.contracts.session import SessionRecord, SessionSource, SessionState
from dayu.engine.events import EventType, StreamEvent
from dayu.host.events import build_app_event_from_stream_event
from dayu.host.startup_preparation import resolve_host_config
from dayu.services.contracts import SessionResolutionPolicy
from dayu.services.internal.session_coordinator import ServiceSessionCoordinator


def _build_session_record(session_id: str, *, scene_name: str | None = None) -> SessionRecord:
    """构造测试用 session 记录。"""

    now = datetime(2026, 4, 14, 12, 0, 0)
    return SessionRecord(
        session_id=session_id,
        source=SessionSource.WEB,
        state=SessionState.ACTIVE,
        scene_name=scene_name,
        created_at=now,
        last_activity_at=now,
    )


class _StubSessionHost:
    """实现 `SessionOperationsProtocol` 的最小测试桩。"""

    def __init__(self) -> None:
        """初始化测试会话存储。"""

        self.sessions: dict[str, SessionRecord] = {}
        self.calls: list[tuple[str, object]] = []

    def create_session(
        self,
        source: SessionSource,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """创建会话记录。"""

        del metadata
        created_id = session_id or f"created_{len(self.sessions) + 1}"
        record = SessionRecord(
            session_id=created_id,
            source=source,
            state=SessionState.ACTIVE,
            scene_name=scene_name,
            created_at=datetime(2026, 4, 14, 12, 0, 0),
            last_activity_at=datetime(2026, 4, 14, 12, 0, 0),
        )
        self.sessions[created_id] = record
        self.calls.append(("create", (created_id, source, scene_name)))
        return record

    def ensure_session(
        self,
        session_id: str,
        source: SessionSource,
        *,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """按确定性 ID 获取或创建会话。"""

        del metadata
        self.calls.append(("ensure", (session_id, source, scene_name)))
        existing = self.sessions.get(session_id)
        if existing is not None:
            return existing
        return self.create_session(source, session_id=session_id, scene_name=scene_name)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """获取会话。"""

        self.calls.append(("get", session_id))
        return self.sessions.get(session_id)

    def list_sessions(
        self,
        *,
        state: SessionState | None = None,
        source: SessionSource | None = None,
        scene_name: str | None = None,
    ) -> list[SessionRecord]:
        """列出会话。"""

        return [
            record
            for record in self.sessions.values()
            if (state is None or record.state == state)
            and (source is None or record.source == source)
            and (scene_name is None or record.scene_name == scene_name)
        ]

    def touch_session(self, session_id: str) -> None:
        """刷新会话活跃时间。"""

        self.calls.append(("touch", session_id))
        current = self.sessions[session_id]
        self.sessions[session_id] = SessionRecord(
            session_id=current.session_id,
            source=current.source,
            state=current.state,
            scene_name=current.scene_name,
            created_at=current.created_at,
            last_activity_at=current.last_activity_at + timedelta(minutes=1),
            metadata=current.metadata,
        )


@pytest.mark.unit
def test_build_app_event_from_stream_event_maps_supported_types() -> None:
    """事件映射应覆盖内容、工具和控制类事件。"""

    content_event = build_app_event_from_stream_event(
        StreamEvent(type=EventType.CONTENT_DELTA, data="hello", metadata={"seq": 1})
    )
    reasoning_event = build_app_event_from_stream_event(
        StreamEvent(type=EventType.REASONING_DELTA, data=None, metadata={"seq": 2})
    )
    final_event = build_app_event_from_stream_event(
        StreamEvent(type=EventType.FINAL_ANSWER, data="final text", metadata={"done": True})
    )
    tool_event = build_app_event_from_stream_event(
        StreamEvent(type=EventType.TOOL_CALL_RESULT, data={"value": 1}, metadata={"tool": "search"})
    )
    warning_event = build_app_event_from_stream_event(StreamEvent(type=EventType.WARNING, data="warn"))
    error_event = build_app_event_from_stream_event(StreamEvent(type=EventType.ERROR, data="error"))
    metadata_event = build_app_event_from_stream_event(StreamEvent(type=EventType.METADATA, data={"tokens": 12}))
    done_event = build_app_event_from_stream_event(StreamEvent(type=EventType.DONE, data={"ok": True}))

    assert content_event is not None and content_event.payload == "hello"
    assert reasoning_event is not None and reasoning_event.payload == ""
    assert final_event is not None and final_event.payload == {"content": "final text", "degraded": False}
    assert tool_event is not None and tool_event.payload["engine_event_type"] == EventType.TOOL_CALL_RESULT.value
    assert warning_event is not None and warning_event.payload == "warn"
    assert error_event is not None and error_event.payload == "error"
    assert metadata_event is not None and metadata_event.payload == {"tokens": 12}
    assert done_event is not None and done_event.payload == {"ok": True}


@pytest.mark.unit
def test_build_app_event_from_stream_event_handles_dict_final_answer_and_unknown_event() -> None:
    """最终答案字典应原样透传，未知事件应被忽略。"""

    passthrough = build_app_event_from_stream_event(
        StreamEvent(type=EventType.FINAL_ANSWER, data={"content": "ok", "degraded": True})
    )
    ignored = build_app_event_from_stream_event(StreamEvent(type=EventType.CONTENT_COMPLETE, data="done"))

    assert passthrough is not None and passthrough.payload == {"content": "ok", "degraded": True}
    assert ignored is None


@pytest.mark.unit
def test_resolve_host_config_supports_defaults_and_nested_overrides() -> None:
    """Host 配置解析应统一支持默认值、嵌套配置与显式 lane 覆盖。"""

    with TemporaryDirectory() as tmp_dir:
        workspace_root = Path(tmp_dir)

        default_config = resolve_host_config(workspace_root=workspace_root, run_config={})
        resolved = resolve_host_config(
            workspace_root=workspace_root,
            run_config={
                "host_config": {
                    "store": {"path": "runtime/host.sqlite"},
                    "lane": {"default": 2, "writer": 3},
                    "pending_turn_resume": {"max_attempts": 5},
                    "pending_turn_retention": {"retention_hours": 72},
                }
            },
            explicit_lane_config={"writer": 7},
        )

        assert default_config.store_path == (workspace_root / ".dayu/host/dayu_host.db").resolve()
        assert default_config.pending_turn_resume_max_attempts == 3
        assert default_config.pending_turn_retention_hours == 168
        assert resolved.store_path == (workspace_root / "runtime/host.sqlite").resolve()
        assert resolved.lane_config["default"] == 2
        assert resolved.lane_config["writer"] == 7
        assert resolved.pending_turn_resume_max_attempts == 5
        assert resolved.pending_turn_retention_hours == 72


@pytest.mark.unit
def test_resolve_host_config_rejects_legacy_and_invalid_shapes() -> None:
    """Host 配置解析应拒绝旧 key 与非法 `host_config` 结构。"""

    with TemporaryDirectory() as tmp_dir:
        workspace_root = Path(tmp_dir)

        with pytest.raises(TypeError, match="host_config"):
            resolve_host_config(
                workspace_root=workspace_root,
                run_config={"host_store_config": {"path": ".dayu/host/dayu_host.db"}},
            )
        with pytest.raises(TypeError, match="host_config 必须是对象"):
            resolve_host_config(workspace_root=workspace_root, run_config={"host_config": []})
        with pytest.raises(TypeError, match="host_config.store 必须是对象"):
            resolve_host_config(
                workspace_root=workspace_root,
                run_config={"host_config": {"store": []}},
            )
        with pytest.raises(TypeError, match="host_config.store.path 必须是字符串"):
            resolve_host_config(
                workspace_root=workspace_root,
                run_config={"host_config": {"store": {"path": 1}}},
            )
        with pytest.raises(TypeError, match="host_config.lane"):
            resolve_host_config(
                workspace_root=workspace_root,
                run_config={"host_config": {"lane": []}},
            )
        with pytest.raises(ValueError, match="必须是正整数"):
            resolve_host_config(
                workspace_root=workspace_root,
                run_config={"host_config": {"pending_turn_resume": {"max_attempts": 0}}},
            )
        with pytest.raises(TypeError, match="pending_turn_retention 必须是对象"):
            resolve_host_config(
                workspace_root=workspace_root,
                run_config={"host_config": {"pending_turn_retention": []}},
            )
        with pytest.raises(ValueError, match="retention_hours 必须是正整数"):
            resolve_host_config(
                workspace_root=workspace_root,
                run_config={
                    "host_config": {"pending_turn_retention": {"retention_hours": 0}}
                },
            )
        with pytest.raises(ValueError, match="retention_hours 必须是正整数"):
            resolve_host_config(
                workspace_root=workspace_root,
                run_config={
                    "host_config": {"pending_turn_retention": {"retention_hours": True}}
                },
            )


@pytest.mark.unit
def test_service_session_coordinator_routes_all_policies() -> None:
    """会话协调器应按策略路由到正确的 host 方法。"""

    host = _StubSessionHost()
    existing = _build_session_record("session_existing", scene_name="interactive")
    host.sessions[existing.session_id] = existing
    coordinator = ServiceSessionCoordinator(host=cast(Any, host), session_source=SessionSource.WEB)

    created = coordinator.resolve(session_id=None, scene_name="prompt", policy=SessionResolutionPolicy.AUTO)
    required = coordinator.resolve(
        session_id="session_existing",
        scene_name="interactive",
        policy=SessionResolutionPolicy.REQUIRE_EXISTING,
    )
    ensured = coordinator.resolve(
        session_id="session_det",
        scene_name="interactive",
        policy=SessionResolutionPolicy.ENSURE_DETERMINISTIC,
    )

    assert created.scene_name == "prompt"
    assert required.session_id == "session_existing"
    assert ensured.session_id == "session_det"
    assert any(call[0] == "touch" for call in host.calls)
    assert any(call[0] == "ensure" for call in host.calls)


@pytest.mark.unit
def test_service_session_coordinator_validates_invalid_policy_combinations() -> None:
    """会话协调器应拒绝非法策略与 session_id 组合。"""

    host = _StubSessionHost()
    coordinator = ServiceSessionCoordinator(host=cast(Any, host), session_source=SessionSource.WEB)

    with pytest.raises(ValueError):
        coordinator.resolve(
            session_id="session_1",
            scene_name=None,
            policy=SessionResolutionPolicy.CREATE_NEW,
        )
    with pytest.raises(ValueError):
        coordinator.resolve(
            session_id=None,
            scene_name=None,
            policy=SessionResolutionPolicy.REQUIRE_EXISTING,
        )
    with pytest.raises(ValueError):
        coordinator.resolve(
            session_id=None,
            scene_name=None,
            policy=SessionResolutionPolicy.ENSURE_DETERMINISTIC,
        )
    with pytest.raises(KeyError):
        coordinator.require_existing("missing")
    with pytest.raises(ValueError):
        coordinator.resolve(
            session_id=None,
            scene_name=None,
            policy=cast(SessionResolutionPolicy, "unknown"),
        )
