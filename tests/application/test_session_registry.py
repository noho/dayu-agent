"""SQLiteSessionRegistry 测试。"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest

from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.host.host_store import HostStore
from dayu.host.session_registry import SQLiteSessionRegistry
from dayu.contracts.session import SessionSource, SessionState
from dayu.log import Log


@pytest.fixture()
def registry(tmp_path: Path) -> SQLiteSessionRegistry:
    """创建一个临时 SessionRegistry。"""
    store = HostStore(tmp_path / "test.db")
    store.initialize_schema()
    return SQLiteSessionRegistry(store)


class TestCreateSession:
    """create_session 测试。"""

    @pytest.mark.unit
    def test_schema_does_not_define_ticker_column(self, registry: SQLiteSessionRegistry) -> None:
        """新建 schema 的 sessions 表不再包含 ticker 列。"""

        conn = registry._host_store.get_connection()
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        assert "ticker" not in columns

    @pytest.mark.unit
    def test_create_session_default_id(self, registry: SQLiteSessionRegistry) -> None:
        """不指定 session_id 时自动生成。"""
        session = registry.create_session(SessionSource.CLI)
        assert session.session_id
        assert session.source == SessionSource.CLI
        assert session.state == SessionState.ACTIVE
        assert session.scene_name is None
        assert session.metadata == {}

    @pytest.mark.unit
    def test_create_session_explicit_id(self, registry: SQLiteSessionRegistry) -> None:
        """指定 session_id。"""
        session = registry.create_session(
            SessionSource.API,
            session_id="my-session-123",
            scene_name="chat",
            metadata=ExecutionDeliveryContext({"chat_key": "value"}),
        )
        assert session.session_id == "my-session-123"
        assert session.source == SessionSource.API
        assert session.scene_name == "chat"
        assert session.metadata == {"chat_key": "value"}

    @pytest.mark.unit
    def test_create_session_timestamps(self, registry: SQLiteSessionRegistry) -> None:
        """创建后 created_at 和 last_activity_at 相等。"""
        session = registry.create_session(SessionSource.CLI)
        assert session.created_at == session.last_activity_at


class TestEnsureSession:
    """ensure_session 幂等测试。"""

    @pytest.mark.unit
    def test_create_if_not_exists(self, registry: SQLiteSessionRegistry) -> None:
        """session 不存在时创建。"""
        session = registry.ensure_session(
            "wechat-room-1",
            SessionSource.WECHAT,
        )
        assert session.session_id == "wechat-room-1"
        assert session.source == SessionSource.WECHAT

    @pytest.mark.unit
    def test_return_existing_if_exists(self, registry: SQLiteSessionRegistry) -> None:
        """session 已存在时返回现有记录并 touch。"""
        original = registry.create_session(
            SessionSource.WECHAT,
            session_id="wechat-room-1",
        )
        ensured = registry.ensure_session(
            "wechat-room-1",
            SessionSource.CLI,  # 源不同也不报错
        )
        assert ensured.session_id == original.session_id
        # source 保持创建时的值
        assert ensured.source == SessionSource.WECHAT


class TestGetAndListSessions:
    """查询和列表测试。"""

    @pytest.mark.unit
    def test_get_session_exists(self, registry: SQLiteSessionRegistry) -> None:
        """查询已存在的 session。"""
        created = registry.create_session(SessionSource.CLI)
        fetched = registry.get_session(created.session_id)
        assert fetched is not None
        assert fetched.session_id == created.session_id

    @pytest.mark.unit
    def test_get_session_not_exists(self, registry: SQLiteSessionRegistry) -> None:
        """查询不存在的 session 返回 None。"""
        assert registry.get_session("nonexistent") is None

    @pytest.mark.unit
    def test_list_sessions_all(self, registry: SQLiteSessionRegistry) -> None:
        """列出所有 session。"""
        registry.create_session(SessionSource.CLI, session_id="s1")
        registry.create_session(SessionSource.API, session_id="s2")
        sessions = registry.list_sessions()
        assert len(sessions) == 2

    @pytest.mark.unit
    def test_list_sessions_filter_by_state(self, registry: SQLiteSessionRegistry) -> None:
        """按状态过滤 session。"""
        registry.create_session(SessionSource.CLI, session_id="s1")
        registry.create_session(SessionSource.CLI, session_id="s2")
        registry.close_session("s1")

        active = registry.list_sessions(state=SessionState.ACTIVE)
        assert len(active) == 1
        assert active[0].session_id == "s2"

        closed = registry.list_sessions(state=SessionState.CLOSED)
        assert len(closed) == 1
        assert closed[0].session_id == "s1"

    @pytest.mark.unit
    def test_list_sessions_filter_by_source_and_scene_name(self, registry: SQLiteSessionRegistry) -> None:
        """按来源和 scene 名称过滤 session。"""

        registry.create_session(SessionSource.CLI, session_id="interactive_1", scene_name="interactive")
        registry.create_session(SessionSource.CLI, session_id="prompt_1", scene_name="prompt")
        registry.create_session(SessionSource.WECHAT, session_id="wechat_1", scene_name="interactive")

        sessions = registry.list_sessions(
            source=SessionSource.CLI,
            scene_name="interactive",
        )

        assert [session.session_id for session in sessions] == ["interactive_1"]


class TestTouchSession:
    """touch_session 测试。"""

    @pytest.mark.unit
    def test_touch_updates_activity(self, registry: SQLiteSessionRegistry) -> None:
        """touch 后 last_activity_at 更新。"""
        session = registry.create_session(SessionSource.CLI)
        original_activity = session.last_activity_at

        import time
        time.sleep(0.01)
        registry.touch_session(session.session_id)

        updated = registry.get_session(session.session_id)
        assert updated is not None
        assert updated.last_activity_at >= original_activity

    @pytest.mark.unit
    def test_touch_nonexistent_raises(self, registry: SQLiteSessionRegistry) -> None:
        """touch 不存在的 session 抛出 KeyError。"""
        with pytest.raises(KeyError, match="session 不存在"):
            registry.touch_session("nonexistent")


class TestCloseSession:
    """close_session 测试。"""

    @pytest.mark.unit
    def test_close_session(self, registry: SQLiteSessionRegistry) -> None:
        """关闭 session 后状态变为 CLOSED。"""
        session = registry.create_session(SessionSource.CLI)
        registry.close_session(session.session_id)

        closed = registry.get_session(session.session_id)
        assert closed is not None
        assert closed.state == SessionState.CLOSED

    @pytest.mark.unit
    def test_close_nonexistent_raises(self, registry: SQLiteSessionRegistry) -> None:
        """关闭不存在的 session 抛 KeyError。"""
        with pytest.raises(KeyError, match="session 不存在"):
            registry.close_session("nonexistent")


class TestCloseIdleSessions:
    """close_idle_sessions 测试。"""

    @pytest.mark.unit
    def test_close_idle_sessions(self, registry: SQLiteSessionRegistry) -> None:
        """空闲超阈值的 session 被关闭。"""
        # 创建两个 session
        registry.create_session(SessionSource.CLI, session_id="idle")
        registry.create_session(SessionSource.CLI, session_id="active")

        # touch active 使其不是最旧的（但实际 close_idle 用的是绝对时间）
        import time
        time.sleep(0.01)
        registry.touch_session("active")

        # 用一个负向阈值（cutoff 取到未来），保证两个 session 的 last_activity_at
        # 都 < cutoff，从而都被关闭。避免 timedelta(0) 在 Windows 微秒级时钟粒度下
        # 与 touch_session 取到同一时刻而产生的边界 flakiness。
        closed_ids = registry.close_idle_sessions(timedelta(microseconds=-1))
        assert set(closed_ids) == {"idle", "active"}

    @pytest.mark.unit
    def test_close_idle_with_large_threshold(self, registry: SQLiteSessionRegistry) -> None:
        """阈值足够大时没有 session 被关闭。"""
        registry.create_session(SessionSource.CLI, session_id="fresh")
        closed_ids = registry.close_idle_sessions(timedelta(hours=24))
        assert closed_ids == []


@pytest.mark.unit
def test_session_registry_emits_lifecycle_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SessionRegistry 关键生命周期动作应输出日志。"""

    store = HostStore(tmp_path / "test.db")
    store.initialize_schema()
    registry = SQLiteSessionRegistry(store)
    debug_mock = Mock()
    monkeypatch.setattr(Log, "debug", debug_mock)

    registry.create_session(SessionSource.CLI, session_id="s1", scene_name="interactive")
    registry.ensure_session("s1", SessionSource.CLI)
    registry.touch_session("s1")
    registry.close_session("s1")

    debug_messages = [call.args[0] for call in debug_mock.call_args_list]

    assert any("创建 session" in message for message in debug_messages)
    assert any("ensure session" in message for message in debug_messages)
    assert any("刷新 session 活跃时间" in message for message in debug_messages)
    assert any("关闭 session" in message for message in debug_messages)
