"""SessionRecord / SessionState / SessionSource 数据模型测试。"""

from __future__ import annotations

from datetime import datetime

import pytest

from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.contracts.session import SessionRecord, SessionSource, SessionState


class TestSessionSource:
    """SessionSource 枚举测试。"""

    @pytest.mark.unit
    def test_all_sources(self) -> None:
        """所有来源枚举值正确。"""
        assert SessionSource.CLI == "cli"
        assert SessionSource.WEB == "web"
        assert SessionSource.WECHAT == "wechat"
        assert SessionSource.GUI == "gui"
        assert SessionSource.API == "api"
        assert SessionSource.INTERNAL == "internal"


class TestSessionState:
    """SessionState 枚举测试。"""

    @pytest.mark.unit
    def test_state_values(self) -> None:
        """状态枚举值正确。"""
        assert SessionState.ACTIVE == "active"
        assert SessionState.CLOSED == "closed"


class TestSessionRecord:
    """SessionRecord 数据结构测试。"""

    @pytest.mark.unit
    def test_create_minimal(self) -> None:
        """最小字段创建。"""
        now = datetime.now()
        record = SessionRecord(
            session_id="sess_001",
            source=SessionSource.CLI,
            state=SessionState.ACTIVE,
            created_at=now,
            last_activity_at=now,
        )
        assert record.session_id == "sess_001"
        assert record.source == SessionSource.CLI
        assert record.is_active()
        assert record.scene_name is None

    @pytest.mark.unit
    def test_create_full(self) -> None:
        """全字段创建。"""
        now = datetime.now()
        record = SessionRecord(
            session_id="wechat_abc123",
            source=SessionSource.WECHAT,
            state=SessionState.ACTIVE,
            scene_name="wechat",
            created_at=now,
            last_activity_at=now,
            metadata=ExecutionDeliveryContext({"chat_key": "user_123"}),
        )
        assert record.scene_name == "wechat"
        assert record.metadata.get("chat_key") == "user_123"

    @pytest.mark.unit
    def test_closed_session(self) -> None:
        """关闭状态检查。"""
        now = datetime.now()
        record = SessionRecord(
            session_id="sess_closed",
            source=SessionSource.CLI,
            state=SessionState.CLOSED,
            created_at=now,
            last_activity_at=now,
        )
        assert not record.is_active()

    @pytest.mark.unit
    def test_metadata_default_empty(self) -> None:
        """metadata 默认空字典。"""
        record = SessionRecord(
            session_id="sess_x",
            source=SessionSource.API,
            state=SessionState.ACTIVE,
        )
        assert record.metadata == {}
