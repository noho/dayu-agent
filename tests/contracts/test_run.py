"""RunRecord / RunState 数据模型测试。"""

from __future__ import annotations

from datetime import datetime

import pytest

from dayu.contracts.run import (
    ACTIVE_STATES,
    TERMINAL_STATES,
    RunCancelReason,
    RunRecord,
    RunState,
    is_valid_transition,
)


class TestRunState:
    """RunState 枚举测试。"""

    @pytest.mark.unit
    def test_terminal_states(self) -> None:
        """终态集合正确。"""
        assert RunState.SUCCEEDED in TERMINAL_STATES
        assert RunState.FAILED in TERMINAL_STATES
        assert RunState.CANCELLED in TERMINAL_STATES
        assert RunState.RUNNING not in TERMINAL_STATES

    @pytest.mark.unit
    def test_active_states(self) -> None:
        """活跃状态集合正确。"""
        assert RunState.CREATED in ACTIVE_STATES
        assert RunState.QUEUED in ACTIVE_STATES
        assert RunState.RUNNING in ACTIVE_STATES
        assert RunState.SUCCEEDED not in ACTIVE_STATES

    @pytest.mark.unit
    def test_string_value(self) -> None:
        """枚举值为字符串。"""
        assert RunState.CREATED == "created"
        assert RunState.RUNNING.value == "running"


class TestStateTransitions:
    """状态转换合法性测试。"""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "from_state, to_state",
        [
            (RunState.CREATED, RunState.QUEUED),
            (RunState.CREATED, RunState.RUNNING),
            (RunState.CREATED, RunState.CANCELLED),
            (RunState.QUEUED, RunState.RUNNING),
            (RunState.QUEUED, RunState.CANCELLED),
            (RunState.RUNNING, RunState.SUCCEEDED),
            (RunState.RUNNING, RunState.FAILED),
            (RunState.RUNNING, RunState.CANCELLED),
        ],
    )
    def test_valid_transitions(self, from_state: RunState, to_state: RunState) -> None:
        """合法状态转换。"""
        assert is_valid_transition(from_state, to_state)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "from_state, to_state",
        [
            (RunState.SUCCEEDED, RunState.RUNNING),
            (RunState.SUCCEEDED, RunState.FAILED),
            (RunState.FAILED, RunState.RUNNING),
            (RunState.CANCELLED, RunState.RUNNING),
            (RunState.RUNNING, RunState.CREATED),
            (RunState.RUNNING, RunState.QUEUED),
        ],
    )
    def test_invalid_transitions(self, from_state: RunState, to_state: RunState) -> None:
        """非法状态转换。"""
        assert not is_valid_transition(from_state, to_state)


class TestRunRecord:
    """RunRecord 数据结构测试。"""

    @pytest.mark.unit
    def test_create_minimal(self) -> None:
        """最小字段创建。"""
        now = datetime.now()
        record = RunRecord(
            run_id="run_abc123",
            session_id=None,
            service_type="prompt",
            scene_name=None,
            state=RunState.CREATED,
            created_at=now,
        )
        assert record.run_id == "run_abc123"
        assert record.is_active()
        assert not record.is_terminal()

    @pytest.mark.unit
    def test_create_full(self) -> None:
        """全字段创建。"""
        now = datetime.now()
        record = RunRecord(
            run_id="run_xyz789",
            session_id="sess_001",
            service_type="write_chapter",
            scene_name="write",
            state=RunState.RUNNING,
            created_at=now,
            started_at=now,
            owner_pid=12345,
            metadata={"delivery_channel": "wechat", "delivery_target": "user_1"},
        )
        assert record.session_id == "sess_001"
        assert record.owner_pid == 12345
        assert record.metadata.get("delivery_channel") == "wechat"
        assert record.is_active()

    @pytest.mark.unit
    def test_terminal_record(self) -> None:
        """终态记录检查。"""
        now = datetime.now()
        record = RunRecord(
            run_id="run_done",
            session_id=None,
            service_type="prompt",
            scene_name=None,
            state=RunState.SUCCEEDED,
            created_at=now,
            completed_at=now,
        )
        assert record.is_terminal()
        assert not record.is_active()

    @pytest.mark.unit
    def test_metadata_default_empty(self) -> None:
        """metadata 默认空字典。"""
        now = datetime.now()
        record = RunRecord(
            run_id="run_1",
            session_id=None,
            service_type="prompt",
            scene_name=None,
            state=RunState.CREATED,
            created_at=now,
        )
        assert record.metadata == {}

    @pytest.mark.unit
    def test_cancel_reason_can_be_recorded(self) -> None:
        """取消原因字段可用。"""

        now = datetime.now()
        record = RunRecord(
            run_id="run_cancelled",
            session_id=None,
            service_type="prompt",
            scene_name=None,
            state=RunState.CANCELLED,
            created_at=now,
            completed_at=now,
            cancel_reason=RunCancelReason.TIMEOUT,
        )
        assert record.cancel_reason == RunCancelReason.TIMEOUT

    @pytest.mark.unit
    def test_cancel_request_fields_can_be_recorded(self) -> None:
        """取消意图字段可用。"""

        now = datetime.now()
        record = RunRecord(
            run_id="run_cancel_requested",
            session_id=None,
            service_type="prompt",
            scene_name=None,
            state=RunState.RUNNING,
            created_at=now,
            cancel_requested_at=now,
            cancel_requested_reason=RunCancelReason.USER_CANCELLED,
        )
        assert record.cancel_requested_at == now
        assert record.cancel_requested_reason == RunCancelReason.USER_CANCELLED
