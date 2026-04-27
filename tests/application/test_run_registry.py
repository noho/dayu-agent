"""SQLiteRunRegistry 测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest

from dayu.host.host_store import HostStore
from dayu.host.pending_turn_store import PendingConversationTurnState, SQLitePendingConversationTurnStore
from dayu.host.run_registry import SQLiteRunRegistry
from dayu.contracts.run import ORPHAN_RUN_ERROR_SUMMARY, RunCancelReason, RunState
from dayu.log import Log


@pytest.fixture()
def registry(tmp_path: Path) -> SQLiteRunRegistry:
    """创建一个临时 RunRegistry。"""
    store = HostStore(tmp_path / "test.db")
    store.initialize_schema()
    return SQLiteRunRegistry(store)


class TestRegisterRun:
    """register_run 测试。"""

    @pytest.mark.unit
    def test_schema_does_not_define_ticker_column(self, registry: SQLiteRunRegistry) -> None:
        """新建 schema 的 runs 表不再包含 ticker 列。"""

        conn = registry._host_store.get_connection()
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        assert "ticker" not in columns
        assert "cancel_requested_at" in columns
        assert "cancel_requested_reason" in columns
        assert "cancel_reason" in columns

    @pytest.mark.unit
    def test_register_run_basic(self, registry: SQLiteRunRegistry) -> None:
        """注册 run，状态为 CREATED。"""
        run = registry.register_run(service_type="chat_turn")
        assert run.run_id.startswith("run_")
        assert run.service_type == "chat_turn"
        assert run.state == RunState.CREATED
        assert run.session_id is None
        assert run.started_at is None
        assert run.completed_at is None
        assert run.error_summary is None
        assert run.cancel_requested_at is None
        assert run.cancel_requested_reason is None
        assert run.cancel_reason is None

    @pytest.mark.unit
    def test_register_run_with_all_fields(self, registry: SQLiteRunRegistry) -> None:
        """注册 run 并传入所有可选字段。"""
        run = registry.register_run(
            session_id="sess-1",
            service_type="write_pipeline",
            scene_name="sec_filing",
            metadata={"delivery_channel": "wechat"},
        )
        assert run.session_id == "sess-1"
        assert run.scene_name == "sec_filing"
        assert run.metadata == {"delivery_channel": "wechat"}

    @pytest.mark.unit
    def test_register_run_has_pid(self, registry: SQLiteRunRegistry) -> None:
        """run 记录包含 owner_pid。"""
        import os
        run = registry.register_run(service_type="test")
        assert run.owner_pid == os.getpid()


class TestRunStateTransitions:
    """状态转换测试。"""

    @pytest.mark.unit
    def test_start_run(self, registry: SQLiteRunRegistry) -> None:
        """CREATED → RUNNING 转换。"""
        run = registry.register_run(service_type="test")
        started = registry.start_run(run.run_id)
        assert started.state == RunState.RUNNING
        assert started.started_at is not None

    @pytest.mark.unit
    def test_complete_run(self, registry: SQLiteRunRegistry) -> None:
        """RUNNING → SUCCEEDED 转换。"""
        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        completed = registry.complete_run(run.run_id)
        assert completed.state == RunState.SUCCEEDED
        assert completed.completed_at is not None

    @pytest.mark.unit
    def test_complete_run_allows_owned_orphan_recovery(self, registry: SQLiteRunRegistry) -> None:
        """当前 owner 可把被 cleanup 误判为 orphan 的 UNSETTLED run 修复为成功。"""

        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        registry.mark_unsettled(run.run_id, error_summary=ORPHAN_RUN_ERROR_SUMMARY)

        recovered = registry.complete_run(run.run_id)

        assert recovered.state == RunState.SUCCEEDED
        assert recovered.completed_at is not None
        assert recovered.error_summary is None

    @pytest.mark.unit
    def test_complete_run_rejects_failed_recovery(self, registry: SQLiteRunRegistry) -> None:
        """FAILED 终态不再被允许修复为 SUCCEEDED（仅 UNSETTLED 可被 owner 修复）。"""

        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        registry.fail_run(run.run_id, error_summary="business failure")

        with pytest.raises(ValueError, match="非法状态转换"):
            registry.complete_run(run.run_id)

    @pytest.mark.unit
    def test_mark_unsettled_from_running(self, registry: SQLiteRunRegistry) -> None:
        """RUNNING → UNSETTLED 合法。"""

        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        unsettled = registry.mark_unsettled(run.run_id, error_summary=ORPHAN_RUN_ERROR_SUMMARY)
        assert unsettled.state == RunState.UNSETTLED
        assert unsettled.completed_at is not None
        assert unsettled.error_summary == ORPHAN_RUN_ERROR_SUMMARY

    @pytest.mark.unit
    def test_unsettled_is_absorbing(self, registry: SQLiteRunRegistry) -> None:
        """UNSETTLED 对非当前 owner / 非 SUCCEEDED 转换都是吸收态。"""

        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        registry.mark_unsettled(run.run_id, error_summary=ORPHAN_RUN_ERROR_SUMMARY)
        # UNSETTLED → FAILED 非法
        with pytest.raises(ValueError, match="非法状态转换"):
            registry.fail_run(run.run_id, error_summary="x")

    @pytest.mark.unit
    def test_fail_run(self, registry: SQLiteRunRegistry) -> None:
        """RUNNING → FAILED 转换。"""
        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        failed = registry.fail_run(run.run_id, error_summary="boom")
        assert failed.state == RunState.FAILED
        assert failed.error_summary == "boom"
        assert failed.completed_at is not None

    @pytest.mark.unit
    def test_mark_cancelled(self, registry: SQLiteRunRegistry) -> None:
        """RUNNING → CANCELLED 转换。"""
        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        registry.request_cancel(run.run_id, cancel_reason=RunCancelReason.TIMEOUT)
        cancelled = registry.mark_cancelled(run.run_id)
        assert cancelled.state == RunState.CANCELLED
        assert cancelled.cancel_requested_reason == RunCancelReason.TIMEOUT
        assert cancelled.cancel_reason == RunCancelReason.USER_CANCELLED

    @pytest.mark.unit
    def test_invalid_transition_raises(self, registry: SQLiteRunRegistry) -> None:
        """非法状态转换抛出 ValueError。"""
        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        registry.complete_run(run.run_id)
        # SUCCEEDED → RUNNING 非法
        with pytest.raises(ValueError, match="非法状态转换"):
            registry.start_run(run.run_id)

    @pytest.mark.unit
    def test_transition_nonexistent_raises(self, registry: SQLiteRunRegistry) -> None:
        """对不存在的 run 做转换抛 KeyError。"""
        with pytest.raises(KeyError, match="run 不存在"):
            registry.start_run("nonexistent")


class TestCancelRequest:
    """request_cancel / is_cancel_requested 测试。"""

    @pytest.mark.unit
    def test_request_cancel_active_run(self, registry: SQLiteRunRegistry) -> None:
        """取消活跃 run 返回 True。"""
        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        assert registry.request_cancel(run.run_id) is True

        updated = registry.get_run(run.run_id)
        assert updated is not None
        assert updated.state == RunState.RUNNING
        assert updated.completed_at is None
        assert updated.cancel_requested_at is not None
        assert updated.cancel_requested_reason == RunCancelReason.USER_CANCELLED
        assert updated.cancel_reason is None

    @pytest.mark.unit
    def test_request_cancel_active_run_with_timeout_reason(self, registry: SQLiteRunRegistry) -> None:
        """取消活跃 run 时可写入 timeout 原因。"""

        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        assert registry.request_cancel(run.run_id, cancel_reason=RunCancelReason.TIMEOUT) is True

        updated = registry.get_run(run.run_id)
        assert updated is not None
        assert updated.cancel_requested_reason == RunCancelReason.TIMEOUT
        assert updated.cancel_reason is None

    @pytest.mark.unit
    def test_request_cancel_terminal_run(self, registry: SQLiteRunRegistry) -> None:
        """取消已完成 run 返回 False。"""
        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        registry.complete_run(run.run_id)
        assert registry.request_cancel(run.run_id) is False

    @pytest.mark.unit
    def test_is_cancel_requested(self, registry: SQLiteRunRegistry) -> None:
        """查询取消状态。"""
        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        assert registry.is_cancel_requested(run.run_id) is False
        registry.request_cancel(run.run_id)
        assert registry.is_cancel_requested(run.run_id) is True

    @pytest.mark.unit
    def test_request_cancel_is_idempotent_and_preserves_first_reason(self, registry: SQLiteRunRegistry) -> None:
        """重复请求取消不应覆盖首次取消原因。"""

        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)

        assert registry.request_cancel(run.run_id, cancel_reason=RunCancelReason.TIMEOUT) is True
        assert registry.request_cancel(run.run_id, cancel_reason=RunCancelReason.USER_CANCELLED) is False

        updated = registry.get_run(run.run_id)
        assert updated is not None
        assert updated.cancel_requested_reason == RunCancelReason.TIMEOUT

    @pytest.mark.unit
    def test_is_cancel_requested_nonexistent(self, registry: SQLiteRunRegistry) -> None:
        """查询不存在的 run 返回 False。"""
        assert registry.is_cancel_requested("nonexistent") is False


class TestQueryAndList:
    """get_run / list_runs / list_active_runs 测试。"""

    @pytest.mark.unit
    def test_get_run_exists(self, registry: SQLiteRunRegistry) -> None:
        """查询存在的 run。"""
        run = registry.register_run(service_type="test")
        fetched = registry.get_run(run.run_id)
        assert fetched is not None
        assert fetched.run_id == run.run_id

    @pytest.mark.unit
    def test_get_run_not_exists(self, registry: SQLiteRunRegistry) -> None:
        """查询不存在的 run 返回 None。"""
        assert registry.get_run("nonexistent") is None

    @pytest.mark.unit
    def test_list_runs_all(self, registry: SQLiteRunRegistry) -> None:
        """列出所有 runs。"""
        registry.register_run(service_type="a")
        registry.register_run(service_type="b")
        assert len(registry.list_runs()) == 2

    @pytest.mark.unit
    def test_list_runs_filter_by_session(self, registry: SQLiteRunRegistry) -> None:
        """按 session_id 过滤。"""
        registry.register_run(session_id="s1", service_type="a")
        registry.register_run(session_id="s2", service_type="b")
        runs = registry.list_runs(session_id="s1")
        assert len(runs) == 1
        assert runs[0].session_id == "s1"

    @pytest.mark.unit
    def test_list_runs_filter_by_state(self, registry: SQLiteRunRegistry) -> None:
        """按状态过滤。"""
        run1 = registry.register_run(service_type="a")
        registry.register_run(service_type="b")
        registry.start_run(run1.run_id)

        running = registry.list_runs(state=RunState.RUNNING)
        assert len(running) == 1
        assert running[0].state == RunState.RUNNING

    @pytest.mark.unit
    def test_list_runs_filter_by_service_type(self, registry: SQLiteRunRegistry) -> None:
        """按 service_type 过滤。"""
        registry.register_run(service_type="chat_turn")
        registry.register_run(service_type="prompt")
        runs = registry.list_runs(service_type="prompt")
        assert len(runs) == 1

    @pytest.mark.unit
    def test_list_active_runs(self, registry: SQLiteRunRegistry) -> None:
        """list_active_runs 只返回活跃 run。"""
        r1 = registry.register_run(service_type="a")
        r2 = registry.register_run(service_type="b")
        registry.start_run(r1.run_id)
        registry.start_run(r2.run_id)
        registry.complete_run(r2.run_id)

        active = registry.list_active_runs()
        assert len(active) == 1
        assert active[0].run_id == r1.run_id

    @pytest.mark.unit
    def test_list_active_runs_for_owner_filters_by_pid(self, registry: SQLiteRunRegistry) -> None:
        """list_active_runs_for_owner 严格按 PID 过滤。"""

        import os as _os

        r1 = registry.register_run(service_type="a")
        r2 = registry.register_run(service_type="b")
        registry.start_run(r1.run_id)
        registry.start_run(r2.run_id)

        # r2 改挂到另一个 PID
        conn = registry._host_store.get_connection()
        conn.execute("UPDATE runs SET owner_pid = ? WHERE run_id = ?", (42, r2.run_id))
        conn.commit()

        mine = registry.list_active_runs_for_owner(_os.getpid())
        other = registry.list_active_runs_for_owner(42)

        mine_ids = {run.run_id for run in mine}
        other_ids = {run.run_id for run in other}

        assert r1.run_id in mine_ids
        assert r2.run_id not in mine_ids
        assert other_ids == {r2.run_id}


@pytest.mark.unit
def test_run_registry_emits_lifecycle_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """RunRegistry 关键生命周期动作应输出日志。"""

    store = HostStore(tmp_path / "test.db")
    store.initialize_schema()
    registry = SQLiteRunRegistry(store)
    debug_mock = Mock()
    monkeypatch.setattr(Log, "debug", debug_mock)

    run = registry.register_run(service_type="chat_turn", session_id="s1", scene_name="wechat")
    registry.start_run(run.run_id)
    registry.request_cancel(run.run_id, cancel_reason=RunCancelReason.TIMEOUT)
    registry.mark_cancelled(run.run_id, cancel_reason=RunCancelReason.TIMEOUT)

    debug_messages = [call.args[0] for call in debug_mock.call_args_list]

    assert any("注册 run" in message and "owner_pid=" in message and "db_path=" in message for message in debug_messages)
    assert any(
        "run 状态迁移" in message and "to_state=running" in message and "db_path=" in message
        for message in debug_messages
    )
    assert any(
        "run 状态迁移" in message and "to_state=cancelled" in message and "db_path=" in message
        for message in debug_messages
    )
    assert any("登记 run 取消请求" in message for message in debug_messages)


class TestCleanupOrphanRuns:
    """cleanup_orphan_runs 测试。"""

    @pytest.mark.unit
    def test_cleanup_no_orphans(self, registry: SQLiteRunRegistry) -> None:
        """当前进程的 run 不会被清理。"""
        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)
        orphans = registry.cleanup_orphan_runs()
        assert orphans == []

    @pytest.mark.unit
    def test_cleanup_dead_pid_run(self, registry: SQLiteRunRegistry) -> None:
        """owner_pid 不存在的活跃 run 被标记 UNSETTLED（不再写 FAILED）。"""
        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)

        # 篡改 owner_pid 为一个不存在的 PID
        conn = registry._host_store.get_connection()
        conn.execute(
            "UPDATE runs SET owner_pid = ?, started_at = ?, created_at = ? WHERE run_id = ?",
            (
                999999,
                (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
                run.run_id,
            ),
        )
        conn.commit()

        orphans = registry.cleanup_orphan_runs()
        assert run.run_id in orphans

        updated = registry.get_run(run.run_id)
        assert updated is not None
        assert updated.state == RunState.UNSETTLED
        assert "orphan" in (updated.error_summary or "")

    @pytest.mark.unit
    def test_cleanup_recent_dead_pid_run_is_deferred(self, registry: SQLiteRunRegistry) -> None:
        """启动恢复不会立即清理刚开始执行的 dead-pid run。"""

        run = registry.register_run(service_type="test")
        registry.start_run(run.run_id)

        conn = registry._host_store.get_connection()
        conn.execute(
            "UPDATE runs SET owner_pid = ? WHERE run_id = ?",
            (999999, run.run_id),
        )
        conn.commit()

        orphans = registry.cleanup_orphan_runs()
        updated = registry.get_run(run.run_id)

        assert orphans == []
        assert updated is not None
        assert updated.state == RunState.RUNNING

    @pytest.mark.unit
    def test_cleanup_orphan_run_keeps_pending_turn_truth_source(self, tmp_path: Path) -> None:
        """orphan 清理不能破坏 pending turn 作为恢复真源。"""

        host_store = HostStore(tmp_path / "test.db")
        host_store.initialize_schema()
        registry = SQLiteRunRegistry(host_store)
        pending_turn_store = SQLitePendingConversationTurnStore(host_store)
        run = registry.register_run(session_id="s1", service_type="chat_turn", scene_name="interactive")
        registry.start_run(run.run_id)
        pending_turn = pending_turn_store.upsert_pending_turn(
            session_id="s1",
            scene_name="interactive",
            user_text="未交付问题",
            source_run_id=run.run_id,
            resumable=True,
            state=PendingConversationTurnState.ACCEPTED_BY_HOST,
        )

        conn = host_store.get_connection()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        conn.execute(
            "UPDATE runs SET owner_pid = ?, started_at = ?, created_at = ? WHERE run_id = ?",
            (999999, old_time, old_time, run.run_id),
        )
        conn.commit()

        orphan_ids = registry.cleanup_orphan_runs()
        persisted = pending_turn_store.get_pending_turn(pending_turn.pending_turn_id)

        assert orphan_ids == [run.run_id]
        assert persisted is not None
        assert persisted.pending_turn_id == pending_turn.pending_turn_id
        assert persisted.source_run_id == run.run_id
