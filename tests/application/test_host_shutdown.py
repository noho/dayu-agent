"""Host.shutdown_active_runs_for_owner 边界测试。

验证当 ``request_cancel`` 已先于本次调用写入过 ``cancel_requested_at``、
返回 False 时，``shutdown_active_runs_for_owner`` 仍必须把仍处于 ACTIVE
状态的 run 收敛为 CANCELLED，而不是凭 ``request_cancel`` 的 False 直接放过。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.contracts.run import RunCancelReason, RunState
from dayu.host.host import Host
from dayu.host.host_store import HostStore
from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore
from dayu.host.run_registry import SQLiteRunRegistry
from tests.application.conftest import (
    StubHostExecutor,
    StubRunRegistry,
    StubSessionRegistry,
)


def _build_host(*, run_registry: StubRunRegistry) -> Host:
    """构造仅用于测试的 Host。"""

    return Host(
        executor=StubHostExecutor(),
        session_registry=StubSessionRegistry(),
        run_registry=run_registry,
        reply_outbox_store=InMemoryReplyOutboxStore(),
    )


@pytest.mark.unit
def test_shutdown_collapses_active_runs_even_when_cancel_already_requested() -> None:
    """已请求取消但仍 RUNNING 的 run 必须被收敛为 CANCELLED。

    回归 finding 064：曾经依赖 ``request_cancel`` 的返回值判断终态，导致
    cancel_requested_at 已被先前调用写入时被错误 ``continue`` 跳过，进而留下
    一个 RUNNING 的 owner run，被下一次 cleanup 误判为 UNSETTLED orphan。
    """

    registry = StubRunRegistry()
    record = registry.register_run(service_type="chat")
    registry.start_run(record.run_id)
    # 模拟外部已经先调用过 request_cancel：cancel_requested_at 已写入，
    # 但 run 仍然 RUNNING。
    assert registry.request_cancel(record.run_id, cancel_reason=RunCancelReason.USER_CANCELLED)
    assert registry.get_run(record.run_id).state == RunState.RUNNING  # type: ignore[union-attr]

    host = _build_host(run_registry=registry)
    cancelled_ids = host.shutdown_active_runs_for_owner()

    assert cancelled_ids == [record.run_id]
    final_record = registry.get_run(record.run_id)
    assert final_record is not None
    assert final_record.state == RunState.CANCELLED


@pytest.mark.unit
def test_shutdown_skips_runs_already_in_terminal_state() -> None:
    """已经是终态的 run 不应被再次 mark_cancelled。"""

    registry = StubRunRegistry()
    record = registry.register_run(service_type="chat")
    registry.start_run(record.run_id)
    registry.complete_run(record.run_id)

    host = _build_host(run_registry=registry)
    # list_active_runs_for_owner 已经过滤掉 SUCCEEDED，因此此处期望空列表。
    assert host.shutdown_active_runs_for_owner() == []
    final_record = registry.get_run(record.run_id)
    assert final_record is not None
    assert final_record.state == RunState.SUCCEEDED


@pytest.mark.unit
def test_mark_cancelled_is_idempotent(tmp_path: Path) -> None:
    """``mark_cancelled`` 对已处于 CANCELLED 状态的 run 应幂等返回。

    回归覆盖：executor 取消路径与 signal handler shutdown 路径竞态时，
    两条路径都可能对同一个 run 调 ``mark_cancelled``；
    若非幂等，第二次调用会触发「cancelled -> cancelled」非法状态转换。
    """

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    registry = SQLiteRunRegistry(host_store)

    record = registry.register_run(service_type="write")
    registry.start_run(record.run_id)

    # 第一次 mark_cancelled：RUNNING -> CANCELLED，应成功。
    first = registry.mark_cancelled(record.run_id, cancel_reason=RunCancelReason.USER_CANCELLED)
    assert first.state == RunState.CANCELLED

    # 第二次 mark_cancelled：CANCELLED -> CANCELLED，应幂等返回而非抛 ValueError。
    second = registry.mark_cancelled(record.run_id, cancel_reason=RunCancelReason.USER_CANCELLED)
    assert second.state == RunState.CANCELLED
    assert second.run_id == record.run_id
