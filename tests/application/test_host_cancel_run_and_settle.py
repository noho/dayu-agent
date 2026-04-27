"""``Host.cancel_run_and_settle`` 单元测试。

覆盖以下场景：
- 活跃 run + 关联 pending turn → run 进 CANCELLED + 关联 pending turn 被清理。
- 已 CANCELLED run 幂等返回原记录。
- run 不存在抛 KeyError。
- run 无 session_id 时跳过 pending turn 清理。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from dayu.contracts.run import RunCancelReason, RunState
from dayu.host.host import Host
from dayu.host.pending_turn_store import InMemoryPendingConversationTurnStore
from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore
from tests.application.conftest import (
    StubHostExecutor,
    StubRunRegistry,
    StubSessionRegistry,
)

if TYPE_CHECKING:
    from dayu.host.host_execution import HostExecutorProtocol
    from dayu.host.protocols import RunRegistryProtocol, SessionRegistryProtocol


def _build_host(
    *,
    run_registry: StubRunRegistry,
    pending_turn_store: InMemoryPendingConversationTurnStore | None = None,
) -> Host:
    """构造仅供测试用 Host。"""

    return Host(
        executor=cast("HostExecutorProtocol", StubHostExecutor()),
        session_registry=cast("SessionRegistryProtocol", StubSessionRegistry()),
        run_registry=cast("RunRegistryProtocol", run_registry),
        pending_turn_store=pending_turn_store,
        reply_outbox_store=InMemoryReplyOutboxStore(),
    )


@pytest.mark.unit
def test_cancel_run_and_settle_pushes_active_run_to_cancelled() -> None:
    """活跃 run 经 cancel_run_and_settle 后必须收敛为 CANCELLED。"""

    registry = StubRunRegistry()
    record = registry.register_run(service_type="chat", session_id="session-1")
    registry.start_run(record.run_id)

    host = _build_host(run_registry=registry)
    settled = host.cancel_run_and_settle(record.run_id)

    assert settled.state == RunState.CANCELLED
    final = registry.get_run(record.run_id)
    assert final is not None
    assert final.state == RunState.CANCELLED


@pytest.mark.unit
def test_cancel_run_and_settle_is_idempotent_on_already_cancelled() -> None:
    """已 CANCELLED run 再次调用应幂等返回原记录。"""

    registry = StubRunRegistry()
    record = registry.register_run(service_type="chat", session_id="session-1")
    registry.start_run(record.run_id)
    registry.mark_cancelled(record.run_id, cancel_reason=RunCancelReason.USER_CANCELLED)

    host = _build_host(run_registry=registry)
    settled = host.cancel_run_and_settle(record.run_id)

    assert settled.state == RunState.CANCELLED
    assert settled.run_id == record.run_id


@pytest.mark.unit
def test_cancel_run_and_settle_raises_key_error_when_run_missing() -> None:
    """run 不存在时必须抛 KeyError。"""

    registry = StubRunRegistry()
    host = _build_host(run_registry=registry)

    with pytest.raises(KeyError):
        host.cancel_run_and_settle("does-not-exist")


@pytest.mark.unit
def test_cancel_run_and_settle_skips_pending_turn_cleanup_when_run_has_no_session() -> None:
    """run 没有 session_id 时不应触发 pending turn 清理。"""

    registry = StubRunRegistry()
    record = registry.register_run(service_type="chat", session_id=None)
    registry.start_run(record.run_id)

    pending_turn_store = InMemoryPendingConversationTurnStore()
    host = _build_host(run_registry=registry, pending_turn_store=pending_turn_store)

    settled = host.cancel_run_and_settle(record.run_id)

    assert settled.state == RunState.CANCELLED
    assert settled.session_id is None
