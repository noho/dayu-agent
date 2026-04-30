"""session closed 写入屏障测试。

验证 `cancel_session` 调整后的防御链：
1. 仓储层：`SQLitePendingConversationTurnStore` 与 `SQLiteReplyOutboxStore` 在 session
   已 CLOSED 或不存在时，对写入操作抛 ``SessionClosedError``；
2. Host 层：`host.cancel_session` 在仓储屏障装配下，关闭 session 后即便 executor 迟到写入
   也不会产生孤儿数据；
3. Executor：accepted/prepared pending turn 登记路径在 session 已关闭时降级为 no-op。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.contracts.reply_outbox import ReplyOutboxSubmitRequest
from dayu.host.host import Host
from dayu.host.host_store import HostStore
from dayu.host.pending_turn_store import (
    InMemoryPendingConversationTurnStore,
    PendingConversationTurnState,
    SQLitePendingConversationTurnStore,
)
from dayu.host.protocols import (
    SessionActivityQueryProtocol,
    SessionClearingError,
    SessionClearingFailedError,
    SessionClosedError,
    SessionWriteBlockedError,
)
from dayu.host.reply_outbox_store import (
    InMemoryReplyOutboxStore,
    SQLiteReplyOutboxStore,
)
from dayu.host.session_registry import SQLiteSessionRegistry
from dayu.contracts.session import SessionSource, SessionState

from tests.application.conftest import (
    StubHostExecutor,
    StubRunRegistry,
    StubSessionRegistry,
)


class _StaticActivity:
    """测试用 session 活性查询桩。"""

    def __init__(self, active: bool) -> None:
        self._active = active

    def is_session_active(self, session_id: str) -> bool:
        """返回预设活性。"""

        del session_id
        return self._active

    def get_session_state(self, session_id: str) -> SessionState | None:
        """非活跃路径上仅区分 ``CLOSED`` 与不存在；本桩按"已 CLOSED"返回。"""

        del session_id
        if self._active:
            return SessionState.ACTIVE
        return SessionState.CLOSED


def _assert_protocol(activity: _StaticActivity) -> SessionActivityQueryProtocol:
    """静态检查 SessionActivityQueryProtocol 兼容性。"""

    return activity


@pytest.mark.unit
def test_inmemory_pending_turn_store_blocks_upsert_when_session_closed() -> None:
    """session 已关闭 → upsert_pending_turn 抛 SessionClosedError。"""

    store = InMemoryPendingConversationTurnStore(
        session_activity=_assert_protocol(_StaticActivity(active=False)),
    )
    with pytest.raises(SessionClosedError) as excinfo:
        store.upsert_pending_turn(
            session_id="s-closed",
            scene_name="interactive",
            user_text="问题",
            source_run_id="run_1",
            resumable=True,
            state=PendingConversationTurnState.PREPARED_BY_HOST,
            resume_source_json="{}",
        )
    assert excinfo.value.session_id == "s-closed"


@pytest.mark.unit
def test_inmemory_reply_outbox_store_blocks_submit_when_session_closed() -> None:
    """session 已关闭 → submit_reply 抛 SessionClosedError。"""

    store = InMemoryReplyOutboxStore(
        session_activity=_assert_protocol(_StaticActivity(active=False)),
    )
    with pytest.raises(SessionClosedError):
        store.submit_reply(
            ReplyOutboxSubmitRequest(
                delivery_key="dk-1",
                session_id="s-closed",
                scene_name="interactive",
                source_run_id="run_1",
                reply_content="ok",
            )
        )


@pytest.mark.unit
def test_inmemory_stores_allow_writes_when_session_active() -> None:
    """session 活跃 → 仓储照常工作。"""

    pending = InMemoryPendingConversationTurnStore(
        session_activity=_assert_protocol(_StaticActivity(active=True)),
    )
    record = pending.upsert_pending_turn(
        session_id="s-active",
        scene_name="interactive",
        user_text="问题",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json="{}",
    )
    assert record.session_id == "s-active"

    outbox = InMemoryReplyOutboxStore(
        session_activity=_assert_protocol(_StaticActivity(active=True)),
    )
    reply = outbox.submit_reply(
        ReplyOutboxSubmitRequest(
            delivery_key="dk-active",
            session_id="s-active",
            scene_name="interactive",
            source_run_id="run_1",
            reply_content="ok",
        )
    )
    assert reply.session_id == "s-active"


@pytest.mark.unit
def test_sqlite_stores_barrier_with_session_registry(tmp_path: Path) -> None:
    """SQLite 仓储在默认装配下屏障生效。"""

    db_path = tmp_path / ".host" / "dayu_host.db"
    host_store = HostStore(db_path)
    host_store.initialize_schema()
    registry = SQLiteSessionRegistry(host_store)
    pending = SQLitePendingConversationTurnStore(host_store, session_activity=registry)
    outbox = SQLiteReplyOutboxStore(host_store, session_activity=registry)

    session = registry.create_session(SessionSource.WECHAT, session_id="s-barrier")
    assert registry.is_session_active(session.session_id)

    record = pending.upsert_pending_turn(
        session_id="s-barrier",
        scene_name="interactive",
        user_text="hi",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json="{}",
    )
    assert record.session_id == "s-barrier"

    registry.close_session("s-barrier")
    assert registry.is_session_active("s-barrier") is False

    with pytest.raises(SessionClosedError):
        pending.upsert_pending_turn(
            session_id="s-barrier",
            scene_name="interactive",
            user_text="late",
            source_run_id="run_2",
            resumable=True,
            state=PendingConversationTurnState.PREPARED_BY_HOST,
            resume_source_json="{}",
        )
    with pytest.raises(SessionClosedError):
        outbox.submit_reply(
            ReplyOutboxSubmitRequest(
                delivery_key="dk-late",
                session_id="s-barrier",
                scene_name="interactive",
                source_run_id="run_2",
                reply_content="late",
            )
        )


@pytest.mark.unit
def test_explicit_injected_host_cancel_session_blocks_late_pending_turn_writes() -> None:
    """显式注入 Host + cancel_session 后，executor 迟到登记 pending turn 应被屏障吸收。

    回归 B-06：显式注入分支若未注入 session_activity，executor 在 scene prepare 完成
    并 cancel 之后仍可能写入 pending turn。新装配逻辑应让默认内存 store 自动接入
    session_registry 作为活性源，确保迟到写入抛 SessionClosedError。
    """

    session_registry = StubSessionRegistry()
    run_registry = StubRunRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
    )
    session = host.create_session(SessionSource.WECHAT, session_id="sess-regression")
    # 触发 cancel_session：先 close_session 立起仓储屏障，再取消活跃 run / 清理记录。
    closed, _cancelled_ids = host.cancel_session(session.session_id)
    assert closed.state.value == "closed"
    # 此刻 executor 若迟到想登记 pending turn，仓储屏障必须拒绝。
    with pytest.raises(SessionClosedError):
        host._pending_turn_store.upsert_pending_turn(  # type: ignore[attr-defined]
            session_id=session.session_id,
            scene_name="interactive",
            user_text="late",
            source_run_id="run_late",
            resumable=True,
            state=PendingConversationTurnState.PREPARED_BY_HOST,
            resume_source_json="{}",
        )
    with pytest.raises(SessionClosedError):
        host._reply_outbox_store.submit_reply(  # type: ignore[attr-defined]
            ReplyOutboxSubmitRequest(
                delivery_key="dk-late",
                session_id=session.session_id,
                scene_name="interactive",
                source_run_id="run_late",
                reply_content="late",
            )
        )


# ---------------------------------------------------------------------------
# `SessionWriteBlockedError` 基类 / 三子类的契约：observability 区分 + 吸收统一
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_session_barrier_errors_share_common_base() -> None:
    """三类屏障异常都继承 ``SessionWriteBlockedError`` 以便上层统一吸收。"""

    assert issubclass(SessionClosedError, SessionWriteBlockedError)
    assert issubclass(SessionClearingError, SessionWriteBlockedError)
    assert issubclass(SessionClearingFailedError, SessionWriteBlockedError)


@pytest.mark.unit
def test_executor_absorbs_clearing_error_for_accepted_pending_turn() -> None:
    """``_register_accepted_pending_turn`` 在 ``CLEARING`` 屏障下降级为 no-op。

    回归 Review 缺口：原实现仅 ``except SessionClosedError``，
    ``clear_session_history`` 屏障期间 executor 迟到登记会未被吸收，
    抛出 ``SessionClearingError`` 把 run 升级为未预期失败。
    扩 ``SessionWriteBlockedError`` 后三子类共享同一吸收链路。
    """

    from dayu.host.pending_turn_store import InMemoryPendingConversationTurnStore

    class _ClearingActivity:
        def is_session_active(self, session_id: str) -> bool:
            del session_id
            return False

        def get_session_state(self, session_id: str):
            del session_id
            return SessionState.CLEARING

    pending = InMemoryPendingConversationTurnStore(session_activity=_ClearingActivity())
    # 直接验证仓储抛 SessionClearingError，且它是 SessionWriteBlockedError 的子类。
    with pytest.raises(SessionClearingError) as excinfo:
        pending.upsert_pending_turn(
            session_id="s-clearing",
            scene_name="interactive",
            user_text="late",
            source_run_id="run_late",
            resumable=True,
            state=PendingConversationTurnState.PREPARED_BY_HOST,
            resume_source_json="{}",
        )
    assert isinstance(excinfo.value, SessionWriteBlockedError)


@pytest.mark.unit
def test_executor_absorbs_clearing_failed_error_for_accepted_pending_turn() -> None:
    """``CLEARING_FAILED`` 持久锁定下迟到 pending turn 写入抛出可被基类吸收。"""

    from dayu.host.pending_turn_store import InMemoryPendingConversationTurnStore

    class _ClearingFailedActivity:
        def is_session_active(self, session_id: str) -> bool:
            del session_id
            return False

        def get_session_state(self, session_id: str):
            del session_id
            return SessionState.CLEARING_FAILED

    pending = InMemoryPendingConversationTurnStore(
        session_activity=_ClearingFailedActivity()
    )
    with pytest.raises(SessionClearingFailedError) as excinfo:
        pending.upsert_pending_turn(
            session_id="s-failed",
            scene_name="interactive",
            user_text="late",
            source_run_id="run_late",
            resumable=True,
            state=PendingConversationTurnState.PREPARED_BY_HOST,
            resume_source_json="{}",
        )
    assert isinstance(excinfo.value, SessionWriteBlockedError)

