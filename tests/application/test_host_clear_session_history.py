"""``Host.clear_session_history`` (#117) 验收测试。

覆盖 ``#117`` 共享设计 §3.7 验收点：

1. 五真源全清 happy path；
2. 四类拒绝（active run / pending turn / outbox / closed）→ ``Rejected`` 且全保留；
3. 场景 a（active run）→ ``Rejected``；
4. 场景 b（compaction 写回竞速 → archive revision 推进）→ ``Stale``；
5. archive 写失败 → 其他四真源不动；
6. archive 写成功后 delete 有界 retry 仍失败 → ``PartiallyApplied`` +
   ``CLEARING_FAILED`` 持久锁定（包含 6a 再次 clear 被拒、6b 新写入被拒、
   6c 屏障保持）；
7. 并发：屏障期间新建 pending turn / outbox / run 一律拒绝；
8. 契约联动：清完后历史读返回 ``[]``；
9. 清完后下一轮 ``persist_turn`` 不复活旧历史；
10. 幂等：连续两次清空第二次也按预检 no-op；
11. 历史读 read model 字段集合不退化（`#118` 结构性反射通过）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.contracts.reply_outbox import ReplyOutboxSubmitRequest
from dayu.contracts.session import SessionSource, SessionState
from dayu.host.conversation_session_archive import (
    ConversationHistoryTurnRecord,
    ConversationSessionArchive,
)
from dayu.host.conversation_store import (
    ConversationTurnRecord,
    FileConversationSessionArchiveStore,
)
from dayu.host.host import Host
from dayu.host.pending_turn_store import (
    InMemoryPendingConversationTurnStore,
    PendingConversationTurnState,
)
from dayu.host.protocols import (
    ConversationClearPartiallyAppliedError,
    ConversationClearRejectedError,
    ConversationClearStaleError,
    SessionClearingError,
    SessionClearingFailedError,
)
from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore
from tests.application.conftest import (
    StubHostExecutor,
    StubRunRegistry,
    StubSessionRegistry,
)


def _build_host(
    archive_store: FileConversationSessionArchiveStore,
    *,
    session_registry: StubSessionRegistry | None = None,
    run_registry: StubRunRegistry | None = None,
    pending_turn_store: InMemoryPendingConversationTurnStore | None = None,
    reply_outbox_store: InMemoryReplyOutboxStore | None = None,
    executor: StubHostExecutor | None = None,
) -> Host:
    """构造 #117 验收用 Host：所有内部仓储均装配 session 屏障。"""

    sr = session_registry or StubSessionRegistry()
    rr = run_registry or StubRunRegistry()
    pts = pending_turn_store or InMemoryPendingConversationTurnStore(session_activity=sr)
    ros = reply_outbox_store or InMemoryReplyOutboxStore(session_activity=sr)
    return Host(
        executor=executor or StubHostExecutor(),
        session_registry=sr,
        run_registry=rr,
        pending_turn_store=pts,
        reply_outbox_store=ros,
        archive_store=archive_store,
    )


def _seed_session_with_archive(
    host: Host,
    store: FileConversationSessionArchiveStore,
    *,
    session_id: str,
    turns: list[tuple[str, str, str]],
) -> ConversationSessionArchive:
    """创建活跃 session + 落盘若干轮 archive。"""

    host.create_session(SessionSource.CLI, session_id=session_id)
    archive = ConversationSessionArchive.create_empty(session_id)
    archive = store.save(archive, expected_revision=None)
    for idx, (user_text, assistant_text, reasoning) in enumerate(turns):
        prev_revision = archive.revision
        runtime_turn = ConversationTurnRecord(
            turn_id=f"turn_{idx}",
            scene_name="interactive",
            user_text=user_text,
            assistant_final=assistant_text,
        )
        history_turn = ConversationHistoryTurnRecord(
            turn_id=runtime_turn.turn_id,
            scene_name=runtime_turn.scene_name,
            user_text=user_text,
            assistant_text=assistant_text,
            assistant_reasoning=reasoning,
            created_at=runtime_turn.created_at,
        )
        archive = archive.with_next_turn(runtime_turn, history_turn)
        archive = store.save(archive, expected_revision=prev_revision)
    return archive


def _enqueue_outbox(host: Host, *, session_id: str, key: str = "k1") -> None:
    """向 reply_outbox_store 注入一条 PENDING_DELIVERY 记录。"""

    host._reply_outbox_store.submit_reply(
        ReplyOutboxSubmitRequest(
            delivery_key=key,
            session_id=session_id,
            scene_name="interactive",
            source_run_id="run_seed",
            reply_content="seeded",
        )
    )


def _enqueue_pending_turn(host: Host, *, session_id: str) -> None:
    """向 pending_turn_store 注入一条 ACCEPTED pending turn。"""

    host._pending_turn_store.upsert_pending_turn(
        session_id=session_id,
        scene_name="interactive",
        user_text="hello",
        source_run_id="run_seed",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )


# ---------------------------------------------------------------------------
# §3.7.1 happy path
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clear_session_history_clears_all_truth_sources(tmp_path: Path) -> None:
    """五真源全清：archive history+runtime / pending / outbox / replay stash。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host,
        store,
        session_id="sess",
        turns=[("Q1", "A1", "R1"), ("Q2", "A2", "R2")],
    )

    host.clear_session_history("sess")

    archive = store.load("sess")
    assert archive is not None
    assert archive.history_archive.turns == ()
    assert archive.runtime_transcript.turns == ()
    assert host._pending_turn_store.list_pending_turns(session_id="sess") == []
    assert host._reply_outbox_store.list_replies(session_id="sess") == []
    # session 仍 ACTIVE，下一轮可继续。
    assert host._session_registry.get_session_state("sess") == SessionState.ACTIVE


@pytest.mark.unit
def test_clear_then_history_read_returns_empty(tmp_path: Path) -> None:
    """契约联动：清完后 ``list_conversation_session_turn_excerpts`` 返回 []。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "R1")]
    )

    host.clear_session_history("sess")

    assert host.list_conversation_session_turn_excerpts("sess", limit=10) == []
    digest = host.get_conversation_session_digest("sess")
    assert digest.turn_count == 0


# ---------------------------------------------------------------------------
# §3.7.2 / §3.7.3 拒绝预检（含场景 a）
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clear_rejected_when_session_missing(tmp_path: Path) -> None:
    """session 不存在 → KeyError，五真源不变。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)

    with pytest.raises(KeyError, match="session 不存在"):
        host.clear_session_history("missing")


@pytest.mark.unit
def test_clear_rejected_when_session_closed(tmp_path: Path) -> None:
    """session 已 CLOSED → Rejected，archive 完整保留。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "")]
    )
    host._session_registry.close_session("sess")

    with pytest.raises(ConversationClearRejectedError):
        host.clear_session_history("sess")

    archive = store.load("sess")
    assert archive is not None
    assert len(archive.history_archive.turns) == 1


@pytest.mark.unit
def test_clear_rejected_when_active_run(tmp_path: Path) -> None:
    """场景 a：存在 active run → Rejected，五真源完整保留。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    sr = StubSessionRegistry()
    rr = StubRunRegistry()
    host = _build_host(store, session_registry=sr, run_registry=rr)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "")]
    )
    run = rr.register_run(session_id="sess", service_type="conversational")
    rr.start_run(run.run_id)  # RUNNING ∈ ACTIVE_STATES

    with pytest.raises(ConversationClearRejectedError) as exc:
        host.clear_session_history("sess")
    assert "active_runs" in exc.value.reason

    # 屏障已释放回 ACTIVE，archive 完整保留。
    assert host._session_registry.get_session_state("sess") == SessionState.ACTIVE
    archive = store.load("sess")
    assert archive is not None
    assert len(archive.history_archive.turns) == 1


@pytest.mark.unit
def test_clear_rejected_when_pending_turn_present(tmp_path: Path) -> None:
    """存在 pending turn → Rejected。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "")]
    )
    _enqueue_pending_turn(host, session_id="sess")

    with pytest.raises(ConversationClearRejectedError) as exc:
        host.clear_session_history("sess")
    assert "pending_turns" in exc.value.reason


@pytest.mark.unit
def test_clear_rejected_when_reply_outbox_present(tmp_path: Path) -> None:
    """存在待投递 reply outbox → Rejected。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "")]
    )
    _enqueue_outbox(host, session_id="sess")

    with pytest.raises(ConversationClearRejectedError) as exc:
        host.clear_session_history("sess")
    assert "reply_outbox" in exc.value.reason


# ---------------------------------------------------------------------------
# §3.7.4 场景 b：archive 乐观锁冲突 → Stale
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clear_stale_when_archive_revision_advanced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """场景 b：clear 在锁内拿到 live.revision 后，模拟 compaction 写回推进 archive
    revision。``archive_store.save`` 应抛 ``ConversationArchiveRevisionConflictError``，
    ``clear_session_history`` 转化为 ``ConversationClearStaleError``，五真源不变。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "R1")]
    )

    real_save = store.save

    def conflicting_save(archive, *, expected_revision=None):  # type: ignore[no-untyped-def]
        # 第一次（清空空 archive）注入冲突，后续放行。
        if len(archive.history_archive.turns) == 0 and expected_revision is not None:
            from dayu.host.protocols import ConversationArchiveRevisionConflictError

            raise ConversationArchiveRevisionConflictError(
                archive.session_id,
                expected_revision=expected_revision,
                actual_revision="rev_other",
            )
        return real_save(archive, expected_revision=expected_revision)

    monkeypatch.setattr(store, "save", conflicting_save)

    with pytest.raises(ConversationClearStaleError):
        host.clear_session_history("sess")

    # 屏障释放、archive 内容完整保留。
    assert host._session_registry.get_session_state("sess") == SessionState.ACTIVE
    archive = store.load("sess")
    assert archive is not None
    assert len(archive.history_archive.turns) == 1


# ---------------------------------------------------------------------------
# §3.7.6 contract B：补偿 retry 失败 → PartiallyApplied + CLEARING_FAILED
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clear_partially_applied_locks_session_in_clearing_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """archive 写已生效但 pending turn delete 始终失败 → PartiallyApplied，
    session 进入 CLEARING_FAILED 持久锁定，且：
    - 再次 clear_session_history 因屏障被 Rejected 拒（不会死循环）；
    - 新 pending turn / outbox 写入被 ``SessionClearingFailedError`` 拒。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "")]
    )

    # 注入 pending turn delete 始终失败。
    def always_fail(session_id: str) -> int:
        del session_id
        raise OSError("simulated delete failure")

    monkeypatch.setattr(host._pending_turn_store, "delete_by_session_id", always_fail)

    with pytest.raises(ConversationClearPartiallyAppliedError) as exc:
        host.clear_session_history("sess")
    assert "pending_turn_store" in exc.value.residual_sources

    # archive 已被清空（切点已过）。
    archive = store.load("sess")
    assert archive is not None
    assert archive.history_archive.turns == ()

    # 6a：session 进入 CLEARING_FAILED 持久锁定。
    assert (
        host._session_registry.get_session_state("sess")
        == SessionState.CLEARING_FAILED
    )

    # 6b：再次 clear 因屏障被拒（begin_clearing 检查 ACTIVE 失败 → Rejected）。
    with pytest.raises(ConversationClearRejectedError):
        host.clear_session_history("sess")

    # 6c：新 pending turn / outbox 写入被 SessionClearingFailedError 拒。
    with pytest.raises(SessionClearingFailedError):
        _enqueue_pending_turn(host, session_id="sess")
    with pytest.raises(SessionClearingFailedError):
        _enqueue_outbox(host, session_id="sess", key="k2")


@pytest.mark.unit
def test_clear_recovers_when_retry_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """补偿 delete 第一次失败、第二次成功 → 仍走 happy path（不抛错）。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "")]
    )

    real_delete = host._pending_turn_store.delete_by_session_id
    call_count = {"n": 0}

    def flaky_delete(session_id: str) -> int:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("simulated transient")
        return real_delete(session_id)

    monkeypatch.setattr(host._pending_turn_store, "delete_by_session_id", flaky_delete)

    host.clear_session_history("sess")

    assert call_count["n"] >= 2
    assert host._session_registry.get_session_state("sess") == SessionState.ACTIVE


# ---------------------------------------------------------------------------
# §3.7.7 屏障期间并发写入被拒
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_writes_during_clearing_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """屏障期间任一新写入路径都应抛 SessionClearingError。

    通过在 archive_store.save 时验证当前屏障状态来观测：在清空进行中，
    pending turn / outbox 写入应抛 SessionClearingError。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "")]
    )

    real_save = store.save
    observed_in_clearing: list[bool] = []

    def save_with_concurrent_check(archive, *, expected_revision=None):  # type: ignore[no-untyped-def]
        # archive 写发生时屏障已立起 → 此刻 pending turn 写入应该被屏障拒绝。
        try:
            _enqueue_pending_turn(host, session_id="sess")
        except SessionClearingError:
            observed_in_clearing.append(True)
        else:
            observed_in_clearing.append(False)
        # 同步检查：outbox 写入也应被屏障拒绝。
        try:
            _enqueue_outbox(host, session_id="sess", key="kc")
        except SessionClearingError:
            observed_in_clearing.append(True)
        else:
            observed_in_clearing.append(False)
        return real_save(archive, expected_revision=expected_revision)

    monkeypatch.setattr(store, "save", save_with_concurrent_check)

    host.clear_session_history("sess")

    assert observed_in_clearing == [True, True]


# ---------------------------------------------------------------------------
# §3.7.9 / §3.7.10 后续状态：下一轮不复活旧历史；幂等
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clear_then_next_persist_does_not_revive_old_history(tmp_path: Path) -> None:
    """清完后下一轮 ``with_next_turn`` 在空 archive 上推进，不复活旧历史。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "R1")]
    )

    host.clear_session_history("sess")

    # 模拟下一轮：load_or_create 拿到空 archive，推进一轮。
    live = store.load_or_create("sess")
    assert live.history_archive.turns == ()

    runtime_turn = ConversationTurnRecord(
        turn_id="turn_new",
        scene_name="interactive",
        user_text="Q-new",
        assistant_final="A-new",
    )
    history_turn = ConversationHistoryTurnRecord(
        turn_id="turn_new",
        scene_name="interactive",
        user_text="Q-new",
        assistant_text="A-new",
        assistant_reasoning="",
        created_at=runtime_turn.created_at,
    )
    next_archive = live.with_next_turn(runtime_turn, history_turn)
    store.save(next_archive, expected_revision=live.revision)

    excerpts = host.list_conversation_session_turn_excerpts("sess", limit=10)
    assert [e.user_text for e in excerpts] == ["Q-new"]


@pytest.mark.unit
def test_clear_is_idempotent(tmp_path: Path) -> None:
    """连续两次 clear 中间无写入 → 第二次也按 happy path 通过（archive 已空）。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "")]
    )

    host.clear_session_history("sess")
    host.clear_session_history("sess")  # 不抛错

    assert host._session_registry.get_session_state("sess") == SessionState.ACTIVE


# ---------------------------------------------------------------------------
# §3.7.11 历史读 read model 字段集合不退化（守 #118 / #116 边界）
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_clear_does_not_alter_read_model_contract(tmp_path: Path) -> None:
    """清空后再播一轮，read model 仍含 reasoning_text 字段（#116 契约不退化）。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)
    _seed_session_with_archive(
        host, store, session_id="sess", turns=[("Q1", "A1", "old-reason")]
    )

    host.clear_session_history("sess")

    live = store.load_or_create("sess")
    runtime_turn = ConversationTurnRecord(
        turn_id="turn_after",
        scene_name="interactive",
        user_text="Q2",
        assistant_final="A2",
    )
    history_turn = ConversationHistoryTurnRecord(
        turn_id="turn_after",
        scene_name="interactive",
        user_text="Q2",
        assistant_text="A2",
        assistant_reasoning="new-reason",
        created_at=runtime_turn.created_at,
    )
    next_archive = live.with_next_turn(runtime_turn, history_turn)
    store.save(next_archive, expected_revision=live.revision)

    excerpts = host.list_conversation_session_turn_excerpts("sess", limit=10)
    assert len(excerpts) == 1
    assert excerpts[0].user_text == "Q2"
    assert excerpts[0].assistant_text == "A2"
    assert excerpts[0].reasoning_text == "new-reason"
