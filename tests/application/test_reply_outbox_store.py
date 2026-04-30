"""reply outbox store 测试。"""

from __future__ import annotations

import threading
from datetime import timedelta
from pathlib import Path

import pytest

from dayu.contracts.reply_outbox import ReplyOutboxRecord, ReplyOutboxState, ReplyOutboxSubmitRequest
from dayu.host.host_store import HostStore
from dayu.host.lease import LeaseExpiredError
from dayu.host.protocols import ReplyOutboxStoreProtocol
from dayu.host.reply_outbox_store import (
    InMemoryReplyOutboxStore,
    MIN_STALE_AGE,
    SQLiteReplyOutboxStore,
    STALE_IN_PROGRESS_ERROR_MESSAGE,
)


def _build_submit_request(*, delivery_key: str = "wechat:run_1") -> ReplyOutboxSubmitRequest:
    """构造最小化提交请求。

    Args:
        delivery_key: 业务幂等键。

    Returns:
        可复用的提交请求。

    Raises:
        无。
    """

    return ReplyOutboxSubmitRequest(
        delivery_key=delivery_key,
        session_id="session_1",
        scene_name="wechat",
        source_run_id="run_1",
        reply_content="分析结论",
        metadata={
            "delivery_channel": "wechat",
            "delivery_target": "user_1",
            "delivery_thread_id": "thread_1",
        },
    )


@pytest.mark.unit
def test_in_memory_reply_outbox_submit_is_idempotent() -> None:
    """相同 delivery_key 与负载重复提交时应幂等返回同一记录。"""

    store = InMemoryReplyOutboxStore()

    first = store.submit_reply(_build_submit_request())
    second = store.submit_reply(_build_submit_request())

    assert second.delivery_id == first.delivery_id
    assert second.state == ReplyOutboxState.PENDING_DELIVERY


@pytest.mark.unit
def test_sqlite_reply_outbox_rejects_conflicting_payload_for_same_delivery_key(tmp_path: Path) -> None:
    """相同 delivery_key 但不同负载时应拒绝覆盖真源。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    store.submit_reply(_build_submit_request())

    with pytest.raises(ValueError, match="delivery_key 已存在且负载不一致"):
        store.submit_reply(
            ReplyOutboxSubmitRequest(
                delivery_key="wechat:run_1",
                session_id="session_1",
                scene_name="wechat",
                source_run_id="run_1",
                reply_content="另一个答案",
                metadata={"delivery_channel": "wechat", "delivery_target": "user_1"},
            )
        )


@pytest.mark.unit
def test_sqlite_reply_outbox_concurrent_submit_same_key_is_idempotent(tmp_path: Path) -> None:
    """SQLite 同一 delivery_key 并发提交时应幂等返回同一记录。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)

    thread_count = 6
    round_count = 5

    for round_index in range(round_count):
        barrier = threading.Barrier(thread_count)
        delivery_ids: list[str] = []
        exceptions: list[BaseException] = []
        lock = threading.Lock()
        delivery_key = f"wechat:run_{round_index}"

        def _worker() -> None:
            """并发提交同一个 reply outbox 幂等键。"""

            try:
                barrier.wait(timeout=5)
                record = store.submit_reply(_build_submit_request(delivery_key=delivery_key))
                with lock:
                    delivery_ids.append(record.delivery_id)
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    exceptions.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not exceptions
        assert len(delivery_ids) == thread_count
        assert len(set(delivery_ids)) == 1

        replies = store.list_replies(session_id="session_1", scene_name="wechat")
        round_replies = [record for record in replies if record.delivery_key == delivery_key]

        assert len(round_replies) == 1
        assert round_replies[0].delivery_id == delivery_ids[0]


@pytest.mark.unit
def test_sqlite_reply_outbox_concurrent_submit_conflicting_payload_returns_value_error(tmp_path: Path) -> None:
    """SQLite 同一 delivery_key 并发提交不同负载时应转化为业务冲突。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    barrier = threading.Barrier(2)
    success_records: list[str] = []
    value_errors: list[ValueError] = []
    other_exceptions: list[BaseException] = []
    lock = threading.Lock()

    def _worker(reply_content: str) -> None:
        """并发提交相同 delivery_key 的不同 payload。"""

        try:
            barrier.wait(timeout=5)
            record = store.submit_reply(
                ReplyOutboxSubmitRequest(
                    delivery_key="wechat:conflict",
                    session_id="session_1",
                    scene_name="wechat",
                    source_run_id="run_1",
                    reply_content=reply_content,
                    metadata={
                        "delivery_channel": "wechat",
                        "delivery_target": "user_1",
                        "delivery_thread_id": "thread_1",
                    },
                )
            )
            with lock:
                success_records.append(record.delivery_id)
        except ValueError as exc:
            with lock:
                value_errors.append(exc)
        except BaseException as exc:  # noqa: BLE001
            with lock:
                other_exceptions.append(exc)

    threads = [
        threading.Thread(target=_worker, args=("答案A",)),
        threading.Thread(target=_worker, args=("答案B",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not other_exceptions
    assert len(success_records) == 1
    assert len(value_errors) == 1

    replies = [record for record in store.list_replies() if record.delivery_key == "wechat:conflict"]

    assert len(replies) == 1
    assert replies[0].delivery_id == success_records[0]


@pytest.mark.unit
def test_sqlite_reply_outbox_submit_after_state_change_returns_existing_record(tmp_path: Path) -> None:
    """记录状态变更后重复 submit 相同 payload 仍应幂等返回既有记录。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)

    created = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(created.delivery_id)
    repeated = store.submit_reply(_build_submit_request())

    assert repeated.delivery_id == created.delivery_id
    assert repeated.state == ReplyOutboxState.DELIVERY_IN_PROGRESS
    assert repeated.delivery_attempt_count == claimed.delivery_attempt_count


@pytest.mark.unit
def test_sqlite_reply_outbox_claim_fail_and_deliver_state_machine(tmp_path: Path) -> None:
    """reply outbox 应支持 pending -> in_progress -> failed_retryable -> in_progress -> delivered。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)

    created = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(created.delivery_id)
    assert claimed.lease_id is not None
    failed = store.mark_failed(
        claimed.delivery_id,
        retryable=True,
        error_message="网络抖动",
        lease_id=claimed.lease_id,
    )
    reclaimed = store.claim_reply(failed.delivery_id)
    assert reclaimed.lease_id is not None
    delivered = store.mark_delivered(reclaimed.delivery_id, lease_id=reclaimed.lease_id)

    assert claimed.state == ReplyOutboxState.DELIVERY_IN_PROGRESS
    assert claimed.delivery_attempt_count == 1
    assert failed.state == ReplyOutboxState.FAILED_RETRYABLE
    assert failed.last_error_message == "网络抖动"
    assert reclaimed.state == ReplyOutboxState.DELIVERY_IN_PROGRESS
    assert reclaimed.delivery_attempt_count == 2
    assert delivered.state == ReplyOutboxState.DELIVERED
    assert delivered.delivery_attempt_count == reclaimed.delivery_attempt_count
    assert delivered.last_error_message is None


@pytest.mark.unit
def test_in_memory_reply_outbox_rejects_ack_before_claim() -> None:
    """内存版 ack 不得绕过 claim 直接完成交付。"""

    store = InMemoryReplyOutboxStore()
    created = store.submit_reply(_build_submit_request())

    with pytest.raises(ValueError, match="当前状态不允许 delivered"):
        store.mark_delivered(created.delivery_id, lease_id="ignored")


@pytest.mark.unit
def test_sqlite_reply_outbox_rejects_ack_before_claim(tmp_path: Path) -> None:
    """SQLite 版 ack 不得绕过 claim 直接完成交付。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    created = store.submit_reply(_build_submit_request())

    with pytest.raises(ValueError, match="当前状态不允许 delivered"):
        store.mark_delivered(created.delivery_id, lease_id="ignored")


@pytest.mark.unit
def test_sqlite_reply_outbox_rejects_ack_from_failed_retryable_without_reclaim(tmp_path: Path) -> None:
    """failed_retryable 记录未经重新 claim 时不得直接 ack。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    created = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(created.delivery_id)
    assert claimed.lease_id is not None
    failed = store.mark_failed(
        claimed.delivery_id,
        retryable=True,
        error_message="网络抖动",
        lease_id=claimed.lease_id,
    )

    with pytest.raises(ValueError, match="当前状态不允许 delivered"):
        store.mark_delivered(failed.delivery_id, lease_id=claimed.lease_id)


@pytest.mark.unit
def test_sqlite_reply_outbox_ack_is_idempotent_after_delivered(tmp_path: Path) -> None:
    """已 delivered 的记录重复 ack 应保持幂等。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    created = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(created.delivery_id)
    assert claimed.lease_id is not None
    delivered = store.mark_delivered(claimed.delivery_id, lease_id=claimed.lease_id)

    repeated = store.mark_delivered(delivered.delivery_id, lease_id=claimed.lease_id)

    assert repeated.delivery_id == delivered.delivery_id
    assert repeated.state == ReplyOutboxState.DELIVERED
    assert repeated.delivery_attempt_count == delivered.delivery_attempt_count
    assert repeated.last_error_message is None


@pytest.mark.unit
def test_sqlite_reply_outbox_ack_accepts_stale_in_progress_snapshot_when_already_delivered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite ack 在并发场景命中已 delivered 记录时应保持幂等。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    created = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(created.delivery_id)
    original_get_reply = store.get_reply
    stale_read_count = 0

    def _stale_then_actual(delivery_id: str):
        """第一次返回陈旧 in_progress 快照，后续返回数据库真实状态。"""

        nonlocal stale_read_count
        if stale_read_count == 0:
            stale_read_count += 1
            return claimed
        return original_get_reply(delivery_id)

    delivered = store.mark_delivered(claimed.delivery_id, lease_id=claimed.lease_id or "")
    monkeypatch.setattr(store, "get_reply", _stale_then_actual)

    repeated = store.mark_delivered(claimed.delivery_id, lease_id=claimed.lease_id or "")

    assert delivered.state == ReplyOutboxState.DELIVERED
    assert repeated.delivery_id == delivered.delivery_id
    assert repeated.state == ReplyOutboxState.DELIVERED
    assert repeated.delivery_attempt_count == delivered.delivery_attempt_count


@pytest.mark.unit
def test_sqlite_reply_outbox_claim_rejects_stale_pending_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLite claim 必须在数据库层原子校验旧状态，不能接受陈旧快照。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    created = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(created.delivery_id)
    original_get_reply = store.get_reply
    stale_read_count = 0

    def _stale_then_actual(delivery_id: str):
        """第一次返回陈旧 pending 快照，后续返回数据库真实状态。"""

        nonlocal stale_read_count
        if stale_read_count == 0:
            stale_read_count += 1
            return created
        return original_get_reply(delivery_id)

    monkeypatch.setattr(store, "get_reply", _stale_then_actual)

    with pytest.raises(ValueError, match="当前状态不允许 claim"):
        store.claim_reply(created.delivery_id)

    refreshed = original_get_reply(created.delivery_id)

    assert refreshed is not None
    assert refreshed.state == ReplyOutboxState.DELIVERY_IN_PROGRESS
    assert refreshed.delivery_attempt_count == claimed.delivery_attempt_count


@pytest.mark.unit
def test_reply_outbox_schema_includes_indexes(tmp_path: Path) -> None:
    """reply outbox schema 应包含幂等键约束和 source_run_id 索引。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    conn = host_store.get_connection()

    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(reply_outbox)").fetchall()
    }
    index_columns = {
        row["name"]: {
            detail["name"]
            for detail in conn.execute(f"PRAGMA index_info({row['name']})").fetchall()
        }
        for row in conn.execute("PRAGMA index_list(reply_outbox)").fetchall()
    }

    assert "delivery_key" in columns
    assert "metadata_json" in columns
    assert any({"source_run_id"} == indexed for indexed in index_columns.values())
    assert any({"delivery_key"} == indexed for indexed in index_columns.values())


@pytest.mark.unit
def test_reply_outbox_store_implements_runtime_protocol(tmp_path: Path) -> None:
    """内存版与 SQLite 版 reply outbox store 都应满足 runtime protocol。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()

    assert isinstance(InMemoryReplyOutboxStore(), ReplyOutboxStoreProtocol)
    assert isinstance(SQLiteReplyOutboxStore(host_store), ReplyOutboxStoreProtocol)


@pytest.mark.unit
def test_inmemory_mark_failed_is_idempotent_on_failed_terminal() -> None:
    """FAILED_TERMINAL 成为吸收态：重复 mark_failed 幂等返回同一记录。"""

    store = InMemoryReplyOutboxStore()
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None
    failed = store.mark_failed(
        submitted.delivery_id,
        retryable=False,
        error_message="fatal",
        lease_id=claimed.lease_id,
    )
    assert failed.state == ReplyOutboxState.FAILED_TERMINAL

    # 再次 mark_failed：不改 last_error_message、不抛错、不改 state；
    # FAILED_TERMINAL 在写路径前就被吸收态拦截，无需校验 lease_id。
    again = store.mark_failed(
        submitted.delivery_id,
        retryable=True,
        error_message="second try",
        lease_id=claimed.lease_id,
    )
    assert again.state == ReplyOutboxState.FAILED_TERMINAL
    assert again.last_error_message == "fatal"


@pytest.mark.unit
def test_sqlite_mark_failed_is_idempotent_on_failed_terminal(tmp_path: Path) -> None:
    """SQLite 实现同样对 FAILED_TERMINAL 幂等。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None
    failed = store.mark_failed(
        submitted.delivery_id,
        retryable=False,
        error_message="fatal",
        lease_id=claimed.lease_id,
    )
    assert failed.state == ReplyOutboxState.FAILED_TERMINAL

    again = store.mark_failed(
        submitted.delivery_id,
        retryable=True,
        error_message="second try",
        lease_id=claimed.lease_id,
    )
    assert again.state == ReplyOutboxState.FAILED_TERMINAL
    assert again.last_error_message == "fatal"


@pytest.mark.unit
def test_inmemory_cleanup_stale_in_progress_reverts_old_records() -> None:
    """超过 max_age 的 DELIVERY_IN_PROGRESS 被回退为 FAILED_RETRYABLE。"""

    store = InMemoryReplyOutboxStore()
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    # 手工把 updated_at 调到很久以前
    old_record = claimed
    store._records[old_record.delivery_id] = type(old_record)(
        delivery_id=old_record.delivery_id,
        delivery_key=old_record.delivery_key,
        session_id=old_record.session_id,
        scene_name=old_record.scene_name,
        source_run_id=old_record.source_run_id,
        reply_content=old_record.reply_content,
        metadata=old_record.metadata,
        state=old_record.state,
        created_at=old_record.created_at,
        updated_at=old_record.updated_at - timedelta(hours=1),
        delivery_attempt_count=old_record.delivery_attempt_count,
        last_error_message=old_record.last_error_message,
        lease_id=old_record.lease_id,
    )

    stale = store.cleanup_stale_in_progress_deliveries(max_age=timedelta(minutes=15))
    assert stale == [old_record.delivery_id]
    refreshed = store.get_reply(old_record.delivery_id)
    assert refreshed is not None
    assert refreshed.state == ReplyOutboxState.FAILED_RETRYABLE
    assert refreshed.last_error_message == STALE_IN_PROGRESS_ERROR_MESSAGE
    # cleanup 把 FAILED_RETRYABLE 的 lease 一并释放（置 NULL）：旧持有者后续
    # ack/nack 用旧 lease 撞 state+lease 双条件 CAS 必失败。
    assert refreshed.lease_id is None


@pytest.mark.unit
def test_sqlite_cleanup_stale_in_progress_reverts_old_records(tmp_path: Path) -> None:
    """SQLite 实现同样按 max_age 回退 stale in_progress。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)

    # 把 updated_at 手工调到 1 小时前
    conn = host_store.get_connection()
    from dayu.host._datetime_utils import serialize_dt
    conn.execute(
        "UPDATE reply_outbox SET updated_at = ? WHERE delivery_id = ?",
        (serialize_dt(claimed.updated_at - timedelta(hours=1)), claimed.delivery_id),
    )
    conn.commit()

    stale = store.cleanup_stale_in_progress_deliveries(max_age=timedelta(minutes=15))
    assert stale == [claimed.delivery_id]
    refreshed = store.get_reply(claimed.delivery_id)
    assert refreshed is not None
    assert refreshed.state == ReplyOutboxState.FAILED_RETRYABLE
    assert refreshed.last_error_message == STALE_IN_PROGRESS_ERROR_MESSAGE
    # cleanup 把 FAILED_RETRYABLE 的 lease 一并释放（置 NULL）：旧持有者后续
    # ack/nack 用旧 lease 撞 state+lease 双条件 CAS 必失败。
    assert refreshed.lease_id is None


@pytest.mark.unit
def test_cleanup_stale_in_progress_skips_fresh_records(tmp_path: Path) -> None:
    """未超 max_age 的 IN_PROGRESS 不会被回退。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    submitted = store.submit_reply(_build_submit_request())
    store.claim_reply(submitted.delivery_id)

    stale = store.cleanup_stale_in_progress_deliveries(max_age=timedelta(hours=1))
    assert stale == []
    refreshed = store.get_reply(submitted.delivery_id)
    assert refreshed is not None
    assert refreshed.state == ReplyOutboxState.DELIVERY_IN_PROGRESS


@pytest.mark.unit
def test_inmemory_cleanup_rejects_max_age_below_min() -> None:
    """InMemory 实现拒绝小于 ``MIN_STALE_AGE`` 的 max_age，避免与 mark_delivered 竞态。"""

    store = InMemoryReplyOutboxStore()
    submitted = store.submit_reply(_build_submit_request())
    store.claim_reply(submitted.delivery_id)
    too_small = MIN_STALE_AGE - timedelta(seconds=1)
    with pytest.raises(ValueError, match="max_age 不能小于"):
        store.cleanup_stale_in_progress_deliveries(max_age=too_small)
    refreshed = store.get_reply(submitted.delivery_id)
    assert refreshed is not None
    # 拒绝后状态保持 IN_PROGRESS，未被错误回退
    assert refreshed.state == ReplyOutboxState.DELIVERY_IN_PROGRESS


@pytest.mark.unit
def test_sqlite_cleanup_rejects_max_age_below_min(tmp_path: Path) -> None:
    """SQLite 实现拒绝小于 ``MIN_STALE_AGE`` 的 max_age，避免误回退。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    submitted = store.submit_reply(_build_submit_request())
    store.claim_reply(submitted.delivery_id)
    too_small = MIN_STALE_AGE - timedelta(seconds=1)
    with pytest.raises(ValueError, match="max_age 不能小于"):
        store.cleanup_stale_in_progress_deliveries(max_age=too_small)
    refreshed = store.get_reply(submitted.delivery_id)
    assert refreshed is not None
    assert refreshed.state == ReplyOutboxState.DELIVERY_IN_PROGRESS


@pytest.mark.unit
def test_sqlite_cleanup_at_min_stale_age_does_not_revert_fresh_record(tmp_path: Path) -> None:
    """``max_age = MIN_STALE_AGE`` 时，刚 claim 的 record 不会被错误回退。

    防回归点：早期实现允许调用方传入很小的 max_age，刚 claim 的 record
    会与 mark_delivered 在 ``state='delivery_in_progress'`` 上竞态。本测
    试断言下界恰好为 ``MIN_STALE_AGE`` 时该 record 仍处于 IN_PROGRESS。
    """

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    submitted = store.submit_reply(_build_submit_request())
    store.claim_reply(submitted.delivery_id)

    stale = store.cleanup_stale_in_progress_deliveries(max_age=MIN_STALE_AGE)
    assert stale == []
    refreshed = store.get_reply(submitted.delivery_id)
    assert refreshed is not None
    assert refreshed.state == ReplyOutboxState.DELIVERY_IN_PROGRESS


@pytest.mark.unit
def test_sqlite_mark_failed_sql_guard_prevents_overwriting_delivered(tmp_path: Path) -> None:
    """mark_failed 在 Python 检查后到 UPDATE 之间若并发被推进到 DELIVERED，应被 SQL 守卫拦截。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)

    # 模拟 TOCTOU 窗口：让 get_reply 返回 IN_PROGRESS 旧 snapshot，
    # 但实际数据库已被并发推进到 DELIVERED。
    assert claimed.lease_id is not None
    delivered = store.mark_delivered(claimed.delivery_id, lease_id=claimed.lease_id)
    assert delivered.state == ReplyOutboxState.DELIVERED

    stale_snapshot = claimed
    real_get_reply = store.get_reply
    call_state: dict[str, int] = {"count": 0}

    def fake_get_reply(delivery_id: str):
        call_state["count"] += 1
        if call_state["count"] == 1:
            return stale_snapshot
        return real_get_reply(delivery_id)

    store.get_reply = fake_get_reply  # type: ignore[method-assign]
    try:
        with pytest.raises(ValueError, match="已完成交付"):
            store.mark_failed(
                claimed.delivery_id,
                retryable=True,
                error_message="late failure",
                lease_id=claimed.lease_id,
            )
    finally:
        store.get_reply = real_get_reply  # type: ignore[method-assign]

    final = store.get_reply(claimed.delivery_id)
    assert final is not None
    assert final.state == ReplyOutboxState.DELIVERED
    assert final.last_error_message is None


@pytest.mark.unit
def test_inmemory_cleanup_invalidates_old_lease_for_ack_and_nack() -> None:
    """cleanup 重新分配 lease 后，旧持有者 ack/nack 必抛 LeaseExpiredError。"""

    store = InMemoryReplyOutboxStore()
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None
    old_lease_id: str = claimed.lease_id

    # 手工把 updated_at 调到很久以前以触发 cleanup。
    old_record = claimed
    store._records[old_record.delivery_id] = type(old_record)(
        delivery_id=old_record.delivery_id,
        delivery_key=old_record.delivery_key,
        session_id=old_record.session_id,
        scene_name=old_record.scene_name,
        source_run_id=old_record.source_run_id,
        reply_content=old_record.reply_content,
        metadata=old_record.metadata,
        state=old_record.state,
        created_at=old_record.created_at,
        updated_at=old_record.updated_at - timedelta(hours=1),
        delivery_attempt_count=old_record.delivery_attempt_count,
        last_error_message=old_record.last_error_message,
        lease_id=old_record.lease_id,
    )
    stale = store.cleanup_stale_in_progress_deliveries(max_age=timedelta(minutes=15))
    assert stale == [old_record.delivery_id]

    # 新 holder 重新 claim，拿到新 lease。
    reclaimed = store.claim_reply(old_record.delivery_id)
    assert reclaimed.lease_id is not None
    assert reclaimed.lease_id != old_lease_id

    # 旧 holder 用旧 lease 走 mark_delivered → LeaseExpiredError（state=IN_PROGRESS，但 lease 不匹配）。
    with pytest.raises(LeaseExpiredError):
        store.mark_delivered(old_record.delivery_id, lease_id=old_lease_id)

    # 旧 holder 用旧 lease 走 mark_failed → LeaseExpiredError。
    with pytest.raises(LeaseExpiredError):
        store.mark_failed(
            old_record.delivery_id,
            retryable=True,
            error_message="stale",
            lease_id=old_lease_id,
        )

    # 新 holder 仍可正常 ack。
    delivered = store.mark_delivered(reclaimed.delivery_id, lease_id=reclaimed.lease_id)
    assert delivered.state == ReplyOutboxState.DELIVERED


@pytest.mark.unit
def test_sqlite_cleanup_invalidates_old_lease_under_threading(tmp_path: Path) -> None:
    """SQLite 跨连接 cleanup 改写 lease 后，旧 holder ack/nack 必失败；新 holder 成功。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None

    # 直接通过 SQL 把 updated_at 调到很久以前。
    from dayu.host._datetime_utils import serialize_dt
    conn = host_store.get_connection()
    conn.execute(
        "UPDATE reply_outbox SET updated_at = ? WHERE delivery_id = ?",
        (serialize_dt(claimed.updated_at - timedelta(hours=1)), claimed.delivery_id),
    )
    conn.commit()

    # 在另一个线程里跑 cleanup（独立连接），验证跨连接 CAS。
    cleanup_result: list[list[str]] = []
    cleanup_error: list[BaseException] = []

    def cleanup_in_thread() -> None:
        try:
            cleanup_result.append(
                store.cleanup_stale_in_progress_deliveries(max_age=timedelta(minutes=15))
            )
        except BaseException as exc:  # pragma: no cover - defensive
            cleanup_error.append(exc)

    t = threading.Thread(target=cleanup_in_thread)
    t.start()
    t.join()
    assert cleanup_error == []
    assert cleanup_result == [[claimed.delivery_id]]

    # cleanup 把记录回退为 FAILED_RETRYABLE 并分配了新 lease；新 holder 重新 claim 后再次进入 IN_PROGRESS。
    reclaimed = store.claim_reply(claimed.delivery_id)
    assert reclaimed.lease_id is not None
    assert reclaimed.lease_id != claimed.lease_id
    old_lease_id: str = claimed.lease_id

    # 旧 holder 用旧 lease 在 IN_PROGRESS 状态下 mark_delivered，state 通过但 lease 不匹配 → LeaseExpiredError。
    with pytest.raises(LeaseExpiredError):
        store.mark_delivered(claimed.delivery_id, lease_id=old_lease_id)
    with pytest.raises(LeaseExpiredError):
        store.mark_failed(
            claimed.delivery_id,
            retryable=True,
            error_message="stale",
            lease_id=old_lease_id,
        )

    # 新 holder 用新 lease 仍能正常 ack。
    delivered = store.mark_delivered(reclaimed.delivery_id, lease_id=reclaimed.lease_id)
    assert delivered.state == ReplyOutboxState.DELIVERED


@pytest.mark.unit
def test_inmemory_mark_failed_rejects_stale_lease_when_record_already_failed_terminal() -> None:
    """旧 holder 在 cleanup 抢占后又看到他人收口为 FAILED_TERMINAL 时必须抛 LeaseExpiredError。"""

    store = InMemoryReplyOutboxStore()
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None
    old_lease_id: str = claimed.lease_id

    # 模拟 cleanup 抢占：手工把 updated_at 提前 + 触发 cleanup。
    old_record = claimed
    store._records[old_record.delivery_id] = type(old_record)(
        delivery_id=old_record.delivery_id,
        delivery_key=old_record.delivery_key,
        session_id=old_record.session_id,
        scene_name=old_record.scene_name,
        source_run_id=old_record.source_run_id,
        reply_content=old_record.reply_content,
        metadata=old_record.metadata,
        state=old_record.state,
        created_at=old_record.created_at,
        updated_at=old_record.updated_at - timedelta(hours=1),
        delivery_attempt_count=old_record.delivery_attempt_count,
        last_error_message=old_record.last_error_message,
        lease_id=old_record.lease_id,
    )
    store.cleanup_stale_in_progress_deliveries(max_age=timedelta(minutes=15))
    new_holder = store.claim_reply(old_record.delivery_id)
    assert new_holder.lease_id is not None
    # 新 holder 把记录收口为 FAILED_TERMINAL。
    new_holder_terminal = store.mark_failed(
        new_holder.delivery_id,
        retryable=False,
        error_message="dropped",
        lease_id=new_holder.lease_id,
    )
    assert new_holder_terminal.state == ReplyOutboxState.FAILED_TERMINAL

    # 旧 holder 拿旧 lease 撞上吸收态：必须 fail-loud。
    with pytest.raises(LeaseExpiredError):
        store.mark_failed(
            old_record.delivery_id,
            retryable=True,
            error_message="stale write after preemption",
            lease_id=old_lease_id,
        )
    # 新 holder 用同一 lease 仍可幂等返回。
    repeated = store.mark_failed(
        new_holder.delivery_id,
        retryable=True,
        error_message="ignored on absorber",
        lease_id=new_holder.lease_id,
    )
    assert repeated.state == ReplyOutboxState.FAILED_TERMINAL
    assert repeated.last_error_message == "dropped"


@pytest.mark.unit
def test_sqlite_mark_failed_rejects_stale_lease_when_record_already_failed_terminal(
    tmp_path: Path,
) -> None:
    """SQLite 路径同样要拒绝旧 lease 撞上他人 FAILED_TERMINAL 收口。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None
    old_lease_id: str = claimed.lease_id

    from dayu.host._datetime_utils import serialize_dt
    conn = host_store.get_connection()
    conn.execute(
        "UPDATE reply_outbox SET updated_at = ? WHERE delivery_id = ?",
        (serialize_dt(claimed.updated_at - timedelta(hours=1)), claimed.delivery_id),
    )
    conn.commit()
    store.cleanup_stale_in_progress_deliveries(max_age=timedelta(minutes=15))
    new_holder = store.claim_reply(claimed.delivery_id)
    assert new_holder.lease_id is not None
    terminal = store.mark_failed(
        new_holder.delivery_id,
        retryable=False,
        error_message="dropped",
        lease_id=new_holder.lease_id,
    )
    assert terminal.state == ReplyOutboxState.FAILED_TERMINAL

    with pytest.raises(LeaseExpiredError):
        store.mark_failed(
            claimed.delivery_id,
            retryable=True,
            error_message="stale write after preemption",
            lease_id=old_lease_id,
        )
    # 新 holder 用同一 lease 仍可幂等返回。
    repeated = store.mark_failed(
        claimed.delivery_id,
        retryable=True,
        error_message="ignored on absorber",
        lease_id=new_holder.lease_id,
    )
    assert repeated.state == ReplyOutboxState.FAILED_TERMINAL


@pytest.mark.unit
def test_inmemory_mark_delivered_rejects_stale_lease_when_record_already_delivered() -> None:
    """旧 holder 在抢占后撞上他人已 DELIVERED 的吸收态时必须抛 LeaseExpiredError。"""

    store = InMemoryReplyOutboxStore()
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None
    old_lease_id: str = claimed.lease_id

    old_record = claimed
    store._records[old_record.delivery_id] = type(old_record)(
        delivery_id=old_record.delivery_id,
        delivery_key=old_record.delivery_key,
        session_id=old_record.session_id,
        scene_name=old_record.scene_name,
        source_run_id=old_record.source_run_id,
        reply_content=old_record.reply_content,
        metadata=old_record.metadata,
        state=old_record.state,
        created_at=old_record.created_at,
        updated_at=old_record.updated_at - timedelta(hours=1),
        delivery_attempt_count=old_record.delivery_attempt_count,
        last_error_message=old_record.last_error_message,
        lease_id=old_record.lease_id,
    )
    store.cleanup_stale_in_progress_deliveries(max_age=timedelta(minutes=15))
    new_holder = store.claim_reply(old_record.delivery_id)
    assert new_holder.lease_id is not None
    delivered = store.mark_delivered(new_holder.delivery_id, lease_id=new_holder.lease_id)
    assert delivered.state == ReplyOutboxState.DELIVERED

    with pytest.raises(LeaseExpiredError):
        store.mark_delivered(old_record.delivery_id, lease_id=old_lease_id)

    repeated = store.mark_delivered(new_holder.delivery_id, lease_id=new_holder.lease_id)
    assert repeated.state == ReplyOutboxState.DELIVERED


@pytest.mark.unit
def test_sqlite_mark_delivered_rejects_stale_lease_when_record_already_delivered(
    tmp_path: Path,
) -> None:
    """SQLite 路径同样要拒绝旧 lease 撞上他人 DELIVERED 吸收态。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None
    old_lease_id: str = claimed.lease_id

    from dayu.host._datetime_utils import serialize_dt
    conn = host_store.get_connection()
    conn.execute(
        "UPDATE reply_outbox SET updated_at = ? WHERE delivery_id = ?",
        (serialize_dt(claimed.updated_at - timedelta(hours=1)), claimed.delivery_id),
    )
    conn.commit()
    store.cleanup_stale_in_progress_deliveries(max_age=timedelta(minutes=15))
    new_holder = store.claim_reply(claimed.delivery_id)
    assert new_holder.lease_id is not None
    delivered = store.mark_delivered(new_holder.delivery_id, lease_id=new_holder.lease_id)
    assert delivered.state == ReplyOutboxState.DELIVERED

    with pytest.raises(LeaseExpiredError):
        store.mark_delivered(claimed.delivery_id, lease_id=old_lease_id)

    repeated = store.mark_delivered(claimed.delivery_id, lease_id=new_holder.lease_id)
    assert repeated.state == ReplyOutboxState.DELIVERED


@pytest.mark.unit
def test_inmemory_mark_failed_retryable_releases_lease_so_old_holder_cannot_remark() -> None:
    """In-memory: FAILED_RETRYABLE 后必须释放 lease，旧 holder 不得继续改写记录。"""

    store = InMemoryReplyOutboxStore()
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None
    old_lease_id: str = claimed.lease_id
    failed = store.mark_failed(
        claimed.delivery_id,
        retryable=True,
        error_message="transient",
        lease_id=old_lease_id,
    )
    assert failed.state == ReplyOutboxState.FAILED_RETRYABLE
    assert failed.lease_id is None

    with pytest.raises(LeaseExpiredError):
        store.mark_failed(
            claimed.delivery_id,
            retryable=False,
            error_message="should not escalate",
            lease_id=old_lease_id,
        )
    with pytest.raises(LeaseExpiredError):
        store.mark_failed(
            claimed.delivery_id,
            retryable=True,
            error_message="should not rewrite",
            lease_id=old_lease_id,
        )

    reclaimed = store.claim_reply(claimed.delivery_id)
    assert reclaimed.lease_id is not None
    assert reclaimed.lease_id != old_lease_id


@pytest.mark.unit
def test_sqlite_mark_failed_retryable_releases_lease_so_old_holder_cannot_remark(
    tmp_path: Path,
) -> None:
    """SQLite: FAILED_RETRYABLE 后必须释放 lease，旧 holder 不得继续改写记录。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None
    old_lease_id: str = claimed.lease_id
    failed = store.mark_failed(
        claimed.delivery_id,
        retryable=True,
        error_message="transient",
        lease_id=old_lease_id,
    )
    assert failed.state == ReplyOutboxState.FAILED_RETRYABLE
    assert failed.lease_id is None

    with pytest.raises(LeaseExpiredError):
        store.mark_failed(
            claimed.delivery_id,
            retryable=False,
            error_message="should not escalate",
            lease_id=old_lease_id,
        )
    with pytest.raises(LeaseExpiredError):
        store.mark_failed(
            claimed.delivery_id,
            retryable=True,
            error_message="should not rewrite",
            lease_id=old_lease_id,
        )

    reclaimed = store.claim_reply(claimed.delivery_id)
    assert reclaimed.lease_id is not None
    assert reclaimed.lease_id != old_lease_id


@pytest.mark.unit
def test_inmemory_cleanup_stale_releases_lease_so_old_holder_cannot_remark() -> None:
    """In-memory: cleanup 抢占将记录回退到 FAILED_RETRYABLE 后 lease 必须为 NULL。"""

    store = InMemoryReplyOutboxStore()
    submitted = store.submit_reply(_build_submit_request())
    claimed = store.claim_reply(submitted.delivery_id)
    assert claimed.lease_id is not None
    old_lease_id: str = claimed.lease_id
    # 强制将 updated_at 置回过去以触发 cleanup
    aged = ReplyOutboxRecord(
        delivery_id=claimed.delivery_id,
        delivery_key=claimed.delivery_key,
        session_id=claimed.session_id,
        scene_name=claimed.scene_name,
        source_run_id=claimed.source_run_id,
        reply_content=claimed.reply_content,
        metadata=claimed.metadata,
        state=claimed.state,
        created_at=claimed.created_at,
        updated_at=claimed.updated_at - timedelta(hours=1),
        delivery_attempt_count=claimed.delivery_attempt_count,
        last_error_message=claimed.last_error_message,
        lease_id=claimed.lease_id,
    )
    store._records[claimed.delivery_id] = aged  # type: ignore[attr-defined]

    rotated = store.cleanup_stale_in_progress_deliveries(max_age=timedelta(minutes=15))
    assert rotated == [claimed.delivery_id]
    after = store.get_reply(claimed.delivery_id)
    assert after is not None
    assert after.state == ReplyOutboxState.FAILED_RETRYABLE
    assert after.lease_id is None

    with pytest.raises(LeaseExpiredError):
        store.mark_failed(
            claimed.delivery_id,
            retryable=False,
            error_message="should not escalate",
            lease_id=old_lease_id,
        )
