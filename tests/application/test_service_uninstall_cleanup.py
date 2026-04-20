"""service uninstall 清理 pending turns / reply outbox / state_dir 的测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.host.host_store import HostStore
from dayu.host.pending_turn_store import (
    InMemoryPendingConversationTurnStore,
    PendingConversationTurnState,
    SQLitePendingConversationTurnStore,
)
from dayu.host.reply_outbox_store import (
    InMemoryReplyOutboxStore,
    SQLiteReplyOutboxStore,
)
from dayu.contracts.reply_outbox import ReplyOutboxSubmitRequest
from dayu.wechat.runtime import _purge_tracked_session_data
from dayu.wechat.state_store import (
    load_tracked_session_ids,
    record_tracked_session_id,
)


# ---------------------------------------------------------------------------
# tracked session_id 读写
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_record_and_load_tracked_session_ids(tmp_path: Path) -> None:
    """追加记录 session_id 并读回，重复写入自动去重。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    state_dir = tmp_path / "state"

    # 空目录返回空列表
    assert load_tracked_session_ids(state_dir) == []

    # 写入两个不同的 session_id
    record_tracked_session_id(state_dir, "sess_a")
    record_tracked_session_id(state_dir, "sess_b")
    assert load_tracked_session_ids(state_dir) == ["sess_a", "sess_b"]

    # 重复写入不产生重复
    record_tracked_session_id(state_dir, "sess_a")
    assert load_tracked_session_ids(state_dir) == ["sess_a", "sess_b"]


@pytest.mark.unit
def test_load_tracked_session_ids_handles_corrupt_file(tmp_path: Path) -> None:
    """文件内容非法时返回空列表而不抛异常。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "tracked_sessions.json").write_text("not json", encoding="utf-8")

    assert load_tracked_session_ids(state_dir) == []


# ---------------------------------------------------------------------------
# InMemory store delete_by_session_id
# ---------------------------------------------------------------------------


def _insert_pending_turn(
    store: InMemoryPendingConversationTurnStore | SQLitePendingConversationTurnStore,
    session_id: str,
    scene_name: str = "wechat",
) -> str:
    """向 pending turn store 插入一条记录并返回 ID。

    Args:
        store: pending turn 仓储。
        session_id: session ID。
        scene_name: 场景名称。

    Returns:
        pending turn ID。

    Raises:
        无。
    """

    record = store.upsert_pending_turn(
        session_id=session_id,
        scene_name=scene_name,
        user_text="test",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
        resume_source_json='{"scene_name": "wechat"}',
    )
    return record.pending_turn_id


@pytest.mark.unit
def test_inmemory_pending_turn_store_delete_by_session_id() -> None:
    """InMemory pending turn store 按 session 删除。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    store = InMemoryPendingConversationTurnStore()
    _insert_pending_turn(store, "s1", "scene_a")
    _insert_pending_turn(store, "s2", "scene_a")

    deleted = store.delete_by_session_id("s1")

    assert deleted == 1
    assert store.list_pending_turns(session_id="s1") == []
    assert len(store.list_pending_turns(session_id="s2")) == 1


@pytest.mark.unit
def test_inmemory_reply_outbox_store_delete_by_session_id() -> None:
    """InMemory reply outbox store 按 session 删除。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    store = InMemoryReplyOutboxStore()
    store.submit_reply(ReplyOutboxSubmitRequest(
        session_id="s1", scene_name="wechat", source_run_id="r1",
        delivery_key="dk1", reply_content="reply1",
    ))
    store.submit_reply(ReplyOutboxSubmitRequest(
        session_id="s2", scene_name="wechat", source_run_id="r2",
        delivery_key="dk2", reply_content="reply2",
    ))

    deleted = store.delete_by_session_id("s1")

    assert deleted == 1
    assert store.list_replies(session_id="s1") == []
    assert len(store.list_replies(session_id="s2")) == 1


# ---------------------------------------------------------------------------
# SQLite store delete_by_session_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sqlite_pending_turn_store_delete_by_session_id(tmp_path: Path) -> None:
    """SQLite pending turn store 按 session 删除。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    host_store = HostStore(tmp_path / "test.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    _insert_pending_turn(store, "s1")
    _insert_pending_turn(store, "s2")

    deleted = store.delete_by_session_id("s1")

    assert deleted == 1
    assert store.list_pending_turns(session_id="s1") == []
    assert len(store.list_pending_turns(session_id="s2")) == 1
    host_store.close()


@pytest.mark.unit
def test_sqlite_reply_outbox_store_delete_by_session_id(tmp_path: Path) -> None:
    """SQLite reply outbox store 按 session 删除。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    host_store = HostStore(tmp_path / "test.db")
    host_store.initialize_schema()
    store = SQLiteReplyOutboxStore(host_store)
    store.submit_reply(ReplyOutboxSubmitRequest(
        session_id="s1", scene_name="wechat", source_run_id="r1",
        delivery_key="dk1", reply_content="reply1",
    ))
    store.submit_reply(ReplyOutboxSubmitRequest(
        session_id="s2", scene_name="wechat", source_run_id="r2",
        delivery_key="dk2", reply_content="reply2",
    ))

    deleted = store.delete_by_session_id("s1")

    assert deleted == 1
    assert store.list_replies(session_id="s1") == []
    assert len(store.list_replies(session_id="s2")) == 1
    host_store.close()


# ---------------------------------------------------------------------------
# _purge_tracked_session_data 集成
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_purge_tracked_session_data_cleans_host_db(tmp_path: Path) -> None:
    """uninstall 时 _purge 按 tracked session_id 清理 Host DB。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    state_dir = tmp_path / "state"

    # 构造 Host DB 并插入数据
    from dayu.workspace_paths import build_host_store_default_path
    db_path = build_host_store_default_path(workspace_root)
    host_store = HostStore(db_path)
    host_store.initialize_schema()
    pending_store = SQLitePendingConversationTurnStore(host_store)
    outbox_store = SQLiteReplyOutboxStore(host_store)

    _insert_pending_turn(pending_store, "sess_target")
    _insert_pending_turn(pending_store, "sess_other")
    outbox_store.submit_reply(ReplyOutboxSubmitRequest(
        session_id="sess_target", scene_name="wechat", source_run_id="r1",
        delivery_key="dk1", reply_content="reply",
    ))
    host_store.close()

    # 记录 tracked session
    record_tracked_session_id(state_dir, "sess_target")

    # 执行清理
    _purge_tracked_session_data(workspace_root=workspace_root, state_dir=state_dir)

    # 验证：target 被删，other 保留
    host_store2 = HostStore(db_path)
    pending_store2 = SQLitePendingConversationTurnStore(host_store2)
    outbox_store2 = SQLiteReplyOutboxStore(host_store2)
    assert pending_store2.list_pending_turns(session_id="sess_target") == []
    assert len(pending_store2.list_pending_turns(session_id="sess_other")) == 1
    assert outbox_store2.list_replies(session_id="sess_target") == []
    host_store2.close()


@pytest.mark.unit
def test_purge_tracked_session_data_no_tracked_sessions(tmp_path: Path) -> None:
    """没有 tracked sessions 时 purge 不会报错。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 不应抛异常
    _purge_tracked_session_data(
        workspace_root=tmp_path / "workspace",
        state_dir=tmp_path / "nonexistent_state",
    )


# ---------------------------------------------------------------------------
# interactive --new-session 清理
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_interactive_new_session_purges_old_session(tmp_path: Path) -> None:
    """interactive --new-session 清理旧 session 的 pending turns 和 reply outbox。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.cli.interactive_state import (
        FileInteractiveStateStore,
        InteractiveSessionState,
        build_interactive_key,
        build_interactive_session_id,
    )
    from dayu.cli.dependency_setup import _resolve_interactive_session_id
    from dayu.workspace_paths import build_host_store_default_path

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()

    # 准备 Host DB
    db_path = build_host_store_default_path(workspace_root)
    host_store = HostStore(db_path)
    host_store.initialize_schema()
    pending_store = SQLitePendingConversationTurnStore(host_store)
    outbox_store = SQLiteReplyOutboxStore(host_store)

    # 先手动写一个 interactive 状态
    from dayu.workspace_paths import build_interactive_state_dir
    state_dir = build_interactive_state_dir(workspace_root)
    int_store = FileInteractiveStateStore(state_dir)
    old_key = build_interactive_key()
    int_store.save(InteractiveSessionState(interactive_key=old_key))
    old_session_id = build_interactive_session_id(old_key)

    # 插入旧 session 的数据
    _insert_pending_turn(pending_store, old_session_id, "interactive")
    outbox_store.submit_reply(ReplyOutboxSubmitRequest(
        session_id=old_session_id, scene_name="interactive", source_run_id="r1",
        delivery_key="dk1", reply_content="reply",
    ))
    # 插入其他 session 的数据
    _insert_pending_turn(pending_store, "other_session", "interactive")
    host_store.close()

    # 执行 new_session 解析（会触发清理）
    new_session_id = _resolve_interactive_session_id(workspace_root, new_session=True)

    # 新 session_id 应不同于旧的
    assert new_session_id != old_session_id

    # 验证旧 session 数据已清理，其他保留
    host_store2 = HostStore(db_path)
    pending_store2 = SQLitePendingConversationTurnStore(host_store2)
    outbox_store2 = SQLiteReplyOutboxStore(host_store2)
    assert pending_store2.list_pending_turns(session_id=old_session_id) == []
    assert outbox_store2.list_replies(session_id=old_session_id) == []
    assert len(pending_store2.list_pending_turns(session_id="other_session")) == 1
    host_store2.close()
