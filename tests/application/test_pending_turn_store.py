"""pending conversation turn store 测试。"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest

import dayu.host.pending_turn_store as pending_turn_store_module
import dayu.host.prepared_turn as prepared_turn_module
from dayu.contracts.agent_execution import AgentCreateArgs, ExecutionDocPermissions, ExecutionPermissions, ExecutionWebPermissions
from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.contracts.model_config import OpenAICompatibleRunnerParams
from dayu.contracts.toolset_config import ToolsetConfigSnapshot
from dayu.host.host_store import HostStore
from dayu.host.lease import LeaseExpiredError
from dayu.host.conversation_store import ConversationTranscript
from dayu.host.pending_turn_store import (
    InMemoryPendingConversationTurnStore,
    PendingConversationTurnState,
    PendingTurnResumeConflictError,
    SQLitePendingConversationTurnStore,
)
from dayu.host.prepared_turn import (
    PreparedAgentTurnSnapshot,
    PreparedConversationSessionSnapshot,
    deserialize_prepared_agent_turn_snapshot,
    serialize_prepared_agent_turn_snapshot,
)
from dayu.host.protocols import PendingConversationTurnStoreProtocol
from dayu.host.run_registry import SQLiteRunRegistry
from dayu.execution.options import ConversationMemorySettings


@pytest.mark.unit
def test_sqlite_pending_turn_store_upsert_and_resume_updates_source_run_id(tmp_path: Path) -> None:
    """相同 session/scene/user_text 恢复时应复用记录并把 source_run_id 指向新 run。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)

    first = store.upsert_pending_turn(
        session_id="s1",
        scene_name="interactive",
        user_text="同一问题",
        source_run_id="run_old",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
        resume_source_json='{"scene_name": "interactive"}',
        metadata={"interactive_key": "default"},
    )
    second = store.upsert_pending_turn(
        session_id="s1",
        scene_name="interactive",
        user_text="同一问题",
        source_run_id="run_new",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json='{"messages": [{"role": "user", "content": "同一问题"}], "scene_name": "interactive"}',
        metadata={"interactive_key": "default"},
    )

    assert second.pending_turn_id == first.pending_turn_id
    assert second.source_run_id == "run_new"
    assert second.state == PendingConversationTurnState.PREPARED_BY_HOST
    assert second.resume_source_json == '{"messages": [{"content": "同一问题", "role": "user"}], "scene_name": "interactive"}'


@pytest.mark.unit
def test_sqlite_pending_turn_store_rejects_non_object_prepared_snapshot(tmp_path: Path) -> None:
    """prepared turn 快照必须是 JSON object。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)

    with pytest.raises(ValueError, match="JSON object"):
        store.upsert_pending_turn(
            session_id="s1",
            scene_name="interactive",
            user_text="同一问题",
            source_run_id="run_old",
            resumable=True,
            state=PendingConversationTurnState.ACCEPTED_BY_HOST,
            resume_source_json='["bad"]',
        )


@pytest.mark.unit
def test_prepared_turn_snapshot_roundtrip_preserves_toolset_configs() -> None:
    """prepared turn 快照 roundtrip 时应以 toolset_configs 作为恢复真源。"""

    snapshot = PreparedAgentTurnSnapshot(
        service_name="chat_turn",
        scene_name="interactive",
        metadata=ExecutionDeliveryContext({"delivery_channel": "cli"}),
        business_concurrency_lane=None,
        timeout_ms=30000,
        resumable=True,
        system_prompt="system",
        messages=[{"role": "user", "content": "hello"}],
        agent_create_args=AgentCreateArgs(runner_type="openai", model_name="gpt-test"),
        selected_toolsets=("doc", "fins", "web"),
        execution_permissions=ExecutionPermissions(
            web=ExecutionWebPermissions(allow_private_network_url=True),
            doc=ExecutionDocPermissions(allowed_read_paths=("/tmp/in",)),
        ),
        trace_settings=None,
        conversation_memory_settings=ConversationMemorySettings(),
        conversation_session=PreparedConversationSessionSnapshot(
            session_id="session-1",
            user_message="hello",
            transcript=ConversationTranscript.create_empty("session-1"),
        ),
        toolset_configs=(
            ToolsetConfigSnapshot(toolset_name="doc", payload={"list_files_max": 21}),
            ToolsetConfigSnapshot(toolset_name="fins", payload={"list_documents_max_items": 34}),
            ToolsetConfigSnapshot(toolset_name="web", payload={"provider": "duckduckgo"}),
        ),
    )

    payload = serialize_prepared_agent_turn_snapshot(snapshot)

    assert "toolset_configs" in payload
    assert "doc_tool_limits" not in payload
    assert "fins_tool_limits" not in payload
    assert "web_tools_config" not in payload

    restored = deserialize_prepared_agent_turn_snapshot(payload)

    assert restored.toolset_configs == snapshot.toolset_configs


@pytest.mark.unit
def test_prepared_turn_snapshot_roundtrip_preserves_runner_params_types() -> None:
    """prepared turn 快照 roundtrip 时应保留 runner_params 的原始类型。"""

    runner_params: dict[str, object] = {
        "endpoint_url": "https://api.example.com/v1/chat/completions",
        "model": "test-model",
        "headers": {"Authorization": "Bearer test-key", "Content-Type": "application/json"},
        "supports_stream": False,
        "timeout": 3600,
        "default_extra_payloads": {"extra_key": "extra_value"},
        "temperature": 0.7,
        "max_retries": 3,
        "supports_tool_calling": True,
        "supports_stream_usage": False,
    }

    snapshot = PreparedAgentTurnSnapshot(
        service_name="chat_turn",
        scene_name="interactive",
        metadata=ExecutionDeliveryContext({"delivery_channel": "cli"}),
        business_concurrency_lane=None,
        timeout_ms=30000,
        resumable=True,
        system_prompt="system",
        messages=[{"role": "user", "content": "hello"}],
        agent_create_args=AgentCreateArgs(
            runner_type="openai_compatible",
            model_name="test-model",
            runner_params=runner_params,  # type: ignore[arg-type]
        ),
        selected_toolsets=("doc",),
        execution_permissions=ExecutionPermissions(
            web=ExecutionWebPermissions(allow_private_network_url=False),
            doc=ExecutionDocPermissions(allowed_read_paths=()),
        ),
        trace_settings=None,
        conversation_memory_settings=ConversationMemorySettings(),
        conversation_session=PreparedConversationSessionSnapshot(
            session_id="session-1",
            user_message="hello",
            transcript=ConversationTranscript.create_empty("session-1"),
        ),
        toolset_configs=(),
    )

    payload = serialize_prepared_agent_turn_snapshot(snapshot)
    restored = deserialize_prepared_agent_turn_snapshot(payload)

    rp = cast(OpenAICompatibleRunnerParams, restored.agent_create_args.runner_params)
    # 嵌套字典必须保持 dict 类型
    assert isinstance(rp.get("headers"), dict)
    assert rp.get("headers") == {"Authorization": "Bearer test-key", "Content-Type": "application/json"}
    assert isinstance(rp.get("default_extra_payloads"), dict)
    assert rp.get("default_extra_payloads") == {"extra_key": "extra_value"}
    # 布尔值必须保持 bool 类型（尤其是 False 不能变成 "False"）
    assert rp.get("supports_stream") is False
    assert rp.get("supports_tool_calling") is True
    assert rp.get("supports_stream_usage") is False
    # 数值必须保持原类型
    assert rp.get("timeout") == 3600
    assert isinstance(rp.get("timeout"), int)
    assert rp.get("temperature") == 0.7
    assert isinstance(rp.get("temperature"), float)
    assert rp.get("max_retries") == 3
    assert isinstance(rp.get("max_retries"), int)
    # 字符串保持不变
    assert rp.get("endpoint_url") == "https://api.example.com/v1/chat/completions"
    assert rp.get("model") == "test-model"


@pytest.mark.unit
def test_prepared_turn_helper_functions_cover_private_parsers(tmp_path: Path) -> None:
    """验证 prepared_turn 私有 helper 的剩余分支。"""

    assert prepared_turn_module._normalize_snapshot_value(Path("/tmp/demo")) == "/tmp/demo"
    assert prepared_turn_module._normalize_optional_text("  demo ") == "demo"
    assert prepared_turn_module._coerce_optional_int(None, field_name="timeout_ms") is None
    assert prepared_turn_module._coerce_optional_float(3, field_name="temperature") == pytest.approx(3.0)
    assert prepared_turn_module._parse_string_tuple(["doc", " ", "fins"]) == ("doc", "fins")
    assert prepared_turn_module._parse_string_dict({"a": "1", " ": "x", "b": 2}) == {"a": "1"}
    assert prepared_turn_module._as_runner_params({"timeout": 30}) == {"timeout": 30}
    assert prepared_turn_module._as_runner_params(None) == {}
    assert prepared_turn_module._as_runner_snapshot({"tool_timeout_seconds": 5.0}) == {"tool_timeout_seconds": 5.0}
    assert prepared_turn_module._as_runner_snapshot(None) == {}
    assert prepared_turn_module._parse_optional_trace_settings(None) is None
    trace_settings = prepared_turn_module._parse_optional_trace_settings(
        {"enabled": True, "output_dir": str(tmp_path / "trace")}
    )
    assert trace_settings is not None
    assert trace_settings.output_dir == (tmp_path / "trace")
    assert prepared_turn_module._parse_optional_trace_identity(None) is None
    trace_identity = prepared_turn_module._parse_optional_trace_identity(
        {
            "agent_name": "interactive_agent",
            "agent_kind": "scene_agent",
            "scene_name": "interactive",
            "model_name": "gpt-test",
            "session_id": "session-1",
        }
    )
    assert trace_identity is not None
    assert trace_identity.session_id == "session-1"
    conversation_session = prepared_turn_module._parse_optional_conversation_session(
        {
            "session_id": "session-1",
            "user_message": "hello",
            "transcript": json.loads(
                json.dumps(asdict(ConversationTranscript.create_empty("session-1")), ensure_ascii=False)
            ),
        }
    )
    assert conversation_session is not None
    assert conversation_session.user_message == "hello"
    toolset_configs = prepared_turn_module._parse_toolset_configs(
        [
            {"toolset_name": "doc", "payload": {"list_files_max": 10}},
            {"toolset_name": "doc", "payload": {"list_files_max": 20}},
        ]
    )
    assert toolset_configs == (ToolsetConfigSnapshot(toolset_name="doc", payload={"list_files_max": 20}),)

    with pytest.raises(ValueError, match="service_name"):
        prepared_turn_module._normalize_required_text("   ", field_name="service_name")
    with pytest.raises(ValueError, match="timeout_ms"):
        prepared_turn_module._coerce_optional_int(True, field_name="timeout_ms")
    with pytest.raises(ValueError, match="temperature"):
        prepared_turn_module._coerce_optional_float("bad", field_name="temperature")
    with pytest.raises(ValueError, match="messages 必须是 JSON array"):
        prepared_turn_module._parse_messages(None)
    with pytest.raises(ValueError, match="message.role"):
        prepared_turn_module._parse_messages([{"role": " ", "content": "x"}])

    class _Unsupported:
        """触发快照值类型错误。"""

    with pytest.raises(ValueError, match="不支持值类型"):
        prepared_turn_module._normalize_snapshot_value(_Unsupported())


@pytest.mark.unit
def test_sqlite_pending_turn_store_rejects_different_user_text_for_same_active_slot(tmp_path: Path) -> None:
    """同一 session/scene 存在活跃 pending turn 时不允许覆盖为不同 user_text。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )

    with pytest.raises(ValueError, match="不同 user_text"):
        store.upsert_pending_turn(
            session_id="s1",
            scene_name="wechat",
            user_text="问题二",
            source_run_id="run_2",
            resumable=True,
            state=PendingConversationTurnState.ACCEPTED_BY_HOST,
        )


@pytest.mark.unit
def test_sqlite_pending_turn_store_concurrent_upsert_is_atomic(tmp_path: Path) -> None:
    """同一 session/scene 并发 upsert 时应原子收敛到单条记录。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)

    thread_count = 6
    round_count = 5

    for round_index in range(round_count):
        barrier = threading.Barrier(thread_count)
        created_ids: list[str] = []
        exceptions: list[BaseException] = []
        lock = threading.Lock()
        session_id = f"session_{round_index}"

        def _worker(worker_index: int) -> None:
            """并发写入同一 pending turn 槽位。"""

            try:
                barrier.wait(timeout=5)
                record = store.upsert_pending_turn(
                    session_id=session_id,
                    scene_name="interactive",
                    user_text="同一个问题",
                    source_run_id=f"run_{round_index}_{worker_index}",
                    resumable=True,
                    state=PendingConversationTurnState.PREPARED_BY_HOST,
                    resume_source_json='{"scene_name": "interactive"}',
                    metadata={"interactive_key": "default"},
                )
                with lock:
                    created_ids.append(record.pending_turn_id)
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    exceptions.append(exc)

        threads = [threading.Thread(target=_worker, args=(index,)) for index in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not exceptions
        assert len(created_ids) == thread_count
        assert len(set(created_ids)) == 1

        pending_turns = store.list_pending_turns(session_id=session_id, scene_name="interactive")

        assert len(pending_turns) == 1
        assert pending_turns[0].pending_turn_id == created_ids[0]
        assert pending_turns[0].user_text == "同一个问题"


@pytest.mark.unit
def test_sqlite_pending_turn_store_update_path_parses_full_row_only_after_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同槽位更新时不应在锁内整行反序列化 pending turn。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    large_snapshot = '{"conversation_session": {"transcript": [' + ','.join('"x"' for _ in range(2000)) + ']}}'
    store.upsert_pending_turn(
        session_id="s1",
        scene_name="interactive",
        user_text="同一问题",
        source_run_id="run_old",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
        resume_source_json=large_snapshot,
    )

    original_row_to_pending_turn = pending_turn_store_module._row_to_pending_turn
    parse_call_count = 0

    def _counting_row_to_pending_turn(row: dict[str, object]):
        """统计完整 pending turn 反序列化调用次数。"""

        nonlocal parse_call_count
        parse_call_count += 1
        return original_row_to_pending_turn(row)

    monkeypatch.setattr(
        pending_turn_store_module,
        "_row_to_pending_turn",
        _counting_row_to_pending_turn,
    )

    updated = store.upsert_pending_turn(
        session_id="s1",
        scene_name="interactive",
        user_text="同一问题",
        source_run_id="run_new",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json=large_snapshot,
    )

    assert updated.source_run_id == "run_new"
    assert parse_call_count == 1


@pytest.mark.unit
def test_sqlite_pending_turn_store_lists_resumable_pending_turns(tmp_path: Path) -> None:
    """列举 pending turn 时应支持按 scene 和交付状态过滤。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    first = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    second = store.upsert_pending_turn(
        session_id="s2",
        scene_name="interactive",
        user_text="问题二",
        source_run_id="run_2",
        resumable=False,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    store.update_state(first.pending_turn_id, state=PendingConversationTurnState.SENT_TO_LLM)

    pending_turns = store.list_pending_turns(scene_name="wechat", resumable_only=True)

    assert [record.pending_turn_id for record in pending_turns] == [first.pending_turn_id]
    assert pending_turns[0].state == PendingConversationTurnState.SENT_TO_LLM


@pytest.mark.unit
def test_pending_turn_schema_includes_resume_source_and_source_run_index(tmp_path: Path) -> None:
    """pending turn schema 应持久化恢复尝试状态与 source_run 索引。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    conn = host_store.get_connection()

    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(pending_conversation_turns)").fetchall()
    }
    index_columns = {
        row["name"]: {
            detail["name"]
            for detail in conn.execute(f"PRAGMA index_info({row['name']})").fetchall()
        }
        for row in conn.execute("PRAGMA index_list(pending_conversation_turns)").fetchall()
    }

    assert "resume_source_json" in columns
    assert "resume_attempt_count" in columns
    assert "last_resume_error_message" in columns
    assert any({"source_run_id"} == indexed for indexed in index_columns.values())


@pytest.mark.unit
def test_sqlite_pending_turn_store_records_resume_attempts_and_failure_message(tmp_path: Path) -> None:
    """pending turn 仓储应持久化恢复次数与最近失败原因。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json='{"scene_name": "wechat"}',
    )

    attempted = store.record_resume_attempt(created.pending_turn_id, max_attempts=3)
    failed = store.record_resume_failure(
        created.pending_turn_id,
        error_message="source run still active",
        lease_id=attempted.resume_lease_id or "",
    )

    assert attempted.resume_attempt_count == 1
    assert attempted.last_resume_error_message is None
    assert failed.resume_attempt_count == 1
    assert failed.last_resume_error_message == "source run still active"


@pytest.mark.unit
def test_sqlite_pending_turn_store_record_resume_attempt_respects_atomic_max_attempts(tmp_path: Path) -> None:
    """恢复尝试自增应由仓储原子守住最大次数。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json='{"scene_name": "wechat"}',
    )

    success_ids: list[str] = []
    errors: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def _worker() -> None:
        try:
            barrier.wait(timeout=5)
            record = store.record_resume_attempt(created.pending_turn_id, max_attempts=1)
            with lock:
                success_ids.append(record.pending_turn_id)
        except ValueError as exc:
            with lock:
                errors.append(str(exc))

    first_thread = threading.Thread(target=_worker)
    second_thread = threading.Thread(target=_worker)
    first_thread.start()
    second_thread.start()
    first_thread.join(timeout=10)
    second_thread.join(timeout=10)

    assert success_ids == [created.pending_turn_id]
    assert len(errors) == 1
    assert "达到上限" in errors[0]
    # 达上限后记录被原子删除，防止超限 pending turn 卡死后续恢复
    current = store.get_pending_turn(created.pending_turn_id)
    assert current is None


@pytest.mark.unit
def test_sqlite_pending_turn_store_exhausted_attempt_deletes_record(tmp_path: Path) -> None:
    """达上限时应原子删除记录并 raise ValueError。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json='{"scene_name": "wechat"}',
    )

    store.record_resume_attempt(created.pending_turn_id, max_attempts=1)

    with pytest.raises(ValueError, match="达到上限"):
        store.record_resume_attempt(created.pending_turn_id, max_attempts=1)

    assert store.get_pending_turn(created.pending_turn_id) is None


@pytest.mark.unit
def test_inmemory_pending_turn_store_exhausted_attempt_deletes_record() -> None:
    """内存实现达上限同样原子删除记录。"""

    store = InMemoryPendingConversationTurnStore()
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )
    store.record_resume_attempt(created.pending_turn_id, max_attempts=1)

    with pytest.raises(ValueError, match="达到上限"):
        store.record_resume_attempt(created.pending_turn_id, max_attempts=1)

    assert store.get_pending_turn(created.pending_turn_id) is None


@pytest.mark.unit
def test_sqlite_pending_turn_store_rejects_concurrent_resume_acquire(tmp_path: Path) -> None:
    """并发 acquire 必须只有一方拿到 RESUMING，另一方抛 PendingTurnResumeConflictError。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json='{"scene_name": "wechat"}',
    )

    success_ids: list[str] = []
    conflicts: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def _worker() -> None:
        try:
            barrier.wait(timeout=5)
            record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
            with lock:
                success_ids.append(record.pending_turn_id)
        except PendingTurnResumeConflictError as exc:
            with lock:
                conflicts.append(str(exc))

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert len(success_ids) == 1
    assert len(conflicts) == 1
    current = store.get_pending_turn(created.pending_turn_id)
    assert current is not None
    assert current.state is PendingConversationTurnState.RESUMING
    assert current.pre_resume_state is PendingConversationTurnState.PREPARED_BY_HOST
    # 只前进一次 attempt count —— 体现互斥而非双写
    assert current.resume_attempt_count == 1


@pytest.mark.unit
def test_inmemory_pending_turn_store_rejects_duplicate_resume_acquire() -> None:
    """内存实现二次 acquire 必须抛 PendingTurnResumeConflictError。"""

    store = InMemoryPendingConversationTurnStore()
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
    )

    first = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    assert first.state is PendingConversationTurnState.RESUMING
    assert first.pre_resume_state is PendingConversationTurnState.PREPARED_BY_HOST

    with pytest.raises(PendingTurnResumeConflictError):
        store.record_resume_attempt(created.pending_turn_id, max_attempts=5)


@pytest.mark.unit
def test_sqlite_pending_turn_store_release_resume_lease_restores_state(tmp_path: Path) -> None:
    """release_resume_lease 把 RESUMING 原子回退到 pre_resume_state。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json='{"scene_name": "wechat"}',
    )
    attempted = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)

    released = store.release_resume_lease(
        created.pending_turn_id, lease_id=attempted.resume_lease_id or ""
    )
    assert released is not None
    assert released.state is PendingConversationTurnState.PREPARED_BY_HOST
    assert released.pre_resume_state is None

    # 同一持有者的 lease 在 release 后已被清空，再调用应抛 LeaseExpiredError
    # （fence token 语义：state 非 RESUMING 时旧 lease 一律视为过期）。
    with pytest.raises(LeaseExpiredError):
        store.release_resume_lease(
            created.pending_turn_id, lease_id=attempted.resume_lease_id or ""
        )


@pytest.mark.unit
def test_sqlite_pending_turn_store_record_resume_failure_releases_lease(tmp_path: Path) -> None:
    """record_resume_failure 在 RESUMING 态应同时写错误消息并回退 state。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.PREPARED_BY_HOST,
        resume_source_json='{"scene_name": "wechat"}',
    )
    attempted = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)

    failed = store.record_resume_failure(
        created.pending_turn_id,
        error_message="boom",
        lease_id=attempted.resume_lease_id or "",
    )
    assert failed.state is PendingConversationTurnState.PREPARED_BY_HOST
    assert failed.pre_resume_state is None
    assert failed.last_resume_error_message == "boom"


@pytest.mark.unit
def test_inmemory_pending_turn_store_release_resume_lease_restores_state() -> None:
    """内存实现的 release_resume_lease 行为与 SQLite 等价。"""

    store = InMemoryPendingConversationTurnStore()
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    attempted = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)

    released = store.release_resume_lease(
        created.pending_turn_id, lease_id=attempted.resume_lease_id or ""
    )
    assert released is not None
    assert released.state is PendingConversationTurnState.ACCEPTED_BY_HOST
    assert released.pre_resume_state is None

    # 第二次回退：state 已不在 RESUMING，旧 lease 视为过期，必抛 LeaseExpiredError。
    with pytest.raises(LeaseExpiredError):
        store.release_resume_lease(
            created.pending_turn_id, lease_id=attempted.resume_lease_id or ""
        )


@pytest.mark.unit
def test_pending_turn_store_implements_runtime_protocol(tmp_path: Path) -> None:
    """SQLite pending turn store 应满足 runtime protocol。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)

    assert isinstance(store, PendingConversationTurnStoreProtocol)


@pytest.mark.unit
def test_run_registry_implements_runtime_protocol(tmp_path: Path) -> None:
    """SQLite run registry 应满足 runtime protocol。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    registry = SQLiteRunRegistry(host_store)

    from dayu.host.protocols import RunRegistryProtocol

    assert isinstance(registry, RunRegistryProtocol)


@pytest.mark.unit
def test_inmemory_release_resume_lease_warns_on_missing_pre_resume_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """pre_resume_state 为 NULL 时应输出告警并按 ACCEPTED_BY_HOST 降级回退。"""

    store = InMemoryPendingConversationTurnStore()
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    # 模拟 pre_resume_state 字段损坏：手动写入 RESUMING 但不附带 pre_resume_state。
    record = store._records[created.pending_turn_id]
    forced_record = type(record)(
        pending_turn_id=record.pending_turn_id,
        session_id=record.session_id,
        scene_name=record.scene_name,
        user_text=record.user_text,
        source_run_id=record.source_run_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        resumable=record.resumable,
        state=PendingConversationTurnState.RESUMING,
        resume_source_json=record.resume_source_json,
        resume_attempt_count=record.resume_attempt_count,
        last_resume_error_message=record.last_resume_error_message,
        pre_resume_state=None,
        metadata=record.metadata,
    )
    store._records[created.pending_turn_id] = forced_record

    with caplog.at_level("WARNING"):
        released = store.cleanup_stale_resuming(
            created.pending_turn_id,
            expected_updated_at=forced_record.updated_at,
        )
    assert released is not None
    assert released.state is PendingConversationTurnState.ACCEPTED_BY_HOST
    assert any("pre_resume_state 缺失" in r.message for r in caplog.records), (
        "pre_resume_state 缺失时应输出告警"
    )


@pytest.mark.unit
def test_inmemory_record_resume_failure_warns_on_missing_pre_resume_state(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """RESUMING 但 pre_resume_state 为 NULL 时，record_resume_failure 应告警并降级回退。"""

    store = InMemoryPendingConversationTurnStore()
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题二",
        source_run_id="run_2",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    forced_lease = "lease_force_inmemory"
    record = store._records[created.pending_turn_id]
    store._records[created.pending_turn_id] = type(record)(
        pending_turn_id=record.pending_turn_id,
        session_id=record.session_id,
        scene_name=record.scene_name,
        user_text=record.user_text,
        source_run_id=record.source_run_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
        resumable=record.resumable,
        state=PendingConversationTurnState.RESUMING,
        resume_source_json=record.resume_source_json,
        resume_attempt_count=record.resume_attempt_count,
        last_resume_error_message=record.last_resume_error_message,
        pre_resume_state=None,
        resume_lease_id=forced_lease,
        metadata=record.metadata,
    )

    with caplog.at_level("WARNING"):
        failed = store.record_resume_failure(
            created.pending_turn_id, error_message="boom", lease_id=forced_lease,
        )
    assert failed.state is PendingConversationTurnState.ACCEPTED_BY_HOST
    assert failed.last_resume_error_message == "boom"
    assert any("pre_resume_state 缺失" in r.message for r in caplog.records), (
        "pre_resume_state 缺失时应输出告警"
    )


@pytest.mark.unit
def test_sqlite_release_resume_lease_warns_on_missing_pre_resume_state(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SQLite 版 cleanup_stale_resuming 在 pre_resume_state 为 NULL 时应告警并降级回退。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)

    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题一",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    # 直接通过底层连接把 state 改成 RESUMING 同时把 pre_resume_state 置 NULL，
    # 模拟数据损坏或 acquire 路径异常导致的字段缺失。
    conn = host_store.get_connection()
    conn.execute(
        "UPDATE pending_conversation_turns SET state = ?, pre_resume_state = NULL WHERE pending_turn_id = ?",
        (PendingConversationTurnState.RESUMING.value, created.pending_turn_id),
    )
    conn.commit()

    refreshed = store.get_pending_turn(created.pending_turn_id)
    assert refreshed is not None
    with caplog.at_level("WARNING"):
        released = store.cleanup_stale_resuming(
            created.pending_turn_id,
            expected_updated_at=refreshed.updated_at,
        )
    assert released is not None
    assert released.state is PendingConversationTurnState.ACCEPTED_BY_HOST
    assert any("pre_resume_state 缺失" in r.message for r in caplog.records), (
        "pre_resume_state 缺失时应输出告警"
    )


@pytest.mark.unit
def test_sqlite_record_resume_failure_warns_on_missing_pre_resume_state(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SQLite 版 record_resume_failure 在 pre_resume_state 为 NULL 时应告警并降级回退。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)

    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题二",
        source_run_id="run_2",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    forced_lease = "lease_force_sqlite"
    conn = host_store.get_connection()
    conn.execute(
        "UPDATE pending_conversation_turns SET state = ?, pre_resume_state = NULL, resume_lease_id = ? WHERE pending_turn_id = ?",
        (PendingConversationTurnState.RESUMING.value, forced_lease, created.pending_turn_id),
    )
    conn.commit()

    with caplog.at_level("WARNING"):
        failed = store.record_resume_failure(
            created.pending_turn_id, error_message="boom", lease_id=forced_lease,
        )
    assert failed.state is PendingConversationTurnState.ACCEPTED_BY_HOST
    assert failed.last_resume_error_message == "boom"
    assert any("pre_resume_state 缺失" in r.message for r in caplog.records), (
        "pre_resume_state 缺失时应输出告警"
    )


@pytest.mark.unit
def test_inmemory_lease_mismatch_release_raises_lease_expired() -> None:
    """In-Memory 实现：A 持有 lease → cleanup_stale_resuming 抢占 → A 迟到 release 必抛 LeaseExpiredError。"""

    store = InMemoryPendingConversationTurnStore()
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    a_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    assert a_record.resume_lease_id

    # cleanup 兜底：把 lease 抢占清空，state 回退到 ACCEPTED_BY_HOST。
    cleaned = store.cleanup_stale_resuming(
        created.pending_turn_id,
        expected_updated_at=a_record.updated_at,
    )
    assert cleaned is not None
    assert cleaned.resume_lease_id is None

    # B 重新 acquire，分配新 lease。
    b_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    assert b_record.resume_lease_id
    assert b_record.resume_lease_id != a_record.resume_lease_id

    # A 迟到的 release / rebind / failure 全部应抛 LeaseExpiredError。
    with pytest.raises(LeaseExpiredError):
        store.release_resume_lease(created.pending_turn_id, lease_id=a_record.resume_lease_id or "")
    with pytest.raises(LeaseExpiredError):
        store.rebind_source_run_id_for_resume(
            created.pending_turn_id,
            new_source_run_id="run_a_late",
            lease_id=a_record.resume_lease_id or "",
        )
    with pytest.raises(LeaseExpiredError):
        store.record_resume_failure(
            created.pending_turn_id,
            error_message="late",
            lease_id=a_record.resume_lease_id or "",
        )

    # B 持有的新 lease 仍能正常 release。
    released = store.release_resume_lease(
        created.pending_turn_id, lease_id=b_record.resume_lease_id or ""
    )
    assert released is not None
    assert released.state is PendingConversationTurnState.ACCEPTED_BY_HOST


@pytest.mark.unit
def test_sqlite_lease_mismatch_release_raises_lease_expired(tmp_path: Path) -> None:
    """SQLite 实现：A → cleanup → B → A 迟到写入路径全部 LeaseExpiredError。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    a_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    assert a_record.resume_lease_id
    assert a_record.state is PendingConversationTurnState.RESUMING

    cleaned = store.cleanup_stale_resuming(
        created.pending_turn_id,
        expected_updated_at=a_record.updated_at,
    )
    assert cleaned is not None
    assert cleaned.resume_lease_id is None
    assert cleaned.state is PendingConversationTurnState.ACCEPTED_BY_HOST

    b_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    assert b_record.resume_lease_id
    assert b_record.resume_lease_id != a_record.resume_lease_id

    with pytest.raises(LeaseExpiredError):
        store.release_resume_lease(created.pending_turn_id, lease_id=a_record.resume_lease_id or "")
    with pytest.raises(LeaseExpiredError):
        store.rebind_source_run_id_for_resume(
            created.pending_turn_id,
            new_source_run_id="run_a_late",
            lease_id=a_record.resume_lease_id or "",
        )
    with pytest.raises(LeaseExpiredError):
        store.record_resume_failure(
            created.pending_turn_id,
            error_message="late",
            lease_id=a_record.resume_lease_id or "",
        )

    # B 仍可正常 release。
    released = store.release_resume_lease(
        created.pending_turn_id, lease_id=b_record.resume_lease_id or ""
    )
    assert released is not None
    assert released.state is PendingConversationTurnState.ACCEPTED_BY_HOST


@pytest.mark.unit
def test_sqlite_lease_threading_double_connection(tmp_path: Path) -> None:
    """跨线程双连接：A acquire → cleanup（另一线程）→ A 迟到 rebind 必抛 LeaseExpiredError。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    a_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)

    cleanup_done = threading.Event()
    cleanup_error: list[BaseException] = []

    def _cleanup_in_other_thread() -> None:
        try:
            store.cleanup_stale_resuming(
                created.pending_turn_id,
                expected_updated_at=a_record.updated_at,
            )
        except BaseException as exc:  # pragma: no cover - 被主线程抓取
            cleanup_error.append(exc)
        finally:
            cleanup_done.set()

    thread = threading.Thread(target=_cleanup_in_other_thread, name="lease-cleanup")
    thread.start()
    thread.join(timeout=5.0)
    assert cleanup_done.is_set()
    assert not cleanup_error

    # B 在主线程重新 acquire，使状态再次进入 RESUMING（持新 lease）。
    b_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    assert b_record.resume_lease_id
    assert b_record.resume_lease_id != a_record.resume_lease_id

    # A 迟到 rebind 必失配。
    with pytest.raises(LeaseExpiredError):
        store.rebind_source_run_id_for_resume(
            created.pending_turn_id,
            new_source_run_id="run_a_late",
            lease_id=a_record.resume_lease_id or "",
        )


@pytest.mark.unit
def test_inmemory_cleanup_stale_resuming_skips_when_updated_at_advanced() -> None:
    """InMemory：cleanup 带 stale ``expected_updated_at`` 时不得抢占 fresh lease。

    覆盖"Host snapshot 判 stale → 期间被合法新 holder 重新 acquire → cleanup 必须 no-op"
    的 TOCTOU 窗口。
    """

    store = InMemoryPendingConversationTurnStore()
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    stale_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    assert stale_record.resume_lease_id

    # 模拟 lease 自动过期被 cleanup 接管，B 立刻重新 acquire 拿到 fresh lease。
    store.cleanup_stale_resuming(
        created.pending_turn_id, expected_updated_at=stale_record.updated_at
    )
    fresh_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    assert fresh_record.resume_lease_id
    assert fresh_record.resume_lease_id != stale_record.resume_lease_id

    # Host 持有的 stale snapshot 触发 cleanup：必须 no-op，不能抹掉 B 的 fresh lease。
    result = store.cleanup_stale_resuming(
        created.pending_turn_id, expected_updated_at=stale_record.updated_at
    )
    assert result is not None
    assert result.state is PendingConversationTurnState.RESUMING
    assert result.resume_lease_id == fresh_record.resume_lease_id

    # B 仍能正常 release。
    released = store.release_resume_lease(
        created.pending_turn_id, lease_id=fresh_record.resume_lease_id or ""
    )
    assert released is not None
    assert released.state is PendingConversationTurnState.ACCEPTED_BY_HOST


@pytest.mark.unit
def test_sqlite_cleanup_stale_resuming_skips_when_updated_at_advanced(tmp_path: Path) -> None:
    """SQLite：同结构 TOCTOU 防御：stale snapshot 不得抹掉 fresh lease。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    stale_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    assert stale_record.resume_lease_id

    store.cleanup_stale_resuming(
        created.pending_turn_id, expected_updated_at=stale_record.updated_at
    )
    fresh_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    assert fresh_record.resume_lease_id
    assert fresh_record.resume_lease_id != stale_record.resume_lease_id

    # Host stale snapshot 再触发 cleanup：必须 no-op。
    result = store.cleanup_stale_resuming(
        created.pending_turn_id, expected_updated_at=stale_record.updated_at
    )
    assert result is not None
    assert result.state is PendingConversationTurnState.RESUMING
    assert result.resume_lease_id == fresh_record.resume_lease_id

    released = store.release_resume_lease(
        created.pending_turn_id, lease_id=fresh_record.resume_lease_id or ""
    )
    assert released is not None
    assert released.state is PendingConversationTurnState.ACCEPTED_BY_HOST


@pytest.mark.unit
def test_inmemory_writes_after_cleanup_before_reacquire_raise_lease_expired() -> None:
    """InMemory：cleanup 后、B 还未 acquire 时，A 的所有迟到写入必抛 LeaseExpiredError。

    覆盖三个缺口：``release_resume_lease`` / ``rebind_source_run_id_for_resume`` /
    ``record_resume_failure`` 在 state 已被 cleanup 回退到非 RESUMING 时，必须统一
    把旧 lease 识别为过期，**不允许**沉默 no-op、不允许误改 ``last_resume_error_message``。
    """

    store = InMemoryPendingConversationTurnStore()
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    a_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)

    # cleanup 把 lease 抢占清空，state 回退；B 尚未 acquire。
    store.cleanup_stale_resuming(
        created.pending_turn_id, expected_updated_at=a_record.updated_at
    )

    snapshot_before = store.get_pending_turn(created.pending_turn_id)
    assert snapshot_before is not None
    assert snapshot_before.state is PendingConversationTurnState.ACCEPTED_BY_HOST
    assert snapshot_before.last_resume_error_message is None

    with pytest.raises(LeaseExpiredError):
        store.release_resume_lease(
            created.pending_turn_id, lease_id=a_record.resume_lease_id or ""
        )
    with pytest.raises(LeaseExpiredError):
        store.rebind_source_run_id_for_resume(
            created.pending_turn_id,
            new_source_run_id="run_a_late",
            lease_id=a_record.resume_lease_id or "",
        )
    with pytest.raises(LeaseExpiredError):
        store.record_resume_failure(
            created.pending_turn_id,
            error_message="late failure",
            lease_id=a_record.resume_lease_id or "",
        )

    snapshot_after = store.get_pending_turn(created.pending_turn_id)
    assert snapshot_after is not None
    # last_resume_error_message 不得被旧 holder 误写。
    assert snapshot_after.last_resume_error_message is None
    assert snapshot_after.state is PendingConversationTurnState.ACCEPTED_BY_HOST


@pytest.mark.unit
def test_sqlite_writes_after_cleanup_before_reacquire_raise_lease_expired(tmp_path: Path) -> None:
    """SQLite：同结构防御：cleanup 后、B 未 acquire 时旧 holder 三类写入全 LeaseExpired。"""

    host_store = HostStore(tmp_path / ".host" / "dayu_host.db")
    host_store.initialize_schema()
    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    a_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)

    store.cleanup_stale_resuming(
        created.pending_turn_id, expected_updated_at=a_record.updated_at
    )

    snapshot_before = store.get_pending_turn(created.pending_turn_id)
    assert snapshot_before is not None
    assert snapshot_before.state is PendingConversationTurnState.ACCEPTED_BY_HOST
    assert snapshot_before.last_resume_error_message is None

    with pytest.raises(LeaseExpiredError):
        store.release_resume_lease(
            created.pending_turn_id, lease_id=a_record.resume_lease_id or ""
        )
    with pytest.raises(LeaseExpiredError):
        store.rebind_source_run_id_for_resume(
            created.pending_turn_id,
            new_source_run_id="run_a_late",
            lease_id=a_record.resume_lease_id or "",
        )
    with pytest.raises(LeaseExpiredError):
        store.record_resume_failure(
            created.pending_turn_id,
            error_message="late failure",
            lease_id=a_record.resume_lease_id or "",
        )

    snapshot_after = store.get_pending_turn(created.pending_turn_id)
    assert snapshot_after is not None
    assert snapshot_after.last_resume_error_message is None
    assert snapshot_after.state is PendingConversationTurnState.ACCEPTED_BY_HOST
