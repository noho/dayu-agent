"""``Host.list_conversation_session_turn_excerpts`` 历史读 read model 契约测试。

Phase 1（``#116``）：

- 历史真源切到 ``archive.history_archive.turns``，**不**从 ``runtime_transcript``
  投影；
- read model ``ConversationSessionTurnExcerpt`` 含 ``reasoning_text``，映射自
  ``history_archive.turns[*].assistant_reasoning``，无 reasoning 的轮次为空字符串；
- 顺序稳定（旧 → 新）；
- 空值语义：archive 缺失 / 空 history → 返回 ``[]``；
- limit 边界：``<= 0`` / 大于 turn 数 / 等于 turn 数。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.host.conversation_session_archive import (
    ConversationHistoryTurnRecord,
    ConversationSessionArchive,
)
from dayu.host.conversation_store import (
    ConversationTurnRecord,
    FileConversationSessionArchiveStore,
)
from dayu.host.host import Host
from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore
from tests.application.conftest import (
    StubHostExecutor,
    StubRunRegistry,
    StubSessionRegistry,
)


def _build_host(archive_store: FileConversationSessionArchiveStore) -> Host:
    """构造仅注入历史读所需依赖的 Host。"""

    return Host(
        executor=StubHostExecutor(),
        session_registry=StubSessionRegistry(),
        run_registry=StubRunRegistry(),
        reply_outbox_store=InMemoryReplyOutboxStore(),
        archive_store=archive_store,
    )


def _seed_archive_with_turns(
    store: FileConversationSessionArchiveStore,
    *,
    session_id: str,
    turns: list[tuple[str, str, str]],
) -> ConversationSessionArchive:
    """以 ``(user, assistant, reasoning)`` 列表种入若干轮历史。

    runtime/history 子视图同步推进，``turn_id`` 一一对应。
    """

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


@pytest.mark.unit
def test_list_excerpts_reads_from_history_archive_with_reasoning(tmp_path: Path) -> None:
    """读源是 ``history_archive``，``reasoning_text`` 透出 ``assistant_reasoning``。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    _seed_archive_with_turns(
        store,
        session_id="sess_a",
        turns=[
            ("Q1", "A1", "R1"),
            ("Q2", "A2", "R2"),
        ],
    )
    host = _build_host(store)

    excerpts = host.list_conversation_session_turn_excerpts("sess_a", limit=10)

    assert [(e.user_text, e.assistant_text, e.reasoning_text) for e in excerpts] == [
        ("Q1", "A1", "R1"),
        ("Q2", "A2", "R2"),
    ]


@pytest.mark.unit
def test_list_excerpts_order_is_old_to_new(tmp_path: Path) -> None:
    """顺序契约：旧 → 新。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    _seed_archive_with_turns(
        store,
        session_id="sess_order",
        turns=[("Q1", "A1", ""), ("Q2", "A2", ""), ("Q3", "A3", "")],
    )
    host = _build_host(store)

    excerpts = host.list_conversation_session_turn_excerpts("sess_order", limit=10)

    assert [e.user_text for e in excerpts] == ["Q1", "Q2", "Q3"]


@pytest.mark.unit
def test_list_excerpts_empty_reasoning_yields_empty_string(tmp_path: Path) -> None:
    """无 reasoning 的轮次 ``reasoning_text == ""``。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    _seed_archive_with_turns(
        store,
        session_id="sess_empty_r",
        turns=[("Q1", "A1", "")],
    )
    host = _build_host(store)

    excerpts = host.list_conversation_session_turn_excerpts("sess_empty_r", limit=1)

    assert len(excerpts) == 1
    assert excerpts[0].reasoning_text == ""


@pytest.mark.unit
def test_list_excerpts_returns_empty_for_missing_archive(tmp_path: Path) -> None:
    """archive 文件缺失时按 §1.5 返回空列表。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    host = _build_host(store)

    assert host.list_conversation_session_turn_excerpts("missing", limit=5) == []


@pytest.mark.unit
def test_list_excerpts_returns_empty_for_empty_history(tmp_path: Path) -> None:
    """archive 存在但 history 为空时返回空列表。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    store.save(ConversationSessionArchive.create_empty("sess_empty"), expected_revision=None)
    host = _build_host(store)

    assert host.list_conversation_session_turn_excerpts("sess_empty", limit=5) == []


@pytest.mark.unit
def test_list_excerpts_limit_non_positive_returns_empty(tmp_path: Path) -> None:
    """``limit <= 0`` 直接返回空列表，不读 archive。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    _seed_archive_with_turns(
        store,
        session_id="sess_limit",
        turns=[("Q1", "A1", "R1")],
    )
    host = _build_host(store)

    assert host.list_conversation_session_turn_excerpts("sess_limit", limit=0) == []
    assert host.list_conversation_session_turn_excerpts("sess_limit", limit=-1) == []


@pytest.mark.unit
def test_list_excerpts_limit_larger_than_turn_count_returns_all(tmp_path: Path) -> None:
    """``limit`` 超过总轮次时返回全部。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    _seed_archive_with_turns(
        store,
        session_id="sess_lim_lg",
        turns=[("Q1", "A1", ""), ("Q2", "A2", "")],
    )
    host = _build_host(store)

    excerpts = host.list_conversation_session_turn_excerpts("sess_lim_lg", limit=100)

    assert [e.user_text for e in excerpts] == ["Q1", "Q2"]


@pytest.mark.unit
def test_list_excerpts_limit_equals_turn_count(tmp_path: Path) -> None:
    """``limit`` 等于总轮次时返回全部。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    _seed_archive_with_turns(
        store,
        session_id="sess_lim_eq",
        turns=[("Q1", "A1", ""), ("Q2", "A2", "")],
    )
    host = _build_host(store)

    excerpts = host.list_conversation_session_turn_excerpts("sess_lim_eq", limit=2)

    assert len(excerpts) == 2
    assert [e.user_text for e in excerpts] == ["Q1", "Q2"]


@pytest.mark.unit
def test_list_excerpts_limit_smaller_takes_most_recent(tmp_path: Path) -> None:
    """``limit`` 小于总轮次时取最近 N 条，仍按旧 → 新。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    _seed_archive_with_turns(
        store,
        session_id="sess_lim_sm",
        turns=[("Q1", "A1", ""), ("Q2", "A2", ""), ("Q3", "A3", "")],
    )
    host = _build_host(store)

    excerpts = host.list_conversation_session_turn_excerpts("sess_lim_sm", limit=2)

    assert [e.user_text for e in excerpts] == ["Q2", "Q3"]


@pytest.mark.unit
def test_list_excerpts_does_not_consume_runtime_transcript(tmp_path: Path) -> None:
    """``runtime_transcript`` 不再是历史真源：人为构造 history 与 runtime 不同
    的 ``user_text`` / ``assistant_text``，验证返回值来自 history 子视图。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    archive = ConversationSessionArchive.create_empty("sess_split")
    archive = store.save(archive, expected_revision=None)
    prev_revision = archive.revision
    runtime_turn = ConversationTurnRecord(
        turn_id="t1",
        scene_name="interactive",
        user_text="runtime-only-user",
        assistant_final="runtime-only-assistant",
    )
    history_turn = ConversationHistoryTurnRecord(
        turn_id="t1",
        scene_name="interactive",
        user_text="history-user",
        assistant_text="history-assistant",
        assistant_reasoning="history-reason",
        created_at=runtime_turn.created_at,
    )
    archive = archive.with_next_turn(runtime_turn, history_turn)
    store.save(archive, expected_revision=prev_revision)

    host = _build_host(store)
    excerpts = host.list_conversation_session_turn_excerpts("sess_split", limit=5)

    assert len(excerpts) == 1
    assert excerpts[0].user_text == "history-user"
    assert excerpts[0].assistant_text == "history-assistant"
    assert excerpts[0].reasoning_text == "history-reason"


@pytest.mark.unit
def test_list_excerpts_repeat_reads_are_stable(tmp_path: Path) -> None:
    """重复读取相同 archive 字段顺序稳定（旧 → 新），多次结果相等。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    _seed_archive_with_turns(
        store,
        session_id="sess_stable",
        turns=[("Q1", "A1", "R1"), ("Q2", "A2", "")],
    )
    host = _build_host(store)

    first = host.list_conversation_session_turn_excerpts("sess_stable", limit=10)
    second = host.list_conversation_session_turn_excerpts("sess_stable", limit=10)

    assert first == second


@pytest.mark.unit
def test_list_excerpts_supports_migrated_legacy_session(tmp_path: Path) -> None:
    """老会话经 ``conversation_archive_init`` 迁移后能完整读出，
    ``reasoning_text == ""``、其他字段齐全。"""

    import json

    from dayu.cli.workspace_migrations.conversation_archive_init import (
        migrate_conversation_archive_init,
    )
    from dayu.workspace_paths import CONVERSATION_STORE_RELATIVE_DIR

    workspace_root = tmp_path / "workspace"
    target_dir = workspace_root / CONVERSATION_STORE_RELATIVE_DIR
    target_dir.mkdir(parents=True)

    legacy_payload = {
        "schema": "conversation_transcript/v1",
        "session_id": "sess_legacy",
        "revision": "rev0",
        "created_at": "2026-04-01T00:00:00+00:00",
        "turns": [
            {
                "turn_id": "t1",
                "scene_name": "interactive",
                "user_text": "老问题",
                "assistant_final": "老回答",
                "created_at": "2026-04-01T00:01:00+00:00",
                "tool_uses": [],
                "warnings": [],
                "errors": [],
                "degraded": False,
            }
        ],
    }
    legacy_path = target_dir / "sess_legacy.json"
    legacy_path.write_text(json.dumps(legacy_payload, ensure_ascii=False), encoding="utf-8")

    rewritten = migrate_conversation_archive_init(workspace_root)
    assert rewritten == 1

    store = FileConversationSessionArchiveStore(target_dir)
    host = _build_host(store)

    excerpts = host.list_conversation_session_turn_excerpts("sess_legacy", limit=10)

    assert len(excerpts) == 1
    assert excerpts[0].user_text == "老问题"
    assert excerpts[0].assistant_text == "老回答"
    assert excerpts[0].reasoning_text == ""
    assert excerpts[0].created_at == "2026-04-01T00:01:00+00:00"


@pytest.mark.unit
def test_list_excerpts_returns_empty_for_corrupt_json(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """JSON 损坏的 archive 文件按 §1.7 读路径列降级为空列表 + warning。"""

    conv_dir = tmp_path / "conv"
    conv_dir.mkdir()
    (conv_dir / "sess_corrupt.json").write_text("{not-json", encoding="utf-8")
    store = FileConversationSessionArchiveStore(conv_dir)
    host = _build_host(store)

    with caplog.at_level("WARNING"):
        result = host.list_conversation_session_turn_excerpts("sess_corrupt", limit=5)

    assert result == []
    assert any("历史读 archive 损坏" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_list_excerpts_returns_empty_for_invalid_schema(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """schema 非法（缺 ``runtime_transcript`` 等）时降级为空列表 + warning。"""

    import json as _json

    conv_dir = tmp_path / "conv"
    conv_dir.mkdir()
    bad_payload = {"schema": "conversation_session_archive/v1", "session_id": "sess_bad"}
    (conv_dir / "sess_bad.json").write_text(_json.dumps(bad_payload), encoding="utf-8")
    store = FileConversationSessionArchiveStore(conv_dir)
    host = _build_host(store)

    with caplog.at_level("WARNING"):
        result = host.list_conversation_session_turn_excerpts("sess_bad", limit=5)

    assert result == []
    assert any("历史读 archive 损坏" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_list_excerpts_returns_empty_for_unmigrated_legacy_schema(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """旧 transcript schema 未迁移时降级为空列表 + warning（不抛 RuntimeError）。"""

    import json as _json

    conv_dir = tmp_path / "conv"
    conv_dir.mkdir()
    legacy_payload = {
        "schema": "conversation_transcript/v1",
        "session_id": "sess_legacy_no_mig",
        "revision": "rev0",
        "created_at": "2026-04-01T00:00:00+00:00",
        "turns": [],
    }
    (conv_dir / "sess_legacy_no_mig.json").write_text(
        _json.dumps(legacy_payload), encoding="utf-8"
    )
    store = FileConversationSessionArchiveStore(conv_dir)
    host = _build_host(store)

    with caplog.at_level("WARNING"):
        result = host.list_conversation_session_turn_excerpts(
            "sess_legacy_no_mig", limit=5
        )

    assert result == []
    assert any("历史读 archive 损坏" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_session_digest_returns_empty_for_corrupt_archive(tmp_path: Path) -> None:
    """``get_conversation_session_digest`` 同样对损坏 archive 降级为空摘要。"""

    conv_dir = tmp_path / "conv"
    conv_dir.mkdir()
    (conv_dir / "sess_dg.json").write_text("{not-json", encoding="utf-8")
    store = FileConversationSessionArchiveStore(conv_dir)
    host = _build_host(store)

    digest = host.get_conversation_session_digest("sess_dg")

    assert digest.turn_count == 0
    assert digest.first_question_preview == ""
    assert digest.last_question_preview == ""


@pytest.mark.unit
def test_session_digest_reads_from_history_archive(tmp_path: Path) -> None:
    """``get_conversation_session_digest`` 与历史读共享 ``history_archive``
    真源（§1.1：运行态子视图不得被任何"读历史"代码路径直接消费）。"""

    store = FileConversationSessionArchiveStore(tmp_path / "conv")
    _seed_archive_with_turns(
        store,
        session_id="sess_dg_hist",
        turns=[("第一问", "第一答", ""), ("第二问", "第二答", "")],
    )
    host = _build_host(store)

    digest = host.get_conversation_session_digest("sess_dg_hist")

    assert digest.turn_count == 2
    assert digest.first_question_preview == "第一问"
    assert digest.last_question_preview == "第二问"
