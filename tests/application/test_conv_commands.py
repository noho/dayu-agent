"""conv_commands 测试。"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from dayu.cli.commands import conv as conv_commands_module
from dayu.cli.conversation_label_locks import ConversationLabelLease
from dayu.cli.conversation_labels import FileConversationLabelRegistry
from dayu.services.contracts import SessionAdminView
from dayu.services.protocols import HostAdminServiceProtocol

pytestmark = pytest.mark.unit


def _build_runtime(
    workspace_root: Path,
    *,
    service: HostAdminServiceProtocol,
) -> SimpleNamespace:
    """构造供 `conv` 命令测试使用的最小 runtime。"""

    return SimpleNamespace(
        paths=SimpleNamespace(workspace_root=workspace_root),
        host_admin_service=service,
    )


def _build_session_view(
    *,
    session_id: str,
    state: str = "active",
    scene_name: str = "interactive",
    last_activity_at: str = "2026-04-22T08:00:00+00:00",
    first_question_preview: str = "",
    last_question_preview: str = "",
) -> SessionAdminView:
    """构造测试使用的 SessionAdminView。"""

    return SessionAdminView(
        session_id=session_id,
        source="cli",
        state=state,
        scene_name=scene_name,
        created_at="2026-04-22T07:00:00+00:00",
        last_activity_at=last_activity_at,
        turn_count=1,
        first_question_preview=first_question_preview,
        last_question_preview=last_question_preview,
    )


def test_run_conv_list_prints_empty_message_when_registry_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """空 registry 下 `conv list` 应稳定返回空列表提示。"""

    service = cast(
        HostAdminServiceProtocol,
        SimpleNamespace(list_sessions=lambda **_kwargs: []),
    )
    monkeypatch.setattr(
        conv_commands_module,
        "_build_host_runtime",
        lambda _args: _build_runtime(tmp_path, service=service),
    )

    exit_code = conv_commands_module._run_conv_list_command(
        argparse.Namespace(base=str(tmp_path), config=None)
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "无 labeled conversation 记录" in captured.out


def test_run_conv_list_only_renders_active_rows_and_prunes_missing_registry_records(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`conv list` 默认只显示 active，并清理已漂移的 registry record。"""

    registry = FileConversationLabelRegistry(tmp_path)
    apple = registry.get_or_create_record(label="apple", scene_name="interactive").record
    ghost = registry.get_or_create_record(label="ghost", scene_name="prompt_mt").record
    closed = registry.get_or_create_record(label="closed", scene_name="interactive").record
    matched_session = _build_session_view(
        session_id=apple.session_id,
        scene_name="interactive",
        first_question_preview="苹果这季度增长来自哪里？",
        last_question_preview="这句不应优先展示",
    )
    closed_session = _build_session_view(
        session_id=closed.session_id,
        state="closed",
        scene_name="interactive",
        first_question_preview="这条默认不该显示",
    )
    service = cast(
        HostAdminServiceProtocol,
        SimpleNamespace(
            list_sessions=lambda **kwargs: (
                [matched_session]
                if kwargs == {"state": "active", "source": "cli"}
                else [matched_session, closed_session]
            ),
            get_session=lambda session_id: {
                apple.session_id: matched_session,
                closed.session_id: closed_session,
            }.get(session_id),
        ),
    )
    monkeypatch.setattr(
        conv_commands_module,
        "_build_host_runtime",
        lambda _args: _build_runtime(tmp_path, service=service),
    )

    exit_code = conv_commands_module._run_conv_list_command(
        argparse.Namespace(base=str(tmp_path), config=None, show_all=False)
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "LABEL" in captured.out
    assert "apple" in captured.out
    assert apple.session_id not in captured.out
    assert "苹果这季度增长来自哪里？" in captured.out
    assert "这句不应优先展示" not in captured.out
    assert "closed" not in captured.out
    assert "ghost" not in captured.out
    assert ghost.session_id not in captured.out
    assert registry.get_record("ghost") is None


def test_run_conv_list_with_all_renders_closed_rows_but_still_prunes_missing_records(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`conv list --all` 应显示 closed，但仍不显示漂移 registry record。"""

    registry = FileConversationLabelRegistry(tmp_path)
    apple = registry.get_or_create_record(label="apple", scene_name="interactive").record
    closed = registry.get_or_create_record(label="closed", scene_name="interactive").record
    registry.get_or_create_record(label="ghost", scene_name="prompt_mt")
    active_session = _build_session_view(session_id=apple.session_id, first_question_preview="active")
    closed_session = _build_session_view(
        session_id=closed.session_id,
        state="closed",
        first_question_preview="closed",
    )
    service = cast(
        HostAdminServiceProtocol,
        SimpleNamespace(
            list_sessions=lambda **kwargs: (
                [active_session, closed_session]
                if kwargs == {"state": None, "source": "cli"}
                else []
            ),
            get_session=lambda session_id: {
                apple.session_id: active_session,
                closed.session_id: closed_session,
            }.get(session_id),
        ),
    )
    monkeypatch.setattr(
        conv_commands_module,
        "_build_host_runtime",
        lambda _args: _build_runtime(tmp_path, service=service),
    )

    exit_code = conv_commands_module._run_conv_list_command(
        argparse.Namespace(base=str(tmp_path), config=None, show_all=True)
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "apple" in captured.out
    assert "closed" in captured.out
    assert "ghost" not in captured.out
    assert registry.get_record("ghost") is None


def test_run_conv_status_renders_matched_label(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`conv status --label` 命中时应展示对应单条记录。"""

    registry = FileConversationLabelRegistry(tmp_path)
    record = registry.get_or_create_record(label="alpha", scene_name="interactive").record
    session = _build_session_view(
        session_id=record.session_id,
        first_question_preview="",
        last_question_preview="最后一问预览",
    )
    service = cast(
        HostAdminServiceProtocol,
        SimpleNamespace(get_session=lambda session_id: session if session_id == record.session_id else None),
    )
    monkeypatch.setattr(
        conv_commands_module,
        "_build_host_runtime",
        lambda _args: _build_runtime(tmp_path, service=service),
    )

    exit_code = conv_commands_module._run_conv_status_command(
        argparse.Namespace(base=str(tmp_path), config=None, label="alpha")
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "LABEL: alpha" in captured.out
    assert f"SESSION_ID: {record.session_id}" in captured.out
    assert "OVERVIEW: 最后一问预览" in captured.out


def test_run_conv_status_returns_error_when_label_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """不存在的 label 应明确报错并返回非 0。"""

    service = cast(
        HostAdminServiceProtocol,
        SimpleNamespace(get_session=lambda _session_id: None),
    )
    monkeypatch.setattr(
        conv_commands_module,
        "_build_host_runtime",
        lambda _args: _build_runtime(tmp_path, service=service),
    )

    exit_code = conv_commands_module._run_conv_status_command(
        argparse.Namespace(base=str(tmp_path), config=None, label="missing")
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "label 不存在: missing" in captured.err


def test_run_conv_status_prunes_missing_registry_record_and_returns_not_found(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """registry 命中漂移 record 时，`conv status` 应清理后按不存在处理。"""

    registry = FileConversationLabelRegistry(tmp_path)
    record = registry.get_or_create_record(label="orphan", scene_name="interactive").record
    service = cast(
        HostAdminServiceProtocol,
        SimpleNamespace(get_session=lambda _session_id: None),
    )
    monkeypatch.setattr(
        conv_commands_module,
        "_build_host_runtime",
        lambda _args: _build_runtime(tmp_path, service=service),
    )

    exit_code = conv_commands_module._run_conv_status_command(
        argparse.Namespace(base=str(tmp_path), config=None, label="orphan")
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert f"label 不存在: {record.label}" in captured.err
    assert registry.get_record("orphan") is None
    assert record.session_id.startswith("cli_conv_")


def test_run_conv_remove_closes_session_and_deletes_label(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`conv remove --label` 应关闭底层 session 并删除 registry record。"""

    registry = FileConversationLabelRegistry(tmp_path)
    record = registry.get_or_create_record(label="alpha", scene_name="interactive").record
    session = _build_session_view(session_id=record.session_id)
    closed_session_ids: list[str] = []
    service = cast(
        HostAdminServiceProtocol,
        SimpleNamespace(
            get_session=lambda session_id: session if session_id == record.session_id else None,
            close_session=lambda session_id: (
                closed_session_ids.append(session_id) or (session, [])
            ),
        ),
    )
    monkeypatch.setattr(
        conv_commands_module,
        "_build_host_runtime",
        lambda _args: _build_runtime(tmp_path, service=service),
    )

    exit_code = conv_commands_module._run_conv_remove_command(
        argparse.Namespace(base=str(tmp_path), config=None, label="alpha")
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "已移除 label: alpha" in captured.out
    assert closed_session_ids == [record.session_id]
    assert registry.get_record("alpha") is None


def test_run_conv_remove_deletes_orphaned_label_without_closing_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`conv remove` 命中漂移 record 时应直接删除 label。"""

    registry = FileConversationLabelRegistry(tmp_path)
    registry.get_or_create_record(label="ghost", scene_name="prompt_mt")
    service = cast(
        HostAdminServiceProtocol,
        SimpleNamespace(get_session=lambda _session_id: None),
    )
    monkeypatch.setattr(
        conv_commands_module,
        "_build_host_runtime",
        lambda _args: _build_runtime(tmp_path, service=service),
    )

    exit_code = conv_commands_module._run_conv_remove_command(
        argparse.Namespace(base=str(tmp_path), config=None, label="ghost")
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "已移除 label: ghost" in captured.out
    assert registry.get_record("ghost") is None


def test_run_conv_remove_rejects_busy_label(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`conv remove` 命中占用中的 label 时应明确失败。"""

    registry = FileConversationLabelRegistry(tmp_path)
    registry.get_or_create_record(label="alpha", scene_name="interactive")
    service = cast(
        HostAdminServiceProtocol,
        SimpleNamespace(get_session=lambda _session_id: None),
    )
    monkeypatch.setattr(
        conv_commands_module,
        "_build_host_runtime",
        lambda _args: _build_runtime(tmp_path, service=service),
    )

    busy_lease = ConversationLabelLease(tmp_path, "alpha")
    busy_lease.acquire()
    try:
        exit_code = conv_commands_module._run_conv_remove_command(
            argparse.Namespace(base=str(tmp_path), config=None, label="alpha")
        )
    finally:
        busy_lease.release()

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "label 正在使用中: alpha" in captured.err
    assert registry.get_record("alpha") is not None
