"""conv_commands 测试。"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from dayu.cli.commands import conv as conv_commands_module
from dayu.cli.conversation_labels import FileConversationLabelRegistry, build_cli_conversation_session_id
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
        conversation_summary="不会被 conv overview 采用",
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


def test_run_conv_list_renders_registry_rows_and_missing_session_stably(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`conv list` 应展示命中 session，并对缺失 session 输出稳定占位值。"""

    registry = FileConversationLabelRegistry(tmp_path)
    apple = registry.get_or_create_record(label="apple", scene_name="interactive")
    ghost = registry.get_or_create_record(label="ghost", scene_name="prompt_mt")
    matched_session = _build_session_view(
        session_id=apple.session_id,
        scene_name="interactive",
        first_question_preview="苹果这季度增长来自哪里？",
        last_question_preview="这句不应优先展示",
    )
    service = cast(
        HostAdminServiceProtocol,
        SimpleNamespace(list_sessions=lambda **_kwargs: [matched_session]),
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
    assert "LABEL" in captured.out
    assert "SESSION_ID" in captured.out
    assert "apple" in captured.out
    assert apple.session_id in captured.out
    assert "苹果这季度增长来自哪里？" in captured.out
    assert "这句不应优先展示" not in captured.out
    assert "ghost" in captured.out
    assert ghost.session_id in captured.out
    assert "missing" in captured.out


def test_run_conv_status_renders_matched_label(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`conv status --label` 命中时应展示对应单条记录。"""

    registry = FileConversationLabelRegistry(tmp_path)
    record = registry.get_or_create_record(label="alpha", scene_name="interactive")
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
    assert "alpha" in captured.out
    assert record.session_id in captured.out
    assert "最后一问预览" in captured.out


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


def test_run_conv_status_keeps_working_when_host_session_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """registry 存在但 Host session 缺失时，`conv status` 仍应输出稳定占位值。"""

    registry = FileConversationLabelRegistry(tmp_path)
    record = registry.get_or_create_record(label="orphan", scene_name="interactive")
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
    assert exit_code == 0
    assert "orphan" in captured.out
    assert record.session_id in captured.out
    assert "missing" in captured.out
    assert build_cli_conversation_session_id("orphan") == record.session_id
