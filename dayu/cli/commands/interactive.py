"""`dayu-cli interactive` 命令实现。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dayu.cli.conversation_labels import FileConversationLabelRegistry
from dayu.cli.dependency_setup import (
    _build_chat_service,
    _build_execution_options,
    _prepare_cli_host_dependencies,
    _resolve_interactive_session_id,
    setup_loglevel,
    setup_paths,
)
from dayu.cli.interactive_ui import interactive
from dayu.log import Log
from dayu.services.host_admin_service import HostAdminService
from dayu.services.protocols import HostAdminServiceProtocol
from dayu.state_dir_lock import StateDirSingleInstanceLock
from dayu.workspace_paths import build_interactive_state_dir

MODULE = "APP.INTERACTIVE"
_INTERACTIVE_SCENE_NAME = "interactive"
_RESTORED_TURN_LIMIT = 1
_RESTORED_MESSAGE_MAX_CHARS = 1200
_RESTORED_MESSAGE_SUFFIX = "\n..."
_RESTORED_PREVIOUS_TURN_HEADER = "----------- 上一轮对话 -----------"
_RESTORED_RESUME_HEADER = "----------- 对话恢复 -----------"
_RESTORED_USER_LABEL = "用户:"
_RESTORED_ASSISTANT_LABEL = "助手:"
_RESTORED_EMPTY_ASSISTANT = "(无最终回答)"


def run_interactive_command(args: argparse.Namespace) -> int:
    """执行交互式 CLI 命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        退出码，`0` 表示成功，`1` 表示单实例锁冲突。

    Raises:
        ValueError: 当 label 非法或 registry record 非法时抛出。
    """

    setup_loglevel(args)
    paths_config = setup_paths(args)
    execution_options = _build_execution_options(args)
    Log.info(f"工作目录: {paths_config.workspace_dir}", module=MODULE)
    (
        _workspace,
        _default_execution_options,
        scene_execution_acceptance_preparer,
        host,
        fins_runtime,
    ) = _prepare_cli_host_dependencies(
        workspace_config=paths_config,
        execution_options=execution_options,
    )
    service = _build_chat_service(
        host=host,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        fins_runtime=fins_runtime,
    )
    host_admin_service = HostAdminService(host=host)
    interactive_model = scene_execution_acceptance_preparer.resolve_scene_model(
        "interactive",
        execution_options,
    )
    Log.info(
        "使用模型: "
        f"{json.dumps(interactive_model, ensure_ascii=False, sort_keys=True)}",
        module=MODULE,
    )
    Log.info("进入交互模式... 按 Ctrl+D 发送 prompt... 按 Enter 换行... 按两次 Ctrl+D 退出...", module=MODULE)
    instance_lock = StateDirSingleInstanceLock(
        state_dir=build_interactive_state_dir(paths_config.workspace_dir),
        lock_file_name=".interactive.lock",
        lock_name="interactive 单实例锁",
    )
    try:
        instance_lock.acquire()
    except RuntimeError as exc:
        Log.error(str(exc), module=MODULE)
        return 1
    try:
        try:
            interactive_session_id, should_print_restore_context = _resolve_interactive_target(
                args,
                workspace_dir=paths_config.workspace_dir,
            )
        except ValueError as exc:
            Log.error(str(exc), module=MODULE)
            return 2
        if should_print_restore_context:
            _print_interactive_session_restore_context(
                host_admin_service=host_admin_service,
                session_id=interactive_session_id,
            )
        interactive(
            service,
            session_id=interactive_session_id,
            execution_options=execution_options,
            show_thinking=bool(getattr(args, "thinking", False)),
        )
        return 0
    finally:
        instance_lock.release()


def _resolve_interactive_target(
    args: argparse.Namespace,
    *,
    workspace_dir: Path,
) -> tuple[str, bool]:
    """根据 CLI 参数解析 interactive 应进入的 session。

    Args:
        args: 解析后的命令行参数。
        workspace_dir: 工作区根目录。

    Returns:
        二元组 ``(session_id, should_print_restore_context)``。

    Raises:
        ValueError: 当 label 非法或 registry record 非法时抛出。
    """

    label = _resolve_interactive_label(args)
    if label is not None:
        session_id, _ = _resolve_labeled_interactive_target(
            args,
            workspace_dir=workspace_dir,
            label=label,
        )
        return session_id, True
    return (
        _resolve_interactive_session_id(
            workspace_dir,
            new_session=bool(getattr(args, "new_session", False)),
        ),
        False,
    )


def _resolve_interactive_label(args: argparse.Namespace) -> str | None:
    """解析 interactive 命令的 label 参数。

    Args:
        args: 解析后的命令行参数。

    Returns:
        规范化后的 label；未提供时返回 ``None``。

    Raises:
        无。
    """

    normalized_label = str(getattr(args, "label", "") or "").strip()
    if not normalized_label:
        return None
    return normalized_label


def _resolve_labeled_interactive_target(
    args: argparse.Namespace,
    *,
    workspace_dir: Path,
    label: str,
) -> tuple[str, str]:
    """解析带 label 的 interactive 会话目标。

    Args:
        args: 解析后的命令行参数。
        workspace_dir: 工作区根目录。
        label: 已规范化的 conversation label。

    Returns:
        二元组 ``(session_id, scene_name)``。

    Raises:
        ValueError: 当 label 非法或 registry record 非法时抛出。
    """

    explicit_session_id = _resolve_label_session_id(args)
    explicit_scene_name = _resolve_label_scene_name(args)
    if explicit_session_id is not None:
        return explicit_session_id, explicit_scene_name
    registry = FileConversationLabelRegistry(workspace_dir)
    record = registry.get_or_create_record(
        label=label,
        scene_name=_INTERACTIVE_SCENE_NAME,
    )
    return record.session_id, record.scene_name


def _resolve_label_session_id(args: argparse.Namespace) -> str | None:
    """解析主代理可能注入的 label session_id。

    Args:
        args: 解析后的命令行参数。

    Returns:
        注入的 session_id；未提供时返回 ``None``。

    Raises:
        无。
    """

    normalized_session_id = str(getattr(args, "label_session_id", "") or "").strip()
    if not normalized_session_id:
        return None
    return normalized_session_id


def _resolve_label_scene_name(args: argparse.Namespace) -> str:
    """解析主代理可能注入的 label scene_name。

    Args:
        args: 解析后的命令行参数。

    Returns:
        label 对应的 scene_name；未提供时返回 ``interactive``。

    Raises:
        无。
    """

    normalized_scene_name = str(getattr(args, "label_scene_name", "") or "").strip()
    if normalized_scene_name:
        return normalized_scene_name
    return _INTERACTIVE_SCENE_NAME


def _print_interactive_session_restore_context(
    *,
    host_admin_service: HostAdminServiceProtocol,
    session_id: str,
) -> None:
    """打印 labeled conversation 恢复时的历史上下文提示。

    Args:
        host_admin_service: 宿主管理服务。
        session_id: labeled conversation 对应的 Host session ID。

    Returns:
        无。

    Raises:
        无。
    """

    turns = host_admin_service.list_session_recent_turns(
        session_id,
        limit=_RESTORED_TURN_LIMIT,
    )
    if not turns:
        return
    print(_RESTORED_PREVIOUS_TURN_HEADER, flush=True)
    for turn in turns:
        print(_RESTORED_USER_LABEL, flush=True)
        print(_truncate_restored_message(turn.user_text), flush=True)
        print(_RESTORED_ASSISTANT_LABEL, flush=True)
        print(_truncate_restored_message(turn.assistant_text) or _RESTORED_EMPTY_ASSISTANT, flush=True)
    print(_RESTORED_RESUME_HEADER, flush=True)


def _truncate_restored_message(text: str) -> str:
    """截断恢复提示中的长消息。

    Args:
        text: 原始消息文本。

    Returns:
        适合在进入 REPL 前展示的消息文本。

    Raises:
        无。
    """

    normalized = str(text or "").strip()
    if len(normalized) <= _RESTORED_MESSAGE_MAX_CHARS:
        return normalized
    content_length = _RESTORED_MESSAGE_MAX_CHARS - len(_RESTORED_MESSAGE_SUFFIX)
    return normalized[:content_length].rstrip() + _RESTORED_MESSAGE_SUFFIX
