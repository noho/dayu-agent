"""`dayu-cli interactive` 命令实现。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dayu.cli.dependency_setup import (
    _bind_interactive_session_id,
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
_CLI_SESSION_SOURCE = "cli"
_ACTIVE_SESSION_STATE = "active"
_RESTORED_TURN_LIMIT = 1
_RESTORED_MESSAGE_MAX_CHARS = 1200
_RESTORED_MESSAGE_SUFFIX = "\n..."
_RESTORED_PREVIOUS_TURN_HEADER = "----------- 上一轮对话 -----------"
_RESTORED_RESUME_HEADER = "----------- 对话恢复 -----------"
_RESTORED_EMPTY_MESSAGE = "未找到已持久化的上一轮对话。"
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
        无。
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
        interactive_session_id = _resolve_interactive_session_id_from_args(
            args,
            workspace_dir=paths_config.workspace_dir,
            host_admin_service=host_admin_service,
        )
        if interactive_session_id is None:
            return 1
        if str(getattr(args, "session_id", "") or "").strip():
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


def _resolve_interactive_session_id_from_args(
    args: argparse.Namespace,
    *,
    workspace_dir: Path,
    host_admin_service: HostAdminServiceProtocol,
) -> str | None:
    """根据 CLI 参数解析 interactive 应进入的 session。

    Args:
        args: 解析后的命令行参数。
        workspace_dir: 工作区根目录。
        host_admin_service: 宿主管理服务。

    Returns:
        目标 session_id；参数非法或 session 不符合 interactive 语义时返回 ``None``。

    Raises:
        无。
    """

    raw_session_id = str(getattr(args, "session_id", "") or "").strip()
    if raw_session_id:
        session = host_admin_service.get_session(raw_session_id)
        if session is None:
            Log.error(f"interactive session 不存在: {raw_session_id}", module=MODULE)
            return None
        if session.source != _CLI_SESSION_SOURCE or session.scene_name != _INTERACTIVE_SCENE_NAME:
            Log.error(f"session 不是 interactive 会话: {raw_session_id}", module=MODULE)
            return None
        if session.state != _ACTIVE_SESSION_STATE:
            Log.error(f"interactive session 已关闭: {raw_session_id}", module=MODULE)
            return None
        return _bind_interactive_session_id(workspace_dir, raw_session_id)
    return _resolve_interactive_session_id(
        workspace_dir,
        new_session=bool(getattr(args, "new_session", False)),
    )


def _print_interactive_session_restore_context(
    *,
    host_admin_service: HostAdminServiceProtocol,
    session_id: str,
) -> None:
    """打印显式恢复 interactive session 时的历史上下文提示。

    Args:
        host_admin_service: 宿主管理服务。
        session_id: 已校验并绑定的 interactive session ID。

    Returns:
        无。

    Raises:
        无。
    """

    turns = host_admin_service.list_interactive_session_recent_turns(
        session_id,
        limit=_RESTORED_TURN_LIMIT,
    )
    print(_RESTORED_PREVIOUS_TURN_HEADER, flush=True)
    if not turns:
        print(_RESTORED_EMPTY_MESSAGE, flush=True)
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
