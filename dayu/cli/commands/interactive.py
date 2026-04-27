"""`dayu-cli interactive` 命令实现。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dayu.cli.conversation_labels import FileConversationLabelRegistry
from dayu.cli.dependency_setup import (
    _build_chat_service,
    _build_execution_options,
    _prepare_cli_host_dependencies,
    _resolve_interactive_session_id,
    get_cli_shutdown_coordinator,
    setup_loglevel,
    setup_paths,
)
from dayu.cli.conversation_label_locks import ConversationLabelLease
from dayu.cli.interactive_ui import interactive
from dayu.cli.labeled_conversations import resolve_labeled_conversation_target
from dayu.contracts.infrastructure import PromptAssetStoreProtocol
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
    label = _resolve_interactive_label(args)
    label_lease: ConversationLabelLease | None = None
    instance_lock = StateDirSingleInstanceLock(
        state_dir=build_interactive_state_dir(paths_config.workspace_dir),
        lock_file_name=".interactive.lock",
        lock_name="interactive 单实例锁",
    )
    try:
        instance_lock.acquire()
    except RuntimeError:
        Log.error(
            _build_interactive_instance_busy_message(
                workspace_dir=paths_config.workspace_dir,
                label=label,
            ),
            module=MODULE,
        )
        return 1
    try:
        if label is not None:
            try:
                label_lease = ConversationLabelLease(paths_config.workspace_dir, label)
                label_lease.acquire()
            except RuntimeError as exc:
                Log.error(str(exc), module=MODULE)
                return 2
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
        try:
            (
                interactive_session_id,
                interactive_scene_name,
                should_print_restore_context,
                label_created,
                recreated_from_closed,
            ) = _resolve_interactive_target(
                args,
                workspace_dir=paths_config.workspace_dir,
                prompt_asset_store=getattr(paths_config, "prompt_asset_store", None),
                host_admin_service=host_admin_service,
            )
        except ValueError as exc:
            Log.error(str(exc), module=MODULE)
            return 2
        interactive_model = scene_execution_acceptance_preparer.resolve_scene_model(
            interactive_scene_name,
            execution_options,
        )
        Log.info(
            "使用模型: "
            f"{json.dumps(asdict(interactive_model), ensure_ascii=False, sort_keys=True)}",
            module=MODULE,
        )
        if label is not None and recreated_from_closed:
            Log.info(
                f"label 对应的旧对话已关闭，现将基于同名 label 创建新对话: {label}",
                module=MODULE,
            )
        Log.info("进入交互模式，按Ctrl+D发送prompt / 按Ctrl+C中断prompt / 按Enter换行 / 按两次Ctrl+D退出", module=MODULE)
        if label is not None:
            Log.info(
                (
                    f"执行带标签 interactive，新创建标签: {label}"
                    if label_created
                    else f"执行带标签 interactive，恢复标签: {label}"
                ),
                module=MODULE,
            )
        if should_print_restore_context:
            _print_interactive_session_restore_context(
                host_admin_service=host_admin_service,
                session_id=interactive_session_id,
            )
        interactive(
            service,
            session_id=interactive_session_id,
            scene_name=interactive_scene_name,
            execution_options=execution_options,
            show_thinking=bool(getattr(args, "thinking", False)),
            run_lifecycle_observer=get_cli_shutdown_coordinator(),
        )
        return 0
    finally:
        if label_lease is not None:
            label_lease.release()
        instance_lock.release()


def _resolve_interactive_target(
    args: argparse.Namespace,
    *,
    workspace_dir: Path,
    prompt_asset_store: PromptAssetStoreProtocol | None = None,
    host_admin_service: HostAdminServiceProtocol | None = None,
) -> tuple[str, str, bool, bool, bool]:
    """根据 CLI 参数解析 interactive 应进入的 session。

    Args:
        args: 解析后的命令行参数。
        workspace_dir: 工作区根目录。
        prompt_asset_store: 可选 prompt 资产仓储；提供时会校验 scene 是否允许多轮会话。
        host_admin_service: 可选 HostAdmin service；提供时会清理不可恢复 registry record。

    Returns:
        五元组 ``(session_id, scene_name, should_print_restore_context, label_created, recreated_from_closed)``。

    Raises:
        ValueError: 当 label 非法或 registry record 非法时抛出。
    """

    label = _resolve_interactive_label(args)
    if label is not None:
        session_id, scene_name, created, recreated_from_closed = _resolve_labeled_interactive_target(
            args,
            workspace_dir=workspace_dir,
            label=label,
            prompt_asset_store=prompt_asset_store,
            host_admin_service=host_admin_service,
        )
        return session_id, scene_name, True, created, recreated_from_closed
    return (
        _resolve_interactive_session_id(
            workspace_dir,
            new_session=bool(getattr(args, "new_session", False)),
        ),
        _INTERACTIVE_SCENE_NAME,
        False,
        False,
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


def _build_interactive_instance_busy_message(*, workspace_dir: Path, label: str | None) -> str:
    """构造 interactive 单实例锁冲突时的用户提示。

    Args:
        workspace_dir: 当前工作区根目录。
        label: 本次命令是否显式提供了 label。

    Returns:
        面向 CLI 用户的稳定错误提示。

    Raises:
        无。
    """

    if label is None:
        return (
            "当前已有 interactive 在运行: "
            f"state_dir={build_interactive_state_dir(workspace_dir)}"
        )
    return (
        "当前已有 interactive 在运行，不能再启动第二个 interactive --label: "
        f"state_dir={build_interactive_state_dir(workspace_dir)}。"
        "请等待其退出后重试；如果只是想并发发起带标签对话，请改用 prompt --label"
    )


def _resolve_labeled_interactive_target(
    args: argparse.Namespace,
    *,
    workspace_dir: Path,
    label: str,
    prompt_asset_store: PromptAssetStoreProtocol | None = None,
    host_admin_service: HostAdminServiceProtocol | None = None,
) -> tuple[str, str, bool, bool]:
    """解析带 label 的 interactive 会话目标。

    Args:
        args: 解析后的命令行参数。
        workspace_dir: 工作区根目录。
        label: 已规范化的 conversation label。
        prompt_asset_store: 可选 prompt 资产仓储；提供时会校验 scene 是否允许多轮会话。
        host_admin_service: 可选 HostAdmin service；提供时会清理不可恢复 registry record。

    Returns:
        四元组 ``(session_id, scene_name, created, recreated_from_closed)``。

    Raises:
        ValueError: 当 label 非法或 registry record 非法时抛出。
    """

    explicit_session_id = _resolve_label_session_id(args)
    explicit_scene_name = _resolve_label_scene_name(args)
    if explicit_session_id is not None:
        return explicit_session_id, explicit_scene_name, False, False
    registry = FileConversationLabelRegistry(workspace_dir)
    target = resolve_labeled_conversation_target(
        registry=registry,
        prompt_asset_store=prompt_asset_store,
        label=label,
        default_scene_name=_INTERACTIVE_SCENE_NAME,
        explicit_session_id=explicit_session_id,
        explicit_scene_name=explicit_scene_name,
        host_admin_service=host_admin_service,
    )
    return target.session_id, target.scene_name, target.created, target.recreated_from_closed


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
