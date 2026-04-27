"""`dayu-cli prompt` 命令实现。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from dayu.cli.dependency_setup import (
    WorkspaceConfig,
    _build_chat_service,
    _build_execution_options,
    _build_prompt_service,
    _prepare_cli_host_dependencies,
    get_cli_shutdown_coordinator,
    setup_loglevel,
    setup_paths,
)
from dayu.cli.conversation_label_locks import ConversationLabelLease
from dayu.cli.conversation_labels import FileConversationLabelRegistry
from dayu.cli.interactive_ui import conversation_prompt as conversation_prompt_command
from dayu.cli.interactive_ui import prompt as prompt_command
from dayu.cli.labeled_conversations import resolve_labeled_conversation_target
from dayu.execution.options import ExecutionOptions
from dayu.log import Log
from dayu.services.host_admin_service import HostAdminService
from dayu.services.protocols import HostAdminServiceProtocol

MODULE = "APP.PROMPT"
_DEFAULT_LABELED_PROMPT_SCENE_NAME = "prompt_mt"


def run_prompt_command(args: argparse.Namespace) -> int:
    """执行单次 prompt 命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        prompt 命令退出码。

    Raises:
        无。
    """

    setup_loglevel(args)
    paths_config = setup_paths(args)
    execution_options = _build_execution_options(args)
    label = _resolve_prompt_label(args)
    if label is not None:
        return _run_labeled_prompt_command(
            args,
            label=label,
            paths_config=paths_config,
            execution_options=execution_options,
        )
    return _run_one_shot_prompt_command(
        args,
        paths_config=paths_config,
        execution_options=execution_options,
    )


def _run_one_shot_prompt_command(
    args: argparse.Namespace,
    *,
    paths_config: WorkspaceConfig,
    execution_options: ExecutionOptions,
) -> int:
    """执行普通 one-shot prompt 命令。

    Args:
        args: 解析后的命令行参数。
        paths_config: 已解析的路径配置。
        execution_options: 请求级执行覆盖参数。

    Returns:
        prompt 命令退出码。

    Raises:
        无。
    """

    Log.info(f"工作目录: {paths_config.workspace_dir}", module=MODULE)
    if paths_config.ticker:
        Log.info(f"公司股票代码: {paths_config.ticker}", module=MODULE)
        if paths_config.has_local_filings:
            Log.info("财报目录: 已检测到本地财报", module=MODULE)
        else:
            Log.info("财报目录: 无本地财报", module=MODULE)
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
    service = _build_prompt_service(
        host=host,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        fins_runtime=fins_runtime,
    )
    prompt_model = scene_execution_acceptance_preparer.resolve_scene_model(
        "prompt",
        execution_options,
    )
    Log.info(
        "使用模型: "
        f"{json.dumps(asdict(prompt_model), ensure_ascii=False, sort_keys=True)}",
        module=MODULE,
    )
    Log.info("执行单次 prompt...", module=MODULE)
    return prompt_command(
        service,
        args.prompt,
        ticker=paths_config.ticker,
        execution_options=execution_options,
        show_thinking=bool(getattr(args, "thinking", False)),
        run_lifecycle_observer=get_cli_shutdown_coordinator(),
    )


def _run_labeled_prompt_command(
    args: argparse.Namespace,
    *,
    label: str,
    paths_config: WorkspaceConfig,
    execution_options: ExecutionOptions,
) -> int:
    """执行带 label 的 conversation prompt 命令。

    Args:
        args: 解析后的命令行参数。
        label: 用户显式提供的 label。
        paths_config: 已解析的路径配置。
        execution_options: 请求级执行覆盖参数。

    Returns:
        prompt 命令退出码。

    Raises:
        无。
    """

    Log.info(f"工作目录: {paths_config.workspace_dir}", module=MODULE)
    if paths_config.ticker:
        Log.info(f"公司股票代码: {paths_config.ticker}", module=MODULE)
        if paths_config.has_local_filings:
            Log.info("财报目录: 已检测到本地财报", module=MODULE)
        else:
            Log.info("财报目录: 无本地财报", module=MODULE)
    try:
        with ConversationLabelLease(paths_config.workspace_dir, label):
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
            label_session_id, scene_name, label_created, recreated_from_closed = _resolve_labeled_prompt_target(
                args,
                workspace_config=paths_config,
                label=label,
                host_admin_service=host_admin_service,
            )
            prompt_model = scene_execution_acceptance_preparer.resolve_scene_model(
                scene_name,
                execution_options,
            )
            Log.info(
                "使用模型: "
                f"{json.dumps(asdict(prompt_model), ensure_ascii=False, sort_keys=True)}",
                module=MODULE,
            )
            if recreated_from_closed:
                Log.info(
                    f"label 对应的旧对话已关闭，现将基于同名 label 创建新对话: {label}",
                    module=MODULE,
                )
            Log.info(
                (
                    f"执行带标签 prompt，新创建标签: {label}"
                    if label_created
                    else f"执行带标签 prompt，恢复标签: {label}"
                ),
                module=MODULE,
            )
            return conversation_prompt_command(
                service,
                args.prompt,
                label=label,
                session_id=label_session_id,
                scene_name=scene_name,
                ticker=paths_config.ticker,
                execution_options=execution_options,
                show_thinking=bool(getattr(args, "thinking", False)),
                run_lifecycle_observer=get_cli_shutdown_coordinator(),
            )
    except ValueError as exc:
        Log.error(str(exc), module=MODULE)
        return 2
    except RuntimeError as exc:
        Log.error(str(exc), module=MODULE)
        return 2


def _resolve_prompt_label(args: argparse.Namespace) -> str | None:
    """解析 prompt 命令的 label 参数。

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


def _resolve_label_session_id(args: argparse.Namespace) -> str | None:
    """解析 labeled prompt 对应的会话 ID。

    Args:
        args: 解析后的命令行参数。

    Returns:
        规范化后的 session_id；未提供时返回 ``None``。

    Raises:
        无。
    """

    normalized_session_id = str(getattr(args, "label_session_id", "") or "").strip()
    if not normalized_session_id:
        return None
    return normalized_session_id


def _resolve_labeled_prompt_target(
    args: argparse.Namespace,
    *,
    workspace_config: WorkspaceConfig,
    label: str,
    host_admin_service: HostAdminServiceProtocol | None = None,
) -> tuple[str, str, bool, bool]:
    """解析 labeled prompt 对应的 session 与 scene。

    Args:
        args: 解析后的命令行参数。
        workspace_config: 已解析的工作区配置。
        label: 已规范化的 conversation label。
        host_admin_service: 可选 HostAdmin service；提供时会清理漂移 registry record。

    Returns:
        四元组 `(session_id, scene_name, created, recreated_from_closed)`。

    Raises:
        ValueError: 当 label 非法或 registry record 非法时抛出。
    """

    explicit_session_id = _resolve_label_session_id(args)
    explicit_scene_name = _resolve_labeled_prompt_scene_name(args)
    if explicit_session_id is not None:
        return explicit_session_id, explicit_scene_name, False, False
    registry = FileConversationLabelRegistry(workspace_config.workspace_dir)
    target = resolve_labeled_conversation_target(
        registry=registry,
        prompt_asset_store=workspace_config.prompt_asset_store,
        label=label,
        default_scene_name=_DEFAULT_LABELED_PROMPT_SCENE_NAME,
        explicit_session_id=explicit_session_id,
        explicit_scene_name=explicit_scene_name,
        host_admin_service=host_admin_service,
    )
    return target.session_id, target.scene_name, target.created, target.recreated_from_closed


def _resolve_labeled_prompt_scene_name(args: argparse.Namespace) -> str:
    """解析 labeled prompt 对应的 scene 名称。

    Args:
        args: 解析后的命令行参数。

    Returns:
        scene 名称；未显式提供时返回 ``prompt_mt``。

    Raises:
        无。
    """

    normalized_scene_name = str(getattr(args, "label_scene_name", "") or "").strip()
    if normalized_scene_name:
        return normalized_scene_name
    return _DEFAULT_LABELED_PROMPT_SCENE_NAME
