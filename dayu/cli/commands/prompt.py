"""`dayu-cli prompt` 命令实现。"""

from __future__ import annotations

import argparse
import json

from dayu.cli.dependency_setup import (
    WorkspaceConfig,
    _build_chat_service,
    _build_execution_options,
    _build_prompt_service,
    _prepare_cli_host_dependencies,
    setup_loglevel,
    setup_paths,
)
from dayu.cli.conversation_labels import FileConversationLabelRegistry
from dayu.cli.interactive_ui import conversation_prompt as conversation_prompt_command
from dayu.cli.interactive_ui import prompt as prompt_command
from dayu.execution.options import ExecutionOptions
from dayu.log import Log

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
        try:
            label_session_id, scene_name = _resolve_labeled_prompt_target(
                args,
                workspace_config=paths_config,
                label=label,
            )
        except ValueError as exc:
            Log.error(str(exc), module=MODULE)
            return 2
        return _run_labeled_prompt_command(
            args,
            label=label,
            label_session_id=label_session_id,
            scene_name=scene_name,
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
        f"{json.dumps(prompt_model, ensure_ascii=False, sort_keys=True)}",
        module=MODULE,
    )
    Log.info("执行单次 prompt...", module=MODULE)
    return prompt_command(
        service,
        args.prompt,
        ticker=paths_config.ticker,
        execution_options=execution_options,
        show_thinking=bool(getattr(args, "thinking", False)),
    )


def _run_labeled_prompt_command(
    args: argparse.Namespace,
    *,
    label: str,
    label_session_id: str,
    scene_name: str,
    paths_config: WorkspaceConfig,
    execution_options: ExecutionOptions,
) -> int:
    """执行带 label 的 conversation prompt 命令。

    Args:
        args: 解析后的命令行参数。
        label: 用户显式提供的 label。
        label_session_id: registry 解析得到的会话 ID。
        scene_name: 当前 label 对应的 scene。
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
    service = _build_chat_service(
        host=host,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        fins_runtime=fins_runtime,
    )
    prompt_model = scene_execution_acceptance_preparer.resolve_scene_model(
        scene_name,
        execution_options,
    )
    Log.info(
        "使用模型: "
        f"{json.dumps(prompt_model, ensure_ascii=False, sort_keys=True)}",
        module=MODULE,
    )
    Log.info(
        f"执行带标签 prompt: label={label}, session_id={label_session_id}, scene={scene_name}",
        module=MODULE,
    )
    return conversation_prompt_command(
        service,
        args.prompt,
        session_id=label_session_id,
        scene_name=scene_name,
        ticker=paths_config.ticker,
        execution_options=execution_options,
        show_thinking=bool(getattr(args, "thinking", False)),
    )


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
) -> tuple[str, str]:
    """解析 labeled prompt 对应的 session 与 scene。

    Args:
        args: 解析后的命令行参数。
        workspace_config: 已解析的工作区配置。
        label: 已规范化的 conversation label。

    Returns:
        二元组 `(session_id, scene_name)`。

    Raises:
        ValueError: 当 label 非法或 registry record 非法时抛出。
    """

    explicit_session_id = _resolve_label_session_id(args)
    explicit_scene_name = _resolve_labeled_prompt_scene_name(args)
    if explicit_session_id is not None:
        return explicit_session_id, explicit_scene_name
    registry = FileConversationLabelRegistry(workspace_config.workspace_dir)
    record = registry.get_or_create_record(
        label=label,
        scene_name=_DEFAULT_LABELED_PROMPT_SCENE_NAME,
    )
    return record.session_id, record.scene_name


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
