"""`dayu-cli interactive` 命令实现。"""

from __future__ import annotations

import argparse
import json

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
from dayu.state_dir_lock import StateDirSingleInstanceLock
from dayu.workspace_paths import build_interactive_state_dir

MODULE = "APP.INTERACTIVE"


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
        interactive_session_id = _resolve_interactive_session_id(
            paths_config.workspace_dir,
            new_session=bool(getattr(args, "new_session", False)),
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
