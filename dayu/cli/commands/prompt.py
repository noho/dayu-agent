"""`dayu-cli prompt` 命令实现。"""

from __future__ import annotations

import argparse
import json

from dayu.cli.dependency_setup import (
    _build_execution_options,
    _build_prompt_service,
    _prepare_cli_host_dependencies,
    setup_loglevel,
    setup_paths,
)
from dayu.cli.interactive_ui import prompt as prompt_command
from dayu.log import Log

MODULE = "APP.PROMPT"


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
