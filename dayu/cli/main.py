#!/usr/bin/env python3
"""报告生成主程序 CLI 入口。

模块职责：
- 解析命令行参数，分发至对应子命令执行路径。
- 参数定义见 ``arg_parsing``，依赖装配见 ``dependency_setup``，财报命令见 ``fins_commands``。

子命令：
- ``interactive``：进入多轮交互终端对话。
- ``prompt``：执行单次 prompt 并输出结果。
- ``write``：按模板逐章写作；显式传 ``--infer`` 时仅刷新公司级 facet 归因并写回 manifest；
  显式传 ``--summary`` 时仅读取已有输出目录并打印上次写作流水线运行报告。
- ``download/upload_filing/upload_filings_from/upload_material/process/process_filing/process_material``：
  财报数据管线命令。
"""
from __future__ import annotations

import json
from time import perf_counter

from dayu.cli.arg_parsing import parse_arguments
from dayu.cli.dependency_setup import (
    MODULE,
    ModelName,
    RunningConfig,
    _build_chat_service,
    _build_execution_options,
    _build_prompt_service,
    _build_write_service,
    _prepare_cli_host_dependencies,
    _resolve_interactive_session_id,
    _resolve_write_output_dir,
    run_write_pipeline,
    setup_loglevel,
    setup_model_name,
    setup_paths,
    setup_write_config,
)
from dayu.cli.fins_commands import _FINS_COMMANDS, _run_fins_command
from dayu.cli.host_commands import run_host_command
from dayu.cli.init_command import run_init
from dayu.cli.interactive_ui import interactive, prompt as prompt_command
from dayu.contracts.cancellation import CancelledError
from dayu.log import Log
from dayu.services import WriteRunConfig
from dayu.services.write_service import WRITE_CANCELLED_EXIT_CODE, WriteService
from dayu.state_dir_lock import StateDirSingleInstanceLock
from dayu.workspace_paths import build_interactive_state_dir


def main() -> int:
    """主函数：解析命令行参数并分发至对应执行路径。

    Args:
        无。

    Returns:
        退出码，0 表示成功。

    Raises:
        无。
    """

    args = parse_arguments()

    # init 子命令不需要 setup_loglevel 等重量级初始化
    if args.command == "init":
        return run_init(args)

    setup_loglevel(args)
    if args.command in _FINS_COMMANDS:
        return _run_fins_command(args)

    # 宿主管理命令（sessions/runs/cancel/host）走独立分支
    _HOST_COMMANDS = {"sessions", "runs", "cancel", "host"}
    if args.command in _HOST_COMMANDS:
        return run_host_command(args)

    paths_config = setup_paths(args)
    model_config = setup_model_name(args) if hasattr(args, "model_name") else ModelName(model_name="")
    execution_options = _build_execution_options(args)

    Log.info(f"工作目录: {paths_config.workspace_dir}", module=MODULE)
    if paths_config.ticker and args.command != "interactive":
        Log.info(f"公司股票代码: {paths_config.ticker}", module=MODULE)
        if paths_config.has_local_filings:
            Log.info("财报目录: 已检测到本地财报", module=MODULE)
        else:
            Log.info("财报目录: 无本地财报", module=MODULE)

    if args.command == "interactive":
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

    if args.command == "prompt":
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

    if args.command == "write":
        if not paths_config.ticker:
            error_message = (
                "write --summary 模式要求必须提供 --ticker"
                if getattr(args, "summary", False)
                else "write 模式要求必须提供 --ticker"
            )
            Log.error(error_message, module=MODULE)
            return 2
        if getattr(args, "summary", False):
            output_dir = _resolve_write_output_dir(
                workspace_dir=paths_config.workspace_dir,
                ticker=paths_config.ticker,
                raw_output=getattr(args, "output", None),
            )
            return WriteService.print_report(output_dir)
        (
            workspace,
            default_execution_options,
            scene_execution_acceptance_preparer,
            host,
            fins_runtime,
        ) = _prepare_cli_host_dependencies(
            workspace_config=paths_config,
            execution_options=execution_options,
        )
        running_config = RunningConfig.from_resolved(default_execution_options)
        write_cli_config = setup_write_config(args, paths_config, running_config)
        service = _build_write_service(
            host=host,
            workspace=workspace,
            scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
            fins_runtime=fins_runtime,
        )
        write_config = WriteRunConfig(
            ticker=paths_config.ticker,
            company=paths_config.ticker,
            template_path=str(write_cli_config.template_path),
            output_dir=str(write_cli_config.output_dir),
            write_max_retries=write_cli_config.write_max_retries,
            web_provider=write_cli_config.web_provider,
            resume=write_cli_config.resume,
            write_model_override_name=model_config.model_name,
            audit_model_override_name=write_cli_config.audit_model_override_name,
            chapter_filter=write_cli_config.chapter_filter,
            fast=write_cli_config.fast,
            force=write_cli_config.force,
            infer=write_cli_config.infer,
        )
        Log.info("公司级 Facet 归因启动..." if write_cli_config.infer else "写作流水线启动...", module=MODULE)
        start_time = perf_counter()
        try:
            exit_code = run_write_pipeline(
                write_config=write_config,
                write_service=service,
            )
            elapsed = perf_counter() - start_time
            if exit_code == 0:
                Log.info(
                    ("公司级 Facet 归因完成" if write_cli_config.infer else "写作流水线完成")
                    + f": exit_code={exit_code}, elapsed={elapsed:.2f}s",
                    module=MODULE,
                )
            elif exit_code == WRITE_CANCELLED_EXIT_CODE:
                Log.warn(
                    ("公司级 Facet 归因已取消" if write_cli_config.infer else "写作模式已取消")
                    + f": exit_code={exit_code}, elapsed={elapsed:.2f}s",
                    module=MODULE,
                )
            else:
                Log.warn(
                    ("公司级 Facet 归因结束但返回非零" if write_cli_config.infer else "写作流水线结束但返回非零")
                    + f": exit_code={exit_code}, elapsed={elapsed:.2f}s",
                    module=MODULE,
                )
            return exit_code
        except CancelledError as exc:
            elapsed = perf_counter() - start_time
            Log.warn(
                ("公司级 Facet 归因已取消" if write_cli_config.infer else "写作模式已取消")
                + f": elapsed={elapsed:.2f}s, reason={exc}",
                module=MODULE,
            )
            return WRITE_CANCELLED_EXIT_CODE
        except Exception as exc:
            elapsed = perf_counter() - start_time
            Log.error(
                ("公司级 Facet 归因执行失败" if write_cli_config.infer else "写作模式执行失败")
                + f": elapsed={elapsed:.2f}s, error={exc}",
                module=MODULE,
            )
            return 2

    return 0
