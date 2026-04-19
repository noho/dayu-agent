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

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from dayu.cli.arg_parsing import parse_arguments
from dayu.contracts.cancellation import CancelledError
from dayu.log import Log
from dayu.state_dir_lock import StateDirSingleInstanceLock
from dayu.workspace_paths import build_interactive_state_dir

if TYPE_CHECKING:
    from dayu.cli.dependency_setup import ModelName, RunningConfig, WorkspaceConfig, WriteCliConfig
    from dayu.execution.options import ExecutionOptions, ResolvedExecutionOptions
    from dayu.fins.service_runtime import DefaultFinsRuntime
    from dayu.host.host import Host
    from dayu.services.chat_service import ChatService
    from dayu.services.prompt_service import PromptService
    from dayu.services.contracts import WriteRunConfig as WriteRunConfigType
    from dayu.services.scene_execution_acceptance import SceneExecutionAcceptancePreparer
    from dayu.services.write_service import WriteService as WriteServiceType
    from dayu.startup.workspace import WorkspaceResources

MODULE = "APP.MAIN"
_FINS_COMMANDS = frozenset(
    {
        "download",
        "upload_filing",
        "upload_filings_from",
        "upload_material",
        "process",
        "process_filing",
        "process_material",
    }
)


class _LazyWriteService:
    """按需暴露写作服务的轻量代理。"""

    @staticmethod
    def print_report(output_dir: str | Path) -> int:
        """延迟导入并打印写作流水线报告。

        Args:
            output_dir: 写作输出目录。

        Returns:
            报告打印退出码。

        Raises:
            无。
        """

        from dayu.services.write_service import WriteService as _RuntimeWriteService

        return _RuntimeWriteService.print_report(output_dir)


WriteService = _LazyWriteService


class _ModelNameLike:
    """`setup_model_name` 返回值的最小只读视图。"""

    def __init__(self, *, model_name: str) -> None:
        """初始化模型名视图。

        Args:
            model_name: 解析后的模型名。

        Returns:
            无。

        Raises:
            无。
        """

        self.model_name = model_name


def setup_loglevel(args: argparse.Namespace) -> None:
    """延迟导入并设置 CLI 日志等级。

    Args:
        args: 解析后的命令行参数。

    Returns:
        无。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import setup_loglevel as _setup_loglevel

    _setup_loglevel(args)


def setup_paths(args: argparse.Namespace) -> WorkspaceConfig:
    """延迟导入并解析 CLI 工作区路径配置。

    Args:
        args: 解析后的命令行参数。

    Returns:
        解析后的工作区配置对象。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import setup_paths as _setup_paths

    return _setup_paths(args)


def setup_model_name(args: argparse.Namespace) -> ModelName:
    """延迟导入并解析 CLI 模型名。

    Args:
        args: 解析后的命令行参数。

    Returns:
        至少包含 ``model_name`` 字段的只读视图。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import setup_model_name as _setup_model_name

    return _setup_model_name(args)


def _build_execution_options(args: argparse.Namespace) -> ExecutionOptions:
    """延迟导入并构建 CLI 执行选项。

    Args:
        args: 解析后的命令行参数。

    Returns:
        运行时执行选项对象。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import _build_execution_options as _build_runtime_execution_options

    return _build_runtime_execution_options(args)


def _prepare_cli_host_dependencies(
    *,
    workspace_config: WorkspaceConfig,
    execution_options: ExecutionOptions,
) -> tuple[
    WorkspaceResources,
    ResolvedExecutionOptions,
    SceneExecutionAcceptancePreparer,
    Host,
    DefaultFinsRuntime,
]:
    """延迟导入并准备 CLI Host 级依赖。

    Args:
        workspace_config: 工作区配置。
        execution_options: 运行时执行选项。

    Returns:
        Host 依赖元组。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import _prepare_cli_host_dependencies as _prepare_dependencies

    return _prepare_dependencies(
        workspace_config=workspace_config,
        execution_options=execution_options,
    )


def _build_chat_service(
    *,
    host: Host,
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer,
    fins_runtime: DefaultFinsRuntime,
) -> ChatService:
    """延迟导入并构建交互式 ChatService。

    Args:
        host: Host 运行时。
        scene_execution_acceptance_preparer: scene 执行接受预处理器。
        fins_runtime: 财报运行时。

    Returns:
        ChatService 实例。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import _build_chat_service as _build_runtime_chat_service

    return _build_runtime_chat_service(
        host=host,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        fins_runtime=fins_runtime,
    )


def _build_prompt_service(
    *,
    host: Host,
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer,
    fins_runtime: DefaultFinsRuntime,
) -> PromptService:
    """延迟导入并构建单次 prompt Service。

    Args:
        host: Host 运行时。
        scene_execution_acceptance_preparer: scene 执行接受预处理器。
        fins_runtime: 财报运行时。

    Returns:
        PromptService 实例。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import _build_prompt_service as _build_runtime_prompt_service

    return _build_runtime_prompt_service(
        host=host,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        fins_runtime=fins_runtime,
    )


def _build_write_service(
    *,
    host: Host,
    workspace: WorkspaceResources,
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer,
    fins_runtime: DefaultFinsRuntime,
) -> "WriteServiceType":
    """延迟导入并构建写作 Service。

    Args:
        host: Host 运行时。
        workspace: 工作区资源。
        scene_execution_acceptance_preparer: scene 执行接受预处理器。
        fins_runtime: 财报运行时。

    Returns:
        WriteService 实例。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import _build_write_service as _build_runtime_write_service

    return _build_runtime_write_service(
        host=host,
        workspace=workspace,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        fins_runtime=fins_runtime,
    )


def _resolve_interactive_session_id(workspace_dir: Path, *, new_session: bool) -> str:
    """延迟导入并解析交互会话 ID。

    Args:
        workspace_dir: 工作区目录。
        new_session: 是否强制新建会话。

    Returns:
        交互会话 ID。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import _resolve_interactive_session_id as _resolve_runtime_session_id

    return _resolve_runtime_session_id(
        workspace_dir,
        new_session=new_session,
    )


def _resolve_write_output_dir(
    *,
    workspace_dir: Path,
    ticker: str,
    raw_output: str | None,
) -> Path:
    """延迟导入并解析写作输出目录。

    Args:
        workspace_dir: 工作区目录。
        ticker: 股票代码。
        raw_output: 原始输出目录参数。

    Returns:
        写作输出目录。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import _resolve_write_output_dir as _resolve_runtime_write_output_dir

    return _resolve_runtime_write_output_dir(
        workspace_dir=workspace_dir,
        ticker=ticker,
        raw_output=raw_output,
    )


def setup_write_config(
    args: argparse.Namespace,
    paths_config: WorkspaceConfig,
    running_config: RunningConfig,
) -> WriteCliConfig:
    """延迟导入并解析写作 CLI 配置。

    Args:
        args: 解析后的命令行参数。
        paths_config: 工作区配置。
        running_config: 运行时配置。

    Returns:
        写作 CLI 配置对象。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import setup_write_config as _setup_runtime_write_config

    return _setup_runtime_write_config(args, paths_config, running_config)


def run_write_pipeline(*, write_config: "WriteRunConfigType", write_service: "WriteServiceType") -> int:
    """延迟导入并执行写作流水线。

    Args:
        write_config: 写作运行配置。
        write_service: 写作服务实例。

    Returns:
        写作流水线退出码。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import run_write_pipeline as _run_runtime_write_pipeline

    return _run_runtime_write_pipeline(
        write_config=write_config,
        write_service=write_service,
    )


def _write_cancelled_exit_code() -> int:
    """延迟导入写作模式取消退出码。

    Args:
        无。

    Returns:
        写作模式取消退出码。

    Raises:
        无。
    """

    from dayu.services.write_service import WRITE_CANCELLED_EXIT_CODE as _runtime_exit_code

    return _runtime_exit_code


def _build_write_run_config(
    *,
    ticker: str,
    company: str,
    template_path: Path,
    output_dir: Path,
    write_max_retries: int,
    web_provider: str,
    resume: bool,
    write_model_override_name: str,
    audit_model_override_name: str,
    chapter_filter: str,
    fast: bool,
    force: bool,
    infer: bool,
) -> "WriteRunConfigType":
    """延迟导入并构建写作运行配置。

    Args:
        ticker: 股票代码。
        company: 公司名。
        template_path: 模板路径。
        output_dir: 输出目录。
        write_max_retries: 章节重试次数。
        web_provider: 联网 provider。
        resume: 是否断点恢复。
        write_model_override_name: 写作模型覆盖名。
        audit_model_override_name: 审计模型覆盖名。
        chapter_filter: 章节过滤表达式。
        fast: 是否快速模式。
        force: 是否强制放宽门禁。
        infer: 是否仅执行公司级 facet 归因。

    Returns:
        写作运行配置对象。

    Raises:
        无。
    """

    from dayu.services.contracts import WriteRunConfig as _RuntimeWriteRunConfig

    return _RuntimeWriteRunConfig(
        ticker=ticker,
        company=company,
        template_path=str(template_path),
        output_dir=str(output_dir),
        write_max_retries=write_max_retries,
        web_provider=web_provider,
        resume=resume,
        write_model_override_name=write_model_override_name,
        audit_model_override_name=audit_model_override_name,
        chapter_filter=chapter_filter,
        fast=fast,
        force=force,
        infer=infer,
    )


def _run_fins_command(args: argparse.Namespace) -> int:
    """延迟导入并执行财报子命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        财报子命令退出码。

    Raises:
        无。
    """

    from dayu.cli.fins_commands import _run_fins_command as _run_runtime_fins_command

    return _run_runtime_fins_command(args)


def run_host_command(args: argparse.Namespace) -> int:
    """延迟导入并执行宿主管理命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        宿主管理命令退出码。

    Raises:
        无。
    """

    from dayu.cli.host_commands import run_host_command as _run_runtime_host_command

    return _run_runtime_host_command(args)


def run_init(args: argparse.Namespace) -> int:
    """延迟导入并执行工作区初始化命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        初始化命令退出码。

    Raises:
        无。
    """

    from dayu.cli.init_command import run_init as _run_runtime_init

    return _run_runtime_init(args)


def interactive(
    service: ChatService,
    *,
    session_id: str,
    execution_options: ExecutionOptions,
    show_thinking: bool,
) -> None:
    """延迟导入并进入交互终端。

    Args:
        service: ChatService。
        session_id: 会话 ID。
        execution_options: 执行选项。
        show_thinking: 是否显示 thinking。

    Returns:
        无。

    Raises:
        无。
    """

    from dayu.cli.interactive_ui import interactive as _interactive_ui

    _interactive_ui(
        service,
        session_id=session_id,
        execution_options=execution_options,
        show_thinking=show_thinking,
    )


def prompt_command(
    service: PromptService,
    prompt_text: str,
    *,
    ticker: str | None,
    execution_options: ExecutionOptions,
    show_thinking: bool,
) -> int:
    """延迟导入并执行单次 prompt。

    Args:
        service: PromptService。
        prompt_text: Prompt 文本。
        ticker: 股票代码。
        execution_options: 执行选项。
        show_thinking: 是否显示 thinking。

    Returns:
        prompt 命令退出码。

    Raises:
        无。
    """

    from dayu.cli.interactive_ui import prompt as _prompt_command

    return _prompt_command(
        service,
        prompt_text,
        ticker=ticker,
        execution_options=execution_options,
        show_thinking=show_thinking,
    )


def _running_config_from_resolved(resolved_options: ResolvedExecutionOptions) -> RunningConfig:
    """延迟导入并从已解析运行时选项构造 CLI 运行配置。

    Args:
        resolved_options: 已解析的运行时执行选项。

    Returns:
        CLI 运行配置对象。

    Raises:
        无。
    """

    from dayu.cli.dependency_setup import RunningConfig

    return RunningConfig.from_resolved(resolved_options)


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
    model_config = setup_model_name(args) if hasattr(args, "model_name") else _ModelNameLike(model_name="")
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
        running_config = _running_config_from_resolved(default_execution_options)
        write_cli_config = setup_write_config(args, paths_config, running_config)
        service = _build_write_service(
            host=host,
            workspace=workspace,
            scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
            fins_runtime=fins_runtime,
        )
        write_config = _build_write_run_config(
            ticker=paths_config.ticker,
            company=paths_config.ticker,
            template_path=write_cli_config.template_path,
            output_dir=write_cli_config.output_dir,
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
            elif exit_code == _write_cancelled_exit_code():
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
            return _write_cancelled_exit_code()
        except Exception as exc:
            elapsed = perf_counter() - start_time
            Log.error(
                ("公司级 Facet 归因执行失败" if write_cli_config.infer else "写作模式执行失败")
                + f": elapsed={elapsed:.2f}s, error={exc}",
                module=MODULE,
            )
            return 2

    return 0
