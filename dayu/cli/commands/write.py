"""`dayu-cli write` 命令实现。"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Callable

from dayu.cli.dependency_setup import (
    RunningConfig,
    WriteCliConfig,
    _build_execution_options,
    _build_write_service,
    _prepare_cli_host_dependencies,
    _resolve_write_output_dir,
    run_write_pipeline,
    setup_loglevel,
    setup_model_name,
    setup_paths,
    setup_write_config,
)
from dayu.contracts.cancellation import CancelledError
from dayu.log import Log
from dayu.services.contracts import WriteRunConfig
from dayu.services.write_service import WRITE_CANCELLED_EXIT_CODE, WriteService

MODULE = "APP.WRITE"


def _resolve_write_model_override_name(args: argparse.Namespace) -> str:
    """解析主写作模型覆盖名。

    Args:
        args: 解析后的命令行参数。

    Returns:
        归一化后的主写作模型覆盖名；未显式配置时返回空字符串。

    Raises:
        无。
    """

    return setup_model_name(args).model_name


def _resolve_write_company_name(
    *,
    ticker: str,
    company_name_resolver: Callable[[str], str],
) -> str:
    """解析写作配置中的公司名称。

    Args:
        ticker: 公司股票代码。
        company_name_resolver: 公司名称解析函数。

    Returns:
        解析后的公司名称；缺失或解析失败时返回空字符串。

    Raises:
        无。
    """

    try:
        return str(company_name_resolver(ticker) or "").strip()
    except Exception:
        return ""


def _build_write_run_config(
    *,
    ticker: str,
    company_name: str,
    write_cli_config: WriteCliConfig,
    write_model_override_name: str,
) -> WriteRunConfig:
    """Build the service contract for one write-pipeline stage."""

    return WriteRunConfig(
        ticker=ticker,
        company=company_name,
        template_path=str(write_cli_config.template_path),
        output_dir=str(write_cli_config.output_dir),
        write_max_retries=write_cli_config.write_max_retries,
        web_provider=write_cli_config.web_provider,
        resume=write_cli_config.resume,
        write_model_override_name=write_model_override_name,
        audit_model_override_name=write_cli_config.audit_model_override_name,
        chapter_filter=write_cli_config.chapter_filter,
        fast=write_cli_config.fast,
        force=write_cli_config.force,
        infer=write_cli_config.infer,
        research_template_requested_name=write_cli_config.research_template_requested_name,
        research_template_resolved_name=write_cli_config.research_template_resolved_name,
        research_template_selection_mode=write_cli_config.research_template_selection_mode,
    )


def _run_write_stage(*, write_config: WriteRunConfig, write_service: WriteService) -> int:
    """Run one infer or writing stage with stable CLI logging and exit codes."""

    infer_mode = write_config.infer
    Log.info("公司级 Facet 归因启动..." if infer_mode else "写作流水线启动...", module=MODULE)
    start_time = perf_counter()
    try:
        exit_code = run_write_pipeline(write_config=write_config, write_service=write_service)
        elapsed = perf_counter() - start_time
        if exit_code == 0:
            Log.info(
                ("公司级 Facet 归因完成" if infer_mode else "写作流水线完成")
                + f": exit_code={exit_code}, elapsed={elapsed:.2f}s",
                module=MODULE,
            )
        elif exit_code == WRITE_CANCELLED_EXIT_CODE:
            Log.warn(
                ("公司级 Facet 归因已取消" if infer_mode else "写作模式已取消")
                + f": exit_code={exit_code}, elapsed={elapsed:.2f}s",
                module=MODULE,
            )
        else:
            Log.warn(
                ("公司级 Facet 归因结束但返回非零" if infer_mode else "写作流水线结束但返回非零")
                + f": exit_code={exit_code}, elapsed={elapsed:.2f}s",
                module=MODULE,
            )
        return exit_code
    except CancelledError as exc:
        elapsed = perf_counter() - start_time
        Log.warn(
            ("公司级 Facet 归因已取消" if infer_mode else "写作模式已取消") + f": elapsed={elapsed:.2f}s, reason={exc}",
            module=MODULE,
        )
        return WRITE_CANCELLED_EXIT_CODE
    except Exception as exc:
        elapsed = perf_counter() - start_time
        Log.error(
            ("公司级 Facet 归因执行失败" if infer_mode else "写作模式执行失败")
            + f": elapsed={elapsed:.2f}s, error={exc}",
            module=MODULE,
        )
        return 2


def _needs_auto_research_bootstrap(args: argparse.Namespace, *, output_dir: Path) -> bool:
    requested = str(getattr(args, "research_template", "") or "").strip().lower()
    return requested == "auto" and not (output_dir / "manifest.json").is_file()


def _build_auto_bootstrap_args(args: argparse.Namespace) -> argparse.Namespace:
    values = dict(vars(args))
    values.update({"template": None, "research_template": None, "infer": True})
    return argparse.Namespace(**values)


def _validate_research_materialization_args(args: argparse.Namespace) -> str | None:
    materialize = bool(getattr(args, "materialize_research", False))
    has_related_option = bool(getattr(args, "research_base", None)) or bool(getattr(args, "overwrite_research", False))
    if has_related_option and not materialize:
        return "--research-base/--overwrite-research 需要同时提供 --materialize-research"
    if not materialize:
        return None
    if not str(getattr(args, "research_template", "") or "").strip():
        return "--materialize-research 需要同时提供 --research-template"
    if bool(getattr(args, "infer", False)):
        return "--materialize-research 不能与 infer-only 的 --infer 同时使用"
    if bool(getattr(args, "summary", False)):
        return "--materialize-research 不能与 --summary 同时使用"
    return None


def _materialize_research_after_write(
    args: argparse.Namespace,
    *,
    workspace_dir: Path,
    ticker: str,
    write_output_dir: Path,
) -> dict[str, object]:
    from dayu.cli.commands.research_template import materialize_research_bundle_from_write_manifest

    research_base_raw = getattr(args, "research_base", None)
    research_base = (
        Path(str(research_base_raw)).expanduser().resolve() if research_base_raw else (workspace_dir / ticker).resolve()
    )
    return materialize_research_bundle_from_write_manifest(
        write_output_dir / "manifest.json",
        workspace_root=research_base,
        overwrite=bool(getattr(args, "overwrite_research", False)),
    )


def run_write_command(args: argparse.Namespace) -> int:
    """执行写作 CLI 命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        写作命令退出码。

    Raises:
        无。
    """

    setup_loglevel(args)
    paths_config = setup_paths(args)
    write_model_override_name = _resolve_write_model_override_name(args)
    execution_options = _build_execution_options(args)
    Log.info(f"工作目录: {paths_config.workspace_dir}", module=MODULE)
    if paths_config.ticker:
        Log.info(f"公司股票代码: {paths_config.ticker}", module=MODULE)
        if paths_config.has_local_filings:
            Log.info("财报目录: 已检测到本地财报", module=MODULE)
        else:
            Log.info("财报目录: 无本地财报", module=MODULE)
    if not paths_config.ticker:
        error_message = (
            "write --summary 模式要求必须提供 --ticker"
            if getattr(args, "summary", False)
            else "write 模式要求必须提供 --ticker"
        )
        Log.error(error_message, module=MODULE)
        return 2
    research_materialization_error = _validate_research_materialization_args(args)
    if research_materialization_error is not None:
        Log.error(research_materialization_error, module=MODULE)
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
        interactive=False,
    )
    running_config = RunningConfig.from_resolved(default_execution_options)
    service = _build_write_service(
        host=host,
        workspace=workspace,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        fins_runtime=fins_runtime,
    )
    company_name = _resolve_write_company_name(
        ticker=paths_config.ticker,
        company_name_resolver=fins_runtime.get_company_name,
    )
    output_dir = _resolve_write_output_dir(
        workspace_dir=paths_config.workspace_dir,
        ticker=paths_config.ticker,
        raw_output=getattr(args, "output", None),
    )
    if _needs_auto_research_bootstrap(args, output_dir=output_dir):
        Log.info("auto 研究模板缺少本地 manifest，先执行公司级 Facet 归因", module=MODULE)
        bootstrap_cli_config = setup_write_config(
            _build_auto_bootstrap_args(args),
            paths_config,
            running_config,
        )
        bootstrap_config = _build_write_run_config(
            ticker=paths_config.ticker,
            company_name=company_name,
            write_cli_config=bootstrap_cli_config,
            write_model_override_name=write_model_override_name,
        )
        bootstrap_exit_code = _run_write_stage(write_config=bootstrap_config, write_service=service)
        if bootstrap_exit_code != 0 or bool(getattr(args, "infer", False)):
            return bootstrap_exit_code

    write_cli_config = setup_write_config(args, paths_config, running_config)
    write_config = _build_write_run_config(
        ticker=paths_config.ticker,
        company_name=company_name,
        write_cli_config=write_cli_config,
        write_model_override_name=write_model_override_name,
    )
    write_exit_code = _run_write_stage(
        write_config=write_config,
        write_service=service,
    )
    if write_exit_code != 0 or not bool(getattr(args, "materialize_research", False)):
        return write_exit_code
    try:
        materialized = _materialize_research_after_write(
            args,
            workspace_dir=paths_config.workspace_dir,
            ticker=paths_config.ticker,
            write_output_dir=Path(write_config.output_dir),
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        Log.error(f"研究工件 materialize 失败: {exc}", module=MODULE)
        return 2
    Log.info(
        f"研究工件 materialize 完成: bundle={materialized['bundle_file']}, workbook={materialized['workbook_file']}",
        module=MODULE,
    )
    return 0
