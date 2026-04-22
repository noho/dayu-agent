"""写作流水线执行模块。

该模块实现 CLI `--write` 的完整写作闭环：
- 中间章节写作 -> 条件占位符补强 -> 审计 -> 失败重写与重审。
- 第10章“是否值得继续深研与待验证问题”决策综合。
- 第0章“投资要点概览”回填。
- 模板包含时才生成末章“来源清单”。
- 中间产物落盘与断点恢复。
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows 不提供 fcntl
    fcntl = None

from dayu.contracts.cancellation import CancelledError
from dayu.host import HostExecutorProtocol
from dayu.host.protocols import HostGovernanceProtocol
from dayu.execution.options import ExecutionOptions, ResolvedExecutionOptions
from dayu.log import Log
from dayu.services.concurrency_lanes import LANE_WRITE_CHAPTER
from dayu.services.internal.write_pipeline.chapter_audit_coordinator import (
    ChapterAuditCoordinator,
)
from dayu.services.internal.write_pipeline.chapter_execution_coordinator import (
    ChapterExecutionCoordinator,
    ChapterExecutionState,
    build_process_state_template,
)
from dayu.services.internal.write_pipeline.models import (
    AuditDecision,
    ChapterResult,
    ChapterTask,
    CompanyFacetProfile,
    EvidenceConfirmationResult,
    RunManifest,
    serialize_scene_models,
    SourceEntry,
    WriteRunConfig,
)
from dayu.services.internal.write_pipeline.source_list_builder import (
    build_source_entries,
    extract_evidence_items,
    render_source_list_chapter,
)
from dayu.services.internal.write_pipeline.template_parser import (
    TemplateLayout,
    parse_template_layout,
)
from dayu.services.scene_execution_acceptance import (
    SceneExecutionAcceptancePreparer,
)
from dayu.contracts.infrastructure import WorkspaceResourcesProtocol
from dayu.services.internal.write_pipeline.audit_formatting import (
    _collect_all_evidence_items,
)
from dayu.services.internal.write_pipeline.artifact_store import (
    ArtifactStore,
    _CHAPTERS_DIR_NAME,
    _manifest_file_lock,
    _MANIFEST_FILE_NAME,
    _OVERVIEW_CHAPTER_TITLE,
    _read_manifest_from_dir,
    _SOURCE_CHAPTER_TITLE,
)
from dayu.services.internal.write_pipeline.execution_summary_builder import ExecutionSummaryBuilder
from dayu.services.internal.write_pipeline.scene_contract_preparer import (
    SceneAgentCreationError,
    SceneContractPreparer,
)
from dayu.services.internal.write_pipeline.scene_executor import (
    ScenePromptRunner,
)
from dayu.services.internal.write_pipeline.prompt_builder import (
    PromptBuilder,
    _build_prior_decision_tasks,
    _DECISION_CHAPTER_TITLE,
)
from dayu.services.internal.write_pipeline.report_assembler import ReportAssembler

MODULE = "APP.WRITE_PIPELINE"

_INFER_COMPANY_META_EXCLUDED_FIELDS = frozenset({"company_id"})


def _sanitize_company_meta_for_infer(company_meta: dict[str, str]) -> dict[str, str]:
    """清洗送给 infer task prompt 的公司基础信息。

    Args:
        company_meta: 原始公司基础信息。

    Returns:
        移除不应暴露给 infer LLM 的字段后的公司基础信息。

    Raises:
        无。
    """

    return {
        key: value
        for key, value in company_meta.items()
        if key not in _INFER_COMPANY_META_EXCLUDED_FIELDS
    }


def _log_write_pipeline_config(
    *,
    write_config: WriteRunConfig,
    resolved_options: ResolvedExecutionOptions,
) -> None:
    """输出写作流水线实际生效参数摘要。

    Args:
        write_config: 写作流水线配置。
        resolved_options: 已解析的运行时配置。

    Returns:
        无。

    Raises:
        无。
    """

    chapter_scope = write_config.chapter_filter if write_config.chapter_filter else "ALL"
    tool_trace_config = resolved_options.trace_settings
    tool_trace_state = "ON" if tool_trace_config.enabled else "OFF"
    runner_running_config = resolved_options.runner_running_config
    agent_running_config = resolved_options.agent_running_config

    Log.info("写作流水线参数摘要:", module=MODULE)
    Log.info(
        f"- ticker={write_config.ticker}, chapter={chapter_scope}, resume={write_config.resume}, "
        f"write_max_retries={write_config.write_max_retries}, fast={write_config.fast}, "
        f"force={write_config.force}, infer={write_config.infer}",
        module=MODULE,
    )
    Log.info(
        f"- scene_models={json.dumps(serialize_scene_models(write_config.scene_models), ensure_ascii=False, sort_keys=True)}",
        module=MODULE,
    )
    Log.info(
        f"- template={write_config.template_path}, output={write_config.output_dir}",
        module=MODULE,
    )
    Log.info(
        f"- web_provider={write_config.web_provider}, "
        f"tool_trace={tool_trace_state}, tool_trace_dir={tool_trace_config.output_dir}",
        module=MODULE,
    )
    Log.info(
        f"- agent_max_iterations={getattr(agent_running_config, 'max_iterations', None)}, "
        f"tool_timeout_seconds={getattr(runner_running_config, 'tool_timeout_seconds', None)}",
        module=MODULE,
    )


class WritePipelineRunner:
    """写作流水线执行器。"""

    def __init__(
        self,
        *,
        workspace: WorkspaceResourcesProtocol,
        resolved_options: ResolvedExecutionOptions,
        write_config: WriteRunConfig,
        scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer,
        host_executor: HostExecutorProtocol,
        host_governance: HostGovernanceProtocol,
        host_session_id: str,
        execution_options: ExecutionOptions | None = None,
        company_name_resolver: Callable[[str], str] | None = None,
        company_meta_summary_resolver: Callable[[str], dict[str, str]] | None = None,
    ) -> None:
        """初始化写作流水线。

        Args:
            workspace: 工作区稳定资源。
            resolved_options: 合并后的运行选项。
            write_config: 写作运行配置。
            scene_execution_acceptance_preparer: scene 执行接受准备器。
            host_executor: 宿主执行器。
            host_governance: Host governance 面；用于读取并发 lane 最大值。
            host_session_id: 当前写作流水线复用的 Host Session。
            execution_options: 请求级覆盖参数。
            company_name_resolver: 可选公司名称解析函数。
            company_meta_summary_resolver: 可选公司基础 meta 摘要解析函数。

        Returns:
            无。

        Raises:
            OSError: 输出目录创建失败时抛出。
        """

        self._workspace = workspace
        self._resolved_options = resolved_options
        self._write_config = write_config
        self._company_name_resolver = company_name_resolver
        self._company_meta_summary_resolver = company_meta_summary_resolver
        self._host_governance = host_governance

        self._output_dir = Path(write_config.output_dir).resolve()
        self._chapters_dir = self._output_dir / _CHAPTERS_DIR_NAME
        self._manifest_path = self._output_dir / _MANIFEST_FILE_NAME
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._chapters_dir.mkdir(parents=True, exist_ok=True)

        self._store = ArtifactStore(
            output_dir=self._output_dir,
            chapters_dir=self._chapters_dir,
            manifest_path=self._manifest_path,
            write_config=self._write_config,
        )

        self._preparer = SceneContractPreparer(
            scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
            write_config=write_config,
            execution_options=execution_options,
            host_session_id=host_session_id,
            company_name_resolver=company_name_resolver,
            config_loader=workspace.config_loader,
        )
        self._prompt_runner = ScenePromptRunner(
            preparer=self._preparer,
            contract_executor=host_executor.run_agent_and_wait,
            write_config=write_config,
        )

        self._prompter = PromptBuilder(
            workspace=workspace,
            write_config=write_config,
            store=self._store,
        )
        self._report_assembler = ReportAssembler(write_config=write_config)
        self._execution_summary_builder = ExecutionSummaryBuilder(write_config=write_config)
        self._chapter_audit_coordinator = ChapterAuditCoordinator(
            write_config=write_config,
            store=self._store,
            preparer=self._preparer,
            prompt_runner=self._prompt_runner,
            prompter=self._prompter,
        )
        self._chapter_execution_coordinator = ChapterExecutionCoordinator(
            write_config=write_config,
            store=self._store,
            prompt_runner=self._prompt_runner,
            prompter=self._prompter,
            audit_coordinator=self._chapter_audit_coordinator,
        )

        _log_write_pipeline_config(
            write_config=self._write_config,
            resolved_options=self._resolved_options,
        )
        self._company_facets: CompanyFacetProfile | None = None
        self._company_facet_catalog: dict[str, list[str]] = {}

    def run(self) -> int:
        """执行完整写作流水线。

        Args:
            无。

        Returns:
            退出码：`0/4/2`。当显式启用 `--infer` 时，仅执行公司级 facet
            归因并写回 manifest，随后直接返回 `0`。

        Raises:
            RuntimeError: 核心流程异常时抛出。
        """

        template_path = Path(self._write_config.template_path).resolve()
        if not template_path.exists() or not template_path.is_file():
            Log.error(f"模板文件不存在: {template_path}", module=MODULE)
            return 2

        template_text = template_path.read_text(encoding="utf-8")
        layout = parse_template_layout(template_text)
        self._company_facet_catalog = dict(layout.company_facet_catalog)
        self._preparer.set_company_facet_catalog(self._company_facet_catalog)
        self._prompter.set_company_facet_catalog(self._company_facet_catalog)
        chapters_by_title = {chapter.title: chapter for chapter in layout.chapters}

        if _OVERVIEW_CHAPTER_TITLE not in chapters_by_title:
            Log.error("模板必须包含“投资要点概览”一级章节", module=MODULE)
            return 2

        has_source_chapter = _SOURCE_CHAPTER_TITLE in chapters_by_title

        chapter_filter = self._write_config.chapter_filter
        if chapter_filter:
            all_titles = {chapter.title for chapter in layout.chapters}
            if chapter_filter not in all_titles:
                Log.error(
                    f"--chapter 指定章节不存在: '{chapter_filter}'"
                    f"，可选内容：{sorted(all_titles)}",
                    module=MODULE,
                )
                return 2
            Log.info(f"[单章模式] 仅写章节: {chapter_filter}", module=MODULE)

        signature = _build_signature(
            template_text=template_text,
            ticker=self._write_config.ticker,
            scene_models=self._write_config.scene_models,
            web_provider=self._write_config.web_provider,
            write_max_retries=self._write_config.write_max_retries,
        )
        manifest = self._store.load_or_create_manifest(signature)

        company_name = self._resolve_company_name()
        self._company_facets = self._ensure_company_facets(manifest=manifest, company_name=company_name)
        self._prompter.set_company_facets(self._company_facets)
        if self._write_config.infer:
            Log.info("`--infer` 已完成公司级 Facet 归因，跳过后续写作阶段", module=MODULE)
            return 0

        middle_tasks = _build_middle_tasks(layout)
        chapter_results: dict[str, ChapterResult] = dict(manifest.chapter_results)
        if chapter_filter:
            chapter_results = self._merge_historical_chapter_results_for_single_chapter(chapter_results)

        if chapter_filter:
            for task in middle_tasks:
                if task.title != chapter_filter:
                    continue
                existing_result = chapter_results.get(task.title)
                if self._should_skip_with_resume(existing_result):
                    Log.info(f"[单章模式] 显式指定章节，强制重跑: {task.title}", module=MODULE)
                try:
                    result = self._run_single_chapter(
                        task=task,
                        company_name=company_name,
                        prompt_name="write_chapter",
                        prompt_inputs=self._prompter.build_chapter_prompt_inputs(task=task, company_name=company_name),
                    )
                except SceneAgentCreationError as exc:
                    Log.error(str(exc), module=MODULE)
                    return 2
                except CancelledError:
                    raise
                except Exception as exc:
                    Log.error(f"章节执行失败，已降级继续: {task.title}, error={exc}", module=MODULE)
                    result = self._build_fallback_result(
                        task=task,
                        reason=str(exc),
                        content=self._store.load_latest_failed_chapter_content(task),
                    )
                chapter_results[task.title] = result
                self._store.persist_chapter_artifacts(result)
                self._store.persist_manifest(manifest=manifest, chapter_results=chapter_results)
        else:
            try:
                chapter_results = self._run_middle_tasks_in_parallel(
                    middle_tasks=middle_tasks,
                    chapter_results=chapter_results,
                    manifest=manifest,
                    company_name=company_name,
                )
            except SceneAgentCreationError as exc:
                Log.error(str(exc), module=MODULE)
                return 2
            except CancelledError:
                raise

        if chapter_filter and chapter_filter not in {_OVERVIEW_CHAPTER_TITLE, _DECISION_CHAPTER_TITLE}:
            # 单章模式且目标既不是第0章概览章也不是决策章，跳过 decision + overview + 来源清单 + 报告组装
            single_result = chapter_results.get(chapter_filter)
            Log.info(f"[单章模式] 完成：{chapter_filter}", module=MODULE)
            return 0 if self._did_chapter_succeed_in_current_mode(single_result) else 4

        decision_result: ChapterResult | None = None
        decision_chapter = chapters_by_title.get(_DECISION_CHAPTER_TITLE)
        if decision_chapter is not None:
            decision_task = ChapterTask(
                index=decision_chapter.index,
                title=decision_chapter.title,
                skeleton=decision_chapter.skeleton,
                report_goal=layout.report_goal,
                audience_profile=layout.audience_profile,
                chapter_goal=decision_chapter.chapter_goal,
                chapter_contract=decision_chapter.chapter_contract,
                item_rules=decision_chapter.item_rules,
            )
            if not chapter_filter or chapter_filter == _DECISION_CHAPTER_TITLE:
                prior_decision_tasks = _build_prior_decision_tasks(layout)
                if chapter_filter == _DECISION_CHAPTER_TITLE and not self._write_config.force:
                    resolved_prior_results, precheck_errors = self._resolve_decision_single_chapter_prerequisites(
                        prior_tasks=prior_decision_tasks,
                        chapter_results=chapter_results,
                    )
                    if precheck_errors:
                        Log.error(
                            "[单章模式] 第10章缺少可用前置章节产物："
                            + "；".join(precheck_errors),
                            module=MODULE,
                        )
                        return 2
                    chapter_results.update(resolved_prior_results)
                elif not self._write_config.force:
                    blocking_titles = self._collect_blocking_task_titles_for_audit_gate(
                        tasks=prior_decision_tasks,
                        chapter_results=chapter_results,
                    )
                    if blocking_titles:
                        Log.error(
                            "[全文模式] 第10章缺少已通过 audit 的前置章节："
                            + "；".join(blocking_titles),
                            module=MODULE,
                        )
                        return 4
                existing_decision = chapter_results.get(_DECISION_CHAPTER_TITLE)
                if self._should_skip_with_resume(existing_decision):
                    if chapter_filter == _DECISION_CHAPTER_TITLE:
                        Log.info(f"[单章模式] 显式指定章节，强制重跑: {_DECISION_CHAPTER_TITLE}", module=MODULE)
                    else:
                        Log.info(f"跳过已完成章节: {_DECISION_CHAPTER_TITLE}", module=MODULE)
                        decision_result = existing_decision
                if decision_result is None:
                    try:
                        decision_result = self._run_single_chapter(
                            task=decision_task,
                            company_name=company_name,
                            prompt_name="write_research_decision",
                            prompt_inputs=self._prompter.build_research_decision_prompt_inputs(
                                task=decision_task,
                                company_name=company_name,
                                layout=layout,
                                chapter_results=chapter_results,
                            ),
                        )
                    except SceneAgentCreationError as exc:
                        Log.error(str(exc), module=MODULE)
                        return 2
                    except CancelledError:
                        raise
                    except Exception as exc:
                        Log.error(f"第10章执行失败，已降级继续: error={exc}", module=MODULE)
                        decision_result = self._build_fallback_result(
                            task=decision_task,
                            reason=str(exc),
                            content=self._store.load_latest_failed_chapter_content(decision_task),
                        )
                    chapter_results[_DECISION_CHAPTER_TITLE] = decision_result
                    self._store.persist_chapter_artifacts(decision_result)
                    self._store.persist_manifest(manifest=manifest, chapter_results=chapter_results)

        if chapter_filter == _DECISION_CHAPTER_TITLE:
            Log.info(f"[单章模式] 完成：{_DECISION_CHAPTER_TITLE}", module=MODULE)
            return 0 if self._did_chapter_succeed_in_current_mode(decision_result) else 4

        overview_task = ChapterTask(
            index=chapters_by_title[_OVERVIEW_CHAPTER_TITLE].index,
            title=_OVERVIEW_CHAPTER_TITLE,
            skeleton=chapters_by_title[_OVERVIEW_CHAPTER_TITLE].skeleton,
            report_goal=layout.report_goal,
            audience_profile=layout.audience_profile,
            chapter_goal=chapters_by_title[_OVERVIEW_CHAPTER_TITLE].chapter_goal,
            chapter_contract=chapters_by_title[_OVERVIEW_CHAPTER_TITLE].chapter_contract,
            item_rules=chapters_by_title[_OVERVIEW_CHAPTER_TITLE].item_rules,
        )
        existing_overview = chapter_results.get(_OVERVIEW_CHAPTER_TITLE)
        if self._should_skip_with_resume(existing_overview):
            Log.info(f"跳过已完成章节: {_OVERVIEW_CHAPTER_TITLE}", module=MODULE)
            overview_result = existing_overview
        else:
            overview_dependency_tasks = _build_overview_dependency_tasks(layout)
            if chapter_filter == _OVERVIEW_CHAPTER_TITLE and not self._write_config.force:
                resolved_overview_results, precheck_errors = self._resolve_overview_single_chapter_prerequisites(
                    dependency_tasks=overview_dependency_tasks,
                    chapter_results=chapter_results,
                )
                if precheck_errors:
                    Log.error(
                        "[单章模式] 第0章缺少可用前置章节产物："
                        + "；".join(precheck_errors),
                        module=MODULE,
                    )
                    return 2
                chapter_results.update(resolved_overview_results)
            elif not chapter_filter and not self._write_config.force:
                blocking_titles = self._collect_blocking_task_titles_for_audit_gate(
                    tasks=overview_dependency_tasks,
                    chapter_results=chapter_results,
                )
                if blocking_titles:
                    Log.error(
                        "[全文模式] 第0章缺少已通过 audit 的依赖章节："
                        + "；".join(blocking_titles),
                        module=MODULE,
                    )
                    return 4
            try:
                overview_result = self._run_single_chapter(
                    task=overview_task,
                    company_name=company_name,
                    prompt_name="fill_overview",
                    prompt_inputs=self._prompter.build_chapter_prompt_inputs(
                        task=overview_task,
                        company_name=company_name,
                        include_item_rules=False,
                        extra_inputs={
                            "prior_chapters_summary_input": self._prompter.build_overview_input(
                                layout=layout,
                                chapter_results=chapter_results,
                            ),
                            "overview_source_of_truth": "prior_chapters_summary_input_only",
                            "overview_allow_new_facts": False,
                            "overview_allow_new_sources": False,
                        },
                    ),
                )
            except SceneAgentCreationError as exc:
                Log.error(str(exc), module=MODULE)
                return 2
            except CancelledError:
                raise
            except Exception as exc:
                Log.error(f"第0章概览回填失败，已降级继续: error={exc}", module=MODULE)
                overview_result = self._build_fallback_result(
                    task=overview_task,
                    reason=str(exc),
                    content=self._store.load_latest_failed_chapter_content(overview_task),
                )
            chapter_results[_OVERVIEW_CHAPTER_TITLE] = overview_result
            self._store.persist_chapter_artifacts(overview_result)
            self._store.persist_manifest(manifest=manifest, chapter_results=chapter_results)

        if chapter_filter == _OVERVIEW_CHAPTER_TITLE:
            # 单章模式且目标为第0章概览章，跳过来源清单和报告组装
            Log.info(f"[单章模式] 完成：{_OVERVIEW_CHAPTER_TITLE}", module=MODULE)
            return 0 if self._did_chapter_succeed_in_current_mode(overview_result) else 4

        source_chapter_markdown: str | None = None
        if has_source_chapter:
            all_evidence_items = _collect_all_evidence_items(
                chapter_results=chapter_results,
                ordered_titles=[chapter.title for chapter in layout.chapters if chapter.title != _SOURCE_CHAPTER_TITLE],
            )
            source_entries = build_source_entries(all_evidence_items)
            source_chapter_markdown = render_source_list_chapter(source_entries)
            self._store.persist_sources_json(source_entries)
        else:
            self._store.remove_sources_json_if_exists()

        final_markdown = self._report_assembler.assemble_report(
            layout,
            chapter_results,
            source_chapter_markdown,
            company_name=company_name,
        )
        output_file = self._output_dir / f"{self._write_config.ticker}_qual_report.md"
        output_file.write_text(final_markdown, encoding="utf-8")

        summary_payload = self._execution_summary_builder.build_summary(
            chapter_results,
            output_file=output_file,
            success_predicate=self._did_chapter_succeed_in_current_mode,
        )
        (self._output_dir / "run_summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        Log.info(f"写作完成: {output_file}", module=MODULE)
        return 4 if summary_payload["failed_count"] > 0 else 0

    def _resolve_middle_worker_limit(self) -> int:
        """通过 Host governance 读取中间章节并发上限。

        Returns:
            ``write_chapter`` lane 的最大并发数。

        Raises:
            RuntimeError: ``write_chapter`` lane 未配置时抛出。
        """

        lane_statuses = self._host_governance.get_all_lane_statuses()
        lane_status = lane_statuses.get(LANE_WRITE_CHAPTER)
        if lane_status is None:
            raise RuntimeError(
                f"缺少 {LANE_WRITE_CHAPTER} lane 配置，无法确定写作并发上限"
            )
        if lane_status.max_concurrent <= 0:
            raise RuntimeError(
                f"{LANE_WRITE_CHAPTER} lane 配置非法: max_concurrent={lane_status.max_concurrent}"
            )
        return lane_status.max_concurrent

    def _run_middle_tasks_in_parallel(
        self,
        *,
        middle_tasks: list[ChapterTask],
        chapter_results: dict[str, ChapterResult],
        manifest: RunManifest,
        company_name: str,
    ) -> dict[str, ChapterResult]:
        """并发执行全文模式下的中间章节批次。

        设计约束：
        - 只并发执行第1-9章等常规中间章节。
        - worker 线程只负责运行单章闭环并返回结果，不直接修改共享 manifest。
        - 主线程按模板顺序统一落盘章节正文与 manifest，保证顺序稳定。

        Args:
            middle_tasks: 中间章节任务列表。
            chapter_results: 当前章节结果映射。
            manifest: 当前运行 manifest。
            company_name: 公司名称。

        Returns:
            更新后的章节结果映射。

        Raises:
            SceneAgentCreationError: 当 scene Agent 创建失败时抛出。
        """

        pending_tasks: list[ChapterTask] = []
        for task in middle_tasks:
            existing_result = chapter_results.get(task.title)
            if self._should_skip_with_resume(existing_result):
                Log.info(f"跳过已完成章节: {task.title}", module=MODULE)
                continue
            pending_tasks.append(task)
        if not pending_tasks:
            return chapter_results

        worker_count = min(self._resolve_middle_worker_limit(), len(pending_tasks))
        completed_results: dict[str, ChapterResult] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_task: dict[concurrent.futures.Future[ChapterResult], ChapterTask] = {}
            pending_index = 0

            while pending_index < len(pending_tasks) and len(future_to_task) < worker_count:
                task = pending_tasks[pending_index]
                future = executor.submit(
                    self._run_middle_task_worker,
                    task=task,
                    company_name=company_name,
                )
                future_to_task[future] = task
                pending_index += 1

            try:
                while future_to_task:
                    completed_futures, _ = concurrent.futures.wait(
                        tuple(future_to_task),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in completed_futures:
                        task = future_to_task.pop(future)
                        completed_results[task.title] = future.result()
                        if pending_index >= len(pending_tasks):
                            continue
                        next_task = pending_tasks[pending_index]
                        next_future = executor.submit(
                            self._run_middle_task_worker,
                            task=next_task,
                            company_name=company_name,
                        )
                        future_to_task[next_future] = next_task
                        pending_index += 1
            except (CancelledError, SceneAgentCreationError):
                self._cancel_pending_middle_task_futures(future_to_task)
                raise

        for task in middle_tasks:
            result = completed_results.get(task.title)
            if result is None:
                continue
            chapter_results[task.title] = result
            self._store.persist_chapter_artifacts(result)
            self._store.persist_manifest(manifest=manifest, chapter_results=chapter_results)
        return chapter_results

    def _cancel_pending_middle_task_futures(
        self,
        future_to_task: dict[concurrent.futures.Future[ChapterResult], ChapterTask],
    ) -> None:
        """取消尚未启动的中间章节 future，避免继续派发无效任务。

        Args:
            future_to_task: 当前仍未完成的 future 到任务映射。

        Returns:
            无。

        Raises:
            无。
        """

        cancelled_titles: list[str] = []
        for future, task in future_to_task.items():
            if future.cancel():
                cancelled_titles.append(task.title)
        if cancelled_titles:
            Log.info(
                f"中止中间章节批次后，已取消尚未启动任务: {cancelled_titles}",
                module=MODULE,
            )

    def _run_middle_task_worker(self, *, task: ChapterTask, company_name: str) -> ChapterResult:
        """执行单个中间章节 worker，并在可恢复异常时生成兜底结果。

        Args:
            task: 中间章节任务。
            company_name: 公司名称。

        Returns:
            单章执行结果。

        Raises:
            SceneAgentCreationError: 当 scene Agent 创建失败时抛出。
        """

        try:
            return self._run_single_chapter(
                task=task,
                company_name=company_name,
                prompt_name="write_chapter",
                prompt_inputs=self._prompter.build_chapter_prompt_inputs(task=task, company_name=company_name),
            )
        except SceneAgentCreationError:
            raise
        except CancelledError:
            raise
        except Exception as exc:
            Log.error(f"章节执行失败，已降级继续: {task.title}, error={exc}", module=MODULE)
            return self._build_fallback_result(
                task=task,
                reason=str(exc),
                content=self._store.load_latest_failed_chapter_content(task),
            )

    def _run_single_chapter(
        self,
        *,
        task: ChapterTask,
        company_name: str,
        prompt_name: str,
        prompt_inputs: dict[str, Any],
    ) -> ChapterResult:
        """执行单个章节完整流水线。

        Args:
            task: 章节任务。
            company_name: 公司名称。
            prompt_name: 初始写作 prompt 名称。
            prompt_inputs: 传入 task prompt contract 的显式字段字典。

        Returns:
            章节执行结果。

        Raises:
            RuntimeError: 当模型调用失败时抛出。
        """

        return self._chapter_execution_coordinator.run_single_chapter(
            task=task,
            company_name=company_name,
            prompt_name=prompt_name,
            prompt_inputs=prompt_inputs,
            company_facets=self._company_facets,
            company_facet_catalog=self._company_facet_catalog,
        )

    def _resolve_decision_single_chapter_prerequisites(
        self,
        *,
        prior_tasks: list[ChapterTask],
        chapter_results: dict[str, ChapterResult],
    ) -> tuple[dict[str, ChapterResult], list[str]]:
        """解析单章模式下第10章所需的前置章节产物。

        该解析优先复用当前 manifest 中的章节结果；若 manifest 缺失但磁盘上已有
        已通过审计的章节正文与 audit 产物，则会从文件系统恢复最小可用结果。

        Args:
            prior_tasks: 第10章依赖的前置章节任务列表。
            chapter_results: 当前已知章节结果。

        Returns:
            `(resolved_results, errors)` 二元组：
            - `resolved_results`：可安全用于第10章的前置章节结果映射。
            - `errors`：缺失/不可用原因列表；为空表示全部前置章节均可用。

        Raises:
            无。
        """

        resolved_results: dict[str, ChapterResult] = {}
        errors: list[str] = []
        for task in prior_tasks:
            result = chapter_results.get(task.title)
            chapter_path = self._store.chapter_file_path(task.index, task.title)
            try:
                file_content = chapter_path.read_text(encoding="utf-8").strip() if chapter_path.exists() else ""
            except OSError as exc:
                errors.append(f"{task.title} 章节文件不可读: {exc}")
                continue
            if result is not None and result.status == "passed" and result.audit_passed and file_content:
                resolved_results[task.title] = ChapterResult(
                    index=task.index,
                    title=task.title,
                    status="passed",
                    content=file_content,
                    audit_passed=True,
                    retry_count=result.retry_count,
                    failure_reason="",
                    evidence_items=result.evidence_items,
                    process_state=result.process_state,
                )
                continue
            recovered_result, recovery_error = self._recover_prior_chapter_result_from_artifacts(task)
            if recovered_result is not None:
                resolved_results[task.title] = recovered_result
                continue
            errors.append(recovery_error or f"{task.title} 缺少可用前置章节产物")
        return resolved_results, errors

    def _resolve_overview_single_chapter_prerequisites(
        self,
        *,
        dependency_tasks: list[ChapterTask],
        chapter_results: dict[str, ChapterResult],
    ) -> tuple[dict[str, ChapterResult], list[str]]:
        """解析单章模式下第0章所需的前置章节产物。

        Args:
            dependency_tasks: 第0章依赖的前置章节任务列表。
            chapter_results: 当前已知章节结果。

        Returns:
            `(resolved_results, errors)` 二元组。

        Raises:
            无。
        """

        resolved_results: dict[str, ChapterResult] = {}
        errors: list[str] = []
        for task in dependency_tasks:
            result = chapter_results.get(task.title)
            chapter_path = self._store.chapter_file_path(task.index, task.title)
            try:
                file_content = chapter_path.read_text(encoding="utf-8").strip() if chapter_path.exists() else ""
            except OSError as exc:
                errors.append(f"{task.title} 章节文件不可读: {exc}")
                continue
            if result is not None and result.status == "passed" and result.audit_passed and file_content:
                resolved_results[task.title] = ChapterResult(
                    index=task.index,
                    title=task.title,
                    status="passed",
                    content=file_content,
                    audit_passed=True,
                    retry_count=result.retry_count,
                    failure_reason="",
                    evidence_items=result.evidence_items,
                    process_state=result.process_state,
                )
                continue
            recovered_result, recovery_error = self._recover_prior_chapter_result_from_artifacts(task)
            if recovered_result is not None:
                resolved_results[task.title] = recovered_result
                continue
            errors.append(recovery_error or f"{task.title} 缺少可用前置章节产物")
        return resolved_results, errors

    def _collect_blocking_task_titles_for_audit_gate(
        self,
        *,
        tasks: list[ChapterTask],
        chapter_results: dict[str, ChapterResult],
    ) -> list[str]:
        """收集未满足第0章/第10章 audit 门禁的章节说明。

        Args:
            tasks: 需要检查的依赖章节任务列表。
            chapter_results: 当前章节结果映射。

        Returns:
            阻塞原因列表。

        Raises:
            无。
        """

        blocking_titles: list[str] = []
        for task in tasks:
            result = chapter_results.get(task.title)
            if self._satisfies_audit_gate_for_current_mode(result):
                continue
            if result is None:
                blocking_titles.append(f"{task.title} 缺少章节结果")
                continue
            if result.status != "passed":
                blocking_titles.append(f"{task.title} 写作失败")
                continue
            if not result.content:
                blocking_titles.append(f"{task.title} 缺少章节正文")
                continue
            blocking_titles.append(f"{task.title} 未通过最终审计")
        return blocking_titles

    def _satisfies_audit_gate_for_current_mode(self, result: ChapterResult | None) -> bool:
        """判断章节是否满足当前模式下的 audit 门禁。

        Args:
            result: 待检查章节结果。

        Returns:
            满足门禁返回 ``True``。

        Raises:
            无。
        """

        if result is None:
            return False
        if result.status != "passed" or not result.content:
            return False
        if self._write_config.fast:
            return True
        return bool(result.audit_passed)

    def _did_chapter_succeed_in_current_mode(self, result: ChapterResult | None) -> bool:
        """判断章节在当前运行模式下是否算成功。

        Args:
            result: 待检查章节结果。

        Returns:
            当前模式下成功返回 ``True``。

        Raises:
            无。
        """

        return self._satisfies_audit_gate_for_current_mode(result)

    def _recover_prior_chapter_result_from_artifacts(
        self, task: ChapterTask
    ) -> tuple[ChapterResult | None, str]:
        """从已落盘章节产物恢复第10章所需的最小前置章节结果。

        Args:
            task: 前置章节任务。

        Returns:
            `(result, error)` 二元组；恢复成功时返回结果对象与空错误信息。

        Raises:
            无。
        """

        chapter_path = self._store.chapter_file_path(task.index, task.title)
        if not chapter_path.exists():
            return None, f"{task.title} 缺少章节文件"
        try:
            content = chapter_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            return None, f"{task.title} 章节文件不可读: {exc}"
        if not content:
            return None, f"{task.title} 章节文件为空"

        audit_path = self._store.chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name="audit",
            extension="json",
        )
        if not audit_path.exists():
            return None, f"{task.title} 缺少最终 audit 产物"
        try:
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"{task.title} audit 产物不可读: {exc}"
        if not isinstance(audit_payload, dict) or not bool(audit_payload.get("pass")):
            return None, f"{task.title} 未通过最终审计"

        return (
            ChapterResult(
                index=task.index,
                title=task.title,
                status="passed",
                content=content,
                audit_passed=True,
            ),
            "",
        )

    def _resolve_company_name(self) -> str:
        """解析公司名称。

        Args:
            无。

        Returns:
            公司名称；缺失时回退 ticker。

        Raises:
            无。
        """

        try:
            if self._company_name_resolver is None:
                return self._write_config.ticker
            resolved_name = str(self._company_name_resolver(self._write_config.ticker) or "").strip()
            return resolved_name or self._write_config.ticker
        except Exception:
            return self._write_config.ticker

    def _resolve_company_meta_summary(self) -> dict[str, str]:
        """解析公司基础 meta 摘要。

        Args:
            无。

        Returns:
            公司基础 meta 摘要；最少包含 `ticker`。

        Raises:
            无。
        """

        try:
            if self._company_meta_summary_resolver is None:
                return {"ticker": self._write_config.ticker}
            raw_meta = self._company_meta_summary_resolver(self._write_config.ticker)
            if not isinstance(raw_meta, dict):
                return {"ticker": self._write_config.ticker}
            normalized = {
                str(key).strip(): str(value or "").strip()
                for key, value in raw_meta.items()
                if str(key).strip()
            }
            normalized.setdefault("ticker", self._write_config.ticker)
            return normalized
        except Exception:
            return {"ticker": self._write_config.ticker}

    def _ensure_company_facets(self, *, manifest: RunManifest, company_name: str) -> CompanyFacetProfile | None:
        """确保当前 ticker 已具备公司级 facet 归因结果。

        Args:
            manifest: 当前运行清单。
            company_name: 公司名称。

        Returns:
            可用的公司级 facet 归因结果；自动兜底场景下若推理失败则返回历史结果或 `None`。

        Raises:
            RuntimeError: 当显式 `--infer` 但推理失败时抛出。
            SceneAgentCreationError: 当 infer scene 无法创建时向上抛出。
        """

        if manifest.company_facets is not None and not self._write_config.infer:
            return manifest.company_facets
        Log.info(
            "开始公司级 Facet 归因: "
            f"ticker={self._write_config.ticker}, company={company_name}, "
            f"force_refresh={self._write_config.infer}",
            module=MODULE,
        )
        try:
            inferred = self._infer_company_facets(company_name=company_name)
        except SceneAgentCreationError:
            raise
        except Exception as exc:  # noqa: BLE001
            if self._write_config.infer:
                raise RuntimeError(f"公司级 Facet 归因失败: {exc}") from exc
            Log.warn(f"公司级 Facet 归因失败，回退到未裁剪写作: error={exc}", module=MODULE)
            return manifest.company_facets
        manifest.company_facets = inferred
        self._store.persist_manifest(manifest=manifest, chapter_results=dict(manifest.chapter_results))
        Log.info(
            "完成公司级 Facet 归因并写回 manifest: "
            f"ticker={self._write_config.ticker}, primary={inferred.primary_facets}, "
            f"cross_cutting={inferred.cross_cutting_facets}",
            module=MODULE,
        )
        return inferred

    def _infer_company_facets(self, *, company_name: str) -> CompanyFacetProfile:
        """执行一次公司级 facet 归因。

        Args:
            company_name: 公司名称。

        Returns:
            结构化 facet 归因结果。

        Raises:
            RuntimeError: 当 infer 调用失败或输出非法时抛出。
        """

        infer_prompt = self._prompter.render_task_prompt(
            prompt_name="infer_company_facets",
            prompt_inputs={
                "company_meta": _sanitize_company_meta_for_infer(self._resolve_company_meta_summary()),
                "business_model_candidates": self._company_facet_catalog.get("business_model_candidates", []),
                "constraint_candidates": self._company_facet_catalog.get("constraint_candidates", []),
            },
        )
        return self._prompt_runner.run_infer_prompt(infer_prompt)

    def _should_skip_with_resume(self, existing_result: Optional[ChapterResult]) -> bool:
        """判断是否应在 resume 模式跳过章节。

        Args:
            existing_result: 历史章节结果。

        Returns:
            应跳过返回 `True`。

        Raises:
            无。
        """

        if not self._write_config.resume:
            return False
        if existing_result is None:
            return False
        # resume 只跳过在“当前运行门槛”下已完成的章节。
        return self._did_chapter_succeed_in_current_mode(existing_result)

    def _merge_historical_chapter_results_for_single_chapter(
        self, chapter_results: dict[str, ChapterResult]
    ) -> dict[str, ChapterResult]:
        """单章模式下合并历史章节结果，避免运行单章时丢失其他章节状态。

        Args:
            chapter_results: 当前会话章节结果。

        Returns:
            合并后的章节结果（当前会话结果优先）。

        Raises:
            无。
        """

        try:
            with _manifest_file_lock(self._output_dir):
                historical_manifest = self._store.read_manifest_from_disk_unlocked()
        except Exception as exc:  # noqa: BLE001
            Log.warn(f"[单章模式] 读取历史 manifest 失败，跳过历史合并: {exc}", module=MODULE)
            return chapter_results
        if historical_manifest is None:
            return chapter_results
        if historical_manifest.config.ticker != self._write_config.ticker:
            Log.warn(
                "[单章模式] 历史 manifest ticker 不匹配，跳过历史合并: "
                f"history={historical_manifest.config.ticker}, current={self._write_config.ticker}",
                module=MODULE,
            )
            return chapter_results
        merged = dict(historical_manifest.chapter_results)
        merged.update(chapter_results)
        preserved_count = len(merged) - len(chapter_results)
        if preserved_count > 0:
            Log.info(f"[单章模式] 已保留历史章节结果: {preserved_count}", module=MODULE)
        return merged

    def _build_fallback_result(self, *, task: ChapterTask, reason: str, content: str = "") -> ChapterResult:
        """构建章节失败兜底结果。

        Args:
            task: 章节任务。
            reason: 失败原因。
            content: 失败时可保留的最新中间稿；为空时表示不落最终正文。

        Returns:
            失败章节结果对象。

        Raises:
            无。
        """

        return ChapterResult(
            index=task.index,
            title=task.title,
            status="failed",
            content=content,
            audit_passed=False,
            retry_count=0,
            failure_reason=reason,
            evidence_items=extract_evidence_items(content),
            process_state={**build_process_state_template(), "fix_reason": "fallback", "final_stage": "runtime_error"},
        )


def run_write_pipeline(
    *,
    workspace: WorkspaceResourcesProtocol,
    resolved_options: ResolvedExecutionOptions,
    write_config: WriteRunConfig,
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer,
    host_executor: HostExecutorProtocol,
    host_governance: HostGovernanceProtocol,
    host_session_id: str,
    execution_options: ExecutionOptions | None = None,
    company_name_resolver: Callable[[str], str] | None = None,
    company_meta_summary_resolver: Callable[[str], dict[str, str]] | None = None,
) -> int:
    """执行写作流水线入口函数。

    Args:
        workspace: 工作区稳定资源。
        resolved_options: 合并后的运行选项。
        write_config: 写作配置。
        scene_execution_acceptance_preparer: scene 执行接受准备器。
        host_executor: 宿主执行器。
        host_session_id: 当前写作流水线复用的 Host Session。
        execution_options: 请求级覆盖参数。
        company_name_resolver: 可选公司名称解析函数。
        company_meta_summary_resolver: 可选公司基础 meta 摘要解析函数。

    Returns:
        流程退出码。

    Raises:
        RuntimeError: 运行异常时抛出。
    """

    runner = WritePipelineRunner(
        workspace=workspace,
        resolved_options=resolved_options,
        write_config=write_config,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        host_executor=host_executor,
        host_governance=host_governance,
        host_session_id=host_session_id,
        execution_options=execution_options,
        company_name_resolver=company_name_resolver,
        company_meta_summary_resolver=company_meta_summary_resolver,
    )
    return runner.run()


def _build_chapter_tasks(
    *,
    layout: TemplateLayout,
    excluded_titles: frozenset[str],
) -> list[ChapterTask]:
    """按模板顺序构建章节任务列表。

    Args:
        layout: 模板布局对象。
        excluded_titles: 需要排除的章节标题集合。

    Returns:
        章节任务列表。

    Raises:
        无。
    """

    tasks: list[ChapterTask] = []
    for chapter in layout.chapters:
        if chapter.title in excluded_titles:
            continue
        tasks.append(
            ChapterTask(
                index=chapter.index,
                title=chapter.title,
                skeleton=chapter.skeleton,
                report_goal=layout.report_goal,
                audience_profile=layout.audience_profile,
                chapter_goal=chapter.chapter_goal,
                chapter_contract=chapter.chapter_contract,
                item_rules=chapter.item_rules,
            )
        )
    return tasks


def _build_middle_tasks(layout: TemplateLayout) -> list[ChapterTask]:
    """构建常规中间章节任务列表。

    这里会跳过：
    - 第0章“投资要点概览”
    - 第10章“是否值得继续深研与待验证问题”
    - 末章“来源清单”

    Args:
        layout: 模板布局对象。

    Returns:
        中间章节任务列表。

    Raises:
        无。
    """

    return _build_chapter_tasks(
        layout=layout,
        excluded_titles=frozenset(
            {
                _OVERVIEW_CHAPTER_TITLE,
                _DECISION_CHAPTER_TITLE,
                _SOURCE_CHAPTER_TITLE,
            }
        ),
    )


def _build_overview_dependency_tasks(layout: TemplateLayout) -> list[ChapterTask]:
    """构建第0章概览回填依赖的章节任务列表。

    约定：
    - 第0章要求第1章至第10章全部先完成。
    - “来源清单”不作为首章回填的前置依赖。

    Args:
        layout: 模板布局对象。

    Returns:
        第0章依赖的章节任务列表，按模板顺序排列。

    Raises:
        无。
    """

    return _build_chapter_tasks(
        layout=layout,
        excluded_titles=frozenset(
            {
                _OVERVIEW_CHAPTER_TITLE,
                _SOURCE_CHAPTER_TITLE,
            }
        ),
    )


def _build_signature(
    *,
    template_text: str,
    ticker: str,
    scene_models: dict[str, Any],
    web_provider: str,
    write_max_retries: int,
) -> str:
    """构建运行签名。

    Args:
        template_text: 模板全文。
        ticker: 股票代码。
        scene_models: 各 scene 实际模型配置映射。
        web_provider: 联网 provider。
        write_max_retries: 重写上限。

    Returns:
        SHA-256 签名。

    Raises:
        无。
    """

    payload = "|".join(
        [
            hashlib.sha256(template_text.encode("utf-8")).hexdigest(),
            ticker,
            json.dumps(serialize_scene_models(scene_models), ensure_ascii=False, sort_keys=True),
            web_provider,
            str(write_max_retries),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def print_write_report(output_dir: str | Path) -> int:
    """打印写作流水线运行报告。

    从指定输出目录读取 manifest.json，统计写作成功/失败数量，并列出失败章节及原因。

    Args:
        output_dir: 写作输出目录路径（字符串或 Path）。

    Returns:
        退出码：0 表示全部通过，4 表示存在失败章节，2 表示 manifest 不存在。

    Raises:
        无。
    """

    resolved = Path(output_dir).resolve()
    manifest = _read_manifest_from_dir(resolved)
    if manifest is None:
        print(f"[REPORT] manifest.json 不存在：{resolved / _MANIFEST_FILE_NAME}")
        return 2

    results = manifest.chapter_results
    total = len(results)
    passed = sum(1 for r in results.values() if r.status == "passed")
    failed_results = [r for r in results.values() if r.status != "passed"]
    failed_count = len(failed_results)

    # 按章节序号排序
    failed_results.sort(key=lambda r: r.index)

    print()
    print("=" * 60)
    print(f"  写作报告  [{manifest.config.ticker}]")
    print("=" * 60)
    print(f"  总章节数   : {total}")
    print(f"  写作成功   : {passed}")
    print(f"  写作失败   : {failed_count}")
    print("-" * 60)

    if failed_results:
        print("  失败章节列表：")
        for r in failed_results:
            # 截断过长的失败原因，避免刷屏
            reason = r.failure_reason or "（未记录原因）"
            if len(reason) > 120:
                reason = reason[:117] + "..."
            status_label = "pending" if r.status == "pending" else "failed"
            print(f"  [{r.index:02d}] {r.title}  [{status_label}]")
            print(f"       原因: {reason}")
    else:
        print("  所有章节均写作成功。")

    print("=" * 60)
    print()

    return 0 if failed_count == 0 else 4
