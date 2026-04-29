"""章节执行协调器。

该模块负责单章初始写作、章节状态机与重写闭环，使 WritePipelineRunner
 退回到全文级 orchestrator 角色。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from dayu.log import Log
from dayu.services.internal.write_pipeline.audit_formatting import (
    _build_allowed_conditional_headings,
    _normalize_chapter_markdown_for_audit,
)
from dayu.services.internal.write_pipeline.audit_rules import (
    RepairOutputError,
)
from dayu.services.internal.write_pipeline.artifact_store import ArtifactStore, _build_phase_artifact_name
from dayu.services.internal.write_pipeline.enums import (
    RepairStrategy,
    WritePhaseName,
    build_rewrite_phase_name,
    normalize_repair_strategy,
)
from dayu.services.internal.write_pipeline.models import (
    AuditDecision,
    ChapterResult,
    ChapterTask,
    CompanyFacetProfile,
    EvidenceConfirmationResult,
    WriteRunConfig,
)
from dayu.services.internal.write_pipeline.prompt_builder import PromptBuilder
from dayu.services.internal.write_pipeline.repair_executor import _apply_repair_plan_with_details
from dayu.services.internal.write_pipeline.scene_executor import ScenePromptRunner
from dayu.services.internal.write_pipeline.source_list_builder import extract_evidence_items

if TYPE_CHECKING:
    from dayu.services.internal.write_pipeline.chapter_audit_coordinator import ChapterAuditCoordinator

MODULE = "APP.WRITE_PIPELINE"
_INITIAL_WRITE_ARTIFACT_NAME = "initial_write"


def _build_chapter_scene_debug_message(
    *,
    chapter_title: str,
    prompt_name: str,
    phase: str,
    scene_name: str,
) -> str:
    """构造章节 scene 调试日志。

    Args:
        chapter_title: 章节标题。
        prompt_name: 当前 prompt 名称。
        phase: 当前阶段名。
        scene_name: 当前调用的 scene 名称。

    Returns:
        统一格式的调试日志文本。

    Raises:
        无。
    """

    return (
        "章节 scene 调用摘要: "
        f"chapter_title={chapter_title}, prompt_name={prompt_name}, "
        f"phase={phase}, scene_name={scene_name}"
    )


class ChapterExecutionStage(StrEnum):
    """章节执行状态机阶段。"""

    EVALUATE_PHASE = "evaluate_phase"
    PREPARE_REWRITE = "prepare_rewrite"
    APPLY_REWRITE = "apply_rewrite"
    COMPLETE = "complete"


@dataclass
class ChapterExecutionState:
    """章节执行状态机上下文。"""

    task: ChapterTask
    company_name: str
    current_content: str
    allowed_conditional_headings: set[str]
    process_state: dict[str, Any]
    phase: str = WritePhaseName.INITIAL
    retry_count: int = 0
    pending_retry_count: int | None = None
    stage: ChapterExecutionStage = ChapterExecutionStage.EVALUATE_PHASE
    rewrite_strategy: RepairStrategy = RepairStrategy.NONE
    audit_decision: AuditDecision | None = None
    suspected_decision: AuditDecision | None = None
    confirmation_result: EvidenceConfirmationResult | None = None
    stop_rewrite_loop: bool = False


def require_audit_decision(
    execution_state: ChapterExecutionState,
    *,
    action: str,
) -> AuditDecision:
    """确保当前阶段已有审计结果。"""

    if execution_state.audit_decision is None:
        raise RuntimeError(f"{action}前缺少审计结果")
    return execution_state.audit_decision


def resolve_rewrite_retry_count(
    execution_state: ChapterExecutionState,
    *,
    action: str,
) -> int:
    """解析当前 rewrite 阶段应使用的重试序号。

    Args:
        execution_state: 当前章节执行状态。
        action: 当前动作名，用于错误提示。

    Returns:
        当前 rewrite 尝试对应的重试序号。

    Raises:
        RuntimeError: 当前上下文既没有待提交重试序号，也没有兼容旧路径的已知重试序号时抛出。
    """

    if execution_state.pending_retry_count is not None:
        return execution_state.pending_retry_count
    if execution_state.retry_count > 0:
        return execution_state.retry_count
    raise RuntimeError(f"{action}前缺少 rewrite 重试序号")


def build_process_state_template() -> dict[str, Any]:
    """构建章节执行过程状态模板。"""

    return {
        "flow_version": "write_phase_state_machine_v5",
        "fix_applied": False,
        "fix_reason": "",
        "audit_history": [],
        "confirm_history": [],
        "anchor_rewrite_history": [],
        "rewrite_history": [],
        "repair_contract_history": [],
        "latest_repair_contract": {},
        "latest_anchor_rewrite": {},
        "final_stage": "unknown",
        "rewrite_exhausted": False,
    }


def build_chapter_execution_state(
    *,
    task: ChapterTask,
    company_name: str,
    current_content: str,
    allowed_conditional_headings: set[str],
) -> ChapterExecutionState:
    """构建章节状态机初始上下文。"""

    return ChapterExecutionState(
        task=task,
        company_name=company_name,
        current_content=current_content,
        allowed_conditional_headings=set(allowed_conditional_headings),
        process_state=build_process_state_template(),
    )


def append_audit_process_state(
    process_state: dict[str, Any],
    *,
    phase: str,
    decision: AuditDecision,
) -> None:
    """追加审计阶段状态。"""

    process_state["audit_history"].append(
        {
            "phase": phase,
            "pass": bool(decision.passed),
            "class": str(decision.category),
            "violations_count": len(decision.violations),
        }
    )
    process_state["repair_contract_history"].append(
        {
            "phase": phase,
            "contract_version": decision.repair_contract.contract_version,
            "preferred_tool_action": decision.repair_contract.preferred_tool_action,
            "retry_scope": decision.repair_contract.retry_scope,
            "missing_evidence_slots_count": len(decision.repair_contract.missing_evidence_slots),
            "offending_claim_spans_count": len(decision.repair_contract.offending_claim_spans),
        }
    )
    process_state["latest_repair_contract"] = asdict(decision.repair_contract)


def append_confirm_process_state(
    process_state: dict[str, Any],
    *,
    phase: str,
    result: EvidenceConfirmationResult,
) -> None:
    """追加证据复核阶段状态。"""

    status_counts: dict[str, int] = {}
    for entry in result.entries:
        status_key = str(entry.status)
        status_counts[status_key] = status_counts.get(status_key, 0) + 1
    process_state.setdefault("confirm_history", []).append(
        {
            "phase": phase,
            "entries_count": len(result.entries),
            "status_counts": status_counts,
        }
    )


def append_anchor_rewrite_process_state(
    process_state: dict[str, Any],
    *,
    phase: str,
    attempted: bool,
    applied: bool,
    skip_reason: str = "",
    failure_reason: str = "",
    resolved_violations_count: int = 0,
) -> None:
    """追加证据锚点轻量修复阶段状态。

    Args:
        process_state: 章节过程状态。
        phase: 当前阶段名。
        attempted: 是否已经真正生成过待落盘重写结果。
        applied: 是否已成功通过后验校验并落盘。
        skip_reason: 未尝试时的跳过原因。
        failure_reason: 已尝试但失败时的失败原因。
        resolved_violations_count: 本次重写成功吸收的问题数量。

    Returns:
        无。

    Raises:
        无。
    """

    entry = {
        "phase": phase,
        "attempted": attempted,
        "applied": applied,
        "skip_reason": skip_reason,
        "failure_reason": failure_reason,
        "resolved_violations_count": resolved_violations_count,
    }
    process_state.setdefault("anchor_rewrite_history", []).append(entry)
    process_state["latest_anchor_rewrite"] = entry


def append_rewrite_process_state(
    process_state: dict[str, Any],
    *,
    retry_count: int,
    trigger_class: str,
    repair_strategy: RepairStrategy,
) -> None:
    """追加重写阶段状态。"""

    process_state["rewrite_history"].append(
        {
            "retry_count": retry_count,
            "trigger_class": trigger_class,
            "repair_strategy": str(repair_strategy),
        }
    )


class ChapterExecutionCoordinator:
    """单章执行状态机协调器。"""

    def __init__(
        self,
        *,
        write_config: WriteRunConfig,
        store: ArtifactStore,
        prompt_runner: ScenePromptRunner,
        prompter: PromptBuilder,
        audit_coordinator: ChapterAuditCoordinator,
    ) -> None:
        """初始化章节执行协调器。

        Args:
            write_config: 写作运行配置。
            store: 写作产物存储器。
            prompt_runner: scene prompt 执行器。
            prompter: prompt 构建器。
            audit_coordinator: 单章审计协调器。

        Returns:
            无。

        Raises:
            无。
        """

        self._write_config = write_config
        self._store = store
        self._prompt_runner = prompt_runner
        self._prompter = prompter
        self._audit_coordinator = audit_coordinator

    def run_single_chapter(
        self,
        *,
        task: ChapterTask,
        company_name: str,
        prompt_name: str,
        prompt_inputs: dict[str, Any],
        company_facets: CompanyFacetProfile | None,
        company_facet_catalog: dict[str, list[str]],
    ) -> ChapterResult:
        """执行单个章节完整流水线。"""

        Log.info(f"开始写章节: {task.title}", module=MODULE)
        Log.debug(
            _build_chapter_scene_debug_message(
                chapter_title=task.title,
                prompt_name=prompt_name,
                phase=WritePhaseName.INITIAL,
                scene_name=(
                    "overview"
                    if prompt_name == "fill_overview"
                    else "decision"
                    if prompt_name == "write_research_decision"
                    else "write"
                ),
            ),
            module=MODULE,
        )
        initial_prompt = self._prompter.render_task_prompt(prompt_name=prompt_name, prompt_inputs=prompt_inputs)
        current_content = self._prompt_runner.run_initial_chapter_prompt(prompt_name=prompt_name, prompt_text=initial_prompt)
        current_content = _normalize_chapter_markdown_for_audit(current_content)
        self._store.persist_write_artifact(task=task, artifact_name=_INITIAL_WRITE_ARTIFACT_NAME, content=current_content)
        Log.info(f"写完章节: {task.title}, phase=initial", module=MODULE)
        execution_state = build_chapter_execution_state(
            task=task,
            company_name=company_name,
            current_content=current_content,
            allowed_conditional_headings=_build_allowed_conditional_headings(task),
        )
        if prompt_name == "fill_overview":
            execution_state.process_state["fix_applied"] = False
            execution_state.process_state["fix_reason"] = "overview_cover_page"
            execution_state.process_state["audit_skipped"] = True
            execution_state.process_state["confirm_skipped"] = True
            execution_state.process_state["repair_skipped"] = True
            execution_state.process_state["final_stage"] = "overview_written"
            Log.info(f"第0章跳过审计链路: {task.title}", module=MODULE)
            return ChapterResult(
                index=task.index,
                title=task.title,
                status="passed",
                content=execution_state.current_content,
                audit_passed=True,
                retry_count=0,
                failure_reason="",
                evidence_items=extract_evidence_items(execution_state.current_content),
                process_state=execution_state.process_state,
            )
        if self._write_config.fast:
            execution_state.process_state["fix_applied"] = False
            execution_state.process_state["fix_reason"] = "fast_mode_skipped"
            execution_state.process_state["audit_skipped"] = True
            execution_state.process_state["confirm_skipped"] = True
            execution_state.process_state["repair_skipped"] = True
            execution_state.process_state["final_stage"] = "fast_written"
            Log.info(f"fast 模式跳过审计链路: {task.title}", module=MODULE)
            # fast 模式下 ``audit_passed`` 强制为 ``False`` 是有意为之：
            # fast 产物不写 audit artifact（未调 ``persist_phase_audit_artifact``），
            # ``WritePipelineRunner._recover_prior_chapter_result_from_artifacts`` 仅承认
            # 落盘 audit artifact 中 ``pass=True`` 的章节作为 resume 复用来源，
            # 因此 fast 产物天然不会被 non-fast resume 误当成"已通过"。
            # 同时 ``_satisfies_audit_gate_for_current_mode`` 在当前进程为 fast 时
            # 直接放行，``audit_passed=False`` 不会影响本进程内的成功判定。
            return ChapterResult(
                index=task.index,
                title=task.title,
                status="passed",
                content=execution_state.current_content,
                audit_passed=False,
                retry_count=0,
                failure_reason="",
                evidence_items=extract_evidence_items(execution_state.current_content),
                process_state=execution_state.process_state,
            )
        execution_state = self._execute_chapter_phase_state_machine(
            execution_state=execution_state,
            company_facets=company_facets,
            company_facet_catalog=company_facet_catalog,
        )
        return self._finalize_chapter_execution(execution_state=execution_state)

    def _execute_chapter_phase_state_machine(
        self,
        *,
        execution_state: ChapterExecutionState,
        company_facets: CompanyFacetProfile | None,
        company_facet_catalog: dict[str, list[str]],
    ) -> ChapterExecutionState:
        """执行章节阶段状态机直到进入完成态。"""

        while execution_state.stage != ChapterExecutionStage.COMPLETE:
            if execution_state.stage == ChapterExecutionStage.EVALUATE_PHASE:
                self._audit_coordinator.evaluate_current_chapter_phase(
                    execution_state=execution_state,
                    company_facets=company_facets,
                    company_facet_catalog=company_facet_catalog,
                )
                if execution_state.audit_decision is None:
                    raise RuntimeError("章节阶段评估未生成审计结果")
                if (
                    execution_state.audit_decision.passed
                    or execution_state.stop_rewrite_loop
                    or execution_state.retry_count >= self._write_config.write_max_retries
                ):
                    execution_state.stage = ChapterExecutionStage.COMPLETE
                else:
                    execution_state.stage = ChapterExecutionStage.PREPARE_REWRITE
                continue

            if execution_state.stage == ChapterExecutionStage.PREPARE_REWRITE:
                self._prepare_next_chapter_rewrite(execution_state=execution_state)
                execution_state.stage = ChapterExecutionStage.APPLY_REWRITE
                continue

            if execution_state.stage == ChapterExecutionStage.APPLY_REWRITE:
                self._apply_pending_chapter_rewrite(execution_state=execution_state)
                execution_state.stage = ChapterExecutionStage.EVALUATE_PHASE
                continue

            raise RuntimeError(f"不支持的章节状态机阶段: {execution_state.stage}")

        return execution_state

    def _prepare_next_chapter_rewrite(self, *, execution_state: ChapterExecutionState) -> None:
        """准备下一轮 rewrite 阶段参数。"""

        audit_decision = require_audit_decision(execution_state, action="准备 rewrite ")
        next_retry_count = execution_state.retry_count + 1
        execution_state.pending_retry_count = next_retry_count
        execution_state.rewrite_strategy = normalize_repair_strategy(
            audit_decision.repair_contract.repair_strategy
        )
        execution_state.phase = build_rewrite_phase_name(
            strategy=execution_state.rewrite_strategy,
            retry_count=next_retry_count,
        )

    def _commit_successful_chapter_rewrite(
        self,
        *,
        execution_state: ChapterExecutionState,
        audit_decision: AuditDecision,
    ) -> None:
        """在 rewrite 成功落盘后提交重试计数与过程状态。

        Args:
            execution_state: 当前章节执行状态。
            audit_decision: 触发本轮 rewrite 的审计结果。

        Returns:
            无。

        Raises:
            RuntimeError: 缺少当前 rewrite 对应的重试序号时抛出。
        """

        applied_retry_count = resolve_rewrite_retry_count(execution_state, action="提交 rewrite ")
        execution_state.retry_count = applied_retry_count
        append_rewrite_process_state(
            execution_state.process_state,
            retry_count=applied_retry_count,
            trigger_class=audit_decision.category,
            repair_strategy=execution_state.rewrite_strategy,
        )
        execution_state.pending_retry_count = None

    def _apply_pending_chapter_rewrite(self, *, execution_state: ChapterExecutionState) -> None:
        """执行当前待处理的 rewrite 阶段。"""

        task = execution_state.task
        phase = execution_state.phase
        audit_decision = require_audit_decision(execution_state, action="执行 rewrite ")
        rewrite_retry_count = resolve_rewrite_retry_count(execution_state, action="执行 rewrite ")
        if execution_state.rewrite_strategy == RepairStrategy.REGENERATE:
            Log.info(f"开始整章重建: {task.title}, retry={rewrite_retry_count}", module=MODULE)
            Log.debug(
                _build_chapter_scene_debug_message(
                    chapter_title=task.title,
                    prompt_name="regenerate_chapter",
                    phase=phase,
                    scene_name="regenerate",
                ),
                module=MODULE,
            )
            regenerate_prompt = self._prompter.render_task_prompt(
                prompt_name="regenerate_chapter",
                prompt_inputs=self._prompter.build_regenerate_prompt_inputs(
                    task=task,
                    company_name=execution_state.company_name,
                    current_content=execution_state.current_content,
                    audit_decision=audit_decision,
                ),
            )
            execution_state.current_content = self._prompt_runner.run_regenerate_prompt(regenerate_prompt)
            execution_state.current_content = _normalize_chapter_markdown_for_audit(execution_state.current_content)
            Log.info(f"完成整章重建: {task.title}, phase={phase}", module=MODULE)
            self._store.persist_write_artifact(
                task=task,
                artifact_name=_build_phase_artifact_name(phase=phase, kind="write"),
                content=execution_state.current_content,
            )
            self._commit_successful_chapter_rewrite(
                execution_state=execution_state,
                audit_decision=audit_decision,
            )
            return

        Log.info(f"开始局部修复: {task.title}, retry={rewrite_retry_count}", module=MODULE)
        Log.debug(
            _build_chapter_scene_debug_message(
                chapter_title=task.title,
                prompt_name="repair_chapter",
                phase=phase,
                scene_name="repair",
            ),
            module=MODULE,
        )
        repair_inputs = self._prompter.build_repair_prompt_inputs(
            task=task,
            company_name=execution_state.company_name,
            current_content=execution_state.current_content,
            audit_decision=audit_decision,
        )
        self._store.persist_repair_input_artifacts(
            task=task,
            retry_count=rewrite_retry_count,
            current_content=execution_state.current_content,
            prompt_inputs=repair_inputs,
        )
        repair_prompt = self._prompter.render_task_prompt(
            prompt_name="repair_chapter",
            prompt_inputs=repair_inputs,
        )
        try:
            repair_plan, repair_raw = self._prompt_runner.run_repair_prompt(repair_prompt)
        except RepairOutputError as exc:
            self._store.persist_repair_plan_artifact(
                task=task,
                retry_count=rewrite_retry_count,
                content=exc.raw_output,
            )
            raise
        repair_apply_result = _apply_repair_plan_with_details(
            chapter_markdown=execution_state.current_content,
            repair_plan=repair_plan,
            repair_contract=audit_decision.repair_contract,
        )
        self._store.persist_repair_plan_artifact(
            task=task,
            retry_count=rewrite_retry_count,
            content=repair_raw,
        )
        self._store.persist_repair_apply_result_artifact(
            task=task,
            retry_count=rewrite_retry_count,
            apply_result=repair_apply_result,
        )
        if repair_apply_result.all_failed:
            raise ValueError(repair_apply_result.error_message)
        execution_state.current_content = _normalize_chapter_markdown_for_audit(repair_apply_result.patched_markdown)
        Log.info(f"完成局部修复: {task.title}, phase={phase}", module=MODULE)
        self._store.persist_write_artifact(
            task=task,
            artifact_name=_build_phase_artifact_name(phase=phase, kind="write"),
            content=execution_state.current_content,
        )
        self._commit_successful_chapter_rewrite(
            execution_state=execution_state,
            audit_decision=audit_decision,
        )

    def _finalize_chapter_execution(self, *, execution_state: ChapterExecutionState) -> ChapterResult:
        """基于状态机上下文构建最终章节结果并落盘最终 audit。"""

        if execution_state.audit_decision is None:
            raise RuntimeError("章节状态机结束时缺少审计结果")

        final_status = "passed" if execution_state.audit_decision.passed else "failed"
        failure_reason = "" if execution_state.audit_decision.passed else execution_state.audit_decision.category
        execution_state.process_state["final_stage"] = (
            "audit_passed" if execution_state.audit_decision.passed else "audit_failed"
        )
        execution_state.process_state["rewrite_exhausted"] = bool(
            (not execution_state.audit_decision.passed)
            and execution_state.retry_count >= self._write_config.write_max_retries
        )

        result = ChapterResult(
            index=execution_state.task.index,
            title=execution_state.task.title,
            status=final_status,
            content=execution_state.current_content,
            audit_passed=execution_state.audit_decision.passed,
            retry_count=execution_state.retry_count,
            failure_reason=failure_reason,
            evidence_items=extract_evidence_items(execution_state.current_content),
            process_state=execution_state.process_state,
        )
        self._store.persist_final_audit_artifact(
            task=execution_state.task,
            audit_decision=execution_state.audit_decision,
        )
        Log.info(
            f"章节最终完成: {execution_state.task.title}, status={final_status}, retry_count={execution_state.retry_count}",
            module=MODULE,
        )
        return result
