"""章节审计协调器。

该模块负责单章执行中的审计、证据复核与轻量证据锚点修复闭环，
将原先集中在 WritePipelineRunner 内部的审计决策链下沉为可复用协作对象。
"""

from __future__ import annotations

from typing import Any

from dayu.log import Log
from dayu.services.internal.write_pipeline.audit_evidence_rewriter import (
    _has_anchor_rewrite_candidates,
    _rewrite_evidence_lines_and_collect_resolved_anchor_issues,
    _validate_anchor_rewrite_postconditions,
)
from dayu.services.internal.write_pipeline.audit_formatting import (
    _extract_evidence_section_block,
    _normalize_chapter_markdown_for_audit,
    _replace_evidence_section_block,
    _should_run_fix_placeholders,
)
from dayu.services.internal.write_pipeline.audit_rules import (
    ConfirmOutputError,
    _collect_confirmable_evidence_violations,
    _drop_resolved_supported_anchor_violations,
    _log_chapter_audit_result,
    _log_chapter_audit_start,
    _log_chapter_confirm_result,
    _log_chapter_confirm_start,
    _merge_confirmed_evidence_results,
    _rebuild_anchor_followup_audit_decision,
    _run_programmatic_audits,
)
from dayu.services.internal.write_pipeline.artifact_store import ArtifactStore, _build_phase_artifact_name
from dayu.services.internal.write_pipeline.chapter_execution_coordinator import (
    ChapterExecutionState,
    append_anchor_rewrite_process_state,
    append_audit_process_state,
    append_confirm_process_state,
)
from dayu.services.internal.write_pipeline.enums import build_audit_scope_rules_payload, is_initial_write_phase
from dayu.services.internal.write_pipeline.company_facets import (
    filter_chapter_contract_by_facets,
    filter_item_rules_by_facets,
)
from dayu.services.internal.write_pipeline.models import (
    AuditDecision,
    ChapterTask,
    CompanyFacetProfile,
    EvidenceConfirmationResult,
    WriteRunConfig,
)
from dayu.services.internal.write_pipeline.prompt_builder import PromptBuilder
from dayu.services.internal.write_pipeline.scene_contract_preparer import SceneContractPreparer
from dayu.services.internal.write_pipeline.scene_executor import ScenePromptRunner
from dayu.services.internal.write_pipeline.source_list_builder import extract_evidence_items

MODULE = "APP.WRITE_PIPELINE"
_FIX_PLACEHOLDERS_ARTIFACT_NAME = "initial_fix_placeholders"


def _return_with_anchor_rewrite_process_state(
    *,
    current_content: str,
    suspected_decision: AuditDecision,
    audit_decision: AuditDecision,
    confirmation_result: EvidenceConfirmationResult | None,
    process_state: dict[str, Any] | None,
    phase: str,
    attempted: bool,
    applied: bool,
    skip_reason: str = "",
    failure_reason: str = "",
    resolved_violations_count: int = 0,
) -> tuple[str, AuditDecision, AuditDecision, EvidenceConfirmationResult | None]:
    """在返回锚点重写结果前写入调用方可观察的过程状态。

    Args:
        current_content: 当前章节正文。
        suspected_decision: 当前疑似审计结果。
        audit_decision: 当前最终审计结果。
        confirmation_result: 当前证据复核结果。
        process_state: 可选章节过程状态对象。
        phase: 当前阶段名。
        attempted: 是否已真正尝试生成并校验重写结果。
        applied: 是否已成功落盘重写结果。
        skip_reason: 未尝试时的跳过原因。
        failure_reason: 已尝试但失败时的失败原因。
        resolved_violations_count: 本次成功吸收的问题数量。

    Returns:
        原函数约定的四元组返回值。

    Raises:
        无。
    """

    if process_state is not None:
        append_anchor_rewrite_process_state(
            process_state,
            phase=phase,
            attempted=attempted,
            applied=applied,
            skip_reason=skip_reason,
            failure_reason=failure_reason,
            resolved_violations_count=resolved_violations_count,
        )
    return current_content, suspected_decision, audit_decision, confirmation_result


class ChapterAuditCoordinator:
    """单章审计与证据修复协调器。"""

    def __init__(
        self,
        *,
        write_config: WriteRunConfig,
        store: ArtifactStore,
        preparer: SceneContractPreparer,
        prompt_runner: ScenePromptRunner,
        prompter: PromptBuilder,
    ) -> None:
        """初始化章节审计协调器。

        Args:
            write_config: 写作运行配置。
            store: 写作产物存储器。
            preparer: scene 契约准备器。
            prompt_runner: scene prompt 执行器。
            prompter: prompt 构建器。

        Returns:
            无。

        Raises:
            无。
        """

        self._write_config = write_config
        self._store = store
        self._preparer = preparer
        self._prompt_runner = prompt_runner
        self._prompter = prompter

    def evaluate_current_chapter_phase(
        self,
        *,
        execution_state: ChapterExecutionState,
        company_facets: CompanyFacetProfile | None,
        company_facet_catalog: dict[str, list[str]],
    ) -> None:
        """评估当前章节阶段并产出审计结果。

        Args:
            execution_state: 章节执行状态机上下文。
            company_facets: 当前公司级 facet 归因结果。
            company_facet_catalog: 当前模板声明的 facet 候选目录。

        Returns:
            无。

        Raises:
            ConfirmOutputError: 当 confirm 输出非法时抛出。
            RuntimeError: 当模型调用失败时抛出。
        """

        phase = execution_state.phase
        task = execution_state.task
        _log_chapter_audit_start(chapter_title=task.title, phase=phase)

        programmatic_fail = _run_programmatic_audits(
            execution_state.current_content,
            skeleton=task.skeleton,
            allowed_conditional_headings=execution_state.allowed_conditional_headings,
        )
        if programmatic_fail is not None:
            self._store.persist_phase_audit_artifact(task=task, phase=phase, audit_decision=programmatic_fail)
            _log_chapter_audit_result(chapter_title=task.title, phase=phase, decision=programmatic_fail)
            execution_state.audit_decision = programmatic_fail
            execution_state.suspected_decision = programmatic_fail
            execution_state.confirmation_result = None
            if is_initial_write_phase(phase):
                Log.warn(
                    f"程序审计失败，跳过初始 LLM 审计直接进入重写: title={task.title!r}",
                    module=MODULE,
                )
                execution_state.process_state["fix_applied"] = False
                execution_state.process_state["fix_reason"] = "programmatic_audit_failed"
            else:
                Log.warn(
                    f"修复后程序审计仍失败，中止当前重试循环: title={task.title!r}, retry={execution_state.retry_count}",
                    module=MODULE,
                )
                execution_state.stop_rewrite_loop = True
            append_audit_process_state(
                execution_state.process_state,
                phase=phase,
                decision=programmatic_fail,
            )
            return

        if is_initial_write_phase(phase):
            self._maybe_apply_placeholder_fix(execution_state=execution_state)

        try:
            filtered_contract = filter_chapter_contract_by_facets(
                task.chapter_contract,
                company_facets,
                company_facet_catalog,
            )
            filtered_item_rules = filter_item_rules_by_facets(
                task.item_rules,
                company_facets,
                company_facet_catalog,
            )
            suspected_decision, audit_decision, confirmation_result = self.audit_and_confirm_chapter(
                chapter_markdown=execution_state.current_content,
                company_name=execution_state.company_name,
                skeleton=task.skeleton,
                chapter_contract=filtered_contract.to_prompt_fields(),
                item_rules=[rule.to_prompt_dict() for rule in filtered_item_rules],
                phase=phase,
                chapter_title=task.title,
                repair_contract=(
                    execution_state.process_state.get("latest_repair_contract")
                    if not is_initial_write_phase(phase)
                    else None
                ),
            )
        except ConfirmOutputError as exc:
            self._store.persist_phase_confirm_raw_artifact(
                task=task,
                phase=phase,
                raw_text=exc.raw_output,
            )
            self._store.persist_phase_confirm_parse_error_artifact(
                task=task,
                phase=phase,
                parse_error=exc.parse_error,
            )
            raise

        (
            execution_state.current_content,
            suspected_decision,
            audit_decision,
            confirmation_result,
        ) = self.maybe_rewrite_evidence_anchors(
            task=task,
            current_content=execution_state.current_content,
            suspected_decision=suspected_decision,
            audit_decision=audit_decision,
            confirmation_result=confirmation_result,
            phase=phase,
            skeleton=task.skeleton,
            allowed_conditional_headings=execution_state.allowed_conditional_headings,
            process_state=execution_state.process_state,
        )
        execution_state.suspected_decision = suspected_decision
        execution_state.audit_decision = audit_decision
        execution_state.confirmation_result = confirmation_result

        self._store.persist_phase_audit_suspect_artifact(
            task=task,
            phase=phase,
            audit_decision=suspected_decision,
        )
        if confirmation_result is not None:
            self._store.persist_phase_confirm_artifact(
                task=task,
                phase=phase,
                confirmation_result=confirmation_result,
            )
            append_confirm_process_state(
                execution_state.process_state,
                phase=phase,
                result=confirmation_result,
            )
        self._store.persist_phase_audit_artifact(task=task, phase=phase, audit_decision=audit_decision)
        _log_chapter_audit_result(chapter_title=task.title, phase=phase, decision=audit_decision)
        append_audit_process_state(
            execution_state.process_state,
            phase=phase,
            decision=audit_decision,
        )

    def audit_and_confirm_chapter(
        self,
        *,
        chapter_markdown: str,
        company_name: str,
        skeleton: str,
        chapter_contract: dict[str, Any],
        item_rules: list[dict[str, Any]],
        phase: str,
        chapter_title: str,
        repair_contract: dict[str, Any] | None = None,
    ) -> tuple[AuditDecision, AuditDecision, EvidenceConfirmationResult | None]:
        """执行“疑似审计 + 证据复核 + 合并”的完整审计链路。

        Args:
            chapter_markdown: 章节正文。
            company_name: 公司名称。
            skeleton: 章节骨架。
            chapter_contract: 章节合同。
            item_rules: 当前章节允许的条件写作规则。
            phase: 阶段名。
            chapter_title: 章节标题。
            repair_contract: 修复后局部复审时使用的修复合同。

        Returns:
            `(suspected_audit_decision, final_audit_decision, confirmation_result)` 三元组。

        Raises:
            RuntimeError: 当审计或证据复核失败时抛出。
        """

        audit_mode = "初始整章审计" if is_initial_write_phase(phase) else "修复后局部复审"
        suspected_decision = self._audit_chapter(
            chapter_markdown,
            company_name,
            skeleton=skeleton,
            chapter_contract=chapter_contract,
            item_rules=item_rules,
            audit_mode=audit_mode,
            repair_contract=repair_contract,
        )
        confirmable_violations = _collect_confirmable_evidence_violations(suspected_decision.violations)
        if confirmable_violations:
            _log_chapter_confirm_start(
                chapter_title=chapter_title,
                phase=phase,
                count=len(confirmable_violations),
            )
        confirmation_result = self._confirm_evidence_violations(
            chapter_markdown=chapter_markdown,
            company_name=company_name,
            audit_decision=suspected_decision,
        )
        if confirmation_result is None:
            return suspected_decision, suspected_decision, None
        _log_chapter_confirm_result(
            chapter_title=chapter_title,
            phase=phase,
            result=confirmation_result,
        )
        return (
            suspected_decision,
            _merge_confirmed_evidence_results(
                audit_decision=suspected_decision,
                confirmation_result=confirmation_result,
            ),
            confirmation_result,
        )

    def maybe_rewrite_evidence_anchors(
        self,
        *,
        task: ChapterTask,
        current_content: str,
        suspected_decision: AuditDecision,
        audit_decision: AuditDecision,
        confirmation_result: EvidenceConfirmationResult | None,
        phase: str,
        skeleton: str,
        allowed_conditional_headings: set[str] | None = None,
        process_state: dict[str, Any] | None = None,
    ) -> tuple[str, AuditDecision, AuditDecision, EvidenceConfirmationResult | None]:
        """在 confirm 后按需执行仅针对证据锚点的轻量修复。

        Args:
            task: 当前章节任务。
            current_content: 当前章节正文。
            suspected_decision: 当前疑似审计结果。
            audit_decision: 当前最终审计结果。
            confirmation_result: 当前 confirm 结果。
            phase: 当前阶段名。
            skeleton: 当前章节骨架。
            allowed_conditional_headings: 允许的条件型可见标题集合。
            process_state: 可选章节过程状态，用于显式暴露 rewrite 尝试结果。

        Returns:
            `(current_content, suspected_decision, audit_decision, confirmation_result)`。

        Raises:
            无。
        """

        if confirmation_result is None:
            return _return_with_anchor_rewrite_process_state(
                current_content=current_content,
                suspected_decision=suspected_decision,
                audit_decision=audit_decision,
                confirmation_result=confirmation_result,
                process_state=process_state,
                phase=phase,
                attempted=False,
                applied=False,
                skip_reason="no_confirmation_result",
            )
        if not _has_anchor_rewrite_candidates(confirmation_result):
            return _return_with_anchor_rewrite_process_state(
                current_content=current_content,
                suspected_decision=suspected_decision,
                audit_decision=audit_decision,
                confirmation_result=confirmation_result,
                process_state=process_state,
                phase=phase,
                attempted=False,
                applied=False,
                skip_reason="no_anchor_rewrite_candidates",
            )

        # 一旦确认存在 anchor rewrite 候选，先把 confirm merge 阶段因
        # ``entry.anchor_fix is not None`` 静默丢弃的 supported_* 违规重建为 S7
        # followup 违规，作为后续所有“未真正落地 rewrite”早退路径的基线 audit_decision。
        # 仅当后续 rewrite 真正成功并通过后验校验时，再交由
        # ``_drop_resolved_supported_anchor_violations`` 把已被吸收的 S7 followup 移除。
        audit_decision = _rebuild_anchor_followup_audit_decision(
            audit_decision=audit_decision,
            confirmation_result=confirmation_result,
        )

        rewritten_evidence_lines, resolved_violation_ids = _rewrite_evidence_lines_and_collect_resolved_anchor_issues(
            chapter_markdown=current_content,
            confirmation_result=confirmation_result,
        )
        if not rewritten_evidence_lines:
            return _return_with_anchor_rewrite_process_state(
                current_content=current_content,
                suspected_decision=suspected_decision,
                audit_decision=audit_decision,
                confirmation_result=confirmation_result,
                process_state=process_state,
                phase=phase,
                attempted=False,
                applied=False,
                skip_reason="no_rewritten_evidence_lines",
            )

        evidence_section = _extract_evidence_section_block(current_content)
        if not evidence_section:
            return _return_with_anchor_rewrite_process_state(
                current_content=current_content,
                suspected_decision=suspected_decision,
                audit_decision=audit_decision,
                confirmation_result=confirmation_result,
                process_state=process_state,
                phase=phase,
                attempted=False,
                applied=False,
                skip_reason="missing_evidence_section",
            )

        current_evidence_lines = extract_evidence_items(current_content)
        if rewritten_evidence_lines == current_evidence_lines:
            return _return_with_anchor_rewrite_process_state(
                current_content=current_content,
                suspected_decision=suspected_decision,
                audit_decision=audit_decision,
                confirmation_result=confirmation_result,
                process_state=process_state,
                phase=phase,
                attempted=False,
                applied=False,
                skip_reason="no_material_change",
            )

        updated_section_lines = ["### 证据与出处", ""]
        updated_section_lines.extend(f"- {line}" for line in rewritten_evidence_lines)
        rewritten_content = _replace_evidence_section_block(
            chapter_markdown=current_content,
            evidence_section="\n".join(updated_section_lines),
        )
        rewritten_content = _normalize_chapter_markdown_for_audit(rewritten_content)
        if rewritten_content == current_content:
            return _return_with_anchor_rewrite_process_state(
                current_content=current_content,
                suspected_decision=suspected_decision,
                audit_decision=audit_decision,
                confirmation_result=confirmation_result,
                process_state=process_state,
                phase=phase,
                attempted=False,
                applied=False,
                skip_reason="no_content_change_after_rewrite",
            )

        Log.info(
            f"执行证据锚点轻量修复: title={task.title!r}, phase={phase}, lines={len(rewritten_evidence_lines)}",
            module=MODULE,
        )
        validation_error = _validate_anchor_rewrite_postconditions(
            original_chapter_markdown=current_content,
            rewritten_chapter_markdown=rewritten_content,
            expected_evidence_lines=rewritten_evidence_lines,
            skeleton=skeleton,
            allowed_conditional_headings=allowed_conditional_headings,
        )
        if validation_error:
            Log.warn(
                f"证据锚点轻量修复后验校验失败，回退原文: title={task.title!r}, phase={phase}, reason={validation_error}",
                module=MODULE,
            )
            return _return_with_anchor_rewrite_process_state(
                current_content=current_content,
                suspected_decision=suspected_decision,
                audit_decision=audit_decision,
                confirmation_result=confirmation_result,
                process_state=process_state,
                phase=phase,
                attempted=True,
                applied=False,
                failure_reason=validation_error,
            )
        self._store.persist_write_artifact(
            task=task,
            artifact_name=_build_phase_artifact_name(phase=phase, kind="write"),
            content=rewritten_content,
        )
        refreshed_audit_decision = _drop_resolved_supported_anchor_violations(
            audit_decision=audit_decision,
            confirmation_result=confirmation_result,
            resolved_violation_ids=resolved_violation_ids,
        )
        return _return_with_anchor_rewrite_process_state(
            current_content=rewritten_content,
            suspected_decision=suspected_decision,
            audit_decision=refreshed_audit_decision,
            confirmation_result=confirmation_result,
            process_state=process_state,
            phase=phase,
            attempted=True,
            applied=True,
            resolved_violations_count=len(resolved_violation_ids),
        )

    def _maybe_apply_placeholder_fix(self, *, execution_state: ChapterExecutionState) -> None:
        """在初始阶段按需执行占位符补强。"""

        if not _should_run_fix_placeholders(execution_state.current_content):
            execution_state.process_state["fix_applied"] = False
            execution_state.process_state["fix_reason"] = "no_placeholder_detected"
            return

        fix_prompt = self._prompter.render_task_prompt(
            prompt_name="fix_placeholders",
            prompt_inputs={
                "chapter": execution_state.task.title,
                "company": execution_state.company_name,
                "ticker": self._write_config.ticker,
                "chapter_markdown": execution_state.current_content,
                "fix_mode": "placeholder_only",
                "rewrite_compliant_content": False,
            },
        )
        execution_state.current_content = self._prompt_runner.run_fix_prompt(fix_prompt)
        execution_state.current_content = _normalize_chapter_markdown_for_audit(execution_state.current_content)
        self._store.persist_write_artifact(
            task=execution_state.task,
            artifact_name=_FIX_PLACEHOLDERS_ARTIFACT_NAME,
            content=execution_state.current_content,
        )
        execution_state.process_state["fix_applied"] = True
        execution_state.process_state["fix_reason"] = "placeholder_detected"

    def _audit_chapter(
        self,
        chapter_markdown: str,
        company_name: str = "",
        *,
        skeleton: str = "",
        chapter_contract: dict[str, Any] | None = None,
        item_rules: list[dict[str, Any]] | None = None,
        audit_mode: str = "初始整章审计",
        repair_contract: dict[str, Any] | None = None,
    ) -> AuditDecision:
        """调用审计 Agent 获取结构化审计结果。

        重试与 replay 兜底由 ``ScenePromptRunner.run_audit_prompt`` 统一负责，
        本方法仅负责 prompt 渲染与转发，避免 helper 之外再保留一份手写 retry。
        """

        audit_prompt = self._prompter.render_task_prompt(
            prompt_name="audit_facts_tone_json",
            prompt_inputs={
                "company": company_name,
                "ticker": self._write_config.ticker,
                "audit_mode": audit_mode,
                "chapter_markdown": chapter_markdown,
                "skeleton": skeleton,
                "chapter_contract": chapter_contract or {},
                "item_rules": item_rules or [],
                "audit_scope_rules": build_audit_scope_rules_payload(),
                "repair_contract": repair_contract or {},
            },
        )
        return self._prompt_runner.run_audit_prompt(audit_prompt)

    def _confirm_evidence_violations(
        self,
        *,
        chapter_markdown: str,
        company_name: str,
        audit_decision: AuditDecision,
    ) -> EvidenceConfirmationResult | None:
        """仅复核疑似 `E1/E2` 违规是否属实。"""

        confirmable_violations = _collect_confirmable_evidence_violations(audit_decision.violations)
        if not confirmable_violations:
            return None
        confirm_prompt = self._prompter.render_task_prompt(
            prompt_name="confirm_evidence_violations",
            prompt_inputs=self._prompter.build_confirm_prompt_inputs(
                company_name=company_name,
                chapter_markdown=chapter_markdown,
                suspected_evidence_violations=confirmable_violations,
            ),
        )
        return self._prompt_runner.run_confirm_prompt(confirm_prompt)