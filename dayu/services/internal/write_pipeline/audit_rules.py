"""审计决策、解析与证据复核纯函数模块。

本模块是审计规则与决策逻辑的真源，包含：
- 审计输出 JSON 解析与违规标准化
- 程序审计（结构/内容/证据三类检查）
- 修复合同推导
- 证据复核结果解析与合并
- 审计异常类与修复数据类
- 审计日志辅助函数
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Optional

from dayu.log import Log
from dayu.services.internal.write_pipeline.audit_formatting import (
    _extract_json_text,
    _extract_markdown_content,
    _has_evidence_section,
    _matches_skeleton_structure,
)
from dayu.services.internal.write_pipeline.enums import (
    AuditCategory,
    AuditRuleCode,
    BLOCKING_CONTENT_AUDIT_RULE_CODES,
    BLOCKING_EVIDENCE_AUDIT_RULE_CODES,
    CONFIRMABLE_EVIDENCE_AUDIT_RULE_CODES,
    EvidenceConfirmationStatus,
    LOW_PRIORITY_AUDIT_RULE_CODES,
    REGENERATE_EVIDENCE_AUDIT_RULE_CODES,
    RepairResolutionMode,
    RepairStrategy,
    RepairTargetKind,
    STRUCTURAL_REPAIR_AUDIT_RULE_CODES,
    normalize_audit_category,
    normalize_audit_rule_code,
    normalize_evidence_confirmation_status,
    normalize_repair_resolution_mode,
    normalize_repair_strategy,
)
from dayu.services.internal.write_pipeline.models import (
    AuditDecision,
    EvidenceAnchorFix,
    EvidenceConfirmationEntry,
    EvidenceConfirmationResult,
    MissingEvidenceSlot,
    OffendingClaimSpan,
    RemediationAction,
    RepairContract,
    Violation,
)


MODULE = "APP.WRITE_PIPELINE"


_CONTENT_MIN_CHARS = 10


_REPAIR_CONTRACT_VERSION = "repair_contract_v1"


class RepairOutputError(RuntimeError):
    """repair 输出异常。

    Attributes:
        raw_output: repair Agent 返回的原始文本，便于失败时落盘留痕。
    """

    def __init__(self, message: str, *, raw_output: str) -> None:
        """初始化 repair 输出异常。

        Args:
            message: 异常描述。
            raw_output: repair Agent 返回的原始文本。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(message)
        self.raw_output = raw_output


class EmptyOutputError(RuntimeError):
    """场景输出为空或脏数据（抽不到代码块、长度过短等）。

    专门用于 markdown 类 scene（write/overview/decision/regenerate/fix）
    的输出校验：当模型只返回工具调用而无文本、或文本未包裹在合法 ```markdown```
    代码块且长度过短时，由 ``parse_markdown_scene_output`` 抛出 ``ValueError``，
    helper 再以本异常包裹，向上层传达「需要 replay 兜底」的语义。

    Attributes:
        raw_output: 模型返回的原始文本，便于失败时落盘留痕。
    """

    def __init__(self, message: str, *, raw_output: str) -> None:
        """初始化空输出异常。

        Args:
            message: 异常描述。
            raw_output: 模型返回的原始文本。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(message)
        self.raw_output = raw_output


class ConfirmOutputError(RuntimeError):
    """confirm 输出异常。

    Attributes:
        raw_output: confirm Agent 返回的原始文本。
        parse_error: 解析异常说明。
    """

    def __init__(self, message: str, *, raw_output: str, parse_error: str) -> None:
        """初始化 confirm 输出异常。

        Args:
            message: 异常描述。
            raw_output: confirm Agent 返回的原始文本。
            parse_error: 解析异常说明。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(message)
        self.raw_output = raw_output
        self.parse_error = parse_error


@dataclass(slots=True)
class RepairPatchApplyRecord:
    """单个 repair patch 的应用结果。

    Attributes:
        patch_index: patch 在 repair plan 中的 1-based 序号。
        target_excerpt: patch 原始目标片段。
        target_kind: patch 目标匹配类型。
        target_section_heading: patch 指定的 section 标题。
        occurrence_index: patch 指定的命中序号。
        matched_count: 当前 patch 在允许作用域内的命中次数。
        status: 应用状态，取值为 ``applied`` 或 ``skipped``。
        skip_reason: 跳过原因；成功应用时为空字符串。
    """

    patch_index: int
    target_excerpt: str
    target_kind: str
    target_section_heading: str
    occurrence_index: int | None
    matched_count: int
    status: str
    skip_reason: str

    def to_dict(self) -> dict[str, Any]:
        """导出为可 JSON 序列化的字典。

        Args:
            无。

        Returns:
            当前 patch 应用结果的字典表示。

        Raises:
            无。
        """

        return {
            "patch_index": self.patch_index,
            "target_excerpt": self.target_excerpt,
            "target_kind": self.target_kind,
            "target_section_heading": self.target_section_heading,
            "occurrence_index": self.occurrence_index,
            "matched_count": self.matched_count,
            "status": self.status,
            "skip_reason": self.skip_reason,
        }


@dataclass(slots=True)
class RepairPlanApplyResult:
    """repair plan 整体应用结果。

    Attributes:
        patched_markdown: 应用 patch 后得到的章节正文。
        total_patches: repair plan 中 patch 总数。
        applied_count: 成功应用的 patch 数量。
        skipped_count: 被跳过的 patch 数量。
        all_failed: 是否全部 patch 均失败。
        error_message: 当全部 patch 失败时的聚合错误消息，否则为空字符串。
        patch_results: 各 patch 的逐条应用结果。
    """

    patched_markdown: str
    total_patches: int
    applied_count: int
    skipped_count: int
    all_failed: bool
    error_message: str
    patch_results: list[RepairPatchApplyRecord]

    def to_dict(self) -> dict[str, Any]:
        """导出为可 JSON 序列化的字典。

        Args:
            无。

        Returns:
            当前 repair plan 应用结果的字典表示。

        Raises:
            无。
        """

        return {
            "patched_markdown": self.patched_markdown,
            "total_patches": self.total_patches,
            "applied_count": self.applied_count,
            "skipped_count": self.skipped_count,
            "all_failed": self.all_failed,
            "error_message": self.error_message,
            "patch_results": [item.to_dict() for item in self.patch_results],
        }


def _derive_repair_resolution_mode(
    *,
    category: AuditCategory,
    violation: Violation,
) -> RepairResolutionMode:
    """根据违规信息推导单条处置模式。

    Args:
        category: 当前审计分类。
        violation: 单条审计违规。

    Returns:
        单条违规的稳定处置模式。

    Raises:
        无。
    """

    raw_mode = violation.resolution_mode.strip()
    if raw_mode:
        return normalize_repair_resolution_mode(raw_mode)

    if category == AuditCategory.EVIDENCE_INSUFFICIENT:
        return RepairResolutionMode.REWRITE_WITH_EXISTING_EVIDENCE
    return RepairResolutionMode.REWRITE_WITH_EXISTING_EVIDENCE


def _derive_target_kind_hint(
    *,
    resolution_mode: RepairResolutionMode,
    excerpt: str,
) -> str:
    """为单条违规生成建议的 patch 粒度。

    Args:
        resolution_mode: 处置模式。
        excerpt: 命中的违规原文。

    Returns:
        建议的 target_kind 字符串。

    Raises:
        无。
    """

    if resolution_mode == RepairResolutionMode.DELETE_CLAIM:
        if excerpt.lstrip().startswith("- "):
            return RepairTargetKind.BULLET.value
        return RepairTargetKind.PARAGRAPH.value
    if "\n" in excerpt:
        return RepairTargetKind.LINE.value
    return RepairTargetKind.SUBSTRING.value


def _build_remediation_action(
    *,
    slot_id: str,
    category: AuditCategory,
    violation: Violation,
) -> RemediationAction | None:
    """把单条违规收口为稳定的修复动作。

    Args:
        slot_id: 当前违规对应的稳定槽位 ID。
        category: 当前审计分类。
        violation: 单条审计违规。

    Returns:
        修复动作；若该违规没有稳定 excerpt，则返回 ``None``。

    Raises:
        无。
    """

    excerpt = violation.excerpt.strip()
    if not excerpt:
        return None
    resolution_mode = _derive_repair_resolution_mode(category=category, violation=violation)
    return RemediationAction(
        action_id=slot_id,
        rule=violation.rule,
        excerpt=excerpt,
        reason=violation.reason,
        rewrite_hint=violation.rewrite_hint,
        confirmation_status=violation.confirmation_status.strip(),
        resolution_mode=resolution_mode.value,
        target_kind_hint=_derive_target_kind_hint(
            resolution_mode=resolution_mode,
            excerpt=excerpt,
        ),
    )


def _log_chapter_audit_start(*, chapter_title: str, phase: str) -> None:
    """输出章节审计开始日志。

    Args:
        chapter_title: 章节标题。
        phase: 审计阶段名（如 initial、repair_1、regenerate_1）。

    Returns:
        无。

    Raises:
        无。
    """

    Log.info(f"开始审计章节: {chapter_title}, phase={phase}", module=MODULE)


def _log_chapter_audit_result(*, chapter_title: str, phase: str, decision: AuditDecision) -> None:
    """输出章节审计结果日志。

    Args:
        chapter_title: 章节标题。
        phase: 审计阶段名（如 initial、repair_1、regenerate_1）。
        decision: 审计结果对象。

    Returns:
        无。

    Raises:
        无。
    """

    if decision.passed:
        Log.info(f"审计成功: {chapter_title}, phase={phase}", module=MODULE)
        return
    Log.warn(
        f"审计失败: {chapter_title}, phase={phase}, class={decision.category}",
        module=MODULE,
    )


def _log_chapter_confirm_start(*, chapter_title: str, phase: str, count: int) -> None:
    """输出章节证据复核开始日志。

    Args:
        chapter_title: 章节标题。
        phase: 所属阶段名（如 initial、repair_1、regenerate_1）。
        count: 待复核的疑似 E 类违规数量。

    Returns:
        无。

    Raises:
        无。
    """

    Log.info(
        f"开始证据复核: {chapter_title}, phase={phase}, suspected_evidence_count={count}",
        module=MODULE,
    )


def _log_chapter_confirm_result(
    *,
    chapter_title: str,
    phase: str,
    result: "EvidenceConfirmationResult",
) -> None:
    """输出章节证据复核结果日志。

    Args:
        chapter_title: 章节标题。
        phase: 所属阶段名（如 initial、repair_1、regenerate_1）。
        result: 证据复核结果。

    Returns:
        无。

    Raises:
        无。
    """

    status_counts: dict[str, int] = {}
    for entry in result.entries:
        status_counts[entry.status] = status_counts.get(entry.status, 0) + 1
    Log.info(
        f"完成证据复核: {chapter_title}, phase={phase}, status_counts={status_counts}",
        module=MODULE,
    )


def _derive_repair_contract(
    *,
    category: AuditCategory | str,
    violations: list[Violation],
    notes: list[str],
) -> RepairContract:
    """从审计结果推导结构化修复合同。

    Args:
        category: 审计分类。
        violations: 违规列表。
        notes: 审计备注。

    Returns:
        标准化后的修复合同。

    Raises:
        无。
    """

    normalized_category = normalize_audit_category(category)
    missing_evidence_slots: list[MissingEvidenceSlot] = []
    offending_claim_spans: list[OffendingClaimSpan] = []
    remediation_actions: list[RemediationAction] = []

    for index, violation in enumerate(violations, start=1):
        rule = violation.rule
        reason = violation.reason
        rewrite_hint = violation.rewrite_hint
        excerpt = violation.excerpt.strip()
        slot_id = f"slot_{index}_{str(rule).replace('.', '_')}"

        if normalized_category == AuditCategory.EVIDENCE_INSUFFICIENT:
            missing_evidence_slots.append(
                MissingEvidenceSlot(
                    slot_id=slot_id,
                    rule=rule,
                    description=reason,
                    required_evidence=rewrite_hint,
                    severity=violation.severity or "high",
                )
            )

        if excerpt:
            offending_claim_spans.append(
                OffendingClaimSpan(
                    rule=rule,
                    excerpt=excerpt,
                    reason=reason,
                )
            )
        remediation_action = _build_remediation_action(
            slot_id=slot_id,
            category=normalized_category,
            violation=violation,
        )
        if remediation_action is not None:
            remediation_actions.append(remediation_action)

    repair_strategy = _derive_repair_strategy(violations)

    if normalized_category == AuditCategory.EVIDENCE_INSUFFICIENT and repair_strategy == RepairStrategy.REGENERATE:
        preferred_tool_action = "regenerate_chapter"
        retry_scope = "chapter_regenerate"
    elif normalized_category == AuditCategory.EVIDENCE_INSUFFICIENT:
        preferred_tool_action = "repair_chapter"
        retry_scope = "targeted_evidence_patch"
    elif normalized_category == AuditCategory.CONTENT_VIOLATION and repair_strategy == RepairStrategy.REGENERATE:
        preferred_tool_action = "regenerate_chapter"
        retry_scope = "chapter_regenerate"
    elif normalized_category == AuditCategory.CONTENT_VIOLATION:
        preferred_tool_action = "repair_chapter"
        retry_scope = "targeted_content_patch"
    elif normalized_category == AuditCategory.STYLE_VIOLATION:
        preferred_tool_action = "repair_chapter"
        retry_scope = "targeted_style_patch"
    elif normalized_category == AuditCategory.OK:
        preferred_tool_action = "none"
        retry_scope = "none"
        repair_strategy = RepairStrategy.NONE
    else:
        preferred_tool_action = "repair_chapter"
        retry_scope = "targeted_patch"

    return RepairContract(
        contract_version=_REPAIR_CONTRACT_VERSION,
        missing_evidence_slots=missing_evidence_slots,
        offending_claim_spans=offending_claim_spans,
        remediation_actions=remediation_actions,
        preferred_tool_action=preferred_tool_action,
        repair_strategy=repair_strategy.value,
        retry_scope=retry_scope,
        notes=list(notes),
    )


def _normalize_repair_contract(
    contract: Optional[dict[str, Any]],
    *,
    category: AuditCategory | str,
    violations: list[Violation],
    notes: list[str],
) -> RepairContract:
    """规范化修复合同并补齐缺省字段。

    ``contract`` 来自 LLM JSON 原始输出，可能为 ``None`` 或不完整的字典；
    函数先从 violations/notes 推导出完整的参考合同，再将 LLM 输出字段优先合入。

    Args:
        contract: LLM 返回的原始修复合同字典。
        category: 审计分类。
        violations: 违规列表。
        notes: 审计备注。

    Returns:
        类型安全的修复合同。

    Raises:
        无。
    """

    derived = _derive_repair_contract(category=category, violations=violations, notes=notes)
    if not isinstance(contract, dict):
        return derived

    # 解析 LLM 返回的子列表，无效时退回 derived 值
    raw_missing = contract.get("missing_evidence_slots")
    missing_evidence_slots = (
        [_parse_raw_missing_evidence_slot(s) for s in raw_missing if isinstance(s, dict)]
        if isinstance(raw_missing, list) and raw_missing
        else derived.missing_evidence_slots
    )
    raw_offending = contract.get("offending_claim_spans")
    offending_claim_spans = (
        [_parse_raw_offending_claim_span(s) for s in raw_offending if isinstance(s, dict)]
        if isinstance(raw_offending, list) and raw_offending
        else derived.offending_claim_spans
    )
    raw_actions = contract.get("remediation_actions")
    remediation_actions = (
        [_parse_raw_remediation_action(a) for a in raw_actions if isinstance(a, dict)]
        if isinstance(raw_actions, list) and raw_actions
        else derived.remediation_actions
    )

    preferred_tool_action = str(contract.get("preferred_tool_action") or derived.preferred_tool_action)
    repair_strategy = normalize_repair_strategy(
        contract.get("repair_strategy") or derived.repair_strategy
    ).value
    retry_scope = str(contract.get("retry_scope") or derived.retry_scope)
    contract_version = str(contract.get("contract_version") or _REPAIR_CONTRACT_VERSION)
    raw_notes = contract.get("notes")
    notes_list = list(raw_notes) if isinstance(raw_notes, list) and raw_notes else list(notes)

    # 若 derived 侧判定需要整章重写，则强制覆盖 LLM 策略
    if derived.repair_strategy == RepairStrategy.REGENERATE.value:
        preferred_tool_action = derived.preferred_tool_action
        repair_strategy = derived.repair_strategy
        retry_scope = derived.retry_scope

    return RepairContract(
        contract_version=contract_version,
        missing_evidence_slots=missing_evidence_slots,
        offending_claim_spans=offending_claim_spans,
        remediation_actions=remediation_actions,
        preferred_tool_action=preferred_tool_action,
        repair_strategy=repair_strategy,
        retry_scope=retry_scope,
        notes=notes_list,
    )


def _parse_raw_missing_evidence_slot(raw: dict[str, Any]) -> MissingEvidenceSlot:
    """从 LLM 原始字典解析缺失证据槽位。

    Args:
        raw: LLM JSON 中单个 missing_evidence_slot 对象。

    Returns:
        类型安全的 ``MissingEvidenceSlot``。

    Raises:
        无。
    """

    return MissingEvidenceSlot(
        slot_id=str(raw.get("slot_id", "")),
        rule=normalize_audit_rule_code(raw.get("rule")),
        description=str(raw.get("description", "")),
        required_evidence=str(raw.get("required_evidence", "")),
        severity=str(raw.get("severity", "")),
    )


def _parse_raw_offending_claim_span(raw: dict[str, Any]) -> OffendingClaimSpan:
    """从 LLM 原始字典解析违规断言片段。

    Args:
        raw: LLM JSON 中单个 offending_claim_span 对象。

    Returns:
        类型安全的 ``OffendingClaimSpan``。

    Raises:
        无。
    """

    return OffendingClaimSpan(
        rule=normalize_audit_rule_code(raw.get("rule")),
        excerpt=str(raw.get("excerpt", "")),
        reason=str(raw.get("reason", "")),
    )


def _parse_raw_remediation_action(raw: dict[str, Any]) -> RemediationAction:
    """从 LLM 原始字典解析修复动作。

    Args:
        raw: LLM JSON 中单个 remediation_action 对象。

    Returns:
        类型安全的 ``RemediationAction``。

    Raises:
        无。
    """

    return RemediationAction(
        action_id=str(raw.get("action_id", "")),
        rule=normalize_audit_rule_code(raw.get("rule")),
        excerpt=str(raw.get("excerpt", "")),
        reason=str(raw.get("reason", "")),
        rewrite_hint=str(raw.get("rewrite_hint", "")),
        confirmation_status=str(raw.get("confirmation_status", "")),
        resolution_mode=str(raw.get("resolution_mode", "")),
        target_kind_hint=str(raw.get("target_kind_hint", "")),
    )


def _derive_repair_strategy(violations: list[Violation]) -> RepairStrategy:
    """根据违规集合推导修复策略。

    Args:
        violations: 审计违规列表。

    Returns:
        `patch` 表示局部修补；`regenerate` 表示整章重建。

    Raises:
        无。
    """

    rule_set = {item.rule for item in violations}
    if rule_set & (STRUCTURAL_REPAIR_AUDIT_RULE_CODES | REGENERATE_EVIDENCE_AUDIT_RULE_CODES):
        return RepairStrategy.REGENERATE
    return RepairStrategy.PATCH


def _normalize_audit_violations(violations: list[dict[str, Any]]) -> list[Violation]:
    """标准化审计违规列表。

    将 LLM 返回的原始字典列表解析为类型安全的 ``Violation`` 列表，
    并根据审计策略调整 severity。

    Args:
        violations: LLM 返回的原始违规字典列表。

    Returns:
        标准化后的违规列表。

    Raises:
        无。
    """

    normalized: list[Violation] = []
    for raw_violation in violations:
        if not isinstance(raw_violation, dict):
            continue
        rule = normalize_audit_rule_code(raw_violation.get("rule"))
        severity = str(raw_violation.get("severity", ""))
        if rule in LOW_PRIORITY_AUDIT_RULE_CODES:
            severity = "low"
        normalized.append(Violation(
            rule=rule,
            severity=severity,
            excerpt=str(raw_violation.get("excerpt", "")),
            reason=str(raw_violation.get("reason", "")),
            rewrite_hint=str(raw_violation.get("rewrite_hint", "")),
            confirmation_status=str(raw_violation.get("confirmation_status", "")),
            resolution_mode=str(raw_violation.get("resolution_mode", "")),
        ))
    return normalized


def _recompute_audit_result(*, violations: list[Violation]) -> tuple[bool, AuditCategory]:
    """根据当前审计策略重新计算通过状态与分类。

    Args:
        violations: 标准化后的违规列表。

    Returns:
        `(passed, category)` 二元组。

    Raises:
        无。
    """

    rules = [item.rule for item in violations]
    severities = [item.severity.strip().lower() for item in violations]

    if any(rule in BLOCKING_EVIDENCE_AUDIT_RULE_CODES for rule in rules):
        return False, AuditCategory.EVIDENCE_INSUFFICIENT
    if any(rule in BLOCKING_CONTENT_AUDIT_RULE_CODES for rule in rules):
        return False, AuditCategory.CONTENT_VIOLATION

    effective_style_pairs = [
        (rule, severity)
        for rule, severity in zip(rules, severities, strict=False)
        if rule not in LOW_PRIORITY_AUDIT_RULE_CODES
    ]
    if any(severity == "high" for _rule, severity in effective_style_pairs):
        return False, AuditCategory.STYLE_VIOLATION
    if sum(1 for _rule, severity in effective_style_pairs if severity == "medium") >= 2:
        return False, AuditCategory.STYLE_VIOLATION
    if sum(1 for rule, _severity in effective_style_pairs if rule == AuditRuleCode.S3) >= 2:
        return False, AuditCategory.STYLE_VIOLATION
    return True, AuditCategory.OK


def _build_research_decision_audit_summary(
    audit_payload: dict[str, Any] | None,
    *,
    audit_passed: bool,
) -> str:
    """构建第10章使用的审计状态摘要。

    Args:
        audit_payload: 最终 audit JSON 负载。
        audit_passed: 章节是否通过最终审计。

    Returns:
        审计状态摘要 Markdown；若无可用信息则返回空字符串。

    Raises:
        无。
    """

    status_text = "通过" if audit_passed else "未通过"
    category = ""
    violations: list[dict[str, Any]] = []
    if isinstance(audit_payload, dict):
        category = str(audit_payload.get("class") or "")
        raw_violations = audit_payload.get("violations")
        if isinstance(raw_violations, list):
            violations = [item for item in raw_violations if isinstance(item, dict)]

    blocks: list[str] = ["### 审计状态摘要", f"- 最终审计：{status_text}"]
    if category:
        blocks.append(f"- 最终类别：{category}")

    unresolved_lines = _extract_high_priority_violation_lines(violations)
    if unresolved_lines:
        blocks.append("")
        blocks.append("### 未解决的高优先级问题")
        blocks.extend(unresolved_lines)
    return "\n".join(blocks).strip()


def _extract_high_priority_violation_lines(violations: list[dict[str, Any]]) -> list[str]:
    """提取中高优先级违规的简要列表。

    Args:
        violations: audit 违规项列表。

    Returns:
        适合写入决策输入的 bullet 行列表。

    Raises:
        无。
    """

    lines: list[str] = []
    for violation in violations:
        severity = str(violation.get("severity") or "").lower()
        if severity not in {"medium", "high"}:
            continue
        rule = str(violation.get("rule") or "").strip()
        excerpt = str(violation.get("excerpt") or "").strip()
        reason = str(violation.get("reason") or "").strip()
        line = f"- [{rule or 'UNKNOWN'}]"
        if excerpt:
            line += f" {excerpt}"
        if reason:
            line += f"：{reason}"
        lines.append(line)
        if len(lines) >= 3:
            break
    return lines


def _build_confirm_artifact_payload(result: EvidenceConfirmationResult) -> dict[str, Any]:
    """构建证据复核产物 JSON 负载。

    Args:
        result: 证据复核结果。

    Returns:
        可直接序列化为 JSON 的字典。

    Raises:
        无。
    """

    return {
        "results": [asdict(entry) for entry in result.entries],
        "notes": list(result.notes),
        "raw": result.raw,
    }



def _run_programmatic_audits(
    content: str,
    *,
    skeleton: str = "",
    allowed_conditional_headings: set[str] | None = None,
) -> AuditDecision | None:
    """按顺序执行所有程序审计，首个失败即短路返回对应决策。

    Args:
        content: 当前章节正文。
        skeleton: 当前章节骨架。
        allowed_conditional_headings: 允许出现的条件型标题集合。

    Returns:
        首个失败的审计决策；若全部通过则返回 ``None``。

    Raises:
        无。
    """

    if not _matches_skeleton_structure(
        content,
        skeleton,
        allowed_conditional_headings=allowed_conditional_headings,
    ):
        reason = "章节内容与骨架结构不匹配"
        violations = [Violation(
            rule=AuditRuleCode.P1,
            severity="high",
            reason=reason,
            rewrite_hint="重新写作，必须严格按骨架标题顺序输出；额外可见标题仅允许来自满足 ITEM_RULE 的条件小节。",
        )]
        notes = [reason]
        return AuditDecision(
            passed=False,
            category=AuditCategory.CONTENT_VIOLATION,
            violations=violations,
            notes=notes,
            repair_contract=_derive_repair_contract(
                category=AuditCategory.CONTENT_VIOLATION,
                violations=violations,
                notes=notes,
            ),
            raw=f'{{"pass": false, "class": "content_violation", "notes": ["{reason}"]}}',
        )

    if len(content.strip()) < _CONTENT_MIN_CHARS:
        reason = f"内容过短：仅 {len(content)} 字符，未达到最小要求 {_CONTENT_MIN_CHARS} 字符"
        violations = [Violation(
            rule=AuditRuleCode.P2,
            severity="high",
            excerpt=content[:80],
            reason=reason,
            rewrite_hint="重新写作，内容必须包含结论要点、详细情况与证据出处",
        )]
        notes = [reason]
        return AuditDecision(
            passed=False,
            category=AuditCategory.CONTENT_VIOLATION,
            violations=violations,
            notes=notes,
            repair_contract=_derive_repair_contract(
                category=AuditCategory.CONTENT_VIOLATION,
                violations=violations,
                notes=notes,
            ),
            raw=f'{{"pass": false, "class": "content_violation", "notes": ["{reason}"]}}',
        )

    if not _has_evidence_section(content):
        reason = '章节缺少「### 证据与出处」小节，无法提取来源条目'
        violations = [Violation(
            rule=AuditRuleCode.P3,
            severity="high",
            reason=reason,
            rewrite_hint='重新写作，必须在章节末尾输出「### 证据与出处」并列出所有引用来源',
        )]
        notes = [reason]
        return AuditDecision(
            passed=False,
            category=AuditCategory.EVIDENCE_INSUFFICIENT,
            violations=violations,
            notes=notes,
            repair_contract=_derive_repair_contract(
                category=AuditCategory.EVIDENCE_INSUFFICIENT,
                violations=violations,
                notes=notes,
            ),
            raw=f'{{"pass": false, "class": "evidence_insufficient", "notes": ["{reason}"]}}',
        )

    return None


def _parse_audit_decision(raw_text: str) -> AuditDecision:
    """解析审计 JSON 输出。

    Args:
        raw_text: 模型返回的原始文本。

    Returns:
        规范化后的审计决策对象。

    Raises:
        无。
    """

    normalized = _extract_markdown_content(raw_text)
    json_text = _extract_json_text(normalized)
    if not json_text:
        violations = [Violation(rule=AuditRuleCode.S1, severity="high", reason="审计输出不是合法 JSON")]
        notes = ["请检查审计模型输出格式"]
        return AuditDecision(
            passed=False,
            category=AuditCategory.STYLE_VIOLATION,
            violations=violations,
            notes=notes,
            repair_contract=_derive_repair_contract(
                category=AuditCategory.STYLE_VIOLATION,
                violations=violations,
                notes=notes,
            ),
            raw=raw_text,
        )

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        violations = [Violation(rule=AuditRuleCode.S1, severity="high", reason="审计 JSON 解析失败")]
        notes = ["请检查审计模型输出格式"]
        return AuditDecision(
            passed=False,
            category=AuditCategory.STYLE_VIOLATION,
            violations=violations,
            notes=notes,
            repair_contract=_derive_repair_contract(
                category=AuditCategory.STYLE_VIOLATION,
                violations=violations,
                notes=notes,
            ),
            raw=raw_text,
        )

    violations = _normalize_audit_violations(list(payload.get("violations", [])))
    notes = list(payload.get("notes", []))
    passed, category = _recompute_audit_result(violations=violations)
    return AuditDecision(
        passed=passed,
        category=category,
        violations=violations,
        notes=notes,
        repair_contract=_normalize_repair_contract(
            payload.get("repair_contract"),
            category=category,
            violations=violations,
            notes=notes,
        ),
        raw=json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _collect_confirmable_evidence_violations(violations: list[Violation]) -> list[dict[str, Any]]:
    """收集需要进入证据复核环节的疑似违规。

    Args:
        violations: 审计违规列表。

    Returns:
        可进入 confirm 的违规子集，带稳定 violation_id；
        返回普通字典以便直接序列化进 LLM prompt。

    Raises:
        无。
    """

    collected: list[dict[str, Any]] = []
    for index, violation in enumerate(violations, start=1):
        rule = violation.rule
        if rule not in CONFIRMABLE_EVIDENCE_AUDIT_RULE_CODES:
            continue
        collected.append(
            {
                "violation_id": f"evidence_{index}",
                "rule": rule.value,
                "severity": violation.severity,
                "excerpt": violation.excerpt,
                "reason": violation.reason,
                "rewrite_hint": violation.rewrite_hint,
            }
        )
    return collected


def _parse_evidence_confirmation_result(raw_text: str) -> EvidenceConfirmationResult:
    """解析证据复核 JSON 输出。

    Args:
        raw_text: 模型返回的原始文本。

    Returns:
        结构化证据复核结果。

    Raises:
        ValueError: 当输出不是合法 JSON 或字段非法时抛出。
    """

    normalized = _extract_markdown_content(raw_text)
    json_text = _extract_json_text(normalized)
    if not json_text:
        raise ValueError("证据复核输出不是合法 JSON")
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError("证据复核 JSON 解析失败") from exc

    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        raise ValueError("证据复核 results 必须为数组")
    entries: list[EvidenceConfirmationEntry] = []
    for item in raw_results:
        if not isinstance(item, dict):
            raise ValueError("证据复核结果项必须为对象")
        violation_id = str(item.get("violation_id", "")).strip()
        rule = normalize_audit_rule_code(item.get("rule"))
        excerpt = str(item.get("excerpt", ""))
        status = normalize_evidence_confirmation_status(item.get("status"))
        reason = str(item.get("reason", "")).strip()
        rewrite_hint = str(item.get("rewrite_hint", "")).strip()
        anchor_fix = _parse_evidence_anchor_fix(item.get("anchor_fix"))
        if not violation_id or rule == AuditRuleCode.UNKNOWN or not excerpt or not status or not reason:
            raise ValueError("证据复核结果缺少必填字段")
        entries.append(
            EvidenceConfirmationEntry(
                violation_id=violation_id,
                rule=rule,
                excerpt=excerpt,
                status=status,
                reason=reason,
                rewrite_hint=rewrite_hint,
                anchor_fix=anchor_fix,
            )
        )
    notes = payload.get("notes", [])
    if not isinstance(notes, list):
        raise ValueError("证据复核 notes 必须为数组")
    return EvidenceConfirmationResult(entries=entries, notes=[str(item) for item in notes], raw=json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_evidence_anchor_fix(raw_value: object) -> EvidenceAnchorFix | None:
    """解析 confirm 返回的结构化证据锚点修复信息。

    Args:
        raw_value: confirm 结果中的 ``anchor_fix`` 字段。

    Returns:
        结构化锚点修复对象；若未提供有效内容则返回 ``None``。

    Raises:
        ValueError: 当 ``anchor_fix`` 结构不合法时抛出。
    """

    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("证据复核 anchor_fix 必须为对象")
    if not raw_value:
        return None
    kind = str(raw_value.get("kind", "")).strip()
    action = str(raw_value.get("action", "")).strip()
    evidence_line = str(raw_value.get("evidence_line", "")).strip()
    section_path = str(raw_value.get("section_path", "")).strip()
    statement_type = str(raw_value.get("statement_type", "")).strip()
    period = str(raw_value.get("period", "")).strip()
    keep_existing_evidence = bool(raw_value.get("keep_existing_evidence", True))
    if not kind and not action and not evidence_line and not section_path and not statement_type and not period:
        raw_rows = raw_value.get("rows", [])
        if raw_rows in (None, [], ()):
            return None
    if kind not in {"same_filing_section", "same_filing_statement", "same_filing_evidence_line"}:
        raise ValueError(f"证据复核 anchor_fix.kind 不受支持: {kind}")
    if action not in {"append", "refine_existing"}:
        raise ValueError(f"证据复核 anchor_fix.action 不受支持: {action}")
    rows = raw_value.get("rows", [])
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        raise ValueError("证据复核 anchor_fix.rows 必须为数组")
    return EvidenceAnchorFix(
        kind=kind,
        action=action,
        keep_existing_evidence=keep_existing_evidence,
        evidence_line=evidence_line,
        section_path=section_path,
        statement_type=statement_type,
        period=period,
        rows=[str(item).strip() for item in rows if str(item).strip()],
    )


def _merge_confirmed_evidence_results(*, audit_decision: AuditDecision, confirmation_result: EvidenceConfirmationResult) -> AuditDecision:
    """将证据复核结果合并回审计决策。

    Args:
        audit_decision: 原始疑似审计结果。
        confirmation_result: 证据复核结果。

    Returns:
        合并后的最终审计结果。

    Raises:
        无。
    """

    confirmable = _collect_confirmable_evidence_violations(audit_decision.violations)
    id_to_entry = {entry.violation_id: entry for entry in confirmation_result.entries}
    merged_violations: list[Violation] = []
    for violation in audit_decision.violations:
        matched = next(
            (
                candidate
                for candidate in confirmable
                if candidate["rule"] == violation.rule.value
                and candidate["excerpt"] == violation.excerpt
            ),
            None,
        )
        if matched is None:
            merged_violations.append(violation)
            continue
        entry = id_to_entry.get(str(matched["violation_id"]))
        if entry is None:
            merged_violations.append(violation)
            continue
        if entry.status == EvidenceConfirmationStatus.SUPPORTED:
            continue
        if entry.status in {
            EvidenceConfirmationStatus.SUPPORTED_BUT_ANCHOR_TOO_COARSE,
            EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
        }:
            if entry.anchor_fix is not None:
                continue
            merged_violations.append(_build_supported_anchor_followup_violation(entry))
            continue
        # 更新违规字段：复核状态、原因、修复建议、处置模式
        updated_fields: dict[str, str] = {
            "confirmation_status": entry.status.value,
            "reason": entry.reason,
        }
        if entry.rewrite_hint:
            updated_fields["rewrite_hint"] = entry.rewrite_hint
        if entry.status == EvidenceConfirmationStatus.CONFIRMED_MISSING:
            updated_fields["resolution_mode"] = RepairResolutionMode.DELETE_CLAIM.value
        merged_violations.append(replace(violation, **updated_fields))
    passed, category = _recompute_audit_result(violations=merged_violations)
    notes = list(audit_decision.notes)
    notes.extend(item for item in confirmation_result.notes if item not in notes)
    merged_payload = {
        "pass": passed,
        "class": category,
        "violations": [asdict(v) for v in merged_violations],
        "notes": notes,
        "confirmation": _build_confirm_artifact_payload(confirmation_result),
    }
    return AuditDecision(
        passed=passed,
        category=category,
        violations=merged_violations,
        notes=notes,
        repair_contract=_normalize_repair_contract(None, category=category, violations=merged_violations, notes=notes),
        raw=json.dumps(merged_payload, ensure_ascii=False, indent=2),
    )


def _build_supported_anchor_followup_violation(entry: EvidenceConfirmationEntry) -> Violation:
    """把 supported_* 条目转成待后续 rewrite 处理的 S7 违规。

    Args:
        entry: 单条证据复核结果。

    Returns:
        规范化后的 S7 违规。

    Raises:
        ValueError: 当条目状态不是 supported_* 时抛出。
    """

    if entry.status not in {
        EvidenceConfirmationStatus.SUPPORTED_BUT_ANCHOR_TOO_COARSE,
        EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
    }:
        raise ValueError(f"仅支持把 supported_* 条目转成 S7，当前状态: {entry.status}")
    return Violation(
        rule=AuditRuleCode.S7,
        severity="low",
        excerpt=entry.excerpt,
        reason=entry.reason,
        confirmation_status=entry.status.value,
        resolution_mode=RepairResolutionMode.ANCHOR_FIX_ONLY.value,
        rewrite_hint=entry.rewrite_hint or (
            "补充同一 filing 内正确锚点，不要删除正文信息。"
            if entry.status == EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING
            else "仅调整证据与出处锚点，不要删除正文信息。"
        ),
    )


def _rebuild_audit_decision_with_confirmation(
    *,
    audit_decision: AuditDecision,
    violations: list[Violation],
    confirmation_result: EvidenceConfirmationResult | None,
) -> AuditDecision:
    """基于更新后的违规列表重建最终审计结果。

    Args:
        audit_decision: 原始审计结果。
        violations: 更新后的违规列表。
        confirmation_result: 当前证据复核结果；若为空则不写 confirmation 负载。

    Returns:
        重建后的审计结果。

    Raises:
        无。
    """

    passed, category = _recompute_audit_result(violations=violations)
    notes = list(audit_decision.notes)
    payload: dict[str, Any] = {
        "pass": passed,
        "class": category,
        "violations": [asdict(v) for v in violations],
        "notes": notes,
    }
    if confirmation_result is not None:
        payload["confirmation"] = _build_confirm_artifact_payload(confirmation_result)
    return AuditDecision(
        passed=passed,
        category=category,
        violations=violations,
        notes=notes,
        repair_contract=_normalize_repair_contract(None, category=category, violations=violations, notes=notes),
        raw=json.dumps(payload, ensure_ascii=False, indent=2),
    )


def _rebuild_anchor_followup_audit_decision(
    *,
    audit_decision: AuditDecision,
    confirmation_result: EvidenceConfirmationResult,
) -> AuditDecision:
    """在锚点轻量修复回退路径上，把被 merge 阶段静默移除的 supported_* + anchor_fix 违规重建回审计决策。

    背景：``_merge_confirmed_evidence_results`` 在 confirm 合并时，对
    ``SUPPORTED_BUT_ANCHOR_TOO_COARSE`` / ``SUPPORTED_ELSEWHERE_IN_SAME_FILING``
    且 ``anchor_fix`` 非空的 entry 直接 ``continue``——预设后续 ``maybe_rewrite_evidence_anchors``
    会用机械 rewrite 吸收这些违规。但若机械 rewrite 后置校验失败、原文回退，
    那些被假设会被吸收的违规仍然真实存在却已不在 audit_decision 里，
    审计 gate 会把章节误判为 passed。

    本 helper 仅在校验失败回退路径调用：基于现有 ``confirmation_result``
    重建对应的 S7 followup 违规并合入当前 audit_decision，
    再用 ``_recompute_audit_result`` 重算 ``passed`` / ``category``。
    若没有需要恢复的 entry，则原样返回。

    Args:
        audit_decision: 当前最终审计结果（已经过 confirm merge）。
        confirmation_result: 本轮证据复核结果。

    Returns:
        若有需要恢复的违规则返回新的 ``AuditDecision``；否则返回原对象。

    Raises:
        无。
    """

    pending_entries = [
        entry
        for entry in confirmation_result.entries
        if entry.status
        in {
            EvidenceConfirmationStatus.SUPPORTED_BUT_ANCHOR_TOO_COARSE,
            EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
        }
        and entry.anchor_fix is not None
    ]
    if not pending_entries:
        return audit_decision
    existing_followups = [
        violation
        for violation in audit_decision.violations
        if any(
            _is_supported_anchor_followup_violation_for_entry(violation=violation, entry=entry)
            for entry in pending_entries
        )
    ]
    missing_entries = [
        entry
        for entry in pending_entries
        if not any(
            _is_supported_anchor_followup_violation_for_entry(violation=violation, entry=entry)
            for violation in existing_followups
        )
    ]
    if not missing_entries:
        return audit_decision
    rebuilt_violations: list[Violation] = list(audit_decision.violations)
    rebuilt_violations.extend(_build_supported_anchor_followup_violation(entry) for entry in missing_entries)
    return _rebuild_audit_decision_with_confirmation(
        audit_decision=audit_decision,
        violations=rebuilt_violations,
        confirmation_result=confirmation_result,
    )


def _drop_resolved_supported_anchor_violations(
    *,
    audit_decision: AuditDecision,
    confirmation_result: EvidenceConfirmationResult,
    resolved_violation_ids: set[str],
) -> AuditDecision:
    """移除已被 anchor rewrite 成功吸收的 supported_* 问题。

    Args:
        audit_decision: 当前最终审计结果。
        confirmation_result: 当前证据复核结果。
        resolved_violation_ids: 本轮已被机械 rewrite 成功吸收的 violation_id 集合。

    Returns:
        移除已解决问题后的最终审计结果；若无变化则返回原对象。

    Raises:
        无。
    """

    if not resolved_violation_ids:
        return audit_decision
    resolved_supported_entries = [
        entry
        for entry in confirmation_result.entries
        if entry.violation_id in resolved_violation_ids
        and entry.status in {
            EvidenceConfirmationStatus.SUPPORTED_BUT_ANCHOR_TOO_COARSE,
            EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
        }
    ]
    if not resolved_supported_entries:
        return audit_decision
    remaining_violations: list[Violation] = []
    for violation in audit_decision.violations:
        matched_entry = next(
            (
                entry
                for entry in resolved_supported_entries
                if _is_supported_anchor_followup_violation_for_entry(violation=violation, entry=entry)
            ),
            None,
        )
        if matched_entry is not None:
            continue
        remaining_violations.append(violation)
    if len(remaining_violations) == len(audit_decision.violations):
        return audit_decision
    return _rebuild_audit_decision_with_confirmation(
        audit_decision=audit_decision,
        violations=remaining_violations,
        confirmation_result=confirmation_result,
    )


def _is_supported_anchor_followup_violation_for_entry(*, violation: Violation, entry: EvidenceConfirmationEntry) -> bool:
    """判断最终审计中的 S7 违规是否对应某条 supported_* confirm 结果。

    Args:
        violation: 最终审计中的单条违规。
        entry: 单条证据复核结果。

    Returns:
        若两者对应同一条 supported_* 后续问题则返回 ``True``。

    Raises:
        无。
    """

    if entry.status not in {
        EvidenceConfirmationStatus.SUPPORTED_BUT_ANCHOR_TOO_COARSE,
        EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
    }:
        return False
    return (
        violation.rule == AuditRuleCode.S7
        and violation.excerpt == entry.excerpt
        and violation.reason == entry.reason
    )
