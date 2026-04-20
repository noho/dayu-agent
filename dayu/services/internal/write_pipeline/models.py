"""写作流水线领域模型。

该模块定义 `--write` 流程所需的数据结构，负责：
- 配置对象建模。
- 公司级 facet 归因结果建模。
- 章节任务与结果状态建模。
- 运行清单（manifest）序列化/反序列化。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from dayu.services.internal.write_pipeline.chapter_contracts import ChapterContract, ItemRule
from dayu.services.internal.write_pipeline.enums import (
    AuditCategory,
    AuditRuleCode,
    EvidenceConfirmationStatus,
    RepairStrategy,
)
from dayu.services.contracts import SceneModelConfig, WriteRunConfig

ChapterStatus = Literal["pending", "passed", "failed"]


@dataclass(frozen=True)
class CompanyFacetProfile:
    """公司级视角标签归因结果。

    Args:
        primary_facets: 主行业/商业模式标签。
        cross_cutting_facets: 横切约束标签。
        confidence_notes: 简短归因说明。

    Returns:
        无。

    Raises:
        无。
    """

    primary_facets: list[str] = field(default_factory=list)
    cross_cutting_facets: list[str] = field(default_factory=list)
    confidence_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典。

        Args:
            无。

        Returns:
            纯字典结果。

        Raises:
            无。
        """

        return {
            "primary_facets": list(self.primary_facets),
            "cross_cutting_facets": list(self.cross_cutting_facets),
            "confidence_notes": self.confidence_notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CompanyFacetProfile":
        """从字典恢复 facet 归因结果。

        Args:
            raw: 原始字典。

        Returns:
            `CompanyFacetProfile` 实例。

        Raises:
            TypeError: 当字段类型非法时抛出。
        """

        primary_facets = list(raw.get("primary_facets", raw.get("business_model_tags", [])))
        cross_cutting_facets = list(raw.get("cross_cutting_facets", raw.get("constraint_tags", [])))
        confidence_notes = str(raw.get("confidence_notes", raw.get("judgement_notes", ""))).strip()
        if any(not isinstance(item, str) for item in primary_facets):
            raise TypeError("company_facets.primary_facets / business_model_tags 必须为字符串列表")
        if any(not isinstance(item, str) for item in cross_cutting_facets):
            raise TypeError("company_facets.cross_cutting_facets / constraint_tags 必须为字符串列表")
        return cls(
            primary_facets=[item.strip() for item in primary_facets if item.strip()],
            cross_cutting_facets=[item.strip() for item in cross_cutting_facets if item.strip()],
            confidence_notes=confidence_notes,
        )

    def all_facets(self) -> list[str]:
        """返回按顺序合并后的全部 facet。

        Args:
            无。

        Returns:
            去重后的 facet 列表。

        Raises:
            无。
        """

        merged: list[str] = []
        for item in [*self.primary_facets, *self.cross_cutting_facets]:
            if item not in merged:
                merged.append(item)
        return merged


def serialize_scene_models(scene_models: dict[str, "SceneModelConfig"]) -> dict[str, dict[str, float | str]]:
    """将 scene 模型配置映射转换为可序列化字典。

    Args:
        scene_models: scene 到模型配置的映射。

    Returns:
        纯字典结果。

    Raises:
        无。
    """

    return {
        scene_name: {
            "name": config.name,
            "temperature": config.temperature,
        }
        for scene_name, config in scene_models.items()
    }


@dataclass
class ChapterTask:
    """章节写作任务。

    Args:
        index: 章节序号（从 1 开始）。
        title: 章节标题。
        report_goal: 全文总目标。
        audience_profile: 全局读者画像。
        chapter_goal: 本章总目标。
        skeleton: 章节模板骨架。
        chapter_contract: 章节级写作合同。
        item_rules: 条件型条目规则。

    Returns:
        无。

    Raises:
        无。
    """

    index: int
    title: str
    skeleton: str
    report_goal: str = ""
    audience_profile: str = ""
    chapter_goal: str = ""
    chapter_contract: ChapterContract = field(default_factory=ChapterContract.empty)
    item_rules: list[ItemRule] = field(default_factory=list)


@dataclass
class Violation:
    """单条审计违规。

    Args:
        rule: 违规规则编号（如 E1、S3、P1）。
        severity: 严重程度（high / medium / low）。
        excerpt: 触发违规的正文片段。
        reason: 违规原因说明。
        rewrite_hint: 修复建议。
        confirmation_status: 证据复核确认状态。
        resolution_mode: 处置模式（如 delete_claim / rewrite_with_existing_evidence）。

    Returns:
        无。

    Raises:
        无。
    """

    rule: AuditRuleCode = AuditRuleCode.UNKNOWN
    severity: str = ""
    excerpt: str = ""
    reason: str = ""
    rewrite_hint: str = ""
    confirmation_status: str = ""
    resolution_mode: str = ""


@dataclass
class MissingEvidenceSlot:
    """修复合同中的缺失证据槽位。

    Args:
        slot_id: 槽位唯一标识。
        rule: 触发此槽位的违规规则。
        description: 缺失证据说明。
        required_evidence: 需要补充的证据类型。
        severity: 严重程度。

    Returns:
        无。

    Raises:
        无。
    """

    slot_id: str = ""
    rule: AuditRuleCode = AuditRuleCode.UNKNOWN
    description: str = ""
    required_evidence: str = ""
    severity: str = ""


@dataclass
class OffendingClaimSpan:
    """修复合同中的违规断言片段。

    Args:
        rule: 触发的违规规则。
        excerpt: 命中的正文片段。
        reason: 违规原因。

    Returns:
        无。

    Raises:
        无。
    """

    rule: AuditRuleCode = AuditRuleCode.UNKNOWN
    excerpt: str = ""
    reason: str = ""


@dataclass
class RemediationAction:
    """修复合同中的单条修复动作。

    Args:
        action_id: 动作唯一标识。
        rule: 触发的违规规则。
        excerpt: 命中的正文片段。
        reason: 违规原因。
        rewrite_hint: 修复建议。
        confirmation_status: 证据复核确认状态。
        resolution_mode: 处置模式。
        target_kind_hint: 建议的 patch 粒度。

    Returns:
        无。

    Raises:
        无。
    """

    action_id: str = ""
    rule: AuditRuleCode = AuditRuleCode.UNKNOWN
    excerpt: str = ""
    reason: str = ""
    rewrite_hint: str = ""
    confirmation_status: str = ""
    resolution_mode: str = ""
    target_kind_hint: str = ""


@dataclass
class RepairContract:
    """结构化修复合同。

    Args:
        contract_version: 合同协议版本。
        missing_evidence_slots: 缺失证据槽位列表。
        offending_claim_spans: 违规断言片段列表。
        remediation_actions: 修复动作列表。
        preferred_tool_action: 建议的修复工具动作。
        repair_strategy: 修复策略（patch / regenerate / none）。
        retry_scope: 重试作用域。
        notes: 备注列表。

    Returns:
        无。

    Raises:
        无。
    """

    contract_version: str = ""
    missing_evidence_slots: list[MissingEvidenceSlot] = field(default_factory=list)
    offending_claim_spans: list[OffendingClaimSpan] = field(default_factory=list)
    remediation_actions: list[RemediationAction] = field(default_factory=list)
    preferred_tool_action: str = ""
    repair_strategy: str = ""
    retry_scope: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class AuditDecision:
    """审计结果对象。

    Args:
        passed: 审计是否通过。
        category: 分类（ok/evidence_insufficient/style_violation）。
        violations: 违规列表。
        notes: 备注列表。
        repair_contract: 结构化修复合同。
        raw: 原始审计文本。

    Returns:
        无。

    Raises:
        无。
    """

    passed: bool
    category: AuditCategory
    violations: list[Violation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    repair_contract: RepairContract = field(default_factory=RepairContract)
    raw: str = ""


@dataclass
class EvidenceAnchorFix:
    """证据锚点修复定位。

    Args:
        kind: 修复类型，当前支持 `same_filing_section`、`same_filing_statement`、
            `same_filing_evidence_line`。
        action: 修复动作，当前支持 `append`、`refine_existing`。
        keep_existing_evidence: 是否保留当前 evidence line。
        evidence_line: 若已能构造完整 evidence line，则直接返回该字段。
        section_path: 同一 filing 内的稳定标题路径。
        statement_type: 财务报表类型，如 `income`、`cash_flow`。
        period: 需要补充的 period 串。
        rows: 需要补充的报表行标签列表。

    Returns:
        无。

    Raises:
        无。
    """

    kind: str
    action: str
    keep_existing_evidence: bool = True
    evidence_line: str = ""
    section_path: str = ""
    statement_type: str = ""
    period: str = ""
    rows: list[str] = field(default_factory=list)


@dataclass
class EvidenceConfirmationEntry:
    """证据违规确认条目。

    Args:
        violation_id: 对应的疑似违规唯一标识。
        rule: 对应的证据规则编号。
        excerpt: 对应的正文触发片段。
        status: 确认状态。
        reason: 确认结论说明。
        rewrite_hint: 修复动作建议。
        anchor_fix: 结构化证据锚点修复定位。

    Returns:
        无。

    Raises:
        无。
    """

    violation_id: str
    rule: AuditRuleCode
    excerpt: str
    status: EvidenceConfirmationStatus
    reason: str
    rewrite_hint: str = ""
    anchor_fix: EvidenceAnchorFix | None = None


@dataclass
class EvidenceConfirmationResult:
    """证据违规确认结果。

    Args:
        entries: 确认条目列表。
        notes: 补充备注。
        raw: 模型原始输出。

    Returns:
        无。

    Raises:
        无。
    """

    entries: list[EvidenceConfirmationEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw: str = ""


@dataclass
class ChapterResult:
    """章节执行结果。

    Args:
        index: 章节序号。
        title: 章节标题。
        status: 最终状态。
        content: 章节正文。
        audit_passed: 最终审计是否通过。
        retry_count: 重写次数。
        failure_reason: 失败原因。
        evidence_items: 提取到的证据条目。
        process_state: 章节加工过程状态（用于 manifest 可追溯）。

    Returns:
        无。

    Raises:
        无。
    """

    index: int
    title: str
    status: ChapterStatus
    content: str
    audit_passed: bool
    retry_count: int = 0
    failure_reason: str = ""
    evidence_items: list[str] = field(default_factory=list)
    process_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceEntry:
    """来源条目对象。

    Args:
        text: 原始来源文本。
        group: 分组名。
        date_text: 日期文本（用于排序）。

    Returns:
        无。

    Raises:
        无。
    """

    text: str
    group: str
    date_text: str = ""


@dataclass
class RunManifest:
    """写作流水线运行清单。

    Args:
        version: 清单版本。
        signature: 运行签名。
        config: 写作配置。
        chapter_results: 章节结果映射，key 为章节标题。
        company_facets: 公司级 facet 归因结果。

    Returns:
        无。

    Raises:
        无。
    """

    version: str
    signature: str
    config: WriteRunConfig
    chapter_results: dict[str, ChapterResult] = field(default_factory=dict)
    company_facets: CompanyFacetProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化字典。

        Args:
            无。

        Returns:
            字典对象。

        Raises:
            无。
        """

        return {
            "version": self.version,
            "signature": self.signature,
            "config": asdict(self.config),
            "chapter_results": {title: asdict(result) for title, result in self.chapter_results.items()},
            "company_facets": self.company_facets.to_dict() if self.company_facets is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunManifest":
        """从字典恢复清单对象。

        Args:
            data: 原始清单字典。

        Returns:
            `RunManifest` 实例。

        Raises:
            KeyError: 字段缺失时抛出。
            TypeError: 字段类型错误时抛出。
        """

        config_dict = dict(data["config"])
        raw_scene_models = dict(config_dict.get("scene_models", {}))
        config_dict["scene_models"] = {
            scene_name: SceneModelConfig(**scene_model)
            for scene_name, scene_model in raw_scene_models.items()
        }
        result_dict = dict(data.get("chapter_results", {}))
        chapter_results: dict[str, ChapterResult] = {}
        for title, raw in result_dict.items():
            chapter_results[title] = ChapterResult(**raw)
        raw_company_facets = data.get("company_facets")
        return cls(
            version=str(data["version"]),
            signature=str(data["signature"]),
            config=WriteRunConfig(**config_dict),
            chapter_results=chapter_results,
            company_facets=(
                CompanyFacetProfile.from_dict(dict(raw_company_facets))
                if isinstance(raw_company_facets, dict)
                else None
            ),
        )
