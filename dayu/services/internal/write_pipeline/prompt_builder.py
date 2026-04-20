"""Prompt 输入构建与渲染模块。

该模块封装写作流水线中所有 task prompt 输入构建与渲染逻辑，
包括章节 prompt、repair prompt、confirm prompt、
regenerate prompt、overview prompt 和 decision prompt。
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any, Mapping, Optional, Sequence

from dayu.contracts.infrastructure import WorkspaceResourcesProtocol
from dayu.services.internal.write_pipeline.artifact_store import (
    ArtifactStore,
    _OVERVIEW_CHAPTER_TITLE,
    _SOURCE_CHAPTER_TITLE,
)
from dayu.services.internal.write_pipeline.audit_formatting import (
    _build_current_visible_headings_block,
    _extract_overview_summary,
    _extract_research_decision_summary,
)
from dayu.services.internal.write_pipeline.audit_rules import (
    _build_research_decision_audit_summary,
)
from dayu.services.internal.write_pipeline.company_facets import (
    filter_chapter_contract_by_facets,
    filter_item_rules_by_facets,
    render_company_facets_for_prompt,
)
from dayu.services.internal.write_pipeline.models import (
    AuditDecision,
    ChapterResult,
    ChapterTask,
    CompanyFacetProfile,
    WriteRunConfig,
)
from dayu.services.internal.write_pipeline.prompt_contracts import (
    TaskPromptContract,
    parse_task_prompt_contract,
    render_task_prompt,
)
from dayu.services.internal.write_pipeline.source_list_builder import extract_evidence_items
from dayu.services.internal.write_pipeline.template_parser import TemplateLayout

_DECISION_SOURCE_OF_TRUTH = "structured_prior_chapter_summaries_v1"


_DECISION_CHAPTER_TITLE = "是否值得继续深研与待验证问题"


def _clone_company_facet_profile(company_facets: CompanyFacetProfile | None) -> CompanyFacetProfile | None:
    """复制公司级 facet 归因结果，避免共享可变列表引用。

    Args:
        company_facets: 原始 facet 归因结果。

    Returns:
        可安全跨线程读取的 facet 副本；当输入为空时返回 ``None``。

    Raises:
        无。
    """

    if company_facets is None:
        return None
    return CompanyFacetProfile(
        primary_facets=list(company_facets.primary_facets),
        cross_cutting_facets=list(company_facets.cross_cutting_facets),
        confidence_notes=company_facets.confidence_notes,
    )


def _clone_company_facet_catalog(company_facet_catalog: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """复制 facet 候选目录，避免外部继续修改内部状态。

    Args:
        company_facet_catalog: 原始 facet 候选目录。

    Returns:
        键和值列表均已复制的新字典。

    Raises:
        无。
    """

    return {
        key: [str(item) for item in values]
        for key, values in company_facet_catalog.items()
    }


def _build_prior_decision_tasks(layout: TemplateLayout) -> list[ChapterTask]:
    """构建第10章决策综合依赖的前置章节任务列表。

    约定：
    - 第0章“投资要点概览”不作为第10章的 source-of-truth。
    - 第10章只依赖其前面所有业务章节，通常即前1–9章。

    Args:
        layout: 模板布局对象。

    Returns:
        前置章节任务列表，按模板顺序排列。

    Raises:
        无。
    """

    decision_chapter = next((chapter for chapter in layout.chapters if chapter.title == _DECISION_CHAPTER_TITLE), None)
    if decision_chapter is None:
        return []

    tasks: list[ChapterTask] = []
    for chapter in layout.chapters:
        if chapter.index >= decision_chapter.index:
            break
        if chapter.title == _OVERVIEW_CHAPTER_TITLE:
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



class PromptBuilder:
    """Prompt 输入构建与渲染器。

    职责：
    - 按 task prompt contract 渲染写作流水线的各类 prompt。
    - 构建 prompt 所需的结构化输入字段。
    - 管理 task prompt 模板缓存。
    """

    def __init__(
        self,
        *,
        workspace: WorkspaceResourcesProtocol,
        write_config: WriteRunConfig,
        store: ArtifactStore,
    ) -> None:
        """初始化 PromptBuilder。

        Args:
            workspace: 工作区稳定资源。
            write_config: 写作运行配置。
            store: 产物存储器。
        """

        self._workspace = workspace
        self._write_config = write_config
        self._store = store
        self._task_prompt_cache: dict[str, tuple[str, TaskPromptContract]] = {}
        self._task_prompt_cache_lock = threading.Lock()
        self._company_facet_state_lock = threading.Lock()
        self._company_facets: CompanyFacetProfile | None = None
        self._company_facet_catalog: dict[str, list[str]] = {}

    def set_company_facets(self, company_facets: CompanyFacetProfile | None) -> None:
        """设置当前运行的公司级 facet 归因结果。

        Args:
            company_facets: 公司级 facet 归因结果；未知时可为 ``None``。

        Returns:
            无。

        Raises:
            无。
        """

        with self._company_facet_state_lock:
            self._company_facets = _clone_company_facet_profile(company_facets)

    def set_company_facet_catalog(self, company_facet_catalog: dict[str, list[str]]) -> None:
        """设置当前模板声明的 facet 候选目录。

        Args:
            company_facet_catalog: facet 候选目录映射。

        Returns:
            无。

        Raises:
            无。
        """

        with self._company_facet_state_lock:
            self._company_facet_catalog = _clone_company_facet_catalog(company_facet_catalog)

    def _get_company_facet_state_snapshot(self) -> tuple[CompanyFacetProfile | None, dict[str, list[str]]]:
        """获取当前 facet 状态的一致性快照。

        Args:
            无。

        Returns:
            `(company_facets, company_facet_catalog)` 的深拷贝快照。

        Raises:
            无。
        """

        with self._company_facet_state_lock:
            return (
                _clone_company_facet_profile(self._company_facets),
                _clone_company_facet_catalog(self._company_facet_catalog),
            )

    def build_chapter_prompt_inputs(
        self,
        *,
        task: ChapterTask,
        company_name: str,
        extra_inputs: Optional[dict[str, Any]] = None,
        include_item_rules: bool = True,
    ) -> dict[str, Any]:
        """公开构建章节类 prompt 输入字段。

        Args:
            task: 章节任务。
            company_name: 公司名称。
            extra_inputs: 附加输入字段。
            include_item_rules: 是否包含 item rules。

        Returns:
            prompt 输入字段字典。

        Raises:
            无。
        """

        return self._build_chapter_prompt_inputs(
            task=task,
            company_name=company_name,
            extra_inputs=extra_inputs,
            include_item_rules=include_item_rules,
        )

    def build_repair_prompt_inputs(
        self,
        *,
        task: ChapterTask,
        company_name: str,
        current_content: str,
        audit_decision: AuditDecision,
    ) -> dict[str, Any]:
        """公开构建 repair prompt 输入字段。

        Args:
            task: 章节任务。
            company_name: 公司名称。
            current_content: 当前章节正文。
            audit_decision: 当前审计结果。

        Returns:
            repair prompt 输入字段字典。

        Raises:
            无。
        """

        return self._build_repair_prompt_inputs(
            task=task,
            company_name=company_name,
            current_content=current_content,
            audit_decision=audit_decision,
        )

    def build_confirm_prompt_inputs(
        self,
        *,
        company_name: str,
        chapter_markdown: str,
        suspected_evidence_violations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """公开构建证据复核 prompt 输入字段。

        Args:
            company_name: 公司名称。
            chapter_markdown: 当前章节正文。
            suspected_evidence_violations: 待复核的疑似证据违规列表。

        Returns:
            证据复核 prompt 输入字段字典。

        Raises:
            无。
        """

        return self._build_confirm_prompt_inputs(
            company_name=company_name,
            chapter_markdown=chapter_markdown,
            suspected_evidence_violations=suspected_evidence_violations,
        )

    def build_regenerate_prompt_inputs(
        self,
        *,
        task: ChapterTask,
        company_name: str,
        current_content: str,
        audit_decision: AuditDecision,
    ) -> dict[str, Any]:
        """公开构建 regenerate prompt 输入字段。

        Args:
            task: 章节任务。
            company_name: 公司名称。
            current_content: 当前章节正文。
            audit_decision: 当前审计结果。

        Returns:
            regenerate prompt 输入字段字典。

        Raises:
            无。
        """

        return self._build_regenerate_prompt_inputs(
            task=task,
            company_name=company_name,
            current_content=current_content,
            audit_decision=audit_decision,
        )

    def build_overview_input(
        self,
        *,
        layout: TemplateLayout,
        chapter_results: dict[str, ChapterResult],
    ) -> str:
        """公开构建第0章概览输入。

        Args:
            layout: 模板布局对象。
            chapter_results: 章节结果映射。

        Returns:
            第0章结构化输入 Markdown。

        Raises:
            无。
        """

        return self._build_overview_input(layout=layout, chapter_results=chapter_results)

    def build_research_decision_prompt_inputs(
        self,
        *,
        task: ChapterTask,
        company_name: str,
        layout: TemplateLayout,
        chapter_results: dict[str, ChapterResult],
    ) -> dict[str, Any]:
        """公开构建第10章研究决策综合输入。

        Args:
            task: 第10章任务。
            company_name: 公司名称。
            layout: 模板布局对象。
            chapter_results: 当前章节结果映射。

        Returns:
            决策综合 prompt 输入字段字典。

        Raises:
            无。
        """

        return self._build_research_decision_prompt_inputs(
            task=task,
            company_name=company_name,
            layout=layout,
            chapter_results=chapter_results,
        )

    def render_task_prompt(self, *, prompt_name: str, prompt_inputs: dict[str, Any]) -> str:
        """公开渲染 task prompt。

        Args:
            prompt_name: task prompt 名称。
            prompt_inputs: prompt 输入字段。

        Returns:
            渲染后的 prompt 文本。

        Raises:
            ValueError: 当 prompt contract 非法或变量未替换完成时抛出。
        """

        return self._render_task_prompt(prompt_name=prompt_name, prompt_inputs=prompt_inputs)

    def _build_chapter_prompt_inputs(
        self,
        *,
        task: ChapterTask,
        company_name: str,
        extra_inputs: Optional[dict[str, Any]] = None,
        include_item_rules: bool = True,
    ) -> dict[str, Any]:
        """构建章节类 prompt 的显式输入字段。

        Args:
            task: 章节任务。
            company_name: 公司名称。
            extra_inputs: 任务特定附加字段。
            include_item_rules: 是否包含局部条件写作规则。

        Returns:
            面向 task prompt renderer 的字段字典。

        Raises:
            无。
        """

        company_facets, company_facet_catalog = self._get_company_facet_state_snapshot()
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
        prompt_inputs: dict[str, Any] = {
            "chapter": task.title,
            "company": company_name,
            "ticker": self._write_config.ticker,
            "company_facets_summary": render_company_facets_for_prompt(company_facets),
            "report_goal": task.report_goal,
            "audience_profile": task.audience_profile,
            "chapter_goal": task.chapter_goal,
            "skeleton": task.skeleton,
            "chapter_contract": filtered_contract.to_prompt_fields(),
        }
        if include_item_rules and filtered_item_rules:
            prompt_inputs["item_rules"] = [rule.to_prompt_dict() for rule in filtered_item_rules]
        if extra_inputs:
            prompt_inputs.update(extra_inputs)
        return prompt_inputs

    def _build_repair_prompt_inputs(
        self,
        *,
        task: ChapterTask,
        company_name: str,
        current_content: str,
        audit_decision: AuditDecision,
    ) -> dict[str, Any]:
        """构建局部 repair prompt 的显式输入字段。

        Args:
            task: 章节任务。
            company_name: 公司名称。
            current_content: 当前章节正文。
            audit_decision: 上轮审计结果。

        Returns:
            repair prompt 输入字段字典。

        Raises:
            无。
        """

        return {
            "chapter": task.title,
            "company": company_name,
            "ticker": self._write_config.ticker,
            "allow_new_facts": True,
            "retry_scope": audit_decision.repair_contract.retry_scope or "targeted_patch",
            "chapter_contract": task.chapter_contract.to_prompt_fields(),
            "last_repair_contract": asdict(audit_decision.repair_contract),
            "current_visible_headings": _build_current_visible_headings_block(current_content),
            "last_wrote_content": current_content,
        }

    def _build_confirm_prompt_inputs(
        self,
        *,
        company_name: str,
        chapter_markdown: str,
        suspected_evidence_violations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """构建证据复核 prompt 输入字段。

        Args:
            company_name: 公司名称。
            chapter_markdown: 当前章节正文。
            suspected_evidence_violations: 待复核的疑似证据违规列表。

        Returns:
            证据复核 prompt 输入字段字典。

        Raises:
            无。
        """

        return {
            "company": company_name,
            "ticker": self._write_config.ticker,
            "chapter_markdown": chapter_markdown,
            "suspected_evidence_violations": suspected_evidence_violations,
            "evidence_items": extract_evidence_items(chapter_markdown),
        }

    def _build_regenerate_prompt_inputs(
        self,
        *,
        task: ChapterTask,
        company_name: str,
        current_content: str,
        audit_decision: AuditDecision,
    ) -> dict[str, Any]:
        """构建整章 regenerate prompt 的显式输入字段。

        Args:
            task: 章节任务。
            company_name: 公司名称。
            current_content: 当前章节正文。
            audit_decision: 上轮审计结果。

        Returns:
            regenerate prompt 输入字段字典。

        Raises:
            无。
        """

        return self._build_chapter_prompt_inputs(
            task=task,
            company_name=company_name,
            extra_inputs={
                "allow_new_facts": True,
                "retry_scope": audit_decision.repair_contract.retry_scope or "chapter_regenerate",
                "last_audit_json_block": audit_decision.raw,
                "last_repair_contract": asdict(audit_decision.repair_contract),
                "last_wrote_content": current_content,
            },
        )

    def _build_overview_input(
        self,
        *,
        layout: TemplateLayout,
        chapter_results: dict[str, ChapterResult],
    ) -> str:
        """构建第0章概览回填的前文章节结构化输入。

        Args:
            layout: 模板布局对象。
            chapter_results: 章节结果映射。

        Returns:
            面向第0章的结构化输入 Markdown。

        Raises:
            无。
        """

        blocks: list[str] = []
        for chapter in layout.chapters:
            title = chapter.title
            if title in {_OVERVIEW_CHAPTER_TITLE, _SOURCE_CHAPTER_TITLE}:
                continue
            result = chapter_results.get(title)
            if result is None:
                continue
            section_summary = _extract_overview_summary(result.content)
            if not section_summary:
                continue
            blocks.append(f"## {title}\n\n{section_summary}")
        return "\n\n".join(blocks).strip()

    def _build_research_decision_prompt_inputs(
        self,
        *,
        task: ChapterTask,
        company_name: str,
        layout: TemplateLayout,
        chapter_results: dict[str, ChapterResult],
    ) -> dict[str, Any]:
        """构建第10章研究决策综合 prompt 的显式输入。

        Args:
            task: 第10章任务对象。
            company_name: 公司名称。
            layout: 模板布局对象。
            chapter_results: 当前已完成章节结果。

        Returns:
            面向决策综合 task prompt 的字段字典。

        Raises:
            无。
        """

        prior_chapters_input = self._build_research_decision_input(layout=layout, chapter_results=chapter_results)
        return self._build_chapter_prompt_inputs(
            task=task,
            company_name=company_name,
            extra_inputs={
                "prior_chapters_input": prior_chapters_input,
                "decision_source_of_truth": _DECISION_SOURCE_OF_TRUTH,
                "decision_allow_new_facts": True,
                "decision_allow_new_sources": True,
            },
        )

    def _build_research_decision_input(
        self,
        *,
        layout: TemplateLayout,
        chapter_results: dict[str, ChapterResult],
    ) -> str:
        """构建第10章的前文章节结构化输入。

        默认实现只抽取前1–9章的结构化摘要，而不直接喂整章全文。未来若要改为
        “结论+证据+未决项”或“章节全文输入”，只需替换此 builder 层，不改主流程。

        Args:
            layout: 模板布局对象。
            chapter_results: 当前已完成章节结果。

        Returns:
            用于第10章 prompt 的结构化 Markdown 输入。

        Raises:
            无。
        """

        blocks: list[str] = []
        for chapter in _build_prior_decision_tasks(layout):
            result = chapter_results.get(chapter.title)
            if result is None:
                continue
            block = self._build_research_decision_chapter_block(result)
            if block:
                blocks.append(block)
        return "\n\n".join(blocks).strip()

    def _build_research_decision_chapter_block(self, result: ChapterResult) -> str:
        """将单章结果转成第10章可消费的结构化输入块。

        Args:
            result: 单章执行结果。

        Returns:
            结构化 Markdown 文本；若无法提取有效摘要则返回空字符串。

        Raises:
            无。
        """

        chapter_content = self._store.load_chapter_content_for_decision(result)
        summary_body = _extract_research_decision_summary(chapter_content)
        audit_summary = ""
        if not self._is_fast_written_result(result):
            audit_payload = self._store.load_final_audit_payload(result)
            audit_summary = _build_research_decision_audit_summary(audit_payload, audit_passed=result.audit_passed)

        blocks: list[str] = [f"## {result.title}"]
        if summary_body:
            blocks.append(summary_body)
        if audit_summary:
            blocks.append(audit_summary)
        return "\n\n".join(blocks).strip()

    def _render_task_prompt(self, *, prompt_name: str, prompt_inputs: dict[str, Any]) -> str:
        """按 task prompt contract 渲染 prompt。

        Args:
            prompt_name: task prompt 名称。
            prompt_inputs: 显式输入字段。

        Returns:
            渲染后的 prompt 文本。

        Raises:
            ValueError: 当 contract 非法、字段不匹配或模板变量未全部替换时抛出。
        """

        prompt_template, prompt_contract = self._load_task_prompt_bundle(prompt_name)
        return render_task_prompt(
            prompt_template=prompt_template,
            prompt_contract=prompt_contract,
            prompt_inputs=prompt_inputs,
        )

    def _load_task_prompt_bundle(self, prompt_name: str) -> tuple[str, TaskPromptContract]:
        """加载 task prompt 模板与 sidecar contract。

        Args:
            prompt_name: task prompt 名称。

        Returns:
            `(prompt_template, prompt_contract)` 二元组。

        Raises:
            ValueError: 当 sidecar contract 非法时抛出。
        """

        with self._task_prompt_cache_lock:
            cached = self._task_prompt_cache.get(prompt_name)
            if cached is not None:
                return cached

            prompt_template = self._workspace.prompt_asset_store.load_task_prompt(prompt_name)
            raw_contract = self._workspace.prompt_asset_store.load_task_prompt_contract(prompt_name)
            prompt_contract = parse_task_prompt_contract(raw_contract, task_name=prompt_name)
            bundle = (prompt_template, prompt_contract)
            self._task_prompt_cache[prompt_name] = bundle
            return bundle

    @staticmethod
    def _is_fast_written_result(result: ChapterResult) -> bool:
        """判断结果是否来自 fast 写作模式。

        Args:
            result: 章节结果。

        Returns:
            若该结果显式标记为 fast 写作则返回 ``True``。

        Raises:
            无。
        """

        return str(result.process_state.get("final_stage") or "") == "fast_written"
