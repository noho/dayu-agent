"""写作流水线模块测试。"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence, cast

import pytest

from dayu.services.internal.write_pipeline import artifact_store as artifact_store_module
from dayu.services.internal.write_pipeline import audit_rules as audit_rules_module
from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedInfrastructureSpec,
    AcceptedModelSpec,
    AcceptedRuntimeSpec,
    AcceptedToolConfigSpec,
)
from dayu.contracts.cancellation import CancelledError
from dayu.contracts.events import AppEvent, AppEventType, AppResult
from dayu.contracts.infrastructure import WorkspaceResourcesProtocol
from dayu.contracts.model_config import ModelConfig
from dayu.contracts.prompt_assets import SceneManifestAsset, TaskPromptContractAsset
from dayu.contracts.tool_configs import DocToolLimits, FinsToolLimits, WebToolsConfig
from dayu.contracts.toolset_config import ToolsetConfigSnapshot, build_toolset_config_snapshot
from dayu.execution.options import (
    ConversationMemorySettings,
    ExecutionOptions,
    ResolvedExecutionOptions,
    TraceSettings,
)
from dayu.execution.runtime_config import AgentRuntimeConfig, OpenAIRunnerRuntimeConfig
from dayu.host.host_execution import HostExecutorProtocol
from dayu.log import Log
from dayu.prompting.scene_definition import load_scene_definition
from dayu.services.internal.write_pipeline.chapter_contracts import ChapterContract, ItemRule, PreferredLens
from dayu.services.internal.write_pipeline.audit_evidence_rewriter import (
    _build_anchor_rewrite_evidence_lines,
    _build_evidence_prefix_from_existing_lines,
    _extract_anchor_locator_from_hint,
    _extract_period_value_from_hint,
    _extract_rows_from_hint,
    _has_anchor_rewrite_candidates,
    _rewrite_evidence_lines_and_collect_resolved_anchor_issues,
    _rewrite_evidence_lines_for_confirmed_anchor_issues,
    _rewrite_financial_statement_evidence_line,
    _validate_anchor_rewrite_postconditions,
)
from dayu.services.internal.write_pipeline.audit_formatting import (
    _build_current_visible_headings_block,
    _collect_all_evidence_items,
    _extract_evidence_section_block,
    _extract_json_text,
    _extract_markdown_content,
    _find_all_occurrences,
    _find_enclosing_heading_section,
    _find_markdown_section_span,
    _find_normalized_line_like_spans,
    _find_normalized_match_spans,
    _find_normalized_paragraph_spans,
    _has_evidence_section,
    _matches_skeleton_structure,
    _normalize_chapter_markdown_for_audit,
    _normalize_evidence_location_segment,
    _normalize_heading_text,
    _replace_evidence_section_block,
    _should_run_fix_placeholders,
    _strip_evidence_section,
    _strip_generated_parenthetical_summary,
)
from dayu.services.internal.write_pipeline.audit_rules import (
    ConfirmOutputError,
    RepairOutputError,
    _collect_confirmable_evidence_violations,
    _merge_confirmed_evidence_results,
    _normalize_audit_violations,
    _parse_audit_decision,
    _parse_evidence_confirmation_result,
    _recompute_audit_result,
    _run_programmatic_audits,
)
from dayu.services.internal.write_pipeline.artifact_store import (
    _read_manifest_from_dir,
)
from dayu.services.internal.write_pipeline.chapter_execution_coordinator import build_process_state_template
from dayu.services.internal.write_pipeline.pipeline import (
    ChapterExecutionState,
    WritePipelineRunner,
    _build_middle_tasks,
    _build_overview_dependency_tasks,
    _build_signature,
    _log_write_pipeline_config,
    print_write_report,
    run_write_pipeline,
)
from dayu.services.internal.write_pipeline.prompt_builder import (
    _build_prior_decision_tasks,
)
from dayu.services.internal.write_pipeline.repair_executor import (
    _apply_repair_plan,
    _apply_repair_plan_with_details,
    _parse_repair_plan,
)
from dayu.services.internal.write_pipeline.scene_contract_preparer import (
    SceneContractPreparer,
    SceneAgentCreationError,
)
from dayu.services.scene_execution_acceptance import AcceptedSceneExecution, SceneExecutionAcceptancePreparer
from dayu.services.internal.write_pipeline.scene_executor import (
    _LLM_RETRY_DELAY_SECONDS,
    _LLM_RETRY_LIMIT,
)
from dayu.services.internal.write_pipeline.company_facets import (
    filter_chapter_contract_by_facets,
    filter_item_rules_by_facets,
    parse_company_facets,
)
from dayu.services.internal.write_pipeline.enums import (
    AuditCategory,
    AuditRuleCode,
    EvidenceConfirmationStatus,
    RepairResolutionMode,
    RepairStrategy,
    RepairTargetKind,
    build_audit_scope_rules_payload,
    normalize_audit_rule_code,
)
from dayu.services.internal.write_pipeline.models import (
    AuditDecision,
    ChapterResult,
    ChapterTask,
    CompanyFacetProfile,
    EvidenceAnchorFix,
    EvidenceConfirmationEntry,
    EvidenceConfirmationResult,
    MissingEvidenceSlot,
    RemediationAction,
    RepairContract,
    RunManifest,
    SceneModelConfig,
    Violation,
    WriteRunConfig,
)
from dayu.services.internal.write_pipeline.template_parser import parse_template_layout


def _build_toolset_configs(
    *,
    doc_tool_limits: DocToolLimits | None = None,
    fins_tool_limits: FinsToolLimits | None = None,
    web_tools_config: WebToolsConfig | None = None,
) -> tuple[ToolsetConfigSnapshot, ...]:
    """构造测试用通用 toolset 配置快照。"""

    return tuple(
        snapshot
        for snapshot in (
            build_toolset_config_snapshot("doc", doc_tool_limits),
            build_toolset_config_snapshot("fins", fins_tool_limits),
            build_toolset_config_snapshot("web", web_tools_config),
        )
        if snapshot is not None
    )

_DEFAULT_SCENE_MODELS = {
    "write": SceneModelConfig(name="deepseek-chat", temperature=0.7),
    "infer": SceneModelConfig(name="deepseek-thinking", temperature=0.7),
    "overview": SceneModelConfig(name="deepseek-chat", temperature=0.7),
    "decision": SceneModelConfig(name="deepseek-thinking", temperature=0.7),
    "regenerate": SceneModelConfig(name="deepseek-chat", temperature=0.7),
    "fix": SceneModelConfig(name="deepseek-chat", temperature=0.7),
    "repair": SceneModelConfig(name="deepseek-chat", temperature=0.7),
    "audit": SceneModelConfig(name="deepseek-thinking", temperature=0.7),
    "confirm": SceneModelConfig(name="deepseek-thinking", temperature=0.7),
}


def _build_scene_models(
    raw: dict[str, SceneModelConfig] | dict[str, tuple[str, float]] | dict[str, str] | None = None,
) -> dict[str, SceneModelConfig]:
    """构建测试用 scene 模型配置映射。"""

    if raw is None:
        return {scene_name: SceneModelConfig(name=config.name, temperature=config.temperature) for scene_name, config in _DEFAULT_SCENE_MODELS.items()}
    scene_models: dict[str, SceneModelConfig] = {}
    for scene_name, value in raw.items():
        if isinstance(value, SceneModelConfig):
            model_name, temperature = value.name, value.temperature
        elif isinstance(value, tuple):
            model_name, temperature = value
        else:
            model_name, temperature = value, 0.7
        scene_models[scene_name] = SceneModelConfig(name=model_name, temperature=temperature)
    return scene_models


def _build_test_write_config(
    tmp_path: Path,
    *,
    scene_models: dict[str, SceneModelConfig] | dict[str, tuple[str, float]] | dict[str, str] | None = None,
    write_model_override_name: str = "",
    audit_model_override_name: str = "",
    chapter_filter: str = "",
    fast: bool = False,
    force: bool = False,
    infer: bool = False,
) -> WriteRunConfig:
    """构建测试用写作配置。

    Args:
        tmp_path: pytest 临时目录。
        scene_models: scene 生效模型映射。
        write_model_override_name: 主写作模型覆盖名。
        audit_model_override_name: 审计模型覆盖名。
        chapter_filter: 单章过滤条件。
        fast: 是否启用 fast 模式。
        force: 是否启用 force 模式。

    Returns:
        测试用写作配置。

    Raises:
        无。
    """

    return WriteRunConfig(
        ticker="AAPL",
        company="Apple",
        template_path=str(tmp_path / "template.md"),
        output_dir=str(tmp_path / "out"),
        write_max_retries=2,
        web_provider="auto",
        resume=True,
        write_model_override_name=write_model_override_name,
        audit_model_override_name=audit_model_override_name,
        scene_models=_build_scene_models(scene_models),
        chapter_filter=chapter_filter,
        fast=fast,
        force=force,
        infer=infer,
    )


class _FakeWorkspaceResources:
    """测试用工作区稳定资源。"""

    def __init__(self, workspace_dir: Path) -> None:
        """初始化测试工作区资源。"""

        self.workspace_dir = workspace_dir
        self.config_root = workspace_dir / "config"
        self.output_dir = workspace_dir / "output"
        self.config_loader = _FakeConfigLoader()
        self.prompt_asset_store = _FakePromptAssetStore()


def _build_test_workspace_resources(workspace_dir: Path) -> WorkspaceResourcesProtocol:
    """构造测试用工作区稳定资源。"""

    return cast(WorkspaceResourcesProtocol, _FakeWorkspaceResources(workspace_dir))


def _build_test_model_config(model_name: str) -> ModelConfig:
    """构造测试用最小模型配置。"""

    return cast(
        ModelConfig,
        {
            "name": model_name,
            "model": model_name,
            "runner_type": "openai_compatible",
            "endpoint_url": "http://example.com",
            "headers": {},
            "max_context_tokens": 131072,
            "runtime_hints": {
                "temperature_profiles": {
                    scene_name: {"temperature": 0.7}
                    for scene_name in (
                        "write",
                        "infer",
                        "regenerate",
                        "decision",
                        "overview",
                        "fix",
                        "repair",
                        "audit",
                        "confirm",
                    )
                }
            },
        },
    )


def _build_test_resolved_options(
    trace_dir: Path,
    *,
    web_provider: str = "auto",
    trace_enabled: bool = False,
    max_iterations: int = 16,
) -> ResolvedExecutionOptions:
    """构造测试用 resolved execution options。"""

    return ResolvedExecutionOptions(
        model_name="deepseek-chat",
        runner_running_config=OpenAIRunnerRuntimeConfig(),
        agent_running_config=AgentRuntimeConfig(max_iterations=max_iterations),
        toolset_configs=_build_toolset_configs(
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider=web_provider),
        ),
        trace_settings=TraceSettings(enabled=trace_enabled, output_dir=trace_dir),
        temperature=0.7,
        conversation_memory_settings=ConversationMemorySettings(),
    )


def _as_scene_execution_acceptance_preparer(
    provider: "_FakeAgentProvider",
) -> SceneExecutionAcceptancePreparer:
    """把测试 provider 收窄为 scene 接受准备器类型。"""

    return cast(SceneExecutionAcceptancePreparer, provider)


def _build_test_host_executor() -> HostExecutorProtocol:
    """构造测试用宿主执行器协议视图。"""

    return cast(HostExecutorProtocol, _FakeHostExecutor())


def _build_test_pipeline_runner(tmp_path: Path) -> WritePipelineRunner:
    """构建最小可调用的写作流水线 runner。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        测试用 WritePipelineRunner。

    Raises:
        无。
    """

    workspace_config = _build_test_workspace_resources(tmp_path)
    resolved_options = _build_test_resolved_options(tmp_path / "tool_trace")
    return WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=resolved_options,
        write_config=_build_test_write_config(tmp_path),
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(_FakeAgentProvider()),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
    )


@pytest.mark.unit
def test_parse_company_facets_allows_more_than_three_controlled_tags() -> None:
    """验证公司级 facet 解析不再限制受控标签数量上限。"""

    raw_text = json.dumps(
        {
            "business_model_tags": ["A", "B", "C", "D"],
            "constraint_tags": ["X", "Y", "Z", "W"],
            "judgement_notes": "允许多标签",
        },
        ensure_ascii=False,
    )

    profile = parse_company_facets(
        raw_text,
        facet_catalog={
            "business_model_candidates": ["A", "B", "C", "D"],
            "constraint_candidates": ["X", "Y", "Z", "W"],
        },
    )

    assert profile.primary_facets == ["A", "B", "C", "D"]
    assert profile.cross_cutting_facets == ["X", "Y", "Z", "W"]


@pytest.mark.unit
def test_filter_chapter_contract_requires_business_model_match_before_constraint_match() -> None:
    """验证带业务类型约束的 lens 不会仅因横切约束而被错误命中。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    company_facets = CompanyFacetProfile(
        primary_facets=["硬件/消费电子", "平台互联网"],
        cross_cutting_facets=["监管敏感", "数据/隐私敏感"],
    )
    contract = ChapterContract(
        preferred_lens=[
            PreferredLens(
                lens="硬件生意优先看品牌、生态和定价权。",
                priority="core",
                facets_any=["硬件/消费电子", "品牌效应明显"],
            ),
            PreferredLens(
                lens="医药业务优先看支付覆盖和监管审批。",
                priority="core",
                facets_any=["医疗服务", "监管敏感"],
            ),
            PreferredLens(
                lens="只要监管环境变化，就先看制度约束是否改写判断。",
                priority="supporting",
                facets_any=["监管敏感"],
            ),
            PreferredLens(
                lens="先找一个最关键的行业共同规律。",
                priority="core",
            ),
        ]
    )
    facet_catalog = {
        "business_model_candidates": ["硬件/消费电子", "平台互联网", "医疗服务"],
        "constraint_candidates": ["监管敏感", "数据/隐私敏感", "品牌效应明显"],
    }

    filtered = filter_chapter_contract_by_facets(contract, company_facets, facet_catalog)
    filtered_texts = [item.lens for item in filtered.preferred_lens]

    assert "硬件生意优先看品牌、生态和定价权。" in filtered_texts
    assert "医药业务优先看支付覆盖和监管审批。" not in filtered_texts
    assert "只要监管环境变化，就先看制度约束是否改写判断。" in filtered_texts
    assert "先找一个最关键的行业共同规律。" in filtered_texts


@pytest.mark.unit
def test_filter_chapter_contract_keeps_all_generic_lenses_then_adds_up_to_six_matched() -> None:
    """验证章节合同先完整保留通用主干 lens，再补最多六条命中的业务/约束 lens。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    company_facets = CompanyFacetProfile(
        primary_facets=["平台互联网"],
        cross_cutting_facets=["监管敏感"],
    )
    contract = ChapterContract(
        preferred_lens=[
            PreferredLens(lens="通用主干一", priority="core"),
            PreferredLens(lens="平台命中一", priority="core", facets_any=["平台互联网"]),
            PreferredLens(lens="通用主干二", priority="supporting"),
            PreferredLens(lens="平台命中二", priority="core", facets_any=["平台互联网", "监管敏感"]),
            PreferredLens(lens="平台命中三", priority="core", facets_any=["平台互联网"]),
            PreferredLens(lens="平台命中四", priority="core", facets_any=["平台互联网"]),
            PreferredLens(lens="平台命中五", priority="supporting", facets_any=["平台互联网"]),
            PreferredLens(lens="平台命中六", priority="supporting", facets_any=["平台互联网"]),
            PreferredLens(lens="平台命中七", priority="supporting", facets_any=["平台互联网"]),
        ]
    )
    facet_catalog = {
        "business_model_candidates": ["平台互联网"],
        "constraint_candidates": ["监管敏感"],
    }

    filtered = filter_chapter_contract_by_facets(contract, company_facets, facet_catalog)
    filtered_texts = [item.lens for item in filtered.preferred_lens]

    assert filtered_texts[:2] == ["通用主干一", "通用主干二"]
    assert filtered_texts[2:] == [
        "平台命中一",
        "平台命中二",
        "平台命中三",
        "平台命中四",
        "平台命中五",
        "平台命中六",
    ]
    assert "平台命中七" not in filtered_texts


@pytest.mark.unit
def test_filter_item_rules_requires_business_model_match_before_constraint_match() -> None:
    """验证带业务类型约束的 item rule 不会仅因横切约束而被错误命中。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    company_facets = CompanyFacetProfile(
        primary_facets=["硬件/消费电子", "平台互联网"],
        cross_cutting_facets=["监管敏感", "数据/隐私敏感"],
    )
    item_rules = [
        ItemRule(
            mode="conditional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="品牌效应、规模效应或定价权是否正在支撑位置判断",
            when="只要有稳定披露且它会改写位置判断就写",
            facets_any=["硬件/消费电子", "品牌效应明显"],
        ),
        ItemRule(
            mode="conditional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="医药业务的支付覆盖与审批约束",
            when="只要有稳定披露且它会改写位置判断就写",
            facets_any=["医疗服务", "监管敏感"],
        ),
        ItemRule(
            mode="optional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="监管变化是否正在抬高制度约束",
            when="只要有稳定披露且它会放大风险就写",
            facets_any=["监管敏感"],
        ),
        ItemRule(
            mode="optional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="必要同业对比",
            when="只要稳定披露能改写判断就写",
        ),
    ]
    facet_catalog = {
        "business_model_candidates": ["硬件/消费电子", "平台互联网", "医疗服务"],
        "constraint_candidates": ["监管敏感", "数据/隐私敏感", "品牌效应明显"],
    }

    filtered = filter_item_rules_by_facets(item_rules, company_facets, facet_catalog)
    filtered_items = [item.item for item in filtered]

    assert "品牌效应、规模效应或定价权是否正在支撑位置判断" in filtered_items
    assert "医药业务的支付覆盖与审批约束" not in filtered_items
    assert "监管变化是否正在抬高制度约束" in filtered_items
    assert "必要同业对比" in filtered_items


@pytest.mark.unit
def test_filter_item_rules_keeps_all_generic_rules_then_adds_matched_by_mode() -> None:
    """验证 ITEM_RULE 先完整保留通用主干规则，再补命中的 conditional/optional。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    company_facets = CompanyFacetProfile(
        primary_facets=["平台互联网"],
        cross_cutting_facets=["监管敏感"],
    )
    item_rules = [
        ItemRule(
            mode="conditional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="通用条件一",
            when="主干",
        ),
        ItemRule(
            mode="conditional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="平台条件一",
            when="命中",
            facets_any=["平台互联网"],
        ),
        ItemRule(
            mode="conditional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="通用条件二",
            when="主干",
        ),
        ItemRule(
            mode="conditional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="平台条件二",
            when="命中",
            facets_any=["平台互联网", "监管敏感"],
        ),
        ItemRule(
            mode="conditional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="平台条件三",
            when="命中",
            facets_any=["平台互联网"],
        ),
        ItemRule(
            mode="conditional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="平台条件四",
            when="命中",
            facets_any=["平台互联网"],
        ),
        ItemRule(
            mode="conditional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="平台条件五",
            when="命中",
            facets_any=["平台互联网"],
        ),
        ItemRule(
            mode="optional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="通用可选一",
            when="主干",
        ),
        ItemRule(
            mode="optional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="平台可选一",
            when="命中",
            facets_any=["平台互联网"],
        ),
        ItemRule(
            mode="optional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="通用可选二",
            when="主干",
        ),
        ItemRule(
            mode="optional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="平台可选二",
            when="命中",
            facets_any=["平台互联网"],
        ),
        ItemRule(
            mode="optional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="平台可选三",
            when="命中",
            facets_any=["平台互联网"],
        ),
        ItemRule(
            mode="optional",
            target_heading="这门生意值不值得看，公司站在什么位置",
            item="平台可选四",
            when="命中",
            facets_any=["平台互联网"],
        ),
    ]
    facet_catalog = {
        "business_model_candidates": ["平台互联网"],
        "constraint_candidates": ["监管敏感"],
    }

    filtered = filter_item_rules_by_facets(item_rules, company_facets, facet_catalog)
    filtered_items = [item.item for item in filtered]

    assert filtered_items == [
        "通用条件一",
        "通用条件二",
        "平台条件一",
        "平台条件二",
        "平台条件三",
        "平台条件四",
        "通用可选一",
        "通用可选二",
        "平台可选一",
        "平台可选二",
        "平台可选三",
    ]
    assert "平台条件五" not in filtered_items
    assert "平台可选四" not in filtered_items


@pytest.mark.unit
def test_log_write_pipeline_config_uses_resolved_agent_iterations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证写作流水线参数摘要打印当前已解析运行配置中的迭代上限。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    logs: list[str] = []

    def _capture_log(message: object, **_kwargs: object) -> None:
        """收集日志文本。

        Args:
            message: 日志消息对象。
            **_kwargs: 关键字参数。

        Returns:
            无。

        Raises:
            无。
        """

        logs.append(str(message))

    write_config = WriteRunConfig(
        ticker="ATAT",
        company="ATAT",
        template_path=str(tmp_path / "template.md"),
        output_dir=str(tmp_path / "draft"),
        write_max_retries=2,
        web_provider="duckduckgo",
        resume=True,
        scene_models=_build_scene_models(),
        chapter_filter="公司做的是什么生意",
        fast=True,
        force=True,
    )
    resolved_options = ResolvedExecutionOptions(
        model_name="deepseek-chat",
        runner_running_config=OpenAIRunnerRuntimeConfig(tool_timeout_seconds=90.0),
        agent_running_config=AgentRuntimeConfig(max_iterations=16),
        toolset_configs=_build_toolset_configs(
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider="duckduckgo"),
        ),
        trace_settings=TraceSettings(enabled=True, output_dir=tmp_path / "trace"),
        temperature=0.7,
    )
    monkeypatch.setattr(Log, "info", _capture_log)

    _log_write_pipeline_config(
        write_config=write_config,
        resolved_options=resolved_options,
    )

    assert any("写作流水线参数摘要" in item for item in logs)
    assert any(
        "chapter=公司做的是什么生意, resume=True, write_max_retries=2, fast=True, force=True" in item
        for item in logs
    )
    assert any("agent_max_iterations=16" in item for item in logs)
    assert any("tool_timeout_seconds=90.0" in item for item in logs)


class _FakePromptAssetStore:
    """测试用 prompt 资产仓储。"""

    def load_scene_manifest(self, scene_name: str) -> SceneManifestAsset:
        """返回 scene manifest。"""

        mapping = {
            "write": {
                "scene": "write",
                "model": {
                    "default_name": "deepseek-chat",
                    "allowed_names": ["deepseek-chat"],
                    "temperature_profile": "write",
                },
                "version": "v1",
                "description": "write",
                "extends": [],
                "tool_selection": {"mode": "select", "tool_tags_any": ["fins", "web"]},
                "defaults": {"missing_fragment_policy": "error"},
                "fragments": [
                    {"id": "base_agents", "type": "AGENTS", "path": "base/agents.md", "order": 100},
                    {"id": "base_tools", "type": "TOOLS", "path": "base/tools.md", "order": 150},
                    {"id": "base_soul", "type": "SOUL", "path": "base/soul.md", "order": 200},
                    {"id": "base_fact_rules", "type": "SOUL", "path": "base/fact_rules.md", "order": 250},
                    {"id": "base_user", "type": "USER", "path": "base/user.md", "order": 300},
                    {"id": "write_scene", "type": "SCENE", "path": "scenes/write.md", "order": 900},
                ],
            },
            "infer": {
                "scene": "infer",
                "model": {
                    "default_name": "deepseek-thinking",
                    "allowed_names": ["deepseek-thinking"],
                    "temperature_profile": "infer",
                },
                "version": "v1",
                "description": "infer",
                "extends": [],
                "tool_selection": {"mode": "select", "tool_tags_any": ["fins"]},
                "context_slots": ["fins_default_subject"],
                "defaults": {"missing_fragment_policy": "error"},
                "fragments": [
                    {"id": "base_agents", "type": "AGENTS", "path": "base/agents.md", "order": 100},
                    {"id": "base_tools", "type": "TOOLS", "path": "base/tools.md", "order": 150},
                    {"id": "base_soul", "type": "SOUL", "path": "base/soul.md", "order": 200},
                    {"id": "base_fact_rules", "type": "SOUL", "path": "base/fact_rules.md", "order": 250},
                    {"id": "infer_scene", "type": "SCENE", "path": "scenes/infer.md", "order": 300},
                    {"id": "fins_default_subject", "type": "USER", "path": "capabilities/fins_default_subject.md", "order": 350},
                ],
            },
            "regenerate": {
                "scene": "regenerate",
                "model": {
                    "default_name": "deepseek-chat",
                    "allowed_names": ["deepseek-chat"],
                    "temperature_profile": "write",
                },
                "version": "v1",
                "description": "regenerate",
                "extends": [],
                "tool_selection": {"mode": "select", "tool_tags_any": ["fins", "web"]},
                "defaults": {"missing_fragment_policy": "error"},
                "fragments": [
                    {"id": "base_agents", "type": "AGENTS", "path": "base/agents.md", "order": 100},
                    {"id": "base_tools", "type": "TOOLS", "path": "base/tools.md", "order": 150},
                    {"id": "base_soul", "type": "SOUL", "path": "base/soul.md", "order": 200},
                    {"id": "base_fact_rules", "type": "SOUL", "path": "base/fact_rules.md", "order": 250},
                    {"id": "base_user", "type": "USER", "path": "base/user.md", "order": 300},
                    {"id": "regenerate_scene", "type": "SCENE", "path": "scenes/regenerate.md", "order": 900},
                ],
            },
            "decision": {
                "scene": "decision",
                "model": {
                    "default_name": "deepseek-thinking",
                    "allowed_names": ["deepseek-thinking"],
                    "temperature_profile": "decision",
                },
                "version": "v1",
                "description": "decision",
                "extends": [],
                "tool_selection": {"mode": "select", "tool_tags_any": ["fins", "web"]},
                "defaults": {"missing_fragment_policy": "error"},
                "fragments": [
                    {"id": "base_agents", "type": "AGENTS", "path": "base/agents.md", "order": 100},
                    {"id": "base_tools", "type": "TOOLS", "path": "base/tools.md", "order": 150},
                    {"id": "base_soul", "type": "SOUL", "path": "base/soul.md", "order": 200},
                    {"id": "base_fact_rules", "type": "SOUL", "path": "base/fact_rules.md", "order": 250},
                    {"id": "base_user", "type": "USER", "path": "base/user.md", "order": 300},
                    {"id": "decision_scene", "type": "SCENE", "path": "scenes/decision.md", "order": 900},
                ],
            },
            "overview": {
                "scene": "overview",
                "model": {
                    "default_name": "deepseek-chat",
                    "allowed_names": ["deepseek-chat"],
                    "temperature_profile": "overview",
                },
                "version": "v1",
                "description": "overview",
                "extends": [],
                "tool_selection": {"mode": "none"},
                "defaults": {"missing_fragment_policy": "error"},
                "fragments": [
                    {"id": "base_agents", "type": "AGENTS", "path": "base/agents.md", "order": 100},
                    {"id": "base_soul", "type": "SOUL", "path": "base/soul.md", "order": 200},
                    {"id": "base_fact_rules", "type": "SOUL", "path": "base/fact_rules.md", "order": 250},
                    {"id": "base_user", "type": "USER", "path": "base/user.md", "order": 300},
                    {"id": "overview_scene", "type": "SCENE", "path": "scenes/overview.md", "order": 900},
                ],
            },
            "fix": {
                "scene": "fix",
                "model": {
                    "default_name": "deepseek-chat",
                    "allowed_names": ["deepseek-chat"],
                    "temperature_profile": "write",
                },
                "version": "v1",
                "description": "fix",
                "extends": [],
                "tool_selection": {"mode": "select", "tool_tags_any": ["fins", "web"]},
                "defaults": {"missing_fragment_policy": "error"},
                "fragments": [
                    {"id": "base_agents", "type": "AGENTS", "path": "base/agents.md", "order": 100},
                    {"id": "base_tools", "type": "TOOLS", "path": "base/tools.md", "order": 150},
                    {"id": "base_soul", "type": "SOUL", "path": "base/soul.md", "order": 200},
                    {"id": "base_fact_rules", "type": "SOUL", "path": "base/fact_rules.md", "order": 250},
                    {"id": "base_user", "type": "USER", "path": "base/user.md", "order": 300},
                    {"id": "fix_scene", "type": "SCENE", "path": "scenes/fix.md", "order": 900},
                ],
            },
            "repair": {
                "scene": "repair",
                "model": {
                    "default_name": "deepseek-chat",
                    "allowed_names": ["deepseek-chat"],
                    "temperature_profile": "write",
                },
                "version": "v1",
                "description": "repair",
                "extends": [],
                "tool_selection": {"mode": "none"},
                "defaults": {"missing_fragment_policy": "error"},
                "fragments": [
                    {"id": "base_agents", "type": "AGENTS", "path": "base/agents.md", "order": 100},
                    {"id": "base_soul", "type": "SOUL", "path": "base/soul.md", "order": 200},
                    {"id": "base_fact_rules", "type": "SOUL", "path": "base/fact_rules.md", "order": 250},
                    {"id": "base_user", "type": "USER", "path": "base/user.md", "order": 300},
                    {"id": "repair_scene", "type": "SCENE", "path": "scenes/repair.md", "order": 900}
                ],
            },
            "audit": {
                "scene": "audit",
                "model": {
                    "default_name": "deepseek-thinking",
                    "allowed_names": ["deepseek-thinking"],
                    "temperature_profile": "audit",
                },
                "version": "v1",
                "description": "audit",
                "extends": [],
                "tool_selection": {"mode": "none"},
                "defaults": {"missing_fragment_policy": "error"},
                "fragments": [
                    {"id": "base_agents", "type": "AGENTS", "path": "base/agents.md", "order": 100},
                    {"id": "base_fact_rules", "type": "SOUL", "path": "base/fact_rules.md", "order": 250},
                    {"id": "audit_scene", "type": "SCENE", "path": "scenes/audit.md", "order": 900}
                ],
            },
            "confirm": {
                "scene": "confirm",
                "model": {
                    "default_name": "deepseek-thinking",
                    "allowed_names": ["deepseek-thinking"],
                    "temperature_profile": "audit",
                },
                "version": "v1",
                "description": "confirm",
                "extends": [],
                "tool_selection": {"mode": "select", "tool_tags_any": ["fins", "web"]},
                "defaults": {"missing_fragment_policy": "error"},
                "fragments": [
                    {"id": "base_agents", "type": "AGENTS", "path": "base/agents.md", "order": 100},
                    {"id": "base_fact_rules", "type": "SOUL", "path": "base/fact_rules.md", "order": 250},
                    {"id": "base_tools", "type": "TOOLS", "path": "base/tools.md", "order": 400},
                    {"id": "confirm_scene", "type": "SCENE", "path": "scenes/confirm.md", "order": 900}
                ],
            }
        }
        return cast(SceneManifestAsset, mapping[scene_name])

    def load_fragment_template(self, fragment_path: str, *, required: bool = True) -> str | None:
        """返回 scene fragment 文本。

        Args:
            fragment_path: fragment 相对路径。
            required: 是否必填。

        Returns:
            对应文本；当 optional 资产不存在时返回 ``None``。

        Raises:
            FileNotFoundError: 必填资产不存在时抛出。
        """

        mapping = {
            "base/agents.md": "你必须严格输出目标格式。",
            "base/soul.md": "证据与出处必须服从统一格式。",
            "base/fact_rules.md": "事实与分析标注、前瞻性表述与证据格式规则。",
            "base/tools.md": "工具使用指引。",
            "base/user.md": "{{current_date_context}}",
            "capabilities/fins_default_subject.md": "# 当前分析对象\n{{ticker_context}}",
            "scenes/audit.md": "你是财报章节审计助手。",
            "scenes/confirm.md": "你是财报章节证据复核助手。",
            "scenes/repair.md": "你是财报章节局部修复助手。",
            "scenes/write.md": "你是财报章节写作助手。",
            "scenes/infer.md": "你是公司业务类型与关键约束判断助手。",
            "scenes/regenerate.md": "你是财报章节整章重建助手。",
            "scenes/decision.md": "你是财报章节研究决策综合助手。",
            "scenes/overview.md": "你是第0章投资要点概览封面页助手。",
            "scenes/fix.md": "你是财报章节占位符补强助手。",
        }
        if fragment_path in mapping:
            return mapping[fragment_path]
        if required:
            raise FileNotFoundError(fragment_path)
        return None

    def load_task_prompt(self, task_name: str) -> str:
        """返回简化 task prompt 模板。

        Args:
            task_name: task prompt 名称。

        Returns:
            模板文本。

        Raises:
            KeyError: 名称非法时抛出。
        """

        mapping: dict[str, str] = {
            "repair_chapter": "{{chapter}} {{chapter_contract_block}} {{last_repair_contract_block}} {{current_visible_headings_block}} {{last_wrote_content_block}}",
            "regenerate_chapter": "{{chapter}} {{company_facets_summary_block}} {{report_goal}} {{audience_profile}} {{chapter_goal}} {{skeleton_block}} {{chapter_contract_block}} {{item_rules_block}} {{last_audit_json_block}} {{last_wrote_content_block}}",
            "fix_placeholders": "{{chapter}} {{chapter_markdown_block}}",
            "audit_facts_tone_json": "{{audit_mode}} {{chapter_markdown_block}} {{skeleton_block}} {{chapter_contract_block}} {{item_rules_block}} {{audit_scope_rules_block}} {{repair_contract_block}}",
            "confirm_evidence_violations": "{{chapter_markdown_block}} {{suspected_evidence_violations_block}} {{evidence_items_block}}",
            "write_chapter": "{{chapter}} {{company_facets_summary_block}} {{report_goal}} {{audience_profile}} {{chapter_goal}} {{skeleton_block}} {{chapter_contract_block}} {{item_rules_block}}",
            "infer_company_facets": "{{company_meta_block}} {{business_model_candidates_block}} {{constraint_candidates_block}}",
            "write_research_decision": "{{chapter}} {{company_facets_summary_block}} {{report_goal}} {{audience_profile}} {{chapter_goal}} {{decision_source_of_truth}} {{decision_allow_new_facts}} {{decision_allow_new_sources}} {{skeleton_block}} {{chapter_contract_block}} {{item_rules_block}} {{prior_chapters_input_block}}",
            "fill_overview": "{{chapter}} {{company_facets_summary_block}} {{report_goal}} {{audience_profile}} {{chapter_goal}} {{chapter_contract_block}} {{prior_chapters_summary_input_block}} {{overview_source_of_truth}}",
        }
        return mapping[task_name]

    def load_task_prompt_contract(self, task_name: str) -> TaskPromptContractAsset:
        """返回简化 task prompt sidecar contract。

        Args:
            task_name: task prompt 名称。

        Returns:
            contract 原始字典。

        Raises:
            KeyError: 名称非法时抛出。
        """

        mapping: dict[str, TaskPromptContractAsset] = {
            "write_chapter": {
                "prompt_name": "write_chapter",
                "version": "v1",
                "inputs": [
                    {"name": "chapter", "type": "scalar", "required": True},
                    {"name": "company", "type": "scalar", "required": True},
                    {"name": "ticker", "type": "scalar", "required": True},
                    {"name": "company_facets_summary", "type": "markdown_block", "required": True},
                    {"name": "report_goal", "type": "scalar", "required": True},
                    {"name": "audience_profile", "type": "scalar", "required": True},
                    {"name": "chapter_goal", "type": "scalar", "required": True},
                    {"name": "skeleton", "type": "markdown_block", "required": True},
                    {"name": "chapter_contract", "type": "json_block", "required": True},
                    {"name": "item_rules", "type": "json_block", "required": False},
                ],
            },
            "repair_chapter": {
                "prompt_name": "repair_chapter",
                "version": "v1",
                "inputs": [
                    {"name": "chapter", "type": "scalar", "required": True},
                    {"name": "company", "type": "scalar", "required": True},
                    {"name": "ticker", "type": "scalar", "required": True},
                    {"name": "allow_new_facts", "type": "scalar", "required": True},
                    {"name": "retry_scope", "type": "scalar", "required": True},
                    {"name": "chapter_contract", "type": "json_block", "required": True},
                    {"name": "last_repair_contract", "type": "json_block", "required": True},
                    {"name": "current_visible_headings", "type": "markdown_block", "required": True},
                    {"name": "last_wrote_content", "type": "markdown_block", "required": True},
                ],
            },
            "regenerate_chapter": {
                "prompt_name": "regenerate_chapter",
                "version": "v1",
                "inputs": [
                    {"name": "chapter", "type": "scalar", "required": True},
                    {"name": "company", "type": "scalar", "required": True},
                    {"name": "ticker", "type": "scalar", "required": True},
                    {"name": "company_facets_summary", "type": "markdown_block", "required": True},
                    {"name": "report_goal", "type": "scalar", "required": True},
                    {"name": "audience_profile", "type": "scalar", "required": True},
                    {"name": "chapter_goal", "type": "scalar", "required": True},
                    {"name": "allow_new_facts", "type": "scalar", "required": True},
                    {"name": "retry_scope", "type": "scalar", "required": True},
                    {"name": "skeleton", "type": "markdown_block", "required": True},
                    {"name": "chapter_contract", "type": "json_block", "required": True},
                    {"name": "item_rules", "type": "json_block", "required": False},
                    {"name": "last_audit_json_block", "type": "markdown_block", "required": True},
                    {"name": "last_repair_contract", "type": "json_block", "required": True},
                    {"name": "last_wrote_content", "type": "markdown_block", "required": True},
                ],
            },
            "infer_company_facets": {
                "prompt_name": "infer_company_facets",
                "version": "v1",
                "inputs": [
                    {"name": "company_meta", "type": "json_block", "required": True},
                    {"name": "business_model_candidates", "type": "json_block", "required": True},
                    {"name": "constraint_candidates", "type": "json_block", "required": True},
                ],
            },
            "fill_overview": {
                "prompt_name": "fill_overview",
                "version": "v1",
                "inputs": [
                    {"name": "chapter", "type": "scalar", "required": True},
                    {"name": "company", "type": "scalar", "required": True},
                    {"name": "ticker", "type": "scalar", "required": True},
                    {"name": "company_facets_summary", "type": "markdown_block", "required": True},
                    {"name": "report_goal", "type": "scalar", "required": True},
                    {"name": "audience_profile", "type": "scalar", "required": True},
                    {"name": "chapter_goal", "type": "scalar", "required": True},
                    {"name": "overview_source_of_truth", "type": "scalar", "required": True},
                    {"name": "overview_allow_new_facts", "type": "scalar", "required": True},
                    {"name": "overview_allow_new_sources", "type": "scalar", "required": True},
                    {"name": "skeleton", "type": "markdown_block", "required": True},
                    {"name": "chapter_contract", "type": "json_block", "required": True},
                    {"name": "prior_chapters_summary_input", "type": "markdown_block", "required": True},
                ],
            },
            "write_research_decision": {
                "prompt_name": "write_research_decision",
                "version": "v1",
                "inputs": [
                    {"name": "chapter", "type": "scalar", "required": True},
                    {"name": "company", "type": "scalar", "required": True},
                    {"name": "ticker", "type": "scalar", "required": True},
                    {"name": "company_facets_summary", "type": "markdown_block", "required": True},
                    {"name": "report_goal", "type": "scalar", "required": True},
                    {"name": "audience_profile", "type": "scalar", "required": True},
                    {"name": "chapter_goal", "type": "scalar", "required": True},
                    {"name": "decision_source_of_truth", "type": "scalar", "required": True},
                    {"name": "decision_allow_new_facts", "type": "scalar", "required": True},
                    {"name": "decision_allow_new_sources", "type": "scalar", "required": True},
                    {"name": "skeleton", "type": "markdown_block", "required": True},
                    {"name": "chapter_contract", "type": "json_block", "required": True},
                    {"name": "item_rules", "type": "json_block", "required": False},
                    {"name": "prior_chapters_input", "type": "markdown_block", "required": True},
                ],
            },
            "fix_placeholders": {
                "prompt_name": "fix_placeholders",
                "version": "v1",
                "inputs": [
                    {"name": "chapter", "type": "scalar", "required": True},
                    {"name": "company", "type": "scalar", "required": True},
                    {"name": "ticker", "type": "scalar", "required": True},
                    {"name": "fix_mode", "type": "scalar", "required": True},
                    {"name": "rewrite_compliant_content", "type": "scalar", "required": True},
                    {"name": "chapter_markdown", "type": "markdown_block", "required": True},
                ],
            },
            "audit_facts_tone_json": {
                "prompt_name": "audit_facts_tone_json",
                "version": "v1",
                "inputs": [
                    {"name": "company", "type": "scalar", "required": True},
                    {"name": "ticker", "type": "scalar", "required": True},
                    {"name": "audit_mode", "type": "scalar", "required": True},
                    {"name": "chapter_markdown", "type": "markdown_block", "required": True},
                    {"name": "skeleton", "type": "markdown_block", "required": True},
                    {"name": "chapter_contract", "type": "json_block", "required": True},
                    {"name": "item_rules", "type": "json_block", "required": True},
                    {"name": "audit_scope_rules", "type": "json_block", "required": True},
                    {"name": "repair_contract", "type": "json_block", "required": False},
                ],
            },
            "confirm_evidence_violations": {
                "prompt_name": "confirm_evidence_violations",
                "version": "v1",
                "inputs": [
                    {"name": "company", "type": "scalar", "required": True},
                    {"name": "ticker", "type": "scalar", "required": True},
                    {"name": "chapter_markdown", "type": "markdown_block", "required": True},
                    {"name": "suspected_evidence_violations", "type": "json_block", "required": True},
                    {"name": "evidence_items", "type": "list_block", "required": True},
                ],
            },
        }
        return mapping[task_name]


class _FakeConfigLoader:
    """测试用应用配置加载器。"""

    def __init__(self) -> None:
        """初始化测试配置加载器。"""

        self.model_env_vars: dict[str, tuple[str, ...]] = {}
        self.model_env_var_calls: list[tuple[str, ...]] = []

    def load_run_config(self) -> dict[str, Any]:
        """返回最小 run 配置。"""

        return {}

    def load_llm_models(self) -> dict[str, ModelConfig]:
        """返回全部测试模型配置。"""

        return {"deepseek-chat": _build_test_model_config("deepseek-chat")}

    def load_llm_model(self, model_name: str) -> ModelConfig:
        """返回最小模型配置。

        Args:
            model_name: 模型名。

        Returns:
            模型配置字典。

        Raises:
            无。
        """

        return _build_test_model_config(model_name)

    def load_toolset_registrars(self) -> dict[str, str]:
        """返回空的 registrar 清单。"""

        return {}

    def collect_model_referenced_env_vars(self, model_names: Iterable[str]) -> tuple[str, ...]:
        """返回指定测试模型声明的环境变量。"""

        normalized_model_names = tuple(str(model_name) for model_name in model_names)
        self.model_env_var_calls.append(normalized_model_names)
        referenced_env_vars: list[str] = []
        for model_name in normalized_model_names:
            referenced_env_vars.extend(self.model_env_vars.get(model_name, ()))
        return tuple(dict.fromkeys(referenced_env_vars))


class _FakeHostExecutor:
    """测试用宿主执行器。"""

    def __init__(self) -> None:
        """初始化测试宿主执行器。"""

        self.last_execution_contract = None

    async def run_agent_and_wait(self, execution_contract) -> AppResult:
        """返回固定结果，并记录最后一次契约。"""

        self.last_execution_contract = execution_contract
        return AppResult(content="ok", errors=[], warnings=[])


class _FakeAgentProvider:
    """测试用 scene 执行接受准备器。"""

    def __init__(
        self,
        *,
        trace_recorder_factory: object | None = None,
        prompt_agent: object | None = None,
    ) -> None:
        """初始化假 provider。"""

        self.trace_recorder_factory = trace_recorder_factory
        self.prompt_agent = prompt_agent if prompt_agent is not None else _FakePromptAgent()
        self.prepare_calls: list[dict[str, object]] = []

    def resolve_execution_options(
        self,
        scene_name: str,
        execution_options: ExecutionOptions | None = None,
    ) -> ResolvedExecutionOptions:
        """返回带当前 scene 生效模型名的 resolved options。"""

        base_resolved = _build_test_resolved_options(Path("/tmp/trace"))
        model_name = (
            str(execution_options.model_name)
            if execution_options is not None and execution_options.model_name
            else f"model-{scene_name}"
        )
        web_provider = (
            str(execution_options.web_provider)
            if execution_options is not None and execution_options.web_provider
            else "auto"
        )
        return ResolvedExecutionOptions(
            model_name=model_name,
            runner_running_config=base_resolved.runner_running_config,
            agent_running_config=base_resolved.agent_running_config,
            toolset_configs=_build_toolset_configs(
                doc_tool_limits=DocToolLimits(),
                fins_tool_limits=FinsToolLimits(),
                web_tools_config=WebToolsConfig(provider=web_provider),
            ),
            trace_settings=base_resolved.trace_settings,
            temperature=base_resolved.temperature,
            conversation_memory_settings=base_resolved.conversation_memory_settings,
        )

    def prepare(
        self,
        scene_name: str,
        execution_options: ExecutionOptions | None = None,
    ) -> AcceptedSceneExecution:
        """记录并返回预设 scene 解析结果。"""

        web_provider = "auto"
        if execution_options is not None and execution_options.web_provider:
            web_provider = str(execution_options.web_provider)

        self.prepare_calls.append(
            {
                "scene_name": scene_name,
                "execution_options": execution_options,
                "web_provider": web_provider,
            }
        )
        trace_dir = Path("/tmp/trace")
        resolved_execution_options = _build_test_resolved_options(
            trace_dir,
            web_provider=web_provider,
        )
        return AcceptedSceneExecution(
            scene_name=scene_name,
            scene_definition=load_scene_definition(_FakePromptAssetStore(), scene_name),
            resolved_execution_options=resolved_execution_options,
            model_config=_build_test_model_config(f"model-{scene_name}"),
            resolved_temperature=0.7,
            accepted_execution_spec=AcceptedExecutionSpec(
                model=AcceptedModelSpec(model_name=f"model-{scene_name}", temperature=0.7),
                runtime=AcceptedRuntimeSpec(
                    runner_running_config={},
                    agent_running_config={},
                ),
                tools=AcceptedToolConfigSpec(
                    toolset_configs=_build_toolset_configs(
                        doc_tool_limits=DocToolLimits(),
                        fins_tool_limits=FinsToolLimits(),
                        web_tools_config=WebToolsConfig(
                            provider=web_provider,
                            allow_private_network_url=False,
                        ),
                    ),
                ),
                infrastructure=AcceptedInfrastructureSpec(
                    trace_settings=TraceSettings(enabled=False, output_dir=trace_dir),
                    conversation_memory_settings=ConversationMemorySettings(),
                ),
            ),
        )


class _FakePromptAgent:
    """测试用 Prompt facade。"""

    def __init__(self, side_effects: list[AppResult] | None = None) -> None:
        """初始化测试 facade。"""

        self._side_effects = list(side_effects or [AppResult(content="ok", errors=[], warnings=[])])
        self.calls: list[dict[str, object]] = []

    async def stream(self, prepared_scene: object, turn_input: object):
        """按序输出应用层事件流。"""

        self.calls.append({"prepared_scene": prepared_scene, "turn_input": turn_input})
        if not self._side_effects:
            raise IndexError("sequence exhausted")
        result = self._side_effects.pop(0)
        for warning in result.warnings:
            yield AppEvent(type=AppEventType.WARNING, payload={"message": warning}, meta={})
        for error in result.errors:
            yield AppEvent(type=AppEventType.ERROR, payload={"message": error}, meta={})
        if result.content or not result.errors:
            yield AppEvent(
                type=AppEventType.FINAL_ANSWER,
                payload={"content": result.content, "degraded": result.degraded},
                meta={},
            )


class _FakeCancelledPromptAgent:
    """测试用取消 Prompt facade。"""

    def __init__(self, cancel_reason: str = "用户取消") -> None:
        """初始化取消 facade。

        Args:
            cancel_reason: 取消原因。
        """

        self._cancel_reason = cancel_reason
        self.calls: list[dict[str, object]] = []

    async def stream(self, prepared_scene: object, turn_input: object):
        """输出单个取消事件。

        Args:
            prepared_scene: 已准备 scene。
            turn_input: 当前轮输入。

        Yields:
            AppEvent: 单个取消事件。
        """

        self.calls.append({"prepared_scene": prepared_scene, "turn_input": turn_input})
        yield AppEvent(
            type=AppEventType.CANCELLED,
            payload={"cancel_reason": self._cancel_reason},
            meta={},
        )


class _SequenceCaller:
    """按顺序返回预设值的可调用对象。"""

    def __init__(self, values: Sequence[Any]) -> None:
        """初始化序列调用器。

        Args:
            values: 预设返回值序列。

        Returns:
            无。

        Raises:
            无。
        """

        self._values: list[Any] = list(values)

    def __call__(self, *_args: object, **_kwargs: object) -> Any:
        """返回下一个预设值。

        Args:
            *_args: 任意参数。
            **_kwargs: 任意参数。

        Returns:
            下一个预设值。

        Raises:
            IndexError: 序列耗尽时抛出。
        """

        if not self._values:
            raise IndexError("sequence exhausted")
        return self._values.pop(0)


class _AtomicWriteRecorder:
    """原子写记录器。"""

    def __init__(self) -> None:
        """初始化记录器。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.path: Path | None = None
        self.content: str = ""
        self.encoding: str = "utf-8"

    def __call__(self, path: Path, content: str, *, encoding: str = "utf-8") -> None:
        """记录一次原子写调用。

        Args:
            path: 目标文件路径。
            content: 待写入内容。
            encoding: 文本编码。

        Returns:
            无。

        Raises:
            无。
        """

        self.path = path
        self.content = content
        self.encoding = encoding


def _write_manifest(path: Path, manifest: RunManifest) -> None:
    """将 manifest 对象写入磁盘。

    Args:
        path: manifest 文件路径。
        manifest: manifest 对象。

    Returns:
        无。

    Raises:
        OSError: 写文件失败时抛出。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _build_runner(
    tmp_path: Path,
    *,
    company_name_resolver: Callable[[str], str] | None = None,
    company_meta_summary_resolver: Callable[[str], dict[str, str]] | None = None,
    infer: bool = False,
) -> WritePipelineRunner:
    """构建测试用写作执行器。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        写作执行器。

    Raises:
        OSError: 目录创建失败时抛出。
    """

    workspace = tmp_path / "workspace"
    (workspace / "portfolio" / "AAPL" / "filings").mkdir(parents=True, exist_ok=True)
    write_config = _build_test_write_config(tmp_path, infer=infer)
    workspace_config = _build_test_workspace_resources(workspace)
    running_config = _build_test_resolved_options(tmp_path / "trace")
    provider = _FakeAgentProvider()
    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=write_config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
        company_name_resolver=company_name_resolver,
        company_meta_summary_resolver=company_meta_summary_resolver,
    )
    cast(Any, runner._prompt_runner)._prompt_agent = provider.prompt_agent
    return runner


@pytest.mark.unit
def test_run_single_chapter_delegates_to_execution_coordinator(tmp_path: Path) -> None:
    """验证 Runner 的单章入口只负责委托章节执行协调器。"""

    runner = _build_runner(tmp_path)
    task = ChapterTask(index=1, title="经营表现与核心驱动", skeleton="## 经营表现与核心驱动")
    expected = ChapterResult(index=1, title=task.title, status="passed", content="## 正文", audit_passed=True)
    runner._company_facets = CompanyFacetProfile(primary_facets=["硬件/消费电子"])
    runner._company_facet_catalog = {"business_model_candidates": ["硬件/消费电子"]}
    captured: dict[str, object] = {}

    def _fake_run_single_chapter(**kwargs: object) -> ChapterResult:
        captured.update(kwargs)
        return expected

    runner._chapter_execution_coordinator.run_single_chapter = _fake_run_single_chapter  # type: ignore[method-assign]

    result = runner._run_single_chapter(
        task=task,
        company_name="Apple",
        prompt_name="write_chapter",
        prompt_inputs={"chapter": task.title},
    )

    assert result is expected
    assert captured["task"] is task
    assert captured["company_name"] == "Apple"
    assert captured["prompt_name"] == "write_chapter"
    assert captured["company_facets"] is runner._company_facets
    assert captured["company_facet_catalog"] is runner._company_facet_catalog


@pytest.mark.unit
def test_chapter_audit_coordinator_audit_and_confirm_chapter_receives_expected_inputs(tmp_path: Path) -> None:
    """验证章节审计协调器入口接收的参数保持完整。"""

    runner = _build_runner(tmp_path)
    suspected = AuditDecision(passed=False, category=AuditCategory.EVIDENCE_INSUFFICIENT)
    final = AuditDecision(passed=True, category=AuditCategory.OK)
    confirmation = EvidenceConfirmationResult(entries=[])
    captured: dict[str, object] = {}

    def _fake_audit_and_confirm_chapter(**kwargs: object) -> tuple[AuditDecision, AuditDecision, EvidenceConfirmationResult | None]:
        captured.update(kwargs)
        return suspected, final, confirmation

    runner._chapter_audit_coordinator.audit_and_confirm_chapter = _fake_audit_and_confirm_chapter  # type: ignore[method-assign]

    result = runner._chapter_audit_coordinator.audit_and_confirm_chapter(
        chapter_markdown="## 正文",
        company_name="Apple",
        skeleton="## 骨架",
        chapter_contract={"must_answer": "核心问题"},
        item_rules=[],
        phase="initial",
        chapter_title="经营表现与核心驱动",
        repair_contract={"retry_scope": "targeted_patch"},
    )

    assert result == (suspected, final, confirmation)
    assert captured["company_name"] == "Apple"
    assert captured["phase"] == "initial"
    assert captured["chapter_title"] == "经营表现与核心驱动"


@pytest.mark.unit
def test_ensure_company_facets_reuses_manifest_value_by_default(tmp_path: Path) -> None:
    """验证 manifest 已有公司级 facet 时，默认直接复用，不重跑推理。"""

    runner = _build_runner(tmp_path)
    manifest = RunManifest(
        version="write_manifest_v1",
        signature="sig",
        config=_build_test_write_config(tmp_path),
        company_facets=None,
    )
    existing = {
        "primary_facets": ["半导体设备/制造"],
        "cross_cutting_facets": ["出口限制敏感"],
        "confidence_notes": "已有归因",
    }
    manifest.company_facets = CompanyFacetProfile.from_dict(existing)
    called = {"count": 0}

    def _unexpected_infer(*, company_name: str) -> object:
        called["count"] += 1
        raise AssertionError("已有 facet 时不应重跑推理")

    runner._infer_company_facets = _unexpected_infer  # type: ignore[method-assign]

    result = runner._ensure_company_facets(manifest=manifest, company_name="Apple")

    assert result == manifest.company_facets
    assert called["count"] == 0


@pytest.mark.unit
def test_ensure_company_facets_auto_infers_when_manifest_missing(tmp_path: Path) -> None:
    """验证 manifest 缺少公司级 facet 时会自动推理并写回 manifest。"""

    runner = _build_runner(tmp_path)
    manifest = RunManifest(
        version="write_manifest_v1",
        signature="sig",
        config=_build_test_write_config(tmp_path),
        company_facets=None,
    )
    inferred = CompanyFacetProfile(
        primary_facets=["平台互联网"],
        cross_cutting_facets=["监管敏感"],
        confidence_notes="自动推理",
    )

    runner._infer_company_facets = lambda *, company_name: inferred  # type: ignore[method-assign]
    persisted: list[RunManifest] = []
    runner._store._persist_manifest = lambda *, manifest, chapter_results: persisted.append(manifest)  # type: ignore[method-assign]

    result = runner._ensure_company_facets(manifest=manifest, company_name="Apple")

    assert result == inferred
    assert manifest.company_facets == inferred
    assert persisted and persisted[-1].company_facets == inferred


@pytest.mark.unit
def test_ensure_company_facets_logs_start_and_finish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 infer 开始与完成写回 manifest 时都会输出日志。"""

    runner = _build_runner(tmp_path, infer=True)
    manifest = RunManifest(
        version="write_manifest_v1",
        signature="sig",
        config=_build_test_write_config(tmp_path, infer=True),
        company_facets=None,
    )
    inferred = CompanyFacetProfile(
        primary_facets=["平台互联网"],
        cross_cutting_facets=["监管敏感"],
        confidence_notes="自动推理",
    )
    logs: list[str] = []

    def _capture_log(message: object, **_kwargs: object) -> None:
        """收集日志文本。

        Args:
            message: 日志消息对象。
            **_kwargs: 其余关键字参数。

        Returns:
            无。

        Raises:
            无。
        """

        logs.append(str(message))

    monkeypatch.setattr(Log, "info", _capture_log)
    runner._infer_company_facets = lambda *, company_name: inferred  # type: ignore[method-assign]
    runner._store._persist_manifest = lambda *, manifest, chapter_results: None  # type: ignore[method-assign]

    result = runner._ensure_company_facets(manifest=manifest, company_name="Apple")

    assert result == inferred
    assert any("开始公司级 Facet 归因: ticker=AAPL, company=Apple, force_refresh=True" in item for item in logs)
    assert any("完成公司级 Facet 归因并写回 manifest: ticker=AAPL" in item for item in logs)


@pytest.mark.unit
def test_infer_prompt_company_meta_excludes_company_id(tmp_path: Path) -> None:
    """验证 infer 送给 LLM 的公司基础信息不会包含 `company_id`。"""

    runner = _build_runner(
        tmp_path,
        company_meta_summary_resolver=lambda _ticker: {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "market": "US",
            "company_id": "320193",
        },
    )
    captured_prompt_inputs: dict[str, object] = {}

    def _capture_rendered_prompt(*, prompt_name: str, prompt_inputs: dict[str, object]) -> str:
        """捕获 infer prompt 输入。

        Args:
            prompt_name: prompt 名称。
            prompt_inputs: prompt 输入。

        Returns:
            固定占位 prompt。

        Raises:
            AssertionError: prompt 名称不符合预期时抛出。
        """

        assert prompt_name == "infer_company_facets"
        captured_prompt_inputs.update(prompt_inputs)
        return "infer prompt"

    runner._company_facet_catalog = {
        "business_model_candidates": ["平台互联网"],
        "constraint_candidates": ["监管敏感"],
    }
    runner._prompter._render_task_prompt = _capture_rendered_prompt  # type: ignore[method-assign]
    runner._prompt_runner.run_infer_prompt = lambda prompt_text: CompanyFacetProfile(  # type: ignore[method-assign]
        primary_facets=["平台互联网"],
        cross_cutting_facets=["监管敏感"],
        confidence_notes="ok",
    )

    runner._infer_company_facets(company_name="Apple")

    company_meta = captured_prompt_inputs["company_meta"]
    assert isinstance(company_meta, dict)
    assert company_meta == {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "market": "US",
    }
    assert "company_id" not in company_meta


@pytest.mark.unit
def test_ensure_company_facets_infer_flag_forces_refresh(tmp_path: Path) -> None:
    """验证 `infer=True` 时即便 manifest 已有 facet 也会强制重跑。"""

    runner = _build_runner(tmp_path, infer=True)
    existing = CompanyFacetProfile(
        primary_facets=["企业软件"],
        cross_cutting_facets=[],
        confidence_notes="旧结果",
    )
    manifest = RunManifest(
        version="write_manifest_v1",
        signature="sig",
        config=_build_test_write_config(tmp_path, infer=True),
        company_facets=existing,
    )
    refreshed = CompanyFacetProfile(
        primary_facets=["垂直软件/创意软件"],
        cross_cutting_facets=["高SBC"],
        confidence_notes="新结果",
    )
    called = {"count": 0}

    def _infer(*, company_name: str) -> object:
        called["count"] += 1
        return refreshed

    runner._infer_company_facets = _infer  # type: ignore[method-assign]
    runner._store._persist_manifest = lambda *, manifest, chapter_results: None  # type: ignore[method-assign]

    result = runner._ensure_company_facets(manifest=manifest, company_name="Apple")

    assert called["count"] == 1
    assert result == refreshed
    assert manifest.company_facets == refreshed


@pytest.mark.unit
def test_ensure_company_facets_infer_flag_raises_on_failure(tmp_path: Path) -> None:
    """验证显式 `infer=True` 时，facet 归因失败会直接报错。"""

    runner = _build_runner(tmp_path, infer=True)
    manifest = RunManifest(
        version="write_manifest_v1",
        signature="sig",
        config=_build_test_write_config(tmp_path, infer=True),
        company_facets=CompanyFacetProfile(
            primary_facets=["企业软件"],
            cross_cutting_facets=[],
            confidence_notes="旧结果",
        ),
    )

    def _raise_error(*, company_name: str) -> object:
        """模拟 infer scene 执行失败。

        Args:
            company_name: 公司名称。

        Returns:
            无。

        Raises:
            RuntimeError: 固定抛出失败异常。
        """

        raise RuntimeError(f"mock infer failed: {company_name}")

    runner._infer_company_facets = _raise_error  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="公司级 Facet 归因失败: mock infer failed: Apple"):
        runner._ensure_company_facets(manifest=manifest, company_name="Apple")


@pytest.mark.unit
def test_run_returns_after_infer_without_entering_write_stage(tmp_path: Path) -> None:
    """验证显式 `--infer` 时只做 facet 归因并直接退出，不进入写作阶段。"""

    runner = _build_runner(tmp_path, infer=True)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "\n".join(
            [
                "## 投资要点概览",
                "",
                "概览骨架",
                "",
                "## 公司做的是什么生意",
                "",
                "正文骨架",
                "",
                "## 来源清单",
                "",
                "来源骨架",
            ]
        ),
        encoding="utf-8",
    )
    inferred = CompanyFacetProfile(
        primary_facets=["平台互联网"],
        cross_cutting_facets=["监管敏感"],
        confidence_notes="infer-only",
    )

    runner._infer_company_facets = lambda *, company_name: inferred  # type: ignore[method-assign]

    def _unexpected_parallel_run(**_kwargs: object) -> object:
        """阻止测试误入正文写作阶段。

        Args:
            **_kwargs: 任意关键字参数。

        Returns:
            无。

        Raises:
            AssertionError: 一旦进入正文写作即失败。
        """

        raise AssertionError("infer-only 模式不应进入中间章节写作")

    def _unexpected_single_chapter(**_kwargs: object) -> object:
        """阻止测试误入单章写作阶段。

        Args:
            **_kwargs: 任意关键字参数。

        Returns:
            无。

        Raises:
            AssertionError: 一旦进入单章写作即失败。
        """

        raise AssertionError("infer-only 模式不应进入单章写作")

    runner._run_middle_tasks_in_parallel = _unexpected_parallel_run  # type: ignore[method-assign]
    runner._run_single_chapter = _unexpected_single_chapter  # type: ignore[method-assign]

    exit_code = runner.run()

    assert exit_code == 0
    manifest = _read_manifest_from_dir(Path(runner._write_config.output_dir))
    assert manifest is not None
    assert manifest.company_facets == inferred
    assert manifest.chapter_results == {}


@pytest.mark.unit
def test_resolve_company_name_uses_injected_resolver(tmp_path: Path) -> None:
    """验证写作流水线通过注入的 resolver 获取公司名称。"""

    runner = _build_runner(
        tmp_path,
        company_name_resolver=lambda ticker: f"{ticker}-Resolved",
    )

    assert runner._resolve_company_name() == "AAPL-Resolved"


@pytest.mark.unit
def test_pipeline_helpers_cover_company_meta_summary_and_audit_gate(tmp_path: Path) -> None:
    """验证 pipeline 的公司 meta 摘要兜底与 audit gate 阻塞说明。"""

    runner = _build_runner(
        tmp_path,
        company_meta_summary_resolver=lambda _ticker: {"company_name": " Apple ", "market": "1", "": "x"},
    )
    tasks = [
        ChapterTask(index=1, title="缺结果", skeleton=""),
        ChapterTask(index=2, title="写作失败", skeleton=""),
        ChapterTask(index=3, title="缺正文", skeleton=""),
        ChapterTask(index=4, title="未过审计", skeleton=""),
    ]
    chapter_results = {
        "写作失败": ChapterResult(
            index=2,
            title="写作失败",
            status="failed",
            content="",
            audit_passed=False,
            failure_reason="boom",
        ),
        "缺正文": ChapterResult(index=3, title="缺正文", status="passed", content="", audit_passed=True),
        "未过审计": ChapterResult(index=4, title="未过审计", status="passed", content="正文", audit_passed=False),
    }

    assert runner._resolve_company_meta_summary() == {
        "company_name": "Apple",
        "market": "1",
        "ticker": "AAPL",
    }
    assert runner._collect_blocking_task_titles_for_audit_gate(
        tasks=tasks,
        chapter_results=chapter_results,
    ) == [
        "缺结果 缺少章节结果",
        "写作失败 写作失败",
        "缺正文 缺少章节正文",
        "未过审计 未通过最终审计",
    ]


@pytest.mark.unit
def test_pipeline_helpers_cover_company_meta_summary_fallback_and_artifact_recovery_errors(tmp_path: Path) -> None:
    """验证公司 meta resolver 异常兜底，以及历史章节产物恢复的错误分支。"""

    runner = _build_runner(
        tmp_path,
        company_meta_summary_resolver=lambda _ticker: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    task = ChapterTask(index=2, title="公司介绍", skeleton="## 公司介绍")

    missing_result, missing_error = runner._recover_prior_chapter_result_from_artifacts(task)
    chapter_path = runner._store.chapter_file_path(task.index, task.title)
    chapter_path.parent.mkdir(parents=True, exist_ok=True)
    chapter_path.write_text("正文", encoding="utf-8")
    no_audit_result, no_audit_error = runner._recover_prior_chapter_result_from_artifacts(task)
    audit_path = runner._store.chapter_phase_artifact_path(
        index=task.index,
        title=task.title,
        artifact_name="audit",
        extension="json",
    )
    audit_path.write_text("{bad json", encoding="utf-8")
    invalid_audit_result, invalid_audit_error = runner._recover_prior_chapter_result_from_artifacts(task)

    assert runner._resolve_company_meta_summary() == {"ticker": "AAPL"}
    assert missing_result is None
    assert missing_error == "公司介绍 缺少章节文件"
    assert no_audit_result is None
    assert no_audit_error == "公司介绍 缺少最终 audit 产物"
    assert invalid_audit_result is None
    assert invalid_audit_error.startswith("公司介绍 audit 产物不可读")


@pytest.mark.unit
def test_run_write_pipeline_wrapper_and_print_write_report_cover_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 run_write_pipeline 入口封装与 print_write_report 的成功/失败分支。"""

    captured: dict[str, object] = {}
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    class _FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def run(self) -> int:
            return 7

    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.WritePipelineRunner", _FakeRunner)

    exit_code = run_write_pipeline(
        workspace=_build_test_workspace_resources(tmp_path),
        resolved_options=_build_test_resolved_options(tmp_path / "trace"),
        write_config=_build_test_write_config(tmp_path),
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(_FakeAgentProvider()),
        host_executor=_build_test_host_executor(),
        host_session_id="host-session",
    )
    missing_report_code = print_write_report(output_dir)

    manifest = RunManifest(
        version="write_manifest_v1",
        signature="sig",
        config=_build_test_write_config(tmp_path),
        chapter_results={
            "第一章": ChapterResult(index=1, title="第一章", status="passed", content="正文", audit_passed=True),
            "第二章": ChapterResult(
                index=2,
                title="第二章",
                status="failed",
                content="",
                audit_passed=False,
                failure_reason="x" * 140,
            ),
        },
    )
    _write_manifest(output_dir / "manifest.json", manifest)
    failed_report_code = print_write_report(output_dir)
    manifest.chapter_results = {
        "第一章": ChapterResult(index=1, title="第一章", status="passed", content="正文", audit_passed=True)
    }
    _write_manifest(output_dir / "manifest.json", manifest)
    passed_report_code = print_write_report(output_dir)

    captured_output = capsys.readouterr().out
    assert exit_code == 7
    assert captured["host_session_id"] == "host-session"
    assert missing_report_code == 2
    assert failed_report_code == 4
    assert passed_report_code == 0
    assert "manifest.json 不存在" in captured_output
    assert "失败章节列表" in captured_output
    assert "所有章节均写作成功" in captured_output


@pytest.mark.unit
def test_strip_evidence_section_removes_section() -> None:
    """验证 _strip_evidence_section 正确剥除"证据与出处"小节。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = (
        "## 投资要点概览\n\n"
        "### 结论要点\n\n- 核心业务\n\n"
        "### 买方跟踪清单\n\n- 跟踪变量\n\n"
        "### 详细情况\n\n- 详情\n\n"
        "### 证据与出处\n\n- 来源1\n- 来源2"
    )
    result = _strip_evidence_section(content)

    assert "### 证据与出处" not in result
    assert "来源1" not in result
    assert "### 结论要点" in result
    assert "### 买方跟踪清单" in result


@pytest.mark.unit
def test_strip_evidence_section_no_section_unchanged() -> None:
    """验证内容中无"证据与出处"时原样返回。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = "## 投资要点概览\n\n### 结论要点\n\n- 内容"
    assert _strip_evidence_section(content) == content


@pytest.mark.unit
def test_run_allows_template_without_source_chapter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证模板缺少来源清单时，流水线会跳过来源清单生成。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\n"
        "x\n"
        "## 第一章\n"
        "x\n"
        "## 是否值得继续深研与待验证问题\n"
        "x\n",
        encoding="utf-8",
    )
    sources_path = Path(runner._write_config.output_dir) / "sources_dedup.json"
    sources_path.parent.mkdir(parents=True, exist_ok=True)
    sources_path.write_text("[]", encoding="utf-8")

    calls: list[str] = []

    def _fake_run_single_chapter(*, task, **_kwargs):
        calls.append(task.title)
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content=f"## {task.title}\n\n### 结论要点\n\n- 内容\n\n### 证据与出处\n\n- 来源",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda _result: None)
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr(runner._store, "_persist_sources_json", lambda _entries: (_ for _ in ()).throw(AssertionError("不应生成来源清单 JSON")))
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.warn", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 0
    assert calls == ["第一章", "是否值得继续深研与待验证问题", "投资要点概览"]
    report_path = Path(runner._write_config.output_dir) / "AAPL_qual_report.md"
    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "## 来源清单" not in report_text
    assert not sources_path.exists()


@pytest.mark.unit
def test_build_middle_tasks_skips_overview_decision_and_source() -> None:
    """验证常规中间章节任务会跳过首章、决策章与来源章。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    layout = parse_template_layout(
        """
## 投资要点概览
x
## 公司介绍
x
## 是否值得继续深研与待验证问题
x
## 来源清单
x
""".strip()
    )

    tasks = _build_middle_tasks(layout)

    assert len(tasks) == 1
    assert tasks[0].title == "公司介绍"


@pytest.mark.unit
def test_build_overview_dependency_tasks_keeps_decision_and_skips_source() -> None:
    """验证概览回填依赖会保留决策章，但跳过概览章与来源章。"""

    layout = parse_template_layout(
        """
## 投资要点概览
x
## 公司介绍
x
## 是否值得继续深研与待验证问题
x
## 来源清单
x
""".strip()
    )

    tasks = _build_overview_dependency_tasks(layout)

    assert [task.title for task in tasks] == ["公司介绍", "是否值得继续深研与待验证问题"]


@pytest.mark.unit
def test_build_prior_decision_tasks_uses_preceding_business_chapters_only() -> None:
    """验证第10章前置章节列表只取其前面的业务章节，不含 overview。"""

    layout = parse_template_layout(
        """
## 投资要点概览
x
## 第一章
x
## 第二章
x
## 是否值得继续深研与待验证问题
x
## 来源清单
x
""".strip()
    )

    tasks = _build_prior_decision_tasks(layout)

    assert [task.title for task in tasks] == ["第一章", "第二章"]


@pytest.mark.unit
def test_build_chapter_prompt_inputs_only_passes_item_rules_when_enabled(tmp_path: Path) -> None:
    """验证 ITEM_RULE 仅进入 write/regenerate 链路，不进入首章回填链路。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner = _build_runner(tmp_path)
    task = ChapterTask(
        index=2,
        title="公司介绍",
        skeleton="## 公司介绍",
        item_rules=[
            ItemRule(
                mode="conditional",
                target_heading="业务结构、收入形成与主次关系",
                item="更细业务拆分",
                when="仅在证据充分时写",
            )
        ],
    )

    write_inputs = runner._prompter._build_chapter_prompt_inputs(task=task, company_name="Apple")
    overview_inputs = runner._prompter._build_chapter_prompt_inputs(
        task=task,
        company_name="Apple",
        include_item_rules=False,
    )

    assert write_inputs["item_rules"][0]["item"] == "更细业务拆分"
    assert "item_rules" not in overview_inputs


@pytest.mark.unit
def test_prompt_builder_facet_catalog_does_not_alias_external_mutation(tmp_path: Path) -> None:
    """验证 PromptBuilder 会复制 facet 状态，避免外部继续修改内部快照。"""

    runner = _build_runner(tmp_path)
    primary_facets = ["平台互联网"]
    runner._prompter.set_company_facets(CompanyFacetProfile(primary_facets=primary_facets))
    external_catalog = {"business_model_candidates": [], "constraint_candidates": []}
    runner._prompter.set_company_facet_catalog(external_catalog)
    primary_facets.append("硬件/消费电子")
    external_catalog["business_model_candidates"].append("平台互联网")
    company_facets, facet_catalog = runner._prompter._get_company_facet_state_snapshot()

    assert company_facets is not None
    assert company_facets.primary_facets == ["平台互联网"]
    assert facet_catalog == {"business_model_candidates": [], "constraint_candidates": []}


@pytest.mark.unit
def test_build_research_decision_input_uses_prior_chapter_summaries_and_audit_status(tmp_path: Path) -> None:
    """验证第10章输入只使用前文章节结构化摘要，并附带审计状态摘要。"""

    runner = _build_runner(tmp_path)
    layout = parse_template_layout(
        """
## 投资要点概览
x
## 第一章
x
## 第二章
x
## 是否值得继续深研与待验证问题
x
## 来源清单
x
""".strip()
    )
    first_result = ChapterResult(
        index=2,
        title="第一章",
        status="passed",
        content="## 第一章\n\n### 结论要点\n\n- 结论A\n\n### 证据与出处\n\n- 来源A",
        audit_passed=True,
    )
    second_result = ChapterResult(
        index=3,
        title="第二章",
        status="failed",
        content="## 第二章\n\n### 结论要点\n\n- 结论B\n\n### 证据与出处\n\n- 来源B",
        audit_passed=False,
    )
    runner._store._chapter_file_path(2, "第一章").write_text(first_result.content, encoding="utf-8")
    runner._store._chapter_file_path(3, "第二章").write_text(second_result.content, encoding="utf-8")
    runner._store._chapter_phase_artifact_path(
        index=2, title="第一章", artifact_name="audit", extension="json"
    ).write_text(json.dumps({"pass": True, "class": "ok", "violations": []}, ensure_ascii=False), encoding="utf-8")
    runner._store._chapter_phase_artifact_path(
        index=3, title="第二章", artifact_name="audit", extension="json"
    ).write_text(
        json.dumps(
            {
                "pass": False,
                "class": "content_violation",
                "violations": [
                    {
                        "rule": "E1",
                        "severity": "high",
                        "excerpt": "缺关键依据",
                        "reason": "关键断言暂无稳定证据",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    decision_input = runner._prompter._build_research_decision_input(
        layout=layout,
        chapter_results={"第一章": first_result, "第二章": second_result},
    )

    assert "## 投资要点概览" not in decision_input
    assert "## 第一章" in decision_input
    assert "## 第二章" in decision_input
    assert "### 结论要点" in decision_input
    assert "### 证据与出处" in decision_input
    assert "### 审计状态摘要" in decision_input
    assert "- 最终审计：通过" in decision_input
    assert "- 最终审计：未通过" in decision_input
    assert "### 未解决的高优先级问题" in decision_input
    assert "[E1] 缺关键依据：关键断言暂无稳定证据" in decision_input


@pytest.mark.unit
def test_build_overview_input_uses_prior_chapter_takeaways_and_includes_decision(tmp_path: Path) -> None:
    """验证第0章输入包含第10章结论要点，但不再喂各章证据区。"""

    runner = _build_runner(tmp_path)
    layout = parse_template_layout(
        "## 投资要点概览\nx\n## 第一章\nx\n## 是否值得继续深研与待验证问题\nx\n## 来源清单\nx\n"
    )
    chapter_results = {
        "是否值得继续深研与待验证问题": ChapterResult(
            index=10,
            title="是否值得继续深研与待验证问题",
            status="passed",
            content="## 是否值得继续深研与待验证问题\n\n### 结论要点\n\n- 动作B\n\n### 证据与出处\n\n- 来源B",
            audit_passed=True,
        ),
        "第一章": ChapterResult(
            index=2,
            title="第一章",
            status="passed",
            content="## 第一章\n\n### 结论要点\n\n- 结论A\n\n### 证据与出处\n\n- 来源A",
            audit_passed=True,
        ),
        "投资要点概览": ChapterResult(
            index=0,
            title="投资要点概览",
            status="passed",
            content="## 投资要点概览\n\n- 已有概览",
            audit_passed=True,
        ),
    }

    overview_input = runner._prompter._build_overview_input(layout=layout, chapter_results=chapter_results)

    assert "## 第一章" in overview_input
    assert "## 是否值得继续深研与待验证问题" in overview_input
    assert overview_input.index("## 第一章") < overview_input.index("## 是否值得继续深研与待验证问题")
    assert "### 结论要点" in overview_input
    assert "- 结论A" in overview_input
    assert "- 动作B" in overview_input
    assert "来源A" not in overview_input
    assert "来源B" not in overview_input
    assert "## 投资要点概览" not in overview_input


@pytest.mark.unit
def test_collect_all_evidence_items_respects_layout_order() -> None:
    """验证来源清单证据聚合按模板顺序消费章节结果。"""

    chapter_results = {
        "第二章": ChapterResult(
            index=3,
            title="第二章",
            status="passed",
            content="## 第二章",
            evidence_items=["来源B"],
            audit_passed=True,
        ),
        "第一章": ChapterResult(
            index=2,
            title="第一章",
            status="passed",
            content="## 第一章",
            evidence_items=["来源A"],
            audit_passed=True,
        ),
    }

    assert _collect_all_evidence_items(
        chapter_results,
        ordered_titles=["第一章", "第二章"],
    ) == ["来源A", "来源B"]


@pytest.mark.unit
def test_fill_overview_routes_to_dedicated_overview_agent(tmp_path: Path) -> None:
    """验证第0章不再复用 write scene，而是走独立 overview scene。"""

    runner = _build_runner(tmp_path)
    calls: list[str] = []

    def _fake_run_overview(prompt_text: str) -> str:
        del prompt_text
        calls.append("overview")
        return "## 投资要点概览\n\n- 概览"

    def _fake_run_write(prompt_text: str) -> str:
        del prompt_text
        calls.append("write")
        return "## 错误"

    runner._prompt_runner.run_overview_prompt = _fake_run_overview  # type: ignore[method-assign]
    runner._prompt_runner.run_write_prompt = _fake_run_write  # type: ignore[method-assign]

    content = runner._prompt_runner.run_initial_chapter_prompt(
        prompt_name="fill_overview",
        prompt_text="prompt",
    )

    assert content == "## 投资要点概览\n\n- 概览"
    assert calls == ["overview"]


@pytest.mark.unit
def test_fill_overview_skips_audit_confirm_and_repair(tmp_path: Path) -> None:
    """验证第0章只写封面页正文，不进入 audit/confirm/repair 链路。"""

    runner = _build_runner(tmp_path)
    task = ChapterTask(
        index=1,
        title="投资要点概览",
        skeleton="- 概览",
    )

    runner._prompter._render_task_prompt = lambda **_kwargs: "prompt"  # type: ignore[method-assign]
    runner._prompt_runner.run_initial_chapter_prompt = lambda **_kwargs: "## 投资要点概览\n\n- 当前研究动作：继续研究"  # type: ignore[method-assign]

    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("第0章不应进入审计链路")

    runner._chapter_audit_coordinator.evaluate_current_chapter_phase = _unexpected_call  # type: ignore[method-assign]

    result = runner._run_single_chapter(
        task=task,
        company_name="Apple",
        prompt_name="fill_overview",
        prompt_inputs={"chapter": "投资要点概览"},
    )

    assert result.status == "passed"
    assert result.audit_passed is True
    assert result.retry_count == 0
    assert result.process_state["audit_skipped"] is True
    assert result.process_state["confirm_skipped"] is True
    assert result.process_state["repair_skipped"] is True
    assert result.process_state["final_stage"] == "overview_written"


@pytest.mark.unit
def test_run_overview_agent_prompt_raw_retries_and_raises_runtime_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证第0章概览 Agent 的错误处理与其它写作 Agent 保持一致。"""

    runner = _build_runner(tmp_path)
    monkeypatch.setattr(runner._preparer, "_create_overview_agent", lambda: object())
    monkeypatch.setattr(
        runner._prompt_runner,
        "run_prepared_scene_prompt",
        lambda **_kwargs: AppResult(content="", errors=["boom"], warnings=[], degraded=False),
    )
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="第0章概览 Agent 执行失败"):
        runner._prompt_runner._run_overview_agent_prompt_raw("prompt")


@pytest.mark.unit
def test_parse_audit_decision_handles_invalid_json() -> None:
    """验证审计 JSON 解析失败时会返回失败结果。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    decision = _parse_audit_decision("not json")

    assert decision.passed is False
    assert decision.category == "style_violation"
    assert decision.violations
    assert decision.repair_contract.contract_version == "repair_contract_v1"
    assert decision.repair_contract.retry_scope == "targeted_style_patch"


@pytest.mark.unit
def test_run_single_chapter_retries_and_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证审计失败后进入修复循环，重审通过即返回 passed。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner = _build_runner(tmp_path)
    info_logs: list[str] = []
    warn_logs: list[str] = []
    write_sequence = _SequenceCaller(
        [
            "初稿内容足够长度用于测试场景\n\n### 证据与出处\n- SEC EDGAR | Form 10-K | Filed 2025-01-01 | Accession 0000000001-25-000001",
        ]
    )
    repair_sequence = _SequenceCaller(
        [
            (
                {
                    "patches": [
                        {
                            "target_excerpt": "初稿内容足够长度用于测试场景",
                            "replacement": "修复后内容足够长度用于测试场景",
                            "reason": "测试 patch 应用",
                        }
                    ],
                    "notes": [],
                },
                json.dumps(
                    {
                        "patches": [
                            {
                                "target_excerpt": "初稿内容足够长度用于测试场景",
                                "replacement": "修复后内容足够长度用于测试场景",
                                "reason": "测试 patch 应用",
                            }
                        ],
                        "notes": [],
                    },
                    ensure_ascii=False,
                ),
            )
        ]
    )
    audit_sequence = _SequenceCaller(
        [
            AuditDecision(
                passed=False,
                category=AuditCategory.EVIDENCE_INSUFFICIENT,
                repair_contract=RepairContract(
                    contract_version="repair_contract_v1",
                    preferred_tool_action="repair_chapter",
                    repair_strategy="patch",
                    retry_scope="targeted_evidence_patch",
                    missing_evidence_slots=[MissingEvidenceSlot(slot_id="slot_1_0_1")],
                ),
                raw='{"pass": false}',
            ),
            AuditDecision(
                passed=True,
                category=AuditCategory.OK,
                repair_contract=RepairContract(
                    contract_version="repair_contract_v1",
                    preferred_tool_action="none",
                    repair_strategy="none",
                    retry_scope="none",
                ),
                raw='{"pass": true}',
            ),
        ]
    )

    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda message, **_kwargs: info_logs.append(str(message)))
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.warn", lambda message, **_kwargs: warn_logs.append(str(message)))
    monkeypatch.setattr(runner._prompt_runner, "run_write_prompt", write_sequence)
    monkeypatch.setattr(runner._prompt_runner, "run_repair_prompt", repair_sequence)
    monkeypatch.setattr(runner._chapter_audit_coordinator, "_audit_chapter", audit_sequence)

    task = ChapterTask(index=2, title="公司介绍", skeleton="## 公司介绍")
    result = runner._run_single_chapter(
        task=task,
        company_name="Apple",
        prompt_name="write_chapter",
        prompt_inputs=runner._prompter._build_chapter_prompt_inputs(task=task, company_name="Apple"),
    )

    assert result.status == "passed"
    assert result.retry_count == 1
    assert result.content.startswith("修复后内容足够长度用于测试场景")
    assert result.process_state["fix_applied"] is False
    assert result.process_state["final_stage"] == "audit_passed"
    assert len(result.process_state["audit_history"]) == 2
    assert len(result.process_state["rewrite_history"]) == 1
    assert len(result.process_state["repair_contract_history"]) == 2
    assert result.process_state["latest_repair_contract"]["retry_scope"] in {
        "targeted_evidence_patch",
        "none",
    }
    chapters_dir = Path(runner._write_config.output_dir) / "chapters"
    assert (chapters_dir / "01_公司介绍.initial_write.md").read_text(encoding="utf-8").startswith(
        "初稿内容足够长度用于测试场景"
    )
    assert (chapters_dir / "01_公司介绍.repair_1_write.md").read_text(encoding="utf-8").startswith(
        "修复后内容足够长度用于测试场景"
    )
    assert (chapters_dir / "01_公司介绍.repair_1_input_write.md").read_text(encoding="utf-8").startswith(
        "初稿内容足够长度用于测试场景"
    )
    repair_plan = json.loads((chapters_dir / "01_公司介绍.repair_1_repair_plan.json").read_text(encoding="utf-8"))
    assert repair_plan["patches"][0]["target_excerpt"] == "初稿内容足够长度用于测试场景"
    repair_context = json.loads((chapters_dir / "01_公司介绍.repair_1_context.json").read_text(encoding="utf-8"))
    assert repair_context["retry_scope"] in {"targeted_evidence_patch", "none"}
    assert repair_context["current_visible_headings"].startswith("- ")
    assert repair_context["input_write_artifact"] == "repair_1_input_write.md"
    repair_apply_result = json.loads((chapters_dir / "01_公司介绍.repair_1_apply_result.json").read_text(encoding="utf-8"))
    assert repair_apply_result["all_failed"] is False
    assert repair_apply_result["applied_count"] == 1
    assert repair_apply_result["patched_markdown"].startswith("修复后内容足够长度用于测试场景")
    assert repair_apply_result["patch_results"][0]["status"] == "applied"
    initial_audit_suspect = json.loads(
        (chapters_dir / "01_公司介绍.initial_audit_suspect.json").read_text(encoding="utf-8")
    )
    initial_audit = json.loads((chapters_dir / "01_公司介绍.initial_audit.json").read_text(encoding="utf-8"))
    rewrite_audit_suspect = json.loads(
        (chapters_dir / "01_公司介绍.repair_1_audit_suspect.json").read_text(encoding="utf-8")
    )
    rewrite_audit = json.loads((chapters_dir / "01_公司介绍.repair_1_audit.json").read_text(encoding="utf-8"))
    final_audit = json.loads((chapters_dir / "01_公司介绍.audit.json").read_text(encoding="utf-8"))
    assert initial_audit_suspect["pass"] is False
    assert initial_audit["pass"] is False
    assert initial_audit["class"] == "evidence_insufficient"
    assert rewrite_audit_suspect["pass"] is True
    assert rewrite_audit["pass"] is True
    assert rewrite_audit["class"] == "ok"
    assert final_audit == rewrite_audit
    assert any("开始写章节: 公司介绍" in item for item in info_logs)
    assert any("写完章节: 公司介绍, phase=initial" in item for item in info_logs)
    assert any("开始审计章节: 公司介绍, phase=initial" in item for item in info_logs)
    assert any("审计失败: 公司介绍, phase=initial" in item for item in warn_logs)
    assert any("开始局部修复: 公司介绍, retry=1" in item for item in info_logs)
    assert any("审计成功: 公司介绍, phase=repair_1" in item for item in info_logs)
    assert any("章节最终完成: 公司介绍, status=passed, retry_count=1" in item for item in info_logs)


@pytest.mark.unit
def test_run_single_chapter_applies_fix_when_placeholder_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证命中占位符特征时会先执行 fix，再进入审计流程。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner = _build_runner(tmp_path)
    write_sequence = _SequenceCaller([
        "初稿含【占位符】并且内容足够长\n\n### 证据与出处\n- SEC EDGAR | Form 10-K | Filed 2025-01-01 | Accession 0000000001-25-000001",
    ])
    fix_sequence = _SequenceCaller([
        "补强后正文内容足够长度用于测试\n\n### 证据与出处\n- SEC EDGAR | Form 10-K | Filed 2025-01-01 | Accession 0000000001-25-000001",
    ])
    audit_sequence = _SequenceCaller([AuditDecision(passed=True, category=AuditCategory.OK, raw='{"pass": true}')])

    monkeypatch.setattr(runner._prompt_runner, "run_write_prompt", write_sequence)
    monkeypatch.setattr(runner._prompt_runner, "run_fix_prompt", fix_sequence)
    monkeypatch.setattr(runner._chapter_audit_coordinator, "_audit_chapter", audit_sequence)

    task = ChapterTask(index=2, title="公司介绍", skeleton="## 公司介绍")
    result = runner._run_single_chapter(
        task=task,
        company_name="Apple",
        prompt_name="write_chapter",
        prompt_inputs=runner._prompter._build_chapter_prompt_inputs(task=task, company_name="Apple"),
    )

    assert result.status == "passed"
    assert result.retry_count == 0
    assert result.content.startswith("补强后正文内容足够长度用于测试")
    assert result.process_state["fix_applied"] is True
    assert result.process_state["fix_reason"] == "placeholder_detected"
    assert result.process_state["final_stage"] == "audit_passed"
    chapters_dir = Path(runner._write_config.output_dir) / "chapters"
    assert (chapters_dir / "01_公司介绍.initial_write.md").read_text(encoding="utf-8").startswith(
        "初稿含【占位符】并且内容足够长"
    )
    assert (chapters_dir / "01_公司介绍.initial_fix_placeholders.md").read_text(encoding="utf-8").startswith(
        "补强后正文内容足够长度用于测试"
    )
    initial_audit_suspect = json.loads(
        (chapters_dir / "01_公司介绍.initial_audit_suspect.json").read_text(encoding="utf-8")
    )
    initial_audit = json.loads((chapters_dir / "01_公司介绍.initial_audit.json").read_text(encoding="utf-8"))
    final_audit = json.loads((chapters_dir / "01_公司介绍.audit.json").read_text(encoding="utf-8"))
    assert initial_audit_suspect["pass"] is True
    assert initial_audit["pass"] is True
    assert final_audit == initial_audit


@pytest.mark.unit
def test_has_evidence_section_detects_heading() -> None:
    """验证 _has_evidence_section 能识别"### 证据与出处"标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    with_heading = "## 公司介绍\n\n正文\n\n### 证据与出处\n- 来源"
    without_heading = "## 公司介绍\n\n正文只有内容没有证据小节"

    assert _has_evidence_section(with_heading) is True
    assert _has_evidence_section(without_heading) is False


@pytest.mark.unit
def test_run_single_chapter_fast_mode_skips_fix_and_audit_chain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 fast 模式只写正文，不进入 fix/audit/confirm/repair。"""

    runner = _build_runner(tmp_path)
    runner._write_config.fast = True

    monkeypatch.setattr(runner._prompt_runner, "run_write_prompt", lambda _prompt: "## 公司介绍\n\n正文\n\n### 证据与出处\n\n- 来源")
    monkeypatch.setattr(
        runner._prompt_runner,
        "run_fix_prompt",
        lambda *_args, **_kwargs: pytest.fail("fast 模式不应执行 fix_prompt"),
    )
    monkeypatch.setattr(
        runner._chapter_audit_coordinator,
        "evaluate_current_chapter_phase",
        lambda *_args, **_kwargs: pytest.fail("fast 模式不应执行 audit_and_confirm"),
    )
    monkeypatch.setattr(
        runner._prompt_runner,
        "run_repair_prompt",
        lambda *_args, **_kwargs: pytest.fail("fast 模式不应执行 repair_prompt"),
    )
    monkeypatch.setattr(
        runner._prompt_runner,
        "run_regenerate_prompt",
        lambda *_args, **_kwargs: pytest.fail("fast 模式不应执行 regenerate_prompt"),
    )

    task = ChapterTask(index=2, title="公司介绍", skeleton="## 公司介绍")
    result = runner._run_single_chapter(
        task=task,
        company_name="Apple",
        prompt_name="write_chapter",
        prompt_inputs=runner._prompter._build_chapter_prompt_inputs(task=task, company_name="Apple"),
    )

    assert result.status == "passed"
    assert result.audit_passed is False
    assert result.process_state["final_stage"] == "fast_written"
    assert result.process_state["audit_skipped"] is True
    assert result.process_state["confirm_skipped"] is True
    assert result.process_state["repair_skipped"] is True


@pytest.mark.unit
def test_run_programmatic_audits_too_short_returns_failed_decision() -> None:
    """验证内容过短时 _run_programmatic_audits 返回失败决策（rule=P2）。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    decision = _run_programmatic_audits("短")

    assert decision is not None
    assert decision.passed is False
    assert decision.violations[0].rule == AuditRuleCode.P2
    assert decision.category == AuditCategory.CONTENT_VIOLATION
    assert decision.repair_contract.preferred_tool_action == "regenerate_chapter"
    assert decision.repair_contract.repair_strategy == "regenerate"
    assert decision.repair_contract.retry_scope == "chapter_regenerate"
    assert decision.repair_contract.missing_evidence_slots == []


@pytest.mark.unit
def test_run_programmatic_audits_missing_evidence_section_returns_failed_decision() -> None:
    """验证缺少证据小节时 _run_programmatic_audits 返回失败决策（rule=P3）。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    long_content = "正文内容足够长但没有证据小节，这是一段合理长度的写作内容供测试使用"
    decision = _run_programmatic_audits(long_content)

    assert decision is not None
    assert decision.passed is False
    assert decision.violations[0].rule == AuditRuleCode.P3
    assert decision.repair_contract.preferred_tool_action == "regenerate_chapter"
    assert decision.repair_contract.repair_strategy == "regenerate"
    assert decision.repair_contract.retry_scope == "chapter_regenerate"


@pytest.mark.unit
def test_run_programmatic_audits_skeleton_mismatch_returns_failed_decision() -> None:
    """验证骨架标题顺序不匹配时程序审计返回 P1。"""

    content = (
        "## 公司做的是什么生意\n\n"
        "### 详细情况\n\n"
        "- test\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2026-01-01 | Accession 0000000000-26-000001 | Part II - Item 8\n"
    )
    skeleton = (
        "## 公司做的是什么生意\n\n"
        "### 结论要点\n\n"
        "### 详细情况\n\n"
        "### 证据与出处\n"
    )

    decision = _run_programmatic_audits(content, skeleton=skeleton)

    assert decision is not None
    assert decision.passed is False
    assert decision.category == AuditCategory.CONTENT_VIOLATION
    assert decision.violations[0].rule == AuditRuleCode.P1


@pytest.mark.unit
def test_run_programmatic_audits_allows_conditional_heading_with_spacing_difference() -> None:
    """验证满足 ITEM_RULE 的条件小节不会被程序审计误判为 P1。"""

    content = (
        "## 最近一年关键变化与当前阶段\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 详细情况\n\n"
        "#### 软件/数据公司的seat、用量、AI商业化或销售效率变化\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2026-01-01 | Accession 0000000000-26-000001 | Part II - Item 7\n"
    )
    skeleton = (
        "## 最近一年关键变化与当前阶段\n\n"
        "### 结论要点\n\n"
        "### 详细情况\n\n"
        "### 证据与出处\n"
    )

    decision = _run_programmatic_audits(
        content,
        skeleton=skeleton,
        allowed_conditional_headings={"软件/数据公司的 seat、用量、AI 商业化或销售效率变化"},
    )

    assert decision is None


@pytest.mark.unit
def test_run_programmatic_audits_valid_content_returns_none() -> None:
    """验证合规内容时 _run_programmatic_audits 返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    valid_content = (
        "## 公司介绍\n\n正文内容足够长，这是一段合理长度的写作内容供测试使用。\n\n"
        "### 证据与出处\n- SEC EDGAR | Form 10-K | Filed 2025-01-01 | Accession 0000000001-25-000001"
    )
    assert _run_programmatic_audits(valid_content) is None


@pytest.mark.unit
def test_run_single_chapter_evidence_section_missing_triggers_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证初稿缺少"证据与出处"时跳过 LLM 初始审计、直接进入重试循环。

    初稿无"### 证据与出处" → 跳过占位符修复与 LLM 审计 → 进入重写 →
    重写稿补上小节 → 审计通过 → 最终 passed。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner = _build_runner(tmp_path)
    content_no_evidence = "正文内容足够长但没有证据小节，这是一段合理长度的写作内容供测试使用"
    content_with_evidence = (
        "重写后正文补充了证据小节\n\n"
        "### 证据与出处\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-01-01 | Accession 0000000001-25-000001"
    )
    regenerate_sequence = _SequenceCaller([content_with_evidence])
    audit_calls: list[object] = []

    def mock_audit(*_args: object, **_kwargs: object) -> AuditDecision:
        audit_calls.append(1)
        return AuditDecision(passed=True, category=AuditCategory.OK, raw='{"pass": true}')

    monkeypatch.setattr(runner._prompt_runner, "run_write_prompt", _SequenceCaller([content_no_evidence]))
    monkeypatch.setattr(runner._prompt_runner, "run_regenerate_prompt", regenerate_sequence)
    monkeypatch.setattr(runner._chapter_audit_coordinator, "_audit_chapter", mock_audit)

    task = ChapterTask(index=2, title="公司介绍", skeleton="## 公司介绍")
    result = runner._run_single_chapter(
        task=task,
        company_name="Apple",
        prompt_name="write_chapter",
        prompt_inputs=runner._prompter._build_chapter_prompt_inputs(task=task, company_name="Apple"),
    )

    # 重写后补上证据小节，审计通过，最终 passed
    assert result.status == "passed"
    assert result.retry_count == 1
    assert result.process_state["fix_reason"] == "programmatic_audit_failed"
    chapters_dir = Path(runner._write_config.output_dir) / "chapters"
    assert (chapters_dir / "01_公司介绍.regenerate_1_write.md").read_text(encoding="utf-8").startswith(
        "重写后正文补充了证据小节"
    )
    # 初始阶段不进入 LLM 审计；重写后进入审计并通过，共调用 1 次
    assert len(audit_calls) == 1


@pytest.mark.unit
def test_should_run_fix_placeholders_detects_common_patterns() -> None:
    """验证占位符检测规则可识别常见占位符文本。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _should_run_fix_placeholders("这里有【占位符：缺口】")
    assert _should_run_fix_placeholders("{{UNFILLED_FIELD}}")
    assert _should_run_fix_placeholders("TODO: 补充事实")
    assert _should_run_fix_placeholders("正文内容完整，无占位") is False


@pytest.mark.unit
def test_create_write_agent_leaves_max_iterations_to_runtime_scene_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证写作 scene 不再由 pipeline 直接注入 max_iterations，而交给 Runtime 按 scene manifest 解析。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    write_config = _build_test_write_config(tmp_path)
    workspace_config = _build_test_workspace_resources(tmp_path)
    running_config = _build_test_resolved_options(tmp_path / "tool_trace", trace_enabled=True)
    del monkeypatch
    provider = _FakeAgentProvider()

    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=write_config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
    )
    runner._preparer._create_write_agent()

    assert provider.prepare_calls
    execution_options = provider.prepare_calls[0]["execution_options"]
    assert isinstance(execution_options, ExecutionOptions)
    assert execution_options.model_name is None
    assert execution_options.max_iterations is None
    assert execution_options.web_provider == "auto"


@pytest.mark.unit
def test_create_write_agent_does_not_inject_global_max_iterations_into_execution_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证写作模式不再由 pipeline 直接把全局 max_iterations 注入 scene 执行选项。"""

    write_config = _build_test_write_config(tmp_path)
    workspace_config = _build_test_workspace_resources(tmp_path)
    running_config = _build_test_resolved_options(tmp_path / "tool_trace", max_iterations=64)
    del monkeypatch
    provider = _FakeAgentProvider()

    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=write_config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
    )
    runner._preparer._create_write_agent()

    assert provider.prepare_calls
    execution_options = provider.prepare_calls[0]["execution_options"]
    assert isinstance(execution_options, ExecutionOptions)
    assert execution_options.model_name is None
    assert execution_options.max_iterations is None
    assert execution_options.web_provider == "auto"


@pytest.mark.unit
@pytest.mark.parametrize("rewrite_strategy", [RepairStrategy.REGENERATE, RepairStrategy.PATCH])
def test_apply_pending_chapter_rewrite_requires_audit_decision(
    tmp_path: Path,
    rewrite_strategy: RepairStrategy,
) -> None:
    """验证执行 rewrite 前若无审计结果会显式失败。"""

    runner = _build_test_pipeline_runner(tmp_path)
    execution_state = ChapterExecutionState(
        task=ChapterTask(index=1, title="公司介绍", skeleton="- bullet"),
        company_name="测试公司",
        current_content="现有正文",
        allowed_conditional_headings=set(),
        process_state={},
        phase="repair_1",
        retry_count=1,
        rewrite_strategy=rewrite_strategy,
    )

    with pytest.raises(RuntimeError, match="执行 rewrite 前缺少审计结果"):
        runner._chapter_execution_coordinator._apply_pending_chapter_rewrite(execution_state=execution_state)


@pytest.mark.unit
def test_prepare_next_chapter_rewrite_keeps_retry_count_until_apply_succeeds(tmp_path: Path) -> None:
    """验证 PREPARE_REWRITE 只生成待执行序号，不提前污染已完成重试计数。"""

    runner = _build_test_pipeline_runner(tmp_path)
    execution_state = ChapterExecutionState(
        task=ChapterTask(index=1, title="公司介绍", skeleton="- bullet"),
        company_name="测试公司",
        current_content="现有正文",
        allowed_conditional_headings=set(),
        process_state={"rewrite_history": []},
        audit_decision=AuditDecision(
            passed=False,
            category=AuditCategory.EVIDENCE_INSUFFICIENT,
            repair_contract=RepairContract(
                repair_strategy="patch",
            ),
            raw='{"pass": false}',
        ),
    )

    runner._chapter_execution_coordinator._prepare_next_chapter_rewrite(execution_state=execution_state)

    assert execution_state.retry_count == 0
    assert execution_state.pending_retry_count == 1
    assert execution_state.phase == "repair_1"
    assert execution_state.rewrite_strategy == RepairStrategy.PATCH
    assert execution_state.process_state["rewrite_history"] == []


@pytest.mark.unit
def test_apply_pending_chapter_rewrite_does_not_commit_retry_count_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 rewrite 执行失败时不会把待执行序号误提交为已完成重试次数。"""

    runner = _build_test_pipeline_runner(tmp_path)
    execution_state = ChapterExecutionState(
        task=ChapterTask(index=1, title="公司介绍", skeleton="- bullet"),
        company_name="测试公司",
        current_content="现有正文",
        allowed_conditional_headings=set(),
        process_state={"rewrite_history": []},
        phase="regenerate_1",
        retry_count=0,
        pending_retry_count=1,
        rewrite_strategy=RepairStrategy.REGENERATE,
        audit_decision=AuditDecision(
            passed=False,
            category=AuditCategory.EVIDENCE_INSUFFICIENT,
            repair_contract=RepairContract(
                repair_strategy="regenerate",
            ),
            raw='{"pass": false}',
        ),
    )

    monkeypatch.setattr(runner._prompter, "render_task_prompt", lambda **_kwargs: "regenerate prompt")
    monkeypatch.setattr(
        runner._prompt_runner,
        "run_regenerate_prompt",
        lambda _prompt: (_ for _ in ()).throw(RuntimeError("regenerate failed")),
    )

    with pytest.raises(RuntimeError, match="regenerate failed"):
        runner._chapter_execution_coordinator._apply_pending_chapter_rewrite(execution_state=execution_state)

    assert execution_state.retry_count == 0
    assert execution_state.pending_retry_count == 1
    assert execution_state.process_state["rewrite_history"] == []


@pytest.mark.unit
def test_create_audit_agent_prepares_scene_with_audit_execution_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证审计 scene 也通过 Runtime facade 准备执行选项。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    write_config = _build_test_write_config(tmp_path)
    workspace_config = _build_test_workspace_resources(tmp_path)
    running_config = _build_test_resolved_options(tmp_path / "tool_trace", trace_enabled=True)
    del monkeypatch
    provider = _FakeAgentProvider()

    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=write_config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
    )
    runner._preparer._create_audit_agent()

    assert provider.prepare_calls
    assert provider.prepare_calls[0]["scene_name"] == "audit"


@pytest.mark.unit
def test_create_audit_agent_does_not_override_web_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证审计 scene 准备阶段不会注入写作专用 web provider 覆盖。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    write_config = _build_test_write_config(tmp_path)
    workspace_config = _build_test_workspace_resources(tmp_path)
    running_config = _build_test_resolved_options(tmp_path / "tool_trace")
    del monkeypatch
    provider = _FakeAgentProvider()

    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=write_config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
    )
    runner._preparer._create_audit_agent()

    assert provider.prepare_calls
    assert provider.prepare_calls[0]["scene_name"] == "audit"
    assert provider.prepare_calls[0]["web_provider"] == "auto"


@pytest.mark.unit
def test_create_repair_agent_uses_repair_scene_without_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 repair 阶段会准备独立的 repair scene。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    write_config = _build_test_write_config(tmp_path)
    workspace_config = _build_test_workspace_resources(tmp_path)
    running_config = _build_test_resolved_options(tmp_path / "tool_trace")

    del monkeypatch
    provider = _FakeAgentProvider()

    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=write_config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
    )
    runner._preparer._create_repair_agent()

    assert any(call["scene_name"] == "repair" for call in provider.prepare_calls)


@pytest.mark.unit
def test_create_confirm_agent_uses_confirm_scene_with_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证证据复核阶段会准备独立的 confirm scene。"""

    write_config = _build_test_write_config(tmp_path)
    workspace_config = _build_test_workspace_resources(tmp_path)
    running_config = _build_test_resolved_options(tmp_path / "tool_trace")

    del monkeypatch
    provider = _FakeAgentProvider()

    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=write_config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
    )
    runner._preparer._create_confirm_agent()

    assert any(call["scene_name"] == "confirm" for call in provider.prepare_calls)


@pytest.mark.unit
def test_create_write_and_audit_agents_use_separate_model_overrides(tmp_path: Path) -> None:
    """验证主写作覆盖不会串到 decision/audit/confirm，审计覆盖只影响这些场景。"""

    write_config = _build_test_write_config(
        tmp_path,
        write_model_override_name="write-override",
        audit_model_override_name="audit-override",
        scene_models=_build_scene_models(
            {
                "write": ("write-override", 0.3),
                "decision": ("audit-override", 0.3),
                "regenerate": ("write-override", 0.3),
                "fix": ("write-override", 0.3),
                "repair": ("write-override", 0.3),
                "audit": ("audit-override", 0.3),
                "confirm": ("audit-override", 0.3),
            }
        ),
    )
    workspace_config = _build_test_workspace_resources(tmp_path)
    running_config = _build_test_resolved_options(tmp_path / "tool_trace")
    provider = _FakeAgentProvider()

    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=write_config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
        execution_options=ExecutionOptions(model_name="write-override", temperature=0.3, debug_sse=True),
    )
    runner._preparer._create_write_agent()
    runner._preparer._create_decision_agent()
    runner._preparer._create_audit_agent()

    write_call = next(call for call in provider.prepare_calls if call["scene_name"] == "write")
    decision_call = next(call for call in provider.prepare_calls if call["scene_name"] == "decision")
    audit_call = next(call for call in provider.prepare_calls if call["scene_name"] == "audit")
    write_execution_options = cast(ExecutionOptions, write_call["execution_options"])
    decision_execution_options = cast(ExecutionOptions, decision_call["execution_options"])
    audit_execution_options = cast(ExecutionOptions, audit_call["execution_options"])
    assert write_execution_options.model_name == "write-override"
    assert decision_execution_options.model_name == "audit-override"
    assert audit_execution_options.model_name == "audit-override"
    assert audit_execution_options.debug_sse is True


@pytest.mark.unit
def test_scene_contract_preparer_build_execution_contract_uses_audit_scene_options(tmp_path: Path) -> None:
    """验证审计 scene 构建 ExecutionContract 时不会因常量名残留而报错。"""

    provider = _FakeAgentProvider()
    preparer = SceneContractPreparer(
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        write_config=_build_test_write_config(
            tmp_path,
            write_model_override_name="write-override",
            audit_model_override_name="audit-override",
        ),
        execution_options=ExecutionOptions(model_name="base-model", temperature=0.3),
        host_session_id="write_session",
    )
    prepared_scene = provider.prepare("audit")

    contract = preparer.build_execution_contract(prepared_scene=prepared_scene, prompt_text="请审计")

    assert contract.scene_name == "audit"
    assert contract.execution_options is not None
    assert contract.execution_options.model_name == "audit-override"


@pytest.mark.unit
def test_scene_contract_preparer_filters_prompt_contributions_for_infer_scene(tmp_path: Path) -> None:
    """验证 infer scene 的 ExecutionContract 只携带 manifest 声明允许的 slot。"""

    preparer = SceneContractPreparer(
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(_FakeAgentProvider()),
        write_config=_build_test_write_config(tmp_path),
        execution_options=ExecutionOptions(model_name="base-model", temperature=0.3),
        host_session_id="write_session",
    )
    infer_scene = preparer.get_or_create_infer_scene()

    contract = preparer.build_execution_contract(prepared_scene=infer_scene, prompt_text="请完成 facet 归因")

    assert list(contract.preparation_spec.prompt_contributions.keys()) == ["fins_default_subject"]
    assert "base_user" not in contract.preparation_spec.prompt_contributions


@pytest.mark.unit
def test_build_signature_uses_scene_models(tmp_path: Path) -> None:
    """验证 scene 模型配置变化会改变运行签名。"""

    base_signature = _build_signature(
        template_text="template",
        ticker="AAPL",
        scene_models=_build_scene_models({"write": ("m1", 0.7), "audit": ("m2", 0.7)}),
        web_provider="auto",
        write_max_retries=2,
    )
    changed_signature = _build_signature(
        template_text="template",
        ticker="AAPL",
        scene_models=_build_scene_models({"write": ("m3", 0.7), "audit": ("m2", 0.7)}),
        web_provider="auto",
        write_max_retries=2,
    )

    assert base_signature != changed_signature


@pytest.mark.unit
def test_build_signature_ignores_fast_flag() -> None:
    """验证 fast 切换不会改变运行签名。"""

    fast_signature = _build_signature(
        template_text="template",
        ticker="AAPL",
        scene_models=_build_scene_models({"write": ("m1", 0.7), "audit": ("m2", 0.7)}),
        web_provider="auto",
        write_max_retries=2,
    )
    non_fast_signature = _build_signature(
        template_text="template",
        ticker="AAPL",
        scene_models=_build_scene_models({"write": ("m1", 0.7), "audit": ("m2", 0.7)}),
        web_provider="auto",
        write_max_retries=2,
    )

    assert fast_signature == non_fast_signature

@pytest.mark.unit
def test_run_manifest_from_dict_rejects_legacy_model_fields() -> None:
    """验证旧 manifest 的模型字段不会被隐式兼容。"""

    with pytest.raises(TypeError, match="write_model_name"):
        RunManifest.from_dict(
            {
                "version": "write_manifest_v1",
                "signature": "sig",
                "config": {
                    "ticker": "META",
                    "company": "META",
                    "template_path": "/tmp/template.md",
                    "output_dir": "/tmp/out",
                    "write_model_name": "deepseek-chat",
                    "audit_model_name": "deepseek-thinking",
                    "write_max_retries": 2,
                    "web_provider": "duckduckgo",
                    "resume": True,
                    "chapter_filter": "是否值得继续深研与待验证问题",
                },
                "chapter_results": {},
            }
        )


@pytest.mark.unit
def test_create_decision_agent_uses_decision_scene_with_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证第10章研究决策综合 Agent 使用独立 decision scene。"""

    write_config = _build_test_write_config(tmp_path)
    workspace_config = _build_test_workspace_resources(tmp_path)
    running_config = _build_test_resolved_options(tmp_path / "tool_trace")

    del monkeypatch
    provider = _FakeAgentProvider()

    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=write_config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
    )
    runner._preparer._create_decision_agent()

    assert any(call["scene_name"] == "decision" for call in provider.prepare_calls)


@pytest.mark.unit
def test_get_or_create_prepared_scene_raises_scene_agent_creation_error_when_prepare_returns_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 scene 准备失败时抛出稳定的 SceneAgentCreationError。"""

    write_config = _build_test_write_config(tmp_path)
    workspace_config = _build_test_workspace_resources(tmp_path)
    running_config = _build_test_resolved_options(tmp_path / "tool_trace")
    provider = _FakeAgentProvider()

    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=write_config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(provider),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
    )
    monkeypatch.setattr(runner._preparer, "_create_write_agent", lambda: None)

    with pytest.raises(SceneAgentCreationError, match="写作 Agent 创建失败: scene=write") as exc_info:
        runner._preparer.get_or_create_prepared_scene(
            scene_name="write",
            agent_label="写作 Agent",
            create_agent=runner._preparer._create_write_agent,
        )

    assert exc_info.value.scene_name == "write"
    assert exc_info.value.agent_label == "写作 Agent"


@pytest.mark.unit
def test_scene_contract_preparer_validates_only_requested_scene_model_env_vars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证模型环境变量校验按 scene 惰性执行，不会提前阻断未使用 scene。"""

    monkeypatch.delenv("DECISION_API_KEY", raising=False)
    config_loader = _FakeConfigLoader()
    config_loader.model_env_vars = {"model-decision": ("DECISION_API_KEY",)}
    preparer = SceneContractPreparer(
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(_FakeAgentProvider()),
        write_config=_build_test_write_config(tmp_path),
        execution_options=ExecutionOptions(model_name="base-model", temperature=0.3),
        host_session_id="write_session",
        config_loader=config_loader,
    )

    write_scene = preparer.get_or_create_write_scene()

    assert write_scene.scene_name == "write"
    assert config_loader.model_env_var_calls == [("model-write",)]

    with pytest.raises(SceneAgentCreationError, match="DECISION_API_KEY"):
        preparer.get_or_create_decision_scene()


@pytest.mark.unit
def test_should_skip_with_resume() -> None:
    """验证 resume 跳过逻辑。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config = WriteRunConfig(
        ticker="AAPL",
        company="Apple",
        template_path="/tmp/t.md",
        output_dir="/tmp/out",
        write_max_retries=2,
        web_provider="auto",
        resume=True,
        scene_models=_build_scene_models(
            {
                "write": ("m1", 0.7),
                "decision": ("m2", 0.7),
                "regenerate": ("m1", 0.7),
                "fix": ("m1", 0.7),
                "repair": ("m1", 0.7),
                "audit": ("m2", 0.7),
                "confirm": ("m2", 0.7),
            }
        ),
    )
    workspace_config = _build_test_workspace_resources(Path("/tmp"))
    running_config = _build_test_resolved_options(Path("/tmp/trace"))
    runner = WritePipelineRunner(
        workspace=workspace_config,
        resolved_options=running_config,
        write_config=config,
        scene_execution_acceptance_preparer=_as_scene_execution_acceptance_preparer(_FakeAgentProvider()),
        host_executor=_build_test_host_executor(),
        host_session_id="write_session",
    )

    passed_result = ChapterResult(
        index=2,
        title="公司介绍",
        status="passed",
        content="## 公司介绍",
        audit_passed=True,
    )
    failed_result = ChapterResult(
        index=3,
        title="风险分析",
        status="failed",
        content="## 风险分析",
        audit_passed=False,
    )
    fast_only_result = ChapterResult(
        index=4,
        title="竞争格局",
        status="passed",
        content="## 竞争格局",
        audit_passed=False,
    )

    assert runner._should_skip_with_resume(passed_result) is True
    assert runner._should_skip_with_resume(failed_result) is False
    assert runner._should_skip_with_resume(fast_only_result) is False

    runner._write_config.fast = True
    assert runner._should_skip_with_resume(fast_only_result) is True


@pytest.mark.unit
def test_resume_skips_overview_when_already_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 resume 模式下首章已通过时不再重写。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\n\n### 结论要点\n\n- \n\n### 证据与出处\n\n- \n\n"
        "## 公司介绍\n\nx\n\n"
        "## 来源清单\n\nx\n",
        encoding="utf-8",
    )

    existing_overview = ChapterResult(
        index=1,
        title="投资要点概览",
        status="passed",
        content="## 投资要点概览\n\n- 已有内容",
        audit_passed=True,
    )

    called: list[str] = []

    def _fake_run_single_chapter(task, **_kwargs):
        called.append(task.title)
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content=f"## {task.title}\n\n- 内容",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._preparer, "_create_write_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_audit_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_confirm_agent", lambda: object())
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda result: None)
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr(runner._store, "_persist_sources_json", lambda _: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)

    # 预置 manifest，首章已通过（使用与 runner 一致的真实 signature）
    manifest_path = runner._manifest_path
    real_signature = _build_signature(
        template_text=template_path.read_text(encoding="utf-8"),
        ticker=runner._write_config.ticker,
        scene_models=runner._write_config.scene_models,
        web_provider=runner._write_config.web_provider,
        write_max_retries=runner._write_config.write_max_retries,
    )
    manifest = RunManifest(
        version="write_manifest_v1",
        signature=real_signature,
        config=runner._write_config,
        chapter_results={"投资要点概览": existing_overview},
    )
    manifest_path.write_text(
        __import__("json").dumps(manifest.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )

    runner.run()

    assert "投资要点概览" not in called, "resume 模式下首章已通过，不应重写"


@pytest.mark.unit
def test_single_chapter_run_updates_target_and_preserves_historical_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证单章模式会更新目标章节并保留历史章节结果。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n## A\nx\n## B\nx\n## 来源清单\nx\n",
        encoding="utf-8",
    )
    runner._write_config.chapter_filter = "B"
    runner._write_config.resume = True

    old_manifest = RunManifest(
        version="write_manifest_v1",
        signature="old_signature",
        config=runner._write_config,
        chapter_results={
            "A": ChapterResult(index=2, title="A", status="passed", content="## A\nold", audit_passed=True),
            "B": ChapterResult(index=3, title="B", status="passed", content="## B\nold", audit_passed=True),
        },
    )
    _write_manifest(runner._manifest_path, old_manifest)

    updated_b = ChapterResult(
        index=3,
        title="B",
        status="passed",
        content="## B\nnew",
        audit_passed=True,
        retry_count=1,
    )
    monkeypatch.setattr(runner._preparer, "_create_write_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_audit_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_confirm_agent", lambda: object())
    monkeypatch.setattr(runner, "_run_single_chapter", lambda **_kwargs: updated_b)

    exit_code = runner.run()

    assert exit_code == 0
    saved_manifest = RunManifest.from_dict(json.loads(runner._manifest_path.read_text(encoding="utf-8")))
    assert set(saved_manifest.chapter_results.keys()) == {"A", "B"}
    assert saved_manifest.chapter_results["A"].content == "## A\nold"
    assert saved_manifest.chapter_results["B"].content == "## B\nnew"
    assert saved_manifest.chapter_results["B"].retry_count == 1


@pytest.mark.unit
def test_single_chapter_run_adds_new_chapter_and_keeps_history_when_no_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证单章模式在 no-resume 下也会新增目标章节并保留历史章节。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n## A\nx\n## H\nx\n## 来源清单\nx\n",
        encoding="utf-8",
    )
    runner._write_config.chapter_filter = "H"
    runner._write_config.resume = False

    old_manifest = RunManifest(
        version="write_manifest_v1",
        signature="old_signature",
        config=runner._write_config,
        chapter_results={
            "A": ChapterResult(index=2, title="A", status="passed", content="## A\nold", audit_passed=True),
        },
    )
    _write_manifest(runner._manifest_path, old_manifest)

    new_h = ChapterResult(
        index=3,
        title="H",
        status="passed",
        content="## H\nnew",
        audit_passed=True,
    )
    monkeypatch.setattr(runner._preparer, "_create_write_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_audit_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_confirm_agent", lambda: object())
    monkeypatch.setattr(runner, "_run_single_chapter", lambda **_kwargs: new_h)

    exit_code = runner.run()

    assert exit_code == 0
    saved_manifest = RunManifest.from_dict(json.loads(runner._manifest_path.read_text(encoding="utf-8")))
    assert set(saved_manifest.chapter_results.keys()) == {"A", "H"}
    assert saved_manifest.chapter_results["A"].content == "## A\nold"
    assert saved_manifest.chapter_results["H"].content == "## H\nnew"


@pytest.mark.unit
def test_full_run_executes_decision_chapter_as_dedicated_post_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证全量运行时第10章走专门 task，而不是普通 write_chapter。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )

    calls: list[tuple[str, str]] = []

    def _fake_run_single_chapter(*, task, prompt_name, prompt_inputs, **_kwargs):
        del prompt_inputs
        calls.append((task.title, prompt_name))
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content=f"## {task.title}\n\n### 结论要点\n\n- 内容\n\n### 证据与出处\n\n- 来源",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._preparer, "_create_write_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_decision_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_regenerate_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_fix_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_repair_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_audit_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_confirm_agent", lambda: object())
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda _result: None)
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr(runner._store, "_persist_sources_json", lambda _entries: None)
    monkeypatch.setattr(runner._report_assembler, "assemble_report", lambda *_a, **_kw: "# report")
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 0
    assert calls == [
        ("第一章", "write_chapter"),
        ("是否值得继续深研与待验证问题", "write_research_decision"),
        ("投资要点概览", "fill_overview"),
    ]


@pytest.mark.unit
def test_full_run_waits_for_all_middle_tasks_before_decision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证全文模式会等待全部中间章批次完成后才进入第10章。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 第二章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )

    middle_titles = {"第一章", "第二章"}
    state_lock = threading.Lock()
    started: set[str] = set()
    completed: list[str] = []
    persisted: list[str] = []
    both_started = threading.Event()
    release_middle = threading.Event()
    result_holder: dict[str, int] = {}
    error_holder: list[BaseException] = []

    def _fake_run_single_chapter(*, task, prompt_name, **_kwargs):
        if task.title in middle_titles:
            with state_lock:
                started.add(task.title)
                if started == middle_titles:
                    both_started.set()
            assert release_middle.wait(timeout=3), "中间章未按预期释放"
            if task.title == "第一章":
                time.sleep(0.05)
            with state_lock:
                completed.append(task.title)
            return ChapterResult(
                index=task.index,
                title=task.title,
                status="passed",
                content=f"## {task.title}\n\n### 结论要点\n\n- 内容\n\n### 证据与出处\n\n- 来源",
                audit_passed=True,
            )
        if task.title == "是否值得继续深研与待验证问题":
            assert prompt_name == "write_research_decision"
            with state_lock:
                assert set(completed) == middle_titles
                assert persisted == ["第一章", "第二章"]
            return ChapterResult(
                index=task.index,
                title=task.title,
                status="passed",
                content="## 是否值得继续深研与待验证问题\n\n### 结论要点\n\n- 动作\n\n### 证据与出处\n\n- 来源",
                audit_passed=True,
            )
        assert task.title == "投资要点概览"
        assert prompt_name == "fill_overview"
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content="## 投资要点概览\n\n### 结论要点\n\n- 内容",
            audit_passed=True,
        )

    def _run_pipeline() -> None:
        try:
            result_holder["exit_code"] = runner.run()
        except BaseException as exc:  # noqa: BLE001
            error_holder.append(exc)

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda result: persisted.append(result.title))
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr(runner._store, "_persist_sources_json", lambda _entries: None)
    monkeypatch.setattr(runner._report_assembler, "assemble_report", lambda *_a, **_kw: "# report")
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)

    run_thread = threading.Thread(target=_run_pipeline)
    run_thread.start()
    assert both_started.wait(timeout=3), "中间章未并发启动"
    release_middle.set()
    run_thread.join(timeout=5)

    assert not run_thread.is_alive()
    assert not error_holder
    assert result_holder["exit_code"] == 0
    assert persisted == ["第一章", "第二章", "是否值得继续深研与待验证问题", "投资要点概览"]


@pytest.mark.unit
def test_full_run_decision_requires_all_prior_business_chapters_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证全文模式下第10章要求前1-9章全部通过 audit，未通过则直接停止。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 第二章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def _fake_run_single_chapter(*, task, **_kwargs):
        calls.append(task.title)
        if task.title == "第二章":
            return ChapterResult(
                index=task.index,
                title=task.title,
                status="failed",
                content="## 第二章\nbad",
                audit_passed=False,
                failure_reason="evidence_insufficient",
            )
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content=f"## {task.title}\n\n### 结论要点\n\n- 内容\n\n### 证据与出处\n\n- 来源",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda _result: None)
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 4
    assert set(calls) == {"第一章", "第二章"}
    assert "是否值得继续深研与待验证问题" not in calls


@pytest.mark.unit
def test_full_run_overview_requires_chapters_one_to_ten_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证全文模式下第0章要求第1-10章全部通过 audit。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def _fake_run_single_chapter(*, task, **_kwargs):
        calls.append(task.title)
        if task.title == "是否值得继续深研与待验证问题":
            return ChapterResult(
                index=task.index,
                title=task.title,
                status="failed",
                content="## 是否值得继续深研与待验证问题\nbad",
                audit_passed=False,
                failure_reason="content_violation",
            )
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content=f"## {task.title}\n\n### 结论要点\n\n- 内容\n\n### 证据与出处\n\n- 来源",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda _result: None)
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 4
    assert calls == ["第一章", "是否值得继续深研与待验证问题"]


@pytest.mark.unit
def test_full_run_force_allows_decision_and_overview_without_audit_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 force 模式会绕过全文第10章与第0章的 audit 门禁。"""

    runner = _build_runner(tmp_path)
    runner._write_config.force = True
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )

    calls: list[str] = []

    def _fake_run_single_chapter(*, task, **_kwargs):
        calls.append(task.title)
        if task.title == "第一章":
            return ChapterResult(
                index=task.index,
                title=task.title,
                status="failed",
                content="## 第一章\nbad",
                audit_passed=False,
                failure_reason="evidence_insufficient",
            )
        if task.title == "是否值得继续深研与待验证问题":
            return ChapterResult(
                index=task.index,
                title=task.title,
                status="failed",
                content="## 是否值得继续深研与待验证问题\nbad",
                audit_passed=False,
                failure_reason="content_violation",
            )
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content="## 投资要点概览\n\n### 结论要点\n\n- 内容",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda _result: None)
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr(runner._store, "_persist_sources_json", lambda _entries: None)
    monkeypatch.setattr(runner._report_assembler, "assemble_report", lambda *_a, **_kw: "# report")
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 4
    assert calls == ["第一章", "是否值得继续深研与待验证问题", "投资要点概览"]


@pytest.mark.unit
def test_single_chapter_run_does_not_eagerly_create_unused_agents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证单章模式不会在 run 开始时预创建整条流水线的 Agent。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n## 第一章\nx\n## 来源清单\nx\n",
        encoding="utf-8",
    )
    runner._write_config.chapter_filter = "第一章"

    created_scenes: list[str] = []

    monkeypatch.setattr(runner._preparer, "_create_write_agent", lambda: created_scenes.append("write") or object())
    monkeypatch.setattr(runner._preparer, "_create_decision_agent", lambda: created_scenes.append("decision") or object())
    monkeypatch.setattr(runner._preparer, "_create_regenerate_agent", lambda: created_scenes.append("regenerate") or object())
    monkeypatch.setattr(runner._preparer, "_create_fix_agent", lambda: created_scenes.append("fix") or object())
    monkeypatch.setattr(runner._preparer, "_create_repair_agent", lambda: created_scenes.append("repair") or object())
    monkeypatch.setattr(runner._preparer, "_create_audit_agent", lambda: created_scenes.append("audit") or object())
    monkeypatch.setattr(runner._preparer, "_create_confirm_agent", lambda: created_scenes.append("confirm") or object())
    monkeypatch.setattr(
        runner,
        "_run_single_chapter",
        lambda **_kwargs: ChapterResult(
            index=2,
            title="第一章",
            status="passed",
            content="## 第一章\nok",
            audit_passed=True,
        ),
    )

    exit_code = runner.run()

    assert exit_code == 0
    assert created_scenes == []


@pytest.mark.unit
def test_single_chapter_decision_requires_existing_prior_passed_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证单章模式跑第10章时，缺少前置章节产物会直接失败。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 第二章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )
    runner._write_config.chapter_filter = "是否值得继续深研与待验证问题"
    runner._write_config.resume = True

    old_manifest = RunManifest(
        version="write_manifest_v1",
        signature="old_signature",
        config=runner._write_config,
        chapter_results={
            "第一章": ChapterResult(
                index=2,
                title="第一章",
                status="passed",
                content="## 第一章\nok",
                audit_passed=True,
            ),
            "第二章": ChapterResult(
                index=3,
                title="第二章",
                status="failed",
                content="## 第二章\nbad",
                audit_passed=False,
            ),
        },
    )
    _write_manifest(runner._manifest_path, old_manifest)
    runner._store._chapter_file_path(2, "第一章").write_text("## 第一章\nok", encoding="utf-8")

    called: list[str] = []
    monkeypatch.setattr(runner._preparer, "_create_write_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_decision_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_regenerate_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_fix_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_repair_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_audit_agent", lambda: object())
    monkeypatch.setattr(runner._preparer, "_create_confirm_agent", lambda: object())
    monkeypatch.setattr(runner, "_run_single_chapter", lambda **_kwargs: called.append("run"))
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 2
    assert called == []


@pytest.mark.unit
def test_single_chapter_decision_force_bypasses_prior_artifact_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证单章模式跑第10章时，force 会绕过前置章节门禁。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 第二章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )
    runner._write_config.chapter_filter = "是否值得继续深研与待验证问题"
    runner._write_config.force = True

    old_manifest = RunManifest(
        version="write_manifest_v1",
        signature="old_signature",
        config=runner._write_config,
        chapter_results={
            "第一章": ChapterResult(
                index=2,
                title="第一章",
                status="failed",
                content="## 第一章\nbad",
                audit_passed=False,
            )
        },
    )
    _write_manifest(runner._manifest_path, old_manifest)

    calls: list[str] = []

    def _fake_run_single_chapter(*, task, **_kwargs):
        calls.append(task.title)
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content=f"## {task.title}\n\n### 结论要点\n\n- 内容\n\n### 证据与出处\n\n- 来源",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda _result: None)
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 0
    assert calls == ["是否值得继续深研与待验证问题"]


@pytest.mark.unit
def test_single_chapter_overview_requires_existing_prior_passed_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证单章模式跑第0章时，缺少第1-10章已通过审计的产物会直接失败。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )
    runner._write_config.chapter_filter = "投资要点概览"
    runner._write_config.resume = True

    old_manifest = RunManifest(
        version="write_manifest_v1",
        signature="old_signature",
        config=runner._write_config,
        chapter_results={
            "第一章": ChapterResult(
                index=2,
                title="第一章",
                status="passed",
                content="## 第一章\nok",
                audit_passed=True,
            ),
            "是否值得继续深研与待验证问题": ChapterResult(
                index=3,
                title="是否值得继续深研与待验证问题",
                status="failed",
                content="## 是否值得继续深研与待验证问题\nbad",
                audit_passed=False,
            ),
        },
    )
    _write_manifest(runner._manifest_path, old_manifest)
    runner._store._chapter_file_path(2, "第一章").write_text("## 第一章\nok", encoding="utf-8")

    called: list[str] = []
    monkeypatch.setattr(runner, "_run_single_chapter", lambda **_kwargs: called.append("run"))
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 2
    assert called == []


@pytest.mark.unit
def test_single_chapter_overview_force_bypasses_prior_artifact_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证单章模式跑第0章时，force 会绕过前置章节门禁。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )
    runner._write_config.chapter_filter = "投资要点概览"
    runner._write_config.force = True

    old_manifest = RunManifest(
        version="write_manifest_v1",
        signature="old_signature",
        config=runner._write_config,
        chapter_results={
            "第一章": ChapterResult(
                index=2,
                title="第一章",
                status="failed",
                content="## 第一章\nbad",
                audit_passed=False,
            ),
            "是否值得继续深研与待验证问题": ChapterResult(
                index=3,
                title="是否值得继续深研与待验证问题",
                status="failed",
                content="## 是否值得继续深研与待验证问题\nbad",
                audit_passed=False,
            ),
        },
    )
    _write_manifest(runner._manifest_path, old_manifest)

    calls: list[str] = []

    def _fake_run_single_chapter(*, task, **_kwargs):
        calls.append(task.title)
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content="## 投资要点概览\n\n### 结论要点\n\n- 内容",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda _result: None)
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 0
    assert calls == ["投资要点概览"]


@pytest.mark.unit
def test_full_run_resume_reuses_manifest_when_force_flag_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证全文模式仅切换 force 时仍会复用既有 manifest。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )

    original_signature = _build_signature(
        template_text=template_path.read_text(encoding="utf-8"),
        ticker=runner._write_config.ticker,
        scene_models=runner._write_config.scene_models,
        web_provider=runner._write_config.web_provider,
        write_max_retries=runner._write_config.write_max_retries,
    )
    old_manifest = RunManifest(
        version="write_manifest_v1",
        signature=original_signature,
        config=runner._write_config,
        chapter_results={
            "第一章": ChapterResult(
                index=2,
                title="第一章",
                status="passed",
                content="## 第一章\nok",
                audit_passed=True,
            )
        },
    )
    _write_manifest(runner._manifest_path, old_manifest)

    runner._write_config.force = True
    calls: list[str] = []

    def _fake_run_single_chapter(*, task, **_kwargs):
        calls.append(task.title)
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content=f"## {task.title}\n\n### 结论要点\n\n- 内容\n\n### 证据与出处\n\n- 来源",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda _result: None)
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr(runner._store, "_persist_sources_json", lambda _entries: None)
    monkeypatch.setattr(runner._report_assembler, "assemble_report", lambda *_a, **_kw: "# report")
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.warn", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 0
    assert calls == ["是否值得继续深研与待验证问题", "投资要点概览"]


@pytest.mark.unit
def test_full_run_resume_reuses_manifest_when_fast_flag_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证全文模式仅切换 fast 时仍会复用既有 manifest。"""

    runner = _build_runner(tmp_path)
    template_path = Path(runner._write_config.template_path)
    template_path.write_text(
        "## 投资要点概览\nx\n"
        "## 第一章\nx\n"
        "## 是否值得继续深研与待验证问题\nx\n"
        "## 来源清单\nx\n",
        encoding="utf-8",
    )

    original_signature = _build_signature(
        template_text=template_path.read_text(encoding="utf-8"),
        ticker=runner._write_config.ticker,
        scene_models=runner._write_config.scene_models,
        web_provider=runner._write_config.web_provider,
        write_max_retries=runner._write_config.write_max_retries,
    )
    old_manifest = RunManifest(
        version="write_manifest_v1",
        signature=original_signature,
        config=runner._write_config,
        chapter_results={
            "第一章": ChapterResult(
                index=2,
                title="第一章",
                status="passed",
                content="## 第一章\nok",
                audit_passed=True,
            )
        },
    )
    _write_manifest(runner._manifest_path, old_manifest)

    runner._write_config.fast = True
    calls: list[str] = []

    def _fake_run_single_chapter(*, task, **_kwargs):
        calls.append(task.title)
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content=f"## {task.title}\n\n### 结论要点\n\n- 内容\n\n### 证据与出处\n\n- 来源",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_single_chapter", _fake_run_single_chapter)
    monkeypatch.setattr(runner._store, "_persist_chapter_artifacts", lambda _result: None)
    monkeypatch.setattr(runner._store, "_persist_manifest", lambda **_: None)
    monkeypatch.setattr(runner._store, "_persist_sources_json", lambda _entries: None)
    monkeypatch.setattr(runner._report_assembler, "assemble_report", lambda *_a, **_kw: "# report")
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.info", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.error", lambda *_a, **_kw: None)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.Log.warn", lambda *_a, **_kw: None)

    exit_code = runner.run()

    assert exit_code == 0
    assert calls == ["是否值得继续深研与待验证问题", "投资要点概览"]


@pytest.mark.unit
def test_persist_manifest_uses_atomic_write(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 manifest 持久化使用原子写接口。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner = _build_runner(tmp_path)
    recorder = _AtomicWriteRecorder()
    monkeypatch.setattr("dayu.services.internal.write_pipeline.artifact_store._atomic_write_text", recorder)

    manifest = RunManifest(
        version="write_manifest_v1",
        signature="sig",
        config=runner._write_config,
        chapter_results={},
    )
    chapter_results = {
        "A": ChapterResult(index=2, title="A", status="passed", content="## A\nok", audit_passed=True)
    }

    runner._store._persist_manifest(manifest=manifest, chapter_results=chapter_results)

    assert recorder.path == runner._manifest_path
    payload = json.loads(recorder.content)
    assert payload["chapter_results"]["A"]["content"] == "## A\nok"
    assert recorder.encoding == "utf-8"


@pytest.mark.unit
def test_persist_manifest_merges_existing_results(tmp_path: Path) -> None:
    """验证 manifest 持久化会保留磁盘上已有章节结果。"""

    runner = _build_runner(tmp_path)
    existing_manifest = RunManifest(
        version="write_manifest_v1",
        signature="sig",
        config=runner._write_config,
        chapter_results={
            "A": ChapterResult(index=2, title="A", status="passed", content="## A\nold", audit_passed=True)
        },
    )
    _write_manifest(runner._manifest_path, existing_manifest)

    new_manifest = RunManifest(
        version="write_manifest_v1",
        signature="sig",
        config=runner._write_config,
        chapter_results={},
    )
    runner._store._persist_manifest(
        manifest=new_manifest,
        chapter_results={
            "B": ChapterResult(index=3, title="B", status="passed", content="## B\nnew", audit_passed=True)
        },
    )

    saved_manifest = RunManifest.from_dict(json.loads(runner._manifest_path.read_text(encoding="utf-8")))
    assert set(saved_manifest.chapter_results.keys()) == {"A", "B"}
    assert saved_manifest.chapter_results["A"].content == "## A\nold"
    assert saved_manifest.chapter_results["B"].content == "## B\nnew"


@pytest.mark.unit
def test_read_manifest_from_dir_uses_manifest_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证打印报告路径读取 manifest 时也会包裹文件锁。"""

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(
        version="write_manifest_v1",
        signature="sig",
        config=_build_test_write_config(tmp_path),
        chapter_results={},
    )
    _write_manifest(output_dir / "manifest.json", manifest)

    events: list[str] = []

    @contextmanager
    def _record_lock(_output_dir: Path):
        events.append("enter")
        try:
            yield None
        finally:
            events.append("exit")

    monkeypatch.setattr("dayu.services.internal.write_pipeline.artifact_store._manifest_file_lock", _record_lock)

    loaded = _read_manifest_from_dir(output_dir)

    assert loaded is not None
    assert events == ["enter", "exit"]


@pytest.mark.unit
def test_manifest_file_lock_uses_msvcrt_when_fcntl_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 Windows 分支会改用 msvcrt 真正加锁，而不是静默跳过。"""

    class _FakeMsvcrt:
        """记录 locking 调用的 Windows 锁实现桩。"""

        LK_NBLCK = 1
        LK_UNLCK = 2

        def __init__(self) -> None:
            """初始化调用记录。"""

            self.calls: list[tuple[int, int]] = []

        def locking(self, fd: int, mode: int, size: int) -> None:
            """记录一次加锁或解锁。"""

            del fd
            self.calls.append((mode, size))

    fake_msvcrt = _FakeMsvcrt()
    monkeypatch.setattr(artifact_store_module.file_lock_module, "_FCNTL", None)
    monkeypatch.setattr(artifact_store_module.file_lock_module, "_MSVCRT", fake_msvcrt)

    output_dir = tmp_path / "out"
    with artifact_store_module._manifest_file_lock(output_dir):
        pass

    lock_path = output_dir / ".manifest.lock"
    assert lock_path.exists()
    assert lock_path.read_text(encoding="utf-8") == "\0"
    assert fake_msvcrt.calls == [
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_UNLCK, 1),
    ]


@pytest.mark.unit
def test_manifest_file_lock_retries_until_windows_lock_is_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 blocking=True 在 Windows 上会持续重试，直到真正拿到锁。"""

    class _RetryingMsvcrt:
        """前两次竞争失败、随后成功的 Windows 锁实现桩。"""

        LK_NBLCK = 1
        LK_UNLCK = 2

        def __init__(self) -> None:
            """初始化失败次数与调用记录。"""

            self.calls: list[tuple[int, int]] = []
            self._remaining_contention_failures = 2

        def locking(self, fd: int, mode: int, size: int) -> None:
            """记录调用，并在前两次加锁时模拟锁竞争。"""

            del fd
            self.calls.append((mode, size))
            if mode == self.LK_NBLCK and self._remaining_contention_failures > 0:
                self._remaining_contention_failures -= 1
                raise BlockingIOError(
                    artifact_store_module.file_lock_module.errno.EAGAIN,
                    "locked",
                )

    fake_msvcrt = _RetryingMsvcrt()
    sleep_calls: list[float] = []
    monkeypatch.setattr(artifact_store_module.file_lock_module, "_FCNTL", None)
    monkeypatch.setattr(artifact_store_module.file_lock_module, "_MSVCRT", fake_msvcrt)
    monkeypatch.setattr(artifact_store_module.file_lock_module.time, "sleep", sleep_calls.append)

    with artifact_store_module._manifest_file_lock(tmp_path / "out"):
        pass

    assert fake_msvcrt.calls == [
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_UNLCK, 1),
    ]
    assert sleep_calls == [
        artifact_store_module.file_lock_module._WINDOWS_LOCK_RETRY_INTERVAL_SEC,
        artifact_store_module.file_lock_module._WINDOWS_LOCK_RETRY_INTERVAL_SEC,
    ]


@pytest.mark.unit
def test_manifest_file_lock_raises_when_no_platform_lock_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证没有任何平台锁实现时会显式失败，而不是静默放过。"""

    monkeypatch.setattr(artifact_store_module.file_lock_module, "_FCNTL", None)
    monkeypatch.setattr(artifact_store_module.file_lock_module, "_MSVCRT", None)

    with pytest.raises(OSError, match="当前平台不支持 manifest 文件锁"):
        with artifact_store_module._manifest_file_lock(tmp_path / "out"):
            pass


@pytest.mark.unit
def test_build_fallback_result_marks_failed(tmp_path: Path) -> None:
    """验证兜底结果会标记为失败，且默认不再回退为章节骨架。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner = _build_runner(tmp_path)
    result = runner._build_fallback_result(
        task=ChapterTask(index=3, title="风险分析", skeleton="## 风险分析\\n\\n### 结论要点\\n- "),
        reason="boom",
    )

    assert result.status == "failed"
    assert result.audit_passed is False
    assert result.failure_reason == "boom"
    assert result.content == ""
    assert result.process_state["final_stage"] == "runtime_error"


@pytest.mark.unit
def test_load_latest_failed_chapter_content_prefers_latest_write_artifact(tmp_path: Path) -> None:
    """验证章节失败时会优先保留最新 write 中间稿。"""

    runner = _build_runner(tmp_path)
    task = ChapterTask(index=5, title="最近一年关键变化与当前阶段", skeleton="## 最近一年关键变化与当前阶段")

    initial_path = runner._store._chapter_phase_artifact_path(
        index=task.index,
        title=task.title,
        artifact_name="initial_write",
        extension="md",
    )
    repair_path = runner._store._chapter_phase_artifact_path(
        index=task.index,
        title=task.title,
        artifact_name="repair_1_write",
        extension="md",
    )
    initial_path.parent.mkdir(parents=True, exist_ok=True)
    initial_path.write_text("初稿", encoding="utf-8")
    repair_path.write_text("修复稿", encoding="utf-8")

    assert runner._store._load_latest_failed_chapter_content(task) == "修复稿"


@pytest.mark.unit
def test_persist_chapter_artifacts_skips_failed_empty_result(tmp_path: Path) -> None:
    """验证失败且正文为空时，不落最终章节文件。"""

    runner = _build_runner(tmp_path)
    result = ChapterResult(
        index=5,
        title="最近一年关键变化与当前阶段",
        status="failed",
        content="",
        audit_passed=False,
    )

    runner._store._persist_chapter_artifacts(result)

    assert not runner._store._chapter_file_path(result.index, result.title).exists()


# ---------------------------------------------------------------------------
# _extract_markdown_content 测试
# ---------------------------------------------------------------------------


class TestExtractMarkdownContent:
    """验证 _extract_markdown_content 对各类输入的提取行为。"""

    def test_single_fence(self) -> None:
        """单对 fence 正常提取。"""
        raw = "```markdown\n第一段\n```"
        assert _extract_markdown_content(raw) == "第一段"

    def test_nested_code_fence(self) -> None:
        """嵌套代码块场景：内部含 json 代码块，贪婪正则应返回完整内容。"""
        raw = (
            "```markdown\n"
            "第一段\n\n"
            "```json\n{\"key\": \"value\"}\n```\n\n"
            "第二段\n"
            "```"
        )
        result = _extract_markdown_content(raw)
        assert "第一段" in result
        assert "第二段" in result
        assert '"key"' in result

    def test_no_fence_returns_stripped(self) -> None:
        """无 fence 时返回 strip 后原文。"""
        raw = "  纯文本内容  "
        assert _extract_markdown_content(raw) == "纯文本内容"

    def test_empty_fence(self) -> None:
        """空 fence 返回空字符串。"""
        raw = "```markdown\n```"
        assert _extract_markdown_content(raw) == ""

    def test_fence_with_language_tag(self) -> None:
        """带 markdown 标签的 fence。"""
        raw = "```Markdown\n内容\n```"
        assert _extract_markdown_content(raw) == "内容"

    def test_truncated_output_short_content(self) -> None:
        """模拟 ASML 场景：模型被截断只输出极短内容 + 嵌套 fence。"""
        raw = "```markdown\n在2025```\nmore content after\n```"
        result = _extract_markdown_content(raw)
        # 贪婪正则应匹配到最后一个 ```，包含完整内容
        assert "在2025" in result
        assert "more content after" in result


@pytest.mark.unit
def test_parse_audit_decision_prefers_payload_repair_contract() -> None:
    """验证 E3 会强制收口为 regenerate，不允许被自定义 repair_contract 降回 patch。"""

    raw = (
        '{"pass": false, "class": "evidence_insufficient", '
        '"violations": [{"rule": "E3", "reason": "来源不可追溯"}], '
        '"notes": ["请补证"], '
        '"repair_contract": {'
        '"contract_version": "custom_v9", '
        '"preferred_tool_action": "read_section", '
        '"retry_scope": "slot_patch", '
        '"missing_evidence_slots": [{"slot_id": "rev_split"}]'
        '}}'
    )

    decision = _parse_audit_decision(raw)

    assert decision.passed is False
    assert decision.repair_contract.contract_version == "custom_v9"
    assert decision.repair_contract.preferred_tool_action == "regenerate_chapter"
    assert decision.repair_contract.repair_strategy == "regenerate"
    assert decision.repair_contract.retry_scope == "chapter_regenerate"
    assert decision.repair_contract.missing_evidence_slots[0].slot_id == "rev_split"


@pytest.mark.unit
def test_parse_repair_plan_accepts_json_object() -> None:
    """验证 repair patch plan 可被正确解析。"""

    raw = json.dumps(
        {
            "patches": [
                {
                    "target_excerpt": "旧句子",
                    "replacement": "新句子",
                    "reason": "删除无证据细节",
                }
            ],
            "notes": ["仅改一处"],
        },
        ensure_ascii=False,
    )

    plan = _parse_repair_plan(raw)

    assert plan["patches"][0]["target_excerpt"] == "旧句子"
    assert plan["patches"][0]["replacement"] == "新句子"
    assert plan["patches"][0]["target_kind"] == "substring"
    assert plan["patches"][0]["target_section_heading"] == ""
    assert plan["patches"][0]["occurrence_index"] is None
    assert plan["notes"] == ["仅改一处"]


@pytest.mark.unit
def test_audit_module_no_longer_hosts_repair_executor_logic() -> None:
    """验证 repair 解析与 apply 唯一真源已收敛到 repair_executor 模块。"""

    assert not hasattr(audit_rules_module, "_parse_repair_plan")
    assert not hasattr(audit_rules_module, "_apply_repair_plan")
    assert not hasattr(audit_rules_module, "_apply_repair_plan_with_details")


@pytest.mark.unit
def test_parse_repair_plan_accepts_optional_section_heading_and_occurrence() -> None:
    """验证 repair patch plan 可解析 section heading 与 occurrence index。"""

    raw = json.dumps(
        {
            "patches": [
                {
                    "target_excerpt": "旧句子",
                    "target_kind": "bullet",
                    "target_section_heading": "详细情况",
                    "occurrence_index": 2,
                    "replacement": "新句子",
                    "reason": "删除重复短语",
                }
            ],
            "notes": [],
        },
        ensure_ascii=False,
    )

    plan = _parse_repair_plan(raw)

    assert plan["patches"][0]["target_kind"] == "bullet"
    assert plan["patches"][0]["target_section_heading"] == "详细情况"
    assert plan["patches"][0]["occurrence_index"] == 2


@pytest.mark.unit
def test_parse_repair_plan_rejects_invalid_target_kind() -> None:
    """验证 repair patch plan 的 target_kind 非法时会报错。"""

    raw = json.dumps(
        {
            "patches": [
                {
                    "target_excerpt": "旧句子",
                    "target_kind": "sentence",
                    "replacement": "新句子",
                    "reason": "测试",
                }
            ],
            "notes": [],
        },
        ensure_ascii=False,
    )

    with pytest.raises(ValueError, match="target_kind 非法"):
        _parse_repair_plan(raw)


@pytest.mark.unit
def test_parse_repair_plan_accepts_fenced_json_with_prefix_text() -> None:
    """验证 repair patch plan 可容忍前缀说明和 fenced json。"""

    raw = (
        "现在我需要修复正文中的客户集中度断言。\n\n"
        "```json\n"
        "{\n"
        '  "patches": [\n'
        '    {"target_excerpt": "旧句子", "replacement": "新句子", "reason": "删除无证据细节"}\n'
        "  ],\n"
        '  "notes": ["仅改一处"]\n'
        "}\n"
        "```"
    )

    plan = _parse_repair_plan(raw)

    assert plan["patches"][0]["target_excerpt"] == "旧句子"
    assert plan["patches"][0]["replacement"] == "新句子"
    assert plan["notes"] == ["仅改一处"]


@pytest.mark.unit
def test_run_single_chapter_persists_repair_raw_output_when_repair_json_invalid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 repair 输出非法时也会先落盘原始返回。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner = _build_runner(tmp_path)
    write_sequence = _SequenceCaller(
        [
            "初稿内容足够长度用于测试场景。\n\n### 证据与出处\n- [1] 年报",
        ]
    )
    audit_sequence = _SequenceCaller(
        [
            AuditDecision(
                passed=False,
                category=AuditCategory.EVIDENCE_INSUFFICIENT,
                violations=[Violation(rule=AuditRuleCode.UNKNOWN, reason="关键数字无证据")],
                notes=[],
                repair_contract=RepairContract(
                    contract_version="repair_contract_v1",
                    repair_strategy="patch",
                    retry_scope="targeted_evidence_patch",
                ),
                raw='{"pass": false}',
            )
        ]
    )

    monkeypatch.setattr(runner._prompt_runner, "run_write_prompt", write_sequence)
    monkeypatch.setattr(runner._chapter_audit_coordinator, "_audit_chapter", audit_sequence)
    monkeypatch.setattr(
        runner._prompt_runner,
        "run_repair_prompt",
        lambda _prompt: (_ for _ in ()).throw(
            RepairOutputError(
                "repair 输出非法: repair 输出必须是 JSON 对象",
                raw_output='前缀说明\n```json\n{"patches":[]}\n```',
            )
        ),
    )

    task = ChapterTask(index=2, title="公司介绍", skeleton="## 公司介绍")
    with pytest.raises(RepairOutputError):
        runner._run_single_chapter(
            task=task,
            company_name="Apple",
            prompt_name="write_chapter",
            prompt_inputs=runner._prompter._build_chapter_prompt_inputs(task=task, company_name="Apple"),
        )

    chapters_dir = Path(runner._write_config.output_dir) / "chapters"
    assert (chapters_dir / "01_公司介绍.repair_1_input_write.md").read_text(encoding="utf-8").startswith("初稿")
    repair_context = json.loads((chapters_dir / "01_公司介绍.repair_1_context.json").read_text(encoding="utf-8"))
    assert repair_context["chapter"] == "公司介绍"
    assert (chapters_dir / "01_公司介绍.repair_1_repair_plan.json").read_text(encoding="utf-8").startswith(
        "前缀说明"
    )


@pytest.mark.unit
def test_run_single_chapter_persists_repair_apply_result_when_all_patches_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 repair apply 全失败时仍会落盘输入快照与逐条失败原因。"""

    runner = _build_runner(tmp_path)
    write_sequence = _SequenceCaller(
        [
            "## 公司介绍\n\n### 结论要点\n- 初稿内容足够长度用于测试场景\n\n### 证据与出处\n- SEC EDGAR | Form 10-K | Filed 2025-01-01 | Accession 0000000001-25-000001",
        ]
    )
    audit_sequence = _SequenceCaller(
        [
            AuditDecision(
                passed=False,
                category=AuditCategory.EVIDENCE_INSUFFICIENT,
                violations=[
                    Violation(
                        rule=AuditRuleCode.S7,
                        severity="high",
                    )
                ],
                repair_contract=RepairContract(
                    contract_version="repair_contract_v1",
                    preferred_tool_action="repair_chapter",
                    retry_scope="targeted_patch",
                    repair_strategy="patch",
                ),
                raw='{"pass": false}',
            )
        ]
    )

    monkeypatch.setattr(runner._prompt_runner, "run_write_prompt", write_sequence)
    monkeypatch.setattr(runner._chapter_audit_coordinator, "_audit_chapter", audit_sequence)
    monkeypatch.setattr(
        runner._prompt_runner,
        "run_repair_prompt",
        lambda _prompt: (
            {
                "patches": [
                    {
                        "target_excerpt": "初稿内容足够长度用于测试场景",
                        "target_kind": "bullet",
                        "target_section_heading": "根本不存在的标题",
                        "occurrence_index": 1,
                        "replacement": "修复后内容",
                        "reason": "标题写错",
                    }
                ],
                "notes": [],
            },
            json.dumps(
                {
                    "patches": [
                        {
                            "target_excerpt": "初稿内容足够长度用于测试场景",
                            "target_kind": "bullet",
                            "target_section_heading": "根本不存在的标题",
                            "occurrence_index": 1,
                            "replacement": "修复后内容",
                            "reason": "标题写错",
                        }
                    ],
                    "notes": [],
                },
                ensure_ascii=False,
            ),
        ),
    )

    task = ChapterTask(index=2, title="公司介绍", skeleton="## 公司介绍")
    with pytest.raises(ValueError, match="所有 1 个 patch 均失败"):
        runner._run_single_chapter(
            task=task,
            company_name="Apple",
            prompt_name="write_chapter",
            prompt_inputs=runner._prompter._build_chapter_prompt_inputs(task=task, company_name="Apple"),
        )

    chapters_dir = Path(runner._write_config.output_dir) / "chapters"
    assert (chapters_dir / "01_公司介绍.repair_1_input_write.md").read_text(encoding="utf-8").startswith(
        "## 公司介绍"
    )
    repair_apply_result = json.loads((chapters_dir / "01_公司介绍.repair_1_apply_result.json").read_text(encoding="utf-8"))
    assert repair_apply_result["all_failed"] is True
    assert repair_apply_result["applied_count"] == 0
    assert repair_apply_result["patch_results"][0]["status"] == "skipped"
    assert repair_apply_result["patch_results"][0]["skip_reason"] == "patch[1] 未找到真实 section 标题: 根本不存在的标题"


@pytest.mark.unit
def test_apply_repair_plan_replaces_exact_excerpt_once() -> None:
    """验证 repair patch plan 会按精确 excerpt 做单次替换。"""

    content = "第一句。\n第二句。\n第三句。"
    plan = {
        "patches": [
            {
                "target_excerpt": "第二句。",
                "replacement": "第二句（已修复）。",
                "reason": "测试",
            }
        ],
        "notes": [],
    }

    result = _apply_repair_plan(chapter_markdown=content, repair_plan=plan)

    assert result == "第一句。\n第二句（已修复）。\n第三句。"


@pytest.mark.unit
def test_apply_repair_plan_with_details_records_skip_reason() -> None:
    """验证带明细的 repair 应用结果会保留逐条跳过原因。"""

    content = "第一句。\n第二句。\n第三句。"
    plan = {
        "patches": [
            {
                "target_excerpt": "不存在的片段",
                "replacement": "替换后",
                "reason": "必定失败",
            }
        ],
        "notes": [],
    }

    result = _apply_repair_plan_with_details(chapter_markdown=content, repair_plan=plan)

    assert result.all_failed is True
    assert result.applied_count == 0
    assert result.skipped_count == 1
    assert result.patch_results[0].status == "skipped"
    assert result.patch_results[0].skip_reason == "patch[1] 目标片段命中次数异常: expected=1 actual=0"


@pytest.mark.unit
def test_apply_repair_plan_with_details_rejects_delete_claim_substring_patch() -> None:
    """验证 delete_claim 不能再通过 substring patch 删除半句。"""

    content = "### 详细情况\n- 地缘政治风险与出口管制可能突然中断订单或现金流，显著影响股东回报的可持续性。\n"
    plan = {
        "patches": [
            {
                "target_excerpt": "地缘政治风险与出口管制可能突然中断订单或现金流",
                "target_kind": "substring",
                "target_section_heading": "详细情况",
                "replacement": "",
                "reason": "错误示例：只删半句",
            }
        ],
        "notes": [],
    }
    repair_contract = RepairContract(
        remediation_actions=[
            RemediationAction(
                action_id="slot_1_E1",
                rule=AuditRuleCode.E1,
                excerpt="地缘政治风险与出口管制可能突然中断订单或现金流",
                resolution_mode=RepairResolutionMode.DELETE_CLAIM.value,
            )
        ]
    )

    result = _apply_repair_plan_with_details(
        chapter_markdown=content,
        repair_plan=plan,
        repair_contract=repair_contract,
    )

    assert result.all_failed is True
    assert result.patch_results[0].status == "skipped"
    assert result.patch_results[0].skip_reason == (
        "patch[1] delete_claim 不允许使用 substring，请改为 line/bullet/paragraph"
    )


@pytest.mark.unit
def test_apply_repair_plan_with_details_rejects_delete_claim_replacement_that_keeps_claim() -> None:
    """验证 delete_claim patch 若仍保留原 unsupported claim，会被执行器拒绝。"""

    content = "### 详细情况\n地缘政治风险与出口管制可能突然中断订单或现金流，显著影响股东回报的可持续性。\n"
    plan = {
        "patches": [
            {
                "target_excerpt": "地缘政治风险与出口管制可能突然中断订单或现金流，显著影响股东回报的可持续性。",
                "target_kind": "paragraph",
                "target_section_heading": "详细情况",
                "replacement": "地缘政治风险与出口管制可能突然中断订单或现金流，但影响程度仍需观察。",
                "reason": "错误示例：弱化但保留 unsupported claim",
            }
        ],
        "notes": [],
    }
    repair_contract = RepairContract(
        remediation_actions=[
            RemediationAction(
                action_id="slot_1_E1",
                rule=AuditRuleCode.E1,
                excerpt="地缘政治风险与出口管制可能突然中断订单或现金流",
                resolution_mode=RepairResolutionMode.DELETE_CLAIM.value,
            )
        ]
    )

    result = _apply_repair_plan_with_details(
        chapter_markdown=content,
        repair_plan=plan,
        repair_contract=repair_contract,
    )

    assert result.all_failed is True
    assert result.patch_results[0].status == "skipped"
    assert result.patch_results[0].skip_reason == (
        "patch[1] delete_claim replacement 仍保留 unsupported claim: 地缘政治风险与出口管制可能突然中断订单或现金流"
    )


@pytest.mark.unit
def test_apply_repair_plan_can_disambiguate_with_section_heading_and_occurrence() -> None:
    """验证 repair patch plan 可借助 section heading 与 occurrence index 命中重复片段。"""

    content = (
        "## 章节\n\n"
        "### 结论要点\n"
        "- 2025年加密交易量市场份额翻倍至6.4%\n\n"
        "### 详细情况\n"
        "- 第一处：2025年加密交易量市场份额翻倍至6.4%\n"
        "- 第二处：2025年加密交易量市场份额翻倍至6.4%\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "2025年加密交易量市场份额翻倍至6.4%",
                "target_section_heading": "详细情况",
                "occurrence_index": 2,
                "replacement": "2025年加密交易量市场份额显著增长",
                "reason": "删除不可锚定数字",
            }
        ],
        "notes": [],
    }

    result = _apply_repair_plan(chapter_markdown=content, repair_plan=plan)

    assert "### 结论要点\n- 2025年加密交易量市场份额翻倍至6.4%" in result
    assert "- 第一处：2025年加密交易量市场份额翻倍至6.4%" in result
    assert "- 第二处：2025年加密交易量市场份额显著增长" in result


@pytest.mark.unit
def test_apply_repair_plan_can_fallback_to_normalized_bullet_match() -> None:
    """验证 repair patch plan 在 exact excerpt 轻微漂移时可借助 bullet 匹配兜底。"""

    content = (
        "## 章节\n\n"
        "### 结论要点\n"
        "- 主要压力来源：短视频平台可能影响用户注意力。\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "主要压力来源：短视频平台可能影响用户注意力",
                "target_kind": "bullet",
                "target_section_heading": "结论要点",
                "occurrence_index": 1,
                "replacement": "- 主要压力来源：短视频平台分流用户注意力。",
                "reason": "测试规范化 bullet 匹配",
            }
        ],
        "notes": [],
    }

    result = _apply_repair_plan(chapter_markdown=content, repair_plan=plan)

    assert "- 主要压力来源：短视频平台分流用户注意力。" in result


@pytest.mark.unit
def test_apply_repair_plan_removes_empty_bullet_after_deletion() -> None:
    """验证删除 bullet 内容后，会清理整条空 bullet 而不留下残片。"""

    content = (
        "### 详细情况\n"
        "- 收入地理分布：美国与加拿大 788.66 亿美元。\n"
        "- 其他内容。\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "- 收入地理分布：美国与加拿大 788.66 亿美元。\n",
                "target_kind": "bullet",
                "target_section_heading": "详细情况",
                "replacement": "",
                "reason": "删除高风险条件项",
            }
        ],
        "notes": [],
    }

    result = _apply_repair_plan(chapter_markdown=content, repair_plan=plan)

    assert "收入地理分布" not in result
    assert "- 其他内容。" in result
    assert "\n-\n" not in result


@pytest.mark.unit
def test_apply_repair_plan_cleans_bullet_punctuation_residue() -> None:
    """验证 patch 删除后若留下空 bullet 标点残片，会被结构清理移除。"""

    content = (
        "### 详细情况\n"
        "- 收入地理分布：美国与加拿大 788.66 亿美元。\n"
        "- 其他内容。\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "收入地理分布：美国与加拿大 788.66 亿美元",
                "target_kind": "substring",
                "target_section_heading": "详细情况",
                "replacement": "",
                "reason": "模拟模型只删除 bullet 正文",
            }
        ],
        "notes": [],
    }

    result = _apply_repair_plan(chapter_markdown=content, repair_plan=plan)

    assert "- 。" not in result
    assert "- 其他内容。" in result


@pytest.mark.unit
def test_apply_repair_plan_cleans_whitespace_only_gap_after_delete_claim() -> None:
    """验证删除段落与标题行后，残留的缩进空白行会被归一化并压平。"""

    content = (
        "### 详细情况\n\n"
        "#### 为什么现在更适合这样做\n\n"
        "- 现在更应该继续研究、暂缓，还是直接放弃\n"
        "  现在更应该继续研究。\n\n"
        "- 这个选择最主要是基于什么判断\n"
        "  公司通过Infinera收购和NVIDIA合作明确了AI基础设施的转型路径，提供了长期增长叙事，但短期盈利能力仍需验证。\n\n"
        "- 为什么现在不是另外两个动作\n"
        "  现在不是放弃，因为关键风险已识别。\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "公司通过Infinera收购和NVIDIA合作明确了AI基础设施的转型路径，提供了长期增长叙事，但短期盈利能力仍需验证。",
                "target_kind": "paragraph",
                "target_section_heading": "为什么现在更适合这样做",
                "occurrence_index": 1,
                "replacement": "",
                "reason": "删除 unsupported claim 段落",
            },
            {
                "target_excerpt": "这个选择最主要是基于什么判断",
                "target_kind": "line",
                "target_section_heading": "为什么现在更适合这样做",
                "occurrence_index": 1,
                "replacement": "",
                "reason": "删除孤立标题行",
            },
        ],
        "notes": [],
    }

    result = _apply_repair_plan(chapter_markdown=content, repair_plan=plan)

    assert "这个选择最主要是基于什么判断" not in result
    assert "\n  \n" not in result
    assert "现在更应该继续研究。\n\n- 为什么现在不是另外两个动作" in result


@pytest.mark.unit
def test_apply_repair_plan_forbids_deleting_evidence_line() -> None:
    """验证 repair patch 不允许删除“证据与出处”中的证据行。"""

    content = (
        "### 详细情况\n"
        "- 正文内容。\n\n"
        "### 证据与出处\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-01-01 | Accession 0000000000-25-000001 | Part I - Item 1 - Business\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "- SEC EDGAR | Form 10-K | Filed 2025-01-01 | Accession 0000000000-25-000001 | Part I - Item 1 - Business\n",
                "target_kind": "line",
                "target_section_heading": "证据与出处",
                "replacement": "",
                "reason": "错误示例：试图直接删除证据行",
            }
        ],
        "notes": [],
    }

    with pytest.raises(ValueError, match="所有 1 个 patch 均失败"):
        _apply_repair_plan(chapter_markdown=content, repair_plan=plan)


@pytest.mark.unit
def test_apply_repair_plan_forbids_replacing_evidence_line() -> None:
    """验证 repair patch 不允许在证据小节内替换 evidence line。"""

    content = (
        "### 详细情况\n"
        "- 正文内容。\n\n"
        "### 证据与出处\n"
        "- 旧证据行\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "- 旧证据行\n",
                "target_kind": "line",
                "target_section_heading": "证据与出处",
                "replacement": "- 新证据行\n",
                "reason": "把证据行替换成更稳的 locator",
            }
        ],
        "notes": [],
    }

    with pytest.raises(ValueError, match="所有 1 个 patch 均失败"):
        _apply_repair_plan(chapter_markdown=content, repair_plan=plan)


@pytest.mark.unit
def test_apply_repair_plan_forbids_modifying_evidence_without_section_heading() -> None:
    """验证即使未显式指定 section heading，命中证据小节也会被拒绝。"""

    content = (
        "### 详细情况\n"
        "- 正文内容。\n\n"
        "### 证据与出处\n"
        "- 旧证据行\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "- 旧证据行\n",
                "target_kind": "line",
                "replacement": "- 新证据行\n",
                "reason": "错误示例：不写 section heading 也想改 evidence",
            }
        ],
        "notes": [],
    }

    with pytest.raises(ValueError, match="所有 1 个 patch 均失败"):
        _apply_repair_plan(chapter_markdown=content, repair_plan=plan)


# ── _normalize_heading_text ──────────────────────────────────────────────────


@pytest.mark.unit
def test_normalize_heading_text_fullwidth_brackets() -> None:
    """验证全角括号被归一化为半角括号。"""

    assert _normalize_heading_text("竞争地位的关键支撑依据（限已披露）") == "竞争地位的关键支撑依据(限已披露)"


@pytest.mark.unit
def test_normalize_heading_text_mixed_punctuation() -> None:
    """验证全角逗号、顿号被归一化为半角逗号，多余空白被合并。"""

    assert _normalize_heading_text("关键指标：A、B，C") == "关键指标:A,B,C"
    assert _normalize_heading_text("A  B　C") == "A B C"


@pytest.mark.unit
def test_normalize_heading_text_common_cn_en_punctuation() -> None:
    """验证常见中英文标题标点会被统一归一化。"""

    assert _normalize_heading_text('解释“为什么是现在”时，最先该看什么') == '解释"为什么是现在"时,最先该看什么'
    assert _normalize_heading_text("解释《为什么是现在》时：最先该看什么？") == '解释"为什么是现在"时:最先该看什么?'
    assert _normalize_heading_text("【关键变化】公司现在走到了哪一步；为什么偏偏是现在。") == "[关键变化]公司现在走到了哪一步;为什么偏偏是现在."


@pytest.mark.unit
def test_matches_skeleton_structure_allows_common_cn_en_punctuation_difference() -> None:
    """验证标题仅存在常见中英文标点差异时不会被误判为 P1。"""

    skeleton = (
        "## 最近一年关键变化与当前阶段\n\n"
        "### 结论要点\n\n"
        "### 详细情况\n\n"
        "#### 解释“为什么是现在”时，最先该看什么\n\n"
        "### 证据与出处\n"
    )
    content = (
        "## 最近一年关键变化与当前阶段\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 详细情况\n\n"
        '#### 解释"为什么是现在"时，最先该看什么\n\n'
        "正文\n\n"
        "### 证据与出处\n\n"
        "- test\n"
    )

    assert _matches_skeleton_structure(content, skeleton) is True


# ── _apply_repair_plan: 标题归一化匹配 ──────────────────────────────────────


@pytest.mark.unit
def test_apply_repair_plan_section_heading_with_fullwidth_brackets() -> None:
    """验证 section heading 在全半角括号不一致时仍能命中。"""

    content = (
        "## 竞争格局\n\n"
        "### 竞争地位的关键支撑依据(限已披露、仅写最关键2-3条)\n"
        "- 待填充\n\n"
        "### 另一个章节\n"
        "- 内容\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "待填充",
                "target_section_heading": "竞争地位的关键支撑依据（限已披露、仅写最关键2-3条）",
                "replacement": "独占EUV光刻技术，护城河极深",
                "reason": "测试全半角括号归一化",
            }
        ],
        "notes": [],
    }

    result = _apply_repair_plan(chapter_markdown=content, repair_plan=plan)

    assert "独占EUV光刻技术，护城河极深" in result
    assert "待填充" not in result


# ── _apply_repair_plan: 单 patch 容错 ───────────────────────────────────────


@pytest.mark.unit
def test_apply_repair_plan_skips_failed_patch_applies_rest() -> None:
    """验证单个 patch 失败时跳过该 patch，其余 patch 正常应用。"""

    content = "第一句。\n第二句。\n第三句。"
    plan = {
        "patches": [
            {
                "target_excerpt": "不存在的片段",
                "replacement": "替换后",
                "reason": "必定失败",
            },
            {
                "target_excerpt": "第二句。",
                "replacement": "第二句（已修复）。",
                "reason": "正常 patch",
            },
        ],
        "notes": [],
    }

    result = _apply_repair_plan(chapter_markdown=content, repair_plan=plan)

    assert "第二句（已修复）。" in result
    assert "不存在的片段" not in result


@pytest.mark.unit
def test_apply_repair_plan_raises_when_all_patches_fail() -> None:
    """验证所有 patch 均失败时才抛出 ValueError。"""

    content = "第一句。\n第二句。\n第三句。"
    plan = {
        "patches": [
            {
                "target_excerpt": "不存在的A",
                "replacement": "替换A",
                "reason": "失败1",
            },
            {
                "target_excerpt": "不存在的B",
                "replacement": "替换B",
                "reason": "失败2",
            },
        ],
        "notes": [],
    }

    with pytest.raises(ValueError, match="所有 2 个 patch 均失败"):
        _apply_repair_plan(chapter_markdown=content, repair_plan=plan)


@pytest.mark.unit
def test_apply_repair_plan_skips_unfound_section_heading() -> None:
    """验证 section heading 找不到时跳过该 patch 而非整章降级。"""

    content = (
        "## 竞争格局\n\n"
        "### 真实标题\n"
        "- 内容A\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "内容A",
                "target_section_heading": "根本不存在的标题",
                "replacement": "内容A（修复）",
                "reason": "section heading 找不到",
            },
        ],
        "notes": [],
    }

    # 唯一一个 patch 失败 → 应抛出 ValueError
    with pytest.raises(ValueError, match="所有 1 个 patch 均失败"):
        _apply_repair_plan(chapter_markdown=content, repair_plan=plan)


# ── _normalize_heading_text: ### 前缀剥离 ───────────────────────────────────


@pytest.mark.unit
def test_normalize_heading_text_strips_markdown_prefix() -> None:
    """验证归一化会剥离误带的 ### 前缀。"""

    assert _normalize_heading_text("### 证据与出处") == "证据与出处"
    assert _normalize_heading_text("## 竞争格局") == "竞争格局"
    assert _normalize_heading_text("#### 行业竞争格局与利润池分布") == "行业竞争格局与利润池分布"


# ── _find_markdown_section_span / _apply_repair_plan: ### 前缀容错 ────────


@pytest.mark.unit
def test_apply_repair_plan_heading_with_markdown_prefix() -> None:
    """验证即使 heading 带 ### 前缀，命中证据小节也会被拒绝。"""

    content = (
        "## 竞争格局\n\n"
        "### 证据与出处\n"
        "- 旧证据行\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "旧证据行",
                "target_section_heading": "### 证据与出处",
                "replacement": "新证据行",
                "reason": "测试 ### 前缀",
            }
        ],
        "notes": [],
    }

    with pytest.raises(ValueError, match="所有 1 个 patch 均失败"):
        _apply_repair_plan(chapter_markdown=content, repair_plan=plan)


# ── _apply_repair_plan: 输出项/标签不得冒充真实标题 ───────────────────────


@pytest.mark.unit
def test_apply_repair_plan_rejects_bullet_label_as_section_heading() -> None:
    """验证 repair 只能命中真实标题，不能把 bullet 标签当成 section heading。"""

    content = (
        "## 竞争格局\n\n"
        "### 详细情况\n\n"
        "#### 公司相对位置、竞争优势与主要压力来源\n\n"
        "- 公司最关键的竞争胜点与主要依据：旧内容A\n"
        "- 竞争地位的关键支撑依据（限已披露、仅写最关键 2-3 条）：旧内容B\n\n"
        "#### 最近一年竞争变化\n\n"
        "- 过去一年竞争格局的关键变化：其他内容\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "旧内容B",
                "target_section_heading": "竞争地位的关键支撑依据（限已披露、仅写最关键2-3条）",
                "replacement": "新内容B",
                "reason": "测试 bullet 标签回退",
            }
        ],
        "notes": [],
    }

    with pytest.raises(ValueError, match="所有 1 个 patch 均失败"):
        _apply_repair_plan(chapter_markdown=content, repair_plan=plan)


@pytest.mark.unit
def test_apply_repair_plan_rejects_output_item_label_as_section_heading() -> None:
    """验证输出项标签不能冒充真实标题。"""

    content = (
        "#### 第一节\n\n"
        "- 竞争地位的关键支撑依据（限已披露、仅写最关键 2-3 条）：\n"
        "- 重复文本\n\n"
        "#### 第二节\n\n"
        "- 重复文本\n"
    )
    plan = {
        "patches": [
            {
                "target_excerpt": "重复文本",
                "target_section_heading": "竞争地位的关键支撑依据（限已披露、仅写最关键2-3条）",
                "replacement": "已替换",
                "reason": "测试 scope 限制",
            }
        ],
        "notes": [],
    }

    with pytest.raises(ValueError, match="所有 1 个 patch 均失败"):
        _apply_repair_plan(chapter_markdown=content, repair_plan=plan)


# ---------------------------------------------------------------------------
# LLM 重试逻辑测试
# ---------------------------------------------------------------------------


def _make_app_result(
    content: str = "",
    errors: list | None = None,
) -> AppResult:
    """构建测试用 AppResult。

    Args:
        content: 回答文本。
        errors: 错误列表，None 表示无错误。

    Returns:
        AppResult 实例。
    """

    normalized_errors = [str(item.get("error") if isinstance(item, dict) else item) for item in (errors or [])]
    return AppResult(
        content=content,
        errors=normalized_errors,
        warnings=[],
    )


def _make_fake_prompt_agent(side_effects: list[AppResult]) -> _FakePromptAgent:
    """构建假 Prompt facade，按序列返回 AppResult。

    Args:
        side_effects: 每次 facade 调用的返回值序列。

    Returns:
        `_FakePromptAgent` 实例。
    """

    return _FakePromptAgent(side_effects)


@pytest.mark.unit
def test_run_agent_prompt_raw_retries_on_llm_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 _run_agent_prompt_raw 在首次 LLM 错误后成功重试。"""

    runner = _build_runner(tmp_path)
    fake_agent = _make_fake_prompt_agent([
        _make_app_result(errors=[{"error": "temporary failure"}]),
        _make_app_result(content="成功的输出"),
    ])
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.time.sleep", lambda _: None)

    result = runner._prompt_runner._run_agent_prompt_raw("测试 prompt")

    assert result == "成功的输出"


@pytest.mark.unit
def test_run_agent_prompt_raw_raises_after_retry_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 _run_agent_prompt_raw 重试耗尽后抛出 RuntimeError。"""

    runner = _build_runner(tmp_path)
    fake_agent = _make_fake_prompt_agent([
        _make_app_result(errors=[{"error": "fail-1"}]),
        _make_app_result(errors=[{"error": "fail-2"}]),
    ])
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="写作 Agent 执行失败"):
        runner._prompt_runner._run_agent_prompt_raw("测试 prompt")


@pytest.mark.unit
def test_scene_prompt_runner_high_level_markdown_wrappers_extract_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证高层 prompt wrapper 会提取 Markdown 正文或解析 facet。"""

    runner = _build_runner(tmp_path)
    markdown = "```markdown\n## 标题\n\n正文\n```"
    facet_json = json.dumps(
        {
            "business_model_tags": ["平台互联网"],
            "constraint_tags": ["监管敏感"],
            "judgement_notes": "note",
        },
        ensure_ascii=False,
    )

    monkeypatch.setattr(runner._prompt_runner, "_run_agent_prompt_raw", lambda _prompt: markdown)
    monkeypatch.setattr(runner._prompt_runner, "_run_decision_agent_prompt_raw", lambda _prompt: markdown)
    monkeypatch.setattr(runner._prompt_runner, "_run_overview_agent_prompt_raw", lambda _prompt: markdown)
    monkeypatch.setattr(runner._prompt_runner, "_run_fix_agent_prompt_raw", lambda _prompt: markdown)
    monkeypatch.setattr(runner._prompt_runner, "_run_regenerate_agent_prompt_raw", lambda _prompt: markdown)
    monkeypatch.setattr(runner._prompt_runner, "_run_infer_agent_prompt_raw", lambda _prompt: facet_json)
    monkeypatch.setattr(
        runner._prompt_runner._preparer,
        "get_company_facet_catalog",
        lambda: {
            "business_model_candidates": ["平台互联网"],
            "constraint_candidates": ["监管敏感"],
        },
    )

    assert runner._prompt_runner.run_write_prompt("prompt") == "## 标题\n\n正文"
    assert runner._prompt_runner.run_decision_prompt("prompt") == "## 标题\n\n正文"
    assert runner._prompt_runner.run_overview_prompt("prompt") == "## 标题\n\n正文"
    assert runner._prompt_runner.run_fix_prompt("prompt") == "## 标题\n\n正文"
    assert runner._prompt_runner.run_regenerate_prompt("prompt") == "## 标题\n\n正文"
    infer_result = runner._prompt_runner.run_infer_prompt("prompt")
    assert infer_result.primary_facets == ["平台互联网"]
    assert infer_result.cross_cutting_facets == ["监管敏感"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "method_name",
    [
        "_run_infer_agent_prompt_raw",
        "_run_decision_agent_prompt_raw",
        "_run_fix_agent_prompt_raw",
        "_run_regenerate_agent_prompt_raw",
    ],
)
def test_scene_prompt_runner_raw_prompt_methods_retry_on_llm_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    method_name: str,
) -> None:
    """验证其余 raw prompt 方法在首次 LLM 错误后会按统一策略重试。"""

    runner = _build_runner(tmp_path)
    fake_agent = _make_fake_prompt_agent([
        _make_app_result(errors=[{"error": "temporary failure"}]),
        _make_app_result(content="成功的输出"),
    ])

    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.scene_executor.time.sleep", lambda _: None)

    result = getattr(runner._prompt_runner, method_name)("测试 prompt")

    assert result == "成功的输出"


@pytest.mark.unit
def test_chapter_audit_coordinator_audit_chapter_retries_on_llm_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证章节审计协调器的 _audit_chapter 在首次 LLM 错误后成功重试。"""

    runner = _build_runner(tmp_path)
    audit_json = json.dumps({
        "pass": True,
        "violations": [],
        "notes": [],
    })
    fake_agent = _make_fake_prompt_agent([
        _make_app_result(errors=[{"error": "audit temporary failure"}]),
        _make_app_result(content=audit_json),
    ])
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.chapter_audit_coordinator.time.sleep", lambda _: None)
    monkeypatch.setattr(runner._prompter, "_render_task_prompt", lambda **_kw: "fake audit prompt")

    result = runner._chapter_audit_coordinator._audit_chapter("## Test\n\n内容", company_name="TestCo")

    assert result.passed is True


@pytest.mark.unit
def test_chapter_audit_coordinator_audit_chapter_raises_after_retry_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证章节审计协调器的 _audit_chapter 重试耗尽后抛出 RuntimeError。"""

    runner = _build_runner(tmp_path)
    fake_agent = _make_fake_prompt_agent([
        _make_app_result(errors=[{"error": "audit-fail-1"}]),
        _make_app_result(errors=[{"error": "audit-fail-2"}]),
    ])
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.chapter_audit_coordinator.time.sleep", lambda _: None)
    monkeypatch.setattr(runner._prompter, "_render_task_prompt", lambda **_kw: "fake audit prompt")

    with pytest.raises(RuntimeError, match="审计 Agent 执行失败"):
        runner._chapter_audit_coordinator._audit_chapter("## Test\n\n内容", company_name="TestCo")


@pytest.mark.unit
def test_chapter_audit_coordinator_audit_chapter_passes_audit_mode_and_repair_contract_to_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 repair 后章节审计协调器会把局部复审模式与修复合同传给 task prompt。"""

    runner = _build_runner(tmp_path)
    captured: dict[str, object] = {}
    audit_json = json.dumps({"pass": True, "violations": [], "notes": []}, ensure_ascii=False)

    def fake_render(*, prompt_name: str, prompt_inputs: dict[str, object]) -> str:
        captured["prompt_name"] = prompt_name
        captured["prompt_inputs"] = prompt_inputs
        return "fake audit prompt"

    fake_agent = _make_fake_prompt_agent([_make_app_result(content=audit_json)])
    monkeypatch.setattr(runner._prompter, "_render_task_prompt", fake_render)
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)

    result = runner._chapter_audit_coordinator._audit_chapter(
        "## Test\n\n内容",
        company_name="TestCo",
        audit_mode="修复后局部复审",
        repair_contract={"retry_scope": "targeted_evidence_patch", "offending_claim_spans": []},
    )

    assert result.passed is True
    assert captured["prompt_name"] == "audit_facts_tone_json"
    prompt_inputs = captured["prompt_inputs"]
    assert isinstance(prompt_inputs, dict)
    assert prompt_inputs["audit_mode"] == "修复后局部复审"
    assert prompt_inputs["audit_scope_rules"] == build_audit_scope_rules_payload()
    assert prompt_inputs["repair_contract"] == {"retry_scope": "targeted_evidence_patch", "offending_claim_spans": []}


@pytest.mark.unit
def test_run_confirm_prompt_retries_on_llm_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 _run_confirm_prompt 在首次 LLM 错误后成功重试。"""

    runner = _build_runner(tmp_path)
    confirm_json = json.dumps(
        {
            "results": [
                {
                    "violation_id": "evidence_1",
                    "rule": "E2",
                    "excerpt": "收入 100",
                    "status": "supported",
                    "reason": "原始证据可直接支持。",
                    "rewrite_hint": "",
                }
            ],
            "notes": [],
        },
        ensure_ascii=False,
    )
    fake_agent = _make_fake_prompt_agent([
        _make_app_result(errors=[{"error": "confirm temporary failure"}]),
        _make_app_result(content=confirm_json),
    ])
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.time.sleep", lambda _: None)

    result = runner._prompt_runner.run_confirm_prompt("测试 confirm prompt")

    assert len(result.entries) == 1
    assert result.entries[0].status == "supported"


@pytest.mark.unit
def test_run_confirm_prompt_does_not_retry_when_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 confirm prompt 遇到取消时直接透传，不进入重试。"""

    runner = _build_runner(tmp_path)
    fake_agent = _FakeCancelledPromptAgent(cancel_reason="测试取消")
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.time.sleep", lambda _: None)

    with pytest.raises(CancelledError, match="写作 Agent 执行被取消: 测试取消"):
        runner._prompt_runner.run_confirm_prompt("测试 confirm prompt")

    assert len(fake_agent.calls) == 1


@pytest.mark.unit
def test_run_confirm_prompt_raises_confirm_output_error_after_parse_retry_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 _run_confirm_prompt JSON 解析重试耗尽后抛出 ConfirmOutputError。"""

    runner = _build_runner(tmp_path)
    fake_agent = _make_fake_prompt_agent([
        _make_app_result(content="非法 JSON 第一次"),
        _make_app_result(content="非法 JSON 第二次"),
    ])
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.time.sleep", lambda _: None)

    with pytest.raises(ConfirmOutputError):
        runner._prompt_runner.run_confirm_prompt("测试 confirm prompt")


@pytest.mark.unit
def test_parse_evidence_confirmation_result_parses_four_statuses() -> None:
    """验证证据复核结果可解析四种确认状态。"""

    raw = json.dumps(
        {
            "results": [
                {
                    "violation_id": "evidence_1",
                    "rule": "E1",
                    "excerpt": "A",
                    "status": "confirmed_missing",
                    "reason": "缺证",
                    "rewrite_hint": "删除",
                },
                {
                    "violation_id": "evidence_2",
                    "rule": "E2",
                    "excerpt": "B",
                    "status": "supported_but_anchor_too_coarse",
                    "reason": "证据存在但锚点过粗",
                    "rewrite_hint": "细化证据行",
                    "anchor_fix": {
                        "kind": "same_filing_statement",
                        "action": "refine_existing",
                        "keep_existing_evidence": True,
                        "statement_type": "income",
                        "period": "FY2025",
                        "rows": ["Services", "Cost of sales:Services"],
                    },
                },
                {
                    "violation_id": "evidence_3",
                    "rule": "E2",
                    "excerpt": "C",
                    "status": "supported_elsewhere_in_same_filing",
                    "reason": "同一 filing 其他 section 已支持",
                    "rewrite_hint": "补锚点",
                },
                {
                    "violation_id": "evidence_4",
                    "rule": "E2",
                    "excerpt": "D",
                    "status": "supported",
                    "reason": "证据充分",
                    "rewrite_hint": "",
                },
            ],
            "notes": ["note"],
        },
        ensure_ascii=False,
    )

    result = _parse_evidence_confirmation_result(raw)

    assert [entry.status for entry in result.entries] == [
        "confirmed_missing",
        "supported_but_anchor_too_coarse",
        "supported_elsewhere_in_same_filing",
        "supported",
    ]
    assert result.entries[1].anchor_fix == EvidenceAnchorFix(
        kind="same_filing_statement",
        action="refine_existing",
        keep_existing_evidence=True,
        statement_type="income",
        period="FY2025",
        rows=["Services", "Cost of sales:Services"],
    )
    assert result.notes == ["note"]


@pytest.mark.unit
def test_audit_parse_evidence_confirmation_result_normalizes_status_values() -> None:
    """验证 audit 模块内部的 confirm 结果解析会正确导入并归一化状态值。"""

    raw = json.dumps(
        {
            "results": [
                {
                    "violation_id": "evidence_1",
                    "rule": "E2",
                    "excerpt": "A",
                    "status": "supported_elsewhere_in_same_filing",
                    "reason": "同一 filing 其他 section 已支持",
                    "rewrite_hint": "补锚点",
                }
            ],
            "notes": [],
        },
        ensure_ascii=False,
    )

    result = _parse_evidence_confirmation_result(raw)

    assert len(result.entries) == 1
    assert result.entries[0].status == "supported_elsewhere_in_same_filing"


@pytest.mark.unit
def test_parse_evidence_confirmation_result_treats_empty_anchor_fix_as_absent() -> None:
    """验证 confirm 返回空 anchor_fix 时按未提供处理，而不是整轮报错。"""

    raw = json.dumps(
        {
            "results": [
                {
                    "violation_id": "evidence_1",
                    "rule": "E2",
                    "excerpt": "A",
                    "status": "supported",
                    "reason": "派生指标由当前证据直接支持",
                    "rewrite_hint": "",
                    "anchor_fix": {},
                }
            ],
            "notes": [],
        },
        ensure_ascii=False,
    )

    result = _parse_evidence_confirmation_result(raw)

    assert len(result.entries) == 1
    assert result.entries[0].anchor_fix is None


@pytest.mark.unit
def test_parse_audit_decision_demotes_low_priority_rules_and_recomputes_pass() -> None:
    """验证 C2 / S4 / S5 / S6 会被降为低优先级且不单独阻断通过。"""

    raw = json.dumps(
        {
            "pass": False,
            "class": "evidence_insufficient",
            "violations": [
                {"rule": "C2", "severity": "high", "reason": "越界"},
                {"rule": "S4", "severity": "medium", "reason": "分析句"},
                {"rule": "S5", "severity": "high", "reason": "前瞻句"},
                {"rule": "S6", "severity": "high", "reason": "前瞻原文"},
            ],
            "notes": [],
        },
        ensure_ascii=False,
    )

    decision = _parse_audit_decision(raw)

    assert decision.passed is True
    assert decision.category == AuditCategory.OK
    assert all(item.severity == "low" for item in decision.violations)


@pytest.mark.unit
def test_recompute_audit_result_still_blocks_remaining_evidence_rules() -> None:
    """验证 E3 等保留证据规则仍会阻断通过。"""

    violations = _normalize_audit_violations(
        [
            {"rule": "E3", "severity": "high", "reason": "来源不可追溯"},
            {"rule": "S4", "severity": "high", "reason": "分析句"},
        ]
    )

    passed, category = _recompute_audit_result(violations=violations)

    assert passed is False
    assert category == AuditCategory.EVIDENCE_INSUFFICIENT


@pytest.mark.unit
def test_parse_audit_decision_routes_e3_to_regenerate() -> None:
    """验证 E3 会把 evidence_insufficient 收口为整章 regenerate。"""

    raw = json.dumps(
        {
            "pass": False,
            "class": "evidence_insufficient",
            "violations": [
                {
                    "rule": "E3",
                    "severity": "high",
                    "excerpt": "证据与出处\n- SEC EDGAR | Form 20-F | Filed 2025-06-13 | Accession 0001144204-25-026024 | Item 1: Business Description",
                    "reason": "evidence line 指向的 accession 不存在，无法定位到当前 ticker 的原始披露。",
                    "rewrite_hint": "删除该来源并整章重写，改用当前 ticker 下可稳定定位的 filing。",
                }
            ],
            "notes": [],
        },
        ensure_ascii=False,
    )

    decision = _parse_audit_decision(raw)

    assert decision.passed is False
    assert decision.category == AuditCategory.EVIDENCE_INSUFFICIENT
    assert decision.repair_contract.preferred_tool_action == "regenerate_chapter"
    assert decision.repair_contract.repair_strategy == "regenerate"
    assert decision.repair_contract.retry_scope == "chapter_regenerate"


@pytest.mark.unit
def test_recompute_audit_result_blocks_content_rules() -> None:
    """验证 C1 会阻断通过并归类为内容合规问题。"""

    violations = _normalize_audit_violations(
        [
            {"rule": "C1", "severity": "high", "reason": "大量占位符"},
            {"rule": "S4", "severity": "high", "reason": "分析句"},
        ]
    )

    passed, category = _recompute_audit_result(violations=violations)

    assert passed is False
    assert category == AuditCategory.CONTENT_VIOLATION


@pytest.mark.unit
def test_normalize_audit_violations_demotes_c2_to_low_priority() -> None:
    """验证 C2 会被归一为低优先级提示。"""

    violations = _normalize_audit_violations(
        [{"rule": "C2", "severity": "high", "reason": "章节越界"}]
    )

    assert violations[0].severity == "low"


@pytest.mark.unit
def test_merge_confirmed_evidence_results_removes_supported_and_demotes_anchor_issue() -> None:
    """验证证据复核会移除已支持的 E 违规，并将锚点过粗降级为 S7。"""

    decision = AuditDecision(
        passed=False,
        category=AuditCategory.EVIDENCE_INSUFFICIENT,
        violations=[
            Violation(rule=AuditRuleCode.E1, severity="high", excerpt="句子 A", reason="疑似缺证", rewrite_hint="删"),
            Violation(rule=AuditRuleCode.E2, severity="high", excerpt="句子 B", reason="疑似数字缺证", rewrite_hint="删"),
            Violation(rule=AuditRuleCode.S2, severity="medium", excerpt="句子 C", reason="摘抄", rewrite_hint="改"),
        ],
        notes=[],
        repair_contract=RepairContract(),
        raw="{}",
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="句子 A",
                status=EvidenceConfirmationStatus.SUPPORTED,
                reason="证据充分",
            ),
            EvidenceConfirmationEntry(
                violation_id="evidence_2",
                rule=AuditRuleCode.E2,
                excerpt="句子 B",
                status=EvidenceConfirmationStatus.SUPPORTED_BUT_ANCHOR_TOO_COARSE,
                reason="锚点过粗",
                rewrite_hint="仅细化证据行",
            ),
        ],
        notes=["confirmed"],
        raw="{}",
    )

    merged = _merge_confirmed_evidence_results(
        audit_decision=decision,
        confirmation_result=confirmation,
    )

    assert all(item.excerpt != "句子 A" for item in merged.violations)
    assert any(item.rule == AuditRuleCode.S7 and item.excerpt == "句子 B" for item in merged.violations)
    assert merged.passed is True
    assert merged.category == AuditCategory.OK
    assert "confirmed" in merged.notes


@pytest.mark.unit
def test_merge_confirmed_evidence_results_demotes_supported_elsewhere_in_same_filing() -> None:
    """验证同一 filing 内可救回的 claim 会降级为 S7，而不是继续保留为 E。"""

    decision = AuditDecision(
        passed=False,
        category=AuditCategory.EVIDENCE_INSUFFICIENT,
        violations=[
            Violation(rule=AuditRuleCode.E1, severity="high", excerpt="句子 A", reason="疑似缺证", rewrite_hint="删"),
        ],
        notes=[],
        repair_contract=RepairContract(),
        raw="{}",
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="句子 A",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="同一 filing 的其它 section 已支持",
                rewrite_hint="补充同一 filing 内正确锚点，不要删除正文信息。",
            )
        ],
        notes=[],
        raw="{}",
    )

    merged = _merge_confirmed_evidence_results(
        audit_decision=decision,
        confirmation_result=confirmation,
    )

    assert merged.passed is True
    assert merged.category == AuditCategory.OK
    assert len(merged.violations) == 1
    assert merged.violations[0].rule == AuditRuleCode.S7
    assert merged.violations[0].excerpt == "句子 A"
    assert "不要删除正文信息" in merged.violations[0].rewrite_hint


@pytest.mark.unit
def test_merge_confirmed_evidence_results_drops_violation_when_structured_anchor_fix_exists() -> None:
    """验证 confirm 已返回结构化 anchor_fix 时，不再把 E1/E2 降成 S7。"""

    decision = AuditDecision(
        passed=False,
        category=AuditCategory.EVIDENCE_INSUFFICIENT,
        violations=[
            Violation(rule=AuditRuleCode.E1, severity="high", excerpt="句子 A", reason="疑似缺证", rewrite_hint="删"),
        ],
        notes=[],
        repair_contract=RepairContract(),
        raw="{}",
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="句子 A",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="同一 filing 的其它 section 已支持",
                rewrite_hint="补充同一 filing 内正确锚点，不要删除正文信息。",
                anchor_fix=EvidenceAnchorFix(
                    kind="same_filing_section",
                    action="append",
                    keep_existing_evidence=True,
                    section_path="Part II - Item 6 | Employees",
                ),
            )
        ],
        notes=[],
        raw="{}",
    )

    merged = _merge_confirmed_evidence_results(
        audit_decision=decision,
        confirmation_result=confirmation,
    )

    assert merged.passed is True
    assert merged.category == AuditCategory.OK
    assert merged.violations == []


@pytest.mark.unit
def test_merge_confirmed_evidence_results_marks_confirmed_missing_as_delete_claim() -> None:
    """验证 confirmed_missing 会被收口成 delete_claim 处置语义。"""

    decision = AuditDecision(
        passed=False,
        category=AuditCategory.EVIDENCE_INSUFFICIENT,
        violations=[
            Violation(
                rule=AuditRuleCode.E1,
                severity="high",
                excerpt="地缘政治风险与出口管制可能突然中断订单或现金流",
                reason="疑似缺证",
                rewrite_hint="删除该断言",
            )
        ],
        notes=[],
        repair_contract=RepairContract(),
        raw="{}",
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="地缘政治风险与出口管制可能突然中断订单或现金流",
                status=EvidenceConfirmationStatus.CONFIRMED_MISSING,
                reason="证据与出处中没有提供地缘政治风险或出口管制的相关证据。",
                rewrite_hint="删除该断言。",
            )
        ],
        notes=[],
        raw="{}",
    )

    merged = _merge_confirmed_evidence_results(
        audit_decision=decision,
        confirmation_result=confirmation,
    )

    assert merged.violations[0].confirmation_status == EvidenceConfirmationStatus.CONFIRMED_MISSING.value
    assert merged.violations[0].resolution_mode == RepairResolutionMode.DELETE_CLAIM.value
    assert merged.repair_contract.remediation_actions[0].resolution_mode == (
        RepairResolutionMode.DELETE_CLAIM.value
    )
    assert merged.repair_contract.remediation_actions[0].target_kind_hint == "paragraph"


@pytest.mark.unit
def test_build_anchor_rewrite_evidence_lines_uses_full_line_or_part_locator() -> None:
    """验证 anchor rewrite 会从 confirm 结果中提取可落盘的新 evidence line。"""

    chapter_markdown = (
        "## 最近一年关键变化与当前阶段\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Financial Statement:cash_flow | Period:FY2025,FY2024 | Rows:Purchases of property and equipment,Net cash provided by operating activities\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="自由现金流从521亿美元降至436亿美元",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="自由现金流数字在同一 filing 的 Part II - Item 7 | Free Cash Flow 中明确列示。",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part II - Item 7 | Free Cash Flow",
            ),
            EvidenceConfirmationEntry(
                violation_id="evidence_2",
                rule=AuditRuleCode.E1,
                excerpt="员工数从2024年41,697人增至2025年43,267人",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="员工数字出现在同一份文件的Part II - Item 8 - Financial Information中，但当前证据条目未引用该部分。",
                rewrite_hint="证据条目应引用Part II - Item 8 - Financial Information，其中包含员工数量表格。",
            ),
        ]
    )

    rewritten = _build_anchor_rewrite_evidence_lines(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    assert any("Part II - Item 7 | Free Cash Flow" in line for line in rewritten)
    assert any("Part II - Item 8 - Financial Information" in line for line in rewritten)


@pytest.mark.unit
def test_rewrite_evidence_lines_for_confirmed_anchor_issues_prefers_structured_anchor_fix() -> None:
    """验证 evidence rewrite 会优先消费结构化 anchor_fix。"""

    chapter_markdown = (
        "## 商业模式的关键机制与约束\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 20-F | Filed 2026-02-11 | Accession 0001628280-26-011378 | Part II - Item 6 | Vulnerability due to certain concentrations\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E2,
                excerpt="公司拥有约10,000名客户支持员工",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="同一 filing 内其它标题已支持。",
                rewrite_hint="补锚点",
                anchor_fix=EvidenceAnchorFix(
                    kind="same_filing_section",
                    action="append",
                    keep_existing_evidence=True,
                    section_path="Part II - Item 6 - Directors, Senior Management and Employees | Innovation and R&D",
                ),
            )
        ]
    )

    rewritten = _rewrite_evidence_lines_for_confirmed_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    assert any("Part II - Item 6 - Directors, Senior Management and Employees | Innovation and R&D" in line for line in rewritten)


@pytest.mark.unit
def test_rewrite_evidence_lines_for_confirmed_anchor_issues_updates_statement_rows_and_periods() -> None:
    """验证证据锚点轻量修复会真正改写已有财务报表 evidence line。"""

    chapter_markdown = (
        "## 财务表现与资本配置\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | "
        "Financial Statement:income | Period:FY2025 | Rows:Net sales, Products, Services\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | "
        "Financial Statement:cash_flow | Period:FY2025 | Rows:Cash generated by operating activities\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E2,
                excerpt="服务业务毛利率达75.4%",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="利润表中已披露服务收入与服务成本，当前证据条目未细分服务成本。",
                rewrite_hint="在证据条目的Rows中添加“Services”成本行，或使用Financial Statement:income | Period:FY2025 | Rows:Services, Cost of sales:Services。",
            ),
            EvidenceConfirmationEntry(
                violation_id="evidence_2",
                rule=AuditRuleCode.E2,
                excerpt="三年平均经营活动现金流1134.26亿美元",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="多年数据存在于同一文件的现金流量表中，但当前证据条目仅引用FY2025。",
                rewrite_hint="在证据条目中扩展期间引用至多年，或使用Financial Statement:cash_flow | Period:FY2025,FY2024,FY2023 | Rows:Cash generated by operating activities。",
            ),
        ]
    )

    rewritten = _rewrite_evidence_lines_for_confirmed_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    normalized_lines = [line.replace(" ", "") for line in rewritten]
    assert any(
        "FinancialStatement:income|Period:FY2025|Rows:Netsales,Products,Services,Costofsales:Services"
        in line
        for line in normalized_lines
    )
    assert any(
        "FinancialStatement:cash_flow|Period:FY2025,FY2024,FY2023|Rows:Cashgeneratedbyoperatingactivities"
        in line
        for line in normalized_lines
    )


@pytest.mark.unit
def test_rewrite_evidence_lines_for_confirmed_anchor_issues_appends_explicit_full_line() -> None:
    """验证 confirm 给出完整 evidence line 时，会被追加到证据列表。"""

    chapter_markdown = (
        "## 经营表现与核心驱动\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | "
        "Financial Statement:cash_flow | Period:FY2025,FY2024 | Rows:Purchases of property and equipment,Net cash provided by operating activities\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="自由现金流从521亿美元降至436亿美元",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="自由现金流数字在同一 filing 的 Part II - Item 7 | Free Cash Flow 中明确列示。",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part II - Item 7 | Free Cash Flow",
            )
        ]
    )

    rewritten = _rewrite_evidence_lines_for_confirmed_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    assert any("Part II - Item 7 | Free Cash Flow" in line for line in rewritten)


@pytest.mark.unit
def test_maybe_rewrite_evidence_anchors_writes_back_updated_evidence_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 confirm 后的轻量 evidence rewrite 会回写章节正文并消除对应 S7。"""

    runner = _build_runner(tmp_path)
    task = ChapterTask(index=5, title="经营表现与核心驱动", skeleton="## 经营表现与核心驱动")
    current_content = (
        "## 经营表现与核心驱动\n\n"
        "### 结论要点\n\n"
        "- 自由现金流下降。\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | "
        "Financial Statement:cash_flow | Period:FY2025,FY2024 | Rows:Purchases of property and equipment,Net cash provided by operating activities\n"
    )
    suspected = AuditDecision(passed=True, category=AuditCategory.OK)
    audit = AuditDecision(
        passed=True,
        category=AuditCategory.OK,
        violations=[
            Violation(
                rule=AuditRuleCode.S7,
                severity="low",
                excerpt="自由现金流从521亿美元降至436亿美元",
                reason="自由现金流数字在同一 filing 的 Part II - Item 7 | Free Cash Flow 中明确列示。",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part II - Item 7 | Free Cash Flow",
            )
        ],
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="自由现金流从521亿美元降至436亿美元",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="自由现金流数字在同一 filing 的 Part II - Item 7 | Free Cash Flow 中明确列示。",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part II - Item 7 | Free Cash Flow",
            )
        ]
    )
    process_state = build_process_state_template()

    rewritten_content, _rewritten_suspected, rewritten_audit, _rewritten_confirmation = runner._chapter_audit_coordinator.maybe_rewrite_evidence_anchors(
        task=task,
        current_content=current_content,
        suspected_decision=suspected,
        audit_decision=audit,
        confirmation_result=confirmation,
        phase="initial",
        skeleton=task.skeleton,
        allowed_conditional_headings=set(),
        process_state=process_state,
    )

    assert "Part II - Item 7 | Free Cash Flow" in rewritten_content
    assert rewritten_audit.passed is True
    assert rewritten_audit.category == AuditCategory.OK
    assert rewritten_audit.violations == []
    assert process_state["latest_anchor_rewrite"] == {
        "phase": "initial",
        "attempted": True,
        "applied": True,
        "skip_reason": "",
        "failure_reason": "",
        "resolved_violations_count": 1,
    }
    artifact_path = runner._store._chapter_phase_artifact_path(
        index=task.index,
        title=task.title,
        artifact_name="initial_write",
        extension=".md",
    )
    assert artifact_path.read_text(encoding="utf-8") == rewritten_content


@pytest.mark.unit
def test_maybe_rewrite_evidence_anchors_does_not_trigger_reaudit_when_all_entries_supported_like(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证全部证据已被 supported / supported* 覆盖时不会再次触发审计链路，且 supported_* 会在 rewrite 成功后消失。"""

    runner = _build_runner(tmp_path)
    task = ChapterTask(index=5, title="经营表现与核心驱动", skeleton="## 经营表现与核心驱动")
    current_content = (
        "## 经营表现与核心驱动\n\n"
        "### 结论要点\n\n"
        "- 自由现金流下降。\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | "
        "Financial Statement:cash_flow | Period:FY2025,FY2024 | Rows:Purchases of property and equipment,Net cash provided by operating activities\n"
    )
    suspected = AuditDecision(passed=True, category=AuditCategory.OK)
    audit = AuditDecision(
        passed=True,
        category=AuditCategory.OK,
        violations=[
            Violation(
                rule=AuditRuleCode.S7,
                severity="low",
                excerpt="自由现金流从521亿美元降至436亿美元",
                reason="自由现金流数字在同一 filing 的 Part II - Item 7 | Free Cash Flow 中明确列示。",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part II - Item 7 | Free Cash Flow",
            )
        ],
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="自由现金流从521亿美元降至436亿美元",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="自由现金流数字在同一 filing 的 Part II - Item 7 | Free Cash Flow 中明确列示。",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part II - Item 7 | Free Cash Flow",
            )
        ]
    )

    def _raise_if_called(**_kwargs: object) -> tuple[AuditDecision, AuditDecision, EvidenceConfirmationResult]:
        """若被调用说明轻量修复错误触发了重审。"""

        raise AssertionError("当 confirm 结果全部为 supported / supported* 时，不应再次触发 audit_and_confirm_chapter")

    monkeypatch.setattr(runner._chapter_audit_coordinator, "audit_and_confirm_chapter", _raise_if_called)

    rewritten_content, rewritten_suspected, rewritten_audit, rewritten_confirmation = runner._chapter_audit_coordinator.maybe_rewrite_evidence_anchors(
        task=task,
        current_content=current_content,
        suspected_decision=suspected,
        audit_decision=audit,
        confirmation_result=confirmation,
        phase="repair_1",
        skeleton=task.skeleton,
        allowed_conditional_headings=set(),
    )

    assert "Part II - Item 7 | Free Cash Flow" in rewritten_content
    assert rewritten_suspected is suspected
    assert rewritten_audit.passed is True
    assert rewritten_audit.category == AuditCategory.OK
    assert rewritten_audit.violations == []
    assert rewritten_confirmation is confirmation


@pytest.mark.unit
def test_maybe_rewrite_evidence_anchors_still_rewrites_when_confirmed_missing_coexists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证存在 confirmed_missing 时，rewrite 成功后只保留 confirmed_missing 对应问题。"""

    runner = _build_runner(tmp_path)
    task = ChapterTask(index=5, title="经营表现与核心驱动", skeleton="## 经营表现与核心驱动")
    current_content = (
        "## 经营表现与核心驱动\n\n"
        "### 结论要点\n\n"
        "- 自由现金流下降。\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | "
        "Financial Statement:cash_flow | Period:FY2025,FY2024 | Rows:Purchases of property and equipment,Net cash provided by operating activities\n"
    )
    suspected = AuditDecision(passed=True, category=AuditCategory.OK)
    audit = AuditDecision(
        passed=False,
        category=AuditCategory.EVIDENCE_INSUFFICIENT,
        violations=[
            Violation(
                rule=AuditRuleCode.S7,
                severity="low",
                excerpt="自由现金流从521亿美元降至436亿美元",
                reason="自由现金流数字在同一 filing 的 Part II - Item 7 | Free Cash Flow 中明确列示。",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part II - Item 7 | Free Cash Flow",
            ),
            Violation(
                rule=AuditRuleCode.E2,
                severity="high",
                excerpt="净债务减少9.6亿欧元",
                reason="当前证据未直接支持净债务变化值。",
                rewrite_hint="删除该具体数字。",
            ),
        ],
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="自由现金流从521亿美元降至436亿美元",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="自由现金流数字在同一 filing 的 Part II - Item 7 | Free Cash Flow 中明确列示。",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part II - Item 7 | Free Cash Flow",
            ),
            EvidenceConfirmationEntry(
                violation_id="evidence_2",
                rule=AuditRuleCode.E2,
                excerpt="净债务减少9.6亿欧元",
                status=EvidenceConfirmationStatus.CONFIRMED_MISSING,
                reason="当前证据未直接支持净债务变化值。",
                rewrite_hint="删除该具体数字。",
            ),
        ]
    )
    def _raise_if_called(**_kwargs: object) -> tuple[AuditDecision, AuditDecision, EvidenceConfirmationResult]:
        """若被调用说明轻量修复错误触发了重审。"""

        raise AssertionError("证据锚点轻量修复后不应再次触发 audit_and_confirm_chapter")

    monkeypatch.setattr(runner._chapter_audit_coordinator, "audit_and_confirm_chapter", _raise_if_called)

    rewritten_content, rewritten_suspected, rewritten_audit, rewritten_confirmation = runner._chapter_audit_coordinator.maybe_rewrite_evidence_anchors(
        task=task,
        current_content=current_content,
        suspected_decision=suspected,
        audit_decision=audit,
        confirmation_result=confirmation,
        phase="repair_1",
        skeleton=task.skeleton,
        allowed_conditional_headings=set(),
    )

    assert "Part II - Item 7 | Free Cash Flow" in rewritten_content
    assert rewritten_suspected is suspected
    assert rewritten_audit.passed is False
    assert rewritten_audit.category == AuditCategory.EVIDENCE_INSUFFICIENT
    assert rewritten_audit.violations == [
        Violation(
            rule=AuditRuleCode.E2,
            severity="high",
            excerpt="净债务减少9.6亿欧元",
            reason="当前证据未直接支持净债务变化值。",
            rewrite_hint="删除该具体数字。",
        )
    ]
    assert rewritten_confirmation is confirmation


@pytest.mark.unit
def test_maybe_rewrite_evidence_anchors_rolls_back_when_post_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证证据锚点轻量修复后验校验失败时会回退原文。"""

    runner = _build_runner(tmp_path)
    task = ChapterTask(index=5, title="经营表现与核心驱动", skeleton="## 经营表现与核心驱动")
    current_content = (
        "## 经营表现与核心驱动\n\n"
        "### 结论要点\n\n"
        "- 自由现金流下降。\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | "
        "Financial Statement:cash_flow | Period:FY2025,FY2024 | Rows:Purchases of property and equipment,Net cash provided by operating activities\n"
    )
    suspected = AuditDecision(passed=True, category=AuditCategory.OK)
    audit = AuditDecision(
        passed=True,
        category=AuditCategory.OK,
        violations=[
            Violation(
                rule=AuditRuleCode.S7,
                severity="low",
                excerpt="自由现金流从521亿美元降至436亿美元",
                reason="自由现金流数字在同一 filing 的 Part II - Item 7 | Free Cash Flow 中明确列示。",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part II - Item 7 | Free Cash Flow",
            )
        ],
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="自由现金流从521亿美元降至436亿美元",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="自由现金流数字在同一 filing 的 Part II - Item 7 | Free Cash Flow 中明确列示。",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part II - Item 7 | Free Cash Flow",
            )
        ]
    )

    failing_decision = AuditDecision(
        passed=False,
        category=AuditCategory.CONTENT_VIOLATION,
        violations=[Violation(rule=AuditRuleCode.P1, severity="high", excerpt="", reason="结构不匹配")],
        notes=["结构不匹配"],
        repair_contract=RepairContract(repair_strategy="regenerate"),
        raw='{"pass": false}',
    )
    monkeypatch.setattr(
        "dayu.services.internal.write_pipeline.audit_evidence_rewriter._run_programmatic_audits",
        lambda *_args, **_kwargs: failing_decision,
    )
    process_state = build_process_state_template()

    rewritten_content, rewritten_suspected, rewritten_audit, rewritten_confirmation = runner._chapter_audit_coordinator.maybe_rewrite_evidence_anchors(
        task=task,
        current_content=current_content,
        suspected_decision=suspected,
        audit_decision=audit,
        confirmation_result=confirmation,
        phase="repair_1",
        skeleton=task.skeleton,
        allowed_conditional_headings=set(),
        process_state=process_state,
    )

    assert rewritten_content == current_content
    assert rewritten_suspected is suspected
    assert rewritten_audit is audit
    assert rewritten_confirmation is confirmation
    assert process_state["latest_anchor_rewrite"] == {
        "phase": "repair_1",
        "attempted": True,
        "applied": False,
        "skip_reason": "",
        "failure_reason": "程序审计失败: content_violation",
        "resolved_violations_count": 0,
    }


@pytest.mark.unit
def test_persist_phase_confirm_parse_error_artifacts(tmp_path: Path) -> None:
    """验证 confirm 非法 JSON 时会落盘原始输出与解析错误。"""

    runner = _build_runner(tmp_path)
    task = ChapterTask(index=5, title="最近一年关键变化与当前阶段", skeleton="## x")

    runner._store._persist_phase_confirm_raw_artifact(
        task=task,
        phase="initial",
        raw_text="不是合法 JSON",
    )
    runner._store._persist_phase_confirm_parse_error_artifact(
        task=task,
        phase="initial",
        parse_error="证据复核输出不是合法 JSON",
    )

    raw_path = runner._store._chapter_phase_artifact_path(
        index=task.index,
        title=task.title,
        artifact_name="initial_confirm_raw",
        extension="txt",
    )
    error_path = runner._store._chapter_phase_artifact_path(
        index=task.index,
        title=task.title,
        artifact_name="initial_confirm_parse_error",
        extension="json",
    )

    assert raw_path.read_text(encoding="utf-8") == "不是合法 JSON"
    error_payload = json.loads(error_path.read_text(encoding="utf-8"))
    assert error_payload["parse_error"] == "证据复核输出不是合法 JSON"


@pytest.mark.unit
def test_collect_confirmable_evidence_violations_only_keeps_e1_e2() -> None:
    """验证只有 E1/E2 会进入证据复核环节。"""

    collected = _collect_confirmable_evidence_violations(
        [
            Violation(rule=AuditRuleCode.E1, excerpt="A"),
            Violation(rule=AuditRuleCode.E2, excerpt="B"),
            Violation(rule=AuditRuleCode.E3, excerpt="C"),
            Violation(rule=AuditRuleCode.S2, excerpt="D"),
        ]
    )

    assert [item["rule"] for item in collected] == ["E1", "E2"]
    assert [item["violation_id"] for item in collected] == ["evidence_1", "evidence_2"]


@pytest.mark.unit
def test_audit_module_programmatic_audits_and_parse_paths_cover_true_source() -> None:
    """验证 audit 真源模块的程序审计与 JSON 解析错误路径。"""

    malformed = audit_rules_module._parse_audit_decision("```json\n{bad json}\n```")
    assert malformed.passed is False
    assert malformed.category == AuditCategory.STYLE_VIOLATION
    assert malformed.violations[0].rule == AuditRuleCode.S1

    valid = audit_rules_module._parse_audit_decision(
        json.dumps(
            {
                "class": "style_violation",
                "violations": [
                    {"rule": "E1", "severity": "high", "excerpt": "句子 A", "reason": "缺证"}
                ],
                "notes": ["需要修复"],
            },
            ensure_ascii=False,
        )
    )
    assert valid.passed is False
    assert valid.category == AuditCategory.EVIDENCE_INSUFFICIENT
    assert valid.notes == ["需要修复"]

    too_short = audit_rules_module._run_programmatic_audits("## 标题\n短", skeleton="## 标题")
    assert too_short is not None
    assert too_short.violations[0].rule == AuditRuleCode.P2

    missing_evidence = audit_rules_module._run_programmatic_audits(
        "## 标题\n\n正文内容已经足够长，但这里没有来源小节。",
        skeleton="## 标题",
    )
    assert missing_evidence is not None
    assert missing_evidence.violations[0].rule == AuditRuleCode.P3

    assert audit_rules_module._run_programmatic_audits(
        "## 标题\n\n正文内容已经足够长。\n\n### 证据与出处\n- 来源 A",
        skeleton="## 标题",
    ) is None


@pytest.mark.unit
def test_audit_module_confirmation_helpers_cover_anchor_fix_parse_and_cleanup() -> None:
    """验证 audit 真源模块会解析 anchor_fix 并清理已解决的 supported follow-up。"""

    assert audit_rules_module._collect_confirmable_evidence_violations(
        [
            Violation(rule=AuditRuleCode.E1, severity="high", excerpt="A", reason="缺证"),
            Violation(rule=AuditRuleCode.S2, severity="low", excerpt="B", reason="样式"),
        ]
    ) == [
        {
            "violation_id": "evidence_1",
            "rule": "E1",
            "severity": "high",
            "excerpt": "A",
            "reason": "缺证",
            "rewrite_hint": "",
        }
    ]
    assert audit_rules_module._parse_evidence_anchor_fix(None) is None
    assert audit_rules_module._parse_evidence_anchor_fix({}) is None
    with pytest.raises(ValueError, match="anchor_fix 必须为对象"):
        audit_rules_module._parse_evidence_anchor_fix("bad")
    with pytest.raises(ValueError, match="kind 不受支持"):
        audit_rules_module._parse_evidence_anchor_fix({"kind": "bad", "action": "append"})
    with pytest.raises(ValueError, match="action 不受支持"):
        audit_rules_module._parse_evidence_anchor_fix({"kind": "same_filing_section", "action": "bad"})
    with pytest.raises(ValueError, match="rows 必须为数组"):
        audit_rules_module._parse_evidence_anchor_fix(
            {"kind": "same_filing_section", "action": "append", "rows": "bad"}
        )

    confirmation = _parse_evidence_confirmation_result(
        json.dumps(
            {
                "results": [
                    {
                        "violation_id": "evidence_1",
                        "rule": "E1",
                        "excerpt": "句子 A",
                        "status": "supported_elsewhere_in_same_filing",
                        "reason": "同一 filing 其它 section 已支持",
                        "rewrite_hint": "补充更精确锚点",
                        "anchor_fix": {
                            "kind": "same_filing_section",
                            "action": "append",
                            "section_path": "Part II | Employees",
                            "rows": ["- 来源 A", "", "- 来源 B"],
                        },
                    }
                ],
                "notes": ["confirmed"],
            },
            ensure_ascii=False,
        )
    )
    assert confirmation.entries[0].anchor_fix is not None
    assert confirmation.entries[0].anchor_fix.rows == ["- 来源 A", "- 来源 B"]
    assert confirmation.entries[0].rule == AuditRuleCode.E1

    followup = audit_rules_module._build_supported_anchor_followup_violation(confirmation.entries[0])
    assert followup.rule == AuditRuleCode.S7
    assert audit_rules_module._is_supported_anchor_followup_violation_for_entry(
        violation=followup,
        entry=confirmation.entries[0],
    ) is True

    pruned = audit_rules_module._drop_resolved_supported_anchor_violations(
        audit_decision=AuditDecision(
            passed=False,
            category=AuditCategory.STYLE_VIOLATION,
            violations=[followup, Violation(rule=AuditRuleCode.S2, excerpt="其它", reason="样式", rewrite_hint="修")],
            notes=["confirmed"],
            repair_contract=RepairContract(),
            raw="{}",
        ),
        confirmation_result=confirmation,
        resolved_violation_ids={"evidence_1"},
    )
    assert pruned.violations == [Violation(rule=AuditRuleCode.S2, excerpt="其它", reason="样式", rewrite_hint="修")]


@pytest.mark.unit
def test_audit_module_merge_rebuild_and_confirm_logs_cover_true_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 audit 真源模块的 merge/rebuild 逻辑与 confirm 日志分支。"""

    decision = AuditDecision(
        passed=False,
        category=AuditCategory.EVIDENCE_INSUFFICIENT,
        violations=[
            Violation(rule=AuditRuleCode.E1, severity="high", excerpt="句子 A", reason="疑似缺证", rewrite_hint="删"),
            Violation(rule=AuditRuleCode.S2, severity="low", excerpt="句子 B", reason="样式", rewrite_hint="修"),
        ],
        notes=["original"],
        repair_contract=RepairContract(),
        raw="{}",
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="句子 A",
                status=EvidenceConfirmationStatus.CONFIRMED_MISSING,
                reason="证据中不存在该说法",
                rewrite_hint="删除该断言",
            )
        ],
        notes=["confirmed"],
        raw="{}",
    )

    merged = audit_rules_module._merge_confirmed_evidence_results(
        audit_decision=decision,
        confirmation_result=confirmation,
    )
    assert merged.violations[0].reason == "证据中不存在该说法"
    assert merged.repair_contract.remediation_actions[0].resolution_mode == (
        RepairResolutionMode.DELETE_CLAIM.value
    )
    assert "confirmed" in merged.notes

    rebuilt = audit_rules_module._rebuild_audit_decision_with_confirmation(
        audit_decision=decision,
        violations=[],
        confirmation_result=None,
    )
    assert rebuilt.passed is True
    assert rebuilt.category == AuditCategory.OK
    assert "confirmation" not in json.loads(rebuilt.raw)


@pytest.mark.unit
def test_audit_rule_code_scope_payload_and_normalize_helper_use_single_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证规则范围摘要与规则码规范化都回到统一真源。"""

    assert build_audit_scope_rules_payload() == {
        "evidence_rules": ["E1", "E2", "E3"],
        "content_rules": ["C1", "C2"],
        "style_rules": ["S1", "S2", "S3", "S4", "S5", "S6", "S7"],
    }
    assert normalize_audit_rule_code(" e2 ") == AuditRuleCode.E2
    assert normalize_audit_rule_code("bad-rule") == AuditRuleCode.UNKNOWN

    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="句子 A",
                status=EvidenceConfirmationStatus.CONFIRMED_MISSING,
                reason="证据中不存在该说法",
            )
        ],
        notes=["confirmed"],
        raw="{}",
    )

    info_messages: list[str] = []
    warn_messages: list[str] = []

    def _fake_info(message: str, *, module: str) -> None:
        info_messages.append(f"{module}:{message}")

    def _fake_warn(message: str, *, module: str) -> None:
        warn_messages.append(f"{module}:{message}")

    monkeypatch.setattr(Log, "info", _fake_info)
    monkeypatch.setattr(Log, "warn", _fake_warn)
    audit_rules_module._log_chapter_confirm_start(chapter_title="第一章", phase="initial", count=2)
    audit_rules_module._log_chapter_confirm_result(
        chapter_title="第一章",
        phase="initial",
        result=confirmation,
    )
    audit_rules_module._log_chapter_confirm_result(
        chapter_title="第一章",
        phase="repair_1",
        result=EvidenceConfirmationResult(entries=[], notes=[], raw="{}"),
    )

    assert any("开始证据复核:" in message for message in info_messages)
    assert any("完成证据复核" in message for message in info_messages)
    assert warn_messages == []


@pytest.mark.unit
def test_normalize_evidence_location_segment_strips_generated_summary_parenthetical() -> None:
    """验证 evidence 定位段会去掉模型生成的摘要型括号说明。"""

    location = "Income Statement (Revenue, Advertising - Family of Apps, Other revenue - Family of Apps)"

    assert _normalize_evidence_location_segment(location) == "Income Statement"


@pytest.mark.unit
def test_normalize_evidence_location_segment_keeps_year_parenthetical() -> None:
    """验证年份型括号标题会被保留。"""

    location = "Consolidated Statements of Operations (2025, 2024, 2023)"

    assert _normalize_evidence_location_segment(location) == location


@pytest.mark.unit
def test_normalize_chapter_markdown_for_audit_rewrites_only_evidence_lines() -> None:
    """验证送审前预处理只清理证据行，不改正文内容。"""

    content = (
        "## 公司做的是什么生意\n\n"
        "### 结论要点\n\n"
        "- 产品收入占比 74%。\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2026-01-01 | Accession 0000000000-26-000001 | "
        "Income Statement (Revenue, Products, Services)\n"
    )

    normalized = _normalize_chapter_markdown_for_audit(content)

    assert "- 产品收入占比 74%。" in normalized
    assert "Income Statement (Revenue, Products, Services)" not in normalized
    assert normalized.endswith("Income Statement")


@pytest.mark.unit
def test_normalize_chapter_markdown_for_audit_coerces_bare_evidence_lines_and_strips_fences() -> None:
    """验证送审前预处理会修正裸 evidence 行并删除 stray fence。"""

    content = (
        "## 是否值得继续深研与待验证问题\n\n"
        "### 结论要点\n\n"
        "继续研究\n\n"
        "### 证据与出处\n\n"
        "SEC EDGAR | Form 10-K | Filed 2026-01-01 | Accession 0000000000-26-000001 | Part II - Item 8\n"
        "```\n"
        "Reuters | 标题 | 访问日期 2026-04-01 | URL:https://example.com/a\n"
    )

    normalized = _normalize_chapter_markdown_for_audit(content)

    assert "```" not in normalized
    assert "\n- SEC EDGAR | Form 10-K | Filed 2026-01-01 | Accession 0000000000-26-000001 | Part II - Item 8" in normalized
    assert "\n- Reuters | 标题 | 访问日期 2026-04-01 | URL:https://example.com/a" in normalized


@pytest.mark.unit
def test_run_repair_prompt_retries_on_llm_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 _run_repair_prompt 在首次 LLM 错误后成功重试。"""

    runner = _build_runner(tmp_path)
    valid_plan = {
        "patches": [{"target_excerpt": "旧", "target_section_heading": "标题", "replacement": "新", "reason": "修"}],
        "notes": [],
    }
    valid_repair_json = json.dumps(valid_plan)
    fake_agent = _make_fake_prompt_agent([
        _make_app_result(errors=[{"error": "repair temporary failure"}]),
        _make_app_result(content=valid_repair_json),
    ])
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.time.sleep", lambda _: None)

    plan, raw = runner._prompt_runner.run_repair_prompt("测试 repair prompt")

    assert len(plan["patches"]) == 1
    assert plan["patches"][0]["target_excerpt"] == "旧"
    assert plan["patches"][0]["replacement"] == "新"
    assert raw == valid_repair_json


@pytest.mark.unit
def test_run_repair_prompt_does_not_retry_when_cancelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 repair prompt 遇到取消时直接透传，不进入重试。"""

    runner = _build_runner(tmp_path)
    fake_agent = _FakeCancelledPromptAgent(cancel_reason="测试取消")
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.time.sleep", lambda _: None)

    with pytest.raises(CancelledError, match="写作 Agent 执行被取消: 测试取消"):
        runner._prompt_runner.run_repair_prompt("测试 repair prompt")

    assert len(fake_agent.calls) == 1


@pytest.mark.unit
def test_run_middle_task_worker_reraises_cancelled_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证中间章节 worker 遇到取消时不会降级为兜底结果。"""

    runner = _build_runner(tmp_path)
    task = ChapterTask(index=3, title="中间章节", skeleton="## 中间章节")

    def _raise_cancelled(*_args: object, **_kwargs: object) -> ChapterResult:
        """测试桩：模拟下游执行被取消。"""

        raise CancelledError("用户取消")

    monkeypatch.setattr(runner, "_run_single_chapter", _raise_cancelled)

    with pytest.raises(CancelledError, match="用户取消"):
        runner._run_middle_task_worker(task=task, company_name="TestCo")


@pytest.mark.unit
def test_run_middle_tasks_in_parallel_stops_dispatching_after_scene_creation_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证中间章节批次遇到 Agent 创建失败后不会继续派发后续任务。"""

    from dayu.services.internal.write_pipeline import pipeline as pipeline_module

    runner = _build_runner(tmp_path)
    monkeypatch.setattr(pipeline_module, "_MIDDLE_CHAPTER_MAX_WORKERS", 2)

    started_titles: list[str] = []
    middle_tasks = [
        ChapterTask(index=1, title="A", skeleton="## A"),
        ChapterTask(index=2, title="B", skeleton="## B"),
        ChapterTask(index=3, title="C", skeleton="## C"),
        ChapterTask(index=4, title="D", skeleton="## D"),
    ]

    def _run_worker(*, task: ChapterTask, company_name: str) -> ChapterResult:
        """测试桩：首个任务失败，其余已启动任务短暂占位。"""

        del company_name
        started_titles.append(task.title)
        if task.title == "A":
            raise SceneAgentCreationError(scene_name="write", agent_label="写作 Agent")
        time.sleep(0.05)
        return ChapterResult(
            index=task.index,
            title=task.title,
            status="passed",
            content=f"## {task.title}\n正文",
            audit_passed=True,
        )

    monkeypatch.setattr(runner, "_run_middle_task_worker", _run_worker)

    with pytest.raises(SceneAgentCreationError, match="写作 Agent 创建失败: scene=write"):
        runner._run_middle_tasks_in_parallel(
            middle_tasks=middle_tasks,
            chapter_results={},
            manifest=RunManifest(
                version="v1",
                signature="sig",
                config=runner._write_config,
            ),
            company_name="TestCo",
        )

    assert "C" not in started_titles
    assert "D" not in started_titles
    assert set(started_titles).issubset({"A", "B"})


@pytest.mark.unit
def test_run_repair_prompt_retries_on_json_parse_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 _run_repair_prompt 在首次 JSON 解析失败后成功重试。"""

    runner = _build_runner(tmp_path)
    valid_plan = {
        "patches": [{"target_excerpt": "旧", "target_section_heading": "标题", "replacement": "新", "reason": "修"}],
        "notes": [],
    }
    valid_repair_json = json.dumps(valid_plan)
    fake_agent = _make_fake_prompt_agent([
        _make_app_result(content="这不是合法的 JSON"),
        _make_app_result(content=valid_repair_json),
    ])
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.time.sleep", lambda _: None)

    plan, raw = runner._prompt_runner.run_repair_prompt("测试 repair prompt")

    assert len(plan["patches"]) == 1
    assert plan["patches"][0]["target_excerpt"] == "旧"
    assert plan["patches"][0]["replacement"] == "新"
    assert raw == valid_repair_json


@pytest.mark.unit
def test_run_repair_prompt_raises_repair_output_error_after_parse_retry_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 _run_repair_prompt JSON 解析重试耗尽后抛出 RepairOutputError。"""

    runner = _build_runner(tmp_path)
    fake_agent = _make_fake_prompt_agent([
        _make_app_result(content="非法 JSON 第一次"),
        _make_app_result(content="非法 JSON 第二次"),
    ])
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", fake_agent)
    monkeypatch.setattr("dayu.services.internal.write_pipeline.pipeline.time.sleep", lambda _: None)

    with pytest.raises(RepairOutputError):
        runner._prompt_runner.run_repair_prompt("测试 repair prompt")


@pytest.mark.unit
def test_extract_json_text_extracts_from_text() -> None:
    """验证 _extract_json_text 从文本中抽取 JSON 子串。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _extract_json_text("prefix {\"a\": 1} suffix") == '{"a": 1}'
    assert _extract_json_text('no json here') == ""
    assert _extract_json_text("{only open") == ""
    assert _extract_json_text("}only close") == ""
    assert _extract_json_text("}only close then {open") == ""
    assert _extract_json_text("{\"ok\": true}") == '{"ok": true}'


@pytest.mark.unit
def test_strip_evidence_section_at_end_of_document() -> None:
    """验证 _strip_evidence_section 剥除文末证据小节。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = "## 第一章\n\n正文\n\n### 证据与出处\n\n- 来源1"
    result = _strip_evidence_section(content)
    assert "### 证据与出处" not in result
    assert "来源1" not in result


@pytest.mark.unit
def test_strip_evidence_section_between_headings() -> None:
    """验证 _strip_evidence_section 在两个同级标题之间剥除。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = (
        "## 公司概况\n\n正文A\n\n"
        "### 证据与出处\n\n- 来源X\n\n"
        "## 财务分析\n\n正文B"
    )
    result = _strip_evidence_section(content)
    assert "### 证据与出处" not in result
    assert "来源X" not in result
    assert "## 财务分析" in result
    assert "正文B" in result


@pytest.mark.unit
def test_strip_generated_parenthetical_summary() -> None:
    """验证 _strip_generated_parenthetical_summary 各分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 无括号：直接返回
    assert _strip_generated_parenthetical_summary("年报 第5页") == "年报 第5页"
    # 空 inner：返回 head
    assert _strip_generated_parenthetical_summary("年报 ()") == "年报"
    # inner 仅有逗号：返回 head
    assert _strip_generated_parenthetical_summary("年报 (,)") == "年报"
    # 全是四位年份：保留
    assert _strip_generated_parenthetical_summary("年报 (2023, 2024)") == "年报 (2023, 2024)"
    # 两个以上非年份 token：截断
    assert _strip_generated_parenthetical_summary("年报 (收入增长, 利润下降, 现金流)") == "年报"
    # 单个非年份 token：保留
    assert _strip_generated_parenthetical_summary("年报 (收入增长)") == "年报 (收入增长)"


@pytest.mark.unit
def test_build_current_visible_headings_block() -> None:
    """验证 _build_current_visible_headings_block 有标题和无标题情况。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 无标题
    result = _build_current_visible_headings_block("纯正文没有标题")
    assert "没有可见标题" in result
    # 有标题
    result = _build_current_visible_headings_block("## 标题A\n\n### 子标题")
    assert "# 标题A" in result
    assert "## 子标题" in result


@pytest.mark.unit
def test_find_enclosing_heading_section_multiple_and_missing() -> None:
    """验证 _find_enclosing_heading_section 各分支覆盖。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 无 headings → None
    assert _find_enclosing_heading_section(
        markdown_text="some text",
        label_text="label",
        headings=[],
    ) is None

    # 多处匹配 → None
    md = "## H1\n\n- slot: 值\n\n## H2\n\n- slot: 值"
    result = _find_enclosing_heading_section(
        markdown_text=md,
        label_text="slot:",
        headings=[
            {"level": 2, "title": "H1", "start": 0, "end": 5},
            {"level": 2, "title": "H2", "start": 17, "end": 22},
        ],
    )
    assert result is None

    # 唯一匹配但无 enclosing heading（label 在所有 heading 之前）
    result = _find_enclosing_heading_section(
        markdown_text="preamble\n- slot: 值\n## H1\n\nbody",
        label_text="slot:",
        headings=[
            {"level": 2, "title": "H1", "start": 18, "end": 23},
        ],
    )
    assert result is None


@pytest.mark.unit
def test_find_enclosing_heading_section_finds_enclosing() -> None:
    """验证 _find_enclosing_heading_section 找到正确的 enclosing heading。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    md = "## H1\n\n- slot: 值\n\n## H2\n\nbody2"
    result = _find_enclosing_heading_section(
        markdown_text=md,
        label_text="slot:",
        headings=[
            {"level": 2, "title": "H1", "start": 0, "end": 5},
            {"level": 2, "title": "H2", "start": 17, "end": 22},
        ],
    )
    assert result == (0, 17)


@pytest.mark.unit
def test_find_markdown_section_span_strategy3_fallback() -> None:
    """验证 _find_markdown_section_span 策略3回退到 enclosing heading。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # heading_text 既不是精确标题也不是归一化标题，而是 bullet 标签
    md = "## H1\n\n- slotA: 值\n\n## H2\n\nbody2"
    result = _find_markdown_section_span(markdown_text=md, heading_text="slotA:")
    # 应 fallback 到 enclosing heading section 查找
    assert result is not None


@pytest.mark.unit
def test_find_markdown_section_span_not_found() -> None:
    """验证 _find_markdown_section_span 找不到标题返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    md = "## H1\n\nbody\n\n## H2\n\nbody2"
    assert _find_markdown_section_span(markdown_text=md, heading_text="不存在的标题") is None


@pytest.mark.unit
def test_find_normalized_line_like_spans() -> None:
    """验证 _find_normalized_line_like_spans 各分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 空 target → []
    assert _find_normalized_line_like_spans("some text", "", bullet_only=False) == []

    # 空行跳过
    text = "第一行\n\n第二行\n"
    spans = _find_normalized_line_like_spans(text, "第二行", bullet_only=False)
    assert len(spans) == 1
    assert spans[0][0] > 0

    # bullet_only=True 但行不是 bullet → 不匹配
    text = "普通行\n- bullet行\n"
    spans = _find_normalized_line_like_spans(text, "普通行", bullet_only=True)
    assert len(spans) == 0

    # bullet_only=True 且匹配（normalize 后包含 bullet 前缀）
    spans = _find_normalized_line_like_spans(text, "-bullet行", bullet_only=True)
    assert len(spans) == 1

    # startswith 匹配（target 是行的前缀）：用非 bullet 行
    text = "来源详情页码\n"
    spans = _find_normalized_line_like_spans(text, "来源详情", bullet_only=False)
    assert len(spans) == 1

    # target 更长于行，target startswith 行（行是 target 前缀）
    text = "短\n"
    spans = _find_normalized_line_like_spans(text, "短文本更多内容", bullet_only=False)
    assert len(spans) == 1


@pytest.mark.unit
def test_find_normalized_paragraph_spans() -> None:
    """验证 _find_normalized_paragraph_spans 各分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 空 target → []
    assert _find_normalized_paragraph_spans("some text", "") == []

    # 完全匹配段落
    text = "段落一\n\n段落二\n"
    spans = _find_normalized_paragraph_spans(text, "段落一")
    assert len(spans) == 1
    assert spans[0] == (0, len("段落一\n"))

    # target 段落 startswith 匹配
    spans = _find_normalized_paragraph_spans(text, "段落")
    assert len(spans) == 2

    # 不匹配
    spans = _find_normalized_paragraph_spans(text, "不存在")
    assert len(spans) == 0

    # 多行段落
    text = "第一行\n第二行\n\n下一节\n"
    spans = _find_normalized_paragraph_spans(text, "第一行第二行")
    assert len(spans) == 1


@pytest.mark.unit
def test_find_normalized_match_spans_all_kinds() -> None:
    """验证 _find_normalized_match_spans 各 target_kind 分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = "普通段落内容\n\n- bullet项目\n"

    # 空 target → []
    assert _find_normalized_match_spans(text, "", RepairTargetKind.SUBSTRING) == []

    # SUBSTRING
    spans = _find_normalized_match_spans(text, "普通段落内容", RepairTargetKind.SUBSTRING)
    assert len(spans) == 1

    # LINE
    spans = _find_normalized_match_spans(text, "普通段落内容", RepairTargetKind.LINE)
    assert len(spans) == 1

    # BULLET
    spans = _find_normalized_match_spans(text, "-bullet项目", RepairTargetKind.BULLET)
    assert len(spans) == 1

    # PARAGRAPH
    spans = _find_normalized_match_spans(text, "普通段落", RepairTargetKind.PARAGRAPH)
    assert len(spans) == 1


@pytest.mark.unit
def test_find_all_occurrences() -> None:
    """验证 _find_all_occurrences 各分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 空 target → []
    assert _find_all_occurrences("abc", "") == []

    # 多次匹配
    assert _find_all_occurrences("abcabc", "abc") == [0, 3]

    # 无匹配
    assert _find_all_occurrences("xyz", "abc") == []

    # 重叠不应出现（非重叠搜索）
    assert _find_all_occurrences("aaa", "aa") == [0]


@pytest.mark.unit
def test_should_run_fix_placeholders_empty() -> None:
    """验证 _should_run_fix_placeholders 空内容返回 False。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _should_run_fix_placeholders("") is False
    assert _should_run_fix_placeholders("   \n  ") is False


@pytest.mark.unit
def test_normalize_evidence_line_no_dash_prefix() -> None:
    """验证 _normalize_evidence_line 非 dash 开头行原样返回。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.services.internal.write_pipeline.audit_formatting import _normalize_evidence_line

    assert _normalize_evidence_line("不是 evidence 行") == "不是 evidence 行"


@pytest.mark.unit
def test_normalize_chapter_markdown_for_audit_strips_code_fences() -> None:
    """验证 _normalize_chapter_markdown_for_audit 在 evidence section 中去除代码围栏。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = (
        "## 公司概况\n\n正文\n\n"
        "### 证据与出处\n\n"
        "```\n"
        "- 来源A | 详情 | 页码\n"
        "```\n"
    )
    result = _normalize_chapter_markdown_for_audit(content)
    assert "```" not in result
    assert "- 来源A" in result


@pytest.mark.unit
def test_normalize_chapter_markdown_for_audit_non_evidence_heading() -> None:
    """验证 evidence section 内遇到新 heading 退出 evidence 模式。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = (
        "### 证据与出处\n\n"
        "- 来源A | 详情 | 页码\n\n"
        "### 下一节标题\n\n"
        "不是 evidence 行\n"
    )
    result = _normalize_chapter_markdown_for_audit(content)
    # "### 下一节标题" 之后的行不应被当作 evidence 处理
    assert "### 下一节标题" in result
    assert "不是 evidence 行" in result


@pytest.mark.unit
def test_extract_evidence_section_block_empty() -> None:
    """验证 _extract_evidence_section_block 无证据小节返回空字符串。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _extract_evidence_section_block("## 只有标题\n\n正文") == ""


@pytest.mark.unit
def test_replace_evidence_section_block_errors() -> None:
    """验证 _replace_evidence_section_block 错误路径。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 新 section 不以 "### 证据与出处" 开头
    with pytest.raises(ValueError, match="证据与出处"):
        _replace_evidence_section_block(
            chapter_markdown="## 正文",
            evidence_section="### 错误标题",
        )

    # 正文缺少 evidence section
    with pytest.raises(ValueError, match="缺少"):
        _replace_evidence_section_block(
            chapter_markdown="## 正文\n\n没有证据小节",
            evidence_section="### 证据与出处\n\n- 来源",
        )


@pytest.mark.unit
def test_collect_all_evidence_items_without_ordered_titles() -> None:
    """验证 _collect_all_evidence_items 不传 ordered_titles 时遍历 values。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    chapter_results = {
        "第一章": ChapterResult(
            index=1,
            title="第一章",
            status="passed",
            content="## 第一章",
            evidence_items=["来源X"],
            audit_passed=True,
        ),
        "第二章": ChapterResult(
            index=2,
            title="第二章",
            status="passed",
            content="## 第二章",
            evidence_items=["来源Y"],
            audit_passed=True,
        ),
    }
    items = _collect_all_evidence_items(chapter_results)
    assert sorted(items) == ["来源X", "来源Y"]


# ── audit_evidence_rewriter 单元函数补充测试 ──────────────────────────────────


@pytest.mark.unit
def test_has_anchor_rewrite_candidates_returns_false_for_empty_entries() -> None:
    """验证空条目列表时 _has_anchor_rewrite_candidates 返回 False。"""

    result = EvidenceConfirmationResult(entries=[])
    assert _has_anchor_rewrite_candidates(result) is False


@pytest.mark.unit
def test_has_anchor_rewrite_candidates_returns_false_for_unrelated_status() -> None:
    """验证条目状态不匹配可重写类型时返回 False。"""

    result = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="v1",
                rule=AuditRuleCode.E1,
                excerpt="测试",
                status=EvidenceConfirmationStatus.SUPPORTED,
                reason="已支持",
                rewrite_hint="",
            )
        ]
    )
    assert _has_anchor_rewrite_candidates(result) is False


@pytest.mark.unit
def test_extract_anchor_locator_from_hint_full_line() -> None:
    """验证 _extract_anchor_locator_from_hint 提取完整证据条目定位。"""

    hint = "将证据条目更改为：SEC EDGAR | Form 10-K | Part II - Item 7"
    locator = _extract_anchor_locator_from_hint(hint)
    assert locator == "SEC EDGAR | Form 10-K | Part II - Item 7"


@pytest.mark.unit
def test_extract_anchor_locator_from_hint_path_match() -> None:
    """验证 _extract_anchor_locator_from_hint 提取 Part 路径。"""

    reason = "数据在同一文件的 Part II - Item 8 - Financial Information 中。"
    locator = _extract_anchor_locator_from_hint(reason)
    assert locator == "Part II - Item 8 - Financial Information"


@pytest.mark.unit
def test_extract_anchor_locator_from_hint_returns_empty() -> None:
    """验证 _extract_anchor_locator_from_hint 无法匹配时返回空串。"""

    locator = _extract_anchor_locator_from_hint("无任何定位信息")
    assert locator == ""


@pytest.mark.unit
def test_extract_period_value_from_hint_matched() -> None:
    """验证 _extract_period_value_from_hint 匹配 Period 串。"""

    value = _extract_period_value_from_hint(
        rewrite_hint="Period:FY2025,FY2024",
        reason="需要扩展期间",
    )
    assert value == "FY2025,FY2024"


@pytest.mark.unit
def test_extract_period_value_from_hint_no_match() -> None:
    """验证 _extract_period_value_from_hint 无匹配时返回空串。"""

    value = _extract_period_value_from_hint(
        rewrite_hint="无期间信息",
        reason="无期间信息",
    )
    assert value == ""


@pytest.mark.unit
def test_extract_rows_from_hint_with_rows_and_quoted() -> None:
    """验证 _extract_rows_from_hint 同时提取 Rows: 和引号行标签。"""

    rows = _extract_rows_from_hint(
        rewrite_hint='Rows:Net sales,Products',
        reason='"Services"行也应包含',
    )
    assert "Net sales" in rows
    assert "Products" in rows
    assert "Services" in rows


@pytest.mark.unit
def test_extract_rows_from_hint_skips_part_and_fs_prefix() -> None:
    """验证 _extract_rows_from_hint 跳过 Part 和 Financial Statement 前缀引号。"""

    rows = _extract_rows_from_hint(
        rewrite_hint="",
        reason='"Part II - Item 7"应引用，"Financial Statement:income"应包含',
    )
    assert "Part II - Item 7" not in rows
    assert "Financial Statement:income" not in rows


@pytest.mark.unit
def test_extract_rows_from_hint_deduplicates() -> None:
    """验证 _extract_rows_from_hint 对重复行标签去重。"""

    rows = _extract_rows_from_hint(
        rewrite_hint='Rows:Net sales,"Net sales"',
        reason="",
    )
    assert rows.count("Net sales") == 1


@pytest.mark.unit
def test_extract_rows_from_hint_empty_quoted() -> None:
    """验证 _extract_rows_from_hint 空引号内容不加入结果。"""

    rows = _extract_rows_from_hint(
        rewrite_hint="",
        reason='""空值',
    )
    assert rows == []


@pytest.mark.unit
def test_rewrite_financial_statement_evidence_line_adds_period() -> None:
    """验证 _rewrite_financial_statement_evidence_line 补入缺失的 Period。"""

    line = "SEC EDGAR | Form 10-K | Filed 2025 | Accession 001 | Financial Statement:income | Rows:Net sales"
    result = _rewrite_financial_statement_evidence_line(
        line, statement_type="income", period_value="FY2025,FY2024", rows=[],
    )
    assert "Period:FY2025,FY2024" in result
    assert "Rows:Net sales" in result


@pytest.mark.unit
def test_rewrite_financial_statement_evidence_line_merges_periods() -> None:
    """验证 _rewrite_financial_statement_evidence_line 合并已有 Period。"""

    line = "SEC EDGAR | Form 10-K | Filed 2025 | Accession 001 | Financial Statement:income | Period:FY2025 | Rows:Net sales"
    result = _rewrite_financial_statement_evidence_line(
        line, statement_type="income", period_value="FY2024,FY2023", rows=[],
    )
    assert "FY2025,FY2024,FY2023" in result


@pytest.mark.unit
def test_rewrite_financial_statement_evidence_line_merges_periods_label() -> None:
    """验证 _rewrite_financial_statement_evidence_line 使用 Periods: 标签合并。"""

    line = "SEC EDGAR | Form 10-K | Filed 2025 | Accession 001 | Financial Statement:income | Periods:FY2025 | Rows:Net sales"
    result = _rewrite_financial_statement_evidence_line(
        line, statement_type="income", period_value="FY2024", rows=[],
    )
    assert "Periods:FY2025,FY2024" in result


@pytest.mark.unit
def test_rewrite_financial_statement_evidence_line_adds_rows() -> None:
    """验证 _rewrite_financial_statement_evidence_line 补入缺失的 Rows。"""

    line = "SEC EDGAR | Form 10-K | Filed 2025 | Accession 001 | Financial Statement:income | Period:FY2025"
    result = _rewrite_financial_statement_evidence_line(
        line, statement_type="income", period_value="", rows=["Products", "Services"],
    )
    assert "Rows:Products,Services" in result


@pytest.mark.unit
def test_rewrite_financial_statement_evidence_line_merges_rows() -> None:
    """验证 _rewrite_financial_statement_evidence_line 合并已有 Rows。"""

    line = "SEC EDGAR | Form 10-K | Filed 2025 | Accession 001 | Financial Statement:income | Period:FY2025 | Rows:Net sales"
    result = _rewrite_financial_statement_evidence_line(
        line, statement_type="income", period_value="", rows=["Products", "Services"],
    )
    assert "Net sales" in result
    assert "Products" in result
    assert "Services" in result


@pytest.mark.unit
def test_rewrite_financial_statement_evidence_line_no_match_returns_original() -> None:
    """验证报表类型不匹配时原样返回。"""

    line = "SEC EDGAR | Form 10-K | Filed 2025 | Accession 001 | Financial Statement:income | Period:FY2025"
    result = _rewrite_financial_statement_evidence_line(
        line, statement_type="cash_flow", period_value="FY2024", rows=[],
    )
    assert result == line


@pytest.mark.unit
def test_rewrite_evidence_lines_no_evidence_returns_empty() -> None:
    """验证无证据条目时返回空列表和空集合。"""

    chapter_markdown = "## 章节标题\n\n无证据条目"
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="v1",
                rule=AuditRuleCode.E1,
                excerpt="测试",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="Part II - Item 7",
                rewrite_hint="",
            )
        ]
    )
    lines, resolved = _rewrite_evidence_lines_and_collect_resolved_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )
    assert lines == []
    assert resolved == set()


@pytest.mark.unit
def test_rewrite_evidence_lines_with_same_filing_statement_anchor_fix() -> None:
    """验证 anchor_fix.kind=same_filing_statement 时从结构化数据补入 period/rows。"""

    chapter_markdown = (
        "## 财务表现\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | "
        "Financial Statement:income | Period:FY2025 | Rows:Net sales\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E2,
                excerpt="毛利率",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="需要补入",
                rewrite_hint="",
                anchor_fix=EvidenceAnchorFix(
                    kind="same_filing_statement",
                    action="append",
                    keep_existing_evidence=True,
                    statement_type="income",
                    period="FY2025,FY2024",
                    rows=["Products", "Services"],
                ),
            )
        ]
    )

    rewritten, resolved = _rewrite_evidence_lines_and_collect_resolved_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    normalized = [line.replace(" ", "") for line in rewritten]
    assert any("FY2025,FY2024" in l for l in normalized)
    assert any("Products" in l for l in normalized)
    assert "evidence_1" in resolved


@pytest.mark.unit
def test_rewrite_evidence_lines_appends_new_statement_line() -> None:
    """验证现有行不匹配报表类型时，anchor_fix 非 None 时补建新 evidence line。"""

    chapter_markdown = (
        "## 财务表现\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | "
        "Financial Statement:cash_flow | Period:FY2025 | Rows:Cash\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E2,
                excerpt="收入增长",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="需要补入利润表",
                rewrite_hint="Financial Statement:income | Period:FY2025 | Rows:Net sales",
                anchor_fix=EvidenceAnchorFix(
                    kind="same_filing_statement",
                    action="append",
                    keep_existing_evidence=True,
                    statement_type="income",
                    period="FY2025",
                    rows=["Net sales"],
                ),
            )
        ]
    )

    rewritten, resolved = _rewrite_evidence_lines_and_collect_resolved_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    assert any("Financial Statement:income" in line for line in rewritten)
    assert "evidence_1" in resolved


@pytest.mark.unit
def test_rewrite_evidence_lines_raw_locator_empty_with_resolved() -> None:
    """验证 raw_locator 为空但已通过报表重写解决时，仍收集 violation_id。"""

    chapter_markdown = (
        "## 财务表现\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | "
        "Financial Statement:income | Period:FY2025 | Rows:Net sales\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E2,
                excerpt="收入增长",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="需要补入",
                rewrite_hint="Financial Statement:income | Period:FY2024 | Rows:Products",
                anchor_fix=EvidenceAnchorFix(
                    kind="same_filing_statement",
                    action="append",
                    keep_existing_evidence=True,
                    statement_type="income",
                    period="FY2024",
                    rows=["Products"],
                ),
            )
        ]
    )

    _, resolved = _rewrite_evidence_lines_and_collect_resolved_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    assert "evidence_1" in resolved


@pytest.mark.unit
def test_rewrite_evidence_lines_sec_edgar_raw_locator() -> None:
    """验证 raw_locator 以 SEC EDGAR 开头时直接作为候选行。"""

    chapter_markdown = (
        "## 章节\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | Part I\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="测试",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="在另一部分",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | Part II - Item 7",
            )
        ]
    )

    rewritten, resolved = _rewrite_evidence_lines_and_collect_resolved_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    assert any("Part II - Item 7" in line for line in rewritten)
    assert "evidence_1" in resolved


@pytest.mark.unit
def test_rewrite_evidence_lines_uploaded_raw_locator() -> None:
    """验证 raw_locator 以 Uploaded 开头时直接作为候选行。"""

    chapter_markdown = (
        "## 章节\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | Part I\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="测试",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="在上传文件中",
                rewrite_hint="将证据条目更改为：Uploaded | 2025 Annual Report | Part 3",
            )
        ]
    )

    rewritten, resolved = _rewrite_evidence_lines_and_collect_resolved_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    assert any("Uploaded | 2025 Annual Report | Part 3" in line for line in rewritten)
    assert "evidence_1" in resolved


@pytest.mark.unit
def test_rewrite_evidence_lines_evidence_prefix_empty_with_resolved() -> None:
    """验证 evidence_prefix 为空但已通过报表重写解决时，仍收集 violation_id。"""

    chapter_markdown = (
        "## 章节\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- short_line\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="测试",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="Part II - Item 7 应引用",
                rewrite_hint="Financial Statement:income | Period:FY2025 | Rows:Net sales",
                anchor_fix=EvidenceAnchorFix(
                    kind="same_filing_statement",
                    action="append",
                    keep_existing_evidence=True,
                    statement_type="income",
                    period="FY2025",
                    rows=["Net sales"],
                ),
            )
        ]
    )

    _, resolved = _rewrite_evidence_lines_and_collect_resolved_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    assert "evidence_1" not in resolved


@pytest.mark.unit
def test_rewrite_evidence_lines_duplicate_locator_skipped() -> None:
    """验证已存在的归一化定位行不重复添加。"""

    chapter_markdown = (
        "## 章节\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | Part II - Item 7 | Free Cash Flow\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="测试",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="已在 Part II - Item 7",
                rewrite_hint="将证据条目更改为：SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | Part II - Item 7 | Free Cash Flow",
            )
        ]
    )

    rewritten = _rewrite_evidence_lines_for_confirmed_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    assert len(rewritten) == 1


@pytest.mark.unit
def test_rewrite_evidence_lines_locator_with_preferred_accession() -> None:
    """验证从 rewrite_hint 提取 preferred_accession 并匹配对应 filing 前缀。"""

    chapter_markdown = (
        "## 章节\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | Part I\n"
        "- SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part I\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="测试",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="在第二个 filing 中",
                rewrite_hint="Accession 0001628280-26-003942 应引用 Part II - Item 7",
            )
        ]
    )

    rewritten = _rewrite_evidence_lines_for_confirmed_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    new_lines = [l for l in rewritten if "Part II - Item 7" in l]
    assert len(new_lines) == 1
    assert "0001628280-26-003942" in new_lines[0]


@pytest.mark.unit
def test_rewrite_evidence_lines_locator_no_prefix_no_resolved() -> None:
    """验证 locator 非 SEC EDGAR/Uploaded 且无可用 prefix 时，未解决的 entry 不加入 resolved。"""

    chapter_markdown = (
        "## 章节\n\n"
        "### 结论要点\n\n"
        "- test\n\n"
        "### 证据与出处\n\n"
        "- short\n"
    )
    confirmation = EvidenceConfirmationResult(
        entries=[
            EvidenceConfirmationEntry(
                violation_id="evidence_1",
                rule=AuditRuleCode.E1,
                excerpt="测试",
                status=EvidenceConfirmationStatus.SUPPORTED_ELSEWHERE_IN_SAME_FILING,
                reason="Part II - Item 7 应引用",
                rewrite_hint="Part II - Item 7",
            )
        ]
    )

    _, resolved = _rewrite_evidence_lines_and_collect_resolved_anchor_issues(
        chapter_markdown=chapter_markdown,
        confirmation_result=confirmation,
    )

    assert "evidence_1" not in resolved


@pytest.mark.unit
def test_build_evidence_prefix_from_existing_lines_matched_accession() -> None:
    """验证 _build_evidence_prefix_from_existing_lines 按 accession 匹配。"""

    lines = [
        "SEC EDGAR | Form 10-K | Filed 2025-10-31 | Accession 0000320193-25-000079 | Part I",
        "SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942 | Part I",
    ]
    prefix = _build_evidence_prefix_from_existing_lines(
        lines, preferred_accession="0001628280-26-003942",
    )
    assert prefix == "SEC EDGAR | Form 10-K | Filed 2026-01-29 | Accession 0001628280-26-003942"


@pytest.mark.unit
def test_build_evidence_prefix_from_existing_lines_strips_dash_prefix() -> None:
    """验证 _build_evidence_prefix_from_existing_lines 剥离 `- ` 前缀。"""

    lines = ["- SEC EDGAR | Form 10-K | Filed 2025 | Accession 001 | Part I"]
    prefix = _build_evidence_prefix_from_existing_lines(lines)
    assert prefix == "SEC EDGAR | Form 10-K | Filed 2025 | Accession 001"


@pytest.mark.unit
def test_build_evidence_prefix_from_existing_lines_short_line_skipped() -> None:
    """验证不足 5 段的行被跳过，最终返回空串。"""

    lines = ["SEC EDGAR | Form 10-K"]
    prefix = _build_evidence_prefix_from_existing_lines(lines)
    assert prefix == ""


@pytest.mark.unit
def test_build_evidence_prefix_from_existing_lines_empty() -> None:
    """验证空列表时返回空串。"""

    prefix = _build_evidence_prefix_from_existing_lines([])
    assert prefix == ""


@pytest.mark.unit
def test_validate_anchor_rewrite_postconditions_no_diff() -> None:
    """验证重写前后无差异时返回失败原因。"""

    error = _validate_anchor_rewrite_postconditions(
        original_chapter_markdown="same content",
        rewritten_chapter_markdown="same content",
        expected_evidence_lines=[],
        skeleton="## 章节",
    )
    assert error == "rewrite 未产生正文差异"


@pytest.mark.unit
def test_validate_anchor_rewrite_postconditions_evidence_mismatch() -> None:
    """验证 evidence section 与期望不一致时返回失败原因。"""

    original = "## 章节\n\n### 证据与出处\n\n- line1\n"
    rewritten = "## 章节\n\n### 证据与出处\n\n- line1\n- line2\n"
    error = _validate_anchor_rewrite_postconditions(
        original_chapter_markdown=original,
        rewritten_chapter_markdown=rewritten,
        expected_evidence_lines=["line1", "line2", "line3"],
        skeleton="## 章节",
    )
    assert error == "evidence section 与期望重写结果不一致"
