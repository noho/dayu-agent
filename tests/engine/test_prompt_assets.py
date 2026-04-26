"""Prompt 资产仓储测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.contracts.prompt_assets import SceneConversationAsset, SceneManifestAsset, SceneToolSelectionAsset
from dayu.prompting.scene_definition import load_scene_definition
from dayu.services.internal.write_pipeline.prompt_contracts import parse_task_prompt_contract
from dayu.startup.config_file_resolver import ConfigFileResolver
from dayu.startup.prompt_assets import FilePromptAssetStore


def _require_tool_selection(manifest: SceneManifestAsset) -> SceneToolSelectionAsset:
    """返回场景 manifest 中必填的工具选择配置。"""

    return manifest["tool_selection"]


def _require_tool_tags_any(manifest: SceneManifestAsset) -> list[str]:
    """返回 select 场景下声明的工具标签。"""

    tool_tags_any = _require_tool_selection(manifest).get("tool_tags_any")
    assert tool_tags_any is not None
    return tool_tags_any


def _require_conversation(manifest: SceneManifestAsset) -> SceneConversationAsset:
    """返回显式声明的 conversation 配置。"""

    conversation = manifest.get("conversation")
    assert conversation is not None
    return conversation


def _require_runtime_agent_max_iterations(manifest: SceneManifestAsset) -> int:
    """返回场景 manifest 显式声明的 Agent 最大迭代数。"""

    runtime = manifest.get("runtime")
    assert runtime is not None
    agent = runtime.get("agent")
    assert agent is not None
    max_iterations = agent.get("max_iterations")
    assert max_iterations is not None
    return max_iterations


def _require_runtime_runner_tool_timeout_seconds(manifest: SceneManifestAsset) -> float:
    """返回场景 manifest 显式声明的 Runner 工具超时秒数。"""

    runtime = manifest.get("runtime")
    assert runtime is not None
    runner = runtime.get("runner")
    assert runner is not None
    tool_timeout_seconds = runner.get("tool_timeout_seconds")
    assert tool_timeout_seconds is not None
    return tool_timeout_seconds


def _require_context_slots(manifest: SceneManifestAsset) -> tuple[str, ...]:
    """返回场景 manifest 显式声明的上下文槽位。"""

    return tuple(manifest["context_slots"])


_EXPECTED_THINKING_ALLOWED_NAMES: list[str] = [
    "mimo-v2.5-pro-thinking",
    "mimo-v2.5-pro-thinking-plan",
    "mimo-v2.5-pro-thinking-plan-sg",
    "deepseek-v4-flash-thinking",
    "deepseek-v4-pro-thinking",
    "qwen-plus-thinking",
    "ollama",
    "gpt-5.4-thinking",
    "claude-sonnet-4-6-thinking",
    "gemini-2.5-flash-thinking",
]

_EXPECTED_CONVERSATION_COMPACTION_ALLOWED_NAMES: list[str] = [
    "mimo-v2.5-pro-thinking",
    "mimo-v2.5-pro-thinking-plan",
    "mimo-v2.5-pro-thinking-plan-sg",
    "deepseek-v4-flash-thinking",
    "deepseek-v4-pro-thinking",
    "qwen-plus-thinking",
    "ollama",
    "gpt-5.4-thinking",
    "claude-sonnet-4-6-thinking",
    "gemini-2.5-flash-thinking",
]

_EXPECTED_WRITE_ALLOWED_NAMES: list[str] = [
    "mimo-v2.5-pro",
    "mimo-v2.5-pro-plan",
    "mimo-v2.5-pro-plan-sg",
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "qwen-plus",
    "ollama",
    "gpt-5.4",
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
]


@pytest.mark.unit
def test_load_task_prompt_normalizes_name_and_extension() -> None:
    """验证 task prompt 读取会补全目录与扩展名。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("write_chapter")

    assert "当前任务" in content
    assert "只基于下方输入写出当前章节完整正文" in content


@pytest.mark.unit
def test_load_task_prompt_contract_normalizes_name_and_extension() -> None:
    """验证 task prompt sidecar contract 读取会补全目录与扩展名。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    raw_contract = store.load_task_prompt_contract("write_chapter")
    contract = parse_task_prompt_contract(raw_contract, task_name="write_chapter")

    assert contract.prompt_name == "write_chapter"
    input_names = {item.name for item in contract.inputs}
    assert {"report_goal", "audience_profile", "chapter_goal"} <= input_names
    assert "company_facets_summary" in input_names


@pytest.mark.unit
def test_infer_company_facets_prompt_uses_controlled_vocab_inputs() -> None:
    """验证公司业务类型与关键约束判断 task prompt 只消费显式输入与受控候选。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("infer_company_facets")
    contract = store.load_task_prompt_contract("infer_company_facets")

    assert "判断这家公司“主要是什么生意”" in content
    assert "高确定性的受控分类任务" in content
    assert "主业务类型候选" in content
    assert "关键约束候选" in content
    assert "先为后续章节写作确定“判断分岔点”" in content
    assert "接下来系统会按固定写作模板逐章生成报告" in content
    assert "`business_model_tags` 的作用是" in content
    assert "`constraint_tags` 的作用是" in content
    assert '"business_model_tags"' in content
    assert '"constraint_tags"' in content
    assert '"suggest_business_model_tags"' not in content
    assert '"suggest_constraint_tags"' not in content
    assert "`business_model_tags` 最多选 3 个" in content
    assert "不要把次要业务、辅助业务、历史遗留业务、投资性业务或叙事热点抬成“主业务类型”" in content
    assert "若显式输入不足以稳定判断，可按财报工具工作流最小检索" in content
    input_names = {item["name"] for item in contract["inputs"]}
    assert input_names == {"company_meta", "business_model_candidates", "constraint_candidates"}


@pytest.mark.unit
def test_infer_scene_manifest_uses_fins_only() -> None:
    """验证 infer scene 只注册 fins 工具，并保留独立 runtime 预算。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("infer")
    definition = load_scene_definition(store, "infer")
    content = store.load_fragment_template("scenes/infer.md")

    assert manifest["scene"] == "infer"
    assert definition.tool_selection_policy.mode.value == "select"
    assert definition.tool_selection_policy.tool_tags_any == ("fins",)
    assert definition.model.default_name == "mimo-v2.5-pro-thinking-plan"
    assert list(definition.model.allowed_names) == _EXPECTED_THINKING_ALLOWED_NAMES
    assert definition.model.temperature_profile == "infer"
    assert definition.runtime.agent.max_iterations == 12
    assert definition.runtime.runner.tool_timeout_seconds == 90.0
    assert _require_runtime_agent_max_iterations(manifest) == 12
    assert _require_runtime_runner_tool_timeout_seconds(manifest) == 90.0
    assert "下游是按固定模板逐章写作" in content
    assert "先为模板写作确定“判断分岔点”" in content
    assert "`business_model_tags` 应让同一章节在面对不同行业" in content
    assert "`constraint_tags` 应让同一行业里的不同公司" in content
    assert "只从给定候选中选出最能代表“主业务类型”和“关键约束”的结果" in content


@pytest.mark.unit
def test_audit_prompt_keeps_web_access_date_and_objective_role_boundary() -> None:
    """验证审计 prompt 放宽客观角色描述，并收窄 E3 到真正不可定位的来源错误。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("audit_facts_tone_json")

    assert "客观角色、技术或生态定位" in content
    assert "C2) 章节越界" in content
    assert "触发时按 low severity 处理" in content
    assert 'S7) "证据与出处"格式可改进' in content
    assert "document id / accession / filing / URL / 标题路径不存在" in content
    assert "明显不属于当前 ticker / 当前公司" in content
    assert "对网页资料，若证据条目已提供页面标题、发布/访问日期和可追溯 URL，则默认不触发 `E3`" in content


@pytest.mark.unit
def test_write_scene_declares_shared_evidence_anchor_actions() -> None:
    """验证 write scene 承载 write/regenerate 共用的证据锚点动作规则。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_fragment_template("scenes/write.md")

    assert "最能直接支撑该句的具体证据锚点" in content
    assert "关键数字、比例、排名或具体断言写入正文时" in content
    assert "若 claim 只在父级 heading 中成立，不得用相邻子节" in content
    assert "必须先选定 statement/xbrl 锚点再写正文" in content
    assert "并在章末列出“### 证据与出处”" in content


@pytest.mark.unit
def test_fact_rules_support_financial_statement_and_xbrl_evidence_formats() -> None:
    """验证共享事实规则已支持 statement / xbrl 证据格式。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_fragment_template("base/fact_rules.md")

    assert "Financial Statement:{statement_type}" in content
    assert "XBRL Facts | Concepts:{concepts}" in content
    assert "Rows" in content
    assert "网页资料：机构/网站 | 页面标题 | 发布/访问日期 | URL:{可追溯链接}" in content
    assert "应使用自然语言让读者能区分“事实”与“分析”" in content
    assert "应使用“预计 / 预期 / 可能 / 到某年 / 指引 / 前瞻”等自然前瞻表述" in content


@pytest.mark.unit
def test_tools_fragment_describes_flat_tool_result_format() -> None:
    """验证工具使用指引与 ToolRegistry 的扁平结果格式保持一致。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_fragment_template("base/tools.md")

    assert "工具返回均为扁平 JSON" in content
    assert "有 `error` 字段 → 失败" in content
    assert "无 `error` 字段 → 成功" in content
    assert "`hint`" in content
    assert "{{directories}}" not in content


@pytest.mark.unit
def test_directories_fragment_is_separate_dynamic_prompt_piece() -> None:
    """验证 directories 已从 tools fragment 中剥离为独立动态片段。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_fragment_template("base/directories.md")

    assert "当前可访问路径" in content
    assert "{{directories}}" in content


@pytest.mark.unit
def test_repair_prompt_forbids_modifying_evidence_section() -> None:
    """验证 repair prompt 明确禁止修改“证据与出处”。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("repair_chapter")

    assert "只修正文，不修改 `### 证据与出处`" in content
    assert "证据锚点过粗、同一 filing 内有更准 section、statement rows/period 缺失等问题，不属于本任务" in content
    assert "不得生成任何命中 `### 证据与出处` 的 patch" in content
    assert "不得改写“章节正文”原有的小节标题文本" in content
    assert "而不是补入新事实" in content
    assert "可使用工具做最小必要检索" in content
    assert "已定位到可用证据，但本任务不改证据区" in content


@pytest.mark.unit
def test_overview_scene_manifest_disables_tools() -> None:
    """验证第0章概览 scene 为无工具轻场景，并复用全局默认 runtime。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("overview")

    assert manifest["scene"] == "overview"
    assert _require_tool_selection(manifest)["mode"] == "none"
    assert _require_runtime_agent_max_iterations(manifest) == 12
    assert _require_runtime_runner_tool_timeout_seconds(manifest) == 90.0


@pytest.mark.unit
def test_prompt_scene_manifest_is_independent_from_interactive() -> None:
    """验证 prompt scene 作为独立单轮场景维护。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("prompt")
    content = store.load_fragment_template("scenes/prompt.md")

    assert manifest["scene"] == "prompt"
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-thinking-plan"
    assert manifest["model"]["temperature_profile"] == "prompt"
    assert _require_tool_selection(manifest)["mode"] == "select"
    assert set(_require_tool_tags_any(manifest)) == {"web", "fins", "ingestion"}
    assert "单轮财报问答任务" in content
    assert "不延展成多轮访谈" in content


@pytest.mark.unit
def test_overview_scene_declares_cover_page_boundary() -> None:
    """验证第0章概览 scene 明确声明封面页边界。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_fragment_template("scenes/overview.md")

    assert "封面页" in content
    assert "不调用任何工具" in content
    assert "不补充新事实" in content
    assert "不输出审计判断、违规分类或证据裁决 JSON" in content


@pytest.mark.unit
def test_audit_prompt_does_not_require_evidence_line_to_repeat_numbers() -> None:
    """验证审计 prompt 不要求证据行重复正文数字。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("audit_facts_tone_json")

    assert "不得仅因为证据行未重复正文中的数字本身" in content
    assert "不要整句一刀切打回" in content
    assert "后续跟踪对象、待验证问题、敏感性变量或需持续观察的信号" in content
    assert "应优先写“修正为……”" in content
    assert "只有在无法稳定确定替代值时，才建议删除、降级或补证" in content


@pytest.mark.unit
def test_audit_prompt_scans_first_then_decides_pass_result() -> None:
    """验证审计 prompt 先列全问题，再统一计算 pass。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("audit_facts_tone_json")

    assert "先做全文扫描" in content
    assert "在当前审计模式下，`violations` 列全之后" in content
    assert "命中即停止" not in content


@pytest.mark.unit
def test_audit_prompt_supports_local_reaudit_after_repair() -> None:
    """验证审计 prompt 区分初始整章审计与修复后局部复审。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("audit_facts_tone_json")
    contract = store.load_task_prompt_contract("audit_facts_tone_json")

    assert "当前审计模式" in content
    assert "修复后局部复审" in content
    assert "不要把未触及且与当前修复无关的整章内容重新按初始整章审计标准全文复扫" in content
    input_names = {item["name"] for item in contract["inputs"]}
    assert {"audit_mode", "repair_contract"} <= input_names


@pytest.mark.unit
def test_audit_prompt_demotes_analysis_and_boundary_rules_to_low_priority() -> None:
    """验证审计 prompt 将 C2 / S4 / S5 / S6 / S7 降为低优先级提示。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("audit_facts_tone_json")

    assert "S4) 将推断、归因或解释性判断写成硬事实" in content
    assert "若句子本身已经明显是判断句，或位于“结论要点 / 当前判断 / 最关键的有利点或硬伤”等天然判断槽位，默认不触发" in content
    assert "S5) 将未来趋势、预测或情景判断写成当前事实" in content
    assert "S6) 引用公司原文中的前瞻性表述时，未明确这是公司披露、管理层预期或公司指引" in content
    assert "S7) \"证据与出处\"格式可改进，但来源与定位仍可追溯；触发时按 low severity 处理。" in content
    assert "若中高优先级写作风格问题累计达到 2 条及以上" in content
    assert "若某个引用前缀仅出现一次，且有助于区分事实、公司原文或公司预期，不触发 `S3`" in content


@pytest.mark.unit
def test_repair_prompt_prefers_natural_rewrite_over_analysis_or_forward_labels() -> None:
    """验证 repair 对 S4/S5/S6 优先自然改写，而非机械添加句首标签。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("repair_chapter")

    assert "默认优先自然改写" in content
    assert "不要通过添加 `【分析】`、`【前瞻】`、`【前瞻（原文）】`" in content


@pytest.mark.unit
def test_regenerate_prompt_keeps_generic_narrative_constraints() -> None:
    """验证 regenerate 只保留通用叙事约束，不夹带章节专用口径。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("regenerate_chapter")

    assert "正文组织服从“章节合同”的叙事方式" in content
    assert "若 `chapter_contract.narrative_mode` 为 `定义→结构→机制`" not in content
    assert "若“是否允许补入新事实”为 true" in content
    assert "若“是否允许补入新事实”为 false" in content
    assert "`section_rules`" not in content
    assert "当某个信息在当前正文中必须保留" in content
    assert "该信息位置必须按占位符统一格式规范化输出" in content


@pytest.mark.unit
def test_write_and_regenerate_prompts_treat_item_rule_as_conditional_visible_structure() -> None:
    """验证 write/regenerate 已将 ITEM_RULE 定义为条件可见结构，而非隐藏槽位。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    write_content = store.load_task_prompt("write_chapter")
    regenerate_content = store.load_task_prompt("regenerate_chapter")

    assert "再判断“条件写作规则”中每条 `ITEM_RULE` 的 `when` 是否成立" in regenerate_content
    assert "再判断“条件写作规则”中每条 `ITEM_RULE` 的 `when` 是否成立" in write_content
    assert "不成立时，整项不输出，不解释、不占位" in write_content
    assert "不成立时整项不输出，不解释、不占位" in regenerate_content
    assert "按照“bullet格式要求”作为独立 bullet 输出" in write_content
    assert "按照“bullet格式要求”作为独立 bullet 输出" in regenerate_content
    assert "`ITEM_RULE.item` 是内部写作提示" in write_content
    assert "`ITEM_RULE.item` 是内部写作提示" in regenerate_content
    assert "不得输出“只有 bullet、没有正文”的空条目" in write_content
    assert "不得输出“只有 bullet、没有正文”的空条目" in regenerate_content
    assert "条件写作规则" in write_content
    assert "隐藏补充槽位" not in write_content
    assert "隐藏补充槽位" not in regenerate_content


@pytest.mark.unit
def test_fix_prompt_only_targets_placeholders_not_structure_or_style() -> None:
    """验证 fix task 只处理占位符，不修改标题、结构或风格。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("fix_placeholders")

    assert "已合规正文必须保持不动" in content
    assert "所有非标占位符必须转换为占位符统一格式或补入实际数据" in content


@pytest.mark.unit
def test_audit_prompt_only_allows_tool_use_for_cited_evidence_verification() -> None:
    """验证 audit 为无工具疑似检查，confirm 才做已引用证据复核。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    scene_content = store.load_fragment_template("scenes/audit.md")
    confirm_scene_content = store.load_fragment_template("scenes/confirm.md")
    confirm_task_content = store.load_task_prompt("confirm_evidence_violations")

    assert "你当前没有任何可调用工具" in scene_content
    assert "基于正文与“证据与出处”文本，输出疑似违规清单" in scene_content
    assert "对 `E1/E2/E3` 只标记“这里可能缺证据”或“证据锚点可能不够准”" in scene_content
    assert "工具只可用于复核当前输入中“证据与出处”已经列出的来源与定位" in confirm_scene_content
    assert "只允许复核“证据与出处”已经列出的 filings / sections / pages / URLs" in confirm_task_content
    assert "若证据条目使用 `Financial Statement:{statement_type}` 格式" in confirm_task_content
    assert "若证据条目使用 `XBRL Facts` 格式" in confirm_task_content
    assert "supported_elsewhere_in_same_filing" in confirm_task_content
    assert '"anchor_fix"' in confirm_task_content
    assert "same_filing_section|same_filing_statement|same_filing_evidence_line" in confirm_task_content
    assert "请省略 `anchor_fix` 字段；不要输出空对象 `{}`" in confirm_task_content
    assert "在同一 item 的直接父级 heading 中即可被支持" in confirm_task_content
    assert "不要新增新的违规点" in confirm_task_content
    assert "条件写作规则" in store.load_task_prompt("audit_facts_tone_json")


@pytest.mark.unit
def test_fact_rules_cover_derived_metrics_and_stronger_synthesis_rules() -> None:
    """验证派生指标与更强合成判断的通用规则由 fact_rules 单点承载。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    fact_rules = store.load_fragment_template("base/fact_rules.md")
    write_scene = store.load_fragment_template("scenes/write.md")
    regenerate_scene = store.load_fragment_template("scenes/regenerate.md")

    expected_derived = "若一句话包含利润率、占比、同比变化、增减幅或其他派生指标"
    expected_stronger = "不得把多个分别成立的事实合成为更强判断"

    assert expected_derived in fact_rules
    assert expected_stronger in fact_rules
    assert expected_derived not in write_scene
    assert expected_stronger not in write_scene
    assert expected_derived not in regenerate_scene
    assert expected_stronger not in regenerate_scene


@pytest.mark.unit
def test_tools_prompt_requires_raw_ref_for_read_section_and_get_table() -> None:
    """验证财报工具指引要求 read_section/get_table 都原样复制 ref。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    tools_content = store.load_fragment_template("base/tools.md")

    assert "禁止猜测或自造`document_id`、`ref`、`table_ref`" in tools_content
    assert "必须先用 `list_documents` 获取 `document_id`" in tools_content


@pytest.mark.unit
def test_repair_scene_and_task_align_on_notes_when_patch_cannot_fully_fix() -> None:
    """验证 repair scene 与 task 对无法完全修复时的 notes 语义保持一致。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    scene_content = store.load_fragment_template("scenes/repair.md")
    task_content = store.load_task_prompt("repair_chapter")

    assert "应在 `notes` 中说明原因" in scene_content
    assert "应在 `notes` 中说明原因" in task_content
    assert "`patches` 必须至少包含 1 条 patch" in task_content
    assert "当前正文中的真实可见标题" in task_content
    assert "只能从“当前正文中的真实可见标题”中选" in task_content
    assert "`delete_claim`：表示该 claim 已被 confirm 证实缺证" in task_content
    assert "`target_kind=substring`：只改某个完整句子/完整 bullet/完整段落中的一小段文字" in task_content


@pytest.mark.unit
def test_confirm_prompt_allows_derived_metrics_supported_by_cited_raw_data() -> None:
    """验证 confirm 允许已引用原始数据直接支撑的派生指标通过。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("confirm_evidence_violations")

    assert "派生指标（如利润率、占比、同比变化）" in content
    assert "分子、分母与计算关系都能由当前 cited evidence 直接支持" in content
    assert "不要仅因该派生指标未被原文逐字披露就判定 `confirmed_missing`" in content


@pytest.mark.unit
def test_fill_overview_prompt_locks_facts_and_numbers_not_surface_phrasing() -> None:
    """验证 fill_overview 锁定事实边界，并把第0章收成封面页而非摘要三段式。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("fill_overview")

    assert "只能复用下方前文章节结构化输入中已出现的事实和数字" in content
    assert "可以在不改变事实边界的前提下压缩、重组或改写表述" in content
    assert "正文默认写成封面页三层结构" in content
    assert "公司简介" in content
    assert "当前经营与财务处在什么状态" in content
    assert "变量写成需要持续观察的观测点" in content
    assert "不输出“证据与出处”小节" in content
    assert "bullet下另起一行输出回答" in content
    assert "二者之间不留空行" in content
    assert "不得在bullet行直接追加回答内容" in content
    assert "禁止写成 `- 标签：回答`" in content
    assert "`当前最大卡点` 要写当前最阻止升级或维持当前动作的现实阻碍" in content
    assert "`下一步先验证什么` 默认只写 1 个最关键问题" in content
    assert "`公司简介` 只保留帮助第一次接触这家公司的人快速建立画像的最必要背景" in content
    assert "预设读者是第一次接触或第一次系统性了解这家公司" in content
    assert "整体语气要像写给买方同事、且对方第一次接触这家公司的封面导语" in content


@pytest.mark.unit
def test_decision_prompt_allows_tentative_conclusion_when_prior_chapters_conflict() -> None:
    """验证第10章在前文冲突较大时允许给出暂定结论，而不是被迫硬下判断。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("write_research_decision")

    assert "先判断当前输入是否足以支持明确拍板" in content
    assert "当前更倾向于..." in content
    assert "不要为了显得果断而硬下结论" in content


@pytest.mark.unit
def test_decision_prompt_requires_complete_takeaway_lines_and_change_conditions() -> None:
    """验证第10章要求结论要点完整落句，且必须写清会改变当前结论的条件。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_task_prompt("write_research_decision")

    assert "先决定当前研究动作，再在必要时补足支撑该动作所需的最小缺口" in content
    assert "默认只写 1 条主判断链" in content
    assert "都必须写清“什么变化会升级、降级或终止当前动作”；不要写“不适用”" in content
    assert "默认写 1 个升级阈值" in content
    assert "默认只写一个主卡点" in content
    assert "默认先写 1 个最关键问题" in content
    assert "bullet下另起一行输出回答" in content
    assert "二者之间不留空行" in content
    assert "不得在bullet行直接追加回答内容" in content
    assert "禁止写成 `- 标签：回答`" in content
    assert "`主卡点` 应写当前最阻止升级或维持当前动作的现实阻碍" in content


@pytest.mark.unit
def test_write_and_regenerate_prompts_require_separate_reader_facing_bullet_labels() -> None:
    """验证普通写作与重建 prompt 要求 reader-facing 的字段标签单独成行。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    write_content = store.load_task_prompt("write_chapter")
    regenerate_content = store.load_task_prompt("regenerate_chapter")

    assert "bullet下另起一行输出回答" in write_content
    assert "二者之间不留空行" in write_content
    assert "不得在bullet行直接追加回答内容" in write_content
    assert "禁止写成 `- 标签：回答`" in write_content
    assert "相邻 bullet 之间保留空行" in write_content
    assert "bullet下另起一行输出回答" in regenerate_content
    assert "二者之间不留空行" in regenerate_content
    assert "不得在bullet行直接追加回答内容" in regenerate_content
    assert "禁止写成 `- 标签：回答`" in regenerate_content
    assert "相邻 bullet 之间保留空行" in regenerate_content


@pytest.mark.unit
def test_load_scene_manifest_reads_interactive_manifest() -> None:
    """验证 scene manifest 可按场景名读取。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("interactive")

    assert manifest["scene"] == "interactive"
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-thinking-plan"
    assert manifest["model"]["allowed_names"] == _EXPECTED_THINKING_ALLOWED_NAMES
    assert manifest["model"]["temperature_profile"] == "interactive"
    assert _require_conversation(manifest)["enabled"] is True
    assert _require_tool_selection(manifest)["mode"] == "select"
    assert set(_require_tool_tags_any(manifest)) == {"fins", "web", "ingestion"}


@pytest.mark.unit
def test_wechat_scene_manifest_is_separate_and_requires_markdown_output() -> None:
    """验证 wechat scene 独立于 interactive，并明确要求输出 Markdown。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("wechat")
    content = store.load_fragment_template("scenes/wechat.md")

    assert manifest["scene"] == "wechat"
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-thinking-plan"
    assert manifest["model"]["temperature_profile"] == "interactive"
    assert _require_conversation(manifest)["enabled"] is True
    assert set(_require_tool_tags_any(manifest)) == {"fins", "web", "ingestion"}
    assert any(fragment["path"] == "scenes/wechat.md" for fragment in manifest["fragments"])
    assert "输出 Markdown 格式" in content


@pytest.mark.unit
def test_load_scene_manifest_reads_conversation_compaction_manifest() -> None:
    """验证 scene manifest 可按场景名读取 conversation_compaction。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("conversation_compaction")

    assert manifest["scene"] == "conversation_compaction"
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-thinking-plan"
    assert manifest["model"]["allowed_names"] == _EXPECTED_CONVERSATION_COMPACTION_ALLOWED_NAMES
    assert manifest["model"]["temperature_profile"] == "conversation_compaction"
    assert _require_tool_selection(manifest)["mode"] == "none"
    assert manifest["fragments"][0]["path"] == "scenes/conversation_compaction.md"


@pytest.mark.unit
def test_load_scene_manifest_reads_repair_manifest() -> None:
    """验证 scene manifest 可按场景名读取 repair。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("repair")

    assert manifest["scene"] == "repair"
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-plan"
    assert manifest["model"]["allowed_names"] == _EXPECTED_WRITE_ALLOWED_NAMES
    assert manifest["model"]["temperature_profile"] == "write"
    assert _require_runtime_agent_max_iterations(manifest) == 16
    assert _require_runtime_runner_tool_timeout_seconds(manifest) == 90.0
    assert _require_tool_selection(manifest)["mode"] == "select"
    assert set(_require_tool_tags_any(manifest)) == {"fins", "web"}


@pytest.mark.unit
def test_load_scene_manifest_reads_decision_manifest() -> None:
    """验证 scene manifest 可按场景名读取 decision。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("decision")

    assert manifest["scene"] == "decision"
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-thinking-plan"
    assert manifest["model"]["allowed_names"] == _EXPECTED_THINKING_ALLOWED_NAMES
    assert manifest["model"]["temperature_profile"] == "decision"
    assert _require_runtime_agent_max_iterations(manifest) == 12
    assert _require_runtime_runner_tool_timeout_seconds(manifest) == 90.0
    assert _require_tool_selection(manifest)["mode"] == "select"
    assert set(_require_tool_tags_any(manifest)) == {"fins", "web"}


@pytest.mark.unit
def test_decision_prompt_contract_exposes_prior_chapter_inputs() -> None:
    """验证第10章决策综合 prompt contract 暴露前文章节输入与决策控制字段。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    contract = store.load_task_prompt_contract("write_research_decision")

    assert contract["prompt_name"] == "write_research_decision"
    input_names = {item["name"] for item in contract["inputs"]}
    assert {
        "company_facets_summary",
        "report_goal",
        "chapter_goal",
        "decision_source_of_truth",
        "decision_allow_new_facts",
        "decision_allow_new_sources",
        "prior_chapters_input",
    } <= input_names


@pytest.mark.unit
def test_fill_overview_prompt_contract_exposes_company_facets_summary() -> None:
    """验证第0章概览 prompt contract 声明公司级 facet 摘要输入。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    contract = store.load_task_prompt_contract("fill_overview")

    assert contract["prompt_name"] == "fill_overview"
    input_names = {item["name"] for item in contract["inputs"]}
    assert "company_facets_summary" in input_names


@pytest.mark.unit
def test_load_scene_manifest_reads_audit_manifest_with_shared_base_fragments() -> None:
    """验证 audit scene 加载共享基础 fragment，且复用全局默认 runtime。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("audit")

    fragment_ids = {fragment["id"] for fragment in manifest["fragments"]}
    assert manifest["scene"] == "audit"
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-thinking-plan"
    assert manifest["model"]["allowed_names"] == _EXPECTED_THINKING_ALLOWED_NAMES
    assert manifest["model"]["temperature_profile"] == "audit"
    assert _require_runtime_agent_max_iterations(manifest) == 16
    assert _require_runtime_runner_tool_timeout_seconds(manifest) == 90.0
    assert _require_tool_selection(manifest)["mode"] == "none"
    assert {"base_agents", "base_fact_rules", "audit_scene"} <= fragment_ids
    assert "base_tools" not in fragment_ids


@pytest.mark.unit
def test_conversation_compaction_scene_requires_strict_json_without_tools() -> None:
    """验证 conversation_compaction scene 明确要求严格 JSON 且禁用工具。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_fragment_template("scenes/conversation_compaction.md")

    assert "不使用任何工具" in content
    assert "严格可解析 JSON 对象" in content
    assert "episode_summary" in content
    assert "pinned_state_patch" in content


@pytest.mark.unit
def test_load_scene_manifest_reads_confirm_manifest_with_shared_base_fragments() -> None:
    """验证 confirm scene 加载共享基础 fragment，并注册复核工具。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("confirm")

    fragment_ids = {fragment["id"] for fragment in manifest["fragments"]}
    assert manifest["scene"] == "confirm"
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-thinking-plan"
    assert manifest["model"]["allowed_names"] == _EXPECTED_THINKING_ALLOWED_NAMES
    assert manifest["model"]["temperature_profile"] == "audit"
    assert _require_runtime_agent_max_iterations(manifest) == 20
    assert _require_runtime_runner_tool_timeout_seconds(manifest) == 90.0
    assert _require_tool_selection(manifest)["mode"] == "select"
    assert set(_require_tool_tags_any(manifest)) == {"fins", "web"}
    assert {"base_agents", "base_fact_rules", "base_tools", "confirm_scene"} <= fragment_ids
    assert "base_directories" not in fragment_ids


@pytest.mark.unit
def test_write_scene_manifest_loads_fact_rules_fragment() -> None:
    """验证 write scene 也加载共享事实与引用规则 fragment。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("write")

    fragment_ids = {fragment["id"] for fragment in manifest["fragments"]}
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-plan"
    assert manifest["model"]["allowed_names"] == _EXPECTED_WRITE_ALLOWED_NAMES
    assert manifest["model"]["temperature_profile"] == "write"
    assert _require_runtime_agent_max_iterations(manifest) == 32
    assert _require_runtime_runner_tool_timeout_seconds(manifest) == 90.0
    assert _require_tool_selection(manifest)["mode"] == "select"
    assert set(_require_tool_tags_any(manifest)) == {"fins", "web"}
    assert {"base_soul", "base_fact_rules", "base_tools"} <= fragment_ids
    assert "base_directories" not in fragment_ids


@pytest.mark.unit
def test_write_scene_only_declares_write_boundary() -> None:
    """验证 write scene 只声明写作任务边界。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    content = store.load_fragment_template("scenes/write.md")

    assert "## 任务目标" in content
    assert "输出结构化章节正文" in content
    assert "缺失但必须保留的事实，按规范保留占位符" in content
    assert "任一事实若无法给出“证据与出处”来源" not in content


@pytest.mark.unit
def test_regenerate_scene_manifest_registers_its_own_contract() -> None:
    """验证 regenerate scene 具备独立 manifest 与 scene 契约。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("regenerate")
    fragment_ids = {fragment["id"] for fragment in manifest["fragments"]}
    content = store.load_fragment_template("scenes/regenerate.md")

    assert manifest["scene"] == "regenerate"
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-plan"
    assert manifest["model"]["allowed_names"] == _EXPECTED_WRITE_ALLOWED_NAMES
    assert manifest["model"]["temperature_profile"] == "write"
    assert _require_runtime_agent_max_iterations(manifest) == 24
    assert _require_runtime_runner_tool_timeout_seconds(manifest) == 90.0
    assert _require_tool_selection(manifest)["mode"] == "select"
    assert set(_require_tool_tags_any(manifest)) == {"fins", "web"}
    assert {"base_agents", "base_tools", "base_soul", "base_fact_rules", "regenerate_scene"} <= fragment_ids
    assert "base_directories" not in fragment_ids
    assert "## 任务目标" in content
    assert "整章重建正文" in content
    assert "不做局部 patch" in content


@pytest.mark.unit
def test_fix_scene_manifest_registers_its_own_tools_and_contract() -> None:
    """验证 fix scene 具备独立 manifest 与 scene 契约。"""

    store = FilePromptAssetStore(ConfigFileResolver())
    manifest = store.load_scene_manifest("fix")
    fragment_ids = {fragment["id"] for fragment in manifest["fragments"]}
    content = store.load_fragment_template("scenes/fix.md")

    assert manifest["scene"] == "fix"
    assert manifest["model"]["default_name"] == "mimo-v2.5-pro-plan"
    assert manifest["model"]["allowed_names"] == _EXPECTED_WRITE_ALLOWED_NAMES
    assert manifest["model"]["temperature_profile"] == "write"
    assert _require_runtime_agent_max_iterations(manifest) == 12
    assert _require_runtime_runner_tool_timeout_seconds(manifest) == 90.0
    assert _require_tool_selection(manifest)["mode"] == "select"
    assert set(_require_tool_tags_any(manifest)) == {"fins", "web"}
    assert {"base_agents", "base_tools", "base_soul", "base_fact_rules", "fix_scene"} <= fragment_ids
    assert "base_directories" not in fragment_ids
    assert "## 任务目标" in content
    assert "仅处理占位符相关问题" in content
    assert "不做完整写作或整章重建" in content
    assert "不输出局部 patch、审计判断或证据复核结论 JSON" in content


@pytest.mark.unit
def test_write_related_manifests_move_dynamic_fragments_after_stable_scene() -> None:
    """验证写作相关 manifests 把动态 slot 从静态 fragments 中剥离。"""

    store = FilePromptAssetStore(ConfigFileResolver())

    expected_fragment_orders = {
        "write": ("base_agents", "base_soul", "base_fact_rules", "base_tools", "write_scene"),
        "regenerate": ("base_agents", "base_soul", "base_fact_rules", "base_tools", "regenerate_scene"),
        "decision": ("base_agents", "base_soul", "base_fact_rules", "base_tools", "decision_scene"),
        "fix": ("base_agents", "base_soul", "base_fact_rules", "base_tools", "fix_scene"),
        "confirm": ("base_agents", "base_fact_rules", "base_tools", "confirm_scene"),
        "overview": ("base_agents", "base_soul", "base_fact_rules", "overview_scene"),
        "audit": ("base_agents", "base_fact_rules", "audit_scene"),
        "repair": ("base_agents", "base_soul", "base_fact_rules", "repair_scene"),
    }

    for scene_name, prefix in expected_fragment_orders.items():
        manifest = store.load_scene_manifest(scene_name)
        fragment_ids = tuple(fragment["id"] for fragment in manifest["fragments"])
        assert fragment_ids[: len(prefix)] == prefix
        assert _require_context_slots(manifest) == ("fins_default_subject", "base_user")


@pytest.mark.unit
def test_manifests_no_longer_declare_tool_call_budget_threshold_context() -> None:
    """验证写作相关 scene 不再声明 tool_call_budget_threshold。"""

    store = FilePromptAssetStore(ConfigFileResolver())

    for scene_name in ("write", "regenerate", "decision", "fix", "confirm", "audit", "repair", "overview", "interactive", "prompt"):
        manifest = store.load_scene_manifest(scene_name)
        for fragment in manifest["fragments"]:
            context_keys = tuple(fragment.get("context_keys", ()))
            assert "tool_call_budget_threshold" not in context_keys


@pytest.mark.unit
def test_base_directories_only_attached_when_scene_really_needs_doc_tools() -> None:
    """验证仅真正需要 doc 工具路径的 scene 才挂载 base_directories。"""

    store = FilePromptAssetStore(ConfigFileResolver())

    for scene_name in ("write", "regenerate", "fix", "decision", "confirm", "overview", "audit", "repair", "interactive", "prompt"):
        manifest = store.load_scene_manifest(scene_name)
        fragment_ids = {fragment["id"] for fragment in manifest["fragments"]}
        assert "base_directories" not in fragment_ids


@pytest.mark.unit
def test_audit_and_repair_still_keep_base_user_slot_for_relative_time_checks() -> None:
    """验证 audit/repair 仍声明 base_user slot，以承载相对时间校验上下文。"""

    store = FilePromptAssetStore(ConfigFileResolver())

    for scene_name in ("audit", "repair"):
        manifest = store.load_scene_manifest(scene_name)
        assert _require_context_slots(manifest) == ("fins_default_subject", "base_user")


@pytest.mark.unit
def test_load_fragment_template_accepts_relative_fragment_path(tmp_path: Path) -> None:
    """验证 fragment 读取按 prompts 根目录解释相对路径。"""

    config_dir = tmp_path / "config"
    fragment_path = config_dir / "prompts" / "base" / "agents.md"
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_text("fragment-body", encoding="utf-8")

    store = FilePromptAssetStore(ConfigFileResolver(config_dir))

    assert store.load_fragment_template("base/agents.md") == "fragment-body"
