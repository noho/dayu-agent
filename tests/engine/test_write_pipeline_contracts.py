"""写作流水线 contract 解析与渲染测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.services.internal.write_pipeline import prompt_contracts as prompt_contracts_module
from dayu.services.internal.write_pipeline.template_parser import TemplateChapter
from dayu.services.internal.write_pipeline.prompt_contracts import (
  PromptInputSpec,
    parse_task_prompt_contract,
    render_task_prompt,
)
from dayu.services.internal.write_pipeline.template_parser import parse_template_layout


def _preferred_lens_texts(chapter: TemplateChapter) -> list[str]:
    """提取章节合同中的 preferred_lens 文本列表。

    Args:
        chapter: 章节对象。

    Returns:
        `preferred_lens` 中的 `lens` 文本列表。

    Raises:
        无。
    """

    return [item.lens for item in chapter.chapter_contract.preferred_lens]


@pytest.mark.unit
def test_parse_template_layout_extracts_chapter_contract_and_item_rule() -> None:
    """验证模板解析会同时提取全文目标、章节目标、骨架、章节合同与局部条件规则。"""

    layout = parse_template_layout(
        """
<!--
REPORT_GOAL
快速重建公司经营全貌，并支持继续研究 / 暂缓 / 放弃的初步判断。
END_REPORT_GOAL
-->

<!--
AUDIENCE_PROFILE
默认读者具备基本买方投资训练，更偏好关键指标与时间序列。
END_AUDIENCE_PROFILE
-->

<!--
COMPANY_FACET_CATALOG
business_model_candidates:
  - 平台互联网
  - 企业软件
constraint_candidates:
  - 监管敏感
  - 高SBC
END_COMPANY_FACET_CATALOG
-->

## 公司是什么生意
<!--
CHAPTER_GOAL
定义这家公司到底是什么生意。
END_CHAPTER_GOAL
-->

<!--
CHAPTER_CONTRACT
narrative_mode: 定义→结构→机制
must_answer:
  - 定义公司生意
must_not_cover:
  - 不分析竞争
required_output_items:
  - 核心产品
preferred_lens:
  default:
    - 先认知入口后财务口径
END_CHAPTER_CONTRACT
-->
### 详细情况
#### 核心产品、平台或关键资产
<!--
ITEM_RULE
mode: conditional
item: 关键资产
when: 仅在必要时写
END_ITEM_RULE
-->
- 
""".strip()
    )

    chapter = layout.chapters[0]

    assert layout.report_goal == "快速重建公司经营全貌，并支持继续研究 / 暂缓 / 放弃的初步判断。"
    assert layout.audience_profile == "默认读者具备基本买方投资训练，更偏好关键指标与时间序列。"
    assert layout.company_facet_catalog == {
        "business_model_candidates": ["平台互联网", "企业软件"],
        "constraint_candidates": ["监管敏感", "高SBC"],
    }
    assert "<!--" not in chapter.skeleton
    assert chapter.chapter_goal == "定义这家公司到底是什么生意。"
    assert chapter.chapter_contract.narrative_mode == "定义→结构→机制"
    assert chapter.chapter_contract.must_answer == ["定义公司生意"]
    assert chapter.chapter_contract.must_not_cover == ["不分析竞争"]
    assert chapter.chapter_contract.required_output_items == ["核心产品"]
    assert _preferred_lens_texts(chapter) == ["先认知入口后财务口径"]
    assert len(chapter.item_rules) == 1
    assert chapter.item_rules[0].mode == "conditional"
    assert chapter.item_rules[0].target_heading == "核心产品、平台或关键资产"
    assert chapter.item_rules[0].item == "关键资产"


@pytest.mark.unit
def test_parse_template_layout_collapses_blank_lines_after_comment_stripping() -> None:
    """验证删除 HTML 注释后，章节骨架不会残留双重空行。"""

    layout = parse_template_layout(
        """
## 财务表现与资本配置
<!--
CHAPTER_CONTRACT
narrative_mode: 数据→变量→现金与资本去向
must_answer:
  - 先看关键财务数据
must_not_cover:
  - 不写流水账
required_output_items:
  - 关键财务数据
preferred_lens:
  default:
    - 先数据后判断
END_CHAPTER_CONTRACT
-->
### 结论要点


<!-- 这段注释不应在 skeleton 中留下空白 -->


#### 先看哪些关键财务数据
- 最能定义财务质量的少数指标
""".strip()
    )

    chapter = layout.chapters[0]

    assert chapter.skeleton == (
        "## 财务表现与资本配置\n\n"
        "### 结论要点\n\n"
        "#### 先看哪些关键财务数据\n"
        "- 最能定义财务质量的少数指标"
    )
    assert "\n\n\n" not in chapter.skeleton


@pytest.mark.unit
def test_parse_template_layout_preserves_intentional_blank_lines_away_from_comments() -> None:
    """验证非注释导致的空行不会被章节骨架提取额外压缩。"""

    layout = parse_template_layout(
        """
## 财务表现与资本配置
### 结论要点


- **关键财务数据与近年走势**
""".strip()
    )

    chapter = layout.chapters[0]

    assert chapter.skeleton == (
        "## 财务表现与资本配置\n"
        "### 结论要点\n\n\n"
        "- **关键财务数据与近年走势**"
    )


@pytest.mark.unit
def test_parse_template_layout_invalid_chapter_contract_fails_fast() -> None:
    """验证章节合同字段缺失时直接失败。"""

    with pytest.raises(ValueError, match="must_not_cover"):
        parse_template_layout(
            """
## 公司是什么生意
<!--
CHAPTER_CONTRACT
narrative_mode: 定义→结构→机制
must_answer:
  - 定义公司生意
required_output_items:
  - 核心产品
preferred_lens:
  default:
    - 先认知入口后财务口径
END_CHAPTER_CONTRACT
-->
### 结论要点
- 
""".strip()
        )


@pytest.mark.unit
def test_parse_template_layout_empty_narrative_mode_fails_fast() -> None:
    """验证 narrative_mode 为空时直接失败。"""

    with pytest.raises(ValueError, match="narrative_mode"):
        parse_template_layout(
            """
## 公司是什么生意
<!--
CHAPTER_CONTRACT
narrative_mode:
must_answer:
  - 定义公司生意
must_not_cover:
  - 不分析竞争
required_output_items:
  - 核心产品
preferred_lens:
  default:
    - 先认知入口后财务口径
END_CHAPTER_CONTRACT
-->
### 结论要点
- 
""".strip()
        )


@pytest.mark.unit
def test_parse_template_layout_accepts_freeform_narrative_mode() -> None:
    """验证解析器允许模板按章节使用自由叙事标签。"""

    layout = parse_template_layout(
        """
## 最近一年关键变化与当前阶段
<!--
CHAPTER_CONTRACT
narrative_mode: 变化→阶段→现在
must_answer:
  - 最近一年发生了什么变化
must_not_cover:
  - 不展开长期竞争格局
required_output_items:
  - 当前阶段判断
preferred_lens:
  default:
    - 先变化后阶段再落到现在
END_CHAPTER_CONTRACT
-->
### 结论要点
- 
""".strip()
    )

    assert layout.chapters[0].chapter_contract.narrative_mode == "变化→阶段→现在"


@pytest.mark.unit
def test_parse_template_layout_rejects_extra_chapter_contract_fields() -> None:
    """验证章节合同默认拒绝超出五字段最小 schema 的额外字段。"""

    with pytest.raises(ValueError, match="未支持字段"):
        parse_template_layout(
            """
## 公司是什么生意
<!--
CHAPTER_CONTRACT
narrative_mode: 定义→结构→机制
must_answer:
  - 定义公司生意
must_not_cover:
  - 不分析竞争
required_output_items:
  - 核心产品
preferred_lens:
  default:
    - 先认知入口后财务口径
optional_output_items:
  - 可选信息
END_CHAPTER_CONTRACT
-->
### 结论要点
- 
""".strip()
        )


@pytest.mark.unit
def test_render_task_prompt_renders_typed_blocks() -> None:
    """验证 task prompt renderer 会按字段类型输出稳定 block。"""

    contract = parse_task_prompt_contract(
        {
            "prompt_name": "write_chapter",
            "version": "v1",
            "inputs": [
                {"name": "chapter", "type": "scalar", "required": True},
                {"name": "report_goal", "type": "scalar", "required": True},
                {"name": "audience_profile", "type": "scalar", "required": True},
                {"name": "chapter_goal", "type": "scalar", "required": True},
                {"name": "skeleton", "type": "markdown_block", "required": True},
                {"name": "chapter_contract", "type": "json_block", "required": True},
                {"name": "item_rules", "type": "json_block", "required": False},
            ],
        },
        task_name="write_chapter",
    )

    rendered = render_task_prompt(
        prompt_template=(
            "章节：{{chapter}}\n"
            "全文目标：{{report_goal}}\n"
            "读者画像：{{audience_profile}}\n"
            "本章回答：{{chapter_goal}}\n"
            "结构：\n{{skeleton_block}}\n"
            "合同：\n{{chapter_contract_block}}\n"
            "条件：\n{{item_rules_block}}\n"
        ),
        prompt_contract=contract,
        prompt_inputs={
            "chapter": "公司是什么生意",
            "report_goal": "支持继续研究 / 暂缓 / 放弃的初步判断",
            "audience_profile": "具备基本买方训练，偏好关键指标与时间序列",
            "chapter_goal": "定义公司生意",
            "skeleton": "## 公司是什么生意\n\n### 结论要点",
            "chapter_contract": {
                "narrative_mode": "定义→结构→机制",
                "must_answer": ["定义公司生意", "说明如何赚钱"],
                "must_not_cover": ["不分析竞争"],
                "required_output_items": ["核心产品"],
                "preferred_lens": {"default": ["先认知入口后财务口径"]},
            },
            "item_rules": [{"mode": "conditional", "target_heading": "核心产品、平台或关键资产", "item": "关键资产", "when": "仅在必要时写"}],
        },
    )

    assert "章节：公司是什么生意" in rendered
    assert "全文目标：支持继续研究 / 暂缓 / 放弃的初步判断" in rendered
    assert "读者画像：具备基本买方训练，偏好关键指标与时间序列" in rendered
    assert "本章回答：定义公司生意" in rendered
    assert "```markdown" in rendered
    assert "```json" in rendered
    assert '"narrative_mode": "定义→结构→机制"' in rendered
    assert '"must_answer"' in rendered
    assert '"default"' in rendered
    assert '"item": "关键资产"' in rendered
    assert "{{" not in rendered


@pytest.mark.unit
def test_parse_template_layout_invalid_item_rule_fails_fast() -> None:
    """验证 ITEM_RULE 非法时直接失败。"""

    with pytest.raises(ValueError, match="ITEM_RULE"):
        parse_template_layout(
            """
## 公司是什么生意
### 详细情况
#### 核心产品
<!--
ITEM_RULE
mode: maybe
item: 关键资产
when: 必要时写
END_ITEM_RULE
-->
- 
""".strip()
        )


@pytest.mark.unit
def test_actual_template_parses_with_expected_top_level_structure() -> None:
    """验证真实模板能够被解析，并具备稳定的顶层结构。"""

    template_path = Path(__file__).resolve().parents[2] / "dayu" / "assets" / "定性分析模板.md"
    layout = parse_template_layout(template_path.read_text(encoding="utf-8"))

    assert layout.report_goal.strip()
    assert layout.audience_profile.strip()
    assert len(layout.chapters) == 11
    assert len({chapter.title for chapter in layout.chapters}) == len(layout.chapters)
    assert layout.chapters[0].title == "投资要点概览"
    assert layout.chapters[-1].title == "是否值得继续深研与待验证问题"
    assert layout.company_facet_catalog["business_model_candidates"]
    assert layout.company_facet_catalog["constraint_candidates"]


@pytest.mark.unit
def test_actual_template_chapters_keep_minimum_contract_structure() -> None:
    """验证真实模板各章节至少保留可解析的目标、骨架与章节合同。"""

    template_path = Path(__file__).resolve().parents[2] / "dayu" / "assets" / "定性分析模板.md"
    layout = parse_template_layout(template_path.read_text(encoding="utf-8"))

    for chapter in layout.chapters:
        assert chapter.chapter_goal.strip()
        assert chapter.skeleton.strip()
        assert chapter.chapter_contract.narrative_mode.strip()
        assert chapter.chapter_contract.must_answer
        assert chapter.chapter_contract.must_not_cover
        assert chapter.chapter_contract.required_output_items
        assert chapter.chapter_contract.preferred_lens


@pytest.mark.unit
def test_actual_template_item_rules_bind_to_visible_headings() -> None:
    """验证真实模板中的 ITEM_RULE 都绑定到对应章节 skeleton 的可见标题。"""

    template_path = Path(__file__).resolve().parents[2] / "dayu" / "assets" / "定性分析模板.md"
    layout = parse_template_layout(template_path.read_text(encoding="utf-8"))

    for chapter in layout.chapters:
        if not chapter.item_rules:
            continue
        for rule in chapter.item_rules:
            assert rule.target_heading in chapter.skeleton


@pytest.mark.unit
def test_render_task_prompt_uses_longer_fence_when_body_contains_backticks() -> None:
    """验证 block 正文含三反引号时会自动提升围栏长度。"""

    contract = parse_task_prompt_contract(
        {
            "prompt_name": "write_chapter",
            "version": "v1",
            "inputs": [
                {"name": "skeleton", "type": "markdown_block", "required": True},
            ],
        },
        task_name="write_chapter",
    )

    rendered = render_task_prompt(
        prompt_template="{{skeleton_block}}",
        prompt_contract=contract,
        prompt_inputs={
            "skeleton": "```markdown\n## 示例\n```",
        },
    )

    assert "````markdown" in rendered


@pytest.mark.unit
def test_prompt_contract_helper_functions_cover_template_names_and_validation_errors() -> None:
    """验证 prompt_contracts helper 的变量名、围栏和错误分支。"""

    assert PromptInputSpec(name="title", input_type="scalar", required=True).template_variable_name == "title"
    assert PromptInputSpec(name="evidence_block", input_type="markdown_block", required=True).template_variable_name == "evidence_block"
    assert PromptInputSpec(name="chapter_contract", input_type="json_block", required=True).template_variable_name == "chapter_contract_block"
    assert prompt_contracts_module._build_fence("") == "```"
    assert prompt_contracts_module._wrap_code_block(body="", language="json") == "```json\n```"
    assert prompt_contracts_module._render_scalar_value(
        spec=PromptInputSpec(name="flag", input_type="scalar", required=True),
        value=True,
    ) == "true"
    assert prompt_contracts_module._render_scalar_value(
        spec=PromptInputSpec(name="empty", input_type="scalar", required=False),
        value=None,
    ) == ""

    with pytest.raises(ValueError, match="prompt_name"):
        parse_task_prompt_contract({"prompt_name": " ", "version": "v1", "inputs": []}, task_name="write_chapter")
    with pytest.raises(ValueError, match="inputs"):
        parse_task_prompt_contract(
            {"prompt_name": "write_chapter", "version": "v1", "inputs": ["bad"]},
            task_name="write_chapter",
        )
    with pytest.raises(ValueError, match="不受支持"):
        parse_task_prompt_contract(
            {
                "prompt_name": "write_chapter",
                "version": "v1",
                "inputs": [{"name": "chapter", "type": "bad_type", "required": True}],
            },
            task_name="write_chapter",
        )
    with pytest.raises(ValueError, match="required 必须为布尔值"):
        parse_task_prompt_contract(
            {
                "prompt_name": "write_chapter",
                "version": "v1",
                "inputs": [{"name": "chapter", "type": "scalar", "required": "yes"}],
            },
            task_name="write_chapter",
        )
    with pytest.raises(ValueError, match="description 必须为字符串"):
        parse_task_prompt_contract(
            {
                "prompt_name": "write_chapter",
                "version": "v1",
                "inputs": [{"name": "chapter", "type": "scalar", "required": True, "description": 1}],
            },
            task_name="write_chapter",
        )


@pytest.mark.unit
def test_render_task_prompt_rejects_unknown_missing_and_bad_input_types() -> None:
    """验证渲染阶段会拒绝未知字段、缺失字段和不匹配的输入类型。"""

    contract = parse_task_prompt_contract(
        {
            "prompt_name": "write_chapter",
            "version": "v1",
            "inputs": [
                {"name": "chapter", "type": "scalar", "required": True},
                {"name": "items", "type": "list_block", "required": False},
                {"name": "mapping", "type": "mapping_block", "required": False},
                {"name": "json_value", "type": "json_block", "required": False},
            ],
        },
        task_name="write_chapter",
    )

    with pytest.raises(ValueError, match="未声明字段"):
        render_task_prompt(
            prompt_template="{{chapter}}",
            prompt_contract=contract,
            prompt_inputs={"chapter": "生意", "unknown": "x"},
        )
    with pytest.raises(ValueError, match="缺少必填字段"):
        render_task_prompt(
            prompt_template="{{chapter}}",
            prompt_contract=contract,
            prompt_inputs={},
        )
    with pytest.raises(ValueError, match="字符串列表"):
        render_task_prompt(
            prompt_template="{{chapter}}\n{{items_block}}",
            prompt_contract=contract,
            prompt_inputs={"chapter": "生意", "items": ["ok", 1]},
        )
    with pytest.raises(ValueError, match="映射"):
        render_task_prompt(
            prompt_template="{{chapter}}\n{{mapping_block}}",
            prompt_contract=contract,
            prompt_inputs={"chapter": "生意", "items": [], "mapping": "bad"},
        )
    with pytest.raises(ValueError, match="无法序列化"):
        render_task_prompt(
            prompt_template="{{chapter}}\n{{json_value_block}}",
            prompt_contract=contract,
            prompt_inputs={"chapter": "生意", "items": [], "mapping": {}, "json_value": {"bad": object()}},
        )
    with pytest.raises(ValueError, match="未替换变量"):
        render_task_prompt(
            prompt_template="{{chapter}}\n{{unknown_block}}",
            prompt_contract=contract,
            prompt_inputs={"chapter": "生意", "items": [], "mapping": {}},
        )
