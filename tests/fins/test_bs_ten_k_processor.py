"""BsTenKFormProcessor 单元测试。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Optional
from unittest.mock import patch

import pytest

from dayu.fins.processors import bs_ten_k_processor as module
from dayu.fins.processors import ten_k_form_common
from dayu.fins.processors.sec_form_section_common import _VirtualSection
from dayu.fins.processors.ten_k_processor import _build_ten_k_markers


class _SourceStub:
    """最小 Source 桩。"""

    uri = "local://test.html"
    media_type = "text/html"
    content_length = 0
    etag = None

    def open(self) -> BinaryIO:
        """打开资源流。

        Args:
            无。

        Returns:
            无。

        Raises:
            OSError: 固定抛错。
        """

        return BytesIO(b"")

    def materialize(self, suffix: Optional[str] = None) -> Path:
        """物化为本地路径。

        Args:
            suffix: 可选后缀。

        Returns:
            固定路径。

        Raises:
            无。
        """

        del suffix
        return Path("/tmp/placeholder.html")


@pytest.mark.unit
def test_ten_k_toc_heading_check_short_circuits_non_toc_context() -> None:
    """验证非目录语境不会进入昂贵的 ToC 正文探测。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "We discuss management's discussion and analysis of financial condition and results of operations "
        "together with other trends affecting the business.\n"
    )
    position = text.lower().index("management's discussion and analysis")

    with patch.object(
        ten_k_form_common,
        "_has_ten_k_substantive_body_after_heading",
        side_effect=AssertionError("非 ToC 语境不应进入正文探测"),
    ):
        assert ten_k_form_common._looks_like_ten_k_toc_heading_context(
            full_text=text,
            position=position,
            matched_text="Management's Discussion and Analysis",
        ) is False


@pytest.mark.unit
def test_ten_k_toc_heading_check_keeps_long_dot_leader_toc_line() -> None:
    """验证长导点目录行仍会进入完整 ToC 判定。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Management's Discussion and Analysis"
        + ("." * 130)
        + " 9 Risk Factors 24 Quantitative and Qualitative Disclosures About Market Risk 32 "
        + "Financial Statements and Supplementary Data 46\n"
    )
    position = text.index("Management's Discussion and Analysis")

    assert ten_k_form_common._looks_like_ten_k_toc_heading_context(
        full_text=text,
        position=position,
        matched_text="Management's Discussion and Analysis",
    ) is True


@pytest.mark.unit
def test_bs_ten_k_init_and_build_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖 BsTenKFormProcessor 构造与 marker 代理逻辑。"""

    captured: dict[str, Any] = {}

    def _fake_init(self: Any, *, source: Any, form_type: Optional[str] = None, media_type: Optional[str] = None) -> None:
        """替代父类构造函数并记录入参。

        Args:
            self: 实例对象。
            source: 来源对象。
            form_type: 表单类型。
            media_type: 媒体类型。

        Returns:
            无。

        Raises:
            无。
        """

        captured["source"] = source
        captured["form_type"] = form_type
        captured["media_type"] = media_type
        self._virtual_sections = []

    monkeypatch.setattr(module._BaseBsReportFormProcessor, "__init__", _fake_init)
    monkeypatch.setattr(module, "_build_ten_k_markers", lambda text: [(123, "Item 1")])
    monkeypatch.setattr(
        module.BsTenKFormProcessor,
        "_collect_document_text",
        lambda self: "document text",
    )
    monkeypatch.setattr(
        module.BsTenKFormProcessor,
        "_postprocess_virtual_sections",
        lambda self, full_text: captured.setdefault("postprocess_text", full_text),
    )

    processor = module.BsTenKFormProcessor(_SourceStub(), form_type="10-K", media_type="text/html")

    assert captured["form_type"] == "10-K"
    assert captured["media_type"] == "text/html"
    assert captured["postprocess_text"] == "document text"
    assert processor._build_markers("dummy") == [(123, "Item 1")]


@pytest.mark.unit
def test_bs_ten_k_postprocess_virtual_sections_delegates_to_common_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 BS 10-K 后处理钩子委托共享 helper。"""

    captured: dict[str, Any] = {}

    def _fake_expand(*, full_text: str, virtual_sections: list[Any]) -> None:
        """记录共享 helper 的调用入参。

        Args:
            full_text: 完整文本。
            virtual_sections: 虚拟章节列表。

        Returns:
            无。

        Raises:
            无。
        """

        captured["full_text"] = full_text
        captured["virtual_sections"] = virtual_sections

    monkeypatch.setattr(module, "expand_ten_k_virtual_sections_content", _fake_expand)
    monkeypatch.setattr(
        module.BsTenKFormProcessor,
        "_assign_tables_to_virtual_sections",
        lambda self: None,
    )

    processor = object.__new__(module.BsTenKFormProcessor)
    processor._virtual_sections = [
        _VirtualSection(
            ref="s_0001",
            title="Part II - Item 7",
            content="sentinel",
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=[],
            start=0,
            end=8,
        )
    ]
    processor._postprocess_virtual_sections("full text")

    assert captured["full_text"] == "full text"
    assert captured["virtual_sections"] is processor._virtual_sections
    assert processor._virtual_section_by_ref["s_0001"] is processor._virtual_sections[0]


@pytest.mark.unit
def test_expand_ten_k_virtual_sections_content_repairs_by_reference_stub() -> None:
    """验证共享 helper 会把 by-reference stub 扩展为同文档正文。"""

    mdna_body = "Margin expansion and segment profitability. " * 90
    risk_body = "Interest rate risk and liquidity profile. " * 80
    text = (
        "ANNUAL REPORT\n"
        + "\nManagement's Discussion and Analysis\n"
        + mdna_body
        + "\nCorporate Risk Profile\n"
        + risk_body
        + "\nPart II\n"
        + "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + 'Information in response to this Item 7 can be found in the 2023 Annual Report under the heading “Management\'s Discussion and Analysis.” That information is incorporated into this report by reference.\n'
        + "Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n"
        + 'Information in response to this Item 7A can be found in the 2023 Annual Report under the heading “Corporate Risk Profile.” That information is incorporated into this report by reference.\n'
        + "Item 8. Financial Statements and Supplementary Data\n"
    )
    item7_start = text.index("Item 7.")
    item7a_start = text.index("Item 7A.")
    item8_start = text.index("Item 8.")
    sections = [
        _VirtualSection(
            ref="s_0001",
            title="Part II - Item 7",
            content=text[item7_start:item7a_start].strip(),
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=[],
            start=item7_start,
            end=item7a_start,
        ),
        _VirtualSection(
            ref="s_0002",
            title="Part II - Item 7A",
            content=text[item7a_start:item8_start].strip(),
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=[],
            start=item7a_start,
            end=item8_start,
        ),
        _VirtualSection(
            ref="s_0003",
            title="Part II - Item 8",
            content=text[item8_start:].strip(),
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=[],
            start=item8_start,
            end=len(text),
        ),
    ]

    ten_k_form_common.expand_ten_k_virtual_sections_content(
        full_text=text,
        virtual_sections=sections,
    )

    assert "Margin expansion and segment profitability." in sections[0].content
    assert "Interest rate risk and liquidity profile." in sections[1].content


@pytest.mark.unit
def test_expand_ten_k_virtual_sections_content_skips_toc_heading_cluster() -> None:
    """验证共享 helper 会跳过目录页码簇，命中真实 7A 正文标题。"""

    market_risk_body = "Interest rate sensitivity and commodity exposure. " * 90
    text = (
        "TABLE OF CONTENTS "
        "Management's Discussion and Analysis 9 "
        "Risk Factors 24 "
        "Quantitative and Qualitative Disclosures About Market Risks 32 "
        "Financial Statements and Supplementary Data 46 "
        + "\nPart II\n"
        + "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + ("Liquidity and segment commentary. " * 120)
        + "\nQuantitative and Qualitative Disclosures About Market Risks\n"
        + market_risk_body
        + "\nItem 8. Financial Statements and Supplementary Data\n"
    )
    item7_start = text.index("Item 7.")
    actual_7a_start = text.index("\nQuantitative and Qualitative Disclosures About Market Risks\n") + 1
    item8_start = text.index("Item 8.")
    stub_7a_start = text.index("Quantitative and Qualitative Disclosures About Market Risks 32")
    sections = [
        _VirtualSection(
            ref="s_0001",
            title="Part II - Item 7",
            content=text[item7_start:stub_7a_start].strip(),
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=[],
            start=item7_start,
            end=stub_7a_start,
        ),
        _VirtualSection(
            ref="s_0002",
            title="Part II - Item 7A",
            content="Quantitative and Qualitative Disclosures About Market Risks 32",
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=[],
            start=stub_7a_start,
            end=item8_start,
        ),
        _VirtualSection(
            ref="s_0003",
            title="Part II - Item 8",
            content=text[item8_start:].strip(),
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=[],
            start=item8_start,
            end=len(text),
        ),
    ]

    ten_k_form_common.expand_ten_k_virtual_sections_content(
        full_text=text,
        virtual_sections=sections,
    )

    assert sections[1].start == actual_7a_start
    assert "Interest rate sensitivity and commodity exposure." in sections[1].content


@pytest.mark.unit
def test_expand_ten_k_virtual_sections_content_accepts_long_leading_by_reference_stub() -> None:
    """验证共享 helper 可识别开头包装句后跟后续 Item 溢出的长 stub。"""

    mdna_body = "Net interest income and fee growth improved. " * 100
    text = (
        "ANNUAL REPORT\n"
        + "\nFinancial Review\n"
        + mdna_body
        + "\nPart II\n"
        + "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + 'Financial Review.” That information is incorporated into this report by reference.\n'
        + "Item 1A.\nRisk Factors\n"
        + "Additional cross reference text.\n"
        + "Item 8. Financial Statements and Supplementary Data\n"
    )
    annual_report_start = text.index("\nFinancial Review\n") + 1
    item7_start = text.index("Item 7.")
    item8_start = text.index("Item 8.")
    sections = [
        _VirtualSection(
            ref="s_0001",
            title="Part II - Item 7",
            content=text[item7_start:item8_start].strip(),
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=[],
            start=item7_start,
            end=item8_start,
        ),
        _VirtualSection(
            ref="s_0002",
            title="Part II - Item 8",
            content=text[item8_start:].strip(),
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=[],
            start=item8_start,
            end=len(text),
        ),
    ]

    ten_k_form_common.expand_ten_k_virtual_sections_content(
        full_text=text,
        virtual_sections=sections,
    )

    assert sections[0].start == annual_report_start
    assert "Net interest income and fee growth improved." in sections[0].content


@pytest.mark.unit
def test_select_ten_k_by_reference_replacement_start_prefers_later_fallback() -> None:
    """验证 by-reference 场景下优先采用后段 fallback 正文。"""

    selected = ten_k_form_common._select_ten_k_by_reference_replacement_start(
        section_start=1500,
        reference_start=620,
        fallback_start=1820,
    )

    assert selected == 1820


@pytest.mark.unit
def test_expand_ten_k_virtual_sections_content_recovers_missing_item7_from_alias_child() -> None:
    """验证缺失 Item 7 时可从 MD&A 别名子标题恢复顶层章节。"""

    prefix_body = "Risk factor preface. " * 60
    item7_body = "Operating outlook and capital allocation discussion. " * 100
    item8_body = "Financial statement discussion. " * 90
    text = (
        "Part I\n"
        + "Item 1A. Risk Factors\n"
        + prefix_body
        + "\nOverview and Outlook\n"
        + item7_body
        + "\nItem 8. Financial Statements and Supplementary Data\n"
        + item8_body
    )
    item1a_start = text.index("Item 1A.")
    alias_start = text.index("Overview and Outlook")
    item8_start = text.index("Item 8.")
    sections = [
        _VirtualSection(
            ref="s_0001",
            title="Part I - Item 1A",
            content=text[item1a_start:item8_start].strip(),
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=["s_0001_c01"],
            start=item1a_start,
            end=item8_start,
        ),
        _VirtualSection(
            ref="s_0001_c01",
            title="Overview and Outlook",
            content=text[alias_start:item8_start].strip(),
            preview="",
            table_refs=[],
            level=2,
            parent_ref="s_0001",
            child_refs=[],
            start=alias_start,
            end=item8_start,
        ),
        _VirtualSection(
            ref="s_0002",
            title="Part II - Item 8",
            content=text[item8_start:].strip(),
            preview="",
            table_refs=[],
            level=1,
            parent_ref=None,
            child_refs=[],
            start=item8_start,
            end=len(text),
        ),
    ]

    ten_k_form_common.expand_ten_k_virtual_sections_content(
        full_text=text,
        virtual_sections=sections,
    )

    recovered_item7 = next(
        section for section in sections
        if section.level == 1 and section.title == "Part II - Item 7"
    )
    updated_item1a = next(
        section for section in sections
        if section.level == 1 and section.title == "Part I - Item 1A"
    )
    child_section = next(section for section in sections if section.ref == "s_0001_c01")

    assert recovered_item7.start == alias_start
    assert "Operating outlook and capital allocation discussion." in recovered_item7.content
    assert updated_item1a.end == alias_start
    assert child_section.parent_ref == recovered_item7.ref


@pytest.mark.unit
def test_build_ten_k_markers_recovers_item_8_from_consolidated_heading() -> None:
    """验证正文仅出现 Consolidated Financial Statements 时可恢复 Item 8。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    toc = (
        "Table of Contents "
        "Item 1A. Risk Factors 10 "
        "Item 7. Management's Discussion and Analysis 60 "
        "Item 8. Financial Statements and Supplementary Data 120 "
    )
    body = (
        "padding " * 400
        + "\nRisk Factors\n" + ("Risk details. " * 80)
        + "\nManagement's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + ("MD&A details. " * 120)
        + "\nQuantitative and Qualitative Disclosures About Risk\n"
        + ("Risk sensitivity details. " * 60)
        + "\nConsolidated Financial Statements\n"
        + ("Financial statement narrative and note references. " * 120)
        + "\nSIGNATURE"
    )

    markers = _build_ten_k_markers(toc + body)
    titles = [str(title or "") for _, title in markers]

    assert "Part I - Item 1A" in titles
    assert "Part II - Item 7" in titles
    assert "Part II - Item 8" in titles


@pytest.mark.unit
def test_build_ten_k_markers_accepts_inline_toc_prefix_before_real_heading_body() -> None:
    """验证真实标题前带少量目录噪声时，仍能恢复关键 Item。"""

    text = (
        "TABLE OF CONTENTS\n"
        "Item 1A. Risk Factors 20\n"
        "Item 7. Management's Discussion and Analysis 36\n"
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk 62\n"
        "Item 8. Financial Statements and Supplementary Data 78\n"
        + ("preface " * 250)
        + "\n20 Table of Contents Index to Financial Statements\n"
        + "Item 1A. RISK FACTORS\n"
        + ("Risk discussion body. " * 120)
        + "\n35 Table of Contents Index to Financial Statements\n"
        + "Item 7. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS\n"
        + ("Operating overview and liquidity discussion. " * 160)
        + "\nQuantitative and Qualitative Disclosures About Market Risk\n"
        + ("Market sensitivity discussion. " * 80)
        + "\nItem 8. Financial Statements and Supplementary Data\n"
        + ("Financial statement body. " * 160)
        + "\nSIGNATURE"
    )

    markers = _build_ten_k_markers(text)
    marker_map = {str(title or ""): pos for pos, title in markers}

    assert marker_map["Part I - Item 1A"] == text.index("RISK FACTORS")
    assert marker_map["Part II - Item 7"] == text.index(
        "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS"
    )
    assert marker_map["Part II - Item 8"] == text.rindex("Financial Statements and Supplementary Data")


@pytest.mark.unit
def test_build_ten_k_markers_recovers_item7_with_curly_or_missing_apostrophe() -> None:
    """验证 Item 7 可识别 Management’s / Management Discussion 变体。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Table of Contents "
        "Item 1A. Risk Factors 12 "
        "Item 7. Management’s Discussion and Analysis 48 "
        "Item 8. Financial Statements and Supplementary Data 90 "
        + "padding " * 400
        + "\nRisk Factors\n" + ("Risk details. " * 60)
        + "\nManagement’s Discussion and Analysis of Financial Condition and Results of Operations\n"
        + ("Operating overview and segment analysis. " * 100)
        + "\nQuantitative and Qualitative Disclosures About Market Risk\n"
        + ("Risk sensitivity details. " * 40)
        + "\nConsolidated Financial Statements\n"
        + ("Financial statement details. " * 100)
        + "\nSIGNATURE"
    )

    markers = _build_ten_k_markers(text)
    titles = [str(title or "") for _, title in markers]

    assert "Part II - Item 7" in titles
    assert "Part II - Item 8" in titles


@pytest.mark.unit
def test_build_ten_k_markers_avoids_item7_cross_reference_snippet() -> None:
    """验证 Item 7 不应落在“in/to/see Item 7”交叉引用句。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "PART II\n"
        "Item 5. Market for Registrant's Common Equity\n"
        + ("Market details. " * 40)
        + "For additional discussion, see Item 7, Management’s Discussion and Analysis of Financial Condition and Results of Operations.\n"
        + ("Transition text. " * 10)
        + "Item 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS\n"
        + ("Management discussion body. " * 120)
        + "Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n"
        + ("Risk section body. " * 60)
        + "Item 8. Financial Statements and Supplementary Data\n"
        + ("Financial statement body. " * 120)
        + "SIGNATURE"
    )

    markers = _build_ten_k_markers(text)
    marker_map = {str(title or ""): pos for pos, title in markers}
    item7_pos = marker_map["Part II - Item 7"]
    item7_head = text[item7_pos : item7_pos + 80].upper()

    assert item7_head.startswith("ITEM 7. MANAGEMENT’S DISCUSSION")


@pytest.mark.unit
def test_build_ten_k_markers_ignores_notes_table_of_contents_for_cutoff() -> None:
    """晚期财务附注目录不应覆盖正文 Item 搜索起点。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "TABLE OF CONTENTS\n"
        "Item 1A. Risk Factors 15\n"
        "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations 43\n"
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk 52\n"
        "Item 8. Financial Statements and Supplementary Data 58\n"
        + ("preface " * 300)
        + "\nITEM 1A. Risk Factors\n"
        + ("Risk discussion. " * 120)
        + "\nITEM 7. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + ("MD&A body. " * 180)
        + "\nITEM 7A. Quantitative and Qualitative Disclosures About Market Risk\n"
        + ("Market risk body. " * 100)
        + "\nITEM 8. Financial Statements and Supplementary Data\n"
        + ("Financial statement body. " * 160)
        + "\nNOTES TO CONSOLIDATED FINANCIAL STATEMENTS\n"
        + "TABLE OF CONTENTS\n"
        + "1. Description of Business 69\n"
        + "2. Basis of Presentation 70\n"
        + ("Notes body. " * 80)
        + "\nSIGNATURE"
    )

    markers = _build_ten_k_markers(text)
    marker_map = {str(title or ""): pos for pos, title in markers}

    assert marker_map["Part I - Item 1A"] < marker_map["Part II - Item 7"]
    assert marker_map["Part II - Item 7"] < marker_map["Part II - Item 8"]


@pytest.mark.unit
def test_build_ten_k_markers_ignores_bare_financial_statement_phrase_inside_sentence() -> None:
    """正文句子里的 financial statements 短语不应误判为 Item 8 标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "TABLE OF CONTENTS\n"
        "Item 1A. Risk Factors 12\n"
        "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations 48\n"
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk 57\n"
        "Item 8. Financial Statements and Supplementary Data 66\n"
        + ("preface " * 280)
        + "\nPART I\n"
        + "Item 1A. Risk Factors\n"
        + ("Risk details. " * 120)
        + "As of December 31, 2020, we had indebtedness outstanding (see notes to the consolidated financial statements included elsewhere in this Annual Report on Form 10-K).\n"
        + ("More risk discussion. " * 200)
        + "\nPART II\n"
        + "Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations\n"
        + ("MD&A body. " * 120)
        + "\nItem 7A. Quantitative and Qualitative Disclosures About Market Risk\n"
        + ("Market risk body. " * 90)
        + "\nItem 8. Financial Statements and Supplementary Data\n"
        + ("Financial statement body. " * 160)
        + "\nSIGNATURE"
    )

    markers = _build_ten_k_markers(text)
    marker_map = {str(title or ""): pos for pos, title in markers}
    item8_pos = marker_map["Part II - Item 8"]

    assert text[item8_pos : item8_pos + 80].upper().startswith("ITEM 8. FINANCIAL STATEMENTS")


@pytest.mark.unit
def test_build_ten_k_markers_signature_uses_furthest_item_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGNATURE 应锚定最靠后的真实 Item，而不是 token 顺序上的最后一个。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 15. Exhibits and Financial Statement Schedules\n"
        + ("Exhibits body. " * 20)
        + "SIGNATURE\n"
        + ("Spacer " * 20)
        + "Item 1A. Risk Factors\n"
        + ("Risk body. " * 60)
        + "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + ("MD&A body. " * 60)
        + "Item 8. Financial Statements and Supplementary Data\n"
        + ("Financial statement body. " * 80)
    )

    monkeypatch.setattr(
        ten_k_form_common,
        "_select_ordered_item_markers_after_toc",
        lambda *args, **kwargs: [("1A", 220), ("7", 520), ("8", 860), ("15", 0)],
    )
    monkeypatch.setattr(
        ten_k_form_common,
        "_repair_ten_k_key_items_with_heading_fallback",
        lambda full_text, item_markers: item_markers,
    )
    monkeypatch.setattr(ten_k_form_common, "_build_part_markers", lambda full_text: [])

    markers = _build_ten_k_markers(text)
    titles = [str(title or "") for _, title in markers]

    assert "SIGNATURE" not in titles
