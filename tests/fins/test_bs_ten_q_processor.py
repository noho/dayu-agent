"""BsTenQFormProcessor 及改进后 10-Q marker 策略的覆盖率测试。

测试覆盖范围：
- ``_build_ten_q_markers`` Part 锚定 + 两阶段有序选取策略；
- ``_find_all_part_heading_positions`` / ``_select_best_part_i_anchor`` Part 标题锚定逻辑；
- TOC 去噪对 10-Q 的生效验证；
- ``BsTenQFormProcessor`` 的 supports / PARSER_VERSION 等基础属性；
- 注册表路由（10-Q 主路径 → BsTenQFormProcessor，回退 → TenQFormProcessor）。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pytest

from dayu.fins.processors.bs_ten_q_processor import BsTenQFormProcessor
from dayu.fins.processors.sec_form_section_common import _VirtualSection
from dayu.fins.processors.ten_q_processor import TenQFormProcessor
from dayu.fins.processors.ten_q_form_common import (
    _anchor_produces_meaningful_items,
    _build_ten_q_markers,
    _find_all_part_heading_positions,
    _select_best_part_i_anchor,
    expand_ten_q_virtual_sections_content,
)
from dayu.fins.processors.ten_q_form_common import (
    _PART_I_HEADING_PATTERN,
    _PART_II_HEADING_PATTERN,
)
from dayu.fins.processors.ten_q_form_common import _html_flexible_word
from dayu.fins.storage.local_file_source import LocalFileSource


def _make_source(path: Path, *, media_type: Optional[str] = "text/html") -> LocalFileSource:
    """构建本地 Source。

    Args:
        path: 文件路径。
        media_type: 媒体类型。

    Returns:
        LocalFileSource 实例。

    Raises:
        OSError: 文件状态读取失败时抛出。
    """

    return LocalFileSource(
        path=path,
        uri=f"local://{path.name}",
        media_type=media_type,
        content_length=path.stat().st_size,
        etag=None,
    )


def _titles_from_markers(
    markers: list[tuple[int, Optional[str]]],
) -> list[str]:
    """从 marker 列表提取标题。

    Args:
        markers: ``(position, title)`` 列表。

    Returns:
        标题字符串列表。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    return [str(title or "") for _, title in markers]


# ────────────────────────────────────────────────────────────────
# _build_ten_q_markers 单元测试
# ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_build_ten_q_markers_basic_two_part_structure() -> None:
    """验证基础 Part I + Part II 结构正确切分。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Cover page content. "
        "Item 1. Financial Statements for the quarter. "
        + "Detailed financial narrative about quarterly results and revenue trends. " * 20
        + "Item 2. Management Discussion and Analysis. "
        + "MD&A narrative content covering operations and financial conditions. " * 20
        + "Item 3. Quantitative disclosures about market risk. "
        + "Market risk analysis including interest rate and currency exposure. " * 20
        + "Item 4. Controls and Procedures. "
        + "Internal controls evaluation and disclosure controls assessment. " * 20
        + "Item 1. Legal Proceedings. "
        + "Legal proceedings details including pending litigation matters. " * 20
        + "Item 1A. Risk Factors. "
        + "Risk factors update covering operational and financial risks. " * 20
        + "Item 2. Unregistered Sales. "
        + "Sales details about equity securities transactions this quarter. " * 20
        + "Item 6. Exhibits. "
        + "Exhibit listing and index. " * 20
        + "SIGNATURE"
    )
    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    # Part I 的 Items 1-4
    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles
    assert "Part I - Item 3" in titles
    assert "Part I - Item 4" in titles
    # Part II 的 Items
    assert "Part II - Item 1" in titles
    assert "Part II - Item 1A" in titles
    assert "Part II - Item 2" in titles
    assert "Part II - Item 6" in titles
    assert "SIGNATURE" in titles

    # 验证顺序：Part I 的所有 Items 在 Part II 之前
    part_i_positions = [pos for pos, title in markers if title and title.startswith("Part I -")]
    part_ii_positions = [pos for pos, title in markers if title and title.startswith("Part II -")]
    assert all(p1 < p2 for p1 in part_i_positions for p2 in part_ii_positions)


@pytest.mark.unit
def test_build_ten_q_markers_toc_denoising() -> None:
    """验证 TOC 区域中的假 Item 被自适应去噪跳过。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 构造带 TOC 的文本：TOC 中 Item 密集且 span 短
    toc_items = (
        "Table of Contents "
        "Item 1. Financial Statements ....... 5 "
        "Item 2. MD&A ....... 20 "
        "Item 3. Quantitative ....... 35 "
        "Item 4. Controls ....... 40 "
        "Item 1. Legal Proceedings ....... 45 "
        "Item 1A. Risk Factors ....... 50 "
        "Item 2. Sales ....... 55 "
        "Item 3. Defaults ....... 58 "
        "Item 4. Mine Safety ....... 60 "
        "Item 5. Other Information ....... 62 "
        "Item 6. Exhibits ....... 65 "
    )
    # 正文 Items 间距较大（正常文档结构）
    body = (
        "Item 1. Financial Statements for the quarter ended. "
        + "Detailed financials. " * 50
        + "Item 2. Management's Discussion and Analysis of Financial Condition. "
        + "MD&A details. " * 50
        + "Item 3. Quantitative and Qualitative Disclosures About Market Risk. "
        + "Risk details. " * 30
        + "Item 4. Controls and Procedures evaluation. "
        + "Controls assessment. " * 30
        + "Item 1. Legal Proceedings pending against the company. "
        + "Legal details. " * 30
        + "Item 1A. Risk Factors that may affect results. "
        + "Risk factors narrative. " * 30
        + "Item 6. Exhibits submitted herewith. "
        + "Exhibit list. " * 10
        + "SIGNATURE"
    )
    text = toc_items + ("x" * 500) + body

    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    # 验证 Part I Items 来自正文而非 TOC
    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles
    # Part II Items 也来自正文
    assert "Part II - Item 1" in titles or "Part II - Item 1A" in titles
    # 无重复：每个标题最多出现一次
    for title in titles:
        assert titles.count(title) == 1, f"标题 '{title}' 出现了 {titles.count(title)} 次"


@pytest.mark.unit
def test_build_ten_q_markers_repairs_item_1_under_part_i_toc_summary() -> None:
    """验证 Part I 目录摘要污染下仍能恢复 Item 1 正文锚点。

    模拟 AEP 10-Q 场景：
    - 目录中存在 ``Items 1,2,3,4 - Financial Statements, Management's Discussion`` 摘要行；
    - 正文 Item 1 使用 ``Condensed Consolidated Financial Statements`` 标题，
      无显式 ``Item 1`` 前缀；
    - 正文 Item 2 使用 ``Management's Discussion and Analysis`` 标题，
      无显式 ``Item 2`` 前缀。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    toc = (
        "INDEX OF QUARTERLY REPORTS ON FORM 10-Q "
        "PART I. FINANCIAL INFORMATION "
        "Items 1, 2, 3 and 4 - Financial Statements, Management's Discussion and Analysis, "
        "Quantitative and Qualitative Disclosures About Market Risk and Controls and Procedures "
        "Item 3. Defaults Upon Senior Securities 164 "
        "Item 4. Mine Safety Disclosures 164 "
        "PART II. OTHER INFORMATION "
        "Item 5. Other Information 164 "
        "Item 6. Exhibits 165 "
    )
    body = (
        "padding " * 400
        + "PART I. FINANCIAL INFORMATION "
        + "Condensed Consolidated Financial Statements "
        + "Balance sheets, income statements, cash flow statements and footnotes. " * 40
        + "Management's Discussion and Analysis of Financial Condition and Results of Operations "
        + "Executive overview, operating results, liquidity and capital resources. " * 45
        + "Item 3. Quantitative and Qualitative Disclosures About Market Risk "
        + "Risk disclosures and sensitivity analysis. " * 20
        + "Item 4. Controls and Procedures "
        + "Controls conclusions and remediation details. " * 20
        + "PART II. OTHER INFORMATION "
        + "Item 5. Other Information "
        + "Other information details. " * 15
        + "Item 6. Exhibits "
        + "Exhibit list details. " * 10
        + "SIGNATURE"
    )

    markers = _build_ten_q_markers(toc + body)
    titles = _titles_from_markers(markers)

    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles

    marker_map = {str(title): int(pos) for pos, title in markers if title}
    item_1_pos = marker_map["Part I - Item 1"]
    item_2_pos = marker_map["Part I - Item 2"]
    assert item_2_pos - item_1_pos >= 120


@pytest.mark.unit
def test_build_ten_q_markers_repairs_part_i_item3_item4_and_part_ii_item5_item6_without_part_anchors() -> None:
    """验证无 Part 锚点时仍能用正文 heading 修复 Item 3/4/5/6。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    toc = (
        "TABLE OF CONTENTS\n"
        "Disclosure Controls and Procedures 84\n"
        "Other Information 174\n"
        "Exhibit Index 175\n"
    )
    body = (
        "Condensed Consolidated Financial Statements\n"
        + "Financial statements narrative. " * 70
        + "\nManagement's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + "Management analysis narrative. " * 120
        + "\nDisclosure Controls and Procedures\n"
        + "Controls conclusions and remediation updates. " * 55
        + "\nOther Information\n"
        + "Other information narrative. " * 70
        + "\nExhibit Index\n"
        + "Exhibit list narrative. " * 30
        + "\nSIGNATURE"
    )

    markers = _build_ten_q_markers(toc + ("padding " * 200) + body)
    titles = _titles_from_markers(markers)
    marker_map = {str(title): int(pos) for pos, title in markers if title}

    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles
    assert "Part I - Item 4" in titles
    assert "Part II - Item 5" in titles
    assert "Part II - Item 6" in titles
    assert marker_map["Part I - Item 2"] < marker_map["Part I - Item 4"]
    assert marker_map["Part I - Item 4"] < marker_map["Part II - Item 5"]
    assert marker_map["Part II - Item 5"] < marker_map["Part II - Item 6"]


@pytest.mark.unit
def test_build_ten_q_markers_repairs_part_i_item3_and_item4_from_body_headings_after_toc() -> None:
    """验证 TOC 污染下 Part I Item 3/4 会回收到正文真实 heading。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    toc = (
        "PART I. FINANCIAL INFORMATION\n"
        "Item 1. Financial Statements 3\n"
        "Item 2. Management's Discussion and Analysis 25\n"
        "Item 3. Quantitative and Qualitative Disclosures About Market Risk 90\n"
        "Item 4. Controls and Procedures 92\n"
        "PART II. OTHER INFORMATION\n"
        "Item 5. Other Information 120\n"
        "Item 6. Exhibits 121\n"
    )
    body = (
        "PART I. FINANCIAL INFORMATION\n"
        "Item 1. Financial Statements\n"
        + "Financial statements narrative. " * 70
        + "\nItem 2. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + "Management analysis narrative. " * 120
        + "\nQuantitative and Qualitative Disclosures About Market Risk\n"
        + "Market risk narrative. " * 55
        + "\nDisclosure Controls and Procedures\n"
        + "Controls narrative. " * 35
        + "\nPART II. OTHER INFORMATION\n"
        + "\nOther Information\n"
        + "Other information narrative. " * 55
        + "\nExhibit Index\n"
        + "Exhibit list narrative. " * 20
        + "\nSIGNATURE"
    )

    markers = _build_ten_q_markers(toc + ("padding " * 250) + body)
    marker_map = {str(title): int(pos) for pos, title in markers if title}

    assert marker_map["Part I - Item 3"] > marker_map["Part I - Item 2"]
    assert marker_map["Part I - Item 4"] > marker_map["Part I - Item 3"]
    assert marker_map["Part II - Item 5"] > marker_map["Part I - Item 4"]
    assert marker_map["Part II - Item 6"] > marker_map["Part II - Item 5"]


@pytest.mark.unit
def test_build_ten_q_markers_supports_unicode_dash_item_heading() -> None:
    """验证 Item 标题使用 Unicode 破折号时仍可识别。

    场景说明：
    - 部分 10-Q 文档使用 ``Item 3 —`` / ``Item 4 –``（em/en dash）；
    - 该写法符合 SEC 披露习惯，但旧正则仅覆盖 ``.:-``，会导致章节漏检。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "PART I — FINANCIAL INFORMATION "
        "Item 1 — Financial Statements. " + "Financial details. " * 30
        + "Item 2 — Management’s Discussion and Analysis. " + "MD&A details. " * 30
        + "Item 3 — Quantitative and Qualitative Disclosures About Market Risk. "
        + "Risk details. " * 20
        + "Item 4 – Controls and Procedures. " + "Controls details. " * 20
        + "PART II — OTHER INFORMATION "
        + "Item 1 — Legal Proceedings. " + "Legal details. " * 20
        + "Item 2 — Unregistered Sales of Equity Securities and Use of Proceeds. "
        + "Sales details. " * 20
        + "Item 6 — Exhibits. " + "Exhibit details. " * 20
        + "SIGNATURE"
    )

    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles
    assert "Part I - Item 3" in titles
    assert "Part I - Item 4" in titles
    assert "Part II - Item 1" in titles
    assert "Part II - Item 2" in titles
    assert "Part II - Item 6" in titles
    assert "SIGNATURE" in titles


@pytest.mark.unit
def test_build_ten_q_markers_missing_part_i_items() -> None:
    """验证 Part I 缺失部分 Items 时仍能正确切分。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # Part I 只有 Item 1 和 Item 2，缺少 Item 3 和 Item 4
    text = (
        "Item 1. Financial Statements. "
        + "Financial content. " * 20
        + "Item 2. MD&A section. "
        + "MD&A content. " * 20
        + "Item 1. Legal Proceedings. "
        + "Legal content. " * 20
        + "Item 1A. Risk Factors updates. "
        + "Risk content. " * 20
        + "Item 5. Other Information. "
        + "Other info. " * 10
        + "Item 6. Exhibits. "
        + "Exhibit list. " * 10
        + "SIGNATURE"
    )
    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    # Part I 应有 Item 1 和 Item 2
    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles
    # Part II 应有 Item 1, 1A, 5, 6
    assert "Part II - Item 1" in titles
    assert "Part II - Item 1A" in titles
    assert "Part II - Item 5" in titles
    assert "Part II - Item 6" in titles


@pytest.mark.unit
def test_build_ten_q_markers_no_part_i_falls_back() -> None:
    """验证完全没有 Part I Items 时，Part II 仍能正确识别。

    当 Phase 1（Part I）未找到任何 Items 时，Phase 2 使用
    ``_select_ordered_item_markers_after_toc`` 进行带 TOC 去噪的搜索。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 只有 Part II Items：1A, 5, 6（不含可被误识别为 Part I 的 Items 1-4）
    # 在真实 10-Q 中 SEC 要求 Part I 存在，此场景仅作为极端回退测试
    text = (
        "Item 1A. Risk Factors update for the period. "
        + "Risk details covering market conditions and regulatory changes. " * 30
        + "Item 5. Other Information disclosure. "
        + "Other details about material events and corporate actions. " * 30
        + "Item 6. Exhibits filed herewith. "
        + "Exhibit list and description of filed documents. " * 20
        + "SIGNATURE"
    )
    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    # 最少应找到 3+ markers
    assert len(markers) >= 3
    # Phase 1 无 Part I items → Phase 2 的 Part II 应有结果
    assert any("Item 1A" in t for t in titles)
    assert any("Item 5" in t for t in titles)
    assert any("Item 6" in t for t in titles)


@pytest.mark.unit
def test_build_ten_q_markers_returns_empty_when_too_few() -> None:
    """验证 Items 不足 3 个时返回空列表。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = "Item 1. Only one item. Some content about financial statements."
    markers = _build_ten_q_markers(text)
    assert markers == []


@pytest.mark.unit
def test_build_ten_q_markers_part_ii_items_after_part_i() -> None:
    """验证 Part II 的 Item 号码不会与 Part I 的 Items 混淆。

    10-Q 中 Item 1, 2, 3, 4 在 Part I 和 Part II 中都出现，
    两阶段策略确保它们被正确分配到各自的 Part。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 1. Financial Statements data. "
        + "Condensed financials. " * 20
        + "Item 2. Management's Discussion section. "
        + "Discussion narrative. " * 20
        + "Item 3. Quantitative Disclosures details. "
        + "Market risk analysis. " * 20
        + "Item 4. Controls and Procedures assessment. "
        + "Internal controls narrative. " * 20
        + "Item 1. Legal Proceedings against company. "
        + "Legal narrative. " * 20
        + "Item 2. Unregistered Sales details. "
        + "Sales narrative. " * 20
        + "Item 3. Defaults on Senior Securities. "
        + "Default details. " * 10
        + "Item 4. Mine Safety Disclosures information. "
        + "Mine safety details. " * 10
        + "SIGNATURE"
    )
    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    # Part I 应识别 Items 1-4
    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles
    assert "Part I - Item 3" in titles
    assert "Part I - Item 4" in titles
    # Part II 应识别 Items 1-4（重复的 Item 编号）
    assert "Part II - Item 1" in titles
    assert "Part II - Item 2" in titles
    assert "Part II - Item 3" in titles
    assert "Part II - Item 4" in titles


@pytest.mark.unit
def test_build_ten_q_markers_signature_appended() -> None:
    """验证 SIGNATURE 标记被添加到末尾。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "PART I — FINANCIAL INFORMATION "
        "Item 1. Financial Statements. "
        + "Content. " * 20
        + "Item 2. MD&A. "
        + "Content. " * 20
        + "Item 3. Quantitative Disclosures. "
        + "Content. " * 20
        + "PART II — OTHER INFORMATION "
        + "Item 1. Legal Proceedings. "
        + "Content. " * 20
        + "Item 1A. Risk Factors. "
        + "Content. " * 20
        + "Item 6. Exhibits. "
        + "Content. " * 10
        + "SIGNATURE Pursuant to the requirements"
    )
    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)
    assert titles[-1] == "SIGNATURE"


@pytest.mark.unit
def test_expand_ten_q_virtual_sections_repairs_item_2_alias_heading() -> None:
    """验证 Item 2 可从目录别名回收到正文真实标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    full_text = "\n".join(
        [
            "TABLE OF CONTENTS",
            "Item 1. Financial Statements 3",
            "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations (Financial Review) 24",
            "",
            "Financial Review",
            "Overview of quarterly results and liquidity.",
            "Operating performance commentary and balance sheet changes." * 20,
            "Item 3. Quantitative and Qualitative Disclosures About Market Risk",
            "Market risk discussion." * 20,
        ]
    )
    toc_item_2_start = full_text.index(
        "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations (Financial Review) 24"
    )
    real_item_2_start = full_text.index(
        "Financial Review",
        toc_item_2_start + len(
            "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations (Financial Review) 24"
        ),
    )
    item_3_start = full_text.index(
        "Item 3. Quantitative and Qualitative Disclosures About Market Risk"
    )
    virtual_sections = [
        _VirtualSection(
            ref="part-i-item-2",
            title="Part I - Item 2",
            content=full_text[toc_item_2_start:real_item_2_start].strip(),
            preview="",
            table_refs=[],
            start=toc_item_2_start,
            end=real_item_2_start,
        ),
        _VirtualSection(
            ref="part-i-item-3",
            title="Part I - Item 3",
            content=full_text[item_3_start:].strip(),
            preview="",
            table_refs=[],
            start=item_3_start,
            end=len(full_text),
        ),
    ]

    expand_ten_q_virtual_sections_content(
        full_text=full_text,
        virtual_sections=virtual_sections,
    )

    repaired = virtual_sections[0]
    assert repaired.start == real_item_2_start
    assert repaired.end == item_3_start
    assert repaired.content.startswith("Financial Review")


@pytest.mark.unit
def test_expand_ten_q_virtual_sections_shrinks_part_ii_item_2_from_toc_start() -> None:
    """验证 Part II Item 2 会从目录起点回收到正文真实起点。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    full_text = "\n".join(
        [
            "TABLE OF CONTENTS",
            "Item 2. Unregistered Sales of Equity Securities and Use of Proceeds 82",
            "",
            "Other narrative before the real Part II items." * 30,
            "Item 2. Unregistered Sales of Equity Securities and Use of Proceeds",
            "Repurchase activity and proceeds discussion." * 20,
            "Item 3. Defaults Upon Senior Securities",
            "Defaults discussion." * 20,
        ]
    )
    toc_item_2_start = full_text.index(
        "Item 2. Unregistered Sales of Equity Securities and Use of Proceeds 82"
    )
    real_item_2_start = full_text.index(
        "Item 2. Unregistered Sales of Equity Securities and Use of Proceeds",
        toc_item_2_start + len(
            "Item 2. Unregistered Sales of Equity Securities and Use of Proceeds 82"
        ),
    )
    item_3_start = full_text.index("Item 3. Defaults Upon Senior Securities")
    virtual_sections = [
        _VirtualSection(
            ref="part-ii-item-2",
            title="Part II - Item 2",
            content=full_text[toc_item_2_start:item_3_start].strip(),
            preview="",
            table_refs=[],
            start=toc_item_2_start,
            end=item_3_start,
        ),
        _VirtualSection(
            ref="part-ii-item-3",
            title="Part II - Item 3",
            content=full_text[item_3_start:].strip(),
            preview="",
            table_refs=[],
            start=item_3_start,
            end=len(full_text),
        ),
    ]

    expand_ten_q_virtual_sections_content(
        full_text=full_text,
        virtual_sections=virtual_sections,
    )

    repaired = virtual_sections[0]
    assert repaired.start == real_item_2_start
    assert repaired.end == item_3_start
    assert repaired.content.startswith(
        "Item 2. Unregistered Sales of Equity Securities and Use of Proceeds"
    )


@pytest.mark.unit
def test_expand_ten_q_virtual_sections_can_recover_backward_from_by_reference_stub() -> None:
    """验证 by-reference stub 可回收到前文真实正文。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    full_text = "\n".join(
        [
            "PART I. FINANCIAL INFORMATION",
            "ITEM 1. FINANCIAL STATEMENTS",
            "Condensed consolidated statements." * 30,
            "ITEM 2. Management's Discussion and Analysis of Financial Condition and Results of Operations",
            "Operating review and liquidity discussion." * 40,
            "Part II interim materials." * 10,
            "Management's Discussion and Analysis of Financial Condition and Results of Operations - Capital Resources and Liquidity included in the Company's Annual Report on Form 10-K for the year ended December 31, 2024.",
        ]
    )
    real_item_2_start = full_text.index(
        "ITEM 2. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )
    stub_item_2_start = full_text.index(
        "Management's Discussion and Analysis of Financial Condition and Results of Operations - Capital Resources and Liquidity included"
    )
    virtual_sections = [
        _VirtualSection(
            ref="part-i-item-2",
            title="Part I - Item 2",
            content=full_text[stub_item_2_start:].strip(),
            preview="",
            table_refs=[],
            start=stub_item_2_start,
            end=len(full_text),
        ),
    ]

    expand_ten_q_virtual_sections_content(
        full_text=full_text,
        virtual_sections=virtual_sections,
    )

    repaired = virtual_sections[0]
    assert repaired.start == real_item_2_start
    assert repaired.content.startswith(
        "ITEM 2. Management's Discussion and Analysis of Financial Condition and Results of Operations"
    )


# ────────────────────────────────────────────────────────────────
# _find_all_part_heading_positions / _select_best_part_i_anchor 单元测试
# ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_find_all_part_heading_positions_basic_single() -> None:
    """验证 Part 标题锚定基本功能（单次出现）。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Cover page. "
        "PART I — FINANCIAL INFORMATION "
        "Item 1. Financial. "
        "PART II — OTHER INFORMATION "
        "Item 1. Legal. "
    )
    pi_pos, pii_pos = _find_all_part_heading_positions(text)
    assert len(pi_pos) == 1
    assert len(pii_pos) == 1
    assert pi_pos[0] < pii_pos[0]
    assert text[pi_pos[0]:pi_pos[0] + 6] == "PART I"
    assert text[pii_pos[0]:pii_pos[0] + 7] == "PART II"


@pytest.mark.unit
def test_find_all_positions_skips_toc_via_best_anchor() -> None:
    """验证 ToC + 正文时能找到所有 Part 位置，best anchor 选中正文。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        # TOC 中的 Part 条目
        "Table of Contents "
        "PART I. FINANCIAL INFORMATION 3 "
        "Item 1. Financial Statements 3 "
        "Item 2. MD&A 20 "
        "PART II. OTHER INFORMATION 38 "
        "Item 1. Legal Proceedings 38 "
        # 正文 Part 标题（含完整 Items，每项 > 1000 chars）
        + "padding " * 250
        + "PART I. FINANCIAL INFORMATION "
        + "Item 1. Actual Financial Statements for the quarter. "
        + "Condensed balance sheet and income statement data here. " * 25
        + "Item 2. Management Discussion and Analysis of Results. "
        + "Operating results and liquidity discussion for period. " * 25
        + "PART II. OTHER INFORMATION "
        + "Item 1. Actual Legal Proceedings and details. "
        + "Legal proceedings description for the reporting period. " * 10
    )
    pi_pos, pii_pos = _find_all_part_heading_positions(text)
    # 应找到 2 个 Part I 和 2 个 Part II
    assert len(pi_pos) == 2
    assert len(pii_pos) == 2
    # best anchor 应选择正文（后面的）候选
    pii_anchor = pii_pos[-1]
    best_pi = _select_best_part_i_anchor(text, pi_pos, pii_anchor)
    assert best_pi is not None
    assert best_pi > 2000  # 在 ToC 之后


@pytest.mark.unit
def test_find_content_part_boundaries_various_formats() -> None:
    """验证 Part 标题模式匹配多种格式变体。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # SEC 法定 Part 标题的常见格式变体
    variants = [
        "PART I — FINANCIAL INFORMATION",
        "PART I. FINANCIAL INFORMATION",
        "PART I - FINANCIAL INFORMATION",
        "PART I: FINANCIAL INFORMATION",
        "Part I Financial Information",
        "PART I—FINANCIAL INFORMATION",
        "PART I – FINANCIAL INFORMATION",
    ]
    for variant in variants:
        assert _PART_I_HEADING_PATTERN.search(variant), (
            f"Pattern should match: {variant}"
        )

    part_ii_variants = [
        "PART II — OTHER INFORMATION",
        "PART II. OTHER INFORMATION",
        "PART II - OTHER INFORMATION",
        "Part II Other Information",
    ]
    for variant in part_ii_variants:
        assert _PART_II_HEADING_PATTERN.search(variant), (
            f"Pattern should match: {variant}"
        )


@pytest.mark.unit
def test_html_flexible_word_basic() -> None:
    """验证 _html_flexible_word 生成正确的灵活匹配模式。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 基本功能：每个字符之间插入 \s*
    pattern = _html_flexible_word("ABC")
    assert pattern == r"A\s*B\s*C"

    # 单字符
    assert _html_flexible_word("X") == "X"

    # 空字符串应抛异常
    with pytest.raises(ValueError):
        _html_flexible_word("")


@pytest.mark.unit
def test_part_heading_pattern_matches_html_word_break() -> None:
    """验证 Part 标题模式能匹配 BSProcessor HTML 文本提取的断字情况。

    BSProcessor 使用 get_text(separator=" ") 提取文本，
    HTML 元素边界可能将单词拆为多段。例如 MSFT 10-Q 中
    ``<span>FINANCI</span><span>AL INFORMATION</span>``
    提取为 ``FINANCI AL INFORMATION``。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # MSFT 10-Q 实际断字场景：FINANCIAL → FINANCI + AL
    broken_variants_pi = [
        "PART I. FINANCI AL INFORMATION",
        "PART I - FINANCI   AL INFORMATION",  # 多空格
        "PART I. FI NANCIAL INFORMATION",  # 不同断点
        "PART I. FINANCIAL IN FORMATION",  # INFORMATION 断字
        "PART I. FINANCI AL IN FORMATION",  # 两处断字
    ]
    for variant in broken_variants_pi:
        assert _PART_I_HEADING_PATTERN.search(variant), (
            f"Part I pattern should match HTML word-break: {variant}"
        )

    broken_variants_pii = [
        "PART II. OTH ER INFORMATION",
        "PART II - OTHER IN FORMATION",
        "PART II. OT HER INFOR MATION",
    ]
    for variant in broken_variants_pii:
        assert _PART_II_HEADING_PATTERN.search(variant), (
            f"Part II pattern should match HTML word-break: {variant}"
        )


@pytest.mark.unit
def test_part_heading_word_break_anchor_selects_body() -> None:
    """验证断字场景下，锚点能正确选择正文而非 ToC。

    模拟 MSFT 10-Q 真实场景：
    - ToC 中 Part I 标题完整：``PART I. FINANCIAL INFORMATION``
    - 正文中 Part I 标题断字：``PART I. FINANCI AL INFORMATION``
    - 锚点应选择正文中的 Part I（质量验证通过）。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    toc = (
        "INDEX Page "
        "PART I. FINANCIAL INFORMATION "
        "Item 1. Financial Statements 3 "
        "Item 2. MD&A 32 "
        "Item 3. Quantitative 47 "
        "Item 4. Controls 47 "
        "PART II. OTHER INFORMATION "
        "Item 1. Legal 48 "
        "Item 1A. Risk Factors 48 "
        "Item 2. Unregistered Sales 63 "
    )
    body = (
        # 正文 Part I 标题（模拟 BSProcessor 断字）
        "PART I. FINANCI AL INFORMATION "
        "ITEM 1. Financial Statements for the quarter. "
        + "Condensed balance sheet data and income statements. " * 50
        + "ITEM 2. Management Discussion and Analysis. "
        + "Operating results and financial condition analysis. " * 50
        + "ITEM 3. Quantitative and Qualitative Disclosures. "
        + "Interest rate and foreign currency risk discussion. " * 25
        + "ITEM 4. Controls and Procedures. "
        + "Assessment of disclosure controls effectiveness. " * 25
        # 正文 Part II 标题（也可能断字）
        + "PART II. OTH ER INFORMATION "
        + "ITEM 1. Legal Proceedings and regulatory matters. "
        + "Legal details. " * 10
        + "ITEM 1A. Risk Factors and uncertainties. "
        + "Updated risk factors disclosure. " * 10
    )
    text = toc + body

    # _find_all_part_heading_positions 应找到断字版本
    pi_pos, pii_pos = _find_all_part_heading_positions(text)
    assert len(pi_pos) == 2, f"Expected 2 Part I positions, got {len(pi_pos)}"
    assert len(pii_pos) == 2, f"Expected 2 Part II positions, got {len(pii_pos)}"

    # best anchor 应选择正文中的 Part I（第二个匹配）
    pii_anchor = pii_pos[-1]
    best_pi = _select_best_part_i_anchor(text, pi_pos, pii_anchor)
    assert best_pi is not None
    assert best_pi > pi_pos[0], "Best anchor should be body heading, not ToC"

    # 完整 markers 应正确构建
    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)
    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles

    # Part I Item 1 应有足够内容（非 ToC 条目）
    item_1_idx = next(i for i, (_, t) in enumerate(markers) if t == "Part I - Item 1")
    item_1_len = markers[item_1_idx + 1][0] - markers[item_1_idx][0]
    assert item_1_len > 1000, f"Part I Item 1 too short: {item_1_len}"


@pytest.mark.unit
def test_find_all_positions_no_headings() -> None:
    """验证无 Part 标题时返回空列表。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = "Item 1. Financial Statements. Item 2. MD&A. Item 3. Risk."
    pi_positions, pii_positions = _find_all_part_heading_positions(text)
    assert pi_positions == []
    assert pii_positions == []
    # 无候选，应返回 None
    assert _select_best_part_i_anchor(text, [], None) is None


@pytest.mark.unit
def test_select_best_anchor_reversed_order_returns_none() -> None:
    """验证 Part I 在 Part II 之后时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "PART II — OTHER INFORMATION "
        + "padding " * 20
        + "PART I — FINANCIAL INFORMATION "
    )
    pi_positions, pii_positions = _find_all_part_heading_positions(text)
    # Part I 在 Part II 之后，无法作为有效锚点
    pii_anchor = pii_positions[-1] if pii_positions else None
    best_pi = _select_best_part_i_anchor(text, pi_positions, pii_anchor)
    assert best_pi is None


@pytest.mark.unit
def test_build_ten_q_markers_with_part_anchors_realistic_toc() -> None:
    """验证修复核心 bug：TOC 缓冲区过宽导致跳过 Part I 内容。

    模拟真实 AAPL/V 场景：TOC 区域紧邻第一个实际 Item 1，
    ``_TABLE_OF_CONTENTS_CUTOFF_BUFFER_CHARS`` 缓冲区会跳过
    实际内容。Part 锚定机制应正确定位内容区边界。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 构造模拟 AAPL/V 的文本结构
    toc = (
        "Table of Contents "
        "PART I. Financial Information 3 "
        "Item 1. Financial Statements 3 "
        "Item 2. MD&A 20 "
        "Item 3. Quantitative 35 "
        "Item 4. Controls 40 "
        "PART II. Other Information 38 "
        "Item 1. Legal 45 "
        "Item 1A. Risk Factors 50 "
        "Item 2. Sales 55 "
        "Item 3. Defaults 58 "
        "Item 4. Mine Safety 60 "
        "Item 5. Other 62 "
        "Item 6. Exhibits 65 "
    )
    # 注意：TOC 后紧接内容区（gap < 1500 chars）
    body = (
        "PART I — FINANCIAL INFORMATION "
        "Item 1. Financial Statements for the quarter. "
        + "Revenue recognition policies and condensed balance sheet. " * 40
        + "Item 2. Management's Discussion and Analysis. "
        + "Financial condition and results of operations discussion. " * 40
        + "Item 3. Quantitative and Qualitative Disclosures. "
        + "Interest rate risk and foreign currency exposure analysis. " * 20
        + "Item 4. Controls and Procedures Assessment. "
        + "Evaluation of disclosure controls and internal controls. " * 20
        + "PART II — OTHER INFORMATION "
        + "Item 1. Legal Proceedings pending. "
        + "Legal proceedings and regulatory matters disclosure. " * 10
        + "Item 1A. Risk Factors updates. "
        + "Changes to risk factors since annual report filing date. " * 10
        + "Item 2. Unregistered Sales of Securities. "
        + "Share repurchase program and equity transactions details. " * 5
        + "Item 3. Defaults Upon Senior Securities. None. "
        "Item 4. Mine Safety Disclosures. Not applicable. "
        "Item 5. Other Information. None. End marker. "
        "Item 6. Exhibits. Exhibit index and descriptions. "
        + "x" * 100
        + "SIGNATURE"
    )
    text = toc + body

    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    # 关键验证：Part I Items 应来自正文（长内容），而非 TOC 条目
    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles

    # Part I Item 1 内容应为正文内容（>1000 chars）
    item_1_idx = next(i for i, (_, t) in enumerate(markers) if t == "Part I - Item 1")
    item_1_len = markers[item_1_idx + 1][0] - markers[item_1_idx][0]
    assert item_1_len > 1000, f"Part I Item 1 content too short: {item_1_len}"

    # Part I Item 2 内容应为正文内容（>1000 chars）
    item_2_idx = next(i for i, (_, t) in enumerate(markers) if t == "Part I - Item 2")
    item_2_len = markers[item_2_idx + 1][0] - markers[item_2_idx][0]
    assert item_2_len > 1000, f"Part I Item 2 content too short: {item_2_len}"

    # 所有 Part I items 在 Part II items 之前
    part_i_positions = [pos for pos, t in markers if t and t.startswith("Part I -")]
    part_ii_positions = [pos for pos, t in markers if t and t.startswith("Part II -")]
    assert all(p1 < p2 for p1 in part_i_positions for p2 in part_ii_positions)

    # Part II Items 应正确识别（不是被 Phase 1 错误占用的 Part II 内容）
    assert "Part II - Item 1" in titles
    assert "Part II - Item 1A" in titles


@pytest.mark.unit
def test_build_ten_q_markers_part_anchor_end_boundary() -> None:
    """验证 Part II 锚定作为 Phase 1 的上界防止越界。

    当 Part I Items 3/4 缺失时，Phase 1 不应越界到 Part II
    区域抓取 Part II 的 Items 3/4。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "PART I — FINANCIAL INFORMATION "
        "Item 1. Financial Statements. "
        + "Condensed financial statements and quarterly results. " * 30
        + "Item 2. Management's Discussion. "
        + "MD&A narrative and operating results for the quarter. " * 30
        # Part I Items 3/4 不存在（部分公司的 10-Q 确实会省略）
        + "PART II — OTHER INFORMATION "
        + "Item 1. Legal Proceedings. "
        + "Legal details. " * 10
        + "Item 3. Defaults Upon Senior Securities. None. "
        "Item 4. Mine Safety. Not applicable. "
        "Item 6. Exhibits. "
        + "Exhibit list. " * 5
        + "SIGNATURE"
    )
    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    # Part I 只应有 Items 1 和 2
    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles
    assert "Part I - Item 3" not in titles  # 不越界
    assert "Part I - Item 4" not in titles  # 不越界

    # Part II 应正确识别 Items 3 和 4
    assert "Part II - Item 3" in titles
    assert "Part II - Item 4" in titles
    assert "Part II - Item 6" in titles


@pytest.mark.unit
def test_build_ten_q_markers_recovers_item2_with_curly_possessive_before_item1() -> None:
    """验证弯引号 possessive 场景下可恢复 Part I Item 2。

    真实 10-Q 中常见：
    - ``ITEM 2. MANAGEMENT’S DISCUSSION ...``（弯引号）；
    - Item 2 可能出现在 Item 1 之前。

    该场景历史上会因 ``management['']s`` 过窄匹配导致 Item 2 漏检。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Table of Contents "
        "PART I. FINANCIAL INFORMATION "
        "Item 1. Financial Statements 35 "
        "Item 2. Management’s Discussion and Analysis 3 "
        "PART II. OTHER INFORMATION "
        "PART I. FINANCIAL INFORMATION "
        "ITEM 2. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS "
        + "Operating trends and balance sheet changes. " * 60
        + "ITEM 1. FINANCIAL STATEMENTS "
        + "Condensed consolidated financial statements and notes. " * 60
        + "ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK "
        + "Market risk commentary. " * 30
        + "ITEM 4. CONTROLS AND PROCEDURES "
        + "Controls discussion. " * 20
        + "PART II. OTHER INFORMATION "
        + "ITEM 1. LEGAL PROCEEDINGS "
        + "Legal updates. " * 15
        + "ITEM 1A. RISK FACTORS "
        + "Risk updates. " * 15
        + "ITEM 2. UNREGISTERED SALES OF EQUITY SECURITIES "
        + "Issuer purchases. " * 10
        + "ITEM 6. EXHIBITS "
        + "SIGNATURE"
    )

    markers = _build_ten_q_markers(text)
    marker_map = {title: pos for pos, title in markers if title}
    titles = _titles_from_markers(markers)
    body_item_2_heading = (
        "ITEM 2. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION "
        "AND RESULTS OF OPERATIONS "
    )

    assert "Part I - Item 2" in titles
    assert "Part I - Item 1" in titles
    assert marker_map["Part I - Item 2"] == text.index(body_item_2_heading)


@pytest.mark.unit
def test_build_ten_q_markers_matches_management_discussion_without_apostrophe() -> None:
    """验证无 apostrophe 的“Management Discussion”标题仍可识别 Item 2。

    部分 10-Q 的纯编号标题使用 ``2. Management Discussion and Analysis``
    （无 ``'s``）。此场景应映射为 ``Part I - Item 2``。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "PART I. FINANCIAL INFORMATION "
        "\n1. Financial Statements "
        + "Quarterly financial statements details. " * 55
        + "\n2. Management Discussion and Analysis "
        + "Management analysis of operations and liquidity. " * 55
        + "\n3. Quantitative and Qualitative Disclosures About Market Risk "
        + "Risk metrics. " * 20
        + "\n4. Controls and Procedures "
        + "Control assessment. " * 20
        + "PART II. OTHER INFORMATION "
        + "\n1. Legal Proceedings "
        + "Legal details. " * 10
        + "\n1A. Risk Factors "
        + "Risk details. " * 10
        + "\n2. Unregistered Sales of Equity Securities and Use of Proceeds "
        + "Issuer repurchase details. " * 10
        + "\n6. Exhibits "
        + "SIGNATURE"
    )

    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    assert "Part I - Item 1" in titles
    assert "Part I - Item 2" in titles
    assert "Part II - Item 2" in titles


@pytest.mark.unit
def test_build_ten_q_markers_prefers_real_item2_heading_over_cross_reference() -> None:
    """验证 Item 2 交叉引用句不会覆盖真实章节标题。

    场景复现：正文较后位置存在
    ``... to Part I, Item 2: \"Management's Discussion ...\"``。
    该句式不是章节边界，不应作为 ``Part I - Item 2`` marker。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "PART I. FINANCIAL INFORMATION "
        "ITEM 2. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS "
        + "Overview and operating results discussion. " * 50
        + "ITEM 1. FINANCIAL STATEMENTS "
        + "Condensed consolidated financial statements and notes. " * 50
        + "The discussion in 2024 compared to 2023 is incorporated by reference to Part I, Item 2: "
        + "“Management’s Discussion and Analysis of Financial Condition and Results of Operations.” "
        + "Additional note text. " * 5
        + "ITEM 3. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK "
        + "Risk section. " * 20
        + "ITEM 4. CONTROLS AND PROCEDURES "
        + "Control section. " * 20
        + "PART II. OTHER INFORMATION "
        + "ITEM 1. LEGAL PROCEEDINGS "
        + "Legal section. " * 12
        + "ITEM 1A. RISK FACTORS "
        + "Risk factor section. " * 12
        + "ITEM 2. UNREGISTERED SALES OF EQUITY SECURITIES "
        + "Issuer purchases section. " * 8
        + "ITEM 6. EXHIBITS "
        + "SIGNATURE"
    )

    markers = _build_ten_q_markers(text)
    marker_map = {title: pos for pos, title in markers if title}
    item2_pos = marker_map["Part I - Item 2"]

    assert item2_pos < marker_map["Part I - Item 1"]
    assert text[item2_pos : item2_pos + 80].upper().startswith(
        "ITEM 2. MANAGEMENT’S DISCUSSION"
    )


@pytest.mark.unit
def test_anchor_quality_allows_short_items_3_4() -> None:
    """验证锚点质量检查允许 Part I Items 3/4 合法地极短。

    V 公司 10-Q 的真实场景：Item 1（Financial Statements）和
    Item 2（MD&A）内容充足（数万字符），但 Item 3（Quantitative
    Disclosures）仅 152 chars、Item 4（Controls & Procedures）
    仅 923 chars。

    SEC 规则允许 Item 3/4 内容极短（如仅引用 10-K），因此锚点质量
    检查应通过——只要 Item 1 和 Item 2 有实质性内容即可。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 模拟 V 10-Q 实际场景：Items 1/2 大、3/4 极短
    # V @24423: Items 1-4 spans = 60206, 33224, 152, 923
    trial_v_content = [("1", 24453), ("2", 84659), ("3", 117883), ("4", 118035)]
    result = _anchor_produces_meaningful_items(
        "x" * 200000, trial_v_content, 118958,
    )
    assert result is True, (
        "V 实际内容锚点应通过——Item 1(60k) 和 Item 2(33k) 有实质性 span"
    )

    # 对比：V TOC 锚点——只有 1 个有意义的 span（Item 4 到 Part II 距离远）
    trial_v_toc = [("1", 23347), ("2", 23905), ("3", 24002), ("4", 24072)]
    result = _anchor_produces_meaningful_items(
        "x" * 200000, trial_v_toc, 118958,
    )
    assert result is False, (
        "V TOC 锚点应被拒——Items 1/2/3 仅百余字符，只有 Item 4 有大 span"
    )


@pytest.mark.unit
def test_build_markers_short_items_3_4_selects_correct_anchor() -> None:
    """验证 Items 3/4 极短时仍选择正确的 Part I 锚点。

    模拟 V 10-Q 结构：TOC + Part I（Items 1/2 大，3/4 短）+ Part II。
    _build_ten_q_markers 应正确选择 Part I 内容锚点，而非回退到
    _select_ordered_item_markers_after_toc 错误地选中 Part II Items。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    toc = (
        "Table of Contents "
        "PART I. Financial Information 3 "
        "Item 1. Financial Statements 3 "
        "Item 2. MD&A 11 "
        "Item 3. Quantitative 27 "
        "Item 4. Controls 37 "
        "PART II. Other Information 38 "
        "Item 1. Legal 38 "
        "Item 1A. Risk Factors 38 "
        "Item 2. Sales 39 "
        "Item 3. Defaults 39 "
        "Item 4. Mine Safety 39 "
        "Item 5. Other 39 "
        "Item 6. Exhibits 40 "
    )
    body = (
        # Part I: Items 1/2 大、3/4 合法地极短（模拟 V 10-Q）
        "PART I. FINANCIAL INFORMATION "
        "Item 1. Financial Statements (Unaudited) "
        + "VISA CONSOLIDATED BALANCE SHEETS revenue data. " * 80
        + "Item 2. Management's Discussion and Analysis. "
        + "Operating results and net income discussion. " * 80
        # Item 3 极短（仅引用 10-K）
        + "Item 3. Quantitative Disclosures about Market Risk. "
        + "No material changes since 10-K. "
        # Item 4 极短
        + "Item 4. Controls and Procedures. "
        + "Effective as of the quarter ended. "
        # Part II 完整
        + "PART II. OTHER INFORMATION "
        + "Item 1. Legal Proceedings. See Note 13. "
        + "Legal matters detail for the period. " * 5
        + "Item 1A. Risk Factors. See 10-K. "
        + "For discussion see Annual Report. " * 3
        + "Item 2. Unregistered Sales. "
        + "Share repurchase program details. " * 5
        + "Item 3. Defaults Upon Senior Securities. None. "
        + "Item 4. Mine Safety. Not applicable. "
        + "Item 5. Other Information. None. "
        + "Item 6. Exhibits. Exhibit index. " * 5
        + "SIGNATURE"
    )
    text = toc + body

    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    # Part I 应完整包含 Items 1-4
    assert "Part I - Item 1" in titles, "Part I Item 1 应被识别"
    assert "Part I - Item 2" in titles, "Part I Item 2 应被识别"
    assert "Part I - Item 3" in titles, "Part I Item 3 应被识别（尽管极短）"
    assert "Part I - Item 4" in titles, "Part I Item 4 应被识别（尽管极短）"

    # Part II 应正确识别（不被 Phase 1 错误占用）
    assert "Part II - Item 1" in titles, "Part II Item 1 应被识别"
    assert "Part II - Item 1A" in titles, "Part II Item 1A 应被识别"

    # Part I Item 1 应有实质性内容（非 TOC 条目）
    item_1_idx = next(i for i, (_, t) in enumerate(markers) if t == "Part I - Item 1")
    item_1_span = markers[item_1_idx + 1][0] - markers[item_1_idx][0]
    assert item_1_span > 1000, f"Part I Item 1 太短: {item_1_span}"

    # Part I Item 3 可以很短——这是合法的
    item_3_idx = next(i for i, (_, t) in enumerate(markers) if t == "Part I - Item 3")
    item_3_span = markers[item_3_idx + 1][0] - markers[item_3_idx][0]
    assert item_3_span < 200, f"Part I Item 3 应为极短内容: {item_3_span}"


# ────────────────────────────────────────────────────────────────
# BsTenQFormProcessor 单元测试
# ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_bs_ten_q_processor_parser_version() -> None:
    """验证 BsTenQFormProcessor 声明独立 parser_version。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert BsTenQFormProcessor.PARSER_VERSION == "bs_ten_q_processor_v1.0.0"


@pytest.mark.unit
def test_bs_ten_q_processor_supported_forms() -> None:
    """验证 BsTenQFormProcessor 仅支持 10-Q。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert BsTenQFormProcessor._SUPPORTED_FORMS == frozenset({"10-Q"})


@pytest.mark.unit
def test_bs_ten_q_processor_supports_10q(tmp_path: Path) -> None:
    """验证 BsTenQFormProcessor.supports 对 10-Q 返回 True。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10q_test.html"
    source_path.write_text("<html><body>report</body></html>", encoding="utf-8")
    source = _make_source(source_path, media_type="text/html")

    assert BsTenQFormProcessor.supports(source, form_type="10-Q", media_type="text/html") is True


@pytest.mark.unit
def test_bs_ten_q_processor_rejects_10k(tmp_path: Path) -> None:
    """验证 BsTenQFormProcessor.supports 对 10-K 返回 False。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10k_test.html"
    source_path.write_text("<html><body>report</body></html>", encoding="utf-8")
    source = _make_source(source_path, media_type="text/html")

    assert BsTenQFormProcessor.supports(source, form_type="10-K", media_type="text/html") is False


@pytest.mark.unit
def test_bs_ten_q_processor_rejects_20f(tmp_path: Path) -> None:
    """验证 BsTenQFormProcessor.supports 对 20-F 返回 False。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "20f_test.html"
    source_path.write_text("<html><body>report</body></html>", encoding="utf-8")
    source = _make_source(source_path, media_type="text/html")

    assert BsTenQFormProcessor.supports(source, form_type="20-F", media_type="text/html") is False


# ────────────────────────────────────────────────────────────────
# 注册表路由测试
# ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_registry_routes_10q_to_bs_primary() -> None:
    """验证注册表路由 10-Q 到 BsTenQFormProcessor（主路径）。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.engine.processors.source import Source
    from dayu.fins.processors.registry import build_fins_processor_registry

    class _DummySource:
        def __init__(self) -> None:
            self.uri = "local://test.html"
            self.media_type = "text/html"
            self.content_length = None
            self.etag = None

        def open(self):
            raise OSError("dummy")

        def materialize(self, suffix=None):
            raise OSError("dummy")

    registry = build_fins_processor_registry()
    source = _DummySource()
    resolved = registry.resolve(source, form_type="10-Q")
    assert resolved is BsTenQFormProcessor


@pytest.mark.unit
def test_registry_10q_fallback_candidates() -> None:
    """验证 10-Q 回退候选列表包含 BsTenQFormProcessor 和 TenQFormProcessor。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.engine.processors.source import Source
    from dayu.fins.processors.registry import build_fins_processor_registry

    class _DummySource:
        def __init__(self) -> None:
            self.uri = "local://test.html"
            self.media_type = "text/html"
            self.content_length = None
            self.etag = None

        def open(self):
            raise OSError("dummy")

        def materialize(self, suffix=None):
            raise OSError("dummy")

    registry = build_fins_processor_registry()
    source = _DummySource()
    candidates = registry.resolve_candidates(source, form_type="10-Q")

    # 前两个候选：BS 主路径 + Sec 回退
    assert len(candidates) >= 2
    assert candidates[0] is BsTenQFormProcessor
    assert candidates[1] is TenQFormProcessor


# ────────────────────────────────────────────────────────────────
# 紧凑 ToC 场景（HIG 类型）：Part I / Part II 锚点均在 ToC 内
# ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_build_ten_q_markers_compact_toc_both_parts_in_toc() -> None:
    """验证 Part I/II 锚点均在紧凑 ToC 内时，Item 3 不会出现在 Item 1 之前。

    场景复现：HIG 10-Q Q1 2025。
    "Part I. Financial Information" 和 "Part II. Other Information"
    在 ToC 里相距不足 _PART_II_ANCHOR_MAX_TOC_SPREAD（5000 chars），
    修复前：Phase 1 以 ToC 内的 Part II 锚点截断选取范围，导致
    正文 Item 1（在 Part II 锚点之后）不可选，最终 ToC 末尾的
    Item 3 出现在正文 Item 1 之前（顺序颠倒）。
    修复后：检测到 compact ToC，不以此截断，Item 1 正确排在 Item 3 之前。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    # ── 紧凑 ToC（Part I / Part II anchor 间距 ~250 chars < 5000）──
    part_i_heading = "Part I. Financial Information\n"
    toc_items = (
        "1. Financial Statements 6\n"
        "2. Management's Discussion and Analysis 49\n"
        "3. Quantitative and Qualitative Disclosures About Market Risk [a]\n"
        "4. Controls and Procedures 100\n"
    )
    part_ii_heading = "Part II. Other Information\n"
    toc_items_ii = (
        "1. Legal Proceedings 101\n"
        "1A. Risk Factors 101\n"
        "2. Unregistered Sales of Equity Securities 102\n"
        "6. Exhibits 103\n"
    )
    compact_toc = part_i_heading + toc_items + part_ii_heading + toc_items_ii

    # ── 正文（Items 按正确顺序，Item 内容 span 足够大）─────────────────
    # Item 3 在正文中出现在 Item 1 之后（真实顺序）
    body_padding_large = "Quarterly financial data and analysis. " * 150  # ~6000 chars
    body_padding_small = "Brief disclosure content. " * 20  # ~520 chars
    body = (
        "\n\nItem 1. Financial Statements\n"
        + body_padding_large
        + "\nItem 2. Management's Discussion and Analysis of Financial Condition "
        "and Results of Operations\n"
        + body_padding_large
        + "\nItem 3. Quantitative and Qualitative Disclosures About Market Risk\n"
        + body_padding_small
        + "\nItem 4. Controls and Procedures\n"
        + body_padding_small
        + "\nItem 1. Legal Proceedings\n"
        + body_padding_small
        + "\nItem 1A. Risk Factors\n"
        + body_padding_small
        + "\nItem 2. Unregistered Sales of Equity Securities\n"
        + "No sales this quarter.\n"
        + "\nItem 6. Exhibits\n"
        + "Exhibit list.\n"
        + "\nSignature\nPursuant to the requirements of the Securities Exchange Act...\n"
    )
    text = compact_toc + body

    markers = _build_ten_q_markers(text)
    titles = _titles_from_markers(markers)

    # Part I Item 1 和 Item 2 必须存在
    assert "Part I - Item 1" in titles, "Part I Item 1 应被识别"
    assert "Part I - Item 2" in titles, "Part I Item 2 应被识别"

    # 核心断言：Item 1 在 Item 2 之前（顺序正确）
    item1_pos = next(pos for pos, title in markers if title == "Part I - Item 1")
    item2_pos = next(pos for pos, title in markers if title == "Part I - Item 2")
    assert item1_pos < item2_pos, (
        f"Item 1 (pos={item1_pos}) 应在 Item 2 (pos={item2_pos}) 之前"
    )

    # 核心断言（HIG bug 回归验证）：如果 Item 3 存在，不能出现在 Item 1 之前
    if "Part I - Item 3" in titles:
        item3_pos = next(pos for pos, title in markers if title == "Part I - Item 3")
        assert item3_pos > item1_pos, (
            f"Item 3 (pos={item3_pos}) 不能出现在 Item 1 (pos={item1_pos}) 之前——"
            "这是 HIG 10-Q compact ToC 场景的顺序颠倒 bug"
        )

    # Part II Items 应在 Part I Items 之后
    part_i_positions = [pos for pos, title in markers if title and title.startswith("Part I -")]
    part_ii_positions = [pos for pos, title in markers if title and title.startswith("Part II -")]
    if part_ii_positions:
        assert max(part_i_positions) < min(part_ii_positions), "Part I Items 应全部在 Part II 之前"
