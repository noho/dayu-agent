"""BsTwentyFFormProcessor 及改进后 20-F marker 策略的覆盖率测试。

测试覆盖范围：
- ``_build_twenty_f_markers`` Part 标签 + Item 描述增强策略；
- ``_build_item_title`` SEC 法定 Part→Item 映射与描述拼装；
- ToC 去噪对 20-F 的生效验证；
- ``BsTwentyFFormProcessor`` 的 supports / PARSER_VERSION 等基础属性；
- 注册表路由（20-F 主路径 → BsTwentyFFormProcessor，回退 → TwentyFFormProcessor）。
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest

from dayu.fins.processors import bs_twenty_f_processor as bs_twenty_f_processor_module
from dayu.fins.processors.bs_twenty_f_processor import (
    BsTwentyFFormProcessor,
    _has_risky_twenty_f_section_profile,
)
from dayu.fins.processors.twenty_f_form_common import (
    _enforce_marker_position_monotonicity,
    _enforce_twenty_f_key_item_priority_monotonicity,
    _extract_twenty_f_cross_reference_guide_snippets,
    _extract_twenty_f_locator_heading_candidates,
    _find_first_valid_twenty_f_heading_position,
    _find_twenty_f_key_heading_position_after,
    _find_twenty_f_locator_heading_position,
    _find_twenty_f_guide_item_spans,
    _looks_like_twenty_f_inline_cross_reference,
    _looks_like_twenty_f_item18_heading_with_body,
    _looks_like_twenty_f_report_suite_cover_marker,
    _looks_like_twenty_f_reference_guide_marker,
    _repair_twenty_f_items_with_cross_reference_guide,
    _repair_twenty_f_item_5_with_subheading_fallback,
    _repair_twenty_f_key_items_with_heading_fallback,
)
from dayu.fins.processors.twenty_f_processor import TwentyFFormProcessor
from dayu.fins.processors.twenty_f_form_common import (
    _build_item_title,
    _build_twenty_f_markers,
    _find_twenty_f_key_heading_positions,
    _TWENTY_F_ITEM_DESCRIPTIONS,
    _TWENTY_F_ITEM_PART_MAP,
)
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


@pytest.mark.unit
def test_has_risky_profile_allows_item18_reference_sentence() -> None:
    """验证 Item 18 的合法引用句不会被误判为目录桩风险。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    sections = [
        SimpleNamespace(
            title="Part IV - Item 18 - Financial Statements",
            content=(
                "ITEM 18. FINANCIAL STATEMENTS\\n"
                "The information required by this item is set forth in our consolidated financial statements "
                "starting on page\\nF-1\\nof this annual report."
            ),
        )
    ]
    assert _has_risky_twenty_f_section_profile(sections) is False


@pytest.mark.unit
def test_has_risky_profile_detects_short_item18_toc_stub() -> None:
    """验证极短 Item 18 目录桩会触发风险回退。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    sections = [
        SimpleNamespace(
            title="Part IV - Item 18 - Financial Statements",
            content="Item 18. Financial Statements 300",
        )
    ]
    assert _has_risky_twenty_f_section_profile(sections) is True


@pytest.mark.unit
def test_has_risky_profile_detects_short_item18_page_range_reference() -> None:
    """验证短 Item 18 页码区间引用会触发风险回退。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    sections = [
        SimpleNamespace(
            title="Part IV - Item 18 - Financial Statements",
            content=(
                "ITEM 18. FINANCIAL STATEMENTS\n"
                "The financial statements filed as part of this Annual Report are included on pages "
                "F-1 through F-86 hereof."
            ),
        )
    ]
    assert _has_risky_twenty_f_section_profile(sections) is True


# ────────────────────────────────────────────────────────────────
# _build_item_title 单元测试
# ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildItemTitle:
    """验证 _build_item_title 的 Part 标签 + 描述拼装。"""

    def test_key_item_with_description(self) -> None:
        """关键 Item 应包含 Part 标签和 SEC 法定描述。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        title = _build_item_title("3")
        assert title == "Part I - Item 3 - Key Information"

    def test_item_5_is_part_ii_with_ofr_description(self) -> None:
        """Item 5（MD&A 等价）应归属 Part II 并附带描述。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        title = _build_item_title("5")
        assert title == "Part II - Item 5 - Operating and Financial Review and Prospects"

    def test_item_18_financial_statements(self) -> None:
        """Item 18（Financial Statements）应归属 Part IV。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        title = _build_item_title("18")
        assert title == "Part IV - Item 18 - Financial Statements"

    def test_item_4a_unresolved_staff_comments(self) -> None:
        """Item 4A 应归属 Part I。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        title = _build_item_title("4A")
        assert title == "Part I - Item 4A - Unresolved Staff Comments"

    def test_governance_item_16a_no_description(self) -> None:
        """治理类 Item 16A 应归属 Part III，无附加描述。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        title = _build_item_title("16A")
        assert title == "Part III - Item 16A"

    def test_all_items_have_part_mapping(self) -> None:
        """所有法定 Item 应有 Part 映射。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        from dayu.fins.processors.twenty_f_form_common import _TWENTY_F_ITEM_ORDER

        for token in _TWENTY_F_ITEM_ORDER:
            assert token in _TWENTY_F_ITEM_PART_MAP, f"Item {token} 缺少 Part 映射"

    def test_case_insensitive_token(self) -> None:
        """token 大小写不敏感。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        assert _build_item_title("4a") == _build_item_title("4A")


# ────────────────────────────────────────────────────────────────
# _build_twenty_f_markers 单元测试
# ────────────────────────────────────────────────────────────────


def _make_twenty_f_body_text(items: list[str], *, padding: int = 200) -> str:
    """构建含多个 Item 的 20-F 模拟正文。

    每个 Item 之间填充足够文本以通过 ToC 检测。

    Args:
        items: Item token 列表（如 ``["3", "4", "5", "18"]``）。
        padding: 每个 Item 后的填充字符数。

    Returns:
        模拟的 20-F 文档正文。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    parts: list[str] = []
    for token in items:
        parts.append(f"Item {token}. ")
        parts.append("X" * padding + " ")
    return "".join(parts)


@pytest.mark.unit
class TestBuildTwentyFMarkers:
    """验证 _build_twenty_f_markers 的切分策略。"""

    def test_basic_item_detection(self) -> None:
        """验证基本的 Item 检测和 Part 标签附加。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = _make_twenty_f_body_text(
            ["1", "2", "3", "4", "5", "6", "7", "8", "18", "19"],
            padding=500,
        )
        markers = _build_twenty_f_markers(text)
        titles = _titles_from_markers(markers)

        # 应包含 Part 前缀
        assert any("Part I" in t for t in titles)
        assert any("Part II" in t for t in titles)
        assert any("Part IV" in t for t in titles)

        # Item 3 应有描述
        assert any("Key Information" in t for t in titles)

        # Item 18 应有描述
        assert any("Financial Statements" in t and "Item 18" in t for t in titles)

    def test_insufficient_items_returns_empty(self) -> None:
        """Item 数量不足时返回空列表触发回退。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = _make_twenty_f_body_text(["1", "3"], padding=500)
        markers = _build_twenty_f_markers(text)
        assert markers == []

    def test_signature_appended(self) -> None:
        """最后一个 Item 之后的 SIGNATURE 应被捕获。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            _make_twenty_f_body_text(
                ["1", "3", "4", "5", "18", "19"],
                padding=500,
            )
            + "SIGNATURE The registrant hereby certifies..."
        )
        markers = _build_twenty_f_markers(text)
        titles = _titles_from_markers(markers)
        assert "SIGNATURE" in titles

    def test_item_heading_with_unicode_dash_is_supported(self) -> None:
        """验证 20-F Item 标题使用 Unicode 破折号可被识别。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Item 1 — Identity of Directors, Senior Management and Advisers. "
            + "X" * 500 + " "
            + "Item 2 — Offer Statistics and Expected Timetable. "
            + "X" * 500 + " "
            + "Item 3 – Key Information. "
            + "X" * 500 + " "
            + "Item 4 — Information on the Company. "
            + "X" * 500 + " "
            + "Item 5 — Operating and Financial Review and Prospects. "
            + "X" * 500 + " "
            + "Item 18 — Financial Statements. "
            + "X" * 500 + " "
            + "Item 19 — Exhibits. "
            + "X" * 500 + " "
            + "SIGNATURE"
        )

        markers = _build_twenty_f_markers(text)
        titles = _titles_from_markers(markers)

        assert any(title.startswith("Part I - Item 1") for title in titles)
        assert any(title.startswith("Part I - Item 3") for title in titles)
        assert any(title.startswith("Part II - Item 5") for title in titles)
        assert any(title.startswith("Part IV - Item 18") for title in titles)
        assert any(title.startswith("Part IV - Item 19") for title in titles)
        assert "SIGNATURE" in titles

    def test_no_ifrs_semantic_markers(self) -> None:
        """v2 策略不再生成 IFRS 语义标题（OFR、Financial Statements 等）。

        IFRS 语义标题与 Item 内容重叠，碎片化结构反而降低 LLM 导航效率。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Item 3. Key Information section. " + "X" * 500 + " "
            "Item 4. Information on the Company. " + "X" * 500 + " "
            "Operating and Financial Review and Prospects discusses trends. " + "X" * 500 + " "
            "Item 5. Operating and Financial Review and Prospects details. " + "X" * 500 + " "
            "Presentation of Financial and Other Information section. " + "X" * 500 + " "
            "Item 18. Financial Statements under IFRS. " + "X" * 500 + " "
            "International Financial Reporting Standards apply. " + "X" * 500 + " "
            "SIGNATURE"
        )
        markers = _build_twenty_f_markers(text)
        titles = _titles_from_markers(markers)

        # 不应有独立 IFRS 语义标题
        assert "Operating and Financial Review and Prospects" not in titles
        assert "Presentation of Financial and Other Information" not in titles
        assert "IFRS Disclosures" not in titles

        # 但 Item 5/18 应正常切分
        assert any("Item 5" in t for t in titles)
        assert any("Item 18" in t for t in titles)

    def test_toc_skipped(self) -> None:
        """Table of Contents 区域应被跳过。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        toc = (
            "TABLE OF CONTENTS\n"
            "Item 1. page 10\n"
            "Item 2. page 15\n"
            "Item 3. page 20\n"
            "Item 4. page 30\n"
            "Item 5. page 40\n"
            "Item 6. page 50\n"
            "Item 7. page 55\n"
            "Item 8. page 60\n"
            "Item 18. page 100\n"
            "Item 19. page 120\n"
        )
        body = _make_twenty_f_body_text(
            ["1", "2", "3", "4", "5", "6", "7", "8", "18", "19"],
            padding=2000,
        )
        text = toc + body
        markers = _build_twenty_f_markers(text)
        titles = _titles_from_markers(markers)

        # 应能成功解析出正文 Items（而非 ToC 条目）
        assert len(markers) >= 8

    def test_item_16_subgroups_detected(self) -> None:
        """Item 16A–16J 治理披露子项应正确检测。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        items = [
            "1", "3", "4", "5", "8",
            "16A", "16B", "16C", "16D", "16E",
            "18", "19",
        ]
        text = _make_twenty_f_body_text(items, padding=500)
        markers = _build_twenty_f_markers(text)
        titles = _titles_from_markers(markers)

        assert any("Item 16A" in t for t in titles)
        assert any("Item 16E" in t for t in titles)
        # 16A–16J 应归属 Part III
        for t in titles:
            if "Item 16" in t and "Item 18" not in t:
                assert "Part III" in t

    def test_item_5_recovered_from_ofr_subheading(self) -> None:
        """无显式 Item 5 时应从 OFR 子标题回退恢复。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Item 3. Key Information. " + "X" * 800 + " "
            "Item 4. Information on the Company. " + "X" * 800 + " "
            "B. Liquidity and Capital Resources " + "X" * 1200 + " "
            "D. Trend Information " + "X" * 1200 + " "
            "Item 10. Additional Information. " + "X" * 800 + " "
            "Item 18. Financial Statements. " + "X" * 800 + " "
            "Item 19. Exhibits. " + "X" * 800 + " "
            "SIGNATURE"
        )

        markers = _build_twenty_f_markers(text)
        titles = _titles_from_markers(markers)
        marker_map = {str(title): int(pos) for pos, title in markers if title}

        assert any("Part II - Item 5" in t for t in titles)
        item_4_pos = next(pos for title, pos in marker_map.items() if "Item 4" in title)
        item_5_pos = next(pos for title, pos in marker_map.items() if "Item 5" in title)
        item_10_pos = next(pos for title, pos in marker_map.items() if "Item 10" in title)
        assert item_4_pos < item_5_pos < item_10_pos

    def test_item_5_fallback_ignores_early_toc_item_10(self) -> None:
        """ToC 早期 Item 10 命中不应阻断 Item 5 子标题回退。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        toc = (
            "TABLE OF CONTENTS "
            "Item 3. page 20 "
            "Item 4. page 35 "
            "Item 10. page 120 "
            "Item 18. page 300 "
        )
        body = (
            "Item 3. Key Information. " + "X" * 800 + " "
            "Item 4. Information on the Company. " + "X" * 800 + " "
            "B. Liquidity and Capital Resources " + "X" * 1200 + " "
            "Item 10. Additional Information. " + "X" * 800 + " "
            "Item 18. Financial Statements. " + "X" * 800 + " "
            "Item 19. Exhibits. " + "X" * 800 + " "
            "SIGNATURE"
        )

        markers = _build_twenty_f_markers(toc + body)
        titles = _titles_from_markers(markers)

        assert any("Part II - Item 5" in title for title in titles)

    def test_front_matter_cross_reference_markers_are_skipped(self) -> None:
        """封面勾选框和 Cross Reference Guide 不应抢占正文 Item。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "If this is an annual report, indicate by check mark which financial statement item "
            "the registrant has elected to follow. Item 17 Item 18 "
            "Form 20-F Cross Reference Guide "
            "Item 1. Not applicable. "
            "Item 2. Not applicable. "
            "Item 3. Response or location in this filing. "
            "Item 4. Response or location in this filing. "
            "Item 5. Response or location in this filing. "
            + ("X" * 2000)
            + " Item 1. Identity of Directors, Senior Management and Advisers. "
            + ("Y" * 1200)
            + " Item 2. Offer Statistics and Expected Timetable. "
            + ("Y" * 1200)
            + " Item 3. Key Information. "
            + ("Y" * 1200)
            + " Item 4. Information on the Company. "
            + ("Y" * 1200)
            + " Item 5. Operating and Financial Review and Prospects. "
            + ("Y" * 1200)
            + " Item 18. Financial Statements. "
            + ("Y" * 1200)
            + " Item 19. Exhibits. "
            + ("Y" * 1200)
            + " SIGNATURE"
        )

        markers = _build_twenty_f_markers(text)
        titles = _titles_from_markers(markers)

        assert titles[0].startswith("Part I - Item 1")
        assert any(title.startswith("Part II - Item 5") for title in titles)
        assert any(title.startswith("Part IV - Item 18") for title in titles)

    def test_split_line_key_item_headings_are_detected(self) -> None:
        """跨换行的 20-F 标题短语也应能被识别。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "3\nKey\nInformation\n"
            + ("alpha details " * 200)
            + "\n4\nInformation on the\nCompany\n"
            + ("beta details " * 200)
            + "\n5\nOperating and Financial Review\nand Prospects\n"
            + ("gamma details " * 200)
            + "\n18\nFinancial\nStatements\n"
            + ("delta details " * 200)
            + "\n19\nExhibits\n"
            + ("epsilon details " * 200)
            + "\nSIGNATURE"
        )

        markers = _build_twenty_f_markers(text)
        titles = _titles_from_markers(markers)

        assert any(title.startswith("Part I - Item 3") for title in titles)
        assert any(title.startswith("Part II - Item 5") for title in titles)
        assert any(title.startswith("Part IV - Item 18") for title in titles)

    def test_single_cover_item18_is_skipped_when_real_body_items_appear_much_later(self) -> None:
        """单个封面 `Item 18` 抢跑时，应继续向后寻找真正正文起点。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "If this is an annual report, indicate by check mark which financial statement item "
            "the registrant has elected to follow. Item 17 Item 18 "
            + ("X" * 25000)
            + " Item 1. Identity of Directors, Senior Management and Advisers. "
            + ("Y" * 1200)
            + " Item 2. Offer Statistics and Expected Timetable. "
            + ("Y" * 1200)
            + " Item 3. Key Information. "
            + ("Y" * 1200)
            + " Item 4. Information on the Company. "
            + ("Y" * 1200)
            + " Item 5. Operating and Financial Review and Prospects. "
            + ("Y" * 1200)
            + " Item 18. Financial Statements. "
            + ("Y" * 1200)
            + " Item 19. Exhibits. "
            + ("Y" * 1200)
            + " SIGNATURE"
        )

        markers = _build_twenty_f_markers(text)
        titles = _titles_from_markers(markers)

        assert titles[0].startswith("Part I - Item 1")
        assert any(title.startswith("Part IV - Item 18") for title in titles)

    def test_reference_guide_financial_statements_prefers_later_real_heading(self) -> None:
        """`Financial Statements` 的 guide 引用不应压过后续真实 Item 18。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Annual Report, Our businesses (18-28), as applicable, "
            "Note 11 to each set of Financial Statements (291 and 410) and "
            "Note 29 to each set of Financial Statements (354 and 475). "
            "See also Supplement (11). "
            + ("X " * 1200)
            + "Item 18. Financial Statements. "
            + ("Y " * 800)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["18"] > 2000

    def test_reference_guide_table_continued_skips_early_item18_locator(self) -> None:
        """跨页 guide 表格中的 `Financial Statements` 不应抢占真实正文锚点。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Form 20-F Cross Reference Guide continued "
            "Item Form 20-F Caption Location in this document Page "
            "Annual Financial Report—Notes to the Consolidated Financial Statements AFR 78-140 "
            "19 Exhibits Exhibits 103-105 "
            "Presentation of Financial and Other Information Financial Information "
            + ("X " * 1200)
            + "Item 4. Information on the Company. "
            + ("Y " * 1200)
            + "Item 18. Financial Statements. "
            + ("Z " * 600)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["4"] > 2000
        assert positions["18"] > positions["4"]

    def test_cross_reference_guide_reconstructs_annual_report_style_markers(self) -> None:
        """guide 型 20-F 应能用年报章节标题重建顶层 Item。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        guide = (
            "Form 20-F caption\n"
            "Location in the document\n"
            "3 Key Information\n"
            "3.D Risk factors | \"Risk factors\" | 79-86\n"
            "4 Information on the company\n"
            "4.B Business overview | \"Strategic Report—Our business model\" | 10-12\n"
            "5 Operating and Financial Review and Prospects\n"
            "5.A Operating results | \"Strategic Report—Financial review\" | 40-55\n"
            "6 Directors, Senior Management and Employees\n"
            "6.A Directors and senior management | \"Governance Report—Board of Directors\" | 90-96\n"
            "18 Financial Statements | \"Financial statements\" | 160-260\n"
            "19 Exhibits | \"Exhibits\" | 300-302\n"
        )
        body = (
            ("preface " * 900)
            + "\nAnnual Report on Form 20-F 2023\n"
            + "Strategic report\n"
            + ("alpha " * 1200)
            + "\nOur business model\n"
            + ("beta " * 1400)
            + "\nFinancial review\n"
            + ("gamma " * 1400)
            + "\nGovernance report\n"
            + ("delta " * 800)
            + "\nBoard of Directors\n"
            + ("epsilon " * 1200)
            + "\nRisk factors\n"
            + ("zeta " * 1200)
            + "\nFinancial statements\n"
            + ("eta " * 1400)
            + "\nExhibits\n"
            + ("theta " * 800)
            + "\nSIGNATURE"
        )

        markers = _build_twenty_f_markers(guide + body)
        marker_map = {str(title): int(position) for position, title in markers if title}

        item_3 = next(position for title, position in marker_map.items() if "Item 3" in title)
        item_4 = next(position for title, position in marker_map.items() if "Item 4" in title)
        item_5 = next(position for title, position in marker_map.items() if "Item 5" in title)
        item_6 = next(position for title, position in marker_map.items() if "Item 6" in title)
        item_18 = next(position for title, position in marker_map.items() if "Item 18" in title)

        assert item_3 < item_4 < item_5 < item_6 < item_18

    def test_cross_reference_table_anchor_reconstructs_annual_report_item18(self) -> None:
        """`cross-reference table` 锚点也应触发 guide repair 并找回 Item 18。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        guide = (
            "In the cross-reference table below, page numbers refer to the Annual Report.\n"
            "Item Form 20-F caption Location in this document\n"
            "3 Key Information | \"Risk factors\" | 79-86\n"
            "5 Operating and Financial Review and Prospects | \"Financial review\" | 40-55\n"
            "18 Financial Statements | \"Financial statements\" | 160-260\n"
            "19 Exhibits | \"Exhibits\" | 300-302\n"
        )
        body = (
            ("opening " * 1000)
            + "\nRisk factors\n"
            + ("alpha " * 1200)
            + "\nFinancial review\n"
            + ("beta " * 1400)
            + "\nFinancial statements\n"
            + ("gamma " * 1500)
            + "\nExhibits\n"
            + ("delta " * 600)
        )

        markers = _build_twenty_f_markers(guide + body + "\nSIGNATURE")
        marker_map = {str(title): int(position) for position, title in markers if title}
        guide_end = len(guide)

        item_18 = next(position for title, position in marker_map.items() if "Item 18" in title)

        assert item_18 > guide_end
        assert item_18 == guide_end + body.index("Financial statements")

    def test_cross_reference_locator_matches_body_heading_split_by_newline(self) -> None:
        """guide locator 回查正文时应允许标题在全文里被换行拆开。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        guide = (
            "In the cross-reference table below, page numbers refer to the Annual Report.\n"
            "Item Form 20-F caption Location in this document\n"
            "3 Key Information | \"Risk factors\" | 79-86\n"
            "5 Operating and Financial Review and Prospects | \"Financial review\" | 40-55\n"
            "18 Financial Statements | \"Financial statements\" | 160-260\n"
        )
        body = (
            ("opening " * 1200)
            + "\nRisk factors\n"
            + ("risk " * 1200)
            + "\nFinancial review\n"
            + ("review " * 1200)
            + "\nFinancial\nstatements\n"
            + ("alpha " * 1500)
        )

        markers = _build_twenty_f_markers(guide + body + "\nSIGNATURE")
        marker_map = {str(title): int(position) for position, title in markers if title}
        item_18 = next(position for title, position in marker_map.items() if "Item 18" in title)
        expected_position = len(guide) + body.index("Financial\nstatements")

        assert item_18 == expected_position

    def test_cross_reference_table_anchor_is_accepted_as_guide_snippet(self) -> None:
        """仅有 `cross-reference table below` 的文本窗口也应被识别为 guide snippet。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "In the cross-reference table below, page numbers refer to the Annual Report.\n"
            "Item Form 20-F caption Location in the document\n"
            "18 Financial Statements | \"Financial statements\" | 160-260\n"
        )

        snippets = _extract_twenty_f_cross_reference_guide_snippets(text)

        assert len(snippets) >= 1
        assert any("cross-reference table below" in snippet for snippet in snippets)

    def test_form_20f_references_anchor_is_accepted_as_guide_snippet(self) -> None:
        """`Form 20-F references` 文本窗口也应被识别为 guide snippet。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Additional information for US listing purposes\n"
            "Form 20-F references\n"
            "Item 3 Key Information 44-50\n"
            "Item 5 Operating and Financial Review and Prospects 32-43\n"
            "Item 18 Financial Statements 106-166\n"
        )

        snippets = _extract_twenty_f_cross_reference_guide_snippets(text)

        assert len(snippets) >= 1
        assert any("Form 20-F references" in snippet for snippet in snippets)

    def test_cross_reference_guide_anchor_is_accepted_as_guide_snippet(self) -> None:
        """`SEC Form 20-F cross reference guide` 也应被识别为 guide snippet。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "SEC Form 20-F cross reference guide\n"
            "Item 3 Key Information 22-31\n"
            "Item 5 Financial performance 32-43\n"
            "Item 18 Financial Statements 251-340\n"
        )

        snippets = _extract_twenty_f_cross_reference_guide_snippets(text)

        assert len(snippets) >= 1
        assert any("cross reference guide" in snippet.lower() for snippet in snippets)

    def test_locator_heading_candidates_split_annual_report_commas(self) -> None:
        """locator 文本中的逗号分隔年报短语应拆出独立正文候选。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        locator_text = (
            "Item 18. Financial Statements. Annual Report, Financial statements (272), "
            "Significant regulated subsidiary and sub-group information (405-407)."
        )

        candidates = _extract_twenty_f_locator_heading_candidates(locator_text)

        assert "Financial statements" in candidates

    def test_tail_cross_reference_guide_prefers_earlier_body_headings(self) -> None:
        """文末 cross-reference guide 不应覆盖前面的真实年报标题锚点。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        body = (
            ("opening " * 1200)
            + "\nStrategic report\n"
            + ("alpha " * 1200)
            + "\nOur business model\n"
            + ("beta " * 1400)
            + "\nFinancial review\n"
            + ("gamma " * 1400)
            + "\nBoard of Directors\n"
            + ("delta " * 1000)
            + "\nRisk factors\n"
            + ("epsilon " * 1200)
            + "\nFinancial statements\n"
            + ("zeta " * 1400)
            + "\nExhibits\n"
            + ("eta " * 800)
            + "\n"
        )
        tail_guide = (
            "Cross-Reference to Form 20-F Continued\n"
            "Item Form 20-F caption Location in this document\n"
            "3 Key Information | \"Risk factors\" | 79-86\n"
            "4 Information on the company | \"Our business model\" | 10-12\n"
            "5 Operating and Financial Review and Prospects | \"Financial review\" | 40-55\n"
            "6 Directors, Senior Management and Employees | \"Board of Directors\" | 90-96\n"
            "18 Financial Statements | \"Financial statements\" | 160-260\n"
            "19 Exhibits | \"Exhibits\" | 300-302\n"
        )
        markers = _build_twenty_f_markers(body + tail_guide + "SIGNATURE")
        marker_map = {str(title): int(position) for position, title in markers if title}
        tail_guide_start = len(body)

        item_4 = next(position for title, position in marker_map.items() if "Item 4" in title)
        item_18 = next(position for title, position in marker_map.items() if "Item 18" in title)

        assert item_4 < tail_guide_start
        assert item_18 < tail_guide_start

    def test_guide_rows_yield_to_later_annual_report_body_headings(self) -> None:
        """guide 行命中后，关键 Item 应优先锚到后面的年报正文标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        guide = (
            "Form 20-F caption\n"
            "Location in this document\n"
            "Item 3 Key Information 5-6\n"
            "Item 4 Information on the Company 7-10\n"
            "Item 5 Operating and Financial Review and Prospects 10-20\n"
            "Item 18 Financial Statements 120-200\n"
            "Item 19 Exhibits 201-202\n"
        )
        body = (
            ("opening " * 1200)
            + "\nPrincipal Risks and Uncertainties\n"
            + ("risk " * 1500)
            + "\nChief Financial Officer’s review\n"
            + ("finance " * 1800)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("audit " * 1600)
            + "\nSIGNATURE"
        )

        markers = _build_twenty_f_markers(guide + body)
        marker_map = {str(title): int(position) for position, title in markers if title}
        guide_end = len(guide)

        item_3 = next(position for title, position in marker_map.items() if "Item 3" in title)
        item_5 = next(position for title, position in marker_map.items() if "Item 5" in title)
        item_18 = next(position for title, position in marker_map.items() if "Item 18" in title)

        assert item_3 > guide_end
        assert item_5 > item_3
        assert item_18 > item_5

    def test_partial_cross_reference_guide_can_merge_existing_key_items(self) -> None:
        """guide 缺少部分关键项时，仍应合并现有 key-item 主链找回 Item 3。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        guide = (
            "Form 20-F caption\n"
            "Location in this document\n"
            "3 Key Information | \"Risk factors\" | 79-86\n"
            "4 Information on the Company | \"Our business model\" | 10-12\n"
            "19 Exhibits | \"Exhibits\" | 300-302\n"
        )
        body = (
            ("opening " * 900)
            + "\nAnnual Report on Form 20-F\n"
            + "\nOur business model\n"
            + ("alpha " * 1200)
            + "\nFinancial review\n"
            + ("beta " * 1300)
            + "\nRisk factors\n"
            + ("gamma " * 1300)
            + "\nItem 18. Financial Statements\n"
            + ("delta " * 1400)
            + "\nExhibits\n"
            + ("theta " * 600)
            + "\nSIGNATURE"
        )

        markers = _build_twenty_f_markers(guide + body)
        marker_map = {str(title): int(position) for position, title in markers if title}

        item_3 = next(position for title, position in marker_map.items() if "Item 3" in title)
        item_5 = next(position for title, position in marker_map.items() if "Item 5" in title)
        item_18 = next(position for title, position in marker_map.items() if "Item 18" in title)

        assert item_3 < item_5 < item_18

    def test_partial_cross_reference_guide_merges_before_key_item_coverage_guard(self) -> None:
        """guide 缺少关键项时，不应在 merge 前被早退短路。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        guide = (
            "Form 20-F caption\n"
            "Location in this document\n"
            "3 Key Information | \"Risk factors\" | 79-86\n"
            "4 Information on the Company | \"Our business model\" | 10-12\n"
            "19 Exhibits | \"Exhibits\" | 300-302\n"
        )
        body = (
            ("opening " * 900)
            + "\nOur business model\n"
            + ("alpha " * 1200)
            + "\nFinancial review\n"
            + ("beta " * 1300)
            + "\nRisk factors\n"
            + ("gamma " * 1300)
            + "\nItem 18. Financial Statements\n"
            + ("delta " * 1400)
            + "\nExhibits\n"
            + ("theta " * 600)
            + "\nSIGNATURE"
        )
        full_text = guide + body
        existing_markers = [
            ("5", full_text.index("Financial review")),
            ("18", full_text.index("Item 18. Financial Statements")),
            ("19", full_text.index("Exhibits")),
        ]

        repaired = _repair_twenty_f_items_with_cross_reference_guide(
            full_text,
            existing_markers,
        )
        repaired_map = {token: position for token, position in repaired}

        assert {"3", "5", "18"}.issubset(repaired_map)
        assert repaired_map["3"] < repaired_map["5"] < repaired_map["18"]

    def test_key_item_priority_monotonicity_preserves_item_18_over_non_key_tail_items(self) -> None:
        """关键 Item 与非关键尾部 Item 冲突时，应优先保留关键 Item。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        repaired = [
            ("3", 0),
            ("4", 100),
            ("5", 200),
            ("9", 900),
            ("10", 1000),
            ("15", 1100),
            ("18", 800),
            ("19", 1200),
        ]

        prioritized = _enforce_twenty_f_key_item_priority_monotonicity(
            repaired=repaired,
            protected_tokens=("3", "5", "18"),
        )
        prioritized_map = {token: position for token, position in prioritized}

        assert {"3", "5", "18"}.issubset(prioritized_map)
        assert prioritized_map["5"] < prioritized_map["18"] < prioritized_map["19"]
        assert "15" not in prioritized_map

    def test_clustered_tail_item_chain_does_not_block_item_18_key_fallback(self) -> None:
        """文末聚簇 Item 链不应否决较早的 Item 18 key-heading fallback。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        body = (
            ("opening " * 900)
            + "\nKey Information\n"
            + ("alpha " * 900)
            + "\nOperating and Financial Review and Prospects\n"
            + ("beta " * 1000)
            + "\nFinancial Statements\n"
            + ("gamma " * 1100)
        )
        tail = (
            "\nItem 1. tail\n"
            "\nItem 2. tail\n"
            "\nItem 3. tail\n"
            "\nItem 4. tail\n"
            "\nItem 4A. tail\n"
            "\nItem 5. tail\n"
            "\nItem 6. tail\n"
            "\nItem 7. tail\n"
            "\nItem 8. tail\n"
            "\nItem 9. tail\n"
            "\nItem 10. tail\n"
            "\nItem 11. tail\n"
            "\nItem 18. tail\n"
            "\nItem 19. tail\n"
        )
        full_text = body + tail
        clustered_markers = [
            ("1", full_text.index("Item 1. tail")),
            ("2", full_text.index("Item 2. tail")),
            ("3", full_text.index("Item 3. tail")),
            ("4", full_text.index("Item 4. tail")),
            ("4A", full_text.index("Item 4A. tail")),
            ("5", full_text.index("Item 5. tail")),
            ("6", full_text.index("Item 6. tail")),
            ("7", full_text.index("Item 7. tail")),
            ("8", full_text.index("Item 8. tail")),
            ("9", full_text.index("Item 9. tail")),
            ("10", full_text.index("Item 10. tail")),
            ("11", full_text.index("Item 11. tail")),
            ("18", full_text.index("Item 18. tail")),
            ("19", full_text.index("Item 19. tail")),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(
            full_text,
            clustered_markers,
        )
        repaired_map = {token: position for token, position in repaired}

        assert {"3", "5", "18"}.issubset(repaired_map)
        assert repaired_map["3"] < repaired_map["5"] < repaired_map["18"]
        assert repaired_map["18"] < clustered_markers[-2][1]

    def test_clustered_tail_item_chain_can_synthesize_item_3_before_item_5(self) -> None:
        """文末聚簇的 Item 1/2/3 不应阻断 synthetic Item 3 的早期起点。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        body = (
            ("opening " * 900)
            + "\nOperating and Financial Review and Prospects\n"
            + ("alpha " * 1000)
            + "\nFinancial Statements\n"
            + ("beta " * 1100)
        )
        tail = (
            "\nItem 1. tail\n"
            "\nItem 2. tail\n"
            "\nItem 3. tail\n"
            "\nItem 4. tail\n"
            "\nItem 4A. tail\n"
            "\nItem 5. tail\n"
            "\nItem 18. tail\n"
            "\nItem 19. tail\n"
        )
        full_text = body + tail
        tail_start = len(body)
        clustered_markers = [
            ("1", full_text.index("Item 1. tail")),
            ("2", full_text.index("Item 2. tail")),
            ("3", full_text.index("Item 3. tail")),
            ("4", full_text.index("Item 4. tail")),
            ("4A", full_text.index("Item 4A. tail")),
            ("5", full_text.index("Item 5. tail")),
            ("18", full_text.index("Item 18. tail")),
            ("19", full_text.index("Item 19. tail")),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(
            full_text,
            clustered_markers,
        )
        repaired_map = {token: position for token, position in repaired}

        assert {"3", "5", "18"}.issubset(repaired_map)
        assert repaired_map["3"] < repaired_map["5"] < repaired_map["18"]
        assert repaired_map["3"] < tail_start

    def test_key_item_repair_keeps_item_5_when_non_key_chain_would_delete_it(self) -> None:
        """key-item repair 不能让非关键项顺序修正再次删掉已恢复的 Item 5。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        full_text = (
            ("preface " * 600)
            + "\nKey Information\n"
            + ("alpha " * 400)
            + "\nInformation on the Company\n"
            + ("beta " * 400)
            + "\nOperating and Financial Review and Prospects\n"
            + ("gamma " * 400)
            + "\nItem 6. Directors, Senior Management and Employees\n"
            + ("delta " * 400)
            + "\nItem 7. Major Shareholders and Related Party Transactions\n"
            + ("epsilon " * 400)
            + "\nItem 8. Financial Information\n"
            + ("zeta " * 400)
        )
        initial = [
            ("1", full_text.index("Information on the Company") + 40),
            ("3", full_text.index("Information on the Company") - 40),
            ("4", full_text.index("Information on the Company")),
            ("6", full_text.index("Item 6.")),
            ("7", full_text.index("Item 7.")),
            ("8", full_text.index("Item 8.")),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(full_text, initial)
        repaired_map = {token: position for token, position in repaired}

        assert repaired_map["5"] < repaired_map["6"]

    def test_key_item_repair_keeps_item_18_when_non_key_tail_would_delete_it(self) -> None:
        """key-item repair 不能让非关键尾标顺序修正再次删掉已恢复的 Item 18。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        full_text = (
            ("preface " * 500)
            + "\nKey Information\n"
            + ("alpha " * 300)
            + "\nInformation on the Company\n"
            + ("beta " * 300)
            + "\nOperating and Financial Review and Prospects\n"
            + ("gamma " * 300)
            + "\nItem 9. The Offer and Listing\n"
            + ("delta " * 250)
            + "\nItem 10. Additional Information\n"
            + ("epsilon " * 250)
            + "\nFinancial Statements\n"
            + ("zeta " * 300)
            + "\nItem 19. Exhibits\n"
            + ("eta " * 200)
        )
        initial = [
            ("1", full_text.index("Information on the Company") + 40),
            ("3", full_text.index("Information on the Company") - 40),
            ("4", full_text.index("Information on the Company")),
            ("5", full_text.index("Operating and Financial Review and Prospects")),
            ("9", full_text.index("Item 9.")),
            ("10", full_text.index("Item 10.")),
            ("19", full_text.index("Item 19.")),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(full_text, initial)
        repaired_map = {token: position for token, position in repaired}

        assert repaired_map["5"] < repaired_map["18"] < repaired_map["19"]

    def test_item_5_accepts_kpi_heading_before_tail_guide(self) -> None:
        """Item 5 应允许用 KPI 标题作为年报正文锚点。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        body = (
            ("opening " * 1000)
            + "\nPrincipal risks and uncertainties\n"
            + ("alpha " * 1200)
            + "\nKey performance indicators\n"
            + ("beta " * 1400)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 1200)
            + "\nSIGNATURE"
        )
        tail_guide = (
            "Form 20-F caption\n"
            "Location in this document\n"
            "3 Key Information | 41-49\n"
            "5 Operating and Financial Review and Prospects | 50-77\n"
            "18 Financial Statements | 137-220\n"
            "19 Exhibits | 221-224\n"
        )

        markers = _build_twenty_f_markers(body + tail_guide)
        marker_map = {str(title): int(position) for position, title in markers if title}
        tail_guide_start = len(body)
        item_5 = next(position for title, position in marker_map.items() if "Item 5" in title)

        assert item_5 < tail_guide_start

    def test_no_guide_annual_report_page_headings_recover_key_items(self) -> None:
        """无显式 guide 时，年报页标题 + 页码 + 正文也应恢复 20-F 关键 Item。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        contents = (
            "Inside This Report\n"
            "For Shareholders and Investors Financial Performance Summary 98\n"
            "Treasury and Cash Flow 104\n"
            "Business Environment Group Principal Risks 116\n"
            "Financial Statements Group Financial Statements Group Companies and Undertakings 301\n"
        )
        body = (
            ("opening " * 800)
            + "\nBusiness Environment Group Principal Risks 116\n"
            + ((
                "These principal risks reflect the main operational, regulatory and market "
                "uncertainties discussed in detail across the annual report. "
            ) * 16)
            + "\nFor Shareholders and Investors Financial Performance Summary 98\n"
            + ((
                "This section reviews revenue, profitability, cash generation and other key "
                "drivers of the Group's financial performance across the reporting period. "
            ) * 16)
            + "\nFinancial Statements Group Financial Statements Group Companies and Undertakings 301\n"
            + ((
                "The consolidated financial statements of the Group, together with supporting "
                "notes and undertaking disclosures, are set out in the following pages. "
            ) * 18)
            + "\nSIGNATURE"
        )
        text = contents + body

        positions = _find_twenty_f_key_heading_positions(text)
        expected_item_3 = text.index(
            "Group Principal Risks",
            len(contents),
        )
        expected_item_5 = text.index(
            "Financial Performance Summary",
            len(contents),
        )
        expected_item_18 = text.index(
            "Financial Statements Group Financial Statements Group Companies and Undertakings 301",
            len(contents),
        )

        assert positions["3"] == expected_item_3
        assert positions["5"] == expected_item_5
        assert positions["18"] == expected_item_18
        assert positions["3"] < positions["5"] < positions["18"]

        markers = _build_twenty_f_markers(text)
        marker_map = {str(title): int(position) for position, title in markers if title}
        item_3 = next(position for title, position in marker_map.items() if "Item 3" in title)
        item_5 = next(position for title, position in marker_map.items() if "Item 5" in title)
        item_18 = next(position for title, position in marker_map.items() if "Item 18" in title)

        assert item_3 == expected_item_3
        assert item_5 == expected_item_5
        assert item_18 == expected_item_18

    def test_key_heading_fallback_synthesizes_early_item3_when_risk_factors_is_late(self) -> None:
        """Item 3 若只命中晚期 Risk factors，应回退到更早正文起点。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 800)
            + "\nStrategic report\n"
            + ("alpha " * 900)
            + "\nKey performance indicators\n"
            + ("beta " * 900)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 900)
            + "\nRisk factors\n"
            + ("delta " * 900)
        )
        tail_markers = [
            ("3", len(text) - 800),
            ("4", len(text) - 600),
            ("5", len(text) - 500),
            ("10", len(text) - 400),
            ("15", len(text) - 300),
            ("17", len(text) - 200),
            ("18", len(text) - 150),
            ("19", len(text) - 100),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(text, tail_markers)
        repaired_map = dict(repaired)

        assert repaired_map["3"] < repaired_map["5"] < repaired_map["18"]
        assert repaired_map["3"] < text.index("Key performance indicators")

    def test_item_5_subheading_fallback_replaces_late_kpi_hit_after_item18(self) -> None:
        """若 Item 5 误命中晚期 KPI 且晚于 Item 18，应改锚到更早经营结果标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + "\nRisk factors\n"
            + ("alpha " * 600)
            + "\nOperating results of 2021 compared to 2020\n"
            + ("beta " * 800)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 800)
            + "\nNon-financial indicators\nKey Performance Indicators (KPIs)\n"
            + ("delta " * 800)
        )
        marker_map = {
            "3": text.index("Risk factors"),
            "5": text.index("Key Performance Indicators"),
            "18": text.index("Report of Independent Registered Public Accounting Firm"),
        }

        repaired = _repair_twenty_f_item_5_with_subheading_fallback(
            full_text=text,
            marker_map=marker_map,
        )

        assert repaired["3"] < repaired["5"] < repaired["18"]
        assert repaired["5"] == text.index("Operating results of 2021 compared to 2020")

    def test_key_heading_fallback_keeps_item5_subheading_repair_when_key_heading_map_empty(self) -> None:
        """即使 key-heading fallback 全部失效，仍应继续执行 Item 5 子标题修复。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 500)
            + "\nLiquidity and capital resources\n"
            + ("alpha " * 700)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("beta " * 700)
        )
        markers = [
            ("18", text.index("Report of Independent Registered Public Accounting Firm")),
            ("19", len(text) - 100),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(text, markers)
        repaired_map = dict(repaired)

        assert repaired_map["5"] == text.index("Liquidity and capital resources")
        assert repaired_map["5"] < repaired_map["18"]

    def test_key_heading_fallback_prefers_complete_monotonic_chain_over_late_markers(self) -> None:
        """完整单调 key-item fallback 应压过更晚的伪 marker。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Form 20-F caption\n"
            "Location in this document\n"
            "Item 1 Not applicable\n"
            + ("front matter " * 160)
            + "\nKey Information\n"
            + ("alpha " * 260)
            + "\nInformation on the Company\n"
            + ("beta " * 260)
            + "\nOperating and Financial Review and Prospects\n"
            + ("gamma " * 260)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("delta " * 260)
            + "\nForm 20-F caption\n"
            + "Location in this document\n"
            + "Item 2. Offer Statistics and Expected Timetable\n"
            + "Not applicable\n"
            + ("epsilon " * 120)
            + "\nSee \"Item 3. Key Information—D. Risk Factors\" for additional discussion.\n"
            + ("zeta " * 120)
            + "\nThe schedule appears in the section entitled \"Item 4. Information on the Company—B. Business Overview\".\n"
            + ("eta " * 120)
            + "\nWe refer you to \"Item 5. Operating and Financial Review and Prospects—B. Liquidity and Capital Resources\".\n"
            + ("theta " * 120)
        )
        markers = [
            ("1", text.index("Item 1 Not applicable")),
            ("3", text.index("Item 3. Key Information—D. Risk Factors")),
            ("4", text.index("Item 4. Information on the Company—B. Business Overview")),
            ("5", text.index("Item 5. Operating and Financial Review and Prospects—B. Liquidity and Capital Resources")),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(text, markers)
        repaired_map = dict(repaired)

        assert repaired_map["3"] == text.index("Key Information")
        assert repaired_map["4"] == text.index("Information on the Company")
        assert repaired_map["5"] == text.index("Operating and Financial Review and Prospects")
        assert repaired_map["18"] == text.index(
            "Report of Independent Registered Public Accounting Firm"
        )
        assert repaired_map["3"] < repaired_map["4"] < repaired_map["5"] < repaired_map["18"]

    def test_key_heading_fallback_keeps_clean_item5_when_only_prior_markers_are_contaminated(self) -> None:
        """若前序仅有 guide 污染 marker，干净 Item 5 fallback 不应在最终顺序修正里丢失。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Financial Review\n"
            + ("alpha " * 260)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("beta " * 260)
            + "\nForm 20-F caption\n"
            + "Location in this document\n"
            + "Item 1 Identity of Directors, Senior Management and Advisers n/a\n"
            + "Item 2 Offer Statistics and Expected Timetable n/a\n"
            + "Item 3 Key Information n/a\n"
            + "Item 4 Information on the Company 10-20\n"
            + "Item 4A Unresolved Staff Comments None\n"
            + "Item 5 Operating and Financial Review and Prospects 30-40\n"
            + "Item 6 Directors, Senior Management and Employees 41-50\n"
            + ("gamma " * 200)
        )
        markers = [
            ("1", text.index("Item 1 Identity of Directors, Senior Management and Advisers n/a")),
            ("2", text.index("Item 2 Offer Statistics and Expected Timetable n/a")),
            ("3", text.index("Item 3 Key Information n/a")),
            ("4", text.index("Item 4 Information on the Company 10-20")),
            ("4A", text.index("Item 4A Unresolved Staff Comments None")),
            ("5", text.index("Item 5 Operating and Financial Review and Prospects 30-40")),
            ("6", text.index("Item 6 Directors, Senior Management and Employees 41-50")),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(text, markers)
        repaired_map = dict(repaired)

        assert repaired_map["5"] == text.index("Financial Review")
        assert repaired_map["5"] < repaired_map["6"]

    def test_key_heading_fallback_researches_item5_after_existing_item4a(self) -> None:
        """若 Item 5 fallback 落在 Item 4A 之前，应在其后重搜并保留合法顺序。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 400)
            + "\nOperating and financial review and prospects\n"
            + ("alpha " * 400)
            + "\nUnresolved Staff Comments\n"
            + ("beta " * 500)
            + "\nLiquidity and capital resources\n"
            + ("gamma " * 600)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("delta " * 600)
        )
        markers = [
            ("3", text.index("opening")),
            ("4", text.index("opening")),
            ("4A", text.index("Unresolved Staff Comments")),
            ("18", text.index("Report of Independent Registered Public Accounting Firm")),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(text, markers)
        repaired_map = dict(repaired)

        assert repaired_map["5"] == text.index("Liquidity and capital resources")
        assert repaired_map["4A"] < repaired_map["5"] < repaired_map["18"]

    def test_key_heading_fallback_keeps_in_order_item5_when_research_result_jumps_before_item4a(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """若重搜 Item 5 回跳到现有 Item 4A 之前，应保留当前顺序内的 Item 5。

        Args:
            monkeypatch: pytest monkeypatch fixture。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        markers = [
            ("3", 30),
            ("4", 100),
            ("4A", 500),
            ("5", 560),
            ("6", 800),
            ("18", 1040),
        ]
        full_text = "x" * 2000

        monkeypatch.setattr(
            "dayu.fins.processors.twenty_f_form_common._find_twenty_f_key_heading_positions",
            lambda text: {"3": 30, "4": 100, "5": 5, "18": 1040},
        )
        monkeypatch.setattr(
            "dayu.fins.processors.twenty_f_form_common._find_twenty_f_key_heading_position_after",
            lambda *, full_text, token, start_at: 400 if token == "5" else None,
        )
        monkeypatch.setattr(
            "dayu.fins.processors.twenty_f_form_common._is_twenty_f_marker_contaminated",
            lambda text, position: position in {500, 560},
        )

        repaired = _repair_twenty_f_key_items_with_heading_fallback(full_text, markers)
        repaired_map = dict(repaired)

        assert repaired_map["4A"] == 500
        assert repaired_map["5"] == 560
        assert repaired_map["4A"] < repaired_map["5"] < repaired_map["6"]

    def test_key_heading_fallback_researches_item18_after_item5_when_early_hit_is_false(self) -> None:
        """若 Item 18 先命中早期 guide 文本，应在 Item 5 之后重搜真正正文锚点。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Cross reference guide\n"
            "Financial statements\n"
            + ("opening " * 500)
            + "\nRisk factors\n"
            + ("alpha " * 600)
            + "\nOperating and financial review and prospects\n"
            + ("beta " * 700)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 800)
        )
        tail_markers = [
            ("3", len(text) - 700),
            ("5", len(text) - 500),
            ("18", len(text) - 900),
            ("19", len(text) - 100),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(text, tail_markers)
        repaired_map = dict(repaired)

        assert repaired_map["3"] == text.index("Risk factors")
        assert repaired_map["5"] == text.index("Operating and financial review and prospects")
        assert repaired_map["18"] == text.index(
            "Report of Independent Registered Public Accounting Firm"
        )
        assert repaired_map["3"] < repaired_map["5"] < repaired_map["18"]

    def test_key_heading_fallback_researches_item18_after_latest_preceding_item(self) -> None:
        """若早期 Item 18 候选落在 Item 15 之前，应在最近前序 Item 之后重搜。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 500)
            + "\nRisk factors\n"
            + ("alpha " * 500)
            + "\nOperating and financial review and prospects\n"
            + ("beta " * 500)
            + "\nSee Item 18. Financial Statements in the annual report.\n"
            + ("gamma " * 500)
            + "\nControls and procedures\n"
            + ("delta " * 500)
            + "\nFinancial statements\n"
            + ("epsilon " * 600)
        )
        markers = [
            ("3", text.index("Risk factors")),
            ("5", text.index("Operating and financial review and prospects")),
            ("15", text.index("Controls and procedures")),
            ("19", len(text) - 100),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(text, markers)
        repaired_map = dict(repaired)

        assert repaired_map["15"] == text.index("Controls and procedures")
        assert repaired_map["18"] == text.index("Financial statements", text.index("Controls and procedures"))
        assert repaired_map["15"] < repaired_map["18"]

    def test_key_heading_positions_prefer_operating_results_over_late_kpi(self) -> None:
        """Item 5 的 key-heading fallback 应优先命中更早经营结果正文。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + "\nRisk factors\n"
            + ("alpha " * 600)
            + "\nOperating results of 2021 compared to 2020\n"
            + ("beta " * 800)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 800)
            + "\nKey performance indicators\n"
            + ("delta " * 800)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["5"] == text.index("Operating results of 2021 compared to 2020")
        assert positions["5"] < positions["18"]

    def test_key_heading_positions_support_financial_performance_for_item5(self) -> None:
        """Item 5 的 key-heading fallback 应支持 Financial Performance 标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + "\nRisk factors\n"
            + ("alpha " * 600)
            + "\nFinancial performance\n"
            + ("beta " * 800)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 800)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["5"] == text.index("Financial performance")
        assert positions["3"] < positions["5"] < positions["18"]

    def test_key_heading_positions_support_summary_of_risk_factors_for_item3(self) -> None:
        """Item 3 的 key-heading fallback 应支持 Summary of Risk Factors 标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + "\nSummary of Risk Factors\n"
            + ("alpha " * 600)
            + "\nOperating and financial review and prospects\n"
            + ("beta " * 800)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 800)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["3"] == text.index("Summary of Risk Factors")
        assert positions["3"] < positions["5"] < positions["18"]

    def test_key_heading_positions_skip_inline_item3_cross_reference(self) -> None:
        """正文中的 `see Item 3 ...` 交叉引用不应抢占真实 Item 3 锚点。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + "\nSee Item 3. Key Information - D. Risk Factors for a discussion of these matters.\n"
            + ("alpha " * 600)
            + "\nSummary of Risk Factors\n"
            + ("beta " * 600)
            + "\nOperating and financial review and prospects\n"
            + ("gamma " * 800)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("delta " * 800)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["3"] == text.index("Summary of Risk Factors")
        assert positions["3"] < positions["5"] < positions["18"]

    def test_key_heading_fallback_researches_item3_after_existing_item2(self) -> None:
        """若 Item 3 fallback 早于已有 Item 2，应在 Item 2 之后重搜正文标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Risk factors may affect our business.\n"
            + ("opening " * 300)
            + "\nOffer Statistics and Expected Timetable\n"
            + ("alpha " * 400)
            + "\nSummary of Risk Factors\n"
            + ("beta " * 500)
            + "\nInformation on the Company\n"
            + ("gamma " * 500)
            + "\nOperating and financial review and prospects\n"
            + ("delta " * 500)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("epsilon " * 500)
        )
        markers = [
            ("1", text.index("opening")),
            ("2", text.index("Offer Statistics and Expected Timetable")),
            ("4", text.index("Information on the Company")),
            ("5", text.index("Operating and financial review and prospects")),
            ("18", text.index("Report of Independent Registered Public Accounting Firm")),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(text, markers)
        repaired_map = dict(repaired)

        assert repaired_map["3"] == text.index("Summary of Risk Factors")
        assert repaired_map["2"] < repaired_map["3"] < repaired_map["4"]

    def test_key_heading_fallback_keeps_item4_and_item5_when_item3_fallback_is_too_late(self) -> None:
        """若 Item 3 fallback 晚到跨过 Item 4/5，不应把已识别正文章节误删。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Item 1. Identity of Directors, Senior Management and Advisers. "
            + ("opening " * 120)
            + "Item 2. Offer Statistics and Expected Timetable. "
            + ("alpha " * 120)
            + "Item 3. page 20 "
            + ("beta " * 120)
            + "\nInformation on the Company\n"
            + ("gamma " * 220)
            + "\nUnresolved Staff Comments\n"
            + ("delta " * 220)
            + "\nOperating and financial review and prospects\n"
            + ("epsilon " * 240)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("zeta " * 260)
            + "\nRisk factors\n"
            + ("eta " * 260)
        )
        markers = [
            ("1", text.index("Item 1. Identity of Directors, Senior Management and Advisers.")),
            ("2", text.index("Item 2. Offer Statistics and Expected Timetable.")),
            ("3", text.index("Item 3. page 20")),
            ("4", text.index("Information on the Company")),
            ("4A", text.index("Unresolved Staff Comments")),
            ("5", text.index("Operating and financial review and prospects")),
            ("18", text.index("Report of Independent Registered Public Accounting Firm")),
        ]

        repaired = _repair_twenty_f_key_items_with_heading_fallback(text, markers)
        repaired_map = dict(repaired)

        assert repaired_map["4"] == text.index("Information on the Company")
        assert repaired_map["5"] == text.index("Operating and financial review and prospects")
        assert repaired_map["3"] < repaired_map["4"] < repaired_map["5"] < repaired_map["18"]

    def test_key_heading_positions_choose_earliest_valid_item5_match(self) -> None:
        """Item 5 应在多个有效正文标题中选择最早命中，而非受模式顺序影响。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + "\nKey performance indicators\n"
            + ("alpha " * 700)
            + "\nFinancial review\n"
            + ("beta " * 700)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 700)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["5"] == text.index("Key performance indicators")
        assert positions["5"] < text.index("Financial review")

    def test_key_heading_positions_skip_inline_item5_cross_reference_and_keep_real_heading(self) -> None:
        """正文里的 `described under Item 5 ...` 不应早于真实 Item 5 标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + "\nNon-GAAP measures are described under Item 5. Operating and Financial Review and Prospects - Adjusted EBITDA.\n"
            + ("alpha " * 600)
            + "\nITEM 5. OPERATING AND FINANCIAL REVIEW AND PROSPECTS\n"
            + ("beta " * 700)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 700)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["5"] == text.index("ITEM 5. OPERATING AND FINANCIAL REVIEW AND PROSPECTS")
        assert positions["5"] < positions["18"]

    def test_key_heading_positions_skip_quoted_item5_cross_reference_and_keep_real_heading(self) -> None:
        """引号中的 Item 5 交叉引用不应抢占真实 Item 5 标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + '\nFor further detail, see "ITEM 5. OPERATING AND FINANCIAL REVIEW AND PROSPECTS - 5.A Operating Results" below.\n'
            + ("alpha " * 600)
            + "\nITEM 5. OPERATING AND FINANCIAL REVIEW AND PROSPECTS\n"
            + ("beta " * 700)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 700)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["5"] == text.index("ITEM 5. OPERATING AND FINANCIAL REVIEW AND PROSPECTS\n")
        assert positions["5"] < positions["18"]

    def test_key_heading_positions_skip_mid_sentence_quoted_item5_cross_reference(self) -> None:
        """句中引号包裹的 Item 5 引用不应抢占真实 Item 5 标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + "\nin which we operate could impact our earnings and adversely affect our operating performance\" and \"ITEM 5. OPERATING AND FINANCIAL REVIEW AND PROSPECTS - 5.A Operating Results\".\n"
            + ("alpha " * 600)
            + "\nITEM 5.\nOPERATING AND FINANCIAL REVIEW AND PROSPECTS\n"
            + ("beta " * 700)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("gamma " * 700)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["5"] == text.index("ITEM 5.\nOPERATING AND FINANCIAL REVIEW AND PROSPECTS")
        assert positions["5"] < positions["18"]

    def test_signature_uses_furthest_item_position_instead_of_token_order_tail(self) -> None:
        """SIGNATURE 应锚定最靠后的 Item，而不是 token 顺序上的最后一个。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "Item 18. Financial Statements. "
            + ("alpha " * 200)
            + "SIGNATURE "
            + ("beta " * 400)
            + "Item 1. Identity of Directors, Senior Management and Advisers. "
            + ("gamma " * 300)
            + "Item 2. Offer Statistics and Expected Timetable. "
            + ("delta " * 300)
            + "Item 3. Key Information. "
            + ("epsilon " * 300)
            + "FINAL SIGNATURE"
        )

        markers = _build_twenty_f_markers(text)
        signature_positions = [
            position
            for position, title in markers
            if title == "SIGNATURE"
        ]
        item_3_position = next(
            position
            for position, title in markers
            if str(title or "").startswith("Part I - Item 3")
        )

        assert len(signature_positions) == 1
        assert signature_positions[0] > item_3_position

    def test_bare_financial_statements_phrase_inside_sentence_is_not_heading(self) -> None:
        """正文句子中的 bare `financial statements` 不应被误判为 Item 18 标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "The Annual Reports include the consolidated financial statements of UBS Group AG "
            "and provide comprehensive information about the firm. "
            + ("alpha " * 1200)
            + "Item 18. Financial Statements. "
            + ("beta " * 800)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["18"] > 5000

    def test_bare_financial_statements_phrase_with_short_prefix_still_prefers_real_heading(self) -> None:
        """短前缀正文句中的 bare 财报短语不应早于真实 Item 18 标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + "\nAegon prepares its consolidated financial statements in accordance with IFRS as adopted by the European Union.\n"
            + ("alpha " * 600)
            + "\nConsolidated financial statements of Aegon N.V.\n"
            + ("beta " * 800)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["18"] == text.index("Consolidated financial statements of Aegon N.V.")

    def test_bare_financial_statements_sentence_with_subject_pronoun_is_not_heading(self) -> None:
        """带主语前缀的正文句不应被识别为 bare Item 18 标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            ("opening " * 600)
            + "\nOur financial statements, particularly our interest-earning assets and interest-bearing liabilities, could be exposed to fluctuations in interest rates.\n"
            + ("alpha " * 600)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("beta " * 800)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["18"] == text.index("Report of Independent Registered Public Accounting Firm")

    def test_bare_financial_statements_page_locator_cluster_is_not_heading(self) -> None:
        """夹在页码 locator 块中的 bare 财报标题不应抢占后续真实 Item 18。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        text = (
            "95 ¨ Results 2021 Americas\n"
            "99 ¨ Results 2021 The Netherlands\n"
            "103 ¨ Results 2021 United Kingdom\n"
            "110 ¨ Results 2021 Asset Management\n"
            "Consolidated financial statements of Aegon N.V.\n"
            "112 Exchange rates\n"
            "113 Consolidated income statement of Aegon N.V.\n"
            "114 Consolidated statement of financial position of Aegon N.V.\n"
            + ("alpha " * 800)
            + "\nReport of Independent Registered Public Accounting Firm\n"
            + ("beta " * 800)
        )

        positions = _find_twenty_f_key_heading_positions(text)

        assert positions["18"] == text.index("Report of Independent Registered Public Accounting Firm")


@pytest.mark.unit
def test_enforce_marker_position_monotonicity_reverts_previous_move_for_clean_later_item() -> None:
    """顺序修复遇到正常后继 Item 时，应回滚前一个被移动的 marker。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 17. Financial Statements\n"
        + ("alpha " * 200)
        + "Item 18. Financial Statements\n"
        + ("beta " * 200)
        + "Item 19. Exhibits\n"
        + ("gamma " * 200)
        + "\nStandalone Financial Statements\n"
        + ("delta " * 200)
    )
    original_positions = {
        "17": text.index("Item 17"),
        "18": text.index("Item 18"),
        "19": text.index("Item 19"),
    }
    repaired = [
        ("17", original_positions["17"]),
        ("18", text.index("Standalone Financial Statements")),
        ("19", original_positions["19"]),
    ]

    normalized = _enforce_marker_position_monotonicity(
        text,
        repaired,
        original_positions,
    )

    assert normalized == [
        ("17", original_positions["17"]),
        ("18", original_positions["18"]),
        ("19", original_positions["19"]),
    ]


@pytest.mark.unit
def test_enforce_marker_position_monotonicity_drops_trailing_contaminated_marker_for_clean_item3() -> None:
    """若 Item 3 被尾部污染 marker 挡住，顺序修复应优先移除污染 marker。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 3. Key Information\n"
        + ("alpha " * 120)
        + "\nItem 5. Operating and Financial Review and Prospects\n"
        + ("beta " * 120)
        + "\nReference guide line for Item 2 300\n"
    )
    contaminated_item2_pos = text.index("Item 2 300")
    item3_pos = text.index("Item 3. Key Information")
    item5_pos = text.index("Item 5. Operating and Financial Review and Prospects")
    repaired = [
        ("2", contaminated_item2_pos),
        ("3", item3_pos),
        ("5", item5_pos),
    ]

    with patch(
        "dayu.fins.processors.twenty_f_form_common._is_twenty_f_marker_contaminated",
        side_effect=lambda full_text, position: position == contaminated_item2_pos,
    ):
        normalized = _enforce_marker_position_monotonicity(
            text,
            repaired,
            {},
        )

    assert normalized == [
        ("3", item3_pos),
        ("5", item5_pos),
    ]


@pytest.mark.unit
def test_enforce_marker_position_monotonicity_drops_multiple_trailing_contaminated_markers() -> None:
    """若尾部存在连续污染 marker，应循环清除直到干净 marker 可以落位。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 3. Key Information\n"
        + ("alpha " * 120)
        + "\nItem 5. Operating and Financial Review and Prospects\n"
        + ("beta " * 120)
        + "\nGuide row for Item 1 100\n"
        + "Guide row for Item 2 200\n"
    )
    contaminated_item1_pos = text.index("Item 1 100")
    contaminated_item2_pos = text.index("Item 2 200")
    item3_pos = text.index("Item 3. Key Information")
    item5_pos = text.index("Item 5. Operating and Financial Review and Prospects")
    repaired = [
        ("1", contaminated_item1_pos),
        ("2", contaminated_item2_pos),
        ("3", item3_pos),
        ("5", item5_pos),
    ]

    with patch(
        "dayu.fins.processors.twenty_f_form_common._is_twenty_f_marker_contaminated",
        side_effect=lambda full_text, position: position in {contaminated_item1_pos, contaminated_item2_pos},
    ):
        normalized = _enforce_marker_position_monotonicity(
            text,
            repaired,
            {},
        )

    assert normalized == [
        ("3", item3_pos),
        ("5", item5_pos),
    ]


@pytest.mark.unit
def test_enforce_marker_position_monotonicity_keeps_synthesized_item18_when_non_contaminated() -> None:
    """顺序修复不应丢弃 annual-report-style 20-F 中后补出的非污染 Item 18。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 3. Key Information\n"
        + ("alpha " * 120)
        + "\nConsolidated financial statements of Example Corp.\n"
        + ("beta " * 120)
        + "\nItem 5. Operating and Financial Review and Prospects\n"
        + ("gamma " * 120)
    )
    repaired = [
        ("3", text.index("Item 3. Key Information")),
        ("5", text.index("Item 5. Operating and Financial Review and Prospects")),
        ("18", text.index("Consolidated financial statements of Example Corp.")),
    ]
    original_positions = {
        "3": text.index("Item 3. Key Information"),
        "5": text.index("Item 5. Operating and Financial Review and Prospects"),
    }

    normalized = _enforce_marker_position_monotonicity(
        text,
        repaired,
        original_positions,
    )

    assert ("18", text.index("Consolidated financial statements of Example Corp.")) in normalized


@pytest.mark.unit
def test_enforce_marker_position_monotonicity_keeps_replacement_item18_when_original_is_contaminated() -> None:
    """若原始 Item 18 落在污染区，顺序修复应保留后补出的真实 Item 18。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 18. Financial Statements\n"
        + ("front matter " * 40)
        + "\nItem 3. Key Information\n"
        + ("alpha " * 120)
        + "\nConsolidated financial statements of Example Corp.\n"
        + ("beta " * 120)
        + "\nItem 5. Operating and Financial Review and Prospects\n"
        + ("gamma " * 120)
    )
    repaired = [
        ("3", text.index("Item 3. Key Information")),
        ("5", text.index("Item 5. Operating and Financial Review and Prospects")),
        ("18", text.index("Consolidated financial statements of Example Corp.")),
    ]
    original_positions = {
        "3": text.index("Item 3. Key Information"),
        "5": text.index("Item 5. Operating and Financial Review and Prospects"),
        "18": text.index("Item 18. Financial Statements"),
    }

    original_item18_pos = original_positions["18"]
    replacement_item18_pos = text.index("Consolidated financial statements of Example Corp.")

    with patch(
        "dayu.fins.processors.twenty_f_form_common._is_twenty_f_marker_contaminated",
        side_effect=lambda full_text, position: position == original_item18_pos,
    ):
        normalized = _enforce_marker_position_monotonicity(
            text,
            repaired,
            original_positions,
        )

    assert ("18", replacement_item18_pos) in normalized


@pytest.mark.unit
def test_key_heading_fallback_keeps_item18_replacement_before_late_item5_when_original_is_contaminated() -> None:
    """若原始 Item 18 污染且 fallback 早于 Item 5，修复阶段仍应保留该替换值。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "front matter placeholder\n"
        + "Item 3. Key Information\n"
        + ("alpha " * 80)
        + "\nConsolidated financial statements of Example Corp.\n"
        + ("beta " * 80)
        + "\nItem 5. Operating and Financial Review and Prospects\n"
        + ("gamma " * 80)
    )
    replacement_item18_pos = text.index("Consolidated financial statements of Example Corp.")
    item_markers = [
        ("3", text.index("Item 3. Key Information")),
        ("5", text.index("Item 5. Operating and Financial Review and Prospects")),
        ("18", 5),
    ]

    with patch(
        "dayu.fins.processors.twenty_f_form_common._find_twenty_f_key_heading_positions",
        return_value={
            "3": text.index("Item 3. Key Information"),
            "5": text.index("Item 5. Operating and Financial Review and Prospects"),
            "18": replacement_item18_pos,
        },
    ), patch(
        "dayu.fins.processors.twenty_f_form_common._find_twenty_f_key_heading_position_after",
        return_value=None,
    ), patch(
        "dayu.fins.processors.twenty_f_form_common._is_twenty_f_marker_contaminated",
        side_effect=lambda full_text, position: position == 5,
    ), patch(
        "dayu.fins.processors.twenty_f_form_common._looks_like_twenty_f_front_matter_marker",
        side_effect=lambda full_text, position: position == 5,
    ):
        repaired = _repair_twenty_f_key_items_with_heading_fallback(text, item_markers)

    assert ("18", replacement_item18_pos) in repaired


# ────────────────────────────────────────────────────────────────
# BsTwentyFFormProcessor 基础属性测试
# ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBsTwentyFFormProcessorProperties:
    """验证 BsTwentyFFormProcessor 的基础属性。"""

    def test_parser_version(self) -> None:
        """验证 PARSER_VERSION 格式正确。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        assert BsTwentyFFormProcessor.PARSER_VERSION.startswith("bs_twenty_f_processor_")

    def test_supported_forms(self) -> None:
        """验证仅支持 20-F。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        assert BsTwentyFFormProcessor._SUPPORTED_FORMS == frozenset({"20-F"})

    def test_supports_20f(self, tmp_path: Path) -> None:
        """验证 supports() 对 20-F  + HTML 返回 True。

        Args:
            tmp_path: pytest 临时目录。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>test</body></html>", encoding="utf-8")
        source = _make_source(html_file)
        assert BsTwentyFFormProcessor.supports(source, form_type="20-F") is True

    def test_rejects_10k(self, tmp_path: Path) -> None:
        """验证 supports() 对 10-K 返回 False。

        Args:
            tmp_path: pytest 临时目录。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>test</body></html>", encoding="utf-8")
        source = _make_source(html_file)
        assert BsTwentyFFormProcessor.supports(source, form_type="10-K") is False

    def test_rejects_10q(self, tmp_path: Path) -> None:
        """验证 supports() 对 10-Q 返回 False。

        Args:
            tmp_path: pytest 临时目录。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        html_file = tmp_path / "test.html"
        html_file.write_text("<html><body>test</body></html>", encoding="utf-8")
        source = _make_source(html_file)
        assert BsTwentyFFormProcessor.supports(source, form_type="10-Q") is False


# ────────────────────────────────────────────────────────────────
# BsTwentyFFormProcessor 集成测试（HTML → sections）
# ────────────────────────────────────────────────────────────────


def _build_twenty_f_html(items: list[str], *, padding: int = 1000) -> str:
    """构建包含多个 Item heading 的 20-F HTML 文档。

    Args:
        items: Item token 列表。
        padding: 每个 Item 后的填充段落字符数。

    Returns:
        HTML 字符串。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    body_parts: list[str] = []
    for token in items:
        body_parts.append(f"<h2>Item {token}.</h2>")
        body_parts.append(f"<p>{'Content for Item ' + token + '. ' * 50}</p>")
        body_parts.append(f"<p>{'Detailed analysis text. ' * (padding // 25)}</p>")
    body_parts.append("<h2>SIGNATURE</h2>")
    body_parts.append("<p>The registrant hereby certifies...</p>")
    return f"<html><body>{''.join(body_parts)}</body></html>"


@pytest.mark.unit
def test_bs_twenty_f_processor_sections(tmp_path: Path) -> None:
    """验证 BsTwentyFFormProcessor 生成带 Part + 描述的虚拟章节。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html_file = tmp_path / "twenty_f.html"
    html_content = _build_twenty_f_html(
        ["1", "3", "4", "5", "8", "11", "18", "19"],
        padding=2000,
    )
    html_file.write_text(html_content, encoding="utf-8")

    processor = BsTwentyFFormProcessor(
        _make_source(html_file),
        form_type="20-F",
        media_type="text/html",
    )
    sections = processor.list_sections()
    titles = [str(s.get("title", "")) for s in sections]

    # 应包含 Part 前缀和描述
    assert any("Part I" in t and "Item 3" in t and "Key Information" in t for t in titles)
    assert any("Part II" in t and "Item 5" in t for t in titles)
    assert any("Part IV" in t and "Item 18" in t and "Financial Statements" in t for t in titles)

    # 应有 SIGNATURE 章节
    assert any("SIGNATURE" in t for t in titles)

    # 章节数应在合理区间
    assert len(sections) >= 5


@pytest.mark.unit
def test_bs_twenty_f_processor_prefers_source_text_when_base_text_is_flattened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 BS 20-F 也会在默认全文抽取失真时改用保留行边界的源文本。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html_file = tmp_path / "twenty_f_source_text_recovery.html"
    html_file.write_text(
        """
        <html><body>
        <table>
          <tr><td>ITEM 3.</td></tr>
          <tr><td>KEY INFORMATION</td></tr>
        </table>
        <p>Annual report detailed discussion.</p>
        <table>
          <tr><td>ITEM 4.</td></tr>
          <tr><td>INFORMATION ON THE COMPANY</td></tr>
        </table>
        <p>Annual report detailed discussion.</p>
        <table>
          <tr><td>ITEM 5.</td></tr>
          <tr><td>OPERATING AND FINANCIAL REVIEW AND PROSPECTS</td></tr>
        </table>
        <p>Annual report detailed discussion.</p>
        <table>
          <tr><td>ITEM 18.</td></tr>
          <tr><td>FINANCIAL STATEMENTS</td></tr>
        </table>
        <p>Annual report detailed discussion.</p>
        <table>
          <tr><td>ITEM 19.</td></tr>
          <tr><td>EXHIBITS</td></tr>
        </table>
        <p>Annual report detailed discussion.</p>
        <p>SIGNATURE</p>
        </body></html>
        """,
        encoding="utf-8",
    )
    flattened_text = ("Annual report detailed discussion. " * 40) + "SIGNATURE"
    monkeypatch.setattr(
        "dayu.fins.processors.bs_report_form_common._BaseBsReportFormProcessor._collect_document_text",
        lambda self: flattened_text,
    )

    processor = BsTwentyFFormProcessor(
        _make_source(html_file),
        form_type="20-F",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert any(title.startswith("Part I - Item 3") for title in titles)
    assert any(title.startswith("Part II - Item 5") for title in titles)
    assert any(title.startswith("Part IV - Item 18") for title in titles)


@pytest.mark.unit
def test_bs_twenty_f_processor_keeps_bs_text_when_minimum_item_quality_is_already_met(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 BS 20-F 在默认全文已达标时不会被更激进的源文本替换。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    parsed_text = _make_twenty_f_body_text(["3", "4", "5", "18"])
    html_file = tmp_path / "twenty_f_bs_text_preferred.html"
    html_file.write_text(
        f"<html><body><pre>{parsed_text}</pre></body></html>",
        encoding="utf-8",
    )

    source_text = _make_twenty_f_body_text(["1", "3", "4", "5", "8", "11", "18", "19"])
    monkeypatch.setattr(
        "dayu.fins.processors.bs_twenty_f_processor._extract_source_text_preserving_lines",
        lambda source: source_text,
    )

    processor = BsTwentyFFormProcessor(
        _make_source(html_file),
        form_type="20-F",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert any(title.startswith("Part I - Item 3") for title in titles)
    assert any(title.startswith("Part II - Item 5") for title in titles)
    assert any(title.startswith("Part IV - Item 18") for title in titles)
    assert not any(title.startswith("Part I - Item 1") for title in titles)
    assert not any(title.startswith("Part II - Item 8") for title in titles)


@pytest.mark.unit
def test_bs_twenty_f_processor_reuses_marker_cache_for_same_full_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 BS 20-F 对同一份全文不会重复重建 marker。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    parsed_text = _make_twenty_f_body_text(["3", "4", "5", "18", "19"])
    html_file = tmp_path / "twenty_f_marker_cache.html"
    html_file.write_text(
        f"<html><body><pre>{parsed_text}</pre></body></html>",
        encoding="utf-8",
    )

    original_build_markers = bs_twenty_f_processor_module._build_twenty_f_markers
    built_texts: list[str] = []

    def _counted_build_twenty_f_markers(full_text: str) -> list[tuple[int, Optional[str]]]:
        """记录 marker 构建输入文本并复用真实实现。

        Args:
            full_text: 待构建 marker 的全文文本。

        Returns:
            真实 marker 列表。

        Raises:
            RuntimeError: 底层 marker 构建失败时抛出。
        """

        built_texts.append(full_text)
        return original_build_markers(full_text)

    monkeypatch.setattr(
        "dayu.fins.processors.bs_twenty_f_processor._build_twenty_f_markers",
        _counted_build_twenty_f_markers,
    )

    processor = BsTwentyFFormProcessor(
        _make_source(html_file),
        form_type="20-F",
        media_type="text/html",
    )

    assert processor.list_sections()
    assert len(built_texts) == 1
    assert built_texts[0].strip() == parsed_text.strip()


@pytest.mark.unit
def test_bs_twenty_f_processor_prefers_dom_text_without_stripping_node_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 BS 20-F 会优先使用未 strip 的 DOM 文本保住 Item 5。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html_file = tmp_path / "twenty_f_bs_dom_text.html"
    html_file.write_text(
        """
        <html><body>
        <div><span>Item 1.</span></div><div><span>Identity of Directors, Senior Management and Advisers</span></div>
        <div><span>Item 3.</span></div><div><span>Key Information</span></div>
        <div><span>Item 4.</span></div><div><span>Information on the Company</span></div>
        <div><span>Item 5.</span></div><div><span>Operating and Financial Review and Prospects</span></div>
        <p>{body}</p>
        <div><span>Item 18.</span></div><div><span>Financial Statements</span></div>
        <p>{body}</p>
        <div><span>Item 19.</span></div><div><span>Exhibits</span></div>
        <p>{body}</p>
        <p>SIGNATURE</p>
        </body></html>
        """.format(body="Detailed annual report discussion. " * 180),
        encoding="utf-8",
    )

    flattened_text = (
        "Item 1. Identity of Directors, Senior Management and Advisers "
        "Item 3. Key Information Item 4. Information on the Company "
        "Item 5. Operating and Financial Review and Prospects "
        "Item 18. Financial Statements Item 19. Exhibits SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.bs_report_form_common._BaseBsReportFormProcessor._collect_document_text",
        lambda self: flattened_text,
    )
    monkeypatch.setattr(
        "dayu.fins.processors.bs_twenty_f_processor._extract_source_text_preserving_lines",
        lambda source: flattened_text,
    )

    processor = BsTwentyFFormProcessor(
        _make_source(html_file),
        form_type="20-F",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert any(title.startswith("Part I - Item 3") for title in titles)
    assert any(title.startswith("Part II - Item 5") for title in titles)
    assert any(title.startswith("Part IV - Item 18") for title in titles)


@pytest.mark.unit
def test_find_twenty_f_guide_item_spans_extracts_locator_blocks_in_order() -> None:
    """验证 20-F guide 片段中的 Item locator block 能按顺序识别。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    snippet = """
    Form 20-F Caption | Location in this document
    Item 3. Key Information | Annual Report pages 12-19
    Item 4. Information on the Company | Annual Report pages 20-38
    Item 5. Operating and Financial Review and Prospects | Annual Report pages 39-58
    Item 18. Financial Statements | Annual Report pages F-1 to F-120
    """

    spans = _find_twenty_f_guide_item_spans(snippet)
    tokens = [token for token, _, _ in spans]

    assert tokens == ["3", "4", "5", "18"]


@pytest.mark.unit
def test_item18_heading_with_attached_hereto_body_is_not_misclassified_as_reference_guide() -> None:
    """真实 Item 18 正文不应因页码定位语句被误判为 guide。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 15. Controls and Procedures\n"
        + ("alpha " * 200)
        + "\nItem 18.\n"
        + "FINANCIAL STATEMENTS\n"
        + (
            "The audited consolidated financial statements as required under Item 18 of this Part III "
            "are attached hereto starting on page F-1 of this annual report on Form 20-F. "
            "The audit report of KPMG, our independent registered public accounting firm, is included "
            "herein preceding the audited consolidated financial statements.\n"
        )
        + "\nItem 19.\n"
        + "EXHIBITS\n"
        + ("beta " * 120)
    )

    item_18_position = text.index("Item 18.\nFINANCIAL STATEMENTS")
    item_19_position = text.index("Item 19.\nEXHIBITS")

    assert _looks_like_twenty_f_reference_guide_marker(text, item_18_position) is False

    markers = _build_twenty_f_markers(text)
    marker_map = {str(title): int(position) for position, title in markers if title}

    assert marker_map["Part IV - Item 18 - Financial Statements"] == item_18_position
    assert marker_map["Part IV - Item 19 - Exhibits"] == item_19_position


@pytest.mark.unit
def test_item18_heading_with_page_f1_body_survives_toc_filters_during_research() -> None:
    """`Item 18` 标题后带 `page F-1` 正文时，重搜不应误判为目录行。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 15.\n"
        "CONTROLS AND PROCEDURES\n"
        + ("alpha " * 120)
        + "\nItem 17.\n"
        + "FINANCIAL STATEMENTS\n"
        + "We have elected to provide financial statements pursuant to Item 18 of this Part III.\n"
        + "Item 18.\n"
        + "FINANCIAL STATEMENTS\n"
        + (
            "The audited consolidated financial statements as required under Item 18 of this Part III "
            "are attached hereto starting on page F-1 of this annual report on Form 20-F.\n"
        )
        + "Item 19.\n"
        + "EXHIBITS\n"
    )

    item_15_position = text.index("Item 15.\nCONTROLS AND PROCEDURES")
    item_18_position = text.index("Item 18.\nFINANCIAL STATEMENTS")

    later_item_18 = _find_twenty_f_key_heading_position_after(
        full_text=text,
        token="18",
        start_at=item_15_position + 1,
    )

    assert later_item_18 == item_18_position


@pytest.mark.unit
def test_non_item18_heading_research_skips_item18_body_whitelist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非 Item 18 的 heading 重搜不应触发 Item 18 正文白名单判定。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 3.\n"
        "KEY INFORMATION\n"
        + ("alpha " * 120)
        + "\nItem 18.\n"
        + "FINANCIAL STATEMENTS\n"
        + ("beta " * 120)
    )

    def _forbidden(*args: object, **kwargs: object) -> bool:
        raise AssertionError("non-Item 18 research should not call Item 18 body whitelist")

    monkeypatch.setattr(
        "dayu.fins.processors.twenty_f_form_common._looks_like_twenty_f_item18_heading_with_body",
        _forbidden,
    )

    position = _find_first_valid_twenty_f_heading_position(
        full_text=text,
        pattern=re.compile(r"(?im)^\s*item\s*3[\.\-:\s]*$"),
        start_at=0,
        token="3",
    )

    assert position == text.index("Item 3.")


@pytest.mark.unit
def test_plain_financial_statements_match_skips_item18_body_whitelist_without_local_item18(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """纯 `Financial Statements` 命中若附近没有 `Item 18`，不应触发正文白名单。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Consolidated Financial Statements\n"
        + ("alpha " * 120)
        + "\nFurther discussion follows.\n"
    )

    def _forbidden(*args: object, **kwargs: object) -> bool:
        raise AssertionError("plain financial statements match should not call Item 18 body whitelist")

    monkeypatch.setattr(
        "dayu.fins.processors.twenty_f_form_common._looks_like_twenty_f_item18_heading_with_body",
        _forbidden,
    )

    position = _find_first_valid_twenty_f_heading_position(
        full_text=text,
        pattern=re.compile(r"(?i)\bfinancial\s+statements\b"),
        start_at=0,
        token="18",
    )

    assert position is None


@pytest.mark.unit
def test_reference_guide_marker_skips_item18_body_whitelist_without_local_item18(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """guide 检测遇到普通 `Financial Statements` 行时，不应触发昂贵 Item 18 白名单。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = "Consolidated Financial Statements\n" + ("alpha " * 60)

    def _forbidden(*args: object, **kwargs: object) -> bool:
        raise AssertionError("reference guide probe should skip Item 18 body whitelist")

    monkeypatch.setattr(
        "dayu.fins.processors.twenty_f_form_common._looks_like_twenty_f_item18_heading_with_body",
        _forbidden,
    )

    position = text.index("Financial Statements")

    assert _looks_like_twenty_f_reference_guide_marker(text, position) is False


@pytest.mark.unit
def test_item18_body_check_short_circuits_when_local_probe_misses() -> None:
    """验证非 `Item 18` 局部窗口不会进入昂贵邻域扫描。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Financial Statements\n"
        + ("alpha " * 240)
        + "\nBusiness Overview\n"
    )
    position = text.index("Financial Statements")

    with patch(
        "dayu.fins.processors.twenty_f_form_common._extract_twenty_f_line_bounds",
        side_effect=AssertionError("非 Item 18 候选不应进入逐行边界扫描"),
    ):
        assert _looks_like_twenty_f_item18_heading_with_body(text, position) is False


@pytest.mark.unit
def test_inline_cross_reference_plain_heading_without_reference_hints_is_false() -> None:
    """普通独立标题若前缀没有引用提示，不应被判为 inline cross reference。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Part II overview.\n"
        "Financial Review\n"
        + ("alpha " * 80)
    )

    position = text.index("Financial Review")

    assert _looks_like_twenty_f_inline_cross_reference(full_text=text, position=position) is False


@pytest.mark.unit
def test_locator_heading_search_skips_guide_check_for_non_standalone_match() -> None:
    """验证 locator 回查会先过滤正文句内命中，再决定是否做 guide 判断。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 18.\n"
        "FINANCIAL STATEMENTS\n"
        "The audited consolidated financial statements are attached hereto.\n"
    )

    with patch(
        "dayu.fins.processors.twenty_f_form_common._looks_like_twenty_f_reference_guide_marker",
        side_effect=AssertionError("正文句内命中应先被 standalone 过滤，避免进入 guide 判定"),
    ):
        assert _find_twenty_f_key_heading_position_after(
            full_text=text,
            token="18",
            start_at=text.index("The audited"),
        ) is None


@pytest.mark.unit
def test_report_suite_cover_marker_detects_prefiling_annual_report_cover() -> None:
    """报告套件封面若位于真实 SEC filing 之前，应被识别为无效锚点。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "GOLD FIELDS LIMITED Integrated Annual Report 2022 Creating enduring value beyond mining IAR 1 "
        "ABOUT OUR COVER SEND US YOUR FEEDBACK REPORTING SUITE CONTENTS "
        "Financial information 76 "
        "Annual Financial Report Governance Report "
        "As filed with the Securities and Exchange Commission on 30 March 2023 "
        "Form 20-F "
        "Financial Information\n"
        + ("alpha " * 120)
    )

    position = text.index("Financial information")

    assert _looks_like_twenty_f_report_suite_cover_marker(
        full_text=text,
        position=position,
    )


@pytest.mark.unit
def test_locator_heading_search_keeps_annual_report_heading_even_with_front_matter_context() -> None:
    """验证 locator 回查仍会保留 front matter 之后的真实 annual-report 页眉。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Form 20-F caption and location in this document\n"
        "Financial Statements 120\n"
        "The audited consolidated financial statements are attached hereto and include accompanying notes.\n"
    )

    assert _find_twenty_f_locator_heading_position(
        full_text=text,
        candidates=["Financial Statements"],
        start_at=0,
    ) == text.index("Financial Statements")


@pytest.mark.unit
def test_key_heading_search_keeps_annual_report_heading_even_with_front_matter_context() -> None:
    """验证 key-item heading fallback 仍会保留 front matter 之后的真实 annual-report 页眉。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Form 20-F caption and location in this document\n"
        "Financial Statements 120\n"
        "The audited consolidated financial statements are attached hereto and include accompanying notes.\n"
    )
    pattern = re.compile(r"(?i)financial\s+statements")

    assert _find_first_valid_twenty_f_heading_position(
        full_text=text,
        pattern=pattern,
        start_at=0,
        token="18",
    ) == text.index("Financial Statements")


@pytest.mark.unit
def test_key_heading_search_prefers_later_real_heading_over_early_annual_report_toc_line() -> None:
    """annual-report 风格 ToC 命中不应挡住后面的真实正文标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    toc = (
        "Annual Report on Form 20-F\n"
        "Item 3. Key Information 7\n"
        "Item 4. Information on the Company 12\n"
    )
    body = (
        "7\n"
        "Item 3. Key Information\n"
        "3.A [Reserved]\n"
        "3.B Capitalization and indebtedness\n"
        "Not applicable.\n"
        "3.C Reasons for the offer and use of proceeds\n"
        "Not applicable.\n"
        "3.D Risk factors\n"
        "Our business faces significant risks and uncertainties.\n"
        "Item 4. Information on the Company\n"
    )
    text = toc + ("\n" * 20) + body
    pattern = re.compile(r"(?i)item\s+3\.\s+key\s+information")

    assert _find_first_valid_twenty_f_heading_position(
        full_text=text,
        pattern=pattern,
        start_at=0,
        token="3",
    ) == text.index("Item 3. Key Information\n3.A [Reserved]")


@pytest.mark.unit
def test_key_heading_fallback_keeps_real_item18_heading_when_body_contains_page_locator() -> None:
    """Item 18 正文含 `page F-1` locator 时，修复逻辑应保留真实标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = (
        "Item 5.\n"
        + "OPERATING AND FINANCIAL REVIEW AND PROSPECTS\n"
        + ("alpha " * 120)
        + "\nItem 15.\n"
        + "CONTROLS AND PROCEDURES\n"
        + ("beta " * 120)
        + "\nItem 17.\n"
        + "FINANCIAL STATEMENTS\n"
        + "We have elected to provide financial statements pursuant to Item 18 of this Part III.\n"
        + "Item 18.\n"
        + "FINANCIAL STATEMENTS\n"
        + (
            "The audited consolidated financial statements as required under Item 18 of this Part III "
            "are attached hereto starting on page F-1 of this annual report on Form 20-F. "
            "The audit report of our independent registered public accounting firm is included herein "
            "preceding the audited consolidated financial statements.\n"
        )
        + "Item 19.\n"
        + "EXHIBITS\n"
    )
    markers = [
        ("5", text.index("Item 5.\nOPERATING AND FINANCIAL REVIEW AND PROSPECTS")),
        ("15", text.index("Item 15.\nCONTROLS AND PROCEDURES")),
        ("17", text.index("Item 17.\nFINANCIAL STATEMENTS")),
        ("18", text.index("Item 18.\nFINANCIAL STATEMENTS")),
        ("19", text.index("Item 19.\nEXHIBITS")),
    ]

    repaired = _repair_twenty_f_key_items_with_heading_fallback(text, markers)
    repaired_map = dict(repaired)

    assert repaired_map["18"] == text.index("Item 18.\nFINANCIAL STATEMENTS")


@pytest.mark.unit
def test_key_heading_fallback_does_not_overwrite_clean_nvs_style_body_markers() -> None:
    """单调 fallback 主链不应覆盖已存在的干净正文 key-item。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    toc = (
        "Annual Report on Form 20-F\n"
        "Item 1. Identity of Directors, Senior Management and Advisers 6\n"
        "Item 2. Offer Statistics and Expected Timetable 7\n"
        "Item 3. Key Information 8\n"
        "Item 4. Information on the Company 22\n"
        "Item 5. Operating and Financial Review and Prospects 46\n"
    )
    long_padding = ("Detailed operating discussion paragraph.\n" * 120)
    body = (
        "Item 1. Identity of Directors, Senior Management and Advisers\n"
        + "Body content for item one continues here.\n"
        + "Item 2. Offer Statistics and Expected Timetable\n"
        + "Body content for item two continues here.\n"
        + long_padding
        + "Item 3. Key Information\n"
        + "3.A [Reserved]\n"
        + "3.B Capitalization and indebtedness\n"
        + "Not applicable.\n"
        + "3.C Reasons for the offer and use of proceeds\n"
        + "Not applicable.\n"
        + "3.D Risk factors\n"
        + "Our business faces significant risks and uncertainties.\n"
        + long_padding
        + "Item 4. Information on the Company\n"
        + "This section describes our company and operations in detail.\n"
        + long_padding
        + "Item 5. Operating and Financial Review and Prospects\n"
        + "Management discusses our financial condition and operating results.\n"
        + "Item 18. Financial Statements\n"
        + "The audited consolidated financial statements follow this heading.\n"
    )
    text = toc + long_padding + body
    markers = [
        ("1", text.index("Item 1. Identity of Directors, Senior Management and Advisers\nBody")),
        ("2", text.index("Item 2. Offer Statistics and Expected Timetable\nBody")),
        ("3", text.index("Item 3. Key Information\n3.A [Reserved]")),
        ("4", text.index("Item 4. Information on the Company\nThis section")),
        ("5", text.index("Item 5. Operating and Financial Review and Prospects\nManagement")),
        ("18", text.index("Item 18. Financial Statements\nThe audited")),
    ]

    repaired = _repair_twenty_f_key_items_with_heading_fallback(text, markers)
    repaired_map = dict(repaired)

    assert repaired_map["3"] == markers[2][1]
    assert repaired_map["4"] == markers[3][1]
    assert repaired_map["5"] >= markers[4][1]


@pytest.mark.unit
def test_find_twenty_f_guide_item_spans_no_catastrophic_backtracking() -> None:
    """验证 guide 片段中不匹配的 token+空白序列不会导致灾难性回溯。

    回归测试：当 snippet 中包含 token 后跟大量空白但无法匹配标准
    description 时，正则必须在合理时间内返回，而非指数级回溯。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    import time

    # 构造包含 token "3" 后跟大量空白的恶意片段
    adversarial_snippet = "3" + " " * 5000 + "unrelated text " * 200
    start = time.monotonic()
    spans = _find_twenty_f_guide_item_spans(adversarial_snippet)
    elapsed = time.monotonic() - start

    # 不应匹配任何 item（因为 description 不存在）
    assert spans == []
    # 必须在 1 秒内完成（修复前会无限卡死）
    assert elapsed < 1.0, f"_find_twenty_f_guide_item_spans 耗时 {elapsed:.2f}s，疑似灾难性回溯"


@pytest.mark.unit
def test_bs_twenty_f_processor_read_section(tmp_path: Path) -> None:
    """验证 read_section 返回有效内容。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html_file = tmp_path / "twenty_f.html"
    html_content = _build_twenty_f_html(
        ["1", "3", "4", "5", "8", "18", "19"],
        padding=2000,
    )
    html_file.write_text(html_content, encoding="utf-8")

    processor = BsTwentyFFormProcessor(
        _make_source(html_file),
        form_type="20-F",
        media_type="text/html",
    )
    sections = processor.list_sections()

    # 读取任一非空章节
    for section in sections:
        ref = section.get("ref")
        if ref is None:
            continue
        content = processor.read_section(ref)
        assert content is not None
        assert content.get("ref") == ref
        text = str(content.get("content", ""))
        # 至少有些内容
        assert len(text) > 0
        break


@pytest.mark.unit
def test_bs_twenty_f_processor_search(tmp_path: Path) -> None:
    """验证 search 能在虚拟章节中命中关键词。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html_file = tmp_path / "twenty_f.html"
    html_content = _build_twenty_f_html(
        ["1", "3", "4", "5", "8", "18", "19"],
        padding=2000,
    )
    html_file.write_text(html_content, encoding="utf-8")

    processor = BsTwentyFFormProcessor(
        _make_source(html_file),
        form_type="20-F",
        media_type="text/html",
    )
    hits = processor.search("Content for Item 3")
    assert len(hits) >= 1
    assert any("Item 3" in str(h.get("section_title", "")) for h in hits)


# ────────────────────────────────────────────────────────────────
# SEC 规则一致性验证
# ────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSecRulesConsistency:
    """验证 SEC 法定映射的完整性和一致性。"""

    def test_part_map_covers_all_items(self) -> None:
        """Part→Item 映射应覆盖所有法定 Item。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        from dayu.fins.processors.twenty_f_form_common import _TWENTY_F_ITEM_ORDER

        for token in _TWENTY_F_ITEM_ORDER:
            assert token in _TWENTY_F_ITEM_PART_MAP, (
                f"Item {token} 不在 _TWENTY_F_ITEM_PART_MAP 中"
            )

    def test_part_boundaries_correct(self) -> None:
        """Part 边界应符合 SEC Form 20-F 法定结构。

        Part I: Items 1–4A
        Part II: Items 5–12
        Part III: Items 13–16J
        Part IV: Items 17–19

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        # Part I
        for token in ("1", "2", "3", "4", "4A"):
            assert _TWENTY_F_ITEM_PART_MAP[token] == "I", f"Item {token} 应属 Part I"

        # Part II
        for token in ("5", "6", "7", "8", "9", "10", "11", "12"):
            assert _TWENTY_F_ITEM_PART_MAP[token] == "II", f"Item {token} 应属 Part II"

        # Part III
        for token in ("13", "14", "15", "16"):
            assert _TWENTY_F_ITEM_PART_MAP[token] == "III", f"Item {token} 应属 Part III"
        for suffix in "ABCDEFGHIJ":
            assert _TWENTY_F_ITEM_PART_MAP[f"16{suffix}"] == "III", (
                f"Item 16{suffix} 应属 Part III"
            )

        # Part IV
        for token in ("17", "18", "19"):
            assert _TWENTY_F_ITEM_PART_MAP[token] == "IV", f"Item {token} 应属 Part IV"

    def test_key_item_descriptions_present(self) -> None:
        """关键分析 Item 应有 SEC 标准描述。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        critical_items = ("3", "4", "5", "8", "11", "18")
        for token in critical_items:
            assert token in _TWENTY_F_ITEM_DESCRIPTIONS, (
                f"关键 Item {token} 缺少 SEC 标准描述"
            )
            assert len(_TWENTY_F_ITEM_DESCRIPTIONS[token]) > 5, (
                f"Item {token} 描述过短"
            )


__all__: list[str] = []
