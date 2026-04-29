"""10-K/10-Q/20-F 专项处理器覆盖率测试。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, TypeAlias

import pytest

from dayu.fins.processors.sec_report_form_common import (
    _extract_source_text_preserving_lines,
    _find_table_of_contents_cutoff,
    _find_toc_cluster_end,
    _is_inline_reference_context,
    _looks_like_inline_toc_snippet,
    _markers_look_like_toc_entries,
    _rank_marker_candidate,
    _normalize_report_form_type,
    _refine_inline_reference_markers,
    _select_ordered_item_markers,
    _select_ordered_item_markers_after_toc,
)
from dayu.fins.processors.sec_form_section_common import (
    _trim_trailing_page_locator,
)
from dayu.fins.processors.bs_twenty_f_processor import (
    _has_minimum_twenty_f_item_quality,
    _has_risky_twenty_f_section_profile,
)
from dayu.fins.processors.ten_k_processor import TenKFormProcessor
from dayu.fins.processors.ten_q_processor import TenQFormProcessor
from dayu.fins.processors.twenty_f_processor import TwentyFFormProcessor
from dayu.fins.processors.twenty_f_form_common import _trim_twenty_f_source_text
from dayu.fins.storage.local_file_source import LocalFileSource


ReportProcessorClass: TypeAlias = (
    type[TenKFormProcessor] | type[TenQFormProcessor] | type[TwentyFFormProcessor]
)


class _FakeSection:
    """测试用章节对象。"""

    def __init__(self, text_value: str) -> None:
        """初始化章节对象。

        Args:
            text_value: 章节文本。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._text_value = text_value
        self.title = "Overview"
        self.name = None
        self.part = None
        self.item = None

    def text(self) -> str:
        """返回章节文本。

        Args:
            无。

        Returns:
            章节文本。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        return self._text_value

    def tables(self) -> list[object]:
        """返回章节内表格。

        Args:
            无。

        Returns:
            空列表。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        return []


class _FakeDocument:
    """测试用文档对象。"""

    def __init__(self, text_value: str) -> None:
        """初始化文档对象。

        Args:
            text_value: 文档全文文本。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.sections = {"only": _FakeSection(text_value)}
        self.tables: list[object] = []
        self._text_value = text_value

    def text(self) -> str:
        """返回全文文本。

        Args:
            无。

        Returns:
            全文文本。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        return self._text_value


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


@pytest.mark.unit
def test_normalize_report_form_type_variants() -> None:
    """验证报告类表单标准化规则。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _normalize_report_form_type("10K") == "10-K"
    assert _normalize_report_form_type(" 10-q ") == "10-Q"
    assert _normalize_report_form_type("20 f") == "20-F"
    assert _normalize_report_form_type(None) is None
    assert _normalize_report_form_type(" ") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("processor_cls", "form_type"),
    [
        (TenKFormProcessor, "10-K"),
        (TenQFormProcessor, "10-Q"),
        (TwentyFFormProcessor, "20-F"),
    ],
)
def test_report_form_processors_support_target_forms(
    tmp_path: Path,
    processor_cls: ReportProcessorClass,
    form_type: str,
) -> None:
    """验证报告类专项处理器支持目标表单。

    Args:
        tmp_path: pytest 临时目录。
        processor_cls: 处理器类型。
        form_type: 目标表单类型。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / f"{form_type}.html"
    source_path.write_text("<html><body>report</body></html>", encoding="utf-8")
    source = _make_source(source_path, media_type="text/html")

    assert processor_cls.supports(source, form_type=form_type, media_type="text/html") is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("processor_cls", "wrong_form"),
    [
        (TenKFormProcessor, "10-Q"),
        (TenQFormProcessor, "20-F"),
        (TwentyFFormProcessor, "10-K"),
    ],
)
def test_report_form_processors_reject_non_target_forms(
    tmp_path: Path,
    processor_cls: ReportProcessorClass,
    wrong_form: str,
) -> None:
    """验证报告类专项处理器拒绝非目标表单。

    Args:
        tmp_path: pytest 临时目录。
        processor_cls: 处理器类型。
        wrong_form: 非目标表单类型。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "wrong_form.html"
    source_path.write_text("<html><body>report</body></html>", encoding="utf-8")
    source = _make_source(source_path, media_type="text/html")

    assert processor_cls.supports(source, form_type=wrong_form, media_type="text/html") is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("processor_cls", "form_type"),
    [
        (TenKFormProcessor, "10-K"),
        (TenQFormProcessor, "10-Q"),
        (TwentyFFormProcessor, "20-F"),
    ],
)
def test_report_form_processors_keep_sec_processor_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    processor_cls: ReportProcessorClass,
    form_type: str,
) -> None:
    """验证报告类专项处理器沿用 SecProcessor 的章节能力。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。
        processor_cls: 处理器类型。
        form_type: 目标表单类型。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / f"{form_type}_detail.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, normalized_form: _FakeDocument(
            f"{form_type} annual report narrative with key disclosures"
        ),
    )

    processor = processor_cls(
        _make_source(source_path),
        form_type=form_type,
        media_type="text/html",
    )
    sections = processor.list_sections()

    assert isinstance(sections, list)


@pytest.mark.unit
def test_report_form_processors_define_parser_versions() -> None:
    """验证报告类专项处理器声明独立 parser_version。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert TenKFormProcessor.PARSER_VERSION == "ten_k_section_processor_v2.0.0"
    assert TenQFormProcessor.PARSER_VERSION == "ten_q_section_processor_v2.0.0"
    assert TwentyFFormProcessor.PARSER_VERSION == "twenty_f_section_processor_v2.1.5"
    assert TwentyFFormProcessor._ENABLE_FAST_SECTION_BUILD is True
    assert TwentyFFormProcessor._FAST_SECTION_BUILD_SINGLE_FULL_TEXT is True


@pytest.mark.unit
def test_find_table_of_contents_cutoff_handles_toc_text() -> None:
    """验证 TOC 截断位置计算。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _find_table_of_contents_cutoff("Table of Contents Item 1") > 0
    assert _find_table_of_contents_cutoff("No toc text") == 0


@pytest.mark.unit
def test_looks_like_inline_toc_snippet_detects_single_line_toc() -> None:
    """验证单行“标题+页码+下一标题”目录片段可被识别。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    toc_like = (
        "Management’s Discussion and Analysis of Financial Condition and Results of Operations 7 "
        "Executive Summary 7 "
        "Item 7A Quantitative and Qualitative Disclosures About Market Risk 22 "
    )
    normal_text = (
        "Management’s Discussion and Analysis of Financial Condition and Results of Operations "
        "describes revenue growth, funding profile and capital actions in detail."
    )
    numeric_body_text = (
        "Operating and Financial Review and Prospects increased 12 percent in 24 markets "
        "and improved margin by 18 basis points without introducing a new item heading."
    )

    assert _looks_like_inline_toc_snippet(toc_like, 0) is True
    assert _looks_like_inline_toc_snippet(normal_text, 0) is False
    assert _looks_like_inline_toc_snippet(numeric_body_text, 0) is False


@pytest.mark.unit
def test_processor_toc_line_detection_supports_inline_toc() -> None:
    """验证 10-K/10-Q/20-F 的 TOC 行判定支持单行目录文本。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.ten_k_form_common import _looks_like_toc_page_line as ten_k_toc
    from dayu.fins.processors.ten_q_form_common import _looks_like_toc_page_line as ten_q_toc
    from dayu.fins.processors.twenty_f_form_common import _looks_like_toc_page_line as twenty_f_toc

    inline_toc = (
        "Item 1A Risk Factors 50 "
        "Item 7 Management's Discussion and Analysis of Financial Condition and Results of Operations 70 "
        "Item 8 Financial Statements and Supplementary Data 140 "
    )

    assert ten_k_toc(inline_toc, inline_toc.index("Item 7")) is True
    assert ten_q_toc(inline_toc, inline_toc.index("Item 7")) is True
    assert twenty_f_toc(inline_toc, inline_toc.index("Item 8")) is True


@pytest.mark.unit
def test_trim_trailing_page_locator_removes_short_item_page_tail() -> None:
    """验证短 Item 章节尾部页码定位符会被清洗。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = "Financial Statements See Item 17 Financial Statements 163"
    title = "Part IV - Item 18 - Financial Statements"
    cleaned = _trim_trailing_page_locator(content, title)
    assert cleaned == "Financial Statements See Item 17 Financial Statements"


@pytest.mark.unit
def test_trim_trailing_page_locator_keeps_non_item_numeric_tail() -> None:
    """验证非 Item 标题场景不会误删尾部数字。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = "Revenue increased to 2024 due to business expansion in key regions"
    title = "Management Overview"
    assert _trim_trailing_page_locator(content, title) == content


@pytest.mark.unit
def test_has_minimum_twenty_f_item_quality_accepts_key_items() -> None:
    """验证 20-F BS 质量门禁在关键 Item 命中时通过。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    titles: list[str | None] = [
        "Cover Page",
        "Part I - Item 3 - Key Information",
        "Part II - Item 5 - Operating and Financial Review and Prospects",
        "Part IV - Item 18 - Financial Statements",
    ]
    assert _has_minimum_twenty_f_item_quality(titles) is True


@pytest.mark.unit
def test_has_minimum_twenty_f_item_quality_rejects_missing_core_item() -> None:
    """验证 20-F BS 质量门禁会拒绝缺失核心 Item 3/5/18 的结果。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    titles: list[str | None] = [
        "Cover Page",
        "Part II - Item 5 - Operating and Financial Review and Prospects",
        "Part IV - Item 18 - Financial Statements",
        "Part IV - Item 19 - Exhibits",
    ]
    assert _has_minimum_twenty_f_item_quality(titles) is False


@pytest.mark.unit
def test_has_minimum_twenty_f_item_quality_rejects_broken_structure() -> None:
    """验证 20-F BS 质量门禁会拒绝缺失 Item 结构的切分结果。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    titles = [
        None,
        "A. McDonald Skadden, Arps, Slate, Meagher & Flom (UK) LLP",
        "1. Education 2",
        "Note 22",
    ]
    assert _has_minimum_twenty_f_item_quality(titles) is False


@pytest.mark.unit
def test_has_risky_twenty_f_section_profile_detects_huge_section() -> None:
    """验证 20-F 风险判定会识别超大章节结构。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    class _Section:
        """测试用章节对象。"""

        def __init__(self, title: str, content: str) -> None:
            """初始化章节对象。

            Args:
                title: 标题。
                content: 内容。

            Returns:
                无。

            Raises:
                ValueError: 参数非法时抛出。
            """

            self.title = title
            self.content = content

    sections = [_Section("Part IV - Item 19 - Exhibits", "A" * 300500)]
    assert _has_risky_twenty_f_section_profile(sections) is True


@pytest.mark.unit
def test_has_risky_twenty_f_section_profile_detects_item18_toc_stub() -> None:
    """验证 20-F 风险判定会识别 Item 18 目录桩文本。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    class _Section:
        """测试用章节对象。"""

        def __init__(self, title: str, content: str) -> None:
            """初始化章节对象。

            Args:
                title: 标题。
                content: 内容。

            Returns:
                无。

            Raises:
                ValueError: 参数非法时抛出。
            """

            self.title = title
            self.content = content

    sections = [
        _Section(
            "Part IV - Item 18 - Financial Statements",
            "Financial Statements and Notes thereto 163",
        )
    ]
    assert _has_risky_twenty_f_section_profile(sections) is True


@pytest.mark.unit
def test_has_risky_twenty_f_section_profile_detects_short_item18_page_range_reference() -> None:
    """验证 20-F 风险判定会识别短 Item 18 页码区间引用桩。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    class _Section:
        """测试用章节对象。"""

        def __init__(self, title: str, content: str) -> None:
            """初始化章节对象。

            Args:
                title: 标题。
                content: 内容。

            Returns:
                无。

            Raises:
                ValueError: 参数非法时抛出。
            """

            self.title = title
            self.content = content

    sections = [
        _Section(
            "Part IV - Item 18 - Financial Statements",
            (
                "ITEM 18. FINANCIAL STATEMENTS\n"
                "The financial statements filed as part of this Annual Report are included on pages\n"
                "F-1\nthrough\nF-86\nhereof."
            ),
        )
    ]
    assert _has_risky_twenty_f_section_profile(sections) is True


@pytest.mark.unit
def test_select_ordered_item_markers_respects_order_and_range() -> None:
    """验证顺序 Item 选择遵循 token 顺序和范围约束。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = (
        "Item 1. Intro "
        "Item 2. Business "
        "Item 3. Risks "
        "Item 4. Controls"
    )
    pattern = re.compile(r"(?i)\bitem\s+(\d)\s*[\.\:\-]")
    selected = _select_ordered_item_markers(
        content,
        item_pattern=pattern,
        ordered_tokens=("1", "2", "3", "4"),
        start_at=0,
        end_at=content.find("Item 4"),
    )
    assert [item[0] for item in selected] == ["1", "2", "3"]


@pytest.mark.unit
def test_select_ordered_item_markers_prefers_non_inline_heading_candidate() -> None:
    """验证顺序 Item 选择优先命中非行内引用候选。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = (
        "Narrative sentence says see Item 3. Key Information for details in this section. "
        "Additional narrative continues.\n\n"
        "ITEM 3. KEY INFORMATION\n"
        "Long heading body content.\n\n"
        "ITEM 4. INFORMATION ON THE COMPANY\n"
        "Long heading body content."
    )
    pattern = re.compile(r"(?i)\bitem\s+([3-4])\s*[\.\:\-]")
    inline_pos = content.index("Item 3. Key Information")
    heading_pos = content.index("ITEM 3. KEY INFORMATION")

    selected = _select_ordered_item_markers(
        content,
        item_pattern=pattern,
        ordered_tokens=("3", "4"),
    )

    assert len(selected) == 2
    assert selected[0][0] == "3"
    assert selected[0][1] == heading_pos
    assert selected[0][1] != inline_pos


@pytest.mark.unit
def test_select_ordered_item_markers_skips_inline_reference_without_punctuation() -> None:
    """验证无标点行内引用（Item X in/and ...）不会抢占标题命中。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = (
        "As discussed in Item 1A in this report, risks may affect results. "
        "Additional narrative before heading.\n\n"
        "ITEM 1A. RISK FACTORS\n"
        "Risk details body text.\n\n"
        "ITEM 2. PROPERTIES\n"
        "Properties details body text."
    )
    pattern = re.compile(r"(?i)\bitem\s+(1A|2)\s*(?:[\.\:\-\u2013\u2014]\s*|\s+(?=[A-Za-z]))")
    inline_pos = content.index("Item 1A in this report")
    heading_pos = content.index("ITEM 1A. RISK FACTORS")

    selected = _select_ordered_item_markers(
        content,
        item_pattern=pattern,
        ordered_tokens=("1A", "2"),
    )

    assert len(selected) == 2
    assert selected[0] == ("1A", heading_pos)
    assert selected[0][1] != inline_pos


@pytest.mark.unit
def test_select_ordered_item_markers_after_toc_prefers_post_toc_sequence() -> None:
    """验证 TOC 后 Item 序列充足时优先使用正文序列。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = (
        "Table of Contents "
        "Item 1. Toc "
        "Item 2. Toc "
        "Item 3. Toc "
        + ("x" * 1800)
        + " Item 1. Business details "
        + "long body text " * 40
        + "Item 2. Properties details "
        + "long body text " * 40
        + "Item 3. Legal Proceedings details "
        + "long body text " * 40
        + "Item 4. Mine Safety details "
    )
    pattern = re.compile(r"(?i)\bitem\s+([1-4])\s*[\.\:\-]")
    selected = _select_ordered_item_markers_after_toc(
        content,
        item_pattern=pattern,
        ordered_tokens=("1", "2", "3", "4"),
        min_items_after_toc=3,
    )
    assert len(selected) >= 3
    assert selected[0][1] > content.lower().find("table of contents")


@pytest.mark.unit
def test_select_after_toc_prefers_non_toc_candidate_when_coverage_equal() -> None:
    """验证当覆盖度相同且存在正文候选时，优先选择非 ToC 候选。

    构造场景：
    - 默认候选（start=0）和 TOC 后候选都可命中完整 Item 序列；
    - 默认候选落在目录区（标题+页码短 span）；
    - 正文候选位于后续正文区域。

    期望：应选择正文候选，避免回退到目录候选。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    toc = (
        "Table of Contents "
        "Item 1. Business 3 "
        "Item 2. Properties 9 "
        "Item 3. Legal Proceedings 15 "
        "Item 4. Mine Safety 20 "
    )
    body = (
        ("x" * 2200)
        + "\nItem 1. Business "
        + ("body " * 400)
        + "\nItem 2. Properties "
        + ("body " * 400)
        + "\nItem 3. Legal Proceedings "
        + ("body " * 400)
        + "\nItem 4. Mine Safety "
        + ("body " * 200)
    )
    content = toc + body
    pattern = re.compile(r"(?i)\bitem\s+([1-4])\s*[\.\:\-]")

    selected = _select_ordered_item_markers_after_toc(
        content,
        item_pattern=pattern,
        ordered_tokens=("1", "2", "3", "4"),
        min_items_after_toc=3,
    )

    assert len(selected) == 4
    assert selected[0][1] > len(toc)


@pytest.mark.unit
def test_select_after_toc_without_explicit_toc_keeps_default_when_skip_drops_too_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证无显式 ToC 时，异常降级跳过结果不会覆盖高覆盖默认结果。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    content = (
        "Item 1. Alpha section body. "
        "Item 2. Beta section body. "
        "Item 3. Gamma section body. "
        "Item 4. Delta section body. "
        "Item 5. Epsilon section body. "
        "Item 6. Zeta section body. "
    )
    pattern = re.compile(r"(?i)\bitem\s+([1-6])\s*[\.\:\-]")
    degraded = [
        ("3", content.index("Item 3.")),
        ("4", content.index("Item 4.")),
        ("5", content.index("Item 5.")),
        ("6", content.index("Item 6.")),
    ]

    monkeypatch.setattr(
        "dayu.fins.processors.sec_report_form_common._skip_toc_like_markers",
        lambda *args, **kwargs: degraded,
    )

    selected = _select_ordered_item_markers_after_toc(
        content,
        item_pattern=pattern,
        ordered_tokens=("1", "2", "3", "4", "5", "6"),
        min_items_after_toc=4,
    )

    assert len(selected) >= 6
    assert selected[0][0] == "1"
    assert selected[0][1] == content.index("Item 1.")


@pytest.mark.unit
def test_rank_marker_candidate_prefers_more_complete_and_better_start() -> None:
    """验证 marker 候选评分在完整度与起点位置上的排序逻辑。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    default_before_toc = [("1", 100), ("2", 200), ("3", 300), ("4", 400)]
    cutoff_candidate = [("1", 4000), ("2", 5000), ("3", 6000), ("4", 7000)]
    cluster_candidate = [("1", 3500), ("2", 4500), ("3", 5500), ("4", 6500)]

    # default 候选在 toc_start 之前，应该被降权
    rank_default = _rank_marker_candidate(default_before_toc, toc_start=1000)
    rank_cutoff = _rank_marker_candidate(cutoff_candidate, toc_start=1000)
    rank_cluster = _rank_marker_candidate(cluster_candidate, toc_start=1000)

    assert rank_default < rank_cutoff
    # 同样长度时，起点更靠前（3500）优于更靠后（4000）
    assert rank_cluster > rank_cutoff


@pytest.mark.unit
def test_markers_look_like_toc_entries_detects_compact_spans() -> None:
    """验证 ToC 条目检测能识别密集短间距 markers。

    当选中的 markers 之间文本跨度大部分低于阈值时，
    应判定为 ToC 区域。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 模拟 ToC 区域：每个 marker 间只有 ~60 字符（标题 + 页码）
    full_text = (
        "Item 1. Business 4 "
        "Item 1A. Risk Factors 15 "
        "Item 2. Properties 27 "
        "Item 7. MD&A 35 "
        "Item 8. Financial Statements 45 "
        + "x" * 100000  # 大量正文在最后
    )
    toc_markers = [
        ("1", 0),
        ("1A", 20),
        ("2", 46),
        ("7", 67),
        ("8", 84),
    ]
    assert _markers_look_like_toc_entries(full_text, toc_markers) is True


@pytest.mark.unit
def test_markers_look_like_toc_entries_accepts_real_content() -> None:
    """验证 ToC 条目检测不误判正文区域的 markers。

    正文区域的 markers 之间有大量文本，不应被误判为 ToC 区域。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    body = "detailed content text " * 100  # 每段 ~2200 字符
    full_text = (
        f"Item 1. Business {body}"
        f"Item 1A. Risk Factors {body}"
        f"Item 2. Properties {body}"
        f"Item 7. MD&A {body}"
        f"Item 8. Financial Statements {body}"
    )
    # 手动计算粗略位置（每段 ~2200+ 字符）
    content_markers = [
        ("1", 0),
        ("1A", 2220),
        ("2", 4450),
        ("7", 6670),
        ("8", 8890),
    ]
    assert _markers_look_like_toc_entries(full_text, content_markers) is False


@pytest.mark.unit
def test_markers_look_like_toc_entries_handles_few_markers() -> None:
    """验证 markers 数量不足时返回 False（无法判定）。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _markers_look_like_toc_entries("text", []) is False
    assert _markers_look_like_toc_entries("text", [("1", 0)]) is False
    assert _markers_look_like_toc_entries("text", [("1", 0), ("2", 10)]) is False


@pytest.mark.unit
def test_markers_look_like_toc_entries_consecutive_start_pattern() -> None:
    """验证连续短 span 检测能识别部分 ToC + 部分正文的混合场景。

    模拟 AXON 2026 类型场景：文档开头有 ToC 列表（5+ 连续短 span），
    之后是正文（大 span），整体短 span 比例低于 80%。
    连续短 span 检测应能捕获此模式。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 构造：开头 6 个连续短 span（ToC），后面 3 个长 span（正文）
    # 短 span 比例 = 6/9 ≈ 67% < 80%（全局比例检测不触发）
    # 但连续数 = 6 ≥ 5（连续检测触发）
    full_text = "x" * 200000
    mixed_markers = [
        ("1", 0),       # → span 30
        ("1A", 30),     # → span 25
        ("1B", 55),     # → span 20
        ("2", 75),      # → span 30
        ("3", 105),     # → span 25
        ("4", 130),     # → span 20
        ("5", 150),     # → span 50000 (正文)
        ("7", 50150),   # → span 40000 (正文)
        ("8", 90150),   # → span 30000 (正文)
        ("9", 120150),  # 最后一个
    ]
    assert _markers_look_like_toc_entries(full_text, mixed_markers) is True


@pytest.mark.unit
def test_markers_look_like_toc_entries_no_false_positive_scattered_short() -> None:
    """验证散布的短 section 不被误判为 ToC。

    正文中可能有少量短 section（如 "None" 或引用语句），
    但它们散布在长 section 之间，不应被判定为 ToC。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    full_text = "x" * 200000
    # 模式：长-短-长-短-长-长-短-长-长
    # 短 span 3/8 = 37.5%，且不连续
    scattered_markers = [
        ("1", 0),         # → span 10000 (长)
        ("1A", 10000),    # → span 50 (短: "None")
        ("1B", 10050),    # → span 15000 (长)
        ("2", 25050),     # → span 100 (短: proxy ref)
        ("3", 25150),     # → span 20000 (长)
        ("7", 45150),     # → span 25000 (长)
        ("7A", 70150),    # → span 200 (短: brief disclosure)
        ("8", 70350),     # → span 50000 (长)
        ("9", 120350),    # 最后一个
    ]
    assert _markers_look_like_toc_entries(full_text, scattered_markers) is False


@pytest.mark.unit
def test_select_after_toc_skips_large_toc_via_adaptive_retry() -> None:
    """验证自适应 ToC 跳过逻辑能处理超大 ToC 区域。

    模拟 AMZN/AXON 场景：ToC 区域远超 1500 字符，标准
    cutoff buffer 不足以跳过全部 ToC 条目。验证新增的
    质量校验 + 重试机制能正确找到正文 Item 标记。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    body = "substantive paragraph content " * 50  # 每段 ~1500 字符
    # 构造一个大型 ToC（远超 1500 字符）
    toc_section = (
        "TABLE OF CONTENTS\n"
        "Item 1. Business 4\n"
        "Item 1A. Risk Factors 15\n"
        "Item 1B. Unresolved Staff Comments 27\n"
        "Item 2. Properties 28\n"
        "Item 3. Legal Proceedings 30\n"
        "Item 4. Mine Safety Disclosures 32\n"
        "Item 5. Market for Registrant's Common Equity 33\n"
        "Item 6. Reserved 35\n"
        "Item 7. Management's Discussion and Analysis of Financial Condition 36\n"
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk 50\n"
        "Item 8. Financial Statements and Supplementary Data 52\n"
        "Item 9. Changes in and Disagreements With Accountants 80\n"
        "Item 9A. Controls and Procedures 81\n"
        "Item 9B. Other Information 82\n"
        "Item 10. Directors, Executive Officers and Corporate Governance 83\n"
        "Item 11. Executive Compensation 84\n"
        "Item 12. Security Ownership 85\n"
        "Item 13. Certain Relationships and Related Transactions 86\n"
        "Item 14. Principal Accounting Fees and Services 87\n"
        "Item 15. Exhibits and Financial Statement Schedules 88\n"
        + "x" * 2000  # 额外填充，确保 ToC 总长度远超 1500
    )
    real_content = (
        f"Item 1. Business\n{body}"
        f"Item 1A. Risk Factors\n{body}"
        f"Item 2. Properties\n{body}"
        f"Item 7. Management's Discussion and Analysis\n{body}"
        f"Item 8. Financial Statements\n{body}"
        f"Item 15. Exhibits\n{body}"
    )
    full_text = toc_section + real_content

    pattern = re.compile(
        r"(?i)\bitem\s+(1A|1B|1C|7A|9A|9B|9C|1[0-5]|[1-9])\s*[\.\:\-]"
    )
    ordered = ("1", "1A", "1B", "2", "3", "4", "5", "6", "7", "7A", "8",
               "9", "9A", "9B", "10", "11", "12", "13", "14", "15")
    selected = _select_ordered_item_markers_after_toc(
        full_text,
        item_pattern=pattern,
        ordered_tokens=ordered,
        min_items_after_toc=4,
    )

    # 验证选出的 markers 在 ToC 区域之后（正文位置）
    toc_end_approx = len(toc_section)
    assert len(selected) >= 4, f"应选出至少 4 个 Item，实际 {len(selected)}"
    assert selected[0][1] >= toc_end_approx - 100, (
        f"第一个 Item 应在正文区域（>= {toc_end_approx - 100}），"
        f"实际位置 {selected[0][1]}"
    )

    # 验证各段内容长度合理（不是 ToC 条目的 ~50 字符）
    for i in range(len(selected) - 1):
        span = selected[i + 1][1] - selected[i][1]
        assert span > 500, (
            f"Item {selected[i][0]} → {selected[i + 1][0]} 跨度仅 {span} 字符，"
            f"疑似仍在 ToC 区域"
        )


@pytest.mark.unit
def test_select_after_toc_handles_no_toc_token_with_toc_like_markers() -> None:
    """验证无 "table of contents" 标记但有隐式 ToC 区域的场景。

    部分 filing 在文档开头有 ToC 列表但无显式 "Table of Contents" 标题。
    验证自适应检测能识别并跳过这些隐式 ToC 区域。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    body = "substantive paragraph content " * 50  # 每段 ~1500 字符
    # 无 "table of contents" 标记，但有密集的 Item 列表（隐式 ToC）
    implicit_toc = (
        "Item 1. Business 4\n"
        "Item 1A. Risk Factors 15\n"
        "Item 2. Properties 28\n"
        "Item 7. MD&A 36\n"
        "Item 8. Financial Statements 52\n"
    )
    real_content = (
        f"Item 1. Business\n{body}"
        f"Item 1A. Risk Factors\n{body}"
        f"Item 2. Properties\n{body}"
        f"Item 7. MD&A\n{body}"
        f"Item 8. Financial Statements\n{body}"
    )
    full_text = implicit_toc + real_content

    pattern = re.compile(
        r"(?i)\bitem\s+(1A|1B|1C|7A|9A|9B|9C|1[0-5]|[1-9])\s*[\.\:\-]"
    )
    ordered = ("1", "1A", "2", "7", "8")
    selected = _select_ordered_item_markers_after_toc(
        full_text,
        item_pattern=pattern,
        ordered_tokens=ordered,
        min_items_after_toc=4,
    )

    # 应选到正文区域的 markers，而非 ToC 条目
    toc_end_approx = len(implicit_toc)
    assert len(selected) >= 4, f"应选出至少 4 个 Item，实际 {len(selected)}"
    assert selected[0][1] >= toc_end_approx - 50, (
        f"第一个 Item 应在正文区域（>= {toc_end_approx - 50}），"
        f"实际位置 {selected[0][1]}"
    )


@pytest.mark.unit
def test_ten_k_processor_splits_part_item_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-K 处理器按 Part+Item 切分章节。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10k_sections.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    text_value = (
        "Table of Contents "
        "Item 1. TOC "
        "Item 1A. TOC "
        + ("x" * 1700)
        + "Part I "
        + "Item 1. Business description and strategy details. "
        + "This section contains extensive narrative text " * 20
        + "Item 1A. Risk Factors and mitigation plans. "
        + "More narrative text for risk factors " * 20
        + "Part II "
        + "Item 7. Management's Discussion and Analysis. "
        + "Detailed analysis text for MD&A section " * 20
        + "Item 8. Financial Statements and Supplementary Data. "
        + "Detailed financial statements narrative " * 20
        + "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenKFormProcessor(
        _make_source(source_path),
        form_type="10-K",
        media_type="text/html",
    )
    sections = processor.list_sections()
    titles = [str(item.get("title") or "") for item in sections]

    assert any(title == "Part I - Item 1" for title in titles)
    assert any(title == "Part I - Item 1A" for title in titles)
    assert any(title == "Part II - Item 7" for title in titles)
    assert any(title == "Part II - Item 8" for title in titles)
    assert any(title == "SIGNATURE" for title in titles)


@pytest.mark.unit
def test_ten_q_processor_splits_part_i_and_part_ii_sections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-Q 处理器按 Part I/II 切分章节。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10q_sections.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    text_value = (
        "Part I "
        "Item 1. Financial Statements for the quarter. "
        "Narrative details for financial statements. "
        "Item 2. Management's Discussion and Analysis. "
        "Narrative details for MD&A. "
        "Part II "
        "Item 1A. Risk Factors updates. "
        "Narrative risk update details. "
        "Item 6. Exhibits. "
        "Additional exhibits details. "
        "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenQFormProcessor(
        _make_source(source_path),
        form_type="10-Q",
        media_type="text/html",
    )
    sections = processor.list_sections()
    titles = [str(item.get("title") or "") for item in sections]

    assert any(title == "Part I - Item 1" for title in titles)
    assert any(title == "Part I - Item 2" for title in titles)
    assert any(title == "Part II - Item 1A" for title in titles)
    assert any(title == "Part II - Item 6" for title in titles)
    assert any(title == "SIGNATURE" for title in titles)


@pytest.mark.unit
def test_ten_k_processor_supports_item_heading_without_punctuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-K 支持无标点的 Item 标题格式。

    场景示例：``Item 1 Business``（无 ``.:-``）。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10k_no_punctuation.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Detailed narrative content. " * 120
    text_value = (
        "Part I "
        f"Item 1 Business {body}"
        f"Item 1A Risk Factors {body}"
        "Part II "
        f"Item 7 Management Discussion and Analysis {body}"
        f"Item 8 Financial Statements and Supplementary Data {body}"
        "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenKFormProcessor(
        _make_source(source_path),
        form_type="10-K",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert any(title == "Part I - Item 1" for title in titles)
    assert any(title == "Part I - Item 1A" for title in titles)
    assert any(title == "Part II - Item 7" for title in titles)
    assert any(title == "Part II - Item 8" for title in titles)
    assert any(title == "SIGNATURE" for title in titles)


@pytest.mark.unit
def test_ten_k_processor_trailing_toc_triggers_heading_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-K 尾部 Item 索引不被误识别为正文标记。

    场景：MCD 等嵌入式年报，正文以自然标题标识章节（Risk Factors 等），
    尾部附简短 Item 索引（Item 1 Business Pages 3-7 ..."）。所有 Item 正则
    命中集中在文档末尾极小区间内（< 5%），应触发标题兜底而非产出仅含目录行的章节。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10k_trailing_toc.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Detailed narrative content. " * 120
    # 正文使用自然标题（无 Item 前缀），内容充实
    text_value = (
        "ANNUAL REPORT\n"
        + body
        + "\nBusiness\n"
        + body
        + "\nRisk Factors\n"
        + body
        + "\nManagement's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + body
        + "\nFinancial Statements and Supplementary Data\n"
        + body
        # 尾部 Item 索引：所有 Item 正则命中聚集在此极小区间
        + "\nItem 1. Business Pages 3-7\n"
        + "Item 1A. Risk Factors Pages 8-20\n"
        + "Item 2. Properties Page 21\n"
        + "Item 3. Legal Proceedings Page 22\n"
        + "Item 7. Management's Discussion and Analysis Pages 23-40\n"
        + "Item 7A. Quantitative and Qualitative Disclosures About Market Risk Page 41\n"
        + "Item 8. Financial Statements Pages 42-90\n"
        + "Item 9. Changes in and Disagreements Page 91\n"
        + "Item 9A. Controls and Procedures Page 92\n"
        + "Item 10. Directors Page 93\n"
        + "Item 11. Executive Compensation Page 94\n"
        + "Item 12. Security Ownership Page 95\n"
        + "Item 13. Certain Relationships Page 96\n"
        + "Item 14. Principal Accountant Page 97\n"
        + "Item 15. Exhibits Page 98\n"
        + "\nSIGNATURE\n"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenKFormProcessor(
        _make_source(source_path),
        form_type="10-K",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    # 应通过标题兜底识别正文中的关键 Item，而非尾部索引
    assert any(title == "Part I - Item 1A" for title in titles)
    assert any(title == "Part II - Item 7" for title in titles)
    assert any(title == "Part II - Item 8" for title in titles)

    # 验证章节内容为正文而非目录行
    for section in processor.list_sections():
        title = section.get("title") or ""
        if "Item 1A" in title:
            content = processor.read_section(section["ref"])
            # 正文内容应远长于目录行
            assert len(content.get("content", "")) > 200, (
                f"Item 1A 内容仅 {len(content.get('content', ''))} chars，疑似目录行"
            )
            break


@pytest.mark.unit
def test_ten_k_processor_trailing_toc_keeps_original_when_heading_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证尾部 Item 索引检测后、标题兜底失败时保留原始 markers。

    场景：C / GE / INTC 等公司的 10-K 有尾部 Item 索引，
    但正文不含独立标题行（如 Risk Factors 不是独立行），
    标题兜底无法命中必需的 1A/7/8。此时应保留原始 Item markers
    继续正常 repair 流程，而非清空导致处理降级。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10k_trailing_no_headings.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    # 超长正文（无独立标题行）
    body = "Detailed corporate content. " * 500
    # 尾部 Item 索引（聚簇在最后一小段）
    text_value = (
        "COVER PAGE\n"
        + body
        + "\nItem 1. Business Pages 3-7\n"
        + "Item 1A. Risk Factors Pages 8-20\n"
        + "Item 2. Properties Page 21\n"
        + "Item 3. Legal Proceedings Page 22\n"
        + "Item 4. Mine Safety Page 23\n"
        + "Item 5. Market for Registrant's Common Equity Page 24\n"
        + "Item 6. Reserved Page 25\n"
        + "Item 7. Management's Discussion and Analysis Pages 26-40\n"
        + "Item 7A. Quantitative and Qualitative Disclosures About Market Risk Page 41\n"
        + "Item 8. Financial Statements Pages 42-90\n"
        + "Item 9. Changes in and Disagreements Page 91\n"
        + "Item 9A. Controls and Procedures Page 92\n"
        + "Item 10. Directors Page 93\n"
        + "\nSIGNATURE\n"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenKFormProcessor(
        _make_source(source_path),
        form_type="10-K",
        media_type="text/html",
    )
    sections = processor.list_sections()
    titles = [str(item.get("title") or "") for item in sections]

    # 原始 Item markers 应保留，确保有 Item 标题（而非退化为 1 section）
    assert len(sections) >= 5, f"应至少有 5 个 Item 章节，实际 {len(sections)}"
    assert any("Item 1A" in t for t in titles), f"Item 1A 缺失: {titles}"
    assert any("Item 7" in t for t in titles), f"Item 7 缺失: {titles}"
    assert any("Item 8" in t for t in titles), f"Item 8 缺失: {titles}"


@pytest.mark.unit
def test_ten_k_processor_expands_heading_stub_sections_with_body_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-K 会把标题/页码 stub 扩展为真正正文。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10k_heading_stub_fix.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Detailed operating discussion. " * 120
    text_value = (
        "TABLE OF CONTENTS\n"
        "Item 1A. Risk Factors 24\n"
        "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations 33-41\n"
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk 42\n"
        "Item 8. Financial Statements and Supplementary Data 50\n"
        + ("preface " * 300)
        + "\nRisk Factors\n"
        + ("Risk factor body. " * 120)
        + "\nManagement's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + body
        + "\nQuantitative and Qualitative Disclosures About Market Risk\n"
        + ("Market risk body. " * 120)
        + "\nFinancial Statements and Supplementary Data\n"
        + ("Financial statement body. " * 160)
        + "\nSIGNATURE\n"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenKFormProcessor(
        _make_source(source_path),
        form_type="10-K",
        media_type="text/html",
    )

    item7_ref = next(
        section["ref"]
        for section in processor.list_sections()
        if section.get("title") == "Part II - Item 7"
    )
    item7_payload = processor.read_section(str(item7_ref))

    assert "Detailed operating discussion." in item7_payload["content"]
    assert len(item7_payload["content"].split()) > 200


@pytest.mark.unit
def test_ten_k_processor_expands_by_reference_sections_from_same_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-K by-reference 包装句会借用同文档被引用正文。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10k_by_reference_fix.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    risk_body = "Enterprise risk narrative. " * 110
    mdna_body = "Loan growth and fee revenue analysis. " * 140
    risk_profile_body = "Interest rate sensitivity and market risk analysis. " * 110
    fs_body = "Consolidated balance sheet and notes discussion. " * 150
    text_value = (
        "ANNUAL REPORT\n"
        + ("opening narrative " * 120)
        + "\nRisk Factors\n"
        + risk_body
        + "\nManagement's Discussion and Analysis\n"
        + mdna_body
        + "\nCorporate Risk Profile\n"
        + risk_profile_body
        + "\nFinancial Statements\n"
        + fs_body
        + "\nPart I\n"
        + "Item 1A. Risk Factors\n"
        + "Information in response to this Item 1A can be found in the 2023 Annual Report on pages 140 to 155 under the heading “Risk Factors.” That information is incorporated into this report by reference.\n"
        + "Part II\n"
        + "Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + "Information in response to this Item 7 can be found in the 2023 Annual Report on pages 22 to 58 under the heading “Management's Discussion and Analysis.” That information is incorporated into this report by reference.\n"
        + "Item 7A. Quantitative and Qualitative Disclosures About Market Risk\n"
        + "Information in response to this Item 7A can be found in the 2023 Annual Report on pages 35 to 55 under the heading “Corporate Risk Profile.” That information is incorporated into this report by reference.\n"
        + "Item 8. Financial Statements and Supplementary Data\n"
        + "Information in response to this Item 8 can be found in the 2023 Annual Report on pages 64 to 139 under the headings “Financial Statements” and “Notes to Financial Statements.” That information is incorporated into this report by reference.\n"
        + "SIGNATURE\n"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenKFormProcessor(
        _make_source(source_path),
        form_type="10-K",
        media_type="text/html",
    )

    payloads = {
        section["title"]: processor.read_section(str(section["ref"]))
        for section in processor.list_sections()
        if section.get("title") in {
            "Part I - Item 1A",
            "Part II - Item 7",
            "Part II - Item 8",
        }
    }

    assert "Enterprise risk narrative." in payloads["Part I - Item 1A"]["content"]
    assert "Loan growth and fee revenue analysis." in payloads["Part II - Item 7"]["content"]
    assert "Consolidated balance sheet and notes discussion." in payloads["Part II - Item 8"]["content"]


@pytest.mark.unit
def test_ten_k_processor_supports_heading_only_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-K 在缺失 Item 前缀时可通过法定标题兜底识别关键 Item。

    场景示例：正文只有 ``Business`` / ``Risk Factors`` / ``Management's Discussion`` /
    ``Financial Statements and Supplementary Data`` 标题，不包含 ``Item 1A/7/8`` 字样。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10k_heading_only.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Detailed narrative content. " * 120
    text_value = (
        "Table of Contents "
        "Risk Factors 1A 13 "
        "Management's Discussion and Analysis of Financial Condition and Results of Operations 7 25 "
        "Financial Statements and Supplementary Data 8 76 "
        + ("x" * 4000)
        + "\nBusiness\n"
        + body
        + "\nRisk Factors\n"
        + body
        + "\nManagement's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + body
        + "\nFinancial Statements and Supplementary Data\n"
        + body
        + "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenKFormProcessor(
        _make_source(source_path),
        form_type="10-K",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert any(title == "Part I - Item 1A" for title in titles)
    assert any(title == "Part II - Item 7" for title in titles)
    assert any(title == "Part II - Item 8" for title in titles)
    assert any(title == "SIGNATURE" for title in titles)


@pytest.mark.unit
def test_ten_k_processor_fills_missing_item_1a_from_heading_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-K 在 Item 1A 缺失时可用 ``Risk Factors`` 标题补齐。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10k_missing_1a_with_heading.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Detailed narrative content. " * 120
    text_value = (
        "Part I\n"
        "Item 1. Business\n"
        + body
        + "\nRisk Factors\n"
        + body
        + "\nItem 2. Properties\n"
        + body
        + "\nPart II\n"
        + "\nItem 7. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + body
        + "\nItem 7A. Quantitative and Qualitative Disclosures About Market Risk\n"
        + body
        + "\nItem 8. Financial Statements and Supplementary Data\n"
        + body
        + "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenKFormProcessor(
        _make_source(source_path),
        form_type="10-K",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert any(title == "Part I - Item 1A" for title in titles)
    assert any(title == "Part II - Item 7" for title in titles)
    assert any(title == "Part II - Item 8" for title in titles)


@pytest.mark.unit
def test_ten_q_processor_supports_numbered_headings_without_item_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-Q 支持 ``1. Financial Statements`` 这类纯编号标题。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10q_numbered_heading.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Quarterly narrative content. " * 100
    text_value = (
        "PART I — FINANCIAL INFORMATION\n"
        "1. Financial Statements\n" + body +
        "\n2. Management's Discussion and Analysis\n" + body +
        "\n3. Quantitative and Qualitative Disclosures About Market Risk\n" + body +
        "\n4. Controls and Procedures\n" + body +
        "\nPART II — OTHER INFORMATION\n"
        "1A. Risk Factors\n" + body +
        "\n6. Exhibits\n" + body +
        "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenQFormProcessor(
        _make_source(source_path),
        form_type="10-Q",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert any(title == "Part I - Item 1" for title in titles)
    assert any(title == "Part I - Item 2" for title in titles)
    assert any(title == "Part II - Item 1A" for title in titles)
    assert any(title == "Part II - Item 6" for title in titles)
    assert any(title == "SIGNATURE" for title in titles)


@pytest.mark.unit
def test_ten_q_processor_repairs_part_i_item_1_from_heading_when_toc_polluted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-Q 在 Part I Item 1 命中目录行时可回退到正文标题。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10q_item1_toc_polluted.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Quarterly statement details. " * 160
    text_value = (
        "PART I — FINANCIAL INFORMATION\n"
        "Item 1. Financial Statements and Supplementary Data 3\n"
        "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations 21\n"
        + ("x" * 3000)
        + "\nFinancial Statements and Supplementary Data\n"
        + body
        + "\nItem 2. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + body
        + "\nPART II — OTHER INFORMATION\n"
        + "\nItem 6. Exhibits\n"
        + body
        + "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenQFormProcessor(
        _make_source(source_path),
        form_type="10-Q",
        media_type="text/html",
    )
    sections = processor.list_sections()

    item_1_ref = None
    for section in sections:
        if str(section.get("title") or "") == "Part I - Item 1":
            item_1_ref = section.get("ref")
            break
    assert item_1_ref is not None

    item_1_payload = processor.read_section(str(item_1_ref))
    item_1_content = str(item_1_payload.get("content") or "")
    assert "Quarterly statement details." in item_1_content
    assert " 3" not in item_1_content[:80]

    item_2_ref = None
    for section in sections:
        if str(section.get("title") or "") == "Part I - Item 2":
            item_2_ref = section.get("ref")
            break
    assert item_2_ref is not None
    item_2_payload = processor.read_section(str(item_2_ref))
    item_2_content = str(item_2_payload.get("content") or "")
    assert "Quarterly statement details." in item_2_content


@pytest.mark.unit
def test_ten_q_processor_repairs_item_1_when_toc_entries_are_inline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 10-Q 可识别“同一行目录条目”并修复 Part I Item 1。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10q_item1_toc_inline.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Quarterly statement details. " * 140
    text_value = (
        "PART I — FINANCIAL INFORMATION\n"
        "Item 1. Financial Statements and Supplementary Data 3 "
        "Item 2. Management's Discussion and Analysis of Financial Condition and Results of Operations 21 "
        + ("x" * 2500)
        + "\nFinancial Statements and Supplementary Data\n"
        + body
        + "\nItem 2. Management's Discussion and Analysis of Financial Condition and Results of Operations\n"
        + body
        + "\nPART II — OTHER INFORMATION\n"
        + "\nItem 6. Exhibits\n"
        + body
        + "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TenQFormProcessor(
        _make_source(source_path),
        form_type="10-Q",
        media_type="text/html",
    )
    sections = processor.list_sections()

    item_1_ref = None
    for section in sections:
        if str(section.get("title") or "") == "Part I - Item 1":
            item_1_ref = section.get("ref")
            break
    assert item_1_ref is not None

    item_1_payload = processor.read_section(str(item_1_ref))
    item_1_content = str(item_1_payload.get("content") or "")
    assert "Quarterly statement details." in item_1_content

    item_2_ref = None
    for section in sections:
        if str(section.get("title") or "") == "Part I - Item 2":
            item_2_ref = section.get("ref")
            break
    assert item_2_ref is not None
    item_2_payload = processor.read_section(str(item_2_ref))
    item_2_content = str(item_2_payload.get("content") or "")
    assert "Quarterly statement details." in item_2_content


@pytest.mark.unit
def test_twenty_f_processor_ignores_cover_checkbox_item_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 20-F 不会将封面勾选框 ``Item 18 ☐`` 误识别为正文 Item。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "20f_checkbox_item18.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Annual report detailed discussion. " * 120
    text_value = (
        "Cover Page Item 17 ☐ Item 18 ☐ "
        f"Item 3 Key Information {body}"
        f"Item 4 Information on the Company {body}"
        f"Item 5 Operating and Financial Review and Prospects {body}"
        f"Item 18 Financial Statements {body}"
        f"Item 19 Exhibits {body}"
        "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TwentyFFormProcessor(
        _make_source(source_path),
        form_type="20-F",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert any("Part I - Item 3" in title for title in titles)
    assert any("Part II - Item 5" in title for title in titles)
    item_18_titles = [title for title in titles if "Item 18" in title]
    assert len(item_18_titles) == 1
    assert "Part IV - Item 18 - Financial Statements" in item_18_titles[0]


@pytest.mark.unit
def test_twenty_f_processor_supports_numbered_headings_without_item_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 20-F 支持 ``3. Key Information`` 这类纯编号标题。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "20f_numbered_heading.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Annual report detailed discussion. " * 120
    text_value = (
        "Table of Contents "
        "3. Key Information ... 10 "
        "5. Operating and Financial Review and Prospects ... 30 "
        "18. Financial Statements ... 120 "
        + ("x" * 3000)
        + "\n3. Key Information\n"
        + body
        + "\n5. Operating and Financial Review and Prospects\n"
        + body
        + "\n18. Financial Statements\n"
        + body
        + "\n19. Exhibits\n"
        + body
        + "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TwentyFFormProcessor(
        _make_source(source_path),
        form_type="20-F",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert any(title.startswith("Part I - Item 3") for title in titles)
    assert any(title.startswith("Part II - Item 5") for title in titles)
    assert any(title.startswith("Part IV - Item 18") for title in titles)
    assert any(title.startswith("Part IV - Item 19") for title in titles)
    assert "SIGNATURE" in titles


@pytest.mark.unit
def test_twenty_f_processor_repairs_item_18_when_toc_entry_is_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 20-F 在 Item 18 命中目录行时可回退到正文标题。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "20f_item18_toc_polluted.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    body = "Annual report detailed discussion. " * 140
    text_value = (
        "Table of Contents "
        "Item 3. Key Information 10 "
        "Item 5. Operating and Financial Review and Prospects 30 "
        "Item 18. Financial Statements 180 "
        "Item 19. Exhibits 185 "
        + ("x" * 3200)
        + "\nItem 3. Key Information\n"
        + body
        + "\nItem 5. Operating and Financial Review and Prospects\n"
        + body
        + "\nFinancial Statements\n"
        + body
        + "\nItem 19. Exhibits\n"
        + body
        + "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TwentyFFormProcessor(
        _make_source(source_path),
        form_type="20-F",
        media_type="text/html",
    )
    sections = processor.list_sections()
    item_18_ref = None
    for section in sections:
        title = str(section.get("title") or "")
        if title.startswith("Part IV - Item 18"):
            item_18_ref = section.get("ref")
            break
    assert item_18_ref is not None
    payload = processor.read_section(str(item_18_ref))
    content = str(payload.get("content") or "")
    assert "Annual report detailed discussion." in content
    assert " 180 " not in content[:120]


@pytest.mark.unit
def test_twenty_f_processor_splits_items_with_part_and_description(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 20-F 处理器按 Item 切分，并附加 Part 标签和 SEC 描述。

    v2 策略不再生成 IFRS 语义标题，改为在 Item 标题中内嵌 Part 和描述。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "20f_sections.html"
    source_path.write_text("<html><body>placeholder</body></html>", encoding="utf-8")
    # 填充足够内容防止被 ToC 检测误判
    padding = "Detailed content for analysis. " * 40
    text_value = (
        f"Item 3. Key Information section details. {padding}"
        f"Item 4. Information on the Company section details. {padding}"
        f"Item 5. Operating and Financial Review and Prospects details. {padding}"
        f"Item 18. Financial Statements under IFRS. {padding}"
        "SIGNATURE The registrant hereby certifies."
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(text_value),
    )

    processor = TwentyFFormProcessor(
        _make_source(source_path),
        form_type="20-F",
        media_type="text/html",
    )
    sections = processor.list_sections()
    titles = [str(item.get("title") or "") for item in sections]

    # Item 3 应包含 Part I 和描述
    assert any("Part I" in t and "Item 3" in t and "Key Information" in t for t in titles)
    # Item 5 应包含 Part II
    assert any("Part II" in t and "Item 5" in t for t in titles)
    # Item 18 应包含 Part IV 和描述
    assert any("Part IV" in t and "Item 18" in t and "Financial Statements" in t for t in titles)
    # SIGNATURE 应存在
    assert any(t == "SIGNATURE" for t in titles)


@pytest.mark.unit
def test_twenty_f_processor_prefers_source_text_when_it_recovers_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 20-F 在解析文本压平失效时会改用保留行边界的源文本。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "20f_source_text_recovery.html"
    source_path.write_text(
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
    flattened_text = (
        "Annual report detailed discussion. " * 40
        + "SIGNATURE"
    )
    monkeypatch.setattr(
        "dayu.fins.processors.sec_processor._parse_document",
        lambda html_content, form_type: _FakeDocument(flattened_text),
    )

    processor = TwentyFFormProcessor(
        _make_source(source_path),
        form_type="20-F",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert any(title.startswith("Part I - Item 3") for title in titles)
    assert any(title.startswith("Part II - Item 5") for title in titles)
    assert any(title.startswith("Part IV - Item 18") for title in titles)


@pytest.mark.unit
def test_trim_twenty_f_source_text_removes_xbrl_preamble_noise() -> None:
    """验证 20-F 原始文本会裁掉前置 XBRL 机器噪声。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    xbrl_prefix = "\n".join(
        f"ifrs-full:Member{index}\nshg:DimensionMember{index}\n2023-12-31\n0001263043"
        for index in range(1600)
    )
    report_text = (
        "Commission file number 001-31529\n"
        "Annual report pursuant to section 13 or 15(d) of the Securities Exchange Act of 1934\n"
        "Securities registered or to be registered pursuant to Section 12(b) of the Act\n"
        "ITEM 1.\nIdentity of Directors, Senior Management and Advisers\n"
    )

    trimmed = _trim_twenty_f_source_text(xbrl_prefix + ("\n" * 8) + report_text)

    assert trimmed.startswith("Commission file number 001-31529")
    assert "ifrs-full:Member0" not in trimmed


# ────────────────────────────────────────────────────────────────
# Step 8 – Part 标签修正（SEC 监管规则）
# ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_correct_part_from_sec_rules_fills_missing() -> None:
    """验证缺失 Part 标签时用 SEC 规则补全。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    from dayu.fins.processors.ten_k_form_common import _correct_part_from_sec_rules

    # Item 1 → Part I（缺失时补全）
    assert _correct_part_from_sec_rules("1", None) == "Part I"
    assert _correct_part_from_sec_rules("1A", None) == "Part I"
    # Items 2, 3, 4 → Part I（SEC Regulation S-K: Properties, Legal Proceedings, Mine Safety）
    assert _correct_part_from_sec_rules("2", None) == "Part I"
    assert _correct_part_from_sec_rules("3", None) == "Part I"
    assert _correct_part_from_sec_rules("4", None) == "Part I"
    # Item 5 → Part II
    assert _correct_part_from_sec_rules("5", None) == "Part II"
    # Item 7 → Part II
    assert _correct_part_from_sec_rules("7", None) == "Part II"
    assert _correct_part_from_sec_rules("7A", None) == "Part II"
    # Item 10 → Part III
    assert _correct_part_from_sec_rules("10", None) == "Part III"
    # Item 15 → Part IV
    assert _correct_part_from_sec_rules("15", None) == "Part IV"


@pytest.mark.unit
def test_correct_part_from_sec_rules_fixes_wrong_part() -> None:
    """验证错误 Part 标签被 SEC 规则修正。

    例如 Item 7A 被 edgartools 标记为 Part I，应修正为 Part II。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    from dayu.fins.processors.ten_k_form_common import _correct_part_from_sec_rules

    # Items 2/3/4 误标为 Part II → 修正为 Part I
    assert _correct_part_from_sec_rules("2", "Part II") == "Part I"
    assert _correct_part_from_sec_rules("3", "Part II") == "Part I"
    assert _correct_part_from_sec_rules("4", "Part II") == "Part I"
    # Item 7A 误标为 Part I → 修正为 Part II
    assert _correct_part_from_sec_rules("7A", "Part I") == "Part II"
    # Item 8 误标为 Part IV → 修正为 Part II
    assert _correct_part_from_sec_rules("8", "Part IV") == "Part II"


@pytest.mark.unit
def test_correct_part_from_sec_rules_preserves_correct() -> None:
    """验证正确的 Part 标签不被修改。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    from dayu.fins.processors.ten_k_form_common import _correct_part_from_sec_rules

    assert _correct_part_from_sec_rules("1", "Part I") == "Part I"
    assert _correct_part_from_sec_rules("2", "Part I") == "Part I"
    assert _correct_part_from_sec_rules("3", "Part I") == "Part I"
    assert _correct_part_from_sec_rules("4", "Part I") == "Part I"
    assert _correct_part_from_sec_rules("7", "Part II") == "Part II"
    assert _correct_part_from_sec_rules("15", "Part IV") == "Part IV"


@pytest.mark.unit
def test_correct_part_unknown_item_preserves_original() -> None:
    """验证未知 Item 编号保留原始推断。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    from dayu.fins.processors.ten_k_form_common import _correct_part_from_sec_rules

    assert _correct_part_from_sec_rules("99", "Part X") == "Part X"
    assert _correct_part_from_sec_rules("99", None) is None


# ────────────────────────────────────────────────────────────────
# 10-Q 锚点质量验证（Running Header 场景）
# ────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_ten_q_anchor_validation_rejects_running_header() -> None:
    """验证 MSFT 式 running header 场景下锚点被正确回退。

    当文档每页重复 'Part I — Financial Information' 作为 running header 时，
    最后一个匹配位于文档末尾，选出的 Items span 极短；
    anchor 质量验证应拒绝此锚点并回退到更早的合格候选。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    from dayu.fins.processors.ten_q_form_common import (
        _anchor_produces_meaningful_items,
        _find_all_part_heading_positions,
        _select_best_part_i_anchor,
    )

    # 构造含 running header 的文档：
    # - 位置 0: ToC 区域的 Part I heading
    # - 位置 ~1000: 真实 Part I 内容区
    # - 位置 ~70000: running header（每页重复 Part I heading）
    # - 位置 ~75000: Part II 内容区
    toc_section = "Part I - Financial Information\n" + "ToC entry " * 90  # ~1000 chars
    part_i_content = (
        "Part I - Financial Information\n"
        "Item 1. Financial Statements\n"
        + "Financial data content. " * 2000  # ~46000 chars
        + "Item 2. Management Discussion\n"
        + "MD&A analysis content. " * 1000  # ~22000 chars
        + "Item 3. Quantitative\n"
        + "Quant disclosure content. " * 200  # ~5200 chars
    )
    # Running headers: 每个只有 Part heading + 极短文本
    running_headers = ""
    for _ in range(3):
        running_headers += "Part I - Financial Information\nPage content line. "

    part_ii_content = (
        "Part II - Other Information\n"
        "Item 1. Legal proceedings\n"
        + "Legal content. " * 500
    )
    full_text = toc_section + part_i_content + running_headers + part_ii_content

    part_i_positions, part_ii_positions = _find_all_part_heading_positions(full_text)
    # 应有多个 Part I 匹配（ToC + 真实 + running headers）
    assert len(part_i_positions) >= 3
    assert len(part_ii_positions) >= 1

    part_ii_anchor = part_ii_positions[-1]
    best_anchor = _select_best_part_i_anchor(full_text, part_i_positions, part_ii_anchor)

    # 最终选中的锚点应该是真实内容区的 Part I（非最后一个 running header）
    assert best_anchor is not None
    # 锚点不应该是最后一个匹配（running header）
    assert best_anchor < part_i_positions[-1]


@pytest.mark.unit
def test_ten_q_anchor_validation_accepts_normal_document() -> None:
    """验证正常文档（无 running header）仍使用最后一个匹配作为锚点。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    from dayu.fins.processors.ten_q_form_common import (
        _find_all_part_heading_positions,
        _select_best_part_i_anchor,
    )

    # 正常文档：只有 ToC + 真实内容各 1 个 Part I heading
    full_text = (
        "Part I - Financial Information\nTable of Contents\n"  # ToC 区域
        + "x" * 2000
        + "Part I – Financial Information\n"  # 真实内容区
        + "Item 1. Statements\n" + "Detail content. " * 1000
        + "Item 2. MD&A\n" + "Analysis. " * 500
        + "Part II – Other Information\n"
        + "Item 1. Legal\n" + "Legal. " * 200
    )

    part_i_positions, part_ii_positions = _find_all_part_heading_positions(full_text)
    part_ii_anchor = part_ii_positions[-1] if part_ii_positions else None
    best_anchor = _select_best_part_i_anchor(full_text, part_i_positions, part_ii_anchor)

    # 应选中最后一个合格候选（即真实内容区的 Part I，跳过 ToC 区域的）
    assert best_anchor is not None
    assert best_anchor == part_i_positions[-1]


@pytest.mark.unit
def test_anchor_produces_meaningful_items_rejects_short_spans() -> None:
    """验证极短 span 的 trial items 被判定为不合格。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    from dayu.fins.processors.ten_q_form_common import (
        _anchor_produces_meaningful_items,
    )

    full_text = "x" * 10000
    # 4 个 Items 紧密排列（span 各 100），全部极短
    short_items = [("1", 0), ("2", 100), ("3", 200), ("4", 300)]
    assert _anchor_produces_meaningful_items(full_text, short_items, 400) is False

    # 4 个 Items 间距合理（span 各 2000+），全部合格
    normal_items = [("1", 0), ("2", 2000), ("3", 4000), ("4", 6000)]
    assert _anchor_produces_meaningful_items(full_text, normal_items, 10000) is True

    # 只有 1 个 Item → 太少，无法判断
    assert _anchor_produces_meaningful_items(full_text, [("1", 0)], 10000) is False


# ---------------------------------------------------------------------------
# _find_toc_cluster_end — 部分 ToC 检测（Detection 1b）
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_find_toc_cluster_end_partial_toc_with_dramatic_jump() -> None:
    """部分 ToC（2-3 个短 span）+ 戏剧性跳跃被检测为 ToC。

    模拟 TSM 20-F 场景：Items 1-3 在 ToC 区域（span ~100），
    Item 4A 直接跳到 body（span ~117K）。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 3 个短 span（100 chars apart），然后一个戏剧性跳跃（150K）
    markers = [
        ("1", 1000),   # Item 1 in ToC
        ("2", 1100),   # Item 2 in ToC, span=100
        ("3", 1200),   # Item 3 in ToC, span=100
        ("4A", 151200),  # Item 4A in body, span=150000
    ]
    # full_text 需要足够长
    full_text = "x" * 200000
    result = _find_toc_cluster_end(full_text, markers)
    # 应检测到前 3 个 marker 在 ToC，跳到 Item 3 之后
    assert result is not None
    assert result == 1201  # markers[3 short spans] position + 1 → markers[2][1] + 1


@pytest.mark.unit
def test_find_toc_cluster_end_partial_toc_no_jump_returns_none() -> None:
    """2 个短 span 但后续跳跃不够大时，不视为部分 ToC。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 2 个短 span + 适度跳跃（span 10x，但 < 50x 阈值）
    markers = [
        ("1", 1000),
        ("2", 1100),  # span=100
        ("3", 2100),  # span=1000（仅 10x）
        ("4", 5000),
    ]
    full_text = "x" * 10000
    result = _find_toc_cluster_end(full_text, markers)
    # 跳跃比不够大（10x < 50x），不应检测为部分 ToC
    assert result is None


@pytest.mark.unit
def test_find_toc_cluster_end_two_consecutive_short_with_huge_jump() -> None:
    """仅 2 个连续短 span + 超大跳跃也触发部分 ToC 检测。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    markers = [
        ("1", 500),
        ("2", 600),     # span=100 (short)
        ("3", 700),     # span=100 (short)
        ("4", 100700),  # span=100000（1000x jump）
    ]
    full_text = "x" * 200000
    result = _find_toc_cluster_end(full_text, markers)
    assert result is not None
    assert result == 701  # markers[2][1] + 1 → skip first 2 short spans


@pytest.mark.unit
def test_find_toc_cluster_end_partial_toc_disabled_after_skip() -> None:
    """已完成一次 ToC 跳过后，禁用 check 1b 不应误判正文短节。

    模拟 SONY 20-F 场景：正文 Items 1-2 内容为 "Not Applicable"（短节），
    Item 3 含 Risk Factors（长节）。短节 + 大跳跃模式与 ToC 完全一致，
    但设置 check_partial_toc=False 后不应触发。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    markers = [
        ("1", 1000),
        ("2", 1084),     # span=84 (short, "Not Applicable")
        ("3", 1154),     # span=70 (short, "Not Applicable")
        ("4", 63154),    # span=62000（Risk Factors 长篇）
    ]
    full_text = "x" * 200000

    # 启用 check_partial_toc 时应触发（与之前行为一致）
    result_with = _find_toc_cluster_end(
        full_text, markers, check_partial_toc=True,
    )
    assert result_with is not None

    # 禁用 check_partial_toc 后不应触发（修复 SONY 误判）
    result_without = _find_toc_cluster_end(
        full_text, markers, check_partial_toc=False,
    )
    assert result_without is None


# ---------------------------------------------------------------------------
# _is_inline_reference_context — 行内引用上下文检测
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_inline_reference_context_after_text() -> None:
    """匹配位置前有普通文本字符，判定为行内引用。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # "see Item 4." — "see " 后紧跟 Item
    text = "For details, see Item 4. Information on the Company"
    pos = text.index("Item 4")
    assert _is_inline_reference_context(text, pos) is True


@pytest.mark.unit
def test_is_inline_reference_context_at_line_start() -> None:
    """匹配位置在行首时，不判定为行内引用。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = "Previous section content.\n\nItem 4. Information on the Company"
    pos = text.index("Item 4")
    assert _is_inline_reference_context(text, pos) is False


@pytest.mark.unit
def test_is_inline_reference_context_at_document_start() -> None:
    """匹配位置在文档起始处，视为行首。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = "Item 1. Business"
    assert _is_inline_reference_context(text, 0) is False


@pytest.mark.unit
def test_is_inline_reference_context_after_whitespace_and_newline() -> None:
    """换行后有缩进空白时仍视为行首。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    text = "End of section.\n   Item 5. Operating Results"
    pos = text.index("Item 5")
    assert _is_inline_reference_context(text, pos) is False


# ---------------------------------------------------------------------------
# _refine_inline_reference_markers — 行内交叉引用修正
# ---------------------------------------------------------------------------


_TEST_ITEM_PATTERN = re.compile(
    r"(?i)\bitem\s+(16[A-J]|4A|1[0-9]|[1-9])\s*[\.\:\-–—]"
)


@pytest.mark.unit
def test_refine_inline_reference_relocates_short_inline_marker() -> None:
    """行内交叉引用产生的短 section 被重定位到行首的真实标题。

    模拟 WB 20-F 场景：正文中出现 "see Item 4. Info—C. Org—Minority"
    后面才是真正的 Item 4 标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 构建模拟文本
    # 0..999: Item 3 正文
    # 1000: "see Item 4. blah" (行内引用，前面有文字)
    # 1100: Item 5 (下一个 marker)
    # 50000: "\nItem 4. Real heading" (真实标题，行首)
    # 100000: 文档结尾
    item3_text = "A" * 1000
    inline_ref = "see Item 4. Information on the Company—C. Organizational"
    gap1 = "B" * (1100 - 1000 - len(inline_ref))
    if len(gap1) < 0:
        gap1 = ""
    item5_marker = "\nItem 5. Operating Results" + "C" * 20000
    gap2 = "D" * (50000 - 1100 - len(item5_marker))
    real_heading = "\nItem 4. Information on the Company\n" + "E" * 49950

    full_text = item3_text + inline_ref + gap1 + item5_marker + gap2 + real_heading
    # 确保文本足够长
    full_text = full_text.ljust(120000, "F")

    # 定位关键位置
    inline_pos = full_text.index("Item 4. Information on the Company—")
    real_pos = full_text.index("\nItem 4. Information on the Company\n") + 1
    item5_pos = full_text.index("Item 5. Operating Results")

    # 原始贪心选中：Item 3 在 0，Item 4 在 inline_pos，Item 5 在 item5_pos
    selected = [
        ("3", 0),
        ("4", inline_pos),   # 行内引用位置
        ("5", item5_pos),
    ]

    refined = _refine_inline_reference_markers(
        full_text, selected, item_pattern=_TEST_ITEM_PATTERN,
    )

    # Item 4 应被重定位到行首的真实标题位置
    refined_dict = {token: pos for token, pos in refined}
    assert refined_dict["4"] == real_pos
    # Item 3 和 Item 5 不变
    assert refined_dict["3"] == 0
    assert refined_dict["5"] == item5_pos


@pytest.mark.unit
def test_refine_inline_reference_keeps_normal_markers() -> None:
    """正常的 markers（span 足够长）不被修改。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 三个 Items 间距足够大，没有行内引用
    full_text = (
        "\nItem 1. Business\n"
        + "A" * 30000
        + "\nItem 2. Properties\n"
        + "B" * 30000
        + "\nItem 3. Legal Proceedings\n"
        + "C" * 30000
    )
    item1_pos = full_text.index("Item 1.")
    item2_pos = full_text.index("Item 2.")
    item3_pos = full_text.index("Item 3.")

    selected = [("1", item1_pos), ("2", item2_pos), ("3", item3_pos)]

    refined = _refine_inline_reference_markers(
        full_text, selected, item_pattern=_TEST_ITEM_PATTERN,
    )

    # 所有 marker 保持不变
    assert refined == selected


@pytest.mark.unit
def test_refine_inline_reference_single_marker_unchanged() -> None:
    """只有一个 marker 时不做处理。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    full_text = "\nItem 1. Business\n" + "A" * 5000
    selected = [("1", 1)]

    refined = _refine_inline_reference_markers(
        full_text, selected, item_pattern=_TEST_ITEM_PATTERN,
    )

    assert refined == selected


@pytest.mark.unit
def test_extract_source_text_preserving_lines_keeps_report_heading_boundaries(tmp_path: Path) -> None:
    """验证报告类原始 HTML 文本提取会保留标题换行结构。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html_path = tmp_path / "sample_20f.htm"
    html_path.write_text(
        """
        <html><body>
        <table>
          <tr><td><span style="font-style:italic">ITEM 2.</span></td></tr>
          <tr><td><span style="font-style:italic">OFFER STATISTICS AND EXPECTED TIMETABLE</span></td></tr>
        </table>
        <p>Not applicable.</p>
        <table>
          <tr><td><span style="font-style:italic">ITEM 5.</span></td></tr>
          <tr><td><span style="font-style:italic">OPERATING AND FINANCIAL REVIEW AND PROSPECTS</span></td></tr>
        </table>
        </body></html>
        """,
        encoding="utf-8",
    )
    source = LocalFileSource(path=html_path, uri=f"local://{html_path.name}", media_type="text/html")

    text = _extract_source_text_preserving_lines(source)

    assert "ITEM 2." in text
    assert "OFFER STATISTICS AND EXPECTED TIMETABLE" in text
    assert "ITEM 5." in text
    assert "OPERATING AND FINANCIAL REVIEW AND PROSPECTS" in text


@pytest.mark.unit
def test_extract_source_text_preserving_lines_skips_script_and_preserves_table_breaks(
    tmp_path: Path,
) -> None:
    """验证流式 HTML 文本抽取会跳过脚本样式并保留表格换行边界。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html_path = tmp_path / "sample_large_20f.htm"
    html_path.write_text(
        """
        <html><body>
        <style>.hidden{display:none;}</style>
        <script>var should_not_be_visible = true;</script>
        <table>
          <tr><td>ITEM 3.</td></tr>
          <tr><td>KEY INFORMATION</td></tr>
        </table>
        <div>Operating data</div>
        <table>
          <tr><td>ITEM 18.</td></tr>
          <tr><td>FINANCIAL STATEMENTS</td></tr>
        </table>
        </body></html>
        """,
        encoding="utf-8",
    )
    source = LocalFileSource(path=html_path, uri=f"local://{html_path.name}", media_type="text/html")

    text = _extract_source_text_preserving_lines(source)

    assert "should_not_be_visible" not in text
    assert "ITEM 3.\nKEY INFORMATION" in text
    assert "Operating data" in text
    assert "ITEM 18.\nFINANCIAL STATEMENTS" in text


# ---------------------------------------------------------------------------
# _looks_like_default_headers 回归测试
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_looks_like_default_headers_unicode_superscript_no_crash() -> None:
    """上标字符 '¹'（isdigit=True 但 int() 会抛 ValueError）不应崩溃，应返回 False。

    回归测试：isdigit() → isdecimal() 修复。
    NGG (National Grid) 财务表格含 ¹ 上标脚注标记，导致 _looks_like_default_headers
    抛 ValueError 使 TwentyFFormProcessor 和 SecProcessor 全部崩溃，
    最终降级到无 XBRL 能力的 FinsBSProcessor。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败或函数抛出 ValueError 时抛出。
    """

    from dayu.fins.processors.sec_table_extraction import _looks_like_default_headers

    # 上标字符 isdigit() 返回 True，但 int() 无法解析
    assert _looks_like_default_headers(["¹", "²", "³"]) is False
    assert _looks_like_default_headers(["¹"]) is False
    assert _looks_like_default_headers(["Revenue", "¹", "Cost"]) is False


@pytest.mark.unit
def test_looks_like_default_headers_accepts_sequential_ascii_integers() -> None:
    """验证连续 ASCII 整数序列被正确识别为自动生成的列头。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.sec_table_extraction import _looks_like_default_headers

    assert _looks_like_default_headers(["1", "2", "3"]) is True
    assert _looks_like_default_headers(["0", "1", "2", "3"]) is True
    assert _looks_like_default_headers(["1", "2", "3", "4", "5"]) is True


@pytest.mark.unit
def test_looks_like_default_headers_rejects_text_headers() -> None:
    """验证有实际语义的文本表头（非数值类）被正确拒绝。

    注意：纯 ASCII 数字（如 '1', '3', '5'）会被 _is_low_information_header 识别为
    低信息表头并提前返回 True，不会走到 isdecimal() 路径。
    文本表头不匹配低信息判定，且 isdecimal() 为 False，最终返回 False。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.sec_table_extraction import _looks_like_default_headers

    # 有语义的文本表头：不是低信息，也不是 decimal → False
    assert _looks_like_default_headers(["Revenue", "Cost", "Profit"]) is False
    # 混合文本和数字
    assert _looks_like_default_headers(["1", "Revenue", "3"]) is False
    # 上标字符（非低信息、非 isdecimal）
    assert _looks_like_default_headers(["¹", "Revenue"]) is False
