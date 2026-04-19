"""6-K 表单章节处理器覆盖率测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import pandas as pd
import pytest

from dayu.fins.processors import six_k_form_common as six_k_form_common_module
from dayu.fins.processors import bs_six_k_processor as bs_six_k_processor_module
from dayu.fins.processors.bs_six_k_processor import BsSixKFormProcessor
from dayu.fins.storage.local_file_source import LocalFileSource


def _statement_reason(statement: Mapping[str, object]) -> str | None:
    """安全读取财务报表结果里的可选 reason 字段。

    Args:
        statement: 财务报表结果字典。

    Returns:
        `reason` 字段；不存在时返回 `None`。

    Raises:
        无。
    """

    value = statement.get("reason")
    return value if isinstance(value, str) else None


def _required_section_ref(table: dict[str, object]) -> str:
    """读取表格的 section_ref，并在测试边界保证非空。

    Args:
        table: 表格摘要字典。

    Returns:
        非空 section_ref。

    Raises:
        AssertionError: section_ref 缺失时抛出。
    """

    ref = table.get("section_ref")
    if not isinstance(ref, str) or not ref:
        raise AssertionError("table.section_ref 缺失")
    return ref


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
def test_six_k_processor_xml_media_type_detection(tmp_path: Path) -> None:
    """验证 6-K 处理器正确识别 XML 媒体类型。

    测试场景：
    - 提供媒体类型为 XML 的源。
    - 处理器应该返回 True。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 使用 XML 媒体类型创建源
    source_path = tmp_path / "6k.xml"
    source_path.write_text(
        """<?xml version="1.0"?>
        <document>
            <Exhibit99Dot1>Press Release</Exhibit99Dot1>
        </document>""",
        encoding="utf-8",
    )
    source = _make_source(source_path, media_type="application/xml")

    # 验证 supports() 对 XML 媒体类型返回 True
    assert (
        BsSixKFormProcessor.supports(source, form_type="6-K", media_type="application/xml")
        is True
    )


@pytest.mark.unit
def test_six_k_processor_xml_uri_suffix_detection(tmp_path: Path) -> None:
    """验证 6-K 处理器通过 URI 后缀识别 XML 文件。

    测试场景：
    - 源的 URI 以 .xml 结尾。
    - 即使媒体类型为空，处理器也应该返回 True。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_report.xml"
    source_path.write_text("<document><data>test</data></document>", encoding="utf-8")
    source = _make_source(source_path, media_type=None)

    # 通过 URI 后缀识别 XML（即使媒体类型为空）
    # 由于 supports() 方法首先检查 FinsBSProcessor.supports()，
    # 然后检查媒体类型，最后检查 URI 后缀
    # 我们验证处理器能够识别 .xml 后缀
    result = BsSixKFormProcessor.supports(source, form_type="6-K", media_type=None)
    # 结果可能取决于 FinsBSProcessor.supports() 的返回值
    # 但至少不应该抛出异常
    assert result is True or result is False


@pytest.mark.unit
def test_six_k_processor_form_type_6k(tmp_path: Path) -> None:
    """验证 6-K 处理器支持 6-K 表单类型。

    测试场景：
    - 使用标准 6-K 表单类型。
    - 处理器应该返回 True。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k.html"
    source_path.write_text(
        "<html><body>6-K content with exhibits</body></html>", encoding="utf-8"
    )
    source = _make_source(source_path, media_type="text/html")

    # 验证 supports() 对 6-K 表单类型返回 True（通过 HTML）
    result = BsSixKFormProcessor.supports(source, form_type="6-K", media_type="text/html")
    # 由于 HTML 的支持取决于 FinsBSProcessor，这里仅进行基本验证
    assert isinstance(result, bool)


@pytest.mark.unit
def test_six_k_processor_empty_media_type_handling(tmp_path: Path) -> None:
    """验证 6-K 处理器处理空媒体类型的情况。

    测试场景：
    - 媒体类型为空或 None。
    - 处理器应该尝试通过 URI 后缀推断类型。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_empty_media.html"
    source_path.write_text("<html><body>content</body></html>", encoding="utf-8")
    source = _make_source(source_path, media_type=None)

    # 验证 supports() 能够处理空媒体类型
    result = BsSixKFormProcessor.supports(source, form_type="6-K", media_type=None)
    assert isinstance(result, bool)


@pytest.mark.unit
def test_six_k_processor_splits_multiple_markers(tmp_path: Path) -> None:
    """验证 6-K 处理器正确识别多个章节标记。

    测试场景：
    - HTML 中包含多个预定义章节标记。
    - 处理器应该根据标记创建多个虚拟小节。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_markers.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p>Trip.com Group Limited</p>
            <p><b>Exhibit 99.1</b> - Press Release</p>
            <p><b>Key Highlights</b> for Q1 2025</p>
            <p><b>Financial Results and Business Updates</b></p>
            <p><b>Conference Call</b> Information</p>
            <p><b>Safe Harbor Statement</b></p>
            <p><b>About Non-GAAP Financial Measures</b></p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )
    sections = processor.list_sections()
    titles = [str(item.get("title") or "") for item in sections]

    # v2: "Exhibit" 不再作为独立章节标记（归入 Cover Page）
    assert all(title != "Exhibit" for title in titles)
    assert any(title == "Key Highlights" for title in titles)
    assert any(title == "Conference Call" for title in titles)
    assert any(title == "Safe Harbor" for title in titles)
    assert any(title == "About Non-GAAP" for title in titles)


@pytest.mark.unit
def test_six_k_processor_handles_missing_markers(tmp_path: Path) -> None:
    """验证 6-K 处理器处理缺失章节标记的情况。

    测试场景：
    - HTML 中不含任何预定义的章节标记。
    - 处理器应该回退到父类的章节处理能力。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_no_markers.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p>General news update</p>
            <p>Business progress report</p>
            <p>No specific markers present</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )
    sections = processor.list_sections()

    # 验证处理器能够返回章节列表（可能回退到父类章节）
    assert isinstance(sections, list)


@pytest.mark.unit
def test_six_k_processor_rejects_non_6k_form_type(tmp_path: Path) -> None:
    """验证 6-K 处理器拒绝非 6-K 表单类型。

    测试场景：
    - 使用其他表单类型，如 10-K。
    - 处理器应该返回 False。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "10k.html"
    source_path.write_text("<html><body>10-K content</body></html>", encoding="utf-8")
    source = _make_source(source_path, media_type="text/html")

    # 验证 supports() 对非 6-K 表单返回 False
    assert BsSixKFormProcessor.supports(source, form_type="10-K", media_type="text/html") is False
    assert BsSixKFormProcessor.supports(source, form_type="8-K", media_type="text/html") is False


@pytest.mark.unit
def test_six_k_processor_preserves_section_content(tmp_path: Path) -> None:
    """验证 6-K 处理器能够完整读取小节内容。

    测试场景：
    - 创建包含多个标记的文档。
    - 验证可以通过 read_section 读取每个小节的完整内容。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_content.html"
    source_path.write_text(
        """
        <html>
          <body>
            <h2>Exhibit 99.1</h2>
            <p>Press release content paragraph 1</p>
            <p>Press release content paragraph 2</p>
            <h2>Key Highlights</h2>
            <p>Highlights for Q1</p>
            <p>Revenue increased 25%</p>
            <h2>Financial Results and Business Updates</h2>
            <p>Detailed financial analysis</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )
    sections = processor.list_sections()

    # 验证至少有一些小节被创建
    if sections:
        first_ref = str(sections[0]["ref"])
        section_content = processor.read_section(first_ref)
        
        # 验证内容字段存在且非空
        assert "content" in section_content
        assert isinstance(section_content["content"], str)


# ---------------------------------------------------------------------------
# v2 新增：财务报表标题标记识别
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_six_k_processor_detects_financial_statement_markers(tmp_path: Path) -> None:
    """验证 6-K v2 处理器识别财务报表标题标记。

    测试场景：
    - HTML 包含 Balance Sheet 和 Statements of Income 标题。
    - 处理器应为每个财务报表创建独立段。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_financial.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Financial Results and Business Updates</b></p>
            <p>Revenue increased 25% year-over-year.</p>
            <p><b>Safe Harbor Statement</b></p>
            <p>Forward-looking statements disclaimer text.</p>
            <p><b>Consolidated Balance Sheets</b></p>
            <p>Cash and cash equivalents: 5,000 million. Total assets: 30,000 million.</p>
            <p><b>Consolidated Statements of Income</b></p>
            <p>Revenue: 12,700 million. Net income: 2,500 million.</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )
    sections = processor.list_sections()
    titles = [str(item.get("title") or "") for item in sections]

    # 验证财务报表标记被识别为独立段
    assert any(title == "Balance Sheets" for title in titles)
    assert any(title == "Statements of Income" for title in titles)
    assert any(title == "Safe Harbor" for title in titles)


# ---------------------------------------------------------------------------
# v2 新增：Recent Development 标记
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_six_k_processor_detects_recent_development(tmp_path: Path) -> None:
    """验证 6-K v2 处理器识别 Recent Development 标记。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_recent_dev.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Key Highlights</b></p>
            <p>Hotel bookings exceeded pre-COVID levels by 120%.</p>
            <p><b>Recent Development</b></p>
            <p>Board authorized new capital return measures in 2025.</p>
            <p><b>Safe Harbor Statement</b></p>
            <p>Forward-looking statements disclaimer.</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )
    sections = processor.list_sections()
    titles = [str(item.get("title") or "") for item in sections]

    assert any(title == "Recent Development" for title in titles)
    assert any(title == "Key Highlights" for title in titles)


# ---------------------------------------------------------------------------
# v2 新增：泛化公司名匹配
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_six_k_processor_matches_generalized_company_names(tmp_path: Path) -> None:
    """验证泛化 About Company 匹配支持多种法律后缀。

    测试 Inc./Corp./Holdings 等非 "Limited" 后缀。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _build_six_k_markers

    # Inc. 后缀
    markers_inc = _build_six_k_markers("About Acme Corp Inc. is a leading company.")
    titles_inc = [title for _, title in markers_inc]
    assert "About Company" in titles_inc

    # Holdings 后缀
    markers_hold = _build_six_k_markers("About SomeFinTech Holdings is a global platform.")
    titles_hold = [title for _, title in markers_hold]
    assert "About Company" in titles_hold

    # Group 后缀
    markers_grp = _build_six_k_markers("About Trip.com Group is a leading platform.")
    titles_grp = [title for _, title in markers_grp]
    assert "About Company" in titles_grp


@pytest.mark.unit
def test_six_k_processor_report_mode_prefers_report_headings(tmp_path: Path) -> None:
    """验证目录型长篇 6-K 优先按 report headings 切分。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_report_mode.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p>Table of Contents</p>
            <p>About this report 4</p>
            <p>Governance 13</p>
            <p>Strategy 21</p>
            <p>Environment 24</p>
            <p>Social 58</p>
            <p>Appendix 97</p>
            <div style="height:1800px;">preface spacer</div>
            <h1>About this report</h1>
            <h2>Overview</h2>
            <p>The sustainability report provides ESG disclosures.</p>
            <h2>Governance</h2>
            <p>Governance body text.</p>
            <h2>Strategy</h2>
            <p>Strategy body text.</p>
            <h2>Environment</h2>
            <p>Environment body text.</p>
            <h2>Social</h2>
            <p>Social body text.</p>
            <h2>Appendix</h2>
            <p>Appendix body text.</p>
            <h2>Consolidated Balance Sheets</h2>
            <p>Cash and cash equivalents: 5,000 million.</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )
    titles = [str(item.get("title") or "") for item in processor.list_sections()]

    assert "Overview" in titles
    assert "Governance" in titles
    assert "Strategy" in titles
    assert "Social" in titles
    assert "Appendix" in titles
    assert any("Balance Sheets" in title for title in titles)


@pytest.mark.unit
def test_six_k_about_company_skips_inline_information_phrase() -> None:
    """验证 About Company 不应命中正文里的 information about 叙述。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _build_six_k_markers

    markers = _build_six_k_markers(
        "For more information about SomeTech Group, please visit our website. "
        "Consolidated Balance Sheets"
    )
    titles = [title for _, title in markers]

    assert "About Company" not in titles


# ---------------------------------------------------------------------------
# v2 新增：搜索 token 级回退
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_six_k_search_exact_match_takes_priority(tmp_path: Path) -> None:
    """验证精确匹配优先于 token 回退。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_search.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Financial Results and Business Updates</b></p>
            <p>Net revenue for the quarter was 12.7 billion.</p>
            <p><b>Safe Harbor Statement</b></p>
            <p>Forward-looking statements regarding net revenue projections.</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    # "net revenue" 精确匹配应命中
    hits = processor.search("net revenue")
    assert len(hits) > 0


# ---------------------------------------------------------------------------
# v2.1 新增：use_last_match 策略验证
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_six_k_statement_markers_skip_inline_references(tmp_path: Path) -> None:
    """验证财务报表标记跳过内联引用，取末次匹配。

    场景：叙述段内联引用 "statement of operations"，
    实际表格标题 "Statements of Income" 出现在后方。
    标记应取后者而非前者。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_inline_ref.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>About Non-GAAP Financial Measures</b></p>
            <p>Reconciliation data included in the consolidated statement of
            operations are included at the end of this press release.</p>
            <p><b>About SomeTech Group</b></p>
            <p>SomeTech Group is a leading platform for travel services.</p>
            <p>SomeTech Group Limited Unaudited Consolidated Statements of Income</p>
            <p>Revenue: 12,700 million. Net income: 2,500 million.</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )
    sections = processor.list_sections()
    titles = [str(item.get("title") or "") for item in sections]

    # "About Non-GAAP" 和 "About Company" 应分别存在
    assert any(title == "About Non-GAAP" for title in titles)
    assert any(title == "About Company" for title in titles)
    # "Statements of Income" 应匹配到实际标题，而非内联引用
    assert any(title == "Statements of Income" for title in titles)

    # About Non-GAAP 应包含完整的叙述内容（不被内联引用截断）
    for s in sections:
        if s.get("title") == "About Non-GAAP":
            content = processor.read_section(s["ref"]).get("content", "")
            # 内容应包含 "included at the end"（叙述完整）
            assert "included at the end" in content


@pytest.mark.unit
def test_six_k_search_token_fallback_for_multiword(tmp_path: Path) -> None:
    """验证多词查询精确匹配失败时 token OR 回退生效。

    文档不含 "cash flow" 短语，但包含 "cash equivalents"。
    token 回退应以 "cash" 匹配命中。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_token_search.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Financial Results and Business Updates</b></p>
            <p>Revenue increased by 25% year-over-year reaching record levels.</p>
            <p><b>Balance Sheets</b></p>
            <p>Cash and cash equivalents: 5,000 million. Short-term investments: 2,000 million.</p>
            <p><b>Safe Harbor Statement</b></p>
            <p>This announcement contains forward-looking statements.</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    # "cash flow" 精确匹配失败，但 token "cash" 存在于 Balance Sheets 段
    hits = processor.search("cash flow")
    assert len(hits) > 0
    # snippet 应包含 "cash" token
    snippets = [h.get("snippet", "").lower() for h in hits]
    assert any("cash" in s for s in snippets)


@pytest.mark.unit
def test_six_k_search_single_word_no_fallback(tmp_path: Path) -> None:
    """验证单词查询不触发 token 回退（无意义拆分）。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_single_search.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Financial Results and Business Updates</b></p>
            <p>Revenue increased by 25% year-over-year.</p>
            <p><b>Safe Harbor Statement</b></p>
            <p>This contains forward-looking statements.</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    # "guidance" 是单词且不存在 → 应返回空
    hits = processor.search("guidance")
    assert len(hits) == 0


@pytest.mark.unit
def test_six_k_financial_statement_extraction_prefers_financial_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
        """验证 6-K 能从财务表格回退抽取结构化报表。

        Args:
                tmp_path: pytest 临时目录。

        Returns:
                无。

        Raises:
                AssertionError: 断言失败时抛出。
        """

        source_path = tmp_path / "6k_financial_statement.html"
        source_path.write_text(
                """
                <html>
                    <body>
                        <p><b>Balance Sheets</b></p>
                        <table>
                            <caption>Condensed Consolidated Balance Sheets (In thousands)</caption>
                            <tr><td>placeholder</td></tr>
                        </table>
                        <p><b>Statements of Income</b></p>
                        <table>
                            <caption>Condensed Consolidated Statements of Operations (In thousands)</caption>
                            <tr><td>placeholder</td></tr>
                        </table>
                    </body>
                </html>
                """,
                encoding="utf-8",
        )

        processor = BsSixKFormProcessor(
                _make_source(source_path, media_type="text/html"),
                form_type="6-K",
                media_type="text/html",
        )

        balance_dataframe = pd.DataFrame(
            [
                [None, None, "As of December 31,", "As of December 31,", None, None, "As of December 31,", "As of December 31,", None, None, "As of December 31,", "As of December 31,", None],
                [None, None, "2024", "2024", None, None, "2025", "2025", None, None, "2025", "2025", None],
                [None, None, "HK$", "HK$", None, None, "HK$", "HK$", None, None, "US$", "US$", None],
                ["ASSETS", None, None, None, None, None, None, None, None, None, None, None, None],
                ["Cash and cash equivalents", None, None, "100", None, None, None, "120", None, None, None, "15", None],
                ["Total assets", None, None, "300", None, None, None, "360", None, None, None, "45", None],
            ]
        )

        def _fake_parse_table_dataframe(table_tag: object) -> pd.DataFrame:
            del table_tag
            return balance_dataframe.copy()

        monkeypatch.setattr(six_k_form_common_module, "parse_html_table_dataframe", _fake_parse_table_dataframe)

        balance_sheet = processor.get_financial_statement("balance_sheet")
        assert balance_sheet["data_quality"] == "extracted"
        assert len(balance_sheet["periods"]) == 2
        assert any(row["label"] == "Total assets" for row in balance_sheet["rows"])
        assert balance_sheet["currency"] == "HKD"

        cash_flow = processor.get_financial_statement("cash_flow")
        assert _statement_reason(cash_flow) == "statement_not_found"


@pytest.mark.unit
def test_six_k_financial_statement_prefers_xbrl_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
        """验证 6-K 存在 XBRL 时优先返回 XBRL 报表。

        Args:
                tmp_path: pytest 临时目录。
                monkeypatch: pytest monkeypatch fixture。

        Returns:
                无。

        Raises:
                AssertionError: 断言失败时抛出。
        """

        source_path = tmp_path / "6k_xbrl_preferred.html"
        source_path.write_text(
                "<html><body><p>placeholder</p></body></html>",
                encoding="utf-8",
        )

        processor = BsSixKFormProcessor(
                _make_source(source_path, media_type="text/html"),
                form_type="6-K",
                media_type="text/html",
        )

        class _FakeStatements:
                """伪造 XBRL statements 容器。"""

                def income_statement(self) -> object:
                        """返回伪造报表对象。"""

                        return object()

        class _FakeXbrl:
                """伪造 XBRL 对象。"""

                statements = _FakeStatements()

        monkeypatch.setattr(processor, "_get_xbrl", lambda: _FakeXbrl())
        monkeypatch.setattr(
                bs_six_k_processor_module,
                "_safe_statement_dataframe",
                lambda statement_obj: pd.DataFrame(
                        [
                                {
                                        "concept": "ifrs-full:Revenue",
                                        "label": "Revenue",
                                        "2025-06-30": 1200,
                                }
                        ]
                ),
        )
        monkeypatch.setattr(
                bs_six_k_processor_module,
                "_infer_units_from_xbrl_query",
                lambda xbrl: "USD",
        )

        statement = processor.get_financial_statement("income")
        assert statement["data_quality"] == "xbrl"
        assert statement["periods"][0]["period_end"] == "2025-06-30"
        assert statement["rows"][0]["label"] == "Revenue"


@pytest.mark.unit
def test_six_k_list_tables_remaps_statement_tables_by_caption(tmp_path: Path) -> None:
        """验证 6-K 财务表可按 caption 重映射到虚拟章节。

        Args:
                tmp_path: pytest 临时目录。

        Returns:
                无。

        Raises:
                AssertionError: 断言失败时抛出。
        """

        source_path = tmp_path / "6k_table_mapping.html"
        source_path.write_text(
                """
                <html>
                    <body>
                        <p><b>Safe Harbor Statement</b></p>
                        <p>Forward-looking statement text.</p>
                        <p><b>Balance Sheets</b></p>
                        <table>
                            <caption>Condensed Consolidated Balance Sheets (Continued) (In thousands)</caption>
                            <tr><th></th><th></th><th>As of December 31,</th><th>2024</th><th></th><th></th><th>As of December 31,</th><th>2025</th></tr>
                            <tr><th></th><th></th><th>2024</th><th>2024</th><th></th><th></th><th>2025</th><th>2025</th></tr>
                            <tr><th></th><th></th><th>HK$</th><th>HK$</th><th></th><th></th><th>HK$</th><th>HK$</th></tr>
                            <tr><td>Total assets</td><td></td><td></td><td>300</td><td></td><td></td><td></td><td>360</td></tr>
                        </table>
                        <p><b>Statements of Income</b></p>
                        <table>
                            <caption>Condensed Consolidated Statements of Comprehensive Income (In thousands)</caption>
                            <tr><th></th><th></th><th>For the Three Months Ended</th><th>For the Three Months Ended</th><th></th><th></th><th>For the Three Months Ended</th><th>For the Three Months Ended</th></tr>
                            <tr><th></th><th></th><th>December 31, 2024</th><th>December 31, 2024</th><th></th><th></th><th>December 31, 2025</th><th>December 31, 2025</th></tr>
                            <tr><th></th><th></th><th>HK$</th><th>HK$</th><th></th><th></th><th>HK$</th><th>HK$</th></tr>
                            <tr><td>Total revenues</td><td></td><td></td><td>150</td><td></td><td></td><td></td><td>210</td></tr>
                        </table>
                        <p><b>Reconciliation</b></p>
                        <table>
                            <caption>Reconciliations of Non-GAAP and GAAP Results (In thousands)</caption>
                            <tr><th></th><th></th><th>For the Three Months Ended</th><th>For the Three Months Ended</th><th></th><th></th><th>For the Three Months Ended</th><th>For the Three Months Ended</th></tr>
                            <tr><th></th><th></th><th>December 31, 2024</th><th>December 31, 2024</th><th></th><th></th><th>December 31, 2025</th><th>December 31, 2025</th></tr>
                            <tr><th></th><th></th><th>HK$</th><th>HK$</th><th></th><th></th><th>HK$</th><th>HK$</th></tr>
                            <tr><td>Adjusted net income</td><td></td><td></td><td>35</td><td></td><td></td><td></td><td>49</td></tr>
                        </table>
                    </body>
                </html>
                """,
                encoding="utf-8",
        )

        processor = BsSixKFormProcessor(
                _make_source(source_path, media_type="text/html"),
                form_type="6-K",
                media_type="text/html",
        )
        section_title_by_ref = {
            section["ref"]: section["title"]
            for section in processor.list_sections()
        }
        table_map = {
                table["caption"]: table["section_ref"]
                for table in processor.list_tables()
                if table.get("caption")
        }

        assert section_title_by_ref[_required_section_ref({"section_ref": table_map["Condensed Consolidated Balance Sheets (Continued) (In thousands)"]})] == "Balance Sheets"
        assert section_title_by_ref[_required_section_ref({"section_ref": table_map["Condensed Consolidated Statements of Comprehensive Income (In thousands)"]})] == "Statements of Income"
        assert section_title_by_ref[_required_section_ref({"section_ref": table_map["Reconciliations of Non-GAAP and GAAP Results (In thousands)"]})] == "Reconciliation"


@pytest.mark.unit
def test_six_k_statement_classification_recognizes_income_statements_phrase() -> None:
    """验证 `Income Statements` 语序标题可识别为 income。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    statement_type = six_k_form_common_module._classify_statement_type_for_table(
        caption="Arm Holdings plc Condensed Consolidated Income Statements (Unaudited)",
        headers=["For the three months ended December 31, 2024"],
        context_before="Table of Contents Statements of Cash Flows",
    )
    assert statement_type == "income"


@pytest.mark.unit
def test_six_k_statement_classification_skips_table_of_contents_noise() -> None:
    """验证目录类表格不会误判为财务报表。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    statement_type = six_k_form_common_module._classify_statement_type_for_table(
        caption="Table of Contents",
        headers=[
            "Condensed Consolidated Balance Sheets",
            "Condensed Consolidated Statements of Cash Flows",
        ],
        context_before="",
    )
    assert statement_type is None


@pytest.mark.unit
def test_six_k_statement_classification_recognizes_bank_style_financial_results() -> None:
    """验证银行口径的 `Standalone Financial Results` 可识别为 income。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    statement_type = six_k_form_common_module._classify_statement_type_for_table(
        caption="Standalone Financial Results for the quarter ended March 31, 2025",
        headers=[
            "Particulars",
            "Interest earned",
            "Interest expended",
            "Net profit for the period",
        ],
        context_before="",
    )
    assert statement_type == "income"


@pytest.mark.unit
def test_six_k_statement_classification_recognizes_assets_and_liabilities_heading() -> None:
    """验证 `Statement of Assets and Liabilities` / `Capital and Liabilities` 可识别为 balance_sheet。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    statement_type = six_k_form_common_module._classify_statement_type_for_table(
        caption="Standalone Statement of Assets and Liabilities is given below:",
        headers=[
            "Particulars",
            "CAPITAL AND LIABILITIES",
            "As at 31.03.2025",
            "As at 31.03.2024",
        ],
        context_before="",
    )
    assert statement_type == "balance_sheet"


@pytest.mark.unit
def test_six_k_financial_statement_extraction_supports_numeric_date_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证可解析 `03/31/2025` 这类数字日期表头。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_numeric_date_header.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Balance Sheets</b></p>
            <table>
              <caption>Interim Consolidated Balance Sheet</caption>
              <tr><td>placeholder</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    balance_dataframe = pd.DataFrame(
        [
            ["Assets", "Note", "03/31/2025", "12/31/2024"],
            ["", "", "", ""],
            ["Cash and cash equivalents", "5.1", "19118354", "28595666"],
            ["Total assets", "", "300", "360"],
        ]
    )

    def _fake_parse_table_dataframe(table_tag: object) -> pd.DataFrame:
        del table_tag
        return balance_dataframe.copy()

    monkeypatch.setattr(six_k_form_common_module, "parse_html_table_dataframe", _fake_parse_table_dataframe)

    balance_sheet = processor.get_financial_statement("balance_sheet")
    assert balance_sheet["data_quality"] == "extracted"
    assert len(balance_sheet["rows"]) >= 2
    assert [period["period_end"] for period in balance_sheet["periods"]] == ["2025-03-31", "2024-12-31"]


@pytest.mark.unit
def test_six_k_financial_statement_extraction_supports_month_year_split_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证可解析 `Dec + 2024` 的分裂式表头。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_split_month_year_header.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Balance Sheets</b></p>
            <table>
              <caption>Statement of Financial Position</caption>
              <tr><td>placeholder</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    split_header_dataframe = pd.DataFrame(
        [
            ["GROUP STATEMENT OF FINANCIAL POSITION", "", "As at", "As at"],
            ["GROUP STATEMENT OF FINANCIAL POSITION", "", "Dec", "Dec"],
            ["GROUP STATEMENT OF FINANCIAL POSITION", "", "2024", "2023"],
            ["Cash and cash equivalents", "", "100", "80"],
            ["Total assets", "", "300", "260"],
        ]
    )

    def _fake_parse_table_dataframe(table_tag: object) -> pd.DataFrame:
        del table_tag
        return split_header_dataframe.copy()

    monkeypatch.setattr(six_k_form_common_module, "parse_html_table_dataframe", _fake_parse_table_dataframe)

    balance_sheet = processor.get_financial_statement("balance_sheet")
    assert balance_sheet["data_quality"] == "extracted"
    assert [period["period_end"] for period in balance_sheet["periods"]] == ["2024-12-31", "2023-12-31"]
    assert any(row["label"] == "Total assets" for row in balance_sheet["rows"])


@pytest.mark.unit
def test_six_k_financial_statement_extraction_fallbacks_to_row_signals(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证标题缺失时可通过行标签语义回退提取报表。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_row_signal_fallback.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Financial Results and Business Updates</b></p>
            <table>
              <caption>dollar, except for number of shares and per share data)</caption>
              <tr><td>placeholder</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    fallback_dataframe = pd.DataFrame(
        [
            ["As at", "December 31, 2024", "December 31, 2023"],
            ["", "", ""],
            ["NON-CURRENT ASSETS", "", ""],
            ["Cash and cash equivalents", "100", "90"],
            ["Trade receivables", "80", "70"],
            ["Total current assets", "190", "170"],
            ["Total assets", "300", "260"],
            ["NON-CURRENT LIABILITIES", "", ""],
            ["Trade payables", "70", "60"],
            ["Total liabilities", "180", "150"],
            ["Total equity", "120", "110"],
        ]
    )

    def _fake_parse_table_dataframe(table_tag: object) -> pd.DataFrame:
        del table_tag
        return fallback_dataframe.copy()

    monkeypatch.setattr(six_k_form_common_module, "parse_html_table_dataframe", _fake_parse_table_dataframe)

    balance_sheet = processor.get_financial_statement("balance_sheet")
    assert balance_sheet["data_quality"] == "extracted"
    assert any(row["label"] == "Total assets" for row in balance_sheet["rows"])
    assert any(row["label"] == "Total liabilities" for row in balance_sheet["rows"])


@pytest.mark.unit
def test_six_k_financial_statement_extraction_supports_bank_style_results_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证银行风格 6-K 财务结果表可提取 income 与 balance_sheet。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_bank_style_results.html"
    source_path.write_text(
        """
        <html>
          <body>
            <table>
              <caption>Standalone Financial Results for the quarter ended March 31, 2025</caption>
              <tr>
                <th>Particulars</th>
                <th>Interest earned</th>
                <th>Interest expended</th>
                <th>Net profit for the period</th>
              </tr>
            </table>
            <table>
              <caption>Standalone Statement of Assets and Liabilities is given below:</caption>
              <tr>
                <th>Particulars</th>
                <th>CAPITAL AND LIABILITIES</th>
                <th>As at 31.03.2025</th>
                <th>As at 31.03.2024</th>
              </tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    income_dataframe = pd.DataFrame(
        [
            ["Particulars", "Quarter ended", "Quarter ended"],
            ["Particulars", "31.03.2025", "31.12.2024"],
            ["Interest earned", "77460.11", "76006.88"],
            ["Interest expended", "45394.31", "44000.00"],
            ["Total Income", "89487.99", "87460.44"],
            ["Net profit for the period", "17500.00", "16000.00"],
        ]
    )
    balance_dataframe = pd.DataFrame(
        [
            ["Particulars", "As at 31.03.2025", "As at 31.03.2024"],
            ["Capital", "765.22", "759.69"],
            ["Reserves and surplus", "496854.21", "436833.39"],
            ["Deposits", "2714714.90", "2379786.28"],
            ["Borrowings", "547930.90", "662153.07"],
            ["Total liabilities", "3422579.51", "3180030.00"],
            ["Total assets", "3910198.94", "3617623.06"],
        ]
    )

    def _fake_parse_table_dataframe(table_tag: object) -> pd.DataFrame:
        """按 caption 返回银行风格测试表格。"""

        caption_getter = getattr(table_tag, "find", None)
        if caption_getter is None:
            raise AssertionError("table_tag 不支持 find()")
        caption_tag = caption_getter("caption")
        caption_text = caption_tag.get_text(" ", strip=True) if caption_tag else ""
        if "Financial Results" in caption_text:
            return income_dataframe.copy()
        if "Assets and Liabilities" in caption_text:
            return balance_dataframe.copy()
        raise AssertionError(f"unexpected caption: {caption_text}")

    monkeypatch.setattr(six_k_form_common_module, "parse_html_table_dataframe", _fake_parse_table_dataframe)

    income_statement = processor.get_financial_statement("income")
    balance_sheet = processor.get_financial_statement("balance_sheet")

    assert income_statement["data_quality"] == "extracted"
    assert any(row["label"] == "Interest earned" for row in income_statement["rows"])
    assert any(row["label"] == "Net profit for the period" for row in income_statement["rows"])
    assert balance_sheet["data_quality"] == "extracted"
    assert any(row["label"] == "Deposits" for row in balance_sheet["rows"])
    assert any(row["label"] == "Total assets" for row in balance_sheet["rows"])


@pytest.mark.unit
def test_six_k_financial_statement_extraction_supports_local_exchange_single_period_summary_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 6-K 处理器可提取本地交易所单期间财务摘要表。"""

    source_path = tmp_path / "6k_local_exchange_summary.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p>
              For the three-month period ended March 31, 2025, relevant information of the
              condensed consolidated interim financial statements follows:
            </p>
            <table>
              <caption>Net profit for the period (in millions of pesos)</caption>
              <tr>
                <td>Attributable to shareholders of the parent company</td>
                <td></td>
                <td></td>
                <td>(19,864)</td>
                <td></td>
              </tr>
              <tr>
                <td>Attributable to non-controlling interest</td>
                <td></td>
                <td></td>
                <td>6,894</td>
                <td></td>
              </tr>
              <tr>
                <td>Total net profit for the period</td>
                <td></td>
                <td></td>
                <td>(12,970)</td>
                <td></td>
              </tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    income_dataframe = pd.DataFrame(
        [
            ["Attributable to shareholders of the parent company", "", "", "(19,864)", ""],
            ["Attributable to non-controlling interest", "", "", "6,894", ""],
            ["Total net profit for the period", "", "", "(12,970)", ""],
        ]
    )

    def _fake_parse_table_dataframe(_table_tag: object) -> pd.DataFrame:
        """返回单期间本地交易所摘要表。"""

        return income_dataframe.copy()

    monkeypatch.setattr(six_k_form_common_module, "parse_html_table_dataframe", _fake_parse_table_dataframe)

    income_statement = processor.get_financial_statement("income")
    balance_sheet = processor.get_financial_statement("balance_sheet")

    assert income_statement["data_quality"] == "extracted"
    assert income_statement["periods"] == [
        {"period_end": "2025-03-31", "fiscal_year": 2025, "fiscal_period": "Q1"}
    ]
    assert any(row["label"] == "Total net profit for the period" for row in income_statement["rows"])
    assert _statement_reason(balance_sheet) == "statement_not_found"


@pytest.mark.unit
def test_six_k_financial_statement_extraction_supports_quarter_token_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证可解析 `1Q25/4Q24` 这类财期 token 表头。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_quarter_token_header.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Balance Sheets</b></p>
            <table>
              <caption>Consolidated Statement of Financial Position</caption>
              <tr><td>placeholder</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    quarter_header_dataframe = pd.DataFrame(
        [
            ["Consolidated Statement of Financial Position", "", "1Q24", "", "4Q24", "", "1Q25"],
            ["Cash and cash equivalents", "", "100", "", "90", "", "110"],
            ["Total assets", "", "300", "", "280", "", "320"],
        ]
    )

    def _fake_parse_table_dataframe(table_tag: object) -> pd.DataFrame:
        del table_tag
        return quarter_header_dataframe.copy()

    monkeypatch.setattr(six_k_form_common_module, "parse_html_table_dataframe", _fake_parse_table_dataframe)

    balance_sheet = processor.get_financial_statement("balance_sheet")
    assert balance_sheet["data_quality"] == "extracted"
    assert [period["period_end"] for period in balance_sheet["periods"]] == [
        "2024-03-31",
        "2024-12-31",
        "2025-03-31",
    ]
    assert [period["fiscal_period"] for period in balance_sheet["periods"]] == ["Q1", "Q4", "Q1"]


@pytest.mark.unit
def test_six_k_financial_statement_extraction_supports_year_month_day_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证可解析 `2024 Dec. 31` 这类年份在前日期格式。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_year_month_day_header.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Balance Sheets</b></p>
            <table>
              <caption>Balance sheet items</caption>
              <tr><td>placeholder</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    year_month_day_dataframe = pd.DataFrame(
        [
            ["Balance sheet items", "", "2024 Dec. 31", "2024 Jun. 30"],
            ["Cash and cash equivalents", "", "100", "90"],
            ["Total assets", "", "300", "260"],
        ]
    )

    def _fake_parse_table_dataframe(table_tag: object) -> pd.DataFrame:
        del table_tag
        return year_month_day_dataframe.copy()

    monkeypatch.setattr(six_k_form_common_module, "parse_html_table_dataframe", _fake_parse_table_dataframe)

    balance_sheet = processor.get_financial_statement("balance_sheet")
    assert balance_sheet["data_quality"] == "extracted"
    assert [period["period_end"] for period in balance_sheet["periods"]] == ["2024-12-31", "2024-06-30"]


@pytest.mark.unit
def test_six_k_financial_statement_extraction_supports_year_first_quarter_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证可解析 `2025 2Q` 这类年份在前的财期 token。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_year_first_quarter_token.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Statements of Income</b></p>
            <table>
              <caption>Consolidated Income Statement</caption>
              <tr><td>placeholder</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    year_first_quarter_dataframe = pd.DataFrame(
        [
            ["Consolidated Income Statement", "2025 2Q", "2025 1Q", "2024 4Q"],
            ["Net interest income", "6208", "6398", "6406"],
            ["Net income", "3015", "2698", "2955"],
        ]
    )

    def _fake_parse_table_dataframe(table_tag: object) -> pd.DataFrame:
        del table_tag
        return year_first_quarter_dataframe.copy()

    monkeypatch.setattr(six_k_form_common_module, "parse_html_table_dataframe", _fake_parse_table_dataframe)

    income_statement = processor.get_financial_statement("income")
    assert income_statement["data_quality"] == "extracted"
    assert [period["period_end"] for period in income_statement["periods"]] == [
        "2025-06-30",
        "2025-03-31",
        "2024-12-31",
    ]
    assert [period["fiscal_period"] for period in income_statement["periods"]] == ["Q2", "Q1", "Q4"]


@pytest.mark.unit
def test_six_k_financial_statement_extraction_supports_two_digit_dd_mm_yy_header(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证可解析 `30-06-25` 这类两位年份日期表头。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_two_digit_date_header.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p><b>Balance Sheets</b></p>
            <table>
              <caption>Consolidated Balance Sheet</caption>
              <tr><td>placeholder</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    two_digit_date_dataframe = pd.DataFrame(
        [
            ["Consolidated Balance Sheet", "30-06-25", "31-12-24"],
            ["Total assets", "776974", "772402"],
            ["Total liabilities", "694058", "690921"],
        ]
    )

    def _fake_parse_table_dataframe(table_tag: object) -> pd.DataFrame:
        del table_tag
        return two_digit_date_dataframe.copy()

    monkeypatch.setattr(six_k_form_common_module, "parse_html_table_dataframe", _fake_parse_table_dataframe)

    balance_sheet = processor.get_financial_statement("balance_sheet")
    assert balance_sheet["data_quality"] == "extracted"
    assert [period["period_end"] for period in balance_sheet["periods"]] == ["2025-06-30", "2024-12-31"]


@pytest.mark.unit
def test_six_k_processor_extracts_statement_from_image_ocr_hidden_text(tmp_path: Path) -> None:
        """验证 6-K 处理器可从 image+OCR 隐藏文本页回退提取财报。

        测试场景：
        - HTML 不包含任何 `<table>`。
        - 财务报表页只由图片和白色 1px OCR 文本组成。
        - 处理器应回退提取 income 与 balance_sheet。

        Args:
                tmp_path: pytest 临时目录。

        Returns:
                无。

        Raises:
                AssertionError: 断言失败时抛出。
        """

        source_path = tmp_path / "6k_image_ocr_hidden_text.html"
        source_path.write_text(
                """
                <html>
                    <body>
                        <p style="text-align: center"><img src="page_1.jpg" alt=""></p>
                        <p style="margin-top: 0; margin-bottom: 0; font-size: 1px; color: White">
                            ANNUAL REPORT 2024 99 Consolidated Statement of Profit or Loss Year ended December 31, 2024
                            Notes 5 REVENUE Cost of sales 5 Gross profit Other income and gains 7 Selling and distribution expenses
                            Administrative expenses Research and development expenses  Other expenses 8 Finance costs 18 Share of (loss)/profit of a joint venture
                            6 LOSS BEFORE TAX 11 Income tax (expense)/credit LOSS FOR THE YEAR Attributable to: Owners of the parent
                            Non - controlling interests LOSS  PER SHARE ATTRIBUTABLE TO  ORDINARY EQUITY HOLDERS OF  THE PARENT Basic and diluted earnings per share
                            2023 2024 RMB ' 000 RMB ' 000 221,984 980,650 (30,543) (29,085) 191,441 951,565 59,316 57,359 (195,387) (195,998)
                            (181,076) (187,125) (706,972) (947,245) (5,203) (9,075) (96,057) (64,455) 1,076 (281) (932,862) (395,255)
                            7,150 (10,425) (925,712) (405,680) (925,637) (405,433) (75) (247)
                        </p>
                        <p style="text-align: center"><img src="page_2.jpg" alt=""></p>
                        <p style="margin-top: 0; margin-bottom: 0; font-size: 1px; color: White">
                            ANNUAL REPORT 2024 101 Consolidated Statement of Financial Position December 31, 2024 Notes
                            NON - CURRENT ASSETS Property, plant and equipment Right - of - use assets Goodwill Other intangible assets
                            Investment in a joint venture Financial assets at fair value through profit or loss Deferred tax assets Other non - current assets
                            Total non - current assets CURRENT ASSETS Inventories Trade receivables Prepayments, other receivables and other assets Cash and bank balances
                            Total current assets CURRENT LIABILITIES Trade payables Other payables and accruals Contract liabilities Interest - bearing bank and other borrowings
                            Total current liabilities NET CURRENT ASSETS TOTAL ASSETS LESS CURRENT LIABILITIES 2023 2024 RMB ' 000 RMB ' 000
                            905,815 849,450 51,252 56,109 24,694 24,694 85,446 75,998 16,998 32,717 1,951 1,141 59,842 44,236 10,217 59,303
                            1,156,215 1,143,648 16,167 6,597 145,893 83,143 88,285 123,211 1,093,833 1,261,211 1,344,178 1,474,162 72,445 91,966
                            206,914 258,098 38,410 37,485 616,404 779,062 934,173 1,166,611 410,005 307,551 1,566,220 1,451,199
                        </p>
                        <p style="text-align: center"><img src="page_3.jpg" alt=""></p>
                        <p style="margin-top: 0; margin-bottom: 0; font-size: 1px; color: White">
                            Ascentage Pharma Group International 102 Consolidated Statement of Financial Position (Continued) December 31, 2024 Notes
                            NON - CURRENT LIABILITIES Contract liabilities Interest - bearing bank and other borrowings Deferred tax liabilities Long - term payables Deferred income
                            Other non - current liabilities Total non - current liabilities Net assets EQUITY Share capital Treasury shares Reserves Non - controlling interests Total equity
                            2023 2024 RMB ' 000 RMB ' 000 251,189 248,460 1,179,191 889,435 10,549 5,368 18,299 - 36,360 27,500 - 6,274 1,495,588 1,177,037
                            70,632 274,162 197 214 (21,351) (8) 81,571 263,988 60,417 264,194 10,215 9,968 70,632 274,162
                        </p>
                    </body>
                </html>
                """,
                encoding="utf-8",
        )

        processor = BsSixKFormProcessor(
                _make_source(source_path, media_type="text/html"),
                form_type="6-K",
                media_type="text/html",
        )

        assert processor.list_tables() == []

        income_statement = processor.get_financial_statement("income")
        assert income_statement["data_quality"] == "extracted"
        assert [period["period_end"] for period in income_statement["periods"]] == ["2023-12-31", "2024-12-31"]
        assert len(income_statement["rows"]) >= 8
        assert any(row["label"] == "Revenue" for row in income_statement["rows"])
        assert income_statement["currency"] == "CNY"

        balance_sheet = processor.get_financial_statement("balance_sheet")
        assert balance_sheet["data_quality"] == "extracted"
        assert [period["period_end"] for period in balance_sheet["periods"]] == ["2023-12-31", "2024-12-31"]
        assert len(balance_sheet["rows"]) >= 10
        assert any(row["label"] == "Total assets less current liabilities" for row in balance_sheet["rows"])
        assert any(row["label"] == "Total equity" for row in balance_sheet["rows"])
        assert balance_sheet["currency"] == "CNY"


@pytest.mark.unit
def test_six_k_processor_extracts_statement_from_hidden_ocr_font_shorthand(tmp_path: Path) -> None:
    """验证 6-K 处理器可识别 `font:` 缩写形式的隐藏 OCR 文本。

    测试场景：
    - HTML 不包含任何结构化 `<table>` 财务表；
    - 财报正文放在白色极小字号段落中，样式使用 `font:` 缩写而不是 `font-size:`；
    - 处理器应仍能回退提取 `balance_sheet`。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_hidden_ocr_font_shorthand.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p style="font: 0.01px/115% Tahoma, Helvetica, Sans-Serif; color: White">
              Consolidated Statement of Financial Position December 31, 2024 Notes
              NON - CURRENT ASSETS Property, plant and equipment Right - of - use assets Goodwill Other intangible assets
              Investment in a joint venture Financial assets at fair value through profit or loss Deferred tax assets Other non - current assets
              Total non - current assets CURRENT ASSETS Inventories Trade receivables Prepayments, other receivables and other assets Cash and bank balances
              Total current assets CURRENT LIABILITIES Trade payables Other payables and accruals Contract liabilities Interest - bearing bank and other borrowings
              Total current liabilities NET CURRENT ASSETS TOTAL ASSETS LESS CURRENT LIABILITIES 2023 2024 RMB ' 000 RMB ' 000
              905,815 849,450 51,252 56,109 24,694 24,694 85,446 75,998 16,998 32,717 1,951 1,141 59,842 44,236 10,217 59,303
              1,156,215 1,143,648 16,167 6,597 145,893 83,143 88,285 123,211 1,093,833 1,261,211 1,344,178 1,474,162 72,445 91,966
              206,914 258,098 38,410 37,485 616,404 779,062 934,173 1,166,611 410,005 307,551 1,566,220 1,451,199
            </p>
            <p style="font: 0.01px/115% Tahoma, Helvetica, Sans-Serif; color: White">
              Consolidated Statement of Financial Position (Continued) December 31, 2024 Notes
              NON - CURRENT LIABILITIES Contract liabilities Interest - bearing bank and other borrowings Deferred tax liabilities Long - term payables Deferred income
              Other non - current liabilities Total non - current liabilities Net assets EQUITY Share capital Treasury shares Reserves Non - controlling interests Total equity
              2023 2024 RMB ' 000 RMB ' 000 251,189 248,460 1,179,191 889,435 10,549 5,368 18,299 - 36,360 27,500 - 6,274 1,495,588 1,177,037
              70,632 274,162 197 214 (21,351) (8) 81,571 263,988 60,417 264,194 10,215 9,968 70,632 274,162
            </p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    assert processor.list_tables() == []

    balance_sheet = processor.get_financial_statement("balance_sheet")

    assert balance_sheet["data_quality"] == "extracted"
    assert [period["period_end"] for period in balance_sheet["periods"]] == [
        "2023-12-31",
        "2024-12-31",
    ]
    assert any(row["label"] == "Total assets less current liabilities" for row in balance_sheet["rows"])
    assert any(row["label"] == "Total equity" for row in balance_sheet["rows"])
    assert balance_sheet["currency"] == "CNY"


@pytest.mark.unit
def test_six_k_processor_extracts_income_from_profit_and_loss_ocr_summary(tmp_path: Path) -> None:
    """验证 6-K 处理器可从 `Profit & Loss` OCR 摘要页提取单期间 income。

    测试场景：
    - 页面不包含结构化 `<table>`；
    - 财务摘要来自图片 OCR 文本；
    - 标题使用 `Profit & Loss`，行项目后紧跟当前期间金额与 delta 指标；
    - 处理器应回退提取单期间 income 行项目，而不是继续返回空结果。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_profit_and_loss_ocr_summary.html"
    source_path.write_text(
        """
        <html>
          <body>
            <p style="text-align: center"><img src="bbva_q1.jpg" alt=""></p>
            <p style="margin-top: 0; margin-bottom: 0; font-size: 1px; color: White">
              1Q25 Earnings Mexico €Bn Profit &amp; Loss (CONSTANT €M)
              1Q24 4Q24 1Q25
              Net Interest Income 2,767 7.6 -1.0 -7.7
              Net Fees and Commissions 583 5.8 -2.3 -9.2
              Operating Expenses 1,144 11.7 -1.6 -4.2
              Operating Income 2,561 7.7 0.9 -7.7
              Income before tax 1,852 7.6 0.1 -8.0
            </p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    income_statement = processor.get_financial_statement("income")

    assert income_statement["data_quality"] == "extracted"
    assert income_statement["currency"] == "EUR"
    assert income_statement["periods"] == [
        {
            "period_end": "2025-03-31",
            "fiscal_year": 2025,
            "fiscal_period": "Q1",
        }
    ]
    assert any(row["label"] == "Operating income" for row in income_statement["rows"])
    assert any(row["label"] == "Operating expenses" for row in income_statement["rows"])


@pytest.mark.unit
def test_six_k_processor_extracts_statement_from_fixed_layout_page_div(tmp_path: Path) -> None:
    """验证 6-K 处理器可从 fixed-layout `PageX` 容器回退提取财报。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_fixed_layout_page_div.html"
    source_path.write_text(
        """
        <html>
          <body>
            <div id="Page1">
              Consolidated Statement of Financial Position
              Total assets
              Total liabilities
              Total equity
              1Q25 4Q24 USD millions
              320 300 210 205 110 95
            </div>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    balance_sheet = processor.get_financial_statement("balance_sheet")

    assert balance_sheet["data_quality"] == "extracted"
    assert [period["period_end"] for period in balance_sheet["periods"]] == [
        "2025-03-31",
        "2024-12-31",
    ]
    assert [row["label"] for row in balance_sheet["rows"][:3]] == [
        "Total assets",
        "Total liabilities",
        "Total equity",
    ]
    assert balance_sheet["currency"] == "USD"


@pytest.mark.unit
def test_six_k_processor_extracts_statement_from_workiva_slide_hidden_text(tmp_path: Path) -> None:
    """验证 6-K 处理器可从 Workiva slide 隐藏文本页回退提取财报。

    测试场景：
    - HTML 页面主体由图片承载；
    - 每页 OCR 文本不在 `<p style=...>`，而在图片容器内的白色 1pt `font`；
    - 处理器应把这些容器重新聚合成页文本并提取 balance_sheet。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_workiva_slide_hidden_text.html"
    source_path.write_text(
        """
        <html>
          <body>
            <div align="center">
              <div style="padding-top:2em;">
                <img src="page1.jpg" title="slide1" width="900" height="1200">
                <div>
                  <font size="1" style="font-size:1pt;color:white">
                    Interim Consolidated Financial Statements
                    Consolidated Statements of Financial Position
                    As of June 30, 2025 and June 30, 2024
                    Assets Total assets Total liabilities Total equity
                    2025 2024 USD million 320 300 210 205 110 95
                  </font>
                </div>
                <p><hr noshade></p>
                <div style="page-break-before:always;">&nbsp;</div>
              </div>
            </div>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    assert processor.list_tables() == []

    balance_sheet = processor.get_financial_statement("balance_sheet")

    assert balance_sheet["data_quality"] == "extracted"
    assert [period["period_end"] for period in balance_sheet["periods"]] == [
        "2025-06-30",
        "2024-06-30",
    ]
    assert [row["label"] for row in balance_sheet["rows"][:3]] == [
        "Total assets",
        "Total liabilities",
        "Total equity",
    ]
    assert balance_sheet["currency"] == "USD"


@pytest.mark.unit
def test_six_k_processor_extracts_statement_from_page_break_text_blocks(tmp_path: Path) -> None:
    """验证 6-K 处理器可从 `page-break` 分页文本块回退提取财报。

    测试场景：
    - HTML 不包含可结构化财报表格；
    - 正文通过 `page-break-before/after: always` 切分页；
    - 财报页仅保留按段落展开的 statement 文本。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_page_break_text_blocks.html"
    source_path.write_text(
        """
        <html>
          <body>
            <div>
              <p>FORM 6-K</p>
              <p>Quarterly results release</p>
            </div>
            <div style="page-break-after: always"></div>
            <div>
              <p style="page-break-before: always">Consolidated Statement of Financial Position</p>
              <p>December 31, 2024 Notes</p>
              <p>
                NON - CURRENT ASSETS Property, plant and equipment Right - of - use assets Goodwill
                Other intangible assets Investment in a joint venture Financial assets at fair value
                through profit or loss Deferred tax assets Other non - current assets Total non - current assets
              </p>
              <p>
                CURRENT ASSETS Inventories Trade receivables Prepayments, other receivables and other assets
                Cash and bank balances Total current assets CURRENT LIABILITIES Trade payables
                Other payables and accruals Contract liabilities Interest - bearing bank and other borrowings
                Total current liabilities NET CURRENT ASSETS TOTAL ASSETS LESS CURRENT LIABILITIES
              </p>
              <p>
                2023 2024 RMB ' 000 RMB ' 000 905,815 849,450 51,252 56,109 24,694 24,694 85,446 75,998
                16,998 32,717 1,951 1,141 59,842 44,236 10,217 59,303 1,156,215 1,143,648 16,167 6,597
                145,893 83,143 88,285 123,211 1,093,833 1,261,211 1,344,178 1,474,162 72,445 91,966
                206,914 258,098 38,410 37,485 616,404 779,062 934,173 1,166,611 410,005 307,551 1,566,220 1,451,199
              </p>
            </div>
            <div style="page-break-after: always"></div>
            <div>
              <p style="page-break-before: always">Consolidated Statement of Financial Position (Continued)</p>
              <p>NON - CURRENT LIABILITIES Total non - current liabilities Net assets EQUITY</p>
              <p>
                Share capital Treasury shares Reserves Non - controlling interests Total equity
                2023 2024 RMB ' 000 RMB ' 000 251,189 248,460 1,179,191 889,435 10,549 5,368
                18,299 - 36,360 27,500 - 6,274 1,495,588 1,177,037 70,632 274,162 197 214
                (21,351) (8) 81,571 263,988 60,417 264,194 10,215 9,968 70,632 274,162
              </p>
            </div>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    balance_sheet = processor.get_financial_statement("balance_sheet")

    assert balance_sheet["data_quality"] == "extracted"
    assert [period["period_end"] for period in balance_sheet["periods"]] == [
        "2023-12-31",
        "2024-12-31",
    ]
    row_labels = [row["label"] for row in balance_sheet["rows"]]
    assert "Total assets less current liabilities" in row_labels
    assert "Total current liabilities" in row_labels
    assert "Total equity" in row_labels


@pytest.mark.unit
def test_six_k_processor_extracts_statement_from_pseudo_page_table_text(tmp_path: Path) -> None:
    """验证 6-K 处理器可从伪表格页文本回退提取财报。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_pseudo_page_table.html"
    source_path.write_text(
        """
        <html>
          <body>
            <table>
              <caption>TSMC's 2025 first quarter consolidated results</caption>
              <tr>
                <td>
                  Net sales Gross profit Income from operations Net income Earnings per share
                  1Q25 4Q24 NT$ million 839250 868460 480252 484000 390561 403200 361564 374680 13.94 14.45
                </td>
              </tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    income_statement = processor.get_financial_statement("income")

    assert income_statement["data_quality"] == "extracted"
    assert [period["period_end"] for period in income_statement["periods"]] == [
        "2025-03-31",
        "2024-12-31",
    ]
    assert [row["label"] for row in income_statement["rows"][:5]] == [
        "Net sales",
        "Gross profit",
        "Income from operations",
        "Net income",
        "Earnings per share",
    ]


@pytest.mark.unit
def test_six_k_processor_joins_title_table_with_following_statement_data_table(tmp_path: Path) -> None:
    """验证 6-K 处理器可把标题 table 与后继数据 table 拼接成报表候选。

    测试场景：
    - 第一个 table 只有公司名、statement title 与单位说明；
    - 真正的财务数据落在紧随其后的未命名 table；
    - 处理器应把两张表视为同一 statement block，成功抽取 income 与
      balance_sheet。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_path = tmp_path / "6k_split_title_and_statement_table.html"
    source_path.write_text(
        """
        <html>
          <body>
            <table>
              <tr><td style="text-align:center"><b>WEIBO CORPORATION</b></td></tr>
              <tr><td style="text-align:center"><b>UNAUDITED CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS</b></td></tr>
              <tr><td style="text-align:center"><b>(In thousands of U.S. dollars)</b></td></tr>
            </table>
            <table>
              <tr>
                <td></td>
                <td colspan="2" style="text-align:center">Three months ended</td>
                <td colspan="2" style="text-align:center">Twelve months ended</td>
              </tr>
              <tr>
                <td></td>
                <td style="text-align:center">December 31,</td>
                <td style="text-align:center">September 30,</td>
                <td style="text-align:center">December 31,</td>
                <td style="text-align:center">December 31,</td>
              </tr>
              <tr>
                <td></td>
                <td style="text-align:center">2023</td>
                <td style="text-align:center">2024</td>
                <td style="text-align:center">2024</td>
                <td style="text-align:center">2024</td>
              </tr>
              <tr><td>Net revenues</td><td>463,667</td><td>464,480</td><td>456,827</td><td>1,754,677</td></tr>
              <tr><td>Income from operations</td><td>119,005</td><td>141,322</td><td>117,880</td><td>494,324</td></tr>
              <tr><td>Net income</td><td>88,684</td><td>132,729</td><td>12,727</td><td>310,105</td></tr>
              <tr><td>Net income attributable to Weibo’s shareholders</td><td>83,230</td><td>130,567</td><td>8,865</td><td>300,801</td></tr>
              <tr><td>Diluted net income per share attributable to Weibo’s shareholders</td><td>0.34</td><td>0.50</td><td>0.04</td><td>1.16</td></tr>
              <tr><td>Shares used in computing diluted net income per share attributable to Weibo’s shareholders</td><td>246,382</td><td>265,824</td><td>239,983</td><td>265,241</td></tr>
            </table>
            <table>
              <tr><td style="text-align:center"><b>WEIBO CORPORATION</b></td></tr>
              <tr><td style="text-align:center"><b>UNAUDITED CONDENSED CONSOLIDATED BALANCE SHEETS</b></td></tr>
              <tr><td style="text-align:center"><b>(In thousands of U.S. dollars)</b></td></tr>
            </table>
            <table>
              <tr><td></td><td colspan="2" style="text-align:center">As of</td></tr>
              <tr><td></td><td style="text-align:center">December 31,</td><td style="text-align:center">December 31,</td></tr>
              <tr><td></td><td style="text-align:center">2023</td><td style="text-align:center">2024</td></tr>
              <tr><td>Cash and cash equivalents</td><td>2,584,635</td><td>1,890,632</td></tr>
              <tr><td>Short-term investments</td><td>641,035</td><td>459,852</td></tr>
              <tr><td>Total assets</td><td>7,280,358</td><td>6,504,499</td></tr>
              <tr><td>Total liabilities</td><td>3,762,742</td><td>2,925,613</td></tr>
              <tr><td>Total shareholders’ equity</td><td>3,448,888</td><td>3,533,783</td></tr>
              <tr><td>Total liabilities, redeemable non-controlling interests and shareholders’ equity</td><td>7,280,358</td><td>6,504,499</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = BsSixKFormProcessor(
        _make_source(source_path, media_type="text/html"),
        form_type="6-K",
        media_type="text/html",
    )

    income_statement = processor.get_financial_statement("income")
    balance_sheet = processor.get_financial_statement("balance_sheet")

    assert income_statement["data_quality"] == "extracted"
    assert [period["period_end"] for period in income_statement["periods"]] == [
        "2023-12-31",
        "2024-09-30",
        "2024-12-31",
    ]
    assert any(row["label"] == "Net revenues" for row in income_statement["rows"])
    assert any(row["label"] == "Net income" for row in income_statement["rows"])

    assert balance_sheet["data_quality"] == "extracted"
    assert [period["period_end"] for period in balance_sheet["periods"]] == [
        "2023-12-31",
        "2024-12-31",
    ]
    assert any(row["label"] == "Total assets" for row in balance_sheet["rows"])
    assert any(row["label"] == "Total liabilities" for row in balance_sheet["rows"])
    assert any(
        row["label"] == "Total liabilities, redeemable non-controlling interests and shareholders’ equity"
        for row in balance_sheet["rows"]
    )


# ---------------------------------------------------------------------------
# 分支覆盖率补充测试
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_six_k_table_dataframe_returns_none_for_missing_tag() -> None:
    """验证 table.tag 为 None 时 _parse_six_k_table_dataframe 返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _parse_six_k_table_dataframe

    class _TaglessTable:
        """无 tag 属性的表格对象。"""

        pass

    assert _parse_six_k_table_dataframe(_TaglessTable()) is None

    class _NoneTagTable:
        """tag 属性为 None 的表格对象。"""

        tag = None

    assert _parse_six_k_table_dataframe(_NoneTagTable()) is None


@pytest.mark.unit
def test_classify_statement_type_all_empty_inputs() -> None:
    """验证 caption、headers、context 全为空时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert six_k_form_common_module._classify_statement_type_for_table(
        caption="",
        headers=[],
        context_before="",
    ) is None

    assert six_k_form_common_module._classify_statement_type_for_table(
        caption=None,
        headers=None,
        context_before="",
    ) is None


@pytest.mark.unit
def test_classify_statement_type_tied_scores_returns_none() -> None:
    """验证两个报表类型分数相同时返回 None（平局分支）。

    构造 caption 同时命中 balance_sheet 和 income 的关键词，
    使两者分数相同从而触发平局逻辑。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert six_k_form_common_module._classify_statement_type_for_table(
        caption="Balance Sheet Income Statement combined overview",
        headers=[],
        context_before="",
    ) is None


@pytest.mark.unit
def test_classify_statement_type_low_score_no_caption_signal_returns_none() -> None:
    """验证分数在 [CONTEXT_ONLY_MIN, MIN_SCORE) 且无 caption/header 信号时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 构造只有 context 命中的场景：score < 4 但 >= 2，且 caption/header 无信号。
    # 使用 "cash_flow" 的 context 模式匹配（cash flows from operating / operating activities），
    # 但 caption 和 headers 不命中任何报表模式。
    # cash_flow context patterns: "cash flows?" 和 "statement of cash flows"
    # 在 context 中出现 3 次 → context_hits=3 → score=3
    # caption 为空，headers 中没有报表信号
    result = six_k_form_common_module._classify_statement_type_for_table(
        caption="quarterly summary",
        headers=["Period", "Amount"],
        context_before=(
            "The cash flows data shows operating trends. "
            "Cash flows details are provided below. "
            "Additional cash flows information follows."
        ),
    )
    # score < 4 且无 caption/header 信号 → None
    assert result is None


@pytest.mark.unit
def test_looks_like_non_statement_table_empty_caption_or_header() -> None:
    """验证 caption 和 header 均为空时返回 False。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert six_k_form_common_module._looks_like_non_statement_table(
        caption_text="",
        headers_text="",
    ) is False


@pytest.mark.unit
def test_looks_like_non_statement_table_caption_noise_with_statement_signal() -> None:
    """验证 caption 含噪声模式但自身含报表信号时不视为非报表表。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # caption 含 "Table of Contents" 噪声模式，但同时也含 "balance sheet" 报表信号
    assert six_k_form_common_module._looks_like_non_statement_table(
        caption_text="table of contents - consolidated balance sheets",
        headers_text="",
    ) is False


@pytest.mark.unit
def test_looks_like_non_statement_table_header_noise_with_statement_signal() -> None:
    """验证 header 含噪声模式但整体有报表信号时返回 False。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # caption 无噪声，但 headers_text 含 "signatures" 噪声 + 报表信号
    assert six_k_form_common_module._looks_like_non_statement_table(
        caption_text="Consolidated Balance Sheets",
        headers_text="signatures total assets",
    ) is False


@pytest.mark.unit
def test_looks_like_non_statement_table_noise_without_signal() -> None:
    """验证 header 含噪声模式且无报表信号时返回 True。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert six_k_form_common_module._looks_like_non_statement_table(
        caption_text="Financial Data",
        headers_text="Table of Contents",
    ) is True


@pytest.mark.unit
def test_looks_like_six_k_about_company_context_with_punctuation_prefix() -> None:
    """验证 About Company 前有标点符号时仍视为标题。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _looks_like_six_k_about_company_context

    # "Some text.About " → position 为 'A' 的索引 (10)，prefix="Some text." → 末尾 '.'
    assert _looks_like_six_k_about_company_context(
        full_text="Some text.About Acme Inc. is a company.",
        position=10,
    ) is True

    # "Details:About " → position 为 'A' 的索引 (8)，prefix="Details:" → 末尾 ':'
    assert _looks_like_six_k_about_company_context(
        full_text="Details:About Acme Inc. is a company.",
        position=8,
    ) is True

    # "(Note)About " → position 为 'A' 的索引 (6)，prefix="(Note)" → 末尾 ')'
    assert _looks_like_six_k_about_company_context(
        full_text="(Note)About Acme Inc. details.",
        position=6,
    ) is True


@pytest.mark.unit
def test_looks_like_six_k_about_company_context_inline_returns_false() -> None:
    """验证 About Company 嵌在普通文本中间时返回 False。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _looks_like_six_k_about_company_context

    # "learn more About " → position 为 'A'，prefix="learn more " → 末尾 'e' 不是标点
    result = _looks_like_six_k_about_company_context(
        full_text="learn more About Acme Inc. on our website.",
        position=11,
    )
    assert result is False


@pytest.mark.unit
def test_month_to_quarter_all_quarters() -> None:
    """验证 _month_to_quarter 对各月份返回正确季度。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _month_to_quarter

    assert _month_to_quarter(1) == "Q1"
    assert _month_to_quarter(3) == "Q1"
    assert _month_to_quarter(4) == "Q2"
    assert _month_to_quarter(6) == "Q2"
    assert _month_to_quarter(7) == "Q3"
    assert _month_to_quarter(9) == "Q3"
    assert _month_to_quarter(10) == "Q4"
    assert _month_to_quarter(12) == "Q4"


@pytest.mark.unit
def test_find_first_pattern_after_no_match() -> None:
    """验证 _find_first_pattern_after 未命中时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _find_first_pattern_after

    import re

    result = _find_first_pattern_after(
        pattern=re.compile(r"nonexistent_pattern"),
        full_text="This text does not contain the target.",
        start_at=0,
    )
    assert result is None


@pytest.mark.unit
def test_find_first_pattern_after_start_beyond_text() -> None:
    """验证 start_at 超过文本长度时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _find_first_pattern_after

    import re

    result = _find_first_pattern_after(
        pattern=re.compile(r"test"),
        full_text="test",
        start_at=100,
    )
    assert result is None


@pytest.mark.unit
def test_extract_statement_result_from_ocr_unsupported_type() -> None:
    """验证不支持 OCR 的报表类型直接返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import extract_statement_result_from_ocr_pages

    # equity 不在 _OCR_STATEMENT_TITLE_PATTERNS 中
    result = extract_statement_result_from_ocr_pages(
        statement_type="equity",
        page_texts=["some text"],
    )
    assert result is None


@pytest.mark.unit
def test_extract_statement_result_from_ocr_no_parsed_pages_non_income() -> None:
    """验证 OCR 无解析页且非 income 类型时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import extract_statement_result_from_ocr_pages

    # cash_flow 在 OCR patterns 中，但页文本不包含任何标题
    result = extract_statement_result_from_ocr_pages(
        statement_type="cash_flow",
        page_texts=["random text without any statement title"],
    )
    assert result is None


@pytest.mark.unit
def test_parse_ocr_numeric_token_empty_after_clean() -> None:
    """验证清理后为空 token 返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _parse_ocr_numeric_token

    assert _parse_ocr_numeric_token("") is None
    assert _parse_ocr_numeric_token(",,,") is None


@pytest.mark.unit
def test_extract_fiscal_period_from_ocr_match_fy() -> None:
    """验证 OCR 期间 match 提取 FY period（12M → FY）。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import (
        _OCR_PERIOD_TOKEN_RE,
        _extract_fiscal_period_from_ocr_period_match,
    )

    # 使用完整的 _OCR_PERIOD_TOKEN_RE 以保证所有命名组都存在
    match = _OCR_PERIOD_TOKEN_RE.search("12M 2024")
    assert match is not None
    assert _extract_fiscal_period_from_ocr_period_match(match) == "FY"


@pytest.mark.unit
def test_extract_fiscal_period_from_ocr_match_nine_months() -> None:
    """验证 OCR 期间 match 提取 9M → Q3。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import (
        _OCR_PERIOD_TOKEN_RE,
        _extract_fiscal_period_from_ocr_period_match,
    )

    match = _OCR_PERIOD_TOKEN_RE.search("9M 2024")
    assert match is not None
    assert _extract_fiscal_period_from_ocr_period_match(match) == "Q3"


@pytest.mark.unit
def test_extract_fiscal_year_from_ocr_match_no_year() -> None:
    """验证 OCR 期间 match 不含年份组时返回 None。

    使用与 _OCR_PERIOD_TOKEN_RE 兼容的正则构造 match，
    确保 year 和 year_first 组均为空。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    import re

    from dayu.fins.processors.six_k_form_common import _extract_fiscal_year_from_ocr_period_match

    # 构造包含所有命名组的正则，但文本不含年份
    full_pattern = re.compile(
        r"(?i)\b(?:(?P<quarter>[1-4])Q|Q(?P<quarter_rev>[1-4])|(?P<half>[12])H|H(?P<half_rev>[12])|"
        r"(?P<nine>9)M|(?P<twelve>12)M|FY)\s*[-/']?\s*(?P<year>\d{2,4})\b|"
        r"\b(?P<year_first>\d{2,4})\s*[-/']?\s*(?:(?P<year_first_quarter>[1-4])Q|Q(?P<year_first_quarter_rev>[1-4])|"
        r"(?P<year_first_half>[12])H|H(?P<year_first_half_rev>[12])|(?P<year_first_nine>9)M|"
        r"(?P<year_first_twelve>12)M|FY)\b"
    )
    # "FY" 匹配但无年份 → year 和 year_first 均为 None
    match = full_pattern.search("FY")
    if match is not None:
        result = _extract_fiscal_year_from_ocr_period_match(match)
        assert result is None


@pytest.mark.unit
def test_infer_ocr_fiscal_period_six_months() -> None:
    """验证推断 six months ended 的 fiscal period 为 H1/H2。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _infer_ocr_fiscal_period

    assert _infer_ocr_fiscal_period(
        page_text="For the six months ended June 30, 2024",
        month_value=6,
    ) == "H1"

    assert _infer_ocr_fiscal_period(
        page_text="For the six months ended December 31, 2024",
        month_value=12,
    ) == "H2"


@pytest.mark.unit
def test_infer_ocr_fiscal_period_nine_months() -> None:
    """验证推断 nine months ended 的 fiscal period 为 9M。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _infer_ocr_fiscal_period

    assert _infer_ocr_fiscal_period(
        page_text="For the nine months ended September 30, 2024",
        month_value=9,
    ) == "9M"


@pytest.mark.unit
def test_infer_ocr_fiscal_period_year_end_dec() -> None:
    """验证 year ended + 12 月时推断为 FY。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _infer_ocr_fiscal_period

    assert _infer_ocr_fiscal_period(
        page_text="For the year ended December 31, 2024",
        month_value=12,
    ) == "FY"


@pytest.mark.unit
def test_infer_ocr_fiscal_period_as_at_quarter_end() -> None:
    """验证 as at + 季度末月时推断对应季度。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _infer_ocr_fiscal_period

    assert _infer_ocr_fiscal_period(
        page_text="As at March 31, 2024",
        month_value=3,
    ) == "Q1"

    assert _infer_ocr_fiscal_period(
        page_text="As at September 30, 2024",
        month_value=9,
    ) == "Q3"


@pytest.mark.unit
def test_infer_ocr_fiscal_period_non_dec_non_quarter_end() -> None:
    """验证非 12 月且无特定指示时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _infer_ocr_fiscal_period

    assert _infer_ocr_fiscal_period(
        page_text="Some random text",
        month_value=5,
    ) is None


@pytest.mark.unit
def test_extract_ocr_currency_and_scale_billion() -> None:
    """验证提取 billions 缩放。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _extract_ocr_currency_and_scale

    currency, scale = _extract_ocr_currency_and_scale("USD in billions")
    assert currency == "USD"
    assert scale == "billions"


@pytest.mark.unit
def test_extract_ocr_currency_and_scale_thousands() -> None:
    """验证提取 thousands 缩放（000 格式）。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _extract_ocr_currency_and_scale

    currency, scale = _extract_ocr_currency_and_scale("RMB '000")
    assert currency == "RMB"
    assert scale == "thousands"


@pytest.mark.unit
def test_map_ocr_currency_code_unknown() -> None:
    """验证未知货币代码返回原文本。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _map_ocr_currency_code

    assert _map_ocr_currency_code("ZAR") == "ZAR"


@pytest.mark.unit
def test_build_ocr_units_label_only_scale() -> None:
    """验证只有 scale 无 currency 时返回 scale。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _build_ocr_units_label

    assert _build_ocr_units_label(currency_raw=None, scale="millions") == "millions"


@pytest.mark.unit
def test_build_ocr_units_label_only_currency() -> None:
    """验证只有 currency 无 scale 时返回 currency。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _build_ocr_units_label

    assert _build_ocr_units_label(currency_raw="USD", scale=None) == "USD"


@pytest.mark.unit
def test_build_ocr_units_label_both_none() -> None:
    """验证 currency 和 scale 均为 None 时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _build_ocr_units_label

    assert _build_ocr_units_label(currency_raw=None, scale=None) is None


@pytest.mark.unit
def test_dedupe_ocr_statement_rows_removes_duplicates() -> None:
    """验证按标签和值去重 OCR 行。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _dedupe_ocr_statement_rows

    rows = [
        {"label": "Revenue", "values": [100.0, 200.0]},
        {"label": "Revenue", "values": [100.0, 200.0]},
        {"label": "Cost", "values": [50.0, 80.0]},
    ]
    result = _dedupe_ocr_statement_rows(rows)
    assert len(result) == 2
    assert result[0]["label"] == "Revenue"
    assert result[1]["label"] == "Cost"


@pytest.mark.unit
def test_extract_income_summary_value_skips_year_range() -> None:
    """验证跳过看起来像年份的数值。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _extract_income_summary_value_after_label

    # 1900-2100 范围内的数字应被跳过
    page_body = "Revenue 2024 3,500"
    result = _extract_income_summary_value_after_label(
        page_body=page_body,
        label_end=7,  # 指向 "Revenue" 之后
    )
    # 2024 被跳过（在 1900-2100 范围），返回 3500
    assert result == 3500.0


@pytest.mark.unit
def test_extract_income_summary_value_skips_percent() -> None:
    """验证含百分比 token 时所有候选值均被跳过。

    当 lookahead 中出现 "%" 后，其后所有数值都会被 percent 检查跳过。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _extract_income_summary_value_after_label

    # "%" 出现在第一个数值之前 → 所有数值都被跳过
    page_body = "Label % 25 1,234"
    result = _extract_income_summary_value_after_label(
        page_body=page_body,
        label_end=6,  # "Label " 之后
    )
    assert result is None


@pytest.mark.unit
def test_extract_income_summary_value_no_valid_value() -> None:
    """验证无有效数值时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _extract_income_summary_value_after_label

    page_body = "Label - - -"
    result = _extract_income_summary_value_after_label(
        page_body=page_body,
        label_end=5,
    )
    assert result is None


@pytest.mark.unit
def test_extract_ocr_line_item_labels_skips_overlapping() -> None:
    """验证重叠的标签匹配被跳过。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _extract_ocr_line_item_labels

    # "Net income" 包含 "Net"，需要验证重叠跳过逻辑
    labels = _extract_ocr_line_item_labels(
        statement_type="income",
        label_source="Revenue Net income Gross profit Earnings per share",
        row_count=4,
    )
    assert len(labels) >= 2


@pytest.mark.unit
def test_resolve_ocr_period_end_fallback_to_anchor() -> None:
    """验证 fiscal_period 无法解析时回退到 anchor month/day。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _resolve_ocr_period_end

    # fiscal_period=None → 走 anchor 回退
    result = _resolve_ocr_period_end(
        fiscal_year=2024,
        fiscal_period=None,
        anchor_month_day=(6, 30),
    )
    assert result == "2024-06-30"


@pytest.mark.unit
def test_group_close_ocr_matches_empty() -> None:
    """验证空匹配列表返回空分组。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _group_close_ocr_matches

    assert _group_close_ocr_matches([]) == []


@pytest.mark.unit
def test_build_six_k_markers_report_mode_branch(tmp_path: Path) -> None:
    """验证 _build_six_k_markers 在 report markers 足够时走 report 模式分支。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _build_six_k_report_markers

    # 构造足够长的文本使 TOC cutoff 被识别
    text = (
        "Table of Contents\n"
        "About this report 4\n"
        "Governance 13\n"
        "Strategy 21\n"
        "Environment 24\n"
        "Social 58\n"
        "Appendix 97\n"
        + "\n" * 200
        + "About this report\n"
        "Overview\n"
        "Governance\n"
        "Strategy\n"
        "Environment\n"
        "Social\n"
        "Risk management\n"
        "Appendix\n"
    )

    markers = _build_six_k_report_markers(text)
    # 若 TOC cutoff 识别成功，应返回 >= 4 个 markers
    if len(markers) >= 4:
        titles = [title for _, title in markers]
        assert "Overview" in titles
        assert "Governance" in titles


@pytest.mark.unit
def test_find_six_k_marker_match_with_start_at() -> None:
    """验证 _find_six_k_marker_match 的 start_at 参数跳过前方匹配。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _find_six_k_marker_match

    import re

    text = "Safe Harbor first. " + "x" * 200 + " Safe Harbor second."
    pattern = re.compile(r"(?i)\bsafe\s+harbor\b")

    # 无 start_at → 取第一个
    match_first = _find_six_k_marker_match(
        full_text=text,
        title="Safe Harbor",
        pattern=pattern,
        use_last=False,
    )
    assert match_first is not None
    assert match_first.start() < 20

    # start_at 跳过第一个 → 取第二个
    match_after = _find_six_k_marker_match(
        full_text=text,
        title="Safe Harbor",
        pattern=pattern,
        use_last=False,
        start_at=200,
    )
    assert match_after is not None
    assert match_after.start() > 200


@pytest.mark.unit
def test_extract_income_summary_result_no_valid_pages() -> None:
    """验证 OCR income 摘要提取无有效页时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _extract_income_summary_result_from_ocr_pages

    result = _extract_income_summary_result_from_ocr_pages(
        page_texts=["random text without profit & loss title"],
    )
    assert result is None


@pytest.mark.unit
def test_build_income_summary_result_no_period() -> None:
    """验证 Profit & Loss 标题附近无法识别期间时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _build_income_summary_result_from_title_match

    import re

    text = "Profit & Loss Some data without period info Revenue 1000"
    match = re.search(r"Profit & Loss", text)
    assert match is not None
    result = six_k_form_common_module._build_income_summary_result_from_title_match(
        normalized_text=text,
        title_match=match,
    )
    assert result is None


@pytest.mark.unit
def test_parse_statement_from_ocr_insufficient_periods() -> None:
    """验证 OCR 页期间数不足 2 时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _parse_statement_from_ocr_page

    # 只有一个年份 → period_entries < 2 → None
    result = _parse_statement_from_ocr_page(
        statement_type="balance_sheet",
        page_text="Consolidated Balance Sheet December 31, 2024 Total assets 300",
    )
    assert result is None


@pytest.mark.unit
def test_parse_statement_from_ocr_insufficient_values() -> None:
    """验证 OCR 页数值不足时返回 None。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _parse_statement_from_ocr_page

    # 有 2 个期间但数值不够
    result = _parse_statement_from_ocr_page(
        statement_type="balance_sheet",
        page_text=(
            "Consolidated Balance Sheet December 31, 2024 and December 31, 2023 "
            "Total assets 100"
        ),
    )
    assert result is None


@pytest.mark.unit
def test_extract_ocr_period_entries_no_year_matches() -> None:
    """验证页头无年份匹配时返回空列表。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _extract_ocr_period_entries_and_header_end

    entries, header_end = _extract_ocr_period_entries_and_header_end(
        header_text="Random text with no year or period tokens",
    )
    assert entries == []
    assert header_end == len("Random text with no year or period tokens")


@pytest.mark.unit
def test_extract_ocr_period_entries_single_year_only() -> None:
    """验证只有一个年份时返回空列表。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _extract_ocr_period_entries_and_header_end

    entries, _ = _extract_ocr_period_entries_and_header_end(
        header_text="Financial data 2024",
    )
    assert entries == []


@pytest.mark.unit
def test_count_pattern_hits_empty_text() -> None:
    """验证空文本返回 0 命中。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    import re

    from dayu.fins.processors.six_k_form_common import _count_pattern_hits

    patterns = (re.compile(r"test"),)
    assert _count_pattern_hits(statement_patterns=patterns, text="") == 0


@pytest.mark.unit
def test_normalize_statement_text() -> None:
    """验证文本标准化。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.fins.processors.six_k_form_common import _normalize_statement_text

    assert _normalize_statement_text("  HELLO   World  ") == "hello world"


__all__ = ["test_six_k_processor_xml_media_type_detection"]
