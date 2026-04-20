"""Web 非 HTML Docling 路径真实集成测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.engine.tools.web_fetch_orchestrator import _docling_convert_to_markdown

pytestmark = pytest.mark.integration


def _fixture_pdf_path() -> Path:
    """返回真实 Docling PDF fixture 路径。

    Args:
        无。

    Returns:
        真实 PDF fixture 文件路径。

    Raises:
        FileNotFoundError: fixture 缺失时抛出。
    """

    fixture_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "docling"
        / "dayu_docling_integration_fixture.pdf"
    )
    if not fixture_path.exists():
        raise FileNotFoundError(f"Docling PDF fixture 不存在: {fixture_path}")
    return fixture_path


def test_docling_convert_to_markdown_reads_real_pdf_bytes() -> None:
    """验证 Web 非 HTML 路径可真实将 PDF 转成 Markdown。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pdf_bytes = _fixture_pdf_path().read_bytes()
    title, markdown, extraction_source = _docling_convert_to_markdown(pdf_bytes, "page.pdf")

    assert extraction_source == "docling"
    assert title == "Dayu Docling Integration Fixture"
    assert "## Financial Summary" in markdown
    assert "| Metric" in markdown
    assert "FY2024" in markdown
    assert "FY2025" in markdown
    assert "| Revenue" in markdown
    assert "| Operating Margin" in markdown
