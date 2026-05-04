"""fins 业务增强处理器测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest

from dayu.fins.processors.fins_bs_processor import FinsBSProcessor
from dayu.fins.processors.fins_docling_processor import FinsDoclingProcessor
from dayu.fins.processors.fins_markdown_processor import FinsMarkdownProcessor
from dayu.fins.processors.sec_html_rules import strip_edgar_sgml_envelope
from dayu.fins.storage.local_file_source import LocalFileSource


def _make_source(path: Path, *, media_type: str) -> LocalFileSource:
    """构建本地 Source。

    Args:
        path: 本地文件路径。
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
def test_fins_bs_processor_relabels_financial_tables(tmp_path: Path) -> None:
    """验证 fins BS 处理器会补充金融语义。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html_path = tmp_path / "sample.html"
    html_path.write_text(
        """
        <html>
          <body>
            <h1>Overview</h1>
            <p>Intro text.</p>
            <table>
              <caption>Consolidated Balance Sheet</caption>
              <tr><th>Item</th><th>2025</th></tr>
              <tr><td>Cash</td><td>100</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = FinsBSProcessor(_make_source(html_path, media_type="text/html"))
    tables = processor.list_tables()
    assert len(tables) == 1
    assert tables[0].get("is_financial") is True
    assert tables[0]["table_type"] == "financial"


@pytest.mark.unit
def test_strip_edgar_sgml_envelope_keeps_only_html_body() -> None:
    """验证共享 SEC HTML helper 会剥离 EDGAR SGML 信封。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    raw = (
        "<DOCUMENT>\n"
        "<TYPE>EX-99.1\n"
        "<SEQUENCE>2\n"
        "<FILENAME>dex991.htm\n"
        "<TEXT>\n"
        "<html><body><p>Content</p></body></html>\n"
        "</TEXT>\n"
        "</DOCUMENT>\n"
    )

    result = strip_edgar_sgml_envelope(raw)
    assert result.startswith("<html>")
    assert "DOCUMENT" not in result
    assert "EX-99.1" not in result
    assert "Content" in result


@pytest.mark.unit
def test_fins_bs_processor_strips_edgar_sgml_metadata(tmp_path: Path) -> None:
    """验证 FinsBSProcessor 会在解析前剥离 EDGAR SGML 元数据。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    raw = (
        "<DOCUMENT>\n"
        "<TYPE>EX-99.1\n"
        "<SEQUENCE>2\n"
        "<FILENAME>dex991.htm\n"
        "<TEXT>\n"
        "<html><body>\n"
        "<h1>Financial Results</h1>\n"
        "<table><tr><th>Revenue</th></tr><tr><td>1000</td></tr></table>\n"
        "</body></html>\n"
        "</TEXT>\n"
        "</DOCUMENT>\n"
    )
    html_path = tmp_path / "exhibit.htm"
    html_path.write_text(raw, encoding="utf-8")

    processor = FinsBSProcessor(_make_source(html_path, media_type="text/html"))
    full_text = processor.get_full_text()

    assert "EX-99.1" not in full_text
    assert "dex991" not in full_text
    assert "SEQUENCE" not in full_text
    assert "Financial Results" in full_text
    assert "Revenue" in full_text


@pytest.mark.unit
def test_fins_bs_processor_filters_sec_layout_tables_with_shared_rules(tmp_path: Path) -> None:
    """验证 FinsBSProcessor 会用共享规则过滤 SEC layout 表格。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html_path = tmp_path / "sec_cover.html"
    html_path.write_text(
        """
        <html>
          <body>
            <table>
              <tr><td>Annual Report Pursuant to Section 13 or 15(d)</td></tr>
              <tr><td>Commission File Number 001-00000 ☒</td></tr>
            </table>
            <table>
              <caption>Consolidated Balance Sheet</caption>
              <tr><th>Item</th><th>2025</th></tr>
              <tr><td>Cash</td><td>100</td></tr>
            </table>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    processor = FinsBSProcessor(_make_source(html_path, media_type="text/html"))
    tables = processor.list_tables()

    assert len(tables) == 1
    assert tables[0]["caption"] == "Consolidated Balance Sheet"


@pytest.mark.unit
def test_fins_markdown_processor_relabels_financial_tables(tmp_path: Path) -> None:
    """验证 fins Markdown 处理器会补充金融语义。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    md_path = tmp_path / "sample.md"
    md_path.write_text(
        "\n".join(
            [
                "# 总览",
                "Consolidated Income Statement",
                "| Item | Value |",
                "| --- | --- |",
                "| Revenue | 100 |",
            ]
        ),
        encoding="utf-8",
    )

    processor = FinsMarkdownProcessor(_make_source(md_path, media_type="text/markdown"))
    tables = processor.list_tables()
    assert len(tables) == 1
    assert tables[0].get("is_financial") is True
    assert tables[0]["table_type"] == "financial"


@dataclass
class _FakeProv:
    """测试用 provenance。"""

    page_no: int


class _FakeLabel:
    """测试用标签对象。"""

    def __init__(self, value: str) -> None:
        """初始化标签。

        Args:
            value: 标签值。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.value = value


class _FakeTextItem:
    """测试用文本 item。"""

    def __init__(self, *, self_ref: str, text: str, label: str, page_no: int) -> None:
        """初始化文本 item。

        Args:
            self_ref: 内部引用。
            text: 文本内容。
            label: 标签值。
            page_no: 页码。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.self_ref = self_ref
        self.text = text
        self.label = _FakeLabel(label)
        self.prov = [_FakeProv(page_no=page_no)]
        self.parent = None


class _FakeTableData:
    """测试用表格 data。"""

    def __init__(self, num_rows: int, num_cols: int) -> None:
        """初始化表格 data。

        Args:
            num_rows: 行数。
            num_cols: 列数。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.num_rows = num_rows
        self.num_cols = num_cols


class _FakeCaption:
    """测试用 caption。"""

    def __init__(self, text: str) -> None:
        """初始化 caption。

        Args:
            text: 标题文本。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.text = text


class _FakeTableItem:
    """测试用表格 item。"""

    def __init__(
        self,
        *,
        self_ref: str,
        page_no: int,
        df: pd.DataFrame,
        markdown: str,
        caption: Optional[str] = None,
    ) -> None:
        """初始化表格 item。

        Args:
            self_ref: 内部引用。
            page_no: 页码。
            df: DataFrame 数据。
            markdown: markdown 内容。
            caption: 可选标题。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.self_ref = self_ref
        self.prov = [_FakeProv(page_no=page_no)]
        self.data = _FakeTableData(num_rows=int(df.shape[0]), num_cols=int(df.shape[1]))
        self.caption = _FakeCaption(caption) if caption else None
        self._df = df
        self._markdown = markdown

    def export_to_dataframe(self, doc: Optional[Any] = None) -> pd.DataFrame:
        """导出 DataFrame。

        Args:
            doc: 预留参数。

        Returns:
            DataFrame 数据。

        Raises:
            RuntimeError: 导出失败时抛出。
        """

        del doc
        return self._df.copy()

    def export_to_markdown(self, doc: Optional[Any] = None) -> str:
        """导出 markdown。

        Args:
            doc: 预留参数。

        Returns:
            markdown 文本。

        Raises:
            RuntimeError: 导出失败时抛出。
        """

        del doc
        return self._markdown


class _FakeDocument:
    """测试用 Docling 文档对象。"""

    def __init__(self, linear_items: list[tuple[Any, int]], tables: list[_FakeTableItem]) -> None:
        """初始化测试文档。

        Args:
            linear_items: 线性 item 列表。
            tables: 表格列表。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._linear_items = linear_items
        self.tables = tables

    def iterate_items(self, with_groups: bool = False) -> list[tuple[Any, int]]:
        """按读取顺序遍历 items。

        Args:
            with_groups: 是否包含 group。

        Returns:
            item 序列。

        Raises:
            RuntimeError: 遍历失败时抛出。
        """

        del with_groups
        return list(self._linear_items)


@pytest.mark.unit
def test_fins_docling_processor_relabels_financial_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 fins Docling 处理器会补充金融语义。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "sample_docling.json"
    json_path.write_text("{}", encoding="utf-8")

    table = _FakeTableItem(
        self_ref="#/tables/0",
        page_no=1,
        df=pd.DataFrame([{"Item": "Revenue", "Value": 100.0}]),
        markdown="|Item|Value|",
        caption="Consolidated Statement of Operations",
    )
    linear_items = [
        (_FakeTextItem(self_ref="#/texts/0", text="Section A", label="section_header", page_no=1), 0),
        (_FakeTextItem(self_ref="#/texts/1", text="正文", label="text", page_no=1), 1),
        (table, 1),
    ]
    fake_doc = _FakeDocument(linear_items=linear_items, tables=[table])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = FinsDoclingProcessor(_make_source(json_path, media_type="application/json"))
    tables = processor.list_tables()
    assert len(tables) == 1
    assert tables[0].get("is_financial") is True
    assert tables[0]["table_type"] == "financial"


@pytest.mark.unit
def test_fins_docling_processor_derives_financial_caption_from_table_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 fins Docling 处理器会从表体补充财报表语义 caption。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "sample_docling.json"
    json_path.write_text("{}", encoding="utf-8")

    table = _FakeTableItem(
        self_ref="#/tables/0",
        page_no=1,
        df=pd.DataFrame(
            [
                {"项目": "一、经营活动产生的现金流量", "金额": None},
                {"项目": "销售商品、提供劳务收到的现金", "金额": 100.0},
                {"项目": "二、投资活动产生的现金流量", "金额": None},
                {"项目": "三、筹资活动产生的现金流量", "金额": None},
            ]
        ),
        markdown=(
            "| 项目 | 金额 |\n"
            "| --- | --- |\n"
            "| 一、经营活动产生的现金流量 | |\n"
            "| 二、投资活动产生的现金流量 | |\n"
            "| 三、筹资活动产生的现金流量 | |"
        ),
        caption=None,
    )
    linear_items = [
        (_FakeTextItem(self_ref="#/texts/0", text="2021年度", label="section_header", page_no=1), 0),
        (table, 1),
    ]
    fake_doc = _FakeDocument(linear_items=linear_items, tables=[table])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = FinsDoclingProcessor(_make_source(json_path, media_type="application/json"))
    tables = processor.list_tables()

    assert len(tables) == 1
    assert tables[0].get("is_financial") is True
    assert tables[0]["table_type"] == "financial"
    assert tables[0]["caption"] == "现金流量表"
