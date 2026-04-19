"""DoclingProcessor 单元测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast
from unittest.mock import Mock

import pandas as pd
import pytest

from dayu.engine.processors import docling_processor
from dayu.engine.processors.docling_processor import DoclingProcessor
from dayu.fins.storage.local_file_source import LocalFileSource


@dataclass
class FakeProv:
    """测试用 provenance。"""

    page_no: int


class FakeLabel:
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


class FakeParentRef:
    """测试用 parent 引用对象。"""

    def __init__(self, ref: str) -> None:
        """初始化 parent 引用。

        Args:
            ref: 引用字符串。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        setattr(self, "$ref", ref)


class FakeTextItem:
    """测试用文本 item。"""

    def __init__(
        self,
        *,
        self_ref: str,
        text: str,
        label: str,
        page_no: int,
        parent_ref: Optional[str] = None,
    ) -> None:
        """初始化文本 item。

        Args:
            self_ref: 内部引用。
            text: 文本内容。
            label: 标签值。
            page_no: 页码。
            parent_ref: 可选父引用。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.self_ref = self_ref
        self.text = text
        self.label = FakeLabel(label)
        self.prov = [FakeProv(page_no=page_no)]
        self.parent = FakeParentRef(parent_ref) if parent_ref is not None else None


class FakeTableData:
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


class FakeCaption:
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


class FakeTableItem:
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
            markdown: markdown 回退内容。
            caption: 可选标题。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.self_ref = self_ref
        self.prov = [FakeProv(page_no=page_no)]
        self.data = FakeTableData(num_rows=int(df.shape[0]), num_cols=int(df.shape[1]))
        self.caption = FakeCaption(caption) if caption else None
        self._df = df
        self._markdown = markdown

    def export_to_dataframe(self, doc: Optional[Any] = None) -> pd.DataFrame:
        """导出 DataFrame。

        Args:
            doc: 预留参数。

        Returns:
            DataFrame。

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
            markdown 字符串。

        Raises:
            RuntimeError: 导出失败时抛出。
        """

        del doc
        return self._markdown


class FakeTableItemMarkdownFail(FakeTableItem):
    """测试用 markdown 导出失败的表格 item。"""

    def export_to_markdown(self, doc: Optional[Any] = None) -> str:
        """模拟 markdown 导出失败。

        Args:
            doc: 预留参数。

        Returns:
            永不返回。

        Raises:
            RuntimeError: 总是抛出以触发回退逻辑。
        """

        del doc
        raise RuntimeError("export_to_markdown failed")


class FakeDocument:
    """测试用 Docling 文档对象。"""

    def __init__(self, linear_items: list[tuple[Any, int]], tables: list[FakeTableItem]) -> None:
        """初始化测试文档。

        Args:
            linear_items: 线性 item 序列。
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



def _make_source(path: Path, media_type: str = "application/json") -> LocalFileSource:
    """构建本地 Source。

    Args:
        path: 文件路径。
        media_type: 媒体类型。

    Returns:
        Source 实例。

    Raises:
        OSError: 构建失败时抛出。
    """

    return LocalFileSource(
        path=path,
        uri=f"local://{path.name}",
        media_type=media_type,
        content_length=path.stat().st_size,
        etag=None,
    )


@pytest.mark.unit
def test_docling_processor_supports_rules(tmp_path: Path) -> None:
    """验证 supports 命中与兜底探测逻辑。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    docling_path = tmp_path / "report_docling.json"
    docling_path.write_text('{"body": {}, "texts": [], "tables": [], "pages": {}}', encoding="utf-8")
    source = _make_source(docling_path)
    assert DoclingProcessor.supports(source) is True

    plain_json = tmp_path / "plain.json"
    plain_json.write_text('{"body": {}, "texts": [], "tables": [], "pages": {}}', encoding="utf-8")
    assert DoclingProcessor.supports(_make_source(plain_json)) is True

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text('{"a": 1}', encoding="utf-8")
    assert DoclingProcessor.supports(_make_source(invalid_json)) is False


@pytest.mark.unit
def test_docling_processor_read_section_replaces_tables_with_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 read_section 会把表格替换成占位符。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "filing_docling.json"
    json_path.write_text("{}", encoding="utf-8")

    table_1 = FakeTableItem(
        self_ref="#/tables/0",
        page_no=2,
        df=pd.DataFrame([{"项目": "现金", "值": 100}]),
        markdown="TABLE_RAW_1",
        caption="资产负债表",
    )
    table_2 = FakeTableItem(
        self_ref="#/tables/1",
        page_no=3,
        df=pd.DataFrame([["A", "B"]], columns=pd.Index(["dup", "dup"], dtype="object")),
        markdown="TABLE_RAW_2",
    )

    linear_items = [
        (FakeTextItem(self_ref="#/texts/0", text="第一章", label="section_header", page_no=1), 0),
        (FakeTextItem(self_ref="#/texts/1", text="章节前文", label="text", page_no=1), 1),
        (table_1, 1),
        (FakeTextItem(self_ref="#/texts/2", text="章节后文", label="text", page_no=2), 1),
        # 表格内部文本，必须被过滤，不应进入章节正文。
        (
            FakeTextItem(
                self_ref="#/texts/3",
                text="表格内部文本",
                label="text",
                page_no=2,
                parent_ref="#/tables/0",
            ),
            2,
        ),
        (table_2, 1),
        (FakeTextItem(self_ref="#/texts/4", text="第二章", label="section_header", page_no=4), 0),
        (FakeTextItem(self_ref="#/texts/5", text="第二章正文", label="text", page_no=4), 1),
    ]

    fake_doc = FakeDocument(linear_items=linear_items, tables=[table_1, table_2])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = DoclingProcessor(_make_source(json_path))
    sections = processor.list_sections()
    assert len(sections) == 2
    assert sections[0]["ref"] == "s_0001"
    assert sections[0].get("page_range") == [1, 3]

    first = processor.read_section("s_0001")
    assert first["tables"] == ["t_0001", "t_0002"]
    assert "[[t_0001]]" in first["content"]
    assert "[[t_0002]]" in first["content"]
    assert "表格内部文本" not in first["content"]
    assert "TABLE_RAW_1" not in first["content"]
    assert "TABLE_RAW_2" not in first["content"]

    # 模拟章节声明了额外表格但正文未命中，验证尾部补占位符。
    processor._sections[0].table_refs.append("t_9999")
    patched = processor.read_section("s_0001")
    assert "[[t_9999]]" in patched["content"]


@pytest.mark.unit
def test_docling_processor_read_table_and_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 read_table records/markdown 与 search 行为。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "material_docling.json"
    json_path.write_text("{}", encoding="utf-8")

    records_table = FakeTableItem(
        self_ref="#/tables/0",
        page_no=5,
        df=pd.DataFrame([{"Item": "Revenue", "Value": 200.0}]),
        markdown="|Item|Value|",
        caption="Income Statement",
    )
    markdown_table = FakeTableItem(
        self_ref="#/tables/1",
        page_no=6,
        df=pd.DataFrame([["X", "Y"]], columns=pd.Index(["col", "col"], dtype="object")),
        markdown="MARKDOWN_FALLBACK",
    )
    linear_items = [
        (FakeTextItem(self_ref="#/texts/0", text="Only body", label="text", page_no=5), 0),
        (records_table, 1),
        (
            FakeTextItem(
                self_ref="#/texts/1",
                text=(
                    "keyword is here. another keyword appears in details. "
                    "final keyword appears in the closing note."
                ),
                label="text",
                page_no=5,
            ),
            1,
        ),
        (markdown_table, 1),
    ]
    fake_doc = FakeDocument(linear_items=linear_items, tables=[records_table, markdown_table])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = DoclingProcessor(_make_source(json_path))
    tables = processor.list_tables()
    assert len(tables) == 2
    assert tables[0]["table_type"] in {"data", "layout"}
    assert tables[1]["table_type"] in {"data", "layout"}

    table_1 = processor.read_table("t_0001")
    assert table_1["data_format"] == "records"
    assert table_1["columns"] == ["Item", "Value"]
    assert table_1.get("page_no") == 5
    table_2 = processor.read_table("t_0002")
    assert table_2["data_format"] == "markdown"
    assert "MARKDOWN_FALLBACK" in str(table_2["data"])

    hits = processor.search("keyword")
    assert hits
    assert hits[0].get("section_ref") == "s_0001"
    assert len(hits) <= 2
    assert all("keyword" in str(hit.get("snippet") or "").lower() for hit in hits)
    assert all(len(str(hit.get("snippet") or "")) <= 360 for hit in hits)

    scoped_hits = processor.search("keyword", within_ref="s_0001")
    assert scoped_hits
    assert all(hit.get("section_ref") == "s_0001" for hit in scoped_hits)


@pytest.mark.unit
def test_docling_processor_section_render_cache_reused_in_search_and_full_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证章节缓存在 search 与 get_full_text 中复用。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "cache_docling.json"
    json_path.write_text("{}", encoding="utf-8")

    table = FakeTableItem(
        self_ref="#/tables/0",
        page_no=1,
        df=pd.DataFrame([{"Item": "Revenue", "Value": 10}]),
        markdown="|Item|Value|",
    )
    linear_items = [
        (FakeTextItem(self_ref="#/texts/0", text="Section One", label="section_header", page_no=1), 0),
        (FakeTextItem(self_ref="#/texts/1", text="Revenue grows strongly", label="text", page_no=1), 1),
        (table, 1),
        (FakeTextItem(self_ref="#/texts/2", text="Section Two", label="section_header", page_no=2), 0),
        (FakeTextItem(self_ref="#/texts/3", text="Margin expands", label="text", page_no=2), 1),
    ]
    fake_doc = FakeDocument(linear_items=linear_items, tables=[table])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = DoclingProcessor(_make_source(json_path))
    original_render = docling_processor._render_section_content
    tracked_render = Mock(side_effect=original_render)
    monkeypatch.setattr(docling_processor, "_render_section_content", tracked_render)

    first = processor.read_section("s_0001")
    hits = processor.search("revenue")
    full_text = processor.get_full_text()

    assert "[[t_0001]]" in first["content"]
    assert hits
    assert "Margin expands" in full_text
    # s_0001 在 read_section 已渲染一次，search/get_full_text 复用缓存；
    # get_full_text 仅会触发 s_0002 的首次渲染。
    assert tracked_render.call_count == 2


@pytest.mark.unit
def test_docling_processor_read_table_markdown_export_failure_falls_back_to_dataframe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 markdown 导出失败时回退 DataFrame to_markdown。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "markdown_fallback_docling.json"
    json_path.write_text("{}", encoding="utf-8")

    table = FakeTableItemMarkdownFail(
        self_ref="#/tables/0",
        page_no=1,
        df=pd.DataFrame([["A", "B"]], columns=pd.Index(["dup", "dup"], dtype="object")),
        markdown="RAW_MD",
    )
    linear_items = [
        (FakeTextItem(self_ref="#/texts/0", text="正文", label="text", page_no=1), 0),
        (table, 1),
    ]
    fake_doc = FakeDocument(linear_items=linear_items, tables=[table])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = DoclingProcessor(_make_source(json_path))
    table_payload = processor.read_table("t_0001")
    assert table_payload["data_format"] == "markdown"
    assert "dup" in str(table_payload["data"])
    assert "|" in str(table_payload["data"])
    assert "RAW_MD" not in str(table_payload["data"])


@pytest.mark.unit
def test_docling_processor_table_headers_fallback_to_column_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证表头提取在行头低信息时回退列头。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "headers_fallback_docling.json"
    json_path.write_text("{}", encoding="utf-8")

    table = FakeTableItem(
        self_ref="#/tables/0",
        page_no=1,
        df=pd.DataFrame(
            [
                ["-", 10],
                ["n/a", 20],
            ],
            columns=pd.Index(["Name", "Value"], dtype="object"),
        ),
        markdown="MD",
    )
    linear_items = [
        (FakeTextItem(self_ref="#/texts/0", text="正文", label="text", page_no=1), 0),
        (table, 1),
    ]
    fake_doc = FakeDocument(linear_items=linear_items, tables=[table])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = DoclingProcessor(_make_source(json_path))
    tables = processor.list_tables()
    assert tables[0]["headers"] == ["Name", "Value"]


@pytest.mark.unit
def test_docling_processor_no_section_header_creates_dummy_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证无 section_header 时返回全文 dummy section。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "dummy_docling.json"
    json_path.write_text("{}", encoding="utf-8")
    table = FakeTableItem(
        self_ref="#/tables/0",
        page_no=1,
        df=pd.DataFrame([{"A": 1}]),
        markdown="md",
    )
    linear_items = [
        (FakeTextItem(self_ref="#/texts/0", text="全文内容", label="text", page_no=1), 0),
        (table, 1),
    ]
    fake_doc = FakeDocument(linear_items=linear_items, tables=[table])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = DoclingProcessor(_make_source(json_path))
    sections = processor.list_sections()
    assert len(sections) == 1
    detail = processor.read_section("s_0001")
    assert detail["contains_full_text"] is True
    assert "[[t_0001]]" in detail["content"]


@pytest.mark.unit
def test_docling_processor_get_page_content_returns_page_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 get_page_content 返回页面级章节/表格/文本上下文。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "page_docling.json"
    json_path.write_text("{}", encoding="utf-8")

    table = FakeTableItem(
        self_ref="#/tables/0",
        page_no=2,
        df=pd.DataFrame([{"项目": "现金", "值": 100}]),
        markdown="TABLE_RAW",
        caption="资产负债表",
    )
    linear_items = [
        (FakeTextItem(self_ref="#/texts/0", text="第一章", label="section_header", page_no=1), 0),
        (FakeTextItem(self_ref="#/texts/1", text="第一页正文", label="text", page_no=1), 1),
        (table, 1),
        (FakeTextItem(self_ref="#/texts/2", text="第二页正文", label="text", page_no=2), 1),
        (FakeTextItem(self_ref="#/texts/3", text="2", label="text", page_no=2), 1),
    ]
    fake_doc = FakeDocument(linear_items=linear_items, tables=[table])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = DoclingProcessor(_make_source(json_path))
    page_2 = processor.get_page_content(2)
    assert page_2["supported"] is True
    assert page_2["has_content"] is True
    assert page_2["total_items"] == 2
    assert page_2["sections"][0]["ref"] == "s_0001"
    assert page_2["sections"][0].get("page_range") == [1, 2]
    assert "第二页正文" in page_2["sections"][0]["preview"]
    assert page_2["tables"][0]["table_ref"] == "t_0001"
    assert page_2["tables"][0]["table_type"] in {"data", "layout"}
    assert "第二页正文" in page_2["text_preview"]
    assert " 2 " not in f" {page_2['text_preview']} "

    page_9 = processor.get_page_content(9)
    assert page_9["supported"] is True
    assert page_9["has_content"] is False
    assert page_9["sections"] == []
    assert page_9["tables"] == []


@pytest.mark.unit
def test_docling_processor_get_page_content_rejects_invalid_page_no(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 get_page_content 会拒绝非法页码。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "page_invalid_docling.json"
    json_path.write_text("{}", encoding="utf-8")
    fake_doc = FakeDocument(linear_items=[], tables=[])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = DoclingProcessor(_make_source(json_path))
    with pytest.raises(ValueError):
        processor.get_page_content(0)


@pytest.mark.unit
def test_docling_processor_get_page_content_skips_invalid_page_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证页码范围非法时不产出章节片段。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "page_range_docling.json"
    json_path.write_text("{}", encoding="utf-8")

    linear_items = [
        (FakeTextItem(self_ref="#/texts/0", text="章节", label="section_header", page_no=1), 0),
        (FakeTextItem(self_ref="#/texts/1", text="正文", label="text", page_no=1), 1),
    ]
    fake_doc = FakeDocument(linear_items=linear_items, tables=[])
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: fake_doc,
    )

    processor = DoclingProcessor(_make_source(json_path))
    cast(Any, processor._sections[0]).page_range = ["1", 2]
    page_payload = processor.get_page_content(1)
    assert page_payload["sections"] == []
    assert "正文" in page_payload["text_preview"]
