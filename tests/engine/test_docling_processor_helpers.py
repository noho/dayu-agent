"""DoclingProcessor 辅助函数与异常分支测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd
import pytest
from docling_core.types.doc.document import DoclingDocument, NodeItem, TableItem

from dayu.engine.processors import docling_processor as dp
from dayu.engine.processors.docling_processor import DoclingProcessor
from dayu.fins.storage.local_file_source import LocalFileSource


@dataclass
class _Prov:
    """测试用页码来源对象。"""

    page_no: int


class _FakeLabel:
    """测试用标签对象。"""

    def __init__(self, value: str) -> None:
        """初始化标签对象。

        Args:
            value: 标签值。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.value = value


class _FakeParent:
    """测试用 parent 对象。"""

    def __init__(self, ref: Optional[str] = None, cref: Optional[str] = None, alt_ref: Optional[str] = None) -> None:
        """初始化 parent。

        Args:
            ref: `$ref` 字段值。
            cref: `cref` 字段值。
            alt_ref: `ref` 字段值。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        if ref is not None:
            setattr(self, "$ref", ref)
        if cref is not None:
            self.cref = cref
        if alt_ref is not None:
            self.ref = alt_ref


class _TextItem:
    """测试用文本 item。"""

    def __init__(
        self,
        *,
        self_ref: Optional[str],
        text: str,
        page_no: Optional[int],
        label: str = "text",
        parent: Any = None,
    ) -> None:
        """初始化文本 item。

        Args:
            self_ref: 内部引用。
            text: 文本。
            page_no: 页码。
            label: 标签。
            parent: 可选父对象。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.self_ref = self_ref
        self.text = text
        self.label = _FakeLabel(label)
        self.parent = parent
        self.prov = [] if page_no is None else [_Prov(page_no=page_no)]


class _MinimalTableItem:
    """最小表格对象。"""

    class _TableData:
        """表格尺寸对象。"""

        def __init__(self, rows: int, cols: int) -> None:
            """初始化尺寸对象。

            Args:
                rows: 行数。
                cols: 列数。

            Returns:
                无。

            Raises:
                ValueError: 参数非法时抛出。
            """

            self.num_rows = rows
            self.num_cols = cols

    def __init__(self, self_ref: str, df: pd.DataFrame, markdown: str, caption: Optional[str] = None) -> None:
        """初始化测试表格对象。

        Args:
            self_ref: 内部引用。
            df: DataFrame。
            markdown: markdown 文本。
            caption: 可选标题。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.self_ref = self_ref
        self._df = df
        self._markdown = markdown
        self.data = self._TableData(int(df.shape[0]), int(df.shape[1]))
        self.caption = None if caption is None else type("Caption", (), {"text": caption})()
        self.prov = [_Prov(page_no=1)]

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
            markdown 文本。

        Raises:
            RuntimeError: 导出失败时抛出。
        """

        del doc
        return self._markdown


class _FakeDocument:
    """测试用文档对象。"""

    def __init__(self, linear_items: list[tuple[Any, int]], tables: list[Any]) -> None:
        """初始化文档。

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
        """返回线性 item。

        Args:
            with_groups: 是否包含 groups。

        Returns:
            item 列表。

        Raises:
            RuntimeError: 遍历失败时抛出。
        """

        del with_groups
        return list(self._linear_items)


class _SourceRaiseOSError:
    """materialize 抛出 OSError 的 source 桩。"""

    uri = "local://bad.json"
    media_type = "application/json"
    content_length = None
    etag = None

    def open(self) -> Any:
        """测试桩不提供 open。"""

        raise OSError("io error")

    def materialize(self, suffix: Optional[str] = None) -> Path:
        """抛出 OSError。

        Args:
            suffix: 预期后缀。

        Returns:
            无。

        Raises:
            OSError: 始终抛出。
        """

        del suffix
        raise OSError("io error")


class _SourceMissing:
    """materialize 返回缺失文件路径的 source 桩。"""

    def __init__(self, missing_path: Path) -> None:
        """初始化缺失 source。"""

        self._missing_path = missing_path
        self.uri = "local://not_exists.json"
        self.media_type = "application/json"
        self.content_length = None
        self.etag = None

    def open(self) -> Any:
        """测试桩不提供 open。"""

        raise OSError("missing")

    def materialize(self, suffix: Optional[str] = None) -> Path:
        """返回缺失文件路径。"""

        del suffix
        return self._missing_path


def _make_source(path: Path, *, uri: Optional[str] = None, media_type: str = "application/json") -> LocalFileSource:
    """构建本地 Source。

    Args:
        path: 文件路径。
        uri: 可选 URI。
        media_type: 媒体类型。

    Returns:
        Source 实例。

    Raises:
        OSError: 文件访问失败时抛出。
    """

    return LocalFileSource(
        path=path,
        uri=uri or f"local://{path.name}",
        media_type=media_type,
        content_length=path.stat().st_size,
        etag=None,
    )


def _build_processor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DoclingProcessor:
    """构建最小可用 DoclingProcessor。

    Args:
        tmp_path: 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        处理器实例。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    json_path = tmp_path / "simple_docling.json"
    json_path.write_text("{}", encoding="utf-8")

    table = _MinimalTableItem(
        self_ref="#/tables/0",
        df=pd.DataFrame([{"A": 1, "B": 2}]),
        markdown="|A|B|",
        caption="Balance Sheet",
    )
    linear_items = [
        (_TextItem(self_ref="#/texts/0", text="章节", page_no=1, label="section_header"), 0),
        (_TextItem(self_ref="#/texts/1", text="正文 keyword", page_no=1), 1),
        (table, 1),
    ]
    monkeypatch.setattr(
        "dayu.engine.processors.docling_processor._load_docling_document",
        lambda _: _FakeDocument(linear_items=linear_items, tables=[table]),
    )
    return DoclingProcessor(_make_source(json_path))


@pytest.mark.unit
def test_supports_and_init_error_branches(tmp_path: Path) -> None:
    """验证 supports 分支与初始化缺失文件异常。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    txt_path = tmp_path / "note.txt"
    txt_path.write_text("{}", encoding="utf-8")
    source_txt = _make_source(txt_path, uri="local://note.txt", media_type="application/json")
    assert DoclingProcessor.supports(source_txt) is False

    plain_path = tmp_path / "doc.json"
    plain_path.write_text("{}", encoding="utf-8")
    source_plain = _make_source(plain_path, media_type="text/plain")
    assert DoclingProcessor.supports(source_plain) is False

    missing = tmp_path / "missing_docling.json"
    missing.write_text("{}", encoding="utf-8")
    source_missing = _make_source(missing)
    missing.unlink()
    with pytest.raises(ValueError, match="Docling JSON 文件不存在"):
        DoclingProcessor(source_missing)


@pytest.mark.unit
def test_read_keyerror_search_shortcuts_and_list_tables_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 read/search 快捷分支与 list_tables 异常封装。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    processor = _build_processor(tmp_path, monkeypatch)

    with pytest.raises(KeyError, match="Section not found"):
        processor.read_section("s_9999")
    with pytest.raises(KeyError, match="Table not found"):
        processor.read_table("t_9999")

    assert processor.search("  ") == []
    assert processor.search("keyword", within_ref="s_9999") == []

    processor._tables = None  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="Docling table parsing failed"):
        processor.list_tables()


@pytest.mark.unit
def test_ref_parent_page_and_item_type_helpers() -> None:
    """验证 item 类型、引用与页码辅助函数。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    picture_item = cast(NodeItem, type("PictureItem", (), {})())
    unknown_item = cast(NodeItem, type("Unknown", (), {})())
    assert dp._resolve_item_type(picture_item) == "picture"
    assert dp._resolve_item_type(unknown_item) is None

    class _RefItem:
        """提供 get_ref 的测试对象。"""

        def get_ref(self) -> str:
            """返回 internal_ref。

            Args:
                无。

            Returns:
                internal_ref。

            Raises:
                RuntimeError: 获取失败时抛出。
            """

            return "#/texts/1"

    assert dp._extract_internal_ref(cast(NodeItem, _RefItem())) == "#/texts/1"

    class _BadRefItem:
        """get_ref 抛错对象。"""

        def get_ref(self) -> str:
            """抛出异常。

            Args:
                无。

            Returns:
                无。

            Raises:
                RuntimeError: 始终抛出。
            """

            raise RuntimeError("bad")

    assert dp._extract_internal_ref(cast(NodeItem, _BadRefItem())) is None
    assert dp._extract_parent_ref(cast(NodeItem, _TextItem(self_ref="#/t", text="x", page_no=1, parent="#/tables/1"))) == "#/tables/1"
    assert dp._extract_parent_ref(cast(NodeItem, _TextItem(self_ref="#/t", text="x", page_no=1, parent=_FakeParent(ref="#/tables/2")))) == "#/tables/2"
    assert dp._extract_parent_ref(cast(NodeItem, _TextItem(self_ref="#/t", text="x", page_no=1, parent=_FakeParent(cref="#/tables/3")))) == "#/tables/3"
    assert dp._extract_parent_ref(cast(NodeItem, _TextItem(self_ref="#/t", text="x", page_no=1, parent=_FakeParent(alt_ref="#/tables/4")))) == "#/tables/4"

    assert dp._extract_page_no(cast(NodeItem, _TextItem(self_ref="#/t", text="x", page_no=None))) is None
    assert dp._extract_page_no(cast(NodeItem, _TextItem(self_ref="#/t", text="x", page_no=1))) == 1


@pytest.mark.unit
def test_dataframe_header_and_markdown_fallback_helpers() -> None:
    """验证 DataFrame、表头与 markdown 回退分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    class _TypeErrorDfExporter:
        """仅支持无参导出的对象。"""

        def export_to_dataframe(self) -> pd.DataFrame:
            """返回 DataFrame。

            Args:
                无。

            Returns:
                DataFrame。

            Raises:
                RuntimeError: 导出失败时抛出。
            """

            return pd.DataFrame([{"A": 1}])

    class _BadDfExporter:
        """导出异常对象。"""

        def export_to_dataframe(self, doc: Any = None) -> Any:
            """抛出异常。

            Args:
                doc: 预留参数。

            Returns:
                无。

            Raises:
                RuntimeError: 始终抛出。
            """

            del doc
            raise RuntimeError("bad")

    fake_document = cast(DoclingDocument, object())

    assert isinstance(dp._safe_table_dataframe(cast(TableItem, _TypeErrorDfExporter()), fake_document), pd.DataFrame)
    assert dp._safe_table_dataframe(cast(TableItem, _BadDfExporter()), fake_document) is None

    df_headers = pd.DataFrame(
        [["-", "x"], ["--", "y"]],
        columns=pd.Index(["unnamed: 0", "unnamed: 1"], dtype="object"),
    )
    table = _MinimalTableItem(self_ref="#/tables/1", df=df_headers, markdown="md")
    assert dp._extract_table_headers(cast(TableItem, table), fake_document) is None

    df_named = pd.DataFrame(
        [["资产", 1], ["负债", 2]],
        columns=pd.Index(["项目", "金额"], dtype="object"),
    )
    table_named = _MinimalTableItem(self_ref="#/tables/2", df=df_named, markdown="md")
    assert dp._extract_table_headers(cast(TableItem, table_named), fake_document) == ["资产", "负债"]

    class _TypeErrorMarkdownExporter:
        """仅支持无参 markdown 导出对象。"""

        def export_to_markdown(self) -> str:
            """返回 markdown。

            Args:
                无。

            Returns:
                markdown。

            Raises:
                RuntimeError: 导出失败时抛出。
            """

            return "|A|"

    class _FallbackMarkdownExporter:
        """markdown 导出失败后走 data 回退。"""

        data = {"rows": 1}

        def export_to_markdown(self, doc: Any = None) -> str:
            """抛出异常。

            Args:
                doc: 预留参数。

            Returns:
                无。

            Raises:
                RuntimeError: 始终抛出。
            """

            del doc
            raise RuntimeError("bad")

        def export_to_dataframe(self, doc: Any = None) -> Any:
            """返回空表。

            Args:
                doc: 预留参数。

            Returns:
                空 DataFrame。

            Raises:
                RuntimeError: 导出失败时抛出。
            """

            del doc
            return pd.DataFrame()

    assert dp._render_markdown_table(cast(TableItem, _TypeErrorMarkdownExporter()), fake_document) == "|A|"
    assert dp._render_markdown_table(cast(TableItem, _FallbackMarkdownExporter()), fake_document) == "{'rows': 1}"


@pytest.mark.unit
def test_text_ref_snippet_and_misc_helpers() -> None:
    """验证字符串、引用、snippet 与表格上下文辅助函数。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert dp._normalize_optional_string(None) is None
    assert dp._normalize_optional_string("   ") is None
    assert dp._normalize_cell_value(None) is None
    assert dp._normalize_cell_value(pd.NA) is None
    assert dp._normalize_cell_value("  a  ") == "a"

    assert dp._is_low_information_header("unnamed: 0") is True
    assert dp._looks_like_default_headers([]) is True
    assert dp._looks_like_default_headers(["1", "2"]) is True
    assert dp._deduplicate_headers(["A", "a", "", "B"]) == ["A", "B"]

    with pytest.raises(ValueError, match="section index"):
        dp._format_section_ref(0)
    with pytest.raises(ValueError, match="table index"):
        dp._format_table_ref(0)

    assert dp._infer_suffix_from_uri("local://a/b/C.JSON") == ".json"
    assert dp._append_missing_placeholders("", ["t_0001"]) == "[[t_0001]]"
    assert dp._pick_snippet_page_no([]) is None
    assert dp._pick_snippet_page_no([0, 2]) is None

    linear_items = [
        dp._LinearItem(index=0, item_type="text", internal_ref="#/t/0", page_no=1, level=1, text="", label="text", object_ref=None),
        dp._LinearItem(index=1, item_type="table", internal_ref="#/tables/0", page_no=1, level=1, text="", label=None, object_ref=None),
    ]
    table = type("Table", (), {"self_ref": None})()
    assert dp._extract_table_context_before(cast(TableItem, table), linear_items) == ""


@pytest.mark.unit
def test_sniff_and_payload_shape_helpers(tmp_path: Path) -> None:
    """验证 Docling payload 形状判断与 sniff 分支。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert dp._looks_like_docling_payload([]) is False
    assert dp._looks_like_docling_payload({"texts": [], "pages": {}}) is False
    assert dp._looks_like_docling_payload({"body": {}, "texts": [], "pages": {}}) is True

    assert dp._sniff_docling_json(_SourceRaiseOSError()) is False

    missing_path = tmp_path / "not_exists.json"
    source_missing = _SourceMissing(missing_path)
    assert dp._sniff_docling_json(source_missing) is False

    invalid_path = tmp_path / "bad.json"
    invalid_path.write_text("{bad", encoding="utf-8")
    source_invalid = _make_source(invalid_path)
    assert dp._sniff_docling_json(source_invalid) is False
