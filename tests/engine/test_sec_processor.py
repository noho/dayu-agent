"""SecProcessor 单元测试。"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast
from unittest.mock import Mock

import pandas as pd
import pytest
from edgar.documents.exceptions import DocumentTooLargeError
from edgar.xbrl import XBRL

from dayu.engine.processors.base import SearchHit, TableSummary
from dayu.fins.processors.financial_base import FinancialStatementResult, XbrlFactsResult
from dayu.fins.storage.local_file_source import LocalFileSource
from dayu.fins.processors import sec_processor
from dayu.fins.processors import sec_section_build
from dayu.fins.processors import sec_table_extraction
from dayu.fins.processors import sec_xbrl_query
from dayu.fins.processors import sec_dom_helpers
from dayu.engine.processors import text_utils
from dayu.fins import xbrl_file_discovery
from dayu.fins.processors.sec_processor import SecProcessor


def _table_is_financial(table: TableSummary) -> object:
    """读取表格摘要中的可选 is_financial。"""

    return table.get("is_financial")


def _hit_section_ref(hit: SearchHit) -> object:
    """读取搜索命中的可选 section_ref。"""

    return hit.get("section_ref")


def _hit_snippet(hit: SearchHit) -> object:
    """读取搜索命中的可选 snippet。"""

    return hit.get("snippet")


def _statement_locator(result: FinancialStatementResult) -> dict[str, Any] | None:
    """读取财务报表结果中的可选 statement_locator。"""

    locator = result.get("statement_locator")
    if isinstance(locator, dict):
        return locator
    return None


def _result_reason(result: FinancialStatementResult | XbrlFactsResult) -> object:
    """读取结果中的可选 reason。"""

    return result.get("reason")


def _as_xbrl(provider: object) -> XBRL:
    """在测试边界把 fake XBRL provider 收窄为生产参数类型。"""

    return cast(XBRL, provider)


@dataclass
class FakeCell:
    """测试用表格单元格。"""

    content: str


class FakeTable:
    """测试用表格对象。"""

    def __init__(
        self,
        text_value: str,
        df: pd.DataFrame,
        *,
        caption: Optional[str] = None,
        is_financial: bool = False,
        semantic_type: str = "TableType.GENERAL",
    ) -> None:
        """初始化表格对象。

        Args:
            text_value: 表格文本。
            df: DataFrame 数据。
            caption: 可选标题。
            is_financial: 是否财务表。
            semantic_type: 语义类型。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._text_value = text_value
        self._df = df
        self.caption = caption
        self.is_financial_table = is_financial
        self.semantic_type = semantic_type
        self.row_count = int(df.shape[0])
        self.col_count = int(df.shape[1])
        self.headers = [
            [FakeCell(content=str(column)) for column in df.columns]
        ] if not df.empty else []

    def text(self) -> str:
        """返回表格文本。

        Args:
            无。

        Returns:
            表格文本。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        return self._text_value

    def to_dataframe(self) -> pd.DataFrame:
        """返回 DataFrame。

        Args:
            无。

        Returns:
            DataFrame。

        Raises:
            RuntimeError: 转换失败时抛出。
        """

        return self._df.copy()

    def to_dict(self) -> dict[str, Any]:
        """返回字典结构。

        Args:
            无。

        Returns:
            字典结构。

        Raises:
            RuntimeError: 转换失败时抛出。
        """

        return {
            "headers": self.headers,
            "data": self._df.to_dict(orient="records"),
        }


class CountingFakeTable(FakeTable):
    """统计 `to_dataframe()` 调用次数的测试表格。"""

    def __init__(
        self,
        text_value: str,
        df: pd.DataFrame,
        *,
        caption: Optional[str] = None,
        is_financial: bool = False,
        semantic_type: str = "TableType.GENERAL",
    ) -> None:
        """初始化计数表格。

        Args:
            text_value: 表格文本。
            df: DataFrame 数据。
            caption: 可选标题。
            is_financial: 是否财务表。
            semantic_type: 语义类型。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        super().__init__(
            text_value=text_value,
            df=df,
            caption=caption,
            is_financial=is_financial,
            semantic_type=semantic_type,
        )
        self.to_dataframe_calls = 0

    def to_dataframe(self) -> pd.DataFrame:
        """返回 DataFrame 并累计调用次数。

        Args:
            无。

        Returns:
            DataFrame。

        Raises:
            RuntimeError: 转换失败时抛出。
        """

        self.to_dataframe_calls += 1
        return super().to_dataframe()


class FakeSection:
    """测试用章节对象。"""

    def __init__(self, text_value: str, tables: list[FakeTable], **attrs: Any) -> None:
        """初始化章节对象。

        Args:
            text_value: 章节文本。
            tables: 章节内表格。
            **attrs: 额外属性（title/name/part/item）。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._text_value = text_value
        self._tables = tables
        self.title = attrs.get("title")
        self.name = attrs.get("name")
        self.part = attrs.get("part")
        self.item = attrs.get("item")

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

    def tables(self) -> list[FakeTable]:
        """返回章节内表格。

        Args:
            无。

        Returns:
            表格列表。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        return list(self._tables)


class FakeDocument:
    """测试用文档对象。"""

    def __init__(
        self,
        sections: dict[str, FakeSection],
        tables: list[FakeTable],
        text_value: str,
        *,
        section_anchor_ids: Optional[dict[str, str]] = None,
    ) -> None:
        """初始化文档对象。

        Args:
            sections: 章节映射。
            tables: 表格列表。
            text_value: 文档全文文本。
            section_anchor_ids: 可选 section -> anchor_id 映射（用于模拟 edgartools 顺序锚点）。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.sections = sections
        self.tables = tables
        self._text_value = text_value
        self._section_anchor_ids = section_anchor_ids or {}

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

    def get_sec_section_info(self, section_name: str) -> Optional[dict[str, Any]]:
        """返回 section 元信息。

        Args:
            section_name: section 键名。

        Returns:
            当 section 存在 anchor_id 时返回 `{\"anchor_id\": ...}`，否则返回 `None`。

        Raises:
            RuntimeError: 无。
        """

        anchor_id = self._section_anchor_ids.get(section_name)
        if not anchor_id:
            return None
        return {"anchor_id": anchor_id}


class FakeStatement:
    """测试用财务报表对象。"""

    def __init__(self, df: pd.DataFrame) -> None:
        """初始化报表对象。

        Args:
            df: 报表 DataFrame。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._df = df

    def to_dataframe(self) -> pd.DataFrame:
        """返回报表 DataFrame。

        Args:
            无。

        Returns:
            DataFrame。

        Raises:
            RuntimeError: 转换失败时抛出。
        """

        return self._df.copy()


class FakeStatements:
    """测试用 statements 容器。"""

    def __init__(self, income_df: pd.DataFrame) -> None:
        """初始化 statements 容器。

        Args:
            income_df: 损益表 DataFrame。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._income_df = income_df

    def income_statement(self) -> FakeStatement:
        """返回损益表对象。

        Args:
            无。

        Returns:
            FakeStatement。

        Raises:
            RuntimeError: 构建失败时抛出。
        """

        return FakeStatement(self._income_df)


class FakeFactQuery:
    """测试用 XBRL 查询对象。"""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """初始化查询对象。

        Args:
            rows: 原始 rows。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._rows = rows
        self._concept: Optional[str] = None
        self._statement_type: Optional[str] = None
        self._fiscal_year: Optional[int] = None
        self._fiscal_period: Optional[str] = None
        self._min_value: Optional[float] = None
        self._max_value: Optional[float] = None

    def by_concept(self, concept: str) -> "FakeFactQuery":
        """按概念筛选。

        Args:
            concept: 概念名。

        Returns:
            自身。

        Raises:
            RuntimeError: 筛选失败时抛出。
        """

        self._concept = concept
        return self

    def by_statement_type(self, statement_type: str) -> "FakeFactQuery":
        """按报表类型筛选。

        Args:
            statement_type: 报表类型。

        Returns:
            自身。

        Raises:
            RuntimeError: 筛选失败时抛出。
        """

        self._statement_type = statement_type
        return self

    def by_fiscal_year(self, fiscal_year: int) -> "FakeFactQuery":
        """按财年筛选。

        Args:
            fiscal_year: 财年。

        Returns:
            自身。

        Raises:
            RuntimeError: 筛选失败时抛出。
        """

        self._fiscal_year = fiscal_year
        return self

    def by_fiscal_period(self, fiscal_period: str) -> "FakeFactQuery":
        """按财季筛选。

        Args:
            fiscal_period: 财季。

        Returns:
            自身。

        Raises:
            RuntimeError: 筛选失败时抛出。
        """

        self._fiscal_period = fiscal_period
        return self

    def by_value(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "FakeFactQuery":
        """按值范围筛选。

        Args:
            min_value: 最小值。
            max_value: 最大值。

        Returns:
            自身。

        Raises:
            RuntimeError: 筛选失败时抛出。
        """

        self._min_value = min_value
        self._max_value = max_value
        return self

    def execute(self) -> list[dict[str, Any]]:
        """执行筛选。

        Args:
            无。

        Returns:
            筛选后 rows。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        result: list[dict[str, Any]] = []
        for row in self._rows:
            if self._concept and self._concept not in str(row.get("concept", "")):
                continue
            if self._statement_type and row.get("statement_type") != self._statement_type:
                continue
            if self._fiscal_year is not None and row.get("fiscal_year") != self._fiscal_year:
                continue
            if self._fiscal_period and row.get("fiscal_period") != self._fiscal_period:
                continue
            numeric_value = row.get("numeric_value")
            if self._min_value is not None and numeric_value is not None and numeric_value < self._min_value:
                continue
            if self._max_value is not None and numeric_value is not None and numeric_value > self._max_value:
                continue
            result.append(row)
        return result


class FakeXbrl:
    """测试用 XBRL 对象。"""

    def __init__(self, statement_df: pd.DataFrame, rows: list[dict[str, Any]]) -> None:
        """初始化 XBRL 对象。

        Args:
            statement_df: 报表 DataFrame。
            rows: facts rows。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.statements = FakeStatements(statement_df)
        self._rows = rows

    def query(self) -> FakeFactQuery:
        """返回查询对象。

        Args:
            无。

        Returns:
            查询对象。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        return FakeFactQuery(self._rows)


class _WarningTable:
    """测试用表格对象：调用 to_dataframe 时触发 PerformanceWarning。"""

    def to_dataframe(self) -> pd.DataFrame:
        """返回 DataFrame 并触发性能告警。

        Args:
            无。

        Returns:
            DataFrame。

        Raises:
            RuntimeError: 转换失败时抛出。
        """

        warnings.warn("table performance warning", pd.errors.PerformanceWarning)
        return pd.DataFrame([{"A": 1}])


class _WarningStatement:
    """测试用报表对象：调用 to_dataframe 时触发 PerformanceWarning。"""

    def to_dataframe(self) -> pd.DataFrame:
        """返回 DataFrame 并触发性能告警。

        Args:
            无。

        Returns:
            DataFrame。

        Raises:
            RuntimeError: 转换失败时抛出。
        """

        warnings.warn("statement performance warning", pd.errors.PerformanceWarning)
        return pd.DataFrame([{"A": 1}])


def _make_source(tmp_path: Path, name: str) -> LocalFileSource:
    """构建本地 Source。

    Args:
        tmp_path: 临时目录。
        name: 文件名。

    Returns:
        本地 Source。

    Raises:
        OSError: 写入失败时抛出。
    """

    path = tmp_path / name
    path.write_text("<html><body><p>placeholder</p></body></html>", encoding="utf-8")
    return LocalFileSource(
        path=path,
        uri=f"local://{name}",
        media_type="text/html" if name.endswith(".html") else "application/xml",
    )


@pytest.mark.unit
def test_sec_processor_supports_matrix(tmp_path: Path) -> None:
    """验证 supports 路由矩阵。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html_source = _make_source(tmp_path, "sample.html")
    xml_source = _make_source(tmp_path, "sample.xml")

    assert SecProcessor.supports(html_source, form_type="10-K") is True
    assert SecProcessor.supports(html_source, form_type="DEF 14A") is True
    assert SecProcessor.supports(html_source, form_type="6-K") is False
    assert SecProcessor.supports(xml_source, form_type="SC 13D/A") is True


@pytest.mark.unit
def test_sec_processor_sections_tables_and_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 sections/tables/read/search 基础能力。

    Args:
        monkeypatch: pytest monkeypatch。
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    table_1 = FakeTable(
        text_value="Cash 100 Revenue 200",
        df=pd.DataFrame([{"Item": "Cash", "Value": 100.0}]),
        caption="Income Data",
        is_financial=True,
        semantic_type="TableType.FINANCIAL",
    )
    table_2 = FakeTable(
        text_value="Unmatched table text",
        df=pd.DataFrame(),
        caption=None,
    )

    section_1 = FakeSection(
        text_value="Part text before Cash 100 Revenue 200 and after.",
        tables=[FakeTable(text_value="Cash 100 Revenue 200", df=pd.DataFrame([{"Item": "Cash"}]))],
        title="part_i_item_1",
        name="part_i_item_1",
        part="I",
        item="1",
    )
    section_2 = FakeSection(
        text_value=(
            "Second section with keyword mentioned in transaction summary. "
            "The keyword appears again in the covenant paragraph. "
            "Final keyword mention appears in closing conditions."
        ),
        tables=[FakeTable(text_value="Unmatched table text", df=pd.DataFrame([{"A": 1}]))],
        title="part_ii_item_7",
        name="part_ii_item_7",
        part="II",
        item="7",
    )
    fake_doc = FakeDocument(
        sections={
            "part_i_item_1": section_1,
            "part_ii_item_7": section_2,
        },
        tables=[table_1, table_2],
        text_value="Global document text.",
    )

    monkeypatch.setattr(sec_processor, "_parse_document", lambda html_content, form_type: fake_doc)
    source = _make_source(tmp_path, "filing.html")
    processor = SecProcessor(source, form_type="10-K")

    sections = processor.list_sections()
    assert sections[0]["ref"] == "s_0001"
    assert sections[1]["ref"] == "s_0002"
    assert sections[0]["title"] == "Part I Item 1"

    tables = processor.list_tables()
    assert tables[0]["table_ref"] == "t_0001"
    assert _table_is_financial(tables[0]) is True
    assert tables[0]["table_type"] == "financial"
    assert "Cash" in (tables[0]["headers"] or [])
    assert "Part text before" in (tables[0]["context_before"] or "")
    # layout 表（t_0002）默认被过滤，不再出现在 list_tables 结果中
    assert len(tables) == 1

    section_1_content = processor.read_section("s_0001")
    assert "[[t_0001]]" in section_1_content["content"]
    assert section_1_content["tables"] == ["t_0001"]

    section_2_content = processor.read_section("s_0002")
    assert "[[t_0002]]" in section_2_content["content"]
    assert section_2_content["tables"] == []

    table_1_content = processor.read_table("t_0001")
    assert table_1_content["data_format"] == "records"
    assert table_1_content["columns"] == ["Item", "Value"]

    table_2_content = processor.read_table("t_0002")
    assert table_2_content["data_format"] == "markdown"
    assert isinstance(table_2_content["data"], str)

    hits = processor.search("keyword", within_ref="s_0002")
    assert hits
    assert _hit_section_ref(hits[0]) == "s_0002"
    assert len(hits) <= 2
    assert all("keyword" in str(_hit_snippet(hit) or "").lower() for hit in hits)
    assert all(len(str(_hit_snippet(hit) or "")) <= 360 for hit in hits)


@pytest.mark.unit
def test_build_sections_orders_by_document_appearance() -> None:
    """验证 `_build_sections` 会按正文出现顺序修复乱序。"""

    section_1 = FakeSection(
        text_value="Item 1. Financial Statements details for quarter.",
        tables=[],
        title="part_i_item_1",
        name="part_i_item_1",
        part="I",
        item="1",
    )
    section_2 = FakeSection(
        text_value="Item 2. Management discussion and analysis details.",
        tables=[],
        title="part_i_item_2",
        name="part_i_item_2",
        part="I",
        item="2",
    )
    section_3 = FakeSection(
        text_value="Item 3. Quantitative and qualitative disclosures.",
        tables=[],
        title="part_i_item_3",
        name="part_i_item_3",
        part="I",
        item="3",
    )
    # 故意将 dict 插入顺序写错，模拟上游 sections 容器乱序。
    fake_doc = FakeDocument(
        sections={
            "part_i_item_1": section_1,
            "part_i_item_3": section_3,
            "part_i_item_2": section_2,
        },
        tables=[],
        text_value=(
            "Item 1. Financial Statements details for quarter. "
            "Item 2. Management discussion and analysis details. "
            "Item 3. Quantitative and qualitative disclosures."
        ),
    )

    sections = sec_processor._build_sections(fake_doc)
    assert [section.title for section in sections] == ["Part I Item 1", "Part I Item 2", "Part I Item 3"]


@pytest.mark.unit
def test_match_section_ref_prefers_longer_table_text_when_fingerprint_conflicts() -> None:
    """验证表格指纹冲突时会尝试用更长表格正文消歧。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    shared_prefix = "Revenue by geography Q1 2024 Q2 2024 " * 8
    table_text = f"{shared_prefix} Latin America 18 Europe 22"
    fingerprint = sec_section_build._table_fingerprint(table_text)
    section_by_ref = {
        "s_0001": sec_processor._SectionBlock(
            ref="s_0001",
            title="Part I Item 3",
            level=1,
            parent_ref=None,
            preview="",
            text=f"First section {shared_prefix} Asia 10 North America 12",
            table_refs=[],
            table_fingerprints={fingerprint},
            contains_full_text=False,
        ),
        "s_0002": sec_processor._SectionBlock(
            ref="s_0002",
            title="Part II Item 5",
            level=1,
            parent_ref=None,
            preview="",
            text=f"Second section {table_text} additional narrative",
            table_refs=[],
            table_fingerprints={fingerprint},
            contains_full_text=False,
        ),
    }

    matched = sec_table_extraction._match_section_ref(
        fingerprint=fingerprint,
        fingerprint_to_sections={fingerprint: ["s_0001", "s_0002"]},
        default_section_ref=None,
        section_by_ref=section_by_ref,
        table_text=table_text,
    )

    assert matched == "s_0002"


@pytest.mark.unit
def test_match_section_ref_uses_dom_context_when_fingerprint_conflicts() -> None:
    """验证表格指纹冲突时会用 DOM 前文做唯一消歧。

    Args:
        无。

    Returns:
        无。

        Raises:
            AssertionError: 断言失败时抛出。
    """

    table_text = "Liquidity and capital resources cash equivalents debt maturities"
    fingerprint = sec_section_build._table_fingerprint(table_text)
    section_by_ref = {
        "s_0001": sec_processor._SectionBlock(
            ref="s_0001",
            title="Part I Item 3",
            level=1,
            parent_ref=None,
            preview="",
            text=(
                "Item 3 content discusses market risk overview before table. "
                "Additional disclosures follow. "
                f"{table_text}"
            ),
            table_refs=[],
            table_fingerprints={fingerprint},
            contains_full_text=False,
        ),
        "s_0002": sec_processor._SectionBlock(
            ref="s_0002",
            title="Part II Item 5",
            level=1,
            parent_ref=None,
            preview="",
            text=(
                "Management discussion and analysis includes covenant leverage bridge before table. "
                "Near-term liquidity planning is discussed immediately before the table. "
                f"{table_text}"
            ),
            table_refs=[],
            table_fingerprints={fingerprint},
            contains_full_text=False,
        ),
    }

    matched = sec_table_extraction._match_section_ref(
        fingerprint=fingerprint,
        fingerprint_to_sections={fingerprint: ["s_0001", "s_0002"]},
        default_section_ref=None,
        section_by_ref=section_by_ref,
        table_text=fingerprint,
        dom_context_before="Near-term liquidity planning is discussed immediately before the table.",
    )

    assert matched == "s_0002"


@pytest.mark.unit
def test_match_section_ref_returns_default_when_collision_remains_ambiguous() -> None:
    """验证无法消歧的指纹冲突不会再偷选第一个章节。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    table_text = "Identical repeated table text for both sections"
    fingerprint = sec_section_build._table_fingerprint(table_text)
    shared_section = sec_processor._SectionBlock(
        ref="s_0001",
        title="Part I Item 3",
        level=1,
        parent_ref=None,
        preview="",
        text=f"Shared section text {table_text}",
        table_refs=[],
        table_fingerprints={fingerprint},
        contains_full_text=False,
    )
    section_by_ref = {
        "s_0001": shared_section,
        "s_0002": sec_processor._SectionBlock(
            ref="s_0002",
            title="Part II Item 5",
            level=1,
            parent_ref=None,
            preview="",
            text=f"Shared section text {table_text}",
            table_refs=[],
            table_fingerprints={fingerprint},
            contains_full_text=False,
        ),
    }

    matched = sec_table_extraction._match_section_ref(
        fingerprint=fingerprint,
        fingerprint_to_sections={fingerprint: ["s_0001", "s_0002"]},
        default_section_ref=None,
        section_by_ref=section_by_ref,
        table_text=table_text,
        dom_context_before="Shared section text",
    )

    assert matched is None


@pytest.mark.unit
def test_sec_processor_build_tables_defers_dataframe_until_table_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证建表阶段不会为元信息充足的表格提前构造 DataFrame。

    Args:
        monkeypatch: pytest monkeypatch。
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    table = CountingFakeTable(
        text_value="Cash and cash equivalents 100",
        df=pd.DataFrame([{"Item": "Cash and cash equivalents", "Value": 100}]),
        caption="Liquidity table",
        is_financial=True,
        semantic_type="TableType.FINANCIAL",
    )
    section = FakeSection(
        text_value="Item 5 discussion before table. Cash and cash equivalents 100",
        tables=[table],
        title="part_ii_item_5",
        name="part_ii_item_5",
        part="II",
        item="5",
    )
    fake_doc = FakeDocument(
        sections={"part_ii_item_5": section},
        tables=[table],
        text_value="Item 5 discussion before table. Cash and cash equivalents 100",
    )

    monkeypatch.setattr(sec_processor, "_parse_document", lambda html_content, form_type: fake_doc)
    source = _make_source(tmp_path, "lazy_table.html")
    processor = SecProcessor(source, form_type="10-K")

    assert table.to_dataframe_calls == 0

    table_content = processor.read_table("t_0001")

    assert table_content["data_format"] == "records"
    assert table.to_dataframe_calls == 1


@pytest.mark.unit
def test_sec_processor_read_section_reuses_render_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 SecProcessor 对同一章节重复读取时复用渲染缓存。"""

    table = FakeTable(
        text_value="Cash 100 Revenue 200",
        df=pd.DataFrame([{"Item": "Cash", "Value": 100}]),
        caption="Financial Table",
        is_financial=True,
    )
    section = FakeSection(
        text_value="Item 8 Financial Statements Cash 100 Revenue 200",
        tables=[table],
        title="part_ii_item_8",
        name="part_ii_item_8",
        part="II",
        item="8",
    )
    fake_doc = FakeDocument(
        sections={"part_ii_item_8": section},
        tables=[table],
        text_value="Item 8 Financial Statements Cash 100 Revenue 200",
    )
    monkeypatch.setattr(sec_processor, "_parse_document", lambda html_content, form_type: fake_doc)
    source = _make_source(tmp_path, "cache_filing.html")
    processor = SecProcessor(source, form_type="10-K")

    original_replace = sec_processor._replace_table_with_placeholder
    tracked_replace = Mock(side_effect=original_replace)
    monkeypatch.setattr(sec_processor, "_replace_table_with_placeholder", tracked_replace)

    first = processor.read_section("s_0001")
    second = processor.read_section("s_0001")

    assert "[[t_0001]]" in first["content"]
    assert first["content"] == second["content"]
    assert tracked_replace.call_count == 1


@pytest.mark.unit
def test_build_sections_falls_back_to_original_order_when_marker_missing() -> None:
    """验证正文定位失败时 `_build_sections` 回退原始顺序。"""

    section_a = FakeSection(text_value="Alpha data block.", tables=[], title="A", name="A")
    section_b = FakeSection(text_value="Beta data block.", tables=[], title="B", name="B")
    fake_doc = FakeDocument(
        sections={
            "section_b": section_b,
            "section_a": section_a,
        },
        tables=[],
        text_value="Document body does not contain section markers.",
    )

    sections = sec_processor._build_sections(fake_doc)
    assert [section.title for section in sections] == ["B", "A"]


@pytest.mark.unit
def test_build_sections_prefers_anchor_sequence_when_available() -> None:
    """验证 `_build_sections` 优先使用 section anchor 序号修正乱序。"""

    section_1 = FakeSection(text_value="Alpha section.", tables=[], title="A", name="A", part="I", item="1")
    section_2 = FakeSection(text_value="Beta section.", tables=[], title="B", name="B", part="I", item="2")
    section_3 = FakeSection(text_value="Gamma section.", tables=[], title="C", name="C", part="I", item="3")
    fake_doc = FakeDocument(
        sections={
            "part_i_item_3": section_3,
            "part_i_item_1": section_1,
            "part_i_item_2": section_2,
        },
        tables=[],
        text_value="header text without explicit markers",
        section_anchor_ids={
            "part_i_item_1": "tx999_3",
            "part_i_item_2": "tx999_4",
            "part_i_item_3": "tx999_5",
        },
    )

    sections = sec_processor._build_sections(fake_doc)
    assert [section.title for section in sections] == ["Part I Item 1", "Part I Item 2", "Part I Item 3"]


@pytest.mark.unit
def test_build_sections_fast_mode_skips_full_text_occurrence_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证快速构建模式不会触发逐章节全文定位扫描。"""

    section_1 = FakeSection(text_value="Alpha section.", tables=[], title="A", name="A", part="I", item="1")
    section_2 = FakeSection(text_value="Beta section.", tables=[], title="B", name="B", part="I", item="2")
    fake_doc = FakeDocument(
        sections={
            "part_i_item_2": section_2,
            "part_i_item_1": section_1,
        },
        tables=[],
        text_value="large document body without marker scan requirement",
        section_anchor_ids={
            "part_i_item_1": "tx999_3",
            "part_i_item_2": "tx999_4",
        },
    )

    def _boom_collect_candidates(**_: Any) -> tuple[list[int], list[int]]:
        """阻断慢路径函数，若被调用则说明 fast mode 失效。"""

        raise AssertionError("fast mode should not call _collect_section_appearance_candidates")

    monkeypatch.setattr(sec_section_build, "_collect_section_appearance_candidates", _boom_collect_candidates)

    sections = sec_processor._build_sections(fake_doc, fast_mode=True)
    assert [section.title for section in sections] == ["Part I Item 1", "Part I Item 2"]


@pytest.mark.unit
def test_sec_processor_opt_in_fast_section_build_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证子类开启开关后会以 fast mode 构建章节。"""

    class _FastSecProcessor(SecProcessor):
        """测试专用处理器：仅用于验证 fast mode 开关传递。"""

        _ENABLE_FAST_SECTION_BUILD = True

    table = FakeTable(
        text_value="Cash 100 Revenue 200",
        df=pd.DataFrame([{"Item": "Cash", "Value": 100}]),
        caption="Financial Table",
        is_financial=True,
    )
    section = FakeSection(
        text_value="Item 8 Financial Statements Cash 100 Revenue 200",
        tables=[table],
        title="part_ii_item_8",
        name="part_ii_item_8",
        part="II",
        item="8",
    )
    fake_doc = FakeDocument(
        sections={"part_ii_item_8": section},
        tables=[table],
        text_value="Item 8 Financial Statements Cash 100 Revenue 200",
    )
    source = _make_source(tmp_path, "fast_mode_filing.html")

    call_flags: list[tuple[bool, bool]] = []
    original_build_sections = sec_processor._build_sections

    def _tracked_build_sections(
        document: Any,
        *,
        fast_mode: bool = False,
        single_full_text: bool = False,
        full_text_override: Optional[str] = None,
    ) -> list[sec_processor._SectionBlock]:
        """记录 fast mode 参数并委托原函数执行。"""

        del full_text_override
        call_flags.append((bool(fast_mode), bool(single_full_text)))
        return original_build_sections(
            document,
            fast_mode=fast_mode,
            single_full_text=single_full_text,
        )

    monkeypatch.setattr(sec_processor, "_parse_document", lambda html_content, form_type: fake_doc)
    monkeypatch.setattr(sec_processor, "_build_sections", _tracked_build_sections)

    processor = _FastSecProcessor(source, form_type="20-F")
    assert processor.list_sections()
    assert call_flags == [(True, False)]


@pytest.mark.unit
def test_build_sections_fast_mode_single_full_text_returns_one_section() -> None:
    """验证快速模式可按配置返回单全文章节。"""

    section_1 = FakeSection(text_value="Alpha section.", tables=[], title="A", name="A")
    section_2 = FakeSection(text_value="Beta section.", tables=[], title="B", name="B")
    fake_doc = FakeDocument(
        sections={
            "section_a": section_1,
            "section_b": section_2,
        },
        tables=[],
        text_value="Alpha section. Beta section.",
    )

    sections = sec_processor._build_sections(
        fake_doc,
        fast_mode=True,
        single_full_text=True,
    )
    assert len(sections) == 1
    assert sections[0].contains_full_text is True
    assert "Alpha section." in sections[0].text


@pytest.mark.unit
def test_build_sections_fast_mode_single_full_text_skips_iter_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证单全文章节模式不会访问底层 section 树。"""

    fake_doc = FakeDocument(
        sections={"only": FakeSection(text_value="Alpha section.", tables=[], title="A", name="A")},
        tables=[],
        text_value="Alpha section.",
    )

    def _boom_iter_sections(document: Any) -> list[tuple[str, Any]]:
        """若被调用则抛错，验证该路径应被短路。"""

        del document
        raise AssertionError("single_full_text mode should bypass _iter_sections")

    monkeypatch.setattr(sec_section_build, "_iter_sections", _boom_iter_sections)
    sections = sec_processor._build_sections(
        fake_doc,
        fast_mode=True,
        single_full_text=True,
    )
    assert len(sections) == 1


@pytest.mark.unit
def test_sec_processor_get_full_text_uses_single_section_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证单全文章节模式下 `get_full_text` 复用初始化缓存。"""

    class _CachedFullTextProcessor(SecProcessor):
        """测试专用处理器：启用 fast + 单全文章节。"""

        _ENABLE_FAST_SECTION_BUILD = True
        _FAST_SECTION_BUILD_SINGLE_FULL_TEXT = True

    class _CountingDocument(FakeDocument):
        """带调用计数的文档对象。"""

        def __init__(self) -> None:
            """初始化计数文档。

            Args:
                无。

            Returns:
                无。

            Raises:
                ValueError: 参数非法时抛出。
            """

            super().__init__(
                sections={"only": FakeSection(text_value="Alpha section.", tables=[], title="A", name="A")},
                tables=[],
                text_value="Alpha section. Beta section.",
            )
            self.text_call_count = 0

        def text(self) -> str:
            """返回全文并记录调用次数。

            Args:
                无。

            Returns:
                全文文本。

            Raises:
                RuntimeError: 第 2 次调用起抛错，用于验证缓存生效。
            """

            self.text_call_count += 1
            if self.text_call_count >= 2:
                raise RuntimeError("text() should not be called after cache is prepared")
            return super().text()

    document = _CountingDocument()
    source = _make_source(tmp_path, "full_text_cache_filing.html")
    monkeypatch.setattr(sec_processor, "_parse_document", lambda html_content, form_type: document)

    processor = _CachedFullTextProcessor(source, form_type="20-F")
    assert processor.get_full_text() == "Alpha section. Beta section."
    assert document.text_call_count == 1


@pytest.mark.unit
def test_sec_processor_single_full_text_preload_falls_back_to_raw_html(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证单全文章节预加载会在空全文时回退到原始 HTML 抽文。

    Args:
        tmp_path: 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    class _FallbackFullTextProcessor(SecProcessor):
        """测试专用处理器：启用单全文章节预加载。"""

        _ENABLE_FAST_SECTION_BUILD = True
        _FAST_SECTION_BUILD_SINGLE_FULL_TEXT = True

    class _EmptyTextDocument(FakeDocument):
        """返回空全文的文档桩。"""

        def __init__(self) -> None:
            """初始化空全文文档。

            Args:
                无。

            Returns:
                无。

            Raises:
                ValueError: 参数非法时抛出。
            """

            super().__init__(
                sections={"only": FakeSection(text_value="", tables=[], title="A", name="A")},
                tables=[],
                text_value="   ",
            )
            self.text_call_count = 0

        def text(self) -> str:
            """返回空全文并记录调用次数。

            Args:
                无。

            Returns:
                空白全文字符串。

            Raises:
                RuntimeError: 读取失败时抛出。
            """

            self.text_call_count += 1
            return "   "

    html_content = "<html><body><p>Recovered full text from raw html</p></body></html>"
    recovered_text = "Recovered full text from raw html"
    source = _make_source(tmp_path, "single_full_text_raw_html.html")
    source.path.write_text(html_content, encoding="utf-8")

    document = _EmptyTextDocument()
    captured_html_inputs: list[str] = []

    monkeypatch.setattr(sec_processor, "_parse_document", lambda html_text, form_type: document)
    monkeypatch.setattr(
        sec_processor,
        "_extract_text_from_raw_html",
        lambda raw_html: captured_html_inputs.append(raw_html) or recovered_text,
    )

    processor = _FallbackFullTextProcessor(source, form_type="20-F")

    section_content = processor.read_section("s_0001")
    assert section_content["contains_full_text"] is True
    assert section_content["content"] == recovered_text
    assert processor.get_full_text() == recovered_text
    assert document.text_call_count == 1
    assert captured_html_inputs == [html_content]


@pytest.mark.unit
def test_sec_processor_get_full_text_falls_back_to_raw_html_when_document_text_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证运行时 `get_full_text()` 会在空全文时回退到原始 HTML 抽文。

    Args:
        tmp_path: 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    class _EmptyTextDocument(FakeDocument):
        """返回空全文的文档桩。"""

        def __init__(self) -> None:
            """初始化空全文文档。

            Args:
                无。

            Returns:
                无。

            Raises:
                ValueError: 参数非法时抛出。
            """

            super().__init__(sections={}, tables=[], text_value=" ")
            self.text_call_count = 0

        def text(self) -> str:
            """返回空全文并记录调用次数。

            Args:
                无。

            Returns:
                空白全文字符串。

            Raises:
                RuntimeError: 读取失败时抛出。
            """

            self.text_call_count += 1
            return " "

    html_content = "<html><body><p>Fallback text from runtime html extraction</p></body></html>"
    recovered_text = "Fallback text from runtime html extraction"
    source = _make_source(tmp_path, "runtime_full_text_raw_html.html")
    source.path.write_text(html_content, encoding="utf-8")

    document = _EmptyTextDocument()
    captured_html_inputs: list[str] = []

    monkeypatch.setattr(sec_processor, "_parse_document", lambda html_text, form_type: document)
    monkeypatch.setattr(
        sec_processor,
        "_extract_text_from_raw_html",
        lambda raw_html: captured_html_inputs.append(raw_html) or recovered_text,
    )

    processor = SecProcessor(source, form_type="10-K")

    assert processor.get_full_text() == recovered_text
    # 普通模式初始化建 section 时会读取一次全文，运行时 get_full_text() 再读取一次。
    assert document.text_call_count == 2
    assert captured_html_inputs == [html_content]


@pytest.mark.unit
def test_is_financial_table_rejects_cover_registration_layout() -> None:
    """验证封面注册信息表不会被误判为 financial。"""

    cover_table = FakeTable(
        text_value=(
            "Title of each class Trading symbol(s) Name of each exchange on which registered "
            "Common Stock, $0.00001 par value per share AAPL"
        ),
        df=pd.DataFrame([{"A": 1}]),
        caption=None,
        is_financial=True,
        semantic_type="TableType.FINANCIAL",
    )
    assert (
        sec_table_extraction._is_financial_table(
            cover_table,
            table_text=cover_table.text(),
            caption=None,
            context_before="Securities registered pursuant to Section 12(b) of the Act:",
        )
        is False
    )
    assert (
        sec_table_extraction._is_financial_table(
            cover_table,
            table_text="CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)",
            caption="CONSOLIDATED STATEMENTS OF OPERATIONS",
            context_before="",
        )
        is True
    )


@pytest.mark.unit
def test_sec_processor_default_header_detection() -> None:
    """验证默认数字表头识别规则。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert sec_table_extraction._looks_like_default_headers(["0", "1", "2"]) is True
    assert sec_table_extraction._looks_like_default_headers(["1", "2"]) is True
    assert sec_table_extraction._looks_like_default_headers(["2021", "2020"]) is True
    assert sec_table_extraction._looks_like_default_headers(["Revenue", "Gross Margin"]) is False


@pytest.mark.unit
def test_sec_processor_deduplicate_headers_keep_first() -> None:
    """验证重复行头仅保留首次出现，不追加后缀。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    deduped = sec_table_extraction._deduplicate_headers(
        ["Products", "Services", "Products", "services", "Total"],
    )
    assert deduped == ["Products", "Services", "Total"]


@pytest.mark.unit
def test_sec_processor_financial_apis(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证财务接口输出结构。

    Args:
        monkeypatch: pytest monkeypatch。
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake_doc = FakeDocument(sections={}, tables=[], text_value="doc")
    monkeypatch.setattr(sec_processor, "_parse_document", lambda html_content, form_type: fake_doc)
    source = _make_source(tmp_path, "financial.html")
    processor = SecProcessor(source, form_type="10-K")

    statement_df = pd.DataFrame(
        [
            {
                "concept": "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
                "label": "Net sales",
                "2024-09-28": 391035000000.0,
                "2023-09-30": 383285000000.0,
            }
        ]
    )
    facts_rows = [
        {
            "fact_key": "k1",
            "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "label": "Net sales",
            "numeric_value": 391035000000.0,
            "unit_ref": "USD",
            "period_end": "2024-09-28",
            "fiscal_year": 2024,
            "fiscal_period": "FY",
            "statement_type": "IncomeStatement",
        }
    ]
    fake_xbrl = FakeXbrl(statement_df=statement_df, rows=facts_rows)
    monkeypatch.setattr(SecProcessor, "_get_xbrl", lambda self: fake_xbrl)

    statement_result = processor.get_financial_statement("income")
    assert statement_result["data_quality"] == "xbrl"
    assert statement_result["periods"]
    assert statement_result["rows"]
    locator = _statement_locator(statement_result)
    assert locator is not None
    assert locator["statement_type"] == "income"
    assert locator["statement_title"] == "Income Statement"
    assert locator["period_labels"] == ["FY2024", "FY2023"]
    assert locator["row_labels"] == ["Net sales"]

    query_result = processor.query_xbrl_facts(
        concepts=["RevenueFromContractWithCustomerExcludingAssessedTax"],
        statement_type="IncomeStatement",
        period_end="2024-09-28",
        fiscal_year=2024,
        fiscal_period="FY",
    )
    assert query_result["total"] == 1
    assert query_result["facts"][0]["concept"]


@pytest.mark.unit
def test_sec_processor_financial_apis_xbrl_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 XBRL 缺失时的降级输出。

    Args:
        monkeypatch: pytest monkeypatch。
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake_doc = FakeDocument(sections={}, tables=[], text_value="doc")
    monkeypatch.setattr(sec_processor, "_parse_document", lambda html_content, form_type: fake_doc)
    monkeypatch.setattr(SecProcessor, "_get_xbrl", lambda self: None)
    source = _make_source(tmp_path, "financial_missing.html")
    processor = SecProcessor(source, form_type="10-K")

    statement_result = processor.get_financial_statement("income")
    assert statement_result["data_quality"] == "partial"
    assert _result_reason(statement_result) == "xbrl_not_available"

    query_result = processor.query_xbrl_facts(concepts=["Revenue"])
    assert query_result["total"] == 0
    assert _result_reason(query_result) == "xbrl_not_available"


@pytest.mark.unit
def test_safe_table_dataframe_suppresses_performance_warning() -> None:
    """验证表格 DataFrame 转换会局部屏蔽 PerformanceWarning。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataframe = sec_table_extraction._safe_table_dataframe(_WarningTable())

    assert isinstance(dataframe, pd.DataFrame)
    assert not any(issubclass(item.category, pd.errors.PerformanceWarning) for item in caught)


@pytest.mark.unit
def test_safe_statement_dataframe_suppresses_performance_warning() -> None:
    """验证报表 DataFrame 转换会局部屏蔽 PerformanceWarning。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataframe = sec_processor._safe_statement_dataframe(_WarningStatement())

    assert isinstance(dataframe, pd.DataFrame)
    assert not any(issubclass(item.category, pd.errors.PerformanceWarning) for item in caught)


class _BadTextObject:
    """text() 抛错的对象桩。"""

    def text(self) -> str:
        """返回文本。

        Args:
            无。

        Returns:
            文本。

        Raises:
            RuntimeError: 人工触发。
        """

        raise RuntimeError("boom")


class _BadIterableTables:
    """不可迭代表格容器桩。"""

    def __iter__(self) -> Any:
        """返回迭代器。

        Args:
            无。

        Returns:
            迭代器。

        Raises:
            RuntimeError: 人工触发。
        """

        raise RuntimeError("iter failed")


class _BadTableDict:
    """to_dict 抛错的表格桩。"""

    def to_dict(self) -> dict[str, Any]:
        """返回字典。

        Args:
            无。

        Returns:
            字典。

        Raises:
            RuntimeError: 人工触发。
        """

        raise RuntimeError("dict failed")


class _BadMarkdownDf:
    """to_markdown 抛错的 DataFrame 桩。"""

    @property
    def empty(self) -> bool:
        """是否为空。

        Args:
            无。

        Returns:
            是否为空。

        Raises:
            RuntimeError: 无。
        """

        return False

    def to_markdown(self, index: bool = False) -> str:
        """转 markdown。

        Args:
            index: 是否输出索引。

        Returns:
            markdown 文本。

        Raises:
            RuntimeError: 人工触发。
        """

        del index
        raise RuntimeError("markdown failed")


class _FakeQueryChain:
    """XBRL 查询链桩。"""

    def __init__(self, rows: list[Any], *, fail_on_execute: bool = False) -> None:
        """初始化查询链。

        Args:
            rows: 返回行。
            fail_on_execute: 是否在 execute 抛错。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._rows = rows
        self._fail_on_execute = fail_on_execute

    def by_concept(self, concept: str) -> "_FakeQueryChain":
        """按概念筛选。

        Args:
            concept: 概念名。

        Returns:
            自身。

        Raises:
            RuntimeError: 无。
        """

        del concept
        return self

    def by_statement_type(self, statement_type: str) -> "_FakeQueryChain":
        """按报表类型筛选。

        Args:
            statement_type: 报表类型。

        Returns:
            自身。

        Raises:
            RuntimeError: 无。
        """

        del statement_type
        return self

    def by_fiscal_year(self, fiscal_year: int) -> "_FakeQueryChain":
        """按财年筛选。

        Args:
            fiscal_year: 财年。

        Returns:
            自身。

        Raises:
            RuntimeError: 无。
        """

        del fiscal_year
        return self

    def by_fiscal_period(self, fiscal_period: str) -> "_FakeQueryChain":
        """按财季筛选。

        Args:
            fiscal_period: 财季。

        Returns:
            自身。

        Raises:
            RuntimeError: 无。
        """

        del fiscal_period
        return self

    def by_value(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> "_FakeQueryChain":
        """按值筛选。

        Args:
            min_value: 最小值。
            max_value: 最大值。

        Returns:
            自身。

        Raises:
            RuntimeError: 无。
        """

        del min_value, max_value
        return self

    def execute(self) -> list[Any]:
        """执行查询。

        Args:
            无。

        Returns:
            行列表。

        Raises:
            RuntimeError: 人工触发。
        """

        if self._fail_on_execute:
            raise RuntimeError("execute failed")
        return self._rows


class _FakeXbrlQueryProvider:
    """提供 query() 的 XBRL 桩。"""

    def __init__(self, rows: list[Any], *, fail_on_execute: bool = False) -> None:
        """初始化对象。

        Args:
            rows: 查询返回行。
            fail_on_execute: execute 是否抛错。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._rows = rows
        self._fail_on_execute = fail_on_execute

    def query(self) -> _FakeQueryChain:
        """构造查询链。

        Args:
            无。

        Returns:
            查询链对象。

        Raises:
            RuntimeError: 无。
        """

        return _FakeQueryChain(self._rows, fail_on_execute=self._fail_on_execute)


class _SequencedParser:
    """按预设序列返回结果/抛错的解析器桩。"""

    side_effects: list[Any] = []
    captured_configs: list[Any] = []

    def __init__(self, config: Any) -> None:
        """记录初始化配置。

        Args:
            config: 解析配置对象。

        Returns:
            无。

        Raises:
            RuntimeError: 无。
        """

        self._config = config
        self.__class__.captured_configs.append(config)

    def parse(self, html_content: str) -> Any:
        """按序执行预设副作用。

        Args:
            html_content: HTML 内容。

        Returns:
            文档对象或预设返回值。

        Raises:
            Exception: 当副作用项为异常实例时抛出。
        """

        del html_content
        if not self.__class__.side_effects:
            return {"status": "ok"}
        effect = self.__class__.side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect

    @classmethod
    def set_side_effects(cls, effects: list[Any]) -> None:
        """重置副作用与配置捕获。

        Args:
            effects: 解析副作用序列。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        cls.side_effects = list(effects)
        cls.captured_configs = []


@pytest.mark.unit
def test_parse_document_and_basic_helpers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证解析失败分支与基础辅助函数分支。

    Args:
        monkeypatch: monkeypatch fixture。
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    class _BoomParser:
        """解析器桩：parse 抛错。"""

        def __init__(self, config: Any) -> None:
            """初始化。

            Args:
                config: 配置对象。

            Returns:
                无。

            Raises:
                RuntimeError: 无。
            """

            del config

        def parse(self, html_content: str) -> Any:
            """解析文本。

            Args:
                html_content: HTML 内容。

            Returns:
                文档对象。

            Raises:
                RuntimeError: 人工触发。
            """

            del html_content
            raise RuntimeError("parse failed")

    monkeypatch.setattr(sec_processor, "HTMLParser", _BoomParser)
    with pytest.raises(RuntimeError, match="SEC document parsing failed"):
        sec_processor._parse_document("<html></html>", "10-K")

    assert sec_processor._normalize_form_type(None) is None
    assert sec_processor._normalize_form_type("  ") is None
    assert sec_processor._normalize_form_type("schedule 13g/a") == "SC 13G/A"
    assert sec_processor._infer_suffix_from_uri("") == ""
    assert sec_processor._infer_suffix_from_uri("local://a/b/file.HTML") == ".html"

    with pytest.raises(ValueError):
        text_utils.format_section_ref(0)
    with pytest.raises(ValueError):
        text_utils.format_table_ref(0)
    assert sec_section_build._table_fingerprint("") == ""

    doc_with_bad_tables = type("Doc", (), {"tables": _BadIterableTables()})()
    assert sec_table_extraction._iter_document_tables(doc_with_bad_tables) == []
    assert sec_processor._safe_document_text(_BadTextObject()) == ""
    assert sec_section_build._safe_section_text(_BadTextObject()) == ""
    assert sec_section_build._safe_table_text(_BadTextObject()) == ""

    file_path = tmp_path / "a.txt"
    file_path.write_text("x", encoding="utf-8")
    assert sec_processor._load_text(file_path) == "x"


@pytest.mark.unit
def test_parse_document_retries_with_doubled_limit_on_document_too_large(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证超限异常会触发一次翻倍重试并成功返回。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    expected_document = {"id": "doc_ok"}
    _SequencedParser.set_side_effects(
        [
            DocumentTooLargeError(size=300 * 1024 * 1024, max_size=256 * 1024 * 1024),
            expected_document,
        ]
    )
    monkeypatch.setattr(sec_processor, "HTMLParser", _SequencedParser)

    actual = sec_processor._parse_document("<html></html>", "10-K")

    assert actual == expected_document
    assert len(_SequencedParser.captured_configs) == 2
    first_config, second_config = _SequencedParser.captured_configs
    assert first_config.max_document_size == 256 * 1024 * 1024
    assert second_config.max_document_size == 512 * 1024 * 1024
    assert first_config.streaming_threshold == 10 * 1024 * 1024
    assert second_config.streaming_threshold == 10 * 1024 * 1024
    assert first_config.form == "10-K"
    assert second_config.form == "10-K"


@pytest.mark.unit
def test_parse_document_raises_runtime_error_when_retry_still_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证超限重试后仍失败时保持统一错误语义。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    _SequencedParser.set_side_effects(
        [
            DocumentTooLargeError(size=300 * 1024 * 1024, max_size=256 * 1024 * 1024),
            RuntimeError("retry failed"),
        ]
    )
    monkeypatch.setattr(sec_processor, "HTMLParser", _SequencedParser)

    with pytest.raises(RuntimeError, match="SEC document parsing failed") as exc_info:
        sec_processor._parse_document("<html></html>", "10-K")

    assert len(_SequencedParser.captured_configs) == 2
    assert _SequencedParser.captured_configs[0].max_document_size == 256 * 1024 * 1024
    assert _SequencedParser.captured_configs[1].max_document_size == 512 * 1024 * 1024
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "retry failed"


@pytest.mark.unit
def test_parse_document_does_not_retry_for_non_size_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证非超限异常不会触发重试。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    _SequencedParser.set_side_effects([RuntimeError("parse failed")])
    monkeypatch.setattr(sec_processor, "HTMLParser", _SequencedParser)

    with pytest.raises(RuntimeError, match="SEC document parsing failed") as exc_info:
        sec_processor._parse_document("<html></html>", "10-K")

    assert len(_SequencedParser.captured_configs) == 1
    assert _SequencedParser.captured_configs[0].max_document_size == 256 * 1024 * 1024
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "parse failed"


@pytest.mark.unit
def test_context_and_header_helper_paths() -> None:
    """验证上下文提取与表头提取的边界路径。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    html = """
    <html>
      <head>
        <style>
          table { border-spacing: 2px; }
          div.largetext { color: blue; }
        </style>
      </head>
      <body>
      <h1>Item 1</h1>
      <p>before table marker context</p>
      <table><tr><td>A</td></tr></table>
      </body>
    </html>
    """
    contexts = sec_processor._extract_dom_table_contexts(html, max_chars=40)
    assert contexts
    assert "before table marker" in contexts[0]
    assert "border-spacing" not in contexts[0]
    assert "largetext" not in contexts[0]

    section = sec_processor._SectionBlock(
        ref="s_0001",
        title="T",
        level=1,
        parent_ref=None,
        preview="p",
        text="abc marker text and more",
        table_refs=[],
        table_fingerprints=set(),
        contains_full_text=False,
    )
    by_ref = {"s_0001": section}
    assert sec_table_extraction._extract_context_before(None, by_ref, "marker") == ""
    assert sec_table_extraction._extract_context_before("sec_x", by_ref, "marker") == ""
    assert sec_table_extraction._extract_context_before("s_0001", by_ref, "marker text") != ""
    assert sec_table_extraction._extract_context_before("s_0001", by_ref, "   ") == ""
    assert sec_table_extraction._resolve_dom_context_by_index([" a "], 2) == ""

    df = pd.DataFrame(columns=pd.Index(["Products", "Products"], dtype="object"))
    assert sec_table_extraction._extract_headers_from_dataframe(df) is None
    assert sec_table_extraction._extract_row_headers_from_table_dict(_BadTableDict()) is None

    row_headers = sec_table_extraction._extract_row_headers_from_table_dict(
        type(
            "Obj",
            (),
            {"to_dict": lambda self: {"data": [{"a": "Products", "b": 1}, {"a": "Products"}]}},
        )(),
    )
    assert row_headers == ["Products"]

    assert sec_table_extraction._is_low_information_header("unnamed: 0") is True
    assert sec_table_extraction._is_low_information_header("Products") is False
    assert sec_table_extraction._normalize_header_list(["A", "a", None, "B"]) == ["A", "B"]
    assert sec_table_extraction._looks_like_default_headers([]) is True
    assert sec_table_extraction._looks_like_default_headers(["2024", "2023"]) is True


@pytest.mark.unit
def test_render_helpers_and_text_snippets(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证渲染与文本片段辅助函数分支。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    replacement = sec_processor._replace_table_with_placeholder("abc", "short", "t_0001")
    assert replacement["replaced"] is False
    replacement_ok = sec_processor._replace_table_with_placeholder(
        "prefix very long table text for replacement suffix",
        "very long table text for replacement",
        "t_0001",
    )
    assert replacement_ok["replaced"] is True

    assert sec_processor._append_missing_placeholders("", ["t_0001"]) == "[[t_0001]]"
    assert sec_processor._append_missing_placeholders("[[t_0001]]", ["t_0001"]) == "[[t_0001]]"

    no_records = sec_processor._render_records_table(type("NoDf", (), {})())
    assert no_records is None

    monkeypatch.setattr(sec_table_extraction, "_safe_table_dataframe", lambda table_obj: _BadMarkdownDf())
    with pytest.raises(RuntimeError, match="SEC 表格 markdown 渲染失败"):
        sec_processor._render_markdown_table(type("T", (), {"to_dict": lambda self: {"a": 1}})(), "")

    assert sec_table_extraction._normalize_optional_string("  a \n b  ") == "a b"


@pytest.mark.unit
def test_xbrl_file_and_statement_helpers(tmp_path: Path) -> None:
    """验证 XBRL 发现与报表辅助函数分支。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    (tmp_path / "report_pre.xml").write_text("", encoding="utf-8")
    (tmp_path / "report_cal.xml").write_text("", encoding="utf-8")
    (tmp_path / "report_def.xml").write_text("", encoding="utf-8")
    (tmp_path / "report_lab.xml").write_text("", encoding="utf-8")
    (tmp_path / "report.xsd").write_text("", encoding="utf-8")
    (tmp_path / "instance.xml").write_text("", encoding="utf-8")
    (tmp_path / "FilingSummary.xml").write_text("", encoding="utf-8")

    discovered = xbrl_file_discovery.discover_xbrl_files(tmp_path)
    assert discovered["instance"] is not None
    assert discovered["schema"] is not None

    fallback_files = xbrl_file_discovery._fallback_instance_files(tmp_path)
    assert any(path.name == "instance.xml" for path in fallback_files)
    assert all(not path.name.endswith("_pre.xml") for path in fallback_files)

    assert xbrl_file_discovery._first_existing([[tmp_path / "missing.xml"]]) is None

    periods = sec_processor._extract_period_columns(["concept", "2024-09-28", "2024/09/28"])
    assert periods == ["2024-09-28"]

    statement_df = pd.DataFrame(
        [
            {"concept": "", "label": "", "2024-09-28": 1},
            {"concept": "us-gaap:Revenue", "label": "Revenue", "2024-09-28": "2"},
        ]
    )
    rows = sec_processor._build_statement_rows(statement_df, ["2024-09-28"])
    assert len(rows) == 1
    assert rows[0]["values"] == [2.0]

    assert sec_processor._build_period_summary("bad-date")["fiscal_year"] is None
    assert sec_xbrl_query._to_optional_float(None) is None
    assert sec_xbrl_query._to_optional_float("   ") is None
    assert sec_xbrl_query._to_optional_float(float("nan")) is None
    assert sec_xbrl_query._to_optional_float("3.5") == 3.5


@pytest.mark.unit
def test_query_and_fact_helper_paths() -> None:
    """验证查询与 fact 标准化辅助函数分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert sec_processor._normalize_query_statement_type("  ") is None
    assert sec_processor._normalize_query_statement_type("cash_flow") == "cash_flow"

    rows: list[Any] = [
        {"fact_key": "a", "concept": "Revenue", "period_end": "2024-09-28", "numeric_value": 1},
        {"fact_key": "a", "concept": "Revenue", "period_end": "2024-09-28", "numeric_value": 1},
        {"fact_key": "b", "concept": "Revenue", "period_end": "2023-09-30", "numeric_value": 2},
        "invalid-row",
    ]
    xbrl = _FakeXbrlQueryProvider(rows=rows, fail_on_execute=False)
    query_rows = sec_processor._query_facts_rows(
        xbrl=xbrl,  # type: ignore[arg-type]
        concepts=["Revenue"],
        statement_type="IncomeStatement",
        period_end="2024-09-28",
        fiscal_year=2024,
        fiscal_period="fy",
        min_value=0,
        max_value=10,
    )
    assert len(query_rows) == 1

    xbrl_fail = _FakeXbrlQueryProvider(rows=[], fail_on_execute=True)
    assert sec_processor._query_facts_rows(
        xbrl=xbrl_fail,  # type: ignore[arg-type]
        concepts=["Revenue"],
        statement_type=None,
        period_end=None,
        fiscal_year=None,
        fiscal_period=None,
        min_value=None,
        max_value=None,
    ) == []

    key = sec_xbrl_query._build_fact_dedup_key({"fact_key": "k", "concept": "c", "period_end": "p", "value": 1})
    assert key == "k|c|p|1"

    normalized = sec_processor._normalize_fact_row(
        {
            "concept": "c",
            "original_label": "L",
            "value": "1",
            "unit_ref": "usd",
            "period_end": "2024-09-28",
        }
    )
    assert normalized["label"] == "L"
    assert normalized["unit"] == "usd"
    assert normalized["numeric_value"] == 1.0
    assert normalized["text_value"] is None
    assert normalized["content_type"] is None

    text_normalized = sec_processor._normalize_fact_row(
        {
            "concept": "c",
            "value": "<div>abc</div>",
        }
    )
    assert text_normalized["numeric_value"] is None
    assert text_normalized["text_value"] == "<div>abc</div>"
    assert text_normalized["content_type"] == "xhtml"

    units = sec_processor._infer_units_from_xbrl_query(
        _as_xbrl(_FakeXbrlQueryProvider(rows=[{"unit_ref": "usd"}], fail_on_execute=False)),
    )
    assert units == "USD"

    units_none = sec_processor._infer_units_from_xbrl_query(
        _as_xbrl(_FakeXbrlQueryProvider(rows=["bad-row"], fail_on_execute=False)),
    )
    assert units_none is None


@pytest.mark.unit
def test_query_facts_rows_filters_textblock_and_requires_exact_local_name() -> None:
    """验证 XBRL facts 查询仅返回数值且按本地名精确匹配。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    rows: list[dict[str, Any]] = [
        {
            "fact_key": "k1",
            "concept": "us-gaap:Revenues",
            "numeric_value": 100,
            "period_end": "2024-09-28",
        },
        {
            "fact_key": "k2",
            "concept": "us-gaap:ScheduleOfRevenuesFromExternalCustomersAndLongLivedAssetsByGeographicalAreasTableTextBlock",
            "value": "<div>table</div>",
            "period_end": "2024-09-28",
        },
        {
            "fact_key": "k3",
            "concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "numeric_value": 90,
            "period_end": "2024-09-28",
        },
    ]
    xbrl = _FakeXbrlQueryProvider(rows=rows, fail_on_execute=False)

    query_rows = sec_processor._query_facts_rows(
        xbrl=_as_xbrl(xbrl),
        concepts=["Revenues"],
        statement_type=None,
        period_end="2024-09-28",
        fiscal_year=None,
        fiscal_period=None,
        min_value=None,
        max_value=None,
    )

    assert len(query_rows) == 1
    assert query_rows[0]["concept"] == "us-gaap:Revenues"
    assert query_rows[0]["numeric_value"] == 100.0

    units_error = sec_processor._infer_units_from_xbrl_query(
        _as_xbrl(_FakeXbrlQueryProvider(rows=[], fail_on_execute=True)),
    )
    assert units_error is None

    assert sec_processor._infer_currency_from_units(None) is None
    assert sec_processor._infer_currency_from_units("shares per USD") == "USD"
    assert sec_processor._infer_currency_from_units("CNY") == "CNY"


@pytest.mark.unit
def test_dataframe_to_records_handles_nan_and_whitespace() -> None:
    """验证 DataFrame 转 records 时的空值与空白标准化。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    dataframe = pd.DataFrame(
        {
            "Item": [" Revenue ", None],
            "Value": [1.0, float("nan")],
        }
    )
    records = sec_table_extraction._dataframe_to_records(dataframe)
    assert records[0]["Item"] == "Revenue"
    assert records[0]["Value"] == 1.0
    assert records[1]["Item"] is None
    assert records[1]["Value"] is None


@pytest.mark.unit
def test_markdown_records_fallback_and_numeric_normalization() -> None:
    """验证 markdown 回解析 records 与数字文本标准化。"""

    payload = sec_table_extraction._render_records_from_markdown_table(
        markdown_text=(
            "| | 2024 |\n"
            "|---|---|\n"
            "| Revenue | 1,234 |\n"
            "| Cost | (567) |\n"
        ),
        allow_generated_columns=True,
    )
    assert payload is not None
    assert payload["columns"] == ["col_1", "col_2"]
    assert payload["data"][0]["col_1"] == "Revenue"
    assert payload["data"][0]["col_2"] == "1234"
    assert payload["data"][1]["col_2"] == "-567"


@pytest.mark.unit
def test_records_quality_check_rejects_large_column_mismatch_and_generated_columns() -> None:
    """验证 records 质量门槛会拒绝列数失真与高占比生成列。"""

    low_quality_payload = {
        "columns": ["col_1"],
        "data": [{"col_1": "94-2404110"} for _ in range(6)],
    }
    assert (
        sec_table_extraction._is_records_payload_quality_ok(
            low_quality_payload,
            aggressive=True,
            expected_col_count=9,
        )
        is False
    )
    acceptable_payload = {
        "columns": ["Item", "Value", "Period"],
        "data": [
            {"Item": "Revenue", "Value": "1000", "Period": "Q2"},
            {"Item": "Net income", "Value": "200", "Period": "Q2"},
        ],
    }
    assert (
        sec_table_extraction._is_records_payload_quality_ok(
            acceptable_payload,
            aggressive=False,
            expected_col_count=3,
        )
        is True
    )


@pytest.mark.unit
def test_should_prioritize_records_for_core_financial_caption() -> None:
    """验证核心财务表标题会触发 records 优先策略。"""

    table_block = sec_table_extraction._TableBlock(
        ref="t_0001",
        table_obj=object(),
        text="Annual data",
        fingerprint="fp",
        caption="CONSOLIDATED STATEMENTS OF OPERATIONS",
        row_count=10,
        col_count=4,
        headers=None,
        section_ref="s_0001",
        context_before="",
        is_financial=False,
        table_type="layout",
    )
    assert sec_processor._should_prioritize_records_output(table_block) is True


@pytest.mark.unit
def test_infer_xbrl_taxonomy_from_query_rows() -> None:
    """验证 taxonomy 可从 query 返回 concept 前缀推断。"""

    provider = _FakeXbrlQueryProvider(
        rows=[{"concept": "ifrs-full:Revenue"}],
        fail_on_execute=False,
    )
    taxonomy = sec_processor._infer_xbrl_taxonomy(provider)  # type: ignore[arg-type]
    assert taxonomy == "ifrs-full"
    assert sec_xbrl_query._extract_taxonomy_from_concept("us-gaap:Assets") == "us-gaap"


@pytest.mark.unit
def test_render_markdown_table_nan_cleaned(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 _render_markdown_table 输出中 NaN 被替换为空字符串。

    MultiIndex 合并单元格在 DataFrame 中表现为 NaN，to_markdown() 会将其
    输出为 'nan' 字符串。修复后应把 NaN 填为空字符串再转 markdown。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    import pandas as pd

    # 构造含 NaN 的 DataFrame，模拟 MultiIndex 合并单元格
    df_with_nan = pd.DataFrame({
        "Period": ["Q1", "Q2"],
        "Revenue": [100, float("nan")],
        "Growth": [float("nan"), 0.05],
    })
    monkeypatch.setattr(
        sec_table_extraction,
        "_safe_table_dataframe",
        lambda table_obj: df_with_nan,
    )
    fake_table = type("T", (), {"to_dict": lambda self: {}})()
    result = sec_processor._render_markdown_table(fake_table, "")

    assert "nan" not in result.lower(), f"NaN should be cleaned from markdown output: {result}"
    # 验证正常值仍然被保留
    assert "Q1" in result
    assert "100" in result
    assert "0.05" in result
