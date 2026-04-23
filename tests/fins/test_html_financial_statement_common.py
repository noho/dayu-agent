"""HTML 财务报表共享解析层测试。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from collections.abc import Callable
from typing import Mapping, Optional

import pandas as pd
import pytest

from dayu.fins.processors import html_financial_statement_common as html_statement_module


@dataclass(frozen=True)
class _StubTable:
    """测试用表格桩对象。"""

    tag: object
    caption: Optional[str]
    context_before: Optional[str] = None


def _build_parse_table_dataframe(
    table_map: dict[object, pd.DataFrame],
) -> Callable[[object], Optional[pd.DataFrame]]:
    """构建测试用 DataFrame 解析回调。

    Args:
        table_map: `table对象 -> dataframe` 的映射。

    Returns:
        供共享解析层注入的回调函数。

    Raises:
        无。
    """

    def _parse_table_dataframe(table: object) -> Optional[pd.DataFrame]:
        """按表格对象返回对应 DataFrame 副本。"""

        dataframe = table_map.get(table)
        if dataframe is None:
            return None
        return dataframe.copy()

    return _parse_table_dataframe


def _statement_locator(result: Mapping[str, object]) -> dict[str, object] | None:
    """安全读取财务报表结果中的 statement_locator。

    Args:
        result: 财务报表结果。

    Returns:
        `statement_locator` 字典；不存在时返回 `None`。

    Raises:
        无。
    """

    locator = result.get("statement_locator")
    if isinstance(locator, dict):
        return locator
    return None


def _row_labels(locator: Mapping[str, object]) -> list[str]:
    """把 statement_locator.row_labels 收窄为字符串列表。"""

    raw_labels = locator.get("row_labels")
    if not isinstance(raw_labels, list):
        return []
    return [label for label in raw_labels if isinstance(label, str)]


@pytest.mark.unit
def test_build_html_statement_result_from_tables_preserves_statement_payload() -> None:
    """验证共享层可稳定构建结构化报表结果。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    table = _StubTable(
        tag=object(),
        caption="Condensed Consolidated Balance Sheets (In thousands of HK$)",
    )
    dataframe = pd.DataFrame(
        [
            ["GROUP STATEMENT OF FINANCIAL POSITION", "", "As at", "As at"],
            ["GROUP STATEMENT OF FINANCIAL POSITION", "", "Dec", "Dec"],
            ["GROUP STATEMENT OF FINANCIAL POSITION", "", "2024", "2023"],
            ["Cash and cash equivalents", "", "100", "80"],
            ["Total assets", "", "300", "260"],
        ]
    )

    result = html_statement_module.build_html_statement_result_from_tables(
        statement_type="balance_sheet",
        tables=[table],
        parse_table_dataframe=_build_parse_table_dataframe({table: dataframe}),
    )

    assert result is not None
    assert result["data_quality"] == "extracted"
    assert result["currency"] == "HKD"
    assert result["units"] == "HK$ in thousands"
    assert result["scale"] == "thousands"
    assert result["periods"] == [
        {"period_end": "2024-12-31", "fiscal_year": 2024, "fiscal_period": "FY"},
        {"period_end": "2023-12-31", "fiscal_year": 2023, "fiscal_period": "FY"},
    ]
    assert result["rows"] == [
        {"concept": "", "label": "Cash and cash equivalents", "values": [100.0, 80.0]},
        {"concept": "", "label": "Total assets", "values": [300.0, 260.0]},
    ]
    locator = _statement_locator(result)
    assert locator is not None
    assert locator["statement_type"] == "balance_sheet"
    assert "Total assets" in _row_labels(locator)


@pytest.mark.unit
def test_select_html_statement_tables_by_row_signals_uses_injected_patterns_and_threshold() -> None:
    """验证共享层按注入参数筛选候选表格。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    strong_table = _StubTable(tag=object(), caption="Strong balance sheet")
    weak_table = _StubTable(tag=object(), caption="Weak balance sheet")
    strong_dataframe = pd.DataFrame(
        [
            ["As at", "December 31, 2024", "December 31, 2023"],
            ["Cash and cash equivalents", "100", "90"],
            ["Trade receivables", "80", "70"],
            ["Total assets", "300", "260"],
            ["Trade payables", "70", "60"],
            ["Total liabilities", "180", "150"],
            ["Total equity", "120", "110"],
        ]
    )
    weak_dataframe = pd.DataFrame(
        [
            ["As at", "December 31, 2024", "December 31, 2023"],
            ["Cash and cash equivalents", "100", "90"],
            ["Trade receivables", "80", "70"],
            ["Inventory", "60", "55"],
            ["Other assets", "40", "35"],
            ["Total assets", "300", "250"],
            ["Share capital", "120", "100"],
        ]
    )
    parse_table_dataframe = _build_parse_table_dataframe(
        {
            strong_table: strong_dataframe,
            weak_table: weak_dataframe,
        }
    )
    line_item_patterns = (
        re.compile(r"(?i)\btotal\s+assets\b"),
        re.compile(r"(?i)\btotal\s+liabilities\b"),
    )

    strong_only = html_statement_module.select_html_statement_tables_by_row_signals(
        tables=[weak_table, strong_table],
        line_item_patterns=line_item_patterns,
        min_hits=2,
        parse_table_dataframe=parse_table_dataframe,
    )
    both_tables = html_statement_module.select_html_statement_tables_by_row_signals(
        tables=[weak_table, strong_table],
        line_item_patterns=line_item_patterns,
        min_hits=1,
        parse_table_dataframe=parse_table_dataframe,
    )

    assert strong_only == [strong_table]
    assert both_tables == [strong_table, weak_table]


@pytest.mark.unit
def test_build_html_statement_result_from_tables_uses_full_table_object_callback() -> None:
    """验证共享层向解析回调传入完整表格对象，而非假设存在 tag。"""

    table = _StubTable(
        tag=object(),
        caption="Consolidated Statements of Operations",
    )
    dataframe = pd.DataFrame(
        [
            ["Year ended", "December 31, 2024"],
            ["Revenue", "100"],
            ["Cost of revenue", "40"],
            ["Gross profit", "60"],
            ["Operating income", "20"],
            ["Net income", "15"],
            ["Earnings per share", "1.5"],
        ]
    )
    received_tables: list[object] = []

    def _parse_table_dataframe(table_obj: object) -> Optional[pd.DataFrame]:
        """记录共享层实际传入的对象。"""

        received_tables.append(table_obj)
        if table_obj is not table:
            return None
        return dataframe.copy()

    result = html_statement_module.build_html_statement_result_from_tables(
        statement_type="income",
        tables=[table],
        parse_table_dataframe=_parse_table_dataframe,
    )

    assert result is not None
    assert received_tables == [table]


@pytest.mark.unit
def test_shared_numeric_parser_supports_currency_percent_and_suffix() -> None:
    """验证迁移后的通用数值解析能力保持不变。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert html_statement_module._parse_optional_numeric("$ 188.6") == pytest.approx(188.6)
    assert html_statement_module._parse_optional_numeric("4.1%") == pytest.approx(4.1)
    assert html_statement_module._parse_optional_numeric("1.06 x") == pytest.approx(1.06)
    assert html_statement_module._parse_optional_numeric("(1,234.5)") == pytest.approx(-1234.5)
    assert html_statement_module._parse_optional_numeric("1.234,5") == pytest.approx(1234.5)


@pytest.mark.unit
def test_statement_date_and_period_helpers_cover_scope_year_and_safe_date_edges() -> None:
    """验证日期/期间 helper 的范围文本、年份提取与非法日期分支。"""

    assert html_statement_module._extract_scope_month_day("As at September 30") == (9, 30)
    assert html_statement_module._extract_scope_month_day("As at Mystery 30") is None
    assert html_statement_module._extract_years("FY 2024 compared with FY 2023 and 2024") == [2024, 2023]
    assert html_statement_module._build_safe_date(year=2024, month=2, day=29) is not None
    assert html_statement_module._build_safe_date(year=2024, month=2, day=30) is None
    assert (
        html_statement_module._normalize_period_end(scope_text="As at September 30", date_text="2024")
        == "2024-09-30"
    )
    assert (
        html_statement_module._normalize_period_end(scope_text="Year ended", date_text="2024")
        == "2024-12-31"
    )
    assert html_statement_module._infer_fiscal_period(
        scope_text="three months ended",
        period_end="2024-03-31",
    ) == "Q1"


@pytest.mark.unit
def test_statement_column_detection_helpers_cover_period_hint_fallback_and_currency() -> None:
    """验证列识别 helper 在低命中表格上会依赖期间信号回退。"""

    hinted_matrix = [
        ["Year ended", "FY 2024", "FY 2023"],
        ["Cash and cash equivalents", "100", "90"],
    ]
    plain_matrix = [
        ["Header", "Column A", "Column B"],
        ["Cash and cash equivalents", "100", "90"],
    ]

    assert html_statement_module._has_period_hint_in_headers(matrix=hinted_matrix, header_row_count=1) is True
    assert html_statement_module._has_period_hint_in_headers(matrix=plain_matrix, header_row_count=1) is False
    assert html_statement_module._detect_value_column_indexes(hinted_matrix, header_row_count=1) == [1, 2]
    assert html_statement_module._detect_value_column_indexes(plain_matrix, header_row_count=1) == []
    assert html_statement_module._extract_currency_for_column(
        scope_text="Amounts in BRL",
        column_header_text="",
    ) == "BRL"
    assert html_statement_module._build_units_label(primary_currency_raw="HK$", scale="thousands") == "HK$ in thousands"


@pytest.mark.unit
def test_statement_date_helpers_cover_multiple_formats_and_header_window_fallback() -> None:
    """验证日期提取与表头窗口 helper 覆盖多种日期格式和回退路径。"""

    assert html_statement_module._collect_header_window_text(
        row=["", "Nine months ended", "", ""],
        column_index=3,
    ) == "Nine months ended"
    assert html_statement_module._collect_header_window_text(
        row=["", "", "", ""],
        column_index=2,
    ) == ""
    assert html_statement_module._extract_first_date("2024/09/30") is not None
    assert html_statement_module._extract_first_date("2024 Sep 30") is not None
    assert html_statement_module._extract_first_date("30/09/2024") is not None
    assert html_statement_module._extract_first_date("09/30/24") is not None
    assert html_statement_module._extract_first_date("September 30, 2024") is not None
    assert html_statement_module._extract_first_date("30 September 2024") is not None
    assert html_statement_module._extract_first_date("31-Dec-24") == html_statement_module._build_safe_date(
        year=2024,
        month=12,
        day=31,
    )
    assert html_statement_module._extract_first_date("Quarter endedSep 2025") == html_statement_module._build_safe_date(
        year=2025,
        month=9,
        day=30,
    )
    assert html_statement_module._extract_first_date("September 2024") == html_statement_module._build_safe_date(
        year=2024,
        month=9,
        day=30,
    )
    assert html_statement_module._extract_first_date("2024 September") == html_statement_module._build_safe_date(
        year=2024,
        month=9,
        day=30,
    )
    assert html_statement_module._extract_first_date("not a date") is None


@pytest.mark.unit
def test_statement_fiscal_period_helpers_cover_token_families_and_period_end_mapping() -> None:
    """验证财期 token 提取覆盖季度、半年度、累计期间与期末映射。"""

    assert html_statement_module._extract_fiscal_period_year("Q2 2024") == ("Q2", 2024)
    assert html_statement_module._extract_fiscal_period_year("2024 3Q") == ("Q3", 2024)
    assert html_statement_module._extract_fiscal_period_year("H1 2024") == ("H1", 2024)
    assert html_statement_module._extract_fiscal_period_year("2024 2H") == ("H2", 2024)
    assert html_statement_module._extract_fiscal_period_year("9M 2024") == ("Q3", 2024)
    assert html_statement_module._extract_fiscal_period_year("6M 24") == ("H1", 2024)
    assert html_statement_module._extract_fiscal_period_year("FY70") == ("FY", 1970)
    assert html_statement_module._extract_fiscal_period_year("Fourth Quarter 2024") == ("Q4", 2024)
    assert html_statement_module._extract_fiscal_period_year("") is None
    assert html_statement_module._extract_fiscal_period_label("FY24") == "FY"
    assert html_statement_module._normalize_year_token("69") == 2069
    assert html_statement_module._normalize_year_token("70") == 1970
    assert html_statement_module._resolve_period_end_from_fiscal_period(
        fiscal_period="Q3",
        fiscal_year=2024,
    ) == html_statement_module._build_safe_date(year=2024, month=9, day=30)
    assert html_statement_module._resolve_period_end_from_fiscal_period(
        fiscal_period="H2",
        fiscal_year=2024,
    ) == html_statement_module._build_safe_date(year=2024, month=12, day=31)
    assert html_statement_module._resolve_period_end_from_fiscal_period(
        fiscal_period="OTHER",
        fiscal_year=2024,
    ) is None


@pytest.mark.unit
def test_statement_currency_scale_and_row_helpers_cover_remaining_branches() -> None:
    """验证货币、单位、行标签与数值清洗 helper 的剩余分支。"""

    assert html_statement_module._extract_currency_for_column(
        scope_text="Amounts in US$",
        column_header_text="",
    ) == "US$"
    assert html_statement_module._extract_currency_for_column(
        scope_text="Reported in RMB",
        column_header_text="",
    ) == "RMB"
    assert html_statement_module._extract_currency_for_column(
        scope_text="Amounts in EUR",
        column_header_text="",
    ) == "EUR"
    assert html_statement_module._extract_currency_for_column(
        scope_text="Reported in GBP",
        column_header_text="",
    ) == "GBP"
    assert html_statement_module._extract_currency_for_column(
        scope_text="Dollar presentation",
        column_header_text="$",
    ) == "$"
    assert html_statement_module._extract_currency_for_column(
        scope_text="No currency",
        column_header_text="",
    ) is None
    assert html_statement_module._map_currency_code("HK$") == "HKD"
    assert html_statement_module._map_currency_code("CAD") == "CAD"
    assert html_statement_module._map_currency_code(None) is None
    assert html_statement_module._build_units_label(primary_currency_raw=None, scale="millions") == "millions"
    assert html_statement_module._build_units_label(primary_currency_raw="US$", scale=None) == "US$"
    assert html_statement_module._build_units_label(primary_currency_raw=None, scale=None) is None
    assert html_statement_module._infer_scale_from_caption("Condensed statements (in millions)") == "millions"
    assert html_statement_module._infer_scale_from_caption("No scale") is None
    assert html_statement_module._extract_row_label(["nan", "123", "Revenue", "456"]) == "Revenue"
    assert html_statement_module._extract_row_label(["", "100", "200"]) is None
    assert html_statement_module.normalize_numeric_separators("1.234,56") == "1234.56"
    assert html_statement_module.normalize_numeric_separators("1,23") == "1.23"
    assert html_statement_module.normalize_numeric_separators("1,234") == "1234"
    assert html_statement_module._count_pattern_hits(
        statement_patterns=(re.compile(r"revenue"), re.compile(r"income")),
        text="revenue operating income",
    ) == 2


@pytest.mark.unit
def test_parse_statement_table_dedupes_duplicate_period_columns_and_skips_empty_payload() -> None:
    """验证单表解析会去重重复期间列，并在无有效数据时返回空。"""

    duplicate_period_table = _StubTable(
        tag=object(),
        caption="Condensed Balance Sheets (in millions of US$)",
    )
    duplicate_period_dataframe = pd.DataFrame(
        [
            ["As at", "September 30, 2024", "September 30, 2024"],
            ["Cash and cash equivalents", "100", "100"],
            ["Total assets", "300", "300"],
        ]
    )
    duplicate_period_result = html_statement_module._parse_statement_table(
        table=duplicate_period_table,
        parse_table_dataframe=_build_parse_table_dataframe({duplicate_period_table: duplicate_period_dataframe}),
    )

    assert duplicate_period_result is not None
    assert len(duplicate_period_result.periods) == 1
    assert duplicate_period_result.periods[0].period_end == "2024-09-30"
    assert duplicate_period_result.currency_raw == "US$"
    assert duplicate_period_result.scale == "millions"
    assert duplicate_period_result.rows == [
        {"concept": "", "label": "Cash and cash equivalents", "values": [100.0]},
        {"concept": "", "label": "Total assets", "values": [300.0]},
    ]

    empty_payload_table = _StubTable(tag=object(), caption="Header only")
    empty_payload_dataframe = pd.DataFrame([["Only header"]])
    assert html_statement_module._parse_statement_table(
        table=empty_payload_table,
        parse_table_dataframe=_build_parse_table_dataframe({empty_payload_table: empty_payload_dataframe}),
    ) is None


@pytest.mark.unit
def test_parse_statement_table_retries_deeper_header_rows_and_textual_quarter_headers() -> None:
    """验证共享层会继续向后探测表头，并支持文本季度标题。"""

    table = _StubTable(
        tag=object(),
        caption="Consolidated Statements of Comprehensive Income",
    )
    dataframe = pd.DataFrame(
        [
            ["COMPANY NAME", "COMPANY NAME", "", "", "", ""],
            ["CONSOLIDATED INCOME STATEMENTS", "CONSOLIDATED INCOME STATEMENTS", "", "", "", ""],
            ["", "", "", "", "", ""],
            ["", "Fourth Quarter", "Fourth Quarter", "Second Quarter", "Second Quarter", ""],
            ["", "2024", "2025", "2024", "2025", ""],
            ["", "KRW", "KRW", "KRW", "KRW", ""],
            ["", "Unaudited", "Unaudited", "Unaudited", "Unaudited", ""],
            ["Revenue", "100", "120", "90", "110", ""],
            ["Operating income", "30", "35", "20", "28", ""],
            ["Net income", "25", "27", "18", "22", ""],
        ]
    )

    result = html_statement_module.build_html_statement_result_from_tables(
        statement_type="income",
        tables=[table],
        parse_table_dataframe=_build_parse_table_dataframe({table: dataframe}),
    )

    assert result is not None
    assert result["periods"] == [
        {"period_end": "2024-12-31", "fiscal_year": 2024, "fiscal_period": "Q4"},
        {"period_end": "2025-12-31", "fiscal_year": 2025, "fiscal_period": "Q4"},
        {"period_end": "2024-06-30", "fiscal_year": 2024, "fiscal_period": "Q2"},
        {"period_end": "2025-06-30", "fiscal_year": 2025, "fiscal_period": "Q2"},
    ]
    assert result["rows"] == [
        {"concept": "", "label": "Revenue", "values": [100.0, 120.0, 90.0, 110.0]},
        {"concept": "", "label": "Operating income", "values": [30.0, 35.0, 20.0, 28.0]},
        {"concept": "", "label": "Net income", "values": [25.0, 27.0, 18.0, 22.0]},
    ]


@pytest.mark.unit
def test_parse_statement_table_supports_hyphenated_day_month_year_headers() -> None:
    """验证共享层支持 `31-Dec-24` 这类带连字符的日期表头。"""

    table = _StubTable(
        tag=object(),
        caption="Consolidated Statements of Financial Position (In millions of KRW and thousands of US$)",
    )
    dataframe = pd.DataFrame(
        [
            ["", "As of", "As of"],
            ["", "31-Dec-24", "31-Mar-25"],
            ["", "(KRW)", "(KRW)"],
            ["Cash and cash equivalents", "228,898", "201,367"],
            ["Total assets", "300,000", "310,500"],
            ["Total liabilities", "120,000", "118,400"],
        ]
    )

    result = html_statement_module.build_html_statement_result_from_tables(
        statement_type="balance_sheet",
        tables=[table],
        parse_table_dataframe=_build_parse_table_dataframe({table: dataframe}),
    )

    assert result is not None
    assert result["periods"] == [
        {"period_end": "2024-12-31", "fiscal_year": 2024, "fiscal_period": "FY"},
        {"period_end": "2025-03-31", "fiscal_year": 2025, "fiscal_period": "Q1"},
    ]
    assert result["rows"] == [
        {"concept": "", "label": "Cash and cash equivalents", "values": [228898.0, 201367.0]},
        {"concept": "", "label": "Total assets", "values": [300000.0, 310500.0]},
        {"concept": "", "label": "Total liabilities", "values": [120000.0, 118400.0]},
    ]


@pytest.mark.unit
def test_parse_statement_table_supports_single_period_summary_value_column_from_scope_text() -> None:
    """验证共享层可从 scope text 为单期间摘要表补出期间列。"""

    table = _StubTable(
        tag=object(),
        caption="Relevant information of the condensed consolidated interim financial statements follows: Net profit for the period",
        context_before="For the three-month period ended March 31, 2025, relevant information follows.",
    )
    dataframe = pd.DataFrame(
        [
            ["Attributable to shareholders of the parent company", "", "", "(19,864)", ""],
            ["Attributable to non-controlling interest", "", "", "6,894", ""],
            ["Total net profit for the period", "", "", "(12,970)", ""],
        ]
    )

    result = html_statement_module.build_html_statement_result_from_tables(
        statement_type="income",
        tables=[table],
        parse_table_dataframe=_build_parse_table_dataframe({table: dataframe}),
    )

    assert result is not None
    assert result["periods"] == [
        {"period_end": "2025-03-31", "fiscal_year": 2025, "fiscal_period": "Q1"}
    ]
    assert any(row["label"] == "Attributable to non-controlling interest" for row in result["rows"])
    assert any(row["label"] == "Total net profit for the period" for row in result["rows"])
