"""HTML 财务报表结构化共享能力。

本模块承载与具体表单类型无关的 HTML 财务表结构化核心能力，供
`6-K` 当前接入，并为后续 `10-K/10-Q/20-F` 的 HTML fallback 预留复用入口。

边界约束：
- 本模块不负责报表类型分类规则，不持有任何 `6-K`/`10-K`/`20-F` 专属模式。
- 本模块不理解虚拟章节、表格重映射或表单路由，只处理“给定候选表格后如何结构化”。
- 所有控制参数（如 line item patterns、min_hits、parse callback）都通过显式参数注入。
"""

from __future__ import annotations

import calendar
from collections.abc import Callable
from dataclasses import dataclass
import datetime as dt
import re
from typing import Any, Optional
import unicodedata

import pandas as pd

from dayu.engine.processors.text_utils import normalize_optional_string as _normalize_optional_string
from dayu.engine.processors.text_utils import normalize_whitespace as _normalize_whitespace

from .financial_base import FinancialStatementResult
from .sec_xbrl_query import build_statement_locator

_CURRENCY_MAP = {
    "HK$": "HKD",
    "US$": "USD",
    "RMB": "CNY",
    "CNY": "CNY",
    "$": "USD",
}
_DEFAULT_HEADER_ROW_COUNT = 3
_MAX_HEADER_SCAN_ROWS = 8
_HEADER_WINDOW_LOOKBACK_COLUMNS = 2
_MIN_NUMERIC_ROWS_PER_VALUE_COLUMN = 2
_MONTH_TOKEN_TO_NUMBER: dict[str, int] = {
    "jan": 1,
    "january": 1,
    "ene": 1,
    "enero": 1,
    "janvier": 1,
    "fev": 2,
    "feb": 2,
    "february": 2,
    "febrero": 2,
    "mar": 3,
    "march": 3,
    "marzo": 3,
    "apr": 4,
    "april": 4,
    "abr": 4,
    "abril": 4,
    "may": 5,
    "mayo": 5,
    "mai": 5,
    "jun": 6,
    "june": 6,
    "junio": 6,
    "jul": 7,
    "july": 7,
    "julio": 7,
    "aug": 8,
    "august": 8,
    "ago": 8,
    "agosto": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "set": 9,
    "setembro": 9,
    "septiembre": 9,
    "oct": 10,
    "october": 10,
    "out": 10,
    "outubro": 10,
    "nov": 11,
    "november": 11,
    "dez": 12,
    "dec": 12,
    "december": 12,
    "dic": 12,
    "diciembre": 12,
}

_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|2100)\b")
_QUARTER_TOKEN_RE = re.compile(
    r"(?i)\b(?:Q([1-4])|([1-4])Q)\s*[-/']?\s*(?:fy)?\s*(\d{2,4})\b"
)
_YEAR_FIRST_QUARTER_TOKEN_RE = re.compile(
    r"(?i)\b(\d{2,4})\s*(?:Q([1-4])|([1-4])Q)\b"
)
_HALF_YEAR_TOKEN_RE = re.compile(
    r"(?i)\b(?:H([12])|([12])H)\s*[-/']?\s*(\d{2,4})\b"
)
_YEAR_FIRST_HALF_YEAR_TOKEN_RE = re.compile(
    r"(?i)\b(\d{2,4})\s*(?:H([12])|([12])H)\b"
)
_NINE_MONTH_TOKEN_RE = re.compile(r"(?i)\b9M\s*[-/']?\s*(\d{2,4})\b")
_SIX_MONTH_TOKEN_RE = re.compile(r"(?i)\b6M\s*[-/']?\s*(\d{2,4})\b")
_FY_TOKEN_RE = re.compile(r"(?i)\b(?:FY|FYE)\s*[-/']?\s*(\d{2,4})\b")
_TEXTUAL_QUARTER_YEAR_RE = re.compile(
    r"(?i)\b(first|second|third|fourth)\s+quarter\s+(\d{2,4})\b"
)
_MONTH_YEAR_RE = re.compile(
    r"(?i)\b([A-Za-zÀ-ÿ]{3,12})\.?\s+(19\d{2}|20\d{2}|2100)\b"
)
_TEXTUAL_DATE_WITH_DASH_RE = re.compile(
    r"(?i)\b(\d{1,2})[-\s]([A-Za-zÀ-ÿ]{3,12})[-\s](19\d{2}|20\d{2}|2100|\d{2})\b"
)
_FUSED_PERIOD_MONTH_RE = re.compile(
    r"(?i)\b(ended|ending|as\s+of|as\s+at)(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b"
)
_NUMERIC_PREFIX_RE = re.compile(
    r"(?i)^(?:us\$|hk\$|nt\$|r\$|ps\.?|cop|usd|eur|gbp|cny|rmb|ars|brl|mxn|chf|jpy|krw)\s*"
)
_NUMERIC_SUFFIX_RE = re.compile(
    r"(?i)\s*(?:%|x|times|bps|pts?|points?)$"
)


@dataclass(frozen=True)
class _StatementTablePeriod:
    """HTML 表格期间列信息。"""

    column_index: int
    period_end: str
    fiscal_period: Optional[str]
    currency_raw: Optional[str]


@dataclass(frozen=True)
class _ParsedStatementTable:
    """单表结构化解析结果。"""

    periods: list[_StatementTablePeriod]
    rows: list[dict[str, Any]]
    currency_raw: Optional[str]
    scale: Optional[str]


def build_html_statement_result_from_tables(
    *,
    statement_type: str,
    tables: list[Any],
    parse_table_dataframe: Callable[[Any], Optional[pd.DataFrame]],
) -> Optional[FinancialStatementResult]:
    """从 HTML 候选表格构建结构化财务报表结果。

    Args:
        statement_type: 报表类型。
        tables: 候选表格列表。
        parse_table_dataframe: 表格对象到 DataFrame 的解析函数。

    Returns:
        结构化财务报表结果；低置信时返回 `None`。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    parsed_tables: list[_ParsedStatementTable] = []
    for table in tables:
        parsed_table = _parse_statement_table(
            table=table,
            parse_table_dataframe=parse_table_dataframe,
        )
        if parsed_table is None:
            continue
        parsed_tables.append(parsed_table)
    if not parsed_tables:
        return None

    primary_currency_raw = _select_primary_currency_raw(parsed_tables)
    grouped_payloads = _group_statement_rows_by_period_signature(
        parsed_tables=parsed_tables,
        primary_currency_raw=primary_currency_raw,
    )
    if not grouped_payloads:
        return None
    _, selected_payload = max(
        grouped_payloads.items(),
        key=lambda item: (
            len(item[1]["rows"]),
            int(item[1]["table_count"]),
            len(item[1]["periods"]),
        ),
    )

    selected_periods = list(selected_payload["periods"])
    rows = _dedupe_statement_rows(list(selected_payload["rows"]))
    if not selected_periods or not rows:
        return None

    period_summaries = [_build_statement_period_summary(period) for period in selected_periods]
    first_scale = parsed_tables[0].scale if parsed_tables else None
    return {
        "statement_type": statement_type,
        "periods": period_summaries,
        "rows": rows,
        "currency": _map_currency_code(primary_currency_raw),
        "units": _build_units_label(
            primary_currency_raw=primary_currency_raw,
            scale=first_scale,
        ),
        "scale": first_scale,
        "data_quality": "extracted",
        "statement_locator": build_statement_locator(
            statement_type=statement_type,
            periods=period_summaries,
            rows=rows,
        ),
    }


def select_html_statement_tables_by_row_signals(
    *,
    tables: list[Any],
    line_item_patterns: tuple[re.Pattern[str], ...],
    min_hits: int,
    min_row_count: int = 6,
    parse_table_dataframe: Callable[[Any], Optional[pd.DataFrame]],
) -> list[Any]:
    """按行标签语义信号筛选候选报表表格。

    Args:
        tables: 原始候选表格列表。
        line_item_patterns: 行标签关键词模式。
        min_hits: 最低命中数阈值。
        min_row_count: 最低有效行数阈值。
        parse_table_dataframe: 表格对象到 DataFrame 的解析函数。

    Returns:
        按置信度排序的候选表格列表。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    if not line_item_patterns:
        return []

    ranked_tables: list[tuple[int, int, Any]] = []
    for table in tables:
        parsed = _parse_statement_table(
            table=table,
            parse_table_dataframe=parse_table_dataframe,
        )
        if parsed is None:
            continue
        if len(parsed.rows) < min_row_count:
            continue
        labels_text = _normalize_match_text(
            " ".join(str(row.get("label", "")) for row in parsed.rows)
        )
        hit_count = _count_pattern_hits(
            statement_patterns=line_item_patterns,
            text=labels_text,
        )
        if hit_count < min_hits:
            continue
        ranked_tables.append((hit_count, len(parsed.rows), table))
    ranked_tables.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [table for _, _, table in ranked_tables]


def _parse_statement_table(
    *,
    table: Any,
    parse_table_dataframe: Callable[[Any], Optional[pd.DataFrame]],
) -> Optional[_ParsedStatementTable]:
    """解析单个 HTML 财务表为中间结构。

    Args:
        table: 内部表格对象。
        parse_table_dataframe: 表格对象到 DataFrame 的解析函数。

    Returns:
        解析结果；低置信时返回 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    dataframe = parse_table_dataframe(table)
    if dataframe is None or dataframe.empty:
        return None

    matrix = _dataframe_to_matrix(dataframe)
    if len(matrix) <= 1:
        return None

    parsed_candidates: list[_ParsedStatementTable] = []
    for header_row_count in _candidate_header_row_counts(matrix):
        parsed_candidate = _parse_statement_table_with_header_rows(
            matrix=matrix,
            header_row_count=header_row_count,
            caption=getattr(table, "caption", None),
            context_before=getattr(table, "context_before", None),
        )
        if parsed_candidate is not None:
            parsed_candidates.append(parsed_candidate)
    if not parsed_candidates:
        return None
    return max(
        parsed_candidates,
        key=lambda item: (len(item.periods), len(item.rows)),
    )


def _candidate_header_row_counts(matrix: list[list[str]]) -> list[int]:
    """生成单表解析要尝试的表头行数候选。

    默认先使用当前启发式推断值；若该值解析不到期间列，再继续向后探测
    更深的表头，覆盖 `TME` 一类前几行为空、真实期间行靠后的结果表。

    Args:
        matrix: 表格矩阵。

    Returns:
        去重后的候选表头行数列表。

    Raises:
        RuntimeError: 生成失败时抛出。
    """

    inferred_count = _infer_header_row_count(matrix)
    candidates: list[int] = []
    max_candidate = min(len(matrix) - 1, _MAX_HEADER_SCAN_ROWS)
    for value in range(inferred_count, max_candidate + 1):
        if value >= 1 and value not in candidates:
            candidates.append(value)
    if inferred_count not in candidates and inferred_count >= 1:
        candidates.insert(0, inferred_count)
    return candidates


def _parse_statement_table_with_header_rows(
    *,
    matrix: list[list[str]],
    header_row_count: int,
    caption: Optional[str],
    context_before: Optional[str],
) -> Optional[_ParsedStatementTable]:
    """按指定表头行数解析单个 HTML 财务表。

    Args:
        matrix: 表格矩阵。
        header_row_count: 当前尝试的表头行数。
        caption: 表格标题。
        context_before: 表格前的局部上下文。

    Returns:
        解析结果；低置信时返回 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    if len(matrix) <= header_row_count:
        return None

    statement_scope_text = _build_statement_scope_text(
        matrix=matrix,
        header_row_count=header_row_count,
        caption=caption,
        context_before=context_before,
    )
    value_column_indexes = _detect_value_column_indexes(
        matrix,
        header_row_count=header_row_count,
    )
    if not value_column_indexes:
        value_column_indexes = _detect_single_period_value_column_indexes(
            matrix=matrix,
            header_row_count=header_row_count,
            statement_scope_text=statement_scope_text,
        )
    if not value_column_indexes:
        return None

    periods: list[_StatementTablePeriod] = []
    seen_period_keys: set[tuple[str, Optional[str], Optional[str]]] = set()
    for column_index in value_column_indexes:
        period = _build_period_for_column(
            matrix=matrix,
            column_index=column_index,
            header_row_count=header_row_count,
            statement_scope_text=statement_scope_text,
        )
        if period is None:
            continue
        period_key = (period.period_end, period.fiscal_period, period.currency_raw)
        if period_key in seen_period_keys:
            continue
        seen_period_keys.add(period_key)
        periods.append(period)
    if not periods:
        single_period = _build_single_scope_period(
            value_column_indexes=value_column_indexes,
            statement_scope_text=statement_scope_text,
        )
        if single_period is None:
            return None
        periods = [single_period]

    rows = _build_rows_from_matrix(
        matrix=matrix,
        periods=periods,
        header_row_count=header_row_count,
    )
    if not rows:
        return None

    primary_currency_raw = periods[0].currency_raw
    return _ParsedStatementTable(
        periods=periods,
        rows=rows,
        currency_raw=primary_currency_raw,
        scale=_infer_scale_from_caption(caption),
    )


def _dataframe_to_matrix(dataframe: pd.DataFrame) -> list[list[str]]:
    """将 DataFrame 归一为字符串矩阵。

    Args:
        dataframe: 表格 DataFrame。

    Returns:
        字符串矩阵。

    Raises:
        RuntimeError: 转换失败时抛出。
    """

    matrix: list[list[str]] = []
    for _, row in dataframe.iterrows():
        current_row: list[str] = []
        for value in row.tolist():
            if pd.isna(value):
                current_row.append("")
                continue
            normalized = _normalize_optional_string(value)
            current_row.append(normalized or "")
        matrix.append(current_row)
    return matrix


def _infer_header_row_count(matrix: list[list[str]]) -> int:
    """推断表头行数。

    Args:
        matrix: 表格矩阵。

    Returns:
        估计的表头行数（至少 1）。

    Raises:
        RuntimeError: 推断失败时抛出。
    """

    scan_limit = min(len(matrix), _MAX_HEADER_SCAN_ROWS)
    for row_index in range(scan_limit):
        row = matrix[row_index]
        if _is_likely_data_row(row):
            return max(1, row_index)
    return min(_DEFAULT_HEADER_ROW_COUNT, max(1, len(matrix) - 1))


def _is_likely_data_row(row: list[str]) -> bool:
    """判断某行是否更像数据行而非表头行。

    Args:
        row: 表格行。

    Returns:
        数据行返回 `True`，否则返回 `False`。

    Raises:
        RuntimeError: 判定失败时抛出。
    """

    label = _extract_row_label(row)
    if label is None:
        return False
    normalized_label = _normalize_free_text(label)
    if any(token in normalized_label for token in ("as of", "as at", "ended", "unaudited", "note")):
        return False

    numeric_values: list[float] = []
    for cell in row[1:]:
        parsed_value = _parse_optional_numeric(cell)
        if parsed_value is not None:
            numeric_values.append(parsed_value)
    if not numeric_values:
        return False
    if all(_is_year_like_numeric(value) for value in numeric_values):
        return False
    return True


def _is_year_like_numeric(value: float) -> bool:
    """判断数值是否看起来像年份。

    Args:
        value: 数值。

    Returns:
        像年份时返回 `True`。

    Raises:
        RuntimeError: 判定失败时抛出。
    """

    rounded = int(value)
    return rounded == value and 1900 <= rounded <= 2100


def _build_statement_scope_text(
    *,
    matrix: list[list[str]],
    header_row_count: int,
    caption: Optional[str],
    context_before: Optional[str],
) -> str:
    """构建表格级范围文本。

    Args:
        matrix: 表格矩阵。
        header_row_count: 表头行数。
        caption: 表格标题。
        context_before: 表格前的局部上下文。

    Returns:
        组合后的范围文本。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    header_rows = matrix[:header_row_count]
    header_text = " ".join(_normalize_whitespace(" ".join(row)) for row in header_rows)
    # 只取紧邻表格前的局部上下文，避免把整篇叙述噪声引进期间解析。
    normalized_context = _normalize_whitespace(context_before or "")
    compact_context = normalized_context[-240:] if normalized_context else ""
    return _normalize_whitespace(f"{compact_context} {caption or ''} {header_text}")


def _detect_value_column_indexes(
    matrix: list[list[str]],
    *,
    header_row_count: int,
) -> list[int]:
    """识别包含数值的候选列。

    Args:
        matrix: 表格矩阵。
        header_row_count: 表头行数。

    Returns:
        数值列下标列表。

    Raises:
        RuntimeError: 检测失败时抛出。
    """

    data_rows = matrix[header_row_count:]
    if not data_rows:
        return []
    value_columns: list[int] = []
    max_col_count = max(len(row) for row in matrix)
    for column_index in range(1, max_col_count):
        numeric_count = 0
        for row in data_rows:
            cell_value = row[column_index] if column_index < len(row) else ""
            if _parse_optional_numeric(cell_value) is not None:
                numeric_count += 1
        if numeric_count >= _MIN_NUMERIC_ROWS_PER_VALUE_COLUMN:
            value_columns.append(column_index)
    if value_columns:
        return value_columns

    if not _has_period_hint_in_headers(matrix=matrix, header_row_count=header_row_count):
        return []
    for column_index in range(1, max_col_count):
        for row in data_rows:
            cell_value = row[column_index] if column_index < len(row) else ""
            if _parse_optional_numeric(cell_value) is not None:
                value_columns.append(column_index)
                break
    return value_columns


def _detect_single_period_value_column_indexes(
    *,
    matrix: list[list[str]],
    header_row_count: int,
    statement_scope_text: str,
) -> list[int]:
    """为单期间摘要表探测唯一数值列。

    某些 `6-K` 本地交易所摘要函只给出单期间、单数值列的净利润/权益摘要，
    表头不再重复写日期，导致常规“多期间表头 + 至少两行数值”启发式无法命中。
    当前回退只在 `scope_text` 已能解析出稳定期间时启用，并且要求整张表只有一列
    含数值，避免把普通说明表误当成财务表。

    Args:
        matrix: 表格矩阵。
        header_row_count: 表头行数。
        statement_scope_text: 表级范围文本。

    Returns:
        唯一数值列下标；不满足条件时返回空列表。

    Raises:
        RuntimeError: 探测失败时抛出。
    """

    if _normalize_period_end(scope_text=statement_scope_text, date_text="") is None:
        return []

    data_rows = matrix[header_row_count:]
    if not data_rows:
        return []

    numeric_columns: list[int] = []
    max_col_count = max(len(row) for row in matrix)
    for column_index in range(1, max_col_count):
        numeric_count = 0
        for row in data_rows:
            cell_value = row[column_index] if column_index < len(row) else ""
            if _parse_optional_numeric(cell_value) is not None:
                numeric_count += 1
        if numeric_count > 0:
            numeric_columns.append(column_index)
    if len(numeric_columns) != 1:
        return []
    return numeric_columns


def _has_period_hint_in_headers(
    *,
    matrix: list[list[str]],
    header_row_count: int,
) -> bool:
    """判断表头是否存在期间信号。

    Args:
        matrix: 表格矩阵。
        header_row_count: 表头行数。

    Returns:
        若表头包含可解析日期或财期 token 则返回 `True`。

    Raises:
        RuntimeError: 判定失败时抛出。
    """

    header_text = _normalize_whitespace(" ".join(" ".join(row) for row in matrix[:header_row_count]))
    if not header_text:
        return False
    if _extract_first_date(header_text) is not None:
        return True
    return _extract_fiscal_period_year(header_text) is not None


def _build_period_for_column(
    *,
    matrix: list[list[str]],
    column_index: int,
    header_row_count: int,
    statement_scope_text: str,
) -> Optional[_StatementTablePeriod]:
    """为某个数值列构建期间元信息。

    Args:
        matrix: 表格矩阵。
        column_index: 列下标。
        header_row_count: 表头行数。
        statement_scope_text: 表格级范围文本。

    Returns:
        期间列信息；无法识别时返回 `None`。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    header_rows = matrix[:header_row_count]
    column_header_text = _collect_column_header_text(
        header_rows=header_rows,
        column_index=column_index,
    )
    period_end = _normalize_period_end(
        scope_text=statement_scope_text,
        date_text=column_header_text,
    )
    if period_end is None:
        return None
    explicit_fiscal_period = _extract_fiscal_period_label(column_header_text)
    if explicit_fiscal_period is None:
        explicit_fiscal_period = _extract_fiscal_period_label(statement_scope_text)
    return _StatementTablePeriod(
        column_index=column_index,
        period_end=period_end,
        fiscal_period=explicit_fiscal_period
        or _infer_fiscal_period(scope_text=statement_scope_text, period_end=period_end),
        currency_raw=_extract_currency_for_column(
            scope_text=statement_scope_text,
            column_header_text=column_header_text,
        ),
    )


def _build_single_scope_period(
    *,
    value_column_indexes: list[int],
    statement_scope_text: str,
) -> Optional[_StatementTablePeriod]:
    """基于范围文本为单期间摘要表构建期间信息。

    Args:
        value_column_indexes: 已识别到的唯一数值列。
        statement_scope_text: 表级范围文本。

    Returns:
        单期间信息；无法解析时返回 `None`。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    if len(value_column_indexes) != 1:
        return None
    period_end = _normalize_period_end(scope_text=statement_scope_text, date_text="")
    if period_end is None:
        return None
    fiscal_period = _extract_fiscal_period_label(statement_scope_text)
    if fiscal_period is None:
        fiscal_period = _infer_fiscal_period(scope_text=statement_scope_text, period_end=period_end)
    return _StatementTablePeriod(
        column_index=value_column_indexes[0],
        period_end=period_end,
        fiscal_period=fiscal_period,
        currency_raw=_extract_currency_for_column(
            scope_text=statement_scope_text,
            column_header_text="",
        ),
    )


def _build_rows_from_matrix(
    *,
    matrix: list[list[str]],
    periods: list[_StatementTablePeriod],
    header_row_count: int,
) -> list[dict[str, Any]]:
    """从矩阵提取报表行。

    Args:
        matrix: 表格矩阵。
        periods: 期间列列表。
        header_row_count: 表头行数。

    Returns:
        报表行列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    rows: list[dict[str, Any]] = []
    for row in matrix[header_row_count:]:
        label = _extract_row_label(row)
        if label is None:
            continue
        values = [
            _parse_optional_numeric(row[period.column_index] if period.column_index < len(row) else "")
            for period in periods
        ]
        if not any(value is not None for value in values):
            continue
        rows.append({
            "concept": "",
            "label": label,
            "values": values,
        })
    return rows


def _select_primary_currency_raw(parsed_tables: list[_ParsedStatementTable]) -> Optional[str]:
    """选择报表的主货币列。

    Args:
        parsed_tables: 已解析表格列表。

    Returns:
        主货币文本；不存在时返回 `None`。

    Raises:
        RuntimeError: 选择失败时抛出。
    """

    counts: dict[str, int] = {}
    for parsed_table in parsed_tables:
        for period in parsed_table.periods:
            if period.currency_raw is None:
                continue
            counts[period.currency_raw] = counts.get(period.currency_raw, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _group_statement_rows_by_period_signature(
    *,
    parsed_tables: list[_ParsedStatementTable],
    primary_currency_raw: Optional[str],
) -> dict[tuple[tuple[str, Optional[str]], ...], dict[str, Any]]:
    """按期间签名聚合可用报表行。

    Args:
        parsed_tables: 已解析表格列表。
        primary_currency_raw: 主货币文本。

    Returns:
        `signature -> payload` 的映射。

    Raises:
        RuntimeError: 聚合失败时抛出。
    """

    grouped: dict[tuple[tuple[str, Optional[str]], ...], dict[str, Any]] = {}
    for parsed_table in parsed_tables:
        matching_periods = [
            period
            for period in parsed_table.periods
            if primary_currency_raw is None or period.currency_raw == primary_currency_raw
        ]
        if not matching_periods:
            matching_periods = list(parsed_table.periods)
        if not matching_periods:
            continue
        signature = tuple((period.period_end, period.fiscal_period) for period in matching_periods)
        selected_rows = _select_row_values(
            rows_payload=parsed_table.rows,
            source_periods=parsed_table.periods,
            target_periods=matching_periods,
        )
        if not selected_rows:
            continue
        payload = grouped.setdefault(
            signature,
            {"periods": matching_periods, "rows": [], "table_count": 0},
        )
        payload["rows"].extend(selected_rows)
        payload["table_count"] = int(payload["table_count"]) + 1
    return grouped


def _dedupe_statement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """对报表行按标签和值去重。

    Args:
        rows: 原始报表行列表。

    Returns:
        去重后的报表行列表。

    Raises:
        RuntimeError: 去重失败时抛出。
    """

    deduped_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[Optional[float], ...]]] = set()
    for row in rows:
        label = _normalize_whitespace(str(row.get("label", "")))
        values = row.get("values", [])
        if not isinstance(values, list):
            continue
        normalized_values = tuple(
            value if isinstance(value, (int, float)) else None
            for value in values
        )
        key = (label, normalized_values)
        if not label or key in seen:
            continue
        seen.add(key)
        deduped_rows.append(
            {
                "concept": row.get("concept", ""),
                "label": label,
                "values": list(normalized_values),
            }
        )
    return deduped_rows


def _select_row_values(
    *,
    rows_payload: list[dict[str, Any]],
    source_periods: list[_StatementTablePeriod],
    target_periods: list[_StatementTablePeriod],
) -> list[dict[str, Any]]:
    """按目标期间列裁剪报表行值。

    Args:
        rows_payload: 原始行载荷。
        source_periods: 原始期间列。
        target_periods: 目标期间列。

    Returns:
        裁剪后的报表行列表。

    Raises:
        RuntimeError: 裁剪失败时抛出。
    """

    selected_indexes = [
        index
        for index, period in enumerate(source_periods)
        if any(
            period.period_end == target.period_end and period.fiscal_period == target.fiscal_period
            for target in target_periods
        )
    ]
    selected_rows: list[dict[str, Any]] = []
    for row_payload in rows_payload:
        values = row_payload.get("values")
        if not isinstance(values, list):
            continue
        clipped_values = [values[index] for index in selected_indexes if index < len(values)]
        if not any(value is not None for value in clipped_values):
            continue
        selected_rows.append(
            {
                "concept": row_payload.get("concept", ""),
                "label": row_payload.get("label", ""),
                "values": clipped_values,
            }
        )
    return selected_rows


def _build_statement_period_summary(period: _StatementTablePeriod) -> dict[str, Any]:
    """构建标准期间摘要。

    Args:
        period: 期间列信息。

    Returns:
        期间摘要字典。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    fiscal_year = int(period.period_end[:4]) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period.period_end) else None
    return {
        "period_end": period.period_end,
        "fiscal_year": fiscal_year,
        "fiscal_period": period.fiscal_period,
    }


def _collect_column_header_text(
    *,
    header_rows: list[list[str]],
    column_index: int,
) -> str:
    """采集某列在多行表头中的文本窗口。

    Args:
        header_rows: 表头行列表。
        column_index: 目标列。

    Returns:
        合并后的列头文本。

    Raises:
        RuntimeError: 采集失败时抛出。
    """

    parts: list[str] = []
    for row in header_rows:
        row_text = _collect_header_window_text(row=row, column_index=column_index)
        if row_text and row_text not in parts:
            parts.append(row_text)
    return _normalize_whitespace(" ".join(parts))


def _collect_header_window_text(
    *,
    row: list[str],
    column_index: int,
) -> str:
    """采集某列附近的单行表头文本窗口。

    Args:
        row: 表头行。
        column_index: 目标列。

    Returns:
        合并后的文本片段。

    Raises:
        RuntimeError: 采集失败时抛出。
    """

    if column_index < len(row):
        current_value = _normalize_whitespace(row[column_index])
        if current_value:
            return current_value

    start_index = max(0, column_index - _HEADER_WINDOW_LOOKBACK_COLUMNS)
    for current_index in range(column_index - 1, start_index - 1, -1):
        if current_index >= len(row):
            continue
        candidate = _normalize_whitespace(row[current_index])
        if candidate:
            return candidate
    return ""


def _normalize_period_end(
    *,
    scope_text: str,
    date_text: str,
) -> Optional[str]:
    """标准化期间结束日期。

    Args:
        scope_text: 范围文本。
        date_text: 日期文本。

    Returns:
        `YYYY-MM-DD` 期间结束日期；无法识别时返回 `None`。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized_scope = _normalize_whitespace(scope_text)
    normalized_date = _normalize_whitespace(date_text)
    if not normalized_scope and not normalized_date:
        return None

    candidate_texts = (
        normalized_date,
        f"{normalized_date} {normalized_scope}",
        f"{normalized_scope} {normalized_date}",
    )
    for candidate_text in candidate_texts:
        parsed = _extract_first_date(candidate_text)
        if parsed is not None:
            return parsed.isoformat()
    for candidate_text in candidate_texts:
        fiscal_period_year = _extract_fiscal_period_year(candidate_text)
        if fiscal_period_year is None:
            continue
        fiscal_period, fiscal_year = fiscal_period_year
        fiscal_period_end = _resolve_period_end_from_fiscal_period(
            fiscal_period=fiscal_period,
            fiscal_year=fiscal_year,
        )
        if fiscal_period_end is not None:
            return fiscal_period_end.isoformat()

    scoped_month_day = _extract_scope_month_day(normalized_scope)
    years = _extract_years(f"{normalized_date} {normalized_scope}")
    if years:
        target_year = years[0]
        if scoped_month_day is not None:
            month, day = scoped_month_day
            candidate = _build_safe_date(year=target_year, month=month, day=day)
            if candidate is not None:
                return candidate.isoformat()
            fallback_day = calendar.monthrange(target_year, month)[1]
            fallback_date = _build_safe_date(year=target_year, month=month, day=fallback_day)
            if fallback_date is not None:
                return fallback_date.isoformat()
        if "as of" in _normalize_free_text(normalized_scope) or "as at" in _normalize_free_text(normalized_scope):
            return dt.date(target_year, 12, 31).isoformat()
        if any(
            token in _normalize_free_text(normalized_scope)
            for token in ("year ended", "fiscal year", "twelve months")
        ):
            return dt.date(target_year, 12, 31).isoformat()

    return None


def _extract_first_date(text: str) -> Optional[dt.date]:
    """从文本中提取首个有效日期。

    Args:
        text: 待解析文本。

    Returns:
        命中时返回 `date`，否则返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    normalized = _normalize_period_date_text(text)
    if not normalized:
        return None

    for match in re.finditer(r"\b(19\d{2}|20\d{2}|2100)[/-](\d{1,2})[/-](\d{1,2})\b", normalized):
        year = int(match.group(1))
        month = int(match.group(2))
        day = int(match.group(3))
        candidate = _build_safe_date(year=year, month=month, day=day)
        if candidate is not None:
            return candidate

    for match in _TEXTUAL_DATE_WITH_DASH_RE.finditer(normalized):
        day = int(match.group(1))
        month = _resolve_month_token(match.group(2))
        if month is None:
            continue
        year = _normalize_year_token(match.group(3))
        candidate = _build_safe_date(year=year, month=month, day=day)
        if candidate is not None:
            return candidate

    for match in re.finditer(
        r"\b(19\d{2}|20\d{2}|2100)\s+([A-Za-zÀ-ÿ]{3,12})\.?\s+(\d{1,2})\b",
        normalized,
    ):
        year = int(match.group(1))
        month = _resolve_month_token(match.group(2))
        if month is None:
            continue
        day = int(match.group(3))
        candidate = _build_safe_date(year=year, month=month, day=day)
        if candidate is not None:
            return candidate

    for match in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](19\d{2}|20\d{2}|2100)\b", normalized):
        first = int(match.group(1))
        second = int(match.group(2))
        year = int(match.group(3))
        if first > 12:
            month = second
            day = first
        elif second > 12:
            month = first
            day = second
        else:
            month = first
            day = second
        candidate = _build_safe_date(year=year, month=month, day=day)
        if candidate is not None:
            return candidate

    for match in re.finditer(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2})\b", normalized):
        first = int(match.group(1))
        second = int(match.group(2))
        year = _normalize_year_token(match.group(3))
        if first > 12:
            month = second
            day = first
        elif second > 12:
            month = first
            day = second
        else:
            month = first
            day = second
        candidate = _build_safe_date(year=year, month=month, day=day)
        if candidate is not None:
            return candidate

    for match in re.finditer(
        r"\b([A-Za-zÀ-ÿ]{3,12})\s+(\d{1,2})(?:,)?\s+(19\d{2}|20\d{2}|2100)\b",
        normalized,
    ):
        month = _resolve_month_token(match.group(1))
        if month is None:
            continue
        day = int(match.group(2))
        year = int(match.group(3))
        candidate = _build_safe_date(year=year, month=month, day=day)
        if candidate is not None:
            return candidate

    for match in re.finditer(
        r"\b(\d{1,2})\s+([A-Za-zÀ-ÿ]{3,12})\s+(19\d{2}|20\d{2}|2100)\b",
        normalized,
    ):
        day = int(match.group(1))
        month = _resolve_month_token(match.group(2))
        if month is None:
            continue
        year = int(match.group(3))
        candidate = _build_safe_date(year=year, month=month, day=day)
        if candidate is not None:
            return candidate

    for match in re.finditer(
        r"\b([A-Za-zÀ-ÿ]{3,12})\s+(19\d{2}|20\d{2}|2100)\b",
        normalized,
    ):
        month = _resolve_month_token(match.group(1))
        if month is None:
            continue
        year = int(match.group(2))
        day = calendar.monthrange(year, month)[1]
        candidate = _build_safe_date(year=year, month=month, day=day)
        if candidate is not None:
            return candidate

    for match in _MONTH_YEAR_RE.finditer(normalized):
        month = _resolve_month_token(match.group(1))
        if month is None:
            continue
        year = _normalize_year_token(match.group(2))
        day = calendar.monthrange(year, month)[1]
        candidate = _build_safe_date(year=year, month=month, day=day)
        if candidate is not None:
            return candidate

    for match in re.finditer(
        r"\b(19\d{2}|20\d{2}|2100)\s+([A-Za-zÀ-ÿ]{3,12})\.?\b",
        normalized,
    ):
        year = int(match.group(1))
        month = _resolve_month_token(match.group(2))
        if month is None:
            continue
        day = calendar.monthrange(year, month)[1]
        candidate = _build_safe_date(year=year, month=month, day=day)
        if candidate is not None:
            return candidate
    return None


def _normalize_period_date_text(text: str) -> str:
    """规范化期间表头文本，修复常见粘连写法。

    一批 6-K 表头会把 `endedSep 2025` 这类 token 粘连在一起，若不先做
    归一化，月份与财期解析会直接漏掉。

    Args:
        text: 原始期间文本。

    Returns:
        归一化后的期间文本。

    Raises:
        RuntimeError: 规范化失败时抛出。
    """

    normalized = _normalize_whitespace(text)
    if not normalized:
        return ""
    return _FUSED_PERIOD_MONTH_RE.sub(r"\1 \2", normalized)


def _extract_fiscal_period_year(text: str) -> Optional[tuple[str, int]]:
    """从文本中提取财期标签与财年。

    Args:
        text: 待解析文本。

    Returns:
        `(fiscal_period, fiscal_year)`；未命中返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    normalized_text = _normalize_period_date_text(text)
    if not normalized_text:
        return None

    quarter_match = _QUARTER_TOKEN_RE.search(normalized_text)
    if quarter_match is not None:
        quarter_raw = quarter_match.group(1) or quarter_match.group(2)
        year_raw = quarter_match.group(3)
        if quarter_raw is None:
            return None
        fiscal_year = _normalize_year_token(year_raw)
        return f"Q{quarter_raw}", fiscal_year

    year_first_quarter_match = _YEAR_FIRST_QUARTER_TOKEN_RE.search(normalized_text)
    if year_first_quarter_match is not None:
        fiscal_year = _normalize_year_token(year_first_quarter_match.group(1))
        quarter_raw = year_first_quarter_match.group(2) or year_first_quarter_match.group(3)
        if quarter_raw is None:
            return None
        return f"Q{quarter_raw}", fiscal_year

    half_match = _HALF_YEAR_TOKEN_RE.search(normalized_text)
    if half_match is not None:
        half_raw = half_match.group(1) or half_match.group(2)
        if half_raw is None:
            return None
        return f"H{half_raw}", _normalize_year_token(half_match.group(3))

    year_first_half_match = _YEAR_FIRST_HALF_YEAR_TOKEN_RE.search(normalized_text)
    if year_first_half_match is not None:
        fiscal_year = _normalize_year_token(year_first_half_match.group(1))
        half_raw = year_first_half_match.group(2) or year_first_half_match.group(3)
        if half_raw is None:
            return None
        return f"H{half_raw}", fiscal_year

    nine_month_match = _NINE_MONTH_TOKEN_RE.search(normalized_text)
    if nine_month_match is not None:
        return "Q3", _normalize_year_token(nine_month_match.group(1))

    six_month_match = _SIX_MONTH_TOKEN_RE.search(normalized_text)
    if six_month_match is not None:
        return "H1", _normalize_year_token(six_month_match.group(1))

    fy_match = _FY_TOKEN_RE.search(normalized_text)
    if fy_match is not None:
        return "FY", _normalize_year_token(fy_match.group(1))

    textual_quarter_match = _TEXTUAL_QUARTER_YEAR_RE.search(normalized_text)
    if textual_quarter_match is not None:
        quarter_token = str(textual_quarter_match.group(1) or "").strip().lower()
        quarter_map = {
            "first": "Q1",
            "second": "Q2",
            "third": "Q3",
            "fourth": "Q4",
        }
        fiscal_period = quarter_map.get(quarter_token)
        if fiscal_period is not None:
            return fiscal_period, _normalize_year_token(textual_quarter_match.group(2))
    return None


def _extract_fiscal_period_label(text: str) -> Optional[str]:
    """从文本中提取财期标签。

    Args:
        text: 待解析文本。

    Returns:
        财期标签；未命中返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    fiscal_period_year = _extract_fiscal_period_year(text)
    if fiscal_period_year is None:
        return None
    return fiscal_period_year[0]


def _normalize_year_token(value: str) -> int:
    """规范化年份 token。

    Args:
        value: 原始年份字符串。

    Returns:
        四位年份整数。

    Raises:
        RuntimeError: 规范化失败时抛出。
    """

    year = int(value)
    if year >= 100:
        return year
    return 2000 + year if year <= 69 else 1900 + year


def _resolve_period_end_from_fiscal_period(
    *,
    fiscal_period: str,
    fiscal_year: int,
) -> Optional[dt.date]:
    """根据财期标签推断 period_end 日期。

    Args:
        fiscal_period: 财期标签。
        fiscal_year: 财年。

    Returns:
        对应的期末日期；无法映射时返回 `None`。

    Raises:
        RuntimeError: 推断失败时抛出。
    """

    mapping: dict[str, tuple[int, int]] = {
        "Q1": (3, 31),
        "Q2": (6, 30),
        "Q3": (9, 30),
        "Q4": (12, 31),
        "H1": (6, 30),
        "H2": (12, 31),
        "FY": (12, 31),
    }
    month_day = mapping.get(fiscal_period)
    if month_day is None:
        return None
    month, day = month_day
    return _build_safe_date(year=fiscal_year, month=month, day=day)


def _extract_scope_month_day(scope_text: str) -> Optional[tuple[int, int]]:
    """从范围文本中提取月日。

    Args:
        scope_text: 范围文本。

    Returns:
        `(month, day)`；无法识别时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    normalized = _normalize_whitespace(scope_text)
    if not normalized:
        return None
    match = re.search(r"\b([A-Za-zÀ-ÿ]{3,12})\s+(\d{1,2})(?:,)?\b", normalized)
    if match is None:
        return None
    month = _resolve_month_token(match.group(1))
    if month is None:
        return None
    day = int(match.group(2))
    if not 1 <= day <= 31:
        return None
    return month, day


def _extract_years(text: str) -> list[int]:
    """提取文本中的年份列表。

    Args:
        text: 待解析文本。

    Returns:
        年份列表。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    years: list[int] = []
    seen: set[int] = set()
    for match in _YEAR_RE.finditer(text):
        year = int(match.group(1))
        if year in seen:
            continue
        seen.add(year)
        years.append(year)
    return years


def _resolve_month_token(token: str) -> Optional[int]:
    """将月份字符串映射为月份数字。

    Args:
        token: 原始月份文本。

    Returns:
        `1~12`；无法识别时返回 `None`。

    Raises:
        RuntimeError: 映射失败时抛出。
    """

    normalized_token = unicodedata.normalize("NFKD", token).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^A-Za-z]", "", normalized_token).lower()
    if not cleaned:
        return None
    if cleaned in _MONTH_TOKEN_TO_NUMBER:
        return _MONTH_TOKEN_TO_NUMBER[cleaned]
    if len(cleaned) >= 3 and cleaned[:3] in _MONTH_TOKEN_TO_NUMBER:
        return _MONTH_TOKEN_TO_NUMBER[cleaned[:3]]
    return None


def _build_safe_date(
    *,
    year: int,
    month: int,
    day: int,
) -> Optional[dt.date]:
    """构建合法日期对象。

    Args:
        year: 年份。
        month: 月份。
        day: 日期。

    Returns:
        合法日期对象；非法输入返回 `None`。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _infer_fiscal_period(
    *,
    scope_text: str,
    period_end: str,
) -> Optional[str]:
    """根据范围文本推断 fiscal_period。

    Args:
        scope_text: 范围文本。
        period_end: 期间结束日期。

    Returns:
        fiscal_period；无法判断时返回 `None`。

    Raises:
        RuntimeError: 推断失败时抛出。
    """

    explicit_period = _extract_fiscal_period_label(scope_text)
    if explicit_period is not None:
        return explicit_period
    normalized_scope = _normalize_free_text(scope_text)
    if any(token in normalized_scope for token in ("three months ended", "quarter ended", "three-month period")):
        return {
            3: "Q1",
            6: "Q2",
            9: "Q3",
            12: "Q4",
        }.get(int(period_end[5:7]))
    if "nine months ended" in normalized_scope and int(period_end[5:7]) == 9:
        return "Q3"
    if "six months ended" in normalized_scope:
        return {
            6: "H1",
            12: "H2",
        }.get(int(period_end[5:7]))
    if any(token in normalized_scope for token in ("twelve months ended", "year ended", "fiscal year")):
        return "FY"
    if "as of" in normalized_scope or "as at" in normalized_scope:
        month = int(period_end[5:7])
        return {
            3: "Q1",
            6: "Q2",
            9: "Q3",
            12: "FY",
        }.get(month)
    return None


def _extract_currency_for_column(
    *,
    scope_text: str,
    column_header_text: str,
) -> Optional[str]:
    """提取列级货币文本。

    Args:
        scope_text: 表格级范围文本。
        column_header_text: 列头文本。

    Returns:
        货币文本；未识别返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    combined = _normalize_whitespace(f"{column_header_text} {scope_text}")
    if not combined:
        return None
    lowered = combined.lower()
    if "hk$" in lowered:
        return _normalize_currency_text("HK$")
    if "us$" in lowered:
        return _normalize_currency_text("US$")
    if "rmb" in lowered:
        return _normalize_currency_text("RMB")
    if re.search(r"(?i)\bbrl\b", combined):
        return _normalize_currency_text("BRL")
    if re.search(r"(?i)\beur\b", combined):
        return _normalize_currency_text("EUR")
    if re.search(r"(?i)\bgbp\b", combined):
        return _normalize_currency_text("GBP")
    if "$" in combined:
        return _normalize_currency_text("$")
    return None


def _normalize_currency_text(raw_currency: str) -> Optional[str]:
    """规范化货币文本。

    Args:
        raw_currency: 原始货币文本。

    Returns:
        规范化后的货币文本；不存在时返回 `None`。

    Raises:
        RuntimeError: 规范化失败时抛出。
    """

    normalized = _normalize_whitespace(raw_currency)
    return normalized or None


def _map_currency_code(raw_currency: Optional[str]) -> Optional[str]:
    """将货币文本映射为标准代码。

    Args:
        raw_currency: 原始货币文本。

    Returns:
        标准货币代码；无法映射时返回原文本或 `None`。

    Raises:
        RuntimeError: 映射失败时抛出。
    """

    if raw_currency is None:
        return None
    return _CURRENCY_MAP.get(raw_currency, raw_currency)


def _build_units_label(
    *,
    primary_currency_raw: Optional[str],
    scale: Optional[str],
) -> Optional[str]:
    """构建 units 文本。

    Args:
        primary_currency_raw: 主货币文本。
        scale: 缩放口径。

    Returns:
        units 文本；不存在时返回 `None`。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    if primary_currency_raw is None and scale is None:
        return None
    if primary_currency_raw is None:
        return scale
    if scale is None:
        return primary_currency_raw
    return f"{primary_currency_raw} in {scale}"


def _infer_scale_from_caption(caption: Optional[str]) -> Optional[str]:
    """从 caption 推断缩放口径。

    Args:
        caption: 表格标题。

    Returns:
        缩放口径；无法识别时返回 `None`。

    Raises:
        RuntimeError: 推断失败时抛出。
    """

    normalized_caption = _normalize_free_text(caption or "")
    if "in thousands" in normalized_caption:
        return "thousands"
    if "in millions" in normalized_caption:
        return "millions"
    return None


def _extract_row_label(row: list[str]) -> Optional[str]:
    """从表格行中提取标签。

    Args:
        row: 表格行。

    Returns:
        行标签；不存在时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    for candidate in row[:6]:
        normalized = _normalize_whitespace(candidate)
        if not normalized or normalized.lower() == "nan":
            continue
        if _parse_optional_numeric(normalized) is not None:
            continue
        return normalized
    return None


def _parse_optional_numeric(value: Any) -> Optional[float]:
    """将单元格值解析为可选数值。

    Args:
        value: 输入值。

    Returns:
        浮点数；无法解析时返回 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    text = _normalize_whitespace(str(value or ""))
    if not text:
        return None
    if text.lower() in {"nan", "none", "nat", "n/a", "na", "nm"}:
        return None
    normalized = text.replace("−", "-").replace("–", "-").replace("—", "-")
    negative = normalized.startswith("(") and normalized.endswith(")")
    normalized = normalized.replace("(", "").replace(")", "").strip()
    normalized = _NUMERIC_PREFIX_RE.sub("", normalized)
    normalized = _NUMERIC_SUFFIX_RE.sub("", normalized)
    normalized = re.sub(r"[$€£¥₩₹]", "", normalized)
    normalized = re.sub(
        r"(?i)\b(?:us\$|hk\$|nt\$|r\$|ps\.?|cop|usd|eur|gbp|cny|rmb|ars|brl|mxn|chf|jpy|krw)\b",
        "",
        normalized,
    )
    normalized = normalized.replace("'", "").replace(" ", "").strip()
    if normalized in {"", "-", "--"}:
        return None
    normalized = normalize_numeric_separators(normalized)
    if normalized in {"", "-", "--"}:
        return None
    try:
        numeric = float(normalized)
    except ValueError:
        return None
    if pd.isna(numeric):
        return None
    return -numeric if negative else numeric


def normalize_numeric_separators(value: str) -> str:
    """规范化数字中的千分位与小数分隔符。

    Args:
        value: 原始数字字符串。

    Returns:
        规范化后的数字字符串。

    Raises:
        RuntimeError: 规范化失败时抛出。
    """

    cleaned = value.strip()
    if "." in cleaned and "," in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            return cleaned.replace(".", "").replace(",", ".")
        return cleaned.replace(",", "")
    if "," in cleaned:
        comma_parts = cleaned.split(",")
        if len(comma_parts) == 2 and 1 <= len(comma_parts[1]) <= 2:
            return cleaned.replace(",", ".")
        return cleaned.replace(",", "")
    return cleaned


def _count_pattern_hits(
    *,
    statement_patterns: tuple[re.Pattern[str], ...],
    text: str,
) -> int:
    """统计文本命中的模式数。

    Args:
        statement_patterns: 目标模式集合。
        text: 待匹配文本。

    Returns:
        命中的模式数量。

    Raises:
        RuntimeError: 统计失败时抛出。
    """

    if not text:
        return 0
    return sum(1 for pattern in statement_patterns if pattern.search(text))


def _normalize_match_text(value: str) -> str:
    """标准化用于匹配的文本。

    Args:
        value: 原始文本。

    Returns:
        标准化文本。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    return _normalize_whitespace(value).lower()


def _normalize_free_text(value: str) -> str:
    """标准化自由文本。

    Args:
        value: 原始文本。

    Returns:
        标准化文本。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    return _normalize_whitespace(value).lower()


__all__ = [
    "build_html_statement_result_from_tables",
    "select_html_statement_tables_by_row_signals",
]
