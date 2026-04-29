"""SEC XBRL 查询与财务报表结构化提取工具函数。

本模块从 ``sec_processor`` 中提取 XBRL 相关的查询、推断与格式化函数，
包括报表类型映射、taxonomy 推断、facts 查询、数值提取与标准化等。
"""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Optional

import pandas as pd
from edgar.xbrl import XBRL

from dayu.engine.processors.text_utils import (
    normalize_optional_string as _normalize_optional_string_base,
    normalize_whitespace as _normalize_whitespace,
)
from .financial_base import FinancialStatementResult, XbrlFactsResult

_STATEMENT_METHODS = {
    "income": "income_statement",
    "balance_sheet": "balance_sheet",
    "cash_flow": "cashflow_statement",
    "equity": "statement_of_equity",
    "comprehensive_income": "comprehensive_income",
}
_QUERY_STATEMENT_TYPES = {
    "income": "IncomeStatement",
    "income_statement": "IncomeStatement",
    "incomestatement": "IncomeStatement",
    "balance_sheet": "BalanceSheet",
    "balancesheet": "BalanceSheet",
    "cash_flow": "CashFlowStatement",
    "cashflowstatement": "CashFlowStatement",
    "statement_of_changes_in_equity": "StatementOfChangesInEquity",
    "statementofchangesinequity": "StatementOfChangesInEquity",
    "equity": "StatementOfChangesInEquity",
    "comprehensive_income": "ComprehensiveIncome",
    "comprehensiveincome": "ComprehensiveIncome",
}
_STATEMENT_TITLE_BY_TYPE = {
    "income": "Income Statement",
    "balance_sheet": "Balance Sheet",
    "cash_flow": "Cash Flow Statement",
    "equity": "Statement of Changes in Equity",
    "comprehensive_income": "Comprehensive Income",
}

# decimals → scale 映射表（与 service 层 _DECIMALS_SCALE_MAP 保持一致）
_DECIMALS_SCALE_MAP: dict[int, str] = {
    -9: "billions",
    -6: "millions",
    -3: "thousands",
    0: "units",
}

# 用于 units/scale 推断的主营收入概念候选（按 US-GAAP 实务命中优先级排序）。
# - ``Revenues``：US-GAAP 2018+ 主流命名。
# - ``Revenue``：部分 IFRS-aligned filer 与子集成器使用。
# - ``SalesRevenueNet`` / ``SalesRevenueGoodsNet``：旧规与细分场景。
_REVENUE_CONCEPT_CANDIDATES: tuple[str, ...] = (
    "Revenues",
    "Revenue",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
)

# ISO 4217 主要货币代码（覆盖 SEC 主要外国发行人语种），用于从 units
# 字符串中识别货币代码。未命中时返回 ``None``，避免把诸如 ``"shares"``
# 之类的非货币单位误当作货币代码传给下游。
_KNOWN_CURRENCY_CODES: frozenset[str] = frozenset(
    {
        "USD", "EUR", "GBP", "JPY", "CNY", "HKD", "TWD", "KRW", "INR",
        "CAD", "AUD", "CHF", "BRL", "MXN", "SGD", "ZAR",
    }
)


def _normalize_optional_string(value: Any) -> Optional[str]:
    """将任意值转为可选字符串，额外处理 pandas NaN/NaT。

    对 ``None``、空字符串、``float('nan')``、``pd.NaT`` 等无意义值统一返回 ``None``。

    Args:
        value: 任意输入值。

    Returns:
        标准化字符串；空值返回 ``None``。
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return _normalize_optional_string_base(value)


def _infer_xbrl_taxonomy(xbrl: XBRL) -> Optional[str]:
    """推断 XBRL taxonomy。

    Args:
        xbrl: XBRL 对象。

    Returns:
        taxonomy（`us-gaap` / `ifrs-full`）或 `None`。

    Raises:
        RuntimeError: 推断失败时抛出。
    """

    probes = ("Assets", "Revenues", "Revenue")
    for probe in probes:
        try:
            rows = xbrl.query().by_concept(probe).execute()
        except Exception:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            concept = str(row.get("concept") or "")
            taxonomy = _extract_taxonomy_from_concept(concept)
            if taxonomy is not None:
                return taxonomy
    return None


def _extract_taxonomy_from_concept(concept: str) -> Optional[str]:
    """从 concept 名称提取 taxonomy 前缀。

    Args:
        concept: concept 名称。

    Returns:
        `us-gaap`、`ifrs-full` 或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    normalized = _normalize_whitespace(concept)
    if ":" not in normalized:
        return None
    prefix = normalized.split(":", 1)[0].strip().lower()
    if prefix.startswith("us-gaap"):
        return "us-gaap"
    if prefix.startswith("ifrs"):
        return "ifrs-full"
    return None


def _extract_period_columns(columns: Any) -> list[str]:
    """识别报表期末列。

    Args:
        columns: DataFrame 列集合。

    Returns:
        期末列名列表。

    Raises:
        RuntimeError: 识别失败时抛出。
    """

    period_columns: list[str] = []
    for column in columns:
        column_str = str(column)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", column_str):
            period_columns.append(column_str)
    return period_columns


def _build_statement_rows(statement_df: pd.DataFrame, period_columns: list[str]) -> list[dict[str, Any]]:
    """构建标准财务行结构。

    Args:
        statement_df: 报表 DataFrame。
        period_columns: 期末列列表。

    Returns:
        行列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    rows: list[dict[str, Any]] = []
    for _, row in statement_df.iterrows():
        concept = _normalize_optional_string(row.get("concept")) or ""
        label = _normalize_optional_string(row.get("label")) or concept
        values = [_to_optional_float(row.get(period)) for period in period_columns]
        if not concept and not label:
            continue
        rows.append(
            {
                "concept": concept,
                "label": label,
                "values": values,
            }
        )
    return rows


def _build_period_summary(period_end: str) -> dict[str, Any]:
    """构建期间摘要。

    Args:
        period_end: 期末日期（YYYY-MM-DD）。

    Returns:
        期间摘要字典。

    Raises:
        ValueError: 日期非法时抛出。
    """

    fiscal_year = int(period_end[:4]) if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period_end) else None
    return {
        "period_end": period_end,
        "fiscal_year": fiscal_year,
        "fiscal_period": "FY" if fiscal_year is not None else None,
    }


def _format_statement_period_label(period_summary: dict[str, Any]) -> str:
    """将期间摘要格式化为稳定的报表期间标签。

    Args:
        period_summary: `_build_period_summary` 生成的期间摘要。

    Returns:
        适合写入 statement locator 的期间标签；优先返回 `FY2025` 这类口径，
        无法归一时退回原始 `period_end`。

    Raises:
        无。
    """

    fiscal_year = period_summary.get("fiscal_year")
    fiscal_period = _normalize_optional_string(period_summary.get("fiscal_period"))
    period_end = _normalize_optional_string(period_summary.get("period_end"))
    if isinstance(fiscal_year, int) and fiscal_period:
        return f"{fiscal_period}{fiscal_year}"
    return period_end or ""


def _extract_statement_row_labels(rows: list[dict[str, Any]]) -> list[str]:
    """从结构化报表行中提取去重后的行标签。

    Args:
        rows: 标准化报表行列表。

    Returns:
        去重且保序的行标签列表。

    Raises:
        无。
    """

    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        label = _normalize_optional_string(row.get("label")) or _normalize_optional_string(row.get("concept")) or ""
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def build_statement_locator(
    *,
    statement_type: str,
    periods: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    statement_title: Optional[str] = None,
) -> dict[str, Any]:
    """构建结构化报表定位信息。

    该定位信息用于：
    - 让 write 在"证据与出处"中稳定表达 `get_financial_statement` 来源；
    - 让 confirm/repair 能以 statement + period + row 的粒度复核证据。

    Args:
        statement_type: 报表类型。
        periods: 报表期间摘要列表。
        rows: 报表行列表。
        statement_title: 可选的人类可读报表标题；为空时按类型映射推断。

    Returns:
        结构化定位信息字典。

    Raises:
        无。
    """

    normalized_statement_type = statement_type.strip().lower()
    resolved_title = statement_title or _STATEMENT_TITLE_BY_TYPE.get(normalized_statement_type) or statement_type
    period_labels = [label for label in (_format_statement_period_label(period) for period in periods) if label]
    row_labels = _extract_statement_row_labels(rows)
    return {
        "statement_type": statement_type,
        "statement_title": resolved_title,
        "period_labels": period_labels,
        "row_labels": row_labels,
    }


def _to_optional_float(value: Any) -> Optional[float]:
    """将值转换为可选浮点数。

    Args:
        value: 输入值。

    Returns:
        浮点数或 `None`。

    Raises:
        ValueError: 转换失败时抛出。
    """

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if pd.isna(numeric):
        return None
    return numeric


def _normalize_query_statement_type(statement_type: Optional[str]) -> Optional[str]:
    """标准化 XBRL 查询报表类型。

    Args:
        statement_type: 输入报表类型。

    Returns:
        标准化报表类型；无法识别返回 `None`。

    Raises:
        ValueError: 输入非法时抛出。
    """

    if statement_type is None:
        return None
    key = re.sub(r"[\s_]+", "", statement_type.strip().lower())
    if not key:
        return None
    return _QUERY_STATEMENT_TYPES.get(key, statement_type)


def _build_xbrl_value_filter(
    min_value: Optional[float],
    max_value: Optional[float],
) -> Callable[[float], bool] | tuple[float, float] | None:
    """构建 edgartools `FactQuery.by_value` 所需过滤参数。

    Args:
        min_value: 可选最小值。
        max_value: 可选最大值。

    Returns:
        双边界存在时返回 `(min, max)` 元组；单边界时返回谓词；都为空时返回 `None`。

    Raises:
        ValueError: 输入非法时抛出。
    """

    if min_value is None and max_value is None:
        return None
    if min_value is not None and max_value is not None:
        return (min_value, max_value)

    def _predicate(value: float) -> bool:
        """判断数值是否满足单边界过滤条件。

        Args:
            value: 待判断数值。

        Returns:
            满足过滤条件返回 `True`，否则返回 `False`。

        Raises:
            ValueError: 输入非法时抛出。
        """

        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
        return True

    return _predicate


def _apply_xbrl_value_filter(
    query_obj: Any,
    min_value: Optional[float],
    max_value: Optional[float],
) -> Any:
    """兼容不同 edgartools `by_value` 签名应用数值过滤。

    Args:
        query_obj: facts 查询链对象。
        min_value: 可选最小值。
        max_value: 可选最大值。

    Returns:
        应用过滤后的查询链对象；无过滤条件时返回原对象。

    Raises:
        AttributeError: 查询对象缺失 `by_value` 时抛出。
    """

    value_filter = _build_xbrl_value_filter(min_value=min_value, max_value=max_value)
    if value_filter is None:
        return query_obj

    by_value = getattr(query_obj, "by_value")
    try:
        parameter_count = len(inspect.signature(by_value).parameters)
    except (TypeError, ValueError):
        parameter_count = 1

    if parameter_count >= 2:
        return by_value(min_value, max_value)
    return by_value(value_filter)


def _query_facts_rows(
    xbrl: XBRL,
    concepts: list[str],
    statement_type: Optional[str],
    period_end: Optional[str],
    fiscal_year: Optional[int],
    fiscal_period: Optional[str],
    min_value: Optional[float],
    max_value: Optional[float],
) -> list[dict[str, Any]]:
    """执行 XBRL facts 查询。

    Args:
        xbrl: XBRL 对象。
        concepts: 概念列表。
        statement_type: 可选报表类型。
        period_end: 可选期末日期。
        fiscal_year: 可选财年。
        fiscal_period: 可选财季。
        min_value: 可选最小值。
        max_value: 可选最大值。

    Returns:
        facts 原始行列表（仅含数值事实，且按 concept 本地名精确匹配）。

    Raises:
        RuntimeError: 查询失败时抛出。
    """

    rows: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    normalized_period_end = _normalize_optional_string(period_end)
    normalized_fiscal_period = _normalize_optional_string(fiscal_period)
    for concept in concepts:
        target_local_name = _extract_concept_local_name(concept)
        if not target_local_name:
            continue
        query_obj = xbrl.query().by_concept(concept)
        if statement_type:
            query_obj = query_obj.by_statement_type(statement_type)
        if fiscal_year is not None:
            query_obj = query_obj.by_fiscal_year(fiscal_year)
        if normalized_fiscal_period:
            query_obj = query_obj.by_fiscal_period(normalized_fiscal_period.upper())
        query_obj = _apply_xbrl_value_filter(
            query_obj,
            min_value=min_value,
            max_value=max_value,
        )
        try:
            result_rows = query_obj.execute()
        except Exception:
            continue
        for row in result_rows:
            if not isinstance(row, dict):
                continue
            row_concept = str(row.get("concept") or "")
            if not _matches_concept_exact_local_name(row_concept, target_local_name):
                continue
            if _is_text_block_concept(row_concept):
                continue
            numeric_value = _extract_numeric_fact_value(row)
            if numeric_value is None:
                continue
            row["numeric_value"] = numeric_value
            if normalized_period_end and str(row.get("period_end") or "") != normalized_period_end:
                continue
            dedup_key = _build_fact_dedup_key(row)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            rows.append(row)
    return rows


def _build_fact_dedup_key(row: dict[str, Any]) -> str:
    """构建 fact 去重键。

    Args:
        row: fact 原始字典。

    Returns:
        去重键字符串。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    parts = [
        str(row.get("fact_key") or ""),
        str(row.get("concept") or ""),
        str(row.get("period_end") or ""),
        str(row.get("numeric_value") or row.get("value") or ""),
    ]
    return "|".join(parts)


def _normalize_fact_row(row: dict[str, Any]) -> dict[str, Any]:
    """标准化单条 fact 输出。

    Args:
        row: 原始 fact 字典。

    Returns:
        标准化 fact 字典。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    concept = str(row.get("concept") or "")
    label = str(row.get("label") or row.get("original_label") or concept)
    numeric_value = _extract_numeric_fact_value(row)
    raw_text_value = row.get("value")
    text_value = None
    content_type = None
    if numeric_value is None and isinstance(raw_text_value, str):
        text_value = raw_text_value
        content_type = _infer_text_content_type(raw_text_value)
    unit = row.get("unit") or row.get("unit_ref")
    return {
        "concept": concept,
        "label": label,
        "numeric_value": numeric_value,
        "text_value": text_value,
        "content_type": content_type,
        "unit": unit,
        "decimals": row.get("decimals"),
        "period_type": row.get("period_type"),
        "period_start": row.get("period_start"),
        "period_end": row.get("period_end"),
        "fiscal_year": row.get("fiscal_year"),
        "fiscal_period": row.get("fiscal_period"),
        "statement_type": row.get("statement_type"),
    }


def _extract_concept_local_name(concept: str) -> str:
    """提取 concept 的本地名。

    Args:
        concept: 原始 concept 名称，支持 `namespace:local` 或 `namespace_local`。

    Returns:
        规范化后的本地名；无法提取时返回空字符串。

    Raises:
        RuntimeError: 无。
    """

    stripped = concept.strip()
    if not stripped:
        return ""
    # 约定：XBRL fact 行里下划线形式的 concept 由 edgartools 等库把唯一的
    # namespace 分隔符 `:` 替换成 `_`，本地名内部仍可含 `_`（自定义 taxonomy 常见）。
    # 因此首个分隔符之后的全部都属本地名，必须用限次 split 而不是 `[-1]`/全文 replace，
    # 否则形如 `company_Custom_Metric` 会被错误截短为 `Metric`。
    if ":" in stripped:
        return stripped.split(":", 1)[1].strip()
    if "_" in stripped:
        return stripped.split("_", 1)[1].strip()
    return stripped


def _normalize_concept_match_key(value: str) -> str:
    """将 concept 匹配键标准化为可比较格式。

    Args:
        value: 输入 concept 名称或本地名。

    Returns:
        标准化键（小写、去空白）；输入为空时返回空字符串。

    Raises:
        RuntimeError: 无。
    """

    local_name = _extract_concept_local_name(value)
    if not local_name:
        return ""
    return local_name.lower()


def _matches_concept_exact_local_name(row_concept: str, target_concept: str) -> bool:
    """判断事实 concept 是否与目标 concept 本地名精确匹配。

    Args:
        row_concept: fact 行中的 concept。
        target_concept: 查询目标 concept。

    Returns:
        两者本地名是否精确一致。

    Raises:
        RuntimeError: 无。
    """

    normalized_row = _normalize_concept_match_key(row_concept)
    normalized_target = _normalize_concept_match_key(target_concept)
    if not normalized_row or not normalized_target:
        return False
    return normalized_row == normalized_target


def _is_text_block_concept(concept: str) -> bool:
    """判断 concept 是否为 TextBlock 非数值概念。

    Args:
        concept: concept 名称。

    Returns:
        若本地名以 `TextBlock` 结尾则返回 `True`，否则返回 `False`。

    Raises:
        RuntimeError: 无。
    """

    local_name = _extract_concept_local_name(concept)
    if not local_name:
        return False
    return local_name.lower().endswith("textblock")


def _extract_numeric_fact_value(row: dict[str, Any]) -> Optional[float]:
    """提取 fact 的可用数值。

    Args:
        row: XBRL fact 原始行。

    Returns:
        可解析数值时返回浮点值；否则返回 `None`。

    Raises:
        RuntimeError: 无。
    """

    numeric_value = _to_optional_float(row.get("numeric_value"))
    if numeric_value is not None:
        return numeric_value
    return _to_optional_float(row.get("value"))


def _infer_text_content_type(value: str) -> str:
    """推断文本值的内容类型。

    Args:
        value: 文本值。

    Returns:
        若疑似 HTML/XHTML 片段返回 `xhtml`，否则返回 `plain`。

    Raises:
        RuntimeError: 无。
    """

    if re.search(r"<\s*/?\s*[a-zA-Z][^>]*>", value):
        return "xhtml"
    return "plain"


def _infer_units_from_xbrl_query(xbrl: XBRL) -> Optional[str]:
    """从 XBRL 查询推断单位。

    依次尝试 ``_REVENUE_CONCEPT_CANDIDATES`` 中的概念，命中即返回该 fact
    的 ``unit`` / ``unit_ref``，扩展对 IFRS-aligned 与旧规命名 filer 的覆盖。

    Args:
        xbrl: XBRL 对象。

    Returns:
        单位字符串或 `None`。

    Raises:
        无。
    """

    for concept in _REVENUE_CONCEPT_CANDIDATES:
        try:
            rows = xbrl.query().by_concept(concept).execute()
        except Exception:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            unit = row.get("unit") or row.get("unit_ref")
            if unit:
                return str(unit).upper()
    return None


def _infer_currency_from_units(units: Optional[str]) -> Optional[str]:
    """从单位字符串中识别 ISO 4217 货币代码。

    在 ``_KNOWN_CURRENCY_CODES`` 范围内做子串匹配，命中即返回标准代码；
    未命中返回 ``None``，避免把诸如 ``"shares"`` 这类非货币单位误当作货币
    代码传给下游。

    Args:
        units: 单位字符串。

    Returns:
        ISO 4217 货币代码或 ``None``。

    Raises:
        无。
    """

    if not units:
        return None
    upper_units = units.upper()
    for code in _KNOWN_CURRENCY_CODES:
        if code in upper_units:
            return code
    return None


def _infer_scale_from_xbrl_query(xbrl: XBRL) -> Optional[str]:
    """从 XBRL Revenue facts 的 decimals 属性推断数值 scale。

    依次尝试 ``_REVENUE_CONCEPT_CANDIDATES``，取首条命中 fact 的 ``decimals``
    字段并按映射表推断 scale（如 ``-6`` → ``millions``）。SEC 实务中单 filing
    内不同报表 scale 一致，因此使用单一概念探测即可。

    Args:
        xbrl: XBRL 对象。

    Returns:
        scale 描述字符串或 ``None``。

    Raises:
        无。
    """

    for concept in _REVENUE_CONCEPT_CANDIDATES:
        try:
            rows = xbrl.query().by_concept(concept).execute()
        except Exception:
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_decimals = row.get("decimals")
            if raw_decimals is None:
                continue
            # 解析 decimals 值
            if isinstance(raw_decimals, str):
                stripped = raw_decimals.strip().upper()
                if stripped == "INF":
                    return "units"
                try:
                    decimals_int = int(stripped)
                except ValueError:
                    continue
            else:
                try:
                    decimals_int = int(raw_decimals)
                except (TypeError, ValueError):
                    continue
            # 查映射表
            exact = _DECIMALS_SCALE_MAP.get(decimals_int)
            if exact is not None:
                return exact
            if decimals_int > 0:
                return "units"
    return None
