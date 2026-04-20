"""服务层通用辅助函数。

该模块包含 FinsToolService 所用的非搜索辅助逻辑：
- 文本标准化（必填/可选/form_type）
- 推荐文档构建
- 章节标准化（children / page_range）
- 财务日期推断（fiscal_year / fiscal_period）
- 表格数据载荷标准化（records / markdown / raw_text）
- XBRL 查询与 fact 标准化（concept 归一化 / 去重 / scale 推断）
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from html import unescape
from typing import Any, Optional, cast

from dayu.engine.exceptions import ToolArgumentError
from dayu.fins._converters import normalize_optional_text
from dayu.engine.processors.base import (
    SectionContent,
    SectionSummary,
    TableContent,
)
from .result_types import NotSupportedResult
from dayu.fins.domain.enums import SourceKind
from dayu.fins.processors.form_type_utils import normalize_form_type

# ---------------------------------------------------------------------------
# 预编译正则
# ---------------------------------------------------------------------------
_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

# ---------------------------------------------------------------------------
# XBRL 默认概念常量
# ---------------------------------------------------------------------------
_GLOBAL_DEFAULT_XBRL_CONCEPTS: tuple[str, ...] = ("Revenues", "NetIncomeLoss", "Assets")

_DEFAULT_XBRL_CONCEPTS_BY_FORM_TAXONOMY: dict[tuple[str, str], tuple[str, ...]] = {
    ("10-K", "us-gaap"): (
        "Revenues",
        "NetIncomeLoss",
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "NetCashProvidedByUsedInOperatingActivities",
    ),
    ("10-Q", "us-gaap"): (
        "Revenues",
        "NetIncomeLoss",
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "NetCashProvidedByUsedInOperatingActivities",
    ),
    ("20-F", "ifrs-full"): (
        "Revenue",
        "ProfitLoss",
        "Assets",
        "Liabilities",
        "Equity",
        "CashAndCashEquivalents",
    ),
}

_DEFAULT_XBRL_CONCEPTS_BY_TAXONOMY: dict[str, tuple[str, ...]] = {
    "us-gaap": (
        "Revenues",
        "NetIncomeLoss",
        "Assets",
        "Liabilities",
        "StockholdersEquity",
    ),
    "ifrs-full": (
        "Revenue",
        "ProfitLoss",
        "Assets",
        "Liabilities",
        "Equity",
    ),
}

# ---------------------------------------------------------------------------
# 推荐文档槽位常量
# ---------------------------------------------------------------------------
_RECOMMENDED_DOCUMENT_KEYS: tuple[str, ...] = (
    "latest_document_id",
    "recommended_for_company_overview_document_id",
    "latest_annual_report_document_id",
    "latest_quarterly_report_document_id",
    "latest_current_report_document_id",
    "latest_proxy_document_id",
    "latest_ownership_document_id",
    "latest_earnings_call_document_id",
    "latest_earnings_presentation_document_id",
    "latest_material_document_id",
)

# ---------------------------------------------------------------------------
# form_type → document_type 映射
#
# document_type 是面向 LLM 的语义字段，屏蔽底层 SEC 表单细节。
# 预留值（目前无 form_type 对应，通过 source_kind 或未来扩展触发）：
#   semi_annual_report — A 股半年报（H1）
#   earnings_call      — 财报电话会议（存放于 materials/）
# ---------------------------------------------------------------------------
_FORM_TYPE_TO_DOCUMENT_TYPE: dict[str, str] = {
    "10-K": "annual_report",
    "10-K/A": "annual_report",
    "20-F": "annual_report",
    "20-F/A": "annual_report",
    "10-Q": "quarterly_report",
    "10-Q/A": "quarterly_report",
    "6-K": "quarterly_report",
    "8-K": "current_report",
    "8-K/A": "current_report",
    "DEF 14A": "proxy",
    "SC 13G": "ownership",
    "SC 13G/A": "ownership",
    "SC 13D": "ownership",
    "SC 13D/A": "ownership",
}

# 港股/A 股上传链路当前将财期直接写入 source meta.form_type。
# 这里把 fiscal_period 语义回收为 LLM-facing document_type，
# 避免 `list_documents` 把年报/中报/季报全部误判为 other。
_CN_FORM_TYPE_TO_DOCUMENT_TYPE: dict[str, str] = {
    "FY": "annual_report",
    "H1": "semi_annual_report",
    "Q1": "quarterly_report",
    "Q2": "quarterly_report",
    "Q3": "quarterly_report",
    "Q4": "quarterly_report",
}

# 缺失 report_date / filing_date 时，按 fiscal_period 做时间顺序回退。
# 数字越大表示同一年内越“新”。
_FISCAL_PERIOD_SORT_ORDER: dict[str, int] = {
    "Q1": 1,
    "Q2": 2,
    "H1": 3,
    "Q3": 4,
    "Q4": 5,
    "FY": 6,
}

# LLM 可传入的合法 document_type 值集合（含预留值）
_VALID_DOCUMENT_TYPES: frozenset[str] = frozenset({
    "annual_report",
    "semi_annual_report",
    "quarterly_report",
    "current_report",
    "proxy",
    "ownership",
    "earnings_call",
    "earnings_presentation",
    "corporate_governance",
    "material",
    "other",
})

# material form_type → document_type 精细映射表
# 未列出的 form_type 回落到通用 "material"
_MATERIAL_FORM_TYPE_TO_DOCUMENT_TYPE: dict[str, str] = {
    "EARNINGS_CALL": "earnings_call",
    "EARNINGS_PRESENTATION": "earnings_presentation",
    "CORPORATE_GOVERNANCE": "corporate_governance",
}

# 历史或人工维护数据中可能出现的 material form_type 变体。
# 工具链路在消费文档元数据时统一归一化，避免脏数据放大到 document_type。
_MATERIAL_FORM_TYPE_ALIASES: dict[str, str] = {
    "EARNING_CALLS": "EARNINGS_CALL",
    "EARNINGS_CALLS": "EARNINGS_CALL",
    "EARNING_PRESENTATIONS": "EARNINGS_PRESENTATION",
    "EARNINGS_PRESENTATIONS": "EARNINGS_PRESENTATION",
}


def _resolve_document_type(form_type: Optional[str], source_kind: str) -> str:
    """根据 form_type 和 source_kind 推导文档类型（document_type）。

    返回值为面向 LLM 的语义枚举，见 _VALID_DOCUMENT_TYPES。

    Args:
        form_type: 标准化后的表单类型。
        source_kind: 文档来源类型（filing / material）。

    Returns:
        document_type 字符串。

    Raises:
        无。
    """

    if source_kind == SourceKind.MATERIAL.value:
        # 特定 material 类型映射到语义更明确的 document_type
        if form_type in _MATERIAL_FORM_TYPE_TO_DOCUMENT_TYPE:
            return _MATERIAL_FORM_TYPE_TO_DOCUMENT_TYPE[form_type]
        return "material"
    if form_type is None:
        return "other"
    if form_type in _CN_FORM_TYPE_TO_DOCUMENT_TYPE:
        return _CN_FORM_TYPE_TO_DOCUMENT_TYPE[form_type]
    return _FORM_TYPE_TO_DOCUMENT_TYPE.get(form_type, "other")


def build_document_recency_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    """构建文档摘要的统一排序键。

    排序目标：
    1. 优先按显式日期排序（`report_date` > `filing_date`）。
    2. 日期缺失时，回退到 `fiscal_year + fiscal_period`。
    3. 若两者都缺失，再回退到 `document_id`，仅用于稳定排序。

    Args:
        item: 文档摘要字典。

    Returns:
        可直接用于 ``list.sort(..., reverse=True)`` 的排序键。

    Raises:
        无。
    """

    report_date = normalize_optional_text(item.get("report_date")) or ""
    filing_date = normalize_optional_text(item.get("filing_date")) or ""
    has_explicit_date = bool(report_date or filing_date)

    fiscal_year = item.get("fiscal_year")
    normalized_fiscal_year = fiscal_year if isinstance(fiscal_year, int) else -1
    normalized_fiscal_period = normalize_optional_text(item.get("fiscal_period"))
    fiscal_period_rank = _FISCAL_PERIOD_SORT_ORDER.get(normalized_fiscal_period or "", 0)
    has_fiscal_recency = normalized_fiscal_year > 0 or fiscal_period_rank > 0
    temporal_rank = 2 if has_explicit_date else 1 if has_fiscal_recency else 0

    primary_date = report_date or filing_date
    secondary_date = filing_date or report_date
    document_id = normalize_optional_text(item.get("document_id")) or ""
    return (
        temporal_rank,
        primary_date,
        secondary_date,
        normalized_fiscal_year,
        fiscal_period_rank,
        document_id,
    )


def resolve_document_type_for_source(*, form_type: Any, source_kind: Any) -> str:
    """根据原始源文档元数据推导稳定 document_type。

    该函数统一封装工具链路的 document_type 推导逻辑：
    先标准化原始 ``form_type``，再结合 ``source_kind`` 映射到
    面向 LLM 的语义 ``document_type``。

    Args:
        form_type: 原始表单类型值。
        source_kind: 原始来源类型值。

    Returns:
        稳定的 ``document_type`` 字符串。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized_form_type = _normalize_form_type_for_matching(form_type)
    normalized_source_kind = normalize_optional_text(source_kind) or ""
    return _resolve_document_type(normalized_form_type, normalized_source_kind)


def _collect_available_document_types(documents: list[dict[str, Any]]) -> list[str]:
    """提取文档列表中出现的所有 document_type（去重、排序）。

    对尚未附加 document_type 字段的原始文档（base_documents）也适用，
    按需从 form_type / source_kind 实时推导。

    Args:
        documents: 文档摘要列表（来自仓储的原始条目）。

    Returns:
        去重后的 document_type 列表（字母序）。

    Raises:
        无。
    """

    doc_types: set[str] = set()
    for doc in documents:
        # 若已附加 document_type 字段则直接取；否则实时推导
        dt = doc.get("document_type")
        if dt is None:
            dt = resolve_document_type_for_source(
                form_type=doc.get("form_type"),
                source_kind=doc.get("source_kind"),
            )
        doc_types.add(dt)
    return sorted(doc_types)


def _collect_parent_titles(
    section: SectionSummary,
    ref_to_section: dict[str, SectionSummary],
) -> list[str]:
    """上溯 parent_ref 链收集父章节标题。

    返回列表从直接父到根的顺序，供 build_section_path 使用
    （build_section_path 内部会反转为正序）。

    Args:
        section: 当前章节。
        ref_to_section: ref → section 索引。

    Returns:
        父标题列表（从直接父到根）。
    """
    titles: list[str] = []
    visited: set[str] = set()
    current_ref = section.get("parent_ref")
    while current_ref and current_ref not in visited:
        visited.add(current_ref)
        parent = ref_to_section.get(current_ref)
        if parent is None:
            break
        parent_title = parent.get("title")
        if parent_title:
            titles.append(parent_title)
        current_ref = parent.get("parent_ref")
    return titles


def _normalize_form_type_for_matching(value: Any) -> Optional[str]:
    """标准化文档表单类型。

    该函数用于工具层统一匹配口径，处理 `SC 13* / SCHEDULE 13* / 10K / 10-Q`
    等别名差异，确保筛选与推荐逻辑稳定。

    Args:
        value: 原始表单类型值。

    Returns:
        标准化表单类型；无法标准化时返回 `None`。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized = normalize_optional_text(value)
    if normalized is None:
        return None
    normalized_form = normalize_form_type(normalized)
    normalized_text = normalize_optional_text(normalized_form)
    if normalized_text is None:
        return None
    return _MATERIAL_FORM_TYPE_ALIASES.get(normalized_text, normalized_text)


def _normalize_document_types(document_types: Optional[list[str]]) -> Optional[list[str]]:
    """标准化 document_types 数组参数。

    仅允许 _VALID_DOCUMENT_TYPES 中定义的枚举值，
    非法值直接丢弃（宽松策略，避免因 LLM 拼写变体导致全部过滤失效）。

    Args:
        document_types: 原始文档类型数组（LLM 传入）。

    Returns:
        去重去空后的合法列表；输入为空时返回 `None`。

    Raises:
        ToolArgumentError: 入参类型非法时抛出。
    """

    if document_types is None:
        return None
    if not isinstance(document_types, list):
        raise ToolArgumentError("list_documents", "document_types", document_types, "Must be a string array")
    result: list[str] = []
    seen: set[str] = set()
    for dt in document_types:
        normalized = normalize_optional_text(dt)
        if normalized is None or normalized not in _VALID_DOCUMENT_TYPES:
            continue
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result or None


def _build_recommended_documents(documents: list[dict[str, Any]]) -> dict[str, Optional[str]]:
    """构建 `list_documents.recommended_documents` 固定槽位。

    Args:
        documents: 已按时间倒序的全量文档列表（附带 `document_type`）。

    Returns:
        推荐文档槽位字典。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    recommendations: dict[str, Optional[str]] = {key: None for key in _RECOMMENDED_DOCUMENT_KEYS}
    if not documents:
        return recommendations

    for item in documents:
        document_id = normalize_optional_text(item.get("document_id"))
        if document_id is None:
            continue
        # document_type 由调用方在过滤循环中已附加
        doc_type = item.get("document_type") or ""

        if recommendations["latest_document_id"] is None:
            recommendations["latest_document_id"] = document_id
        if recommendations["latest_annual_report_document_id"] is None and doc_type == "annual_report":
            recommendations["latest_annual_report_document_id"] = document_id
        if recommendations["latest_quarterly_report_document_id"] is None and doc_type in {"quarterly_report", "semi_annual_report"}:
            recommendations["latest_quarterly_report_document_id"] = document_id
        if recommendations["latest_current_report_document_id"] is None and doc_type == "current_report":
            recommendations["latest_current_report_document_id"] = document_id
        if recommendations["latest_proxy_document_id"] is None and doc_type == "proxy":
            recommendations["latest_proxy_document_id"] = document_id
        if recommendations["latest_ownership_document_id"] is None and doc_type == "ownership":
            recommendations["latest_ownership_document_id"] = document_id
        if recommendations["latest_earnings_call_document_id"] is None and doc_type == "earnings_call":
            recommendations["latest_earnings_call_document_id"] = document_id
        if (
            recommendations["latest_earnings_presentation_document_id"] is None
            and doc_type == "earnings_presentation"
        ):
            recommendations["latest_earnings_presentation_document_id"] = document_id
        if recommendations["latest_material_document_id"] is None and doc_type == "material":
            recommendations["latest_material_document_id"] = document_id

    recommendations["recommended_for_company_overview_document_id"] = (
        recommendations["latest_annual_report_document_id"]
        or recommendations["latest_quarterly_report_document_id"]
        or recommendations["latest_proxy_document_id"]
        or recommendations["latest_current_report_document_id"]
        or recommendations["latest_ownership_document_id"]
        or recommendations["latest_document_id"]
    )
    return recommendations


def resolve_has_financial_data(
    *,
    has_financial_data: Any = None,
    availability: Any = None,
    has_financial_statement: Any = None,
    has_xbrl: Any = None,
    has_structured_financial_statements: Any = None,
    has_financial_statement_sections: Any = None,
) -> Optional[bool]:
    """保守推导 has_financial_data。

    设计原则：宁可返回 `None`，也不在能力语义不清时误导 LLM。

    判定优先级：
    1. 直接的 `has_financial_data` 显式字段
    2. 内部 `financial_statement_availability` 枚举
    3. 内部布尔：`has_structured_financial_statements` / `has_financial_statement_sections`
    4. 兼容旧字段：`has_xbrl` / `has_financial_statement`

    Args:
        has_financial_data: 直接的 has_financial_data 字段。
        availability: 内部 availability 枚举。
        has_financial_statement: 旧能力布尔。
        has_xbrl: 旧 XBRL 能力布尔。
        has_structured_financial_statements: 内部结构化能力布尔。
        has_financial_statement_sections: 内部章节级能力布尔。

    Returns:
        `True`（可调用 get_financial_statement）/ `False`（无数据）/ `None`（无法判定）。

    Raises:
        无。
    """

    # 优先级 1：直接字段
    if has_financial_data is not None:
        return bool(has_financial_data)

    # 优先级 2：内部 availability 枚举
    norm_avail = normalize_optional_text(availability) if availability is not None else None
    if norm_avail in ("structured_data_available", "statement_sections_available"):
        return True
    if norm_avail == "not_available":
        return False

    # 优先级 3：内部布尔
    if has_structured_financial_statements is True:
        return True
    if has_financial_statement_sections is True:
        return True

    # 优先级 4：兼容旧字段
    if has_financial_statement is False:
        return False
    if has_xbrl is True:
        return True

    # 无法保守判定
    return None


def build_search_next_section_fields(
    *,
    matches: list[dict[str, Any]],
    queries: Optional[list[str]] = None,
 ) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Optional[dict[str, Any]]]]]:
    """基于搜索命中构建下一步阅读章节字段。

    Args:
        matches: `search_document` 的 matches 列表。
        queries: 多查询模式下的原始查询列表；单查询传 `None`。

    Returns:
        `(next_section_to_read, next_section_by_query)`。

        - 单查询时：`next_section_to_read` 为对象或 `None`，`next_section_by_query` 为 `None`
        - 多查询时：`next_section_to_read` 为 `None`，`next_section_by_query` 为 `query -> object|None` 映射

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    section_stats: dict[str, dict[str, Any]] = {}
    query_section_stats: dict[str, dict[str, dict[str, Any]]] = {}

    for index, match in enumerate(matches):
        if not isinstance(match, Mapping):
            continue
        section = match.get("section")
        if not isinstance(section, Mapping):
            continue
        section_ref = normalize_optional_text(section.get("ref"))
        if section_ref is None:
            continue
        matched_query = normalize_optional_text(match.get("matched_query"))
        is_exact_phrase = bool(match.get("is_exact_phrase"))
        stat = section_stats.setdefault(
            section_ref,
            {
                "section": {
                    "ref": section_ref,
                    "title": normalize_optional_text(section.get("title")),
                    "item": normalize_optional_text(section.get("item")),
                    "topic": normalize_optional_text(section.get("topic")),
                },
                "evidence_hit_count": 0,
                "_exact_match_count": 0,
                "_first_index": index,
            },
        )
        stat["evidence_hit_count"] += 1
        if is_exact_phrase:
            stat["_exact_match_count"] += 1

        if matched_query is not None:
            per_query = query_section_stats.setdefault(matched_query, {})
            per_query_stat = per_query.setdefault(
                section_ref,
                {
                    "section": stat["section"],
                    "evidence_hit_count": 0,
                    "_exact_match_count": 0,
                    "_first_index": index,
                },
            )
            per_query_stat["evidence_hit_count"] += 1
            if is_exact_phrase:
                per_query_stat["_exact_match_count"] += 1

    ranked_sections = sorted(
        section_stats.values(),
        key=lambda item: (
            -int(item["evidence_hit_count"]),
            -int(item["_exact_match_count"]),
            int(item["_first_index"]),
        ),
    )

    if queries is None:
        next_section_to_read = (
            _strip_search_section_internal_fields(ranked_sections[0])
            if ranked_sections else None
        )
        return next_section_to_read, None

    next_section_by_query: dict[str, Optional[dict[str, Any]]] = {}
    for query in queries:
        normalized_query = normalize_optional_text(query)
        if normalized_query is None:
            continue
        candidate_stats = query_section_stats.get(normalized_query, {})
        if not candidate_stats:
            next_section_by_query[normalized_query] = None
            continue
        next_section_by_query[normalized_query] = _strip_search_section_internal_fields(
            sorted(
                candidate_stats.values(),
                key=lambda item: (
                    -int(item["evidence_hit_count"]),
                    -int(item["_exact_match_count"]),
                    int(item["_first_index"]),
                ),
            )[0]
        )
    return None, next_section_by_query


def _strip_search_section_internal_fields(section_stat: dict[str, Any]) -> dict[str, Any]:
    """移除 search section 聚合内部字段。

    Args:
        section_stat: 含内部统计字段的 section 聚合字典。

    Returns:
        仅包含对 LLM 有决策价值字段的字典。

    Raises:
        RuntimeError: 清洗失败时抛出。
    """

    return {
        "section": dict(section_stat.get("section") or {}),
        "evidence_hit_count": int(section_stat.get("evidence_hit_count") or 0),
    }


# =====================================================================
# 章节标准化
# =====================================================================

def _normalize_section_children(raw_children: Any) -> list[dict[str, Any]]:
    """标准化 `read_section.children` 字段。

    仅保留 ref + title 最小导航字段：
    - ref：read_section 入参，LLM 必须持有。
    - title：帮助 LLM 判断子章节主题相关性。
    - level / preview / parent_ref 等已去除：
      level 恒等于 parent+1（零信息增量），preview 与 title 高重复
      （与 get_document_sections 去 preview 的 T1 决策对齐），
      parent_ref 就是当前 section（零信息增量）。

    Args:
        raw_children: 原始 children 值。

    Returns:
        标准化后的 children 列表；非法输入返回空列表。
    """

    if not isinstance(raw_children, list):
        return []
    normalized: list[dict[str, Any]] = []
    for child in raw_children:
        if not isinstance(child, Mapping):
            continue
        ref = normalize_optional_text(child.get("ref"))
        if ref is None:
            continue
        title = normalize_optional_text(child.get("title"))
        normalized.append({"ref": ref, "title": title})
    return normalized


def _normalize_periods(periods: Optional[list[str]]) -> Optional[list[str]]:
    """标准化财期数组。

    Args:
        periods: 原始财期数组。

    Returns:
        规范化财期数组；输入为空时返回 `None`。

    Raises:
        ToolArgumentError: 入参类型非法时抛出。
    """

    if periods is None:
        return None
    if not isinstance(periods, list):
        raise ToolArgumentError("list_documents", "fiscal_periods", periods, "Must be a string array")
    result: list[str] = []
    for period in periods:
        normalized = normalize_optional_text(period)
        if normalized is None:
            continue
        result.append(normalized)
    return result or None


def _build_not_supported_result(
    *,
    ticker: str,
    document_id: str,
    feature: str,
    payload: Optional[dict[str, Any]] = None,
) -> NotSupportedResult:
    """构建能力不支持结果。

    Args:
        ticker: 股票代码。
        document_id: 文档 ID。
        feature: 能力名称。
        payload: 额外回显字段。

    Returns:
        ``not_supported`` 结构化结果。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    message = f"Current document processor does not support feature: {feature}"
    result: dict[str, Any] = {
        "ticker": ticker,
        "document_id": document_id,
        "supported": False,
        "error": {
            "code": "not_supported",
            "message": message,
        },
    }
    if payload:
        result.update(payload)
    # 已知回显字段（page_no / statement_type / concepts）在 NotSupportedResult 中声明；
    # payload 来自各调用方的确定性字典，运行时结构已保证。
    return cast(NotSupportedResult, result)


def _extract_page_range(section: SectionSummary | SectionContent | Mapping[str, Any]) -> Optional[list[int]]:
    """从章节结构提取页码范围。

    Args:
        section: 章节结构对象。

    Returns:
        页码范围；不存在返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    raw = section.get("page_range")
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    start, end = raw
    if isinstance(start, int) and isinstance(end, int) and start > 0 and end > 0:
        return [start, end]
    return None


# =====================================================================
# 财务日期推断
# =====================================================================

def _infer_fiscal_period(meta: dict[str, Any]) -> Optional[str]:
    """推断财期。

    Args:
        meta: 文档元数据。

    Returns:
        财期字符串或 `None`。

    Raises:
        RuntimeError: 推断失败时抛出。
    """

    raw_period = normalize_optional_text(meta.get("fiscal_period"))
    if raw_period is not None:
        return raw_period

    form_type = normalize_optional_text(meta.get("form_type"))
    if form_type in {"10-K", "20-F"}:
        return "FY"
    return None


def _resolve_fiscal_year_with_fallback(raw_value: Any, inferred_year: Optional[int]) -> Optional[int]:
    """解析 fiscal_year，空值时回退到推断值。

    Args:
        raw_value: 源 meta 中的 fiscal_year 原始值。
        inferred_year: 由 `report_date` 等信息推断出的 fiscal_year。

    Returns:
        可用 fiscal_year；当原始值为空或非法时返回 `inferred_year`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    if isinstance(raw_value, bool):
        return inferred_year
    if isinstance(raw_value, int):
        return raw_value if raw_value > 0 else inferred_year
    text = normalize_optional_text(raw_value)
    if text is None:
        return inferred_year
    try:
        parsed = int(text)
    except ValueError:
        return inferred_year
    if parsed <= 0:
        return inferred_year
    return parsed


def _resolve_fiscal_period_with_fallback(raw_value: Any, inferred_period: Optional[str]) -> Optional[str]:
    """解析 fiscal_period，空值时回退到推断值。

    Args:
        raw_value: 源 meta 中的 fiscal_period 原始值。
        inferred_period: 推断出的 fiscal_period。

    Returns:
        可用 fiscal_period；原始值为空时返回 `inferred_period`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    normalized = normalize_optional_text(raw_value)
    if normalized is not None:
        return normalized
    return inferred_period


def _infer_fiscal_year(meta: dict[str, Any], fiscal_period: Optional[str]) -> Optional[int]:
    """推断财年。

    Args:
        meta: 文档元数据。
        fiscal_period: 已推断财期。

    Returns:
        财年或 `None`。

    Raises:
        RuntimeError: 推断失败时抛出。
    """

    raw_year = meta.get("fiscal_year")
    if isinstance(raw_year, int):
        return raw_year

    del fiscal_period
    return None


def _extract_year(iso_date: str) -> Optional[int]:
    """从 ISO 日期提取年份。

    Args:
        iso_date: ISO 日期字符串。

    Returns:
        年份整数；无法提取时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    parts = iso_date.split("-")
    if len(parts) < 2:
        return None
    try:
        year = int(parts[0])
    except ValueError:
        return None
    if year <= 0:
        return None
    return year


def _to_optional_float(value: Any) -> Optional[float]:
    """将任意值转换为可选浮点数。

    Args:
        value: 原始值。

    Returns:
        可解析时返回浮点值，否则返回 `None`。

    Raises:
        RuntimeError: 转换失败时抛出。
    """

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if numeric != numeric:  # NaN
        return None
    return numeric


# =====================================================================
# 表格标准化
# =====================================================================

def _build_table_data_payload(table_raw: TableContent | Mapping[str, Any]) -> dict[str, Any]:
    """构建 `get_table.data` 的自解释结构。

    Args:
        table_raw: 处理器返回的原始表格内容。

    Returns:
        统一后的 `data` 结构，固定为以下三类之一：
        - `records`：包含 `columns` 与 `rows`
        - `markdown`：包含 `markdown`
        - `raw_text`：包含 `text`

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    raw_format = normalize_optional_text(table_raw.get("data_format"))
    normalized_format = (raw_format or "unknown").lower()
    raw_data = table_raw.get("data")
    raw_columns = table_raw.get("columns")

    # 复杂逻辑说明：先按处理器声明格式分支，再对内容做兜底识别，避免 data_format 与 data 实际形态不一致。
    if normalized_format == "records":
        return _build_records_data_payload(
            raw_data=raw_data,
            raw_columns=raw_columns,
        )
    if normalized_format == "markdown":
        text = _coerce_table_text(raw_data)
        if _looks_like_markdown_table(text):
            return {
                "kind": "markdown",
                "description": "Markdown table text, ready to render.",
                "markdown": text,
            }
        return {
            "kind": "raw_text",
            "description": "Raw text content; does not meet standard Markdown table structure.",
            "text": text,
        }

    if isinstance(raw_data, list):
        return _build_records_data_payload(
            raw_data=raw_data,
            raw_columns=raw_columns,
        )
    text = _coerce_table_text(raw_data)
    if _looks_like_markdown_table(text):
        return {
            "kind": "markdown",
            "description": "Markdown table text, ready to render.",
            "markdown": text,
        }
    return {
        "kind": "raw_text",
        "description": "Raw text content; does not meet standard Markdown table structure.",
        "text": text,
    }


def _build_records_data_payload(
    *,
    raw_data: Any,
    raw_columns: Any,
) -> dict[str, Any]:
    """构建 `records` 类型的表格数据。

    Args:
        raw_data: 原始数据体，期望为记录数组。
        raw_columns: 原始列名信息。

    Returns:
        `records` 形态的数据体。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    rows = _normalize_table_rows(raw_data)
    columns = _normalize_table_columns(raw_columns, rows)
    return {
        "kind": "records",
        "description": "Structured table data; rows are row-level objects, columns define column order.",
        "columns": columns,
        "rows": rows,
    }


def _normalize_table_rows(raw_data: Any) -> list[dict[str, Any]]:
    """标准化 records 行数据。

    Args:
        raw_data: 原始行数据。

    Returns:
        归一化后的行对象列表。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    if not isinstance(raw_data, list):
        return []

    normalized_rows: list[dict[str, Any]] = []
    for row in raw_data:
        if isinstance(row, Mapping):
            normalized_row: dict[str, Any] = {}
            for key, value in row.items():
                normalized_key = normalize_optional_text(key) if key is not None else None
                normalized_row[normalized_key or str(key)] = value
            normalized_rows.append(normalized_row)
            continue
        if isinstance(row, list):
            indexed_row = {str(index): value for index, value in enumerate(row)}
            normalized_rows.append(indexed_row)
            continue
        normalized_rows.append({"value": row})
    return normalized_rows


def _normalize_table_columns(
    raw_columns: Any,
    rows: list[dict[str, Any]],
) -> list[str]:
    """标准化 records 列名列表。

    Args:
        raw_columns: 原始列名候选。
        rows: 已标准化行数据。

    Returns:
        列名列表。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized_columns: list[str] = []
    if isinstance(raw_columns, list):
        seen: set[str] = set()
        for column in raw_columns:
            if column is None:
                continue
            normalized = normalize_optional_text(column)
            if normalized is None or normalized in seen:
                continue
            seen.add(normalized)
            normalized_columns.append(normalized)
    if normalized_columns:
        return normalized_columns
    if not rows:
        return []
    return list(rows[0].keys())


def _coerce_table_text(raw_data: Any) -> str:
    """将任意表格内容兜底转换为文本。

    Args:
        raw_data: 原始数据体。

    Returns:
        文本形态内容。

    Raises:
        RuntimeError: 转换失败时抛出。
    """

    if raw_data is None:
        return ""
    if isinstance(raw_data, str):
        return raw_data
    return str(raw_data)


def _looks_like_markdown_table(text: str) -> bool:
    """判断文本是否接近 Markdown 表格结构。

    Args:
        text: 候选文本。

    Returns:
        `True` 表示可视为 Markdown 表格，`False` 表示更像普通原文文本。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    stripped = text.strip()
    if not stripped:
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    header_line = lines[0]
    separator_line = lines[1]
    if "|" not in header_line or "|" not in separator_line:
        return False
    return bool(re.match(r"^\|?[\s:\-|]+\|?$", separator_line))


def _normalize_table_type(raw_table_type: Any) -> Optional[str]:
    """标准化表格类型字段。

    Args:
        raw_table_type: 处理器输出的原始表格类型。

    Returns:
        合法类型字符串（`layout/data/financial`）或 `None`。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized = normalize_optional_text(raw_table_type)
    if normalized is None:
        return None
    lowered = normalized.lower()
    if lowered not in {"layout", "data", "financial"}:
        return None
    return lowered


# =====================================================================
# XBRL 辅助
# =====================================================================

def _resolve_processor_taxonomy(processor: Any) -> Optional[str]:
    """从处理器中读取 XBRL taxonomy。

    Args:
        processor: 处理器实例。

    Returns:
        标准化 taxonomy（`us-gaap` / `ifrs-full`）或 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    taxonomy_method = getattr(processor, "get_xbrl_taxonomy", None)
    if callable(taxonomy_method):
        try:
            return _normalize_taxonomy_name(taxonomy_method())
        except Exception:
            return None
    raw_taxonomy = getattr(processor, "xbrl_taxonomy", None)
    return _normalize_taxonomy_name(raw_taxonomy)


def _normalize_taxonomy_name(taxonomy: Any) -> Optional[str]:
    """标准化 taxonomy 名称。

    Args:
        taxonomy: 原始 taxonomy 值。

    Returns:
        标准化后的 taxonomy；未知时返回 `None`。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized = normalize_optional_text(taxonomy)
    if normalized is None:
        return None
    lowered = normalized.lower()
    if lowered.startswith("us-gaap"):
        return "us-gaap"
    if lowered.startswith("ifrs"):
        return "ifrs-full"
    return None


def _resolve_default_xbrl_concepts(*, form_type: Optional[str], taxonomy: Optional[str]) -> list[str]:
    """按 `(form_type, taxonomy)` 解析默认 concept 包。

    Args:
        form_type: 可选 SEC form。
        taxonomy: 可选 taxonomy。

    Returns:
        默认 concept 列表（非空）。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    normalized_form = normalize_optional_text(form_type)
    normalized_taxonomy = _normalize_taxonomy_name(taxonomy)
    if normalized_form and normalized_taxonomy:
        matched = _DEFAULT_XBRL_CONCEPTS_BY_FORM_TAXONOMY.get((normalized_form, normalized_taxonomy))
        if matched:
            return list(matched)
    if normalized_taxonomy:
        taxonomy_defaults = _DEFAULT_XBRL_CONCEPTS_BY_TAXONOMY.get(normalized_taxonomy)
        if taxonomy_defaults:
            return list(taxonomy_defaults)
    return list(_GLOBAL_DEFAULT_XBRL_CONCEPTS)


def _normalize_xbrl_query_payload(
    *,
    payload: Mapping[str, Any] | dict[str, Any],
    default_concepts: list[str],
) -> dict[str, Any]:
    """标准化 `query_xbrl_facts` 的输出载荷。

    Args:
        payload: 处理器返回载荷。
        default_concepts: 本次查询实际使用的概念列表。

    Returns:
        结构稳定、已去重且文本已清洗的载荷。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    query_params_raw = payload.get("query_params")
    query_params = dict(query_params_raw) if isinstance(query_params_raw, Mapping) else {}
    query_params["concepts"] = _normalize_concepts_for_query(query_params.get("concepts"), default_concepts)

    facts_raw = payload.get("facts")
    if not isinstance(facts_raw, list):
        facts_raw = []

    normalized_pairs: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for index, raw_fact in enumerate(facts_raw):
        if not isinstance(raw_fact, Mapping):
            continue
        normalized_fact = _normalize_single_fact(raw_fact)
        if normalized_fact is None:
            continue
        normalized_pairs.append((normalized_fact, dict(raw_fact), index))

    deduped_facts = _deduplicate_xbrl_facts(normalized_pairs)
    normalized_payload = dict(payload)
    normalized_payload["query_params"] = query_params
    normalized_payload["facts"] = deduped_facts
    normalized_payload["total"] = len(deduped_facts)
    return normalized_payload


def _normalize_concepts_for_query(raw_concepts: Any, default_concepts: list[str]) -> list[str]:
    """标准化查询概念列表。

    Args:
        raw_concepts: 原始概念字段。
        default_concepts: 默认概念列表。

    Returns:
        标准化后的概念列表（保证非空）。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    if not isinstance(raw_concepts, list):
        return list(default_concepts)
    normalized: list[str] = []
    for item in raw_concepts:
        concept = normalize_optional_text(item)
        if concept is None:
            continue
        normalized.append(concept)
    return normalized or list(default_concepts)


def _normalize_single_fact(raw_fact: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """标准化单条 fact。

    Args:
        raw_fact: 原始 fact 对象。

    Returns:
        标准化 fact；若不包含可用数值/文本则返回 `None`。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    concept = str(raw_fact.get("concept") or "")
    label = str(raw_fact.get("label") or raw_fact.get("original_label") or concept)
    numeric_value = _to_optional_float(raw_fact.get("numeric_value"))
    if numeric_value is None:
        numeric_value = _to_optional_float(raw_fact.get("value"))

    raw_text_value: Optional[str] = None
    candidate_text = raw_fact.get("text_value")
    if isinstance(candidate_text, str):
        raw_text_value = candidate_text
    elif isinstance(raw_fact.get("value"), str):
        raw_text_value = str(raw_fact.get("value"))

    text_value: Optional[str] = None
    content_type: Optional[str] = None
    if numeric_value is None and raw_text_value is not None:
        cleaned = _clean_fact_text_value(raw_text_value)
        text_value = cleaned or None
        if text_value is not None:
            content_type = "xhtml" if _looks_like_html_text(raw_text_value) else "plain"

    if numeric_value is None and text_value is None:
        return None

    # 解析 decimals 并推断 scale
    raw_decimals = raw_fact.get("decimals")
    decimals = _parse_xbrl_decimals_value(raw_decimals)
    scale = _infer_scale_from_decimals(decimals) if numeric_value is not None else None

    return {
        "concept": concept,
        "label": label,
        "numeric_value": numeric_value,
        "text_value": text_value,
        "content_type": content_type,
        "unit": raw_fact.get("unit") or raw_fact.get("unit_ref"),
        "decimals": decimals,
        "scale": scale,
        "period_type": raw_fact.get("period_type"),
        "period_start": raw_fact.get("period_start"),
        "period_end": raw_fact.get("period_end"),
        "fiscal_year": raw_fact.get("fiscal_year"),
        "fiscal_period": raw_fact.get("fiscal_period"),
        "statement_type": raw_fact.get("statement_type"),
    }


def _clean_fact_text_value(text: str) -> str:
    """清洗 fact 文本值（去标签、反转义、压缩空白）。

    Args:
        text: 原始文本。

    Returns:
        清洗后的可读文本。

    Raises:
        RuntimeError: 清洗失败时抛出。
    """

    if not text:
        return ""
    stripped = _HTML_TAG_PATTERN.sub(" ", text)
    unescaped = unescape(stripped)
    return re.sub(r"\s+", " ", unescaped).strip()


def _looks_like_html_text(text: str) -> bool:
    """判断文本是否包含 HTML/XHTML 标签。

    Args:
        text: 候选文本。

    Returns:
        若包含标签返回 `True`，否则返回 `False`。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if not text:
        return False
    return bool(_HTML_TAG_PATTERN.search(text))


def _deduplicate_xbrl_facts(
    normalized_pairs: list[tuple[dict[str, Any], dict[str, Any], int]]
) -> list[dict[str, Any]]:
    """按确定性策略去重 XBRL facts。

    去重键：`(canonical_concept, period_start, period_end, fiscal_year, dedup_fiscal_period, unit, segment_signature)`。
    其中当 `period_end` 存在时，`dedup_fiscal_period` 固定为空，避免同一期末仅因 fiscal_period 缺失而重复。
    保留优先级：数值型 > fiscal_period 非空 > statement_type 非空 > 有 segment > decimals 更优 > 先出现项。

    Args:
        normalized_pairs: `(normalized_fact, raw_fact, source_index)` 三元组列表。

    Returns:
        去重后的标准化 fact 列表。

    Raises:
        RuntimeError: 去重失败时抛出。
    """

    selected: dict[
        tuple[str, str, str, str, str, str, str],
        tuple[dict[str, Any], int, tuple[int, int, int, int, int]],
    ] = {}
    first_seen_index: dict[tuple[str, str, str, str, str, str, str], int] = {}

    for normalized_fact, raw_fact, source_index in normalized_pairs:
        dedup_key = _build_fact_dedup_key(normalized_fact, raw_fact)
        score = _build_fact_selection_score(normalized_fact, raw_fact)
        current = selected.get(dedup_key)
        if current is None or score > current[2]:
            selected[dedup_key] = (normalized_fact, source_index, score)
            if dedup_key not in first_seen_index:
                first_seen_index[dedup_key] = source_index
            continue
        if score == current[2] and source_index < current[1]:
            selected[dedup_key] = (normalized_fact, source_index, score)
            if dedup_key not in first_seen_index:
                first_seen_index[dedup_key] = source_index

    ordered_items = sorted(selected.items(), key=lambda item: first_seen_index.get(item[0], item[1][1]))
    return [item[1][0] for item in ordered_items]


def _build_fact_dedup_key(
    normalized_fact: Mapping[str, Any],
    raw_fact: Mapping[str, Any],
) -> tuple[str, str, str, str, str, str, str]:
    """构建 fact 去重键。

    Args:
        normalized_fact: 标准化 fact。
        raw_fact: 原始 fact。

    Returns:
        去重键元组（当存在 `period_end` 时，不使用 `fiscal_period` 区分）。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    canonical_concept = _canonicalize_concept(normalized_fact.get("concept"))
    period_start = str(raw_fact.get("period_start") or "")
    period_end = str(normalized_fact.get("period_end") or "")
    fiscal_year = str(normalized_fact.get("fiscal_year") or "")
    fiscal_period = str(normalized_fact.get("fiscal_period") or "")
    dedup_fiscal_period = fiscal_period if not period_end else ""
    unit = str(normalized_fact.get("unit") or "")
    segment_signature = _build_segment_signature(raw_fact.get("segment") or raw_fact.get("dimensions"))
    return (
        canonical_concept,
        period_start,
        period_end,
        fiscal_year,
        dedup_fiscal_period,
        unit,
        segment_signature,
    )


def _canonicalize_concept(concept: Any) -> str:
    """将 concept 归一化为 canonical key。

    Args:
        concept: 原始 concept。

    Returns:
        归一化本地名（小写）。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized = normalize_optional_text(concept) or ""
    if ":" in normalized:
        normalized = normalized.split(":")[-1]
    return normalized.strip().lower()


def _build_segment_signature(segment: Any) -> str:
    """构建 segment 稳定签名。

    Args:
        segment: segment/dimensions 原始对象。

    Returns:
        稳定签名字符串；空对象返回空字符串。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    if segment is None:
        return ""
    try:
        return json.dumps(segment, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(segment)


def _build_fact_selection_score(
    normalized_fact: Mapping[str, Any],
    raw_fact: Mapping[str, Any],
) -> tuple[int, int, int, int, int]:
    """构建 fact 保留优先级评分。

    Args:
        normalized_fact: 标准化 fact。
        raw_fact: 原始 fact。

    Returns:
        可比较评分元组；值越大优先级越高。

    Raises:
        RuntimeError: 评分失败时抛出。
    """

    numeric_score = 1 if normalized_fact.get("numeric_value") is not None else 0
    fiscal_period_score = 1 if normalize_optional_text(normalized_fact.get("fiscal_period")) else 0
    statement_type_score = 1 if normalize_optional_text(normalized_fact.get("statement_type")) else 0
    segment_score = 1 if _build_segment_signature(raw_fact.get("segment") or raw_fact.get("dimensions")) else 0
    precision_score = _parse_xbrl_decimals(raw_fact.get("decimals"))
    return (numeric_score, fiscal_period_score, statement_type_score, segment_score, precision_score)


def _parse_xbrl_decimals(raw_decimals: Any) -> int:
    """解析 XBRL decimals 精度评分。

    Args:
        raw_decimals: 原始 decimals 值。

    Returns:
        精度评分，越大表示精度越高。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    if raw_decimals is None:
        return -100000
    if isinstance(raw_decimals, str) and raw_decimals.strip().upper() == "INF":
        return 100000
    try:
        return int(str(raw_decimals).strip())
    except ValueError:
        return -100000


def _parse_xbrl_decimals_value(raw_decimals: Any) -> Optional[int]:
    """解析 XBRL decimals 为实际整数值。

    与 ``_parse_xbrl_decimals`` 不同，本函数返回原始语义值而非评分值。
    ``INF`` 返回 ``None``（表示无限精度）。

    Args:
        raw_decimals: 原始 decimals 值（int / str / None）。

    Returns:
        整数 decimals 或 ``None``。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    if raw_decimals is None:
        return None
    if isinstance(raw_decimals, str) and raw_decimals.strip().upper() == "INF":
        return None
    try:
        return int(str(raw_decimals).strip())
    except ValueError:
        return None


# decimals → scale 映射表
_DECIMALS_SCALE_MAP: dict[int, str] = {
    -9: "billions",
    -6: "millions",
    -3: "thousands",
    0: "units",
}


# ---------------------------------------------------------------------------
# match_quality 匹配质量标签常量
# ---------------------------------------------------------------------------
_MATCH_QUALITY_EXACT = "exact"
_MATCH_QUALITY_MIXED = "mixed"
_MATCH_QUALITY_EXPANSION_ONLY = "expansion_only"
_MATCH_QUALITY_NONE = "none"


def _build_match_quality(
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    """根据 post-cap 匹配列表构建面向 LLM 的匹配质量摘要。

    计数基于 exact 优先限流后的实际匹配列表，确保
    ``exact_phrase_matches + expansion_matches == total_matches``，
    消除跨语义层级的认知歧义。

    Args:
        matches: 经过排序、去重、exact 优先限流后的匹配列表
            （每条含 ``is_exact_phrase`` 字段）。

    Returns:
        包含 ``exact_phrase_matches``、``expansion_matches``、``primary_source``
        三个字段的字典。
    """
    exact_count = sum(1 for m in matches if m.get("is_exact_phrase"))
    expansion_count = len(matches) - exact_count

    if not matches:
        primary_source = _MATCH_QUALITY_NONE
    elif exact_count > 0 and expansion_count == 0:
        primary_source = _MATCH_QUALITY_EXACT
    elif exact_count == 0 and expansion_count > 0:
        primary_source = _MATCH_QUALITY_EXPANSION_ONLY
    else:
        primary_source = _MATCH_QUALITY_MIXED

    return {
        "exact_phrase_matches": exact_count,
        "expansion_matches": expansion_count,
        "primary_source": primary_source,
    }


def _extract_top_section_ref(matches: list[dict[str, Any]]) -> Optional[str]:
    """从 matches 列表取第一个有效 section ref。

    Args:
        matches: search_document 的 matches 列表。

    Returns:
        第一个非空 section ref；无可用结果时返回 ``None``。
    """
    for match in matches:
        if not isinstance(match, Mapping):
            continue
        section = match.get("section")
        if not isinstance(section, Mapping):
            continue
        ref = normalize_optional_text(section.get("ref"))
        if ref:
            return ref
    return None


def _build_search_hint(
    matches: list[dict[str, Any]],
    primary_source: str,
) -> Optional[str]:
    """根据匹配质量生成面向 LLM 的操作引导提示。

    引导原则：在最低认知负担下让 LLM 做对下一步动作。
    仅在结果存在噪声风险或 fetch_more 价值低时生成提示。

    Args:
        matches: post-cap 匹配列表。
        primary_source: 匹配质量来源标签（exact / mixed / expansion_only / none）。

    Returns:
        提示字符串；无需提示时返回 ``None``。
    """
    if len(matches) > 40:
        top_ref = _extract_top_section_ref(matches)
        if top_ref:
            return (
                f"当前命中 {len(matches)} 条，范围过大。"
                f"若继续搜索，下一次必须带 `within_section_ref` 收窄；"
                f"或先直接调用 read_section(ref='{top_ref}') 读取最相关章节，再决定是否继续检索。"
            )
        return (
            f"当前命中 {len(matches)} 条，范围过大。"
            "若继续搜索，下一次必须带 `within_section_ref` 收窄；"
            "或先直接读取 `next_section_to_read` / `next_section_by_query` 指向的最相关章节。"
        )
    if primary_source in (_MATCH_QUALITY_NONE, _MATCH_QUALITY_EXACT):
        return None
    if primary_source == _MATCH_QUALITY_EXPANSION_ONLY:
        # 取 top match 的 section ref，直接给出可执行的 read_section 指令
        top_ref = _extract_top_section_ref(matches)
        if top_ref:
            return (
                f"目标：先读最相关章节。允许动作：直接调用 read_section(ref='{top_ref}')。"
                f"不允许：逐条阅读这 {len(matches)} 条扩展命中，也不要先 fetch_more。"
                "下一步：直接读取最相关章节。"
            )
        return (
            f"目标：先读最相关章节。允许动作：直接使用 next_section_to_read 或 next_section_by_query 里的 ref 调 read_section。"
            f"不允许：逐条阅读这 {len(matches)} 条扩展命中，也不要先 fetch_more。"
            "下一步：直接读取最相关章节。"
        )
    # mixed
    exact_count = sum(1 for m in matches if m.get("is_exact_phrase"))
    top_ref = _extract_top_section_ref(matches)
    next_step = (
        f"下一步：先调用 read_section(ref='{top_ref}')。"
        if top_ref
        else "下一步：先读取最相关命中的章节。"
    )
    return (
        f"目标：先读最相关的精确命中。允许动作：先读取前面的精确命中。"
        f"不允许：先 fetch_more 去枚举后面的低相关扩展命中。"
        f"{next_step} 只有在你必须枚举全部出现位置时再 fetch_more。"
    )


def _infer_scale_from_decimals(decimals: Optional[int]) -> Optional[str]:
    """根据 XBRL decimals 推断数值 scale。

    映射规则：
    - ``-9`` → ``"billions"``
    - ``-6`` → ``"millions"``
    - ``-3`` → ``"thousands"``
    - ``0`` 或正数 → ``"units"``
    - ``None`` 或其他负数 → ``None``

    Args:
        decimals: 已解析的 decimals 值。

    Returns:
        scale 描述字符串或 ``None``。

    Raises:
        RuntimeError: 推断失败时抛出。
    """

    if decimals is None:
        return None
    exact = _DECIMALS_SCALE_MAP.get(decimals)
    if exact is not None:
        return exact
    # 正数表示小数位数，即原始单位
    if decimals > 0:
        return "units"
    return None
