"""SEC 文档表格提取、渲染与 records 构建。

本模块承接 ``sec_processor.py`` 中表格相关的全部逻辑，包括：
- 表格内部数据结构（``_TableDataFrameProvider``、``_TableBlock``）
- 表格构建流程（``_build_tables`` 及其依赖链）
- 表格章节匹配与消歧
- 表格渲染（records / markdown / HTML 三路径）
- DataFrame / HTML / Markdown → records 转换
- 财务表判定与表格类型分类

维护说明(不拆分本模块):
    本模块约 2200 行, 内部可识别 section matching / header extraction /
    classification / render 四个区域, 但核心入口 _build_tables 同时
    调用三个区域共 11 个函数, 拆分子模块会引入大量跨模块 import 而
    无法降低耦合. 外部消费者仅 4 个文件, API 表面很小.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import pandas as pd
from pandas.errors import PerformanceWarning
from bs4 import BeautifulSoup

from dayu.engine.processors.text_utils import (
    PREVIEW_MAX_CHARS as _PREVIEW_MAX_CHARS,
    clean_page_header_noise as _clean_page_header_noise,
    format_table_placeholder as _format_table_placeholder,
    format_table_ref as _format_table_ref,
    infer_caption_from_context as _infer_caption_from_context,
    normalize_optional_string as _normalize_optional_string_base,
    normalize_whitespace as _normalize_whitespace,
)
from dayu.fins.processors.sec_html_rules import is_sec_cover_page_table, is_sec_section_heading_table
from dayu.fins.processors.html_financial_statement_common import normalize_numeric_separators
from dayu.fins.processors.sec_section_build import (
    _SectionBlock,
    _normalize_searchable_text,
    _normalize_table_objects,
    _safe_table_text,
    _table_fingerprint,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOW_INFO_TOKENS = {"-", "--", "—", "n/a", "na", "none", "nil", "☐", "☒", "nan"}
_NUMERIC_LIKE_PATTERN = re.compile(r"^[\d\s,\.\-\+\(\)%$¥€]+$")
_STRICT_NUMERIC_TEXT_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
_MARKDOWN_SEPARATOR_PATTERN = re.compile(r"^\|?[\s:\-|]+\|?$")
_GENERATED_COLUMN_PATTERN = re.compile(r"^col_\d+$")
_CORE_FINANCIAL_TABLE_KEYWORDS = (
    "consolidated statements of operations",
    "consolidated statement of operations",
    "consolidated statements of income",
    "consolidated statement of income",
    "consolidated balance sheets",
    "consolidated balance sheet",
    "consolidated statements of cash flows",
    "consolidated statement of cash flows",
    "consolidated statements of shareholders",
    "consolidated statements of stockholders",
    "statement of changes in equity",
    "statement of comprehensive income",
    "comprehensive income",
)
_FINANCIAL_NEGATIVE_KEYWORDS = (
    "securities registered pursuant to section 12(b)",
    "title of each class",
    "trading symbol(s)",
    "name of each exchange on which registered",
    "large accelerated filer",
    "accelerated filer",
    "smaller reporting company",
    "emerging growth company",
    "check whether the registrant",
    "incorporated by reference",
    "exhibit number",
    "exhibit description",
)
_CONTEXT_TAIL_MARKER_WORD_COUNTS = (24, 20, 16, 12, 8)
_CONTEXT_TAIL_MARKER_MIN_CHARS = 48
_TABLE_FINGERPRINT_MAX_CHARS = 240
_TABLE_DISAMBIGUATION_MARKER_CHARS = (720, 480, 360, 300)
_CURRENCY_SYMBOL_TOKENS = frozenset({"$", "¥", "€", "£", "%", "%%"})


# ---------------------------------------------------------------------------
# Helper: pandas NaN-aware normalize_optional_string
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _TableDataFrameProvider:
    """按需读取并缓存单表 DataFrame。"""

    table_obj: Any
    _dataframe: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)
    _resolved: bool = field(default=False, init=False, repr=False)

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """返回表格 DataFrame，并保证单表最多解析一次。

        Args:
            无。

        Returns:
            DataFrame；解析失败或不可用时返回 `None`。

        Raises:
            RuntimeError: DataFrame 读取失败时由底层 helper 处理。
        """

        if not self._resolved:
            self._dataframe = _safe_table_dataframe(self.table_obj)
            self._resolved = True
        return self._dataframe


@dataclass
class _TableBlock:
    """内部表格结构。"""

    ref: str
    table_obj: Any
    text: str
    fingerprint: str
    caption: Optional[str]
    row_count: int
    col_count: int
    headers: Optional[list[str]]
    section_ref: Optional[str]
    context_before: str
    is_financial: bool
    table_type: str
    dataframe: Optional[pd.DataFrame] = None
    dataframe_provider: Optional[_TableDataFrameProvider] = field(default=None, repr=False)

    def resolve_dataframe(self) -> Optional[pd.DataFrame]:
        """按需解析并缓存当前表格的 DataFrame。

        Args:
            无。

        Returns:
            DataFrame；若不可解析则返回 `None`。

        Raises:
            RuntimeError: DataFrame 读取失败时由底层 helper 处理。
        """

        if self.dataframe is not None:
            return self.dataframe
        if self.dataframe_provider is None:
            return None
        self.dataframe = self.dataframe_provider.get_dataframe()
        return self.dataframe


# ---------------------------------------------------------------------------
# Safe DataFrame helpers
# ---------------------------------------------------------------------------


def _safe_table_dataframe(table_obj: Any) -> Optional[pd.DataFrame]:
    """安全读取表格 DataFrame。

    Args:
        table_obj: 表格对象。

    Returns:
        DataFrame 或 `None`。

    Raises:
        RuntimeError: 转换失败时抛出。
    """

    if not hasattr(table_obj, "to_dataframe"):
        return None
    try:
        # edgartools 在部分复杂表格上会触发 pandas PerformanceWarning，
        # 该告警只反映性能风险，不影响解析正确性；这里做最小范围屏蔽。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerformanceWarning)
            df = table_obj.to_dataframe()
    except Exception:
        return None
    if isinstance(df, pd.DataFrame):
        return df
    return None


def _safe_statement_dataframe(statement_obj: Any) -> Optional[pd.DataFrame]:
    """安全读取财务报表 DataFrame。

    Args:
        statement_obj: 报表对象。

    Returns:
        DataFrame 或 `None`。

    Raises:
        RuntimeError: 转换失败时抛出。
    """

    if not hasattr(statement_obj, "to_dataframe"):
        return None
    try:
        # 财务报表 DataFrame 转换同样可能触发 pandas PerformanceWarning，
        # 与业务正确性无关，局部屏蔽以避免日志噪声。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=PerformanceWarning)
            df = statement_obj.to_dataframe()
    except Exception:
        return None
    if isinstance(df, pd.DataFrame):
        return df
    return None


# ---------------------------------------------------------------------------
# Document iteration
# ---------------------------------------------------------------------------


def _iter_document_tables(document: Any) -> list[Any]:
    """安全遍历文档表格。

    Args:
        document: edgartools 文档对象。

    Returns:
        表格对象列表。

    Raises:
        RuntimeError: 访问失败时抛出。
    """

    tables_obj = getattr(document, "tables", None)
    if tables_obj is None:
        return []
    if isinstance(tables_obj, list):
        return list(tables_obj)
    try:
        return list(tables_obj)
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------


def _build_tables(
    document: Any,
    sections: list[_SectionBlock],
    dom_table_contexts: Optional[list[str]] = None,
) -> list[_TableBlock]:
    """从文档对象构建表格列表并挂载 section 关系。

    Args:
        document: edgartools 文档对象。
        sections: 章节块列表。

    Returns:
        表格块列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    section_by_ref = {section.ref: section for section in sections}
    fingerprint_to_sections = _build_fingerprint_section_mapping(sections)
    default_section_ref = _get_default_section_ref(sections)
    tables: list[_TableBlock] = []

    for index, table_obj in enumerate(_iter_document_tables(document), start=1):
        table_text = _normalize_whitespace(_safe_table_text(table_obj))
        caption = _normalize_optional_string(getattr(table_obj, "caption", None))
        fingerprint = _table_fingerprint(table_text)
        dataframe_provider = _TableDataFrameProvider(table_obj)
        row_count, col_count = _resolve_table_dimensions(table_obj, dataframe_provider)
        headers = _extract_table_headers(table_obj, dataframe_provider)
        dom_context_before = _resolve_dom_context_by_index(dom_table_contexts, index)
        section_ref = _match_section_ref(
            fingerprint=fingerprint,
            fingerprint_to_sections=fingerprint_to_sections,
            default_section_ref=default_section_ref,
            section_by_ref=section_by_ref,
            table_text=table_text,
            dom_context_before=dom_context_before,
        )
        context_before = _extract_context_before(
            section_ref=section_ref,
            section_by_ref=section_by_ref,
            table_text=table_text,
            dom_context_before=dom_context_before,
        )
        # Step 10/12: 自适应清除 context_before 中的页眉页脚噪声
        context_before = _clean_page_header_noise(context_before)
        # Step 7: 当 edgartools 未提供 caption 时从前文推断
        if caption is None and context_before:
            caption = _infer_caption_from_context(context_before)
        is_financial = _is_financial_table(
            table_obj,
            table_text=table_text,
            caption=caption,
            context_before=context_before,
        )
        table_type = _classify_table_type(
            is_financial=is_financial,
            row_count=row_count,
            col_count=col_count,
            headers=headers,
            table_text=table_text,
        )
        table_block = _TableBlock(
            ref=_format_table_ref(index),
            table_obj=table_obj,
            text=table_text,
            fingerprint=fingerprint,
            caption=caption,
            row_count=row_count,
            col_count=col_count,
            headers=headers,
            section_ref=section_ref,
            context_before=context_before,
            is_financial=is_financial,
            table_type=table_type,
            dataframe_provider=dataframe_provider,
        )
        tables.append(table_block)
        if section_ref and section_ref in section_by_ref:
            section_by_ref[section_ref].table_refs.append(table_block.ref)
    return tables


# ---------------------------------------------------------------------------
# Section matching
# ---------------------------------------------------------------------------


def _build_fingerprint_section_mapping(sections: list[_SectionBlock]) -> dict[str, list[str]]:
    """构建表格指纹到章节 ref 的映射。

    Args:
        sections: 章节块列表。

    Returns:
        指纹映射。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    mapping: dict[str, list[str]] = {}
    for section in sections:
        for fingerprint in section.table_fingerprints:
            if not fingerprint:
                continue
            mapping.setdefault(fingerprint, []).append(section.ref)
    return mapping


def _match_section_ref(
    fingerprint: str,
    fingerprint_to_sections: dict[str, list[str]],
    default_section_ref: Optional[str],
    section_by_ref: Optional[dict[str, _SectionBlock]] = None,
    table_text: str = "",
    dom_context_before: str = "",
) -> Optional[str]:
    """根据表格指纹匹配 section ref。

    Args:
        fingerprint: 表格文本指纹。
        fingerprint_to_sections: 指纹映射。
        default_section_ref: 默认章节 ref。
        section_by_ref: 章节映射。
        table_text: 表格全文文本。
        dom_context_before: DOM 维度提取的前文。

    Returns:
        section ref 或 `None`。

    Raises:
        RuntimeError: 匹配失败时抛出。
    """

    if fingerprint:
        refs = fingerprint_to_sections.get(fingerprint, [])
        if len(refs) == 1:
            return refs[0]
        if refs:
            resolved_ref = _resolve_ambiguous_section_ref(
                candidate_refs=refs,
                section_by_ref=section_by_ref,
                table_text=table_text,
                dom_context_before=dom_context_before,
            )
            if resolved_ref is not None:
                return resolved_ref
    return default_section_ref


def _resolve_ambiguous_section_ref(
    candidate_refs: Sequence[str],
    section_by_ref: Optional[dict[str, _SectionBlock]],
    table_text: str,
    dom_context_before: str,
) -> Optional[str]:
    """在指纹冲突时尝试用更强信号消歧章节归属。

    Args:
        candidate_refs: 指纹候选章节列表。
        section_by_ref: 章节映射。
        table_text: 表格全文文本。
        dom_context_before: DOM 前文。

    Returns:
        唯一命中的章节 ref；无法确认时返回 `None`。

    Raises:
        RuntimeError: 消歧失败时抛出。
    """

    if not candidate_refs or not section_by_ref:
        return None

    resolved_ref = _match_unique_candidate_section_ref(
        candidate_refs=candidate_refs,
        section_by_ref=section_by_ref,
        markers=_build_table_text_disambiguation_markers(table_text),
    )
    if resolved_ref is not None:
        return resolved_ref

    return _match_unique_candidate_section_ref(
        candidate_refs=candidate_refs,
        section_by_ref=section_by_ref,
        markers=_build_context_tail_markers(dom_context_before),
    )


def _match_unique_candidate_section_ref(
    candidate_refs: Sequence[str],
    section_by_ref: dict[str, _SectionBlock],
    markers: Sequence[str],
) -> Optional[str]:
    """使用一组文本 marker 查找唯一匹配的章节候选。

    Args:
        candidate_refs: 候选章节 ref。
        section_by_ref: 章节映射。
        markers: 候选 marker 列表。

    Returns:
        唯一命中的章节 ref；否则返回 `None`。

    Raises:
        RuntimeError: 匹配失败时抛出。
    """

    if not candidate_refs or not markers:
        return None

    normalized_section_texts: dict[str, str] = {}
    for ref in candidate_refs:
        section = section_by_ref.get(ref)
        if section is None:
            continue
        normalized_section_texts[ref] = _normalize_searchable_text(section.text)

    if not normalized_section_texts:
        return None

    for marker in markers:
        matched_refs = [
            ref
            for ref in candidate_refs
            if marker and marker in normalized_section_texts.get(ref, "")
        ]
        if len(matched_refs) == 1:
            return matched_refs[0]
    return None


def _build_table_text_disambiguation_markers(table_text: str) -> list[str]:
    """为冲突表格生成比指纹更强的正文 marker。

    Args:
        table_text: 表格全文文本。

    Returns:
        从长到短排序的 marker 列表。

    Raises:
        RuntimeError: marker 构建失败时抛出。
    """

    normalized = _normalize_searchable_text(table_text)
    if len(normalized) <= _TABLE_FINGERPRINT_MAX_CHARS:
        return []

    marker_lengths = {len(normalized)}
    marker_lengths.update(
        max_chars
        for max_chars in _TABLE_DISAMBIGUATION_MARKER_CHARS
        if len(normalized) >= max_chars > _TABLE_FINGERPRINT_MAX_CHARS
    )

    markers: list[str] = []
    seen: set[str] = set()
    for marker_len in sorted(marker_lengths, reverse=True):
        marker = normalized[:marker_len].strip()
        if len(marker) <= _TABLE_FINGERPRINT_MAX_CHARS or marker in seen:
            continue
        seen.add(marker)
        markers.append(marker)
    return markers


def _build_context_tail_markers(context_text: str) -> list[str]:
    """为 DOM 前文构建尾部 marker，用于表格章节消歧。

    Args:
        context_text: 表格前文。

    Returns:
        marker 列表；无有效内容时返回空列表。

    Raises:
        RuntimeError: marker 构建失败时抛出。
    """

    normalized = _normalize_searchable_text(context_text)
    if not normalized:
        return []

    words = normalized.split()
    if not words:
        return []

    markers: list[str] = []
    seen: set[str] = set()
    for word_count in _CONTEXT_TAIL_MARKER_WORD_COUNTS:
        if len(words) < word_count:
            continue
        marker = " ".join(words[-word_count:]).strip()
        if len(marker) < _CONTEXT_TAIL_MARKER_MIN_CHARS or marker in seen:
            continue
        seen.add(marker)
        markers.append(marker)

    if markers:
        return markers

    fallback_marker = " ".join(words[-min(8, len(words)):]).strip()
    if len(fallback_marker) >= 36:
        return [fallback_marker]
    return []


def _get_default_section_ref(sections: list[_SectionBlock]) -> Optional[str]:
    """计算默认 section ref。

    Args:
        sections: 章节块列表。

    Returns:
        默认章节 ref 或 `None`。

    Raises:
        RuntimeError: 计算失败时抛出。
    """

    if len(sections) != 1:
        return None
    section = sections[0]
    if section.contains_full_text:
        return section.ref
    return None


def _resolve_dom_context_by_index(
    dom_table_contexts: Optional[list[str]],
    table_index: int,
) -> str:
    """按表格序号获取 DOM 上下文。

    Args:
        dom_table_contexts: DOM 上下文列表。
        table_index: 1-based 表格序号。

    Returns:
        上下文字符串。

    Raises:
        RuntimeError: 读取失败时抛出。
    """

    if not dom_table_contexts:
        return ""
    idx = table_index - 1
    if idx < 0 or idx >= len(dom_table_contexts):
        return ""
    return _normalize_whitespace(dom_table_contexts[idx])


def _extract_context_before(
    section_ref: Optional[str],
    section_by_ref: dict[str, _SectionBlock],
    table_text: str,
    dom_context_before: str = "",
) -> str:
    """提取表格前文。

    Args:
        section_ref: 所属章节引用。
        section_by_ref: 章节映射。
        table_text: 表格文本。
        dom_context_before: DOM 维度提取的前文。

    Returns:
        前文（最多 200 字）。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    if dom_context_before:
        return dom_context_before[:_PREVIEW_MAX_CHARS]
    if section_ref is None or not table_text:
        return ""
    section = section_by_ref.get(section_ref)
    if section is None:
        return ""

    marker = _build_marker_text(table_text, max_chars=120)
    if not marker:
        return ""
    index = section.text.lower().find(marker.lower())
    if index <= 0:
        return ""
    return section.text[max(0, index - _PREVIEW_MAX_CHARS):index].strip()


def _build_marker_text(text: str, max_chars: int) -> str:
    """构建定位用文本标记。

    Args:
        text: 原始文本。
        max_chars: 最大字符数。

    Returns:
        文本标记。

    Raises:
        ValueError: 参数非法时抛出。
    """

    normalized = _normalize_whitespace(text)
    if not normalized:
        return ""
    return normalized[:max_chars]


# ---------------------------------------------------------------------------
# Table dimensions and headers
# ---------------------------------------------------------------------------


def _resolve_table_dimensions(
    table_obj: Any,
    table_df: Optional[pd.DataFrame | _TableDataFrameProvider],
) -> tuple[int, int]:
    """解析表格行列数。

    Args:
        table_obj: 表格对象。
        table_df: 可选 DataFrame。

    Returns:
        `(row_count, col_count)`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    raw_rows = int(getattr(table_obj, "row_count", 0) or 0)
    raw_cols = int(getattr(table_obj, "col_count", 0) or 0)
    if raw_rows > 0 and raw_cols > 0:
        return raw_rows, raw_cols
    resolved_table_df = _resolve_table_dataframe(table_df)
    if resolved_table_df is None:
        return max(raw_rows, 0), max(raw_cols, 0)
    return int(resolved_table_df.shape[0]), int(resolved_table_df.shape[1])


def _extract_table_headers(
    table_obj: Any,
    table_df: Optional[pd.DataFrame | _TableDataFrameProvider],
) -> Optional[list[str]]:
    """提取表格行头（复用 `headers` 字段）。

    Args:
        table_obj: 表格对象。
        table_df: 可选 DataFrame。

    Returns:
        表头列表或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    row_headers_from_obj = _extract_row_headers_from_table_object(table_obj)
    if row_headers_from_obj:
        return row_headers_from_obj

    row_headers_from_dict = _extract_row_headers_from_table_dict(table_obj)
    if row_headers_from_dict:
        return row_headers_from_dict

    resolved_table_df = _resolve_table_dataframe(table_df)
    row_headers_from_df = _extract_row_headers_from_dataframe(resolved_table_df)
    if row_headers_from_df:
        return row_headers_from_df

    headers_from_obj = _extract_headers_from_table_object(table_obj)
    if headers_from_obj:
        return headers_from_obj

    headers_from_dict = _extract_headers_from_table_dict(table_obj)
    if headers_from_dict:
        return headers_from_dict

    if resolved_table_df is not None:
        headers_from_df = _extract_headers_from_dataframe(resolved_table_df)
        if headers_from_df:
            return headers_from_df

    return None


def _resolve_table_dataframe(
    table_df: Optional[pd.DataFrame | _TableDataFrameProvider],
) -> Optional[pd.DataFrame]:
    """收敛直接 DataFrame 或惰性 DataFrame 提供者。

    Args:
        table_df: DataFrame 或惰性提供者。

    Returns:
        DataFrame；不可用时返回 `None`。

    Raises:
        RuntimeError: DataFrame 解析失败时由底层 helper 处理。
    """

    if isinstance(table_df, _TableDataFrameProvider):
        return table_df.get_dataframe()
    return table_df


def _extract_row_headers_from_dataframe(table_df: Optional[pd.DataFrame]) -> Optional[list[str]]:
    """从 DataFrame 提取行头。

    Args:
        table_df: DataFrame。

    Returns:
        行头列表或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    if table_df is None or table_df.empty:
        return None

    row_headers: list[str] = []
    for row in table_df.itertuples(index=False, name=None):
        header = _pick_row_header_from_values(list(row))
        if header:
            row_headers.append(header)
    normalized = _normalize_header_list(row_headers)
    if not normalized or _looks_like_default_headers(normalized):
        return None
    return normalized


def _extract_row_headers_from_table_dict(table_obj: Any) -> Optional[list[str]]:
    """从 `table.to_dict()` 提取行头。

    Args:
        table_obj: 表格对象。

    Returns:
        行头列表或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    if not hasattr(table_obj, "to_dict"):
        return None
    try:
        table_dict = table_obj.to_dict()
    except Exception:
        return None
    if not isinstance(table_dict, dict):
        return None
    data_rows = table_dict.get("data")
    if not isinstance(data_rows, list):
        return None

    row_headers: list[str] = []
    for row in data_rows:
        if isinstance(row, dict):
            values = list(row.values())
        elif isinstance(row, list):
            values = row
        else:
            values = [row]
        header = _pick_row_header_from_values(values)
        if header:
            row_headers.append(header)
    normalized = _normalize_header_list(row_headers)
    if not normalized or _looks_like_default_headers(normalized):
        return None
    return normalized


def _extract_row_headers_from_table_object(table_obj: Any) -> Optional[list[str]]:
    """从 `table.headers` 结构提取可能的行头。

    Args:
        table_obj: 表格对象。

    Returns:
        行头列表或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    headers = getattr(table_obj, "headers", None)
    if not isinstance(headers, list) or len(headers) <= 1:
        return None

    row_headers: list[str] = []
    for row in headers:
        if isinstance(row, list):
            values = [_extract_cell_content(cell) for cell in row]
        else:
            values = [_extract_cell_content(row)]
        header = _pick_row_header_from_values(values)
        if header:
            row_headers.append(header)
    normalized = _normalize_header_list(row_headers)
    if not normalized or _looks_like_default_headers(normalized):
        return None
    return normalized


def _extract_headers_from_table_object(table_obj: Any) -> Optional[list[str]]:
    """从 table.headers 提取表头。

    Args:
        table_obj: 表格对象。

    Returns:
        表头列表或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    headers = getattr(table_obj, "headers", None)
    if not isinstance(headers, list) or not headers:
        return None

    flat_headers: list[str] = []
    for row in headers:
        if isinstance(row, list):
            for cell in row:
                content = _extract_cell_content(cell)
                if content:
                    flat_headers.append(content)
        else:
            content = _extract_cell_content(row)
            if content:
                flat_headers.append(content)
    normalized = _normalize_header_list(flat_headers)
    if not normalized:
        return None
    return normalized


def _extract_headers_from_dataframe(table_df: pd.DataFrame) -> Optional[list[str]]:
    """从 DataFrame 列名提取表头。

    Args:
        table_df: DataFrame。

    Returns:
        表头列表或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    columns = [_normalize_optional_string(column) for column in table_df.columns]
    normalized = _normalize_header_list(columns)
    if not normalized:
        return None
    # 若去重后列数变化，records 路径将无法安全对齐，回退为 None。
    if len(normalized) != len(columns):
        return None
    if _looks_like_default_headers(normalized):
        return None
    return normalized


def _extract_headers_from_table_dict(table_obj: Any) -> Optional[list[str]]:
    """从 `table.to_dict()` 提取表头。

    Args:
        table_obj: 表格对象。

    Returns:
        表头列表或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    if not hasattr(table_obj, "to_dict"):
        return None
    try:
        table_dict = table_obj.to_dict()
    except Exception:
        return None
    if not isinstance(table_dict, dict):
        return None
    headers = table_dict.get("headers")
    if not isinstance(headers, list):
        return None
    flat_headers: list[str] = []
    for row in headers:
        if isinstance(row, list):
            for cell in row:
                content = _extract_cell_content(cell)
                if content:
                    flat_headers.append(content)
        else:
            content = _extract_cell_content(row)
            if content:
                flat_headers.append(content)
    normalized = _normalize_header_list(flat_headers)
    if not normalized:
        return None
    return normalized


def _extract_cell_content(cell: Any) -> str:
    """提取单元格可读内容。

    Args:
        cell: 单元格对象或字符串。

    Returns:
        单元格文本。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    if cell is None:
        return ""
    if isinstance(cell, str):
        return _normalize_whitespace(cell)
    if hasattr(cell, "content"):
        return _normalize_whitespace(str(getattr(cell, "content", "")))
    return _normalize_whitespace(str(cell))


def _pick_row_header_from_values(values: list[Any]) -> str:
    """从一行值中挑选最有信息量的行头。

    Args:
        values: 行内候选值列表。

    Returns:
        行头文本；若未找到则返回空字符串。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    for value in values:
        candidate = _normalize_optional_string(value)
        if not candidate:
            continue
        if _is_low_information_header(candidate):
            continue
        return candidate
    return ""


def _is_low_information_header(value: str) -> bool:
    """判断文本是否属于低信息头部。

    Args:
        value: 待判断文本。

    Returns:
        是否低信息。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized in _LOW_INFO_TOKENS:
        return True
    if normalized.startswith("unnamed"):
        return True
    if _NUMERIC_LIKE_PATTERN.fullmatch(normalized):
        return True
    return False


def _normalize_header_list(headers: Sequence[Optional[str]]) -> list[str]:
    """标准化表头列表。

    Args:
        headers: 原始表头列表。

    Returns:
        标准化后的表头列表。

    Raises:
        RuntimeError: 处理失败时抛出。
    """

    normalized: list[str] = []
    for header in headers:
        value = _normalize_optional_string(header)
        if not value:
            continue
        normalized.append(value)
    deduped = _deduplicate_headers(normalized)
    if not deduped:
        return []
    return deduped[:10]


def _deduplicate_headers(headers: list[str]) -> list[str]:
    """去重表头（保留首次出现顺序）。

    Args:
        headers: 原始表头列表。

    Returns:
        去重后的表头列表（重复项直接移除）。

    Raises:
        RuntimeError: 去重失败时抛出。
    """

    result: list[str] = []
    seen: set[str] = set()
    for item in headers:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _looks_like_default_headers(headers: list[str]) -> bool:
    """判断是否是默认数字表头。

    Args:
        headers: 表头列表。

    Returns:
        是否为默认数字表头。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if not headers:
        return True
    cleaned = [str(item).strip() for item in headers if str(item).strip()]
    if not cleaned:
        return True
    if all(_is_low_information_header(item) for item in cleaned):
        return True
    # 必须用 isdecimal() 而非 isdigit()：Python 的 isdigit() 对上标字符（'¹'、'²' 等）
    # 也返回 True，但 int() 无法解析这类 Unicode 数字符号，会抛 ValueError。
    # isdecimal() 只接受可参与十进制运算的字符（ASCII 0-9），行为与 int() 一致。
    if all(item.isdecimal() for item in cleaned):
        numbers = [int(item) for item in cleaned]
        start = numbers[0]
        return numbers == list(range(start, start + len(numbers)))
    return False


# ---------------------------------------------------------------------------
# Financial table detection and classification
# ---------------------------------------------------------------------------


def _is_financial_table(
    table_obj: Any,
    *,
    table_text: str = "",
    caption: Optional[str] = None,
    context_before: str = "",
) -> bool:
    """判定表格是否财务表。

    Args:
        table_obj: 表格对象。
        table_text: 表格正文文本。
        caption: 表格标题。
        context_before: 表格前文。

    Returns:
        是否财务表。

    Raises:
        RuntimeError: 判定失败时抛出。
    """

    combined_text = _normalize_searchable_text(" ".join([caption or "", context_before or "", table_text or ""]))
    explicit_financial = bool(getattr(table_obj, "is_financial_table", False))
    semantic_type = str(getattr(table_obj, "semantic_type", "") or "").upper()
    semantic_financial = "FINANCIAL" in semantic_type
    negative_signal = _has_financial_negative_signal(combined_text)
    core_signal = _has_financial_core_signal(combined_text)

    if negative_signal and not core_signal:
        return False
    if explicit_financial and not negative_signal:
        return True
    if semantic_financial and not negative_signal:
        return True
    if core_signal:
        return True
    return False


def _has_financial_negative_signal(text: str) -> bool:
    """判断文本是否命中 financial 负向特征。

    Args:
        text: 归一化文本。

    Returns:
        命中负向关键词返回 `True`。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if not text:
        return False
    return any(keyword in text for keyword in _FINANCIAL_NEGATIVE_KEYWORDS)


def _has_financial_core_signal(text: str) -> bool:
    """判断文本是否命中核心财务表信号。

    Args:
        text: 归一化文本。

    Returns:
        命中核心财务关键词返回 `True`。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if not text:
        return False
    return any(keyword in text for keyword in _CORE_FINANCIAL_TABLE_KEYWORDS)


def _classify_table_type(
    *,
    is_financial: bool,
    row_count: int,
    col_count: int,
    headers: Optional[list[str]],
    table_text: str,
) -> str:
    """对表格做轻量类型分类。

    分类规则优先级：
    1. 显式财务标记 → ``financial``
    2. 极小表格 / 默认列头 / 无列头少行 → ``layout``
    3. Python dict repr 残留（edgartools 解析失败） → ``layout``
    4. Section heading 横线表（``Item N. Title ────``） → ``layout``
    5. SEC 封面页元数据表（法律声明 / 勾选框） → ``layout``
    6. 其他 → ``data``

    Args:
        is_financial: 是否财务表。
        row_count: 行数。
        col_count: 列数。
        headers: 行头列表。
        table_text: 表格文本。

    Returns:
        ``financial``、``data`` 或 ``layout``。

    Raises:
        RuntimeError: 分类失败时抛出。
    """

    if is_financial:
        return "financial"
    normalized_text = _normalize_whitespace(table_text)
    # 原有规则：极小表格
    if row_count <= 2 and col_count <= 3 and (not normalized_text or len(normalized_text) < 16):
        return "layout"
    if headers and _looks_like_default_headers(headers):
        return "layout"
    if not headers and row_count <= 3:
        return "layout"
    # 新增规则：Python dict repr 残留（edgartools 对空表输出 str(dict)）
    if normalized_text and normalized_text.lstrip().startswith("{'type':"):
        return "layout"
    # 新增规则：section heading 横线表（如 "Item 7. MD&A ──────"）
    if is_sec_section_heading_table(normalized_text or ""):
        return "layout"
    # 新增规则：SEC 封面页法律声明 / 勾选框表（少行条件下）
    if row_count <= 5 and is_sec_cover_page_table(normalized_text or ""):
        return "layout"
    return "data"


def _replace_table_with_placeholder(content: str, table_text: str, table_ref: str) -> dict[str, Any]:
    """尝试将表格文本替换为占位符。

    Args:
        content: section 文本。
        table_text: 表格文本。
        table_ref: 表格引用。

    Returns:
        `{\"content\": str, \"replaced\": bool}`。

    Raises:
        RuntimeError: 替换失败时抛出。
    """

    normalized_table_text = _normalize_whitespace(table_text)
    if len(normalized_table_text) < 24:
        return {"content": content, "replaced": False}
    placeholder = _format_table_placeholder(table_ref)
    if normalized_table_text in content:
        replaced_content = content.replace(normalized_table_text, placeholder, 1)
        return {"content": replaced_content, "replaced": True}
    return {"content": content, "replaced": False}


def _should_prioritize_records_output(table: _TableBlock) -> bool:
    """判断是否应优先输出 records。

    Args:
        table: 表格块。

    Returns:
        若属于核心财务表或已判定为 financial/data 表则返回 `True`。

    Raises:
        RuntimeError: 判定失败时抛出。
    """

    if table.table_type in {"financial", "data"}:
        return True
    if table.is_financial:
        return True
    joined_text = " ".join(
        [
            _normalize_whitespace(table.caption or ""),
            _normalize_whitespace(table.context_before or ""),
            _normalize_whitespace(table.text or "")[:240],
        ]
    ).lower()
    return any(keyword in joined_text for keyword in _CORE_FINANCIAL_TABLE_KEYWORDS)


# ---------------------------------------------------------------------------
# Records rendering
# ---------------------------------------------------------------------------


def _render_records_table(
    table_obj: Any,
    *,
    fallback_text: str = "",
    allow_generated_columns: bool = False,
    aggressive_fallback: bool = False,
    precomputed_dataframe: Optional[pd.DataFrame] = None,
) -> Optional[dict[str, Any]]:
    """尝试将表格渲染为 records。

    Args:
        table_obj: 表格对象。
        fallback_text: 回退文本（通常为 table.text()）。
        allow_generated_columns: 是否允许生成 `col_1...` 占位列名。
        aggressive_fallback: 是否启用更激进 fallback（用于核心财务表）。
        precomputed_dataframe: 可复用的预计算 DataFrame。

    Returns:
        `{\"columns\": list[str], \"data\": list[dict[str, Any]]}` 或 `None`。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    expected_col_count = _resolve_expected_table_col_count(table_obj)
    dataframe_payload = _render_records_from_dataframe(
        table_obj=table_obj,
        allow_generated_columns=allow_generated_columns,
        precomputed_dataframe=precomputed_dataframe,
    )
    if dataframe_payload is not None and _is_records_payload_quality_ok(
        dataframe_payload,
        aggressive=aggressive_fallback,
        expected_col_count=expected_col_count,
    ):
        return dataframe_payload

    if aggressive_fallback:
        html_payload = _render_records_from_html_table(
            table_obj=table_obj,
            allow_generated_columns=allow_generated_columns,
        )
        if html_payload is not None and _is_records_payload_quality_ok(
            html_payload,
            aggressive=True,
            expected_col_count=expected_col_count,
        ):
            return html_payload

    markdown_text = _render_markdown_table(table_obj, fallback_text)
    markdown_payload = _render_records_from_markdown_table(
        markdown_text=markdown_text,
        allow_generated_columns=allow_generated_columns,
    )
    if markdown_payload is not None and _is_records_payload_quality_ok(
        markdown_payload,
        aggressive=aggressive_fallback,
        expected_col_count=expected_col_count,
    ):
        return markdown_payload
    return None


def _resolve_expected_table_col_count(table_obj: Any) -> Optional[int]:
    """读取表格声明列数。

    Args:
        table_obj: 表格对象。

    Returns:
        声明列数；不可用时返回 `None`。

    Raises:
        RuntimeError: 读取失败时抛出。
    """

    raw_col_count = getattr(table_obj, "col_count", None)
    if not isinstance(raw_col_count, int) or raw_col_count <= 0:
        return None
    return raw_col_count


def _recover_index_as_column(df: pd.DataFrame) -> pd.DataFrame:
    """将有意义的 DataFrame index 恢复为数据列。

    SEC 财务表经 edgartools 解析后，行标签（如月份名、科目名）可能
    存储在 DataFrame 的 index 而非数据列中。当 index 满足以下条件时
    将其恢复为第一列：

    - index 非默认 ``RangeIndex``
    - index 包含至少一个非空文本值（非纯数字）

    列名取 ``index.name``；多级 index 则拼接非空 level name；若为空则使用
    ``"Item"``。实现上避免直接调用 ``reset_index()``，以绕开大表
    ``MultiIndex`` 上的高成本重排。

    Args:
        df: 待处理的 DataFrame。

    Returns:
        行标签已恢复为首列的 DataFrame（或原对象）。
    """

    if isinstance(df.index, pd.RangeIndex):
        return df
    if not _index_contains_meaningful_text(df.index):
        return df
    recovered_values = [_render_index_value(value) for value in df.index]
    result = df.copy()
    result.index = pd.RangeIndex(len(result))
    result.insert(
        0,
        _resolve_recovered_index_column_name(df.index),
        pd.Index(recovered_values, dtype="object"),
        allow_duplicates=True,
    )
    return result


def _index_contains_meaningful_text(index: pd.Index) -> bool:
    """判断 index 中是否包含值得恢复的文本标签。

    Args:
        index: 待检查的 pandas index。

    Returns:
        若存在至少一个非纯数字文本标签则返回 `True`，否则返回 `False`。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    for value in index:
        if _index_value_has_meaningful_text(value):
            return True
    return False


def _index_value_has_meaningful_text(value: Any) -> bool:
    """判断单个 index 值是否包含有意义文本。

    Args:
        value: index 值，可能是标量或多级 tuple。

    Returns:
        若包含非空、非纯数字文本则返回 `True`，否则返回 `False`。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if isinstance(value, tuple):
        return any(_index_value_has_meaningful_text(part) for part in value)
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return not text.replace(".", "").replace("-", "").isdigit()


def _render_index_value(value: Any) -> str:
    """将 index 值渲染为可序列化文本。

    Args:
        value: index 值，可能是标量或多级 tuple。

    Returns:
        适合写入首列的文本表示。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    if isinstance(value, tuple):
        parts = [
            str(part).strip()
            for part in value
            if part is not None and str(part).strip() and str(part).strip().lower() != "nan"
        ]
        return " | ".join(parts)
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _resolve_recovered_index_column_name(index: pd.Index) -> str:
    """为恢复出的 index 列生成稳定列名。

    Args:
        index: 原始 pandas index。

    Returns:
        单级 index 返回其名称；多级 index 返回拼接后的名称；不可用时返回
        ``"Item"``。

    Raises:
        RuntimeError: 名称解析失败时抛出。
    """

    if isinstance(index, pd.MultiIndex):
        names = [
            str(name).strip()
            for name in index.names
            if name is not None and str(name).strip() and str(name).strip().lower() != "none"
        ]
        return " | ".join(names) if names else "Item"
    raw_name = index.name
    if raw_name is None:
        return "Item"
    text = str(raw_name).strip()
    if not text or text.lower() in {"index", "none"}:
        return "Item"
    return text


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """展平 MultiIndex 列名为可读字符串。

    当 DataFrame 的列为 ``pd.MultiIndex`` 时（常见于 SEC 财务表的
    多级表头），将每级非空标签以 ``" | "`` 拼接为单级字符串。
    例如 ``("Revenue", "2024")`` → ``"Revenue | 2024"``。

    若列非 MultiIndex 则原样返回，不做拷贝。

    Args:
        df: 待处理的 DataFrame。

    Returns:
        列名已展平的 DataFrame（新副本或原对象）。
    """

    if not isinstance(df.columns, pd.MultiIndex):
        return df
    flat_names: list[str] = []
    for col_tuple in df.columns:
        # 过滤空级（空字符串 / None / NaN / 纯空白）
        parts: list[str] = []
        for level_val in col_tuple:
            text = str(level_val).strip() if level_val is not None else ""
            if text and text.lower() != "nan":
                parts.append(text)
        flat_names.append(" | ".join(parts) if parts else "")
    result = df.copy()
    result.columns = pd.Index(flat_names)
    return result


def _collapse_ghost_columns(df: pd.DataFrame) -> pd.DataFrame:
    """合并因 HTML colspan 产生的同名幽灵列。

    SEC 财务表经 edgartools 解析后，同一语义列因 colspan 被拆分为
    多个同名列（经 _uniquify_columns 后变为 ``col``, ``col_2``,
    ``col_3``）。本函数检测相邻同名列组，将每行中散落的非 null 值
    合并到首列，然后丢弃剩余幽灵列。

    同时处理 ``$`` / ``%`` 等符号被独立成列的情况：若合并后某列的
    所有非 null 值均为纯符号，将其追加到相邻数值列的值前/后缀，
    然后丢弃符号列。

    Args:
        df: 待处理的 DataFrame（列名可能含 ``_2``/``_3`` 后缀）。

    Returns:
        消除幽灵列后的 DataFrame。
    """

    if df.empty or len(df.columns) <= 1:
        return df

    # --- 第一步：识别相邻同名列组 ---
    col_names = [str(c) for c in df.columns]
    groups: list[list[int]] = []  # 每组为同基名相邻列的索引列表
    current_group: list[int] = [0]
    base_name_0 = _ghost_column_base_name(col_names[0])

    for i in range(1, len(col_names)):
        base_i = _ghost_column_base_name(col_names[i])
        if base_i == base_name_0:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
            base_name_0 = base_i
    groups.append(current_group)

    # 如果没有任何多列组则无需处理
    if all(len(g) == 1 for g in groups):
        return df

    # --- 第二步：合并同组列 ---
    result_cols: list[str] = []
    result_data: dict[str, list[Any]] = {}

    for group in groups:
        primary_idx = group[0]
        primary_name = col_names[primary_idx]
        if len(group) == 1:
            result_cols.append(primary_name)
            result_data[primary_name] = list(df.iloc[:, primary_idx])
            continue

        # 合并同组各行：取第一个非 null 值
        merged_values: list[Any] = []
        for row_idx in range(len(df)):
            merged = None
            for col_idx in group:
                val = df.iloc[row_idx, col_idx]
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    cell_text = str(val).strip() if not isinstance(val, (int, float)) else val
                    if cell_text == "" or cell_text == "nan":
                        continue
                    if merged is None:
                        merged = val
                    # 若已有值，尝试合并符号
                    elif isinstance(val, str) and val.strip() in _CURRENCY_SYMBOL_TOKENS:
                        merged = f"{val.strip()}{merged}" if isinstance(merged, str) else merged
                    elif isinstance(merged, str) and merged.strip() in _CURRENCY_SYMBOL_TOKENS:
                        merged = f"{merged.strip()}{val}" if isinstance(val, str) else val
            merged_values.append(merged)

        # 使用基名（去掉 _2/_3 后缀）作为列名
        clean_name = _ghost_column_base_name(primary_name)
        # 避免与已有列名冲突
        if clean_name in result_data:
            clean_name = primary_name
        result_cols.append(clean_name)
        result_data[clean_name] = merged_values

    return pd.DataFrame(result_data, columns=pd.Index(result_cols, dtype="object"))


def _ghost_column_base_name(col_name: Any) -> str:
    """提取列名的基名（去掉 ``_2``/``_3`` 等去重后缀）。

    Args:
        col_name: 原始列名（可能为 int 等非字符串类型）。

    Returns:
        去除尾部 ``_\\d+`` 后缀的基名。
    """

    name_str = str(col_name)
    return re.sub(r"_\d+$", "", name_str)


def _render_records_from_dataframe(
    *,
    table_obj: Any,
    allow_generated_columns: bool,
    precomputed_dataframe: Optional[pd.DataFrame] = None,
) -> Optional[dict[str, Any]]:
    """从 DataFrame 渲染 records。

    Args:
        table_obj: 表格对象。
        allow_generated_columns: 是否允许生成占位列名。
        precomputed_dataframe: 可复用的预计算 DataFrame。

    Returns:
        records 载荷；失败返回 `None`。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    table_df = precomputed_dataframe
    if table_df is None:
        table_df = _safe_table_dataframe(table_obj)
    if table_df is None or table_df.empty:
        return None

    # 行标签恢复：将有意义的 index 转为数据列
    table_df = _recover_index_as_column(table_df)

    # MultiIndex 列名展平（消除 tuple repr 问题）
    table_df = _flatten_multiindex_columns(table_df)

    if table_df.columns.has_duplicates:
        table_df = table_df.copy()
        table_df.columns = _uniquify_columns([str(column) for column in table_df.columns])

    # 合并因 colspan 产生的幽灵列（消除 _2/_3 空列和符号分裂）
    table_df = _collapse_ghost_columns(table_df)

    columns = _extract_headers_from_dataframe(table_df)
    normalized_df = table_df.copy()
    if not columns:
        candidate_columns = [_normalize_optional_string(column) for column in normalized_df.columns]
        columns = _build_table_columns(
            candidate_columns=candidate_columns,
            col_count=normalized_df.shape[1],
            allow_generated=allow_generated_columns,
        )
        if columns is None:
            return None
    normalized_df.columns = columns
    records = _dataframe_to_records(normalized_df)
    if not records:
        return None
    return {"columns": columns, "data": records}


def _render_records_from_html_table(
    *,
    table_obj: Any,
    allow_generated_columns: bool,
) -> Optional[dict[str, Any]]:
    """从 HTML 表格结构渲染 records。

    Args:
        table_obj: 表格对象。
        allow_generated_columns: 是否允许生成占位列名。

    Returns:
        records 载荷；失败返回 `None`。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    html_text = _extract_table_html(table_obj)
    if not html_text:
        return None
    soup = BeautifulSoup(html_text, "html.parser")
    table_tag = soup.find("table")
    if table_tag is None:
        return None

    header_rows: list[list[str]] = []
    data_rows: list[list[str]] = []
    for row_tag in table_tag.find_all("tr"):
        row_cells: list[str] = []
        header_cells: list[str] = []
        for th in row_tag.find_all("th"):
            header_cells.append(_normalize_whitespace(th.get_text(" ", strip=True)))
        for td in row_tag.find_all("td"):
            row_cells.append(_normalize_whitespace(td.get_text(" ", strip=True)))
        if header_cells:
            header_rows.append(header_cells)
        if row_cells:
            data_rows.append(row_cells)

    if not data_rows:
        return None
    col_count = max(len(row) for row in data_rows)
    collapsed_headers = _collapse_header_rows(header_rows, col_count)
    columns = _build_table_columns(
        candidate_columns=collapsed_headers,
        col_count=col_count,
        allow_generated=allow_generated_columns,
    )
    if columns is None:
        return None
    records = _matrix_rows_to_records(
        rows=data_rows,
        columns=columns,
    )
    if not records:
        return None
    return {"columns": columns, "data": records}


def _render_records_from_markdown_table(
    *,
    markdown_text: str,
    allow_generated_columns: bool,
) -> Optional[dict[str, Any]]:
    """从 markdown 文本回解析 records。

    Args:
        markdown_text: markdown 表格文本。
        allow_generated_columns: 是否允许生成占位列名。

    Returns:
        records 载荷；解析失败返回 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    if not markdown_text:
        return None
    lines = [line.strip() for line in markdown_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    if not all("|" in line for line in lines[:2]):
        return None

    header_cells = _split_markdown_row(lines[0])
    separator = lines[1]
    if not _MARKDOWN_SEPARATOR_PATTERN.fullmatch(separator):
        return None
    data_lines = lines[2:]
    if not data_lines:
        return None

    col_count = max(len(header_cells), max(len(_split_markdown_row(line)) for line in data_lines))
    columns = _build_table_columns(
        candidate_columns=header_cells,
        col_count=col_count,
        allow_generated=allow_generated_columns,
    )
    if columns is None:
        return None
    matrix_rows = [_split_markdown_row(line) for line in data_lines]
    records = _matrix_rows_to_records(rows=matrix_rows, columns=columns)
    if not records:
        return None
    return {"columns": columns, "data": records}


def _split_markdown_row(row: str) -> list[str]:
    """拆分 markdown 行。

    Args:
        row: 行文本。

    Returns:
        单元格列表。

    Raises:
        RuntimeError: 拆分失败时抛出。
    """

    trimmed = row.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [_normalize_whitespace(cell) for cell in trimmed.split("|")]


def _extract_table_html(table_obj: Any) -> str:
    """从表格对象提取 HTML 文本。

    Args:
        table_obj: 表格对象。

    Returns:
        HTML 字符串；无法提取时返回空字符串。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    for attr_name in ("html", "table_html", "raw_html"):
        raw_attr = getattr(table_obj, attr_name, None)
        if isinstance(raw_attr, str) and raw_attr.strip():
            return raw_attr
    to_html = getattr(table_obj, "to_html", None)
    if callable(to_html):
        try:
            html_value = to_html()
        except Exception:
            return ""
        if isinstance(html_value, str):
            return html_value
    return ""


def _collapse_header_rows(header_rows: list[list[str]], col_count: int) -> list[Optional[str]]:
    """合并多行表头。

    Args:
        header_rows: 表头行列表。
        col_count: 目标列数。

    Returns:
        合并后的列名候选列表。

    Raises:
        RuntimeError: 合并失败时抛出。
    """

    if col_count <= 0:
        return []
    merged: list[list[str]] = [[] for _ in range(col_count)]
    for row in header_rows:
        for index in range(col_count):
            token = _normalize_optional_string(row[index] if index < len(row) else None)
            if not token:
                continue
            if token in merged[index]:
                continue
            merged[index].append(token)
    return [" / ".join(parts) if parts else None for parts in merged]


def _build_table_columns(
    *,
    candidate_columns: Sequence[Optional[str]],
    col_count: int,
    allow_generated: bool,
) -> Optional[list[str]]:
    """构建可用列名。

    Args:
        candidate_columns: 候选列名。
        col_count: 目标列数。
        allow_generated: 是否允许自动生成列名。

    Returns:
        列名列表；不可用时返回 `None`。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    if col_count <= 0:
        return None
    normalized_columns: list[str] = []
    for index in range(col_count):
        candidate = _normalize_optional_string(candidate_columns[index] if index < len(candidate_columns) else None)
        if candidate and not _is_low_information_header(candidate):
            normalized_columns.append(candidate)
            continue
        if allow_generated:
            normalized_columns.append(f"col_{index + 1}")
            continue
        return None
    return _uniquify_columns(normalized_columns)


def _uniquify_columns(columns: list[str]) -> list[str]:
    """将列名去重并保持顺序。

    Args:
        columns: 原始列名。

    Returns:
        唯一化后的列名列表。

    Raises:
        RuntimeError: 去重失败时抛出。
    """

    seen: dict[str, int] = {}
    result: list[str] = []
    for column in columns:
        key = column.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            result.append(column)
            continue
        result.append(f"{column}_{seen[key]}")
    return result


def _matrix_rows_to_records(
    *,
    rows: list[list[str]],
    columns: list[str],
) -> list[dict[str, Any]]:
    """将二维文本矩阵转换为 records。

    Args:
        rows: 行矩阵。
        columns: 列名列表。

    Returns:
        records 列表。

    Raises:
        RuntimeError: 转换失败时抛出。
    """

    records: list[dict[str, Any]] = []
    for row in rows:
        normalized_row: dict[str, Any] = {}
        for index, column in enumerate(columns):
            cell = row[index] if index < len(row) else ""
            normalized_row[column] = _normalize_table_cell_value(cell)
        records.append(normalized_row)
    return records


def _is_records_payload_quality_ok(
    payload: dict[str, Any],
    *,
    aggressive: bool,
    expected_col_count: Optional[int] = None,
) -> bool:
    """评估 records 载荷质量。

    Args:
        payload: records 载荷。
        aggressive: 是否使用宽松阈值（核心财务表）。
        expected_col_count: 原始表格声明列数。

    Returns:
        质量是否达标。

    Raises:
        RuntimeError: 评估失败时抛出。
    """

    columns = payload.get("columns")
    data = payload.get("data")
    if not isinstance(columns, list) or not columns:
        return False
    if not isinstance(data, list) or not data:
        return False
    total_cells = len(columns) * len(data)
    if total_cells <= 0:
        return False
    if expected_col_count is not None and expected_col_count > 0:
        ratio = len(columns) / expected_col_count
        min_ratio = 0.5 if aggressive else 0.65
        if ratio < min_ratio:
            return False
    generated_count = sum(1 for column in columns if _GENERATED_COLUMN_PATTERN.fullmatch(str(column)))
    generated_ratio = generated_count / len(columns)
    generated_threshold = 0.8 if aggressive else 0.6
    if generated_ratio > generated_threshold:
        return False
    non_empty_cells = 0
    for row in data:
        if not isinstance(row, dict):
            continue
        for column in columns:
            if row.get(column) is not None:
                non_empty_cells += 1
    density = non_empty_cells / total_cells
    threshold = 0.15 if aggressive else 0.3
    return density >= threshold


def _dataframe_to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """将 DataFrame 转换为 records，并稳定处理空值与数字文本。

    Args:
        dataframe: 待转换的 DataFrame。

    Returns:
        规范化后的 records 列表。

    Raises:
        RuntimeError: 转换失败时抛出。
    """

    records: list[dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        normalized_row: dict[str, Any] = {}
        for column in dataframe.columns:
            normalized_row[str(column)] = _normalize_table_cell_value(row[column])
        records.append(normalized_row)
    return records


def _normalize_table_cell_value(value: Any) -> Any:
    """标准化表格单元格值。

    Args:
        value: 原始单元格值。

    Returns:
        标准化后的值：空值返回 `None`，数字文本转为可解析字符串，其余返回清洗文本。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    if value is None:
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return value
    if pd.isna(value):
        return None
    text = _normalize_whitespace(str(value))
    if not text:
        return None
    lowered = text.lower()
    if lowered in _LOW_INFO_TOKENS:
        return None
    normalized_numeric = _normalize_numeric_cell_text(text)
    if normalized_numeric is not None:
        return normalized_numeric
    return text


def _normalize_numeric_cell_text(text: str) -> Optional[str]:
    """将数字样式文本标准化。

    规则：处理货币符号、千分位、括号负数与常见脚注标记；无法确定为数字时返回 `None`。

    Args:
        text: 原始文本。

    Returns:
        规范化数字字符串；非数字返回 `None`。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    cleaned = _strip_trailing_footnote(text)
    if "%" in cleaned:
        return None
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    if negative:
        cleaned = cleaned[1:-1].strip()
    # 先剥离货币符号与空白，再由共享规范化辅助函数处理千分位/小数分隔符，避免欧洲格式 "1,23" 被误删逗号丢失小数语义。
    cleaned = cleaned.replace("$", "").replace("¥", "").replace("€", "").replace(" ", "")
    cleaned = normalize_numeric_separators(cleaned)
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    if not cleaned:
        return None
    if not _STRICT_NUMERIC_TEXT_PATTERN.fullmatch(cleaned):
        return None
    if negative:
        cleaned = f"-{cleaned}"
    return cleaned


def _strip_trailing_footnote(text: str) -> str:
    """移除尾部脚注标记。

    Args:
        text: 原始文本。

    Returns:
        去除常见脚注标记后的文本。

    Raises:
        RuntimeError: 清洗失败时抛出。
    """

    cleaned = re.sub(r"(?<=\d)\[(?:\d+|[a-zA-Z])\]\s*$", "", text).strip()
    cleaned = re.sub(r"(?<=\d)\((?:\d+|[a-zA-Z])\)\s*$", "", cleaned).strip()
    # 仅剥离裸尾小写单字母脚注（SEC 财报续序脚注约定为 a/b/c…）；
    # 保留大写单字母后缀（M/K/B/T magnitude、Q/K form 标识等），由 _STRICT_NUMERIC_TEXT_PATTERN
    # 在 _normalize_numeric_text 内主动判 None，避免数量级被静默丢失。
    cleaned = re.sub(r"(?<=\d)[a-z]\s*$", "", cleaned).strip()
    return cleaned


def _render_markdown_table(table_obj: Any, fallback_text: str) -> str:
    """将表格渲染为 markdown 文本。

    Args:
        table_obj: 表格对象。
        fallback_text: 回退文本。

    Returns:
        markdown 字符串。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    table_df = _safe_table_dataframe(table_obj)
    if table_df is not None and not table_df.empty:
        try:
            from tabulate import tabulate as _tabulate

            del _tabulate
            # NaN 在 MultiIndex 合并单元格中常见，转 markdown 前统一填为空字符串
            cleaned_df = table_df.fillna("")
            markdown = cleaned_df.to_markdown(index=False)
            if isinstance(markdown, str) and markdown.strip():
                return markdown
            raise RuntimeError("SEC 表格 markdown 渲染结果为空")
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 tabulate 依赖，无法渲染 SEC 表格 markdown") from exc
        except Exception as exc:
            raise RuntimeError(f"SEC 表格 markdown 渲染失败: {exc}") from exc
    if fallback_text:
        return fallback_text
    return _normalize_whitespace(str(getattr(table_obj, "to_dict", lambda: {})()))
