"""SEC 文档处理器实现。

该模块基于 `edgartools` 提供 SEC 主流程文档的结构化读取能力：
- `list_sections/list_tables/read_section/read_table/search`
- `get_financial_statement/query_xbrl_facts`（XBRL 能力）

设计目标：
- 严格对齐 `DocumentProcessor(Source)` 协议。
- 仅做"单文档解析"，不承担下载、路由、仓储写入职责。
- 对 `DEF 14A/6-K` 显式让位给 `BSProcessor`，优先保障 LLM 输入可读性。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from edgar.documents import HTMLParser, ParserConfig
from edgar.documents.exceptions import DocumentTooLargeError
from edgar.xbrl import XBRL

from dayu.engine.processors.base import (
    SearchHit,
    SectionContent,
    SectionSummary,
    TableContent,
    TableSummary,
    build_table_content,
)
from .financial_base import (
    FinancialMeta,
    FinancialStatementResult,
    XbrlFactsResult,
)
from dayu.engine.processors.text_utils import (
    append_missing_table_placeholders as _append_missing_placeholders,
    infer_suffix_from_uri as _infer_suffix_from_uri,
)
from dayu.engine.processors.search_utils import enrich_hits_by_section
from dayu.engine.processors.source import Source
from dayu.engine.processors.perf_utils import ProcessorStageProfiler, is_processor_profile_enabled
from dayu.fins.xbrl_file_discovery import discover_xbrl_files
from dayu.log import Log
from dayu.fins.processors.form_type_utils import normalize_form_type as _normalize_form_type

# --- 子模块导入 ---
from dayu.fins.processors.sec_xbrl_query import (
    _STATEMENT_METHODS,
    _build_period_summary,
    _build_statement_rows,
    _extract_period_columns,
    _infer_currency_from_units,
    _infer_scale_from_xbrl_query,
    _infer_units_from_xbrl_query,
    _infer_xbrl_taxonomy,
    _normalize_fact_row,
    _normalize_query_statement_type,
    _query_facts_rows,
    build_statement_locator,
)
from dayu.fins.processors.sec_section_build import (
    _SectionBlock,
    _build_sections,
    _safe_document_text,
)
from dayu.fins.processors.sec_dom_helpers import (
    _extract_dom_table_contexts,
    _extract_text_from_raw_html,
)
from dayu.fins.processors.sec_table_extraction import (
    _build_tables,
    _render_markdown_table,
    _render_records_table,
    _replace_table_with_placeholder,
    _safe_statement_dataframe,
    _should_prioritize_records_output,
)


# --- 本模块独用常量 ---

_SUPPORTED_FORMS = frozenset(
    {
        "10-K",
        "10-Q",
        "20-F",
        "8-K",
        "DEF 14A",
        "SC 13D",
        "SC 13D/A",
        "SC 13G",
        "SC 13G/A",
    }
)
_MEDIA_TYPE_TOKENS = ("html", "xhtml", "xml")
_FILE_SUFFIXES = {".htm", ".html", ".xhtml", ".xml"}
_EDGAR_MAX_DOCUMENT_SIZE_BYTES = 256 * 1024 * 1024
_EDGAR_MAX_DOCUMENT_SIZE_RETRY_MULTIPLIER = 2
_SECTION_RENDER_CACHE_MAX_ENTRIES = 256

class SecProcessor:
    """SEC 文档处理器。"""

    PARSER_VERSION = "sec_processor_v1.0.0"
    MODULE = "FINS.SEC_PROCESSOR"
    _ENABLE_FAST_SECTION_BUILD = False
    _FAST_SECTION_BUILD_SINGLE_FULL_TEXT = False

    @classmethod
    def get_parser_version(cls) -> str:
        """返回处理器 parser version。"""

        return str(cls.PARSER_VERSION)

    def __init__(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> None:
        """初始化处理器。

        Args:
            source: 文档来源抽象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            None。

        Raises:
            ValueError: 源文件不存在或参数非法时抛出。
            RuntimeError: 解析失败时抛出。
        """

        self._source = source
        self._form_type = _normalize_form_type(form_type)
        self._media_type = media_type or source.media_type
        self._profiler = ProcessorStageProfiler(
            component=self.__class__.__name__,
            enabled=is_processor_profile_enabled(),
        )
        self._full_text_cache: Optional[str] = None
        single_full_text_enabled = self._should_use_single_full_text_section()
        preloaded_full_text: Optional[str] = None

        suffix = _infer_suffix_from_uri(source.uri) or ".html"
        source_path = source.materialize(suffix=suffix)
        if not source_path.exists() or not source_path.is_file():
            raise ValueError(f"源文件不存在: {source_path}")
        self._source_path = source_path

        with self._profiler.stage("load_source_html"):
            html_content = _load_text(source_path)
        with self._profiler.stage("parse_document"):
            self._document = _parse_document(html_content, self._form_type)
        if single_full_text_enabled:
            with self._profiler.stage("preload_full_text"):
                preloaded_full_text = _safe_document_text(self._document)
                # edgartools document.text() 对部分大型/复杂 XBRL filing 返回空文本，
                # 回退到直接从原始 HTML 提取纯文本，保证后续 marker 检测有基底。
                if not preloaded_full_text.strip():
                    preloaded_full_text = _extract_text_from_raw_html(html_content)

        with self._profiler.stage("build_sections"):
            self._sections = _build_sections(
                self._document,
                fast_mode=self._should_use_fast_section_build(),
                single_full_text=single_full_text_enabled,
                full_text_override=preloaded_full_text,
            )
        with self._profiler.stage("extract_dom_table_contexts"):
            dom_table_contexts = _extract_dom_table_contexts(html_content)
        with self._profiler.stage("build_tables"):
            self._tables = _build_tables(
                document=self._document,
                sections=self._sections,
                dom_table_contexts=dom_table_contexts,
            )
        self._section_by_ref = {section.ref: section for section in self._sections}
        self._table_by_ref = {table.ref: table for table in self._tables}
        self._listable_table_refs = {
            table.ref for table in self._tables if table.table_type != "layout"
        }
        self._section_render_cache: OrderedDict[str, str] = OrderedDict()

        self._xbrl: Optional[XBRL] = None
        self._xbrl_loaded = False
        self._xbrl_taxonomy: Optional[str] = None
        self._xbrl_taxonomy_loaded = False
        if single_full_text_enabled and preloaded_full_text is not None:
            self._full_text_cache = preloaded_full_text
        self._profiler.log_summary(
            extra=(
                f"uri={self._source.uri} "
                f"fast_sections={self._should_use_fast_section_build()} "
                f"single_full_text={single_full_text_enabled}"
            ),
        )

    def get_section_title(self, ref: str) -> Optional[str]:
        """根据 section ref 获取章节标题。

        Args:
            ref: 章节引用。

        Returns:
            章节标题字符串；ref 不存在时返回 None。
        """
        section = self._section_by_ref.get(ref)
        return section.title if section else None

    def _should_use_fast_section_build(self) -> bool:
        """判断当前实例是否启用快速章节构建。

        Args:
            无。

        Returns:
            启用快速构建返回 ``True``，否则返回 ``False``。

        Raises:
            RuntimeError: 判断失败时抛出。
        """

        return bool(getattr(self.__class__, "_ENABLE_FAST_SECTION_BUILD", False))

    def _should_use_single_full_text_section(self) -> bool:
        """判断快速构建是否使用“单全文章节”策略。

        Args:
            无。

        Returns:
            启用单全文章节策略返回 ``True``，否则返回 ``False``。

        Raises:
            RuntimeError: 判断失败时抛出。
        """

        return bool(getattr(self.__class__, "_FAST_SECTION_BUILD_SINGLE_FULL_TEXT", False))

    @classmethod
    def supports(
        cls,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> bool:
        """判断是否支持处理该文件。

        Args:
            source: 文档来源抽象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            是否支持。

        Raises:
            OSError: 文件访问失败时可能抛出。
        """

        normalized_form = _normalize_form_type(form_type)
        if normalized_form is None:
            return False
        # 设计约束：6-K 统一路由到 BSProcessor，避免 edgartools 分段结果
        # 在部分材料型文档上给 LLM 产生低质量输入。
        if normalized_form in {"6-K"}:
            return False
        if normalized_form not in _SUPPORTED_FORMS:
            return False

        resolved_media_type = str(media_type or source.media_type or "").lower()
        if any(token in resolved_media_type for token in _MEDIA_TYPE_TOKENS):
            return True

        return _infer_suffix_from_uri(source.uri) in _FILE_SUFFIXES

    def list_sections(self) -> list[SectionSummary]:
        """读取章节列表。

        Args:
            无。

        Returns:
            章节摘要列表。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        try:
            return [
                {
                    "ref": section.ref,
                    "title": section.title,
                    "level": section.level,
                    "parent_ref": section.parent_ref,
                    "preview": section.preview,
                }
                for section in self._sections
            ]
        except Exception as exc:  # pragma: no cover - 兜底保护
            Log.warn(f"list_sections 失败: {exc}", module=self.MODULE)
            raise RuntimeError("SEC section parsing failed") from exc

    def list_tables(self) -> list[TableSummary]:
        """读取表格列表。

        Args:
            无。

        Returns:
            表格摘要列表。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        # 默认过滤 layout 表（SEC 封面元数据、横线表、空表等）
        try:
            return [
                {
                    "table_ref": table.ref,
                    "caption": table.caption,
                    "context_before": table.context_before,
                    "row_count": table.row_count,
                    "col_count": table.col_count,
                    "is_financial": table.is_financial,
                    "table_type": table.table_type,
                    "headers": table.headers,
                    "section_ref": table.section_ref,
                }
                for table in self._tables
                if table.ref in self._listable_table_refs
            ]
        except Exception as exc:  # pragma: no cover - 兜底保护
            Log.warn(f"list_tables 失败: {exc}", module=self.MODULE)
            raise RuntimeError("SEC table parsing failed") from exc

    def read_section(self, ref: str) -> SectionContent:
        """按 ref 读取章节内容。

        Args:
            ref: 章节引用。

        Returns:
            章节内容字典。

        Raises:
            KeyError: 章节不存在时抛出。
            RuntimeError: 读取失败时抛出。
        """

        section = self._section_by_ref.get(ref)
        if section is None:
            raise KeyError(f"Section not found: {ref}")

        try:
            with self._profiler.stage("read_section"):
                content = self._get_or_render_section_content(section)
            visible_table_refs = [
                table_ref
                for table_ref in section.table_refs
                if table_ref in self._listable_table_refs
            ]
            word_count = len(content.split())
            return {
                "ref": section.ref,
                "title": section.title,
                "content": content,
                "tables": visible_table_refs,
                "word_count": word_count,
                "contains_full_text": section.contains_full_text,
            }
        except Exception as exc:  # pragma: no cover - 兜底保护
            Log.warn(f"read_section 失败: ref={ref} exc={exc}", module=self.MODULE)
            raise RuntimeError(f"Section read failed: {ref}") from exc

    def read_table(self, table_ref: str) -> TableContent:
        """按 ref 读取表格内容。

        Args:
            table_ref: 表格引用。

        Returns:
            表格内容字典。

        Raises:
            KeyError: 表格不存在时抛出。
            RuntimeError: 渲染失败时抛出。
        """

        table = self._table_by_ref.get(table_ref)
        if table is None:
            raise KeyError(f"Table not found: {table_ref}")

        try:
            prefer_records = _should_prioritize_records_output(table)
            records_payload = _render_records_table(
                table.table_obj,
                fallback_text=table.text,
                allow_generated_columns=prefer_records,
                aggressive_fallback=prefer_records,
                precomputed_dataframe=table.resolve_dataframe(),
            )
            if records_payload is not None:
                return build_table_content(
                    table_ref=table.ref,
                    caption=table.caption,
                    data_format="records",
                    data=records_payload["data"],
                    columns=records_payload["columns"],
                    row_count=table.row_count,
                    col_count=table.col_count,
                    section_ref=table.section_ref,
                    table_type=table.table_type,
                    is_financial=table.is_financial,
                )
            markdown_text = _render_markdown_table(table.table_obj, table.text)
            return build_table_content(
                table_ref=table.ref,
                caption=table.caption,
                data_format="markdown",
                data=markdown_text,
                columns=None,
                row_count=table.row_count,
                col_count=table.col_count,
                section_ref=table.section_ref,
                table_type=table.table_type,
                is_financial=table.is_financial,
            )
        except Exception as exc:  # pragma: no cover - 兜底保护
            Log.warn(f"read_table 失败: table_ref={table_ref} exc={exc}", module=self.MODULE)
            raise RuntimeError(f"Table read failed: {table_ref}") from exc

    def search(self, query: str, within_ref: Optional[str] = None) -> list[SearchHit]:
        """在文档中搜索。

        Args:
            query: 搜索词。
            within_ref: 可选章节范围。

        Returns:
            搜索命中列表。

        Raises:
            RuntimeError: 搜索失败时抛出。
        """

        # TODO(phase-2): 扩展同义词/词形变化的智能匹配。
        if not query.strip():
            return []
        if within_ref is not None and within_ref not in self._section_by_ref:
            return []

        target_sections = (
            [self._section_by_ref[within_ref]]
            if within_ref is not None
            else self._sections
        )
        normalized_query = query.strip()
        hits_raw: list[SearchHit] = []
        section_content_map: dict[str, str] = {}
        with self._profiler.stage("search"):
            # 在循环外预编译 query 正则，避免每个 section 迭代都重复 re.escape + re.compile。
            query_pattern = re.compile(re.escape(normalized_query), flags=re.IGNORECASE)
            for section in target_sections:
                if query_pattern.search(section.text) is None:
                    continue
                section_content_map[section.ref] = section.text
                hits_raw.append(
                    {
                        "section_ref": section.ref,
                        "section_title": section.title,
                        "snippet": normalized_query,
                    }
                )
        return enrich_hits_by_section(
            hits_raw=hits_raw,
            section_content_map=section_content_map,
            query=normalized_query,
        )

    def _get_or_render_section_content(self, section: _SectionBlock) -> str:
        """读取或渲染章节正文缓存。

        Args:
            section: 章节对象。

        Returns:
            章节正文文本。

        Raises:
            RuntimeError: 渲染失败时抛出。
        """

        cached = self._section_render_cache.get(section.ref)
        if cached is not None:
            self._section_render_cache.move_to_end(section.ref, last=True)
            return cached

        content = section.text
        unresolved_refs: list[str] = []
        for table_ref in section.table_refs:
            table = self._table_by_ref.get(table_ref)
            if table is None:
                continue
            replaced = _replace_table_with_placeholder(content, table.text, table_ref)
            content = replaced["content"]
            if not replaced["replaced"]:
                unresolved_refs.append(table_ref)
        rendered = _append_missing_placeholders(content, unresolved_refs)
        self._section_render_cache[section.ref] = rendered
        self._section_render_cache.move_to_end(section.ref, last=True)
        while len(self._section_render_cache) > _SECTION_RENDER_CACHE_MAX_ENTRIES:
            self._section_render_cache.popitem(last=False)
        return rendered

    def get_full_text(self) -> str:
        """获取文档的完整纯文本内容（包含表格内文本）。

        委托 edgartools ``Document.text()`` 获取全文，该文本默认
        ``include_tables=True``，保留表格内所有文本。
        若 edgartools 返回空文本，回退到从原始 HTML 直接提取。

        Args:
            无。

        Returns:
            文档完整纯文本字符串。

        Raises:
            RuntimeError: 提取失败时抛出。
        """

        if self._full_text_cache is not None:
            return str(self._full_text_cache).strip()
        try:
            text = self._document.text()
        except Exception as exc:  # pragma: no cover - edgartools 内部异常
            Log.warn(f"SecProcessor 全文提取失败: {exc}", module=self.MODULE)
            raise RuntimeError("SecProcessor 全文提取失败") from exc
        result = str(text or "").strip()
        # edgartools document.text() 对部分复杂 filing 返回空文本，
        # 回退到直接从原始 HTML 提取纯文本。
        if not result:
            result = _extract_text_from_raw_html(_load_text(self._source_path)).strip()
        return result

    def get_full_text_with_table_markers(self) -> str:
        """获取带表格占位符的全文（SecProcessor 不支持）。

        SecProcessor 基于 edgartools 解析，不具备 DOM 级表格标记
        注入能力，返回空字符串表示不支持。

        Args:
            无。

        Returns:
            空字符串。
        """

        return ""

    def get_financial_statement(
        self,
        statement_type: str,
        financials: Optional[dict[str, Any]] = None,
        *,
        meta: Optional[FinancialMeta] = None,
    ) -> FinancialStatementResult:
        """获取标准财务报表。

        Args:
            statement_type: 报表类型。
            financials: 预留参数，当前实现不使用。
            meta: 预留参数，当前实现不使用。

        Returns:
            财务报表结果。

        Raises:
            RuntimeError: XBRL 读取或转换失败时抛出。
        """

        del financials
        del meta

        normalized_statement_type = statement_type.strip().lower()
        method_name = _STATEMENT_METHODS.get(normalized_statement_type)
        base_result: FinancialStatementResult = {
            "statement_type": statement_type,
            "periods": [],
            "rows": [],
            "currency": None,
            "units": None,
            "scale": None,
            "data_quality": "partial",
        }
        if method_name is None:
            base_result["reason"] = "unsupported_statement_type"
            return base_result

        xbrl = self._get_xbrl()
        if xbrl is None:
            base_result["reason"] = "xbrl_not_available"
            return base_result

        statements = getattr(xbrl, "statements", None)
        method = getattr(statements, method_name, None)
        if not callable(method):
            base_result["reason"] = "statement_method_missing"
            return base_result

        statement_obj = method()
        if statement_obj is None:
            base_result["reason"] = "statement_not_found"
            return base_result

        statement_df = _safe_statement_dataframe(statement_obj)
        if statement_df is None or statement_df.empty:
            base_result["reason"] = "statement_empty"
            return base_result

        period_columns = _extract_period_columns(statement_df.columns)
        rows = _build_statement_rows(statement_df, period_columns)
        periods = [_build_period_summary(period) for period in period_columns]
        units = _infer_units_from_xbrl_query(xbrl)
        currency = _infer_currency_from_units(units)
        scale = _infer_scale_from_xbrl_query(xbrl)

        return {
            "statement_type": statement_type,
            "periods": periods,
            "rows": rows,
            "currency": currency,
            "units": units,
            "scale": scale,
            "data_quality": "xbrl" if rows else "partial",
            "statement_locator": build_statement_locator(
                statement_type=statement_type,
                periods=periods,
                rows=rows,
            ),
        }

    def query_xbrl_facts(
        self,
        concepts: list[str],
        statement_type: Optional[str] = None,
        period_end: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> XbrlFactsResult:
        """查询 XBRL facts。

        Args:
            concepts: XBRL 概念列表。
            statement_type: 可选报表类型。
            period_end: 可选期末日期（YYYY-MM-DD）。
            fiscal_year: 可选财年。
            fiscal_period: 可选财季。
            min_value: 可选最小值筛选。
            max_value: 可选最大值筛选。

        Returns:
            XBRL 查询结果（仅含可解析数值 facts）。

        Raises:
            RuntimeError: 查询执行失败时抛出。
        """

        normalized_concepts = [str(item).strip() for item in concepts if str(item).strip()]
        normalized_statement_type = _normalize_query_statement_type(statement_type)
        query_params = {
            "concepts": normalized_concepts,
            "statement_type": normalized_statement_type or statement_type,
            "filters_applied": {
                "period_end": period_end,
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
                "min_value": min_value,
                "max_value": max_value,
            },
        }
        if not normalized_concepts:
            return {
                "query_params": query_params,
                "facts": [],
                "total": 0,
            }

        xbrl = self._get_xbrl()
        if xbrl is None:
            return {
                "query_params": query_params,
                "facts": [],
                "total": 0,
                "data_quality": "partial",
                "reason": "xbrl_not_available",
            }

        rows = _query_facts_rows(
            xbrl=xbrl,
            concepts=normalized_concepts,
            statement_type=normalized_statement_type,
            period_end=period_end,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            min_value=min_value,
            max_value=max_value,
        )
        facts = [_normalize_fact_row(row) for row in rows]
        return {
            "query_params": query_params,
            "facts": facts,
            "total": len(facts),
        }

    def _get_xbrl(self) -> Optional[XBRL]:
        """延迟加载并缓存 XBRL 对象。

        Args:
            无。

        Returns:
            `XBRL` 实例或 `None`。

        Raises:
            RuntimeError: XBRL 构建失败时抛出。
        """

        if self._xbrl_loaded:
            return self._xbrl

        self._xbrl_loaded = True
        xbrl_files = discover_xbrl_files(self._source_path.parent)
        instance_file = xbrl_files.get("instance")
        schema_file = xbrl_files.get("schema")
        if instance_file is None or schema_file is None:
            self._xbrl = None
            return None
        try:
            self._xbrl = XBRL.from_files(
                instance_file=instance_file,
                schema_file=schema_file,
                presentation_file=xbrl_files.get("presentation"),
                calculation_file=xbrl_files.get("calculation"),
                definition_file=xbrl_files.get("definition"),
                label_file=xbrl_files.get("label"),
            )
        except Exception as exc:
            Log.warn(f"XBRL 加载失败，将降级为无 XBRL 模式: {exc}", module=self.MODULE)
            self._xbrl = None
        return self._xbrl

    def get_xbrl_taxonomy(self) -> Optional[str]:
        """读取当前文档 XBRL taxonomy。

        Args:
            无。

        Returns:
            taxonomy（`us-gaap` / `ifrs-full`）或 `None`。

        Raises:
            RuntimeError: 解析失败时抛出。
        """

        if self._xbrl_taxonomy_loaded:
            return self._xbrl_taxonomy
        self._xbrl_taxonomy_loaded = True
        xbrl = self._get_xbrl()
        if xbrl is None:
            self._xbrl_taxonomy = None
            return None
        self._xbrl_taxonomy = _infer_xbrl_taxonomy(xbrl)
        return self._xbrl_taxonomy


def _parse_document(html_content: str, form_type: Optional[str]) -> Any:
    """调用 edgartools 解析文档。

    Args:
        html_content: HTML/XML 文本内容。
        form_type: 已标准化的 SEC 表单类型。

    Returns:
        edgartools 文档对象。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    config = _build_parser_config(
        form_type=form_type,
        max_document_size=_EDGAR_MAX_DOCUMENT_SIZE_BYTES,
    )
    try:
        # return parse_html(html_content, config=config)
        return HTMLParser(config).parse(html_content)
    except DocumentTooLargeError:
        # 超大文档专用重试：仅在触发体积上限时放大一次阈值，避免掩盖其他解析错误。
        retry_max_document_size = _EDGAR_MAX_DOCUMENT_SIZE_BYTES * _EDGAR_MAX_DOCUMENT_SIZE_RETRY_MULTIPLIER
        retry_config = _build_parser_config(
            form_type=form_type,
            max_document_size=retry_max_document_size,
        )
        try:
            return HTMLParser(retry_config).parse(html_content)
        except Exception as retry_exc:
            raise RuntimeError("SEC document parsing failed") from retry_exc
    except Exception as exc:
        raise RuntimeError("SEC document parsing failed") from exc


def _build_parser_config(form_type: Optional[str], max_document_size: int) -> ParserConfig:
    """构建 edgartools 解析配置。

    Args:
        form_type: 已标准化的 SEC 表单类型。
        max_document_size: 本次解析允许的最大文档体积（字节）。

    Returns:
        解析配置对象。

    Raises:
        ValueError: 参数非法时抛出。
    """

    if form_type:
        return ParserConfig(form=form_type, max_document_size=max_document_size)
    return ParserConfig(max_document_size=max_document_size)


def _load_text(source_path: Path) -> str:
    """读取文本文件内容。

    Args:
        source_path: 源文件路径。

    Returns:
        文本内容。

    Raises:
        OSError: 读取失败时抛出。
    """

    return source_path.read_text(encoding="utf-8", errors="ignore")

