"""Docling JSON 文档处理器实现。

该模块实现 `DocumentProcessor` 协议，用于读取单个 `*_docling.json` 文档，
提供章节、表格、章节内容与搜索能力。
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypeVar, cast, overload

import pandas as pd

if TYPE_CHECKING:
    from docling_core.types.doc.document import DoclingDocument, NodeItem, TableItem

from .base import (
    PageContentResult,
    SearchHit,
    SectionContent,
    SectionSummary,
    TableContent,
    TableSummary,
    build_page_content_result,
    build_search_hit,
    build_section_content,
    build_section_summary,
    build_table_content,
    build_table_summary,
)
from .search_utils import enrich_hits_by_section
from .source import Source
from .text_utils import (
    PREVIEW_MAX_CHARS as _PREVIEW_MAX_CHARS,
    append_missing_table_placeholders as _append_missing_placeholders,
    format_section_ref as _format_section_ref,
    format_table_placeholder as _format_table_placeholder,
    format_table_ref as _format_table_ref,
    infer_suffix_from_uri as _infer_suffix_from_uri,
    normalize_optional_string as _normalize_optional_string,
    normalize_whitespace as _normalize_whitespace,
)
from .perf_utils import ProcessorStageProfiler, is_processor_profile_enabled

_LOW_INFO_TOKENS = {"", "-", "--", "—", "n/a", "na", "none", "nil"}
_SECTION_CONTENT_CACHE_MAX_ENTRIES = 256

_CellValueT = TypeVar("_CellValueT")


@dataclass
class _LinearItem:
    """线性阅读序内部结构。"""

    index: int
    item_type: str
    internal_ref: Optional[str]
    page_no: Optional[int]
    level: int
    text: str
    label: Optional[str]
    object_ref: Optional[NodeItem]


@dataclass
class _SectionBlock:
    """章节索引内部结构。"""

    ref: str
    title: Optional[str]
    level: int
    parent_ref: Optional[str]
    preview: str
    start_index: int
    end_index: int
    page_range: Optional[list[int]]
    table_refs: list[str]
    contains_full_text: bool
    header_internal_ref: Optional[str]


@dataclass
class _TableBlock:
    """表格索引内部结构。"""

    table_ref: str
    internal_ref: Optional[str]
    table_item: TableItem
    page_no: Optional[int]
    row_count: int
    col_count: int
    headers: Optional[list[str]]
    section_ref: Optional[str]
    context_before: str
    table_type: str
    caption: Optional[str]


class DoclingProcessor:
    """Docling JSON 文档处理器。"""

    PARSER_VERSION = "docling_processor_v1.1.0"

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
            无。

        Raises:
            ValueError: 源文件不存在或格式非法时抛出。
            RuntimeError: Docling 依赖缺失或文档解析失败时抛出。
        """

        self._source = source
        self._form_type = form_type
        self._media_type = media_type or source.media_type
        self._profiler = ProcessorStageProfiler(
            component=self.__class__.__name__,
            enabled=is_processor_profile_enabled(),
        )
        source_path = source.materialize(suffix=".json")
        if not source_path.exists() or not source_path.is_file():
            raise ValueError(f"Docling JSON 文件不存在: {source_path}")
        self._source_path = source_path

        with self._profiler.stage("load_docling_document"):
            self._document = _load_docling_document(source_path)
        with self._profiler.stage("build_linear_items"):
            self._linear_items = _build_linear_items(self._document)
        with self._profiler.stage("build_tables"):
            self._tables, table_internal_ref_map = _build_tables(self._document, self._linear_items)
        with self._profiler.stage("build_sections"):
            self._sections = _build_sections(self._linear_items, table_internal_ref_map)
        with self._profiler.stage("attach_table_sections"):
            self._tables = _attach_table_sections(self._tables, self._sections)

        self._section_by_ref = {section.ref: section for section in self._sections}
        self._table_by_ref = {table.table_ref: table for table in self._tables}
        self._table_ref_by_internal_ref = {
            table.internal_ref: table.table_ref
            for table in self._tables
            if table.internal_ref
        }
        self._section_content_cache: OrderedDict[str, tuple[tuple[str, ...], str]] = OrderedDict()
        self._full_text_cache: Optional[str] = None
        self._profiler.log_summary(extra=f"uri={self._source.uri}")

    def get_section_title(self, ref: str) -> Optional[str]:
        """根据 section ref 获取章节标题。

        Args:
            ref: 章节引用。

        Returns:
            章节标题字符串；ref 不存在时返回 None。
        """
        section = self._section_by_ref.get(ref)
        return section.title if section else None

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
            form_type: 可选表单类型（当前不参与判断）。
            media_type: 可选媒体类型。

        Returns:
            是否支持。

        Raises:
            OSError: 访问文件失败时可能抛出。
        """

        del form_type
        uri = str(source.uri or "").strip().lower()
        if uri.endswith("_docling.json"):
            return True

        resolved_media_type = str(media_type or source.media_type or "").lower()
        if "json" not in resolved_media_type:
            return False
        if _infer_suffix_from_uri(source.uri) != ".json":
            return False
        return _sniff_docling_json(source)

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
                build_section_summary(
                    ref=section.ref,
                    title=section.title,
                    level=section.level,
                    parent_ref=section.parent_ref,
                    preview=section.preview,
                    page_range=section.page_range,
                    internal_ref=section.header_internal_ref,
                )
                for section in self._sections
            ]
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError("Docling section parsing failed") from exc

    def _extra_table_fields(self, table: _TableBlock) -> dict:
        """返回嵌入到表格输出字典的额外字段（供子类覆盖）。

        Args:
            table: 内部表格对象。

        Returns:
            额外字段字典，默认为空字典。
        """
        return {}

    def list_tables(self) -> list[TableSummary]:
        """读取表格列表。

        Args:
            无。

        Returns:
            表格摘要列表。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        try:
            result = []
            for table in self._tables:
                summary = build_table_summary(
                    table_ref=table.table_ref,
                    caption=table.caption,
                    context_before=table.context_before,
                    row_count=table.row_count,
                    col_count=table.col_count,
                    table_type=table.table_type,
                    headers=table.headers,
                    section_ref=table.section_ref,
                    page_no=table.page_no,
                    internal_ref=table.internal_ref,
                )
                extra = self._extra_table_fields(table)
                if extra.get("is_financial") is not None:
                    summary["is_financial"] = bool(extra["is_financial"])
                result.append(summary)
            return result
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError("Docling table parsing failed") from exc

    def read_section(self, ref: str) -> SectionContent:
        """按 ref 读取章节内容。

        Args:
            ref: 章节引用。

        Returns:
            章节内容字典。

        Raises:
            KeyError: 章节不存在时抛出。
            RuntimeError: 渲染失败时抛出。
        """

        section = self._section_by_ref.get(ref)
        if section is None:
            raise KeyError(f"Section not found: {ref}")

        try:
            with self._profiler.stage("read_section"):
                content = self._get_or_render_section_content(section)
            word_count = len(content.split())
            return build_section_content(
                ref=section.ref,
                title=section.title,
                content=content,
                tables=list(section.table_refs),
                word_count=word_count,
                contains_full_text=section.contains_full_text,
                page_range=section.page_range,
                internal_ref=section.header_internal_ref,
            )
        except Exception as exc:  # pragma: no cover - 兜底保护
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
            # docling 独有的额外字段
            docling_extra = {
                "page_no": table.page_no,
                "internal_ref": table.internal_ref,
                **self._extra_table_fields(table),
            }
            records_payload = _render_records_table(table.table_item, self._document)
            if records_payload is not None:
                return build_table_content(
                    table_ref=table.table_ref,
                    caption=table.caption,
                    data_format="records",
                    data=records_payload["data"],
                    columns=records_payload["columns"],
                    row_count=table.row_count,
                    col_count=table.col_count,
                    section_ref=table.section_ref,
                    table_type=table.table_type,
                    **docling_extra,
                )
            markdown_text = _render_markdown_table(table.table_item, self._document)
            return build_table_content(
                table_ref=table.table_ref,
                caption=table.caption,
                data_format="markdown",
                data=markdown_text,
                columns=None,
                row_count=table.row_count,
                col_count=table.col_count,
                section_ref=table.section_ref,
                table_type=table.table_type,
                **docling_extra,
            )
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError(f"Table read failed: {table_ref}") from exc

    def search(self, query: str, within_ref: Optional[str] = None) -> list[SearchHit]:
        """在文档中搜索关键词。

        Args:
            query: 搜索词。
            within_ref: 可选章节范围。

        Returns:
            命中列表。

        Raises:
            RuntimeError: 搜索失败时抛出。
        """

        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []
        if within_ref is not None and within_ref not in self._section_by_ref:
            return []

        target_sections = (
            [self._section_by_ref[within_ref]]
            if within_ref is not None
            else self._sections
        )
        hits_raw: list[SearchHit] = []
        section_content_map: dict[str, str] = {}
        with self._profiler.stage("search"):
            for section in target_sections:
                content = self._get_or_render_section_content(section)
                title_text = section.title or ""
                title_hit = bool(title_text) and re.search(re.escape(normalized_query), title_text, flags=re.IGNORECASE) is not None
                content_hit = re.search(re.escape(normalized_query), content, flags=re.IGNORECASE) is not None
                if not title_hit and not content_hit:
                    continue
                # 若 title 命中而 content 无命中，将 title 前置进搜索文本，确保 snippet 能定位到匹配词。
                searchable_text = (
                    (title_text + "\n" + content).strip()
                    if title_hit and not content_hit
                    else content
                )
                section_content_map[section.ref] = searchable_text
                hits_raw.append(
                    build_search_hit(
                        section_ref=section.ref,
                        section_title=section.title,
                        snippet=normalized_query,
                        page_no=_pick_snippet_page_no(section.page_range),
                    )
                )
        return enrich_hits_by_section(
            hits_raw=hits_raw,
            section_content_map=section_content_map,
            query=normalized_query,
        )

    def get_full_text(self) -> str:
        """获取文档的完整纯文本内容。

        遍历所有章节读取内容并拼接为完整全文。

        Args:
            无。

        Returns:
            文档完整纯文本字符串。

        Raises:
            RuntimeError: 提取失败时抛出。
        """

        if self._full_text_cache is not None:
            return self._full_text_cache
        parts: list[str] = []
        with self._profiler.stage("get_full_text"):
            for section in self._sections:
                text = self._get_or_render_section_content(section).strip()
                if text:
                    parts.append(text)
        self._full_text_cache = "\n".join(parts)
        return self._full_text_cache

    def _get_or_render_section_content(self, section: _SectionBlock) -> str:
        """读取或渲染章节正文缓存。

        Args:
            section: 章节对象。

        Returns:
            章节正文文本。

        Raises:
            RuntimeError: 渲染失败时抛出。
        """

        cache_key_refs = tuple(section.table_refs)
        cached_entry = self._section_content_cache.get(section.ref)
        if cached_entry is not None and cached_entry[0] == cache_key_refs:
            self._section_content_cache.move_to_end(section.ref, last=True)
            return cached_entry[1]
        section_items = self._linear_items[section.start_index:section.end_index]
        rendered = _render_section_content(
            section_items=section_items,
            table_ref_by_internal_ref=self._table_ref_by_internal_ref,
            declared_table_refs=section.table_refs,
        )
        content = str(rendered.get("content", "") or "")
        self._section_content_cache[section.ref] = (cache_key_refs, content)
        self._section_content_cache.move_to_end(section.ref, last=True)
        while len(self._section_content_cache) > _SECTION_CONTENT_CACHE_MAX_ENTRIES:
            self._section_content_cache.popitem(last=False)
        return content

    def get_full_text_with_table_markers(self) -> str:
        """获取带表格占位符的全文（DoclingProcessor 不支持）。

        DoclingProcessor 基于 JSON 结构解析，不具备 DOM 级表格标记
        注入能力，返回空字符串表示不支持。

        Args:
            无。

        Returns:
            空字符串。
        """

        return ""

    def get_page_content(self, page_no: int) -> PageContentResult:
        """读取指定页码的上下文内容。

        Args:
            page_no: 页码（1-based）。

        Returns:
            页面内容结果。

        Raises:
            ValueError: `page_no` 非法时抛出。
            RuntimeError: 读取失败时抛出。
        """

        if not isinstance(page_no, int) or page_no <= 0:
            raise ValueError("page_no must be a positive integer")

        try:
            sections = _build_page_sections(
                page_no=page_no,
                sections=self._sections,
                linear_items=self._linear_items,
                table_ref_by_internal_ref=self._table_ref_by_internal_ref,
            )
            tables = _build_page_tables(page_no=page_no, tables=self._tables)
            text_preview = _build_page_text_preview(page_no=page_no, linear_items=self._linear_items)
            has_content = bool(sections or tables or text_preview)
            return build_page_content_result(
                page_no=page_no,
                sections=sections,
                tables=tables,
                text_preview=text_preview,
                has_content=has_content,
                total_items=len(sections) + len(tables),
                supported=True,
            )
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError(f"页面内容读取失败: page_no={page_no}") from exc


def _load_docling_document(source_path: Path) -> "DoclingDocument":
    """加载 DoclingDocument。

    Args:
        source_path: Docling JSON 文件路径。

    Returns:
        DoclingDocument 实例。

    Raises:
        RuntimeError: docling-core 缺失或加载失败时抛出。
    """

    try:
        from docling_core.types.doc.document import DoclingDocument
    except ImportError as exc:  # pragma: no cover - 依赖缺失保护
        raise RuntimeError("docling-core 未安装，无法读取 Docling JSON") from exc

    try:
        return cast("DoclingDocument", DoclingDocument.load_from_json(str(source_path)))
    except Exception as exc:  # pragma: no cover - 第三方异常兜底
        raise RuntimeError(f"Docling JSON parsing failed: {source_path.name}") from exc


def _build_linear_items(document: DoclingDocument) -> list[_LinearItem]:
    """构建线性阅读序列。

    Args:
        document: DoclingDocument 实例。

    Returns:
        线性 item 列表。

    Raises:
        RuntimeError: 遍历失败时抛出。
    """

    linear_items: list[_LinearItem] = []
    iterator = document.iterate_items(with_groups=False)
    for index, (item, level) in enumerate(iterator):
        item_type = _resolve_item_type(item)
        if item_type is None:
            continue
        if item_type == "text" and _is_text_inside_table(item):
            # 复杂逻辑说明：表格子节点文本不应进入 section 正文，避免 read_section 泄漏表格内容。
            continue

        item_text = _extract_item_text(item) if item_type == "text" else ""
        label = _normalize_label(getattr(item, "label", None))
        internal_ref = _extract_internal_ref(item)
        linear_items.append(
            _LinearItem(
                index=index,
                item_type=item_type,
                internal_ref=internal_ref,
                page_no=_extract_page_no(item),
                level=max(1, int(level) + 1),
                text=item_text,
                label=label,
                object_ref=item,
            )
        )
    return linear_items


def _build_tables(
    document: DoclingDocument,
    linear_items: list[_LinearItem],
) -> tuple[list[_TableBlock], dict[str, str]]:
    """构建表格索引。

    Args:
        document: DoclingDocument 实例。
        linear_items: 线性 items。

    Returns:
        (表格块列表, internal_ref->table_ref 映射)。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    tables: list[_TableBlock] = []
    internal_ref_map: dict[str, str] = {}
    table_items = _iter_document_tables(document)

    for index, table_item in enumerate(table_items, start=1):
        table_ref = _format_table_ref(index)
        internal_ref = _extract_internal_ref(table_item)
        if internal_ref:
            internal_ref_map[internal_ref] = table_ref

        row_count, col_count = _resolve_table_dimensions(table_item, document)
        headers = _extract_table_headers(table_item, document)
        context_before = _extract_table_context_before(table_item, linear_items)
        caption = _extract_table_caption(table_item)
        table_type = _classify_table_type(
            row_count=row_count,
            col_count=col_count,
            headers=headers,
            context_before=context_before,
        )

        tables.append(
            _TableBlock(
                table_ref=table_ref,
                internal_ref=internal_ref,
                table_item=table_item,
                page_no=_extract_page_no(table_item),
                row_count=row_count,
                col_count=col_count,
                headers=headers,
                section_ref=None,
                context_before=context_before,
                table_type=table_type,
                caption=caption,
            )
        )
    return tables, internal_ref_map


def _build_sections(linear_items: list[_LinearItem], table_internal_ref_map: dict[str, str]) -> list[_SectionBlock]:
    """根据线性序构建章节索引。

    Args:
        linear_items: 线性 item 列表。
        table_internal_ref_map: 表格内部引用映射。

    Returns:
        章节块列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    header_indices = [
        item.index
        for item in linear_items
        if item.item_type == "text" and item.label == "section_header"
    ]

    if not linear_items:
        return [
            _SectionBlock(
                ref=_format_section_ref(1),
                title=None,
                level=1,
                parent_ref=None,
                preview="",
                start_index=0,
                end_index=0,
                page_range=None,
                table_refs=[],
                contains_full_text=True,
                header_internal_ref=None,
            )
        ]

    if not header_indices:
        table_refs = _collect_section_table_refs(
            section_items=linear_items,
            table_ref_by_internal_ref=table_internal_ref_map,
        )
        preview = _build_section_preview(linear_items, table_internal_ref_map)
        return [
            _SectionBlock(
                ref=_format_section_ref(1),
                title=None,
                level=1,
                parent_ref=None,
                preview=preview,
                start_index=0,
                end_index=len(linear_items),
                page_range=_extract_page_range(linear_items),
                table_refs=table_refs,
                contains_full_text=True,
                header_internal_ref=None,
            )
        ]

    sections: list[_SectionBlock] = []
    level_stack: list[tuple[int, str]] = []
    linear_index_map = {item.index: idx for idx, item in enumerate(linear_items)}

    for section_idx, header_linear_index in enumerate(header_indices, start=1):
        start = linear_index_map[header_linear_index]
        next_linear_index = header_indices[section_idx] if section_idx < len(header_indices) else None
        end = len(linear_items) if next_linear_index is None else linear_index_map[next_linear_index]
        section_items = linear_items[start:end]
        header_item = section_items[0]
        section_level = max(1, int(header_item.level))

        while level_stack and level_stack[-1][0] >= section_level:
            level_stack.pop()
        parent_ref = level_stack[-1][1] if level_stack else None
        section_ref = _format_section_ref(section_idx)
        level_stack.append((section_level, section_ref))

        table_refs = _collect_section_table_refs(
            section_items=section_items,
            table_ref_by_internal_ref=table_internal_ref_map,
        )
        sections.append(
            _SectionBlock(
                ref=section_ref,
                title=header_item.text or None,
                level=section_level,
                parent_ref=parent_ref,
                preview=_build_section_preview(section_items, table_internal_ref_map),
                start_index=start,
                end_index=end,
                page_range=_extract_page_range(section_items),
                table_refs=table_refs,
                contains_full_text=False,
                header_internal_ref=header_item.internal_ref,
            )
        )

    return sections


def _attach_table_sections(tables: list[_TableBlock], sections: list[_SectionBlock]) -> list[_TableBlock]:
    """把表格挂载到所属章节。

    Args:
        tables: 表格块列表。
        sections: 章节块列表。

    Returns:
        更新后的表格块列表。

    Raises:
        RuntimeError: 挂载失败时抛出。
    """

    table_to_section: dict[str, str] = {}
    for section in sections:
        for table_ref in section.table_refs:
            table_to_section.setdefault(table_ref, section.ref)

    for table in tables:
        table.section_ref = table_to_section.get(table.table_ref)
    return tables


def _render_section_content(
    *,
    section_items: list[_LinearItem],
    table_ref_by_internal_ref: dict[str, str],
    declared_table_refs: list[str],
) -> dict[str, Any]:
    """渲染章节正文并替换表格占位符。

    Args:
        section_items: 章节线性 item 切片。
        table_ref_by_internal_ref: 表格 internal_ref 映射。
        declared_table_refs: 章节声明的表格 ref。

    Returns:
        包含 `content` 与 `used_table_refs` 的字典。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    parts: list[str] = []
    used_table_refs: list[str] = []

    for item in section_items:
        if item.item_type == "table":
            table_ref = _resolve_table_ref_for_item(item, table_ref_by_internal_ref)
            if table_ref is None:
                continue
            placeholder = _format_table_placeholder(table_ref)
            parts.append(placeholder)
            if table_ref not in used_table_refs:
                used_table_refs.append(table_ref)
            continue
        if item.item_type != "text":
            continue
        text = _normalize_whitespace(item.text)
        if text:
            parts.append(text)

    content = _normalize_whitespace(" ".join(parts))
    unresolved_refs = [ref for ref in declared_table_refs if ref not in used_table_refs]
    content = _append_missing_placeholders(content, unresolved_refs)
    return {
        "content": content,
        "used_table_refs": used_table_refs,
    }


def _render_records_table(
    table_item: TableItem,
    document: DoclingDocument,
) -> Optional[dict[str, Any]]:
    """渲染 records 表格数据。

    Args:
        table_item: Docling 表格对象。
        document: Docling 文档对象。

    Returns:
        records 载荷；无法稳定渲染时返回 `None`。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    table_df = _safe_table_dataframe(table_item, document)
    if table_df is None or table_df.empty:
        return None
    if table_df.columns.has_duplicates:
        return None

    columns = [str(column) for column in table_df.columns.tolist()]
    records = table_df.to_dict(orient="records")
    normalized_records: list[dict[str, Any]] = []
    for row in records:
        normalized_row = {
            str(key): _normalize_cell_value(value)
            for key, value in row.items()
        }
        normalized_records.append(normalized_row)
    return {
        "columns": columns,
        "data": normalized_records,
    }


def _render_markdown_table(
    table_item: TableItem,
    document: DoclingDocument,
) -> str:
    """渲染 markdown 表格数据。

    Args:
        table_item: Docling 表格对象。
        document: Docling 文档对象。

    Returns:
        markdown 字符串。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    exporter = getattr(table_item, "export_to_markdown", None)
    if callable(exporter):
        try:
            markdown = exporter(doc=document)
            if isinstance(markdown, str) and markdown.strip():
                return markdown
        except TypeError:
            markdown = exporter()
            if isinstance(markdown, str) and markdown.strip():
                return markdown
        except Exception:
            pass

    table_df = _safe_table_dataframe(table_item, document)
    if table_df is not None and not table_df.empty:
        try:
            from tabulate import tabulate as _tabulate

            del _tabulate
            # NaN 在合并单元格中常见，转 markdown 前统一填为空字符串
            cleaned_df = table_df.fillna("")
            markdown = cleaned_df.to_markdown(index=False)
            if isinstance(markdown, str) and markdown.strip():
                return markdown
            raise RuntimeError("Docling 表格 markdown 渲染结果为空")
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少 tabulate 依赖，无法渲染 Docling 表格 markdown") from exc
        except Exception as exc:
            raise RuntimeError(f"Docling 表格 markdown 渲染失败: {exc}") from exc

    raw_dict = getattr(table_item, "data", None)
    return _normalize_whitespace(str(raw_dict or {}))


def _resolve_item_type(item: NodeItem) -> Optional[str]:
    """解析 item 类型。

    Args:
        item: Docling item。

    Returns:
        `text/table/picture` 或 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    class_name = item.__class__.__name__.lower()
    if "table" in class_name:
        return "table"
    if "text" in class_name or "sectionheader" in class_name:
        return "text"
    if "picture" in class_name:
        return "picture"
    return None


def _extract_item_text(item: NodeItem) -> str:
    """提取文本 item 内容。

    Args:
        item: Docling item。

    Returns:
        文本内容。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    text = getattr(item, "text", "")
    return _normalize_whitespace(str(text or ""))


def _extract_internal_ref(item: NodeItem) -> Optional[str]:
    """提取 item 的 internal_ref。

    Args:
        item: Docling item。

    Returns:
        internal_ref 字符串；不存在时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    self_ref = getattr(item, "self_ref", None)
    normalized = _normalize_optional_string(self_ref)
    if normalized is not None:
        return normalized

    get_ref_method = getattr(item, "get_ref", None)
    if callable(get_ref_method):
        try:
            raw_ref = get_ref_method()
        except Exception:
            return None
        return _normalize_optional_string(raw_ref)
    return None


def _extract_parent_ref(item: NodeItem) -> Optional[str]:
    """提取 item parent 引用。

    Args:
        item: Docling item。

    Returns:
        parent 引用字符串。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    parent = getattr(item, "parent", None)
    if parent is None:
        return None
    if isinstance(parent, str):
        return _normalize_optional_string(parent)

    parent_ref = getattr(parent, "$ref", None)
    normalized = _normalize_optional_string(parent_ref)
    if normalized is not None:
        return normalized

    cref = getattr(parent, "cref", None)
    normalized = _normalize_optional_string(cref)
    if normalized is not None:
        return normalized

    ref_attr = getattr(parent, "ref", None)
    return _normalize_optional_string(ref_attr)


def _is_text_inside_table(item: NodeItem) -> bool:
    """判断文本项是否位于表格内部。

    Args:
        item: Docling item。

    Returns:
        是否属于表格子项。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    parent_ref = _extract_parent_ref(item)
    if parent_ref is None:
        return False
    return parent_ref.startswith("#/tables/")


def _extract_page_no(item: NodeItem) -> Optional[int]:
    """提取 item 所在页码。

    Args:
        item: Docling item。

    Returns:
        页码（1-based）；不存在时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    prov_list = getattr(item, "prov", None)
    if not isinstance(prov_list, list) or not prov_list:
        return None
    first = prov_list[0]
    page_no = getattr(first, "page_no", None)
    if isinstance(page_no, int) and page_no > 0:
        return page_no
    return None


def _normalize_label(raw_label: object) -> Optional[str]:
    """标准化 item 标签。

    Args:
        raw_label: 原始标签对象。

    Returns:
        小写标签；空值返回 `None`。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    if raw_label is None:
        return None
    if hasattr(raw_label, "value"):
        raw_label = getattr(raw_label, "value")
    normalized = _normalize_optional_string(raw_label)
    if normalized is None:
        return None
    return normalized.lower()


def _iter_document_tables(document: DoclingDocument) -> list[TableItem]:
    """遍历文档表格对象。

    Args:
        document: DoclingDocument 对象。

    Returns:
        表格对象列表。

    Raises:
        RuntimeError: 遍历失败时抛出。
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


def _resolve_table_dimensions(
    table_item: TableItem,
    document: DoclingDocument,
) -> tuple[int, int]:
    """解析表格行列数。

    Args:
        table_item: 表格对象。
        document: Docling 文档对象。

    Returns:
        (row_count, col_count)。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    table_data = getattr(table_item, "data", None)
    row_count = int(getattr(table_data, "num_rows", 0) or 0)
    col_count = int(getattr(table_data, "num_cols", 0) or 0)
    if row_count > 0 and col_count > 0:
        return row_count, col_count

    table_df = _safe_table_dataframe(table_item, document)
    if table_df is None:
        return row_count, col_count
    return int(table_df.shape[0]), int(table_df.shape[1])


def _extract_table_headers(
    table_item: TableItem,
    document: DoclingDocument,
) -> Optional[list[str]]:
    """提取表头（优先行头）。

    Args:
        table_item: 表格对象。
        document: Docling 文档对象。

    Returns:
        表头列表；无法提取时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    table_df = _safe_table_dataframe(table_item, document)
    if table_df is None or table_df.empty:
        return None

    row_headers: list[str] = []
    first_column = table_df.iloc[:, 0].tolist()
    for raw_value in first_column:
        text = _normalize_optional_string(raw_value)
        if text is None:
            continue
        if _is_low_information_header(text):
            continue
        row_headers.append(text)
        if len(row_headers) >= 10:
            break
    deduped_row_headers = _deduplicate_headers(row_headers)
    if deduped_row_headers:
        return deduped_row_headers

    columns = [_normalize_optional_string(column) for column in table_df.columns.tolist()]
    normalized_columns = [column for column in columns if column is not None]
    if not normalized_columns or _looks_like_default_headers(normalized_columns):
        return None
    return _deduplicate_headers(normalized_columns[:10])


def _extract_table_caption(table_item: TableItem) -> Optional[str]:
    """提取表格标题。

    Args:
        table_item: 表格对象。

    Returns:
        标题字符串；不存在时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    caption_obj = getattr(table_item, "caption", None)
    if caption_obj is None:
        return None
    caption_text = getattr(caption_obj, "text", caption_obj)
    return _normalize_optional_string(caption_text)


def _extract_table_context_before(
    table_item: TableItem,
    linear_items: list[_LinearItem],
    max_chars: int = 200,
) -> str:
    """提取表格前文。

    Args:
        table_item: 表格对象。
        linear_items: 线性 item 列表。
        max_chars: 前文最大长度。

    Returns:
        前文字符串。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    table_ref = _extract_internal_ref(table_item)
    if table_ref is None:
        return ""

    target_index = None
    for idx, item in enumerate(linear_items):
        if item.internal_ref == table_ref and item.item_type == "table":
            target_index = idx
            break
    if target_index is None:
        return ""

    text_parts: list[str] = []
    total_len = 0
    for idx in range(target_index - 1, -1, -1):
        item = linear_items[idx]
        if item.item_type != "text":
            continue
        if item.label == "section_header":
            break
        text = _normalize_whitespace(item.text)
        if not text:
            continue
        text_parts.append(text)
        total_len += len(text)
        if total_len >= max_chars:
            break

    if not text_parts:
        return ""
    text_parts.reverse()
    combined = _normalize_whitespace(" ".join(text_parts))
    if len(combined) <= max_chars:
        return combined
    return combined[-max_chars:]


def _collect_section_table_refs(
    section_items: list[_LinearItem],
    table_ref_by_internal_ref: dict[str, str],
) -> list[str]:
    """收集章节包含的表格引用列表。

    Args:
        section_items: 章节 item 列表。
        table_ref_by_internal_ref: 表格映射。

    Returns:
        表格 ref 列表（按出现顺序，去重）。

    Raises:
        RuntimeError: 收集失败时抛出。
    """

    refs: list[str] = []
    for item in section_items:
        if item.item_type != "table":
            continue
        table_ref = _resolve_table_ref_for_item(item, table_ref_by_internal_ref)
        if table_ref is None:
            continue
        if table_ref in refs:
            continue
        refs.append(table_ref)
    return refs


def _resolve_table_ref_for_item(
    item: _LinearItem,
    table_ref_by_internal_ref: dict[str, str],
) -> Optional[str]:
    """为线性 item 解析 table_ref。

    Args:
        item: 线性 item。
        table_ref_by_internal_ref: 表格映射。

    Returns:
        table_ref 或 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    if item.internal_ref is None:
        return None
    return table_ref_by_internal_ref.get(item.internal_ref)


def _extract_page_range(section_items: list[_LinearItem]) -> Optional[list[int]]:
    """提取章节页码范围。

    Args:
        section_items: 章节 item 列表。

    Returns:
        `[start_page, end_page]` 或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    page_nos = [item.page_no for item in section_items if isinstance(item.page_no, int)]
    if not page_nos:
        return None
    return [min(page_nos), max(page_nos)]


def _build_section_preview(
    section_items: list[_LinearItem],
    table_ref_by_internal_ref: dict[str, str],
    max_chars: int = 200,
) -> str:
    """构建章节 preview。

    Args:
        section_items: 章节 item 列表。
        table_ref_by_internal_ref: 表格映射。
        max_chars: 预览最大长度。

    Returns:
        预览文本。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    rendered = _render_section_content(
        section_items=section_items,
        table_ref_by_internal_ref=table_ref_by_internal_ref,
        declared_table_refs=[],
    )
    preview = str(rendered.get("content", ""))
    if len(preview) <= max_chars:
        return preview
    return preview[:max_chars]


def _pick_snippet_page_no(page_range: Optional[list[int]]) -> Optional[int]:
    """从章节页码范围选择命中页码。

    Args:
        page_range: 章节页码范围。

    Returns:
        命中页码。

    Raises:
        RuntimeError: 选择失败时抛出。
    """

    if not page_range or len(page_range) < 1:
        return None
    first_page = page_range[0]
    if isinstance(first_page, int) and first_page > 0:
        return first_page
    return None


def _build_page_sections(
    *,
    page_no: int,
    sections: list[_SectionBlock],
    linear_items: list[_LinearItem],
    table_ref_by_internal_ref: dict[str, str],
) -> list[SectionSummary]:
    """构建页面章节片段列表。

    Args:
        page_no: 目标页码（1-based）。
        sections: 全量章节块。
        linear_items: 文档线性 item。
        table_ref_by_internal_ref: 表格映射。

    Returns:
        页面章节片段列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    payloads: list[SectionSummary] = []
    for section in sections:
        page_range = section.page_range
        if not page_range or len(page_range) != 2:
            continue
        start_page, end_page = page_range
        if not isinstance(start_page, int) or not isinstance(end_page, int):
            continue
        if page_no < start_page or page_no > end_page:
            continue
        preview = _build_page_section_preview(
            page_no=page_no,
            section=section,
            linear_items=linear_items,
            table_ref_by_internal_ref=table_ref_by_internal_ref,
            max_chars=200,
        )
        payloads.append(
            build_section_summary(
                ref=section.ref,
                title=section.title,
                level=section.level,
                parent_ref=section.parent_ref,
                preview=preview,
                page_range=section.page_range,
                internal_ref=section.header_internal_ref,
            )
        )
    return payloads


def _build_page_section_preview(
    *,
    page_no: int,
    section: _SectionBlock,
    linear_items: list[_LinearItem],
    table_ref_by_internal_ref: dict[str, str],
    max_chars: int,
) -> str:
    """构建章节在指定页的预览文本。

    Args:
        page_no: 目标页码。
        section: 章节块。
        linear_items: 文档线性 item。
        table_ref_by_internal_ref: 表格映射。
        max_chars: 预览最大字符数。

    Returns:
        预览文本。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    scoped_items = linear_items[section.start_index:section.end_index]
    page_items = [
        item
        for item in scoped_items
        if isinstance(item.page_no, int) and item.page_no == page_no
    ]
    if not page_items:
        return ""
    rendered = _render_section_content(
        section_items=page_items,
        table_ref_by_internal_ref=table_ref_by_internal_ref,
        declared_table_refs=[],
    )
    preview = str(rendered.get("content", ""))
    if len(preview) <= max_chars:
        return preview
    return preview[:max_chars]


def _build_page_tables(*, page_no: int, tables: list[_TableBlock]) -> list[TableSummary]:
    """构建页面表格列表。

    Args:
        page_no: 目标页码（1-based）。
        tables: 全量表格块。

    Returns:
        页面表格列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    payloads: list[TableSummary] = []
    for table in tables:
        if table.page_no != page_no:
            continue
        payloads.append(
            build_table_summary(
                table_ref=table.table_ref,
                caption=table.caption,
                context_before=table.context_before[:_PREVIEW_MAX_CHARS],
                row_count=table.row_count,
                col_count=table.col_count,
                table_type=table.table_type,
                headers=table.headers,
                section_ref=table.section_ref,
                page_no=table.page_no,
                internal_ref=table.internal_ref,
            )
        )
    return payloads


def _build_page_text_preview(*, page_no: int, linear_items: list[_LinearItem], max_chars: int = 500) -> str:
    """构建页面文本预览。

    Args:
        page_no: 目标页码（1-based）。
        linear_items: 文档线性 item。
        max_chars: 预览最大字符数。

    Returns:
        页面文本预览。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    parts: list[str] = []
    for item in linear_items:
        if item.item_type != "text":
            continue
        if item.page_no != page_no:
            continue
        text = _normalize_whitespace(item.text)
        if _is_noise_page_text(text):
            continue
        parts.append(text)
    preview = _normalize_whitespace(" ".join(parts))
    if len(preview) <= max_chars:
        return preview
    return preview[:max_chars]


def _is_noise_page_text(text: str) -> bool:
    """判断文本是否为页面噪音。

    Args:
        text: 原始文本。

    Returns:
        是噪音时返回 `True`。

    Raises:
        RuntimeError: 判定失败时抛出。
    """

    normalized = _normalize_whitespace(text).lower()
    if not normalized:
        return True
    if re.fullmatch(r"\d+", normalized):
        return True
    if normalized in {"page", "页", "of"}:
        return True
    return False


def _safe_table_dataframe(
    table_item: TableItem,
    document: DoclingDocument,
) -> Optional[pd.DataFrame]:
    """安全导出表格 DataFrame。

    Args:
        table_item: 表格对象。
        document: Docling 文档对象。

    Returns:
        DataFrame 或 `None`。

    Raises:
        RuntimeError: 导出失败时抛出。
    """

    exporter = getattr(table_item, "export_to_dataframe", None)
    if not callable(exporter):
        return None

    try:
        df = exporter(doc=document)
    except TypeError:
        try:
            df = exporter()
        except Exception:
            return None
    except Exception:
        return None

    if isinstance(df, pd.DataFrame):
        return df
    return None




def _classify_table_type(
    *,
    row_count: int,
    col_count: int,
    headers: Optional[list[str]],
    context_before: str,
) -> str:
    """轻量分类表格类型。

    Args:
        row_count: 行数。
        col_count: 列数。
        headers: 表头列表。
        context_before: 表格前文。

    Returns:
        `data` 或 `layout`。

    Raises:
        RuntimeError: 分类失败时抛出。
    """

    if row_count <= 2 and col_count <= 3:
        return "layout"
    if headers and _looks_like_default_headers(headers):
        return "layout"
    if not headers and len(context_before.strip()) < 12:
        return "layout"
    return "data"


@overload
def _normalize_cell_value(value: None) -> None:
    ...


@overload
def _normalize_cell_value(value: str) -> str | None:
    ...


@overload
def _normalize_cell_value(value: float) -> float | None:
    ...


@overload
def _normalize_cell_value(value: _CellValueT) -> _CellValueT | None:
    ...


def _normalize_cell_value(value: _CellValueT | str | float | None) -> _CellValueT | str | float | None:
    """标准化表格单元格值。

    Args:
        value: 原始单元格值。

    Returns:
        标准化后的值。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    if value is None:
        return None
    is_missing = pd.isna(value)
    if isinstance(is_missing, bool) and is_missing:
        return None
    if isinstance(value, str):
        normalized = _normalize_whitespace(value)
        return normalized or None
    return value


def _is_low_information_header(value: str) -> bool:
    """判断表头是否低信息量。

    Args:
        value: 表头文本。

    Returns:
        是否低信息量。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    normalized = _normalize_whitespace(value).lower()
    if normalized in _LOW_INFO_TOKENS:
        return True
    if re.fullmatch(r"\d+", normalized):
        return True
    if normalized.startswith("unnamed"):
        return True
    return False


def _looks_like_default_headers(headers: list[str]) -> bool:
    """判断是否看起来像默认列名。

    Args:
        headers: 表头列表。

    Returns:
        是否默认列名。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if not headers:
        return True
    normalized = [_normalize_whitespace(item).lower() for item in headers]
    if all(re.fullmatch(r"\d+", item or "") for item in normalized):
        return True
    if all(item.startswith("unnamed") for item in normalized if item):
        return True
    return False


def _deduplicate_headers(headers: list[str]) -> list[str]:
    """对表头去重并保持首次出现顺序。

    Args:
        headers: 原始表头列表。

    Returns:
        去重后的表头列表。

    Raises:
        RuntimeError: 去重失败时抛出。
    """

    deduped: list[str] = []
    seen: set[str] = set()
    for header in headers:
        normalized = _normalize_optional_string(header)
        if normalized is None:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        deduped.append(normalized)
    return deduped


def _sniff_docling_json(source: Source) -> bool:
    """轻量探测 JSON 是否为 Docling 产物。

    Args:
        source: 文档来源抽象。

    Returns:
        是否满足 Docling 核心字段特征。

    Raises:
        OSError: 读取源文件失败时可能抛出。
    """

    try:
        source_path = source.materialize(suffix=".json")
    except OSError:
        return False
    if not source_path.exists() or not source_path.is_file():
        return False

    try:
        payload = json.loads(source_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return False
    return _looks_like_docling_payload(payload)


def _looks_like_docling_payload(payload: Any) -> bool:
    """判断 JSON 数据是否符合 Docling 基础结构。

    Args:
        payload: JSON 反序列化对象。

    Returns:
        是否符合特征。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if not isinstance(payload, dict):
        return False
    if "body" not in payload:
        return False
    has_core_array = any(
        key in payload and isinstance(payload.get(key), list)
        for key in ("texts", "tables", "pictures", "groups")
    )
    has_pages = isinstance(payload.get("pages"), (dict, list))
    return has_core_array and has_pages
