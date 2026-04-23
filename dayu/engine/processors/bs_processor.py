"""BeautifulSoup HTML 处理器实现。

该模块实现 DocumentProcessor 协议，提供通用 HTML 文档的章节、表格、
章节内容与检索能力。

维护说明(不拆分本模块):
    本模块约 2000 行, 由 BSProcessor 类(545 行)和 45 个模块级私有
    工具函数组成. 工具函数覆盖 DOM 清洗 / 章节构建 / 表格提取 /
    渲染等区域, 但核心构建函数跨区域调用多个工具函数, 拆分只会增加
    import 复杂度. 外部仅通过 BSProcessor 类消费, 模块级函数不对外.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterable, Optional, TypeVar, overload

import pandas as pd
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from .source import Source
from .text_utils import (
    PREVIEW_MAX_CHARS as _PREVIEW_MAX_CHARS,
    clean_page_header_noise as _clean_page_header_noise,
    format_section_ref as _format_section_ref,
    format_table_ref as _format_table_ref,
    infer_caption_from_context as _infer_caption_from_context,
    infer_suffix_from_uri as _infer_suffix_from_uri,
    normalize_whitespace as _normalize_whitespace,
)

from .base import (
    SearchHit,
    SectionContent,
    SectionSummary,
    TableContent,
    TableSummary,
    build_section_content,
    build_section_summary,
    build_table_content,
    build_table_summary,
)
from .search_utils import enrich_hits_by_section, run_titled_section_search
from .perf_utils import ProcessorStageProfiler, is_processor_profile_enabled
from .table_utils import parse_html_table_dataframe

# HTML 解析器：lxml 比 Python 自带 html.parser 快约 2-5 倍
_HTML_PARSER = "lxml"

_HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
_IX_REMOVE_TAGS = {"ix:header", "ix:hidden", "ix:references", "ix:resources"}
_HIDDEN_STYLE_TOKENS = ("display:none", "visibility:hidden")
_LOW_INFO_TOKENS = {"-", "--", "—", "n/a", "na", "none", "nil", "☐", "☒"}
_SECTION_TEXT_CACHE_MAX_ENTRIES = 256

_CellValueT = TypeVar("_CellValueT")
_SECTION_TEXT_CACHE_MAX_CHARS = 12_000_000
_TABLE_RENDER_CACHE_MAX_ENTRIES = 2048


@dataclass
class _SectionBlock:
    """内部章节结构。"""

    ref: str
    title: Optional[str]
    level: int
    parent_ref: Optional[str]
    preview: str
    heading_tag: Optional[Tag]
    next_heading_tag: Optional[Tag]
    table_refs: list[str]
    contains_full_text: bool


@dataclass
class _TableBlock:
    """内部表格结构。"""

    ref: str
    tag: Tag
    caption: Optional[str]
    row_count: int
    col_count: int
    headers: Optional[list[str]]
    section_ref: Optional[str]
    context_before: str
    table_type: str
    has_spans: bool


class BSProcessor:
    """BeautifulSoup 文档处理器。"""

    PARSER_VERSION = "bs_processor_v1.1.0"

    @classmethod
    def get_parser_version(cls) -> str:
        """返回处理器 parser version。"""

        return str(cls.PARSER_VERSION)

    # 子类可覆盖此属性以使用不同的 HTML 解析器。
    # 默认使用模块常量 _HTML_PARSER（lxml），个别表单类型
    # （如 6-K）若因 lxml 解析差异导致评分回归，可回退为 "html.parser"。
    _html_parser: str = _HTML_PARSER

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
            ValueError: 文件不存在或不是文件时抛出。
        """

        html_path = source.materialize(suffix=".html")
        if not html_path.exists() or not html_path.is_file():
            raise ValueError(f"HTML 文件不存在: {html_path}")

        self._source = source
        self._form_type = form_type
        self._media_type = media_type or source.media_type
        self._profiler = ProcessorStageProfiler(
            component=self.__class__.__name__,
            enabled=is_processor_profile_enabled(),
        )
        with self._profiler.stage("load_html"):
            html_content = self._load_html_content(html_path)
        with self._profiler.stage("parse_html"):
            self._soup = BeautifulSoup(html_content, self._html_parser)
        with self._profiler.stage("sanitize_html"):
            _sanitize_soup(self._soup)
        with self._profiler.stage("resolve_root"):
            self._root = _get_body_or_root(self._soup)

        self._sections: list[_SectionBlock]
        self._tables: list[_TableBlock]
        self._section_by_ref: dict[str, _SectionBlock]
        self._table_by_ref: dict[str, _TableBlock]
        self._table_ref_by_tag_id: dict[int, str]
        self._section_text_cache: OrderedDict[str, str] = OrderedDict()
        self._section_text_cache_chars = 0
        self._table_render_cache: OrderedDict[str, tuple[str, Any, Optional[list[str]]]] = OrderedDict()

        with self._profiler.stage("build_sections"):
            self._sections = _build_sections(self._root)
        with self._profiler.stage("build_tables"):
            heading_ref_map = _build_heading_ref_map(self._sections)
            default_section_ref = _get_default_section_ref(self._sections)
            self._tables = _build_tables(
                self._root,
                heading_ref_map,
                default_section_ref,
                extra_layout_check=self._extra_layout_table_check,
            )
        with self._profiler.stage("attach_tables_to_sections"):
            self._sections = _attach_tables_to_sections(self._sections, self._tables)

        self._section_by_ref = {section.ref: section for section in self._sections}
        self._table_by_ref = {table.ref: table for table in self._tables}
        self._table_ref_by_tag_id = {id(table.tag): table.ref for table in self._tables}
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
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            是否支持。

        Raises:
            OSError: 读取文件失败时可能抛出。
        """

        resolved_media_type = media_type or source.media_type
        if resolved_media_type and "html" in resolved_media_type.lower():
            return True

        uri_suffix = _infer_suffix_from_uri(source.uri)
        if uri_suffix in {".htm", ".html", ".xhtml"}:
            return True

        return False

    @staticmethod
    def _extra_layout_table_check(row_count: int, col_count: int, text: str) -> bool:
        """业务域扩展的 layout 表格检测钩子。

        engine 层默认不做额外检测。子类（如 FinsBSProcessor）可覆盖此方法
        注入业务域特定规则。

        Args:
            row_count: 表格行数。
            col_count: 表格列数。
            text: 表格规范化后的纯文本。

        Returns:
            是否为 layout 表格。
        """
        return False

    def _extra_table_fields(self, table: _TableBlock) -> dict[str, Any]:
        """返回嵌入到表格输出字典的额外字段。

        engine 层默认不注入额外字段。子类（如 ``FinsBSProcessor``）
        可覆盖此方法添加业务域特有字段（如 ``is_financial``）。

        Args:
            table: 内部表格对象。

        Returns:
            额外字段字典，默认为空字典。
        """
        return {}

    def _load_html_content(self, source_path: Path) -> str:
        """读取 HTML 文件内容。

        Engine 默认只做通用文件读取，不附加业务域预处理。业务域子类如需
        在解析前做额外规范化，应通过覆写此钩子完成。

        Args:
            source_path: HTML 文件路径。

        Returns:
            HTML 内容字符串。

        Raises:
            OSError: 读取失败时抛出。
        """

        return source_path.read_text(encoding="utf-8", errors="ignore")

    def list_sections(self) -> list[SectionSummary]:
        """读取章节列表。

        Args:
            无。

        Returns:
            章节摘要列表。

        Raises:
            RuntimeError: 处理失败时抛出。
        """

        try:
            return [_section_to_summary(section) for section in self._sections]
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError("解析章节失败") from exc

    def list_tables(self) -> list[TableSummary]:
        """读取表格列表。

        Args:
            无。

        Returns:
            表格摘要列表。

        Raises:
            RuntimeError: 处理失败时抛出。
        """

        # 默认过滤 layout 表格，仅返回有实质数据的表格。
        try:
            result = []
            for table in self._tables:
                if table.table_type != "layout":
                    summary = _table_to_summary(table)
                    extra = self._extra_table_fields(table)
                    if isinstance(extra.get("page_no"), int):
                        summary["page_no"] = extra["page_no"]
                    if isinstance(extra.get("internal_ref"), str):
                        summary["internal_ref"] = extra["internal_ref"]
                    if isinstance(extra.get("is_financial"), bool):
                        summary["is_financial"] = extra["is_financial"]
                    result.append(summary)
            return result
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError("解析表格失败") from exc

    def read_section(self, ref: str) -> SectionContent:
        """按 ref 读取章节内容。

        Args:
            ref: 章节引用。

        Returns:
            章节内容。

        Raises:
            KeyError: 章节不存在时抛出。
        """

        section = self._section_by_ref.get(ref)
        if not section:
            raise KeyError(f"Section not found: {ref}")

        # TODO(phase-2): 细化 HTML 清理策略（例如保留部分结构化标记）。
        with self._profiler.stage("read_section"):
            content = self._get_section_text(section)
        word_count = len(content.split())

        return build_section_content(
            ref=section.ref,
            title=section.title,
            content=content,
            tables=list(section.table_refs),
            word_count=word_count,
            contains_full_text=section.contains_full_text,
        )

    def read_table(self, table_ref: str) -> TableContent:
        """按 ref 读取表格内容。

        Args:
            table_ref: 表格引用。

        Returns:
            表格内容。

        Raises:
            KeyError: 表格不存在时抛出。
        """

        table = self._table_by_ref.get(table_ref)
        if not table:
            raise KeyError(f"Table not found: {table_ref}")

        cached_rendered = self._get_cached_rendered_table(table_ref)
        if cached_rendered is None:
            with self._profiler.stage("read_table_render"):
                rendered = _render_table_data(table)
            self._set_cached_rendered_table(table_ref, rendered)
            data_format, data, columns = rendered
        else:
            data_format, data, columns = cached_rendered
        cloned_data = deepcopy(data)
        cloned_columns = deepcopy(columns)

        return build_table_content(
            table_ref=table.ref,
            caption=table.caption,
            data_format=data_format,
            data=cloned_data,
            columns=cloned_columns,
            row_count=table.row_count,
            col_count=table.col_count,
            section_ref=table.section_ref,
            table_type=table.table_type,
            **self._extra_table_fields(table),
        )

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

        # TODO(phase-2): 扩展为智能匹配（复数/时态/同义词）。
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return []

        if within_ref and within_ref not in self._section_by_ref:
            return []

        with self._profiler.stage("search"):
            sections = self._sections if not within_ref else [self._section_by_ref[within_ref]]
            hits_raw, section_content_map = run_titled_section_search(
                sections=sections,
                normalized_query=normalized_query,
                get_text=self._get_section_text,
            )
        return enrich_hits_by_section(
            hits_raw=hits_raw,
            section_content_map=section_content_map,
            query=normalized_query,
        )

    def _get_section_text(self, section: _SectionBlock) -> str:
        """获取章节文本内容并带表格占位符。

        Args:
            section: 章节对象。

        Returns:
            章节文本。

        Raises:
            RuntimeError: 解析失败时抛出。
        """

        cached = self._get_cached_section_text(section.ref)
        if cached is not None:
            return cached
        try:
            rendered = _render_section_text(
                self._root,
                section,
                table_ref_by_tag_id=self._table_ref_by_tag_id,
            )
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError("读取章节内容失败") from exc
        self._set_cached_section_text(section.ref, rendered)
        return rendered

    def _get_cached_section_text(self, ref: str) -> Optional[str]:
        """读取章节文本缓存。

        Args:
            ref: 章节引用。

        Returns:
            命中时返回章节文本，否则返回 `None`。

        Raises:
            RuntimeError: 缓存访问失败时抛出。
        """

        cached = self._section_text_cache.get(ref)
        if cached is None:
            return None
        self._section_text_cache.move_to_end(ref, last=True)
        return cached

    def _set_cached_section_text(self, ref: str, content: str) -> None:
        """写入章节文本缓存并按 LRU 淘汰。

        Args:
            ref: 章节引用。
            content: 章节文本。

        Returns:
            无。

        Raises:
            RuntimeError: 缓存写入失败时抛出。
        """

        existing = self._section_text_cache.pop(ref, None)
        if existing is not None:
            self._section_text_cache_chars -= len(existing)
        self._section_text_cache[ref] = content
        self._section_text_cache_chars += len(content)
        while (
            len(self._section_text_cache) > _SECTION_TEXT_CACHE_MAX_ENTRIES
            or self._section_text_cache_chars > _SECTION_TEXT_CACHE_MAX_CHARS
        ):
            _, removed_content = self._section_text_cache.popitem(last=False)
            self._section_text_cache_chars -= len(removed_content)

    def _get_cached_rendered_table(
        self,
        table_ref: str,
    ) -> Optional[tuple[str, Any, Optional[list[str]]]]:
        """读取表格渲染缓存。

        Args:
            table_ref: 表格引用。

        Returns:
            命中时返回 `(data_format, data, columns)`，否则返回 `None`。

        Raises:
            RuntimeError: 缓存访问失败时抛出。
        """

        cached = self._table_render_cache.get(table_ref)
        if cached is None:
            return None
        self._table_render_cache.move_to_end(table_ref, last=True)
        return cached

    def _set_cached_rendered_table(
        self,
        table_ref: str,
        rendered: tuple[str, Any, Optional[list[str]]],
    ) -> None:
        """写入表格渲染缓存并按 LRU 淘汰。

        Args:
            table_ref: 表格引用。
            rendered: 表格渲染结果 `(data_format, data, columns)`。

        Returns:
            无。

        Raises:
            RuntimeError: 缓存写入失败时抛出。
        """

        if table_ref in self._table_render_cache:
            self._table_render_cache.pop(table_ref, None)
        self._table_render_cache[table_ref] = rendered
        while len(self._table_render_cache) > _TABLE_RENDER_CACHE_MAX_ENTRIES:
            self._table_render_cache.popitem(last=False)

    def get_full_text(self) -> str:
        """获取文档的完整纯文本内容（包含表格内文本）。

        与 ``read_section()`` 不同，本方法不会将表格替换为占位符，
        而是保留表格内所有文本。适用于需要全文分析的场景（如章节切分
        marker 检测），因为 SEC 文档中部分 Item 标题可能位于 table 布局
        内（如 AMZN 的 table-based layout）。

        借鉴 edgartools 的 ``Document.text(include_tables=True)`` 策略：
        全文提取时保留表格内文本，确保 marker 信息不丢失。

        Args:
            无。

        Returns:
            文档完整纯文本字符串。

        Raises:
            RuntimeError: 提取失败时抛出。
        """

        return _normalize_whitespace(
            self._root.get_text(separator=" ", strip=True)
        )

    def get_full_text_with_table_markers(self) -> str:
        """获取文档全文，非 layout 表格用 ``[[t_XXXX]]`` 占位符替代。

        与 ``get_full_text()`` 不同，本方法将每个非 layout ``<table>``
        标签替换为对应的 ``[[t_XXXX]]`` 占位符后再提取文本。Layout
        表格直接移除，不注入标记——与 ``list_tables()`` 过滤策略一致，
        避免在虚拟章节中出现 ``list_tables()`` 不返回的悬挂引用。

        占位符编号与 ``list_tables()`` 返回的 ``table_ref`` 一致
        （按 DOM 出现顺序编号）。

        用途：虚拟章节处理器在全文切分后，通过解析占位符确定每个表格
        落入哪个虚拟章节，从而建立 table→virtual_section 映射。

        实现细节：临时在 DOM 中替换 ``<table>`` 标签，提取文本后立即恢复，
        避免影响其他方法的正常行为。

        Args:
            无。

        Returns:
            带 ``[[t_XXXX]]`` 占位符的文档全文字符串。

        Raises:
            RuntimeError: 提取失败时抛出。
        """

        # 构建 layout 表格 ref 集合（快速查找）
        layout_refs = {t.ref for t in self._tables if t.table_type == "layout"}

        saved: list[tuple[NavigableString, Tag]] = []
        try:
            for idx, table_tag in enumerate(self._root.find_all("table")):
                ref = _format_table_ref(idx + 1)
                if ref in layout_refs:
                    # layout 表格：移除文本，不注入标记
                    marker = NavigableString(" ")
                else:
                    # 数据表格：注入 [[t_XXXX]] 占位符
                    marker = NavigableString(f" [[{ref}]] ")
                table_tag.replace_with(marker)
                saved.append((marker, table_tag))
            return _normalize_whitespace(
                self._root.get_text(separator=" ", strip=True)
            )
        finally:
            # 逆序恢复，保证 DOM 树完整性
            for marker, table_tag in reversed(saved):
                marker.replace_with(table_tag)

def _sanitize_soup(soup: BeautifulSoup) -> None:
    """清理 HTML 解析树，移除隐藏与噪音节点。

    Args:
        soup: BeautifulSoup 解析对象。

    Returns:
        None。

    Raises:
        RuntimeError: 清理失败时抛出。
    """

    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()

    # 性能优化：合并两次 find_all(True) 为一次遍历，
    # 减少对大型 DOM（如 10-30MB 的 SEC 20-F 文件）的重复扫描。
    # 使用 list() 物化，因为循环体中 decompose/unwrap 会修改树。
    for tag in list(soup.find_all(True)):
        # 跳过已被父节点 decompose 而脱离文档树的标签
        if tag.parent is None:
            continue
        if _is_hidden_tag(tag):
            tag.decompose()
            continue
        tag_name = (tag.name or "").lower()
        if not tag_name.startswith("ix:"):
            continue
        if tag_name in _IX_REMOVE_TAGS:
            tag.decompose()
        else:
            tag.unwrap()


def _is_hidden_tag(tag: Tag) -> bool:
    """判断标签是否隐藏。

    Args:
        tag: HTML 标签。

    Returns:
        是否隐藏。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if tag.name in {"html", "body"}:
        return False
    if tag.attrs is None:
        return False
    if tag.has_attr("hidden"):
        return True
    aria_hidden = tag.get("aria-hidden")
    if aria_hidden is not None and str(aria_hidden).strip().lower() == "true":
        return True
    # 仅当 style 属性存在时才做字符串处理，避免对大量无 style 标签的无谓开销
    raw_style = tag.get("style")
    if raw_style:
        style = str(raw_style).replace(" ", "").lower()
        if any(token in style for token in _HIDDEN_STYLE_TOKENS):
            return True
    return False


def _get_body_or_root(soup: BeautifulSoup) -> Tag:
    """获取 body 标签或返回根节点。

    Args:
        soup: BeautifulSoup 解析对象。

    Returns:
        body 标签或根节点。

    Raises:
        RuntimeError: 获取失败时抛出。
    """

    if soup.body:
        return soup.body
    return soup


def _get_default_section_ref(sections: list[_SectionBlock]) -> Optional[str]:
    """计算无标题场景的默认 section ref。

    Args:
        sections: 章节列表。

    Returns:
        默认 section ref 或 None。

    Raises:
        RuntimeError: 计算失败时抛出。
    """

    if len(sections) != 1:
        return None
    section = sections[0]
    if section.contains_full_text:
        return section.ref
    return None


def _build_sections(root: Tag) -> list[_SectionBlock]:
    """构建章节结构。

    Args:
        root: HTML 根节点（body 或 soup）。

    Returns:
        章节列表。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    headings = _extract_heading_tags(root)
    if not headings:
        full_text = _normalize_whitespace(root.get_text(separator=" ", strip=True))
        preview = full_text[:_PREVIEW_MAX_CHARS]
        return [
            _SectionBlock(
                ref=_format_section_ref(1),
                title=None,
                level=1,
                parent_ref=None,
                preview=preview,
                heading_tag=None,
                next_heading_tag=None,
                table_refs=[],
                contains_full_text=True,
            )
        ]

    parent_refs = _compute_parent_refs(headings)
    # 预计算所有 table 标签的 id 集合，加速 _is_within_table 判断
    table_tag_ids = frozenset(id(t) for t in root.find_all("table"))
    sections: list[_SectionBlock] = []

    for index, heading_tag in enumerate(headings):
        next_heading = headings[index + 1] if index + 1 < len(headings) else None
        title = _normalize_whitespace(heading_tag.get_text(separator=" ", strip=True))
        preview = _extract_preview_text(heading_tag, next_heading, _PREVIEW_MAX_CHARS, table_tag_ids)
        ref = _format_section_ref(index + 1)
        level = _heading_level(heading_tag)
        sections.append(
            _SectionBlock(
                ref=ref,
                title=title,
                level=level,
                parent_ref=parent_refs.get(id(heading_tag)),
                preview=preview,
                heading_tag=heading_tag,
                next_heading_tag=next_heading,
                table_refs=[],
                contains_full_text=False,
            )
        )

    return sections


def _build_heading_ref_map(sections: list[_SectionBlock]) -> dict[int, str]:
    """构建标题 Tag 到 section ref 的映射。

    Args:
        sections: 章节列表。

    Returns:
        标题 tag id 到 ref 的映射。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    mapping: dict[int, str] = {}
    for section in sections:
        if section.heading_tag is None:
            continue
        mapping[id(section.heading_tag)] = section.ref
    return mapping


def _attach_tables_to_sections(
    sections: list[_SectionBlock],
    tables: list[_TableBlock],
) -> list[_SectionBlock]:
    """将表格 ref 挂载到章节。

    Args:
        sections: 章节列表。
        tables: 表格列表。

    Returns:
        更新后的章节列表。

    Raises:
        RuntimeError: 处理失败时抛出。
    """

    section_by_ref = {section.ref: section for section in sections}
    fallback_full_text_section_ref: Optional[str] = None
    for section in sections:
        section.table_refs = []
        if section.contains_full_text:
            fallback_full_text_section_ref = section.ref

    for table in tables:
        target_ref = table.section_ref
        if target_ref and target_ref in section_by_ref:
            section_by_ref[target_ref].table_refs.append(table.ref)
            continue
        if fallback_full_text_section_ref is not None:
            section_by_ref[fallback_full_text_section_ref].table_refs.append(table.ref)

    return sections


def _build_tables(
    root: Tag,
    heading_ref_map: dict[int, str],
    default_section_ref: Optional[str],
    *,
    extra_layout_check: Optional[Callable[[int, int, str], bool]] = None,
) -> list[_TableBlock]:
    """构建表格结构。

    Args:
        root: HTML 根节点（body 或 soup）。
        heading_ref_map: 标题 tag id 到章节 ref 的映射。
        default_section_ref: 默认 section ref。
        extra_layout_check: 可选的业务域扩展 layout 检测回调，
            签名 ``(row_count, col_count, normalized_text) -> bool``。

    Returns:
        表格列表。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    tables: list[_TableBlock] = []
    tables_with_refs = _collect_tables_with_section_refs(
        root=root,
        heading_ref_map=heading_ref_map,
        default_section_ref=default_section_ref,
    )
    # 预计算所有 table 标签的 id 集合，加速 _extract_context_before 中 _is_within_table 判断
    table_tag_ids = frozenset(id(t) for t in root.find_all("table"))

    for index, (table_tag, section_ref) in enumerate(tables_with_refs):
        ref = _format_table_ref(index + 1)
        caption = _extract_caption(table_tag)
        # 性能优化：build 阶段仅用 matrix 提取维度和表头，
        # DataFrame 延迟到 _render_table_data（读取时）按需解析，
        # 避免对全量表格执行 pd.read_html(StringIO(str(tag)))。
        matrix = _extract_table_matrix(table_tag)
        row_count, col_count = _count_table_dimensions(None, matrix)
        headers = _extract_headers(None, matrix, table_tag)
        has_spans = _has_complex_spans(table_tag)
        context_before = _extract_context_before(table_tag, table_tag_ids=table_tag_ids)
        # 清除页眉噪声并在 caption 缺失时从前文推断
        context_before = _clean_page_header_noise(context_before)
        if caption is None and context_before:
            caption = _infer_caption_from_context(context_before)
        # 提取表格全文本用于增强分类
        table_text = _safe_table_text(table_tag)
        table_type = _classify_table_type(
            row_count=row_count,
            col_count=col_count,
            headers=headers,
            context_before=context_before,
            table_text=table_text,
            extra_layout_check=extra_layout_check,
        )

        tables.append(
            _TableBlock(
                ref=ref,
                tag=table_tag,
                caption=caption,
                row_count=row_count,
                col_count=col_count,
                headers=headers,
                section_ref=section_ref,
                context_before=context_before,
                table_type=table_type,
                has_spans=has_spans,
            )
        )

    return tables


def _collect_tables_with_section_refs(
    *,
    root: Tag,
    heading_ref_map: dict[int, str],
    default_section_ref: Optional[str],
) -> list[tuple[Tag, Optional[str]]]:
    """单次遍历 DOM 收集表格及所属章节映射。

    该实现按 ``root.descendants`` 线性扫描文档，仅维护一个
    ``current_section_ref`` 游标：

    1. 命中标题节点时更新当前章节；
    2. 命中 ``table`` 节点时记录 ``(table_tag, section_ref)``。

    相比“每张表反向 find_previous 查标题”的方式，该实现将复杂度从
    ``O(table_count * reverse_dom_scan)`` 降为 ``O(dom_size)``。

    Args:
        root: HTML 根节点（body 或 soup）。
        heading_ref_map: 标题 tag id 到章节 ref 的映射。
        default_section_ref: 无标题文档或兜底章节 ref。

    Returns:
        按 DOM 顺序排列的 ``(table_tag, section_ref)`` 列表。

    Raises:
        RuntimeError: 遍历失败时抛出。
    """

    table_entries: list[tuple[Tag, Optional[str]]] = []
    current_section_ref: Optional[str] = default_section_ref
    for node in root.descendants:
        if not isinstance(node, Tag):
            continue
        section_ref = heading_ref_map.get(id(node))
        if section_ref is not None:
            current_section_ref = section_ref
            continue
        if node.name != "table":
            continue
        table_entries.append((node, current_section_ref))
    return table_entries


def _extract_heading_tags(root: Tag) -> list[Tag]:
    """提取有效标题标签。

    Args:
        root: HTML 根节点（body 或 soup）。

    Returns:
        标题标签列表。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    headings: list[Tag] = []
    for tag in root.find_all(_HEADING_TAGS):
        title = _normalize_whitespace(tag.get_text(separator=" ", strip=True))
        if len(title) < 3:
            continue
        headings.append(tag)
    return headings


def _compute_parent_refs(headings: list[Tag]) -> dict[int, Optional[str]]:
    """根据标题层级计算 parent_ref。

    Args:
        headings: 标题标签列表。

    Returns:
        标题 tag id 到 parent_ref 的映射。

    Raises:
        RuntimeError: 计算失败时抛出。
    """

    stack: list[tuple[int, str, Tag]] = []
    parent_refs: dict[int, Optional[str]] = {}

    for index, heading_tag in enumerate(headings):
        level = _heading_level(heading_tag)
        ref = _format_section_ref(index + 1)
        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_refs[id(heading_tag)] = stack[-1][1] if stack else None
        stack.append((level, ref, heading_tag))

    return parent_refs


def _heading_level(tag: Tag) -> int:
    """获取标题层级。

    Args:
        tag: 标题标签。

    Returns:
        标题层级（1-based）。

    Raises:
        ValueError: 标签不是标题时抛出。
    """

    if not tag.name or not tag.name.startswith("h"):
        raise ValueError("非标题标签")
    return int(tag.name[1])


def _extract_preview_text(
    heading_tag: Tag,
    next_heading_tag: Optional[Tag],
    max_chars: int,
    table_tag_ids: Optional[frozenset[int]] = None,
) -> str:
    """提取章节预览文本。

    Args:
        heading_tag: 当前标题标签。
        next_heading_tag: 下一个标题标签。
        max_chars: 最大字符数。
        table_tag_ids: 可选的预计算 table 标签 id 集合，加速 _is_within_table 判断。

    Returns:
        预览文本。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    text_parts: list[str] = []
    for node in _iter_elements_between(heading_tag, next_heading_tag):
        if isinstance(node, Tag) and node.name in _HEADING_TAGS:
            break
        if _is_within_table(node, table_tag_ids):
            continue
        if isinstance(node, Tag):
            text = _normalize_whitespace(node.get_text(separator=" ", strip=True))
            if text:
                text_parts.append(text)
        if isinstance(node, NavigableString):
            text = _normalize_whitespace(str(node))
            if text:
                text_parts.append(text)
        if sum(len(part) for part in text_parts) >= max_chars:
            break

    preview = " ".join(text_parts)
    return preview[:max_chars]


def _iter_nodes_between(start: Tag, end: Optional[Tag]) -> Iterable[Any]:
    """遍历两个节点之间的兄弟节点。

    Args:
        start: 起始节点。
        end: 终止节点（不包含）。

    Returns:
        节点迭代器。

    Raises:
        RuntimeError: 遍历失败时抛出。
    """

    current = start.find_next_sibling()
    while current and current != end:
        yield current
        current = current.find_next_sibling()


def _iter_elements_between(start: Tag, end: Optional[Tag]) -> Iterable[Any]:
    """遍历两个节点之间的文档顺序元素。

    Args:
        start: 起始节点。
        end: 终止节点（不包含）。

    Returns:
        节点迭代器。

    Raises:
        RuntimeError: 遍历失败时抛出。
    """

    for node in start.next_elements:
        if node == end:
            break
        yield node


def _extract_caption(table_tag: Tag) -> Optional[str]:
    """提取表格标题。

    Args:
        table_tag: 表格标签。

    Returns:
        caption 文本。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    if table_tag.caption:
        caption_text = table_tag.caption.get_text(separator=" ", strip=True)
        return _normalize_whitespace(caption_text) or None
    return None


def _extract_table_matrix(table_tag: Tag) -> list[list[str]]:
    """从 HTML 提取表格矩阵。

    Args:
        table_tag: 表格标签。

    Returns:
        行列矩阵。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    rows: list[list[str]] = []
    for row in table_tag.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if not cells:
            continue
        rows.append([_normalize_whitespace(cell.get_text(separator=" ", strip=True)) for cell in cells])
    return rows


def _count_table_dimensions(
    df: Optional[pd.DataFrame],
    matrix: list[list[str]],
) -> tuple[int, int]:
    """统计表格行列数。

    Args:
        df: DataFrame。
        matrix: HTML 矩阵。

    Returns:
        行数、列数。

    Raises:
        RuntimeError: 统计失败时抛出。
    """

    if df is not None:
        return len(df.index), len(df.columns)
    if not matrix:
        return 0, 0
    return len(matrix), max(len(row) for row in matrix)


def _extract_headers(
    df: Optional[pd.DataFrame],
    matrix: list[list[str]],
    table_tag: Tag,
) -> Optional[list[str]]:
    """提取行头（复用 `headers` 字段）。

    Args:
        df: DataFrame。
        matrix: HTML 矩阵。
        table_tag: 表格标签。

    Returns:
        表头列表或 None。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    row_headers_from_df = _extract_row_headers_from_dataframe(df)
    if row_headers_from_df:
        return row_headers_from_df

    row_headers_from_matrix = _extract_row_headers_from_matrix(matrix)
    if row_headers_from_matrix:
        return row_headers_from_matrix

    th_headers = _extract_headers_from_th(table_tag)
    if th_headers:
        return th_headers[:10]

    if df is not None:
        headers = [str(item) for item in df.columns.tolist()]
        if _looks_like_default_headers(headers):
            return _select_matrix_headers(matrix)
        return headers[:10] if headers else None

    return _select_matrix_headers(matrix)


def _extract_row_headers_from_dataframe(df: Optional[pd.DataFrame]) -> Optional[list[str]]:
    """从 DataFrame 中提取行头。

    Args:
        df: DataFrame。

    Returns:
        行头列表或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    if df is None or df.empty:
        return None
    candidates: list[str] = []
    for row in df.itertuples(index=False, name=None):
        header = _pick_row_header_from_values(list(row))
        if header:
            candidates.append(header)
    normalized = _normalize_header_candidates(candidates)
    if not normalized or _looks_like_default_headers(normalized):
        return None
    return normalized[:10]


def _extract_row_headers_from_matrix(matrix: list[list[str]]) -> Optional[list[str]]:
    """从矩阵第一列提取行头。

    Args:
        matrix: 表格矩阵。

    Returns:
        行头列表或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    if not matrix:
        return None
    candidates: list[str] = []
    for row in matrix:
        if not row:
            continue
        header = _pick_row_header_from_values(row[:3])
        if header:
            candidates.append(header)
    normalized = _normalize_header_candidates(candidates)
    if not normalized or _looks_like_default_headers(normalized):
        return None
    return normalized[:10]


def _pick_row_header_from_values(values: list[Any]) -> str:
    """从单行值中挑选首个高信息量候选。

    Args:
        values: 行内候选值。

    Returns:
        行头文本，未命中返回空字符串。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    for value in values:
        text = _normalize_whitespace(str(value))
        if not text:
            continue
        if _is_low_information_header(text):
            continue
        return text
    return ""


def _normalize_header_candidates(candidates: list[str]) -> list[str]:
    """标准化候选头部文本并去重。

    Args:
        candidates: 候选列表。

    Returns:
        标准化列表。

    Raises:
        RuntimeError: 处理失败时抛出。
    """

    normalized: list[str] = []
    for item in candidates:
        value = _normalize_whitespace(item)
        if not value:
            continue
        normalized.append(value)
    return _deduplicate_headers(normalized)


def _deduplicate_headers(headers: list[str]) -> list[str]:
    """按出现顺序去重并保留首次出现。

    Args:
        headers: 原始头部列表。

    Returns:
        去重列表（重复项直接移除）。

    Raises:
        RuntimeError: 去重失败时抛出。
    """

    seen: set[str] = set()
    deduped: list[str] = []
    for item in headers:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _has_complex_spans(table_tag: Tag) -> bool:
    """判断表格是否存在跨行跨列。

    Args:
        table_tag: 表格标签。

    Returns:
        是否包含跨行跨列。

    Raises:
        RuntimeError: 检测失败时抛出。
    """

    # 性能优化：使用 descendants 惰性迭代替代 find_all 全量收集，
    # 遇到首个跨行/跨列即返回，避免构建完整 cell 列表。
    for node in table_tag.descendants:
        if not isinstance(node, Tag):
            continue
        if node.name not in ("th", "td"):
            continue
        if node.has_attr("rowspan") or node.has_attr("colspan"):
            try:
                if _coerce_span_value(node.get("rowspan")) > 1 or _coerce_span_value(node.get("colspan")) > 1:
                    return True
            except ValueError:
                return True
    return False


def _coerce_span_value(value: Any) -> int:
    """将 HTML span 属性值转换为整数。

    Args:
        value: 原始 span 属性值。

    Returns:
        转换后的整数；空值默认返回 1。

    Raises:
        ValueError: 值非法时抛出。
    """

    if value is None:
        return 1
    normalized_value = str(value).strip()
    if not normalized_value:
        return 1
    return int(normalized_value)


def _extract_headers_from_th(table_tag: Tag) -> Optional[list[str]]:
    """从 th 标签提取表头。

    Args:
        table_tag: 表格标签。

    Returns:
        表头列表或 None。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    for row in table_tag.find_all("tr"):
        header_cells = row.find_all("th")
        if not header_cells:
            continue
        headers = [_normalize_whitespace(cell.get_text(separator=" ", strip=True)) for cell in header_cells]
        headers = [item for item in headers if item]
        if headers:
            return headers
    return None


def _looks_like_default_headers(headers: list[str]) -> bool:
    """判断是否为默认索引表头（如 0,1,2 或 Unnamed）。

    Args:
        headers: 表头列表。

    Returns:
        是否为默认表头。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    cleaned = [header.strip() for header in headers if str(header).strip()]
    if not cleaned:
        return True
    if all(_is_low_information_header(item) for item in cleaned):
        return True
    if all(str(item).lower().startswith("unnamed") for item in cleaned):
        return True
    # 必须用 isdecimal() 而非 isdigit()：isdigit() 会把 `①`、`¹` 等
    # Unicode 数字符号也判为 True，但 int() 无法解析这些字符。
    if all(str(item).isdecimal() for item in cleaned):
        numbers = [int(item) for item in cleaned]
        start = numbers[0]
        return numbers == list(range(start, start + len(numbers)))
    return False


def _is_low_information_header(value: str) -> bool:
    """判断是否低信息头部文本。

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
    if re.fullmatch(r"[\d\s,\.\-\+\(\)%$¥€]+", normalized):
        return True
    return False


def _select_matrix_headers(matrix: list[list[str]]) -> Optional[list[str]]:
    """从矩阵中选择合理的表头行。

    Args:
        matrix: HTML 矩阵。

    Returns:
        表头列表或 None。

    Raises:
        RuntimeError: 选择失败时抛出。
    """

    for row in matrix:
        normalized = [_normalize_whitespace(cell) for cell in row]
        normalized = [cell for cell in normalized if cell]
        if not normalized:
            continue
        if _looks_like_default_headers(normalized):
            continue
        return row[:10]
    return None


def _is_within_table(node: Any, table_tag_ids: Optional[frozenset[int]] = None) -> bool:
    """判断节点是否位于 table 内部。

    当提供 ``table_tag_ids``（所有 ``<table>`` 标签的 ``id()`` 集合）时，
    通过向上遍历父节点并做 hash 查找，避免 BS4 ``find_parent("table")``
    的字符串比较开销。

    Args:
        node: BeautifulSoup 节点。
        table_tag_ids: 可选的预计算 table 标签 id 集合。

    Returns:
        是否位于 table 内部。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if table_tag_ids is not None:
        # 快速路径：向上遍历父节点检查 id 是否在 table 集合中
        tag = node if isinstance(node, Tag) else getattr(node, 'parent', None)
        while tag is not None:
            if id(tag) in table_tag_ids:
                return True
            tag = tag.parent
        return False
    # 回退路径：无预计算集合时使用 BS4 原生 find_parent
    if isinstance(node, Tag):
        return node.find_parent("table") is not None or node.name == "table"
    if isinstance(node, NavigableString):
        parent = node.parent
        if isinstance(parent, Tag):
            return parent.find_parent("table") is not None or parent.name == "table"
    return False


def _extract_context_before(
    table_tag: Tag,
    max_chars: int = 200,
    table_tag_ids: Optional[frozenset[int]] = None,
) -> str:
    """提取表格前的上下文文本。

    Args:
        table_tag: 表格标签。
        max_chars: 最大字符数。
        table_tag_ids: 可选的预计算 table 标签 id 集合，加速 _is_within_table 判断。

    Returns:
        上下文文本。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    text_parts: list[str] = []
    total_len = 0

    for node in table_tag.previous_elements:
        if node == table_tag:
            continue
        if isinstance(node, Tag):
            if node.name in _HEADING_TAGS:
                break
            if node.name == "table":
                break
            if _is_hidden_tag(node):
                continue
        if _is_within_table(node, table_tag_ids):
            continue
        if isinstance(node, NavigableString):
            text = _normalize_whitespace(str(node))
            if not text:
                continue
            text_parts.append(text)
            total_len += len(text)
            if total_len >= max_chars:
                break

    if not text_parts:
        # 回退到兄弟节点扫描，提升复杂 DOM 下的命中率。
        sibling = table_tag.find_previous_sibling()
        while sibling is not None:
            if isinstance(sibling, Tag):
                if sibling.name in _HEADING_TAGS:
                    break
                if sibling.name == "table":
                    break
                if not _is_hidden_tag(sibling):
                    text = _normalize_whitespace(sibling.get_text(separator=" ", strip=True))
                    if text:
                        text_parts.append(text)
                        if sum(len(part) for part in text_parts) >= max_chars:
                            break
            sibling = sibling.find_previous_sibling()
    if not text_parts:
        return ""
    text_parts.reverse()
    full_text = " ".join(text_parts)
    if len(full_text) > max_chars:
        return full_text[-max_chars:]
    return full_text


def _classify_table_type(
    *,
    row_count: int,
    col_count: int,
    headers: Optional[list[str]],
    context_before: str,
    table_text: str = "",
    extra_layout_check: Optional[Callable[[int, int, str], bool]] = None,
) -> str:
    """对表格做轻量类型分类。

    分类规则优先级：
    1. 极小表格（≤2 行 ≤3 列 且文本 < 16 字符） → ``layout``
    2. 默认列头（数字索引 / Unnamed） → ``layout``
    3. 无列头且前文过短 → ``layout``
    4. 业务域扩展规则（通过 ``extra_layout_check`` 回调注入） → ``layout``
    5. 其他 → ``data``

    Args:
        row_count: 行数。
        col_count: 列数。
        headers: 行头列表。
        context_before: 前文。
        table_text: 表格全文本（用于增强分类）。
        extra_layout_check: 可选的业务域扩展回调。
            签名 ``(row_count, col_count, normalized_text) -> bool``，
            返回 ``True`` 表示该表格应标记为 ``layout``。
            用于允许上层（如 fins 层）注入业务域特定的 layout 判定规则，
            而不在 engine 通用层中硬编码业务知识。

    Returns:
        ``data`` 或 ``layout``。

    Raises:
        RuntimeError: 分类失败时抛出。
    """

    normalized_text = _normalize_whitespace(table_text)
    # 规则 1: 极小表格（行少、列少、文本极短）
    if row_count <= 2 and col_count <= 3 and (not normalized_text or len(normalized_text) < 16):
        return "layout"
    # 规则 2: 默认索引表头（如 0,1,2 或 Unnamed）
    if headers and _looks_like_default_headers(headers):
        return "layout"
    # 规则 3: 无列头且前文过短
    if not headers and len(context_before.strip()) < 12:
        return "layout"
    # 规则 4: 业务域扩展规则（由调用方注入）
    if extra_layout_check and extra_layout_check(row_count, col_count, normalized_text or ""):
        return "layout"
    return "data"

def _safe_table_text(table_tag: Tag) -> str:
    """安全提取表格纯文本（用于分类检测）。

    Args:
        table_tag: 表格 HTML 标签。

    Returns:
        表格文本，拐取失败时返回空字符串。

    Raises:
        RuntimeError: 处理失败时抛出。
    """
    try:
        return table_tag.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def _render_section_text(
    root: Tag,
    section: _SectionBlock,
    *,
    table_ref_by_tag_id: Optional[dict[int, str]] = None,
) -> str:
    """渲染章节文本，表格以占位符替换。

    Args:
        root: HTML 根节点（body 或 soup）。
        section: 章节对象。
        table_ref_by_tag_id: 可选的 `table_tag_id -> table_ref` 映射。

    Returns:
        章节文本。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    table_refs = list(section.table_refs)
    table_counter = 0
    text_parts: list[str] = []

    def _append_node_text(node: Any) -> None:
        nonlocal table_counter
        if isinstance(node, NavigableString):
            normalized = _normalize_whitespace(str(node))
            if normalized:
                text_parts.append(normalized)
            return
        if not isinstance(node, Tag):
            return
        if node.name == "table":
            mapped_ref = None
            if table_ref_by_tag_id is not None:
                mapped_ref = table_ref_by_tag_id.get(id(node))
            if mapped_ref is None:
                mapped_ref = (
                    table_refs[table_counter]
                    if table_counter < len(table_refs)
                    else _format_table_ref(table_counter + 1)
                )
            text_parts.append(f"[[{mapped_ref}]]")
            table_counter += 1
            return
        for child in node.children:
            _append_node_text(child)

    if section.heading_tag is None:
        for node in root.children:
            _append_node_text(node)
    else:
        for node in _iter_nodes_between(section.heading_tag, section.next_heading_tag):
            _append_node_text(node)
    return _normalize_whitespace(" ".join(text_parts))


def _section_to_summary(section: _SectionBlock) -> SectionSummary:
    """章节对象转摘要。

    Args:
        section: 章节对象。

    Returns:
        章节摘要。

    Raises:
        RuntimeError: 转换失败时抛出。
    """

    return build_section_summary(
        ref=section.ref,
        title=section.title,
        level=section.level,
        parent_ref=section.parent_ref,
        preview=section.preview,
    )


def _table_to_summary(table: _TableBlock) -> TableSummary:
    """表格对象转摘要。

    Args:
        table: 表格对象。

    Returns:
        表格摘要。

    Raises:
        RuntimeError: 转换失败时抛出。
    """

    return build_table_summary(
        table_ref=table.ref,
        caption=table.caption,
        context_before=table.context_before,
        row_count=table.row_count,
        col_count=table.col_count,
        table_type=table.table_type,
        headers=table.headers,
        section_ref=table.section_ref,
    )


def _render_table_data(
    table: _TableBlock,
) -> tuple[str, Any, Optional[list[str]]]:
    """渲染表格数据。

    性能优化：先用表格属性预判是否走 markdown 路径，
    若确定走 markdown 则跳过昂贵的 pd.read_html 调用。
    仅在 records 路径时才解析 DataFrame。

    Args:
        table: 表格对象。

    Returns:
        data_format、data、columns。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    # 快速路径：已知走 markdown 的场景直接用 matrix，跳过 pd.read_html
    if _can_skip_dataframe(table):
        matrix = _extract_table_matrix(table.tag)
        markdown = _build_markdown_table(matrix)
        return "markdown", markdown, None

    # 常规路径：需要 DataFrame 判断是否有重复列等
    df = parse_html_table_dataframe(table.tag)
    matrix = _extract_table_matrix(table.tag)

    use_markdown = _should_use_markdown(df, table, matrix)
    if use_markdown:
        markdown = _build_markdown_table(matrix)
        return "markdown", markdown, None

    records, columns = _build_records(df, matrix)
    return "records", records, columns


def _can_skip_dataframe(table: _TableBlock) -> bool:
    """判断渲染时是否可跳过 DataFrame 解析。

    当表格属性已可确定走 markdown 路径时返回 True，
    避免执行代价高昂的 ``pd.read_html(StringIO(str(tag)))``。

    Args:
        table: 表格对象。

    Returns:
        是否可跳过 DataFrame 解析。
    """

    # has_spans → _should_use_markdown 必定返回 True
    if table.has_spans:
        return True
    # 超大表格 → _should_use_markdown 必定返回 True
    if table.col_count > 25 or table.row_count > 500:
        return True
    # 无行列 → matrix 为空 → _should_use_markdown 必定返回 True
    if table.row_count == 0:
        return True
    return False


def _should_use_markdown(
    df: Optional[pd.DataFrame],
    table: _TableBlock,
    matrix: list[list[str]],
) -> bool:
    """判断是否使用 markdown 输出。

    Args:
        df: DataFrame。
        table: 表格对象。
        matrix: HTML 矩阵。

    Returns:
        是否使用 markdown。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if table.has_spans:
        return True
    if df is not None and df.columns.has_duplicates:
        return True
    if table.col_count > 25 or table.row_count > 500:
        return True
    if not matrix:
        return True
    return False


def _build_records(
    df: Optional[pd.DataFrame],
    matrix: list[list[str]],
) -> tuple[list[dict[str, Any]], Optional[list[str]]]:
    """构建 records 输出。

    Args:
        df: DataFrame。
        matrix: HTML 矩阵。

    Returns:
        records 列表与 columns。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    if df is None:
        columns, records = _records_from_matrix(matrix)
        return records, columns

    records = df.to_dict(orient="records")
    normalized_records: list[dict[str, Any]] = []
    for row in records:
        normalized_row = {str(key): _normalize_cell_value(value) for key, value in row.items()}
        normalized_records.append(normalized_row)
    columns = [str(col) for col in df.columns.tolist()]
    return normalized_records, columns


def _records_from_matrix(matrix: list[list[str]]) -> tuple[list[str], list[dict[str, Any]]]:
    """从矩阵构建 records。

    Args:
        matrix: HTML 矩阵。

    Returns:
        columns 与 records。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    if not matrix:
        return [], []
    headers = matrix[0]
    body = matrix[1:]
    columns = headers
    records: list[dict[str, Any]] = []
    for row in body:
        row_values = row + [""] * max(0, len(columns) - len(row))
        record = {columns[i]: row_values[i] for i in range(len(columns))}
        records.append(record)
    return columns, records


def _build_markdown_table(matrix: list[list[str]]) -> str:
    """构建 markdown 表格。

    Args:
        matrix: HTML 矩阵。

    Returns:
        markdown 字符串。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    if not matrix:
        return ""

    col_count = max(len(row) for row in matrix)
    header = matrix[0] + [""] * max(0, col_count - len(matrix[0]))
    body = matrix[1:]

    lines = [
        _render_markdown_row(header, col_count),
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in body:
        lines.append(_render_markdown_row(row, col_count))
    return "\n".join(lines)


def _render_markdown_row(row: list[str], col_count: int) -> str:
    """渲染 markdown 表格行。

    Args:
        row: 行数据。
        col_count: 列数。

    Returns:
        markdown 行字符串。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    row_values = row + [""] * max(0, col_count - len(row))
    return "| " + " | ".join(row_values) + " |"


@overload
def _normalize_cell_value(value: None) -> None:
    ...


@overload
def _normalize_cell_value(value: str) -> str:
    ...


@overload
def _normalize_cell_value(value: float) -> float | None:
    ...


@overload
def _normalize_cell_value(value: _CellValueT) -> _CellValueT:
    ...


def _normalize_cell_value(value: _CellValueT | str | float | None) -> _CellValueT | str | float | None:
    """规范化表格单元格值。

    Args:
        value: 原始值。

    Returns:
        规范化后的值。

    Raises:
        RuntimeError: 处理失败时抛出。
    """

    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        return _normalize_whitespace(value)
    return value
