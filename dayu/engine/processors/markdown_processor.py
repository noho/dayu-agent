"""Markdown 文档处理器实现。

该模块实现 `DocumentProcessor` 协议，用于解析 `*.md/*.markdown` 文档，
提供章节、表格、章节内容、表格内容与搜索能力。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
from .perf_utils import ProcessorStageProfiler, is_processor_profile_enabled
from .search_utils import enrich_hits_by_section, run_titled_section_search
from .source import Source
from .text_utils import (
    PREVIEW_MAX_CHARS as _PREVIEW_MAX_CHARS,
    format_section_ref as _format_section_ref,
    format_table_ref as _format_table_ref,
    normalize_whitespace as _normalize_whitespace,
)

_SECTION_CONTENT_CACHE_MAX_ENTRIES = 256


@dataclass
class _SectionBlock:
    """章节内部结构。"""

    ref: str
    title: Optional[str]
    level: int
    parent_ref: Optional[str]
    preview: str
    start_line: int
    end_line: int
    contains_full_text: bool
    table_refs: list[str]


@dataclass
class _TableBlock:
    """表格内部结构。"""

    table_ref: str
    start_line: int
    end_line: int
    caption: Optional[str]
    context_before: str
    row_count: int
    col_count: int
    headers: Optional[list[str]]
    section_ref: Optional[str]
    table_type: str
    markdown: str


class MarkdownProcessor:
    """Markdown 文档处理器。"""

    PARSER_VERSION = "markdown_processor_v1.1.0"

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
            ValueError: 文件不存在时抛出。
            OSError: 文件读取失败时抛出。
        """

        self._source = source
        self._form_type = form_type
        self._media_type = media_type or source.media_type
        self._profiler = ProcessorStageProfiler(
            component=self.__class__.__name__,
            enabled=is_processor_profile_enabled(),
        )

        source_path = source.materialize(suffix=".md")
        if not source_path.exists() or not source_path.is_file():
            raise ValueError(f"Markdown 文件不存在: {source_path}")
        self._source_path = source_path
        with self._profiler.stage("read_lines"):
            self._lines = _read_lines(source_path)

        with self._profiler.stage("build_tables"):
            self._tables = _build_tables(self._lines)
        with self._profiler.stage("build_sections"):
            self._sections = _build_sections(self._lines, self._tables)
        with self._profiler.stage("attach_tables_to_sections"):
            self._tables = _attach_tables_to_sections(self._tables, self._sections)

        self._section_by_ref = {section.ref: section for section in self._sections}
        self._table_by_ref = {table.table_ref: table for table in self._tables}
        self._section_content_cache: OrderedDict[str, str] = OrderedDict()
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
            OSError: 文件访问失败时可能抛出。
        """

        del form_type
        resolved_media_type = str(media_type or source.media_type or "").lower()
        if "markdown" in resolved_media_type or resolved_media_type == "text/md":
            return True
        suffix = Path(str(source.uri or "").split("?", 1)[0]).suffix.lower()
        return suffix in {".md", ".markdown"}

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
                )
                for section in self._sections
            ]
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError("Markdown section parsing failed") from exc

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
                )
                extra = self._extra_table_fields(table)
                if extra.get("is_financial") is not None:
                    summary["is_financial"] = bool(extra["is_financial"])
                result.append(summary)
            return result
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError("Markdown table parsing failed") from exc

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
            return build_section_content(
                ref=section.ref,
                title=section.title,
                content=content,
                tables=list(section.table_refs),
                word_count=len(content.split()),
                contains_full_text=section.contains_full_text,
            )
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError(f"章节读取失败: {ref}") from exc

    def read_table(self, table_ref: str) -> TableContent:
        """按 ref 读取表格内容。

        Args:
            table_ref: 表格引用。

        Returns:
            表格内容字典。

        Raises:
            KeyError: 表格不存在时抛出。
            RuntimeError: 读取失败时抛出。
        """

        table = self._table_by_ref.get(table_ref)
        if table is None:
            raise KeyError(f"Table not found: {table_ref}")

        try:
            md_extra = self._extra_table_fields(table)
            headers, rows = _parse_markdown_table(table.markdown)
            if headers and len(set(headers)) == len(headers):
                records = _rows_to_records(headers, rows)
                return build_table_content(
                    table_ref=table.table_ref,
                    caption=table.caption,
                    data_format="records",
                    data=records,
                    columns=headers,
                    row_count=table.row_count,
                    col_count=table.col_count,
                    section_ref=table.section_ref,
                    table_type=table.table_type,
                    **md_extra,
                )
            return build_table_content(
                table_ref=table.table_ref,
                caption=table.caption,
                data_format="markdown",
                data=table.markdown,
                columns=None,
                row_count=table.row_count,
                col_count=table.col_count,
                section_ref=table.section_ref,
                table_type=table.table_type,
                **md_extra,
            )
        except Exception as exc:  # pragma: no cover - 兜底保护
            raise RuntimeError(f"表格读取失败: {table_ref}") from exc

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

        sections = [self._section_by_ref[within_ref]] if within_ref else self._sections
        hits_raw: list[SearchHit] = []
        section_content_map: dict[str, str] = {}
        with self._profiler.stage("search"):
            hits_raw, section_content_map = run_titled_section_search(
                sections=sections,
                normalized_query=normalized_query,
                get_text=self._get_or_render_section_content,
            )
        return enrich_hits_by_section(
            hits_raw=hits_raw,
            section_content_map=section_content_map,
            query=normalized_query,
        )

    def get_full_text(self) -> str:
        """获取文档的完整纯文本内容。

        从原始 Markdown 行数据拼接完整全文。

        Args:
            无。

        Returns:
            文档完整纯文本字符串。

        Raises:
            RuntimeError: 提取失败时抛出。
        """

        return "\n".join(self._lines)

    def _get_or_render_section_content(self, section: _SectionBlock) -> str:
        """读取或渲染章节正文缓存。

        Args:
            section: 章节对象。

        Returns:
            章节正文文本。

        Raises:
            RuntimeError: 渲染失败时抛出。
        """

        cached = self._section_content_cache.get(section.ref)
        if cached is not None:
            self._section_content_cache.move_to_end(section.ref, last=True)
            return cached
        content = _render_section_content(self._lines, section, self._tables)
        self._section_content_cache[section.ref] = content
        self._section_content_cache.move_to_end(section.ref, last=True)
        while len(self._section_content_cache) > _SECTION_CONTENT_CACHE_MAX_ENTRIES:
            self._section_content_cache.popitem(last=False)
        return content

    def get_full_text_with_table_markers(self) -> str:
        """获取带表格占位符的全文（MarkdownProcessor 不支持）。

        MarkdownProcessor 基于纯文本行解析，不具备 DOM 级表格标记
        注入能力，返回空字符串表示不支持。

        Args:
            无。

        Returns:
            空字符串。
        """

        return ""


def _read_lines(path: Path) -> list[str]:
    """读取 Markdown 行列表。

    Args:
        path: 文件路径。

    Returns:
        行列表。

    Raises:
        OSError: 读取失败时抛出。
    """

    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _build_sections(lines: list[str], tables: list[_TableBlock]) -> list[_SectionBlock]:
    """构建章节索引。

    Args:
        lines: Markdown 行列表。
        tables: 表格列表。

    Returns:
        章节列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    del tables
    heading_indices = _collect_heading_indices(lines)
    if not heading_indices:
        content = _normalize_whitespace("\n".join(lines))
        return [
            _SectionBlock(
                ref=_format_section_ref(1),
                title=None,
                level=1,
                parent_ref=None,
                preview=content[:_PREVIEW_MAX_CHARS],
                start_line=0,
                end_line=max(0, len(lines) - 1),
                contains_full_text=True,
                table_refs=[],
            )
        ]

    sections: list[_SectionBlock] = []
    stack: list[tuple[int, str]] = []

    for idx, (line_no, level, title) in enumerate(heading_indices, start=1):
        next_line_no = heading_indices[idx][0] if idx < len(heading_indices) else len(lines)
        start_line = line_no
        end_line = max(line_no, next_line_no - 1)

        while stack and stack[-1][0] >= level:
            stack.pop()
        parent_ref = stack[-1][1] if stack else None
        section_ref = _format_section_ref(idx)
        stack.append((level, section_ref))

        preview_text = _normalize_whitespace("\n".join(lines[start_line:end_line + 1]))[:_PREVIEW_MAX_CHARS]
        sections.append(
            _SectionBlock(
                ref=section_ref,
                title=title,
                level=level,
                parent_ref=parent_ref,
                preview=preview_text,
                start_line=start_line,
                end_line=end_line,
                contains_full_text=False,
                table_refs=[],
            )
        )

    return sections


def _collect_heading_indices(lines: list[str]) -> list[tuple[int, int, str]]:
    """提取标题行。

    Args:
        lines: Markdown 行列表。

    Returns:
        `(line_no, level, title)` 列表。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    results: list[tuple[int, int, str]] = []
    for line_no, line in enumerate(lines):
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        level = len(match.group(1))
        title = _normalize_whitespace(match.group(2))
        if not title:
            continue
        results.append((line_no, level, title))
    return results


def _build_tables(lines: list[str]) -> list[_TableBlock]:
    """构建表格索引。

    Args:
        lines: Markdown 行列表。

    Returns:
        表格列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    tables: list[_TableBlock] = []
    idx = 0
    table_index = 0

    while idx < len(lines):
        if idx + 1 >= len(lines):
            break
        header_line = lines[idx]
        separator_line = lines[idx + 1]
        if not _looks_like_markdown_table_header(header_line, separator_line):
            idx += 1
            continue

        end_line = idx + 1
        while end_line + 1 < len(lines) and _looks_like_table_row(lines[end_line + 1]):
            end_line += 1

        table_index += 1
        markdown = "\n".join(lines[idx:end_line + 1])
        headers, rows = _parse_markdown_table(markdown)
        row_count = len(rows)
        col_count = len(headers)
        caption = _extract_table_caption(lines, idx)
        context_before = _extract_context_before(lines, idx)
        table_type = _classify_table_type(
            row_count=row_count,
            col_count=col_count,
            headers=headers,
            context_before=context_before,
        )
        tables.append(
            _TableBlock(
                table_ref=_format_table_ref(table_index),
                start_line=idx,
                end_line=end_line,
                caption=caption,
                context_before=context_before,
                row_count=row_count,
                col_count=col_count,
                headers=headers if headers else None,
                section_ref=None,
                table_type=table_type,
                markdown=markdown,
            )
        )
        idx = end_line + 1

    return tables


def _attach_tables_to_sections(tables: list[_TableBlock], sections: list[_SectionBlock]) -> list[_TableBlock]:
    """将表格挂载到章节。

    Args:
        tables: 表格列表。
        sections: 章节列表。

    Returns:
        更新后的表格列表。

    Raises:
        RuntimeError: 挂载失败时抛出。
    """

    if not sections:
        return tables

    sorted_sections = sorted(sections, key=lambda section: section.start_line)
    sorted_tables = sorted(tables, key=lambda table: table.start_line)
    for section in sorted_sections:
        section.table_refs = []

    section_index = 0
    for table in sorted_tables:
        while (
            section_index < len(sorted_sections)
            and table.start_line > sorted_sections[section_index].end_line
        ):
            section_index += 1
        if section_index >= len(sorted_sections):
            break
        section = sorted_sections[section_index]
        if table.start_line < section.start_line:
            continue
        table.section_ref = section.ref
        section.table_refs.append(table.table_ref)
    return tables


def _render_section_content(lines: list[str], section: _SectionBlock, tables: list[_TableBlock]) -> str:
    """渲染章节正文并替换表格占位符。

    Args:
        lines: Markdown 行列表。
        section: 章节对象。
        tables: 表格列表。

    Returns:
        渲染后的章节正文。

    Raises:
        RuntimeError: 渲染失败时抛出。
    """

    table_by_start_line: dict[int, _TableBlock] = {
        table.start_line: table
        for table in tables
        if table.start_line >= section.start_line and table.start_line <= section.end_line
    }

    parts: list[str] = []
    current_line = section.start_line
    while current_line <= section.end_line and current_line < len(lines):
        table = table_by_start_line.get(current_line)
        if table is not None:
            parts.append(f"[[{table.table_ref}]]")
            current_line = table.end_line + 1
            continue
        parts.append(lines[current_line])
        current_line += 1

    return _normalize_whitespace("\n".join(parts))


def _looks_like_markdown_table_header(header_line: str, separator_line: str) -> bool:
    """判断是否为 markdown 表头 + 分隔行。

    Args:
        header_line: 表头行。
        separator_line: 分隔行。

    Returns:
        是否命中。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if "|" not in header_line or "|" not in separator_line:
        return False
    cells = _split_table_cells(separator_line)
    if len(cells) < 2:
        return False
    for cell in cells:
        token = cell.strip()
        if not token:
            return False
        if re.fullmatch(r":?-{3,}:?", token) is None:
            return False
    return True


def _looks_like_table_row(line: str) -> bool:
    """判断一行是否像表格行。

    Args:
        line: 单行文本。

    Returns:
        是否像表格行。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if "|" not in line:
        return False
    cells = _split_table_cells(line)
    return len(cells) >= 2


def _split_table_cells(line: str) -> list[str]:
    """拆分 markdown 表格单元格。

    Args:
        line: 表格行。

    Returns:
        单元格列表。

    Raises:
        RuntimeError: 拆分失败时抛出。
    """

    trimmed = line.strip()
    if trimmed.startswith("|"):
        trimmed = trimmed[1:]
    if trimmed.endswith("|"):
        trimmed = trimmed[:-1]
    return [cell.strip() for cell in trimmed.split("|")]


def _parse_markdown_table(markdown: str) -> tuple[list[str], list[list[str]]]:
    """解析 markdown 表格。

    Args:
        markdown: 表格 markdown 文本。

    Returns:
        `(headers, rows)`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    lines = [line for line in markdown.splitlines() if line.strip()]
    if len(lines) < 2:
        return [], []
    headers = [_normalize_whitespace(item) for item in _split_table_cells(lines[0])]

    rows: list[list[str]] = []
    for line in lines[2:]:
        row = [_normalize_whitespace(item) for item in _split_table_cells(line)]
        if len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))
        elif len(row) > len(headers):
            row = row[: len(headers)]
        rows.append(row)
    return headers, rows


def _rows_to_records(headers: list[str], rows: list[list[str]]) -> list[dict[str, Optional[str]]]:
    """把表格行转换为 records。

    Args:
        headers: 列名列表。
        rows: 数据行列表。

    Returns:
        records 列表。

    Raises:
        RuntimeError: 转换失败时抛出。
    """

    records: list[dict[str, Optional[str]]] = []
    for row in rows:
        record: dict[str, Optional[str]] = {}
        for index, header in enumerate(headers):
            value = row[index] if index < len(row) else ""
            record[header] = value or None
        records.append(record)
    return records


def _extract_table_caption(lines: list[str], table_start_line: int) -> Optional[str]:
    """提取表格标题。

    Args:
        lines: 全部行。
        table_start_line: 表格起始行号。

    Returns:
        标题字符串或 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    current_line = table_start_line - 1
    while current_line >= 0:
        text = _normalize_whitespace(lines[current_line])
        if not text:
            current_line -= 1
            continue
        if re.match(r"^#{1,6}\s+", text):
            return None
        return text[:120]
    return None


def _extract_context_before(lines: list[str], table_start_line: int, max_chars: int = 200) -> str:
    """提取表格前文。

    Args:
        lines: 全部行。
        table_start_line: 表格起始行。
        max_chars: 最大字符数。

    Returns:
        前文文本。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    parts: list[str] = []
    total_len = 0
    current_line = table_start_line - 1
    while current_line >= 0:
        raw = _normalize_whitespace(lines[current_line])
        if not raw:
            current_line -= 1
            continue
        if re.match(r"^#{1,6}\s+", raw):
            break
        parts.append(raw)
        total_len += len(raw)
        if total_len >= max_chars:
            break
        current_line -= 1
    if not parts:
        return ""
    parts.reverse()
    merged = _normalize_whitespace(" ".join(parts))
    if len(merged) <= max_chars:
        return merged
    return merged[-max_chars:]




def _classify_table_type(
    *,
    row_count: int,
    col_count: int,
    headers: list[str],
    context_before: str,
) -> str:
    """分类表格类型。

    Args:
        row_count: 行数。
        col_count: 列数。
        headers: 表头。
        context_before: 前文。

    Returns:
        `data`/`layout`。

    Raises:
        RuntimeError: 分类失败时抛出。
    """

    if row_count <= 1 and col_count <= 2:
        return "layout"
    if not headers and len(context_before) < 10:
        return "layout"
    return "data"
