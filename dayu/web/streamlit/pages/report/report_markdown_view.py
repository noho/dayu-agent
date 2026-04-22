"""分析报告 Markdown 展示辅助模块。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html import escape
from types import TracebackType
from typing import Protocol, cast

import streamlit as streamlit_module

_MARKDOWN_HEADING_PATTERN = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_MARKDOWN_FENCE_PREFIXES = ("```", "~~~")
_REPORT_LAYOUT_COLUMN_WIDTHS = [1, 3]
_TOC_INDENT_REM = 1.0
_REPORT_PANEL_MIN_HEIGHT_PX = 1200
_REPORT_PANEL_MAX_HEIGHT_PX = 4800
_REPORT_PANEL_BASE_HEIGHT_PX = 1200
_REPORT_PANEL_HEIGHT_PER_CONTENT_LINE_PX = 0.2
_REPORT_PANEL_HEIGHT_PER_HEADING_PX = 4
_REPORT_PANEL_HEIGHT_CALIBRATION_FACTOR = 1.0


class _ColumnContextProtocol(Protocol):
    """列上下文协议。"""

    def __enter__(self) -> "_ColumnContextProtocol":
        """进入列上下文。"""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """退出列上下文。"""
        ...


class _ContainerContextProtocol(Protocol):
    """容器上下文协议。"""

    def __enter__(self) -> "_ContainerContextProtocol":
        """进入容器上下文。"""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        """退出容器上下文。"""
        ...


class _StreamlitMarkdownViewProtocol(Protocol):
    """Markdown 展示最小 Streamlit 协议。"""

    def columns(self, spec: list[int], *, gap: str = "small") -> list[_ColumnContextProtocol]:
        """创建列。"""
        ...

    def markdown(self, body: str, unsafe_allow_html: bool = False) -> None:
        """渲染 Markdown。"""
        ...

    def container(self, *, border: bool = False, height: int | None = None) -> _ContainerContextProtocol:
        """创建容器。"""
        ...


st = cast(_StreamlitMarkdownViewProtocol, streamlit_module)


@dataclass(frozen=True)
class MarkdownHeading:
    """Markdown 标题目录项。"""

    line_index: int
    level: int
    title: str
    anchor: str


def extract_markdown_headings(markdown_content: str) -> list[MarkdownHeading]:
    """从 Markdown 文本中提取标题目录。"""

    headings: list[MarkdownHeading] = []
    anchor_counts: dict[str, int] = {}
    in_fenced_code_block = False

    for line_index, line in enumerate(markdown_content.splitlines()):
        stripped_line = line.strip()
        if is_markdown_fence_line(stripped_line):
            in_fenced_code_block = not in_fenced_code_block
            continue
        if in_fenced_code_block:
            continue

        match = _MARKDOWN_HEADING_PATTERN.match(line)
        if match is None:
            continue

        title = normalize_markdown_heading_title(match.group("title"))
        if not title:
            continue

        level = len(match.group("marks"))
        base_anchor = slugify_markdown_heading(title)
        duplicate_count = anchor_counts.get(base_anchor, 0)
        anchor_counts[base_anchor] = duplicate_count + 1
        anchor = base_anchor if duplicate_count == 0 else f"{base_anchor}-{duplicate_count + 1}"
        headings.append(
            MarkdownHeading(
                line_index=line_index,
                level=level,
                title=title,
                anchor=anchor,
            )
        )

    return headings


def is_markdown_fence_line(stripped_line: str) -> bool:
    """判断当前行是否为 Markdown 围栏代码块分隔符。"""

    return stripped_line.startswith(_MARKDOWN_FENCE_PREFIXES)


def normalize_markdown_heading_title(raw_title: str) -> str:
    """规范化 Markdown 标题文本。"""

    return raw_title.strip().rstrip("#").strip()


def slugify_markdown_heading(title: str) -> str:
    """将标题文本转换为稳定锚点。"""

    normalized = unicodedata.normalize("NFKC", title).lower()
    slug_characters: list[str] = []

    for character in normalized:
        if character.isalnum():
            slug_characters.append(character)
            continue
        if character in {" ", "-", "_"}:
            slug_characters.append("-")

    collapsed_slug = re.sub(r"-{2,}", "-", "".join(slug_characters)).strip("-")
    if collapsed_slug:
        return collapsed_slug
    return "section"


def build_report_toc_html(headings: list[MarkdownHeading]) -> str:
    """构建报告目录 HTML。"""

    toc_headings = [heading for heading in headings if heading.level <= 4]
    if not toc_headings:
        return (
            '<div style="font-size: 0.875rem; color: #6b7280;">'
            "当前报告未检测到可展示的 Markdown 目录。"
            "</div>"
        )

    base_level = min(heading.level for heading in toc_headings)
    html_lines = [
        (
            '<div class="report-toc-root" '
            'style="font-weight: 600; margin-bottom: 0.35rem; overflow-anchor: none;">'
            "点击条目可跳转到对应章节"
            "</div>"
        ),
    ]
    for heading in toc_headings:
        padding_left = max(heading.level - base_level, 0) * _TOC_INDENT_REM
        html_lines.append(
            (
                f'<div style="padding-left: {padding_left:.1f}rem; margin: 0.3rem 0;">'
                f'<a href="#{escape(heading.anchor, quote=True)}" '
                'onclick="this.blur();" '
                'style="text-decoration: none;">'
                f"{escape(heading.title)}"
                "</a>"
                "</div>"
            )
        )
    return "\n".join(html_lines)


def inject_heading_anchors(markdown_content: str, headings: list[MarkdownHeading]) -> str:
    """为 Markdown 标题注入锚点。"""

    if not headings:
        return markdown_content

    heading_anchor_map = {heading.line_index: heading.anchor for heading in headings}
    rendered_lines: list[str] = []

    for line_index, line in enumerate(markdown_content.splitlines()):
        anchor = heading_anchor_map.get(line_index)
        if anchor is not None:
            heading_match = _MARKDOWN_HEADING_PATTERN.match(line)
            if heading_match is not None:
                heading_marks = heading_match.group("marks")
                heading_title = normalize_markdown_heading_title(heading_match.group("title"))
                rendered_lines.append(
                    f'{heading_marks} <span id="{escape(anchor, quote=True)}"></span>{heading_title}'
                )
                continue
        rendered_lines.append(line)

    return "\n".join(rendered_lines)


def render_markdown_report(markdown_content: str) -> None:
    """按“目录 + 正文”布局渲染 Markdown 报告。"""

    headings = extract_markdown_headings(markdown_content)
    anchored_markdown = inject_heading_anchors(markdown_content, headings)
    panel_height_px = get_report_panel_container_height_px(markdown_content, headings)
    toc_column, content_column = st.columns(_REPORT_LAYOUT_COLUMN_WIDTHS, gap="large")

    with toc_column:
        st.markdown("#### 目录")
        with st.container(border=True, height=panel_height_px):
            st.markdown(build_report_toc_html(headings), unsafe_allow_html=True)

    with content_column:
        st.markdown("#### 正文")
        with st.container(border=True, height=panel_height_px):
            st.markdown(anchored_markdown, unsafe_allow_html=True)


def get_report_panel_container_height_px(markdown_content: str, headings: list[MarkdownHeading]) -> int:
    """动态计算报告双栏容器高度。"""

    content_line_count = len(markdown_content.splitlines())
    estimated_height_px = (
        _REPORT_PANEL_BASE_HEIGHT_PX
        + content_line_count * _REPORT_PANEL_HEIGHT_PER_CONTENT_LINE_PX
        + len(headings) * _REPORT_PANEL_HEIGHT_PER_HEADING_PX
    )
    estimated_height_px = int(estimated_height_px * _REPORT_PANEL_HEIGHT_CALIBRATION_FACTOR)
    return clamp_report_panel_height_px(estimated_height_px)


def clamp_report_panel_height_px(estimated_height_px: int) -> int:
    """将动态估算高度约束在可读区间内。"""

    if estimated_height_px < _REPORT_PANEL_MIN_HEIGHT_PX:
        return _REPORT_PANEL_MIN_HEIGHT_PX
    if estimated_height_px > _REPORT_PANEL_MAX_HEIGHT_PX:
        return _REPORT_PANEL_MAX_HEIGHT_PX
    return estimated_height_px
