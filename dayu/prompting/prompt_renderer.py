"""Prompt 模板渲染模块。

本模块只负责 Markdown prompt 模板的条件块处理与变量替换，
不依赖 Engine 运行时、PromptComposer 或 scene 定义对象。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from dayu.prompt_template_rendering import replace_template_variables


class PromptParseError(Exception):
    """Prompt 模板解析错误。"""


GuidanceParseError = PromptParseError


# 模板条件块的固定字面量正则；提升到模块级以避免每次解析重复编译。
_WHEN_TOOL_OPEN_PATTERN = re.compile(r"<when_tool\s+([a-zA-Z_][a-zA-Z0-9_]*)>")
_WHEN_TOOL_CLOSE_PATTERN = re.compile(r"</when_tool>")
_WHEN_TAG_OPEN_PATTERN = re.compile(r"<when_tag\s+([a-zA-Z0-9_\-,\s]+)>")
_WHEN_TAG_CLOSE_PATTERN = re.compile(r"</when_tag>")


@dataclass
class ConditionalBlock:
    """条件块表示。

    Args:
        tool_name: 工具名称条件。
        content: 块内容。
        start_pos: 开始位置。
        end_pos: 结束位置。

    Returns:
        无。

    Raises:
        无。
    """

    tool_name: str
    content: str
    start_pos: int
    end_pos: int


def parse_when_tool_blocks(template: str, tool_names: set[str]) -> str:
    """解析并处理 `<when_tool>` 条件块。

    Args:
        template: 模板字符串。
        tool_names: 已注册工具名称集合。

    Returns:
        处理后的模板文本。

    Raises:
        PromptParseError: 标签不匹配或格式错误时抛出。
    """

    open_tag_pattern = _WHEN_TOOL_OPEN_PATTERN
    close_tag_pattern = _WHEN_TOOL_CLOSE_PATTERN

    result_parts: list[str] = []
    position = 0

    while position < len(template):
        open_match = open_tag_pattern.search(template, position)
        if open_match is None:
            result_parts.append(template[position:])
            break

        open_line_start = template.rfind("\n", 0, open_match.start()) + 1
        open_line_end_pos = template.find("\n", open_match.end())
        if open_line_end_pos == -1:
            open_line_end_pos = len(template)

        before_tag = template[open_line_start:open_match.start()]
        after_tag = template[open_match.end():open_line_end_pos]
        open_is_standalone = before_tag.strip() == "" and after_tag.strip() == ""

        tool_name = open_match.group(1)

        if open_is_standalone:
            result_parts.append(template[position:open_line_start])
        else:
            result_parts.append(template[position:open_match.start()])

        if open_is_standalone:
            block_start = open_line_end_pos + 1 if open_line_end_pos < len(template) else open_line_end_pos
        else:
            block_start = open_match.end()

        block_content, block_end = extract_block_content(
            template,
            block_start,
            open_tag_pattern,
            close_tag_pattern,
            "when_tool",
        )

        close_tag_start = block_end - len("</when_tool>")
        close_line_start = template.rfind("\n", 0, close_tag_start) + 1
        close_line_end_pos = template.find("\n", block_end)
        if close_line_end_pos == -1:
            close_line_end_pos = len(template)

        before_close = template[close_line_start:close_tag_start]
        after_close = template[block_end:close_line_end_pos]
        close_is_standalone = before_close.strip() == "" and after_close.strip() == ""

        if close_is_standalone and close_line_start > 0:
            actual_content = template[block_start:close_line_start - 1]
        else:
            actual_content = block_content

        processed_content = parse_when_tool_blocks(actual_content, tool_names)
        if tool_name in tool_names:
            result_parts.append(processed_content)
            if close_is_standalone and not processed_content.endswith("\n"):
                result_parts.append("\n")

        if close_is_standalone:
            if close_line_end_pos < len(template):
                position = close_line_end_pos + 1
            else:
                position = len(template)
        else:
            position = block_end

    return "".join(result_parts)


def parse_when_tag_blocks(template: str, tag_names: set[str]) -> str:
    """解析并处理 `<when_tag>` 条件块。

    Args:
        template: 模板字符串。
        tag_names: 已启用 tag 名称集合。

    Returns:
        处理后的模板文本。

    Raises:
        PromptParseError: 标签不匹配或格式错误时抛出。
    """

    open_tag_pattern = _WHEN_TAG_OPEN_PATTERN
    close_tag_pattern = _WHEN_TAG_CLOSE_PATTERN

    result_parts: list[str] = []
    position = 0

    while position < len(template):
        open_match = open_tag_pattern.search(template, position)
        if open_match is None:
            result_parts.append(template[position:])
            break

        open_line_start = template.rfind("\n", 0, open_match.start()) + 1
        open_line_end_pos = template.find("\n", open_match.end())
        if open_line_end_pos == -1:
            open_line_end_pos = len(template)

        before_tag = template[open_line_start:open_match.start()]
        after_tag = template[open_match.end():open_line_end_pos]
        open_is_standalone = before_tag.strip() == "" and after_tag.strip() == ""

        tag_expr = open_match.group(1)
        tags = [tag.strip() for tag in tag_expr.split(",") if tag.strip()]

        if open_is_standalone:
            result_parts.append(template[position:open_line_start])
        else:
            result_parts.append(template[position:open_match.start()])

        if open_is_standalone:
            block_start = open_line_end_pos + 1 if open_line_end_pos < len(template) else open_line_end_pos
        else:
            block_start = open_match.end()

        block_content, block_end = extract_block_content(
            template,
            block_start,
            open_tag_pattern,
            close_tag_pattern,
            "when_tag",
        )

        close_tag_start = block_end - len("</when_tag>")
        close_line_start = template.rfind("\n", 0, close_tag_start) + 1
        close_line_end_pos = template.find("\n", block_end)
        if close_line_end_pos == -1:
            close_line_end_pos = len(template)

        before_close = template[close_line_start:close_tag_start]
        after_close = template[block_end:close_line_end_pos]
        close_is_standalone = before_close.strip() == "" and after_close.strip() == ""

        if close_is_standalone and close_line_start > 0:
            actual_content = template[block_start:close_line_start - 1]
        else:
            actual_content = block_content

        processed_content = parse_when_tag_blocks(actual_content, tag_names)
        if tags and all(tag in tag_names for tag in tags):
            result_parts.append(processed_content)
            if close_is_standalone and processed_content and not processed_content.endswith("\n"):
                result_parts.append("\n")

        if close_is_standalone:
            if close_line_end_pos < len(template):
                position = close_line_end_pos + 1
            else:
                position = len(template)
        else:
            position = block_end

    return "".join(result_parts)


def extract_block_content(
    template: str,
    start_pos: int,
    open_tag_pattern: re.Pattern[str],
    close_tag_pattern: re.Pattern[str],
    tag_label: str,
) -> tuple[str, int]:
    """提取嵌套条件块内容。

    Args:
        template: 模板字符串。
        start_pos: 块内容起始位置。
        open_tag_pattern: 开标签正则。
        close_tag_pattern: 闭标签正则。
        tag_label: 当前标签名，仅用于错误提示。

    Returns:
        `(块内容, 块结束位置)`。

    Raises:
        PromptParseError: 标签不匹配时抛出。
    """

    depth = 1
    position = start_pos

    while position < len(template) and depth > 0:
        next_open = open_tag_pattern.search(template, position)
        next_close = close_tag_pattern.search(template, position)

        if next_open and next_close:
            if next_open.start() < next_close.start():
                depth += 1
                position = next_open.end()
            else:
                depth -= 1
                if depth == 0:
                    content = template[start_pos:next_close.start()]
                    return content, next_close.end()
                position = next_close.end()
        elif next_close:
            depth -= 1
            if depth == 0:
                content = template[start_pos:next_close.start()]
                return content, next_close.end()
            position = next_close.end()
        elif next_open:
            raise PromptParseError(f"未找到匹配的 </{tag_label}> 标签，位置 {start_pos}")
        else:
            break

    raise PromptParseError(f"未找到匹配的 </{tag_label}> 标签，位置 {start_pos}")


def load_prompt(
    template: str,
    variables: dict[str, Any] | None = None,
    tool_names: set[str] | None = None,
    tag_names: set[str] | None = None,
) -> str:
    """加载并渲染 Markdown prompt 模板。

    Args:
        template: Markdown 模板文本。
        variables: 可选变量字典。
        tool_names: 可选工具名称集合。
        tag_names: 可选 tag 名称集合。

    Returns:
        渲染后的 prompt 文本。

    Raises:
        PromptParseError: 模板解析错误时抛出。
    """

    if not template:
        return ""
    if tool_names is not None:
        template = parse_when_tool_blocks(template, tool_names)
    if tag_names is not None:
        template = parse_when_tag_blocks(template, tag_names)
    if variables:
        template = replace_template_variables(template, variables)
    return template.strip()


__all__ = [
    "ConditionalBlock",
    "GuidanceParseError",
    "PromptParseError",
    "extract_block_content",
    "load_prompt",
    "parse_when_tag_blocks",
    "parse_when_tool_blocks",
]
