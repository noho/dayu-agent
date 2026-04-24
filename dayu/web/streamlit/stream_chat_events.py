"""Streamlit 聊天事件折叠工具。

负责把 Service 层 `AppEvent` 序列折叠成：
1. 助手主文文本；
2. 侧边提示消息列表（warning / error / cancelled）；
3. 内容过滤标记。
"""

from __future__ import annotations

import re

from dayu.contracts.events import AppEvent, AppEventType

_CANCELLED_DEFAULT_MESSAGE = "执行已取消"
_TEXT_PAYLOAD_KEYS: tuple[str, ...] = ("content", "text", "answer")

# Markdown 语法解析相关正则表达式, 修复 Markdown 语法解析错误。
_CODE_FENCE_PATTERN = re.compile(r"(```[\s\S]*?```)")
_INLINE_HEADING_PATTERN = re.compile(r"([^\n])(#{2,6})(?=[^#\s])")
_HEADING_SPACE_PATTERN = re.compile(r"(?m)^(#{1,6})([^ #\n])")
_INLINE_STAR_LIST_PATTERN = re.compile(r"(\S)(?<!\*)(\* )")
_INLINE_SEC_BULLET_PATTERN = re.compile(r"(\S)(- SEC EDGAR)")
_HEADING_INLINE_BOLD_DASH_LIST_PATTERN = re.compile(r"(?m)^(#{1,6}\s[^\n|]+?)(-\s+\*\*)")
_HEADING_INLINE_TABLE_PATTERN = re.compile(r"(?m)^(#{1,6}\s[^\n|]+?)(\|)")
_INLINE_TABLE_START_PATTERN = re.compile(r"^([^|\n][^|\n]*?)(\|.+\|)$")
_INLINE_TABLE_ROW_SPLIT_PATTERN = re.compile(r"\|\s+\|")
_SEC_LINE_WITHOUT_BULLET_PATTERN = re.compile(r"(?m)^(SEC EDGAR \|)")


def _normalize_markdown_outside_code_fence(text: str) -> str:
    """规整非代码块区域的 Markdown 字符串。

    参数:
        text: 原始 Markdown 文本。

    返回值:
        规整后的 Markdown 文本。

    异常:
        无。
    """

    normalized = text.replace("\\n", "\n")
    normalized = _INLINE_HEADING_PATTERN.sub(r"\1\n\2", normalized)
    normalized = _HEADING_SPACE_PATTERN.sub(r"\1 \2", normalized)
    normalized = _INLINE_STAR_LIST_PATTERN.sub(r"\1\n\2", normalized)
    normalized = _INLINE_SEC_BULLET_PATTERN.sub(r"\1\n\2", normalized)
    normalized = _HEADING_INLINE_BOLD_DASH_LIST_PATTERN.sub(r"\1\n\2", normalized)
    normalized = _HEADING_INLINE_TABLE_PATTERN.sub(r"\1\n\2", normalized)
    normalized = normalized.replace("\n\n##", "\n##")
    if "|---|---| |" in normalized:
        normalized = normalized.replace("|---|---| |", "|---|---|\n|")
    if " | | " in normalized:
        normalized = normalized.replace(" | | ", " |\n| ")
    ends_with_newline = normalized.endswith("\n")
    normalized_lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = raw_line
        if ("|---" in line) and (line.count("|") >= 6):
            table_start_match = _INLINE_TABLE_START_PATTERN.match(line)
            if table_start_match is not None:
                line = f"{table_start_match.group(1)}\n{table_start_match.group(2)}"
            line = _INLINE_TABLE_ROW_SPLIT_PATTERN.sub("|\n| ", line)
        normalized_lines.append(line)
    normalized = "\n".join(normalized_lines)
    if ends_with_newline and (not normalized.endswith("\n")):
        normalized = f"{normalized}\n"
    normalized = re.sub(r"(?m)^\|\s{2,}", "| ", normalized)
    if "- SEC EDGAR |" in normalized:
        normalized = _SEC_LINE_WITHOUT_BULLET_PATTERN.sub(r"- \1", normalized)
    return normalized


def normalize_stream_text_for_markdown(text: str) -> str:
    """规整流式 Markdown 文本，避免结构被压扁。

    参数:
        text: 原始流式文本。

    返回值:
        规整后的 Markdown 文本；代码块内容保持原样。

    异常:
        无。
    """

    if not text:
        return ""
    parts = _CODE_FENCE_PATTERN.split(text)
    normalized_parts: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("```") and part.endswith("```"):
            normalized_parts.append(part.replace("\\n", "\n"))
            continue
        normalized_parts.append(_normalize_markdown_outside_code_fence(part))
    return "".join(normalized_parts)

def _payload_to_text(payload: str | dict[str, str | bool]) -> str:
    """将事件负载规范化为文本。

    Args:
        payload: 事件负载，支持字符串或字典。

    Returns:
        规范化后的文本；无法提取时返回空字符串。

    Raises:
        无。
    """

    if isinstance(payload, str):
        if payload.strip():
            return normalize_stream_text_for_markdown(payload)
    if isinstance(payload, dict):
        for key in _TEXT_PAYLOAD_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, str):
                if candidate.strip():
                    return normalize_stream_text_for_markdown(candidate)
    return ""


def _payload_message(payload: str | dict[str, str | bool]) -> str:
    """提取 warning/error 事件的 message 字段。

    Args:
        payload: 事件负载，支持字符串或字典。

    Returns:
        提取到的消息文本；无法提取时返回空字符串。

    Raises:
        无。
    """

    if isinstance(payload, str):
        return payload.strip()
    if isinstance(payload, dict):
        for key in ("message", "error", "detail", "content"):
            candidate = payload.get(key)
            if isinstance(candidate, str):
                normalized_message = candidate.strip()
                if normalized_message:
                    return normalized_message
        return str(payload).strip()


def _format_cancelled_message(payload: str | dict[str, str | bool]) -> str:
    """格式化取消事件文案。

    Args:
        payload: 取消事件负载。

    Returns:
        用户可读的取消提示文案。

    Raises:
        无。
    """

    if isinstance(payload, dict):
        reason = payload.get("cancel_reason")
        if isinstance(reason, str) and reason.strip():
            return f"{_CANCELLED_DEFAULT_MESSAGE}：{reason.strip()}"
    return _CANCELLED_DEFAULT_MESSAGE


def fold_app_events_to_assistant_text(events: list[AppEvent]) -> tuple[str, list[str], bool]:
    """把事件流折叠为主文、侧边消息与过滤标记。

    Args:
        events: 应用层事件列表。

    Returns:
        三元组 `(assistant_text, side_messages, filtered)`。

    Raises:
        无。
    """

    text_parts: list[str] = []
    side_messages: list[str] = []
    filtered = False

    for event in events:
        payload = event.payload

        if event.type in (AppEventType.REASONING_DELTA, AppEventType.CONTENT_DELTA):
            chunk_text = _payload_to_text(payload if isinstance(payload, (dict, str)) else "")
            if chunk_text:
                text_parts.append(chunk_text)
            continue

        if event.type == AppEventType.FINAL_ANSWER:
            if isinstance(payload, dict):
                filtered_payload = payload.get("filtered")
                if isinstance(filtered_payload, bool):
                    filtered = filtered_payload
            if not text_parts:
                final_text = _payload_to_text(payload if isinstance(payload, (dict, str)) else "").strip()
                if final_text:
                    text_parts.append(final_text)
            continue

        if event.type in (AppEventType.WARNING, AppEventType.ERROR):
            message = _payload_message(payload if isinstance(payload, (dict, str)) else "")
            if message:
                side_messages.append(message)
            continue

        if event.type == AppEventType.CANCELLED:
            cancelled_message = _format_cancelled_message(payload if isinstance(payload, (dict, str)) else "")
            side_messages.append(cancelled_message)

    return "".join(text_parts), side_messages, filtered


__all__ = ["fold_app_events_to_assistant_text", "normalize_stream_text_for_markdown"]
