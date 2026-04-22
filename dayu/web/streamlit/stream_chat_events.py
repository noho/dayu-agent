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
_CANCELLED_REASON_PREFIX = "执行已取消："
_TEXT_PAYLOAD_KEYS: tuple[str, ...] = ("content", "text", "answer")
_FENCED_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
_INLINE_HEADING_WITHOUT_BREAK_PATTERN = re.compile(r"(?m)([^\n`])\s*(#{2,6})(?=[^\s#])")
_MARKDOWN_HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})([^\s#])")
_HEADING_INLINE_LIST_PATTERN = re.compile(r"(?m)^(#{1,6}\s[^\n*]+)\s*(\*\s+)")


def _normalize_markdown_headings(text: str) -> str:
    """规范化常见标题格式，避免 Streamlit Markdown 解析失败。

    Args:
        text: 已完成换行解码的文本。

    Returns:
        规范化后的文本。

    Raises:
        无。
    """

    with_heading_break = _INLINE_HEADING_WITHOUT_BREAK_PATTERN.sub(r"\1\n\2", text)
    heading_spaced = _MARKDOWN_HEADING_PATTERN.sub(r"\1 \2", with_heading_break)
    return _HEADING_INLINE_LIST_PATTERN.sub(r"\1\n\2", heading_spaced)


def _normalize_markdown_structures_outside_code_blocks(text: str) -> str:
    """只在代码块外执行 Markdown 结构纠偏。

    Args:
        text: 已完成换行解码的文本。

    Returns:
        规范化后的文本。

    Raises:
        无。
    """

    normalized_parts: list[str] = []
    last_end = 0
    for block in _FENCED_CODE_BLOCK_PATTERN.finditer(text):
        normalized_parts.append(_normalize_markdown_headings(text[last_end:block.start()]))
        normalized_parts.append(block.group(0))
        last_end = block.end()
    normalized_parts.append(_normalize_markdown_headings(text[last_end:]))
    return "".join(normalized_parts)


def normalize_stream_text_for_markdown(text: str) -> str:
    """把常见转义换行规范化为 Markdown 可渲染换行。

    Args:
        text: 原始文本片段。

    Returns:
        规范化后的文本。

    Raises:
        无。
    """

    unescaped_text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return _normalize_markdown_structures_outside_code_blocks(unescaped_text)


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
        normalized = normalize_stream_text_for_markdown(payload)
        return normalized if normalized.strip() else ""
    if isinstance(payload, dict):
        for key in _TEXT_PAYLOAD_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, str):
                normalized = normalize_stream_text_for_markdown(candidate)
                if normalized.strip():
                    return normalized
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
    return ""


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
            return f"{_CANCELLED_REASON_PREFIX}{reason.strip()}"
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
