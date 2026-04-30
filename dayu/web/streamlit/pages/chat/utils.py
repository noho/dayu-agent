"""聊天模块公共工具函数。

纯函数，无 Streamlit 依赖，不访问 session_state。
"""

from __future__ import annotations

import hashlib
import re
import time

from dayu.contracts.events import AppEvent, AppEventType
from dayu.fins.ticker_normalization import normalize_ticker, ticker_to_company_id

MODULE = "dayu.web.streamlit.pages.chat.utils"

_TEXT_PAYLOAD_KEYS: tuple[str, ...] = ("content", "text", "answer")
_CANCELLED_DEFAULT_MESSAGE = "执行已取消"

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

    参数:
        payload: 事件负载，支持字符串或字典。

    返回值:
        规范化后的文本；无法提取时返回空字符串。

    异常:
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

    参数:
        payload: 事件负载，支持字符串或字典。

    返回值:
        提取到的消息文本；无法提取时返回空字符串。

    异常:
        无。
    """

    if isinstance(payload, str):
        return payload.strip()
    elif isinstance(payload, dict):
        for key in ("message", "error", "detail", "content"):
            candidate = payload.get(key)
            if isinstance(candidate, str):
                normalized_message = candidate.strip()
                if normalized_message:
                    return normalized_message
        return str(payload).strip()
    else:
        return ""


def _format_cancelled_message(payload: str | dict[str, str | bool]) -> str:
    """格式化取消事件文案。

    参数:
        payload: 取消事件负载。

    返回值:
        用户可读的取消提示文案。

    异常:
        无。
    """

    if isinstance(payload, dict):
        reason = payload.get("cancel_reason")
        if isinstance(reason, str) and reason.strip():
            return f"{_CANCELLED_DEFAULT_MESSAGE}：{reason.strip()}"
    return _CANCELLED_DEFAULT_MESSAGE


def fold_app_events_to_assistant_text(events: list[AppEvent]) -> tuple[str, list[str], bool]:
    """把事件流折叠为主文、侧边消息与过滤标记。

    注意:
        REASONING_DELTA 与 CONTENT_DELTA 均被折叠到同一主文文本中；
        如需分区展示，应直接消费 ``StreamQueueItem`` 并区分 ``kind``。

    参数:
        events: 应用层事件列表。

    返回值:
        三元组 ``(assistant_text, side_messages, filtered)``。

    异常:
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


def extract_stream_text(payload: str | dict[str, str | bool]) -> str:
    """提取流式事件可展示文本，并保留 Markdown 所需空白。

    参数:
        payload: 流式事件负载，支持字符串或字典。

    返回值:
        可展示文本；当负载仅包含空白时返回空字符串。

    异常:
        无。
    """

    return _payload_to_text(payload)


def build_request_trace_id(*, ticker: str, user_text: str) -> str:
    """构建单次提交的日志追踪 ID。

    参数:
        ticker: 股票代码。
        user_text: 用户输入文本。

    返回值:
        格式为 ``TICKER-TIMESTAMP-HASH`` 的追踪标识。

    异常:
        无。
    """

    prefix = ticker.strip().upper()[:8] or "UNKNOWN"
    text_hash = int(hashlib.md5(user_text.encode()).hexdigest(), 16) % 100000
    return f"{prefix}-{int(time.time() * 1000)}-{text_hash:05d}"


def build_chat_session_id(ticker: str) -> str:
    """根据股票代码生成固定的聊天会话标识。

    使用 ticker 归一化后的 canonical 编码生成 session_id，确保同一公司的
    不同 ticker 写法（如 ``AAPL`` 与 ``AAPL.US``）映射到同一会话。
    格式: ``streamlit-web-{COMPANY_ID}``。

    参数:
        ticker: 股票代码，支持常见市场前后缀变体。

    返回值:
        固定格式的会话标识。

    异常:
        无；归一化失败时回退到 ``ticker.strip().upper()``。
    """

    try:
        normalized = normalize_ticker(ticker)
        company_id = ticker_to_company_id(normalized)
    except ValueError:
        company_id = ticker.strip().upper()
    return f"streamlit-web-{company_id}"


def summarize_user_text(user_text: str) -> str:
    """生成用户输入脱敏摘要，避免日志打印完整内容。

    参数:
        user_text: 原始用户输入。

    返回值:
        包含长度与预览片段的摘要字符串。

    异常:
        无。
    """

    normalized = " ".join(user_text.strip().split())
    preview = normalized[:48]
    suffix = "..." if len(normalized) > len(preview) else ""
    return f"len={len(user_text)}, preview={preview!r}{suffix}"


def should_keep_current_frame_for_side_effects(*, assistant_text: str, side_messages: list[str]) -> bool:
    """判断是否需要保留当前页面帧以展示侧边错误。

    参数:
        assistant_text: 助手回复正文文本。
        side_messages: 流式副作用消息列表（warning/error/cancelled）。

    返回值:
        当回复为空且存在副作用消息时返回 ``True``，表示应保留当前帧。

    异常:
        无。
    """

    return (not assistant_text.strip()) and bool(side_messages)
