"""Host 内部共享的轻量规范化工具。"""

from __future__ import annotations

from datetime import datetime, timezone


def _coerce_string_tuple(values: object) -> tuple[str, ...]:
    """将任意列表值规范化为字符串元组。

    Args:
        values: 原始值。

    Returns:
        过滤空白后的字符串元组；非列表输入返回空元组。

    Raises:
        无。
    """

    if not isinstance(values, list):
        return ()
    normalized: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO 字符串。

    Args:
        无。

    Returns:
        ISO 8601 字符串。

    Raises:
        无。
    """

    return datetime.now(timezone.utc).isoformat()


def _normalize_session_id(session_id: str) -> str:
    """规范化 session_id。

    Args:
        session_id: 原始会话 ID。

    Returns:
        去除首尾空白后的会话 ID。

    Raises:
        ValueError: 当会话 ID 为空时抛出。
    """

    normalized = str(session_id or "").strip()
    if not normalized:
        raise ValueError("session_id 不能为空")
    return normalized


__all__ = ["_coerce_string_tuple", "_normalize_session_id", "_utc_now_iso"]