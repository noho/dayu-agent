"""Host 层共享日期时间工具函数。

提供 UTC 时间获取、ISO 序列化与解析的统一实现，
供 session_registry、run_registry、pending_turn_store、
reply_outbox_store、concurrency 等模块共用。
"""

from __future__ import annotations

from datetime import datetime, timezone


def now_utc() -> datetime:
    """返回当前 UTC 时间。

    Args:
        无。

    Returns:
        当前 UTC 时间。

    Raises:
        无。
    """

    return datetime.now(timezone.utc)


def serialize_dt(value: datetime) -> str:
    """将时间戳序列化为 ISO 8601 文本。

    Args:
        value: 待序列化时间。

    Returns:
        ISO 8601 文本。

    Raises:
        无。
    """

    return value.isoformat()


def parse_dt(value: str) -> datetime:
    """将 ISO 8601 文本解析为时间戳。

    Args:
        value: ISO 8601 文本。

    Returns:
        解析后的时间戳。

    Raises:
        ValueError: 文本非法时抛出。
    """

    return datetime.fromisoformat(value)


def parse_dt_optional(value: str | None) -> datetime | None:
    """将 ISO 8601 文本解析为时间戳，None 输入返回 None。

    Args:
        value: ISO 8601 文本或 None。

    Returns:
        解析后的时间戳；None 输入返回 None。

    Raises:
        ValueError: 文本非法时抛出。
    """

    if value is None:
        return None
    return datetime.fromisoformat(value)
