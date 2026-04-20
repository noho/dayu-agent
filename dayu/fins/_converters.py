"""fins 通用标量转换工具。

提供跨 fins 子模块共享的可选整数解析与可选文本标准化函数，
避免各处重复定义。
"""

from __future__ import annotations

from typing import SupportsInt, cast


def optional_int(value: object) -> int | None:
    """把可选标量安全收敛为整数。

    Args:
        value: 原始标量值，可为 ``None``、空字符串或任何可转换为整数的值。

    Returns:
        成功解析时返回对应整数；无法解析或为空时返回 ``None``。

    Raises:
        无。
    """

    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float, str, bytes, bytearray)):
            return int(value)
        if hasattr(value, "__int__"):
            return int(cast(SupportsInt, value))
        return None
    except (TypeError, ValueError):
        return None


def int_or_zero(value: object) -> int:
    """把任意可选标量收敛为整数，失败时回落为 0。

    Args:
        value: 原始标量值。

    Returns:
        成功解析时返回对应整数；否则返回 ``0``。

    Raises:
        无。
    """

    normalized = optional_int(value)
    return normalized if normalized is not None else 0


def normalize_optional_text(value: object) -> str | None:
    """标准化可选文本：去首尾空白，空值归 None。

    Args:
        value: 原始值，可为 ``None``、空字符串或任意可 ``str()`` 的对象。

    Returns:
        去空白后的非空字符串；值为 ``None`` 或去空白后为空时返回 ``None``。

    Raises:
        无。
    """

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


__all__ = [
    "optional_int",
    "int_or_zero",
    "normalize_optional_text",
]
