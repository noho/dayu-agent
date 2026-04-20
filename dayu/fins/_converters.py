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


def require_non_empty_text(value: object, *, empty_error: Exception) -> str:
    """标准化必填文本：仅拒绝 `None` 与空白字符串。

    该函数复用 `normalize_optional_text()` 的语义：
    - `None` 与去空白后为空的字符串视为缺失；
    - 其它值先执行 `str(...).strip()` 后保留，因此 `0` / `False`
      不会被误判为空。

    Args:
        value: 原始值。
        empty_error: 值缺失时需要抛出的异常实例。

    Returns:
        去空白后的非空字符串。

    Raises:
        Exception: 当值缺失时抛出调用方提供的异常。
    """

    normalized = normalize_optional_text(value)
    if normalized is None:
        raise empty_error
    return normalized


__all__ = [
    "optional_int",
    "int_or_zero",
    "normalize_optional_text",
    "require_non_empty_text",
]
