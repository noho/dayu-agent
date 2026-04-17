"""Toolset 通用配置快照契约。

该模块定义跨层流动的 toolset 配置真源：
- Service / Contract preparation 只负责把工具配置收敛为通用快照。
- Host 只负责在 scene preparation 中把单个 toolset 快照交给 registrar。
- 各 toolset adapter 负责把快照反序列化为所属包的专用配置对象。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import TypeAlias

ToolsetConfigScalar: TypeAlias = str | int | float | bool | None
ToolsetConfigValue: TypeAlias = (
    ToolsetConfigScalar | list["ToolsetConfigValue"] | dict[str, "ToolsetConfigValue"]
)
ToolsetConfigPayload: TypeAlias = dict[str, ToolsetConfigValue]


@dataclass(frozen=True)
class ToolsetConfigSnapshot:
    """单个 toolset 的通用配置快照。

    Args:
        toolset_name: toolset 稳定名称。
        version: 快照版本号。
        payload: 当前 toolset 的 JSON 兼容配置载荷。

    Returns:
        无。

    Raises:
        无。
    """

    toolset_name: str
    version: str = "1"
    payload: ToolsetConfigPayload = field(default_factory=dict)


def normalize_toolset_name(toolset_name: str) -> str:
    """规范化 toolset 名称。

    Args:
        toolset_name: 原始 toolset 名称。

    Returns:
        去首尾空白后的 toolset 名称。

    Raises:
        ValueError: 当名称为空时抛出。
    """

    normalized_name = str(toolset_name or "").strip()
    if not normalized_name:
        raise ValueError("toolset_name 不能为空")
    return normalized_name


def find_toolset_config(
    snapshots: tuple[ToolsetConfigSnapshot, ...],
    toolset_name: str,
) -> ToolsetConfigSnapshot | None:
    """按 toolset 名称查找配置快照。

    Args:
        snapshots: toolset 配置快照序列。
        toolset_name: 待查找的 toolset 名称。

    Returns:
        命中的 toolset 配置快照；不存在时返回 ``None``。

    Raises:
        无。
    """

    normalized_name = str(toolset_name or "").strip()
    if not normalized_name:
        return None
    for snapshot in snapshots:
        if snapshot.toolset_name == normalized_name:
            return snapshot
    return None


def replace_toolset_config(
    snapshots: tuple[ToolsetConfigSnapshot, ...],
    snapshot: ToolsetConfigSnapshot,
) -> tuple[ToolsetConfigSnapshot, ...]:
    """用同名快照替换或追加到序列中。

    Args:
        snapshots: 原始 toolset 配置快照序列。
        snapshot: 待写入的新快照。

    Returns:
        替换后的新序列。

    Raises:
        无。
    """

    normalized: list[ToolsetConfigSnapshot] = []
    replaced = False
    for existing in snapshots:
        if existing.toolset_name == snapshot.toolset_name:
            if not replaced:
                normalized.append(snapshot)
                replaced = True
            continue
        normalized.append(existing)
    if not replaced:
        normalized.append(snapshot)
    return tuple(normalized)


def normalize_toolset_configs(
    snapshots: tuple[ToolsetConfigSnapshot, ...],
) -> tuple[ToolsetConfigSnapshot, ...]:
    """规范化并去重 toolset 配置快照序列。

    Args:
        snapshots: 原始 toolset 配置快照序列。

    Returns:
        规范化后的快照序列；同名项以后出现者覆盖先前项。

    Raises:
        ValueError: 当某个 toolset 名称为空时抛出。
    """

    normalized: tuple[ToolsetConfigSnapshot, ...] = ()
    for snapshot in snapshots:
        normalized = replace_toolset_config(
            normalized,
            ToolsetConfigSnapshot(
                toolset_name=normalize_toolset_name(snapshot.toolset_name),
                version=str(snapshot.version or "1").strip() or "1",
                payload={
                    key: serialize_toolset_config_payload_value(value)
                    for key, value in snapshot.payload.items()
                },
            ),
        )
    return normalized


def serialize_toolset_config_payload_value(value: ToolsetConfigValue) -> ToolsetConfigValue:
    """把 toolset 配置值递归标准化为 JSON 兼容值。

    Args:
        value: 原始配置值。

    Returns:
        标准化后的配置值。

    Raises:
        TypeError: 当值类型不受支持时抛出。
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, list):
        return [serialize_toolset_config_payload_value(item) for item in value]
    if isinstance(value, tuple):
        return [serialize_toolset_config_payload_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): serialize_toolset_config_payload_value(item)
            for key, item in value.items()
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            str(key): serialize_toolset_config_payload_value(item)
            for key, item in asdict(value).items()
        }
    raise TypeError(f"不支持的 toolset 配置值类型: {type(value).__name__}")


def coerce_toolset_config_int(
    value: ToolsetConfigValue,
    *,
    field_name: str,
    default: int,
) -> int:
    """把 toolset 配置值收敛为整数。

    Args:
        value: 原始 toolset 配置值。
        field_name: 字段名，用于报错。
        default: 缺失或空字符串时使用的默认值。

    Returns:
        规范化后的整数值。

    Raises:
        TypeError: 当值无法收敛为整数时抛出。
    """

    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是整数，不能是布尔值")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise TypeError(f"{field_name} 必须是整数，当前值为小数")
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return default
        try:
            return int(normalized)
        except ValueError as exc:
            raise TypeError(f"{field_name} 必须是整数") from exc
    raise TypeError(f"{field_name} 必须是整数")


def coerce_toolset_config_float(
    value: ToolsetConfigValue,
    *,
    field_name: str,
    default: float,
) -> float:
    """把 toolset 配置值收敛为浮点数。

    Args:
        value: 原始 toolset 配置值。
        field_name: 字段名，用于报错。
        default: 缺失或空字符串时使用的默认值。

    Returns:
        规范化后的浮点数值。

    Raises:
        TypeError: 当值无法收敛为浮点数时抛出。
    """

    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f"{field_name} 必须是数字，不能是布尔值")
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return default
        try:
            return float(normalized)
        except ValueError as exc:
            raise TypeError(f"{field_name} 必须是数字") from exc
    raise TypeError(f"{field_name} 必须是数字")


def build_toolset_config_snapshot(
    toolset_name: str,
    payload: object | None,
    *,
    version: str = "1",
) -> ToolsetConfigSnapshot | None:
    """从专用配置对象构造通用 toolset 配置快照。

    Args:
        toolset_name: toolset 稳定名称。
        payload: 专用配置对象或 JSON 兼容对象；为空时返回 ``None``。
        version: 快照版本号。

    Returns:
        构造后的 toolset 配置快照；当 payload 为空时返回 ``None``。

    Raises:
        TypeError: 当 payload 类型不受支持时抛出。
        ValueError: 当 toolset_name 为空时抛出。
    """

    if payload is None:
        return None
    normalized_name = normalize_toolset_name(toolset_name)
    if is_dataclass(payload) and not isinstance(payload, type):
        normalized_payload = {
            str(key): serialize_toolset_config_payload_value(value)
            for key, value in asdict(payload).items()
        }
    elif isinstance(payload, dict):
        normalized_payload = {
            str(key): serialize_toolset_config_payload_value(value)
            for key, value in payload.items()
        }
    else:
        raise TypeError(f"无法从类型 {type(payload).__name__} 构造 toolset 配置快照")
    return ToolsetConfigSnapshot(
        toolset_name=normalized_name,
        version=str(version or "1").strip() or "1",
        payload=normalized_payload,
    )


__all__ = [
    "build_toolset_config_snapshot",
    "coerce_toolset_config_float",
    "coerce_toolset_config_int",
    "ToolsetConfigPayload",
    "ToolsetConfigScalar",
    "ToolsetConfigSnapshot",
    "ToolsetConfigValue",
    "find_toolset_config",
    "normalize_toolset_configs",
    "normalize_toolset_name",
    "replace_toolset_config",
    "serialize_toolset_config_payload_value",
]
