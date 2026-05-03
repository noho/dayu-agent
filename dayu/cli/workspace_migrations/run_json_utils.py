"""run.json 工作区迁移的共享 JSON 读写工具。

本模块只承载迁移脚本之间共同需要的低层能力：类型化读取 JSON、
以缩进格式写回 JSON，以及使用临时文件完成原子替换。具体迁移规则
仍应放在各自模块内，避免把业务语义堆进共享工具。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypeAlias, cast


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def load_json_value(path: Path) -> JsonValue:
    """读取 JSON 文件并返回类型化 JSON 值。

    Args:
        path: 待读取的 JSON 文件路径。

    Returns:
        解析后的 JSON 值。

    Raises:
        OSError: 读取文件失败时抛出。
        json.JSONDecodeError: 文件内容不是合法 JSON 时抛出。
    """

    return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))


def as_json_object(value: JsonValue) -> JsonObject | None:
    """把 JSON 值收窄为对象字典。

    Args:
        value: 待检查的 JSON 值。

    Returns:
        当 ``value`` 是 JSON object 时返回字典，否则返回 None。

    Raises:
        无。
    """

    if isinstance(value, dict):
        return cast(JsonObject, value)
    return None


def write_json_value(path: Path, payload: JsonValue) -> None:
    """用项目迁移格式写回 JSON 文件。

    Args:
        path: 目标 JSON 文件路径。
        payload: 待序列化的 JSON 值。

    Returns:
        无。

    Raises:
        OSError: 写入或重命名失败时抛出。
        TypeError: ``payload`` 不可 JSON 序列化时抛出。
    """

    new_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, new_text)


def atomic_write_text(target_path: Path, content: str) -> None:
    """使用临时文件 + ``os.replace`` 原子替换 ``target_path``。

    Args:
        target_path: 目标文件路径。
        content: 待写入文本。

    Returns:
        无。

    Raises:
        OSError: 写入、刷盘或重命名失败时抛出。
    """

    temp_path = target_path.with_name(f".{target_path.name}.migrate.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as fp:
            fp.write(content)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp_path, target_path)
    finally:
        # 仅清理悬挂临时文件；replace 成功后该路径已不存在，unlink 会 ENOENT。
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
