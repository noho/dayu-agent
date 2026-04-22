"""CLI label conversation registry。

本模块实现 CLI 层自有的 labeled conversation registry 持久化能力，
用于把用户可读 label 绑定到具体的 CLI session 标识与 scene 信息。

该模块只服务于 CLI，不向 Service、Host、Agent 暴露依赖，也不参与
底层 session schema 设计。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dayu.workspace_paths import (
    build_cli_conversation_label_record_path,
    build_cli_conversation_registry_dir,
)


CLI_CONVERSATION_SOURCE: Literal["cli"] = "cli"
CLI_CONVERSATION_SESSION_ID_PREFIX = "cli_conv_"
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
RECORD_FILE_SUFFIX = ".json"


def validate_conversation_label(label: str) -> str:
    """校验并规范化 CLI conversation label。

    Args:
        label: 待校验的 conversation label。

    Returns:
        去除首尾空白后的合法 label。

    Raises:
        ValueError: 当 label 为空或不满足命名规则时抛出。
    """

    normalized_label = str(label or "").strip()
    if not normalized_label:
        raise ValueError("label 不能为空")
    if LABEL_PATTERN.fullmatch(normalized_label) is None:
        raise ValueError("label 必须以字母或数字开头，且只能包含字母、数字、_、-")
    return normalized_label


def generate_cli_conversation_session_id() -> str:
    """生成新的 CLI conversation session_id。

    Args:
        无。

    Returns:
        新生成的 CLI conversation session_id。

    Raises:
        无。
    """

    return f"{CLI_CONVERSATION_SESSION_ID_PREFIX}{uuid.uuid4().hex}"


def _validate_conversation_session_id(session_id: str) -> str:
    """校验 CLI conversation session_id。

    Args:
        session_id: 待校验的 session_id。

    Returns:
        去除首尾空白后的合法 session_id。

    Raises:
        ValueError: 当 session_id 为空或不满足前缀约束时抛出。
    """

    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        raise ValueError("record.session_id 不能为空")
    if not normalized_session_id.startswith(CLI_CONVERSATION_SESSION_ID_PREFIX):
        raise ValueError("record.session_id 必须以 cli_conv_ 开头")
    return normalized_session_id


def build_registry_timestamp() -> str:
    """生成 registry record 使用的 UTC 时间戳。

    Args:
        无。

    Returns:
        `YYYY-MM-DDTHH:MM:SSZ` 格式的 UTC 时间戳。

    Raises:
        无。
    """

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ConversationLabelRecord:
    """CLI label conversation registry record。"""

    label: str
    session_id: str
    source: Literal["cli"]
    scene_name: str
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        """校验 record 的基础字段约束。

        Args:
            无。

        Returns:
            无。

        Raises:
            ValueError: 当任一字段不满足约束时抛出。
        """

        validate_conversation_label(self.label)
        _validate_conversation_session_id(self.session_id)
        if self.source != CLI_CONVERSATION_SOURCE:
            raise ValueError("record.source 必须为 cli")
        if not str(self.scene_name or "").strip():
            raise ValueError("record.scene_name 不能为空")
        if not str(self.created_at or "").strip():
            raise ValueError("record.created_at 不能为空")
        if not str(self.updated_at or "").strip():
            raise ValueError("record.updated_at 不能为空")

    def to_dict(self) -> dict[str, str]:
        """把 record 转为 JSON 可写入字典。

        Args:
            无。

        Returns:
            只包含字符串字段的字典。

        Raises:
            无。
        """

        return {
            "label": self.label,
            "session_id": self.session_id,
            "source": self.source,
            "scene_name": self.scene_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ConversationLabelResolution:
    """CLI label registry 解析结果。"""

    record: ConversationLabelRecord
    created: bool


class FileConversationLabelRegistry:
    """基于工作区文件系统的 CLI conversation label registry。"""

    def __init__(self, workspace_root: Path) -> None:
        """初始化 registry。

        Args:
            workspace_root: 工作区根目录。

        Returns:
            无。

        Raises:
            无。
        """

        self._workspace_root = Path(workspace_root).expanduser().resolve()
        self._registry_dir = build_cli_conversation_registry_dir(self._workspace_root)

    @property
    def registry_dir(self) -> Path:
        """返回 registry 目录路径。

        Args:
            无。

        Returns:
            registry 目录路径。

        Raises:
            无。
        """

        return self._registry_dir

    def get_record(self, label: str) -> ConversationLabelRecord | None:
        """读取指定 label 的 registry record。

        Args:
            label: 待读取的 conversation label。

        Returns:
            找到时返回对应 record；不存在时返回 `None`。

        Raises:
            ValueError: 当 label 非法或 record 文件损坏时抛出。
        """

        normalized_label = validate_conversation_label(label)
        record_path = build_cli_conversation_label_record_path(self._workspace_root, normalized_label)
        if not record_path.exists():
            return None
        return _load_record_from_path(record_path, expected_label=normalized_label)

    def delete_record(self, label: str) -> bool:
        """删除指定 label 的 registry record。

        Args:
            label: 待删除的 conversation label。

        Returns:
            实际删除了文件时返回 ``True``；record 不存在时返回 ``False``。

        Raises:
            ValueError: 当 label 非法时抛出。
            OSError: 当删除文件失败时抛出。
        """

        normalized_label = validate_conversation_label(label)
        record_path = build_cli_conversation_label_record_path(self._workspace_root, normalized_label)
        if not record_path.exists():
            return False
        record_path.unlink()
        return True

    def get_or_create_record(self, *, label: str, scene_name: str) -> ConversationLabelResolution:
        """读取现有 record，若不存在则首次创建。

        Args:
            label: conversation label。
            scene_name: 调用方显式传入的 scene 名称。

        Returns:
            已存在或新创建的 registry 解析结果。

        Raises:
            ValueError: 当 label 非法、scene_name 为空或 record 文件损坏时抛出。
        """

        normalized_label = validate_conversation_label(label)
        existing_record = self.get_record(normalized_label)
        if existing_record is not None:
            return ConversationLabelResolution(record=existing_record, created=False)
        normalized_scene_name = _normalize_scene_name(scene_name)
        timestamp = build_registry_timestamp()
        created_record = ConversationLabelRecord(
            label=normalized_label,
            session_id=generate_cli_conversation_session_id(),
            source=CLI_CONVERSATION_SOURCE,
            scene_name=normalized_scene_name,
            created_at=timestamp,
            updated_at=timestamp,
        )
        self._write_record(created_record)
        return ConversationLabelResolution(record=created_record, created=True)

    def list_records(self) -> tuple[ConversationLabelRecord, ...]:
        """按 label 升序列出全部 CLI conversation records。

        Args:
            无。

        Returns:
            按 label 升序排列的 record 元组。

        Raises:
            ValueError: 当某个 record 文件损坏时抛出。
        """

        if not self._registry_dir.is_dir():
            return ()
        record_paths = sorted(
            path for path in self._registry_dir.iterdir() if path.is_file() and path.suffix == RECORD_FILE_SUFFIX
        )
        records = [
            _load_record_from_path(
                record_path,
                expected_label=_extract_label_from_record_path(record_path),
            )
            for record_path in record_paths
        ]
        return tuple(sorted(records, key=lambda record: record.label))

    def _write_record(self, record: ConversationLabelRecord) -> None:
        """以原子替换方式写入 registry record。

        Args:
            record: 待写入的 registry record。

        Returns:
            无。

        Raises:
            OSError: 当写入或替换失败时抛出。
        """

        record_path = build_cli_conversation_label_record_path(self._workspace_root, record.label)
        self._registry_dir.mkdir(parents=True, exist_ok=True)
        serialized_record = json.dumps(record.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        _write_text_atomic(record_path, serialized_record)


def _normalize_scene_name(scene_name: str) -> str:
    """校验并规范化 scene_name。

    Args:
        scene_name: 调用方传入的 scene 名称。

    Returns:
        去除首尾空白后的 scene_name。

    Raises:
        ValueError: 当 scene_name 为空时抛出。
    """

    normalized_scene_name = str(scene_name or "").strip()
    if not normalized_scene_name:
        raise ValueError("scene_name 不能为空")
    return normalized_scene_name


def _extract_label_from_record_path(record_path: Path) -> str:
    """从 record 文件路径提取 label。

    Args:
        record_path: record 文件路径。

    Returns:
        文件名中去掉 `.json` 后的 label。

    Raises:
        ValueError: 当文件名无法表示合法 label 时抛出。
    """

    file_name = record_path.name
    if not file_name.endswith(RECORD_FILE_SUFFIX):
        raise ValueError(f"非法 CLI conversation record 文件名: {file_name}")
    label = file_name[: -len(RECORD_FILE_SUFFIX)]
    return validate_conversation_label(label)


def _load_record_from_path(record_path: Path, *, expected_label: str) -> ConversationLabelRecord:
    """从指定文件路径加载并校验 registry record。

    Args:
        record_path: record 文件路径。
        expected_label: 调用方根据文件名推导出的期望 label。

    Returns:
        校验通过的 registry record。

    Raises:
        ValueError: 当文件内容不是合法 record 时抛出。
    """

    normalized_expected_label = validate_conversation_label(expected_label)
    raw_payload: object = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError(f"CLI conversation record 损坏: {record_path.name} 顶层必须是 JSON 对象")
    label = _read_required_string_field(raw_payload, "label", record_path)
    session_id = _read_required_string_field(raw_payload, "session_id", record_path)
    source = _read_required_string_field(raw_payload, "source", record_path)
    scene_name = _read_required_string_field(raw_payload, "scene_name", record_path)
    created_at = _read_required_string_field(raw_payload, "created_at", record_path)
    updated_at = _read_required_string_field(raw_payload, "updated_at", record_path)
    if label != normalized_expected_label:
        raise ValueError(f"CLI conversation record 损坏: {record_path.name} 的 label 与文件名不一致")
    if source != CLI_CONVERSATION_SOURCE:
        raise ValueError(f"CLI conversation record 损坏: {record_path.name} 的 source 必须为 cli")
    return ConversationLabelRecord(
        label=label,
        session_id=session_id,
        source=CLI_CONVERSATION_SOURCE,
        scene_name=scene_name,
        created_at=created_at,
        updated_at=updated_at,
    )


def _read_required_string_field(raw_payload: dict[object, object], field_name: str, record_path: Path) -> str:
    """读取 record 中的必填字符串字段。

    Args:
        raw_payload: 反序列化后的 JSON 对象。
        field_name: 字段名。
        record_path: 当前加载的 record 文件路径。

    Returns:
        去除首尾空白后的字段值。

    Raises:
        ValueError: 当字段缺失、不是字符串或为空时抛出。
    """

    raw_value = raw_payload.get(field_name)
    if not isinstance(raw_value, str):
        raise ValueError(f"CLI conversation record 损坏: {record_path.name} 缺少字段 {field_name}")
    normalized_value = raw_value.strip()
    if not normalized_value:
        raise ValueError(f"CLI conversation record 损坏: {record_path.name} 的字段 {field_name} 不能为空")
    return normalized_value


def _write_text_atomic(target_path: Path, content: str) -> None:
    """以原子替换方式写入 UTF-8 文本文件。

    Args:
        target_path: 目标文件路径。
        content: 待写入文本。

    Returns:
        无。

    Raises:
        OSError: 当写入或替换失败时抛出。
    """

    target_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_path_str = tempfile.mkstemp(
        prefix=f".{target_path.name}.",
        suffix=".tmp",
        dir=target_path.parent,
    )
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise
