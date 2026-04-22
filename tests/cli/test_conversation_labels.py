"""CLI label conversation registry 测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dayu.cli.conversation_labels import (
    CLI_CONVERSATION_SOURCE,
    FileConversationLabelRegistry,
    generate_cli_conversation_session_id,
    validate_conversation_label,
)
from dayu.cli.conversation_label_locks import ConversationLabelLease
from dayu.workspace_paths import build_cli_conversation_label_record_path

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("raw_label", "expected_label"),
    [
        ("alpha", "alpha"),
        ("A-1", "A-1"),
        ("9_report", "9_report"),
        ("  cli-label_1  ", "cli-label_1"),
    ],
)
def test_validate_conversation_label_accepts_valid_inputs(raw_label: str, expected_label: str) -> None:
    """合法 label 应被接受并返回规范化结果。"""

    assert validate_conversation_label(raw_label) == expected_label


@pytest.mark.parametrize(
    "raw_label",
    [
        "",
        "   ",
        "_alpha",
        "-alpha",
        "alpha space",
        "alpha.dot",
        "alpha/1",
    ],
)
def test_validate_conversation_label_rejects_invalid_inputs(raw_label: str) -> None:
    """非法 label 应稳定抛出 ValueError。"""

    with pytest.raises(ValueError, match="label"):
        validate_conversation_label(raw_label)


def test_generate_cli_conversation_session_id_returns_prefixed_unique_value() -> None:
    """每次生成都应返回新的 CLI session_id。"""

    first = generate_cli_conversation_session_id()
    second = generate_cli_conversation_session_id()

    assert first != second
    assert first.startswith("cli_conv_")
    assert second.startswith("cli_conv_")


def test_registry_get_record_returns_none_when_missing(tmp_path: Path) -> None:
    """不存在的 label 应返回 None。"""

    registry = FileConversationLabelRegistry(tmp_path)

    assert registry.get_record("missing") is None


def test_registry_get_or_create_creates_record_and_get_record_reuses_it(tmp_path: Path) -> None:
    """首次创建后应可重复读取，并保持同一条 record 的 session_id。"""

    registry = FileConversationLabelRegistry(tmp_path)

    created = registry.get_or_create_record(label="apple-1", scene_name="prompt_mt")
    loaded = registry.get_record("apple-1")

    assert loaded == created.record
    assert created.created is True
    assert created.record.label == "apple-1"
    assert created.record.session_id.startswith("cli_conv_")
    assert created.record.source == CLI_CONVERSATION_SOURCE
    assert created.record.scene_name == "prompt_mt"
    assert created.record.created_at == created.record.updated_at
    assert created.record.created_at.endswith("Z")


def test_registry_get_or_create_returns_existing_record_without_overwriting_scene(tmp_path: Path) -> None:
    """已存在 record 时应直接复用，不覆盖既有 scene。"""

    registry = FileConversationLabelRegistry(tmp_path)

    first = registry.get_or_create_record(label="apple", scene_name="interactive")
    second = registry.get_or_create_record(label="apple", scene_name="prompt_mt")

    assert first.created is True
    assert second.created is False
    assert second.record == first.record
    assert second.record.scene_name == "interactive"


def test_registry_list_records_returns_label_sorted_result(tmp_path: Path) -> None:
    """list_records 应按 label 升序返回。"""

    registry = FileConversationLabelRegistry(tmp_path)
    registry.get_or_create_record(label="charlie", scene_name="interactive")
    registry.get_or_create_record(label="alpha", scene_name="prompt_mt")
    registry.get_or_create_record(label="bravo", scene_name="interactive")

    records = registry.list_records()

    assert tuple(record.label for record in records) == ("alpha", "bravo", "charlie")


def test_registry_delete_record_removes_existing_label(tmp_path: Path) -> None:
    """delete_record 命中现有 label 时应删除对应文件。"""

    registry = FileConversationLabelRegistry(tmp_path)
    registry.get_or_create_record(label="alpha", scene_name="interactive")

    deleted = registry.delete_record("alpha")

    assert deleted is True
    assert registry.get_record("alpha") is None


def test_registry_recreate_same_label_allocates_new_session_id(tmp_path: Path) -> None:
    """删除后重建同名 label 时应分配新的 session_id。"""

    registry = FileConversationLabelRegistry(tmp_path)
    first = registry.get_or_create_record(label="alpha", scene_name="interactive").record

    assert registry.delete_record("alpha") is True

    second = registry.get_or_create_record(label="alpha", scene_name="interactive").record

    assert first.session_id != second.session_id
    assert second.session_id.startswith("cli_conv_")


def test_registry_delete_record_returns_false_when_label_is_absent(tmp_path: Path) -> None:
    """delete_record 命中缺失 label 时应返回 False。"""

    registry = FileConversationLabelRegistry(tmp_path)

    deleted = registry.delete_record("alpha")

    assert deleted is False


def test_registry_rejects_invalid_label_when_loading(tmp_path: Path) -> None:
    """非法 label 读取应稳定抛出 ValueError。"""

    registry = FileConversationLabelRegistry(tmp_path)

    with pytest.raises(ValueError, match="label"):
        registry.get_record("_bad")


def test_registry_rejects_blank_scene_name_when_creating(tmp_path: Path) -> None:
    """创建 record 时 scene_name 不能为空。"""

    registry = FileConversationLabelRegistry(tmp_path)

    with pytest.raises(ValueError, match="scene_name"):
        registry.get_or_create_record(label="alpha", scene_name="   ")


def test_registry_rejects_corrupted_record_on_get(tmp_path: Path) -> None:
    """损坏 record 在 get 时应稳定抛出 ValueError。"""

    record_path = build_cli_conversation_label_record_path(tmp_path, "alpha")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {
                "label": "beta",
                "session_id": generate_cli_conversation_session_id(),
                "source": CLI_CONVERSATION_SOURCE,
                "scene_name": "interactive",
                "created_at": "2026-04-22T00:00:00Z",
                "updated_at": "2026-04-22T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    registry = FileConversationLabelRegistry(tmp_path)

    with pytest.raises(ValueError, match="label 与文件名不一致"):
        registry.get_record("alpha")


def test_registry_rejects_corrupted_record_on_list(tmp_path: Path) -> None:
    """损坏 record 在 list 时应稳定抛出 ValueError。"""

    record_path = build_cli_conversation_label_record_path(tmp_path, "alpha")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text("[]", encoding="utf-8")

    registry = FileConversationLabelRegistry(tmp_path)

    with pytest.raises(ValueError, match="JSON 对象"):
        registry.list_records()


def test_conversation_label_lease_rejects_second_acquire_until_release(tmp_path: Path) -> None:
    """同一个 label 的第二次获取应被拒绝，释放后可再次获取。"""

    first = ConversationLabelLease(tmp_path, "alpha")
    second = ConversationLabelLease(tmp_path, "alpha")

    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="label 正在使用中: alpha"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()
