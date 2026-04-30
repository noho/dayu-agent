"""结构性回归保险：禁止历史展示字段污染运行态真源。

`#118` 把 `assistant_reasoning` 作为展示侧字段独立放在 `ConversationHistoryArchive`
里，不能扩散到 `ConversationTurnRecord` / pending turn / prepared turn /
agent execution 序列化等运行态契约模块。本文件以源码反射方式硬性守住该边界。
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from dayu.host.conversation_store import ConversationTurnRecord

_FORBIDDEN_TOKENS = (
    "assistant_reasoning",
    "ConversationHistoryTurnRecord",
    "ConversationHistoryArchive",
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_GUARDED_FILES = (
    "dayu/host/conversation_memory.py",
    "dayu/host/pending_turn_store.py",
    "dayu/host/prepared_turn.py",
    "dayu/contracts/agent_execution.py",
    "dayu/contracts/agent_execution_serialization.py",
    "dayu/contracts/agent_types.py",
)


@pytest.mark.unit
@pytest.mark.parametrize("relative_path", _GUARDED_FILES)
def test_runtime_modules_must_not_reference_history_archive_tokens(
    relative_path: str,
) -> None:
    """运行态契约/记忆/恢复模块不得出现历史展示字段相关 token。"""

    source = (_PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    for token in _FORBIDDEN_TOKENS:
        assert token not in source, (
            f"{relative_path} 不允许出现历史展示字段 token: {token}"
        )


@pytest.mark.unit
def test_conversation_turn_record_field_set_excludes_assistant_reasoning() -> None:
    """`ConversationTurnRecord` 字段集严禁含 `assistant_reasoning`。"""

    field_names = {f.name for f in fields(ConversationTurnRecord)}
    assert "assistant_reasoning" not in field_names
