# -*- coding: utf-8 -*-
"""AsyncCliRunner 辅助方法测试"""

from pathlib import Path
from typing import cast

import pytest

from dayu.contracts.agent_types import AgentMessage
from dayu.contracts.protocols import ToolExecutor
from dayu.engine.async_cli_runner import AsyncCliRunner
from dayu.engine.events import EventType
from dayu.engine.events import StreamEvent


class _NoopExecutor:
    """满足 ToolExecutor 协议的最小测试桩。"""

    def execute(self, name: str, arguments: dict[str, object], context=None) -> dict[str, object]:
        """测试桩不实际执行工具。"""

        del name, arguments, context
        return {}

    def get_schemas(self) -> list[dict[str, object]]:
        """返回空 schema 列表。"""

        return []

    def clear_cursors(self) -> None:
        """测试桩不维护游标。"""

    def get_dup_call_spec(self, name: str):
        """测试桩默认无重复调用策略。"""

        del name
        return None

    def get_execution_context_param_name(self, name: str) -> str | None:
        """测试桩默认不注入 execution context。"""

        del name
        return None

    def get_tool_display_info(self, name: str) -> tuple[str, list[str] | None]:
        """返回默认展示信息。"""

        return name, None

    def register_response_middleware(self, callback) -> None:
        """测试桩忽略 response middleware。"""

        del callback


def _messages(items: list[AgentMessage]) -> list[AgentMessage]:
    """把测试消息显式收窄为 AgentMessage 列表。"""

    return items


def _require_event(event: StreamEvent | None) -> StreamEvent:
    """断言解析结果非空并返回事件。"""

    assert event is not None
    return event


def test_build_command_overrides(tmp_path):
    runner = AsyncCliRunner(
        command=["codex", "exec"],
        working_dir=tmp_path,
        model="gpt-4",
        full_auto=False,
        reasoning_effort="medium",
    )

    cmd = runner._build_command(full_auto=True, reasoning_effort="high")

    assert "--json" in cmd
    assert "--color" in cmd and "never" in cmd
    assert "--model" in cmd and "gpt-4" in cmd
    assert "--full-auto" in cmd
    assert "model_reasoning_effort=\"high\"" in " ".join(cmd)
    assert cmd[-1] == "-"


def test_format_messages_writes_agents_md(tmp_path):
    runner = AsyncCliRunner(command=["codex", "exec"], working_dir=tmp_path)

    prompt, ok = runner._format_messages(_messages([
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]))

    agents_path = Path(tmp_path) / "AGENTS.md"
    assert ok is True
    assert agents_path.exists()
    assert agents_path.read_text(encoding="utf-8") == "SYSTEM"
    assert "hello" in prompt
    assert "ASSISTANT: hi" in prompt


def test_parse_json_event_variants(tmp_path):
    runner = AsyncCliRunner(command=["codex", "exec"], working_dir=tmp_path)

    event = _require_event(
        runner._parse_json_event('{"type":"item.completed","item":{"type":"agent_message","text":"hi"}}')
    )
    assert event.type == EventType.CONTENT_DELTA
    assert event.data == "hi"

    done = _require_event(runner._parse_json_event('{"type":"turn.completed","usage":{"prompt_tokens":1}}'))
    assert done.type == EventType.DONE

    err = _require_event(runner._parse_json_event('{"type":"error","message":"boom"}'))
    assert err.type == EventType.ERROR

    invalid = _require_event(runner._parse_json_event("{invalid json"))
    assert invalid.type == EventType.ERROR


def test_annotate_event(tmp_path):
    runner = AsyncCliRunner(command=["codex", "exec"], working_dir=tmp_path)
    event = runner._annotate_event(
        _require_event(runner._parse_json_event('{"type":"turn.completed","usage":{}}')),
        {"run_id": "r", "iteration_id": "t", "request_id": "q"},
    )
    assert event.metadata["run_id"] == "r"
    assert event.metadata["iteration_id"] == "t"
    assert event.metadata["request_id"] == "q"


def test_set_tools_logs_warning(caplog: pytest.LogCaptureFixture):
    """测试 set_tools 会记录警告日志。"""

    import logging

    caplog.set_level(logging.WARNING)

    runner = AsyncCliRunner(command=["test"])
    runner.set_tools(cast(ToolExecutor, _NoopExecutor()))

    assert any("不支持工具调用机制" in record.message for record in caplog.records)

def test_format_messages_agents_md_write_failure(tmp_path):
    """测试 AGENTS.md 写入失败时仍返回格式化消息。"""

    from unittest.mock import patch

    runner = AsyncCliRunner(command=["echo"], working_dir=tmp_path)
    messages = _messages([
        {"role": "system", "content": "You are a helper"},
        {"role": "user", "content": "hello"},
    ])

    with patch.object(Path, "write_text", side_effect=PermissionError("Access denied")):
        result, ok = runner._format_messages(messages)

    assert ok is False
    assert result == "hello"
