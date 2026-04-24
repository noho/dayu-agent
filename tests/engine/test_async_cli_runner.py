"""
测试 AsyncCliRunner
"""
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from dayu.contracts.agent_types import AgentMessage
from dayu.contracts.protocols import ToolExecutionContext
from dayu.engine.async_cli_runner import AsyncCliRunner, create_codex_runner
from dayu.engine import EventType, content_complete, content_delta, done_event


class FakeStdout:
    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]

    async def readline(self):
        if not self._lines:
            return b""
        return self._lines.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)


class FakeStdin:
    def __init__(self):
        self.buffer = b""

    def write(self, data):
        self.buffer += data

    async def drain(self):
        return None

    def close(self):
        return None


class FakeStderr:
    def __init__(self, payload=b""):
        self._payload = payload

    async def read(self):
        return self._payload


class FakeProcess:
    def __init__(self, lines, returncode=0, stderr=b""):
        self.stdin = FakeStdin()
        self.stdout = FakeStdout(lines)
        self.stderr = FakeStderr(stderr)
        self.returncode = returncode
        self.killed = False

    async def wait(self):
        return self.returncode

    def kill(self):
        self.killed = True


class _NoopExecutor:
    """满足 ToolExecutor 协议的最小测试桩。"""

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """测试桩不实际执行工具。"""

        del name, arguments, context
        return {}

    def get_schemas(self) -> list[dict[str, Any]]:
        """返回空 schema 列表。"""

        return []

    def clear_cursors(self) -> None:
        """测试桩不维护截断游标。"""

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


class TestAsyncCliRunnerInit:
    """测试 AsyncCliRunner 初始化"""

    def test_init_minimal(self):
        """测试最小参数初始化"""
        runner = AsyncCliRunner(command=["echo", "test"])

        assert runner.command == ["echo", "test"]
        assert runner.timeout == 3600
        assert runner.working_dir == Path.cwd()

    def test_init_full(self):
        """测试完整参数初始化"""
        runner = AsyncCliRunner(
            command=["codex", "exec"],
            working_dir=Path("/tmp"),
            env={"KEY": "value"},
            timeout=1800,
            model="gpt-4",
            full_auto=True,
            reasoning_effort="high",
        )

        assert runner.command == ["codex", "exec"]
        assert runner.working_dir == Path("/tmp")
        assert "KEY" in runner.env
        assert runner.env["KEY"] == "value"
        assert runner.timeout == 1800
        assert runner.model == "gpt-4"
        assert runner.full_auto is True
        assert runner.reasoning_effort == "high"

    def test_set_tools(self):
        """测试设置工具"""
        runner = AsyncCliRunner(command=["test"])

        # set_tools是no-op（CLI不需要工具）
        runner.set_tools(_NoopExecutor())
        # 验证不抛出异常即可


class TestAsyncCliRunnerBuildCommand:
    """测试命令构建"""

    def test_build_command_minimal(self):
        """测试最小命令构建"""
        runner = AsyncCliRunner(command=["codex", "exec"])
        cmd = runner._build_command()

        assert cmd == [
            "codex",
            "exec",
            "--json",
            "--color",
            "never",
            "--config",
            "model_reasoning_effort=\"medium\"",
            "-",
        ]

    def test_build_command_with_model(self):
        """测试带模型的命令构建"""
        runner = AsyncCliRunner(command=["codex", "exec"], model="gpt-4")
        cmd = runner._build_command()

        assert cmd == [
            "codex",
            "exec",
            "--json",
            "--color",
            "never",
            "--model",
            "gpt-4",
            "--config",
            "model_reasoning_effort=\"medium\"",
            "-",
        ]

    def test_build_command_with_full_auto_and_reasoning(self):
        """测试 full_auto 和 reasoning_effort 参数"""
        runner = AsyncCliRunner(command=["codex", "exec"])
        cmd = runner._build_command(full_auto=True, reasoning_effort="high")

        assert "--full-auto" in cmd
        assert "model_reasoning_effort=\"high\"" in cmd


class TestAsyncCliRunnerFormatMessages:
    """测试消息格式化"""

    def test_format_single_user_message(self):
        """测试单条用户消息"""
        runner = AsyncCliRunner(command=["test"])
        messages = _messages([{"role": "user", "content": "hello"}])

        result, ok = runner._format_messages(messages)
        assert ok is True

        # 用户消息不加前缀
        assert result == "hello"
        assert "[USER]" not in result

    def test_format_system_and_user_messages(self, tmp_path):
        """测试格式化系统和用户消息（system写入AGENTS.md）"""
        runner = AsyncCliRunner(command=["echo"], working_dir=tmp_path)
        messages = _messages([
            {"role": "system", "content": "You are a helper"},
            {"role": "user", "content": "hello"},
        ])

        result, ok = runner._format_messages(messages)
        assert ok is True

        agents_path = tmp_path / "AGENTS.md"
        assert agents_path.exists()
        assert agents_path.read_text(encoding="utf-8") == "You are a helper"
        # system role不进入最终prompt
        assert "You are a helper" not in result
        assert result == "hello"

    def test_format_conversation(self, tmp_path):
        """测试对话消息"""
        runner = AsyncCliRunner(command=["test"], working_dir=tmp_path)
        messages = _messages([
            {"role": "system", "content": "helper"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "how are you"},
        ])

        result, ok = runner._format_messages(messages)
        assert ok is True

        agents_path = tmp_path / "AGENTS.md"
        assert agents_path.exists()
        assert agents_path.read_text(encoding="utf-8") == "helper"
        # system role不进入最终prompt
        assert "helper" not in result
        # 用户消息不加前缀
        assert "hi" in result
        # assistant消息加ASSISTANT:前缀
        assert "ASSISTANT: hello" in result
        assert "how are you" in result


@pytest.mark.asyncio
class TestAsyncCliRunnerCall:
    """测试 CLI 调用"""

    async def test_call_success(self):
        """测试成功调用"""
        runner = AsyncCliRunner(command=["codex", "exec"])

        messages = _messages([{"role": "user", "content": "test"}])

        captured = {}

        async def fake_run_streaming(command, prompt_text):
            captured["command"] = command
            captured["prompt_text"] = prompt_text
            yield content_delta("ok")
            yield content_complete("ok")
            yield done_event(summary={"usage": {"total_tokens": 1}})

        with patch.object(runner, "_run_streaming", fake_run_streaming):
            events = []
            async for event in runner.call(
                messages,
                full_auto=True,
                reasoning_effort="high",
                trace_context={"run_id": "run_test", "iteration_id": "iteration_test"},
            ):
                events.append(event)

        # 应该有 CONTENT_DELTA, CONTENT_COMPLETE, DONE 事件
        event_types = [e.type for e in events]
        assert EventType.CONTENT_DELTA in event_types
        assert EventType.CONTENT_COMPLETE in event_types
        assert EventType.DONE in event_types
        assert "--full-auto" in captured["command"]
        assert "model_reasoning_effort=\"high\"" in captured["command"]
        for event in events:
            assert event.metadata.get("run_id") == "run_test"
            assert event.metadata.get("iteration_id") == "iteration_test"
            assert event.metadata.get("request_id", "").startswith("cli_")


class TestCreateCodexRunner:
    """测试 create_codex_runner 工厂函数"""

    def test_create_minimal(self):
        """测试最小参数创建"""
        runner = create_codex_runner()

        assert "codex" in runner.command
        assert "exec" in runner.command
        assert runner.supports_tool_calling is False

    def test_create_with_model(self):
        """测试带模型创建"""
        runner = create_codex_runner(model="gpt-4")

        assert "--model" in runner.command
        assert "gpt-4" in runner.command

    def test_create_with_reasoning_effort(self):
        """测试带推理强度创建"""
        runner = create_codex_runner(reasoning_effort="high")

        assert "--config" in runner.command
        # 检查是否有 reasoning_effort 配置
        config_found = any("reasoning_effort" in str(arg) for arg in runner.command)
        assert config_found

    def test_create_with_working_dir(self):
        """测试带工作目录创建"""
        working_dir = Path("/tmp/test")
        runner = create_codex_runner(working_dir=working_dir)

        assert runner.working_dir == working_dir


@pytest.mark.asyncio
class TestAsyncCliRunnerRunStreaming:
    """测试 _run_streaming 的事件顺序"""

    async def test_run_streaming_emits_content_complete_before_done(self):
        """测试 content_complete 出现在 done_event 之前"""
        runner = AsyncCliRunner(command=["codex", "exec"])
        lines = [
            '{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}\n',
            '{"type":"turn.completed","usage":{"total_tokens":3}}\n',
        ]
        fake_process = FakeProcess(lines)

        with patch("asyncio.create_subprocess_exec", return_value=fake_process):
            events = []
            async for event in runner._run_streaming(["codex", "exec"], "prompt"):
                events.append(event)

        assert [e.type for e in events] == [
            EventType.CONTENT_DELTA,
            EventType.CONTENT_COMPLETE,
            EventType.DONE,
        ]
        assert events[1].data == "Hello"

    async def test_run_streaming_nonzero_exit_code(self):
        """测试非零退出码返回 error_event"""
        runner = AsyncCliRunner(command=["codex", "exec"])
        fake_process = FakeProcess([], returncode=1, stderr=b"boom")

        with patch("asyncio.create_subprocess_exec", return_value=fake_process):
            events = []
            async for event in runner._run_streaming(["codex", "exec"], "prompt"):
                events.append(event)

        assert len(events) == 1
        assert events[0].type == EventType.ERROR

    async def test_run_streaming_missing_done_event_emits_synthetic_done_event(self):
        """测试缺少 turn.completed 时补发 done_event"""
        runner = AsyncCliRunner(command=["codex", "exec"])
        lines = [
            '{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}\n',
        ]
        fake_process = FakeProcess(lines)

        with patch("asyncio.create_subprocess_exec", return_value=fake_process):
            events = []
            async for event in runner._run_streaming(["codex", "exec"], "prompt"):
                events.append(event)

        assert [e.type for e in events] == [
            EventType.CONTENT_DELTA,
            EventType.CONTENT_COMPLETE,
            EventType.WARNING,
            EventType.DONE,
        ]
        assert events[-1].data.get("inferred") is True

    async def test_run_streaming_iteration_failed_stops_early(self):
        """测试 turn.failed 触发 error_event 后立即结束"""
        runner = AsyncCliRunner(command=["codex", "exec"])
        lines = [
            '{"type":"turn.failed","error":{"message":"boom"}}\n',
            '{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}\n',
            '{"type":"turn.completed","usage":{"total_tokens":3}}\n',
        ]
        fake_process = FakeProcess(lines)

        with patch("asyncio.create_subprocess_exec", return_value=fake_process):
            events = []
            async for event in runner._run_streaming(["codex", "exec"], "prompt"):
                events.append(event)

        assert [e.type for e in events] == [EventType.ERROR]

    async def test_run_streaming_json_parse_error_stops_early(self):
        """测试 JSON 解析失败触发 error_event 后立即结束"""
        runner = AsyncCliRunner(command=["codex", "exec"])
        lines = [
            "not-json\n",
            '{"type":"turn.completed","usage":{"total_tokens":3}}\n',
        ]
        fake_process = FakeProcess(lines)

        with patch("asyncio.create_subprocess_exec", return_value=fake_process):
            events = []
            async for event in runner._run_streaming(["codex", "exec"], "prompt"):
                events.append(event)

        assert [e.type for e in events] == [EventType.ERROR]
