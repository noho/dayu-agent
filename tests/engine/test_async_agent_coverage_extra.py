"""AsyncAgent 补充覆盖测试。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator, Optional

import pytest

from dayu.contracts.agent_types import AgentMessage
from dayu.contracts.protocols import ToolExecutionContext
from dayu.engine.async_agent import AgentRunningConfig, AsyncAgent
from dayu.engine.duplicate_call_guard import _make_tool_signature
from dayu.engine.events import (
    EventType,
    StreamEvent,
    content_complete,
    done_event,
    error_event,
    tool_call_delta,
    tool_call_result,
    tool_call_start,
    tool_calls_batch_done,
)


class _RunnerStub:
    """Runner 桩。"""

    def __init__(self, batches: list[list[StreamEvent]]) -> None:
        self._batches = list(batches)
        self.calls: list[dict[str, Any]] = []

    def is_supports_tool_calling(self) -> bool:
        return True

    def set_tools(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def close(self) -> None:
        return None

    async def call(
        self,
        messages: list[AgentMessage],
        *,
        stream: bool = True,
        **extra_payloads: Any,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append({"messages": messages, "stream": stream, "extra_payloads": extra_payloads})
        batch = self._batches.pop(0)
        for event in batch:
            yield event


class _ToolExecutorStub:
    """工具执行器桩。"""

    def __init__(self) -> None:
        """初始化工具执行器桩。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.clear_count = 0

    def get_schemas(self) -> list[dict[str, Any]]:
        """返回空工具 schema 列表。

        Args:
            无。

        Returns:
            空列表。

        Raises:
            无。
        """

        return []

    def get_tool_guidance(self) -> str:
        """返回空工具指导。

        Args:
            无。

        Returns:
            空字符串。

        Raises:
            无。
        """

        return ""

    def clear_cursors(self) -> None:
        """记录游标清理次数。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.clear_count += 1

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """返回最小成功结果。

        Args:
            name: 工具名称。
            arguments: 工具参数。
            context: 工具执行上下文。

        Returns:
            空 value 的成功信封。

        Raises:
            无。
        """

        _ = (name, arguments, context)
        return {"ok": True, "value": {}}

    def get_dup_call_spec(self, name: str) -> None:
        """返回空重复调用策略。

        Args:
            name: 工具名称。

        Returns:
            `None`。

        Raises:
            无。
        """

        _ = name
        return None

    def get_execution_context_param_name(self, name: str) -> None:
        """返回空 execution context 参数名。

        Args:
            name: 工具名称。

        Returns:
            `None`。

        Raises:
            无。
        """

        _ = name
        return None

    def get_tool_display_info(self, name: str) -> tuple[str, list[str] | None]:
        """返回默认展示元数据。

        Args:
            name: 工具名称。

        Returns:
            ``(name, None)`` 二元组。

        Raises:
            无。
        """

        return (name, None)

    def register_response_middleware(self, callback: Any) -> None:
        """忽略 response middleware 注册。

        Args:
            callback: middleware 回调。

        Returns:
            无。

        Raises:
            无。
        """

        _ = callback


def _messages(*contents: str) -> list[AgentMessage]:
    """构造强类型消息列表。

    Args:
        *contents: 用户消息正文。

    Returns:
        强类型消息列表。

    Raises:
        无。
    """

    return [{"role": "user", "content": content} for content in contents]


@pytest.mark.asyncio
async def test_run_loop_non_stream_and_unknown_event_branch() -> None:
    """覆盖非流式日志分支、tool start/delta 与未知事件分支。"""

    runner = _RunnerStub(
        [[tool_call_start("tool", "call_1"), tool_call_delta("call_1", "tool", "{}"), StreamEvent(type=EventType.METADATA, data={}), content_complete("ok"), done_event()]]
    )
    agent = AsyncAgent(runner)

    events = []
    async for event in agent.run("prompt", stream=False):
        events.append(event)

    types = [event.type for event in events]
    assert EventType.TOOL_CALL_START in types
    assert EventType.TOOL_CALL_DELTA in types
    assert EventType.FINAL_ANSWER in types


@pytest.mark.asyncio
async def test_run_loop_batch_done_without_calls_emits_error() -> None:
    """覆盖 TOOL_CALLS_BATCH_DONE 但无调用数据的错误分支。"""

    runner = _RunnerStub([[tool_calls_batch_done([], ok=0, error=0, timeout=0, cancelled=0)]])
    agent = AsyncAgent(runner)

    events = []
    async for event in agent.run("prompt"):
        events.append(event)

    assert any(e.type == EventType.ERROR for e in events)


@pytest.mark.asyncio
async def test_run_loop_tool_calls_without_batch_done_emits_error() -> None:
    """覆盖已收到 tool_call_result 但缺少 batch_done 的错误路径。"""

    runner = _RunnerStub(
        [[tool_call_result("call_1", {"ok": True, "value": {}}, name="tool", arguments={}, index_in_iteration=0), content_complete(""), done_event()]]
    )
    agent = AsyncAgent(runner)

    events = []
    async for event in agent.run("prompt"):
        events.append(event)

    assert any(e.type == EventType.ERROR for e in events)


@pytest.mark.asyncio
async def test_run_and_wait_collects_error_entries() -> None:
    """覆盖 run_and_wait 中 ERROR 事件收集分支。"""

    runner = _RunnerStub([[error_event("fatal", recoverable=False)]])
    agent = AsyncAgent(runner)

    result = await agent.run_and_wait("prompt")
    assert result.errors


@pytest.mark.asyncio
async def test_run_and_wait_clears_tool_executor_cursors_before_each_run() -> None:
    """覆盖 run_and_wait 在每轮开始前清理工具游标。"""

    runner = _RunnerStub(
        [
            [content_complete("first"), done_event()],
            [content_complete("second"), done_event()],
        ]
    )
    tool_executor = _ToolExecutorStub()
    agent = AsyncAgent(runner, tool_executor=tool_executor)

    first = await agent.run_and_wait("prompt-1")
    second = await agent.run_and_wait("prompt-2")

    assert first.content == "first"
    assert second.content == "second"
    assert tool_executor.clear_count == 2


@pytest.mark.asyncio
async def test_run_messages_accepts_session_id_without_duplicate_kwargs() -> None:
    """覆盖 run_messages 透传 session_id 时不会重复传参。"""

    runner = _RunnerStub([[content_complete("ok"), done_event()]])
    agent = AsyncAgent(runner)

    events = []
    async for event in agent.run_messages(
        _messages("hello"),
        session_id="sess_manual",
    ):
        events.append(event)

    assert events
    assert runner.calls[0]["extra_payloads"].get("session_id") is None
    assert runner.calls[0]["extra_payloads"]["trace_context"]["run_id"].startswith("run_")


@pytest.mark.asyncio
async def test_run_accepts_explicit_session_id() -> None:
    """覆盖 run 显式传入 session_id 的路径。"""

    runner = _RunnerStub([[content_complete("ok"), done_event()]])
    agent = AsyncAgent(runner)

    events = []
    async for event in agent.run("hello", session_id="sess_run"):
        events.append(event)

    assert events
    assert runner.calls[0]["extra_payloads"].get("session_id") is None


@pytest.mark.asyncio
async def test_run_and_wait_accepts_explicit_session_id() -> None:
    """覆盖 run_and_wait 显式传入 session_id 的路径。"""

    runner = _RunnerStub([[content_complete("ok"), done_event()]])
    agent = AsyncAgent(runner)

    result = await agent.run_and_wait("hello", session_id="sess_wait")

    assert result.content == "ok"
    assert runner.calls[0]["extra_payloads"].get("session_id") is None


@pytest.mark.asyncio
async def test_run_messages_rejects_reserved_extra_payload_keys() -> None:
    """覆盖 run_messages 拒绝内部保留透传字段的路径。"""

    runner = _RunnerStub([[content_complete("ok"), done_event()]])
    agent = AsyncAgent(runner)

    with pytest.raises(ValueError, match="内部保留字段"):
        async for _ in agent.run_messages(
            _messages("hello"),
            trace_context={"run_id": "fake", "iteration_id": "fake_iteration"},
        ):
            pass


@pytest.mark.asyncio
async def test_run_rejects_trace_context_in_extra_payloads() -> None:
    """覆盖 run 拒绝通过 extra_payloads 注入 trace_context 的路径。"""

    runner = _RunnerStub([[content_complete("ok"), done_event()]])
    agent = AsyncAgent(runner)

    with pytest.raises(ValueError, match="trace_context"):
        async for _ in agent.run("hello", trace_context={"run_id": "bad", "iteration_id": "bad_iteration"}):
            pass


def test_make_tool_signature_non_serializable_fallback() -> None:
    """覆盖 _make_tool_signature JSON 序列化失败分支。"""

    class _Bad:
        def __str__(self) -> str:
            return "bad"

    signature = _make_tool_signature("tool", {"x": _Bad()})
    assert signature.startswith("tool:")


def test_init_preserves_explicit_running_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖 AsyncAgent 显式 running_config 构造分支。"""

    monkeypatch.setattr("dayu.engine.async_agent.Log.verbose", lambda *args, **kwargs: None)

    class _CreatedRunner:
        def is_supports_tool_calling(self) -> bool:
            return False

        def set_tools(self, executor: object | None) -> None:
            """设置工具执行器。

            Args:
                executor: 工具执行器。

            Returns:
                无。

            Raises:
                无。
            """

            _ = executor

        async def close(self) -> None:
            """关闭 Runner（桩实现，无操作）。"""
            return None

        async def call(
            self,
            messages: list[AgentMessage],
            *,
            stream: bool = True,
            **extra_payloads: Any,
        ) -> AsyncIterator[StreamEvent]:
            """返回空事件流。

            Args:
                messages: 消息列表。
                stream: 是否流式。
                **extra_payloads: 额外参数。

            Yields:
                不产出任何事件。

            Raises:
                无。
            """

            _ = (messages, stream, extra_payloads)
            if False:
                yield done_event()

    agent = AsyncAgent(
        runner=_CreatedRunner(),
        running_config=AgentRunningConfig(max_iterations=7),
    )
    assert isinstance(agent.runner, _CreatedRunner)
    assert agent.running_config.max_iterations == 7
