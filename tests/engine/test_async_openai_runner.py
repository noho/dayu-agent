"""
测试 AsyncOpenAIRunner
"""
from __future__ import annotations

from typing import Any, Callable, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dayu.contracts.agent_types import AgentMessage
from dayu.contracts.protocols import ToolExecutionContext
from dayu.engine.async_openai_runner import AsyncOpenAIRunner
from dayu.engine import EventType
from dayu.engine.tool_contracts import DupCallSpec

# 检查 aiohttp 是否可用
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class DummyToolExecutor:
    """用于测试的最小工具执行器桩。"""

    def __init__(self, result="ok"):
        """初始化执行器桩。

        Args:
            result: 工具执行返回值。

        Returns:
            无。

        Raises:
            无。
        """

        self.result = result

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """执行测试工具调用。

        Args:
            name: 工具名称。
            arguments: 工具参数。
            context: 可选执行上下文。

        Returns:
            工具执行结果。

        Raises:
            无。
        """

        del name, arguments, context
        if isinstance(self.result, dict):
            return self.result
        return {"ok": True, "value": str(self.result)}

    def get_schemas(self) -> list[dict[str, Any]]:
        """返回空工具 schema 列表。"""

        return []

    def clear_cursors(self) -> None:
        """测试桩不维护 cursor，无需处理。"""

    def get_dup_call_spec(self, name: str) -> DupCallSpec | None:
        """返回空重复调用策略。"""

        del name
        return None

    def get_execution_context_param_name(self, name: str) -> str | None:
        """测试桩不声明上下文参数名。"""

        del name
        return None

    def get_tool_display_info(self, name: str) -> tuple[str, list[str] | None]:
        """返回默认展示信息（fallback 到原始名）。"""

        return name, None

    def register_response_middleware(
        self,
        callback: Callable[[str, dict[str, Any], ToolExecutionContext | None], dict[str, Any]],
    ) -> None:
        """测试桩不消费 response middleware。"""

        del callback


class FakeSSEContent:
    """最小 SSE content 桩。"""

    def __init__(self, chunks):
        """初始化 chunk 列表。"""

        self._chunks = [
            chunk if isinstance(chunk, bytes) else chunk.encode("utf-8")
            for chunk in chunks
        ]

    async def iter_chunked(self, size: int):
        """按 chunk 迭代 SSE 字节流。"""

        del size
        for chunk in self._chunks:
            yield chunk


class FakeSSEResponse:
    """最小 SSE 响应桩。"""

    def __init__(self, chunks):
        """初始化响应内容。"""

        self.content = FakeSSEContent(chunks)


def _fake_client_response(chunks: list[str | bytes]) -> Any:
    """构造满足 `_process_sse_stream` 调用边界的假响应。

    Args:
        chunks: SSE 原始 chunk 列表。

    Returns:
        以 `Any` 收窄后的假响应对象。

    Raises:
        无。
    """

    return cast(Any, FakeSSEResponse(chunks))


def _message_list(*messages: AgentMessage) -> list[AgentMessage]:
    """构造强类型 AgentMessage 列表。

    Args:
        *messages: 消息序列。

    Returns:
        强类型消息列表。

    Raises:
        无。
    """

    return list(messages)


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
class TestAsyncOpenAIRunnerInit:
    """测试 AsyncOpenAIRunner 初始化"""

    def test_init_minimal(self):
        """测试最小参数初始化"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com/v1/chat/completions",
            model="gpt-4",
            headers={"Authorization": "Bearer test-key"},
        )

        assert runner.endpoint_url == "https://api.example.com/v1/chat/completions"
        assert runner.model == "gpt-4"
        assert runner.headers["Authorization"] == "Bearer test-key"
        assert runner.temperature == 0.7
        assert runner.timeout == 3600

    def test_init_full(self):
        """测试完整参数初始化"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com/v1/chat/completions",
            model="gpt-4",
            headers={"Authorization": "Bearer test-key"},
            temperature=0.5,
            timeout=1800,
        )

        assert runner.temperature == 0.5
        assert runner.timeout == 1800

    def test_set_tools(self):
        """测试设置工具"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        executor = DummyToolExecutor()
        runner.set_tools(executor)

        assert runner._tool_executor == executor


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
class TestAsyncOpenAIRunnerToolSpec:
    """测试工具规范转换"""

    def test_tool_to_openai_spec_simple(self):
        """测试简单格式转换"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        tool = {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object"},
        }

        spec = runner._tool_to_openai_spec(tool)

        assert spec["type"] == "function"
        assert spec["function"]["name"] == "read_file"
        assert spec["function"]["description"] == "Read a file"
        assert spec["function"]["parameters"] == {"type": "object"}

    def test_tool_to_openai_spec_already_formatted(self):
        """测试已格式化的工具"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        tool = {
            "type": "function",
            "function": {"name": "read_file", "description": "Read a file"},
        }

        spec = runner._tool_to_openai_spec(tool)

        assert spec == tool


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
@pytest.mark.asyncio
class TestAsyncOpenAIRunnerProcessNonStream:
    """测试非 streaming 响应处理"""

    async def test_process_simple_response(self):
        """测试简单响应"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {
            "choices": [
                {"message": {"role": "assistant", "content": "Hello, world!"}}
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        # 检查事件类型
        event_types = [e.type for e in events]
        assert EventType.CONTENT_DELTA in event_types
        assert EventType.CONTENT_COMPLETE in event_types
        assert EventType.DONE in event_types

        # 检查内容
        content_events = [e for e in events if e.type == EventType.CONTENT_DELTA]
        assert content_events[0].data == "Hello, world!"
        for event in events:
            assert event.metadata.get("run_id") == "run_test"
            assert event.metadata.get("iteration_id") == "run_test_iteration"
            assert event.metadata.get("request_id") == "test_request"

    async def test_process_tool_call_response(self):
        """测试工具调用响应"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        runner.set_tools(DummyToolExecutor(result="tool-ok"))

        result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "tool", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        event_types = [e.type for e in events]
        assert EventType.CONTENT_COMPLETE in event_types
        assert EventType.TOOL_CALL_START in event_types
        assert EventType.TOOL_CALL_DISPATCHED in event_types
        assert EventType.TOOL_CALLS_BATCH_READY in event_types
        assert EventType.TOOL_CALL_RESULT in event_types
        assert EventType.TOOL_CALLS_BATCH_DONE in event_types
        assert EventType.DONE in event_types

    async def test_process_tool_call_response_with_dict_arguments(self):
        """测试工具调用参数为对象时可被兼容处理。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        runner.set_tools(DummyToolExecutor(result="tool-ok"))

        result = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {"name": "tool", "arguments": {"path": "a.txt"}},
                            }
                        ],
                    }
                }
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        event_types = [e.type for e in events]
        assert EventType.TOOL_CALL_RESULT in event_types
        assert EventType.DONE in event_types

    async def test_process_tool_calls_item_is_not_object(self):
        """测试 tool_calls 条目不是对象时返回错误。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {
            "choices": [
                {"message": {"role": "assistant", "content": "", "tool_calls": ["bad-item"]}}
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        assert events
        assert events[0].type == EventType.ERROR
        assert events[0].metadata.get("error_type") == "tool_call_incomplete"

    async def test_process_empty_response(self):
        """测试空响应"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {"choices": []}

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        # 应该有错误事件
        error_events = [e for e in events if e.type == EventType.ERROR]
        assert len(error_events) > 0

    async def test_process_non_stream_message_field_not_dict(self):
        """choices[0].message 非 dict（例如字符串）时应产出 response_error，不抛 AttributeError。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {"choices": [{"message": "oops"}]}

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        assert events
        assert events[0].type == EventType.ERROR
        assert events[0].metadata.get("error_type") == "response_error"

    async def test_process_content_none(self):
        """测试 content 为 None 时归一为字符串"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {
            "choices": [
                {"message": {"role": "assistant", "content": None}}
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        assert [e.type for e in events] == [
            EventType.CONTENT_COMPLETE,
            EventType.DONE,
        ]
        assert events[0].data == ""
        for event in events:
            assert event.metadata.get("run_id") == "run_test"
            assert event.metadata.get("iteration_id") == "run_test_iteration"
            assert event.metadata.get("request_id") == "test_request"

    async def test_process_tool_calls_none(self):
        """测试 tool_calls 为 None 时归一为空列表"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {
            "choices": [
                {"message": {"role": "assistant", "content": "", "tool_calls": None}}
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        assert [e.type for e in events] == [
            EventType.CONTENT_COMPLETE,
            EventType.DONE,
        ]
        assert events[1].data.get("tool_calls") == 0

    async def test_process_tool_calls_invalid_type(self):
        """测试 tool_calls 不是列表时返回错误"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {
            "choices": [
                {"message": {"role": "assistant", "content": "", "tool_calls": {"id": "x"}}}
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        assert events
        assert events[0].type == EventType.ERROR
        assert events[0].metadata.get("error_type") == "tool_call_invalid"

    @pytest.mark.parametrize(
        "tool_calls",
        [
            [{"function": {"name": "tool", "arguments": "{}"}}],
            [{"id": "call_1", "function": {"name": "tool", "arguments": "{"}}],
            [{"id": "call_1", "function": {"name": "tool", "arguments": "[]"}}],
            [{"id": None, "function": {"name": "tool", "arguments": "{}"}}],
            [{"id": 123, "function": {"name": "tool", "arguments": "{}"}}],
        ],
    )
    async def test_process_tool_calls_incomplete_or_invalid(self, tool_calls):
        """测试 tool_calls 缺失/无效参数时返回错误"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {
            "choices": [
                {"message": {"role": "assistant", "content": "", "tool_calls": tool_calls}}
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        assert events
        assert events[0].type == EventType.ERROR
        assert events[0].metadata.get("error_type") == "tool_call_incomplete"

    async def test_non_stream_done_event_truncated_true_on_finish_reason_length(self):
        """验证非流式响应 finish_reason=length 时 done_event summary 包含 truncated=True。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "partial output"},
                    "finish_reason": "length",
                }
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        done_events = [e for e in events if e.type == EventType.DONE]
        assert len(done_events) == 1
        assert done_events[0].data["truncated"] is True

    async def test_non_stream_done_event_truncated_false_on_finish_reason_stop(self):
        """验证非流式响应 finish_reason=stop 时 done_event summary 包含 truncated=False。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "full output"},
                    "finish_reason": "stop",
                }
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        done_events = [e for e in events if e.type == EventType.DONE]
        assert len(done_events) == 1
        assert done_events[0].data["truncated"] is False

    async def test_non_stream_done_event_marks_content_filtered(self):
        """验证非流式响应 finish_reason=content_filter 时 done_event summary 带 filtered 标志。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )

        result = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "partial output"},
                    "finish_reason": "content_filter",
                }
            ]
        }

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_non_stream(result, "test_request", trace_meta):
            events.append(event)

        done_events = [e for e in events if e.type == EventType.DONE]
        assert len(done_events) == 1
        assert done_events[0].data["truncated"] is False
        assert done_events[0].data["content_filtered"] is True
        assert done_events[0].data["finish_reason"] == "content_filter"


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
@pytest.mark.asyncio
class TestAsyncOpenAIRunnerProcessSSE:
    """测试 SSE 响应处理"""

    async def test_process_sse_content_none(self):
        """测试 SSE content 为 None 时归一为空字符串"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"content":null},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        assert [e.type for e in events] == [
            EventType.CONTENT_COMPLETE,
            EventType.DONE,
        ]
        assert events[0].data == ""

    async def test_process_sse_reasoning_content_in_metadata(self):
        """测试 SSE 流中 reasoning_content 产出 REASONING_DELTA 事件并透传到 content_complete metadata"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="deepseek-reasoner",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"reasoning_content":"思考"}}]}\n\n',
            'data: {"choices":[{"delta":{"reasoning_content":"中"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"result"}}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        event_types = [e.type for e in events]
        # REASONING_DELTA 事件应被透传
        reasoning_events = [e for e in events if e.type == EventType.REASONING_DELTA]
        assert len(reasoning_events) == 2
        assert reasoning_events[0].data == "思考"
        assert reasoning_events[1].data == "中"
        # content_complete metadata 应携带完整 reasoning_content
        cc_events = [e for e in events if e.type == EventType.CONTENT_COMPLETE]
        assert len(cc_events) == 1
        assert cc_events[0].metadata.get("reasoning_content") == "思考中"

    async def test_process_sse_no_reasoning_content_clean(self):
        """测试 SSE 流不含 reasoning_content 时 content_complete metadata 无噪声"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        # 不应有 REASONING_DELTA 事件
        assert all(e.type != EventType.REASONING_DELTA for e in events)
        # content_complete metadata 不应含 reasoning_content
        cc_events = [e for e in events if e.type == EventType.CONTENT_COMPLETE]
        assert "reasoning_content" not in cc_events[0].metadata

    async def test_process_sse_tool_calls_none(self):
        """测试 SSE tool_calls 为 None 时归一为空列表"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"tool_calls":null},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        assert [e.type for e in events] == [
            EventType.CONTENT_COMPLETE,
            EventType.DONE,
        ]
        assert events[1].data.get("tool_calls") == 0

    async def test_process_sse_no_choices_error(self):
        """测试 SSE payload 缺少 choices 时返回错误"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"id":"x"}\n\n',
            'data: [DONE]\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        assert events
        assert events[0].type == EventType.ERROR
        assert events[0].metadata.get("error_type") == "response_error"

    async def test_process_sse_empty_delta_without_done(self):
        """测试 SSE 只有空 delta 且无 [DONE] 时仍返回完成事件"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{}}]}\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        assert [e.type for e in events] == [
            EventType.CONTENT_COMPLETE,
            EventType.DONE,
        ]

    async def test_process_sse_tool_calls_missing_id_or_name(self):
        """测试 SSE 工具调用缺失 id/name 时返回错误"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            'data: [DONE]\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        assert events
        # T2 fix: validation_errors 路径也先发 content_complete 再发 error
        assert events[-2].type == EventType.CONTENT_COMPLETE
        assert events[-1].type == EventType.ERROR
        assert events[-1].metadata.get("error_type") == "tool_call_incomplete"

    async def test_process_sse_tool_calls_invalid_json(self):
        """测试 SSE 工具调用参数 JSON 无效时返回错误"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"tool","arguments":"{"}}]},"finish_reason":"tool_calls"}]}\n\n',
            'data: [DONE]\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        assert events
        assert EventType.TOOL_CALL_START in [e.type for e in events]
        # T2 fix: validation_errors 路径也先发 content_complete 再发 error
        assert events[-2].type == EventType.CONTENT_COMPLETE
        assert events[-1].type == EventType.ERROR
        assert events[-1].metadata.get("error_type") == "tool_call_incomplete"

    async def test_process_sse_tool_calls_not_list_trailing_line(self):
        """测试 SSE tool_calls 非列表且无换行结尾时返回结构化错误。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"tool_calls":{}}}]}',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        assert len(events) == 2
        # T2 fix: 错误路径也先发 content_complete 再发 error
        assert events[0].type == EventType.CONTENT_COMPLETE
        assert events[1].type == EventType.ERROR
        assert events[1].metadata.get("error_type") == "tool_call_invalid"

    async def test_process_sse_invalid_utf8_returns_response_error(self):
        """测试 SSE 非法 UTF-8 不再静默忽略。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            b'data: {"choices":[{"delta":{"content":"\xff"}}]}\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        # T2 fix: protocol_errors 路径先发 content_complete 再发 error
        assert len(events) == 2
        assert events[0].type == EventType.CONTENT_COMPLETE
        assert events[1].type == EventType.ERROR
        assert events[1].metadata.get("error_type") == "response_error"

    async def test_process_sse_utf8_split_preserves_multibyte_content(self):
        """测试 UTF-8 多字节字符跨 chunk 时内容无损。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        payload = 'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'.encode("utf-8")
        split_at = payload.index("你".encode("utf-8")) + 1
        response = _fake_client_response([
            payload[:split_at],
            payload[split_at:],
            b"data: [DONE]\n\n",
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        content_events = [event for event in events if event.type == EventType.CONTENT_COMPLETE]
        assert len(content_events) == 1
        assert content_events[0].data == "你好"

    async def test_sse_done_event_truncated_true_on_finish_reason_length(self):
        """验证 SSE 流式响应 finish_reason=length 时 done_event summary 包含 truncated=True。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":"length"}]}\n\n',
            'data: [DONE]\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        done_events = [e for e in events if e.type == EventType.DONE]
        assert len(done_events) == 1
        assert done_events[0].data["truncated"] is True

        # 内容完成事件仍正常发射（不改变控制流）
        content_complete_events = [e for e in events if e.type == EventType.CONTENT_COMPLETE]
        assert len(content_complete_events) == 1
        assert content_complete_events[0].data == "partial"

    async def test_sse_done_event_truncated_false_on_finish_reason_stop(self):
        """验证 SSE 流式响应 finish_reason=stop 时 done_event summary 包含 truncated=False。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"content":"full"},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        done_events = [e for e in events if e.type == EventType.DONE]
        assert len(done_events) == 1
        assert done_events[0].data["truncated"] is False

    async def test_sse_done_event_marks_content_filtered(self):
        """验证 SSE 流式响应 finish_reason=content_filter 时 done_event summary 带 filtered 标志。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":"content_filter"}]}\n\n',
            'data: [DONE]\n\n',
        ])

        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        done_events = [e for e in events if e.type == EventType.DONE]
        assert len(done_events) == 1
        assert done_events[0].data["truncated"] is False
        assert done_events[0].data["content_filtered"] is True
        assert done_events[0].data["finish_reason"] == "content_filter"


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
@pytest.mark.asyncio
class TestAsyncOpenAIRunnerCall:
    """测试 API 调用（Mock）"""

    async def test_call_non_stream(self):
        """测试非 streaming 调用"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com/v1/chat/completions",
            model="gpt-4",
            headers={"Authorization": "Bearer test"},
            default_extra_payloads={"max_tokens": 10, "foo": "bar"},
        )

        mock_response = {
            "choices": [
                {"message": {"role": "assistant", "content": "test response"}}
            ]
        }

        with patch("aiohttp.ClientSession") as mock_session:
            # Mock 响应
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.headers = {"Content-Type": "application/json"}
            mock_resp.json = AsyncMock(return_value=mock_response)

            mock_post = AsyncMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_post.__aexit__ = AsyncMock()

            mock_session_inst = MagicMock()
            mock_session_inst.post = MagicMock(return_value=mock_post)
            mock_session_inst.close = AsyncMock()
            mock_session_inst.closed = False
            mock_session_inst.__aenter__ = AsyncMock(return_value=mock_session_inst)
            mock_session_inst.__aexit__ = AsyncMock()

            mock_session.return_value = mock_session_inst

            # 调用
            messages = _message_list(cast(AgentMessage, {"role": "user", "content": "test"}))
            events = []

            async for event in runner.call(messages, stream=False, foo="override"):
                events.append(event)

            # 验证
            assert len(events) > 0
            event_types = [e.type for e in events]
            assert EventType.CONTENT_DELTA in event_types
            assert EventType.DONE in event_types

            payload = mock_session_inst.post.call_args.kwargs["json"]
            assert payload["foo"] == "override"
            assert payload["max_tokens"] == 10
            mock_session_inst.close.assert_not_awaited()

            await runner.close()

            mock_session_inst.close.assert_awaited_once()


def test_aiohttp_import_error():
    """测试 aiohttp 未安装时的错误"""
    if AIOHTTP_AVAILABLE:
        pytest.skip("aiohttp is installed")

    with pytest.raises(ImportError):
        AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )


@pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
@pytest.mark.asyncio
class TestSSEErrorPathContentComplete:
    """T2: SSE 错误路径在 error 前先发 content_complete 事件。"""

    async def test_protocol_error_emits_content_complete_first(self):
        """protocol_errors 路径先 yield content_complete 再 yield error。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        # 构造一个触发 protocol_error 的 SSE 流：无效 JSON payload
        response = _fake_client_response([
            b'data: not valid json\n\n',
            b'data: [DONE]\n\n',
        ])
        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        types = [e.type for e in events]
        # 倒数第二个是 content_complete，最后一个是 error
        assert types[-2] == EventType.CONTENT_COMPLETE
        assert types[-1] == EventType.ERROR

    async def test_validation_error_emits_content_complete_first(self):
        """validation_errors 路径先 yield content_complete 再 yield error。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        # 工具调用缺少 id → 触发 validation_errors
        response = _fake_client_response([
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            'data: [DONE]\n\n',
        ])
        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        types = [e.type for e in events]
        assert types[-2] == EventType.CONTENT_COMPLETE
        assert types[-1] == EventType.ERROR
        assert events[-1].metadata.get("error_type") == "tool_call_incomplete"

    async def test_empty_output_done_event_has_truncated_field(self):
        """T8: 空输出的 done_event 应包含 truncated 字段。"""
        runner = AsyncOpenAIRunner(
            endpoint_url="https://api.example.com",
            model="gpt-4",
            headers={},
        )
        # 发送一个有 choices 但 delta 为空的 SSE 流
        response = _fake_client_response([
            'data: {"choices":[{"delta":{}}]}\n\n',
            'data: [DONE]\n\n',
        ])
        events = []
        trace_meta = {"run_id": "run_test", "iteration_id": "run_test_iteration", "request_id": "test_request"}
        async for event in runner._process_sse_stream(response, "test_request", trace_meta):
            events.append(event)

        done_events = [e for e in events if e.type == EventType.DONE]
        assert len(done_events) == 1
        assert "truncated" in done_events[0].data
