"""
测试 AsyncAgent
"""
import asyncio
from typing import Any

import pytest

from dayu.engine import AgentResult, AsyncAgent
from dayu.engine.async_cli_runner import AsyncCliRunner
from dayu.engine import (
    EventType,
    content_complete,
    content_delta,
    done_event,
    error_event,
    final_answer_event,
    metadata_event,
    tool_call_dispatched,
    tool_call_result,
    tool_calls_batch_done,
)
from dayu.engine.async_agent import AgentRunningConfig
from dayu.contracts.cancellation import CancelledError, CancellationToken
from dayu.engine.events import reasoning_delta
from dayu.engine.tool_contracts import DupCallSpec


class DummyRunner:
    def __init__(self, event_batches, supports_tools=False):
        self.event_batches = list(event_batches)
        self.calls = []
        self.set_tools_calls = []
        self._supports_tools = supports_tools

    def is_supports_tool_calling(self):
        return self._supports_tools

    def set_tools(self, *args, **kwargs):
        self.set_tools_calls.append((args, kwargs))

    async def close(self) -> None:
        return None

    async def call(self, messages, stream=True, **extra_payloads):
        self.calls.append(
            {"messages": messages, "stream": stream, "extra_payloads": extra_payloads}
        )
        batch = self.event_batches.pop(0)
        for event in batch:
            yield event


class DummyDupSpecToolExecutor:
    """用于重复调用策略测试的最小工具执行器。"""

    def __init__(self, dup_specs=None):
        """初始化工具执行器桩。

        Args:
            dup_specs: 工具名到 DupCallSpec 的映射。

        Returns:
            无。

        Raises:
            无。
        """

        self._dup_specs = dict(dup_specs or {})

    def get_schemas(self):
        """返回空 schema 列表。"""

        return []

    def execute(self, name, arguments, context=None):
        """兼容 ToolExecutor 协议的桩实现。"""

        _ = (name, arguments, context)
        return {"ok": True, "value": {}}

    def clear_cursors(self):
        """兼容 ToolExecutor 协议的 no-op。"""

        return None

    def register_response_middleware(self, callback):
        """兼容 ToolExecutor 协议的 no-op。"""

        _ = callback
        return None

    def get_dup_call_spec(self, name):
        """按工具名返回 DupCallSpec。"""

        return self._dup_specs.get(name)

    def get_execution_context_param_name(self, name):
        """兼容 ToolExecutor 协议的 no-op。"""

        _ = name
        return None


class BlockingRunner:
    """用于并发运行防护测试的阻塞 Runner。"""

    def __init__(self) -> None:
        """初始化阻塞 Runner。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """
        self.calls: list[dict] = []
        self._release_event = asyncio.Event()

    def is_supports_tool_calling(self) -> bool:
        """声明不支持工具调用。

        Args:
            无。

        Returns:
            False。

        Raises:
            无。
        """
        return False

    def set_tools(self, *args, **kwargs) -> None:
        """兼容 Runner 协议的 no-op。

        Args:
            *args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            无。

        Raises:
            无。
        """
        _ = (args, kwargs)

    async def close(self) -> None:
        return None

    async def call(self, messages, stream=True, **extra_payloads):
        """返回一个会被阻塞的事件流。

        Args:
            messages: 输入消息。
            stream: 是否流式。
            **extra_payloads: 额外参数。

        Yields:
            StreamEvent 事件。

        Raises:
            无。
        """
        self.calls.append(
            {"messages": messages, "stream": stream, "extra_payloads": extra_payloads}
        )
        yield content_delta("hold")
        await self._release_event.wait()
        yield content_complete("hold")
        yield done_event()

    def release(self) -> None:
        """释放阻塞，允许事件流结束。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """
        self._release_event.set()


async def _consume_agent_run_until_done(
    *,
    agent: AsyncAgent,
    prompt: str,
    first_delta_seen: asyncio.Event,
) -> None:
    """消费一次 Agent 运行，直到事件流结束。

    Args:
        agent: 待运行的 Agent。
        prompt: 用户输入。
        first_delta_seen: 首个 CONTENT_DELTA 到达时置位的事件。

    Returns:
        无。

    Raises:
        无。
    """

    async for event in agent.run(prompt):
        if event.type == EventType.CONTENT_DELTA:
            first_delta_seen.set()


class DummyToolExecutor:
    def __init__(self, schemas=None):
        self._schemas = schemas or []
        self._middlewares = []

    def get_schemas(self):
        return self._schemas

    def get_tool_guidance(self):
        return "Use tools carefully."

    def execute(self, name, arguments, context=None):
        _ = (name, arguments, context)
        return {"ok": True, "value": {}}

    def clear_cursors(self):
        pass

    def get_dup_call_spec(self, name):
        _ = name
        return None

    def get_execution_context_param_name(self, name):
        _ = name
        return None

    def register_response_middleware(self, callback):
        self._middlewares.append(callback)
        return None


class DummyToolTraceRecorder:
    """工具调用追踪 recorder 桩。"""

    def __init__(self) -> None:
        """初始化追踪 recorder 桩。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.started_iterations: list[dict[str, Any]] = []
        self.dispatched: list[dict] = []
        self.results: list[dict] = []
        self.iteration_usages: list[dict] = []
        self.finished_iterations: list[dict] = []
        self.final_responses: list[dict] = []
        self.closed = False

    def start_iteration(
        self,
        *,
        iteration_id: str,
        model_input_messages,
        tool_schemas,
    ) -> None:
        """记录 start_iteration。

        Args:
            iteration_id: iteration ID。
            model_input_messages: 送模消息。
            tool_schemas: 原始工具 schema。

        Returns:
            无。

        Raises:
            无。
        """

        self.started_iterations.append(
            {
                "iteration_id": iteration_id,
                "model_input_messages": model_input_messages,
                "tool_schemas": tool_schemas,
            }
        )

    def on_tool_dispatched(
        self,
        *,
        iteration_id: str,
        payload,
    ) -> None:
        """记录工具请求事件。

        Args:
            iteration_id: iteration ID。
            payload: 原始事件载荷。

        Returns:
            无。

        Raises:
            无。
        """

        self.dispatched.append(
            {
                "iteration_id": iteration_id,
                "payload": payload,
            }
        )

    def on_tool_result(
        self,
        *,
        iteration_id: str,
        payload,
    ) -> None:
        """记录工具返回事件。

        Args:
            iteration_id: iteration ID。
            payload: 原始事件载荷。

        Returns:
            无。

        Raises:
            无。
        """

        self.results.append(
            {
                "iteration_id": iteration_id,
                "payload": payload,
            }
        )

    def record_iteration_usage(
        self,
        *,
        iteration_id: str,
        usage,
        budget_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """记录 iteration_usage。

        Args:
            iteration_id: iteration ID。
            usage: token 用量。
            budget_snapshot: 预算快照。

        Returns:
            无。

        Raises:
            无。
        """

        self.iteration_usages.append(
            {
                "iteration_id": iteration_id,
                "usage": usage,
                "budget_snapshot": budget_snapshot,
            }
        )

    def record_final_response(
        self,
        *,
        iteration_id: str,
        content: str,
        degraded: bool,
    ) -> None:
        """记录最终回答。

        Args:
            iteration_id: iteration ID。
            content: 最终回答内容。
            degraded: 是否降级回答。

        Returns:
            无。

        Raises:
            无。
        """

        self.final_responses.append(
            {
                "iteration_id": iteration_id,
                "content": content,
                "degraded": degraded,
            }
        )

    def finish_iteration(self, *, iteration_id: str, iteration_index: int) -> None:
        """记录 finish_iteration。

        Args:
            iteration_id: iteration ID。
            iteration_index: iteration 序号。

        Returns:
            无。

        Raises:
            无。
        """

        self.finished_iterations.append(
            {
                "iteration_id": iteration_id,
                "iteration_index": iteration_index,
            }
        )

    def close(self) -> None:
        """记录 close 调用。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.closed = True


class DummyToolTraceRecorderFactory:
    """工具调用追踪 recorder 工厂桩。"""

    def __init__(self) -> None:
        """初始化工厂桩。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.created_recorders: list[dict[str, Any]] = []

    def create_recorder(
        self,
        *,
        run_id: str,
        session_id: str,
        agent_metadata: dict[str, Any] | None = None,
    ) -> DummyToolTraceRecorder:
        """创建新的 recorder。

        Args:
            run_id: 运行 ID。
            session_id: 会话 ID。
            agent_metadata: Agent 元数据。

        Returns:
            新建的 recorder 桩。

        Raises:
            无。
        """

        recorder = DummyToolTraceRecorder()
        self.created_recorders.append(
            {
                "run_id": run_id,
                "session_id": session_id,
                "agent_metadata": agent_metadata or {},
                "recorder": recorder,
            }
        )
        return recorder


class TestAgentResult:
    """测试 AgentResult"""
    
    def test_create_result(self):
        """测试创建结果"""
        result = AgentResult(
            content="test content",
            tool_calls=[{"id": "1", "name": "test"}],
            errors=[],
            messages=[{"role": "user", "content": "hi"}]
        )
        
        assert result.content == "test content"
        assert len(result.tool_calls) == 1
        assert len(result.errors) == 0
        assert len(result.messages) == 1
    
    def test_success_property(self):
        """测试 success 属性"""
        result_success = AgentResult(
            content="test",
            tool_calls=[],
            errors=[],
            messages=[]
        )
        
        result_failed = AgentResult(
            content="test",
            tool_calls=[],
            errors=[{"message": "error"}],
            messages=[]
        )
        
        assert result_success.success is True
        assert result_failed.success is False
    
    def test_repr(self):
        """测试字符串表示"""
        result = AgentResult(
            content="a" * 100,
            tool_calls=[{"id": "1"}],
            errors=[],
            messages=[]
        )
        
        repr_str = repr(result)
        
        assert "AgentResult" in repr_str
        assert "tool_calls=1" in repr_str
        assert "errors=0" in repr_str
        # 无 warnings 时不展示
        assert "warnings" not in repr_str

    def test_repr_with_warnings(self):
        """有 warnings 时 repr 展示 warnings 计数。"""
        result = AgentResult(
            content="a" * 100,
            tool_calls=[],
            errors=[],
            messages=[],
            warnings=["w1", "w2"],
        )
        repr_str = repr(result)
        assert "warnings=2" in repr_str


class TestAsyncAgentInit:
    """测试 AsyncAgent 初始化"""
    
    def test_init_minimal(self):
        """测试最小参数初始化"""
        runner = DummyRunner([])
        agent = AsyncAgent(runner)
        
        assert agent.runner == runner
        assert agent.running_config.max_iterations == 16
        # AsyncAgent是无状态的，不保存_messages
    
    def test_init_with_system_message_removed_from_constructor(self):
        """测试 system_prompt 不再由构造函数注入。"""
        runner = DummyRunner([])
        agent = AsyncAgent(runner)
        
        assert isinstance(agent, AsyncAgent)
        # AsyncAgent是无状态的，messages在run()时构造
    
    def test_init_with_tools_support(self):
        """测试带工具能力 Runner 时可正常初始化。"""
        runner = DummyRunner([], supports_tools=True)
        agent = AsyncAgent(runner)

        assert agent.runner is runner
    
    def test_init_with_tool_executor_without_system_message(self):
        """测试带 executor 但不显式传入 system_message。"""
        runner = DummyRunner([], supports_tools=True)
        executor = DummyToolExecutor([{"name": "test_tool"}])
        
        # 新逻辑：不再从 executor 获取 tool_guidance
        agent = AsyncAgent(runner, tool_executor=executor)
        
        assert agent.tool_executor is executor
    
    def test_init_with_executor(self):
        """测试带 executor 初始化。"""
        runner = DummyRunner([], supports_tools=True)
        executor = DummyToolExecutor([{"name": "test_tool"}])
        
        agent = AsyncAgent(runner, tool_executor=executor)
        
        assert agent.tool_executor is executor
    
    def test_init_with_tool_executor(self):
        """测试带工具执行器初始化"""
        runner = DummyRunner([])
        executor = DummyToolExecutor([{"name": "test_tool"}])

        agent = AsyncAgent(runner, tool_executor=executor)

        assert agent.tool_executor == executor
    
    def test_init_with_max_iterations(self):
        """测试自定义最大迭代次数"""
        runner = DummyRunner([])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(max_iterations=5)
        agent = AsyncAgent(runner, running_config=config)
        
        assert agent.running_config.max_iterations == 5

    def test_default_running_config_not_shared(self):
        """测试默认运行配置实例不共享。"""
        runner_a = DummyRunner([])
        runner_b = DummyRunner([])
        agent_a = AsyncAgent(runner_a)
        agent_b = AsyncAgent(runner_b)

        agent_a.running_config.max_iterations = 3

        assert agent_b.running_config.max_iterations == 16


@pytest.mark.asyncio
class TestAsyncAgentRun:
    """测试 AsyncAgent run 方法"""
    
    async def test_run_streaming(self):
        """测试 streaming 模式"""
        runner = DummyRunner(
            [[content_delta("hi"), content_complete("hi"), done_event()]]
        )
        agent = AsyncAgent(runner)
        
        events = []
        async for event in agent.run("test prompt"):
            events.append(event)
        
        event_types = [e.type for e in events]
        assert EventType.CONTENT_DELTA in event_types
        assert EventType.CONTENT_COMPLETE in event_types
        assert EventType.DONE in event_types
        assert EventType.FINAL_ANSWER in event_types
        assert events[-1].data == final_answer_event("hi", degraded=False).data
        run_ids = {event.metadata.get("run_id") for event in events}
        iteration_ids = {event.metadata.get("iteration_id") for event in events}
        assert run_ids == {next(iter(run_ids))}
        assert None not in run_ids
        assert iteration_ids == {next(iter(iteration_ids))}
        assert None not in iteration_ids
    
    async def test_run_non_streaming(self):
        """测试非 streaming 模式"""
        runner = DummyRunner(
            [[content_delta("hi"), content_complete("hi"), done_event()]]
        )
        agent = AsyncAgent(runner)
        
        result = await agent.run_and_wait("test prompt")
        
        assert isinstance(result, AgentResult)
        assert isinstance(result.content, str)
        assert isinstance(result.tool_calls, list)
        assert isinstance(result.errors, list)
        assert result.content == "hi"
        assert runner.calls[0]["stream"] is True
    
    async def test_run_is_stateless(self):
        """测试 run 是无状态的（每次调用独立）"""
        runner = DummyRunner([
            [content_delta("first"), content_complete("first"), done_event()],
            [content_delta("second"), content_complete("second"), done_event()],
        ])
        agent = AsyncAgent(runner)
        
        # 第一次调用
        async for _ in agent.run("first prompt", system_prompt="You are a helper"):
            pass
        
        # 第二次调用（应该是独立的，不包含第一次的消息）
        async for _ in agent.run("second prompt", system_prompt="You are a helper"):
            pass
        
        assert len(runner.calls) == 2
        assert runner.calls[0]["messages"] == [
            {"role": "system", "content": "You are a helper"},
            {"role": "user", "content": "first prompt"},
        ]
        assert runner.calls[1]["messages"] == [
            {"role": "system", "content": "You are a helper"},
            {"role": "user", "content": "second prompt"},
        ]

    async def test_run_raises_when_same_agent_runs_concurrently(self):
        """验证同一 AsyncAgent 实例并发运行会抛出异常。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        runner = BlockingRunner()
        agent = AsyncAgent(runner)
        first_delta_seen = asyncio.Event()
        first_task = asyncio.create_task(
            _consume_agent_run_until_done(
                agent=agent,
                prompt="first prompt",
                first_delta_seen=first_delta_seen,
            )
        )
        await asyncio.wait_for(first_delta_seen.wait(), timeout=1.0)

        with pytest.raises(RuntimeError, match="不支持并发运行"):
            async for _ in agent.run("second prompt"):
                pass

        runner.release()
        await first_task

    async def test_agent_controls_next_iteration_after_tool_batch(self):
        """测试下一次 agent iteration 由 Agent 触发（Runner 只负责单次调用）"""
        tool_args = {"path": "test.txt"}
        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("done"), content_complete("done"), done_event()],
        ])
        agent = AsyncAgent(runner)

        events = []
        async for event in agent.run("test prompt"):
            events.append(event)

        assert len(runner.calls) == 2
        second_messages = runner.calls[1]["messages"]
        assistant_message = next(m for m in second_messages if m.get("role") == "assistant")
        tool_message = next(m for m in second_messages if m.get("role") == "tool")

        tool_calls = assistant_message.get("tool_calls", [])
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "tool"
        assert "path" in tool_calls[0]["function"]["arguments"]
        assert "ok" in tool_message.get("content", "")
        run_ids = {event.metadata.get("run_id") for event in events}
        iteration_ids = {event.metadata.get("iteration_id") for event in events}
        assert len(run_ids) == 1
        assert len(iteration_ids) >= 2

    async def test_agent_projects_bytes_tool_result_to_json_safe_payload(self):
        """验证 bytes 工具结果会以 base64 结构注入下一轮消息。"""
        tool_args = {"path": "binary.bin"}
        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": b"abc"},
                    name="tool",
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("done"), content_complete("done"), done_event()],
        ])
        agent = AsyncAgent(runner)

        async for _ in agent.run("test prompt"):
            pass

        second_messages = runner.calls[1]["messages"]
        tool_message = next(message for message in second_messages if message.get("role") == "tool")
        assert '"content_base64": "YWJj"' in tool_message["content"]
        assert '"content_encoding": "base64"' in tool_message["content"]

    async def test_agent_predictively_caps_tool_results_before_next_iteration(self):
        """验证 Agent 会在注入下一轮消息前按预算预测性截断工具结果。"""

        tool_args = {"path": "oversized.txt"}
        large_text = "x" * 30000
        runner = DummyRunner([
            [
                metadata_event("usage", {"prompt_tokens": 8800, "completion_tokens": 500}),
                tool_call_dispatched("call_1", "tool", tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": large_text},
                    name="tool",
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("done"), content_complete("done"), done_event()],
        ], supports_tools=True)
        agent = AsyncAgent(runner, running_config=AgentRunningConfig(max_context_tokens=10000))

        events = []
        async for event in agent.run("test prompt"):
            events.append(event)

        warnings = [
            event.data.get("message", "")
            for event in events
            if event.type == EventType.WARNING and isinstance(event.data, dict)
        ]
        second_messages = runner.calls[1]["messages"]
        tool_message = next(message for message in second_messages if message.get("role") == "tool")

        assert any("预测性截断" in message for message in warnings)
        assert "CONTEXT_BUDGET_TRUNCATED" in tool_message["content"]
        assert "search_document" not in tool_message["content"]
        assert "within_section_ref" not in tool_message["content"]

    async def test_duplicate_tool_call_triggers_early_exit(self):
        """测试重复工具调用触发提前退出"""
        tool_args = {"query": "same"}
        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                tool_call_dispatched("call_2", "tool", tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_2",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_2"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
        ])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(fallback_mode="raise_error", max_duplicate_tool_calls=1)
        agent = AsyncAgent(runner, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.ERROR in event_types
        assert EventType.FINAL_ANSWER not in event_types
        tool_events = [event for event in events if event.type == EventType.TOOL_CALL_RESULT]
        assert tool_events
        assert tool_events[0].metadata.get("tool_call_id") == "call_1"

    async def test_duplicate_tool_call_soft_hint_then_continue(self):
        """测试首次无增量重复调用触发软提醒，并继续到下一轮生成答案。"""
        tool_args = {"query": "same"}
        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                tool_call_dispatched("call_2", "tool", tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_2",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_2"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("final answer"), content_complete("final answer"), done_event()],
        ])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(
            fallback_mode="raise_error",
            max_duplicate_tool_calls=2,
        )
        agent = AsyncAgent(runner, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.WARNING in event_types
        assert EventType.ERROR not in event_types
        assert EventType.FINAL_ANSWER in event_types
        assert len(runner.calls) == 3
        warning_messages = [
            event.data.get("message", "")
            for event in events
            if event.type == EventType.WARNING
        ]
        assert any(message.startswith("⚠️ 检测到重复工具调用") for message in warning_messages)

        third_iteration_messages = runner.calls[2]["messages"]
        assert any(
            msg.get("role") == "user" and "same tool (tool)" in msg.get("content", "")
            for msg in third_iteration_messages
        ), f"Expected duplicate hint with 'same tool (tool)' in messages: {[m.get('content','')[:80] for m in third_iteration_messages if m.get('role')=='user']}"

    async def test_duplicate_tool_call_with_information_gain_not_early_exit(self):
        """测试重复调用若结果有信息增量，不触发重复调用提前退出。"""
        tool_args = {"query": "same"}
        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": "v1"},
                    name="tool",
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                tool_call_dispatched("call_2", "tool", tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_2",
                    {"ok": True, "value": "v2"},
                    name="tool",
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_2"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("final answer"), content_complete("final answer"), done_event()],
        ])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(
            fallback_mode="raise_error",
            max_duplicate_tool_calls=1,
        )
        agent = AsyncAgent(runner, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.ERROR not in event_types
        assert EventType.FINAL_ANSWER in event_types
        assert len(runner.calls) == 3

    async def test_polling_tool_running_status_skips_duplicate_hint(self):
        """测试声明为 polling 的工具在未终态时允许重复轮询。"""

        tool_name = "get_financial_filing_download_job_status"
        tool_args = {"job_id": "job_1"}
        running_result = {
            "ok": True,
            "value": {
                "job": {"job_id": "job_1", "status": "running"},
                "next_step": {"action": "poll_status", "tool_name": tool_name},
            },
        }
        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", tool_name, tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    running_result,
                    name=tool_name,
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                tool_call_dispatched("call_2", tool_name, tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_2",
                    running_result,
                    name=tool_name,
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_2"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("final answer"), content_complete("final answer"), done_event()],
        ])
        from dayu.engine.async_agent import AgentRunningConfig

        tool_executor = DummyDupSpecToolExecutor(
            dup_specs={
                tool_name: DupCallSpec(
                    mode="poll_until_terminal",
                    status_path="job.status",
                    terminal_values=["succeeded", "failed", "cancelled"],
                )
            }
        )
        config = AgentRunningConfig(
            fallback_mode="raise_error",
            max_duplicate_tool_calls=1,
        )
        agent = AsyncAgent(runner, tool_executor=tool_executor, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.ERROR not in event_types
        assert EventType.WARNING not in event_types
        assert EventType.FINAL_ANSWER in event_types
        assert len(runner.calls) == 3

    async def test_polling_tool_terminal_status_restores_duplicate_guard(self):
        """测试 polling 工具进入终态后会恢复重复调用保护。"""

        tool_name = "get_financial_filing_download_job_status"
        tool_args = {"job_id": "job_1"}
        terminal_result = {
            "ok": True,
            "value": {
                "job": {"job_id": "job_1", "status": "succeeded"},
                "next_step": {"action": "stop", "tool_name": None},
            },
        }
        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", tool_name, tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    terminal_result,
                    name=tool_name,
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                tool_call_dispatched("call_2", tool_name, tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_2",
                    terminal_result,
                    name=tool_name,
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_2"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
        ])
        from dayu.engine.async_agent import AgentRunningConfig

        tool_executor = DummyDupSpecToolExecutor(
            dup_specs={
                tool_name: DupCallSpec(
                    mode="poll_until_terminal",
                    status_path="job.status",
                    terminal_values=["succeeded", "failed", "cancelled"],
                )
            }
        )
        config = AgentRunningConfig(
            fallback_mode="raise_error",
            max_duplicate_tool_calls=1,
        )
        agent = AsyncAgent(runner, tool_executor=tool_executor, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.ERROR in event_types
        assert EventType.FINAL_ANSWER not in event_types

    async def test_max_duplicate_tool_calls_zero_uses_guard_value(self):
        """测试重复调用阈值配置为 0 时会被保护为 1，而非首轮误触发。"""
        tool_args = {"query": "same"}
        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", tool_args, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments=tool_args,
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("final answer"), content_complete("final answer"), done_event()],
        ])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(
            fallback_mode="raise_error",
            max_duplicate_tool_calls=0,
        )
        agent = AsyncAgent(runner, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.ERROR not in event_types
        assert EventType.FINAL_ANSWER in event_types
        assert len(runner.calls) == 2

    async def test_consecutive_failed_tool_batches_trigger_raise_error(self):
        """测试连续失败工具批次达到阈值后会触发 raise_error。"""
        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", {"iter": 1}, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": False, "error": "FAIL", "message": "fail"},
                    name="tool",
                    arguments={"iter": 1},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=0, error=1, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                tool_call_dispatched("call_2", "tool", {"iter": 2}, index_in_iteration=0),
                tool_call_result(
                    "call_2",
                    {"ok": False, "error": "FAIL", "message": "fail"},
                    name="tool",
                    arguments={"iter": 2},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_2"], ok=0, error=1, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
        ])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(
            fallback_mode="raise_error",
            max_consecutive_failed_tool_batches=2,
        )
        agent = AsyncAgent(runner, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.ERROR in event_types
        assert EventType.FINAL_ANSWER not in event_types
        assert len(runner.calls) == 2

    async def test_consecutive_failed_tool_batches_trigger_force_answer(self):
        """测试连续失败工具批次达到阈值后会触发 force_answer。"""

        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", {"iter": 1}, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": False, "error": "FAIL", "message": "fail"},
                    name="tool",
                    arguments={"iter": 1},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=0, error=1, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                tool_call_dispatched("call_2", "tool", {"iter": 2}, index_in_iteration=0),
                tool_call_result(
                    "call_2",
                    {"ok": False, "error": "FAIL", "message": "fail"},
                    name="tool",
                    arguments={"iter": 2},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_2"], ok=0, error=1, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("forced answer"), content_complete("forced answer"), done_event()],
        ])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(
            fallback_mode="force_answer",
            max_consecutive_failed_tool_batches=2,
        )
        agent = AsyncAgent(runner, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.WARNING in event_types
        assert EventType.FINAL_ANSWER in event_types

    async def test_cancelled_before_final_answer_suppresses_final_event(self):
        """在 final_answer 落外部事实前取消时，不应再产出 final_answer。"""

        cancellation_token = CancellationToken()

        class _CancelBeforeFinalRunner(DummyRunner):
            async def call(self, messages, stream=True, **extra_payloads):
                async for event in super().call(messages, stream=stream, **extra_payloads):
                    yield event
                cancellation_token.cancel()

        runner = _CancelBeforeFinalRunner([
            [content_delta("final body"), content_complete("final body"), done_event()],
        ])
        agent = AsyncAgent(runner, cancellation_token=cancellation_token)

        events = []
        with pytest.raises(CancelledError):
            async for event in agent.run("prompt"):
                events.append(event)

        event_types = [event.type for event in events]
        assert EventType.FINAL_ANSWER not in event_types

    async def test_runner_cancelled_mid_stream_propagates_and_suppresses_final_answer(self):
        """Runner 在流式阶段抛协作式取消时，Agent 不应再产出 final_answer。"""

        class _CancelledMidStreamRunner(DummyRunner):
            async def call(self, messages, stream=True, **extra_payloads):
                self.calls.append(
                    {"messages": messages, "stream": stream, "extra_payloads": extra_payloads}
                )
                yield content_delta("partial")
                raise CancelledError("runner cancelled during stream")

        runner = _CancelledMidStreamRunner([])
        agent = AsyncAgent(runner)

        events = []
        with pytest.raises(CancelledError):
            async for event in agent.run("prompt"):
                events.append(event)

        event_types = [event.type for event in events]
        assert EventType.CONTENT_DELTA in event_types
        assert EventType.FINAL_ANSWER not in event_types

    async def test_cancelled_before_force_answer_suppresses_degraded_final_event(self):
        """force_answer 降级路径在最终回答前取消时，不应再产出 degraded final_answer。"""

        cancellation_token = CancellationToken()

        class _CancelBeforeForceAnswerRunner(DummyRunner):
            async def call(self, messages, stream=True, **extra_payloads):
                async for event in super().call(messages, stream=stream, **extra_payloads):
                    yield event
                if len(self.calls) == 2:
                    cancellation_token.cancel()

        runner = _CancelBeforeForceAnswerRunner([
            [
                tool_call_dispatched("call_1", "tool", {"iter": 1}, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": False, "error": "FAIL", "message": "fail"},
                    name="tool",
                    arguments={"iter": 1},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=0, error=1, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("forced answer"), content_complete("forced answer"), done_event()],
        ])
        from dayu.engine.async_agent import AgentRunningConfig

        config = AgentRunningConfig(
            fallback_mode="force_answer",
            max_consecutive_failed_tool_batches=1,
        )
        agent = AsyncAgent(
            runner,
            running_config=config,
            cancellation_token=cancellation_token,
        )

        events = []
        with pytest.raises(CancelledError):
            async for event in agent.run("prompt"):
                events.append(event)

        event_types = [event.type for event in events]
        assert EventType.FINAL_ANSWER not in event_types
        assert len(runner.calls) == 2

    async def test_successful_tool_batch_resets_failed_batch_counter(self):
        """测试某轮工具批次成功会清零连续失败计数。"""

        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", {"iter": 1}, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": False, "error": "FAIL", "message": "fail"},
                    name="tool",
                    arguments={"iter": 1},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=0, error=1, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                tool_call_dispatched("call_2", "tool", {"iter": 2}, index_in_iteration=0),
                tool_call_result(
                    "call_2",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments={"iter": 2},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_2"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                tool_call_dispatched("call_3", "tool", {"iter": 3}, index_in_iteration=0),
                tool_call_result(
                    "call_3",
                    {"ok": False, "error": "FAIL", "message": "fail"},
                    name="tool",
                    arguments={"iter": 3},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_3"], ok=0, error=1, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("final answer"), content_complete("final answer"), done_event()],
        ])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(
            fallback_mode="raise_error",
            max_consecutive_failed_tool_batches=2,
        )
        agent = AsyncAgent(runner, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.ERROR not in event_types
        assert EventType.FINAL_ANSWER in event_types
        assert len(runner.calls) == 4

    async def test_non_tool_iteration_does_not_count_as_failed_batch(self):
        """测试无工具调用的 iteration 不计入连续失败工具批次。"""

        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", {"iter": 1}, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": False, "error": "FAIL", "message": "fail"},
                    name="tool",
                    arguments={"iter": 1},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=0, error=1, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                content_delta("partial"),
                content_complete("partial"),
                done_event(summary={"truncated": True}),
            ],
            [
                tool_call_dispatched("call_2", "tool", {"iter": 2}, index_in_iteration=0),
                tool_call_result(
                    "call_2",
                    {"ok": False, "error": "FAIL", "message": "fail"},
                    name="tool",
                    arguments={"iter": 2},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_2"], ok=0, error=1, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [content_delta("final answer"), content_complete("final answer"), done_event()],
        ])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(
            fallback_mode="raise_error",
            max_consecutive_failed_tool_batches=2,
        )
        agent = AsyncAgent(runner, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.ERROR not in event_types
        assert EventType.FINAL_ANSWER in event_types
        assert len(runner.calls) == 4

    async def test_fallback_error_stops_without_final_answer(self):
        """测试降级路径遇到 error_event 不再产出 final_answer"""
        runner = DummyRunner([
            [
                tool_call_dispatched("call_1", "tool", {}, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments={},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                content_complete(""),
                done_event(),
            ],
            [
                error_event("fallback failed", recoverable=False),
            ],
        ])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(max_iterations=1, max_compactions=0, fallback_mode="force_answer")
        agent = AsyncAgent(runner, running_config=config)

        events = []
        async for event in agent.run("test prompt"):
            events.append(event)

        event_types = [e.type for e in events]
        assert EventType.ERROR in event_types
        assert EventType.FINAL_ANSWER not in event_types

    async def test_content_filter_yields_filtered_final_answer_without_continuation(self):
        """验证 content_filter 不触发续写，且最终答案带 filtered 标记。"""
        runner = DummyRunner([
            [
                content_delta("partial"),
                content_complete("partial"),
                done_event(summary={"content_filtered": True, "finish_reason": "content_filter", "truncated": False}),
            ],
        ])
        agent = AsyncAgent(runner)

        events = []
        async for event in agent.run("test prompt"):
            events.append(event)

        warnings = [event for event in events if event.type == EventType.WARNING]
        final_events = [event for event in events if event.type == EventType.FINAL_ANSWER]

        assert len(runner.calls) == 1
        assert warnings
        assert len(final_events) == 1
        assert final_events[0].data == {
            "content": "partial",
            "degraded": True,
            "filtered": True,
            "finish_reason": "content_filter",
        }


class TestAsyncAgentExplicitConstruction:
    """测试 AsyncAgent 显式构造。"""

    def test_init_cli_runner(self):
        """测试使用显式 CLI runner 构造 Agent。"""

        runner = AsyncCliRunner(
            command=["codex", "exec"],
            timeout=3600,
            model="gpt-5.4",
            full_auto=False,
            reasoning_effort="medium",
        )
        agent = AsyncAgent(runner=runner)
        assert isinstance(agent, AsyncAgent)
        assert isinstance(agent.runner, AsyncCliRunner)


@pytest.mark.asyncio
class TestAsyncAgentStreaming:
    """测试 AsyncAgent streaming 功能"""
    
    async def test_streaming_collects_content(self):
        """测试 streaming 收集内容"""
        runner = DummyRunner(
            [[content_delta("test output"), content_complete("test output"), done_event()]]
        )
        agent = AsyncAgent(runner)
        
        content_parts = []
        async for event in agent.run("test"):
            if event.type == EventType.CONTENT_DELTA:
                content_parts.append(event.data)
        
        # 应该收集到内容
        full_content = "".join(content_parts)
        assert len(full_content) > 0
    
    async def test_non_streaming_returns_result(self):
        """测试非 streaming 返回完整结果"""
        runner = DummyRunner(
            [[content_delta("test"), content_complete("test"), done_event()]]
        )
        agent = AsyncAgent(runner)
        
        result = await agent.run_and_wait("test prompt")
        
        assert isinstance(result, AgentResult)
        assert result.success is True
        assert len(result.content) > 0

    async def test_tool_trace_recorder_records_lifecycle_events(self):
        """验证工具调用事件会写入追踪 recorder。"""

        runner = DummyRunner(
            [
                [
                    tool_call_dispatched("call_1", "tool", {"k": "v"}, index_in_iteration=0),
                    tool_call_result(
                        "call_1",
                        {"ok": True, "value": "ok"},
                        name="tool",
                        arguments={"k": "v"},
                        index_in_iteration=0,
                    ),
                    tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                    content_complete(""),
                    done_event(),
                ],
                [
                    content_delta("done"),
                    content_complete("done"),
                    done_event(),
                ],
            ]
        )
        trace_factory = DummyToolTraceRecorderFactory()
        agent = AsyncAgent(runner, tool_trace_recorder_factory=trace_factory)

        result = await agent.run_and_wait("test prompt")
        recorder = trace_factory.created_recorders[0]["recorder"]

        assert result.success is True
        assert len(trace_factory.created_recorders) == 1
        assert len(recorder.started_iterations) >= 1
        assert len(recorder.dispatched) == 1
        assert len(recorder.results) == 1
        assert len(recorder.iteration_usages) == 0
        assert len(recorder.finished_iterations) >= 1
        assert len(recorder.final_responses) == 1
        assert recorder.started_iterations[0]["model_input_messages"][-1]["content"] == "test prompt"
        assert recorder.dispatched[0]["payload"]["id"] == "call_1"
        assert recorder.results[0]["payload"]["result"]["ok"] is True
        assert recorder.final_responses[0]["content"] == "done"
        assert recorder.final_responses[0]["degraded"] is False
        assert recorder.closed is True

    async def test_tool_trace_recorder_close_on_unrecoverable_error(self):
        """验证不可恢复错误时也会触发 recorder.close。"""

        runner = DummyRunner([[error_event("fatal", recoverable=False)]])
        trace_factory = DummyToolTraceRecorderFactory()
        agent = AsyncAgent(runner, tool_trace_recorder_factory=trace_factory)

        result = await agent.run_and_wait("test prompt")
        recorder = trace_factory.created_recorders[0]["recorder"]

        assert result.success is False
        assert len(recorder.finished_iterations) == 1
        assert len(recorder.final_responses) == 0
        assert recorder.closed is True

    async def test_tool_trace_recorder_start_iteration_receives_raw_tool_schemas(self):
        """验证 start_iteration 会收到当前轮真实工具 schema。"""

        runner = DummyRunner(
            [[content_delta("done"), content_complete("done"), done_event()]],
            supports_tools=True,
        )
        trace_factory = DummyToolTraceRecorderFactory()
        tool_executor = DummyToolExecutor(
            schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "list_documents",
                        "description": "列出文档",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )
        agent = AsyncAgent(
            runner,
            tool_executor=tool_executor,
            tool_trace_recorder_factory=trace_factory,
        )

        result = await agent.run_and_wait("test prompt")
        recorder = trace_factory.created_recorders[0]["recorder"]

        assert result.success is True
        assert len(recorder.started_iterations) == 1
        assert recorder.started_iterations[0]["tool_schemas"][0]["function"]["name"] == "list_documents"

    async def test_fallback_raise_error(self):
        """测试达到上限后抛出错误"""
        runner = DummyRunner([
            [
                content_complete(""),
                tool_call_dispatched("call_1", "tool", {"iter": 1}, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments={"iter": 1},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                done_event(),
            ],
            [
                content_complete(""),
                tool_call_dispatched("call_2", "tool", {"iter": 2}, index_in_iteration=0),
                tool_call_result(
                    "call_2",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments={"iter": 2},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_2"], ok=1, error=0, timeout=0, cancelled=0),
                done_event(),
            ],
        ])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(max_iterations=2, max_compactions=2, fallback_mode="raise_error")
        agent = AsyncAgent(runner, running_config=config)

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        assert events[-1].type == EventType.ERROR

    async def test_fallback_force_answer(self):
        """测试达到上限后强制生成答案"""
        runner = DummyRunner([
            [
                content_complete(""),
                tool_call_dispatched("call_1", "tool", {"iter": 1}, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments={"iter": 1},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                done_event(),
            ],
            [content_delta("final answer")],
        ])
        executor = DummyToolExecutor([{"name": "tool"}])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(max_iterations=1, max_compactions=0, fallback_mode="force_answer")
        agent = AsyncAgent(
            runner,
            tool_executor=executor,
            running_config=config,
        )

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        event_types = [event.type for event in events]
        assert EventType.WARNING in event_types
        assert EventType.CONTENT_DELTA in event_types
        assert EventType.FINAL_ANSWER in event_types
        assert events[-1].data.get("degraded") is True

    async def test_fallback_force_answer_restores_tools(self):
        """T3: force_answer 结束后恢复 runner 上的工具状态。"""
        runner = DummyRunner([
            [
                content_complete(""),
                tool_call_dispatched("call_1", "tool", {"iter": 1}, index_in_iteration=0),
                tool_call_result(
                    "call_1",
                    {"ok": True, "value": "ok"},
                    name="tool",
                    arguments={"iter": 1},
                    index_in_iteration=0,
                ),
                tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                done_event(),
            ],
            [content_delta("forced answer"), content_complete("forced answer"), done_event()],
        ])
        executor = DummyToolExecutor([{"name": "tool"}])
        from dayu.engine.async_agent import AgentRunningConfig
        config = AgentRunningConfig(max_iterations=1, max_compactions=0, fallback_mode="force_answer")
        agent = AsyncAgent(
            runner,
            tool_executor=executor,
            running_config=config,
        )

        events = []
        async for event in agent.run("prompt"):
            events.append(event)

        # set_tools 调用序列：初始设置 → 清除(None) → 恢复(executor)
        set_tools_args = [call[0][0] for call in runner.set_tools_calls]
        # 最后一次 set_tools 应传入 executor（非 None）
        assert set_tools_args[-1] is executor, (
            f"force_answer 后未恢复工具状态，set_tools 序列: {set_tools_args}"
        )


@pytest.mark.asyncio
class TestAsyncAgentReasoningContent:
    """测试 reasoning_content 在 thinking 模式 tool-call 轮次的传递"""

    async def test_reasoning_delta_forwarded_to_caller(self):
        """验证 REASONING_DELTA 事件被透传给调用者"""
        runner = DummyRunner([
            [
                reasoning_delta("思考中"),
                content_delta("answer"),
                content_complete("answer"),
                done_event(),
            ],
        ])
        agent = AsyncAgent(runner)

        events = []
        async for event in agent.run("test"):
            events.append(event)

        event_types = [e.type for e in events]
        assert EventType.REASONING_DELTA in event_types
        rd_events = [e for e in events if e.type == EventType.REASONING_DELTA]
        assert rd_events[0].data == "思考中"

    async def test_reasoning_content_in_assistant_message_on_tool_call(self):
        """验证工具调用轮次中 assistant 消息正确包含 reasoning_content"""
        runner = DummyRunner(
            [
                # Turn 1: thinking + tool call
                [
                    reasoning_delta("让我查一下"),
                    tool_call_dispatched("call_1", "search", {"q": "test"}, index_in_iteration=0),
                    tool_call_result(
                        "call_1",
                        {"ok": True, "value": "found"},
                        name="search",
                        arguments={"q": "test"},
                        index_in_iteration=0,
                    ),
                    tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                    content_complete("", reasoning_content="让我查一下"),
                    done_event(),
                ],
                # Turn 2: final answer
                [
                    content_delta("结论"),
                    content_complete("结论"),
                    done_event(),
                ],
            ],
            supports_tools=True,
        )
        schemas = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        tool_executor = DummyToolExecutor(schemas=schemas)
        agent = AsyncAgent(runner, tool_executor=tool_executor)

        events = []
        async for event in agent.run("test"):
            events.append(event)

        # Turn 2 的 messages 应包含 reasoning_content
        assert len(runner.calls) == 2
        second_messages = runner.calls[1]["messages"]
        assistant_msgs = [m for m in second_messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[0].get("reasoning_content") == "让我查一下"

    async def test_no_reasoning_content_clean_assistant_message(self):
        """验证无 thinking 模式时 assistant 消息不含 reasoning_content 字段"""
        runner = DummyRunner(
            [
                # Turn 1: tool call without reasoning
                [
                    tool_call_dispatched("call_1", "search", {"q": "test"}, index_in_iteration=0),
                    tool_call_result(
                        "call_1",
                        {"ok": True, "value": "found"},
                        name="search",
                        arguments={"q": "test"},
                        index_in_iteration=0,
                    ),
                    tool_calls_batch_done(["call_1"], ok=1, error=0, timeout=0, cancelled=0),
                    content_complete(""),
                    done_event(),
                ],
                # Turn 2: final answer
                [
                    content_delta("ok"),
                    content_complete("ok"),
                    done_event(),
                ],
            ],
            supports_tools=True,
        )
        schemas = [{"type": "function", "function": {"name": "search", "parameters": {}}}]
        tool_executor = DummyToolExecutor(schemas=schemas)
        agent = AsyncAgent(runner, tool_executor=tool_executor)

        events = []
        async for event in agent.run("test"):
            events.append(event)

        assert len(runner.calls) == 2
        second_messages = runner.calls[1]["messages"]
        assistant_msgs = [m for m in second_messages if m.get("role") == "assistant"]
        assert "reasoning_content" not in assistant_msgs[0]


class TestAsyncAgentImport:
    """测试 AsyncAgent 导入"""
    
    def test_can_import_async_agent(self):
        """测试可以导入 AsyncAgent"""
        from dayu.engine import AsyncAgent
        
        runner = DummyRunner([])
        agent = AsyncAgent(runner)
        
        assert isinstance(agent, AsyncAgent)
