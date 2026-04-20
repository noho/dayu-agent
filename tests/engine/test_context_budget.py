"""上下文预算治理测试。

覆盖范围：
- ContextBudgetState 数据类属性与方法
- _compact_messages / _build_compaction_summary 压缩函数
- Agent 截断续写（truncated → continuation）
- Agent 上下文溢出压缩（context_overflow → compaction）
- Agent 主动预算压缩（soft limit → proactive compaction）
- Agent 显式 running_config 读取模型静态能力
- Runner _detect_context_overflow 检测
- Runner stream_options 注入
- SSE Parser usage 捕获
- tool_trace record_iteration_usage 记录
- 预测性工具结果截断（_estimate_chars_to_tokens / _truncate_tool_result_str / _cap_tool_results_for_budget）
- run_and_wait WARNING 事件收集
- AgentResult.warnings 字段
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, cast

import pytest

from dayu.contracts.agent_types import AgentMessage
from dayu.engine.context_budget import ContextBudgetState, ToolResultBudgetCapper
from dayu.engine.async_agent import (
    AgentRunningConfig,
    AsyncAgent,
    _build_compaction_summary,
    _compact_messages,
    _find_safe_split_point,
    AgentResult,
)
from dayu.engine.async_openai_runner import AsyncOpenAIRunnerRunningConfig, _detect_context_overflow
from dayu.engine.events import (
    EventType,
    StreamEvent,
    content_complete,
    done_event,
    error_event,
    metadata_event,
    warning_event,
)
from dayu.engine.sse_parser import SSEStreamParser
from dayu.engine.tool_trace import JsonlToolTraceStore, V2ToolTraceRecorder, TRACE_TYPE_ITERATION_USAGE

if TYPE_CHECKING:
    from aiohttp import ClientResponse


# ---------- 辅助类 ----------


@dataclass
class _SSERunningConfigStub(AsyncOpenAIRunnerRunningConfig):
    """SSE 解析器运行配置桩。"""

    debug_sse: bool = False
    debug_sse_sample_rate: float = 1.0
    debug_sse_throttle_sec: float = 0.0
    debug_tool_delta: bool = False


class _RunnerStub:
    """Runner 桩，按顺序返回预置的事件批次。

    Attributes:
        calls: 记录每次 call() 的参数列表。
    """

    def __init__(self, batches: list[list[StreamEvent]]) -> None:
        """初始化 RunnerStub。

        Args:
            batches: 每次调用返回的事件列表。
        """
        self._batches = list(batches)
        self.calls: list[dict[str, Any]] = []

    def is_supports_tool_calling(self) -> bool:
        """是否支持工具调用。"""
        return False

    def set_tools(self, *args: Any, **kwargs: Any) -> None:
        """设置工具（桩实现，无操作）。"""
        pass

    async def close(self) -> None:
        """关闭 Runner（桩实现，无操作）。"""
        return None

    async def call(
        self,
        messages: list[AgentMessage],
        *,
        stream: bool = True,
        **extra_payloads: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """模拟 Runner.call()，返回预置事件。

        Args:
            messages: 消息列表。
            stream: 是否流式。
            **extra_payloads: 额外参数。

        Yields:
            预置的 StreamEvent。
        """
        self.calls.append({"messages": list(messages), "stream": stream, "extra_payloads": extra_payloads})
        if self._batches:
            batch = self._batches.pop(0)
            for event in batch:
                yield event


def _read_trace_records(output_dir: Path) -> list[dict[str, Any]]:
    """读取 trace 目录下所有 JSONL 记录。

    Args:
        output_dir: 追踪输出目录。

    Returns:
        解析后的记录列表。
    """
    records = []
    for f in sorted((output_dir / "sessions").rglob("*.jsonl*")):
        if f.suffix == ".gz":
            text = gzip.decompress(f.read_bytes()).decode("utf-8")
        else:
            text = f.read_text("utf-8")
        for line in text.splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


async def _empty_stream_events() -> AsyncGenerator[StreamEvent, None]:
    """返回空的事件流。"""

    if False:
        yield warning_event("unused")


def _as_agent_messages(messages: list[dict[str, Any]]) -> list[AgentMessage]:
    """把测试消息列表显式收窄为 AgentMessage 列表。"""

    return cast(list[AgentMessage], messages)


def _message_content(message: AgentMessage) -> str:
    """安全读取测试消息内容。"""

    return str(message.get("content") or "")


def _message_role(message: AgentMessage) -> str:
    """安全读取测试消息角色。"""

    return str(message.get("role") or "")


def _parse_sse_stream(
    parser: SSEStreamParser,
    response: object,
) -> AsyncGenerator[StreamEvent, None]:
    """把测试响应桩收窄为 SSE 解析器可接受的流生成器。"""

    return cast(
        AsyncGenerator[StreamEvent, None],
        parser.parse_stream(cast("ClientResponse", response)),
    )


# ========== ContextBudgetState 测试 ==========


class TestContextBudgetState:
    """ContextBudgetState 数据类测试。"""

    def test_default_budget_disabled(self) -> None:
        """默认 max_context_tokens=0 时预算治理关闭。"""
        state = ContextBudgetState()
        assert not state.is_budget_enabled
        assert state.soft_limit_tokens == 0
        assert state.hard_limit_tokens == 0
        assert not state.is_over_soft_limit
        assert not state.is_over_hard_limit

    def test_budget_enabled_with_limits(self) -> None:
        """启用预算后正确计算软硬阈值。"""
        state = ContextBudgetState(
            max_context_tokens=100000,
            soft_limit_ratio=0.75,
            hard_limit_ratio=0.90,
        )
        assert state.is_budget_enabled
        assert state.soft_limit_tokens == 75000
        assert state.hard_limit_tokens == 90000

    def test_is_over_soft_limit(self) -> None:
        """prompt_tokens 超过软阈值时返回 True。"""
        state = ContextBudgetState(max_context_tokens=100000, soft_limit_ratio=0.75)
        state.current_prompt_tokens = 75000
        assert state.is_over_soft_limit

    def test_is_not_over_soft_limit(self) -> None:
        """prompt_tokens 未超过软阈值时返回 False。"""
        state = ContextBudgetState(max_context_tokens=100000, soft_limit_ratio=0.75)
        state.current_prompt_tokens = 74999
        assert not state.is_over_soft_limit

    def test_is_over_hard_limit(self) -> None:
        """prompt_tokens 超过硬阈值时返回 True。"""
        state = ContextBudgetState(max_context_tokens=100000, hard_limit_ratio=0.90)
        state.current_prompt_tokens = 90000
        assert state.is_over_hard_limit

    def test_record_usage(self) -> None:
        """record_usage 正确累计用量。"""
        state = ContextBudgetState(max_context_tokens=131072)
        state.record_usage({"prompt_tokens": 5000, "completion_tokens": 1000})
        assert state.current_prompt_tokens == 5000
        assert state.latest_completion_tokens == 1000
        assert state.total_prompt_tokens == 5000
        assert state.total_completion_tokens == 1000
        assert state.iteration_count == 1

        state.record_usage({"prompt_tokens": 8000, "completion_tokens": 2000})
        assert state.current_prompt_tokens == 8000
        assert state.total_prompt_tokens == 13000
        assert state.total_completion_tokens == 3000
        assert state.iteration_count == 2

    def test_record_usage_missing_fields(self) -> None:
        """usage 缺少字段时默认为 0。"""
        state = ContextBudgetState()
        state.record_usage({})
        assert state.current_prompt_tokens == 0
        assert state.iteration_count == 1


# ========== 消息压缩测试 ==========


class TestCompactMessages:
    """_compact_messages / _build_compaction_summary 测试。"""

    def test_short_messages_not_compressed(self) -> None:
        """消息数不足时原样返回，且 compacted=False。"""
        messages: list[AgentMessage] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result, compacted = _compact_messages(messages, recent_keep=6)
        assert result == messages
        assert compacted is False

    def test_compaction_preserves_system_and_first_user(self) -> None:
        """压缩保留 system message 和首条 user message。"""
        messages: list[AgentMessage] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task goal"},
        ]
        # 添加足够多的中间消息和尾部消息
        for i in range(10):
            messages.append({"role": "assistant", "content": f"resp_{i}", "tool_calls": []})
            messages.append({"role": "user", "content": f"follow_up_{i}"})

        result, compacted = _compact_messages(messages, recent_keep=4)
        # 首条 system + 首条 user + 1 摘要 + 4 recent = 7
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[1] == {"role": "user", "content": "task goal"}
        assert "[Context Compaction Summary]" in _message_content(result[2])
        assert len(result) == 7
        assert compacted is True

    def test_compaction_preserves_recent_tail(self) -> None:
        """压缩保留最近 N 条消息不变。"""
        messages: list[AgentMessage] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "goal"},
        ]
        tail_messages: list[AgentMessage] = []
        for i in range(4):
            msg: AgentMessage = {"role": "assistant", "content": f"tail_{i}"}
            tail_messages.append(msg)
            messages.append({"role": "assistant", "content": f"old_{i}"})
        messages.extend(tail_messages)

        result, compacted = _compact_messages(messages, recent_keep=4)
        assert result[-4:] == tail_messages
        assert compacted is True

    def test_build_compaction_summary_with_tool_calls(self) -> None:
        """摘要正确统计工具调用信息。"""
        messages: list[AgentMessage] = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_search_document",
                        "type": "function",
                        "function": {"name": "search_document", "arguments": "{}"},
                    },
                    {
                        "id": "call_fetch_web_page",
                        "type": "function",
                        "function": {"name": "fetch_web_page", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "1", "content": "result1"},
            {"role": "tool", "tool_call_id": "2", "content": "result2"},
            {"role": "user", "content": "continue"},
        ]
        summary = _build_compaction_summary(messages)
        assert "assistant=1" in summary
        assert "tool_call=2" in summary
        assert "tool_result=2" in summary
        assert "user=1" in summary
        assert "search_document" in summary
        assert "fetch_web_page" in summary

    def test_compact_messages_with_gap_between_system_and_first_user(self) -> None:
        """T7: system~first_user 之间有消息时，这些消息进入摘要而非被丢弃。"""
        messages: list[AgentMessage] = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "early_assistant"},
            {"role": "user", "content": "task goal"},
        ]
        # 添加足够多的后续消息触发压缩
        for i in range(10):
            messages.append({"role": "assistant", "content": f"resp_{i}"})
            messages.append({"role": "user", "content": f"follow_up_{i}"})

        result, compacted = _compact_messages(messages, recent_keep=4)
        assert compacted is True
        # system + first_user + 摘要 + 4 recent = 7
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[1] == {"role": "user", "content": "task goal"}
        # 摘要中应包含 early_assistant 的信息
        summary = _message_content(result[2])
        assert "[Context Compaction Summary]" in summary
        assert "assistant=" in summary

    def test_compact_messages_with_gap_does_not_duplicate_first_user(self) -> None:
        """T10: 默认 recent_keep 下，首条 user 不会同时落入摘要保留区和 recent tail。"""
        messages: list[AgentMessage] = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "early_assistant_1"},
            {"role": "assistant", "content": "early_assistant_2"},
            {"role": "user", "content": "task goal"},
            {"role": "assistant", "content": "resp_1"},
            {"role": "user", "content": "follow_up_1"},
            {"role": "assistant", "content": "resp_2"},
            {"role": "user", "content": "follow_up_2"},
            {"role": "assistant", "content": "resp_3"},
        ]

        result, compacted = _compact_messages(messages)

        assert compacted is True
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[1] == {"role": "user", "content": "task goal"}
        assert sum(1 for msg in result if msg.get("content") == "task goal") == 1
        assert len(result) == 8


# ========== Agent 截断续写测试 ==========


@pytest.mark.asyncio
async def test_agent_truncated_continuation() -> None:
    """finish_reason=length 时 Agent 自动续写。"""
    # 第一轮：截断返回
    batch_1 = [
        content_complete("partial content"),
        done_event(summary={"truncated": True, "usage": {"prompt_tokens": 5000, "completion_tokens": 1000}}),
    ]
    # 第二轮：正常返回
    batch_2 = [
        content_complete("continued content"),
        done_event(summary={"truncated": False, "usage": {"prompt_tokens": 6000, "completion_tokens": 500}}),
    ]
    runner = _RunnerStub([batch_1, batch_2])
    agent = AsyncAgent(
        runner,
        running_config=AgentRunningConfig(max_continuations=3),
    )

    events = []
    async for event in agent.run("prompt"):
        events.append(event)

    types = [e.type for e in events]
    # 应有 warning 事件表明截断续写
    assert EventType.WARNING in types
    warning_msgs = [e.data.get("message", "") if isinstance(e.data, dict) else str(e.data)
                    for e in events if e.type == EventType.WARNING]
    assert any("截断" in msg or "续写" in msg for msg in warning_msgs)
    # 最终应有 final_answer
    assert EventType.FINAL_ANSWER in types
    # Bug #2 fix: final_answer 应包含前序 + 续写的完整拼接内容
    final_events = [e for e in events if e.type == EventType.FINAL_ANSWER]
    assert len(final_events) == 1
    full = final_events[0].data["content"]
    assert "partial content" in full
    assert "continued content" in full
    assert full == "partial contentcontinued content"
    # Runner 应被调用 2 次
    assert len(runner.calls) == 2
    # 第二次调用的 messages 应包含 continuation prompt
    second_call_msgs = runner.calls[1]["messages"]
    assert any("truncated" in m.get("content", "") for m in second_call_msgs if m.get("role") == "user")


@pytest.mark.asyncio
async def test_agent_truncated_max_continuations_exceeded() -> None:
    """续写次数达到上限后直接返回 final_answer。"""
    # 所有轮次都截断
    truncated_batch = [
        content_complete("partial"),
        done_event(summary={"truncated": True, "usage": {"prompt_tokens": 5000, "completion_tokens": 1000}}),
    ]
    runner = _RunnerStub([truncated_batch] * 4)  # max_continuations=2，所以第 3 轮不续写
    agent = AsyncAgent(
        runner,
        running_config=AgentRunningConfig(max_continuations=2),
    )

    events = []
    async for event in agent.run("prompt"):
        events.append(event)

    types = [e.type for e in events]
    assert EventType.FINAL_ANSWER in types
    # 应调用 3 次（1 初始 + 2 续写），第 3 次截断但不再续写
    assert len(runner.calls) == 3


# ========== Agent context_overflow 压缩测试 ==========


@pytest.mark.asyncio
async def test_agent_context_overflow_compaction() -> None:
    """context_overflow 错误时 Agent 自动压缩并重试。"""
    # 先通过多轮截断续写积累 messages，再触发 overflow，最后成功
    # 每轮截断增加 2 条 messages（assistant + continuation user）
    truncated_batches = [
        [
            content_complete(f"trunc_{i}"),
            done_event(summary={"truncated": True, "usage": {"prompt_tokens": 5000, "completion_tokens": 1000}}),
        ]
        for i in range(4)  # 4 轮截断 → messages 从 2 增长到 10
    ]
    overflow_batch = [
        error_event("context length exceeded", recoverable=False, error_type="context_overflow"),
    ]
    normal_batch = [
        content_complete("success"),
        done_event(summary={"truncated": False, "usage": {"prompt_tokens": 3000, "completion_tokens": 500}}),
    ]
    runner = _RunnerStub(truncated_batches + [overflow_batch, normal_batch])

    agent = AsyncAgent(
        runner,
        running_config=AgentRunningConfig(max_continuations=5, max_compactions=3),
    )

    events = []
    async for event in agent.run("task goal"):
        events.append(event)

    types = [e.type for e in events]
    # 应有 warning 说明压缩
    assert EventType.WARNING in types
    warning_msgs = [e.data.get("message", "") if isinstance(e.data, dict) else str(e.data)
                    for e in events if e.type == EventType.WARNING]
    assert any("压缩" in msg for msg in warning_msgs)
    # 最终应有 final_answer
    assert EventType.FINAL_ANSWER in types


@pytest.mark.asyncio
async def test_agent_context_overflow_max_compactions_exceeded() -> None:
    """压缩次数达到上限后 context_overflow 错误不可恢复。"""
    # 先积累足够 messages，再连续 overflow 直到 compaction 预算耗尽
    truncated_batches = [
        [
            content_complete(f"trunc_{i}"),
            done_event(summary={"truncated": True, "usage": {"prompt_tokens": 5000, "completion_tokens": 1000}}),
        ]
        for i in range(4)
    ]
    overflow_batch = [
        error_event("context length exceeded", recoverable=False, error_type="context_overflow"),
    ]
    # max_compactions=2 → 最多压缩 2 次，第 3 次 overflow 不可恢复
    runner = _RunnerStub(truncated_batches + [overflow_batch] * 5)
    agent = AsyncAgent(
        runner,
        running_config=AgentRunningConfig(max_continuations=5, max_compactions=2),
    )

    events = []
    async for event in agent.run("prompt"):
        events.append(event)

    types = [e.type for e in events]
    # 最终应有不可恢复的 ERROR
    assert EventType.ERROR in types


@pytest.mark.asyncio
async def test_agent_context_overflow_iteration_id_unique() -> None:
    """T1/T9: context_overflow 压缩重试后 iteration_id 不冲突。

    验证即使 iteration 回退（iteration -= 1），iteration_id 仍然单调递增且唯一。
    """
    # 先通过多轮截断续写积累足够 messages 使压缩生效
    truncated_batches = [
        [
            content_complete(f"part_{i}"),
            done_event(summary={"truncated": True, "usage": {"prompt_tokens": 5000, "completion_tokens": 1000}}),
        ]
        for i in range(4)
    ]
    # 触发 context_overflow
    overflow_batch = [
        error_event("context length exceeded", recoverable=False, error_type="context_overflow"),
    ]
    # 压缩后重试：正常返回
    normal_batch = [
        content_complete("final"),
        done_event(summary={"truncated": False, "usage": {"prompt_tokens": 3000, "completion_tokens": 500}}),
    ]
    runner = _RunnerStub(truncated_batches + [overflow_batch, normal_batch])
    agent = AsyncAgent(
        runner,
        running_config=AgentRunningConfig(max_continuations=5, max_compactions=3),
    )

    events = []
    async for event in agent.run("task goal"):
        events.append(event)

    # 收集所有 iteration_id
    iteration_ids = []
    for e in events:
        iteration_id = e.metadata.get("iteration_id") if e.metadata else None
        if iteration_id and iteration_id not in iteration_ids:
            iteration_ids.append(iteration_id)

    # 所有 iteration_id 应唯一
    assert len(iteration_ids) == len(set(iteration_ids)), f"iteration_id 重复: {iteration_ids}"
    # 4 轮截断 + 1 overflow + 1 正常 = 至少 6 个不同 iteration_id
    assert len(iteration_ids) >= 6, f"预期至少 6 个 iteration_id，实际 {len(iteration_ids)}: {iteration_ids}"
    # 验证 iteration_id 序号单调递增
    iteration_nums = [int(iteration_id.rsplit("_", 1)[1]) for iteration_id in iteration_ids]
    assert iteration_nums == sorted(iteration_nums), f"iteration_id 序号非单调递增: {iteration_nums}"


# ========== Agent 主动预算压缩测试 ==========


@pytest.mark.asyncio
async def test_agent_proactive_soft_limit_compaction() -> None:
    """超过软阈值时 Agent 在调用前主动压缩。"""
    # 第一轮：返回高用量（超过软阈值 75000）
    batch_1 = [
        content_complete("r1"),
        done_event(summary={
            "truncated": False,
            "usage": {"prompt_tokens": 80000, "completion_tokens": 1000},
        }),
    ]
    # 理论上 Agent 查询后 tool_calls 继续循环，但这里没有工具调用，
    # 所以会直接 final_answer。我们用截断来继续循环。
    # 让第一轮截断以触发续写，续写时检查预算
    batch_1_trunc = [
        content_complete("partial"),
        done_event(summary={
            "truncated": True,
            "usage": {"prompt_tokens": 80000, "completion_tokens": 1000},
        }),
    ]
    batch_2 = [
        content_complete("final"),
        done_event(summary={
            "truncated": False,
            "usage": {"prompt_tokens": 50000, "completion_tokens": 500},
        }),
    ]
    runner = _RunnerStub([batch_1_trunc, batch_2])

    # 构造足够多 messages 使压缩有效
    agent = AsyncAgent(
        runner,
        running_config=AgentRunningConfig(
            max_context_tokens=100000,
            budget_soft_limit_ratio=0.75,
            max_continuations=3,
            max_compactions=3,
        ),
    )

    events = []
    async for event in agent.run("goal " + " ".join(f"padding_{i}" for i in range(20))):
        events.append(event)

    assert EventType.FINAL_ANSWER in [e.type for e in events]


# ========== Agent 显式 running_config 模型能力读取测试 ==========


def test_init_reads_explicit_running_config_capabilities() -> None:
    """显式 running_config 会直接驱动上下文窗口能力。"""

    class _CreatedRunner:
        """Runner 桩。"""

        def call(self, messages: list[AgentMessage], *, stream: bool = True, **extra_payloads: Any) -> AsyncGenerator[StreamEvent, None]:
            """提供协议所需的空调用实现。"""

            del messages, stream, extra_payloads
            return _empty_stream_events()

        def set_tools(self, executor: Any | None) -> None:
            """提供协议所需的空 set_tools 实现。"""

            del executor

        def is_supports_tool_calling(self) -> bool:
            return False

        async def close(self) -> None:
            """关闭 Runner（桩实现，无操作）。"""
            return None

    agent = AsyncAgent(
        runner=_CreatedRunner(),
        running_config=AgentRunningConfig(
            max_context_tokens=131072,
            max_output_tokens=8192,
        ),
    )
    assert agent.running_config.max_context_tokens == 131072
    assert agent.running_config.max_output_tokens == 8192


def test_init_builds_running_config_without_hidden_overrides() -> None:
    """显式 running_config 不应再依赖隐藏装配逻辑。"""

    class _CreatedRunner:
        """Runner 桩。"""

        def call(self, messages: list[AgentMessage], *, stream: bool = True, **extra_payloads: Any) -> AsyncGenerator[StreamEvent, None]:
            """提供协议所需的空调用实现。"""

            del messages, stream, extra_payloads
            return _empty_stream_events()

        def set_tools(self, executor: Any | None) -> None:
            """提供协议所需的空 set_tools 实现。"""

            del executor

        def is_supports_tool_calling(self) -> bool:
            return False

        async def close(self) -> None:
            """关闭 Runner（桩实现，无操作）。"""
            return None

    agent = AsyncAgent(
        runner=_CreatedRunner(),
        running_config=AgentRunningConfig(
            max_context_tokens=50000,
            max_output_tokens=4096,
        ),
    )
    assert agent.running_config.max_context_tokens == 50000
    assert agent.running_config.max_output_tokens == 4096


# ========== Runner _detect_context_overflow 测试 ==========


class TestDetectContextOverflow:
    """_detect_context_overflow 检测逻辑测试。"""

    def test_detect_by_error_code(self) -> None:
        """通过 JSON error.code 检测。"""
        body = json.dumps({
            "error": {
                "code": "context_length_exceeded",
                "message": "This model's maximum context length is 131072 tokens.",
            }
        })
        assert _detect_context_overflow(body) is True

    def test_detect_by_text_fallback(self) -> None:
        """通过文本回退检测。"""
        body = "Error: maximum context length is 131072 tokens. Requested 132000 tokens."
        assert _detect_context_overflow(body) is True

    def test_detect_by_mimo_text_fallback(self) -> None:
        """通过 MiMo 风格上下文超限文本检测。"""
        body = "Total message token length exceed model limit"
        assert _detect_context_overflow(body) is True

    def test_not_detected_for_other_400(self) -> None:
        """其他 400 错误不误判。"""
        body = json.dumps({
            "error": {
                "code": "invalid_api_key",
                "message": "Invalid API key.",
            }
        })
        assert _detect_context_overflow(body) is False

    def test_empty_body(self) -> None:
        """空 body 不误判。"""
        assert _detect_context_overflow("") is False

    def test_malformed_json(self) -> None:
        """非 JSON body 但不含关键词不误判。"""
        assert _detect_context_overflow("{broken json") is False


# ========== SSE Parser usage 捕获测试 ==========


class _ChunkedContentStub:
    """模拟 aiohttp content 对象。"""

    def __init__(self, chunks: list[bytes]) -> None:
        """初始化。

        Args:
            chunks: 预置的字节串序列。
        """
        self._chunks = chunks

    async def iter_chunked(self, _: int):
        """异步产出 chunk。

        Args:
            _: 未使用参数。

        Yields:
            bytes 数据块。
        """
        for chunk in self._chunks:
            yield chunk


class _SSEResponseStub:
    """模拟 aiohttp 响应对象。"""

    def __init__(self, chunks: list[bytes]) -> None:
        """初始化。

        Args:
            chunks: 预置的字节串序列。
        """
        self.content = _ChunkedContentStub(chunks)


class TestSSEParserUsageCapture:
    """SSEStreamParser usage 捕获测试。"""

    @pytest.mark.asyncio
    async def test_usage_captured_from_final_chunk(self) -> None:
        """从流式最终 chunk 中捕获 usage。"""
        parser = SSEStreamParser(
            name="test", request_id="req_usage", running_config=_SSERunningConfigStub(),
        )
        content_chunk = json.dumps({"choices": [{"delta": {"content": "hello"}, "index": 0}]})
        finish_chunk = json.dumps({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]})
        usage_data = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        usage_chunk = json.dumps({"choices": [], "usage": usage_data})

        chunks = [
            f"data: {content_chunk}\n\n".encode(),
            f"data: {finish_chunk}\n\n".encode(),
            f"data: {usage_chunk}\n\n".encode(),
            b"data: [DONE]\n\n",
        ]
        response = _SSEResponseStub(chunks)

        events = [e async for e in _parse_sse_stream(parser, response)]
        result = parser.get_result()
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50

    @pytest.mark.asyncio
    async def test_no_usage_when_not_provided(self) -> None:
        """没有 usage 数据时返回 None。"""
        parser = SSEStreamParser(
            name="test", request_id="req_no_usage", running_config=_SSERunningConfigStub(),
        )
        finish_chunk = json.dumps({"choices": [{"delta": {"content": "hi"}, "index": 0, "finish_reason": "stop"}]})
        chunks = [
            f"data: {finish_chunk}\n\n".encode(),
            b"data: [DONE]\n\n",
        ]
        response = _SSEResponseStub(chunks)

        events = [e async for e in _parse_sse_stream(parser, response)]
        result = parser.get_result()
        assert result.usage is None


# ========== tool_trace record_iteration_usage 测试 ==========


class TestToolTraceRecordIterationUsage:
    """V2ToolTraceRecorder.record_iteration_usage 测试。"""

    def test_record_iteration_usage(self, tmp_path: Path) -> None:
        """正确写入 iteration_usage trace 记录。"""
        store = JsonlToolTraceStore(tmp_path)
        recorder = V2ToolTraceRecorder(run_id="run_01", session_id="run_01", store=store)
        usage = {"prompt_tokens": 5000, "completion_tokens": 1000, "total_tokens": 6000}
        budget_snapshot = {
            "max_context_tokens": 131072,
            "current_prompt_tokens": 5000,
            "compaction_count": 0,
        }
        recorder.record_iteration_usage(
            iteration_id="run_01_iteration_1",
            usage=usage,
            budget_snapshot=budget_snapshot,
        )

        records = _read_trace_records(tmp_path)
        assert len(records) == 1
        record = records[0]
        assert record["trace_type"] == TRACE_TYPE_ITERATION_USAGE
        assert record["run_id"] == "run_01"
        assert record["iteration_id"] == "run_01_iteration_1"
        assert record["usage"]["prompt_tokens"] == 5000
        assert record["budget_snapshot"]["max_context_tokens"] == 131072

    def test_record_iteration_usage_without_budget(self, tmp_path: Path) -> None:
        """不提供 budget_snapshot 时记录中不含该字段。"""
        store = JsonlToolTraceStore(tmp_path)
        recorder = V2ToolTraceRecorder(run_id="run_02", session_id="run_02", store=store)
        recorder.record_iteration_usage(
            iteration_id="run_02_iteration_1",
            usage={"prompt_tokens": 1000, "completion_tokens": 200},
        )

        records = _read_trace_records(tmp_path)
        assert len(records) == 1
        assert "budget_snapshot" not in records[0]


# ========== Bug #1: tool 消息组原子性测试 ==========


class TestCompactMessagesToolGroupSafety:
    """_compact_messages / _find_safe_split_point 工具消息组原子性测试。"""

    def test_tool_group_not_split_across_boundary(self) -> None:
        """assistant(tool_calls) + tool 组横跨 recent_keep 边界时整组进入保留区。"""
        messages: list[AgentMessage] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "goal"},
            # 中间填充
            {"role": "assistant", "content": "resp_0"},
            {"role": "user", "content": "q_1"},
            {"role": "assistant", "content": "resp_1"},
            {"role": "user", "content": "q_2"},
            # tool 组：assistant + 2 个 tool —— 应不可拆分
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "tc1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
                {"id": "tc2", "type": "function", "function": {"name": "fetch", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "tc1", "content": "result1"},
            {"role": "tool", "tool_call_id": "tc2", "content": "result2"},
            # 后续消息
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "final_resp"},
        ]
        # recent_keep=4 → raw split 在 len(12)-4=8，即 tool(tc2) 上
        # 安全调整应将 split 移到 assistant(tool_calls) 位置 (idx=6)
        result, compacted = _compact_messages(messages, recent_keep=4)
        assert compacted is True
        # 检查结果中不存在孤立的 tool 消息（每个 tool 前面必须有一个 assistant(tool_calls)）
        for i, msg in enumerate(result):
            if msg.get("role") == "tool":
                # 前面必须有 assistant 或另一个同组 tool
                prev = result[i - 1]
                assert prev.get("role") in ("assistant", "tool"), (
                    f"孤立的 tool 消息在索引 {i}，前一条是 role={prev.get('role')}"
                )

    def test_find_safe_split_point_on_tool_message(self) -> None:
        """split 落在 tool 消息上时向前调整到 assistant(tool_calls)。"""
        messages: list[AgentMessage] = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "tc1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "r"},
            {"role": "user", "content": "follow"},
        ]
        # target=2 指向 tool → 应回退到 1 (assistant)
        assert _find_safe_split_point(messages, 2) == 1

    def test_find_safe_split_point_on_safe_boundary(self) -> None:
        """split 已在安全边界时不调整。"""
        messages: list[AgentMessage] = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "resp"},
            {"role": "user", "content": "follow"},
        ]
        assert _find_safe_split_point(messages, 2) == 2

    def test_find_safe_split_point_multi_tool_group(self) -> None:
        """多个连续 tool 消息构成的组回退到 assistant。"""
        messages: list[AgentMessage] = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "b", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
                    {"id": "c", "type": "function", "function": {"name": "tool_c", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "r1"},
            {"role": "tool", "tool_call_id": "b", "content": "r2"},
            {"role": "tool", "tool_call_id": "c", "content": "r3"},
            {"role": "user", "content": "continue"},
        ]
        # target=3 指向第 2 个 tool → 应回退到 1 (assistant)
        assert _find_safe_split_point(messages, 3) == 1
        # target=4 指向第 3 个 tool → 应回退到 1
        assert _find_safe_split_point(messages, 4) == 1

    def test_compaction_with_multiple_tool_rounds(self) -> None:
        """多轮工具调用的消息列表压缩后仍为合法序列。"""
        messages: list[AgentMessage] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "goal"},
        ]
        # 添加 5 轮工具调用
        for i in range(5):
            tc_id = f"tc_{i}"
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": tc_id, "type": "function",
                                "function": {"name": f"tool_{i}", "arguments": "{}"}}],
            })
            messages.append({"role": "tool", "tool_call_id": tc_id, "content": f"result_{i}"})
            messages.append({"role": "user", "content": f"follow_{i}"})
        # 添加最终回复
        messages.append({"role": "assistant", "content": "final"})

        result, compacted = _compact_messages(messages, recent_keep=4)
        assert compacted is True
        # 验证合法性：每个 tool 消息前面必须有 assistant(tool_calls) 或同组 tool
        for i, msg in enumerate(result):
            if msg.get("role") == "tool":
                prev = result[i - 1]
                assert prev.get("role") in ("assistant", "tool"), (
                    f"非法 tool 消息在压缩结果索引 {i}，"
                    f"前一条 role={_message_role(prev)}, content={_message_content(prev)[:50]}"
                )


# ========== Bug #2: 截断续写内容累积测试 ==========


@pytest.mark.asyncio
async def test_agent_truncated_triple_continuation_accumulates() -> None:
    """3 次续写后 final_answer 包含全部 4 段内容。"""
    batches = []
    for i in range(3):
        batches.append([
            content_complete(f"part_{i} "),
            done_event(summary={"truncated": True, "usage": {"prompt_tokens": 5000, "completion_tokens": 1000}}),
        ])
    # 最后一轮正常结束
    batches.append([
        content_complete("part_3"),
        done_event(summary={"truncated": False, "usage": {"prompt_tokens": 6000, "completion_tokens": 500}}),
    ])
    runner = _RunnerStub(batches)
    agent = AsyncAgent(
        runner,
        running_config=AgentRunningConfig(max_continuations=3),
    )

    events = []
    async for event in agent.run("prompt"):
        events.append(event)

    final_events = [e for e in events if e.type == EventType.FINAL_ANSWER]
    assert len(final_events) == 1
    full = final_events[0].data["content"]
    # 应包含所有 4 段
    assert full == "part_0 part_1 part_2 part_3"
    # Runner 应被调用 4 次
    assert len(runner.calls) == 4


# ========== Bug #4: system~first_user 间消息保留测试 ==========


class TestCompactMessagesSystemUserGap:
    """system 和 first_user 之间有消息时的压缩行为。"""

    def test_messages_between_system_and_first_user_in_summary(self) -> None:
        """system~first_user 间的消息不被丢弃，纳入摘要。"""
        messages: list[AgentMessage] = [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "prelude"},  # system~user 之间
            {"role": "user", "content": "goal"},
        ]
        # 添加足够多消息触发压缩
        for i in range(10):
            messages.append({"role": "assistant", "content": f"resp_{i}"})
            messages.append({"role": "user", "content": f"q_{i}"})

        result, compacted = _compact_messages(messages, recent_keep=4)
        assert compacted is True
        # system 和 first_user 应被保留
        assert result[0] == {"role": "system", "content": "sys"}
        assert result[1] == {"role": "user", "content": "goal"}
        # 摘要应包含 assistant prelude 的统计
        summary_msg = result[2]
        assert _message_role(summary_msg) == "user"
        assert "[Context Compaction Summary]" in _message_content(summary_msg)
        # "prelude" 本身在摘要统计中应被计为 1 个 assistant
        assert "assistant=" in _message_content(summary_msg)


# ========== Bug #5: overflow 检测精度测试 ==========


class TestDetectContextOverflowPrecision:
    """_detect_context_overflow 不误判非 overflow 的 400 错误。"""

    def test_no_false_positive_for_general_maximum_mention(self) -> None:
        """包含 'maximum' 但非 context 超限的 400 响应不误判。"""
        body = '{"error": {"code": "invalid_request", "message": "maximum number of retries reached"}}'
        assert _detect_context_overflow(body) is False

    def test_precise_match_with_is_suffix(self) -> None:
        """精确匹配 'maximum context length is' 信号。"""
        body = "Error: This model's maximum context length is 131072 tokens. Requested 150000."
        assert _detect_context_overflow(body) is True

    def test_context_length_exceeded_in_message(self) -> None:
        """文本中包含 'context length exceeded' 时正确检测。"""
        body = "Your input exceeds the context length exceeded limit for this model."
        assert _detect_context_overflow(body) is True

    def test_model_max_context_phrasing(self) -> None:
        """匹配 \"model's maximum context length\" 变体。"""
        body = "The model's maximum context length is 8192 tokens."
        assert _detect_context_overflow(body) is True

    def test_qwen_range_of_input_length_signal(self) -> None:
        """匹配 Qwen 标准报错中的 Range of input length 信号。"""
        body = "400-InvalidParameter: Range of input length should be [1, 131072]"
        assert _detect_context_overflow(body) is True

    def test_ollama_model_requires_more_context_signal(self) -> None:
        """匹配 Ollama 报错中的 model requires more context 信号。"""
        body = '{"error":"model requires more context than is available"}'
        assert _detect_context_overflow(body) is True


# ========== 续写路径压缩 warning 事件测试 ==========


@pytest.mark.asyncio
async def test_continuation_compaction_emits_warning() -> None:
    """续写路径触发压缩时 emit warning 事件（含 '续写前' 关键词）。

    通过多轮截断累积足够消息，使 _compact_messages 在续写前实际执行压缩，
    验证新增的 warning_event 被正确发射。
    """
    # 4 轮截断 → messages 从 2 增长到 10，足以让 _compact_messages 实际压缩
    truncated_batches = [
        [
            content_complete(f"part_{i}"),
            done_event(summary={
                "truncated": True,
                "usage": {"prompt_tokens": 80000, "completion_tokens": 500},
            }),
        ]
        for i in range(4)
    ]
    final_batch = [
        content_complete("done"),
        done_event(summary={
            "truncated": False,
            "usage": {"prompt_tokens": 50000, "completion_tokens": 500},
        }),
    ]
    runner = _RunnerStub(truncated_batches + [final_batch])

    agent = AsyncAgent(
        runner,
        running_config=AgentRunningConfig(
            max_context_tokens=100000,
            budget_soft_limit_ratio=0.75,  # soft_limit = 75000
            max_continuations=5,
            max_compactions=3,
        ),
    )

    events: list[StreamEvent] = []
    async for event in agent.run("goal"):
        events.append(event)

    warnings = [
        e.data.get("message", "") if isinstance(e.data, dict) else str(e.data)
        for e in events
        if e.type == EventType.WARNING
    ]
    # 续写路径的压缩 warning 应包含 "续写前" 关键词
    assert any("续写前" in msg for msg in warnings), (
        f"未找到包含 '续写前' 的 warning，实际 warnings: {warnings}"
    )
    assert EventType.FINAL_ANSWER in [e.type for e in events]


# ========== 预测性工具结果截断测试 ==========


class TestEstimateCharsToTokens:
    """ToolResultBudgetCapper.estimate_chars_to_tokens 单元测试。"""

    def test_basic_conversion(self) -> None:
        """1000 chars ≈ 250 tokens（按 4 chars/token）。"""
        assert ToolResultBudgetCapper.estimate_chars_to_tokens(1000) == 250

    def test_zero(self) -> None:
        """0 chars → 0 tokens。"""
        assert ToolResultBudgetCapper.estimate_chars_to_tokens(0) == 0

    def test_small_remainder(self) -> None:
        """不足一个 token 的字符数向下取整。"""
        assert ToolResultBudgetCapper.estimate_chars_to_tokens(3) == 0
        assert ToolResultBudgetCapper.estimate_chars_to_tokens(4) == 1


class TestTruncateToolResultStr:
    """ToolResultBudgetCapper.truncate_result_str 单元测试。"""

    def test_no_truncation_within_limit(self) -> None:
        """未超限时原样返回。"""
        text = "short text"
        assert ToolResultBudgetCapper.truncate_result_str(text, 100) == text

    def test_truncation_appends_note(self) -> None:
        """超限时截断并追加 CONTEXT_BUDGET_TRUNCATED 提示。"""
        text = "a" * 200
        result = ToolResultBudgetCapper.truncate_result_str(text, 50)
        assert len(result) > 50  # 截断后有追加提示
        assert result.startswith("a" * 50)
        assert "CONTEXT_BUDGET_TRUNCATED" in result
        assert "original=200" in result
        assert "kept=50" in result
        assert "search_document" not in result
        assert "within_section_ref" not in result

    def test_exact_boundary(self) -> None:
        """恰好等于限制时不截断。"""
        text = "x" * 100
        assert ToolResultBudgetCapper.truncate_result_str(text, 100) == text


class TestCapToolResultsForBudget:
    """ToolResultBudgetCapper.cap_results_for_budget 单元测试。"""

    def test_no_cap_when_budget_sufficient(self) -> None:
        """预算充足时不截断。"""
        state = ContextBudgetState(
            max_context_tokens=131072,
            soft_limit_ratio=0.75,
        )
        state.current_prompt_tokens = 10000
        state.latest_completion_tokens = 500
        pairs: list[tuple[dict[str, object], str]] = [
            ({"id": "c1"}, "small result"),
            ({"id": "c2"}, "another small result"),
        ]
        result, capped = ToolResultBudgetCapper.cap_results_for_budget(pairs, state)
        assert not capped
        assert result == pairs

    def test_cap_large_results(self) -> None:
        """大结果被截断，小结果保留。"""
        state = ContextBudgetState(
            max_context_tokens=100000,
            soft_limit_ratio=0.75,  # soft = 75000 tokens → available = 75000 - 70000 - 500 = 4500 tokens → 18000 chars
        )
        state.current_prompt_tokens = 70000
        state.latest_completion_tokens = 500
        large_result = "x" * 50000  # 50K chars，远超预算
        small_result = "y" * 100
        pairs: list[tuple[dict[str, object], str]] = [
            ({"id": "c1"}, large_result),
            ({"id": "c2"}, small_result),
        ]
        result, capped = ToolResultBudgetCapper.cap_results_for_budget(pairs, state)
        assert capped
        # 大结果被截断
        assert len(result[0][1]) < len(large_result)
        assert "CONTEXT_BUDGET_TRUNCATED" in result[0][1]
        # 小结果完整保留
        assert result[1][1] == small_result

    def test_ascending_fair_share_strategy(self) -> None:
        """升序公平分配：小结果完整保留，大结果获得剩余全部预算。"""
        state = ContextBudgetState(
            max_context_tokens=100000,
            soft_limit_ratio=0.75,  # soft = 75000 tokens
        )
        # available = 75000 - 60000 - 500 = 14500 tokens → 58000 chars
        state.current_prompt_tokens = 60000
        state.latest_completion_tokens = 500
        small_a = "a" * 100
        small_b = "b" * 500
        large_c = "c" * 200000
        pairs: list[tuple[dict[str, object], str]] = [
            ({"id": "c1"}, large_c),    # 大结果放前面
            ({"id": "c2"}, small_a),
            ({"id": "c3"}, small_b),
        ]
        result, capped = ToolResultBudgetCapper.cap_results_for_budget(pairs, state)
        assert capped
        # 小结果完整保留
        assert result[1][1] == small_a
        assert result[2][1] == small_b
        # 大结果获得剩余预算（58000 - 100 - 500 = 57400），远多于均分的 58000/3 ≈ 19333
        large_kept = result[0][1]
        assert "CONTEXT_BUDGET_TRUNCATED" in large_kept
        # 截断后保留字符数应接近 57400（大于均分值 19333）
        kept_chars = int(large_kept.split("kept=")[1].split(" ")[0])
        assert kept_chars > 50000, f"大结果应获得远多于均分的预算，实际 {kept_chars}"

    def test_empty_results_not_capped(self) -> None:
        """空结果不受影响。"""
        state = ContextBudgetState(
            max_context_tokens=100000,
            soft_limit_ratio=0.75,
        )
        state.current_prompt_tokens = 70000
        state.latest_completion_tokens = 500
        pairs: list[tuple[dict[str, object], str]] = [
            ({"id": "c1"}, ""),
            ({"id": "c2"}, "small"),
        ]
        result, capped = ToolResultBudgetCapper.cap_results_for_budget(pairs, state)
        assert not capped
        assert result[0][1] == ""

    def test_minimum_chars_preserved(self) -> None:
        """即使预算耗尽，每个结果至少保留 _MIN_RESULT_CHARS 字符。"""
        state = ContextBudgetState(
            max_context_tokens=100000,
            soft_limit_ratio=0.75,  # soft = 75000
        )
        # 预算已经用完（prompt 已超 soft limit）
        state.current_prompt_tokens = 80000
        state.latest_completion_tokens = 500
        large_result = "z" * 50000
        pairs: list[tuple[dict[str, object], str]] = [({"id": "c1"}, large_result)]
        result, capped = ToolResultBudgetCapper.cap_results_for_budget(pairs, state)
        assert capped
        # 截断后保留至少 4000 chars（_MIN_RESULT_CHARS）
        # 加上 CONTEXT_BUDGET_TRUNCATED 注释
        assert result[0][1].startswith("z" * 4000)


# ========== run_and_wait WARNING 收集测试 ==========


@pytest.mark.asyncio
async def test_run_and_wait_collects_warnings() -> None:
    """run_and_wait 正确收集 WARNING 事件到 AgentResult.warnings。"""
    # 第一轮截断 → 触发续写 warning
    truncated_batch = [
        content_complete("partial"),
        done_event(summary={
            "truncated": True,
            "usage": {"prompt_tokens": 80000, "completion_tokens": 500},
        }),
    ]
    final_batch = [
        content_complete("done"),
        done_event(summary={
            "truncated": False,
            "usage": {"prompt_tokens": 50000, "completion_tokens": 500},
        }),
    ]
    runner = _RunnerStub([truncated_batch, final_batch])

    agent = AsyncAgent(
        runner,
        running_config=AgentRunningConfig(
            max_context_tokens=100000,
            budget_soft_limit_ratio=0.75,
            max_continuations=3,
        ),
    )

    result = await agent.run_and_wait("goal")
    assert result.content == "partialdone"
    # 应收集到续写 warning
    assert len(result.warnings) > 0
    assert any("截断" in w or "续写" in w for w in result.warnings)


# ========== AgentResult.warnings 字段测试 ==========


class TestAgentResultWarnings:
    """AgentResult warnings 字段测试。"""

    def test_default_empty_warnings(self) -> None:
        """不传 warnings 时默认为空列表。"""
        result = AgentResult(
            content="ok",
            tool_calls=[],
            errors=[],
            messages=[],
        )
        assert result.warnings == []

    def test_explicit_warnings(self) -> None:
        """显式传入 warnings。"""
        result = AgentResult(
            content="ok",
            tool_calls=[],
            errors=[],
            messages=[],
            warnings=["w1", "w2"],
        )
        assert result.warnings == ["w1", "w2"]
