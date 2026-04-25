"""sse_parser 模块单元测试。

该测试文件针对 SSE 解析器的低频分支进行覆盖，包括：
- `data: [DONE]` 终止逻辑与尾行处理
- 非 JSON payload 的跳过与调试日志分支
- 多 choices / finish_reason / tool_calls=None 的分支行为
- 工具调用增量与最终组装校验分支
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator, AsyncIterator, Dict, List, cast

import pytest

from dayu.engine.async_openai_runner import AsyncOpenAIRunnerRunningConfig
from dayu.contracts.cancellation import CancelledError as EngineCancelledError, CancellationToken
from dayu.engine.events import EventType, StreamEvent
from dayu.log import Log
from dayu.engine.sse_parser import SSEStreamParser

if TYPE_CHECKING:
    from aiohttp import ClientResponse


@dataclass
class _RunningConfigStub(AsyncOpenAIRunnerRunningConfig):
    """SSE 解析器运行配置桩对象。"""

    debug_sse: bool = False
    debug_sse_sample_rate: float = 1.0
    debug_sse_throttle_sec: float = 0.0
    debug_tool_delta: bool = False
    stream_idle_heartbeat_sec: float | None = 10.0


class _ChunkedContentStub:
    """模拟 aiohttp `response.content` 的 chunk 迭代器。"""

    def __init__(self, chunks: List[bytes], delays: List[float] | None = None) -> None:
        """初始化 chunk 容器。

        Args:
            chunks: 预置的网络分片序列。
            delays: 每个 chunk 产出前的等待秒数。

        Returns:
            None。

        Raises:
            无。
        """
        self._chunks = chunks
        self._delays = delays or [0.0] * len(chunks)

    async def iter_chunked(self, _: int) -> AsyncIterator[bytes]:
        """按顺序异步产出 chunk。

        Args:
            _: 与真实接口兼容的 chunk 大小参数。

        Returns:
            AsyncIterator[bytes]: chunk 异步迭代器。

        Raises:
            无。
        """
        for delay, chunk in zip(self._delays, self._chunks):
            if delay > 0:
                await asyncio.sleep(delay)
            yield chunk


class _ResponseStub:
    """模拟 aiohttp 响应对象，仅保留 `content` 字段。"""

    def __init__(self, chunks: List[bytes], delays: List[float] | None = None) -> None:
        """构建响应桩对象。

        Args:
            chunks: 将被 `content.iter_chunked` 依次返回的字节串。
            delays: 每个 chunk 产出前的等待秒数。

        Returns:
            None。

        Raises:
            无。
        """
        self.content = _ChunkedContentStub(chunks, delays=delays)


class _LogCollector:
    """日志收集器，用于替换 `Log.debug` / `Log.warn`。"""

    def __init__(self) -> None:
        """初始化日志收集器。

        Args:
            无。

        Returns:
            None。

        Raises:
            无。
        """
        self.messages: List[str] = []

    def debug(self, message: str, *, module: str = "APP") -> None:
        """收集 debug 级别日志。

        Args:
            message: 日志文本。
            module: 模块名（为兼容 `Log.debug` 签名而保留）。

        Returns:
            None。

        Raises:
            无。
        """
        del module
        self.messages.append(message)

    def warn(self, message: str, *, module: str = "APP") -> None:
        """收集 warning 级别日志。

        Args:
            message: 日志文本。
            module: 模块名（为兼容 `Log.warn` 签名而保留）。

        Returns:
            None。

        Raises:
            无。
        """
        del module
        self.messages.append(message)


async def _collect_events(async_iter: AsyncIterator[StreamEvent]) -> List[StreamEvent]:
    """收集异步事件迭代器中的所有事件。

    Args:
        async_iter: 异步事件迭代器。

    Returns:
        List[StreamEvent]: 收集到的事件列表。

    Raises:
        无。
    """
    return [event async for event in async_iter]


def _parse_stream(
    parser: SSEStreamParser,
    response: object,
) -> AsyncGenerator[StreamEvent, None]:
    """把测试响应桩收窄为解析器可接受的流生成器。"""

    return cast(
        AsyncGenerator[StreamEvent, None],
        parser.parse_stream(cast("ClientResponse", response)),
    )


@pytest.mark.asyncio
async def test_parse_stream_stops_after_done_even_with_following_chunks() -> None:
    """验证收到 `[DONE]` 后会提前停止处理后续 chunk。

    Args:
        无。

    Returns:
        None。

    Raises:
        AssertionError: 当解析结果与预期不一致时抛出。
    """
    parser = SSEStreamParser(
        name="test",
        request_id="req_done_break",
        running_config=_RunningConfigStub(),
    )
    response = _ResponseStub(
        [
            b"data: [DONE]\n",
            b'data: {"choices":[{"delta":{"content":"late"}}]}\n',
        ]
    )

    events = await _collect_events(_parse_stream(parser, response))
    result = parser.get_result()

    assert events == []
    assert result.done_received is True
    assert result.content == ""


@pytest.mark.asyncio
async def test_parse_stream_handles_trailing_line_variants_without_newline() -> None:
    """验证尾行无换行时 `[DONE]` 与普通 payload 都能被正确处理。

    Args:
        无。

    Returns:
        None。

    Raises:
        AssertionError: 当尾行解析行为不符合预期时抛出。
    """
    done_parser = SSEStreamParser(
        name="test",
        request_id="req_trailing_done",
        running_config=_RunningConfigStub(),
    )
    done_events = await _collect_events(_parse_stream(done_parser, _ResponseStub([b"data: [DONE]"])))
    done_result = done_parser.get_result()

    payload_parser = SSEStreamParser(
        name="test",
        request_id="req_trailing_payload",
        running_config=_RunningConfigStub(),
    )
    payload = json.dumps({"choices": [{"delta": {"content": "tail"}}]})
    payload_events = await _collect_events(
        _parse_stream(payload_parser, _ResponseStub([f"data: {payload}".encode("utf-8")]))
    )
    payload_result = payload_parser.get_result()

    assert done_events == []
    assert done_result.done_received is True

    assert len(payload_events) == 1
    assert payload_events[0].type == EventType.CONTENT_DELTA
    assert payload_events[0].data == "tail"
    assert payload_result.content == "tail"


@pytest.mark.asyncio
async def test_parse_stream_supports_multiline_data_event() -> None:
    """验证一个 SSE 事件包含多行 `data:` 时可正确聚合解析。

    Args:
        无。

    Returns:
        None。

    Raises:
        AssertionError: 当多行 data 聚合解析不符合预期时抛出。
    """
    parser = SSEStreamParser(
        name="test",
        request_id="req_multiline_data",
        running_config=_RunningConfigStub(),
    )
    response = _ResponseStub(
        [
            b'data: {"choices":[{"delta":\n',
            b'data: {"content":"multi"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    events = await _collect_events(_parse_stream(parser, response))
    result = parser.get_result()

    assert len(events) == 1
    assert events[0].type == EventType.CONTENT_DELTA
    assert events[0].data == "multi"
    assert result.content == "multi"
    assert result.done_received is True


@pytest.mark.asyncio
async def test_handle_payload_logs_and_skips_non_json_when_debug_enabled(monkeypatch: Any) -> None:
    """验证非法 JSON payload 会记录协议错误，而不是静默跳过。

    Args:
        monkeypatch: pytest monkeypatch 工具，用于替换日志函数。

    Returns:
        None。

    Raises:
        AssertionError: 当日志未记录或产出事件不符合预期时抛出。
    """
    parser = SSEStreamParser(
        name="test",
        request_id="req_non_json",
        running_config=_RunningConfigStub(debug_sse=True, debug_sse_sample_rate=1.0),
    )

    events = await _collect_events(parser._handle_payload("this is not json"))

    assert events == []
    result = parser.get_result()
    assert result.protocol_errors[0]["error_type"] == "response_error"


@pytest.mark.asyncio
async def test_handle_payload_logs_multi_choices_finish_reason_and_delta_absence(monkeypatch: Any) -> None:
    """验证多 choice 警告、finish_reason 调试日志和无 delta 场景。

    Args:
        monkeypatch: pytest monkeypatch 工具，用于替换日志函数。

    Returns:
        None。

    Raises:
        AssertionError: 当事件或日志不符合预期时抛出。
    """
    warn_collector = _LogCollector()
    debug_collector = _LogCollector()
    monkeypatch.setattr(Log, "warn", warn_collector.warn)
    monkeypatch.setattr(Log, "debug", debug_collector.debug)

    parser = SSEStreamParser(
        name="test",
        request_id="req_choices",
        running_config=_RunningConfigStub(debug_sse=True, debug_sse_sample_rate=1.0),
    )

    multi_choice_payload = json.dumps(
        {
            "choices": [
                {"finish_reason": "stop", "delta": {"content": "A"}},
                {"delta": {"content": "B"}},
            ]
        }
    )
    events = await _collect_events(parser._handle_payload(multi_choice_payload))

    no_delta_payload = json.dumps({"choices": [{"finish_reason": None}]})
    no_delta_events = await _collect_events(parser._handle_payload(no_delta_payload))

    assert len(events) == 1
    assert events[0].type == EventType.CONTENT_DELTA
    assert events[0].data == "A"
    assert no_delta_events == []

    assert any("多个 choices" in message for message in warn_collector.messages)
    assert any("finish_reason" in message for message in debug_collector.messages)


@pytest.mark.asyncio
async def test_handle_payload_logs_tool_calls_none_when_debug_enabled(monkeypatch: Any) -> None:
    """验证 `tool_calls=None` 分支会被忽略并输出调试日志。

    Args:
        monkeypatch: pytest monkeypatch 工具，用于替换日志函数。

    Returns:
        None。

    Raises:
        AssertionError: 当日志或事件不符合预期时抛出。
    """
    debug_collector = _LogCollector()
    monkeypatch.setattr(Log, "debug", debug_collector.debug)

    parser = SSEStreamParser(
        name="test",
        request_id="req_tool_calls_none",
        running_config=_RunningConfigStub(debug_sse=True, debug_sse_sample_rate=1.0),
    )
    payload = json.dumps({"choices": [{"delta": {"tool_calls": None}}]})

    events = await _collect_events(parser._handle_payload(payload))

    assert events == []
    assert any("tool_calls=None" in message for message in debug_collector.messages)


@pytest.mark.asyncio
async def test_handle_payload_records_protocol_error_when_tool_calls_not_list() -> None:
    """验证 `tool_calls` 非列表时记录协议错误。"""
    parser = SSEStreamParser(
        name="test",
        request_id="req_tool_calls_invalid",
        running_config=_RunningConfigStub(),
    )
    payload = json.dumps({"choices": [{"delta": {"tool_calls": {}}}]})

    events = await _collect_events(parser._handle_payload(payload))
    result = parser.get_result()

    assert events == []
    assert result.protocol_errors[0]["error_type"] == "tool_call_invalid"


@pytest.mark.asyncio
async def test_handle_payload_skips_invalid_tool_call_entry_but_continues_following_entries() -> None:
    """非法 tool_call entry 不应阻断同一 chunk 中后续合法 entry 的处理。"""

    parser = SSEStreamParser(
        name="test",
        request_id="req_tool_calls_mixed_entries",
        running_config=_RunningConfigStub(),
    )
    payload = json.dumps(
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            "invalid-entry",
                            {
                                "index": 0,
                                "id": "tc_1",
                                "function": {
                                    "name": "search",
                                    "arguments": "{\"q\":\"abc\"}",
                                },
                            },
                        ]
                    }
                }
            ]
        }
    )

    events = await _collect_events(parser._handle_payload(payload))
    result = parser.get_result()

    assert [event.type for event in events] == [EventType.TOOL_CALL_START, EventType.TOOL_CALL_DELTA]
    assert result.protocol_errors[0]["error_type"] == "tool_call_incomplete"
    assert "index 0" in result.protocol_errors[0]["body"]


@pytest.mark.asyncio
async def test_handle_tool_call_delta_covers_missing_index_and_debug_logs(monkeypatch: Any) -> None:
    """验证工具增量在缺索引与调试模式下的行为。

    Args:
        monkeypatch: pytest monkeypatch 工具，用于替换日志函数。

    Returns:
        None。

    Raises:
        AssertionError: 当事件或日志不符合预期时抛出。
    """
    warn_collector = _LogCollector()
    debug_collector = _LogCollector()
    monkeypatch.setattr(Log, "warn", warn_collector.warn)
    monkeypatch.setattr(Log, "debug", debug_collector.debug)

    parser = SSEStreamParser(
        name="test",
        request_id="req_tool_delta",
        running_config=_RunningConfigStub(debug_tool_delta=True),
    )

    missing_index_events = await _collect_events(parser._handle_tool_call_delta({"id": "tc_1"}))

    full_events = await _collect_events(
        parser._handle_tool_call_delta(
            {
                "index": 0,
                "id": "tc_2",
                "function": {"name": "read_file", "arguments": "{}"},
            }
        )
    )

    assert missing_index_events == []
    assert any("index" in message and ("缺失" in message or "异常" in message) for message in warn_collector.messages)

    assert [event.type for event in full_events] == [EventType.TOOL_CALL_START, EventType.TOOL_CALL_DELTA]
    assert any("记录 tool_call_id" in message for message in debug_collector.messages)
    assert any("记录工具名" in message for message in debug_collector.messages)


@pytest.mark.asyncio
async def test_handle_tool_call_delta_replays_buffered_prefix_when_id_name_arrive_late() -> None:
    """验证 `id/name` 晚于 arguments 到达时，会补发已缓存的参数前缀。"""
    parser = SSEStreamParser(
        name="test",
        request_id="req_tool_delta_replay",
        running_config=_RunningConfigStub(),
    )

    early_args_events = await _collect_events(
        parser._handle_tool_call_delta(
            {
                "index": 0,
                "function": {"arguments": '{"a":'},
            }
        )
    )
    late_identity_events = await _collect_events(
        parser._handle_tool_call_delta(
            {
                "index": 0,
                "id": "tc_3",
                "function": {"name": "tool"},
            }
        )
    )
    trailing_events = await _collect_events(
        parser._handle_tool_call_delta(
            {
                "index": 0,
                "function": {"arguments": "1}"},
            }
        )
    )
    result = parser.get_result()

    assert early_args_events == []
    assert [event.type for event in late_identity_events] == [
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_DELTA,
    ]
    assert late_identity_events[1].data["arguments_delta"] == '{"a":'
    assert [event.type for event in trailing_events] == [EventType.TOOL_CALL_DELTA]
    assert trailing_events[0].data["arguments_delta"] == "1}"
    assert result.tool_calls == [
        {
            "id": "tc_3",
            "name": "tool",
            "arguments": {"a": 1},
            "index_in_iteration": 0,
        }
    ]


@pytest.mark.asyncio
async def test_handle_tool_call_delta_records_protocol_error_for_invalid_function_or_arguments() -> None:
    """验证流式 tool_call 子结构非法时记录协议错误，而不是抛异常。"""
    parser = SSEStreamParser(
        name="test",
        request_id="req_tool_delta_invalid",
        running_config=_RunningConfigStub(),
    )

    invalid_function_events = await _collect_events(
        parser._handle_tool_call_delta(
            {
                "index": 0,
                "id": "tc_bad_func",
                "function": None,
            }
        )
    )
    invalid_arguments_events = await _collect_events(
        parser._handle_tool_call_delta(
            {
                "index": 1,
                "id": "tc_bad_args",
                "function": {"name": "tool", "arguments": {"a": 1}},
            }
        )
    )
    result = parser.get_result()

    assert invalid_function_events == []
    assert invalid_arguments_events == []
    assert len(result.protocol_errors) == 2
    assert result.protocol_errors[0]["error_type"] == "tool_call_incomplete"
    assert "missing function object" in result.protocol_errors[0]["body"]
    assert "arguments type is dict" in result.protocol_errors[1]["body"]


@pytest.mark.asyncio
async def test_parse_stream_logs_idle_heartbeat_before_next_chunk(monkeypatch: Any) -> None:
    """验证流式响应长时间空闲时会输出心跳日志。

    Args:
        monkeypatch: pytest monkeypatch 工具，用于替换日志函数。

    Returns:
        None。

    Raises:
        AssertionError: 当心跳日志未输出时抛出。
    """

    debug_collector = _LogCollector()
    monkeypatch.setattr(Log, "debug", debug_collector.debug)

    parser = SSEStreamParser(
        name="test",
        request_id="req_idle_heartbeat",
        running_config=_RunningConfigStub(stream_idle_heartbeat_sec=0.01),
    )
    response = _ResponseStub(
        [b'data: {"choices":[{"delta":{"content":"late"}}]}\n\n', b"data: [DONE]\n\n"],
        delays=[0.02, 0.0],
    )

    events = await _collect_events(_parse_stream(parser, response))
    result = parser.get_result()

    assert len(events) == 1
    assert events[0].type == EventType.CONTENT_DELTA
    assert result.content == "late"
    assert any("SSE 流空闲等待中" in message for message in debug_collector.messages)


@pytest.mark.asyncio
async def test_parse_stream_ignores_empty_data_heartbeat() -> None:
    """验证空 `data:` heartbeat 不会被错误标记为坏 JSON。"""
    parser = SSEStreamParser(
        name="test",
        request_id="req_empty_data_heartbeat",
        running_config=_RunningConfigStub(),
    )
    response = _ResponseStub(
        [
            b"data:\n\n",
            b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    events = await _collect_events(_parse_stream(parser, response))
    result = parser.get_result()

    assert [event.type for event in events] == [EventType.CONTENT_DELTA]
    assert events[0].data == "ok"
    assert result.protocol_errors == []
    assert result.done_received is True


@pytest.mark.asyncio
async def test_parse_stream_preserves_utf8_multibyte_across_chunks() -> None:
    """验证 UTF-8 多字节字符跨 chunk 时不会丢字。"""
    parser = SSEStreamParser(
        name="test",
        request_id="req_utf8_split",
        running_config=_RunningConfigStub(),
    )
    payload = 'data: {"choices":[{"delta":{"content":"你好"}}]}\n\n'.encode("utf-8")
    split_at = payload.index("你".encode("utf-8")) + 1
    response = _ResponseStub([payload[:split_at], payload[split_at:], b"data: [DONE]\n\n"])

    events = await _collect_events(_parse_stream(parser, response))
    result = parser.get_result()

    assert [event.data for event in events] == ["你好"]
    assert result.content == "你好"


@pytest.mark.asyncio
async def test_parse_stream_invalid_utf8_becomes_protocol_error() -> None:
    """验证非法 UTF-8 被上报为协议错误。"""
    parser = SSEStreamParser(
        name="test",
        request_id="req_invalid_utf8",
        running_config=_RunningConfigStub(),
    )
    response = _ResponseStub([b'data: {"choices":[{"delta":{"content":"\xff"}}]}\n\n'])

    events = await _collect_events(_parse_stream(parser, response))
    result = parser.get_result()

    assert events == []
    assert result.protocol_errors[0]["error_type"] == "response_error"


@pytest.mark.asyncio
async def test_parse_stream_cancels_pending_task_on_early_close() -> None:
    """验证生成器提前关闭时会取消未完成的 chunk 读取 task，不留悬空 task。

    Args:
        无。

    Returns:
        None。

    Raises:
        AssertionError: 当关闭后仍存在悬空 task 时抛出。
    """

    class _SlowContentStub:
        """模拟一个极慢的 chunk 源，确保 pending_chunk_task 处于等待中。"""

        @staticmethod
        async def iter_chunked(_: int) -> AsyncIterator[bytes]:
            """产出前等待极长时间，模拟空闲挂起。"""
            await asyncio.sleep(100)
            yield b"data: [DONE]\n"

    class _SlowResponseStub:
        content = _SlowContentStub()

    parser = SSEStreamParser(
        name="test",
        request_id="req_early_close",
        running_config=_RunningConfigStub(stream_idle_heartbeat_sec=0.01),
    )
    gen = _parse_stream(parser, _SlowResponseStub())

    # 启动生成器并等一小段时间让 pending_chunk_task 被创建
    try:
        await asyncio.wait_for(gen.__anext__(), timeout=0.05)
    except (asyncio.TimeoutError, StopAsyncIteration):
        pass

    # 关闭生成器
    await gen.aclose()

    # 检查不应有悬空 task
    await asyncio.sleep(0.01)
    dangling = [
        t
        for t in asyncio.all_tasks()
        if not t.done() and t is not asyncio.current_task()
    ]
    assert len(dangling) == 0, f"发现悬空 task: {[t.get_coro() for t in dangling]}"


@pytest.mark.asyncio
async def test_parse_stream_raises_cancelled_during_idle_heartbeat_wait() -> None:
    """验证空闲 heartbeat 等待期间会优先响应取消。"""

    token = CancellationToken()

    class _SlowContentStub:
        @staticmethod
        async def iter_chunked(_: int) -> AsyncIterator[bytes]:
            await asyncio.sleep(100)
            yield b"data: [DONE]\n"

    class _SlowResponseStub:
        content = _SlowContentStub()

    parser = SSEStreamParser(
        name="test",
        request_id="req_cancel_idle_wait",
        running_config=_RunningConfigStub(stream_idle_heartbeat_sec=0.01),
        cancellation_token=token,
    )
    consume_task = asyncio.create_task(_collect_events(_parse_stream(parser, _SlowResponseStub())))

    await asyncio.sleep(0.05)
    token.cancel()
    with pytest.raises(EngineCancelledError):
        await consume_task


@pytest.mark.asyncio
async def test_parse_stream_outer_task_cancel_cleans_inflight_next_chunk_task() -> None:
    """验证外层任务取消时，下一块读取对应的子任务也会被取消收口。"""

    token = CancellationToken()
    started = asyncio.Event()
    inner_cancelled = asyncio.Event()

    class _SlowContentStub:
        @staticmethod
        async def iter_chunked(_: int) -> AsyncIterator[bytes]:
            started.set()
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                inner_cancelled.set()
                raise
            yield b"data: [DONE]\n"

    class _SlowResponseStub:
        content = _SlowContentStub()

    parser = SSEStreamParser(
        name="test",
        request_id="req_outer_cancel_next_chunk",
        running_config=_RunningConfigStub(stream_idle_heartbeat_sec=None),
        cancellation_token=token,
    )
    consume_task = asyncio.create_task(_collect_events(_parse_stream(parser, _SlowResponseStub())))

    await started.wait()
    consume_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consume_task
    assert inner_cancelled.is_set()


@pytest.mark.asyncio
async def test_parse_stream_raises_cancelled_before_next_chunk_and_leaves_no_dangling_task() -> None:
    """验证读到首个 chunk 后，下一次 chunk 等待期间取消会清理 pending task。"""

    token = CancellationToken()
    payload = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
    parser = SSEStreamParser(
        name="test",
        request_id="req_cancel_next_chunk",
        running_config=_RunningConfigStub(stream_idle_heartbeat_sec=0.01),
        cancellation_token=token,
    )
    response = _ResponseStub(
        [payload, b"data: [DONE]\n\n"],
        delays=[0.0, 100.0],
    )

    gen = _parse_stream(parser, response)
    first_event = await gen.__anext__()
    assert first_event.type == EventType.CONTENT_DELTA

    token.cancel()
    with pytest.raises(EngineCancelledError):
        await gen.__anext__()

    await asyncio.sleep(0.01)
    dangling = [
        task
        for task in asyncio.all_tasks()
        if not task.done() and task is not asyncio.current_task()
    ]
    assert len(dangling) == 0, f"发现悬空 task: {[task.get_coro() for task in dangling]}"


@pytest.mark.asyncio
async def test_parse_stream_success_unregisters_cancellation_callback() -> None:
    """验证成功解析后不会把取消回调残留在复用 token 上。"""

    token = CancellationToken()
    parser = SSEStreamParser(
        name="test",
        request_id="req_success_unregister_callback",
        running_config=_RunningConfigStub(stream_idle_heartbeat_sec=None),
        cancellation_token=token,
    )
    response = _ResponseStub([b"data: [DONE]\n\n"])

    events = [event async for event in _parse_stream(parser, response)]

    assert events == []
    assert token._callbacks == []


def test_assemble_tool_calls_reports_missing_name_and_non_object_arguments() -> None:
    """验证工具调用组装阶段的 name 缺失与参数非对象错误。

    Args:
        无。

    Returns:
        None。

    Raises:
        AssertionError: 当组装输出与预期不一致时抛出。
    """
    parser = SSEStreamParser(
        name="test",
        request_id="req_assemble",
        running_config=_RunningConfigStub(),
    )
    parser._tool_calls_buffer = {
        0: {"id": "tc_missing_name", "name": "", "arguments_buf": "{}"},
        1: {"id": "tc_not_object", "name": "tool", "arguments_buf": "[]"},
        2: {"id": "tc_valid", "name": "tool", "arguments_buf": '{"a": 1}'},
    }

    tool_calls, validation_errors = parser._assemble_tool_calls()

    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "tc_valid"
    assert any("missing name" in message for message in validation_errors)
    assert any("arguments is not object" in message for message in validation_errors)


def test_assemble_tool_calls_rejects_non_string_or_empty_id() -> None:
    """验证 tool_call id 必须为非空字符串；null / 整数 / 空串均应被拦截。

    Args:
        无。

    Returns:
        None。

    Raises:
        AssertionError: 当非字符串或空字符串 id 未被拦截时抛出。
    """

    parser = SSEStreamParser(
        name="test",
        request_id="req_assemble_id",
        running_config=_RunningConfigStub(),
    )
    parser._tool_calls_buffer = {
        0: {"id": None, "name": "tool", "arguments_buf": "{}"},
        1: {"id": 123, "name": "tool", "arguments_buf": "{}"},
        2: {"id": "", "name": "tool", "arguments_buf": "{}"},
        3: {"id": "tc_valid", "name": "tool", "arguments_buf": '{"a": 1}'},
    }

    tool_calls, validation_errors = parser._assemble_tool_calls()

    assert len(tool_calls) == 1
    assert tool_calls[0]["id"] == "tc_valid"
    missing_id_count = sum(1 for message in validation_errors if "missing id" in message)
    assert missing_id_count == 3


@pytest.mark.asyncio
async def test_handle_tool_call_delta_ignores_non_string_id() -> None:
    """验证 delta 阶段遇到 null / 非字符串 id 时不会写入 entry。

    Args:
        无。

    Returns:
        None。

    Raises:
        AssertionError: 当非字符串 id 被写入 entry 时抛出。
    """

    parser = SSEStreamParser(
        name="test",
        request_id="req_delta_non_string_id",
        running_config=_RunningConfigStub(),
    )

    await _collect_events(
        parser._handle_tool_call_delta({"index": 0, "id": None, "function": {"name": "tool"}})
    )
    await _collect_events(
        parser._handle_tool_call_delta({"index": 0, "id": 123, "function": {"name": "tool"}})
    )
    assert parser._tool_calls_buffer[0]["id"] is None

    await _collect_events(
        parser._handle_tool_call_delta({"index": 0, "id": "tc_real", "function": {}})
    )
    assert parser._tool_calls_buffer[0]["id"] == "tc_real"


# ---------------------------------------------------------------------------
# finish_reason=length → truncated 标记 + WARN 日志
# ---------------------------------------------------------------------------


def test_handle_payload_sets_truncated_on_finish_reason_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 finish_reason=length 时设置 stream_state["truncated"]=True 并打印 WARN 日志。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        None。

    Raises:
        AssertionError: 当 truncated 标记或日志内容与预期不符时抛出。
    """

    warn_messages: list[str] = []
    monkeypatch.setattr(Log, "warn", lambda msg, *, module="APP": warn_messages.append(msg))

    parser = SSEStreamParser(
        name="test",
        request_id="req_length",
        running_config=_RunningConfigStub(debug_sse=True),
    )

    # 构造含 finish_reason=length 的 SSE payload
    payload = json.dumps({
        "choices": [
            {
                "index": 0,
                "delta": {"content": "partial"},
                "finish_reason": "length",
            }
        ]
    })

    # 收集事件（_handle_payload 是异步生成器）
    import asyncio

    async def _collect():
        return [event async for event in parser._handle_payload(payload)]

    events = asyncio.run(_collect())

    # 验证 stream_state
    assert parser._stream_state.get("finish_reason") == "length"
    assert parser._stream_state.get("truncated") is True
    # tool_calls_finished 不应被设置
    assert parser._stream_state.get("tool_calls_finished") is not True

    # 验证内容增量事件
    content_events = [e for e in events if e.type == EventType.CONTENT_DELTA]
    assert len(content_events) == 1
    assert content_events[0].data == "partial"

    # 验证 WARN 日志
    assert len(warn_messages) >= 1
    assert any("finish_reason=length" in msg for msg in warn_messages)


def test_handle_payload_no_truncated_on_finish_reason_stop() -> None:
    """验证 finish_reason=stop 时不设置 truncated 标记。

    Args:
        无。

    Returns:
        None。

    Raises:
        AssertionError: 当 truncated 标记被意外设置时抛出。
    """

    parser = SSEStreamParser(
        name="test",
        request_id="req_stop",
        running_config=_RunningConfigStub(debug_sse=True),
    )

    payload = json.dumps({
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ]
    })

    import asyncio

    async def _collect():
        return [event async for event in parser._handle_payload(payload)]

    asyncio.run(_collect())

    assert parser._stream_state.get("finish_reason") == "stop"
    assert parser._stream_state.get("truncated") is not True


# ── reasoning_content（thinking 模式思维链）────────────────────────────


@pytest.mark.asyncio
async def test_handle_payload_extracts_reasoning_content_delta() -> None:
    """验证 delta 包含 reasoning_content 时，产出 REASONING_DELTA 事件并累积到缓冲区。"""
    parser = SSEStreamParser(
        name="test",
        request_id="req_rc",
        running_config=_RunningConfigStub(),
    )

    payload = json.dumps({
        "choices": [{"delta": {"reasoning_content": "思考中..."}}]
    })

    events = await _collect_events(parser._handle_payload(payload))

    assert len(events) == 1
    assert events[0].type == EventType.REASONING_DELTA
    assert events[0].data == "思考中..."
    assert parser._reasoning_content_buffer == ["思考中..."]


@pytest.mark.asyncio
async def test_reasoning_content_accumulated_across_multiple_deltas() -> None:
    """验证多条 reasoning_content delta 正确拼接到 get_result()。"""
    parser = SSEStreamParser(
        name="test",
        request_id="req_rc_multi",
        running_config=_RunningConfigStub(),
    )

    chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"part1"}}]}\n\n',
        b'data: {"choices":[{"delta":{"reasoning_content":"part2"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n',
        b"data: [DONE]\n",
    ]
    response = _ResponseStub(chunks)

    events = await _collect_events(_parse_stream(parser, response))
    result = parser.get_result()

    # 事件顺序：REASONING_DELTA × 2, CONTENT_DELTA × 1
    event_types = [e.type for e in events]
    assert event_types == [
        EventType.REASONING_DELTA,
        EventType.REASONING_DELTA,
        EventType.CONTENT_DELTA,
    ]
    assert result.reasoning_content == "part1part2"
    assert result.content == "answer"


@pytest.mark.asyncio
async def test_no_reasoning_content_yields_empty_string() -> None:
    """验证不含 reasoning_content 时，结果字段为空字符串且无 REASONING_DELTA 事件。"""
    parser = SSEStreamParser(
        name="test",
        request_id="req_no_rc",
        running_config=_RunningConfigStub(),
    )

    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b"data: [DONE]\n",
    ]
    response = _ResponseStub(chunks)

    events = await _collect_events(_parse_stream(parser, response))
    result = parser.get_result()

    assert result.reasoning_content == ""
    assert all(e.type != EventType.REASONING_DELTA for e in events)


@pytest.mark.asyncio
async def test_reasoning_content_empty_string_ignored() -> None:
    """验证 delta 中 reasoning_content 为空字符串时不产出事件。"""
    parser = SSEStreamParser(
        name="test",
        request_id="req_rc_empty",
        running_config=_RunningConfigStub(),
    )

    payload = json.dumps({
        "choices": [{"delta": {"reasoning_content": "", "content": "hi"}}]
    })

    events = await _collect_events(parser._handle_payload(payload))

    assert len(events) == 1
    assert events[0].type == EventType.CONTENT_DELTA
    assert parser._reasoning_content_buffer == []


@pytest.mark.asyncio
async def test_indented_data_line_not_parsed() -> None:
    """T5: 前导空格的行不应被识别为 SSE data 字段（SSE 规范合规）。

    移除 lstrip() 后，带前导空格的 "  data: ..." 行应被忽略，
    只有行首即为 "data:" 的行才会被解析。
    """
    parser = SSEStreamParser(
        name="test",
        request_id="req_indented",
        running_config=_RunningConfigStub(),
    )
    payload = json.dumps({"choices": [{"delta": {"content": "visible"}}]})
    # 正常 data 行 + 一行带前导空格的 "data:" + [DONE]
    response = _ResponseStub([
        f"data: {payload}\n\n".encode("utf-8"),
        f"  data: {payload}\n\n".encode("utf-8"),
        b"data: [DONE]\n\n",
    ])

    events = await _collect_events(_parse_stream(parser, response))
    result = parser.get_result()

    # 只有第一行无前导空格的 data 被解析为内容
    content_deltas = [e for e in events if e.type == EventType.CONTENT_DELTA]
    assert len(content_deltas) == 1
    assert content_deltas[0].data == "visible"
    assert result.done_received is True


@pytest.mark.asyncio
async def test_parse_stream_gemini_parallel_tool_calls_without_index() -> None:
    """验证 Gemini 风格的并行工具调用（无 index 字段）在 parse_stream 层正确解析。

    Gemini OpenAI 兼容模式不发 index 字段，每个 tool call 在一个 delta 中一次给全。
    本测试模拟一个 chunk 内并行调用 3 个工具的场景，验证上游自动补齐 index 后
    下游正确组装出 3 个独立的工具调用。
    """
    parser = SSEStreamParser(
        name="gemini",
        request_id="req_gemini_parallel",
        running_config=_RunningConfigStub(),
    )

    # 模拟 Gemini 并行 3 个 tool call（真实抓包结构，无 index 字段）
    payload = json.dumps({
        "choices": [{
            "delta": {
                "tool_calls": [
                    {
                        "id": "function-call-001",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"杭州天气"}',
                        },
                    },
                    {
                        "id": "function-call-002",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"北京天气"}',
                        },
                    },
                    {
                        "id": "function-call-003",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"query":"上海天气"}',
                        },
                    },
                ],
            },
            "finish_reason": "tool_calls",
        }],
    })

    response = _ResponseStub([
        f"data: {payload}\n\n".encode("utf-8"),
        b"data: [DONE]\n\n",
    ])

    events = await _collect_events(_parse_stream(parser, response))
    result = parser.get_result()

    # 应产出 3 组 tool_call_start + tool_call_delta 事件
    start_events = [e for e in events if e.type == EventType.TOOL_CALL_START]
    delta_events = [e for e in events if e.type == EventType.TOOL_CALL_DELTA]
    assert len(start_events) == 3
    assert len(delta_events) == 3

    # 验证组装结果：3 个独立的工具调用，各自参数正确
    assert len(result.tool_calls) == 3
    assert result.tool_calls[0]["id"] == "function-call-001"
    assert result.tool_calls[0]["arguments"] == {"query": "杭州天气"}
    assert result.tool_calls[1]["id"] == "function-call-002"
    assert result.tool_calls[1]["arguments"] == {"query": "北京天气"}
    assert result.tool_calls[2]["id"] == "function-call-003"
    assert result.tool_calls[2]["arguments"] == {"query": "上海天气"}

    # 无 protocol_errors 和 validation_errors
    assert result.protocol_errors == []
    assert result.validation_errors == []
