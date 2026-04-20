"""
SSE (Server-Sent Events) 流式响应解析器

从 AsyncOpenAIRunner 中提取的 SSE 解析逻辑，负责：
- SSE 行缓冲与 data 行解析
- JSON payload 解析与增量事件产出（content_delta、tool_call_start/delta）
- 流结束后的工具调用组装与验证

解析器不负责工具执行和事件注解，这些由 Runner 处理。

典型使用流程：
    parser = SSEStreamParser(name="model", request_id="req_xxx", running_config=config)
    async for event in parser.parse_stream(response):
        yield annotate(event)
    result = parser.get_result()
    # 根据 result 处理工具调用、完成事件等
"""

from __future__ import annotations

import asyncio
import codecs
import inspect
import json
import time
from dataclasses import dataclass, field
from contextlib import suppress
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List

from .events import (
    StreamEvent,
    content_delta,
    reasoning_delta,
    tool_call_delta,
    tool_call_start,
)
from dayu.contracts.cancellation import CancelledError as EngineCancelledError, CancellationToken
from dayu.log import Log

if TYPE_CHECKING:
    from aiohttp import ClientResponse
    from .async_openai_runner import AsyncOpenAIRunnerRunningConfig

MODULE = "ENGINE.SSE_PARSER"
_DEFAULT_STREAM_IDLE_HEARTBEAT_SEC = 10.0


from dayu.engine.cancellation import resolve_cancellation_waiter as _resolve_cancellation_waiter
from dayu.engine.cancellation import cancel_task_and_wait as _cancel_task_and_wait


@dataclass
class SSEParseResult:
    """
    SSE 流解析的最终结果。

    Attributes:
        content: 拼接后的完整文本内容。
        tool_calls: 验证通过的工具调用列表（每项含 id/name/arguments/index_in_iteration）。
        stream_state: 流式状态字典（finish_reason、saw_choice 等）。
        done_received: 是否收到 [DONE] 标记。
        validation_errors: 工具调用验证失败的错误列表。
        protocol_errors: SSE 协议层错误列表（坏 UTF-8 / 坏 JSON / 非法 tool_calls 结构）。
        usage: 流式响应中捕获的 token 用量统计（需 stream_options.include_usage=true）。
        raw_tool_call_count: 模型尝试的工具调用数（含验证失败的），用于判断是否进入工具调用分支。
    """

    content: str = ""
    reasoning_content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stream_state: dict[str, Any] = field(default_factory=dict)
    done_received: bool = False
    validation_errors: list[str] = field(default_factory=list)
    protocol_errors: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] | None = None
    raw_tool_call_count: int = 0


def should_log_debug(
    running_config: "AsyncOpenAIRunnerRunningConfig",
    stream_state: Dict[str, Any],
) -> bool:
    """
    判断是否输出 SSE 高频调试日志（采样 + 节流）。

    Args:
        running_config: Runner 运行时配置（debug_sse / sample_rate / throttle_sec）。
        stream_state: 流式状态字典，用于记录采样计数与节流时间戳。

    Returns:
        是否应输出调试日志。
    """
    if not running_config.debug_sse:
        return False
    sample_rate = running_config.debug_sse_sample_rate
    if sample_rate <= 0:
        return False

    # 采样计数
    counter = stream_state.get("sse_debug_counter", 0) + 1
    stream_state["sse_debug_counter"] = counter

    if sample_rate < 1.0:
        sample_every = max(1, int(1 / sample_rate))
        if counter % sample_every != 0:
            return False

    # 节流
    if running_config.debug_sse_throttle_sec > 0:
        now = time.monotonic()
        last_ts = stream_state.get("sse_debug_last_ts", 0.0)
        if now - last_ts < running_config.debug_sse_throttle_sec:
            return False
        stream_state["sse_debug_last_ts"] = now

    return True


class SSEStreamParser:
    """
    SSE 流式响应解析器。

    职责边界：
    - 负责：行缓冲、payload JSON 解析、内容/工具增量累积、工具调用组装验证
    - 不负责：事件注解（trace_meta）、工具执行、完成/错误事件的产出

    使用方式：
        parser = SSEStreamParser(
            name="deepseek_chat",
            request_id="req_abc",
            running_config=running_config,
        )
        async for event in parser.parse_stream(response):
            yield annotate(event)  # Runner 负责注解
        result = parser.get_result()
    """

    def __init__(
        self,
        *,
        name: str,
        request_id: str,
        running_config: "AsyncOpenAIRunnerRunningConfig",
        cancellation_token: CancellationToken | None = None,
    ):
        """
        初始化 SSE 流解析器。

        Args:
            name: Runner 名称（用于日志前缀）。
            request_id: 请求 ID（用于日志前缀）。
            running_config: Runner 运行时配置。
        """
        self._name = name
        self._request_id = request_id
        self._running_config = running_config
        self._cancellation_token = cancellation_token
        self._log_prefix = f"[{name}][{request_id}]"

        # 内部缓冲（在 parse_stream 期间累积）
        self._content_buffer: List[str] = []
        self._reasoning_content_buffer: List[str] = []
        self._tool_calls_buffer: Dict[int, Dict] = {}
        self._stream_state: Dict[str, Any] = {
            "tool_calls_finished": False,
            "finish_reason": None,
            "saw_choice": False,
        }
        self._done_received: bool = False
        # 流式 usage 捕获（需上游设置 stream_options.include_usage=true）
        self._usage: Dict[str, Any] | None = None
        self._protocol_errors: list[dict[str, Any]] = []

    def _raise_if_cancelled(self) -> None:
        """检查当前解析流程是否已收到取消信号。"""

        if self._cancellation_token is not None:
            self._cancellation_token.raise_if_cancelled()

    def _create_cancellation_waiter(self) -> tuple[asyncio.Future[None] | None, Callable[[], None] | None]:
        """创建与当前取消令牌联动的等待 future。

        参数:
            无。

        返回值:
            tuple[asyncio.Future[None] | None, Callable[[], None] | None]:
                取消等待 future 与回调注销函数；未配置取消令牌时返回 `(None, None)`。

        异常:
            RuntimeError: 当前事件循环不可用时由底层抛出。
        """

        if self._cancellation_token is None:
            return None, None
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()

        def _on_cancel() -> None:
            loop.call_soon_threadsafe(_resolve_cancellation_waiter, waiter)

        unregister = self._cancellation_token.on_cancel(_on_cancel)
        if self._cancellation_token.is_cancelled():
            _resolve_cancellation_waiter(waiter)
        return waiter, unregister

    async def _await_or_cancel(
        self,
        awaitable: Any,
        *,
        operation_name: str,
        cancellation_waiter: asyncio.Future[None] | None,
    ) -> Any:
        """等待单个 awaitable，并在取消时优先终止。"""

        try:
            self._raise_if_cancelled()
        except Exception:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise
        if cancellation_waiter is None:
            return await awaitable

        task = asyncio.ensure_future(awaitable)
        try:
            done, _ = await asyncio.wait(
                {task, cancellation_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_waiter in done and self._cancellation_token is not None and self._cancellation_token.is_cancelled():
                await _cancel_task_and_wait(task)
                Log.info(
                    f"{self._log_prefix} 解析等待点已因取消中止: {operation_name}",
                    module=MODULE,
                )
                raise EngineCancelledError(f"operation cancelled: {operation_name}")
            return await task
        except asyncio.CancelledError:
            await _cancel_task_and_wait(task)
            raise

    def get_result(self) -> SSEParseResult:
        """
        获取解析结果（应在 parse_stream 迭代完成后调用）。

        Returns:
            SSEParseResult 包含完整内容、工具调用、状态等。
        """
        content = "".join(self._content_buffer)
        reasoning_content = "".join(self._reasoning_content_buffer)
        tool_calls, validation_errors = self._assemble_tool_calls()
        return SSEParseResult(
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            stream_state=dict(self._stream_state),
            done_received=self._done_received,
            validation_errors=validation_errors,
            protocol_errors=list(self._protocol_errors),
            usage=self._usage,
            raw_tool_call_count=len(self._tool_calls_buffer),
        )

    async def parse_stream(self, response: "ClientResponse") -> AsyncIterator[StreamEvent]:
        """
        解析 SSE 流式响应，逐个产出增量事件。

        SSE 格式：
        - 行分隔符: \\r\\n、\\r 或 \\n
        - data: {...JSON...}  ← payload 行（支持多行 data 聚合）
        - data: [DONE]        ← 结束标记
        - 空行和以 : 开头的行为注释

        Args:
            response: aiohttp 响应对象。

        Yields:
            StreamEvent — content_delta、tool_call_start、tool_call_delta 事件。
        """
        # 行缓冲：处理跨网络数据包的 SSE 行
        line_buffer = ""
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        # SSE 事件级 data 缓冲：支持一个事件包含多行 data:
        event_data_lines: list[str] = []
        chunk_iter = response.content.iter_chunked(1024).__aiter__()
        idle_log_count = 0
        idle_log_interval = self._get_stream_idle_heartbeat_sec()
        pending_chunk_task: asyncio.Task[bytes] | None = None
        cancellation_waiter, unregister_cancellation_waiter = self._create_cancellation_waiter()

        try:
            while True:
                self._raise_if_cancelled()
                if self._done_received:
                    break

                try:
                    if idle_log_interval is None:
                        chunk = await self._await_or_cancel(
                            chunk_iter.__anext__(),
                            operation_name="sse_next_chunk",
                            cancellation_waiter=cancellation_waiter,
                        )
                    else:
                        if pending_chunk_task is None:
                            pending_chunk_task = asyncio.create_task(
                                chunk_iter.__anext__()
                            )
                        wait_set: set[asyncio.Future[Any] | asyncio.Task[bytes]] = {pending_chunk_task}
                        if cancellation_waiter is not None:
                            wait_set.add(cancellation_waiter)
                        done, _ = await asyncio.wait(
                            wait_set,
                            timeout=idle_log_interval,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if cancellation_waiter is not None and cancellation_waiter in done and self._cancellation_token is not None and self._cancellation_token.is_cancelled():
                            if pending_chunk_task is not None and not pending_chunk_task.done():
                                pending_chunk_task.cancel()
                                with suppress(asyncio.CancelledError, Exception):
                                    await pending_chunk_task
                            raise EngineCancelledError("operation cancelled: sse_idle_wait")
                        if not done:
                            idle_log_count += 1
                            waited_seconds = int(idle_log_count * idle_log_interval)
                            Log.debug(
                                (
                                    f"{self._log_prefix} SSE 流空闲等待中，"
                                    f"已连续 {waited_seconds} 秒未收到新 chunk"
                                ),
                                module=MODULE,
                            )
                            continue
                        chunk = pending_chunk_task.result()
                        pending_chunk_task = None
                except StopAsyncIteration:
                    break

                # 增量 UTF-8 解码，确保多字节字符跨 chunk 时不丢字节。
                try:
                    chunk_text = decoder.decode(chunk, final=False)
                except UnicodeDecodeError as exc:
                    self._record_protocol_error(
                        "response_error",
                        "Invalid UTF-8 in SSE stream",
                        body=str(exc),
                    )
                    break
                line_buffer += chunk_text
                idle_log_count = 0

                # 统一换行符并切分为行
                normalized = line_buffer.replace("\r\n", "\n").replace("\r", "\n")
                lines = normalized.split("\n")

                # 保留不完整行用于下一个 chunk
                line_buffer = lines[-1]

                # 处理完整行
                for line_text in lines[:-1]:
                    # SSE 规范：空行表示事件结束，触发 data 聚合处理
                    if not line_text.strip():
                        async for event in self._flush_event_data_lines(event_data_lines):
                            yield event
                        if self._done_received:
                            break
                        continue

                    # 跳过注释行（SSE 规范：以 ':' 开头的行为注释）
                    if line_text.startswith(":"):
                        continue

                    # 仅聚合 data 字段；event/id/retry 等字段当前忽略
                    if line_text.startswith("data:"):
                        if self._is_event_payload_complete(event_data_lines):
                            async for event in self._flush_event_data_lines(
                                event_data_lines
                            ):
                                yield event
                            if self._done_received:
                                break
                        data_str = line_text[5:]
                        if data_str.startswith(" "):
                            data_str = data_str[1:]
                        event_data_lines.append(data_str)

                if self._done_received:
                    break
        finally:
            # 清理：取消尚未完成的 chunk 读取 task，防止生成器提前关闭时 task 泄漏
            if pending_chunk_task is not None and not pending_chunk_task.done():
                pending_chunk_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await pending_chunk_task
            if unregister_cancellation_waiter is not None:
                unregister_cancellation_waiter()
            if cancellation_waiter is not None and not cancellation_waiter.done():
                cancellation_waiter.cancel()

        try:
            line_buffer += decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            self._record_protocol_error(
                "response_error",
                "Invalid UTF-8 in SSE stream",
                body=str(exc),
            )

        if self._protocol_errors:
            return

        # 处理末尾残留行（无换行结尾）
        trailing_line = line_buffer
        trailing_normalized = trailing_line.lstrip()
        if trailing_normalized.startswith("data:"):
            data_str = trailing_normalized[5:]
            if data_str.startswith(" "):
                data_str = data_str[1:]
            event_data_lines.append(data_str)

        async for event in self._flush_event_data_lines(event_data_lines):
            yield event

    def _get_stream_idle_heartbeat_sec(self) -> float | None:
        """返回流式空闲心跳日志间隔。

        Args:
            无。

        Returns:
            float | None: 心跳间隔秒数；`None` 表示禁用。

        Raises:
            无。
        """
        raw_value = getattr(
            self._running_config,
            "stream_idle_heartbeat_sec",
            _DEFAULT_STREAM_IDLE_HEARTBEAT_SEC,
        )
        if raw_value is None:
            return None
        try:
            normalized = float(raw_value)
        except (TypeError, ValueError):
            return _DEFAULT_STREAM_IDLE_HEARTBEAT_SEC
        if normalized <= 0:
            return None
        return normalized

    async def _flush_event_data_lines(
        self,
        event_data_lines: list[str],
    ) -> AsyncIterator[StreamEvent]:
        """
        将聚合后的 SSE data 行拼接成一个事件并解析。

        Args:
            event_data_lines: 当前事件聚合的 data 行列表。

        Yields:
            StreamEvent — 由 payload 解析得到的事件。
        """
        if not event_data_lines:
            return

        payload = "\n".join(event_data_lines)
        event_data_lines.clear()
        if not payload.strip():
            return

        if payload.strip() == "[DONE]":
            self._done_received = True
            return

        async for event in self._handle_payload(payload):
            yield event

    def _record_protocol_error(
        self,
        error_type: str,
        message: str,
        *,
        body: str = "",
    ) -> None:
        """记录 SSE 协议层错误，供 Runner 统一映射为 error_event。

        Args:
            error_type: 错误分类。
            message: 错误消息。
            body: 可选错误上下文。

        Returns:
            无。
        """
        error_record = {
            "error_type": error_type,
            "message": message,
        }
        if body:
            error_record["body"] = body
        self._protocol_errors.append(error_record)

    def _is_event_payload_complete(self, event_data_lines: list[str]) -> bool:
        """
        判断当前聚合的 data 行是否已构成一个完整事件。

        兼容两种上游行为：
        1. 标准 SSE：事件间有空行分隔（此方法多为 False）。
        2. 行分隔 JSON：每个 `data:` 行即一个完整 JSON 事件（无空行）。

        Args:
            event_data_lines: 当前事件聚合的 data 行列表。

        Returns:
            `True` 表示应先 flush，再接收下一条 data 行。
        """
        if not event_data_lines:
            return False

        payload = "\n".join(event_data_lines).strip()
        if not payload:
            return False
        if payload == "[DONE]":
            return True
        try:
            json.loads(payload)
        except json.JSONDecodeError:
            return False
        return True

    async def _handle_payload(self, payload: str) -> AsyncIterator[StreamEvent]:
        """
        解析单个 SSE payload（data 行内容），产出内容与工具调用增量事件。

        Args:
            payload: SSE data 行载荷（JSON 字符串）。

        Yields:
            StreamEvent — content_delta、tool_call_start、tool_call_delta。
        """
        # 1) 解析 JSON payload
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            self._record_protocol_error(
                "response_error",
                "Invalid JSON SSE payload",
                body=payload[:500],
            )
            return

        # 2) 捕获 token 用量（流式 include_usage 最后一个 chunk 携带，choices 可能为空）
        usage = data.get("usage")
        if usage and isinstance(usage, dict):
            self._usage = usage

        # 3) 选择首个包含 delta 的 choice
        choices = data.get("choices", [])
        if not choices:
            return
        self._stream_state["saw_choice"] = True
        if len(choices) > 1:
            Log.warn(
                f"{self._log_prefix} SSE 分片包含多个 choices 字段，仅使用第一个 delta",
                module=MODULE,
            )

        for choice in choices:
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                if self._stream_state.get("finish_reason") != finish_reason:
                    if should_log_debug(self._running_config, self._stream_state):
                        Log.debug(
                            f"{self._log_prefix} 收到 finish_reason: {finish_reason}",
                            module=MODULE,
                        )
                self._stream_state["finish_reason"] = finish_reason
                if finish_reason == "tool_calls":
                    # 标记看到了工具调用完成的 finish_reason
                    self._stream_state["tool_calls_finished"] = True
                elif finish_reason == "length":
                    # 模型输出被截断（上下文窗口耗尽或 max_tokens 不足）
                    Log.warn(
                        f"{self._log_prefix} 输出被截断 (finish_reason=length)，"
                        f"模型可能因上下文窗口不足而无法生成完整内容",
                        module=MODULE,
                    )
                    self._stream_state["truncated"] = True
                elif finish_reason == "content_filter":
                    Log.warn(
                        f"{self._log_prefix} 输出命中内容过滤 (finish_reason=content_filter)，"
                        f"本轮结果将标记为 filtered",
                        module=MODULE,
                    )
                    self._stream_state["content_filtered"] = True
                break

        # 仅处理第一个包含 delta 的 choice
        delta = None
        for choice in choices:
            candidate = choice.get("delta")
            if candidate is not None:
                delta = candidate
                break
        if delta is None:
            return

        # 3) 累积内容增量
        if "content" in delta and delta["content"]:
            text = delta["content"]
            self._content_buffer.append(text)
            yield content_delta(text)

        # 3.5) 累积推理内容增量（thinking 模式思维链）
        if "reasoning_content" in delta and delta["reasoning_content"]:
            text = delta["reasoning_content"]
            self._reasoning_content_buffer.append(text)
            yield reasoning_delta(text)

        # 4) 处理工具调用增量（可能跨多条 delta）
        if "tool_calls" in delta:
            tool_calls = delta.get("tool_calls")
            if tool_calls is None:
                if should_log_debug(self._running_config, self._stream_state):
                    Log.debug(
                        f"{self._log_prefix} 收到 tool_calls=None 的增量，已忽略",
                        module=MODULE,
                    )
                return
            if not isinstance(tool_calls, list):
                Log.warn(
                    f"{self._log_prefix} tool_calls 字段不是列表，已忽略",
                    module=MODULE,
                )
                self._record_protocol_error(
                    "tool_call_invalid",
                    "tool_calls must be a list",
                    body=json.dumps({"tool_calls": tool_calls}, ensure_ascii=False, default=str),
                )
                return
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    self._record_protocol_error(
                        "tool_call_incomplete",
                        "Tool call arguments incomplete or invalid",
                        body=json.dumps(
                            ["tool_call entry is not object"],
                            ensure_ascii=False,
                        ),
                    )
                    return
                async for event in self._handle_tool_call_delta(tc):
                    yield event

    async def _handle_tool_call_delta(self, tc: Dict) -> AsyncIterator[StreamEvent]:
        """
        处理单个工具调用增量（从 tool_calls 数组中的一项）。

        Args:
            tc: 工具调用增量字典（含 index、id、function 等）。

        Yields:
            StreamEvent — tool_call_start、tool_call_delta。
        """
        # 获取工具调用索引
        tool_index = tc.get("index")
        if tool_index is None or not isinstance(tool_index, int):
            Log.warn(
                f"{self._log_prefix} 工具调用增量 index 缺失或类型异常: {tool_index!r}",
                module=MODULE,
            )
            self._record_protocol_error(
                "tool_call_incomplete",
                "Tool call arguments incomplete or invalid",
                body=json.dumps(
                    [f"tool_call entry invalid index: {tool_index!r}"],
                    ensure_ascii=False,
                ),
            )
            return

        entry = self._tool_calls_buffer.setdefault(
            tool_index,
            {
                "id": None,
                "name": "",
                "arguments_buf": "",
                "index_in_iteration": tool_index,
                "started": False,
            },
        )

        func = tc.get("function", {})
        if not isinstance(func, dict):
            self._record_protocol_error(
                "tool_call_incomplete",
                "Tool call arguments incomplete or invalid",
                body=json.dumps(
                    [f"tool_index {tool_index}: missing function object"],
                    ensure_ascii=False,
                ),
            )
            return

        buffered_args_before_current = entry["arguments_buf"]
        args_delta: str | None = None
        if "arguments" in func:
            raw_args_delta = func["arguments"]
            if not isinstance(raw_args_delta, str):
                self._record_protocol_error(
                    "tool_call_incomplete",
                    "Tool call arguments incomplete or invalid",
                    body=json.dumps(
                        [
                            (
                                f"tool_index {tool_index}: "
                                f"arguments type is {type(raw_args_delta).__name__}"
                            )
                        ],
                        ensure_ascii=False,
                    ),
                )
                return
            args_delta = raw_args_delta

        # 记录 id（首次出现即写入）
        tc_id = tc.get("id")
        if tc_id and not entry["id"]:
            entry["id"] = tc_id
            if self._running_config.debug_tool_delta:
                Log.debug(
                    f"{self._log_prefix} 记录 tool_call_id: {tc_id}（index={tool_index}）",
                    module=MODULE,
                )

        # 记录 name（首次出现即写入）
        func_name = func.get("name", "")
        if func_name and not entry["name"]:
            entry["name"] = func_name
            if self._running_config.debug_tool_delta:
                Log.debug(
                    f"{self._log_prefix} 记录工具名: {func_name}（index={tool_index}）",
                    module=MODULE,
                )

        # id + name 就绪后触发 tool_call_start
        if entry["id"] and entry["name"] and not entry["started"]:
            entry["started"] = True
            yield tool_call_start(
                tool_name=entry["name"],
                tool_call_id=entry["id"],
            )
            if buffered_args_before_current:
                # id/name 晚于 arguments 到达时，补发已缓存前缀，避免事件流丢前缀。
                yield tool_call_delta(
                    tool_call_id=entry["id"],
                    name=entry["name"],
                    arguments_delta=buffered_args_before_current,
                )

        # 累积参数增量
        if args_delta:
            entry["arguments_buf"] += args_delta
            if entry["id"] and entry["name"]:
                yield tool_call_delta(
                    tool_call_id=entry["id"],
                    name=entry["name"],
                    arguments_delta=args_delta,
                )

    def _assemble_tool_calls(self) -> tuple[list[dict[str, Any]], list[str]]:
        """
        组装并验证缓冲区中的工具调用。

        Returns:
            (tool_calls, validation_errors) 元组：
            - tool_calls: 验证通过的工具调用列表
            - validation_errors: 验证错误描述列表
        """
        if not self._tool_calls_buffer:
            return [], []

        validation_errors: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        sorted_indices = sorted(self._tool_calls_buffer.keys())
        # 校验 index 规范性：非 int / 非 0 起始 / 不连续 均视为协议异常
        if sorted_indices and isinstance(sorted_indices[0], int) and sorted_indices[0] != 0:
            validation_errors.append(
                f"tool_call index 未从 0 开始: 首个 index={sorted_indices[0]}"
            )
        for i in range(1, len(sorted_indices)):
            prev, curr = sorted_indices[i - 1], sorted_indices[i]
            if not isinstance(prev, int) or not isinstance(curr, int):
                validation_errors.append(
                    f"tool_call index 类型异常: {prev!r}, {curr!r}"
                )
                break
            if curr != prev + 1:
                validation_errors.append(
                    f"tool_call index 不连续: {prev} -> {curr}"
                )

        for tool_index in sorted_indices:
            tc_data = self._tool_calls_buffer[tool_index]
            tc_id = tc_data.get("id")
            name = tc_data.get("name")
            args_buf = tc_data.get("arguments_buf", "")

            if not tc_id:
                validation_errors.append(f"tool_index {tool_index}: missing id")
                continue
            if not name:
                validation_errors.append(f"tool_index {tool_index}: missing name")
                continue
            try:
                args_obj = json.loads(args_buf)
            except json.JSONDecodeError as exc:
                validation_errors.append(
                    f"tool_index {tool_index}: invalid arguments JSON ({exc})"
                )
                continue
            if not isinstance(args_obj, dict):
                validation_errors.append(
                    f"tool_index {tool_index}: arguments is not object"
                )
                continue

            tool_calls.append({
                "id": tc_id,
                "name": name,
                "arguments": args_obj,
                "index_in_iteration": tool_index,
            })

        return tool_calls, validation_errors
