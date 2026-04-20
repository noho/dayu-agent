"""AsyncOpenAIRunner 调用路径分支测试。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from dayu.engine import async_openai_runner as aor
from dayu.contracts.protocols import ToolExecutionContext
from dayu.contracts.cancellation import CancelledError as EngineCancelledError, CancellationToken
from dayu.engine.events import EventType, StreamEvent


class _DummyExecutor:
    """工具执行器桩。"""

    def __init__(self, schemas: Optional[list[dict[str, Any]]] = None) -> None:
        """初始化执行器。

        Args:
            schemas: 工具 schema 列表。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._schemas = schemas or []

    def get_schemas(self) -> list[dict[str, Any]]:
        """返回工具 schema。

        Args:
            无。

        Returns:
            schema 列表。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        return self._schemas

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """执行工具。

        Args:
            name: 工具名。
            arguments: 参数。
            context: 上下文。

        Returns:
            执行结果。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        del context
        return {"ok": True, "value": {"tool": name, "arguments": arguments}}

    def clear_cursors(self) -> None:
        """清理游标（测试桩无操作）。"""

    def get_dup_call_spec(self, name: str) -> None:
        """返回空重复调用策略。"""

        _ = name
        return None

    def get_execution_context_param_name(self, name: str) -> None:
        """返回空 execution context 参数名。"""

        _ = name
        return None

    def register_response_middleware(self, callback: Any) -> None:
        """注册 response middleware（测试桩无操作）。"""

        _ = callback


class _FakeContent:
    """SSE 内容桩。"""

    def __init__(self, chunks: Optional[list[bytes]] = None) -> None:
        """初始化内容桩。

        Args:
            chunks: 分片字节列表。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._chunks = chunks or []

    async def iter_chunked(self, size: int) -> Any:
        """按块迭代。

        Args:
            size: 块大小。

        Returns:
            异步迭代器。

        Raises:
            RuntimeError: 迭代失败时抛出。
        """

        del size
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    """HTTP 响应桩。"""

    def __init__(
        self,
        *,
        status: int,
        headers: Optional[dict[str, str]] = None,
        json_data: Optional[dict[str, Any]] = None,
        text_data: str = "",
        chunks: Optional[list[bytes]] = None,
    ) -> None:
        """初始化响应。

        Args:
            status: HTTP 状态码。
            headers: 响应头。
            json_data: JSON 内容。
            text_data: 文本内容。
            chunks: SSE 分片。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self.status = status
        self.headers = headers or {}
        self._json_data = json_data or {}
        self._text_data = text_data
        self.content = _FakeContent(chunks=chunks)

    async def json(self) -> dict[str, Any]:
        """返回 JSON 内容。

        Args:
            无。

        Returns:
            JSON 字典。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        return self._json_data

    async def text(self) -> str:
        """返回文本内容。

        Args:
            无。

        Returns:
            响应文本。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        return self._text_data


@dataclass
class _PostStep:
    """一次 post 行为配置。"""

    response: Optional[_FakeResponse] = None
    error: Optional[Exception] = None


class _FakePostContext:
    """post 请求上下文管理器桩。"""

    def __init__(self, step: _PostStep) -> None:
        """初始化上下文。

        Args:
            step: 行为配置。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._step = step

    async def __aenter__(self) -> _FakeResponse:
        """进入上下文。

        Args:
            无。

        Returns:
            响应对象。

        Raises:
            Exception: 按配置抛出。
        """

        if self._step.error is not None:
            raise self._step.error
        if self._step.response is None:
            raise RuntimeError("response 不能为空")
        return self._step.response

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """退出上下文。

        Args:
            exc_type: 异常类型。
            exc: 异常对象。
            tb: traceback。

        Returns:
            False（不吞异常）。

        Raises:
            RuntimeError: 退出失败时抛出。
        """

        del exc_type, exc, tb
        return False


class _FakeClientSession:
    """aiohttp.ClientSession 桩。"""

    def __init__(self, steps: list[_PostStep], post_calls: list[dict[str, Any]]) -> None:
        """初始化会话。

        Args:
            steps: 顺序响应配置。
            post_calls: post 调用记录。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._steps = steps
        self._post_calls = post_calls
        self.closed = False

    async def __aenter__(self) -> "_FakeClientSession":
        """进入会话上下文。

        Args:
            无。

        Returns:
            自身。

        Raises:
            RuntimeError: 进入失败时抛出。
        """

        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        """退出会话上下文。

        Args:
            exc_type: 异常类型。
            exc: 异常对象。
            tb: traceback。

        Returns:
            False（不吞异常）。

        Raises:
            RuntimeError: 退出失败时抛出。
        """

        del exc_type, exc, tb
        return False

    async def close(self) -> None:
        """关闭会话并记录关闭状态。"""

        self.closed = True

    def post(self, endpoint_url: str, **kwargs: Any) -> _FakePostContext:
        """发起 post。

        Args:
            endpoint_url: URL。
            **kwargs: 请求参数。

        Returns:
            post 上下文管理器。

        Raises:
            RuntimeError: 无剩余步骤时抛出。
        """

        self._post_calls.append({"endpoint_url": endpoint_url, **kwargs})
        if not self._steps:
            raise RuntimeError("没有可用 post step")
        return _FakePostContext(self._steps.pop(0))


class _FakeClientError(Exception):
    """aiohttp.ClientError 桩。"""


def _build_runner_with_steps(
    monkeypatch: pytest.MonkeyPatch,
    steps: list[_PostStep],
    *,
    supports_stream: bool = True,
    supports_tool_calling: bool = True,
    cancellation_token: CancellationToken | None = None,
    created_sessions: list[_FakeClientSession] | None = None,
) -> tuple[aor.AsyncOpenAIRunner, list[dict[str, Any]]]:
    """构建带可控网络步骤的 Runner。

    Args:
        monkeypatch: monkeypatch fixture。
        steps: post 行为步骤。
        supports_stream: 是否支持流式。
        supports_tool_calling: 是否支持工具调用。

    Returns:
        (runner, post_calls)。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    post_calls: list[dict[str, Any]] = []

    def _session_factory() -> _FakeClientSession:
        """创建会话桩。

        Args:
            无。

        Returns:
            会话对象。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        session = _FakeClientSession(steps=steps, post_calls=post_calls)
        if created_sessions is not None:
            created_sessions.append(session)
        return session

    fake_aiohttp = SimpleNamespace(
        ClientSession=_session_factory,
        ClientTimeout=lambda **kwargs: kwargs,
        ClientError=_FakeClientError,
    )
    monkeypatch.setattr(aor, "aiohttp", fake_aiohttp)
    runner = aor.AsyncOpenAIRunner(
        endpoint_url="https://example.com/v1/chat/completions",
        model="test-model",
        headers={"Authorization": "Bearer x"},
        max_retries=1,
        supports_stream=supports_stream,
        supports_tool_calling=supports_tool_calling,
        cancellation_token=cancellation_token,
    )
    return runner, post_calls


async def _collect_events(stream: Any) -> list[StreamEvent]:
    """收集异步事件流。

    Args:
        stream: 异步事件流。

    Returns:
        事件列表。

    Raises:
        RuntimeError: 迭代失败时抛出。
    """

    events: list[StreamEvent] = []
    async for event in stream:
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_call_handles_non_retriable_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 400 非重试错误路径。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    runner, _ = _build_runner_with_steps(
        monkeypatch,
        steps=[_PostStep(response=_FakeResponse(status=400, text_data="bad request"))],
    )
    events = await _collect_events(runner.call([{"role": "user", "content": "hi"}], stream=False))
    assert events
    assert events[0].type == EventType.ERROR
    assert events[0].metadata.get("error_type") == "invalid_request"


@pytest.mark.asyncio
async def test_call_retriable_then_success_and_tools_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 429 重试后成功，并携带工具定义。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    async def _fake_sleep(delay: float) -> None:
        """替代 sleep。

        Args:
            delay: 延迟秒数。

        Returns:
            无。

        Raises:
            RuntimeError: 无。
        """

        del delay
        return None

    monkeypatch.setattr(aor.asyncio, "sleep", _fake_sleep)

    success_json = {
        "choices": [{"message": {"content": "ok"}}]
    }
    runner, calls = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(response=_FakeResponse(status=429, headers={"Retry-After": "1"}, text_data="rate limit")),
            _PostStep(
                response=_FakeResponse(
                    status=200,
                    headers={"Content-Type": "application/json"},
                    json_data=success_json,
                )
            ),
        ],
    )
    runner.set_tools(
        _DummyExecutor(
            schemas=[
                {
                    "name": "echo",
                    "description": "echo tool",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        )
    )
    events = await _collect_events(runner.call([{"role": "user", "content": "hi"}], stream=False))
    event_types = [event.type for event in events]
    assert EventType.WARNING in event_types
    assert EventType.CONTENT_COMPLETE in event_types
    assert EventType.DONE in event_types
    assert len(calls) == 2
    assert "tools" in calls[-1]["json"]


@pytest.mark.asyncio
async def test_call_cancelled_during_retry_sleep_raises_cancelled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 HTTP 重试等待期间会优先响应协作式取消。"""

    token = CancellationToken()
    runner, calls = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(response=_FakeResponse(status=429, headers={"Retry-After": "30"}, text_data="rate limit")),
            _PostStep(
                response=_FakeResponse(
                    status=200,
                    headers={"Content-Type": "application/json"},
                    json_data={"choices": [{"message": {"content": "ok"}}]},
                )
            ),
        ],
        cancellation_token=token,
    )

    stream = runner.call([{"role": "user", "content": "hi"}], stream=False)
    first_event = await stream.__anext__()

    assert first_event.type == EventType.WARNING
    token.cancel()
    with pytest.raises(EngineCancelledError):
        await stream.__anext__()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_call_cancelled_during_post_enter_raises_cancelled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 request enter 挂起期间收到取消会直接中止。"""

    token = CancellationToken()
    started = asyncio.Event()
    post_calls: list[dict[str, Any]] = []

    class _BlockingPostContext:
        async def __aenter__(self) -> _FakeResponse:
            started.set()
            await asyncio.sleep(100)
            return _FakeResponse(status=200, headers={"Content-Type": "application/json"})

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            del exc_type, exc, tb
            return False

    class _BlockingClientSession:
        async def close(self) -> None:
            return None

        def post(self, endpoint_url: str, **kwargs: Any) -> _BlockingPostContext:
            post_calls.append({"endpoint_url": endpoint_url, **kwargs})
            return _BlockingPostContext()

    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda: _BlockingClientSession(),
        ClientTimeout=lambda **kwargs: kwargs,
        ClientError=_FakeClientError,
    )
    monkeypatch.setattr(aor, "aiohttp", fake_aiohttp)

    runner = aor.AsyncOpenAIRunner(
        endpoint_url="https://example.com/v1/chat/completions",
        model="test-model",
        headers={"Authorization": "Bearer x"},
        max_retries=1,
        cancellation_token=token,
    )
    consume_task = asyncio.create_task(
        _collect_events(runner.call([{"role": "user", "content": "hi"}], stream=False))
    )

    await started.wait()
    token.cancel()
    with pytest.raises(EngineCancelledError):
        await consume_task
    assert len(post_calls) == 1


@pytest.mark.asyncio
async def test_call_outer_task_cancel_cleans_inflight_post_enter_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证外层任务取消时，建连等待对应的子任务也会被取消收口。"""

    token = CancellationToken()
    started = asyncio.Event()
    inner_cancelled = asyncio.Event()

    class _BlockingPostContext:
        async def __aenter__(self) -> _FakeResponse:
            started.set()
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                inner_cancelled.set()
                raise
            return _FakeResponse(status=200, headers={"Content-Type": "application/json"})

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            del exc_type, exc, tb
            return False

    class _BlockingClientSession:
        async def close(self) -> None:
            return None

        def post(self, endpoint_url: str, **kwargs: Any) -> _BlockingPostContext:
            del endpoint_url, kwargs
            return _BlockingPostContext()

    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda: _BlockingClientSession(),
        ClientTimeout=lambda **kwargs: kwargs,
        ClientError=_FakeClientError,
    )
    monkeypatch.setattr(aor, "aiohttp", fake_aiohttp)

    runner = aor.AsyncOpenAIRunner(
        endpoint_url="https://example.com/v1/chat/completions",
        model="test-model",
        headers={"Authorization": "Bearer x"},
        max_retries=1,
        cancellation_token=token,
    )
    consume_task = asyncio.create_task(
        _collect_events(runner.call([{"role": "user", "content": "hi"}], stream=False))
    )

    await started.wait()
    consume_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consume_task
    assert inner_cancelled.is_set()


@pytest.mark.asyncio
async def test_call_success_unregisters_cancellation_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证成功调用后不会把取消回调残留在复用 token 上。"""

    token = CancellationToken()
    runner, _ = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(
                response=_FakeResponse(
                    status=200,
                    headers={"Content-Type": "application/json"},
                    json_data={"choices": [{"message": {"content": "ok"}}]},
                )
            ),
            _PostStep(
                response=_FakeResponse(
                    status=200,
                    headers={"Content-Type": "application/json"},
                    json_data={"choices": [{"message": {"content": "ok2"}}]},
                )
            ),
        ],
        cancellation_token=token,
    )

    await _collect_events(runner.call([{"role": "user", "content": "hi"}], stream=False))

    assert token._callbacks == []

    await _collect_events(runner.call([{"role": "user", "content": "again"}], stream=False))

    assert token._callbacks == []


@pytest.mark.asyncio
async def test_call_reuses_session_until_runner_close(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 Runner 会复用实例级 session，并在显式 close 后重建。"""

    created_sessions: list[_FakeClientSession] = []
    runner, post_calls = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(
                response=_FakeResponse(
                    status=200,
                    headers={"Content-Type": "application/json"},
                    json_data={"choices": [{"message": {"content": "first"}}]},
                )
            ),
            _PostStep(
                response=_FakeResponse(
                    status=200,
                    headers={"Content-Type": "application/json"},
                    json_data={"choices": [{"message": {"content": "second"}}]},
                )
            ),
            _PostStep(
                response=_FakeResponse(
                    status=200,
                    headers={"Content-Type": "application/json"},
                    json_data={"choices": [{"message": {"content": "third"}}]},
                )
            ),
        ],
        created_sessions=created_sessions,
    )

    await _collect_events(runner.call([{"role": "user", "content": "first"}], stream=False))
    await _collect_events(runner.call([{"role": "user", "content": "second"}], stream=False))

    assert len(created_sessions) == 1
    assert len(post_calls) == 2
    assert created_sessions[0].closed is False

    await runner.close()

    assert created_sessions[0].closed is True

    await _collect_events(runner.call([{"role": "user", "content": "third"}], stream=False))

    assert len(created_sessions) == 2
    assert created_sessions[1].closed is False


@pytest.mark.asyncio
async def test_call_cancelled_during_response_body_read_raises_cancelled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证响应体读取挂起期间收到取消会直接中止。"""

    token = CancellationToken()
    started = asyncio.Event()

    class _BlockingErrorResponse(_FakeResponse):
        async def text(self) -> str:
            started.set()
            await asyncio.sleep(100)
            return await super().text()

    runner, _ = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(
                response=_BlockingErrorResponse(status=400, text_data="bad request"),
            )
        ],
        cancellation_token=token,
    )
    consume_task = asyncio.create_task(
        _collect_events(runner.call([{"role": "user", "content": "hi"}], stream=False))
    )

    await started.wait()
    token.cancel()
    with pytest.raises(EngineCancelledError):
        await consume_task


@pytest.mark.asyncio
async def test_call_prefers_cancelled_over_timeout_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 timeout 重试路径与取消竞争时，以取消为准。"""

    token = CancellationToken()
    runner, calls = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(error=asyncio.TimeoutError()),
            _PostStep(error=asyncio.TimeoutError()),
        ],
        cancellation_token=token,
    )
    stream = runner.call([{"role": "user", "content": "hi"}], stream=False)
    first_event = await stream.__anext__()

    assert first_event.type == EventType.WARNING
    token.cancel()
    with pytest.raises(EngineCancelledError):
        await stream.__anext__()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_call_retriable_exhausted_and_unknown_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 5xx 重试耗尽与未知状态码分支。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    async def _fake_sleep(delay: float) -> None:
        """替代 sleep。

        Args:
            delay: 延迟秒数。

        Returns:
            无。

        Raises:
            RuntimeError: 无。
        """

        del delay
        return None

    monkeypatch.setattr(aor.asyncio, "sleep", _fake_sleep)

    runner_500, _ = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(response=_FakeResponse(status=500, text_data="err-1")),
            _PostStep(response=_FakeResponse(status=500, text_data="err-2")),
        ],
    )
    events_500 = await _collect_events(runner_500.call([{"role": "user", "content": "hi"}], stream=False))
    assert events_500[-1].type == EventType.ERROR
    assert events_500[-1].metadata.get("error_type") == "server_error"

    runner_418, _ = _build_runner_with_steps(
        monkeypatch,
        steps=[_PostStep(response=_FakeResponse(status=418, text_data="teapot"))],
    )
    events_418 = await _collect_events(runner_418.call([{"role": "user", "content": "hi"}], stream=False))
    assert events_418[-1].type == EventType.ERROR
    assert events_418[-1].metadata.get("error_type") == "unknown_http_status"


@pytest.mark.asyncio
async def test_call_timeout_client_error_and_unexpected_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证超时、网络错误和未知异常分支。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    async def _fake_sleep(delay: float) -> None:
        """替代 sleep。

        Args:
            delay: 延迟秒数。

        Returns:
            无。

        Raises:
            RuntimeError: 无。
        """

        del delay
        return None

    monkeypatch.setattr(aor.asyncio, "sleep", _fake_sleep)

    timeout_runner, _ = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(error=asyncio.TimeoutError()),
            _PostStep(error=asyncio.TimeoutError()),
        ],
    )
    timeout_events = await _collect_events(timeout_runner.call([{"role": "user", "content": "hi"}], stream=False))
    assert timeout_events[-1].metadata.get("error_type") == "timeout"

    network_runner, _ = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(error=_FakeClientError("net-1")),
            _PostStep(error=_FakeClientError("net-2")),
        ],
    )
    network_events = await _collect_events(network_runner.call([{"role": "user", "content": "hi"}], stream=False))
    assert network_events[-1].metadata.get("error_type") == "network_error"

    unknown_runner, _ = _build_runner_with_steps(
        monkeypatch,
        steps=[_PostStep(error=ValueError("bad"))],
    )
    unknown_events = await _collect_events(unknown_runner.call([{"role": "user", "content": "hi"}], stream=False))
    assert unknown_events[-1].metadata.get("error_type") == "unknown_error"


@pytest.mark.asyncio
async def test_call_stream_fallback_and_sse_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证流式降级与 SSE 解析路径。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n',
        b"data: [DONE]\n",
    ]
    runner_stream_false, _ = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(
                response=_FakeResponse(
                    status=200,
                    headers={"Content-Type": "text/event-stream"},
                    chunks=sse_chunks,
                )
            )
        ],
        supports_stream=False,
        supports_tool_calling=False,
    )
    runner_stream_false.set_tools(_DummyExecutor(schemas=[{"name": "t", "parameters": {"type": "object"}}]))
    events_false = await _collect_events(
        runner_stream_false.call([{"role": "user", "content": "hi"}], stream=True, n=2)
    )
    assert events_false[-1].type in {EventType.DONE, EventType.ERROR}

    runner_unknown_ct, _ = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(
                response=_FakeResponse(
                    status=200,
                    headers={"Content-Type": "text/plain"},
                    chunks=sse_chunks,
                )
            )
        ],
    )
    events_unknown_ct = await _collect_events(
        runner_unknown_ct.call([{"role": "user", "content": "hi"}], stream=True)
    )
    assert any(event.type == EventType.CONTENT_DELTA for event in events_unknown_ct)


@pytest.mark.asyncio
async def test_run_tool_call_cancelled_and_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证工具执行取消分支与显式构造。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake_aiohttp = SimpleNamespace(
        ClientSession=lambda: None,
        ClientTimeout=lambda **kwargs: kwargs,
        ClientError=_FakeClientError,
    )
    monkeypatch.setattr(aor, "aiohttp", fake_aiohttp)

    runner = aor.AsyncOpenAIRunner(
        endpoint_url="https://example.com/v1/chat/completions",
        model="x",
        headers={"Authorization": "Bearer x"},
        temperature=0.2,
        timeout=30,
        max_retries=2,
        default_extra_payloads={"max_tokens": 128},
        supports_stream=True,
        supports_tool_calling=True,
        running_config=aor.AsyncOpenAIRunnerRunningConfig(tool_timeout_seconds=5.0),
    )
    runner.set_tools(_DummyExecutor())

    async def _fake_to_thread(*args: Any, **kwargs: Any) -> Any:
        """替代 to_thread。

        Args:
            *args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            无。

        Raises:
            asyncio.CancelledError: 人工触发。
        """

        del args, kwargs
        raise asyncio.CancelledError()

    monkeypatch.setattr(aor.asyncio, "to_thread", _fake_to_thread)
    result = await runner._run_tool_call(
        {"id": "t1", "name": "tool", "arguments": {}, "index_in_iteration": 0},
        "req1",
        {"run_id": "r", "iteration_id": "t", "request_id": "q"},
    )
    assert result["result"]["ok"] is False
    assert result["result"]["error"] == "cancelled"


@pytest.mark.asyncio
async def test_call_sets_sock_read_timeout_for_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证流式请求会把 `stream_idle_timeout` 映射到 `sock_read`。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    runner, post_calls = _build_runner_with_steps(
        monkeypatch,
        steps=[
            _PostStep(
                response=_FakeResponse(
                    status=200,
                    headers={"Content-Type": "text/event-stream"},
                    chunks=sse_chunks,
                )
            )
        ],
    )
    runner.stream_idle_timeout = 17.0

    events = await _collect_events(
        runner.call([{"role": "user", "content": "hi"}], stream=True)
    )

    assert events[-1].type == EventType.DONE
    assert post_calls[0]["timeout"]["total"] == 3600
    assert post_calls[0]["timeout"]["sock_read"] == 17.0


@pytest.mark.asyncio
async def test_call_uses_model_total_timeout_and_stream_idle_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证请求层同时使用模型总超时与流式空闲超时。

    Args:
        monkeypatch: monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    sse_chunks = [
        b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    post_calls: list[dict[str, Any]] = []

    def _session_factory() -> _FakeClientSession:
        return _FakeClientSession(
            steps=[
                _PostStep(
                    response=_FakeResponse(
                        status=200,
                        headers={"Content-Type": "text/event-stream"},
                        chunks=sse_chunks,
                    )
                )
            ],
            post_calls=post_calls,
        )

    fake_aiohttp = SimpleNamespace(
        ClientSession=_session_factory,
        ClientTimeout=lambda **kwargs: kwargs,
        ClientError=_FakeClientError,
    )
    monkeypatch.setattr(aor, "aiohttp", fake_aiohttp)

    runner = aor.AsyncOpenAIRunner(
        endpoint_url="https://example.com/v1/chat/completions",
        model="test-model",
        headers={"Authorization": "Bearer x"},
        temperature=0.2,
        timeout=91,
        running_config=aor.AsyncOpenAIRunnerRunningConfig(stream_idle_timeout=19.0),
    )

    events = await _collect_events(
        runner.call([{"role": "user", "content": "hi"}], stream=True)
    )

    assert events[-1].type == EventType.DONE
    assert post_calls[0]["timeout"]["total"] == 91
    assert post_calls[0]["timeout"]["sock_read"] == 19.0
