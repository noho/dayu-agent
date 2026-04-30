"""聊天流式运行时管理。

管理后台 worker 线程、事件队列、流式帧状态与轮询逻辑。
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Literal

import streamlit as st

from dayu.contracts.events import AppEvent, AppEventType
from dayu.log import Log
from dayu.services.contracts import ChatTurnRequest, SessionResolutionPolicy
from dayu.services.protocols import ChatServiceProtocol

from dayu.web.streamlit.pages.chat.utils import (
    extract_stream_text,
    fold_app_events_to_assistant_text,
    summarize_user_text,
)

MODULE = "dayu.web.streamlit.pages.chat.stream_runtime"

_SCENE_NAME_INTERACTIVE = "interactive"
_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS = 90.0
_STREAM_CHUNK_TIMEOUT_SECONDS = 45.0
_STREAM_TIMEOUT_MESSAGE = "交互式分析等待模型输出超时，请检查模型 API Key、网络连接或稍后重试。"
_STREAM_EVENT_BATCH_LIMIT = 256


@dataclass(frozen=True)
class StreamQueueItem:
    """线程桥接队列元素。"""

    done: bool
    kind: Literal["content", "reasoning"] = "content"
    chunk: str = ""
    event_type: Literal["chunk", "session_id", "side_message", "filtered", "error", "done"] = "chunk"
    flag: bool = False


@dataclass
class ChatStreamFrameState:
    """聊天流式渲染帧状态。"""

    trace_id: str
    session_id: str
    reasoning_text: str
    answer_text: str
    side_messages: list[str]
    filtered_flags: list[bool]
    done: bool
    error_message: str


@dataclass
class ChatStreamRuntimeHandle:
    """后台流式 worker 运行时句柄。"""

    worker: threading.Thread
    event_queue: Queue[StreamQueueItem]
    cancel_event: threading.Event
    started_at: float
    last_chunk_at: float
    has_received_chunk: bool


CHAT_STREAM_RUNTIME_HANDLES: dict[str, ChatStreamRuntimeHandle] = {}


async def consume_chat_event_stream(
    *,
    chat_service: ChatServiceProtocol,
    user_text: str,
    session_id: str | None,
    ticker: str,
    event_queue: Queue[StreamQueueItem],
    cancel_event: threading.Event,
    trace_id: str,
) -> None:
    """异步消费聊天事件流并写入线程队列。

    参数:
        chat_service: 聊天服务协议实例。
        user_text: 用户输入文本。
        session_id: 当前会话标识，首次可为 ``None``。
        ticker: 股票代码。
        event_queue: 线程安全队列，用于向主线程投递流式事件。
        cancel_event: 取消信号事件；主线程设置后协程应尽快退出。
        trace_id: 请求链路追踪标识。

    返回值:
        无。

    异常:
        无；异常由调用方通过 ``err_out`` 捕获。
    """

    request_started_at = time.perf_counter()
    Log.info(
        f"[{trace_id}] 提交聊天请求: ticker={ticker}, has_session={bool(session_id)}, "
        f"user_text_summary={summarize_user_text(user_text)}",
        module=MODULE,
    )
    request = ChatTurnRequest(
        user_text=user_text,
        session_id=session_id,
        ticker=ticker,
        scene_name=_SCENE_NAME_INTERACTIVE,
        session_resolution_policy=SessionResolutionPolicy.ENSURE_DETERMINISTIC,
    )
    submission = await chat_service.submit_turn(request)
    submit_elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
    Log.info(
        f"[{trace_id}] submit_turn 已返回: session_id={submission.session_id}, elapsed_ms={submit_elapsed_ms}",
        module=MODULE,
    )
    event_queue.put(
        StreamQueueItem(
            done=False,
            event_type="session_id",
            chunk=submission.session_id,
        )
    )

    buffered_events: list[AppEvent] = []
    has_streamed_chunks = False
    content_delta_count = 0
    reasoning_delta_count = 0
    warning_count = 0
    error_count = 0
    cancelled_count = 0
    first_chunk_latency_ms: int | None = None
    async for event in submission.event_stream:
        if cancel_event.is_set():
            Log.info(f"[{trace_id}] 收到取消信号，终止事件流消费", module=MODULE)
            break
        buffered_events.append(event)
        if event.type == AppEventType.CONTENT_DELTA:
            content_delta_count += 1
        elif event.type == AppEventType.REASONING_DELTA:
            reasoning_delta_count += 1
        elif event.type == AppEventType.WARNING:
            warning_count += 1
        elif event.type == AppEventType.ERROR:
            error_count += 1
        elif event.type == AppEventType.CANCELLED:
            cancelled_count += 1

        if event.type in (AppEventType.CONTENT_DELTA, AppEventType.REASONING_DELTA):
            payload = event.payload
            chunk_text = extract_stream_text(payload) if isinstance(payload, (dict, str)) else ""
            if chunk_text:
                has_streamed_chunks = True
                if first_chunk_latency_ms is None:
                    first_chunk_latency_ms = int((time.perf_counter() - request_started_at) * 1000)
                    Log.info(
                        f"[{trace_id}] 收到首个可展示增量: latency_ms={first_chunk_latency_ms}, "
                        f"event_type={event.type.value}",
                        module=MODULE,
                    )
                chunk_kind: Literal["content", "reasoning"] = (
                    "reasoning" if event.type == AppEventType.REASONING_DELTA else "content"
                )
                event_queue.put(
                    StreamQueueItem(
                        done=False,
                        kind=chunk_kind,
                        chunk=chunk_text,
                        event_type="chunk",
                    )
                )

    folded_text, side_messages, filtered = fold_app_events_to_assistant_text(buffered_events)
    if (not has_streamed_chunks) and folded_text:
        event_queue.put(
            StreamQueueItem(
                done=False,
                kind="content",
                chunk=folded_text,
                event_type="chunk",
            )
        )
    for side_message in side_messages:
        event_queue.put(
            StreamQueueItem(
                done=False,
                event_type="side_message",
                chunk=side_message,
            )
        )
    event_queue.put(
        StreamQueueItem(
            done=False,
            event_type="filtered",
            flag=filtered,
        )
    )
    total_elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
    Log.info(
        f"[{trace_id}] 事件流消费完成: total_events={len(buffered_events)}, "
        f"content_delta={content_delta_count}, reasoning_delta={reasoning_delta_count}, "
        f"warning={warning_count}, error={error_count}, cancelled={cancelled_count}, "
        f"side_messages={len(side_messages)}, filtered={filtered}, "
        f"has_streamed_chunks={has_streamed_chunks}, folded_text_len={len(folded_text)}, "
        f"elapsed_ms={total_elapsed_ms}",
        module=MODULE,
    )


def run_stream_worker(
    *,
    chat_service: ChatServiceProtocol,
    user_text: str,
    session_id: str | None,
    ticker: str,
    event_queue: Queue[StreamQueueItem],
    cancel_event: threading.Event,
    trace_id: str,
) -> None:
    """在线程中运行异步事件消费协程。

    参数:
        chat_service: 聊天服务协议实例。
        user_text: 用户输入文本。
        session_id: 当前会话标识，首次可为 ``None``。
        ticker: 股票代码。
        event_queue: 线程安全队列，用于向主线程投递流式事件。
        cancel_event: 取消信号事件；接收后应尽快停止消费。
        trace_id: 请求链路追踪标识。

    返回值:
        无。

    异常:
        无；异常被捕获并通过 ``error`` 事件上报。
    """

    try:
        asyncio.run(
            consume_chat_event_stream(
                chat_service=chat_service,
                user_text=user_text,
                session_id=session_id,
                ticker=ticker,
                event_queue=event_queue,
                cancel_event=cancel_event,
                trace_id=trace_id,
            )
        )
    except BaseException as exception:  # noqa: BLE001
        Log.error(f"[{trace_id}] 后台事件消费失败: {exception}", exc_info=True, module=MODULE)
        event_queue.put(
            StreamQueueItem(
                done=False,
                event_type="error",
                chunk=str(exception),
            )
        )
    finally:
        event_queue.put(
            StreamQueueItem(
                done=True,
                event_type="done",
            )
        )


def new_stream_frame_state(*, trace_id: str, session_id: str) -> ChatStreamFrameState:
    """创建新的流式渲染帧状态。

    参数:
        trace_id: 请求追踪标识。
        session_id: 当前会话标识。

    返回值:
        初始的流式渲染帧状态对象。

    异常:
        无。
    """

    return ChatStreamFrameState(
        trace_id=trace_id,
        session_id=session_id,
        reasoning_text="",
        answer_text="",
        side_messages=[],
        filtered_flags=[],
        done=False,
        error_message="",
    )


def read_stream_frame_state(state_key: str) -> ChatStreamFrameState | None:
    """读取流式渲染帧状态。

    参数:
        state_key: 会话状态键。

    返回值:
        当前流式渲染帧状态；不存在或类型不合法时返回 ``None``。

    异常:
        无。
    """

    raw_state = st.session_state.get(state_key)
    if isinstance(raw_state, ChatStreamFrameState):
        return raw_state
    return None


def clear_chat_stream_runtime(*, ticker: str, stream_state_key: str) -> None:
    """清理聊天流式运行时句柄与会话状态。

    参数:
        ticker: 股票代码。
        stream_state_key: 会话状态键。

    返回值:
        无。

    异常:
        无。
    """

    runtime = CHAT_STREAM_RUNTIME_HANDLES.pop(ticker, None)
    if runtime is not None:
        if runtime.worker.is_alive():
            runtime.cancel_event.set()
            runtime.worker.join(timeout=0.1)
        else:
            runtime.worker.join()
    if stream_state_key in st.session_state:
        del st.session_state[stream_state_key]


def start_chat_stream_runtime(
    *,
    chat_service: ChatServiceProtocol,
    ticker: str,
    user_text: str,
    session_id: str,
    trace_id: str,
    stream_state_key: str,
) -> None:
    """启动聊天流式 worker 并初始化渲染帧状态。

    参数:
        chat_service: 聊天服务协议实例。
        ticker: 股票代码。
        user_text: 用户输入文本。
        session_id: 会话标识。
        trace_id: 请求追踪标识。
        stream_state_key: 会话状态键。

    返回值:
        无。

    异常:
        无。
    """

    clear_chat_stream_runtime(ticker=ticker, stream_state_key=stream_state_key)
    event_queue: Queue[StreamQueueItem] = Queue()
    cancel_event = threading.Event()
    worker = threading.Thread(
        target=run_stream_worker,
        kwargs={
            "chat_service": chat_service,
            "user_text": user_text,
            "session_id": session_id,
            "ticker": ticker,
            "event_queue": event_queue,
            "cancel_event": cancel_event,
            "trace_id": trace_id,
        },
        daemon=True,
    )
    started_at = time.perf_counter()
    CHAT_STREAM_RUNTIME_HANDLES[ticker] = ChatStreamRuntimeHandle(
        worker=worker,
        event_queue=event_queue,
        cancel_event=cancel_event,
        started_at=started_at,
        last_chunk_at=started_at,
        has_received_chunk=False,
    )
    st.session_state[stream_state_key] = new_stream_frame_state(trace_id=trace_id, session_id=session_id)
    worker.start()
    Log.info(
        f"[{trace_id}] 启动流式 worker: ticker={ticker}, session_id={session_id}",
        module=MODULE,
    )


def poll_chat_stream_events(*, ticker: str, stream_state_key: str) -> ChatStreamFrameState | None:
    """轮询并消费聊天流式事件。

    参数:
        ticker: 股票代码。
        stream_state_key: 会话状态键。

    返回值:
        更新后的流式渲染帧状态；若无活动流则返回 ``None``。

    异常:
        无。
    """

    state = read_stream_frame_state(stream_state_key)
    if state is None:
        return None

    runtime = CHAT_STREAM_RUNTIME_HANDLES.get(ticker)
    if runtime is None:
        state.done = True
        if not state.error_message:
            state.error_message = "流式任务运行句柄丢失，请重试。"
        st.session_state[stream_state_key] = state
        return state

    processed_events = 0
    while processed_events < _STREAM_EVENT_BATCH_LIMIT:
        try:
            event = runtime.event_queue.get_nowait()
        except Empty:
            break
        processed_events += 1
        if event.event_type == "chunk":
            if event.kind == "reasoning":
                state.reasoning_text = f"{state.reasoning_text}{event.chunk}"
            else:
                state.answer_text = f"{state.answer_text}{event.chunk}"
            runtime.has_received_chunk = True
            runtime.last_chunk_at = time.perf_counter()
            continue
        if event.event_type == "session_id":
            state.session_id = event.chunk
            continue
        if event.event_type == "side_message":
            state.side_messages.append(event.chunk)
            continue
        if event.event_type == "filtered":
            state.filtered_flags.append(event.flag)
            continue
        if event.event_type == "error":
            state.error_message = event.chunk or "交互式分析执行失败"
            state.done = True
            continue
        if event.event_type == "done" or event.done:
            state.done = True

    now = time.perf_counter()
    if runtime.worker.is_alive() and (not state.done):
        elapsed_since_start = now - runtime.started_at
        elapsed_since_chunk = now - runtime.last_chunk_at
        if (not runtime.has_received_chunk) and (elapsed_since_start >= _STREAM_FIRST_CHUNK_TIMEOUT_SECONDS):
            runtime.cancel_event.set()
            state.done = True
            state.error_message = _STREAM_TIMEOUT_MESSAGE
        elif runtime.has_received_chunk and (elapsed_since_chunk >= _STREAM_CHUNK_TIMEOUT_SECONDS):
            runtime.cancel_event.set()
            state.done = True
            state.error_message = _STREAM_TIMEOUT_MESSAGE

    if state.done and (not runtime.worker.is_alive()):
        while True:
            try:
                trailing_event = runtime.event_queue.get_nowait()
            except Empty:
                break
            if trailing_event.event_type == "side_message":
                state.side_messages.append(trailing_event.chunk)
            elif trailing_event.event_type == "filtered":
                state.filtered_flags.append(trailing_event.flag)
            elif trailing_event.event_type == "error":
                state.error_message = trailing_event.chunk or "交互式分析执行失败"
    st.session_state[stream_state_key] = state
    return state
