"""交互式分析页的异步事件流桥接工具。"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from collections.abc import Iterator
from typing import Literal

import streamlit as st

from dayu.contracts.events import AppEvent, AppEventType
from dayu.log import Log
from dayu.services.contracts import ChatTurnRequest, SessionResolutionPolicy
from dayu.services.protocols import ChatServiceProtocol
from dayu.web.streamlit.stream_chat_events import (
    fold_app_events_to_assistant_text,
    normalize_stream_text_for_markdown,
)

MODULE = "dayu.web.streamlit.pages.chat_stream_bridge"
_SCENE_NAME_INTERACTIVE = "interactive"
_FILTERED_INFO_MESSAGE = "本轮输出触发内容过滤，结果可能不完整。"
_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS = 90.0
_STREAM_CHUNK_TIMEOUT_SECONDS = 45.0
_STREAM_TIMEOUT_MESSAGE = "交互式分析等待模型输出超时，请检查模型 API Key、网络连接或稍后重试。"
_STREAM_TEXT_KEYS: tuple[str, ...] = ("content", "text", "answer")


@dataclass(frozen=True)
class StreamQueueItem:
    """线程桥接队列元素。"""

    done: bool
    kind: Literal["content", "reasoning"] = "content"
    chunk: str = ""


def extract_stream_text(payload: str | dict[str, str | bool]) -> str:
    """提取流式事件可展示文本，并保留 Markdown 所需空白。

    参数:
        payload: 流式事件负载，支持字符串或字典。

    返回值:
        可展示文本；当负载仅包含空白时返回空字符串。

    异常:
        无。
    """

    if isinstance(payload, str):
        return normalize_stream_text_for_markdown(payload) if payload.strip() else ""
    for key in _STREAM_TEXT_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, str):
            if candidate.strip():
                return normalize_stream_text_for_markdown(candidate)
    return ""


def build_request_trace_id(*, ticker: str, user_text: str) -> str:
    """构建单次提交的日志追踪 ID。"""

    prefix = ticker.strip().upper()[:8] or "UNKNOWN"
    text_hash = abs(hash(user_text)) % 100000
    return f"{prefix}-{int(time.time() * 1000)}-{text_hash:05d}"


def summarize_user_text(user_text: str) -> str:
    """生成用户输入脱敏摘要，避免日志打印完整内容。"""

    normalized = " ".join(user_text.strip().split())
    preview = normalized[:48]
    suffix = "..." if len(normalized) > len(preview) else ""
    return f"len={len(user_text)}, preview={preview!r}{suffix}"


async def _consume_chat_event_stream(
    *,
    chat_service: ChatServiceProtocol,
    user_text: str,
    session_id: str | None,
    ticker: str,
    chunk_queue: Queue[StreamQueueItem],
    session_id_out: list[str],
    sides_out: list[str],
    filtered_holder: list[bool],
    trace_id: str,
) -> None:
    """异步消费聊天事件流并写入线程队列。"""

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
        session_resolution_policy=SessionResolutionPolicy.AUTO,
    )
    submission = await chat_service.submit_turn(request)
    submit_elapsed_ms = int((time.perf_counter() - request_started_at) * 1000)
    Log.info(
        f"[{trace_id}] submit_turn 已返回: session_id={submission.session_id}, elapsed_ms={submit_elapsed_ms}",
        module=MODULE,
    )
    session_id_out.append(submission.session_id)

    buffered_events: list[AppEvent] = []
    has_streamed_chunks = False
    content_delta_count = 0
    reasoning_delta_count = 0
    warning_count = 0
    error_count = 0
    cancelled_count = 0
    first_chunk_latency_ms: int | None = None
    async for event in submission.event_stream:
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
                chunk_queue.put(StreamQueueItem(done=False, kind=chunk_kind, chunk=chunk_text))

    folded_text, side_messages, filtered = fold_app_events_to_assistant_text(buffered_events)
    if (not has_streamed_chunks) and folded_text:
        chunk_queue.put(StreamQueueItem(done=False, kind="content", chunk=folded_text))
    sides_out.extend(side_messages)
    filtered_holder.append(filtered)
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


def _run_stream_worker(
    *,
    chat_service: ChatServiceProtocol,
    user_text: str,
    session_id: str | None,
    ticker: str,
    chunk_queue: Queue[StreamQueueItem],
    session_id_out: list[str],
    sides_out: list[str],
    filtered_holder: list[bool],
    err_out: list[BaseException],
    trace_id: str,
) -> None:
    """在线程中运行异步事件消费协程。"""

    try:
        asyncio.run(
            _consume_chat_event_stream(
                chat_service=chat_service,
                user_text=user_text,
                session_id=session_id,
                ticker=ticker,
                chunk_queue=chunk_queue,
                session_id_out=session_id_out,
                sides_out=sides_out,
                filtered_holder=filtered_holder,
                trace_id=trace_id,
            )
        )
    except BaseException as exception:  # noqa: BLE001
        Log.error(f"[{trace_id}] 后台事件消费失败: {exception}", exc_info=True, module=MODULE)
        err_out.append(exception)
    finally:
        chunk_queue.put(StreamQueueItem(done=True))


def sync_stream_via_asyncio(
    chat_service: ChatServiceProtocol,
    *,
    user_text: str,
    session_id: str | None,
    ticker: str,
    session_id_out: list[str],
    sides_out: list[str],
    filtered_holder: list[bool],
    err_out: list[BaseException],
    trace_id: str,
) -> Iterator[StreamQueueItem]:
    """把异步事件流桥接成同步可消费的流式迭代器。

    参数:
        chat_service: 聊天服务协议实例。
        user_text: 用户输入文本。
        session_id: 会话标识，首次提问可为 ``None``。
        ticker: 股票代码。
        session_id_out: 输出参数；后台消费成功后写入最新会话标识。
        sides_out: 输出参数；写入 warning/error/cancelled 侧边消息。
        filtered_holder: 输出参数；写入内容过滤标记。
        err_out: 输出参数；写入后台消费异常。
        trace_id: 请求链路追踪标识。

    返回值:
        逐块产出 ``StreamQueueItem`` 的同步迭代器。

    异常:
        TimeoutError: 首包或后续增量在超时时间内未到达且后台线程仍存活。
        BaseException: 后台异步消费抛出的原始异常。
    """

    chunk_queue: Queue[StreamQueueItem] = Queue()
    Log.info(
        f"[{trace_id}] 启动线程桥接消费: first_chunk_timeout_seconds={_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS}, "
        f"chunk_timeout_seconds={_STREAM_CHUNK_TIMEOUT_SECONDS}",
        module=MODULE,
    )
    worker = threading.Thread(
        target=_run_stream_worker,
        kwargs={
            "chat_service": chat_service,
            "user_text": user_text,
            "session_id": session_id,
            "ticker": ticker,
            "chunk_queue": chunk_queue,
            "session_id_out": session_id_out,
            "sides_out": sides_out,
            "filtered_holder": filtered_holder,
            "err_out": err_out,
            "trace_id": trace_id,
        },
        daemon=True,
    )
    worker.start()

    has_received_chunk = False
    yielded_chunk_count = 0
    while True:
        timeout_seconds = _STREAM_CHUNK_TIMEOUT_SECONDS if has_received_chunk else _STREAM_FIRST_CHUNK_TIMEOUT_SECONDS
        try:
            item = chunk_queue.get(timeout=timeout_seconds)
        except Empty as exception:
            if worker.is_alive():
                timeout_phase = "后续增量" if has_received_chunk else "首个增量"
                Log.error(
                    f"[{trace_id}] 等待流式{timeout_phase}超时，worker 仍在运行: timeout_seconds={timeout_seconds}",
                    module=MODULE,
                )
                raise TimeoutError(_STREAM_TIMEOUT_MESSAGE) from exception
            break
        if item.done:
            break
        has_received_chunk = True
        yielded_chunk_count += 1
        yield item

    if worker.is_alive():
        worker.join(timeout=0.1)
    else:
        worker.join()
    if err_out:
        raise err_out[0]
    Log.info(
        f"[{trace_id}] 线程桥接结束: yielded_chunk_count={yielded_chunk_count}, "
        f"side_messages={len(sides_out)}, filtered_flags={len(filtered_holder)}, "
        f"session_id_out={len(session_id_out)}",
        module=MODULE,
    )


def present_stream_side_effects(side_messages: list[str], filtered_flags: list[bool]) -> None:
    """展示流式输出副作用信息。"""

    for message in side_messages:
        st.warning(message)
    if any(filtered_flags):
        st.info(_FILTERED_INFO_MESSAGE)


__all__ = [
    "StreamQueueItem",
    "build_request_trace_id",
    "extract_stream_text",
    "present_stream_side_effects",
    "summarize_user_text",
    "sync_stream_via_asyncio",
]
