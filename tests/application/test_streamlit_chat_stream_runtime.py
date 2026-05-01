"""聊天流式运行时测试。

覆盖 ``chat/stream_runtime.py`` 的线程桥接、超时、取消与错误分支。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from queue import Empty, Queue
from typing import AsyncIterator, Iterator, cast

import pytest

from dayu.contracts.events import AppEvent, AppEventType
from dayu.host.protocols import ConversationSessionTurnExcerpt
from dayu.services.contracts import ChatPendingTurnView, ChatResumeRequest, ChatTurnRequest, ChatTurnSubmission
from dayu.services.protocols import ChatServiceProtocol
from dayu.web.streamlit.pages.chat import stream_runtime
from dayu.web.streamlit.pages.chat.stream_runtime import (
    CHAT_STREAM_RUNTIME_HANDLES,
    consume_chat_event_stream,
    new_stream_frame_state,
    run_stream_worker,
    ChatStreamRuntimeHandle,
    StreamQueueItem,
    poll_chat_stream_events,
)


@dataclass
class _FakeStreamlitState:
    """用于测试的精简 Streamlit 状态容器。"""

    session_state: dict[str, stream_runtime.ChatStreamFrameState]


class _StubWorker:
    """用于测试轮询逻辑的 worker 存根。"""

    def __init__(self, *, alive: bool) -> None:
        """初始化 worker 存根。

        参数:
            alive: 是否报告为存活状态。

        返回值:
            无。

        异常:
            无。
        """

        self._alive = alive

    def is_alive(self) -> bool:
        """返回 worker 是否仍在运行。

        参数:
            无。

        返回值:
            当前存活状态。

        异常:
            无。
        """

        return self._alive


class _StubChatService(ChatServiceProtocol):
    """用于测试 `_consume_chat_event_stream` 的聊天服务存根。"""

    def __init__(self, submission: ChatTurnSubmission) -> None:
        """初始化服务存根。

        参数:
            submission: 预置的提交结果。

        返回值:
            无。

        异常:
            无。
        """

        self._submission = submission

    async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
        """返回预置提交结果。"""

        _ = request
        return self._submission

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        """测试桩不支持恢复 pending turn。"""

        _ = request
        raise NotImplementedError

    def list_resumable_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
    ) -> list[ChatPendingTurnView]:
        """返回空 pending turn 列表。"""

        _ = (session_id, scene_name)
        return []

    def cleanup_stale_pending_turns(self, *, session_id: str | None = None) -> list[str]:
        """返回空清理结果。"""

        _ = session_id
        return []

    def list_conversation_session_turn_excerpts(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> list[ConversationSessionTurnExcerpt]:
        """返回空会话摘录。"""

        _ = (session_id, limit)
        return []

    def clear_session_history(self, session_id: str) -> None:
        """测试桩不支持清空历史。"""

        _ = session_id
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _reset_stream_runtime(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """为每个用例隔离 stream runtime 全局状态。"""

    CHAT_STREAM_RUNTIME_HANDLES.clear()
    monkeypatch.setattr(stream_runtime, "st", _FakeStreamlitState(session_state={}))
    yield
    CHAT_STREAM_RUNTIME_HANDLES.clear()


@pytest.mark.unit
class TestStreamQueueItem:
    """StreamQueueItem dataclass 测试。"""

    def test_default_values(self) -> None:
        item = StreamQueueItem(done=False)
        assert item.done is False
        assert item.kind == "content"
        assert item.chunk == ""
        assert item.event_type == "chunk"
        assert item.flag is False

    def test_reasoning_kind(self) -> None:
        item = StreamQueueItem(done=False, kind="reasoning", chunk="think")
        assert item.kind == "reasoning"
        assert item.chunk == "think"

    def test_done_item(self) -> None:
        item = StreamQueueItem(done=True, event_type="done")
        assert item.done is True
        assert item.event_type == "done"

    def test_frozen(self) -> None:
        item = StreamQueueItem(done=False)
        with pytest.raises(Exception):
            item.done = True  # type: ignore[misc]


@pytest.mark.unit
class TestChatStreamRuntimePolling:
    """stream_runtime 核心轮询路径测试。"""

    def test_first_chunk_timeout_sets_cancel_and_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stream_state_key = "stream-state"
        ticker = "AAPL"
        stream_runtime.st.session_state[stream_state_key] = new_stream_frame_state(trace_id="t1", session_id="s1")
        cancel_event = threading.Event()
        queue: Queue[StreamQueueItem] = Queue()
        runtime = ChatStreamRuntimeHandle(
            worker=cast(threading.Thread, _StubWorker(alive=True)),
            event_queue=queue,
            cancel_event=cancel_event,
            started_at=0.0,
            last_chunk_at=0.0,
            has_received_chunk=False,
        )
        CHAT_STREAM_RUNTIME_HANDLES[ticker] = runtime
        monkeypatch.setattr(stream_runtime.time, "perf_counter", lambda: 100.0)
        monkeypatch.setattr(stream_runtime, "_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS", 1.0)

        state = poll_chat_stream_events(ticker=ticker, stream_state_key=stream_state_key)

        assert state is not None
        assert state.done is True
        assert "超时" in state.error_message
        assert cancel_event.is_set() is True

    def test_chunk_interval_timeout_sets_cancel_and_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stream_state_key = "stream-state"
        ticker = "AAPL"
        stream_runtime.st.session_state[stream_state_key] = new_stream_frame_state(trace_id="t2", session_id="s2")
        cancel_event = threading.Event()
        queue: Queue[StreamQueueItem] = Queue()
        runtime = ChatStreamRuntimeHandle(
            worker=cast(threading.Thread, _StubWorker(alive=True)),
            event_queue=queue,
            cancel_event=cancel_event,
            started_at=10.0,
            last_chunk_at=0.0,
            has_received_chunk=True,
        )
        CHAT_STREAM_RUNTIME_HANDLES[ticker] = runtime
        monkeypatch.setattr(stream_runtime.time, "perf_counter", lambda: 100.0)
        monkeypatch.setattr(stream_runtime, "_STREAM_CHUNK_TIMEOUT_SECONDS", 1.0)

        state = poll_chat_stream_events(ticker=ticker, stream_state_key=stream_state_key)

        assert state is not None
        assert state.done is True
        assert "超时" in state.error_message
        assert cancel_event.is_set() is True

    def test_error_event_marks_state_done(self) -> None:
        stream_state_key = "stream-state"
        ticker = "AAPL"
        stream_runtime.st.session_state[stream_state_key] = new_stream_frame_state(trace_id="t3", session_id="s3")
        queue: Queue[StreamQueueItem] = Queue()
        queue.put(StreamQueueItem(done=False, event_type="error", chunk="boom"))
        runtime = ChatStreamRuntimeHandle(
            worker=cast(threading.Thread, _StubWorker(alive=True)),
            event_queue=queue,
            cancel_event=threading.Event(),
            started_at=0.0,
            last_chunk_at=0.0,
            has_received_chunk=False,
        )
        CHAT_STREAM_RUNTIME_HANDLES[ticker] = runtime

        state = poll_chat_stream_events(ticker=ticker, stream_state_key=stream_state_key)

        assert state is not None
        assert state.done is True
        assert state.error_message == "boom"

    def test_trailing_events_drained_after_worker_exit(self) -> None:
        stream_state_key = "stream-state"
        ticker = "AAPL"
        state = new_stream_frame_state(trace_id="t4", session_id="s4")
        state.done = True
        stream_runtime.st.session_state[stream_state_key] = state
        queue: Queue[StreamQueueItem] = Queue()
        queue.put(StreamQueueItem(done=False, event_type="side_message", chunk="warn"))
        queue.put(StreamQueueItem(done=False, event_type="filtered", flag=True))
        queue.put(StreamQueueItem(done=False, event_type="error", chunk="late error"))
        runtime = ChatStreamRuntimeHandle(
            worker=cast(threading.Thread, _StubWorker(alive=False)),
            event_queue=queue,
            cancel_event=threading.Event(),
            started_at=0.0,
            last_chunk_at=0.0,
            has_received_chunk=True,
        )
        CHAT_STREAM_RUNTIME_HANDLES[ticker] = runtime

        polled = poll_chat_stream_events(ticker=ticker, stream_state_key=stream_state_key)

        assert polled is not None
        assert polled.side_messages == ["warn"]
        assert polled.filtered_flags == [True]
        assert polled.error_message == "late error"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consume_chat_event_stream_stops_when_cancelled() -> None:
    """取消信号置位后应停止继续投递后续 chunk。"""

    event_queue: Queue[StreamQueueItem] = Queue()
    cancel_event = threading.Event()

    async def _event_stream() -> AsyncIterator[AppEvent]:
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="first")
        cancel_event.set()
        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="second")

    submission = ChatTurnSubmission(session_id="session-new", event_stream=_event_stream())
    chat_service = _StubChatService(submission=submission)

    await consume_chat_event_stream(
        chat_service=chat_service,
        user_text="hello",
        session_id=None,
        ticker="AAPL",
        event_queue=event_queue,
        cancel_event=cancel_event,
        trace_id="trace-test",
    )

    events: list[StreamQueueItem] = []
    while True:
        try:
            events.append(event_queue.get_nowait())
        except Empty:
            break

    chunk_values = [event.chunk for event in events if event.event_type == "chunk"]
    assert "first" in chunk_values
    assert "second" not in chunk_values


@pytest.mark.unit
def test_run_stream_worker_emits_error_and_done_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """worker 协程异常时应投递 error 与 done 事件。"""

    event_queue: Queue[StreamQueueItem] = Queue()
    cancel_event = threading.Event()

    async def _raise_in_consumer(
        *,
        chat_service: ChatServiceProtocol,
        user_text: str,
        session_id: str | None,
        ticker: str,
        event_queue: Queue[StreamQueueItem],
        cancel_event: threading.Event,
        trace_id: str,
    ) -> None:
        _ = (chat_service, user_text, session_id, ticker, event_queue, cancel_event, trace_id)
        raise RuntimeError("worker failed")

    monkeypatch.setattr(stream_runtime, "consume_chat_event_stream", _raise_in_consumer)

    run_stream_worker(
        chat_service=_StubChatService(
            submission=ChatTurnSubmission(session_id="s", event_stream=_empty_event_stream())
        ),
        user_text="u",
        session_id=None,
        ticker="AAPL",
        event_queue=event_queue,
        cancel_event=cancel_event,
        trace_id="trace-err",
    )

    first = event_queue.get_nowait()
    second = event_queue.get_nowait()
    assert first.event_type == "error"
    assert "worker failed" in first.chunk
    assert second.event_type == "done"


async def _empty_event_stream() -> AsyncIterator[AppEvent]:
    """返回一个空事件流。"""

    if False:
        yield AppEvent(type=AppEventType.DONE, payload="")
