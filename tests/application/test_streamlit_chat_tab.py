"""Streamlit 交互式分析 Tab 的集成与 UI 相关单测。

完整导入 ``dayu.web.streamlit.pages.chat_tab`` 依赖 Python 3.11+（标准库
``datetime.UTC``、``enum.StrEnum`` 等）；在 3.10 环境下本模块在导入阶段跳过。
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time

import pytest

if sys.version_info < (3, 11):
    pytest.skip("chat_tab 集成测试需要 Python 3.11+", allow_module_level=True)

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from dayu.contracts.events import AppEvent, AppEventType
from dayu.services.contracts import (
    ChatPendingTurnView,
    ChatResumeRequest,
    ChatTurnRequest,
    ChatTurnSubmission,
    SessionResolutionPolicy,
)
from dayu.services.protocols import ChatServiceProtocol, HostAdminServiceProtocol, ReplyDeliveryServiceProtocol
from dayu.web.streamlit.pages import chat_tab
from dayu.web.streamlit.pages.chat import chat_stream_bridge
from dayu.web.streamlit.pages.chat_tab import (
    ChatServiceClient,
    create_chat_service_client,
)


def test_create_chat_service_client_and_reserved_accessors() -> None:
    """验证工厂装配与预留访问器返回同一实例。"""

    chat = MagicMock(spec=ChatServiceProtocol)
    admin = MagicMock(spec=HostAdminServiceProtocol)
    delivery = MagicMock(spec=ReplyDeliveryServiceProtocol)
    client = create_chat_service_client(
        chat_service=chat,
        host_admin_service=admin,
        reply_delivery_service=delivery,
    )
    assert isinstance(client, ChatServiceClient)
    assert client.chat_service is chat
    assert client.reserved_admin_service() is admin
    assert client.reserved_reply_delivery_service() is delivery


@dataclass
class _FakeChatService:
    """测试用聊天服务：返回固定事件流。"""

    events: list[AppEvent]
    last_request: ChatTurnRequest | None = None

    async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
        """返回包含预置事件的提交句柄。"""

        self.last_request = request

        async def _stream() -> AsyncIterator[AppEvent]:
            for event in self.events:
                yield event

        return ChatTurnSubmission(session_id="fake-session", event_stream=_stream())

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        """未在单测中使用。"""

        raise NotImplementedError

    def list_resumable_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
    ) -> list[ChatPendingTurnView]:
        """未在单测中使用。"""

        return []


def _make_selected_stock(ticker: str) -> chat_tab.WatchlistItem:
    """构造测试用自选股条目。"""

    return chat_tab.WatchlistItem(
        ticker=ticker,
        company_name=f"{ticker} Inc.",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


def test_sync_stream_via_asyncio_yields_text_chunks() -> None:
    """验证后台线程消费异步流后，同步迭代器按块产出。"""

    fake = _FakeChatService(
        events=[
            AppEvent(type=AppEventType.CONTENT_DELTA, payload="p1", meta={}),
            AppEvent(type=AppEventType.CONTENT_DELTA, payload="p2", meta={}),
        ]
    )
    session_holder: list[str] = []
    sides_out: list[str] = []
    filtered_holder: list[bool] = []
    err_out: list[BaseException] = []
    stream_items = list(
        chat_tab._sync_stream_via_asyncio(
            fake,
            user_text="hello",
            session_id=None,
            ticker="TEST",
            session_id_out=session_holder,
            sides_out=sides_out,
            filtered_holder=filtered_holder,
            err_out=err_out,
            trace_id="test-trace",
        )
    )
    assert [item.kind for item in stream_items] == ["content", "content"]
    assert [item.chunk for item in stream_items] == ["p1", "p2"]
    assert session_holder == ["fake-session"]
    assert sides_out == []
    assert filtered_holder == [False]
    assert err_out == []
    assert fake.last_request is not None
    assert fake.last_request.session_resolution_policy is SessionResolutionPolicy.AUTO
    assert fake.last_request.session_id is None


def test_sync_stream_via_asyncio_streams_before_all_events_finished() -> None:
    """验证同步桥接会边消费边产出，而不是等待事件流整体结束。"""

    @dataclass
    class _DelayedChunkChat(_FakeChatService):
        """测试用聊天服务：在两段内容间引入可观测延迟。"""

        inter_chunk_delay_seconds: float = 0.0

        async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
            """返回含分段延迟的事件流。"""

            self.last_request = request

            async def _stream() -> AsyncIterator[AppEvent]:
                yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="first", meta={})
                await asyncio.sleep(self.inter_chunk_delay_seconds)
                yield AppEvent(type=AppEventType.CONTENT_DELTA, payload="second", meta={})

            return ChatTurnSubmission(session_id="delayed-session", event_stream=_stream())

    delayed_chat = _DelayedChunkChat(events=[], inter_chunk_delay_seconds=0.12)
    session_holder: list[str] = []
    sides_out: list[str] = []
    filtered_holder: list[bool] = []
    err_out: list[BaseException] = []
    stream_iter = chat_tab._sync_stream_via_asyncio(
        delayed_chat,
        user_text="hello",
        session_id=None,
        ticker="TEST",
        session_id_out=session_holder,
        sides_out=sides_out,
        filtered_holder=filtered_holder,
        err_out=err_out,
        trace_id="test-trace",
    )

    started_at = time.perf_counter()
    first_item = next(stream_iter)
    first_yield_elapsed = time.perf_counter() - started_at
    assert first_item.chunk == "first"
    assert first_yield_elapsed < 0.08

    second_item = next(stream_iter)
    total_elapsed = time.perf_counter() - started_at
    assert second_item.chunk == "second"
    assert total_elapsed >= 0.10
    assert session_holder == ["delayed-session"]
    assert sides_out == []
    assert filtered_holder == [False]
    assert err_out == []

    with pytest.raises(StopIteration):
        next(stream_iter)


def test_sync_stream_via_asyncio_yields_text_field_chunks() -> None:
    """验证 content_delta 为 dict(text) 时仍能进入同步迭代器。"""

    fake = _FakeChatService(
        events=[
            AppEvent(type=AppEventType.CONTENT_DELTA, payload={"text": "p1"}, meta={}),
            AppEvent(type=AppEventType.CONTENT_DELTA, payload={"text": "p2"}, meta={}),
        ]
    )
    session_holder: list[str] = []
    sides_out: list[str] = []
    filtered_holder: list[bool] = []
    err_out: list[BaseException] = []
    stream_items = list(
        chat_tab._sync_stream_via_asyncio(
            fake,
            user_text="hello",
            session_id=None,
            ticker="TEST",
            session_id_out=session_holder,
            sides_out=sides_out,
            filtered_holder=filtered_holder,
            err_out=err_out,
            trace_id="test-trace",
        )
    )
    assert [item.kind for item in stream_items] == ["content", "content"]
    assert [item.chunk for item in stream_items] == ["p1", "p2"]
    assert session_holder == ["fake-session"]
    assert err_out == []


def test_sync_stream_via_asyncio_yields_reasoning_chunks() -> None:
    """验证 reasoning_delta 会进入同步迭代器（与 Web 主文展示一致）。"""

    fake = _FakeChatService(
        events=[
            AppEvent(type=AppEventType.REASONING_DELTA, payload="r1", meta={}),
            AppEvent(type=AppEventType.REASONING_DELTA, payload="r2", meta={}),
        ]
    )
    session_holder: list[str] = []
    sides_out: list[str] = []
    filtered_holder: list[bool] = []
    err_out: list[BaseException] = []
    stream_items = list(
        chat_tab._sync_stream_via_asyncio(
            fake,
            user_text="hello",
            session_id=None,
            ticker="TEST",
            session_id_out=session_holder,
            sides_out=sides_out,
            filtered_holder=filtered_holder,
            err_out=err_out,
            trace_id="test-trace",
        )
    )
    assert [item.kind for item in stream_items] == ["reasoning", "reasoning"]
    assert [item.chunk for item in stream_items] == ["r1", "r2"]
    assert session_holder == ["fake-session"]
    assert err_out == []


def test_sync_stream_via_asyncio_normalizes_escaped_newline_chunks() -> None:
    """验证流式文本块中的字面量 ``\\n`` 会转为真实换行。"""

    fake = _FakeChatService(
        events=[
            AppEvent(type=AppEventType.CONTENT_DELTA, payload="itive...\\n- SEC EDGAR", meta={}),
        ]
    )
    session_holder: list[str] = []
    sides_out: list[str] = []
    filtered_holder: list[bool] = []
    err_out: list[BaseException] = []
    stream_items = list(
        chat_tab._sync_stream_via_asyncio(
            fake,
            user_text="hello",
            session_id=None,
            ticker="TEST",
            session_id_out=session_holder,
            sides_out=sides_out,
            filtered_holder=filtered_holder,
            err_out=err_out,
            trace_id="test-trace",
        )
    )
    assert [item.kind for item in stream_items] == ["content"]
    assert [item.chunk for item in stream_items] == ["itive...\n- SEC EDGAR"]
    assert session_holder == ["fake-session"]
    assert err_out == []


def test_sync_stream_via_asyncio_forwards_session_id_for_followup() -> None:
    """续聊时应把显式 ``session_id`` 原样传入 ``ChatTurnRequest``。"""

    fake = _FakeChatService(events=[AppEvent(type=AppEventType.CONTENT_DELTA, payload="z", meta={})])
    session_holder: list[str] = []
    sides_out: list[str] = []
    filtered_holder: list[bool] = []
    err_out: list[BaseException] = []
    list(
        chat_tab._sync_stream_via_asyncio(
            fake,
            user_text="q",
            session_id="prior-sid",
            ticker="X",
            session_id_out=session_holder,
            sides_out=sides_out,
            filtered_holder=filtered_holder,
            err_out=err_out,
            trace_id="test-trace",
        )
    )
    assert fake.last_request is not None
    assert fake.last_request.session_id == "prior-sid"
    assert fake.last_request.session_resolution_policy is SessionResolutionPolicy.AUTO


def test_sync_stream_via_asyncio_propagates_submit_error() -> None:
    """验证 submit_turn 异常会冒泡到同步消费端。"""

    class _BrokenChat(_FakeChatService):
        """总是抛出异常的假服务。"""

        async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
            """抛出固定异常。"""

            raise ValueError("bad turn")

    broken = _BrokenChat(events=[])
    session_holder: list[str] = []
    sides_out: list[str] = []
    filtered_holder: list[bool] = []
    err_out: list[BaseException] = []
    gen = chat_tab._sync_stream_via_asyncio(
        broken,
        user_text="x",
        session_id=None,
        ticker="T",
        session_id_out=session_holder,
        sides_out=sides_out,
        filtered_holder=filtered_holder,
        err_out=err_out,
        trace_id="test-trace",
    )
    with pytest.raises(ValueError, match="bad turn"):
        list(gen)


def test_sync_stream_via_asyncio_raises_timeout_when_worker_stalled(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证后台线程长时间无输出时会抛出超时异常。"""

    class _StalledThread:
        """不会执行目标函数且始终存活的线程桩。"""

        def __init__(self, *, target: object, kwargs: dict[str, object], daemon: bool) -> None:
            del target, kwargs, daemon

        def start(self) -> None:
            """启动空操作。"""

        def is_alive(self) -> bool:
            """始终返回存活，模拟线程卡住。"""

            return True

        def join(self, timeout: float | None = None) -> None:
            """空实现 join。"""

            del timeout

    monkeypatch.setattr(chat_stream_bridge.threading, "Thread", _StalledThread)
    monkeypatch.setattr(chat_stream_bridge, "_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(chat_stream_bridge, "_STREAM_CHUNK_TIMEOUT_SECONDS", 0.01)

    session_holder: list[str] = []
    sides_out: list[str] = []
    filtered_holder: list[bool] = []
    err_out: list[BaseException] = []
    fake_chat = MagicMock(spec=ChatServiceProtocol)
    generator = chat_tab._sync_stream_via_asyncio(
        fake_chat,
        user_text="timeout-case",
        session_id=None,
        ticker="TEST",
        session_id_out=session_holder,
        sides_out=sides_out,
        filtered_holder=filtered_holder,
        err_out=err_out,
        trace_id="test-trace",
    )
    with pytest.raises(TimeoutError, match="等待模型输出超时"):
        list(generator)


def test_sync_stream_via_asyncio_allows_slow_first_chunk_within_first_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证首包延迟超过常规增量超时但小于首包超时时仍可成功返回。"""

    @dataclass
    class _SlowFirstChunkChat(_FakeChatService):
        """首包前先等待一小段时间的假服务。"""

        first_chunk_delay_seconds: float = 0.0

        async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
            """返回含首包延迟的事件流。"""

            self.last_request = request

            async def _stream() -> AsyncIterator[AppEvent]:
                await asyncio.sleep(self.first_chunk_delay_seconds)
                for event in self.events:
                    yield event

            return ChatTurnSubmission(session_id="slow-first-session", event_stream=_stream())

    monkeypatch.setattr(chat_stream_bridge, "_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(chat_stream_bridge, "_STREAM_CHUNK_TIMEOUT_SECONDS", 0.05)

    fake = _SlowFirstChunkChat(
        events=[AppEvent(type=AppEventType.CONTENT_DELTA, payload="late", meta={})],
        first_chunk_delay_seconds=0.08,
    )
    session_holder: list[str] = []
    sides_out: list[str] = []
    filtered_holder: list[bool] = []
    err_out: list[BaseException] = []

    stream_items = list(
        chat_tab._sync_stream_via_asyncio(
            fake,
            user_text="slow-first-chunk",
            session_id=None,
            ticker="TEST",
            session_id_out=session_holder,
            sides_out=sides_out,
            filtered_holder=filtered_holder,
            err_out=err_out,
            trace_id="test-trace",
        )
    )

    assert [item.kind for item in stream_items] == ["content"]
    assert [item.chunk for item in stream_items] == ["late"]
    assert session_holder == ["slow-first-session"]
    assert err_out == []


@patch.object(chat_tab.st, "session_state", new_callable=dict)
def test_stream_and_collect_assistant_reply_renders_markdown_incrementally(
    _mock_session_state: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证助手流式回复会按 Markdown 逐步刷新占位符。"""

    class _FakePlaceholder:
        """测试用 Markdown 占位符。"""

        def __init__(self) -> None:
            self.rendered: list[str] = []

        def markdown(self, body: str) -> None:
            """记录每次渲染文本。"""

            self.rendered.append(body)

    thinking_placeholder = _FakePlaceholder()
    answer_placeholder = _FakePlaceholder()

    def _fake_sync_stream_via_asyncio(*args: object, **kwargs: object) -> Iterator[chat_tab._StreamQueueItem]:
        """返回固定文本块的同步迭代器桩。"""

        del args, kwargs
        return iter(
            [
                chat_tab._StreamQueueItem(done=False, kind="reasoning", chunk="先审题。\n"),
                chat_tab._StreamQueueItem(done=False, kind="content", chunk="## 结论\n"),
                chat_tab._StreamQueueItem(done=False, kind="content", chunk="- 要点一\n"),
                chat_tab._StreamQueueItem(done=False, kind="content", chunk="- 要点二"),
            ]
        )

    monkeypatch.setattr(chat_tab, "_sync_stream_via_asyncio", _fake_sync_stream_via_asyncio)
    placeholders = iter([thinking_placeholder, answer_placeholder])
    monkeypatch.setattr(chat_tab.st, "empty", lambda: next(placeholders))

    fake_client = create_chat_service_client(
        chat_service=MagicMock(spec=ChatServiceProtocol),
        host_admin_service=MagicMock(spec=HostAdminServiceProtocol),
        reply_delivery_service=MagicMock(spec=ReplyDeliveryServiceProtocol),
    )

    monkeypatch.setattr(chat_tab.st, "expander", lambda _label, expanded=False: contextlib.nullcontext())
    assistant_reasoning_text, assistant_text, side_messages, filtered_flags = chat_tab._stream_and_collect_assistant_reply(
        service_client=fake_client,
        user_text="请输出 markdown",
        ticker="AAPL",
        session_id=None,
        session_id_key="session-key",
        trace_id="trace-id",
    )

    assert assistant_reasoning_text == "先审题。\n"
    assert assistant_text == "## 结论\n- 要点一\n- 要点二"
    assert side_messages == []
    assert filtered_flags == []
    assert thinking_placeholder.rendered == [
        "先审题。\n",
    ]
    assert answer_placeholder.rendered == [
        "## 结论\n",
        "## 结论\n- 要点一\n",
        "## 结论\n- 要点一\n- 要点二",
    ]


@patch.object(chat_tab.st, "session_state", new_callable=dict)
def test_stream_and_collect_assistant_reply_normalizes_cross_chunk_escaped_newline(
    _mock_session_state: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 ``\\`` 与 ``n`` 跨 chunk 分裂时仍会被规范化为换行。"""

    class _FakePlaceholder:
        """测试用 Markdown 占位符。"""

        def __init__(self) -> None:
            self.rendered: list[str] = []

        def markdown(self, body: str) -> None:
            """记录每次渲染文本。"""

            self.rendered.append(body)

    thinking_placeholder = _FakePlaceholder()
    answer_placeholder = _FakePlaceholder()

    def _fake_sync_stream_via_asyncio(*args: object, **kwargs: object) -> Iterator[chat_tab._StreamQueueItem]:
        """返回跨 chunk 分裂 ``\\n`` 的文本块。"""

        del args, kwargs
        return iter(
            [
                chat_tab._StreamQueueItem(done=False, kind="content", chunk="“ Performance”  \\"),
                chat_tab._StreamQueueItem(done=False, kind="content", chunk="n- SEC EDGAR | Form”"),
            ]
        )

    monkeypatch.setattr(chat_tab, "_sync_stream_via_asyncio", _fake_sync_stream_via_asyncio)
    placeholders = iter([thinking_placeholder, answer_placeholder])
    monkeypatch.setattr(chat_tab.st, "empty", lambda: next(placeholders))

    fake_client = create_chat_service_client(
        chat_service=MagicMock(spec=ChatServiceProtocol),
        host_admin_service=MagicMock(spec=HostAdminServiceProtocol),
        reply_delivery_service=MagicMock(spec=ReplyDeliveryServiceProtocol),
    )

    monkeypatch.setattr(chat_tab.st, "expander", lambda _label, expanded=False: contextlib.nullcontext())
    assistant_reasoning_text, assistant_text, side_messages, filtered_flags = chat_tab._stream_and_collect_assistant_reply(
        service_client=fake_client,
        user_text="请输出 markdown",
        ticker="AAPL",
        session_id=None,
        session_id_key="session-key",
        trace_id="trace-id",
    )

    assert assistant_reasoning_text == ""
    assert assistant_text == "“ Performance”  \n- SEC EDGAR | Form”"
    assert side_messages == []
    assert filtered_flags == []
    assert thinking_placeholder.rendered == []
    assert answer_placeholder.rendered == [
        "“ Performance”  \\",
        "“ Performance”  \n- SEC EDGAR | Form”",
    ]


@patch.object(chat_tab.st, "code")
@patch.object(chat_tab.st, "markdown")
@patch.object(chat_tab.st, "expander", return_value=contextlib.nullcontext())
@patch.object(chat_tab.st, "chat_message", return_value=contextlib.nullcontext())
@patch.object(
    chat_tab.st,
    "columns",
    return_value=[contextlib.nullcontext(), contextlib.nullcontext()],
)
def test_render_message_history_renders_copyable_markdown_for_assistant(
    _mock_columns: MagicMock,
    _mock_chat_message: MagicMock,
    _mock_expander: MagicMock,
    _mock_markdown: MagicMock,
    mock_code: MagicMock,
) -> None:
    """助手消息应额外渲染可复制的截图版 Markdown。"""

    chat_tab._render_message_history(
        [
            chat_tab._ChatMessage(role="assistant", content="## 结论\n- 要点一"),
        ]
    )
    mock_code.assert_called_once_with("## 结论\n- 要点一", language="markdown")


@patch.object(chat_tab.st, "code")
@patch.object(chat_tab.st, "markdown")
@patch.object(chat_tab.st, "expander", return_value=contextlib.nullcontext())
@patch.object(chat_tab.st, "chat_message", return_value=contextlib.nullcontext())
@patch.object(
    chat_tab.st,
    "columns",
    return_value=[contextlib.nullcontext(), contextlib.nullcontext()],
)
def test_render_message_history_skips_copyable_markdown_for_user(
    _mock_columns: MagicMock,
    _mock_chat_message: MagicMock,
    _mock_expander: MagicMock,
    _mock_markdown: MagicMock,
    mock_code: MagicMock,
) -> None:
    """用户消息不应渲染截图版 Markdown 复制区域。"""

    chat_tab._render_message_history(
        [
            chat_tab._ChatMessage(role="user", content="用户问题"),
        ]
    )
    mock_code.assert_not_called()


@patch.object(chat_tab.st, "info")
@patch.object(chat_tab.st, "warning")
def test_present_stream_side_effects_calls_streamlit(
    mock_warning: MagicMock,
    mock_info: MagicMock,
) -> None:
    """验证侧边展示会调用 ``st.warning`` 与 ``st.info``。"""

    chat_tab._present_stream_side_effects(["a", "b"], [True])
    mock_warning.assert_any_call("a")
    mock_warning.assert_any_call("b")
    mock_info.assert_called_once()


@patch.object(chat_tab.st, "info")
@patch.object(chat_tab.st, "warning")
def test_present_stream_side_effects_skips_info_when_not_filtered(
    mock_warning: MagicMock,
    mock_info: MagicMock,
) -> None:
    """验证未过滤时不调用 ``st.info``。"""

    chat_tab._present_stream_side_effects([], [False])
    mock_warning.assert_not_called()
    mock_info.assert_not_called()


@patch.object(chat_tab.st, "info")
@patch.object(
    chat_tab.st,
    "columns",
    return_value=[contextlib.nullcontext(), contextlib.nullcontext()],
)
@patch.object(chat_tab.st, "caption")
@patch.object(chat_tab.st, "button", return_value=False)
@patch.object(chat_tab.st, "text_area", return_value="")
def test_render_chat_tab_without_submit_keeps_service_warning_hidden(
    _mock_text_area: MagicMock,
    _mock_button: MagicMock,
    _mock_caption: MagicMock,
    _mock_columns: MagicMock,
    mock_info: MagicMock,
) -> None:
    """未点击提交时，即使服务缺失也不应出现服务告警。"""

    chat_tab.render_chat_tab(
        selected_stock=_make_selected_stock("AAA"),
        service_client=None,
    )
    mock_info.assert_not_called()


@patch.object(chat_tab.st, "text_area", return_value="用户问题")
@patch.object(chat_tab.st, "button", return_value=True)
@patch.object(
    chat_tab.st,
    "columns",
    return_value=[contextlib.nullcontext(), contextlib.nullcontext()],
)
@patch.object(chat_tab.st, "warning")
def test_render_chat_tab_when_service_none(
    mock_warning: MagicMock,
    _mock_columns: MagicMock,
    _mock_button: MagicMock,
    _mock_text_area: MagicMock,
) -> None:
    """聊天客户端缺失且已提交时应告警。"""

    chat_tab.render_chat_tab(
        selected_stock=_make_selected_stock("AAA"),
        service_client=None,
    )
    mock_warning.assert_called()


@patch.object(chat_tab.st, "session_state", new_callable=dict)
@patch.object(chat_tab.st, "markdown")
@patch.object(
    chat_tab.st,
    "columns",
    return_value=[contextlib.nullcontext(), contextlib.nullcontext()],
)
@patch.object(chat_tab.st, "button", return_value=False)
@patch.object(chat_tab.st, "text_area")
def test_render_chat_tab_first_screen_widgets(
    mock_text_area: MagicMock,
    mock_button: MagicMock,
    mock_columns: MagicMock,
    mock_markdown: MagicMock,
    mock_session_state: dict[str, object],
) -> None:
    """首屏无历史时应渲染引导与输入控件。"""

    fake_client = create_chat_service_client(
        chat_service=_FakeChatService([]),
        host_admin_service=MagicMock(spec=HostAdminServiceProtocol),
        reply_delivery_service=MagicMock(spec=ReplyDeliveryServiceProtocol),
    )
    chat_tab.render_chat_tab(
        selected_stock=_make_selected_stock("BBB"),
        service_client=fake_client,
    )
    assert mock_markdown.call_count >= 2
    assert mock_button.call_count >= 1
    mock_button.assert_called_with("🚀 开始分析", type="primary", key="chat_tab_BBB_send_button")
    mock_columns.assert_not_called()
    mock_text_area.assert_called_once()


@patch.object(chat_tab.st, "session_state", new_callable=dict)
@patch.object(chat_tab.st, "text_area")
@patch.object(chat_tab.st, "button", return_value=False)
def test_render_chat_tab_text_area_uses_non_empty_label(
    _mock_button: MagicMock,
    mock_text_area: MagicMock,
    mock_session_state: dict[str, object],
) -> None:
    """输入框应使用非空 label，避免 Streamlit 空标签告警。"""

    fake_client = create_chat_service_client(
        chat_service=_FakeChatService([]),
        host_admin_service=MagicMock(spec=HostAdminServiceProtocol),
        reply_delivery_service=MagicMock(spec=ReplyDeliveryServiceProtocol),
    )
    chat_tab.render_chat_tab(
        selected_stock=_make_selected_stock("BBB"),
        service_client=fake_client,
    )
    mock_text_area.assert_called_once_with(
        "输入你的分析问题",
        key="chat_tab_BBB_input_text",
        placeholder="例如：公司的核心竞争力是什么？增长的主要驱动因素有哪些？",
        height=120,
    )
    assert isinstance(mock_session_state["chat_tab_BBB_input_text"], str)


@patch.object(chat_tab.st, "session_state", new_callable=dict)
def test_apply_pending_input_reset_clears_input_before_widget_instantiation(
    mock_session_state: dict[str, object],
) -> None:
    """验证延迟清空标记会在控件实例化前清空输入。"""

    input_key = "chat_tab_AAPL_input_text"
    clear_input_key = "chat_tab_AAPL_clear_input_pending"
    mock_session_state[input_key] = "待清空文本"
    mock_session_state[clear_input_key] = True

    chat_tab._apply_pending_input_reset(
        input_key=input_key,
        clear_input_key=clear_input_key,
    )

    assert mock_session_state[input_key] == ""
    assert mock_session_state[clear_input_key] is False


@patch.object(chat_tab.st, "session_state", new_callable=dict)
@patch.object(chat_tab.st, "rerun")
@patch.object(chat_tab.st, "error")
@patch.object(
    chat_tab.st,
    "columns",
    return_value=[contextlib.nullcontext(), contextlib.nullcontext()],
)
@patch.object(chat_tab.st, "chat_message")
@patch.object(chat_tab.st, "text_area", return_value="用户问题")
@patch.object(chat_tab.st, "button", return_value=True)
@patch.object(chat_tab, "_stream_and_collect_assistant_reply", return_value=("", "助手回复", [], []))
def test_render_chat_tab_marks_clear_pending_and_reruns_after_submit(
    _mock_stream: MagicMock,
    _mock_button: MagicMock,
    _mock_text_area: MagicMock,
    mock_chat_message: MagicMock,
    _mock_columns: MagicMock,
    mock_error: MagicMock,
    mock_rerun: MagicMock,
    mock_session_state: dict[str, object],
) -> None:
    """验证提交后通过延迟标记 + rerun 清空输入框。"""

    message_ctx = MagicMock()
    mock_chat_message.return_value = message_ctx
    fake_client = create_chat_service_client(
        chat_service=_FakeChatService([]),
        host_admin_service=MagicMock(spec=HostAdminServiceProtocol),
        reply_delivery_service=MagicMock(spec=ReplyDeliveryServiceProtocol),
    )

    chat_tab.render_chat_tab(
        selected_stock=_make_selected_stock("AAPL"),
        service_client=fake_client,
    )

    assert mock_session_state["chat_tab_AAPL_clear_input_pending"] is True
    mock_error.assert_not_called()
    mock_rerun.assert_called_once()


@patch.object(chat_tab.st, "session_state", new_callable=dict)
@patch.object(chat_tab.st, "rerun")
@patch.object(chat_tab.st, "error")
@patch.object(chat_tab.st, "warning")
@patch.object(
    chat_tab.st,
    "columns",
    return_value=[contextlib.nullcontext(), contextlib.nullcontext()],
)
@patch.object(chat_tab.st, "chat_message")
@patch.object(chat_tab.st, "text_area", return_value="用户问题")
@patch.object(chat_tab.st, "button", return_value=True)
@patch.object(chat_tab, "_stream_and_collect_assistant_reply", return_value=("", "", ["Invalid API Key"], []))
def test_render_chat_tab_keeps_error_visible_when_stream_failed(
    _mock_stream: MagicMock,
    _mock_button: MagicMock,
    _mock_text_area: MagicMock,
    mock_chat_message: MagicMock,
    _mock_columns: MagicMock,
    mock_warning: MagicMock,
    mock_error: MagicMock,
    mock_rerun: MagicMock,
    mock_session_state: dict[str, object],
) -> None:
    """验证空回复且存在侧边错误时不触发 rerun，避免错误提示被瞬间覆盖。"""

    message_ctx = MagicMock()
    mock_chat_message.return_value = message_ctx
    fake_client = create_chat_service_client(
        chat_service=_FakeChatService([]),
        host_admin_service=MagicMock(spec=HostAdminServiceProtocol),
        reply_delivery_service=MagicMock(spec=ReplyDeliveryServiceProtocol),
    )

    chat_tab.render_chat_tab(
        selected_stock=_make_selected_stock("AAPL"),
        service_client=fake_client,
    )

    mock_error.assert_not_called()
    mock_warning.assert_any_call("本轮未收到可展示的回复，请稍后重试或检查模型与网络配置。")
    mock_warning.assert_any_call("Invalid API Key")
    mock_rerun.assert_not_called()
    assert mock_session_state["chat_tab_AAPL_clear_input_pending"] is False


@patch.object(chat_tab.st, "session_state", new_callable=dict)
@patch.object(chat_tab.st, "rerun")
@patch.object(chat_tab.st, "error")
@patch.object(
    chat_tab.st,
    "columns",
    return_value=[contextlib.nullcontext(), contextlib.nullcontext()],
)
@patch.object(chat_tab.st, "chat_message")
@patch.object(chat_tab.st, "text_area", return_value="用户问题")
@patch.object(chat_tab.st, "button", return_value=True)
@patch.object(chat_tab, "_stream_and_collect_assistant_reply", side_effect=RuntimeError("boom"))
def test_render_chat_tab_shows_error_when_stream_raises_exception(
    _mock_stream: MagicMock,
    _mock_button: MagicMock,
    _mock_text_area: MagicMock,
    mock_chat_message: MagicMock,
    _mock_columns: MagicMock,
    mock_error: MagicMock,
    mock_rerun: MagicMock,
    mock_session_state: dict[str, object],
) -> None:
    """验证流式执行抛异常时会展示错误且不触发 rerun。"""

    message_ctx = MagicMock()
    mock_chat_message.return_value = message_ctx
    fake_client = create_chat_service_client(
        chat_service=_FakeChatService([]),
        host_admin_service=MagicMock(spec=HostAdminServiceProtocol),
        reply_delivery_service=MagicMock(spec=ReplyDeliveryServiceProtocol),
    )

    chat_tab.render_chat_tab(
        selected_stock=_make_selected_stock("AAPL"),
        service_client=fake_client,
    )

    mock_error.assert_called_once()
    mock_rerun.assert_not_called()
    assert mock_session_state["chat_tab_AAPL_clear_input_pending"] is False


def test_stream_chat_events_exports_fold() -> None:
    """事件折叠函数应从 stream_chat_events 模块导入。"""

    from dayu.web.streamlit.stream_chat_events import fold_app_events_to_assistant_text

    text, _s, _f = fold_app_events_to_assistant_text([AppEvent(type=AppEventType.CONTENT_DELTA, payload="z", meta={})])
    assert text == "z"
