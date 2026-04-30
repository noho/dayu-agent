"""聊天页逻辑测试。

覆盖 ``chat_tab.py`` 中与 UI 分支强相关的纯逻辑行为。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from dayu.host.protocols import (
    ConversationClearPartiallyAppliedError,
    ConversationClearRejectedError,
    ConversationClearStaleError,
    ConversationSessionTurnExcerpt,
)
from dayu.services.protocols import ChatServiceProtocol
from dayu.web.streamlit.pages import chat_tab as chat_tab_module
from dayu.web.streamlit.pages.chat.utils import build_chat_session_id
from dayu.web.streamlit.pages.chat_tab import ChatMessage, load_history_for_ticker, perform_clear_session_history


@pytest.mark.unit
class TestChatMessage:
    """ChatMessage dataclass 测试。"""

    def test_default_reasoning_content(self) -> None:
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.reasoning_content == ""

    def test_with_reasoning_content(self) -> None:
        msg = ChatMessage(role="assistant", content="reply", reasoning_content="thinking...")
        assert msg.role == "assistant"
        assert msg.content == "reply"
        assert msg.reasoning_content == "thinking..."

    def test_frozen(self) -> None:
        msg = ChatMessage(role="user", content="hello")
        with pytest.raises(Exception):
            msg.content = "changed"  # type: ignore[misc]


class _ClearHistoryServiceStub:
    """用于测试清空历史异常分支的服务桩。"""

    def __init__(self, clear_side_effects: list[Exception | None]) -> None:
        """初始化服务桩。

        参数:
            clear_side_effects: 每次调用 ``clear_session_history`` 的行为序列；
                ``None`` 表示成功，异常对象表示抛错。

        返回值:
            无。

        异常:
            无。
        """

        self._clear_side_effects = list(clear_side_effects)
        self.clear_call_count = 0

    def clear_session_history(self, session_id: str) -> None:
        """按预置序列执行清空历史。

        参数:
            session_id: 会话 ID。

        返回值:
            无。

        异常:
            Exception: 预置序列项为异常时抛出对应异常。
        """

        _ = session_id
        self.clear_call_count += 1
        if not self._clear_side_effects:
            return
        current = self._clear_side_effects.pop(0)
        if current is not None:
            raise current


class _HistoryServiceStub:
    """用于测试历史加载分支的服务桩。"""

    def __init__(
        self,
        *,
        turns: list[ConversationSessionTurnExcerpt] | None = None,
        error: Exception | None = None,
    ) -> None:
        """初始化历史加载服务桩。

        参数:
            turns: 预置历史轮次。
            error: 预置异常；存在时优先抛出。

        返回值:
            无。

        异常:
            无。
        """

        self._turns = turns or []
        self._error = error
        self.called_session_id: str | None = None
        self.called_limit: int | None = None

    def list_conversation_session_turn_excerpts(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> list[ConversationSessionTurnExcerpt]:
        """返回预置历史轮次或抛出预置异常。"""

        self.called_session_id = session_id
        self.called_limit = limit
        if self._error is not None:
            raise self._error
        return self._turns


@dataclass
class _FakeChatTabStreamlit:
    """用于测试 chat_tab UI 分支的 Streamlit 存根。"""

    session_state: dict[str, object]
    warnings: list[str]
    errors: list[str]
    rerun_called: bool = False

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def rerun(self) -> None:
        self.rerun_called = True


@pytest.mark.unit
class TestPerformClearSessionHistory:
    """_perform_clear_session_history 异常分支测试。"""

    def test_key_error_shows_warning_without_rerun(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_st = _FakeChatTabStreamlit(
            session_state={"messages": ["existing"], "input": "hello", "clear_input": True},
            warnings=[],
            errors=[],
        )
        service = _ClearHistoryServiceStub(clear_side_effects=[KeyError("missing")])
        clear_calls: list[str] = []
        monkeypatch.setattr(chat_tab_module, "st", fake_st)
        monkeypatch.setattr(
            chat_tab_module,
            "clear_chat_stream_runtime",
            lambda *, ticker, stream_state_key: clear_calls.append(f"{ticker}:{stream_state_key}"),
        )

        perform_clear_session_history(
            chat_service=cast(ChatServiceProtocol, service),
            session_id="sid",
            ticker="AAPL",
            stream_state_key="stream",
            message_state_key="messages",
            input_key="input",
            clear_input_key="clear_input",
        )

        assert service.clear_call_count == 1
        assert fake_st.warnings == ["会话不存在或已被删除。"]
        assert fake_st.errors == []
        assert fake_st.rerun_called is False
        assert clear_calls == []

    def test_rejected_error_shows_reason_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_st = _FakeChatTabStreamlit(session_state={}, warnings=[], errors=[])
        service = _ClearHistoryServiceStub(
            clear_side_effects=[ConversationClearRejectedError("sid", reason="存在 active run")]
        )
        monkeypatch.setattr(chat_tab_module, "st", fake_st)
        monkeypatch.setattr(chat_tab_module, "clear_chat_stream_runtime", lambda *, ticker, stream_state_key: None)

        perform_clear_session_history(
            chat_service=cast(ChatServiceProtocol, service),
            session_id="sid",
            ticker="AAPL",
            stream_state_key="stream",
            message_state_key="messages",
            input_key="input",
            clear_input_key="clear_input",
        )

        assert service.clear_call_count == 1
        assert fake_st.warnings == ["当前无法清空会话：存在 active run。请等回复完成后再试。"]
        assert fake_st.errors == []
        assert fake_st.rerun_called is False

    def test_stale_error_retries_silently_then_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_st = _FakeChatTabStreamlit(
            session_state={"messages": ["m"], "input": "x", "clear_input": True},
            warnings=[],
            errors=[],
        )
        service = _ClearHistoryServiceStub(
            clear_side_effects=[
                ConversationClearStaleError("sid", expected_revision="r1", actual_revision="r2"),
                None,
            ]
        )
        clear_calls: list[str] = []
        sleep_calls: list[float] = []
        monkeypatch.setattr(chat_tab_module, "st", fake_st)
        monkeypatch.setattr(chat_tab_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))
        monkeypatch.setattr(
            chat_tab_module,
            "clear_chat_stream_runtime",
            lambda *, ticker, stream_state_key: clear_calls.append(f"{ticker}:{stream_state_key}"),
        )

        perform_clear_session_history(
            chat_service=cast(ChatServiceProtocol, service),
            session_id="sid",
            ticker="AAPL",
            stream_state_key="stream",
            message_state_key="messages",
            input_key="input",
            clear_input_key="clear_input",
        )

        assert service.clear_call_count == 2
        assert sleep_calls == [0.1]
        assert fake_st.warnings == []
        assert fake_st.errors == []
        assert clear_calls == ["AAPL:stream"]
        assert fake_st.session_state["messages"] == []
        assert fake_st.session_state["input"] == ""
        assert fake_st.session_state["clear_input"] is False
        assert fake_st.rerun_called is True

    def test_partially_applied_error_shows_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_st = _FakeChatTabStreamlit(session_state={}, warnings=[], errors=[])
        service = _ClearHistoryServiceStub(
            clear_side_effects=[
                ConversationClearPartiallyAppliedError(
                    "sid",
                    residual_sources=("pending_turn_store", "reply_outbox_store"),
                )
            ]
        )
        monkeypatch.setattr(chat_tab_module, "st", fake_st)
        monkeypatch.setattr(chat_tab_module, "clear_chat_stream_runtime", lambda *, ticker, stream_state_key: None)

        perform_clear_session_history(
            chat_service=cast(ChatServiceProtocol, service),
            session_id="sid",
            ticker="AAPL",
            stream_state_key="stream",
            message_state_key="messages",
            input_key="input",
            clear_input_key="clear_input",
        )

        assert service.clear_call_count == 1
        assert fake_st.warnings == []
        assert fake_st.errors == ["会话历史已清空，但部分残留待修复。请联系管理员。"]
        assert fake_st.rerun_called is False

    def test_stale_retry_failure_shows_retry_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_st = _FakeChatTabStreamlit(session_state={}, warnings=[], errors=[])
        service = _ClearHistoryServiceStub(
            clear_side_effects=[
                ConversationClearStaleError("sid", expected_revision="r1", actual_revision="r2"),
                RuntimeError("retry failed"),
            ]
        )
        sleep_calls: list[float] = []
        monkeypatch.setattr(chat_tab_module, "st", fake_st)
        monkeypatch.setattr(chat_tab_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))
        monkeypatch.setattr(chat_tab_module, "clear_chat_stream_runtime", lambda *, ticker, stream_state_key: None)

        perform_clear_session_history(
            chat_service=cast(ChatServiceProtocol, service),
            session_id="sid",
            ticker="AAPL",
            stream_state_key="stream",
            message_state_key="messages",
            input_key="input",
            clear_input_key="clear_input",
        )

        assert service.clear_call_count == 2
        assert sleep_calls == [0.1]
        assert fake_st.warnings == ["清空会话失败，请稍后重试。"]
        assert fake_st.errors == []
        assert fake_st.rerun_called is False


@pytest.mark.unit
class TestLoadHistoryForTicker:
    """load_history_for_ticker 历史加载测试。"""

    def test_load_history_maps_turns_to_chat_messages(self) -> None:
        service = _HistoryServiceStub(
            turns=[
                ConversationSessionTurnExcerpt(
                    user_text="问题1",
                    assistant_text="回答1",
                    reasoning_text="思考1",
                    created_at="2026-01-01T00:00:00Z",
                ),
                ConversationSessionTurnExcerpt(
                    user_text="问题2",
                    assistant_text="回答2",
                    reasoning_text="",
                    created_at="2026-01-01T00:01:00Z",
                ),
            ]
        )

        messages = load_history_for_ticker(
            chat_service=cast(ChatServiceProtocol, service),
            ticker="AAPL",
        )

        assert service.called_session_id == build_chat_session_id("AAPL")
        assert service.called_limit == 50
        assert messages == [
            ChatMessage(role="user", content="问题1"),
            ChatMessage(role="assistant", content="回答1", reasoning_content="思考1"),
            ChatMessage(role="user", content="问题2"),
            ChatMessage(role="assistant", content="回答2", reasoning_content=""),
        ]

    def test_load_history_returns_empty_when_no_turns(self) -> None:
        service = _HistoryServiceStub(turns=[])

        messages = load_history_for_ticker(
            chat_service=cast(ChatServiceProtocol, service),
            ticker="MSFT",
        )

        assert service.called_session_id == build_chat_session_id("MSFT")
        assert service.called_limit == 50
        assert messages == []

    def test_load_history_returns_empty_on_exception(self) -> None:
        service = _HistoryServiceStub(error=RuntimeError("backend failed"))

        messages = load_history_for_ticker(
            chat_service=cast(ChatServiceProtocol, service),
            ticker="TSLA",
        )

        assert service.called_session_id == build_chat_session_id("TSLA")
        assert service.called_limit == 50
        assert messages == []
