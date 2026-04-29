"""interactive pending turn 恢复测试。"""

from __future__ import annotations

from typing import AsyncIterator

import pytest

from dayu.contracts.events import AppEvent, AppEventType
from dayu.cli.interactive_ui import _RenderState, _render_stream_event, _resume_interactive_pending_turn_if_needed
from dayu.services.contracts import ChatPendingTurnView, ChatResumeRequest


class _FakeChatService:
    """测试用可恢复聊天服务。"""

    def __init__(self, *, expected_scene_name: str = "interactive") -> None:
        self.resume_requests: list[ChatResumeRequest] = []
        self._expected_scene_name = expected_scene_name
        self.list_scene_names: list[str | None] = []

    def list_resumable_pending_turns(self, *, session_id: str | None = None, scene_name: str | None = None) -> list[ChatPendingTurnView]:
        assert session_id == "interactive-session"
        assert scene_name == self._expected_scene_name
        self.list_scene_names.append(scene_name)
        return [
            ChatPendingTurnView(
                pending_turn_id="pending-1",
                session_id="interactive-session",
                scene_name=self._expected_scene_name,
                user_text="未完成问题",
                source_run_id="run-old",
                resumable=True,
                state="sent_to_llm",
                metadata={"delivery_channel": "interactive"},
            )
        ]

    async def resume_pending_turn(self, request: ChatResumeRequest):
        self.resume_requests.append(request)

        async def _stream() -> AsyncIterator[AppEvent]:
            yield AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "恢复结果", "degraded": False}, meta={})

        class _Submission:
            session_id = "interactive-session"
            event_stream = _stream()

        return _Submission()


class _RejectingChatService(_FakeChatService):
    """测试用拒绝恢复的聊天服务。"""

    async def resume_pending_turn(self, request: ChatResumeRequest):
        """模拟 Host gate 拒绝恢复。"""

        self.resume_requests.append(request)
        raise ValueError("pending turn 当前不可恢复")


class _AutoCleaningRejectingChatService(_FakeChatService):
    """模拟恢复失败后 Host 已清理 pending turn 的聊天服务。"""

    def __init__(self) -> None:
        super().__init__()
        self._cleared = False

    def list_resumable_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
    ) -> list[ChatPendingTurnView]:
        if self._cleared:
            assert session_id == "interactive-session"
            assert scene_name == self._expected_scene_name
            self.list_scene_names.append(scene_name)
            return []
        return super().list_resumable_pending_turns(session_id=session_id, scene_name=scene_name)

    async def resume_pending_turn(self, request: ChatResumeRequest):
        self.resume_requests.append(request)
        self._cleared = True
        raise ValueError("pending turn resume_source_json 不是合法 JSON object")


@pytest.mark.unit
def test_interactive_startup_resumes_pending_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """进入 interactive REPL 前应先恢复当前 session 的 pending turn。"""

    service = _FakeChatService()
    rendered: list[str] = []
    monkeypatch.setattr(
        "dayu.cli.interactive_ui._render_stream_event",
        lambda event, state: rendered.append(str(event.payload)),
    )

    _resume_interactive_pending_turn_if_needed(
        service,  # type: ignore[arg-type]
        session_id="interactive-session",
        show_thinking=False,
    )

    assert [request.pending_turn_id for request in service.resume_requests] == ["pending-1"]
    assert [request.session_id for request in service.resume_requests] == ["interactive-session"]
    assert rendered == ["{'content': '恢复结果', 'degraded': False}"]


@pytest.mark.unit
def test_interactive_startup_continues_when_resume_fails_but_pending_turn_still_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host gate 拒绝恢复但 pending turn 仍可恢复时，interactive 应给出错误提示并继续进入 REPL。"""

    service = _RejectingChatService()
    rendered: list[str] = []
    monkeypatch.setattr(
        "dayu.cli.interactive_ui._render_warning_or_error",
        lambda current_state, message: rendered.append(message),
    )

    # 不应抛异常，REPL 启动流程能继续
    _resume_interactive_pending_turn_if_needed(
        service,  # type: ignore[arg-type]
        session_id="interactive-session",
        show_thinking=False,
    )

    assert [request.pending_turn_id for request in service.resume_requests] == ["pending-1"]
    assert [request.session_id for request in service.resume_requests] == ["interactive-session"]
    assert rendered == [
        "[error] 上一轮 pending turn 恢复失败，会话继续可用，可在下一轮输入后重试"
    ]


@pytest.mark.unit
def test_interactive_startup_allows_session_when_failed_pending_turn_has_been_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """恢复失败后若 pending turn 已被 Host 清理，interactive 不应中断整个会话。"""

    service = _AutoCleaningRejectingChatService()
    rendered: list[str] = []
    monkeypatch.setattr(
        "dayu.cli.interactive_ui._render_warning_or_error",
        lambda current_state, message: rendered.append(message),
    )

    _resume_interactive_pending_turn_if_needed(
        service,  # type: ignore[arg-type]
        session_id="interactive-session",
        show_thinking=True,
    )

    assert [request.pending_turn_id for request in service.resume_requests] == ["pending-1"]
    assert rendered == ["[warning] 上一轮 pending turn 恢复失败，但记录已被清理；当前会话继续可用"]


@pytest.mark.unit
def test_interactive_startup_propagates_custom_scene_name_when_resume_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非默认 scene 的 interactive resume 失败但仍 resumable 时，二次探测必须沿用调用方传入的 scene_name。"""

    service = _RejectingChatService(expected_scene_name="custom-scene")
    rendered: list[str] = []
    monkeypatch.setattr(
        "dayu.cli.interactive_ui._render_warning_or_error",
        lambda current_state, message: rendered.append(message),
    )

    _resume_interactive_pending_turn_if_needed(
        service,  # type: ignore[arg-type]
        session_id="interactive-session",
        scene_name="custom-scene",
        show_thinking=False,
    )

    # 关键：list_resumable_pending_turns 必须以 custom-scene 调用两次
    # （首次启动探测 + 失败后二次确认），否则会误判 pending turn 已清理
    assert service.list_scene_names == ["custom-scene", "custom-scene"]
    assert [request.pending_turn_id for request in service.resume_requests] == ["pending-1"]
    assert rendered == [
        "[error] 上一轮 pending turn 恢复失败，会话继续可用，可在下一轮输入后重试"
    ]


@pytest.mark.unit
def test_render_stream_event_marks_filtered_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """interactive UI 收到 filtered final_answer 时应输出 filtered 提示。"""

    state = _RenderState(show_thinking=False)
    rendered: list[str] = []
    monkeypatch.setattr(
        "dayu.cli.interactive_ui._render_warning_or_error",
        lambda current_state, message: rendered.append(message),
    )
    monkeypatch.setattr(
        "dayu.cli.interactive_ui._render_content_delta",
        lambda current_state, text: rendered.append(text),
    )

    _render_stream_event(
        AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload={"content": "部分答案", "degraded": True, "filtered": True},
            meta={},
        ),
        state,
    )

    assert state.final_content == "部分答案"
    assert state.filtered is True
    assert rendered == ["部分答案", "[filtered] 本轮输出触发内容过滤，结果可能不完整"]


@pytest.mark.unit
def test_render_stream_event_renders_cancelled_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """interactive UI 收到 CANCELLED 时应输出明确取消提示。"""

    state = _RenderState(show_thinking=False)
    rendered: list[str] = []
    monkeypatch.setattr(
        "dayu.cli.interactive_ui._render_warning_or_error",
        lambda current_state, message: rendered.append(message),
    )

    _render_stream_event(
        AppEvent(
            type=AppEventType.CANCELLED,
            payload={"cancel_reason": "timeout"},
            meta={},
        ),
        state,
    )

    assert rendered == ["[cancelled] 执行已取消: timeout"]
