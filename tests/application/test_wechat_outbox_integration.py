"""WeChat reply outbox 集成测试。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

from dayu.contracts.events import AppEvent, AppEventType
from dayu.contracts.reply_outbox import ReplyOutboxState
from dayu.host.host import Host
from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore
from dayu.services.contracts import ChatPendingTurnView, ChatResumeRequest, ChatTurnRequest, ChatTurnSubmission
from dayu.services.reply_delivery_service import ReplyDeliveryService
from dayu.wechat.daemon import WeChatDaemon
from dayu.wechat.ilink_client import IlinkApiError, QRCodeLoginStatus, QRCodeLoginTicket
from dayu.wechat.state_store import FileWeChatStateStore, WeChatDaemonState, build_wechat_session_id
from tests.application.conftest import StubHostExecutor, StubRunRegistry, StubSessionRegistry


@dataclass(frozen=True)
class _ScriptedTurn:
    """测试用脚本化轮次。"""

    events: tuple[AppEvent, ...]


class _FakeChatService:
    """测试用 ChatService。"""

    def __init__(self, scripted_turns: list[_ScriptedTurn]) -> None:
        self._scripted_turns = scripted_turns
        self.requests: list[ChatTurnRequest] = []
        self.submit_turn_requests: list[ChatTurnRequest] = []

    async def _build_scripted_event_stream(self, request: ChatTurnRequest) -> AsyncIterator[AppEvent]:
        """返回脚本化事件流。"""

        self.requests.append(request)
        turn = self._scripted_turns.pop(0)
        for event in turn.events:
            yield event

    async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
        """按 ChatServiceProtocol 返回提交句柄。"""

        self.submit_turn_requests.append(request)
        session_id = request.session_id or build_wechat_session_id("user@im.wechat")
        return ChatTurnSubmission(
            session_id=session_id,
            event_stream=self._build_scripted_event_stream(request),
        )

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        """当前测试不覆盖 pending turn 恢复路径。"""

        del request
        raise AssertionError("当前测试不应调用 resume_pending_turn")

    def list_resumable_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
    ) -> list[ChatPendingTurnView]:
        """当前测试默认没有可恢复 pending turn。"""

        del session_id, scene_name
        return []

    def cleanup_stale_pending_turns(
        self,
        *,
        session_id: str | None = None,
    ) -> list[str]:
        """测试桩不需要清理 pending turn。"""

        del session_id
        return []


class _FakeIlinkClient:
    """测试用 iLink client。"""

    def __init__(self, *, updates_payloads: list[dict[str, Any]], fail_send: Exception | None = None) -> None:
        self.updates_payloads = updates_payloads
        self.fail_send = fail_send
        self.login_ticket = QRCodeLoginTicket(qrcode="qr-1", url=None)
        self.login_status = QRCodeLoginStatus(status="confirmed", bot_token="token-1", base_url="https://ilink.example")
        self.sent_messages: list[dict[str, Any]] = []

    def update_auth(self, *, base_url: str | None, bot_token: str | None) -> None:
        """记录登录态更新。"""

        del base_url, bot_token

    async def aclose(self) -> None:
        """关闭客户端。"""

    async def get_bot_qrcode(self) -> QRCodeLoginTicket:
        """返回二维码。"""

        return self.login_ticket

    async def get_qrcode_status(self, qrcode: str) -> QRCodeLoginStatus:
        """返回固定登录状态。"""

        assert qrcode == self.login_ticket.qrcode
        return self.login_status

    async def get_updates(self, *, get_updates_buf: str) -> dict[str, Any]:
        """返回脚本化轮询结果。"""

        del get_updates_buf
        return self.updates_payloads.pop(0)

    async def send_text_message(
        self,
        *,
        to_user_id: str,
        context_token: str,
        text: str,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        """记录或故意失败发送调用。"""

        if self.fail_send is not None:
            raise self.fail_send
        self.sent_messages.append(
            {
                "to_user_id": to_user_id,
                "context_token": context_token,
                "text": text,
                "group_id": group_id,
            }
        )
        return {"ret": 0}

    async def get_typing_ticket(self, *, ilink_user_id: str, context_token: str | None = None) -> str | None:
        """返回空 typing ticket。"""

        del ilink_user_id, context_token
        return None

    async def send_typing(self, *, ilink_user_id: str, typing_ticket: str, status: int = 1) -> dict[str, Any]:
        """typing 在该测试中不参与。"""

        del ilink_user_id, typing_ticket, status
        return {"ret": 0}


def _build_text_message(*, text: str, context_token: str, from_user_id: str = "user@im.wechat") -> dict[str, Any]:
    """构建测试用入站文本消息。"""

    return {
        "from_user_id": from_user_id,
        "message_type": 1,
        "context_token": context_token,
        "item_list": [{"type": 1, "text_item": {"text": text}}],
    }


def _build_reply_delivery_service() -> ReplyDeliveryService:
    """构造测试用 ReplyDeliveryService。"""

    host = Host(
        executor=StubHostExecutor(),
        session_registry=StubSessionRegistry(),
        run_registry=StubRunRegistry(),
        reply_outbox_store=InMemoryReplyOutboxStore(),
    )
    return ReplyDeliveryService(host=host)


@pytest.mark.unit
def test_wechat_process_once_uses_reply_outbox_and_marks_delivered(tmp_path: Path) -> None:
    """WeChat daemon 接入 reply outbox 后应走 submit/claim/delivered 闭环。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    reply_delivery_service = _build_reply_delivery_service()
    chat_service = _FakeChatService(
        scripted_turns=[
            _ScriptedTurn(
                events=(
                    AppEvent(
                        type=AppEventType.FINAL_ANSWER,
                        payload={"content": "答复", "degraded": False},
                        meta={"run_id": "run_wechat_1"},
                    ),
                )
            )
        ]
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="问题", context_token="ctx-1")], "get_updates_buf": "cursor-1"}
        ]
    )
    daemon = WeChatDaemon(
        chat_service=chat_service,
        reply_delivery_service=reply_delivery_service,
        state_store=store,
        client=client,
    )

    asyncio.run(daemon.process_once())

    deliveries = reply_delivery_service.list_deliveries(session_id=build_wechat_session_id("user@im.wechat"))
    assert len(deliveries) == 1
    assert deliveries[0].state == ReplyOutboxState.DELIVERED
    assert deliveries[0].delivery_key == "wechat:run_wechat_1"
    assert len(chat_service.submit_turn_requests) == 1
    assert client.sent_messages[0]["text"] == "答复"


@pytest.mark.unit
def test_wechat_process_once_marks_retryable_failure_when_delivery_errors(tmp_path: Path) -> None:
    """WeChat daemon 发送失败时应把 outbox 标记为可重试失败。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    reply_delivery_service = _build_reply_delivery_service()
    chat_service = _FakeChatService(
        scripted_turns=[
            _ScriptedTurn(
                events=(
                    AppEvent(
                        type=AppEventType.FINAL_ANSWER,
                        payload={"content": "答复", "degraded": False},
                        meta={"run_id": "run_wechat_2"},
                    ),
                )
            )
        ]
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="问题", context_token="ctx-2")], "get_updates_buf": "cursor-2"}
        ],
        fail_send=IlinkApiError("temporary failure", status_code=500),
    )
    daemon = WeChatDaemon(
        chat_service=chat_service,
        reply_delivery_service=reply_delivery_service,
        state_store=store,
        client=client,
    )

    asyncio.run(daemon.process_once())

    deliveries = reply_delivery_service.list_deliveries(session_id=build_wechat_session_id("user@im.wechat"))
    assert len(deliveries) == 1
    assert deliveries[0].state == ReplyOutboxState.FAILED_RETRYABLE
    assert deliveries[0].last_error_message == "temporary failure"
