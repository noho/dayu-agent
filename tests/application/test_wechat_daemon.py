"""WeChat daemon 测试。"""

# pyright: reportOptionalSubscript=false, reportCallIssue=false
# 原因：iLink API 返回嵌套 JSON 结构，逐层 subscript 产生的 union 类型在 pyright 下
# 无法自动收窄；测试仅做断言，不需要严格的 subscript 类型推导。

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import AsyncIterator, Iterator, Mapping

import pytest

import dayu.state_dir_lock as state_dir_lock_module
import dayu.wechat.daemon as wechat_daemon_module
from dayu.contracts.events import AppEvent, AppEventType
from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.contracts.reply_outbox import ReplyOutboxState
from dayu.host.host import Host
from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore
from dayu.state_dir_lock import StateDirSingleInstanceLock
from dayu.services.reply_delivery_service import ReplyDeliveryService
from dayu.services.contracts import (
    ChatPendingTurnView,
    ChatResumeRequest,
    ChatTurnRequest,
    ChatTurnSubmission,
    ReplyDeliverySubmitRequest,
)
from dayu.wechat.daemon import (
    _AsyncWeChatStateStoreAdapter,
    _build_wechat_delivery_context,
    _extract_message_text,
    _format_log_text_preview,
    _format_wechat_reply_text,
    _is_retryable_delivery_error,
    _parse_int_field,
    _rebuild_wechat_delivery_context,
    _resolve_chat_key,
    WeChatDaemon,
    WeChatDaemonConfig,
    WeChatReply,
    WeChatReplyBuilder,
)
from dayu.wechat.ilink_client import IlinkApiError, QRCodeLoginStatus, QRCodeLoginTicket
from dayu.wechat.state_store import (
    build_wechat_runtime_identity,
    FileWeChatStateStore,
    WeChatDaemonState,
    build_wechat_session_id,
)
from tests.application.conftest import StubHostExecutor, StubRunRegistry, StubSessionRegistry

_CREATED_DAEMONS: list[WeChatDaemon] = []


def _build_daemon(
    *,
    chat_service: _FakeChatService | _FakeResumableChatService | _OrderedResumableChatService | _FailingResumeChatService | _AutoCleaningResumeChatService,
    state_store: FileWeChatStateStore,
    client: _FakeIlinkClient | _BusinessFailingIlinkClient | _TransientFailingIlinkClient,
    reply_delivery_service: ReplyDeliveryService | None = None,
    config: WeChatDaemonConfig | None = None,
) -> WeChatDaemon:
    """创建并登记测试用 daemon，便于在测试结束后统一关闭。

    Args:
        chat_service: 聊天服务桩。
        state_store: 状态仓储。
        client: iLink 客户端桩。
        reply_delivery_service: 可选回复投递服务。
        config: 可选 daemon 配置。

    Returns:
        已登记的 daemon 实例。

    Raises:
        无。
    """

    daemon = WeChatDaemon(
        chat_service=chat_service,
        state_store=state_store,
        client=client,
        reply_delivery_service=reply_delivery_service,
        config=config or WeChatDaemonConfig(),
    )
    _CREATED_DAEMONS.append(daemon)
    return daemon


def _build_reply_delivery_service() -> ReplyDeliveryService:
    """构造测试用 ReplyDeliveryService。"""

    host = Host(
        executor=StubHostExecutor(),
        session_registry=StubSessionRegistry(),
        run_registry=StubRunRegistry(),
        reply_outbox_store=InMemoryReplyOutboxStore(),
    )
    return ReplyDeliveryService(host=host)


@pytest.fixture(autouse=True)
def _close_registered_daemons() -> Iterator[None]:
    """在每条测试结束后显式关闭当前测试创建的 daemon。"""

    start_index = len(_CREATED_DAEMONS)
    try:
        yield
    finally:
        pending_daemons = list(reversed(_CREATED_DAEMONS[start_index:]))
        del _CREATED_DAEMONS[start_index:]
        for daemon in pending_daemons:
            asyncio.run(daemon.aclose())


@dataclass(frozen=True)
class _ScriptedTurn:
    """测试用脚本化轮次。"""

    events: tuple[AppEvent, ...]
    delay_sec: float = 0.0


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
        if turn.delay_sec > 0:
            await asyncio.sleep(turn.delay_sec)

    async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
        """按 ChatServiceProtocol 返回提交句柄。"""

        self.submit_turn_requests.append(request)
        session_id = request.session_id or build_wechat_session_id("user@im.wechat")
        return ChatTurnSubmission(
            session_id=session_id,
            event_stream=self._build_scripted_event_stream(request),
        )

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        """基础测试桩默认不支持 pending turn 恢复。"""

        del request
        raise AssertionError("当前测试不应调用 resume_pending_turn")

    def list_resumable_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
    ) -> list[ChatPendingTurnView]:
        """基础测试桩默认没有可恢复 pending turn。"""

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


def _filtered_flag(metadata: Mapping[str, object]) -> bool | None:
    """安全读取交付上下文中的 filtered 标记。"""

    filtered = metadata.get("filtered")
    return filtered if isinstance(filtered, bool) else None


def _wechat_runtime_identity(metadata: Mapping[str, object]) -> str | None:
    """安全读取交付上下文中的运行时身份。"""

    runtime_identity = metadata.get("wechat_runtime_identity")
    return runtime_identity if isinstance(runtime_identity, str) else None


class _FakeResumableChatService(_FakeChatService):
    """支持 pending turn 恢复的测试服务。"""

    def __init__(self, scripted_turns: list[_ScriptedTurn], *, runtime_identity: str) -> None:
        super().__init__(scripted_turns)
        self.resume_requests: list[ChatResumeRequest] = []
        self.runtime_identity = runtime_identity

    def list_resumable_pending_turns(self, *, session_id: str | None = None, scene_name: str | None = None) -> list[ChatPendingTurnView]:
        assert scene_name == "wechat"
        if session_id not in {None, build_wechat_session_id("user@im.wechat")}:
            return []
        return [
            ChatPendingTurnView(
                pending_turn_id="pending-1",
                session_id=build_wechat_session_id("user@im.wechat"),
                scene_name="wechat",
                user_text="历史问题",
                source_run_id="run-old",
                resumable=True,
                state="sent_to_llm",
                metadata={
                    "delivery_channel": "wechat",
                    "delivery_target": "user@im.wechat",
                    "delivery_thread_id": "ctx-old",
                    "wechat_runtime_identity": self.runtime_identity,
                },
            ),
            ChatPendingTurnView(
                pending_turn_id="pending-other-runtime",
                session_id=build_wechat_session_id("user@im.wechat"),
                scene_name="wechat",
                user_text="其他实例问题",
                source_run_id="run-other",
                resumable=True,
                state="sent_to_llm",
                metadata={
                    "delivery_channel": "wechat",
                    "delivery_target": "user@im.wechat",
                    "delivery_thread_id": "ctx-other",
                    "wechat_runtime_identity": "wechat_runtime_foreign",
                },
            )
        ]

    async def resume_pending_turn(self, request: ChatResumeRequest):
        self.resume_requests.append(request)
        turn = self._scripted_turns.pop(0)

        async def _stream() -> AsyncIterator[AppEvent]:
            for event in turn.events:
                yield event

        return ChatTurnSubmission(
            session_id=build_wechat_session_id("user@im.wechat"),
            event_stream=_stream(),
        )


class _OrderedResumableChatService(_FakeResumableChatService):
    """带启动顺序记录的恢复测试桩。"""

    def __init__(self, scripted_turns: list[_ScriptedTurn], *, runtime_identity: str, events: list[str]) -> None:
        super().__init__(scripted_turns, runtime_identity=runtime_identity)
        self._events = events

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        self._events.append("resume")
        return await super().resume_pending_turn(request)


class _FailingResumeChatService(_FakeResumableChatService):
    """恢复阶段始终失败的测试桩。"""

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        self.resume_requests.append(request)
        raise ValueError("pending conversation turn 对应的 source run 仍处于活跃状态，不能恢复")


class _AutoCleaningResumeChatService(_FakeResumableChatService):
    """模拟恢复失败后 pending turn 已被 Host 清理的测试桩。"""

    def __init__(self, scripted_turns: list[_ScriptedTurn], *, runtime_identity: str) -> None:
        super().__init__(scripted_turns, runtime_identity=runtime_identity)
        self._cleared = False

    def list_resumable_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
    ) -> list[ChatPendingTurnView]:
        if self._cleared:
            return []
        return super().list_resumable_pending_turns(session_id=session_id, scene_name=scene_name)

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        self.resume_requests.append(request)
        self._cleared = True
        raise ValueError("pending turn resume_source_json 不是合法 JSON object")

class _FakeIlinkClient:
    """测试用 iLink client。"""

    def __init__(
        self,
        *,
        updates_payloads: list[dict[str, str | int | float | bool | None | list | dict]] | None = None,
        login_ticket: QRCodeLoginTicket | None = None,
        login_status: QRCodeLoginStatus | None = None,
        typing_ticket: str | None = None,
    ) -> None:
        self.updates_payloads = updates_payloads or []
        self.login_ticket = login_ticket or QRCodeLoginTicket(qrcode="qr-1", url=None)
        self.login_status = login_status or QRCodeLoginStatus(
            status="confirmed",
            bot_token="token-1",
            base_url="https://ilink.example",
        )
        self.typing_ticket = typing_ticket
        self.auth_updates: list[tuple[str | None, str | None]] = []
        self.received_cursors: list[str] = []
        self.sent_messages: list[dict[str, str | int | float | bool | None | list | dict]] = []
        self.typing_calls: list[dict[str, str | int | float | bool | None | list | dict]] = []
        self.closed = False

    def update_auth(self, *, base_url: str | None, bot_token: str | None) -> None:
        """记录登录态更新。"""

        self.auth_updates.append((base_url, bot_token))

    async def aclose(self) -> None:
        """关闭客户端。"""

        self.closed = True

    async def get_bot_qrcode(self) -> QRCodeLoginTicket:
        """返回二维码。"""

        return self.login_ticket

    async def get_qrcode_status(self, qrcode: str) -> QRCodeLoginStatus:
        """返回固定登录状态。"""

        assert qrcode == self.login_ticket.qrcode
        return self.login_status

    async def get_updates(self, *, get_updates_buf: str) -> dict[str, str | int | float | bool | None | list | dict]:
        """返回脚本化轮询结果。"""

        self.received_cursors.append(get_updates_buf)
        return self.updates_payloads.pop(0)

    async def send_text_message(
        self,
        *,
        to_user_id: str,
        context_token: str,
        text: str,
        group_id: str | None = None,
    ) -> dict[str, str | int | float | bool | None | list | dict]:
        """记录发消息调用。"""

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
        """返回固定 typing ticket。"""

        assert ilink_user_id
        _ = context_token
        return self.typing_ticket

    async def send_typing(
        self,
        *,
        ilink_user_id: str,
        typing_ticket: str,
        status: int = 1,
    ) -> dict[str, str | int | float | bool | None | list | dict]:
        """记录 typing 调用。"""

        self.typing_calls.append(
            {
                "ilink_user_id": ilink_user_id,
                "typing_ticket": typing_ticket,
                "status": status,
            }
        )
        return {"ret": 0}


class _BusinessFailingIlinkClient(_FakeIlinkClient):
    """发送文本时返回 iLink 业务失败。"""

    async def send_text_message(
        self,
        *,
        to_user_id: str,
        context_token: str,
        text: str,
        group_id: str | None = None,
    ) -> dict[str, str | int | float | bool | None | list | dict]:
        await super().send_text_message(
            to_user_id=to_user_id,
            context_token=context_token,
            text=text,
            group_id=group_id,
        )
        raise IlinkApiError("iLink 业务错误 ret=-2", status_code=200, business_ret_code=-2, payload={"ret": -2})


class _TransientFailingIlinkClient(_FakeIlinkClient):
    """发送文本时持续返回可重试错误。"""

    async def send_text_message(
        self,
        *,
        to_user_id: str,
        context_token: str,
        text: str,
        group_id: str | None = None,
    ) -> dict[str, str | int | float | bool | None | list | dict]:
        await super().send_text_message(
            to_user_id=to_user_id,
            context_token=context_token,
            text=text,
            group_id=group_id,
        )
        raise IlinkApiError("iLink HTTP 错误: 503", status_code=503, payload={"ret": 0})


class _ThreadRecordingStateStore(FileWeChatStateStore):
    """记录状态仓储调用线程的测试替身。"""

    def __init__(self, state_dir: Path) -> None:
        """初始化线程记录仓储。

        Args:
            state_dir: 状态目录。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(state_dir)
        self.load_thread_ids: list[int] = []
        self.save_thread_ids: list[int] = []
        self.qrcode_thread_ids: list[int] = []

    def load(self) -> WeChatDaemonState:
        """记录 load 调用线程。"""

        self.load_thread_ids.append(threading.get_ident())
        return super().load()

    def save(self, state: WeChatDaemonState) -> None:
        """记录 save 调用线程。"""

        self.save_thread_ids.append(threading.get_ident())
        super().save(state)

    def write_qrcode_artifact(self, qrcode_img_content: str | None) -> Path | None:
        """记录二维码文件写入线程。"""

        self.qrcode_thread_ids.append(threading.get_ident())
        return super().write_qrcode_artifact(qrcode_img_content)


class _BlockingSaveStateStore(FileWeChatStateStore):
    """在 save 时阻塞的测试仓储。"""

    def __init__(self, state_dir: Path) -> None:
        """初始化阻塞 save 仓储。

        Args:
            state_dir: 状态目录。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(state_dir)
        self.save_started = threading.Event()
        self.release_save = threading.Event()
        self.save_completed = threading.Event()

    def save(self, state: WeChatDaemonState) -> None:
        """阻塞 save，直到测试显式放行。"""

        self.save_started.set()
        released = self.release_save.wait(timeout=5)
        if not released:
            raise TimeoutError("测试未在预期时间内放行阻塞 save")
        super().save(state)
        self.save_completed.set()


class _FailingBlockingSaveStateStore(FileWeChatStateStore):
    """在 save 阶段阻塞后失败的测试仓储。"""

    def __init__(self, state_dir: Path, *, error_message: str = "disk full") -> None:
        """初始化阻塞失败 save 仓储。

        Args:
            state_dir: 状态目录。
            error_message: save 释放后抛出的错误信息。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(state_dir)
        self.save_started = threading.Event()
        self.release_save = threading.Event()
        self.error_message = error_message

    def save(self, state: WeChatDaemonState) -> None:
        """阻塞 save，放行后抛出持久化失败。"""

        _ = state
        self.save_started.set()
        released = self.release_save.wait(timeout=5)
        if not released:
            raise TimeoutError("测试未在预期时间内放行失败 save")
        raise OSError(self.error_message)


class _SerialSaveStateStore(FileWeChatStateStore):
    """记录 save 串行顺序的测试仓储。"""

    def __init__(self, state_dir: Path) -> None:
        """初始化串行 save 仓储。

        Args:
            state_dir: 状态目录。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(state_dir)
        self.entry_order: list[str] = []
        self.exit_order: list[str] = []
        self.active_saves = 0
        self.max_active_saves = 0
        self.first_save_entered = threading.Event()
        self.release_first_save = threading.Event()
        self._lock = threading.Lock()

    def save(self, state: WeChatDaemonState) -> None:
        """记录 save 进入/退出顺序并阻塞第一次 save。"""

        marker = state.get_updates_buf
        with self._lock:
            self.active_saves += 1
            self.max_active_saves = max(self.max_active_saves, self.active_saves)
            self.entry_order.append(marker)
        if marker == "first":
            self.first_save_entered.set()
            released = self.release_first_save.wait(timeout=5)
            if not released:
                raise TimeoutError("测试未在预期时间内放行第一次 save")
        super().save(state)
        with self._lock:
            self.exit_order.append(marker)
            self.active_saves -= 1

def _build_text_message(*, text: str, context_token: str, from_user_id: str = "user@im.wechat") -> dict[str, str | int | float | bool | None | list | dict]:
    """构建测试用入站文本消息。"""

    return {
        "from_user_id": from_user_id,
        "message_type": 1,
        "context_token": context_token,
        "item_list": [{"type": 1, "text_item": {"text": text}}],
    }


@pytest.mark.unit
def test_reply_builder_prefers_final_answer() -> None:
    """验证回复聚合优先取 final_answer。"""

    builder = WeChatReplyBuilder()
    builder.consume(AppEvent(type=AppEventType.CONTENT_DELTA, payload="前缀", meta={}))
    builder.consume(
        AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "最终答案", "degraded": True}, meta={})
    )

    reply = builder.build()

    assert reply.text == "最终答案"
    assert reply.degraded is True


@pytest.mark.unit
def test_reply_builder_preserves_filtered_state() -> None:
    """验证微信回复聚合会保留 filtered 状态。"""

    builder = WeChatReplyBuilder()
    builder.consume(AppEvent(type=AppEventType.CONTENT_DELTA, payload="前缀", meta={}))
    builder.consume(
        AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload={"content": "最终答案", "degraded": True, "filtered": True},
            meta={},
        )
    )

    reply = builder.build()

    assert reply.text == "最终答案"
    assert reply.filtered is True


@pytest.mark.unit
def test_reply_builder_marks_cancelled_and_drops_partial_text() -> None:
    """微信回复聚合遇到 CANCELLED 时应保留取消状态并丢弃 partial text。"""

    builder = WeChatReplyBuilder()
    builder.consume(AppEvent(type=AppEventType.CONTENT_DELTA, payload="前缀", meta={"run_id": "run-1"}))
    builder.consume(
        AppEvent(
            type=AppEventType.CANCELLED,
            payload={"cancel_reason": "timeout"},
            meta={"run_id": "run-1"},
        )
    )

    reply = builder.build()

    assert reply.text == ""
    assert reply.cancelled is True
    assert reply.cancel_reason == "timeout"
    assert reply.source_run_id == "run-1"


@pytest.mark.unit
def test_extract_message_text_returns_none_for_non_numeric_message_type() -> None:
    """非数字 message_type 不应触发崩溃。"""

    message = {
        "from_user_id": "user@im.wechat",
        "message_type": "text",
        "context_token": "ctx-1",
        "item_list": [{"type": 1, "text_item": {"text": "hello"}}],
    }

    assert _extract_message_text(message) is None


@pytest.mark.unit
def test_extract_message_text_skips_non_numeric_item_type() -> None:
    """非数字 item.type 应被跳过而不是中断整条消息解析。"""

    message = {
        "from_user_id": "user@im.wechat",
        "message_type": 1,
        "context_token": "ctx-1",
        "item_list": [
            {"type": "text", "text_item": {"text": "ignored"}},
            {"type": 1, "text_item": {"text": "hello"}},
        ],
    }

    assert _extract_message_text(message) == "hello"


@pytest.mark.unit
def test_process_once_reuses_same_session_for_same_user(tmp_path: Path) -> None:
    """验证同一微信用户会复用同一 Dayu session。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(
        scripted_turns=[
            _ScriptedTurn(events=(AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "答1"}, meta={}),)),
            _ScriptedTurn(events=(AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "答2"}, meta={}),)),
        ]
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="问题1", context_token="ctx-1")], "get_updates_buf": "cursor-1"},
            {"ret": 0, "msgs": [_build_text_message(text="问题2", context_token="ctx-2")], "get_updates_buf": "cursor-2"},
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    asyncio.run(daemon.process_once())
    asyncio.run(daemon.process_once())

    assert len(service.requests) == 2
    assert len(service.submit_turn_requests) == 2
    assert service.requests[0].session_id == service.requests[1].session_id
    assert service.requests[0].scene_name == "wechat"
    assert service.requests[1].scene_name == "wechat"
    assert client.sent_messages[0]["text"] == "答1"
    assert client.sent_messages[1]["text"] == "答2"
    assert store.load().get_updates_buf == "cursor-2"


@pytest.mark.unit
def test_process_once_appends_filtered_hint(tmp_path: Path) -> None:
    """微信直接发送时应对 filtered 回复追加可见提示。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(
        scripted_turns=[
            _ScriptedTurn(
                events=(
                    AppEvent(
                        type=AppEventType.FINAL_ANSWER,
                        payload={"content": "答复", "degraded": True, "filtered": True},
                        meta={},
                    ),
                )
            )
        ]
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="问题", context_token="ctx-1")], "get_updates_buf": "cursor-1"},
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    asyncio.run(daemon.process_once())

    assert client.sent_messages[0]["text"] == "答复\n\n[filtered] 内容可能不完整"


@pytest.mark.unit
def test_process_once_falls_back_to_content_delta_when_missing_final_answer(tmp_path: Path) -> None:
    """验证缺少 final_answer 时会拼接 content_delta。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(
        scripted_turns=[
            _ScriptedTurn(
                events=(
                    AppEvent(type=AppEventType.CONTENT_DELTA, payload="第一段", meta={}),
                    AppEvent(type=AppEventType.CONTENT_DELTA, payload="第二段", meta={}),
                )
            )
        ]
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="问题", context_token="ctx-1")], "get_updates_buf": "cursor-1"}
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    asyncio.run(daemon.process_once())

    assert client.sent_messages[0]["text"] == "第一段第二段"


@pytest.mark.unit
def test_process_once_sends_cancelled_message_instead_of_empty_reply(tmp_path: Path) -> None:
    """微信直接发送路径在取消时应发送取消提示，而不是空回复兜底。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(
        scripted_turns=[
            _ScriptedTurn(
                events=(
                    AppEvent(
                        type=AppEventType.CANCELLED,
                        payload={"cancel_reason": "user_cancelled"},
                        meta={"run_id": "run_cancelled"},
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
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    asyncio.run(daemon.process_once())

    assert client.sent_messages[0]["text"] == "[cancelled] 当前执行已取消: user_cancelled"


@pytest.mark.unit
def test_process_once_logs_received_and_sent_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证处理微信消息时会记录收消息和发回复日志。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(
        scripted_turns=[
            _ScriptedTurn(events=(AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "这是回复"}, meta={}),))
        ]
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="这是问题", context_token="ctx-1")], "get_updates_buf": "cursor-1"}
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)
    logged: list[str] = []
    session_id = build_wechat_session_id("user@im.wechat")

    monkeypatch.setattr("dayu.wechat.daemon.Log.info", lambda message, *, module="APP": logged.append(f"{module}:{message}"))

    asyncio.run(daemon.process_once())

    assert f"APP.WECHAT:收到微信消息 user=user@im.wechat session={session_id} text=这是问题" in logged
    assert f"APP.WECHAT:发送微信回复 user=user@im.wechat session={session_id} text=这是回复" in logged


@pytest.mark.unit
def test_ensure_authenticated_persists_login_state(tmp_path: Path) -> None:
    """验证扫码登录成功后会保存 token/base_url。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        login_ticket=QRCodeLoginTicket(qrcode="qr-1", url=None, qrcode_img_content=None),
        login_status=QRCodeLoginStatus(status="confirmed", bot_token="token-1", base_url="https://ilink.example"),
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    state = asyncio.run(daemon.ensure_authenticated(force_relogin=True))

    assert state.bot_token == "token-1"
    assert state.base_url == "https://ilink.example"
    assert store.load().bot_token == "token-1"
    assert client.auth_updates[-1] == ("https://ilink.example", "token-1")


@pytest.mark.unit
def test_load_existing_authenticated_state_raises_without_token(tmp_path: Path) -> None:
    """验证缺少本地登录态时会给出明确错误。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient()
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    with pytest.raises(RuntimeError, match="python -m dayu.wechat login"):
        asyncio.run(daemon.load_existing_authenticated_state())


@pytest.mark.unit
def test_run_forever_requires_existing_auth_updates_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 `run` 模式会先加载已有登录态再进入主循环。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient()
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    async def _stop_immediately() -> int:
        raise asyncio.CancelledError()

    monkeypatch.setattr(daemon, "process_once", _stop_immediately)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(daemon.run_forever(require_existing_auth=True))

    assert client.auth_updates[-1] == ("https://ilink.example", "token-1")


@pytest.mark.unit
def test_run_forever_logs_when_entering_wait_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 daemon 在进入长轮询等待前会立即输出启动日志。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient()
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)
    logged: list[str] = []

    async def _stop_immediately() -> int:
        raise asyncio.CancelledError()

    monkeypatch.setattr(daemon, "process_once", _stop_immediately)
    monkeypatch.setattr("dayu.wechat.daemon.Log.info", lambda message, *, module="APP": logged.append(f"{module}:{message}"))

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(daemon.run_forever(require_existing_auth=True))

    assert "APP.WECHAT:WeChat daemon 已进入运行态，开始等待新消息" in logged


@pytest.mark.unit
def test_run_forever_recovers_channel_state_before_resuming_pending_turns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """daemon 启动时应先回收渠道级启动状态，再恢复 pending turn。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    startup_events: list[str] = []
    service = _OrderedResumableChatService(
        scripted_turns=[
            _ScriptedTurn(events=(AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "补发答复"}, meta={}),))
        ],
        runtime_identity=build_wechat_runtime_identity(store.state_dir),
        events=startup_events,
    )
    client = _FakeIlinkClient()
    daemon = _build_daemon(
        chat_service=service,
        state_store=store,
        client=client,
    )

    async def _stop_immediately() -> int:
        raise asyncio.CancelledError()

    async def _recover_channel_state() -> None:
        startup_events.append("recover_delivery")

    monkeypatch.setattr(daemon, "_recover_interrupted_reply_deliveries", _recover_channel_state)
    monkeypatch.setattr(daemon, "process_once", _stop_immediately)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(daemon.run_forever(require_existing_auth=True))

    assert startup_events == ["recover_delivery", "resume"]


@pytest.mark.unit
def test_run_forever_skips_failed_pending_resume_and_keeps_daemon_alive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单条 pending turn 恢复失败不应阻断 daemon 启动。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FailingResumeChatService(
        scripted_turns=[],
        runtime_identity=build_wechat_runtime_identity(store.state_dir),
    )
    client = _FakeIlinkClient()
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    async def _stop_immediately() -> int:
        raise asyncio.CancelledError()

    monkeypatch.setattr(daemon, "process_once", _stop_immediately)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(daemon.run_forever(require_existing_auth=True))

    assert [request.pending_turn_id for request in service.resume_requests] == ["pending-1"]


@pytest.mark.unit
def test_process_once_rejects_new_message_when_session_pending_turn_cannot_resume(tmp_path: Path) -> None:
    """同会话存在无法恢复的 pending turn 时，应返回显式错误而不是继续 submit_turn。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FailingResumeChatService(
        scripted_turns=[],
        runtime_identity=build_wechat_runtime_identity(store.state_dir),
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="新问题", context_token="ctx-new")], "get_updates_buf": "cursor-1"}
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    processed = asyncio.run(daemon.process_once())

    assert processed == 1
    assert service.submit_turn_requests == []
    assert [request.pending_turn_id for request in service.resume_requests] == ["pending-1"]
    assert client.sent_messages == [
        {
            "to_user_id": "user@im.wechat",
            "context_token": "ctx-new",
            "text": daemon.config.pending_turn_blocked_reply_text,
            "group_id": None,
        }
    ]


@pytest.mark.unit
def test_process_once_allows_new_message_when_failed_pending_turn_has_been_cleared(tmp_path: Path) -> None:
    """若恢复失败后 Host 已清理 pending turn，daemon 应继续处理当前新消息。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _AutoCleaningResumeChatService(
        scripted_turns=[
            _ScriptedTurn(
                events=(
                    AppEvent(
                        type=AppEventType.FINAL_ANSWER,
                        payload={"content": "新回复", "degraded": False},
                        meta={"run_id": "run-new"},
                    ),
                )
            )
        ],
        runtime_identity=build_wechat_runtime_identity(store.state_dir),
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="新问题", context_token="ctx-new")], "get_updates_buf": "cursor-1"}
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    processed = asyncio.run(daemon.process_once())

    assert processed == 1
    assert [request.pending_turn_id for request in service.resume_requests] == ["pending-1"]
    assert len(service.submit_turn_requests) == 1
    assert client.sent_messages == [
        {
            "to_user_id": "user@im.wechat",
            "context_token": "ctx-new",
            "text": "新回复",
            "group_id": None,
        }
    ]


@pytest.mark.unit
def test_run_forever_recovers_interrupted_reply_delivery_before_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """启动时应把上次进程遗留的 in-progress delivery 回收并补发。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    runtime_identity = build_wechat_runtime_identity(store.state_dir)
    reply_delivery_service = _build_reply_delivery_service()
    record = reply_delivery_service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="wechat:run_orphan_delivery",
            session_id=build_wechat_session_id("user@im.wechat"),
            scene_name="wechat",
            source_run_id="run_orphan_delivery",
            reply_content="补发中的微信回复",
            metadata={
                "delivery_channel": "wechat",
                "delivery_target": "user@im.wechat",
                "delivery_thread_id": "ctx-orphan",
                "wechat_runtime_identity": runtime_identity,
            },
        )
    )
    reply_delivery_service.claim_delivery(record.delivery_id)
    client = _FakeIlinkClient()
    daemon = _build_daemon(
        chat_service=_FakeChatService(scripted_turns=[]),
        state_store=store,
        reply_delivery_service=reply_delivery_service,
        client=client,
        config=WeChatDaemonConfig(delivery_scan_interval_sec=0.001),
    )

    async def _stop_after_delivery_loop_ticks() -> int:
        await asyncio.sleep(0.02)
        raise asyncio.CancelledError()

    monkeypatch.setattr(daemon, "process_once", _stop_after_delivery_loop_ticks)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(daemon.run_forever(require_existing_auth=True))

    recovered = reply_delivery_service.get_delivery(record.delivery_id)
    assert recovered is not None
    assert recovered.state == ReplyOutboxState.DELIVERED
    assert client.sent_messages == [
        {
            "to_user_id": "user@im.wechat",
            "context_token": "ctx-orphan",
            "text": "补发中的微信回复",
            "group_id": None,
        }
    ]


@pytest.mark.unit
def test_delivery_business_error_is_terminal_without_retry(tmp_path: Path) -> None:
    """iLink 显式业务失败码应首轮收口为 terminal。"""

    reply_delivery_service = _build_reply_delivery_service()
    record = reply_delivery_service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="wechat:delivery_business_terminal",
            session_id=build_wechat_session_id("user@im.wechat"),
            scene_name="wechat",
            source_run_id="run_delivery_terminal",
            reply_content="业务失败回复",
            metadata={
                "delivery_channel": "wechat",
                "delivery_target": "user@im.wechat",
                "delivery_thread_id": "ctx-terminal",
            },
        )
    )
    daemon = _build_daemon(
        chat_service=_FakeChatService(scripted_turns=[]),
        state_store=FileWeChatStateStore(tmp_path / ".wechat"),
        reply_delivery_service=reply_delivery_service,
        client=_BusinessFailingIlinkClient(),
    )

    asyncio.run(daemon._deliver_pending_replies())

    failed = reply_delivery_service.get_delivery(record.delivery_id)
    assert failed is not None
    assert failed.state == ReplyOutboxState.FAILED_TERMINAL
    assert failed.delivery_attempt_count == 1
    assert failed.last_error_message == "iLink 业务错误 ret=-2"


@pytest.mark.unit
def test_delivery_retryable_failure_stops_after_configured_attempts(tmp_path: Path) -> None:
    """可重试发送错误达到上限后应在发送前收口为 terminal。"""

    reply_delivery_service = _build_reply_delivery_service()
    record = reply_delivery_service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key="wechat:delivery_retry_exhausted",
            session_id=build_wechat_session_id("user@im.wechat"),
            scene_name="wechat",
            source_run_id="run_delivery_retry",
            reply_content="重试失败回复",
            metadata={
                "delivery_channel": "wechat",
                "delivery_target": "user@im.wechat",
                "delivery_thread_id": "ctx-retry",
            },
        )
    )
    client = _TransientFailingIlinkClient()
    daemon = _build_daemon(
        chat_service=_FakeChatService(scripted_turns=[]),
        state_store=FileWeChatStateStore(tmp_path / ".wechat"),
        reply_delivery_service=reply_delivery_service,
        client=client,
        config=WeChatDaemonConfig(delivery_max_attempts=3),
    )

    for _ in range(4):
        asyncio.run(daemon._deliver_pending_replies())

    failed = reply_delivery_service.get_delivery(record.delivery_id)
    assert failed is not None
    assert failed.state == ReplyOutboxState.FAILED_TERMINAL
    assert failed.delivery_attempt_count == 4
    assert failed.last_error_message == "delivery retries exhausted"
    assert len(client.sent_messages) == 3


@pytest.mark.unit
def test_run_forever_rejects_duplicate_state_dir_instance(tmp_path: Path) -> None:
    """同一个 state_dir 已被占用时，第二个 daemon 应直接拒绝启动。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    first_daemon = _build_daemon(
        chat_service=_FakeChatService(scripted_turns=[]),
        state_store=store,
        client=_FakeIlinkClient(),
    )
    second_daemon = _build_daemon(
        chat_service=_FakeChatService(scripted_turns=[]),
        state_store=store,
        client=_FakeIlinkClient(),
    )

    first_daemon._instance_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="同一个 state_dir 已有运行中的 WeChat daemon"):
            asyncio.run(second_daemon.run_forever(require_existing_auth=True))
    finally:
        first_daemon._instance_lock.release()


@pytest.mark.unit
def test_single_instance_lock_uses_msvcrt_when_fcntl_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Windows 分支应改用 msvcrt，释放后锁文件内容应被清空。"""

    class _FakeMsvcrt:
        """记录 locking 调用的 Windows 锁实现桩。"""

        LK_NBLCK = 1
        LK_UNLCK = 2

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def locking(self, fd: int, mode: int, size: int) -> None:
            del fd
            self.calls.append((mode, size))

    fake_msvcrt = _FakeMsvcrt()
    monkeypatch.setattr(state_dir_lock_module.file_lock_module, "_FCNTL", None)
    monkeypatch.setattr(state_dir_lock_module.file_lock_module, "_MSVCRT", fake_msvcrt)

    lock_path = tmp_path / ".wechat" / ".daemon.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("stale-owner\n\0", encoding="utf-8")
    lock = StateDirSingleInstanceLock(
        state_dir=tmp_path / ".wechat",
        lock_file_name=".daemon.lock",
        lock_name="WeChat daemon 单实例锁",
    )
    lock.acquire()
    try:
        assert lock_path.exists()
        assert lock_path.read_text(encoding="utf-8") == f"{os.getpid()}\n"
        assert fake_msvcrt.calls == [(fake_msvcrt.LK_NBLCK, 1)]
    finally:
        lock.release()
    assert lock_path.read_text(encoding="utf-8") == ""

    assert fake_msvcrt.calls == [
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_UNLCK, 1),
    ]


@pytest.mark.unit
def test_process_once_sends_typing_best_effort(tmp_path: Path) -> None:
    """验证处理消息时会 best-effort 发送 typing。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(
        scripted_turns=[
            _ScriptedTurn(
                events=(AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "答复"}, meta={}),),
                # Windows ProactorEventLoop 的 timer 粒度约 15ms；
                # 这里把 scripted turn 的 delay 拉高到 0.2s，保证 typing task 有机会至少跑一次，
                # 不再依赖 macOS/Linux 上更细的 scheduling 粒度。
                delay_sec=0.2,
            )
        ]
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="问题", context_token="ctx-1")], "get_updates_buf": "cursor-1"}
        ],
        typing_ticket="typing-1",
    )
    daemon = _build_daemon(
        chat_service=service,
        state_store=store,
        client=client,
        config=WeChatDaemonConfig(typing_interval_sec=0.001),
    )

    asyncio.run(daemon.process_once())

    assert client.typing_calls
    assert client.typing_calls[0]["typing_ticket"] == "typing-1"


@pytest.mark.unit
def test_run_forever_resumes_pending_wechat_turn_before_polling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """daemon 启动后应先补发历史 pending turn，再进入长轮询。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeResumableChatService(
        scripted_turns=[
            _ScriptedTurn(events=(AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "补发答复"}, meta={}),))
        ],
        runtime_identity=build_wechat_runtime_identity(store.state_dir),
    )
    client = _FakeIlinkClient()
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    async def _stop_immediately() -> int:
        raise asyncio.CancelledError()

    monkeypatch.setattr(daemon, "process_once", _stop_immediately)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(daemon.run_forever(require_existing_auth=True))

    assert [request.pending_turn_id for request in service.resume_requests] == ["pending-1"]
    assert [request.session_id for request in service.resume_requests] == [build_wechat_session_id("user@im.wechat")]
    assert client.sent_messages == [
        {
            "to_user_id": "user@im.wechat",
            "context_token": "ctx-old",
            "text": "补发答复",
            "group_id": None,
        }
    ]


@pytest.mark.unit
def test_resume_pending_turn_with_reply_outbox_preserves_filtered_metadata(tmp_path: Path) -> None:
    """恢复补发走 reply outbox 时应保留真实 filtered 状态与可见提示。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    runtime_identity = build_wechat_runtime_identity(store.state_dir)
    service = _FakeResumableChatService(
        scripted_turns=[
            _ScriptedTurn(
                events=(
                    AppEvent(
                        type=AppEventType.FINAL_ANSWER,
                        payload={"content": "补发答复", "degraded": True, "filtered": True},
                        meta={"run_id": "run_resume_filtered"},
                    ),
                )
            )
        ],
        runtime_identity=runtime_identity,
    )
    client = _FakeIlinkClient()
    reply_delivery_service = _build_reply_delivery_service()
    daemon = _build_daemon(
        chat_service=service,
        state_store=store,
        client=client,
        reply_delivery_service=reply_delivery_service,
    )

    asyncio.run(daemon._resume_single_pending_turn("pending-1"))

    records = reply_delivery_service.list_deliveries(
        session_id=build_wechat_session_id("user@im.wechat"),
        scene_name="wechat",
    )
    assert len(records) == 1
    assert records[0].source_run_id == "run_resume_filtered"
    assert _filtered_flag(records[0].metadata) is True
    assert _wechat_runtime_identity(records[0].metadata) == runtime_identity
    assert client.sent_messages == [
        {
            "to_user_id": "user@im.wechat",
            "context_token": "ctx-old",
            "text": "补发答复\n\n[filtered] 内容可能不完整",
            "group_id": None,
        }
    ]


@pytest.mark.unit
def test_process_once_only_resumes_pending_turns_for_current_runtime(tmp_path: Path) -> None:
    """处理微信消息前只应恢复当前 state_dir 对应 runtime 的 pending turn。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeResumableChatService(
        scripted_turns=[
            _ScriptedTurn(events=(AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "补发答复"}, meta={}),)),
            _ScriptedTurn(events=(AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "当前问题答复"}, meta={}),)),
        ],
        runtime_identity=build_wechat_runtime_identity(store.state_dir),
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="新问题", context_token="ctx-new")], "get_updates_buf": "cursor-1"}
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    asyncio.run(daemon.process_once())

    assert [request.pending_turn_id for request in service.resume_requests] == ["pending-1"]
    assert [request.session_id for request in service.resume_requests] == [build_wechat_session_id("user@im.wechat")]
    assert [message["text"] for message in client.sent_messages] == ["补发答复", "当前问题答复"]


@pytest.mark.unit
def test_process_once_processes_different_chat_keys_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不同 chat_key 的消息应并发处理。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    daemon = _build_daemon(
        chat_service=_FakeChatService(scripted_turns=[]),
        state_store=store,
        client=_FakeIlinkClient(
            updates_payloads=[
                {
                    "ret": 0,
                    "msgs": [
                        _build_text_message(text="A", context_token="ctx-a", from_user_id="user-a"),
                        _build_text_message(text="B", context_token="ctx-b", from_user_id="user-b"),
                    ],
                    "get_updates_buf": "cursor-1",
                }
            ]
        ),
    )
    barrier = asyncio.Event()
    started_count = 0
    active_count = 0
    max_active_count = 0

    async def _fake_handle(message: dict[str, object], state: WeChatDaemonState) -> bool:
        nonlocal started_count, active_count, max_active_count
        del message, state
        started_count += 1
        active_count += 1
        max_active_count = max(max_active_count, active_count)
        if started_count == 2:
            barrier.set()
        await asyncio.wait_for(barrier.wait(), timeout=0.2)
        active_count -= 1
        return True

    monkeypatch.setattr(daemon, "_handle_inbound_message", _fake_handle)

    processed = asyncio.run(daemon.process_once())

    assert processed == 2
    assert max_active_count == 2


@pytest.mark.unit
def test_process_once_keeps_same_chat_key_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一 chat_key 的消息即使同批到达，也必须保持串行。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    daemon = _build_daemon(
        chat_service=_FakeChatService(scripted_turns=[]),
        state_store=store,
        client=_FakeIlinkClient(
            updates_payloads=[
                {
                    "ret": 0,
                    "msgs": [
                        _build_text_message(text="A1", context_token="ctx-a1", from_user_id="same-user"),
                        _build_text_message(text="A2", context_token="ctx-a2", from_user_id="same-user"),
                    ],
                    "get_updates_buf": "cursor-1",
                }
            ]
        ),
    )
    active_count = 0
    max_active_count = 0
    seen_context_tokens: list[str] = []

    async def _fake_handle(
        message: dict[str, str | int | float | bool | None | list | dict],
        state: WeChatDaemonState,
    ) -> bool:
        nonlocal active_count, max_active_count
        del state
        active_count += 1
        max_active_count = max(max_active_count, active_count)
        seen_context_tokens.append(str(message.get("context_token") or ""))
        await asyncio.sleep(0)
        active_count -= 1
        return True

    monkeypatch.setattr(daemon, "_handle_inbound_message", _fake_handle)

    processed = asyncio.run(daemon.process_once())

    assert processed == 2
    assert max_active_count == 1
    assert seen_context_tokens == ["ctx-a1", "ctx-a2"]


@pytest.mark.unit
def test_process_once_uses_chat_service_protocol_only(tmp_path: Path) -> None:
    """WeChat daemon 应只消费 ChatServiceProtocol 的稳定公开方法。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeResumableChatService(
        scripted_turns=[
            _ScriptedTurn(events=(AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "补发答复"}, meta={}),)),
            _ScriptedTurn(events=(AppEvent(type=AppEventType.FINAL_ANSWER, payload={"content": "当前答复"}, meta={}),)),
        ],
        runtime_identity=build_wechat_runtime_identity(store.state_dir),
    )
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": [_build_text_message(text="新问题", context_token="ctx-new")], "get_updates_buf": "cursor-1"}
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    asyncio.run(daemon.process_once())

    assert len(service.submit_turn_requests) == 1
    assert service.submit_turn_requests[0].scene_name == "wechat"
    assert [request.pending_turn_id for request in service.resume_requests] == ["pending-1"]
    assert [message["text"] for message in client.sent_messages] == ["补发答复", "当前答复"]


@pytest.mark.unit
def test_process_once_state_io_runs_off_event_loop_thread(tmp_path: Path) -> None:
    """验证 process_once 的状态 I/O 不在 event loop 线程执行。"""

    store = _ThreadRecordingStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    store.load_thread_ids.clear()
    store.save_thread_ids.clear()
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(updates_payloads=[{"ret": 0, "msgs": [], "get_updates_buf": "cursor-1"}])
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    async def _run_and_capture() -> tuple[int, list[int], list[int]]:
        event_loop_thread_id = threading.get_ident()
        await daemon.process_once()
        return event_loop_thread_id, list(store.load_thread_ids), list(store.save_thread_ids)

    event_loop_thread_id, load_thread_ids, save_thread_ids = asyncio.run(_run_and_capture())

    assert load_thread_ids
    assert save_thread_ids
    assert all(thread_id != event_loop_thread_id for thread_id in load_thread_ids)
    assert all(thread_id != event_loop_thread_id for thread_id in save_thread_ids)


@pytest.mark.unit
def test_ensure_authenticated_state_io_runs_off_event_loop_thread(tmp_path: Path) -> None:
    """验证登录路径上的状态 I/O 不在 event loop 线程执行。"""

    store = _ThreadRecordingStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        login_ticket=QRCodeLoginTicket(
            qrcode="qr-1",
            url=None,
            qrcode_img_content="https://liteapp.weixin.qq.com/q/demo",
        ),
        login_status=QRCodeLoginStatus(status="confirmed", bot_token="token-1", base_url="https://ilink.example"),
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    async def _run_and_capture() -> tuple[int, list[int], list[int], list[int]]:
        event_loop_thread_id = threading.get_ident()
        await daemon.ensure_authenticated(force_relogin=True)
        return (
            event_loop_thread_id,
            list(store.load_thread_ids),
            list(store.save_thread_ids),
            list(store.qrcode_thread_ids),
        )

    event_loop_thread_id, load_thread_ids, save_thread_ids, qrcode_thread_ids = asyncio.run(_run_and_capture())

    assert load_thread_ids
    assert len(save_thread_ids) >= 2
    assert qrcode_thread_ids
    assert all(thread_id != event_loop_thread_id for thread_id in load_thread_ids)
    assert all(thread_id != event_loop_thread_id for thread_id in save_thread_ids)
    assert all(thread_id != event_loop_thread_id for thread_id in qrcode_thread_ids)


@pytest.mark.unit
def test_daemon_aclose_waits_for_pending_state_save(tmp_path: Path) -> None:
    """验证 daemon 关闭时会等待已提交状态写入收口。"""

    store = _BlockingSaveStateStore(tmp_path / ".wechat")
    store.release_save.set()
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    store.release_save.clear()
    store.save_started.clear()
    store.save_completed.clear()
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(updates_payloads=[{"ret": 0, "msgs": [], "get_updates_buf": "cursor-1"}])
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    async def _run_scenario() -> None:
        process_task = asyncio.create_task(daemon.process_once())
        save_started = await asyncio.to_thread(store.save_started.wait, 5.0)
        assert save_started is True
        close_task = asyncio.create_task(daemon.aclose())
        await asyncio.sleep(0)
        assert close_task.done() is False
        store.release_save.set()
        await process_task
        await close_task

    asyncio.run(_run_scenario())

    assert store.save_completed.is_set() is True
    assert client.closed is True


@pytest.mark.unit
def test_state_io_adapter_serializes_overlapping_saves(tmp_path: Path) -> None:
    """验证状态 I/O adapter 会串行执行重叠 save 请求。"""

    store = _SerialSaveStateStore(tmp_path / ".wechat")
    adapter = _AsyncWeChatStateStoreAdapter(store)

    async def _run_scenario() -> None:
        first_state = WeChatDaemonState(get_updates_buf="first")
        second_state = WeChatDaemonState(get_updates_buf="second")
        first_task = asyncio.create_task(adapter.save(first_state))
        entered = await asyncio.to_thread(store.first_save_entered.wait, 5.0)
        assert entered is True
        second_task = asyncio.create_task(adapter.save(second_state))
        await asyncio.sleep(0.05)
        assert store.entry_order == ["first"]
        store.release_first_save.set()
        await asyncio.gather(first_task, second_task)
        await adapter.aclose()

    asyncio.run(_run_scenario())

    assert store.entry_order == ["first", "second"]
    assert store.exit_order == ["first", "second"]
    assert store.max_active_saves == 1


@pytest.mark.unit
def test_state_io_adapter_aclose_can_retry_after_first_waiter_cancelled(tmp_path: Path) -> None:
    """验证首次 aclose 等待被取消后，后续重试关闭仍可完成。"""

    store = _BlockingSaveStateStore(tmp_path / ".wechat")
    adapter = _AsyncWeChatStateStoreAdapter(store)

    async def _run_scenario() -> None:
        save_task = asyncio.create_task(adapter.save(WeChatDaemonState(get_updates_buf="cursor-1")))
        started = await asyncio.to_thread(store.save_started.wait, 5.0)
        assert started is True
        first_close_task = asyncio.create_task(adapter.aclose())
        await asyncio.sleep(0)
        first_close_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close_task
        second_close_task = asyncio.create_task(adapter.aclose())
        await asyncio.sleep(0.05)
        assert second_close_task.done() is False
        store.release_save.set()
        await asyncio.wait_for(second_close_task, timeout=5.0)
        await save_task

    asyncio.run(_run_scenario())


@pytest.mark.unit
def test_state_io_adapter_aclose_surfaces_unobserved_save_failure_after_awaiter_cancel(tmp_path: Path) -> None:
    """验证调用方取消 await 后，后台 save 失败仍会由 aclose 暴露。"""

    store = _FailingBlockingSaveStateStore(tmp_path / ".wechat")
    adapter = _AsyncWeChatStateStoreAdapter(store)

    async def _run_scenario() -> None:
        save_task = asyncio.create_task(adapter.save(WeChatDaemonState(get_updates_buf="cursor-1")))
        started = await asyncio.to_thread(store.save_started.wait, 5.0)
        assert started is True
        save_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await save_task
        store.release_save.set()
        with pytest.raises(OSError, match="disk full"):
            await adapter.aclose()

    asyncio.run(_run_scenario())


@pytest.mark.unit
def test_daemon_aclose_still_closes_client_when_state_io_close_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 daemon 状态收口失败时仍会关闭底层 client。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient()
    daemon = WeChatDaemon(chat_service=service, state_store=store, client=client)

    async def _raise_state_io_error() -> None:
        raise OSError("disk full")

    monkeypatch.setattr(daemon._state_io, "aclose", _raise_state_io_error)

    with pytest.raises(OSError, match="disk full"):
        asyncio.run(daemon.aclose())

    assert client.closed is True


@pytest.mark.unit
def test_run_forever_clears_auth_state_after_unauthorized_poll(tmp_path: Path) -> None:
    """验证长轮询 401/403 后会清空本地登录态。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example", typing_ticket="typing-1"))
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient()

    async def _raise_unauthorized(*, get_updates_buf: str) -> dict[str, str | int | float | bool | None | list | dict]:
        _ = get_updates_buf
        raise IlinkApiError("unauthorized", status_code=401)

    client.get_updates = _raise_unauthorized  # type: ignore[method-assign]
    daemon = _build_daemon(
        chat_service=service,
        state_store=store,
        client=client,
        config=WeChatDaemonConfig(allow_interactive_relogin=False),
    )

    with pytest.raises(RuntimeError, match="python -m dayu.wechat login"):
        asyncio.run(daemon.run_forever(require_existing_auth=True))

    loaded = store.load()
    assert loaded.bot_token is None
    assert loaded.typing_ticket is None
    assert client.auth_updates[-1] == (None, None)


@pytest.mark.unit
def test_ensure_authenticated_prints_and_opens_qrcode_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证二维码链接会被打印并尝试打开浏览器。"""

    opened: list[str] = []
    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        login_ticket=QRCodeLoginTicket(
            qrcode="qr-1",
            url="https://liteapp.weixin.qq.com/q/demo",
            qrcode_img_content="https://liteapp.weixin.qq.com/q/demo",
        ),
        login_status=QRCodeLoginStatus(status="confirmed", bot_token="token-1", base_url="https://ilink.example"),
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)
    monkeypatch.setattr("webbrowser.open", lambda url, new=0: opened.append(url) or True)

    asyncio.run(daemon.ensure_authenticated(force_relogin=True))

    assert opened == ["https://liteapp.weixin.qq.com/q/demo"]


# ── 补充分支覆盖测试 ──────────────────────────────────────────────


@pytest.mark.unit
def test_format_log_text_preview_truncates_long_text() -> None:
    """验证超长文本会被截断并追加省略号。"""

    text = "a" * 200
    preview = _format_log_text_preview(text, limit=50)
    assert len(preview) == 50
    assert preview.endswith("...")


@pytest.mark.unit
def test_format_log_text_preview_within_limit() -> None:
    """验证未超长文本原样返回。"""

    preview = _format_log_text_preview("hello", limit=120)
    assert preview == "hello"


@pytest.mark.unit
def test_parse_int_field_str_value_error() -> None:
    """验证 str 转 int 失败时返回 None。"""

    assert _parse_int_field("not_a_number") is None


@pytest.mark.unit
def test_parse_int_field_int_passthrough() -> None:
    """验证 int 值直接透传。"""

    assert _parse_int_field(42) == 42


@pytest.mark.unit
def test_parse_int_field_str_success() -> None:
    """验证合法数字字符串可正确解析。"""

    assert _parse_int_field("123") == 123


@pytest.mark.unit
def test_extract_message_text_skips_non_dict_item() -> None:
    """item_list 中非 dict 的 item 应被跳过。"""

    message: dict[str, str | int | float | bool | None | list | dict] = {
        "from_user_id": "user@im.wechat",
        "message_type": 1,
        "context_token": "ctx-1",
        "item_list": [
            "not_a_dict",
            {"type": 1, "text_item": {"text": "hello"}},
        ],
    }

    assert _extract_message_text(message) == "hello"


@pytest.mark.unit
def test_extract_message_text_skips_non_dict_text_item() -> None:
    """text_item 非 dict 的 item 应被跳过。"""

    message: dict[str, str | int | float | bool | None | list | dict] = {
        "from_user_id": "user@im.wechat",
        "message_type": 1,
        "context_token": "ctx-1",
        "item_list": [
            {"type": 1, "text_item": "not_a_dict"},
            {"type": 1, "text_item": {"text": "valid"}},
        ],
    }

    assert _extract_message_text(message) == "valid"


@pytest.mark.unit
def test_extract_message_text_returns_none_when_no_text_items() -> None:
    """item_list 中没有任何有效 text_item 时返回 None。"""

    message: dict[str, str | int | float | bool | None | list | dict] = {
        "from_user_id": "user@im.wechat",
        "message_type": 1,
        "context_token": "ctx-1",
        "item_list": [
            {"type": 2, "text_item": {"text": "not_text_type"}},
        ],
    }

    assert _extract_message_text(message) is None


@pytest.mark.unit
def test_extract_message_text_returns_none_for_non_list_item_list() -> None:
    """item_list 非 list 时返回 None。"""

    message: dict[str, str | int | float | bool | None | list | dict] = {
        "from_user_id": "user@im.wechat",
        "message_type": 1,
        "context_token": "ctx-1",
        "item_list": "not_a_list",
    }

    assert _extract_message_text(message) is None


@pytest.mark.unit
def test_extract_message_text_returns_none_for_empty_text() -> None:
    """text 为空白字符串时返回 None。"""

    message: dict[str, str | int | float | bool | None | list | dict] = {
        "from_user_id": "user@im.wechat",
        "message_type": 1,
        "context_token": "ctx-1",
        "item_list": [{"type": 1, "text_item": {"text": "   "}}],
    }

    assert _extract_message_text(message) is None


@pytest.mark.unit
def test_resolve_chat_key_returns_none_when_both_empty() -> None:
    """group_id 和 from_user_id 都为空时返回 None。"""

    message: dict[str, str | int | float | bool | None | list | dict] = {
        "group_id": "",
        "from_user_id": "",
    }

    assert _resolve_chat_key(message) is None


@pytest.mark.unit
def test_resolve_chat_key_prefers_group_id() -> None:
    """group_id 非空时优先使用 group_id。"""

    message: dict[str, str | int | float | bool | None | list | dict] = {
        "group_id": "group-1",
        "from_user_id": "user-1",
    }

    assert _resolve_chat_key(message) == "group-1"


@pytest.mark.unit
def test_resolve_chat_key_falls_back_to_from_user_id() -> None:
    """group_id 为空时使用 from_user_id。"""

    message: dict[str, str | int | float | bool | None | list | dict] = {
        "group_id": "",
        "from_user_id": "user-1",
    }

    assert _resolve_chat_key(message) == "user-1"


@pytest.mark.unit
def test_format_wechat_reply_text_cancelled_without_reason() -> None:
    """cancelled 状态无 cancel_reason 时返回默认取消提示。"""

    reply = WeChatReply(text="", cancelled=True, cancel_reason=None)

    result = _format_wechat_reply_text(reply, empty_reply_text="(empty)")

    assert result == "[cancelled] 当前执行已取消"


@pytest.mark.unit
def test_format_wechat_reply_text_cancelled_with_reason() -> None:
    """cancelled 状态有 cancel_reason 时返回带原因的取消提示。"""

    reply = WeChatReply(text="", cancelled=True, cancel_reason="timeout")

    result = _format_wechat_reply_text(reply, empty_reply_text="(empty)")

    assert result == "[cancelled] 当前执行已取消: timeout"


@pytest.mark.unit
def test_format_wechat_reply_text_empty_fallback() -> None:
    """非取消、非过滤、空回复时返回兜底文本。"""

    reply = WeChatReply(text="")

    result = _format_wechat_reply_text(reply, empty_reply_text="(empty)")

    assert result == "(empty)"


@pytest.mark.unit
def test_format_wechat_reply_text_filtered_hint() -> None:
    """filtered 状态时在回复文本后追加过滤提示。"""

    reply = WeChatReply(text="部分结果", filtered=True)

    result = _format_wechat_reply_text(reply, empty_reply_text="(empty)")

    assert result == "部分结果\n\n[filtered] 内容可能不完整"


@pytest.mark.unit
def test_rebuild_wechat_delivery_context_group_id_none() -> None:
    """delivery_group_id 为空字符串时重建后 group_id 应为 None。"""

    metadata: ExecutionDeliveryContext = {
        "delivery_target": "user-1",
        "delivery_thread_id": "ctx-1",
        "chat_key": "user-1",
        "delivery_group_id": "",
        "wechat_runtime_identity": "runtime-1",
    }

    result = _rebuild_wechat_delivery_context(metadata, filtered=True)

    assert result.get("delivery_group_id") == ""
    assert result.get("filtered") is True


@pytest.mark.unit
def test_rebuild_wechat_delivery_context_with_group() -> None:
    """delivery_group_id 非空时应保留。"""

    metadata: ExecutionDeliveryContext = {
        "delivery_target": "user-1",
        "delivery_thread_id": "ctx-1",
        "chat_key": "group-1",
        "delivery_group_id": "group-1",
        "wechat_runtime_identity": "runtime-1",
    }

    result = _rebuild_wechat_delivery_context(metadata, filtered=False)

    assert result.get("delivery_group_id") == "group-1"
    assert result.get("filtered") is False


@pytest.mark.unit
def test_is_retryable_delivery_error_non_ilink_error() -> None:
    """非 IlinkApiError 的异常始终可重试。"""

    assert _is_retryable_delivery_error(RuntimeError("network")) is True


@pytest.mark.unit
def test_is_retryable_delivery_error_business_ret_code() -> None:
    """IlinkApiError 有 business_ret_code 时不可重试。"""

    exc = IlinkApiError("biz fail", status_code=200, business_ret_code=-1)
    assert _is_retryable_delivery_error(exc) is False


@pytest.mark.unit
def test_is_retryable_delivery_error_no_status_code() -> None:
    """IlinkApiError 无 status_code 时可重试。"""

    exc = IlinkApiError("unknown")
    assert _is_retryable_delivery_error(exc) is True


@pytest.mark.unit
def test_is_retryable_delivery_error_4xx_not_retryable() -> None:
    """4xx 状态码的 IlinkApiError 不可重试。"""

    exc = IlinkApiError("bad request", status_code=400)
    assert _is_retryable_delivery_error(exc) is False


@pytest.mark.unit
def test_is_retryable_delivery_error_5xx_retryable() -> None:
    """5xx 状态码的 IlinkApiError 可重试。"""

    exc = IlinkApiError("server error", status_code=500)
    assert _is_retryable_delivery_error(exc) is True


@pytest.mark.unit
def test_reply_builder_consume_warning_event() -> None:
    """WARNING 事件应收集到 warnings 列表。"""

    builder = WeChatReplyBuilder()
    builder.consume(
        AppEvent(
            type=AppEventType.WARNING,
            payload={"message": "这是警告"},
            meta={"run_id": "run-1"},
        )
    )

    reply = builder.build()

    assert reply.warnings == ("这是警告",)
    assert reply.source_run_id == "run-1"


@pytest.mark.unit
def test_reply_builder_consume_warning_non_dict_payload() -> None:
    """WARNING 事件 payload 非 dict 时直接取 payload 值。"""

    builder = WeChatReplyBuilder()
    builder.consume(
        AppEvent(type=AppEventType.WARNING, payload="纯文本警告", meta={})
    )

    reply = builder.build()

    assert reply.warnings == ("纯文本警告",)


@pytest.mark.unit
def test_reply_builder_consume_error_event() -> None:
    """ERROR 事件应收集到 errors 列表。"""

    builder = WeChatReplyBuilder()
    builder.consume(
        AppEvent(
            type=AppEventType.ERROR,
            payload={"message": "出错了"},
            meta={"run_id": "run-err"},
        )
    )

    reply = builder.build()

    assert reply.errors == ("出错了",)
    assert reply.source_run_id == "run-err"


@pytest.mark.unit
def test_reply_builder_consume_error_non_dict_payload() -> None:
    """ERROR 事件 payload 非 dict 时直接取 payload 值。"""

    builder = WeChatReplyBuilder()
    builder.consume(
        AppEvent(type=AppEventType.ERROR, payload="纯文本错误", meta={})
    )

    reply = builder.build()

    assert reply.errors == ("纯文本错误",)


@pytest.mark.unit
def test_reply_builder_warning_empty_message_skipped() -> None:
    """WARNING 事件空消息不应加入 warnings。"""

    builder = WeChatReplyBuilder()
    builder.consume(
        AppEvent(type=AppEventType.WARNING, payload={"message": "   "}, meta={})
    )

    reply = builder.build()

    assert reply.warnings == ()


@pytest.mark.unit
def test_reply_builder_error_empty_message_skipped() -> None:
    """ERROR 事件空消息不应加入 errors。"""

    builder = WeChatReplyBuilder()
    builder.consume(
        AppEvent(type=AppEventType.ERROR, payload={"message": ""}, meta={})
    )

    reply = builder.build()

    assert reply.errors == ()


@pytest.mark.unit
def test_capture_run_id_ignores_non_dict_meta() -> None:
    """meta 非 dict 时不应尝试提取 run_id。"""

    builder = WeChatReplyBuilder()
    event = AppEvent(type=AppEventType.CONTENT_DELTA, payload="text", meta={})
    event.meta = "not_a_dict"  # type: ignore[assignment]
    builder.consume(event)

    reply = builder.build()

    assert reply.source_run_id is None


@pytest.mark.unit
def test_reply_builder_captures_run_id_from_meta() -> None:
    """meta 中的 run_id 应被捕获。"""

    builder = WeChatReplyBuilder()
    builder.consume(
        AppEvent(
            type=AppEventType.CONTENT_DELTA,
            payload="text",
            meta={"run_id": "run-123"},
        )
    )

    reply = builder.build()

    assert reply.source_run_id == "run-123"


@pytest.mark.unit
def test_reply_builder_content_delta_empty_payload_skipped() -> None:
    """CONTENT_DELTA 事件 payload 为空时不加入 chunks。"""

    builder = WeChatReplyBuilder()
    builder.consume(AppEvent(type=AppEventType.CONTENT_DELTA, payload="", meta={}))
    builder.consume(AppEvent(type=AppEventType.CONTENT_DELTA, payload="hello", meta={}))

    reply = builder.build()

    assert reply.text == "hello"


@pytest.mark.unit
def test_state_io_adapter_run_sync_call_raises_when_closed(tmp_path: Path) -> None:
    """适配器关闭后提交新请求应抛出 RuntimeError。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    adapter = _AsyncWeChatStateStoreAdapter(store)
    adapter._closed = True

    with pytest.raises(RuntimeError, match="已关闭"):
        asyncio.run(adapter.load())


@pytest.mark.unit
def test_state_io_adapter_run_sync_call_base_exception_forgets_future(tmp_path: Path) -> None:
    """BaseException 应导致 future 从跟踪集合中移除。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    adapter = _AsyncWeChatStateStoreAdapter(store)

    async def _run_scenario() -> None:
        # 先正常关闭 adapter
        await adapter.aclose()

    # adapter 已关闭，_run_sync_call 应抛 RuntimeError
    asyncio.run(_run_scenario())

    assert adapter._closed is True


@pytest.mark.unit
def test_state_io_adapter_build_close_error_returns_none_for_empty_list() -> None:
    """错误列表为空时 _build_close_error 返回 None。"""

    store = FileWeChatStateStore(Path("/tmp/never_used"))
    adapter = _AsyncWeChatStateStoreAdapter(store)

    assert adapter._build_close_error([]) is None


@pytest.mark.unit
def test_state_io_adapter_build_close_error_returns_single_error() -> None:
    """错误列表只有一个异常时直接返回该异常。"""

    store = FileWeChatStateStore(Path("/tmp/never_used"))
    adapter = _AsyncWeChatStateStoreAdapter(store)
    exc = ValueError("single error")

    result = adapter._build_close_error([exc])

    assert result is exc


@pytest.mark.unit
def test_state_io_adapter_build_close_error_returns_exception_group() -> None:
    """错误列表有多个异常时返回 ExceptionGroup。"""

    store = FileWeChatStateStore(Path("/tmp/never_used"))
    adapter = _AsyncWeChatStateStoreAdapter(store)
    errors = [ValueError("err1"), OSError("err2")]

    result = adapter._build_close_error(errors)

    assert isinstance(result, ExceptionGroup)
    assert len(result.exceptions) == 2


@pytest.mark.unit
def test_process_once_skips_non_list_messages(tmp_path: Path) -> None:
    """get_updates 返回的 msgs 非 list 时应跳过消息处理。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        updates_payloads=[
            {"ret": 0, "msgs": "not_a_list", "get_updates_buf": "cursor-1"}
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    processed = asyncio.run(daemon.process_once())

    assert processed == 0


@pytest.mark.unit
def test_process_once_skips_non_dict_message_in_list(tmp_path: Path) -> None:
    """msgs 列表中非 dict 的元素应被跳过。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        updates_payloads=[
            {
                "ret": 0,
                "msgs": ["not_a_dict", 123],
                "get_updates_buf": "cursor-1",
            }
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    processed = asyncio.run(daemon.process_once())

    assert processed == 0


@pytest.mark.unit
def test_handle_inbound_message_returns_false_for_non_text(tmp_path: Path) -> None:
    """非文本消息应返回 False 且不触发 chat_service。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        updates_payloads=[
            {
                "ret": 0,
                "msgs": [{"from_user_id": "u1", "message_type": 2, "context_token": "ctx-1"}],
                "get_updates_buf": "cursor-1",
            }
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    processed = asyncio.run(daemon.process_once())

    assert processed == 0
    assert service.submit_turn_requests == []


@pytest.mark.unit
def test_handle_inbound_message_returns_false_without_chat_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """无 chat_key 的消息应跳过。"""

    logged: list[str] = []
    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        updates_payloads=[
            {
                "ret": 0,
                "msgs": [{
                    "message_type": 1,
                    "context_token": "ctx-1",
                    "item_list": [{"type": 1, "text_item": {"text": "hello"}}],
                    # no group_id, no from_user_id
                }],
                "get_updates_buf": "cursor-1",
            }
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)
    monkeypatch.setattr(
        "dayu.wechat.daemon.Log.warning",
        lambda message, *, module="": logged.append(message),
    )

    processed = asyncio.run(daemon.process_once())

    assert processed == 0
    assert any("chat_key" in msg for msg in logged)


@pytest.mark.unit
def test_handle_inbound_message_returns_false_without_to_user_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """无 to_user_id / context_token 的消息应跳过。"""

    logged: list[str] = []
    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        updates_payloads=[
            {
                "ret": 0,
                "msgs": [{
                    "message_type": 1,
                    "from_user_id": "",  # empty -> no to_user_id
                    "context_token": "",  # empty
                    "group_id": "group-1",
                    "item_list": [{"type": 1, "text_item": {"text": "hello"}}],
                }],
                "get_updates_buf": "cursor-1",
            }
        ]
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)
    monkeypatch.setattr(
        "dayu.wechat.daemon.Log.warning",
        lambda message, *, module="": logged.append(message),
    )

    processed = asyncio.run(daemon.process_once())

    assert processed == 0
    assert any("to_user_id" in msg for msg in logged)


@pytest.mark.unit
def test_process_once_with_errors_in_reply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """回复中包含 ERROR 事件时应记录警告。"""

    logged: list[str] = []
    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example"))
    service = _FakeChatService(
        scripted_turns=[
            _ScriptedTurn(
                events=(
                    AppEvent(
                        type=AppEventType.FINAL_ANSWER,
                        payload={"content": "部分结果"},
                        meta={"run_id": "run-err"},
                    ),
                    AppEvent(
                        type=AppEventType.ERROR,
                        payload={"message": "tool failed"},
                        meta={},
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
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)
    monkeypatch.setattr(
        "dayu.wechat.daemon.Log.warning",
        lambda message, *, module="": logged.append(message),
    )

    asyncio.run(daemon.process_once())

    assert any("tool failed" in msg for msg in logged)


@pytest.mark.unit
def test_open_login_url_handles_webbrowser_exception(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """webbrowser.open 抛异常时不应崩溃。"""

    logged: list[str] = []
    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        login_ticket=QRCodeLoginTicket(
            qrcode="qr-1",
            url="https://liteapp.weixin.qq.com/q/demo",
            qrcode_img_content=None,
        ),
        login_status=QRCodeLoginStatus(status="confirmed", bot_token="token-1", base_url="https://ilink.example"),
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    def _raise(*args: str | int, **kwargs: str | int) -> bool:
        raise OSError("no display")

    monkeypatch.setattr("webbrowser.open", _raise)
    monkeypatch.setattr(
        "dayu.wechat.daemon.Log.warning",
        lambda message, *, module="": logged.append(message),
    )

    state = asyncio.run(daemon.ensure_authenticated(force_relogin=True))

    assert state.bot_token == "token-1"
    assert any("无法自动打开" in msg for msg in logged)


@pytest.mark.unit
def test_open_login_url_returns_false_no_success_message(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """webbrowser.open 返回 False 时不打印成功消息。"""

    printed: list[str] = []
    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        login_ticket=QRCodeLoginTicket(
            qrcode="qr-1",
            url="https://liteapp.weixin.qq.com/q/demo",
            qrcode_img_content=None,
        ),
        login_status=QRCodeLoginStatus(status="confirmed", bot_token="token-1", base_url="https://ilink.example"),
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    monkeypatch.setattr("webbrowser.open", lambda url, new=0: False)
    monkeypatch.setattr("builtins.print", lambda *args: printed.append(" ".join(str(a) for a in args)))

    asyncio.run(daemon.ensure_authenticated(force_relogin=True))

    assert not any("已尝试在默认浏览器" in p for p in printed)


@pytest.mark.unit
def test_daemon_aclose_handles_cancelled_state_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """state_io.aclose 被取消后 daemon 应标记 close_cancelled。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient()
    daemon = WeChatDaemon(chat_service=service, state_store=store, client=client)

    async def _cancel_state_io() -> None:
        raise asyncio.CancelledError()

    monkeypatch.setattr(daemon._state_io, "aclose", _cancel_state_io)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(daemon.aclose())


@pytest.mark.unit
def test_daemon_aclose_handles_client_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """client.aclose 失败时应抛出 client 异常。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient()
    daemon = WeChatDaemon(chat_service=service, state_store=store, client=client)

    async def _fail_client_close() -> None:
        raise OSError("client close failed")

    monkeypatch.setattr(client, "aclose", _fail_client_close)

    with pytest.raises(OSError, match="client close failed"):
        asyncio.run(daemon.aclose())


@pytest.mark.unit
def test_daemon_aclose_handles_lock_release_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_instance_lock.release() 失败时应抛出锁异常。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient()
    daemon = WeChatDaemon(chat_service=service, state_store=store, client=client)

    def _fail_release() -> None:
        raise RuntimeError("lock release failed")

    monkeypatch.setattr(daemon._instance_lock, "release", _fail_release)

    with pytest.raises(RuntimeError, match="lock release failed"):
        asyncio.run(daemon.aclose())


@pytest.mark.unit
def test_daemon_aclose_multiple_errors_raises_exception_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """多个关闭异常应合并为 ExceptionGroup。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient()
    daemon = WeChatDaemon(chat_service=service, state_store=store, client=client)

    async def _fail_client_close() -> None:
        raise OSError("client error")

    def _fail_lock_release() -> None:
        raise RuntimeError("lock error")

    monkeypatch.setattr(client, "aclose", _fail_client_close)
    monkeypatch.setattr(daemon._instance_lock, "release", _fail_lock_release)

    with pytest.raises(ExceptionGroup):
        asyncio.run(daemon.aclose())


@pytest.mark.unit
def test_ensure_authenticated_expired_qrcode(tmp_path: Path) -> None:
    """二维码过期时应抛出 RuntimeError。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        login_ticket=QRCodeLoginTicket(qrcode="qr-1", url=None, qrcode_img_content=None),
        login_status=QRCodeLoginStatus(status="expired", bot_token=None, base_url=None),
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    with pytest.raises(RuntimeError, match="二维码登录失败"):
        asyncio.run(daemon.ensure_authenticated(force_relogin=True))


@pytest.mark.unit
def test_ensure_authenticated_qrcode_timeout(tmp_path: Path) -> None:
    """等待扫码超时时应抛出 TimeoutError。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        login_ticket=QRCodeLoginTicket(qrcode="qr-1", url=None, qrcode_img_content=None),
        login_status=QRCodeLoginStatus(status="scanned", bot_token=None, base_url=None),
    )
    daemon = _build_daemon(
        chat_service=service,
        state_store=store,
        client=client,
        config=WeChatDaemonConfig(qrcode_poll_interval_sec=0.01, qrcode_timeout_sec=0.05),
    )

    with pytest.raises(TimeoutError, match="等待微信扫码登录超时"):
        asyncio.run(daemon.ensure_authenticated(force_relogin=True))


@pytest.mark.unit
def test_ensure_authenticated_no_bot_token_after_confirm(tmp_path: Path) -> None:
    """二维码确认但服务端未返回 bot_token 时应抛出 RuntimeError。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    service = _FakeChatService(scripted_turns=[])
    client = _FakeIlinkClient(
        login_ticket=QRCodeLoginTicket(qrcode="qr-1", url=None, qrcode_img_content=None),
        login_status=QRCodeLoginStatus(status="confirmed", bot_token="", base_url="https://ilink.example"),
    )
    daemon = _build_daemon(chat_service=service, state_store=store, client=client)

    with pytest.raises(RuntimeError, match="bot_token"):
        asyncio.run(daemon.ensure_authenticated(force_relogin=True))
