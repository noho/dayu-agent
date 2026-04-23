"""web 路由辅助逻辑测试。"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, AsyncIterator, cast

import pytest

from dayu.contracts.fins import (
    DownloadProgressPayload,
    FinsCommandName,
    FinsEvent,
    FinsEventType,
    FinsProgressEventName,
    ProcessResultData,
)
from dayu.contracts.events import AppEvent, AppEventType
from dayu.services.contracts import ChatPendingTurnView, ChatTurnSubmission, FinsSubmission, PromptSubmission, RunAdminView
from dayu.services.protocols import (
    ChatServiceProtocol,
    FinsServiceProtocol,
    HostAdminServiceProtocol,
    PromptServiceProtocol,
    ReplyDeliveryServiceProtocol,
)
from dayu.web.fastapi_app import create_fastapi_app
from dayu.web.routes.chat import create_chat_router
from dayu.web.routes.fins import create_fins_router
from dayu.web.routes.prompt import create_prompt_router
from dayu.web.routes.runs import _build_run_response_payload, _parse_run_state
from dayu.web.routes.sessions import _parse_session_state, create_session_router
from dayu.web.routes.write import create_write_router


@dataclass(frozen=True)
class _NamedDependency:
    """组合根测试用依赖标识对象。"""

    name: str


class _CapturingRouter:
    """记录 handler 的最小 APIRouter 测试桩。"""

    def __init__(self, *, prefix: str, tags: list[str]) -> None:
        """初始化路由信息。"""

        self.prefix = prefix
        self.tags = tags
        self.routes: list[tuple[str, str]] = []
        self.handlers: dict[str, object] = {}

    def _record_handler(self, method: str, path: str, func: object) -> object:
        """记录 handler，并保留 method+path 的精确索引。"""

        self.routes.append((method, path))
        self.handlers[f"{method} {path}"] = func
        self.handlers.setdefault(path, func)
        return func

    def post(self, path: str, **_kwargs):
        """记录 post handler。"""

        def _decorator(func):
            return self._record_handler("POST", path, func)

        return _decorator

    def get(self, path: str, **_kwargs):
        """记录 get handler。"""

        def _decorator(func):
            return self._record_handler("GET", path, func)

        return _decorator

    def delete(self, path: str, **_kwargs):
        """记录 delete handler。"""

        def _decorator(func):
            return self._record_handler("DELETE", path, func)

        return _decorator


class _FakeHTTPException(Exception):
    """最小 HTTPException 测试桩。"""

    def __init__(self, *, status_code: int, detail: str) -> None:
        """记录 HTTP 错误载荷。"""

        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _FakeBaseModel:
    """最小 BaseModel 测试桩。"""

    def __init__(self, **data: object) -> None:
        """按字段写入实例属性。"""

        for key, value in data.items():
            setattr(self, key, value)


class _FakeStreamingResponse:
    """最小 StreamingResponse 测试桩。"""

    def __init__(self, content: AsyncIterator[str], media_type: str) -> None:
        """记录流式内容与媒体类型。"""

        self.body_iterator = content
        self.media_type = media_type


class _StringlessEventType:
    """用于覆盖无 `.value` 分支的事件类型测试桩。"""

    def __str__(self) -> str:
        """返回稳定字符串表示。"""

        return "custom-event-fallback"


@dataclass(frozen=True)
class _EventLike:
    """最小事件对象测试桩。"""

    type: object
    payload: object


@dataclass(frozen=True)
class _ChatBody:
    """chat handler 测试请求体。"""

    user_text: str
    ticker: str | None = None
    scene_name: str | None = None
    session_id: str | None = None


@dataclass(frozen=True)
class _ChatResumeBody:
    """chat resume handler 测试请求体。"""

    session_id: str
    pending_turn_id: str


@dataclass(frozen=True)
class _FinsDownloadBody:
    """download handler 测试请求体。"""

    ticker: str
    forms: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class _FinsProcessBody:
    """process handler 测试请求体。"""

    ticker: str
    overwrite: bool = False
    ci: bool = False


@dataclass(frozen=True)
class _PromptBody:
    """prompt handler 测试请求体。"""

    user_text: str
    ticker: str | None = None


@dataclass(frozen=True)
class _WriteBody:
    """write handler 测试请求体。"""

    ticker: str
    template_path: str = ""
    output_dir: str = ""


async def _empty_app_stream():
    """返回空 AppEvent 流。"""

    if False:
        yield cast(AppEvent, None)


async def _empty_fins_stream():
    """返回空 FinsEvent 流。"""

    if False:
        yield None


async def _single_fins_stream():
    """返回单元素 FinsEvent 流。"""

    yield FinsEvent(
        type=FinsEventType.RESULT,
        command=FinsCommandName.PROCESS,
        payload=ProcessResultData(pipeline="sec", status="ok", ticker="AAPL"),
    )


def _install_fake_route_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """安装用于 route handler 单测的 fastapi/pydantic 测试桩。"""

    fake_fastapi = ModuleType("fastapi")
    cast(Any, fake_fastapi).APIRouter = _CapturingRouter
    cast(Any, fake_fastapi).HTTPException = _FakeHTTPException
    fake_fastapi_responses = ModuleType("fastapi.responses")
    cast(Any, fake_fastapi_responses).StreamingResponse = _FakeStreamingResponse
    fake_pydantic = ModuleType("pydantic")
    cast(Any, fake_pydantic).BaseModel = _FakeBaseModel
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
    monkeypatch.setitem(sys.modules, "fastapi.responses", fake_fastapi_responses)
    monkeypatch.setitem(sys.modules, "pydantic", fake_pydantic)


async def _collect_text_chunks(stream: AsyncIterator[str]) -> list[str]:
    """收集异步文本流中的所有片段。"""

    return [chunk async for chunk in stream]


def _build_run_admin_view(
    *,
    run_id: str = "run_1",
    session_id: str | None = "session_1",
    state: str = "running",
    service_type: str = "chat_turn",
    cancel_requested_at: str | None = None,
    cancel_requested_reason: str | None = None,
    cancel_reason: str | None = None,
    scene_name: str | None = "interactive",
    created_at: str = "2026-04-03T08:00:00+00:00",
    started_at: str | None = None,
    finished_at: str | None = None,
    error_summary: str | None = None,
) -> RunAdminView:
    """构造 run 管理视图测试数据。"""

    return RunAdminView(
        run_id=run_id,
        session_id=session_id,
        service_type=service_type,
        state=state,
        cancel_requested_at=cancel_requested_at,
        cancel_requested_reason=cancel_requested_reason,
        cancel_reason=cancel_reason,
        scene_name=scene_name,
        created_at=created_at,
        started_at=started_at,
        finished_at=finished_at,
        error_summary=error_summary,
    )


def _build_session_admin_view(
    *,
    session_id: str = "session_1",
    source: str = "web",
    state: str = "active",
    scene_name: str | None = "interactive",
    created_at: str = "2026-04-03T08:00:00+00:00",
    last_activity_at: str = "2026-04-03T08:01:00+00:00",
):
    """构造 session 管理视图测试数据。"""

    from dayu.services.contracts import SessionAdminView

    return SessionAdminView(
        session_id=session_id,
        source=source,
        state=state,
        scene_name=scene_name,
        created_at=created_at,
        last_activity_at=last_activity_at,
    )


def _record_background_task(create_task_calls: list[object], coroutine: object) -> object:
    """记录后台任务并关闭协程，避免测试泄漏未 awaited 警告。"""

    create_task_calls.append(coroutine)
    close = getattr(coroutine, "close", None)
    if callable(close):
        close()
    return object()


def _as_chat_service(dependency: _NamedDependency) -> ChatServiceProtocol:
    """在组合根边界把依赖标识对象收窄为聊天服务协议。"""

    return cast(ChatServiceProtocol, dependency)


def _as_prompt_service(dependency: _NamedDependency) -> PromptServiceProtocol:
    """在组合根边界把依赖标识对象收窄为 Prompt 服务协议。"""

    return cast(PromptServiceProtocol, dependency)


def _as_fins_service(dependency: _NamedDependency) -> FinsServiceProtocol:
    """在组合根边界把依赖标识对象收窄为财报服务协议。"""

    return cast(FinsServiceProtocol, dependency)


def _as_host_admin_service(dependency: _NamedDependency) -> HostAdminServiceProtocol:
    """在组合根边界把依赖标识对象收窄为宿主管理服务协议。"""

    return cast(HostAdminServiceProtocol, dependency)


def _as_reply_delivery_service(dependency: _NamedDependency) -> ReplyDeliveryServiceProtocol:
    """在组合根边界把依赖标识对象收窄为 reply delivery 服务协议。"""

    return cast(ReplyDeliveryServiceProtocol, dependency)


@pytest.mark.unit
def test_build_run_response_payload_uses_finished_at() -> None:
    """run 响应载荷应从管理视图序列化 finished_at。"""

    payload = _build_run_response_payload(
        RunAdminView(
            run_id="run_1",
            session_id="session_1",
            service_type="prompt",
            state="succeeded",
            cancel_requested_at=None,
            cancel_requested_reason=None,
            scene_name="prompt",
            created_at="2026-04-03T08:00:00+00:00",
            started_at="2026-04-03T08:01:00+00:00",
            finished_at="2026-04-03T09:00:00+00:00",
            error_summary=None,
            cancel_reason=None,
        )
    )

    assert payload["finished_at"] == "2026-04-03T09:00:00+00:00"
    assert payload["cancel_requested_at"] is None
    assert payload["cancel_requested_reason"] is None
    assert payload["cancel_reason"] is None


@pytest.mark.unit
def test_build_run_response_payload_does_not_expose_ticker() -> None:
    """run 响应载荷不应暴露领域 ticker 字段。"""

    payload = _build_run_response_payload(
        RunAdminView(
            run_id="run_2",
            session_id="session_2",
            service_type="chat_turn",
            state="running",
            cancel_requested_at="2026-04-03T10:02:00+00:00",
            cancel_requested_reason="timeout",
            scene_name="interactive",
            created_at="2026-04-03T10:00:00+00:00",
            started_at=None,
            finished_at=None,
            error_summary=None,
            cancel_reason=None,
        )
    )

    assert "ticker" not in payload


@pytest.mark.unit
def test_parse_run_state_accepts_string_value() -> None:
    """run 状态过滤应接受合法字符串。"""

    assert _parse_run_state("running") == "running"


@pytest.mark.unit
def test_parse_run_state_rejects_invalid_value() -> None:
    """非法 run 状态应抛出 `ValueError`。"""

    with pytest.raises(ValueError):
        _parse_run_state("bad-state")


@pytest.mark.unit
def test_parse_session_state_accepts_string_value() -> None:
    """session 状态过滤应接受合法字符串。"""

    assert _parse_session_state("active") == "active"


@pytest.mark.unit
def test_parse_session_state_rejects_invalid_value() -> None:
    """非法 session 状态应抛出 `ValueError`。"""

    with pytest.raises(ValueError):
        _parse_session_state("bad-state")


@pytest.mark.unit
def test_create_write_router_does_not_depend_on_removed_application_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """write router 工厂不应依赖已删除的 `get_application`。"""

    class _FakeRouter:
        """最小 APIRouter 测试桩。"""

        def __init__(self, *, prefix: str, tags: list[str]) -> None:
            """记录初始化参数。"""

            self.prefix = prefix
            self.tags = tags
            self.routes: list[tuple[str, str]] = []

        def post(self, path: str, **_kwargs):
            """记录 post 路由装饰。"""

            def _decorator(func):
                self.routes.append(("POST", path))
                return func

            return _decorator

    class _FakeBaseModel:
        """最小 BaseModel 测试桩。"""

    class _FakeFactoryHTTPException(Exception):
        """满足 write router 导入的最小 HTTPException 测试桩。"""

    fake_fastapi = ModuleType("fastapi")
    cast(Any, fake_fastapi).APIRouter = _FakeRouter
    cast(Any, fake_fastapi).HTTPException = _FakeFactoryHTTPException
    fake_pydantic = ModuleType("pydantic")
    cast(Any, fake_pydantic).BaseModel = _FakeBaseModel

    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)
    monkeypatch.setitem(sys.modules, "pydantic", fake_pydantic)

    router = create_write_router()

    assert router.prefix == "/api"
    assert router.tags == ["write"]
    assert router.routes == [("POST", "/write")]


@pytest.mark.unit
def test_create_fastapi_app_injects_narrow_services_into_route_factories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FastAPI app 组合根应显式把窄 Service 依赖传给各 router 工厂。"""

    captured_calls: list[tuple[str, object]] = []

    class _FakeApp:
        """最小 FastAPI 测试桩。"""

        def __init__(self, *, title: str) -> None:
            """记录标题与已挂载 router。"""

            self.title = title
            self.routers: list[object] = []

        def get(self, _path: str):
            """模拟 FastAPI `get` 装饰器。"""

            def _decorator(func):
                return func

            return _decorator

        def include_router(self, router: object) -> None:
            """记录 router 挂载。"""

            self.routers.append(router)

    def _capture_single(name: str):
        """生成记录单依赖调用的 router 工厂。"""

        def _factory(service: object) -> object:
            captured_calls.append((name, service))
            return f"{name}_router"

        return _factory

    def _capture_zero(name: str):
        """生成记录零依赖调用的 router 工厂。"""

        def _factory() -> object:
            captured_calls.append((name, None))
            return f"{name}_router"

        return _factory

    fake_fastapi = ModuleType("fastapi")
    cast(Any, fake_fastapi).FastAPI = _FakeApp
    monkeypatch.setitem(sys.modules, "fastapi", fake_fastapi)

    monkeypatch.setattr("dayu.web.fastapi_app.create_session_router", _capture_single("sessions"))
    monkeypatch.setattr("dayu.web.fastapi_app.create_run_router", _capture_single("runs"))
    monkeypatch.setattr("dayu.web.fastapi_app.create_events_router", _capture_single("events"))
    def _capture_chat(chat_service: object, reply_delivery_service: object) -> object:
        captured_calls.append(("chat", chat_service))
        captured_calls.append(("reply_outbox_for_chat", reply_delivery_service))
        return "chat_router"

    monkeypatch.setattr("dayu.web.fastapi_app.create_chat_router", _capture_chat)
    monkeypatch.setattr("dayu.web.fastapi_app.create_prompt_router", _capture_single("prompt"))
    monkeypatch.setattr("dayu.web.fastapi_app.create_reply_outbox_router", _capture_single("reply_outbox"))
    monkeypatch.setattr("dayu.web.fastapi_app.create_fins_router", _capture_single("fins"))
    monkeypatch.setattr("dayu.web.fastapi_app.create_write_router", _capture_zero("write"))

    chat_service = _NamedDependency(name="chat")
    prompt_service = _NamedDependency(name="prompt")
    fins_service = _NamedDependency(name="fins")
    host_admin_service = _NamedDependency(name="admin")
    reply_delivery_service = _NamedDependency(name="reply_delivery")

    app = create_fastapi_app(
        chat_service=_as_chat_service(chat_service),
        prompt_service=_as_prompt_service(prompt_service),
        fins_service=_as_fins_service(fins_service),
        host_admin_service=_as_host_admin_service(host_admin_service),
        reply_delivery_service=_as_reply_delivery_service(reply_delivery_service),
    )

    typed_app = cast(_FakeApp, app)

    assert typed_app.title == "Dayu Web"
    assert captured_calls == [
        ("sessions", host_admin_service),
        ("runs", host_admin_service),
        ("events", host_admin_service),
        ("chat", chat_service),
        ("reply_outbox_for_chat", reply_delivery_service),
        ("prompt", prompt_service),
        ("reply_outbox", reply_delivery_service),
        ("write", None),
        ("fins", fins_service),
    ]
    assert typed_app.routers == [
        "sessions_router",
        "runs_router",
        "events_router",
        "chat_router",
        "prompt_router",
        "reply_outbox_router",
        "write_router",
        "fins_router",
    ]


@pytest.mark.unit
def test_chat_route_maps_value_error_to_400_without_creating_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat 路由应把提交期 ValueError 转为 400，且不启动后台任务。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.chat.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _FailingChatService:
        async def submit_turn(self, request):
            del request
            raise ValueError("聊天输入不能为空")

    class _ReplyDeliveryService:
        def submit_reply_for_delivery(self, request):
            del request
            raise AssertionError("当前测试不应投递 reply")

    router = cast(
        _CapturingRouter,
        create_chat_router(
            cast(ChatServiceProtocol, _FailingChatService()),
            cast(ReplyDeliveryServiceProtocol, _ReplyDeliveryService()),
        ),
    )
    handler = cast(Any, router.handlers["/chat"])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler(_ChatBody(user_text="   ")))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "聊天输入不能为空"
    assert create_task_calls == []


@pytest.mark.unit
def test_chat_route_creates_task_only_after_successful_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat 路由只应在 submit 成功后调度后台任务。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.chat.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _SuccessfulChatService:
        async def submit_turn(self, request):
            del request
            return ChatTurnSubmission(session_id="session_chat", event_stream=_empty_app_stream())

    class _ReplyDeliveryService:
        def submit_reply_for_delivery(self, request):
            del request
            raise AssertionError("当前测试不应投递 reply")

    router = cast(
        _CapturingRouter,
        create_chat_router(
            cast(ChatServiceProtocol, _SuccessfulChatService()),
            cast(ReplyDeliveryServiceProtocol, _ReplyDeliveryService()),
        ),
    )
    handler = cast(Any, router.handlers["/chat"])

    response = asyncio.run(handler(_ChatBody(user_text="hello")))

    assert response.session_id == "session_chat"
    assert len(create_task_calls) == 1


@pytest.mark.unit
def test_chat_route_normalizes_empty_session_id_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """空字符串 / 全空白 session_id 都应在路由层归一化为 None，避免下游出现未定义行为。"""

    _install_fake_route_modules(monkeypatch)
    monkeypatch.setattr(
        "dayu.web.routes.chat.asyncio.create_task",
        lambda coroutine: _record_background_task([], coroutine),
    )

    captured_requests: list[object] = []

    class _CapturingChatService:
        async def submit_turn(self, request):
            captured_requests.append(request)
            return ChatTurnSubmission(session_id="session_new", event_stream=_empty_app_stream())

    class _NoopReplyDeliveryService:
        def submit_reply_for_delivery(self, request):
            del request

    router = cast(
        _CapturingRouter,
        create_chat_router(
            cast(ChatServiceProtocol, _CapturingChatService()),
            cast(ReplyDeliveryServiceProtocol, _NoopReplyDeliveryService()),
        ),
    )
    handler = cast(Any, router.handlers["/chat"])

    asyncio.run(handler(_ChatBody(user_text="hello", session_id="")))
    asyncio.run(handler(_ChatBody(user_text="hello", session_id="   ")))
    asyncio.run(handler(_ChatBody(user_text="hello", session_id="\t\n")))

    assert len(captured_requests) == 3
    assert all(getattr(req, "session_id") is None for req in captured_requests)


@pytest.mark.unit
def test_chat_route_maps_invalid_submission_to_500_without_creating_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat 路由应把非法 Service 返回值映射为 500，且不启动后台任务。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.chat.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _BrokenChatService:
        async def submit_turn(self, request):
            del request
            return None

    class _ReplyDeliveryService:
        def submit_reply_for_delivery(self, request):
            del request
            raise AssertionError("当前测试不应投递 reply")

    router = cast(
        _CapturingRouter,
        create_chat_router(
            cast(ChatServiceProtocol, _BrokenChatService()),
            cast(ReplyDeliveryServiceProtocol, _ReplyDeliveryService()),
        ),
    )
    handler = cast(Any, router.handlers["/chat"])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler(_ChatBody(user_text="hello")))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "chat service returned invalid submission"
    assert create_task_calls == []


@pytest.mark.unit
def test_chat_resume_route_creates_task_only_after_successful_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat resume 路由只应在 resume 成功后调度后台任务。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.chat.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _SuccessfulChatService:
        async def submit_turn(self, request):
            del request
            raise AssertionError("当前测试不应调用 submit_turn")

        def list_resumable_pending_turns(self, *, session_id: str | None = None, scene_name: str | None = None):
            del scene_name
            assert session_id == "session_chat"
            return [
                ChatPendingTurnView(
                    pending_turn_id="pending_1",
                    session_id="session_chat",
                    scene_name="custom_scene",
                    user_text="old question",
                    source_run_id="run_old",
                    resumable=True,
                    state="sent_to_llm",
                    metadata={"delivery_channel": "web"},
                )
            ]

        async def resume_pending_turn(self, request):
            assert request.session_id == "session_chat"
            assert request.pending_turn_id == "pending_1"
            return ChatTurnSubmission(session_id="session_chat", event_stream=_empty_app_stream())

    class _ReplyDeliveryService:
        def submit_reply_for_delivery(self, request):
            del request
            raise AssertionError("当前测试不应投递 reply")

    router = cast(
        _CapturingRouter,
        create_chat_router(
            cast(ChatServiceProtocol, _SuccessfulChatService()),
            cast(ReplyDeliveryServiceProtocol, _ReplyDeliveryService()),
        ),
    )
    handler = cast(Any, router.handlers["POST /chat/resume"])

    response = asyncio.run(handler(_ChatResumeBody(session_id="session_chat", pending_turn_id="pending_1")))

    assert response.session_id == "session_chat"
    assert len(create_task_calls) == 1


@pytest.mark.unit
def test_chat_resume_route_uses_pending_turn_scene_name_for_background_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """chat resume 路由应把 pending turn 的真实 scene_name 传给后台消费器。"""

    _install_fake_route_modules(monkeypatch)
    captured_scene_names: list[str | None] = []
    monkeypatch.setattr(
        "dayu.web.routes.chat._start_chat_stream_consumer",
        lambda *, stream, reply_delivery_service, session_id, scene_name: (
            captured_scene_names.append(scene_name)
        ),
    )

    class _SuccessfulChatService:
        async def submit_turn(self, request):
            del request
            raise AssertionError("当前测试不应调用 submit_turn")

        def list_resumable_pending_turns(self, *, session_id: str | None = None, scene_name: str | None = None):
            del scene_name
            assert session_id == "session_chat"
            return [
                ChatPendingTurnView(
                    pending_turn_id="pending_1",
                    session_id="session_chat",
                    scene_name="earnings_review",
                    user_text="old question",
                    source_run_id="run_old",
                    resumable=True,
                    state="sent_to_llm",
                    metadata={"delivery_channel": "web"},
                )
            ]

        async def resume_pending_turn(self, request):
            assert request.session_id == "session_chat"
            assert request.pending_turn_id == "pending_1"
            return ChatTurnSubmission(session_id="session_chat", event_stream=_empty_app_stream())

    class _ReplyDeliveryService:
        def submit_reply_for_delivery(self, request):
            del request
            raise AssertionError("当前测试不应投递 reply")

    router = cast(
        _CapturingRouter,
        create_chat_router(
            cast(ChatServiceProtocol, _SuccessfulChatService()),
            cast(ReplyDeliveryServiceProtocol, _ReplyDeliveryService()),
        ),
    )
    handler = cast(Any, router.handlers["POST /chat/resume"])

    response = asyncio.run(handler(_ChatResumeBody(session_id="session_chat", pending_turn_id="pending_1")))

    assert response.session_id == "session_chat"
    assert captured_scene_names == ["earnings_review"]


@pytest.mark.unit
def test_chat_resume_route_maps_missing_pending_turn_to_404(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat resume 路由应把缺失 pending turn 映射为 404。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.chat.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _FailingChatService:
        async def submit_turn(self, request):
            del request
            raise AssertionError("当前测试不应调用 submit_turn")

        def list_resumable_pending_turns(self, *, session_id: str | None = None, scene_name: str | None = None):
            del session_id, scene_name
            return []

        async def resume_pending_turn(self, request):
            del request
            raise KeyError("missing pending turn")

    class _ReplyDeliveryService:
        def submit_reply_for_delivery(self, request):
            del request
            raise AssertionError("当前测试不应投递 reply")

    router = cast(
        _CapturingRouter,
        create_chat_router(
            cast(ChatServiceProtocol, _FailingChatService()),
            cast(ReplyDeliveryServiceProtocol, _ReplyDeliveryService()),
        ),
    )
    handler = cast(Any, router.handlers["POST /chat/resume"])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler(_ChatResumeBody(session_id="session_chat", pending_turn_id="pending_1")))

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "pending turn not found"
    assert create_task_calls == []


@pytest.mark.unit
def test_prompt_route_maps_value_error_to_400_without_creating_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """prompt 路由应把提交期 ValueError 转为 400，且不启动后台任务。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.prompt.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _FailingPromptService:
        async def submit(self, request):
            del request
            raise ValueError("Prompt 输入不能为空")

    router = cast(_CapturingRouter, create_prompt_router(cast(PromptServiceProtocol, _FailingPromptService())))
    handler = cast(Any, router.handlers["/prompt"])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler(_PromptBody(user_text="   ")))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Prompt 输入不能为空"
    assert create_task_calls == []


@pytest.mark.unit
def test_prompt_route_creates_task_only_after_successful_submit(monkeypatch: pytest.MonkeyPatch) -> None:
    """prompt 路由只应在 submit 成功后调度后台任务。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.prompt.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _SuccessfulPromptService:
        async def submit(self, request):
            del request
            return PromptSubmission(session_id="session_prompt", event_stream=_empty_app_stream())

    router = cast(_CapturingRouter, create_prompt_router(cast(PromptServiceProtocol, _SuccessfulPromptService())))
    handler = cast(Any, router.handlers["/prompt"])

    response = asyncio.run(handler(_PromptBody(user_text="hello")))

    assert response.session_id == "session_prompt"
    assert len(create_task_calls) == 1


@pytest.mark.unit
def test_write_route_returns_501_not_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """write 路由应明确返回 Web 当前不支持在线写作。"""

    _install_fake_route_modules(monkeypatch)
    router = cast(_CapturingRouter, create_write_router())
    handler = cast(Any, router.handlers["/write"])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler(_WriteBody(ticker="AAPL")))

    assert exc_info.value.status_code == 501
    assert exc_info.value.detail == "Web 端暂不支持在线写作"


@pytest.mark.unit
def test_fins_download_route_maps_value_error_to_400_without_creating_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """fins download 路由应把提交期 ValueError 转为 400，且不启动后台任务。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.fins.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _FailingFinsService:
        def submit(self, request):
            del request
            raise ValueError("ticker 不能为空")

    router = cast(_CapturingRouter, create_fins_router(cast(FinsServiceProtocol, _FailingFinsService())))
    handler = cast(Any, router.handlers["/download"])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler(_FinsDownloadBody(ticker="   ")))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "ticker 不能为空"
    assert create_task_calls == []


@pytest.mark.unit
def test_fins_process_route_maps_value_error_to_400_without_creating_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """fins process 路由应把提交期 ValueError 转为 400，且不启动后台任务。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.fins.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _FailingFinsService:
        def submit(self, request):
            del request
            raise ValueError("ticker 不能为空")

    router = cast(_CapturingRouter, create_fins_router(cast(FinsServiceProtocol, _FailingFinsService())))
    handler = cast(Any, router.handlers["/process"])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler(_FinsProcessBody(ticker="   ")))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "ticker 不能为空"
    assert create_task_calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("route_key", "body", "command_name", "expected_payload_type"),
    [
        ("/download", _FinsDownloadBody(ticker="AAPL", forms=["10-K"], overwrite=True), "DOWNLOAD", "DownloadCommandPayload"),
        ("/process", _FinsProcessBody(ticker="AAPL", overwrite=True, ci=True), "PROCESS", "ProcessCommandPayload"),
    ],
)
def test_fins_routes_schedule_background_consumer_after_success(
    monkeypatch: pytest.MonkeyPatch,
    route_key: str,
    body: object,
    command_name: str,
    expected_payload_type: str,
) -> None:
    """fins 路由只应在 submit 成功后创建后台消费任务。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.fins.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _SuccessfulFinsService:
        def submit(self, request):
            assert request.command.name.name == command_name
            assert type(request.command.payload).__name__ == expected_payload_type
            return FinsSubmission(session_id="session_fins", execution=_single_fins_stream())

    router = cast(_CapturingRouter, create_fins_router(cast(FinsServiceProtocol, _SuccessfulFinsService())))
    handler = cast(Any, router.handlers[route_key])

    response = asyncio.run(handler(body))

    assert response.session_id == "session_fins"
    assert len(create_task_calls) == 1


@pytest.mark.unit
def test_fins_route_maps_invalid_submission_to_500_without_creating_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """fins 路由应把非法 Service 返回值映射为 500，且不启动后台任务。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.fins.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _BrokenFinsService:
        def submit(self, request):
            del request
            return None

    router = cast(_CapturingRouter, create_fins_router(cast(FinsServiceProtocol, _BrokenFinsService())))
    handler = cast(Any, router.handlers["/download"])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler(_FinsDownloadBody(ticker="AAPL")))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "fins service returned invalid submission"
    assert create_task_calls == []


@pytest.mark.unit
def test_fins_route_rejects_non_stream_execution_before_background_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """fins 路由应在同步结果误入异步端点时直接返回 500。"""

    _install_fake_route_modules(monkeypatch)
    create_task_calls: list[object] = []
    monkeypatch.setattr(
        "dayu.web.routes.fins.asyncio.create_task",
        lambda coroutine: _record_background_task(create_task_calls, coroutine),
    )

    class _BrokenFinsService:
        def submit(self, request):
            del request
            return FinsSubmission(
                session_id="session_fins",
                execution=cast(Any, ProcessResultData(pipeline="sec", status="ok", ticker="AAPL")),
            )

    router = cast(_CapturingRouter, create_fins_router(cast(FinsServiceProtocol, _BrokenFinsService())))
    handler = cast(Any, router.handlers["/process"])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler(_FinsProcessBody(ticker="AAPL")))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "fins service returned invalid execution"
    assert create_task_calls == []


@pytest.mark.unit
def test_fins_consume_stream_rejects_non_streaming_result() -> None:
    """后台消费者收到同步结果时应显式失败。"""

    from dayu.contracts.fins import DownloadResultData, FinsCommandName, FinsResult
    from dayu.web.routes.fins import _consume_stream

    result = FinsResult(
        command=FinsCommandName.DOWNLOAD,
        data=DownloadResultData(pipeline="sec", status="ok", ticker="AAPL"),
    )

    with pytest.raises(TypeError):
        asyncio.run(_consume_stream(result))


@pytest.mark.unit
def test_fins_consume_stream_drains_async_iterator() -> None:
    """后台消费者应完整消费流式事件。"""

    from dayu.web.routes.fins import _consume_stream

    observed: list[int] = []

    async def _stream() -> AsyncIterator[int]:
        """生成可观察的测试流。"""

        for value in (1, 2, 3):
            observed.append(value)
            yield value

    asyncio.run(_consume_stream(_stream()))

    assert observed == [1, 2, 3]


@pytest.mark.unit
def test_sessions_create_route_maps_invalid_source_to_400(monkeypatch: pytest.MonkeyPatch) -> None:
    """sessions 创建路由应把非法 source 显式映射为 400。"""

    _install_fake_route_modules(monkeypatch)

    @dataclass(frozen=True)
    class _CreateSessionBody:
        source: str
        scene_name: str | None = None

    class _FailingHostAdminService:
        def create_session(self, *, source: str = "web", scene_name: str | None = None):
            del scene_name
            raise ValueError(source)

    router = cast(
        _CapturingRouter,
        create_session_router(cast(HostAdminServiceProtocol, _FailingHostAdminService())),
    )
    handler = cast(Any, router.handlers["POST "])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler(_CreateSessionBody(source="bad-source")))

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "invalid session source: bad-source"


@pytest.mark.unit
def test_events_router_streams_run_events_as_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    """run 事件路由应把事件流编码成 SSE。"""

    _install_fake_route_modules(monkeypatch)

    async def _run_events() -> AsyncIterator[AppEvent]:
        """生成测试用 run 事件流。"""

        yield AppEvent(type=AppEventType.CONTENT_DELTA, payload={"text": "hello"})

    class _HostAdminService:
        def subscribe_run_events(self, run_id: str) -> AsyncIterator[AppEvent]:
            assert run_id == "run_1"
            return _run_events()

    from dayu.web.routes.events import create_events_router

    router = cast(_CapturingRouter, create_events_router(cast(HostAdminServiceProtocol, _HostAdminService())))
    handler = cast(Any, router.handlers["/runs/{run_id}/events"])

    response = asyncio.run(handler("run_1"))
    typed_response = cast(_FakeStreamingResponse, response)
    chunks = asyncio.run(_collect_text_chunks(typed_response.body_iterator))

    assert typed_response.media_type == "text/event-stream"
    assert chunks == ['data: {"type": "content_delta", "payload": {"text": "hello"}}\n\n']


@pytest.mark.unit
def test_events_router_streams_fins_events_as_sse(monkeypatch: pytest.MonkeyPatch) -> None:
    """run 事件路由应能序列化 FinsEvent 的 dataclass payload。"""

    _install_fake_route_modules(monkeypatch)

    async def _run_events() -> AsyncIterator[FinsEvent]:
        """生成测试用 Fins 事件流。"""

        yield FinsEvent(
            type=FinsEventType.RESULT,
            command=FinsCommandName.PROCESS,
            payload=ProcessResultData(pipeline="sec", status="ok", ticker="AAPL"),
        )

    class _HostAdminService:
        def subscribe_run_events(self, run_id: str) -> AsyncIterator[FinsEvent]:
            assert run_id == "run_1"
            return _run_events()

    from dayu.web.routes.events import create_events_router

    router = cast(_CapturingRouter, create_events_router(cast(HostAdminServiceProtocol, _HostAdminService())))
    handler = cast(Any, router.handlers["/runs/{run_id}/events"])

    response = asyncio.run(handler("run_1"))
    typed_response = cast(_FakeStreamingResponse, response)
    chunks = asyncio.run(_collect_text_chunks(typed_response.body_iterator))
    payload = json.loads(chunks[0].removeprefix("data: ").strip())

    assert payload == {
        "type": "result",
        "command": "process",
        "payload": {
            "pipeline": "sec",
            "status": "ok",
            "ticker": "AAPL",
            "overwrite": False,
            "ci": False,
            "filings": [],
            "filing_summary": {
                "total": 0,
                "processed": 0,
                "failed": 0,
                "skipped": 0,
                "todo": False,
            },
            "materials": [],
            "material_summary": {
                "total": 0,
                "processed": 0,
                "failed": 0,
                "skipped": 0,
                "todo": False,
            },
        },
    }


@pytest.mark.unit
def test_events_router_preserves_fins_command_discriminator(monkeypatch: pytest.MonkeyPatch) -> None:
    """run 事件路由应保留 FinsEvent 的 command 判别字段。"""

    _install_fake_route_modules(monkeypatch)

    async def _run_events() -> AsyncIterator[FinsEvent]:
        """生成测试用 Fins progress 事件流。"""

        yield FinsEvent(
            type=FinsEventType.PROGRESS,
            command=FinsCommandName.DOWNLOAD,
            payload=DownloadProgressPayload(
                event_type=FinsProgressEventName.FILE_DOWNLOADED,
                ticker="AAPL",
                document_id="doc_1",
                name="aapl-10k.htm",
                file_count=1,
            ),
        )

    class _HostAdminService:
        def subscribe_run_events(self, run_id: str) -> AsyncIterator[FinsEvent]:
            assert run_id == "run_1"
            return _run_events()

    from dayu.web.routes.events import create_events_router

    router = cast(_CapturingRouter, create_events_router(cast(HostAdminServiceProtocol, _HostAdminService())))
    handler = cast(Any, router.handlers["/runs/{run_id}/events"])

    response = asyncio.run(handler("run_1"))
    typed_response = cast(_FakeStreamingResponse, response)
    chunks = asyncio.run(_collect_text_chunks(typed_response.body_iterator))
    payload = json.loads(chunks[0].removeprefix("data: ").strip())

    assert payload == {
        "type": "progress",
        "command": "download",
        "payload": {
            "event_type": "file_downloaded",
            "ticker": "AAPL",
            "document_id": "doc_1",
            "action": None,
            "name": "aapl-10k.htm",
            "form_type": None,
            "file_count": 1,
            "size": None,
            "message": None,
            "reason": None,
            "filing_result": None,
        },
    }


@pytest.mark.unit
def test_events_router_uses_string_fallback_for_nonstandard_event_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """session 事件路由应在无 `.value` 时退回 `str(type)`。"""

    _install_fake_route_modules(monkeypatch)

    async def _session_events() -> AsyncIterator[_EventLike]:
        """生成无 `.value` 的事件流。"""

        yield _EventLike(type=_StringlessEventType(), payload={"seq": 1})

    class _HostAdminService:
        def subscribe_session_events(self, session_id: str) -> AsyncIterator[_EventLike]:
            assert session_id == "session_1"
            return _session_events()

    from dayu.web.routes.events import create_events_router

    router = cast(_CapturingRouter, create_events_router(cast(HostAdminServiceProtocol, _HostAdminService())))
    handler = cast(Any, router.handlers["/sessions/{session_id}/events"])

    response = asyncio.run(handler("session_1"))
    typed_response = cast(_FakeStreamingResponse, response)
    chunks = asyncio.run(_collect_text_chunks(typed_response.body_iterator))
    payload = json.loads(chunks[0].removeprefix("data: ").strip())

    assert payload == {"type": "custom-event-fallback", "payload": {"seq": 1}}


@pytest.mark.unit
def test_events_router_maps_runtime_error_to_501(monkeypatch: pytest.MonkeyPatch) -> None:
    """事件订阅不受支持时应返回 501。"""

    _install_fake_route_modules(monkeypatch)

    class _HostAdminService:
        def subscribe_run_events(self, run_id: str) -> AsyncIterator[AppEvent]:
            del run_id
            raise RuntimeError("SSE not enabled")

    from dayu.web.routes.events import create_events_router

    router = cast(_CapturingRouter, create_events_router(cast(HostAdminServiceProtocol, _HostAdminService())))
    handler = cast(Any, router.handlers["/runs/{run_id}/events"])

    with pytest.raises(_FakeHTTPException) as exc_info:
        asyncio.run(handler("run_1"))

    assert exc_info.value.status_code == 501
    assert exc_info.value.detail == "SSE not enabled"


@pytest.mark.unit
def test_run_router_list_get_and_cancel_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """run 路由应正确转发 list/get/cancel 成功路径。"""

    _install_fake_route_modules(monkeypatch)
    calls: list[tuple[str, object]] = []
    listed = [_build_run_admin_view(run_id="run_1"), _build_run_admin_view(run_id="run_2", state="failed")]

    class _HostAdminService:
        def list_runs(
            self,
            *,
            session_id: str | None = None,
            state: str | None = None,
            service_type: str | None = None,
        ) -> list[RunAdminView]:
            calls.append(("list", (session_id, state, service_type)))
            return listed

        def get_run(self, run_id: str) -> RunAdminView | None:
            calls.append(("get", run_id))
            return listed[0] if run_id == "run_1" else None

        def cancel_run(self, run_id: str) -> RunAdminView:
            calls.append(("cancel", run_id))
            if run_id != "run_1":
                raise KeyError(run_id)
            return _build_run_admin_view(run_id=run_id, state="cancelled", cancel_reason="user_request")

    from dayu.web.routes.runs import create_run_router

    router = cast(_CapturingRouter, create_run_router(cast(HostAdminServiceProtocol, _HostAdminService())))
    list_handler = cast(Any, router.handlers["GET "])
    get_handler = cast(Any, router.handlers["GET /{run_id}"])
    cancel_handler = cast(Any, router.handlers["POST /{run_id}/cancel"])

    listed_response = asyncio.run(list_handler(session_id="session_1", state="RUNNING", service_type="chat_turn"))
    get_response = asyncio.run(get_handler("run_1"))
    cancel_response = asyncio.run(cancel_handler("run_1"))

    assert [item.run_id for item in listed_response] == ["run_1", "run_2"]
    assert get_response.run_id == "run_1"
    assert cancel_response.state == "cancelled"
    assert cancel_response.cancel_reason == "user_request"
    assert calls == [
        ("list", ("session_1", "running", "chat_turn")),
        ("get", "run_1"),
        ("cancel", "run_1"),
    ]


@pytest.mark.unit
def test_run_router_maps_invalid_state_and_missing_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """run 路由应把非法状态和缺失记录映射成 HTTP 错误。"""

    _install_fake_route_modules(monkeypatch)

    class _HostAdminService:
        def list_runs(self, **_kwargs) -> list[RunAdminView]:
            return []

        def get_run(self, run_id: str) -> None:
            del run_id
            return None

        def cancel_run(self, run_id: str) -> RunAdminView:
            raise KeyError(run_id)

    from dayu.web.routes.runs import create_run_router

    router = cast(_CapturingRouter, create_run_router(cast(HostAdminServiceProtocol, _HostAdminService())))
    list_handler = cast(Any, router.handlers["GET "])
    get_handler = cast(Any, router.handlers["GET /{run_id}"])
    cancel_handler = cast(Any, router.handlers["POST /{run_id}/cancel"])

    with pytest.raises(_FakeHTTPException) as invalid_exc:
        asyncio.run(list_handler(state="bad-state"))
    with pytest.raises(_FakeHTTPException) as get_exc:
        asyncio.run(get_handler("missing"))
    with pytest.raises(_FakeHTTPException) as cancel_exc:
        asyncio.run(cancel_handler("missing"))

    assert invalid_exc.value.status_code == 400
    assert invalid_exc.value.detail == "invalid run state: bad-state"
    assert get_exc.value.status_code == 404
    assert get_exc.value.detail == "run not found"
    assert cancel_exc.value.status_code == 404
    assert cancel_exc.value.detail == "run not found"


@pytest.mark.unit
def test_session_router_create_list_get_and_close_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """session 路由应正确转发 create/list/get/close 成功路径。"""

    _install_fake_route_modules(monkeypatch)
    calls: list[tuple[str, object]] = []
    active = _build_session_admin_view()
    closed = _build_session_admin_view(session_id="session_2", state="closed")

    @dataclass(frozen=True)
    class _CreateSessionBody:
        source: str = "web"
        scene_name: str | None = "interactive"

    class _HostAdminService:
        def create_session(self, *, source: str = "web", scene_name: str | None = None):
            calls.append(("create", (source, scene_name)))
            return active

        def list_sessions(self, *, state: str | None = None):
            calls.append(("list", state))
            return [active, closed]

        def get_session(self, session_id: str):
            calls.append(("get", session_id))
            return active if session_id == "session_1" else None

        def close_session(self, session_id: str):
            calls.append(("close", session_id))
            if session_id != "session_1":
                raise KeyError(session_id)
            return closed, ("run_1",)

    router = cast(_CapturingRouter, create_session_router(cast(HostAdminServiceProtocol, _HostAdminService())))
    create_handler = cast(Any, router.handlers["POST "])
    list_handler = cast(Any, router.handlers["GET "])
    get_handler = cast(Any, router.handlers["GET /{session_id}"])
    close_handler = cast(Any, router.handlers["DELETE /{session_id}"])

    created = asyncio.run(create_handler(_CreateSessionBody()))
    listed_response = asyncio.run(list_handler(state="ACTIVE"))
    loaded = asyncio.run(get_handler("session_1"))
    closed_response = asyncio.run(close_handler("session_1"))

    assert created.session_id == "session_1"
    assert [item.session_id for item in listed_response] == ["session_1", "session_2"]
    assert loaded.source == "web"
    assert closed_response.state == "closed"
    assert calls == [
        ("create", ("web", "interactive")),
        ("list", "active"),
        ("get", "session_1"),
        ("close", "session_1"),
    ]


@pytest.mark.unit
def test_session_router_maps_invalid_state_and_missing_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """session 路由应把非法状态和缺失记录映射成 HTTP 错误。"""

    _install_fake_route_modules(monkeypatch)

    class _HostAdminService:
        def list_sessions(self, *, state: str | None = None):
            del state
            return []

        def get_session(self, session_id: str):
            del session_id
            return None

        def close_session(self, session_id: str):
            raise KeyError(session_id)

    router = cast(_CapturingRouter, create_session_router(cast(HostAdminServiceProtocol, _HostAdminService())))
    list_handler = cast(Any, router.handlers["GET "])
    get_handler = cast(Any, router.handlers["GET /{session_id}"])
    close_handler = cast(Any, router.handlers["DELETE /{session_id}"])

    with pytest.raises(_FakeHTTPException) as state_exc:
        asyncio.run(list_handler(state="bad-state"))
    with pytest.raises(_FakeHTTPException) as get_exc:
        asyncio.run(get_handler("missing"))
    with pytest.raises(_FakeHTTPException) as close_exc:
        asyncio.run(close_handler("missing"))

    assert state_exc.value.status_code == 400
    assert state_exc.value.detail == "invalid session state: bad-state"
    assert get_exc.value.status_code == 404
    assert get_exc.value.detail == "session not found"
    assert close_exc.value.status_code == 404
    assert close_exc.value.detail == "session not found"
