"""WeChat UI daemon。

该模块把 iLink 消息通道接到 Dayu ChatService：
- 用户输入来自微信消息
- Agent 走 WeChat 专用 scene 的 ChatService 链路
- 回复在微信侧表现为 typing + 单条最终文本
"""

from __future__ import annotations

import asyncio
import traceback
from concurrent.futures import Future as ConcurrentFuture, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Protocol, TypeVar
import webbrowser

from dayu.text import strip_markdown_fence
from dayu.execution.options import ExecutionOptions
from dayu.log import Log
from dayu.contracts.reply_outbox import ReplyOutboxState
from dayu.contracts.events import AppEvent, AppEventType
from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.services.pending_turns import has_resumable_pending_turn
from dayu.state_dir_lock import StateDirSingleInstanceLock
from dayu.services.contracts import (
    ChatResumeRequest,
    ChatTurnRequest,
    ReplyDeliveryFailureRequest,
    ReplyDeliverySubmitRequest,
    ReplyDeliveryView,
    SessionResolutionPolicy,
)
from dayu.services.protocols import ChatServiceProtocol, ReplyDeliveryServiceProtocol
from dayu.wechat.ilink_client import IlinkApiClient, IlinkApiError, QRCodeLoginStatus, QRCodeLoginTicket
from dayu.wechat.state_store import (
    build_wechat_runtime_identity,
    build_wechat_session_id,
    record_tracked_session_id,
    FileWeChatStateStore,
    WeChatDaemonState,
)

MODULE = "APP.WECHAT"
ResultType = TypeVar("ResultType")
_WECHAT_DAEMON_LOCK_FILE_NAME = ".daemon.lock"
_WECHAT_DAEMON_LOCK_REGION_BYTES = 1
DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS = 3


def _format_log_text_preview(text: str, *, limit: int = 120) -> str:
    """生成适合日志输出的文本预览。

    Args:
        text: 原始文本。
        limit: 最大预览长度。

    Returns:
        单行文本预览；超长时截断并追加省略号。

    Raises:
        无。
    """

    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 0)] + "..."


def _parse_int_field(value: object) -> int | None:
    """安全解析 iLink 载荷中的整数字段。

    Args:
        value: 原始字段值。

    Returns:
        解析成功时返回整数；无法解析时返回 `None`。

    Raises:
        无。
    """

    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


class _PendingTurnRecoveryError(RuntimeError):
    """表示当前微信会话存在无法自动恢复的 pending turn。"""

    def __init__(
        self,
        *,
        session_id: str,
        pending_turn_id: str,
        source_run_id: str,
        message: str,
    ) -> None:
        """初始化恢复失败异常。

        Args:
            session_id: 关联会话 ID。
            pending_turn_id: 待恢复 turn ID。
            source_run_id: 来源 run ID。
            message: 错误说明。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(message)
        self.session_id = session_id
        self.pending_turn_id = pending_turn_id
        self.source_run_id = source_run_id


def _extract_message_text(message: dict[str, Any]) -> str | None:
    """从 iLink 消息中提取首个文本内容。

    Args:
        message: iLink 消息对象。

    Returns:
        文本内容；不是文本消息时返回 `None`。

    Raises:
        无。
    """

    if _parse_int_field(message.get("message_type")) != 1:
        return None
    item_list = message.get("item_list")
    if not isinstance(item_list, list):
        return None
    for item in item_list:
        if not isinstance(item, dict):
            continue
        if _parse_int_field(item.get("type")) != 1:
            continue
        text_item = item.get("text_item")
        if not isinstance(text_item, dict):
            continue
        text = str(text_item.get("text") or "").strip()
        if text:
            return text
    return None


def _resolve_chat_key(message: dict[str, Any]) -> str | None:
    """为微信消息解析稳定会话键。

    Args:
        message: iLink 消息对象。

    Returns:
        群聊优先使用 `group_id`，否则使用 `from_user_id`；都缺失时返回 `None`。

    Raises:
        无。
    """

    group_id = str(message.get("group_id") or "").strip()
    if group_id:
        return group_id
    from_user_id = str(message.get("from_user_id") or "").strip()
    if from_user_id:
        return from_user_id
    return None


def _group_inbound_messages_by_chat_key(
    messages: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """按 chat_key 对入站消息分组。

    不同 `chat_key` 的消息允许并发处理；同一 `chat_key` 仍保持原始顺序串行，
    以守住 session 级恢复与发送语义。

    Args:
        messages: 当前长轮询批次中的消息列表。

    Returns:
        分组后的消息列表，组间顺序按首条消息出现顺序稳定保留。

    Raises:
        无。
    """

    grouped_messages: dict[str, list[dict[str, Any]]] = {}
    ordered_keys: list[str] = []
    for index, message in enumerate(messages):
        chat_key = _resolve_chat_key(message)
        group_key = chat_key if chat_key else f"__ungrouped__:{index}"
        if group_key not in grouped_messages:
            grouped_messages[group_key] = []
            ordered_keys.append(group_key)
        grouped_messages[group_key].append(message)
    return [grouped_messages[key] for key in ordered_keys]


def _build_wechat_delivery_context(
    *,
    to_user_id: str,
    context_token: str,
    chat_key: str,
    group_id: str | None,
    runtime_identity: str,
    filtered: bool = False,
) -> ExecutionDeliveryContext:
    """构造微信回复的交付上下文。"""

    return {
        "delivery_channel": "wechat",
        "delivery_target": to_user_id,
        "delivery_thread_id": context_token,
        "chat_key": chat_key,
        "delivery_group_id": group_id or "",
        "wechat_runtime_identity": runtime_identity,
        "filtered": filtered,
    }


def _format_wechat_reply_text(reply: WeChatReply, *, empty_reply_text: str) -> str:
    """根据回复状态生成微信侧最终展示文本。

    Args:
        reply: 聚合后的微信回复。
        empty_reply_text: 空回复兜底文本。

    Returns:
        最终发送给微信用户的文本。

    Raises:
        无。
    """

    if reply.cancelled:
        if reply.cancel_reason:
            return f"[cancelled] 当前执行已取消: {reply.cancel_reason}"
        return "[cancelled] 当前执行已取消"
    reply_text = reply.text or empty_reply_text
    if reply.filtered:
        return f"{reply_text}\n\n[filtered] 内容可能不完整"
    return reply_text


def _rebuild_wechat_delivery_context(
    metadata: ExecutionDeliveryContext,
    *,
    filtered: bool,
) -> ExecutionDeliveryContext:
    """基于已有交付上下文重建微信 delivery metadata。

    Args:
        metadata: 现有交付上下文。
        filtered: 本轮真实 filtered 状态。

    Returns:
        只包含微信稳定字段的新交付上下文。

    Raises:
        无。
    """

    return _build_wechat_delivery_context(
        to_user_id=str(metadata.get("delivery_target") or "").strip(),
        context_token=str(metadata.get("delivery_thread_id") or "").strip(),
        chat_key=str(metadata.get("chat_key") or "").strip(),
        group_id=str(metadata.get("delivery_group_id") or "").strip() or None,
        runtime_identity=str(metadata.get("wechat_runtime_identity") or "").strip(),
        filtered=filtered,
    )


_NON_RETRYABLE_DELIVERY_PROGRAMMING_ERRORS: tuple[type[BaseException], ...] = (
    TypeError,
    AttributeError,
    ValueError,
    KeyError,
    NotImplementedError,
)


def _is_retryable_delivery_error(exc: Exception) -> bool:
    """判断发送失败是否允许后续重试。

    判定规则：
    - `IlinkApiError`：iLink 已返回错误响应，按业务约定一律不可重试。
    - 编程类错误（`TypeError`/`AttributeError`/`ValueError`/`KeyError`/
      `NotImplementedError`）：重试无法恢复，直接判为不可重试。
    - 其它异常（网络、超时、未知运行时错误）：保留可重试兜底。

    Args:
        exc: 发送链路上抛出的异常实例。

    Returns:
        允许后续重试返回 ``True``；否则返回 ``False``。

    Raises:
        无。
    """

    if isinstance(exc, IlinkApiError):
        return False
    if isinstance(exc, _NON_RETRYABLE_DELIVERY_PROGRAMMING_ERRORS):
        return False
    return True


@dataclass(frozen=True)
class WeChatReply:
    """聚合后的微信回复。"""

    text: str
    source_run_id: str | None = None
    degraded: bool = False
    filtered: bool = False
    cancelled: bool = False
    cancel_reason: str | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class WeChatReplyBuilder:
    """把 AppEvent 流聚合为单条微信回复。"""

    def __init__(self) -> None:
        """初始化回复聚合器。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self._content_chunks: list[str] = []
        self._final_answer: str = ""
        self._source_run_id: str | None = None
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._degraded = False
        self._filtered = False
        self._cancelled = False
        self._cancel_reason: str | None = None

    def consume(self, event: AppEvent) -> None:
        """消费一个应用层事件。

        Args:
            event: AppEvent 对象。

        Returns:
            无。

        Raises:
            无。
        """

        if event.type == AppEventType.CONTENT_DELTA:
            text = str(event.payload or "")
            if text:
                self._content_chunks.append(text)
            self._capture_run_id(event)
            return
        if event.type == AppEventType.FINAL_ANSWER:
            payload = event.payload if isinstance(event.payload, dict) else {"content": str(event.payload)}
            self._final_answer = str(payload.get("content") or "")
            self._degraded = bool(payload.get("degraded", False))
            self._filtered = bool(payload.get("filtered", False))
            self._capture_run_id(event)
            return
        if event.type == AppEventType.WARNING:
            message = event.payload.get("message") if isinstance(event.payload, dict) else event.payload
            normalized = str(message or "").strip()
            if normalized:
                self._warnings.append(normalized)
            self._capture_run_id(event)
            return
        if event.type == AppEventType.ERROR:
            message = event.payload.get("message") if isinstance(event.payload, dict) else event.payload
            normalized = str(message or "").strip()
            if normalized:
                self._errors.append(normalized)
            self._capture_run_id(event)
            return
        if event.type == AppEventType.CANCELLED:
            self._cancelled = True
            if isinstance(event.payload, dict):
                cancel_reason = str(event.payload.get("cancel_reason") or "").strip()
                self._cancel_reason = cancel_reason or None
            self._capture_run_id(event)

    def _capture_run_id(self, event: AppEvent) -> None:
        """从事件元数据中捕获 source run ID。"""

        if not isinstance(event.meta, dict):
            return
        run_id = str(event.meta.get("run_id") or "").strip()
        if run_id:
            self._source_run_id = run_id

    def build(self) -> WeChatReply:
        """生成最终微信回复。

        Args:
            无。

        Returns:
            聚合后的单条回复。

        Raises:
            无。
        """

        final_text = "" if self._cancelled else strip_markdown_fence(
            self._final_answer or "".join(self._content_chunks)
        )
        return WeChatReply(
            text=final_text.strip(),
            source_run_id=self._source_run_id,
            degraded=self._degraded,
            filtered=self._filtered,
            cancelled=self._cancelled,
            cancel_reason=self._cancel_reason,
            warnings=tuple(self._warnings),
            errors=tuple(self._errors),
        )


class IlinkClientProtocol(Protocol):
    """daemon 依赖的 iLink 客户端协议。"""

    def update_auth(self, *, base_url: str | None, bot_token: str | None) -> None:
        """更新登录态。"""
        ...

    async def aclose(self) -> None:
        """关闭客户端。"""
        ...

    async def get_bot_qrcode(self) -> QRCodeLoginTicket:
        """获取登录二维码。"""
        ...

    async def get_qrcode_status(self, qrcode: str) -> QRCodeLoginStatus:
        """轮询二维码状态。"""
        ...

    async def get_updates(self, *, get_updates_buf: str) -> dict[str, Any]:
        """长轮询收消息。"""
        ...

    async def send_text_message(
        self,
        *,
        to_user_id: str,
        context_token: str,
        text: str,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        """发送文本消息。"""
        ...

    async def get_typing_ticket(
        self,
        *,
        ilink_user_id: str,
        context_token: str | None = None,
    ) -> str | None:
        """获取 typing ticket。"""
        ...

    async def send_typing(
        self,
        *,
        ilink_user_id: str,
        typing_ticket: str,
        status: int = 1,
    ) -> dict[str, Any]:
        """发送 typing。"""
        ...


@dataclass(frozen=True)
class WeChatDaemonConfig:
    """WeChat daemon 运行配置。"""

    scene_name: str = "wechat"
    allow_interactive_relogin: bool = True
    execution_options: ExecutionOptions | None = None
    qrcode_poll_interval_sec: float = 1.0
    qrcode_timeout_sec: float | None = None
    typing_interval_sec: float = 8.0
    delivery_scan_interval_sec: float = 2.0
    delivery_max_attempts: int = DEFAULT_WECHAT_DELIVERY_MAX_ATTEMPTS
    idle_retry_delay_sec: float = 2.0
    error_reply_text: str = "当前处理失败，请稍后再试。"
    pending_turn_blocked_reply_text: str = "当前会话存在未恢复的上一轮任务，请联系管理员处理后再试。"
    empty_reply_text: str = "当前没有生成可发送的文本结果。"


class _AsyncWeChatStateStoreAdapter:
    """为 WeChat daemon 提供可关闭的异步状态 I/O 包装。"""

    def __init__(self, state_store: FileWeChatStateStore) -> None:
        """初始化异步状态仓储适配器。

        Args:
            state_store: 同步文件状态仓储。

        Returns:
            无。

        Raises:
            无。
        """

        self._state_store = state_store
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wechat-state-io")
        self._lifecycle_lock = Lock()
        self._tracked_futures: set[ConcurrentFuture[Any]] = set()
        self._closed = False
        self._close_waiter: asyncio.Future[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._close_error: Exception | None = None

    def __del__(self) -> None:
        """在调用方遗漏显式关闭时 best-effort 回收线程执行器。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            with self._lifecycle_lock:
                self._closed = True
            self._executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            pass

    def _forget_future(self, future: ConcurrentFuture[Any]) -> None:
        """把已被调用方观测完成的 future 从跟踪集合中移除。

        Args:
            future: 已被观测完成的并发 future。

        Returns:
            无。

        Raises:
            无。
        """

        with self._lifecycle_lock:
            self._tracked_futures.discard(future)

    def _consume_wrapped_future_exception(self, future: asyncio.Future[ResultType]) -> None:
        """在调用方取消等待后显式消费 wrap_future 的完成结果，避免未检索异常告警。

        Args:
            future: 由 `asyncio.wrap_future(...)` 生成的 asyncio Future。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            future.result()
        except BaseException:
            return

    def _build_close_error(self, errors: list[Exception]) -> Exception | None:
        """把关闭阶段收集到的异常规整为单个可抛出错误。

        Args:
            errors: 关闭阶段捕获到的异常列表。

        Returns:
            单个异常；无异常时返回 `None`。

        Raises:
            无。
        """

        if not errors:
            return None
        if len(errors) == 1:
            return errors[0]
        return ExceptionGroup("WeChat 状态 I/O 关闭阶段发生多个异常", errors)

    async def _finish_close(self, tracked_futures: tuple[ConcurrentFuture[Any], ...], close_waiter: asyncio.Future[None]) -> None:
        """在后台完成状态 I/O 收口并关闭线程执行器。

        Args:
            tracked_futures: 关闭开始时仍需收口的 future 快照。
            close_waiter: 对外广播关闭完成的等待器。

        Returns:
            无。

        Raises:
            无。
        """

        close_errors: list[Exception] = []
        try:
            if tracked_futures:
                results = await asyncio.gather(
                    *(asyncio.wrap_future(future) for future in tracked_futures),
                    return_exceptions=True,
                )
                for result in results:
                    if isinstance(result, Exception):
                        close_errors.append(result)
        except Exception as exc:
            close_errors.append(exc)
        finally:
            with self._lifecycle_lock:
                for future in tracked_futures:
                    self._tracked_futures.discard(future)
            try:
                self._executor.shutdown(wait=True, cancel_futures=False)
            except Exception as exc:
                close_errors.append(exc)
            self._close_error = self._build_close_error(close_errors)
            if not close_waiter.done():
                close_waiter.set_result(None)

    async def _run_sync_call(self, func: Callable[..., ResultType], *args: Any) -> ResultType:
        """把同步状态仓储调用串行提交到专用线程执行器。

        Args:
            func: 待执行的同步函数。
            *args: 传给同步函数的参数。

        Returns:
            同步函数返回值。

        Raises:
            RuntimeError: 当适配器已经关闭时抛出。
            Exception: 透传底层同步调用异常。
        """

        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("WeChat 状态 I/O 已关闭，不能继续提交新请求")
            future = self._executor.submit(func, *args)
            self._tracked_futures.add(future)
        # 一旦请求提交，就应由 daemon 自己负责收口，不能让调用方取消把底层写入一起撤销。
        wrapped_future = asyncio.wrap_future(future)
        try:
            result = await asyncio.shield(wrapped_future)
        except asyncio.CancelledError:
            wrapped_future.add_done_callback(self._consume_wrapped_future_exception)
            raise
        except BaseException:
            self._forget_future(future)
            raise
        self._forget_future(future)
        return result

    async def load(self) -> WeChatDaemonState:
        """异步加载 daemon 状态。

        Args:
            无。

        Returns:
            当前持久化状态。

        Raises:
            RuntimeError: 当适配器已经关闭时抛出。
            ValueError: 当状态文件格式非法时抛出。
        """

        return await self._run_sync_call(self._state_store.load)

    async def save(self, state: WeChatDaemonState) -> None:
        """异步保存 daemon 状态。

        Args:
            state: 待持久化状态。

        Returns:
            无。

        Raises:
            RuntimeError: 当适配器已经关闭时抛出。
            Exception: 透传底层持久化异常。
        """

        await self._run_sync_call(self._state_store.save, state)

    async def clear_auth(self) -> None:
        """异步清除本地登录态。

        Args:
            无。

        Returns:
            无。

        Raises:
            RuntimeError: 当适配器已经关闭时抛出。
            Exception: 透传底层持久化异常。
        """

        await self._run_sync_call(self._state_store.clear_auth)

    async def write_qrcode_artifact(self, qrcode_img_content: str | None) -> Path | None:
        """异步写出登录二维码文件。

        Args:
            qrcode_img_content: 服务端返回的二维码内容。

        Returns:
            生成的二维码文件路径；无内容时返回 `None`。

        Raises:
            RuntimeError: 当适配器已经关闭时抛出。
            Exception: 透传底层文件写入异常。
        """

        return await self._run_sync_call(self._state_store.write_qrcode_artifact, qrcode_img_content)

    async def aclose(self) -> None:
        """关闭适配器并等待已提交状态 I/O 收口。

        Args:
            无。

        Returns:
            无。

        Raises:
            Exception: 当待收口状态 I/O 失败时抛出首个非取消异常。
        """

        with self._lifecycle_lock:
            close_waiter = self._close_waiter
            close_task = self._close_task
            if close_waiter is None:
                loop = asyncio.get_running_loop()
                close_waiter = loop.create_future()
                self._close_waiter = close_waiter
                self._closed = True
                tracked_futures = tuple(self._tracked_futures)
                close_task = loop.create_task(self._finish_close(tracked_futures, close_waiter))
                self._close_task = close_task
        try:
            await asyncio.shield(close_waiter)
        except asyncio.CancelledError:
            raise
        if self._close_error is not None:
            raise self._close_error


@dataclass
class WeChatDaemon:
    """WeChat 到 Dayu ChatService 的桥接 daemon。"""

    chat_service: ChatServiceProtocol
    state_store: FileWeChatStateStore
    reply_delivery_service: ReplyDeliveryServiceProtocol | None = None
    config: WeChatDaemonConfig = field(default_factory=WeChatDaemonConfig)
    client: IlinkClientProtocol = field(default_factory=IlinkApiClient)
    _state_io: _AsyncWeChatStateStoreAdapter = field(init=False, repr=False)
    _instance_lock: StateDirSingleInstanceLock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """初始化 daemon 内部协作者。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self._state_io = _AsyncWeChatStateStoreAdapter(self.state_store)
        self._instance_lock = StateDirSingleInstanceLock(
            state_dir=self.state_store.state_dir,
            lock_file_name=_WECHAT_DAEMON_LOCK_FILE_NAME,
            lock_name="WeChat daemon 单实例锁",
            lock_region_bytes=_WECHAT_DAEMON_LOCK_REGION_BYTES,
        )

    def _runtime_identity(self) -> str:
        """返回当前 daemon 的稳定 runtime identity。

        Args:
            无。

        Returns:
            当前状态目录对应的 runtime identity。

        Raises:
            无。
        """

        return build_wechat_runtime_identity(self.state_store.state_dir)

    async def aclose(self) -> None:
        """关闭 daemon 依赖资源。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        state_io_error: Exception | None = None
        client_error: Exception | None = None
        lock_error: Exception | None = None
        close_cancelled = False
        try:
            await self._state_io.aclose()
        except asyncio.CancelledError:
            close_cancelled = True
        except Exception as exc:
            state_io_error = exc
        try:
            await self.client.aclose()
        except asyncio.CancelledError:
            close_cancelled = True
        except Exception as exc:
            client_error = exc
        try:
            self._instance_lock.release()
        except Exception as exc:
            lock_error = exc
        close_errors = [error for error in (state_io_error, client_error, lock_error) if error is not None]
        if len(close_errors) > 1:
            raise ExceptionGroup("WeChat daemon 关闭阶段发生多个异常", close_errors)
        if state_io_error is not None:
            raise state_io_error
        if client_error is not None:
            raise client_error
        if lock_error is not None:
            raise lock_error
        if close_cancelled:
            raise asyncio.CancelledError()

    async def ensure_authenticated(self, *, force_relogin: bool = False) -> WeChatDaemonState:
        """确保当前 daemon 已登录到 iLink。

        Args:
            force_relogin: 是否忽略已有 token 强制重新登录。

        Returns:
            最新状态对象。

        Raises:
            TimeoutError: 当扫码等待超时。
            RuntimeError: 当二维码过期或登录失败。
        """

        state = await self._state_io.load()
        if state.bot_token and not force_relogin:
            self.client.update_auth(base_url=state.base_url, bot_token=state.bot_token)
            return state
        if force_relogin:
            state.bot_token = None
            state.typing_ticket = None
            state.base_url = state.base_url or "https://ilinkai.weixin.qq.com"
            await self._state_io.save(state)
        ticket = await self.client.get_bot_qrcode()
        artifact_path = await self._state_io.write_qrcode_artifact(ticket.qrcode_img_content)
        if artifact_path is not None:
            print(f"WeChat 登录二维码已写入: {artifact_path}")
        if ticket.url:
            print(f"WeChat 登录链接: {ticket.url}")
            self._open_login_url(ticket.url)
        deadline = None
        if self.config.qrcode_timeout_sec is not None:
            deadline = asyncio.get_running_loop().time() + float(self.config.qrcode_timeout_sec)
        while True:
            status = await self.client.get_qrcode_status(ticket.qrcode)
            if status.status == "confirmed":
                if not status.bot_token:
                    raise RuntimeError("二维码已确认，但服务端未返回 bot_token")
                state.bot_token = status.bot_token
                state.base_url = status.base_url or state.base_url
                state.typing_ticket = None
                await self._state_io.save(state)
                self.client.update_auth(base_url=state.base_url, bot_token=state.bot_token)
                Log.info("WeChat iLink 登录成功", module=MODULE)
                return state
            if status.status in {"expired", "cancelled", "canceled", "rejected", "failed"}:
                raise RuntimeError(f"二维码登录失败，状态={status.status}")
            if deadline is not None and asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("等待微信扫码登录超时")
            await asyncio.sleep(self.config.qrcode_poll_interval_sec)

    async def load_existing_authenticated_state(self) -> WeChatDaemonState:
        """加载已有登录态，并更新客户端鉴权信息。

        Args:
            无。

        Returns:
            已存在的登录态。

        Raises:
            RuntimeError: 当本地不存在可用登录态时抛出。
        """

        state = await self._state_io.load()
        if not state.bot_token:
            raise RuntimeError("未检测到 iLink 登录态，请先执行 `python -m dayu.wechat login`")
        self.client.update_auth(base_url=state.base_url, bot_token=state.bot_token)
        return state

    async def run_forever(self, *, require_existing_auth: bool = False) -> None:
        """启动 daemon 主循环。

        Args:
            require_existing_auth: 是否要求使用已有登录态启动，而不是现场扫码。

        Returns:
            无。

        Raises:
            无。
        """

        self._instance_lock.acquire()
        if require_existing_auth:
            await self.load_existing_authenticated_state()
        else:
            await self.ensure_authenticated(force_relogin=False)
        await self._recover_startup_state()
        await self._resume_pending_turns()
        stop_delivery_event = asyncio.Event()
        delivery_task = asyncio.create_task(self._run_delivery_loop(stop_delivery_event))
        Log.info("WeChat daemon 已进入运行态，开始等待新消息", module=MODULE)
        try:
            while True:
                try:
                    await self.process_once()
                except IlinkApiError as exc:
                    if exc.status_code in {401, 403}:
                        await self._reset_auth_state()
                        if self.config.allow_interactive_relogin:
                            Log.warning("iLink 登录态失效，准备重新扫码登录", module=MODULE)
                            await self.ensure_authenticated(force_relogin=True)
                            continue
                        message = "iLink 登录态已失效，请先执行 `python -m dayu.wechat login`"
                        Log.error(message, module=MODULE)
                        raise RuntimeError(message) from exc
                    Log.warning(f"iLink 长轮询失败: {exc}", module=MODULE)
                    await asyncio.sleep(self.config.idle_retry_delay_sec)
                except Exception as exc:
                    Log.error(f"WeChat daemon 发生未处理异常: {exc}", exc_info=True, module=MODULE)
                    await asyncio.sleep(self.config.idle_retry_delay_sec)
        finally:
            stop_delivery_event.set()
            delivery_task.cancel()
            with suppress(asyncio.CancelledError):
                await delivery_task

    async def _recover_startup_state(self) -> None:
        """在启动主循环前回收上一进程遗留的渠道级交付状态。"""

        await self._recover_interrupted_reply_deliveries()

    async def _recover_interrupted_reply_deliveries(self) -> None:
        """把上一进程遗留的 in-progress reply delivery 回收到可重试状态。"""

        if self.reply_delivery_service is None:
            return
        runtime_identity = self._runtime_identity()
        in_progress_records = self.reply_delivery_service.list_deliveries(
            scene_name=self.config.scene_name,
            state=ReplyOutboxState.DELIVERY_IN_PROGRESS.value,
        )
        recovered_delivery_ids: list[str] = []
        for record in in_progress_records:
            if record.metadata.get("delivery_channel") != "wechat":
                continue
            if record.metadata.get("wechat_runtime_identity") != runtime_identity:
                continue
            try:
                self.reply_delivery_service.mark_delivery_failed(
                    ReplyDeliveryFailureRequest(
                        delivery_id=record.delivery_id,
                        retryable=True,
                        error_message="上一进程在发送阶段退出，启动时回收为可重试 delivery",
                    )
                )
            except Exception as exc:
                Log.warning(
                    "微信 reply delivery 启动回收失败"
                    f" delivery_id={record.delivery_id}"
                    f" error={exc}",
                    module=MODULE,
                )
                continue
            recovered_delivery_ids.append(record.delivery_id)
        if recovered_delivery_ids:
            Log.info(
                "WeChat daemon 已回收中断中的 reply delivery"
                f" count={len(recovered_delivery_ids)}"
                f" delivery_ids={','.join(recovered_delivery_ids)}",
                module=MODULE,
            )

    async def process_once(self) -> int:
        """执行一次长轮询批次处理。

        Args:
            无。

        Returns:
            当前批次处理的文本消息数量。

        Raises:
            IlinkApiError: 当长轮询请求失败时抛出。
        """

        state = await self._state_io.load()
        self.client.update_auth(base_url=state.base_url, bot_token=state.bot_token)
        payload = await self.client.get_updates(get_updates_buf=state.get_updates_buf)
        messages = payload.get("msgs")
        if not isinstance(messages, list):
            messages = []
        structured_messages = [message for message in messages if isinstance(message, dict)]
        group_results = await asyncio.gather(
            *(
                self._handle_inbound_message_group(message_group, state)
                for message_group in _group_inbound_messages_by_chat_key(structured_messages)
            ),
            return_exceptions=True,
        )
        processed_count = 0
        for group_result in group_results:
            if isinstance(group_result, BaseException):
                tb_text = "".join(
                    traceback.format_exception(
                        type(group_result), group_result, group_result.__traceback__
                    )
                )
                Log.error(
                    f"处理 chat_key 群组失败，本组按 0 条计入: {group_result!r}\n{tb_text}",
                    module=MODULE,
                )
                continue
            processed_count += group_result
        next_cursor = payload.get("get_updates_buf")
        if isinstance(next_cursor, str) and next_cursor:
            state.get_updates_buf = next_cursor
        await self._state_io.save(state)
        await self._deliver_pending_replies()
        return processed_count

    async def _handle_inbound_message_group(
        self,
        messages: list[dict[str, Any]],
        state: WeChatDaemonState,
    ) -> int:
        """串行处理同一 chat_key 下的一组消息。

        Args:
            messages: 同一 `chat_key` 下、顺序稳定的一组消息。
            state: 当前 daemon 状态。

        Returns:
            本组中被成功处理为文本会话的消息数量。

        Raises:
            无。
        """

        processed_count = 0
        for message in messages:
            handled = await self._handle_inbound_message(message, state)
            if handled:
                processed_count += 1
        return processed_count

    async def _handle_inbound_message(self, message: dict[str, Any], state: WeChatDaemonState) -> bool:
        """处理单条入站消息。

        Args:
            message: iLink 消息对象。
            state: 当前 daemon 状态。

        Returns:
            是否成功处理为一轮文本会话。

        Raises:
            无。
        """

        user_text = _extract_message_text(message)
        if not user_text:
            return False
        chat_key = _resolve_chat_key(message)
        if not chat_key:
            Log.warning("跳过缺少 chat_key 的微信消息", module=MODULE)
            return False
        to_user_id = str(message.get("from_user_id") or "").strip()
        context_token = str(message.get("context_token") or "").strip()
        if not to_user_id or not context_token:
            Log.warning("跳过缺少 to_user_id/context_token 的微信消息", module=MODULE)
            return False
        session_id = build_wechat_session_id(chat_key)
        record_tracked_session_id(self.state_store.state_dir, session_id)
        group_id = str(message.get("group_id") or "").strip() or None
        try:
            await self._resume_pending_turns(session_id=session_id, fail_fast=True)
        except _PendingTurnRecoveryError as exc:
            Log.error(
                "当前微信会话存在无法恢复的 pending turn，已拒绝新消息"
                f" session_id={exc.session_id}"
                f" pending_turn_id={exc.pending_turn_id}"
                f" source_run_id={exc.source_run_id}"
                f" error={exc}",
                module=MODULE,
            )
            await self.client.send_text_message(
                to_user_id=to_user_id,
                context_token=context_token,
                text=self.config.pending_turn_blocked_reply_text,
                group_id=group_id,
            )
            return True
        Log.info(
            "收到微信消息"
            f" user={to_user_id}"
            f" session={session_id}"
            f" text={_format_log_text_preview(user_text)}",
            module=MODULE,
        )

        stop_event = asyncio.Event()
        typing_task = asyncio.create_task(
            self._run_typing_loop(
                to_user_id=to_user_id,
                context_token=context_token,
                group_id=group_id,
                state=state,
                stop_event=stop_event,
            )
        )
        delivery_context = _build_wechat_delivery_context(
            to_user_id=to_user_id,
            context_token=context_token,
            chat_key=chat_key,
            group_id=group_id,
            runtime_identity=self._runtime_identity(),
        )
        try:
            reply = await self._run_chat_turn(
                user_text=user_text,
                session_id=session_id,
                delivery_context=delivery_context,
            )
            reply_text = _format_wechat_reply_text(reply, empty_reply_text=self.config.empty_reply_text)
            if reply.errors:
                Log.warning(f"本轮 ChatService 返回错误事件: {reply.errors}", module=MODULE)
        except Exception as exc:
            Log.error(f"处理微信消息失败: {exc}", exc_info=True, module=MODULE)
            reply_text = self.config.error_reply_text
            reply = WeChatReply(text=reply_text)
        finally:
            stop_event.set()
            typing_task.cancel()
            with suppress(asyncio.CancelledError):
                await typing_task

        if self.reply_delivery_service is None or not reply.source_run_id:
            await self.client.send_text_message(
                to_user_id=to_user_id,
                context_token=context_token,
                text=reply_text,
                group_id=group_id,
            )
            Log.info(
                "发送微信回复"
                f" user={to_user_id}"
                f" session={session_id}"
                f" text={_format_log_text_preview(reply_text)}",
                module=MODULE,
            )
            return True

        self._submit_reply_for_delivery(
            session_id=session_id,
            scene_name=self.config.scene_name,
            source_run_id=reply.source_run_id,
            reply_text=reply_text,
            metadata=_rebuild_wechat_delivery_context(delivery_context, filtered=reply.filtered),
            filtered=reply.filtered,
        )
        await self._deliver_pending_replies(session_id=session_id)
        return True

    async def _run_chat_turn(
        self,
        *,
        user_text: str,
        session_id: str,
        delivery_context: ExecutionDeliveryContext,
    ) -> WeChatReply:
        """执行一轮 Dayu ChatService，并聚合回复。

        Args:
            user_text: 用户输入文本。
            session_id: Dayu 会话 ID。
            delivery_context: 交付上下文。

        Returns:
            聚合后的微信回复。

        Raises:
            无。
        """

        builder = WeChatReplyBuilder()
        request = ChatTurnRequest(
            session_id=session_id,
            user_text=user_text,
            execution_options=self.config.execution_options,
            scene_name=self.config.scene_name,
            session_resolution_policy=SessionResolutionPolicy.ENSURE_DETERMINISTIC,
            delivery_context=delivery_context,
        )
        submission = await self.chat_service.submit_turn(request)
        async for event in submission.event_stream:
            builder.consume(event)
        return builder.build()

    async def _resume_pending_turns(self, *, session_id: str | None = None, fail_fast: bool = False) -> None:
        """恢复微信 scene 下尚未完成的 pending turn。"""

        pending_turns = self.chat_service.list_resumable_pending_turns(
            session_id=session_id,
            scene_name=self.config.scene_name,
        )
        runtime_identity = self._runtime_identity()
        for pending_turn in pending_turns:
            if pending_turn.metadata.get("delivery_channel") != "wechat":
                continue
            if pending_turn.metadata.get("wechat_runtime_identity") != runtime_identity:
                continue
            try:
                await self._resume_single_pending_turn(pending_turn.pending_turn_id)
            except _PendingTurnRecoveryError as exc:
                if not has_resumable_pending_turn(
                    self.chat_service,
                    session_id=pending_turn.session_id,
                    scene_name=self.config.scene_name,
                    pending_turn_id=pending_turn.pending_turn_id,
                ):
                    Log.warning(
                        "恢复 pending 微信 turn 失败，但 Host 已清理记录，继续放行后续消息"
                        f" pending_turn_id={pending_turn.pending_turn_id}"
                        f" source_run_id={pending_turn.source_run_id}"
                        f" error={exc}",
                        module=MODULE,
                    )
                    continue
                if fail_fast:
                    raise
                Log.warning(
                    "恢复 pending 微信 turn 失败，已跳过"
                    f" pending_turn_id={exc.pending_turn_id}"
                    f" source_run_id={exc.source_run_id}"
                    f" error={exc}",
                    module=MODULE,
                )
            except Exception as exc:
                if not has_resumable_pending_turn(
                    self.chat_service,
                    session_id=pending_turn.session_id,
                    scene_name=self.config.scene_name,
                    pending_turn_id=pending_turn.pending_turn_id,
                ):
                    Log.warning(
                        "恢复 pending 微信 turn 失败，但 Host 已清理记录，继续放行后续消息"
                        f" pending_turn_id={pending_turn.pending_turn_id}"
                        f" source_run_id={pending_turn.source_run_id}"
                        f" error={exc}",
                        module=MODULE,
                    )
                    continue
                if fail_fast:
                    raise _PendingTurnRecoveryError(
                        session_id=pending_turn.session_id,
                        pending_turn_id=pending_turn.pending_turn_id,
                        source_run_id=pending_turn.source_run_id,
                        message=str(exc),
                    ) from exc
                Log.warning(
                    "恢复 pending 微信 turn 失败，已跳过"
                    f" pending_turn_id={pending_turn.pending_turn_id}"
                    f" source_run_id={pending_turn.source_run_id}"
                    f" error={exc}",
                    module=MODULE,
                )

    async def _resume_single_pending_turn(self, pending_turn_id: str) -> None:
        """恢复单个微信 pending turn 并重新发送回复。"""

        pending_turns = self.chat_service.list_resumable_pending_turns(scene_name=self.config.scene_name)
        pending_turn = next((item for item in pending_turns if item.pending_turn_id == pending_turn_id), None)
        if pending_turn is None:
            return
        to_user_id = str(pending_turn.metadata.get("delivery_target") or "").strip()
        context_token = str(pending_turn.metadata.get("delivery_thread_id") or "").strip()
        group_id = str(pending_turn.metadata.get("delivery_group_id") or "").strip() or None
        if not to_user_id or not context_token:
            raise _PendingTurnRecoveryError(
                session_id=pending_turn.session_id,
                pending_turn_id=pending_turn.pending_turn_id,
                source_run_id=pending_turn.source_run_id,
                message="pending 微信 turn 缺少 delivery_target 或 delivery_thread_id",
            )
        submission = await self.chat_service.resume_pending_turn(
            ChatResumeRequest(
                session_id=pending_turn.session_id,
                pending_turn_id=pending_turn_id,
            )
        )
        builder = WeChatReplyBuilder()
        async for event in submission.event_stream:
            builder.consume(event)
        reply = builder.build()
        reply_text = _format_wechat_reply_text(reply, empty_reply_text=self.config.empty_reply_text)
        if self.reply_delivery_service is None or not reply.source_run_id:
            await self.client.send_text_message(
                to_user_id=to_user_id,
                context_token=context_token,
                text=reply_text,
                group_id=group_id,
            )
            Log.info(
                "恢复并补发微信回复"
                f" pending_turn={pending_turn_id}"
                f" session={submission.session_id}"
                f" text={_format_log_text_preview(reply_text)}",
                module=MODULE,
            )
            return
        self._submit_reply_for_delivery(
            session_id=submission.session_id,
            scene_name=self.config.scene_name,
            source_run_id=reply.source_run_id,
            reply_text=reply_text,
            metadata=_rebuild_wechat_delivery_context(pending_turn.metadata, filtered=reply.filtered),
            filtered=reply.filtered,
        )
        await self._deliver_pending_replies(session_id=submission.session_id)

    def _submit_reply_for_delivery(
        self,
        *,
        session_id: str,
        scene_name: str,
        source_run_id: str,
        reply_text: str,
        metadata: ExecutionDeliveryContext,
        filtered: bool,
    ) -> ReplyDeliveryView:
        """向 reply delivery Service 提交待发送回复。

        Args:
            session_id: Dayu 会话 ID。
            scene_name: scene 名称。
            source_run_id: 源 run ID。
            reply_text: 要投递的回复文本。
            metadata: 已带有投递上下文的 metadata；不依赖其 ``filtered``
                字段取值，以避免 metadata 链路丢失该字段后语义漂移。
            filtered: 显式标注当前回复是否因内容安全过滤而降级。

        Returns:
            delivery service 返回的视图。
        """

        if self.reply_delivery_service is None:
            raise RuntimeError("reply_delivery_service 未配置，不能提交微信 delivery")
        return self.reply_delivery_service.submit_reply_for_delivery(
            ReplyDeliverySubmitRequest(
                delivery_key=f"wechat:{source_run_id}",
                session_id=session_id,
                scene_name=scene_name,
                source_run_id=source_run_id,
                reply_content=reply_text,
                metadata=_rebuild_wechat_delivery_context(
                    metadata,
                    filtered=filtered,
                ),
            )
        )

    async def _deliver_pending_replies(self, *, session_id: str | None = None) -> None:
        """发送当前可投递的微信 outbox 记录。"""

        if self.reply_delivery_service is None:
            return
        pending_records = self.reply_delivery_service.list_deliveries(
            session_id=session_id,
            scene_name=self.config.scene_name,
            state=ReplyOutboxState.PENDING_DELIVERY.value,
        )
        retryable_records = self.reply_delivery_service.list_deliveries(
            session_id=session_id,
            scene_name=self.config.scene_name,
            state=ReplyOutboxState.FAILED_RETRYABLE.value,
        )
        for record in [*pending_records, *retryable_records]:
            try:
                claimed = self.reply_delivery_service.claim_delivery(record.delivery_id)
            except (KeyError, ValueError):
                continue
            if claimed.delivery_attempt_count > self.config.delivery_max_attempts:
                self.reply_delivery_service.mark_delivery_failed(
                    ReplyDeliveryFailureRequest(
                        delivery_id=claimed.delivery_id,
                        retryable=False,
                        error_message="delivery retries exhausted",
                    )
                )
                Log.warning(
                    "微信 reply delivery 已达到最大重试次数，收口为 terminal"
                    f" delivery_id={claimed.delivery_id}"
                    f" attempts={claimed.delivery_attempt_count}"
                    f" max_attempts={self.config.delivery_max_attempts}",
                    module=MODULE,
                )
                continue
            to_user_id = str(claimed.metadata.get("delivery_target") or "").strip()
            context_token = str(claimed.metadata.get("delivery_thread_id") or "").strip()
            group_id = str(claimed.metadata.get("delivery_group_id") or "").strip() or None
            if not to_user_id or not context_token:
                self.reply_delivery_service.mark_delivery_failed(
                    ReplyDeliveryFailureRequest(
                        delivery_id=claimed.delivery_id,
                        retryable=False,
                        error_message="缺少 delivery_target 或 delivery_thread_id",
                    )
                )
                continue
            try:
                await self.client.send_text_message(
                    to_user_id=to_user_id,
                    context_token=context_token,
                    text=claimed.reply_content,
                    group_id=group_id,
                )
                self.reply_delivery_service.mark_delivery_delivered(claimed.delivery_id)
                Log.info(
                    "发送微信回复"
                    f" user={to_user_id}"
                    f" session={claimed.session_id}"
                    f" text={_format_log_text_preview(claimed.reply_content)}",
                    module=MODULE,
                )
            except Exception as exc:
                self.reply_delivery_service.mark_delivery_failed(
                    ReplyDeliveryFailureRequest(
                        delivery_id=claimed.delivery_id,
                        retryable=_is_retryable_delivery_error(exc),
                        error_message=str(exc),
                    )
                )
                Log.warning(f"微信 reply delivery 发送失败: {exc}", module=MODULE)

    async def _run_delivery_loop(self, stop_event: asyncio.Event) -> None:
        """周期性扫描并发送 reply outbox。"""

        while not stop_event.is_set():
            try:
                await self._deliver_pending_replies()
            except Exception as exc:
                Log.warning(f"微信 reply delivery 后台扫描失败: {exc}", module=MODULE)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.config.delivery_scan_interval_sec)
            except asyncio.TimeoutError:
                continue

    async def _run_typing_loop(
        self,
        *,
        to_user_id: str,
        context_token: str,
        group_id: str | None,
        state: WeChatDaemonState,
        stop_event: asyncio.Event,
    ) -> None:
        """周期性发送 typing。

        Args:
            to_user_id: 接收方用户 ID。
            context_token: 上游上下文 token。
            group_id: 群聊 ID。
            state: 当前 daemon 状态。
            stop_event: 停止信号。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            if not state.typing_ticket:
                state.typing_ticket = await self.client.get_typing_ticket(
                    ilink_user_id=to_user_id,
                    context_token=context_token,
                )
                await self._state_io.save(state)
            if not state.typing_ticket:
                return
            while not stop_event.is_set():
                try:
                    await self.client.send_typing(
                        ilink_user_id=to_user_id,
                        typing_ticket=state.typing_ticket,
                        status=1,
                    )
                except Exception as exc:
                    # typing 是辅助能力，失败后直接静默降级，不能阻塞主消息闭环。
                    Log.warning(f"发送 typing 失败，已降级为仅发最终答案: {exc}", module=MODULE)
                    return
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=self.config.typing_interval_sec)
                except asyncio.TimeoutError:
                    continue
        except Exception as exc:
            Log.warning(f"初始化 typing 失败: {exc}", module=MODULE)

    async def _reset_auth_state(self) -> None:
        """清除当前登录态。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        await self._state_io.clear_auth()
        self.client.update_auth(base_url=None, bot_token=None)

    def _open_login_url(self, login_url: str) -> None:
        """best-effort 打开二维码链接。

        Args:
            login_url: 登录二维码页面 URL。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            opened = webbrowser.open(login_url, new=2)
        except Exception as exc:
            Log.warning(f"无法自动打开二维码链接，请手动访问: {exc}", module=MODULE)
            return
        if opened:
            print("已尝试在默认浏览器中打开二维码页面，请直接用手机微信扫描。")


__all__ = [
    "WeChatDaemon",
    "WeChatDaemonConfig",
    "WeChatReply",
    "WeChatReplyBuilder",
]
