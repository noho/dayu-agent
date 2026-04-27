"""交互式终端入口。

该模块负责：
- 交互式终端 UI
- 单次 prompt 终端 UI
- 消费 application 会话事件流并渲染到终端
"""

from __future__ import annotations

import asyncio
import threading
import time
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any
import sys

from dayu.contracts.events import AppEventType, extract_cancel_reason
from dayu.process_lifecycle import EXIT_CODE_SIGINT, RunLifecycleObserver
from dayu.text import strip_markdown_fence
from dayu.execution.options import ExecutionOptions
from dayu.log import Log
from dayu.services.contracts import ChatResumeRequest, ChatTurnRequest, PromptRequest, SessionResolutionPolicy
from dayu.services.pending_turns import has_resumable_pending_turn
from dayu.services.protocols import ChatServiceProtocol, PromptServiceProtocol

MODULE = "APP.INTERACTIVE"
_WIDE_EAST_ASIAN_WIDTHS = frozenset(("F", "W"))




class _StatusLineController:
    """动态状态行控制器。

    - 单行覆写（\\r + ANSI 清行），不向上滚动
    - braille 点阵动画（⠋⠙⠹⠸⠼⠴⠦⠧）
    - 支持 update() 更新文本、pause() 暂停显示、stop() 永久停止
    - 可恢复：pause() 后 update() 可重新激活
    - 超时提示：工具执行超 5 秒时追加 (已等待 Xs)

    线程模型：
    - 动画线程只负责写帧，不做任何清行操作
    - pause() / stop() 先 join 线程（等待最后一帧写完），再由主线程清行
    - join 完成后 stdout 完全归主线程，正文输出与清行严格串行，无竞争
    - asyncio 事件消费在主线程，join 阻塞主线程期间事件流同样暂停，
      不会有并发的 update() 调用，join 期间不存在重启线程的竞争
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧"
    _FRAME_INTERVAL = 0.08
    _TIMEOUT_THRESHOLD = 5.0
    _MIN_DISPLAY_SEC = 1.0

    def __init__(self) -> None:
        self._label: str = "思考中..."
        self._pending_label: str | None = None
        self._lock = threading.Lock()
        self._active: bool = False
        self._stopped: bool = False
        self._thread: threading.Thread | None = None
        self._last_update_time: float = 0.0
        self._label_set_time: float = 0.0

    def update(self, text: str) -> None:
        """更新状态行文本并启动动画（如果尚未运行）。

        如果上一次文本设置不足 _MIN_DISPLAY_SEC，新文本缓存为 pending，
        由动画线程在最小停留时间后自动切换，避免快速工具状态行闪烁。
        """

        with self._lock:
            if self._stopped:
                return
            now = time.monotonic()
            if self._active and (now - self._label_set_time) < self._MIN_DISPLAY_SEC:
                self._pending_label = text
            else:
                self._label = text
                self._pending_label = None
                self._label_set_time = now
                self._last_update_time = now
            if not self._active:
                self._active = True
                self._label_set_time = now
                self._last_update_time = now
                self._thread = threading.Thread(target=self._spin, daemon=True)
                self._thread.start()

    def pause(self) -> None:
        """暂停动画并清行。join 后主线程独占 stdout，与正文输出严格串行。"""

        with self._lock:
            if self._stopped or not self._active:
                return
            self._active = False
            self._pending_label = None
            thread = self._thread
            self._thread = None

        # 等线程完全退出，此后 stdout 无竞争
        if thread is not None:
            thread.join()

        # 主线程清行，后续正文 print 直接跟上
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def stop(self) -> None:
        """永久停止状态行（turn 结束时调用）。"""

        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            was_active = self._active
            self._active = False
            self._pending_label = None
            thread = self._thread
            self._thread = None

        if thread is not None:
            thread.join()

        # 只有动画还在显示时才需要清行；内容已输出后不能清，否则会抹掉最后一行
        if was_active:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def _spin(self) -> None:
        """动画循环。退出时清行，确保 stdout 干净交还主线程。"""

        idx = 0
        while True:
            with self._lock:
                if not self._active or self._stopped:
                    break
                now = time.monotonic()
                if self._pending_label is not None and (now - self._label_set_time) >= self._MIN_DISPLAY_SEC:
                    self._label = self._pending_label
                    self._pending_label = None
                    self._label_set_time = now
                    self._last_update_time = now
                label = self._label
                elapsed = now - self._last_update_time

            display = f"{label} (已等待 {int(elapsed)}s)" if elapsed > self._TIMEOUT_THRESHOLD else label
            frame = self._FRAMES[idx % len(self._FRAMES)]
            # \r 回行首，\033[K 清到行尾，再写新内容。
            # write/flush 在 lock 外执行；安全性由 pause()/stop() 先 join 线程
            # 再操作 stdout 来保证——动画运行期间主线程不会写 stdout。
            sys.stdout.write(f"\r\033[K{frame} {display}")
            sys.stdout.flush()
            time.sleep(self._FRAME_INTERVAL)
            idx += 1


@dataclass
class _RenderState:
    """终端事件渲染状态。"""

    show_thinking: bool = False
    status_line: _StatusLineController | None = None
    content_streamed: bool = False
    reasoning_streamed: bool = False
    reasoning_line_open: bool = False
    line_open: bool = False
    final_content: str = ""
    filtered: bool = False
    tool_calls_seen: int = 0
    _pending_content_delta: str | None = None


def _measure_display_width(text: str) -> int:
    """计算文本在等宽终端中的显示宽度。

    Args:
        text: 待计算文本。

    Returns:
        终端显示宽度；全角/宽字符按 2 列计算，其余字符按 1 列计算。

    Raises:
        无。
    """

    return sum(2 if unicodedata.east_asian_width(char) in _WIDE_EAST_ASIAN_WIDTHS else 1 for char in text)


def _print_label_hint_box(label: str) -> None:
    """在 prompt 输出末尾打印可恢复标签提示框。

    Args:
        label: 当前 conversation label。

    Returns:
        无。

    Raises:
        无。
    """

    line = f"标签: {label}"
    line_width = _measure_display_width(line)
    content_width = line_width + 2
    trailing_padding_width = content_width - 1 - line_width
    top_bottom = f"+{'-' * content_width}+"
    middle = f"| {line}{' ' * trailing_padding_width}|"
    print(top_bottom)
    print(middle)
    print(top_bottom)



def _ensure_newline(state: _RenderState) -> None:
    """在当前行为内容流时补一个换行。"""

    if not state.line_open:
        return
    print()
    state.line_open = False


def _ensure_reasoning_newline(state: _RenderState) -> None:
    """在当前行为 reasoning 流时补一个换行。

    Args:
        state: 渲染状态。

    Returns:
        无。

    Raises:
        无。
    """

    if not state.reasoning_line_open:
        return
    print(file=sys.stderr, flush=True)
    state.reasoning_line_open = False


def _render_content_delta(state: _RenderState, text: str) -> None:
    """渲染内容增量事件。"""

    if not text:
        return
    # 过滤纯空白 delta：
    # - 尚未输出实质内容时：完全跳过，避免工具调用前的 \n 产生空行
    # - 已输出内容但当前行未开（上一个 delta 已换行）：跳过，避免工具循环间的 \n\n 叠加产生大段空行
    if not text.strip():
        if not state.content_streamed:
            return
        if not state.line_open:
            return
    if state.status_line is not None:
        state.status_line.pause()
    if state.reasoning_streamed and not state.content_streamed:
        _ensure_reasoning_newline(state)
        print(flush=True)  # reasoning 和 content 之间加空行
    print(text, end="", flush=True)
    state.content_streamed = True
    state.line_open = not text.endswith("\n")


def _render_reasoning_delta(state: _RenderState, text: str) -> None:
    """渲染 reasoning 增量事件。"""

    if not state.show_thinking:
        return
    if not text:
        return
    if state.status_line is not None:
        state.status_line.pause()
    if not state.reasoning_streamed:
        print("Thinking...", file=sys.stderr, flush=True)
    print(text, end="", file=sys.stderr, flush=True)
    state.reasoning_streamed = True
    state.reasoning_line_open = not text.endswith("\n")


def _render_warning_or_error(state: _RenderState, message: str) -> None:
    """渲染告警或错误。"""

    if state.status_line is not None:
        state.status_line.pause()
    _ensure_reasoning_newline(state)
    _ensure_newline(state)
    print(message, file=sys.stderr, flush=True)


def _get_event_type_token(event: Any) -> str:
    """提取事件类型 token。"""

    event_type = getattr(event, "type", "")
    if isinstance(event_type, Enum):
        return str(event_type.value)
    return str(event_type)


def _get_event_payload(event: object) -> object | None:
    """提取事件负载（兼容 AppEvent/StreamEvent）。

    Args:
        event: 事件对象。

    Returns:
        事件负载；若不存在则返回 ``None``。

    Raises:
        无。
    """

    payload = getattr(event, "payload", None)
    if payload is not None:
        return payload
    return getattr(event, "data", None)


def _handle_content_delta(state: _RenderState, text: str) -> None:
    """处理内容增量：首个 delta 暂存检测围栏，围栏模式下持续缓冲。

    首个内容 delta 若以 ````` `` 开头，进入缓冲模式：所有后续 delta
    累积到 ``_pending_content_delta``，不在终端渲染，等待
    ``FINAL_ANSWER`` 确认是否为围栏。若确认是围栏，丢弃缓冲并渲染
    剥离后的内容；若非围栏，刷出全部缓冲内容。

    Args:
        state: 渲染状态。
        text: 本次增量文本。

    Raises:
        无。
    """

    if not text:
        return
    if state._pending_content_delta is not None:
        # 缓冲模式：累积 delta，不渲染
        state._pending_content_delta += text
        return
    if not state.content_streamed and text.lstrip().startswith("```"):
        # 首个内容 delta 且疑似围栏头，进入缓冲模式
        state._pending_content_delta = text
        return
    _render_content_delta(state, text)


def _render_stream_event(event: Any, state: _RenderState) -> None:
    """将单个事件渲染到终端。

    Args:
        event: 流式事件。
        state: 渲染状态。

    Raises:
        无。
    """

    event_type = _get_event_type_token(event)
    payload = _get_event_payload(event)

    if event_type == AppEventType.CONTENT_DELTA.value:
        _handle_content_delta(state, str(payload or ""))
        return

    if event_type == AppEventType.FINAL_ANSWER.value:
        raw_content = str(payload.get("content", "")) if isinstance(payload, dict) else str(payload)
        stripped = strip_markdown_fence(raw_content)
        state.final_content = stripped
        state.filtered = bool(payload.get("filtered", False)) if isinstance(payload, dict) else False
        if state._pending_content_delta is not None:
            # 有暂存 delta（疑似围栏模式），根据完整内容判断
            if raw_content.startswith("```"):
                # 确认是围栏：丢弃缓冲，渲染剥离后的正文
                state._pending_content_delta = None
                state.content_streamed = False
                if stripped:
                    _render_content_delta(state, stripped)
            else:
                # 非围栏：刷出缓冲（已是完整内容），无需再渲染
                _render_content_delta(state, state._pending_content_delta)
                state._pending_content_delta = None
        elif stripped and not state.content_streamed:
            _render_content_delta(state, stripped)
        if state.filtered:
            _render_warning_or_error(state, "[filtered] 本轮输出触发内容过滤，结果可能不完整")
        return

    if event_type == AppEventType.REASONING_DELTA.value:
        _render_reasoning_delta(state, str(payload or ""))
        return

    if event_type == AppEventType.WARNING.value:
        message = payload.get("message", "") if isinstance(payload, dict) else str(payload)
        _render_warning_or_error(state, f"[warning] {message}")
        return

    if event_type == AppEventType.ERROR.value:
        if isinstance(payload, dict):
            message = str(payload.get("message", payload))
        else:
            message = str(payload)
        _render_warning_or_error(state, f"[error] {message}")
        return

    if event_type == AppEventType.CANCELLED.value:
        _render_warning_or_error(state, _format_cancelled_message(payload))
        return

    if event_type == AppEventType.TOOL_EVENT.value:
        if state.status_line is not None and isinstance(payload, dict):
            summary = _format_tool_event_summary(payload, state)
            if summary:
                # reasoning / content 未换行时先推到新行，避免动画 \r\033[K 清掉已输出文字
                _ensure_reasoning_newline(state)
                if state.line_open:
                    print(flush=True)
                    state.line_open = False
                state.status_line.update(summary)
        return

    if event_type == AppEventType.ITERATION_START.value:
        if state.status_line is not None and isinstance(payload, dict):
            iteration = int(payload.get("iteration", 1))
            if iteration >= 2 and state.tool_calls_seen > 0:
                _ensure_reasoning_newline(state)
                if state.line_open:
                    print(flush=True)
                    state.line_open = False
                state.status_line.update(f"思考中... [已调用 {state.tool_calls_seen} 次工具]")
        return


def _format_tool_event_summary(payload: dict[str, Any], state: _RenderState) -> str | None:
    """将 TOOL_EVENT payload 格式化为状态行文本。

    从 engine 层事件自带的 display_name / param_preview 读取展示信息。
    """

    sub_type = payload.get("engine_event_type", "")
    data = payload.get("data") or {}

    if sub_type == "tool_call_dispatched" and isinstance(data, dict):
        display = str(data.get("display_name") or data.get("name", ""))
        param_preview = str(data.get("param_preview", ""))
        state.tool_calls_seen += 1
        if param_preview:
            return f"{display} — {param_preview}"
        return display

    return None


def _format_cancelled_message(payload: Any) -> str:
    """将取消事件负载格式化为 CLI 提示。"""

    reason = extract_cancel_reason(payload)
    if reason:
        return f"[cancelled] 执行已取消: {reason}"
    return "[cancelled] 执行已取消"


def _extract_run_id_from_event(event: object) -> str | None:
    """从事件对象中提取稳定的 ``run_id``。

    Args:
        event: 事件流中产生的事件对象，期望其 ``meta`` 字段为 ``dict``，
            且包含字符串 ``run_id`` 键。

    Returns:
        非空字符串形式的 ``run_id``；事件未携带合法 ``run_id`` 时返回 ``None``。

    Raises:
        无。
    """

    meta = getattr(event, "meta", None)
    if isinstance(meta, dict):
        candidate = meta.get("run_id")
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


class _RunIdLifecycleTracker:
    """在事件流期间登记/清理当前 run 的小型生命周期跟踪器。

    职责：把 `_consume_chat_turn_stream` 与 `_consume_prompt_stream` 中
    重复的「首帧 ``meta["run_id"]`` 登记 → finally clear」样板抽到一处，
    避免在两个事件消费器内重复维护协调器交互逻辑。
    """

    def __init__(self, observer: RunLifecycleObserver | None) -> None:
        """初始化跟踪器。

        Args:
            observer: 可选的进程级 run 生命周期观察者；为 ``None`` 时
                所有跟踪操作都是 no-op。

        Returns:
            无。

        Raises:
            无。
        """

        self._observer = observer
        self._registered_run_id: str | None = None

    def observe(self, event: object) -> None:
        """检查事件并在拿到首个 ``run_id`` 时登记到观察者。

        Args:
            event: 事件流中产生的事件对象。

        Returns:
            无。

        Raises:
            无。
        """

        if self._observer is None or self._registered_run_id is not None:
            return
        run_id = _extract_run_id_from_event(event)
        if run_id is None:
            return
        self._observer.register_active_run(run_id)
        self._registered_run_id = run_id

    def clear(self) -> None:
        """事件流结束后从观察者清理已登记的 run。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        if self._observer is None or self._registered_run_id is None:
            return
        self._observer.clear_active_run(self._registered_run_id)
        self._registered_run_id = None

async def _consume_chat_turn_stream(
    session: ChatServiceProtocol,
    user_input: str,
    state: _RenderState,
    *,
    session_id: str | None,
    scene_name: str = "interactive",
    ticker: str | None = None,
    execution_options: ExecutionOptions | None = None,
    run_lifecycle_observer: RunLifecycleObserver | None = None,
) -> tuple[str, str]:
    """消费单轮 chat 事件流并实时渲染。

    Args:
        session: 聊天会话服务。
        user_input: 用户输入文本。
        state: 渲染状态。
        session_id: 会话 ID；首轮可为空。
        scene_name: 本轮执行使用的 scene 名称。
        ticker: 股票代码。
        execution_options: 请求级执行覆盖参数。
        run_lifecycle_observer: 可选的进程级 run 生命周期观察者，
            用于在拿到 ``meta["run_id"]`` 后登记当前 run，配合 Ctrl-C
            协作式取消。

    Returns:
        `(最终答案文本, 本轮解析后的 session_id)`。

    Raises:
        ValueError: 输入为空时抛出。
        RuntimeError: Agent 创建失败时抛出。
    """

    request = ChatTurnRequest(
        session_id=session_id,
        user_text=user_input,
        ticker=ticker,
        execution_options=execution_options,
        scene_name=scene_name,
        session_resolution_policy=SessionResolutionPolicy.ENSURE_DETERMINISTIC,
    )
    submission = await session.submit_turn(request)
    tracker = _RunIdLifecycleTracker(run_lifecycle_observer)
    try:
        async for event in submission.event_stream:
            tracker.observe(event)
            _render_stream_event(event, state)
    finally:
        tracker.clear()
    return state.final_content, submission.session_id

def _resume_interactive_pending_turn_if_needed(
    session: ChatServiceProtocol,
    *,
    session_id: str | None,
    scene_name: str = "interactive",
    show_thinking: bool,
) -> None:
    """在进入 REPL 前恢复当前 interactive session 的 pending turn。

    Args:
        session: interactive 使用的 ChatService 协议实现。
        session_id: 当前 interactive 绑定的 Host session ID；为空时直接跳过恢复。
        scene_name: 本次 interactive 会话对应的 scene 名称，默认使用 `interactive`。
        show_thinking: 是否展示 thinking 流。

    Returns:
        无。

    Raises:
        Exception: 当 pending turn 仍然存在且恢复失败时，继续向上抛出原始异常。
    """

    if session_id is None:
        return
    pending_turns = session.list_resumable_pending_turns(
        session_id=session_id,
        scene_name=scene_name,
    )
    if not pending_turns:
        return
    pending_turn = pending_turns[0]
    state = _RenderState(show_thinking=show_thinking)
    status_line = _StatusLineController()
    status_line.update("思考中...")
    state.status_line = status_line
    try:
        async def _resume_and_consume() -> str:
            submission = await session.resume_pending_turn(
                ChatResumeRequest(
                    session_id=session_id,
                    pending_turn_id=pending_turn.pending_turn_id,
                )
            )
            async for event in submission.event_stream:
                _render_stream_event(event, state)
            return submission.session_id

        try:
            asyncio.run(_resume_and_consume())
        except Exception as exc:
            if not has_resumable_pending_turn(
                session,
                session_id=pending_turn.session_id,
                scene_name="interactive",
                pending_turn_id=pending_turn.pending_turn_id,
            ):
                Log.warning(
                    "interactive pending turn 恢复失败，但记录已被 Host 清理，继续进入会话"
                    f" session_id={pending_turn.session_id}"
                    f" pending_turn_id={pending_turn.pending_turn_id}"
                    f" error={exc}",
                    module=MODULE,
                )
                _render_warning_or_error(
                    state,
                    "[warning] 上一轮 pending turn 恢复失败，但记录已被清理；当前会话继续可用",
                )
                return
            raise
    finally:
        status_line.stop()
        state.status_line = None
        _ensure_reasoning_newline(state)
        _ensure_newline(state)


async def _consume_prompt_stream(
    session: PromptServiceProtocol,
    user_input: str,
    state: _RenderState,
    *,
    ticker: str | None,
    execution_options: ExecutionOptions | None = None,
    run_lifecycle_observer: RunLifecycleObserver | None = None,
) -> str:
    """消费单次 prompt 的事件流并实时渲染。

    Args:
        session: Prompt 服务。
        user_input: 用户输入文本。
        state: 渲染状态。
        ticker: 股票代码。
        execution_options: 请求级执行覆盖参数。
        run_lifecycle_observer: 可选的 run 生命周期观察者；当事件流首帧
            携带 ``meta["run_id"]`` 时登记到协调器，让 Ctrl-C 能驱动协作式取消。

    Returns:
        最终答案文本。

    Raises:
        ValueError: 输入为空时抛出。
        RuntimeError: Agent 创建失败时抛出。
    """

    request = PromptRequest(
        user_text=user_input,
        ticker=ticker,
        execution_options=execution_options,
    )
    submission = await session.submit(request)
    tracker = _RunIdLifecycleTracker(run_lifecycle_observer)
    try:
        async for event in submission.event_stream:
            tracker.observe(event)
            _render_stream_event(event, state)
    finally:
        tracker.clear()
    return state.final_content


def _run_chat_turn_stream(
    session: ChatServiceProtocol,
    user_input: str,
    *,
    session_id: str | None,
    scene_name: str = "interactive",
    ticker: str | None = None,
    execution_options: ExecutionOptions | None = None,
    show_thinking: bool = False,
    run_lifecycle_observer: RunLifecycleObserver | None = None,
) -> tuple[str, str]:
    """执行单轮 chat 的同步包装入口。

    Args:
        session: 聊天会话服务。
        user_input: 用户输入文本。
        session_id: 会话 ID；首轮可为空。
        scene_name: 本轮执行使用的 scene 名称。
        ticker: 股票代码。
        execution_options: 请求级执行覆盖参数。
        show_thinking: 是否回显 thinking 增量。
        run_lifecycle_observer: 可选的 run 生命周期观察者，用于配合
            进程级协调器在 Ctrl-C 时触发协作式取消。

    Returns:
        `(最终答案文本, 本轮解析后的 session_id)`。

    Raises:
        ValueError: 输入为空时抛出。
        RuntimeError: Agent 创建失败时抛出。
    """

    state = _RenderState(show_thinking=show_thinking)
    status_line = _StatusLineController()
    status_line.update("思考中...")
    state.status_line = status_line
    try:
        return asyncio.run(
            _consume_chat_turn_stream(
                session,
                user_input,
                state,
                session_id=session_id,
                scene_name=scene_name,
                ticker=ticker,
                execution_options=execution_options,
                run_lifecycle_observer=run_lifecycle_observer,
            )
        )
    finally:
        status_line.stop()
        state.status_line = None
        _ensure_reasoning_newline(state)
        _ensure_newline(state)


def _run_prompt_stream(
    session: PromptServiceProtocol,
    user_input: str,
    *,
    ticker: str | None,
    execution_options: ExecutionOptions | None = None,
    show_thinking: bool = False,
    run_lifecycle_observer: RunLifecycleObserver | None = None,
) -> str:
    """执行单次 prompt 的同步包装入口。

    Args:
        session: Prompt 服务。
        user_input: 用户输入文本。
        ticker: 股票代码。
        execution_options: 请求级执行覆盖参数。
        show_thinking: 是否回显 thinking 增量。
        run_lifecycle_observer: 可选的 run 生命周期观察者；事件流首帧
            带 ``meta["run_id"]`` 时登记到协调器，让 Ctrl-C 走协作式取消。

    Returns:
        最终答案文本。

    Raises:
        ValueError: 输入为空时抛出。
        RuntimeError: Agent 创建失败时抛出。
    """

    state = _RenderState(show_thinking=show_thinking)
    status_line = _StatusLineController()
    status_line.update("思考中...")
    state.status_line = status_line
    try:
        return asyncio.run(
            _consume_prompt_stream(
                session,
                user_input,
                state,
                ticker=ticker,
                execution_options=execution_options,
                run_lifecycle_observer=run_lifecycle_observer,
            )
        )
    finally:
        status_line.stop()
        state.status_line = None
        _ensure_reasoning_newline(state)
        _ensure_newline(state)

def interactive(
    agent_session: ChatServiceProtocol,
    *,
    session_id: str | None = None,
    scene_name: str = "interactive",
    execution_options: ExecutionOptions | None = None,
    show_thinking: bool = False,
    run_lifecycle_observer: RunLifecycleObserver | None = None,
) -> None:
    """执行交互式多轮输入循环。

    Args:
        agent_session: 已装配的聊天会话服务。
        session_id: 可选初始会话 ID。
        scene_name: 本轮 turn 使用的 scene 名称。
        execution_options: 请求级执行覆盖参数。
        show_thinking: 是否回显 thinking 增量。
        run_lifecycle_observer: 可选的进程级 run 生命周期观察者，
            用于让 Ctrl-C 在事件流中触发 cooperative cancel。

    Returns:
        无。

    Raises:
        无。
    """

    if not sys.stdin.isatty():
        Log.error("交互模式需要 TTY 输入，请在终端中运行。", module=MODULE)
        return

    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
    except Exception:
        Log.error("prompt_toolkit 未安装，无法进入交互模式", module=MODULE)
        return

    kb = KeyBindings()

    @kb.add("enter")
    def _insert_newline(event) -> None:
        event.app.current_buffer.insert_text("\n")

    @kb.add("c-d")
    def _accept_or_eof(event) -> None:
        buffer = event.app.current_buffer
        if buffer.text:
            event.app.exit(result=buffer.text)
        else:
            event.app.exit(result=None)

    session = PromptSession(multiline=True, key_bindings=kb)
    consecutive_eof = 0
    try:
        _resume_interactive_pending_turn_if_needed(
            agent_session,
            session_id=session_id,
            scene_name=scene_name,
            show_thinking=show_thinking,
        )
    except KeyboardInterrupt:
        print("\n[interrupted]")

    while True:
        try:
            user_input = session.prompt(">>> ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            continue
        if user_input is None:
            consecutive_eof += 1
            if consecutive_eof >= 2:
                break
            continue

        user_input = user_input.strip()
        if not user_input:
            consecutive_eof = 0
            continue

        consecutive_eof = 0

        try:
            _final_content, session_id = _run_chat_turn_stream(
                agent_session,
                user_input,
                session_id=session_id,
                scene_name=scene_name,
                execution_options=execution_options,
                show_thinking=show_thinking,
                run_lifecycle_observer=run_lifecycle_observer,
            )
        except KeyboardInterrupt:
            # 信号路径已通过 ProcessShutdownCoordinator 触发协作式取消，
            # 这里只负责把控制权交还 REPL；事件流中的 CANCELLED 事件已由
            # _consume_chat_turn_stream 处理，run_id 也已 clear。
            print("\n[interrupted]")
            continue
        except ValueError as exc:
            Log.error(str(exc), module=MODULE)
            continue
        except RuntimeError as exc:
            Log.error(f"{exc}，跳过当前轮次", module=MODULE)
            continue


def prompt(
    prompt_service: PromptServiceProtocol,
    user_input: str,
    *,
    ticker: str | None = None,
    execution_options: ExecutionOptions | None = None,
    show_thinking: bool = False,
    run_lifecycle_observer: RunLifecycleObserver | None = None,
) -> int:
    """执行单次 prompt 命令。

    Args:
        prompt_service: 已装配的单轮 prompt 服务。
        user_input: 单次输入文本。
        ticker: 股票代码。
        execution_options: 请求级执行覆盖参数。
        show_thinking: 是否回显 thinking 增量。
        run_lifecycle_observer: 可选的 run 生命周期观察者；用于让 Ctrl-C
            通过事件 ``meta["run_id"]`` 触发协作式取消，与 chat 路径一致。

    Returns:
        退出码，``0`` 表示成功，``2`` 表示失败。

    Raises:
        无。
    """

    try:
        _run_prompt_stream(
            prompt_service,
            user_input,
            ticker=ticker,
            execution_options=execution_options,
            show_thinking=show_thinking,
            run_lifecycle_observer=run_lifecycle_observer,
        )
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return EXIT_CODE_SIGINT
    except ValueError as exc:
        Log.error(str(exc), module=MODULE)
        return 2
    except RuntimeError as exc:
        Log.error(f"{exc}，退出 prompt 模式", module=MODULE)
        return 2
    return 0


def conversation_prompt(
    chat_service: ChatServiceProtocol,
    user_input: str,
    *,
    label: str,
    session_id: str,
    scene_name: str,
    ticker: str | None = None,
    execution_options: ExecutionOptions | None = None,
    show_thinking: bool = False,
    run_lifecycle_observer: RunLifecycleObserver | None = None,
) -> int:
    """执行单轮 conversation prompt 命令。

    Args:
        chat_service: 已装配的聊天服务。
        user_input: 单次输入文本。
        label: 当前可恢复对话标签。
        session_id: label registry 解析得到的确定性会话 ID。
        scene_name: 本轮 turn 使用的 scene 名称。
        ticker: 股票代码。
        execution_options: 请求级执行覆盖参数。
        show_thinking: 是否回显 thinking 增量。
        run_lifecycle_observer: 可选的进程级 run 生命周期观察者，
            用于让 Ctrl-C 触发 cooperative cancel。

    Returns:
        退出码，``0`` 表示成功，``2`` 表示失败。

    Raises:
        无。
    """

    try:
        _run_chat_turn_stream(
            chat_service,
            user_input,
            session_id=session_id,
            scene_name=scene_name,
            ticker=ticker,
            execution_options=execution_options,
            show_thinking=show_thinking,
            run_lifecycle_observer=run_lifecycle_observer,
        )
        _print_label_hint_box(label)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return EXIT_CODE_SIGINT
    except ValueError as exc:
        Log.error(str(exc), module=MODULE)
        return 2
    except RuntimeError as exc:
        Log.error(f"{exc}，退出 prompt 模式", module=MODULE)
        return 2
    return 0
