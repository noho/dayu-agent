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
from dayu.execution.options import ExecutionOptions
from dayu.log import Log
from dayu.services.contracts import ChatResumeRequest, ChatTurnRequest, PromptRequest, SessionResolutionPolicy
from dayu.services.pending_turns import has_resumable_pending_turn
from dayu.services.protocols import ChatServiceProtocol, PromptServiceProtocol

MODULE = "APP.INTERACTIVE"
_WIDE_EAST_ASIAN_WIDTHS = frozenset(("F", "W"))





class _SpinnerController:
    """命令行旋转指示器。"""

    def __init__(self, *, label: str = "Waiting") -> None:
        """初始化旋转指示器。

        Args:
            label: 旋转指示器前缀文本。

        Returns:
            无。

        Raises:
            无。
        """

        self._label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """启动旋转指示器。"""

        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """停止旋转指示器。"""

        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None

    def _spin(self) -> None:
        """循环输出旋转动画。"""

        frames = "|/-\\"
        idx = 0
        while not self._stop.is_set():
            print(f"\r{self._label} {frames[idx % len(frames)]}", end="", flush=True)
            time.sleep(0.1)
            idx += 1
        print("\r" + " " * (len(self._label) + 2) + "\r", end="", flush=True)


@dataclass
class _RenderState:
    """终端事件渲染状态。"""

    show_thinking: bool = False
    spinner: _SpinnerController | None = None
    content_streamed: bool = False
    reasoning_streamed: bool = False
    reasoning_line_open: bool = False
    line_open: bool = False
    final_content: str = ""
    filtered: bool = False


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


def _stop_spinner_if_needed(state: _RenderState) -> None:
    """在首次可见输出前停止 spinner。"""

    if state.spinner is None:
        return
    state.spinner.stop()
    state.spinner = None


def _start_spinner_if_needed(state: _RenderState, *, enabled: bool) -> None:
    """按需启动等待 spinner。

    Args:
        state: 渲染状态。
        enabled: 是否允许启动 spinner。

    Returns:
        无。

    Raises:
        无。
    """

    if not enabled or state.spinner is not None:
        return
    state.spinner = _SpinnerController()
    state.spinner.start()


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
    _stop_spinner_if_needed(state)
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
    _stop_spinner_if_needed(state)
    if not state.reasoning_streamed:
        print("Thinking...", file=sys.stderr, flush=True)
    print(text, end="", file=sys.stderr, flush=True)
    state.reasoning_streamed = True
    state.reasoning_line_open = not text.endswith("\n")


def _render_warning_or_error(state: _RenderState, message: str) -> None:
    """渲染告警或错误。"""

    _stop_spinner_if_needed(state)
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
        _render_content_delta(state, str(payload or ""))
        return

    if event_type == AppEventType.FINAL_ANSWER.value:
        state.final_content = str(payload.get("content", "")) if isinstance(payload, dict) else str(payload)
        state.filtered = bool(payload.get("filtered", False)) if isinstance(payload, dict) else False
        if state.final_content and not state.content_streamed:
            _render_content_delta(state, state.final_content)
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


def _format_cancelled_message(payload: Any) -> str:
    """将取消事件负载格式化为 CLI 提示。"""

    reason = extract_cancel_reason(payload)
    if reason:
        return f"[cancelled] 执行已取消: {reason}"
    return "[cancelled] 执行已取消"

async def _consume_chat_turn_stream(
    session: ChatServiceProtocol,
    user_input: str,
    state: _RenderState,
    *,
    session_id: str | None,
    scene_name: str = "interactive",
    ticker: str | None = None,
    execution_options: ExecutionOptions | None = None,
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
    async for event in submission.event_stream:
        _render_stream_event(event, state)
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
    _start_spinner_if_needed(state, enabled=not show_thinking)
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
        _stop_spinner_if_needed(state)
        _ensure_reasoning_newline(state)
        _ensure_newline(state)


async def _consume_prompt_stream(
    session: PromptServiceProtocol,
    user_input: str,
    state: _RenderState,
    *,
    ticker: str | None,
    execution_options: ExecutionOptions | None = None,
) -> str:
    """消费单次 prompt 的事件流并实时渲染。

    Args:
        session: Prompt 服务。
        user_input: 用户输入文本。
        state: 渲染状态。
        ticker: 股票代码。
        execution_options: 请求级执行覆盖参数。

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
    async for event in submission.event_stream:
        _render_stream_event(event, state)
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
    show_waiting_spinner: bool = False,
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
        show_waiting_spinner: 是否在首个可见输出前显示等待 spinner。

    Returns:
        `(最终答案文本, 本轮解析后的 session_id)`。

    Raises:
        ValueError: 输入为空时抛出。
        RuntimeError: Agent 创建失败时抛出。
    """

    state = _RenderState(show_thinking=show_thinking)
    _start_spinner_if_needed(state, enabled=show_waiting_spinner)
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
            )
        )
    finally:
        _stop_spinner_if_needed(state)
        _ensure_reasoning_newline(state)
        _ensure_newline(state)


def _run_prompt_stream(
    session: PromptServiceProtocol,
    user_input: str,
    *,
    ticker: str | None,
    execution_options: ExecutionOptions | None = None,
    show_thinking: bool = False,
    show_waiting_spinner: bool = False,
) -> str:
    """执行单次 prompt 的同步包装入口。

    Args:
        session: Prompt 服务。
        user_input: 用户输入文本。
        ticker: 股票代码。
        execution_options: 请求级执行覆盖参数。
        show_thinking: 是否回显 thinking 增量。
        show_waiting_spinner: 是否在首个可见输出前显示等待 spinner。

    Returns:
        最终答案文本。

    Raises:
        ValueError: 输入为空时抛出。
        RuntimeError: Agent 创建失败时抛出。
    """

    state = _RenderState(show_thinking=show_thinking)
    _start_spinner_if_needed(state, enabled=show_waiting_spinner)
    try:
        return asyncio.run(
            _consume_prompt_stream(
                session,
                user_input,
                state,
                ticker=ticker,
                execution_options=execution_options,
            )
        )
    finally:
        _stop_spinner_if_needed(state)
        _ensure_reasoning_newline(state)
        _ensure_newline(state)

def interactive(
    agent_session: ChatServiceProtocol,
    *,
    session_id: str | None = None,
    scene_name: str = "interactive",
    execution_options: ExecutionOptions | None = None,
    show_thinking: bool = False,
) -> None:
    """执行交互式多轮输入循环。

    Args:
        agent_session: 已装配的聊天会话服务。
        session_id: 可选初始会话 ID。
        scene_name: 本轮 turn 使用的 scene 名称。
        execution_options: 请求级执行覆盖参数。
        show_thinking: 是否回显 thinking 增量。

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
    _resume_interactive_pending_turn_if_needed(
        agent_session,
        session_id=session_id,
        scene_name=scene_name,
        show_thinking=show_thinking,
    )

    while True:
        try:
            user_input = session.prompt(">>> ")
        except EOFError:
            break
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
                show_waiting_spinner=not show_thinking,
            )
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
) -> int:
    """执行单次 prompt 命令。

    Args:
        prompt_service: 已装配的单轮 prompt 服务。
        user_input: 单次输入文本。
        ticker: 股票代码。
        execution_options: 请求级执行覆盖参数。
        show_thinking: 是否回显 thinking 增量。

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
            show_waiting_spinner=not show_thinking,
        )
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
            show_waiting_spinner=not show_thinking,
        )
        _print_label_hint_box(label)
    except ValueError as exc:
        Log.error(str(exc), module=MODULE)
        return 2
    except RuntimeError as exc:
        Log.error(f"{exc}，退出 prompt 模式", module=MODULE)
        return 2
    return 0
