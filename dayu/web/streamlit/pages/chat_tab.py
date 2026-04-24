"""Streamlit 交互式分析 Tab。"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import time
import streamlit as st

from dayu.log import Log
from dayu.services.protocols import ChatServiceProtocol
from dayu.web.streamlit.components.sidebar import WatchlistItem
from dayu.web.streamlit.pages.chat.chat_client import ChatServiceClient, create_chat_service_client
from dayu.web.streamlit.pages.chat.chat_stream_bridge import (
    StreamQueueItem,
    build_request_trace_id,
    present_stream_side_effects,
    summarize_user_text,
    sync_stream_via_asyncio,
)
from dayu.web.streamlit.stream_chat_events import normalize_stream_text_for_markdown

MODULE = "dayu.web.streamlit.pages.chat_tab"
_WELCOME_MARKDOWN = "大禹 Agent 将基于当前股票的财报及相关材料进行交互式分析。"
_INPUT_PLACEHOLDER = "例如：公司的核心竞争力是什么？增长的主要驱动因素有哪些？"
_INPUT_LABEL = "输入你的分析问题"
_EMPTY_INPUT_WARNING = "请输入问题后再提交。"
_MISSING_SERVICE_WARNING = "交互式分析服务未就绪，请检查服务初始化状态。"
_EMPTY_ASSISTANT_REPLY_WARNING = "本轮未收到可展示的回复，请稍后重试或检查模型与网络配置。"
_THINKING_EXPANDER_TITLE = "思考内容"
_USER_MESSAGE_COLUMN_SPEC: list[int] = [1, 3]
_ASSISTANT_MESSAGE_COLUMN_SPEC: list[int] = [4, 1]
_STREAM_RENDER_MIN_INTERVAL_SECONDS = 0.08
_STREAM_RENDER_MIN_CHAR_DELTA = 24
_StreamQueueItem = StreamQueueItem


@dataclass(frozen=True)
class _ChatMessage:
    """聊天消息视图模型。"""

    role: str
    content: str
    reasoning_content: str = ""


def _build_state_key(ticker: str, suffix: str) -> str:
    """构建按股票代码隔离的会话状态键。"""

    return f"chat_tab_{ticker}_{suffix}"


def _apply_pending_input_reset(*, input_key: str, clear_input_key: str) -> None:
    """在输入控件实例化前应用延迟清空请求。"""

    raw_pending = st.session_state.get(clear_input_key)
    if isinstance(raw_pending, bool) and raw_pending:
        st.session_state[input_key] = ""
        st.session_state[clear_input_key] = False
        Log.info(f"应用延迟输入清空: input_key={input_key}", module=MODULE)


def _ensure_messages(state_key: str) -> list[_ChatMessage]:
    """确保会话消息列表存在。"""

    if state_key not in st.session_state:
        st.session_state[state_key] = []
    raw_messages = st.session_state[state_key]
    if isinstance(raw_messages, list):
        return raw_messages
    st.session_state[state_key] = []
    reset_messages = st.session_state[state_key]
    if isinstance(reset_messages, list):
        return reset_messages
    return []


def _render_message_history(messages: list[_ChatMessage]) -> None:
    """渲染历史消息。"""

    for message in messages:
        if message.role == "user":
            _user_spacer_column, user_column = st.columns(_USER_MESSAGE_COLUMN_SPEC, gap="small")
            target_column = user_column
        else:
            assistant_column, _assistant_spacer_column = st.columns(_ASSISTANT_MESSAGE_COLUMN_SPEC, gap="small")
            target_column = assistant_column
        with target_column:
            with st.chat_message(message.role):
                if message.role == "assistant" and message.reasoning_content.strip():
                    with st.expander(_THINKING_EXPANDER_TITLE, expanded=True):
                        st.markdown(message.reasoning_content)
                if message.role == "assistant":
                    st.markdown(message.content)
                else:
                    st.markdown(message.content)


def _sync_stream_via_asyncio(
    chat_service: ChatServiceProtocol,
    *,
    user_text: str,
    session_id: str | None,
    ticker: str,
    session_id_out: list[str],
    sides_out: list[str],
    filtered_holder: list[bool],
    err_out: list[BaseException],
    trace_id: str,
) -> Iterator[StreamQueueItem]:
    """兼容旧调用形态：返回迭代器。"""

    return iter(
        sync_stream_via_asyncio(
            chat_service,
            user_text=user_text,
            session_id=session_id,
            ticker=ticker,
            session_id_out=session_id_out,
            sides_out=sides_out,
            filtered_holder=filtered_holder,
            err_out=err_out,
            trace_id=trace_id,
        )
    )


def _render_stream_markdown(chunks: Iterator[StreamQueueItem]) -> tuple[str, str]:
    """按 Markdown 语义分区流式渲染思考与正文。

    参数:
        chunks: 后端产出的流式分片迭代器。

    返回值:
        二元组 ``(reasoning_text, answer_text)``，分别表示思考与正文完整文本。

    异常:
        无。
    """

    with st.expander(_THINKING_EXPANDER_TITLE, expanded=True):
        reasoning_placeholder = st.empty()
    answer_placeholder = st.empty()
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    rendered_reasoning_length = 0
    rendered_answer_length = 0
    last_render_timestamp = 0.0

    def _should_flush_stream(*, accumulated_text: str, rendered_length: int, current_timestamp: float) -> bool:
        """判断是否需要刷新 Markdown 占位符。

        参数:
            accumulated_text: 当前完整累计文本。
            rendered_length: 上一次已渲染的文本长度。
            current_timestamp: 当前时间戳（秒）。

        返回值:
            满足最小时间间隔或结构边界条件时返回 ``True``。

        异常:
            无。
        """

        delta_length = len(accumulated_text) - rendered_length
        if delta_length <= 0:
            return False
        has_markdown_boundary = (
            ("\n\n" in accumulated_text[rendered_length:])
            or ("\n- " in accumulated_text[rendered_length:])
            or ("\n* " in accumulated_text[rendered_length:])
            or ("\n# " in accumulated_text[rendered_length:])
            or ("```" in accumulated_text[rendered_length:])
        )
        interval_reached = (current_timestamp - last_render_timestamp) >= _STREAM_RENDER_MIN_INTERVAL_SECONDS
        return (
            delta_length >= _STREAM_RENDER_MIN_CHAR_DELTA
            or has_markdown_boundary
            or interval_reached
        )

    for item in chunks:
        current_timestamp = time.perf_counter()
        if item.kind == "reasoning":
            reasoning_parts.append(item.chunk)
            reasoning_text = "".join(reasoning_parts)
            rendered_reasoning_text = normalize_stream_text_for_markdown(reasoning_text)
            if _should_flush_stream(
                accumulated_text=reasoning_text,
                rendered_length=rendered_reasoning_length,
                current_timestamp=current_timestamp,
            ):
                reasoning_placeholder.markdown(rendered_reasoning_text)
                rendered_reasoning_length = len(reasoning_text)
                last_render_timestamp = current_timestamp
            continue
        answer_parts.append(item.chunk)
        answer_text = "".join(answer_parts)
        rendered_answer_text = normalize_stream_text_for_markdown(answer_text)
        if _should_flush_stream(
            accumulated_text=answer_text,
            rendered_length=rendered_answer_length,
            current_timestamp=current_timestamp,
        ):
            answer_placeholder.markdown(rendered_answer_text)
            rendered_answer_length = len(answer_text)
            last_render_timestamp = current_timestamp
    final_reasoning_text = normalize_stream_text_for_markdown("".join(reasoning_parts))
    final_answer_text = normalize_stream_text_for_markdown("".join(answer_parts))
    if len(final_reasoning_text) != rendered_reasoning_length:
        reasoning_placeholder.markdown(final_reasoning_text)
    if len(final_answer_text) != rendered_answer_length:
        answer_placeholder.markdown(final_answer_text)
    return (
        final_reasoning_text,
        final_answer_text,
    )


def _stream_and_collect_assistant_reply(
    *,
    service_client: ChatServiceClient,
    user_text: str,
    ticker: str,
    session_id: str | None,
    session_id_key: str,
    trace_id: str,
) -> tuple[str, str, list[str], list[bool]]:
    """执行一次聊天流式请求并收集助手输出。"""

    next_session_id: list[str] = []
    side_messages: list[str] = []
    filtered_flags: list[bool] = []
    stream_errors: list[BaseException] = []
    chunks = _sync_stream_via_asyncio(
        service_client.chat_service,
        user_text=user_text,
        session_id=session_id,
        ticker=ticker,
        session_id_out=next_session_id,
        sides_out=side_messages,
        filtered_holder=filtered_flags,
        err_out=stream_errors,
        trace_id=trace_id,
    )

    assistant_reasoning_text, assistant_answer_text = _render_stream_markdown(chunks)
    if next_session_id:
        st.session_state[session_id_key] = next_session_id[-1]
    Log.info(
        f"[{trace_id}] 前端 markdown 分区流式渲染完成: answer_len={len(assistant_answer_text)}, "
        f"reasoning_len={len(assistant_reasoning_text)}, "
        f"side_messages={len(side_messages)}, filtered_flags={len(filtered_flags)}, "
        f"session_updated={bool(next_session_id)}",
        module=MODULE,
    )
    return assistant_reasoning_text, assistant_answer_text, side_messages, filtered_flags


def _present_stream_side_effects(side_messages: list[str], filtered_flags: list[bool]) -> None:
    """展示流式副作用信息。"""

    present_stream_side_effects(side_messages, filtered_flags)


def _should_keep_current_frame_for_side_effects(*, assistant_text: str, side_messages: list[str]) -> bool:
    """判断是否需要保留当前页面帧以展示侧边错误。

    参数:
        assistant_text: 助手回复正文文本。
        side_messages: 流式副作用消息列表（warning/error/cancelled）。

    返回值:
        当回复为空且存在副作用消息时返回 ``True``，表示应保留当前帧。

    异常:
        无。
    """

    return (not assistant_text.strip()) and bool(side_messages)


def render_chat_tab(
    *,
    selected_stock: WatchlistItem,
    service_client: ChatServiceClient | None,
) -> None:
    """渲染交互式分析 Tab。"""

    ticker = selected_stock.ticker
    message_state_key = _build_state_key(ticker, "messages")
    session_id_key = _build_state_key(ticker, "session_id")
    input_key = _build_state_key(ticker, "input_text")
    clear_input_key = _build_state_key(ticker, "clear_input_pending")

    messages = _ensure_messages(message_state_key)
    Log.verbose(
        f"渲染交互式分析页: ticker={ticker}, message_count={len(messages)}",
        module=MODULE,
    )
    if session_id_key not in st.session_state:
        st.session_state[session_id_key] = None
    if input_key not in st.session_state:
        st.session_state[input_key] = ""
    if clear_input_key not in st.session_state:
        st.session_state[clear_input_key] = False
    _apply_pending_input_reset(input_key=input_key, clear_input_key=clear_input_key)

    st.markdown(f"### {selected_stock.company_name} ({selected_stock.ticker}) - 交互式分析")
    history_container = st.container()
    with history_container:
        if not messages:
            st.markdown(_WELCOME_MARKDOWN)
        else:
            _render_message_history(messages)

    user_text = st.text_area(
        _INPUT_LABEL,
        key=input_key,
        placeholder=_INPUT_PLACEHOLDER,
        height=120,
    )
    send_button_key = _build_state_key(ticker, "send_button")
    if not st.button("🚀开始分析", type="primary", key=send_button_key):
        return

    normalized_user_text = user_text.strip()
    trace_id = build_request_trace_id(ticker=ticker, user_text=normalized_user_text)
    Log.info(
        f"[{trace_id}] 用户点击开始分析: ticker={ticker}, "
        f"user_text_summary={summarize_user_text(normalized_user_text)}",
        module=MODULE,
    )
    if not normalized_user_text:
        Log.warning(f"[{trace_id}] 提交被拒绝：输入为空", module=MODULE)
        st.warning(_EMPTY_INPUT_WARNING)
        return

    if service_client is None:
        Log.warning(f"[{trace_id}] 提交被拒绝：服务未初始化", module=MODULE)
        st.warning(_MISSING_SERVICE_WARNING)
        return

    user_message = _ChatMessage(role="user", content=normalized_user_text)
    messages.append(user_message)
    with history_container:
        _user_spacer_column, user_column = st.columns(_USER_MESSAGE_COLUMN_SPEC, gap="small")
        with user_column:
            with st.chat_message("user"):
                st.markdown(normalized_user_text)

    current_session_id = st.session_state[session_id_key]
    if not isinstance(current_session_id, str):
        current_session_id = None
    Log.info(
        f"[{trace_id}] 开始请求流式回复: ticker={ticker}, has_session={bool(current_session_id)}",
        module=MODULE,
    )

    try:
        with history_container:
            assistant_column, _assistant_spacer_column = st.columns(_ASSISTANT_MESSAGE_COLUMN_SPEC, gap="small")
            with assistant_column:
                with st.chat_message("assistant"):
                    assistant_reasoning_text, assistant_text, side_messages, filtered_flags = _stream_and_collect_assistant_reply(
                        service_client=service_client,
                        user_text=normalized_user_text,
                        ticker=ticker,
                        session_id=current_session_id,
                        session_id_key=session_id_key,
                        trace_id=trace_id,
                    )
        if not assistant_text.strip():
            Log.warning(f"[{trace_id}] 回复完成但文本为空", module=MODULE)
            st.warning(_EMPTY_ASSISTANT_REPLY_WARNING)
        _present_stream_side_effects(side_messages, filtered_flags)
        if _should_keep_current_frame_for_side_effects(assistant_text=assistant_text, side_messages=side_messages):
            Log.warning(
                f"[{trace_id}] 检测到空回复且存在副作用消息，保留当前页面帧展示错误，不执行 rerun",
                module=MODULE,
            )
            return
        messages.append(
            _ChatMessage(
                role="assistant",
                content=assistant_text,
                reasoning_content=assistant_reasoning_text,
            )
        )
    except Exception as exception:
        Log.error(f"[{trace_id}] 交互式分析执行失败: {exception}", exc_info=True, module=MODULE)
        st.error(f"交互式分析执行失败：{exception}")
        return
    Log.info(
        f"[{trace_id}] 交互式分析完成，准备 rerun: assistant_len={len(assistant_text)}, "
        f"side_messages={len(side_messages)}, filtered={any(filtered_flags)}",
        module=MODULE,
    )
    st.session_state[clear_input_key] = True
    st.rerun()


__all__ = [
    "ChatServiceClient",
    "StreamQueueItem",
    "create_chat_service_client",
    "render_chat_tab",
]
