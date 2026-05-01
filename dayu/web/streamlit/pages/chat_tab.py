"""交互式分析聊天 Tab 页面。

包含页面渲染与交互处理；工具函数与流式运行时管理已抽取至 ``chat/`` 目录。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import streamlit as st

from dayu.host.protocols import (
    ConversationClearPartiallyAppliedError,
    ConversationClearRejectedError,
    ConversationClearStaleError,
)
from dayu.log import Log
from dayu.services.protocols import ChatServiceProtocol
from dayu.web.streamlit.components.watchlist import WatchlistItem
from dayu.web.streamlit.pages.chat.stream_runtime import (
    ChatStreamFrameState,
    read_stream_frame_state,
    clear_chat_stream_runtime,
    poll_chat_stream_events,
    start_chat_stream_runtime,
)
from dayu.web.streamlit.pages.chat.utils import (
    build_chat_session_id,
    build_request_trace_id,
    normalize_stream_text_for_markdown,
    should_keep_current_frame_for_side_effects,
    summarize_user_text,
)

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
_FILTERED_INFO_MESSAGE = "本轮输出触发内容过滤，结果可能不完整。"


@dataclass(frozen=True)
class ChatMessage:
    """聊天消息视图模型。"""

    role: str
    content: str
    reasoning_content: str = ""


def present_stream_side_effects(side_messages: list[str], filtered_flags: list[bool]) -> None:
    """展示流式输出副作用信息。

    参数:
        side_messages: 副作用消息列表。
        filtered_flags: 内容过滤标记列表。

    返回值:
        无。

    异常:
        无。
    """

    for message in side_messages:
        st.warning(message)
    if any(filtered_flags):
        st.info(_FILTERED_INFO_MESSAGE)


def _build_state_key(ticker: str, suffix: str) -> str:
    """构建按股票代码隔离的会话状态键。

    参数:
        ticker: 股票代码。
        suffix: 键后缀标识。

    返回值:
        格式为 ``chat_tab_{ticker}_{suffix}`` 的 session_state 键。

    异常:
        无。
    """

    return f"chat_tab_{ticker}_{suffix}"


def _apply_pending_input_reset(*, input_key: str, clear_input_key: str) -> None:
    """在输入控件实例化前应用延迟清空请求。

    参数:
        input_key: 输入文本 session_state 键。
        clear_input_key: 清空标记 session_state 键。

    返回值:
        无。

    异常:
        无。
    """

    raw_pending = st.session_state.get(clear_input_key)
    if isinstance(raw_pending, bool) and raw_pending:
        st.session_state[input_key] = ""
        st.session_state[clear_input_key] = False
        Log.info(f"应用延迟输入清空: input_key={input_key}", module=MODULE)


def _ensure_messages(state_key: str) -> list[ChatMessage]:
    """确保会话消息列表存在，且元素类型为 ``_ChatMessage``。

    首次访问时初始化为空列表；类型不匹配时自动修复为合法状态。

    参数:
        state_key: session_state 中消息列表的键。

    返回值:
        合法的 ``_ChatMessage`` 列表，保证非 ``None``。

    异常:
        无。
    """

    if state_key not in st.session_state:
        st.session_state[state_key] = []
    raw_messages = st.session_state[state_key]
    if isinstance(raw_messages, list) and all(isinstance(m, ChatMessage) for m in raw_messages):
        return raw_messages
    st.session_state[state_key] = []
    reset_messages = st.session_state[state_key]
    if isinstance(reset_messages, list):
        return reset_messages
    return []


def _render_message_history(messages: list[ChatMessage]) -> None:
    """渲染历史消息。

    按对话轮次依次渲染用户消息与助手回复；助手消息中若含
    ``reasoning_content`` 则以折叠面板展示。

    参数:
        messages: 待渲染的聊天消息列表。

    返回值:
        无。

    异常:
        无。
    """

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


def _render_stream_frame(*, title, state: ChatStreamFrameState) -> None:
    """渲染当前流式帧的思考与正文。

    参数:
        title: 思考内容折叠面板标题。
        state: 流式渲染帧状态。

    返回值:
        无。

    异常:
        无。
    """

    with st.expander(title, expanded=True):
        if state.reasoning_text.strip():
            st.markdown(normalize_stream_text_for_markdown(state.reasoning_text))
        elif not state.done:
            st.markdown("正在思考...")
    st.markdown(normalize_stream_text_for_markdown(state.answer_text))


def load_history_for_ticker(
    *,
    chat_service: ChatServiceProtocol,
    ticker: str,
) -> list[ChatMessage]:
    """根据 ticker 对应的固定 session_id 加载历史会话消息。

    通过 ``ChatServiceProtocol.list_conversation_session_turn_excerpts``
    读取历史真源，不直接访问 Host 层或 archive 文件。

    参数:
        chat_service: 聊天服务协议实例。
        ticker: 股票代码。

    返回值:
        历史会话消息列表。

    异常:
        无；读路径 fail-soft，异常情况返回空列表。
    """

    session_id = build_chat_session_id(ticker)
    try:
        turns = chat_service.list_conversation_session_turn_excerpts(session_id=session_id, limit=50)
        if not turns:
            return []

        messages: list[ChatMessage] = []
        for turn in turns:
            messages.append(ChatMessage(role="user", content=turn.user_text))
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=turn.assistant_text,
                    reasoning_content=turn.reasoning_text,
                )
            )
        Log.info(
            f"加载历史会话成功: ticker={ticker}, session_id={session_id}, messages={len(messages)}",
            module=MODULE,
        )
        return messages
    except Exception as e:
        Log.error(f"加载历史会话失败: ticker={ticker}, session_id={session_id}, error={e}", module=MODULE)
        return []
    

def perform_clear_session_history(
    *,
    chat_service: ChatServiceProtocol,
    session_id: str,
    ticker: str,
    stream_state_key: str,
    message_state_key: str,
    input_key: str,
    clear_input_key: str,
) -> None:
    """执行清空会话历史，封装异常处置。

    通过 ``ChatServiceProtocol.clear_session_history`` 调用 Service 层
    清空接口，遵循 ``#117`` 共享设计 §3 异常分支语义：

    - ``KeyError``：session 不存在，提示用户。
    - ``ConversationClearRejectedError``：预检拒绝（active run / pending turn /
      outbox / 非 ACTIVE），提示用户当前不可清空。
    - ``ConversationClearStaleError``：archive 乐观锁冲突，静默重试一次。
    - ``ConversationClearPartiallyAppliedError``：archive 已清但补偿未收敛，
      提示用户并告警。

    清空成功后主动清理本地 session_state 缓存并触发 rerun。

    参数:
        chat_service: 聊天服务协议实例。
        session_id: 目标会话标识。
        ticker: 股票代码。
        stream_state_key: stream_state 的 session_state 键。
        message_state_key: messages 的 session_state 键。
        input_key: input_text 的 session_state 键。
        clear_input_key: clear_input_pending 的 session_state 键。

    返回值:
        无。

    异常:
        无；所有异常在函数内处置并转为 UI 提示。
    """

    try:
        chat_service.clear_session_history(session_id)
    except KeyError:
        Log.warning(
            f"清空会话失败——session 不存在: ticker={ticker}, session_id={session_id}",
            module=MODULE,
        )
        st.warning("会话不存在或已被删除。")
        return
    except ConversationClearRejectedError as exc:
        Log.info(
            f"清空会话被拒绝: ticker={ticker}, session_id={session_id}, reason={exc.reason}",
            module=MODULE,
        )
        st.warning(f"当前无法清空会话：{exc.reason}。请等回复完成后再试。")
        return
    except ConversationClearStaleError:
        Log.info(
            f"清空会话 archive 乐观锁冲突，静默重试: ticker={ticker}, session_id={session_id}",
            module=MODULE,
        )
        time.sleep(0.1)
        try:
            chat_service.clear_session_history(session_id)
        except Exception as retry_exc:
            Log.warning(
                f"清空会话重试失败: ticker={ticker}, session_id={session_id}, error={retry_exc}",
                module=MODULE,
            )
            st.warning("清空会话失败，请稍后重试。")
            return
    except ConversationClearPartiallyAppliedError as exc:
        Log.error(
            f"清空会话部分生效: ticker={ticker}, session_id={session_id}, "
            f"residual_sources={exc.residual_sources}",
            module=MODULE,
        )
        st.error("会话历史已清空，但部分残留待修复。请联系管理员。")
        return

    Log.info(
        f"清空会话成功: ticker={ticker}, session_id={session_id}",
        module=MODULE,
    )
    clear_chat_stream_runtime(ticker=ticker, stream_state_key=stream_state_key)
    st.session_state[message_state_key] = []
    st.session_state[input_key] = ""
    st.session_state[clear_input_key] = False
    st.rerun()


@st.fragment(run_every=0.5)
def _render_stream_polling_fragment(
    *,
    ticker: str,
    stream_state_key: str,
    message_state_key: str,
    clear_input_key: str,
) -> None:
    """渲染流式帧并轮询事件直到本轮完成。

    该 fragment 独立于主页面渲染，通过 ``run_every`` 自动刷新，
    仅重执行自身而非全部 Tab。流结束后触发全量 rerun，
    fragment 不再渲染，定时器自动停止。

    参数:
        ticker: 股票代码。
        stream_state_key: stream_state 的 session_state 键。
        message_state_key: messages 的 session_state 键。
        clear_input_key: clear_input_pending 的 session_state 键。

    返回值:
        无。

    异常:
        无。
    """

    stream_state = poll_chat_stream_events(ticker=ticker, stream_state_key=stream_state_key)
    if stream_state is None:
        return

    assistant_column, _ = st.columns(_ASSISTANT_MESSAGE_COLUMN_SPEC, gap="small")
    with assistant_column:
        with st.chat_message("assistant"):
            _render_stream_frame(title=_THINKING_EXPANDER_TITLE, state=stream_state)

    if not stream_state.done:
        return

    trace_id = stream_state.trace_id
    if stream_state.error_message.strip():
        Log.error(
            f"[{trace_id}] 交互式分析执行失败: {stream_state.error_message}",
            module=MODULE,
        )
        st.error(f"交互式分析执行失败：{stream_state.error_message}")
        clear_chat_stream_runtime(ticker=ticker, stream_state_key=stream_state_key)
        return

    assistant_text = normalize_stream_text_for_markdown(stream_state.answer_text)
    assistant_reasoning_text = normalize_stream_text_for_markdown(stream_state.reasoning_text)
    side_messages = stream_state.side_messages
    filtered_flags = stream_state.filtered_flags

    if not assistant_text.strip():
        Log.warning(f"[{trace_id}] 回复完成但文本为空", module=MODULE)
        st.warning(_EMPTY_ASSISTANT_REPLY_WARNING)
    present_stream_side_effects(side_messages, filtered_flags)
    if should_keep_current_frame_for_side_effects(assistant_text=assistant_text, side_messages=side_messages):
        Log.warning(
            f"[{trace_id}] 检测到空回复且存在副作用消息，保留当前页面帧展示错误，不执行 rerun",
            module=MODULE,
        )
        clear_chat_stream_runtime(ticker=ticker, stream_state_key=stream_state_key)
        return

    messages = _ensure_messages(message_state_key)
    messages.append(
        ChatMessage(
            role="assistant",
            content=assistant_text,
            reasoning_content=assistant_reasoning_text,
        )
    )
    Log.info(
        f"[{trace_id}] 交互式分析完成，准备 rerun: assistant_len={len(assistant_text)}, "
        f"side_messages={len(side_messages)}, filtered={any(filtered_flags)}",
        module=MODULE,
    )
    st.session_state[clear_input_key] = True
    clear_chat_stream_runtime(ticker=ticker, stream_state_key=stream_state_key)
    st.rerun()


def render_chat_tab(
    *,
    selected_stock: WatchlistItem,
    chat_service: ChatServiceProtocol | None = None,
) -> None:
    """渲染交互式分析 Tab。"""

    ticker = selected_stock.ticker
    message_state_key = _build_state_key(ticker, "messages")
    input_key = _build_state_key(ticker, "input_text")
    clear_input_key = _build_state_key(ticker, "clear_input_pending")
    stream_state_key = _build_state_key(ticker, "stream_state")

    stream_state = read_stream_frame_state(stream_state_key)

    messages = _ensure_messages(message_state_key)
    Log.verbose(
        f"渲染交互式分析页: ticker={ticker}, message_count={len(messages)}",
        module=MODULE,
    )

    if (stream_state is None) and (st.session_state.get(message_state_key) is not None) and (chat_service is not None):
        messages = load_history_for_ticker(
            chat_service=chat_service,
            ticker=ticker,
        )
        st.session_state[message_state_key] = messages
        messages = _ensure_messages(message_state_key)

    if input_key not in st.session_state:
        st.session_state[input_key] = ""
    if clear_input_key not in st.session_state:
        st.session_state[clear_input_key] = False
    _apply_pending_input_reset(input_key=input_key, clear_input_key=clear_input_key)

    title_col, clear_col = st.columns([9, 1], gap="small", vertical_alignment="center")
    with title_col:
        st.markdown(f"### {selected_stock.company_name} ({selected_stock.ticker}) - 交互式分析")
    with clear_col:
        clear_button_key = _build_state_key(ticker, "clear_button")
        if st.button("清空会话", icon="🗑", key=clear_button_key, width="stretch"):
            if stream_state is not None and (not stream_state.done):
                st.warning("当前回答仍在生成中，请等待完成后再清空会话。")
            elif chat_service is None:
                st.warning("聊天服务未就绪，无法清空后端历史。")
            else:
                session_id = build_chat_session_id(ticker)
                perform_clear_session_history(
                    chat_service=chat_service,
                    session_id=session_id,
                    ticker=ticker,
                    stream_state_key=stream_state_key,
                    message_state_key=message_state_key,
                    input_key=input_key,
                    clear_input_key=clear_input_key,
                )

    history_container = st.container()
    with history_container:
        if not messages:
            st.markdown(_WELCOME_MARKDOWN)
        else:
            _render_message_history(messages)
        if stream_state is not None:
            _render_stream_polling_fragment(
                ticker=ticker,
                stream_state_key=stream_state_key,
                message_state_key=message_state_key,
                clear_input_key=clear_input_key,
            )

    user_text = st.text_area(
        _INPUT_LABEL,
        key=input_key,
        placeholder=_INPUT_PLACEHOLDER,
        height=120,
    )
    send_button_key = _build_state_key(ticker, "send_button")
    is_running = stream_state is not None and (not stream_state.done)
    send_clicked = st.button(
        "🚀正在分析中。。。" if is_running else "🚀开始分析",
        type="primary",
        key=send_button_key,
        disabled=is_running,
    )
    if send_clicked:
        if stream_state is not None and (not stream_state.done):
            st.warning("当前回答仍在生成中，请稍候再提交新问题。")
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
        if chat_service is None:
            Log.warning(f"[{trace_id}] 提交被拒绝：服务未初始化", module=MODULE)
            st.warning(_MISSING_SERVICE_WARNING)
            return
        messages.append(ChatMessage(role="user", content=normalized_user_text))
        session_id = build_chat_session_id(ticker)
        Log.info(
            f"[{trace_id}] 开始请求流式回复: ticker={ticker}, session_id={session_id}",
            module=MODULE,
        )
        start_chat_stream_runtime(
            chat_service=chat_service,
            ticker=ticker,
            user_text=normalized_user_text,
            session_id=session_id,
            trace_id=trace_id,
            stream_state_key=stream_state_key,
        )
        st.rerun()
