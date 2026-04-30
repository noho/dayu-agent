"""聊天模块纯函数测试。

覆盖 ``chat/utils.py``、``chat/stream_runtime.py``、``chat_tab.py`` 中
不依赖 st.session_state 与 Streamlit 渲染的纯逻辑函数与 dataclass。
"""

from __future__ import annotations

import re
import pytest

from dayu.contracts.events import AppEvent, AppEventType
from dayu.web.streamlit.pages.chat.stream_runtime import (
    StreamQueueItem,
)
from dayu.web.streamlit.pages.chat.utils import (
    build_chat_session_id,
    build_request_trace_id,
    extract_stream_text,
    fold_app_events_to_assistant_text,
    normalize_stream_text_for_markdown,
    should_keep_current_frame_for_side_effects,
    summarize_user_text,
)
from dayu.web.streamlit.pages.chat_tab import ChatMessage


# ═════════════════════════════════════════════════════════════════════════
# chat_tab.py — _ChatMessage
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestChatMessage:
    """_ChatMessage dataclass 测试。"""

    def test_default_reasoning_content(self) -> None:
        """不传 reasoning_content 时应默认为空字符串。"""
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.reasoning_content == ""

    def test_with_reasoning_content(self) -> None:
        """传入 reasoning_content 时应正确存储。"""
        msg = ChatMessage(role="assistant", content="reply", reasoning_content="thinking...")
        assert msg.role == "assistant"
        assert msg.content == "reply"
        assert msg.reasoning_content == "thinking..."

    def test_frozen(self) -> None:
        """_ChatMessage 应为 frozen dataclass。"""
        msg = ChatMessage(role="user", content="hello")
        with pytest.raises(Exception):
            msg.content = "changed"  # type: ignore[misc]


# ═════════════════════════════════════════════════════════════════════════
# chat/utils.py — build_chat_session_id
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestBuildChatSessionId:
    """build_chat_session_id 测试。"""

    def test_format_prefix(self) -> None:
        """返回的 session_id 应以 streamlit-web- 开头。"""
        sid = build_chat_session_id("AAPL")
        assert sid.startswith("streamlit-web-")

    def test_same_ticker_same_id(self) -> None:
        """同一 ticker 多次调用应返回相同 session_id。"""
        assert build_chat_session_id("AAPL") == build_chat_session_id("AAPL")

    def test_different_ticker_different_id(self) -> None:
        """不同 ticker 应返回不同 session_id。"""
        assert build_chat_session_id("AAPL") != build_chat_session_id("GOOGL")

    def test_ticker_whitespace_trim(self) -> None:
        """带空白的 ticker 应被归一化。"""
        sid_trimmed = build_chat_session_id("  AAPL  ")
        sid_normal = build_chat_session_id("AAPL")
        assert sid_trimmed == sid_normal

    def test_lowercase_ticker_normalized(self) -> None:
        """小写 ticker 应与大写等同。"""
        assert build_chat_session_id("aapl") == build_chat_session_id("AAPL")


# ═════════════════════════════════════════════════════════════════════════
# chat/utils.py — build_request_trace_id
# ═════════════════════════════════════════════════════════════════════════


_TRACE_ID_PATTERN = re.compile(r"^[A-Z0-9]+-\d{13}-\d{5}$")


@pytest.mark.unit
class TestBuildRequestTraceId:
    """build_request_trace_id 测试。"""

    def test_format(self) -> None:
        """trace_id 应符合 TICKER-TIMESTAMP-HASH 格式。"""
        tid = build_request_trace_id(ticker="AAPL", user_text="hello")
        assert _TRACE_ID_PATTERN.match(tid), f"got {tid!r}"

    def test_different_input_different_trace_id(self) -> None:
        """不同输入应生成不同 trace_id。"""
        tid1 = build_request_trace_id(ticker="AAPL", user_text="hello")
        tid2 = build_request_trace_id(ticker="AAPL", user_text="world")
        assert tid1 != tid2

    def test_empty_ticker_unknown_prefix(self) -> None:
        """空 ticker 应回退为 UNKNOWN 前缀。"""
        tid = build_request_trace_id(ticker="   ", user_text="test")
        assert tid.startswith("UNKNOWN-")

    def test_same_input_same_hash(self) -> None:
        """同一输入多次调用应产生相同 hash 段。"""
        tid1 = build_request_trace_id(ticker="AAPL", user_text="hello")
        tid2 = build_request_trace_id(ticker="AAPL", user_text="hello")
        # 时间戳可能不同，所以只比较 hash 段（最后 5 位）
        assert tid1[-5:] == tid2[-5:]

    def test_ticker_truncated_to_8_chars(self) -> None:
        """过长 ticker 前缀应截断至 8 字符。"""
        tid = build_request_trace_id(ticker="VERYLONGTICKER", user_text="test")
        prefix = tid.split("-")[0]
        assert len(prefix) <= 8


# ═════════════════════════════════════════════════════════════════════════
# chat/utils.py — summarize_user_text
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSummarizeUserText:
    """summarize_user_text 测试。"""

    def test_short_text_no_ellipsis(self) -> None:
        """短文本不应追加省略号。"""
        result = summarize_user_text("hello")
        assert "len=" in result
        assert "preview=" in result
        assert not result.endswith("...'")

    def test_long_text_has_ellipsis(self) -> None:
        """超长（>48 字符）文本应在 repr 闭合引号后追加省略号。"""
        long_text = "hello world " * 20
        result = summarize_user_text(long_text)
        # preview 截断到 48 字符后以 !r 输出，省略号追加在闭合引号之后
        assert result.endswith("...")

    def test_newline_collapsed(self) -> None:
        """换行符应被空格替代。"""
        result = summarize_user_text("line1\nline2\n  line3")
        assert "line1 line2 line3" in result

    def test_len_field_matches_input(self) -> None:
        """len 字段应反映原始输入长度。"""
        result = summarize_user_text("abcde")
        assert "len=5" in result


# ═════════════════════════════════════════════════════════════════════════
# chat/utils.py — should_keep_current_frame_for_side_effects
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestShouldKeepCurrentFrame:
    """should_keep_current_frame_for_side_effects 测试。"""

    def test_empty_text_no_side_effects(self) -> None:
        """空回复且无副作用 → False。"""
        assert should_keep_current_frame_for_side_effects(assistant_text="", side_messages=[]) is False

    def test_empty_text_with_side_effects(self) -> None:
        """空回复但有副作用 → True。"""
        assert should_keep_current_frame_for_side_effects(assistant_text="", side_messages=["err"]) is True

    def test_non_empty_text_no_side_effects(self) -> None:
        """有回复无副作用 → False。"""
        assert should_keep_current_frame_for_side_effects(assistant_text="hello", side_messages=[]) is False

    def test_non_empty_text_with_side_effects(self) -> None:
        """有回复有副作用 → False。"""
        assert should_keep_current_frame_for_side_effects(assistant_text="hello", side_messages=["warn"]) is False

    def test_whitespace_only_text_no_side_effects(self) -> None:
        """纯空白回复且无副作用 → False。"""
        assert should_keep_current_frame_for_side_effects(assistant_text="   ", side_messages=[]) is False

    def test_whitespace_only_text_with_side_effects(self) -> None:
        """纯空白回复有副作用 → True。"""
        assert should_keep_current_frame_for_side_effects(assistant_text="  \n ", side_messages=["x"]) is True


# ═════════════════════════════════════════════════════════════════════════
# chat/utils.py — extract_stream_text
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestExtractStreamText:
    """extract_stream_text 测试（委托 _payload_to_text）。"""

    def test_string_payload(self) -> None:
        """字符串负载应原样返回（经 markdown 规整）。"""
        assert extract_stream_text("hello world") == "hello world"

    def test_empty_string(self) -> None:
        """空字符串应返回空。"""
        assert extract_stream_text("") == ""

    def test_dict_content_key(self) -> None:
        """字典含 content 键时应提取其值。"""
        assert extract_stream_text({"content": "result"}) == "result"

    def test_dict_text_key(self) -> None:
        """字典含 text 键时应提取其值。"""
        assert extract_stream_text({"text": "answer"}) == "answer"

    def test_dict_answer_key(self) -> None:
        """字典含 answer 键时应提取其值。"""
        assert extract_stream_text({"answer": "final"}) == "final"

    def test_dict_priority_content_over_text(self) -> None:
        """content 键优先级高于 text 键。"""
        result = extract_stream_text({"content": "first", "text": "second"})
        assert result == "first"

    def test_dict_whitespace_only_content(self) -> None:
        """字典 content 为纯空白时跳过并尝试下一键。"""
        result = extract_stream_text({"content": "   ", "text": "valid"})
        assert result == "valid"

    def test_empty_dict(self) -> None:
        """空字典返回空字符串。"""
        assert extract_stream_text({}) == ""

    def test_dict_no_known_keys(self) -> None:
        """无可识别键的字典返回空字符串。"""
        assert extract_stream_text({"unknown": "val"}) == ""

    def test_whitespace_only_string(self) -> None:
        """纯空白字符串返回空。"""
        assert extract_stream_text("   \n  ") == ""


# ═════════════════════════════════════════════════════════════════════════
# chat/utils.py — fold_app_events_to_assistant_text
# ═════════════════════════════════════════════════════════════════════════


def _make_event(event_type: AppEventType, payload: str | dict[str, str | bool] = "") -> AppEvent:
    """构造测试用 AppEvent。

    参数:
        event_type: 应用事件类型。
        payload: 事件负载，支持字符串或 ``{key: str|bool}`` 字典。

    返回值:
        填充了指定类型与负载的 AppEvent 实例。

    异常:
        无。
    """
    return AppEvent(type=event_type, payload=payload)


@pytest.mark.unit
class TestFoldAppEventsToAssistantText:
    """fold_app_events_to_assistant_text 测试。"""

    def test_content_delta_concatenation(self) -> None:
        """多个 CONTENT_DELTA 事件应拼接为同一文本。"""
        events = [
            _make_event(AppEventType.CONTENT_DELTA, "hello "),
            _make_event(AppEventType.CONTENT_DELTA, "world"),
        ]
        text, side, filtered = fold_app_events_to_assistant_text(events)
        assert text == "hello world"
        assert side == []
        assert filtered is False

    def test_reasoning_delta_folded_to_text(self) -> None:
        """REASONING_DELTA 应被折叠到主文本中。"""
        events = [
            _make_event(AppEventType.REASONING_DELTA, "think..."),
            _make_event(AppEventType.CONTENT_DELTA, "answer"),
        ]
        text, side, filtered = fold_app_events_to_assistant_text(events)
        assert "think..." in text
        assert "answer" in text

    def test_final_answer_sets_filtered_flag(self) -> None:
        """FINAL_ANSWER 含 filtered=True 时应置位 filtered 标记。"""
        events = [_make_event(AppEventType.FINAL_ANSWER, {"filtered": True})]
        _text, _side, filtered = fold_app_events_to_assistant_text(events)
        assert filtered is True

    def test_final_answer_text_fallback(self) -> None:
        """无 CONTENT_DELTA 时 FINAL_ANSWER 文本应作为回退。"""
        events = [_make_event(AppEventType.FINAL_ANSWER, {"content": "final answer"})]
        text, _side, _filtered = fold_app_events_to_assistant_text(events)
        assert "final answer" in text

    def test_warning_event_collected_as_side_message(self) -> None:
        """WARNING 事件应进入 side_messages。"""
        events = [_make_event(AppEventType.WARNING, {"message": "careful"})]
        _text, side, _filtered = fold_app_events_to_assistant_text(events)
        assert "careful" in side

    def test_error_event_collected_as_side_message(self) -> None:
        """ERROR 事件应进入 side_messages。"""
        events = [_make_event(AppEventType.ERROR, {"error": "failed"})]
        _text, side, _filtered = fold_app_events_to_assistant_text(events)
        assert "failed" in side

    def test_cancelled_event_default_message(self) -> None:
        """CANCELLED 事件无 reason 时应使用默认消息。"""
        events = [_make_event(AppEventType.CANCELLED, {})]
        _text, side, _filtered = fold_app_events_to_assistant_text(events)
        assert any("取消" in m for m in side)

    def test_cancelled_event_with_reason(self) -> None:
        """CANCELLED 事件有 cancel_reason 时应包含原因。"""
        events = [_make_event(AppEventType.CANCELLED, {"cancel_reason": "timeout"})]
        _text, side, _filtered = fold_app_events_to_assistant_text(events)
        assert any("timeout" in m for m in side)

    def test_empty_events(self) -> None:
        """空事件列表返回三元空组。"""
        text, side, filtered = fold_app_events_to_assistant_text([])
        assert text == ""
        assert side == []
        assert filtered is False

    def test_mixed_content_and_warning(self) -> None:
        """混合内容与副作用事件各自正确归类。"""
        events = [
            _make_event(AppEventType.CONTENT_DELTA, "data"),
            _make_event(AppEventType.WARNING, {"message": "warn1"}),
        ]
        text, side, filtered = fold_app_events_to_assistant_text(events)
        assert "data" in text
        assert "warn1" in side
        assert filtered is False

    def test_dict_content_delta(self) -> None:
        """CONTENT_DELTA 带字典负载应正确提取文本。"""
        events = [_make_event(AppEventType.CONTENT_DELTA, {"content": "chunk"})]
        text, _side, _filtered = fold_app_events_to_assistant_text(events)
        assert "chunk" in text


# ═════════════════════════════════════════════════════════════════════════
# chat/utils.py — normalize_stream_text_for_markdown
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestNormalizeStreamTextForMarkdown:
    """normalize_stream_text_for_markdown 测试。"""

    def test_empty_string(self) -> None:
        """空输入返回空字符串。"""
        assert normalize_stream_text_for_markdown("") == ""

    def test_plain_text_unchanged(self) -> None:
        """纯文本无特殊字符时原样返回。"""
        assert normalize_stream_text_for_markdown("hello world") == "hello world"

    def test_heading_gets_space(self) -> None:
        """## 后缺少空格时自动补全。"""
        result = normalize_stream_text_for_markdown("##Title")
        assert result == "## Title"

    def test_escaped_newline_to_real_newline(self) -> None:
        """\\\\n 转义换行转为真实换行。"""
        result = normalize_stream_text_for_markdown(r"line1\nline2")
        assert "\n" in result

    def test_code_fence_preserved(self) -> None:
        """代码块内容不被规整破坏。"""
        result = normalize_stream_text_for_markdown("```\n# code\n```")
        assert "```" in result
        assert "# code" in result

    def test_code_fence_escaped_newline(self) -> None:
        """代码块内 \\\\n 转为真实换行。"""
        result = normalize_stream_text_for_markdown("```\\npython\\n```")
        lines = result.split("\n")
        assert lines[0] == "```"
        assert lines[1] == "python"
        assert lines[2] == "```"

    def test_inline_heading_split(self) -> None:
        """行内 ## 前无换行且后随非空白字符时：先拆行再补空格。"""
        result = normalize_stream_text_for_markdown("text##Heading")
        assert "text\n## Heading" in result

    def test_inline_heading_with_space_no_split(self) -> None:
        """## 后已有空格时不做拆分（已是合法 Markdown）。"""
        result = normalize_stream_text_for_markdown("text## heading")
        assert result == "text## heading"

    def test_inline_star_list_split(self) -> None:
        """行内 * 列表前插入换行。"""
        result = normalize_stream_text_for_markdown("text* item")
        # "* " 前有非空白字符时应在 * 前插入换行
        assert "\n* item" in result

    def test_inline_hash_heading_with_space(self) -> None:
        """# 后无空格时补空格。"""
        result = normalize_stream_text_for_markdown("#heading")
        assert result == "# heading"

    def test_multiple_hash_heading(self) -> None:
        """### 后无空格时：首个 # 视作行内字符，## 拆行并补空格。"""
        result = normalize_stream_text_for_markdown("###sub")
        assert "\n## sub" in result


# ═════════════════════════════════════════════════════════════════════════
# chat/stream_runtime.py — StreamQueueItem
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestStreamQueueItem:
    """StreamQueueItem dataclass 测试。"""

    def test_default_values(self) -> None:
        """默认字段值应正确。"""
        item = StreamQueueItem(done=False)
        assert item.done is False
        assert item.kind == "content"
        assert item.chunk == ""
        assert item.event_type == "chunk"
        assert item.flag is False

    def test_reasoning_kind(self) -> None:
        """reasoning 类事件应正确设置 kind。"""
        item = StreamQueueItem(done=False, kind="reasoning", chunk="think")
        assert item.kind == "reasoning"
        assert item.chunk == "think"

    def test_done_item(self) -> None:
        """done=True 事件应正确标记。"""
        item = StreamQueueItem(done=True, event_type="done")
        assert item.done is True
        assert item.event_type == "done"

    def test_frozen(self) -> None:
        """StreamQueueItem 为 frozen dataclass。"""
        item = StreamQueueItem(done=False)
        with pytest.raises(Exception):
            item.done = True  # type: ignore[misc]

