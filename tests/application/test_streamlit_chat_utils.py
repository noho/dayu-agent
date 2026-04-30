"""聊天工具函数测试。"""

from __future__ import annotations

import re

import pytest

from dayu.contracts.events import AppEvent, AppEventType
from dayu.web.streamlit.pages.chat.utils import (
    build_chat_session_id,
    build_request_trace_id,
    extract_stream_text,
    fold_app_events_to_assistant_text,
    normalize_stream_text_for_markdown,
    should_keep_current_frame_for_side_effects,
    summarize_user_text,
)


_TRACE_ID_PATTERN = re.compile(r"^[A-Z0-9]+-\d{13}-\d{5}$")


@pytest.mark.unit
class TestBuildChatSessionId:
    def test_format_prefix(self) -> None:
        assert build_chat_session_id("AAPL").startswith("streamlit-web-")

    def test_same_ticker_same_id(self) -> None:
        assert build_chat_session_id("AAPL") == build_chat_session_id("AAPL")

    def test_different_ticker_different_id(self) -> None:
        assert build_chat_session_id("AAPL") != build_chat_session_id("GOOGL")

    def test_ticker_whitespace_trim(self) -> None:
        assert build_chat_session_id("  AAPL  ") == build_chat_session_id("AAPL")

    def test_lowercase_ticker_normalized(self) -> None:
        assert build_chat_session_id("aapl") == build_chat_session_id("AAPL")


@pytest.mark.unit
class TestBuildRequestTraceId:
    def test_format(self) -> None:
        tid = build_request_trace_id(ticker="AAPL", user_text="hello")
        assert _TRACE_ID_PATTERN.match(tid), f"got {tid!r}"

    def test_different_input_different_trace_id(self) -> None:
        assert build_request_trace_id(ticker="AAPL", user_text="hello") != build_request_trace_id(
            ticker="AAPL", user_text="world"
        )

    def test_empty_ticker_unknown_prefix(self) -> None:
        assert build_request_trace_id(ticker="   ", user_text="test").startswith("UNKNOWN-")

    def test_same_input_same_hash(self) -> None:
        tid1 = build_request_trace_id(ticker="AAPL", user_text="hello")
        tid2 = build_request_trace_id(ticker="AAPL", user_text="hello")
        assert tid1[-5:] == tid2[-5:]

    def test_ticker_truncated_to_8_chars(self) -> None:
        prefix = build_request_trace_id(ticker="VERYLONGTICKER", user_text="test").split("-")[0]
        assert len(prefix) <= 8


@pytest.mark.unit
class TestSummarizeUserText:
    def test_short_text_no_ellipsis(self) -> None:
        result = summarize_user_text("hello")
        assert "len=" in result
        assert "preview=" in result
        assert not result.endswith("...'")

    def test_long_text_has_ellipsis(self) -> None:
        assert summarize_user_text("hello world " * 20).endswith("...")

    def test_newline_collapsed(self) -> None:
        assert "line1 line2 line3" in summarize_user_text("line1\nline2\n  line3")

    def test_len_field_matches_input(self) -> None:
        assert "len=5" in summarize_user_text("abcde")


@pytest.mark.unit
class TestShouldKeepCurrentFrame:
    def test_empty_text_no_side_effects(self) -> None:
        assert should_keep_current_frame_for_side_effects(assistant_text="", side_messages=[]) is False

    def test_empty_text_with_side_effects(self) -> None:
        assert should_keep_current_frame_for_side_effects(assistant_text="", side_messages=["err"]) is True

    def test_non_empty_text_no_side_effects(self) -> None:
        assert should_keep_current_frame_for_side_effects(assistant_text="hello", side_messages=[]) is False

    def test_non_empty_text_with_side_effects(self) -> None:
        assert should_keep_current_frame_for_side_effects(assistant_text="hello", side_messages=["warn"]) is False

    def test_whitespace_only_text_no_side_effects(self) -> None:
        assert should_keep_current_frame_for_side_effects(assistant_text="   ", side_messages=[]) is False

    def test_whitespace_only_text_with_side_effects(self) -> None:
        assert should_keep_current_frame_for_side_effects(assistant_text="  \n ", side_messages=["x"]) is True


@pytest.mark.unit
class TestExtractStreamText:
    def test_string_payload(self) -> None:
        assert extract_stream_text("hello world") == "hello world"

    def test_empty_string(self) -> None:
        assert extract_stream_text("") == ""

    def test_dict_content_key(self) -> None:
        assert extract_stream_text({"content": "result"}) == "result"

    def test_dict_text_key(self) -> None:
        assert extract_stream_text({"text": "answer"}) == "answer"

    def test_dict_answer_key(self) -> None:
        assert extract_stream_text({"answer": "final"}) == "final"

    def test_dict_priority_content_over_text(self) -> None:
        assert extract_stream_text({"content": "first", "text": "second"}) == "first"

    def test_dict_whitespace_only_content(self) -> None:
        assert extract_stream_text({"content": "   ", "text": "valid"}) == "valid"

    def test_empty_dict(self) -> None:
        assert extract_stream_text({}) == ""

    def test_dict_no_known_keys(self) -> None:
        assert extract_stream_text({"unknown": "val"}) == ""

    def test_whitespace_only_string(self) -> None:
        assert extract_stream_text("   \n  ") == ""


def _make_event(event_type: AppEventType, payload: str | dict[str, str | bool] = "") -> AppEvent:
    return AppEvent(type=event_type, payload=payload)


@pytest.mark.unit
class TestFoldAppEventsToAssistantText:
    def test_content_delta_concatenation(self) -> None:
        text, side, filtered = fold_app_events_to_assistant_text(
            [
                _make_event(AppEventType.CONTENT_DELTA, "hello "),
                _make_event(AppEventType.CONTENT_DELTA, "world"),
            ]
        )
        assert text == "hello world"
        assert side == []
        assert filtered is False

    def test_reasoning_delta_folded_to_text(self) -> None:
        text, side, filtered = fold_app_events_to_assistant_text(
            [
                _make_event(AppEventType.REASONING_DELTA, "think..."),
                _make_event(AppEventType.CONTENT_DELTA, "answer"),
            ]
        )
        assert "think..." in text
        assert "answer" in text
        assert side == []
        assert filtered is False

    def test_final_answer_sets_filtered_flag(self) -> None:
        _text, _side, filtered = fold_app_events_to_assistant_text(
            [_make_event(AppEventType.FINAL_ANSWER, {"filtered": True})]
        )
        assert filtered is True

    def test_final_answer_text_fallback(self) -> None:
        text, _side, _filtered = fold_app_events_to_assistant_text(
            [_make_event(AppEventType.FINAL_ANSWER, {"content": "final answer"})]
        )
        assert "final answer" in text

    def test_warning_event_collected_as_side_message(self) -> None:
        _text, side, _filtered = fold_app_events_to_assistant_text(
            [_make_event(AppEventType.WARNING, {"message": "careful"})]
        )
        assert "careful" in side

    def test_error_event_collected_as_side_message(self) -> None:
        _text, side, _filtered = fold_app_events_to_assistant_text(
            [_make_event(AppEventType.ERROR, {"error": "failed"})]
        )
        assert "failed" in side

    def test_cancelled_event_default_message(self) -> None:
        _text, side, _filtered = fold_app_events_to_assistant_text([_make_event(AppEventType.CANCELLED, {})])
        assert any("取消" in message for message in side)

    def test_cancelled_event_with_reason(self) -> None:
        _text, side, _filtered = fold_app_events_to_assistant_text(
            [_make_event(AppEventType.CANCELLED, {"cancel_reason": "timeout"})]
        )
        assert any("timeout" in message for message in side)

    def test_empty_events(self) -> None:
        text, side, filtered = fold_app_events_to_assistant_text([])
        assert text == ""
        assert side == []
        assert filtered is False

    def test_mixed_content_and_warning(self) -> None:
        text, side, filtered = fold_app_events_to_assistant_text(
            [
                _make_event(AppEventType.CONTENT_DELTA, "data"),
                _make_event(AppEventType.WARNING, {"message": "warn1"}),
            ]
        )
        assert "data" in text
        assert "warn1" in side
        assert filtered is False

    def test_dict_content_delta(self) -> None:
        text, _side, _filtered = fold_app_events_to_assistant_text(
            [_make_event(AppEventType.CONTENT_DELTA, {"content": "chunk"})]
        )
        assert "chunk" in text


@pytest.mark.unit
class TestNormalizeStreamTextForMarkdown:
    def test_empty_string(self) -> None:
        assert normalize_stream_text_for_markdown("") == ""

    def test_plain_text_unchanged(self) -> None:
        assert normalize_stream_text_for_markdown("hello world") == "hello world"

    def test_heading_gets_space(self) -> None:
        assert normalize_stream_text_for_markdown("##Title") == "## Title"

    def test_escaped_newline_to_real_newline(self) -> None:
        assert "\n" in normalize_stream_text_for_markdown(r"line1\nline2")

    def test_code_fence_preserved(self) -> None:
        result = normalize_stream_text_for_markdown("```\n# code\n```")
        assert "```" in result
        assert "# code" in result

    def test_code_fence_escaped_newline(self) -> None:
        lines = normalize_stream_text_for_markdown("```\\npython\\n```").split("\n")
        assert lines[0] == "```"
        assert lines[1] == "python"
        assert lines[2] == "```"

    def test_inline_heading_split(self) -> None:
        assert "text\n## Heading" in normalize_stream_text_for_markdown("text##Heading")

    def test_inline_heading_with_space_no_split(self) -> None:
        assert normalize_stream_text_for_markdown("text## heading") == "text## heading"

    def test_inline_star_list_split(self) -> None:
        assert "\n* item" in normalize_stream_text_for_markdown("text* item")

    def test_inline_hash_heading_with_space(self) -> None:
        assert normalize_stream_text_for_markdown("#heading") == "# heading"

    def test_multiple_hash_heading(self) -> None:
        assert "\n## sub" in normalize_stream_text_for_markdown("###sub")
