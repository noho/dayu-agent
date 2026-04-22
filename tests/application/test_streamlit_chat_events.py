"""``stream_chat_events`` 纯函数单元测试（无 Streamlit / 无 3.11 强依赖）。"""

from __future__ import annotations

from dayu.contracts.events import AppEvent, AppEventType
from dayu.web.streamlit.stream_chat_events import (
    fold_app_events_to_assistant_text,
    normalize_stream_text_for_markdown,
)


def test_fold_app_events_concatenates_reasoning_deltas() -> None:
    """验证 reasoning_delta 进入主文（避免仅思考链模型在 Streamlit 中空白）。"""

    events = [
        AppEvent(type=AppEventType.REASONING_DELTA, payload="think", meta={}),
        AppEvent(type=AppEventType.REASONING_DELTA, payload="ing", meta={}),
    ]
    text, sides, filtered = fold_app_events_to_assistant_text(events)
    assert text == "thinking"
    assert sides == []
    assert filtered is False


def test_fold_app_events_concatenates_content_deltas() -> None:
    """验证多个 content_delta 会按顺序拼接为主文。"""

    events = [
        AppEvent(type=AppEventType.CONTENT_DELTA, payload="ab", meta={}),
        AppEvent(type=AppEventType.CONTENT_DELTA, payload="cd", meta={}),
    ]
    text, sides, filtered = fold_app_events_to_assistant_text(events)
    assert text == "abcd"
    assert sides == []
    assert filtered is False


def test_fold_app_events_final_answer_fills_when_no_delta() -> None:
    """验证仅有 final_answer 时主文取 content 字段。"""

    events = [
        AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload={"content": "  全文  ", "filtered": False},
            meta={},
        ),
    ]
    text, sides, filtered = fold_app_events_to_assistant_text(events)
    assert text == "全文"
    assert sides == []
    assert filtered is False


def test_fold_app_events_final_answer_accepts_string_payload() -> None:
    """验证仅有 final_answer 且 payload 为字符串时可回填主文。"""

    events = [
        AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload="  纯文本最终答案  ",
            meta={},
        ),
    ]
    text, sides, filtered = fold_app_events_to_assistant_text(events)
    assert text == "纯文本最终答案"
    assert sides == []
    assert filtered is False


def test_fold_app_events_final_answer_skipped_when_delta_exists() -> None:
    """验证已有 delta 时不再把 final_answer 的 content 重复拼入主文。"""

    events = [
        AppEvent(type=AppEventType.CONTENT_DELTA, payload="x", meta={}),
        AppEvent(
            type=AppEventType.FINAL_ANSWER,
            payload={"content": "y", "filtered": True},
            meta={},
        ),
    ]
    text, sides, filtered = fold_app_events_to_assistant_text(events)
    assert text == "x"
    assert sides == []
    assert filtered is True


def test_fold_app_events_content_delta_supports_text_field() -> None:
    """验证 content_delta 负载为 dict 且使用 text 字段时仍可输出主文。"""

    events = [
        AppEvent(type=AppEventType.CONTENT_DELTA, payload={"text": "片段A"}, meta={}),
        AppEvent(type=AppEventType.CONTENT_DELTA, payload={"text": "片段B"}, meta={}),
    ]
    text, sides, filtered = fold_app_events_to_assistant_text(events)
    assert text == "片段A片段B"
    assert sides == []
    assert filtered is False


def test_fold_app_events_normalizes_escaped_newline_to_markdown_break() -> None:
    """验证 content_delta 中的字面量 ``\\n`` 会转为真实换行。"""

    events = [
        AppEvent(type=AppEventType.CONTENT_DELTA, payload="itive...\\n- SEC EDGAR", meta={}),
    ]
    text, sides, filtered = fold_app_events_to_assistant_text(events)
    assert text == "itive...\n- SEC EDGAR"
    assert sides == []
    assert filtered is False


def test_fold_app_events_collects_side_messages() -> None:
    """验证 warning、error、cancelled 进入侧边列表。"""

    events = [
        AppEvent(type=AppEventType.WARNING, payload={"message": "w1"}, meta={}),
        AppEvent(type=AppEventType.ERROR, payload={"message": "e1"}, meta={}),
        AppEvent(type=AppEventType.CANCELLED, payload={"cancel_reason": "用户中止"}, meta={}),
    ]
    text, sides, filtered = fold_app_events_to_assistant_text(events)
    assert text == ""
    assert sides == ["w1", "e1", "执行已取消：用户中止"]
    assert filtered is False


def test_fold_app_events_collects_side_messages_from_error_key() -> None:
    """验证 warning/error 可从 error 字段提取侧边提示。"""

    events = [
        AppEvent(type=AppEventType.ERROR, payload={"error": "模型调用失败"}, meta={}),
    ]
    text, sides, filtered = fold_app_events_to_assistant_text(events)
    assert text == ""
    assert sides == ["模型调用失败"]
    assert filtered is False


def test_normalize_stream_text_for_markdown_fixes_heading_and_list() -> None:
    """验证无空格标题与内联列表会被规整为可渲染 Markdown。"""

    normalized = normalize_stream_text_for_markdown("如下：\\n\\n##硬件产品* iPhone")
    assert normalized == "如下：\n## 硬件产品\n* iPhone"


def test_normalize_stream_text_for_markdown_splits_inline_heading() -> None:
    """验证正文与标题黏连时会自动断行为独立标题。"""

    normalized = normalize_stream_text_for_markdown("核心业务分部###1.汽车（Automotive）")
    assert normalized == "核心业务分部\n### 1.汽车（Automotive）"


def test_normalize_stream_text_for_markdown_keeps_fenced_code_unchanged() -> None:
    """验证代码块中的井号内容不会被误判为标题。"""

    source = "说明\\n```python\\ntext = \"核心业务分部###1.汽车\"\\n```\\n结尾"
    normalized = normalize_stream_text_for_markdown(source)
    assert normalized == "说明\n```python\ntext = \"核心业务分部###1.汽车\"\n```\n结尾"


def test_fold_app_events_cancelled_without_reason() -> None:
    """验证取消事件无原因时的默认文案。"""

    events = [AppEvent(type=AppEventType.CANCELLED, payload={}, meta={})]
    _text, sides, _filtered = fold_app_events_to_assistant_text(events)
    assert sides == ["执行已取消"]
