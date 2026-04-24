# -*- coding: utf-8 -*-
"""AsyncOpenAIRunner 工具执行相关测试"""

import asyncio
from typing import Any, cast

import pytest

from dayu.contracts.protocols import ToolExecutor
from dayu.engine import async_openai_runner as aor
from dayu.engine.events import EventType


class _DummyExecutor:
    def __init__(self, result):
        self._result = result

    def get_schemas(self):
        return []

    def execute(self, name, arguments, context=None):
        if isinstance(self._result, Exception):
            raise self._result
        return cast(dict[str, Any], self._result)

    def clear_cursors(self) -> None:
        pass

    def get_dup_call_spec(self, name: str):
        del name
        return None

    def get_execution_context_param_name(self, name: str) -> str | None:
        del name
        return None

    def get_tool_display_info(self, name: str) -> tuple[str, list[str] | None]:
        return name, None

    def register_response_middleware(self, callback) -> None:
        del callback


def _make_runner():
    if aor.aiohttp is None:
        aor.aiohttp = object()
    return aor.AsyncOpenAIRunner(
        endpoint_url="http://example.com",
        model="test-model",
        headers={},
        supports_stream=True,
        supports_tool_calling=True,
    )


@pytest.mark.asyncio
async def test_run_tool_call_success():
    runner = _make_runner()
    runner.set_tools(_DummyExecutor({"ok": True, "value": "ok"}))

    tool_call = {
        "id": "t1",
        "name": "echo",
        "arguments": {"x": 1},
        "index_in_iteration": 0,
    }

    result = await runner._run_tool_call(tool_call, "req1", {"run_id": "r", "iteration_id": "t"})
    assert result["result"]["ok"] is True
    assert result["result"]["value"] == "ok"
    assert result["result"]["meta"]["tool"] == "echo"


@pytest.mark.asyncio
async def test_run_tool_call_failure():
    runner = _make_runner()
    runner.set_tools(_DummyExecutor(RuntimeError("boom")))

    tool_call = {
        "id": "t1",
        "name": "echo",
        "arguments": {"x": 1},
        "index_in_iteration": 0,
    }

    result = await runner._run_tool_call(tool_call, "req1", {"run_id": "r", "iteration_id": "t"})
    assert result["result"]["ok"] is False
    assert result["result"]["error"] == "execution_error"


@pytest.mark.asyncio
async def test_run_tool_call_invalid_result():
    runner = _make_runner()
    runner.set_tools(_DummyExecutor("not_dict"))

    tool_call = {
        "id": "t1",
        "name": "echo",
        "arguments": {"x": 1},
        "index_in_iteration": 0,
    }

    result = await runner._run_tool_call(tool_call, "req1", {"run_id": "r", "iteration_id": "t"})
    assert result["result"]["ok"] is False
    assert result["result"]["error"] == "invalid_result"


@pytest.mark.asyncio
async def test_run_tool_call_legacy_result_rejected():
    runner = _make_runner()
    runner.set_tools(_DummyExecutor({"success": True, "data": {"type": "text", "value": "ok"}}))

    tool_call = {
        "id": "t1",
        "name": "echo",
        "arguments": {"x": 1},
        "index_in_iteration": 0,
    }

    result = await runner._run_tool_call(tool_call, "req1", {"run_id": "r", "iteration_id": "t"})
    assert result["result"]["ok"] is False
    assert result["result"]["error"] == "invalid_result"


@pytest.mark.asyncio
async def test_run_tool_call_invalid_error_and_none_data():
    runner = _make_runner()
    runner.set_tools(_DummyExecutor({"ok": False, "error": "bad", "message": ""}))

    tool_call = {
        "id": "t1",
        "name": "echo",
        "arguments": {"x": 1},
        "index_in_iteration": 0,
    }

    result = await runner._run_tool_call(tool_call, "req1", {"run_id": "r", "iteration_id": "t"})
    assert result["result"]["error"] == "invalid_result"

    runner.set_tools(_DummyExecutor({"ok": True, "value": None}))
    result = await runner._run_tool_call(tool_call, "req1", {"run_id": "r", "iteration_id": "t"})
    assert result["result"]["value"] is None


@pytest.mark.asyncio
async def test_run_tool_call_wraps_bytes_and_tuple():
    runner = _make_runner()
    runner.set_tools(_DummyExecutor({"ok": True, "value": b"abc"}))

    tool_call = {
        "id": "t1",
        "name": "echo",
        "arguments": {"x": 1},
        "index_in_iteration": 0,
    }

    result = await runner._run_tool_call(tool_call, "req1", {"run_id": "r", "iteration_id": "t"})
    assert result["result"]["value"] == b"abc"

    runner.set_tools(_DummyExecutor({"ok": True, "value": (1, 2)}))
    result = await runner._run_tool_call(tool_call, "req1", {"run_id": "r", "iteration_id": "t"})
    assert result["result"]["value"] == (1, 2)


@pytest.mark.asyncio
async def test_run_tool_call_timeout(monkeypatch):
    runner = _make_runner()
    runner.tool_timeout_seconds = 1.0
    runner.set_tools(_DummyExecutor({"ok": True, "value": "ok"}))

    async def fake_wait_for(*args, **kwargs):
        await args[0]
        raise asyncio.TimeoutError()

    import asyncio as _asyncio
    monkeypatch.setattr(_asyncio, "wait_for", fake_wait_for)

    tool_call = {
        "id": "t1",
        "name": "echo",
        "arguments": {"x": 1},
        "index_in_iteration": 0,
    }

    result = await runner._run_tool_call(tool_call, "req1", {"run_id": "r", "iteration_id": "t"})
    assert result["result"]["ok"] is False
    assert result["result"]["error"] == "tool_execution_timeout"
    assert result["result"]["hint"] == "tool execution may still be running; do not blindly retry"
    assert result["result"]["meta"]["execution_may_continue"] is True


@pytest.mark.asyncio
async def test_emit_tool_batch_missing_executor():
    runner = _make_runner()
    events = []
    async for event in runner._emit_tool_batch(
        tool_calls=[{"id": "t1", "name": "echo", "arguments": {}, "index_in_iteration": 0}],
        request_id="req1",
        trace_meta={"run_id": "r", "iteration_id": "t", "request_id": "q"},
    ):
        events.append(event)

    assert events
    assert events[0].type == EventType.ERROR


@pytest.mark.asyncio
async def test_emit_tool_batch_empty():
    runner = _make_runner()
    runner.set_tools(_DummyExecutor({"ok": True, "value": "ok"}))
    events = []
    async for event in runner._emit_tool_batch([], "req1", {"run_id": "r", "iteration_id": "t", "request_id": "q"}):
        events.append(event)
    assert events == []


@pytest.mark.asyncio
async def test_emit_tool_batch_success():
    runner = _make_runner()
    runner.set_tools(_DummyExecutor({"ok": True, "value": "ok"}))

    tool_calls = [
        {"id": "t1", "name": "echo", "arguments": {}, "index_in_iteration": 1},
        {"id": "t0", "name": "echo", "arguments": {}, "index_in_iteration": 0},
    ]

    events = []
    async for event in runner._emit_tool_batch(
        tool_calls=tool_calls,
        request_id="req1",
        trace_meta={"run_id": "r", "iteration_id": "t", "request_id": "q"},
    ):
        events.append(event.type)

    assert EventType.TOOL_CALL_DISPATCHED in events
    assert EventType.TOOL_CALLS_BATCH_READY in events
    assert EventType.TOOL_CALL_RESULT in events
    assert EventType.TOOL_CALLS_BATCH_DONE in events


@pytest.mark.asyncio
async def test_emit_tool_batch_aborts_quickly_on_cancellation():
    """取消令牌触发后，_emit_tool_batch 应在很短时间内抛出 EngineCancelledError，不等待工具 timeout。"""

    from dayu.contracts.cancellation import CancelledError as EngineCancelledError, CancellationToken

    class _SlowExecutor:
        def get_schemas(self):
            return []

        def execute(self, name, arguments, context=None):
            del name, arguments, context
            # 模拟远大于测试预算的工具耗时
            import time
            time.sleep(5.0)
            return {"ok": True, "value": "late"}

        def clear_cursors(self) -> None:
            pass

        def get_dup_call_spec(self, name: str):
            del name
            return None

        def get_execution_context_param_name(self, name: str) -> str | None:
            del name
            return None

        def get_tool_display_info(self, name: str) -> tuple[str, list[str] | None]:
            return name, None

        def register_response_middleware(self, callback) -> None:
            del callback

    token = CancellationToken()
    runner = _make_runner()
    runner.cancellation_token = token
    runner.set_tools(cast(ToolExecutor, _SlowExecutor()))

    async def cancel_soon() -> None:
        await asyncio.sleep(0.05)
        token.cancel()

    tool_calls = [
        {"id": "t1", "name": "slow", "arguments": {}, "index_in_iteration": 0},
    ]

    async def drain() -> list[Any]:
        collected: list[Any] = []
        async for event in runner._emit_tool_batch(
            tool_calls=tool_calls,
            request_id="req1",
            trace_meta={"run_id": "r", "iteration_id": "t", "request_id": "q"},
        ):
            collected.append(event)
        return collected

    start = asyncio.get_event_loop().time()
    cancel_task = asyncio.create_task(cancel_soon())
    with pytest.raises(EngineCancelledError):
        await drain()
    elapsed = asyncio.get_event_loop().time() - start
    await cancel_task
    # 取消路径不等待 tool timeout；留出一定余量但远小于 5 秒。
    assert elapsed < 2.0, f"cancellation response too slow: {elapsed}s"


# ---------------------------------------------------------------------------
# _compact_args 测试
# ---------------------------------------------------------------------------

class TestCompactArgs:
    """_compact_args 辅助函数单元测试"""

    def test_basic_dict(self):
        """普通参数格式化为 key=val 形式"""
        result = aor._compact_args({"ticker": "V", "form_type": "10-K"})
        assert "ticker='V'" in result
        assert "form_type='10-K'" in result

    def test_none_values_filtered(self):
        """None 值应被过滤"""
        result = aor._compact_args({"ticker": "V", "form_type": None})
        assert "None" not in result
        assert "form_type" not in result

    def test_long_string_truncated(self):
        """超长字符串应被截断并附加省略号"""
        long_val = "a" * 60
        result = aor._compact_args({"key": long_val})
        assert "…" in result
        assert len(result) < 60 + 20  # 有截断

    def test_total_length_capped(self):
        """整体输出超过 max_len 时截断"""
        args = {f"k{i}": f"value_{i}" for i in range(20)}
        result = aor._compact_args(args, max_len=50)
        assert len(result) <= 51  # 50 + 1 省略号

    def test_non_dict_input(self):
        """非 dict 输入直接转字符串"""
        result = aor._compact_args("raw_string")
        assert result == "raw_string"

    def test_empty_dict(self):
        """空 dict 返回空字符串"""
        result = aor._compact_args({})
        assert result == ""

    def test_int_value(self):
        """整数值正确格式化"""
        result = aor._compact_args({"count": 5})
        assert "count=5" in result


# ---------------------------------------------------------------------------
# _result_summary 测试
# ---------------------------------------------------------------------------

class TestResultSummary:
    """_result_summary 辅助函数单元测试"""

    def test_success_with_json_value(self):
        """成功：从 value 提取基本类型字段"""
        result = aor._result_summary({
            "ok": True,
            "value": {"ticker": "V", "ref": "s_0001"},
        })
        assert "ticker='V'" in result
        assert "ref='s_0001'" in result

    def test_success_truncated_true(self):
        """成功且被截断时，truncated=True 出现在摘要中"""
        result = aor._result_summary({
            "ok": True,
            "value": {},
            "truncation": {"reason": "text_chars"},
        })
        assert "truncated=True" in result

    def test_success_list_and_dict_fields_skipped(self):
        """value 中的 list/dict 字段应被跳过"""
        result = aor._result_summary({
            "ok": True,
            "value": {"ticker": "V", "documents": [1, 2, 3], "nested": {"x": 1}},
        })
        assert "documents" not in result
        assert "nested" not in result
        assert "ticker='V'" in result

    def test_success_no_data_value(self):
        """成功但 value 非 dict 时返回 content 摘要。"""
        result = aor._result_summary({"ok": True, "value": "hello"})
        assert result == "content='hello'"

    def test_success_max_fields(self):
        """超过 4 个字段时只显示前 4 个"""
        value = {f"k{i}": f"v{i}" for i in range(10)}
        result = aor._result_summary({
            "ok": True,
            "value": value,
        })
        # 最多 4 个 value 字段
        assert result.count("=") <= 4

    def test_failure_with_code_and_message(self):
        """失败：返回 error_code: message"""
        result = aor._result_summary({
            "ok": False,
            "error": "cursor_not_found",
            "message": "cursor not found",
        })
        assert result == "cursor_not_found: cursor not found"

    def test_business_failure_with_ok_false(self):
        """业务层 ok=false 时，输出失败摘要。"""
        result = aor._result_summary(
            {
                "ok": False,
                "error": "permission_denied",
                "message": "URL is blocked by fetch safety policy.",
            }
        )
        assert result == "permission_denied: URL is blocked by fetch safety policy."

    def test_failure_code_only(self):
        """失败：仅有 code 时不带冒号"""
        result = aor._result_summary({
            "ok": False,
            "error": "tool_execution_timeout",
        })
        assert result == "tool_execution_timeout"

    def test_long_string_value_truncated(self):
        """value 中超长字符串应被截断"""
        long_val = "x" * 100
        result = aor._result_summary({
            "ok": True,
            "value": {"content": long_val},
        })
        assert "…" in result

    def test_bool_field_no_quotes(self):
        """bool 字段不加引号显示"""
        result = aor._result_summary({
            "ok": True,
            "value": {"has_page_info": False},
        })
        assert "has_page_info=False" in result
