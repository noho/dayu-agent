"""truncation_manager 模块单元测试。

本文件聚焦 `TruncationManager` 的低频分支与异常路径，
以提升截断与游标续读相关代码的分支覆盖率。
"""

from __future__ import annotations

from typing import Any, Dict, List, cast

import pytest

from dayu.contracts.protocols import ToolExecutionContext
from dayu.engine.tool_contracts import TRUNCATION_STRATEGIES, ToolTruncateSpec
from dayu.engine.truncation_manager import TruncationManager


class _DeterministicUUID:
    """用于提供可预测 UUID 的辅助对象。"""

    hex = "fixed_cursor_id"


def _build_manager() -> TruncationManager:
    """创建截断管理器实例。"""
    return TruncationManager()


def _build_spec(strategy: str, limit_key: str, limit: int) -> ToolTruncateSpec:
    """创建合法截断配置。

    Args:
        strategy: 截断策略名称。
        limit_key: 策略对应的限制字段名。
        limit: 限制值。

    Returns:
        ToolTruncateSpec: 可直接用于 `apply_truncation` 的配置对象。
    """
    return ToolTruncateSpec(enabled=True, strategy=strategy, limits={limit_key: limit})


# ------------------------------------------------------------------
# apply_truncation — 目标提取失败 / 未知策略
# ------------------------------------------------------------------

def test_apply_truncation_returns_original_when_target_extraction_fails() -> None:
    """验证不同策略在提取不到目标时返回原值且无截断信息。"""
    manager = _build_manager()
    args = {"q": "demo"}

    # text_chars 策略，传入无文本字段的 dict → 提取失败
    text_spec = _build_spec("text_chars", "max_chars", 5)
    raw_text_miss = {"k": 1}
    result_text, trunc_text = manager.apply_truncation("tool", args, raw_text_miss, None, text_spec)

    # list_items 策略，传入纯字符串 → 提取失败
    list_spec = _build_spec("list_items", "max_items", 1)
    raw_list_miss = "abc"
    result_list, trunc_list = manager.apply_truncation("tool", args, raw_list_miss, None, list_spec)

    # binary_bytes 策略，传入非 bytes 的 dict → 提取失败
    binary_spec = _build_spec("binary_bytes", "max_bytes", 1)
    raw_binary_miss = {"k": 1}
    result_binary, trunc_binary = manager.apply_truncation("tool", args, raw_binary_miss, None, binary_spec)

    assert result_text == raw_text_miss and trunc_text is None
    assert result_list == raw_list_miss and trunc_list is None
    assert result_binary == raw_binary_miss and trunc_binary is None


def test_apply_truncation_returns_original_for_unhandled_strategy(monkeypatch: Any) -> None:
    """验证新增但未实现的策略会走兜底分支返回原结果。"""
    manager = _build_manager()
    monkeypatch.setitem(
        TRUNCATION_STRATEGIES,
        "noop_strategy",
        {"limit_key": "max_items", "unit": "items", "reason": "noop"},
    )

    spec = _build_spec("noop_strategy", "max_items", 1)
    raw = {"rows": [1, 2, 3]}

    result, truncation = manager.apply_truncation("tool", {}, raw, None, spec)

    assert result == raw
    assert truncation is None


# ------------------------------------------------------------------
# _extract_text_target
# ------------------------------------------------------------------

def test_extract_text_target_covers_non_text_and_invalid_paths(monkeypatch: Any) -> None:
    """验证文本目标提取中非字符串和兜底分支。"""
    manager = _build_manager()

    # 传入 bytes（非 str、非 dict）→ 返回 None
    non_text = manager._extract_text_target(b"abc")
    assert non_text == (None, None, None)

    # dict 中字段值非 str → 返回 None
    monkeypatch.setattr(manager, "_select_largest_text_field", lambda value: "field")
    invalid_text = manager._extract_text_target({"field": 123})
    assert invalid_text == (None, None, None)

    # 纯字符串 → 直接返回
    text_value = manager._extract_text_target("hello")
    assert text_value == ("hello", None, None)

    # 传入 int（非 str、非 dict）→ 返回 None
    unsupported_value = manager._extract_text_target(42)
    assert unsupported_value == (None, None, None)


def test_extract_text_target_uses_explicit_target_field() -> None:
    """验证 _extract_text_target 优先使用 target_field 而非启发式选择。"""
    manager = _build_manager()
    value = {"short": "x", "content": "long text here"}

    # 不指定 target_field → 启发式选最长
    text_h, _, path_h = manager._extract_text_target(value)
    assert path_h == ["content"]

    # 指定 target_field → 即使短也用它
    text_t, _, path_t = manager._extract_text_target(value, target_field="short")
    assert path_t == ["short"]
    assert text_t == "x"


def test_extract_text_target_fallback_when_target_field_missing() -> None:
    """验证 target_field 不在 value 中时回退到启发式。"""
    manager = _build_manager()
    value = {"content": "hello"}

    text, _, path = manager._extract_text_target(value, target_field="nonexistent")
    assert path == ["content"]
    assert text == "hello"


# ------------------------------------------------------------------
# _extract_list_target
# ------------------------------------------------------------------

def test_extract_list_target_covers_non_list_and_write_fail_paths(monkeypatch: Any) -> None:
    """验证列表目标提取的非列表与写入失败分支。"""
    manager = _build_manager()

    # 纯字符串 → 返回 None
    non_list = manager._extract_list_target("x")
    assert non_list == (None, None, None)

    # dict 中字段非列表 → 返回 None
    monkeypatch.setattr(manager, "_select_largest_list_path", lambda value: ["a"])
    non_list2 = manager._extract_list_target({"a": "not_list"})
    assert non_list2 == (None, None, None)

    # write_nested_dict_value 失败 → 返回 None
    monkeypatch.setattr(manager, "_write_nested_dict_value", lambda value, path, data: False)
    write_fail = manager._extract_list_target({"a": [1, 2]})
    assert write_fail == (None, None, None)

    # 标量 → 返回 None
    scalar = manager._extract_list_target(1)
    assert scalar == (None, None, None)


def test_extract_list_target_uses_explicit_target_field() -> None:
    """验证 _extract_list_target 优先使用 target_field。"""
    manager = _build_manager()
    value = {"small": [1], "big": [1, 2, 3, 4, 5]}

    # 不指定 → 选最大
    _, _, path_h = manager._extract_list_target(value)
    assert path_h == ["big"]

    # 指定 small
    items, _, path_t = manager._extract_list_target(value, target_field="small")
    assert path_t == ["small"]
    assert items == [1]


def test_extract_list_target_fallback_when_target_field_not_list() -> None:
    """验证 target_field 指向非列表字段时回退到启发式。"""
    manager = _build_manager()
    value = {"name": "str", "items": [1, 2]}

    _, _, path = manager._extract_list_target(value, target_field="name")
    assert path == ["items"]


# ------------------------------------------------------------------
# _extract_binary_target
# ------------------------------------------------------------------

def test_extract_binary_target_covers_non_bytes_paths() -> None:
    """验证二进制目标提取中的非 bytes 分支。"""
    manager = _build_manager()

    # bytes → 正常返回
    data, _, _ = manager._extract_binary_target(b"abc")
    assert data == b"abc"

    # bytearray → 正常返回
    data2, _, _ = manager._extract_binary_target(bytearray(b"xy"))
    assert data2 == b"xy"

    # 非 bytes → 返回 None
    data3, _, _ = manager._extract_binary_target("abc")
    assert data3 is None

    # int → 返回 None
    data4, _, _ = manager._extract_binary_target(123)
    assert data4 is None


# ------------------------------------------------------------------
# 嵌套路径辅助
# ------------------------------------------------------------------

def test_nested_path_helpers_cover_non_string_keys_and_failure_paths() -> None:
    """验证嵌套路径辅助方法的低频失败分支。"""
    manager = _build_manager()

    selected_path = manager._select_largest_list_path(cast(Any, {1: [1, 2, 3], "outer": {"inner": [1]}}))
    assert selected_path == ["outer", "inner"]

    assert manager._read_nested_dict_value({"a": [1]}, ["a", "b"]) is None

    assert manager._write_nested_dict_value({"a": 1}, [], "x") is False
    assert manager._write_nested_dict_value({"a": 1}, ["a", "b", "c"], "x") is False
    assert manager._write_nested_dict_value({"a": 1}, ["a", "b"], "x") is False


# ------------------------------------------------------------------
# 截断不超限
# ------------------------------------------------------------------

def test_truncate_text_lines_and_binary_return_original_when_not_exceeding_limit() -> None:
    """验证 text_lines 与 binary_bytes 在未超限时直接返回原值。"""
    manager = _build_manager()

    lines_value, lines_trunc = manager._truncate_text_lines(
        text="a\nb\n",
        limit=3,
        template=None,
        field_path=None,
        context=None,
        scope_hash="scope",
    )
    assert lines_value == "a\nb\n"
    assert lines_trunc is None

    binary_value, binary_trunc = manager._truncate_binary_bytes(
        data=b"ab",
        limit=3,
        template=None,
        field_path=None,
        context=None,
        scope_hash="scope",
    )
    assert binary_value == b"ab"
    assert binary_trunc is None


# ------------------------------------------------------------------
# _apply_chunk_to_template
# ------------------------------------------------------------------

def test_apply_chunk_to_template_returns_template_when_middle_node_not_dict() -> None:
    """验证模板中间节点非 dict 时会返回模板副本而非写入 chunk。"""
    manager = _build_manager()
    template = {"a": "leaf"}

    output = manager._apply_chunk_to_template(template, ["a", "b", "c"], "chunk")

    assert output == {"a": "leaf"}
    assert output is not template


# ------------------------------------------------------------------
# scope hash 编码
# ------------------------------------------------------------------

def test_build_scope_hash_uses_repr_fallback_for_circular_arguments() -> None:
    """验证循环引用参数会触发 scope hash 的 repr 兜底编码。"""
    manager = _build_manager()
    circular: Dict[str, Any] = {}
    circular["self"] = circular

    value = manager._build_scope_hash("tool", circular)

    assert isinstance(value, str)
    assert len(value) == 64


# ------------------------------------------------------------------
# 游标存储与过期
# ------------------------------------------------------------------

def test_store_cursor_uses_context_timeout_and_cleans_expired_entries(monkeypatch: Any) -> None:
    """验证游标存储会使用 context.timeout 并清理过期游标。"""
    manager = _build_manager()
    manager._cursor_store["expired"] = {"expires_at": 50.0}

    monkeypatch.setattr("dayu.engine.truncation_manager.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("dayu.engine.truncation_manager.uuid.uuid4", lambda: _DeterministicUUID())

    cursor = manager._store_cursor(
        tool_name="tool",
        scope_hash="scope",
        reason="max_items",
        unit="items",
        limit=2,
        total=5,
        data=[1, 2, 3],
        offset=0,
        template=None,
        field_path=None,
        mode="list",
        context=ToolExecutionContext(
            run_id="r1",
            iteration_id="i1",
            tool_call_id="c1",
            timeout_seconds=10,
        ),
    )

    assert cursor == "fixed_cursor_id"
    assert "expired" not in manager._cursor_store
    assert manager._cursor_store[cursor]["expires_at"] == 110.0
    assert manager._cursor_store[cursor]["iteration_id"] == "i1"


# ------------------------------------------------------------------
# _build_chunk
# ------------------------------------------------------------------

def test_build_chunk_covers_text_lines_and_binary_modes() -> None:
    """验证 `_build_chunk` 的 text_lines 与 binary 分支返回值。"""
    manager = _build_manager()

    text_chunk, text_size = manager._build_chunk(mode="text_lines", data=["a\n", "b\n"], offset=0, limit=1)
    binary_chunk, binary_size = manager._build_chunk(mode="binary", data=b"abcd", offset=1, limit=2)

    assert text_chunk == "a\n"
    assert text_size == 1
    assert binary_chunk == b"bc"
    assert binary_size == 2


# ------------------------------------------------------------------
# 清理
# ------------------------------------------------------------------

def test_cleanup_expired_cursors_removes_entries() -> None:
    """验证显式清理函数会删除已过期游标。"""
    manager = _build_manager()
    manager._cursor_store = {
        "old": {"expires_at": 10.0},
        "new": {"expires_at": 200.0},
    }

    manager._cleanup_expired_cursors(now=100.0)

    assert "old" not in manager._cursor_store
    assert "new" in manager._cursor_store


# ------------------------------------------------------------------
# build_truncation_info
# ------------------------------------------------------------------

def test_build_truncation_info_contains_fetch_more_args_and_continuation_hints() -> None:
    """验证截断信息包含续读参数与强续读提示字段。"""
    manager = _build_manager()

    info = manager._build_truncation_info(
        cursor="c1",
        reason="max_chars",
        limit=10,
        unit="chars",
        total=100,
        has_more=True,
        scope_token="s1",
    )

    assert info["fetch_more_args"] == {"cursor": "c1", "scope_token": "s1"}
    assert info["continuation_required"] is True
    assert info["continuation_priority"] == "high"
    assert info["next_action"] == "fetch_more"


# ------------------------------------------------------------------
# clear_cursors
# ------------------------------------------------------------------

def test_clear_cursors_removes_all_entries() -> None:
    """验证 clear_cursors 清空全部游标记录。"""
    manager = _build_manager()
    manager._cursor_store["a"] = {"expires_at": 9999.0}
    manager._cursor_store["b"] = {"expires_at": 9999.0}

    manager.clear_cursors()

    assert len(manager._cursor_store) == 0


# ------------------------------------------------------------------
# fetch_more 续读（新信封格式）
# ------------------------------------------------------------------

def test_fetch_more_returns_new_cursor_when_has_more() -> None:
    """验证 fetch_more 在 has_more=True 时返回新 cursor，旧 cursor 失效。"""
    manager = _build_manager()
    # data=[1,2,3,4,5,6], limit=2 → 首次截断返回 [1,2]，游标 offset=2
    spec = _build_spec("list_items", "max_items", 2)
    raw = [1, 2, 3, 4, 5, 6]
    _, trunc = manager.apply_truncation("t", {}, raw, None, spec)
    assert trunc is not None
    cursor_a = trunc["cursor"]
    scope_a = trunc["fetch_more_args"]["scope_token"]

    # fetch: offset 2→4, still has_more (total=6)
    result = manager.execute_fetch_more(
        {"cursor": cursor_a, "scope_token": scope_a}, context=None,
    )
    assert result["ok"] is True
    assert result.get("truncation") is not None
    cursor_b = result["truncation"]["cursor"]
    scope_b = result["truncation"]["fetch_more_args"]["scope_token"]
    # 新旧 cursor 不同
    assert cursor_b != cursor_a
    # 旧 cursor 已失效
    old_result = manager.execute_fetch_more(
        {"cursor": cursor_a, "scope_token": scope_a}, context=None,
    )
    assert old_result["ok"] is False
    assert old_result["error"] == "cursor_not_found"
    # 新 cursor 可继续
    final = manager.execute_fetch_more(
        {"cursor": cursor_b, "scope_token": scope_b}, context=None,
    )
    assert final["ok"] is True
    assert final.get("truncation") is None
    assert final["value"] == [5, 6]
