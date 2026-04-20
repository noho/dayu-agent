"""
ToolRegistry 测试 — 适配单层信封格式
"""
from pathlib import Path
from typing import Any, cast

import pytest

from dayu.contracts.protocols import ToolExecutionContext
from dayu.engine import ToolRegistry, ConfigError
from dayu.engine.tool_contracts import ToolTruncateSpec
from dayu.engine.tool_errors import ToolBusinessError
from dayu.engine.tool_result import is_tool_success, validate_tool_result_contract


def _simple_schema(name: str):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "test",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    }


def _attach_tool_extra(func: object, *, file_path_params: list[str] | None, truncate: object) -> None:
    """在测试边界为工具函数挂载 `__tool_extra__`。"""

    cast(Any, func).__tool_extra__ = type(
        "ToolExtra",
        (),
        {"__file_path_params__": file_path_params, "__truncate__": truncate},
    )()


def test_register_allowed_paths_accepts_dir_and_file(tmp_path: Path):
    test_dir = tmp_path / "docs"
    test_dir.mkdir()
    test_file = tmp_path / "one.txt"
    test_file.write_text("hi", encoding="utf-8")

    registry = ToolRegistry()
    registry.register_allowed_paths([test_dir, test_file])

    allowed = registry.get_allowed_paths()
    assert str(test_dir.resolve()) in allowed
    assert str(test_file.resolve()) in allowed


def test_register_allowed_paths_missing_path_raises(tmp_path: Path):
    missing = tmp_path / "missing"
    registry = ToolRegistry()
    with pytest.raises(ConfigError):
        registry.register_allowed_paths([missing])


def test_execute_auto_path_validation_success(tmp_path: Path):
    test_dir = tmp_path / "docs"
    test_dir.mkdir()
    test_file = test_dir / "ok.txt"
    test_file.write_text("ok", encoding="utf-8")

    registry = ToolRegistry()
    registry.register_allowed_paths([test_dir])

    def read_tool(file_path: str):
        return {"file_path": file_path}

    _attach_tool_extra(read_tool, file_path_params=["file_path"], truncate=None)
    registry.register("read_tool", read_tool, _simple_schema("read_tool"))

    result = registry.execute("read_tool", {"file_path": str(test_file)})
    assert result["ok"] is True
    assert result["value"]["file_path"] == str(test_file.resolve())


def test_execute_auto_path_validation_denied(tmp_path: Path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("no", encoding="utf-8")

    registry = ToolRegistry()
    registry.register_allowed_paths([allowed_dir])

    def read_tool(file_path: str):
        return {"file_path": file_path}

    _attach_tool_extra(read_tool, file_path_params=["file_path"], truncate=None)
    registry.register("read_tool", read_tool, _simple_schema("read_tool"))

    result = registry.execute("read_tool", {"file_path": str(outside_file)})
    assert result["ok"] is False
    assert result["error"] == "permission_denied"


def test_execute_rejects_file_tool_without_allowed_paths(tmp_path: Path):
    """验证声明路径参数的工具在未注册白名单时会 fail-closed。"""
    test_file = tmp_path / "data.txt"
    test_file.write_text("secret", encoding="utf-8")

    registry = ToolRegistry()

    def read_tool(file_path: str):
        return {"file_path": file_path}

    _attach_tool_extra(read_tool, file_path_params=["file_path"], truncate=None)
    registry.register("read_tool", read_tool, _simple_schema("read_tool"))

    result = registry.execute("read_tool", {"file_path": str(test_file)})
    assert result["ok"] is False
    assert result["error"] == "permission_denied"
    assert "未配置路径白名单" in result["message"]


def test_execute_logs_business_error_with_explicit_url(monkeypatch: pytest.MonkeyPatch):
    """验证业务错误日志会优先显式打印完整 URL。"""

    registry = ToolRegistry()
    captured: dict[str, str] = {}

    def _capture_warn(message: str, *, module: str = "APP") -> None:
        captured["message"] = message
        captured["module"] = module

    def fetch_tool() -> dict[str, str]:
        raise ToolBusinessError(
            "http_error",
            "HTTPSConnectionPool(host='www.21jingji.com', port=443): Max retries exceeded with url: /article/demo.html",
            url="https://www.21jingji.com/article/demo.html",
        )

    monkeypatch.setattr("dayu.engine.tool_registry.Log.warn", _capture_warn)
    registry.register(
        "fetch_web_page",
        fetch_tool,
        {
            "type": "function",
            "function": {
                "name": "fetch_web_page",
                "description": "test",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    )

    result = registry.execute("fetch_web_page", {})

    assert result["ok"] is False
    assert result["error"] == "http_error"
    assert captured["module"] == "ENGINE.TOOL_REGISTRY"
    assert "url=https://www.21jingji.com/article/demo.html" in captured["message"]


def test_apply_truncation_text_lines():
    registry = ToolRegistry()
    value = "a\n\nb\n\nc\n"
    spec = ToolTruncateSpec(enabled=True, strategy="text_lines", limits={"max_lines": 1})

    result, trunc = registry._truncation_manager.apply_truncation(
        name="demo",
        arguments={},
        value=value,
        context=None,
        truncate_spec=spec,
    )

    assert trunc is not None
    assert trunc["fetch_more_args"]["cursor"] == trunc["cursor"]
    assert trunc["continuation_required"] is True
    assert trunc["continuation_priority"] == "high"
    assert trunc["next_action"] == "fetch_more"
    assert isinstance(result, str)
    assert result.strip() == "a"


def test_apply_truncation_list_and_fetch_more():
    registry = ToolRegistry()
    value = [1, 2, 3]
    spec = ToolTruncateSpec(enabled=True, strategy="list_items", limits={"max_items": 2})

    result, trunc = registry._truncation_manager.apply_truncation(
        name="demo",
        arguments={},
        value=value,
        context=None,
        truncate_spec=spec,
    )

    assert trunc is not None
    assert trunc["fetch_more_args"]["cursor"] == trunc["cursor"]
    assert trunc["fetch_more_args"]["scope_token"]
    assert trunc["continuation_required"] is True
    assert trunc["next_action"] == "fetch_more"
    assert result == [1, 2]

    fetch = registry._truncation_manager.execute_fetch_more(
        {
            "cursor": trunc["cursor"],
            "scope_token": trunc["fetch_more_args"]["scope_token"],
            "limit": 2,
        },
        context=None,
    )
    assert fetch["ok"] is True
    assert fetch.get("truncation") is None
    assert fetch["value"] == [3]


def test_apply_truncation_nested_list_field_and_fetch_more():
    registry = ToolRegistry()
    value = {
        "table_ref": "t_0001",
        "data": {
            "kind": "records",
            "rows": [{"id": 1}, {"id": 2}, {"id": 3}],
        },
    }
    spec = ToolTruncateSpec(enabled=True, strategy="list_items", limits={"max_items": 2})

    result, trunc = registry._truncation_manager.apply_truncation(
        name="demo",
        arguments={},
        value=value,
        context=None,
        truncate_spec=spec,
    )

    assert trunc is not None
    assert trunc["fetch_more_args"]["cursor"] == trunc["cursor"]
    assert trunc["fetch_more_args"]["scope_token"]
    assert trunc["continuation_required"] is True
    assert trunc["next_action"] == "fetch_more"
    assert result["data"]["rows"] == [{"id": 1}, {"id": 2}]

    fetch = registry._truncation_manager.execute_fetch_more(
        {
            "cursor": trunc["cursor"],
            "scope_token": trunc["fetch_more_args"]["scope_token"],
            "limit": 2,
        },
        context=None,
    )
    assert fetch["ok"] is True
    assert fetch.get("truncation") is None
    assert fetch["value"]["data"]["rows"] == [{"id": 3}]


def test_apply_truncation_binary_bytes():
    registry = ToolRegistry()
    value = b"abcd"
    spec = ToolTruncateSpec(enabled=True, strategy="binary_bytes", limits={"max_bytes": 2})

    result, trunc = registry._truncation_manager.apply_truncation(
        name="demo",
        arguments={},
        value=value,
        context=None,
        truncate_spec=spec,
    )

    assert trunc is not None
    assert trunc["fetch_more_args"]["cursor"] == trunc["cursor"]
    assert trunc["fetch_more_args"]["scope_token"]
    assert trunc["continuation_required"] is True
    assert trunc["next_action"] == "fetch_more"
    assert isinstance(result, bytes)
    assert result == b"ab"


def test_execute_fetch_more_error_paths():
    registry = ToolRegistry()

    r1 = registry._truncation_manager.execute_fetch_more({"cursor": ""}, context=None)
    assert r1["error"] == "invalid_cursor"

    r2 = registry._truncation_manager.execute_fetch_more({"cursor": "missing"}, context=None)
    assert r2["error"] == "cursor_not_found"

    cursor = registry._truncation_manager._store_cursor(
        tool_name="text",
        scope_hash="scope",
        reason="max_chars",
        unit="chars",
        limit=1,
        total=2,
        data="ab",
        offset=1,
        template=None,
        field_path=None,
        mode="text",
        context=None,
    )
    registry._truncation_manager._cursor_store[cursor]["expires_at"] = 0
    r3 = registry._truncation_manager.execute_fetch_more({"cursor": cursor}, context=None)
    assert r3["error"] == "cursor_expired"

    cursor2 = registry._truncation_manager._store_cursor(
        tool_name="text",
        scope_hash="scope",
        reason="max_chars",
        unit="chars",
        limit=1,
        total=2,
        data="ab",
        offset=1,
        template=None,
        field_path=None,
        mode="text",
        context=ToolExecutionContext(run_id="r1"),
    )
    scope_token2 = registry._truncation_manager._cursor_store[cursor2]["scope_token"]
    mismatch = registry._truncation_manager.execute_fetch_more(
        {"cursor": cursor2, "scope_token": scope_token2},
        context=ToolExecutionContext(run_id="r2"),
    )
    assert mismatch["error"] == "cursor_scope_mismatch"

    # 同一 run 内跨 iteration 续读应成功
    cross_turn_ok = registry._truncation_manager.execute_fetch_more(
        {"cursor": cursor2, "scope_token": scope_token2},
        context=ToolExecutionContext(run_id="r1", iteration_id="iteration-next"),
    )
    assert cross_turn_ok["ok"] is True

    cursor3 = registry._truncation_manager._store_cursor(
        tool_name="text",
        scope_hash="scope",
        reason="max_chars",
        unit="chars",
        limit=1,
        total=2,
        data="ab",
        offset=1,
        template=None,
        field_path=None,
        mode="text",
        context=None,
    )
    scope_mismatch = registry._truncation_manager.execute_fetch_more({"cursor": cursor3}, context=None)
    assert scope_mismatch["error"] == "cursor_scope_mismatch"


def test_validate_and_coerce_arguments_non_dict():
    registry = ToolRegistry()
    result = registry._argument_validator.validate_and_coerce("bad", None)
    assert result["ok"] is False
    assert result["error"] == "invalid_argument"


def test_calculate_depth_exceeded():
    registry = ToolRegistry()
    av = registry._argument_validator
    deep = current = {}
    for _ in range(av.ARGUMENTS_MAX_DEPTH + 1):
        current["x"] = {}
        current = current["x"]
    result = av.validate_and_coerce(deep, None)
    assert result["ok"] is False
    assert result["meta"]["issues"][0]["reason"] == "depth_exceeded"
    assert "Reduce argument nesting to at most" in result["hint"]
    assert validate_tool_result_contract(result) is None


def test_coerce_value_with_union_and_enum():
    registry = ToolRegistry()
    schema = {
        "type": "object",
        "properties": {
            "mode": {"type": ["string", "integer"], "enum": ["a", 1]},
        },
        "required": ["mode"],
    }
    ok, coerced, issues = registry._argument_validator._coerce_value({"mode": "a"}, schema, path="$")
    assert ok is True
    assert coerced["mode"] == "a"

    ok, coerced, issues = registry._argument_validator._coerce_value({"mode": 1}, schema, path="$")
    assert ok is True
    assert coerced["mode"] == 1

    ok, coerced, issues = registry._argument_validator._coerce_value({"mode": "b"}, schema, path="$")
    assert ok is False
    assert issues[0]["reason"] == "enum_mismatch"


def test_coerce_value_array_and_object_limits():
    registry = ToolRegistry()
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    ok, coerced, issues = registry._argument_validator._coerce_value({"items": [1]}, schema, path="$")
    assert ok is False
    assert issues[0]["reason"] == "array_too_small"

    ok, coerced, issues = registry._argument_validator._coerce_value({"items": [1, 2], "extra": 1}, schema, path="$")
    assert ok is False
    assert issues[0]["reason"] == "additional_properties"


def test_build_scope_hash_non_json():
    registry = ToolRegistry()
    class Bad:
        def __repr__(self):
            return "<bad>"
    value = registry._truncation_manager._build_scope_hash("tool", {"x": Bad()})
    assert isinstance(value, str)
    assert len(value) == 64


def test_apply_chunk_to_template_non_dict_path():
    registry = ToolRegistry()
    template = {"a": None}
    output = registry._truncation_manager._apply_chunk_to_template(template, ["a", "b"], "x")
    assert output == template


def test_check_generic_limits_string_too_long():
    registry = ToolRegistry()
    long_str = "a" * (registry._argument_validator.SCHEMA_MAX_STRING_LENGTH + 1)
    issues = registry._argument_validator._check_generic_limits(long_str, path="$")
    assert issues[0]["reason"] == "string_too_long"


def test_check_generic_limits_array_too_large():
    registry = ToolRegistry()
    long_list = [0] * (registry._argument_validator.SCHEMA_MAX_ARRAY_ITEMS + 1)
    issues = registry._argument_validator._check_generic_limits(long_list, path="$")
    assert issues[0]["reason"] == "array_too_large"


def test_coerce_value_for_type_string_bounds():
    registry = ToolRegistry()
    schema = {"type": "string", "minLength": 2, "maxLength": 3}
    ok, _, issues = registry._argument_validator._coerce_value_for_type("a", schema, path="$")
    assert ok is False
    assert issues[0]["reason"] == "string_too_short"
    ok, _, issues = registry._argument_validator._coerce_value_for_type("abcd", schema, path="$")
    assert ok is False
    assert issues[0]["reason"] == "string_too_long"


def test_coerce_value_for_type_number_and_boolean():
    registry = ToolRegistry()
    ok, coerced, _ = registry._argument_validator._coerce_value_for_type("1.5", {"type": "number"}, path="$")
    assert ok is True
    assert coerced == 1.5

    ok, coerced, _ = registry._argument_validator._coerce_value_for_type("true", {"type": "boolean"}, path="$")
    assert ok is True
    assert coerced is True

    ok, _, issues = registry._argument_validator._coerce_value_for_type(2, {"type": "boolean"}, path="$")
    assert ok is False
    assert issues[0]["reason"] == "type_mismatch"


def test_execute_fetch_more_chunk_size_zero():
    registry = ToolRegistry()
    cursor = registry._truncation_manager._store_cursor(
        tool_name="text",
        scope_hash="scope",
        reason="max_chars",
        unit="chars",
        limit=1,
        total=1,
        data="a",
        offset=1,
        template=None,
        field_path=None,
        mode="text",
        context=None,
    )
    scope_token = registry._truncation_manager._cursor_store[cursor]["scope_token"]
    result = registry._truncation_manager.execute_fetch_more(
        {"cursor": cursor, "scope_token": scope_token},
        context=None,
    )
    assert result["ok"] is True
    assert result.get("truncation") is None


def test_extract_targets():
    """验证新 _extract_*_target 方法的核心路径。"""
    registry = ToolRegistry()

    # 文本提取：纯字符串
    text, template, field_path = registry._truncation_manager._extract_text_target("hello")
    assert text == "hello"
    assert template is None
    assert field_path is None

    # 文本提取：字典中最长字段
    text, template, field_path = registry._truncation_manager._extract_text_target({"a": "short", "b": "much longer"})
    assert text == "much longer"
    assert field_path == ["b"]
    assert template is not None
    assert template["b"] is None

    # 列表提取：顶级列表
    items, template, field_path = registry._truncation_manager._extract_list_target([1, 2, 3])
    assert items == [1, 2, 3]
    assert template is None

    # 列表提取：字典中最大列表
    items, template, field_path = registry._truncation_manager._extract_list_target({"x": [1], "y": [1, 2, 3]})
    assert items == [1, 2, 3]
    assert field_path == ["y"]

    # 二进制提取：bytes
    data, template, field_path = registry._truncation_manager._extract_binary_target(b"abc")
    assert data == b"abc"
    assert template is None

    # 二进制提取：非 bytes 返回 None
    data, template, field_path = registry._truncation_manager._extract_binary_target("not bytes")
    assert data is None


def test_apply_chunk_to_template_none_paths():
    registry = ToolRegistry()
    assert registry._truncation_manager._apply_chunk_to_template(None, None, "x") == "x"
    assert registry._truncation_manager._apply_chunk_to_template({"a": None}, None, "x") == "x"


def test_resolve_fetch_limit():
    registry = ToolRegistry()
    assert registry._truncation_manager._resolve_fetch_limit(None, 3) == 3
    assert registry._truncation_manager._resolve_fetch_limit(0, 3) == 3


def test_apply_truncation_text_chars_selects_largest_field():
    registry = ToolRegistry()
    value = {"short": "ab", "long": "abcdefghij"}
    spec = ToolTruncateSpec(enabled=True, strategy="text_chars", limits={"max_chars": 5})

    result, trunc = registry._truncation_manager.apply_truncation(
        name="demo",
        arguments={},
        value=value,
        context=None,
        truncate_spec=spec,
    )

    assert trunc is not None
    assert result["long"] == "abcde"
    assert result["short"] == "ab"


class TestToolRegistryBoundary:
    def test_register_duplicate_tool_overwrites(self):
        """验证重复注册同名工具会覆盖（带 warning），不抛异常。"""
        registry = ToolRegistry()
        schema = {"type": "function", "function": {"name": "foo", "description": "test", "parameters": {"type": "object", "properties": {}, "required": []}}}
        registry.register("foo", lambda: "v1", schema)
        registry.register("foo", lambda: "v2", schema)
        result = registry.execute("foo", {})
        assert result["ok"] is True
        assert result["value"] == "v2"

    def test_execute_nonexistent_tool(self):
        registry = ToolRegistry()
        result = registry.execute("no_such_tool", {})
        assert result["ok"] is False
        assert result["error"] == "tool_not_found"

    def test_register_with_empty_name(self):
        registry = ToolRegistry()
        schema = {"type": "function", "function": {"name": "", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}}
        with pytest.raises(ConfigError, match="非空字符串"):
            registry.register("", lambda: None, schema)

    def test_register_with_mismatched_name(self):
        registry = ToolRegistry()
        schema = {"type": "function", "function": {"name": "other_name", "description": "", "parameters": {"type": "object", "properties": {}, "required": []}}}
        with pytest.raises(ConfigError, match="不匹配"):
            registry.register("my_name", lambda: None, schema)

    def test_validate_path_with_nonexistent_file(self, tmp_path: Path):
        registry = ToolRegistry()
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        registry.register_allowed_paths([allowed])
        with pytest.raises(FileNotFoundError):
            registry._validate_path(str(allowed / "nonexistent.txt"))

    def test_validate_path_with_symlink_traversal(self, tmp_path: Path):
        registry = ToolRegistry()
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")
        link = allowed / "link.txt"
        link.symlink_to(outside)
        registry.register_allowed_paths([allowed])
        with pytest.raises(PermissionError):
            registry._validate_path(str(link))

    def test_validate_path_with_exact_file_match(self, tmp_path: Path):
        registry = ToolRegistry()
        file = tmp_path / "ok.txt"
        file.write_text("ok", encoding="utf-8")
        registry.register_allowed_paths([file])
        assert registry._validate_path(str(file)) == file.resolve()

    def test_get_tool_schemas_format(self):
        registry = ToolRegistry()
        schema = {"type": "function", "function": {"name": "foo", "description": "d", "parameters": {"type": "object", "properties": {}, "required": []}}}
        registry.register("foo", lambda: None, schema)
        schemas = registry.get_schemas()
        # schemas 包含 foo 和自动注册的 fetch_more
        foo_schemas = [s for s in schemas if s["function"]["name"] == "foo"]
        assert len(foo_schemas) == 1

    def test_get_allowed_paths_empty(self):
        registry = ToolRegistry()
        assert registry.get_allowed_paths() == []

    def test_get_allowed_paths_multiple(self, tmp_path: Path):
        registry = ToolRegistry()
        d1 = tmp_path / "d1"
        d1.mkdir()
        d2 = tmp_path / "d2"
        d2.mkdir()
        registry.register_allowed_paths([d1, d2])
        assert len(registry.get_allowed_paths()) == 2


class TestToolRegistrySchemaValidation:
    """验证 schema 严格校验行为。"""

    def test_register_with_invalid_schema_type(self):
        registry = ToolRegistry()
        with pytest.raises(ConfigError, match="schema 必须是 dict"):
            registry.register("foo", lambda: None, "not a dict")

    def test_register_with_missing_parameters_coerced(self):
        """_coerce_tool_schema 在 function 无 parameters 时默认补空 dict，
        但 _validate_tool_schema 要求 parameters.type=='object'，因此应 raise。"""
        registry = ToolRegistry()
        schema = {"function": {"name": "foo"}}
        with pytest.raises(ConfigError, match="parameters"):
            registry.register("foo", lambda: None, schema)

    def test_register_with_valid_minimal_schema(self):
        """最小合法 schema。"""
        registry = ToolRegistry()
        schema = {
            "type": "function",
            "function": {
                "name": "foo",
                "description": "test",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        registry.register("foo", lambda: None, schema)
        assert registry.schemas["foo"]["function"]["name"] == "foo"

    def test_register_with_invalid_description_type_raises(self):
        registry = ToolRegistry()
        schema = {
            "type": "function",
            "function": {
                "name": "foo",
                "description": 123,
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        with pytest.raises(ConfigError, match="description"):
            registry.register("foo", lambda: None, schema)

    def test_register_with_invalid_parameters_type_raises(self):
        registry = ToolRegistry()
        schema = {"type": "function", "function": {"name": "foo", "parameters": "bad"}}
        with pytest.raises(ConfigError, match="parameters"):
            registry.register("foo", lambda: None, schema)

    def test_register_with_missing_properties_raises(self):
        registry = ToolRegistry()
        schema = {"type": "function", "function": {"name": "foo", "parameters": {"type": "object"}}}
        with pytest.raises(ConfigError, match="properties"):
            registry.register("foo", lambda: None, schema)

    def test_register_with_invalid_required_type_raises(self):
        registry = ToolRegistry()
        schema = {
            "type": "function",
            "function": {
                "name": "foo",
                "parameters": {"type": "object", "properties": {}, "required": "not_list"},
            },
        }
        with pytest.raises(ConfigError, match="required"):
            registry.register("foo", lambda: None, schema)

    def test_register_with_valid_required_referencing_missing_property(self):
        """required 引用不存在的 property 目前不校验（由 LLM schema 语义决定）。"""
        registry = ToolRegistry()
        schema = {
            "type": "function",
            "function": {
                "name": "foo",
                "parameters": {"type": "object", "properties": {}, "required": ["missing"]},
            },
        }
        registry.register("foo", lambda: None, schema)
        assert "missing" in registry.schemas["foo"]["function"]["parameters"]["required"]


class TestToolRegistryExecution:
    def _register_tool(self, registry, name="echo", func=None, schema=None):
        if func is None:
            func = lambda **kwargs: kwargs
        if schema is None:
            schema = {"type": "function", "function": {"name": name, "description": "test", "parameters": {"type": "object", "properties": {"msg": {"type": "string"}}}}}
        registry.register(name, func, schema)

    def test_execute_with_tool_exception(self):
        registry = ToolRegistry()
        def bad_tool(**kwargs):
            raise RuntimeError("boom")
        self._register_tool(registry, name="bad_tool", func=bad_tool)
        result = registry.execute("bad_tool", {"msg": "hi"})
        assert result["ok"] is False
        assert result["error"] == "execution_error"
        assert "RuntimeError" in result["message"]

    def test_execute_with_dict_arguments(self):
        registry = ToolRegistry()
        self._register_tool(registry)
        result = registry.execute("echo", {"msg": "test"})
        assert result["ok"] is True
        assert result["value"]["msg"] == "test"

    def test_execute_with_invalid_json_arguments(self):
        registry = ToolRegistry()
        self._register_tool(registry)
        result = registry.execute("echo", cast(Any, "not json"))
        assert result["ok"] is False

    def test_execute_fetch_more_tool(self):
        registry = ToolRegistry()
        def my_tool(**kwargs):
            return [1, 2, 3, 4, 5]

        schema = {"type": "function", "function": {"name": "my_tool", "description": "test", "parameters": {"type": "object", "properties": {}}}}
        _attach_tool_extra(
            my_tool,
            file_path_params=None,
            truncate=ToolTruncateSpec(enabled=True, strategy="list_items", limits={"max_items": 2}),
        )
        registry.register("my_tool", my_tool, schema)

        result = registry.execute("my_tool", {})
        assert result["ok"] is True
        assert result["value"] == [1, 2]
        assert result["truncation"] is not None
        cursor = result["truncation"]["cursor"]
        scope_token = result["truncation"]["fetch_more_args"]["scope_token"]

        # fetch_more
        fetch_schema = {"type": "function", "function": {"name": "fetch_more", "description": "fetch more", "parameters": {"type": "object", "properties": {"cursor": {"type": "string"}, "scope_token": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["cursor", "scope_token"]}}}
        registry.register("fetch_more", lambda: (_ for _ in ()).throw(RuntimeError("should not be called")), fetch_schema)
        fetch_result = registry.execute("fetch_more", {"cursor": cursor, "scope_token": scope_token})
        assert fetch_result["ok"] is True
        assert fetch_result["value"] == [3, 4]
        assert fetch_result.get("truncation") is not None

    def test_execute_with_missing_required_argument(self):
        registry = ToolRegistry()
        schema = {
            "type": "function",
            "function": {
                "name": "strict",
                "description": "test",
                "parameters": {
                    "type": "object",
                    "properties": {"required_param": {"type": "string"}},
                    "required": ["required_param"],
                },
            },
        }
        registry.register("strict", lambda **kwargs: kwargs, schema)
        result = registry.execute("strict", {})
        assert result["ok"] is False

    def test_execute_with_tool_business_error(self):
        """验证 ToolBusinessError 被正确转换为 build_error 信封。"""
        registry = ToolRegistry()
        def biz_tool(**kwargs):
            raise ToolBusinessError(
                code="not_found",
                message="文档不存在",
                hint="请检查 document_id",
            )
        schema = {"type": "function", "function": {"name": "biz_tool", "description": "test", "parameters": {"type": "object", "properties": {}}}}
        registry.register("biz_tool", biz_tool, schema)
        result = registry.execute("biz_tool", {})
        assert result["ok"] is False
        assert result["error"] == "not_found"
        assert result["message"] == "文档不存在"
        assert result["hint"] == "请检查 document_id"


def test_fetch_more_placeholder_raises_runtime_error():
    registry = ToolRegistry()
    with pytest.raises(RuntimeError, match="fetch_more"):
        registry._fetch_more_placeholder(cursor="c", scope_token="s")
