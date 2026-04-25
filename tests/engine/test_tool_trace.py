"""工具调用追踪模块测试（V2）。"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, cast

import pytest

from dayu.engine import tool_trace as tool_trace_module
from dayu.engine.tool_trace import JsonlToolTraceStore, V2ToolTraceRecorder


def _read_trace_records(output_dir: Path) -> list[dict[str, Any]]:
    """读取测试目录中的 trace 记录。

    Args:
        output_dir: trace 输出目录。

    Returns:
        解析后的 JSON 记录列表。

    Raises:
        FileNotFoundError: 未找到 trace 文件时抛出。
        ValueError: JSON 解析失败时抛出。
    """

    trace_files = sorted((output_dir / "sessions").rglob("tool_calls_*.jsonl*"))
    if not trace_files:
        raise FileNotFoundError("trace file not found")
    payloads: list[dict[str, Any]] = []
    for trace_file in trace_files:
        if trace_file.suffix == ".gz":
            raw_text = gzip.decompress(trace_file.read_bytes()).decode("utf-8")
        else:
            raw_text = trace_file.read_text(encoding="utf-8")
        for line in raw_text.splitlines():
            if not line.strip():
                continue
            payloads.append(json.loads(line))
    return payloads


def _build_recorder(
    output_dir: Path,
    *,
    run_id: str = "run_1",
    session_id: str = "sess_1",
) -> tuple[JsonlToolTraceStore, V2ToolTraceRecorder]:
    """构造测试使用的 trace store 与 recorder。

    Args:
        output_dir: trace 输出目录。
        run_id: 运行 ID。
        session_id: 会话 ID。

    Returns:
        ``(store, recorder)`` 二元组。

    Raises:
        无。
    """

    store = JsonlToolTraceStore(output_dir=output_dir)
    recorder = V2ToolTraceRecorder(
        run_id=run_id,
        session_id=session_id,
        store=store,
        agent_metadata={
            "agent_name": "interactive_agent",
            "agent_kind": "scene_agent",
            "scene_name": "interactive",
            "model_name": "mimo-v2.5-pro-thinking",
            "enabled_capabilities": ["fins"],
        },
    )
    return store, recorder


def _raise_append_error(record: dict[str, Any]) -> None:
    """模拟追加写盘失败。

    Args:
        record: 待写入记录。

    Returns:
        无。

    Raises:
        OSError: 始终抛出磁盘异常。
    """

    del record
    raise OSError("disk full")


@pytest.mark.unit
def test_v2_tool_trace_recorder_pairs_dispatched_and_result(tmp_path: Path) -> None:
    """验证请求与返回可正常配对写盘。"""

    output_dir = tmp_path / "trace"
    _, recorder = _build_recorder(output_dir)
    recorder.start_iteration(
        iteration_id="run_1_iteration_1",
        model_input_messages=[{"role": "user", "content": "分析 AAPL"}],
        tool_schemas=[],
    )
    recorder.on_tool_dispatched(
        iteration_id="run_1_iteration_1",
        payload={
            "id": "call_1",
            "index_in_iteration": 0,
            "name": "list_documents",
            "arguments": {"ticker": "AAPL"},
        },
    )
    recorder.on_tool_result(
        iteration_id="run_1_iteration_1",
        payload={
            "id": "call_1",
            "index_in_iteration": 0,
            "name": "list_documents",
            "arguments": {"ticker": "AAPL"},
            "result": {"ok": True, "value": {"ok": 1}},
        },
    )

    records = _read_trace_records(output_dir)
    assert len(records) == 1
    record = records[0]
    assert record["trace_schema_version"] == "tool_trace_v2"
    assert record["trace_type"] == "tool_call"
    assert record["run_id"] == "run_1"
    assert record["session_id"] == "sess_1"
    assert record["iteration_id"] == "run_1_iteration_1"
    assert record["index_in_iteration"] == 0
    assert record["tool_call_id"] == "call_1"
    assert record["tool_name"] == "list_documents"
    assert record["arguments"] == {"ticker": "AAPL"}
    assert record["agent_name"] == "interactive_agent"
    assert record["scene_name"] == "interactive"
    assert record["enabled_capabilities"] == ["fins"]
    assert record["result_fact"]["status"] == "success"
    assert "success" not in record["result_fact"]
    assert record["result_fact"]["raw_result_ref"]["blob_id"].startswith("tool_result_raw:")


@pytest.mark.unit
def test_v2_tool_trace_recorder_marks_business_error_as_failure(tmp_path: Path) -> None:
    """验证外层执行成功但业务层 `ok=false` 时，trace 仍按失败记录。"""

    output_dir = tmp_path / "trace"
    _, recorder = _build_recorder(output_dir, run_id="run_business_error", session_id="sess_business_error")
    result_payload = {
        "ok": False,
        "error": "permission_denied",
        "message": "URL is blocked by fetch safety policy.",
    }
    recorder.on_tool_dispatched(
        iteration_id="run_business_error_iteration_1",
        payload={
            "id": "call_1",
            "index_in_iteration": 0,
            "name": "fetch_web_page",
            "arguments": {"url": "https://www.apple.com/investor"},
        },
    )
    recorder.on_tool_result(
        iteration_id="run_business_error_iteration_1",
        payload={
            "id": "call_1",
            "index_in_iteration": 0,
            "name": "fetch_web_page",
            "arguments": {"url": "https://www.apple.com/investor"},
            "result": result_payload,
        },
    )

    records = _read_trace_records(output_dir)
    record = records[0]
    assert record["result_fact"]["status"] == "error"
    assert "success" not in record["result_fact"]
    assert record["result_fact"]["error_code"] == "permission_denied"


@pytest.mark.unit
def test_v2_tool_trace_recorder_records_iteration_context_and_final_response(tmp_path: Path) -> None:
    """验证 iteration_context_snapshot 与 final_response 可写入。"""

    output_dir = tmp_path / "trace"
    _, recorder = _build_recorder(output_dir, run_id="run_prompt", session_id="sess_prompt")
    recorder.start_iteration(
        iteration_id="run_prompt_iteration_1",
        model_input_messages=[
            {"role": "system", "content": "你是一个财报分析助手。"},
            {"role": "user", "content": "请分析 AAPL 最新季度风险"},
        ],
        tool_schemas=[
            {
                "type": "function",
                "function": {
                    "name": "list_documents",
                    "description": "列出文档",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_section",
                    "description": "读取段落",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ],
    )
    recorder.finish_iteration(iteration_id="run_prompt_iteration_1", iteration_index=1)
    recorder.record_final_response(
        iteration_id="run_prompt_iteration_2",
        content="结论：短期库存风险可控。",
        degraded=False,
    )

    records = _read_trace_records(output_dir)
    assert len(records) == 2
    by_type = {record["trace_type"]: record for record in records}

    iteration_context_record = by_type["iteration_context_snapshot"]
    assert iteration_context_record["run_id"] == "run_prompt"
    assert iteration_context_record["session_id"] == "sess_prompt"
    assert iteration_context_record["iteration_id"] == "run_prompt_iteration_1"
    assert iteration_context_record["iteration_index"] == 1
    assert iteration_context_record["current_user_message"] == "请分析 AAPL 最新季度风险"
    assert iteration_context_record["agent_name"] == "interactive_agent"
    assert iteration_context_record["enabled_capabilities"] == ["fins"]
    assert "raw_input_ref" in iteration_context_record
    assert iteration_context_record["tool_schema_names"] == ["list_documents", "read_section"]
    assert iteration_context_record["raw_tool_schemas_ref"]["blob_id"].startswith("tool_schemas_final:")
    assert iteration_context_record["model_input_messages_summary"]

    final_record = by_type["final_response"]
    assert final_record["run_id"] == "run_prompt"
    assert final_record["iteration_id"] == "run_prompt_iteration_2"
    assert final_record["agent_name"] == "interactive_agent"
    assert final_record["final_response"]["content"] == "结论：短期库存风险可控。"
    assert final_record["final_response"]["degraded"] is False


@pytest.mark.unit
def test_v2_tool_trace_recorder_uses_shorter_excerpt_for_system_messages(tmp_path: Path) -> None:
    """验证系统消息 excerpt 比当前轮用户消息更短。"""

    output_dir = tmp_path / "trace"
    _, recorder = _build_recorder(output_dir, run_id="run_excerpt", session_id="sess_excerpt")
    long_policy = "P" * 160
    long_user = "U" * 220

    recorder.start_iteration(
        iteration_id="run_excerpt_iteration_1",
        model_input_messages=[
            {"role": "system", "content": long_policy},
            {"role": "assistant", "content": "上一轮已经确认分析范围。"},
            {"role": "user", "content": long_user},
        ],
        tool_schemas=[],
    )
    recorder.finish_iteration(iteration_id="run_excerpt_iteration_1", iteration_index=1)

    records = _read_trace_records(output_dir)
    iteration_context_record = next(record for record in records if record["trace_type"] == "iteration_context_snapshot")
    summaries = iteration_context_record["model_input_messages_summary"]
    policy_summary = next(item for item in summaries if item["source_tag"] == "policy")
    current_iteration_summary = next(item for item in summaries if item["source_tag"] == "current_iteration")

    assert len(policy_summary["excerpt"]) == 64
    assert len(current_iteration_summary["excerpt"]) == 200
    assert policy_summary["excerpt"] == long_policy[:64]
    assert current_iteration_summary["excerpt"] == long_user[:200]


@pytest.mark.unit
def test_v2_tool_trace_recorder_supports_result_before_dispatched(tmp_path: Path) -> None:
    """验证先到返回再到请求时仍可正确配对。"""

    output_dir = tmp_path / "trace"
    _, recorder = _build_recorder(output_dir, run_id="run_2", session_id="sess_2")
    recorder.start_iteration(
        iteration_id="run_2_iteration_1",
        model_input_messages=[{"role": "user", "content": "读取 sec_1"}],
        tool_schemas=[],
    )
    recorder.on_tool_result(
        iteration_id="run_2_iteration_1",
        payload={
            "id": "call_2",
            "index_in_iteration": 1,
            "name": "read_section",
            "arguments": {"ref": "sec_1"},
            "result": {"ok": True, "value": {"content": "x"}},
        },
    )
    recorder.on_tool_dispatched(
        iteration_id="run_2_iteration_1",
        payload={
            "id": "call_2",
            "index_in_iteration": 1,
            "name": "read_section",
            "arguments": {"ref": "sec_1"},
        },
    )

    records = _read_trace_records(output_dir)
    assert len(records) == 1
    record = records[0]
    assert record["tool_call_id"] == "call_2"
    assert record["arguments"] == {"ref": "sec_1"}
    assert record["result_fact"]["status"] == "success"
    assert "success" not in record["result_fact"]


@pytest.mark.unit
def test_v2_tool_trace_recorder_close_flushes_unpaired_records(tmp_path: Path) -> None:
    """验证 close 会输出未配对请求与返回的异常记录。"""

    output_dir = tmp_path / "trace"
    _, recorder = _build_recorder(output_dir, run_id="run_3", session_id="sess_3")
    recorder.on_tool_dispatched(
        iteration_id="run_3_iteration_1",
        payload={
            "id": "call_3a",
            "index_in_iteration": 0,
            "name": "get_table",
            "arguments": {"table_ref": "tbl_1"},
        },
    )
    recorder.on_tool_result(
        iteration_id="run_3_iteration_1",
        payload={
            "id": "call_3b",
            "index_in_iteration": 1,
            "name": "get_table",
            "arguments": {"table_ref": "tbl_2"},
            "result": {"ok": True, "value": {"rows": 10}},
        },
    )
    recorder.close()

    records = _read_trace_records(output_dir)
    assert len(records) == 2
    by_call_id = {record["tool_call_id"]: record for record in records}
    assert by_call_id["call_3a"]["result_fact"]["error_code"] == "RESULT_MISSING"
    assert by_call_id["call_3b"]["result_fact"]["error_code"] == "REQUEST_MISSING"
    assert by_call_id["call_3b"]["arguments"] == {"table_ref": "tbl_2"}


@pytest.mark.unit
def test_v2_tool_trace_recorder_swallow_write_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 recorder 写盘失败不会向上抛出异常。"""

    output_dir = tmp_path / "trace"
    store, recorder = _build_recorder(output_dir, run_id="run_4", session_id="sess_4")
    monkeypatch.setattr(store, "append_record", _raise_append_error)

    recorder.on_tool_dispatched(
        iteration_id="run_4_iteration_1",
        payload={
            "id": "call_4",
            "index_in_iteration": 0,
            "name": "search_document",
            "arguments": {"query": "risk"},
        },
    )
    recorder.on_tool_result(
        iteration_id="run_4_iteration_1",
        payload={
            "id": "call_4",
            "index_in_iteration": 0,
            "name": "search_document",
            "arguments": {"query": "risk"},
            "result": {"ok": True, "value": {"hits": []}},
        },
    )
    recorder.record_iteration_usage(
        iteration_id="run_4_iteration_1",
        usage={"prompt_tokens": 1, "completion_tokens": 2},
    )
    recorder.record_final_response(
        iteration_id="run_4_iteration_1",
        content="done",
        degraded=False,
    )
    recorder.finish_iteration(iteration_id="run_4_iteration_1", iteration_index=1)
    recorder.close()


@pytest.mark.unit
def test_jsonl_tool_trace_store_exposes_current_trace_file_path(tmp_path: Path) -> None:
    """验证可获取当前本地日期对应的 trace 分片路径。"""

    output_dir = tmp_path / "trace"
    store = JsonlToolTraceStore(output_dir=output_dir)

    file_path = store.get_current_trace_file_path()

    assert file_path.parent == output_dir / "sessions" / "_session_unknown"
    assert file_path.name.startswith("tool_calls_")
    assert file_path.suffix == ".jsonl"


@pytest.mark.unit
def test_tool_trace_helper_functions_cover_message_and_result_edges() -> None:
    """验证 tool_trace 顶部 helper 的剩余边界分支。"""

    class _NonSerializable:
        """触发 JSON 序列化失败。"""

        def __str__(self) -> str:
            """返回稳定字符串表示。"""

            return "non-serializable"

    assert tool_trace_module._normalize_session_partition(" ") == "_session_unknown"
    assert tool_trace_module._normalize_session_partition("session:/a b") == "session__a_b"
    assert tool_trace_module._safe_json_dumps(_NonSerializable()) == "non-serializable"
    assert tool_trace_module._compute_sha256({"a": 1}).startswith("sha256:")

    normalized_error = tool_trace_module._normalize_result_payload("bad-result")
    assert normalized_error["ok"] is False
    assert normalized_error["error"] == "invalid_result"

    success_result = {"ok": True, "value": {"rows": 1}, "meta": {"latency_ms": 12}}
    error_result = {"ok": False, "error": "denied", "message": "permission denied"}
    assert tool_trace_module._build_result_summary(success_result) == "success(type=dict, truncated=False)"
    assert tool_trace_module._build_result_summary(error_result) == "error(code=denied, message=permission denied)"
    assert tool_trace_module._extract_result_data(success_result) == {"rows": 1}
    assert tool_trace_module._extract_result_data({"ok": True, "value": [1, 2]}) is None

    messages = [
        {"role": "system", "name": "summary", "content": "[Context Compaction Summary] short"},
        {"role": "system", "name": "memory", "content": "memory payload"},
        {"role": "tool", "content": {"section": "risk"}},
        {"role": "assistant", "content": "上一轮回复"},
        {"role": "user", "content": "当前问题"},
    ]
    summaries = tool_trace_module._build_messages_summary(messages)
    assert [item["source_tag"] for item in summaries] == [
        "summary",
        "memory",
        "tool_context",
        "recent_history",
        "current_iteration",
    ]
    assert tool_trace_module._extract_message_text(cast(Any, {"content": None})) == ""
    assert tool_trace_module._extract_message_text(cast(Any, {"content": ["a", 1]})) == '["a", 1]'
    assert tool_trace_module._extract_current_user_message(messages) == "当前问题"
    assert tool_trace_module._build_context_meta(messages) == {
        "summary_present": True,
        "summary_version": 1,
        "recent_history_count": 1,
        "memory_keys": ["memory"],
        "tool_context_count": 1,
    }
    assert tool_trace_module._extract_tool_schema_names(
        [
            {"function": {"name": "list_documents"}},
            {"function": {"name": "  "}},
            {"bad": "schema"},
        ]
    ) == ["list_documents"]
    assert tool_trace_module._get_message_excerpt_limit("policy") == 64
    assert tool_trace_module._get_message_excerpt_limit("unknown") == 96
