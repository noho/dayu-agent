"""V2 Tool Trace 分析脚本。

本脚本只面向 `tool_trace_v2`。它的目标不是做通用日志统计，而是围绕
「工具设计是否让一个无状态、会犯错、会走捷径、上下文有限、偏好模式匹配的推理器，在最低认知负担下稳定做对下一步动作」这一北极星，分析 trace 并输出优化建议。

分析重点：
1. 工具是否被稳定选中并成功完成。
2. 工具返回是否过大、过重、难以续读。
3. 截断 contract 是否真的把模型引导到了下一步动作。
4. 工具 schema / 结果摘要是否足以避免重复调用和回摆。
5. trace 本身是否完整、是否足以支撑回放和诊断。

用法：
    python -m utils.analyze_tool_trace [选项]
    python utils/analyze_tool_trace.py [选项]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Optional


TRACE_SCHEMA_VERSION = "tool_trace_v2"
TRACE_TYPE_TOOL_CALL = "tool_call"
TRACE_TYPE_ITERATION_USAGE = "iteration_usage"
TRACE_TYPE_ITERATION_CONTEXT = "iteration_context_snapshot"
TRACE_TYPE_FINAL_RESPONSE = "final_response"
# 注意：该字面量必须与 `dayu.engine.tool_trace.TRACE_TYPE_SSE_PROTOCOL_ERROR` 保持一致。
TRACE_TYPE_SSE_PROTOCOL_ERROR = "sse_protocol_error"

TOP_RECORD_LIMIT = 12
MAX_EXCERPT_CHARS = 80
LARGE_PAYLOAD_PERCENTILE = 0.90
LARGE_INPUT_PERCENTILE = 0.90
HIGH_FAILURE_RATE = 0.20
HIGH_DEGRADE_RATE = 0.20


@dataclass(slots=True)
class TraceIdentity:
    """trace 身份信息。"""

    agent_name: str = ""
    agent_kind: str = ""
    scene_name: str = ""
    model_name: str = ""
    enabled_capabilities: tuple[str, ...] = ()


@dataclass(slots=True)
class RawRef:
    """冷存引用。"""

    blob_id: str
    content_hash: str
    storage_uri: str
    bytes: int


@dataclass(slots=True)
class ToolCallInfo:
    """单次工具调用的 V2 记录。"""

    run_id: str
    session_id: str
    iteration_id: str
    index_in_iteration: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    status: str
    truncated: bool
    latency_ms: int
    error_code: Optional[str]
    result_summary: str
    result_data: Optional[dict[str, Any]]
    raw_result_ref: Optional[RawRef]
    recorded_at: str
    identity: TraceIdentity

    @property
    def is_successful(self) -> bool:
        """返回当前工具调用是否成功。"""

        return self.status == "success"


@dataclass(slots=True)
class IterationUsageInfo:
    """单次 agent iteration 的 token 与预算快照。"""

    run_id: str
    session_id: str
    iteration_id: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cached_tokens: int
    max_context_tokens: int
    current_prompt_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    iteration_count: int
    compaction_count: int
    continuation_count: int
    is_over_soft_limit: bool
    tool_call_budget: Optional[int]
    tool_calls_remaining: Optional[int]
    recorded_at: str
    identity: TraceIdentity


@dataclass(slots=True)
class IterationContextInfo:
    """单次 agent iteration 的送模上下文快照。"""

    run_id: str
    session_id: str
    iteration_id: str
    iteration_index: int
    current_user_message: str
    context_meta: dict[str, Any]
    model_input_messages_summary: list[dict[str, Any]]
    raw_input_ref: Optional[RawRef]
    tool_schema_names: tuple[str, ...]
    raw_tool_schemas_ref: Optional[RawRef]
    tool_calls_summary: list[dict[str, Any] | None]
    recorded_at: str
    identity: TraceIdentity


@dataclass(slots=True)
class FinalResponseInfo:
    """最终响应记录。"""

    run_id: str
    session_id: str
    iteration_id: str
    content: str
    degraded: bool
    filtered: bool
    finish_reason: str | None
    recorded_at: str
    identity: TraceIdentity


@dataclass(slots=True)
class SSEProtocolErrorInfo:
    """SSE 协议错误记录。"""

    run_id: str
    session_id: str
    iteration_id: str
    error_type: str
    partial_tool_name: str | None
    partial_arguments_ref: RawRef | None
    request_id: str
    attempt: int | None
    recorded_at: str
    identity: TraceIdentity


@dataclass(slots=True)
class RunInfo:
    """单个 run 的聚合视图。"""

    run_id: str
    session_id: str
    identity: TraceIdentity
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    iteration_usages: list[IterationUsageInfo] = field(default_factory=list)
    iteration_contexts: list[IterationContextInfo] = field(default_factory=list)
    final_response: FinalResponseInfo | None = None
    sse_protocol_errors: list[SSEProtocolErrorInfo] = field(default_factory=list)

    @property
    def start_time(self) -> str:
        """返回 run 的起始时间。"""

        timestamps = [
            item.recorded_at
            for item in [
                *self.tool_calls,
                *self.iteration_usages,
                *self.iteration_contexts,
                *self.sse_protocol_errors,
            ]
            if item.recorded_at
        ]
        if self.final_response and self.final_response.recorded_at:
            timestamps.append(self.final_response.recorded_at)
        return min(timestamps) if timestamps else ""

    @property
    def end_time(self) -> str:
        """返回 run 的结束时间。"""

        timestamps = [
            item.recorded_at
            for item in [
                *self.tool_calls,
                *self.iteration_usages,
                *self.iteration_contexts,
                *self.sse_protocol_errors,
            ]
            if item.recorded_at
        ]
        if self.final_response and self.final_response.recorded_at:
            timestamps.append(self.final_response.recorded_at)
        return max(timestamps) if timestamps else ""

    @property
    def has_final_response(self) -> bool:
        """是否存在 final_response。"""

        return self.final_response is not None

    @property
    def degraded(self) -> bool:
        """是否降级输出。"""

        return bool(self.final_response and self.final_response.degraded)

    @property
    def filtered(self) -> bool:
        """最终响应是否被内容过滤命中。"""

        return bool(self.final_response and self.final_response.filtered)

    @property
    def finish_reason(self) -> str | None:
        """最终响应的 finish_reason；缺失返回 ``None``。"""

        return self.final_response.finish_reason if self.final_response else None

    @property
    def iteration_count(self) -> int:
        """返回最终 agent iteration 次数。"""

        usage_iteration_count = max((item.iteration_count for item in self.iteration_usages), default=0)
        snapshot_iteration_count = max((item.iteration_index for item in self.iteration_contexts), default=0)
        return max(usage_iteration_count, snapshot_iteration_count)

    @property
    def total_tool_calls(self) -> int:
        """返回工具调用总数。"""

        return len(self.tool_calls)

    @property
    def successful_tool_calls(self) -> int:
        """返回成功工具调用数。"""

        return sum(1 for item in self.tool_calls if item.is_successful)

    @property
    def total_prompt_tokens(self) -> int:
        """返回累计 prompt tokens。"""

        return max((item.total_prompt_tokens for item in self.iteration_usages), default=0)

    @property
    def total_completion_tokens(self) -> int:
        """返回累计 completion tokens。"""

        return max((item.total_completion_tokens for item in self.iteration_usages), default=0)

    @property
    def max_raw_input_bytes(self) -> int:
        """返回 run 内最大的输入快照字节数。"""

        values = [
            item.raw_input_ref.bytes
            for item in self.iteration_contexts
            if item.raw_input_ref is not None and item.raw_input_ref.bytes >= 0
        ]
        return max(values) if values else 0

    @property
    def max_raw_result_bytes(self) -> int:
        """返回 run 内最大的工具结果字节数。"""

        values = [
            item.raw_result_ref.bytes
            for item in self.tool_calls
            if item.raw_result_ref is not None and item.raw_result_ref.bytes >= 0
        ]
        return max(values) if values else 0

    @property
    def first_user_message(self) -> str:
        """返回首个 agent iteration 的用户输入摘要。"""

        if not self.iteration_contexts:
            return ""
        ordered = sorted(self.iteration_contexts, key=lambda item: item.iteration_index)
        return ordered[0].current_user_message

    @property
    def sse_protocol_error_count(self) -> int:
        """返回 SSE 协议错误条数。"""

        return len(self.sse_protocol_errors)

    @property
    def latest_sse_protocol_error(self) -> Optional[SSEProtocolErrorInfo]:
        """返回最近一条 SSE 协议错误。"""

        if not self.sse_protocol_errors:
            return None
        return max(self.sse_protocol_errors, key=lambda item: item.recorded_at)


@dataclass(slots=True)
class ToolStats:
    """工具级汇总统计。"""

    tool_name: str
    call_count: int
    success_count: int
    success_rate: float
    truncation_count: int
    truncation_rate: float
    median_latency_ms: int
    median_result_bytes: int
    p90_result_bytes: int
    median_argument_keys: int
    top_error_codes: list[tuple[str, int]]


@dataclass(slots=True)
class AnalysisBundle:
    """聚合后的分析结果。"""

    runs: list[RunInfo]
    sse_protocol_errors: list[SSEProtocolErrorInfo]
    tool_stats: list[ToolStats]
    duplicate_calls: list[dict[str, Any]]
    truncation_issues: list[dict[str, Any]]
    large_payload_calls: list[dict[str, Any]]
    large_prompt_iterations: list[dict[str, Any]]
    failure_patterns: list[dict[str, Any]]
    detailed_failure_patterns: list[dict[str, Any]]
    context_pressure_runs: list[dict[str, Any]]
    trace_integrity_issues: list[dict[str, Any]]
    prompt_pairs: list[dict[str, Any]]
    tool_schemas: list[dict[str, Any]]
    search_quality: dict[str, Any]
    list_documents_quality: dict[str, Any]
    fetch_more_quality: dict[str, Any]
    web_fetch_gaps: list[dict[str, Any]]
    web_search_empty_streaks: list[dict[str, Any]]
    recommendations: list[dict[str, str]]


def _load_jsonl_files(path: Path) -> list[dict[str, Any]]:
    """加载 trace 文件。

    Args:
        path: 单个 JSONL 文件路径，或包含多个 JSONL/JSONL.GZ 文件的目录。

    Returns:
        解析成功的原始记录列表。

    Raises:
        FileNotFoundError: 路径不存在时抛出。
    """

    if not path.exists():
        raise FileNotFoundError(f"trace 路径不存在: {path}")
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            [
                file
                for file in path.rglob("*")
                if file.is_file() and (file.suffix == ".jsonl" or file.name.endswith(".jsonl.gz"))
            ]
        )

    records: list[dict[str, Any]] = []
    for file in files:
        lines = _read_text_lines(file)
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _read_text_lines(path: Path) -> list[str]:
    """读取文本行，兼容 `.jsonl` 与 `.jsonl.gz`。

    Args:
        path: 文件路径。

    Returns:
        文本行列表。

    Raises:
        OSError: 文件读取失败时向上传播。
    """

    if path.name.endswith(".gz"):
        return gzip.decompress(path.read_bytes()).decode("utf-8").splitlines()
    return path.read_text(encoding="utf-8").splitlines()


def _parse_trace_identity(record: dict[str, Any]) -> TraceIdentity:
    """解析 trace 身份信息。"""

    enabled_capabilities = record.get("enabled_capabilities")
    if not isinstance(enabled_capabilities, list):
        enabled_capabilities = []
    return TraceIdentity(
        agent_name=str(record.get("agent_name") or ""),
        agent_kind=str(record.get("agent_kind") or ""),
        scene_name=str(record.get("scene_name") or ""),
        model_name=str(record.get("model_name") or ""),
        enabled_capabilities=tuple(str(item) for item in enabled_capabilities if str(item or "")),
    )


def _parse_raw_ref(payload: Any) -> Optional[RawRef]:
    """解析冷存引用。"""

    if not isinstance(payload, dict):
        return None
    return RawRef(
        blob_id=str(payload.get("blob_id") or ""),
        content_hash=str(payload.get("content_hash") or ""),
        storage_uri=str(payload.get("storage_uri") or ""),
        bytes=_to_int(payload.get("bytes"), default=-1),
    )


def _to_int(value: Any, *, default: int = 0) -> int:
    """安全转换为整数。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, *, default: float = 0.0) -> float:
    """安全转换为浮点数。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_tool_call(record: dict[str, Any]) -> Optional[ToolCallInfo]:
    """解析 V2 `tool_call` 记录。"""

    if record.get("trace_type") != TRACE_TYPE_TOOL_CALL:
        return None
    if record.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        return None

    result_fact = record.get("result_fact")
    if not isinstance(result_fact, dict):
        result_fact = {}
    arguments = record.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    result_data = record.get("result_data")
    if not isinstance(result_data, dict):
        result_data = None

    return ToolCallInfo(
        run_id=str(record.get("run_id") or ""),
        session_id=str(record.get("session_id") or ""),
        iteration_id=str(record.get("iteration_id") or ""),
        index_in_iteration=_to_int(record.get("index_in_iteration"), default=0),
        tool_call_id=str(record.get("tool_call_id") or ""),
        tool_name=str(record.get("tool_name") or ""),
        arguments=arguments,
        status=str(result_fact.get("status") or ""),
        truncated=bool(result_fact.get("truncated")),
        latency_ms=_to_int(result_fact.get("latency_ms"), default=-1),
        error_code=_to_optional_text(result_fact.get("error_code")),
        result_summary=str(record.get("result_summary") or ""),
        result_data=result_data,
        raw_result_ref=_parse_raw_ref(result_fact.get("raw_result_ref")),
        recorded_at=str(record.get("recorded_at") or ""),
        identity=_parse_trace_identity(record),
    )


def _parse_iteration_usage(record: dict[str, Any]) -> Optional[IterationUsageInfo]:
    """解析 V2 `iteration_usage` 记录。"""

    if record.get("trace_type") != TRACE_TYPE_ITERATION_USAGE:
        return None
    if record.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        return None

    usage = record.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    budget = record.get("budget_snapshot")
    if not isinstance(budget, dict):
        budget = {}
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}
    prompt_details = usage.get("prompt_tokens_details")
    if not isinstance(prompt_details, dict):
        prompt_details = {}

    return IterationUsageInfo(
        run_id=str(record.get("run_id") or ""),
        session_id=str(record.get("session_id") or ""),
        iteration_id=str(record.get("iteration_id") or ""),
        prompt_tokens=_to_int(usage.get("prompt_tokens")),
        completion_tokens=_to_int(usage.get("completion_tokens")),
        reasoning_tokens=_to_int(completion_details.get("reasoning_tokens")),
        cached_tokens=_to_int(prompt_details.get("cached_tokens")),
        max_context_tokens=_to_int(budget.get("max_context_tokens")),
        current_prompt_tokens=_to_int(budget.get("current_prompt_tokens")),
        total_prompt_tokens=_to_int(budget.get("total_prompt_tokens")),
        total_completion_tokens=_to_int(budget.get("total_completion_tokens")),
        iteration_count=_to_int(budget.get("iteration_count")),
        compaction_count=_to_int(budget.get("compaction_count")),
        continuation_count=_to_int(budget.get("continuation_count")),
        is_over_soft_limit=bool(budget.get("is_over_soft_limit")),
        tool_call_budget=_to_optional_int(budget.get("tool_call_budget")),
        tool_calls_remaining=_to_optional_int(budget.get("tool_calls_remaining")),
        recorded_at=str(record.get("recorded_at") or ""),
        identity=_parse_trace_identity(record),
    )


def _parse_iteration_context(record: dict[str, Any]) -> Optional[IterationContextInfo]:
    """解析 V2 `iteration_context_snapshot` 记录。"""

    if record.get("trace_type") != TRACE_TYPE_ITERATION_CONTEXT:
        return None
    if record.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        return None

    context_meta = record.get("context_meta")
    if not isinstance(context_meta, dict):
        context_meta = {}
    summaries = record.get("model_input_messages_summary")
    if not isinstance(summaries, list):
        summaries = []
    tool_calls_summary = record.get("tool_calls")
    if not isinstance(tool_calls_summary, list):
        tool_calls_summary = []
    tool_schema_names = record.get("tool_schema_names")
    if not isinstance(tool_schema_names, list):
        tool_schema_names = []

    return IterationContextInfo(
        run_id=str(record.get("run_id") or ""),
        session_id=str(record.get("session_id") or ""),
        iteration_id=str(record.get("iteration_id") or ""),
        iteration_index=_to_int(record.get("iteration_index"), default=0),
        current_user_message=str(record.get("current_user_message") or ""),
        context_meta=context_meta,
        model_input_messages_summary=[item for item in summaries if isinstance(item, dict)],
        raw_input_ref=_parse_raw_ref(record.get("raw_input_ref")),
        tool_schema_names=tuple(str(item) for item in tool_schema_names if str(item or "")),
        raw_tool_schemas_ref=_parse_raw_ref(record.get("raw_tool_schemas_ref")),
        tool_calls_summary=tool_calls_summary,
        recorded_at=str(record.get("recorded_at") or ""),
        identity=_parse_trace_identity(record),
    )


def _parse_final_response(record: dict[str, Any]) -> Optional[FinalResponseInfo]:
    """解析 V2 `final_response` 记录。"""

    if record.get("trace_type") != TRACE_TYPE_FINAL_RESPONSE:
        return None
    if record.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        return None

    final_response = record.get("final_response")
    if not isinstance(final_response, dict):
        final_response = {}
    raw_finish_reason = final_response.get("finish_reason")
    finish_reason = str(raw_finish_reason).strip() if raw_finish_reason else None
    return FinalResponseInfo(
        run_id=str(record.get("run_id") or ""),
        session_id=str(record.get("session_id") or ""),
        iteration_id=str(record.get("iteration_id") or ""),
        content=str(final_response.get("content") or ""),
        degraded=bool(final_response.get("degraded")),
        filtered=bool(final_response.get("filtered")),
        finish_reason=finish_reason or None,
        recorded_at=str(record.get("recorded_at") or ""),
        identity=_parse_trace_identity(record),
    )


def _parse_sse_protocol_error(record: dict[str, Any]) -> Optional[SSEProtocolErrorInfo]:
    """解析 V2 `sse_protocol_error` 记录。"""

    if record.get("trace_type") != TRACE_TYPE_SSE_PROTOCOL_ERROR:
        return None
    if record.get("trace_schema_version") != TRACE_SCHEMA_VERSION:
        return None

    return SSEProtocolErrorInfo(
        run_id=str(record.get("run_id") or ""),
        session_id=str(record.get("session_id") or ""),
        iteration_id=str(record.get("iteration_id") or ""),
        error_type=str(record.get("error_type") or ""),
        partial_tool_name=_to_optional_text(record.get("partial_tool_name")),
        partial_arguments_ref=_parse_raw_ref(record.get("partial_arguments_ref")),
        request_id=str(record.get("request_id") or ""),
        attempt=_to_optional_int(record.get("attempt")),
        recorded_at=str(record.get("recorded_at") or ""),
        identity=_parse_trace_identity(record),
    )


def _to_optional_text(value: Any) -> Optional[str]:
    """转换为可选字符串。"""

    text = str(value).strip() if value is not None else ""
    return text or None


def _to_optional_int(value: Any) -> Optional[int]:
    """转换为可选整数。"""

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _group_runs(records: list[dict[str, Any]]) -> list[RunInfo]:
    """将原始记录聚合为 run 视图。"""

    runs: dict[str, RunInfo] = {}
    for record in records:
        run_id = str(record.get("run_id") or "").strip()
        session_id = str(record.get("session_id") or "").strip()
        if not run_id:
            continue
        identity = _parse_trace_identity(record)
        run = runs.setdefault(
            run_id,
            RunInfo(
                run_id=run_id,
                session_id=session_id,
                identity=identity,
            ),
        )
        if not run.session_id and session_id:
            run.session_id = session_id
        if not run.identity.agent_name and identity.agent_name:
            run.identity = identity

        tool_call = _parse_tool_call(record)
        if tool_call is not None:
            run.tool_calls.append(tool_call)
            continue

        iteration_usage = _parse_iteration_usage(record)
        if iteration_usage is not None:
            run.iteration_usages.append(iteration_usage)
            continue

        iteration_context = _parse_iteration_context(record)
        if iteration_context is not None:
            run.iteration_contexts.append(iteration_context)
            continue

        final_response = _parse_final_response(record)
        if final_response is not None:
            run.final_response = final_response
            continue

        sse_protocol_error = _parse_sse_protocol_error(record)
        if sse_protocol_error is not None:
            run.sse_protocol_errors.append(sse_protocol_error)
            continue

    return sorted(runs.values(), key=lambda item: item.start_time)


def _summarize_tool_stats(runs: list[RunInfo]) -> list[ToolStats]:
    """聚合工具级统计。"""

    grouped: dict[str, list[ToolCallInfo]] = defaultdict(list)
    for run in runs:
        for call in run.tool_calls:
            grouped[call.tool_name].append(call)

    stats: list[ToolStats] = []
    for tool_name, calls in sorted(grouped.items()):
        latencies = [item.latency_ms for item in calls if item.latency_ms >= 0]
        result_bytes = [
            item.raw_result_ref.bytes
            for item in calls
            if item.raw_result_ref is not None and item.raw_result_ref.bytes >= 0
        ]
        error_counter = Counter(item.error_code for item in calls if item.error_code)
        success_count = sum(1 for item in calls if item.is_successful)
        truncation_count = sum(1 for item in calls if item.truncated)
        stats.append(
            ToolStats(
                tool_name=tool_name,
                call_count=len(calls),
                success_count=success_count,
                success_rate=_safe_ratio(success_count, len(calls)),
                truncation_count=truncation_count,
                truncation_rate=_safe_ratio(truncation_count, len(calls)),
                median_latency_ms=int(median(latencies)) if latencies else -1,
                median_result_bytes=int(median(result_bytes)) if result_bytes else 0,
                p90_result_bytes=_percentile_int(result_bytes, LARGE_PAYLOAD_PERCENTILE),
                median_argument_keys=int(median([len(item.arguments) for item in calls])) if calls else 0,
                top_error_codes=error_counter.most_common(3),
            )
        )
    return sorted(stats, key=lambda item: (-item.call_count, item.tool_name))


def _safe_ratio(numerator: int, denominator: int) -> float:
    """安全计算比例。"""

    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _percentile_int(values: list[int], percentile: float) -> int:
    """计算整数百分位。"""

    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    percentile = min(max(percentile, 0.0), 1.0)
    index = int(math.ceil((len(ordered) - 1) * percentile))
    return ordered[index]


def _build_duplicate_calls(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """检测同一 run 内的重复调用。"""

    findings: list[dict[str, Any]] = []
    for run in runs:
        grouped: dict[tuple[str, str], list[ToolCallInfo]] = defaultdict(list)
        for call in run.tool_calls:
            grouped[(call.tool_name, _stable_json(call.arguments))].append(call)
        for (tool_name, stable_args), calls in grouped.items():
            if len(calls) <= 1:
                continue
            ordered = sorted(calls, key=_tool_call_sort_key)
            findings.append(
                {
                    "run_id": run.run_id,
                    "session_id": run.session_id,
                    "tool_name": tool_name,
                    "call_count": len(ordered),
                    "iteration_ids": [item.iteration_id for item in ordered],
                    "args_summary": _args_summary(tool_name, ordered[0].arguments),
                    "reason": "同一 run 内相同工具 + 相同参数被重复调用",
                }
            )
    findings.sort(key=lambda item: (-item["call_count"], item["tool_name"], item["run_id"]))
    return findings


def _stable_json(payload: Any) -> str:
    """稳定序列化为 JSON。"""

    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return str(payload)


def _args_summary(tool_name: str, arguments: dict[str, Any], *, max_chars: int = 120) -> str:
    """生成可读参数摘要。"""

    preferred_fields = {
        "list_documents": ("ticker", "document_types", "fiscal_years", "fiscal_periods"),
        "search_document": ("ticker", "document_id", "query", "queries", "mode", "within_section_ref"),
        "read_section": ("ticker", "document_id", "ref"),
        "get_document_sections": ("ticker", "document_id"),
        "get_financial_statement": ("ticker", "document_id", "statement_type"),
        "fetch_more": ("scope_token", "cursor"),
        "search_web": ("query",),
    }.get(tool_name, tuple(arguments.keys()))

    parts: list[str] = []
    for key in preferred_fields:
        if key not in arguments:
            continue
        value = arguments.get(key)
        text = _compact_value(value)
        parts.append(f"{key}={text}")
    summary = ", ".join(parts) or _compact_value(arguments)
    return summary[:max_chars]


def _compact_value(value: Any) -> str:
    """压缩单个值的展示文本。"""

    if isinstance(value, list):
        items = ", ".join(_compact_value(item) for item in value[:3])
        if len(value) > 3:
            items += f", …+{len(value) - 3}"
        return f"[{items}]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={_compact_value(v)}" for k, v in list(value.items())[:3]) + "}"
    text = str(value)
    if len(text) > 36:
        text = text[:33] + "..."
    return repr(text)


def _build_truncation_issues(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """检测截断后未明显续读的问题。"""

    findings: list[dict[str, Any]] = []
    for run in runs:
        ordered = sorted(run.tool_calls, key=_tool_call_sort_key)
        for index, call in enumerate(ordered):
            if not call.truncated:
                continue
            next_call = ordered[index + 1] if index + 1 < len(ordered) else None
            used_fetch_more = next_call is not None and next_call.tool_name == "fetch_more"
            raw_payload = _load_raw_payload(call.raw_result_ref)
            raw_truncation = raw_payload.get("truncation")
            truncation: dict[str, Any] = raw_truncation if isinstance(raw_truncation, dict) else {}
            has_fetch_more_args = isinstance(truncation.get("fetch_more_args"), dict) and bool(truncation.get("fetch_more_args"))
            next_action = _to_optional_text(truncation.get("next_action"))
            hint_text = _to_optional_text(raw_payload.get("hint"))
            findings.append(
                {
                    "run_id": run.run_id,
                    "iteration_id": call.iteration_id,
                    "tool_name": call.tool_name,
                    "args_summary": _args_summary(call.tool_name, call.arguments),
                    "next_tool_name": next_call.tool_name if next_call else None,
                    "used_fetch_more": used_fetch_more,
                    "has_fetch_more_args": has_fetch_more_args,
                    "hint_text": hint_text,
                    "has_hint": bool(hint_text),
                    "next_action": next_action,
                    "raw_result_bytes": call.raw_result_ref.bytes if call.raw_result_ref else -1,
                }
            )
    findings.sort(
        key=lambda item: (
            item["used_fetch_more"],
            item["has_hint"],
            item["next_action"] == "fetch_more",
            item["run_id"],
        )
    )
    return findings


def _load_raw_payload(raw_ref: Optional[RawRef]) -> dict[str, Any]:
    """按需读取冷存原始载荷。"""

    payload = _load_raw_json(raw_ref)
    return payload if isinstance(payload, dict) else {}


def _load_raw_json(raw_ref: Optional[RawRef]) -> Any:
    """按需读取冷存 JSON 载荷，返回原始 JSON 结构。"""

    if raw_ref is None or not raw_ref.storage_uri:
        return None
    path = Path(raw_ref.storage_uri)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload


def _extract_sse_partial_arguments_excerpt(sse_error: SSEProtocolErrorInfo) -> str:
    """从 SSE 协议错误冷存中提取部分 arguments 前缀摘要。"""

    payload = _load_raw_json(sse_error.partial_arguments_ref)
    if not isinstance(payload, dict):
        return "-"
    tool_calls = payload.get("tool_calls")
    if not isinstance(tool_calls, list):
        return "-"

    target_tool_name = sse_error.partial_tool_name
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_payload = tool_call.get("function")
        if not isinstance(function_payload, dict):
            continue
        tool_name = _to_optional_text(function_payload.get("name"))
        if target_tool_name and tool_name != target_tool_name:
            continue
        arguments = function_payload.get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            return _excerpt(arguments, max_chars=100)

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_payload = tool_call.get("function")
        if not isinstance(function_payload, dict):
            continue
        arguments = function_payload.get("arguments")
        if isinstance(arguments, str) and arguments.strip():
            return _excerpt(arguments, max_chars=100)
    return "-"


def _build_large_payload_calls(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """检测结果负载过大的工具调用。"""

    raw_bytes = [
        call.raw_result_ref.bytes
        for run in runs
        for call in run.tool_calls
        if call.raw_result_ref is not None and call.raw_result_ref.bytes > 0
    ]
    threshold = _adaptive_large_threshold(raw_bytes)
    findings: list[dict[str, Any]] = []
    for run in runs:
        for call in run.tool_calls:
            if call.raw_result_ref is None or call.raw_result_ref.bytes <= 0:
                continue
            if call.raw_result_ref.bytes < threshold:
                continue
            findings.append(
                {
                    "run_id": run.run_id,
                    "iteration_id": call.iteration_id,
                    "tool_name": call.tool_name,
                    "raw_result_bytes": call.raw_result_ref.bytes,
                    "threshold": threshold,
                    "args_summary": _args_summary(call.tool_name, call.arguments),
                    "result_summary": call.result_summary,
                }
            )
    findings.sort(key=lambda item: (-item["raw_result_bytes"], item["tool_name"]))
    return findings


def _build_large_prompt_iterations(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """检测输入上下文过大的 agent iteration。"""

    input_bytes = [
        item.raw_input_ref.bytes
        for run in runs
        for item in run.iteration_contexts
        if item.raw_input_ref is not None and item.raw_input_ref.bytes > 0
    ]
    threshold = _adaptive_large_threshold(input_bytes, percentile=LARGE_INPUT_PERCENTILE)
    usage_index = _index_iteration_usage_by_iteration_id(runs)
    findings: list[dict[str, Any]] = []
    for run in runs:
        for context in run.iteration_contexts:
            if context.raw_input_ref is None or context.raw_input_ref.bytes <= 0:
                continue
            if context.raw_input_ref.bytes < threshold:
                continue
            usage = usage_index.get((context.run_id, context.iteration_id))
            findings.append(
                {
                    "run_id": context.run_id,
                    "iteration_id": context.iteration_id,
                    "iteration_index": context.iteration_index,
                    "raw_input_bytes": context.raw_input_ref.bytes,
                    "threshold": threshold,
                    "current_user_message": _excerpt(context.current_user_message),
                    "summary_present": bool(context.context_meta.get("summary_present")),
                    "recent_history_count": _to_int(context.context_meta.get("recent_history_count")),
                    "is_over_soft_limit": bool(usage.is_over_soft_limit) if usage else False,
                    "compaction_count": usage.compaction_count if usage else 0,
                    "message_mix": _summarize_message_mix(context.model_input_messages_summary),
                }
            )
    findings.sort(key=lambda item: (-item["raw_input_bytes"], item["run_id"], item["iteration_index"]))
    return findings


def _adaptive_large_threshold(values: list[int], *, percentile: float = LARGE_PAYLOAD_PERCENTILE) -> int:
    """使用自适应规则计算“大载荷”阈值。"""

    positive = sorted(value for value in values if value > 0)
    if not positive:
        return 0
    median_value = int(median(positive))
    percentile_value = _percentile_int(positive, percentile)
    return max(percentile_value, median_value * 2)


def _index_iteration_usage_by_iteration_id(runs: list[RunInfo]) -> dict[tuple[str, str], IterationUsageInfo]:
    """构建 `(run_id, iteration_id)` 到 `IterationUsageInfo` 的索引。"""

    index: dict[tuple[str, str], IterationUsageInfo] = {}
    for run in runs:
        for usage in run.iteration_usages:
            index[(usage.run_id, usage.iteration_id)] = usage
    return index


def _build_failure_patterns(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """聚合失败模式。"""

    counter: Counter[tuple[str, str]] = Counter()
    for run in runs:
        for call in run.tool_calls:
            if call.is_successful:
                continue
            counter[(call.tool_name, call.error_code or "UNKNOWN")] += 1
    findings = [
        {"tool_name": tool_name, "error_code": error_code, "count": count}
        for (tool_name, error_code), count in counter.most_common()
    ]
    return findings


def _build_detailed_failure_patterns(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """聚合可直接指导工具优化的详细失败签名。"""

    counter: Counter[tuple[str, str, str, str, str]] = Counter()
    samples: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for run in runs:
        for call in run.tool_calls:
            if call.is_successful:
                continue
            detail = _extract_detailed_error(call)
            signature = (
                call.tool_name,
                detail["error_signature"],
                detail["http_status_text"],
                detail["message"],
                detail["url"],
            )
            counter[signature] += 1
            samples.setdefault(
                signature,
                {
                    "tool_name": call.tool_name,
                    "error_signature": detail["error_signature"],
                    "http_status": detail["http_status_text"],
                    "message": detail["message"],
                    "url": detail["url"],
                    "detail_excerpt": detail["detail_excerpt"],
                    "count": 0,
                },
            )

    findings: list[dict[str, Any]] = []
    for signature, count in counter.most_common():
        sample = dict(samples[signature])
        sample["count"] = count
        findings.append(sample)
    return findings


def _summarize_message_mix(summaries: list[dict[str, Any]]) -> str:
    """按 source_tag 汇总消息结构，不依赖 excerpt 长度。"""

    counter: Counter[str] = Counter()
    for item in summaries:
        source_tag = str(item.get("source_tag") or "unknown")
        counter[source_tag] += 1
    if not counter:
        return "-"
    ordered = sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))
    return ", ".join(f"{key}:{value}" for key, value in ordered)


def _extract_detailed_error(call: ToolCallInfo) -> dict[str, str]:
    """从 raw result 中提取更细粒度的错误签名。

    设计意图：
    - `result_fact.error_code` 通常过粗，例如统一落成 `EXECUTION_ERROR`。
    - 真正能指导工具优化的是更细的失败原因，例如 URL 拦截、HTTP 403、HTTP 429、超时等。
    """

    raw_payload = _load_raw_payload(call.raw_result_ref)
    raw_error = raw_payload.get("error")
    error = raw_error if isinstance(raw_error, dict) else {}
    business = _extract_business_error_payload(call=call, raw_payload=raw_payload)
    raw_meta = raw_payload.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    raw_code = (
        _to_optional_text(error.get("code"))
        or (_to_optional_text(raw_error) if not isinstance(raw_error, dict) else None)
        or _to_optional_text(business.get("error_code"))
        or call.error_code
        or "UNKNOWN"
    )
    raw_message = (
        _to_optional_text(error.get("message"))
        or _to_optional_text(raw_payload.get("message"))
        or _to_optional_text(business.get("message"))
        or "UNKNOWN"
    )
    raw_detail = (
        _to_optional_text(error.get("detail"))
        or _to_optional_text(raw_payload.get("detail"))
        or _to_optional_text(raw_payload.get("hint"))
        or _to_optional_text(business.get("detail"))
        or _to_optional_text(business.get("hint"))
        or _summarize_error_meta(meta)
        or ""
    )
    error_url = (
        _to_optional_text(business.get("url"))
        or _to_optional_text(raw_payload.get("url"))
        or _to_optional_text(call.arguments.get("url"))
        or "-"
    )
    http_status = _extract_http_status(raw_payload=raw_payload, error=error, business=business)
    error_signature = _classify_error_signature(
        raw_code=raw_code,
        raw_message=raw_message,
        raw_detail=raw_detail,
        http_status=http_status,
    )
    return {
        "error_signature": error_signature,
        "http_status_text": str(http_status) if http_status is not None else "-",
        "message": raw_message,
        "url": error_url,
        "detail_excerpt": _excerpt(raw_detail, max_chars=120) or "-",
    }


def _summarize_error_meta(meta: dict[str, Any]) -> str:
    """从错误 meta 中提炼对调试最有帮助的摘要。"""

    if not meta:
        return ""

    raw_repair_hint = meta.get("repair_hint")
    repair_hint: dict[str, Any] = raw_repair_hint if isinstance(raw_repair_hint, dict) else {}
    raw_issues = meta.get("issues")
    issues: list[Any] = raw_issues if isinstance(raw_issues, list) else []
    issue_parts: list[str] = []
    for issue in issues[:2]:
        if not isinstance(issue, dict):
            continue
        fields = issue.get("fields") if isinstance(issue.get("fields"), list) else []
        allowed_fields = issue.get("allowed_fields") if isinstance(issue.get("allowed_fields"), list) else []
        reason = _to_optional_text(issue.get("reason")) or "unknown"
        path = _to_optional_text(issue.get("path")) or "$"
        detail = f"path={path}, reason={reason}"
        if fields:
            detail += f", fields={','.join(str(field) for field in fields)}"
        if allowed_fields:
            detail += f", allowed={','.join(str(field) for field in allowed_fields[:8])}"
        issue_parts.append(detail)

    repair_message = _to_optional_text(repair_hint.get("message")) or ""
    repair_action = _to_optional_text(repair_hint.get("action")) or ""
    if repair_action and repair_message:
        issue_parts.append(f"repair={repair_action}: {repair_message}")
    elif repair_action:
        issue_parts.append(f"repair={repair_action}")
    elif repair_message:
        issue_parts.append(repair_message)

    return " | ".join(issue_parts)


def _extract_business_error_payload(*, call: ToolCallInfo, raw_payload: dict[str, Any]) -> dict[str, Any]:
    """提取业务层失败载荷。

    设计意图：
    - V2 trace 下，很多高价值失败信息并不在外层 ``error``，而在业务层
      ``data.value`` / ``result_data``，例如 ``permission_denied`` 与被拒 URL。
    - analyzer 应优先消费这些结构化字段，而不是退化成只看外层异常包装。
    """

    if isinstance(call.result_data, dict):
        return call.result_data
    data = raw_payload.get("data")
    if not isinstance(data, dict):
        return {}
    value = data.get("value")
    return value if isinstance(value, dict) else {}


def _extract_http_status(
    *, raw_payload: dict[str, Any], error: dict[str, Any], business: dict[str, Any]
) -> Optional[int]:
    """从原始错误载荷中提取 HTTP 状态码。"""

    candidates = [
        error.get("http_status"),
        error.get("status"),
        error.get("status_code"),
        business.get("http_status"),
        business.get("status"),
        business.get("status_code"),
        raw_payload.get("http_status"),
        raw_payload.get("status"),
        raw_payload.get("status_code"),
    ]
    for value in candidates:
        status = _to_optional_int(value)
        if status is not None and 100 <= status <= 599:
            return status

    text_candidates = [
        _to_optional_text(error.get("detail")) or "",
        _to_optional_text(error.get("message")) or "",
    ]
    patterns = [
        re.compile(r"HTTP(?:\s+status)?\s*(\d{3})", re.IGNORECASE),
        re.compile(r"status(?:\s+code)?\s*[:=]?\s*(\d{3})", re.IGNORECASE),
        re.compile(r"\b(\d{3})\b"),
    ]
    for text in text_candidates:
        if not text:
            continue
        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            status = _to_optional_int(match.group(1))
            if status is not None and 100 <= status <= 599:
                return status
    return None


def _classify_error_signature(
    *,
    raw_code: str,
    raw_message: str,
    raw_detail: str,
    http_status: Optional[int],
) -> str:
    """将粗粒度错误映射为更可操作的失败签名。"""

    detail = raw_detail.lower()
    message = raw_message.lower()

    if "不允许访问的 url" in detail or "not allowed url" in detail:
        return "URL_NOT_ALLOWED"
    if raw_code == "permission_denied" and "blocked by fetch safety policy" in message:
        return "URL_BLOCKED_BY_POLICY"
    if "timeout" in detail or "timed out" in detail or "超时" in raw_detail:
        return "TIMEOUT"
    if "dns" in detail or "name or service not known" in detail:
        return "DNS_ERROR"
    if "ssl" in detail or "certificate" in detail:
        return "SSL_ERROR"
    if http_status is not None:
        if http_status == 403:
            return "HTTP_403"
        if http_status == 404:
            return "HTTP_404"
        if http_status == 429:
            return "HTTP_429"
        if 500 <= http_status <= 599:
            return "HTTP_5XX"
        return f"HTTP_{http_status}"
    if raw_code == "EXECUTION_ERROR" and ("valueerror" in message or "valueerror" in detail):
        return "VALUE_ERROR"
    return raw_code or "UNKNOWN"


def _build_context_pressure_runs(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """识别上下文压力较高的 run。"""

    findings: list[dict[str, Any]] = []
    for run in runs:
        over_soft_limit = any(item.is_over_soft_limit for item in run.iteration_usages)
        compaction_count = max((item.compaction_count for item in run.iteration_usages), default=0)
        continuation_count = max((item.continuation_count for item in run.iteration_usages), default=0)
        has_sse_protocol_error = run.sse_protocol_error_count > 0
        if not (
            over_soft_limit
            or compaction_count > 0
            or continuation_count > 0
            or run.degraded
            or run.filtered
            or (not run.has_final_response and not has_sse_protocol_error)
        ):
            continue
        findings.append(
            {
                "run_id": run.run_id,
                "scene_name": run.identity.scene_name,
                "model_name": run.identity.model_name,
                "iteration_count": run.iteration_count,
                "tool_calls": run.total_tool_calls,
                "compaction_count": compaction_count,
                "continuation_count": continuation_count,
                "over_soft_limit": over_soft_limit,
                "degraded": run.degraded,
                "filtered": run.filtered,
                "finish_reason": run.finish_reason,
                "has_final_response": run.has_final_response,
                "has_sse_protocol_error": has_sse_protocol_error,
                "max_raw_input_bytes": run.max_raw_input_bytes,
            }
        )
    findings.sort(
        key=lambda item: (
            not item["degraded"],
            not item["filtered"],
            item["has_final_response"],
            -item["compaction_count"],
            -item["max_raw_input_bytes"],
        )
    )
    return findings


def _build_trace_integrity_issues(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """检查 trace 自身是否完整。"""

    findings: list[dict[str, Any]] = []
    for run in runs:
        context_index = {(item.run_id, item.iteration_id): item for item in run.iteration_contexts}
        grouped_calls: dict[tuple[str, str], list[ToolCallInfo]] = defaultdict(list)
        for call in run.tool_calls:
            grouped_calls[(call.run_id, call.iteration_id)].append(call)

        for key, context in context_index.items():
            snapshot_call_count = len(context.tool_calls_summary)
            snapshot_null_count = sum(1 for item in context.tool_calls_summary if item is None)
            actual_call_count = len(grouped_calls.get(key, []))
            if snapshot_null_count > 0:
                findings.append(
                    {
                        "issue": "iteration_context_snapshot.tool_calls 含空值",
                        "run_id": context.run_id,
                        "iteration_id": context.iteration_id,
                        "detail": f"snapshot_call_count={snapshot_call_count}, null_count={snapshot_null_count}",
                    }
                )
            if actual_call_count != snapshot_call_count:
                findings.append(
                    {
                        "issue": "iteration_context_snapshot.tool_calls 与 tool_call 记录数不一致",
                        "run_id": context.run_id,
                        "iteration_id": context.iteration_id,
                        "detail": f"snapshot={snapshot_call_count}, actual={actual_call_count}",
                    }
                )
            if context.raw_input_ref is None:
                findings.append(
                    {
                        "issue": "iteration_context_snapshot 缺少 raw_input_ref",
                        "run_id": context.run_id,
                        "iteration_id": context.iteration_id,
                        "detail": "无法回放真实送模输入",
                    }
                )

        for call in run.tool_calls:
            if call.raw_result_ref is None:
                findings.append(
                    {
                        "issue": "tool_call 缺少 raw_result_ref",
                        "run_id": call.run_id,
                        "iteration_id": call.iteration_id,
                        "detail": f"tool={call.tool_name}, tool_call_id={call.tool_call_id}",
                    }
                )
        for sse_error in run.sse_protocol_errors:
            if sse_error.partial_arguments_ref is None:
                findings.append(
                    {
                        "issue": "sse_protocol_error 缺少 partial_arguments_ref",
                        "run_id": sse_error.run_id,
                        "iteration_id": sse_error.iteration_id,
                        "detail": (
                            f"error_type={sse_error.error_type}, partial_tool_name="
                            f"{sse_error.partial_tool_name or '-'}"
                        ),
                    }
                )
    return findings


def _collect_sse_protocol_errors(runs: list[RunInfo]) -> list[SSEProtocolErrorInfo]:
    """收集并排序全部 SSE 协议错误记录。"""

    errors = [item for run in runs for item in run.sse_protocol_errors]
    return sorted(
        errors,
        key=lambda item: (
            item.recorded_at,
            item.run_id,
            _extract_iteration_index(item.iteration_id),
        ),
    )


def _build_prompt_pairs(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """按 scene 提取一组代表性的 system prompt / user prompt 对。"""

    findings: list[dict[str, Any]] = []
    seen_scenes: set[str] = set()
    for run in runs:
        ordered_contexts = sorted(run.iteration_contexts, key=lambda item: (item.iteration_index, item.recorded_at))
        for context in ordered_contexts:
            scene_name = context.identity.scene_name or context.session_id or "unknown_scene"
            if scene_name in seen_scenes:
                continue
            messages = _load_raw_messages(context)
            system_messages = [
                str(item.get("content") or "")
                for item in messages
                if str(item.get("role") or "") == "system" and str(item.get("content") or "").strip()
            ]
            user_messages = [
                str(item.get("content") or "")
                for item in messages
                if str(item.get("role") or "") == "user" and str(item.get("content") or "").strip()
            ]
            system_prompt = "\n\n".join(system_messages).strip()
            user_prompt = _select_current_user_prompt(
                current_user_message=context.current_user_message,
                user_messages=user_messages,
            )
            if not system_prompt and not user_prompt:
                continue
            seen_scenes.add(scene_name)
            findings.append(
                {
                    "run_id": context.run_id,
                    "iteration_id": context.iteration_id,
                    "iteration_index": context.iteration_index,
                    "scene_name": scene_name,
                    "model_name": context.identity.model_name,
                    "system_prompt": system_prompt,
                    "system_message_count": len(system_messages),
                    "user_prompt": user_prompt,
                }
            )
    return findings


def _load_raw_messages(context: IterationContextInfo) -> list[dict[str, Any]]:
    """从冷存输入中提取原始消息列表。"""

    payload = _load_raw_json(context.raw_input_ref)
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _select_current_user_prompt(*, current_user_message: str, user_messages: list[str]) -> str:
    """选择当前 iteration 对应的 user prompt。"""

    normalized_current = re.sub(r"\s+", " ", current_user_message or "").strip()
    if normalized_current:
        for message in reversed(user_messages):
            normalized_message = re.sub(r"\s+", " ", message).strip()
            if normalized_message == normalized_current:
                return message
        return current_user_message
    return user_messages[-1] if user_messages else ""


def _build_tool_schemas(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """聚合 trace 中出现过的工具 schema。"""

    grouped: dict[str, dict[str, Any]] = {}
    for run in runs:
        ordered_contexts = sorted(run.iteration_contexts, key=lambda item: (item.iteration_index, item.recorded_at))
        for context in ordered_contexts:
            raw_schemas = _load_raw_json(context.raw_tool_schemas_ref)
            loaded_tool_names: set[str] = set()
            if isinstance(raw_schemas, list):
                for schema in raw_schemas:
                    if not isinstance(schema, dict):
                        continue
                    function_payload = schema.get("function")
                    if not isinstance(function_payload, dict):
                        continue
                    tool_name = str(function_payload.get("name") or "").strip()
                    if not tool_name:
                        continue
                    loaded_tool_names.add(tool_name)
                    entry = grouped.setdefault(
                        tool_name,
                        {
                            "tool_name": tool_name,
                            "variants": {},
                        },
                    )
                    schema_key = _stable_json(schema)
                    variant = entry["variants"].setdefault(
                        schema_key,
                        {
                            "schema": schema,
                            "seen_in": [],
                        },
                    )
                    variant["seen_in"].append(
                        {
                            "run_id": context.run_id,
                            "iteration_id": context.iteration_id,
                        }
                    )

            for tool_name in context.tool_schema_names:
                entry = grouped.setdefault(
                    tool_name,
                    {
                        "tool_name": tool_name,
                        "variants": {},
                    },
                )
                if tool_name in loaded_tool_names:
                    continue
                variant = entry["variants"].setdefault(
                    "__missing__",
                    {
                        "schema": None,
                        "seen_in": [],
                    },
                )
                variant["seen_in"].append(
                    {
                        "run_id": context.run_id,
                        "iteration_id": context.iteration_id,
                    }
                )

    findings: list[dict[str, Any]] = []
    for tool_name in sorted(grouped):
        variants = list(grouped[tool_name]["variants"].values())
        variants.sort(key=lambda item: (_stable_json(item["schema"]) if item["schema"] is not None else ""))
        findings.append(
            {
                "tool_name": tool_name,
                "variants": variants,
            }
        )
    return findings


def _analyze_search_document_quality(runs: list[RunInfo]) -> dict[str, Any]:
    """分析 `search_document` 的可操作性质量。"""

    calls = [call for run in runs for call in run.tool_calls if call.tool_name == "search_document" and isinstance(call.result_data, dict)]
    if not calls:
        return {"call_count": 0}

    exact_ratios: list[float] = []
    exact_top_ranked_count = 0
    matched_query_coverage = 0
    is_exact_phrase_coverage = 0
    shown_counts: list[int] = []
    total_matches: list[int] = []
    problematic_calls: list[dict[str, Any]] = []

    for call in calls:
        result_data = call.result_data or {}
        matches = result_data.get("matches")
        if not isinstance(matches, list):
            matches = []
        shown_count = len(matches)
        shown_counts.append(shown_count)
        total_matches.append(_to_int(result_data.get("total_matches"), default=shown_count))

        exact_flags = [bool(item.get("is_exact_phrase")) for item in matches if isinstance(item, dict)]
        matched_query_flags = [("matched_query" in item) for item in matches if isinstance(item, dict)]
        exact_count = sum(1 for flag in exact_flags if flag)
        exact_ratio = _to_float(exact_count / shown_count if shown_count else 0.0)
        exact_ratios.append(exact_ratio)

        if matched_query_flags and all(matched_query_flags):
            matched_query_coverage += 1
        if exact_flags and len(exact_flags) == shown_count:
            is_exact_phrase_coverage += 1

        if _is_exact_top_ranked(matches):
            exact_top_ranked_count += 1
        else:
            problematic_calls.append(
                {
                    "run_id": call.run_id,
                    "iteration_id": call.iteration_id,
                    "tool_name": call.tool_name,
                    "reason": "exact / expansion 混排，模型很难直接判断优先读哪条",
                    "args_summary": _args_summary(call.tool_name, call.arguments),
                }
            )

    return {
        "call_count": len(calls),
        "avg_exact_ratio": round(sum(exact_ratios) / len(exact_ratios), 3) if exact_ratios else 0.0,
        "exact_top_ranked_rate": round(_safe_ratio(exact_top_ranked_count, len(calls)), 3),
        "matched_query_coverage_rate": round(_safe_ratio(matched_query_coverage, len(calls)), 3),
        "is_exact_phrase_coverage_rate": round(_safe_ratio(is_exact_phrase_coverage, len(calls)), 3),
        "median_shown_count": int(median(shown_counts)) if shown_counts else 0,
        "median_total_matches": int(median(total_matches)) if total_matches else 0,
        "problematic_calls": problematic_calls[:TOP_RECORD_LIMIT],
    }


def _is_exact_top_ranked(matches: list[Any]) -> bool:
    """判断 exact 结果是否全部排在 expansion 结果之前。"""

    seen_non_exact = False
    for item in matches:
        if not isinstance(item, dict):
            continue
        is_exact = bool(item.get("is_exact_phrase"))
        if not is_exact:
            seen_non_exact = True
            continue
        if seen_non_exact:
            return False
    return True


def _analyze_list_documents_quality(runs: list[RunInfo]) -> dict[str, Any]:
    """分析 `list_documents` 的下一步动作友好性。"""

    calls = [call for run in runs for call in run.tool_calls if call.tool_name == "list_documents" and isinstance(call.result_data, dict)]
    if not calls:
        return {"call_count": 0}

    recommended_count = 0
    raw_bytes: list[int] = []
    documents_count: list[int] = []
    large_calls: list[dict[str, Any]] = []
    large_threshold = _adaptive_large_threshold(
        [call.raw_result_ref.bytes for call in calls if call.raw_result_ref is not None and call.raw_result_ref.bytes > 0]
    )

    for call in calls:
        result_data = call.result_data or {}
        recommended = result_data.get("recommended_documents")
        if isinstance(recommended, dict) and recommended:
            recommended_count += 1
        documents = result_data.get("documents")
        if isinstance(documents, list):
            documents_count.append(len(documents))
        if call.raw_result_ref is not None and call.raw_result_ref.bytes > 0:
            raw_bytes.append(call.raw_result_ref.bytes)
            if call.raw_result_ref.bytes >= large_threshold:
                large_calls.append(
                    {
                        "run_id": call.run_id,
                        "iteration_id": call.iteration_id,
                        "raw_result_bytes": call.raw_result_ref.bytes,
                        "args_summary": _args_summary(call.tool_name, call.arguments),
                    }
                )

    return {
        "call_count": len(calls),
        "recommended_documents_coverage_rate": round(_safe_ratio(recommended_count, len(calls)), 3),
        "median_documents_count": int(median(documents_count)) if documents_count else 0,
        "median_result_bytes": int(median(raw_bytes)) if raw_bytes else 0,
        "large_calls": large_calls[:TOP_RECORD_LIMIT],
    }


def _analyze_fetch_more_quality(runs: list[RunInfo]) -> dict[str, Any]:
    """分析 `fetch_more` 的使用情况。"""

    calls = [call for run in runs for call in run.tool_calls if call.tool_name == "fetch_more" and isinstance(call.result_data, dict)]
    if not calls:
        return {"call_count": 0}

    exact_counts: list[int] = []
    expansion_counts: list[int] = []
    fetched_counts: list[int] = []
    for call in calls:
        result_data = call.result_data or {}
        matches = result_data.get("matches")
        if not isinstance(matches, list):
            matches = []
        fetched_counts.append(len(matches))
        exact_counts.append(sum(1 for item in matches if isinstance(item, dict) and bool(item.get("is_exact_phrase"))))
        expansion_counts.append(sum(1 for item in matches if isinstance(item, dict) and not bool(item.get("is_exact_phrase"))))
    return {
        "call_count": len(calls),
        "median_fetched_count": int(median(fetched_counts)) if fetched_counts else 0,
        "median_exact_count": int(median(exact_counts)) if exact_counts else 0,
        "median_expansion_count": int(median(expansion_counts)) if expansion_counts else 0,
    }


def _build_web_fetch_gaps(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """检测 search_web 命中后未继续 fetch_web_page 的链路。

    规则：
    - 只看 `search_web.total > 0` 且存在首选结果或 `results` 非空的调用；
    - 若在下一次 `search_web` 之前都没有出现 `fetch_web_page`，判为可疑。
    - 若工具结果已显式给出 `preferred_result + next_action=fetch_web_page`，则判为
      “模型忽略显式抓正文信号”；
      否则判为“工具结果仍缺少显式抓正文信号”。
    """

    findings: list[dict[str, Any]] = []
    for run in runs:
        ordered = sorted(run.tool_calls, key=_tool_call_sort_key)
        for index, call in enumerate(ordered):
            if call.tool_name != "search_web" or not isinstance(call.result_data, dict):
                continue
            preferred_result = call.result_data.get("preferred_result")
            if not isinstance(preferred_result, dict):
                preferred_result = {}
            results = call.result_data.get("results")
            total = _to_int(call.result_data.get("total"))
            if not isinstance(results, list):
                results = []
            lead_result = preferred_result or (results[0] if results and isinstance(results[0], dict) else {})
            if total <= 0 or not isinstance(lead_result, dict) or not lead_result:
                continue
            preferred_summary = _to_optional_text(call.result_data.get("preferred_result_summary"))
            next_action = _to_optional_text(call.result_data.get("next_action"))
            next_action_args = call.result_data.get("next_action_args")
            if not isinstance(next_action_args, dict):
                next_action_args = {}
            next_action_url = _to_optional_text(next_action_args.get("url"))
            lead_url = _to_optional_text(lead_result.get("url"))
            has_preferred_result = bool(preferred_result)
            has_explicit_fetch_signal = (
                has_preferred_result
                and next_action == "fetch_web_page"
                and bool(next_action_url)
                and (lead_url is None or next_action_url == lead_url)
            )

            next_fetch: Optional[ToolCallInfo] = None
            for follower in ordered[index + 1 :]:
                if follower.tool_name == "fetch_web_page":
                    next_fetch = follower
                    break
                if follower.tool_name == "search_web":
                    break

            if next_fetch is not None:
                continue

            gap_kind = (
                "ignored_explicit_fetch_signal"
                if has_explicit_fetch_signal
                else "missing_explicit_fetch_signal"
            )
            reason = (
                "search_web 已显式返回 preferred_result 与 `next_action=fetch_web_page`，但后续仍未继续抓正文。"
                if gap_kind == "ignored_explicit_fetch_signal"
                else "search_web 已命中网页入口，但返回结果没有把“下一步就抓这页正文”压缩成足够显式的动作。"
            )
            findings.append(
                {
                    "run_id": run.run_id,
                    "iteration_id": call.iteration_id,
                    "query": str(call.result_data.get("query") or call.arguments.get("query") or ""),
                    "result_count": total,
                    "gap_kind": gap_kind,
                    "has_preferred_result": has_preferred_result,
                    "preferred_summary": preferred_summary or "",
                    "next_action": next_action or "",
                    "next_action_url": next_action_url or "",
                    "top_url": str(lead_result.get("url") or ""),
                    "top_title": str(lead_result.get("title") or ""),
                    "reason": reason,
                }
            )
    return findings


def _build_web_search_empty_streaks(runs: list[RunInfo]) -> list[dict[str, Any]]:
    """检测连续多次 `search_web` 空结果的空转链路。"""

    findings: list[dict[str, Any]] = []
    for run in runs:
        ordered = sorted(run.tool_calls, key=_tool_call_sort_key)
        streak: list[ToolCallInfo] = []
        for call in ordered:
            is_empty_search = (
                call.tool_name == "search_web"
                and isinstance(call.result_data, dict)
                and _to_int(call.result_data.get("total")) == 0
                and not (call.result_data.get("results") or [])
            )
            if is_empty_search:
                streak.append(call)
                continue
            if len(streak) >= 2:
                findings.append(
                    {
                        "run_id": run.run_id,
                        "start_iteration_id": streak[0].iteration_id,
                        "end_iteration_id": streak[-1].iteration_id,
                        "count": len(streak),
                        "queries": [str((item.result_data or {}).get("query") or item.arguments.get("query") or "") for item in streak],
                        "reason": "连续多次 search_web 均为空结果，模型正在空转改 query，而不是切换到更稳的取证路径。",
                    }
                )
            streak = []
        if len(streak) >= 2:
            findings.append(
                {
                    "run_id": run.run_id,
                    "start_iteration_id": streak[0].iteration_id,
                    "end_iteration_id": streak[-1].iteration_id,
                    "count": len(streak),
                    "queries": [str((item.result_data or {}).get("query") or item.arguments.get("query") or "") for item in streak],
                    "reason": "连续多次 search_web 均为空结果，模型正在空转改 query，而不是切换到更稳的取证路径。",
                }
            )
    return findings


def _build_recommendations(bundle: AnalysisBundle) -> list[dict[str, str]]:
    """根据分析结果生成优化建议。"""

    recommendations: list[dict[str, str]] = []
    missing_web_fetch_signals = [
        item for item in bundle.web_fetch_gaps if item.get("gap_kind") == "missing_explicit_fetch_signal"
    ]
    ignored_web_fetch_signals = [
        item for item in bundle.web_fetch_gaps if item.get("gap_kind") == "ignored_explicit_fetch_signal"
    ]

    if bundle.trace_integrity_issues:
        recommendations.append(
            {
                "priority": "P0",
                "title": "先修 trace 完整性，再做工具优化判断",
                "reason": "当前 trace 存在 `iteration_context_snapshot.tool_calls` 空值或计数不一致问题；这会污染对“模型看到了什么、下一步为什么这么做”的判断。",
                "action": "修正 trace 导出，使 `iteration_context_snapshot.tool_calls` 与同一 iteration 的 `tool_call` 记录一一对应，并保证不写入空值。",
            }
        )

    if bundle.sse_protocol_errors:
        recommendations.append(
            {
                "priority": "P0",
                "title": "优先检查 SSE 协议错误现场，确认问题在工具参数生成还是流式输出稳定性",
                "reason": "当前 trace 已记录到 `sse_protocol_error`。这类 run 的失败点通常不是工具执行失败，而是模型在流式 tool call 阶段输出了未闭合或非法的 arguments；如果不先看 `partial_tool_name` 与 `sse_error_*.json`，后续所有“工具成功率/降级率”判断都容易偏题。",
                "action": "优先按报告中的 `partial_tool_name / request_id / attempt / raw_blob` 回放失败现场；先确认是哪一个工具、哪段 arguments 前缀被截断，再决定是收窄 schema、缩短参数面，还是排查上游模型/网关的 SSE 稳定性。",
            }
        )

    if bundle.large_payload_calls:
        recommendations.append(
            {
                "priority": "P1",
                "title": "结果返回改成 summary-first + handle-first，降低首屏认知负担",
                "reason": "存在结果载荷显著高于同类调用分布的工具调用；模型在首屏读到过大的 payload 时，更容易走捷径、误抓局部文本、或者直接触发上下文压力。",
                "action": "优先返回“下一步决策所需的最小摘要 + 稳定句柄/游标”；把大正文、长列表、完整证据下放到续读或定点读取工具。",
            }
        )

    missing_fetch_more_contract_cases = [
        item
        for item in bundle.truncation_issues
        if not item["has_hint"]
        and item["next_action"] == "fetch_more"
        and item["tool_name"] != "fetch_more"
        and not item["used_fetch_more"]
    ]
    if missing_fetch_more_contract_cases:
        recommendations.append(
            {
                "priority": "P1",
                "title": "强化截断 contract，把“下一步该怎么读”表达成机械可执行动作",
                "reason": "trace 中存在“无 hint 且 `truncation.next_action=fetch_more`，但模型没有继续 `fetch_more`”的情况。这说明截断 contract 仍不足以把“继续同一结果”压缩成足够低认知负担的机械动作。",
                "action": "确保截断结果固定包含简短摘要、`has_more`、`fetch_more_args`，并让 `result_summary` 明确提示“若继续看同一结果，请直接 fetch_more”。",
            }
        )

    if bundle.duplicate_calls:
        recommendations.append(
            {
                "priority": "P1",
                "title": "减少重复调用：让 schema 和 result_summary 更强地支持复用已有结果",
                "reason": "同一 run 内出现相同工具 + 相同参数的重复调用，说明模型没有稳定地判断“已有结果已足够支撑下一步动作”。",
                "action": "收窄参数面、增强结果摘要中的“已拿到什么/下一步建议读什么”，必要时让结果显式暴露可复用的 section_ref / document_id / handle。",
            }
        )

    if bundle.context_pressure_runs:
        recommendations.append(
            {
                "priority": "P1",
                "title": "把工具设计目标从“给全”改成“让下一步更稳”",
                "reason": "trace 中出现上下文压缩、软限超标、续写或降级 run，说明当前 prompt 与工具结果组合后的输入负担偏高。",
                "action": "优先减少目录类工具和搜索类工具的首屏返回规模，把长结果拆成推荐入口 + 定点读取；避免让模型在同一 agent iteration 同时面对大量候选和大量正文。",
            }
        )

    if bundle.failure_patterns:
        top_failure = bundle.failure_patterns[0]
        recommendations.append(
            {
                "priority": "P2",
                "title": "把失败做成可恢复分支，而不是模糊错误",
                "reason": f"当前失败最集中在 `{top_failure['tool_name']}` / `{top_failure['error_code']}`。如果失败信息不够机械可操作，模型通常只会重试或改换无关工具。",
                "action": "错误返回中优先暴露 machine-actionable 字段：缺什么参数、推荐改用哪个工具、推荐最小修复动作；避免仅返回人类可读长文本。",
            }
        )

    fetch_web_page_failures = [
        item for item in bundle.detailed_failure_patterns if item["tool_name"] == "fetch_web_page"
    ]
    if fetch_web_page_failures:
        top_fetch_failure = fetch_web_page_failures[0]
        recommendations.append(
            {
                "priority": "P2",
                "title": "把 fetch_web_page 的失败类型做成可机械分流的错误 contract",
                "reason": f"`fetch_web_page` 当前高频失败签名是 `{top_fetch_failure['error_signature']}`。仅暴露粗粒度 `EXECUTION_ERROR` 对工具优化没有帮助，模型也难以学会稳定换路。",
                "action": "在错误返回中稳定暴露更细粒度字段，例如 `http_status`、`failure_kind`、`retryable`、`allowed_url`；若是 URL 白名单拦截，应直接给出“禁止抓取，改用 search_web 或财报工具”的 next action。",
            }
        )

    if bundle.search_quality.get("call_count", 0) > 0 and bundle.search_quality.get("exact_top_ranked_rate", 1.0) < 1.0:
        recommendations.append(
            {
                "priority": "P2",
                "title": "把 search_document 的排序语义做得更直接",
                "reason": "若 exact 与 expansion 混排，模型需要额外推理“哪条更值得先读”，这增加了认知负担和误判概率。",
                "action": "优先保证 exact 命中排在 expansion 前，并持续返回 `matched_query` / `is_exact_phrase` 等低成本判别信号。",
            }
        )

    if bundle.list_documents_quality.get("call_count", 0) > 0 and bundle.list_documents_quality.get("large_calls"):
        recommendations.append(
            {
                "priority": "P2",
                "title": "目录类工具优先返回推荐入口，而不是完整清单",
                "reason": "`list_documents` 的部分结果载荷偏大；对模型来说，完整清单往往不是下一步动作所需的最小信息。",
                "action": "保留 `recommended_documents` 作为首屏主路径，把完整文档清单降为次级信息，或支持显式分页/过滤。",
            }
        )

    if missing_web_fetch_signals:
        recommendations.append(
            {
                "priority": "P2",
                "title": "让 web 检索结果显式驱动下一步抓正文",
                "reason": "存在 `search_web` 已命中候选网页，但返回结果仍没有把“下一步就抓这页正文”表达成足够低认知负担的显式动作。",
                "action": "在 `search_web` 返回中强化首选结果摘要与 next action 提示；必要时返回更强的推荐字段（例如 recommended_url / recommended_reason）。",
            }
        )

    if ignored_web_fetch_signals:
        recommendations.append(
            {
                "priority": "P2",
                "title": "让模型优先执行 search_web 已给出的抓正文动作",
                "reason": "存在 `search_web` 已显式返回 `preferred_result + next_action=fetch_web_page`，但模型后续仍未继续抓正文的链路。这说明问题已不主要在工具返回结构，而在工具使用约束或执行稳定性。",
                "action": "在 tools guidance / task prompt / result_summary 中明确：当 `search_web.next_action=fetch_web_page` 且 `next_action_args.url` 存在时，优先直接抓正文；只有抓取失败或明显不相关时才改 query 或换来源。",
            }
        )

    if bundle.web_search_empty_streaks:
        recommendations.append(
            {
                "priority": "P2",
                "title": "为空 web 检索提供更明确的停损与换路信号",
                "reason": "存在连续多次 `search_web` 结果全空的空转链路。模型没有稳定判断“继续改 query 已经不划算，应该换回财报证据或直接承认无公开数据”。",
                "action": "为空结果统一返回更强的 next-action 信号，例如 `continue_without_web` / `change_source`，并减少让模型自行盲目改 query 的空间。",
            }
        )

    return recommendations


def _analyze_manifest(manifest_path: Optional[Path]) -> Optional[dict[str, Any]]:
    """加载 manifest 摘要。"""

    if manifest_path is None or not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(manifest, dict):
        return None
    config = manifest.get("config")
    if not isinstance(config, dict):
        config = {}
    chapter_results = manifest.get("chapter_results")
    if not isinstance(chapter_results, dict):
        chapter_results = {}

    chapters: list[dict[str, Any]] = []
    for title, chapter in chapter_results.items():
        if not isinstance(chapter, dict):
            continue
        process_state = chapter.get("process_state")
        if not isinstance(process_state, dict):
            process_state = {}
        audit_history = process_state.get("audit_history")
        if not isinstance(audit_history, list):
            audit_history = []
        chapters.append(
            {
                "title": str(title),
                "status": str(chapter.get("status") or "unknown"),
                "audit_passed": chapter.get("audit_passed"),
                "retry_count": _to_int(chapter.get("retry_count")),
                "rewrite_exhausted": bool(process_state.get("rewrite_exhausted")),
                "audit_rounds": len(audit_history),
                "failure_reason": str(chapter.get("failure_reason") or ""),
            }
        )
    return {
        "ticker": str(config.get("ticker") or ""),
        "write_model": str(config.get("write_model_name") or "（未记录）"),
        "audit_model": str(config.get("audit_model_name") or "（未记录）"),
        "chapters": chapters,
    }


def _build_analysis_bundle(runs: list[RunInfo]) -> AnalysisBundle:
    """构建完整分析结果。"""

    tool_stats = _summarize_tool_stats(runs)
    duplicate_calls = _build_duplicate_calls(runs)
    truncation_issues = _build_truncation_issues(runs)
    large_payload_calls = _build_large_payload_calls(runs)
    large_prompt_iterations = _build_large_prompt_iterations(runs)
    failure_patterns = _build_failure_patterns(runs)
    detailed_failure_patterns = _build_detailed_failure_patterns(runs)
    context_pressure_runs = _build_context_pressure_runs(runs)
    trace_integrity_issues = _build_trace_integrity_issues(runs)
    prompt_pairs = _build_prompt_pairs(runs)
    tool_schemas = _build_tool_schemas(runs)
    search_quality = _analyze_search_document_quality(runs)
    list_documents_quality = _analyze_list_documents_quality(runs)
    fetch_more_quality = _analyze_fetch_more_quality(runs)
    web_fetch_gaps = _build_web_fetch_gaps(runs)
    web_search_empty_streaks = _build_web_search_empty_streaks(runs)
    sse_protocol_errors = _collect_sse_protocol_errors(runs)
    bundle = AnalysisBundle(
        runs=runs,
        sse_protocol_errors=sse_protocol_errors,
        tool_stats=tool_stats,
        duplicate_calls=duplicate_calls,
        truncation_issues=truncation_issues,
        large_payload_calls=large_payload_calls,
        large_prompt_iterations=large_prompt_iterations,
        failure_patterns=failure_patterns,
        detailed_failure_patterns=detailed_failure_patterns,
        context_pressure_runs=context_pressure_runs,
        trace_integrity_issues=trace_integrity_issues,
        prompt_pairs=prompt_pairs,
        tool_schemas=tool_schemas,
        search_quality=search_quality,
        list_documents_quality=list_documents_quality,
        fetch_more_quality=fetch_more_quality,
        web_fetch_gaps=web_fetch_gaps,
        web_search_empty_streaks=web_search_empty_streaks,
        recommendations=[],
    )
    bundle.recommendations = _build_recommendations(bundle)
    return bundle


def analyze(
    trace_path: Path,
    manifest_path: Optional[Path] = None,
    ticker: Optional[str] = None,
) -> str:
    """执行 trace 分析并返回 Markdown。

    Args:
        trace_path: trace 文件或目录。
        manifest_path: 可选 manifest 路径。
        ticker: 可选 ticker，用于报告标题。

    Returns:
        Markdown 报告文本。

    Raises:
        FileNotFoundError: trace 路径不存在时抛出。
        RuntimeError: 未发现任何 V2 记录时抛出。
    """

    records = _load_jsonl_files(trace_path)
    v2_records = [record for record in records if record.get("trace_schema_version") == TRACE_SCHEMA_VERSION]
    if not v2_records:
        raise RuntimeError("未发现 tool_trace_v2 记录；当前 analyzer 已按 V2 schema 收口")
    runs = _group_runs(v2_records)
    manifest = _analyze_manifest(manifest_path)
    bundle = _build_analysis_bundle(runs)
    return _render_report(bundle=bundle, ticker=ticker, manifest=manifest, trace_path=trace_path)


def _render_report(
    *,
    bundle: AnalysisBundle,
    ticker: Optional[str],
    manifest: Optional[dict[str, Any]],
    trace_path: Path,
) -> str:
    """渲染最终 Markdown 报告。"""

    title_ticker = ticker or (manifest or {}).get("ticker") or "未知标的"
    lines: list[str] = [
        f"# Tool Trace 分析报告（{title_ticker}）",
        "",
        "## 0. 分析目标",
        "",
        "本报告只基于 `tool_trace_v2`，目标不是泛化日志统计，而是回答一个更具体的问题：",
        "",
        "> 当前的 tool schema 与 tool 返回数据，是否让一个无状态、会犯错、会走捷径、上下文有限、偏好模式匹配的推理器，",
        "> 在最低认知负担下稳定做对下一步动作？",
        "",
        f"- **trace_path**: `{trace_path}`",
        f"- **run 数量**: {len(bundle.runs)}",
        f"- **scene**: {_fmt_set({run.identity.scene_name for run in bundle.runs if run.identity.scene_name})}",
        f"- **model**: {_fmt_set({run.identity.model_name for run in bundle.runs if run.identity.model_name})}",
        f"- **agent**: {_fmt_set({run.identity.agent_name for run in bundle.runs if run.identity.agent_name})}",
        f"- **capabilities**: {_fmt_set({cap for run in bundle.runs for cap in run.identity.enabled_capabilities})}",
        "",
    ]

    lines.extend(_render_executive_summary(bundle))
    lines.extend(_render_run_overview(bundle))
    lines.extend(_render_tool_stats(bundle))
    lines.extend(_render_cognitive_load_signals(bundle))
    lines.extend(_render_reliability_signals(bundle))
    lines.extend(_render_trace_integrity(bundle))
    lines.extend(_render_run_chains(bundle))
    lines.extend(_render_recommendations(bundle))
    lines.extend(_render_prompt_pairs(bundle))
    lines.extend(_render_tool_schemas(bundle))
    if manifest:
        lines.extend(_render_manifest_section(manifest))
    return "\n".join(lines).rstrip() + "\n"


def _render_executive_summary(bundle: AnalysisBundle) -> list[str]:
    """渲染执行摘要。"""

    degraded_runs = sum(1 for run in bundle.runs if run.degraded)
    interrupted_runs = sum(1 for run in bundle.runs if not run.has_final_response)
    sse_protocol_error_runs = sum(1 for run in bundle.runs if run.sse_protocol_error_count > 0)
    total_tool_calls = sum(run.total_tool_calls for run in bundle.runs)
    successful_tool_calls = sum(run.successful_tool_calls for run in bundle.runs)
    lines = [
        "## 1. 执行摘要",
        "",
        f"- **总工具调用数**: {total_tool_calls}",
        f"- **工具成功率**: {_fmt_ratio(successful_tool_calls, total_tool_calls)}",
        f"- **降级 run 数**: {degraded_runs}/{len(bundle.runs)}",
        f"- **中断 run 数**: {interrupted_runs}/{len(bundle.runs)}",
        f"- **SSE 协议错误 run 数**: {sse_protocol_error_runs}/{len(bundle.runs)}",
    ]
    if bundle.recommendations:
        lines.append("- **优先建议**:")
        for item in bundle.recommendations[:3]:
            lines.append(f"  - `{item['priority']}` {item['title']}：{item['reason']}")
    lines.append("")
    return lines


def _fmt_ratio(numerator: int, denominator: int) -> str:
    """格式化比例。"""

    if denominator <= 0:
        return "0/0 (0.0%)"
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def _render_run_overview(bundle: AnalysisBundle) -> list[str]:
    """渲染 run 总览。"""

    lines = [
        "## 2. Run 总览",
        "",
        "| run_id | scene | model | iterations | tool_calls | degraded | final_response | sse_error | max_input_bytes | 首轮用户问题 |",
        "|---|---|---:|---:|---:|---|---|---|---:|---|",
    ]
    for run in bundle.runs:
        lines.append(
            f"| `{run.run_id}` | `{run.identity.scene_name or '-'}` | `{run.identity.model_name or '-'}` "
            f"| {run.iteration_count} | {run.total_tool_calls} | {'是' if run.degraded else '否'} "
            f"| {'是' if run.has_final_response else '否'} | {'是' if run.sse_protocol_error_count > 0 else '否'} "
            f"| {run.max_raw_input_bytes:,} | {_md_escape(_excerpt(run.first_user_message))} |"
        )
    lines.append("")
    return lines


def _render_tool_stats(bundle: AnalysisBundle) -> list[str]:
    """渲染工具级总表。"""

    lines = [
        "## 3. 工具级诊断总表",
        "",
        "| tool | calls | success_rate | truncation_rate | median_latency_ms | median_result_bytes | p90_result_bytes | median_arg_keys | top_errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for stat in bundle.tool_stats:
        top_errors = ", ".join(f"{code}:{count}" for code, count in stat.top_error_codes) or "-"
        lines.append(
            f"| `{stat.tool_name}` | {stat.call_count} | {stat.success_rate:.1%} | {stat.truncation_rate:.1%} "
            f"| {stat.median_latency_ms} | {stat.median_result_bytes:,} | {stat.p90_result_bytes:,} "
            f"| {stat.median_argument_keys} | {top_errors} |"
        )
    lines.append("")
    return lines


def _render_cognitive_load_signals(bundle: AnalysisBundle) -> list[str]:
    """渲染认知负担信号。"""

    lines = [
        "## 4. 认知负担信号",
        "",
        "### 4.1 大结果负载",
        "",
    ]
    if not bundle.large_payload_calls:
        lines.append("- 未发现显著高于整体分布的大结果负载调用。")
    else:
        lines.append("| run_id | iteration_id | tool | result_bytes | 阈值 | 参数摘要 | 结果摘要 |")
        lines.append("|---|---|---|---:|---:|---|---|")
        for item in bundle.large_payload_calls[:TOP_RECORD_LIMIT]:
            lines.append(
                f"| `{item['run_id']}` | `{item['iteration_id']}` | `{item['tool_name']}` | {item['raw_result_bytes']:,} "
                f"| {item['threshold']:,} | {_md_escape(item['args_summary'])} | {_md_escape(_excerpt(item['result_summary']))} |"
            )

    lines.extend(
        [
            "",
            "### 4.2 大输入上下文",
            "",
            "- 这里不按 `model_input_messages_summary.excerpt` 的长度做判断；V2 trace 已按 `source_tag` 分级截断 excerpt。",
            "- 输入结构判断应以 `raw_input_ref.bytes`、`context_meta`、`message_mix` 为主，而不是把各类 excerpt 当成等长文本。",
            "",
        ]
    )
    if not bundle.large_prompt_iterations:
        lines.append("- 未发现显著高于整体分布的大输入快照。")
    else:
        lines.append("| run_id | iteration_id | input_bytes | 阈值 | over_soft_limit | compaction_count | message_mix | current_user_message |")
        lines.append("|---|---|---:|---:|---|---:|---|---|")
        for item in bundle.large_prompt_iterations[:TOP_RECORD_LIMIT]:
            lines.append(
                f"| `{item['run_id']}` | `{item['iteration_id']}` | {item['raw_input_bytes']:,} | {item['threshold']:,} "
                f"| {'是' if item['is_over_soft_limit'] else '否'} | {item['compaction_count']} "
                f"| {_md_escape(item['message_mix'])} | {_md_escape(item['current_user_message'])} |"
            )

    lines.extend(
        [
            "",
            "### 4.3 截断与续读",
            "",
        ]
    )
    if not bundle.truncation_issues:
        lines.append("- 未发现截断工具调用。")
    else:
        lines.append("| run_id | iteration_id | tool | hint | next_action | fetch_more_args | 后续是否 fetch_more | 下一工具 | result_bytes | 参数摘要 |")
        lines.append("|---|---|---|---|---|---|---|---|---:|---|")
        for item in bundle.truncation_issues[:TOP_RECORD_LIMIT]:
            lines.append(
                f"| `{item['run_id']}` | `{item['iteration_id']}` | `{item['tool_name']}` | {'有' if item['has_hint'] else '无'} "
                f"| {_md_escape(item['next_action'] or '-')} | {'有' if item['has_fetch_more_args'] else '无'} "
                f"| {'是' if item['used_fetch_more'] else '否'} | `{item['next_tool_name'] or '-'}` | {item['raw_result_bytes']:,} "
                f"| {_md_escape(item['args_summary'])} |"
            )

    lines.extend(
        [
            "",
            "### 4.4 重复调用",
            "",
        ]
    )
    if not bundle.duplicate_calls:
        lines.append("- 未发现显著重复调用。")
    else:
        lines.append("| run_id | tool | count | iteration_ids | 参数摘要 |")
        lines.append("|---|---|---:|---|---|")
        for item in bundle.duplicate_calls[:TOP_RECORD_LIMIT]:
            lines.append(
                f"| `{item['run_id']}` | `{item['tool_name']}` | {item['call_count']} | "
                f"{_md_escape(', '.join(item['iteration_ids']))} | {_md_escape(item['args_summary'])} |"
            )
    lines.append("")
    return lines


def _render_reliability_signals(bundle: AnalysisBundle) -> list[str]:
    """渲染可靠性信号。"""

    lines = [
        "## 5. 可靠性与降级信号",
        "",
        "### 5.1 失败模式",
        "",
    ]
    if not bundle.failure_patterns:
        lines.append("- 未发现工具失败记录。")
    else:
        lines.append("| tool | error_code | count |")
        lines.append("|---|---|---:|")
        for item in bundle.failure_patterns[:TOP_RECORD_LIMIT]:
            lines.append(f"| `{item['tool_name']}` | `{item['error_code']}` | {item['count']} |")

    lines.extend(["", "### 5.1.1 详细失败签名", ""])
    if not bundle.detailed_failure_patterns:
        lines.append("- 未发现可细化的失败记录。")
    else:
        lines.append("| tool | error_signature | http_status | message | url | count | detail_excerpt |")
        lines.append("|---|---|---:|---|---|---:|---|")
        for item in bundle.detailed_failure_patterns[:TOP_RECORD_LIMIT]:
            lines.append(
                f"| `{item['tool_name']}` | `{item['error_signature']}` | {item['http_status']} "
                f"| {_md_escape(item['message'])} | {_md_escape(item['url'])} | {item['count']} | {_md_escape(item['detail_excerpt'])} |"
            )

    lines.extend(["", "### 5.2 上下文压力 run", ""])
    if not bundle.context_pressure_runs:
        lines.append("- 未发现明显上下文压力 run。")
    else:
        lines.append("| run_id | tool_calls | iterations | compaction | continuation | over_soft_limit | degraded | final_response | max_input_bytes |")
        lines.append("|---|---:|---:|---:|---:|---|---|---|---:|")
        for item in bundle.context_pressure_runs[:TOP_RECORD_LIMIT]:
            lines.append(
                f"| `{item['run_id']}` | {item['tool_calls']} | {item['iteration_count']} | {item['compaction_count']} | "
                f"{item['continuation_count']} | {'是' if item['over_soft_limit'] else '否'} | {'是' if item['degraded'] else '否'} "
                f"| {'是' if item['has_final_response'] else '否'} | {item['max_raw_input_bytes']:,} |"
            )

    lines.extend(["", "### 5.3 SSE 协议错误", ""])
    if not bundle.sse_protocol_errors:
        lines.append("- 当前 trace 未记录到 `sse_protocol_error`。")
    else:
        lines.append("| run_id | iteration_id | error_type | partial_tool_name | attempt | request_id | arguments_prefix | raw_blob |")
        lines.append("|---|---|---|---|---:|---|---|---|")
        for item in bundle.sse_protocol_errors[:TOP_RECORD_LIMIT]:
            raw_blob = item.partial_arguments_ref.storage_uri if item.partial_arguments_ref is not None else "-"
            lines.append(
                f"| `{item.run_id}` | `{item.iteration_id}` | `{item.error_type or '-'}` | "
                f"`{item.partial_tool_name or '-'}` | {item.attempt or 0} | `{item.request_id or '-'}` | "
                f"{_md_escape(_extract_sse_partial_arguments_excerpt(item))} | {_md_escape(raw_blob)} |"
            )

    lines.extend(["", "### 5.4 工具特定质量信号", ""])
    lines.extend(_render_search_document_quality(bundle.search_quality))
    lines.extend(_render_list_documents_quality(bundle.list_documents_quality))
    lines.extend(_render_fetch_more_quality(bundle.fetch_more_quality))
    lines.extend(_render_web_search_quality(bundle))
    lines.append("")
    return lines


def _render_search_document_quality(search_quality: dict[str, Any]) -> list[str]:
    """渲染 search_document 质量分析。"""

    if search_quality.get("call_count", 0) <= 0:
        return ["- `search_document`: 当前 trace 中无调用。"]
    lines = [
        f"- `search_document` 调用数: {search_quality['call_count']}",
        f"- 平均 exact 占比: {search_quality['avg_exact_ratio']:.1%}",
        f"- exact 排序正确率: {search_quality['exact_top_ranked_rate']:.1%}",
        f"- `matched_query` 覆盖率: {search_quality['matched_query_coverage_rate']:.1%}",
        f"- `is_exact_phrase` 覆盖率: {search_quality['is_exact_phrase_coverage_rate']:.1%}",
        f"- 中位 returned matches: {search_quality['median_shown_count']}",
        f"- 中位 total_matches: {search_quality['median_total_matches']}",
    ]
    if search_quality.get("problematic_calls"):
        lines.append("- 存在排序或判别信号不足的调用：")
        for item in search_quality["problematic_calls"]:
            lines.append(
                f"  - `{item['run_id']}` / `{item['iteration_id']}`: {_md_escape(item['reason'])}；"
                f"参数={_md_escape(item['args_summary'])}"
            )
    return lines


def _render_list_documents_quality(list_quality: dict[str, Any]) -> list[str]:
    """渲染 list_documents 质量分析。"""

    if list_quality.get("call_count", 0) <= 0:
        return ["- `list_documents`: 当前 trace 中无调用。"]
    lines = [
        f"- `list_documents` 调用数: {list_quality['call_count']}",
        f"- `recommended_documents` 覆盖率: {list_quality['recommended_documents_coverage_rate']:.1%}",
        f"- 中位 documents 数量: {list_quality['median_documents_count']}",
        f"- 中位结果字节数: {list_quality['median_result_bytes']:,}",
    ]
    if list_quality.get("large_calls"):
        lines.append("- 结果负载偏大的目录调用：")
        for item in list_quality["large_calls"]:
            lines.append(
                f"  - `{item['run_id']}` / `{item['iteration_id']}`: {item['raw_result_bytes']:,} bytes；"
                f"参数={_md_escape(item['args_summary'])}"
            )
    return lines


def _render_fetch_more_quality(fetch_quality: dict[str, Any]) -> list[str]:
    """渲染 fetch_more 质量分析。"""

    if fetch_quality.get("call_count", 0) <= 0:
        return ["- `fetch_more`: 当前 trace 中无调用。"]
    return [
        f"- `fetch_more` 调用数: {fetch_quality['call_count']}",
        f"- 中位续读条数: {fetch_quality['median_fetched_count']}",
        f"- 中位 exact 条数: {fetch_quality['median_exact_count']}",
        f"- 中位 expansion 条数: {fetch_quality['median_expansion_count']}",
    ]


def _render_web_search_quality(bundle: AnalysisBundle) -> list[str]:
    """渲染 web 搜索链路质量。"""

    lines: list[str] = []
    if not bundle.web_fetch_gaps and not bundle.web_search_empty_streaks:
        lines.append("- `search_web / fetch_web_page`: 未发现明显的 web 链路可疑模式。")
        return lines

    missing_web_fetch_signals = [
        item for item in bundle.web_fetch_gaps if item.get("gap_kind") == "missing_explicit_fetch_signal"
    ]
    ignored_web_fetch_signals = [
        item for item in bundle.web_fetch_gaps if item.get("gap_kind") == "ignored_explicit_fetch_signal"
    ]

    if missing_web_fetch_signals:
        lines.append("- `search_web` 命中后未继续 `fetch_web_page`，且返回未给出显式抓正文信号：")
        for item in missing_web_fetch_signals[:TOP_RECORD_LIMIT]:
            summary_preview = _excerpt(item.get("preferred_summary", ""), max_chars=70)
            lines.append(
                f"  - `{item['run_id']}` / `{item['iteration_id']}`: 命中 {item['result_count']} 条，"
                f"但后续未抓正文；query={_md_escape(_excerpt(item['query'], max_chars=100))}；"
                f"next_action={_md_escape(item.get('next_action') or '-')}；"
                f"preferred_result={'有' if item.get('has_preferred_result') else '无'}；"
                f"preferred_summary={_md_escape(summary_preview or '-')}"
            )

    if ignored_web_fetch_signals:
        lines.append("- `search_web` 已给出显式 `fetch_web_page` 信号，但模型仍未执行：")
        for item in ignored_web_fetch_signals[:TOP_RECORD_LIMIT]:
            summary_preview = _excerpt(item.get("preferred_summary", ""), max_chars=70)
            lines.append(
                f"  - `{item['run_id']}` / `{item['iteration_id']}`: 命中 {item['result_count']} 条，"
                f"但后续未抓正文；query={_md_escape(_excerpt(item['query'], max_chars=100))}；"
                f"next_action={_md_escape(item.get('next_action') or '-')}；"
                f"url={_md_escape(_excerpt(item.get('next_action_url') or item.get('top_url') or '-', max_chars=80))}；"
                f"preferred_summary={_md_escape(summary_preview or '-')}"
            )

    if bundle.web_search_empty_streaks:
        lines.append("- 连续空结果 `search_web` 空转链路：")
        for item in bundle.web_search_empty_streaks[:TOP_RECORD_LIMIT]:
            query_preview = " | ".join(_excerpt(query, max_chars=40) for query in item["queries"][:3])
            lines.append(
                f"  - `{item['run_id']}`: {item['count']} 次连续空检索"
                f"（{item['start_iteration_id']} -> {item['end_iteration_id']}）；queries={_md_escape(query_preview)}"
            )

    return lines


def _render_trace_integrity(bundle: AnalysisBundle) -> list[str]:
    """渲染 trace 完整性问题。"""

    lines = [
        "## 6. Trace 完整性",
        "",
    ]
    if not bundle.trace_integrity_issues:
        lines.append("- 未发现明显的 trace 完整性问题。")
        lines.append("")
        return lines

    lines.append("| issue | run_id | iteration_id | detail |")
    lines.append("|---|---|---|---|")
    for item in bundle.trace_integrity_issues[:TOP_RECORD_LIMIT]:
        lines.append(
            f"| {_md_escape(item['issue'])} | `{item['run_id']}` | `{item['iteration_id']}` | {_md_escape(item['detail'])} |"
        )
    lines.append("")
    return lines


def _render_run_chains(bundle: AnalysisBundle) -> list[str]:
    """渲染 run 级调用链。"""

    lines = [
        "## 7. Run 级调用链",
        "",
    ]
    for run in bundle.runs:
        lines.append(f"### {run.run_id}")
        lines.append("")
        lines.append(
            f"- scene=`{run.identity.scene_name or '-'}`, model=`{run.identity.model_name or '-'}`, "
            f"agent=`{run.identity.agent_name or '-'}`, degraded={'是' if run.degraded else '否'}, "
            f"final_response={'是' if run.has_final_response else '否'}, "
            f"sse_protocol_error={'是' if run.sse_protocol_error_count > 0 else '否'}"
        )
        if run.first_user_message:
            lines.append(f"- 首轮用户问题: {_md_escape(_excerpt(run.first_user_message, max_chars=140))}")
        latest_sse_protocol_error = run.latest_sse_protocol_error
        if latest_sse_protocol_error is not None:
            raw_blob = (
                latest_sse_protocol_error.partial_arguments_ref.storage_uri
                if latest_sse_protocol_error.partial_arguments_ref is not None
                else "-"
            )
            lines.append(
                "- 最近一次 SSE 协议错误: "
                f"error_type=`{latest_sse_protocol_error.error_type or '-'}`, "
                f"partial_tool_name=`{latest_sse_protocol_error.partial_tool_name or '-'}`, "
                f"attempt={latest_sse_protocol_error.attempt or '-'}, "
                f"request_id=`{latest_sse_protocol_error.request_id or '-'}`"
            )
            lines.append(
                f"- 部分 arguments 前缀: {_md_escape(_extract_sse_partial_arguments_excerpt(latest_sse_protocol_error))}"
            )
            lines.append(f"- SSE raw blob: {_md_escape(raw_blob)}")
        lines.append("")
        if not run.tool_calls:
            lines.append("- 本 run 无成功落盘的 `tool_call` 记录。")
            lines.append("")
            continue
        lines.append("| iteration | idx | tool | status | truncated | latency_ms | result_bytes | 参数摘要 | 结果摘要 |")
        lines.append("|---|---:|---|---|---|---:|---:|---|---|")
        ordered_calls = sorted(run.tool_calls, key=_tool_call_sort_key)
        for call in ordered_calls:
            result_bytes = call.raw_result_ref.bytes if call.raw_result_ref else -1
            lines.append(
                f"| `{call.iteration_id}` | {call.index_in_iteration} | `{call.tool_name}` | `{call.status or '-'}` "
                f"| {'是' if call.truncated else '否'} | {call.latency_ms} | {result_bytes:,} | "
                f"{_md_escape(_args_summary(call.tool_name, call.arguments))} | {_md_escape(_excerpt(call.result_summary))} |"
            )
        lines.append("")
    return lines


def _render_recommendations(bundle: AnalysisBundle) -> list[str]:
    """渲染优化建议。"""

    lines = [
        "## 8. 工具优化建议",
        "",
    ]
    if not bundle.recommendations:
        lines.append("- 当前 trace 未触发明显的结构性优化建议。")
        lines.append("")
        return lines

    for index, item in enumerate(bundle.recommendations, start=1):
        lines.extend(
            [
                f"### 8.{index} {item['priority']} - {item['title']}",
                "",
                f"- **为什么**: {item['reason']}",
                f"- **建议动作**: {item['action']}",
                "",
            ]
        )
    return lines


def _render_manifest_section(manifest_data: dict[str, Any]) -> list[str]:
    """渲染 manifest 章节状态。"""

    lines = [
        "## 11. 章节状态（来自 manifest）",
        "",
        f"- **write_model**: `{manifest_data['write_model']}`",
        f"- **audit_model**: `{manifest_data['audit_model']}`",
        "",
        "| 章节 | 状态 | audit_passed | retry_count | audit_轮数 | 重写耗尽 | 失败原因 |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for item in manifest_data["chapters"]:
        audit_passed = "✓" if item.get("audit_passed") is True else ("✗" if item.get("audit_passed") is False else "-")
        exhausted = "是" if item.get("rewrite_exhausted") else "否"
        failure_reason = item.get("failure_reason") or "-"
        lines.append(
            f"| {_md_escape(item['title'])} | `{item['status']}` | {audit_passed} | {item['retry_count']} "
            f"| {item['audit_rounds']} | {exhausted} | {_md_escape(failure_reason)} |"
        )
    lines.append("")
    return lines


def _render_prompt_pairs(bundle: AnalysisBundle) -> list[str]:
    """渲染提取到的 prompt 对。"""

    lines = [
        "## 9. Prompt",
        "",
    ]
    if not bundle.prompt_pairs:
        lines.append("- 未从 trace 冷存输入中提取到成对的 system prompt / user prompt。")
        lines.append("")
        return lines

    for index, item in enumerate(bundle.prompt_pairs, start=1):
        lines.extend(
            [
                f"### 9.{index} `{item['scene_name']}`",
                "",
                f"- model=`{item['model_name'] or '-'}`, run_id=`{item['run_id']}`, iteration_id=`{item['iteration_id']}`, iteration_index={item['iteration_index']}, system_messages={item['system_message_count']}",
                "",
                "#### System Prompt",
                "",
            ]
        )
        lines.extend(_render_fenced_block(item["system_prompt"], language="text"))
        lines.extend(
            [
                "",
                "#### User Prompt",
                "",
            ]
        )
        lines.extend(_render_fenced_block(item["user_prompt"], language="text"))
        lines.append("")
    return lines


def _render_tool_schemas(bundle: AnalysisBundle) -> list[str]:
    """渲染 trace 中出现过的工具 schema。"""

    lines = [
        "## 10. Schema",
        "",
    ]
    if not bundle.tool_schemas:
        lines.append("- 未从 trace 中提取到工具 schema。")
        lines.append("")
        return lines

    section_index = 1
    for item in bundle.tool_schemas:
        lines.extend(
            [
                f"### 10.{section_index} `{item['tool_name']}`",
                "",
            ]
        )
        section_index += 1
        if not item["variants"]:
            lines.append("- 当前 trace 只记录到了工具名，未拿到对应 raw tool schema。")
            lines.append("")
            continue

        for variant_index, variant in enumerate(item["variants"], start=1):
            seen_preview = ", ".join(
                f"{entry['run_id']}/{entry['iteration_id']}" for entry in variant["seen_in"][:3]
            )
            if len(variant["seen_in"]) > 3:
                seen_preview += f", …+{len(variant['seen_in']) - 3}"
            lines.append(
                f"- 变体 {variant_index}: 出现 {len(variant['seen_in'])} 次；示例 iteration: {seen_preview or '-'}"
            )
            if variant["schema"] is None:
                lines.append("- 该变体未能从 raw payload 读取到完整 schema。")
                continue
            lines.append("")
            lines.extend(
                _render_fenced_block(
                    json.dumps(variant["schema"], ensure_ascii=False, indent=2),
                    language="json",
                )
            )
        lines.append("")
    return lines


def _fmt_set(values: set[str]) -> str:
    """格式化集合。"""

    normalized = sorted(value for value in values if value)
    return ", ".join(f"`{value}`" for value in normalized) if normalized else "（无）"


def _tool_call_sort_key(call: ToolCallInfo) -> tuple[int, int, str]:
    """返回工具调用的稳定排序键，避免 iteration_id 的字符串序污染时序。"""

    return (_extract_iteration_index(call.iteration_id), call.index_in_iteration, call.recorded_at)


def _extract_iteration_index(iteration_id: str) -> int:
    """从 iteration_id 中提取数值 iteration 序号。"""

    match = re.search(r"_iteration_(\d+)$", str(iteration_id or ""))
    if match is None:
        return sys.maxsize
    return _to_int(match.group(1), default=sys.maxsize)


def _excerpt(text: str, *, max_chars: int = MAX_EXCERPT_CHARS) -> str:
    """截断文本摘要。"""

    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


def _md_escape(text: str) -> str:
    """转义 Markdown 表格敏感字符。"""

    return str(text).replace("|", "\\|").replace("\n", " ")


def _render_fenced_block(text: str, *, language: str) -> list[str]:
    """渲染带围栏的代码块，避免内容中的反引号破坏 Markdown。"""

    normalized = str(text or "")
    fence = "~~~~"
    while fence in normalized:
        fence += "~"
    return [f"{fence}{language}", normalized, fence]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="分析 tool_trace_v2，输出面向工具优化的 Markdown 报告。",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./workspace/output/tool_call_traces"),
        help="tool trace JSONL 文件或目录，默认 `./workspace/output/tool_call_traces`",
    )
    parser.add_argument("--manifest", type=Path, help="manifest.json 路径（可选）")
    parser.add_argument("--output", "-o", type=Path, help="输出 Markdown 文件路径（可选）")
    parser.add_argument("--ticker", type=str, help="股票代码（可选，用于标题）")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 入口。"""

    args = parse_args(argv)
    try:
        report = analyze(
            trace_path=args.input,
            manifest_path=args.manifest,
            ticker=args.ticker,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[analyze_tool_trace] {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
