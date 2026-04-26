"""工具调用追踪模块（V2）。

该模块提供在线请求/响应链路的结构化落盘能力，
用于沉淀真实请求样本以优化处理器与提示词策略。

V2 设计约束：
- 主记录采用 `iteration_context_snapshot` + `tool_call` + `iteration_usage` + `final_response`。
- `iteration_context_snapshot` 记录轻量可读摘要与冷存引用，不在热层重复落全量原文。
- 原始输入与工具原始返回落到本地冷存目录，通过 `raw_*_ref` 可精确回放。
- 追踪失败不影响主链路，采用 best-effort 策略。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Optional

from dayu.contracts.agent_types import AgentMessage
from dayu.contracts.protocols import ToolTraceRecorder, ToolTraceRecorderFactory
from dayu.log import Log
from .tool_result import (
    build_error,
    get_error_code,
    get_error_message,
    get_value,
    is_tool_success,
)

MODULE = "ENGINE.TOOL_TRACE"
TRACE_SCHEMA_VERSION = "tool_trace_v2"
TRACE_TYPE_TOOL_CALL = "tool_call"
TRACE_TYPE_ITERATION_CONTEXT_SNAPSHOT = "iteration_context_snapshot"
TRACE_TYPE_FINAL_RESPONSE = "final_response"
TRACE_TYPE_ITERATION_USAGE = "iteration_usage"
TRACE_TYPE_SSE_PROTOCOL_ERROR = "sse_protocol_error"

_MISSING_RESULT_CODE = "RESULT_MISSING"
_MISSING_REQUEST_CODE = "REQUEST_MISSING"
_INVALID_RESULT_CODE = "invalid_result"
_RAW_PAYLOAD_DIR_NAME = "raw_payloads"
_SESSION_DIR_NAME = "sessions"
_SESSION_UNKNOWN = "_session_unknown"
_SESSION_FILE_PREFIX = "tool_calls_"
_SESSION_FILE_PATTERN = re.compile(r"^tool_calls_(\d{6})\.jsonl(?:\.gz)?$")
_DEFAULT_MESSAGE_EXCERPT_LIMIT = 96
_CURRENT_ITERATION_EXCERPT_LIMIT = 200
_RECENT_HISTORY_EXCERPT_LIMIT = 120
_TOOL_CONTEXT_EXCERPT_LIMIT = 120
_SYSTEM_MESSAGE_EXCERPT_LIMIT = 64


def _normalize_session_partition(value: Any) -> str:
    """规范化 session 分区键。

    Args:
        value: 原始 session 标识。

    Returns:
        仅包含安全字符的分区键；为空时返回默认分区。

    Raises:
        无。
    """

    raw = str(value or "").strip()
    if not raw:
        return _SESSION_UNKNOWN
    return re.sub(r"[^0-9A-Za-z._-]", "_", raw)


def _safe_json_dumps(payload: Any) -> str:
    """将对象稳定序列化为 JSON 字符串。

    Args:
        payload: 任意可 JSON 化对象。

    Returns:
        稳定排序后的 JSON 字符串；序列化失败时回退为 ``str(payload)``。

    Raises:
        无。
    """

    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(payload)


def _compute_sha256(payload: Any) -> str:
    """计算载荷的 SHA256 哈希。

    Args:
        payload: 任意对象，将先稳定序列化为字符串。

    Returns:
        ``sha256:<hex>`` 格式字符串。

    Raises:
        无。
    """

    raw = _safe_json_dumps(payload)
    return f"sha256:{hashlib.sha256(raw.encode('utf-8')).hexdigest()}"


def _normalize_result_payload(result: Any) -> dict[str, Any]:
    """规范化工具返回载荷，确保为字典结构。

    Args:
        result: 原始工具返回。

    Returns:
        规范化后的字典结构。

    Raises:
        无。
    """

    if isinstance(result, dict):
        return result
    return build_error(
        _INVALID_RESULT_CODE,
        "tool result must be dict",
        hint=str(result),
    )


def _build_message_source_tag(message: AgentMessage, *, is_last_user: bool) -> str:
    """推断消息来源标签。

    Args:
        message: 单条模型输入消息。
        is_last_user: 当前消息是否为最后一条 user 消息。

    Returns:
        来源标签字符串。

    Raises:
        无。
    """

    role = str(message.get("role") or "")
    name = str(message.get("name") or "")
    content = str(message.get("content") or "")

    if role == "tool":
        return "tool_context"
    if role == "system":
        if name == "summary" or "[Context Compaction Summary]" in content:
            return "summary"
        if name == "memory":
            return "memory"
        return "policy"
    if role == "user" and is_last_user:
        return "current_iteration"
    if role in {"user", "assistant"}:
        return "recent_history"
    return "unknown"


def _extract_message_text(message: AgentMessage) -> str:
    """提取消息可读文本并做稳定序列化。

    Args:
        message: 单条消息。

    Returns:
        文本内容字符串。

    Raises:
        无。
    """

    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return _safe_json_dumps(content)


def _get_message_excerpt_limit(source_tag: str) -> int:
    """按消息来源标签返回 excerpt 截断长度。

    Args:
        source_tag: 消息来源标签。

    Returns:
        对应来源标签的 excerpt 最大长度。

    Raises:
        无。
    """

    if source_tag == "current_iteration":
        return _CURRENT_ITERATION_EXCERPT_LIMIT
    if source_tag == "recent_history":
        return _RECENT_HISTORY_EXCERPT_LIMIT
    if source_tag == "tool_context":
        return _TOOL_CONTEXT_EXCERPT_LIMIT
    if source_tag in {"policy", "memory", "summary"}:
        return _SYSTEM_MESSAGE_EXCERPT_LIMIT
    return _DEFAULT_MESSAGE_EXCERPT_LIMIT


def _build_messages_summary(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """构建模型输入消息的结构化摘要。

    Args:
        messages: 本轮送模的完整消息列表。

    Returns:
        每条消息的摘要列表，包含角色、来源、摘要片段与哈希。

    Raises:
        无。
    """

    last_user_index = -1
    for idx, message in enumerate(messages):
        if str(message.get("role") or "") == "user":
            last_user_index = idx

    summaries: list[dict[str, Any]] = []
    for idx, message in enumerate(messages):
        text = _extract_message_text(message)
        source_tag = _build_message_source_tag(message, is_last_user=(idx == last_user_index))
        excerpt_limit = _get_message_excerpt_limit(source_tag)
        summaries.append(
            {
                "idx": idx,
                "role": str(message.get("role") or ""),
                "source_tag": source_tag,
                "excerpt": text[:excerpt_limit],
                "content_hash": _compute_sha256(text),
            }
        )
    return summaries


def _build_result_fact(
    *,
    result: dict[str, Any],
    raw_result_ref: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """构建工具结果事实摘要。

    Args:
        result: 规范化后的工具结果。
        raw_result_ref: 原始返回冷存引用。

    Returns:
        结构化事实字典。

    Raises:
        无。
    """

    meta_value = result.get("meta")
    meta: dict[str, Any] = meta_value if isinstance(meta_value, dict) else {}
    success = is_tool_success(result)
    status = "success" if success else "error"
    return {
        "status": status,
        "error_code": get_error_code(result),
        "truncated": result.get("truncation") is not None,
        "latency_ms": meta.get("latency_ms"),
        "result_hash": _compute_sha256(result),
        "raw_result_ref": raw_result_ref,
    }


def _build_result_summary(result: dict[str, Any]) -> str:
    """生成工具结果简短摘要。

    Args:
        result: 规范化后的工具结果。

    Returns:
        可读摘要字符串。

    Raises:
        无。
    """

    success = is_tool_success(result)
    if success:
        value = get_value(result)
        value_type = type(value).__name__ if value is not None else "null"
        truncated = result.get("truncation") is not None
        return f"success(type={value_type}, truncated={truncated})"
    code = get_error_code(result) or "UNKNOWN"
    message = get_error_message(result) or ""
    return f"error(code={code}, message={message[:80]})"


def _normalize_partial_tool_name(partial_tool_name: str | None) -> str | None:
    """规范化部分 tool call 的工具名。

    Args:
        partial_tool_name: 原始工具名。

    Returns:
        去除空白后的工具名；为空时返回 `None`。

    Raises:
        无。
    """

    if partial_tool_name is None:
        return None
    normalized_tool_name = partial_tool_name.strip()
    if not normalized_tool_name:
        return None
    return normalized_tool_name


def _extract_result_data(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """提取工具返回中的业务数据。

    Args:
        result: 规范化后的工具结果。

    Returns:
        工具返回值为字典时返回该字典，否则返回 ``None``。

    Raises:
        无。
    """

    value = get_value(result)
    if isinstance(value, dict):
        return value
    return None


def _extract_current_user_message(messages: list[AgentMessage]) -> str:
    """提取当前 iteration 最后一条用户输入文本。

    Args:
        messages: 本轮送模消息。

    Returns:
        最后一条 user 文本，未找到时返回空字符串。

    Raises:
        无。
    """

    for message in reversed(messages):
        if str(message.get("role") or "") != "user":
            continue
        return _extract_message_text(message)
    return ""


def _build_context_meta(messages: list[AgentMessage]) -> dict[str, Any]:
    """从消息列表推断上下文元信息。

    Args:
        messages: 本轮送模消息。

    Returns:
        上下文元信息字典。

    Raises:
        无。
    """

    summary_present = False
    summary_version = 0
    memory_keys: list[str] = []
    tool_context_count = 0
    recent_history_count = 0

    for message in messages:
        role = str(message.get("role") or "")
        name = str(message.get("name") or "")
        content = _extract_message_text(message)

        if role == "tool":
            tool_context_count += 1
            continue

        if role == "system":
            if name == "summary" or "[Context Compaction Summary]" in content:
                summary_present = True
                if summary_version == 0:
                    summary_version = 1
            if name == "memory":
                memory_keys.append("memory")
            continue

        if role in {"user", "assistant"}:
            recent_history_count += 1

    return {
        "summary_present": summary_present,
        "summary_version": summary_version,
        "recent_history_count": max(0, recent_history_count - 1),
        "memory_keys": sorted(set(memory_keys)),
        "tool_context_count": tool_context_count,
    }


def _extract_tool_schema_names(tool_schemas: list[dict[str, Any]]) -> list[str]:
    """从工具 schema 列表中提取工具名。

    Args:
        tool_schemas: 原始工具 schema 列表。

    Returns:
        已过滤空值后的工具名列表。

    Raises:
        无。
    """

    tool_schema_names: list[str] = []
    for schema in tool_schemas:
        if not isinstance(schema, dict):
            continue
        function_block = schema.get("function")
        if not isinstance(function_block, dict):
            continue
        tool_name = str(function_block.get("name") or "").strip()
        if tool_name:
            tool_schema_names.append(tool_name)
    return tool_schema_names


def _build_iteration_tool_call_summary(
    *,
    tool_call_id: str,
    tool_name: str,
    result: dict[str, Any],
    raw_result_ref: Optional[dict[str, Any]],
    result_summary: str,
) -> dict[str, Any]:
    """构建 iteration_context_snapshot 中的单条工具调用摘要。

    Args:
        tool_call_id: 工具调用 ID。
        tool_name: 工具名。
        result: 规范化后的工具结果。
        raw_result_ref: 工具原始返回冷存引用。
        result_summary: 工具结果摘要。

    Returns:
        轻量工具调用摘要字典。

    Raises:
        无。
    """

    return {
        "call_id": tool_call_id,
        "tool_name": tool_name,
        "status": "success" if is_tool_success(result) else "error",
        "result_fact": _build_result_fact(result=result, raw_result_ref=raw_result_ref),
        "result_summary": result_summary,
    }


def _build_tool_call_record(
    *,
    request: "_ToolCallRequestState",
    result: dict[str, Any],
    raw_result_ref: Optional[dict[str, Any]],
    result_summary: Optional[str],
    session_id: str,
    agent_metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """构建单条 `tool_call` 落盘记录。

    Args:
        request: 请求侧状态。
        result: 规范化后的工具结果。
        raw_result_ref: 原始结果冷存引用。
        result_summary: 工具结果摘要。
        session_id: 会话 ID。
        agent_metadata: Agent 固定元数据。

    Returns:
        统一结构化记录。

    Raises:
        无。
    """

    fact = _build_result_fact(result=result, raw_result_ref=raw_result_ref)
    summary = result_summary or _build_result_summary(result)
    record = {
        "trace_schema_version": TRACE_SCHEMA_VERSION,
        "trace_type": TRACE_TYPE_TOOL_CALL,
        "recorded_at": datetime.now(UTC).isoformat(),
        "run_id": request.run_id,
        "session_id": session_id,
        "iteration_id": request.iteration_id,
        "tool_call_id": request.tool_call_id,
        "index_in_iteration": request.index_in_iteration,
        "tool_name": request.tool_name,
        "arguments": request.arguments,
        "result_fact": fact,
        "result_summary": summary,
        "result_data": _extract_result_data(result),
    }
    if agent_metadata:
        record.update(agent_metadata)
    return record


@dataclass(slots=True)
class _ToolCallRequestState:
    """单条工具请求的运行时状态。"""

    run_id: str
    iteration_id: str
    tool_call_id: str
    index_in_iteration: int
    tool_name: str
    arguments: Any


@dataclass(slots=True)
class _ToolCallResultState:
    """单条工具返回的运行时状态。"""

    iteration_id: str
    tool_call_id: str
    index_in_iteration: int
    tool_name: str
    arguments: Any
    result: dict[str, Any]
    raw_result_ref: Optional[dict[str, Any]]
    result_summary: str


@dataclass(slots=True)
class _IterationTraceState:
    """单次 agent iteration 的 trace 聚合状态。"""

    model_input_messages: list[AgentMessage]
    tool_schema_names: list[str] = field(default_factory=list)
    raw_tool_schemas_ref: Optional[dict[str, Any]] = None
    tool_calls_by_id: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_call_order: dict[str, int] = field(default_factory=dict)


class JsonlToolTraceStore:
    """基于 JSONL 的 trace 落盘存储。

    该对象只负责两件事：
    - 追加写入结构化 trace 记录。
    - 写入 raw payload 冷存并返回引用。
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        max_file_bytes: int = 64 * 1024 * 1024,
        retention_days: int = 30,
        compress_rolled: bool = True,
        partition_by_session: bool = True,
    ) -> None:
        """初始化 JSONL trace 存储。

        Args:
            output_dir: JSONL 文件输出目录。
            max_file_bytes: 单文件最大字节数，超过后滚动。
            retention_days: 文件保留天数。
            compress_rolled: 滚动后是否压缩旧分片。
            partition_by_session: 是否按 session 分区。

        Returns:
            无。

        Raises:
            OSError: 目录创建失败时抛出。
        """

        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._sessions_root = self._output_dir / _SESSION_DIR_NAME
        self._sessions_root.mkdir(parents=True, exist_ok=True)
        self._raw_dir = self._output_dir / _RAW_PAYLOAD_DIR_NAME
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._max_file_bytes = max(1024, int(max_file_bytes))
        self._retention_days = max(1, int(retention_days))
        self._compress_rolled = bool(compress_rolled)
        self._partition_by_session = bool(partition_by_session)
        self._lock = Lock()
        self._active_trace_files: dict[str, Path] = {}
        self._active_trace_indices: dict[str, int] = {}
        self._last_cleanup_date = ""
        with self._lock:
            self._cleanup_expired_files_locked(force=True)

    def get_current_trace_file_path(self) -> Path:
        """获取默认分区当前 trace 输出文件路径。

        Args:
            无。

        Returns:
            当前本地日期分片文件路径。

        Raises:
            无。
        """

        with self._lock:
            return self._ensure_active_file_locked(_SESSION_UNKNOWN)

    def store_raw_payload(
        self,
        *,
        run_id: str,
        iteration_id: str,
        payload_type: str,
        payload: Any,
    ) -> dict[str, Any]:
        """写入冷存原始载荷并返回引用。

        Args:
            run_id: 运行 ID。
            iteration_id: iteration ID。
            payload_type: 载荷类型。
            payload: 待写入原始载荷。

        Returns:
            原始载荷引用字典，包含 blob_id/content_hash/storage_uri/bytes。

        Raises:
            无。
        """

        resolved_iteration_id = str(iteration_id or "").strip()
        if not resolved_iteration_id:
            raise ValueError("iteration_id 不能为空")

        raw_text = _safe_json_dumps(payload)
        content_hash = _compute_sha256(raw_text)
        hash_hex = content_hash.split(":", 1)[1]
        blob_id = f"{payload_type}:{run_id}:{resolved_iteration_id}:{hash_hex[:12]}"
        run_dir = self._raw_dir / run_id / resolved_iteration_id
        run_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"{payload_type}_{hash_hex}.json"
        file_path = run_dir / file_name
        file_path.write_text(raw_text, encoding="utf-8")
        return {
            "blob_id": blob_id,
            "content_hash": content_hash,
            "storage_uri": str(file_path),
            "bytes": len(raw_text.encode("utf-8")),
        }

    def append_record(self, record: dict[str, Any]) -> None:
        """追加写入一条结构化 trace 记录。

        Args:
            record: 待写入记录。

        Returns:
            无。

        Raises:
            OSError: 文件写入失败时抛出。
            TypeError: JSON 序列化失败时抛出。
            ValueError: JSON 序列化失败时抛出。
        """

        with self._lock:
            self._append_record_locked(record)

    def _append_record_locked(self, record: dict[str, Any]) -> None:
        """在锁内写入一条 JSONL 记录。

        Args:
            record: 待写入记录。

        Returns:
            无。

        Raises:
            OSError: 文件写入失败时抛出。
            TypeError: JSON 序列化失败时抛出。
            ValueError: JSON 序列化失败时抛出。
        """

        self._cleanup_expired_files_locked(force=False)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = line.encode("utf-8")
        session_key = self._resolve_session_key(record)
        file_path = self._ensure_active_file_locked(session_key)
        if file_path.exists() and (file_path.stat().st_size + len(line_bytes) > self._max_file_bytes):
            file_path = self._rollover_file_locked(session_key)
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    def _now_local(self) -> datetime:
        """获取当前本地时区时间。

        Args:
            无。

        Returns:
            带本地时区信息的当前时间。

        Raises:
            无。
        """

        return datetime.now().astimezone()

    def _resolve_session_key(self, record: dict[str, Any]) -> str:
        """解析记录对应的 session 分区键。

        Args:
            record: 单条 trace 记录。

        Returns:
            分区键字符串。

        Raises:
            无。
        """

        if not self._partition_by_session:
            return _SESSION_UNKNOWN
        session_id = record.get("session_id") or record.get("run_id") or _SESSION_UNKNOWN
        return _normalize_session_partition(session_id)

    def _ensure_active_file_locked(self, session_key: str) -> Path:
        """确保指定 session 的活动分片存在。

        Args:
            session_key: session 分区键。

        Returns:
            当前活动 JSONL 文件路径。

        Raises:
            无。
        """

        current = self._active_trace_files.get(session_key)
        if current is not None:
            return current
        session_dir = self._sessions_root / session_key
        session_dir.mkdir(parents=True, exist_ok=True)
        next_index = self._compute_next_index_locked(session_dir)
        file_path = session_dir / f"{_SESSION_FILE_PREFIX}{next_index:06d}.jsonl"
        self._active_trace_files[session_key] = file_path
        self._active_trace_indices[session_key] = next_index
        return file_path

    def _compute_next_index_locked(self, session_dir: Path) -> int:
        """计算 session 目录下的下一个分片序号。

        Args:
            session_dir: session 分区目录。

        Returns:
            下一个分片序号。

        Raises:
            无。
        """

        max_index = 0
        for path in session_dir.glob(f"{_SESSION_FILE_PREFIX}*.jsonl*"):
            matched = _SESSION_FILE_PATTERN.match(path.name)
            if matched is None:
                continue
            try:
                index = int(matched.group(1))
            except ValueError:
                continue
            max_index = max(max_index, index)
        return max_index + 1

    def _rollover_file_locked(self, session_key: str) -> Path:
        """滚动指定 session 的 trace 分片。

        Args:
            session_key: session 分区键。

        Returns:
            新的活动分片路径。

        Raises:
            无。
        """

        current = self._ensure_active_file_locked(session_key)
        if self._compress_rolled and current.exists() and current.stat().st_size > 0:
            gz_path = current.with_suffix(".jsonl.gz")
            with current.open("rb") as src, gzip.open(gz_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            current.unlink(missing_ok=True)
        current_index = self._active_trace_indices.get(session_key, 0)
        next_index = current_index + 1
        next_path = current.parent / f"{_SESSION_FILE_PREFIX}{next_index:06d}.jsonl"
        self._active_trace_files[session_key] = next_path
        self._active_trace_indices[session_key] = next_index
        return next_path

    def _cleanup_expired_files_locked(self, *, force: bool) -> None:
        """按保留期清理过期 trace 与 raw payload。

        Args:
            force: 是否强制执行本次清理。

        Returns:
            无。

        Raises:
            无。
        """

        now = self._now_local()
        day_key = now.strftime("%Y%m%d")
        if not force and self._last_cleanup_date == day_key:
            return
        self._last_cleanup_date = day_key
        cutoff = now.timestamp() - float(self._retention_days * 24 * 3600)
        for path in self._sessions_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix not in {".jsonl", ".gz"}:
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
        for run_dir in self._raw_dir.iterdir():
            if not run_dir.is_dir():
                continue
            try:
                if run_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(run_dir, ignore_errors=True)
            except OSError:
                continue


class V2ToolTraceRecorder:
    """`tool_trace_v2` 的 per-run recorder 实现。"""

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        store: JsonlToolTraceStore,
        agent_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """初始化单次 run 的 trace recorder。

        Args:
            run_id: 本次运行 ID。
            session_id: 当前会话 ID。
            store: JSONL trace 存储。
            agent_metadata: Agent 固定身份元数据。

        Returns:
            无。

        Raises:
            无。
        """

        self._run_id = run_id
        self._session_id = session_id or run_id
        self._store = store
        self._agent_metadata = dict(agent_metadata or {})
        self._iterations: dict[str, _IterationTraceState] = {}
        self._pending_requests: dict[tuple[str, str], _ToolCallRequestState] = {}
        self._pending_results: dict[tuple[str, str], _ToolCallResultState] = {}
        self._raw_paths: set[Path] = set()

    def start_iteration(
        self,
        *,
        iteration_id: str,
        model_input_messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
    ) -> None:
        """开始记录一次 agent iteration 的送模上下文。

        Args:
            iteration_id: iteration ID。
            model_input_messages: 本次 iteration 的真实送模消息。
            tool_schemas: 当前实际注册给模型的原始工具 schema 列表。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            raw_tool_schemas_ref: Optional[dict[str, Any]] = None
            tool_schema_names = _extract_tool_schema_names(tool_schemas)
            if tool_schemas:
                raw_tool_schemas_ref = self._store.store_raw_payload(
                    run_id=self._run_id,
                    iteration_id=iteration_id,
                    payload_type="tool_schemas_final",
                    payload=tool_schemas,
                )
                self._remember_raw_ref(raw_tool_schemas_ref)
            self._iterations[iteration_id] = _IterationTraceState(
                model_input_messages=list(model_input_messages),
                tool_schema_names=tool_schema_names,
                raw_tool_schemas_ref=raw_tool_schemas_ref,
            )
        except Exception as exc:
            Log.warn(
                f"tool trace start_iteration 失败: run_id={self._run_id}, iteration_id={iteration_id}, error={exc}",
                module=MODULE,
            )

    def on_tool_dispatched(self, *, iteration_id: str, payload: Any) -> None:
        """观察到工具请求发起事件。

        Args:
            iteration_id: iteration ID。
            payload: `tool_call_dispatched` 事件数据。

        Returns:
            无。

        Raises:
            无。
        """

        if not isinstance(payload, dict):
            return
        tool_call_id = str(payload.get("id") or "")
        if not tool_call_id:
            return
        try:
            request = _ToolCallRequestState(
                run_id=self._run_id,
                iteration_id=iteration_id,
                tool_call_id=tool_call_id,
                index_in_iteration=int(payload.get("index_in_iteration") or 0),
                tool_name=str(payload.get("name") or ""),
                arguments=payload.get("arguments"),
            )
            key = (iteration_id, tool_call_id)
            result_state = self._pending_results.pop(key, None)
            if result_state is None:
                self._pending_requests[key] = request
                return
            self._append_tool_call_record(
                request=request,
                result=result_state.result,
                raw_result_ref=result_state.raw_result_ref,
                result_summary=result_state.result_summary,
            )
        except Exception as exc:
            Log.warn(
                f"tool trace 记录请求失败: run_id={self._run_id}, iteration_id={iteration_id}, tool_call_id={tool_call_id}, error={exc}",
                module=MODULE,
            )

    def on_tool_result(self, *, iteration_id: str, payload: Any) -> None:
        """观察到工具返回事件。

        Args:
            iteration_id: iteration ID。
            payload: `tool_call_result` 事件数据。

        Returns:
            无。

        Raises:
            无。
        """

        if not isinstance(payload, dict):
            return
        tool_call_id = str(payload.get("id") or "")
        if not tool_call_id:
            return
        try:
            normalized_result = _normalize_result_payload(payload.get("result"))
            raw_result_ref = self._store.store_raw_payload(
                run_id=self._run_id,
                iteration_id=iteration_id,
                payload_type="tool_result_raw",
                payload=payload.get("result"),
            )
            self._remember_raw_ref(raw_result_ref)
            result_summary = _build_result_summary(normalized_result)
            self._update_iteration_tool_summary(
                iteration_id=iteration_id,
                tool_call_id=tool_call_id,
                tool_name=str(payload.get("name") or ""),
                index_in_iteration=int(payload.get("index_in_iteration") or 0),
                result=normalized_result,
                raw_result_ref=raw_result_ref,
                result_summary=result_summary,
            )
            key = (iteration_id, tool_call_id)
            request = self._pending_requests.pop(key, None)
            if request is None:
                self._pending_results[key] = _ToolCallResultState(
                    iteration_id=iteration_id,
                    tool_call_id=tool_call_id,
                    index_in_iteration=int(payload.get("index_in_iteration") or 0),
                    tool_name=str(payload.get("name") or ""),
                    arguments=payload.get("arguments"),
                    result=normalized_result,
                    raw_result_ref=raw_result_ref,
                    result_summary=result_summary,
                )
                return
            self._append_tool_call_record(
                request=request,
                result=normalized_result,
                raw_result_ref=raw_result_ref,
                result_summary=result_summary,
            )
        except Exception as exc:
            Log.warn(
                f"tool trace 记录返回失败: run_id={self._run_id}, iteration_id={iteration_id}, tool_call_id={tool_call_id}, error={exc}",
                module=MODULE,
            )

    def record_iteration_usage(
        self,
        *,
        iteration_id: str,
        usage: dict[str, Any],
        budget_snapshot: Optional[dict[str, Any]] = None,
    ) -> None:
        """记录单次 iteration token 用量。

        Args:
            iteration_id: iteration ID。
            usage: token 用量。
            budget_snapshot: 可选预算快照。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            record: dict[str, Any] = {
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "trace_type": TRACE_TYPE_ITERATION_USAGE,
                "recorded_at": datetime.now(UTC).isoformat(),
                "run_id": self._run_id,
                "session_id": self._session_id,
                "iteration_id": iteration_id,
                "usage": usage,
            }
            if budget_snapshot:
                record["budget_snapshot"] = budget_snapshot
            if self._agent_metadata:
                record.update(self._agent_metadata)
            self._store.append_record(record)
        except Exception as exc:
            Log.warn(
                f"tool trace 记录 iteration_usage 失败: run_id={self._run_id}, iteration_id={iteration_id}, error={exc}",
                module=MODULE,
            )

    def record_final_response(
        self,
        *,
        iteration_id: str,
        content: str,
        degraded: bool,
        filtered: bool = False,
        finish_reason: str | None = None,
    ) -> None:
        """记录最终回答。

        Args:
            iteration_id: iteration ID。
            content: 最终回答文本。
            degraded: 是否降级回答。
            filtered: 是否被内容过滤命中。
            finish_reason: 上游 runner 报告的 finish_reason（如 length / stop / content_filter）。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            record = {
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "trace_type": TRACE_TYPE_FINAL_RESPONSE,
                "recorded_at": datetime.now(UTC).isoformat(),
                "run_id": self._run_id,
                "session_id": self._session_id,
                "iteration_id": iteration_id,
                "final_response": {
                    "content": content,
                    "degraded": degraded,
                    "filtered": filtered,
                    "finish_reason": finish_reason,
                },
            }
            if self._agent_metadata:
                record.update(self._agent_metadata)
            self._store.append_record(record)
        except Exception as exc:
            Log.warn(
                f"tool trace 记录 final_response 失败: run_id={self._run_id}, iteration_id={iteration_id}, error={exc}",
                module=MODULE,
            )

    def record_sse_protocol_error(
        self,
        *,
        iteration_id: str,
        error_type: str,
        partial_tool_name: str | None,
        partial_tool_calls: list[dict[str, Any]],
        request_id: str,
        attempt: int | None = None,
    ) -> None:
        """记录 SSE 协议错误现场。

        Args:
            iteration_id: iteration ID。
            error_type: 协议错误类型。
            partial_tool_name: 首个可识别的部分工具名。
            partial_tool_calls: 截止失败点累计的部分 tool call 片段。
            request_id: 当前请求 ID。
            attempt: 当前请求尝试次数。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            partial_arguments_ref = self._store.store_raw_payload(
                run_id=self._run_id,
                iteration_id=iteration_id,
                payload_type="sse_error",
                payload={
                    "error_type": error_type,
                    "request_id": request_id,
                    "attempt": attempt,
                    "tool_calls": partial_tool_calls,
                },
            )
            self._remember_raw_ref(partial_arguments_ref)
            record: dict[str, Any] = {
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "trace_type": TRACE_TYPE_SSE_PROTOCOL_ERROR,
                "recorded_at": datetime.now(UTC).isoformat(),
                "run_id": self._run_id,
                "session_id": self._session_id,
                "iteration_id": iteration_id,
                "error_type": error_type,
                "partial_tool_name": _normalize_partial_tool_name(partial_tool_name),
                "partial_arguments_ref": partial_arguments_ref,
                "request_id": request_id,
                "attempt": attempt,
            }
            if self._agent_metadata:
                record.update(self._agent_metadata)
            self._store.append_record(record)
        except Exception as exc:
            Log.warn(
                f"tool trace 记录 sse_protocol_error 失败: run_id={self._run_id}, iteration_id={iteration_id}, error={exc}",
                module=MODULE,
            )

    def finish_iteration(
        self,
        *,
        iteration_id: str,
        iteration_index: int,
        termination_reason: str | None = None,
    ) -> None:
        """结束一次 iteration 并输出上下文快照。

        Args:
            iteration_id: iteration ID。
            iteration_index: iteration 序号。
            termination_reason: iteration 结束原因（如 final_answer / tool_calls / error / cancelled / max_iterations / overflow_exhausted）。
                可选，缺省 None 兼容历史调用。

        Returns:
            无。

        Raises:
            无。
        """

        iteration_state = self._iterations.get(iteration_id)
        if iteration_state is None:
            return
        try:
            raw_input_ref = self._store.store_raw_payload(
                run_id=self._run_id,
                iteration_id=iteration_id,
                payload_type="model_input_messages_final",
                payload=iteration_state.model_input_messages,
            )
            self._remember_raw_ref(raw_input_ref)
            tool_calls = [
                payload
                for _, payload in sorted(iteration_state.tool_calls_by_id.items(), key=self._sort_iteration_tool_call)
            ]
            record = {
                "trace_schema_version": TRACE_SCHEMA_VERSION,
                "trace_type": TRACE_TYPE_ITERATION_CONTEXT_SNAPSHOT,
                "recorded_at": datetime.now(UTC).isoformat(),
                "run_id": self._run_id,
                "session_id": self._session_id,
                "iteration_id": iteration_id,
                "iteration_index": int(iteration_index),
                "current_user_message": _extract_current_user_message(iteration_state.model_input_messages),
                "context_meta": _build_context_meta(iteration_state.model_input_messages),
                "model_input_messages_summary": _build_messages_summary(iteration_state.model_input_messages),
                "raw_input_ref": raw_input_ref,
                "tool_schema_names": list(iteration_state.tool_schema_names),
                "raw_tool_schemas_ref": iteration_state.raw_tool_schemas_ref,
                "tool_calls": tool_calls,
                "termination_reason": termination_reason,
            }
            if self._agent_metadata:
                record.update(self._agent_metadata)
            self._store.append_record(record)
        except Exception as exc:
            Log.warn(
                f"tool trace 记录 iteration_context_snapshot 失败: run_id={self._run_id}, iteration_id={iteration_id}, error={exc}",
                module=MODULE,
            )

    def close(self) -> None:
        """关闭 recorder，并补偿未配对记录。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            for key, request in list(self._pending_requests.items()):
                self._append_tool_call_record(
                    request=request,
                    result=build_error(
                        _MISSING_RESULT_CODE,
                        "tool result not received before run closed",
                    ),
                    raw_result_ref=None,
                    result_summary="result missing",
                )
                self._pending_requests.pop(key, None)

            for key, result_state in list(self._pending_results.items()):
                request = _ToolCallRequestState(
                    run_id=self._run_id,
                    iteration_id=result_state.iteration_id,
                    tool_call_id=result_state.tool_call_id,
                    index_in_iteration=result_state.index_in_iteration,
                    tool_name=result_state.tool_name,
                    arguments=result_state.arguments,
                )
                self._append_tool_call_record(
                    request=request,
                    result=build_error(
                        _MISSING_REQUEST_CODE,
                        "tool request not received before tool result",
                    ),
                    raw_result_ref=result_state.raw_result_ref,
                    result_summary="request missing",
                )
                self._pending_results.pop(key, None)

            for raw_path in sorted(self._raw_paths):
                if not raw_path.exists():
                    Log.warn(
                        f"tool trace raw payload 丢失: run_id={self._run_id}, path={raw_path}",
                        module=MODULE,
                    )
        except Exception as exc:
            Log.warn(f"tool trace close 失败: run_id={self._run_id}, error={exc}", module=MODULE)

    def _append_tool_call_record(
        self,
        *,
        request: _ToolCallRequestState,
        result: dict[str, Any],
        raw_result_ref: Optional[dict[str, Any]],
        result_summary: Optional[str],
    ) -> None:
        """输出单条 `tool_call` 记录。

        Args:
            request: 请求侧状态。
            result: 工具结果。
            raw_result_ref: 原始结果冷存引用。
            result_summary: 结果摘要。

        Returns:
            无。

        Raises:
            无。
        """

        record = _build_tool_call_record(
            request=request,
            result=_normalize_result_payload(result),
            raw_result_ref=raw_result_ref,
            result_summary=result_summary,
            session_id=self._session_id,
            agent_metadata=self._agent_metadata,
        )
        self._store.append_record(record)

    def _update_iteration_tool_summary(
        self,
        *,
        iteration_id: str,
        tool_call_id: str,
        tool_name: str,
        index_in_iteration: int,
        result: dict[str, Any],
        raw_result_ref: Optional[dict[str, Any]],
        result_summary: str,
    ) -> None:
        """更新单次 iteration 上下文中的工具摘要。

        Args:
            iteration_id: iteration ID。
            tool_call_id: 工具调用 ID。
            tool_name: 工具名。
            index_in_iteration: iteration 内顺序。
            result: 规范化后的工具结果。
            raw_result_ref: 原始结果冷存引用。
            result_summary: 结果摘要。

        Returns:
            无。

        Raises:
            无。
        """

        iteration_state = self._iterations.get(iteration_id)
        if iteration_state is None:
            iteration_state = _IterationTraceState(model_input_messages=[])
            self._iterations[iteration_id] = iteration_state
        iteration_state.tool_call_order[tool_call_id] = int(index_in_iteration)
        iteration_state.tool_calls_by_id[tool_call_id] = _build_iteration_tool_call_summary(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=result,
            raw_result_ref=raw_result_ref,
            result_summary=result_summary,
        )

    def _sort_iteration_tool_call(self, item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
        """生成单次 iteration 工具摘要排序键。

        Args:
            item: ``(tool_call_id, summary_payload)`` 元组。

        Returns:
            ``(index_in_iteration, tool_call_id)`` 排序键。

        Raises:
            无。
        """

        tool_call_id, _ = item
        return (self._iterations_order_value(tool_call_id), tool_call_id)

    def _iterations_order_value(self, tool_call_id: str) -> int:
        """查找工具调用在单次 iteration 内的顺序值。

        Args:
            tool_call_id: 工具调用 ID。

        Returns:
            顺序值；未命中时返回较大默认值。

        Raises:
            无。
        """

        for iteration_state in self._iterations.values():
            if tool_call_id in iteration_state.tool_call_order:
                return iteration_state.tool_call_order[tool_call_id]
        return 1 << 30

    def _remember_raw_ref(self, raw_ref: Optional[dict[str, Any]]) -> None:
        """登记 raw payload 路径，便于 close 时校验。

        Args:
            raw_ref: raw payload 引用。

        Returns:
            无。

        Raises:
            无。
        """

        if not isinstance(raw_ref, dict):
            return
        storage_uri = raw_ref.get("storage_uri")
        if not storage_uri:
            return
        self._raw_paths.add(Path(str(storage_uri)))


class JsonlToolTraceRecorderFactory:
    """基于 `JsonlToolTraceStore` 的 recorder 工厂。"""

    def __init__(self, store: JsonlToolTraceStore) -> None:
        """初始化 recorder 工厂。

        Args:
            store: 共享 JSONL trace 存储。

        Returns:
            无。

        Raises:
            无。
        """

        self._store = store

    def create_recorder(
        self,
        *,
        run_id: str,
        session_id: str,
        agent_metadata: Optional[dict[str, Any]] = None,
    ) -> ToolTraceRecorder:
        """创建单次 run 使用的 recorder。

        Args:
            run_id: 本次运行 ID。
            session_id: 当前会话 ID。
            agent_metadata: Agent 固定身份元数据。

        Returns:
            新的 per-run recorder。

        Raises:
            无。
        """

        return V2ToolTraceRecorder(
            run_id=run_id,
            session_id=session_id,
            store=self._store,
            agent_metadata=agent_metadata,
        )


__all__ = [
    "TRACE_SCHEMA_VERSION",
    "TRACE_TYPE_TOOL_CALL",
    "TRACE_TYPE_ITERATION_CONTEXT_SNAPSHOT",
    "TRACE_TYPE_FINAL_RESPONSE",
    "TRACE_TYPE_ITERATION_USAGE",
    "TRACE_TYPE_SSE_PROTOCOL_ERROR",
    "ToolTraceRecorder",
    "ToolTraceRecorderFactory",
    "JsonlToolTraceStore",
    "JsonlToolTraceRecorderFactory",
    "V2ToolTraceRecorder",
]
