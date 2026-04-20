"""跨层共享的执行选项契约。

该模块只承载会在 ``Service / Contracts / Host`` 之间流动的稳定执行选项数据
结构，不包含运行时合并算法。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Mapping, TypeAlias, cast

from dayu.contracts.tool_configs import DocToolLimits, FinsToolLimits, WebToolsConfig, build_legacy_toolset_configs
from dayu.contracts.toolset_config import (
    ToolsetConfigSnapshot,
    ToolsetConfigValue,
    build_toolset_config_snapshot,
    normalize_toolset_configs,
    replace_toolset_config,
    serialize_toolset_config_payload_value,
)

ExecutionOptionsSnapshotValue: TypeAlias = (
    str
    | int
    | float
    | bool
    | None
    | list["ExecutionOptionsSnapshotValue"]
    | dict[str, "ExecutionOptionsSnapshotValue"]
)
ExecutionOptionsSnapshot: TypeAlias = dict[str, ExecutionOptionsSnapshotValue]

_REMOVED_EXECUTION_OPTION_SNAPSHOT_FIELDS = frozenset({"max_consecutive_tool_failures"})


@dataclass(frozen=True)
class TraceSettings:
    """工具调用追踪配置。

    Args:
        enabled: 是否开启工具调用追踪。
        output_dir: 追踪输出目录。
        max_file_bytes: 单个 trace 文件大小上限。
        retention_days: trace 文件保留天数。
        compress_rolled: 是否压缩轮转后的 trace 文件。
        partition_by_session: 是否按 session 分区写 trace。

    Returns:
        无。

    Raises:
        无。
    """

    enabled: bool
    output_dir: Path
    max_file_bytes: int = 64 * 1024 * 1024
    retention_days: int = 30
    compress_rolled: bool = True
    partition_by_session: bool = True


@dataclass(frozen=True)
class ConversationMemorySettings:
    """多轮会话分层记忆配置。

    Args:
        working_memory_max_turns: 工作记忆保留的最大轮数。
        working_memory_token_budget_ratio: 工作记忆预算比例。
        working_memory_token_budget_floor: 工作记忆预算下限。
        working_memory_token_budget_cap: 工作记忆预算上限。
        episodic_memory_token_budget_ratio: 情节记忆预算比例。
        episodic_memory_token_budget_floor: 情节记忆预算下限。
        episodic_memory_token_budget_cap: 情节记忆预算上限。
        compaction_trigger_turn_count: 触发压缩的轮数阈值。
        compaction_trigger_token_ratio: 触发压缩的 token 比例阈值。
        compaction_tail_preserve_turns: 压缩后保留的尾部轮数。
        compaction_context_episode_window: 压缩时携带的 episode 窗口数。
        compaction_scene_name: 执行压缩使用的 scene 名称。

    Returns:
        无。

    Raises:
        无。
    """

    working_memory_max_turns: int = 6
    working_memory_token_budget_ratio: float = 0.08
    working_memory_token_budget_floor: int = 1500
    working_memory_token_budget_cap: int = 6000
    episodic_memory_token_budget_ratio: float = 0.02
    episodic_memory_token_budget_floor: int = 2000
    episodic_memory_token_budget_cap: int = 12000
    compaction_trigger_turn_count: int = 8
    compaction_trigger_token_ratio: float = 1.5
    compaction_tail_preserve_turns: int = 4
    compaction_context_episode_window: int = 2
    compaction_scene_name: str = "conversation_compaction"


@dataclass(frozen=True)
class ConversationMemoryConfig:
    """多轮会话记忆配置集合。

    Args:
        default: 默认会话记忆配置。

    Returns:
        无。

    Raises:
        无。
    """

    default: ConversationMemorySettings = field(default_factory=ConversationMemorySettings)


@dataclass(frozen=True)
class ExecutionOptions:
    """请求级执行覆盖参数。

    Args:
        model_name: 覆盖模型名。
        temperature: 覆盖温度。
        debug_sse: 是否开启 SSE 调试。
        debug_tool_delta: 是否开启工具 delta 调试。
        debug_sse_sample_rate: SSE 调试采样率。
        debug_sse_throttle_sec: SSE 调试节流秒数。
        tool_timeout_seconds: 单次工具调用超时秒数。
        max_iterations: Agent 最大迭代次数。
        fallback_mode: fallback 策略。
        fallback_prompt: fallback prompt。
        max_consecutive_failed_tool_batches: 连续失败工具批次数阈值。
        max_duplicate_tool_calls: 重复工具调用阈值。
        duplicate_tool_hint_prompt: 重复工具提示词。
        web_provider: 联网 provider 覆盖。
        trace_enabled: 是否开启 trace。
        trace_output_dir: trace 输出目录。
        toolset_configs: 通用 toolset 配置快照。
        toolset_config_overrides: 通用 toolset 覆盖快照。
        doc_tool_limits: 旧式 doc 配置。
        fins_tool_limits: 旧式 fins 配置。
        web_tools_config: 旧式 web 配置。

    Returns:
        无。

    Raises:
        TypeError: 当 toolset 配置无法序列化时抛出。
        ValueError: 当 toolset 名称非法时抛出。
    """

    model_name: str | None = None
    temperature: float | None = None
    debug_sse: bool = False
    debug_tool_delta: bool = False
    debug_sse_sample_rate: float | None = None
    debug_sse_throttle_sec: float | None = None
    tool_timeout_seconds: float | None = None
    max_iterations: int | None = None
    fallback_mode: str | None = None
    fallback_prompt: str | None = None
    max_consecutive_failed_tool_batches: int | None = None
    max_duplicate_tool_calls: int | None = None
    duplicate_tool_hint_prompt: str | None = None
    web_provider: str | None = None
    trace_enabled: bool | None = None
    trace_output_dir: Path | None = None
    toolset_configs: tuple[ToolsetConfigSnapshot, ...] = field(default_factory=tuple)
    toolset_config_overrides: tuple[ToolsetConfigSnapshot, ...] = field(default_factory=tuple)
    doc_tool_limits: DocToolLimits | None = None
    fins_tool_limits: FinsToolLimits | None = None
    web_tools_config: WebToolsConfig | None = None

    def __post_init__(self) -> None:
        """规范化请求级 toolset 配置快照。

        Args:
            无。

        Returns:
            无。

        Raises:
            TypeError: 当 toolset 配置无法序列化时抛出。
            ValueError: 当 toolset 名称非法时抛出。
        """

        normalized_configs = build_legacy_toolset_configs(
            doc_tool_limits=self.doc_tool_limits,
            fins_tool_limits=self.fins_tool_limits,
            web_tools_config=self.web_tools_config,
        )
        for snapshot in normalize_toolset_configs(self.toolset_configs):
            normalized_configs = replace_toolset_config(normalized_configs, snapshot)
        object.__setattr__(self, "toolset_configs", normalized_configs)
        object.__setattr__(
            self,
            "toolset_config_overrides",
            normalize_toolset_configs(self.toolset_config_overrides),
        )


def _snapshot_optional_str(value: ExecutionOptionsSnapshotValue) -> str | None:
    """从快照值读取可选字符串。

    Args:
        value: 原始快照值。

    Returns:
        字符串或 ``None``。

    Raises:
        ValueError: 当值不是字符串时抛出。
    """

    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError("execution option snapshot value must be string")


def _snapshot_optional_float(value: ExecutionOptionsSnapshotValue) -> float | None:
    """从快照值读取可选浮点数。

    Args:
        value: 原始快照值。

    Returns:
        浮点数或 ``None``。

    Raises:
        ValueError: 当值不是数字时抛出。
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("execution option snapshot value must be number")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError("execution option snapshot value must be number")


def _snapshot_optional_int(value: ExecutionOptionsSnapshotValue) -> int | None:
    """从快照值读取可选整数。

    Args:
        value: 原始快照值。

    Returns:
        整数或 ``None``。

    Raises:
        ValueError: 当值不是整数时抛出。
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("execution option snapshot value must be int")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError("execution option snapshot value must be int")


def _snapshot_optional_bool(value: ExecutionOptionsSnapshotValue) -> bool | None:
    """从快照值读取可选布尔值。

    Args:
        value: 原始快照值。

    Returns:
        布尔值或 ``None``。

    Raises:
        ValueError: 当值不是布尔值时抛出。
    """

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise ValueError("execution option snapshot value must be bool")


def _snapshot_optional_mapping(
    value: ExecutionOptionsSnapshotValue | None,
) -> dict[str, ExecutionOptionsSnapshotValue] | None:
    """从快照值读取可选对象。

    Args:
        value: 原始快照值。

    Returns:
        对象映射或 ``None``。

    Raises:
        ValueError: 当值不是对象时抛出。
    """

    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    raise ValueError("execution option snapshot value must be object")


def _snapshot_optional_list(
    value: ExecutionOptionsSnapshotValue | None,
) -> list[ExecutionOptionsSnapshotValue] | None:
    """从快照值读取可选数组。

    Args:
        value: 原始快照值。

    Returns:
        数组或 ``None``。

    Raises:
        ValueError: 当值不是数组时抛出。
    """

    if value is None:
        return None
    if isinstance(value, list):
        return value
    raise ValueError("execution option snapshot value must be list")


def _serialize_dataclass_snapshot(
    value: object | None,
) -> dict[str, ExecutionOptionsSnapshotValue] | None:
    """把 dataclass 序列化为快照对象。

    Args:
        value: 待序列化对象。

    Returns:
        JSON 兼容对象或 ``None``。

    Raises:
        TypeError: 当输入不是 dataclass 实例时抛出。
    """

    if value is None:
        return None
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("value must be dataclass instance")
    return cast(dict[str, ExecutionOptionsSnapshotValue], asdict(value))


def _coerce_toolset_config_payload(
    payload: Mapping[str, ExecutionOptionsSnapshotValue],
) -> dict[str, ToolsetConfigValue]:
    """把执行参数层快照对象收窄为 toolset 配置 payload。

    Args:
        payload: 原始快照对象。

    Returns:
        规范化后的 toolset payload。

    Raises:
        TypeError: 当 payload 中存在不支持的值类型时抛出。
    """

    return {
        str(key): cast(
            ToolsetConfigValue,
            serialize_toolset_config_payload_value(cast(ToolsetConfigValue, value)),
        )
        for key, value in payload.items()
    }


def _build_toolset_config_from_snapshot(
    payload: Mapping[str, ExecutionOptionsSnapshotValue],
) -> ToolsetConfigSnapshot:
    """从执行参数快照对象恢复单个 toolset 配置。

    Args:
        payload: 单个 toolset 快照对象。

    Returns:
        恢复后的 toolset 配置快照。

    Raises:
        ValueError: 当对象缺少必要字段时抛出。
    """

    toolset_name = _snapshot_optional_str(payload.get("toolset_name"))
    if toolset_name is None:
        raise ValueError("execution_options.toolset_configs[].toolset_name 不能为空")
    version = _snapshot_optional_str(payload.get("version")) or "1"
    config_payload = _snapshot_optional_mapping(payload.get("payload")) or {}
    snapshot = build_toolset_config_snapshot(
        toolset_name,
        _coerce_toolset_config_payload(config_payload),
        version=version,
    )
    if snapshot is None:
        raise ValueError("execution_options.toolset_configs[] 必须包含 payload")
    return snapshot


def _build_toolset_configs_from_snapshot(
    toolset_configs_payload: list[ExecutionOptionsSnapshotValue] | None,
) -> tuple[ToolsetConfigSnapshot, ...]:
    """从执行参数快照恢复通用 toolset 配置序列。

    Args:
        toolset_configs_payload: toolset 配置快照数组。

    Returns:
        规范化后的 toolset 配置序列。

    Raises:
        ValueError: 当快照结构非法时抛出。
    """

    snapshots: tuple[ToolsetConfigSnapshot, ...] = ()
    for item in toolset_configs_payload or []:
        item_payload = _snapshot_optional_mapping(item)
        if item_payload is None:
            raise ValueError("execution_options.toolset_configs[] 必须是 JSON object")
        snapshots = replace_toolset_config(snapshots, _build_toolset_config_from_snapshot(item_payload))
    return normalize_toolset_configs(snapshots)


def _serialize_toolset_configs_snapshot(
    toolset_configs: tuple[ToolsetConfigSnapshot, ...],
) -> list[ExecutionOptionsSnapshotValue] | None:
    """把通用 toolset 配置序列序列化为执行参数快照数组。

    Args:
        toolset_configs: 原始 toolset 配置序列。

    Returns:
        快照数组；空序列时返回 ``None``。

    Raises:
        无。
    """

    if not toolset_configs:
        return None
    return cast(list[ExecutionOptionsSnapshotValue], [asdict(snapshot) for snapshot in toolset_configs])


def serialize_execution_options_snapshot(
    execution_options: ExecutionOptions | None,
) -> ExecutionOptionsSnapshot:
    """把请求级执行参数序列化为可持久化快照。

    Args:
        execution_options: 原始执行参数。

    Returns:
        仅包含显式字段的 JSON 兼容快照。

    Raises:
        TypeError: 当 dataclass 序列化失败时抛出。
    """

    if execution_options is None:
        return {}
    snapshot: ExecutionOptionsSnapshot = {
        "model_name": execution_options.model_name,
        "temperature": execution_options.temperature,
        "debug_sse": execution_options.debug_sse,
        "debug_tool_delta": execution_options.debug_tool_delta,
        "debug_sse_sample_rate": execution_options.debug_sse_sample_rate,
        "debug_sse_throttle_sec": execution_options.debug_sse_throttle_sec,
        "tool_timeout_seconds": execution_options.tool_timeout_seconds,
        "max_iterations": execution_options.max_iterations,
        "fallback_mode": execution_options.fallback_mode,
        "fallback_prompt": execution_options.fallback_prompt,
        "max_consecutive_failed_tool_batches": execution_options.max_consecutive_failed_tool_batches,
        "max_duplicate_tool_calls": execution_options.max_duplicate_tool_calls,
        "duplicate_tool_hint_prompt": execution_options.duplicate_tool_hint_prompt,
        "web_provider": execution_options.web_provider,
        "trace_enabled": execution_options.trace_enabled,
        "trace_output_dir": (
            str(execution_options.trace_output_dir)
            if execution_options.trace_output_dir is not None
            else None
        ),
        "toolset_configs": _serialize_toolset_configs_snapshot(execution_options.toolset_configs),
        "toolset_config_overrides": _serialize_toolset_configs_snapshot(
            execution_options.toolset_config_overrides
        ),
    }
    return {key: value for key, value in snapshot.items() if value is not None}


def deserialize_execution_options_snapshot(
    snapshot: Mapping[str, ExecutionOptionsSnapshotValue] | None,
) -> ExecutionOptions | None:
    """把执行参数快照恢复为请求级执行参数。

    Args:
        snapshot: 持久化的 JSON 兼容快照。

    Returns:
        恢复后的执行参数；快照为空时返回 ``None``。

    Raises:
        ValueError: 当快照包含已删除字段或结构非法时抛出。
    """

    if not snapshot:
        return None
    removed_fields = sorted(set(snapshot.keys()).intersection(_REMOVED_EXECUTION_OPTION_SNAPSHOT_FIELDS))
    if removed_fields:
        removed_list = ", ".join(removed_fields)
        raise ValueError(f"execution option snapshot contains removed fields: {removed_list}")
    toolset_configs_payload = _snapshot_optional_list(snapshot.get("toolset_configs"))
    toolset_config_overrides_payload = _snapshot_optional_list(snapshot.get("toolset_config_overrides"))
    toolset_configs = _build_toolset_configs_from_snapshot(toolset_configs_payload)
    toolset_config_overrides = _build_toolset_configs_from_snapshot(toolset_config_overrides_payload)
    trace_output_dir_text = _snapshot_optional_str(snapshot.get("trace_output_dir"))
    return ExecutionOptions(
        model_name=_snapshot_optional_str(snapshot.get("model_name")),
        temperature=_snapshot_optional_float(snapshot.get("temperature")),
        debug_sse=_snapshot_optional_bool(snapshot.get("debug_sse")) or False,
        debug_tool_delta=_snapshot_optional_bool(snapshot.get("debug_tool_delta")) or False,
        debug_sse_sample_rate=_snapshot_optional_float(snapshot.get("debug_sse_sample_rate")),
        debug_sse_throttle_sec=_snapshot_optional_float(snapshot.get("debug_sse_throttle_sec")),
        tool_timeout_seconds=_snapshot_optional_float(snapshot.get("tool_timeout_seconds")),
        max_iterations=_snapshot_optional_int(snapshot.get("max_iterations")),
        fallback_mode=_snapshot_optional_str(snapshot.get("fallback_mode")),
        fallback_prompt=_snapshot_optional_str(snapshot.get("fallback_prompt")),
        max_consecutive_failed_tool_batches=_snapshot_optional_int(
            snapshot.get("max_consecutive_failed_tool_batches")
        ),
        max_duplicate_tool_calls=_snapshot_optional_int(snapshot.get("max_duplicate_tool_calls")),
        duplicate_tool_hint_prompt=_snapshot_optional_str(snapshot.get("duplicate_tool_hint_prompt")),
        web_provider=_snapshot_optional_str(snapshot.get("web_provider")),
        trace_enabled=_snapshot_optional_bool(snapshot.get("trace_enabled")),
        trace_output_dir=Path(trace_output_dir_text) if trace_output_dir_text is not None else None,
        toolset_configs=toolset_configs,
        toolset_config_overrides=toolset_config_overrides,
    )


__all__ = [
    "ConversationMemoryConfig",
    "ConversationMemorySettings",
    "ExecutionOptions",
    "ExecutionOptionsSnapshot",
    "ExecutionOptionsSnapshotValue",
    "TraceSettings",
    "deserialize_execution_options_snapshot",
    "serialize_execution_options_snapshot",
]
