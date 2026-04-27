"""运行时选项与合并逻辑。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
import math
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

from dayu.contracts.execution_options import (
    ConversationMemoryConfig,
    ConversationMemorySettings,
    ExecutionOptions,
    TraceSettings,
)
from dayu.contracts.model_config import ConversationMemoryRuntimeHints, ModelConfig, RunnerType, normalize_runner_type
from dayu.contracts.tool_configs import (
    DocToolLimits,
    FinsToolLimits,
    WebToolsConfig,
    build_doc_tool_limits,
    build_fins_tool_limits,
    build_legacy_toolset_configs,
    build_web_tools_config,
)
from dayu.contracts.toolset_config import (
    ToolsetConfigSnapshot,
    ToolsetConfigValue,
    build_toolset_config_snapshot,
    find_toolset_config,
    normalize_toolset_configs,
    replace_toolset_config,
    serialize_toolset_config_payload_value,
)
from dayu.execution.runtime_config import AgentRuntimeConfig, OpenAIRunnerRuntimeConfig, RunnerRuntimeConfig

@dataclass(frozen=True)
class _ToolTraceConfigDefaults:
    """工具调用追踪默认配置。"""

    enabled: bool
    output_dir: str
    max_file_bytes: int
    retention_days: int
    compress_rolled: bool
    partition_by_session: bool


@dataclass(frozen=True)
class _DefaultRunConfig:
    """运行时默认配置真源。"""

    runner_running_config: OpenAIRunnerRuntimeConfig
    agent_running_config: AgentRuntimeConfig
    doc_tool_limits: DocToolLimits
    fins_tool_limits: FinsToolLimits
    web_tools_config: WebToolsConfig
    tool_trace_config: _ToolTraceConfigDefaults
    conversation_memory: ConversationMemoryConfig

_MIN_TEMPERATURE = 0.0
_MAX_TEMPERATURE = 2.0

ExecutionOptionsOverrideValue: TypeAlias = str | int | float | bool | None
ExecutionOptionsOverridePayload: TypeAlias = dict[str, ExecutionOptionsOverrideValue]
ModelRunnerRuntimeOverrideValue: TypeAlias = str | int | float | bool | None

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


def _resolve_optional_workspace_path(path_value: Any, *, workspace_dir: Path) -> str:
    """解析可选路径配置。

    Args:
        path_value: 原始路径值。
        workspace_dir: 工作区根目录。

    Returns:
        解析后的绝对路径字符串；空值时返回空字符串。

    Raises:
        无。
    """

    if path_value is None:
        return ""
    raw = str(path_value).strip()
    if not raw:
        return ""
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    return str((workspace_dir / candidate).resolve())


_DEFAULT_RUN_CONFIG = _DefaultRunConfig(
    runner_running_config=OpenAIRunnerRuntimeConfig(
        debug_sse=False,
        debug_tool_delta=False,
        debug_sse_sample_rate=1.0,
        debug_sse_throttle_sec=0.0,
        tool_timeout_seconds=90.0,
    ),
    agent_running_config=AgentRuntimeConfig(),
    doc_tool_limits=DocToolLimits(
        list_files_max=200,
        get_sections_max=200,
        search_files_max_results=50,
        read_file_max_chars=80000,
        read_file_section_max_chars=50000,
    ),
    fins_tool_limits=FinsToolLimits(
        processor_cache_max_entries=128,
        list_documents_max_items=300,
        get_document_sections_max_items=1200,
        search_document_max_items=20,
        list_tables_max_items=50,
        read_section_max_chars=80000,
        get_page_content_max_chars=80000,
        get_table_max_items=800,
        get_financial_statement_max_items=1200,
        query_xbrl_facts_max_items=1200,
    ),
    web_tools_config=WebToolsConfig(
        provider="auto",
        request_timeout_seconds=12.0,
        max_search_results=20,
        fetch_truncate_chars=80000,
        allow_private_network_url=False,
        playwright_channel="chrome",
        playwright_storage_state_dir="output/web_diagnostics/storage_states",
    ),
    tool_trace_config=_ToolTraceConfigDefaults(
        enabled=False,
        output_dir="output/tool_call_traces",
        max_file_bytes=64 * 1024 * 1024,
        retention_days=30,
        compress_rolled=True,
        partition_by_session=True,
    ),
    conversation_memory=ConversationMemoryConfig(
        default=ConversationMemorySettings(
            memory_token_budget_ratio=0.10,
            memory_token_budget_floor=4000,
            memory_token_budget_cap=60000,
            recent_turns_floor=2,
            compaction_trigger_context_ratio=0.70,
            compaction_tail_preserve_turns=4,
            compaction_context_episode_window=2,
            compaction_scene_name="conversation_compaction",
        )
    ),
)


def normalize_temperature(
    value: Any,
    *,
    field_name: str = "temperature",
) -> float | None:
    """标准化 temperature 配置。

    Args:
        value: 原始 temperature 值。
        field_name: 字段名，仅用于错误提示。

    Returns:
        标准化后的 temperature；输入为空时返回 ``None``。

    Raises:
        ValueError: 当值不是有限数值或不在 ``[0, 2]`` 范围内时抛出。
    """

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是数值")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是数值") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} 必须是有限数值")
    if normalized < _MIN_TEMPERATURE or normalized > _MAX_TEMPERATURE:
        raise ValueError(f"{field_name} 必须位于 [{_MIN_TEMPERATURE}, {_MAX_TEMPERATURE}]")
    return normalized


def _parse_override_int(
    value: ExecutionOptionsOverrideValue,
    *,
    field_name: str,
) -> int | None:
    """解析请求级 override 中的整数值。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} 必须是整数")
    return value


def _parse_override_float(
    value: ExecutionOptionsOverrideValue,
    *,
    field_name: str,
) -> float | None:
    """解析请求级 override 中的浮点值。"""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} 必须是数值")
    return float(value)


def _parse_override_bool(
    value: ExecutionOptionsOverrideValue,
    *,
    field_name: str,
) -> bool | None:
    """解析请求级 override 中的布尔值。"""

    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是布尔值")
    return value


def _parse_override_str(
    value: ExecutionOptionsOverrideValue,
    *,
    field_name: str,
) -> str | None:
    """解析请求级 override 中的字符串值。"""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是字符串")
    return value


def _validate_override_keys(
    payload: Mapping[str, ExecutionOptionsOverrideValue],
    *,
    allowed_keys: frozenset[str],
    field_name: str,
) -> None:
    """校验请求级 override 的键集合。"""

    unknown_keys = sorted(set(payload.keys()) - allowed_keys)
    if unknown_keys:
        raise ValueError(f"{field_name} 包含未知字段: {unknown_keys}")


def resolve_scene_temperature(
    *,
    resolved_temperature: float | None,
    model_config: Mapping[str, Any],
    temperature_profile: str,
    scene_name: str,
    model_name: str,
) -> float:
    """按统一规则解析 scene 最终 temperature。

    Args:
        resolved_temperature: 已经过请求级覆盖与基础配置合并后的显式 temperature。
        model_config: 当前模型配置。
        temperature_profile: scene manifest 声明的 temperature profile 名称。
        scene_name: scene 名称，仅用于错误提示。
        model_name: 当前模型名，仅用于错误提示。

    Returns:
        当前 scene 最终生效的 temperature。

    Raises:
        ValueError: 当模型配置缺少 temperature profile 或 temperature 非法时抛出。
    """

    if resolved_temperature is not None:
        # 防御性兜底：``normalize_temperature`` 在输入为空白字符串时会返回 ``None``；
        # 当前外层已过滤 ``None``，但不过滤空串。这里显式 ``or 0.0`` 保证返回严格 float，
        # 同时避免向调用方泄漏 "空字符串" 的语义。
        return normalize_temperature(
            resolved_temperature,
            field_name="resolved_execution_options.temperature",
        ) or 0.0
    runtime_hints = model_config.get("runtime_hints")
    if not isinstance(runtime_hints, dict):
        raise ValueError(
            f"scene={scene_name}, model={model_name} 缺少 llm_models.runtime_hints"
        )
    temperature_profiles = runtime_hints.get("temperature_profiles")
    if not isinstance(temperature_profiles, dict):
        raise ValueError(
            f"scene={scene_name}, model={model_name} 缺少 temperature_profiles"
        )
    raw_profile = temperature_profiles.get(temperature_profile)
    if not isinstance(raw_profile, dict):
        raise ValueError(
            f"scene={scene_name}, model={model_name} 缺少 "
            f"temperature_profiles[{temperature_profile}]"
        )
    normalized = normalize_temperature(
        raw_profile.get("temperature"),
        field_name=f"temperature_profiles[{temperature_profile}].temperature",
    )
    if normalized is None:
        raise ValueError(
            f"scene={scene_name}, model={model_name} 缺少 temperature"
        )
    return normalized


def _build_doc_tool_limits_from_override(
    payload: Mapping[str, ExecutionOptionsOverrideValue] | None,
    *,
    base: DocToolLimits | None = None,
) -> DocToolLimits | None:
    """从请求级原始 override 稀疏覆盖文档工具限制。

    未指定的字段保留 base 中的值；base 为 None 时回退到库默认值。

    Args:
        payload: 请求级原始 override 载荷。
        base: 基础配置，来自 run.json 合并后的运行时选项。

    Returns:
        合并后的文档工具限制；payload 为 None 时返回 None。

    Raises:
        ValueError: 当 override 包含非法字段或值时抛出。
    """

    if payload is None:
        return None
    _validate_override_keys(
        payload,
        allowed_keys=frozenset(
            {
                "list_files_max",
                "get_sections_max",
                "search_files_max_results",
                "read_file_max_chars",
                "read_file_section_max_chars",
            }
        ),
        field_name="execution_options.doc_tool_limits_override",
    )
    defaults = base or DocToolLimits()
    # 仅收集请求中显式指定的字段，未指定的保留 base 值
    overrides: dict[str, int] = {}
    for key in payload:
        parsed = _parse_override_int(
            payload[key],
            field_name=f"execution_options.doc_tool_limits_override.{key}",
        )
        if parsed is not None:
            overrides[key] = parsed
    return replace(defaults, **overrides)


def _build_fins_tool_limits_from_override(
    payload: Mapping[str, ExecutionOptionsOverrideValue] | None,
    *,
    base: FinsToolLimits | None = None,
) -> FinsToolLimits | None:
    """从请求级原始 override 稀疏覆盖财报工具限制。

    未指定的字段保留 base 中的值；base 为 None 时回退到库默认值。

    Args:
        payload: 请求级原始 override 载荷。
        base: 基础配置，来自 run.json 合并后的运行时选项。

    Returns:
        合并后的财报工具限制；payload 为 None 时返回 None。

    Raises:
        ValueError: 当 override 包含非法字段或值时抛出。
    """

    if payload is None:
        return None
    _validate_override_keys(
        payload,
        allowed_keys=frozenset(
            {
                "processor_cache_max_entries",
                "list_documents_max_items",
                "get_document_sections_max_items",
                "search_document_max_items",
                "list_tables_max_items",
                "read_section_max_chars",
                "get_page_content_max_chars",
                "get_table_max_items",
                "get_financial_statement_max_items",
                "query_xbrl_facts_max_items",
            }
        ),
        field_name="execution_options.fins_tool_limits_override",
    )
    defaults = base or FinsToolLimits()
    # 仅收集请求中显式指定的字段，未指定的保留 base 值
    overrides: dict[str, int] = {}
    for key in payload:
        parsed = _parse_override_int(
            payload[key],
            field_name=f"execution_options.fins_tool_limits_override.{key}",
        )
        if parsed is not None:
            overrides[key] = parsed
    return replace(defaults, **overrides)


# -- WebToolsConfig override 字段类型映射 --
_WEB_OVERRIDE_STR_FIELDS = frozenset({"provider", "playwright_channel", "playwright_storage_state_dir"})
_WEB_OVERRIDE_FLOAT_FIELDS = frozenset({"request_timeout_seconds"})
_WEB_OVERRIDE_INT_FIELDS = frozenset({"max_search_results", "fetch_truncate_chars"})
_WEB_OVERRIDE_BOOL_FIELDS = frozenset({"allow_private_network_url"})
_WEB_OVERRIDE_ALL_FIELDS = (
    _WEB_OVERRIDE_STR_FIELDS | _WEB_OVERRIDE_FLOAT_FIELDS | _WEB_OVERRIDE_INT_FIELDS | _WEB_OVERRIDE_BOOL_FIELDS
)


def _build_web_tools_config_from_override(
    payload: Mapping[str, ExecutionOptionsOverrideValue] | None,
    *,
    base: WebToolsConfig | None = None,
) -> WebToolsConfig | None:
    """从请求级原始 override 稀疏覆盖联网工具配置。

    未指定的字段保留 base 中的值；base 为 None 时回退到库默认值。

    Args:
        payload: 请求级原始 override 载荷。
        base: 基础配置，来自 run.json 合并后的运行时选项。

    Returns:
        合并后的联网工具配置；payload 为 None 时返回 None。

    Raises:
        ValueError: 当 override 包含非法字段或值时抛出。
    """

    if payload is None:
        return None
    _validate_override_keys(
        payload,
        allowed_keys=_WEB_OVERRIDE_ALL_FIELDS,
        field_name="execution_options.web_tools_config_override",
    )
    defaults = base or WebToolsConfig()
    _FN = "execution_options.web_tools_config_override"
    # 按类型分组解析，仅收集显式指定的字段
    overrides: dict[str, str | int | float | bool] = {}
    for key in payload:
        if key in _WEB_OVERRIDE_STR_FIELDS:
            val_s = _parse_override_str(payload[key], field_name=f"{_FN}.{key}")
            if val_s is not None:
                overrides[key] = val_s
        elif key in _WEB_OVERRIDE_FLOAT_FIELDS:
            val_f = _parse_override_float(payload[key], field_name=f"{_FN}.{key}")
            if val_f is not None:
                overrides[key] = val_f
        elif key in _WEB_OVERRIDE_INT_FIELDS:
            val_i = _parse_override_int(payload[key], field_name=f"{_FN}.{key}")
            if val_i is not None:
                overrides[key] = val_i
        elif key in _WEB_OVERRIDE_BOOL_FIELDS:
            val_b = _parse_override_bool(payload[key], field_name=f"{_FN}.{key}")
            if val_b is not None:
                overrides[key] = val_b
    return replace(defaults, **overrides)


def _snapshot_optional_str(value: ExecutionOptionsSnapshotValue) -> str | None:
    """从执行参数快照中读取可选字符串。"""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _snapshot_optional_float(value: ExecutionOptionsSnapshotValue) -> float | None:
    """从执行参数快照中读取可选浮点数。

    Args:
        value: 原始快照值。

    Returns:
        浮点数或 ``None``。

    Raises:
        ValueError: 当值类型不是 ``None`` / ``int`` / ``float`` 时抛出。``bool``
            属于 ``int`` 的子类但语义上不是数字，混入 float 快照通常意味着上游
            序列化 bug，直接 raise 以便尽早暴露（与 ``contracts.execution_options``
            / ``contracts.agent_execution_serialization`` 中的同名函数保持一致的
            严格策略）。
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("execution option snapshot value must be number")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError("execution option snapshot value must be number")


def _snapshot_optional_int(value: ExecutionOptionsSnapshotValue) -> int | None:
    """从执行参数快照中读取可选整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _snapshot_optional_bool(value: ExecutionOptionsSnapshotValue) -> bool | None:
    """从执行参数快照中读取可选布尔值。"""

    if not isinstance(value, bool):
        return None
    return value


def _snapshot_optional_mapping(
    value: ExecutionOptionsSnapshotValue | None,
) -> dict[str, ExecutionOptionsSnapshotValue] | None:
    """从执行参数快照中读取可选对象映射。"""

    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def _snapshot_optional_list(
    value: ExecutionOptionsSnapshotValue | None,
) -> list[ExecutionOptionsSnapshotValue] | None:
    """从执行参数快照中读取可选数组。"""

    if not isinstance(value, list):
        return None
    return list(value)


def _serialize_dataclass_snapshot(value: object | None) -> dict[str, ExecutionOptionsSnapshotValue] | None:
    """把嵌套 dataclass 序列化为快照对象。"""

    if value is None:
        return None
    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError(f"不支持序列化的 dataclass 快照类型: {type(value).__name__}")
    return cast(dict[str, ExecutionOptionsSnapshotValue], asdict(value))


def _narrow_override_payload(
    mapping: dict[str, ExecutionOptionsSnapshotValue] | None,
) -> ExecutionOptionsOverridePayload | None:
    """将快照映射收窄为标量级 override 载荷。

    Args:
        mapping: 从快照读取的可选映射。

    Returns:
        仅包含标量值的 override 载荷；输入为空时返回 None。

    Raises:
        ValueError: 当映射包含嵌套对象时抛出。
    """

    if mapping is None:
        return None
    result: ExecutionOptionsOverridePayload = {}
    for key, value in mapping.items():
        if isinstance(value, dict | list):
            raise ValueError(f"override payload 不支持嵌套对象: {key}")
        result[key] = value
    return result or None


def _snapshot_int_or_default(
    value: ExecutionOptionsSnapshotValue | None,
    *,
    default: int,
) -> int:
    """从快照值读取整数，缺失时回退到默认值。

    Args:
        value: 原始快照值。
        default: 默认值。

    Returns:
        合法整数或默认值。

    Raises:
        无。
    """

    parsed = _snapshot_optional_int(value)
    if parsed is None:
        return default
    return parsed


def _snapshot_float_or_default(
    value: ExecutionOptionsSnapshotValue | None,
    *,
    default: float,
) -> float:
    """从快照值读取浮点数，缺失时回退到默认值。

    Args:
        value: 原始快照值。
        default: 默认值。

    Returns:
        合法浮点数或默认值。

    Raises:
        无。
    """

    parsed = _snapshot_optional_float(value)
    if parsed is None:
        return default
    return parsed


def _snapshot_bool_or_default(
    value: ExecutionOptionsSnapshotValue | None,
    *,
    default: bool,
) -> bool:
    """从快照值读取布尔值，缺失时回退到默认值。

    Args:
        value: 原始快照值。
        default: 默认值。

    Returns:
        合法布尔值或默认值。

    Raises:
        无。
    """

    parsed = _snapshot_optional_bool(value)
    if parsed is None:
        return default
    return parsed


def _snapshot_str_or_default(
    value: ExecutionOptionsSnapshotValue | None,
    *,
    default: str,
) -> str:
    """从快照值读取字符串，缺失时回退到默认值。

    Args:
        value: 原始快照值。
        default: 默认值。

    Returns:
        合法字符串或默认值。

    Raises:
        无。
    """

    parsed = _snapshot_optional_str(value)
    if parsed is None:
        return default
    return parsed


def _resolve_toolset_override_payload(
    snapshots: tuple[ToolsetConfigSnapshot, ...],
    *,
    toolset_name: str,
) -> ExecutionOptionsOverridePayload | None:
    """从通用 toolset override 快照提取单个 toolset 的稀疏 override。

    Args:
        snapshots: 通用 toolset override 快照序列。
        toolset_name: 目标 toolset 名称。

    Returns:
        命中的稀疏 override；不存在时返回 ``None``。

    Raises:
        ValueError: 当 payload 含嵌套对象或列表时抛出。
    """

    snapshot = find_toolset_config(snapshots, toolset_name)
    if snapshot is None:
        return None
    override_payload: ExecutionOptionsOverridePayload = {}
    for key, value in snapshot.payload.items():
        if isinstance(value, dict | list):
            raise ValueError(f"toolset override payload 不支持嵌套结构: {toolset_name}.{key}")
        override_payload[str(key)] = value
    return override_payload or None


def _resolve_doc_tool_limits_from_toolset_configs(
    toolset_configs: tuple[ToolsetConfigSnapshot, ...],
) -> DocToolLimits | None:
    """从通用 toolset 快照恢复文档工具限制。"""

    snapshot = find_toolset_config(toolset_configs, "doc")
    if snapshot is None:
        return None
    return build_doc_tool_limits(snapshot)


def resolve_doc_tool_limits_from_toolset_configs(
    toolset_configs: tuple[ToolsetConfigSnapshot, ...],
) -> DocToolLimits | None:
    """从通用 toolset 快照恢复文档工具限制。

    Args:
        toolset_configs: 通用 toolset 配置快照。

    Returns:
        恢复后的文档工具限制；不存在时返回 ``None``。

    Raises:
        无。
    """

    return _resolve_doc_tool_limits_from_toolset_configs(toolset_configs)


def _resolve_fins_tool_limits_from_toolset_configs(
    toolset_configs: tuple[ToolsetConfigSnapshot, ...],
) -> FinsToolLimits | None:
    """从通用 toolset 快照恢复财报工具限制。"""

    snapshot = find_toolset_config(toolset_configs, "fins")
    if snapshot is None:
        return None
    return build_fins_tool_limits(snapshot)


def resolve_fins_tool_limits_from_toolset_configs(
    toolset_configs: tuple[ToolsetConfigSnapshot, ...],
) -> FinsToolLimits | None:
    """从通用 toolset 快照恢复财报工具限制。

    Args:
        toolset_configs: 通用 toolset 配置快照。

    Returns:
        恢复后的财报工具限制；不存在时返回 ``None``。

    Raises:
        无。
    """

    return _resolve_fins_tool_limits_from_toolset_configs(toolset_configs)


def _resolve_web_tools_config_from_toolset_configs(
    toolset_configs: tuple[ToolsetConfigSnapshot, ...],
) -> WebToolsConfig | None:
    """从通用 toolset 快照恢复联网工具配置。"""

    snapshot = find_toolset_config(toolset_configs, "web")
    if snapshot is None:
        return None
    return build_web_tools_config(snapshot)


def resolve_web_tools_config_from_toolset_configs(
    toolset_configs: tuple[ToolsetConfigSnapshot, ...],
) -> WebToolsConfig | None:
    """从通用 toolset 快照恢复联网工具配置。

    Args:
        toolset_configs: 通用 toolset 配置快照。

    Returns:
        恢复后的联网工具配置；不存在时返回 ``None``。

    Raises:
        无。
    """

    return _resolve_web_tools_config_from_toolset_configs(toolset_configs)


def _coerce_toolset_config_payload(
    payload: Mapping[str, ExecutionOptionsSnapshotValue],
) -> dict[str, ToolsetConfigValue]:
    """把执行参数层快照对象收窄为 toolset 配置 payload。"""

    return {
        str(key): cast(ToolsetConfigValue, serialize_toolset_config_payload_value(cast(ToolsetConfigValue, value)))
        for key, value in payload.items()
    }


def _build_toolset_config_from_snapshot(
    payload: Mapping[str, ExecutionOptionsSnapshotValue],
) -> ToolsetConfigSnapshot:
    """从执行参数快照对象恢复单个 toolset 配置。"""

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
    """从执行参数快照恢复通用 toolset 配置序列。"""

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
    """把通用 toolset 配置序列序列化为执行参数快照数组。"""

    if not toolset_configs:
        return None
    return cast(
        list[ExecutionOptionsSnapshotValue],
        [asdict(snapshot) for snapshot in toolset_configs],
    )


def _build_toolset_config_overrides_from_snapshot(
    toolset_config_overrides_payload: list[ExecutionOptionsSnapshotValue] | None,
) -> tuple[ToolsetConfigSnapshot, ...]:
    """从快照恢复通用 toolset override 序列。

    Args:
        toolset_config_overrides_payload: 新格式的通用 override 数组。

    Returns:
        规范化后的通用 toolset override 序列。

    Raises:
        ValueError: 当快照结构非法时抛出。
    """

    snapshots: tuple[ToolsetConfigSnapshot, ...] = ()
    for item in toolset_config_overrides_payload or []:
        item_payload = _snapshot_optional_mapping(item)
        if item_payload is None:
            raise ValueError("execution_options.toolset_config_overrides[] 必须是 JSON object")
        snapshots = replace_toolset_config(snapshots, _build_toolset_config_from_snapshot(item_payload))
    return normalize_toolset_configs(snapshots)


def _merge_toolset_configs(
    base_configs: tuple[ToolsetConfigSnapshot, ...],
    incoming_configs: tuple[ToolsetConfigSnapshot, ...],
) -> tuple[ToolsetConfigSnapshot, ...]:
    """按 toolset 名称把 incoming 快照覆盖到 base 快照。"""

    merged_configs = normalize_toolset_configs(base_configs)
    for snapshot in normalize_toolset_configs(incoming_configs):
        merged_configs = replace_toolset_config(merged_configs, snapshot)
    return merged_configs


def serialize_execution_options_snapshot(
    execution_options: ExecutionOptions | None,
) -> ExecutionOptionsSnapshot:
    """把请求级执行参数序列化为可持久化快照。

    Args:
        execution_options: 原始执行参数。

    Returns:
        仅包含显式字段的 JSON 兼容快照。

    Raises:
        无。
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
    return {
        key: value
        for key, value in snapshot.items()
        if value is not None
    }


def deserialize_execution_options_snapshot(
    snapshot: Mapping[str, ExecutionOptionsSnapshotValue] | None,
) -> ExecutionOptions | None:
    """把执行参数快照恢复为请求级执行参数。

    Args:
        snapshot: 持久化的 JSON 兼容快照。

    Returns:
        恢复后的执行参数；快照为空时返回 ``None``。

    Raises:
        无。
    """

    if not snapshot:
        return None
    removed_fields = sorted(set(snapshot.keys()).intersection(_REMOVED_EXECUTION_OPTION_SNAPSHOT_FIELDS))
    if removed_fields:
        removed_list = ", ".join(removed_fields)
        raise ValueError(f"execution option snapshot contains removed fields: {removed_list}")
    toolset_configs_payload = _snapshot_optional_list(snapshot.get("toolset_configs"))
    toolset_config_overrides_payload = _snapshot_optional_list(snapshot.get("toolset_config_overrides"))
    toolset_config_overrides = _build_toolset_config_overrides_from_snapshot(
        toolset_config_overrides_payload,
    )
    toolset_configs = _build_toolset_configs_from_snapshot(toolset_configs_payload)
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


@dataclass(frozen=True)
class ResolvedExecutionOptions:
    """合并后的执行选项。"""

    model_name: str
    runner_running_config: RunnerRuntimeConfig
    agent_running_config: AgentRuntimeConfig
    trace_settings: TraceSettings
    toolset_configs: tuple[ToolsetConfigSnapshot, ...] = ()
    temperature: float | None = None
    conversation_memory_config: ConversationMemoryConfig = field(
        default_factory=ConversationMemoryConfig
    )
    conversation_memory_settings: ConversationMemorySettings = field(
        default_factory=ConversationMemorySettings
    )

    def __post_init__(self) -> None:
        """执行冻结 dataclass 的派生字段规范化。

        Args:
            无。

        Returns:
            无。

        Raises:
            TypeError: 当 toolset 配置无法序列化时抛出。
            ValueError: 当 toolset 名称非法时抛出。
        """

        object.__setattr__(self, "toolset_configs", normalize_toolset_configs(self.toolset_configs))


def _merge_section(base: dict[str, Any], incoming: Any) -> dict[str, Any]:
    """合并单个配置段。"""

    if not isinstance(incoming, dict):
        return dict(base)
    return {**base, **incoming}


def _build_conversation_memory_settings(section: dict[str, Any]) -> ConversationMemorySettings:
    """从配置字典构建会话记忆设置。

    Args:
        section: 单个会话记忆配置段。

    Returns:
        规范化后的会话记忆设置。

    Raises:
        KeyError: 当缺少必要字段时抛出。
        ValueError: 当字段值无法转换为目标类型时抛出。
    """

    return ConversationMemorySettings(
        memory_token_budget_ratio=float(section["memory_token_budget_ratio"]),
        memory_token_budget_floor=int(section["memory_token_budget_floor"]),
        memory_token_budget_cap=int(section["memory_token_budget_cap"]),
        recent_turns_floor=int(section["recent_turns_floor"]),
        compaction_trigger_context_ratio=float(section["compaction_trigger_context_ratio"]),
        compaction_tail_preserve_turns=int(section["compaction_tail_preserve_turns"]),
        compaction_context_episode_window=int(section["compaction_context_episode_window"]),
        compaction_scene_name=str(section["compaction_scene_name"]).strip() or "conversation_compaction",
    )


def _merge_conversation_memory_config(section: Any) -> ConversationMemoryConfig:
    """合并 `conversation_memory` 配置。

    Args:
        section: `run.json` 中的 `conversation_memory` 原始值。

    Returns:
        规范化后的会话记忆配置集合。

    Raises:
        无。
    """

    base_section = asdict(_DEFAULT_RUN_CONFIG.conversation_memory)
    incoming = section if isinstance(section, dict) else {}
    merged_default = _merge_section(base_section["default"], incoming.get("default"))
    return ConversationMemoryConfig(default=_build_conversation_memory_settings(merged_default))


def _normalize_runtime_hints_conversation_memory(
    model_config: ModelConfig,
) -> ConversationMemoryRuntimeHints:
    """从模型配置提取 conversation_memory runtime hints。

    Args:
        model_config: 单个模型配置字典。

    Returns:
        仅包含 conversation_memory 覆盖字段的字典。

    Raises:
        ValueError: 当 runtime_hints 配置结构非法时抛出。
    """

    runtime_hints = model_config.get("runtime_hints")
    if runtime_hints is None:
        return {}
    conversation_memory = runtime_hints.get("conversation_memory")
    if conversation_memory is None:
        return {}
    return conversation_memory


def _validate_conversation_memory_settings(settings: ConversationMemorySettings) -> None:
    """校验分层记忆配置的静态约束。

    Args:
        settings: 已规范化的分层记忆配置。

    Returns:
        无。

    Raises:
        ValueError: 当配置违反静态约束时抛出。
    """

    if settings.memory_token_budget_floor <= 0:
        raise ValueError("conversation_memory.memory_token_budget_floor 必须大于 0")
    if not math.isfinite(settings.memory_token_budget_ratio) or settings.memory_token_budget_ratio < 0:
        raise ValueError("conversation_memory.memory_token_budget_ratio 必须是非负有限数值")
    if settings.memory_token_budget_ratio > 1.0:
        raise ValueError("conversation_memory.memory_token_budget_ratio 不能大于 1")
    if settings.memory_token_budget_cap < settings.memory_token_budget_floor:
        raise ValueError("conversation_memory.memory_token_budget_cap 不能小于 floor")
    if settings.recent_turns_floor < 0:
        raise ValueError("conversation_memory.recent_turns_floor 必须是非负整数")
    if not math.isfinite(settings.compaction_trigger_context_ratio):
        raise ValueError("conversation_memory.compaction_trigger_context_ratio 必须是有限数值")
    if settings.compaction_trigger_context_ratio <= 0 or settings.compaction_trigger_context_ratio > 1.0:
        raise ValueError("conversation_memory.compaction_trigger_context_ratio 必须落在 (0, 1] 区间")
    if settings.compaction_tail_preserve_turns < 0:
        raise ValueError("conversation_memory.compaction_tail_preserve_turns 必须是非负整数")
    if settings.compaction_context_episode_window < 0:
        raise ValueError("conversation_memory.compaction_context_episode_window 必须是非负整数")
    if not settings.compaction_scene_name.strip():
        raise ValueError("conversation_memory.compaction_scene_name 不能为空")


def resolve_conversation_memory_settings(
    *,
    conversation_memory_config: ConversationMemoryConfig,
    model_config: ModelConfig | None,
) -> ConversationMemorySettings:
    """根据全局默认值与模型 runtime hints 解析最终分层记忆配置。

    注意：
        这里刻意采用“字段级合并”而不是“单一真源优先级覆盖”。
        `run.json.conversation_memory.default` 提供全局公式，
        `llm_models.runtime_hints.conversation_memory` 只覆盖少数字段。
        后续若改成整对象覆盖，会直接改变 memory policy 的设计语义。

    Args:
        conversation_memory_config: 来自 ``run.json`` 的全局默认 policy。
        model_config: 当前模型配置字典。

    Returns:
        合并模型特例后的分层记忆配置。

    Raises:
        ValueError: 当 runtime hints 非法时抛出。
    """

    # 这里故意先取 run.json 默认公式，再按字段叠加模型 runtime hints，
    # 避免模型特例把整套 memory policy 意外替换掉。
    base_section = vars(conversation_memory_config.default)
    hint_overrides = _normalize_runtime_hints_conversation_memory(model_config or {})
    settings = _build_conversation_memory_settings(_merge_section(base_section, hint_overrides))
    _validate_conversation_memory_settings(settings)
    return settings


def _parse_model_runner_positive_float(
    value: ModelRunnerRuntimeOverrideValue,
    *,
    field_name: str,
) -> float | None:
    """解析模型级 Runner 正数秒值。

    Args:
        value: 模型配置中的原始秒值。
        field_name: 字段名，仅用于错误提示。

    Returns:
        解析后的正数秒值；未配置时返回 ``None``。

    Raises:
        ValueError: 当值不是正有限数值时抛出。
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} 必须是正数")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{field_name} 必须是正数")
        try:
            normalized = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{field_name} 必须是正数") from exc
    elif isinstance(value, int | float):
        normalized = float(value)
    else:
        raise ValueError(f"{field_name} 必须是正数")
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} 必须是正有限数值")
    return normalized


def apply_model_runner_runtime_overrides(
    *,
    resolved_execution_options: ResolvedExecutionOptions,
    model_config: ModelConfig,
) -> ResolvedExecutionOptions:
    """按模型配置覆盖 Runner 运行时参数。

    Args:
        resolved_execution_options: 已解析的执行选项基线。
        model_config: 当前模型配置。

    Returns:
        应用模型级 Runner 覆盖后的执行选项。

    Raises:
        ValueError: 当 runner_type 或模型级 Runner 运行时配置非法时抛出。
    """

    runner_type = normalize_runner_type(model_config.get("runner_type"))
    if runner_type != RunnerType.OPENAI_COMPATIBLE:
        return resolved_execution_options
    runner_running_config = resolved_execution_options.runner_running_config
    if not isinstance(runner_running_config, OpenAIRunnerRuntimeConfig):
        return resolved_execution_options
    stream_idle_timeout = _parse_model_runner_positive_float(
        cast(ModelRunnerRuntimeOverrideValue, model_config.get("stream_idle_timeout")),
        field_name="stream_idle_timeout",
    )
    stream_idle_heartbeat_sec = _parse_model_runner_positive_float(
        cast(ModelRunnerRuntimeOverrideValue, model_config.get("stream_idle_heartbeat_sec")),
        field_name="stream_idle_heartbeat_sec",
    )
    next_runner_running_config = replace(
        runner_running_config,
        stream_idle_timeout=(
            stream_idle_timeout
            if stream_idle_timeout is not None
            else runner_running_config.stream_idle_timeout
        ),
        stream_idle_heartbeat_sec=(
            stream_idle_heartbeat_sec
            if stream_idle_heartbeat_sec is not None
            else runner_running_config.stream_idle_heartbeat_sec
        ),
    )
    if next_runner_running_config == runner_running_config:
        return resolved_execution_options
    return replace(
        resolved_execution_options,
        runner_running_config=next_runner_running_config,
    )


def _resolve_trace_output_dir(path_value: str | Path, *, workspace_dir: Path) -> Path:
    """解析 trace 输出目录。"""

    expanded = Path(path_value).expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (workspace_dir / expanded).resolve()


def build_base_execution_options(
    *,
    workspace_dir: Path,
    run_config: dict[str, Any],
) -> ResolvedExecutionOptions:
    """根据 `run.json` 构建基础运行选项。"""

    merged_runner = _merge_section(asdict(_DEFAULT_RUN_CONFIG.runner_running_config), run_config.get("runner_running_config"))
    merged_agent = _merge_section(asdict(_DEFAULT_RUN_CONFIG.agent_running_config), run_config.get("agent_running_config"))
    merged_doc_limits = _merge_section(asdict(_DEFAULT_RUN_CONFIG.doc_tool_limits), run_config.get("doc_tool_limits"))
    merged_fins_limits = _merge_section(asdict(_DEFAULT_RUN_CONFIG.fins_tool_limits), run_config.get("fins_tool_limits"))
    merged_web = _merge_section(asdict(_DEFAULT_RUN_CONFIG.web_tools_config), run_config.get("web_tools_config"))
    merged_web["playwright_storage_state_dir"] = _resolve_optional_workspace_path(
        merged_web.get("playwright_storage_state_dir"),
        workspace_dir=workspace_dir,
    )
    merged_trace = _merge_section(asdict(_DEFAULT_RUN_CONFIG.tool_trace_config), run_config.get("tool_trace_config"))
    merged_conversation_memory = _merge_conversation_memory_config(run_config.get("conversation_memory"))
    trace_output_dir = _resolve_trace_output_dir(str(merged_trace["output_dir"]), workspace_dir=workspace_dir)

    return ResolvedExecutionOptions(
        model_name="",
        temperature=None,
        runner_running_config=OpenAIRunnerRuntimeConfig(**merged_runner),
        agent_running_config=AgentRuntimeConfig(**merged_agent),
        toolset_configs=build_legacy_toolset_configs(
            doc_tool_limits=DocToolLimits(**merged_doc_limits),
            fins_tool_limits=FinsToolLimits(**merged_fins_limits),
            web_tools_config=WebToolsConfig(**merged_web),
        ),
        trace_settings=TraceSettings(
            enabled=bool(merged_trace["enabled"]),
            output_dir=trace_output_dir,
            max_file_bytes=int(merged_trace.get("max_file_bytes", 64 * 1024 * 1024)),
            retention_days=int(merged_trace.get("retention_days", 30)),
            compress_rolled=bool(merged_trace.get("compress_rolled", True)),
            partition_by_session=bool(merged_trace.get("partition_by_session", True)),
        ),
        conversation_memory_config=merged_conversation_memory,
        conversation_memory_settings=merged_conversation_memory.default,
    )


def merge_execution_options(
    *,
    base_options: ResolvedExecutionOptions,
    workspace_dir: Path,
    execution_options: ExecutionOptions | None,
) -> ResolvedExecutionOptions:
    """合并请求级执行覆盖参数。

    注意：
        大多数字段遵循“request 覆盖 base”的单一优先级。
        但 `web_provider` 是一个刻意保留的局部覆盖特例：
        它只覆盖 `web_tools_config.provider`，不会替换整个 `web_tools_config` 对象。
    """

    if execution_options is None:
        return base_options

    runner_running_config = base_options.runner_running_config
    agent_running_config = base_options.agent_running_config
    base_doc_tool_limits = resolve_doc_tool_limits_from_toolset_configs(base_options.toolset_configs) or DocToolLimits()
    base_fins_tool_limits = resolve_fins_tool_limits_from_toolset_configs(base_options.toolset_configs) or FinsToolLimits()
    base_web_tools_config = resolve_web_tools_config_from_toolset_configs(base_options.toolset_configs) or WebToolsConfig()
    request_doc_tool_limits = (
        _resolve_doc_tool_limits_from_toolset_configs(execution_options.toolset_configs)
        or execution_options.doc_tool_limits
    )
    request_fins_tool_limits = (
        _resolve_fins_tool_limits_from_toolset_configs(execution_options.toolset_configs)
        or execution_options.fins_tool_limits
    )
    request_web_tools_config = (
        _resolve_web_tools_config_from_toolset_configs(execution_options.toolset_configs)
        or execution_options.web_tools_config
    )

    runner_kwargs = dict(vars(runner_running_config))
    agent_kwargs = dict(vars(agent_running_config))

    if execution_options.debug_sse:
        runner_kwargs["debug_sse"] = True
    if execution_options.debug_tool_delta:
        runner_kwargs["debug_tool_delta"] = True
    if execution_options.debug_sse_sample_rate is not None:
        runner_kwargs["debug_sse_sample_rate"] = execution_options.debug_sse_sample_rate
    if execution_options.debug_sse_throttle_sec is not None:
        runner_kwargs["debug_sse_throttle_sec"] = execution_options.debug_sse_throttle_sec
    if execution_options.tool_timeout_seconds is not None:
        runner_kwargs["tool_timeout_seconds"] = execution_options.tool_timeout_seconds

    if execution_options.max_iterations is not None:
        agent_kwargs["max_iterations"] = execution_options.max_iterations
    if execution_options.fallback_mode is not None:
        agent_kwargs["fallback_mode"] = execution_options.fallback_mode
    if execution_options.fallback_prompt is not None:
        agent_kwargs["fallback_prompt"] = execution_options.fallback_prompt
    if execution_options.max_consecutive_failed_tool_batches is not None:
        agent_kwargs["max_consecutive_failed_tool_batches"] = (
            execution_options.max_consecutive_failed_tool_batches
        )
    if execution_options.max_duplicate_tool_calls is not None:
        agent_kwargs["max_duplicate_tool_calls"] = execution_options.max_duplicate_tool_calls
    if execution_options.duplicate_tool_hint_prompt is not None:
        agent_kwargs["duplicate_tool_hint_prompt"] = execution_options.duplicate_tool_hint_prompt

    doc_tool_limits_override = _resolve_toolset_override_payload(
        execution_options.toolset_config_overrides,
        toolset_name="doc",
    )
    fins_tool_limits_override = _resolve_toolset_override_payload(
        execution_options.toolset_config_overrides,
        toolset_name="fins",
    )
    web_tools_config_override = _resolve_toolset_override_payload(
        execution_options.toolset_config_overrides,
        toolset_name="web",
    )

    doc_tool_limits = request_doc_tool_limits or _build_doc_tool_limits_from_override(
        doc_tool_limits_override,
        base=base_doc_tool_limits,
    )
    fins_tool_limits = request_fins_tool_limits or _build_fins_tool_limits_from_override(
        fins_tool_limits_override,
        base=base_fins_tool_limits,
    )
    web_tools_config = request_web_tools_config or _build_web_tools_config_from_override(
        web_tools_config_override,
        base=base_web_tools_config,
    ) or base_web_tools_config
    if execution_options.web_provider is not None:
        # `web_provider` 只允许局部覆盖 provider。
        # 这里不能顺手把整个 web config 重建成“只保留 provider 的新对象”，
        # 否则会把 run.json 里其他联网参数一并抹掉。
        web_tools_config = WebToolsConfig(
            provider=execution_options.web_provider,
            request_timeout_seconds=web_tools_config.request_timeout_seconds,
            max_search_results=web_tools_config.max_search_results,
            fetch_truncate_chars=web_tools_config.fetch_truncate_chars,
            allow_private_network_url=web_tools_config.allow_private_network_url,
            playwright_channel=web_tools_config.playwright_channel,
            playwright_storage_state_dir=web_tools_config.playwright_storage_state_dir,
        )

    trace_enabled = base_options.trace_settings.enabled
    if execution_options.trace_enabled is not None:
        trace_enabled = bool(execution_options.trace_enabled)
    trace_output_dir = base_options.trace_settings.output_dir
    if execution_options.trace_output_dir is not None:
        trace_output_dir = _resolve_trace_output_dir(execution_options.trace_output_dir, workspace_dir=workspace_dir)
    trace_settings = TraceSettings(
        enabled=trace_enabled,
        output_dir=trace_output_dir,
        max_file_bytes=base_options.trace_settings.max_file_bytes,
        retention_days=base_options.trace_settings.retention_days,
        compress_rolled=base_options.trace_settings.compress_rolled,
        partition_by_session=base_options.trace_settings.partition_by_session,
    )

    model_name = execution_options.model_name.strip() if isinstance(execution_options.model_name, str) else ""
    if not model_name:
        model_name = base_options.model_name
    temperature = base_options.temperature
    if execution_options.temperature is not None:
        temperature = normalize_temperature(execution_options.temperature, field_name="execution_options.temperature")

    merged_toolset_configs = _merge_toolset_configs(base_options.toolset_configs, execution_options.toolset_configs)
    merged_toolset_configs = _merge_toolset_configs(
        merged_toolset_configs,
        build_legacy_toolset_configs(
            doc_tool_limits=doc_tool_limits or base_doc_tool_limits,
            fins_tool_limits=fins_tool_limits or base_fins_tool_limits,
            web_tools_config=web_tools_config,
        ),
    )

    return ResolvedExecutionOptions(
        model_name=model_name,
        temperature=temperature,
        runner_running_config=type(runner_running_config)(**runner_kwargs),
        agent_running_config=AgentRuntimeConfig(**agent_kwargs),
        toolset_configs=merged_toolset_configs,
        trace_settings=trace_settings,
        conversation_memory_config=base_options.conversation_memory_config,
        conversation_memory_settings=base_options.conversation_memory_settings,
    )


def resolve_scene_execution_options(
    *,
    base_execution_options: ResolvedExecutionOptions,
    workspace_dir: Path,
    execution_options: ExecutionOptions | None,
    default_model_name: str,
    allowed_model_names: tuple[str, ...],
    scene_agent_max_iterations: int | None,
    scene_agent_max_consecutive_failed_tool_batches: int | None,
    scene_runner_tool_timeout_seconds: float | None,
    scene_name: str,
) -> ResolvedExecutionOptions:
    """按统一规则解析 scene 级执行选项。

    Args:
        base_execution_options: 基础执行选项。
        workspace_dir: 工作区根目录。
        execution_options: 请求级显式覆盖参数。
        default_model_name: scene 默认模型名。
        allowed_model_names: scene 允许的模型列表。
        scene_agent_max_iterations: scene manifest 声明的 Agent 迭代上限。
        scene_agent_max_consecutive_failed_tool_batches: scene manifest 声明的连续失败工具批次上限。
        scene_runner_tool_timeout_seconds: scene manifest 声明的 runner 工具超时秒数。
        scene_name: scene 名称，仅用于错误提示。

    Returns:
        解析后的 scene 级执行选项。

    Raises:
        ValueError: 当模型不在 scene allowlist 中时抛出。
    """

    base_options_without_model = replace(base_execution_options, model_name="")
    resolved_execution_options = merge_execution_options(
        base_options=base_options_without_model,
        workspace_dir=workspace_dir,
        execution_options=execution_options,
    )
    if not resolved_execution_options.model_name:
        resolved_execution_options = replace(
            resolved_execution_options,
            model_name=default_model_name,
        )

    agent_running_config = resolved_execution_options.agent_running_config
    if scene_agent_max_iterations is not None and (
        execution_options is None or execution_options.max_iterations is None
    ):
        agent_running_config = replace(
            agent_running_config,
            max_iterations=scene_agent_max_iterations,
        )
    if scene_agent_max_consecutive_failed_tool_batches is not None and (
        execution_options is None or execution_options.max_consecutive_failed_tool_batches is None
    ):
        agent_running_config = replace(
            agent_running_config,
            max_consecutive_failed_tool_batches=scene_agent_max_consecutive_failed_tool_batches,
        )
    runner_running_config = resolved_execution_options.runner_running_config
    if isinstance(runner_running_config, OpenAIRunnerRuntimeConfig) and (
        scene_runner_tool_timeout_seconds is not None
        and (execution_options is None or execution_options.tool_timeout_seconds is None)
    ):
        runner_running_config = replace(
            runner_running_config,
            tool_timeout_seconds=scene_runner_tool_timeout_seconds,
        )
    if agent_running_config != resolved_execution_options.agent_running_config:
        resolved_execution_options = replace(
            resolved_execution_options,
            agent_running_config=agent_running_config,
        )
    if runner_running_config != resolved_execution_options.runner_running_config:
        resolved_execution_options = replace(
            resolved_execution_options,
            runner_running_config=runner_running_config,
        )

    if resolved_execution_options.model_name not in allowed_model_names:
        raise ValueError(
            "当前执行请求的模型不在 scene allowlist 中，"
            f"scene={scene_name}, model={resolved_execution_options.model_name}, "
            f"allowed_names={list(allowed_model_names)}"
        )
    return resolved_execution_options


__all__ = [
    "ConversationMemoryConfig",
    "ConversationMemorySettings",
    "ExecutionOptions",
    "ExecutionOptionsOverridePayload",
    "ExecutionOptionsOverrideValue",
    "ExecutionOptionsSnapshot",
    "ExecutionOptionsSnapshotValue",
    "ModelRunnerRuntimeOverrideValue",
    "ResolvedExecutionOptions",
    "TraceSettings",
    "apply_model_runner_runtime_overrides",
    "build_base_execution_options",
    "deserialize_execution_options_snapshot",
    "merge_execution_options",
    "normalize_temperature",
    "resolve_scene_temperature",
    "resolve_scene_execution_options",
    "resolve_conversation_memory_settings",
    "serialize_execution_options_snapshot",
]
