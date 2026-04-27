"""Agent 执行契约的序列化与反序列化。

从 ``agent_execution`` 拆分出的纯数据转换层：把 ``ExecutionContract``
与 JSON 兼容的快照对象互相转换，用于跨进程 / 磁盘持久化。
本模块不承载业务语义，只做结构层面的归一化。
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, is_dataclass
from pathlib import Path
from typing import Literal, Mapping, cast

from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedInfrastructureSpec,
    AcceptedModelSpec,
    AcceptedRuntimeSpec,
    AcceptedToolConfigSpec,
    ExecutionContract,
    ExecutionContractSnapshot,
    ExecutionContractSnapshotValue,
    ExecutionDocPermissions,
    ExecutionHostPolicy,
    ExecutionMessageInputs,
    ExecutionPermissions,
    ExecutionWebPermissions,
    ScenePreparationSpec,
)
from dayu.contracts.execution_metadata import normalize_execution_delivery_context
from dayu.contracts.host_execution import ConcurrencyAcquirePolicy
from dayu.contracts.execution_options import (
    ConversationMemorySettings,
    ExecutionOptionsSnapshot,
    ExecutionOptionsSnapshotValue,
    TraceSettings,
    deserialize_execution_options_snapshot,
    serialize_execution_options_snapshot,
)
from dayu.contracts.runtime_config_snapshot import AgentRunningConfigSnapshot, RunnerRunningConfigSnapshot
from dayu.contracts.toolset_config import (
    ToolsetConfigSnapshot,
    ToolsetConfigValue,
    build_toolset_config_snapshot,
    normalize_toolset_configs,
    replace_toolset_config,
    serialize_toolset_config_payload_value,
)


def _trace_settings_default_int(field_name: str) -> int:
    """读取 `TraceSettings` 指定整数字段的 dataclass 默认值。

    Args:
        field_name: `TraceSettings` 中的目标字段名。

    Returns:
        对应字段的整数默认值。

    Raises:
        KeyError: 字段不存在时抛出。
        TypeError: 字段默认值不是整数时抛出。
        ValueError: 字段未声明默认值时抛出。
    """

    field_info = TraceSettings.__dataclass_fields__[field_name]
    default_value = field_info.default
    if default_value is MISSING:
        raise ValueError(f"TraceSettings.{field_name} 未声明默认值")
    if isinstance(default_value, bool) or not isinstance(default_value, int):
        raise TypeError(f"TraceSettings.{field_name} 默认值不是整数")
    return default_value


def _trace_settings_default_bool(field_name: str) -> bool:
    """读取 `TraceSettings` 指定布尔字段的 dataclass 默认值。

    Args:
        field_name: `TraceSettings` 中的目标字段名。

    Returns:
        对应字段的布尔默认值。

    Raises:
        KeyError: 字段不存在时抛出。
        TypeError: 字段默认值不是布尔值时抛出。
        ValueError: 字段未声明默认值时抛出。
    """

    field_info = TraceSettings.__dataclass_fields__[field_name]
    default_value = field_info.default
    if default_value is MISSING:
        raise ValueError(f"TraceSettings.{field_name} 未声明默认值")
    if not isinstance(default_value, bool):
        raise TypeError(f"TraceSettings.{field_name} 默认值不是布尔值")
    return default_value


def _normalize_snapshot_value(value: object) -> ExecutionContractSnapshotValue:
    """把对象标准化为 JSON 兼容快照值。"""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {
            str(key): _normalize_snapshot_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_normalize_snapshot_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return cast(
            dict[str, ExecutionContractSnapshotValue],
            _normalize_snapshot_value(asdict(value)),
        )
    raise ValueError(f"不支持序列化为 ExecutionContract 快照的值类型: {type(value).__name__}")


def _snapshot_optional_object(
    value: ExecutionContractSnapshotValue | None,
) -> dict[str, ExecutionContractSnapshotValue] | None:
    """从快照值中读取对象。"""

    if not isinstance(value, dict):
        return None
    return {str(key): item for key, item in value.items()}


def _snapshot_optional_list(
    value: ExecutionContractSnapshotValue | None,
) -> list[ExecutionContractSnapshotValue] | None:
    """从快照值中读取可选数组。"""

    if not isinstance(value, list):
        return None
    return list(value)


def _snapshot_required_object(
    value: ExecutionContractSnapshotValue | None,
    *,
    field_name: str,
) -> dict[str, ExecutionContractSnapshotValue]:
    """从快照值中读取必填对象。"""

    payload = _snapshot_optional_object(value)
    if payload is None:
        raise ValueError(f"{field_name} 必须是 JSON object")
    return payload


def _snapshot_optional_str(value: ExecutionContractSnapshotValue | None) -> str | None:
    """从快照值中读取可选字符串。"""

    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _snapshot_required_str(
    value: ExecutionContractSnapshotValue | None,
    *,
    field_name: str,
) -> str:
    """从快照值中读取必填字符串。"""

    normalized = _snapshot_optional_str(value)
    if normalized is None:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def _snapshot_optional_int(value: ExecutionContractSnapshotValue | None) -> int | None:
    """从快照值中读取可选整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _snapshot_optional_float(value: ExecutionContractSnapshotValue | None) -> float | None:
    """从快照值中读取可选浮点数。

    Args:
        value: 原始快照值。

    Returns:
        浮点数或 ``None``。

    Raises:
        ValueError: 当值类型不是 ``None`` / ``int`` / ``float`` 时抛出。``bool``
            属于 ``int`` 的子类但语义上不是数字，混入 float 快照通常意味着上游
            序列化 bug，直接 raise 以便尽早暴露（与 ``contracts.execution_options``
            / ``execution.options`` 中的同名函数保持一致的严格策略）。
    """

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("execution contract snapshot value must be number")
    if isinstance(value, int | float):
        return float(value)
    raise ValueError("execution contract snapshot value must be number")


def _snapshot_optional_bool(value: ExecutionContractSnapshotValue | None) -> bool | None:
    """从快照值中读取可选布尔值。"""

    if not isinstance(value, bool):
        return None
    return value


def _snapshot_str_tuple(value: ExecutionContractSnapshotValue | None) -> tuple[str, ...]:
    """从快照值中读取字符串元组。"""

    if not isinstance(value, list):
        return ()
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized = item.strip()
            if normalized:
                result.append(normalized)
    return tuple(result)


def _snapshot_string_dict(value: ExecutionContractSnapshotValue | None) -> dict[str, str]:
    """从快照值中读取字符串字典。"""

    payload = _snapshot_optional_object(value)
    if payload is None:
        return {}
    result: dict[str, str] = {}
    for key, item in payload.items():
        if isinstance(item, str):
            result[str(key)] = item
    return result


def _snapshot_int_or_default(
    value: ExecutionContractSnapshotValue | None,
    *,
    default: int,
) -> int:
    """从契约快照值读取整数，缺失时回退到默认值。

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
    value: ExecutionContractSnapshotValue | None,
    *,
    default: float,
) -> float:
    """从契约快照值读取浮点数，缺失时回退到默认值。

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


def _snapshot_str_or_default(
    value: ExecutionContractSnapshotValue | None,
    *,
    default: str,
) -> str:
    """从契约快照值读取字符串，缺失时回退到默认值。

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


def _coerce_execution_options_snapshot_value(
    value: ExecutionContractSnapshotValue | None,
) -> ExecutionOptionsSnapshotValue:
    """把契约快照值收窄为执行参数快照值。

    Args:
        value: 契约层快照值。

    Returns:
        执行参数层允许的快照值。

    Raises:
        无。
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_coerce_execution_options_snapshot_value(item) for item in value]
    return {
        str(key): _coerce_execution_options_snapshot_value(item)
        for key, item in value.items()
    }


def _coerce_execution_options_snapshot(
    payload: Mapping[str, ExecutionContractSnapshotValue],
) -> ExecutionOptionsSnapshot:
    """把契约层 execution options 快照收窄为执行参数快照。

    Args:
        payload: 契约层 execution options 对象。

    Returns:
        执行参数快照对象。

    Raises:
        ValueError: 当对象包含执行参数快照不支持的值时抛出。
    """

    return {
        str(key): _coerce_execution_options_snapshot_value(item)
        for key, item in payload.items()
    }


def _coerce_toolset_config_payload(
    payload: Mapping[str, ExecutionContractSnapshotValue],
) -> dict[str, ToolsetConfigValue]:
    """把契约层快照对象收窄为 toolset 通用 payload。"""

    return {
        str(key): cast(ToolsetConfigValue, serialize_toolset_config_payload_value(cast(ToolsetConfigValue, value)))
        for key, value in payload.items()
    }


def _build_toolset_config_from_snapshot(
    payload: Mapping[str, ExecutionContractSnapshotValue],
) -> ToolsetConfigSnapshot:
    """从契约快照对象恢复单个 toolset 配置快照。"""

    toolset_name = _snapshot_optional_str(payload.get("toolset_name"))
    if toolset_name is None:
        raise ValueError("tools.toolset_configs[].toolset_name 不能为空")
    version = _snapshot_optional_str(payload.get("version")) or "1"
    config_payload = _snapshot_optional_object(payload.get("payload")) or {}
    snapshot = build_toolset_config_snapshot(
        toolset_name,
        _coerce_toolset_config_payload(config_payload),
        version=version,
    )
    if snapshot is None:
        raise ValueError("tools.toolset_configs[] 必须包含 payload")
    return snapshot


def _build_toolset_configs_from_snapshot(
    toolset_configs_payload: list[ExecutionContractSnapshotValue] | None,
) -> tuple[ToolsetConfigSnapshot, ...]:
    """从契约快照恢复通用 toolset 配置序列。"""

    snapshots: tuple[ToolsetConfigSnapshot, ...] = ()
    for item in toolset_configs_payload or []:
        item_payload = _snapshot_optional_object(item)
        if item_payload is None:
            raise ValueError("tools.toolset_configs[] 必须是 JSON object")
        snapshots = replace_toolset_config(snapshots, _build_toolset_config_from_snapshot(item_payload))
    return normalize_toolset_configs(snapshots)


def _build_conversation_memory_settings_from_snapshot(
    payload: Mapping[str, ExecutionContractSnapshotValue] | None,
) -> ConversationMemorySettings | None:
    """从契约快照对象恢复会话记忆配置。

    Args:
        payload: 会话记忆配置快照对象。

    Returns:
        恢复后的会话记忆配置；输入为空时返回 ``None``。

    Raises:
        无。
    """

    if payload is None:
        return None
    defaults = ConversationMemorySettings()
    return ConversationMemorySettings(
        memory_token_budget_ratio=_snapshot_float_or_default(
            payload.get("memory_token_budget_ratio"),
            default=defaults.memory_token_budget_ratio,
        ),
        memory_token_budget_floor=_snapshot_int_or_default(
            payload.get("memory_token_budget_floor"),
            default=defaults.memory_token_budget_floor,
        ),
        memory_token_budget_cap=_snapshot_int_or_default(
            payload.get("memory_token_budget_cap"),
            default=defaults.memory_token_budget_cap,
        ),
        recent_turns_floor=_snapshot_int_or_default(
            payload.get("recent_turns_floor"),
            default=defaults.recent_turns_floor,
        ),
        compaction_trigger_context_ratio=_snapshot_float_or_default(
            payload.get("compaction_trigger_context_ratio"),
            default=defaults.compaction_trigger_context_ratio,
        ),
        compaction_tail_preserve_turns=_snapshot_int_or_default(
            payload.get("compaction_tail_preserve_turns"),
            default=defaults.compaction_tail_preserve_turns,
        ),
        compaction_context_episode_window=_snapshot_int_or_default(
            payload.get("compaction_context_episode_window"),
            default=defaults.compaction_context_episode_window,
        ),
        compaction_scene_name=_snapshot_str_or_default(
            payload.get("compaction_scene_name"),
            default=defaults.compaction_scene_name,
        ),
    )


def _serialize_accepted_execution_spec(
    spec: AcceptedExecutionSpec,
) -> dict[str, ExecutionContractSnapshotValue]:
    """序列化已接受执行规格。"""

    return {
        "model": cast(
            dict[str, ExecutionContractSnapshotValue],
            _normalize_snapshot_value(spec.model),
        ),
        "runtime": cast(
            dict[str, ExecutionContractSnapshotValue],
            _normalize_snapshot_value(spec.runtime),
        ),
        "tools": cast(
            dict[str, ExecutionContractSnapshotValue],
            _normalize_snapshot_value(spec.tools),
        ),
        "infrastructure": cast(
            dict[str, ExecutionContractSnapshotValue],
            _normalize_snapshot_value(spec.infrastructure),
        ),
    }


def _deserialize_accepted_execution_spec(
    payload: Mapping[str, ExecutionContractSnapshotValue],
) -> AcceptedExecutionSpec:
    """反序列化已接受执行规格。"""

    model_payload = _snapshot_required_object(payload.get("model"), field_name="model")
    runtime_payload = _snapshot_optional_object(payload.get("runtime")) or {}
    tools_payload = _snapshot_optional_object(payload.get("tools")) or {}
    infrastructure_payload = _snapshot_optional_object(payload.get("infrastructure")) or {}
    toolset_configs_payload = _snapshot_optional_list(tools_payload.get("toolset_configs"))
    trace_settings_payload = _snapshot_optional_object(infrastructure_payload.get("trace_settings"))
    conversation_memory_settings_payload = _snapshot_optional_object(
        infrastructure_payload.get("conversation_memory_settings")
    )
    trace_output_dir_raw = (
        _snapshot_optional_str(trace_settings_payload.get("output_dir"))
        if trace_settings_payload is not None
        else None
    )
    trace_output_dir = Path(trace_output_dir_raw) if trace_output_dir_raw is not None else None
    trace_enabled = (
        _snapshot_optional_bool(trace_settings_payload.get("enabled"))
        if trace_settings_payload is not None
        else None
    )
    trace_max_file_bytes = (
        _snapshot_optional_int(trace_settings_payload.get("max_file_bytes"))
        if trace_settings_payload is not None
        else None
    )
    trace_retention_days = (
        _snapshot_optional_int(trace_settings_payload.get("retention_days"))
        if trace_settings_payload is not None
        else None
    )
    trace_compress_rolled = (
        _snapshot_optional_bool(trace_settings_payload.get("compress_rolled"))
        if trace_settings_payload is not None
        else None
    )
    trace_partition_by_session = (
        _snapshot_optional_bool(trace_settings_payload.get("partition_by_session"))
        if trace_settings_payload is not None
        else None
    )
    return AcceptedExecutionSpec(
        model=AcceptedModelSpec(
            model_name=_snapshot_optional_str(model_payload.get("model_name")) or "",
            temperature=_snapshot_optional_float(model_payload.get("temperature")),
        ),
        runtime=AcceptedRuntimeSpec(
            runner_running_config=cast(
                RunnerRunningConfigSnapshot,
                dict(_snapshot_optional_object(runtime_payload.get("runner_running_config")) or {}),
            ),
            agent_running_config=cast(
                AgentRunningConfigSnapshot,
                dict(_snapshot_optional_object(runtime_payload.get("agent_running_config")) or {}),
            ),
        ),
        tools=AcceptedToolConfigSpec(
            toolset_configs=_build_toolset_configs_from_snapshot(toolset_configs_payload),
        ),
        infrastructure=AcceptedInfrastructureSpec(
            trace_settings=(
                TraceSettings(
                    enabled=bool(trace_enabled),
                    output_dir=trace_output_dir,
                    max_file_bytes=(
                        trace_max_file_bytes
                        if trace_max_file_bytes is not None
                        else _trace_settings_default_int("max_file_bytes")
                    ),
                    retention_days=(
                        trace_retention_days
                        if trace_retention_days is not None
                        else _trace_settings_default_int("retention_days")
                    ),
                    compress_rolled=(
                        trace_compress_rolled
                        if trace_compress_rolled is not None
                        else _trace_settings_default_bool("compress_rolled")
                    ),
                    partition_by_session=(
                        trace_partition_by_session
                        if trace_partition_by_session is not None
                        else _trace_settings_default_bool("partition_by_session")
                    ),
                )
                if trace_settings_payload is not None and trace_output_dir is not None
                else None
            ),
            conversation_memory_settings=_build_conversation_memory_settings_from_snapshot(
                conversation_memory_settings_payload
            ),
        ),
    )


def serialize_execution_contract_snapshot(
    execution_contract: ExecutionContract,
) -> ExecutionContractSnapshot:
    """把 ExecutionContract 序列化为可持久化快照。"""

    snapshot_payload: ExecutionContractSnapshot = {
        "service_name": execution_contract.service_name,
        "scene_name": execution_contract.scene_name,
        "host_policy": {
            "session_key": execution_contract.host_policy.session_key,
            "business_concurrency_lane": execution_contract.host_policy.business_concurrency_lane,
            "concurrency_acquire_policy": {
                "mode": execution_contract.host_policy.concurrency_acquire_policy.mode,
                "timeout_seconds": execution_contract.host_policy.concurrency_acquire_policy.timeout_seconds,
            },
            "timeout_ms": execution_contract.host_policy.timeout_ms,
            "resumable": execution_contract.host_policy.resumable,
        },
        "preparation_spec": {
            "selected_toolsets": cast(
                list[ExecutionContractSnapshotValue],
                _normalize_snapshot_value(list(execution_contract.preparation_spec.selected_toolsets)),
            ),
            "execution_permissions": cast(
                dict[str, ExecutionContractSnapshotValue],
                _normalize_snapshot_value(execution_contract.preparation_spec.execution_permissions),
            ),
            "prompt_contributions": cast(
                dict[str, ExecutionContractSnapshotValue],
                _normalize_snapshot_value(dict(execution_contract.preparation_spec.prompt_contributions)),
            ),
        },
        "message_inputs": {
            "user_message": execution_contract.message_inputs.user_message,
        },
        "accepted_execution_spec": _serialize_accepted_execution_spec(
            execution_contract.accepted_execution_spec
        ),
        "execution_options": cast(
            dict[str, ExecutionContractSnapshotValue],
            _normalize_snapshot_value(
                serialize_execution_options_snapshot(execution_contract.execution_options)
            ),
        ),
        "metadata": cast(
            dict[str, ExecutionContractSnapshotValue],
            _normalize_snapshot_value(dict(execution_contract.metadata)),
        ),
    }
    return snapshot_payload


def deserialize_execution_contract_snapshot(
    snapshot: Mapping[str, ExecutionContractSnapshotValue],
) -> ExecutionContract:
    """把 ExecutionContract 快照恢复为契约对象。"""

    host_policy = _snapshot_required_object(snapshot.get("host_policy"), field_name="host_policy")
    preparation_spec = _snapshot_required_object(snapshot.get("preparation_spec"), field_name="preparation_spec")
    message_inputs = _snapshot_required_object(snapshot.get("message_inputs"), field_name="message_inputs")
    accepted_execution_spec_payload = _snapshot_required_object(
        snapshot.get("accepted_execution_spec"),
        field_name="accepted_execution_spec",
    )
    execution_permissions_payload = _snapshot_required_object(
        preparation_spec.get("execution_permissions"),
        field_name="preparation_spec.execution_permissions",
    )
    web_permissions_payload = _snapshot_optional_object(execution_permissions_payload.get("web")) or {}
    doc_permissions_payload = _snapshot_optional_object(execution_permissions_payload.get("doc")) or {}
    execution_options_payload = _snapshot_optional_object(snapshot.get("execution_options")) or {}
    metadata_payload = _snapshot_optional_object(snapshot.get("metadata")) or {}
    concurrency_acquire_policy_payload = _snapshot_required_object(
        host_policy.get("concurrency_acquire_policy"),
        field_name="host_policy.concurrency_acquire_policy",
    )
    return ExecutionContract(
        service_name=_snapshot_optional_str(snapshot.get("service_name")) or "",
        scene_name=_snapshot_optional_str(snapshot.get("scene_name")) or "",
        host_policy=ExecutionHostPolicy(
            session_key=_snapshot_optional_str(host_policy.get("session_key")),
            business_concurrency_lane=_snapshot_optional_str(host_policy.get("business_concurrency_lane")),
            concurrency_acquire_policy=_deserialize_concurrency_acquire_policy(
                concurrency_acquire_policy_payload
            ),
            timeout_ms=_snapshot_optional_int(host_policy.get("timeout_ms")),
            resumable=bool(_snapshot_optional_bool(host_policy.get("resumable"))),
        ),
        preparation_spec=ScenePreparationSpec(
            selected_toolsets=_snapshot_str_tuple(preparation_spec.get("selected_toolsets")),
            execution_permissions=ExecutionPermissions(
                web=ExecutionWebPermissions(
                    allow_private_network_url=bool(
                        _snapshot_optional_bool(web_permissions_payload.get("allow_private_network_url"))
                    ),
                ),
                doc=ExecutionDocPermissions(
                    allowed_read_paths=_snapshot_str_tuple(doc_permissions_payload.get("allowed_read_paths")),
                    allow_file_write=bool(_snapshot_optional_bool(doc_permissions_payload.get("allow_file_write"))),
                    allowed_write_paths=_snapshot_str_tuple(doc_permissions_payload.get("allowed_write_paths")),
                ),
            ),
            prompt_contributions=_snapshot_string_dict(preparation_spec.get("prompt_contributions")),
        ),
        message_inputs=ExecutionMessageInputs(
            user_message=_snapshot_optional_str(message_inputs.get("user_message")),
        ),
        accepted_execution_spec=_deserialize_accepted_execution_spec(accepted_execution_spec_payload),
        execution_options=deserialize_execution_options_snapshot(
            _coerce_execution_options_snapshot(execution_options_payload)
        ),
        metadata=normalize_execution_delivery_context(metadata_payload),
    )


def _deserialize_concurrency_acquire_policy(
    payload: Mapping[str, ExecutionContractSnapshotValue],
) -> ConcurrencyAcquirePolicy:
    """把序列化的并发等待策略恢复为强类型对象。"""

    mode = cast(
        Literal["host_default", "timeout", "unbounded"],
        _snapshot_required_str(
            payload.get("mode"),
            field_name="host_policy.concurrency_acquire_policy.mode",
        ),
    )
    return ConcurrencyAcquirePolicy(
        mode=mode,
        timeout_seconds=_snapshot_optional_float(payload.get("timeout_seconds")),
    )


__all__ = [
    "deserialize_execution_contract_snapshot",
    "serialize_execution_contract_snapshot",
]
