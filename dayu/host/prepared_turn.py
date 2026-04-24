"""Host 内部 pending turn 恢复快照。

该模块定义 Host 在 pending conversation turn 生命周期中使用的两类恢复真源：

- accepted snapshot：Host 已接受 resumable turn，但 scene preparation 还未成功完成。
- prepared snapshot：Host 已完成 scene preparation，可直接恢复 Agent 输入。

两类快照都属于 Host 内部机械恢复真源，不向 Service / UI 暴露。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

from dayu.contracts.agent_execution import (
    AcceptedInfrastructureSpec,
    AcceptedExecutionSpec,
    AcceptedModelSpec,
    AcceptedRuntimeSpec,
    AcceptedToolConfigSpec,
    AgentCreateArgs,
    ExecutionDocPermissions,
    ExecutionHostPolicy,
    ExecutionMessageInputs,
    ExecutionPermissions,
    ExecutionWebPermissions,
    ScenePreparationSpec,
)
from dayu.contracts.agent_types import AgentMessage, AgentTraceIdentity
from dayu.contracts.execution_metadata import ExecutionDeliveryContext, normalize_execution_delivery_context
from dayu.contracts.model_config import RunnerParams
from dayu.contracts.toolset_config import (
    ToolsetConfigSnapshot,
    ToolsetConfigValue,
    build_toolset_config_snapshot,
    normalize_toolset_configs,
    replace_toolset_config,
    serialize_toolset_config_payload_value,
)
from dayu.execution.options import (
    ConversationMemorySettings,
    ExecutionOptions,
    ExecutionOptionsSnapshot,
    ExecutionOptionsSnapshotValue,
    TraceSettings,
    deserialize_execution_options_snapshot,
    serialize_execution_options_snapshot,
)
from dayu.contracts.runtime_config_snapshot import AgentRunningConfigSnapshot, RunnerRunningConfigSnapshot
from dayu.host.conversation_store import ConversationTranscript

PendingTurnSnapshotScalar: TypeAlias = str | int | float | bool | None
PendingTurnSnapshotValue: TypeAlias = (
    PendingTurnSnapshotScalar | list["PendingTurnSnapshotValue"] | dict[str, "PendingTurnSnapshotValue"]
)
PendingTurnRawValue: TypeAlias = PendingTurnSnapshotValue | None


@dataclass(frozen=True)
class AcceptedAgentTurnSnapshot:
    """Host 已接受但尚未完成 scene preparation 的恢复快照。"""

    service_name: str
    scene_name: str
    host_policy: ExecutionHostPolicy
    preparation_spec: ScenePreparationSpec
    message_inputs: ExecutionMessageInputs
    accepted_execution_spec: AcceptedExecutionSpec
    execution_options: ExecutionOptions | None = None
    metadata: ExecutionDeliveryContext | None = None


@dataclass(frozen=True)
class PreparedConversationSessionSnapshot:
    """prepared turn 的会话持久化快照。"""

    session_id: str
    user_message: str
    transcript: ConversationTranscript


@dataclass(frozen=True)
class PreparedAgentTurnSnapshot:
    """Host 已完成 scene preparation 后的可恢复快照。"""

    service_name: str
    scene_name: str
    metadata: ExecutionDeliveryContext
    business_concurrency_lane: str | None
    timeout_ms: int | None
    resumable: bool
    system_prompt: str
    messages: list[AgentMessage]
    agent_create_args: AgentCreateArgs
    selected_toolsets: tuple[str, ...]
    execution_permissions: ExecutionPermissions
    toolset_configs: tuple[ToolsetConfigSnapshot, ...]
    trace_settings: TraceSettings | None
    conversation_memory_settings: ConversationMemorySettings
    trace_identity: AgentTraceIdentity | None = None
    conversation_session: PreparedConversationSessionSnapshot | None = None

    def __post_init__(self) -> None:
        """规范化 prepared turn 中的 toolset 配置快照。"""

        object.__setattr__(self, "toolset_configs", normalize_toolset_configs(self.toolset_configs))


def serialize_accepted_agent_turn_snapshot(
    snapshot: AcceptedAgentTurnSnapshot,
) -> dict[str, PendingTurnSnapshotValue]:
    """把 accepted snapshot 序列化为 JSON 对象。"""

    payload = {
        "service_name": snapshot.service_name,
        "scene_name": snapshot.scene_name,
        "host_policy": asdict(snapshot.host_policy),
        "preparation_spec": asdict(snapshot.preparation_spec),
        "message_inputs": asdict(snapshot.message_inputs),
        "accepted_execution_spec": asdict(snapshot.accepted_execution_spec),
        "execution_options": (
            serialize_execution_options_snapshot(snapshot.execution_options)
            if snapshot.execution_options is not None
            else None
        ),
        "metadata": normalize_execution_delivery_context(snapshot.metadata),
    }
    normalized_payload = _normalize_snapshot_value(payload)
    if not isinstance(normalized_payload, dict):
        raise ValueError("accepted snapshot 序列化结果必须是 JSON object")
    return cast(dict[str, PendingTurnSnapshotValue], normalized_payload)


def deserialize_accepted_agent_turn_snapshot(
    payload: dict[str, PendingTurnSnapshotValue],
) -> AcceptedAgentTurnSnapshot:
    """从 JSON 对象反序列化 accepted snapshot。"""

    execution_options_payload = _snapshot_optional_object(payload.get("execution_options"))
    return AcceptedAgentTurnSnapshot(
        service_name=_normalize_required_text(payload.get("service_name"), field_name="service_name"),
        scene_name=_normalize_required_text(payload.get("scene_name"), field_name="scene_name"),
        host_policy=_parse_execution_host_policy(_as_object(payload.get("host_policy"))),
        preparation_spec=_parse_scene_preparation_spec(_as_object(payload.get("preparation_spec"))),
        message_inputs=_parse_execution_message_inputs(_as_object(payload.get("message_inputs"))),
        accepted_execution_spec=_parse_accepted_execution_spec(_as_object(payload.get("accepted_execution_spec"))),
        execution_options=(
            deserialize_execution_options_snapshot(_coerce_execution_options_snapshot(execution_options_payload))
            if execution_options_payload is not None
            else None
        ),
        metadata=normalize_execution_delivery_context(_as_object(payload.get("metadata"))),
    )


def serialize_prepared_agent_turn_snapshot(
    snapshot: PreparedAgentTurnSnapshot,
) -> dict[str, PendingTurnSnapshotValue]:
    """把 prepared turn 快照序列化为 JSON 对象。"""

    payload = _normalize_snapshot_value(asdict(snapshot))
    if not isinstance(payload, dict):
        raise ValueError("prepared turn 快照序列化结果必须是 JSON object")
    return cast(dict[str, PendingTurnSnapshotValue], payload)


def deserialize_prepared_agent_turn_snapshot(
    payload: dict[str, PendingTurnSnapshotValue],
) -> PreparedAgentTurnSnapshot:
    """从 JSON 对象反序列化 prepared turn 快照。"""

    service_name = _normalize_required_text(payload.get("service_name"), field_name="service_name")
    scene_name = _normalize_required_text(payload.get("scene_name"), field_name="scene_name")
    system_prompt = str(payload.get("system_prompt") or "")
    business_concurrency_lane = _normalize_optional_text(payload.get("business_concurrency_lane"))
    timeout_ms = _coerce_optional_int(payload.get("timeout_ms"), field_name="timeout_ms")
    resumable = bool(payload.get("resumable"))
    metadata = normalize_execution_delivery_context(_as_object(payload.get("metadata")))
    messages = _parse_messages(payload.get("messages"))
    agent_create_args = _parse_agent_create_args(_as_object(payload.get("agent_create_args")))
    execution_permissions = _parse_execution_permissions(_as_object(payload.get("execution_permissions")))
    selected_toolsets = _parse_string_tuple(payload.get("selected_toolsets"))
    toolset_configs = _parse_toolset_configs(payload.get("toolset_configs"))
    trace_settings = _parse_optional_trace_settings(payload.get("trace_settings"))
    conversation_memory_settings = _parse_conversation_memory_settings(
        _as_object(payload.get("conversation_memory_settings"))
    )
    trace_identity = _parse_optional_trace_identity(payload.get("trace_identity"))
    conversation_session = _parse_optional_conversation_session(payload.get("conversation_session"))
    return PreparedAgentTurnSnapshot(
        service_name=service_name,
        scene_name=scene_name,
        metadata=metadata,
        business_concurrency_lane=business_concurrency_lane,
        timeout_ms=timeout_ms,
        resumable=resumable,
        system_prompt=system_prompt,
        messages=messages,
        agent_create_args=agent_create_args,
        selected_toolsets=selected_toolsets,
        execution_permissions=execution_permissions,
        toolset_configs=toolset_configs,
        trace_settings=trace_settings,
        conversation_memory_settings=conversation_memory_settings,
        trace_identity=trace_identity,
        conversation_session=conversation_session,
    )


def _normalize_snapshot_value(value: object) -> PendingTurnSnapshotValue:
    """把对象递归标准化为 JSON 兼容值。"""

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
        return cast(dict[str, PendingTurnSnapshotValue], _normalize_snapshot_value(asdict(value)))
    raise ValueError(f"pending turn 快照不支持值类型: {type(value).__name__}")


def _normalize_required_text(value: PendingTurnRawValue, *, field_name: str) -> str:
    """规范化必填文本字段。"""

    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} 不能为空")
    return normalized


def _normalize_optional_text(value: PendingTurnRawValue) -> str | None:
    """规范化可选文本字段。"""

    normalized = str(value or "").strip()
    return normalized or None


def _coerce_optional_int(value: PendingTurnRawValue, *, field_name: str) -> int | None:
    """规范化可选整数。"""

    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} 必须是整数或空")
    return value


def _coerce_optional_float(value: PendingTurnRawValue, *, field_name: str) -> float | None:
    """规范化可选浮点数。"""

    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} 必须是浮点数或空")
    return float(value)


def _as_object(value: PendingTurnRawValue) -> dict[str, PendingTurnSnapshotValue]:
    """把值收窄为 JSON object。"""

    if not isinstance(value, dict):
        raise ValueError("pending turn 快照字段必须是 JSON object")
    return cast(dict[str, PendingTurnSnapshotValue], value)


def _snapshot_optional_object(value: PendingTurnRawValue) -> dict[str, PendingTurnSnapshotValue] | None:
    """把值收窄为可选 JSON object。"""

    if not isinstance(value, dict):
        return None
    return cast(dict[str, PendingTurnSnapshotValue], value)


def _parse_string_tuple(value: PendingTurnRawValue) -> tuple[str, ...]:
    """解析字符串元组。"""

    if not isinstance(value, list):
        return ()
    normalized: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _parse_string_dict(value: PendingTurnRawValue) -> dict[str, str]:
    """解析字符串字典。"""

    if not isinstance(value, dict):
        return {}
    normalized: dict[str, str] = {}
    for key, item in value.items():
        key_text = str(key or "").strip()
        if not key_text or not isinstance(item, str):
            continue
        normalized[key_text] = item
    return normalized


def _parse_messages(value: PendingTurnRawValue) -> list[AgentMessage]:
    """解析送模消息列表。"""

    if not isinstance(value, list):
        raise ValueError("messages 必须是 JSON array")
    messages: list[AgentMessage] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("messages 项必须是 JSON object")
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "")
        if not role:
            raise ValueError("message.role 不能为空")
        messages.append(cast(AgentMessage, {"role": role, "content": content}))
    return messages


def _parse_execution_host_policy(value: dict[str, PendingTurnSnapshotValue]) -> ExecutionHostPolicy:
    """解析 Host 生命周期策略。"""

    return ExecutionHostPolicy(
        session_key=_normalize_optional_text(value.get("session_key")),
        business_concurrency_lane=_normalize_optional_text(value.get("business_concurrency_lane")),
        timeout_ms=_coerce_optional_int(value.get("timeout_ms"), field_name="timeout_ms"),
        resumable=bool(value.get("resumable", False)),
    )


def _parse_scene_preparation_spec(value: dict[str, PendingTurnSnapshotValue]) -> ScenePreparationSpec:
    """解析 scene preparation 说明。"""

    return ScenePreparationSpec(
        selected_toolsets=_parse_string_tuple(value.get("selected_toolsets")),
        execution_permissions=_parse_execution_permissions(_as_object(value.get("execution_permissions"))),
        prompt_contributions=_parse_string_dict(value.get("prompt_contributions")),
    )


def _parse_execution_message_inputs(value: dict[str, PendingTurnSnapshotValue]) -> ExecutionMessageInputs:
    """解析当前轮消息输入。"""

    return ExecutionMessageInputs(
        user_message=_normalize_optional_text(value.get("user_message")),
    )


def _parse_accepted_execution_spec(value: dict[str, PendingTurnSnapshotValue]) -> AcceptedExecutionSpec:
    """解析 Service 已接受的执行规格。"""

    model_payload = _as_object(value.get("model"))
    runtime_payload = _snapshot_optional_object(value.get("runtime")) or {}
    tools_payload = _snapshot_optional_object(value.get("tools")) or {}
    infrastructure_payload = _snapshot_optional_object(value.get("infrastructure")) or {}
    toolset_configs = _parse_toolset_configs(tools_payload.get("toolset_configs"))
    return AcceptedExecutionSpec(
        model=AcceptedModelSpec(
            model_name=_normalize_required_text(model_payload.get("model_name"), field_name="model.model_name"),
            temperature=_coerce_optional_float(model_payload.get("temperature"), field_name="model.temperature"),
        ),
        runtime=AcceptedRuntimeSpec(
            runner_running_config=cast(
                RunnerRunningConfigSnapshot,
                _as_runner_snapshot(runtime_payload.get("runner_running_config")),
            ),
            agent_running_config=cast(
                AgentRunningConfigSnapshot,
                _as_runner_snapshot(runtime_payload.get("agent_running_config")),
            ),
        ),
        tools=AcceptedToolConfigSpec(
            toolset_configs=toolset_configs,
        ),
        infrastructure=AcceptedInfrastructureSpec(
            trace_settings=_parse_optional_trace_settings(infrastructure_payload.get("trace_settings")),
            conversation_memory_settings=(
                _parse_conversation_memory_settings(
                    _as_object(infrastructure_payload.get("conversation_memory_settings"))
                )
                if infrastructure_payload.get("conversation_memory_settings") is not None
                else None
            ),
        ),
    )


def _coerce_execution_options_snapshot_value(
    value: PendingTurnSnapshotValue | None,
) -> ExecutionOptionsSnapshotValue:
    """把 pending turn 快照值收窄为 execution options 快照值。

    Args:
        value: pending turn 中保存的原始快照值。

    Returns:
        满足 execution options 快照契约的结构化值。

    Raises:
        ValueError: 当值结构不符合 execution options 快照约束时抛出。
    """

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_coerce_execution_options_snapshot_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _coerce_execution_options_snapshot_value(item)
            for key, item in value.items()
        }
    raise ValueError("execution_options 快照不支持列表结构")


def _coerce_execution_options_snapshot(
    payload: dict[str, PendingTurnSnapshotValue],
) -> ExecutionOptionsSnapshot:
    """把 pending turn execution_options 对象收窄为执行参数快照。

    Args:
        payload: pending turn 中保存的 execution_options 原始对象。

    Returns:
        满足 execution options 快照契约的对象。

    Raises:
        ValueError: 当对象中存在非法嵌套结构时抛出。
    """

    return {
        str(key): _coerce_execution_options_snapshot_value(item)
        for key, item in payload.items()
    }


def _parse_agent_create_args(value: dict[str, PendingTurnSnapshotValue]) -> AgentCreateArgs:
    """解析 AgentCreateArgs。"""

    return AgentCreateArgs(
        runner_type=_normalize_required_text(value.get("runner_type"), field_name="runner_type"),
        model_name=_normalize_required_text(value.get("model_name"), field_name="model_name"),
        max_turns=_coerce_optional_int(value.get("max_turns"), field_name="max_turns"),
        max_context_tokens=_coerce_optional_int(value.get("max_context_tokens"), field_name="max_context_tokens"),
        temperature=_coerce_optional_float(value.get("temperature"), field_name="temperature"),
        runner_params=_as_runner_params(value.get("runner_params")),
        runner_running_config=cast(
            RunnerRunningConfigSnapshot,
            _as_runner_snapshot(value.get("runner_running_config")),
        ),
        agent_running_config=cast(
            AgentRunningConfigSnapshot,
            _as_runner_snapshot(value.get("agent_running_config")),
        ),
    )


def _as_runner_params(value: PendingTurnRawValue) -> RunnerParams:
    """解析 runner 参数字典。"""

    if not isinstance(value, dict):
        return cast(RunnerParams, {})
    return cast(RunnerParams, {str(key): item for key, item in value.items()})


def _as_runner_snapshot(value: PendingTurnRawValue) -> dict[str, Any]:
    """解析运行配置快照。"""

    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _parse_execution_permissions(value: dict[str, PendingTurnSnapshotValue]) -> ExecutionPermissions:
    """解析执行权限。"""

    web_payload = _as_object(value.get("web")) if value.get("web") is not None else {}
    doc_payload = _as_object(value.get("doc")) if value.get("doc") is not None else {}
    return ExecutionPermissions(
        web=ExecutionWebPermissions(
            allow_private_network_url=bool(web_payload.get("allow_private_network_url")),
        ),
        doc=ExecutionDocPermissions(
            allowed_read_paths=_parse_string_tuple(doc_payload.get("allowed_read_paths")),
            allow_file_write=bool(doc_payload.get("allow_file_write")),
            allowed_write_paths=_parse_string_tuple(doc_payload.get("allowed_write_paths")),
        ),
    )


def _coerce_toolset_config_payload(
    payload: dict[str, PendingTurnSnapshotValue],
) -> dict[str, ToolsetConfigValue]:
    """把 pending turn payload 收窄为 toolset 配置 payload。"""

    return {
        str(key): cast(ToolsetConfigValue, serialize_toolset_config_payload_value(cast(ToolsetConfigValue, value)))
        for key, value in payload.items()
    }


def _parse_toolset_configs(value: PendingTurnRawValue) -> tuple[ToolsetConfigSnapshot, ...]:
    """解析通用 toolset 配置快照列表。"""

    if not isinstance(value, list):
        return ()
    snapshots: tuple[ToolsetConfigSnapshot, ...] = ()
    for item in value:
        item_payload = _as_object(item)
        toolset_name = _normalize_required_text(
            item_payload.get("toolset_name"),
            field_name="toolset_configs[].toolset_name",
        )
        version = _normalize_optional_text(item_payload.get("version")) or "1"
        payload = _snapshot_optional_object(item_payload.get("payload")) or {}
        snapshot = build_toolset_config_snapshot(
            toolset_name,
            _coerce_toolset_config_payload(payload),
            version=version,
        )
        if snapshot is None:
            raise ValueError("toolset_configs[] 必须包含 payload")
        snapshots = replace_toolset_config(snapshots, snapshot)
    return normalize_toolset_configs(snapshots)


def _parse_optional_trace_settings(value: PendingTurnRawValue) -> TraceSettings | None:
    """解析可选 TraceSettings。"""

    if value is None:
        return None
    payload = _coerce_dataclass_kwargs(_as_object(value))
    output_dir = payload.get("output_dir")
    payload["output_dir"] = Path(str(output_dir or ""))
    return TraceSettings(**payload)


def _parse_conversation_memory_settings(
    value: dict[str, PendingTurnSnapshotValue],
) -> ConversationMemorySettings:
    """解析会话记忆配置。"""

    return ConversationMemorySettings(**_coerce_dataclass_kwargs(value))


def _parse_optional_trace_identity(value: PendingTurnRawValue) -> AgentTraceIdentity | None:
    """解析可选 trace identity。"""

    if value is None:
        return None
    return AgentTraceIdentity(**_coerce_dataclass_kwargs(_as_object(value)))


def _parse_optional_conversation_session(value: PendingTurnRawValue) -> PreparedConversationSessionSnapshot | None:
    """解析可选会话恢复快照。"""

    if value is None:
        return None
    payload = _as_object(value)
    transcript_payload = _as_object(payload.get("transcript"))
    return PreparedConversationSessionSnapshot(
        session_id=_normalize_required_text(payload.get("session_id"), field_name="session_id"),
        user_message=_normalize_required_text(payload.get("user_message"), field_name="user_message"),
        transcript=ConversationTranscript.from_dict(_coerce_dataclass_kwargs(transcript_payload)),
    )


def _coerce_dataclass_kwargs(value: dict[str, PendingTurnSnapshotValue]) -> dict[str, Any]:
    """把 JSON object 转成 dataclass 构造 kwargs。"""

    return {str(key): item for key, item in value.items()}


__all__ = [
    "AcceptedAgentTurnSnapshot",
    "PendingTurnRawValue",
    "PreparedAgentTurnSnapshot",
    "PreparedConversationSessionSnapshot",
    "PendingTurnSnapshotValue",
    "deserialize_accepted_agent_turn_snapshot",
    "deserialize_prepared_agent_turn_snapshot",
    "serialize_accepted_agent_turn_snapshot",
    "serialize_prepared_agent_turn_snapshot",
]