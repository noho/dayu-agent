"""Agent 执行路径公共契约。

该模块定义从 ``Service -> Host -> scene preparation -> Agent`` 之间传递的
稳定数据结构：

- ``ExecutionContract``：Service 输出给 Host 的单个 Agent 子执行契约。
- ``AgentInput``：scene preparation 收敛后交给 Agent 的最低可执行输入。

这些对象不负责业务解释，只负责承载已经完成业务解释后的执行决策。
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

from dayu.contracts.agent_types import (
    AgentMessage,
    AgentRuntimeLimits,
    AgentTraceIdentity,
    ConversationTurnPersistenceProtocol,
)
from dayu.contracts.cancellation import CancellationToken
from dayu.contracts.execution_metadata import ExecutionDeliveryContext, normalize_execution_delivery_context
from dayu.contracts.execution_options import (
    ConversationMemorySettings,
    ExecutionOptions,
    ExecutionOptionsSnapshot,
    ExecutionOptionsSnapshotValue,
    TraceSettings,
    deserialize_execution_options_snapshot,
    serialize_execution_options_snapshot,
)
from dayu.contracts.model_config import RunnerParams
from dayu.contracts.protocols import ToolExecutor, ToolTraceRecorderFactory
from dayu.contracts.toolset_config import (
    ToolsetConfigSnapshot,
    ToolsetConfigValue,
    build_toolset_config_snapshot,
    normalize_toolset_configs,
    replace_toolset_config,
    serialize_toolset_config_payload_value,
)
from dayu.execution.runtime_config import AgentRunningConfigSnapshot, RunnerRunningConfigSnapshot

ExecutionContractSnapshotScalar: TypeAlias = str | int | float | bool | None
ExecutionContractSnapshotValue: TypeAlias = (
    ExecutionContractSnapshotScalar
    | list["ExecutionContractSnapshotValue"]
    | dict[str, "ExecutionContractSnapshotValue"]
)
ExecutionContractSnapshot: TypeAlias = dict[str, ExecutionContractSnapshotValue]


def _empty_runner_running_config_snapshot() -> RunnerRunningConfigSnapshot:
    """返回空的 runner 运行配置快照。

    Args:
        无。

    Returns:
        空的 runner 运行配置快照。

    Raises:
        无。
    """

    return {}


def _empty_agent_running_config_snapshot() -> AgentRunningConfigSnapshot:
    """返回空的 agent 运行配置快照。

    Args:
        无。

    Returns:
        空的 agent 运行配置快照。

    Raises:
        无。
    """

    return {}


def _empty_runner_params() -> RunnerParams:
    """返回空的 runner 参数快照。

    Args:
        无。

    Returns:
        空的 runner 参数快照。

    Raises:
        无。
    """

    return {}


def _empty_execution_delivery_context() -> ExecutionDeliveryContext:
    """返回空的交付上下文。

    Args:
        无。

    Returns:
        空的交付上下文映射。

    Raises:
        无。
    """

    return {}


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


@dataclass(frozen=True)
class ExecutionWebPermissions:
    """单次执行下 Web 工具域的动态权限策略。

    Args:
        allow_private_network_url: 是否允许访问私网 URL。

    Returns:
        无。

    Raises:
        无。
    """

    allow_private_network_url: bool = False


@dataclass(frozen=True)
class ExecutionDocPermissions:
    """单次执行下文档工具域的动态权限策略。

    Args:
        allowed_read_paths: 允许读取的路径白名单。
        allow_file_write: 是否允许写文件。
        allowed_write_paths: 允许写入的路径白名单。

    Returns:
        无。

    Raises:
        无。
    """

    allowed_read_paths: tuple[str, ...] = ()
    allow_file_write: bool = False
    allowed_write_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionPermissions:
    """单次执行的动态权限收窄策略。

    Args:
        web: Web 工具域的动态权限策略。
        doc: 文档工具域的动态权限策略。

    Returns:
        无。

    Raises:
        无。
    """

    web: ExecutionWebPermissions = field(default_factory=ExecutionWebPermissions)
    doc: ExecutionDocPermissions = field(default_factory=ExecutionDocPermissions)


@dataclass(frozen=True)
class ScenePreparationSpec:
    """scene preparation 所需的机械装配说明。

    Args:
        selected_toolsets: 本次执行显式启用的工具集合名。
        execution_permissions: 单次执行的动态权限收窄结果。
        prompt_contributions: 由 Service 提供的动态 prompt 片段。

    Returns:
        无。

    Raises:
        无。
    """

    selected_toolsets: tuple[str, ...] = ()
    execution_permissions: ExecutionPermissions = field(default_factory=ExecutionPermissions)
    prompt_contributions: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AcceptedModelSpec:
    """Service 已接受的模型选择规格。

    Args:
        model_name: 当前 scene 生效的模型名。
        temperature: 当前 scene 生效的 temperature。

    Returns:
        无。

    Raises:
        无。
    """

    model_name: str
    temperature: float | None = None


@dataclass(frozen=True)
class AcceptedRuntimeSpec:
    """Service 已接受的运行时快照规格。

    Args:
        runner_running_config: 已接受的 runner 运行配置快照。
        agent_running_config: 已接受的 agent 运行配置快照。

    Returns:
        无。

    Raises:
        无。
    """

    runner_running_config: RunnerRunningConfigSnapshot = field(
        default_factory=_empty_runner_running_config_snapshot
    )
    agent_running_config: AgentRunningConfigSnapshot = field(
        default_factory=_empty_agent_running_config_snapshot
    )


@dataclass(frozen=True, init=False)
class AcceptedToolConfigSpec:
    """Service 已接受的工具域配置。

    Args:
        toolset_configs: 已接受的通用 toolset 配置快照。

    Returns:
        无。

    Raises:
        无。
    """

    toolset_configs: tuple[ToolsetConfigSnapshot, ...] = field(default_factory=tuple)

    def __init__(self, toolset_configs: tuple[ToolsetConfigSnapshot, ...] = ()) -> None:
        """初始化已接受的工具域配置。

        Args:
            toolset_configs: 已接受的通用 toolset 配置快照。

        Returns:
            无。

        Raises:
            TypeError: 当 toolset 配置值无法序列化为通用快照时抛出。
            ValueError: 当 toolset 名称非法时抛出。
        """

        object.__setattr__(self, "toolset_configs", normalize_toolset_configs(toolset_configs))


@dataclass(frozen=True)
class AcceptedInfrastructureSpec:
    """Service 已接受的基础设施配置。

    Args:
        trace_settings: 已接受的工具追踪配置。
        conversation_memory_settings: 已接受的会话记忆配置。

    Returns:
        无。

    Raises:
        无。
    """

    trace_settings: TraceSettings | None = None
    conversation_memory_settings: ConversationMemorySettings | None = None


@dataclass(frozen=True, init=False)
class AcceptedExecutionSpec:
    """Service 已接受的执行规格。

    这里承载的是 Service 基于 scene 规则、显式请求参数和默认配置完成接受后的
    执行结果。Host 只能消费这些结果并继续机械装配，不能回头重新解释业务语义。

    Args:
        model: 已接受的模型选择规格。
        runtime: 已接受的运行时快照规格。
        tools: 已接受的工具域配置。
        infrastructure: 已接受的基础设施配置。
    """

    model: AcceptedModelSpec
    runtime: AcceptedRuntimeSpec = field(default_factory=AcceptedRuntimeSpec)
    tools: AcceptedToolConfigSpec = field(default_factory=AcceptedToolConfigSpec)
    infrastructure: AcceptedInfrastructureSpec = field(default_factory=AcceptedInfrastructureSpec)

    def __init__(
        self,
        model: AcceptedModelSpec,
        runtime: AcceptedRuntimeSpec | None = None,
        tools: AcceptedToolConfigSpec | None = None,
        infrastructure: AcceptedInfrastructureSpec | None = None,
    ) -> None:
        """初始化已接受执行规格。

        Args:
            model: 已接受的模型选择规格。
            runtime: 已接受的运行时快照规格。
            tools: 已接受的工具域配置。
            infrastructure: 已接受的基础设施配置。

        Returns:
            无。

        Raises:
            无。
        """

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "runtime", runtime or AcceptedRuntimeSpec())
        object.__setattr__(self, "tools", tools or AcceptedToolConfigSpec())
        object.__setattr__(self, "infrastructure", infrastructure or AcceptedInfrastructureSpec())


@dataclass(frozen=True)
class ExecutionHostPolicy:
    """Host 侧生命周期治理策略。

    Args:
        session_key: Host Session 键。
        business_concurrency_lane: 业务并发通道名称。

            该字段仅用于 Service 声明业务并发通道（如 ``write_chapter`` /
            ``sec_download``）；``llm_api`` 属于 Host 自治 lane，由 Host 根据
            ExecutionContract 的调用路径自动叠加，禁止在此字段写入 Host 自治
            lane 名。
        timeout_ms: 本次执行超时。
        resumable: 是否允许恢复。

    Returns:
        无。

    Raises:
        无。
    """

    session_key: str | None = None
    business_concurrency_lane: str | None = None
    timeout_ms: int | None = None
    resumable: bool = False


@dataclass(frozen=True)
class ExecutionMessageInputs:
    """Service 交给 scene preparation 的当前轮消息输入。

    Args:
        user_message: 当前轮用户输入。

    Returns:
        无。

    Raises:
        无。
    """

    user_message: str | None = None


@dataclass(frozen=True)
class AgentCreateArgs:
    """构造 AsyncAgent/AsyncRunner 所需的完整参数对象。

    这里允许携带少量内部运行配置字段，以避免 Agent 再次理解配置文件结构。

    Args:
        runner_type: Runner 类型。
        model_name: 逻辑模型名。
        max_turns: 最大工具轮次。
        max_context_tokens: 最大上下文长度。
        max_output_tokens: 最大输出长度。
        temperature: 最终 temperature。
        runner_params: Runner 构造参数。
        runner_running_config: Runner 运行时配置。
        agent_running_config: Agent 运行时配置。

    Returns:
        无。

    Raises:
        无。
    """

    runner_type: str
    model_name: str
    max_turns: int | None = None
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    runner_params: RunnerParams = field(default_factory=_empty_runner_params)
    runner_running_config: RunnerRunningConfigSnapshot = field(default_factory=_empty_runner_running_config_snapshot)
    agent_running_config: AgentRunningConfigSnapshot = field(default_factory=_empty_agent_running_config_snapshot)


@dataclass(frozen=True)
class ExecutionContract:
    """Service 输出给 Host 的单个 Agent 子执行契约。

    Args:
        service_name: 发起该子执行的 Service 名。
        scene_name: 目标 scene 名称。
        host_policy: Host 生命周期治理策略。
        preparation_spec: scene preparation 装配说明。
        message_inputs: 当前轮消息输入。
        accepted_execution_spec: Service 已接受的执行规格。
        execution_options: Service 已接受的通用执行显式参数。该字段对 Host /
            scene preparation 是不透明透传对象，不参与契约层解释。
        metadata: 宿主侧交付上下文。

    Returns:
        无。

    Raises:
        无。
    """

    service_name: str
    scene_name: str
    host_policy: ExecutionHostPolicy
    preparation_spec: ScenePreparationSpec
    message_inputs: ExecutionMessageInputs
    accepted_execution_spec: AcceptedExecutionSpec
    execution_options: ExecutionOptions | None = None
    metadata: ExecutionDeliveryContext = field(default_factory=_empty_execution_delivery_context)


@dataclass(frozen=True)
class AgentInput:
    """scene preparation 收敛后的最低可执行输入。

    Args:
        system_prompt: 最终 system prompt。
        messages: 最终送模消息。
        tools: 最终工具执行器。
        agent_create_args: 用于构造 Agent 的参数对象。
        session_state: Host Session 下的会话状态快照。
        runtime_limits: 运行限制。
        cancellation_handle: 取消句柄。
        tool_trace_recorder_factory: tool trace recorder 工厂。
        trace_identity: trace 身份元数据。

    Returns:
        无。

    Raises:
        无。
    """

    system_prompt: str
    messages: list[AgentMessage]
    tools: ToolExecutor | None = None
    agent_create_args: AgentCreateArgs = field(default_factory=lambda: AgentCreateArgs(runner_type="", model_name=""))
    session_state: ConversationTurnPersistenceProtocol | None = None
    runtime_limits: AgentRuntimeLimits = field(default_factory=AgentRuntimeLimits)
    cancellation_handle: CancellationToken | None = None
    tool_trace_recorder_factory: ToolTraceRecorderFactory | None = None
    trace_identity: AgentTraceIdentity | None = None


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


def _snapshot_optional_int(value: ExecutionContractSnapshotValue | None) -> int | None:
    """从快照值中读取可选整数。"""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _snapshot_optional_float(value: ExecutionContractSnapshotValue | None) -> float | None:
    """从快照值中读取可选浮点数。"""

    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


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


def _snapshot_bool_or_default(
    value: ExecutionContractSnapshotValue | None,
    *,
    default: bool,
) -> bool:
    """从契约快照值读取布尔值，缺失时回退到默认值。

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
        ValueError: 当值包含执行参数快照不支持的列表结构时抛出。
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
        working_memory_max_turns=_snapshot_int_or_default(
            payload.get("working_memory_max_turns"),
            default=defaults.working_memory_max_turns,
        ),
        working_memory_token_budget_ratio=_snapshot_float_or_default(
            payload.get("working_memory_token_budget_ratio"),
            default=defaults.working_memory_token_budget_ratio,
        ),
        working_memory_token_budget_floor=_snapshot_int_or_default(
            payload.get("working_memory_token_budget_floor"),
            default=defaults.working_memory_token_budget_floor,
        ),
        working_memory_token_budget_cap=_snapshot_int_or_default(
            payload.get("working_memory_token_budget_cap"),
            default=defaults.working_memory_token_budget_cap,
        ),
        episodic_memory_token_budget_ratio=_snapshot_float_or_default(
            payload.get("episodic_memory_token_budget_ratio"),
            default=defaults.episodic_memory_token_budget_ratio,
        ),
        episodic_memory_token_budget_floor=_snapshot_int_or_default(
            payload.get("episodic_memory_token_budget_floor"),
            default=defaults.episodic_memory_token_budget_floor,
        ),
        episodic_memory_token_budget_cap=_snapshot_int_or_default(
            payload.get("episodic_memory_token_budget_cap"),
            default=defaults.episodic_memory_token_budget_cap,
        ),
        compaction_trigger_turn_count=_snapshot_int_or_default(
            payload.get("compaction_trigger_turn_count"),
            default=defaults.compaction_trigger_turn_count,
        ),
        compaction_trigger_token_ratio=_snapshot_float_or_default(
            payload.get("compaction_trigger_token_ratio"),
            default=defaults.compaction_trigger_token_ratio,
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
    return ExecutionContract(
        service_name=_snapshot_optional_str(snapshot.get("service_name")) or "",
        scene_name=_snapshot_optional_str(snapshot.get("scene_name")) or "",
        host_policy=ExecutionHostPolicy(
            session_key=_snapshot_optional_str(host_policy.get("session_key")),
            business_concurrency_lane=_snapshot_optional_str(host_policy.get("business_concurrency_lane")),
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


__all__ = [
    "AcceptedInfrastructureSpec",
    "AcceptedModelSpec",
    "AcceptedRuntimeSpec",
    "AcceptedToolConfigSpec",
    "AcceptedExecutionSpec",
    "AgentCreateArgs",
    "AgentInput",
    "ExecutionContractSnapshot",
    "ExecutionContractSnapshotValue",
    "ExecutionContract",
    "ExecutionDocPermissions",
    "ExecutionHostPolicy",
    "ExecutionMessageInputs",
    "ExecutionPermissions",
    "ExecutionWebPermissions",
    "ScenePreparationSpec",
    "deserialize_execution_contract_snapshot",
    "serialize_execution_contract_snapshot",
]
