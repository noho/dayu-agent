"""Agent 执行路径公共契约。

该模块定义从 ``Service -> Host -> scene preparation -> Agent`` 之间传递的
稳定数据结构：

- ``ExecutionContract``：Service 输出给 Host 的单个 Agent 子执行契约。
- ``AgentInput``：scene preparation 收敛后交给 Agent 的最低可执行输入。

这些对象不负责业务解释，只负责承载已经完成业务解释后的执行决策。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

from dayu.contracts.agent_types import (
    AgentMessage,
    AgentRuntimeLimits,
    AgentTraceIdentity,
    ConversationTurnPersistenceProtocol,
)
from dayu.contracts.cancellation import CancellationToken
from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.contracts.execution_options import (
    ConversationMemorySettings,
    ExecutionOptions,
    TraceSettings,
)
from dayu.contracts.model_config import RunnerParams
from dayu.contracts.protocols import ToolExecutor, ToolTraceRecorderFactory
from dayu.contracts.toolset_config import ToolsetConfigSnapshot, normalize_toolset_configs
from dayu.contracts.runtime_config_snapshot import AgentRunningConfigSnapshot, RunnerRunningConfigSnapshot

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
            scene preparation 是不透明透传对象，**不参与契约层解释**，也**不参与
            Host 执行路径**——Host 始终以 ``accepted_execution_spec`` 作为执行
            参数来源。此处保留 ``execution_options`` 仅用于审计/调试快照，便于
            从 trace 追溯 Service 原始请求意图；任何 Host 侧的行为差异都应通过
            ``accepted_execution_spec`` 表达，避免快照恢复时出现语义漂移。
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
]
