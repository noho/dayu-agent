"""Host 内部的 conversation memory / compaction 运行时共享契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

from dayu.contracts.agent_execution import AgentCreateArgs
from dayu.contracts.execution_options import ExecutionOptions
from dayu.contracts.agent_types import AgentMessage, AgentTraceIdentity
from dayu.contracts.infrastructure import PromptAssetStoreProtocol
from dayu.contracts.model_config import ModelConfig
from dayu.contracts.protocols import PromptToolExecutorProtocol, ToolTraceRecorderFactory
from dayu.contracts.tool_configs import WebToolsConfig
from dayu.engine.events import StreamEvent
from dayu.execution.options import ConversationMemorySettings
from dayu.prompting.scene_definition import SceneDefinition


class ConversationPreparedSceneProtocol(Protocol):
    """conversation memory 可见的最小静态 scene 视图。"""

    @property
    def scene_name(self) -> str:
        """返回 scene 名称。"""
        ...

    @property
    def model_config(self) -> ModelConfig:
        """返回当前生效的模型配置。"""
        ...

    @property
    def agent_create_args(self) -> AgentCreateArgs:
        """返回当前 scene 的 Agent 构造参数。"""
        ...

    @property
    def conversation_memory_settings(self) -> ConversationMemorySettings:
        """返回当前 scene 的会话记忆配置。"""
        ...


class ConversationCompactionSceneProtocol(ConversationPreparedSceneProtocol, Protocol):
    """conversation compaction 可见的静态 scene 视图。"""

    @property
    def scene_definition(self) -> SceneDefinition:
        """返回 scene 定义。"""
        ...

    @property
    def prompt_asset_store(self) -> PromptAssetStoreProtocol:
        """返回 prompt 资产仓储。"""
        ...

    @property
    def tool_registry(self) -> PromptToolExecutorProtocol:
        """返回当前 scene 的工具执行与快照视图。"""
        ...

    @property
    def tool_trace_recorder_factory(self) -> ToolTraceRecorderFactory | None:
        """返回工具追踪 recorder 工厂。"""
        ...

    @property
    def trace_identity(self) -> AgentTraceIdentity | None:
        """返回 trace 身份信息。"""
        ...


@dataclass(frozen=True)
class ConversationCompactionRequest:
    """conversation compaction 的显式请求对象。"""

    session_id: str


class ConversationCompactionAgentProtocol(Protocol):
    """conversation compaction 可见的最小 Agent 协议。"""

    def run_messages(
        self,
        messages: list[AgentMessage],
        *,
        session_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """执行 compaction 所需的一轮消息交互。"""

        ...


@dataclass(frozen=True)
class ConversationCompactionAgentHandle:
    """conversation compaction 可见的最小 Agent 句柄。"""

    agent: ConversationCompactionAgentProtocol
    system_prompt: str


class ConversationRuntimeProtocol(Protocol):
    """conversation memory 复用的 Host 内部运行时协议。"""

    def prepare_compaction_scene(
        self,
        scene_name: str,
        execution_options: ExecutionOptions | None = None,
        web_tools_config: WebToolsConfig | None = None,
    ) -> ConversationCompactionSceneProtocol:
        """返回 conversation compaction 所需的静态 scene 结果。

        Args:
            scene_name: 目标 scene 名称。
            execution_options: 可选的执行参数覆盖。
            web_tools_config: 可选的联网工具配置覆盖。

        Returns:
            可供 compaction 构造 Agent 的静态 scene 视图。

        Raises:
            ValueError: scene 配置不合法时抛出。
        """
        ...

    def prepare_compaction_agent(
        self,
        prepared_scene: ConversationCompactionSceneProtocol,
        request: ConversationCompactionRequest,
    ) -> ConversationCompactionAgentHandle:
        """基于静态 scene 与请求构造 compaction Agent。

        Args:
            prepared_scene: 由 ``prepare_compaction_scene`` 返回的静态 scene。
            request: 本次 compaction 请求。

        Returns:
            可执行的 compaction Agent 句柄。

        Raises:
            RuntimeError: Agent 构造失败时抛出。
        """
        ...


__all__ = [
    "ConversationCompactionAgentProtocol",
    "ConversationCompactionAgentHandle",
    "ConversationCompactionRequest",
    "ConversationCompactionSceneProtocol",
    "ConversationPreparedSceneProtocol",
    "ConversationRuntimeProtocol",
]
