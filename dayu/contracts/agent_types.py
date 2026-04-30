"""Agent 执行路径共享类型。

该模块承载 ``Service -> Host -> Agent`` 主链路会复用的轻量类型，避免在
公共契约上继续使用 ``Any`` 和无约束字典袋子。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, Protocol, TypeAlias, TypedDict


JsonScalar: TypeAlias = str | int | float | bool | None
"""JSON 标量值。"""


JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
"""递归 JSON 值。"""


ExtraContentPayload: TypeAlias = dict[str, dict[str, JsonValue]]
"""provider 私有 ``extra_content`` 透传负载。"""


class FunctionToolCallPayload(TypedDict):
    """工具调用中的 function 负载。"""

    name: str
    arguments: str


class ToolCallPayload(TypedDict):
    """assistant message 中的单个工具调用。"""

    id: str
    type: Literal["function"]
    function: FunctionToolCallPayload
    extra_content: NotRequired[ExtraContentPayload]


class SystemChatMessage(TypedDict):
    """system role 消息。"""

    role: Literal["system"]
    content: str


class UserChatMessage(TypedDict):
    """user role 消息。"""

    role: Literal["user"]
    content: str


class AssistantChatMessage(TypedDict, total=False):
    """assistant role 消息。"""

    role: Literal["assistant"]
    content: str | None
    tool_calls: list[ToolCallPayload]
    reasoning_content: str


class ToolChatMessage(TypedDict, total=False):
    """tool role 消息。"""

    role: Literal["tool"]
    tool_call_id: str
    content: str
    name: str


AgentMessage: TypeAlias = (
    SystemChatMessage
    | UserChatMessage
    | AssistantChatMessage
    | ToolChatMessage
)


def build_system_chat_message(content: str) -> SystemChatMessage:
    """构造 system role 消息。"""

    return {"role": "system", "content": content}


def build_user_chat_message(content: str) -> UserChatMessage:
    """构造 user role 消息。"""

    return {"role": "user", "content": content}


def build_assistant_chat_message(
    *,
    content: str | None,
    tool_calls: list[ToolCallPayload] | None = None,
    reasoning_content: str | None = None,
) -> AssistantChatMessage:
    """构造 assistant role 消息。"""

    message: AssistantChatMessage = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return message


def build_tool_chat_message(
    *,
    tool_call_id: str,
    content: str,
    name: str | None = None,
) -> ToolChatMessage:
    """构造 tool role 消息。"""

    message: ToolChatMessage = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content,
    }
    if name:
        message["name"] = name
    return message


@dataclass(frozen=True)
class AgentRuntimeLimits:
    """Host 传给 Agent 的显式运行限制。"""

    timeout_ms: int | None = None


@dataclass(frozen=True)
class AgentTraceIdentity:
    """Host 传给 Agent / trace recorder 的固定身份信息。"""

    agent_name: str
    agent_kind: str
    scene_name: str
    model_name: str
    session_id: str | None = None

    def to_metadata(self) -> dict[str, str]:
        """转换为 trace recorder 使用的元数据字典。"""

        metadata = {
            "agent_name": self.agent_name,
            "agent_kind": self.agent_kind,
            "scene_name": self.scene_name,
            "model_name": self.model_name,
        }
        if self.session_id:
            metadata["session_id"] = self.session_id
        return metadata


class ConversationTurnPersistenceProtocol(Protocol):
    """Host 会话状态需要暴露给执行器的最小能力。"""

    def persist_turn(
        self,
        *,
        final_content: str,
        degraded: bool,
        tool_uses: tuple[object, ...],
        warnings: tuple[str, ...],
        errors: tuple[str, ...],
    ) -> None:
        """持久化当前 conversation turn 的执行结果。"""

    def record_reasoning_delta(self, chunk: str) -> None:
        """累积本轮 reasoning 增量到展示侧 buffer。

        语义说明：reasoning 仅作为历史展示字段持久化，**不**进入运行态送模、
        memory、compaction、resume 决策链路。executor 在收到 reasoning 增量
        事件时调用本方法做 buffer，``persist_turn`` 时统一刷入展示侧子视图。

        Args:
            chunk: 单次 reasoning 增量文本。

        Returns:
            无。

        Raises:
            无。
        """


__all__ = [
    "AgentMessage",
    "AgentRuntimeLimits",
    "AgentTraceIdentity",
    "AssistantChatMessage",
    "build_assistant_chat_message",
    "build_system_chat_message",
    "build_tool_chat_message",
    "build_user_chat_message",
    "ConversationTurnPersistenceProtocol",
    "FunctionToolCallPayload",
    "SystemChatMessage",
    "ToolCallPayload",
    "ToolChatMessage",
    "UserChatMessage",
]
