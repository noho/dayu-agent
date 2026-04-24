"""跨层稳定协议定义。

本模块承载跨层共享的稳定协议，以及少量需要多层共同消费的
强类型执行上下文契约。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Collection, Optional, Protocol, Sequence

from dayu.contracts.agent_types import AgentMessage
from dayu.contracts.cancellation import CancellationToken


@dataclass(frozen=True)
class ToolExecutionContext:
    """单次 tool call 的强类型执行上下文。

    Args:
        run_id: 当前 Host run ID。
        iteration_id: 当前 Engine iteration ID。
        tool_call_id: 当前工具调用 ID。
        index_in_iteration: 当前工具在本轮中的顺序索引。
        timeout_seconds: 当前工具调用预算秒数。
        cancellation_token: 当前工具调用可观察的取消令牌。

    Returns:
        无。

    Raises:
        无。
    """

    run_id: str | None = None
    iteration_id: str | None = None
    tool_call_id: str | None = None
    index_in_iteration: int = 0
    timeout_seconds: float | None = None
    cancellation_token: CancellationToken | None = None


class DupCallSpecProtocol(Protocol):
    """重复调用策略的最小结构协议。"""

    mode: str
    status_path: str | None
    terminal_values: list[str] | None


class PromptToolCatalogProtocol(Protocol):
    """供 Prompt 装配读取工具快照的最小协议。"""

    def get_tool_names(self) -> Collection[str]:
        """返回当前可见的工具名集合。"""

        ...

    def get_tool_tags(self) -> Collection[str]:
        """返回当前可见的工具标签集合。"""

        ...

    def get_allowed_paths(self) -> Sequence[str]:
        """返回当前工具允许访问的路径列表。"""

        ...


class ToolTraceRecorder(Protocol):
    """单次 run 的工具追踪记录器协议。"""

    def start_iteration(
        self,
        *,
        iteration_id: str,
        model_input_messages: list[AgentMessage],
        tool_schemas: list[dict[str, Any]],
    ) -> None:
        """开始记录一次 agent iteration 的送模上下文。"""

        ...

    def on_tool_dispatched(self, *, iteration_id: str, payload: Any) -> None:
        """观察到工具请求发起事件。"""

        ...

    def on_tool_result(self, *, iteration_id: str, payload: Any) -> None:
        """观察到工具返回事件。"""

        ...

    def record_iteration_usage(
        self,
        *,
        iteration_id: str,
        usage: dict[str, Any],
        budget_snapshot: Optional[dict[str, Any]] = None,
    ) -> None:
        """记录单次 iteration token 用量。"""

        ...

    def record_final_response(
        self,
        *,
        iteration_id: str,
        content: str,
        degraded: bool,
        filtered: bool = False,
        finish_reason: str | None = None,
    ) -> None:
        """记录最终回答。"""

        ...

    def finish_iteration(
        self,
        *,
        iteration_id: str,
        iteration_index: int,
        termination_reason: str | None = None,
    ) -> None:
        """结束一次 iteration 并输出上下文快照。"""

        ...

    def close(self) -> None:
        """关闭 recorder，并补偿未配对记录。"""

        ...


class ToolTraceRecorderFactory(Protocol):
    """工具追踪记录器工厂协议。"""

    def create_recorder(
        self,
        *,
        run_id: str,
        session_id: str,
        agent_metadata: Optional[dict[str, Any]] = None,
    ) -> ToolTraceRecorder:
        """创建单次 run 使用的 recorder。"""

        ...


class ToolExecutor(Protocol):
    """工具执行器接口协议。"""

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> dict[str, Any]:
        """执行工具并返回结构化结果。"""

        ...

    def get_schemas(self) -> list[dict[str, Any]]:
        """获取所有工具的 schema 定义。"""

        ...

    def clear_cursors(self) -> None:
        """清除截断游标，释放关联的数据引用。"""

        ...

    def get_dup_call_spec(self, name: str) -> Optional[DupCallSpecProtocol]:
        """按工具名读取重复调用策略声明。"""

        ...

    def get_execution_context_param_name(self, name: str) -> str | None:
        """按工具名读取 execution context 注入参数名。"""

        ...

    def get_tool_display_info(self, name: str) -> tuple[str, list[str] | None]:
        """按工具名读取面向用户的展示元数据。

        Args:
            name: 工具名称。

        Returns:
            ``(display_name, summary_params)`` 二元组；display_name fallback 到 name，
            summary_params 为 None 表示不展示参数摘要。
        """

        ...

    def register_response_middleware(
        self,
        callback: Callable[[str, dict[str, Any], ToolExecutionContext | None], dict[str, Any]],
    ) -> None:
        """注册 response middleware，在工具执行成功后链式调用。"""

        ...


class PromptToolExecutorProtocol(ToolExecutor, PromptToolCatalogProtocol, Protocol):
    """同时具备工具执行与 Prompt 工具快照能力的最小协议。"""

    ...
