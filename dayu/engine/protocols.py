"""协议定义模块 - Engine 接口契约。

基于 Protocol（结构化子类型）定义 Runner 的最小能力集，
并导出跨层共享的 `ToolExecutor` 协议。

设计目标：
- 解耦实现与依赖：Agent 只依赖接口。
- 易于测试：可使用 Mock Runner / Executor。
- 易于扩展：支持多种 LLM Runner 与工具实现。

关键协议：
- `AsyncRunner`: 约束 `call()` / `set_tools()` / `is_supports_tool_calling()`。
- `ToolExecutor`: 由 `dayu.contracts.protocols` 提供稳定定义。
"""
from typing import TYPE_CHECKING, AsyncIterator, List, Optional, Protocol

from dayu.contracts.agent_types import AgentMessage
from dayu.contracts.protocols import ToolExecutor

if TYPE_CHECKING:
    from .events import StreamEvent


class AsyncRunner(Protocol):
    """异步 Runner 接口协议。

    定义 Agent 需要的异步 LLM 调用能力。
    任何实现了这些方法的类都可以作为 Agent 的 runner。

    当前稳定实现示例：
    - `AsyncOpenAIRunner`: OpenAI 兼容 API。
    """
    
    def call(
        self,
        messages: List[AgentMessage],
        *,
        stream: bool = True,
        **extra_payloads,
    ) -> AsyncIterator["StreamEvent"]:
        """
        调用 LLM 并返回 streaming 事件流
        
        Args:
            messages: 消息列表（OpenAI 格式）
            stream: 是否启用 streaming
            **extra_payloads: 额外参数
        
        Yields:
            StreamEvent: 流式事件
        """
        ...
    
    def set_tools(self, executor: Optional["ToolExecutor"]) -> None:
        """
        设置工具执行器（工具定义从 executor.get_schemas() 获取）
        
        Args:
            executor: 工具执行器实例；传 None 表示清空工具能力
        """
        ...

    def is_supports_tool_calling(self) -> bool:
        """
        是否支持 Tool Calling
        
        Returns:
            bool: True 表示支持工具调用
        """
        ...

    async def close(self) -> None:
        """关闭 Runner 持有的异步资源。

        Returns:
            无。
        """
        ...
