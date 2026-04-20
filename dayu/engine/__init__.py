"""Engine 核心引擎导出。"""

# 核心组件
from .async_agent import AsyncAgent, AgentResult
from .tool_registry import ToolRegistry
from .tools.doc_tools import register_doc_tools
from .tools.utils_tools import register_utils_builtin_tools
from .tools.web_tools import register_web_tools
from dayu.contracts.tool_configs import DocToolLimits, WebToolsConfig

# 事件模型
from .events import (
    EventType,
    StreamEvent,
    content_delta,
    content_complete,
    reasoning_delta,
    tool_call_start,
    tool_call_delta,
    tool_call_dispatched,
    tool_call_result,
    tool_calls_batch_ready,
    tool_calls_batch_done,
    error_event,
    warning_event,
    done_event,
    metadata_event,
    final_answer_event,
)

# 取消原语
from dayu.contracts.cancellation import CancelledError, CancellationToken

# 异常
from .exceptions import (
    EngineError,
    ConfigError,
    ToolError,
    ToolNotFoundError,
    ToolExecutionError,
    ToolArgumentError,
    FileAccessError,
)

# 协议
from dayu.contracts.protocols import ToolExecutor
from .protocols import AsyncRunner as AsyncRunnerProtocol

__all__ = [
    # 核心类
    "AsyncAgent",
    "AgentResult",
    "ToolRegistry",
    "register_doc_tools",
    "register_utils_builtin_tools",
    "register_web_tools",
    "WebToolsConfig",
    "DocToolLimits",
    
    # 事件
    "EventType",
    "StreamEvent",
    "content_delta",
    "content_complete",
    "reasoning_delta",
    "tool_call_start",
    "tool_call_delta",
    "tool_call_dispatched",
    "tool_call_result",
    "tool_calls_batch_ready",
    "tool_calls_batch_done",
    "error_event",
    "warning_event",
    "done_event",
    "metadata_event",
    "final_answer_event",
    
    # 取消原语
    "CancellationToken",
    "CancelledError",
    
    # 异常
    "EngineError",
    "ConfigError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolArgumentError",
    "FileAccessError",
    
    # 协议
    "ToolExecutor",
    "AsyncRunnerProtocol",
]
