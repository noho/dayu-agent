"""跨层传递的 runtime config 快照契约。

该模块只承载 Service / Host / Agent 之间传递的纯结构化快照 TypedDict，
作为契约层的稳定真源使用。execution 层的运行配置值对象与快照转换函数
仍归属于 ``dayu.execution.runtime_config``；此处只提供结构定义，避免
contracts 反向依赖 execution。
"""

from __future__ import annotations

from typing import TypedDict


class RunnerRunningConfigSnapshot(TypedDict, total=False):
    """跨层传递的 runner 运行配置快照。

    Attributes:
        debug_sse: 是否启用 SSE 调试日志。
        debug_tool_delta: 是否启用工具 delta 调试日志。
        debug_sse_sample_rate: SSE 调试采样率。
        debug_sse_throttle_sec: SSE 调试节流间隔（秒）。
        tool_timeout_seconds: 工具执行超时（秒）。
        stream_idle_timeout: 流式空闲超时（秒）。
        stream_idle_heartbeat_sec: 流式空闲心跳间隔（秒）。
    """

    debug_sse: bool
    debug_tool_delta: bool
    debug_sse_sample_rate: float
    debug_sse_throttle_sec: float
    tool_timeout_seconds: float
    stream_idle_timeout: float
    stream_idle_heartbeat_sec: float


class AgentRunningConfigSnapshot(TypedDict, total=False):
    """跨层传递的 agent 运行配置快照。

    Attributes:
        max_iterations: Agent 最大迭代次数。
        fallback_mode: Agent 降级模式。
        fallback_prompt: 降级 prompt。
        duplicate_tool_hint_prompt: 重复工具调用提醒 prompt。
        continuation_prompt: 续写 prompt。
        compaction_summary_header: 上下文压缩摘要头。
        compaction_summary_instruction: 上下文压缩指令。
        max_consecutive_failed_tool_batches: 连续失败工具批次上限。
        max_duplicate_tool_calls: 最大重复工具调用次数。
        budget_soft_limit_ratio: 软限额比例。
        budget_hard_limit_ratio: 硬限额比例。
        max_continuations: 最大续写次数。
        max_compactions: 最大压缩次数。
        max_context_tokens: 最大上下文 token 数。
    """

    max_iterations: int
    fallback_mode: str
    fallback_prompt: str
    duplicate_tool_hint_prompt: str
    continuation_prompt: str
    compaction_summary_header: str
    compaction_summary_instruction: str
    max_consecutive_failed_tool_batches: int
    max_duplicate_tool_calls: int
    budget_soft_limit_ratio: float
    budget_hard_limit_ratio: float
    max_continuations: int
    max_compactions: int
    max_context_tokens: int


__all__ = [
    "AgentRunningConfigSnapshot",
    "RunnerRunningConfigSnapshot",
]
