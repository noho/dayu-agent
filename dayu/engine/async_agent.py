"""
异步 Agent - 负责推理循环与事件聚合

提供统一的异步接口，屏蔽 Runner 差异，负责：
- 构造系统/用户消息，驱动多轮推理
- 处理工具调用批次，按顺序回填 tool messages
- 管理迭代上限与降级策略，产出 final_answer
- 透传 Runner 事件流，供调用者实时消费

核心特性:
- 无状态执行：每次 run() 独立构造 messages
- 支持多轮会话：可直接传入外部维护的 messages
- 工具调用闭环：等待 TOOL_CALLS_BATCH_DONE 后回填
- 降级策略：达到迭代上限后移除工具能力并强制回答
- 事件对齐：内容/工具/错误事件顺序与 Runner 规范一致

主要入口:
1. run(prompt, system_prompt="", session_id=None, stream=True, **extra_payloads) -> AsyncIterator[StreamEvent]
2. run_messages(messages, session_id=None, stream=True, **extra_payloads) -> AsyncIterator[StreamEvent]
3. run_and_wait(prompt, system_prompt="", session_id=None, **extra_payloads) -> AgentResult
重要事件:
- content_delta / content_complete: 文本输出
- tool_call_dispatched / tool_call_result / tool_calls_batch_done: 工具调用闭环
- final_answer: 该轮推理结束（含 degraded 标记）
"""

import copy
import json
import uuid
from threading import Lock
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Tuple, cast

from dataclasses import dataclass

from dayu.contracts.agent_types import (
    AgentMessage,
    AgentTraceIdentity,
    ToolCallPayload,
    build_assistant_chat_message,
    build_system_chat_message,
    build_tool_chat_message,
    build_user_chat_message,
)
from .duplicate_call_guard import DuplicateCallGuard
from .context_budget import ContextBudgetState, PREDICTIVE_OVERHEAD_TOKENS, ToolResultBudgetCapper
from .events import (
    EventType,
    StreamEvent,
    error_event,
    final_answer_event,
    iteration_start_event,
    warning_event,
)

MODULE = "ENGINE.ASYNC_AGENT"
from .protocols import AsyncRunner, ToolExecutor
from dayu.log import Log
from .tool_contracts import DupCallSpec
from .tool_result import (
    is_tool_success,
    project_for_llm,
)
from dayu.contracts.cancellation import CancellationToken
from .tool_trace import ToolTraceRecorder, ToolTraceRecorderFactory

DEFAULT_FALLBACK_PROMPT = (
    "Based on the information gathered, answer the question directly. "
    "Do not fabricate if information is insufficient."
)
DEFAULT_DUPLICATE_TOOL_HINT_PROMPT = (
    "You just called the same tool ({{tool_name}}) with identical parameters. "
    "Reuse the existing results first. Only call a tool again if you clearly "
    "need new information, and provide your conclusion promptly."
)
_RESERVED_EXTRA_PAYLOAD_KEYS = frozenset({"session_id", "run_id", "iteration_id", "trace_context"})


def _validate_extra_payload_keys(extra_payloads: Dict[str, Any]) -> None:
    """校验外部透传参数中是否包含内部保留字段。

    Args:
        extra_payloads: 调用方传入的额外透传参数。

    Returns:
        无。

    Raises:
        ValueError: 当调用方试图注入内部保留字段时抛出。
    """

    duplicated_keys = sorted(_RESERVED_EXTRA_PAYLOAD_KEYS.intersection(extra_payloads.keys()))
    if duplicated_keys:
        duplicated = ", ".join(duplicated_keys)
        raise ValueError(f"extra_payloads 包含内部保留字段: {duplicated}")
DEFAULT_CONTINUATION_PROMPT = (
    "Your previous response was truncated (finish_reason=length). "
    "Continue from where you left off without repeating content already produced."
)
DEFAULT_COMPACTION_SUMMARY_HEADER = "[Context Compaction Summary]"
DEFAULT_COMPACTION_SUMMARY_INSTRUCTION = (
    "Continue reasoning based on recent context. "
    "Avoid repeating tool calls that have already been completed."
)

# 压缩中保留的最近 message 条数（system 和首条 user 单独保留）
_COMPACT_RECENT_KEEP = 6
_COMPACTION_TOOL_RESULT_MAX_LINES = 4
_COMPACTION_VALUE_SUMMARY_MAX_CHARS = 160
_COMPACTION_ERROR_SUMMARY_MAX_CHARS = 120


def _normalize_system_prompt(system_prompt: Optional[str]) -> str:
    """规范化系统提示词文本。

    Args:
        system_prompt: 调用方传入的系统提示词。

    Returns:
        去除首尾空白后的系统提示词；为空时返回空字符串。

    Raises:
        无。
    """

    return (system_prompt or "").strip()


def _build_messages_from_prompt(
    *,
    prompt: str,
    system_prompt: Optional[str],
) -> List[AgentMessage]:
    """根据单轮输入构建 messages。

    Args:
        prompt: 用户输入。
        system_prompt: 可选系统提示词。

    Returns:
        可直接传给 Runner 的 messages 列表。

    Raises:
        ValueError: prompt 为空字符串时。
    """

    if not prompt.strip():
        raise ValueError("prompt 不能为空")
    messages: List[AgentMessage] = []
    normalized_system_prompt = _normalize_system_prompt(system_prompt)
    if normalized_system_prompt:
        messages.append(build_system_chat_message(normalized_system_prompt))
    messages.append(build_user_chat_message(prompt))
    return messages


def _build_tool_trace_budget_snapshot(
    *,
    budget_state: ContextBudgetState,
    iteration: int,
    max_iterations: int,
) -> Dict[str, Any]:
    """构建 trace 使用的预算快照。

    Args:
        budget_state: 当前上下文预算状态。
        iteration: 当前已完成轮次数。
        max_iterations: 最大工具调用轮次预算。

    Returns:
        预算快照字典。

    Raises:
        无。
    """

    return {
        "max_context_tokens": budget_state.max_context_tokens,
        "current_prompt_tokens": budget_state.current_prompt_tokens,
        "total_prompt_tokens": budget_state.total_prompt_tokens,
        "total_completion_tokens": budget_state.total_completion_tokens,
        "iteration_count": budget_state.iteration_count,
        "compaction_count": budget_state.compaction_count,
        "continuation_count": budget_state.continuation_count,
        "is_over_soft_limit": budget_state.is_over_soft_limit,
        "tool_call_budget": max_iterations,
        "tool_calls_remaining": max(0, max_iterations - iteration),
    }


def _build_tool_calls_payload(
    ordered_tool_calls: List[Dict[str, Any]],
) -> list[ToolCallPayload]:
    """根据已排序的工具调用数据构造 OpenAI tool_calls payload。

    Args:
        ordered_tool_calls: 已按 index_in_iteration 排序的工具调用数据列表，
            每项至少包含 ``id`` / ``name`` / ``arguments`` 字段。

    Returns:
        可直接放入 assistant message 的 tool_calls 列表；参数若非字符串，
        会通过 ``json.dumps`` 序列化保持 OpenAI 协议兼容。

    Raises:
        KeyError: 当某个 tool call 缺少必需字段 ``id`` / ``name`` 时抛出。
    """

    tool_calls_payload: list[ToolCallPayload] = []
    for tc in ordered_tool_calls:
        raw_args = tc.get("arguments", "")
        if not isinstance(raw_args, str):
            raw_args = json.dumps(raw_args)
        tool_calls_payload.append({
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": raw_args,
            },
        })
    return tool_calls_payload


def _serialize_tool_results_for_llm(
    ordered_tool_calls: List[Dict[str, Any]],
    *,
    budget_remaining: int,
) -> list[tuple[Dict[str, Any], str]]:
    """序列化工具调用结果为文本，供注入到 tool message。

    Args:
        ordered_tool_calls: 已按 index_in_iteration 排序的工具调用数据列表。
        budget_remaining: 当前 run 剩余工具调用轮次，会透传给 ``project_for_llm``
            用于注入"剩余轮次"信号。

    Returns:
        ``(tool_call_data, serialized_text)`` 对；若 ``result`` 已是字符串则原样
        保留，若为 None 则对应空字符串，否则通过 ``project_for_llm`` 投影后再
        ``json.dumps``。

    Raises:
        无。
    """

    serialized_pairs: list[tuple[Dict[str, Any], str]] = []
    for tc in ordered_tool_calls:
        tool_result = tc.get("result")
        if tool_result is None:
            serialized_pairs.append((tc, ""))
            continue
        if not isinstance(tool_result, str):
            tool_result = json.dumps(
                project_for_llm(tool_result, budget=budget_remaining),
                ensure_ascii=False,
            )
        serialized_pairs.append((tc, tool_result))
    return serialized_pairs


def _compute_predictive_budget_stats(
    budget_state: ContextBudgetState,
    serialized_pairs: list[tuple[Dict[str, Any], str]],
) -> tuple[int, int, int]:
    """估算工具结果注入后的 prompt tokens 使用量。

    Args:
        budget_state: 当前上下文预算状态，需包含最新 ``current_prompt_tokens``
            和 ``latest_completion_tokens``。
        serialized_pairs: 序列化后的 ``(tool_call, text)`` 对列表。

    Returns:
        ``(total_result_chars, estimated_injection_tokens, projected_tokens)``
        三元组，分别表示工具结果总字符数、估算注入 tokens 数以及注入后预计的
        总 prompt tokens（已加上 ``PREDICTIVE_OVERHEAD_TOKENS``）。

    Raises:
        无。
    """

    total_result_chars = sum(len(s) for _, s in serialized_pairs)
    estimated_injection_tokens = ToolResultBudgetCapper.estimate_chars_to_tokens(total_result_chars)
    projected_tokens = (
        budget_state.current_prompt_tokens
        + budget_state.latest_completion_tokens
        + estimated_injection_tokens
        + PREDICTIVE_OVERHEAD_TOKENS
    )
    return total_result_chars, estimated_injection_tokens, projected_tokens


def _normalize_trace_identity(trace_identity: Optional[AgentTraceIdentity]) -> dict[str, str]:
    """规范化 trace 身份元数据。

    Args:
        trace_identity: 调用方传入的 trace 身份元数据。

    Returns:
        只包含允许字段的规范化字典。

    Raises:
        无。
    """

    if trace_identity is None:
        return {}
    return trace_identity.to_metadata()


@dataclass
class AgentRunningConfig:
    """Agent 运行时控制参数。"""

    # 单次 run 最多允许的工具调用轮次；16 轮足够覆盖绝大多数财报分析任务
    # （实测 p99 在 10 轮以内），同时防止 LLM 进入无限工具循环。
    max_iterations: int = 16
    fallback_mode: Literal["force_answer", "raise_error"] = "force_answer"
    fallback_prompt: Optional[str] = DEFAULT_FALLBACK_PROMPT
    duplicate_tool_hint_prompt: Optional[str] = DEFAULT_DUPLICATE_TOOL_HINT_PROMPT
    continuation_prompt: Optional[str] = DEFAULT_CONTINUATION_PROMPT
    compaction_summary_header: str = DEFAULT_COMPACTION_SUMMARY_HEADER
    compaction_summary_instruction: str = DEFAULT_COMPACTION_SUMMARY_INSTRUCTION
    # 连续失败 2 批即终止——连续失败通常意味着工具本身不可用或参数系统性错误，
    # 继续重试只会浪费 token。
    max_consecutive_failed_tool_batches: int = 2
    # 同一工具+参数组合重复调用 2 次即触发 hard stop——第 1 次 hint 提醒，
    # 第 2 次强制终止，避免 LLM 陷入"反复调相同工具"的死循环。
    max_duplicate_tool_calls: int = 2
    # 上下文预算治理参数（max_context_tokens=0 表示禁用）
    max_context_tokens: int = 0
    budget_soft_limit_ratio: float = 0.75
    budget_hard_limit_ratio: float = 0.90
    max_continuations: int = 3
    max_compactions: int = 3

    def __post_init__(self) -> None:
        """校验运行时参数约束。

        Raises:
            ValueError: 当 max_compactions 大于 max_iterations 时抛出。
        """

        if self.max_compactions > self.max_iterations:
            raise ValueError(
                f"max_compactions ({self.max_compactions}) 不能大于 "
                f"max_iterations ({self.max_iterations})，"
                "否则 context overflow 重试可能绕过迭代预算限制"
            )


class AsyncAgent:
    """
    异步 Agent - 支持 streaming 和工具调用
    
    使用示例：
    
    ```python
    # 创建 Agent
    agent = AsyncAgent(runner=runner, running_config=running_config)
    
    # Streaming 模式（默认）
    async for event in agent.run("分析一下 Tesla 2024 Q3 财报", system_prompt="你是一个助手"):
        if event.type == EventType.CONTENT_DELTA:
            print(event.data, end="", flush=True)
        elif event.type == EventType.TOOL_CALL_START:
            print(f"\\n[调用工具: {event.data['name']}]")
    
    # 非 Streaming 模式
    result = await agent.run_and_wait("分析一下 Tesla 2024 Q3 财报", system_prompt="你是一个助手")
    print(result.content)
    ```
    """
    
    def __init__(
        self,
        runner: AsyncRunner,
        *,
        tool_executor: Optional[ToolExecutor] = None,
        tool_trace_recorder_factory: Optional[ToolTraceRecorderFactory] = None,
        running_config: Optional[AgentRunningConfig] = None,
        trace_identity: Optional[AgentTraceIdentity] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ):
        """
        Args:
            runner: 异步 Runner（AsyncCliRunner 或 AsyncOpenAIRunner）
            tool_executor: 工具执行器（自动通过 get_schemas() 获取工具定义）
            tool_trace_recorder_factory: 工具调用追踪 recorder 工厂（可选）。
            running_config: Agent 运行配置（AgentRunningConfig），包含：
                - max_iterations: 最大迭代次数（默认 16）
                - fallback_mode: 超限处理模式（默认 "force_answer"）
                    * "force_answer": 强制生成答案（生产环境推荐）
                    * "raise_error": 抛出错误（调试模式）
                - fallback_prompt: 超限时的自定义提示（仅 force_answer 模式有效）
                - duplicate_tool_hint_prompt: 检测到重复调用后给模型的软提醒提示
                - max_consecutive_failed_tool_batches: 连续失败工具批次上限（默认 2）
                - max_duplicate_tool_calls: 同一工具“无信息增量”的连续重复调用上限（默认 2）
        """
        self.runner = runner
        self.tool_executor = tool_executor
        self.tool_trace_recorder_factory = tool_trace_recorder_factory
        self.running_config = running_config or AgentRunningConfig()
        self.trace_identity = _normalize_trace_identity(trace_identity)
        Log.verbose(f"Agent 最大迭代次数设置为: {self.running_config.max_iterations}", module=MODULE)
        self.fallback_mode = self.running_config.fallback_mode
        self.fallback_prompt = self.running_config.fallback_prompt or DEFAULT_FALLBACK_PROMPT
        self.duplicate_tool_hint_prompt = (
            self.running_config.duplicate_tool_hint_prompt or DEFAULT_DUPLICATE_TOOL_HINT_PROMPT
        )
        Log.verbose(
            (
                "Agent 降级模式设置为: "
                f"{self.fallback_mode}, "
                f"fallback_prompt='{self.fallback_prompt}', "
                f"duplicate_tool_hint_prompt='{self.duplicate_tool_hint_prompt}'"
            ),
            module=MODULE,
        )
        self.max_consecutive_failed_tool_batches = max(
            1,
            self.running_config.max_consecutive_failed_tool_batches,
        )
        self.max_duplicate_tool_calls = max(1, self.running_config.max_duplicate_tool_calls)
        self.cancellation_token = cancellation_token
        self._active_run_id: Optional[str] = None
        self._run_guard_lock = Lock()

    def _acquire_run_slot(self, run_id: str) -> None:
        """申请运行槽位，禁止同一 Agent 实例并发运行。

        Args:
            run_id: 当前运行 ID。

        Returns:
            无。

        Raises:
            RuntimeError: 当前实例已有进行中的 run/run_and_wait 时抛出。
        """
        with self._run_guard_lock:
            if self._active_run_id is not None:
                raise RuntimeError(
                    f"AsyncAgent 不支持并发运行：active_run_id={self._active_run_id}，incoming_run_id={run_id}"
                )
            self._active_run_id = run_id

    def _release_run_slot(self, run_id: str) -> None:
        """释放运行槽位。

        Args:
            run_id: 当前运行 ID。

        Returns:
            无。

        Raises:
            无。
        """
        with self._run_guard_lock:
            if self._active_run_id == run_id:
                self._active_run_id = None
    
    async def run(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        session_id: str | None = None,
        stream: bool = True,
        run_id: str | None = None,
        **extra_payloads,
    ) -> AsyncIterator[StreamEvent]:
        """执行单轮 Agent 调用。

        Args:
            prompt: 用户输入。
            system_prompt: 本次调用使用的系统提示词。
            session_id: 会话标识；为空时默认使用本次运行的 `run_id`。
            stream: 是否启用 streaming（默认 `True`）。
            **extra_payloads: 透传给底层 Runner 的额外参数。

        Returns:
            异步事件流。

        Raises:
            ValueError: `extra_payloads` 包含内部保留字段时抛出。
            RuntimeError: 同一 Agent 实例并发运行时抛出。
        """
        _validate_extra_payload_keys(extra_payloads)
        messages = _build_messages_from_prompt(prompt=prompt, system_prompt=system_prompt)
        async for event in self.run_messages(
            messages,
            session_id=session_id,
            stream=stream,
            run_id=run_id,
            **extra_payloads,
        ):
            yield event

    async def run_messages(
        self,
        messages: List[AgentMessage],
        *,
        session_id: str | None = None,
        stream: bool = True,
        run_id: str | None = None,
        **extra_payloads,
    ) -> AsyncIterator[StreamEvent]:
        """执行 Agent（调用方直接传入 messages，用于多轮会话）。

        Args:
            messages: 调用方维护的消息列表。该列表会在运行中被原地追加消息。
            session_id: 会话标识；为空时默认使用本次运行的 `run_id`。
            stream: 是否启用 streaming（默认 True）。
            **extra_payloads: 透传给底层 Runner 的额外参数。

        Yields:
            StreamEvent: 运行事件流。

        Raises:
            ValueError: `extra_payloads` 包含内部保留字段时抛出。
            RuntimeError: 同一 Agent 实例并发运行时抛出。
        """

        effective_run_id = str(run_id or "").strip() or f"run_{uuid.uuid4().hex[:8]}"
        _validate_extra_payload_keys(extra_payloads)
        call_payloads = dict(extra_payloads)
        effective_session_id = str(session_id or effective_run_id)
        trace_recorder = (
            self.tool_trace_recorder_factory.create_recorder(
                run_id=effective_run_id,
                session_id=effective_session_id,
                agent_metadata=self.trace_identity,
            )
            if self.tool_trace_recorder_factory is not None
            else None
        )
        self._acquire_run_slot(effective_run_id)
        if self.tool_executor is not None:
            self.tool_executor.clear_cursors()
        Log.verbose(f"[{effective_run_id}] 开始运行 Agent，消息数={len(messages)}", module=MODULE)
        try:
            async for event in self._run_loop(
                messages,
                stream=stream,
                run_id=effective_run_id,
                session_id=effective_session_id,
                trace_recorder=trace_recorder,
                **call_payloads,
            ):
                yield event
        finally:
            await self.runner.close()
            if trace_recorder is not None:
                trace_recorder.close()
            self._release_run_slot(effective_run_id)

    async def _run_loop(
        self,
        messages: List[AgentMessage],
        *,
        stream: bool,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        trace_recorder: Optional[ToolTraceRecorder] = None,
        **extra_payloads,
    ) -> AsyncIterator[StreamEvent]:
        """统一推理循环（供 streaming / non-streaming 复用）。

        .. note::

            本方法约 550 行，承担了 tool batch / continuation / compaction /
            duplicate detection / force answer 全部逻辑。已知需要拆分为更小的
            子方法（_handle_tool_batch / _handle_continuation / _handle_compaction 等），
            但因改动面大、与 AsyncRunner 交互紧密，暂不在常规修复批次中处理。
            参见 code-review H1。
        """
        run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        session_id = session_id or run_id
        duplicate_call_guard = DuplicateCallGuard(
            max_duplicate_tool_calls=self.max_duplicate_tool_calls,
        )

        # 上下文预算状态（max_context_tokens=0 时不生效）
        budget_state = ContextBudgetState(
            max_context_tokens=self.running_config.max_context_tokens,
            soft_limit_ratio=self.running_config.budget_soft_limit_ratio,
            hard_limit_ratio=self.running_config.budget_hard_limit_ratio,
        )

        # 无状态设计：每次 run 时设置工具
        if self.tool_executor:
            self.runner.set_tools(self.tool_executor)
        
        iteration = 0
        consecutive_failed_tool_batches = 0
        # iteration_counter 单调递增，不受 context_overflow 回退影响，保证 iteration_id 全局唯一。
        iteration_counter = 0
        iteration_id = f"{run_id}_iteration_0"
        # 跨轮次累积内容，确保续写场景下 final_answer_event 包含完整内容
        accumulated_content_parts: List[str] = []
        call_payloads = dict(extra_payloads)
        
        while iteration < self.running_config.max_iterations:
            # 协作式取消检查：每轮迭代起始时检查取消令牌
            self._raise_if_cancelled()

            iteration += 1
            iteration_counter += 1
            iteration_id = f"{run_id}_iteration_{iteration_counter}"
            yield self._annotate_event(
                iteration_start_event(iteration=iteration, run_id=run_id),
                run_id=run_id, iteration_id=iteration_id,
            )
            if stream:
                Log.debug(f"[{iteration_id}] 开始第 {iteration} 次 agent iteration，消息数={len(messages)}", module=MODULE)
            else:
                Log.debug(f"[{iteration_id}] 非流式第 {iteration} 次 agent iteration，消息数={len(messages)}", module=MODULE)
            
            content_buffer: List[str] = []
            content_complete_seen = False
            content_complete_text: Optional[str] = None
            reasoning_content: Optional[str] = None
            done_event_seen = False
            done_event_summary: Dict[str, Any] = {}
            tool_calls_batch_done_seen = False
            tool_calls_data: Dict[str, Dict] = {}  # tool_call_id → tool_call data
            early_exit_reason: Optional[str] = None
            early_exit_error_type: Optional[str] = None
            duplicate_hint_tool_name: Optional[str] = None
            context_overflow_handled = False
            iteration_tool_schemas: List[Dict[str, Any]] = []

            # 主动预算检查：超过软阈值时在调用前压缩，避免触发 400
            if (
                budget_state.is_over_soft_limit
                and budget_state.compaction_count < self.running_config.max_compactions
            ):
                proactive_msg = (
                    f"⚠️ prompt tokens ({budget_state.current_prompt_tokens}) "
                    f"超过软阈值 ({budget_state.soft_limit_tokens})，主动压缩消息..."
                )
                yield self._annotate_event(
                    warning_event(proactive_msg),
                    run_id=run_id, iteration_id=iteration_id,
                )
                Log.warn(f"[{iteration_id}] {proactive_msg}", module=MODULE)
                messages, actually_compacted = _compact_messages(
                    messages,
                    summary_header=self.running_config.compaction_summary_header,
                    summary_instruction=self.running_config.compaction_summary_instruction,
                )
                if actually_compacted:
                    budget_state.compaction_count += 1

            iteration_tool_schemas = self._get_registered_tool_schemas()
            if trace_recorder is not None:
                trace_recorder.start_iteration(
                    iteration_id=iteration_id,
                    model_input_messages=copy.deepcopy(messages),
                    tool_schemas=iteration_tool_schemas,
                )
            call_payloads["trace_context"] = {
                "run_id": run_id,
                "iteration_id": iteration_id,
            }
            async for event in self.runner.call(
                messages=messages,
                stream=stream,
                **call_payloads,
            ):
                if event.type == EventType.CONTENT_DELTA:
                    content_buffer.append(event.data)
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                
                elif event.type == EventType.REASONING_DELTA:
                    # 推理增量（thinking 模式思维链）—— 透传给调用者供 UI 展示
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                
                elif event.type == EventType.TOOL_CALL_START:
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                
                elif event.type == EventType.TOOL_CALL_DELTA:
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                
                elif event.type == EventType.TOOL_CALL_DISPATCHED:
                    if trace_recorder is not None:
                        trace_recorder.on_tool_dispatched(iteration_id=iteration_id, payload=event.data)
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                
                elif event.type == EventType.TOOL_CALL_RESULT:
                    if not isinstance(event.data, dict):
                        Log.warn(
                            f"[{iteration_id}] TOOL_CALL_RESULT 事件 data 非 dict，已跳过: type={type(event.data).__name__}",
                            module=MODULE,
                        )
                        continue
                    raw_tool_call_id = event.data.get("id")
                    if not isinstance(raw_tool_call_id, str) or not raw_tool_call_id:
                        Log.warn(
                            f"[{iteration_id}] TOOL_CALL_RESULT 事件缺少有效 id，已跳过",
                            module=MODULE,
                        )
                        continue
                    tool_call_id = raw_tool_call_id
                    tool_calls_data.setdefault(tool_call_id, {})["id"] = tool_call_id
                    tool_calls_data[tool_call_id]["name"] = event.data.get("name", "")
                    tool_calls_data[tool_call_id]["arguments"] = event.data.get("arguments", {})
                    tool_calls_data[tool_call_id]["index_in_iteration"] = event.data.get("index_in_iteration", 0)
                    tool_calls_data[tool_call_id]["result"] = event.data.get("result")
                    tool_name = tool_calls_data[tool_call_id]["name"]
                    tool_args = tool_calls_data[tool_call_id]["arguments"]
                    result = tool_calls_data[tool_call_id].get("result", {})
                    dup_call_spec: DupCallSpec | None = (
                        cast(DupCallSpec | None, self.tool_executor.get_dup_call_spec(tool_name))
                        if self.tool_executor is not None
                        else None
                    )
                    duplicate_decision = duplicate_call_guard.evaluate(
                        tool_name=tool_name,
                        arguments=tool_args,
                        result=result,
                        spec=dup_call_spec,
                    )
                    if duplicate_decision.emit_hint and early_exit_reason is None:
                        duplicate_hint_tool_name = duplicate_decision.hint_tool_name or tool_name
                    if duplicate_decision.hard_stop and early_exit_reason is None:
                        early_exit_reason = duplicate_decision.reason
                        early_exit_error_type = "tool_call_duplicate"

                    if trace_recorder is not None:
                        trace_recorder.on_tool_result(iteration_id=iteration_id, payload=event.data)
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)

                elif event.type == EventType.TOOL_CALLS_BATCH_READY:
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)

                elif event.type == EventType.TOOL_CALLS_BATCH_DONE:
                    tool_calls_batch_done_seen = True
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                
                elif event.type == EventType.CONTENT_COMPLETE:
                    content_complete_seen = True
                    content_complete_text = event.data
                    # 提取 reasoning_content（thinking 模式下由 Runner 通过 metadata 传递）
                    rc = event.metadata.get("reasoning_content") if event.metadata else None
                    if rc:
                        reasoning_content = rc
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                
                elif event.type == EventType.DONE:
                    done_event_seen = True
                    done_event_summary = event.data if isinstance(event.data, dict) else {}
                    # 更新预算状态
                    usage = done_event_summary.get("usage")
                    if usage and isinstance(usage, dict):
                        budget_state.record_usage(usage)
                        if trace_recorder is not None:
                            trace_recorder.record_iteration_usage(
                                iteration_id=iteration_id,
                                usage=usage,
                                budget_snapshot=_build_tool_trace_budget_snapshot(
                                    budget_state=budget_state,
                                    iteration=iteration,
                                    max_iterations=self.running_config.max_iterations,
                                ),
                            )
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)

                elif event.type == EventType.METADATA:
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                
                elif event.type == EventType.WARNING:
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                
                elif event.type == EventType.ERROR:
                    error_data = event.data if isinstance(event.data, dict) else {}
                    error_meta = event.metadata if isinstance(event.metadata, dict) else {}
                    err_type = error_meta.get("error_type", "")
                    # context_overflow 特殊处理：尝试压缩后重试
                    overflow_exhausted = False
                    if err_type == "context_overflow":
                        if budget_state.compaction_count < self.running_config.max_compactions:
                            messages, actually_compacted = _compact_messages(
                                messages,
                                summary_header=self.running_config.compaction_summary_header,
                                summary_instruction=self.running_config.compaction_summary_instruction,
                            )
                            if actually_compacted:
                                budget_state.compaction_count += 1
                                overflow_msg = (
                                    f"⚠️ 上下文超长（prompt_tokens={budget_state.current_prompt_tokens}），"
                                    f"正在执行第 {budget_state.compaction_count} 次消息压缩..."
                                )
                                yield self._annotate_event(
                                    warning_event(overflow_msg),
                                    run_id=run_id, iteration_id=iteration_id,
                                )
                                Log.warn(f"[{iteration_id}] {overflow_msg}", module=MODULE)
                                context_overflow_handled = True
                                break  # 退出 async for，回到 while 循环重试
                            # 还有配额但消息已压不动
                            overflow_exhausted = True
                        else:
                            # max_compactions 配额已耗尽（含 max_compactions=0）
                            overflow_exhausted = True

                    if overflow_exhausted:
                        # 区分“真实 context 超长（首次未尝试压缩）”与“压缩策略已不可用”：
                        # 不可用包括 1) 配额已耗尽 2) 配额未耗尽但 _compact_messages 返回未压缩。
                        exhausted_metadata = dict(error_meta)
                        exhausted_metadata["error_type"] = "context_overflow_exhausted"
                        event = StreamEvent(
                            type=event.type,
                            data=event.data,
                            metadata=exhausted_metadata,
                        )

                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                    if not error_data.get("recoverable", False):
                        if trace_recorder is not None:
                            trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration)
                        return
                
                else:
                    Log.warn(f"[{iteration_id}] 未知事件类型: {event}", module=MODULE)
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
            
            # context_overflow 压缩后重试：不计入迭代次数（但 iteration_counter 已递增，保证下一轮 iteration_id 唯一）
            if context_overflow_handled:
                if trace_recorder is not None:
                    trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration_counter)
                iteration -= 1
                continue

            final_content = content_complete_text if content_complete_text is not None else "".join(content_buffer)

            if tool_calls_batch_done_seen:
                assistant_content = final_content or None
                ordered_tool_calls = sorted(
                    tool_calls_data.values(),
                    key=lambda tc: tc.get("index_in_iteration", 0),
                )
                if not ordered_tool_calls:
                    Log.error(f"[{iteration_id}] 收到 TOOL_CALLS_BATCH_DONE 但未收集到任何工具调用", module=MODULE)
                    error = error_event(
                        "Tool batch done received without any tool calls",
                        recoverable=False,
                        error_type="tool_calls_missing",
                    )
                    yield self._annotate_event(error, run_id=run_id, iteration_id=iteration_id)
                    if trace_recorder is not None:
                        trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration)
                    return

                if any(is_tool_success(tc.get("result")) for tc in ordered_tool_calls):
                    consecutive_failed_tool_batches = 0
                else:
                    consecutive_failed_tool_batches += 1
                    if (
                        consecutive_failed_tool_batches >= self.max_consecutive_failed_tool_batches
                        and early_exit_reason is None
                    ):
                        early_exit_reason = self._build_failed_tool_batches_reason(
                            consecutive_failed_tool_batches,
                        )
                        early_exit_error_type = "consecutive_failed_tool_batches"

                tool_calls_payload = _build_tool_calls_payload(ordered_tool_calls)

                assistant_message = build_assistant_chat_message(
                    content=assistant_content,
                    tool_calls=tool_calls_payload,
                )
                # 这里保留 reasoning_content 不是随意污染通用 OpenAI messages。
                # DeepSeek 思考模式的工具调用要求在同一用户问题的后续子请求中回传
                # reasoning_content，否则会返回 400；MiMo 文档也建议在思考模式下的
                # 多轮工具调用里保留历史 reasoning_content；Qwen 在开启
                # preserve_thinking 时也会消费 assistant message 中的 reasoning_content。
                # 因此这里按 provider 兼容约束保留该字段，但它只应用于当前 tool loop
                # 的后续请求，不代表新的用户问题也应无条件继续携带历史 thinking。
                if reasoning_content:
                    assistant_message = build_assistant_chat_message(
                        content=assistant_content,
                        tool_calls=tool_calls_payload,
                        reasoning_content=reasoning_content,
                    )
                messages.append(assistant_message)
                
                # Pass 1: 序列化所有工具结果
                budget_remaining = max(0, self.running_config.max_iterations - iteration)
                serialized_pairs = _serialize_tool_results_for_llm(
                    ordered_tool_calls,
                    budget_remaining=budget_remaining,
                )

                # Pass 1.5: 预测性预算检查 —— 估算工具结果注入后是否会超过硬阈值
                if budget_state.is_budget_enabled:
                    total_result_chars, estimated_injection_tokens, projected_tokens = (
                        _compute_predictive_budget_stats(budget_state, serialized_pairs)
                    )
                    if projected_tokens > budget_state.hard_limit_tokens:
                        serialized_pairs, was_capped = ToolResultBudgetCapper.cap_results_for_budget(
                            serialized_pairs, budget_state,
                        )
                        if was_capped:
                            cap_msg = (
                                f"⚠️ 预测性截断：工具结果 {total_result_chars} chars "
                                f"(≈{estimated_injection_tokens} tokens) 注入后 "
                                f"预计 {projected_tokens} tokens > "
                                f"硬阈值 {budget_state.hard_limit_tokens}，"
                                f"已截断至 soft_limit 预算范围"
                            )
                            yield self._annotate_event(
                                warning_event(cap_msg),
                                run_id=run_id, iteration_id=iteration_id,
                            )
                            Log.warn(f"[{iteration_id}] {cap_msg}", module=MODULE)

                # Pass 2: 注入到 messages
                # OpenAI 协议要求 assistant.tool_calls 与 role=tool 消息一一对应，
                # 不能因为序列化结果为空字符串就跳过；对空结果统一注入占位文案，
                # 既保证协议完整，也便于模型识别“工具返回为空”。
                for tc, result_str in serialized_pairs:
                    tool_call_id = str(tc["id"])
                    messages.append(
                        build_tool_chat_message(
                            tool_call_id=tool_call_id,
                            content=result_str if result_str else "(empty result)",
                        )
                    )

                if duplicate_hint_tool_name and not early_exit_reason:
                    duplicate_warning = warning_event(
                        f"⚠️ 检测到重复工具调用: {duplicate_hint_tool_name}，已注入防重复提示并继续推理"
                    )
                    yield self._annotate_event(duplicate_warning, run_id=run_id, iteration_id=iteration_id)
                    messages.append(
                        build_user_chat_message(
                            self._build_duplicate_tool_hint_prompt(duplicate_hint_tool_name)
                        )
                    )
                    Log.warn(
                        (
                            f"[{iteration_id}] ⚠️ 检测到重复工具调用: {duplicate_hint_tool_name}，"
                            "已注入提示引导模型复用已有结果"
                        ),
                        module=MODULE,
                    )

                if early_exit_reason:
                    duplicate_log_reason = (
                        f"⚠️ {early_exit_reason}"
                        if early_exit_error_type == "tool_call_duplicate"
                        else early_exit_reason
                    )
                    if self.fallback_mode == "raise_error":
                        Log.error(f"[{iteration_id}] {duplicate_log_reason}", module=MODULE)
                        error = error_event(
                            early_exit_reason,
                            recoverable=False,
                            error_type=early_exit_error_type or "tool_call_early_exit",
                        )
                        yield self._annotate_event(error, run_id=run_id, iteration_id=iteration_id)
                        if trace_recorder is not None:
                            trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration)
                        return

                    if self.fallback_mode == "force_answer":
                        if trace_recorder is not None:
                            trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration)
                        async for fallback_event in self._run_force_answer(
                            messages,
                            stream=stream,
                            run_id=run_id,
                            iteration_id=f"{iteration_id}_fallback",
                            session_id=session_id,
                            trace_recorder=trace_recorder,
                            warning_message=f"{duplicate_log_reason}，将基于现有上下文生成最终答案",
                            log_message=f"[{iteration_id}] {duplicate_log_reason}，进入降级模式生成最终答案。",
                            **extra_payloads,
                        ):
                            yield fallback_event
                        return

                if trace_recorder is not None:
                    trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration)
                continue

            if tool_calls_data and not tool_calls_batch_done_seen:
                Log.error(f"[{iteration_id}] 已派发工具调用但未收到 TOOL_CALLS_BATCH_DONE", module=MODULE)
                error = error_event(
                    "Tool calls dispatched but batch done not received",
                    recoverable=False,
                    error_type="tool_calls_batch_missing",
                )
                yield self._annotate_event(error, run_id=run_id, iteration_id=iteration_id)
                if trace_recorder is not None:
                    trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration)
                return

            if (content_complete_seen or done_event_seen) and not tool_calls_data:
                # 这里刻意把“无工具调用的本轮”视为非失败工具批次，并清零连续失败计数。
                # 纯文本回答、截断续写等路径都依赖这条稳定语义；而 tool_call 协议错误
                # 会在更早的 ERROR 分支直接终止，不会走到这个 reset 分支。
                consecutive_failed_tool_batches = 0
                # 截断续写：finish_reason=length 时自动续写而非直接终止
                is_truncated = done_event_summary.get("truncated", False)
                is_filtered = bool(done_event_summary.get("content_filtered", False))
                finish_reason = str(done_event_summary.get("finish_reason") or "").strip() or None
                if (
                    is_truncated
                    and budget_state.continuation_count < self.running_config.max_continuations
                ):
                    budget_state.continuation_count += 1
                    continuation_msg = (
                        f"⚠️ 回答被截断（finish_reason=length），"
                        f"正在进行第 {budget_state.continuation_count} 次续写..."
                    )
                    yield self._annotate_event(
                        warning_event(continuation_msg),
                        run_id=run_id, iteration_id=iteration_id,
                    )
                    Log.warn(f"[{iteration_id}] {continuation_msg}", module=MODULE)
                    # 将截断的部分内容累积到跨轮缓冲
                    if final_content:
                        accumulated_content_parts.append(final_content)
                    # 将截断的部分内容作为 assistant message
                    if final_content:
                        messages.append(build_assistant_chat_message(content=final_content))
                    messages.append(
                        build_user_chat_message(
                            self.running_config.continuation_prompt or DEFAULT_CONTINUATION_PROMPT
                        )
                    )
                    # 续写前如果超过软阈值则压缩
                    if (
                        budget_state.is_over_soft_limit
                        and budget_state.compaction_count < self.running_config.max_compactions
                    ):
                        messages, actually_compacted = _compact_messages(
                            messages,
                            summary_header=self.running_config.compaction_summary_header,
                            summary_instruction=self.running_config.compaction_summary_instruction,
                        )
                        if actually_compacted:
                            budget_state.compaction_count += 1
                            cont_compact_msg = (
                                f"⚠️ 续写前 prompt tokens ({budget_state.current_prompt_tokens}) "
                                f"超过软阈值，执行第 {budget_state.compaction_count} 次消息压缩"
                            )
                            yield self._annotate_event(
                                warning_event(cont_compact_msg),
                                run_id=run_id, iteration_id=iteration_id,
                            )
                            Log.warn(f"[{iteration_id}] {cont_compact_msg}", module=MODULE)
                    if trace_recorder is not None:
                        trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration)
                    continue  # 回到 while 循环进行下一轮

                # 拼接所有轮次的内容（续写场景下 accumulated_content_parts 含前序内容）
                full_content = "".join(accumulated_content_parts) + final_content
                if is_filtered:
                    filtered_msg = "⚠️ 回答触发内容过滤，以下结果可能不完整。"
                    yield self._annotate_event(
                        warning_event(filtered_msg),
                        run_id=run_id, iteration_id=iteration_id,
                    )
                    Log.warn(f"[{iteration_id}] {filtered_msg}", module=MODULE)
                self._raise_if_cancelled()
                final_event = final_answer_event(
                    full_content,
                    degraded=is_filtered,
                    filtered=is_filtered,
                    finish_reason=finish_reason,
                )
                if trace_recorder is not None:
                    trace_recorder.record_final_response(
                        iteration_id=iteration_id,
                        content=full_content,
                        degraded=is_filtered,
                        filtered=is_filtered,
                        finish_reason=finish_reason,
                    )
                yield self._annotate_event(final_event, run_id=run_id, iteration_id=iteration_id)
                if trace_recorder is not None:
                    trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration)
                return
            
        if iteration >= self.running_config.max_iterations:
            if self.fallback_mode == "raise_error":
                Log.error(
                    f"[{iteration_id}] ⚠️ 达到最大迭代次数 {self.running_config.max_iterations}，停止运行",
                    module=MODULE,
                )
                error = error_event(
                    f"达到最大迭代次数 {self.running_config.max_iterations}",
                    recoverable=False,
                    error_type="max_iterations",
                )
                yield self._annotate_event(error, run_id=run_id, iteration_id=iteration_id)
                if trace_recorder is not None:
                    trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration)
                return

            if self.fallback_mode == "force_answer":
                if trace_recorder is not None:
                    trace_recorder.finish_iteration(iteration_id=iteration_id, iteration_index=iteration)
                async for fallback_event in self._run_force_answer(
                    messages,
                    stream=stream,
                    run_id=run_id,
                    iteration_id=f"{iteration_id}_fallback",
                    session_id=session_id,
                    trace_recorder=trace_recorder,
                    warning_message=f"⚠️ 已达到最大工具调用次数 {self.running_config.max_iterations}，将基于现有上下文生成最终答案",
                    log_message=f"[{iteration_id}] ⚠️ 已达到最大工具调用次数 {self.running_config.max_iterations}，进入降级模式生成最终答案。",
                    **extra_payloads,
                ):
                    yield fallback_event
                return

    def _build_duplicate_tool_hint_prompt(self, tool_name: str) -> str:
        """构建重复调用软干预提示词。

        Args:
            tool_name: 触发重复调用的工具名称。

        Returns:
            供下一轮注入到消息上下文的提示词文本。

        Raises:
            无。
        """
        template = self.duplicate_tool_hint_prompt
        return template.replace("{{tool_name}}", tool_name or "unknown_tool")

    def _build_failed_tool_batches_reason(self, failure_count: int) -> str:
        """构建连续失败工具批次的提前退出原因。

        Args:
            failure_count: 当前连续失败工具批次数。

        Returns:
            可直接用于日志与错误事件的原因描述。

        Raises:
            无。
        """

        return f"连续 {failure_count} 轮工具批次全部失败，停止继续调用工具"

    def _annotate_event(self, event: StreamEvent, *, run_id: str, iteration_id: str) -> StreamEvent:
        """为事件补充执行级元数据。

        Args:
            event: 原始事件。
            run_id: Host 执行 ID。
            iteration_id: 当前 agent iteration 的标识。

        Returns:
            补充了元数据的事件。

        Raises:
            无。
        """

        metadata = dict(event.metadata) if event.metadata else {}
        metadata.setdefault("run_id", run_id)
        metadata.setdefault("iteration_id", iteration_id)
        if event.type in {
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_DELTA,
            EventType.TOOL_CALL_DISPATCHED,
            EventType.TOOL_CALL_RESULT,
        }:
            tool_call_id = event.data.get("id") if isinstance(event.data, dict) else None
            if tool_call_id:
                metadata.setdefault("tool_call_id", tool_call_id)
        event.metadata = metadata
        return event

    def _raise_if_cancelled(self) -> None:
        """在提交关键外部事实前执行协作式取消检查。

        Args:
            无。

        Returns:
            无。

        Raises:
            CancelledError: 当前 agent run 已被取消时抛出。
        """

        if self.cancellation_token is not None:
            self.cancellation_token.raise_if_cancelled()

    async def _run_force_answer(
        self,
        messages: List[AgentMessage],
        *,
        stream: bool,
        run_id: str,
        iteration_id: str,
        warning_message: str,
        log_message: str,
        session_id: str | None = None,
        trace_recorder: Optional[ToolTraceRecorder] = None,
        **extra_payloads,
    ) -> AsyncIterator[StreamEvent]:
        warning = warning_event(warning_message)
        yield self._annotate_event(warning, run_id=run_id, iteration_id=iteration_id)
        Log.warn(log_message, module=MODULE)
        messages.append(build_user_chat_message(self.fallback_prompt))
        Log.debug(f"[{iteration_id}] 进入降级模式，追加 fallback prompt 并生成最终答案", module=MODULE)

        if self.tool_executor:
            self.runner.set_tools(None)

        try:
            content_buffer = []
            content_complete_seen = False
            content_complete_text = None
            done_summary: Dict[str, Any] = {}
            call_payloads = dict(extra_payloads)
            call_payloads["trace_context"] = {"run_id": run_id, "iteration_id": iteration_id}
            async for event in self.runner.call(
                messages=messages,
                stream=stream,
                **call_payloads,
            ):
                if event.type == EventType.CONTENT_COMPLETE:
                    content_complete_seen = True
                    content_complete_text = event.data
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                elif event.type == EventType.CONTENT_DELTA:
                    content_buffer.append(event.data)
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                elif event.type == EventType.DONE:
                    # 降级分支也需要捕获截断/过滤信号，透传给 final_answer_event
                    if isinstance(event.data, dict):
                        done_summary = event.data
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                elif event.type == EventType.ERROR:
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)
                    return
                else:
                    yield self._annotate_event(event, run_id=run_id, iteration_id=iteration_id)

            if content_complete_seen or content_buffer:
                final_content = content_complete_text if content_complete_text is not None else "".join(content_buffer)
                self._raise_if_cancelled()
                is_filtered = bool(done_summary.get("content_filtered", False))
                finish_reason = str(done_summary.get("finish_reason") or "").strip() or None
                final_event = final_answer_event(
                    final_content,
                    degraded=True,
                    filtered=is_filtered,
                    finish_reason=finish_reason,
                )
                if trace_recorder is not None:
                    trace_recorder.record_final_response(
                        iteration_id=iteration_id,
                        content=final_content,
                        degraded=True,
                        filtered=is_filtered,
                        finish_reason=finish_reason,
                    )
                yield self._annotate_event(final_event, run_id=run_id, iteration_id=iteration_id)
            else:
                # 降级调用未产生任何内容，显式 yield ERROR 避免静默空结果
                yield self._annotate_event(
                    error_event(
                        "降级模式未产生任何内容",
                        recoverable=False,
                        error_type="force_answer_empty",
                    ),
                    run_id=run_id,
                    iteration_id=iteration_id,
                )
        finally:
            # 恢复工具状态，避免后续（如 run_and_wait 的 warnings 收集等）丢失工具能力
            if self.tool_executor:
                self.runner.set_tools(self.tool_executor)
    

    async def run_and_wait(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        session_id: str | None = None,
        run_id: str | None = None,
        **extra_payloads,
    ) -> "AgentResult":
        """运行并等待完成，聚合完整结果。

        Args:
            prompt: 用户输入。
            system_prompt: 本次调用使用的系统提示词。
            session_id: 会话标识；为空时默认使用本次运行的 `run_id`。
            **extra_payloads: 透传给底层 Runner 的额外参数。

        Returns:
            聚合后的 Agent 结果对象。

        Raises:
            ValueError: `extra_payloads` 包含内部保留字段时抛出。
            RuntimeError: 同一 Agent 实例并发运行时抛出。
        """
        _validate_extra_payload_keys(extra_payloads)
        messages = _build_messages_from_prompt(prompt=prompt, system_prompt=system_prompt)
        tool_calls = []
        errors = []
        warnings: List[str] = []
        final_content = ""
        degraded = False
        filtered = False

        async for event in self.run_messages(
            messages,
            session_id=session_id,
            run_id=run_id,
            # 关键说明：run_and_wait 仅表示“调用方等待完整结果返回”，
            # 不表示 Runner 必须关闭流式输出。这里保持 stream=True，
            # 以复用流式事件路径（含工具调用增量事件）并与 run() 语义对齐。
                stream=True,
                **extra_payloads,
        ):
            if event.type == EventType.FINAL_ANSWER:
                final_content = event.data.get("content", "")
                degraded = event.data.get("degraded", False)
                filtered = bool(event.data.get("filtered", False)) if isinstance(event.data, dict) else False
            elif event.type == EventType.TOOL_CALL_RESULT:
                tool_calls.append(event.data)
            elif event.type == EventType.WARNING:
                msg = event.data.get("message", "") if isinstance(event.data, dict) else str(event.data)
                warnings.append(msg)
            elif event.type == EventType.ERROR:
                errors.append(event.data)

        return AgentResult(
            content=final_content,
            tool_calls=tool_calls,
            errors=errors,
            warnings=warnings,
            messages=messages.copy(),
            degraded=degraded,
            filtered=filtered,
        )

    def _get_registered_tool_schemas(self) -> List[Dict[str, Any]]:
        """返回当前实际注册给模型的原始工具 schema 列表。

        Args:
            无。

        Returns:
            原始工具 schema 列表；不支持工具调用或未注册工具时返回空列表。

        Raises:
            无。
        """

        if self.tool_executor is None:
            return []
        if not self.runner.is_supports_tool_calling():
            return []
        return self.tool_executor.get_schemas()
# ---------- 消息压缩工具函数 ----------


def _find_safe_split_point(messages: List[AgentMessage], target_idx: int) -> int:
    """在 messages 中找到不拆散 tool 消息组的安全切分点。

    assistant(tool_calls=[...]) 和其后续所有 tool 消息构成一个原子组，
    切分点不能落在组内。如果 target_idx 落在组内，则向前调整到组起始位置。

    Args:
        messages: 消息列表。
        target_idx: 期望的切分索引（该位置及之后的消息将被保留）。

    Returns:
        调整后的安全切分索引（<= target_idx）。
    """
    if target_idx <= 0 or target_idx >= len(messages):
        return target_idx
    # 如果 target_idx 指向 tool 消息，说明其前面的 assistant(tool_calls) 被切到了压缩侧
    # 需要向前回溯到该 tool 组的 assistant 消息位置
    idx = target_idx
    while idx > 0 and messages[idx].get("role") == "tool":
        idx -= 1
    # idx 现在要么指向 assistant(tool_calls) 要么指向非 tool 消息
    # 如果是 assistant 且含 tool_calls，则从它开始保留
    if idx < target_idx and messages[idx].get("role") == "assistant" and messages[idx].get("tool_calls"):
        return idx
    return target_idx


def _compact_messages(
    messages: List[AgentMessage],
    recent_keep: int = _COMPACT_RECENT_KEEP,
    summary_header: str = DEFAULT_COMPACTION_SUMMARY_HEADER,
    summary_instruction: str = DEFAULT_COMPACTION_SUMMARY_INSTRUCTION,
) -> Tuple[List[AgentMessage], bool]:
    """规则化压缩消息列表，降低 token 占用。

    保留策略：
    1. system message（首条，如果有）
    2. 首条 user message（任务目标）
    3. 中间消息压缩为单条结构化摘要
    4. 最近 recent_keep 条消息原样保留

    切分时保证 assistant(tool_calls) + 对应 tool 消息的原子性，
    不会将一组工具调用/结果拆散到压缩区和保留区两侧。

    Args:
        messages: 原始消息列表。
        recent_keep: 尾部原样保留的条数。

    Returns:
        (压缩后的新消息列表, 是否实际执行了压缩)。不修改原列表。
    """
    if len(messages) <= recent_keep + 2:
        return list(messages), False

    result: List[AgentMessage] = []
    start_idx = 0

    # 保留连续前导 system 段（static system prompt + episodic memory block 等）。
    # Host 层可能注入多条 system message（例如 build_messages 会追加一条
    # `[Conversation Memory]` block）。这些都属于"框架条件"，不能落入中段被压缩；
    # 否则 episodic memory 会被降级为"摘要之摘要"，丢失 pinned_state / episode
    # title 等结构化信息。
    while start_idx < len(messages) and messages[start_idx].get("role") == "system":
        result.append(messages[start_idx])
        start_idx += 1

    # 保留首条 user message
    first_user_idx = None
    for i in range(start_idx, len(messages)):
        if messages[i].get("role") == "user":
            first_user_idx = i
            break

    if first_user_idx is not None:
        result.append(messages[first_user_idx])
        # 压缩区起始：首条 user 之后，但 system~first_user 之间若有消息也需纳入
        if first_user_idx > start_idx:
            compress_start = start_idx
        else:
            compress_start = first_user_idx + 1
    else:
        compress_start = start_idx

    # 划分需要压缩的中间范围和保留的尾部范围
    raw_recent_start = max(compress_start, len(messages) - recent_keep)
    if first_user_idx is not None:
        # 首条 user 已单独保留；recent tail 不能再从它开始，否则会重复注入任务目标。
        raw_recent_start = max(raw_recent_start, first_user_idx + 1)
    # Bug #1 fix: 调整切分点到 tool 消息组边界，防止拆散 assistant(tool_calls)+tool 配对
    recent_start = _find_safe_split_point(messages, raw_recent_start)
    middle_messages = messages[compress_start:recent_start]
    # 从 middle 中移除首条 user（已单独保留）
    if first_user_idx is not None and compress_start <= first_user_idx < recent_start:
        middle_messages = [
            m for i, m in enumerate(messages[compress_start:recent_start], start=compress_start)
            if i != first_user_idx
        ]

    if middle_messages:
        summary = _build_compaction_summary(
            middle_messages,
            header=summary_header,
            instruction=summary_instruction,
        )
        result.append(build_user_chat_message(summary))

    # 保留最近 recent_keep 条消息
    result.extend(messages[recent_start:])
    return result, True


def _build_compaction_summary(
    messages: List[AgentMessage],
    header: str = DEFAULT_COMPACTION_SUMMARY_HEADER,
    instruction: str = DEFAULT_COMPACTION_SUMMARY_INSTRUCTION,
) -> str:
    """为被压缩的中间消息构建结构化摘要。

    生成通用摘要（不依赖业务类型），记录消息类型统计和已调用的工具名。

    Args:
        messages: 被压缩的消息列表。
        header: 摘要标题行。
        instruction: 摘要尾部指令。

    Returns:
        摘要文本。
    """
    assistant_count = 0
    tool_call_count = 0
    tool_result_count = 0
    user_count = 0
    tool_names: set[str] = set()
    tool_call_names_by_id: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            assistant_count += 1
            tool_calls = msg.get("tool_calls", [])
            tool_call_count += len(tool_calls)
            for tc in tool_calls:
                tool_call_id = str(tc.get("id") or "").strip()
                fn = tc.get("function", {})
                tool_name = str(fn.get("name") or "").strip()
                if tool_name:
                    tool_names.add(tool_name)
                    if tool_call_id:
                        tool_call_names_by_id[tool_call_id] = tool_name
        elif role == "tool":
            tool_result_count += 1
        elif role == "user":
            user_count += 1

    lines = [
        header,
        (
            f"Compacted {len(messages)} history messages "
            f"(assistant={assistant_count}, tool_call={tool_call_count}, "
            f"tool_result={tool_result_count}, user={user_count})."
        ),
    ]
    if tool_names:
        lines.append(f"Tools called: {', '.join(sorted(tool_names))}.")
    tool_result_lines = _summarize_compacted_tool_results(
        messages,
        tool_call_names_by_id=tool_call_names_by_id,
    )
    lines.extend(tool_result_lines)
    lines.append(instruction)
    return "\n".join(lines)


def _summarize_compacted_tool_results(
    messages: List[AgentMessage],
    *,
    tool_call_names_by_id: dict[str, str],
) -> list[str]:
    """提取被压缩区间内的工具结果核心语义摘要。

    Args:
        messages: 被压缩区间内的消息列表。
        tool_call_names_by_id: `tool_call_id -> tool_name` 映射。

    Returns:
        可直接拼入 compaction summary 的文本行列表；没有可摘要内容时返回空列表。

    Raises:
        无。
    """

    summarized_results: list[str] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        summarized_result = _summarize_single_tool_result(
            message,
            tool_call_names_by_id=tool_call_names_by_id,
        )
        if summarized_result is None:
            continue
        summarized_results.append(summarized_result)
        if len(summarized_results) >= _COMPACTION_TOOL_RESULT_MAX_LINES:
            break
    if not summarized_results:
        return []
    return ["Tool result highlights:"] + [f"- {summary}" for summary in summarized_results]


def _summarize_single_tool_result(
    message: AgentMessage,
    *,
    tool_call_names_by_id: dict[str, str],
) -> str | None:
    """为单条 tool message 构建紧凑摘要。

    Args:
        message: 单条工具消息。
        tool_call_names_by_id: `tool_call_id -> tool_name` 映射。

    Returns:
        单行摘要；若消息内容为空且无法提炼语义，则返回 `None`。

    Raises:
        无。
    """

    tool_call_id = str(message.get("tool_call_id") or "").strip()
    tool_name = tool_call_names_by_id.get(tool_call_id, tool_call_id or "unknown_tool")
    parsed_payload = _parse_compaction_tool_payload(message.get("content"))
    status_parts = _build_compaction_tool_status_parts(parsed_payload)
    if not status_parts:
        text_summary = _summarize_compaction_scalar_value(message.get("content"))
        if text_summary is None:
            return None
        return f"{tool_name}: value={text_summary}"
    return f"{tool_name}: {', '.join(status_parts)}"


def _parse_compaction_tool_payload(content: object) -> object:
    """把 tool message 内容解析为便于摘要的对象。

    Args:
        content: 原始 tool message 内容。

    Returns:
        若内容是 JSON 字符串则返回解析后的对象；否则返回原始值或规范化后的字符串。

    Raises:
        无。
    """

    if isinstance(content, str):
        stripped_content = content.strip()
        if not stripped_content:
            return ""
        try:
            return json.loads(stripped_content)
        except ValueError:
            return stripped_content
    return content


def _build_compaction_tool_status_parts(parsed_payload: object) -> list[str]:
    """从工具结果载荷中抽取 `ok/error/value` 关键字段。

    Args:
        parsed_payload: 已解析的工具结果载荷。

    Returns:
        状态片段列表，供上层以逗号拼接。

    Raises:
        无。
    """

    if not isinstance(parsed_payload, dict):
        scalar_summary = _summarize_compaction_scalar_value(parsed_payload)
        return [f"value={scalar_summary}"] if scalar_summary is not None else []

    status_parts: list[str] = []
    ok_value = parsed_payload.get("ok")
    if isinstance(ok_value, bool):
        status_parts.append(f"ok={str(ok_value).lower()}")
    error_summary = _summarize_compaction_scalar_value(
        parsed_payload.get("error"),
        max_chars=_COMPACTION_ERROR_SUMMARY_MAX_CHARS,
    )
    if error_summary is not None:
        status_parts.append(f"error={error_summary}")
    value_summary = _summarize_compaction_scalar_value(
        parsed_payload.get("value"),
        max_chars=_COMPACTION_VALUE_SUMMARY_MAX_CHARS,
    )
    if value_summary is not None:
        status_parts.append(f"value={value_summary}")
    if not status_parts:
        fallback_summary = _summarize_compaction_scalar_value(parsed_payload)
        if fallback_summary is not None:
            status_parts.append(f"value={fallback_summary}")
    return status_parts


def _summarize_compaction_scalar_value(
    value: object,
    *,
    max_chars: int = _COMPACTION_VALUE_SUMMARY_MAX_CHARS,
) -> str | None:
    """把任意值压缩为单行短摘要。

    Args:
        value: 待摘要的值。
        max_chars: 最长字符数。

    Returns:
        单行摘要；若值为空或规范化后为空字符串，则返回 `None`。

    Raises:
        无。
    """

    if value is None:
        return None
    if isinstance(value, str):
        normalized = " ".join(value.split())
    else:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        normalized = " ".join(serialized.split())
    if not normalized:
        return None
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


class AgentResult:
    """Agent 执行结果（非 streaming 模式）。

    Attributes:
        content: 最终回答文本。
        tool_calls: 工具调用记录列表。
        errors: 错误记录列表。
        warnings: 警告消息列表（压缩/续写/截断等治理事件）。
        messages: 完整消息历史。
        degraded: 是否为降级回答。
        filtered: 是否为受过滤完成态。
    """
    
    def __init__(
        self,
        content: str,
        tool_calls: List[Dict],
        errors: List[Dict],
        messages: List[AgentMessage],
        degraded: bool = False,
        filtered: bool = False,
        warnings: Optional[List[str]] = None,
    ):
        self.content = content
        self.tool_calls = tool_calls
        self.errors = errors
        self.warnings = warnings or []
        self.messages = messages
        self.degraded = degraded
        self.filtered = filtered
    
    @property
    def success(self) -> bool:
        """是否成功（无错误）"""
        return len(self.errors) == 0
    
    def __repr__(self) -> str:
        degraded_str = ", degraded=True" if self.degraded else ""
        filtered_str = ", filtered=True" if self.filtered else ""
        warnings_str = f", warnings={len(self.warnings)}" if self.warnings else ""
        return (
            f"AgentResult(content={self.content[:50]}..., "
            f"tool_calls={len(self.tool_calls)}, errors={len(self.errors)}"
            f"{warnings_str}{degraded_str}{filtered_str})"
        )
