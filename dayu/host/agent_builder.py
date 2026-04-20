"""Host 内部的 Agent 构造模块。"""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, cast

from dayu.contracts.agent_execution import AgentCreateArgs
from dayu.contracts.agent_types import AgentTraceIdentity
from dayu.contracts.model_config import (
    CliRunnerParams,
    ModelConfig,
    OpenAICompatibleModelConfig,
    OpenAICompatibleRunnerParams,
    RunnerType,
    ensure_runner_type_enabled,
)
from dayu.engine.async_agent import AgentRunningConfig, AsyncAgent
from dayu.contracts.cancellation import CancellationToken
from dayu.engine.protocols import AsyncRunner, ToolExecutor
from dayu.engine.runner_factory import create_runner
from dayu.engine.tool_trace import ToolTraceRecorderFactory
from dayu.execution.runtime_config import (
    FallbackMode,
    build_agent_running_config_snapshot,
    build_runner_running_config_snapshot,
    normalize_fallback_mode,
)
from dayu.execution.options import ResolvedExecutionOptions


def build_async_runner(
    agent_create_args: AgentCreateArgs,
    *,
    cancellation_token: CancellationToken | None = None,
) -> AsyncRunner:
    """根据 ``AgentCreateArgs`` 构造底层 Runner。

    Args:
        agent_create_args: 已解析完成的 Agent 创建参数。
        cancellation_token: 可选取消令牌。

    Returns:
        底层异步 Runner。

    Raises:
        ValueError: ``runner_type`` 不支持时抛出。
    """

    return create_runner(agent_create_args, cancellation_token=cancellation_token)


def build_agent_running_config(agent_create_args: AgentCreateArgs) -> AgentRunningConfig:
    """根据 ``AgentCreateArgs`` 构造 ``AgentRunningConfig``。

    Args:
        agent_create_args: 已解析完成的 Agent 创建参数。

    Returns:
        Agent 运行配置对象。

    Raises:
        TypeError: 配置字段类型非法时抛出。
    """

    running_config = _build_agent_running_config_from_snapshot(agent_create_args)
    if agent_create_args.max_turns is not None:
        running_config.max_iterations = int(agent_create_args.max_turns)
    if agent_create_args.max_context_tokens is not None:
        running_config.max_context_tokens = int(agent_create_args.max_context_tokens)
    if agent_create_args.max_output_tokens is not None:
        running_config.max_output_tokens = int(agent_create_args.max_output_tokens)
    return running_config


def build_agent_create_args(
    *,
    resolved_execution_options: ResolvedExecutionOptions,
    model_config: ModelConfig,
) -> AgentCreateArgs:
    """从 resolved execution options 与模型配置构造 ``AgentCreateArgs``。"""

    runner_type = ensure_runner_type_enabled(model_config.get("runner_type"))
    runner_params = _build_runner_params(
        runner_type=runner_type,
        model_name=resolved_execution_options.model_name,
        temperature=resolved_execution_options.temperature,
        model_config=model_config,
    )
    agent_running = replace(
        resolved_execution_options.agent_running_config,
        max_context_tokens=int(model_config.get("max_context_tokens") or 0),
        max_output_tokens=int(model_config.get("max_output_tokens") or 0),
    )
    return AgentCreateArgs(
        runner_type=runner_type.value,
        model_name=resolved_execution_options.model_name,
        max_turns=agent_running.max_iterations,
        max_context_tokens=agent_running.max_context_tokens,
        max_output_tokens=agent_running.max_output_tokens,
        temperature=resolved_execution_options.temperature,
        runner_params=runner_params,
        runner_running_config=build_runner_running_config_snapshot(
            resolved_execution_options.runner_running_config
        ),
        agent_running_config=build_agent_running_config_snapshot(agent_running),
    )


def build_async_agent(
    *,
    agent_create_args: AgentCreateArgs,
    tool_executor: ToolExecutor | None = None,
    tool_trace_recorder_factory: ToolTraceRecorderFactory | None = None,
    trace_identity: AgentTraceIdentity | None = None,
    cancellation_token: CancellationToken | None = None,
) -> AsyncAgent:
    """根据 ``AgentCreateArgs`` 构造 ``AsyncAgent``。

    Args:
        agent_create_args: 已解析完成的 Agent 创建参数。
        tool_executor: 工具执行器。
        tool_trace_recorder_factory: 工具追踪 recorder 工厂。
        trace_identity: 追踪身份信息。
        cancellation_token: 取消令牌。

    Returns:
        可执行的 ``AsyncAgent``。

    Raises:
        ValueError: Runner 配置非法时抛出。
    """

    return AsyncAgent(
        runner=build_async_runner(
            agent_create_args,
            cancellation_token=cancellation_token,
        ),
        tool_executor=tool_executor,
        tool_trace_recorder_factory=tool_trace_recorder_factory,
        running_config=build_agent_running_config(agent_create_args),
        trace_identity=trace_identity,
        cancellation_token=cancellation_token,
    )


def _build_runner_params(
    *,
    runner_type: RunnerType,
    model_name: str,
    temperature: float | None,
    model_config: ModelConfig,
) -> OpenAICompatibleRunnerParams | CliRunnerParams:
    """构造 runner 专属参数。"""

    if runner_type == RunnerType.OPENAI_COMPATIBLE:
        openai_model_config = cast(OpenAICompatibleModelConfig, model_config)
        endpoint_url = openai_model_config.get("endpoint_url")
        target_model = openai_model_config.get("model")
        headers = openai_model_config.get("headers")
        if endpoint_url is None:
            raise ValueError("openai_compatible model_config 缺少 endpoint_url")
        if target_model is None:
            raise ValueError("openai_compatible model_config 缺少 model")
        if headers is None:
            raise ValueError("openai_compatible model_config 缺少 headers")
        openai_runner_params: OpenAICompatibleRunnerParams = {
            "endpoint_url": endpoint_url,
            "model": target_model,
            "headers": dict(headers),
            "name": openai_model_config.get("name") or model_name,
            "temperature": temperature,
            "default_extra_payloads": dict(openai_model_config.get("extra_payloads", {})),
            "timeout": openai_model_config.get("timeout", 3600),
            "max_retries": openai_model_config.get("max_retries", 3),
            "supports_stream": bool(openai_model_config.get("supports_stream", True)),
            "supports_tool_calling": bool(openai_model_config.get("supports_tool_calling", True)),
            "supports_stream_usage": bool(openai_model_config.get("supports_stream_usage", False)),
        }
        return openai_runner_params
    raise ValueError(f"不支持的 runner_type: {runner_type}")


def _build_agent_running_config_from_snapshot(
    agent_create_args: AgentCreateArgs,
) -> AgentRunningConfig:
    """从快照构造 Agent 运行时配置。

    Args:
        agent_create_args: 已解析完成的 Agent 创建参数。

    Returns:
        Agent 运行时配置对象。

    Raises:
        无。
    """

    snapshot = agent_create_args.agent_running_config
    fallback_mode = normalize_fallback_mode(snapshot.get("fallback_mode", FallbackMode.FORCE_ANSWER))
    fallback_mode_literal: Literal["force_answer", "raise_error"]
    if fallback_mode == FallbackMode.RAISE_ERROR:
        fallback_mode_literal = "raise_error"
    else:
        fallback_mode_literal = "force_answer"
    return AgentRunningConfig(
        max_iterations=snapshot.get("max_iterations", 16),
        fallback_mode=fallback_mode_literal,
        fallback_prompt=snapshot.get("fallback_prompt"),
        duplicate_tool_hint_prompt=snapshot.get("duplicate_tool_hint_prompt"),
        continuation_prompt=snapshot.get("continuation_prompt"),
        compaction_summary_header=snapshot.get("compaction_summary_header", "[Context Compaction Summary]"),
        compaction_summary_instruction=snapshot.get(
            "compaction_summary_instruction",
            "Continue reasoning based on recent context. Avoid repeating tool calls that have already been completed.",
        ),
        max_consecutive_failed_tool_batches=snapshot.get("max_consecutive_failed_tool_batches", 2),
        max_duplicate_tool_calls=snapshot.get("max_duplicate_tool_calls", 2),
        max_context_tokens=snapshot.get("max_context_tokens", 0),
        max_output_tokens=snapshot.get("max_output_tokens", 0),
        budget_soft_limit_ratio=snapshot.get("budget_soft_limit_ratio", 0.75),
        budget_hard_limit_ratio=snapshot.get("budget_hard_limit_ratio", 0.9),
        max_continuations=snapshot.get("max_continuations", 3),
        max_compactions=snapshot.get("max_compactions", 3),
    )


__all__ = [
    "build_agent_create_args",
    "build_agent_running_config",
    "build_async_agent",
    "build_async_runner",
]
