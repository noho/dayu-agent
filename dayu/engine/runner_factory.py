"""Engine Runner 工厂。

Host 通过本模块的 ``create_runner`` 获取 ``AsyncRunner`` 实例，
无需直接依赖具体 Runner 实现类。
"""

from __future__ import annotations

from typing import cast

from dayu.contracts.agent_execution import AgentCreateArgs
from dayu.contracts.model_config import (
    OpenAICompatibleRunnerParams,
    RunnerType,
    ensure_runner_type_enabled,
)
from dayu.engine.async_openai_runner import AsyncOpenAIRunner, AsyncOpenAIRunnerRunningConfig
from dayu.contracts.cancellation import CancellationToken
from dayu.engine.protocols import AsyncRunner


def create_runner(
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

    runner_type = ensure_runner_type_enabled(agent_create_args.runner_type)
    if runner_type == RunnerType.OPENAI_COMPATIBLE:
        runner_params = cast(OpenAICompatibleRunnerParams, agent_create_args.runner_params)
        endpoint_url = runner_params.get("endpoint_url")
        model = runner_params.get("model")
        headers = runner_params.get("headers")
        if endpoint_url is None:
            raise ValueError("openai_compatible runner_params 缺少 endpoint_url")
        if model is None:
            raise ValueError("openai_compatible runner_params 缺少 model")
        if headers is None:
            raise ValueError("openai_compatible runner_params 缺少 headers")
        temperature = runner_params.get("temperature")
        return AsyncOpenAIRunner(
            endpoint_url=str(endpoint_url),
            model=str(model),
            headers=dict(headers),
            name=runner_params.get("name"),
            temperature=float(agent_create_args.temperature or 0.0) if temperature is None else float(temperature),
            default_extra_payloads=dict(runner_params.get("default_extra_payloads") or {}),
            timeout=int(runner_params.get("timeout", 3600)),
            max_retries=int(runner_params.get("max_retries", 3)),
            supports_stream=bool(runner_params.get("supports_stream", True)),
            supports_tool_calling=bool(runner_params.get("supports_tool_calling", True)),
            supports_stream_usage=bool(runner_params.get("supports_stream_usage", False)),
            running_config=_build_openai_runner_running_config(agent_create_args),
            cancellation_token=cancellation_token,
        )
    raise ValueError(f"不支持的 runner_type: {runner_type}")


def _build_openai_runner_running_config(
    agent_create_args: AgentCreateArgs,
) -> AsyncOpenAIRunnerRunningConfig:
    """从快照构造 OpenAI Runner 运行时配置。"""

    snapshot = agent_create_args.runner_running_config
    return AsyncOpenAIRunnerRunningConfig(
        debug_sse=snapshot.get("debug_sse", False),
        debug_tool_delta=snapshot.get("debug_tool_delta", False),
        debug_sse_sample_rate=snapshot.get("debug_sse_sample_rate", 1.0),
        debug_sse_throttle_sec=snapshot.get("debug_sse_throttle_sec", 0.0),
        tool_timeout_seconds=snapshot.get("tool_timeout_seconds"),
        stream_idle_timeout=snapshot.get("stream_idle_timeout"),
        stream_idle_heartbeat_sec=snapshot.get("stream_idle_heartbeat_sec"),
    )


__all__ = ["create_runner"]
