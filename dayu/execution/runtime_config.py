"""执行层的纯运行配置模型。

该模块只承载跨 Service / Host 传递的稳定运行配置结构，
不直接依赖 engine 内部实现类。Host 在最后一跳负责把这些
纯配置模型转换为具体的 engine Runner / Agent 配置对象。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Mapping, TypeAlias

from dayu.contracts.runtime_config_snapshot import (
    AgentRunningConfigSnapshot,
    RunnerRunningConfigSnapshot,
)


class FallbackMode(StrEnum):
    """Agent 超限后的降级处理模式。"""

    FORCE_ANSWER = "force_answer"
    RAISE_ERROR = "raise_error"


def normalize_fallback_mode(raw_value: object) -> FallbackMode:
    """把原始 fallback_mode 规范化为枚举。

    Args:
        raw_value: 原始 fallback_mode 值。

    Returns:
        规范化后的 fallback 模式；未知值回退为 ``FORCE_ANSWER``。

    Raises:
        无。
    """

    try:
        return FallbackMode(str(raw_value or FallbackMode.FORCE_ANSWER).strip() or FallbackMode.FORCE_ANSWER)
    except ValueError:
        return FallbackMode.FORCE_ANSWER


def _coerce_float(value: object, *, default: float) -> float:
    """把 snapshot 值转换为 float。

    Args:
        value: 原始 snapshot 值。
        default: 值缺失时使用的默认值。

    Returns:
        转换后的浮点数。

    Raises:
        TypeError: 值类型无法转换时抛出。
        ValueError: 字符串值无法解析为浮点数时抛出。
    """

    if value is None:
        return default
    if isinstance(value, (str, int, float)):
        return float(value)
    raise TypeError(f"snapshot value is not convertible to float: {value!r}")


def _coerce_optional_float(value: object) -> float | None:
    """把 snapshot 值转换为可选 float。"""

    if value is None:
        return None
    return _coerce_float(value, default=0.0)


def _coerce_int(value: object, *, default: int) -> int:
    """把 snapshot 值转换为 int。

    Args:
        value: 原始 snapshot 值。
        default: 值缺失时使用的默认值。

    Returns:
        转换后的整数。

    Raises:
        TypeError: 值类型无法转换时抛出。
        ValueError: 字符串值无法解析为整数时抛出。
    """

    if value is None:
        return default
    if isinstance(value, (str, int, float)):
        return int(value)
    raise TypeError(f"snapshot value is not convertible to int: {value!r}")


@dataclass(frozen=True)
class OpenAIRunnerRuntimeConfig:
    """OpenAI 兼容 runner 的纯运行配置。"""

    debug_sse: bool = False
    debug_tool_delta: bool = False
    debug_sse_sample_rate: float = 1.0
    debug_sse_throttle_sec: float = 0.0
    tool_timeout_seconds: float | None = None
    stream_idle_timeout: float | None = None
    stream_idle_heartbeat_sec: float | None = None


@dataclass(frozen=True)
class CliRunnerRuntimeConfig:
    """CLI runner 的纯运行配置。

    当前 CLI runner 没有额外运行时字段，但仍保留独立类型，
    避免上层继续把 runner 类型知识编码回 engine 实现类。
    """


RunnerRuntimeConfig: TypeAlias = OpenAIRunnerRuntimeConfig | CliRunnerRuntimeConfig


@dataclass(frozen=True)
class AgentRuntimeConfig:
    """Agent 的纯运行配置。"""

    max_iterations: int = 16
    fallback_mode: FallbackMode = FallbackMode.FORCE_ANSWER
    fallback_prompt: str | None = (
        "Based on the information gathered, answer the question directly. "
        "Do not fabricate if information is insufficient."
    )
    duplicate_tool_hint_prompt: str | None = (
        "You just called the same tool ({{tool_name}}) with identical parameters. "
        "Reuse the existing results first. Only call a tool again if you clearly "
        "need new information, and provide your conclusion promptly."
    )
    continuation_prompt: str | None = (
        "Your previous response was truncated (finish_reason=length). "
        "Continue from where you left off without repeating content already produced."
    )
    compaction_summary_header: str = "[Context Compaction Summary]"
    compaction_summary_instruction: str = (
        "Continue reasoning based on recent context. "
        "Avoid repeating tool calls that have already been completed."
    )
    max_consecutive_failed_tool_batches: int = 2
    max_duplicate_tool_calls: int = 2
    max_context_tokens: int = 0
    budget_soft_limit_ratio: float = 0.75
    budget_hard_limit_ratio: float = 0.90
    max_continuations: int = 3
    max_compactions: int = 3


def build_runner_running_config_snapshot(
    running_config: RunnerRuntimeConfig,
) -> RunnerRunningConfigSnapshot:
    """把 runner 纯运行配置转换为跨层快照。"""

    if isinstance(running_config, OpenAIRunnerRuntimeConfig):
        snapshot: RunnerRunningConfigSnapshot = {
            "debug_sse": running_config.debug_sse,
            "debug_tool_delta": running_config.debug_tool_delta,
            "debug_sse_sample_rate": running_config.debug_sse_sample_rate,
            "debug_sse_throttle_sec": running_config.debug_sse_throttle_sec,
        }
        if running_config.tool_timeout_seconds is not None:
            snapshot["tool_timeout_seconds"] = running_config.tool_timeout_seconds
        if running_config.stream_idle_timeout is not None:
            snapshot["stream_idle_timeout"] = running_config.stream_idle_timeout
        if running_config.stream_idle_heartbeat_sec is not None:
            snapshot["stream_idle_heartbeat_sec"] = running_config.stream_idle_heartbeat_sec
        return snapshot
    return {}


def build_agent_running_config_snapshot(
    running_config: AgentRuntimeConfig,
) -> AgentRunningConfigSnapshot:
    """把 agent 纯运行配置转换为跨层快照。"""

    snapshot: AgentRunningConfigSnapshot = {
        "max_iterations": running_config.max_iterations,
        "fallback_mode": str(running_config.fallback_mode),
        "compaction_summary_header": running_config.compaction_summary_header,
        "compaction_summary_instruction": running_config.compaction_summary_instruction,
        "max_consecutive_failed_tool_batches": running_config.max_consecutive_failed_tool_batches,
        "max_duplicate_tool_calls": running_config.max_duplicate_tool_calls,
        "max_context_tokens": running_config.max_context_tokens,
        "budget_soft_limit_ratio": running_config.budget_soft_limit_ratio,
        "budget_hard_limit_ratio": running_config.budget_hard_limit_ratio,
        "max_continuations": running_config.max_continuations,
        "max_compactions": running_config.max_compactions,
    }
    if running_config.fallback_prompt is not None:
        snapshot["fallback_prompt"] = running_config.fallback_prompt
    if running_config.duplicate_tool_hint_prompt is not None:
        snapshot["duplicate_tool_hint_prompt"] = running_config.duplicate_tool_hint_prompt
    if running_config.continuation_prompt is not None:
        snapshot["continuation_prompt"] = running_config.continuation_prompt
    return snapshot


def build_runner_running_config_from_snapshot(
    snapshot: Mapping[str, object],
    *,
    base_config: RunnerRuntimeConfig,
) -> RunnerRuntimeConfig:
    """根据快照和基线配置恢复 runner 纯运行配置。"""

    if isinstance(base_config, OpenAIRunnerRuntimeConfig):
        return OpenAIRunnerRuntimeConfig(
            debug_sse=bool(snapshot.get("debug_sse", False)),
            debug_tool_delta=bool(snapshot.get("debug_tool_delta", False)),
            debug_sse_sample_rate=_coerce_float(snapshot.get("debug_sse_sample_rate"), default=1.0),
            debug_sse_throttle_sec=_coerce_float(snapshot.get("debug_sse_throttle_sec"), default=0.0),
            tool_timeout_seconds=_coerce_optional_float(snapshot.get("tool_timeout_seconds")),
            stream_idle_timeout=_coerce_optional_float(snapshot.get("stream_idle_timeout")),
            stream_idle_heartbeat_sec=_coerce_optional_float(snapshot.get("stream_idle_heartbeat_sec")),
        )
    return CliRunnerRuntimeConfig()


def build_agent_running_config_from_snapshot(
    snapshot: Mapping[str, object],
) -> AgentRuntimeConfig:
    """根据快照恢复 agent 纯运行配置。"""

    return AgentRuntimeConfig(
        max_iterations=_coerce_int(snapshot.get("max_iterations"), default=16),
        fallback_mode=normalize_fallback_mode(snapshot.get("fallback_mode", FallbackMode.FORCE_ANSWER)),
        fallback_prompt=(
            str(snapshot["fallback_prompt"])
            if snapshot.get("fallback_prompt") is not None
            else None
        ),
        duplicate_tool_hint_prompt=(
            str(snapshot["duplicate_tool_hint_prompt"])
            if snapshot.get("duplicate_tool_hint_prompt") is not None
            else None
        ),
        continuation_prompt=(
            str(snapshot["continuation_prompt"])
            if snapshot.get("continuation_prompt") is not None
            else None
        ),
        compaction_summary_header=str(
            snapshot.get("compaction_summary_header", "[Context Compaction Summary]")
        ),
        compaction_summary_instruction=str(
            snapshot.get(
                "compaction_summary_instruction",
                "Continue reasoning based on recent context. Avoid repeating tool calls that have already been completed.",
            )
        ),
        max_consecutive_failed_tool_batches=_coerce_int(
            snapshot.get("max_consecutive_failed_tool_batches"),
            default=2,
        ),
        max_duplicate_tool_calls=_coerce_int(snapshot.get("max_duplicate_tool_calls"), default=2),
        max_context_tokens=_coerce_int(snapshot.get("max_context_tokens"), default=0),
        budget_soft_limit_ratio=_coerce_float(snapshot.get("budget_soft_limit_ratio"), default=0.75),
        budget_hard_limit_ratio=_coerce_float(snapshot.get("budget_hard_limit_ratio"), default=0.9),
        max_continuations=_coerce_int(snapshot.get("max_continuations"), default=3),
        max_compactions=_coerce_int(snapshot.get("max_compactions"), default=3),
    )


__all__ = [
    "AgentRuntimeConfig",
    "FallbackMode",
    "CliRunnerRuntimeConfig",
    "OpenAIRunnerRuntimeConfig",
    "RunnerRuntimeConfig",
    "build_agent_running_config_from_snapshot",
    "build_agent_running_config_snapshot",
    "build_runner_running_config_from_snapshot",
    "build_runner_running_config_snapshot",
    "normalize_fallback_mode",
]