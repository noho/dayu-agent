"""模型目录与 Runner 参数共享类型。

该模块描述 `llm_models.json` 在跨层边界上真正稳定的结构化字段，避免
`model_catalog` 与 `AgentCreateArgs.runner_params` 继续以无约束字典袋子流转。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, TypeAlias, TypedDict


class RunnerType(StrEnum):
    """Runner 类型。"""

    OPENAI_COMPATIBLE = "openai_compatible"
    CLI = "cli"


DISABLED_RUNNER_TYPES: frozenset[RunnerType] = frozenset({RunnerType.CLI})


def normalize_runner_type(raw_value: object) -> RunnerType:
    """把原始 runner_type 规范化为枚举。

    Args:
        raw_value: 原始 runner_type 值。

    Returns:
        规范化后的 Runner 类型；缺失值回退为 OpenAI 兼容 Runner。

    Raises:
        ValueError: 当 runner_type 不受支持时抛出。
    """

    normalized = str(raw_value or RunnerType.OPENAI_COMPATIBLE).strip() or RunnerType.OPENAI_COMPATIBLE
    try:
        return RunnerType(normalized)
    except ValueError as exc:
        raise ValueError(f"不支持的 runner_type: {normalized}") from exc


def ensure_runner_type_enabled(raw_value: object) -> RunnerType:
    """规范化 runner_type 并拒绝已禁用的运行器。

    Args:
        raw_value: 原始 runner_type 值。

    Returns:
        已启用的 Runner 类型。

    Raises:
        ValueError: 当 runner_type 不受支持或已被禁用时抛出。
    """

    runner_type = normalize_runner_type(raw_value)
    if runner_type in DISABLED_RUNNER_TYPES:
        raise ValueError(
            f"runner_type '{runner_type.value}' 已彻底禁用，不允许再配置或使用 CLI runner"
        )
    return runner_type


ModelConfigScalar: TypeAlias = str | int | float | bool | None
ModelConfigJsonValue: TypeAlias = (
    ModelConfigScalar
    | list["ModelConfigJsonValue"]
    | dict[str, "ModelConfigJsonValue"]
)


class TemperatureProfileConfig(TypedDict):
    """单个 scene 温度档位配置。"""

    temperature: float


class ConversationMemoryRuntimeHints(TypedDict, total=False):
    """模型 runtime hints 中允许覆盖的 conversation memory 字段。

    单总池两层结构：``pinned_state`` 独立、不计入 token 池；其余历史共享一个总池，
    `budget = clamp(window * memory_token_budget_ratio, memory_token_budget_floor,
    memory_token_budget_cap)`，``recent_turns_floor`` 给出最近 N 轮强制保留下限。
    """

    memory_token_budget_ratio: float
    memory_token_budget_floor: int
    memory_token_budget_cap: int
    recent_turns_floor: int
    compaction_trigger_context_ratio: float
    compaction_tail_preserve_turns: int
    compaction_context_episode_window: int
    compaction_scene_name: str


class ModelRuntimeHints(TypedDict, total=False):
    """模型配置中的 runtime hints。"""

    temperature_profiles: dict[str, TemperatureProfileConfig]
    conversation_memory: ConversationMemoryRuntimeHints


class BaseModelConfig(TypedDict, total=False):
    """所有模型配置共享的稳定字段。"""

    name: str
    model: str
    timeout: int | float
    max_context_tokens: int
    description: str
    runtime_hints: ModelRuntimeHints


class OpenAICompatibleModelConfig(BaseModelConfig, total=False):
    """OpenAI 兼容 Runner 的模型配置。"""

    runner_type: Literal["openai_compatible"]
    endpoint_url: str
    headers: dict[str, str]
    stream_idle_timeout: float
    stream_idle_heartbeat_sec: float
    supports_stream: bool
    supports_tool_calling: bool
    supports_usage: bool
    supports_stream_usage: bool
    extra_payloads: dict[str, ModelConfigJsonValue]
    max_retries: int


class CliModelConfig(BaseModelConfig, total=False):
    """CLI Runner 的模型配置。"""

    runner_type: Literal["cli"]
    command: list[str]
    working_dir: str
    env: dict[str, str]
    full_auto: bool
    reasoning_effort: str


ModelConfig: TypeAlias = OpenAICompatibleModelConfig | CliModelConfig


class OpenAICompatibleRunnerParams(TypedDict, total=False):
    """传给 `AsyncOpenAIRunner` 的稳定参数。"""

    endpoint_url: str
    model: str
    headers: dict[str, str]
    name: str
    temperature: float | None
    default_extra_payloads: dict[str, ModelConfigJsonValue]
    timeout: int | float
    max_retries: int
    supports_stream: bool
    supports_tool_calling: bool
    supports_stream_usage: bool


class CliRunnerParams(TypedDict, total=False):
    """传给 `AsyncCliRunner` 的稳定参数。"""

    command: list[str]
    working_dir: str
    env: dict[str, str]
    timeout: int | float
    model: str
    full_auto: bool
    reasoning_effort: str
    name: str


RunnerParams: TypeAlias = OpenAICompatibleRunnerParams | CliRunnerParams


__all__ = [
    "BaseModelConfig",
    "CliModelConfig",
    "CliRunnerParams",
    "ConversationMemoryRuntimeHints",
    "ModelConfig",
    "ModelConfigJsonValue",
    "ModelConfigScalar",
    "ModelRuntimeHints",
    "OpenAICompatibleModelConfig",
    "OpenAICompatibleRunnerParams",
    "ensure_runner_type_enabled",
    "RunnerType",
    "RunnerParams",
    "TemperatureProfileConfig",
    "normalize_runner_type",
]