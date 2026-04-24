"""execution.runtime_config 测试。"""

from __future__ import annotations

import pytest

from dayu.execution.runtime_config import (
    AgentRuntimeConfig,
    CliRunnerRuntimeConfig,
    FallbackMode,
    OpenAIRunnerRuntimeConfig,
    build_agent_running_config_from_snapshot,
    build_agent_running_config_snapshot,
    build_runner_running_config_from_snapshot,
    build_runner_running_config_snapshot,
)


@pytest.mark.unit
def test_build_runner_running_config_snapshot_preserves_openai_fields() -> None:
    """OpenAI runner 纯配置应稳定序列化为快照。"""

    snapshot = build_runner_running_config_snapshot(
        OpenAIRunnerRuntimeConfig(
            debug_sse=True,
            debug_tool_delta=True,
            debug_sse_sample_rate=0.25,
            debug_sse_throttle_sec=1.5,
            tool_timeout_seconds=12.0,
            stream_idle_timeout=33.0,
            stream_idle_heartbeat_sec=4.0,
        )
    )

    assert snapshot == {
        "debug_sse": True,
        "debug_tool_delta": True,
        "debug_sse_sample_rate": 0.25,
        "debug_sse_throttle_sec": 1.5,
        "tool_timeout_seconds": 12.0,
        "stream_idle_timeout": 33.0,
        "stream_idle_heartbeat_sec": 4.0,
    }


@pytest.mark.unit
def test_build_runner_running_config_snapshot_for_cli_is_empty() -> None:
    """CLI runner 纯配置当前不应暴露额外快照字段。"""

    assert build_runner_running_config_snapshot(CliRunnerRuntimeConfig()) == {}


@pytest.mark.unit
def test_build_runner_running_config_from_snapshot_uses_base_runner_kind() -> None:
    """runner 快照恢复应由 execution 侧基线类型决定最终配置类型。"""

    recovered = build_runner_running_config_from_snapshot(
        {
            "debug_sse": True,
            "tool_timeout_seconds": 18.0,
        },
        base_config=OpenAIRunnerRuntimeConfig(),
    )

    assert isinstance(recovered, OpenAIRunnerRuntimeConfig)
    assert recovered.debug_sse is True
    assert recovered.tool_timeout_seconds == pytest.approx(18.0)

    cli_recovered = build_runner_running_config_from_snapshot(
        {"debug_sse": True},
        base_config=CliRunnerRuntimeConfig(),
    )
    assert isinstance(cli_recovered, CliRunnerRuntimeConfig)


@pytest.mark.unit
def test_build_runner_running_config_from_snapshot_accepts_string_numbers() -> None:
    """runner 快照中的字符串数字应按当前语义恢复为数值。"""

    recovered = build_runner_running_config_from_snapshot(
        {
            "debug_sse_sample_rate": "0.5",
            "debug_sse_throttle_sec": "1.25",
            "tool_timeout_seconds": "18.5",
            "stream_idle_timeout": "30",
            "stream_idle_heartbeat_sec": "4.0",
        },
        base_config=OpenAIRunnerRuntimeConfig(),
    )

    assert isinstance(recovered, OpenAIRunnerRuntimeConfig)
    assert recovered.debug_sse_sample_rate == pytest.approx(0.5)
    assert recovered.debug_sse_throttle_sec == pytest.approx(1.25)
    assert recovered.tool_timeout_seconds == pytest.approx(18.5)
    assert recovered.stream_idle_timeout == pytest.approx(30.0)
    assert recovered.stream_idle_heartbeat_sec == pytest.approx(4.0)


@pytest.mark.unit
def test_build_agent_running_config_snapshot_and_restore_roundtrip() -> None:
    """agent 纯配置应可在快照和恢复之间保持关键治理字段。"""

    original = AgentRuntimeConfig(
        max_iterations=9,
        fallback_mode=FallbackMode.RAISE_ERROR,
        fallback_prompt="fallback",
        duplicate_tool_hint_prompt="dup",
        continuation_prompt="continue",
        compaction_summary_header="header",
        compaction_summary_instruction="instruction",
        max_consecutive_failed_tool_batches=5,
        max_duplicate_tool_calls=4,
        max_context_tokens=64000,
        budget_soft_limit_ratio=0.6,
        budget_hard_limit_ratio=0.85,
        max_continuations=7,
        max_compactions=2,
    )

    snapshot = build_agent_running_config_snapshot(original)
    recovered = build_agent_running_config_from_snapshot(snapshot)

    assert recovered == original


@pytest.mark.unit
def test_build_agent_running_config_from_snapshot_normalizes_invalid_fallback_mode() -> None:
    """非法 fallback_mode 应被收敛到稳定默认值。"""

    recovered = build_agent_running_config_from_snapshot({"fallback_mode": "unknown"})

    assert recovered.fallback_mode == "force_answer"
    assert recovered.max_iterations == 16


@pytest.mark.unit
def test_build_agent_running_config_from_snapshot_accepts_string_numbers() -> None:
    """agent 快照中的字符串数字应按当前语义恢复为数值。"""

    recovered = build_agent_running_config_from_snapshot(
        {
            "max_iterations": "12",
            "max_consecutive_failed_tool_batches": "4",
            "max_duplicate_tool_calls": "5",
            "max_context_tokens": "64000",
            "budget_soft_limit_ratio": "0.61",
            "budget_hard_limit_ratio": "0.88",
            "max_continuations": "6",
            "max_compactions": "2",
        }
    )

    assert recovered.max_iterations == 12
    assert recovered.max_consecutive_failed_tool_batches == 4
    assert recovered.max_duplicate_tool_calls == 5
    assert recovered.max_context_tokens == 64000
    assert recovered.budget_soft_limit_ratio == pytest.approx(0.61)
    assert recovered.budget_hard_limit_ratio == pytest.approx(0.88)
    assert recovered.max_continuations == 6
    assert recovered.max_compactions == 2