"""`host.agent_builder` 额外覆盖测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from dayu.contracts.agent_execution import AgentCreateArgs
from dayu.contracts.agent_types import AgentTraceIdentity
from dayu.contracts.model_config import OpenAICompatibleRunnerParams, RunnerType
from dayu.contracts.toolset_config import ToolsetConfigSnapshot
from dayu.execution.options import ConversationMemorySettings, ResolvedExecutionOptions, TraceSettings
from dayu.execution.runtime_config import AgentRuntimeConfig, OpenAIRunnerRuntimeConfig
from dayu.host import agent_builder as module


@pytest.mark.unit
@pytest.mark.parametrize(
    "runner_params, expected_message",
    [
        ({"model": "demo", "headers": {}}, "endpoint_url"),
        ({"endpoint_url": "http://example.com", "headers": {}}, "model"),
        ({"endpoint_url": "http://example.com", "model": "demo"}, "headers"),
    ],
)
def test_build_async_runner_validates_required_openai_runner_params(
    runner_params: OpenAICompatibleRunnerParams,
    expected_message: str,
) -> None:
    """OpenAI runner 缺少关键字段时应显式失败。"""

    with pytest.raises(ValueError, match=expected_message):
        module.build_async_runner(
            AgentCreateArgs(
                runner_type=RunnerType.OPENAI_COMPATIBLE,
                model_name="demo",
                runner_params=cast(Any, runner_params),
            )
        )


@pytest.mark.unit
def test_build_agent_create_args_maps_execution_options_to_openai_runner_params() -> None:
    """构造 AgentCreateArgs 时应正确吸收模型与执行选项。"""

    resolved = ResolvedExecutionOptions(
        model_name="resolved-model",
        runner_running_config=OpenAIRunnerRuntimeConfig(tool_timeout_seconds=12.0),
        agent_running_config=AgentRuntimeConfig(max_iterations=7),
        trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp/trace")),
        temperature=0.4,
        conversation_memory_settings=ConversationMemorySettings(),
        toolset_configs=(ToolsetConfigSnapshot("doc", payload={"limit": 1}),),
    )
    model_config = cast(
        dict[str, object],
        {
            "runner_type": RunnerType.OPENAI_COMPATIBLE,
            "endpoint_url": "http://example.com",
            "model": "provider-model",
            "headers": {"Authorization": "Bearer token"},
            "name": "configured-name",
            "max_context_tokens": 8192,
            "extra_payloads": {"reasoning": True},
            "timeout": 30,
        },
    )

    created = module.build_agent_create_args(
        resolved_execution_options=resolved,
        model_config=cast(Any, model_config),
    )

    assert created.runner_type == RunnerType.OPENAI_COMPATIBLE.value
    assert created.model_name == "resolved-model"
    assert created.max_turns == 7
    assert created.max_context_tokens == 8192
    assert created.temperature == 0.4
    runner_params = cast(dict[str, object], created.runner_params)
    assert runner_params.get("endpoint_url") == "http://example.com"
    assert runner_params.get("model") == "provider-model"
    assert runner_params.get("name") == "configured-name"
    assert runner_params.get("default_extra_payloads") == {"reasoning": True}


@pytest.mark.unit
def test_build_async_agent_delegates_runner_and_running_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """构造 AsyncAgent 时应把 runner、运行配置和追踪身份完整透传。"""

    captured: dict[str, object] = {}
    fake_runner = object()
    fake_running_config = object()
    trace_identity = AgentTraceIdentity(
        agent_name="prompt-agent",
        agent_kind="interactive",
        scene_name="prompt",
        model_name="demo-model",
        session_id="trace-session",
    )

    monkeypatch.setattr(module, "build_async_runner", lambda *args, **kwargs: fake_runner)
    monkeypatch.setattr(module, "build_agent_running_config", lambda args: fake_running_config)

    class _FakeAsyncAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(module, "AsyncAgent", _FakeAsyncAgent)

    created = module.build_async_agent(
        agent_create_args=AgentCreateArgs(
            runner_type=RunnerType.OPENAI_COMPATIBLE,
            model_name="demo",
            runner_params={
                "endpoint_url": "http://example.com",
                "model": "demo",
                "headers": {},
            },
        ),
        trace_identity=trace_identity,
    )

    assert isinstance(created, _FakeAsyncAgent)
    assert captured["runner"] is fake_runner
    assert captured["running_config"] is fake_running_config
    assert captured["trace_identity"] is trace_identity