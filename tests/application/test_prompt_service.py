"""PromptService 测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedInfrastructureSpec,
    AcceptedModelSpec,
    AcceptedRuntimeSpec,
    AcceptedToolConfigSpec,
    ExecutionContract,
)
from dayu.contracts.model_config import OpenAICompatibleModelConfig
from dayu.contracts.toolset_config import ToolsetConfigSnapshot, build_toolset_config_snapshot
from dayu.contracts.events import AppEventType
from dayu.contracts.session import SessionSource
from dayu.execution.runtime_config import (
    AgentRunningConfigSnapshot,
    AgentRuntimeConfig,
    OpenAIRunnerRuntimeConfig,
)
from dayu.execution.options import ConversationMemorySettings, ExecutionOptions, ResolvedExecutionOptions
from dayu.execution.options import (
    DocToolLimits,
    FinsToolLimits,
    TraceSettings,
    WebToolsConfig,
)
from dayu.host.host import Host
from dayu.host.host_execution import HostExecutorProtocol
from dayu.host.protocols import RunRegistryProtocol, SessionRegistryProtocol
from dayu.prompting.scene_definition import (
    SceneConversationDefinition,
    SceneDefinition,
    SceneModelDefinition,
)
from dayu.services.contracts import PromptRequest
from dayu.services.prompt_service import PromptService
from dayu.services.protocols import PromptServiceProtocol
from dayu.services.scene_execution_acceptance import AcceptedSceneExecution, SceneExecutionAcceptancePreparer
from tests.application.conftest import StubHostExecutor, StubRunRegistry, StubSessionRegistry


def _build_toolset_configs(
    *,
    doc_tool_limits: DocToolLimits | None = None,
    fins_tool_limits: FinsToolLimits | None = None,
    web_tools_config: WebToolsConfig | None = None,
) -> tuple[ToolsetConfigSnapshot, ...]:
    """构造测试用通用 toolset 配置快照。"""

    return tuple(
        snapshot
        for snapshot in (
            build_toolset_config_snapshot("doc", doc_tool_limits),
            build_toolset_config_snapshot("fins", fins_tool_limits),
            build_toolset_config_snapshot("web", web_tools_config),
        )
        if snapshot is not None
    )


def _build_scene_definition() -> SceneDefinition:
    """构造测试用 scene 定义。"""

    return SceneDefinition(
        name="prompt",
        model=SceneModelDefinition(default_name="test-model"),
        version="v1",
        description="test scene",
        context_slots=("fins_default_subject", "base_user"),
        conversation=SceneConversationDefinition(enabled=False),
    )


def _build_model_config() -> OpenAICompatibleModelConfig:
    """构造测试用模型配置。"""

    return {
        "name": "test-model",
        "model": "gpt-5.4",
        "runner_type": "openai_compatible",
        "max_context_tokens": 32000,
        "max_output_tokens": 4096,
    }


def _host(gateway: object) -> Host:
    """把服务持有的 Host gateway 收窄为真实 Host。"""

    return cast(Host, gateway)


def _executor(gateway: object):
    """返回测试用 Host executor。"""

    return cast(StubHostExecutor, _host(gateway)._executor)


def _session_registry(gateway: object):
    """返回测试用 session registry。"""

    return cast(StubSessionRegistry, _host(gateway)._session_registry)


def _last_execution_contract(gateway: object) -> ExecutionContract:
    """返回最近一次执行契约。"""

    contract = _executor(gateway).last_execution_contract
    assert contract is not None
    return contract


def _build_accepted_execution_spec(
    *,
    model_name: str,
    temperature: float | None,
    agent_running_config: AgentRunningConfigSnapshot,
    doc_tool_limits: DocToolLimits | None,
    fins_tool_limits: FinsToolLimits | None,
    web_tools_config: WebToolsConfig | None,
    trace_settings: TraceSettings | None,
    conversation_memory_settings: ConversationMemorySettings | None,
) -> AcceptedExecutionSpec:
    """构造分组式 accepted execution spec。"""

    return AcceptedExecutionSpec(
        model=AcceptedModelSpec(model_name=model_name, temperature=temperature),
        runtime=AcceptedRuntimeSpec(
            runner_running_config={},
            agent_running_config=agent_running_config,
        ),
        tools=AcceptedToolConfigSpec(
            toolset_configs=_build_toolset_configs(
                doc_tool_limits=doc_tool_limits,
                fins_tool_limits=fins_tool_limits,
                web_tools_config=web_tools_config,
            ),
        ),
        infrastructure=AcceptedInfrastructureSpec(
            trace_settings=trace_settings,
            conversation_memory_settings=conversation_memory_settings,
        ),
    )


class _FakeSceneExecutionAcceptancePreparer:
    """测试用 scene 执行接受准备器。"""

    def __init__(self) -> None:
        """初始化测试桩。"""

        self.calls: list[dict[str, object]] = []

    def prepare(self, scene_name: str, execution_options: ExecutionOptions | None = None) -> AcceptedSceneExecution:
        """返回固定的 accepted_execution_spec。"""

        self.calls.append(
            {
                "scene_name": scene_name,
                "execution_options": execution_options,
            }
        )
        resolved_execution_options = ResolvedExecutionOptions(
            model_name="test-model",
            runner_running_config=OpenAIRunnerRuntimeConfig(),
            agent_running_config=AgentRuntimeConfig(),
            toolset_configs=_build_toolset_configs(
                doc_tool_limits=DocToolLimits(),
                fins_tool_limits=FinsToolLimits(),
                web_tools_config=WebToolsConfig(provider="off"),
            ),
            trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp/tool-trace")),
            temperature=0.2,
        )
        return AcceptedSceneExecution(
            scene_name=scene_name,
            scene_definition=_build_scene_definition(),
            resolved_execution_options=resolved_execution_options,
            model_config=_build_model_config(),
            resolved_temperature=0.2,
            accepted_execution_spec=_build_accepted_execution_spec(
                model_name="test-model",
                temperature=0.2,
                agent_running_config={
                    "max_iterations": 8,
                    "max_context_tokens": 32000,
                    "max_output_tokens": 4096,
                },
                doc_tool_limits=DocToolLimits(),
                fins_tool_limits=FinsToolLimits(),
                web_tools_config=WebToolsConfig(provider="off"),
                trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp/tool-trace")),
                conversation_memory_settings=resolved_execution_options.conversation_memory_settings,
            ),
        )


def _build_service() -> tuple[PromptService, _FakeSceneExecutionAcceptancePreparer]:
    """构建测试服务实例。"""

    resolver = _FakeSceneExecutionAcceptancePreparer()
    session_registry = StubSessionRegistry()
    host_executor = StubHostExecutor()
    run_registry = StubRunRegistry()
    host = Host(
        executor=cast(HostExecutorProtocol, host_executor),
        session_registry=cast(SessionRegistryProtocol, session_registry),
        run_registry=cast(RunRegistryProtocol, run_registry),
    )
    service = PromptService(
        host=host,
        scene_execution_acceptance_preparer=cast(SceneExecutionAcceptancePreparer, resolver),
        company_name_resolver=lambda ticker: f"{ticker}-NAME",
    )
    return service, resolver


async def _submit_and_collect(
    service: PromptService,
    request: PromptRequest,
) -> list:
    """提交单轮 Prompt 并收集事件。"""

    submission = await service.submit(request)
    events = []
    async for event in submission.event_stream:
        events.append(event)
    return events


@pytest.mark.unit
def test_stream_prompt_outputs_execution_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 PromptService 会直接生成 ExecutionContract。"""

    monkeypatch.setattr(
        "dayu.services.prompt_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )

    service, resolver = _build_service()

    events = asyncio.run(_submit_and_collect(service, PromptRequest(user_text=" hello ", ticker="AAPL")))

    assert [event.type for event in events] == [AppEventType.CONTENT_DELTA, AppEventType.FINAL_ANSWER]
    assert resolver.calls == [{"scene_name": "prompt", "execution_options": None}]
    contract = _last_execution_contract(service.host)
    assert contract.service_name == "prompt"
    assert contract.scene_name == "prompt"
    assert contract.host_policy.business_concurrency_lane is None
    assert contract.host_policy.timeout_ms is None
    assert contract.host_policy.resumable is False
    assert contract.message_inputs.user_message == "hello"
    assert contract.accepted_execution_spec.model.model_name == "test-model"
    assert contract.preparation_spec.selected_toolsets == ()
    assert contract.preparation_spec.execution_permissions.web.allow_private_network_url is False
    assert contract.preparation_spec.execution_permissions.doc.allowed_read_paths == ()
    assert contract.preparation_spec.prompt_contributions["base_user"] == "# 用户与运行时上下文\n当前时间：2026年04月03日。"
    assert contract.preparation_spec.prompt_contributions["fins_default_subject"] == "# 当前分析对象\n你正在分析的是 AAPL（AAPL-NAME）。"
    assert contract.metadata == {}
    session = _session_registry(service.host).get_session(contract.host_policy.session_key or "")
    assert session is not None
    assert session.source == SessionSource.API
    assert session.scene_name == "prompt"


@pytest.mark.unit
def test_prompt_service_implements_protocol() -> None:
    """验证 PromptService 满足 PromptServiceProtocol 协议。"""

    service, _ = _build_service()
    assert isinstance(service, PromptServiceProtocol)


@pytest.mark.unit
def test_stream_prompt_reuses_explicit_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 session_id 时应复用既有 Host Session。"""

    monkeypatch.setattr(
        "dayu.services.prompt_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )

    service, _ = _build_service()
    session = _session_registry(service.host).create_session(
        source=SessionSource.WEB,
        session_id="session_prompt",
    )

    asyncio.run(
        _submit_and_collect(
            service,
            PromptRequest(user_text=" hello ", ticker="AAPL", session_id=session.session_id),
        )
    )

    contract = _last_execution_contract(service.host)
    assert contract.host_policy.session_key == session.session_id


@pytest.mark.unit
def test_stream_prompt_raises_when_input_empty() -> None:
    """验证空输入会触发 ValueError。"""

    service, _ = _build_service()

    with pytest.raises(ValueError):
        asyncio.run(_submit_and_collect(service, PromptRequest(user_text="   ", ticker="AAPL")))
