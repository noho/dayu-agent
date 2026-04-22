"""ChatService 测试。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from typing import Mapping, cast

import pytest

from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedInfrastructureSpec,
    AcceptedModelSpec,
    AcceptedRuntimeSpec,
    AcceptedToolConfigSpec,
    AgentCreateArgs,
    ExecutionContract,
)
from dayu.contracts.execution_metadata import ExecutionDeliveryContext, normalize_execution_delivery_context
from dayu.contracts.model_config import ModelConfig
from dayu.contracts.events import AppEventType
from dayu.contracts.run import RunCancelReason
from dayu.contracts.session import SessionSource
from dayu.contracts.toolset_config import ToolsetConfigSnapshot, build_toolset_config_snapshot
from dayu.execution.runtime_config import (
    AgentRuntimeConfig,
    OpenAIRunnerRuntimeConfig,
    build_agent_running_config_snapshot,
    build_runner_running_config_snapshot,
)
from dayu.execution.options import ConversationMemorySettings, ExecutionOptions, ResolvedExecutionOptions
from dayu.execution.options import (
    DocToolLimits,
    FinsToolLimits,
    TraceSettings,
    WebToolsConfig,
)
from dayu.host.host import Host
from dayu.host.conversation_store import ConversationTranscript
from dayu.host.pending_turn_store import PendingConversationTurnState
from dayu.host.prepared_turn import (
    AcceptedAgentTurnSnapshot,
    PreparedAgentTurnSnapshot,
    PreparedConversationSessionSnapshot,
    serialize_accepted_agent_turn_snapshot,
    serialize_prepared_agent_turn_snapshot,
)
from dayu.log import Log
from dayu.prompting.scene_definition import (
    SceneConversationDefinition,
    SceneDefinition,
    SceneModelDefinition,
)
from dayu.services.chat_service import ChatService
from dayu.services.contracts import ChatResumeRequest, ChatTurnRequest, SessionResolutionPolicy
from dayu.services.protocols import ChatServiceProtocol
from dayu.services.scene_execution_acceptance import (
    AcceptedSceneExecution,
    SceneExecutionAcceptancePreparer,
)
from tests.application.conftest import StubHostExecutor, StubPendingTurnStore, StubRunRegistry, StubSessionRegistry


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


def _build_test_scene_definition(scene_name: str) -> SceneDefinition:
    """构造测试用 scene 定义。"""

    return SceneDefinition(
        name=scene_name,
        model=SceneModelDefinition(default_name="chat-model"),
        version="v1",
        description="chat service test scene",
        context_slots=("fins_default_subject", "base_user"),
        conversation=SceneConversationDefinition(enabled=True),
    )


def _build_test_model_config() -> ModelConfig:
    """构造测试用模型配置。"""

    return {
        "name": "chat-model",
        "model": "chat-model",
        "runner_type": "openai_compatible",
        "supports_stream": True,
        "supports_tool_calling": True,
        "supports_usage": True,
        "supports_stream_usage": True,
    }


def _build_test_resolved_execution_options() -> ResolvedExecutionOptions:
    """构造测试用已解析执行选项。"""

    return ResolvedExecutionOptions(
        model_name="chat-model",
        runner_running_config=OpenAIRunnerRuntimeConfig(),
        agent_running_config=AgentRuntimeConfig(),
        toolset_configs=_build_toolset_configs(
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider="off"),
        ),
        trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp/tool-trace")),
        temperature=0.3,
    )


def _build_test_accepted_execution_spec(
    resolved_execution_options: ResolvedExecutionOptions,
) -> AcceptedExecutionSpec:
    """构造测试用 accepted execution spec。"""

    return AcceptedExecutionSpec(
        model=AcceptedModelSpec(
            model_name=resolved_execution_options.model_name,
            temperature=resolved_execution_options.temperature,
        ),
        runtime=AcceptedRuntimeSpec(
            runner_running_config=build_runner_running_config_snapshot(
                resolved_execution_options.runner_running_config
            ),
            agent_running_config=build_agent_running_config_snapshot(
                resolved_execution_options.agent_running_config
            ),
        ),
        tools=AcceptedToolConfigSpec(toolset_configs=resolved_execution_options.toolset_configs),
        infrastructure=AcceptedInfrastructureSpec(
            trace_settings=resolved_execution_options.trace_settings,
            conversation_memory_settings=resolved_execution_options.conversation_memory_settings,
        ),
    )


def _delivery_context(context: Mapping[str, object] | None = None) -> ExecutionDeliveryContext:
    """构造测试用交付上下文。"""

    return normalize_execution_delivery_context(context)


def _host(service: ChatService) -> Host:
    """把服务上的 Host 协议收窄为测试使用的具体 Host。"""

    return cast(Host, service.host)


def _executor(service: ChatService) -> StubHostExecutor:
    """返回测试用 HostExecutor stub。"""

    return cast(StubHostExecutor, _host(service)._executor)


def _run_registry(service: ChatService) -> StubRunRegistry:
    """返回测试用 RunRegistry stub。"""

    return cast(StubRunRegistry, _host(service)._run_registry)


def _session_registry(service: ChatService) -> StubSessionRegistry:
    """返回测试用 SessionRegistry stub。"""

    return cast(StubSessionRegistry, _host(service)._session_registry)


def _pending_turn_store(service: ChatService) -> StubPendingTurnStore:
    """返回测试用 PendingTurnStore stub。"""

    return cast(StubPendingTurnStore, _host(service)._pending_turn_store)


def _last_execution_contract(service: ChatService) -> ExecutionContract:
    """返回最近一次执行的 execution contract。"""

    execution_contract = _executor(service).last_execution_contract
    assert execution_contract is not None
    return execution_contract


def _last_prepared_turn(service: ChatService) -> PreparedAgentTurnSnapshot:
    """返回最近一次执行的 prepared turn。"""

    prepared_turn = _executor(service).last_prepared_turn
    assert prepared_turn is not None
    return prepared_turn


def _conversation_session(
    prepared_turn: PreparedAgentTurnSnapshot,
) -> PreparedConversationSessionSnapshot:
    """返回 prepared turn 中的对话会话快照。"""

    conversation_session = prepared_turn.conversation_session
    assert conversation_session is not None
    return conversation_session


class _FakeSceneExecutionAcceptancePreparer:
    """测试用 scene 执行接受准备器。"""

    def __init__(self) -> None:
        """初始化测试桩。"""

        self.calls: list[dict[str, object]] = []

    def prepare(self, scene_name: str, execution_options: ExecutionOptions | None = None) -> AcceptedSceneExecution:
        """返回固定 accepted_execution_spec。"""

        self.calls.append(
            {
                "scene_name": scene_name,
                "execution_options": execution_options,
            }
        )
        resolved_execution_options = _build_test_resolved_execution_options()
        return AcceptedSceneExecution(
            scene_name=scene_name,
            scene_definition=_build_test_scene_definition(scene_name),
            resolved_execution_options=resolved_execution_options,
            model_config=_build_test_model_config(),
            resolved_temperature=0.3,
            accepted_execution_spec=_build_test_accepted_execution_spec(resolved_execution_options),
        )


def _build_service(
    *,
    session_source: SessionSource = SessionSource.API,
) -> tuple[ChatService, _FakeSceneExecutionAcceptancePreparer]:
    """构建测试服务实例。"""

    resolver = _FakeSceneExecutionAcceptancePreparer()
    session_registry = StubSessionRegistry()
    session_registry.create_session(source=SessionSource.CLI, session_id="s1")
    host_executor = StubHostExecutor()
    run_registry = StubRunRegistry()
    pending_turn_store = StubPendingTurnStore()
    host = Host(
        executor=host_executor,  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=run_registry,  # type: ignore[arg-type]
        pending_turn_store=pending_turn_store,  # type: ignore[arg-type]
    )
    service = ChatService(
        host=host,
        scene_execution_acceptance_preparer=cast(SceneExecutionAcceptancePreparer, resolver),
        company_name_resolver=lambda ticker: f"{ticker}-NAME",
        session_source=session_source,
    )
    return service, resolver


async def _consume_submission(
    service: ChatService,
    pending_turn_id: str,
    *,
    session_id: str | None = None,
) -> list:
    """执行一次 pending turn 恢复并收集事件。"""

    pending_turn = service.host.get_pending_turn(pending_turn_id)
    assert pending_turn is not None
    submission = await service.resume_pending_turn(
        ChatResumeRequest(
            session_id=session_id or pending_turn.session_id,
            pending_turn_id=pending_turn_id,
        )
    )
    events = []
    async for event in submission.event_stream:
        events.append(event)
    return events


async def _submit_turn_and_collect(
    service: ChatService,
    request: ChatTurnRequest,
) -> list:
    """提交聊天单轮并收集事件。"""

    submission = await service.submit_turn(request)
    events = []
    async for event in submission.event_stream:
        events.append(event)
    return events


def _seed_pending_turn_with_contract(
    service: ChatService,
    *,
    execution_contract: ExecutionContract,
    run_state: str,
    cancel_reason: RunCancelReason | None = None,
) -> str:
    """基于 Host prepared snapshot 预置一条可测试的 pending turn。"""

    run = _run_registry(service).register_run(
        session_id=execution_contract.host_policy.session_key,
        service_type=execution_contract.service_name,
        scene_name=execution_contract.scene_name,
    )
    _run_registry(service).start_run(run.run_id)
    if run_state == "failed":
        source_run = _run_registry(service).fail_run(run.run_id, error_summary="boom")
    elif run_state == "succeeded":
        source_run = _run_registry(service).complete_run(run.run_id)
    elif run_state == "timeout_cancelled":
        _run_registry(service).request_cancel(run.run_id, cancel_reason=RunCancelReason.TIMEOUT)
        source_run = _run_registry(service).mark_cancelled(run.run_id, cancel_reason=RunCancelReason.TIMEOUT)
    elif run_state == "user_cancelled":
        _run_registry(service).request_cancel(run.run_id, cancel_reason=RunCancelReason.USER_CANCELLED)
        source_run = _run_registry(service).mark_cancelled(run.run_id, cancel_reason=RunCancelReason.USER_CANCELLED)
    elif run_state == "running":
        source_run = _run_registry(service).get_run(run.run_id)
    else:
        raise AssertionError(f"unknown run_state: {run_state}")
    assert source_run is not None
    session_id = str(execution_contract.host_policy.session_key or "").strip()
    prepared_turn = PreparedAgentTurnSnapshot(
        service_name=execution_contract.service_name,
        scene_name=execution_contract.scene_name,
        metadata=_delivery_context({
            str(key): str(value)
            for key, value in execution_contract.metadata.items()
        }),
        business_concurrency_lane=execution_contract.host_policy.business_concurrency_lane,
        timeout_ms=execution_contract.host_policy.timeout_ms,
        resumable=bool(execution_contract.host_policy.resumable),
        system_prompt="resume-system",
        messages=[{"role": "user", "content": str(execution_contract.message_inputs.user_message or "")}],
        agent_create_args=AgentCreateArgs(
            runner_type="openai",
            model_name=execution_contract.accepted_execution_spec.model.model_name,
        ),
        selected_toolsets=execution_contract.preparation_spec.selected_toolsets,
        execution_permissions=execution_contract.preparation_spec.execution_permissions,
        toolset_configs=execution_contract.accepted_execution_spec.tools.toolset_configs,
        trace_settings=execution_contract.accepted_execution_spec.infrastructure.trace_settings,
        conversation_memory_settings=ConversationMemorySettings(),
        conversation_session=PreparedConversationSessionSnapshot(
            session_id=session_id,
            user_message=str(execution_contract.message_inputs.user_message or ""),
            transcript=ConversationTranscript.create_empty(session_id),
        ),
    )
    pending_turn = _pending_turn_store(service).seed_pending_turn(
        session_id=session_id,
        scene_name=execution_contract.scene_name,
        user_text=str(execution_contract.message_inputs.user_message or ""),
        source_run_id=source_run.run_id,
        resume_source_json=json.dumps(
            serialize_prepared_agent_turn_snapshot(prepared_turn),
            ensure_ascii=False,
            sort_keys=True,
        ),
        metadata=_delivery_context({
            str(key): str(value)
            for key, value in execution_contract.metadata.items()
        }),
        state=PendingConversationTurnState.SENT_TO_LLM,
    )
    return pending_turn.pending_turn_id


def _seed_accepted_pending_turn_with_contract(
    service: ChatService,
    *,
    execution_contract: ExecutionContract,
    run_state: str,
) -> str:
    """基于 Host accepted snapshot 预置一条可恢复 pending turn。"""

    run = _run_registry(service).register_run(
        session_id=execution_contract.host_policy.session_key,
        service_type=execution_contract.service_name,
        scene_name=execution_contract.scene_name,
    )
    _run_registry(service).start_run(run.run_id)
    if run_state == "failed":
        source_run = _run_registry(service).fail_run(run.run_id, error_summary="boom")
    else:
        raise AssertionError(f"unknown run_state: {run_state}")
    assert source_run is not None
    accepted_turn = AcceptedAgentTurnSnapshot(
        service_name=execution_contract.service_name,
        scene_name=execution_contract.scene_name,
        host_policy=execution_contract.host_policy,
        preparation_spec=execution_contract.preparation_spec,
        message_inputs=execution_contract.message_inputs,
        accepted_execution_spec=execution_contract.accepted_execution_spec,
        execution_options=execution_contract.execution_options,
        metadata=_delivery_context({
            str(key): str(value)
            for key, value in execution_contract.metadata.items()
        }),
    )
    pending_turn = _pending_turn_store(service).seed_pending_turn(
        session_id=str(execution_contract.host_policy.session_key or ""),
        scene_name=execution_contract.scene_name,
        user_text=str(execution_contract.message_inputs.user_message or ""),
        source_run_id=source_run.run_id,
        resume_source_json=json.dumps(
            serialize_accepted_agent_turn_snapshot(accepted_turn),
            ensure_ascii=False,
            sort_keys=True,
        ),
        metadata=_delivery_context({
            str(key): str(value)
            for key, value in execution_contract.metadata.items()
        }),
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    return pending_turn.pending_turn_id


@pytest.mark.unit
def test_stream_turn_outputs_execution_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 ChatService 会直接生成 ExecutionContract。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )

    service, resolver = _build_service()

    events = asyncio.run(
        _submit_turn_and_collect(
            service,
            ChatTurnRequest(session_id="s1", user_text=" hello ", ticker="AAPL"),
        )
    )

    assert [event.type for event in events] == [AppEventType.CONTENT_DELTA, AppEventType.FINAL_ANSWER]
    assert resolver.calls == [{"scene_name": "interactive", "execution_options": None}]
    contract = _last_execution_contract(service)
    assert contract.service_name == "chat_turn"
    assert contract.scene_name == "interactive"
    assert contract.host_policy.session_key == "s1"
    assert contract.host_policy.timeout_ms is None
    assert contract.host_policy.resumable is True
    assert contract.message_inputs.user_message == "hello"
    assert contract.accepted_execution_spec.model.model_name == "chat-model"
    assert contract.preparation_spec.selected_toolsets == ()
    assert contract.preparation_spec.execution_permissions.web.allow_private_network_url is False
    assert contract.preparation_spec.execution_permissions.doc.allowed_read_paths == ()
    assert contract.preparation_spec.prompt_contributions["base_user"] == "# 用户与运行时上下文\n当前时间：2026年04月03日。"
    assert contract.preparation_spec.prompt_contributions["fins_default_subject"] == "# 当前分析对象\n你正在分析的是 AAPL（AAPL-NAME）。"
    assert contract.metadata == {}


@pytest.mark.unit
def test_stream_turn_persists_delivery_context_in_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证聊天请求的交付上下文会透传进 ExecutionContract.metadata。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )

    service, _resolver = _build_service()

    asyncio.run(
        _submit_turn_and_collect(
            service,
            ChatTurnRequest(
                session_id="s1",
                user_text=" hello ",
                scene_name="wechat",
                delivery_context=_delivery_context({
                    "delivery_channel": "wechat",
                    "delivery_target": "user-1",
                }),
            ),
        )
    )

    assert _last_execution_contract(service).metadata == {
        "delivery_channel": "wechat",
        "delivery_target": "user-1",
    }


@pytest.mark.unit
def test_stream_turn_supports_scene_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 ChatService 支持请求级 scene 覆盖。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )

    service, resolver = _build_service()

    asyncio.run(
        _submit_turn_and_collect(
            service,
            ChatTurnRequest(session_id="s1", user_text=" hello ", ticker="AAPL", scene_name="wechat"),
        )
    )

    assert resolver.calls == [{"scene_name": "wechat", "execution_options": None}]
    assert _last_execution_contract(service).scene_name == "wechat"
    assert _last_execution_contract(service).host_policy.resumable is True


@pytest.mark.unit
def test_chat_service_implements_protocol() -> None:
    """验证 ChatService 满足 ChatServiceProtocol 协议。"""

    service, _ = _build_service()
    assert isinstance(service, ChatServiceProtocol)


@pytest.mark.unit
def test_stream_turn_passes_same_host_session_on_multiple_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证多次调用会持续复用同一个 Host Session。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )

    service, _ = _build_service()

    async def _run_pair() -> None:
        await _submit_turn_and_collect(service, ChatTurnRequest(session_id="s1", user_text="q1", ticker="AAPL"))
        await _submit_turn_and_collect(service, ChatTurnRequest(session_id="s1", user_text="q2", ticker="AAPL"))

    asyncio.run(_run_pair())

    assert _last_execution_contract(service).host_policy.session_key == "s1"


@pytest.mark.unit
def test_stream_turn_raises_when_input_empty() -> None:
    """验证空聊天输入会触发 ValueError。"""

    service, _ = _build_service()

    with pytest.raises(ValueError):
        asyncio.run(
            _submit_turn_and_collect(
                service,
                ChatTurnRequest(session_id="s1", user_text="   ", ticker="AAPL"),
            )
        )


@pytest.mark.unit
def test_submit_turn_rejects_empty_input_before_creating_session() -> None:
    """验证 submit_turn 会在创建 session 前同步拒绝空输入。"""

    service, _resolver = _build_service()
    before_sessions = _session_registry(service).list_sessions()

    async def _submit() -> None:
        await service.submit_turn(ChatTurnRequest(user_text="   ", ticker="AAPL"))

    with pytest.raises(ValueError, match="聊天输入不能为空"):
        asyncio.run(_submit())

    after_sessions = _session_registry(service).list_sessions()
    assert [session.session_id for session in after_sessions] == [session.session_id for session in before_sessions]


@pytest.mark.unit
def test_submit_turn_rejects_missing_scene_before_creating_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 submit_turn 会在创建 session 前同步拒绝不存在的 scene。"""

    service, resolver = _build_service()
    before_sessions = _session_registry(service).list_sessions()

    def _raise_missing_scene(scene_name: str, execution_options: ExecutionOptions | None = None) -> AcceptedSceneExecution:
        del execution_options
        raise FileNotFoundError(scene_name)

    monkeypatch.setattr(resolver, "prepare", _raise_missing_scene)

    async def _submit() -> None:
        await service.submit_turn(ChatTurnRequest(user_text="hello", scene_name="missing_scene"))

    with pytest.raises(ValueError, match="scene 不存在: missing_scene"):
        asyncio.run(_submit())

    after_sessions = _session_registry(service).list_sessions()
    assert [session.session_id for session in after_sessions] == [session.session_id for session in before_sessions]


@pytest.mark.unit
def test_submit_turn_can_ensure_deterministic_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 `ENSURE_DETERMINISTIC` 时应由 Service 幂等获取或创建 session。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )

    service, _resolver = _build_service(session_source=SessionSource.WECHAT)

    asyncio.run(
        _submit_turn_and_collect(
            service,
            ChatTurnRequest(
                session_id="wechat_session_1",
                user_text=" hello ",
                ticker="AAPL",
                scene_name="wechat",
                session_resolution_policy=SessionResolutionPolicy.ENSURE_DETERMINISTIC,
            ),
        )
    )

    session = _session_registry(service).get_session("wechat_session_1")
    assert session is not None
    assert session.source == SessionSource.WECHAT
    assert _last_execution_contract(service).host_policy.session_key == "wechat_session_1"


@pytest.mark.unit
def test_resume_pending_turn_replays_prepared_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证恢复 pending turn 时会直接重放 Host prepared snapshot。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service(session_source=SessionSource.WECHAT)
    original_execution_options = ExecutionOptions(model_name="resume-model", max_iterations=7)

    asyncio.run(
        _submit_turn_and_collect(
            service,
            ChatTurnRequest(
                session_id="s1",
                user_text="历史问题",
                ticker="AAPL",
                scene_name="wechat",
                execution_options=original_execution_options,
                delivery_context=_delivery_context({
                    "delivery_channel": "wechat",
                    "delivery_target": "user-1",
                }),
            ),
        )
    )

    original_contract = _last_execution_contract(service)
    pending_turn_id = _seed_pending_turn_with_contract(
        service,
        execution_contract=original_contract,
        run_state="failed",
    )
    service.company_name_resolver = lambda _ticker: "CHANGED-NAME"

    events = asyncio.run(_consume_submission(service, pending_turn_id))

    assert [event.type for event in events] == [AppEventType.CONTENT_DELTA, AppEventType.FINAL_ANSWER]
    prepared_turn = _last_prepared_turn(service)
    conversation_session = _conversation_session(prepared_turn)
    assert conversation_session.user_message == "历史问题"
    assert conversation_session.session_id == "s1"
    assert prepared_turn.metadata == {
        "delivery_channel": "wechat",
        "delivery_target": "user-1",
    }


@pytest.mark.unit
def test_resume_pending_turn_replays_accepted_snapshot_via_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 accepted snapshot 恢复会走 prepare 路径，而不是丢失原始 accepted 真源。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service(session_source=SessionSource.WECHAT)
    original_execution_options = ExecutionOptions(model_name="resume-model", max_iterations=7)

    asyncio.run(
        _submit_turn_and_collect(
            service,
            ChatTurnRequest(
                session_id="s1",
                user_text="历史问题",
                ticker="AAPL",
                scene_name="wechat",
                execution_options=original_execution_options,
                delivery_context=_delivery_context({
                    "delivery_channel": "wechat",
                    "delivery_target": "user-1",
                }),
            ),
        )
    )

    original_contract = _last_execution_contract(service)
    pending_turn_id = _seed_accepted_pending_turn_with_contract(
        service,
        execution_contract=original_contract,
        run_state="failed",
    )
    service.company_name_resolver = lambda _ticker: "CHANGED-NAME"

    events = asyncio.run(_consume_submission(service, pending_turn_id))

    assert [event.type for event in events] == [AppEventType.CONTENT_DELTA, AppEventType.FINAL_ANSWER]
    resumed_contract = _last_execution_contract(service)
    assert resumed_contract.message_inputs.user_message == "历史问题"
    assert resumed_contract.host_policy.session_key == "s1"
    assert resumed_contract.preparation_spec.prompt_contributions["fins_default_subject"] == (
        "# 当前分析对象\n你正在分析的是 AAPL（AAPL-NAME）。"
    )


@pytest.mark.unit
def test_resume_pending_turn_uses_host_owned_prepared_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 pending turn 恢复后不再依赖 ExecutionOptions 快照。

    Args:
        monkeypatch: pytest monkeypatch 对象。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service(session_source=SessionSource.WECHAT)
    original_execution_options = ExecutionOptions(
        model_name="resume-model",
        max_iterations=7,
        toolset_config_overrides=(
            ToolsetConfigSnapshot(toolset_name="doc", payload={"list_files_max": 10}),
            ToolsetConfigSnapshot(toolset_name="fins", payload={"list_documents_max_items": 12}),
            ToolsetConfigSnapshot(toolset_name="web", payload={"max_search_results": 2}),
        ),
    )

    asyncio.run(
        _submit_turn_and_collect(
            service,
            ChatTurnRequest(
                session_id="s1",
                user_text="历史问题",
                ticker="AAPL",
                scene_name="wechat",
                execution_options=original_execution_options,
                delivery_context=_delivery_context({
                    "delivery_channel": "wechat",
                    "delivery_target": "user-1",
                }),
            ),
        )
    )

    original_contract = _last_execution_contract(service)
    pending_turn_id = _seed_pending_turn_with_contract(
        service,
        execution_contract=original_contract,
        run_state="failed",
    )

    events = asyncio.run(_consume_submission(service, pending_turn_id))

    assert [event.type for event in events] == [AppEventType.CONTENT_DELTA, AppEventType.FINAL_ANSWER]
    resumed_turn = _last_prepared_turn(service)
    assert resumed_turn.agent_create_args.model_name == "chat-model"
    assert _conversation_session(resumed_turn).user_message == "历史问题"


@pytest.mark.unit
def test_resume_pending_turn_rejects_active_source_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """活跃 source run 的 pending turn 不能恢复。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service()

    asyncio.run(_submit_turn_and_collect(service, ChatTurnRequest(session_id="s1", user_text="q", ticker="AAPL")))
    pending_turn_id = _seed_pending_turn_with_contract(
        service,
        execution_contract=_last_execution_contract(service),
        run_state="running",
    )

    with pytest.raises(ValueError, match="活跃状态"):
        asyncio.run(_consume_submission(service, pending_turn_id))
    pending_turn_record = _pending_turn_store(service).get_pending_turn_record(pending_turn_id)
    assert pending_turn_record is not None
    assert pending_turn_record.resume_attempt_count == 0
    assert pending_turn_record.last_resume_error_message is None


@pytest.mark.unit
def test_resume_pending_turn_deletes_record_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """连续执行期恢复失败达到上限后应删除 pending turn，避免卡住同槽位。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service()

    asyncio.run(_submit_turn_and_collect(service, ChatTurnRequest(session_id="s1", user_text="q", ticker="AAPL")))
    pending_turn_id = _seed_pending_turn_with_contract(
        service,
        execution_contract=_last_execution_contract(service),
        run_state="failed",
    )
    host_executor = _executor(service)

    async def _failing_run_prepared_turn_stream(_prepared_turn):
        raise RuntimeError("network boom")
        yield  # pragma: no cover

    monkeypatch.setattr(host_executor, "run_prepared_turn_stream", _failing_run_prepared_turn_stream)

    with pytest.raises(RuntimeError, match="network boom"):
        asyncio.run(_consume_submission(service, pending_turn_id))
    with pytest.raises(RuntimeError, match="network boom"):
        asyncio.run(_consume_submission(service, pending_turn_id))
    with pytest.raises(ValueError, match="已达到最大恢复次数"):
        asyncio.run(_consume_submission(service, pending_turn_id))

    assert _pending_turn_store(service).get_pending_turn_record(pending_turn_id) is None


@pytest.mark.unit
def test_resume_pending_turn_rejects_succeeded_source_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """source run 已成功完成的 pending turn 在 V1 中不能恢复。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service()

    asyncio.run(_submit_turn_and_collect(service, ChatTurnRequest(session_id="s1", user_text="q", ticker="AAPL")))
    pending_turn_id = _seed_pending_turn_with_contract(
        service,
        execution_contract=_last_execution_contract(service),
        run_state="succeeded",
    )

    with pytest.raises(ValueError, match="已成功完成"):
        asyncio.run(_consume_submission(service, pending_turn_id))
    assert _pending_turn_store(service).get_pending_turn_record(pending_turn_id) is None


@pytest.mark.unit
def test_resume_pending_turn_rejects_user_cancelled_source_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """用户主动取消的 pending turn 不能恢复。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service()

    asyncio.run(_submit_turn_and_collect(service, ChatTurnRequest(session_id="s1", user_text="q", ticker="AAPL")))
    pending_turn_id = _seed_pending_turn_with_contract(
        service,
        execution_contract=_last_execution_contract(service),
        run_state="user_cancelled",
    )

    with pytest.raises(ValueError, match="不是 timeout 取消"):
        asyncio.run(_consume_submission(service, pending_turn_id))
    assert _pending_turn_store(service).get_pending_turn_record(pending_turn_id) is None


@pytest.mark.unit
def test_resume_pending_turn_deletes_record_immediately_when_snapshot_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    """恢复快照损坏属于永久错误，应立即删除 pending turn 而不是消耗 attempt。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service()

    asyncio.run(_submit_turn_and_collect(service, ChatTurnRequest(session_id="s1", user_text="q", ticker="AAPL")))
    pending_turn_id = _seed_pending_turn_with_contract(
        service,
        execution_contract=_last_execution_contract(service),
        run_state="failed",
    )
    pending_turn_store = _pending_turn_store(service)
    pending_turn_record = pending_turn_store.get_pending_turn_record(pending_turn_id)
    assert pending_turn_record is not None
    pending_turn_store._records[pending_turn_id] = replace(
        pending_turn_record,
        resume_source_json='{"broken": [}',
    )

    with pytest.raises(ValueError, match="resume_source_json"):
        asyncio.run(_consume_submission(service, pending_turn_id))

    assert pending_turn_store.get_pending_turn_record(pending_turn_id) is None


@pytest.mark.unit
def test_resume_pending_turn_allows_timeout_cancelled_source_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """timeout 取消的 pending turn 允许恢复。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service()

    asyncio.run(_submit_turn_and_collect(service, ChatTurnRequest(session_id="s1", user_text="q", ticker="AAPL")))
    pending_turn_id = _seed_pending_turn_with_contract(
        service,
        execution_contract=_last_execution_contract(service),
        run_state="timeout_cancelled",
    )

    events = asyncio.run(_consume_submission(service, pending_turn_id))
    assert [event.type for event in events] == [AppEventType.CONTENT_DELTA, AppEventType.FINAL_ANSWER]


@pytest.mark.unit
def test_resume_pending_turn_emits_verbose_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """pending turn 恢复应输出恢复入口与恢复分支的 verbose 日志。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service()

    asyncio.run(_submit_turn_and_collect(service, ChatTurnRequest(session_id="s1", user_text="q", ticker="AAPL")))
    pending_turn_id = _seed_pending_turn_with_contract(
        service,
        execution_contract=_last_execution_contract(service),
        run_state="failed",
    )

    log_calls: list[str] = []
    monkeypatch.setattr(
        Log,
        "verbose",
        lambda message, *, module: log_calls.append(f"{module}:{message}"),
    )

    events = asyncio.run(_consume_submission(service, pending_turn_id))

    assert [event.type for event in events] == [AppEventType.CONTENT_DELTA, AppEventType.FINAL_ANSWER]
    assert any("HOST:开始恢复 pending turn" in item for item in log_calls)
    assert any("HOST:pending turn 按 prepared snapshot 恢复" in item for item in log_calls)


@pytest.mark.unit
def test_resume_pending_turn_rejects_cross_session_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    """pending turn 恢复必须校验 session ownership。"""

    monkeypatch.setattr(
        "dayu.services.chat_service.build_base_user_contribution",
        lambda: "# 用户与运行时上下文\n当前时间：2026年04月03日。",
    )
    service, _resolver = _build_service()

    asyncio.run(_submit_turn_and_collect(service, ChatTurnRequest(session_id="s1", user_text="q", ticker="AAPL")))
    pending_turn_id = _seed_pending_turn_with_contract(
        service,
        execution_contract=_last_execution_contract(service),
        run_state="failed",
    )

    with pytest.raises(ValueError, match="不属于当前 session"):
        asyncio.run(_consume_submission(service, pending_turn_id, session_id="other-session"))
