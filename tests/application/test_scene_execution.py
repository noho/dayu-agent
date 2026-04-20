"""Host 装配测试。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedInfrastructureSpec,
    AcceptedModelSpec,
    AcceptedRuntimeSpec,
    AcceptedToolConfigSpec,
    ExecutionDocPermissions,
    ExecutionPermissions,
    ExecutionWebPermissions,
)
from dayu.contracts.infrastructure import ModelCatalogProtocol, PromptAssetStoreProtocol
from dayu.contracts.model_config import ModelConfig, RunnerType
from dayu.contracts.protocols import PromptToolExecutorProtocol
from dayu.contracts.toolset_config import ToolsetConfigSnapshot, build_toolset_config_snapshot
from dayu.services.contract_preparation import prepare_execution_contract
from dayu.host.agent_builder import build_agent_running_config, build_async_runner
import dayu.host.scene_preparer as scene_preparer_module
from dayu.host.scene_preparer import DefaultScenePreparer, PreparedSceneState, _resolve_enabled_toolsets
from dayu.contracts.agent_execution import AgentCreateArgs
from dayu.engine.async_agent import AgentRunningConfig
from dayu.engine.tool_registry import ToolRegistry
from dayu.contracts.cancellation import CancellationToken
from dayu.engine.async_openai_runner import AsyncOpenAIRunner
from dayu.execution.runtime_config import (
    AgentRunningConfigSnapshot,
    AgentRuntimeConfig,
    FallbackMode,
    OpenAIRunnerRuntimeConfig,
    RunnerRunningConfigSnapshot,
)
from dayu.execution.options import (
    ConversationMemorySettings,
    DocToolLimits,
    ExecutionOptions,
    FinsToolLimits,
    ResolvedExecutionOptions,
    TraceSettings,
    WebToolsConfig,
    resolve_doc_tool_limits_from_toolset_configs,
    resolve_fins_tool_limits_from_toolset_configs,
    resolve_scene_temperature,
    resolve_scene_execution_options,
    resolve_web_tools_config_from_toolset_configs,
)
from dayu.host.host_execution import HostedRunContext
from dayu.prompting import SceneConversationDefinition, SceneDefinition, SceneModelDefinition
from dayu.prompting.prompt_plan import PromptAssemblyPlan
from dayu.prompting.scene_definition import ToolSelectionMode, ToolSelectionPolicy
from dayu.startup.workspace import WorkspaceResources


def _build_accepted_execution_spec(
    *,
    model_name: str,
    temperature: float | None,
    runner_running_config: RunnerRunningConfigSnapshot | None = None,
    agent_running_config: AgentRunningConfigSnapshot | None = None,
    doc_tool_limits: DocToolLimits | None = None,
    fins_tool_limits: FinsToolLimits | None = None,
    web_tools_config: WebToolsConfig | None = None,
    trace_settings: TraceSettings | None = None,
    conversation_memory_settings: ConversationMemorySettings | None = None,
) -> AcceptedExecutionSpec:
    """构造分组式 accepted execution spec。"""

    return AcceptedExecutionSpec(
        model=AcceptedModelSpec(model_name=model_name, temperature=temperature),
        runtime=AcceptedRuntimeSpec(
            runner_running_config=runner_running_config or {},
            agent_running_config=agent_running_config or {},
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


class _ConfigLoaderStub:
    """最小配置加载器桩。"""

    def load_run_config(self) -> dict[str, object]:
        """返回空运行配置。

        Args:
            无。

        Returns:
            空配置字典。

        Raises:
            无。
        """

        return {}

    def load_llm_models(self) -> dict[str, ModelConfig]:
        """返回空模型清单。

        Args:
            无。

        Returns:
            空模型映射。

        Raises:
            无。
        """

        return {}

    def load_llm_model(self, model_name: str) -> ModelConfig:
        """按名称读取模型配置。

        Args:
            model_name: 模型名称。

        Returns:
            测试模型配置。

        Raises:
            KeyError: 当前测试不应走到该分支时抛出。
        """

        raise KeyError(model_name)

    def load_toolset_registrars(self) -> dict[str, str]:
        """返回测试使用的 registrar 清单。

        Args:
            无。

        Returns:
            registrar 名称到导入路径的映射。

        Raises:
            无。
        """

        return {
            "web": "dayu.engine.toolset_registrars.register_web_toolset",
            "doc": "dayu.engine.toolset_registrars.register_doc_toolset",
            "utils": "dayu.engine.toolset_registrars.register_utils_toolset",
        }

    def collect_model_referenced_env_vars(self, model_names: Iterable[str]) -> tuple[str, ...]:
        """测试场景默认不声明模型环境变量依赖。"""

        del model_names
        return ()


class _MissingRegistrarConfigLoader(_ConfigLoaderStub):
    """返回缺失 registrar 清单的配置加载器桩。"""

    def load_toolset_registrars(self) -> dict[str, str]:
        """返回缺失 web registrar 的清单。

        Args:
            无。

        Returns:
            仅含 doc registrar 的映射。

        Raises:
            无。
        """

        return {"doc": "x.y.z"}


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


def test_build_async_runner_rejects_cli() -> None:
    """显式 AgentCreateArgs 若指定 CLI runner，应在 Host 层被拒绝。"""

    with pytest.raises(ValueError, match="CLI runner"):
        build_async_runner(
            AgentCreateArgs(
                runner_type=RunnerType.CLI,
                model_name="codex_cli",
                runner_params={
                    "command": ["codex", "exec"],
                    "timeout": 3600,
                },
            )
        )


def test_build_async_runner_openai() -> None:
    """显式 AgentCreateArgs 应在 Host 层装配 OpenAI 兼容 runner。"""

    runner = build_async_runner(
        AgentCreateArgs(
            runner_type=RunnerType.OPENAI_COMPATIBLE,
            model_name="test",
            temperature=0.2,
            runner_running_config={"tool_timeout_seconds": 15.0},
            runner_params={
                "endpoint_url": "http://example.com",
                "model": "test-model",
                "headers": {},
            },
        )
    )
    assert isinstance(runner, AsyncOpenAIRunner)


def test_build_async_runner_unsupported_type() -> None:
    """不支持的 runner_type 应在 Host 层被拒绝。"""

    with pytest.raises(ValueError, match="不支持的 runner_type"):
        build_async_runner(
            AgentCreateArgs(
                runner_type="unsupported_type",
                model_name="test",
            )
        )


def test_build_agent_running_config() -> None:
    """显式 AgentCreateArgs 应映射为 AgentRunningConfig。"""

    running_config = build_agent_running_config(
        AgentCreateArgs(
            runner_type=RunnerType.OPENAI_COMPATIBLE,
            model_name="test",
            max_turns=9,
            max_context_tokens=50000,
            max_output_tokens=4096,
            agent_running_config={"fallback_mode": FallbackMode.FORCE_ANSWER},
        )
    )
    assert running_config.max_iterations == 9
    assert running_config.max_context_tokens == 50000
    assert running_config.max_output_tokens == 4096
    assert running_config.fallback_mode == FallbackMode.FORCE_ANSWER


def test_resolve_scene_execution_options_applies_failed_batch_override() -> None:
    """scene manifest 应可覆盖连续失败工具批次上限。"""

    base_options = ResolvedExecutionOptions(
        model_name="",
        runner_running_config=OpenAIRunnerRuntimeConfig(),
        agent_running_config=AgentRuntimeConfig(),
        toolset_configs=_build_toolset_configs(
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider="off"),
        ),
        trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp/trace")),
        conversation_memory_settings=ConversationMemorySettings(),
    )

    resolved_options = resolve_scene_execution_options(
        base_execution_options=base_options,
        workspace_dir=Path("/tmp/workspace"),
        execution_options=None,
        default_model_name="test-model",
        allowed_model_names=("test-model",),
        scene_agent_max_iterations=24,
        scene_agent_max_consecutive_failed_tool_batches=5,
        scene_runner_tool_timeout_seconds=45.0,
        scene_name="prompt",
    )

    assert resolved_options.model_name == "test-model"
    assert resolved_options.agent_running_config.max_iterations == 24
    assert resolved_options.agent_running_config.max_consecutive_failed_tool_batches == 5
    assert cast(OpenAIRunnerRuntimeConfig, resolved_options.runner_running_config).tool_timeout_seconds == pytest.approx(45.0)


def test_resolve_scene_execution_options_prefers_request_failed_batch_override() -> None:
    """请求级覆盖应优先于 scene manifest 的失败工具批次上限。"""

    base_options = ResolvedExecutionOptions(
        model_name="",
        runner_running_config=OpenAIRunnerRuntimeConfig(),
        agent_running_config=AgentRuntimeConfig(),
        toolset_configs=_build_toolset_configs(
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider="off"),
        ),
        trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp/trace")),
        conversation_memory_settings=ConversationMemorySettings(),
    )

    resolved_options = resolve_scene_execution_options(
        base_execution_options=base_options,
        workspace_dir=Path("/tmp/workspace"),
        execution_options=ExecutionOptions(max_consecutive_failed_tool_batches=3),
        default_model_name="test-model",
        allowed_model_names=("test-model",),
        scene_agent_max_iterations=24,
        scene_agent_max_consecutive_failed_tool_batches=5,
        scene_runner_tool_timeout_seconds=45.0,
        scene_name="prompt",
    )

    assert resolved_options.agent_running_config.max_consecutive_failed_tool_batches == 3


def test_resolve_scene_execution_options_accepts_toolset_config_overrides() -> None:
    """请求级 toolset_config_overrides 应稀疏覆盖 base，未指定字段保留 base 值。"""

    base_options = ResolvedExecutionOptions(
        model_name="",
        runner_running_config=OpenAIRunnerRuntimeConfig(),
        agent_running_config=AgentRuntimeConfig(),
        toolset_configs=_build_toolset_configs(
            doc_tool_limits=DocToolLimits(list_files_max=500, get_sections_max=999),
            fins_tool_limits=FinsToolLimits(list_documents_max_items=800, read_section_max_chars=50000),
            web_tools_config=WebToolsConfig(provider="off", fetch_truncate_chars=12345),
        ),
        trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp/trace")),
        conversation_memory_settings=ConversationMemorySettings(),
    )

    resolved_options = resolve_scene_execution_options(
        base_execution_options=base_options,
        workspace_dir=Path("/tmp/workspace"),
        execution_options=ExecutionOptions(
            toolset_config_overrides=(
                ToolsetConfigSnapshot(
                    toolset_name="doc",
                    payload={"list_files_max": 33, "read_file_max_chars": 4096},
                ),
                ToolsetConfigSnapshot(
                    toolset_name="fins",
                    payload={"list_documents_max_items": 17, "read_section_max_chars": 1200},
                ),
                ToolsetConfigSnapshot(
                    toolset_name="web",
                    payload={"provider": "duckduckgo", "max_search_results": 5},
                ),
            ),
        ),
        default_model_name="test-model",
        allowed_model_names=("test-model",),
        scene_agent_max_iterations=24,
        scene_agent_max_consecutive_failed_tool_batches=5,
        scene_runner_tool_timeout_seconds=45.0,
        scene_name="prompt",
    )

    resolved_doc_limits = resolve_doc_tool_limits_from_toolset_configs(resolved_options.toolset_configs)
    resolved_fins_limits = resolve_fins_tool_limits_from_toolset_configs(resolved_options.toolset_configs)
    resolved_web_config = resolve_web_tools_config_from_toolset_configs(resolved_options.toolset_configs)

    # 显式覆盖的字段
    assert resolved_doc_limits is not None
    assert resolved_fins_limits is not None
    assert resolved_web_config is not None
    assert resolved_doc_limits.list_files_max == 33
    assert resolved_doc_limits.read_file_max_chars == 4096
    assert resolved_fins_limits.list_documents_max_items == 17
    assert resolved_fins_limits.read_section_max_chars == 1200
    assert resolved_web_config.provider == "duckduckgo"
    assert resolved_web_config.max_search_results == 5
    # 未覆盖的字段必须保留 base 值，而非回退到库默认值
    assert resolved_doc_limits.get_sections_max == 999
    assert resolved_fins_limits.read_section_max_chars == 1200
    assert resolved_web_config.fetch_truncate_chars == 12345


def test_resolve_scene_temperature_prefers_explicit_override() -> None:
    """显式 temperature 应优先于模型 profile。"""

    resolved_temperature = resolve_scene_temperature(
        resolved_temperature=0.4,
        model_config={
            "runtime_hints": {
                "temperature_profiles": {
                    "analysis": {"temperature": 0.9},
                }
            }
        },
        temperature_profile="analysis",
        scene_name="prompt",
        model_name="test-model",
    )

    assert resolved_temperature == pytest.approx(0.4)


def test_resolve_scene_temperature_uses_profile_when_request_omitted() -> None:
    """未显式覆盖时应回退到模型 temperature profile。"""

    resolved_temperature = resolve_scene_temperature(
        resolved_temperature=None,
        model_config={
            "runtime_hints": {
                "temperature_profiles": {
                    "analysis": {"temperature": 0.35},
                }
            }
        },
        temperature_profile="analysis",
        scene_name="prompt",
        model_name="test-model",
    )

    assert resolved_temperature == pytest.approx(0.35)


def test_resolve_scene_temperature_rejects_missing_profile() -> None:
    """缺失 profile 时应抛出稳定错误，避免 Service 与 Host 各自漂移。"""

    with pytest.raises(ValueError, match="temperature_profiles\\[analysis\\]"):
        resolve_scene_temperature(
            resolved_temperature=None,
            model_config={
                "runtime_hints": {
                    "temperature_profiles": {},
                }
            },
            temperature_profile="analysis",
            scene_name="prompt",
            model_name="test-model",
        )


def _build_scene_preparer(tmp_path: Path, *, web_provider: str) -> DefaultScenePreparer:
    """构造最小可用的 `DefaultScenePreparer`。"""

    workspace_dir = tmp_path / "workspace"
    config_root = workspace_dir / "config"
    output_dir = workspace_dir / "output"
    config_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = WorkspaceResources(
        workspace_dir=workspace_dir,
        config_root=config_root,
        output_dir=output_dir,
        config_loader=_ConfigLoaderStub(),
        prompt_asset_store=cast(PromptAssetStoreProtocol, SimpleNamespace()),
    )
    resolved_options = ResolvedExecutionOptions(
        model_name="test-model",
        runner_running_config=OpenAIRunnerRuntimeConfig(),
        agent_running_config=AgentRuntimeConfig(),
        toolset_configs=_build_toolset_configs(
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider=web_provider),
        ),
        trace_settings=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
        conversation_memory_settings=ConversationMemorySettings(),
    )
    return DefaultScenePreparer(
        workspace=workspace,  # type: ignore[arg-type]
        model_catalog=cast(ModelCatalogProtocol, SimpleNamespace()),
        default_execution_options=resolved_options,
        tool_registry_factory=lambda: ToolRegistry(),
    )


@pytest.mark.unit
def test_resolve_enabled_toolsets_select_mode_uses_manifest_when_selected_empty() -> None:
    """验证 select 模式在 selected_toolsets 为空时直接使用 manifest 候选集合。"""

    scene_definition = SceneDefinition(
        name="prompt",
        model=SceneModelDefinition(default_name="test-model"),
        version="v1",
        description="test",
        tool_selection_policy=ToolSelectionPolicy(
            mode=ToolSelectionMode.SELECT,
            tool_tags_any=("web", "fins", "ingestion"),
        ),
    )

    assert _resolve_enabled_toolsets(
        scene_definition=scene_definition,
        selected_toolsets=(),
        installed_toolsets=("doc", "web", "fins", "ingestion"),
    ) == ("web", "fins", "ingestion")


@pytest.mark.unit
def test_resolve_enabled_toolsets_select_mode_intersects_selected_toolsets() -> None:
    """验证 select 模式会与 selected_toolsets 做求交。"""

    scene_definition = SceneDefinition(
        name="prompt",
        model=SceneModelDefinition(default_name="test-model"),
        version="v1",
        description="test",
        tool_selection_policy=ToolSelectionPolicy(
            mode=ToolSelectionMode.SELECT,
            tool_tags_any=("web", "fins", "ingestion"),
        ),
    )

    assert _resolve_enabled_toolsets(
        scene_definition=scene_definition,
        selected_toolsets=("doc", "fins"),
        installed_toolsets=("doc", "web", "fins", "ingestion"),
    ) == ("fins",)


@pytest.mark.unit
def test_scene_preparer_skips_web_tools_when_provider_off(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 provider=off 时 Host 不会注册 web tools。"""

    preparer = _build_scene_preparer(tmp_path, web_provider="off")
    calls: list[dict[str, object]] = []

    def _record_register_web_tools(*_args, **kwargs) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr("dayu.engine.toolset_registrars.register_web_tools", _record_register_web_tools)

    registry = preparer._build_tool_registry(
        scene_definition=SceneDefinition(
            name="prompt",
            model=SceneModelDefinition(default_name="test-model"),
            version="v1",
            description="test",
            tool_selection_policy=ToolSelectionPolicy(
                mode=ToolSelectionMode.SELECT,
                tool_tags_any=("web",),
            ),
        ),
        selected_toolsets=(),
        execution_permissions=ExecutionPermissions(
            web=ExecutionWebPermissions(allow_private_network_url=False),
            doc=ExecutionDocPermissions(),
        ),
        resolved_options=preparer.default_execution_options,
        tool_timeout_seconds=None,
    )

    tool_names = set(registry.get_tool_names())
    assert "search_web" not in tool_names
    assert "fetch_web_page" not in tool_names
    assert calls == []


@pytest.mark.unit
def test_scene_preparer_raises_when_enabled_toolset_has_no_registrar(
    tmp_path: Path,
) -> None:
    """验证最终启用的 toolset 若缺少 registrar，会在 Host 主链显式失败。"""

    preparer = _build_scene_preparer(tmp_path, web_provider="off")
    preparer.workspace = WorkspaceResources(
        workspace_dir=preparer.workspace.workspace_dir,
        config_root=preparer.workspace.config_root,
        output_dir=preparer.workspace.output_dir,
        config_loader=_MissingRegistrarConfigLoader(),
        prompt_asset_store=preparer.workspace.prompt_asset_store,
    )

    with pytest.raises(ValueError, match="缺失 registrar"):
        preparer._build_tool_registry(
            scene_definition=SceneDefinition(
                name="prompt",
                model=SceneModelDefinition(default_name="test-model"),
                version="v1",
                description="test",
                tool_selection_policy=ToolSelectionPolicy(
                    mode=ToolSelectionMode.SELECT,
                    tool_tags_any=("web",),
                ),
            ),
            selected_toolsets=(),
            execution_permissions=ExecutionPermissions(
                web=ExecutionWebPermissions(allow_private_network_url=False),
                doc=ExecutionDocPermissions(),
            ),
            resolved_options=preparer.default_execution_options,
            tool_timeout_seconds=None,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scene_preparer_uses_manifest_conversation_flag_for_custom_scene(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证多轮判定来自 manifest 的 conversation.enabled，而不是 scene 名硬编码。"""

    preparer = _build_scene_preparer(tmp_path, web_provider="off")
    scene_definition = SceneDefinition(
        name="custom_scene",
        model=SceneModelDefinition(default_name="test-model"),
        version="v1",
        description="test",
        conversation=SceneConversationDefinition(enabled=True),
    )
    prepared_scene = PreparedSceneState(
        scene_name="custom_scene",
        scene_definition=scene_definition,
        resolved_options=preparer.default_execution_options,
        model_config=cast(ModelConfig, {"name": "test-model", "temperature": 0.2}),
        prompt_asset_store=cast(PromptAssetStoreProtocol, object()),
        tool_registry=cast(PromptToolExecutorProtocol, SimpleNamespace()),
        agent_create_args=AgentCreateArgs(runner_type="openai_compatible", model_name="test-model"),
        conversation_memory_settings=ConversationMemorySettings(),
    )

    monkeypatch.setattr("dayu.host.scene_preparer.load_scene_definition", lambda *_args, **_kwargs: scene_definition)
    monkeypatch.setattr(
        preparer,
        "_prepare_scene_state_from_contract",
        lambda **_kwargs: prepared_scene,
    )
    monkeypatch.setattr(
        preparer,
        "_compose_system_prompt",
        lambda **_kwargs: "SYS",
    )

    async def _prepare_transcript(**kwargs):
        return kwargs["transcript"]

    monkeypatch.setattr(preparer._memory_manager, "prepare_transcript", _prepare_transcript)
    monkeypatch.setattr(
        preparer._memory_manager,
        "build_messages",
        lambda **_kwargs: [{"role": "system", "content": "multi-turn"}],
    )

    contract = prepare_execution_contract(
        service_name="chat_turn",
        scene_name="custom_scene",
        accepted_execution_spec=_build_accepted_execution_spec(
            model_name="test-model",
            temperature=0.2,
            runner_running_config={},
            agent_running_config={"max_iterations": 8},
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider="off"),
            trace_settings=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
            conversation_memory_settings=ConversationMemorySettings(),
        ),
        prompt_contributions={"base_user": "x"},
        user_message="hello",
        session_key="session-custom",
        concurrency_lane="llm_api",
        timeout_ms=None,
        resumable=True,
    )

    prepared_execution = await preparer.prepare(
        contract,
        HostedRunContext(run_id="run_1", cancellation_token=CancellationToken()),
    )
    agent_input = prepared_execution.agent_input

    assert agent_input.session_state is not None
    assert agent_input.messages == [{"role": "system", "content": "multi-turn"}]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scene_preparer_rejects_resumable_when_conversation_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 resumable 不能用于 conversation.enabled=false 的 scene。"""

    preparer = _build_scene_preparer(tmp_path, web_provider="off")
    scene_definition = SceneDefinition(
        name="single_turn_scene",
        model=SceneModelDefinition(default_name="test-model"),
        version="v1",
        description="test",
        conversation=SceneConversationDefinition(enabled=False),
    )

    monkeypatch.setattr("dayu.host.scene_preparer.load_scene_definition", lambda *_args, **_kwargs: scene_definition)

    contract = prepare_execution_contract(
        service_name="chat_turn",
        scene_name="single_turn_scene",
        accepted_execution_spec=_build_accepted_execution_spec(
            model_name="test-model",
            temperature=0.2,
            runner_running_config={},
            agent_running_config={"max_iterations": 8},
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider="off"),
            trace_settings=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
            conversation_memory_settings=ConversationMemorySettings(),
        ),
        prompt_contributions={"base_user": "x"},
        user_message="hello",
        session_key="session-custom",
        concurrency_lane="llm_api",
        timeout_ms=None,
        resumable=True,
    )

    with pytest.raises(ValueError, match="conversation.enabled=true"):
        await preparer.prepare(
            contract,
            HostedRunContext(run_id="run_1", cancellation_token=CancellationToken()),
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scene_preparer_rejects_resumable_without_session_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 resumable 要求稳定 session_key。"""

    preparer = _build_scene_preparer(tmp_path, web_provider="off")
    scene_definition = SceneDefinition(
        name="custom_scene",
        model=SceneModelDefinition(default_name="test-model"),
        version="v1",
        description="test",
        conversation=SceneConversationDefinition(enabled=True),
    )

    monkeypatch.setattr("dayu.host.scene_preparer.load_scene_definition", lambda *_args, **_kwargs: scene_definition)

    contract = prepare_execution_contract(
        service_name="chat_turn",
        scene_name="custom_scene",
        accepted_execution_spec=_build_accepted_execution_spec(
            model_name="test-model",
            temperature=0.2,
            runner_running_config={},
            agent_running_config={"max_iterations": 8},
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider="off"),
            trace_settings=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
            conversation_memory_settings=ConversationMemorySettings(),
        ),
        prompt_contributions={"base_user": "x"},
        user_message="hello",
        session_key=None,
        concurrency_lane="llm_api",
        timeout_ms=None,
        resumable=True,
    )

    with pytest.raises(ValueError, match="session_key"):
        await preparer.prepare(
            contract,
            HostedRunContext(run_id="run_1", cancellation_token=CancellationToken()),
        )


@pytest.mark.unit
def test_scene_preparer_warns_and_ignores_unknown_prompt_contributions_slots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 Host 遇到未声明 slot 时只告警并忽略，而不会把 scene 组装升级成失败。"""

    preparer = _build_scene_preparer(tmp_path, web_provider="off")
    warnings: list[str] = []
    scene_definition = SceneDefinition(
        name="custom_scene",
        model=SceneModelDefinition(default_name="test-model"),
        version="v1",
        description="test",
        context_slots=("base_user",),
    )
    prepared_scene = PreparedSceneState(
        scene_name="custom_scene",
        scene_definition=scene_definition,
        resolved_options=preparer.default_execution_options,
        model_config=cast(ModelConfig, {"name": "test-model", "supports_tool_calling": False}),
        prompt_asset_store=cast(PromptAssetStoreProtocol, object()),
        tool_registry=cast(
            PromptToolExecutorProtocol,
            SimpleNamespace(
                get_tool_names=lambda: (),
                get_tool_tags=lambda: (),
                get_allowed_paths=lambda: (),
            ),
        ),
        agent_create_args=AgentCreateArgs(runner_type="openai_compatible", model_name="test-model"),
        conversation_memory_settings=ConversationMemorySettings(),
    )

    monkeypatch.setattr(
        "dayu.host.scene_preparer.build_prompt_assembly_plan",
        lambda **_kwargs: PromptAssemblyPlan(name="custom_scene", version="v1", fragments=()),
    )
    monkeypatch.setattr(
        "dayu.host.scene_preparer.Log.warn",
        lambda message, **_kwargs: warnings.append(str(message)),
    )

    system_prompt = preparer._compose_system_prompt(
        prepared_scene=prepared_scene,
        prompt_contributions={
            "base_user": "# 用户与运行时上下文\n当前时间：2026年04月13日。",
            "unexpected": "should be ignored",
        },
    )

    assert system_prompt == "# 用户与运行时上下文\n当前时间：2026年04月13日。"
    assert warnings == [
        "检测到未在 scene manifest 中声明的 Prompt Contributions slot，已忽略: scene=custom_scene, slots=['unexpected']"
    ]


@pytest.mark.unit
def test_scene_preparer_helper_functions_cover_toolset_merge_and_registrar_validation() -> None:
    """验证 scene_preparer helper 的工具集合与 registrar 分支。"""

    scene_definition_all = SceneDefinition(
        name="prompt",
        model=SceneModelDefinition(default_name="test-model"),
        version="v1",
        description="test",
        tool_selection_policy=ToolSelectionPolicy(mode=ToolSelectionMode.ALL),
    )
    scene_definition_none = SceneDefinition(
        name="prompt",
        model=SceneModelDefinition(default_name="test-model"),
        version="v1",
        description="test",
        tool_selection_policy=ToolSelectionPolicy(mode=ToolSelectionMode.NONE),
    )

    merged = scene_preparer_module._merge_toolset_configs(
        (ToolsetConfigSnapshot(toolset_name="doc", payload={"list_files_max": 10}),),
        (
            ToolsetConfigSnapshot(toolset_name="doc", payload={"list_files_max": 20}),
            ToolsetConfigSnapshot(toolset_name="web", payload={"provider": "duckduckgo"}),
        ),
    )

    assert scene_preparer_module._normalize_toolset_names(["doc", " ", "doc", "web"]) == ("doc", "web")
    assert _resolve_enabled_toolsets(
        scene_definition=scene_definition_all,
        selected_toolsets=("doc",),
        installed_toolsets=("doc", "web"),
    ) == ("doc",)
    assert _resolve_enabled_toolsets(
        scene_definition=scene_definition_none,
        selected_toolsets=("doc",),
        installed_toolsets=("doc", "web"),
    ) == ()
    assert merged == (
        ToolsetConfigSnapshot(toolset_name="doc", payload={"list_files_max": 20}),
        ToolsetConfigSnapshot(toolset_name="web", payload={"provider": "duckduckgo"}),
    )

    with pytest.raises(ValueError, match="非法 toolset registrar"):
        scene_preparer_module._load_toolset_registrar("bad-path")
    with pytest.raises(ImportError, match="无法导入"):
        scene_preparer_module._load_toolset_registrar("missing.module.registrar")


@pytest.mark.unit
def test_scene_preparer_helper_functions_cover_single_turn_and_execution_option_edges(
    tmp_path: Path,
) -> None:
    """验证 scene_preparer 的单轮消息、tool summary 与 execution option helper。"""

    base_options = ResolvedExecutionOptions(
        model_name="base-model",
        runner_running_config=OpenAIRunnerRuntimeConfig(tool_timeout_seconds=12.0),
        agent_running_config=AgentRuntimeConfig(max_iterations=5),
        toolset_configs=_build_toolset_configs(
            doc_tool_limits=DocToolLimits(list_files_max=20),
            web_tools_config=WebToolsConfig(provider="off"),
        ),
        trace_settings=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
        conversation_memory_settings=ConversationMemorySettings(),
    )
    scene_definition = SceneDefinition(
        name="prompt",
        model=SceneModelDefinition(default_name="scene-model", allowed_names=("scene-model", "base-model")),
        version="v1",
        description="test",
    )
    contract = prepare_execution_contract(
        service_name="chat_turn",
        scene_name="prompt",
        accepted_execution_spec=_build_accepted_execution_spec(
            model_name="accepted-model",
            temperature=0.35,
            runner_running_config={"tool_timeout_seconds": 18.0},
            agent_running_config={"max_iterations": 9},
            doc_tool_limits=DocToolLimits(list_files_max=33),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider="off"),
            trace_settings=TraceSettings(enabled=False, output_dir=tmp_path / "trace-accepted"),
            conversation_memory_settings=ConversationMemorySettings(compaction_trigger_turn_count=6),
        ),
        prompt_contributions={},
        user_message="hello",
        session_key=None,
        concurrency_lane=None,
        timeout_ms=4000,
        resumable=False,
    )

    assert scene_preparer_module._build_single_turn_messages(system_prompt="", user_message="hello") == [
        {"role": "user", "content": "hello"}
    ]
    assert scene_preparer_module._build_single_turn_messages(system_prompt="SYS", user_message="hello") == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hello"},
    ]
    tool_summary = scene_preparer_module.ConversationToolUseSummary(name="search")
    assert scene_preparer_module._coerce_tool_use_summaries((tool_summary,)) == (tool_summary,)
    with pytest.raises(TypeError, match="ConversationToolUseSummary"):
        scene_preparer_module._coerce_tool_use_summaries((object(),))
    assert scene_preparer_module._extract_tool_timeout_seconds(
        AgentCreateArgs(
            runner_type="openai_compatible",
            model_name="test-model",
            runner_running_config={"tool_timeout_seconds": 7},
        )
    ) == pytest.approx(7.0)
    assert (
        scene_preparer_module._extract_tool_timeout_seconds(
            AgentCreateArgs(
                runner_type="openai_compatible",
                model_name="test-model",
                runner_running_config={"tool_timeout_seconds": True},
            )
        )
        is None
    )

    resolved_options = scene_preparer_module._resolve_scene_execution_options(
        base_execution_options=base_options,
        workspace_dir=tmp_path,
        scene_definition=scene_definition,
        execution_options=ExecutionOptions(max_consecutive_failed_tool_batches=4),
    )
    rebuilt_options = scene_preparer_module._build_resolved_execution_options_from_contract(
        base_options=base_options,
        execution_contract=contract,
    )

    assert resolved_options.model_name == "scene-model"
    assert resolved_options.agent_running_config.max_consecutive_failed_tool_batches == 4
    assert rebuilt_options.model_name == "accepted-model"
    assert rebuilt_options.temperature == pytest.approx(0.35)


@pytest.mark.unit
def test_scene_preparer_conversation_session_state_persist_turn_saves_and_schedules_compaction(
    tmp_path: Path,
) -> None:
    """验证 conversation session state 会持久化 turn 并调度压缩。"""

    preparer = _build_scene_preparer(tmp_path, web_provider="off")
    prepared_scene = PreparedSceneState(
        scene_name="prompt",
        scene_definition=SceneDefinition(
            name="prompt",
            model=SceneModelDefinition(default_name="test-model"),
            version="v1",
            description="test",
        ),
        resolved_options=preparer.default_execution_options,
        model_config=cast(ModelConfig, {"name": "test-model"}),
        prompt_asset_store=cast(PromptAssetStoreProtocol, object()),
        tool_registry=cast(PromptToolExecutorProtocol, SimpleNamespace()),
        agent_create_args=AgentCreateArgs(runner_type="openai_compatible", model_name="test-model"),
        conversation_memory_settings=ConversationMemorySettings(),
    )
    transcript = scene_preparer_module.ConversationTranscript.create_empty("session-1")
    persisted: list[object] = []
    scheduled: list[object] = []

    def _save(next_transcript, *, expected_revision):
        persisted.extend([next_transcript, expected_revision])
        return next_transcript

    def _schedule_compaction(*, session_id, prepared_scene, transcript):
        scheduled.extend([session_id, prepared_scene, transcript])

    session_state = scene_preparer_module.ConversationSessionState(
        session_id="session-1",
        scene_name="prompt",
        transcript=transcript,
        conversation_store=cast(scene_preparer_module.ConversationStore, SimpleNamespace(save=_save)),
        memory_manager=cast(scene_preparer_module.DefaultConversationMemoryManager, SimpleNamespace(schedule_compaction=_schedule_compaction)),
        prepared_scene=prepared_scene,
        user_message="hello",
    )

    session_state.persist_turn(
        final_content="done",
        degraded=False,
        tool_uses=(scene_preparer_module.ConversationToolUseSummary(name="search"),),
        warnings=("warn",),
        errors=("err",),
    )

    saved_transcript = cast(scene_preparer_module.ConversationTranscript, persisted[0])
    assert persisted[1] == transcript.revision
    assert len(saved_transcript.turns) == 1
    assert saved_transcript.turns[0].assistant_final == "done"
    assert saved_transcript.turns[0].warnings == ("warn",)
    assert scheduled[0] == "session-1"
    assert scheduled[1] is prepared_scene
    assert scheduled[2] is saved_transcript


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scene_preparer_prepare_restore_and_host_owned_state_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 host-owned state、prepared state 恢复与 restore_prepared_execution helper。"""

    preparer = _build_scene_preparer(tmp_path, web_provider="off")
    scene_definition = SceneDefinition(
        name="prompt",
        model=SceneModelDefinition(default_name="test-model", allowed_names=("test-model",)),
        version="v1",
        description="test",
    )
    tool_registry = cast(
        PromptToolExecutorProtocol,
        SimpleNamespace(
            get_tool_names=lambda: (),
            get_tool_tags=lambda: (),
            get_allowed_paths=lambda: (),
        ),
    )
    preparer.model_catalog = cast(
        ModelCatalogProtocol,
        SimpleNamespace(load_model=lambda _name: {"name": "test-model", "supports_tool_calling": False}),
    )
    monkeypatch.setattr("dayu.host.scene_preparer.load_scene_definition", lambda *_args, **_kwargs: scene_definition)
    monkeypatch.setattr(
        "dayu.host.scene_preparer.build_agent_create_args",
        lambda **_kwargs: AgentCreateArgs(
            runner_type="openai_compatible",
            model_name="test-model",
            temperature=0.25,
            runner_running_config={"tool_timeout_seconds": 11.0},
            agent_running_config={"max_iterations": 7},
        ),
    )
    monkeypatch.setattr(preparer, "_build_tool_registry", lambda **_kwargs: tool_registry)
    monkeypatch.setattr(preparer._trace_provider, "get_or_create", lambda _settings: "trace-factory")

    contract = prepare_execution_contract(
        service_name="chat_turn",
        scene_name="prompt",
        accepted_execution_spec=_build_accepted_execution_spec(
            model_name="test-model",
            temperature=0.25,
            runner_running_config={"tool_timeout_seconds": 11.0},
            agent_running_config={"max_iterations": 7},
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider="off"),
            trace_settings=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
            conversation_memory_settings=ConversationMemorySettings(compaction_trigger_turn_count=5),
        ),
        prompt_contributions={},
        user_message="hello",
        session_key=None,
        concurrency_lane=None,
        timeout_ms=2000,
        resumable=False,
    )

    prepared_scene = preparer._prepare_scene_state_from_contract(
        execution_contract=contract,
        scene_definition=scene_definition,
    )
    host_owned_scene = preparer._prepare_host_owned_scene_state(
        scene_name="prompt",
        execution_options=ExecutionOptions(temperature=0.1, max_consecutive_failed_tool_batches=3),
        web_tools_config=WebToolsConfig(provider="duckduckgo", allow_private_network_url=True),
    )

    prepared_turn = scene_preparer_module.PreparedAgentTurnSnapshot(
        service_name="chat_turn",
        scene_name="prompt",
        metadata={"delivery_channel": "interactive"},
        concurrency_lane=None,
        timeout_ms=1234,
        resumable=True,
        system_prompt="SYS",
        messages=[{"role": "user", "content": "hello"}],
        agent_create_args=AgentCreateArgs(
            runner_type="openai_compatible",
            model_name="test-model",
            temperature=0.25,
            runner_running_config={"tool_timeout_seconds": 11.0},
            agent_running_config={"max_iterations": 7},
        ),
        selected_toolsets=("doc",),
        execution_permissions=ExecutionPermissions(
            web=ExecutionWebPermissions(allow_private_network_url=False),
            doc=ExecutionDocPermissions(),
        ),
        toolset_configs=(ToolsetConfigSnapshot(toolset_name="doc", payload={"list_files_max": 9}),),
        trace_settings=TraceSettings(enabled=False, output_dir=tmp_path / "trace-restored"),
        conversation_memory_settings=ConversationMemorySettings(compaction_trigger_turn_count=4),
        trace_identity=scene_preparer_module.AgentTraceIdentity(
            agent_name="prompt_agent",
            agent_kind="scene_agent",
            scene_name="prompt",
            model_name="test-model",
            session_id="session-1",
        ),
        conversation_session=scene_preparer_module.PreparedConversationSessionSnapshot(
            session_id="session-1",
            user_message="hello",
            transcript=scene_preparer_module.ConversationTranscript.create_empty("session-1"),
        ),
    )
    restored_input = await preparer.restore_prepared_execution(
        prepared_turn,
        HostedRunContext(run_id="run-1", cancellation_token=CancellationToken()),
    )

    assert prepared_scene.tool_trace_recorder_factory == "trace-factory"
    assert prepared_scene.trace_identity is not None
    assert host_owned_scene.scene_name == "prompt"
    assert host_owned_scene.tool_trace_recorder_factory == "trace-factory"
    assert restored_input.system_prompt == "SYS"
    assert restored_input.session_state is not None
    restored_session_state = cast(scene_preparer_module.ConversationSessionState, restored_input.session_state)
    assert restored_session_state.session_id == "session-1"


@pytest.mark.unit
def test_scene_preparer_conversation_runtime_adapter_covers_compaction_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 conversation runtime adapter 会复用 host-owned scene 与 compaction agent 组装。"""

    preparer = _build_scene_preparer(tmp_path, web_provider="off")
    prepared_scene = PreparedSceneState(
        scene_name="prompt",
        scene_definition=SceneDefinition(
            name="prompt",
            model=SceneModelDefinition(default_name="test-model"),
            version="v1",
            description="test",
        ),
        resolved_options=preparer.default_execution_options,
        model_config=cast(ModelConfig, {"name": "test-model", "supports_tool_calling": False}),
        prompt_asset_store=cast(PromptAssetStoreProtocol, object()),
        tool_registry=cast(PromptToolExecutorProtocol, SimpleNamespace()),
        agent_create_args=AgentCreateArgs(runner_type="openai_compatible", model_name="test-model"),
        conversation_memory_settings=ConversationMemorySettings(),
        trace_identity=scene_preparer_module.AgentTraceIdentity(
            agent_name="prompt_agent",
            agent_kind="scene_agent",
            scene_name="prompt",
            model_name="test-model",
            session_id="original-session",
        ),
    )

    monkeypatch.setattr(preparer, "_prepare_host_owned_scene_state", lambda **_kwargs: prepared_scene)
    monkeypatch.setattr(preparer, "_compose_system_prompt", lambda **_kwargs: "SYS")
    monkeypatch.setattr(
        "dayu.host.scene_preparer.build_async_agent",
        lambda **kwargs: kwargs,
    )

    resolved_scene = preparer._conversation_runtime.prepare_compaction_scene("prompt")
    handle = preparer._conversation_runtime.prepare_compaction_agent(
        prepared_scene,
        cast(scene_preparer_module.ConversationCompactionRequest, SimpleNamespace(session_id=" session-2 ")),
    )

    assert resolved_scene is prepared_scene
    assert handle.system_prompt == "SYS"
    agent_payload = cast(dict[str, object], handle.agent)
    trace_identity = cast(scene_preparer_module.AgentTraceIdentity, agent_payload["trace_identity"])
    assert trace_identity.session_id == "session-2"
