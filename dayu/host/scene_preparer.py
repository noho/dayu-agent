"""Host 内部的 scene preparation 实现。"""

from __future__ import annotations

from importlib import import_module
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, cast, runtime_checkable

from dayu.contracts.agent_execution import (
    AgentCreateArgs,
    AgentInput,
    ExecutionContract,
    ExecutionDocPermissions,
    ExecutionPermissions,
    ExecutionWebPermissions,
)
from dayu.contracts.agent_types import AgentMessage, AgentRuntimeLimits, AgentTraceIdentity
from dayu.contracts.execution_options import ExecutionOptions
from dayu.contracts.model_config import ModelConfig
from dayu.contracts.protocols import PromptToolExecutorProtocol, ToolTraceRecorderFactory
from dayu.contracts.tool_configs import WebToolsConfig
from dayu.contracts.toolset_config import (
    build_toolset_config_snapshot,
    find_toolset_config,
    normalize_toolset_configs,
    replace_toolset_config,
)
from dayu.contracts.toolset_registrar import (
    ToolRegistryProtocol,
    ToolsetRegistrarProtocol,
    ToolsetRegistrationContext,
)
from dayu.log import Log
from dayu.execution.runtime_config import build_agent_running_config_from_snapshot, build_runner_running_config_from_snapshot
from dayu.execution.options import (
    ConversationMemorySettings,
    ResolvedExecutionOptions,
    resolve_web_tools_config_from_toolset_configs,
    resolve_scene_execution_options,
    resolve_scene_temperature,
    resolve_conversation_memory_settings,
)
from dayu.host.agent_builder import build_agent_create_args, build_async_agent
from dayu.host.conversation_memory import DefaultConversationMemoryManager
from dayu.host.conversation_runtime import (
    ConversationCompactionAgentHandle,
    ConversationCompactionRequest,
    ConversationCompactionSceneProtocol,
)
from dayu.host.conversation_store import (
    ConversationStore,
    ConversationToolUseSummary,
    ConversationTranscript,
    ConversationTurnRecord,
    FileConversationStore,
)
from dayu.host.host_execution import HostedRunContext
from dayu.host.prepared_turn import PreparedAgentTurnSnapshot, PreparedConversationSessionSnapshot
from dayu.host.trace_infrastructure import TraceRecorderFactoryProvider
from dayu.prompting.prompt_composer import PromptComposeContext, PromptComposer
from dayu.prompting.prompt_contribution_slots import select_prompt_contributions
from dayu.prompting.prompt_plan import build_prompt_assembly_plan
from dayu.prompting.scene_definition import SceneDefinition, ToolSelectionMode, load_scene_definition
from dayu.prompting.tool_snapshot import build_prompt_tool_snapshot
from dayu.contracts.infrastructure import ModelCatalogProtocol, PromptAssetStoreProtocol, WorkspaceResourcesProtocol
from dayu.workspace_paths import build_conversation_store_dir

MODULE = "HOST.SCENE_PREPARER"


def _merge_toolset_configs(
    base_configs: tuple[object, ...],
    incoming_configs: tuple[object, ...],
) -> tuple[object, ...]:
    """按 toolset 名称把 incoming 快照覆盖到 base 快照。"""

    merged_configs = normalize_toolset_configs(cast(tuple[Any, ...], base_configs))
    for snapshot in normalize_toolset_configs(cast(tuple[Any, ...], incoming_configs)):
        merged_configs = replace_toolset_config(merged_configs, snapshot)
    return cast(tuple[object, ...], merged_configs)


def _validate_resumable_host_policy(
    *,
    execution_contract: ExecutionContract,
    scene_definition: SceneDefinition,
) -> None:
    """校验 resumable 的 Host 生效前提。

    Args:
        execution_contract: Service 提交给 Host 的执行契约。
        scene_definition: 当前 scene 定义。

    Returns:
        无。

    Raises:
        ValueError: 当 resumable 请求不满足 Host 恢复前提时抛出。
    """

    if not execution_contract.host_policy.resumable:
        return
    if not scene_definition.conversation.enabled:
        raise ValueError(
            "resumable 仅允许用于 conversation.enabled=true 的 scene: "
            f"scene={scene_definition.name}"
        )
    session_key = str(execution_contract.host_policy.session_key or "").strip()
    if not session_key:
        raise ValueError(
            "resumable 要求非空 session_key，Host 只能在稳定 session 下恢复 pending conversation turn"
        )


@runtime_checkable
class ScenePreparationProtocol(Protocol):
    """scene preparation 协议。"""

    async def prepare(
        self,
        execution_contract: ExecutionContract,
        run_context: HostedRunContext,
    ) -> "PreparedAgentExecution":
        """把 ``ExecutionContract`` 收敛为可执行输入与可恢复快照。"""
        ...

    async def restore_prepared_execution(
        self,
        prepared_turn: PreparedAgentTurnSnapshot,
        run_context: HostedRunContext,
    ) -> AgentInput:
        """从 prepared turn 快照重建 ``AgentInput``。"""
        ...


@runtime_checkable
class PreparedToolRegistryProtocol(ToolRegistryProtocol, PromptToolExecutorProtocol, Protocol):
    """scene preparation 需要的最小工具注册表协议。"""


ToolRegistryFactory = Callable[[], PreparedToolRegistryProtocol]


@dataclass(frozen=True)
class PreparedAgentExecution:
    """scene preparation 的产物。

    Args:
        agent_input: 立即可执行的 Agent 输入。
        resume_snapshot: Host 用于持久化恢复的 prepared turn 快照。

    Returns:
        无。

    Raises:
        无。
    """

    agent_input: AgentInput
    resume_snapshot: PreparedAgentTurnSnapshot | None


@dataclass(frozen=True)
class PreparedSceneState:
    """Host 内部静态 scene 状态。"""

    scene_name: str
    scene_definition: SceneDefinition
    resolved_options: ResolvedExecutionOptions
    model_config: ModelConfig
    prompt_asset_store: PromptAssetStoreProtocol
    tool_registry: PromptToolExecutorProtocol
    agent_create_args: AgentCreateArgs
    conversation_memory_settings: ConversationMemorySettings
    tool_trace_recorder_factory: ToolTraceRecorderFactory | None = None
    trace_identity: AgentTraceIdentity | None = None


@dataclass
class ConversationSessionState:
    """多轮 Host Session 的会话状态快照。"""

    session_id: str
    scene_name: str
    transcript: ConversationTranscript
    conversation_store: ConversationStore
    memory_manager: DefaultConversationMemoryManager
    prepared_scene: PreparedSceneState
    user_message: str

    def persist_turn(
        self,
        *,
        final_content: str,
        degraded: bool,
        tool_uses: tuple[object, ...],
        warnings: tuple[str, ...],
        errors: tuple[str, ...],
    ) -> None:
        """把当前轮结果写回 transcript 并调度后台压缩。"""

        turn_record = ConversationTurnRecord(
            turn_id=f"turn_{uuid.uuid4().hex[:8]}",
            scene_name=self.scene_name,
            user_text=self.user_message,
            assistant_final=final_content,
            assistant_degraded=degraded,
            tool_uses=_coerce_tool_use_summaries(tool_uses),
            warnings=warnings,
            errors=errors,
        )
        next_transcript = self.transcript.append_turn(turn_record)
        persisted = self.conversation_store.save(next_transcript, expected_revision=self.transcript.revision)
        self.memory_manager.schedule_compaction(
            session_id=self.session_id,
            prepared_scene=self.prepared_scene,
            transcript=persisted,
        )


def _normalize_toolset_names(items: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """规整并去重 toolset 名称序列。"""

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in items:
        item = str(raw_item or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _resolve_enabled_toolsets(
    *,
    scene_definition: SceneDefinition,
    selected_toolsets: tuple[str, ...],
    installed_toolsets: tuple[str, ...],
) -> tuple[str, ...]:
    """解析当前 scene 最终启用的 toolset 名单。

    Args:
        scene_definition: 当前 scene 定义。
        selected_toolsets: Service 显式传入的 toolset 收窄集合。
        installed_toolsets: 当前运行时已安装的 toolset 集合。

    Returns:
        最终启用的 toolset 名单。

    Raises:
        无。
    """

    normalized_selected = _normalize_toolset_names(list(selected_toolsets))
    normalized_installed = _normalize_toolset_names(list(installed_toolsets))
    policy = scene_definition.tool_selection_policy
    if policy.mode == ToolSelectionMode.NONE:
        return ()
    if policy.mode == ToolSelectionMode.SELECT:
        manifest_toolsets = _normalize_toolset_names(list(policy.tool_tags_any))
        if not normalized_selected:
            return manifest_toolsets
        selected_set = frozenset(normalized_selected)
        return tuple(item for item in manifest_toolsets if item in selected_set)
    if normalized_selected:
        return normalized_selected
    return normalized_installed


def _load_toolset_registrar(import_path: str) -> ToolsetRegistrarProtocol:
    """按导入路径加载 toolset adapter。"""

    normalized_path = str(import_path or "").strip()
    if not normalized_path or "." not in normalized_path:
        raise ValueError(f"非法 toolset registrar 导入路径: {import_path!r}")
    module_name, attr_name = normalized_path.rsplit(".", 1)
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"无法导入 toolset registrar 模块: {module_name}") from exc
    try:
        registrar = getattr(module, attr_name)
    except AttributeError as exc:
        raise AttributeError(f"toolset registrar 不存在: {normalized_path}") from exc
    if not callable(registrar):
        raise TypeError(f"toolset registrar 必须可调用: {normalized_path}")
    return cast(ToolsetRegistrarProtocol, registrar)


@dataclass
class _ConversationRuntimeAdapter:
    """供 conversation memory 复用的 Host 内部适配器。"""

    scene_preparer: "DefaultScenePreparer"

    def prepare_compaction_scene(
        self,
        scene_name: str,
        execution_options: ExecutionOptions | None = None,
        web_tools_config: WebToolsConfig | None = None,
    ) -> ConversationCompactionSceneProtocol:
        """构造 conversation compaction 所需的静态 scene 状态。"""

        return self.scene_preparer._prepare_host_owned_scene_state(
            scene_name=scene_name,
            execution_options=execution_options,
            web_tools_config=web_tools_config,
        )

    def prepare_compaction_agent(
        self,
        prepared_scene: ConversationCompactionSceneProtocol,
        request: ConversationCompactionRequest,
    ) -> ConversationCompactionAgentHandle:
        """为 conversation compaction 构造最小 Agent 句柄。"""

        session_id = request.session_id.strip()
        system_prompt = self.scene_preparer._compose_system_prompt(
            prepared_scene=prepared_scene,
            prompt_contributions={},
        )
        agent = build_async_agent(
            agent_create_args=prepared_scene.agent_create_args,
            tool_executor=prepared_scene.tool_registry,
            tool_trace_recorder_factory=prepared_scene.tool_trace_recorder_factory,
            trace_identity=(
                AgentTraceIdentity(
                    agent_name=prepared_scene.trace_identity.agent_name,
                    agent_kind=prepared_scene.trace_identity.agent_kind,
                    scene_name=prepared_scene.trace_identity.scene_name,
                    model_name=prepared_scene.trace_identity.model_name,
                    session_id=session_id or prepared_scene.trace_identity.session_id,
                )
                if prepared_scene.trace_identity is not None
                else None
            ),
        )
        return ConversationCompactionAgentHandle(agent=agent, system_prompt=system_prompt)


@dataclass
class DefaultScenePreparer(ScenePreparationProtocol):
    """默认 Host scene preparation。"""

    workspace: WorkspaceResourcesProtocol
    model_catalog: ModelCatalogProtocol
    default_execution_options: ResolvedExecutionOptions
    tool_registry_factory: ToolRegistryFactory
    conversation_store: ConversationStore | None = None

    def __post_init__(self) -> None:
        """初始化辅助依赖。"""

        self._prompt_composer = PromptComposer()
        self._trace_provider = TraceRecorderFactoryProvider()
        self._conversation_store_impl = self.conversation_store or FileConversationStore(
            build_conversation_store_dir(self.workspace.workspace_dir)
        )
        self._conversation_runtime = _ConversationRuntimeAdapter(self)
        self._memory_manager = DefaultConversationMemoryManager(
            self._conversation_runtime,
            conversation_store=self._conversation_store_impl,
        )

    async def prepare(
        self,
        execution_contract: ExecutionContract,
        run_context: HostedRunContext,
    ) -> PreparedAgentExecution:
        """将 ``ExecutionContract`` 收敛为可执行输入与可恢复快照。"""

        user_message = str(execution_contract.message_inputs.user_message or "").strip()
        if not user_message:
            raise ValueError("ExecutionContract.message_inputs.user_message 不能为空")

        scene_definition = load_scene_definition(
            self.workspace.prompt_asset_store,
            execution_contract.scene_name,
        )
        _validate_resumable_host_policy(
            execution_contract=execution_contract,
            scene_definition=scene_definition,
        )
        prepared_scene = self._prepare_scene_state_from_contract(
            execution_contract=execution_contract,
            scene_definition=scene_definition,
        )
        system_prompt = self._compose_system_prompt(
            prepared_scene=prepared_scene,
            prompt_contributions=execution_contract.preparation_spec.prompt_contributions,
        )
        session_key = str(execution_contract.host_policy.session_key or "").strip()
        session_state: ConversationSessionState | None = None
        if prepared_scene.scene_definition.conversation.enabled and session_key:
            await self._memory_manager.cancel_pending_compaction(session_key)
            existing_transcript = self._conversation_store_impl.load(session_key)
            transcript = existing_transcript or ConversationTranscript.create_empty(session_key)
            transcript = await self._memory_manager.prepare_transcript(
                session_id=session_key,
                prepared_scene=prepared_scene,
                transcript=transcript,
            )
            messages = self._memory_manager.build_messages(
                prepared_scene=prepared_scene,
                transcript=transcript,
                system_prompt=system_prompt,
                user_text=user_message,
            )
            session_state = ConversationSessionState(
                session_id=session_key,
                scene_name=execution_contract.scene_name,
                transcript=transcript,
                conversation_store=self._conversation_store_impl,
                memory_manager=self._memory_manager,
                prepared_scene=prepared_scene,
                user_message=user_message,
            )
        else:
            messages = _build_single_turn_messages(system_prompt=system_prompt, user_message=user_message)

        agent_input = AgentInput(
            system_prompt=system_prompt,
            messages=messages,
            tools=prepared_scene.tool_registry,
            agent_create_args=prepared_scene.agent_create_args,
            session_state=session_state,
            runtime_limits=AgentRuntimeLimits(timeout_ms=execution_contract.host_policy.timeout_ms),
            cancellation_handle=run_context.cancellation_token,
            tool_trace_recorder_factory=prepared_scene.tool_trace_recorder_factory,
            trace_identity=(
                AgentTraceIdentity(
                    agent_name=prepared_scene.trace_identity.agent_name,
                    agent_kind=prepared_scene.trace_identity.agent_kind,
                    scene_name=prepared_scene.trace_identity.scene_name,
                    model_name=prepared_scene.trace_identity.model_name,
                    session_id=session_key or prepared_scene.trace_identity.session_id,
                )
                if prepared_scene.trace_identity is not None
                else None
            ),
        )
        return PreparedAgentExecution(
            agent_input=agent_input,
            resume_snapshot=_build_prepared_turn_snapshot(
                execution_contract=execution_contract,
                prepared_scene=prepared_scene,
                system_prompt=system_prompt,
                messages=messages,
                session_state=session_state,
            )
            if execution_contract.host_policy.resumable
            else None,
        )

    async def restore_prepared_execution(
        self,
        prepared_turn: PreparedAgentTurnSnapshot,
        run_context: HostedRunContext,
    ) -> AgentInput:
        """从 prepared turn 快照重建 ``AgentInput``。"""

        prepared_scene = self._restore_prepared_scene_state(prepared_turn)
        session_state: ConversationSessionState | None = None
        session_snapshot = prepared_turn.conversation_session
        if session_snapshot is not None:
            session_state = ConversationSessionState(
                session_id=session_snapshot.session_id,
                scene_name=prepared_turn.scene_name,
                transcript=session_snapshot.transcript,
                conversation_store=self._conversation_store_impl,
                memory_manager=self._memory_manager,
                prepared_scene=prepared_scene,
                user_message=session_snapshot.user_message,
            )
        return AgentInput(
            system_prompt=prepared_turn.system_prompt,
            messages=list(prepared_turn.messages),
            tools=prepared_scene.tool_registry,
            agent_create_args=prepared_turn.agent_create_args,
            session_state=session_state,
            runtime_limits=AgentRuntimeLimits(timeout_ms=prepared_turn.timeout_ms),
            cancellation_handle=run_context.cancellation_token,
            tool_trace_recorder_factory=prepared_scene.tool_trace_recorder_factory,
            trace_identity=prepared_turn.trace_identity,
        )

    def _prepare_scene_state_from_contract(
        self,
        *,
        execution_contract: ExecutionContract,
        scene_definition: SceneDefinition,
    ) -> PreparedSceneState:
        """根据 Service 提供的 contract 构造静态 scene 状态。"""

        accepted_execution_spec = execution_contract.accepted_execution_spec
        model_config = self.model_catalog.load_model(accepted_execution_spec.model.model_name)
        resolved_options = _build_resolved_execution_options_from_contract(
            base_options=self.default_execution_options,
            execution_contract=execution_contract,
        )
        agent_create_args = build_agent_create_args(
            resolved_execution_options=resolved_options,
            model_config=model_config,
        )
        tool_registry = self._build_tool_registry(
            scene_definition=scene_definition,
            selected_toolsets=execution_contract.preparation_spec.selected_toolsets,
            execution_permissions=execution_contract.preparation_spec.execution_permissions,
            resolved_options=resolved_options,
            tool_timeout_seconds=_extract_tool_timeout_seconds(agent_create_args),
        )
        trace_settings = (
            accepted_execution_spec.infrastructure.trace_settings
            or self.default_execution_options.trace_settings
        )
        tool_trace_recorder_factory = self._trace_provider.get_or_create(trace_settings)
        return PreparedSceneState(
            scene_name=execution_contract.scene_name,
            scene_definition=scene_definition,
            resolved_options=resolved_options,
            model_config=model_config,
            prompt_asset_store=self.workspace.prompt_asset_store,
            tool_registry=tool_registry,
            agent_create_args=agent_create_args,
            conversation_memory_settings=resolved_options.conversation_memory_settings,
            tool_trace_recorder_factory=tool_trace_recorder_factory,
            trace_identity=AgentTraceIdentity(
                agent_name=f"{execution_contract.scene_name}_agent",
                agent_kind="scene_agent",
                scene_name=execution_contract.scene_name,
                model_name=agent_create_args.model_name,
            ),
        )

    def _prepare_host_owned_scene_state(
        self,
        *,
        scene_name: str,
        execution_options: ExecutionOptions | None,
        web_tools_config: WebToolsConfig | None = None,
    ) -> PreparedSceneState:
        """为 Host 自己发起的 Agent 子执行构造静态 scene 状态。"""

        scene_definition = load_scene_definition(self.workspace.prompt_asset_store, scene_name)
        resolved_options = _resolve_scene_execution_options(
            base_execution_options=self.default_execution_options,
            workspace_dir=self.workspace.workspace_dir,
            scene_definition=scene_definition,
            execution_options=execution_options,
        )
        model_name = str(resolved_options.model_name or "").strip()
        model_config = self.model_catalog.load_model(model_name)
        resolved_temperature = resolve_scene_temperature(
            resolved_temperature=resolved_options.temperature,
            model_config=model_config,
            temperature_profile=scene_definition.model.temperature_profile,
            scene_name=scene_definition.name,
            model_name=resolved_options.model_name,
        )
        conversation_memory_settings = resolve_conversation_memory_settings(
            conversation_memory_config=resolved_options.conversation_memory_config,
            model_config=model_config,
        )
        effective_web_tools_config = (
            web_tools_config
            or resolve_web_tools_config_from_toolset_configs(resolved_options.toolset_configs)
            or WebToolsConfig()
        )
        resolved_options = replace(
            resolved_options,
            temperature=resolved_temperature,
            conversation_memory_settings=conversation_memory_settings,
            toolset_configs=_merge_toolset_configs(
                resolved_options.toolset_configs,
                (
                    build_toolset_config_snapshot("web", effective_web_tools_config),
                )
                if build_toolset_config_snapshot("web", effective_web_tools_config) is not None
                else (),
            ),
        )
        agent_create_args = build_agent_create_args(
            resolved_execution_options=resolved_options,
            model_config=model_config,
        )
        tool_registry = self._build_tool_registry(
            scene_definition=scene_definition,
            selected_toolsets=(),
            execution_permissions=ExecutionPermissions(
                web=ExecutionWebPermissions(
                    allow_private_network_url=bool(effective_web_tools_config.allow_private_network_url),
                ),
                doc=ExecutionDocPermissions(),
            ),
            resolved_options=resolved_options,
            tool_timeout_seconds=_extract_tool_timeout_seconds(agent_create_args),
        )
        tool_trace_recorder_factory = self._trace_provider.get_or_create(resolved_options.trace_settings)
        return PreparedSceneState(
            scene_name=scene_name,
            scene_definition=scene_definition,
            resolved_options=resolved_options,
            model_config=model_config,
            prompt_asset_store=self.workspace.prompt_asset_store,
            tool_registry=tool_registry,
            agent_create_args=agent_create_args,
            conversation_memory_settings=conversation_memory_settings,
            tool_trace_recorder_factory=tool_trace_recorder_factory,
            trace_identity=AgentTraceIdentity(
                agent_name=f"{scene_name}_agent",
                agent_kind="scene_agent",
                scene_name=scene_name,
                model_name=agent_create_args.model_name,
            ),
        )

    def _build_tool_registry(
        self,
        *,
        scene_definition: SceneDefinition,
        selected_toolsets: tuple[str, ...],
        execution_permissions: ExecutionPermissions,
        resolved_options: ResolvedExecutionOptions,
        tool_timeout_seconds: float | None,
    ) -> PromptToolExecutorProtocol:
        """构建当前 scene 的工具执行与快照视图。"""

        toolset_registrars = self.workspace.config_loader.load_toolset_registrars()
        enabled_toolsets = _resolve_enabled_toolsets(
            scene_definition=scene_definition,
            selected_toolsets=selected_toolsets,
            installed_toolsets=tuple(toolset_registrars.keys()),
        )
        registry = self.tool_registry_factory()
        for toolset_name in enabled_toolsets:
            registrar_path = toolset_registrars.get(toolset_name)
            if registrar_path is None:
                raise ValueError(
                    "当前 scene 最终启用了缺失 registrar 的 toolset: "
                    f"scene={scene_definition.name} toolset={toolset_name} "
                    f"selected_toolsets={selected_toolsets}"
                )
            registrar = _load_toolset_registrar(registrar_path)
            registrar(
                ToolsetRegistrationContext(
                    toolset_name=toolset_name,
                    registry=registry,
                    workspace=self.workspace,
                    toolset_config=find_toolset_config(resolved_options.toolset_configs, toolset_name),
                    execution_permissions=execution_permissions,
                    tool_timeout_seconds=tool_timeout_seconds,
                )
            )
        return registry

    def _restore_prepared_scene_state(
        self,
        prepared_turn: PreparedAgentTurnSnapshot,
    ) -> PreparedSceneState:
        """从 prepared turn 快照恢复静态 scene 状态。"""

        scene_definition = load_scene_definition(self.workspace.prompt_asset_store, prepared_turn.scene_name)
        model_config = self.model_catalog.load_model(prepared_turn.agent_create_args.model_name)
        runner_running_config = build_runner_running_config_from_snapshot(
            prepared_turn.agent_create_args.runner_running_config,
            base_config=self.default_execution_options.runner_running_config,
        )
        agent_running_config = build_agent_running_config_from_snapshot(
            prepared_turn.agent_create_args.agent_running_config,
        )
        trace_settings = prepared_turn.trace_settings or self.default_execution_options.trace_settings
        merged_toolset_configs = _merge_toolset_configs(
            self.default_execution_options.toolset_configs,
            prepared_turn.toolset_configs,
        )
        resolved_options = replace(
            self.default_execution_options,
            model_name=prepared_turn.agent_create_args.model_name,
            temperature=(
                prepared_turn.agent_create_args.temperature
                if prepared_turn.agent_create_args.temperature is not None
                else self.default_execution_options.temperature
            ),
            runner_running_config=runner_running_config,
            agent_running_config=agent_running_config,
            toolset_configs=cast(tuple[Any, ...], merged_toolset_configs),
            trace_settings=trace_settings,
            conversation_memory_settings=prepared_turn.conversation_memory_settings,
        )
        tool_registry = self._build_tool_registry(
            scene_definition=scene_definition,
            selected_toolsets=prepared_turn.selected_toolsets,
            execution_permissions=prepared_turn.execution_permissions,
            resolved_options=resolved_options,
            tool_timeout_seconds=_extract_tool_timeout_seconds(prepared_turn.agent_create_args),
        )
        return PreparedSceneState(
            scene_name=prepared_turn.scene_name,
            scene_definition=scene_definition,
            resolved_options=resolved_options,
            model_config=model_config,
            prompt_asset_store=self.workspace.prompt_asset_store,
            tool_registry=tool_registry,
            agent_create_args=prepared_turn.agent_create_args,
            conversation_memory_settings=prepared_turn.conversation_memory_settings,
            tool_trace_recorder_factory=self._trace_provider.get_or_create(trace_settings),
            trace_identity=prepared_turn.trace_identity,
        )

    def _compose_system_prompt(
        self,
        *,
        prepared_scene: PreparedSceneState | ConversationCompactionSceneProtocol,
        prompt_contributions: dict[str, str],
    ) -> str:
        """基于 scene 定义与 Prompt Contributions 组装 system prompt。"""

        prompt_contribution_selection = select_prompt_contributions(
            prompt_contributions=prompt_contributions,
            context_slots=prepared_scene.scene_definition.context_slots,
        )
        if prompt_contribution_selection.ignored_slots:
            Log.warn(
                "检测到未在 scene manifest 中声明的 Prompt Contributions slot，已忽略: "
                f"scene={prepared_scene.scene_name}, slots={list(prompt_contribution_selection.ignored_slots)}",
                module=MODULE,
            )
        tool_snapshot = build_prompt_tool_snapshot(
            prepared_scene.tool_registry,
            supports_tool_calling=bool(prepared_scene.model_config.get("supports_tool_calling", False)),
        )
        context: dict[str, str] = {}
        if tool_snapshot.allowed_paths:
            context["directories"] = ", ".join(tool_snapshot.allowed_paths)
        plan = build_prompt_assembly_plan(
            asset_store=prepared_scene.prompt_asset_store,
            scene_definition=prepared_scene.scene_definition,
        )
        composed = self._prompt_composer.compose(
            plan=plan,
            context=PromptComposeContext(values=context),
            tool_snapshot=tool_snapshot,
            prompt_contributions=prompt_contribution_selection.selected_contributions,
            context_slots=prepared_scene.scene_definition.context_slots,
        )
        return composed.system_message


def _resolve_scene_execution_options(
    *,
    base_execution_options: ResolvedExecutionOptions,
    workspace_dir: Path,
    scene_definition: SceneDefinition,
    execution_options: ExecutionOptions | None,
) -> ResolvedExecutionOptions:
    """为 Host 自己发起的子执行解析 scene 级执行选项。"""

    return resolve_scene_execution_options(
        base_execution_options=base_execution_options,
        workspace_dir=workspace_dir,
        execution_options=execution_options,
        default_model_name=scene_definition.model.default_name,
        allowed_model_names=scene_definition.model.allowed_names,
        scene_agent_max_iterations=scene_definition.runtime.agent.max_iterations,
        scene_agent_max_consecutive_failed_tool_batches=(
            scene_definition.runtime.agent.max_consecutive_failed_tool_batches
        ),
        scene_runner_tool_timeout_seconds=scene_definition.runtime.runner.tool_timeout_seconds,
        scene_name=scene_definition.name,
    )


def _build_resolved_execution_options_from_contract(
    *,
    base_options: ResolvedExecutionOptions,
    execution_contract: ExecutionContract,
) -> ResolvedExecutionOptions:
    """根据 Service 提供的 contract 还原 Host 可消费的 resolved options。"""

    accepted_execution_spec = execution_contract.accepted_execution_spec
    runner_running_config = build_runner_running_config_from_snapshot(
        accepted_execution_spec.runtime.runner_running_config,
        base_config=base_options.runner_running_config,
    )
    agent_running_config = build_agent_running_config_from_snapshot(
        accepted_execution_spec.runtime.agent_running_config,
    )
    merged_toolset_configs = _merge_toolset_configs(
        base_options.toolset_configs,
        accepted_execution_spec.tools.toolset_configs,
    )
    return replace(
        base_options,
        model_name=accepted_execution_spec.model.model_name,
        temperature=accepted_execution_spec.model.temperature,
        runner_running_config=runner_running_config,
        agent_running_config=agent_running_config,
        toolset_configs=cast(tuple[Any, ...], merged_toolset_configs),
        trace_settings=(
            accepted_execution_spec.infrastructure.trace_settings or base_options.trace_settings
        ),
        conversation_memory_settings=(
            accepted_execution_spec.infrastructure.conversation_memory_settings
            or base_options.conversation_memory_settings
        ),
    )
def _build_single_turn_messages(
    *,
    system_prompt: str,
    user_message: str,
) -> list[AgentMessage]:
    """构建单轮送模消息。"""

    messages: list[AgentMessage] = []
    normalized_system_prompt = str(system_prompt or "").strip()
    if normalized_system_prompt:
        messages.append({"role": "system", "content": normalized_system_prompt})
    messages.append({"role": "user", "content": user_message})
    return messages


def _coerce_tool_use_summaries(
    tool_uses: tuple[object, ...],
) -> tuple[ConversationToolUseSummary, ...]:
    """把协议层 tool_uses 收窄为 transcript 可持久化的工具摘要。

    Args:
        tool_uses: 协议层传入的工具摘要元组。

    Returns:
        经过校验的工具摘要元组。

    Raises:
        TypeError: 当存在非 ``ConversationToolUseSummary`` 条目时抛出。
    """

    normalized: list[ConversationToolUseSummary] = []
    for tool_use in tool_uses:
        if not isinstance(tool_use, ConversationToolUseSummary):
            raise TypeError("session_state.persist_turn 仅支持 ConversationToolUseSummary")
        normalized.append(tool_use)
    return tuple(normalized)


def _extract_tool_timeout_seconds(agent_create_args: AgentCreateArgs) -> float | None:
    """从 ``AgentCreateArgs`` 提取工具超时预算秒数。"""

    raw_timeout = dict(agent_create_args.runner_running_config).get("tool_timeout_seconds")
    if raw_timeout is None or isinstance(raw_timeout, bool) or not isinstance(raw_timeout, int | float):
        return None
    return float(raw_timeout)


def _build_prepared_turn_snapshot(
    *,
    execution_contract: ExecutionContract,
    prepared_scene: PreparedSceneState,
    system_prompt: str,
    messages: list[AgentMessage],
    session_state: ConversationSessionState | None,
) -> PreparedAgentTurnSnapshot:
    """根据已完成的 scene preparation 构造可恢复快照。"""

    conversation_session: PreparedConversationSessionSnapshot | None = None
    if session_state is not None:
        conversation_session = PreparedConversationSessionSnapshot(
            session_id=session_state.session_id,
            user_message=session_state.user_message,
            transcript=session_state.transcript,
        )
    return PreparedAgentTurnSnapshot(
        service_name=execution_contract.service_name,
        scene_name=execution_contract.scene_name,
        metadata=execution_contract.metadata,
        concurrency_lane=execution_contract.host_policy.concurrency_lane,
        timeout_ms=execution_contract.host_policy.timeout_ms,
        resumable=bool(execution_contract.host_policy.resumable),
        system_prompt=system_prompt,
        messages=list(messages),
        agent_create_args=prepared_scene.agent_create_args,
        selected_toolsets=execution_contract.preparation_spec.selected_toolsets,
        execution_permissions=execution_contract.preparation_spec.execution_permissions,
        toolset_configs=execution_contract.accepted_execution_spec.tools.toolset_configs,
        trace_settings=(
            execution_contract.accepted_execution_spec.infrastructure.trace_settings
            or prepared_scene.resolved_options.trace_settings
        ),
        conversation_memory_settings=prepared_scene.conversation_memory_settings,
        trace_identity=(
            AgentTraceIdentity(
                agent_name=prepared_scene.trace_identity.agent_name,
                agent_kind=prepared_scene.trace_identity.agent_kind,
                scene_name=prepared_scene.trace_identity.scene_name,
                model_name=prepared_scene.trace_identity.model_name,
                session_id=(session_state.session_id if session_state is not None else prepared_scene.trace_identity.session_id),
            )
            if prepared_scene.trace_identity is not None
            else None
        ),
        conversation_session=conversation_session,
    )


__all__ = [
    "ConversationSessionState",
    "DefaultScenePreparer",
    "PreparedAgentExecution",
    "PreparedSceneState",
    "ScenePreparationProtocol",
]
