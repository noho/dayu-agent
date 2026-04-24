"""Service 侧 scene 执行接受模块。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedInfrastructureSpec,
    AcceptedModelSpec,
    AcceptedRuntimeSpec,
    AcceptedToolConfigSpec,
)
from dayu.contracts.infrastructure import ModelCatalogProtocol
from dayu.contracts.model_config import ModelConfig
from dayu.execution.runtime_config import OpenAIRunnerRuntimeConfig
from dayu.execution.runtime_config import build_agent_running_config_snapshot, build_runner_running_config_snapshot
from dayu.execution.options import (
    ExecutionOptions,
    ResolvedExecutionOptions,
    resolve_scene_temperature,
    resolve_scene_execution_options,
)
from dayu.prompting.scene_definition import SceneDefinition
from dayu.services.conversation_policy_reader import ConversationPolicyReader
from dayu.services.contracts import SceneModelConfig
from dayu.services.scene_definition_reader import SceneDefinitionReader
from dayu.log import Log

MODULE = "SERVICE.SCENE_ACCEPTANCE"


def _build_scene_acceptance_debug_message(
    *,
    accepted_scene: "AcceptedSceneExecution",
) -> str:
    """构造 scene 接受结果调试日志。

    Args:
        accepted_scene: 已接受的 scene 执行结果。

    Returns:
        统一格式的调试日志文本。

    Raises:
        无。
    """

    runner_running_config = accepted_scene.resolved_execution_options.runner_running_config
    tool_timeout_seconds: float | None = None
    if isinstance(runner_running_config, OpenAIRunnerRuntimeConfig):
        tool_timeout_seconds = runner_running_config.tool_timeout_seconds
    return (
        "scene 接受结果: "
        f"scene_name={accepted_scene.scene_name}, "
        f"model_name={accepted_scene.accepted_execution_spec.model.model_name}, "
        f"temperature={accepted_scene.resolved_temperature}, "
        f"max_iterations={accepted_scene.resolved_execution_options.agent_running_config.max_iterations}, "
        f"tool_timeout_seconds={tool_timeout_seconds}, "
        f"resumable={accepted_scene.default_resumable}"
    )


@dataclass(frozen=True)
class AcceptedSceneExecution:
    """Service 侧单个 scene 的接受结果。"""

    scene_name: str
    scene_definition: SceneDefinition
    resolved_execution_options: ResolvedExecutionOptions
    model_config: ModelConfig
    resolved_temperature: float
    accepted_execution_spec: AcceptedExecutionSpec

    @property
    def scene_model(self) -> SceneModelConfig:
        """返回用于展示的模型摘要。"""

        return SceneModelConfig(
            name=self.accepted_execution_spec.model.model_name,
            temperature=self.resolved_temperature,
        )

    @property
    def default_resumable(self) -> bool:
        """返回该 scene 的默认 resumable 策略。"""

        return self.scene_definition.conversation.enabled


@dataclass(frozen=True)
class SceneExecutionAcceptancePreparer:
    """把 scene 规则与显式参数收敛为已接受执行规格。"""

    workspace_dir: Path
    base_execution_options: ResolvedExecutionOptions
    model_catalog: ModelCatalogProtocol
    scene_definition_reader: SceneDefinitionReader
    conversation_policy_reader: ConversationPolicyReader

    def read_scene_definition(self, scene_name: str) -> SceneDefinition:
        """读取 scene 定义。"""

        return self.scene_definition_reader.read(scene_name)

    def resolve_execution_options(
        self,
        scene_name: str,
        execution_options: ExecutionOptions | None = None,
    ) -> ResolvedExecutionOptions:
        """按 scene 真源规则解析已接受执行选项。"""

        scene_definition = self.read_scene_definition(scene_name)
        return resolve_scene_execution_options(
            base_execution_options=self.base_execution_options,
            workspace_dir=self.workspace_dir,
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

    def prepare(
        self,
        scene_name: str,
        execution_options: ExecutionOptions | None = None,
    ) -> AcceptedSceneExecution:
        """准备单个 scene 的接受执行规格。"""

        scene_definition = self.read_scene_definition(scene_name)
        resolved_execution_options = self.resolve_execution_options(scene_name, execution_options)
        model_name = str(resolved_execution_options.model_name or "").strip()
        if not model_name:
            raise ValueError("当前执行缺少 model_name")
        model_config = self.model_catalog.load_model(model_name)
        resolved_temperature = resolve_scene_temperature(
            resolved_temperature=resolved_execution_options.temperature,
            model_config=model_config,
            temperature_profile=scene_definition.model.temperature_profile,
            scene_name=scene_definition.name,
            model_name=resolved_execution_options.model_name,
        )
        conversation_memory_settings = self.conversation_policy_reader.resolve(
            resolved_execution_options=resolved_execution_options,
            model_config=model_config,
        )
        resolved_execution_options = replace(
            resolved_execution_options,
            conversation_memory_settings=conversation_memory_settings,
        )
        accepted_scene = AcceptedSceneExecution(
            scene_name=scene_name,
            scene_definition=scene_definition,
            resolved_execution_options=resolved_execution_options,
            model_config=model_config,
            resolved_temperature=resolved_temperature,
            accepted_execution_spec=_build_accepted_execution_spec(
                resolved_execution_options=resolved_execution_options,
                model_config=model_config,
                resolved_temperature=resolved_temperature,
            ),
        )
        Log.debug(
            _build_scene_acceptance_debug_message(accepted_scene=accepted_scene),
            module=MODULE,
        )
        return accepted_scene

    def resolve_scene_model(
        self,
        scene_name: str,
        execution_options: ExecutionOptions | None = None,
    ) -> SceneModelConfig:
        """解析用于展示的 scene 模型摘要。"""

        return self.prepare(scene_name, execution_options).scene_model


def _build_accepted_execution_spec(
    *,
    resolved_execution_options: ResolvedExecutionOptions,
    model_config: ModelConfig,
    resolved_temperature: float,
) -> AcceptedExecutionSpec:
    """根据已接受执行选项构造 AcceptedExecutionSpec。"""

    accepted_agent_running_config = replace(
        resolved_execution_options.agent_running_config,
        max_context_tokens=int(model_config.get("max_context_tokens") or 0),
    )
    return AcceptedExecutionSpec(
        model=AcceptedModelSpec(
            model_name=resolved_execution_options.model_name,
            temperature=resolved_temperature,
        ),
        runtime=AcceptedRuntimeSpec(
            runner_running_config=build_runner_running_config_snapshot(
                resolved_execution_options.runner_running_config
            ),
            agent_running_config=build_agent_running_config_snapshot(accepted_agent_running_config),
        ),
        tools=AcceptedToolConfigSpec(
            toolset_configs=resolved_execution_options.toolset_configs,
        ),
        infrastructure=AcceptedInfrastructureSpec(
            trace_settings=resolved_execution_options.trace_settings,
            conversation_memory_settings=resolved_execution_options.conversation_memory_settings,
        ),
    )


__all__ = [
    "AcceptedSceneExecution",
    "SceneExecutionAcceptancePreparer",
]
