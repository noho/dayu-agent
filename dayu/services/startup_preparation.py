"""Service 层对启动期暴露的 preparation API。

本模块是 Service 层的 public surface，供 UI 启动期装配调用。
它负责把稳定输入收敛成 Service 请求期需要的公开依赖，
但不把内部 reader / preparer 的实现细节泄漏给 `startup/` 或 UI。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dayu.contracts.infrastructure import ModelCatalogProtocol, PromptAssetStoreProtocol
from dayu.execution.options import ExecutionOptions, ResolvedExecutionOptions
from dayu.fins.service_runtime import DefaultFinsRuntime, FinsRuntimeProtocol
from dayu.host import Host, resolve_host_config
from dayu.services.concurrency_lanes import SERVICE_DEFAULT_LANE_CONFIG
from dayu.services.conversation_policy_reader import ConversationPolicyReader
from dayu.services.host_admin_service import HostAdminService
from dayu.services.scene_definition_reader import SceneDefinitionReader
from dayu.services.scene_execution_acceptance import SceneExecutionAcceptancePreparer
from dayu.services.startup_recovery import recover_host_startup_state
from dayu.execution.options import build_base_execution_options, merge_execution_options
from dayu.startup.config_file_resolver import ConfigFileResolver
from dayu.startup.config_loader import ConfigLoader
from dayu.startup.model_catalog import ConfigLoaderModelCatalog
from dayu.startup.paths import StartupPaths, resolve_startup_paths
from dayu.startup.prompt_assets import FilePromptAssetStore
from dayu.startup.workspace import WorkspaceResources


@dataclass(frozen=True)
class PreparedHostAdminDependencies:
    """Host 管理子命令使用的轻量依赖集合。

    Args:
        paths: 启动期解析出的路径信息（供 CLI 展示）。
        host_admin_service: 已装配的宿主管理服务。
    """

    paths: StartupPaths
    host_admin_service: HostAdminService


@dataclass(frozen=True)
class PreparedHostRuntimeDependencies:
    """共享 Host 运行时依赖集合。

    Args:
        workspace: 工作区稳定资源。
        default_execution_options: 启动期解析后的默认执行选项。
        scene_execution_acceptance_preparer: scene 执行接受准备器。
        host: Host 实例。
        fins_runtime: 财报领域运行时。
    """

    workspace: WorkspaceResources
    default_execution_options: ResolvedExecutionOptions
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer
    host: Host
    fins_runtime: FinsRuntimeProtocol


def prepare_scene_execution_acceptance_preparer(
    *,
    workspace_root: Path,
    default_execution_options: ResolvedExecutionOptions,
    model_catalog: ModelCatalogProtocol,
    prompt_asset_store: PromptAssetStoreProtocol,
) -> SceneExecutionAcceptancePreparer:
    """准备 Service 侧 scene 执行接受准备器。

    Args:
        workspace_root: 当前工作区根目录。
        default_execution_options: 启动期已解析的默认执行选项。
        model_catalog: 启动期模型目录对象。
        prompt_asset_store: prompt 资产仓储对象。

    Returns:
        已完成内部 reader 装配的 `SceneExecutionAcceptancePreparer`。

    Raises:
        无。
    """

    scene_definition_reader = SceneDefinitionReader(prompt_asset_store)
    conversation_policy_reader = ConversationPolicyReader()
    return SceneExecutionAcceptancePreparer(
        workspace_dir=workspace_root,
        base_execution_options=default_execution_options,
        model_catalog=model_catalog,
        scene_definition_reader=scene_definition_reader,
        conversation_policy_reader=conversation_policy_reader,
    )


def prepare_host_runtime_dependencies(
    *,
    workspace_root: Path,
    config_root: Path | None,
    execution_options: ExecutionOptions | None,
    runtime_label: str,
    log_module: str,
) -> PreparedHostRuntimeDependencies:
    """准备 CLI / WeChat 共用的 Host 运行时稳定依赖。

    Args:
        workspace_root: 工作区根目录。
        config_root: 可选配置根目录。
        execution_options: 请求级执行选项。
        runtime_label: startup recovery 的运行时标签。
        log_module: recovery 日志模块名。

    Returns:
        已完成 Host、scene preparation 与 fins runtime 装配的共享依赖集合。

    Raises:
        无。
    """

    paths = resolve_startup_paths(
        workspace_root=workspace_root,
        config_root=config_root,
    )
    resolver = ConfigFileResolver(paths.config_root)
    config_loader = ConfigLoader(resolver)
    prompt_asset_store = FilePromptAssetStore(resolver)
    workspace = WorkspaceResources(
        workspace_dir=paths.workspace_root,
        config_root=paths.config_root,
        output_dir=paths.output_dir,
        config_loader=config_loader,
        prompt_asset_store=prompt_asset_store,
    )
    model_catalog = ConfigLoaderModelCatalog(config_loader)
    run_config = config_loader.load_run_config()
    base_execution_options = build_base_execution_options(
        workspace_dir=paths.workspace_root,
        run_config=run_config,
    )
    default_execution_options = merge_execution_options(
        base_options=base_execution_options,
        workspace_dir=paths.workspace_root,
        execution_options=execution_options,
    )
    scene_execution_acceptance_preparer = prepare_scene_execution_acceptance_preparer(
        workspace_root=paths.workspace_root,
        default_execution_options=default_execution_options,
        model_catalog=model_catalog,
        prompt_asset_store=prompt_asset_store,
    )
    fins_runtime = DefaultFinsRuntime.create(workspace_root=paths.workspace_root)
    host_config = resolve_host_config(
        workspace_root=paths.workspace_root,
        run_config=run_config,
        service_lane_defaults=dict(SERVICE_DEFAULT_LANE_CONFIG),
        explicit_lane_config=None,
    )
    host = Host(
        workspace=workspace,
        model_catalog=model_catalog,
        default_execution_options=default_execution_options,
        host_store_path=host_config.store_path,
        lane_config=host_config.lane_config,
        pending_turn_resume_max_attempts=host_config.pending_turn_resume_max_attempts,
        pending_turn_retention_hours=host_config.pending_turn_retention_hours,
        cancellation_bridge_poll_interval_seconds=(
            host_config.cancellation_bridge_poll_interval_seconds
        ),
        cancellation_bridge_failure_grace_period_seconds=(
            host_config.cancellation_bridge_failure_grace_period_seconds
        ),
        event_bus=None,
    )
    recover_host_startup_state(
        HostAdminService(host=host),
        runtime_label=runtime_label,
        log_module=log_module,
    )
    return PreparedHostRuntimeDependencies(
        workspace=workspace,
        default_execution_options=default_execution_options,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        host=host,
        fins_runtime=fins_runtime,
    )


def prepare_host_admin_dependencies(
    *,
    workspace_root: Path,
    config_root: Path | None,
) -> PreparedHostAdminDependencies:
    """为 Host 管理子命令准备轻量依赖。

    本入口不构造执行期所需的 scene preparer 与 fins runtime，
    仅装配 Host 并封装出 `HostAdminService`，供 CLI 展示与运维命令使用。

    Args:
        workspace_root: 工作区根目录。
        config_root: 可选配置根目录。

    Returns:
        已装配的宿主管理服务以及配套路径信息。

    Raises:
        TypeError: `run.json` 配置结构非法时抛出。
        ValueError: `run.json` 配置值非法时抛出。
    """

    paths = resolve_startup_paths(
        workspace_root=workspace_root,
        config_root=config_root,
    )
    resolver = ConfigFileResolver(paths.config_root)
    config_loader = ConfigLoader(resolver)
    run_config = config_loader.load_run_config()
    host_config = resolve_host_config(
        workspace_root=paths.workspace_root,
        run_config=run_config,
        service_lane_defaults=dict(SERVICE_DEFAULT_LANE_CONFIG),
        explicit_lane_config=None,
    )
    host = Host(
        host_store_path=host_config.store_path,
        lane_config=host_config.lane_config,
        pending_turn_resume_max_attempts=host_config.pending_turn_resume_max_attempts,
        pending_turn_retention_hours=host_config.pending_turn_retention_hours,
        cancellation_bridge_poll_interval_seconds=(
            host_config.cancellation_bridge_poll_interval_seconds
        ),
        cancellation_bridge_failure_grace_period_seconds=(
            host_config.cancellation_bridge_failure_grace_period_seconds
        ),
        event_bus=None,
    )
    return PreparedHostAdminDependencies(
        paths=paths,
        host_admin_service=HostAdminService(host=host),
    )


__all__ = [
    "PreparedHostAdminDependencies",
    "PreparedHostRuntimeDependencies",
    "prepare_host_admin_dependencies",
    "prepare_host_runtime_dependencies",
    "prepare_scene_execution_acceptance_preparer",
]
