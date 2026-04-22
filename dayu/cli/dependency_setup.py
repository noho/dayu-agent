"""CLI 依赖装配与配置解析模块。

模块职责：
- 定义 CLI 侧数据类型（``WorkspaceConfig``、``RunningConfig``、``ModelName``、``WriteCliConfig``）。
- 解析工作区路径、日志级别、模型名称、写作配置。
- 装配 Host 级依赖并构建 Chat / Prompt / Write / Fins 各 Service 实例。
- 解析 write 场景模型配置。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from dayu.cli.interactive_state import (
    FileInteractiveStateStore,
    InteractiveSessionState,
    build_interactive_key,
    resolve_interactive_session_id as resolve_interactive_state_session_id,
)
from dayu.cli.graceful_shutdown import register_cli_shutdown_hook
from dayu.contracts.infrastructure import ConfigLoaderProtocol, PromptAssetStoreProtocol
from dayu.contracts.session import SessionSource
from dayu.contracts.tool_configs import DocToolLimits, FinsToolLimits
from dayu.execution.cli_execution_options import build_execution_options_from_args
from dayu.execution.options import (
    ExecutionOptions,
    ResolvedExecutionOptions,
    TraceSettings,
    build_base_execution_options,
    merge_execution_options,
    resolve_doc_tool_limits_from_toolset_configs,
    resolve_fins_tool_limits_from_toolset_configs,
    resolve_web_tools_config_from_toolset_configs,
)
from dayu.execution.runtime_config import AgentRuntimeConfig, RunnerRuntimeConfig
from dayu.fins.domain.enums import SourceKind
from dayu.fins.service_runtime import DefaultFinsRuntime
from dayu.fins.storage import FsSourceDocumentRepository
from dayu.host import Host
from dayu.log import Log, set_level_from_flags
from dayu.services import prepare_host_runtime_dependencies, WriteRunConfig
from dayu.services.chat_service import ChatService
from dayu.services.fins_service import FinsService
from dayu.services.prompt_service import PromptService
from dayu.services.scene_execution_acceptance import SceneExecutionAcceptancePreparer
from dayu.services.write_service import WriteService
from dayu.services.contracts import WriteRequest
from dayu.startup.config_file_resolver import ConfigFileResolver
from dayu.startup.config_loader import ConfigLoader
from dayu.startup.config_file_resolver import resolve_package_assets_path
from dayu.startup.paths import resolve_startup_paths
from dayu.startup.workspace import WorkspaceResources
from dayu.workspace_paths import build_host_store_default_path, build_interactive_state_dir



MODULE = "APP.MAIN"


_COMMANDS_ALLOW_MISSING_FILINGS_DIR = frozenset(
    {
        "download",
        "interactive",
        "prompt",
        "upload_filing",
        "upload_filings_from",
        "upload_material",
    }
)


_COMMANDS_WARN_ON_MISSING_FILINGS_DIR = frozenset(
    {
        "interactive",
        "prompt",
    }
)


@dataclass(frozen=True)
class WorkspaceConfig:
    """CLI 工作区路径配置。"""

    workspace_dir: Path
    output_dir: Path
    config_root: Path | None = None
    ticker: str | None = None
    has_local_filings: bool = False
    config_loader: ConfigLoaderProtocol | None = None
    prompt_asset_store: PromptAssetStoreProtocol | None = None


def _has_local_filing_storage_root(workspace_dir: Path, ticker: str) -> bool:
    """通过源文档仓储判断本地 filing 根目录是否存在。

    Args:
        workspace_dir: 工作区根目录。
        ticker: 股票代码。

    Returns:
        若 filing 根目录存在且为目录则返回 `True`，否则返回 `False`。

    Raises:
        NotADirectoryError: filing 根路径存在但不是目录时抛出。
        OSError: 仓储检查失败时抛出。
    """

    source_repository = FsSourceDocumentRepository(
        workspace_dir,
        create_directories=False,
    )
    return source_repository.has_source_storage_root(ticker, SourceKind.FILING)


class _WebToolsConfigLike(Protocol):
    """CLI 侧只关心 ``provider`` 字段的 web 配置视图。"""

    @property
    def provider(self) -> str:
        """返回当前 web provider 名称。"""

        ...


@dataclass(frozen=True)
class _DefaultWebToolsConfig:
    """CLI 本地兜底的最小 web 配置快照。"""

    provider: str = "auto"


@dataclass(frozen=True)
class RunningConfig:
    """CLI 侧可见的运行配置快照。"""

    runner_running_config: RunnerRuntimeConfig
    agent_running_config: AgentRuntimeConfig
    doc_tool_limits: DocToolLimits | None
    fins_tool_limits: FinsToolLimits | None
    web_tools_config: _WebToolsConfigLike
    tool_trace_config: TraceSettings
    model_name: str = ""
    temperature: float | None = None

    @property
    def trace_settings(self) -> TraceSettings:
        """兼容 runtime 命名。"""

        return self.tool_trace_config

    @classmethod
    def from_resolved(cls, resolved: ResolvedExecutionOptions) -> "RunningConfig":
        """从 runtime 选项转换为 CLI 运行配置。"""

        return cls(
            runner_running_config=resolved.runner_running_config,
            agent_running_config=resolved.agent_running_config,
            doc_tool_limits=resolve_doc_tool_limits_from_toolset_configs(resolved.toolset_configs),
            fins_tool_limits=resolve_fins_tool_limits_from_toolset_configs(resolved.toolset_configs),
            web_tools_config=(
                resolve_web_tools_config_from_toolset_configs(resolved.toolset_configs)
                or _DefaultWebToolsConfig()
            ),
            tool_trace_config=TraceSettings(
                enabled=resolved.trace_settings.enabled,
                output_dir=resolved.trace_settings.output_dir,
                max_file_bytes=resolved.trace_settings.max_file_bytes,
                retention_days=resolved.trace_settings.retention_days,
                compress_rolled=resolved.trace_settings.compress_rolled,
                partition_by_session=resolved.trace_settings.partition_by_session,
            ),
            model_name=resolved.model_name,
            temperature=resolved.temperature,
        )


@dataclass(frozen=True)
class ModelName:
    """模型名包装对象。"""

    model_name: str

@dataclass
class WriteCliConfig:
    """CLI 写作模式配置。"""

    enabled: bool
    template_path: Path
    output_dir: Path
    audit_model_override_name: str
    write_max_retries: int
    resume: bool
    web_provider: str
    chapter_filter: str = ""
    fast: bool = False
    force: bool = False
    infer: bool = False


def _build_execution_options(args: argparse.Namespace) -> ExecutionOptions:
    """从 CLI 参数构建请求级执行选项。

    Args:
        args: 命令行参数对象。

    Returns:
        执行选项对象。

    Raises:
        SystemExit: limits JSON 非法时退出。
    """

    return build_execution_options_from_args(args)


def _build_interactive_state_store(workspace_dir: Path) -> FileInteractiveStateStore:
    """构造 interactive 状态仓储。

    Args:
        workspace_dir: 工作区根目录。

    Returns:
        interactive 状态仓储。

    Raises:
        无。
    """

    return FileInteractiveStateStore(build_interactive_state_dir(workspace_dir))


def _purge_old_interactive_session(
    store: FileInteractiveStateStore,
    workspace_dir: Path,
) -> None:
    """清理旧 interactive 会话在 Host DB 中的残留数据。

    读取当前 interactive 状态，若存在旧 session_id，则从 Host DB
    中删除对应的 pending turns 和 reply outbox 记录。

    Args:
        store: interactive 状态仓储。
        workspace_dir: 工作区根目录，用于定位 Host DB。

    Returns:
        无。

    Raises:
        无。清理失败仅记录日志。
    """

    old_state: InteractiveSessionState | None = None
    try:
        old_state = store.load()
    except (ValueError, OSError):
        # 状态文件损坏，无法定位旧 session_id，跳过清理
        pass
    if old_state is None:
        return
    old_session_id = resolve_interactive_state_session_id(old_state)
    host_db_path = build_host_store_default_path(workspace_dir)

    from dayu.host.host_cleanup import purge_sessions_from_host_db

    total_pending, total_outbox = purge_sessions_from_host_db(
        host_db_path=host_db_path,
        session_ids=[old_session_id],
    )
    if total_pending or total_outbox:
        Log.info(
            f"已清理旧 interactive 会话数据: pending_turns={total_pending}, reply_outbox={total_outbox}",
            module=MODULE,
        )
    Log.info("开启新会话，清理旧会话绑定", module=MODULE)


def _resolve_interactive_session_id(workspace_dir: Path, *, new_session: bool) -> str:
    """解析 interactive 启动时应绑定的 session_id。

    当 ``new_session=True`` 时，先清理旧会话在 Host DB 中残留的
    pending turns 和 reply outbox，再重建会话绑定。

    Args:
        workspace_dir: 工作区根目录。
        new_session: 是否显式要求开启新会话。

    Returns:
        供 interactive 使用的确定性 session_id。

    Raises:
        ValueError: 当状态文件损坏时抛出。
    """

    store = _build_interactive_state_store(workspace_dir)
    if new_session:
        _purge_old_interactive_session(store, workspace_dir)
        store.clear()
    state = store.load()
    if state is None:
        state = InteractiveSessionState(interactive_key=build_interactive_key())
        store.save(state)
    return resolve_interactive_state_session_id(state)


def _bind_interactive_session_id(workspace_dir: Path, session_id: str) -> str:
    """把 interactive 本地绑定切换到指定 Host session。

    Args:
        workspace_dir: 工作区根目录。
        session_id: 目标 Host session ID。

    Returns:
        已绑定的 Host session ID。

    Raises:
        ValueError: 当 `session_id` 为空时抛出。
    """

    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        raise ValueError("session_id 不能为空")
    store = _build_interactive_state_store(workspace_dir)
    state = InteractiveSessionState(
        interactive_key=build_interactive_key(),
        session_id=normalized_session_id,
    )
    store.save(state)
    return normalized_session_id


def setup_paths(args: argparse.Namespace) -> WorkspaceConfig:
    """处理并验证工作区路径。

    Args:
        args: 命令行参数对象。

    Returns:
        CLI 工作区路径配置。

    Raises:
        SystemExit: 路径非法时退出。
    """

    workspace_dir = Path(args.base).expanduser().resolve()
    if not workspace_dir.exists():
        Log.error(f"输入目录不存在: {workspace_dir}", module=MODULE)
        raise SystemExit(1)
    if not workspace_dir.is_dir():
        Log.error(f"输入路径不是目录: {workspace_dir}", module=MODULE)
        raise SystemExit(1)

    raw_config_root = getattr(args, "config", None)
    config_root = Path(raw_config_root).expanduser().resolve() if raw_config_root else None
    output_dir = (workspace_dir / "output").resolve()

    raw_ticker = getattr(args, "ticker", None)
    ticker = str(raw_ticker).strip().upper() if raw_ticker else None
    has_local_filings = False
    command_name = str(getattr(args, "command", "") or "").strip()
    if ticker:
        try:
            has_local_filings = _has_local_filing_storage_root(workspace_dir, ticker)
        except NotADirectoryError as exc:
            Log.error(f"财报目录不是目录: {exc}", module=MODULE)
            raise SystemExit(1) from exc
        except OSError as exc:
            Log.error(f"财报目录检查失败: ticker={ticker}, error={exc}", module=MODULE)
            raise SystemExit(1) from exc
        if not has_local_filings and command_name not in _COMMANDS_ALLOW_MISSING_FILINGS_DIR:
            Log.error(f"财报目录不存在: ticker={ticker}", module=MODULE)
            raise SystemExit(1)
        if not has_local_filings and command_name in _COMMANDS_WARN_ON_MISSING_FILINGS_DIR:
            Log.warning(f"财报目录不存在，将按无本地财报继续: ticker={ticker}", module=MODULE)

    return WorkspaceConfig(
        workspace_dir=workspace_dir,
        config_root=config_root,
        output_dir=output_dir,
        ticker=ticker,
        has_local_filings=has_local_filings,
    )




def load_running_config(args: argparse.Namespace, paths_config: WorkspaceConfig) -> RunningConfig:
    """通过启动期依赖解析并返回默认执行选项。

    Args:
        args: 命令行参数对象。
        paths_config: 已解析的工作区路径配置。

    Returns:
        CLI 可见的运行配置快照。

    Raises:
        无。
    """

    default_execution_options = _prepare_cli_default_execution_options(
        workspace_config=paths_config,
        execution_options=_build_execution_options(args),
    )
    return RunningConfig.from_resolved(default_execution_options)


def _resolve_tool_trace_output_dir(raw_output_dir: str, workspace_dir: Path) -> Path:
    """解析工具追踪输出目录。

    Args:
        raw_output_dir: 配置中的目录字符串。
        workspace_dir: 工作区根目录。

    Returns:
        规范化后的绝对路径。

    Raises:
        无。
    """

    candidate = Path(raw_output_dir).expanduser()
    if candidate.is_absolute():
        return candidate
    return (workspace_dir / candidate).resolve()


def _build_scene_execution_options(
    *,
    execution_options: ExecutionOptions | None,
    model_name: str,
) -> ExecutionOptions | None:
    """为指定 scene 构建模型覆盖后的执行选项。

    Args:
        execution_options: 原始请求级执行选项。
        model_name: 目标模型名；为空时表示不注入请求级模型覆盖。

    Returns:
        适用于该 scene 的执行选项。

    Raises:
        无。
    """

    normalized_model_name = str(model_name or "").strip()
    stripped_options = execution_options
    if stripped_options is not None:
        stripped_options = replace(stripped_options, model_name=None)
    if not normalized_model_name:
        return stripped_options
    if stripped_options is None:
        return ExecutionOptions(model_name=normalized_model_name)
    return replace(stripped_options, model_name=normalized_model_name)




def _prepare_cli_default_execution_options(
    *,
    workspace_config: WorkspaceConfig,
    execution_options: ExecutionOptions | None,
) -> ResolvedExecutionOptions:
    """准备 CLI 默认执行基线。

    Args:
        workspace_config: 工作区路径配置。
        execution_options: 请求级执行选项。

    Returns:
        解析后的默认执行选项。

    Raises:
        无。
    """

    paths = resolve_startup_paths(
        workspace_root=workspace_config.workspace_dir,
        config_root=workspace_config.config_root,
    )
    resolver = ConfigFileResolver(paths.config_root)
    config_loader = ConfigLoader(resolver)
    base_execution_options = build_base_execution_options(
        workspace_dir=paths.workspace_root,
        run_config=config_loader.load_run_config(),
    )
    return merge_execution_options(
        base_options=base_execution_options,
        workspace_dir=paths.workspace_root,
        execution_options=execution_options,
    )


def _prepare_cli_host_dependencies(
    *,
    workspace_config: WorkspaceConfig,
    execution_options: ExecutionOptions | None,
) -> tuple[
    WorkspaceResources,
    ResolvedExecutionOptions,
    SceneExecutionAcceptancePreparer,
    Host,
    DefaultFinsRuntime,
]:
    """准备 CLI 的 Host 级稳定依赖。

    Args:
        workspace_config: 工作区路径配置。
        execution_options: 请求级执行选项。

    Returns:
        `(
            workspace,
            default_execution_options,
            scene_execution_acceptance_preparer,
            host,
            fins_runtime,
        )`。

    Raises:
        无。
    """

    prepared = prepare_host_runtime_dependencies(
        workspace_root=workspace_config.workspace_dir,
        config_root=workspace_config.config_root,
        execution_options=execution_options,
        runtime_label="CLI Host runtime",
        log_module=MODULE,
    )
    register_cli_shutdown_hook(prepared.host)
    return (
        prepared.workspace,
        prepared.default_execution_options,
        prepared.scene_execution_acceptance_preparer,
        prepared.host,
        prepared.fins_runtime,
    )


def _build_chat_service(
    *,
    host: Host,
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer,
    fins_runtime: DefaultFinsRuntime,
) -> ChatService:
    """构建交互聊天服务。

    Args:
        host: 宿主对象。
        scene_execution_acceptance_preparer: scene 执行参数接受器。
        fins_runtime: 财报运行时。

    Returns:
        聊天服务实例。

    Raises:
        无。
    """

    return ChatService(
        host=host,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        company_name_resolver=fins_runtime.get_company_name,
        session_source=SessionSource.CLI,
    )


def _build_prompt_service(
    *,
    host: Host,
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer,
    fins_runtime: DefaultFinsRuntime,
) -> PromptService:
    """构建单轮 prompt 服务。

    Args:
        host: 宿主对象。
        scene_execution_acceptance_preparer: scene 执行参数接受器。
        fins_runtime: 财报运行时。

    Returns:
        prompt 服务实例。

    Raises:
        无。
    """

    return PromptService(
        host=host,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        company_name_resolver=fins_runtime.get_company_name,
        session_source=SessionSource.CLI,
    )


def _build_write_service(
    *,
    host: Host,
    workspace: WorkspaceResources,
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer,
    fins_runtime: DefaultFinsRuntime,
) -> WriteService:
    """构建写作服务。

    Args:
        host: 宿主对象。
        workspace: 工作区稳定资源。
        scene_execution_acceptance_preparer: scene 执行参数接受器。
        fins_runtime: 财报运行时。

    Returns:
        写作服务实例。

    Raises:
        无。
    """

    return WriteService(
        host=host,
        host_governance=host,
        workspace=workspace,
        scene_execution_acceptance_preparer=scene_execution_acceptance_preparer,
        company_name_resolver=fins_runtime.get_company_name,
        company_meta_summary_resolver=fins_runtime.get_company_meta_summary,
    )


def _build_fins_ops_service(args: argparse.Namespace) -> FinsService:
    """构建财报服务。

    Args:
        args: 命令行参数对象。

    Returns:
        财报服务实例。

    Raises:
        无。
    """

    workspace_config = setup_paths(args)
    (
        _workspace,
        _default_execution_options,
        _scene_execution_acceptance_preparer,
        host,
        fins_runtime,
    ) = _prepare_cli_host_dependencies(
        workspace_config=workspace_config,
        execution_options=None,
    )
    return FinsService(
        host=host,
        fins_runtime=fins_runtime,
        session_source=SessionSource.CLI,
    )


def setup_model_name(args: argparse.Namespace) -> ModelName:
    """从 CLI 参数构建模型名称。

    Args:
        args: 命令行参数对象。

    Returns:
        ModelName 实例。

    Raises:
        无。
    """

    return ModelName(model_name=str(getattr(args, "model_name", "") or "").strip())


def setup_write_config(args: argparse.Namespace, paths_config: WorkspaceConfig, running_config: RunningConfig) -> WriteCliConfig:
    """构建写作模式配置。

    Args:
        args: 命令行参数对象。
        paths_config: 路径配置对象。
        running_config: 运行时配置对象，用于读取 `web_tools_config` 默认值。

    Returns:
        `WriteCliConfig` 写作配置对象。

    Raises:
        SystemExit: 当写作参数非法时退出。
    """

    raw_output = getattr(args, "output", None)
    raw_template = getattr(args, "template", None)
    raw_write_max_retries = int(getattr(args, "write_max_retries", 2))
    raw_resume = bool(getattr(args, "resume", True))
    raw_web_provider = getattr(args, "web_provider", None)
    raw_audit_model_name = str(getattr(args, "audit_model_name", "") or "").strip()
    raw_chapter_filter = str(getattr(args, "chapter", None) or "")
    raw_fast = bool(getattr(args, "fast", False))
    raw_force = bool(getattr(args, "force", False))
    raw_infer = bool(getattr(args, "infer", False))

    output_dir = _resolve_write_output_dir(
        workspace_dir=paths_config.workspace_dir,
        ticker=paths_config.ticker,
        raw_output=raw_output,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if raw_template is not None:
        template_path = Path(raw_template).expanduser()
        if not template_path.is_absolute():
            template_path = (Path.cwd() / template_path).resolve()
    else:
        template_path = _resolve_default_write_template_path(paths_config.workspace_dir)
    if not template_path.exists() or not template_path.is_file():
        Log.error(f"模板文件不存在: {template_path}", module=MODULE)
        raise SystemExit(2)

    if raw_write_max_retries < 0:
        Log.error("--write-max-retries 不能为负数", module=MODULE)
        raise SystemExit(2)

    return WriteCliConfig(
        enabled=(args.command == "write"),
        template_path=template_path,
        output_dir=output_dir,
        audit_model_override_name=raw_audit_model_name,
        write_max_retries=raw_write_max_retries,
        resume=raw_resume,
        web_provider=str(raw_web_provider or running_config.web_tools_config.provider),
        chapter_filter=raw_chapter_filter,
        fast=raw_fast,
        force=raw_force,
        infer=raw_infer,
    )


def _resolve_default_write_template_path(workspace_dir: Path) -> Path:
    """解析默认写作模板路径。

    Args:
        workspace_dir: 工作区根目录。

    Returns:
        优先使用工作区模板；若不存在，则返回包内默认模板路径。

    Raises:
        无。
    """

    workspace_template = workspace_dir / "assets" / "定性分析模板.md"
    if workspace_template.exists():
        return workspace_template
    return resolve_package_assets_path() / "定性分析模板.md"


def _resolve_write_output_dir(*, workspace_dir: Path, ticker: str | None, raw_output: str | None) -> Path:
    """解析写作相关命令的输出目录。

    Args:
        workspace_dir: 工作区根目录。
        ticker: 当前股票代码；存在时默认落到 `draft/{ticker}`。
        raw_output: CLI 显式传入的输出目录。

    Returns:
        规范化后的输出目录绝对路径。

    Raises:
        无。
    """

    if raw_output is not None:
        return Path(raw_output).expanduser().resolve()
    default_output_dir = (workspace_dir / "draft").resolve()
    if ticker:
        return (default_output_dir / ticker).resolve()
    return default_output_dir


def setup_loglevel(args: argparse.Namespace) -> None:
    """根据命令行参数设置日志级别。

    Args:
        args: argparse 解析结果对象。

    Returns:
        无。

    Raises:
        KeyError: 传入未知日志级别名称时抛出。
    """

    # 即使无显式 flag，也需调用 set_level 以触发第三方库抑制逻辑。
    set_level_from_flags(
        log_level=getattr(args, "log_level", None),
        debug=bool(getattr(args, "debug", False)),
        verbose=bool(getattr(args, "verbose", False)),
        info=bool(getattr(args, "info", False)),
        quiet=bool(getattr(args, "quiet", False)),
    )


def run_write_pipeline(
    *,
    write_config: WriteRunConfig,
    write_service: WriteService | None = None,
) -> int:
    """执行写作流水线入口。

    Args:
        write_config: 写作配置对象。
        write_service: 预装配写作服务。

    Returns:
        流水线退出码。

    Raises:
        RuntimeError: 写作执行异常时透传。
    """

    if write_service is None:
        raise ValueError("run_write_pipeline 需要注入 write_service")
    return write_service.run(WriteRequest(write_config=write_config))
