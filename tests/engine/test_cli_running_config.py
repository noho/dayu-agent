"""dayu.cli 运行配置加载测试。"""

from __future__ import annotations

import inspect
import json
import sys
from argparse import Namespace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest

from dayu.contracts.cancellation import CancelledError
from dayu.cli.arg_parsing import parse_arguments
from dayu.cli.commands.interactive import run_interactive_command
from dayu.cli.commands.prompt import run_prompt_command
from dayu.cli.commands.write import run_write_command
from dayu.cli.dependency_setup import (
    ModelName,
    RunningConfig,
    WorkspaceConfig,
    _prepare_cli_host_dependencies,
    _has_local_filing_storage_root,
    _resolve_interactive_session_id,
    _resolve_write_output_dir,
    _resolve_tool_trace_output_dir,
    load_running_config,
    run_write_pipeline,
    setup_loglevel,
    setup_model_name,
    setup_paths,
    setup_write_config,
)
from dayu.cli.commands.fins import _build_fins_command, run_fins_command
from dayu.cli.main import main
from dayu.cli.interactive_state import (
    FileInteractiveStateStore,
    InteractiveSessionState,
    build_interactive_session_id,
)
from dayu.services.contracts import FinsSubmission, SceneModelConfig, WriteRunConfig
from dayu.services.scene_execution_acceptance import SceneExecutionAcceptancePreparer
from dayu.services.write_service import WriteService
from dayu.startup.workspace import WorkspaceResources
from dayu.host.protocols import HostedExecutionGatewayProtocol
from dayu.host import Host
from dayu.contracts.agent_types import AgentMessage, AgentTraceIdentity
from dayu.contracts.fins import (
    DownloadCommandPayload,
    DownloadProgressPayload,
    DownloadResultData,
    DownloadSummary,
    FinsEvent,
    FinsEventType,
    FinsProgressEventName,
    UploadFilingCommandPayload,
    UploadFilingProgressPayload,
    UploadFilingResultData,
    UploadMaterialCommandPayload,
)
from dayu.contracts.tool_configs import DocToolLimits, FinsToolLimits, WebToolsConfig
from dayu.contracts.toolset_config import build_toolset_config_snapshot
from dayu.engine.doc_access_policy import build_effective_doc_allowed_paths
from dayu.engine.events import content_delta, final_answer_event
from dayu.engine.toolset_registrars import (
    register_doc_toolset as _register_doc_toolset,
    register_web_toolset as _register_web_toolset,
)
from dayu.execution.runtime_config import (
    AgentRuntimeConfig as AgentRunningConfig,
    OpenAIRunnerRuntimeConfig as AsyncOpenAIRunnerRunningConfig,
)
from dayu.execution.options import ResolvedExecutionOptions, TraceSettings
from dayu.fins.service_runtime import DefaultFinsRuntime
from dayu.startup.config_file_resolver import ConfigFileResolver
from dayu.startup.config_loader import ConfigLoader
from dayu.startup.prompt_assets import FilePromptAssetStore
from dayu.fins.toolset_registrars import register_fins_read_toolset as _register_fins_read_toolset
from dayu.services.startup_preparation import PreparedHostRuntimeDependencies
class _CallCollector:
    """测试调用记录器。"""

    def __init__(self) -> None:
        """初始化记录器。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.levels: list[str] = []
        self.interactive_calls = 0
        self.info_logs: list[str] = []
        self.warn_logs: list[str] = []
        self.error_logs: list[str] = []

    def capture_level(self, level: object) -> None:
        """记录日志级别。

        Args:
            level: 日志级别对象。

        Returns:
            无。

        Raises:
            无。
        """

        self.levels.append(getattr(level, "name", str(level)))

    def mark_interactive(self, *_args: object, **_kwargs: object) -> None:
        """记录 interactive 调用。

        Args:
            *_args: 位置参数。
            **_kwargs: 关键字参数。

        Returns:
            无。

        Raises:
            无。
        """

        self.interactive_calls += 1

    def capture_info(self, message: object, **_kwargs: object) -> None:
        """记录 info 日志内容。

        Args:
            message: 日志消息对象。
            **_kwargs: 关键字参数。

        Returns:
            无。

        Raises:
            无。
        """

        self.info_logs.append(str(message))

    def capture_warn(self, message: object, **_kwargs: object) -> None:
        """记录 warn 日志内容。

        Args:
            message: 日志消息对象。
            **_kwargs: 关键字参数。

        Returns:
            无。

        Raises:
            无。
        """

        self.warn_logs.append(str(message))

    def capture_error(self, message: object, **_kwargs: object) -> None:
        """记录 error 日志内容。

        Args:
            message: 日志消息对象。
            **_kwargs: 关键字参数。

        Returns:
            无。

        Raises:
            无。
        """

        self.error_logs.append(str(message))


class _FakeCliHostDependencies:
    """测试用 CLI Host 稳定依赖集合。"""

    def __init__(
        self,
        *,
        running_config: RunningConfig | None = None,
        scene_model_names: dict[str, object] | None = None,
    ) -> None:
        """初始化测试用 CLI Host 稳定依赖。

        Args:
            running_config: CLI 运行配置快照。
            scene_model_names: scene 到模型配置的映射。

        Returns:
            无。

        Raises:
            无。
        """

        resolved_running_config = running_config or RunningConfig(
            runner_running_config=AsyncOpenAIRunnerRunningConfig(),
            agent_running_config=AgentRunningConfig(),
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(provider="auto"),
            tool_trace_config=TraceSettings(enabled=False, output_dir=Path("/tmp/trace")),
        )
        self._scene_model_names = dict(scene_model_names or {})
        self.default_execution_options = _running_config_to_resolved_namespace(resolved_running_config)
        self.workspace = SimpleNamespace()
        self.host = SimpleNamespace(
            create_session=lambda **_kwargs: SimpleNamespace(session_id="fake-session-id"),
        )
        self.fins_runtime = SimpleNamespace(
            get_company_name=lambda *_args, **_kwargs: "",
            get_company_meta_summary=lambda *_args, **_kwargs: "",
        )
        self.scene_execution_acceptance_preparer = SimpleNamespace(
            resolve_scene_model=self.resolve_scene_model_config,
        )

    def resolve_scene_execution_options(self, scene_name: str, _execution_options=None):
        """返回预置 scene 模型解析结果。"""

        model_config = self._scene_model_names.get(scene_name, "")
        if isinstance(model_config, dict):
            return SimpleNamespace(
                model_name=model_config.get("name", ""),
                temperature=model_config.get("temperature"),
            )
        return SimpleNamespace(model_name=model_config, temperature=None)

    def resolve_scene_model_config(self, scene_name: str, _execution_options=None) -> SceneModelConfig:
        """返回预置 scene 最终模型配置。"""

        model_config = self._scene_model_names.get(scene_name, "")
        if isinstance(model_config, dict):
            return SceneModelConfig(
                name=str(model_config.get("name", "")),
                temperature=float(model_config.get("temperature") or 0.0),
            )
        return SceneModelConfig(
            name=str(model_config),
            temperature=0.0,
        )

    def as_tuple(self) -> tuple[object, object, object, object, object]:
        """返回与 CLI Host 依赖装配函数一致的元组。

        Args:
            无。

        Returns:
            `(workspace, default_execution_options, scene_execution_acceptance_preparer, host, fins_runtime)`。

        Raises:
            无。
        """

        return (
            self.workspace,
            self.default_execution_options,
            self.scene_execution_acceptance_preparer,
            self.host,
            self.fins_runtime,
        )


class _PromptAgentE2ERecorder:
    """记录 prompt 全链路测试里送往 Agent 的最终输入。"""

    def __init__(self) -> None:
        """初始化记录器。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.agent_create_args: Any | None = None
        self.tool_executor: Any | None = None
        self.tool_trace_recorder_factory: Any | None = None
        self.trace_identity: dict[str, str] | None = None
        self.messages: list[dict[str, Any]] | None = None
        self.session_id: str | None = None
        self.run_id: str | None = None


class _MockPromptAgent:
    """替换真实 Agent 的最小测试桩。"""

    def __init__(self, recorder: _PromptAgentE2ERecorder) -> None:
        """初始化测试桩。

        Args:
            recorder: 贯通测试记录器。

        Returns:
            无。

        Raises:
            无。
        """

        self._recorder = recorder

    async def run_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        stream: bool = True,
    ):
        """记录传入消息并返回最小可消费事件流。

        Args:
            messages: 送入 Agent 的最终消息列表。
            session_id: 会话标识。
            run_id: Host run 标识。
            stream: 是否流式执行。

        Returns:
            异步事件流。

        Raises:
            AssertionError: 当上游未按流式模式调用时抛出。
        """

        assert stream is True
        self._recorder.messages = list(messages)
        self._recorder.session_id = session_id
        self._recorder.run_id = run_id
        yield content_delta("mock content")
        yield final_answer_event("mock final answer")


class _MockPromptAgentBuilder:
    """替换 `build_async_agent` 的测试构造器。"""

    def __init__(self, recorder: _PromptAgentE2ERecorder) -> None:
        """初始化构造器。

        Args:
            recorder: 贯通测试记录器。

        Returns:
            无。

        Raises:
            无。
        """

        self._recorder = recorder

    def __call__(
        self,
        *,
        agent_create_args: Any,
        tool_executor: Any = None,
        tool_trace_recorder_factory: Any | None = None,
        trace_identity: AgentTraceIdentity | None = None,
        cancellation_token: Any | None = None,
    ) -> _MockPromptAgent:
        """记录 Agent 构造参数并返回 MockAgent。

        Args:
            agent_create_args: 最终 Agent 创建参数。
            tool_executor: 最终工具执行器。
            tool_trace_recorder_factory: trace 工厂。
            trace_identity: trace 身份。
            cancellation_token: 取消令牌。

        Returns:
            可执行的 MockAgent。

        Raises:
            无。
        """

        _ = cancellation_token
        self._recorder.agent_create_args = agent_create_args
        self._recorder.tool_executor = tool_executor
        self._recorder.tool_trace_recorder_factory = tool_trace_recorder_factory
        self._recorder.trace_identity = trace_identity.to_metadata() if trace_identity is not None else {}
        return _MockPromptAgent(self._recorder)


class _WebToolsRegistrationRecorder:
    """记录 scene preparation 注册 web tools 时的关键参数。"""

    def __init__(self) -> None:
        """初始化记录器。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.calls: list[dict[str, Any]] = []

    def __call__(self, context: Any) -> int:
        """记录 web tools 注册参数并调用真实注册逻辑。

        Args:
            context: toolset 注册上下文。

        Returns:
            无。

        Raises:
            无。
        """

        payload = context.toolset_config.payload if context.toolset_config is not None else {}
        self.calls.append(
            {
                "provider": payload.get("provider"),
                "request_timeout_seconds": payload.get("request_timeout_seconds"),
                "max_search_results": payload.get("max_search_results"),
                "fetch_truncate_chars": payload.get("fetch_truncate_chars"),
                "timeout_budget": context.tool_timeout_seconds,
                "allow_private_network_url": context.execution_permissions.web.allow_private_network_url,
                "playwright_channel": payload.get("playwright_channel"),
                "playwright_storage_state_dir": payload.get("playwright_storage_state_dir"),
            }
        )
        return _register_web_toolset(context)


class _DocToolsRegistrationRecorder:
    """记录 scene preparation 注册 doc tools 时的关键参数。"""

    def __init__(self) -> None:
        """初始化记录器。"""

        self.calls: list[dict[str, Any]] = []

    def __call__(self, context: Any) -> int:
        """记录 doc tools 注册参数并调用真实注册逻辑。

        Args:
            context: toolset 注册上下文。

        Returns:
            无。

        Raises:
            无。
        """

        payload = context.toolset_config.payload if context.toolset_config is not None else {}
        self.calls.append(
            {
                "list_files_max": payload.get("list_files_max"),
                "get_sections_max": payload.get("get_sections_max"),
                "search_files_max_results": payload.get("search_files_max_results"),
                "read_file_max_chars": payload.get("read_file_max_chars"),
                "read_file_section_max_chars": payload.get("read_file_section_max_chars"),
                "allowed_paths": [
                    str(path)
                    for path in build_effective_doc_allowed_paths(
                        workspace=context.workspace,
                        doc_permissions=context.execution_permissions.doc,
                    )
                ],
                "allow_file_write": context.execution_permissions.doc.allow_file_write,
                "allowed_write_paths": list(context.execution_permissions.doc.allowed_write_paths),
                "timeout_budget": context.tool_timeout_seconds,
            }
        )
        return _register_doc_toolset(context)


class _FinsReadToolsRegistrationRecorder:
    """记录 scene preparation 注册 fins 读取工具时的关键参数。"""

    def __init__(self) -> None:
        """初始化记录器。"""

        self.calls: list[dict[str, Any]] = []

    def __call__(self, context: Any) -> int:
        """记录 fins 读取工具注册参数并调用真实注册逻辑。

        Args:
            context: toolset 注册上下文。

        Returns:
            无。

        Raises:
            无。
        """

        payload = context.toolset_config.payload if context.toolset_config is not None else {}
        self.calls.append(
            {
            "processor_cache_max_entries": payload.get("processor_cache_max_entries"),
            "list_documents_max_items": payload.get("list_documents_max_items"),
            "get_document_sections_max_items": payload.get("get_document_sections_max_items"),
            "search_document_max_items": payload.get("search_document_max_items"),
            "read_section_max_chars": payload.get("read_section_max_chars"),
            "get_page_content_max_chars": payload.get("get_page_content_max_chars"),
                "timeout_budget": context.tool_timeout_seconds,
                "has_service": True,
                "has_repository": False,
                "has_processor_registry": False,
            }
        )
        return _register_fins_read_toolset(context)


def _running_config_to_resolved_namespace(running_config: RunningConfig) -> SimpleNamespace:
    """将 `RunningConfig` 转成 CLI 启动依赖里的 resolved options。"""

    return SimpleNamespace(
        runner_running_config=running_config.runner_running_config,
        agent_running_config=running_config.agent_running_config,
        toolset_configs=tuple(
            snapshot
            for snapshot in (
                build_toolset_config_snapshot("doc", running_config.doc_tool_limits),
                build_toolset_config_snapshot("fins", running_config.fins_tool_limits),
                build_toolset_config_snapshot("web", running_config.web_tools_config),
            )
            if snapshot is not None
        ),
        trace_settings=running_config.tool_trace_config,
        model_name=running_config.model_name,
        temperature=running_config.temperature,
    )


def _as_fins_tool_limits(value: object | None) -> FinsToolLimits:
    """把 CLI 运行配置中的财报 limits 收窄为测试断言所需类型。"""

    return cast(FinsToolLimits, value)


def _as_web_tools_config(value: object) -> WebToolsConfig:
    """把 CLI 运行配置中的 web 配置收窄为测试断言所需类型。"""

    return cast(WebToolsConfig, value)


def _as_openai_runner_config(value: object) -> AsyncOpenAIRunnerRunningConfig:
    """把 RunnerRuntimeConfig 收窄为 OpenAI runner 配置。"""

    return cast(AsyncOpenAIRunnerRunningConfig, value)


def _expected_doc_allowed_paths(workspace_dir: Path) -> list[str]:
    """根据当前文件系统状态构造 doc 工具允许读取的路径列表。

    Args:
        workspace_dir: 工作区目录。

    Returns:
        当前实现下会被 Host 放行的读取白名单路径列表。

    Raises:
        无。
    """

    return [str(workspace_dir.resolve())]


def _write_json_file(file_path: Path, payload: object) -> None:
    """写入测试用 JSON 文件。

    Args:
        file_path: 目标文件路径。
        payload: 待写入 JSON 对象。

    Returns:
        无。

    Raises:
        OSError: 文件写入失败时抛出。
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text_file(file_path: Path, content: str) -> None:
    """写入测试用文本文件。

    Args:
        file_path: 目标文件路径。
        content: 待写入文本。

    Returns:
        无。

    Raises:
        OSError: 文件写入失败时抛出。
    """

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")


def _extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    """从最终消息列表里提取 system prompt。

    Args:
        messages: 送往 Agent 的最终消息列表。

    Returns:
        system 消息内容。

    Raises:
        AssertionError: 未找到 system 消息时抛出。
    """

    for message in messages:
        if message.get("role") == "system":
            return str(message.get("content") or "")
    raise AssertionError("未找到 system prompt")


def _return_value(value: object, *_args: object, **_kwargs: object) -> object:
    """返回预置值（用于 monkeypatch）。

    Args:
        value: 预置返回值。
        *_args: 任意位置参数。
        **_kwargs: 任意关键字参数。

    Returns:
        预置返回值。

    Raises:
        无。
    """

    return value


def _raise_runtime_error_for_write_pipeline(**_kwargs: object) -> int:
    """抛出固定异常，覆盖写作流水线失败分支。

    Args:
        **_kwargs: 关键字参数。

    Returns:
        逻辑上不会返回。

    Raises:
        RuntimeError: 固定抛出。
    """

    raise RuntimeError("boom")


@pytest.mark.unit
def test_main_dispatches_host_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """host 子命令应直接分发到宿主管理入口。"""

    args = Namespace(command="host")
    monkeypatch.setattr("dayu.cli.main.parse_arguments", partial(_return_value, args))
    monkeypatch.setattr("dayu.cli.commands.host.run_host_command", lambda args: 0)

    assert main() == 0


@pytest.mark.unit
def test_main_dispatches_init_without_runtime_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`init` 子命令不应触发运行时装配路径。"""

    args = Namespace(command="init")
    monkeypatch.setattr("dayu.cli.main.parse_arguments", partial(_return_value, args))
    monkeypatch.setattr("dayu.cli.commands.init.run_init_command", lambda args: 7)

    assert main() == 7


def _build_args() -> Namespace:
    """构造 `load_running_config` 所需参数对象。

    Args:
        无。

    Returns:
        仅包含运行配置覆盖字段的命令行参数对象。

    Raises:
        无。
    """

    return Namespace(
        debug_sse=False,
        debug_tool_delta=False,
        debug_sse_sample_rate=None,
        debug_sse_throttle_sec=None,
        tool_timeout_seconds=None,
        enable_tool_trace=False,
        tool_trace_dir=None,
        max_iterations=None,
        fallback_mode=None,
        fallback_prompt=None,
        max_consecutive_failed_tool_batches=None,
        max_duplicate_tool_calls=None,
        duplicate_tool_hint_prompt=None,
        web_provider=None,
    )


def _build_workspace_config(workspace_dir: Path) -> WorkspaceConfig:
    """构造测试用 `WorkspaceConfig`。

    Args:
        workspace_dir: 工作区根目录。

    Returns:
        可供 `load_running_config` 调用的路径配置对象。

    Raises:
        OSError: 目录创建失败时抛出。
    """

    filings_dir = workspace_dir / "portfolio" / "AAPL" / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    output_dir = workspace_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return WorkspaceConfig(
        workspace_dir=workspace_dir,
        has_local_filings=True,
        output_dir=output_dir,
        config_loader=ConfigLoader(ConfigFileResolver(workspace_dir / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(workspace_dir / "config")),
        ticker="AAPL",
    )


@pytest.mark.unit
def test_load_running_config_reads_fins_tool_limits_from_run_json(tmp_path: Path) -> None:
    """验证 `fins_tool_limits` 可从 `run.json` 读取并合并默认值。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = config_dir / "run.json"
    run_config_path.write_text(
        json.dumps(
            {
                "fins_tool_limits": {
                    "processor_cache_max_entries": 64,
                    "list_documents_max_items": 999,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)
    fins_tool_limits = _as_fins_tool_limits(loaded.fins_tool_limits)

    assert fins_tool_limits.processor_cache_max_entries == 64
    assert fins_tool_limits.list_documents_max_items == 999
    # 复杂逻辑说明：未在 run.json 中显式声明的字段应继续使用默认值。
    assert fins_tool_limits.get_table_max_items == 800
    assert fins_tool_limits.read_section_max_chars == 80000


@pytest.mark.unit
def test_load_running_config_uses_default_fins_tool_limits_when_missing(tmp_path: Path) -> None:
    """验证未配置 `fins_tool_limits` 时会回退到默认值。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text("{}", encoding="utf-8")

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)
    fins_tool_limits = _as_fins_tool_limits(loaded.fins_tool_limits)

    assert fins_tool_limits.processor_cache_max_entries == 128
    assert fins_tool_limits.get_page_content_max_chars == 80000
    assert fins_tool_limits.read_section_max_chars == 80000


@pytest.mark.unit
def test_load_running_config_reads_tool_timeout_seconds_from_run_json(tmp_path: Path) -> None:
    """验证 `tool_timeout_seconds` 可从 `run.json` 读取。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = config_dir / "run.json"
    run_config_path.write_text(
        json.dumps(
            {
                "runner_running_config": {
                    "tool_timeout_seconds": 120.0,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)
    runner_running_config = _as_openai_runner_config(loaded.runner_running_config)

    assert runner_running_config.tool_timeout_seconds == 120.0


@pytest.mark.unit
def test_load_running_config_uses_default_web_tools_provider_when_missing(tmp_path: Path) -> None:
    """验证未配置 `web_tools_config` 时默认 provider 为 `auto`。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text("{}", encoding="utf-8")

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)
    web_tools_config = _as_web_tools_config(loaded.web_tools_config)

    assert web_tools_config.provider == "auto"
    assert web_tools_config.allow_private_network_url is False
    assert web_tools_config.playwright_channel == "chrome"
    assert web_tools_config.playwright_storage_state_dir == str(
        (tmp_path / "output" / "web_diagnostics" / "storage_states").resolve()
    )


@pytest.mark.unit
def test_load_running_config_reads_allow_private_network_url(tmp_path: Path) -> None:
    """验证 `run.json.web_tools_config.allow_private_network_url` 会被正确加载。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text(
        json.dumps({"web_tools_config": {"allow_private_network_url": True}}, ensure_ascii=False),
        encoding="utf-8",
    )

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)
    web_tools_config = _as_web_tools_config(loaded.web_tools_config)

    assert web_tools_config.allow_private_network_url is True


@pytest.mark.unit
def test_load_running_config_cli_overrides_web_tools_provider(tmp_path: Path) -> None:
    """验证 CLI 参数 `--web-provider` 可覆盖 `run.json`。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text(
        json.dumps({"web_tools_config": {"provider": "duckduckgo"}}, ensure_ascii=False),
        encoding="utf-8",
    )

    args = _build_args()
    args.web_provider = "serper"

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(args, workspace_config)
    web_tools_config = _as_web_tools_config(loaded.web_tools_config)

    assert web_tools_config.provider == "serper"


@pytest.mark.unit
def test_load_running_config_reads_playwright_channel(tmp_path: Path) -> None:
    """验证 `run.json.web_tools_config.playwright_channel` 会被正确加载。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text(
        json.dumps({"web_tools_config": {"playwright_channel": ""}}, ensure_ascii=False),
        encoding="utf-8",
    )

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)
    web_tools_config = _as_web_tools_config(loaded.web_tools_config)

    assert web_tools_config.playwright_channel == ""


@pytest.mark.unit
def test_load_running_config_resolves_playwright_storage_state_dir(tmp_path: Path) -> None:
    """验证 `run.json.web_tools_config.playwright_storage_state_dir` 会解析为工作区绝对路径。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text(
        json.dumps(
            {"web_tools_config": {"playwright_storage_state_dir": "states"}},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)
    web_tools_config = _as_web_tools_config(loaded.web_tools_config)

    assert web_tools_config.playwright_storage_state_dir == str((tmp_path / "states").resolve())


@pytest.mark.unit
def test_load_running_config_uses_default_tool_timeout_seconds_when_missing(tmp_path: Path) -> None:
    """验证未配置 `tool_timeout_seconds` 时会使用默认值 90.0。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text("{}", encoding="utf-8")

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)
    runner_running_config = _as_openai_runner_config(loaded.runner_running_config)

    assert runner_running_config.tool_timeout_seconds == 90.0


@pytest.mark.unit
def test_load_running_config_uses_duplicate_tool_defaults_when_missing(tmp_path: Path) -> None:
    """验证未配置重复调用策略时会使用默认阈值与默认提示词。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text("{}", encoding="utf-8")

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)
    duplicate_tool_hint_prompt = loaded.agent_running_config.duplicate_tool_hint_prompt

    assert loaded.agent_running_config.max_duplicate_tool_calls == 2
    assert duplicate_tool_hint_prompt is not None
    assert "{{tool_name}}" in duplicate_tool_hint_prompt


@pytest.mark.unit
def test_load_running_config_uses_tool_trace_defaults_when_missing(tmp_path: Path) -> None:
    """验证未配置追踪时使用默认关闭与默认目录。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text("{}", encoding="utf-8")

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)

    assert loaded.tool_trace_config.enabled is False
    assert loaded.tool_trace_config.output_dir == (tmp_path / "output" / "tool_call_traces").resolve()


@pytest.mark.unit
def test_load_running_config_reads_tool_trace_config_from_run_json(tmp_path: Path) -> None:
    """验证 `tool_trace_config` 可从 `run.json` 读取。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = config_dir / "run.json"
    run_config_path.write_text(
        json.dumps(
            {
                "tool_trace_config": {
                    "enabled": True,
                    "output_dir": "output/custom_trace",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(_build_args(), workspace_config)

    assert loaded.tool_trace_config.enabled is True
    assert loaded.tool_trace_config.output_dir == (tmp_path / "output" / "custom_trace").resolve()


@pytest.mark.unit
def test_load_running_config_cli_overrides_tool_trace_config(tmp_path: Path) -> None:
    """验证 CLI 参数可覆盖 `tool_trace_config`。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = config_dir / "run.json"
    run_config_path.write_text(
        json.dumps(
            {
                "tool_trace_config": {
                    "enabled": False,
                    "output_dir": "output/from_run_json",
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    args = _build_args()
    args.enable_tool_trace = True
    cli_dir = (tmp_path / "cli_trace_output").resolve()
    args.tool_trace_dir = str(cli_dir)

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(args, workspace_config)

    assert loaded.tool_trace_config.enabled is True
    assert loaded.tool_trace_config.output_dir == cli_dir


@pytest.mark.unit
def test_parse_arguments_supports_tool_trace_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证命令行参数解析支持工具追踪开关。

    Args:
        monkeypatch: pytest monkeypatch 对象。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "interactive",
            "--base",
            "./workspace",
            "--enable-tool-trace",
            "--tool-trace-dir",
            "output/custom",
            "--model-name",
            "mimo-v2-flash",
            "--temperature",
            "0.2",
        ],
    )
    parsed = parse_arguments()
    assert parsed.enable_tool_trace is True
    assert parsed.tool_trace_dir == "output/custom"
    assert getattr(parsed, "ticker", None) is None
    assert parsed.model_name == "mimo-v2-flash"
    assert parsed.temperature == 0.2
    assert parsed.thinking is False


@pytest.mark.unit
def test_parse_arguments_supports_write_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证命令行解析支持写作模式参数。

    Args:
        monkeypatch: pytest monkeypatch 对象。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "write",
            "--base",
            "./workspace",
            "--ticker",
            "aapl",
            "--template",
            "./定性分析模板.md",
            "--output",
            "./workspace/draft",
            "--audit-model-name",
            "deepseek-thinking",
            "--write-max-retries",
            "3",
            "--web-provider",
            "serper",
            "--no-resume",
            "--temperature",
            "0.4",
        ],
    )
    parsed = parse_arguments()

    assert parsed.command == "write"
    assert parsed.template.endswith("定性分析模板.md")
    assert parsed.output == "./workspace/draft"
    assert parsed.audit_model_name == "deepseek-thinking"
    assert parsed.write_max_retries == 3
    assert parsed.web_provider == "serper"
    assert parsed.resume is False
    assert parsed.model_name is None
    assert parsed.temperature == 0.4


@pytest.mark.unit
def test_parse_arguments_supports_write_summary_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证写作命令支持 `--summary` 退化入口。

    Args:
        monkeypatch: pytest monkeypatch 对象。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "write",
            "--ticker",
            "aapl",
            "--summary",
            "--output",
            "./workspace/draft/custom",
        ],
    )

    parsed = parse_arguments()

    assert parsed.command == "write"
    assert parsed.ticker == "aapl"
    assert parsed.summary is True
    assert parsed.output == "./workspace/draft/custom"


@pytest.mark.unit
def test_parse_arguments_rejects_removed_summary_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证已移除的 `summary` 子命令会被直接拒绝。

    Args:
        monkeypatch: pytest monkeypatch 对象。

    Returns:
        无。

    Raises:
        SystemExit: 参数解析失败时抛出。
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "summary",
            "--ticker",
            "aapl",
        ],
    )

    with pytest.raises(SystemExit):
        parse_arguments()


@pytest.mark.unit
def test_parse_arguments_supports_prompt_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证命令行解析支持 prompt 子命令。

    Args:
        monkeypatch: pytest monkeypatch 对象。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "prompt",
            "请总结最新财报风险",
            "--ticker",
            "aapl",
            "--model-name",
            "mimo-v2-flash-thinking",
            "--thinking",
            "--temperature",
            "0.1",
        ],
    )

    parsed = parse_arguments()

    assert parsed.command == "prompt"
    assert parsed.prompt == "请总结最新财报风险"
    assert parsed.ticker == "aapl"
    assert parsed.model_name == "mimo-v2-flash-thinking"
    assert parsed.temperature == 0.1
    assert parsed.thinking is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("argv", "ticker"),
    [
        (["cli.py", "download", "--ticker", "BABA,9988", "--infer"], "BABA,9988"),
        (
            [
                "cli.py",
                "upload_filing",
                "--ticker",
                "0300",
                "--infer",
                "--fiscal-year",
                "2025",
                "--fiscal-period",
                "FY",
            ],
            "0300",
        ),
        (
            [
                "cli.py",
                "upload_material",
                "--ticker",
                "AAPL,APC",
                "--infer",
                "--action",
                "create",
                "--forms",
                "MATERIAL_OTHER",
                "--material-name",
                "deck",
            ],
            "AAPL,APC",
        ),
        (
            [
                "cli.py",
                "upload_filings_from",
                "--ticker",
                "BABA,9988",
                "--infer",
                "--from",
                "./workspace/source",
            ],
            "BABA,9988",
        ),
    ],
)
def test_parse_arguments_supports_fins_infer_flag(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    ticker: str,
) -> None:
    """验证顶层 CLI 为 fins 命令暴露 `--infer`。"""

    monkeypatch.setattr(sys, "argv", argv)

    parsed = parse_arguments()

    assert parsed.infer is True
    assert parsed.ticker == ticker


@pytest.mark.unit
def test_parse_arguments_supports_no_thinking_for_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证交互模式支持显式关闭 thinking 回显。

    Args:
        monkeypatch: pytest monkeypatch 对象。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "interactive",
            "--thinking",
            "--no-thinking",
        ],
    )

    parsed = parse_arguments()

    assert parsed.command == "interactive"
    assert getattr(parsed, "ticker", None) is None
    assert parsed.thinking is False
    assert parsed.model_name is None


@pytest.mark.unit
def test_resolve_interactive_session_id_reuses_saved_interactive_key(tmp_path: Path) -> None:
    """验证 interactive 会话解析会复用状态文件中的 interactive_key。"""

    store = FileInteractiveStateStore(tmp_path / ".dayu" / "interactive")
    state = InteractiveSessionState(interactive_key="interactive_key_saved")
    store.save(state)

    session_id = _resolve_interactive_session_id(tmp_path, new_session=False)

    assert session_id == build_interactive_session_id("interactive_key_saved")
    assert store.load() == state


@pytest.mark.unit
def test_resolve_interactive_session_id_rotates_after_new_session(tmp_path: Path) -> None:
    """验证 `--new-session` 会删除旧绑定并生成新的 interactive_key。"""

    store = FileInteractiveStateStore(tmp_path / ".dayu" / "interactive")
    store.save(InteractiveSessionState(interactive_key="interactive_key_old"))

    session_id = _resolve_interactive_session_id(tmp_path, new_session=True)

    reloaded = store.load()
    assert reloaded is not None
    assert reloaded.interactive_key != "interactive_key_old"
    assert session_id == build_interactive_session_id(reloaded.interactive_key)



@pytest.mark.unit
def test_parse_arguments_rejects_removed_processor_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证财报 process CLI 不再接受 `--processor-hint`。

    Args:
        monkeypatch: pytest monkeypatch 对象。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cli.py",
            "process",
            "--ticker",
            "aapl",
            "--processor-hint",
            "bs",
        ],
    )

    with pytest.raises(SystemExit):
        parse_arguments()


@pytest.mark.unit
def test_resolve_tool_trace_output_dir_handles_relative_and_absolute(tmp_path: Path) -> None:
    """验证工具追踪目录解析支持相对路径和绝对路径。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    relative = _resolve_tool_trace_output_dir("output/tool_trace", tmp_path)
    absolute_target = (tmp_path / "absolute_trace").resolve()
    absolute = _resolve_tool_trace_output_dir(str(absolute_target), tmp_path)

    assert relative == (tmp_path / "output" / "tool_trace").resolve()
    assert absolute == absolute_target


@pytest.mark.unit
def test_load_running_config_cli_tool_trace_dir_resolves_from_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 CLI 显式指定 `tool_trace_dir` 时按当前目录解析。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch 对象。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = (tmp_path / "workspace").resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    config_dir = workspace_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    args = _build_args()
    args.enable_tool_trace = True
    args.tool_trace_dir = "./workspace/tmp/ASML"

    workspace_config = _build_workspace_config(workspace_dir)
    loaded = load_running_config(args, workspace_config)

    expected = (workspace_dir / "tmp" / "ASML").resolve()
    assert loaded.tool_trace_config.output_dir == expected


@pytest.mark.unit
def test_setup_model_name_builds_model_config() -> None:
    """验证模型配置构建逻辑。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    args = Namespace(model_name="test-model")
    model_config = setup_model_name(args)
    assert isinstance(model_config, ModelName)
    assert model_config.model_name == "test-model"


@pytest.mark.unit
def test_setup_paths_returns_workspace_config_success(tmp_path: Path) -> None:
    """验证 `setup_paths` 在目录合法时返回完整路径配置。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "config").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "portfolio" / "AAPL" / "filings").mkdir(parents=True, exist_ok=True)

    args = Namespace(base=str(workspace_dir), ticker="aapl")
    paths_config = setup_paths(args)

    assert paths_config.workspace_dir == workspace_dir.resolve()
    assert paths_config.ticker == "AAPL"
    assert paths_config.has_local_filings is True
    assert paths_config.output_dir == (workspace_dir / "output")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("command_name", "ticker"),
    [
        ("download", "AAPL"),
        ("upload_filing", "0300"),
        ("upload_filings_from", "3606"),
        ("upload_material", "AAPL"),
    ],
)
def test_setup_paths_allows_missing_filings_dir_for_ingest_commands(
    tmp_path: Path,
    command_name: str,
    ticker: str,
) -> None:
    """验证创建或导入源文档的命令不要求预先存在 filings 目录。

    Args:
        tmp_path: pytest 临时目录。
        command_name: 命令名称。
        ticker: 股票代码。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    args = Namespace(base=str(workspace_dir), ticker=ticker, command=command_name)
    paths_config = setup_paths(args)

    assert paths_config.workspace_dir == workspace_dir.resolve()
    assert paths_config.ticker == ticker
    assert paths_config.has_local_filings is False
    assert paths_config.output_dir == (workspace_dir / "output")
    assert not (workspace_dir / "portfolio").exists()


@pytest.mark.unit
@pytest.mark.parametrize("command_name", ["interactive", "prompt"])
def test_setup_paths_warns_and_allows_missing_filings_dir_for_agent_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command_name: str,
) -> None:
    """验证 agent 命令在缺少 `filings` 目录时仅告警并继续。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch 对象。
        command_name: 命令名称。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    collector = _CallCollector()
    monkeypatch.setattr("dayu.cli.dependency_setup.Log.warning", collector.capture_warn)

    args = Namespace(base=str(workspace_dir), ticker="aapl", command=command_name)
    paths_config = setup_paths(args)

    assert paths_config.workspace_dir == workspace_dir.resolve()
    assert paths_config.ticker == "AAPL"
    assert paths_config.has_local_filings is False
    assert collector.warn_logs == ["财报目录不存在，将按无本地财报继续: ticker=AAPL"]
    assert not (workspace_dir / "portfolio").exists()


@pytest.mark.unit
@pytest.mark.parametrize("command_name", ["interactive", "prompt"])
def test_setup_paths_keeps_existing_filings_dir_for_agent_commands(
    tmp_path: Path,
    command_name: str,
) -> None:
    """验证 agent 命令在目录存在时仍会挂载本地 `filings` 目录。

    Args:
        tmp_path: pytest 临时目录。
        command_name: 命令名称。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    filings_dir = workspace_dir / "portfolio" / "AAPL" / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)

    args = Namespace(base=str(workspace_dir), ticker="aapl", command=command_name)
    paths_config = setup_paths(args)

    assert paths_config.workspace_dir == workspace_dir.resolve()
    assert paths_config.ticker == "AAPL"
    assert paths_config.has_local_filings is True


@pytest.mark.unit
def test_has_local_filing_storage_root_keeps_read_only_probe_side_effect_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证只读 filings 探测不会触发 storage recovery。"""

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "portfolio" / "AAPL" / "filings").mkdir(parents=True, exist_ok=True)
    (workspace_dir / ".dayu").mkdir(parents=True, exist_ok=True)

    def _fail_ensure_batch_recovery(_self: object) -> tuple[str, ...]:
        """若只读探测触发 recovery，则立即让测试失败。"""

        raise AssertionError("create_directories=False 的只读探测不应触发 recovery")

    monkeypatch.setattr(
        "dayu.fins.storage._fs_repository_factory.FsStorageCore.ensure_batch_recovery",
        _fail_ensure_batch_recovery,
    )

    assert _has_local_filing_storage_root(workspace_dir, "aapl") is True


@pytest.mark.unit
def test_setup_write_config_uses_workspace_draft_ticker_by_default(tmp_path: Path) -> None:
    """验证写作模式默认输出目录为 `workspace/draft/{ticker}`。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "config").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "portfolio" / "AAPL" / "filings").mkdir(parents=True, exist_ok=True)
    template_path = tmp_path / "template.md"
    template_path.write_text("## 投资要点概览\n\n## 来源清单\n", encoding="utf-8")

    paths_config = WorkspaceConfig(
        workspace_dir=workspace_dir,
        output_dir=workspace_dir / "output",
        config_loader=ConfigLoader(ConfigFileResolver(workspace_dir / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(workspace_dir / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    args = Namespace(
        command="write",
        output=None,
        template=str(template_path),
        write_max_retries=2,
        resume=True,
        web_provider="auto",
        audit_model_name="deepseek-thinking",
        fast=True,
        force=True,
        infer=True,
    )

    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    write_config = setup_write_config(args, paths_config, running_config)

    assert write_config.web_provider == "auto"
    assert write_config.output_dir == (workspace_dir / "draft" / "AAPL").resolve()
    assert write_config.audit_model_override_name == "deepseek-thinking"
    assert write_config.fast is True
    assert write_config.force is True
    assert write_config.infer is True


@pytest.mark.unit
def test_setup_write_config_reads_web_provider_from_running_config(tmp_path: Path) -> None:
    """验证写作配置会继承 `RunningConfig.web_tools_config.provider`。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "config").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "portfolio" / "AAPL" / "filings").mkdir(parents=True, exist_ok=True)
    template_path = tmp_path / "template.md"
    template_path.write_text("## 投资要点概览\n\n## 来源清单\n", encoding="utf-8")

    paths_config = WorkspaceConfig(
        workspace_dir=workspace_dir,
        output_dir=workspace_dir / "output",
        config_loader=ConfigLoader(ConfigFileResolver(workspace_dir / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(workspace_dir / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    args = Namespace(
        command="write",
        output=None,
        template=str(template_path),
        write_max_retries=2,
        resume=True,
        web_provider=None,
        audit_model_name="deepseek-thinking",
    )

    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="duckduckgo"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    write_config = setup_write_config(args, paths_config, running_config)

    assert write_config.web_provider == "duckduckgo"


@pytest.mark.unit
def test_setup_write_config_keeps_empty_audit_override_when_cli_not_provided(tmp_path: Path) -> None:
    """验证未显式传入审计模型时不写入 CLI 覆盖。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "config").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "portfolio" / "AAPL" / "filings").mkdir(parents=True, exist_ok=True)
    template_path = tmp_path / "template.md"
    template_path.write_text("## 投资要点概览\n\n## 来源清单\n", encoding="utf-8")
    paths_config = WorkspaceConfig(
        workspace_dir=workspace_dir,
        output_dir=workspace_dir / "output",
        config_loader=ConfigLoader(ConfigFileResolver(workspace_dir / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(workspace_dir / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    args = Namespace(
        command="write",
        output=None,
        template=str(template_path),
        write_max_retries=2,
        resume=True,
        web_provider=None,
        audit_model_name=None,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )

    write_config = setup_write_config(args, paths_config, running_config)

    assert write_config.audit_model_override_name == ""


@pytest.mark.unit
def test_setup_write_config_raises_system_exit_when_template_missing(tmp_path: Path) -> None:
    """验证缺失模板文件时抛出 `SystemExit(2)`。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "config").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "portfolio" / "AAPL" / "filings").mkdir(parents=True, exist_ok=True)
    missing_template_path = tmp_path / "missing-template.md"
    paths_config = WorkspaceConfig(
        workspace_dir=workspace_dir,
        output_dir=workspace_dir / "output",
        config_loader=ConfigLoader(ConfigFileResolver(workspace_dir / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(workspace_dir / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    args = Namespace(
        command="write",
        output=None,
        template=str(missing_template_path),
        write_max_retries=2,
        resume=True,
        web_provider=None,
        audit_model_name=None,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )

    with pytest.raises(SystemExit) as exc_info:
        setup_write_config(args, paths_config, running_config)

    assert exc_info.value.code == 2


@pytest.mark.unit
def test_setup_write_config_raises_system_exit_when_retry_count_is_negative(tmp_path: Path) -> None:
    """验证负数重试次数时抛出 `SystemExit(2)`。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "config").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "portfolio" / "AAPL" / "filings").mkdir(parents=True, exist_ok=True)
    template_path = tmp_path / "template.md"
    template_path.write_text("## 投资要点概览\n\n## 来源清单\n", encoding="utf-8")
    paths_config = WorkspaceConfig(
        workspace_dir=workspace_dir,
        output_dir=workspace_dir / "output",
        config_loader=ConfigLoader(ConfigFileResolver(workspace_dir / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(workspace_dir / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    args = Namespace(
        command="write",
        output=None,
        template=str(template_path),
        write_max_retries=-1,
        resume=True,
        web_provider=None,
        audit_model_name=None,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )

    with pytest.raises(SystemExit) as exc_info:
        setup_write_config(args, paths_config, running_config)

    assert exc_info.value.code == 2


@pytest.mark.unit
def test_setup_write_config_prefers_workspace_template_when_cli_not_provided(tmp_path: Path) -> None:
    """验证未传 `--template` 时优先使用工作区模板。"""

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "config").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "assets").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "assets" / "定性分析模板.md").write_text("# workspace", encoding="utf-8")
    (workspace_dir / "portfolio" / "AAPL" / "filings").mkdir(parents=True, exist_ok=True)
    paths_config = WorkspaceConfig(
        workspace_dir=workspace_dir,
        output_dir=workspace_dir / "output",
        config_loader=ConfigLoader(ConfigFileResolver(workspace_dir / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(workspace_dir / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    args = Namespace(
        command="write",
        output=None,
        template=None,
        write_max_retries=2,
        resume=True,
        web_provider=None,
        audit_model_name=None,
        chapter=None,
        fast=False,
        force=False,
        infer=False,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )

    write_config = setup_write_config(args, paths_config, running_config)

    assert write_config.template_path == (workspace_dir / "assets" / "定性分析模板.md")


@pytest.mark.unit
def test_setup_write_config_falls_back_to_package_template_when_workspace_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证工作区模板缺失时回退到包内默认模板。"""

    workspace_dir = tmp_path / "workspace"
    (workspace_dir / "config").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "portfolio" / "AAPL" / "filings").mkdir(parents=True, exist_ok=True)
    package_assets_dir = tmp_path / "package_assets"
    package_assets_dir.mkdir(parents=True, exist_ok=True)
    (package_assets_dir / "定性分析模板.md").write_text("# package", encoding="utf-8")
    monkeypatch.setattr(
        "dayu.cli.dependency_setup.resolve_package_assets_path",
        lambda: package_assets_dir,
    )

    paths_config = WorkspaceConfig(
        workspace_dir=workspace_dir,
        output_dir=workspace_dir / "output",
        config_loader=ConfigLoader(ConfigFileResolver(workspace_dir / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(workspace_dir / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    args = Namespace(
        command="write",
        output=None,
        template=None,
        write_max_retries=2,
        resume=True,
        web_provider=None,
        audit_model_name=None,
        chapter=None,
        fast=False,
        force=False,
        infer=False,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )

    write_config = setup_write_config(args, paths_config, running_config)

    assert write_config.template_path == (package_assets_dir / "定性分析模板.md")


@pytest.mark.unit
def test_resolve_write_output_dir_uses_ticker_subdir_by_default(tmp_path: Path) -> None:
    """验证写作输出目录默认落到 `draft/{ticker}`。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    resolved = _resolve_write_output_dir(
        workspace_dir=tmp_path,
        ticker="AAPL",
        raw_output=None,
    )

    assert resolved == (tmp_path / "draft" / "AAPL").resolve()


@pytest.mark.unit
def test_load_running_config_applies_all_cli_overrides(tmp_path: Path) -> None:
    """验证 CLI 覆盖字段全部生效。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text("{}", encoding="utf-8")

    args = _build_args()
    args.debug_sse = True
    args.debug_tool_delta = True
    args.debug_sse_sample_rate = 0.5
    args.debug_sse_throttle_sec = 2.0
    args.tool_timeout_seconds = 9.5
    args.max_iterations = 3
    args.fallback_mode = "raise_error"
    args.fallback_prompt = "fallback"
    args.max_consecutive_failed_tool_batches = 4
    args.max_duplicate_tool_calls = 5
    args.duplicate_tool_hint_prompt = "hint"

    workspace_config = _build_workspace_config(tmp_path)
    loaded = load_running_config(args, workspace_config)
    runner_running_config = _as_openai_runner_config(loaded.runner_running_config)

    assert runner_running_config.debug_sse is True
    assert runner_running_config.debug_tool_delta is True
    assert runner_running_config.debug_sse_sample_rate == 0.5
    assert runner_running_config.debug_sse_throttle_sec == 2.0
    assert runner_running_config.tool_timeout_seconds == 9.5
    assert loaded.agent_running_config.max_iterations == 3
    assert loaded.agent_running_config.fallback_mode == "raise_error"
    assert loaded.agent_running_config.fallback_prompt == "fallback"
    assert loaded.agent_running_config.max_consecutive_failed_tool_batches == 4
    assert loaded.agent_running_config.max_duplicate_tool_calls == 5
    assert loaded.agent_running_config.duplicate_tool_hint_prompt == "hint"


@pytest.mark.unit
def test_setup_loglevel_selects_expected_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 `setup_loglevel` 的优先级分支。

    Args:
        monkeypatch: pytest monkeypatch 对象。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    collector = _CallCollector()
    monkeypatch.setattr("dayu.cli.dependency_setup.Log.set_level", collector.capture_level)

    setup_loglevel(Namespace(log_level="info", debug=False, verbose=False, info=False, quiet=False))
    setup_loglevel(Namespace(log_level=None, debug=True, verbose=False, info=False, quiet=False))
    setup_loglevel(Namespace(log_level=None, debug=False, verbose=True, info=False, quiet=False))
    setup_loglevel(Namespace(log_level=None, debug=False, verbose=False, info=True, quiet=False))
    setup_loglevel(Namespace(log_level=None, debug=False, verbose=False, info=False, quiet=True))
    setup_loglevel(Namespace(log_level=None, debug=False, verbose=False, info=False, quiet=False))

    assert collector.levels == ["INFO", "DEBUG", "VERBOSE", "INFO", "ERROR", "INFO"]


@pytest.mark.unit
def test_main_interactive_path_returns_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `main` 在 interactive 模式会调用交互入口并返回 0。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    filings_dir = tmp_path / "portfolio" / "AAPL" / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    (filings_dir / "sample.txt").write_text("x", encoding="utf-8")
    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(
        command="interactive",
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        thinking=True,
        model_name="deepseek-thinking",
        new_session=False,
    )

    collector = _CallCollector()
    interactive_kwargs: dict[str, object] = {}

    def _capture_interactive(*_args: object, **kwargs: object) -> None:
        """记录 interactive 调用参数。

        Args:
            *_args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            无。

        Raises:
            无。
        """

        collector.mark_interactive()
        interactive_kwargs.update(kwargs)

    monkeypatch.setattr("dayu.cli.commands.interactive.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.interactive.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr(
        "dayu.cli.commands.interactive._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    fake_dependencies = _FakeCliHostDependencies(
        running_config=running_config,
        scene_model_names={"interactive": "scene-interactive-model"},
    )
    monkeypatch.setattr(
        "dayu.cli.commands.interactive._prepare_cli_host_dependencies",
        lambda **_kwargs: fake_dependencies.as_tuple(),
    )
    monkeypatch.setattr("dayu.cli.commands.interactive._build_chat_service", lambda **_kwargs: object())
    monkeypatch.setattr("dayu.cli.commands.interactive.Log.info", collector.capture_info)
    monkeypatch.setattr("dayu.cli.commands.interactive.interactive", _capture_interactive)

    assert run_interactive_command(args) == 0
    assert collector.interactive_calls == 1
    state = FileInteractiveStateStore(tmp_path / ".dayu" / "interactive").load()
    assert state is not None
    interactive_execution_options = cast(Any, interactive_kwargs["execution_options"])
    assert interactive_kwargs["session_id"] == build_interactive_session_id(state.interactive_key)
    assert interactive_kwargs["show_thinking"] is True
    assert interactive_execution_options.model_name == "deepseek-thinking"
    assert any('使用模型: {"name": "scene-interactive-model", "temperature": 0.0}' in item for item in collector.info_logs)


@pytest.mark.unit
def test_main_interactive_path_rejects_second_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """interactive 命令遇到单实例锁冲突时应显式失败。"""

    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker="AAPL",
        has_local_filings=False,
    )
    args = Namespace(
        command="interactive",
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        thinking=False,
        model_name="deepseek-thinking",
        new_session=False,
    )
    error_logs: list[str] = []

    monkeypatch.setattr("dayu.cli.commands.interactive.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.interactive.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr(
        "dayu.cli.commands.interactive._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    monkeypatch.setattr(
        "dayu.cli.commands.interactive._prepare_cli_host_dependencies",
        lambda **_kwargs: _FakeCliHostDependencies(
            running_config=RunningConfig(
                runner_running_config=AsyncOpenAIRunnerRunningConfig(),
                agent_running_config=AgentRunningConfig(),
                doc_tool_limits=DocToolLimits(),
                fins_tool_limits=FinsToolLimits(),
                web_tools_config=WebToolsConfig(provider="auto"),
                tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
            ),
            scene_model_names={"interactive": "scene-interactive-model"},
        ).as_tuple(),
    )
    monkeypatch.setattr("dayu.cli.commands.interactive._build_chat_service", lambda **_kwargs: object())
    monkeypatch.setattr(
        "dayu.cli.commands.interactive.StateDirSingleInstanceLock.acquire",
        lambda self: (_ for _ in ()).throw(RuntimeError("同一个 state_dir 已有运行中的 interactive 单实例锁")),
    )
    monkeypatch.setattr("dayu.cli.commands.interactive.Log.error", lambda message, **_kwargs: error_logs.append(str(message)))
    monkeypatch.setattr("dayu.cli.commands.interactive.interactive", lambda *_args, **_kwargs: pytest.fail("不应进入 interactive"))

    assert run_interactive_command(args) == 1
    assert any("interactive 单实例锁" in message for message in error_logs)


@pytest.mark.unit
def test_main_prompt_path_returns_prompt_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `main` 在 prompt 模式会调用单次 prompt 入口并返回其退出码。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    filings_dir = tmp_path / "portfolio" / "AAPL" / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    (filings_dir / "sample.txt").write_text("x", encoding="utf-8")
    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(
        command="prompt",
        prompt="请总结风险",
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        thinking=False,
        model_name="deepseek-thinking",
    )

    prompt_kwargs: dict[str, object] = {}

    def _capture_prompt(*_args: object, **kwargs: object) -> int:
        """记录 prompt 调用参数并返回固定退出码。

        Args:
            *_args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            固定退出码。

        Raises:
            无。
        """

        prompt_kwargs.update(kwargs)
        return 7

    monkeypatch.setattr("dayu.cli.commands.prompt.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr(
        "dayu.cli.commands.prompt._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    fake_dependencies = _FakeCliHostDependencies(
        running_config=running_config,
        scene_model_names={"prompt": "scene-prompt-model"},
    )
    monkeypatch.setattr(
        "dayu.cli.commands.prompt._prepare_cli_host_dependencies",
        lambda **_kwargs: fake_dependencies.as_tuple(),
    )
    monkeypatch.setattr("dayu.cli.commands.prompt._build_prompt_service", lambda **_kwargs: object())
    monkeypatch.setattr("dayu.cli.commands.prompt.prompt_command", _capture_prompt)

    assert run_prompt_command(args) == 7
    prompt_execution_options = cast(Any, prompt_kwargs["execution_options"])
    assert prompt_kwargs["ticker"] == "AAPL"
    assert prompt_kwargs["show_thinking"] is False
    assert prompt_execution_options.model_name == "deepseek-thinking"


@pytest.mark.unit
def test_prepare_cli_host_dependencies_runs_unified_startup_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """CLI Host 依赖装配应委托 Service 共享启动 API。"""

    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_root=tmp_path / "config",
    )
    fake_workspace = cast(WorkspaceResources, object())
    fake_default_execution_options = cast(ResolvedExecutionOptions, object())
    fake_scene_preparer = cast(SceneExecutionAcceptancePreparer, object())
    fake_fins_runtime = cast(DefaultFinsRuntime, object())
    fake_host = cast(Host, object())
    captured_call: dict[str, object] = {}

    monkeypatch.setattr(
        "dayu.cli.dependency_setup.prepare_host_runtime_dependencies",
        lambda **kwargs: (
            captured_call.update(kwargs)
            or PreparedHostRuntimeDependencies(
                workspace=fake_workspace,
                default_execution_options=fake_default_execution_options,
                scene_execution_acceptance_preparer=fake_scene_preparer,
                host=fake_host,
                fins_runtime=fake_fins_runtime,
            )
        ),
    )

    prepared = _prepare_cli_host_dependencies(
        workspace_config=workspace_config,
        execution_options=None,
    )

    assert prepared == (
        fake_workspace,
        fake_default_execution_options,
        fake_scene_preparer,
        fake_host,
        fake_fins_runtime,
    )
    assert captured_call == {
        "workspace_root": workspace_config.workspace_dir,
        "config_root": workspace_config.config_root,
        "execution_options": None,
        "runtime_label": "CLI Host runtime",
        "log_module": "APP.MAIN",
    }


@pytest.mark.unit
def test_main_prompt_path_allows_missing_filings_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `main` 在 prompt 模式下可接受缺失的本地 `filings` 目录。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker="AAPL",
        has_local_filings=False,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(
        command="prompt",
        prompt="请总结风险",
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        thinking=False,
        model_name="deepseek-thinking",
    )

    prompt_kwargs: dict[str, object] = {}

    def _capture_prompt(*_args: object, **kwargs: object) -> int:
        """记录 prompt 调用参数并返回固定退出码。

        Args:
            *_args: 位置参数。
            **kwargs: 关键字参数。

        Returns:
            固定退出码。

        Raises:
            无。
        """

        prompt_kwargs.update(kwargs)
        return 9

    monkeypatch.setattr("dayu.cli.commands.prompt.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr(
        "dayu.cli.commands.prompt._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    fake_dependencies = _FakeCliHostDependencies(
        running_config=running_config,
        scene_model_names={"prompt": "scene-prompt-model"},
    )
    monkeypatch.setattr(
        "dayu.cli.commands.prompt._prepare_cli_host_dependencies",
        lambda **_kwargs: fake_dependencies.as_tuple(),
    )
    monkeypatch.setattr("dayu.cli.commands.prompt._build_prompt_service", lambda **_kwargs: object())
    monkeypatch.setattr("dayu.cli.commands.prompt.prompt_command", _capture_prompt)

    assert run_prompt_command(args) == 9
    assert prompt_kwargs["ticker"] == "AAPL"


@pytest.mark.unit
def test_main_prompt_path_propagates_cli_options_to_mock_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 prompt CLI 显式参数会贯通到 scene preparation 与最终 Agent 输入。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)

    recorder = _PromptAgentE2ERecorder()
    web_tools_recorder = _WebToolsRegistrationRecorder()
    doc_tools_recorder = _DocToolsRegistrationRecorder()
    fins_read_tools_recorder = _FinsReadToolsRegistrationRecorder()
    trace_dir = (tmp_path / "explicit_trace").resolve()
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "prompt",
            "--base",
            str(workspace_dir),
            "--ticker",
            "AAPL",
            "--model-name",
            "gpt-5.4",
            "--temperature",
            "0.35",
            "--tool-timeout-seconds",
            "12",
            "--max-iterations",
            "3",
            "--fallback-mode",
            "raise_error",
            "--fallback-prompt",
            "fallback by cli",
            "--max-consecutive-failed-tool-batches",
            "4",
            "--max-duplicate-tool-calls",
            "5",
            "--duplicate-tool-hint-prompt",
            "dup hint by cli",
            "--web-provider",
            "duckduckgo",
            "--enable-tool-trace",
            "--tool-trace-dir",
            str(trace_dir),
            "--doc-limits-json",
            '{"list_files_max": 123, "read_file_max_chars": 4567}',
            "--fins-limits-json",
            '{"list_documents_max_items": 77, "read_section_max_chars": 2222, "processor_cache_max_entries": 64}',
            "--thinking",
            "请总结最新风险",
        ],
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", _MockPromptAgentBuilder(recorder))
    monkeypatch.setattr("dayu.engine.toolset_registrars.register_web_toolset", web_tools_recorder)
    monkeypatch.setattr("dayu.engine.toolset_registrars.register_doc_toolset", doc_tools_recorder)
    monkeypatch.setattr("dayu.fins.toolset_registrars.register_fins_read_toolset", fins_read_tools_recorder)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.error", lambda *_args, **_kwargs: None)

    assert main() == 0
    assert recorder.agent_create_args is not None
    assert recorder.messages is not None
    assert recorder.tool_executor is not None
    assert recorder.tool_trace_recorder_factory is not None
    assert recorder.session_id
    assert recorder.run_id
    assert recorder.agent_create_args.model_name == "gpt-5.4"
    assert recorder.agent_create_args.temperature == pytest.approx(0.35)
    assert recorder.agent_create_args.max_turns == 3
    assert recorder.agent_create_args.runner_running_config["tool_timeout_seconds"] == pytest.approx(12.0)
    assert recorder.agent_create_args.agent_running_config["fallback_mode"] == "raise_error"
    assert recorder.agent_create_args.agent_running_config["fallback_prompt"] == "fallback by cli"
    assert recorder.agent_create_args.agent_running_config["max_consecutive_failed_tool_batches"] == 4
    assert recorder.agent_create_args.agent_running_config["max_duplicate_tool_calls"] == 5
    assert recorder.agent_create_args.agent_running_config["duplicate_tool_hint_prompt"] == "dup hint by cli"
    assert recorder.trace_identity is not None
    assert recorder.trace_identity["agent_name"] == "prompt_agent"
    assert recorder.trace_identity["agent_kind"] == "scene_agent"
    assert recorder.trace_identity["scene_name"] == "prompt"
    assert recorder.trace_identity["model_name"] == "gpt-5.4"
    assert recorder.trace_identity["session_id"] == recorder.session_id
    assert recorder.tool_trace_recorder_factory is not None
    assert recorder.tool_trace_recorder_factory._store._output_dir == trace_dir
    assert recorder.tool_trace_recorder_factory._store._partition_by_session is True
    assert recorder.messages[-1] == {"role": "user", "content": "请总结最新风险"}
    assert "search_web" in recorder.tool_executor.tools
    assert "fetch_web_page" in recorder.tool_executor.tools
    assert doc_tools_recorder.calls == []
    assert fins_read_tools_recorder.calls == [
        {
            "processor_cache_max_entries": 64,
            "list_documents_max_items": 77,
            "get_document_sections_max_items": 1200,
            "search_document_max_items": 20,
            "read_section_max_chars": 2222,
            "get_page_content_max_chars": 80000,
            "timeout_budget": pytest.approx(12.0),
            "has_service": True,
            "has_repository": False,
            "has_processor_registry": False,
        }
    ]
    assert web_tools_recorder.calls == [
        {
            "provider": "duckduckgo",
            "request_timeout_seconds": pytest.approx(12.0),
            "max_search_results": 20,
            "fetch_truncate_chars": 80000,
            "timeout_budget": pytest.approx(12.0),
            "allow_private_network_url": False,
            "playwright_channel": "chrome",
            "playwright_storage_state_dir": str(
                (workspace_dir / "output" / "web_diagnostics" / "storage_states").resolve()
            ),
        }
    ]


@pytest.mark.unit
def test_main_prompt_path_sparse_override_preserves_run_json_base(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 CLI 稀疏 override 只覆盖指定字段，未指定字段保留 run.json 中的非默认值。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    config_dir = workspace_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    # run.json 设置非默认的 limits 值
    (config_dir / "run.json").write_text(
        json.dumps(
            {
                "doc_tool_limits": {
                    "list_files_max": 500,
                    "get_sections_max": 999,
                    "search_files_max_results": 88,
                    "read_file_max_chars": 300000,
                },
                "fins_tool_limits": {
                    "processor_cache_max_entries": 256,
                    "list_documents_max_items": 600,
                    "get_document_sections_max_items": 2400,
                    "read_section_max_chars": 120000,
                },
                "web_tools_config": {
                    "provider": "off",
                    "max_search_results": 50,
                    "fetch_truncate_chars": 99999,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    recorder = _PromptAgentE2ERecorder()
    web_tools_recorder = _WebToolsRegistrationRecorder()
    doc_tools_recorder = _DocToolsRegistrationRecorder()
    fins_read_tools_recorder = _FinsReadToolsRegistrationRecorder()
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    # CLI 只覆盖少数字段
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "prompt",
            "--base",
            str(workspace_dir),
            "--ticker",
            "AAPL",
            "--doc-limits-json",
            '{"list_files_max": 10}',
            "--fins-limits-json",
            '{"read_section_max_chars": 5000}',
            "--web-provider",
            "duckduckgo",
            "请总结最新风险",
        ],
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", _MockPromptAgentBuilder(recorder))
    monkeypatch.setattr("dayu.engine.toolset_registrars.register_web_toolset", web_tools_recorder)
    monkeypatch.setattr("dayu.engine.toolset_registrars.register_doc_toolset", doc_tools_recorder)
    monkeypatch.setattr("dayu.fins.toolset_registrars.register_fins_read_toolset", fins_read_tools_recorder)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.error", lambda *_args, **_kwargs: None)

    assert main() == 0

    # prompt scene 不启用 doc toolset，因此不应调用 doc adapter
    assert doc_tools_recorder.calls == []

    # fins: read_section_max_chars 被覆盖，其余字段保留 run.json 的非默认值
    assert fins_read_tools_recorder.calls[0]["processor_cache_max_entries"] == 256
    assert fins_read_tools_recorder.calls[0]["list_documents_max_items"] == 600
    assert fins_read_tools_recorder.calls[0]["get_document_sections_max_items"] == 2400
    assert fins_read_tools_recorder.calls[0]["read_section_max_chars"] == 5000

    # web: --web-provider 覆盖 provider，其余保留 run.json 的非默认值
    assert web_tools_recorder.calls[0]["provider"] == "duckduckgo"
    assert web_tools_recorder.calls[0]["max_search_results"] == 50
    assert web_tools_recorder.calls[0]["fetch_truncate_chars"] == 99999


@pytest.mark.unit
def test_main_prompt_path_propagates_execution_permissions_to_web_tool_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `run.json.web_tools_config.allow_private_network_url` 会贯通到 web 工具注册。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    config_dir = workspace_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text(
        json.dumps(
            {
                "web_tools_config": {
                    "provider": "duckduckgo",
                    "allow_private_network_url": True,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    recorder = _PromptAgentE2ERecorder()
    web_tools_recorder = _WebToolsRegistrationRecorder()
    doc_tools_recorder = _DocToolsRegistrationRecorder()
    fins_read_tools_recorder = _FinsReadToolsRegistrationRecorder()
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "prompt",
            "--base",
            str(workspace_dir),
            "--ticker",
            "AAPL",
            "请总结最新风险",
        ],
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", _MockPromptAgentBuilder(recorder))
    monkeypatch.setattr("dayu.engine.toolset_registrars.register_web_toolset", web_tools_recorder)
    monkeypatch.setattr("dayu.engine.toolset_registrars.register_doc_toolset", doc_tools_recorder)
    monkeypatch.setattr("dayu.fins.toolset_registrars.register_fins_read_toolset", fins_read_tools_recorder)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.error", lambda *_args, **_kwargs: None)

    assert main() == 0
    assert recorder.messages is not None
    assert recorder.messages[-1] == {"role": "user", "content": "请总结最新风险"}
    assert recorder.tool_executor is not None
    assert "search_web" in recorder.tool_executor.tools
    assert "fetch_web_page" in recorder.tool_executor.tools
    assert web_tools_recorder.calls == [
        {
            "provider": "duckduckgo",
            "request_timeout_seconds": pytest.approx(12.0),
            "max_search_results": 20,
            "fetch_truncate_chars": 80000,
            "timeout_budget": pytest.approx(90.0),
            "allow_private_network_url": True,
            "playwright_channel": "chrome",
            "playwright_storage_state_dir": str(
                (workspace_dir / "output" / "web_diagnostics" / "storage_states").resolve()
            ),
        }
    ]
    assert doc_tools_recorder.calls == []
    assert fins_read_tools_recorder.calls == [
        {
            "processor_cache_max_entries": 128,
            "list_documents_max_items": 300,
            "get_document_sections_max_items": 1200,
            "search_document_max_items": 20,
            "read_section_max_chars": 80000,
            "get_page_content_max_chars": 80000,
            "timeout_budget": pytest.approx(90.0),
            "has_service": True,
            "has_repository": False,
            "has_processor_registry": False,
        }
    ]


@pytest.mark.unit
def test_main_prompt_path_propagates_run_json_defaults_to_host_and_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `run.json` 配置会真实贯通到 Host 工具注册与最终 Agent 参数。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    config_dir = workspace_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "run.json").write_text(
        json.dumps(
            {
                "runner_running_config": {
                    "tool_timeout_seconds": 21.5,
                },
                "agent_running_config": {
                    "max_iterations": 7,
                    "fallback_mode": "raise_error",
                    "fallback_prompt": "fallback by run",
                    "max_consecutive_failed_tool_batches": 6,
                    "max_duplicate_tool_calls": 8,
                    "duplicate_tool_hint_prompt": "dup hint by run",
                },
                "doc_tool_limits": {
                    "list_files_max": 321,
                    "read_file_max_chars": 6543,
                },
                "fins_tool_limits": {
                    "processor_cache_max_entries": 77,
                    "list_documents_max_items": 88,
                    "read_section_max_chars": 9999,
                    "get_page_content_max_chars": 4444,
                },
                "web_tools_config": {
                    "provider": "serper",
                    "request_timeout_seconds": 18.0,
                    "max_search_results": 9,
                    "fetch_truncate_chars": 1234,
                    "allow_private_network_url": True,
                    "playwright_channel": "",
                    "playwright_storage_state_dir": "states",
                },
                "tool_trace_config": {
                    "enabled": True,
                    "output_dir": "output/custom_trace",
                    "partition_by_session": False,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    recorder = _PromptAgentE2ERecorder()
    web_tools_recorder = _WebToolsRegistrationRecorder()
    doc_tools_recorder = _DocToolsRegistrationRecorder()
    fins_read_tools_recorder = _FinsReadToolsRegistrationRecorder()
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "prompt",
            "--base",
            str(workspace_dir),
            "--ticker",
            "AAPL",
            "请总结最新风险",
        ],
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", _MockPromptAgentBuilder(recorder))
    monkeypatch.setattr("dayu.engine.toolset_registrars.register_web_toolset", web_tools_recorder)
    monkeypatch.setattr("dayu.engine.toolset_registrars.register_doc_toolset", doc_tools_recorder)
    monkeypatch.setattr("dayu.fins.toolset_registrars.register_fins_read_toolset", fins_read_tools_recorder)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.error", lambda *_args, **_kwargs: None)

    assert main() == 0
    assert recorder.agent_create_args is not None
    assert recorder.agent_create_args.max_turns == 24
    assert recorder.agent_create_args.runner_running_config["tool_timeout_seconds"] == pytest.approx(90.0)
    assert recorder.agent_create_args.agent_running_config["fallback_mode"] == "raise_error"
    assert recorder.agent_create_args.agent_running_config["fallback_prompt"] == "fallback by run"
    assert recorder.agent_create_args.agent_running_config["max_consecutive_failed_tool_batches"] == 6
    assert recorder.agent_create_args.agent_running_config["max_duplicate_tool_calls"] == 8
    assert recorder.agent_create_args.agent_running_config["duplicate_tool_hint_prompt"] == "dup hint by run"
    assert recorder.tool_trace_recorder_factory is not None
    assert recorder.tool_trace_recorder_factory._store._output_dir == (
        workspace_dir / "output" / "custom_trace"
    ).resolve()
    assert recorder.tool_trace_recorder_factory._store._partition_by_session is False
    assert doc_tools_recorder.calls == []
    assert fins_read_tools_recorder.calls == [
        {
            "processor_cache_max_entries": 77,
            "list_documents_max_items": 88,
            "get_document_sections_max_items": 1200,
            "search_document_max_items": 20,
            "read_section_max_chars": 9999,
            "get_page_content_max_chars": 4444,
            "timeout_budget": pytest.approx(90.0),
            "has_service": True,
            "has_repository": False,
            "has_processor_registry": False,
        }
    ]
    assert web_tools_recorder.calls == [
        {
            "provider": "serper",
            "request_timeout_seconds": pytest.approx(18.0),
            "max_search_results": 9,
            "fetch_truncate_chars": 1234,
            "timeout_budget": pytest.approx(90.0),
            "allow_private_network_url": True,
            "playwright_channel": "",
            "playwright_storage_state_dir": str((workspace_dir / "states").resolve()),
        }
    ]


@pytest.mark.unit
def test_main_prompt_path_propagates_scene_manifest_prompt_assets_and_llm_model_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 scene manifest、prompt assets 与 llm_models 配置会端到端贯通到最终 Agent 输入。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_dir = tmp_path / "workspace"
    config_dir = workspace_dir / "config"
    prompts_dir = config_dir / "prompts"
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    _write_json_file(config_dir / "run.json", {})
    _write_json_file(
        config_dir / "llm_models.json",
        {
            "custom-prompt-model": {
                "runner_type": "openai_compatible",
                "endpoint_url": "https://example.invalid/v1",
                "model": "custom-prompt-model",
                "headers": {},
                "supports_stream": True,
                "supports_tool_calling": True,
                "supports_stream_usage": True,
                "max_context_tokens": 64000,
                "max_output_tokens": 4096,
                "runtime_hints": {
                    "temperature_profiles": {
                        "prompt_e2e": {
                            "temperature": 0.17,
                        }
                    }
                },
            }
        },
    )
    _write_json_file(
        prompts_dir / "manifests" / "prompt.json",
        {
            "scene": "prompt",
            "model": {
                "default_name": "custom-prompt-model",
                "allowed_names": ["custom-prompt-model"],
                "temperature_profile": "prompt_e2e"
            },
            "runtime": {
                "agent": {
                    "max_iterations": 11
                },
                "runner": {
                    "tool_timeout_seconds": 90.0
                }
            },
            "version": "v1",
            "description": "自定义单轮问答场景",
            "extends": [],
            "tool_selection": {
                "mode": "select",
                "tool_tags_any": ["web"],
            },
            "defaults": {
                "missing_fragment_policy": "error",
            },
            "fragments": [
                {
                    "id": "base_agents",
                    "type": "AGENTS",
                    "path": "base/agents.md",
                    "required": True,
                    "order": 100,
                },
                {
                    "id": "base_tools",
                    "type": "TOOLS",
                    "path": "base/tools.md",
                    "required": True,
                    "order": 200,
                },
                {
                    "id": "prompt_scene",
                    "type": "SCENE",
                    "path": "scenes/prompt.md",
                    "required": True,
                    "order": 300,
                },
            ],
            "context_slots": ["fins_default_subject", "base_user"],
        },
    )
    _write_text_file(
        prompts_dir / "scenes" / "prompt.md",
        "# 自定义单轮问答执行契约\n\n- 这是工作区覆盖的 prompt scene 片段。\n- 只回答当前问题。\n",
    )

    recorder = _PromptAgentE2ERecorder()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "prompt",
            "--base",
            str(workspace_dir),
            "--ticker",
            "AAPL",
            "请总结自定义 prompt",
        ],
    )
    monkeypatch.setattr(
        "dayu.services.prompt_service.build_base_user_contribution",
        lambda: "# BASE_USER_SLOT\n当前时间：测试时间。",
    )
    monkeypatch.setattr(
        "dayu.services.prompt_service.build_optional_fins_subject_contribution",
        lambda **_kwargs: "# SUBJECT_SLOT\n你正在分析的是 AAPL。",
    )
    monkeypatch.setattr("dayu.host.executor.build_async_agent", _MockPromptAgentBuilder(recorder))
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("dayu.cli.commands.prompt.Log.error", lambda *_args, **_kwargs: None)

    assert main() == 0
    assert recorder.agent_create_args is not None
    assert recorder.messages is not None
    assert recorder.tool_executor is not None

    system_prompt = _extract_system_prompt(recorder.messages)

    # 复杂逻辑说明：这里同时验证四类真源的端到端落地：
    # 1. llm_models.json 的 temperature_profiles -> accepted_execution_spec -> AgentCreateArgs.temperature
    # 2. scene manifest 的 default_name / temperature_profile / runtime.agent.max_iterations -> 最终 AgentCreateArgs
    # 3. scene manifest 的 tool_selection -> 最终 ToolRegistry
    # 4. prompt assets + context_slots -> 最终 system prompt
    assert recorder.agent_create_args.model_name == "custom-prompt-model"
    assert recorder.agent_create_args.temperature == pytest.approx(0.17)
    assert recorder.agent_create_args.max_turns == 11
    assert recorder.messages[-1] == {"role": "user", "content": "请总结自定义 prompt"}
    assert "这是工作区覆盖的 prompt scene 片段" in system_prompt
    assert "严格输出当前任务要求的唯一结果" in system_prompt
    assert system_prompt.index("# SUBJECT_SLOT") < system_prompt.index("# BASE_USER_SLOT")
    assert "search_web" in recorder.tool_executor.tools
    assert "fetch_web_page" in recorder.tool_executor.tools
    assert "list_documents" not in recorder.tool_executor.tools
    assert "list_files" not in recorder.tool_executor.tools


@pytest.mark.unit
def test_main_non_interactive_path_returns_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `main` 在非 interactive 模式直接返回 0。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker=None,
        has_local_filings=False,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(command=None, log_level=None, debug=False, verbose=False, info=False, quiet=False)

    monkeypatch.setattr("dayu.cli.main.parse_arguments", partial(_return_value, args))
    assert main() == 0


@pytest.mark.unit
def test_main_write_mode_requires_ticker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证写作模式缺少 ticker 时返回参数错误码。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker=None,
        has_local_filings=False,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(
        command="write",
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        output=None,
        template="./定性分析模板.md",
        write_max_retries=2,
        resume=True,
        web_provider="auto",
        audit_model_name="deepseek-thinking",
    )

    monkeypatch.setattr("dayu.cli.commands.write.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.write.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr("dayu.cli.commands.write.setup_model_name", partial(_return_value, model_name))
    monkeypatch.setattr(
        "dayu.cli.commands.write._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    monkeypatch.setattr("dayu.cli.commands.write.Log.error", lambda *args, **kwargs: None)

    assert run_write_command(args) == 2


@pytest.mark.unit
def test_main_write_summary_mode_requires_ticker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `write --summary` 缺少 ticker 时返回参数错误码。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker=None,
        has_local_filings=False,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(
        command="write",
        summary=True,
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        output=None,
        template="./定性分析模板.md",
        write_max_retries=2,
        resume=True,
        web_provider="auto",
        audit_model_name="deepseek-thinking",
    )

    collector = _CallCollector()
    monkeypatch.setattr("dayu.cli.commands.write.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.write.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr("dayu.cli.commands.write.setup_model_name", partial(_return_value, model_name))
    monkeypatch.setattr(
        "dayu.cli.commands.write._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    monkeypatch.setattr("dayu.cli.commands.write.Log.error", collector.capture_error)

    assert run_write_command(args) == 2
    assert any("write --summary 模式要求必须提供 --ticker" in item for item in collector.error_logs)


@pytest.mark.unit
def test_main_write_summary_mode_calls_print_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `write --summary` 会调用写作服务的报告打印入口。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    filings_dir = tmp_path / "portfolio" / "AAPL" / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(
        command="write",
        summary=True,
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        output=None,
        template="./定性分析模板.md",
        write_max_retries=2,
        resume=True,
        web_provider="auto",
        audit_model_name="deepseek-thinking",
    )

    captured_output_dir: dict[str, Path] = {}

    class _FakeWriteService:
        """测试用写作服务。"""

        @staticmethod
        def print_report(output_dir: Path) -> int:
            """记录输出目录并返回固定退出码。

            Args:
                output_dir: 写作输出目录。

            Returns:
                固定退出码。

            Raises:
                无。
            """

            captured_output_dir["value"] = output_dir
            return 6

    monkeypatch.setattr("dayu.cli.commands.write.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.write.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr("dayu.cli.commands.write.setup_model_name", partial(_return_value, model_name))
    monkeypatch.setattr(
        "dayu.cli.commands.write._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    monkeypatch.setattr("dayu.cli.commands.write.WriteService.print_report", _FakeWriteService.print_report)
    monkeypatch.setattr("dayu.cli.commands.write.run_write_pipeline", lambda **_kwargs: pytest.fail("summary 分支不应进入写作流水线"))

    assert run_write_command(args) == 6
    assert captured_output_dir["value"] == (tmp_path / "draft" / "AAPL").resolve()


@pytest.mark.unit
def test_main_write_mode_calls_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证写作模式会调用写作流水线并透传返回码。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    filings_dir = tmp_path / "portfolio" / "AAPL" / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(
        command="write",
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        output=str(tmp_path / "draft"),
        template=None,
        write_max_retries=2,
        resume=True,
        web_provider="auto",
        audit_model_name="deepseek-thinking",
    )

    collector = _CallCollector()
    monkeypatch.setattr("dayu.cli.commands.write.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.write.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr("dayu.cli.commands.write.setup_model_name", partial(_return_value, model_name))
    monkeypatch.setattr(
        "dayu.cli.commands.write._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    fake_dependencies = _FakeCliHostDependencies(running_config=running_config)
    monkeypatch.setattr(
        "dayu.cli.commands.write._prepare_cli_host_dependencies",
        lambda **_kwargs: fake_dependencies.as_tuple(),
    )
    monkeypatch.setattr("dayu.cli.commands.write._build_write_service", lambda **_kwargs: object())
    monkeypatch.setattr("dayu.cli.commands.write.Log.info", collector.capture_info)
    monkeypatch.setattr("dayu.cli.commands.write.Log.warn", collector.capture_warn)
    monkeypatch.setattr("dayu.cli.commands.write.run_write_pipeline", lambda **_kwargs: 4)

    assert run_write_command(args) == 4
    assert any("写作流水线启动" in item for item in collector.info_logs)
    assert any("写作流水线结束但返回非零" in item for item in collector.warn_logs)


@pytest.mark.unit
def test_main_write_mode_uses_resolved_company_name_and_normalized_model_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证写作 CLI 会传递公司名而非 ticker，并显式归一化主写作模型覆盖名。"""

    filings_dir = tmp_path / "portfolio" / "AAPL" / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    captured_write_config: dict[str, object] = {}
    args = Namespace(
        command="write",
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        output=str(tmp_path / "draft"),
        template=None,
        write_max_retries=2,
        resume=True,
        web_provider="auto",
        audit_model_name="deepseek-thinking",
        model_name=None,
    )

    monkeypatch.setattr("dayu.cli.commands.write.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.write.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr(
        "dayu.cli.commands.write.setup_model_name",
        partial(_return_value, ModelName(model_name="")),
    )
    monkeypatch.setattr(
        "dayu.cli.commands.write._build_execution_options",
        lambda _args: SimpleNamespace(model_name=None),
    )
    fake_dependencies = _FakeCliHostDependencies(running_config=running_config)
    fake_dependencies.fins_runtime = SimpleNamespace(
        get_company_name=lambda *_args, **_kwargs: "Apple Inc.",
        get_company_meta_summary=lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        "dayu.cli.commands.write._prepare_cli_host_dependencies",
        lambda **_kwargs: fake_dependencies.as_tuple(),
    )
    monkeypatch.setattr("dayu.cli.commands.write._build_write_service", lambda **_kwargs: object())
    monkeypatch.setattr(
        "dayu.cli.commands.write.run_write_pipeline",
        lambda **kwargs: captured_write_config.update({"write_config": kwargs["write_config"]}) or 0,
    )

    assert run_write_command(args) == 0
    write_config = cast(WriteRunConfig, captured_write_config["write_config"])
    assert write_config.company == "Apple Inc."
    assert write_config.ticker == "AAPL"
    assert write_config.write_model_override_name == ""


@pytest.mark.unit
def test_main_write_mode_logs_success_when_pipeline_returns_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证写作模式返回 0 时输出完成日志。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    filings_dir = tmp_path / "portfolio" / "AAPL" / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=True, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(
        command="write",
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        output=str(tmp_path / "draft"),
        template=None,
        write_max_retries=2,
        resume=True,
        web_provider="auto",
        audit_model_name="deepseek-thinking",
    )

    collector = _CallCollector()
    monkeypatch.setattr("dayu.cli.commands.write.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.write.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr("dayu.cli.commands.write.setup_model_name", partial(_return_value, model_name))
    monkeypatch.setattr(
        "dayu.cli.commands.write._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    fake_dependencies = _FakeCliHostDependencies(running_config=running_config)
    monkeypatch.setattr(
        "dayu.cli.commands.write._prepare_cli_host_dependencies",
        lambda **_kwargs: fake_dependencies.as_tuple(),
    )
    monkeypatch.setattr("dayu.cli.commands.write._build_write_service", lambda **_kwargs: object())
    monkeypatch.setattr("dayu.cli.commands.write.Log.info", collector.capture_info)
    monkeypatch.setattr("dayu.cli.commands.write.Log.warn", collector.capture_warn)
    monkeypatch.setattr("dayu.cli.commands.write.run_write_pipeline", lambda **_kwargs: 0)

    assert run_write_command(args) == 0
    assert any("写作流水线完成: exit_code=0" in item for item in collector.info_logs)
    assert not collector.warn_logs


@pytest.mark.unit
def test_main_write_mode_logs_elapsed_when_pipeline_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证写作模式异常时输出带耗时的错误日志并返回 2。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    filings_dir = tmp_path / "portfolio" / "AAPL" / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(
        command="write",
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        output=str(tmp_path / "draft"),
        template=None,
        write_max_retries=2,
        resume=True,
        web_provider="auto",
        audit_model_name="deepseek-thinking",
    )

    collector = _CallCollector()

    monkeypatch.setattr("dayu.cli.commands.write.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.write.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr("dayu.cli.commands.write.setup_model_name", partial(_return_value, model_name))
    monkeypatch.setattr(
        "dayu.cli.commands.write._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    fake_dependencies = _FakeCliHostDependencies(running_config=running_config)
    monkeypatch.setattr(
        "dayu.cli.commands.write._prepare_cli_host_dependencies",
        lambda **_kwargs: fake_dependencies.as_tuple(),
    )
    monkeypatch.setattr("dayu.cli.commands.write._build_write_service", lambda **_kwargs: object())
    monkeypatch.setattr("dayu.cli.commands.write.Log.info", collector.capture_info)
    monkeypatch.setattr("dayu.cli.commands.write.Log.error", collector.capture_error)
    monkeypatch.setattr("dayu.cli.commands.write.run_write_pipeline", _raise_runtime_error_for_write_pipeline)

    assert run_write_command(args) == 2
    assert any("写作模式执行失败: elapsed=" in item for item in collector.error_logs)


@pytest.mark.unit
def test_main_write_returns_130_when_run_write_pipeline_is_cancelled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证写作 CLI 在真实服务取消路径上返回 130 并记录取消日志。"""

    filings_dir = tmp_path / "portfolio" / "AAPL" / "filings"
    filings_dir.mkdir(parents=True, exist_ok=True)
    workspace_config = WorkspaceConfig(
        workspace_dir=tmp_path,
        output_dir=tmp_path / "output",
        config_loader=ConfigLoader(ConfigFileResolver(tmp_path / "config")),
        prompt_asset_store=FilePromptAssetStore(ConfigFileResolver(tmp_path / "config")),
        ticker="AAPL",
        has_local_filings=True,
    )
    running_config = RunningConfig(
        runner_running_config=AsyncOpenAIRunnerRunningConfig(),
        agent_running_config=AgentRunningConfig(),
        doc_tool_limits=DocToolLimits(),
        fins_tool_limits=FinsToolLimits(),
        web_tools_config=WebToolsConfig(provider="auto"),
        tool_trace_config=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
    )
    model_name = ModelName(model_name="mimo-v2-flash")
    args = Namespace(
        command="write",
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
        output=str(tmp_path / "draft"),
        template=None,
        write_max_retries=2,
        resume=True,
        web_provider="auto",
        audit_model_name="deepseek-thinking",
    )

    collector = _CallCollector()

    monkeypatch.setattr("dayu.cli.commands.write.setup_loglevel", lambda _args: None)
    monkeypatch.setattr("dayu.cli.commands.write.setup_paths", partial(_return_value, workspace_config))
    monkeypatch.setattr("dayu.cli.commands.write.setup_model_name", partial(_return_value, model_name))
    monkeypatch.setattr(
        "dayu.cli.commands.write._build_execution_options",
        lambda _args: SimpleNamespace(model_name="deepseek-thinking"),
    )
    fake_dependencies = _FakeCliHostDependencies(running_config=running_config)
    monkeypatch.setattr(
        "dayu.cli.commands.write._prepare_cli_host_dependencies",
        lambda **_kwargs: fake_dependencies.as_tuple(),
    )

    class _CancellingWriteHost:
        """测试用写作宿主，模拟真实同步取消收口。"""

        def create_session(self, _source: object) -> SimpleNamespace:
            """返回固定 session。"""

            return SimpleNamespace(session_id="cancelled-session")

        def run_operation_sync(
            self,
            *,
            spec: object,
            operation: object,
            on_cancel: object | None = None,
        ) -> int:
            """直接执行取消回调，模拟 Host 收口。"""

            del spec, operation
            if on_cancel is None:
                raise AssertionError("测试前提不成立：缺少 on_cancel 回调")
            cancel_callback = cast(Callable[[], int], on_cancel)
            return cancel_callback()

    monkeypatch.setattr(
        "dayu.cli.commands.write._build_write_service",
        lambda **_kwargs: WriteService(
            host=cast(HostedExecutionGatewayProtocol, _CancellingWriteHost()),
            host_governance=cast(Any, _CancellingWriteHost()),
            workspace=cast(Any, fake_dependencies.workspace),
            scene_execution_acceptance_preparer=cast(Any, fake_dependencies.scene_execution_acceptance_preparer),
        ),
    )
    monkeypatch.setattr("dayu.cli.commands.write.Log.info", collector.capture_info)
    monkeypatch.setattr("dayu.cli.commands.write.Log.warn", collector.capture_warn)
    monkeypatch.setattr("dayu.cli.commands.write.Log.error", collector.capture_error)

    assert run_write_command(args) == 130
    assert any("写作模式已取消: exit_code=130" in item for item in collector.warn_logs)
    assert not any("写作流水线结束但返回非零" in item for item in collector.warn_logs)


@pytest.mark.unit
def test_run_write_pipeline_accepts_only_real_write_arguments() -> None:
    """CLI 写作薄入口不应继续暴露未消费的兼容关键字。"""

    parameter_names = tuple(inspect.signature(run_write_pipeline).parameters)

    assert parameter_names == ("write_config", "write_service")

    entrypoint: Callable[..., int] = run_write_pipeline
    with pytest.raises(TypeError):
        entrypoint(
            workspace_config=object(),
            running_config=object(),
            model_config=object(),
            write_config=object(),
            write_service=object(),
        )


@pytest.mark.unit
def test_main_fins_command_bypasses_setup_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证财报命令分支不会调用 setup_paths。"""

    args = Namespace(
        command="download",
        base="./workspace",
        ticker="AAPL",
        form_type=["10Q"],
        start_date=None,
        end_date=None,
        overwrite=False,
        rebuild=False,
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
    )

    monkeypatch.setattr("dayu.cli.main.parse_arguments", partial(_return_value, args))
    monkeypatch.setattr("dayu.cli.commands.fins.run_fins_command", lambda _args: 0)

    def _fail_setup_paths(_args: object) -> object:
        raise AssertionError("fins 命令不应调用 setup_paths")

    monkeypatch.setattr("dayu.cli.commands.interactive.setup_paths", _fail_setup_paths)

    assert main() == 0


@pytest.mark.unit
def test_build_fins_command_normalizes_download_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `_build_fins_command` 会先做 download 参数规范化。"""

    def _fake_prepare_cli_args(namespace: Namespace) -> None:
        namespace.base = str(tmp_path)
        namespace.ticker = "BABA"
        namespace.ticker_aliases = ["9988", "9988.HK"]

    monkeypatch.setattr("dayu.cli.commands.fins.prepare_cli_args", _fake_prepare_cli_args)

    command = _build_fins_command(
        Namespace(
            command="download",
            base=str(tmp_path),
            ticker="BABA,9988,9988.HK",
            form_type=["10-K"],
            start_date=None,
            end_date=None,
            overwrite=False,
            rebuild=False,
            infer=True,
            log_level=None,
            debug=False,
            verbose=False,
            info=False,
            quiet=False,
        )
    )

    assert isinstance(command.payload, DownloadCommandPayload)
    assert command.payload.ticker == "BABA"
    assert command.payload.ticker_aliases == ("9988", "9988.HK")
    assert command.payload.infer is True


@pytest.mark.unit
def test_build_fins_command_preserves_upload_filing_infer_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `_build_fins_command` 会把上传命令的 infer 规范化结果写入 payload。"""

    def _fake_prepare_cli_args(namespace: Namespace) -> None:
        namespace.base = str(tmp_path)
        namespace.ticker = "AAPL"
        namespace.company_name = "Apple Inc."
        namespace.ticker_aliases = ["AAPL.US"]

    monkeypatch.setattr("dayu.cli.commands.fins.prepare_cli_args", _fake_prepare_cli_args)

    command = _build_fins_command(
        Namespace(
            command="upload_filing",
            base=str(tmp_path),
            ticker="AAPL,APC",
            files=["report.pdf"],
            fiscal_year=2024,
            fiscal_period="FY",
            action="create",
            amended=False,
            filing_date=None,
            report_date=None,
            company_id="1",
            company_name="",
            infer=True,
            overwrite=False,
            log_level=None,
            debug=False,
            verbose=False,
            info=False,
            quiet=False,
        )
    )

    assert isinstance(command.payload, UploadFilingCommandPayload)
    assert command.payload.ticker == "AAPL"
    assert command.payload.company_name == "Apple Inc."
    assert command.payload.ticker_aliases == ("AAPL.US",)
    assert command.payload.infer is True


@pytest.mark.unit
def test_build_fins_command_allows_upload_filing_delete_without_files(tmp_path: Path) -> None:
    """验证 `upload_filing --action delete` 不会因 `files=None` 提前失败。"""

    command = _build_fins_command(
        Namespace(
            command="upload_filing",
            base=str(tmp_path),
            ticker="AAPL",
            files=None,
            fiscal_year=2024,
            fiscal_period="FY",
            action="delete",
            amended=False,
            filing_date=None,
            report_date=None,
            company_id="1",
            company_name="Apple Inc.",
            infer=False,
            overwrite=False,
            log_level=None,
            debug=False,
            verbose=False,
            info=False,
            quiet=False,
        )
    )

    assert isinstance(command.payload, UploadFilingCommandPayload)
    assert command.payload.files == ()


@pytest.mark.unit
def test_build_fins_command_allows_upload_material_delete_without_files(tmp_path: Path) -> None:
    """验证 `upload_material --action delete` 不会因 `files=None` 提前失败。"""

    command = _build_fins_command(
        Namespace(
            command="upload_material",
            base=str(tmp_path),
            ticker="AAPL",
            files=None,
            action="delete",
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            document_id="mat_1",
            internal_document_id=None,
            fiscal_year=None,
            fiscal_period=None,
            filing_date=None,
            report_date=None,
            company_id="1",
            company_name="Apple Inc.",
            infer=False,
            overwrite=False,
            log_level=None,
            debug=False,
            verbose=False,
            info=False,
            quiet=False,
        )
    )

    assert isinstance(command.payload, UploadMaterialCommandPayload)
    assert command.payload.files == ()


@pytest.mark.unit
def test_run_fins_command_stream_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """验证 `run_fins_command` 可消费流式结果并输出最终摘要。"""

    class _FakeService:
        def __init__(self) -> None:
            pass

        def submit(self, request):
            command = request.command

            async def _stream():
                yield FinsEvent(
                    type=FinsEventType.PROGRESS,
                    command=command.name,
                    payload=DownloadProgressPayload(
                        event_type=FinsProgressEventName.PIPELINE_STARTED,
                        ticker="AAPL",
                    ),
                )
                yield FinsEvent(
                    type=FinsEventType.RESULT,
                    command=command.name,
                    payload=DownloadResultData(
                        pipeline="fake",
                        status="ok",
                        ticker="AAPL",
                        summary=DownloadSummary(total=0, downloaded=0, skipped=0, failed=0),
                    ),
                )

            return FinsSubmission(session_id="test-session", execution=_stream())

    monkeypatch.setattr("dayu.cli.commands.fins._build_fins_ops_service", lambda _args: _FakeService())
    monkeypatch.setattr("dayu.cli.commands.fins.format_fins_cli_result", lambda command, result: f"{command.value}:{result.status}")

    args = Namespace(
        command="download",
        base=str(tmp_path),
        ticker="AAPL",
        form_type=["10Q"],
        start_date=None,
        end_date=None,
        overwrite=False,
        rebuild=False,
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
    )

    assert run_fins_command(args) == 0
    output = capsys.readouterr().out
    assert "download:ok" in output


@pytest.mark.unit
def test_run_fins_command_upload_stream_logs_progress_at_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证上传类流式财报命令会在 INFO 级别输出进度。"""

    class _FakeService:
        def submit(self, request):
            command = request.command

            async def _stream():
                yield FinsEvent(
                    type=FinsEventType.PROGRESS,
                    command=command.name,
                    payload=UploadFilingProgressPayload(
                        event_type=FinsProgressEventName.UPLOAD_STARTED,
                        ticker="AAPL",
                        action="create",
                        file_count=1,
                    ),
                )
                yield FinsEvent(
                    type=FinsEventType.PROGRESS,
                    command=command.name,
                    payload=UploadFilingProgressPayload(
                        event_type=FinsProgressEventName.FILE_UPLOADED,
                        ticker="AAPL",
                        name="report.pdf",
                        size=123,
                    ),
                )
                yield FinsEvent(
                    type=FinsEventType.RESULT,
                    command=command.name,
                    payload=UploadFilingResultData(
                        pipeline="fake",
                        status="ok",
                        ticker="AAPL",
                        filing_action="create",
                    ),
                )

            return FinsSubmission(session_id="test-session", execution=_stream())

    info_lines: list[str] = []
    verbose_lines: list[str] = []

    def _capture_info(message: str, *, module: str = "APP") -> None:
        del module
        info_lines.append(message)

    def _capture_verbose(message: str, *, module: str = "APP") -> None:
        del module
        verbose_lines.append(message)

    monkeypatch.setattr("dayu.cli.commands.fins._build_fins_ops_service", lambda _args: _FakeService())
    monkeypatch.setattr("dayu.cli.commands.fins.format_fins_cli_result", lambda command, result: f"{command.value}:{result.status}")
    monkeypatch.setattr("dayu.cli.commands.fins.Log.info", _capture_info)
    monkeypatch.setattr("dayu.cli.commands.fins.Log.verbose", _capture_verbose)

    args = Namespace(
        command="upload_filing",
        base=str(tmp_path),
        ticker="AAPL",
        action="create",
        files=[],
        fiscal_year=2025,
        fiscal_period="FY",
        amended=False,
        filing_date=None,
        report_date=None,
        company_id="1",
        company_name="Apple",
        overwrite=False,
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
    )

    assert run_fins_command(args) == 0
    output = capsys.readouterr().out
    assert "upload_filing:ok" in output
    assert any("[upload_filing] upload_started ticker=AAPL action=create file_count=1" in line for line in info_lines)
    assert any("[upload_filing] file_uploaded ticker=AAPL name=report.pdf size=123" in line for line in info_lines)
    assert verbose_lines == []


@pytest.mark.unit
def test_run_fins_command_download_allows_missing_filings_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 `download` 在空工作区也能完成 CLI 装配并进入财报服务执行。

    Args:
        monkeypatch: pytest monkeypatch 对象。
        tmp_path: pytest 临时目录。
        capsys: 标准输出捕获器。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    class _FakeService:
        """模拟财报服务。"""

        def submit(self, request):
            """返回 `download` 提交句柄。

            Args:
                request: 财报服务提交请求。

            Returns:
                财报提交句柄。

            Raises:
                无。
            """

            command = request.command

            async def _stream():
                yield FinsEvent(
                    type=FinsEventType.RESULT,
                    command=command.name,
                    payload=DownloadResultData(
                        pipeline="fake",
                        status="ok",
                        ticker="ONC",
                        summary=DownloadSummary(total=0, downloaded=0, skipped=0, failed=0),
                    ),
                )

            return FinsSubmission(session_id="test-session", execution=_stream())

    monkeypatch.setattr("dayu.cli.commands.fins._build_fins_ops_service", lambda _args: _FakeService())
    monkeypatch.setattr("dayu.cli.commands.fins.format_fins_cli_result", lambda command, result: f"{command.value}:{result.status}")

    args = Namespace(
        command="download",
        base=str(tmp_path),
        ticker="ONC",
        form_type=None,
        start_date=None,
        end_date=None,
        overwrite=False,
        rebuild=False,
        config=None,
        log_level=None,
        debug=False,
        verbose=False,
        info=False,
        quiet=False,
    )

    assert run_fins_command(args) == 0
    output = capsys.readouterr().out
    assert "download:ok" in output
