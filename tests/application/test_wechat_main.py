"""WeChat 启动入口测试。"""

from __future__ import annotations

import asyncio
import argparse
import builtins
import importlib
import signal
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from dayu.contracts.toolset_config import ToolsetConfigSnapshot
from dayu.execution.options import ExecutionOptions, ResolvedExecutionOptions
from dayu.fins.service_runtime import DefaultFinsRuntime
from dayu.host import Host
from dayu.services.scene_execution_acceptance import SceneExecutionAcceptancePreparer
from dayu.services.startup_preparation import PreparedHostRuntimeDependencies
from dayu.startup.workspace import WorkspaceResources
from dayu.wechat.service_manager import InstalledServiceDefinition, ServiceSpec, ServiceStatus
from dayu.wechat.state_store import WeChatDaemonState

wechat_arg_module = importlib.import_module("dayu.wechat.arg_parsing")
wechat_main_module = importlib.import_module("dayu.wechat.main")
wechat_runtime_module = importlib.import_module("dayu.wechat.runtime")
wechat_login_module = importlib.import_module("dayu.wechat.commands.login")
wechat_run_module = importlib.import_module("dayu.wechat.commands.run")
wechat_service_module = importlib.import_module("dayu.wechat.commands.service")


@pytest.mark.unit
def test_build_execution_options_supports_interactive_like_overrides() -> None:
    """验证入口会解析 interactive 相关执行覆盖项。"""

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(
        [
            "run",
            "--model-name",
            "mimo-v2-flash",
            "--temperature",
            "0.4",
            "--web-provider",
            "duckduckgo",
            "--debug-sse",
            "--debug-tool-delta",
            "--debug-sse-sample-rate",
            "0.25",
            "--debug-sse-throttle-sec",
            "1.5",
            "--tool-timeout-seconds",
            "12",
            "--max-iterations",
            "7",
            "--fallback-mode",
            "raise_error",
            "--fallback-prompt",
            "answer briefly",
            "--max-duplicate-tool-calls",
            "4",
            "--duplicate-tool-hint-prompt",
            "avoid repeats",
            "--enable-tool-trace",
            "--doc-limits-json",
            '{"list_files_max": 12}',
            "--fins-limits-json",
            '{"list_documents_max_items": 23}',
        ]
    )

    options = wechat_arg_module._build_execution_options(args)

    assert isinstance(options, ExecutionOptions)
    assert options.model_name == "mimo-v2-flash"
    assert options.temperature == 0.4
    assert options.web_provider == "duckduckgo"
    assert options.debug_sse is True
    assert options.debug_tool_delta is True
    assert options.debug_sse_sample_rate == 0.25
    assert options.debug_sse_throttle_sec == 1.5
    assert options.tool_timeout_seconds == 12
    assert options.max_iterations == 7
    assert options.fallback_mode == "raise_error"
    assert options.fallback_prompt == "answer briefly"
    assert options.max_duplicate_tool_calls == 4
    assert options.duplicate_tool_hint_prompt == "avoid repeats"
    assert options.trace_enabled is True
    assert options.doc_tool_limits is None
    assert options.fins_tool_limits is None
    assert options.toolset_config_overrides == (
        ToolsetConfigSnapshot(toolset_name="doc", version="1", payload={"list_files_max": 12}),
        ToolsetConfigSnapshot(toolset_name="fins", version="1", payload={"list_documents_max_items": 23}),
    )


@pytest.mark.unit
def test_create_parser_uses_python_module_prog() -> None:
    """验证顶层 usage 使用 `python -m dayu.wechat`，不暴露 `__main__.py`。"""

    parser = wechat_arg_module._create_parser()

    assert parser.prog == "python -m dayu.wechat"
    assert "__main__.py" not in parser.format_usage()


@pytest.mark.unit
def test_create_parser_missing_command_prints_help_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """验证缺少命令时会输出完整帮助和命令摘要。"""

    parser = wechat_arg_module._create_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([])

    captured = capsys.readouterr()

    assert exc_info.value.code == 2
    assert "python -m dayu.wechat" in captured.err
    assert "__main__.py" not in captured.err
    assert "login" in captured.err
    assert "run" in captured.err
    assert "service" in captured.err
    assert "错误: 缺少命令" in captured.err


@pytest.mark.unit
def test_resolve_state_dir_defaults_to_workspace_dayu_wechat_default(tmp_path: Path) -> None:
    """验证默认状态目录为 `<workspace>/.dayu/wechat-default`。"""

    resolved = wechat_arg_module._resolve_state_dir(tmp_path, "default")

    assert resolved == (tmp_path / ".dayu" / "wechat-default").resolve()


@pytest.mark.unit
def test_resolve_instance_label_rejects_invalid_characters() -> None:
    """验证实例标签禁止包含路径穿越类非法字符。"""

    with pytest.raises(SystemExit, match="2"):
        wechat_arg_module._resolve_instance_label("bad/path")


@pytest.mark.unit
def test_resolve_workspace_root_rejects_missing_dir(tmp_path: Path) -> None:
    """验证缺失工作区目录时直接退出。"""

    missing_dir = tmp_path / "missing"

    with pytest.raises(SystemExit):
        wechat_arg_module._resolve_workspace_root(str(missing_dir))


@pytest.mark.unit
def test_run_login_command_triggers_login(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `login` 命令只执行扫码登录，不创建应用服务。"""

    captured: dict[str, object] = {}
    context = wechat_arg_module.ResolvedWechatContext(
        workspace_root=tmp_path,
        config_root=tmp_path / "config",
        state_dir=tmp_path / ".wechat",
        execution_options=ExecutionOptions(),
    )

    class _FakeDaemon:
        async def ensure_authenticated(self, *, force_relogin: bool = False) -> None:
            captured["force_relogin"] = force_relogin

        async def aclose(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(wechat_login_module, "_resolve_command_context", lambda _args: context)
    monkeypatch.setattr(wechat_login_module, "_create_login_daemon", lambda _args, _context: _FakeDaemon())

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["login", "--relogin"])
    exit_code = asyncio.run(wechat_login_module._run_login_command(args))

    assert exit_code == 0
    assert captured == {"force_relogin": True, "closed": True}


@pytest.mark.unit
def test_run_command_requires_existing_login(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `run` 命令缺少登录态时会直接失败。"""

    context = wechat_arg_module.ResolvedWechatContext(
        workspace_root=tmp_path,
        config_root=tmp_path / "config",
        state_dir=tmp_path / ".wechat",
        execution_options=ExecutionOptions(),
    )

    class _FakeStore:
        def __init__(self, _state_dir: Path) -> None:
            self.state_dir = _state_dir

        def load(self) -> WeChatDaemonState:
            return WeChatDaemonState(bot_token=None, base_url="https://ilink.example")

    monkeypatch.setattr(wechat_run_module, "_resolve_command_context", lambda _args: context)
    monkeypatch.setattr(wechat_run_module, "FileWeChatStateStore", _FakeStore)
    monkeypatch.setattr(
        wechat_run_module,
        "_create_run_daemon",
        lambda _args, _context: (_ for _ in ()).throw(AssertionError("不应创建 daemon")),
    )

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["run"])
    exit_code = asyncio.run(wechat_run_module._run_run_command(args))

    assert exit_code == 1


@pytest.mark.unit
def test_run_command_uses_existing_login(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `run` 命令会在已有登录态下进入长轮询。"""

    captured: dict[str, object] = {}
    context = wechat_arg_module.ResolvedWechatContext(
        workspace_root=tmp_path,
        config_root=tmp_path / "config",
        state_dir=tmp_path / ".wechat",
        execution_options=ExecutionOptions(),
    )

    class _FakeStore:
        def __init__(self, _state_dir: Path) -> None:
            self.state_dir = _state_dir

        def load(self) -> WeChatDaemonState:
            return WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example")

    fake_daemon = object()

    async def _fake_run_daemon_with_graceful_shutdown(_daemon, *, require_existing_auth: bool) -> int:
        captured["daemon"] = _daemon
        captured["require_existing_auth"] = require_existing_auth
        return 0

    monkeypatch.setattr(wechat_run_module, "_resolve_command_context", lambda _args: context)
    monkeypatch.setattr(wechat_run_module, "FileWeChatStateStore", _FakeStore)
    monkeypatch.setattr(wechat_run_module, "_create_run_daemon", lambda _args, _context: fake_daemon)
    monkeypatch.setattr(
        wechat_run_module,
        "_run_daemon_with_graceful_shutdown",
        _fake_run_daemon_with_graceful_shutdown,
    )

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["run"])
    exit_code = asyncio.run(wechat_run_module._run_run_command(args))

    assert exit_code == 0
    assert captured == {"daemon": fake_daemon, "require_existing_auth": True}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("signal_name", "exit_code"),
    [("SIGTERM", 0), ("SIGINT", 130)],
)
def test_run_daemon_with_graceful_shutdown_maps_signal_to_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    signal_name: str,
    exit_code: int,
) -> None:
    """验证前台 daemon 会把退出信号收敛到统一的受控退出码。"""

    captured: dict[str, object] = {}

    class _FakeDaemon:
        async def run_forever(self, *, require_existing_auth: bool = False) -> None:
            captured["require_existing_auth"] = require_existing_auth
            await asyncio.sleep(3600)

        async def aclose(self) -> None:
            captured["closed"] = True

    def _fake_install_signal_handlers(loop, run_task, shutdown_state):
        loop.call_soon(
            lambda: wechat_run_module._request_daemon_shutdown(
                run_task,
                shutdown_state,
                signal_name=signal_name,
                exit_code=exit_code,
            )
        )
        return [signal.SIGTERM]

    monkeypatch.setattr(wechat_run_module, "_install_daemon_signal_handlers", _fake_install_signal_handlers)
    monkeypatch.setattr(wechat_run_module, "_remove_daemon_signal_handlers", lambda _loop, _signals: None)

    actual_exit_code = asyncio.run(
        wechat_run_module._run_daemon_with_graceful_shutdown(
            _FakeDaemon(),
            require_existing_auth=True,
        )
    )

    assert actual_exit_code == exit_code
    assert captured == {"require_existing_auth": True, "closed": True}


@pytest.mark.unit
def test_install_daemon_signal_handlers_passes_positional_callback_args(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 asyncio signal handler 安装使用的位置参数可直接调用回调。"""

    captured: dict[str, object] = {}
    loop = asyncio.new_event_loop()
    shutdown_state = wechat_run_module.DaemonShutdownState()

    class _FakeTask:
        def cancel(self) -> None:
            captured["cancelled"] = True

    fake_task = _FakeTask()

    def _fake_add_signal_handler(os_signal, callback, *args):
        captured["signal"] = os_signal
        callback(*args)

    monkeypatch.setattr(loop, "add_signal_handler", _fake_add_signal_handler)
    monkeypatch.setattr(wechat_run_module.Log, "info", lambda *_args, **_kwargs: None)

    installed = wechat_run_module._install_daemon_signal_handlers(loop, fake_task, shutdown_state)

    assert installed == [signal.SIGINT, signal.SIGTERM]
    assert captured["cancelled"] is True
    assert shutdown_state.signal_name == "SIGINT"
    assert shutdown_state.exit_code == 130

    loop.close()


@pytest.mark.unit
def test_build_run_cli_arguments_includes_overrides(tmp_path: Path) -> None:
    """验证 service 运行参数会带上 run 命令需要的关键覆盖项。"""

    context = wechat_arg_module.ResolvedWechatContext(
        workspace_root=tmp_path / "workspace",
        config_root=tmp_path / "workspace" / "config",
        state_dir=tmp_path / "workspace" / ".dayu" / "wechat-ops",
        execution_options=ExecutionOptions(),
        instance_label="ops",
    )
    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(
        [
            "service",
            "install",
            "--typing-interval-sec",
            "3",
            "--delivery-max-attempts",
            "5",
            "--model-name",
            "deepseek-chat",
            "--debug-sse",
            "--debug-tool-delta",
            "--debug-sse-sample-rate",
            "0.5",
            "--debug-sse-throttle-sec",
            "2.0",
            "--fallback-mode",
            "raise_error",
            "--fallback-prompt",
            "force concise answer",
            "--max-duplicate-tool-calls",
            "6",
            "--duplicate-tool-hint-prompt",
            "use new evidence",
            "--enable-tool-trace",
            "--log-level",
            "debug",
        ]
    )

    cli_arguments = wechat_runtime_module._build_run_cli_arguments(args, context)

    assert cli_arguments[:7] == [
        "run",
        "--base",
        str(context.workspace_root),
        "--config",
        str(context.config_root),
        "--label",
        "ops",
    ]
    assert "--typing-interval-sec" in cli_arguments
    assert "3.0" in cli_arguments
    assert "--delivery-max-attempts" in cli_arguments
    assert "5" in cli_arguments
    assert "--model-name" in cli_arguments
    assert "deepseek-chat" in cli_arguments
    assert "--debug-sse" in cli_arguments
    assert "--debug-tool-delta" in cli_arguments
    assert "--debug-sse-sample-rate" in cli_arguments
    assert "0.5" in cli_arguments
    assert "--debug-sse-throttle-sec" in cli_arguments
    assert "2.0" in cli_arguments
    assert "--fallback-mode" in cli_arguments
    assert "raise_error" in cli_arguments
    assert "--fallback-prompt" in cli_arguments
    assert "force concise answer" in cli_arguments
    assert "--max-duplicate-tool-calls" in cli_arguments
    assert "6" in cli_arguments
    assert "--duplicate-tool-hint-prompt" in cli_arguments
    assert "use new evidence" in cli_arguments
    assert "--enable-tool-trace" in cli_arguments
    assert "--log-level" in cli_arguments


@pytest.mark.unit
def test_build_run_cli_arguments_persists_non_default_context_delivery_max_attempts(tmp_path: Path) -> None:
    """当 context 已经携带非默认 delivery 重试次数时，应继续写回 run 参数。"""

    context = wechat_arg_module.ResolvedWechatContext(
        workspace_root=tmp_path / "workspace",
        config_root=tmp_path / "workspace" / "config",
        state_dir=tmp_path / "workspace" / ".dayu" / "wechat-ops",
        execution_options=ExecutionOptions(),
        delivery_max_attempts=5,
        instance_label="ops",
    )
    args = argparse.Namespace(
        typing_interval_sec=wechat_arg_module.DEFAULT_TYPING_INTERVAL_SEC,
        model_name=None,
        temperature=None,
        web_provider=None,
        debug_sse=False,
        debug_tool_delta=False,
        debug_sse_sample_rate=None,
        debug_sse_throttle_sec=None,
        tool_timeout_seconds=None,
        max_iterations=None,
        fallback_mode=None,
        fallback_prompt=None,
        max_duplicate_tool_calls=None,
        duplicate_tool_hint_prompt=None,
        enable_tool_trace=False,
        tool_trace_dir=None,
        doc_limits_json=None,
        fins_limits_json=None,
        log_level=None,
    )

    cli_arguments = wechat_runtime_module._build_run_cli_arguments(args, context)

    assert "--delivery-max-attempts" in cli_arguments
    assert "5" in cli_arguments


@pytest.mark.unit
def test_service_start_requires_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证未安装 service 时不会尝试启动。"""

    identity = wechat_runtime_module.ResolvedWechatServiceIdentity(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "com.dayu.wechat.test.plist",
        state_dir=tmp_path / ".wechat",
        instance_label="ops",
    )

    monkeypatch.setattr(wechat_service_module, "_resolve_service_identity", lambda _args: identity)
    monkeypatch.setattr(
        wechat_service_module,
        "_query_installed_service_status",
        lambda _identity: ServiceStatus(
            backend=identity.backend,
            label=identity.label,
            definition_path=identity.definition_path,
            installed=False,
            loaded=False,
        ),
    )
    monkeypatch.setattr(
        wechat_service_module,
        "start_service",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("不应启动 service")),
    )

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "start"])

    assert wechat_service_module._run_service_start_command(args) == 1


@pytest.mark.unit
def test_service_start_returns_running_when_service_already_loaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `service start` 在已运行时不会触发重启。"""

    identity = wechat_runtime_module.ResolvedWechatServiceIdentity(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "com.dayu.wechat.test.plist",
        state_dir=tmp_path / ".wechat",
        instance_label="ops",
    )
    logged: list[str] = []

    monkeypatch.setattr(wechat_service_module, "_resolve_service_identity", lambda _args: identity)
    monkeypatch.setattr(
        wechat_service_module,
        "_query_installed_service_status",
        lambda _identity: ServiceStatus(
            backend=identity.backend,
            label=identity.label,
            definition_path=identity.definition_path,
            installed=True,
            loaded=True,
            pid=123,
        ),
    )
    monkeypatch.setattr(
        wechat_service_module,
        "FileWeChatStateStore",
        lambda _state_dir: (_ for _ in ()).throw(AssertionError("已运行时不应再读取登录态")),
    )
    monkeypatch.setattr(
        wechat_service_module,
        "start_service",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("已运行时不应再次启动 service")),
    )
    monkeypatch.setattr(
        wechat_service_module.Log,
        "info",
        lambda message, *, module="APP": logged.append(f"{module}:{message}"),
    )

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "start"])

    assert wechat_service_module._run_service_start_command(args) == 0
    assert logged == ["APP.WECHAT.MAIN:macOS launchd 服务实例已在运行: ops"]


@pytest.mark.unit
def test_service_start_launchd_loaded_without_pid_still_calls_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 launchd 已 bootstrap 但无 pid 时，`service start` 仍会走恢复路径。"""

    identity = wechat_runtime_module.ResolvedWechatServiceIdentity(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "com.dayu.wechat.test.plist",
        state_dir=tmp_path / ".wechat",
        instance_label="ops",
    )
    printed: list[str] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(wechat_service_module, "_resolve_service_identity", lambda _args: identity)
    monkeypatch.setattr(
        wechat_service_module,
        "_query_installed_service_status",
        lambda _identity: ServiceStatus(
            backend=identity.backend,
            label=identity.label,
            definition_path=identity.definition_path,
            installed=True,
            loaded=True,
            pid=None,
        ),
    )
    monkeypatch.setattr(wechat_service_module, "_has_persisted_wechat_login", lambda _state_dir: True)
    monkeypatch.setattr(
        wechat_service_module,
        "start_service",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        wechat_service_module.Log,
        "info",
        lambda message, *, module="APP": (_ for _ in ()).throw(AssertionError("无 pid 时不应短路成已运行日志")),
    )
    monkeypatch.setattr(builtins, "print", lambda message: printed.append(message))

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "start"])

    assert wechat_service_module._run_service_start_command(args) == 0
    assert captured == {
        "label": identity.label,
        "definition_path": identity.definition_path,
        "backend": identity.backend,
    }
    assert printed == ["已启动 macOS launchd 服务实例: ops"]


@pytest.mark.unit
def test_service_restart_requires_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证未安装 service 时不会尝试重启。"""

    identity = wechat_runtime_module.ResolvedWechatServiceIdentity(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "com.dayu.wechat.test.plist",
        state_dir=tmp_path / ".wechat",
    )

    monkeypatch.setattr(wechat_service_module, "_resolve_service_identity", lambda _args: identity)
    monkeypatch.setattr(
        wechat_service_module,
        "query_service_status",
        lambda **_kwargs: ServiceStatus(
            backend=identity.backend,
            label=identity.label,
            definition_path=identity.definition_path,
            installed=False,
            loaded=False,
        ),
    )
    monkeypatch.setattr(
        wechat_service_module,
        "restart_service",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("不应重启 service")),
    )

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "restart"])

    assert wechat_service_module._run_service_restart_command(args) == 1


@pytest.mark.unit
def test_service_restart_restarts_loaded_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `service restart` 会调用 restart_service 并打印重启结果。"""

    identity = wechat_runtime_module.ResolvedWechatServiceIdentity(
        backend="systemd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "com.dayu.wechat.test.service",
        state_dir=tmp_path / ".wechat",
        instance_label="ops",
    )
    printed: list[str] = []
    captured: dict[str, object] = {}

    monkeypatch.setattr(wechat_service_module, "_resolve_service_identity", lambda _args: identity)
    monkeypatch.setattr(
        wechat_service_module,
        "_query_installed_service_status",
        lambda _identity: ServiceStatus(
            backend=identity.backend,
            label=identity.label,
            definition_path=identity.definition_path,
            installed=True,
            loaded=True,
            pid=456,
        ),
    )
    monkeypatch.setattr(wechat_service_module, "_has_persisted_wechat_login", lambda _state_dir: True)
    monkeypatch.setattr(
        wechat_service_module,
        "restart_service",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(builtins, "print", lambda message: printed.append(message))

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "restart"])

    assert wechat_service_module._run_service_restart_command(args) == 0
    assert captured == {
        "label": identity.label,
        "definition_path": identity.definition_path,
        "backend": identity.backend,
    }
    assert printed == ["已重启 Linux systemd --user 服务实例: ops"]


@pytest.mark.unit
def test_service_install_uses_systemd_backend_on_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `service install` 在 Linux 上会自动走 systemd backend。"""

    captured: dict[str, object] = {}
    context = wechat_arg_module.ResolvedWechatContext(
        workspace_root=tmp_path,
        config_root=tmp_path / "config",
        state_dir=tmp_path / ".dayu" / "wechat-default",
        execution_options=ExecutionOptions(),
    )
    fake_spec = ServiceSpec(
        backend="systemd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "service.service",
        working_directory=tmp_path,
        program_arguments=("/usr/bin/python3", "-m", "dayu.wechat", "run"),
    )
    printed: list[str] = []

    monkeypatch.setattr(wechat_service_module, "_resolve_command_context", lambda _args: context)
    monkeypatch.setattr(wechat_service_module, "detect_service_backend", lambda: "systemd")
    monkeypatch.setattr(wechat_service_module, "_resolve_repo_root", lambda: tmp_path)
    monkeypatch.setattr(wechat_service_module, "_collect_service_environment_variables", lambda _context: {"MIMO_API_KEY": "secret-1"})

    def _fake_build_service_spec(**kwargs) -> ServiceSpec:
        captured["backend"] = kwargs["backend"]
        captured["environment_variables"] = kwargs["environment_variables"]
        return fake_spec

    monkeypatch.setattr(
        wechat_service_module,
        "build_service_spec",
        _fake_build_service_spec,
    )
    monkeypatch.setattr(wechat_service_module, "install_service", lambda spec: captured.setdefault("spec", spec))
    monkeypatch.setattr(builtins, "print", lambda message: printed.append(message))

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "install"])

    assert wechat_service_module._run_service_install_command(args) == 0
    assert captured["backend"] == "systemd"
    assert captured["environment_variables"] == {"MIMO_API_KEY": "secret-1"}
    assert captured["spec"] == fake_spec
    assert printed[0] == "已安装 Linux systemd --user 服务实例: default"


@pytest.mark.unit
def test_service_status_prints_launchd_log_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `service status` 会打印 macOS 后台日志文件路径。"""

    identity = wechat_runtime_module.ResolvedWechatServiceIdentity(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "com.dayu.wechat.test.plist",
        state_dir=tmp_path / ".wechat",
    )
    printed: list[str] = []

    class _FakeStore:
        def __init__(self, _state_dir: Path) -> None:
            self.state_dir = _state_dir

        def load(self) -> WeChatDaemonState:
            return WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example")

    monkeypatch.setattr(wechat_service_module, "_resolve_service_identity", lambda _args: identity)
    monkeypatch.setattr(
        wechat_service_module,
        "query_service_status",
        lambda **_kwargs: ServiceStatus(
            backend=identity.backend,
            label=identity.label,
            definition_path=identity.definition_path,
            installed=True,
            loaded=True,
            pid=123,
        ),
    )
    monkeypatch.setattr(wechat_service_module, "FileWeChatStateStore", _FakeStore)
    monkeypatch.setattr(builtins, "print", lambda message: printed.append(message))

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "status"])

    assert wechat_service_module._run_service_status_command(args) == 0
    assert f"log_stdout: {(tmp_path / '.wechat' / 'logs' / 'launchd.stdout.log').resolve()}" in printed
    assert f"log_stderr: {(tmp_path / '.wechat' / 'logs' / 'launchd.stderr.log').resolve()}" in printed


@pytest.mark.unit
def test_service_status_prints_systemd_log_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `service status` 会打印 Linux journal 查看命令。"""

    identity = wechat_runtime_module.ResolvedWechatServiceIdentity(
        backend="systemd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "com.dayu.wechat.test.service",
        state_dir=tmp_path / ".wechat",
    )
    printed: list[str] = []

    class _FakeStore:
        def __init__(self, _state_dir: Path) -> None:
            self.state_dir = _state_dir

        def load(self) -> WeChatDaemonState:
            return WeChatDaemonState(bot_token="token-1", base_url="https://ilink.example")

    monkeypatch.setattr(wechat_service_module, "_resolve_service_identity", lambda _args: identity)
    monkeypatch.setattr(
        wechat_service_module,
        "query_service_status",
        lambda **_kwargs: ServiceStatus(
            backend=identity.backend,
            label=identity.label,
            definition_path=identity.definition_path,
            installed=True,
            loaded=True,
            pid=456,
        ),
    )
    monkeypatch.setattr(wechat_service_module, "FileWeChatStateStore", _FakeStore)
    monkeypatch.setattr(builtins, "print", lambda message: printed.append(message))

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "status"])

    assert wechat_service_module._run_service_status_command(args) == 0
    assert "log_backend: journal" in printed
    assert "log_follow_command: journalctl --user -u com.dayu.wechat.test.service -f" in printed


@pytest.mark.unit
def test_service_list_only_prints_installed_instances(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `service list` 以已安装 definition 为真源，而非依赖状态目录存在。"""

    printed: list[str] = []

    class _FakeStore:
        def __init__(self, state_dir: Path) -> None:
            self._state_dir = state_dir

        def load(self) -> WeChatDaemonState:
            token = "token-1" if self._state_dir.name == "wechat-alpha" else None
            return WeChatDaemonState(bot_token=token, base_url="https://ilink.example")

    def _fake_query_service_status(*, label: str, definition_path: Path, backend: str) -> ServiceStatus:
        assert backend == "launchd"
        installed = label != "svc-wechat-beta"
        loaded = label == "svc-wechat-alpha"
        return ServiceStatus(
            backend="launchd",
            label=label,
            definition_path=definition_path,
            installed=installed,
            loaded=loaded,
            pid=123 if loaded else None,
        )

    monkeypatch.setattr(wechat_runtime_module, "detect_service_backend", lambda: "launchd")
    monkeypatch.setattr(
        wechat_runtime_module,
        "list_installed_service_definitions",
        lambda _backend: (
            InstalledServiceDefinition(
                backend="launchd",
                label="svc-wechat-alpha",
                definition_path=tmp_path / "svc-wechat-alpha.plist",
                program_arguments=(
                    "/usr/bin/python3",
                    "-m",
                    "dayu.wechat",
                    "run",
                    "--base",
                    str(tmp_path),
                    "--label",
                    "alpha",
                ),
            ),
            InstalledServiceDefinition(
                backend="launchd",
                label="svc-wechat-beta",
                definition_path=tmp_path / "svc-wechat-beta.plist",
                program_arguments=(
                    "/usr/bin/python3",
                    "-m",
                    "dayu.wechat",
                    "run",
                    "--base",
                    str(tmp_path),
                    "--label",
                    "beta",
                ),
            ),
            InstalledServiceDefinition(
                backend="launchd",
                label="svc-other-workspace",
                definition_path=tmp_path / "svc-other-workspace.plist",
                program_arguments=(
                    "/usr/bin/python3",
                    "-m",
                    "dayu.wechat",
                    "run",
                    "--base",
                    str(tmp_path / "other"),
                    "--label",
                    "other",
                ),
            ),
            InstalledServiceDefinition(
                backend="launchd",
                label="svc-malformed",
                definition_path=tmp_path / "svc-malformed.plist",
                program_arguments=("/usr/bin/python3", "-m", "dayu.wechat", "login"),
            ),
            InstalledServiceDefinition(
                backend="launchd",
                label="svc-bad-label",
                definition_path=tmp_path / "svc-bad-label.plist",
                program_arguments=(
                    "/usr/bin/python3",
                    "-m",
                    "dayu.wechat",
                    "run",
                    "--base",
                    str(tmp_path),
                    "--label",
                    "bad/path",
                ),
            ),
        ),
    )
    monkeypatch.setattr(wechat_runtime_module, "query_service_status", _fake_query_service_status)
    monkeypatch.setattr(wechat_runtime_module, "FileWeChatStateStore", _FakeStore)
    monkeypatch.setattr(builtins, "print", lambda message: printed.append(message))

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "list", "--base", str(tmp_path)])

    assert wechat_service_module._run_service_list_command(args) == 0
    assert "instance_label: alpha" in printed
    assert "service: 运行中" in printed
    assert "logged_in: yes" in printed
    assert all("beta" not in line for line in printed)
    assert all("other" not in line for line in printed)
    assert all("bad/path" not in line for line in printed)


@pytest.mark.unit
def test_main_returns_130_on_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证主入口在 Ctrl+C 时静默返回 130，而不是抛出堆栈。"""

    parser = wechat_arg_module._create_parser()
    logged: list[str] = []

    monkeypatch.setattr(wechat_main_module, "parse_arguments", lambda _argv=None: parser.parse_args(["login"]))
    monkeypatch.setattr(wechat_main_module, "setup_loglevel", lambda _args: None)
    monkeypatch.setattr(
        wechat_main_module,
        "_dispatch_command",
        lambda _args: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    monkeypatch.setattr(wechat_main_module.Log, "info", lambda message, *, module="APP": logged.append(f"{module}:{message}"))

    exit_code = wechat_main_module.main(["login"])

    assert exit_code == 130
    assert logged == ["APP.WECHAT.MAIN:收到中断信号，WeChat daemon 正在退出"]


@pytest.mark.unit
def test_wechat_main_helper_functions_cover_noop_services_context_and_daemon_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 wechat main 中剩余的 no-op helper 与 daemon/context 组装分支。"""

    parser = wechat_arg_module._create_parser()
    login_args = parser.parse_args(["login", "--base", str(tmp_path)])
    run_args = parser.parse_args(
        [
            "run",
            "--base",
            str(tmp_path),
            "--typing-interval-sec",
            "2.5",
            "--delivery-max-attempts",
            "4",
        ]
    )
    context = wechat_arg_module._resolve_command_context(login_args)
    run_context = wechat_arg_module._resolve_command_context(run_args)
    created: dict[str, object] = {}

    monkeypatch.setattr(
        wechat_runtime_module,
        "FileWeChatStateStore",
        lambda state_dir: ("store", state_dir),
    )
    monkeypatch.setattr(
        wechat_runtime_module,
        "WeChatDaemon",
        lambda **kwargs: created.setdefault("daemon", kwargs),
    )

    daemon_config = wechat_runtime_module._build_daemon_config(
        run_args,
        run_context,
        allow_interactive_relogin=False,
    )
    login_daemon = wechat_runtime_module._create_login_daemon(login_args, context)

    assert context.workspace_root == tmp_path.resolve()
    assert context.config_root == (tmp_path / "config").resolve()
    assert context.state_dir == (tmp_path / ".dayu" / "wechat-default").resolve()
    assert context.instance_label == "default"
    assert daemon_config.allow_interactive_relogin is False
    assert daemon_config.typing_interval_sec == pytest.approx(2.5)
    assert daemon_config.delivery_max_attempts == 4
    daemon_kwargs = cast(dict[str, object], created["daemon"])
    assert login_daemon == daemon_kwargs
    assert daemon_kwargs["state_store"] == ("store", context.state_dir)
    assert isinstance(daemon_kwargs["chat_service"], wechat_runtime_module.NoOpChatService)
    assert isinstance(daemon_kwargs["reply_delivery_service"], wechat_runtime_module.NoOpReplyDeliveryService)
    module_file = wechat_main_module.__file__
    assert module_file is not None
    assert wechat_runtime_module._resolve_repo_root() == Path(module_file).resolve().parents[2]


@pytest.mark.unit
def test_wechat_main_noop_services_raise_for_all_operations() -> None:
    """验证 login 模式下的 no-op service 在误调用时统一抛错。"""

    chat_service = wechat_runtime_module.NoOpChatService()
    reply_service = wechat_runtime_module.NoOpReplyDeliveryService()

    with pytest.raises(RuntimeError, match="ChatService"):
        asyncio.run(chat_service.submit_turn(cast(object, object())))
    with pytest.raises(RuntimeError, match="ChatService"):
        asyncio.run(chat_service.resume_pending_turn(cast(object, object())))
    with pytest.raises(RuntimeError, match="ChatService"):
        chat_service.list_resumable_pending_turns(session_id="session-1", scene_name="wechat")
    with pytest.raises(RuntimeError, match="ReplyDeliveryService"):
        reply_service.submit_reply_for_delivery(cast(object, object()))
    with pytest.raises(RuntimeError, match="ReplyDeliveryService"):
        reply_service.get_delivery("delivery-1")
    with pytest.raises(RuntimeError, match="ReplyDeliveryService"):
        reply_service.list_deliveries(session_id="session-1", scene_name="wechat", state="pending")
    with pytest.raises(RuntimeError, match="ReplyDeliveryService"):
        reply_service.claim_delivery("delivery-1")
    with pytest.raises(RuntimeError, match="ReplyDeliveryService"):
        reply_service.mark_delivery_delivered("delivery-1")
    with pytest.raises(RuntimeError, match="ReplyDeliveryService"):
        reply_service.mark_delivery_failed(cast(object, object()))


@pytest.mark.unit
def test_prepare_wechat_host_dependencies_runs_unified_startup_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """WeChat Host 依赖装配应委托 Service 共享启动 API。"""

    context = wechat_arg_module.ResolvedWechatContext(
        workspace_root=tmp_path,
        config_root=tmp_path / "config",
        state_dir=tmp_path / ".dayu" / "wechat-default",
        execution_options=ExecutionOptions(),
    )
    fake_workspace = cast(WorkspaceResources, object())
    fake_default_execution_options = cast(ResolvedExecutionOptions, object())
    fake_scene_preparer = cast(SceneExecutionAcceptancePreparer, object())
    fake_fins_runtime = cast(DefaultFinsRuntime, object())
    fake_host = cast(Host, object())
    captured_call: dict[str, object] = {}

    monkeypatch.setattr(
        wechat_runtime_module,
        "prepare_host_runtime_dependencies",
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

    prepared = wechat_runtime_module._prepare_wechat_host_dependencies(context)

    assert prepared == (
        fake_workspace,
        fake_default_execution_options,
        fake_scene_preparer,
        fake_host,
        fake_fins_runtime,
    )
    assert captured_call == {
        "workspace_root": context.workspace_root,
        "config_root": context.config_root,
        "execution_options": context.execution_options,
        "runtime_label": "WeChat Host runtime",
        "log_module": "APP.WECHAT.MAIN",
    }


@pytest.mark.unit
def test_wechat_main_helper_functions_cover_env_identity_parsing_and_signal_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证环境变量采集、service identity、参数错误与 signal cleanup helper。"""

    context = wechat_arg_module.ResolvedWechatContext(
        workspace_root=tmp_path,
        config_root=tmp_path / "config",
        state_dir=tmp_path / ".wechat",
        execution_options=ExecutionOptions(),
    )
    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "status", "--base", str(tmp_path)])
    errors: list[str] = []
    removed_signals: list[signal.Signals] = []

    monkeypatch.setattr(wechat_runtime_module, "ConfigFileResolver", lambda config_root: ("resolver", config_root))
    monkeypatch.setattr(
        wechat_runtime_module,
        "ConfigLoader",
        lambda resolver: SimpleNamespace(collect_referenced_env_vars=lambda: ["MIMO_API_KEY", "EMPTY_ENV"]),
    )
    monkeypatch.setattr(wechat_runtime_module.os.environ, "get", lambda name: {
        "MIMO_API_KEY": " secret-1 ",
        "EMPTY_ENV": "   ",
    }.get(name))
    monkeypatch.setattr(wechat_runtime_module, "detect_service_backend", lambda: "launchd")
    monkeypatch.setattr(wechat_runtime_module, "list_installed_service_definitions", lambda _backend: ())
    monkeypatch.setattr(wechat_runtime_module, "build_service_label", lambda _state_dir: "com.dayu.wechat.test")
    monkeypatch.setattr(
        wechat_runtime_module,
        "resolve_service_definition_path",
        lambda label, *, backend: tmp_path / f"{label}.{backend}",
    )
    monkeypatch.setattr(wechat_arg_module.Log, "error", lambda message, **_kwargs: errors.append(str(message)))

    class _Loop:
        def remove_signal_handler(self, os_signal: signal.Signals) -> None:
            removed_signals.append(os_signal)
            if os_signal == signal.SIGTERM:
                raise RuntimeError("ignore")

    captured_environment = wechat_runtime_module._collect_service_environment_variables(context)
    identity = wechat_runtime_module._resolve_service_identity(args)

    assert captured_environment == {"MIMO_API_KEY": "secret-1"}
    assert identity.instance_label == "default"
    assert identity.label == "com.dayu.wechat.test"
    assert identity.definition_path == (tmp_path / "com.dayu.wechat.test.launchd")
    from dayu.cli.arg_parsing import parse_limits_override, parse_temperature_argument
    with pytest.raises(SystemExit, match="2"):
        parse_limits_override("{bad json}", field_name="--doc-limits-json")
    with pytest.raises(SystemExit, match="2"):
        parse_limits_override('[1, 2]', field_name="--doc-limits-json")
    with pytest.raises(SystemExit, match="2"):
        parse_limits_override('{"nested": []}', field_name="--doc-limits-json")
    with pytest.raises(SystemExit, match="2"):
        parse_temperature_argument("bad", field_name="--temperature")
    wechat_run_module._remove_daemon_signal_handlers(_Loop(), [signal.SIGINT, signal.SIGTERM])

    assert any("不是合法 JSON" in message for message in errors)
    assert any("必须是 JSON 对象" in message for message in errors)
    assert any("只允许 JSON 标量值" in message for message in errors)
    assert any("--temperature" in message for message in errors)
    assert removed_signals == [signal.SIGINT, signal.SIGTERM]


@pytest.mark.unit
def test_resolve_service_identity_prefers_installed_definition_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 service 身份解析优先复用已安装 definition 的真实 label。"""

    parser = wechat_arg_module._create_parser()
    args = parser.parse_args(["service", "uninstall", "--base", str(tmp_path)])
    expected_state_dir = (tmp_path / ".dayu" / "wechat-default").resolve()

    monkeypatch.setattr(wechat_runtime_module, "detect_service_backend", lambda: "launchd")
    monkeypatch.setattr(
        wechat_runtime_module,
        "list_installed_service_definitions",
        lambda _backend: (
            InstalledServiceDefinition(
                backend="launchd",
                label="com.dayu.wechat.legacy-default",
                definition_path=tmp_path / "legacy-default.plist",
                program_arguments=(
                    "/usr/bin/python3",
                    "-m",
                    "dayu.wechat",
                    "run",
                    "--base",
                    str(tmp_path),
                ),
            ),
        ),
    )
    monkeypatch.setattr(wechat_runtime_module, "build_service_label", lambda _state_dir: "com.dayu.wechat.recomputed")

    identity = wechat_runtime_module._resolve_service_identity(args)

    assert identity.instance_label == "default"
    assert identity.state_dir == expected_state_dir
    assert identity.label == "com.dayu.wechat.legacy-default"
    assert identity.definition_path == (tmp_path / "legacy-default.plist")


@pytest.mark.unit
def test_wechat_main_helper_functions_cover_service_dispatch_and_log_levels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 service stop/uninstall、主命令 dispatch 与日志级别设置。"""

    parser = wechat_arg_module._create_parser()
    stop_args = parser.parse_args(["service", "stop", "--base", str(tmp_path)])
    uninstall_args = parser.parse_args(["service", "uninstall", "--base", str(tmp_path)])
    identity = wechat_runtime_module.ResolvedWechatServiceIdentity(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "com.dayu.wechat.test.plist",
        state_dir=tmp_path / ".wechat",
    )
    printed: list[str] = []
    set_levels: list[object] = []
    stop_results = iter([True, False])
    uninstall_results = iter([True, False])

    monkeypatch.setattr(wechat_service_module, "_resolve_service_identity", lambda _args: identity)
    monkeypatch.setattr(wechat_service_module, "stop_service", lambda **_kwargs: next(stop_results))
    monkeypatch.setattr(wechat_service_module, "uninstall_service", lambda **_kwargs: next(uninstall_results))
    monkeypatch.setattr(builtins, "print", lambda message: printed.append(message))
    monkeypatch.setattr(wechat_arg_module.Log, "set_level", lambda level: set_levels.append(level))

    assert wechat_service_module._run_service_stop_command(stop_args) == 0
    assert wechat_service_module._run_service_stop_command(stop_args) == 0
    assert wechat_service_module._run_service_uninstall_command(uninstall_args) == 0
    assert wechat_service_module._run_service_uninstall_command(uninstall_args) == 0

    monkeypatch.setattr(wechat_service_module, "_run_service_install_command", lambda _args: 11)
    monkeypatch.setattr(wechat_service_module, "_run_service_start_command", lambda _args: 12)
    monkeypatch.setattr(wechat_service_module, "_run_service_restart_command", lambda _args: 13)
    monkeypatch.setattr(wechat_service_module, "_run_service_stop_command", lambda _args: 14)
    monkeypatch.setattr(wechat_service_module, "_run_service_status_command", lambda _args: 15)
    monkeypatch.setattr(wechat_service_module, "_run_service_list_command", lambda _args: 16)
    monkeypatch.setattr(wechat_service_module, "_run_service_uninstall_command", lambda _args: 17)

    assert wechat_service_module.run_service_command(SimpleNamespace(service_command="install")) == 11
    assert wechat_service_module.run_service_command(SimpleNamespace(service_command="start")) == 12
    assert wechat_service_module.run_service_command(SimpleNamespace(service_command="restart")) == 13
    assert wechat_service_module.run_service_command(SimpleNamespace(service_command="stop")) == 14
    assert wechat_service_module.run_service_command(SimpleNamespace(service_command="status")) == 15
    assert wechat_service_module.run_service_command(SimpleNamespace(service_command="list")) == 16
    assert wechat_service_module.run_service_command(SimpleNamespace(service_command="uninstall")) == 17
    with pytest.raises(ValueError, match="未知 service 子命令"):
        wechat_service_module.run_service_command(SimpleNamespace(service_command="bad"))

    monkeypatch.setattr("dayu.wechat.commands.login.run_login_command", lambda _args: 21)
    monkeypatch.setattr("dayu.wechat.commands.run.run_run_command", lambda _args: 22)
    monkeypatch.setattr("dayu.wechat.commands.service.run_service_command", lambda _args: 23)

    assert wechat_main_module._dispatch_command(SimpleNamespace(command="login")) == 21
    assert wechat_main_module._dispatch_command(SimpleNamespace(command="run")) == 22
    assert wechat_main_module._dispatch_command(SimpleNamespace(command="service")) == 23
    with pytest.raises(ValueError, match="未知命令"):
        wechat_main_module._dispatch_command(SimpleNamespace(command="bad"))

    for namespace in (
        SimpleNamespace(log_level="debug", debug=False, verbose=False, info=False, quiet=False),
        SimpleNamespace(log_level=None, debug=True, verbose=False, info=False, quiet=False),
        SimpleNamespace(log_level=None, debug=False, verbose=True, info=False, quiet=False),
        SimpleNamespace(log_level=None, debug=False, verbose=False, info=True, quiet=False),
        SimpleNamespace(log_level=None, debug=False, verbose=False, info=False, quiet=True),
        SimpleNamespace(log_level=None, debug=False, verbose=False, info=False, quiet=False),
    ):
        wechat_arg_module.setup_loglevel(namespace)

    assert printed == [
        "已停止 macOS launchd 服务实例: default",
        "macOS launchd 服务实例未运行: default",
        "已卸载 macOS launchd 服务实例: default",
        "macOS launchd 服务实例尚未安装: default",
    ]
    assert set_levels == [
        wechat_arg_module.LogLevel.DEBUG,
        wechat_arg_module.LogLevel.DEBUG,
        wechat_arg_module.LogLevel.VERBOSE,
        wechat_arg_module.LogLevel.INFO,
        wechat_arg_module.LogLevel.ERROR,
        wechat_arg_module.LogLevel.INFO,
    ]
