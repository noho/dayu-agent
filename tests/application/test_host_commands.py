"""host_commands 测试。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dayu.cli.arg_parsing import DayuCliArgumentParser, _register_host_subcommands
from dayu.cli.commands import host as host_commands_module
from dayu.cli.commands.host import _build_host_runtime
from dayu.host.executor import DefaultHostExecutor
from dayu.host.host import Host
from dayu.host.host_store import HostStore
from dayu.host.session_registry import SQLiteSessionRegistry
from dayu.services.contracts import RunAdminView, SessionAdminView
from dayu.services.host_admin_service import HostAdminService
from dayu.services.protocols import HostAdminServiceProtocol


def _host_admin_service(runtime: Any) -> HostAdminService:
    """把 host runtime 收窄为测试所需的 HostAdminService。"""

    return cast(HostAdminService, runtime.host_admin_service)


def _host(service: HostAdminService) -> Host:
    """把 HostAdminService 收窄为测试所需的 Host。"""

    return cast(Host, service.host)


def _executor(host: Host) -> DefaultHostExecutor:
    """收窄 Host 内部 executor 到默认实现。"""

    return cast(DefaultHostExecutor, host._executor)


def _session_registry(host: Host) -> SQLiteSessionRegistry:
    """收窄 Host 内部 session registry 到 SQLite 实现。"""

    return cast(SQLiteSessionRegistry, host._session_registry)


def _host_store(session_registry: SQLiteSessionRegistry) -> HostStore:
    """收窄 session registry 内部 host store。"""

    return session_registry._host_store


def _write_minimal_startup_config(config_root: Path, *, run_config_text: str) -> None:
    """写入最小可启动配置。

    Args:
        config_root: 配置目录。
        run_config_text: `run.json` 文本。

    Returns:
        无。

    Raises:
        OSError: 文件写入失败时抛出。
    """

    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "run.json").write_text(run_config_text, encoding="utf-8")
    (config_root / "llm_models.json").write_text(
        '{"defaults":{"temperature":0.2},"models":{"default":{"runner_type":"openai_compatible","endpoint_url":"http://example.com","model":"test","headers":{},"max_context_tokens":32000}}}',
        encoding="utf-8",
    )


@pytest.mark.unit
def test_register_host_subcommands_can_attach_global_args() -> None:
    """宿主管理子命令应保留全局参数字段。"""

    parser = DayuCliArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    _register_host_subcommands(subparsers)

    host_args = parser.parse_args(["host", "status"])
    sessions_args = parser.parse_args(["sessions"])
    filtered_sessions_args = parser.parse_args(["sessions", "--source", "cli", "--scene", "interactive"])

    assert host_args.base == "./workspace"
    assert host_args.log_level is None
    assert sessions_args.base == "./workspace"
    assert sessions_args.config is None
    assert filtered_sessions_args.source == "cli"
    assert filtered_sessions_args.scene == "interactive"


@pytest.mark.unit
def test_build_host_runtime_returns_startup_dependencies(tmp_path: Path) -> None:
    """宿主管理命令应通过稳定输入拿到 Host 默认装配结果。"""

    workspace_root = (tmp_path / "workspace").resolve()
    config_root = (workspace_root / "config").resolve()
    _write_minimal_startup_config(
        config_root,
        run_config_text=(
            '{"host_config":{"store":{"path":".dayu/host/dayu_host.db"},'
            '"lane":{"llm_api":2,"sec_download":3}}}'
        ),
    )

    args = argparse.Namespace(
        base=str(workspace_root),
        config=str(config_root),
    )

    runtime = _build_host_runtime(args)
    host = _host(_host_admin_service(runtime))
    executor = _executor(host)
    session_registry = _session_registry(host)

    assert runtime.paths.workspace_root == workspace_root
    assert runtime.paths.config_root == config_root
    assert session_registry is not None
    assert host._concurrency_governor is not None
    assert executor is not None
    assert executor.scene_preparation is None
    assert host._concurrency_governor.get_lane_status("llm_api").max_concurrent == 2
    assert host._concurrency_governor.get_lane_status("sec_download").max_concurrent == 3
    assert _host_store(session_registry).db_path == (workspace_root / ".dayu" / "host" / "dayu_host.db").resolve()


@pytest.mark.unit
def test_host_default_assembly_builds_internal_sqlite_components(tmp_path: Path) -> None:
    """Host 默认构造应在内部装配 SQLite registry 与默认 executor。"""

    workspace_root = (tmp_path / "workspace").resolve()
    config_root = (workspace_root / "config").resolve()
    _write_minimal_startup_config(
        config_root,
        run_config_text=(
            '{"host_config":{"store":{"path":".dayu/host/dayu_host.db"},'
            '"lane":{"llm_api":2,"sec_download":3}}}'
        ),
    )

    args = argparse.Namespace(
        base=str(workspace_root),
        config=str(config_root),
    )

    runtime = _build_host_runtime(args)
    host = _host(_host_admin_service(runtime))

    assert _session_registry(host).__class__.__name__ == "SQLiteSessionRegistry"
    assert host._run_registry.__class__.__name__ == "SQLiteRunRegistry"
    assert host._concurrency_governor.__class__.__name__ == "SQLiteConcurrencyGovernor"
    assert _executor(host).__class__.__name__ == "DefaultHostExecutor"


@pytest.mark.unit
def test_build_host_runtime_uses_default_host_store_path_when_missing_config(tmp_path: Path) -> None:
    """宿主管理命令未配置 `host_config.store.path` 时应回退到默认路径。"""

    workspace_root = (tmp_path / "workspace").resolve()
    config_root = (workspace_root / "config").resolve()
    _write_minimal_startup_config(
        config_root,
        run_config_text='{"host_config":{"lane":{"llm_api":2,"sec_download":3}}}',
    )

    args = argparse.Namespace(
        base=str(workspace_root),
        config=str(config_root),
    )

    runtime = _build_host_runtime(args)
    host = _host(_host_admin_service(runtime))

    assert _host_store(_session_registry(host)).db_path == (workspace_root / ".dayu" / "host" / "dayu_host.db").resolve()


@pytest.mark.unit
def test_build_host_runtime_resolves_custom_relative_host_store_path(tmp_path: Path) -> None:
    """宿主管理命令应把相对 `host_config.store.path` 解析到 workspace 下。"""

    workspace_root = (tmp_path / "workspace").resolve()
    config_root = (workspace_root / "config").resolve()
    _write_minimal_startup_config(
        config_root,
        run_config_text=(
            '{"host_config":{"store":{"path":"data/runtime/host.sqlite3"},'
            '"lane":{"llm_api":2,"sec_download":3}}}'
        ),
    )

    args = argparse.Namespace(
        base=str(workspace_root),
        config=str(config_root),
    )

    runtime = _build_host_runtime(args)
    host = _host(_host_admin_service(runtime))

    assert _host_store(_session_registry(host)).db_path == (workspace_root / "data" / "runtime" / "host.sqlite3").resolve()


@pytest.mark.unit
def test_run_runs_command_does_not_render_ticker_column(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`host runs` 输出不应再依赖 ticker 列。"""

    run = RunAdminView(
        run_id="run_test",
        session_id="session_test",
        service_type="prompt",
        scene_name="prompt",
        state="succeeded",
        cancel_requested_at=None,
        cancel_requested_reason=None,
        cancel_reason=None,
        created_at="2026-04-03T08:00:00+00:00",
        started_at="2026-04-03T08:01:00+00:00",
        finished_at="2026-04-03T08:02:00+00:00",
        error_summary=None,
    )
    fake_runtime = SimpleNamespace(
        host_admin_service=SimpleNamespace(
            list_runs=lambda **_kwargs: [run],
        )
    )
    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: fake_runtime)

    exit_code = host_commands_module._run_runs_command(
        argparse.Namespace(
            show_all=True,
            session_id=None,
            base="./workspace",
            config=None,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "TICKER" not in captured.out
    assert "AAPL" not in captured.out
    assert "RUN_ID" in captured.out
    assert "SERVICE" in captured.out


@pytest.mark.unit
def test_run_sessions_command_does_not_render_ticker_column(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`host sessions` 输出不应再依赖 ticker 列。"""

    session = SessionAdminView(
        session_id="session_test",
        source="web",
        state="active",
        scene_name="interactive",
        created_at="2026-04-03T08:00:00+00:00",
        last_activity_at="2026-04-03T08:05:00+00:00",
    )
    fake_runtime = SimpleNamespace(
        host_admin_service=SimpleNamespace(
            list_sessions=lambda **_kwargs: [session],
        )
    )
    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: fake_runtime)

    exit_code = host_commands_module._run_sessions_command(
        argparse.Namespace(
            show_all=True,
            sessions_action=None,
            source=None,
            scene=None,
            base="./workspace",
            config=None,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "TICKER" not in captured.out
    assert "SESSION_ID" in captured.out
    assert "SOURCE" in captured.out
    assert "SCENE" in captured.out
    assert "TURNS" in captured.out
    assert "OVERVIEW" in captured.out


@pytest.mark.unit
def test_run_sessions_command_renders_generic_session_digest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`sessions` 应展示通用会话摘要视图。"""

    session = SessionAdminView(
        session_id="interactive_test",
        source="cli",
        state="active",
        scene_name="interactive",
        created_at="2026-04-03T08:00:00+00:00",
        last_activity_at="2026-04-03T08:05:00+00:00",
        turn_count=2,
        first_question_preview="第一问",
        last_question_preview="最后一问",
    )
    list_calls: list[dict[str, object | None]] = []
    fake_runtime = SimpleNamespace(
        host_admin_service=SimpleNamespace(
            list_sessions=lambda **kwargs: list_calls.append(kwargs) or [session],
        )
    )
    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: fake_runtime)

    exit_code = host_commands_module._run_sessions_command(
        argparse.Namespace(
            show_all=True,
            sessions_action=None,
            source="cli",
            scene="interactive",
            base="./workspace",
            config=None,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert list_calls == [{"state": None, "source": "cli", "scene": "interactive"}]
    assert "SESSION_ID" in captured.out
    assert "SOURCE" in captured.out
    assert "SCENE" in captured.out
    assert "TURNS" in captured.out
    assert "OVERVIEW" in captured.out
    assert "interactive_test" in captured.out
    assert "cli" in captured.out
    assert "interactive" in captured.out
    assert "第一问" in captured.out
    assert "最后一问" not in captured.out


@pytest.mark.unit
def test_run_sessions_command_filters_active_sessions_by_default(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`sessions` 未传 `--all` 时应默认只请求 active 会话。"""

    list_calls: list[dict[str, object | None]] = []
    fake_runtime = SimpleNamespace(
        host_admin_service=SimpleNamespace(
            list_sessions=lambda **kwargs: list_calls.append(kwargs) or [],
        )
    )
    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: fake_runtime)

    exit_code = host_commands_module._run_sessions_command(
        argparse.Namespace(
            show_all=False,
            sessions_action=None,
            source="cli",
            scene="prompt_mt",
            base="./workspace",
            config=None,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert list_calls == [{"state": "active", "source": "cli", "scene": "prompt_mt"}]
    assert "无会话记录" in captured.out


@pytest.mark.unit
def test_run_host_command_dispatches_known_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证顶层命令分发会路由到对应处理函数。"""

    monkeypatch.setattr(host_commands_module, "setup_loglevel", lambda _args: None)
    monkeypatch.setattr(host_commands_module, "_run_sessions_command", lambda _args: 11)
    monkeypatch.setattr(host_commands_module, "_run_runs_command", lambda _args: 22)
    monkeypatch.setattr(host_commands_module, "_run_cancel_command", lambda _args: 33)
    monkeypatch.setattr(host_commands_module, "_run_host_command", lambda _args: 44)

    assert host_commands_module.run_host_command(argparse.Namespace(command="sessions")) == 11
    assert host_commands_module.run_host_command(argparse.Namespace(command="runs")) == 22
    assert host_commands_module.run_host_command(argparse.Namespace(command="cancel")) == 33
    assert host_commands_module.run_host_command(argparse.Namespace(command="host")) == 44
    assert host_commands_module.run_host_command(argparse.Namespace(command="unknown")) == 1


@pytest.mark.unit
def test_run_host_command_configures_loglevel_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证宿主管理命令入口会先配置日志级别。"""

    setup_calls: list[argparse.Namespace] = []
    monkeypatch.setattr(host_commands_module, "setup_loglevel", lambda args: setup_calls.append(args))
    monkeypatch.setattr(host_commands_module, "_run_sessions_command", lambda _args: 11)

    args = argparse.Namespace(command="sessions")

    assert host_commands_module.run_host_command(args) == 11
    assert setup_calls == [args]


@pytest.mark.unit
def test_resolve_host_admin_service_prefers_runtime_service() -> None:
    """验证运行时已提供 host_admin_service 时直接复用。"""

    service = cast(HostAdminServiceProtocol, SimpleNamespace(name="service"))

    resolved = host_commands_module._resolve_host_admin_service(
        cast(host_commands_module._HostCliRuntimeLike, SimpleNamespace(host_admin_service=service))
    )

    assert resolved is service


@pytest.mark.unit
def test_resolve_host_admin_service_requires_service() -> None:
    """验证运行时缺少 host_admin_service 时抛出错误。"""

    with pytest.raises(AttributeError, match="host_admin_service"):
        host_commands_module._resolve_host_admin_service(
            cast(host_commands_module._HostCliRuntimeLike, SimpleNamespace())
        )


@pytest.mark.unit
def test_resolve_host_admin_service_does_not_fallback_to_host() -> None:
    """验证 CLI 宿主管理命令不会回退到直接消费 Host。"""

    with pytest.raises(AttributeError, match="host_admin_service"):
        host_commands_module._resolve_host_admin_service(
            cast(host_commands_module._HostCliRuntimeLike, SimpleNamespace(host=SimpleNamespace()))
        )


@pytest.mark.unit
def test_run_cancel_command_supports_session_scope(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 cancel --session 会调用批量取消分支。"""

    fake_runtime = SimpleNamespace(
        host_admin_service=SimpleNamespace(
            get_session=lambda _session_id: SimpleNamespace(session_id=_session_id),
            cancel_session_runs=lambda _session_id: ["run-1", "run-2"],
        )
    )
    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: fake_runtime)

    exit_code = host_commands_module._run_cancel_command(
        argparse.Namespace(
            session_id="session-1",
            run_id=None,
            base="./workspace",
            config=None,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "2 个 run" in captured.out


@pytest.mark.unit
def test_run_cancel_command_session_not_found_returns_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 cancel --session 对不存在 session_id 返回非零退出码并打印错误。"""

    fake_runtime = SimpleNamespace(
        host_admin_service=SimpleNamespace(
            get_session=lambda _session_id: None,
            cancel_session_runs=lambda _session_id: pytest.fail("不应进入批量取消分支"),
        )
    )
    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: fake_runtime)

    exit_code = host_commands_module._run_cancel_command(
        argparse.Namespace(
            session_id="session-missing",
            run_id=None,
            base="./workspace",
            config=None,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "session session-missing 不存在" in captured.err


@pytest.mark.unit
def test_run_cancel_command_requires_run_or_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 cancel 缺少 run_id 和 session 时返回错误。"""

    fake_runtime = SimpleNamespace(host_admin_service=SimpleNamespace())
    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: fake_runtime)

    exit_code = host_commands_module._run_cancel_command(
        argparse.Namespace(
            session_id=None,
            run_id=None,
            base="./workspace",
            config=None,
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "请指定 run_id 或 --session" in captured.err


@pytest.mark.unit
def test_run_cancel_command_handles_missing_or_inactive_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 cancel run 的缺失和非活跃分支。"""

    service = SimpleNamespace()

    def _cancel_missing(_run_id: str) -> Any:
        """模拟缺失 run。"""

        raise KeyError("missing")

    service.cancel_run = _cancel_missing
    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: SimpleNamespace(host_admin_service=service))

    missing_code = host_commands_module._run_cancel_command(
        argparse.Namespace(session_id=None, run_id="run-404", base="./workspace", config=None)
    )
    missing_output = capsys.readouterr()

    service.cancel_run = lambda _run_id: SimpleNamespace(state="running")
    inactive_code = host_commands_module._run_cancel_command(
        argparse.Namespace(session_id=None, run_id="run-1", base="./workspace", config=None)
    )
    inactive_output = capsys.readouterr()

    assert missing_code == 1
    assert "run run-404 不存在" in missing_output.err
    assert inactive_code == 1
    assert "无法取消" in inactive_output.err


@pytest.mark.unit
def test_run_cancel_command_succeeds_for_cancelled_run(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 cancel run 成功分支。"""

    fake_runtime = SimpleNamespace(
        host_admin_service=SimpleNamespace(
            cancel_run=lambda _run_id: SimpleNamespace(state="cancelled"),
        )
    )
    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: fake_runtime)

    exit_code = host_commands_module._run_cancel_command(
        argparse.Namespace(session_id=None, run_id="run-1", base="./workspace", config=None)
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "已请求取消 run run-1" in captured.out


@pytest.mark.unit
def test_run_host_command_requires_action(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 host 子命令缺少 action 时输出用法。"""

    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: SimpleNamespace())

    exit_code = host_commands_module._run_host_command(
        argparse.Namespace(host_action=None, base="./workspace", config=None)
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "dayu host {cleanup|status}" in captured.err


@pytest.mark.unit
def test_run_host_command_cleanup_and_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 host cleanup 与 status 的输出分支。"""

    status = SimpleNamespace(
        active_session_count=2,
        total_session_count=5,
        active_run_count=3,
        active_runs_by_type={"prompt": 2, "audit": 1},
        lane_statuses={
            "llm_api": SimpleNamespace(active=1, max_concurrent=4),
        },
    )
    service = SimpleNamespace(
        cleanup=lambda: SimpleNamespace(orphan_run_ids=["run-1"], stale_permit_ids=["permit-1", "permit-2"]),
        get_status=lambda: status,
    )
    monkeypatch.setattr(host_commands_module, "_build_host_runtime", lambda _args: SimpleNamespace(host_admin_service=service))

    cleanup_code = host_commands_module._run_host_command(
        argparse.Namespace(host_action="cleanup", base="./workspace", config=None)
    )
    cleanup_output = capsys.readouterr()
    status_code = host_commands_module._run_host_command(
        argparse.Namespace(host_action="status", base="./workspace", config=None)
    )
    status_output = capsys.readouterr()

    assert cleanup_code == 0
    assert "1 个孤儿 run, 2 个过期 permit" in cleanup_output.out
    assert status_code == 0
    assert "活跃会话: 2 / 总计: 5" in status_output.out
    assert "prompt: 2" in status_output.out
    assert "llm_api: 1/4" in status_output.out


@pytest.mark.unit
def test_run_host_command_returns_one_for_unknown_action(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证未知 host action 返回 1。"""

    monkeypatch.setattr(
        host_commands_module,
        "_build_host_runtime",
        lambda _args: SimpleNamespace(host_admin_service=SimpleNamespace()),
    )

    assert host_commands_module._run_host_command(
        argparse.Namespace(host_action="unknown", base="./workspace", config=None)
    ) == 1


@pytest.mark.unit
def test_format_helpers_cover_invalid_and_elapsed_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证时间格式化辅助函数覆盖异常和分钟分支。"""

    class _FrozenDatetime(datetime):
        """固定当前时间，方便验证未完成 run 的时长。"""

        @classmethod
        def now(cls, tz: object = None) -> datetime:
            """返回固定当前时间。"""

            del tz
            return datetime(2026, 4, 3, 8, 2, 30, tzinfo=timezone.utc)

    monkeypatch.setattr(host_commands_module, "datetime", _FrozenDatetime)

    assert host_commands_module._format_datetime_iso(None) == "-"
    assert host_commands_module._format_datetime_iso("not-iso") == "not-iso"
    assert host_commands_module._format_duration_iso(None, None) == "-"
    assert host_commands_module._format_duration_iso(
        "2026-04-03T08:00:00+00:00",
        "2026-04-03T08:01:05+00:00",
    ) == "1m5s"
    assert host_commands_module._format_duration_iso(
        "2026-04-03T08:00:00+00:00",
        None,
    ) == "2m30s"


@pytest.mark.unit
def test_default_host_assembly_shares_archive_store_with_scene_preparer(tmp_path: Path) -> None:
    """Host 默认装配应让 ScenePreparer 与 Host 本体持有同一 archive_store 实例。

    回归 review 条目 080：此前 `DefaultScenePreparer.__post_init__` 会在未收到外部
    archive_store 时自建实例，导致 Host 与其内部 ScenePreparer
    指向同一目录的两个独立实例，未来引入 transcript 内存缓存会触发 stale read。
    """

    from dayu.execution.options import ResolvedExecutionOptions, TraceSettings
    from dayu.execution.runtime_config import (
        AgentRuntimeConfig,
        FallbackMode,
        OpenAIRunnerRuntimeConfig,
    )
    from dayu.host.host import _build_default_host_components
    from dayu.host.conversation_store import FileConversationSessionArchiveStore
    from dayu.host.scene_preparer import DefaultScenePreparer
    from dayu.workspace_paths import build_conversation_store_dir

    workspace_root = (tmp_path / "workspace").resolve()
    workspace = cast(
        Any,
        SimpleNamespace(
            workspace_dir=workspace_root,
            prompt_asset_store=object(),
        ),
    )
    resolved = ResolvedExecutionOptions(
        model_name="default",
        runner_running_config=OpenAIRunnerRuntimeConfig(),
        agent_running_config=AgentRuntimeConfig(fallback_mode=FallbackMode.FORCE_ANSWER),
        trace_settings=TraceSettings(enabled=False, output_dir=tmp_path / "trace"),
        temperature=None,
    )
    shared_store = FileConversationSessionArchiveStore(build_conversation_store_dir(workspace_root))
    components = _build_default_host_components(
        workspace=workspace,
        model_catalog=cast(Any, object()),
        default_execution_options=resolved,
        host_store_path=tmp_path / "host.sqlite3",
        lane_config={"llm_api": 1},
        event_bus=None,
        archive_store=shared_store,
    )

    executor = components._executor
    scene_preparation = cast(DefaultScenePreparer, executor.scene_preparation)
    assert scene_preparation is not None
    assert scene_preparation._archive_store_impl is shared_store
