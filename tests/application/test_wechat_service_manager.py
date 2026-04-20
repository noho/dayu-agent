"""WeChat 系统 service manager 测试。"""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from dayu.wechat import service_manager

# WeChat service 仅支持 macOS (launchd) 与 Linux (systemd)；Windows 上 service_manager
# 依赖 os.getuid() 等 POSIX 专属 API，不参与运行或测试。
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="WeChat service 不支持 Windows")


@pytest.mark.unit
def test_detect_service_backend_supports_linux() -> None:
    """验证 Linux 平台会路由到 systemd backend。"""

    assert service_manager.detect_service_backend("Linux") == "systemd"


@pytest.mark.unit
def test_build_service_label_is_stable(tmp_path: Path) -> None:
    """验证相同状态目录会生成稳定 label。"""

    state_dir = tmp_path / ".wechat"

    first = service_manager.build_service_label(state_dir)
    second = service_manager.build_service_label(state_dir)

    assert first == second
    assert first.startswith("com.dayu.wechat.")


@pytest.mark.unit
def test_build_launchd_service_spec_builds_expected_paths(tmp_path: Path) -> None:
    """验证 launchd spec 会生成 plist 与日志路径。"""

    state_dir = tmp_path / ".wechat"
    working_directory = tmp_path / "repo"

    spec = service_manager.build_launchd_service_spec(
        state_dir=state_dir,
        working_directory=working_directory,
        python_executable="/usr/bin/python3",
        run_arguments=["run", "--base", "/tmp/workspace"],
        environment_variables={"MIMO_API_KEY": "secret-1"},
    )

    assert spec.backend == "launchd"
    assert spec.label.startswith("com.dayu.wechat.")
    assert spec.definition_path.name == f"{spec.label}.plist"
    assert spec.program_arguments == (
        "/usr/bin/python3",
        "-m",
        "dayu.wechat",
        "run",
        "--base",
        "/tmp/workspace",
    )
    assert spec.environment_variables == (("MIMO_API_KEY", "secret-1"),)
    assert spec.stdout_path == state_dir.resolve() / "logs" / "launchd.stdout.log"
    assert spec.stderr_path == state_dir.resolve() / "logs" / "launchd.stderr.log"


@pytest.mark.unit
def test_build_systemd_service_spec_builds_expected_unit_path(tmp_path: Path) -> None:
    """验证 systemd spec 会生成 user unit 路径。"""

    state_dir = tmp_path / ".wechat"
    working_directory = tmp_path / "repo"

    spec = service_manager.build_systemd_service_spec(
        state_dir=state_dir,
        working_directory=working_directory,
        python_executable="/usr/bin/python3",
        run_arguments=["run", "--base", "/tmp/workspace"],
        environment_variables={"MIMO_API_KEY": "secret-1"},
    )

    assert spec.backend == "systemd"
    assert spec.label.startswith("com.dayu.wechat.")
    assert spec.definition_path.name == f"{spec.label}.service"
    assert spec.environment_variables == (("MIMO_API_KEY", "secret-1"),)
    assert spec.stdout_path is None
    assert spec.stderr_path is None


@pytest.mark.unit
def test_list_installed_launchd_service_definitions_reads_program_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 launchd definition 枚举会从 plist 读取 ProgramArguments。"""

    launch_agents_dir = tmp_path / "LaunchAgents"
    plist_path = launch_agents_dir / "com.dayu.wechat.alpha.plist"
    launch_agents_dir.mkdir(parents=True)
    with plist_path.open("wb") as handle:
        plistlib.dump(
            {
                "Label": "com.dayu.wechat.alpha",
                "WorkingDirectory": str(tmp_path / "repo"),
                "ProgramArguments": [
                    "/usr/bin/python3",
                    "-m",
                    "dayu.wechat",
                    "run",
                    "--base",
                    "/tmp/workspace",
                    "--label",
                    "alpha",
                ],
            },
            handle,
        )

    monkeypatch.setattr(
        service_manager,
        "resolve_launch_agent_plist_path",
        lambda label: launch_agents_dir / f"{label}.plist",
    )

    definitions = service_manager.list_installed_service_definitions("launchd")

    assert definitions == (
        service_manager.InstalledServiceDefinition(
            backend="launchd",
            label="com.dayu.wechat.alpha",
            definition_path=plist_path.resolve(),
            program_arguments=(
                "/usr/bin/python3",
                "-m",
                "dayu.wechat",
                "run",
                "--base",
                "/tmp/workspace",
                "--label",
                "alpha",
            ),
            working_directory=(tmp_path / "repo").resolve(),
        ),
    )


@pytest.mark.unit
def test_list_installed_systemd_service_definitions_reads_exec_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 systemd definition 枚举会从 unit 读取 ExecStart。"""

    unit_dir = tmp_path / "systemd"
    unit_path = unit_dir / "com.dayu.wechat.alpha.service"
    unit_dir.mkdir(parents=True)
    unit_path.write_text(
        "\n".join(
            [
                "[Service]",
                f"WorkingDirectory={tmp_path / 'repo'}",
                'ExecStart=/usr/bin/python3 -m dayu.wechat run --base /tmp/workspace --label alpha',
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        service_manager,
        "resolve_systemd_user_unit_path",
        lambda label: unit_dir / f"{label}.service",
    )

    definitions = service_manager.list_installed_service_definitions("systemd")

    assert definitions == (
        service_manager.InstalledServiceDefinition(
            backend="systemd",
            label="com.dayu.wechat.alpha",
            definition_path=unit_path.resolve(),
            program_arguments=(
                "/usr/bin/python3",
                "-m",
                "dayu.wechat",
                "run",
                "--base",
                "/tmp/workspace",
                "--label",
                "alpha",
            ),
            working_directory=(tmp_path / "repo").resolve(),
        ),
    )


@pytest.mark.unit
def test_query_launchd_service_status_returns_not_installed(tmp_path: Path) -> None:
    """验证 plist 不存在时 launchd 状态为未安装。"""

    status = service_manager.query_launchd_service_status(
        label="com.dayu.wechat.test",
        plist_path=tmp_path / "missing.plist",
    )

    assert status.installed is False
    assert status.loaded is False
    assert status.pid is None


@pytest.mark.unit
def test_query_launchd_service_status_extracts_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `launchctl print` 输出中的 pid 会被解析出来。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")

    def _fake_run_launchctl(_arguments, *, check: bool):
        assert check is False
        return subprocess.CompletedProcess(
            args=["launchctl"],
            returncode=0,
            stdout="service = { pid = 123; state = running; }",
            stderr="",
        )

    monkeypatch.setattr(service_manager, "_run_launchctl", _fake_run_launchctl)

    status = service_manager.query_launchd_service_status(
        label="com.dayu.wechat.test",
        plist_path=plist_path,
    )

    assert status.installed is True
    assert status.loaded is True
    assert status.pid == 123


@pytest.mark.unit
def test_query_systemd_service_status_extracts_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 `systemctl --user show` 输出中的 MainPID 会被解析出来。"""

    unit_path = tmp_path / "service.service"
    unit_path.write_text("[Service]\n", encoding="utf-8")

    def _fake_run_systemctl(_arguments, *, check: bool):
        assert check is False
        return subprocess.CompletedProcess(
            args=["systemctl", "--user"],
            returncode=0,
            stdout="LoadState=loaded\nActiveState=active\nMainPID=456\n",
            stderr="",
        )

    monkeypatch.setattr(service_manager, "_run_systemctl_user", _fake_run_systemctl)

    status = service_manager.query_systemd_service_status(
        label="com.dayu.wechat.test",
        unit_path=unit_path,
    )

    assert status.installed is True
    assert status.loaded is True
    assert status.pid == 456


@pytest.mark.unit
def test_is_service_running_returns_true_for_launchd_with_pid(tmp_path: Path) -> None:
    """验证 launchd 仅在存在 pid 时才视为运行中。"""

    status = service_manager.ServiceStatus(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "service.plist",
        installed=True,
        loaded=True,
        pid=123,
    )

    assert service_manager.is_service_running(status) is True


@pytest.mark.unit
def test_is_service_running_returns_false_for_launchd_without_pid(tmp_path: Path) -> None:
    """验证 launchd 即使 loaded 但无 pid，仍要允许上层继续走启动恢复。"""

    status = service_manager.ServiceStatus(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "service.plist",
        installed=True,
        loaded=True,
        pid=None,
    )

    assert service_manager.is_service_running(status) is False


@pytest.mark.unit
def test_is_service_running_returns_true_for_loaded_systemd(tmp_path: Path) -> None:
    """验证 systemd 的 loaded 运行态按后端语义直接视为运行中。"""

    status = service_manager.ServiceStatus(
        backend="systemd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "service.service",
        installed=True,
        loaded=True,
        pid=None,
    )

    assert service_manager.is_service_running(status) is True


@pytest.mark.unit
def test_install_launchd_service_writes_plist(tmp_path: Path) -> None:
    """验证安装 launchd service 时会写入 plist 文件。"""

    plist_path = tmp_path / "LaunchAgents" / "com.dayu.wechat.test.plist"
    spec = service_manager.ServiceSpec(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=plist_path,
        working_directory=tmp_path / "repo",
        program_arguments=("/usr/bin/python3", "-m", "dayu.wechat", "run"),
        environment_variables=(("MIMO_API_KEY", "secret-1"),),
        stdout_path=tmp_path / "logs" / "stdout.log",
        stderr_path=tmp_path / "logs" / "stderr.log",
    )

    written_path = service_manager.install_launchd_service(spec)

    assert written_path == plist_path
    assert plist_path.exists()
    payload = __import__("plistlib").load(plist_path.open("rb"))
    assert payload["EnvironmentVariables"]["MIMO_API_KEY"] == "secret-1"


@pytest.mark.unit
def test_install_systemd_service_writes_unit_and_reload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证安装 systemd service 时会写入 unit 文件并执行 daemon-reload。"""

    unit_path = tmp_path / "systemd" / "com.dayu.wechat.test.service"
    spec = service_manager.ServiceSpec(
        backend="systemd",
        label="com.dayu.wechat.test",
        definition_path=unit_path,
        working_directory=tmp_path / "repo",
        program_arguments=("/usr/bin/python3", "-m", "dayu.wechat", "run"),
        environment_variables=(("MIMO_API_KEY", "secret-1"),),
    )
    command_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        service_manager,
        "_run_systemctl_user",
        lambda arguments, *, check: command_calls.append(tuple(arguments))
        or subprocess.CompletedProcess(args=["systemctl", "--user", *arguments], returncode=0, stdout="", stderr=""),
    )

    written_path = service_manager.install_systemd_service(spec)

    assert written_path == unit_path
    assert unit_path.exists()
    assert "ExecStart=/usr/bin/python3 -m dayu.wechat run" in unit_path.read_text(encoding="utf-8")
    assert 'Environment="MIMO_API_KEY=secret-1"' in unit_path.read_text(encoding="utf-8")
    assert command_calls == [("daemon-reload",)]


@pytest.mark.unit
def test_start_launchd_service_noops_when_process_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 launchd service start 在已运行时不会触发重启。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")
    command_calls: list[tuple[tuple[str, ...], bool]] = []

    monkeypatch.setattr(
        service_manager,
        "query_launchd_service_status",
        lambda **_kwargs: service_manager.ServiceStatus(
            backend="launchd",
            label="com.dayu.wechat.test",
            definition_path=plist_path,
            installed=True,
            loaded=True,
            pid=123,
        ),
    )
    monkeypatch.setattr(
        service_manager,
        "_run_launchctl",
        lambda arguments, *, check: command_calls.append((tuple(arguments), check))
        or subprocess.CompletedProcess(args=["launchctl", *arguments], returncode=0, stdout="", stderr=""),
    )

    service_manager.start_launchd_service(
        label="com.dayu.wechat.test",
        plist_path=plist_path,
    )

    assert command_calls == []


@pytest.mark.unit
def test_restart_launchd_service_uses_kickstart_k(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 launchd service restart 会显式执行 kickstart -k。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")
    command_calls: list[tuple[tuple[str, ...], bool]] = []

    monkeypatch.setattr(
        service_manager,
        "query_launchd_service_status",
        lambda **_kwargs: service_manager.ServiceStatus(
            backend="launchd",
            label="com.dayu.wechat.test",
            definition_path=plist_path,
            installed=True,
            loaded=True,
            pid=123,
        ),
    )
    monkeypatch.setattr(
        service_manager,
        "_run_launchctl",
        lambda arguments, *, check: command_calls.append((tuple(arguments), check))
        or subprocess.CompletedProcess(args=["launchctl", *arguments], returncode=0, stdout="", stderr=""),
    )

    service_manager.restart_launchd_service(
        label="com.dayu.wechat.test",
        plist_path=plist_path,
    )

    assert command_calls == [
        (("kickstart", "-k", f"gui/{service_manager.os.getuid()}/com.dayu.wechat.test"), True),
    ]


@pytest.mark.unit
def test_start_systemd_service_calls_systemctl_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 systemd service start 会调用 `systemctl --user start`。"""

    unit_path = tmp_path / "service.service"
    unit_path.write_text("[Service]\n", encoding="utf-8")
    command_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        service_manager,
        "_run_systemctl_user",
        lambda arguments, *, check: command_calls.append(tuple(arguments))
        or subprocess.CompletedProcess(args=["systemctl", "--user", *arguments], returncode=0, stdout="", stderr=""),
    )

    service_manager.start_systemd_service(
        label="com.dayu.wechat.test",
        unit_path=unit_path,
    )

    assert command_calls == [
        ("daemon-reload",),
        ("start", "com.dayu.wechat.test.service"),
    ]


@pytest.mark.unit
def test_restart_systemd_service_calls_systemctl_restart_when_loaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证运行中的 systemd service restart 会调用 `systemctl --user restart`。"""

    unit_path = tmp_path / "service.service"
    unit_path.write_text("[Service]\n", encoding="utf-8")
    command_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        service_manager,
        "query_systemd_service_status",
        lambda **_kwargs: service_manager.ServiceStatus(
            backend="systemd",
            label="com.dayu.wechat.test",
            definition_path=unit_path,
            installed=True,
            loaded=True,
            pid=456,
        ),
    )
    monkeypatch.setattr(
        service_manager,
        "_run_systemctl_user",
        lambda arguments, *, check: command_calls.append(tuple(arguments))
        or subprocess.CompletedProcess(args=["systemctl", "--user", *arguments], returncode=0, stdout="", stderr=""),
    )

    service_manager.restart_systemd_service(
        label="com.dayu.wechat.test",
        unit_path=unit_path,
    )

    assert command_calls == [
        ("daemon-reload",),
        ("restart", "com.dayu.wechat.test.service"),
    ]


@pytest.mark.unit
def test_stop_launchd_service_sends_sigterm_before_bootout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 launchd service stop 会先发 SIGTERM，再卸载 launchd 定义。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")
    command_calls: list[tuple[tuple[str, ...], bool]] = []

    monkeypatch.setattr(
        service_manager,
        "query_launchd_service_status",
        lambda **_kwargs: service_manager.ServiceStatus(
            backend="launchd",
            label="com.dayu.wechat.test",
            definition_path=plist_path,
            installed=True,
            loaded=True,
            pid=123,
        ),
    )
    monkeypatch.setattr(service_manager, "_wait_for_launchd_service_process_exit", lambda **_kwargs: True)

    def _fake_run_launchctl(arguments, *, check: bool):
        command_calls.append((tuple(arguments), check))
        return subprocess.CompletedProcess(args=["launchctl"], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(service_manager, "_run_launchctl", _fake_run_launchctl)

    stopped = service_manager.stop_launchd_service(
        label="com.dayu.wechat.test",
        plist_path=plist_path,
    )

    assert stopped is True
    assert command_calls == [
        (("kill", "SIGTERM", f"gui/{service_manager.os.getuid()}/com.dayu.wechat.test"), True),
        (("bootout", f"gui/{service_manager.os.getuid()}", str(plist_path.resolve())), True),
    ]


@pytest.mark.unit
def test_stop_launchd_service_logs_warning_when_graceful_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 launchd service stop 等待优雅退出超时后会输出 warning。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")
    warnings: list[str] = []

    monkeypatch.setattr(
        service_manager,
        "query_launchd_service_status",
        lambda **_kwargs: service_manager.ServiceStatus(
            backend="launchd",
            label="com.dayu.wechat.test",
            definition_path=plist_path,
            installed=True,
            loaded=True,
            pid=123,
        ),
    )
    monkeypatch.setattr(service_manager, "_wait_for_launchd_service_process_exit", lambda **_kwargs: False)
    monkeypatch.setattr(
        service_manager.Log,
        "warning",
        lambda message, *, module="APP": warnings.append(f"{module}:{message}"),
    )
    monkeypatch.setattr(
        service_manager,
        "_run_launchctl",
        lambda arguments, *, check: subprocess.CompletedProcess(args=["launchctl", *arguments], returncode=0, stdout="", stderr=""),
    )

    stopped = service_manager.stop_launchd_service(
        label="com.dayu.wechat.test",
        plist_path=plist_path,
    )

    assert stopped is True
    assert warnings == [
        f"APP.WECHAT.SERVICE:WeChat daemon 在 {service_manager.STOP_WAIT_TIMEOUT_SEC:.1f}s 内未完成优雅退出，继续执行 launchd bootout"
    ]


@pytest.mark.unit
def test_stop_systemd_service_calls_systemctl_stop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 systemd service stop 会调用 `systemctl --user stop`。"""

    unit_path = tmp_path / "service.service"
    unit_path.write_text("[Service]\n", encoding="utf-8")
    command_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        service_manager,
        "query_systemd_service_status",
        lambda **_kwargs: service_manager.ServiceStatus(
            backend="systemd",
            label="com.dayu.wechat.test",
            definition_path=unit_path,
            installed=True,
            loaded=True,
            pid=456,
        ),
    )
    monkeypatch.setattr(
        service_manager,
        "_run_systemctl_user",
        lambda arguments, *, check: command_calls.append(tuple(arguments))
        or subprocess.CompletedProcess(args=["systemctl", "--user", *arguments], returncode=0, stdout="", stderr=""),
    )

    stopped = service_manager.stop_systemd_service(
        label="com.dayu.wechat.test",
        unit_path=unit_path,
    )

    assert stopped is True
    assert command_calls == [("stop", "com.dayu.wechat.test.service")]


@pytest.mark.unit
def test_service_manager_generic_wrappers_dispatch_launchd_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 backend 无关 wrapper 会正确路由到 launchd 实现。"""

    definition_path = tmp_path / "service.plist"
    spec = service_manager.ServiceSpec(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=definition_path,
        working_directory=tmp_path,
        program_arguments=("/usr/bin/python3", "-m", "dayu.wechat", "run"),
    )
    status = service_manager.ServiceStatus(
        backend="launchd",
        label=spec.label,
        definition_path=definition_path,
        installed=True,
        loaded=True,
        pid=123,
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        service_manager,
        "resolve_launch_agent_plist_path",
        lambda label: calls.append(("resolve", label)) or definition_path,
    )
    monkeypatch.setattr(
        service_manager,
        "build_launchd_service_spec",
        lambda **kwargs: calls.append(("build", kwargs["state_dir"])) or spec,
    )
    monkeypatch.setattr(
        service_manager,
        "install_launchd_service",
        lambda incoming: calls.append(("install", incoming.label)) or definition_path,
    )
    monkeypatch.setattr(
        service_manager,
        "query_launchd_service_status",
        lambda **kwargs: calls.append(("query", kwargs["label"])) or status,
    )
    monkeypatch.setattr(
        service_manager,
        "start_launchd_service",
        lambda **kwargs: calls.append(("start", kwargs["label"])),
    )
    monkeypatch.setattr(
        service_manager,
        "restart_launchd_service",
        lambda **kwargs: calls.append(("restart", kwargs["label"])),
    )
    monkeypatch.setattr(
        service_manager,
        "stop_launchd_service",
        lambda **kwargs: calls.append(("stop", kwargs["label"])) or True,
    )
    monkeypatch.setattr(
        service_manager,
        "uninstall_launchd_service",
        lambda **kwargs: calls.append(("uninstall", kwargs["label"])) or True,
    )

    assert service_manager.resolve_service_definition_path(spec.label, backend="launchd") == definition_path
    assert (
        service_manager.build_service_spec(
            state_dir=tmp_path / ".wechat",
            working_directory=tmp_path,
            python_executable="/usr/bin/python3",
            run_arguments=["run"],
            backend="launchd",
        )
        == spec
    )
    assert service_manager.install_service(spec) == definition_path
    assert (
        service_manager.query_service_status(
            label=spec.label,
            definition_path=definition_path,
            backend="launchd",
        )
        == status
    )
    service_manager.start_service(label=spec.label, definition_path=definition_path, backend="launchd")
    service_manager.restart_service(label=spec.label, definition_path=definition_path, backend="launchd")
    assert service_manager.stop_service(label=spec.label, definition_path=definition_path, backend="launchd") is True
    assert (
        service_manager.uninstall_service(
            label=spec.label,
            definition_path=definition_path,
            backend="launchd",
        )
        is True
    )
    assert [name for name, _ in calls] == [
        "resolve",
        "build",
        "install",
        "query",
        "start",
        "restart",
        "stop",
        "uninstall",
    ]


@pytest.mark.unit
def test_service_manager_generic_wrappers_dispatch_systemd_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 backend 无关 wrapper 会正确路由到 systemd 实现。"""

    definition_path = tmp_path / "service.service"
    spec = service_manager.ServiceSpec(
        backend="systemd",
        label="com.dayu.wechat.test",
        definition_path=definition_path,
        working_directory=tmp_path,
        program_arguments=("/usr/bin/python3", "-m", "dayu.wechat", "run"),
    )
    status = service_manager.ServiceStatus(
        backend="systemd",
        label=spec.label,
        definition_path=definition_path,
        installed=True,
        loaded=True,
        pid=456,
    )
    calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        service_manager,
        "resolve_systemd_user_unit_path",
        lambda label: calls.append(("resolve", label)) or definition_path,
    )
    monkeypatch.setattr(
        service_manager,
        "build_systemd_service_spec",
        lambda **kwargs: calls.append(("build", kwargs["state_dir"])) or spec,
    )
    monkeypatch.setattr(
        service_manager,
        "install_systemd_service",
        lambda incoming: calls.append(("install", incoming.label)) or definition_path,
    )
    monkeypatch.setattr(
        service_manager,
        "query_systemd_service_status",
        lambda **kwargs: calls.append(("query", kwargs["label"])) or status,
    )
    monkeypatch.setattr(
        service_manager,
        "start_systemd_service",
        lambda **kwargs: calls.append(("start", kwargs["label"])),
    )
    monkeypatch.setattr(
        service_manager,
        "restart_systemd_service",
        lambda **kwargs: calls.append(("restart", kwargs["label"])),
    )
    monkeypatch.setattr(
        service_manager,
        "stop_systemd_service",
        lambda **kwargs: calls.append(("stop", kwargs["label"])) or True,
    )
    monkeypatch.setattr(
        service_manager,
        "uninstall_systemd_service",
        lambda **kwargs: calls.append(("uninstall", kwargs["label"])) or True,
    )

    assert service_manager.resolve_service_definition_path(spec.label, backend="systemd") == definition_path
    assert (
        service_manager.build_service_spec(
            state_dir=tmp_path / ".wechat",
            working_directory=tmp_path,
            python_executable="/usr/bin/python3",
            run_arguments=["run"],
            backend="systemd",
        )
        == spec
    )
    assert service_manager.install_service(spec) == definition_path
    assert (
        service_manager.query_service_status(
            label=spec.label,
            definition_path=definition_path,
            backend="systemd",
        )
        == status
    )
    service_manager.start_service(label=spec.label, definition_path=definition_path, backend="systemd")
    service_manager.restart_service(label=spec.label, definition_path=definition_path, backend="systemd")
    assert service_manager.stop_service(label=spec.label, definition_path=definition_path, backend="systemd") is True
    assert (
        service_manager.uninstall_service(
            label=spec.label,
            definition_path=definition_path,
            backend="systemd",
        )
        is True
    )
    assert [name for name, _ in calls] == [
        "resolve",
        "build",
        "install",
        "query",
        "start",
        "restart",
        "stop",
        "uninstall",
    ]


@pytest.mark.unit
def test_service_manager_generic_wrappers_reject_unknown_backend(tmp_path: Path) -> None:
    """验证 backend 无关 wrapper 遇到未知 backend 会稳定失败。"""

    spec = service_manager.ServiceSpec(
        backend=cast(Any, "bad"),
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "service.bad",
        working_directory=tmp_path,
        program_arguments=("/usr/bin/python3",),
    )

    with pytest.raises(ValueError, match="未知 service backend"):
        service_manager.resolve_service_definition_path("com.dayu.wechat.test", backend=cast(Any, "bad"))
    with pytest.raises(ValueError, match="未知 service backend"):
        service_manager.build_service_spec(
            state_dir=tmp_path / ".wechat",
            working_directory=tmp_path,
            python_executable="/usr/bin/python3",
            run_arguments=["run"],
            backend=cast(Any, "bad"),
        )
    with pytest.raises(ValueError, match="未知 service backend"):
        service_manager.install_service(spec)
    with pytest.raises(ValueError, match="未知 service backend"):
        service_manager.query_service_status(
            label=spec.label,
            definition_path=spec.definition_path,
            backend=cast(Any, "bad"),
        )
    with pytest.raises(ValueError, match="未知 service backend"):
        service_manager.start_service(label=spec.label, definition_path=spec.definition_path, backend=cast(Any, "bad"))
    with pytest.raises(ValueError, match="未知 service backend"):
        service_manager.restart_service(label=spec.label, definition_path=spec.definition_path, backend=cast(Any, "bad"))
    with pytest.raises(ValueError, match="未知 service backend"):
        service_manager.stop_service(label=spec.label, definition_path=spec.definition_path, backend=cast(Any, "bad"))
    with pytest.raises(ValueError, match="未知 service backend"):
        service_manager.uninstall_service(label=spec.label, definition_path=spec.definition_path, backend=cast(Any, "bad"))


@pytest.mark.unit
def test_service_manager_command_runners_cover_success_and_checked_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 launchctl/systemctl 命令封装的成功路径与 checked 失败路径。"""

    calls: list[tuple[str, ...]] = []

    def _fake_run(arguments, *, check: bool, capture_output: bool, text: bool):
        del check
        del capture_output
        del text
        calls.append(tuple(arguments))
        if arguments[0] == "launchctl":
            return subprocess.CompletedProcess(args=arguments, returncode=0, stdout="ok", stderr="")
        return subprocess.CompletedProcess(args=arguments, returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(service_manager.subprocess, "run", _fake_run)

    launch_result = service_manager._run_launchctl(("print", "gui/1/x"), check=False)
    assert launch_result.stdout == "ok"
    with pytest.raises(RuntimeError, match="systemctl --user 执行失败"):
        service_manager._run_systemctl_user(("status", "x.service"), check=True)
    assert calls == [
        ("launchctl", "print", "gui/1/x"),
        ("systemctl", "--user", "status", "x.service"),
    ]


@pytest.mark.unit
def test_is_service_running_returns_false_when_not_loaded(tmp_path: Path) -> None:
    """验证 service 未 loaded 时 is_service_running 返回 False。"""

    status = service_manager.ServiceStatus(
        backend="systemd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "service.service",
        installed=True,
        loaded=False,
    )

    assert service_manager.is_service_running(status) is False


@pytest.mark.unit
def test_combine_process_output_with_stdout_and_stderr() -> None:
    """验证 stdout 和 stderr 同时有内容时合并输出。"""

    result = subprocess.CompletedProcess(
        args=["cmd"],
        returncode=0,
        stdout="out message",
        stderr="err message",
    )

    combined = service_manager._combine_process_output(result)

    assert combined == "out message\nerr message"


@pytest.mark.unit
def test_combine_process_output_with_only_stdout() -> None:
    """验证仅 stdout 有内容时返回 stdout。"""

    result = subprocess.CompletedProcess(args=["cmd"], returncode=0, stdout="only out", stderr="")

    assert service_manager._combine_process_output(result) == "only out"


@pytest.mark.unit
def test_combine_process_output_with_only_stderr() -> None:
    """验证仅 stderr 有内容时返回 stderr。"""

    result = subprocess.CompletedProcess(args=["cmd"], returncode=0, stdout="", stderr="only err")

    assert service_manager._combine_process_output(result) == "only err"


@pytest.mark.unit
def test_extract_launchd_pid_returns_none_on_no_match() -> None:
    """验证 launchctl 输出中无 pid 时返回 None。"""

    assert service_manager._extract_launchd_pid("state = running;") is None
    assert service_manager._extract_launchd_pid("") is None


@pytest.mark.unit
def test_build_service_log_lines_launchd(tmp_path: Path) -> None:
    """验证 build_service_log_lines 在 launchd 后端返回日志路径。"""

    lines = service_manager.build_service_log_lines(
        label="com.dayu.wechat.test",
        state_dir=tmp_path / ".wechat",
        backend="launchd",
    )

    assert len(lines) == 2
    assert lines[0].startswith("log_stdout:")
    assert lines[1].startswith("log_stderr:")


@pytest.mark.unit
def test_build_service_log_lines_systemd() -> None:
    """验证 build_service_log_lines 在 systemd 后端返回 journal 命令。"""

    lines = service_manager.build_service_log_lines(
        label="com.dayu.wechat.test",
        state_dir=Path("/tmp/state"),
        backend="systemd",
    )

    assert len(lines) == 2
    assert lines[0] == "log_backend: journal"
    assert "journalctl" in lines[1]


@pytest.mark.unit
def test_build_service_log_lines_unknown_backend(tmp_path: Path) -> None:
    """验证 build_service_log_lines 遇到未知 backend 时抛出 ValueError。"""

    with pytest.raises(ValueError, match="未知 service backend"):
        service_manager.build_service_log_lines(
            label="com.dayu.wechat.test",
            state_dir=tmp_path,
            backend=cast(Any, "unknown"),
        )


@pytest.mark.unit
def test_normalize_program_arguments_filters_empty_values() -> None:
    """验证空参数在标准化时被过滤。"""

    result = service_manager._normalize_program_arguments(["python", "", "  ", None])

    assert result == ("python",)


@pytest.mark.unit
def test_normalize_optional_path_returns_none_for_empty() -> None:
    """验证空路径标准化返回 None。"""

    assert service_manager._normalize_optional_path(None) is None
    assert service_manager._normalize_optional_path("") is None
    assert service_manager._normalize_optional_path("  ") is None




@pytest.mark.unit
def test_detect_service_backend_raises_on_unknown_platform() -> None:
    """验证未知平台时 detect_service_backend 抛出 RuntimeError。"""

    with pytest.raises(RuntimeError, match="当前仅支持 macOS"):
        service_manager.detect_service_backend("Windows")


@pytest.mark.unit
def test_detect_service_backend_supports_darwin() -> None:
    """验证 Darwin 平台会路由到 launchd backend。"""

    assert service_manager.detect_service_backend("Darwin") == "launchd"


@pytest.mark.unit
def test_describe_service_backend_launchd() -> None:
    """验证 launchd backend 描述。"""

    assert service_manager.describe_service_backend("launchd") == "macOS launchd"


@pytest.mark.unit
def test_describe_service_backend_systemd() -> None:
    """验证 systemd backend 描述。"""

    assert service_manager.describe_service_backend("systemd") == "Linux systemd --user"


@pytest.mark.unit
def test_describe_service_backend_raises_on_unknown() -> None:
    """验证未知 backend 描述时抛出 ValueError。"""

    with pytest.raises(ValueError, match="未知 service backend"):
        service_manager.describe_service_backend(cast(Any, "unknown"))


@pytest.mark.unit
def test_parse_systemd_pid_returns_none_for_empty() -> None:
    """验证空 MainPID 返回 None。"""

    assert service_manager._parse_systemd_pid(None) is None
    assert service_manager._parse_systemd_pid("") is None


@pytest.mark.unit
def test_parse_systemd_pid_returns_none_for_zero() -> None:
    """验证 MainPID=0 返回 None。"""

    assert service_manager._parse_systemd_pid("0") is None


@pytest.mark.unit
def test_parse_systemd_pid_returns_none_for_non_digit() -> None:
    """验证非数字 MainPID 返回 None。"""

    assert service_manager._parse_systemd_pid("not_a_pid") is None


@pytest.mark.unit
def test_parse_systemd_pid_returns_valid_pid() -> None:
    """验证有效的 MainPID 返回整数 pid。"""

    assert service_manager._parse_systemd_pid("1234") == 1234


@pytest.mark.unit
def test_parse_systemd_show_output_skips_non_matching_lines() -> None:
    """验证非 key=value 行被忽略。"""

    result = service_manager._parse_systemd_show_output("invalid line\n=missing_key\nValidKey=value")

    assert result == {"ValidKey": "value"}


@pytest.mark.unit
def test_render_systemd_unit_raises_on_non_systemd_spec(tmp_path: Path) -> None:
    """验证 _render_systemd_unit 对非 systemd 规格抛出 ValueError。"""

    spec = service_manager.ServiceSpec(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "service.plist",
        working_directory=tmp_path,
        program_arguments=("/usr/bin/python3",),
    )

    with pytest.raises(ValueError, match="_render_systemd_unit 仅接受 systemd 规格"):
        service_manager._render_systemd_unit(spec)


@pytest.mark.unit
def test_render_systemd_environment_line_escapes_special_chars() -> None:
    """验证环境变量值中的特殊字符被正确转义。"""

    line = service_manager._render_systemd_environment_line("KEY", 'val"ue\\x')

    assert line == 'Environment="KEY=val\\"ue\\\\x"'


@pytest.mark.unit
def test_build_launchd_service_spec_raises_on_empty_executable(tmp_path: Path) -> None:
    """验证空 python_executable 时 build_launchd_service_spec 抛出 ValueError。"""

    with pytest.raises(ValueError, match="python_executable 不能为空"):
        service_manager.build_launchd_service_spec(
            state_dir=tmp_path / ".wechat",
            working_directory=tmp_path,
            python_executable="",
            run_arguments=["run"],
        )


@pytest.mark.unit
def test_build_systemd_service_spec_raises_on_empty_executable(tmp_path: Path) -> None:
    """验证空 python_executable 时 build_systemd_service_spec 抛出 ValueError。"""

    with pytest.raises(ValueError, match="python_executable 不能为空"):
        service_manager.build_systemd_service_spec(
            state_dir=tmp_path / ".wechat",
            working_directory=tmp_path,
            python_executable="",
            run_arguments=["run"],
        )


@pytest.mark.unit
def test_install_launchd_service_raises_on_non_launchd_spec(tmp_path: Path) -> None:
    """验证 install_launchd_service 对非 launchd 规格抛出 ValueError。"""

    spec = service_manager.ServiceSpec(
        backend="systemd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "service.service",
        working_directory=tmp_path,
        program_arguments=("/usr/bin/python3",),
    )

    with pytest.raises(ValueError, match="install_launchd_service 仅接受 launchd 规格"):
        service_manager.install_launchd_service(spec)


@pytest.mark.unit
def test_install_launchd_service_raises_on_missing_log_paths(tmp_path: Path) -> None:
    """验证 launchd spec 缺少日志路径时 install_launchd_service 抛出 ValueError。"""

    spec = service_manager.ServiceSpec(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "LaunchAgents" / "service.plist",
        working_directory=tmp_path,
        program_arguments=("/usr/bin/python3",),
        stdout_path=None,
        stderr_path=None,
    )

    with pytest.raises(ValueError, match="launchd service 需要 stdout_path / stderr_path"):
        service_manager.install_launchd_service(spec)


@pytest.mark.unit
def test_install_systemd_service_raises_on_non_systemd_spec(tmp_path: Path) -> None:
    """验证 install_systemd_service 对非 systemd 规格抛出 ValueError。"""

    spec = service_manager.ServiceSpec(
        backend="launchd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "service.plist",
        working_directory=tmp_path,
        program_arguments=("/usr/bin/python3",),
    )

    with pytest.raises(ValueError, match="install_systemd_service 仅接受 systemd 规格"):
        service_manager.install_systemd_service(spec)


@pytest.mark.unit
def test_build_launchctl_domain_target_raises_on_win32(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证在 win32 上 _build_launchctl_domain_target 抛出 RuntimeError。"""

    monkeypatch.setattr(service_manager.sys, "platform", "win32")

    with pytest.raises(RuntimeError, match="launchctl 仅在 macOS 上可用"):
        service_manager._build_launchctl_domain_target()


@pytest.mark.unit
def test_launchd_perror_lines_not_covered_elsewhere() -> None:
    """验证 extract_launchd_pid 能正确提取 pid。"""

    assert service_manager._extract_launchd_pid("pid = 999;") == 999


@pytest.mark.unit
def test_launchd_service_not_installed_status_fields(tmp_path: Path) -> None:
    """验证 launchd 未安装的 status 字段完整性。"""

    status = service_manager.query_launchd_service_status(
        label="com.dayu.wechat.test",
        plist_path=tmp_path / "nonexistent.plist",
    )

    assert status.backend == "launchd"
    assert status.installed is False
    assert status.loaded is False
    assert status.pid is None
    assert status.raw_output == ""


@pytest.mark.unit
def test_launchd_service_installed_but_not_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 launchd 已安装但 launchctl 返回非零时状态为 installed 但 not loaded。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")

    monkeypatch.setattr(
        service_manager,
        "_run_launchctl",
        lambda _args, **_kw: subprocess.CompletedProcess(
            args=["launchctl"], returncode=1, stdout="", stderr="not found",
        ),
    )

    status = service_manager.query_launchd_service_status(
        label="com.dayu.wechat.test",
        plist_path=plist_path,
    )

    assert status.installed is True
    assert status.loaded is False
    assert "not found" in status.raw_output


@pytest.mark.unit
def test_systemd_service_not_installed_status_fields(tmp_path: Path) -> None:
    """验证 systemd 未安装的 status 字段完整性。"""

    status = service_manager.query_systemd_service_status(
        label="com.dayu.wechat.test",
        unit_path=tmp_path / "nonexistent.service",
    )

    assert status.backend == "systemd"
    assert status.installed is False
    assert status.loaded is False


@pytest.mark.unit
def test_systemd_service_installed_but_not_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 systemd 已安装但 systemctl 返回非零时状态为 installed 但 not loaded。"""

    unit_path = tmp_path / "service.service"
    unit_path.write_text("[Service]\n", encoding="utf-8")

    monkeypatch.setattr(
        service_manager,
        "_run_systemctl_user",
        lambda _args, **_kw: subprocess.CompletedProcess(
            args=["systemctl", "--user"], returncode=1, stdout="", stderr="unit not found",
        ),
    )

    status = service_manager.query_systemd_service_status(
        label="com.dayu.wechat.test",
        unit_path=unit_path,
    )

    assert status.installed is True
    assert status.loaded is False
    assert "unit not found" in status.raw_output


@pytest.mark.unit
def test_is_service_running_systemd_loaded_without_pid(tmp_path: Path) -> None:
    """验证 systemd loaded 无 pid 时仍视为运行中。"""

    status = service_manager.ServiceStatus(
        backend="systemd",
        label="com.dayu.wechat.test",
        definition_path=tmp_path / "service.service",
        installed=True,
        loaded=True,
        pid=None,
    )

    assert service_manager.is_service_running(status) is True


@pytest.mark.unit
def test_launchd_start_service_bootstraps_when_not_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 launchd service 在未加载时执行 bootstrap 启动。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")
    command_calls: list[tuple[tuple[str, ...], bool]] = []

    monkeypatch.setattr(
        service_manager,
        "query_launchd_service_status",
        lambda **_kw: service_manager.ServiceStatus(
            backend="launchd",
            label="com.dayu.wechat.test",
            definition_path=plist_path,
            installed=True,
            loaded=False,
        ),
    )
    monkeypatch.setattr(
        service_manager,
        "_run_launchctl",
        lambda arguments, *, check: command_calls.append((tuple(arguments), check))
        or subprocess.CompletedProcess(args=["launchctl", *arguments], returncode=0, stdout="", stderr=""),
    )

    service_manager.start_launchd_service(label="com.dayu.wechat.test", plist_path=plist_path)

    assert len(command_calls) == 1
    assert command_calls[0][0][0] == "bootstrap"


@pytest.mark.unit
def test_launchd_restart_service_bootstraps_when_not_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 launchd service restart 在未加载时执行 bootstrap。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")
    command_calls: list[tuple[tuple[str, ...], bool]] = []

    monkeypatch.setattr(
        service_manager,
        "query_launchd_service_status",
        lambda **_kw: service_manager.ServiceStatus(
            backend="launchd",
            label="com.dayu.wechat.test",
            definition_path=plist_path,
            installed=True,
            loaded=False,
        ),
    )
    monkeypatch.setattr(
        service_manager,
        "_run_launchctl",
        lambda arguments, *, check: command_calls.append((tuple(arguments), check))
        or subprocess.CompletedProcess(args=["launchctl", *arguments], returncode=0, stdout="", stderr=""),
    )

    service_manager.restart_launchd_service(label="com.dayu.wechat.test", plist_path=plist_path)

    assert len(command_calls) == 1
    assert command_calls[0][0][0] == "bootstrap"


@pytest.mark.unit
def test_launchd_start_service_kickstart_when_loaded_without_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 launchd 已加载但无 pid 时执行 kickstart。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")
    command_calls: list[tuple[tuple[str, ...], bool]] = []

    monkeypatch.setattr(
        service_manager,
        "query_launchd_service_status",
        lambda **_kw: service_manager.ServiceStatus(
            backend="launchd",
            label="com.dayu.wechat.test",
            definition_path=plist_path,
            installed=True,
            loaded=True,
            pid=None,
        ),
    )
    monkeypatch.setattr(
        service_manager,
        "_run_launchctl",
        lambda arguments, *, check: command_calls.append((tuple(arguments), check))
        or subprocess.CompletedProcess(args=["launchctl", *arguments], returncode=0, stdout="", stderr=""),
    )

    service_manager.start_launchd_service(label="com.dayu.wechat.test", plist_path=plist_path)

    assert command_calls == [
        (("kickstart", f"gui/{service_manager.os.getuid()}/com.dayu.wechat.test"), True),
    ]


@pytest.mark.unit
def test_stop_launchd_service_returns_false_when_not_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 launchd 未加载时 stop 返回 False。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")

    monkeypatch.setattr(
        service_manager,
        "query_launchd_service_status",
        lambda **_kw: service_manager.ServiceStatus(
            backend="launchd",
            label="com.dayu.wechat.test",
            definition_path=plist_path,
            installed=True,
            loaded=False,
        ),
    )

    assert service_manager.stop_launchd_service(label="com.dayu.wechat.test", plist_path=plist_path) is False


@pytest.mark.unit
def test_stop_launchd_service_skips_sigterm_when_no_pid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 launchd 已加载但无 pid 时跳过 SIGTERM 直接 bootout。"""

    plist_path = tmp_path / "service.plist"
    plist_path.write_text("plist", encoding="utf-8")
    command_calls: list[tuple[tuple[str, ...], bool]] = []

    monkeypatch.setattr(
        service_manager,
        "query_launchd_service_status",
        lambda **_kw: service_manager.ServiceStatus(
            backend="launchd",
            label="com.dayu.wechat.test",
            definition_path=plist_path,
            installed=True,
            loaded=True,
            pid=None,
        ),
    )
    monkeypatch.setattr(
        service_manager,
        "_run_launchctl",
        lambda arguments, *, check: command_calls.append((tuple(arguments), check))
        or subprocess.CompletedProcess(args=["launchctl", *arguments], returncode=0, stdout="", stderr=""),
    )

    stopped = service_manager.stop_launchd_service(label="com.dayu.wechat.test", plist_path=plist_path)

    assert stopped is True
    assert command_calls == [
        (("bootout", f"gui/{service_manager.os.getuid()}", str(plist_path.resolve())), True),
    ]


@pytest.mark.unit
def test_stop_systemd_service_returns_false_when_not_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 systemd 未加载时 stop 返回 False。"""

    unit_path = tmp_path / "service.service"
    unit_path.write_text("[Service]\n", encoding="utf-8")

    monkeypatch.setattr(
        service_manager,
        "query_systemd_service_status",
        lambda **_kw: service_manager.ServiceStatus(
            backend="systemd",
            label="com.dayu.wechat.test",
            definition_path=unit_path,
            installed=True,
            loaded=False,
        ),
    )

    assert service_manager.stop_systemd_service(label="com.dayu.wechat.test", unit_path=unit_path) is False


@pytest.mark.unit
def test_restart_systemd_service_starts_when_not_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 systemd 未加载时 restart 执行 start。"""

    unit_path = tmp_path / "service.service"
    unit_path.write_text("[Service]\n", encoding="utf-8")
    command_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        service_manager,
        "_run_systemctl_user",
        lambda arguments, *, check: command_calls.append(tuple(arguments))
        or subprocess.CompletedProcess(args=["systemctl", "--user", *arguments], returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        service_manager,
        "query_systemd_service_status",
        lambda **_kw: service_manager.ServiceStatus(
            backend="systemd",
            label="com.dayu.wechat.test",
            definition_path=unit_path,
            installed=True,
            loaded=False,
        ),
    )

    service_manager.restart_systemd_service(label="com.dayu.wechat.test", unit_path=unit_path)

    assert command_calls == [
        ("daemon-reload",),
        ("start", "com.dayu.wechat.test.service"),
    ]


@pytest.mark.unit
def test_launchd_service_start_raises_on_missing_plist(tmp_path: Path) -> None:
    """验证 plist 不存在时 start_launchd_service 抛出 FileNotFoundError。"""

    with pytest.raises(FileNotFoundError, match="launchd plist 不存在"):
        service_manager.start_launchd_service(
            label="com.dayu.wechat.test",
            plist_path=tmp_path / "missing.plist",
        )


@pytest.mark.unit
def test_launchd_service_restart_raises_on_missing_plist(tmp_path: Path) -> None:
    """验证 plist 不存在时 restart_launchd_service 抛出 FileNotFoundError。"""

    with pytest.raises(FileNotFoundError, match="launchd plist 不存在"):
        service_manager.restart_launchd_service(
            label="com.dayu.wechat.test",
            plist_path=tmp_path / "missing.plist",
        )


@pytest.mark.unit
def test_systemd_service_start_raises_on_missing_unit(tmp_path: Path) -> None:
    """验证 unit 不存在时 start_systemd_service 抛出 FileNotFoundError。"""

    with pytest.raises(FileNotFoundError, match="systemd unit 不存在"):
        service_manager.start_systemd_service(
            label="com.dayu.wechat.test",
            unit_path=tmp_path / "missing.service",
        )


@pytest.mark.unit
def test_systemd_service_restart_raises_on_missing_unit(tmp_path: Path) -> None:
    """验证 unit 不存在时 restart_systemd_service 抛出 FileNotFoundError。"""

    with pytest.raises(FileNotFoundError, match="systemd unit 不存在"):
        service_manager.restart_systemd_service(
            label="com.dayu.wechat.test",
            unit_path=tmp_path / "missing.service",
        )


@pytest.mark.unit
def test_resolve_launch_agent_plist_path_raises_on_empty_label() -> None:
    """验证空 label 时 resolve_launch_agent_plist_path 抛出 ValueError。"""

    with pytest.raises(ValueError, match="label 不能为空"):
        service_manager.resolve_launch_agent_plist_path("")


@pytest.mark.unit
def test_resolve_systemd_user_unit_path_raises_on_empty_label() -> None:
    """验证空 label 时 resolve_systemd_user_unit_path 抛出 ValueError。"""

    with pytest.raises(ValueError, match="label 不能为空"):
        service_manager.resolve_systemd_user_unit_path("")