"""`utils.build_offline_bundle` 模块测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils import build_offline_bundle as module

pytestmark = pytest.mark.unit


class TestWriteInstallScript:
    """`_write_install_script()` 测试。"""

    def test_writes_unix_install_script_with_binary_only_guard(self, tmp_path: Path) -> None:
        """Unix 安装脚本应显式要求仅安装 wheel。"""

        module._write_install_script(tmp_path, version="0.1.2", platform_id="macos-arm64")

        script_text = (tmp_path / "install.sh").read_text(encoding="utf-8")

        assert module._PIP_ONLY_BINARY_ALL in script_text
        assert "dayu-agent[browser]==0.1.2" in script_text

    def test_writes_windows_install_script_with_binary_only_guard(self, tmp_path: Path) -> None:
        """Windows 安装脚本应显式要求仅安装 wheel。"""

        module._write_install_script(tmp_path, version="0.1.2", platform_id="windows-x64")

        script_text = (tmp_path / "install.cmd").read_text(encoding="utf-8")

        assert module._PIP_ONLY_BINARY_ALL in script_text
        assert "dayu-agent[browser]==0.1.2" in script_text


class TestBuildSourceDistributionWheels:
    """`_build_source_distribution_wheels()` 测试。"""

    def test_skips_when_wheelhouse_contains_only_wheels(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """若 wheelhouse 里只有 wheel，则不应触发 `pip wheel`。"""

        (tmp_path / "already-built-1.0.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")

        command_calls: list[list[str]] = []

        def _fake_run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
            del env
            command_calls.append(command)

        monkeypatch.setattr(module, "_run_command", _fake_run_command)

        module._build_source_distribution_wheels(tmp_path)

        assert command_calls == []

    def test_builds_source_distributions_and_removes_archives(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """应把源码分发包预构建为 wheel，并删除原始源码归档。"""

        source_tar = tmp_path / "demo_pkg-1.0.0.tar.gz"
        source_zip = tmp_path / "other_pkg-2.0.0.zip"
        existing_wheel = tmp_path / "keep_pkg-3.0.0-py3-none-any.whl"
        source_tar.write_text("tar", encoding="utf-8")
        source_zip.write_text("zip", encoding="utf-8")
        existing_wheel.write_text("wheel", encoding="utf-8")

        command_calls: list[list[str]] = []

        def _fake_run_command(command: list[str], *, env: dict[str, str] | None = None) -> None:
            del env
            command_calls.append(command)
            wheel_dir = Path(command[command.index("--wheel-dir") + 1])
            source_paths = [Path(path_text) for path_text in command[command.index(str(wheel_dir)) + 1 :]]
            for source_path in source_paths:
                wheel_name = f"{source_path.stem}-py3-none-any.whl"
                (wheel_dir / wheel_name).write_text("built", encoding="utf-8")

        monkeypatch.setattr(module, "_run_command", _fake_run_command)

        module._build_source_distribution_wheels(tmp_path)

        assert len(command_calls) == 1
        assert command_calls[0][:5] == [module.sys.executable, "-m", "pip", "wheel", "--no-deps"]
        assert not source_tar.exists()
        assert not source_zip.exists()
        assert existing_wheel.exists()
        assert (tmp_path / "demo_pkg-1.0.0.tar-py3-none-any.whl").exists()
        assert (tmp_path / "other_pkg-2.0.0-py3-none-any.whl").exists()
