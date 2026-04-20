"""入口模块覆盖测试。"""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import ModuleType
from typing import Any, cast

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_main_module(monkeypatch: pytest.MonkeyPatch, module_name: str, return_code: int) -> None:
    """安装可控的 `main` 模块测试桩。"""

    if "." in module_name:
        package_name, attr_name = module_name.rsplit(".", 1)
        package_module = ModuleType(package_name)
        cast(Any, package_module).__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package_module)
    else:
        package_name = ""
        attr_name = module_name

    module = ModuleType(module_name)

    def _main() -> int:
        """返回预设退出码。"""

        return return_code

    cast(Any, module).main = _main
    monkeypatch.setitem(sys.modules, module_name, module)
    if package_name:
        cast(Any, sys.modules[package_name]).__dict__[attr_name] = module


@pytest.mark.unit
@pytest.mark.parametrize(
    ("entrypoint_path", "dependency_module", "return_code"),
    [
        (_REPO_ROOT / "dayu/__main__.py", "dayu.cli.main", 11),
        (_REPO_ROOT / "dayu/cli/__main__.py", "dayu.cli.main", 12),
        (_REPO_ROOT / "dayu/wechat/__main__.py", "dayu.wechat.main", 13),
    ],
)
def test_module_entrypoints_raise_system_exit(
    monkeypatch: pytest.MonkeyPatch,
    entrypoint_path: Path,
    dependency_module: str,
    return_code: int,
) -> None:
    """包入口应把 `main()` 返回值透传为 `SystemExit.code`。"""

    _install_main_module(monkeypatch, dependency_module, return_code)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(entrypoint_path), run_name="__main__")

    assert exc_info.value.code == return_code


@pytest.mark.unit
def test_tool_limits_reexports_stable_limit_types() -> None:
    """稳定导出模块应直接重导出 doc/fins limits 类型。"""

    from dayu.contracts.tool_configs import DocToolLimits, FinsToolLimits
    from dayu.tool_limits import __all__, DocToolLimits as ExportedDocToolLimits, FinsToolLimits as ExportedFinsToolLimits

    assert ExportedDocToolLimits is DocToolLimits
    assert ExportedFinsToolLimits is FinsToolLimits
    assert __all__ == ["DocToolLimits", "FinsToolLimits"]


@pytest.mark.unit
def test_import_dayu_cli_main_keeps_heavy_runtime_modules_lazy() -> None:
    """导入 `dayu.cli.main` 时不应抢先导入重运行时模块。"""

    command = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "import dayu.cli.main; "
            "print(json.dumps({"
            "\"host_commands\": \"dayu.cli.commands.host\" in sys.modules, "
            "\"dependency_setup\": \"dayu.cli.dependency_setup\" in sys.modules, "
            "\"services_package\": \"dayu.services\" in sys.modules, "
            "\"write_service\": \"dayu.services.write_service\" in sys.modules"
            "}))"
        ),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )

    observed = json.loads(completed.stdout.strip())
    assert observed == {
        "host_commands": False,
        "dependency_setup": False,
        "services_package": False,
        "write_service": False,
    }


@pytest.mark.unit
def test_import_dayu_wechat_main_keeps_daemon_module_lazy() -> None:
    """导入 `dayu.wechat.main` 时不应抢先导入 daemon。"""

    command = [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "import dayu.wechat.main; "
            "print(json.dumps({"
            "\"wechat_package\": \"dayu.wechat\" in sys.modules, "
            "\"wechat_daemon\": \"dayu.wechat.daemon\" in sys.modules, "
            "\"host_runtime\": \"dayu.host.host\" in sys.modules, "
            "\"services_package\": \"dayu.services\" in sys.modules, "
            "\"fins_runtime\": \"dayu.fins.service_runtime\" in sys.modules"
            "}))"
        ),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )

    observed = json.loads(completed.stdout.strip())
    assert observed == {
        "wechat_package": True,
        "wechat_daemon": False,
        "host_runtime": False,
        "services_package": False,
        "fins_runtime": False,
    }
