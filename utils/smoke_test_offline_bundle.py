"""验证 Dayu Agent 离线安装包。

本脚本会把离线安装包解压到临时目录，创建干净虚拟环境，执行离线安装脚本，
并验证最小 import / CLI smoke 流程是否通过。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
import tempfile
import venv
import zipfile
from pathlib import Path
from typing import Mapping, Sequence


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    参数：
        无。

    返回值：
        argparse.Namespace：解析后的参数。

    异常：
        SystemExit：命令行参数不合法时抛出。
    """

    parser = argparse.ArgumentParser(description="在干净虚拟环境中验证离线安装包。")
    parser.add_argument("--archive", type=Path, required=True, help="离线安装包归档路径。")
    return parser.parse_args()


def _run_command(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> None:
    """执行外部命令并在失败时抛错。

    参数：
        command：命令与参数序列。
        env：可选环境变量覆盖。

    返回值：
        无。

    异常：
        subprocess.CalledProcessError：命令执行失败时抛出。
    """

    subprocess.run(command, check=True, env=dict(env) if env is not None else None)


def _extract_archive(archive_path: Path, extraction_root: Path) -> Path:
    """解压离线安装包并返回包根目录。

    参数：
        archive_path：离线安装包路径。
        extraction_root：解压目标目录。

    返回值：
        Path：离线包根目录。

    异常：
        FileNotFoundError：归档不存在时抛出。
        RuntimeError：解压后未找到唯一包根目录时抛出。
        tarfile.TarError / zipfile.BadZipFile：归档损坏时抛出。
    """

    if not archive_path.is_file():
        raise FileNotFoundError(f"未找到离线安装包：{archive_path}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as archive_file:
            archive_file.extractall(extraction_root)
    else:
        with tarfile.open(archive_path, "r:gz") as archive_file:
            archive_file.extractall(extraction_root)
    children = [path for path in extraction_root.iterdir() if path.is_dir()]
    if len(children) != 1:
        raise RuntimeError(f"解压后未得到唯一根目录：{children}")
    return children[0]


def _venv_paths(venv_root: Path) -> tuple[Path, Path]:
    """返回虚拟环境的 Python 与脚本目录。

    参数：
        venv_root：虚拟环境根目录。

    返回值：
        tuple[Path, Path]：Python 可执行文件路径与脚本目录路径。

    异常：
        RuntimeError：虚拟环境结构异常时抛出。
    """

    if os.name == "nt":
        python_path = venv_root / "Scripts" / "python.exe"
        scripts_dir = venv_root / "Scripts"
    else:
        python_path = venv_root / "bin" / "python"
        scripts_dir = venv_root / "bin"
    if not python_path.exists() or not scripts_dir.exists():
        raise RuntimeError(f"虚拟环境结构不完整：{venv_root}")
    return python_path, scripts_dir


def _run_install_script(bundle_root: Path, python_path: Path) -> None:
    """执行离线安装脚本。

    参数：
        bundle_root：离线包根目录。
        python_path：虚拟环境 Python 路径。

    返回值：
        无。

    异常：
        FileNotFoundError：安装脚本不存在时抛出。
        subprocess.CalledProcessError：安装失败时抛出。
    """

    env = dict(os.environ)
    env["PYTHON_BIN"] = str(python_path)
    if os.name == "nt":
        script_path = bundle_root / "install.cmd"
        if not script_path.is_file():
            raise FileNotFoundError(f"未找到安装脚本：{script_path}")
        _run_command(["cmd.exe", "/c", str(script_path)], env=env)
        return
    script_path = bundle_root / "install.sh"
    if not script_path.is_file():
        raise FileNotFoundError(f"未找到安装脚本：{script_path}")
    _run_command(["/bin/sh", str(script_path)], env=env)


def _run_smoke_checks(python_path: Path, scripts_dir: Path) -> None:
    """执行 import / CLI smoke 验证。

    参数：
        python_path：虚拟环境 Python 路径。
        scripts_dir：虚拟环境脚本目录。

    返回值：
        无。

    异常：
        subprocess.CalledProcessError：任一 smoke 命令失败时抛出。
    """

    _run_command([str(python_path), "-c", "import dayu"])
    dayu_cli = scripts_dir / ("dayu-cli.exe" if os.name == "nt" else "dayu-cli")
    dayu_wechat = scripts_dir / ("dayu-wechat.exe" if os.name == "nt" else "dayu-wechat")
    dayu_render = scripts_dir / ("dayu-render.exe" if os.name == "nt" else "dayu-render")
    _run_command([str(dayu_cli), "--help"])
    _run_command([str(dayu_wechat), "--help"])
    _run_command([str(dayu_render), "--help"])
    _run_command([str(dayu_cli), "init", "--help"])


def main() -> None:
    """执行离线安装包 smoke test。

    参数：
        无。

    返回值：
        无。

    异常：
        FileNotFoundError：输入文件不存在时抛出。
        RuntimeError：解压结构或虚拟环境结构异常时抛出。
        subprocess.CalledProcessError：安装或 smoke 验证失败时抛出。
    """

    args = _parse_args()
    archive_path = args.archive.resolve()
    with tempfile.TemporaryDirectory(prefix="dayu-offline-smoke-") as temp_dir_name:
        temp_root = Path(temp_dir_name)
        extraction_root = temp_root / "bundle"
        extraction_root.mkdir(parents=True, exist_ok=True)
        bundle_root = _extract_archive(archive_path, extraction_root)
        venv_root = temp_root / "venv"
        builder = venv.EnvBuilder(with_pip=True, clear=True)
        builder.create(venv_root)
        python_path, scripts_dir = _venv_paths(venv_root)
        _run_install_script(bundle_root, python_path)
        _run_smoke_checks(python_path, scripts_dir)


if __name__ == "__main__":
    main()
