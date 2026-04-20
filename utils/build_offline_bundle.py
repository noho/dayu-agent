"""构建 Dayu Agent 离线安装包。

本脚本用于把已构建好的项目 wheel、对应平台的 wheelhouse、安装脚本与
简短说明打包为单个离线交付物，供 CI / 发布流程复用。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path
from typing import Mapping, Sequence

_SOURCE_DISTRIBUTION_SUFFIXES: tuple[str, ...] = (".tar.gz", ".zip", ".tar.bz2", ".tar.xz", ".tgz")
_PIP_ONLY_BINARY_ALL = "--only-binary=:all:"


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。

    参数：
        无。

    返回值：
        argparse.Namespace：解析后的命令行参数。

    异常：
        SystemExit：命令行参数不合法时抛出。
    """

    parser = argparse.ArgumentParser(description="构建 Dayu Agent 离线安装包。")
    parser.add_argument("--wheel", type=Path, required=True, help="已构建好的项目 wheel 路径。")
    parser.add_argument("--constraints", type=Path, required=True, help="用于离线包的锁定 constraints 文件。")
    parser.add_argument("--platform-id", required=True, help="平台标识，例如 macos-arm64 / linux-x64。")
    parser.add_argument("--output-dir", type=Path, required=True, help="离线包输出目录。")
    return parser.parse_args()


def _read_wheel_metadata(wheel_path: Path) -> tuple[str, str]:
    """读取 wheel 中的包名与版本。

    参数：
        wheel_path：项目 wheel 文件路径。

    返回值：
        tuple[str, str]：标准化后的包名与版本号。

    异常：
        FileNotFoundError：wheel 文件不存在时抛出。
        RuntimeError：wheel 中缺少 METADATA 或缺少必要字段时抛出。
    """

    if not wheel_path.is_file():
        raise FileNotFoundError(f"未找到 wheel：{wheel_path}")
    with zipfile.ZipFile(wheel_path) as wheel_zip:
        metadata_name = next(
            (name for name in wheel_zip.namelist() if name.endswith("METADATA")),
            None,
        )
        if metadata_name is None:
            raise RuntimeError(f"wheel 缺少 METADATA：{wheel_path}")
        metadata_text = wheel_zip.read(metadata_name).decode("utf-8")
    parsed = Parser().parsestr(metadata_text)
    package_name = parsed.get("Name")
    version = parsed.get("Version")
    if package_name is None or version is None:
        raise RuntimeError(f"wheel 元数据缺少 Name/Version：{wheel_path}")
    return package_name, version


def _bundle_stem(package_name: str, version: str, platform_id: str) -> str:
    """计算离线包目录名与归档名前缀。

    参数：
        package_name：项目包名。
        version：项目版本号。
        platform_id：平台标识。

    返回值：
        str：离线包目录名。

    异常：
        无。
    """

    normalized_name = package_name.replace("_", "-")
    return f"{normalized_name}-{version}-{platform_id}-offline"


def _archive_path(output_dir: Path, bundle_stem: str, platform_id: str) -> Path:
    """根据平台生成归档文件路径。

    参数：
        output_dir：归档输出目录。
        bundle_stem：离线包目录名前缀。
        platform_id：平台标识。

    返回值：
        Path：最终归档文件路径。

    异常：
        无。
    """

    suffix = ".zip" if platform_id.startswith("windows-") else ".tar.gz"
    return output_dir / f"{bundle_stem}{suffix}"


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

    subprocess.run(
        command,
        check=True,
        env=dict(env) if env is not None else None,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )


def _flatten_constraints_text(constraints_path: Path) -> str:
    """展开 constraints 文件中的嵌套 `-c` / `--constraint` 引用。

    参数：
        constraints_path：待展开的 constraints 文件路径。

    返回值：
        str：展开后的完整 constraints 文本。

    异常：
        FileNotFoundError：constraints 文件不存在时抛出。
        RuntimeError：遇到循环引用时抛出。
        OSError：文件读取失败时抛出。
    """

    visited: set[Path] = set()

    def _expand(path: Path) -> list[str]:
        resolved_path = path.resolve()
        if resolved_path in visited:
            raise RuntimeError(f"constraints 存在循环引用：{resolved_path}")
        if not resolved_path.is_file():
            raise FileNotFoundError(f"未找到 constraints 文件：{resolved_path}")
        visited.add(resolved_path)
        expanded_lines: list[str] = []
        for raw_line in resolved_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("-c ") or stripped.startswith("--constraint "):
                _, include_value = stripped.split(maxsplit=1)
                include_path = (resolved_path.parent / include_value).resolve()
                expanded_lines.extend(_expand(include_path))
                continue
            expanded_lines.append(stripped)
        visited.remove(resolved_path)
        return expanded_lines

    return "\n".join(_expand(constraints_path)).strip() + "\n"


def _write_install_script(bundle_dir: Path, *, version: str, platform_id: str) -> None:
    """写入平台安装脚本。

    参数：
        bundle_dir：离线包工作目录。
        version：项目版本号。
        platform_id：平台标识。

    返回值：
        无。

    异常：
        OSError：文件写入失败时抛出。
    """

    package_spec = f"dayu-agent[browser]=={version}"
    if platform_id.startswith("windows-"):
        script_path = bundle_dir / "install.cmd"
        script_text = (
            "@echo off\r\n"
            "setlocal\r\n"
            "set \"ROOT=%~dp0\"\r\n"
            "if \"%PYTHON_BIN%\"==\"\" set \"PYTHON_BIN=python\"\r\n"
            "\"%PYTHON_BIN%\" -m pip install --no-index --find-links \"%ROOT%wheelhouse\" "
            f"{_PIP_ONLY_BINARY_ALL} --constraint \"%ROOT%constraints.txt\" \"{package_spec}\"\r\n"
        )
    else:
        script_path = bundle_dir / "install.sh"
        script_text = (
            "#!/usr/bin/env sh\n"
            "set -eu\n"
            'ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"\n'
            'PYTHON_BIN="${PYTHON_BIN:-python3}"\n'
            '"$PYTHON_BIN" -m pip install --no-index --find-links "$ROOT/wheelhouse" '
            f'{_PIP_ONLY_BINARY_ALL} --constraint "$ROOT/constraints.txt" "{package_spec}"\n'
        )
    script_path.write_text(script_text, encoding="utf-8", newline="")
    if not platform_id.startswith("windows-"):
        script_path.chmod(0o755)


def _write_bundle_readme(bundle_dir: Path, *, package_name: str, version: str, platform_id: str) -> None:
    """写入离线安装包说明文件。

    参数：
        bundle_dir：离线包工作目录。
        package_name：项目包名。
        version：项目版本号。
        platform_id：平台标识。

    返回值：
        无。

    异常：
        OSError：文件写入失败时抛出。
    """

    install_command = ".\\install.cmd" if platform_id.startswith("windows-") else "./install.sh"
    wheel_name_pattern = f"{package_name.replace('-', '_')}-{version}-*.whl"
    text = f"""# Dayu Agent 离线安装包

版本：`{version}`
平台：`{platform_id}`

## 内容

- `wheelhouse/`：本平台完整 wheelhouse
- 项目 wheel：离线包根目录下的 `{wheel_name_pattern}`
- `constraints.txt`：本平台锁定约束
- `install.sh` / `install.cmd`：离线安装脚本

## 使用方式

1. 解压当前归档。
2. 在解压目录中运行 `{install_command}`。
3. 安装完成后验证：
   - `dayu-cli --help`
   - `dayu-wechat --help`
   - `dayu-cli init --help`
   - `dayu-render --help`

说明：
- 安装脚本只会从当前目录下的 `wheelhouse/` 读取 wheel，不访问公网，也不会现场构建源码包。
- 浏览器回退抓取默认依赖 Chromium；安装完成后仍需单独执行 `playwright install chromium`。
"""
    (bundle_dir / "README.txt").write_text(text, encoding="utf-8")


def _is_source_distribution(artifact_path: Path) -> bool:
    """判断给定文件是否为源码分发包。

    参数：
        artifact_path：待判断的归档路径。

    返回值：
        bool：若为源码分发包则返回 `True`，否则返回 `False`。

    异常：
        无。
    """

    normalized_name = artifact_path.name.lower()
    return any(normalized_name.endswith(suffix) for suffix in _SOURCE_DISTRIBUTION_SUFFIXES)


def _build_source_distribution_wheels(wheelhouse_dir: Path) -> None:
    """把 wheelhouse 中的源码分发包预构建为 wheel 并删除源码归档。

    参数：
        wheelhouse_dir：wheelhouse 目录。

    返回值：
        无。

    异常：
        subprocess.CalledProcessError：`pip wheel` 构建失败时抛出。
        OSError：删除源码归档失败时抛出。
    """

    source_distributions = tuple(
        path
        for path in sorted(wheelhouse_dir.iterdir())
        if path.is_file() and _is_source_distribution(path)
    )
    if not source_distributions:
        return
    _run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse_dir),
            *[str(path) for path in source_distributions],
        ]
    )
    for source_distribution in source_distributions:
        source_distribution.unlink()


def _download_wheelhouse(bundle_dir: Path, *, wheel_path: Path, constraints_path: Path) -> None:
    """下载离线包所需的完整 wheelhouse。

    参数：
        bundle_dir：离线包工作目录。
        wheel_path：项目 wheel 路径。
        constraints_path：锁定约束文件路径。

    返回值：
        无。

    异常：
        subprocess.CalledProcessError：`pip download` 失败时抛出。
        OSError：中间文件写入失败时抛出。
    """

    wheelhouse_dir = bundle_dir / "wheelhouse"
    wheelhouse_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel_path, bundle_dir / wheel_path.name)
    shutil.copy2(wheel_path, wheelhouse_dir / wheel_path.name)
    flattened_constraints_text = _flatten_constraints_text(constraints_path)
    (bundle_dir / "constraints.txt").write_text(flattened_constraints_text, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="dayu-offline-build-") as temp_dir_name:
        requirements_path = Path(temp_dir_name) / "requirements.txt"
        requirements_path.write_text(
            f"dayu-agent[browser] @ {wheel_path.resolve().as_uri()}\n",
            encoding="utf-8",
        )
        _run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--dest",
                str(wheelhouse_dir),
                "--constraint",
                str(bundle_dir / "constraints.txt"),
                "-r",
                str(requirements_path),
            ]
        )
    _build_source_distribution_wheels(wheelhouse_dir)


def _create_archive(bundle_dir: Path, archive_path: Path) -> None:
    """把离线包目录归档为发布文件。

    参数：
        bundle_dir：离线包工作目录。
        archive_path：目标归档文件路径。

    返回值：
        无。

    异常：
        OSError：归档写入失败时抛出。
    """

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive_file:
            for path in bundle_dir.rglob("*"):
                archive_file.write(path, path.relative_to(bundle_dir.parent))
        return
    with tarfile.open(archive_path, "w:gz") as archive_file:
        archive_file.add(bundle_dir, arcname=bundle_dir.name)


def main() -> None:
    """执行离线包构建流程。

    参数：
        无。

    返回值：
        无。

    异常：
        FileNotFoundError：输入文件不存在时抛出。
        RuntimeError：wheel 元数据异常时抛出。
        subprocess.CalledProcessError：依赖下载失败时抛出。
        OSError：文件写入或归档失败时抛出。
    """

    args = _parse_args()
    wheel_path = args.wheel.resolve()
    constraints_path = args.constraints.resolve()
    package_name, version = _read_wheel_metadata(wheel_path)
    bundle_stem = _bundle_stem(package_name, version, args.platform_id)
    archive_path = _archive_path(args.output_dir.resolve(), bundle_stem, args.platform_id)

    with tempfile.TemporaryDirectory(prefix="dayu-offline-bundle-") as temp_dir_name:
        staging_root = Path(temp_dir_name)
        bundle_dir = staging_root / bundle_stem
        bundle_dir.mkdir(parents=True, exist_ok=True)
        _download_wheelhouse(bundle_dir, wheel_path=wheel_path, constraints_path=constraints_path)
        _write_install_script(bundle_dir, version=version, platform_id=args.platform_id)
        _write_bundle_readme(
            bundle_dir,
            package_name=package_name,
            version=version,
            platform_id=args.platform_id,
        )
        _create_archive(bundle_dir, archive_path)

    print(archive_path)


if __name__ == "__main__":
    main()
