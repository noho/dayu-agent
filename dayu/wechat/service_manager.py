"""WeChat 系统 service 管理。

该模块负责 `dayu.wechat` 在受支持平台上的用户级托管：
- macOS: `launchd` LaunchAgent
- Linux: `systemd --user` unit

模块提供两层能力：
- 面向 CLI 的统一 service 管理接口
- 面向各平台后端的细粒度实现函数，便于单测精确覆盖
"""

from __future__ import annotations

import hashlib
import os
import plistlib
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from dayu.log import Log

ServiceBackend = Literal["launchd", "systemd"]

SERVICE_LABEL_PREFIX = "com.dayu.wechat"
STOP_WAIT_INTERVAL_SEC = 0.2
STOP_WAIT_TIMEOUT_SEC = 10.0
_PID_PATTERN = re.compile(r"\bpid\s*=\s*(\d+)")
_SYSTEMD_KEY_VALUE_PATTERN = re.compile(r"^(?P<key>[A-Za-z][A-Za-z0-9]*)=(?P<value>.*)$")
_SYSTEMD_RUNNING_STATES = frozenset({"active", "activating", "reloading"})
_PLATFORM_TO_BACKEND: dict[str, ServiceBackend] = {
    "Darwin": "launchd",
    "Linux": "systemd",
}
MODULE = "APP.WECHAT.SERVICE"


@dataclass(frozen=True)
class ServiceSpec:
    """系统 service 安装规格。"""

    backend: ServiceBackend
    label: str
    definition_path: Path
    working_directory: Path
    program_arguments: tuple[str, ...]
    environment_variables: tuple[tuple[str, str], ...] = ()
    stdout_path: Path | None = None
    stderr_path: Path | None = None


@dataclass(frozen=True)
class ServiceStatus:
    """系统 service 状态快照。"""

    backend: ServiceBackend
    label: str
    definition_path: Path
    installed: bool
    loaded: bool
    pid: int | None = None
    raw_output: str = ""


@dataclass(frozen=True)
class InstalledServiceDefinition:
    """已安装 service definition 的只读视图。"""

    backend: ServiceBackend
    label: str
    definition_path: Path
    program_arguments: tuple[str, ...]
    working_directory: Path | None = None


def is_service_running(status: ServiceStatus) -> bool:
    """判断 service 状态快照是否表示当前已处于运行态。

    Args:
        status: service 状态快照。

    Returns:
        `True` 表示当前无需再次执行 start；`False` 表示仍应执行启动恢复路径。

    Raises:
        无。
    """

    if not status.loaded:
        return False
    if status.backend == "launchd":
        return status.pid is not None
    return True


def resolve_launchd_log_paths(state_dir: Path) -> tuple[Path, Path]:
    """解析 launchd stdout/stderr 日志路径。

    Args:
        state_dir: WeChat 状态目录。

    Returns:
        ``(stdout_path, stderr_path)``。

    Raises:
        ValueError: 当状态目录为空时抛出。
    """

    resolved_state_dir = Path(state_dir).expanduser().resolve()
    normalized = str(resolved_state_dir).strip()
    if not normalized:
        raise ValueError("state_dir 不能为空")
    log_dir = resolved_state_dir / "logs"
    return log_dir / "launchd.stdout.log", log_dir / "launchd.stderr.log"


def detect_service_backend(platform_name: str | None = None) -> ServiceBackend:
    """根据当前平台解析 service backend。

    Args:
        platform_name: 可选的平台名覆盖；未传时使用当前运行平台。

    Returns:
        当前平台对应的 service backend。

    Raises:
        RuntimeError: 当当前平台尚未支持时抛出。
    """

    resolved_platform_name = str(platform_name or __import__("platform").system()).strip()
    backend = _PLATFORM_TO_BACKEND.get(resolved_platform_name)
    if backend is None:
        raise RuntimeError("当前仅支持 macOS launchd 和 Linux systemd --user；Windows 暂未实现")
    return backend


def describe_service_backend(backend: ServiceBackend) -> str:
    """返回面向用户的 backend 描述。

    Args:
        backend: service backend。

    Returns:
        适合打印到 CLI 的简短描述。

    Raises:
        ValueError: 当 backend 非法时抛出。
    """

    if backend == "launchd":
        return "macOS launchd"
    if backend == "systemd":
        return "Linux systemd --user"
    raise ValueError(f"未知 service backend: {backend}")


def build_service_label(state_dir: Path) -> str:
    """根据状态目录生成稳定 service label。

    Args:
        state_dir: WeChat 状态目录。

    Returns:
        稳定的 service label。

    Raises:
        ValueError: 当状态目录为空时抛出。
    """

    normalized = str(Path(state_dir).expanduser().resolve()).strip()
    if not normalized:
        raise ValueError("state_dir 不能为空")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{SERVICE_LABEL_PREFIX}.{digest}"


def build_service_log_lines(*, label: str, state_dir: Path, backend: ServiceBackend) -> tuple[str, ...]:
    """构建适合 CLI 打印的日志定位信息。

    Args:
        label: service label。
        state_dir: WeChat 状态目录。
        backend: service backend。

    Returns:
        适合逐行打印的日志提示。

    Raises:
        ValueError: 当 backend 非法时抛出。
    """

    if backend == "launchd":
        stdout_path, stderr_path = resolve_launchd_log_paths(state_dir)
        return (
            f"log_stdout: {stdout_path}",
            f"log_stderr: {stderr_path}",
        )
    if backend == "systemd":
        unit_name = _build_systemd_unit_name(label)
        follow_command = shlex.join(["journalctl", "--user", "-u", unit_name, "-f"])
        return (
            "log_backend: journal",
            f"log_follow_command: {follow_command}",
        )
    raise ValueError(f"未知 service backend: {backend}")


def resolve_service_definition_path(label: str, *, backend: ServiceBackend) -> Path:
    """解析当前 backend 的 service 定义文件路径。

    Args:
        label: service label。
        backend: service backend。

    Returns:
        service 定义文件绝对路径。

    Raises:
        ValueError: 当 backend 非法时抛出。
    """

    if backend == "launchd":
        return resolve_launch_agent_plist_path(label)
    if backend == "systemd":
        return resolve_systemd_user_unit_path(label)
    raise ValueError(f"未知 service backend: {backend}")


def list_installed_service_definitions(backend: ServiceBackend) -> tuple[InstalledServiceDefinition, ...]:
    """枚举当前 backend 下已写入磁盘的 WeChat service definition。

    Args:
        backend: service backend。

    Returns:
        已安装 definition 视图元组。

    Raises:
        ValueError: 当 backend 非法时抛出。
    """

    if backend == "launchd":
        return _list_launchd_service_definitions()
    if backend == "systemd":
        return _list_systemd_service_definitions()
    raise ValueError(f"未知 service backend: {backend}")


def _list_launchd_service_definitions() -> tuple[InstalledServiceDefinition, ...]:
    """枚举 launchd WeChat service definition。"""

    launch_agent_root = resolve_launch_agent_plist_path(f"{SERVICE_LABEL_PREFIX}.probe").parent
    if not launch_agent_root.is_dir():
        return ()
    definitions: list[InstalledServiceDefinition] = []
    for plist_path in sorted(launch_agent_root.glob(f"{SERVICE_LABEL_PREFIX}.*.plist")):
        definition = _read_launchd_service_definition(plist_path)
        if definition is None:
            continue
        definitions.append(definition)
    return tuple(definitions)


def _read_launchd_service_definition(plist_path: Path) -> InstalledServiceDefinition | None:
    """读取单个 launchd plist 并提取 WeChat definition 视图。"""

    try:
        with Path(plist_path).expanduser().resolve().open("rb") as handle:
            payload = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException):
        return None
    if not isinstance(payload, dict):
        return None
    label = str(payload.get("Label") or "").strip()
    if not _is_wechat_service_label(label):
        return None
    raw_program_arguments = payload.get("ProgramArguments")
    if not isinstance(raw_program_arguments, list):
        return None
    program_arguments = _normalize_program_arguments(raw_program_arguments)
    if not program_arguments:
        return None
    raw_working_directory = payload.get("WorkingDirectory")
    working_directory = _normalize_optional_path(raw_working_directory)
    return InstalledServiceDefinition(
        backend="launchd",
        label=label,
        definition_path=Path(plist_path).expanduser().resolve(),
        program_arguments=program_arguments,
        working_directory=working_directory,
    )


def _list_systemd_service_definitions() -> tuple[InstalledServiceDefinition, ...]:
    """枚举 systemd WeChat service definition。"""

    unit_root = resolve_systemd_user_unit_path(f"{SERVICE_LABEL_PREFIX}.probe").parent
    if not unit_root.is_dir():
        return ()
    definitions: list[InstalledServiceDefinition] = []
    for unit_path in sorted(unit_root.glob(f"{SERVICE_LABEL_PREFIX}.*.service")):
        definition = _read_systemd_service_definition(unit_path)
        if definition is None:
            continue
        definitions.append(definition)
    return tuple(definitions)


def _read_systemd_service_definition(unit_path: Path) -> InstalledServiceDefinition | None:
    """读取单个 systemd unit 并提取 WeChat definition 视图。"""

    resolved_unit_path = Path(unit_path).expanduser().resolve()
    label = resolved_unit_path.stem
    if not _is_wechat_service_label(label):
        return None
    try:
        unit_text = resolved_unit_path.read_text(encoding="utf-8")
    except OSError:
        return None
    working_directory_text, exec_start_text = _parse_systemd_unit_runtime_fields(unit_text)
    if exec_start_text is None:
        return None
    try:
        program_arguments = tuple(shlex.split(exec_start_text))
    except ValueError:
        return None
    if not program_arguments:
        return None
    return InstalledServiceDefinition(
        backend="systemd",
        label=label,
        definition_path=resolved_unit_path,
        program_arguments=program_arguments,
        working_directory=_normalize_optional_path(working_directory_text),
    )


def _parse_systemd_unit_runtime_fields(raw_unit_text: str) -> tuple[str | None, str | None]:
    """从 systemd unit 文本中提取运行时字段。"""

    working_directory: str | None = None
    exec_start: str | None = None
    for raw_line in raw_unit_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("WorkingDirectory="):
            working_directory = line.removeprefix("WorkingDirectory=").strip() or None
            continue
        if line.startswith("ExecStart="):
            exec_start = line.removeprefix("ExecStart=").strip() or None
    return working_directory, exec_start


def _normalize_program_arguments(raw_program_arguments: list[object]) -> tuple[str, ...]:
    """把 definition 中的 ProgramArguments 标准化为字符串元组。"""

    normalized_arguments: list[str] = []
    for raw_argument in raw_program_arguments:
        argument = str(raw_argument or "").strip()
        if not argument:
            continue
        normalized_arguments.append(argument)
    return tuple(normalized_arguments)


def _normalize_optional_path(raw_path: object) -> Path | None:
    """把可选路径字段标准化为绝对路径。"""

    normalized_path = str(raw_path or "").strip()
    if not normalized_path:
        return None
    return Path(normalized_path).expanduser().resolve()


def _is_wechat_service_label(label: str) -> bool:
    """判断 label 是否属于 Dayu WeChat service。"""

    return bool(label.startswith(f"{SERVICE_LABEL_PREFIX}."))


def build_service_spec(
    *,
    state_dir: Path,
    working_directory: Path,
    python_executable: str,
    run_arguments: Sequence[str],
    environment_variables: Mapping[str, str] | None = None,
    backend: ServiceBackend,
) -> ServiceSpec:
    """构建当前 backend 的 service 安装规格。

    Args:
        state_dir: WeChat 状态目录。
        working_directory: 运行目录。
        python_executable: Python 可执行文件路径。
        run_arguments: `python -m dayu.wechat` 后续参数。
        backend: service backend。

    Returns:
        统一的 service 安装规格。

    Raises:
        ValueError: 当 backend 非法或关键参数为空时抛出。
    """

    if backend == "launchd":
        return build_launchd_service_spec(
            state_dir=state_dir,
            working_directory=working_directory,
            python_executable=python_executable,
            run_arguments=run_arguments,
            environment_variables=environment_variables,
        )
    if backend == "systemd":
        return build_systemd_service_spec(
            state_dir=state_dir,
            working_directory=working_directory,
            python_executable=python_executable,
            run_arguments=run_arguments,
            environment_variables=environment_variables,
        )
    raise ValueError(f"未知 service backend: {backend}")


def _normalize_environment_variables(environment_variables: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    """标准化 service 需要注入的环境变量映射。

    Args:
        environment_variables: 原始环境变量映射。

    Returns:
        过滤空 key/value 后的稳定有序键值对。

    Raises:
        无。
    """

    normalized_items: list[tuple[str, str]] = []
    for raw_key, raw_value in dict(environment_variables or {}).items():
        key = str(raw_key or "").strip()
        value = str(raw_value or "")
        if not key or not value:
            continue
        normalized_items.append((key, value))
    normalized_items.sort(key=lambda item: item[0])
    return tuple(normalized_items)


def install_service(spec: ServiceSpec) -> Path:
    """写入 service 定义文件。

    Args:
        spec: service 安装规格。

    Returns:
        写入后的定义文件路径。

    Raises:
        OSError: 当写文件失败时抛出。
        RuntimeError: 当后台管理命令失败时抛出。
    """

    if spec.backend == "launchd":
        return install_launchd_service(spec)
    if spec.backend == "systemd":
        return install_systemd_service(spec)
    raise ValueError(f"未知 service backend: {spec.backend}")


def query_service_status(*, label: str, definition_path: Path, backend: ServiceBackend) -> ServiceStatus:
    """查询当前 backend 的 service 状态。

    Args:
        label: service label。
        definition_path: service 定义文件路径。
        backend: service backend。

    Returns:
        统一的 service 状态快照。

    Raises:
        RuntimeError: 当后台管理命令执行失败时抛出。
    """

    if backend == "launchd":
        return query_launchd_service_status(label=label, plist_path=definition_path)
    if backend == "systemd":
        return query_systemd_service_status(label=label, unit_path=definition_path)
    raise ValueError(f"未知 service backend: {backend}")


def start_service(*, label: str, definition_path: Path, backend: ServiceBackend) -> None:
    """启动当前 backend 的 service。

    Args:
        label: service label。
        definition_path: service 定义文件路径。
        backend: service backend。

    Returns:
        无。

    Raises:
        FileNotFoundError: 当定义文件不存在时抛出。
        RuntimeError: 当后台管理命令执行失败时抛出。
    """

    if backend == "launchd":
        start_launchd_service(label=label, plist_path=definition_path)
        return
    if backend == "systemd":
        start_systemd_service(label=label, unit_path=definition_path)
        return
    raise ValueError(f"未知 service backend: {backend}")


def restart_service(*, label: str, definition_path: Path, backend: ServiceBackend) -> None:
    """重启当前 backend 的 service。

    Args:
        label: service label。
        definition_path: service 定义文件路径。
        backend: service backend。

    Returns:
        无。

    Raises:
        FileNotFoundError: 当定义文件不存在时抛出。
        RuntimeError: 当后台管理命令执行失败时抛出。
    """

    if backend == "launchd":
        restart_launchd_service(label=label, plist_path=definition_path)
        return
    if backend == "systemd":
        restart_systemd_service(label=label, unit_path=definition_path)
        return
    raise ValueError(f"未知 service backend: {backend}")


def stop_service(*, label: str, definition_path: Path, backend: ServiceBackend) -> bool:
    """停止当前 backend 的 service。

    Args:
        label: service label。
        definition_path: service 定义文件路径。
        backend: service backend。

    Returns:
        `True` 表示本次执行了停止动作；`False` 表示服务本来就未运行。

    Raises:
        RuntimeError: 当后台管理命令执行失败时抛出。
    """

    if backend == "launchd":
        return stop_launchd_service(label=label, plist_path=definition_path)
    if backend == "systemd":
        return stop_systemd_service(label=label, unit_path=definition_path)
    raise ValueError(f"未知 service backend: {backend}")


def uninstall_service(*, label: str, definition_path: Path, backend: ServiceBackend) -> bool:
    """卸载当前 backend 的 service。

    Args:
        label: service label。
        definition_path: service 定义文件路径。
        backend: service backend。

    Returns:
        `True` 表示确实删除了定义文件；`False` 表示定义文件原本不存在。

    Raises:
        RuntimeError: 当后台管理命令执行失败时抛出。
        OSError: 当删除定义文件失败时抛出。
    """

    if backend == "launchd":
        return uninstall_launchd_service(label=label, plist_path=definition_path)
    if backend == "systemd":
        return uninstall_systemd_service(label=label, unit_path=definition_path)
    raise ValueError(f"未知 service backend: {backend}")


def resolve_launch_agent_plist_path(label: str) -> Path:
    """解析 LaunchAgent plist 文件路径。

    Args:
        label: service label。

    Returns:
        plist 绝对路径。

    Raises:
        ValueError: 当 label 为空时抛出。
    """

    normalized = str(label or "").strip()
    if not normalized:
        raise ValueError("label 不能为空")
    return Path("~/Library/LaunchAgents").expanduser().resolve() / f"{normalized}.plist"


def build_launchd_service_spec(
    *,
    state_dir: Path,
    working_directory: Path,
    python_executable: str,
    run_arguments: Sequence[str],
    environment_variables: Mapping[str, str] | None = None,
) -> ServiceSpec:
    """构建 launchd 安装规格。

    Args:
        state_dir: WeChat 状态目录。
        working_directory: 运行目录。
        python_executable: Python 可执行文件路径。
        run_arguments: `python -m dayu.wechat` 后续参数。

    Returns:
        launchd 安装规格。

    Raises:
        ValueError: 当参数为空时抛出。
    """

    resolved_state_dir = Path(state_dir).expanduser().resolve()
    resolved_working_directory = Path(working_directory).expanduser().resolve()
    executable = str(python_executable or "").strip()
    if not executable:
        raise ValueError("python_executable 不能为空")
    label = build_service_label(resolved_state_dir)
    stdout_path, stderr_path = resolve_launchd_log_paths(resolved_state_dir)
    return ServiceSpec(
        backend="launchd",
        label=label,
        definition_path=resolve_launch_agent_plist_path(label),
        working_directory=resolved_working_directory,
        program_arguments=tuple([executable, "-m", "dayu.wechat", *run_arguments]),
        environment_variables=_normalize_environment_variables(environment_variables),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def install_launchd_service(spec: ServiceSpec) -> Path:
    """写入 launchd plist 文件。

    Args:
        spec: launchd 安装规格。

    Returns:
        写入后的 plist 路径。

    Raises:
        OSError: 当写文件失败时抛出。
        ValueError: 当规格不属于 launchd 时抛出。
    """

    if spec.backend != "launchd":
        raise ValueError("install_launchd_service 仅接受 launchd 规格")
    if spec.stdout_path is None or spec.stderr_path is None:
        raise ValueError("launchd service 需要 stdout_path / stderr_path")
    spec.definition_path.parent.mkdir(parents=True, exist_ok=True)
    spec.stdout_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Label": spec.label,
        "ProgramArguments": list(spec.program_arguments),
        "WorkingDirectory": str(spec.working_directory),
        "RunAtLoad": True,
        "StandardOutPath": str(spec.stdout_path),
        "StandardErrorPath": str(spec.stderr_path),
        "ProcessType": "Background",
        "EnvironmentVariables": {
            "PYTHONUNBUFFERED": "1",
            **{key: value for key, value in spec.environment_variables},
        },
    }
    with spec.definition_path.open("wb") as handle:
        plistlib.dump(payload, handle, sort_keys=True)
    return spec.definition_path


def query_launchd_service_status(*, label: str, plist_path: Path) -> ServiceStatus:
    """查询 launchd 服务状态。

    Args:
        label: service label。
        plist_path: plist 路径。

    Returns:
        服务状态快照。

    Raises:
        RuntimeError: 当 `launchctl` 执行异常时抛出。
    """

    resolved_plist_path = Path(plist_path).expanduser().resolve()
    installed = resolved_plist_path.exists()
    if not installed:
        return ServiceStatus(
            backend="launchd",
            label=label,
            definition_path=resolved_plist_path,
            installed=False,
            loaded=False,
        )
    target = _build_launchctl_service_target(label)
    result = _run_launchctl(["print", target], check=False)
    combined_output = _combine_process_output(result)
    if result.returncode != 0:
        return ServiceStatus(
            backend="launchd",
            label=label,
            definition_path=resolved_plist_path,
            installed=True,
            loaded=False,
            raw_output=combined_output,
        )
    return ServiceStatus(
        backend="launchd",
        label=label,
        definition_path=resolved_plist_path,
        installed=True,
        loaded=True,
        pid=_extract_launchd_pid(combined_output),
        raw_output=combined_output,
    )


def start_launchd_service(*, label: str, plist_path: Path) -> None:
    """启动 launchd 服务。

    Args:
        label: service label。
        plist_path: plist 路径。

    Returns:
        无。

    Raises:
        FileNotFoundError: 当 plist 不存在时抛出。
        RuntimeError: 当 `launchctl` 执行失败时抛出。
    """

    resolved_plist_path = Path(plist_path).expanduser().resolve()
    if not resolved_plist_path.exists():
        raise FileNotFoundError(f"launchd plist 不存在: {resolved_plist_path}")
    status = query_launchd_service_status(label=label, plist_path=resolved_plist_path)
    if status.loaded:
        if status.pid is None:
            _run_launchctl(["kickstart", _build_launchctl_service_target(label)], check=True)
        return
    _run_launchctl(["bootstrap", _build_launchctl_domain_target(), str(resolved_plist_path)], check=True)


def restart_launchd_service(*, label: str, plist_path: Path) -> None:
    """重启 launchd 服务。

    Args:
        label: service label。
        plist_path: plist 路径。

    Returns:
        无。

    Raises:
        FileNotFoundError: 当 plist 不存在时抛出。
        RuntimeError: 当 `launchctl` 执行失败时抛出。
    """

    resolved_plist_path = Path(plist_path).expanduser().resolve()
    if not resolved_plist_path.exists():
        raise FileNotFoundError(f"launchd plist 不存在: {resolved_plist_path}")
    status = query_launchd_service_status(label=label, plist_path=resolved_plist_path)
    if status.loaded:
        _run_launchctl(["kickstart", "-k", _build_launchctl_service_target(label)], check=True)
        return
    _run_launchctl(["bootstrap", _build_launchctl_domain_target(), str(resolved_plist_path)], check=True)


def stop_launchd_service(*, label: str, plist_path: Path) -> bool:
    """停止 launchd 服务。

    Args:
        label: service label。
        plist_path: plist 路径。

    Returns:
        `True` 表示本次执行了停止动作；`False` 表示服务本来就未加载。

    Raises:
        RuntimeError: 当 `launchctl` 执行失败时抛出。
    """

    resolved_plist_path = Path(plist_path).expanduser().resolve()
    status = query_launchd_service_status(label=label, plist_path=resolved_plist_path)
    if not status.loaded:
        return False
    if status.pid is not None:
        _run_launchctl(["kill", "SIGTERM", _build_launchctl_service_target(label)], check=True)
        stopped_gracefully = _wait_for_launchd_service_process_exit(
            label=label,
            plist_path=resolved_plist_path,
            timeout_sec=STOP_WAIT_TIMEOUT_SEC,
        )
        if not stopped_gracefully:
            Log.warning(
                f"WeChat daemon 在 {STOP_WAIT_TIMEOUT_SEC:.1f}s 内未完成优雅退出，继续执行 launchd bootout",
                module=MODULE,
            )
    _run_launchctl(["bootout", _build_launchctl_domain_target(), str(resolved_plist_path)], check=True)
    return True


def _wait_for_launchd_service_process_exit(*, label: str, plist_path: Path, timeout_sec: float) -> bool:
    """等待 launchd service 进程在收到 SIGTERM 后退出。

    Args:
        label: service label。
        plist_path: plist 路径。
        timeout_sec: 最长等待秒数。

    Returns:
        `True` 表示观察到 pid 已退出；`False` 表示等待超时。

    Raises:
        无。
    """

    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while time.monotonic() < deadline:
        status = query_launchd_service_status(label=label, plist_path=plist_path)
        if status.pid is None:
            return True
        time.sleep(STOP_WAIT_INTERVAL_SEC)
    return False


def uninstall_launchd_service(*, label: str, plist_path: Path) -> bool:
    """卸载 launchd 服务。

    Args:
        label: service label。
        plist_path: plist 路径。

    Returns:
        `True` 表示确实删除了 plist；`False` 表示 plist 原本不存在。

    Raises:
        RuntimeError: 当 `launchctl` 执行失败时抛出。
        OSError: 当删除 plist 失败时抛出。
    """

    resolved_plist_path = Path(plist_path).expanduser().resolve()
    status = query_launchd_service_status(label=label, plist_path=resolved_plist_path)
    if status.loaded:
        _run_launchctl(["bootout", _build_launchctl_domain_target(), str(resolved_plist_path)], check=True)
    if not resolved_plist_path.exists():
        return False
    resolved_plist_path.unlink()
    return True


def resolve_systemd_user_unit_path(label: str) -> Path:
    """解析 systemd user unit 文件路径。

    Args:
        label: service label。

    Returns:
        unit 文件绝对路径。

    Raises:
        ValueError: 当 label 为空时抛出。
    """

    normalized = str(label or "").strip()
    if not normalized:
        raise ValueError("label 不能为空")
    return Path("~/.config/systemd/user").expanduser().resolve() / f"{normalized}.service"


def build_systemd_service_spec(
    *,
    state_dir: Path,
    working_directory: Path,
    python_executable: str,
    run_arguments: Sequence[str],
    environment_variables: Mapping[str, str] | None = None,
) -> ServiceSpec:
    """构建 systemd --user 安装规格。

    Args:
        state_dir: WeChat 状态目录。
        working_directory: 运行目录。
        python_executable: Python 可执行文件路径。
        run_arguments: `python -m dayu.wechat` 后续参数。

    Returns:
        systemd user service 安装规格。

    Raises:
        ValueError: 当参数为空时抛出。
    """

    resolved_state_dir = Path(state_dir).expanduser().resolve()
    resolved_working_directory = Path(working_directory).expanduser().resolve()
    executable = str(python_executable or "").strip()
    if not executable:
        raise ValueError("python_executable 不能为空")
    label = build_service_label(resolved_state_dir)
    return ServiceSpec(
        backend="systemd",
        label=label,
        definition_path=resolve_systemd_user_unit_path(label),
        working_directory=resolved_working_directory,
        program_arguments=tuple([executable, "-m", "dayu.wechat", *run_arguments]),
        environment_variables=_normalize_environment_variables(environment_variables),
    )


def _render_systemd_environment_line(key: str, value: str) -> str:
    """渲染单条 systemd `Environment=` 配置。

    Args:
        key: 环境变量名。
        value: 环境变量值。

    Returns:
        一条可直接写入 unit 的 `Environment=` 配置行。

    Raises:
        无。
    """

    assignment = f"{key}={value}"
    escaped = assignment.replace("\\", "\\\\").replace('"', '\\"')
    return f'Environment="{escaped}"'


def install_systemd_service(spec: ServiceSpec) -> Path:
    """写入 systemd user unit 文件并刷新 daemon-reload。

    Args:
        spec: systemd 安装规格。

    Returns:
        写入后的 unit 文件路径。

    Raises:
        OSError: 当写文件失败时抛出。
        RuntimeError: 当 `systemctl --user daemon-reload` 失败时抛出。
        ValueError: 当规格不属于 systemd 时抛出。
    """

    if spec.backend != "systemd":
        raise ValueError("install_systemd_service 仅接受 systemd 规格")
    spec.definition_path.parent.mkdir(parents=True, exist_ok=True)
    spec.definition_path.write_text(_render_systemd_unit(spec), encoding="utf-8")
    _run_systemctl_user(["daemon-reload"], check=True)
    return spec.definition_path


def query_systemd_service_status(*, label: str, unit_path: Path) -> ServiceStatus:
    """查询 systemd --user 服务状态。

    Args:
        label: service label。
        unit_path: unit 文件路径。

    Returns:
        服务状态快照。

    Raises:
        RuntimeError: 当 `systemctl` 执行异常时抛出。
    """

    resolved_unit_path = Path(unit_path).expanduser().resolve()
    installed = resolved_unit_path.exists()
    if not installed:
        return ServiceStatus(
            backend="systemd",
            label=label,
            definition_path=resolved_unit_path,
            installed=False,
            loaded=False,
        )
    result = _run_systemctl_user(
        [
            "show",
            _build_systemd_unit_name(label),
            "--property=LoadState",
            "--property=ActiveState",
            "--property=MainPID",
        ],
        check=False,
    )
    combined_output = _combine_process_output(result)
    if result.returncode != 0:
        return ServiceStatus(
            backend="systemd",
            label=label,
            definition_path=resolved_unit_path,
            installed=True,
            loaded=False,
            raw_output=combined_output,
        )
    fields = _parse_systemd_show_output(combined_output)
    active_state = str(fields.get("ActiveState", "")).strip()
    pid = _parse_systemd_pid(fields.get("MainPID"))
    return ServiceStatus(
        backend="systemd",
        label=label,
        definition_path=resolved_unit_path,
        installed=True,
        loaded=active_state in _SYSTEMD_RUNNING_STATES,
        pid=pid,
        raw_output=combined_output,
    )


def start_systemd_service(*, label: str, unit_path: Path) -> None:
    """启动 systemd --user 服务。

    Args:
        label: service label。
        unit_path: unit 文件路径。

    Returns:
        无。

    Raises:
        FileNotFoundError: 当 unit 文件不存在时抛出。
        RuntimeError: 当 `systemctl` 执行失败时抛出。
    """

    resolved_unit_path = Path(unit_path).expanduser().resolve()
    if not resolved_unit_path.exists():
        raise FileNotFoundError(f"systemd unit 不存在: {resolved_unit_path}")
    _run_systemctl_user(["daemon-reload"], check=True)
    _run_systemctl_user(["start", _build_systemd_unit_name(label)], check=True)


def restart_systemd_service(*, label: str, unit_path: Path) -> None:
    """重启 systemd --user 服务。

    Args:
        label: service label。
        unit_path: unit 文件路径。

    Returns:
        无。

    Raises:
        FileNotFoundError: 当 unit 文件不存在时抛出。
        RuntimeError: 当 `systemctl` 执行失败时抛出。
    """

    resolved_unit_path = Path(unit_path).expanduser().resolve()
    if not resolved_unit_path.exists():
        raise FileNotFoundError(f"systemd unit 不存在: {resolved_unit_path}")
    _run_systemctl_user(["daemon-reload"], check=True)
    status = query_systemd_service_status(label=label, unit_path=resolved_unit_path)
    command = "restart" if status.loaded else "start"
    _run_systemctl_user([command, _build_systemd_unit_name(label)], check=True)


def stop_systemd_service(*, label: str, unit_path: Path) -> bool:
    """停止 systemd --user 服务。

    Args:
        label: service label。
        unit_path: unit 文件路径。

    Returns:
        `True` 表示本次执行了停止动作；`False` 表示服务本来就未运行。

    Raises:
        RuntimeError: 当 `systemctl` 执行失败时抛出。
    """

    resolved_unit_path = Path(unit_path).expanduser().resolve()
    status = query_systemd_service_status(label=label, unit_path=resolved_unit_path)
    if not status.loaded:
        return False
    _run_systemctl_user(["stop", _build_systemd_unit_name(label)], check=True)
    return True


def uninstall_systemd_service(*, label: str, unit_path: Path) -> bool:
    """卸载 systemd --user 服务。

    Args:
        label: service label。
        unit_path: unit 文件路径。

    Returns:
        `True` 表示确实删除了 unit 文件；`False` 表示 unit 文件原本不存在。

    Raises:
        RuntimeError: 当 `systemctl` 执行失败时抛出。
        OSError: 当删除 unit 文件失败时抛出。
    """

    resolved_unit_path = Path(unit_path).expanduser().resolve()
    status = query_systemd_service_status(label=label, unit_path=resolved_unit_path)
    if status.loaded:
        _run_systemctl_user(["stop", _build_systemd_unit_name(label)], check=True)
    if not resolved_unit_path.exists():
        return False
    resolved_unit_path.unlink()
    _run_systemctl_user(["daemon-reload"], check=True)
    return True


def _render_systemd_unit(spec: ServiceSpec) -> str:
    """渲染 systemd user unit 内容。

    Args:
        spec: systemd 安装规格。

    Returns:
        unit 文件文本。

    Raises:
        ValueError: 当规格不属于 systemd 时抛出。
    """

    if spec.backend != "systemd":
        raise ValueError("_render_systemd_unit 仅接受 systemd 规格")
    exec_start = shlex.join(spec.program_arguments)
    return "\n".join(
        [
            "[Unit]",
            f"Description=Dayu WeChat daemon ({spec.label})",
            "After=default.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={spec.working_directory}",
            f"ExecStart={exec_start}",
            "Environment=PYTHONUNBUFFERED=1",
            *[_render_systemd_environment_line(key, value) for key, value in spec.environment_variables],
            "KillSignal=SIGTERM",
            f"TimeoutStopSec={int(max(STOP_WAIT_TIMEOUT_SEC, 1.0))}",
            "Restart=no",
            "StandardOutput=journal",
            "StandardError=journal",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ]
    )


def _build_systemd_unit_name(label: str) -> str:
    """构造 systemd unit 名称。

    Args:
        label: service label。

    Returns:
        unit 名称。

    Raises:
        无。
    """

    return f"{label}.service"


def _parse_systemd_show_output(raw_output: str) -> dict[str, str]:
    """解析 `systemctl show` 的 key=value 输出。

    Args:
        raw_output: 原始输出文本。

    Returns:
        解析后的字段字典。

    Raises:
        无。
    """

    parsed: dict[str, str] = {}
    for line in str(raw_output or "").splitlines():
        match = _SYSTEMD_KEY_VALUE_PATTERN.match(line.strip())
        if match is None:
            continue
        parsed[match.group("key")] = match.group("value")
    return parsed


def _parse_systemd_pid(raw_value: str | None) -> int | None:
    """解析 systemd `MainPID` 字段。

    Args:
        raw_value: `MainPID` 原始文本。

    Returns:
        pid；无效或为 0 时返回 `None`。

    Raises:
        无。
    """

    normalized = str(raw_value or "").strip()
    if not normalized:
        return None
    if not normalized.isdigit():
        return None
    pid = int(normalized)
    if pid <= 0:
        return None
    return pid


def _build_launchctl_domain_target() -> str:
    """构造当前用户的 launchctl domain target。

    Args:
        无。

    Returns:
        launchctl domain target。

    Raises:
        RuntimeError: 在不支持 launchctl 的平台上调用时抛出。
    """

    if sys.platform == "win32":
        raise RuntimeError("launchctl 仅在 macOS 上可用")
    return f"gui/{os.getuid()}"


def _build_launchctl_service_target(label: str) -> str:
    """构造 launchctl service target。

    Args:
        label: service label。

    Returns:
        launchctl service target。

    Raises:
        无。
    """

    return f"{_build_launchctl_domain_target()}/{label}"


def _run_launchctl(arguments: Sequence[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    """执行一次 launchctl 命令。

    Args:
        arguments: launchctl 参数列表。
        check: 失败时是否抛异常。

    Returns:
        子进程执行结果。

    Raises:
        RuntimeError: 当命令失败且 `check=True` 时抛出。
    """

    result = subprocess.run(
        ["launchctl", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError("launchctl 执行失败: " + _combine_process_output(result).strip())
    return result


def _run_systemctl_user(arguments: Sequence[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    """执行一次 `systemctl --user` 命令。

    Args:
        arguments: `systemctl --user` 参数列表。
        check: 失败时是否抛异常。

    Returns:
        子进程执行结果。

    Raises:
        RuntimeError: 当命令失败且 `check=True` 时抛出。
    """

    result = subprocess.run(
        ["systemctl", "--user", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError("systemctl --user 执行失败: " + _combine_process_output(result).strip())
    return result


def _combine_process_output(result: subprocess.CompletedProcess[str]) -> str:
    """合并子进程 stdout/stderr。

    Args:
        result: 子进程执行结果。

    Returns:
        合并后的文本。

    Raises:
        无。
    """

    stdout = str(result.stdout or "").strip()
    stderr = str(result.stderr or "").strip()
    if stdout and stderr:
        return f"{stdout}\n{stderr}"
    return stdout or stderr


def _extract_launchd_pid(raw_output: str) -> int | None:
    """从 `launchctl print` 输出中提取 pid。

    Args:
        raw_output: 原始输出文本。

    Returns:
        pid；未匹配到时返回 `None`。

    Raises:
        无。
    """

    match = _PID_PATTERN.search(str(raw_output or ""))
    if not match:
        return None
    return int(match.group(1))


__all__ = [
    "ServiceBackend",
    "build_service_log_lines",
    "InstalledServiceDefinition",
    "ServiceSpec",
    "ServiceStatus",
    "build_service_label",
    "build_service_spec",
    "build_launchd_service_spec",
    "build_systemd_service_spec",
    "describe_service_backend",
    "detect_service_backend",
    "is_service_running",
    "install_service",
    "install_launchd_service",
    "install_systemd_service",
    "list_installed_service_definitions",
    "query_service_status",
    "query_launchd_service_status",
    "query_systemd_service_status",
    "resolve_launch_agent_plist_path",
    "resolve_launchd_log_paths",
    "resolve_service_definition_path",
    "resolve_systemd_user_unit_path",
    "restart_service",
    "restart_launchd_service",
    "restart_systemd_service",
    "start_service",
    "start_launchd_service",
    "start_systemd_service",
    "stop_service",
    "stop_launchd_service",
    "stop_systemd_service",
    "uninstall_service",
    "uninstall_launchd_service",
    "uninstall_systemd_service",
]
