"""跨平台进程判活与进程身份采集辅助。

`os.kill(pid, 0)` 仅在 POSIX 上是一个「发送 0 号信号」的存在性探测；
在 Windows 上 CPython 将 signal=0 路由到 `TerminateProcess`，无法用于判活，
对不存在的 PID 也不会稳定抛出 `ProcessLookupError`，可能阻塞或误判。

本模块提供两类稳定入口：

判活：
    - `is_pid_alive(pid)`：仅基于 PID 的存活判定，保持向后语义。

进程身份（OwnerIdentity，#106）：
    - `OwnerIdentity`：`pid + process_start_time + boot_id` 三元组。
    - `current_owner_identity()`：采集当前进程的完整身份。
    - `is_owner_identity_alive(identity)`：完整身份等值 + PID 判活；
      任一字段为 NULL 时退化为其余非 NULL 字段比对，最差等价于 `is_pid_alive`。

所有平台细节（私有 `_get_*` helper）一律封装在本模块内部，**禁止上层模块**
（`run_registry` / `concurrency` / `host` / 其它）直接 import 私有 helper，
统一通过上述四个公开入口与 `OwnerIdentity` 数据类进行交互。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# Windows Win32 API 常量
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5
_ERROR_INVALID_PARAMETER = 87


def _is_pid_alive_posix(pid: int) -> bool:
    """POSIX 平台下基于 `os.kill(pid, 0)` 的存在性探测。

    Args:
        pid: 目标进程 ID。

    Returns:
        `True` 表示进程存活；`False` 表示进程已退出。

    Raises:
        OSError: 底层非预期错误原样抛出。
    """

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # 进程存在但无权限发信号。
        return True
    return True


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL

    def _is_pid_alive_windows(pid: int) -> bool:
        """Windows 平台下基于 Win32 `OpenProcess` + `GetExitCodeProcess` 的存活判定。

        Args:
            pid: 目标进程 ID。

        Returns:
            `True` 表示进程仍在运行；`False` 表示 PID 不存在或进程已退出。

        Raises:
            OSError: 底层 Win32 调用发生非预期错误时抛出。
        """

        handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            last_error = ctypes.get_last_error()
            if last_error == _ERROR_INVALID_PARAMETER:
                # 对应不存在或已回收的 PID。
                return False
            if last_error == _ERROR_ACCESS_DENIED:
                # 进程存在但当前令牌无权限查询。
                return True
            raise OSError(last_error, f"OpenProcess 失败：winerror={last_error}")
        try:
            exit_code = wintypes.DWORD()
            if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                last_error = ctypes.get_last_error()
                raise OSError(last_error, f"GetExitCodeProcess 失败：winerror={last_error}")
            return exit_code.value == _STILL_ACTIVE
        finally:
            _kernel32.CloseHandle(handle)


def is_pid_alive(pid: int) -> bool:
    """跨平台判断指定 PID 对应的进程是否存活。

    Args:
        pid: 目标进程 ID。

    Returns:
        `True` 表示进程仍在运行；`False` 表示进程已退出或 PID 不存在。

    Raises:
        OSError: 底层系统调用发生非预期错误时抛出。
    """

    if sys.platform == "win32":
        return _is_pid_alive_windows(pid)
    return _is_pid_alive_posix(pid)


# ---------------------------------------------------------------------------
# OwnerIdentity（#106）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnerIdentity:
    """进程身份三元组，用于跨进程持久化记录的 owner 等值匹配。

    Attributes:
        pid: 操作系统进程 ID。
        process_start_time: 进程创建时间（epoch 秒）；若采集失败为 `None`。
        boot_id: 系统 boot 标识（Linux 取 `/proc/sys/kernel/random/boot_id`；
            macOS / Windows 取 `psutil.boot_time()` 序列化字符串）；若采集失败为 `None`。
    """

    pid: int
    process_start_time: float | None
    boot_id: str | None

    def matches(self, other: "OwnerIdentity") -> bool:
        """判断两个 ``OwnerIdentity`` 是否指向同一个 owner。

        判定规则（与 :func:`is_owner_identity_alive` 共用退化策略）：
          * PID 必须严格相等；不等直接返回 ``False``。
          * ``process_start_time`` / ``boot_id`` 若双方均非 ``None`` 则参与
            等值校验；任一方为 ``None`` 时视为「未采集到」，不参与比对，
            自动退化为剩余非 NULL 字段比对，最差等价于仅 PID 比对。

        ``process_start_time`` 用 ``!=`` 直接比较 ``float`` 是安全的：
        SQLite ``REAL`` 类型是 IEEE 754 binary64，与 Python ``float``
        位等价，写入再读出 round-trip 不引入精度漂移；``psutil``
        在同一平台同一 PID 下也保证读到同一个 ``create_time()`` 值。

        Args:
            other: 待比对的另一个 owner 身份。

        Returns:
            ``True`` 表示两者指向同一个 owner；``False`` 表示明确不匹配。
        """

        if self.pid != other.pid:
            return False
        if (
            self.process_start_time is not None
            and other.process_start_time is not None
            and self.process_start_time != other.process_start_time
        ):
            return False
        if (
            self.boot_id is not None
            and other.boot_id is not None
            and self.boot_id != other.boot_id
        ):
            return False
        return True


def _get_process_start_time(pid: int) -> float | None:
    """跨平台采集指定 PID 的进程创建时间（epoch 秒）。

    使用 `psutil.Process(pid).create_time()`，封装所有平台差异。
    任意 `psutil` 异常或解析错误一律返回 `None`，由上层等值校验自动退化。

    Args:
        pid: 目标进程 ID。

    Returns:
        进程创建时间（epoch 秒）；采集失败返回 `None`。
    """

    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # pragma: no cover - 平台兜底，psutil 可能抛多种异常
        return None


def _get_boot_id_linux() -> str | None:
    """Linux 平台读 `/proc/sys/kernel/random/boot_id`。

    Returns:
        boot_id 字符串；读取失败返回 `None`。
    """

    try:
        with open("/proc/sys/kernel/random/boot_id", "r", encoding="ascii") as fp:
            value = fp.read().strip()
        return value or None
    except OSError:
        return None


def _get_boot_id_via_boot_time() -> str | None:
    """macOS / Windows 平台基于 `psutil.boot_time()` 推导 boot 标识。

    boot_time 返回系统启动时间（epoch 秒），将其序列化为字符串作为 boot_id。
    精度为秒级；进程创建必然晚于 boot，且重启会改变 boot_time，足以区分重启窗口。

    Returns:
        boot 时间序列化字符串；采集失败返回 `None`。
    """

    try:
        import psutil

        return f"{psutil.boot_time():.0f}"
    except Exception:  # pragma: no cover - 平台兜底
        return None


def _get_current_boot_id() -> str | None:
    """按平台分派，采集当前系统的 boot 标识。

    Returns:
        boot_id 字符串；采集失败返回 `None`。
    """

    if sys.platform.startswith("linux"):
        return _get_boot_id_linux()
    return _get_boot_id_via_boot_time()


def current_owner_identity() -> OwnerIdentity:
    """采集当前进程的 `OwnerIdentity`。

    任一平台采集失败的字段以 `None` 返回；不会抛错向上层污染。

    Returns:
        当前进程身份。
    """

    pid = os.getpid()
    return OwnerIdentity(
        pid=pid,
        process_start_time=_get_process_start_time(pid),
        boot_id=_get_current_boot_id(),
    )


def is_owner_identity_alive(identity: OwnerIdentity) -> bool:
    """判断给定 `OwnerIdentity` 对应的 owner 进程是否仍是「同一个」活进程。

    判定规则：
      1. PID 必须仍然活着（`is_pid_alive`）；否则直接 `False`。
      2. 重新采集当前 PID 的身份，与给定 identity 调用 :meth:`OwnerIdentity.matches`
         做按字段等值；任一字段为 `None`（无论是给定值还是当前采集值）则该字段
         不参与比对，最差等价于仅 PID 判活。

    Args:
        identity: 持久化记录里保存的 owner 身份。

    Returns:
        `True` 表示该 identity 仍指向同一个活进程；`False` 表示进程已退出
        或 PID 已被复用为其它进程。
    """

    if not is_pid_alive(identity.pid):
        return False

    current_at_pid = OwnerIdentity(
        pid=identity.pid,
        process_start_time=_get_process_start_time(identity.pid),
        boot_id=_get_current_boot_id(),
    )
    return identity.matches(current_at_pid)


__all__ = [
    "OwnerIdentity",
    "current_owner_identity",
    "is_owner_identity_alive",
    "is_pid_alive",
]
