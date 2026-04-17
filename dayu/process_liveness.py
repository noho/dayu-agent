"""跨平台进程判活辅助。

`os.kill(pid, 0)` 仅在 POSIX 上是一个「发送 0 号信号」的存在性探测；
在 Windows 上 CPython 将 signal=0 路由到 `TerminateProcess`，无法用于判活，
对不存在的 PID 也不会稳定抛出 `ProcessLookupError`，可能阻塞或误判。

本模块把两类平台都收敛到 `is_pid_alive` 一个入口：

- POSIX：沿用 `os.kill(pid, 0)` 的 `ProcessLookupError` / `PermissionError` 语义。
- Windows：通过 `ctypes` 调用 `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, ...)`
  + `GetExitCodeProcess`，根据 `STILL_ACTIVE` 判定存活；`ACCESS_DENIED` 视为存在。
"""

from __future__ import annotations

import os
import sys

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


__all__ = ["is_pid_alive"]
