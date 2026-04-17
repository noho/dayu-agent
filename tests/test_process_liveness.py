"""测试跨平台进程判活 `dayu.process_liveness.is_pid_alive`。

设计原则：
- 唯一的「真实进程」用例是当前进程判活，因为 `os.getpid()` 必然对应当前运行的 Python 进程，
  任何 OS/CI 环境都能给出确定结果。
- 其余分支（PID 不存在、权限被拒、Win32 错误码等）全部通过 monkeypatch 直接桩住
  `os.kill` 或 Win32 `OpenProcess` 返回值，不依赖任何「某个 PID 当前空闲」或
  「子进程 PID 不会被立刻回收」这类与 OS 调度相关的不确定性。
"""

from __future__ import annotations

import os
import sys

import pytest

from dayu import process_liveness as liveness_module
from dayu.process_liveness import is_pid_alive

_POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason="POSIX 分支专用")
_WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="Windows 分支专用")


@pytest.mark.unit
def test_is_pid_alive_returns_true_for_current_process() -> None:
    """当前进程的 PID 一定被判为存活。"""

    assert is_pid_alive(os.getpid()) is True


@pytest.mark.unit
@_POSIX_ONLY
def test_posix_branch_returns_false_on_process_lookup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX 下 `os.kill` 抛 `ProcessLookupError` 时判定为已退出。"""

    def _raise_lookup(pid: int, sig: int) -> None:
        del pid, sig
        raise ProcessLookupError

    monkeypatch.setattr(liveness_module.os, "kill", _raise_lookup)
    assert liveness_module._is_pid_alive_posix(12345) is False


@pytest.mark.unit
@_POSIX_ONLY
def test_posix_branch_returns_true_on_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX 下 `os.kill` 抛 `PermissionError` 表示目标存在、仅无权限发信号。"""

    def _raise_permission(pid: int, sig: int) -> None:
        del pid, sig
        raise PermissionError

    monkeypatch.setattr(liveness_module.os, "kill", _raise_permission)
    assert liveness_module._is_pid_alive_posix(12345) is True


@pytest.mark.unit
@_POSIX_ONLY
def test_posix_branch_returns_true_when_kill_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """POSIX 下 `os.kill(pid, 0)` 正常返回即表示目标进程存活。"""

    def _noop(pid: int, sig: int) -> None:
        del pid, sig

    monkeypatch.setattr(liveness_module.os, "kill", _noop)
    assert liveness_module._is_pid_alive_posix(12345) is True


@pytest.mark.unit
@_WINDOWS_ONLY
def test_windows_branch_returns_false_for_invalid_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows 下 `OpenProcess` 失败且 last error 为 INVALID_PARAMETER 时判定为不存在。"""

    import ctypes

    set_last_error = getattr(ctypes, "set_last_error")
    kernel32 = getattr(liveness_module, "_kernel32")
    is_alive_windows = getattr(liveness_module, "_is_pid_alive_windows")

    def _fake_open_process(access: int, inherit: bool, pid: int) -> int:
        del access, inherit, pid
        set_last_error(liveness_module._ERROR_INVALID_PARAMETER)
        return 0

    monkeypatch.setattr(kernel32, "OpenProcess", _fake_open_process)
    assert is_alive_windows(999999) is False


@pytest.mark.unit
@_WINDOWS_ONLY
def test_windows_branch_returns_true_for_access_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows 下 `OpenProcess` 失败且 last error 为 ACCESS_DENIED 时表示目标存在。"""

    import ctypes

    set_last_error = getattr(ctypes, "set_last_error")
    kernel32 = getattr(liveness_module, "_kernel32")
    is_alive_windows = getattr(liveness_module, "_is_pid_alive_windows")

    def _fake_open_process(access: int, inherit: bool, pid: int) -> int:
        del access, inherit, pid
        set_last_error(liveness_module._ERROR_ACCESS_DENIED)
        return 0

    monkeypatch.setattr(kernel32, "OpenProcess", _fake_open_process)
    assert is_alive_windows(12345) is True


@pytest.mark.unit
@_WINDOWS_ONLY
def test_windows_branch_raises_on_unexpected_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows 下 `OpenProcess` 出现未预期的 winerror 时必须外抛 OSError。"""

    import ctypes

    set_last_error = getattr(ctypes, "set_last_error")
    kernel32 = getattr(liveness_module, "_kernel32")
    is_alive_windows = getattr(liveness_module, "_is_pid_alive_windows")

    _UNEXPECTED_WINERROR = 1234

    def _fake_open_process(access: int, inherit: bool, pid: int) -> int:
        del access, inherit, pid
        set_last_error(_UNEXPECTED_WINERROR)
        return 0

    monkeypatch.setattr(kernel32, "OpenProcess", _fake_open_process)
    with pytest.raises(OSError):
        is_alive_windows(12345)
