"""测试跨平台进程判活 `dayu.process_liveness.is_pid_alive`。

设计原则：
- 唯一的「真实进程」用例是当前进程判活，因为 `os.getpid()` 必然对应当前运行的 Python 进程，
  任何 OS/CI 环境都能给出确定结果。
- 其余分支（PID 不存在、权限被拒、Win32 错误码等）全部通过 monkeypatch 直接桩住
  `os.kill` 或 Win32 `OpenProcess` 返回值，不依赖任何「某个 PID 当前空闲」或
  「子进程 PID 不会被立刻回收」这类与 OS 调度相关的不确定性。
- Windows 分支通过 `importlib.reload` 在 mock 了 `sys.platform` 和 `ctypes` 之后重新
  导入模块来覆盖，使得 `if sys.platform == "win32":` 块及其内部函数得以执行。
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol
from unittest.mock import MagicMock

import pytest

from dayu import process_liveness as liveness_module
from dayu.process_liveness import is_pid_alive


class _CtypesPointerLike(Protocol):
    """ctypes 指针对象的最小协议，支持 `.contents.value` 读写。"""

    @property
    def contents(self) -> _CtypesValueLike: ...


class _CtypesValueLike(Protocol):
    """ctypes 值对象的最小协议，支持 `.value` 读写。"""

    value: int

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


# ---------------------------------------------------------------------------
# macOS/Linux 上模拟 Windows 分支的测试
#
# 因为 `if sys.platform == "win32":` 块在非 Windows 平台上不会执行，
# 需要在 mock 了 sys.platform 与 ctypes 之后通过 importlib.reload
# 重新加载模块来覆盖 Windows 分支代码。
# ---------------------------------------------------------------------------


def _build_fake_kernel32(
    open_process_fn: Callable[[int, bool, int], int],
    get_exit_code_fn: Callable[[int, _CtypesPointerLike], int],
    close_handle_fn: Callable[[int], int],
) -> MagicMock:
    """构造一个伪造的 kernel32 WinDLL 对象。

    Args:
        open_process_fn: 伪造的 OpenProcess 实现。
        get_exit_code_fn: 伪造的 GetExitCodeProcess 实现。
        close_handle_fn: 伪造的 CloseHandle 实现。

    Returns:
        配置好 argtypes/restype 的 MagicMock 实例。
    """

    fake = MagicMock()
    fake.OpenProcess = open_process_fn
    fake.GetExitCodeProcess = get_exit_code_fn
    fake.CloseHandle = close_handle_fn
    return fake


class _PtrProxy:
    """代理对象，使 `.contents.value = x` 实际写入原始 ctypes 对象。

    用于在 macOS 上替代 `ctypes.byref(wintypes.DWORD())` 的返回值，
    使得 fake GetExitCodeProcess 可以通过 `ptr.contents.value = ...` 写入退出码，
    而真实代码随后读取 `exit_code.value` 能拿到相同的值。
    """

    def __init__(self, target: _CtypesValueLike) -> None:
        self._target = target

    @property
    def contents(self) -> _CtypesValueLike:
        return self._target


def _fake_byref(obj: _CtypesValueLike) -> _PtrProxy:
    """伪造的 ctypes.byref，返回代理原始对象的指针容器。

    Args:
        obj: 原始 ctypes 对象（如 wintypes.DWORD()）。

    Returns:
        代理对象，`.contents` 返回原始对象。
    """

    return _PtrProxy(obj)


@contextmanager
def _reloaded_as_windows(
    monkeypatch: pytest.MonkeyPatch,
    fake_kernel32: MagicMock,
) -> Iterator[types.ModuleType]:
    """临时将 process_liveness 模块重载为 Windows 版本。

    通过 mock sys.platform 和 ctypes.WinDLL，然后 importlib.reload 使
    `if sys.platform == "win32":` 块执行，从而覆盖 Windows 分支代码。

    同时处理 macOS 上缺失的 ctypes.WinDLL、ctypes.set_last_error 以及
    ctypes.byref 返回值类型差异。

    上下文退出时恢复原始模块状态。

    Args:
        monkeypatch: pytest 的 monkeypatch fixture。
        fake_kernel32: 伪造的 kernel32 WinDLL 对象。

    Yields:
        重载后的 Windows 版 process_liveness 模块。
    """

    # 备份当前模块。
    original_module = sys.modules["dayu.process_liveness"]

    import ctypes  # pyright: ignore[reportRedeclaration]

    # --- 注入 macOS 上缺失的 ctypes 属性 ---
    _added_attrs: list[str] = []
    _saved_attrs: dict[str, object] = {}

    def _ensure_attr(name: str, value: object) -> None:
        """如果 ctypes 没有指定属性则注入，否则备份原值。"""

        if hasattr(ctypes, name):
            _saved_attrs[name] = getattr(ctypes, name)
        else:
            _added_attrs.append(name)
        setattr(ctypes, name, value)

    def _fake_win_dll(_name: str | None = None, **kwargs: bool) -> MagicMock:
        del _name, kwargs
        return fake_kernel32

    _ensure_attr("WinDLL", _fake_win_dll)

    # set_last_error / get_last_error：macOS 上 set_last_error 不存在，
    # 需要提供一个有状态的 mock 使得 fake OpenProcess 设置的错误码能被
    # 模块内的 ctypes.get_last_error() 读取到。
    _last_error_state = [0]

    def _mock_set_last_error(code: int) -> None:
        _last_error_state[0] = code

    def _mock_get_last_error() -> int:
        return _last_error_state[0]

    _ensure_attr("set_last_error", _mock_set_last_error)
    _ensure_attr("get_last_error", _mock_get_last_error)

    _ensure_attr("byref", _fake_byref)

    # mock sys.platform 为 win32。
    monkeypatch.setattr(sys, "platform", "win32")

    try:
        # 重新加载模块，此时 sys.platform == "win32"，if 块会执行。
        reloaded = importlib.reload(liveness_module)
        yield reloaded
    finally:
        # 恢复原始模块。
        monkeypatch.setattr(sys, "platform", "darwin")
        sys.modules["dayu.process_liveness"] = original_module
        importlib.reload(liveness_module)
        # 恢复 ctypes 属性。
        for name in _added_attrs:
            delattr(ctypes, name)
        for name, orig_value in _saved_attrs.items():
            setattr(ctypes, name, orig_value)


@pytest.mark.unit
def test_windows_returns_false_for_invalid_pid_via_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通过 reload 覆盖 Windows 分支：OpenProcess 返回 0 + INVALID_PARAMETER → False。"""

    import ctypes

    def _fake_open_process(access: int, inherit: bool, pid: int) -> int:
        del access, inherit, pid
        ctypes.set_last_error(liveness_module._ERROR_INVALID_PARAMETER)  # type: ignore[attr-defined]
        return 0

    fake_k32 = _build_fake_kernel32(
        open_process_fn=_fake_open_process,
        get_exit_code_fn=MagicMock(),
        close_handle_fn=MagicMock(),
    )

    with _reloaded_as_windows(monkeypatch, fake_k32) as win_mod:
        assert win_mod._is_pid_alive_windows(999999) is False


@pytest.mark.unit
def test_windows_returns_true_for_access_denied_via_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通过 reload 覆盖 Windows 分支：OpenProcess 返回 0 + ACCESS_DENIED → True。"""

    import ctypes

    def _fake_open_process(access: int, inherit: bool, pid: int) -> int:
        del access, inherit, pid
        ctypes.set_last_error(liveness_module._ERROR_ACCESS_DENIED)  # type: ignore[attr-defined]
        return 0

    fake_k32 = _build_fake_kernel32(
        open_process_fn=_fake_open_process,
        get_exit_code_fn=MagicMock(),
        close_handle_fn=MagicMock(),
    )

    with _reloaded_as_windows(monkeypatch, fake_k32) as win_mod:
        assert win_mod._is_pid_alive_windows(12345) is True


@pytest.mark.unit
def test_windows_raises_on_unexpected_error_via_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通过 reload 覆盖 Windows 分支：OpenProcess 返回 0 + 未预期 winerror → OSError。"""

    import ctypes

    _UNEXPECTED_WINERROR = 1234

    def _fake_open_process(access: int, inherit: bool, pid: int) -> int:
        del access, inherit, pid
        ctypes.set_last_error(_UNEXPECTED_WINERROR)  # type: ignore[attr-defined]
        return 0

    fake_k32 = _build_fake_kernel32(
        open_process_fn=_fake_open_process,
        get_exit_code_fn=MagicMock(),
        close_handle_fn=MagicMock(),
    )

    with _reloaded_as_windows(monkeypatch, fake_k32) as win_mod:
        with pytest.raises(OSError):
            win_mod._is_pid_alive_windows(12345)


@pytest.mark.unit
def test_windows_handle_valid_still_active_via_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通过 reload 覆盖 Windows 分支：handle 有效 + GetExitCodeProcess 返回 STILL_ACTIVE。

    覆盖 _is_pid_alive_windows 中 handle 非零、exit_code == STILL_ACTIVE 的完整路径。
    """

    import ctypes

    fake_handle = 12345
    close_calls: list[int] = []

    def _fake_open_process(access: int, inherit: bool, pid: int) -> int:
        del access, inherit, pid
        return fake_handle

    def _fake_get_exit_code(handle: int, exit_code_ptr: _CtypesPointerLike) -> int:
        assert handle == fake_handle
        exit_code_ptr.contents.value = liveness_module._STILL_ACTIVE
        return 1

    def _fake_close_handle(handle: int) -> int:
        close_calls.append(handle)
        return 1

    fake_k32 = _build_fake_kernel32(_fake_open_process, _fake_get_exit_code, _fake_close_handle)

    with _reloaded_as_windows(monkeypatch, fake_k32) as win_mod:
        assert win_mod._is_pid_alive_windows(42) is True
        assert close_calls == [fake_handle]


@pytest.mark.unit
def test_windows_handle_valid_process_exited_via_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通过 reload 覆盖 Windows 分支：handle 有效 + exit_code != STILL_ACTIVE。

    覆盖 _is_pid_alive_windows 中进程已退出返回 False 的分支。
    """

    import ctypes

    fake_handle = 67890
    close_calls: list[int] = []

    def _fake_open_process(access: int, inherit: bool, pid: int) -> int:
        del access, inherit, pid
        return fake_handle

    def _fake_get_exit_code(handle: int, exit_code_ptr: _CtypesPointerLike) -> int:
        assert handle == fake_handle
        exit_code_ptr.contents.value = 0
        return 1

    def _fake_close_handle(handle: int) -> int:
        close_calls.append(handle)
        return 1

    fake_k32 = _build_fake_kernel32(_fake_open_process, _fake_get_exit_code, _fake_close_handle)

    with _reloaded_as_windows(monkeypatch, fake_k32) as win_mod:
        assert win_mod._is_pid_alive_windows(42) is False
        assert close_calls == [fake_handle]


@pytest.mark.unit
def test_windows_get_exit_code_fails_via_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通过 reload 覆盖 Windows 分支：GetExitCodeProcess 返回 0 时抛 OSError。

    同时验证 CloseHandle 在 finally 块中被调用。
    """

    import ctypes

    fake_handle = 11111
    close_calls: list[int] = []

    def _fake_open_process(access: int, inherit: bool, pid: int) -> int:
        del access, inherit, pid
        return fake_handle

    def _fake_get_exit_code(handle: int, exit_code_ptr: _CtypesPointerLike) -> int:
        del handle, exit_code_ptr
        ctypes.set_last_error(997)  # type: ignore[attr-defined]
        return 0

    def _fake_close_handle(handle: int) -> int:
        close_calls.append(handle)
        return 1

    fake_k32 = _build_fake_kernel32(_fake_open_process, _fake_get_exit_code, _fake_close_handle)

    with _reloaded_as_windows(monkeypatch, fake_k32) as win_mod:
        with pytest.raises(OSError):
            win_mod._is_pid_alive_windows(42)
        # CloseHandle 应在 finally 中被调用。
        assert close_calls == [fake_handle]


@pytest.mark.unit
def test_is_pid_alive_dispatches_to_windows_branch_via_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """通过 reload 后 is_pid_alive 应调用 _is_pid_alive_windows。"""

    import ctypes

    fake_handle = 99999

    def _fake_open_process(access: int, inherit: bool, pid: int) -> int:
        del access, inherit, pid
        return fake_handle

    def _fake_get_exit_code(handle: int, exit_code_ptr: _CtypesPointerLike) -> int:
        del handle
        exit_code_ptr.contents.value = liveness_module._STILL_ACTIVE
        return 1

    def _fake_close_handle(handle: int) -> int:
        del handle
        return 1

    fake_k32 = _build_fake_kernel32(_fake_open_process, _fake_get_exit_code, _fake_close_handle)

    with _reloaded_as_windows(monkeypatch, fake_k32) as win_mod:
        assert win_mod.is_pid_alive(42) is True


# ---------------------------------------------------------------------------
# OwnerIdentity（#106）
# ---------------------------------------------------------------------------


from dayu.process_liveness import (
    OwnerIdentity,
    current_owner_identity,
    is_owner_identity_alive,
)


@pytest.mark.unit
def test_current_owner_identity_returns_current_pid() -> None:
    """current_owner_identity 必须返回当前进程的 PID。"""

    identity = current_owner_identity()
    assert identity.pid == os.getpid()


@pytest.mark.unit
def test_current_owner_identity_collects_at_least_one_field() -> None:
    """主流平台至少能采集到 process_start_time 或 boot_id 之一。"""

    identity = current_owner_identity()
    assert (identity.process_start_time is not None) or (identity.boot_id is not None)


@pytest.mark.unit
def test_is_owner_identity_alive_returns_true_for_current_process() -> None:
    """当前进程身份重新采集后应判活成功。"""

    identity = current_owner_identity()
    assert is_owner_identity_alive(identity) is True


@pytest.mark.unit
def test_is_owner_identity_alive_returns_false_when_pid_dead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID 已死时直接返回 False，不再做后续字段比对。"""

    monkeypatch.setattr(liveness_module, "is_pid_alive", lambda pid: False)
    identity = OwnerIdentity(pid=99999, process_start_time=1.0, boot_id="b")
    assert is_owner_identity_alive(identity) is False


@pytest.mark.unit
def test_is_owner_identity_alive_returns_false_when_start_time_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID 活但 process_start_time 不一致 → PID 复用 → False。"""

    monkeypatch.setattr(liveness_module, "is_pid_alive", lambda pid: True)
    monkeypatch.setattr(liveness_module, "_get_process_start_time", lambda pid: 222.0)
    monkeypatch.setattr(liveness_module, "_get_current_boot_id", lambda: None)
    identity = OwnerIdentity(pid=12345, process_start_time=111.0, boot_id=None)
    assert is_owner_identity_alive(identity) is False


@pytest.mark.unit
def test_is_owner_identity_alive_returns_false_when_boot_id_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PID 活但 boot_id 不一致 → 重启过 → False。"""

    monkeypatch.setattr(liveness_module, "is_pid_alive", lambda pid: True)
    monkeypatch.setattr(liveness_module, "_get_process_start_time", lambda pid: None)
    monkeypatch.setattr(liveness_module, "_get_current_boot_id", lambda: "boot-new")
    identity = OwnerIdentity(pid=12345, process_start_time=None, boot_id="boot-old")
    assert is_owner_identity_alive(identity) is False


@pytest.mark.unit
def test_is_owner_identity_alive_degrades_to_pid_when_all_fields_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """所有字段都 NULL 时退化为仅 PID 判活，行为等价 is_pid_alive。"""

    monkeypatch.setattr(liveness_module, "is_pid_alive", lambda pid: True)
    monkeypatch.setattr(liveness_module, "_get_process_start_time", lambda pid: None)
    monkeypatch.setattr(liveness_module, "_get_current_boot_id", lambda: None)
    identity = OwnerIdentity(pid=12345, process_start_time=None, boot_id=None)
    assert is_owner_identity_alive(identity) is True


@pytest.mark.unit
def test_is_owner_identity_alive_skips_field_when_either_side_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任一侧字段 NULL 则该字段不参与等值；剩余字段一致即视为同 owner。"""

    monkeypatch.setattr(liveness_module, "is_pid_alive", lambda pid: True)
    # stored has start_time but current采集失败 → start_time 不参与
    monkeypatch.setattr(liveness_module, "_get_process_start_time", lambda pid: None)
    monkeypatch.setattr(liveness_module, "_get_current_boot_id", lambda: "boot-x")
    identity = OwnerIdentity(pid=12345, process_start_time=111.0, boot_id="boot-x")
    assert is_owner_identity_alive(identity) is True
