"""``register_process_shutdown_hook`` 单元测试。"""

from __future__ import annotations

import signal
from typing import Callable

import pytest

from dayu.process_lifecycle import EXIT_CODE_SIGINT, ProcessShutdownCoordinator
from dayu.process_lifecycle.sync_signals import (
    _reset_registration_for_testing,
    register_process_shutdown_hook,
)


class _FakeHost:
    """实现 ``cancel_run_and_settle`` 的伪 Host。"""

    def __init__(self) -> None:
        self.settled_runs: list[str] = []
        self.owner_active_run_ids: list[str] = []

    def cancel_run_and_settle(self, run_id: str) -> object:
        self.settled_runs.append(run_id)
        return None

    def list_active_run_ids_for_current_owner(self) -> list[str]:
        return list(self.owner_active_run_ids)


def _patch_signal_and_atexit(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[int, object],
    reset_calls: list[int] | None = None,
) -> None:
    """统一 patch ``signal.signal`` / ``atexit.register``。

    Args:
        monkeypatch: pytest fixture。
        captured: 收集 (signum, handler)。
        reset_calls: 若提供，记录被还原 SIG_DFL 的 signum。
    """

    def _fake_signal(signum, handler):
        if not callable(handler) and reset_calls is not None:
            reset_calls.append(int(signum))
        captured[int(signum)] = handler
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "signal", _fake_signal)
    monkeypatch.setattr("atexit.register", lambda *_a, **_kw: None)


@pytest.mark.unit
def test_interactive_sigint_settles_and_raises_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """interactive=True 时 SIGINT 触发 settle 并抛 KeyboardInterrupt，不还原 SIG_DFL。"""

    _reset_registration_for_testing()
    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")

    captured: dict[int, object] = {}
    reset_calls: list[int] = []
    _patch_signal_and_atexit(monkeypatch, captured, reset_calls)

    register_process_shutdown_hook(coordinator, interactive=True)

    sigint_handler = captured.get(int(signal.SIGINT))
    assert callable(sigint_handler)

    reset_calls.clear()
    with pytest.raises(KeyboardInterrupt):
        sigint_handler(int(signal.SIGINT), None)

    assert host.settled_runs == ["run-1"]
    assert int(signal.SIGINT) not in reset_calls

    _reset_registration_for_testing()


@pytest.mark.unit
def test_interactive_multiple_sigint_each_settle_active_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """interactive 多次 SIGINT 都命中自定义 handler 并 settle 当时活跃 run。"""

    _reset_registration_for_testing()
    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)

    captured: dict[int, object] = {}
    _patch_signal_and_atexit(monkeypatch, captured)

    register_process_shutdown_hook(coordinator, interactive=True)
    sigint_handler = captured.get(int(signal.SIGINT))
    assert callable(sigint_handler)

    coordinator.register_active_run("run-1")
    with pytest.raises(KeyboardInterrupt):
        sigint_handler(int(signal.SIGINT), None)
    assert host.settled_runs == ["run-1"]

    coordinator.clear_active_run("run-1")
    coordinator.register_active_run("run-2")
    with pytest.raises(KeyboardInterrupt):
        sigint_handler(int(signal.SIGINT), None)
    assert host.settled_runs == ["run-1", "run-2"]

    _reset_registration_for_testing()


@pytest.mark.unit
def test_non_interactive_sigint_settles_and_raises_systemexit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """interactive=False 时 SIGINT 触发 settle 并抛 SystemExit(EXIT_CODE_SIGINT)。"""

    _reset_registration_for_testing()
    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")

    captured: dict[int, object] = {}
    _patch_signal_and_atexit(monkeypatch, captured)

    register_process_shutdown_hook(coordinator, interactive=False)

    sigint_handler = captured.get(int(signal.SIGINT))
    assert callable(sigint_handler)

    with pytest.raises(SystemExit) as excinfo:
        sigint_handler(int(signal.SIGINT), None)
    assert excinfo.value.code == EXIT_CODE_SIGINT
    assert host.settled_runs == ["run-1"]

    _reset_registration_for_testing()


@pytest.mark.unit
def test_sigterm_settles_resets_handler_and_raises_systemexit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM 一律 settle + 还原 SIG_DFL + SystemExit。"""

    _reset_registration_for_testing()
    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")

    captured: dict[int, object] = {}
    reset_calls: list[int] = []
    _patch_signal_and_atexit(monkeypatch, captured, reset_calls)

    register_process_shutdown_hook(coordinator, interactive=True)

    sigterm_handler = captured.get(int(signal.SIGTERM))
    assert callable(sigterm_handler)

    reset_calls.clear()
    with pytest.raises(SystemExit) as excinfo:
        sigterm_handler(int(signal.SIGTERM), None)
    assert excinfo.value.code == 0
    assert host.settled_runs == ["run-1"]
    assert int(signal.SIGTERM) in reset_calls

    _reset_registration_for_testing()


@pytest.mark.unit
def test_register_process_shutdown_hook_only_registers_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """重复调用 register_process_shutdown_hook 只注册一次。"""

    _reset_registration_for_testing()
    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)

    register_calls: list[int] = []

    def _fake_signal(signum, _handler):
        register_calls.append(int(signum))
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "signal", _fake_signal)
    monkeypatch.setattr("atexit.register", lambda *_a, **_kw: None)

    register_process_shutdown_hook(coordinator, interactive=False)
    first_count = len(register_calls)
    register_process_shutdown_hook(coordinator, interactive=False)

    assert first_count > 0
    assert len(register_calls) == first_count

    _reset_registration_for_testing()


@pytest.mark.unit
def test_atexit_hook_invokes_settle_active_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """atexit 钩子触发 ``settle_active_runs(trigger="atexit")``。"""

    _reset_registration_for_testing()
    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")

    captured_atexit: list[Callable[[], None]] = []

    def _capture_atexit(func: Callable[[], None], *_a: object, **_kw: object) -> Callable[[], None]:
        captured_atexit.append(func)
        return func

    monkeypatch.setattr(signal, "signal", lambda *_a, **_kw: signal.SIG_DFL)
    monkeypatch.setattr("atexit.register", _capture_atexit)

    register_process_shutdown_hook(coordinator, interactive=False)

    assert captured_atexit, "应当注册 atexit 钩子"
    captured_atexit[0]()
    assert host.settled_runs == ["run-1"]

    _reset_registration_for_testing()
