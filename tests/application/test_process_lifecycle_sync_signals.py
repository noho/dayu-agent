"""``install_sync_signal_handlers`` 与 ``register_process_shutdown_hook`` 单元测试。"""

from __future__ import annotations

import signal

import pytest

from dayu.process_lifecycle import ProcessShutdownCoordinator
from dayu.process_lifecycle.sync_signals import (
    _reset_registration_for_testing,
    install_sync_signal_handlers,
    register_process_shutdown_hook,
)


class _FakeHost:
    def __init__(self) -> None:
        self.cancelled_runs: list[str] = []
        self.shutdown_calls: int = 0

    def cancel_run(self, run_id: str) -> bool:
        self.cancelled_runs.append(run_id)
        return True

    def shutdown_active_runs_for_owner(self) -> list[str]:
        self.shutdown_calls += 1
        return []


@pytest.mark.unit
def test_install_sync_signal_handlers_runs_cancel_then_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """信号 handler 顺序：协作式取消 → 强收敛 → KeyboardInterrupt。"""

    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")

    captured: dict[str, object] = {"installed": []}

    original_signal = signal.signal

    def _fake_signal(signum, handler):
        captured["installed"].append((signum, handler))  # type: ignore[union-attr]
        return original_signal(signum, signal.SIG_DFL) if not callable(handler) else None

    monkeypatch.setattr(signal, "signal", _fake_signal)
    monkeypatch.setattr(signal, "getsignal", lambda _signum: signal.SIG_DFL)

    with install_sync_signal_handlers(coordinator):
        # 找到注册的 SIGINT handler 并触发
        sigint_entries = [item for item in captured["installed"] if item[0] == signal.SIGINT]  # type: ignore[index]
        assert sigint_entries, "应当注册了 SIGINT handler"
        handler = sigint_entries[0][1]
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)

    assert host.cancelled_runs == ["run-1"]
    assert host.shutdown_calls >= 1


@pytest.mark.unit
def test_register_process_shutdown_hook_only_registers_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """register_process_shutdown_hook 重复调用只生效一次。"""

    _reset_registration_for_testing()
    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)

    register_calls: list[tuple[int, object]] = []

    def _fake_signal(signum, handler):
        register_calls.append((signum, handler))
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "signal", _fake_signal)
    monkeypatch.setattr("atexit.register", lambda *_args, **_kwargs: None)

    register_process_shutdown_hook(coordinator)
    first_count = len(register_calls)
    register_process_shutdown_hook(coordinator)

    assert first_count > 0
    assert len(register_calls) == first_count

    _reset_registration_for_testing()


@pytest.mark.unit
def test_install_sync_signal_handlers_sigterm_raises_systemexit_not_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGTERM 必须 ``SystemExit`` 退出进程，不能落到 ``KeyboardInterrupt``。

    如果回退成 ``KeyboardInterrupt``，``dayu interactive`` 收到 SIGTERM
    会被 REPL 的 ``except KeyboardInterrupt`` 吞掉继续等待输入，
    ``dayu prompt`` 也会错误地返回 ``EXIT_CODE_SIGINT``。
    """

    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")

    captured: dict[int, object] = {}
    original_signal = signal.signal

    def _fake_signal(signum, handler):
        captured[signum] = handler
        return original_signal(signum, signal.SIG_DFL) if not callable(handler) else None

    monkeypatch.setattr(signal, "signal", _fake_signal)
    monkeypatch.setattr(signal, "getsignal", lambda _signum: signal.SIG_DFL)

    with install_sync_signal_handlers(coordinator):
        sigterm_handler = captured.get(int(signal.SIGTERM))
        assert callable(sigterm_handler), "SIGTERM 必须注册 handler"
        with pytest.raises(SystemExit) as excinfo:
            sigterm_handler(int(signal.SIGTERM), None)
        assert excinfo.value.code == 0  # EXIT_CODE_SIGTERM

    assert host.cancelled_runs == ["run-1"]
    assert host.shutdown_calls >= 1


@pytest.mark.unit
def test_install_sync_signal_handlers_sigint_still_raises_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGINT 仍然必须维持 ``KeyboardInterrupt`` 语义供 REPL 使用。"""

    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)

    captured: dict[int, object] = {}
    original_signal = signal.signal

    def _fake_signal(signum, handler):
        captured[signum] = handler
        return original_signal(signum, signal.SIG_DFL) if not callable(handler) else None

    monkeypatch.setattr(signal, "signal", _fake_signal)
    monkeypatch.setattr(signal, "getsignal", lambda _signum: signal.SIG_DFL)

    with install_sync_signal_handlers(coordinator):
        sigint_handler = captured.get(int(signal.SIGINT))
        assert callable(sigint_handler)
        with pytest.raises(KeyboardInterrupt):
            sigint_handler(int(signal.SIGINT), None)
