"""CLI 优雅退出钩子测试。"""

from __future__ import annotations

import signal
from types import SimpleNamespace
from typing import cast

import pytest

from dayu.cli import graceful_shutdown
from dayu.cli.graceful_shutdown import (
    _ShutdownCoordinator,
    install_cli_signal_handlers,
    register_cli_shutdown_hook,
)


class _FakeHost:
    """收敛用 Host 桩。"""

    def __init__(self, *, raise_on_shutdown: bool = False) -> None:
        """初始化桩。

        Args:
            raise_on_shutdown: 是否在 shutdown 时抛异常。

        Returns:
            无。

        Raises:
            无。
        """

        self.shutdown_calls = 0
        self._raise = raise_on_shutdown

    def shutdown_active_runs_for_owner(self) -> list[str]:
        """模拟 Host.shutdown_active_runs_for_owner。"""

        self.shutdown_calls += 1
        if self._raise:
            raise RuntimeError("boom")
        return ["run_001", "run_002"]


@pytest.mark.unit
def test_coordinator_runs_once_even_if_triggered_multiple_times() -> None:
    """_ShutdownCoordinator 在多次触发下只收敛一次。"""

    host = _FakeHost()
    coordinator = _ShutdownCoordinator(host)

    first = coordinator.run_once(trigger="test-a")
    second = coordinator.run_once(trigger="test-b")

    assert first == ["run_001", "run_002"]
    assert second == []
    assert host.shutdown_calls == 1


@pytest.mark.unit
def test_coordinator_swallows_shutdown_exception() -> None:
    """Host 抛异常时 coordinator 返回空列表且不外泄。"""

    host = _FakeHost(raise_on_shutdown=True)
    coordinator = _ShutdownCoordinator(host)

    assert coordinator.run_once(trigger="err") == []
    assert host.shutdown_calls == 1


@pytest.mark.unit
def test_install_cli_signal_handlers_runs_on_context_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    """上下文正常退出时会触发一次 shutdown。"""

    host = _FakeHost()
    installed: dict[int, object] = {}

    def _fake_signal(signum: int, handler: object) -> object:
        previous = installed.get(signum, signal.SIG_DFL)
        installed[signum] = handler
        return previous

    monkeypatch.setattr(signal, "signal", _fake_signal)
    monkeypatch.setattr(signal, "getsignal", lambda s: installed.get(s, signal.SIG_DFL))

    with install_cli_signal_handlers(host):
        pass

    assert host.shutdown_calls == 1


@pytest.mark.unit
def test_install_cli_signal_handlers_is_robust_to_signal_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """signal.signal 抛错时上下文不崩溃，atexit 仍会在退出时收敛。"""

    host = _FakeHost()

    def _broken_signal(_signum: int, _handler: object) -> object:
        raise OSError("not main thread")

    monkeypatch.setattr(signal, "signal", _broken_signal)
    monkeypatch.setattr(signal, "getsignal", lambda s: signal.SIG_DFL)

    with install_cli_signal_handlers(host):
        pass

    # 所有 signal 安装失败，但 context-exit 路径仍应调用 shutdown
    assert host.shutdown_calls == 1


@pytest.mark.unit
def test_register_cli_shutdown_hook_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    """register_cli_shutdown_hook 重复调用只生效一次。"""

    graceful_shutdown._reset_registration_for_testing()

    atexit_calls: list[object] = []
    monkeypatch.setattr(graceful_shutdown.atexit, "register", lambda func, **kwargs: atexit_calls.append((func, kwargs)))
    monkeypatch.setattr(signal, "signal", lambda _s, _h: signal.SIG_DFL)

    host = _FakeHost()
    register_cli_shutdown_hook(cast(object, host))  # type: ignore[arg-type]
    register_cli_shutdown_hook(cast(object, host))  # type: ignore[arg-type]
    register_cli_shutdown_hook(cast(object, host))  # type: ignore[arg-type]

    # 只应产生一次 atexit.register 调用
    assert len(atexit_calls) == 1
    graceful_shutdown._reset_registration_for_testing()


@pytest.mark.unit
def test_install_cli_signal_handlers_handler_triggers_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """手工调用注册的 handler 应先收敛再抛 KeyboardInterrupt。"""

    host = _FakeHost()
    captured: dict[int, object] = {}

    def _fake_signal(signum: int, handler: object) -> object:
        captured[signum] = handler
        return signal.SIG_DFL

    monkeypatch.setattr(signal, "signal", _fake_signal)
    monkeypatch.setattr(signal, "getsignal", lambda s: signal.SIG_DFL)

    with install_cli_signal_handlers(host):
        # 取出注册的 SIGTERM handler 手工触发
        sigterm = getattr(signal, "SIGTERM")
        handler = captured.get(sigterm)
        assert handler is not None and callable(handler)
        with pytest.raises(KeyboardInterrupt):
            cast(object, handler).__call__(int(sigterm), SimpleNamespace())  # type: ignore[attr-defined]

    # signal handler + context-exit 应都幂等触发同一次
    assert host.shutdown_calls == 1
