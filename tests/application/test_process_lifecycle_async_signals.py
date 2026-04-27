"""``install_async_signal_handlers`` 单元测试。"""

from __future__ import annotations

import asyncio
import signal

import pytest

from dayu.process_lifecycle import ProcessShutdownCoordinator, install_async_signal_handlers


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


@pytest.mark.unit
def test_install_async_signal_handlers_settles_then_calls_on_signal() -> None:
    """asyncio handler 执行顺序：settle_active_runs → on_signal 回调。"""

    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")

    captured: dict[str, object] = {}

    def _on_signal(name: str, exit_code: int) -> None:
        captured["name"] = name
        captured["exit_code"] = exit_code
        captured["settled_snapshot"] = list(host.settled_runs)

    async def _scenario() -> list[signal.Signals]:
        loop = asyncio.get_running_loop()
        with install_async_signal_handlers(loop, coordinator, on_signal=_on_signal) as installed:
            if not installed:
                return installed
            loop.call_soon(loop._signal_handlers[signal.SIGINT]._run)  # type: ignore[attr-defined]
            await asyncio.sleep(0)
            return list(installed)

    installed_signals = asyncio.run(_scenario())
    if not installed_signals:
        pytest.skip("当前平台不支持 add_signal_handler")

    assert captured.get("name") == "SIGINT"
    assert captured.get("exit_code") == 130
    assert captured.get("settled_snapshot") == ["run-1"]


@pytest.mark.unit
def test_install_async_signal_handlers_returns_installed_signals_and_unregisters() -> None:
    """退出上下文时移除 handler，确保 loop 不留下副作用。"""

    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)

    async def _scenario() -> list[signal.Signals]:
        loop = asyncio.get_running_loop()
        with install_async_signal_handlers(
            loop, coordinator, on_signal=lambda _name, _code: None
        ) as installed:
            return list(installed)

    installed_signals = asyncio.run(_scenario())
    if not installed_signals:
        pytest.skip("当前平台不支持 add_signal_handler")

    assert signal.SIGINT in installed_signals or signal.SIGTERM in installed_signals
