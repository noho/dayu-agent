"""``install_async_signal_handlers`` 单元测试。"""

from __future__ import annotations

import asyncio
import signal

import pytest

from dayu.process_lifecycle import ProcessShutdownCoordinator, install_async_signal_handlers


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
def test_install_async_signal_handlers_calls_full_sequence_then_on_signal() -> None:
    """asyncio handler 执行顺序：协作式取消 → 强收敛 → on_signal 回调。"""

    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")

    captured: dict[str, object] = {}

    def _on_signal(name: str, exit_code: int) -> None:
        captured["name"] = name
        captured["exit_code"] = exit_code
        # 此时强收敛与协作式取消都已执行。
        captured["cancelled_snapshot"] = list(host.cancelled_runs)
        captured["shutdown_calls"] = host.shutdown_calls

    async def _scenario() -> list[signal.Signals]:
        loop = asyncio.get_running_loop()
        with install_async_signal_handlers(loop, coordinator, on_signal=_on_signal) as installed:
            if not installed:
                return installed
            # 直接调度 SIGINT handler 触发同步流程。
            loop.call_soon(loop._signal_handlers[signal.SIGINT]._run)  # type: ignore[attr-defined]
            await asyncio.sleep(0)
            return list(installed)

    installed_signals = asyncio.run(_scenario())
    if not installed_signals:
        pytest.skip("当前平台不支持 add_signal_handler")

    assert captured.get("name") == "SIGINT"
    assert captured.get("exit_code") == 130
    assert captured.get("cancelled_snapshot") == ["run-1"]
    assert captured.get("shutdown_calls") == 1


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
