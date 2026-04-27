"""``ProcessShutdownCoordinator`` 单元测试。"""

from __future__ import annotations

import pytest

from dayu.process_lifecycle import ProcessShutdownCoordinator


class _FakeHost:
    """同时实现 cancel_run 与 shutdown_active_runs_for_owner 的伪 Host。"""

    def __init__(self) -> None:
        self.cancelled_runs: list[str] = []
        self.shutdown_calls: int = 0
        self.shutdown_returns: list[str] = []
        self.cancel_raises: dict[str, Exception] = {}
        self.shutdown_raises: Exception | None = None

    def cancel_run(self, run_id: str) -> bool:
        if run_id in self.cancel_raises:
            raise self.cancel_raises[run_id]
        self.cancelled_runs.append(run_id)
        return True

    def shutdown_active_runs_for_owner(self) -> list[str]:
        self.shutdown_calls += 1
        if self.shutdown_raises is not None:
            raise self.shutdown_raises
        return list(self.shutdown_returns)


@pytest.mark.unit
def test_register_and_clear_active_run_is_idempotent() -> None:
    """重复登记同一 run 视为一次，clear 不存在的 run 静默忽略。"""

    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")
    coordinator.register_active_run("run-1")
    coordinator.register_active_run("")

    assert coordinator.snapshot_active_runs() == ["run-1"]

    coordinator.clear_active_run("does-not-exist")
    coordinator.clear_active_run("run-1")
    assert coordinator.snapshot_active_runs() == []


@pytest.mark.unit
def test_cancel_active_runs_invokes_host_for_each_registered_run() -> None:
    """cancel_active_runs 会调用 host.cancel_run；异常仅记录日志。"""

    host = _FakeHost()
    host.cancel_raises["run-bad"] = RuntimeError("boom")
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")
    coordinator.register_active_run("run-bad")
    coordinator.register_active_run("run-2")

    cancelled = coordinator.cancel_active_runs(trigger="test")

    assert cancelled == ["run-1", "run-2"]
    assert host.cancelled_runs == ["run-1", "run-2"]


@pytest.mark.unit
def test_shutdown_owner_runs_is_idempotent() -> None:
    """重复调用 shutdown_owner_runs 只会真正生效一次。"""

    host = _FakeHost()
    host.shutdown_returns = ["run-1"]
    coordinator = ProcessShutdownCoordinator(host=host)

    first = coordinator.shutdown_owner_runs(trigger="first")
    second = coordinator.shutdown_owner_runs(trigger="second")

    assert first == ["run-1"]
    assert second == []
    assert host.shutdown_calls == 1


@pytest.mark.unit
def test_run_full_shutdown_sequence_runs_cancel_then_shutdown() -> None:
    """完整退出流程先协作式取消再强收敛。"""

    host = _FakeHost()
    host.shutdown_returns = ["run-2"]
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")

    cancelled, owner_cancelled = coordinator.run_full_shutdown_sequence(trigger="signal:SIGINT")

    assert cancelled == ["run-1"]
    assert owner_cancelled == ["run-2"]
    assert host.cancelled_runs == ["run-1"]
    assert host.shutdown_calls == 1


@pytest.mark.unit
def test_shutdown_owner_runs_swallows_host_exception() -> None:
    """host.shutdown_active_runs_for_owner 异常不向外抛。"""

    host = _FakeHost()
    host.shutdown_raises = RuntimeError("boom")
    coordinator = ProcessShutdownCoordinator(host=host)

    assert coordinator.shutdown_owner_runs(trigger="x") == []
    # 即使首次失败也仍然算作已触发，避免循环重试。
    assert coordinator.shutdown_owner_runs(trigger="y") == []
