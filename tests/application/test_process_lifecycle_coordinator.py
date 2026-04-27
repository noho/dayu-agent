"""``ProcessShutdownCoordinator`` 单元测试。"""

from __future__ import annotations

import pytest

from dayu.process_lifecycle import ProcessShutdownCoordinator


class _FakeHost:
    """实现 ``cancel_run_and_settle`` 的伪 Host。"""

    def __init__(self) -> None:
        self.settled_runs: list[str] = []
        self.settle_raises: dict[str, Exception] = {}
        self.owner_active_run_ids: list[str] = []

    def cancel_run_and_settle(self, run_id: str) -> object:
        if run_id in self.settle_raises:
            raise self.settle_raises[run_id]
        self.settled_runs.append(run_id)
        return None

    def list_active_run_ids_for_current_owner(self) -> list[str]:
        return list(self.owner_active_run_ids)


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
def test_settle_active_runs_invokes_settle_hook_for_each_registered_run() -> None:
    """settle_active_runs 对每个已登记 run 调 cancel_run_and_settle。"""

    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")
    coordinator.register_active_run("run-2")

    settled = coordinator.settle_active_runs(trigger="test")

    assert settled == ["run-1", "run-2"]
    assert host.settled_runs == ["run-1", "run-2"]


@pytest.mark.unit
def test_settle_active_runs_swallows_hook_exception() -> None:
    """单个 run 的 settle 异常不影响其他 run，也不向外抛。"""

    host = _FakeHost()
    host.settle_raises["run-bad"] = RuntimeError("boom")
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")
    coordinator.register_active_run("run-bad")
    coordinator.register_active_run("run-2")

    settled = coordinator.settle_active_runs(trigger="test")

    assert settled == ["run-1", "run-2"]
    assert host.settled_runs == ["run-1", "run-2"]


@pytest.mark.unit
def test_settle_active_runs_is_safe_on_empty_state() -> None:
    """无登记时 settle 直接返回空列表。"""

    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)

    assert coordinator.settle_active_runs(trigger="empty") == []
    assert host.settled_runs == []


@pytest.mark.unit
def test_settle_active_runs_can_be_called_multiple_times() -> None:
    """多次调用 settle 安全；底层 host.cancel_run_and_settle 自身幂等。"""

    host = _FakeHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-1")

    first = coordinator.settle_active_runs(trigger="first")
    second = coordinator.settle_active_runs(trigger="second")

    assert first == ["run-1"]
    assert second == ["run-1"]
    assert host.settled_runs == ["run-1", "run-1"]


@pytest.mark.unit
def test_settle_active_runs_falls_back_to_owner_active_runs() -> None:
    """observer 未登记时，owner-pid 兜底扫描必须命中并 settle。

    覆盖：fins 直接调 host.run_operation_*、interactive/prompt 在事件流首帧
    ``meta["run_id"]`` 之前的窗口期；这两类场景下 ``_active_runs`` 为空，
    必须由 ``Host.list_active_run_ids_for_current_owner()`` 兜底。
    """

    host = _FakeHost()
    host.owner_active_run_ids = ["run-orphan-1", "run-orphan-2"]
    coordinator = ProcessShutdownCoordinator(host=host)

    settled = coordinator.settle_active_runs(trigger="signal:SIGINT")

    assert settled == ["run-orphan-1", "run-orphan-2"]
    assert host.settled_runs == ["run-orphan-1", "run-orphan-2"]


@pytest.mark.unit
def test_settle_active_runs_merges_observer_and_owner_lists_without_duplication() -> None:
    """observer 登记 + owner 兜底合并去重，observer 路径优先保留顺序。"""

    host = _FakeHost()
    host.owner_active_run_ids = ["run-observer", "run-orphan"]
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-observer")

    settled = coordinator.settle_active_runs(trigger="signal:SIGINT")

    assert settled == ["run-observer", "run-orphan"]
    assert host.settled_runs == ["run-observer", "run-orphan"]


@pytest.mark.unit
def test_settle_active_runs_swallows_owner_scan_failure() -> None:
    """owner 兜底扫描抛异常时，已登记 run 仍要继续 settle。"""

    class _ScanFailureHost(_FakeHost):
        def list_active_run_ids_for_current_owner(self) -> list[str]:
            raise RuntimeError("registry unreachable")

    host = _ScanFailureHost()
    coordinator = ProcessShutdownCoordinator(host=host)
    coordinator.register_active_run("run-observer")

    settled = coordinator.settle_active_runs(trigger="signal:SIGINT")

    assert settled == ["run-observer"]
    assert host.settled_runs == ["run-observer"]
