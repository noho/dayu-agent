"""``DefaultHostExecutor`` run 资源注册表语义单元测试（覆盖 #46）。

覆盖：
- ``release_resources_for_run`` 从注册表 atomic-pop 后调用 governor.release /
  watcher.stop / bridge.stop；
- 与 ``_finish_run`` 互相幂等：先后两路调用只会真正释放一次；
- 未注册 run_id 静默 no-op，不抛异常。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import cast

import pytest

from dayu.host.cancellation_bridge import CancellationBridge
from dayu.host.executor import DefaultHostExecutor, _RunResources
from dayu.host.protocols import ConcurrencyGovernorProtocol, ConcurrencyPermit, LaneStatus
from dayu.host.executor import RunDeadlineWatcher
from tests.application.conftest import StubRunRegistry


@dataclass
class _SpyBridge:
    """记录 ``stop`` 调用次数的伪 CancellationBridge。"""

    stop_calls: int = 0

    def stop(self) -> None:
        """仅计数，无副作用。"""

        self.stop_calls += 1

    def start(self) -> None:
        """测试不需要启动，留空。"""


@dataclass
class _SpyWatcher:
    """记录 ``stop`` 调用次数的伪 RunDeadlineWatcher。"""

    stop_calls: int = 0

    def stop(self) -> None:
        """仅计数，无副作用。"""

        self.stop_calls += 1

    def start(self) -> None:
        """测试不需要启动，留空。"""


@dataclass
class _SpyGovernor:
    """记录 ``release`` 调用次数的伪 ConcurrencyGovernor。"""

    released: list[ConcurrencyPermit] = field(default_factory=list)
    raise_on_release: bool = False

    def release(self, permit: ConcurrencyPermit) -> None:
        """记录被释放的 permit，按需抛异常以验证错误分支。"""

        if self.raise_on_release:
            raise RuntimeError("permit 释放故意失败")
        self.released.append(permit)

    def acquire(self, lane: str, *, timeout: float | None = None) -> ConcurrencyPermit:
        """测试不会触发，未实现。"""

        raise NotImplementedError

    def acquire_many(
        self,
        lanes: list[str],
        *,
        timeout: float | None = None,
        cancellation_token: object | None = None,
    ) -> list[ConcurrencyPermit]:
        """测试不会触发，未实现。"""

        raise NotImplementedError

    def try_acquire(self, lane: str) -> ConcurrencyPermit | None:
        """测试不会触发，未实现。"""

        raise NotImplementedError

    def get_lane_status(self, lane: str) -> LaneStatus:
        """测试不会触发，未实现。"""

        raise NotImplementedError

    def get_all_status(self) -> dict[str, LaneStatus]:
        """测试不会触发，未实现。"""

        raise NotImplementedError

    def cleanup_stale_permits(self) -> list[str]:
        """测试不会触发，未实现。"""

        raise NotImplementedError


def _build_executor(governor: _SpyGovernor | None = None) -> DefaultHostExecutor:
    """构造仅测试用的最小 ``DefaultHostExecutor`` 实例。"""

    return DefaultHostExecutor(
        run_registry=cast("object", StubRunRegistry()),  # type: ignore[arg-type]
        concurrency_governor=cast(ConcurrencyGovernorProtocol, governor) if governor is not None else None,
    )


def _build_permit(permit_id: str, lane: str) -> ConcurrencyPermit:
    """构造测试 permit。"""

    return ConcurrencyPermit(permit_id=permit_id, lane=lane, acquired_at=datetime.now(tz=timezone.utc))


def _inject_resources(
    *,
    executor: DefaultHostExecutor,
    run_id: str,
    bridge: _SpyBridge,
    watcher: _SpyWatcher,
    permits: list[ConcurrencyPermit],
) -> None:
    """直接把伪资源注入到 executor 注册表，绕过 ``_start_run``。"""

    executor._run_resources[run_id] = _RunResources(
        bridge=cast(CancellationBridge, bridge),
        deadline_watcher=cast(RunDeadlineWatcher, watcher),
        permits=list(permits),
    )


@pytest.mark.unit
def test_release_resources_for_run_drains_registry_and_stops_components() -> None:
    """同步释放路径：从注册表 pop 后释放 permit / 停 watcher / 停 bridge。"""

    governor = _SpyGovernor()
    executor = _build_executor(governor)
    bridge = _SpyBridge()
    watcher = _SpyWatcher()
    permits = [_build_permit("p1", "lane_a"), _build_permit("p2", "lane_b")]
    _inject_resources(executor=executor, run_id="r1", bridge=bridge, watcher=watcher, permits=permits)

    executor.release_resources_for_run("r1")

    assert "r1" not in executor._run_resources
    assert [p.permit_id for p in governor.released] == ["p2", "p1"]  # reversed
    assert watcher.stop_calls == 1
    assert bridge.stop_calls == 1


@pytest.mark.unit
def test_release_resources_for_run_unknown_run_is_noop() -> None:
    """未注册 run_id 静默退出。"""

    governor = _SpyGovernor()
    executor = _build_executor(governor)

    executor.release_resources_for_run("missing")

    assert governor.released == []
    assert executor._run_resources == {}


@pytest.mark.unit
def test_release_resources_for_run_then_finish_run_is_idempotent() -> None:
    """先同步释放后异步 ``_finish_run``：第二步必须 no-op。"""

    governor = _SpyGovernor()
    executor = _build_executor(governor)
    bridge = _SpyBridge()
    watcher = _SpyWatcher()
    permits = [_build_permit("p1", "lane_a")]
    _inject_resources(executor=executor, run_id="r1", bridge=bridge, watcher=watcher, permits=permits)

    executor.release_resources_for_run("r1")
    # 第二次（异步 finally 路径）要 no-op
    executor._finish_run(
        bridge=cast(CancellationBridge, bridge),
        deadline_watcher=cast(RunDeadlineWatcher, watcher),
        permits=permits,
        run_id="r1",
    )

    assert len(governor.released) == 1
    assert watcher.stop_calls == 1
    assert bridge.stop_calls == 1


@pytest.mark.unit
def test_finish_run_then_release_resources_for_run_is_idempotent() -> None:
    """先异步 ``_finish_run`` 后同步释放：第二步必须 no-op。"""

    governor = _SpyGovernor()
    executor = _build_executor(governor)
    bridge = _SpyBridge()
    watcher = _SpyWatcher()
    permits = [_build_permit("p1", "lane_a")]
    _inject_resources(executor=executor, run_id="r1", bridge=bridge, watcher=watcher, permits=permits)

    executor._finish_run(
        bridge=cast(CancellationBridge, bridge),
        deadline_watcher=cast(RunDeadlineWatcher, watcher),
        permits=permits,
        run_id="r1",
    )
    executor.release_resources_for_run("r1")

    assert len(governor.released) == 1
    assert watcher.stop_calls == 1
    assert bridge.stop_calls == 1


@pytest.mark.unit
def test_release_resources_for_run_logs_and_continues_on_permit_release_failure() -> None:
    """permit 释放抛异常仅记日志，不影响 watcher / bridge.stop。"""

    governor = _SpyGovernor(raise_on_release=True)
    executor = _build_executor(governor)
    bridge = _SpyBridge()
    watcher = _SpyWatcher()
    permits = [_build_permit("p1", "lane_a")]
    _inject_resources(executor=executor, run_id="r1", bridge=bridge, watcher=watcher, permits=permits)

    executor.release_resources_for_run("r1")

    assert watcher.stop_calls == 1
    assert bridge.stop_calls == 1
    assert "r1" not in executor._run_resources


@pytest.mark.unit
def test_host_cancel_run_and_settle_invokes_executor_release_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host.cancel_run_and_settle 必须把资源释放转发给 DefaultHostExecutor。"""

    from typing import cast as _cast

    from dayu.contracts.run import RunState
    from dayu.host.host import Host
    from dayu.host.host_execution import HostExecutorProtocol
    from dayu.host.protocols import RunRegistryProtocol, SessionRegistryProtocol
    from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore
    from tests.application.conftest import StubSessionRegistry

    registry = StubRunRegistry()
    record = registry.register_run(service_type="chat", session_id="session-1")
    registry.start_run(record.run_id)

    executor = DefaultHostExecutor(run_registry=_cast(RunRegistryProtocol, registry))
    host = Host(
        executor=_cast(HostExecutorProtocol, executor),
        session_registry=_cast(SessionRegistryProtocol, StubSessionRegistry()),
        run_registry=_cast(RunRegistryProtocol, registry),
        reply_outbox_store=InMemoryReplyOutboxStore(),
    )

    release_calls: list[str] = []

    def _spy(run_id: str) -> None:
        release_calls.append(run_id)

    monkeypatch.setattr(executor, "release_resources_for_run", _spy)

    settled = host.cancel_run_and_settle(record.run_id)

    assert settled.state == RunState.CANCELLED
    assert release_calls == [record.run_id]
