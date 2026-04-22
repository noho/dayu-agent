"""Host executor 多 lane 叠加 + permit 回滚语义测试。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pytest

from dayu.contracts.host_execution import HostedRunSpec
from dayu.host.concurrency import HOST_AGENT_LANE
from dayu.host.executor import DefaultHostExecutor, _required_lanes_for_spec


@dataclass
class _Permit:
    """测试用 permit。"""

    permit_id: str
    lane: str


class _RecordingGovernor:
    """记录 acquire/release 顺序的测试用 governor。"""

    def __init__(
        self,
        *,
        acquire_failure: Callable[[str], bool] | None = None,
    ) -> None:
        self.acquired: list[str] = []
        self.released: list[str] = []
        self._acquire_failure = acquire_failure or (lambda _lane: False)

    def acquire(self, lane: str, *, timeout: float | None = None) -> _Permit:
        """模拟 acquire，`acquire_failure(lane)` 返回 True 时抛 TimeoutError。"""

        del timeout
        if self._acquire_failure(lane):
            raise TimeoutError(f"模拟 acquire 失败: {lane}")
        self.acquired.append(lane)
        return _Permit(permit_id=f"permit-{lane}", lane=lane)

    def acquire_many(
        self, lanes: list[str], *, timeout: float | None = None
    ) -> list[_Permit]:
        """原子多 lane acquire 的测试实现。

        任何 lane 触发 ``acquire_failure`` 视为整体失败：既不记录已获取的 lane，
        也不返回 permit，语义对齐 SQLite 实现"要么全拿要么全不拿"。
        """

        del timeout
        for lane_name in lanes:
            if self._acquire_failure(lane_name):
                raise TimeoutError(f"模拟 acquire_many 失败: {lane_name}")
        permits: list[_Permit] = []
        for lane_name in lanes:
            self.acquired.append(lane_name)
            permits.append(_Permit(permit_id=f"permit-{lane_name}", lane=lane_name))
        return permits

    def try_acquire(self, lane: str):
        """非阻塞尝试占位实现。"""

        del lane
        return None

    def release(self, permit: _Permit) -> None:
        """模拟 release。"""

        self.released.append(permit.lane)

    def get_lane_status(self, lane: str):
        """未使用。"""

        raise NotImplementedError()

    def get_all_status(self):
        """未使用。"""

        raise NotImplementedError()

    def cleanup_stale_permits(self):
        """返回空列表。"""

        return []


class _StubEventBus:
    """占位事件总线。"""

    def publish(self, run_id: str, event: object) -> None:
        """占位实现。"""

        del run_id, event

    def subscribe(self, *, run_id: str | None = None, session_id: str | None = None):
        """未使用。"""

        raise NotImplementedError()


class _StubRunRegistry:
    """最小 run 注册表。"""

    def __init__(self) -> None:
        self.started: list[str] = []

    def start_run(self, run_id: str) -> None:
        """记录 run 已启动，满足 executor 行为。"""

        self.started.append(run_id)

    def mark_finished(self, run_id: str, **kwargs: object) -> None:
        """占位实现。"""

        del run_id, kwargs


@pytest.mark.unit
def test_required_lanes_agent_without_business_lane_returns_agent_lane_only() -> None:
    """Agent 执行 + business_concurrency_lane=None → 仅持 llm_api。"""

    spec = HostedRunSpec(operation_name="chat", session_id="s1", business_concurrency_lane=None)
    assert _required_lanes_for_spec(spec, include_agent_lane=True) == [HOST_AGENT_LANE]


@pytest.mark.unit
def test_required_lanes_agent_with_business_lane_returns_sorted_pair() -> None:
    """Agent 执行 + business=write_chapter → 按字母序 ['llm_api', 'write_chapter']。"""

    spec = HostedRunSpec(
        operation_name="write_pipeline",
        session_id="s1",
        business_concurrency_lane="write_chapter",
    )
    assert _required_lanes_for_spec(spec, include_agent_lane=True) == [
        HOST_AGENT_LANE,
        "write_chapter",
    ]


@pytest.mark.unit
def test_required_lanes_non_agent_with_business_lane_returns_business_only() -> None:
    """非 Agent 执行 + business=sec_download → 仅持 sec_download。"""

    spec = HostedRunSpec(
        operation_name="fins_download",
        session_id="s1",
        business_concurrency_lane="sec_download",
    )
    assert _required_lanes_for_spec(spec, include_agent_lane=False) == ["sec_download"]


@pytest.mark.unit
def test_required_lanes_raises_when_business_lane_equals_host_agent_lane() -> None:
    """Service 误填 llm_api 到 business 字段应被守卫拒绝。"""

    spec = HostedRunSpec(
        operation_name="chat",
        session_id="s1",
        business_concurrency_lane=HOST_AGENT_LANE,
    )
    with pytest.raises(ValueError, match="business_concurrency_lane"):
        _required_lanes_for_spec(spec, include_agent_lane=True)


@pytest.mark.unit
def test_start_run_propagates_timeout_without_leaking_permits() -> None:
    """acquire_many 失败时 executor 不自己回滚；governor 原子保证无 permit 残留。

    新 contract：多 lane 获取由 governor 在单事务内完成，要么全拿要么全不拿。
    executor 层只需把 TimeoutError 透传上去，不再承担 permit 回滚。
    `_RecordingGovernor.acquire_many` 实现对齐 SQLite：任一 lane 失败就整体抛错、
    不记录任何 lane；该断言同时锁住"executor 不再重复调用 release"。
    """

    def _fail_on_write_chapter(lane: str) -> bool:
        return lane == "write_chapter"

    governor = _RecordingGovernor(acquire_failure=_fail_on_write_chapter)
    executor = DefaultHostExecutor(
        run_registry=_StubRunRegistry(),  # type: ignore[arg-type]
        concurrency_governor=governor,  # type: ignore[arg-type]
        event_bus=_StubEventBus(),  # type: ignore[arg-type]
    )
    spec = HostedRunSpec(
        operation_name="write_pipeline",
        session_id="s1",
        business_concurrency_lane="write_chapter",
    )

    with pytest.raises(TimeoutError):
        executor._start_run(spec=spec, run_id="run-rollback", include_agent_lane=True)

    # acquire_many 原子失败：governor 没记录任何 lane，executor 也不会调用 release。
    assert governor.acquired == []
    assert governor.released == []
