"""Fins PDF 下载段 lane gate 测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from dayu.contracts.cancellation import CancelledError, CancellationToken
from dayu.host.protocols import ConcurrencyPermit, LaneStatus
from dayu.services.concurrency_lanes import LANE_CN_DOWNLOAD, LANE_HK_DOWNLOAD
from dayu.services.fins_download_lane_gate import GovernorCnDownloadPdfGate


@dataclass
class _FakeGovernor:
    """测试用并发 governor。"""

    busy: bool = False
    cleanup_releases: bool = False
    acquired: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)
    cleanup_calls: int = 0

    def try_acquire(self, lane: str) -> ConcurrencyPermit | None:
        """尝试获取测试 permit。"""

        if self.busy:
            return None
        self.busy = True
        self.acquired.append(lane)
        return ConcurrencyPermit(
            permit_id=f"permit_{lane}",
            lane=lane,
            acquired_at=datetime.now(timezone.utc),
        )

    def release(self, permit: ConcurrencyPermit) -> None:
        """释放测试 permit。"""

        self.busy = False
        self.released.append(permit.lane)

    def acquire(
        self,
        lane: str,
        *,
        timeout: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> ConcurrencyPermit:
        """测试中不走阻塞 acquire。"""

        del lane, timeout, cancellation_token
        raise AssertionError("PDF gate 应使用可轮询取消的 try_acquire")

    def acquire_many(
        self,
        lanes: list[str],
        *,
        timeout: float | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> list[ConcurrencyPermit]:
        """测试中不走多 lane acquire。"""

        del lanes, timeout, cancellation_token
        raise AssertionError("PDF gate 不应使用 acquire_many")

    def get_lane_status(self, lane: str) -> LaneStatus:
        """返回测试 lane 状态。"""

        return LaneStatus(lane=lane, max_concurrent=1, active=1 if self.busy else 0)

    def get_all_status(self) -> dict[str, LaneStatus]:
        """返回全部测试 lane 状态。"""

        return {}

    def cleanup_stale_permits(self) -> list[str]:
        """记录 stale permit 清理调用。"""

        self.cleanup_calls += 1
        if self.cleanup_releases:
            self.busy = False
            return ["permit_stale"]
        return []


def test_pdf_gate_acquires_provider_lane_and_releases() -> None:
    """PDF gate 应按 provider 获取对应 lane 并在退出时释放。"""

    governor = _FakeGovernor()
    gate = GovernorCnDownloadPdfGate(governor=governor, acquire_timeout_seconds=0.01)

    with gate.lease_for_provider("cninfo"):
        assert governor.acquired == [LANE_CN_DOWNLOAD]
        assert governor.busy is True

    assert governor.released == [LANE_CN_DOWNLOAD]
    assert governor.busy is False


def test_pdf_gate_wait_is_cancelable() -> None:
    """等待 PDF 下载 lane 时应响应取消。"""

    governor = _FakeGovernor(busy=True)
    gate = GovernorCnDownloadPdfGate(governor=governor, acquire_timeout_seconds=1.0)

    with pytest.raises(CancelledError):
        with gate.lease_for_provider("hkexnews", cancel_checker=lambda: True):
            pass

    assert governor.acquired == []


def test_pdf_gate_reaps_stale_permit_while_waiting() -> None:
    """等待 PDF 下载 lane 时应按节流周期清理 stale permit。"""

    governor = _FakeGovernor(busy=True, cleanup_releases=True)
    gate = GovernorCnDownloadPdfGate(
        governor=governor,
        acquire_timeout_seconds=1.0,
        stale_reap_interval_seconds=0.0,
    )

    with gate.lease_for_provider("hkexnews"):
        assert governor.cleanup_calls >= 1
        assert governor.acquired == [LANE_HK_DOWNLOAD]
