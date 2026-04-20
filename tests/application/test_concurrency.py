"""SQLiteConcurrencyGovernor 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.host.concurrency import SQLiteConcurrencyGovernor
from dayu.host.host_store import HostStore


@pytest.fixture()
def host_store(tmp_path: Path) -> HostStore:
    """创建临时 HostStore。"""
    store = HostStore(tmp_path / "test.db")
    store.initialize_schema()
    return store


@pytest.fixture()
def governor(host_store: HostStore) -> SQLiteConcurrencyGovernor:
    """创建并发治理器，lane 容量设小方便测试。"""
    return SQLiteConcurrencyGovernor(
        host_store,
        lane_config={"llm_api": 2, "sec_download": 1},
    )


class TestTryAcquireAndRelease:
    """try_acquire / release 基本测试。"""

    @pytest.mark.unit
    def test_try_acquire_success(self, governor: SQLiteConcurrencyGovernor) -> None:
        """获取许可成功。"""
        permit = governor.try_acquire("llm_api")
        assert permit is not None
        assert permit.lane == "llm_api"
        assert permit.permit_id.startswith("permit_")

    @pytest.mark.unit
    def test_try_acquire_respects_limit(self, governor: SQLiteConcurrencyGovernor) -> None:
        """超出容量时返回 None。"""
        p1 = governor.try_acquire("sec_download")
        assert p1 is not None
        p2 = governor.try_acquire("sec_download")
        assert p2 is None

    @pytest.mark.unit
    def test_release_frees_slot(self, governor: SQLiteConcurrencyGovernor) -> None:
        """释放后另一个请求可以获取。"""
        permit = governor.try_acquire("sec_download")
        assert permit is not None
        governor.release(permit)
        # 释放后重新可用
        p2 = governor.try_acquire("sec_download")
        assert p2 is not None

    @pytest.mark.unit
    def test_try_acquire_unknown_lane_raises(self, governor: SQLiteConcurrencyGovernor) -> None:
        """未配置的 lane 抛出 ValueError。"""
        with pytest.raises(ValueError, match="未配置的并发通道"):
            governor.try_acquire("unknown_lane")

    @pytest.mark.unit
    def test_multiple_permits_same_lane(self, governor: SQLiteConcurrencyGovernor) -> None:
        """llm_api 容量为 2，可同时获取 2 个许可。"""
        p1 = governor.try_acquire("llm_api")
        p2 = governor.try_acquire("llm_api")
        assert p1 is not None
        assert p2 is not None
        p3 = governor.try_acquire("llm_api")
        assert p3 is None


class TestAcquireBlocking:
    """acquire 阻塞/超时测试。"""

    @pytest.mark.unit
    def test_acquire_success(self, governor: SQLiteConcurrencyGovernor) -> None:
        """有容量时 acquire 立即返回。"""
        permit = governor.acquire("llm_api", timeout=1.0)
        assert permit is not None

    @pytest.mark.unit
    def test_acquire_timeout(self, governor: SQLiteConcurrencyGovernor) -> None:
        """无容量时 acquire 超时抛出 TimeoutError。"""
        governor.try_acquire("sec_download")
        with pytest.raises(TimeoutError, match="获取并发许可超时"):
            governor.acquire("sec_download", timeout=0.2)

    @pytest.mark.unit
    def test_acquire_without_explicit_timeout_keeps_current_infinite_wait_contract(
        self,
        governor: SQLiteConcurrencyGovernor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """显式不传 timeout 时，governor 仍保持协议约定的无限等待语义。"""

        governor.try_acquire("sec_download")

        def _raise_after_first_sleep(_seconds: float) -> None:
            """测试桩：避免测试真的无限阻塞。"""

            raise RuntimeError("stop loop")

        monkeypatch.setattr("dayu.host.concurrency.time.sleep", _raise_after_first_sleep)

        with pytest.raises(RuntimeError, match="stop loop"):
            governor.acquire("sec_download")


class TestLaneStatus:
    """get_lane_status / get_all_status 测试。"""

    @pytest.mark.unit
    def test_get_lane_status_initial(self, governor: SQLiteConcurrencyGovernor) -> None:
        """初始状态 active=0。"""
        status = governor.get_lane_status("llm_api")
        assert status.lane == "llm_api"
        assert status.max_concurrent == 2
        assert status.active == 0

    @pytest.mark.unit
    def test_get_lane_status_after_acquire(self, governor: SQLiteConcurrencyGovernor) -> None:
        """获取许可后 active 增加。"""
        governor.try_acquire("llm_api")
        status = governor.get_lane_status("llm_api")
        assert status.active == 1

    @pytest.mark.unit
    def test_get_all_status(self, governor: SQLiteConcurrencyGovernor) -> None:
        """get_all_status 返回所有已配置 lane。"""
        all_status = governor.get_all_status()
        assert "llm_api" in all_status
        assert "sec_download" in all_status


class TestCleanupStalePermits:
    """cleanup_stale_permits 测试。"""

    @pytest.mark.unit
    def test_cleanup_no_stale(self, governor: SQLiteConcurrencyGovernor) -> None:
        """当前进程的许可不会被清理。"""
        governor.try_acquire("llm_api")
        stale = governor.cleanup_stale_permits()
        assert stale == []

    @pytest.mark.unit
    def test_cleanup_dead_pid_permit(
        self,
        governor: SQLiteConcurrencyGovernor,
        host_store: HostStore,
    ) -> None:
        """owner_pid 不存在的许可被清理。"""
        permit = governor.try_acquire("llm_api")
        assert permit is not None

        # 篡改 owner_pid 为一个不存在的 PID
        conn = host_store.get_connection()
        conn.execute(
            "UPDATE permits SET owner_pid = ? WHERE permit_id = ?",
            (999999, permit.permit_id),
        )
        conn.commit()

        stale = governor.cleanup_stale_permits()
        assert permit.permit_id in stale

        # 清理后 lane 重新可用
        status = governor.get_lane_status("llm_api")
        assert status.active == 0
