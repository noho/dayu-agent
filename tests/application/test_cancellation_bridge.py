"""CancellationBridge 测试。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

import pytest

from dayu.host.cancellation_bridge import (
    CancellationBridge,
    _MAX_CONSECUTIVE_POLL_FAILURES,
)
from dayu.host.protocols import RunRegistryProtocol
from dayu.contracts.run import RunRecord, RunState
from dayu.contracts.cancellation import CancellationToken


@dataclass
class _MockRunRegistry:
    """用于测试的 RunRegistry mock。

    通过修改内部字段控制返回的 run 状态与取消请求意图。
    """

    _state: RunState = RunState.RUNNING
    _deleted: bool = False
    _cancel_requested_at: datetime | None = None

    def get_run(self, run_id: str) -> RunRecord | None:
        """返回 mock run 或 None。"""
        if self._deleted:
            return None
        return RunRecord(
            run_id=run_id,
            session_id=None,
            service_type="test",
            scene_name=None,
            state=self._state,
            created_at=datetime.now(timezone.utc),
            cancel_requested_at=self._cancel_requested_at,
            owner_pid=1,
        )


def _wait_until(predicate: Callable[[], bool], timeout_seconds: float, interval_seconds: float = 0.01) -> bool:
    """轮询等待条件成立。

    Args:
        predicate: 返回布尔值的条件函数。
        timeout_seconds: 最大等待秒数。
        interval_seconds: 轮询间隔秒数。

    Returns:
        条件是否在超时前成立。

    Raises:
        无。
    """

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_seconds)
    return predicate()


class TestCancellationBridgePolling:
    """CancellationBridge 轮询行为测试。"""

    @pytest.mark.unit
    def test_detects_cancel_and_triggers_token(self) -> None:
        """检测到取消请求意图后触发 CancellationToken。"""
        mock_registry = _MockRunRegistry(_state=RunState.RUNNING)
        token = CancellationToken()
        bridge = CancellationBridge(
            run_registry=cast(RunRegistryProtocol, mock_registry),
            run_id="run_test",
            token=token,
            poll_interval=0.05,
        )
        bridge.start()
        try:
            assert not token.is_cancelled()

            # 模拟外部写入取消请求
            mock_registry._cancel_requested_at = datetime.now(timezone.utc)

            assert _wait_until(token.is_cancelled, timeout_seconds=1.0)
        finally:
            bridge.stop()

    @pytest.mark.unit
    def test_stops_on_succeeded(self) -> None:
        """run 进入 SUCCEEDED 后自动停止轮询。"""
        mock_registry = _MockRunRegistry(_state=RunState.RUNNING)
        token = CancellationToken()
        bridge = CancellationBridge(
            run_registry=cast(RunRegistryProtocol, mock_registry),
            run_id="run_test",
            token=token,
            poll_interval=0.05,
        )
        bridge.start()

        mock_registry._state = RunState.SUCCEEDED
        time.sleep(0.2)

        # token 不应被取消
        assert not token.is_cancelled()
        # 线程应已退出
        assert bridge._thread is None or not bridge._thread.is_alive()
        bridge.stop()

    @pytest.mark.unit
    def test_stops_on_deleted_run(self) -> None:
        """run 被删除后自动停止轮询。"""
        mock_registry = _MockRunRegistry(_state=RunState.RUNNING)
        token = CancellationToken()
        bridge = CancellationBridge(
            run_registry=cast(RunRegistryProtocol, mock_registry),
            run_id="run_test",
            token=token,
            poll_interval=0.05,
        )
        bridge.start()

        mock_registry._deleted = True
        time.sleep(0.2)

        assert not token.is_cancelled()
        bridge.stop()

    @pytest.mark.unit
    def test_stop_is_reentrant(self) -> None:
        """stop() 可重入，多次调用安全。"""
        mock_registry = _MockRunRegistry(_state=RunState.RUNNING)
        token = CancellationToken()
        bridge = CancellationBridge(
            run_registry=cast(RunRegistryProtocol, mock_registry),
            run_id="run_test",
            token=token,
            poll_interval=0.05,
        )
        bridge.start()
        bridge.stop()
        bridge.stop()  # 不抛异常

    @pytest.mark.unit
    def test_start_is_idempotent(self) -> None:
        """start() 重复调用不创建多个线程。"""
        mock_registry = _MockRunRegistry(_state=RunState.RUNNING)
        token = CancellationToken()
        bridge = CancellationBridge(
            run_registry=cast(RunRegistryProtocol, mock_registry),
            run_id="run_test",
            token=token,
            poll_interval=0.05,
        )
        bridge.start()
        thread_1 = bridge._thread
        bridge.start()  # 线程仍在运行，应忽略
        thread_2 = bridge._thread
        assert thread_1 is thread_2
        bridge.stop()

    @pytest.mark.unit
    def test_registry_exception_does_not_crash(self) -> None:
        """查询异常不中断轮询。"""

        class _FailingRegistry:
            """首次查询抛异常，之后返回带取消请求意图的运行记录。"""

            def __init__(self) -> None:
                self._call_count = 0

            def get_run(self, run_id: str) -> RunRecord | None:
                self._call_count += 1
                if self._call_count <= 2:
                    raise RuntimeError("db error")
                return RunRecord(
                    run_id=run_id,
                    session_id=None,
                    service_type="test",
                    scene_name=None,
                    state=RunState.RUNNING,
                    created_at=datetime.now(timezone.utc),
                    cancel_requested_at=datetime.now(timezone.utc),
                    owner_pid=1,
                )

        token = CancellationToken()
        bridge = CancellationBridge(
            run_registry=cast(RunRegistryProtocol, _FailingRegistry()),
            run_id="run_test",
            token=token,
            poll_interval=0.05,
        )
        bridge.start()
        time.sleep(0.5)

        # 最终检测到取消请求
        assert token.is_cancelled()
        bridge.stop()

    @pytest.mark.unit
    def test_stops_after_consecutive_failures_reach_threshold(self) -> None:
        """持续异常达到阈值后退出轮询，不再无限重试。"""

        class _AlwaysFailingRegistry:
            """每次查询都抛异常的 registry。"""

            def __init__(self) -> None:
                self.call_count = 0

            def get_run(self, run_id: str) -> RunRecord | None:
                self.call_count += 1
                _ = run_id
                raise RuntimeError("db error")

        registry = _AlwaysFailingRegistry()
        token = CancellationToken()
        bridge = CancellationBridge(
            run_registry=cast(RunRegistryProtocol, registry),
            run_id="run_test",
            token=token,
            poll_interval=0.01,
        )
        bridge.start()
        # 等待轮询线程退出
        thread = bridge._thread
        assert thread is not None
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "持续失败达阈值后应退出轮询"
        # token 不应被取消（无法判定取消请求时不应误判取消）
        assert not token.is_cancelled()
        # 调用次数应在阈值附近（允许少量偏差，但远小于无限循环）
        assert registry.call_count >= _MAX_CONSECUTIVE_POLL_FAILURES
        assert registry.call_count <= _MAX_CONSECUTIVE_POLL_FAILURES + 2
        bridge.stop()

    @pytest.mark.unit
    def test_failure_counter_resets_on_success(self) -> None:
        """偶发异常后能成功一次查询，失败计数应清零，不会过早退出。"""

        class _IntermittentRegistry:
            """前若干次查询失败，之后稳定成功。"""

            def __init__(self, fail_times: int) -> None:
                self._fail_times = fail_times
                self.call_count = 0

            def get_run(self, run_id: str) -> RunRecord | None:
                self.call_count += 1
                if self.call_count <= self._fail_times:
                    raise RuntimeError("transient error")
                return RunRecord(
                    run_id=run_id,
                    session_id=None,
                    service_type="test",
                    scene_name=None,
                    state=RunState.RUNNING,
                    created_at=datetime.now(timezone.utc),
                    cancel_requested_at=None,
                    owner_pid=1,
                )

        # 失败次数小于阈值，且后续始终成功；轮询线程应保持运行。
        fail_times = max(1, _MAX_CONSECUTIVE_POLL_FAILURES - 2)
        registry = _IntermittentRegistry(fail_times=fail_times)
        token = CancellationToken()
        bridge = CancellationBridge(
            run_registry=cast(RunRegistryProtocol, registry),
            run_id="run_test",
            token=token,
            poll_interval=0.01,
        )
        bridge.start()
        # 给足时间让失败次数累计后再清零，并继续多轮成功轮询。
        time.sleep(0.3)
        thread = bridge._thread
        assert thread is not None
        assert thread.is_alive(), "失败计数清零后轮询线程应继续运行"
        assert not token.is_cancelled()
        bridge.stop()
