"""CancellationBridge 测试。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

import pytest

from dayu.host.cancellation_bridge import CancellationBridge
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
