"""AsyncQueueEventBus 测试。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast
from unittest.mock import Mock

import pytest

from dayu.contracts.fins import FinsCommandName, FinsEvent, FinsEventType, ProcessResultData
from dayu.host import event_bus as event_bus_module
from dayu.host.event_bus import AsyncQueueEventBus
from dayu.host.protocols import RunRegistryProtocol
from dayu.contracts.events import AppEvent, AppEventType
from dayu.contracts.run import RunRecord, RunState
from dayu.log import Log


def _make_event(event_type: AppEventType = AppEventType.CONTENT_DELTA, payload: str = "hello") -> AppEvent:
    """创建测试用 AppEvent。"""
    return AppEvent(type=event_type, payload=payload)


def _make_fins_event() -> FinsEvent:
    """创建测试用 FinsEvent。"""

    return FinsEvent(
        type=FinsEventType.RESULT,
        command=FinsCommandName.PROCESS,
        payload=ProcessResultData(pipeline="sec", status="ok", ticker="AAPL"),
    )


@dataclass
class _MockRunRegistry:
    """模拟 RunRegistry，用于解析 run→session 映射。"""

    _runs: dict[str, str | None]  # run_id → session_id

    def get_run(self, run_id: str) -> RunRecord | None:
        """通过 run_id 返回 mock RunRecord。"""
        if run_id not in self._runs:
            return None
        return RunRecord(
            run_id=run_id,
            session_id=self._runs[run_id],
            service_type="test",
            scene_name=None,
            state=RunState.RUNNING,
            created_at=datetime.now(timezone.utc),
            owner_pid=1,
        )


def _run_registry(mock: _MockRunRegistry) -> RunRegistryProtocol:
    """在测试装配边界把 mock registry 收窄为 RunRegistryProtocol。"""

    return cast(RunRegistryProtocol, mock)


def _subscription(subscription: object) -> event_bus_module._Subscription:
    """把订阅句柄收窄为内部 Subscription 以覆盖溢出分支。"""

    return cast(event_bus_module._Subscription, subscription)


class TestPublishAndSubscribe:
    """publish / subscribe 基本流程。"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subscribe_receives_event(self) -> None:
        """订阅者收到发布的事件。"""
        bus = AsyncQueueEventBus()
        sub = bus.subscribe(run_id="run_1")
        event = _make_event()
        bus.publish("run_1", event)
        sub.close()  # 放入 sentinel 以终止迭代

        events = [e async for e in sub]
        assert len(events) == 1
        assert events[0].payload == "hello"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subscribe_filter_by_run_id(self) -> None:
        """只收到匹配 run_id 的事件。"""
        bus = AsyncQueueEventBus()
        sub = bus.subscribe(run_id="run_1")

        bus.publish("run_1", _make_event(payload="yes"))
        bus.publish("run_2", _make_event(payload="no"))
        sub.close()

        events = [e async for e in sub]
        assert len(events) == 1
        assert events[0].payload == "yes"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subscribe_all_runs(self) -> None:
        """run_id=None 时收到所有 run 的事件。"""
        bus = AsyncQueueEventBus()
        sub = bus.subscribe()

        bus.publish("run_1", _make_event(payload="a"))
        bus.publish("run_2", _make_event(payload="b"))
        sub.close()

        events = [e async for e in sub]
        assert len(events) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subscribe_accepts_fins_event(self) -> None:
        """事件总线应允许转发非 AppEvent 的稳定运行事件。"""

        bus = AsyncQueueEventBus()
        sub = bus.subscribe(run_id="run_1")
        event = _make_fins_event()

        bus.publish("run_1", event)
        sub.close()

        events = [e async for e in sub]
        assert len(events) == 1
        assert isinstance(events[0], FinsEvent)
        assert events[0].payload == event.payload

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subscribe_filter_by_session_id(self) -> None:
        """按 session_id 过滤事件。"""
        mock = _MockRunRegistry(_runs={"run_1": "sess_a", "run_2": "sess_b"})
        bus = AsyncQueueEventBus(run_registry=_run_registry(mock))
        sub = bus.subscribe(session_id="sess_a")

        bus.publish("run_1", _make_event(payload="yes"))
        bus.publish("run_2", _make_event(payload="no"))
        sub.close()

        events = [e async for e in sub]
        assert len(events) == 1
        assert events[0].payload == "yes"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_subscribers(self) -> None:
        """多个 subscriber 各自独立收到事件。"""
        bus = AsyncQueueEventBus()
        sub1 = bus.subscribe(run_id="run_1")
        sub2 = bus.subscribe(run_id="run_1")

        bus.publish("run_1", _make_event(payload="shared"))
        sub1.close()
        sub2.close()

        events1 = [e async for e in sub1]
        events2 = [e async for e in sub2]
        assert len(events1) == 1
        assert len(events2) == 1


class TestSubscriptionLifecycle:
    """Subscription 关闭和溢出行为。"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_closed_subscription_skipped(self) -> None:
        """已关闭的订阅不再接收事件。"""
        bus = AsyncQueueEventBus()
        sub = bus.subscribe(run_id="run_1")
        sub.close()

        bus.publish("run_1", _make_event())
        events = [e async for e in sub]
        assert events == []

    @pytest.mark.unit
    def test_is_closed(self) -> None:
        """close() 后 is_closed 为 True。"""
        bus = AsyncQueueEventBus()
        sub = bus.subscribe()
        assert not sub.is_closed
        sub.close()
        assert sub.is_closed

    @pytest.mark.unit
    def test_close_is_idempotent(self) -> None:
        """多次 close 不抛异常。"""
        bus = AsyncQueueEventBus()
        sub = bus.subscribe()
        sub.close()
        sub.close()  # 不抛异常

    @pytest.mark.unit
    def test_queue_overflow_logs_with_project_log(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """队列溢出时通过项目统一日志入口输出 warning。"""
        warning_mock = Mock()
        monkeypatch.setattr(Log, "warning", warning_mock)

        bus = AsyncQueueEventBus()
        sub = bus.subscribe(run_id="run_1")
        internal_sub = _subscription(sub)
        internal_sub._queue = asyncio.Queue(maxsize=1)
        internal_sub._queue.put_nowait(_make_event(payload="old"))

        accepted = internal_sub.try_put(_make_event(payload="new"))

        assert accepted is False
        warning_mock.assert_called_once_with(
            "事件队列溢出，丢弃旧事件: run_id=run_1",
            module="HOST.EVENT_BUS",
        )


class TestCleanupClosed:
    """_cleanup_closed 内部维护。"""

    @pytest.mark.unit
    def test_cleanup_removes_closed_subscriptions(self) -> None:
        """_cleanup_closed 清除已关闭的订阅。"""
        bus = AsyncQueueEventBus()
        sub1 = bus.subscribe()
        sub2 = bus.subscribe()
        sub1.close()

        assert len(bus._subscriptions) == 2
        bus._cleanup_closed()
        assert len(bus._subscriptions) == 1

    @pytest.mark.unit
    def test_publish_reaps_closed_subscriptions(self) -> None:
        """publish 应顺带回收已关闭订阅，防止长驻进程内存泄漏。

        回归 finding 009：曾经 _cleanup_closed 只作为内部方法存在但没有
        任何调用点，导致 subscribe → close → publish 反复循环时，
        _subscriptions 长度持续增长。
        """

        bus = AsyncQueueEventBus()
        _ = bus.subscribe()  # 保留一个活跃订阅
        # 模拟多轮 subscribe → close → publish
        for _ in range(5):
            sub = bus.subscribe()
            sub.close()
            bus.publish("run_x", _make_event())
        # 只剩 1 个活跃订阅（首次保留的），5 个已关闭的都应被回收。
        assert len(bus._subscriptions) == 1
