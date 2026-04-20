"""Host 层事件辅助函数。"""

from __future__ import annotations

from typing import Optional

from dayu.contracts.events import AppEvent, AppEventType
from dayu.engine.events import EventType, StreamEvent


def build_app_event_from_stream_event(event: StreamEvent) -> Optional[AppEvent]:
    """将 Engine 事件映射为 Host 对外事件。

    Args:
        event: Engine 事件对象。

    Returns:
        映射后的对外事件；若事件无需暴露则返回 ``None``。

    Raises:
        无。
    """

    if event.type == EventType.CONTENT_DELTA:
        return AppEvent(type=AppEventType.CONTENT_DELTA, payload=str(event.data or ""), meta=dict(event.metadata or {}))
    if event.type == EventType.REASONING_DELTA:
        return AppEvent(type=AppEventType.REASONING_DELTA, payload=str(event.data or ""), meta=dict(event.metadata or {}))
    if event.type == EventType.FINAL_ANSWER:
        payload = event.data if isinstance(event.data, dict) else {"content": str(event.data), "degraded": False}
        return AppEvent(type=AppEventType.FINAL_ANSWER, payload=payload, meta=dict(event.metadata or {}))
    if event.type in {
        EventType.TOOL_CALL_START,
        EventType.TOOL_CALL_DELTA,
        EventType.TOOL_CALL_DISPATCHED,
        EventType.TOOL_CALL_RESULT,
        EventType.TOOL_CALLS_BATCH_READY,
        EventType.TOOL_CALLS_BATCH_DONE,
    }:
        payload = {
            "engine_event_type": event.type.value,
            "data": event.data,
        }
        return AppEvent(type=AppEventType.TOOL_EVENT, payload=payload, meta=dict(event.metadata or {}))
    if event.type == EventType.WARNING:
        return AppEvent(type=AppEventType.WARNING, payload=event.data, meta=dict(event.metadata or {}))
    if event.type == EventType.ERROR:
        return AppEvent(type=AppEventType.ERROR, payload=event.data, meta=dict(event.metadata or {}))
    if event.type == EventType.METADATA:
        return AppEvent(type=AppEventType.METADATA, payload=event.data, meta=dict(event.metadata or {}))
    if event.type == EventType.DONE:
        return AppEvent(type=AppEventType.DONE, payload=event.data, meta=dict(event.metadata or {}))
    return None

__all__ = [
    "build_app_event_from_stream_event",
]
