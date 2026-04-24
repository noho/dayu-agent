"""应用层公共事件契约。

该模块只定义 UI / Service / Host 共享的数据模型，
不负责把 Engine 事件映射成应用层事件，避免 ``contracts -> engine`` 反向依赖。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Protocol, runtime_checkable


class AppEventType(Enum):
    """应用层事件类型。"""

    CONTENT_DELTA = "content_delta"
    REASONING_DELTA = "reasoning_delta"
    FINAL_ANSWER = "final_answer"
    CANCELLED = "cancelled"
    TOOL_EVENT = "tool_event"
    ITERATION_START = "iteration_start"
    WARNING = "warning"
    ERROR = "error"
    METADATA = "metadata"
    DONE = "done"


@runtime_checkable
class PublishedRunEventProtocol(Protocol):
    """Host 事件总线可发布的稳定事件包络。"""

    @property
    def type(self) -> object:
        """返回事件类型对象。"""

        ...

    @property
    def payload(self) -> object:
        """返回事件负载对象。"""

        ...


@dataclass
class AppEvent:
    """应用层标准事件。

    Attributes:
        type: 事件类型。
        payload: 事件负载。
        meta: 额外元数据。
    """

    type: AppEventType
    payload: Any
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AppResult:
    """应用层一次执行结果。

    Attributes:
        content: 最终文本内容。
        errors: 错误列表。
        warnings: 告警列表。
        degraded: 是否降级。
        filtered: 是否为受过滤完成态。
    """

    content: str
    errors: list[str]
    warnings: list[str]
    degraded: bool = False
    filtered: bool = False


def extract_cancel_reason(payload: Any) -> str | None:
    """从 CANCELLED 事件负载中提取 ``cancel_reason`` 字段。

    该 helper 只做纯解析，不拼装任何面向人的提示文案；展示文案留在各
    消费端（CLI / 写作流水线 / Web 等）自行组装，避免把 UI 文案约定带进
    ``contracts`` 稳定边界。

    Args:
        payload: 事件 payload；通常为 dict，也可能是任意对象。

    Returns:
        非空 ``cancel_reason`` 字符串；若 payload 非 dict、字段缺失或去空后为空则返回 ``None``。

    Raises:
        无。
    """

    if not isinstance(payload, dict):
        return None
    cancel_reason = str(payload.get("cancel_reason") or "").strip()
    return cancel_reason or None


__all__ = [
    "AppEvent",
    "AppEventType",
    "AppResult",
    "PublishedRunEventProtocol",
    "extract_cancel_reason",
]
