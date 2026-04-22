"""Service → Host 宿主执行边界数据类型。

定义 Service 层提交宿主执行时使用的稳定数据契约：
- ``HostedRunSpec``：描述一次宿主执行的 run 规格。
- ``HostedRunContext``：宿主执行传递给业务 handler 的上下文。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dayu.contracts.cancellation import CancellationToken
from dayu.contracts.execution_metadata import (
    ExecutionDeliveryContext,
    empty_execution_delivery_context,
    normalize_execution_delivery_context,
)


@dataclass(frozen=True)
class HostedRunSpec:
    """宿主执行所需的 run 描述。

    Attributes:
        operation_name: 操作名称，用于 run registry 标识。
        session_id: 关联的 Host session ID。
        scene_name: 关联的 scene 名称。
        metadata: 结构化交付元数据。
        business_concurrency_lane: 业务并发通道名称；``llm_api`` 由 Host 根据
            调用路径自动叠加，Service 禁止在此字段写入 Host 自治 lane 名。
        timeout_ms: 超时毫秒数。
        publish_events: 是否发布事件到 event bus。
        error_summary_limit: 错误摘要字符上限。
    """

    operation_name: str
    session_id: str | None = None
    scene_name: str | None = None
    metadata: ExecutionDeliveryContext = field(default_factory=empty_execution_delivery_context)
    business_concurrency_lane: str | None = None
    timeout_ms: int | None = None
    publish_events: bool = True
    error_summary_limit: int = 500

    def __post_init__(self) -> None:
        """规范化交付元数据。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        object.__setattr__(self, "metadata", normalize_execution_delivery_context(self.metadata))


@dataclass(frozen=True)
class HostedRunContext:
    """宿主执行传递给业务 handler 的上下文。

    Attributes:
        run_id: 当前 Host run ID。
        cancellation_token: 取消令牌。
    """

    run_id: str
    cancellation_token: CancellationToken


__all__ = [
    "HostedRunContext",
    "HostedRunSpec",
]
