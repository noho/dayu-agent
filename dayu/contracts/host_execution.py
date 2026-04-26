"""Service → Host 宿主执行边界数据类型。

定义 Service 层提交宿主执行时使用的稳定数据契约：
- ``ConcurrencyAcquirePolicy``：声明 Host 并发许可的等待策略。
- ``HostedRunSpec``：描述一次宿主执行的 run 规格。
- ``HostedRunContext``：宿主执行传递给业务 handler 的上下文。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dayu.contracts.cancellation import CancellationToken
from dayu.contracts.execution_metadata import (
    ExecutionDeliveryContext,
    empty_execution_delivery_context,
    normalize_execution_delivery_context,
)

_CONCURRENCY_ACQUIRE_MODE_HOST_DEFAULT: Literal["host_default"] = "host_default"
_CONCURRENCY_ACQUIRE_MODE_TIMEOUT: Literal["timeout"] = "timeout"
_CONCURRENCY_ACQUIRE_MODE_UNBOUNDED: Literal["unbounded"] = "unbounded"


@dataclass(frozen=True)
class ConcurrencyAcquirePolicy:
    """Host 并发 permit 的等待策略。

    Attributes:
        mode: 等待模式。
            - ``"host_default"``：由 Host 按自身默认治理规则决定。
            - ``"timeout"``：调用方显式要求有限等待。
            - ``"unbounded"``：调用方显式要求无限等待。
        timeout_seconds: 当 ``mode="timeout"`` 时生效的最大等待秒数。
    """

    mode: Literal["host_default", "timeout", "unbounded"] = (
        _CONCURRENCY_ACQUIRE_MODE_HOST_DEFAULT
    )
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        """校验并规范化等待策略。

        Args:
            无。

        Returns:
            无。

        Raises:
            ValueError: 模式非法，或 ``timeout_seconds`` 与模式不匹配时抛出。
        """

        if self.mode == _CONCURRENCY_ACQUIRE_MODE_TIMEOUT:
            raw_timeout_seconds = self.timeout_seconds
            if (
                isinstance(raw_timeout_seconds, bool)
                or not isinstance(raw_timeout_seconds, int | float)
                or raw_timeout_seconds <= 0
            ):
                raise ValueError(
                    "ConcurrencyAcquirePolicy.timeout_seconds 必须是正数"
                )
            object.__setattr__(self, "timeout_seconds", float(raw_timeout_seconds))
            return
        if self.mode not in (
            _CONCURRENCY_ACQUIRE_MODE_HOST_DEFAULT,
            _CONCURRENCY_ACQUIRE_MODE_UNBOUNDED,
        ):
            raise ValueError(f"未知的 ConcurrencyAcquirePolicy.mode: {self.mode}")
        if self.timeout_seconds is not None:
            raise ValueError(
                "ConcurrencyAcquirePolicy 仅在 mode='timeout' 时允许设置 timeout_seconds"
            )

    @classmethod
    def use_host_default(cls) -> "ConcurrencyAcquirePolicy":
        """构造沿用 Host 默认治理规则的等待策略。

        Args:
            无。

        Returns:
            ``mode='host_default'`` 的等待策略。

        Raises:
            无。
        """

        return cls(mode=_CONCURRENCY_ACQUIRE_MODE_HOST_DEFAULT)

    @classmethod
    def with_timeout(cls, timeout_seconds: float) -> "ConcurrencyAcquirePolicy":
        """构造有限等待策略。

        Args:
            timeout_seconds: 最大等待秒数，必须为正数。

        Returns:
            ``mode='timeout'`` 的等待策略。

        Raises:
            ValueError: ``timeout_seconds`` 非正数时抛出。
        """

        return cls(
            mode=_CONCURRENCY_ACQUIRE_MODE_TIMEOUT,
            timeout_seconds=timeout_seconds,
        )

    @classmethod
    def unbounded(cls) -> "ConcurrencyAcquirePolicy":
        """构造无限等待策略。

        Args:
            无。

        Returns:
            ``mode='unbounded'`` 的等待策略。

        Raises:
            无。
        """

        return cls(mode=_CONCURRENCY_ACQUIRE_MODE_UNBOUNDED)


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
        concurrency_acquire_policy: Host 并发 permit 的等待策略。Service 只通过
            contracts 层声明意图，不感知 Host 的具体实现细节。
        timeout_ms: 超时毫秒数。
        publish_events: 是否发布事件到 event bus。
        error_summary_limit: 错误摘要字符上限。
    """

    operation_name: str
    session_id: str | None = None
    scene_name: str | None = None
    metadata: ExecutionDeliveryContext = field(default_factory=empty_execution_delivery_context)
    business_concurrency_lane: str | None = None
    concurrency_acquire_policy: ConcurrencyAcquirePolicy = field(
        default_factory=ConcurrencyAcquirePolicy.use_host_default
    )
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
    "ConcurrencyAcquirePolicy",
    "HostedRunContext",
    "HostedRunSpec",
]
