"""跨进程 fence token / lease 公共原语。

为 Host 内部 record 级强一致 CAS 提供最小公共原语。

设计取向（与上层调用方约定）：

- 不抽象 LeaseManager 类，不抽 SQL 模板：reply outbox 与 pending conversation turn
  的状态机不同，硬抽公共层只会制造胶水；本模块只共享异常类型与 lease_id 生成器。
- ``LeaseExpiredError`` 在写路径双条件 CAS（``state + lease_id``）mismatch 时由 store
  抛出，调用方据此明确感知"持有 lease 已被 cleanup 抢占改写"。
- ``generate_lease_id`` 一律返回 uuid4 hex 字符串；即便后续切换实现也保持等长不可
  预测的语义不变。
"""

from __future__ import annotations

import uuid


class LeaseExpiredError(RuntimeError):
    """持有的 lease 已失效。

    在 store 层执行 ``state + lease_id`` 双条件 CAS 时，若 lease_id 不再匹配（典型
    场景：cleanup 抢占已分配新 lease；旧持有者迟到回写），抛出该异常。
    """

    def __init__(
        self,
        message: str,
        *,
        record_id: str,
        lease_id: str | None,
    ) -> None:
        """初始化异常。

        Args:
            message: 错误描述文本。
            record_id: 触发该错误的真源 record 标识，便于日志定位。
            lease_id: 调用方传入的过期 lease_id；为 ``None`` 时表示调用方完全未带 lease。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__(message)
        self.record_id = record_id
        self.lease_id = lease_id


def generate_lease_id() -> str:
    """生成新的 lease_id。

    Args:
        无。

    Returns:
        uuid4 hex 字符串，长度 32，全小写。

    Raises:
        无。
    """

    return uuid.uuid4().hex


__all__ = [
    "LeaseExpiredError",
    "generate_lease_id",
]
