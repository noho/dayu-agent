"""Host 仓储写入前的 session 活性屏障辅助函数。

该模块集中承载 pending turn / reply outbox 等 Host 内部真源仓储共用的
session 活性校验逻辑，避免重复实现同一套“session 已关闭则拒绝写入”的
屏障分支。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dayu.log import Log


if TYPE_CHECKING:
    from dayu.host.protocols import SessionActivityQueryProtocol


def ensure_session_active(
    session_activity: "SessionActivityQueryProtocol | None",
    *,
    session_id: str,
    operation: str,
    module: str,
    target_name: str,
) -> None:
    """在 Host 仓储写入前校验 session 活性。

    Args:
        session_activity: 可选的 session 活性查询协议实现；``None`` 表示未装配屏障。
        session_id: 目标 session ID（调用方应先完成规范化）。
        operation: 触发屏障的写入操作名，仅用于诊断日志。
        module: 当前调用模块名，用于日志归属。
        target_name: 被拒绝写入的真源名称，如 ``pending turn`` / ``reply outbox``。

    Returns:
        无。

    Raises:
        SessionClosedError: session 不存在或已 ``CLOSED`` 时抛出。
    """

    if session_activity is None:
        return
    if session_activity.is_session_active(session_id):
        return

    # 延迟 import 避免 Host 私有模块与 protocols 形成包级循环。
    from dayu.host.protocols import SessionClosedError

    Log.verbose(
        f"session 已关闭或不存在，拒绝 {target_name} 写入: "
        f"session_id={session_id}, operation={operation}",
        module=module,
    )
    raise SessionClosedError(session_id)


__all__ = ["ensure_session_active"]
