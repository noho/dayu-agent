"""Host 仓储写入前的 session 活性屏障辅助函数。

该模块集中承载 pending turn / reply outbox 等 Host 内部真源仓储共用的
session 活性校验逻辑，避免重复实现同一套"session 已关闭则拒绝写入"的
屏障分支。

屏障识别遵循 ``#117`` 共享设计 §3.3：仅 ``ACTIVE`` 状态允许写入；
``CLEARING`` / ``CLEARING_FAILED`` / ``CLOSED`` 任一状态都拒绝写入，
并按 session 当前状态抛出对应异常以便观测面区分原因。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dayu.contracts.session import SessionState
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
        SessionClearingError: session 处于 ``CLEARING`` 临时屏障时抛出。
        SessionClearingFailedError: session 处于 ``CLEARING_FAILED`` 持久锁定屏障时抛出。
    """

    if session_activity is None:
        return

    state = session_activity.get_session_state(session_id)
    if state == SessionState.ACTIVE:
        return

    # 延迟 import 避免 Host 私有模块与 protocols 形成包级循环。
    from dayu.host.protocols import (
        SessionClearingError,
        SessionClearingFailedError,
        SessionClosedError,
    )

    if state == SessionState.CLEARING:
        Log.verbose(
            f"session 处于 CLEARING 屏障，拒绝 {target_name} 写入: "
            f"session_id={session_id}, operation={operation}",
            module=module,
        )
        raise SessionClearingError(session_id)
    if state == SessionState.CLEARING_FAILED:
        Log.verbose(
            f"session 处于 CLEARING_FAILED 锁定，拒绝 {target_name} 写入: "
            f"session_id={session_id}, operation={operation}",
            module=module,
        )
        raise SessionClearingFailedError(session_id)

    # state is None（session 不存在）或 SessionState.CLOSED：统一按"已关闭"
    # 语义拒绝写入，保持与 SessionClosedError 文案一致。
    Log.verbose(
        f"session 已关闭或不存在，拒绝 {target_name} 写入: "
        f"session_id={session_id}, operation={operation}",
        module=module,
    )
    raise SessionClosedError(session_id)


__all__ = ["ensure_session_active"]
