"""pending turn 恢复辅助函数。"""

from __future__ import annotations

from dayu.services.protocols import ChatServiceProtocol


def has_resumable_pending_turn(
    chat_service: ChatServiceProtocol,
    *,
    session_id: str,
    scene_name: str,
    pending_turn_id: str,
) -> bool:
    """判断指定 pending turn 是否仍然存在于 Service 可恢复视图中。

    Args:
        chat_service: ChatService 稳定协议实现。
        session_id: 所属会话 ID。
        scene_name: 所属 scene 名称。
        pending_turn_id: 目标 pending turn ID。

    Returns:
        若该 pending turn 仍可在 Service 公开视图中观察到，则返回 `True`；
        否则返回 `False`。

    Raises:
        无。
    """

    pending_turns = chat_service.list_resumable_pending_turns(
        session_id=session_id,
        scene_name=scene_name,
    )
    return any(view.pending_turn_id == pending_turn_id for view in pending_turns)


__all__ = ["has_resumable_pending_turn"]
