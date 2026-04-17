"""Host 层会话数据清理工具。

提供按 session_id 批量清理 pending turns 和 reply outbox 的能力，
供 UI 层在 service uninstall 时调用。
"""

from __future__ import annotations

from pathlib import Path

from dayu.log import Log

MODULE = "host.cleanup"


def purge_sessions_from_host_db(
    *,
    host_db_path: Path,
    session_ids: list[str],
) -> tuple[int, int]:
    """按 session_id 列表清理 Host DB 中的 pending turns 和 reply outbox。

    Args:
        host_db_path: Host DB 文件路径。
        session_ids: 待清理的 session_id 列表。

    Returns:
        (已删 pending turns 数, 已删 reply outbox 数)。

    Raises:
        无。内部异常会被捕获并记录日志。
    """

    if not session_ids or not host_db_path.exists():
        return (0, 0)

    from dayu.host.host_store import HostStore
    from dayu.host.pending_turn_store import SQLitePendingConversationTurnStore
    from dayu.host.reply_outbox_store import SQLiteReplyOutboxStore

    try:
        host_store = HostStore(host_db_path)
        host_store.initialize_schema()
        pending_store = SQLitePendingConversationTurnStore(host_store)
        outbox_store = SQLiteReplyOutboxStore(host_store)
        total_pending = 0
        total_outbox = 0
        for sid in session_ids:
            total_pending += pending_store.delete_by_session_id(sid)
            total_outbox += outbox_store.delete_by_session_id(sid)
        host_store.close()
        return (total_pending, total_outbox)
    except Exception as exc:
        Log.warning(f"清理 Host DB 数据时出错: {exc}", module=MODULE)
        return (0, 0)
