"""宿主管理服务。

该服务把 Host 的管理能力收口为 Service-owned 契约，
供 Web / CLI 管理面调用，避免 UI 请求期直接触碰 Host。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator

from dayu.contracts.events import AppEvent, PublishedRunEventProtocol
from dayu.contracts.run import RunRecord, RunState
from dayu.contracts.session import SessionRecord, SessionSource, SessionState
from dayu.host.protocols import EventSubscription, HostAdminOperationsProtocol
from dayu.services.contracts import (
    HostCleanupResult,
    HostStatusView,
    LaneStatusView,
    RunAdminView,
    SessionAdminView,
    SessionTurnExcerptView,
)
from dayu.services.protocols import HostAdminServiceProtocol


def _format_optional_datetime(value: datetime | None) -> str:
    """把可选时间格式化为管理视图使用的字符串。

    Args:
        value: 原始时间对象。

    Returns:
        存在时返回 ISO 8601 字符串，否则返回空字符串。

    Raises:
        无。
    """

    if value is None:
        return ""
    return value.isoformat()


def _parse_session_state(state: str | None) -> SessionState | None:
    """解析 session 状态字符串。

    Args:
        state: 原始状态字符串。

    Returns:
        解析后的 `SessionState`，未传时返回 `None`。

    Raises:
        ValueError: 状态值非法时抛出。
    """

    if state is None:
        return None
    return SessionState(str(state).strip().lower())


def _parse_run_state(state: str | None) -> RunState | None:
    """解析 run 状态字符串。

    Args:
        state: 原始状态字符串。

    Returns:
        解析后的 `RunState`，未传时返回 `None`。

    Raises:
        ValueError: 状态值非法时抛出。
    """

    if state is None:
        return None
    return RunState(str(state).strip().lower())


def _parse_session_source(source: str) -> SessionSource:
    """解析会话来源字符串。

    Args:
        source: 原始来源字符串。

    Returns:
        解析后的 `SessionSource`。

    Raises:
        ValueError: 来源值非法时抛出。
    """

    return SessionSource(str(source).strip().lower())


def _parse_optional_session_source(source: str | None) -> SessionSource | None:
    """解析可选会话来源字符串。

    Args:
        source: 原始来源字符串。

    Returns:
        解析后的 `SessionSource`；未传或空白时返回 `None`。

    Raises:
        ValueError: 来源值非法时抛出。
    """

    if source is None:
        return None
    normalized = str(source).strip()
    if not normalized:
        return None
    return SessionSource(normalized.lower())


def _normalize_scene_name(scene: str | None) -> str | None:
    """规范化可选 scene 过滤条件。

    Args:
        scene: 原始 scene 文本。

    Returns:
        规范化后的 scene 文本；未传或空白时返回 `None`。

    Raises:
        无。
    """

    if scene is None:
        return None
    normalized = str(scene).strip()
    if not normalized:
        return None
    return normalized


def _to_session_view(
    record: SessionRecord,
    *,
    turn_count: int,
    first_question_preview: str,
    last_question_preview: str,
) -> SessionAdminView:
    """把 Host 会话记录与 conversation 摘要转换为管理视图。

    Args:
        record: Host 会话记录。
        turn_count: 已持久化的 conversation turn 数量。
        first_question_preview: 第一轮问题预览。
        last_question_preview: 最后一轮问题预览。
    Returns:
        管理面会话视图。

    Raises:
        无。
    """

    created_at = record.created_at
    last_activity_at = record.last_activity_at
    return SessionAdminView(
        session_id=record.session_id,
        source=record.source.value,
        state=record.state.value,
        scene_name=record.scene_name,
        created_at=_format_optional_datetime(created_at),
        last_activity_at=_format_optional_datetime(last_activity_at),
        turn_count=turn_count,
        first_question_preview=first_question_preview,
        last_question_preview=last_question_preview,
    )


def _to_session_turn_excerpt_view(
    *,
    user_text: str,
    assistant_text: str,
    created_at: str,
) -> SessionTurnExcerptView:
    """把 Host conversation 单轮摘录转换为管理视图。

    Args:
        user_text: 用户输入文本。
        assistant_text: 助手最终回答文本。
        created_at: 该轮创建时间。

    Returns:
        通用单轮对话摘录视图。

    Raises:
        无。
    """

    return SessionTurnExcerptView(
        user_text=user_text,
        assistant_text=assistant_text,
        created_at=created_at,
    )


def _to_run_view(record: RunRecord) -> RunAdminView:
    """把 Host 运行记录转换为管理视图。

    Args:
        record: Host 运行记录。

    Returns:
        管理面运行视图。

    Raises:
        无。
    """

    created_at = record.created_at
    started_at = record.started_at
    completed_at = record.completed_at
    cancel_requested_at = record.cancel_requested_at
    cancel_requested_reason = record.cancel_requested_reason
    cancel_reason = record.cancel_reason
    return RunAdminView(
        run_id=record.run_id,
        session_id=record.session_id,
        service_type=record.service_type,
        state=record.state.value,
        cancel_requested_at=(
            cancel_requested_at.isoformat()
            if cancel_requested_at is not None
            else None
        ),
        cancel_requested_reason=(
            cancel_requested_reason.value
            if cancel_requested_reason is not None
            else None
        ),
        cancel_reason=(
            cancel_reason.value
            if cancel_reason is not None
            else None
        ),
        scene_name=record.scene_name,
        created_at=created_at.isoformat() if created_at is not None else str(created_at),
        started_at=started_at.isoformat() if started_at is not None else None,
        finished_at=completed_at.isoformat() if completed_at is not None else None,
        error_summary=record.error_summary,
    )


async def _stream_subscription_events(subscription: EventSubscription) -> AsyncIterator[PublishedRunEventProtocol]:
    """把 Host 事件订阅句柄包装成稳定的异步事件流。

    Args:
        subscription: Host 事件订阅句柄。

    Yields:
        稳定运行事件包络。

    Raises:
        无。
    """

    try:
        async for event in subscription:
            yield event
    finally:
        subscription.close()


@dataclass
class HostAdminService(HostAdminServiceProtocol):
    """宿主管理服务实现。"""

    host: HostAdminOperationsProtocol

    def _build_session_view(self, record: SessionRecord) -> SessionAdminView:
        """构造带 digest 的会话管理视图。

        Args:
            record: Host 会话记录。

        Returns:
            带 conversation digest 字段的会话视图。

        Raises:
            无。
        """

        digest = self.host.get_conversation_session_digest(record.session_id)
        return _to_session_view(
            record,
            turn_count=digest.turn_count,
            first_question_preview=digest.first_question_preview,
            last_question_preview=digest.last_question_preview,
        )

    def create_session(self, *, source: str = "web", scene_name: str | None = None) -> SessionAdminView:
        """创建宿主会话。

        Args:
            source: 会话来源字符串。
            scene_name: 可选 scene 名称。

        Returns:
            新创建的会话视图。

        Raises:
            ValueError: 来源值非法时抛出。
        """

        record = self.host.create_session(
            _parse_session_source(source),
            scene_name=scene_name,
        )
        return self._build_session_view(record)

    def list_sessions(
        self,
        *,
        state: str | None = None,
        source: str | None = None,
        scene: str | None = None,
    ) -> list[SessionAdminView]:
        """列出宿主会话摘要。

        Args:
            state: 可选状态过滤。
            source: 可选来源过滤。
            scene: 可选 scene 过滤。

        Returns:
            匹配的会话视图列表。

        Raises:
            ValueError: 状态值或来源值非法时抛出。
        """

        parsed_state = _parse_session_state(state)
        parsed_source = _parse_optional_session_source(source)
        normalized_scene = _normalize_scene_name(scene)
        records = self.host.list_sessions(state=parsed_state)
        matched_records = [
            record
            for record in records
            if (parsed_source is None or record.source == parsed_source)
            and (normalized_scene is None or record.scene_name == normalized_scene)
        ]
        return [self._build_session_view(record) for record in matched_records]

    def list_session_recent_turns(
        self,
        session_id: str,
        *,
        limit: int = 1,
    ) -> list[SessionTurnExcerptView]:
        """列出指定会话最近对话轮次。

        Args:
            session_id: 会话 ID。
            limit: 最多返回的轮次数量。

        Returns:
            最近对话轮次，按时间从旧到新排列；会话不存在时返回空列表。

        Raises:
            无。
        """

        record = self.host.get_session(session_id)
        if record is None:
            return []
        excerpts = self.host.list_conversation_session_turn_excerpts(session_id, limit=limit)
        return [
            _to_session_turn_excerpt_view(
                user_text=excerpt.user_text,
                assistant_text=excerpt.assistant_text,
                created_at=excerpt.created_at,
            )
            for excerpt in excerpts
        ]

    def get_session(self, session_id: str) -> SessionAdminView | None:
        """获取单个宿主会话。

        Args:
            session_id: 会话 ID。

        Returns:
            会话视图；不存在时返回 `None`。

        Raises:
            无。
        """

        record = self.host.get_session(session_id)
        if record is None:
            return None
        return self._build_session_view(record)

    def close_session(self, session_id: str) -> tuple[SessionAdminView, list[str]]:
        """关闭宿主会话并取消其下活跃运行。

        Args:
            session_id: 会话 ID。

        Returns:
            `(关闭后的会话视图, 被取消的 run_id 列表)`。

        Raises:
            KeyError: 会话不存在时抛出。
        """

        record, cancelled_run_ids = self.host.cancel_session(session_id)
        return self._build_session_view(record), cancelled_run_ids

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        state: str | None = None,
        service_type: str | None = None,
        active_only: bool = False,
    ) -> list[RunAdminView]:
        """列出宿主运行记录。

        Args:
            session_id: 可选会话过滤条件。
            state: 可选状态过滤。
            service_type: 可选服务类型过滤。
            active_only: 是否只返回活跃运行。

        Returns:
            匹配的运行视图列表。

        Raises:
            ValueError: 状态值非法时抛出。
        """

        if active_only:
            records = self.host.list_active_runs()
            if session_id is not None:
                records = [record for record in records if record.session_id == session_id]
            if service_type is not None:
                records = [record for record in records if record.service_type == service_type]
        else:
            parsed_state = _parse_run_state(state)
            records = self.host.list_runs(
                session_id=session_id,
                state=parsed_state,
                service_type=service_type,
            )
        return [_to_run_view(record) for record in records]

    def get_run(self, run_id: str) -> RunAdminView | None:
        """获取单个运行记录。

        Args:
            run_id: 运行 ID。

        Returns:
            运行视图；不存在时返回 `None`。

        Raises:
            无。
        """

        record = self.host.get_run(run_id)
        if record is None:
            return None
        return _to_run_view(record)

    def cancel_run(self, run_id: str) -> RunAdminView:
        """取消指定运行。

        Args:
            run_id: 运行 ID。

        Returns:
            更新后的运行视图。

        Raises:
            KeyError: 运行不存在时抛出。
        """

        return _to_run_view(self.host.cancel_run(run_id))

    def cancel_session_runs(self, session_id: str) -> list[str]:
        """取消指定会话下的所有活跃运行。

        Args:
            session_id: 会话 ID。

        Returns:
            被成功请求取消的 run_id 列表。

        Raises:
            无。
        """

        return self.host.cancel_session_runs(session_id)

    def cleanup(self) -> HostCleanupResult:
        """执行宿主清理。

        Args:
            无。

        Returns:
            清理结果。

        Raises:
            无。
        """

        orphan_run_ids = tuple(self.host.cleanup_orphan_runs())
        stale_permit_ids = tuple(self.host.cleanup_stale_permits())
        # best-effort：对齐 run / permit 的启动恢复，避免 IN_PROGRESS 回退后
        # 整条 reply outbox 流水线卡死。失败在 Host 层已被 Log.warn 吞掉。
        self.host.cleanup_stale_reply_outbox_deliveries()
        stale_pending_turn_ids = tuple(self.host.cleanup_stale_pending_turns())
        return HostCleanupResult(
            orphan_run_ids=orphan_run_ids,
            stale_permit_ids=stale_permit_ids,
            stale_pending_turn_ids=stale_pending_turn_ids,
        )

    def get_status(self) -> HostStatusView:
        """获取宿主状态快照。

        Args:
            无。

        Returns:
            宿主状态视图。

        Raises:
            无。
        """

        sessions = self.host.list_sessions()
        active_runs = self.host.list_active_runs()
        active_runs_by_type = dict(Counter(record.service_type for record in active_runs))
        lane_statuses: dict[str, LaneStatusView] = {}
        for lane_name, snapshot in self.host.get_all_lane_statuses().items():
            lane_statuses[lane_name] = LaneStatusView(
                lane=lane_name,
                active=snapshot.active,
                max_concurrent=snapshot.max_concurrent,
            )
        return HostStatusView(
            active_session_count=sum(1 for session in sessions if session.state == SessionState.ACTIVE),
            total_session_count=len(sessions),
            active_run_count=len(active_runs),
            active_runs_by_type=active_runs_by_type,
            lane_statuses=lane_statuses,
        )

    def subscribe_run_events(self, run_id: str) -> AsyncIterator[PublishedRunEventProtocol]:
        """订阅单个运行的事件流。

        Args:
            run_id: 运行 ID。

        Yields:
            稳定运行事件包络。

        Raises:
            RuntimeError: 未启用事件总线时抛出。
        """

        subscription = self.host.subscribe_run_events(run_id)
        return _stream_subscription_events(subscription)

    def subscribe_session_events(self, session_id: str) -> AsyncIterator[PublishedRunEventProtocol]:
        """订阅单个会话下所有运行的事件流。

        Args:
            session_id: 会话 ID。

        Yields:
            稳定运行事件包络。

        Raises:
            RuntimeError: 未启用事件总线时抛出。
        """

        subscription = self.host.subscribe_session_events(session_id)
        return _stream_subscription_events(subscription)


__all__ = ["HostAdminService"]
