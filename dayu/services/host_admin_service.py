"""宿主管理服务。

该服务把 Host 的管理能力收口为 Service-owned 契约，
供 Web / CLI 管理面调用，避免 UI 请求期直接触碰 Host。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
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
)
from dayu.services.protocols import HostAdminServiceProtocol


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


def _to_session_view(record: SessionRecord) -> SessionAdminView:
    """把 Host 会话记录转换为管理视图。

    Args:
        record: Host 会话记录。

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
        created_at=created_at.isoformat() if created_at is not None else str(created_at),
        last_activity_at=(
            last_activity_at.isoformat()
            if last_activity_at is not None
            else str(last_activity_at)
        ),
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
        return _to_session_view(record)

    def list_sessions(self, *, state: str | None = None) -> list[SessionAdminView]:
        """列出宿主会话。

        Args:
            state: 可选状态过滤。

        Returns:
            匹配的会话视图列表。

        Raises:
            ValueError: 状态值非法时抛出。
        """

        parsed_state = _parse_session_state(state)
        return [_to_session_view(record) for record in self.host.list_sessions(state=parsed_state)]

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
        return _to_session_view(record)

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
        return _to_session_view(record), cancelled_run_ids

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
        return HostCleanupResult(
            orphan_run_ids=orphan_run_ids,
            stale_permit_ids=stale_permit_ids,
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

        active_sessions = self.host.list_sessions(state=SessionState.ACTIVE)
        total_sessions = self.host.list_sessions()
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
            active_session_count=len(active_sessions),
            total_session_count=len(total_sessions),
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
