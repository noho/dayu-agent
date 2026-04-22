"""服务层协议定义。"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from dayu.contracts.events import AppEvent, PublishedRunEventProtocol
from dayu.services.contracts import (
    ChatPendingTurnView,
    ChatResumeRequest,
    ChatTurnRequest,
    ChatTurnSubmission,
    FinsSubmitRequest,
    FinsSubmission,
    HostCleanupResult,
    HostStatusView,
    PromptRequest,
    PromptSubmission,
    ReplyDeliveryFailureRequest,
    ReplyDeliverySubmitRequest,
    ReplyDeliveryView,
    RunAdminView,
    SessionAdminView,
    SessionTurnExcerptView,
    WriteRequest,
)


@runtime_checkable
class BaseServiceProtocol(Protocol):
    """服务基础协议。"""


@runtime_checkable
class ChatServiceProtocol(BaseServiceProtocol, Protocol):
    """聊天服务协议。"""

    async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
        """提交聊天单轮请求并返回会话句柄。

        Args:
            request: 聊天单轮请求。

        Returns:
            包含 `session_id` 与事件流句柄的提交结果。
        """
        ...

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        """恢复指定 pending conversation turn 并返回事件流句柄。"""
        ...

    def list_resumable_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
    ) -> list[ChatPendingTurnView]:
        """列出可恢复的 pending conversation turn。"""
        ...


@runtime_checkable
class PromptServiceProtocol(BaseServiceProtocol, Protocol):
    """单轮 Prompt 服务协议。"""

    async def submit(self, request: PromptRequest) -> PromptSubmission:
        """提交单轮 Prompt 请求并返回会话句柄。

        Args:
            request: Prompt 请求。

        Returns:
            包含 `session_id` 与事件流句柄的提交结果。
        """
        ...


@runtime_checkable
class WriteServiceProtocol(BaseServiceProtocol, Protocol):
    """写作服务协议。"""

    def run(self, request: WriteRequest) -> int:
        """执行写作流程。"""
        ...

    @staticmethod
    def print_report(output_dir: str) -> int:
        """打印写作报告。"""
        ...


@runtime_checkable
class FinsServiceProtocol(BaseServiceProtocol, Protocol):
    """财报服务协议。"""

    def submit(self, request: FinsSubmitRequest) -> FinsSubmission:
        """提交财报命令并返回执行句柄。

        Args:
            request: 财报服务提交请求。

        Returns:
            包含 `session_id` 与执行句柄的提交结果。
        """
        ...


@runtime_checkable
class HostAdminServiceProtocol(BaseServiceProtocol, Protocol):
    """宿主管理服务协议。"""

    def create_session(self, *, source: str = "web", scene_name: str | None = None) -> SessionAdminView:
        """创建宿主会话。"""
        ...

    def list_sessions(
        self,
        *,
        state: str | None = None,
        source: str | None = None,
        scene: str | None = None,
    ) -> list[SessionAdminView]:
        """列出宿主会话摘要视图。"""
        ...

    def list_session_recent_turns(
        self,
        session_id: str,
        *,
        limit: int = 1,
    ) -> list[SessionTurnExcerptView]:
        """列出指定会话最近对话轮次。"""
        ...

    def get_session(self, session_id: str) -> SessionAdminView | None:
        """获取单个宿主会话。"""
        ...

    def close_session(self, session_id: str) -> tuple[SessionAdminView, list[str]]:
        """关闭宿主会话并取消其下活跃运行。"""
        ...

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        state: str | None = None,
        service_type: str | None = None,
        active_only: bool = False,
    ) -> list[RunAdminView]:
        """列出宿主运行记录。"""
        ...

    def get_run(self, run_id: str) -> RunAdminView | None:
        """获取单个运行记录。"""
        ...

    def cancel_run(self, run_id: str) -> RunAdminView:
        """取消指定运行。"""
        ...

    def cancel_session_runs(self, session_id: str) -> list[str]:
        """取消指定会话下的所有活跃运行。"""
        ...

    def cleanup(self) -> HostCleanupResult:
        """执行宿主清理。"""
        ...

    def get_status(self) -> HostStatusView:
        """获取宿主状态快照。"""
        ...

    def subscribe_run_events(self, run_id: str) -> AsyncIterator[PublishedRunEventProtocol]:
        """订阅单个运行的事件流。"""
        ...

    def subscribe_session_events(self, session_id: str) -> AsyncIterator[PublishedRunEventProtocol]:
        """订阅单个会话下所有运行的事件流。"""
        ...


@runtime_checkable
class ReplyDeliveryServiceProtocol(BaseServiceProtocol, Protocol):
    """渠道层使用的 reply delivery 服务协议。"""

    def submit_reply_for_delivery(self, request: ReplyDeliverySubmitRequest) -> ReplyDeliveryView:
        """显式提交待交付回复。"""
        ...

    def get_delivery(self, delivery_id: str) -> ReplyDeliveryView | None:
        """按 ID 查询交付记录。"""
        ...

    def list_deliveries(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: str | None = None,
    ) -> list[ReplyDeliveryView]:
        """列出交付记录。"""
        ...

    def claim_delivery(self, delivery_id: str) -> ReplyDeliveryView:
        """把交付记录推进到发送中状态。"""
        ...

    def mark_delivery_delivered(self, delivery_id: str) -> ReplyDeliveryView:
        """标记交付完成。"""
        ...

    def mark_delivery_failed(self, request: ReplyDeliveryFailureRequest) -> ReplyDeliveryView:
        """标记交付失败。"""
        ...


__all__ = [
    "BaseServiceProtocol",
    "ChatServiceProtocol",
    "FinsServiceProtocol",
    "HostAdminServiceProtocol",
    "PromptServiceProtocol",
    "ReplyDeliveryServiceProtocol",
    "WriteServiceProtocol",
]
