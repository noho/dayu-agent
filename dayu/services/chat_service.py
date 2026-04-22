"""聊天服务实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Callable

from dayu.contracts.agent_execution import ExecutionContract
from dayu.contracts.events import AppEvent
from dayu.contracts.session import SessionSource
from dayu.host.protocols import ConversationalExecutionGatewayProtocol, PendingTurnSummary
from dayu.contracts.execution_metadata import normalize_execution_delivery_context
from dayu.services.concurrency_lanes import resolve_contract_concurrency_lane
from dayu.services.contract_preparation import prepare_execution_contract
from dayu.services.contracts import (
    ChatPendingTurnView,
    ChatResumeRequest,
    ChatTurnRequest,
    ChatTurnSubmission,
)
from dayu.services.internal.session_coordinator import ServiceSessionCoordinator
from dayu.services.prompt_contributions import (
    build_base_user_contribution,
    build_optional_fins_subject_contribution,
)
from dayu.services.protocols import ChatServiceProtocol
from dayu.services.scene_execution_acceptance import AcceptedSceneExecution, SceneExecutionAcceptancePreparer


@dataclass(frozen=True)
class _PreparedChatTurnContext:
    """聊天单轮提交前已完成的请求级准备结果。"""

    scene_name: str
    user_message: str
    accepted_scene: AcceptedSceneExecution


@dataclass
class ChatService(ChatServiceProtocol):
    """聊天服务。"""

    host: ConversationalExecutionGatewayProtocol
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer
    company_name_resolver: Callable[[str], str] | None = None
    session_source: SessionSource = SessionSource.API

    async def submit_turn(self, request: ChatTurnRequest) -> ChatTurnSubmission:
        """提交聊天单轮并返回事件流句柄。

        Args:
            request: 聊天单轮请求。

        Returns:
            聊天单轮提交句柄。

        Raises:
            ValueError: 用户输入为空时抛出。
            KeyError: 显式续聊但 session 不存在时抛出。
        """

        prepared_context = self._prepare_turn_context(request)
        session = self._session_coordinator().resolve(
            session_id=request.session_id,
            scene_name=prepared_context.scene_name,
            policy=request.session_resolution_policy,
        )
        return ChatTurnSubmission(
            session_id=session.session_id,
            event_stream=self._stream_turn_in_session(
                request=request,
                session_id=session.session_id,
                prepared_context=prepared_context,
            ),
        )

    async def resume_pending_turn(self, request: ChatResumeRequest) -> ChatTurnSubmission:
        """恢复指定 pending conversation turn。

        Args:
            request: 恢复请求。

        Returns:
            聊天单轮提交句柄。

        Raises:
            KeyError: pending turn 不存在时抛出。
            ValueError: pending turn 不可恢复时抛出。
        """

        pending_turn = self.host.get_pending_turn(request.pending_turn_id)
        if pending_turn is None:
            raise KeyError(f"pending conversation turn 不存在: {request.pending_turn_id}")
        session_id = str(request.session_id or "").strip()
        if pending_turn.session_id != session_id:
            raise ValueError(
                "pending conversation turn 不属于当前 session: "
                f"pending_turn_id={request.pending_turn_id}, session_id={session_id}"
            )
        return ChatTurnSubmission(
            session_id=pending_turn.session_id,
            event_stream=self.host.resume_pending_turn_stream(request.pending_turn_id, session_id=session_id),
        )

    def list_resumable_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
    ) -> list[ChatPendingTurnView]:
        """列出可恢复的 pending conversation turn。"""

        return [
            _to_pending_turn_view(record)
            for record in self.host.list_pending_turns(
                session_id=session_id,
                scene_name=scene_name,
                resumable_only=True,
            )
        ]

    def _session_coordinator(self) -> ServiceSessionCoordinator:
        """构造当前服务使用的会话协调器。"""

        return ServiceSessionCoordinator(
            host=self.host,
            session_source=self.session_source,
        )

    def _prepare_turn_context(self, request: ChatTurnRequest) -> _PreparedChatTurnContext:
        """在提交前完成聊天请求级校验与 scene 接受。

        Args:
            request: 聊天单轮请求。

        Returns:
            已规范化用户输入与 accepted scene 的准备结果。

        Raises:
            ValueError: 用户输入为空或 scene 非法时抛出。
        """

        scene_name = str(request.scene_name or "").strip() or "interactive"
        user_message = str(request.user_text or "").strip()
        if not user_message:
            raise ValueError("聊天输入不能为空")
        try:
            accepted_scene = self.scene_execution_acceptance_preparer.prepare(scene_name, request.execution_options)
        except FileNotFoundError as exc:
            raise ValueError(f"scene 不存在: {scene_name}") from exc
        return _PreparedChatTurnContext(
            scene_name=scene_name,
            user_message=user_message,
            accepted_scene=accepted_scene,
        )

    async def _stream_turn_in_session(
        self,
        *,
        request: ChatTurnRequest,
        session_id: str,
        prepared_context: _PreparedChatTurnContext,
    ) -> AsyncIterator[AppEvent]:
        """在给定 session 中执行已完成受理校验的聊天单轮。"""

        prompt_contributions = {
            "base_user": build_base_user_contribution(),
        }
        subject_text = build_optional_fins_subject_contribution(
            ticker=request.ticker,
            company_name_resolver=self.company_name_resolver,
        )
        if subject_text:
            prompt_contributions["fins_default_subject"] = subject_text

        execution_contract = prepare_execution_contract(
            service_name="chat_turn",
            scene_name=prepared_context.scene_name,
            accepted_execution_spec=prepared_context.accepted_scene.accepted_execution_spec,
            prompt_contributions=prompt_contributions,
            context_slots=prepared_context.accepted_scene.scene_definition.context_slots,
            selected_toolsets=(),
            user_message=prepared_context.user_message,
            session_key=session_id,
            business_concurrency_lane=resolve_contract_concurrency_lane(prepared_context.scene_name),
            metadata=request.delivery_context,
            execution_options=request.execution_options,
            timeout_ms=None,
            resumable=prepared_context.accepted_scene.default_resumable,
        )
        async for event in self._stream_execution_contract(execution_contract):
            yield event

    async def _stream_execution_contract(
        self,
        execution_contract: ExecutionContract,
    ) -> AsyncIterator[AppEvent]:
        """执行已准备好的 ExecutionContract。"""

        async for event in self.host.run_agent_stream(execution_contract):
            yield event


def _to_pending_turn_view(record: PendingTurnSummary) -> ChatPendingTurnView:
    """把 Host pending turn 记录转换为 Service 视图。"""

    return ChatPendingTurnView(
        pending_turn_id=record.pending_turn_id,
        session_id=record.session_id,
        scene_name=record.scene_name,
        user_text=record.user_text,
        source_run_id=record.source_run_id,
        resumable=record.resumable,
        state=record.state,
        metadata=normalize_execution_delivery_context(record.metadata),
    )
__all__ = ["ChatService"]
