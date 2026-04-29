"""聊天服务实现。"""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable

from dayu.contracts.agent_execution import ExecutionContract
from dayu.contracts.events import AppEvent
from dayu.contracts.session import SessionSource
from dayu.host.protocols import ConversationalExecutionGatewayProtocol, PendingTurnSummary
from dayu.contracts.execution_metadata import normalize_execution_delivery_context
from dayu.prompting.scene_definition import PromptManifestError
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
    default_scene_name: str
    company_name_resolver: Callable[[str], str] | None = None
    session_source: SessionSource = SessionSource.API
    # 以弱引用持有每 session 的串行锁：只要存在至少一个生成器/任务引用该锁对象，
    # 锁就保留在表中；所有引用者结束后，对应条目由 GC 自动清理，避免长驻进程中
    # 随 session 数单调增长。
    _session_locks: "weakref.WeakValueDictionary[str, asyncio.Lock]" = field(
        default_factory=weakref.WeakValueDictionary
    )

    def _lock_for_session(self, session_id: str) -> asyncio.Lock:
        """获取或创建指定 session 的串行锁。

        Args:
            session_id: 目标 session ID。

        Returns:
            对应 session 的 ``asyncio.Lock``；调用方需保持强引用直到使用完毕，
            否则条目可能被 GC 清理（符合"无活跃使用即释放"的语义）。

        Raises:
            无。
        """

        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

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
        # 同 session 并发提交按 FIFO 串行，避免在 Host 层因 revision 乐观锁
        # 冲突而静默丢弃后到达的请求。
        # 锁在事件流首次迭代时才 acquire，避免调用方拿到 submission 后放弃消费
        # 导致锁永久不释放（隐式泄漏）。
        session_lock = self._lock_for_session(session.session_id)
        return ChatTurnSubmission(
            session_id=session.session_id,
            event_stream=self._stream_turn_in_session_with_lock(
                request=request,
                session_id=session.session_id,
                prepared_context=prepared_context,
                session_lock=session_lock,
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

    def cleanup_stale_pending_turns(
        self,
        *,
        session_id: str | None = None,
    ) -> list[str]:
        """清理关联 run 已终态、且按调和规则应删除的 pending turn。

        Args:
            session_id: 若提供，仅扫描该 session 下的 pending turn；
                为 ``None`` 时全量扫描。

        Returns:
            被清理的 pending_turn_id 列表。

        Raises:
            无。
        """

        return self.host.cleanup_stale_pending_turns(session_id=session_id)

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

        # scene_name 语义对齐：契约中 None 表示"未指定"，应走装配期注入的默认；
        # 显式传入空串 / 装配期注入空白串均属契约违规，必须立刻抛出明确错误，
        # 不能被静默 strip 掉后退化成下游 "scene 不存在" 这类误导性诊断。
        raw_scene_name = request.scene_name
        if raw_scene_name is None:
            stripped_default = self.default_scene_name.strip()
            if not stripped_default:
                raise ValueError(
                    "ChatService.default_scene_name 装配期未注入有效值；"
                    "请由装配点显式传入非空 scene_name"
                )
            scene_name = stripped_default
        else:
            stripped_scene_name = str(raw_scene_name).strip()
            if not stripped_scene_name:
                raise ValueError("scene_name 不能为空字符串；如需默认 scene 请传 None")
            scene_name = stripped_scene_name
        user_message = str(request.user_text or "").strip()
        if not user_message:
            raise ValueError("聊天输入不能为空")
        try:
            accepted_scene = self.scene_execution_acceptance_preparer.prepare(scene_name, request.execution_options)
        except FileNotFoundError as exc:
            raise ValueError(f"scene 不存在: {scene_name}") from exc
        except PromptManifestError as exc:
            raise ValueError(f"scene manifest 配置损坏: scene={scene_name}, 详情={exc}") from exc
        return _PreparedChatTurnContext(
            scene_name=scene_name,
            user_message=user_message,
            accepted_scene=accepted_scene,
        )

    async def _stream_turn_in_session_with_lock(
        self,
        *,
        request: ChatTurnRequest,
        session_id: str,
        prepared_context: _PreparedChatTurnContext,
        session_lock: asyncio.Lock,
    ) -> AsyncIterator[AppEvent]:
        """在持有 session 锁的前提下产出事件流，流结束后释放锁。

        Args:
            request: 聊天单轮请求。
            session_id: 已 resolve 的 session ID。
            prepared_context: 已经过校验的请求级准备结果。
            session_lock: 由 ``submit_turn`` 准备好的 session 锁；在首次迭代时 acquire。

        Returns:
            事件流异步迭代器。

        Raises:
            无：异常会沿事件流向上抛出，``finally`` 仍会释放锁。
        """

        await session_lock.acquire()
        try:
            async for event in self._stream_turn_in_session(
                request=request,
                session_id=session_id,
                prepared_context=prepared_context,
            ):
                yield event
        finally:
            session_lock.release()

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
