"""单轮 Prompt 服务实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Callable

from dayu.contracts.events import AppEvent
from dayu.contracts.session import SessionSource
from dayu.host.protocols import HostedExecutionGatewayProtocol
from dayu.services.concurrency_lanes import resolve_contract_concurrency_lane
from dayu.services.contract_preparation import prepare_execution_contract
from dayu.services.contracts import PromptRequest, PromptSubmission
from dayu.services.internal.session_coordinator import ServiceSessionCoordinator
from dayu.services.prompt_contributions import (
    build_base_user_contribution,
    build_optional_fins_subject_contribution,
)
from dayu.services.protocols import PromptServiceProtocol
from dayu.services.scene_execution_acceptance import SceneExecutionAcceptancePreparer


@dataclass
class PromptService(PromptServiceProtocol):
    """单轮 Prompt 服务。"""

    host: HostedExecutionGatewayProtocol
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer
    company_name_resolver: Callable[[str], str] | None = None
    session_source: SessionSource = SessionSource.API

    async def submit(self, request: PromptRequest) -> PromptSubmission:
        """提交单轮 Prompt 请求并返回事件流句柄。

        Args:
            request: Prompt 请求。

        Returns:
            Prompt 提交句柄。

        Raises:
            ValueError: 输入为空时抛出。
            KeyError: 显式续聊但 session 不存在时抛出。
        """

        session = self._resolve_session(request)
        return PromptSubmission(
            session_id=session.session_id,
            event_stream=self._stream_in_session(request=request, session_id=session.session_id),
        )

    def _resolve_session(self, request: PromptRequest):
        """解析 Prompt 请求对应的 Host session。"""

        user_message = str(request.user_text or "").strip()
        if not user_message:
            raise ValueError("Prompt 输入不能为空")

        coordinator = ServiceSessionCoordinator(
            host=self.host,
            session_source=self.session_source,
        )
        return coordinator.resolve(
            session_id=request.session_id,
            scene_name="prompt",
            policy=request.session_resolution_policy,
        )

    async def _stream_in_session(
        self,
        *,
        request: PromptRequest,
        session_id: str,
    ) -> AsyncIterator[AppEvent]:
        """在给定 session 中执行单轮 Prompt。"""

        user_message = str(request.user_text or "").strip()
        if not user_message:
            raise ValueError("Prompt 输入不能为空")

        accepted_scene = self.scene_execution_acceptance_preparer.prepare("prompt", request.execution_options)
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
            service_name="prompt",
            scene_name="prompt",
            accepted_execution_spec=accepted_scene.accepted_execution_spec,
            prompt_contributions=prompt_contributions,
            context_slots=accepted_scene.scene_definition.context_slots,
            selected_toolsets=(),
            user_message=user_message,
            session_key=session_id,
            business_concurrency_lane=resolve_contract_concurrency_lane("prompt"),
            execution_options=request.execution_options,
            timeout_ms=None,
            resumable=accepted_scene.default_resumable,
        )
        async for event in self.host.run_agent_stream(execution_contract):
            yield event


__all__ = ["PromptService"]
