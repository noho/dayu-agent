"""财报服务实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Callable

from dayu.contracts.fins import FinsCommand, FinsEvent, FinsEventType, FinsResult
from dayu.contracts.session import SessionRecord, SessionSource
from dayu.fins.service_runtime import FinsRuntimeProtocol
from dayu.contracts.host_execution import HostedRunContext, HostedRunSpec
from dayu.host.protocols import HostedExecutionGatewayProtocol
from dayu.services.concurrency_lanes import resolve_hosted_run_concurrency_lane
from dayu.services.contracts import FinsSubmission, FinsSubmitRequest, SessionResolutionPolicy
from dayu.services.internal.session_coordinator import ServiceSessionCoordinator
from dayu.services.protocols import FinsServiceProtocol


def _build_cancel_checker(context: HostedRunContext) -> Callable[[], bool]:
    """基于 HostedRunContext 构造同步执行链的取消检查函数。"""

    return context.cancellation_token.is_cancelled


@dataclass
class FinsService(FinsServiceProtocol):
    """财报服务。"""

    host: HostedExecutionGatewayProtocol
    fins_runtime: FinsRuntimeProtocol
    session_source: SessionSource = SessionSource.API

    def submit(self, request: FinsSubmitRequest) -> FinsSubmission:
        """提交财报命令并返回执行句柄。

        Args:
            request: 财报服务提交请求。

        Returns:
            财报执行句柄。

        Raises:
            KeyError: 显式续聊但 session 不存在时抛出。
            ValueError: 命令参数非法时抛出。
        """

        self.fins_runtime.validate_command(request.command)
        session = self._resolve_session(
            command=request.command,
            policy=request.session_resolution_policy,
        )
        spec = HostedRunSpec(
            operation_name=f"fins_{request.command.name}",
            session_id=session.session_id,
            scene_name=request.command.name,
            business_concurrency_lane=resolve_hosted_run_concurrency_lane(
                f"fins_{request.command.name}"
            ),
        )

        if request.command.stream:
            execution = self._execute_stream(request.command, spec)
        else:
            execution = self._execute_sync(request.command, spec)
        return FinsSubmission(
            session_id=session.session_id,
            execution=execution,
        )

    def execute(self, command: FinsCommand) -> FinsResult | AsyncIterator[FinsEvent]:
        """兼容旧调用方的财报命令接口。

        Args:
            command: 财报命令。

        Returns:
            同步结果或流式事件句柄。

        Raises:
            KeyError: 显式续聊但 session 不存在时抛出。
        """

        submission = self.submit(FinsSubmitRequest(command=command))
        return submission.execution

    def _resolve_session(
        self,
        *,
        command: FinsCommand,
        policy: SessionResolutionPolicy,
    ) -> SessionRecord:
        """解析本次财报命令对应的宿主 session。

        Args:
            command: 财报命令。
            policy: session 解析策略。

        Returns:
            对应的 Host session。

        Raises:
            KeyError: 显式续聊但 session 不存在时抛出。
            ValueError: session 策略非法时抛出。
        """

        coordinator = ServiceSessionCoordinator(
            host=self.host,
            session_source=self.session_source,
        )
        return coordinator.resolve(
            session_id=command.session_id,
            scene_name=command.name,
            policy=policy,
        )

    def _execute_sync(
        self,
        command: FinsCommand,
        spec: HostedRunSpec,
    ) -> FinsResult:
        """同步执行财报命令。"""

        return self.host.run_operation_sync(
            spec=spec,
            operation=lambda context: self._execute_command_sync(
                command,
                cancel_checker=_build_cancel_checker(context),
            ),
        )

    async def _execute_stream(
        self,
        command: FinsCommand,
        spec: HostedRunSpec,
    ) -> AsyncIterator[FinsEvent]:
        """流式执行财报命令。"""

        async for event in self.host.run_operation_stream(
            spec=spec,
            event_stream_factory=lambda _context: self._execute_command_stream(command),
        ):
            yield event

    def _execute_command_sync(
        self,
        command: FinsCommand,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> FinsResult:
        """执行同步财报命令。"""

        result = self.fins_runtime.execute(command, cancel_checker=cancel_checker)
        if not isinstance(result, FinsResult):
            raise TypeError(f"同步执行应返回 FinsResult，实际得到 {type(result).__name__}")
        return result

    async def _execute_command_stream(self, command: FinsCommand) -> AsyncIterator[FinsEvent]:
        """执行流式财报命令。"""

        result = self.fins_runtime.execute(command)
        if isinstance(result, FinsResult):
            raise TypeError(f"流式执行不应返回 FinsResult，应为 AsyncIterator")
        async for event in result:
            yield event


__all__ = [
    "FinsCommand",
    "FinsEvent",
    "FinsEventType",
    "FinsResult",
    "FinsService",
]
