"""WritePipeline 章节内取消（issue #114 Phase A）单元测试。

聚焦 ``_HostScenePromptContractExecutor`` 的两个新增能力：

1. 调 ``run_agent_and_wait_replayable`` / ``replay_agent_and_wait`` **前**
   一次 ``raise_if_cancelled``——token 命中即退当前 LLM。
2. 捕获 ``SessionWriteBlockedError`` 子类时双门控映射成
   ``CancelledError``（token 已取消 + 异常属于白名单基类），保留原异常
   作 ``__cause__``；token 未取消则原样上抛，避免误吞。
"""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock

import pytest

from dayu.contracts.agent_execution import ExecutionContract, ReplayHandle
from dayu.contracts.cancellation import CancelledError, CancellationToken
from dayu.contracts.events import AppResult
from dayu.host.host_execution import HostExecutorProtocol
from dayu.host.protocols import (
    SessionClearingError,
    SessionClearingFailedError,
    SessionClosedError,
)
from dayu.services.internal.write_pipeline.pipeline import (
    _HostScenePromptContractExecutor,
)


def _make_app_result() -> AppResult:
    """构造一个最小 AppResult 桩。"""

    return AppResult(content="ok", errors=[], warnings=[])


def _make_handle() -> ReplayHandle:
    """构造一个最小 ReplayHandle 桩。"""

    return ReplayHandle(handle_id="handle-1")


def _make_contract() -> ExecutionContract:
    """构造一个不会被 executor 解析的占位 ExecutionContract。

    Executor 仅把契约透传给 host，不读取任何字段；测试中只需要任意
    可作为参数传递的对象，不必构造完整契约。
    """

    return cast(ExecutionContract, object())


class _StubHostExecutor:
    """最小 HostExecutorProtocol 桩——仅实现 executor 用到的三个方法。"""

    def __init__(self) -> None:
        self.run_replayable = AsyncMock(
            return_value=(_make_app_result(), _make_handle())
        )
        self.replay = AsyncMock(
            return_value=(_make_app_result(), _make_handle())
        )
        self.discard_calls: list[ReplayHandle] = []

    async def run_agent_and_wait_replayable(
        self, execution_contract: ExecutionContract
    ) -> tuple[AppResult, ReplayHandle]:
        return await self.run_replayable(execution_contract)

    async def replay_agent_and_wait(
        self,
        handle: ReplayHandle,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        return await self.replay(handle, execution_contract)

    def discard_replay_state(self, handle: ReplayHandle) -> None:
        self.discard_calls.append(handle)


def _build_executor(
    *, token: CancellationToken | None
) -> tuple[_HostScenePromptContractExecutor, _StubHostExecutor]:
    """组装 executor + stub host_executor。"""

    stub = _StubHostExecutor()
    executor = _HostScenePromptContractExecutor(
        host_executor=cast(HostExecutorProtocol, stub),
        cancellation_token=token,
    )
    return executor, stub


@pytest.mark.asyncio
async def test_run_replayable_raises_cancelled_when_token_already_cancelled() -> None:
    """token 在 LLM 入口已置位时，executor 直接抛 CancelledError，不发起 host 调用。"""

    token = CancellationToken()
    token.cancel()
    executor, stub = _build_executor(token=token)

    with pytest.raises(CancelledError):
        await executor.run_replayable(_make_contract())

    stub.run_replayable.assert_not_awaited()


@pytest.mark.asyncio
async def test_replay_raises_cancelled_when_token_already_cancelled() -> None:
    """replay 路径同样在入口自查 token，命中即退。"""

    token = CancellationToken()
    token.cancel()
    executor, stub = _build_executor(token=token)

    with pytest.raises(CancelledError):
        await executor.replay(_make_handle(), _make_contract())

    stub.replay.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_replayable_passes_through_when_token_not_cancelled() -> None:
    """token 未触发时，run_replayable 正常透传到 host_executor。"""

    token = CancellationToken()
    executor, stub = _build_executor(token=token)

    result, handle = await executor.run_replayable(_make_contract())

    assert result.content == "ok"
    assert handle.handle_id == "handle-1"
    stub.run_replayable.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_replayable_without_token_skips_check() -> None:
    """``cancellation_token=None`` 时退化为不做入口自查与异常映射。"""

    executor, stub = _build_executor(token=None)
    result, _ = await executor.run_replayable(_make_contract())

    assert result.content == "ok"
    stub.run_replayable.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: SessionClosedError("sess-1"),
        lambda: SessionClearingError("sess-1"),
    ],
)
async def test_cancel_induced_barrier_mapped_to_cancelled_when_token_set(
    exc_factory,
) -> None:
    """token 已取消 + 白名单 barrier 异常 → 映射成 CancelledError。

    白名单仅含 ``SessionClosedError`` / ``SessionClearingError``：这两类
    在 cancel_run_and_settle 触发后下一次 register_run 路径上抛，语义等价
    "运行已被取消"。``SessionClearingFailedError`` 由独立用例守住——它是
    持久锁定故障态、不进白名单。

    用 ``_LateCancelToken`` 模拟"自查通过、host 调用阶段被 settle 触发"的
    时序：入口检查时 token 未触发，进入 host 调用后再 ``fire`` 并抛 barrier。
    """

    class _LateCancelToken(CancellationToken):
        def __init__(self) -> None:
            super().__init__()
            self._fired = False

        def is_cancelled(self) -> bool:
            return self._fired

        def raise_if_cancelled(self) -> None:
            if self._fired:
                raise CancelledError("操作已被取消")

        def fire(self) -> None:
            self._fired = True

    late_token = _LateCancelToken()
    stub = _StubHostExecutor()
    executor = _HostScenePromptContractExecutor(
        host_executor=cast(HostExecutorProtocol, stub),
        cancellation_token=late_token,
    )
    raised = exc_factory()

    async def _side_effect_then_fire(_contract: ExecutionContract):
        late_token.fire()
        raise raised

    stub.run_replayable.side_effect = _side_effect_then_fire

    with pytest.raises(CancelledError) as exc_info:
        await executor.run_replayable(_make_contract())

    assert exc_info.value.__cause__ is raised


@pytest.mark.asyncio
async def test_session_closed_error_propagates_when_token_not_cancelled() -> None:
    """token 未取消时，barrier 异常原样上抛，禁止被误吞为 CancelledError。"""

    token = CancellationToken()
    executor, stub = _build_executor(token=token)
    raised = SessionClosedError("sess-1")
    stub.run_replayable.side_effect = raised

    with pytest.raises(SessionClosedError) as exc_info:
        await executor.run_replayable(_make_contract())

    assert exc_info.value is raised


@pytest.mark.asyncio
async def test_session_clearing_failed_error_never_mapped_even_when_cancelled() -> None:
    """``SessionClearingFailedError`` 不进 cancel-induced 白名单。

    它是 host 持久锁定故障态、需人工恢复——即便外层 token 已取消也必须
    原样上抛，避免被掩盖成普通用户取消导致 actionable 信号丢失。
    """

    class _LateCancelToken(CancellationToken):
        def __init__(self) -> None:
            super().__init__()
            self._fired = False

        def is_cancelled(self) -> bool:
            return self._fired

        def raise_if_cancelled(self) -> None:
            if self._fired:
                raise CancelledError("操作已被取消")

        def fire(self) -> None:
            self._fired = True

    late_token = _LateCancelToken()
    stub = _StubHostExecutor()
    executor = _HostScenePromptContractExecutor(
        host_executor=cast(HostExecutorProtocol, stub),
        cancellation_token=late_token,
    )
    raised = SessionClearingFailedError("sess-1")

    async def _side_effect(_contract: ExecutionContract):
        late_token.fire()
        raise raised

    stub.run_replayable.side_effect = _side_effect

    with pytest.raises(SessionClearingFailedError) as exc_info:
        await executor.run_replayable(_make_contract())

    assert exc_info.value is raised


@pytest.mark.asyncio
async def test_replay_maps_cancel_induced_barrier_under_double_gate() -> None:
    """replay 路径同样在双门控下把 barrier 映射成 CancelledError。"""

    class _LateCancelToken(CancellationToken):
        def __init__(self) -> None:
            super().__init__()
            self._fired = False

        def is_cancelled(self) -> bool:
            return self._fired

        def raise_if_cancelled(self) -> None:
            if self._fired:
                raise CancelledError("操作已被取消")

        def fire(self) -> None:
            self._fired = True

    token = _LateCancelToken()
    executor, stub = _build_executor(token=token)
    raised = SessionClearingError("sess-1")

    async def _side_effect(_handle: ReplayHandle, _contract: ExecutionContract):
        token.fire()
        raise raised

    stub.replay.side_effect = _side_effect

    with pytest.raises(CancelledError) as exc_info:
        await executor.replay(_make_handle(), _make_contract())

    assert exc_info.value.__cause__ is raised


@pytest.mark.asyncio
async def test_replay_session_closed_error_propagates_without_token() -> None:
    """``cancellation_token=None`` 时 replay 路径不做映射，barrier 异常原样上抛。"""

    executor, stub = _build_executor(token=None)
    raised = SessionClosedError("sess-1")
    stub.replay.side_effect = raised

    with pytest.raises(SessionClosedError):
        await executor.replay(_make_handle(), _make_contract())


def test_discard_delegates_to_host_executor() -> None:
    """discard 仅做透传，不引入额外语义。"""

    executor, stub = _build_executor(token=None)
    handle = _make_handle()
    executor.discard(handle)
    assert stub.discard_calls == [handle]
