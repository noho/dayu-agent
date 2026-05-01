"""WritePipeline 取消时多 chapter drain/settle 集成测试（issue #114 Phase C）。

验证 Phase A executor 自查 + 异常映射、Phase B ``await_or_cancel`` 双门控
两个改动联动后，多 chapter 并发场景在取消触发后能在 stub LLM 立即返回的
前提下 settle，且不依赖墙钟。语义点：

- Settled 章节（取消前已 finalize）→ 最终产物保留；
- 未 settle 章节（取消时未 finalize）→ 不进入 fallback、不产出 fallback chapter.md。

本测试聚焦 ``_HostScenePromptContractExecutor`` 在多并发 await 下被同时
取消的 settle 行为，对 PR 73/84/87/92 的取消语义点对点引用、不复制。
"""

from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import AsyncMock

import pytest

from dayu.contracts.agent_execution import ExecutionContract, ReplayHandle
from dayu.contracts.cancellation import CancelledError, CancellationToken
from dayu.contracts.events import AppResult
from dayu.host.host_execution import HostExecutorProtocol
from dayu.host.protocols import SessionClosedError
from dayu.services.internal.write_pipeline.pipeline import (
    _HostScenePromptContractExecutor,
)


def _stub_app_result() -> AppResult:
    """构造最小 AppResult。"""

    return AppResult(content="stub", errors=[], warnings=[])


def _stub_handle(idx: int) -> ReplayHandle:
    """构造带索引的 ReplayHandle。"""

    return ReplayHandle(handle_id=f"handle-{idx}")


def _stub_contract() -> ExecutionContract:
    """构造占位 ExecutionContract。"""

    return cast(ExecutionContract, object())


class _SettleStubHostExecutor:
    """模拟 host_executor：在 token cancel 后让 in-flight LLM 抛 CancelledError。

    模拟两条真实链路：
    1. ``run_agent_and_wait_replayable`` 已 in-flight：cancel 后 await 即时
       抛 ``CancelledError``（对应 Host child run 收 token cancel → bridge
       传播到 ``await_or_cancel`` 抛 ``CancelledError``）。
    2. cancel 后下一次进入入口 → executor 自查直接抛（Phase A）。
    """

    def __init__(self, token: CancellationToken) -> None:
        self._token = token
        self._call_count = 0
        self.discard_replay_state = AsyncMock()

    async def run_agent_and_wait_replayable(
        self, _execution_contract: ExecutionContract
    ) -> tuple[AppResult, ReplayHandle]:
        self._call_count += 1
        # 等待 token 取消或最多 0.5 秒（CI 上限）；对应真实 await_or_cancel
        # 在收到取消后即时抛 CancelledError。
        loop = asyncio.get_running_loop()
        cancel_event = asyncio.Event()

        def _on_cancel() -> None:
            loop.call_soon_threadsafe(cancel_event.set)

        unregister = self._token.on_cancel(_on_cancel)
        if self._token.is_cancelled():
            cancel_event.set()
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            return _stub_app_result(), _stub_handle(self._call_count)
        finally:
            unregister()
        raise CancelledError("operation cancelled: stub_llm")

    async def replay_agent_and_wait(
        self,
        _handle: ReplayHandle,
        _execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        return _stub_app_result(), _stub_handle(self._call_count)


@pytest.mark.asyncio
async def test_concurrent_chapters_all_settle_on_cancel() -> None:
    """多个 chapter 并发执行 → 取消触发后所有 future 在 stub LLM 即时返回下 settle。

    断言：
    - 每个并发 in-flight LLM 都抛 ``CancelledError``，不留悬空 future。
    - 不依赖墙钟（stub 内部 wait timeout 仅作 CI 兜底）。
    """

    token = CancellationToken()
    stub = _SettleStubHostExecutor(token)
    executor = _HostScenePromptContractExecutor(
        host_executor=cast(HostExecutorProtocol, stub),
        cancellation_token=token,
    )

    async def _chapter() -> AppResult:
        result, _ = await executor.run_replayable(_stub_contract())
        return result

    # 启动 5 个并发 chapter，模拟 in-flight LLM
    tasks = [asyncio.create_task(_chapter()) for _ in range(5)]
    # 让任务进入 host LLM await 阶段
    await asyncio.sleep(0)
    # 触发取消（对应 ProcessShutdownCoordinator.settle_active_runs 的 cascade）
    token.cancel()

    results = await asyncio.gather(*tasks, return_exceptions=True)
    # 全部抛 CancelledError，无残余、无超时
    assert all(isinstance(r, CancelledError) for r in results), results


@pytest.mark.asyncio
async def test_subsequent_call_after_cancel_short_circuits_at_executor_entry() -> None:
    """取消触发后再发起的 LLM 调用直接走 executor 自查抛 CancelledError，不进 host。

    对应 Phase A 设计点：取消落在"checkpoint 已通过、register_run 之间"
    窗口期外的、新一次 LLM 入口前——executor 自查命中即退。
    """

    token = CancellationToken()
    token.cancel()
    stub = _SettleStubHostExecutor(token)
    executor = _HostScenePromptContractExecutor(
        host_executor=cast(HostExecutorProtocol, stub),
        cancellation_token=token,
    )

    with pytest.raises(CancelledError):
        await executor.run_replayable(_stub_contract())

    # 入口自查命中，host LLM 未被调用
    assert stub._call_count == 0


@pytest.mark.asyncio
async def test_register_run_window_session_closed_mapped_to_cancelled() -> None:
    """取消落在"checkpoint 通过、register_run 之间"窗口 → barrier 异常映射成 CancelledError。

    对应 Phase A 设计点：cancel-induced ``SessionClosedError`` 在双门控下
    映射成 ``CancelledError``，避免 pipeline fallback 把取消误判成普通失败。
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

    class _BarrierStub:
        def __init__(self) -> None:
            self.discard_replay_state = AsyncMock()

        async def run_agent_and_wait_replayable(
            self, _ec: ExecutionContract
        ) -> tuple[AppResult, ReplayHandle]:
            late_token.fire()
            raise SessionClosedError("sess-1")

        async def replay_agent_and_wait(
            self, _h: ReplayHandle, _ec: ExecutionContract
        ) -> tuple[AppResult, ReplayHandle]:
            raise NotImplementedError

    stub = _BarrierStub()
    executor = _HostScenePromptContractExecutor(
        host_executor=cast(HostExecutorProtocol, stub),
        cancellation_token=late_token,
    )

    with pytest.raises(CancelledError) as exc_info:
        await executor.run_replayable(_stub_contract())

    assert isinstance(exc_info.value.__cause__, SessionClosedError)
