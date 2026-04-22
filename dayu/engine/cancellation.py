"""协作式取消辅助函数。"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import suppress
from typing import Any, Awaitable, Callable, TypeVar

from dayu.contracts.cancellation import CancelledError as EngineCancelledError, CancellationToken
from dayu.log import Log

MODULE = "ENGINE.CANCELLATION"

_AwaitableResult = TypeVar("_AwaitableResult")


def resolve_cancellation_waiter(waiter: asyncio.Future[None]) -> None:
    """安全地完成取消等待 future。"""

    if not waiter.done():
        waiter.set_result(None)


async def cancel_task_and_wait(task: asyncio.Future[Any]) -> None:
    """取消并等待子任务收口。

    参数:
        task: 需要取消的异步任务或 future。

    返回值:
        无。

    异常:
        无。取消或等待中的异常会被内部吞掉。
    """

    if task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


def create_cancellation_waiter(
    cancellation_token: CancellationToken | None,
) -> tuple[asyncio.Future[None] | None, Callable[[], None] | None]:
    """创建与取消令牌联动的等待 future。

    参数:
        cancellation_token: 可选取消令牌；为空时表示当前流程不支持协作式取消。

    返回值:
        取消等待 future 与其注销回调；未配置取消令牌时返回 ``(None, None)``。

    异常:
        RuntimeError: 当前事件循环不可用时由底层抛出。
    """

    if cancellation_token is None:
        return None, None
    loop = asyncio.get_running_loop()
    waiter: asyncio.Future[None] = loop.create_future()

    def _on_cancel() -> None:
        loop.call_soon_threadsafe(resolve_cancellation_waiter, waiter)

    unregister = cancellation_token.on_cancel(_on_cancel)
    if cancellation_token.is_cancelled():
        resolve_cancellation_waiter(waiter)
    return waiter, unregister


async def await_or_cancel(
    awaitable: Awaitable[_AwaitableResult],
    *,
    operation_name: str,
    cancellation_waiter: asyncio.Future[None] | None,
    cancellation_token: CancellationToken | None,
    raise_if_cancelled: Callable[[], None],
    log_prefix: str,
    log_module: str = MODULE,
) -> _AwaitableResult:
    """等待 awaitable，并在取消信号先到时优先中止。

    参数:
        awaitable: 需要等待的协程或 awaitable。
        operation_name: 当前等待点名称，用于日志与异常消息。
        cancellation_waiter: 与取消令牌绑定的等待 future。
        cancellation_token: 当前取消令牌。
        raise_if_cancelled: 调用前执行的取消检查函数。
        log_prefix: 取消日志前缀。
        log_module: 日志模块名。

    返回值:
        awaitable 的返回结果。

    异常:
        EngineCancelledError: 当取消信号先到达时抛出。
        asyncio.CancelledError: 当前任务被外层取消时抛出。
        Exception: 透传业务 awaitable 的原始异常。
    """

    try:
        raise_if_cancelled()
    except Exception:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        raise
    if cancellation_waiter is None:
        return await awaitable

    task = asyncio.ensure_future(awaitable)
    try:
        done, _ = await asyncio.wait(
            {task, cancellation_waiter},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if cancellation_waiter in done and cancellation_token is not None and cancellation_token.is_cancelled():
            await cancel_task_and_wait(task)
            Log.info(
                f"{log_prefix} 等待点已因取消中止: {operation_name}",
                module=log_module,
            )
            raise EngineCancelledError(f"operation cancelled: {operation_name}")
        return await task
    except asyncio.CancelledError:
        await cancel_task_and_wait(task)
        raise


__all__ = [
    "await_or_cancel",
    "cancel_task_and_wait",
    "create_cancellation_waiter",
    "resolve_cancellation_waiter",
]
