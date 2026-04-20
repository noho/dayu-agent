"""协作式取消辅助函数。"""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any


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


__all__ = ["cancel_task_and_wait", "resolve_cancellation_waiter"]
