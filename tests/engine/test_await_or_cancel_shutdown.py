"""``await_or_cancel`` 双 Ctrl-C 双门控收口测试（issue #114 Phase B）。

第二次 SIGINT 后 asyncio 默认 executor shutdown，DNS ``getaddrinfo`` 等路径
仍走 ``executor.submit`` → 撞 ``RuntimeError: cannot schedule new futures after
shutdown``。``await_or_cancel`` 在双门控（token 已取消 + 错误文本严格匹配）
下把该 RuntimeError 转 ``CancelledError``，其他情形原样上抛。
"""

from __future__ import annotations

import asyncio

import pytest

from dayu.contracts.cancellation import CancelledError, CancellationToken
from dayu.engine.cancellation import await_or_cancel


def _noop_check() -> None:
    """raise_if_cancelled 桩：始终通过。"""


async def _raise_runtime_error(message: str) -> None:
    """构造一个抛出指定 RuntimeError 的协程。"""

    raise RuntimeError(message)


@pytest.mark.asyncio
async def test_runtime_error_mapped_when_token_cancelled_and_message_matches() -> None:
    """token 已取消 + 错误文本严格匹配 → 转 CancelledError。"""

    token = CancellationToken()
    token.cancel()
    waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    waiter.set_result(None)

    # 让 cancellation_waiter 先 done，但因为 token 已取消会先走 cancel 路径，
    # 故这里不用 waiter 通道——把 waiter 设为 pending，让 task 先抛错。
    pending_waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    with pytest.raises(CancelledError) as exc_info:
        await await_or_cancel(
            _raise_runtime_error("cannot schedule new futures after shutdown"),
            operation_name="dns_lookup",
            cancellation_waiter=pending_waiter,
            cancellation_token=token,
            raise_if_cancelled=_noop_check,
            log_prefix="[TEST]",
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_runtime_error_propagates_when_token_not_cancelled() -> None:
    """token 未取消 → RuntimeError 原样上抛，禁止误吞。"""

    token = CancellationToken()
    pending_waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        await await_or_cancel(
            _raise_runtime_error("cannot schedule new futures after shutdown"),
            operation_name="dns_lookup",
            cancellation_waiter=pending_waiter,
            cancellation_token=token,
            raise_if_cancelled=_noop_check,
            log_prefix="[TEST]",
        )


@pytest.mark.asyncio
async def test_runtime_error_propagates_when_message_mismatches() -> None:
    """token 已取消但错误文本不匹配 → 仍原样上抛，避免误吞业务异常。"""

    token = CancellationToken()
    token.cancel()
    pending_waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    with pytest.raises(RuntimeError, match="some other error"):
        await await_or_cancel(
            _raise_runtime_error("some other error"),
            operation_name="dns_lookup",
            cancellation_waiter=pending_waiter,
            cancellation_token=token,
            raise_if_cancelled=_noop_check,
            log_prefix="[TEST]",
        )


@pytest.mark.asyncio
async def test_runtime_error_propagates_when_token_is_none() -> None:
    """无 token + cancellation_waiter=None 场景下，RuntimeError 原样上抛。"""

    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        await await_or_cancel(
            _raise_runtime_error("cannot schedule new futures after shutdown"),
            operation_name="dns_lookup",
            cancellation_waiter=None,
            cancellation_token=None,
            raise_if_cancelled=_noop_check,
            log_prefix="[TEST]",
        )


@pytest.mark.asyncio
async def test_runtime_error_propagates_when_message_only_contains_phrase() -> None:
    """token 已取消 + 错误文本仅包含目标短语作为子串 → 原样上抛。

    收口子串匹配的退化风险：业务侧若抛出含 "cannot schedule new futures
    after shutdown" 子串但根因不同的 RuntimeError（例如外层 wrapper 拼接
    了上下文前缀），不能在取消窗口期被折叠成 CancelledError 掩盖根因。
    """

    token = CancellationToken()
    token.cancel()
    pending_waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()

    with pytest.raises(RuntimeError, match="wrapper error: cannot schedule new futures after shutdown"):
        await await_or_cancel(
            _raise_runtime_error(
                "wrapper error: cannot schedule new futures after shutdown"
            ),
            operation_name="dns_lookup",
            cancellation_waiter=pending_waiter,
            cancellation_token=token,
            raise_if_cancelled=_noop_check,
            log_prefix="[TEST]",
        )
