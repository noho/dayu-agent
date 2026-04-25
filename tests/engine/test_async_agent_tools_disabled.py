"""``AsyncAgent._tools_disabled`` 与 ``run_messages(disable_tools=True)`` 的单元测试。

覆盖目标：
1. ``_tools_disabled`` 上下文进入时调用 ``runner.set_tools(None)``，退出时恢复
   原 ``tool_executor``，且无 ``tool_executor`` 时仅 yield 不操作 runner。
2. ``run_messages(disable_tools=True)`` 在执行期间使 runner 的工具列表为 None，
   退出后恢复（与 force_answer 共享同一原语）。
3. ``run_messages(disable_tools=False)`` 不触发禁用/恢复对调用，保持现有行为。
"""

from __future__ import annotations

import pytest

from dayu.engine import AsyncAgent
from dayu.engine import (
    content_complete,
    content_delta,
    done_event,
)
from tests.engine.test_async_agent import DummyRunner, DummyToolExecutor


pytestmark = pytest.mark.asyncio


async def test_tools_disabled_toggles_set_tools_around_block() -> None:
    """``_tools_disabled`` 进入时清空 runner 工具，退出时恢复。"""

    runner = DummyRunner([])
    executor = DummyToolExecutor([{"name": "tool"}])
    agent = AsyncAgent(runner, tool_executor=executor)

    runner.set_tools_calls.clear()
    async with agent._tools_disabled():
        assert runner.set_tools_calls[-1] == ((None,), {})

    args_seq = [call[0][0] for call in runner.set_tools_calls]
    assert args_seq == [None, executor]


async def test_tools_disabled_no_op_without_executor() -> None:
    """无 ``tool_executor`` 时，``_tools_disabled`` 不应触发任何 set_tools 调用。"""

    runner = DummyRunner([])
    agent = AsyncAgent(runner, tool_executor=None)

    runner.set_tools_calls.clear()
    async with agent._tools_disabled():
        pass

    assert runner.set_tools_calls == []


async def test_tools_disabled_restores_after_exception() -> None:
    """``_tools_disabled`` 块内异常时也必须恢复工具能力。"""

    runner = DummyRunner([])
    executor = DummyToolExecutor([{"name": "tool"}])
    agent = AsyncAgent(runner, tool_executor=executor)

    runner.set_tools_calls.clear()
    with pytest.raises(RuntimeError, match="boom"):
        async with agent._tools_disabled():
            raise RuntimeError("boom")

    args_seq = [call[0][0] for call in runner.set_tools_calls]
    assert args_seq[0] is None
    assert args_seq[-1] is executor


async def test_run_messages_disable_tools_wraps_loop() -> None:
    """``run_messages(disable_tools=True)`` 期间 runner 的工具应被禁用。"""

    runner = DummyRunner(
        [[content_delta("hi"), content_complete("hi"), done_event()]]
    )
    executor = DummyToolExecutor([{"name": "tool"}])
    agent = AsyncAgent(runner, tool_executor=executor)

    runner.set_tools_calls.clear()
    events = []
    async for event in agent.run_messages([], disable_tools=True):
        events.append(event)

    args_seq = [call[0][0] for call in runner.set_tools_calls]
    # _run_loop 内部还会主动 set_tools(executor) 一次以装配工具，所以
    # 序列应当包含 None 段并最终恢复到 executor。
    assert None in args_seq
    assert args_seq[-1] is executor
    # disable 段必须出现在最终恢复之前。
    last_none_index = max(i for i, v in enumerate(args_seq) if v is None)
    last_exec_index = max(i for i, v in enumerate(args_seq) if v is executor)
    assert last_none_index < last_exec_index


async def test_run_messages_disable_tools_default_false_does_not_clear() -> None:
    """默认 ``disable_tools=False`` 时不应额外触发禁用恢复对调用。"""

    runner = DummyRunner(
        [[content_delta("hi"), content_complete("hi"), done_event()]]
    )
    executor = DummyToolExecutor([{"name": "tool"}])
    agent = AsyncAgent(runner, tool_executor=executor)

    runner.set_tools_calls.clear()
    async for _event in agent.run_messages([]):
        pass

    args_seq = [call[0][0] for call in runner.set_tools_calls]
    # 默认路径下不应该出现 None 段。
    assert None not in args_seq


async def test_run_messages_disable_tools_keeps_tools_none_inside_loop() -> None:
    """``run_messages(disable_tools=True)`` 期间 ``_run_loop`` 不能把工具又装回去。

    修复点：``_run_loop`` 会在每轮 iteration 入口无条件 ``set_tools(executor)``，
    导致 replay 路径仍然可能发起工具调用。这里通过观察 runner.call 时刻的工具
    状态来回归该问题。
    """

    runner = DummyRunner(
        [[content_delta("text"), content_complete("text"), done_event()]]
    )
    executor = DummyToolExecutor([{"name": "tool"}])
    agent = AsyncAgent(runner, tool_executor=executor)

    runner.set_tools_calls.clear()

    # 包一层 runner.call，记录每次 call 入口处最近一次 set_tools 的参数。
    original_call = runner.call
    snapshots: list[object] = []

    def patched_call(messages, stream=True, **extra_payloads):
        last_set = (
            runner.set_tools_calls[-1][0][0]
            if runner.set_tools_calls
            else "no-set-tools"
        )
        snapshots.append(last_set)
        return original_call(messages, stream=stream, **extra_payloads)

    runner.call = patched_call  # type: ignore[assignment]

    async for _event in agent.run_messages([], disable_tools=True):
        pass

    # 在 runner.call 真正发起模型请求之前，runner 上的工具必须仍然是 None；
    # 否则 disable_tools 语义形同虚设。
    assert snapshots, "runner.call 应至少被触发一次"
    assert snapshots[0] is None
