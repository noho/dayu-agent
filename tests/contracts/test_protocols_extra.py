"""`contracts.protocols` 额外覆盖测试。"""

from __future__ import annotations

import pytest

from dayu.contracts.cancellation import CancellationToken
from dayu.contracts.protocols import ToolExecutionContext


@pytest.mark.unit
def test_tool_execution_context_retains_explicit_field_values() -> None:
    """显式构造的上下文应原样保留各字段。"""

    token = CancellationToken()
    context = ToolExecutionContext(
        run_id="run_1",
        iteration_id="2",
        tool_call_id="tool_1",
        index_in_iteration=3,
        timeout_seconds=4.5,
        cancellation_token=token,
    )

    assert context.run_id == "run_1"
    assert context.iteration_id == "2"
    assert context.tool_call_id == "tool_1"
    assert context.index_in_iteration == 3
    assert context.timeout_seconds == 4.5
    assert context.cancellation_token is token


@pytest.mark.unit
def test_tool_execution_context_defaults_are_stable() -> None:
    """上下文对象的默认值应保持稳定且可预测。"""

    context = ToolExecutionContext(run_id="run_1", index_in_iteration=1)

    assert context.run_id == "run_1"
    assert context.index_in_iteration == 1
    assert context.iteration_id is None
    assert context.tool_call_id is None
    assert context.timeout_seconds is None
    assert context.cancellation_token is None


@pytest.mark.unit
def test_tool_execution_context_supports_absent_optional_fields() -> None:
    """可选字段省略时应保持空值，不再隐式做映射归一化。"""

    context = ToolExecutionContext()

    assert context.run_id is None
    assert context.iteration_id is None
    assert context.tool_call_id is None
    assert context.index_in_iteration == 0
    assert context.timeout_seconds is None
