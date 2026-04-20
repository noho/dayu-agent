"""Host reply outbox façade 测试。"""

from __future__ import annotations

import asyncio

import pytest

from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedModelSpec,
    ExecutionContract,
    ExecutionHostPolicy,
    ExecutionMessageInputs,
    ScenePreparationSpec,
)
from dayu.contracts.reply_outbox import ReplyOutboxState, ReplyOutboxSubmitRequest
from dayu.host.host import Host
from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore
from tests.application.conftest import StubHostExecutor, StubRunRegistry, StubSessionRegistry


def _build_host() -> Host:
    """构造显式注入依赖的 Host。

    Args:
        无。

    Returns:
        可用于单元测试的 Host。

    Raises:
        无。
    """

    return Host(
        executor=StubHostExecutor(),
        session_registry=StubSessionRegistry(),
        run_registry=StubRunRegistry(),
        reply_outbox_store=InMemoryReplyOutboxStore(),
    )


def _build_execution_contract() -> ExecutionContract:
    """构造最小化 ExecutionContract。

    Args:
        无。

    Returns:
        可被 StubHostExecutor 接收的执行契约。

    Raises:
        无。
    """

    return ExecutionContract(
        service_name="chat",
        scene_name="wechat",
        host_policy=ExecutionHostPolicy(session_key="session_1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=AcceptedExecutionSpec(model=AcceptedModelSpec(model_name="test-model")),
        metadata={"delivery_channel": "wechat", "delivery_target": "user_1"},
    )


@pytest.mark.unit
def test_host_reply_outbox_facade_round_trip() -> None:
    """Host façade 应显式托管 reply outbox 的提交流程与状态流转。"""

    host = _build_host()
    created = host.submit_reply_for_delivery(
        ReplyOutboxSubmitRequest(
            delivery_key="wechat:run_1",
            session_id="session_1",
            scene_name="wechat",
            source_run_id="run_1",
            reply_content="结论",
            metadata={"delivery_channel": "wechat", "delivery_target": "user_1"},
        )
    )
    claimed = host.claim_reply_delivery(created.delivery_id)
    failed = host.mark_reply_delivery_failed(
        claimed.delivery_id,
        retryable=True,
        error_message="网络重试",
    )
    delivered = host.mark_reply_delivered(host.claim_reply_delivery(failed.delivery_id).delivery_id)

    listed = host.list_reply_outbox(session_id="session_1")

    assert created.state == ReplyOutboxState.PENDING_DELIVERY
    assert claimed.delivery_attempt_count == 1
    assert failed.state == ReplyOutboxState.FAILED_RETRYABLE
    assert delivered.state == ReplyOutboxState.DELIVERED
    assert [record.delivery_id for record in listed] == [created.delivery_id]


@pytest.mark.unit
def test_host_agent_success_does_not_auto_enqueue_reply_outbox() -> None:
    """Host internal success 只返回结果，不会自动写入 reply outbox。"""

    host = _build_host()

    result = asyncio.run(host.run_agent_and_wait(_build_execution_contract()))

    assert result.content == "done"
    assert host.list_reply_outbox() == []
