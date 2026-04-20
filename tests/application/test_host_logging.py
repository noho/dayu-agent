"""Host 门面日志测试。"""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest

from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedModelSpec,
    ExecutionContract,
    ExecutionHostPolicy,
    ExecutionMessageInputs,
    ScenePreparationSpec,
)
from dayu.contracts.session import SessionSource
from dayu.host.host import Host
from dayu.log import Log
from tests.application.conftest import StubHostExecutor, StubRunRegistry, StubSessionRegistry


def _build_host() -> Host:
    """构造显式注入依赖的 Host。"""

    return Host(
        executor=StubHostExecutor(),
        session_registry=StubSessionRegistry(),
        run_registry=StubRunRegistry(),
    )


def _build_execution_contract() -> ExecutionContract:
    """构造最小化 ExecutionContract。"""

    return ExecutionContract(
        service_name="chat_turn",
        scene_name="wechat",
        host_policy=ExecutionHostPolicy(session_key="session_1", resumable=True),
        preparation_spec=ScenePreparationSpec(),
        message_inputs=ExecutionMessageInputs(user_message="问题"),
        accepted_execution_spec=AcceptedExecutionSpec(model=AcceptedModelSpec(model_name="test-model")),
    )


@pytest.mark.unit
def test_host_emits_logs_for_session_and_run_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host 门面关键动作应输出 debug/info 日志。"""

    host = _build_host()
    debug_mock = Mock()
    info_mock = Mock()
    monkeypatch.setattr(Log, "debug", debug_mock)
    monkeypatch.setattr(Log, "info", info_mock)

    host.create_session(SessionSource.CLI, session_id="session_1", scene_name="interactive")
    host.ensure_session("session_1", SessionSource.CLI, scene_name="interactive")
    asyncio.run(host.run_agent_and_wait(_build_execution_contract()))
    run = host._run_registry.register_run(session_id="session_1", service_type="chat_turn")
    host.cancel_run(run.run_id)

    debug_messages = [call.args[0] for call in debug_mock.call_args_list]
    info_messages = [call.args[0] for call in info_mock.call_args_list]

    assert any("Host 创建 session" in message for message in debug_messages)
    assert any("Host ensure session" in message for message in debug_messages)
    assert any("Host 启动 agent sync wait" in message for message in debug_messages)
    assert any("Host 请求取消 run" in message for message in info_messages)
