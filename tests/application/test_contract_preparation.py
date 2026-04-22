"""contract_preparation 测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    AcceptedInfrastructureSpec,
    AcceptedModelSpec,
    AcceptedRuntimeSpec,
    AcceptedToolConfigSpec,
)
from dayu.contracts.tool_configs import DocToolLimits, FinsToolLimits, WebToolsConfig
from dayu.contracts.toolset_config import build_toolset_config_snapshot
from dayu.execution.options import (
    ConversationMemorySettings,
    TraceSettings,
    resolve_web_tools_config_from_toolset_configs,
)
from dayu.services.contract_preparation import (
    prepare_execution_contract,
    prepare_execution_permissions,
    prepare_message_inputs,
)


def _build_accepted_execution_spec(*, allow_private_network_url: bool) -> AcceptedExecutionSpec:
    """构建测试用 accepted_execution_spec。"""

    return AcceptedExecutionSpec(
        model=AcceptedModelSpec(model_name="test-model", temperature=0.2),
        runtime=AcceptedRuntimeSpec(
            runner_running_config={"tool_timeout_seconds": 30.0},
            agent_running_config={"max_iterations": 8},
        ),
        tools=AcceptedToolConfigSpec(
            toolset_configs=tuple(
                snapshot
                for snapshot in (
                    build_toolset_config_snapshot("doc", DocToolLimits()),
                    build_toolset_config_snapshot("fins", FinsToolLimits()),
                    build_toolset_config_snapshot(
                        "web",
                        WebToolsConfig(
                            provider="auto",
                            allow_private_network_url=allow_private_network_url,
                        ),
                    ),
                )
                if snapshot is not None
            ),
        ),
        infrastructure=AcceptedInfrastructureSpec(
            trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp/trace")),
            conversation_memory_settings=ConversationMemorySettings(),
        ),
    )


@pytest.mark.unit
def test_prepare_execution_permissions_uses_web_tools_private_network_flag() -> None:
    """验证联网私网权限从 accepted_execution_spec 同步进入 execution_permissions。"""

    permissions = prepare_execution_permissions(
        accepted_execution_spec=_build_accepted_execution_spec(allow_private_network_url=True),
    )

    assert permissions.web.allow_private_network_url is True


@pytest.mark.unit
def test_prepare_execution_contract_uses_execution_permissions_as_private_network_truth() -> None:
    """验证 contract preparation 会把私网权限写入 execution_permissions 真源。"""

    contract = prepare_execution_contract(
        service_name="prompt",
        scene_name="prompt",
        accepted_execution_spec=_build_accepted_execution_spec(allow_private_network_url=True),
        prompt_contributions={"base_user": "x"},
        user_message="hello",
        session_key="session-1",
        business_concurrency_lane=None,
    )

    web_tools_config = resolve_web_tools_config_from_toolset_configs(
        contract.accepted_execution_spec.tools.toolset_configs
    )
    assert web_tools_config is not None
    assert bool(web_tools_config.allow_private_network_url) is True
    assert contract.preparation_spec.selected_toolsets == ()
    assert contract.preparation_spec.execution_permissions.web.allow_private_network_url is True


@pytest.mark.unit
def test_prepare_execution_contract_filters_prompt_contributions_by_context_slots() -> None:
    """验证 Service contract preparation 只把 scene manifest 允许的 slot 写入契约。"""

    contract = prepare_execution_contract(
        service_name="prompt",
        scene_name="infer",
        accepted_execution_spec=_build_accepted_execution_spec(allow_private_network_url=False),
        prompt_contributions={
            "base_user": "# 用户与运行时上下文",
            "fins_default_subject": "# 当前分析对象",
            "unexpected": "should be ignored",
        },
        context_slots=("fins_default_subject",),
        user_message="hello",
        session_key="session-1",
        business_concurrency_lane=None,
    )

    assert contract.preparation_spec.prompt_contributions == {
        "fins_default_subject": "# 当前分析对象",
    }


@pytest.mark.unit
def test_prepare_message_inputs_rejects_blank_user_message() -> None:
    """验证空白消息会被 contract preparation 拒绝。"""

    with pytest.raises(ValueError):
        prepare_message_inputs(user_message="   ")
