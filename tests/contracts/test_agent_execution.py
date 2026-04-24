"""ExecutionContract 快照测试。"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from dayu.contracts.agent_execution import (
    AcceptedInfrastructureSpec,
    AcceptedExecutionSpec,
    AcceptedModelSpec,
    AcceptedRuntimeSpec,
    AcceptedToolConfigSpec,
    ExecutionContract,
    ExecutionContractSnapshotValue,
    ExecutionDocPermissions,
    ExecutionHostPolicy,
    ExecutionMessageInputs,
    ExecutionPermissions,
    ExecutionWebPermissions,
    ScenePreparationSpec,
)
from dayu.contracts.agent_execution_serialization import (
    deserialize_execution_contract_snapshot,
    serialize_execution_contract_snapshot,
)
from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.contracts.tool_configs import DocToolLimits, FinsToolLimits, WebToolsConfig
from dayu.contracts.toolset_config import ToolsetConfigSnapshot, build_toolset_config_snapshot
from dayu.execution.options import (
    ExecutionOptionsSnapshot,
    ConversationMemorySettings,
    ExecutionOptions,
    TraceSettings,
)


def _build_toolset_configs(
    *,
    doc_tool_limits: DocToolLimits | None = None,
    fins_tool_limits: FinsToolLimits | None = None,
    web_tools_config: WebToolsConfig | None = None,
) -> tuple[ToolsetConfigSnapshot, ...]:
    """构造测试用通用 toolset 配置快照。"""

    return tuple(
        snapshot
        for snapshot in (
            build_toolset_config_snapshot("doc", doc_tool_limits),
            build_toolset_config_snapshot("fins", fins_tool_limits),
            build_toolset_config_snapshot("web", web_tools_config),
        )
        if snapshot is not None
    )


def _read_attribute(target: object, name: str) -> object:
    """读取测试目标的属性值。

    Args:
        target: 待读取属性的对象。
        name: 属性名。

    Returns:
        属性值。

    Raises:
        AttributeError: 属性不存在时抛出。
    """

    return getattr(target, name)


@pytest.mark.unit
def test_execution_contract_snapshot_roundtrip() -> None:
    """ExecutionContract 快照应能无损恢复关键治理与 prompt 信息。"""

    contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="wechat",
        host_policy=ExecutionHostPolicy(
            session_key="session-1",
            business_concurrency_lane=None,
            timeout_ms=None,
            resumable=True,
        ),
        preparation_spec=ScenePreparationSpec(
            selected_toolsets=("doc", "fins"),
            execution_permissions=ExecutionPermissions(
                web=ExecutionWebPermissions(allow_private_network_url=True),
                doc=ExecutionDocPermissions(
                    allowed_read_paths=("/tmp/a", "/tmp/b"),
                    allow_file_write=True,
                    allowed_write_paths=("/tmp/out",),
                ),
            ),
            prompt_contributions={
                "base_user": "# 用户与运行时上下文",
                "fins_default_subject": "# 当前分析对象\n你正在分析的是 AAPL。",
            },
        ),
        message_inputs=ExecutionMessageInputs(user_message="请继续"),
        accepted_execution_spec=AcceptedExecutionSpec(
            model=AcceptedModelSpec(model_name="gpt-test", temperature=0.2),
            runtime=AcceptedRuntimeSpec(
                runner_running_config={"tool_timeout_seconds": 12.0},
                agent_running_config={"max_iterations": 6},
            ),
            tools=AcceptedToolConfigSpec(
                toolset_configs=_build_toolset_configs(
                    doc_tool_limits=DocToolLimits(search_files_max_results=9),
                    fins_tool_limits=FinsToolLimits(list_documents_max_items=12),
                    web_tools_config=WebToolsConfig(provider="duckduckgo"),
                ),
            ),
            infrastructure=AcceptedInfrastructureSpec(
                trace_settings=TraceSettings(enabled=True, output_dir=Path("/tmp/trace")),
                conversation_memory_settings=ConversationMemorySettings(
                    compaction_scene_name="conversation_compaction"
                ),
            ),
        ),
        execution_options=ExecutionOptions(
            model_name="gpt-test",
            max_iterations=6,
            trace_output_dir=Path("/tmp/request-trace"),
            web_tools_config=WebToolsConfig(provider="duckduckgo"),
        ),
        metadata=ExecutionDeliveryContext({
            "delivery_channel": "wechat",
            "delivery_target": "user-1",
        }),
    )

    snapshot = serialize_execution_contract_snapshot(contract)
    restored = deserialize_execution_contract_snapshot(snapshot)

    assert restored.service_name == "chat_turn"
    assert restored.scene_name == "wechat"
    assert restored.host_policy == contract.host_policy
    assert restored.preparation_spec.selected_toolsets == ("doc", "fins")
    assert restored.preparation_spec.execution_permissions == contract.preparation_spec.execution_permissions
    assert restored.preparation_spec.prompt_contributions == contract.preparation_spec.prompt_contributions
    assert restored.message_inputs == contract.message_inputs
    assert restored.accepted_execution_spec.model.model_name == "gpt-test"
    assert restored.accepted_execution_spec.infrastructure.trace_settings == TraceSettings(
        enabled=True,
        output_dir=Path("/tmp/trace"),
    )
    assert restored.execution_options is not None
    assert contract.execution_options is not None
    assert restored.execution_options.toolset_configs == contract.execution_options.toolset_configs
    assert restored.execution_options.web_tools_config is None
    assert restored.metadata == contract.metadata


@pytest.mark.unit
def test_accepted_execution_spec_rejects_flat_compatibility_accessors() -> None:
    """AcceptedExecutionSpec 只暴露分组后的稳定契约，不再提供兼容平铺属性。"""

    spec = AcceptedExecutionSpec(
        model=AcceptedModelSpec(model_name="gpt-test", temperature=0.2),
        runtime=AcceptedRuntimeSpec(
            runner_running_config={"tool_timeout_seconds": 12.0},
            agent_running_config={"max_iterations": 6},
        ),
        infrastructure=AcceptedInfrastructureSpec(
            trace_settings=TraceSettings(enabled=True, output_dir=Path("/tmp/trace")),
            conversation_memory_settings=ConversationMemorySettings(
                compaction_scene_name="conversation_compaction"
            ),
        ),
    )

    assert spec.model.model_name == "gpt-test"
    assert spec.runtime.runner_running_config == {"tool_timeout_seconds": 12.0}
    assert spec.infrastructure.trace_settings == TraceSettings(
        enabled=True,
        output_dir=Path("/tmp/trace"),
    )

    with pytest.raises(AttributeError):
        _read_attribute(spec, "model_name")

    with pytest.raises(AttributeError):
        _read_attribute(spec, "temperature")

    with pytest.raises(AttributeError):
        _read_attribute(spec, "runner_running_config")

    with pytest.raises(AttributeError):
        _read_attribute(spec, "agent_running_config")

    with pytest.raises(AttributeError):
        _read_attribute(spec, "trace_settings")

    with pytest.raises(AttributeError):
        _read_attribute(spec, "conversation_memory_settings")


@pytest.mark.unit
def test_execution_contract_snapshot_rejects_removed_failed_tool_field() -> None:
    """旧字段名进入 execution option snapshot 时应显式失败。"""

    contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="wechat",
        host_policy=ExecutionHostPolicy(
            session_key="session-1",
            business_concurrency_lane=None,
            timeout_ms=None,
            resumable=True,
        ),
        preparation_spec=ScenePreparationSpec(
            selected_toolsets=(),
            execution_permissions=ExecutionPermissions(
                web=ExecutionWebPermissions(allow_private_network_url=False),
                doc=ExecutionDocPermissions(),
            ),
            prompt_contributions={},
        ),
        message_inputs=ExecutionMessageInputs(user_message="请继续"),
        accepted_execution_spec=AcceptedExecutionSpec(
            model=AcceptedModelSpec(model_name="gpt-test", temperature=0.2),
            runtime=AcceptedRuntimeSpec(agent_running_config={"max_iterations": 6}),
            tools=AcceptedToolConfigSpec(
                toolset_configs=_build_toolset_configs(
                    doc_tool_limits=DocToolLimits(),
                    fins_tool_limits=FinsToolLimits(),
                    web_tools_config=WebToolsConfig(provider="duckduckgo"),
                ),
            ),
            infrastructure=AcceptedInfrastructureSpec(
                trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp/trace")),
                conversation_memory_settings=ConversationMemorySettings(),
            ),
        ),
        execution_options=ExecutionOptions(model_name="gpt-test", max_iterations=6),
    )

    snapshot = serialize_execution_contract_snapshot(contract)
    execution_options_payload = dict(cast(ExecutionOptionsSnapshot, snapshot["execution_options"]))
    execution_options_payload.pop("max_iterations", None)
    execution_options_payload["max_consecutive_tool_failures"] = 2
    snapshot = {**snapshot, "execution_options": execution_options_payload}

    with pytest.raises(ValueError, match="removed fields"):
        deserialize_execution_contract_snapshot(snapshot)


@pytest.mark.unit
def test_execution_contract_snapshot_roundtrip_toolset_config_overrides() -> None:
    """ExecutionOptions 的 toolset_config_overrides 快照应能无损恢复。"""

    contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="wechat",
        host_policy=ExecutionHostPolicy(
            session_key="session-1",
            business_concurrency_lane=None,
            timeout_ms=None,
            resumable=True,
        ),
        preparation_spec=ScenePreparationSpec(
            selected_toolsets=(),
            execution_permissions=ExecutionPermissions(
                web=ExecutionWebPermissions(allow_private_network_url=False),
                doc=ExecutionDocPermissions(),
            ),
            prompt_contributions={},
        ),
        message_inputs=ExecutionMessageInputs(user_message="请继续"),
        accepted_execution_spec=AcceptedExecutionSpec(
            model=AcceptedModelSpec(model_name="gpt-test", temperature=0.2),
            runtime=AcceptedRuntimeSpec(agent_running_config={"max_iterations": 6}),
            tools=AcceptedToolConfigSpec(
                toolset_configs=_build_toolset_configs(
                    doc_tool_limits=DocToolLimits(),
                    fins_tool_limits=FinsToolLimits(),
                    web_tools_config=WebToolsConfig(provider="duckduckgo"),
                ),
            ),
            infrastructure=AcceptedInfrastructureSpec(
                trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp/trace")),
                conversation_memory_settings=ConversationMemorySettings(),
            ),
        ),
        execution_options=ExecutionOptions(
            toolset_config_overrides=(
                ToolsetConfigSnapshot(
                    toolset_name="doc",
                    payload={"search_files_max_results": 9},
                ),
                ToolsetConfigSnapshot(
                    toolset_name="fins",
                    payload={"list_documents_max_items": 12},
                ),
                ToolsetConfigSnapshot(
                    toolset_name="web",
                    payload={"provider": "off", "max_search_results": 3},
                ),
            ),
        ),
    )

    snapshot = serialize_execution_contract_snapshot(contract)
    restored = deserialize_execution_contract_snapshot(snapshot)

    assert restored.execution_options is not None
    assert restored.execution_options.toolset_config_overrides == (
        ToolsetConfigSnapshot(toolset_name="doc", version="1", payload={"search_files_max_results": 9}),
        ToolsetConfigSnapshot(toolset_name="fins", version="1", payload={"list_documents_max_items": 12}),
        ToolsetConfigSnapshot(
            toolset_name="web",
            version="1",
            payload={"provider": "off", "max_search_results": 3},
        ),
    )


@pytest.mark.unit
def test_execution_contract_snapshot_uses_trace_settings_dataclass_defaults_when_missing_optional_fields() -> None:
    """缺失 trace 可选字段时应回退到 `TraceSettings` dataclass 默认值。"""

    contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(
            session_key="session-1",
            business_concurrency_lane=None,
            timeout_ms=None,
            resumable=True,
        ),
        preparation_spec=ScenePreparationSpec(
            selected_toolsets=(),
            execution_permissions=ExecutionPermissions(
                web=ExecutionWebPermissions(allow_private_network_url=False),
                doc=ExecutionDocPermissions(),
            ),
            prompt_contributions={},
        ),
        message_inputs=ExecutionMessageInputs(user_message="hello"),
        accepted_execution_spec=AcceptedExecutionSpec(
            model=AcceptedModelSpec(model_name="gpt-test", temperature=0.2),
            runtime=AcceptedRuntimeSpec(agent_running_config={"max_iterations": 6}),
            tools=AcceptedToolConfigSpec(toolset_configs=()),
            infrastructure=AcceptedInfrastructureSpec(
                trace_settings=TraceSettings(
                    enabled=True,
                    output_dir=Path("/tmp/trace"),
                    max_file_bytes=0,
                    retention_days=0,
                    compress_rolled=False,
                    partition_by_session=False,
                ),
                conversation_memory_settings=ConversationMemorySettings(),
            ),
        ),
        execution_options=ExecutionOptions(),
    )

    snapshot = serialize_execution_contract_snapshot(contract)
    accepted_execution_spec_payload = cast(
        dict[str, ExecutionContractSnapshotValue],
        snapshot["accepted_execution_spec"],
    )
    infrastructure_payload = dict(
        cast(dict[str, ExecutionContractSnapshotValue], accepted_execution_spec_payload["infrastructure"])
    )
    trace_settings_payload = dict(
        cast(dict[str, ExecutionContractSnapshotValue], infrastructure_payload["trace_settings"])
    )
    for field_name in ("max_file_bytes", "retention_days", "compress_rolled", "partition_by_session"):
        trace_settings_payload.pop(field_name, None)
    infrastructure_payload["trace_settings"] = trace_settings_payload
    accepted_execution_spec_payload["infrastructure"] = infrastructure_payload
    snapshot["accepted_execution_spec"] = accepted_execution_spec_payload

    restored = deserialize_execution_contract_snapshot(snapshot)

    assert restored.accepted_execution_spec.infrastructure.trace_settings == TraceSettings(
        enabled=True,
        output_dir=Path("/tmp/trace"),
    )
    assert restored.execution_options is not None
    assert restored.execution_options.doc_tool_limits is None
    assert restored.execution_options.fins_tool_limits is None
    assert restored.execution_options.web_tools_config is None


@pytest.mark.unit
def test_execution_contract_snapshot_drops_trace_settings_when_output_dir_missing() -> None:
    """缺失 trace output_dir 时不应恢复为当前工作目录。"""

    contract = ExecutionContract(
        service_name="chat_turn",
        scene_name="interactive",
        host_policy=ExecutionHostPolicy(
            session_key="session-1",
            business_concurrency_lane=None,
            timeout_ms=None,
            resumable=True,
        ),
        preparation_spec=ScenePreparationSpec(
            selected_toolsets=(),
            execution_permissions=ExecutionPermissions(
                web=ExecutionWebPermissions(allow_private_network_url=False),
                doc=ExecutionDocPermissions(),
            ),
            prompt_contributions={},
        ),
        message_inputs=ExecutionMessageInputs(user_message="hello"),
        accepted_execution_spec=AcceptedExecutionSpec(
            model=AcceptedModelSpec(model_name="gpt-test", temperature=0.2),
            runtime=AcceptedRuntimeSpec(agent_running_config={"max_iterations": 6}),
            tools=AcceptedToolConfigSpec(toolset_configs=()),
            infrastructure=AcceptedInfrastructureSpec(
                trace_settings=TraceSettings(enabled=True, output_dir=Path("/tmp/trace")),
                conversation_memory_settings=ConversationMemorySettings(),
            ),
        ),
        execution_options=ExecutionOptions(),
    )

    snapshot = serialize_execution_contract_snapshot(contract)
    accepted_execution_spec_payload = cast(
        dict[str, ExecutionContractSnapshotValue],
        snapshot["accepted_execution_spec"],
    )
    infrastructure_payload = dict(
        cast(dict[str, ExecutionContractSnapshotValue], accepted_execution_spec_payload["infrastructure"])
    )
    trace_settings_payload = dict(
        cast(dict[str, ExecutionContractSnapshotValue], infrastructure_payload["trace_settings"])
    )
    trace_settings_payload.pop("output_dir", None)
    infrastructure_payload["trace_settings"] = trace_settings_payload
    accepted_execution_spec_payload["infrastructure"] = infrastructure_payload
    snapshot["accepted_execution_spec"] = accepted_execution_spec_payload

    restored = deserialize_execution_contract_snapshot(snapshot)

    assert restored.accepted_execution_spec.infrastructure.trace_settings is None


@pytest.mark.unit
def test_snapshot_optional_float_rejects_bool_uniformly() -> None:
    """三个模块的 ``_snapshot_optional_float`` 遇到 bool 均应统一 raise。

    动机：
        bool 属于 ``int`` 的子类，但语义上不是数字。若一处 return None、一处 raise，
        同一份快照数据在不同反序列化路径会出现不一致的行为。本用例锁定"统一 raise"
        契约，防止任何一处退化回静默降级。
    """

    from dayu.contracts.agent_execution_serialization import (
        _snapshot_optional_float as _contract_snapshot_optional_float,
    )
    from dayu.contracts.execution_options import (
        _snapshot_optional_float as _options_contract_snapshot_optional_float,
    )
    from dayu.execution.options import (
        _snapshot_optional_float as _execution_snapshot_optional_float,
    )

    for fn in (
        _contract_snapshot_optional_float,
        _options_contract_snapshot_optional_float,
        _execution_snapshot_optional_float,
    ):
        with pytest.raises(ValueError):
            fn(True)
        with pytest.raises(ValueError):
            fn(False)
        assert fn(None) is None
        assert fn(1.5) == 1.5
        assert fn(3) == 3.0
