"""Service 侧 ExecutionContract 机械装配模块。"""

from __future__ import annotations

from dayu.contracts.agent_execution import (
    AcceptedExecutionSpec,
    ExecutionContract,
    ExecutionDocPermissions,
    ExecutionHostPolicy,
    ExecutionMessageInputs,
    ExecutionPermissions,
    ExecutionWebPermissions,
    ScenePreparationSpec,
)
from dayu.contracts.execution_metadata import ExecutionDeliveryContext, normalize_execution_delivery_context
from dayu.execution.options import ExecutionOptions, resolve_web_tools_config_from_toolset_configs
from dayu.prompting.prompt_contribution_slots import select_prompt_contributions


def prepare_execution_permissions(
    *,
    accepted_execution_spec: AcceptedExecutionSpec,
    doc_permissions: ExecutionDocPermissions | None = None,
) -> ExecutionPermissions:
    """准备执行权限真源。"""

    web_tools_config = resolve_web_tools_config_from_toolset_configs(
        accepted_execution_spec.tools.toolset_configs
    )
    allow_private_network_url = False
    if web_tools_config is not None:
        allow_private_network_url = web_tools_config.allow_private_network_url
    return ExecutionPermissions(
        web=ExecutionWebPermissions(
            allow_private_network_url=allow_private_network_url,
        ),
        doc=doc_permissions or ExecutionDocPermissions(),
    )


def prepare_scene_preparation_spec(
    *,
    accepted_execution_spec: AcceptedExecutionSpec,
    prompt_contributions: dict[str, str],
    context_slots: tuple[str, ...] | None = None,
    selected_toolsets: tuple[str, ...] = (),
    execution_permissions: ExecutionPermissions | None = None,
) -> ScenePreparationSpec:
    """准备 scene preparation 装配说明。"""

    normalized_prompt_contributions = dict(prompt_contributions)
    if context_slots is not None:
        normalized_prompt_contributions = select_prompt_contributions(
            prompt_contributions=prompt_contributions,
            context_slots=context_slots,
        ).selected_contributions

    return ScenePreparationSpec(
        selected_toolsets=selected_toolsets,
        execution_permissions=execution_permissions
        or prepare_execution_permissions(accepted_execution_spec=accepted_execution_spec),
        prompt_contributions=normalized_prompt_contributions,
    )


def prepare_host_policy(
    *,
    session_key: str | None,
    business_concurrency_lane: str | None,
    timeout_ms: int | None = None,
    resumable: bool = False,
) -> ExecutionHostPolicy:
    """准备 Host 生命周期治理策略。"""

    return ExecutionHostPolicy(
        session_key=session_key,
        business_concurrency_lane=business_concurrency_lane,
        timeout_ms=timeout_ms,
        resumable=resumable,
    )


def prepare_message_inputs(
    *,
    user_message: str,
) -> ExecutionMessageInputs:
    """准备当前轮消息输入。"""

    normalized_user_message = str(user_message or "").strip()
    if not normalized_user_message:
        raise ValueError("当前轮 user_message 不能为空")
    return ExecutionMessageInputs(user_message=normalized_user_message)


def prepare_execution_contract(
    *,
    service_name: str,
    scene_name: str,
    accepted_execution_spec: AcceptedExecutionSpec,
    prompt_contributions: dict[str, str],
    context_slots: tuple[str, ...] | None = None,
    user_message: str,
    session_key: str | None,
    business_concurrency_lane: str | None,
    metadata: ExecutionDeliveryContext | None = None,
    selected_toolsets: tuple[str, ...] = (),
    execution_permissions: ExecutionPermissions | None = None,
    execution_options: ExecutionOptions | None = None,
    timeout_ms: int | None = None,
    resumable: bool = False,
) -> ExecutionContract:
    """准备完整 ExecutionContract。"""

    return ExecutionContract(
        service_name=service_name,
        scene_name=scene_name,
        host_policy=prepare_host_policy(
            session_key=session_key,
            business_concurrency_lane=business_concurrency_lane,
            timeout_ms=timeout_ms,
            resumable=resumable,
        ),
        preparation_spec=prepare_scene_preparation_spec(
            accepted_execution_spec=accepted_execution_spec,
            prompt_contributions=prompt_contributions,
            context_slots=context_slots,
            selected_toolsets=selected_toolsets,
            execution_permissions=execution_permissions,
        ),
        message_inputs=prepare_message_inputs(user_message=user_message),
        accepted_execution_spec=accepted_execution_spec,
        execution_options=execution_options,
        metadata=normalize_execution_delivery_context(metadata),
    )


__all__ = [
    "prepare_execution_contract",
    "prepare_execution_permissions",
    "prepare_host_policy",
    "prepare_message_inputs",
    "prepare_scene_preparation_spec",
]
