"""写作服务实现。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from dayu.contracts.session import SessionSource
from dayu.execution.options import ExecutionOptions
from dayu.contracts.host_execution import HostedRunSpec
from dayu.host.protocols import HostedExecutionGatewayProtocol, HostGovernanceProtocol
from dayu.services.concurrency_lanes import resolve_hosted_run_concurrency_lane
from dayu.services.contracts import WriteRequest
from dayu.services.internal.write_pipeline.enums import (
    AUDIT_WRITE_SCENES,
    PRIMARY_MODEL_WRITE_SCENES,
    WriteSceneName,
)
from dayu.services.internal.write_pipeline.execution_options import build_execution_options_with_model_override
from dayu.services.contracts import SceneModelConfig, WriteRunConfig
from dayu.services.internal.write_pipeline.pipeline import print_write_report, run_write_pipeline
from dayu.services.protocols import WriteServiceProtocol
from dayu.services.scene_execution_acceptance import SceneExecutionAcceptancePreparer
from dayu.contracts.infrastructure import WorkspaceResourcesProtocol


WRITE_CANCELLED_EXIT_CODE = 130


@dataclass
class WriteService(WriteServiceProtocol):
    """写作服务。"""

    host: HostedExecutionGatewayProtocol
    host_governance: HostGovernanceProtocol
    workspace: WorkspaceResourcesProtocol
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer
    company_name_resolver: Callable[[str], str] | None = None
    company_meta_summary_resolver: Callable[[str], dict[str, str]] | None = None

    def run(self, request: WriteRequest) -> int:
        """执行写作流水线。

        Args:
            request: 写作执行请求。

        Returns:
            写作流程退出码；若宿主在同步执行阶段收口为取消，则返回显式取消退出码。

        Raises:
            RuntimeError: 写作流水线主体抛出的运行时异常会继续向上传播。
        """

        session = self.host.create_session(SessionSource.API)
        spec = HostedRunSpec(
            operation_name="write_pipeline",
            session_id=session.session_id,
            scene_name=WriteSceneName.WRITE,
            business_concurrency_lane=resolve_hosted_run_concurrency_lane("write_pipeline"),
        )
        return self.host.run_operation_sync(
            spec=spec,
            operation=lambda _context: self._run_pipeline(request, host_session_id=session.session_id),
            on_cancel=lambda: WRITE_CANCELLED_EXIT_CODE,
        )

    def _run_pipeline(self, request: WriteRequest, *, host_session_id: str) -> int:
        """执行写作流水线主体。"""

        main_execution_options = build_execution_options_with_model_override(
            execution_options=request.execution_options,
            model_name=request.write_config.write_model_override_name,
        )
        audit_execution_options = build_execution_options_with_model_override(
            execution_options=request.execution_options,
            model_name=request.write_config.audit_model_override_name,
        )
        resolved_options = self.scene_execution_acceptance_preparer.resolve_execution_options(
            WriteSceneName.WRITE,
            main_execution_options,
        )
        scene_models = _resolve_write_scene_models(
            scene_execution_acceptance_preparer=self.scene_execution_acceptance_preparer,
            main_execution_options=main_execution_options,
            audit_execution_options=audit_execution_options,
        )
        write_config = replace(request.write_config, scene_models=scene_models)

        return run_write_pipeline(
            workspace=self.workspace,
            resolved_options=resolved_options,
            write_config=write_config,
            scene_execution_acceptance_preparer=self.scene_execution_acceptance_preparer,
            host_executor=self.host,
            host_governance=self.host_governance,
            host_session_id=host_session_id,
            execution_options=main_execution_options,
            company_name_resolver=self.company_name_resolver,
            company_meta_summary_resolver=self.company_meta_summary_resolver,
        )

    @staticmethod
    def print_report(output_dir: str | Path) -> int:
        """打印写作流水线报告。"""

        return print_write_report(output_dir)


def _resolve_write_scene_models(
    *,
    scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer,
    main_execution_options: ExecutionOptions | None,
    audit_execution_options: ExecutionOptions | None,
) -> dict[str, SceneModelConfig]:
    """解析写作流水线各 scene 的实际模型配置。"""

    scene_models: dict[str, SceneModelConfig] = {}
    for scene_name in PRIMARY_MODEL_WRITE_SCENES:
        scene_models[scene_name] = scene_execution_acceptance_preparer.resolve_scene_model(
            scene_name,
            main_execution_options,
        )
    for scene_name in AUDIT_WRITE_SCENES:
        scene_models[scene_name] = scene_execution_acceptance_preparer.resolve_scene_model(
            scene_name,
            audit_execution_options,
        )
    return scene_models


__all__ = ["WRITE_CANCELLED_EXIT_CODE", "WriteRunConfig", "WriteService"]
