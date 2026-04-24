"""WriteService 测试。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, cast, overload

import pytest

from dayu.contracts.infrastructure import ModelCatalogProtocol
from dayu.contracts.model_config import ModelConfig
from dayu.contracts.prompt_assets import SceneManifestAsset, TaskPromptContractAsset
from dayu.contracts.session import SessionSource
from dayu.execution.options import ExecutionOptions
from dayu.execution.options import ResolvedExecutionOptions, build_base_execution_options
from dayu.host.host import Host
from dayu.host.host_execution import HostedRunContext, HostedRunSpec
from dayu.host.protocols import HostedExecutionGatewayProtocol, HostGovernanceProtocol
from dayu.host.protocols import RunRegistryProtocol
from dayu.services.conversation_policy_reader import ConversationPolicyReader
from dayu.services.scene_definition_reader import SceneDefinitionReader
from dayu.services.contracts import SceneModelConfig, WriteRequest, WriteRunConfig
from dayu.services.scene_execution_acceptance import SceneExecutionAcceptancePreparer
from dayu.services.write_service import WRITE_CANCELLED_EXIT_CODE, WriteService
from dayu.services.protocols import WriteServiceProtocol
from dayu.startup.workspace import WorkspaceResources


class _FakeCompanyMetaProvider:
    """测试用公司 meta provider。"""

    def get_company_name(self, ticker: str) -> str:
        return f"{ticker}-NAME"

    def get_company_meta_summary(self, ticker: str) -> dict[str, str]:
        return {"ticker": ticker, "company_name": f"{ticker}-NAME"}


@dataclass(frozen=True)
class _FakeCreatedSession:
    """测试用已创建 Host session。"""

    session_id: str


class _CancellingHostedGateway:
    """测试用宿主网关，模拟同步路径在 Host 内部收口为取消退出码。"""

    def __init__(self) -> None:
        """初始化测试网关。"""

        self.last_spec: HostedRunSpec | None = None
        self.sync_call_count = 0

    def create_session(self, source: SessionSource) -> _FakeCreatedSession:
        """返回固定 session。

        Args:
            source: session 来源。

        Returns:
            固定 session 记录。

        Raises:
            无。
        """

        del source
        return _FakeCreatedSession(session_id="cancelled-session")

    def run_operation_sync(
        self,
        *,
        spec: HostedRunSpec,
        operation: Callable[[HostedRunContext], int],
        on_cancel: Callable[[], int] | None = None,
    ) -> int:
        """直接走 Host 取消收口回调。

        Args:
            spec: 托管同步执行规格。
            operation: 同步执行体。
            on_cancel: 取消收口回调。

        Returns:
            取消退出码。

        Raises:
            AssertionError: 未提供取消回调时抛出。
        """

        del operation
        self.last_spec = spec
        self.sync_call_count += 1
        if on_cancel is None:
            raise AssertionError("测试前提不成立：缺少 on_cancel 回调")
        return on_cancel()


class _FakeConfigLoader:
    """测试用配置加载器。"""

    def load_run_config(self) -> dict[str, Any]:
        """返回最小 run 配置。"""

        return {}

    def load_llm_models(self) -> dict[str, ModelConfig]:
        """返回空模型表。"""

        return {}

    def load_llm_model(self, model_name: str) -> ModelConfig:
        """返回最小模型配置。"""

        return {"model": model_name, "max_context_tokens": 32000}

    def load_toolset_registrars(self) -> dict[str, str]:
        """返回空 registrar 配置。"""

        return {}

    def collect_model_referenced_env_vars(self, model_names: Iterable[str]) -> tuple[str, ...]:
        """返回指定模型引用的环境变量。"""

        del model_names
        return ()


class _FakePromptAssetStore:
    """测试用 prompt 资产仓储。"""

    def load_scene_manifest(self, scene_name: str) -> SceneManifestAsset:
        """当前测试不应读取 scene manifest。"""

        return {
            "scene": scene_name,
            "model": {
                "default_name": "test-model",
                "allowed_names": ["test-model"],
                "temperature_profile": "default",
            },
            "runtime": {
                "agent": {"max_iterations": 4},
                "runner": {"tool_timeout_seconds": 90.0},
            },
            "fragments": [],
            "context_slots": [],
            "tool_selection": {"mode": "allow_all"},
        }

    @overload
    def load_fragment_template(self, fragment_path: str, *, required: Literal[True] = True) -> str:
        ...

    @overload
    def load_fragment_template(self, fragment_path: str, *, required: Literal[False]) -> str | None:
        ...

    def load_fragment_template(self, fragment_path: str, *, required: bool = True) -> str | None:
        """当前测试不应读取 fragment。"""

        del fragment_path, required
        return None

    def load_task_prompt(self, task_name: str) -> str:
        """当前测试不应读取 task prompt。"""

        raise AssertionError(f"当前测试不应读取 task prompt: {task_name}")

    def load_task_prompt_contract(self, task_name: str) -> TaskPromptContractAsset:
        """当前测试不应读取 task contract。"""

        return {
            "prompt_name": task_name,
            "version": "1",
            "inputs": [],
        }


class _FakeModelCatalog(ModelCatalogProtocol):
    """测试用模型目录。"""

    def load_model(self, model_name: str) -> ModelConfig:
        """返回最小模型配置。"""

        return {"model": model_name, "max_context_tokens": 32000}

    def load_models(self) -> dict[str, ModelConfig]:
        """返回空模型表。"""

        return {}


class _FakeSceneDefinitionReader(SceneDefinitionReader):
    """测试用 scene reader。"""

    def __init__(self) -> None:
        """初始化空 reader。"""

        pass

    def read(self, scene_name: str):
        """当前测试不会调用 read。"""

        raise AssertionError(f"当前测试不应读取 scene 定义: {scene_name}")


class _FakeConversationPolicyReader(ConversationPolicyReader):
    """测试用 conversation policy reader。"""

    def __init__(self) -> None:
        """初始化空 reader。"""

        pass

    def resolve(self, *, resolved_execution_options: ResolvedExecutionOptions, model_config: ModelConfig | None):
        """返回当前 resolved options 自带的 memory settings。"""

        del model_config
        return resolved_execution_options.conversation_memory_settings


class _FakeSceneExecutionAcceptancePreparer(SceneExecutionAcceptancePreparer):
    """测试用 scene 执行接受器。"""

    def __init__(self, *, workspace_dir: Path) -> None:
        """构造只覆盖被测方法的最小 preparer。"""

        super().__init__(
            workspace_dir=workspace_dir,
            base_execution_options=build_base_execution_options(workspace_dir=workspace_dir, run_config={}),
            model_catalog=_FakeModelCatalog(),
            scene_definition_reader=_FakeSceneDefinitionReader(),
            conversation_policy_reader=_FakeConversationPolicyReader(),
        )

    def resolve_execution_options(
        self,
        scene_name: str,
        execution_options: ExecutionOptions | None = None,
    ) -> ResolvedExecutionOptions:
        """返回最小可用的 resolved execution options。"""

        del scene_name, execution_options
        return build_base_execution_options(workspace_dir=self.workspace_dir, run_config={})

    def resolve_scene_model(
        self,
        scene_name: str,
        execution_options: ExecutionOptions | None = None,
    ) -> SceneModelConfig:
        """返回稳定 scene model 摘要。"""

        del execution_options
        return SceneModelConfig(name=f"model-{scene_name}", temperature=0.2)


def _build_workspace(workspace_dir: Path) -> WorkspaceResources:
    """构造最小 WorkspaceResources。"""

    return WorkspaceResources(
        workspace_dir=workspace_dir,
        config_root=workspace_dir / "config",
        output_dir=workspace_dir / "output",
        config_loader=_FakeConfigLoader(),
        prompt_asset_store=_FakePromptAssetStore(),
    )


def _run_registry() -> RunRegistryProtocol:
    """构造满足 Host 依赖的 run registry。"""

    from tests.application.conftest import StubRunRegistry

    return StubRunRegistry()


def _build_request() -> WriteRequest:
    """构建最小写作请求。"""

    return WriteRequest(
        write_config=WriteRunConfig(
            ticker="AAPL",
            company="Apple",
            template_path="/tmp/template.md",
            output_dir="/tmp/output",
            write_max_retries=1,
            web_provider="off",
            resume=False,
        ),
    )


@pytest.mark.unit
def test_write_service_runs_pipeline_via_host_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """WriteService 应通过 host executor 托管同步执行。"""

    from tests.application.conftest import StubHostExecutor, StubSessionRegistry

    host_executor = StubHostExecutor()
    workspace = _build_workspace(Path("/tmp/dayu-write-service"))
    host = Host(
        executor=host_executor,
        session_registry=StubSessionRegistry(),
        run_registry=_run_registry(),
    )
    service = WriteService(
        host=host,
        host_governance=host,
        workspace=workspace,
        scene_execution_acceptance_preparer=_FakeSceneExecutionAcceptancePreparer(
            workspace_dir=workspace.workspace_dir,
        ),
        company_name_resolver=lambda ticker: f"{ticker}-NAME",
        company_meta_summary_resolver=lambda ticker: {"ticker": ticker},
    )

    monkeypatch.setattr(
        "dayu.services.write_service.run_write_pipeline",
        lambda **kwargs: 0,
    )

    result = service.run(_build_request())

    assert result == 0
    assert host_executor.sync_call_count == 1
    assert host_executor.last_spec is not None
    assert host_executor.last_spec.operation_name == "write_pipeline"
    assert host_executor.last_spec.business_concurrency_lane == "write_chapter"
    assert host_executor.last_spec.metadata == {}

    assert isinstance(service, WriteServiceProtocol)


@pytest.mark.unit
def test_write_service_resolves_overview_scene_with_primary_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """WriteService 应为 overview scene 解析主写作模型配置。"""

    from tests.application.conftest import StubHostExecutor, StubSessionRegistry

    workspace = _build_workspace(Path("/tmp/dayu-write-service-overview"))
    host = Host(
        executor=StubHostExecutor(),
        session_registry=StubSessionRegistry(),
        run_registry=_run_registry(),
    )
    scene_queries: list[tuple[str, ExecutionOptions | None]] = []
    captured_scene_models: dict[str, SceneModelConfig] = {}

    class _CapturingSceneExecutionAcceptancePreparer(_FakeSceneExecutionAcceptancePreparer):
        """记录 scene model 查询的测试用 preparer。"""

        def resolve_scene_model(
            self,
            scene_name: str,
            execution_options: ExecutionOptions | None = None,
        ) -> SceneModelConfig:
            """记录查询并返回稳定 scene model 摘要。"""

            scene_queries.append((scene_name, execution_options))
            return SceneModelConfig(name=f"model-{scene_name}", temperature=0.2)

    service = WriteService(
        host=host,
        host_governance=host,
        workspace=workspace,
        scene_execution_acceptance_preparer=_CapturingSceneExecutionAcceptancePreparer(
            workspace_dir=workspace.workspace_dir,
        ),
    )

    def _capture_run_write_pipeline(**kwargs: object) -> int:
        write_config = kwargs["write_config"]
        assert isinstance(write_config, WriteRunConfig)
        captured_scene_models.update(write_config.scene_models)
        return 0

    monkeypatch.setattr("dayu.services.write_service.run_write_pipeline", _capture_run_write_pipeline)

    result = service.run(_build_request())

    assert result == 0
    assert "overview" in captured_scene_models
    assert captured_scene_models["overview"].name == "model-overview"
    assert any(scene_name == "overview" for scene_name, _execution_options in scene_queries)


@pytest.mark.unit
def test_write_service_returns_cancelled_exit_code_when_host_sync_path_is_cancelled() -> None:
    """WriteService 应在真实宿主同步取消收口路径上返回显式取消退出码。"""

    workspace = _build_workspace(Path("/tmp/dayu-write-service-cancelled"))
    host_gateway = _CancellingHostedGateway()
    service = WriteService(
        host=cast(HostedExecutionGatewayProtocol, host_gateway),
        host_governance=cast(HostGovernanceProtocol, host_gateway),
        workspace=workspace,
        scene_execution_acceptance_preparer=_FakeSceneExecutionAcceptancePreparer(
            workspace_dir=workspace.workspace_dir,
        ),
    )

    result = service.run(_build_request())

    assert result == WRITE_CANCELLED_EXIT_CODE
    assert host_gateway.sync_call_count == 1
    assert host_gateway.last_spec is not None
    assert host_gateway.last_spec.operation_name == "write_pipeline"
