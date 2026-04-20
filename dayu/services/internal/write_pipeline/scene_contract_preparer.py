"""Scene 契约准备模块。

该模块负责写作流水线中 Scene 的 Service 层准备工作：
- 按需创建与缓存 AcceptedSceneExecution。
- 为每个 scene 调用构造 ExecutionContract。
- 管理 prompt contributions 和执行选项构建。

不持有任何 Host 层引用，仅产出契约数据。
"""

from __future__ import annotations

import os
import threading
from typing import Callable

from dayu.contracts.agent_execution import ExecutionContract
from dayu.contracts.infrastructure import ConfigLoaderProtocol
from dayu.execution.options import ExecutionOptions
from dayu.services.contracts import WriteRunConfig
from dayu.services.contract_preparation import prepare_execution_contract
from dayu.services.internal.write_pipeline.execution_options import (
    build_execution_options_with_scene_overrides,
)
from dayu.services.internal.write_pipeline.enums import AUDIT_WRITE_SCENES, WriteSceneName
from dayu.services.prompt_contributions import (
    build_base_user_contribution,
    build_fins_default_subject_contribution,
)
from dayu.services.scene_execution_acceptance import (
    AcceptedSceneExecution,
    SceneExecutionAcceptancePreparer,
)

class SceneAgentCreationError(RuntimeError):
    """scene Agent 创建失败异常。"""

    def __init__(self, *, scene_name: str, agent_label: str, detail: str = "") -> None:
        """初始化 scene Agent 创建失败异常。

        Args:
            scene_name: scene 名称。
            agent_label: 面向日志的人类可读 Agent 名称。
            detail: 失败细节。
        """

        message = f"{agent_label} 创建失败: scene={scene_name}"
        if detail:
            message = f"{message}, error={detail}"
        super().__init__(message)
        self.scene_name = scene_name
        self.agent_label = agent_label
        self.detail = detail


class SceneContractPreparer:
    """Scene 契约准备器（纯 Service 层）。

    职责：
    - 按需创建与缓存 Scene 执行信息（AcceptedSceneExecution）。
    - 为每个 scene 调用构造 ExecutionContract。
    - 管理 prompt contributions 和执行选项的构建。

    不持有 Host 层引用；仅产出契约数据供执行侧消费。
    """

    def __init__(
        self,
        *,
        scene_execution_acceptance_preparer: SceneExecutionAcceptancePreparer,
        write_config: WriteRunConfig,
        execution_options: ExecutionOptions | None = None,
        host_session_id: str = "",
        company_name_resolver: Callable[[str], str] | None = None,
        company_facet_catalog: dict[str, list[str]] | None = None,
        config_loader: ConfigLoaderProtocol | None = None,
    ) -> None:
        """初始化 Scene 契约准备器。

        Args:
            scene_execution_acceptance_preparer: scene 执行接受准备器。
            write_config: 写作运行配置。
            execution_options: 请求级覆盖参数。
            host_session_id: 当前写作流水线复用的 Host Session。
            company_name_resolver: 可选公司名称解析函数。
            company_facet_catalog: 公司级 facet 目录。
            config_loader: 可选配置加载器，用于按 scene 惰性校验模型环境变量。
        """

        self._scene_execution_acceptance_preparer = scene_execution_acceptance_preparer
        self._write_config = write_config
        self._execution_options = execution_options
        self._host_session_id = host_session_id
        self._company_name_resolver = company_name_resolver
        self._company_facet_catalog = company_facet_catalog or {}
        self._config_loader = config_loader
        self._resolved_scene_executions: dict[str, AcceptedSceneExecution] = {}
        self._validated_model_names: set[str] = set()
        self._resolved_scene_lock = threading.Lock()

    def set_company_facet_catalog(self, company_facet_catalog: dict[str, list[str]]) -> None:
        """设置当前模板声明的公司级 facet 候选目录。

        Args:
            company_facet_catalog: facet 候选目录映射。

        Returns:
            无。

        Raises:
            无。
        """

        self._company_facet_catalog = dict(company_facet_catalog)

    def get_company_facet_catalog(self) -> dict[str, list[str]]:
        """返回当前模板声明的公司级 facet 候选目录。

        Args:
            无。

        Returns:
            facet 候选目录映射副本。

        Raises:
            无。
        """

        return dict(self._company_facet_catalog)

    def get_or_create_write_scene(self) -> AcceptedSceneExecution:
        """获取或创建写作 scene 的已解析执行信息。"""

        return self.get_or_create_prepared_scene(
            scene_name=WriteSceneName.WRITE,
            agent_label="写作 Agent",
            create_agent=self._create_write_agent,
        )

    def get_or_create_infer_scene(self) -> AcceptedSceneExecution:
        """获取或创建 infer scene 的已解析执行信息。"""

        return self.get_or_create_prepared_scene(
            scene_name=WriteSceneName.INFER,
            agent_label="公司级 Facet 归因 Agent",
            create_agent=self._create_infer_agent,
        )

    def get_or_create_decision_scene(self) -> AcceptedSceneExecution:
        """获取或创建 decision scene 的已解析执行信息。"""

        return self.get_or_create_prepared_scene(
            scene_name=WriteSceneName.DECISION,
            agent_label="研究决策综合 Agent",
            create_agent=self._create_decision_agent,
        )

    def get_or_create_overview_scene(self) -> AcceptedSceneExecution:
        """获取或创建 overview scene 的已解析执行信息。"""

        return self.get_or_create_prepared_scene(
            scene_name=WriteSceneName.OVERVIEW,
            agent_label="第0章概览 Agent",
            create_agent=self._create_overview_agent,
        )

    def get_or_create_regenerate_scene(self) -> AcceptedSceneExecution:
        """获取或创建 regenerate scene 的已解析执行信息。"""

        return self.get_or_create_prepared_scene(
            scene_name=WriteSceneName.REGENERATE,
            agent_label="整章重建 Agent",
            create_agent=self._create_regenerate_agent,
        )

    def get_or_create_audit_scene(self) -> AcceptedSceneExecution:
        """获取或创建审计 scene 的已解析执行信息。

        Args:
            无。

        Returns:
            已解析的审计 scene 执行信息。

        Raises:
            SceneAgentCreationError: 当审计 scene 创建失败时抛出。
        """

        return self.get_or_create_prepared_scene(
            scene_name=WriteSceneName.AUDIT,
            agent_label="审计 Agent",
            create_agent=self._create_audit_agent,
        )

    def get_or_create_fix_scene(self) -> AcceptedSceneExecution:
        """获取或创建 fix scene 的已解析执行信息。"""

        return self.get_or_create_prepared_scene(
            scene_name=WriteSceneName.FIX,
            agent_label="占位符补强 Agent",
            create_agent=self._create_fix_agent,
        )

    def get_or_create_confirm_scene(self) -> AcceptedSceneExecution:
        """获取或创建 confirm scene 的已解析执行信息。"""

        return self.get_or_create_prepared_scene(
            scene_name=WriteSceneName.CONFIRM,
            agent_label="证据复核 Agent",
            create_agent=self._create_confirm_agent,
        )

    def get_or_create_repair_scene(self) -> AcceptedSceneExecution:
        """获取或创建 repair scene 的已解析执行信息。"""

        return self.get_or_create_prepared_scene(
            scene_name=WriteSceneName.REPAIR,
            agent_label="repair Agent",
            create_agent=self._create_repair_agent,
        )

    # ------------------------------------------------------------------
    # Scene 创建方法
    # ------------------------------------------------------------------

    def _create_write_agent(self) -> AcceptedSceneExecution:
        """解析写作 scene 的静态执行信息。"""

        return self._scene_execution_acceptance_preparer.prepare(
            WriteSceneName.WRITE,
            self._build_primary_scene_execution_options(),
        )

    def _create_infer_agent(self) -> AcceptedSceneExecution:
        """解析公司级 facet 归因 scene 的静态执行信息。"""

        return self._scene_execution_acceptance_preparer.prepare(
            WriteSceneName.INFER,
            self._build_audit_scene_execution_options(),
        )

    def _create_decision_agent(self) -> AcceptedSceneExecution:
        """解析研究决策综合 scene 的静态执行信息。"""

        return self._scene_execution_acceptance_preparer.prepare(
            WriteSceneName.DECISION,
            self._build_audit_scene_execution_options(),
        )

    def _create_overview_agent(self) -> AcceptedSceneExecution:
        """解析第0章概览 scene 的静态执行信息。"""

        return self._scene_execution_acceptance_preparer.prepare(
            WriteSceneName.OVERVIEW,
            self._build_primary_scene_execution_options(),
        )

    def _create_regenerate_agent(self) -> AcceptedSceneExecution:
        """解析整章重建 scene 的静态执行信息。"""

        return self._scene_execution_acceptance_preparer.prepare(
            WriteSceneName.REGENERATE,
            self._build_primary_scene_execution_options(),
        )

    def _create_audit_agent(self) -> AcceptedSceneExecution:
        """解析审计 scene 的静态执行信息。"""

        return self._scene_execution_acceptance_preparer.prepare(
            WriteSceneName.AUDIT,
            self._build_audit_scene_execution_options(),
        )

    def _create_fix_agent(self) -> AcceptedSceneExecution:
        """解析占位符补强 scene 的静态执行信息。"""

        return self._scene_execution_acceptance_preparer.prepare(
            WriteSceneName.FIX,
            self._build_primary_scene_execution_options(),
        )

    def _create_confirm_agent(self) -> AcceptedSceneExecution:
        """解析证据复核 scene 的静态执行信息。"""

        return self._scene_execution_acceptance_preparer.prepare(
            WriteSceneName.CONFIRM,
            self._build_audit_scene_execution_options(),
        )

    def _create_repair_agent(self) -> AcceptedSceneExecution:
        """解析 repair scene 的静态执行信息。"""

        return self._scene_execution_acceptance_preparer.prepare(
            WriteSceneName.REPAIR,
            self._build_primary_scene_execution_options(),
        )

    # ------------------------------------------------------------------
    # 缓存与查找
    # ------------------------------------------------------------------

    def get_or_create_prepared_scene(
        self,
        *,
        scene_name: str,
        agent_label: str,
        create_agent: Callable[[], AcceptedSceneExecution],
    ) -> AcceptedSceneExecution:
        """获取或延迟解析 scene 执行信息。

        Args:
            scene_name: scene 名称。
            agent_label: 面向日志的人类可读 Agent 名称。
            create_agent: 延迟创建回调。

        Returns:
            已解析的 AcceptedSceneExecution。

        Raises:
            SceneAgentCreationError: 当创建回调返回 None 时抛出。
        """

        with self._resolved_scene_lock:
            cached_scene = self._resolved_scene_executions.get(scene_name)
            if cached_scene is not None:
                return cached_scene

            self._ensure_scene_environment_ready(scene_name=scene_name, agent_label=agent_label)

            resolved_scene = create_agent()
            if resolved_scene is None:
                raise SceneAgentCreationError(scene_name=scene_name, agent_label=agent_label)
            self._resolved_scene_executions[scene_name] = resolved_scene
            return resolved_scene

    # ------------------------------------------------------------------
    # ExecutionContract 构建
    # ------------------------------------------------------------------

    def build_execution_contract(
        self,
        *,
        prepared_scene: AcceptedSceneExecution,
        prompt_text: str,
    ) -> ExecutionContract:
        """为写作流水线中的单个 scene 调用构造 ExecutionContract。

        Args:
            prepared_scene: 已解析的 scene 执行信息。
            prompt_text: 当前轮用户输入。

        Returns:
            单个 Agent 子执行契约。

        Raises:
            ValueError: 当 prompt 文本为空时抛出。
        """

        user_message = str(prompt_text or "").strip()
        if not user_message:
            raise ValueError("写作流水线 prompt 不能为空")
        prompt_contributions = self._build_prompt_contributions()
        return prepare_execution_contract(
            service_name="write_pipeline",
            scene_name=prepared_scene.scene_name,
            accepted_execution_spec=prepared_scene.accepted_execution_spec,
            prompt_contributions=prompt_contributions,
            context_slots=prepared_scene.scene_definition.context_slots,
            selected_toolsets=(),
            user_message=user_message,
            session_key=self._host_session_id,
            concurrency_lane="llm_api",
            execution_options=self._build_execution_options_for_scene(prepared_scene.scene_name),
            timeout_ms=None,
            resumable=prepared_scene.default_resumable,
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_prompt_contributions(self) -> dict[str, str]:
        """构建写作流水线公共 Prompt Contributions。"""

        company_name = str(self._write_config.company or "").strip()
        if not company_name and self._company_name_resolver is not None:
            company_name = str(self._company_name_resolver(self._write_config.ticker) or "").strip()
        prompt_contributions = {
            "base_user": build_base_user_contribution(),
        }
        subject_text = build_fins_default_subject_contribution(
            ticker=self._write_config.ticker,
            company_name=company_name,
        )
        if subject_text:
            prompt_contributions["fins_default_subject"] = subject_text
        return prompt_contributions

    def _build_execution_options_for_scene(self, scene_name: str) -> ExecutionOptions | None:
        """按 scene 名称返回对应的执行选项。"""

        if scene_name in AUDIT_WRITE_SCENES:
            return self._build_audit_scene_execution_options()
        return self._build_primary_scene_execution_options()

    def _build_primary_scene_execution_options(self) -> ExecutionOptions | None:
        """构建主写作 scene 的执行选项。

        Returns:
            主写作 scene 使用的执行选项。
        """

        return build_execution_options_with_scene_overrides(
            execution_options=self._execution_options,
            model_name=self._write_config.write_model_override_name,
            web_provider=self._write_config.web_provider,
        )

    def _build_audit_scene_execution_options(self) -> ExecutionOptions | None:
        """构建审计 scene 的执行选项。

        Returns:
            审计 scene 使用的执行选项。
        """

        return build_execution_options_with_scene_overrides(
            execution_options=self._execution_options,
            model_name=self._write_config.audit_model_override_name,
            web_provider=None,
        )

    def _ensure_scene_environment_ready(self, *, scene_name: str, agent_label: str) -> None:
        """在首次创建 scene 前校验当前模型所需环境变量。

        Args:
            scene_name: 即将创建的 scene 名称。
            agent_label: 人类可读 Agent 名称。

        Returns:
            无。

        Raises:
            SceneAgentCreationError: 当前 scene 所需模型缺少环境变量时抛出。
        """

        if self._config_loader is None:
            return

        resolved_options = self._scene_execution_acceptance_preparer.resolve_execution_options(
            scene_name,
            self._build_execution_options_for_scene(scene_name),
        )
        model_name = str(resolved_options.model_name or "").strip()
        if not model_name or model_name in self._validated_model_names:
            return

        required_env_vars = self._config_loader.collect_model_referenced_env_vars((model_name,))
        missing_env_vars = tuple(
            env_var_name
            for env_var_name in required_env_vars
            if not str(os.environ.get(env_var_name) or "").strip()
        )
        if missing_env_vars:
            raise SceneAgentCreationError(
                scene_name=scene_name,
                agent_label=agent_label,
                detail=(
                    f"模型 {model_name} 缺少环境变量: {', '.join(missing_env_vars)}"
                ),
            )

        self._validated_model_names.add(model_name)
