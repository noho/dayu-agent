"""Service 侧并发 lane 业务真源。

该模块收口 Service 层对"业务并发通道"的声明：

- ``LANE_WRITE_CHAPTER`` / ``LANE_SEC_DOWNLOAD``：业务 lane 名称常量。
- ``SERVICE_DEFAULT_LANE_CONFIG``：Service 启动期交给 Host 作为默认 lane 配置
  的一部分；Host 不感知这些业务语义。
- ``resolve_contract_concurrency_lane`` / ``resolve_hosted_run_concurrency_lane``：
  调用点用于把"scene / operation 名称"映射到业务 lane。

分层约束（CLAUDE.md 硬约束）：

- 模块**完全不出现** ``"llm_api"`` 字面量；``llm_api`` 属于 Host 自治 lane，
  由 Host 根据 ExecutionContract 的调用路径自动叠加，Service 不参与决策。
- 本模块不 import 任何 Host 私有常量，依赖方向仍是 Service → Host。
"""

from __future__ import annotations

from dayu.contracts.fins import FinsCommandName
from dayu.services.internal.write_pipeline.enums import WriteSceneName


LANE_WRITE_CHAPTER: str = "write_chapter"
"""章节写作业务 lane 名称。"""


LANE_SEC_DOWNLOAD: str = "sec_download"
"""SEC 原始文件下载业务 lane 名称。"""


SERVICE_DEFAULT_LANE_CONFIG: dict[str, int] = {
    LANE_WRITE_CHAPTER: 5,
    LANE_SEC_DOWNLOAD: 1,
}
"""Service 启动期交给 Host 的业务 lane 默认配置。

Host 会把该 dict 与自有 lane、``run.json.host_config.lane`` 以及显式覆盖
按既定优先级合并，不做任何业务判断。
"""


_WRITE_PIPELINE_SCENES: frozenset[str] = frozenset(scene.value for scene in WriteSceneName)


def resolve_contract_concurrency_lane(scene_name: str) -> str | None:
    """解析 ExecutionContract 的业务 lane。

    Args:
        scene_name: scene 名称。

    Returns:
        写作流水线 scene 返回 ``LANE_WRITE_CHAPTER``；其他场景返回 ``None``，
        由 Host 根据调用路径自动补齐自治 lane。

    Raises:
        无。
    """

    normalized = (scene_name or "").strip()
    if normalized in _WRITE_PIPELINE_SCENES:
        return LANE_WRITE_CHAPTER
    return None


def resolve_hosted_run_concurrency_lane(operation_name: str) -> str | None:
    """解析 HostedRunSpec 的业务 lane。

    Args:
        operation_name: Service 提交的操作名。

    Returns:
        - ``"write_pipeline"`` → ``LANE_WRITE_CHAPTER``
        - ``"fins_download"`` → ``LANE_SEC_DOWNLOAD``
        - 其他宿主操作 → ``None``，由 Host 根据调用路径自动补齐自治 lane。

    Raises:
        无。
    """

    normalized = (operation_name or "").strip()
    if normalized == "write_pipeline":
        return LANE_WRITE_CHAPTER
    if normalized == f"fins_{FinsCommandName.DOWNLOAD}":
        return LANE_SEC_DOWNLOAD
    return None


__all__ = [
    "LANE_SEC_DOWNLOAD",
    "LANE_WRITE_CHAPTER",
    "SERVICE_DEFAULT_LANE_CONFIG",
    "resolve_contract_concurrency_lane",
    "resolve_hosted_run_concurrency_lane",
]
