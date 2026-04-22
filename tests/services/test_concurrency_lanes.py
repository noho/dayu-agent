"""Service 侧并发 lane resolver 单元测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.services.concurrency_lanes import (
    LANE_SEC_DOWNLOAD,
    LANE_WRITE_CHAPTER,
    SERVICE_DEFAULT_LANE_CONFIG,
    resolve_contract_concurrency_lane,
    resolve_hosted_run_concurrency_lane,
)
from dayu.services.internal.write_pipeline.enums import WriteSceneName


@pytest.mark.unit
def test_service_default_lane_config_contains_business_lanes_only() -> None:
    """验证 Service 默认 lane 只声明业务 lane，不出现 Host 自治 lane。"""

    assert set(SERVICE_DEFAULT_LANE_CONFIG.keys()) == {LANE_WRITE_CHAPTER, LANE_SEC_DOWNLOAD}
    assert all(value > 0 for value in SERVICE_DEFAULT_LANE_CONFIG.values())


@pytest.mark.unit
def test_resolve_contract_concurrency_lane_returns_write_chapter_for_all_write_scenes() -> None:
    """写作流水线全部 scene 都应映射到 write_chapter 业务 lane。"""

    for scene in WriteSceneName:
        assert resolve_contract_concurrency_lane(scene.value) == LANE_WRITE_CHAPTER


@pytest.mark.unit
@pytest.mark.parametrize(
    "scene_name",
    ["chat", "prompt", "interactive", "", "custom_scene"],
)
def test_resolve_contract_concurrency_lane_returns_none_for_non_write_scenes(
    scene_name: str,
) -> None:
    """非写作场景返回 None，由 Host 根据调用路径决定是否补 llm_api。"""

    assert resolve_contract_concurrency_lane(scene_name) is None


@pytest.mark.unit
def test_resolve_hosted_run_concurrency_lane_maps_known_operations() -> None:
    """HostedRunSpec 层 resolver 覆盖三条业务分支。"""

    assert resolve_hosted_run_concurrency_lane("write_pipeline") == LANE_WRITE_CHAPTER
    assert resolve_hosted_run_concurrency_lane("fins_download") == LANE_SEC_DOWNLOAD
    assert resolve_hosted_run_concurrency_lane("fins_analyze") is None
    assert resolve_hosted_run_concurrency_lane("") is None


@pytest.mark.unit
def test_concurrency_lanes_module_has_no_llm_api_literal() -> None:
    """守卫：dayu/services/concurrency_lanes.py 代码正文不出现 llm_api 字面量。

    仅在模块 docstring 中以反引号形式说明"本模块不出现此字面量"是允许的。
    """

    module_path = (
        Path(__file__).resolve().parents[2]
        / "dayu"
        / "services"
        / "concurrency_lanes.py"
    )
    source = module_path.read_text(encoding="utf-8")
    # 排除 docstring 中自述句中的 ``"llm_api"``
    without_literal_docstring = source.replace('``"llm_api"``', "")
    assert '"llm_api"' not in without_literal_docstring
