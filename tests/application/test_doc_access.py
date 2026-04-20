"""Doc 工具访问边界辅助函数测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import pytest

from dayu.contracts.agent_execution import ExecutionDocPermissions
from dayu.contracts.infrastructure import WorkspaceResourcesProtocol
from dayu.engine.doc_access_policy import (
    build_doc_tool_allowed_paths,
    build_effective_doc_allowed_paths,
    path_is_covered_by_allowed_roots,
    resolve_permission_path,
)


@dataclass(frozen=True)
class _WorkspaceStub:
    """实现 `WorkspaceResourcesProtocol` 必需路径属性的最小测试桩。"""

    workspace_dir: Path
    config_root: Path
    output_dir: Path
    config_loader: object = object()
    prompt_asset_store: object = object()


@pytest.mark.unit
def test_build_doc_tool_allowed_paths_deduplicates_nested_roots() -> None:
    """宿主允许路径应去重并忽略不存在的输出目录。"""

    with TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir)
        config_root = workspace_dir / "config"
        config_root.mkdir()
        output_dir = workspace_dir / "output"
        workspace = _WorkspaceStub(workspace_dir=workspace_dir, config_root=config_root, output_dir=output_dir)

        allowed = build_doc_tool_allowed_paths(cast(WorkspaceResourcesProtocol, workspace))

        assert allowed == (workspace_dir.resolve(),)


@pytest.mark.unit
def test_build_effective_doc_allowed_paths_filters_outside_roots_and_deduplicates() -> None:
    """动态权限路径应只保留宿主允许范围内的有效根目录。"""

    with TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir)
        config_root = workspace_dir / "config"
        output_dir = workspace_dir / "output"
        docs_dir = workspace_dir / "docs"
        nested_docs_dir = docs_dir / "nested"
        outside_dir = workspace_dir.parent / "outside"

        config_root.mkdir()
        output_dir.mkdir()
        docs_dir.mkdir()
        nested_docs_dir.mkdir()
        outside_dir.mkdir(exist_ok=True)

        workspace = _WorkspaceStub(workspace_dir=workspace_dir, config_root=config_root, output_dir=output_dir)
        permissions = ExecutionDocPermissions(
            allowed_read_paths=(
                "docs",
                "docs/nested",
                str(config_root),
                str(outside_dir),
                "   ",
            )
        )

        allowed = build_effective_doc_allowed_paths(
            workspace=cast(WorkspaceResourcesProtocol, workspace),
            doc_permissions=permissions,
        )

        assert allowed == (docs_dir.resolve(), config_root.resolve())


@pytest.mark.unit
def test_build_effective_doc_allowed_paths_falls_back_to_host_defaults() -> None:
    """未提供动态读权限时应直接回退到宿主默认白名单。"""

    with TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir)
        config_root = workspace_dir / "config"
        output_dir = workspace_dir / "output"
        config_root.mkdir()
        output_dir.mkdir()
        workspace = _WorkspaceStub(workspace_dir=workspace_dir, config_root=config_root, output_dir=output_dir)

        allowed = build_effective_doc_allowed_paths(
            workspace=cast(WorkspaceResourcesProtocol, workspace),
            doc_permissions=ExecutionDocPermissions(),
        )

        assert allowed == (workspace_dir.resolve(),)


@pytest.mark.unit
def test_resolve_permission_path_and_path_coverage() -> None:
    """权限路径解析与覆盖判断应支持空值、相对路径和绝对路径。"""

    with TemporaryDirectory() as tmp_dir:
        workspace_dir = Path(tmp_dir)
        config_root = workspace_dir / "config"
        output_dir = workspace_dir / "output"
        config_root.mkdir()
        output_dir.mkdir()
        workspace = _WorkspaceStub(workspace_dir=workspace_dir, config_root=config_root, output_dir=output_dir)

        typed_workspace = cast(WorkspaceResourcesProtocol, workspace)
        assert resolve_permission_path(workspace=typed_workspace, raw_path="   ") is None

        relative = resolve_permission_path(workspace=typed_workspace, raw_path="docs/file.md")
        absolute = resolve_permission_path(workspace=typed_workspace, raw_path=str(config_root))

        assert relative == (workspace_dir / "docs/file.md").resolve()
        assert absolute == config_root.resolve()
        assert path_is_covered_by_allowed_roots(config_root.resolve(), workspace_dir.resolve()) is True
        assert path_is_covered_by_allowed_roots(workspace_dir.parent.resolve(), workspace_dir.resolve()) is False
