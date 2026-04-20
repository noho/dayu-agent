"""Doc toolset 访问边界辅助函数。

该模块负责从通用执行上下文中推导 doc tool 所需的路径白名单。
规则归属保持在 doc toolset 边界内，不向 Host 泄漏 doc domain 细节。
"""

from __future__ import annotations

from pathlib import Path

from dayu.contracts.agent_execution import ExecutionDocPermissions
from dayu.contracts.infrastructure import WorkspaceResourcesProtocol


def build_doc_tool_allowed_paths(workspace: WorkspaceResourcesProtocol) -> tuple[Path, ...]:
    """构建 doc 工具的宿主级读取白名单。

    Args:
        workspace: 工作区稳定资源。

    Returns:
        宿主允许的读取路径根目录元组。

    Raises:
        无。
    """

    candidates = [
        workspace.workspace_dir,
        workspace.config_root,
        *([workspace.output_dir] if workspace.output_dir.exists() else []),
    ]
    resolved_paths: list[Path] = []
    for candidate in candidates:
        resolved_candidate = candidate.resolve()
        if any(path_is_covered_by_allowed_roots(resolved_candidate, existing) for existing in resolved_paths):
            continue
        resolved_paths.append(resolved_candidate)
    return tuple(resolved_paths)


def build_effective_doc_allowed_paths(
    *,
    workspace: WorkspaceResourcesProtocol,
    doc_permissions: ExecutionDocPermissions,
) -> tuple[Path, ...]:
    """构建 doc 工具最终生效的读取白名单。

    Args:
        workspace: 工作区稳定资源。
        doc_permissions: 当前执行的 doc 动态权限。

    Returns:
        最终生效的读取路径根目录元组。

    Raises:
        无。
    """

    host_allowed_paths = build_doc_tool_allowed_paths(workspace)
    if not doc_permissions.allowed_read_paths:
        return host_allowed_paths

    effective_paths: list[Path] = []
    for raw_path in doc_permissions.allowed_read_paths:
        candidate = resolve_permission_path(workspace=workspace, raw_path=raw_path)
        if candidate is None:
            continue
        if not any(path_is_covered_by_allowed_roots(candidate, allowed_root) for allowed_root in host_allowed_paths):
            continue
        if any(path_is_covered_by_allowed_roots(candidate, existing) for existing in effective_paths):
            continue
        effective_paths.append(candidate)
    return tuple(effective_paths)


def resolve_permission_path(
    *,
    workspace: WorkspaceResourcesProtocol,
    raw_path: str,
) -> Path | None:
    """把权限路径文本解析为绝对路径。

    Args:
        workspace: 工作区稳定资源。
        raw_path: 权限路径原文。

    Returns:
        解析后的绝对路径；空值时返回 ``None``。

    Raises:
        无。
    """

    normalized = str(raw_path or "").strip()
    if not normalized:
        return None
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        return (workspace.workspace_dir / candidate).resolve()
    return candidate.resolve()


def path_is_covered_by_allowed_roots(path: Path, allowed_root: Path) -> bool:
    """判断路径是否已被某个白名单根目录覆盖。

    Args:
        path: 目标路径。
        allowed_root: 白名单根目录。

    Returns:
        若 ``path`` 位于 ``allowed_root`` 下则返回 ``True``。

    Raises:
        无。
    """

    try:
        path.relative_to(allowed_root)
        return True
    except ValueError:
        return False


__all__ = [
    "build_doc_tool_allowed_paths",
    "build_effective_doc_allowed_paths",
    "path_is_covered_by_allowed_roots",
    "resolve_permission_path",
]
