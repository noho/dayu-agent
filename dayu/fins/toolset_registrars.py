"""Fins toolset adapter。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import cast

from dayu.contracts.tool_configs import build_fins_tool_limits
from dayu.contracts.toolset_registrar import ToolsetRegistrationContext
from dayu.engine.tool_registry import ToolRegistry
from dayu.fins.service_runtime import DefaultFinsRuntime
from dayu.fins.tools.fins_tools import register_fins_ingestion_tools, register_fins_read_tools


@lru_cache(maxsize=8)
def _get_cached_fins_runtime(workspace_root: str) -> DefaultFinsRuntime:
    """按工作区缓存 Fins runtime。

    Args:
        workspace_root: 工作区根目录绝对路径。

    Returns:
        缓存后的 Fins runtime。

    Raises:
        无。
    """

    return DefaultFinsRuntime.create(workspace_root=Path(workspace_root))


def register_fins_read_toolset(context: ToolsetRegistrationContext) -> int:
    """注册 fins 读取 toolset。

    Args:
        context: toolset 注册上下文。

    Returns:
        实际注册的工具数量。

    Raises:
        无。
    """

    runtime = _get_cached_fins_runtime(str(context.workspace.workspace_dir.resolve()))
    fins_tool_limits = build_fins_tool_limits(context.toolset_config)
    before_count = len(context.registry.tools)
    register_fins_read_tools(
        cast(ToolRegistry, context.registry),
        service=runtime.get_tool_service(
            processor_cache_max_entries=fins_tool_limits.processor_cache_max_entries
        ),
        limits=fins_tool_limits,
        timeout_budget=context.tool_timeout_seconds,
    )
    return len(context.registry.tools) - before_count


def register_fins_ingestion_toolset(context: ToolsetRegistrationContext) -> int:
    """注册 fins ingestion toolset。

    Args:
        context: toolset 注册上下文。

    Returns:
        实际注册的工具数量。

    Raises:
        无。
    """

    runtime = _get_cached_fins_runtime(str(context.workspace.workspace_dir.resolve()))
    before_count = len(context.registry.tools)
    register_fins_ingestion_tools(
        cast(ToolRegistry, context.registry),
        service_factory=runtime.build_ingestion_service_factory(),
        manager_key=runtime.get_ingestion_manager_key(),
        timeout_budget=context.tool_timeout_seconds,
    )
    return len(context.registry.tools) - before_count


__all__ = [
    "register_fins_ingestion_toolset",
    "register_fins_read_toolset",
]
