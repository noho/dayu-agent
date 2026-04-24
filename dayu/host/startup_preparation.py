"""Host 层对启动期暴露的 preparation API。

本模块收口 Host 启动期需要的稳定装配入口，
让 `startup/` 与 UI 入口只依赖 Host public surface，
而不触碰 Host 内部默认常量或规范化细节。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dayu.host.concurrency import DEFAULT_LANE_CONFIG
from dayu.workspace_paths import HOST_STORE_RELATIVE_PATH, build_host_store_default_path

DEFAULT_PENDING_TURN_RESUME_MAX_ATTEMPTS = 3
# pending turn 在 ACCEPTED_BY_HOST / PREPARED_BY_HOST 状态下最长保留时间（小时）。
# 超过该阈值后 `Host.cleanup_stale_pending_turns` 会兜底删除，避免因 UI 层始终
# 未询问用户"是否重发"导致库无限累积。168 = 7 天，预留给 UI 的询问窗口，
# 同时远大于正常 CLI / Web / WeChat 的实际询问周期（一般 <72 小时）。
DEFAULT_PENDING_TURN_RETENTION_HOURS = 168


@dataclass(frozen=True)
class ResolvedHostConfig:
    """Host 启动期的已解析配置。"""

    store_path: Path
    lane_config: dict[str, int]
    pending_turn_resume_max_attempts: int
    pending_turn_retention_hours: int


def resolve_host_config(
    *,
    workspace_root: Path,
    run_config: dict[str, Any],
    service_lane_defaults: dict[str, int] | None = None,
    explicit_lane_config: dict[str, int] | None = None,
) -> ResolvedHostConfig:
    """解析 Host 启动配置。

    Args:
        workspace_root: 当前工作区根目录。
        run_config: 已加载的 `run.json` 配置对象。
        service_lane_defaults: Service 启动期注入的业务 lane 默认配置；
            Host 不感知其业务语义，仅按 lane_config 合并顺序叠加。
        explicit_lane_config: UI 启动期额外传入的 lane 覆盖配置。

    Returns:
        已规范化的 Host 配置。

    Raises:
        TypeError: 配置结构非法时抛出。
        ValueError: 配置值非法时抛出。
    """

    _reject_legacy_host_config_keys(run_config)
    raw_host_config = run_config.get("host_config")
    if raw_host_config is None:
        raw_host_config = {}
    if not isinstance(raw_host_config, dict):
        raise TypeError("run.json.host_config 必须是对象")
    return ResolvedHostConfig(
        store_path=_resolve_store_path(workspace_root=workspace_root, raw_host_config=raw_host_config),
        lane_config=_resolve_lane_config(
            raw_host_config=raw_host_config,
            service_lane_defaults=service_lane_defaults,
            explicit_lane_config=explicit_lane_config,
        ),
        pending_turn_resume_max_attempts=_resolve_pending_turn_resume_max_attempts(raw_host_config),
        pending_turn_retention_hours=_resolve_pending_turn_retention_hours(raw_host_config),
    )


def _reject_legacy_host_config_keys(run_config: dict[str, Any]) -> None:
    """拒绝旧版 Host 顶层配置键。

    Args:
        run_config: 已加载的运行配置。

    Returns:
        无。

    Raises:
        TypeError: 发现旧版顶层 Host 配置时抛出。
    """

    legacy_keys = tuple(
        key for key in ("host_store_config", "lane_config") if key in run_config
    )
    if not legacy_keys:
        return
    legacy_key_text = ", ".join(legacy_keys)
    raise TypeError(
        "run.json 的 Host 配置已收口到 host_config，"
        f"不再接受旧顶层键: {legacy_key_text}"
    )


def _resolve_store_path(*, workspace_root: Path, raw_host_config: dict[str, Any]) -> Path:
    """解析 Host SQLite 存储路径。"""

    raw_store_config = raw_host_config.get("store")
    if raw_store_config is None:
        return build_host_store_default_path(workspace_root).resolve()
    if not isinstance(raw_store_config, dict):
        raise TypeError("run.json.host_config.store 必须是对象")
    raw_path = raw_store_config.get("path", str(HOST_STORE_RELATIVE_PATH))
    if not isinstance(raw_path, str):
        raise TypeError("run.json.host_config.store.path 必须是字符串")
    normalized_path = raw_path.strip() or str(HOST_STORE_RELATIVE_PATH)
    resolved_path = Path(normalized_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = workspace_root / resolved_path
    return resolved_path.resolve()


def _resolve_lane_config(
    *,
    raw_host_config: dict[str, Any],
    service_lane_defaults: dict[str, int] | None,
    explicit_lane_config: dict[str, int] | None,
) -> dict[str, int]:
    """解析 Host 最终生效的并发 lane 配置。

    合并顺序（后者覆盖前者）：
        1. Host ``DEFAULT_LANE_CONFIG``（仅 Host 自治 lane）。
        2. ``service_lane_defaults``：Service 启动期注入的业务 lane 默认值。
        3. ``run.json.host_config.lane``：用户可覆盖任何一层。
        4. ``explicit_lane_config``：UI/CLI 启动期最强覆盖。
    """

    resolved = dict(DEFAULT_LANE_CONFIG)
    if service_lane_defaults is not None:
        resolved.update(
            _normalize_lane_config(
                service_lane_defaults,
                source_name="resolve_host_config(service_lane_defaults)",
            )
        )
    raw_lane_config = raw_host_config.get("lane")
    if raw_lane_config is not None:
        resolved.update(
            _normalize_lane_config(
                raw_lane_config,
                source_name="run.json.host_config.lane",
            )
        )
    if explicit_lane_config is not None:
        resolved.update(
            _normalize_lane_config(
                explicit_lane_config,
                source_name="resolve_host_config(explicit_lane_config)",
            )
        )
    return resolved


def _resolve_pending_turn_resume_max_attempts(raw_host_config: dict[str, Any]) -> int:
    """解析 pending turn resume 最大尝试次数。"""

    raw_resume_config = raw_host_config.get("pending_turn_resume")
    if raw_resume_config is None:
        return DEFAULT_PENDING_TURN_RESUME_MAX_ATTEMPTS
    if not isinstance(raw_resume_config, dict):
        raise TypeError("run.json.host_config.pending_turn_resume 必须是对象")
    raw_max_attempts = raw_resume_config.get(
        "max_attempts",
        DEFAULT_PENDING_TURN_RESUME_MAX_ATTEMPTS,
    )
    if (
        isinstance(raw_max_attempts, bool)
        or not isinstance(raw_max_attempts, int)
        or raw_max_attempts <= 0
    ):
        raise ValueError("run.json.host_config.pending_turn_resume.max_attempts 必须是正整数")
    return raw_max_attempts


def _resolve_pending_turn_retention_hours(raw_host_config: dict[str, Any]) -> int:
    """解析 pending turn 超保留期阈值（小时）。"""

    raw_retention_config = raw_host_config.get("pending_turn_retention")
    if raw_retention_config is None:
        return DEFAULT_PENDING_TURN_RETENTION_HOURS
    if not isinstance(raw_retention_config, dict):
        raise TypeError("run.json.host_config.pending_turn_retention 必须是对象")
    raw_retention_hours = raw_retention_config.get(
        "retention_hours",
        DEFAULT_PENDING_TURN_RETENTION_HOURS,
    )
    if (
        isinstance(raw_retention_hours, bool)
        or not isinstance(raw_retention_hours, int)
        or raw_retention_hours <= 0
    ):
        raise ValueError(
            "run.json.host_config.pending_turn_retention.retention_hours 必须是正整数"
        )
    return raw_retention_hours


def _normalize_lane_config(raw_lane_config: Any, *, source_name: str) -> dict[str, int]:
    """规范化并发 lane 配置。"""

    if not isinstance(raw_lane_config, dict):
        raise TypeError(f"{source_name} 必须是对象映射")

    normalized: dict[str, int] = {}
    for lane_name, max_concurrent in raw_lane_config.items():
        if not isinstance(lane_name, str) or not lane_name.strip():
            raise ValueError(f"{source_name} 的 lane 名必须是非空字符串")
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int) or max_concurrent <= 0:
            raise ValueError(f"{source_name}.{lane_name} 必须是正整数")
        normalized[lane_name] = max_concurrent
    return normalized


__all__ = [
    "DEFAULT_PENDING_TURN_RESUME_MAX_ATTEMPTS",
    "DEFAULT_PENDING_TURN_RETENTION_HOURS",
    "ResolvedHostConfig",
    "resolve_host_config",
]
