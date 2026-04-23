"""服务层启动恢复辅助。

本模块负责把 Host-owned 的启动恢复策略收口为统一 helper，
避免不同 UI/runtime 在启动阶段各自直接拼接 cleanup 逻辑。
"""

from __future__ import annotations

from dayu.log import Log
from dayu.services.contracts import HostCleanupResult
from dayu.services.protocols import HostAdminServiceProtocol


def recover_host_startup_state(
    host_admin_service: HostAdminServiceProtocol,
    *,
    runtime_label: str,
    log_module: str,
) -> HostCleanupResult:
    """执行 Host-owned 启动恢复。

    启动恢复当前只包含 orphan run 与 stale permit 的统一清理。
    该 helper 采用 best-effort 语义：失败时只告警并继续启动，
    由调用方自行处理后续渠道级恢复逻辑。

    Args:
        host_admin_service: 宿主管理服务。
        runtime_label: 当前启动中的 runtime 名称，用于日志。
        log_module: 日志模块名。

    Returns:
        启动恢复结果；若恢复失败则返回空结果。

    Raises:
        无。
    """

    try:
        cleanup_result = host_admin_service.cleanup()
    except Exception as exc:
        Log.warning(f"{runtime_label} 启动恢复失败，将继续启动: {exc}", module=log_module)
        return HostCleanupResult(
            orphan_run_ids=(),
            stale_permit_ids=(),
            stale_pending_turn_ids=(),
        )
    if (
        cleanup_result.orphan_run_ids
        or cleanup_result.stale_permit_ids
        or cleanup_result.stale_pending_turn_ids
    ):
        Log.info(
            f"{runtime_label} 启动恢复完成"
            f" orphan_runs={len(cleanup_result.orphan_run_ids)}"
            f" stale_permits={len(cleanup_result.stale_permit_ids)}"
            f" stale_pending_turns={len(cleanup_result.stale_pending_turn_ids)}",
            module=log_module,
        )
    return cleanup_result


__all__ = ["recover_host_startup_state"]
