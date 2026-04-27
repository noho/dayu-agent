"""进程级优雅退出协调器。

模块职责：
- 把 sync CLI、async daemon、atexit 三种进程入口的退出语义收口在一处。
- 提供 ``ProcessShutdownCoordinator`` 把"收敛活跃 run"做成幂等动作：
  先取 observer 已登记 run，再合并 ``Host.list_active_run_ids_for_current_owner()``
  兜底覆盖未登记窗口，对每个 run 同步执行 ``Host.cancel_run_and_settle``（写
  ``cancel_requested_at`` + ``mark_cancelled`` + 清理关联 pending turn）。
- 把信号映射成统一退出码。

模块边界：
- 协调器与具体进程入口解耦，cli / wechat 通过 ``sync_signals`` /
  ``async_signals`` 这两条入口适配自身执行模型。
- 不覆盖 SIGKILL / 断电等不可捕获场景，这部分仍由
  ``RunRegistry.cleanup_orphan_runs`` 在下次启动时收敛。
"""

from __future__ import annotations

from dayu.process_lifecycle.coordinator import (
    HostSettleHook,
    ProcessShutdownCoordinator,
    RunLifecycleObserver,
)
from dayu.process_lifecycle.exit_codes import (
    EXIT_CODE_SIGINT,
    EXIT_CODE_SIGTERM,
    map_signal_to_exit_code,
)
from dayu.process_lifecycle.sync_signals import register_process_shutdown_hook
from dayu.process_lifecycle.async_signals import install_async_signal_handlers


__all__ = [
    "EXIT_CODE_SIGINT",
    "EXIT_CODE_SIGTERM",
    "HostSettleHook",
    "ProcessShutdownCoordinator",
    "RunLifecycleObserver",
    "install_async_signal_handlers",
    "map_signal_to_exit_code",
    "register_process_shutdown_hook",
]
