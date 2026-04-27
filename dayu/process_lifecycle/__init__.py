"""进程级优雅退出协调器。

模块职责：
- 把 sync CLI、async daemon、atexit 三种进程入口的退出语义收口在一处。
- 提供 ``ProcessShutdownCoordinator`` 协调三件正交动作：
    1. 取消当前进程登记的 active run（cooperative，调用 ``Host.cancel_run``）。
    2. 强制收敛同 owner 的剩余 active run（``Host.shutdown_active_runs_for_owner``）。
    3. 把信号映射成统一退出码。

模块边界：
- 协调器与具体进程入口解耦，cli / wechat 通过 ``sync_signals`` /
  ``async_signals`` 这两条入口适配自身执行模型。
- 不覆盖 SIGKILL / 断电等不可捕获场景，这部分仍由
  ``RunRegistry.cleanup_orphan_runs`` 在下次启动时收敛。
"""

from __future__ import annotations

from dayu.process_lifecycle.coordinator import (
    HostCancelRunHook,
    HostShutdownHook,
    ProcessShutdownCoordinator,
    RunLifecycleObserver,
)
from dayu.process_lifecycle.exit_codes import (
    EXIT_CODE_SIGINT,
    EXIT_CODE_SIGTERM,
    map_signal_to_exit_code,
)
from dayu.process_lifecycle.sync_signals import (
    install_sync_signal_handlers,
    register_process_shutdown_hook,
)
from dayu.process_lifecycle.async_signals import install_async_signal_handlers


__all__ = [
    "EXIT_CODE_SIGINT",
    "EXIT_CODE_SIGTERM",
    "HostCancelRunHook",
    "HostShutdownHook",
    "ProcessShutdownCoordinator",
    "RunLifecycleObserver",
    "install_async_signal_handlers",
    "install_sync_signal_handlers",
    "map_signal_to_exit_code",
    "register_process_shutdown_hook",
]
