"""CancellationBridge：跨进程取消桥接。

轮询 SQLite run 状态，发现 cancel 标记时触发进程内 CancellationToken。
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from dayu.contracts.run import TERMINAL_STATES, RunState
from dayu.contracts.cancellation import CancellationToken

if TYPE_CHECKING:
    from dayu.host.protocols import RunRegistryProtocol


class CancellationBridge:
    """跨进程取消桥接器。

    在后台 daemon 线程中轮询 SQLite run 状态，
    当检测到 run 已写入取消请求意图时触发进程内 CancellationToken。
    当 run 进入其他终态（SUCCEEDED/FAILED）时自动停止轮询。

    线程安全，stop() 可重入。
    """

    def __init__(
        self,
        run_registry: RunRegistryProtocol,
        run_id: str,
        token: CancellationToken,
        poll_interval: float = 0.5,
    ) -> None:
        """初始化 CancellationBridge。

        Args:
            run_registry: 用于查询 run 状态的注册表。
            run_id: 监听的 run ID。
            token: 进程内取消令牌。
            poll_interval: 轮询间隔（秒）。
        """

        self._run_registry = run_registry
        self._run_id = run_id
        self._token = token
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """启动后台轮询线程。

        重复调用是安全的：如果已在运行则忽略。
        """

        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name=f"cancel-bridge-{self._run_id}",
        )
        self._thread.start()

    def stop(self) -> None:
        """停止轮询。

        可重入：多次调用安全。
        """

        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._poll_interval * 2)
        self._thread = None

    def _poll_loop(self) -> None:
        """后台轮询循环。"""

        while not self._stop_event.is_set():
            try:
                run = self._run_registry.get_run(self._run_id)
                if run is None:
                    # run 被删除，停止轮询
                    break
                if run.cancel_requested_at is not None:
                    self._token.cancel()
                    break
                if run.state in TERMINAL_STATES:
                    # run 已完成（SUCCEEDED/FAILED），无需继续轮询
                    break
            except Exception:  # noqa: BLE001
                # 查询失败不中断轮询，下次重试
                pass
            self._stop_event.wait(timeout=self._poll_interval)


__all__ = ["CancellationBridge"]
