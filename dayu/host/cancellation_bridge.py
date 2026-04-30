"""CancellationBridge：跨进程取消桥接。

轮询 SQLite run 状态，发现 cancel 标记时触发进程内 CancellationToken。
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING

from dayu.contracts.run import TERMINAL_STATES
from dayu.contracts.cancellation import CancellationToken
from dayu.log import Log

if TYPE_CHECKING:
    from dayu.host.protocols import RunRegistryProtocol


_MODULE = "HOST.CANCELLATION_BRIDGE"


class CancellationBridge:
    """跨进程取消桥接器。

    在后台 daemon 线程中轮询 SQLite run 状态，
    当检测到 run 已写入取消请求意图时触发进程内 CancellationToken。
    当 run 进入其他终态（SUCCEEDED/FAILED）时自动停止轮询。

    线程安全，stop() 可重入。

    失败降级策略：
        若底层 ``run_registry.get_run`` 持续抛非预期异常，bridge 会按
        ``failure_grace_period_seconds`` 估算可接受的取消探测空窗，连续失败
        超出该窗口后停止轮询并通过 ``Log.error`` 告知，避免在系统性异常
        下空转消耗资源。一旦成功查询一次，失败计数立即清零。
    """

    def __init__(
        self,
        run_registry: RunRegistryProtocol,
        run_id: str,
        token: CancellationToken,
        poll_interval: float = 0.5,
        failure_grace_period_seconds: float = 5.0,
    ) -> None:
        """初始化 CancellationBridge。

        Args:
            run_registry: 用于查询 run 状态的注册表。
            run_id: 监听的 run ID。
            token: 进程内取消令牌。
            poll_interval: 轮询间隔（秒），必须为正数。
            failure_grace_period_seconds: 容忍底层连续失败的时间窗口（秒），
                轮询连续失败累计超过该窗口后停止线程。必须为正数。

        Raises:
            ValueError: 当 ``poll_interval`` 或 ``failure_grace_period_seconds``
                非正数时抛出。
        """

        if poll_interval <= 0.0:
            raise ValueError("poll_interval 必须为正数")
        if failure_grace_period_seconds <= 0.0:
            raise ValueError("failure_grace_period_seconds 必须为正数")

        self._run_registry = run_registry
        self._run_id = run_id
        self._token = token
        self._poll_interval = poll_interval
        self._failure_grace_period_seconds = failure_grace_period_seconds
        # 由「时间窗口 / 轮询间隔」推导的连续失败阈值，至少为 1，避免出现
        # grace 比 poll_interval 还小的边界配置导致永不退出。
        self._max_consecutive_failures = max(
            1,
            math.ceil(failure_grace_period_seconds / poll_interval),
        )
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

        consecutive_failures = 0
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
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                # 查询失败不立即中断轮询，但累计失败次数；持续失败到阈值后退出，
                # 避免在系统性异常下空转消耗资源。
                consecutive_failures += 1
                Log.warn(
                    "CancellationBridge 轮询失败: "
                    f"run_id={self._run_id}, "
                    f"consecutive_failures={consecutive_failures}, "
                    f"max_consecutive_failures={self._max_consecutive_failures}, "
                    f"poll_interval_seconds={self._poll_interval}, "
                    f"failure_grace_period_seconds={self._failure_grace_period_seconds}, "
                    f"error={exc}",
                    module=_MODULE,
                )
                if consecutive_failures >= self._max_consecutive_failures:
                    Log.error(
                        "CancellationBridge 连续轮询失败已超出容忍窗口，停止轮询: "
                        f"run_id={self._run_id}, "
                        f"max_consecutive_failures={self._max_consecutive_failures}, "
                        f"poll_interval_seconds={self._poll_interval}, "
                        f"failure_grace_period_seconds={self._failure_grace_period_seconds}",
                        module=_MODULE,
                    )
                    break
            self._stop_event.wait(timeout=self._poll_interval)


__all__ = ["CancellationBridge"]
