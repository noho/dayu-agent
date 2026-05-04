"""Fins 下载段业务 lane gate 实现。

该模块位于 Service 层，负责把启动期已装配好的 Host 并发 governor 适配成
CN/HK pipeline 需要的 PDF 下载段 gate。Fins pipeline 只依赖窄协议，不感知
HostStore、run.json 或具体 governor 实现。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Callable, Final

from dayu.contracts.cancellation import CancelledError
from dayu.fins.pipelines.cn_download_models import CnSourceProvider
from dayu.fins.pipelines.cn_download_pdf_gate import CnDownloadPdfGateProtocol
from dayu.host.protocols import ConcurrencyGovernorProtocol, ConcurrencyPermit
from dayu.services.concurrency_lanes import LANE_CN_DOWNLOAD, LANE_HK_DOWNLOAD

_PDF_GATE_POLL_INTERVAL_SECONDS: Final[float] = 0.1
_PDF_GATE_ACQUIRE_TIMEOUT_SECONDS: Final[float] = 300.0
_PDF_GATE_STALE_REAP_INTERVAL_SECONDS: Final[float] = 5.0


@dataclass(frozen=True)
class GovernorCnDownloadPdfGate(CnDownloadPdfGateProtocol):
    """基于 Host 并发 governor 的 CN/HK PDF 下载段 gate。"""

    governor: ConcurrencyGovernorProtocol
    acquire_timeout_seconds: float = _PDF_GATE_ACQUIRE_TIMEOUT_SECONDS
    stale_reap_interval_seconds: float = _PDF_GATE_STALE_REAP_INTERVAL_SECONDS

    def lease_for_provider(
        self,
        provider: CnSourceProvider,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> "_GovernorCnDownloadPdfLease":
        """返回指定 provider 对应的下载段 lease。

        Args:
            provider: CN/HK 主源 provider。
            cancel_checker: 可选取消检查函数，等待 permit 时会短轮询检查。

        Returns:
            进入后持有对应 PDF 下载 lane 的上下文管理器。

        Raises:
            ValueError: provider 非法时抛出。
        """

        return _GovernorCnDownloadPdfLease(
            governor=self.governor,
            lane=_lane_for_provider(provider),
            timeout_seconds=self.acquire_timeout_seconds,
            stale_reap_interval_seconds=self.stale_reap_interval_seconds,
            cancel_checker=cancel_checker,
        )


@dataclass
class _GovernorCnDownloadPdfLease:
    """单次 PDF 下载的业务 lane lease。"""

    governor: ConcurrencyGovernorProtocol
    lane: str
    timeout_seconds: float
    stale_reap_interval_seconds: float
    cancel_checker: Callable[[], bool] | None
    _permit: ConcurrencyPermit | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> None:
        """获取 PDF 下载 lane permit。

        Args:
            无。

        Returns:
            无。

        Raises:
            TimeoutError: 等待 permit 超时时抛出。
            CancelledError: 等待期间收到取消请求时抛出。
            ValueError: lane 未配置时抛出。
        """

        deadline = time.monotonic() + self.timeout_seconds
        last_reap_started_at = time.monotonic()
        while True:
            _raise_if_cancelled(self.cancel_checker)
            permit = self.governor.try_acquire(self.lane)
            if permit is not None:
                self._permit = permit
                return None
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"获取 PDF 下载 lane 超时: lane={self.lane}, timeout={self.timeout_seconds}s"
                )
            last_reap_started_at = _maybe_cleanup_stale_permits(
                governor=self.governor,
                last_reap_started_at=last_reap_started_at,
                stale_reap_interval_seconds=self.stale_reap_interval_seconds,
            )
            time.sleep(_PDF_GATE_POLL_INTERVAL_SECONDS)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """释放 PDF 下载 lane permit。

        Args:
            exc_type: with 块异常类型。
            exc: with 块异常实例。
            traceback: with 块异常 traceback。

        Returns:
            无。

        Raises:
            无。
        """

        del exc_type, exc, traceback
        permit = self._permit
        if permit is not None:
            self.governor.release(permit)
            self._permit = None


def _raise_if_cancelled(cancel_checker: Callable[[], bool] | None) -> None:
    """检查取消状态。

    Args:
        cancel_checker: 可选取消检查函数。

    Returns:
        无。

    Raises:
        CancelledError: 取消检查命中时抛出。
    """

    if cancel_checker is not None and cancel_checker():
        raise CancelledError("PDF 下载 lane 等待已取消")


def _lane_for_provider(provider: CnSourceProvider) -> str:
    """解析主源 provider 对应的 Service 业务 lane。

    Args:
        provider: CN/HK 主源 provider。

    Returns:
        Service 层配置的 PDF 下载业务 lane 名称。

    Raises:
        ValueError: provider 非法时抛出。
    """

    if provider == "cninfo":
        return LANE_CN_DOWNLOAD
    if provider == "hkexnews":
        return LANE_HK_DOWNLOAD
    raise ValueError(f"不支持的 CN/HK 下载 provider: {provider}")


def _maybe_cleanup_stale_permits(
    *,
    governor: ConcurrencyGovernorProtocol,
    last_reap_started_at: float,
    stale_reap_interval_seconds: float,
) -> float:
    """按节流频率清理已死亡进程遗留的 permit。

    Args:
        governor: Host 并发 governor 协议。
        last_reap_started_at: 上一次启动清理的 monotonic 时间。
        stale_reap_interval_seconds: 清理节流间隔。

    Returns:
        最新清理启动时间；未到清理周期时原样返回。

    Raises:
        无。清理失败会降级为下一轮继续等待。
    """

    now = time.monotonic()
    if now - last_reap_started_at < stale_reap_interval_seconds:
        return last_reap_started_at
    try:
        governor.cleanup_stale_permits()
    except Exception:
        return now
    return now


__all__ = ["GovernorCnDownloadPdfGate"]
