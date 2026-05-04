"""CN/HK PDF 下载段 gate 协议。

本模块只定义 Fins pipeline 需要的窄协议与空实现，不依赖 Host、Service、
配置加载或具体并发治理实现。真实跨进程 gate 由启动装配层注入。
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Callable, Protocol

from .cn_download_models import CnSourceProvider


class CnDownloadPdfGateProtocol(Protocol):
    """CN/HK PDF 下载段 gate 协议。"""

    def lease_for_provider(
        self,
        provider: CnSourceProvider,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> AbstractContextManager[None]:
        """返回指定 provider 对应的下载段 lease。

        Args:
            provider: CN/HK 主源 provider。
            cancel_checker: 可选取消检查函数，等待 permit 时必须尽快响应。

        Returns:
            进入后持有对应 PDF 下载 lane 的上下文管理器。

        Raises:
            ValueError: provider 非法时抛出。
            CancelledError: 等待 permit 时收到取消请求时抛出。
        """

        ...


@dataclass(frozen=True)
class NoopCnDownloadPdfGate(CnDownloadPdfGateProtocol):
    """不做并发限制的 PDF 下载段 gate。"""

    def lease_for_provider(
        self,
        provider: CnSourceProvider,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> AbstractContextManager[None]:
        """返回空 lease。

        Args:
            provider: CN/HK 主源 provider。
            cancel_checker: 可选取消检查函数；空 gate 不主动调用。

        Returns:
            不获取任何 permit 的上下文管理器。

        Raises:
            ValueError: provider 非法时抛出。
        """

        del cancel_checker
        del provider
        return nullcontext()


__all__ = [
    "CnDownloadPdfGateProtocol",
    "NoopCnDownloadPdfGate",
]
