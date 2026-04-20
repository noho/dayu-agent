"""Pipeline → Ingestion 后端适配层。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, AsyncIterator, Callable, Optional, Protocol

from .process_events import ProcessEvent

if TYPE_CHECKING:
    from dayu.fins.pipelines.download_events import DownloadEvent


class PipelineIngestionSourceProtocol(Protocol):
    """支持长事务私有实现的 pipeline 协议。"""

    def download_stream_impl(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: Optional[list[str]] = None,
        *,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> AsyncIterator["DownloadEvent"]:
        """执行底层下载流。

        Args:
            ticker: 股票代码。
            form_type: 可选表单过滤。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否覆盖。
            rebuild: 是否重建本地 meta/manifest。
            ticker_aliases: 可选公司 alias 列表。
            cancel_checker: 可选取消检查函数。

        Returns:
            下载事件异步迭代器。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...

    def process_stream_impl(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: Optional[list[str]] = None,
        *,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> AsyncIterator[ProcessEvent]:
        """执行底层预处理流。

        Args:
            ticker: 股票代码。
            overwrite: 是否覆盖。
            ci: 是否导出 CI 附加快照。
            document_ids: 可选文档 ID 列表；传入时仅处理这些文档。
            cancel_checker: 可选取消检查函数。

        Returns:
            预处理事件异步迭代器。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...


@dataclass
class PipelineIngestionBackend:
    """基于 pipeline 私有实现的适配后端。"""

    pipeline: PipelineIngestionSourceProtocol

    async def download_stream(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: Optional[list[str]] = None,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> AsyncIterator["DownloadEvent"]:
        """转发到 pipeline 私有下载实现。

        Args:
            ticker: 股票代码。
            form_type: 可选表单过滤。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否覆盖。
            rebuild: 是否重建本地 meta/manifest。
            ticker_aliases: 可选公司 alias 列表。
            cancel_checker: 可选取消检查函数。

        Returns:
            下载事件异步迭代器。

        Raises:
            RuntimeError: pipeline 执行失败时抛出。
        """

        async for event in self.pipeline.download_stream_impl(
            ticker=ticker,
            form_type=form_type,
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
            rebuild=rebuild,
            ticker_aliases=ticker_aliases,
            cancel_checker=cancel_checker,
        ):
            yield event

    async def process_stream(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: Optional[list[str]] = None,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> AsyncIterator[ProcessEvent]:
        """转发到 pipeline 私有预处理实现。

        Args:
            ticker: 股票代码。
            overwrite: 是否覆盖。
            ci: 是否导出 CI 附加快照。
            document_ids: 可选文档 ID 列表；传入时仅处理这些文档。
            cancel_checker: 可选取消检查函数。

        Returns:
            预处理事件异步迭代器。

        Raises:
            RuntimeError: pipeline 执行失败时抛出。
        """

        async for event in self.pipeline.process_stream_impl(
            ticker=ticker,
            overwrite=overwrite,
            ci=ci,
            document_ids=document_ids,
            cancel_checker=cancel_checker,
        ):
            yield event
