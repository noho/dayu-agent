"""长事务共享编排服务。"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Optional, Protocol

from .process_events import ProcessEvent

if TYPE_CHECKING:
    from dayu.fins.pipelines.download_events import DownloadEvent


class IngestionBackendProtocol(Protocol):
    """长事务后端协议。

    由市场/管线适配层实现，负责真正执行 `download/process` 流式逻辑。
    """

    def download_stream(
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
        """执行流式下载。"""

        ...

    def process_stream(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: Optional[list[str]] = None,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> AsyncIterator[ProcessEvent]:
        """执行流式预处理。"""

        ...


class FinsIngestionService:
    """财报长事务共享编排服务。

    该服务不关心文档读取字段逻辑，只负责：
    - 调用后端的流式长事务接口。
    - 提供同步聚合包装器。
    - 为 job runner 提供可选取消检查入口。
    """

    def __init__(self, *, backend: IngestionBackendProtocol) -> None:
        """初始化服务。

        Args:
            backend: 长事务后端实现。

        Returns:
            无。

        Raises:
            ValueError: 后端缺失时抛出。
        """

        if backend is None:
            raise ValueError("backend 不能为空")
        self._backend = backend

    async def download_stream(
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
        """执行流式下载。

        Args:
            ticker: 股票代码。
            form_type: 可选表单过滤。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否覆盖。
            rebuild: 是否重建本地 meta/manifest。
            ticker_aliases: 可选公司 alias 列表。
            cancel_checker: 可选取消检查函数，仅供 job runner 使用。

        Yields:
            下载事件流。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        async for event in self._backend.download_stream(
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

    def download(
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
    ) -> dict[str, Any]:
        """执行同步下载。

        Args:
            ticker: 股票代码。
            form_type: 可选表单过滤。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否覆盖。
            rebuild: 是否重建本地 meta/manifest。
            ticker_aliases: 可选公司 alias 列表。
            cancel_checker: 可选取消检查函数，用于同步聚合链路下传。

        Returns:
            最终结果字典。

        Raises:
            RuntimeError: 当前线程已有运行中的事件循环时抛出。
        """

        return _run_async_ingestion_sync(
            _collect_pipeline_result_from_stream(
                self.download_stream(
                    ticker=ticker,
                    form_type=form_type,
                    start_date=start_date,
                    end_date=end_date,
                    overwrite=overwrite,
                    rebuild=rebuild,
                    ticker_aliases=ticker_aliases,
                    cancel_checker=cancel_checker,
                ),
                stream_name="download_stream",
            )
        )

    async def process_stream(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: Optional[list[str]] = None,
        *,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> AsyncIterator[ProcessEvent]:
        """执行流式预处理。

        Args:
            ticker: 股票代码。
            overwrite: 是否覆盖。
            ci: 是否导出 CI 附加快照。
            document_ids: 可选文档 ID 列表；传入时仅处理这些文档。
            cancel_checker: 可选取消检查函数，仅供 job runner 使用。

        Yields:
            预处理事件流。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        async for event in self._backend.process_stream(
            ticker=ticker,
            overwrite=overwrite,
            ci=ci,
            document_ids=document_ids,
            cancel_checker=cancel_checker,
        ):
            yield event

    def process(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """执行同步预处理。

        Args:
            ticker: 股票代码。
            overwrite: 是否覆盖。
            ci: 是否导出 CI 附加快照。
            document_ids: 可选文档 ID 列表；传入时仅处理这些文档。

        Returns:
            最终结果字典。

        Raises:
            RuntimeError: 当前线程已有运行中的事件循环时抛出。
        """

        return _run_async_ingestion_sync(
            _collect_pipeline_result_from_stream(
                self.process_stream(
                    ticker=ticker,
                    overwrite=overwrite,
                    ci=ci,
                    document_ids=document_ids,
                ),
                stream_name="process_stream",
            )
        )


async def _collect_pipeline_result_from_stream(
    stream: AsyncIterator["DownloadEvent | ProcessEvent"],
    *,
    stream_name: str,
) -> dict[str, Any]:
    """从事件流中提取最终结果。

    Args:
        stream: 事件流。
        stream_name: 事件流名称。

    Returns:
        `pipeline_completed` 事件中的结果字典。

    Raises:
        RuntimeError: 未收到最终结果时抛出。
    """

    result: Optional[dict[str, Any]] = None
    async for event in stream:
        if event.event_type != "pipeline_completed":
            continue
        payload_result = event.payload.get("result")
        if isinstance(payload_result, dict):
            result = payload_result
    if result is None:
        raise RuntimeError(f"{stream_name} 未返回 pipeline_completed 结果")
    return result


def _run_async_ingestion_sync(coro: Any) -> dict[str, Any]:
    """在同步上下文运行长事务协程。

    Args:
        coro: 协程对象。

    Returns:
        协程结果字典。

    Raises:
        RuntimeError: 当前线程已有运行中的事件循环时抛出。
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("检测到正在运行的事件循环，请改用 stream 异步接口")
