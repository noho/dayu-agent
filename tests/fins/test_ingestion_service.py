"""ingestion.service 测试。"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Optional

import pytest

from dayu.fins.ingestion.process_events import ProcessEvent
from dayu.fins.ingestion.process_events import ProcessEventType
from dayu.fins.ingestion.service import FinsIngestionService
from dayu.fins.pipelines.download_events import DownloadEvent
from dayu.fins.pipelines.download_events import DownloadEventType


class _FakeIngestionBackend:
    """共享事务后端桩。"""

    def __init__(self) -> None:
        """初始化调用记录。"""

        self.download_calls: list[dict[str, Any]] = []
        self.process_calls: list[dict[str, Any]] = []

    async def download_stream(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: Optional[list[str]] = None,
        cancel_checker: Optional[Any] = None,
    ) -> AsyncIterator[DownloadEvent]:
        """返回固定下载事件流。"""

        self.download_calls.append(
            {
                "ticker": ticker,
                "form_type": form_type,
                "start_date": start_date,
                "end_date": end_date,
                "overwrite": overwrite,
                "rebuild": rebuild,
                "ticker_aliases": ticker_aliases,
                "cancel_checker": cancel_checker,
            }
        )
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_STARTED,
            ticker=ticker,
            payload={"overwrite": overwrite, "rebuild": rebuild},
        )
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_COMPLETED,
            ticker=ticker,
            payload={
                "result": {
                    "action": "download",
                    "ticker": ticker,
                    "status": "ok",
                    "summary": {"total": 1, "downloaded": 1, "skipped": 0, "failed": 0},
                    "filings": [{"document_id": "fil_1", "status": "downloaded", "downloaded_files": 1}],
                }
            },
        )

    async def process_stream(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: Optional[list[str]] = None,
        cancel_checker: Optional[Any] = None,
    ) -> AsyncIterator[ProcessEvent]:
        """返回固定预处理事件流。"""

        self.process_calls.append(
            {
                "ticker": ticker,
                "overwrite": overwrite,
                "ci": ci,
                "document_ids": document_ids,
                "cancel_checker": cancel_checker,
            }
        )
        yield ProcessEvent(
            event_type=ProcessEventType.PIPELINE_STARTED,
            ticker=ticker,
            payload={"overwrite": overwrite, "ci": ci, "total_documents": 1},
        )
        yield ProcessEvent(
            event_type=ProcessEventType.PIPELINE_COMPLETED,
            ticker=ticker,
            payload={
                "result": {
                    "action": "process",
                    "ticker": ticker,
                    "status": "ok",
                    "filing_summary": {"total": 1, "processed": 1, "skipped": 0, "failed": 0},
                    "material_summary": {"total": 0, "processed": 0, "skipped": 0, "failed": 0},
                }
            },
        )


@pytest.mark.unit
def test_download_sync_result_matches_stream_collection() -> None:
    """验证 `download()` 与 `download_stream()` 的最终结果一致。"""

    backend = _FakeIngestionBackend()
    service = FinsIngestionService(backend=backend)

    sync_result = service.download(
        ticker="AAPL",
        form_type="10-K",
        start_date="2024-01-01",
        end_date="2024-12-31",
        overwrite=True,
    )
    async_result = asyncio.run(
        _collect_pipeline_result(
            service.download_stream(
                ticker="AAPL",
                form_type="10-K",
                start_date="2024-01-01",
                end_date="2024-12-31",
                overwrite=True,
            )
        )
    )

    assert sync_result == async_result
    assert backend.download_calls[0]["ticker"] == "AAPL"
    assert backend.download_calls[0]["overwrite"] is True


@pytest.mark.unit
def test_download_sync_passes_cancel_checker_to_stream() -> None:
    """同步下载包装器必须把取消检查函数下传给流式后端。"""

    backend = _FakeIngestionBackend()
    service = FinsIngestionService(backend=backend)

    def cancel_checker() -> bool:
        """测试取消检查函数。"""

        return True

    service.download(ticker="600519", cancel_checker=cancel_checker)

    assert backend.download_calls[0]["cancel_checker"] is cancel_checker


@pytest.mark.unit
def test_process_sync_result_matches_stream_collection() -> None:
    """验证 `process()` 与 `process_stream()` 的最终结果一致。"""

    backend = _FakeIngestionBackend()
    service = FinsIngestionService(backend=backend)

    sync_result = service.process(
        ticker="AAPL",
        overwrite=True,
        ci=False,
        document_ids=["fil_001"],
    )
    async_result = asyncio.run(
        _collect_pipeline_result(
            service.process_stream(
                ticker="AAPL",
                overwrite=True,
                ci=False,
                document_ids=["fil_001"],
            )
        )
    )

    assert sync_result == async_result
    assert backend.process_calls[0]["ticker"] == "AAPL"
    assert backend.process_calls[0]["overwrite"] is True
    assert backend.process_calls[0]["document_ids"] == ["fil_001"]


async def _collect_pipeline_result(
    stream: AsyncIterator[DownloadEvent | ProcessEvent],
) -> dict[str, Any]:
    """从事件流中提取最终结果。"""

    result: Optional[dict[str, Any]] = None
    async for event in stream:
        if event.event_type != "pipeline_completed":
            continue
        payload_result = event.payload.get("result")
        if isinstance(payload_result, dict):
            result = payload_result
    if result is None:
        raise AssertionError("事件流未返回最终结果")
    return result
