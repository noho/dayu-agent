"""SecPipeline 异步下载事件流测试。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Optional

from dayu.fins.domain.document_models import FileObjectMeta
from dayu.fins.downloaders.sec_downloader import DownloaderEvent, RemoteFileDescriptor, SecDownloader
from dayu.fins.pipelines.download_events import DownloadEvent
from dayu.fins.pipelines.sec_pipeline import SEC_PIPELINE_DOWNLOAD_VERSION, SecPipeline
from dayu.fins.processors.registry import build_fins_processor_registry
from dayu.fins.storage.fs_source_document_repository import FsSourceDocumentRepository


class StreamStubDownloader(SecDownloader):
    """用于验证 `download_stream` 的下载器桩。"""

    def __init__(self) -> None:
        """初始化下载器桩。"""

        self.configure_called = False

    def configure(self, user_agent: Optional[str], sleep_seconds: float, max_retries: int) -> None:
        """记录配置调用。"""

        del user_agent, sleep_seconds, max_retries
        self.configure_called = True

    def normalize_ticker(self, ticker: str) -> str:
        """标准化 ticker。"""

        return ticker.strip().upper()

    async def resolve_company(self, ticker: str) -> tuple[str, str, str]:
        """返回固定公司信息。"""

        del ticker
        return ("320193", "Apple Inc.", "0000320193")

    async def fetch_submissions(self, cik10: str) -> dict[str, Any]:
        """返回固定 submissions。"""

        del cik10
        return {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2025-02-01"],
                    "reportDate": ["2024-12-31"],
                    "accessionNumber": ["0000000000-25-000001"],
                    "primaryDocument": ["sample-10k.htm"],
                },
                "files": [],
            }
        }

    async def list_filing_files(
        self,
        cik: str,
        accession_no_dash: str,
        primary_document: str,
        form_type: str,
        include_xbrl: bool = True,
        include_exhibits: bool = True,
        include_http_metadata: bool = True,
    ) -> list[RemoteFileDescriptor]:
        """返回固定远端文件列表。"""

        del cik, accession_no_dash, primary_document, form_type, include_xbrl, include_exhibits, include_http_metadata
        return [
            RemoteFileDescriptor(
                name="sample-10k.htm",
                source_url="https://example.com/sample-10k.htm",
                http_etag="etag-v1",
                http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
                remote_size=100,
                http_status=200,
            )
        ]

    async def download_files_stream(
        self,
        remote_files: list[RemoteFileDescriptor],
        overwrite: bool,
        store_file: Callable[[str, BinaryIO], FileObjectMeta],
        existing_files: Optional[dict[str, dict[str, Any]]] = None,
        primary_document: Optional[str] = None,
    ) -> AsyncIterator[DownloaderEvent]:
        """输出单文件下载事件。"""

        del overwrite, existing_files, primary_document
        descriptor = remote_files[0]
        file_meta = store_file(descriptor.name, BytesIO(b"payload"))
        yield DownloaderEvent(
            event_type="file_downloaded",
            name=descriptor.name,
            source_url=descriptor.source_url,
            http_etag=descriptor.http_etag,
            http_last_modified=descriptor.http_last_modified,
            http_status=descriptor.http_status,
            file_meta=file_meta,
        )


class StreamXbrlStubDownloader(StreamStubDownloader):
    """用于验证 download_stream 的 XBRL 落盘路径。"""

    async def list_filing_files(
        self,
        cik: str,
        accession_no_dash: str,
        primary_document: str,
        form_type: str,
        include_xbrl: bool = True,
        include_exhibits: bool = True,
        include_http_metadata: bool = True,
    ) -> list[RemoteFileDescriptor]:
        """返回 HTML 与 XBRL instance 两个远端文件。"""

        del cik, accession_no_dash, primary_document, form_type, include_xbrl, include_exhibits, include_http_metadata
        return [
            RemoteFileDescriptor(
                name="sample-10k.htm",
                source_url="https://example.com/sample-10k.htm",
                http_etag="etag-v1",
                http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
                remote_size=100,
                http_status=200,
            ),
            RemoteFileDescriptor(
                name="sample_htm.xml",
                source_url="https://example.com/sample_htm.xml",
                http_etag="etag-xbrl",
                http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
                remote_size=80,
                http_status=200,
            ),
        ]

    async def download_files_stream(
        self,
        remote_files: list[RemoteFileDescriptor],
        overwrite: bool,
        store_file: Callable[[str, BinaryIO], FileObjectMeta],
        existing_files: Optional[dict[str, dict[str, Any]]] = None,
        primary_document: Optional[str] = None,
    ) -> AsyncIterator[DownloaderEvent]:
        """输出 HTML 与 XBRL instance 两个下载事件。"""

        del overwrite, existing_files, primary_document
        payload_by_name = {
            "sample-10k.htm": b"<html>payload</html>",
            "sample_htm.xml": b"<xbrl></xbrl>",
        }
        for descriptor in remote_files:
            file_meta = store_file(descriptor.name, BytesIO(payload_by_name[descriptor.name]))
            yield DownloaderEvent(
                event_type="file_downloaded",
                name=descriptor.name,
                source_url=descriptor.source_url,
                http_etag=descriptor.http_etag,
                http_last_modified=descriptor.http_last_modified,
                http_status=descriptor.http_status,
                file_meta=file_meta,
            )


class _SpySourceRepository(FsSourceDocumentRepository):
    """记录 has_filing_xbrl_instance 调用的源文档仓储 spy。"""

    def __init__(self, workspace_root: Path) -> None:
        """初始化 spy。"""

        super().__init__(workspace_root)
        self.has_filing_xbrl_instance_calls: list[tuple[str, str]] = []

    def has_filing_xbrl_instance(self, ticker: str, document_id: str) -> bool:
        """记录调用后转发到真实实现。"""

        self.has_filing_xbrl_instance_calls.append((ticker, document_id))
        return super().has_filing_xbrl_instance(ticker, document_id)

    async def download_files_stream(
        self,
        remote_files: list[RemoteFileDescriptor],
        overwrite: bool,
        store_file: Any,
        existing_files: Optional[dict[str, dict[str, Any]]] = None,
        primary_document: Optional[str] = None,
    ):
        """输出单文件下载事件。"""

        del overwrite, existing_files, primary_document
        descriptor = remote_files[0]
        file_meta = store_file(descriptor.name, BytesIO(b"payload"))
        yield DownloaderEvent(
            event_type="file_downloaded",
            name=descriptor.name,
            source_url=descriptor.source_url,
            http_etag=descriptor.http_etag,
            http_last_modified=descriptor.http_last_modified,
            http_status=descriptor.http_status,
            file_meta=file_meta,
        )


async def _collect_events(pipeline: SecPipeline, ticker: str) -> list[DownloadEvent]:
    """收集异步下载事件。"""

    events: list[DownloadEvent] = []
    async for event in pipeline.download_stream(ticker=ticker, overwrite=False):
        events.append(event)
    return events


def test_download_stream_emits_ordered_events(tmp_path: Path) -> None:
    """验证事件顺序与完成事件负载。"""

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=StreamStubDownloader(),
        processor_registry=build_fins_processor_registry(),
    )
    import asyncio

    events = asyncio.run(_collect_events(pipeline, ticker="AAPL"))
    event_types = [event.event_type for event in events]
    assert event_types[0] == "pipeline_started"
    assert "company_resolved" in event_types
    assert "filing_started" in event_types
    assert "file_downloaded" in event_types
    assert "filing_completed" in event_types
    assert event_types[-1] == "pipeline_completed"
    final_result = events[-1].payload["result"]
    assert final_result["summary"]["downloaded"] == 1


def test_download_sync_wrapper_aggregates_stream_result(tmp_path: Path) -> None:
    """验证同步 download 包装器可返回事件流最终结果。"""

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=StreamStubDownloader(),
        processor_registry=build_fins_processor_registry(),
    )
    result = pipeline.download(ticker="AAPL", overwrite=False)
    assert result["action"] == "download"
    assert result["summary"]["downloaded"] == 1


def test_download_stream_filing_skip_event_exposes_reason_fields(tmp_path: Path) -> None:
    """验证 filing 跳过事件会同时暴露扁平与嵌套的原因字段。"""

    document_dir = tmp_path / "portfolio" / "AAPL" / "filings" / "fil_0000000000-25-000001"
    document_dir.mkdir(parents=True, exist_ok=True)
    (document_dir / "meta.json").write_text(
        json.dumps(
            {
                "document_version": "v1",
                "source_fingerprint": "fp-ready",
                "download_version": SEC_PIPELINE_DOWNLOAD_VERSION,
                "ingest_complete": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=StreamStubDownloader(),
        processor_registry=build_fins_processor_registry(),
    )

    import asyncio

    events = asyncio.run(_collect_events(pipeline, ticker="AAPL"))
    filing_event = next(event for event in events if event.event_type == "filing_completed")
    assert filing_event.payload["skip_reason"] == "already_downloaded_complete"
    assert filing_event.payload["reason_code"] == "already_downloaded_complete"
    assert "完整下载结果" in str(filing_event.payload["reason_message"])
    assert filing_event.payload["filing_result"]["skip_reason"] == "already_downloaded_complete"


def test_download_stream_resolves_has_xbrl_via_source_repository(tmp_path: Path) -> None:
    """验证下载完成后的 has_xbrl 由源文档仓储事实接口回填。"""

    source_repository = _SpySourceRepository(tmp_path)
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=StreamXbrlStubDownloader(),
        source_repository=source_repository,
        processor_registry=build_fins_processor_registry(),
    )

    import asyncio

    events = asyncio.run(_collect_events(pipeline, ticker="AAPL"))
    filing_event = next(event for event in events if event.event_type == "filing_completed")

    assert filing_event.payload["has_xbrl"] is True
    assert source_repository.has_filing_xbrl_instance_calls == [("AAPL", "fil_0000000000-25-000001")]
