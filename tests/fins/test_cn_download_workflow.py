"""CN download workflow 单元测试。

覆盖 A4 的核心语义：主流程事件序列、完成态 fast skip、远端 fingerprint 变化但
PDF SHA 相同的只读 skip、PDF 中间态恢复，以及 overwrite ticker 级清理。
"""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import BinaryIO, Optional, TypeAlias
from io import BytesIO
from pathlib import Path
from types import TracebackType

import pytest

from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.fins.domain.document_models import (
    FileObjectMeta,
    FilingUpdateRequest,
    RejectedFilingArtifact,
    RejectedFilingArtifactUpsertRequest,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.cn_download_models import (
    CN_PIPELINE_DOWNLOAD_VERSION,
    CnCompanyProfile,
    CnFiscalPeriod,
    CnReportCandidate,
    CnReportQuery,
    CnSourceProvider,
    DownloadedReportAsset,
)
from dayu.fins.pipelines.cn_download_pdf_gate import CnDownloadPdfGateProtocol, NoopCnDownloadPdfGate
from dayu.fins.pipelines.cn_pipeline import CnPipeline
from dayu.fins.pipelines.docling_upload_service import build_cn_filing_ids
from dayu.fins.pipelines.download_events import DownloadEvent, DownloadEventType
from dayu.fins.storage import FilingMaintenanceRepositoryProtocol
from tests.fins.storage_testkit import build_fs_storage_test_context

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

_PDF_BYTES = b"%PDF-1.7\n" + b"0" * 2048
_DOCLING_BYTES = b'{"document": "ok"}'


@dataclass
class _FakeDiscoveryClient:
    """CN discovery fake。"""

    temp_dir: Path
    candidates: tuple[CnReportCandidate, ...]
    pdf_bytes: bytes = _PDF_BYTES
    download_calls: int = 0
    failed_source_ids: set[str] = field(default_factory=set)
    list_error: RuntimeError | None = None
    delete_pdf_before_return: bool = False

    def resolve_company(self, query: CnReportQuery) -> CnCompanyProfile:
        """返回固定公司元数据。"""

        return CnCompanyProfile(
            provider="cninfo",
            company_id="CNINFO:9900000600",
            company_name="贵州茅台",
            ticker=query.normalized_ticker,
        )

    def list_report_candidates(
        self,
        query: CnReportQuery,
        profile: CnCompanyProfile,
    ) -> tuple[CnReportCandidate, ...]:
        """返回测试候选。"""

        del query, profile
        if self.list_error is not None:
            raise self.list_error
        return self.candidates

    def download_report_pdf(self, candidate: CnReportCandidate) -> DownloadedReportAsset:
        """写入临时 PDF 并返回下载资产。"""

        self.download_calls += 1
        if candidate.source_id in self.failed_source_ids:
            raise RuntimeError(f"download failed: {candidate.source_id}")
        path = self.temp_dir / f"{candidate.source_id}_{self.download_calls}.pdf"
        path.write_bytes(self.pdf_bytes)
        if self.delete_pdf_before_return:
            path.unlink()
        return DownloadedReportAsset(
            candidate=candidate,
            pdf_path=path,
            sha256=hashlib.sha256(self.pdf_bytes).hexdigest(),
            content_length=len(self.pdf_bytes),
            downloaded_at="2026-05-02T00:00:00+00:00",
        )


@dataclass
class _FakeConverter:
    """Docling 转换 fake。"""

    fail_once: bool = False
    calls: int = 0

    def __call__(self, raw_data: bytes, stream_name: str) -> bytes:
        """返回固定 Docling JSON。"""

        del raw_data, stream_name
        self.calls += 1
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("docling failed")
        return _DOCLING_BYTES


@dataclass
class _RecordingPdfGate(CnDownloadPdfGateProtocol):
    """记录 PDF 下载 gate 持有状态。"""

    active: bool = False
    enter_count: int = 0
    exit_count: int = 0

    def lease_for_provider(
        self,
        provider: CnSourceProvider,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> AbstractContextManager[None]:
        """返回记录型 lease。"""

        del cancel_checker
        assert provider in {"cninfo", "hkexnews"}
        return _RecordingPdfGateLease(self)


@dataclass
class _RecordingPdfGateLease:
    """测试用 PDF gate lease。"""

    gate: _RecordingPdfGate

    def __enter__(self) -> None:
        """标记 gate 已进入。"""

        self.gate.active = True
        self.gate.enter_count += 1
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """标记 gate 已退出。"""

        del exc_type, exc, traceback
        self.gate.active = False
        self.gate.exit_count += 1


@dataclass
class _GateAwareConverter(_FakeConverter):
    """验证 Docling 转换不在 PDF 下载 gate 内执行。"""

    gate: _RecordingPdfGate = field(default_factory=_RecordingPdfGate)

    def __call__(self, raw_data: bytes, stream_name: str) -> bytes:
        """断言转换阶段没有持有 PDF 下载 gate。"""

        assert self.gate.active is False
        return super().__call__(raw_data, stream_name)


@dataclass
class _CountingMaintenanceRepository:
    """记录 clear 调用并委托真实维护仓储。"""

    delegate: FilingMaintenanceRepositoryProtocol
    cleared_tickers: list[str] = field(default_factory=list)

    def clear_filing_documents(self, ticker: str) -> None:
        """记录 ticker 级清理。"""

        self.cleared_tickers.append(ticker)
        self.delegate.clear_filing_documents(ticker)

    def load_download_rejection_registry(self, ticker: str) -> dict[str, dict[str, str]]:
        """委托读取下载拒绝注册表。"""

        return self.delegate.load_download_rejection_registry(ticker)

    def save_download_rejection_registry(
        self,
        ticker: str,
        registry: dict[str, dict[str, str]],
    ) -> None:
        """委托保存下载拒绝注册表。"""

        self.delegate.save_download_rejection_registry(ticker, registry)

    def store_rejected_filing_file(
        self,
        ticker: str,
        document_id: str,
        filename: str,
        data: BinaryIO,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> FileObjectMeta:
        """委托写入 rejected filing 文件。"""

        return self.delegate.store_rejected_filing_file(
            ticker,
            document_id,
            filename,
            data,
            content_type=content_type,
            metadata=metadata,
        )

    def upsert_rejected_filing_artifact(
        self,
        req: RejectedFilingArtifactUpsertRequest,
    ) -> RejectedFilingArtifact:
        """委托写入 rejected filing artifact。"""

        return self.delegate.upsert_rejected_filing_artifact(req)

    def get_rejected_filing_artifact(
        self,
        ticker: str,
        document_id: str,
    ) -> RejectedFilingArtifact:
        """委托读取 rejected filing artifact。"""

        return self.delegate.get_rejected_filing_artifact(ticker, document_id)

    def list_rejected_filing_artifacts(self, ticker: str) -> list[RejectedFilingArtifact]:
        """委托列出 rejected filing artifacts。"""

        return self.delegate.list_rejected_filing_artifacts(ticker)

    def read_rejected_filing_file_bytes(
        self,
        ticker: str,
        document_id: str,
        filename: str,
    ) -> bytes:
        """委托读取 rejected filing 文件。"""

        return self.delegate.read_rejected_filing_file_bytes(ticker, document_id, filename)

    def cleanup_stale_filing_documents(
        self,
        ticker: str,
        *,
        active_form_types: set[str],
        valid_document_ids: set[str],
    ) -> int:
        """委托清理 stale filing。"""

        return self.delegate.cleanup_stale_filing_documents(
            ticker,
            active_form_types=active_form_types,
            valid_document_ids=valid_document_ids,
        )


def _candidate(
    *,
    source_id: str = "A1",
    etag: str = '"v1"',
    fiscal_year: int = 2024,
    fiscal_period: CnFiscalPeriod = "FY",
    filing_date: str | None = None,
) -> CnReportCandidate:
    """构造 CN 候选。"""

    return CnReportCandidate(
        provider="cninfo",
        source_id=source_id,
        source_url=f"https://static.cninfo.test/{source_id}.pdf",
        title=f"贵州茅台：{fiscal_year}年{fiscal_period}报告",
        language="zh",
        filing_date=filing_date or f"{fiscal_year + 1}-04-01",
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        amended=False,
        content_length=len(_PDF_BYTES),
        etag=etag,
        last_modified="Wed, 01 Apr 2026 00:00:00 GMT",
    )


def _build_pipeline(
    *,
    tmp_path: Path,
    discovery: _FakeDiscoveryClient,
    converter: _FakeConverter,
    maintenance: FilingMaintenanceRepositoryProtocol | None = None,
    pdf_download_gate: CnDownloadPdfGateProtocol | None = None,
) -> CnPipeline:
    """构造注入 fake downloader / converter 的 CnPipeline。"""

    context = build_fs_storage_test_context(tmp_path)
    return CnPipeline(
        workspace_root=tmp_path,
        processor_registry=ProcessorRegistry(),
        company_repository=context.company_repository,
        source_repository=context.source_repository,
        processed_repository=context.processed_repository,
        blob_repository=context.blob_repository,
        filing_maintenance_repository=maintenance or context.filing_maintenance_repository,
        cn_discovery_client=discovery,
        pdf_download_gate=pdf_download_gate or NoopCnDownloadPdfGate(),
        convert_pdf_to_docling_json=converter,
    )


def _collect_events(
    pipeline: CnPipeline,
    *,
    overwrite: bool = False,
    form_type: str = "FY",
    cancel_checker: Callable[[], bool] | None = None,
) -> list[DownloadEvent]:
    """同步收集 download_stream 事件。"""

    async def collect() -> list[DownloadEvent]:
        events: list[DownloadEvent] = []
        async for event in pipeline.download_stream(
            ticker="600519",
            form_type=form_type,
            start_date="2024",
            end_date="2026",
            overwrite=overwrite,
            cancel_checker=cancel_checker,
        ):
            events.append(event)
        return events

    return asyncio.run(collect())


def _final_result(events: list[DownloadEvent]) -> JsonObject:
    """读取最终 pipeline result。"""

    payload = events[-1].payload.get("result")
    assert isinstance(payload, dict)
    return {str(key): value for key, value in payload.items()}


def test_cn_download_workflow_commits_pdf_and_docling(tmp_path: Path) -> None:
    """主流程应按事件序列完成 PDF + Docling + ingest_complete commit。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    events = _collect_events(pipeline)

    assert [event.event_type for event in events] == [
        DownloadEventType.PIPELINE_STARTED,
        DownloadEventType.COMPANY_RESOLVED,
        DownloadEventType.FILING_STARTED,
        DownloadEventType.FILE_DOWNLOADED,
        DownloadEventType.FILING_COMPLETED,
        DownloadEventType.PIPELINE_COMPLETED,
    ]
    result = _final_result(events)
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert summary["downloaded"] == 1
    assert summary["converted"] == 1
    assert discovery.download_calls == 1
    assert converter.calls == 1
    started = [event for event in events if event.event_type == DownloadEventType.FILING_STARTED]
    completed = [event for event in events if event.event_type == DownloadEventType.FILING_COMPLETED]
    company_info = result["company_info"]
    assert isinstance(company_info, dict)
    assert company_info["company_id"] == "600519_SSE"
    assert company_info["provider_company_id"] == "CNINFO:9900000600"
    company_meta = pipeline._company_repository.get_company_meta("600519")  # type: ignore[attr-defined]
    assert company_meta.company_id == "600519_SSE"
    document_id, _ = build_cn_filing_ids(
        ticker="600519",
        form_type="FY",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
    )
    source_meta = pipeline._source_repository.get_source_meta("600519", document_id, SourceKind.FILING)  # type: ignore[attr-defined]
    assert started[-1].document_id == document_id
    assert completed[-1].document_id == document_id
    assert source_meta["company_id"] == "600519_SSE"
    assert source_meta["provider_company_id"] == "CNINFO:9900000600"
    assert source_meta["document_version"] == "v1"


def test_cn_download_pdf_gate_does_not_cover_docling_convert(tmp_path: Path) -> None:
    """PDF 下载 gate 只应覆盖远端 PDF 下载，不应覆盖 Docling 转换。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    gate = _RecordingPdfGate()
    converter = _GateAwareConverter(gate=gate)
    pipeline = _build_pipeline(
        tmp_path=tmp_path,
        discovery=discovery,
        converter=converter,
        pdf_download_gate=gate,
    )

    result = _final_result(_collect_events(pipeline))

    summary = result["summary"]
    assert isinstance(summary, dict)
    assert summary["downloaded"] == 1
    assert summary["converted"] == 1
    assert gate.enter_count == 1
    assert gate.exit_count == 1
    assert gate.active is False
    assert converter.calls == 1


def test_cn_download_logs_match_sec_download_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CN download 应输出与 SEC download 对齐的入口和 filing 终态日志。"""

    info_logs: list[str] = []

    def capture_info(message: str, *, module: str) -> None:
        """捕获 ``Log.info`` 消息。"""

        info_logs.append(f"{module} {message}")

    monkeypatch.setattr(
        "dayu.fins.pipelines.cn_download_workflow.Log.info",
        capture_info,
    )
    monkeypatch.setattr(
        "dayu.fins.pipelines.cn_download_filing_workflow.Log.info",
        capture_info,
    )
    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    _collect_events(pipeline)

    assert any("FINS.CN_PIPELINE 进入CN/HK下载流程: ticker=600519" in item for item in info_logs)
    assert any(
        "FINS.CN_PIPELINE filing 下载完成: ticker=600519 "
        "document_id=fil_cn_" in item
        and "status=downloaded form=FY" in item
        and "downloaded_files=2 skipped_files=0 failed_files=0" in item
        for item in info_logs
    )
    assert any(
        "FINS.CN_PIPELINE 开始 Docling 转换: ticker=600519 document_id=fil_cn_" in item
        and "form=FY filing_date=2025-04-01" in item
        and "source_file=fil_cn_" in item
        for item in info_logs
    )
    assert any(
        "FINS.CN_PIPELINE CN/HK 下载完成: ticker=600519 total=1 downloaded=1 skipped=0 failed=0 elapsed_ms="
        in item
        for item in info_logs
    )


def test_cn_download_fast_skip_uses_remote_fingerprint(tmp_path: Path) -> None:
    """完成态版本与 remote_fingerprint 命中时不下载 PDF。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)
    _collect_events(pipeline)

    events = _collect_events(pipeline)

    completed = [event for event in events if event.event_type == DownloadEventType.FILING_COMPLETED]
    assert completed[-1].payload["reason_code"] == "remote_fingerprint_matched"
    assert discovery.download_calls == 1
    assert converter.calls == 1


def test_cn_download_pdf_sha_skip_commits_remote_meta_for_next_fast_skip(tmp_path: Path) -> None:
    """PDF 内容一致时跳过 Docling，但推进远端 meta 让下次 fast skip。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(etag='"v1"'),))
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)
    _collect_events(pipeline)
    discovery.candidates = (_candidate(source_id="A2", etag='"v2"'),)
    document_id = build_cn_filing_ids(
        ticker="600519",
        form_type="FY",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
    )[0]
    context = build_fs_storage_test_context(tmp_path)
    before_meta = context.source_repository.get_source_meta("600519", document_id, SourceKind.FILING)

    events = _collect_events(pipeline)

    completed = [event for event in events if event.event_type == DownloadEventType.FILING_COMPLETED]
    assert completed[-1].payload["reason_code"] == "pdf_sha256_matched"
    assert discovery.download_calls == 2
    assert converter.calls == 1
    after_meta = context.source_repository.get_source_meta("600519", document_id, SourceKind.FILING)
    assert after_meta["source_id"] == "A2"
    assert after_meta["remote_fingerprint"] != before_meta["remote_fingerprint"]
    assert after_meta["source_fingerprint"] == before_meta["source_fingerprint"]
    assert after_meta["document_version"] == before_meta["document_version"]

    third_events = _collect_events(pipeline)

    third_completed = [event for event in third_events if event.event_type == DownloadEventType.FILING_COMPLETED]
    assert third_completed[-1].payload["reason_code"] == "remote_fingerprint_matched"
    assert discovery.download_calls == 2
    assert converter.calls == 1


def test_cn_download_candidate_failure_does_not_fail_pipeline(tmp_path: Path) -> None:
    """单个 candidate PDF 下载失败时应继续处理后续候选。"""

    discovery = _FakeDiscoveryClient(
        temp_dir=tmp_path,
        candidates=(
            _candidate(source_id="A1", fiscal_year=2024),
            _candidate(source_id="A2", fiscal_year=2023),
        ),
        failed_source_ids={"A1"},
    )
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    events = _collect_events(pipeline)

    result = _final_result(events)
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert result["status"] == "ok"
    assert summary["failed"] == 1
    assert summary["downloaded"] == 1
    assert [event.event_type for event in events].count(DownloadEventType.FILING_FAILED) == 1
    assert [event.event_type for event in events].count(DownloadEventType.FILING_COMPLETED) == 1
    assert discovery.download_calls == 2


def test_cn_download_workflow_keeps_multi_year_periodic_candidates(tmp_path: Path) -> None:
    """workflow 不应再次截断 downloader 返回的跨年 H1/季度候选。"""

    discovery = _FakeDiscoveryClient(
        temp_dir=tmp_path,
        candidates=(
            _candidate(source_id="H1-2024", fiscal_year=2024, fiscal_period="H1"),
            _candidate(source_id="H1-2023", fiscal_year=2023, fiscal_period="H1"),
        ),
    )
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    events = _collect_events(pipeline, form_type="H1")

    started = [event for event in events if event.event_type == DownloadEventType.FILING_STARTED]
    result = _final_result(events)
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert [event.payload["fiscal_year"] for event in started] == [2024, 2023]
    assert summary["downloaded"] == 2
    assert discovery.download_calls == 2
    assert converter.calls == 2


def test_cn_download_workflow_marks_missing_independent_quarters_skipped(tmp_path: Path) -> None:
    """请求 Q2/Q4 但主源无独立报告时应 skipped，不应 failed 或用 H1/FY 冒充。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=())
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    events = _collect_events(pipeline, form_type="Q2 Q4")

    completed = [event for event in events if event.event_type == DownloadEventType.FILING_COMPLETED]
    result = _final_result(events)
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert [(event.payload["form_type"], event.payload["status"]) for event in completed] == [
        ("Q2", "skipped"),
        ("Q4", "skipped"),
    ]
    assert result["status"] == "ok"
    assert summary["skipped"] == 2
    assert summary["failed"] == 0
    assert discovery.download_calls == 0
    assert converter.calls == 0


def test_cn_download_default_window_limits_interim_to_two_years(tmp_path: Path) -> None:
    """默认窗口下半年报/季报只保留 end 年和上一 fiscal_year 候选。"""

    discovery = _FakeDiscoveryClient(
        temp_dir=tmp_path,
        candidates=(
            _candidate(
                source_id="H1-2026",
                fiscal_year=2026,
                fiscal_period="H1",
                filing_date="2026-08-30",
            ),
            _candidate(
                source_id="H1-2025",
                fiscal_year=2025,
                fiscal_period="H1",
                filing_date="2025-08-30",
            ),
            _candidate(
                source_id="H1-2024",
                fiscal_year=2024,
                fiscal_period="H1",
                filing_date="2024-08-30",
            ),
        ),
    )
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    async def collect() -> list[DownloadEvent]:
        events: list[DownloadEvent] = []
        async for event in pipeline.download_stream(
            ticker="600519",
            form_type="H1",
            start_date=None,
            end_date="2026-12-31",
            overwrite=False,
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    started = [event for event in events if event.event_type == DownloadEventType.FILING_STARTED]
    result = _final_result(events)
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert [event.payload["fiscal_year"] for event in started] == [2026, 2025]
    assert summary["downloaded"] == 2
    assert discovery.download_calls == 2
    assert converter.calls == 2


def test_cn_download_default_window_limits_annual_to_five_reports(tmp_path: Path) -> None:
    """默认窗口下 FY 只保留最近 5 份年报候选。"""

    discovery = _FakeDiscoveryClient(
        temp_dir=tmp_path,
        candidates=tuple(
            _candidate(source_id=f"FY-{year}", fiscal_year=year, fiscal_period="FY")
            for year in (2025, 2024, 2023, 2022, 2021, 2020)
        ),
    )
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    async def collect() -> list[DownloadEvent]:
        events: list[DownloadEvent] = []
        async for event in pipeline.download_stream(
            ticker="600519",
            form_type="FY",
            start_date=None,
            end_date="2026-05-01",
            overwrite=False,
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    started = [event for event in events if event.event_type == DownloadEventType.FILING_STARTED]
    result = _final_result(events)
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert [event.payload["fiscal_year"] for event in started] == [2025, 2024, 2023, 2022, 2021]
    assert summary["downloaded"] == 5
    assert discovery.download_calls == 5
    assert converter.calls == 5


def test_cn_download_version_mismatch_redownloads(tmp_path: Path) -> None:
    """完成态 download_version 不一致时禁止 fast skip 和 PDF skip。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)
    _collect_events(pipeline)
    context = build_fs_storage_test_context(tmp_path)
    document_id = build_cn_filing_ids(
        ticker="600519",
        form_type="FY",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
    )[0]
    meta = context.source_repository.get_source_meta("600519", document_id, SourceKind.FILING)
    meta["download_version"] = "cn_pipeline_download_v0.0.0"
    meta["first_ingested_at"] = "2020-01-01T00:00:00+00:00"
    meta["created_at"] = "2020-01-01T00:00:01+00:00"
    context.source_repository.replace_source_meta("600519", document_id, SourceKind.FILING, meta)

    events = _collect_events(pipeline)

    completed = [event for event in events if event.event_type == DownloadEventType.FILING_COMPLETED]
    assert completed[-1].payload["status"] == "downloaded"
    assert discovery.download_calls == 2
    assert converter.calls == 2
    updated_meta = context.source_repository.get_source_meta("600519", document_id, SourceKind.FILING)
    assert updated_meta["first_ingested_at"] == "2020-01-01T00:00:00+00:00"
    assert updated_meta["created_at"] == "2020-01-01T00:00:01+00:00"


def test_cn_download_resumes_staged_pdf_after_docling_failure(tmp_path: Path) -> None:
    """Docling 失败后下一次应复用 staged PDF 并完成 commit。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter(fail_once=True)
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)
    first_events = _collect_events(pipeline)
    assert any(event.event_type == DownloadEventType.FILING_FAILED for event in first_events)

    second_events = _collect_events(pipeline)

    file_events = [event for event in second_events if event.event_type == DownloadEventType.FILE_DOWNLOADED]
    assert file_events[-1].payload["reused"] is True
    completed = [event for event in second_events if event.event_type == DownloadEventType.FILING_COMPLETED]
    assert completed[-1].payload["downloaded_files"] == 1
    assert completed[-1].payload["skipped_files"] == 1
    assert completed[-1].payload["reused_pdf"] is True
    assert completed[-1].payload["reused_docling"] is False
    assert discovery.download_calls == 1
    assert converter.calls == 2


def test_cn_download_stage_cancel_returns_cancelled_not_failed(tmp_path: Path) -> None:
    """阶段内取消应返回 cancelled，不应记作 failed。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)
    calls = {"count": 0}

    def cancel_checker() -> bool:
        calls["count"] += 1
        return calls["count"] >= 3

    events = _collect_events(pipeline, cancel_checker=cancel_checker)

    result = _final_result(events)
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert result["status"] == "cancelled"
    assert summary["failed"] == 0
    assert any(event.event_type == DownloadEventType.FILE_DOWNLOADED for event in events)
    assert not any(event.event_type == DownloadEventType.FILING_FAILED for event in events)


def test_cn_download_cancel_after_docling_convert_prevents_commit(tmp_path: Path) -> None:
    """Docling 转换后收到取消信号时不得继续提交完成态 source meta。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    def cancel_checker() -> bool:
        """转换完成后返回取消。"""

        return converter.calls > 0

    events = _collect_events(pipeline, cancel_checker=cancel_checker)

    result = _final_result(events)
    document_id = build_cn_filing_ids(
        ticker="600519",
        form_type="FY",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
    )[0]
    source_meta = pipeline._source_repository.get_source_meta("600519", document_id, SourceKind.FILING)  # type: ignore[attr-defined]
    assert result["status"] == "cancelled"
    assert source_meta["ingest_complete"] is False


def test_cn_download_commits_when_pdf_and_docling_are_staged(tmp_path: Path) -> None:
    """PDF 与 Docling JSON 都已落盘但 ingest_complete=False 时应直接 commit。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter(fail_once=True)
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)
    _collect_events(pipeline)
    context = build_fs_storage_test_context(tmp_path)
    document_id = build_cn_filing_ids(
        ticker="600519",
        form_type="FY",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
    )[0]
    handle = context.source_repository.get_source_handle("600519", document_id, SourceKind.FILING)
    docling_meta = context.blob_repository.store_file(
        handle,
        f"{document_id}_docling.json",
        BytesIO(_DOCLING_BYTES),
        content_type="application/json",
        metadata={"source": "docling"},
    )
    staged_meta = context.source_repository.get_source_meta("600519", document_id, SourceKind.FILING)
    existing_files = staged_meta.get("files")
    assert isinstance(existing_files, list)
    context.source_repository.update_source_document(
        FilingUpdateRequest(
            ticker="600519",
            document_id=document_id,
            internal_document_id=str(staged_meta["internal_document_id"]),
            form_type="FY",
            primary_document=f"{document_id}.pdf",
            file_entries=[
                *[item for item in existing_files if isinstance(item, dict)],
                {
                    "name": f"{document_id}_docling.json",
                    "uri": docling_meta.uri,
                    "etag": docling_meta.etag,
                    "last_modified": docling_meta.last_modified,
                    "size": docling_meta.size,
                    "content_type": docling_meta.content_type,
                    "sha256": docling_meta.sha256,
                },
            ],
            meta={"ingest_complete": False},
        ),
        source_kind=SourceKind.FILING,
    )

    events = _collect_events(pipeline)

    completed = [event for event in events if event.event_type == DownloadEventType.FILING_COMPLETED]
    assert completed[-1].payload["status"] == "downloaded"
    assert completed[-1].payload["reused_docling"] is True
    assert discovery.download_calls == 1
    assert converter.calls == 1


def test_cn_download_reuses_unlisted_docling_blob_after_crash(tmp_path: Path) -> None:
    """Docling blob 已落盘但 meta 未列出时，下次应复用 blob 并 commit。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter(fail_once=True)
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)
    _collect_events(pipeline)
    context = build_fs_storage_test_context(tmp_path)
    document_id = build_cn_filing_ids(
        ticker="600519",
        form_type="FY",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
    )[0]
    handle = context.source_repository.get_source_handle("600519", document_id, SourceKind.FILING)
    context.blob_repository.store_file(
        handle,
        f"{document_id}_docling.json",
        BytesIO(_DOCLING_BYTES),
        content_type="application/json",
        metadata={"source": "docling"},
    )

    events = _collect_events(pipeline)

    completed = [event for event in events if event.event_type == DownloadEventType.FILING_COMPLETED]
    assert completed[-1].payload["status"] == "downloaded"
    assert completed[-1].payload["reused_docling"] is True
    assert discovery.download_calls == 1
    assert converter.calls == 1


def test_cn_download_does_not_reuse_docling_when_staged_pdf_sha_differs(tmp_path: Path) -> None:
    """当前 PDF SHA 与 staged meta 不一致时，旧 Docling JSON 不能复用。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter(fail_once=True)
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)
    _collect_events(pipeline)
    context = build_fs_storage_test_context(tmp_path)
    document_id = build_cn_filing_ids(
        ticker="600519",
        form_type="FY",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
    )[0]
    handle = context.source_repository.get_source_handle("600519", document_id, SourceKind.FILING)
    context.blob_repository.store_file(
        handle,
        f"{document_id}_docling.json",
        BytesIO(b'{"document": "old"}'),
        content_type="application/json",
        metadata={"source": "docling"},
    )
    staged_meta = context.source_repository.get_source_meta("600519", document_id, SourceKind.FILING)
    staged_meta["staging_pdf_sha256"] = "0" * 64
    context.source_repository.replace_source_meta("600519", document_id, SourceKind.FILING, staged_meta)

    events = _collect_events(pipeline)

    completed = [event for event in events if event.event_type == DownloadEventType.FILING_COMPLETED]
    assert completed[-1].payload["status"] == "downloaded"
    assert completed[-1].payload["reused_docling"] is False
    assert discovery.download_calls == 2
    assert converter.calls == 2


def test_cn_download_overwrite_clears_ticker_and_redownloads(tmp_path: Path) -> None:
    """overwrite=True 应触发 ticker 级 clear，并禁止复用完成态。"""

    context = build_fs_storage_test_context(tmp_path)
    maintenance = _CountingMaintenanceRepository(context.filing_maintenance_repository)
    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=ProcessorRegistry(),
        company_repository=context.company_repository,
        source_repository=context.source_repository,
        processed_repository=context.processed_repository,
        blob_repository=context.blob_repository,
        filing_maintenance_repository=maintenance,
        cn_discovery_client=discovery,
        convert_pdf_to_docling_json=converter,
    )
    _collect_events(pipeline)

    _collect_events(pipeline, overwrite=True)

    assert maintenance.cleared_tickers == ["600519"]
    assert discovery.download_calls == 2
    meta = context.source_repository.get_source_meta(
        ticker="600519",
        document_id=build_cn_filing_ids(
            ticker="600519",
            form_type="FY",
            fiscal_year=2024,
            fiscal_period="FY",
            amended=False,
        )[0],
        source_kind=SourceKind.FILING,
    )
    assert meta["ingest_complete"] is True
    assert meta["download_version"] == CN_PIPELINE_DOWNLOAD_VERSION
    assert str(meta["primary_document"]).endswith("_docling.json")


def test_cn_download_overwrite_does_not_clear_when_discovery_fails(tmp_path: Path) -> None:
    """overwrite=True 遇到候选发现失败时不得先清空本地已完成 filing。"""

    context = build_fs_storage_test_context(tmp_path)
    maintenance = _CountingMaintenanceRepository(context.filing_maintenance_repository)
    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=ProcessorRegistry(),
        company_repository=context.company_repository,
        source_repository=context.source_repository,
        processed_repository=context.processed_repository,
        blob_repository=context.blob_repository,
        filing_maintenance_repository=maintenance,
        cn_discovery_client=discovery,
        convert_pdf_to_docling_json=converter,
    )
    _collect_events(pipeline)
    document_id = build_cn_filing_ids(
        ticker="600519",
        form_type="FY",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
    )[0]
    assert context.source_repository.get_source_meta("600519", document_id, SourceKind.FILING)

    discovery.list_error = RuntimeError("remote discovery unavailable")
    events = _collect_events(pipeline, overwrite=True)

    result = _final_result(events)
    assert result["status"] == "failed"
    assert maintenance.cleared_tickers == []
    assert context.source_repository.get_source_meta("600519", document_id, SourceKind.FILING)


def test_cn_download_pdf_temp_file_read_failure_is_filing_failed(tmp_path: Path) -> None:
    """PDF 下载后临时文件不可读时应产出 filing failed，而不是未处理异常。"""

    discovery = _FakeDiscoveryClient(
        temp_dir=tmp_path,
        candidates=(_candidate(),),
        delete_pdf_before_return=True,
    )
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    events = _collect_events(pipeline)

    failed_events = [event for event in events if event.event_type == DownloadEventType.FILING_FAILED]
    result = _final_result(events)
    summary = result["summary"]
    assert isinstance(summary, dict)
    assert failed_events[-1].payload["reason_code"] == "pdf_read_failed"
    assert summary["failed"] == 1


def test_cn_download_unsupported_ticker_raises_value_error(tmp_path: Path) -> None:
    """非 CN/HK ticker 应与 SEC 一样作为请求级错误直接抛出。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    async def collect() -> list[DownloadEvent]:
        events: list[DownloadEvent] = []
        async for event in pipeline.download_stream(ticker="AAPL", form_type="FY"):
            events.append(event)
        return events

    with pytest.raises(ValueError, match="不支持"):
        asyncio.run(collect())


def test_cn_download_rebuild_local_meta_manifest_without_redownload(tmp_path: Path) -> None:
    """CN/HK `download --rebuild` 应基于本地完成态重建且不访问远端下载。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)
    _collect_events(pipeline)
    context = build_fs_storage_test_context(tmp_path)
    document_id = build_cn_filing_ids(
        ticker="600519",
        form_type="FY",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
    )[0]
    meta = context.source_repository.get_source_meta("600519", document_id, SourceKind.FILING)
    meta["download_version"] = "legacy_download_version"
    meta["staging_remote_fingerprint"] = "legacy_stage"
    meta["staging_pdf_sha256"] = "legacy_pdf"
    context.source_repository.replace_source_meta("600519", document_id, SourceKind.FILING, meta)

    async def collect() -> list[DownloadEvent]:
        events: list[DownloadEvent] = []
        async for event in pipeline.download_stream(
            ticker="600519",
            form_type="FY",
            start_date="2024",
            end_date="2026",
            overwrite=False,
            rebuild=True,
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    result = _final_result(events)
    filters = result["filters"]
    summary = result["summary"]
    assert isinstance(filters, dict)
    assert isinstance(summary, dict)
    rebuilt_meta = context.source_repository.get_source_meta("600519", document_id, SourceKind.FILING)
    assert [event.event_type for event in events] == [
        DownloadEventType.PIPELINE_STARTED,
        DownloadEventType.FILING_COMPLETED,
        DownloadEventType.PIPELINE_COMPLETED,
    ]
    assert result["status"] == "ok"
    assert filters["rebuild"] is True
    assert summary["downloaded"] == 1
    assert discovery.download_calls == 1
    assert converter.calls == 1
    assert rebuilt_meta["download_version"] == CN_PIPELINE_DOWNLOAD_VERSION
    assert rebuilt_meta["staging_remote_fingerprint"] is None
    assert rebuilt_meta["staging_pdf_sha256"] is None


def test_cn_download_rebuild_honors_cancel_checker(tmp_path: Path) -> None:
    """rebuild 遍历本地 filing 时应响应取消并返回 cancelled。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=(_candidate(),))
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)
    _collect_events(pipeline)

    async def collect() -> list[DownloadEvent]:
        events: list[DownloadEvent] = []
        async for event in pipeline.download_stream(
            ticker="600519",
            form_type="FY",
            start_date="2024",
            end_date="2026",
            overwrite=False,
            rebuild=True,
            cancel_checker=lambda: True,
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    result = _final_result(events)
    assert [event.event_type for event in events] == [
        DownloadEventType.PIPELINE_STARTED,
        DownloadEventType.PIPELINE_COMPLETED,
    ]
    assert result["status"] == "cancelled"


def test_cn_download_post_loop_cancel_checker_error_yields_failed_result(tmp_path: Path) -> None:
    """最终状态检查时 cancel_checker 失败也必须产出 PIPELINE_COMPLETED。"""

    discovery = _FakeDiscoveryClient(temp_dir=tmp_path, candidates=())
    converter = _FakeConverter()
    pipeline = _build_pipeline(tmp_path=tmp_path, discovery=discovery, converter=converter)

    async def collect() -> list[DownloadEvent]:
        events: list[DownloadEvent] = []

        def cancel_checker() -> bool:
            """模拟取消通道关闭。"""

            raise RuntimeError("cancel channel closed")

        async for event in pipeline.download_stream(
            ticker="600519",
            form_type="FY",
            start_date="2024",
            end_date="2026",
            cancel_checker=cancel_checker,
        ):
            events.append(event)
        return events

    events = asyncio.run(collect())

    result = _final_result(events)
    assert events[-1].event_type == DownloadEventType.PIPELINE_COMPLETED
    assert result["status"] == "failed"
    assert result["reason_code"] == "cn_download_failed"
