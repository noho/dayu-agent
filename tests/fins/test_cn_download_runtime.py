"""CN download runtime 级接入测试。"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from dayu.contracts.fins import (
    DownloadCommandPayload,
    DownloadFilingResultStatus,
    DownloadProgressPayload,
    DownloadResultData,
    FinsCommand,
    FinsCommandName,
    FinsEvent,
    FinsEventType,
    FinsProgressEventName,
    FinsResult,
)
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.fins.pipelines import PipelineProtocol
from dayu.fins.pipelines.cn_download_models import (
    CnCompanyProfile,
    CnReportCandidate,
    CnReportQuery,
    DownloadedReportAsset,
)
from dayu.fins.pipelines.cn_pipeline import CnPipeline
from dayu.fins.processors.registry import build_fins_processor_registry
from dayu.fins.service_runtime import DefaultFinsRuntime
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    DocumentBlobRepositoryProtocol,
    FilingMaintenanceRepositoryProtocol,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)
from dayu.fins.ticker_normalization import NormalizedTicker
import dayu.fins.service_runtime as service_runtime_module

_PDF_BYTES = b"%PDF-1.7\n" + b"1" * 2048
_DOCLING_BYTES = b'{"document": "runtime-ok"}'


@dataclass
class _RuntimeDownloadFakeDiscoveryClient:
    """runtime 接入测试用 CN discovery fake。"""

    temp_dir: Path
    download_calls: int = 0

    def resolve_company(self, query: CnReportQuery) -> CnCompanyProfile:
        """返回固定公司元数据。

        Args:
            query: 下载查询。

        Returns:
            公司元数据。

        Raises:
            无。
        """

        return CnCompanyProfile(
            provider="cninfo",
            company_id="CNINFO:runtime-org",
            company_name="平安银行",
            ticker=query.normalized_ticker,
        )

    def list_report_candidates(
        self,
        query: CnReportQuery,
        profile: CnCompanyProfile,
    ) -> tuple[CnReportCandidate, ...]:
        """返回固定年度报告候选。

        Args:
            query: 下载查询。
            profile: 公司元数据。

        Returns:
            候选报告 tuple。

        Raises:
            无。
        """

        del profile
        return (
            CnReportCandidate(
                provider="cninfo",
                source_id="runtime-a1",
                source_url="https://static.cninfo.test/runtime-a1.pdf",
                title="平安银行：2025年年度报告",
                language="zh",
                filing_date="2026-04-01",
                fiscal_year=2025,
                fiscal_period="FY",
                amended=False,
                content_length=len(_PDF_BYTES),
                etag='"runtime-v1"',
                last_modified="Wed, 01 Apr 2026 00:00:00 GMT",
            ),
        )

    def download_report_pdf(self, candidate: CnReportCandidate) -> DownloadedReportAsset:
        """返回本地 PDF 资产。

        Args:
            candidate: 远端候选。

        Returns:
            已下载 PDF 资产。

        Raises:
            OSError: 临时文件写入失败时抛出。
        """

        self.download_calls += 1
        pdf_path = self.temp_dir / f"{candidate.source_id}_{self.download_calls}.pdf"
        pdf_path.write_bytes(_PDF_BYTES)
        return DownloadedReportAsset(
            candidate=candidate,
            pdf_path=pdf_path,
            sha256=hashlib.sha256(_PDF_BYTES).hexdigest(),
            content_length=len(_PDF_BYTES),
            downloaded_at="2026-05-02T00:00:00+00:00",
        )


@dataclass
class _RuntimeDownloadFakeConverter:
    """runtime 接入测试用 Docling fake。"""

    calls: int = 0

    def __call__(self, raw_data: bytes, stream_name: str) -> bytes:
        """返回固定 Docling JSON 字节。

        Args:
            raw_data: PDF 字节。
            stream_name: 流名称。

        Returns:
            Docling JSON 字节。

        Raises:
            无。
        """

        del raw_data, stream_name
        self.calls += 1
        return _DOCLING_BYTES


@dataclass
class _RuntimeCnPipelineFactory:
    """为 runtime 测试注入真实 CnPipeline 与 fake downloader。"""

    temp_dir: Path
    discovery: _RuntimeDownloadFakeDiscoveryClient = field(init=False)
    converter: _RuntimeDownloadFakeConverter = field(default_factory=_RuntimeDownloadFakeConverter)

    def __post_init__(self) -> None:
        """初始化 fake discovery。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.discovery = _RuntimeDownloadFakeDiscoveryClient(temp_dir=self.temp_dir)

    def build_pipeline(
        self,
        normalized_ticker: NormalizedTicker,
        workspace_root: Path,
        processor_hint: str | None = None,
        company_repository: CompanyMetaRepositoryProtocol | None = None,
        source_repository: SourceDocumentRepositoryProtocol | None = None,
        processed_repository: ProcessedDocumentRepositoryProtocol | None = None,
        blob_repository: DocumentBlobRepositoryProtocol | None = None,
        filing_maintenance_repository: FilingMaintenanceRepositoryProtocol | None = None,
        processor_registry: ProcessorRegistry | None = None,
    ) -> PipelineProtocol:
        """构建测试用真实 CN pipeline。

        Args:
            normalized_ticker: runtime 已归一化的 ticker。
            workspace_root: 工作区根目录。
            processor_hint: 处理器路线提示；本测试使用显式 registry。
            company_repository: 公司元数据仓储。
            source_repository: source 文档仓储。
            processed_repository: processed 文档仓储。
            blob_repository: blob 仓储。
            filing_maintenance_repository: filing 维护仓储。
            processor_registry: 处理器注册表。

        Returns:
            绑定 fake discovery 的 ``CnPipeline``。

        Raises:
            AssertionError: 非 CN ticker 进入测试 factory 时抛出。
        """

        del processor_hint
        if normalized_ticker.market != "CN":
            raise AssertionError(f"预期 CN ticker，收到 {normalized_ticker.market}")
        return CnPipeline(
            workspace_root=workspace_root,
            processor_registry=processor_registry or build_fins_processor_registry(),
            company_repository=company_repository,
            source_repository=source_repository,
            processed_repository=processed_repository,
            blob_repository=blob_repository,
            filing_maintenance_repository=filing_maintenance_repository,
            cn_discovery_client=self.discovery,
            convert_pdf_to_docling_json=self.converter,
        )


def _require_sync_result(result: FinsResult | AsyncIterator[FinsEvent]) -> FinsResult:
    """收窄 runtime 同步执行结果。

    Args:
        result: ``DefaultFinsRuntime.execute`` 返回值。

    Returns:
        同步执行结果。

    Raises:
        AssertionError: 返回值为流式迭代器时抛出。
    """

    if not isinstance(result, FinsResult):
        raise AssertionError("预期同步执行返回 FinsResult")
    return result


def _require_stream_result(
    result: FinsResult | AsyncIterator[FinsEvent],
) -> AsyncIterator[FinsEvent]:
    """收窄 runtime 流式执行结果。

    Args:
        result: ``DefaultFinsRuntime.execute`` 返回值。

    Returns:
        流式事件迭代器。

    Raises:
        AssertionError: 返回值为同步结果时抛出。
    """

    if isinstance(result, FinsResult):
        raise AssertionError("预期流式执行返回 AsyncIterator[FinsEvent]")
    return result


def _require_download_progress(event: FinsEvent) -> DownloadProgressPayload:
    """收窄 download progress 事件负载。

    Args:
        event: runtime 统一事件。

    Returns:
        download progress 负载。

    Raises:
        AssertionError: 事件不是 download progress 时抛出。
    """

    if event.type != FinsEventType.PROGRESS:
        raise AssertionError("预期 progress 事件")
    if not isinstance(event.payload, DownloadProgressPayload):
        raise AssertionError("预期 DownloadProgressPayload")
    return event.payload


def test_runtime_download_sync_uses_cn_pipeline_and_builds_contract_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 runtime 同步 download 进入真实 CN workflow 并输出契约结果。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: 临时工作区。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    factory = _RuntimeCnPipelineFactory(temp_dir=tmp_path)
    monkeypatch.setattr(
        service_runtime_module,
        "get_pipeline_from_normalized_ticker",
        factory.build_pipeline,
    )
    runtime = DefaultFinsRuntime.create(workspace_root=tmp_path)

    result = _require_sync_result(
        runtime.execute(
            FinsCommand(
                name=FinsCommandName.DOWNLOAD,
                payload=DownloadCommandPayload(
                    ticker="000001",
                    form_type=("FY",),
                    start_date="2025-01-01",
                    end_date="2026-12-31",
                    overwrite=True,
                    ticker_aliases=("平安银行",),
                ),
            )
        )
    )

    assert isinstance(result.data, DownloadResultData)
    assert result.data.pipeline == "cn"
    assert result.data.status == "ok"
    assert result.data.ticker == "000001"
    assert result.data.company_info.company_id == "000001_SZSE"
    assert result.data.filters.forms == ("FY",)
    assert result.data.filters.end_date == "2026-12-31"
    assert result.data.filters.overwrite is True
    assert result.data.summary.total == 1
    assert result.data.summary.downloaded == 1
    assert result.data.filings[0].status == DownloadFilingResultStatus.DOWNLOADED
    assert factory.discovery.download_calls == 1
    assert factory.converter.calls == 1


@pytest.mark.asyncio
async def test_runtime_download_stream_uses_cn_pipeline_and_emits_result_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 runtime 流式 download 透传 CN pipeline 事件并收口结果。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        tmp_path: 临时工作区。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    factory = _RuntimeCnPipelineFactory(temp_dir=tmp_path)
    monkeypatch.setattr(
        service_runtime_module,
        "get_pipeline_from_normalized_ticker",
        factory.build_pipeline,
    )
    runtime = DefaultFinsRuntime.create(workspace_root=tmp_path)

    stream = _require_stream_result(
        runtime.execute(
            FinsCommand(
                name=FinsCommandName.DOWNLOAD,
                payload=DownloadCommandPayload(
                    ticker="000001",
                    form_type=("FY",),
                    start_date="2025-01-01",
                    end_date="2026-12-31",
                ),
                stream=True,
            )
        )
    )
    events = [event async for event in stream]

    progress_names = [
        _require_download_progress(event).event_type
        for event in events
        if event.type == FinsEventType.PROGRESS
    ]
    assert progress_names == [
        FinsProgressEventName.PIPELINE_STARTED,
        FinsProgressEventName.COMPANY_RESOLVED,
        FinsProgressEventName.FILING_STARTED,
        FinsProgressEventName.FILE_DOWNLOADED,
        FinsProgressEventName.FILING_COMPLETED,
        FinsProgressEventName.PIPELINE_COMPLETED,
    ]
    assert events[-1].type == FinsEventType.RESULT
    assert isinstance(events[-1].payload, DownloadResultData)
    assert events[-1].payload.status == "ok"
    assert events[-1].payload.summary.downloaded == 1
    assert factory.discovery.download_calls == 1
    assert factory.converter.calls == 1
