"""CnPipeline 行为测试。

本文件只测 pipeline wrapper / 上传入口的边界行为；CN download 单元测试必须注入
fake discovery client，避免默认 ``CninfoDiscoveryClient`` 访问真实巨潮接口。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest

from dayu.fins.domain.enums import SourceKind
from dayu.fins.downloaders.hkexnews_downloader import HkexnewsDiscoveryClient
from dayu.fins.pipelines.cn_download_models import (
    CnCompanyProfile,
    CnReportCandidate,
    CnReportQuery,
    DownloadedReportAsset,
)
from dayu.fins.pipelines.download_events import DownloadEventType
from dayu.fins.pipelines.upload_filing_events import UploadFilingEventType
from dayu.fins.pipelines.upload_material_events import UploadMaterialEventType
from dayu.fins.pipelines.cn_pipeline import CnPipeline
from dayu.fins.processors.registry import build_fins_processor_registry

_PDF_BYTES = b"%PDF-1.7\n" + b"0" * 2048
_DOCLING_BYTES = b'{"document": "ok"}'


@dataclass
class _PipelineDownloadFakeDiscoveryClient:
    """CnPipeline wrapper 测试用 fake discovery client。"""

    temp_dir: Path
    download_calls: int = 0

    def resolve_company(self, query: CnReportQuery) -> CnCompanyProfile:
        """返回固定公司元数据。

        Args:
            query: 下载查询。

        Returns:
            公司基础元数据。

        Raises:
            无。
        """

        return CnCompanyProfile(
            provider="cninfo",
            company_id="CNINFO:9900000001",
            company_name="平安银行",
            ticker=query.normalized_ticker,
        )

    def list_report_candidates(
        self,
        query: CnReportQuery,
        profile: CnCompanyProfile,
    ) -> tuple[CnReportCandidate, ...]:
        """返回一份固定 FY 候选。

        Args:
            query: 下载查询。
            profile: 公司元数据。

        Returns:
            候选 tuple。

        Raises:
            无。
        """

        del profile
        return (
            CnReportCandidate(
                provider="cninfo",
                source_id="A1",
                source_url="https://static.cninfo.test/A1.pdf",
                title="平安银行：2025年年度报告",
                language="zh",
                filing_date="2026-04-01",
                fiscal_year=2025,
                fiscal_period="FY",
                amended=False,
                content_length=len(_PDF_BYTES),
                etag='"v1"',
                last_modified="Wed, 01 Apr 2026 00:00:00 GMT",
            ),
        )

    def download_report_pdf(self, candidate: CnReportCandidate) -> DownloadedReportAsset:
        """返回本地临时 PDF 资产。

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
class _PipelineDownloadFakeConverter:
    """CnPipeline wrapper 测试用 Docling fake。"""

    calls: int = 0

    def __call__(self, raw_data: bytes, stream_name: str) -> bytes:
        """返回固定 Docling JSON。

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
class _PipelineDownloadFakeHkDiscoveryClient:
    """CnPipeline HK wrapper 测试用 fake discovery client。"""

    temp_dir: Path
    download_calls: int = 0

    def resolve_company(self, query: CnReportQuery) -> CnCompanyProfile:
        """返回固定 HK 公司元数据。

        Args:
            query: 下载查询。

        Returns:
            公司基础元数据。

        Raises:
            无。
        """

        return CnCompanyProfile(
            provider="hkexnews",
            company_id="HKEX:7609",
            company_name="騰訊控股",
            ticker=query.normalized_ticker,
        )

    def list_report_candidates(
        self,
        query: CnReportQuery,
        profile: CnCompanyProfile,
    ) -> tuple[CnReportCandidate, ...]:
        """返回一份固定 HK FY 候选。

        Args:
            query: 下载查询。
            profile: 公司元数据。

        Returns:
            候选 tuple。

        Raises:
            无。
        """

        del profile
        return (
            CnReportCandidate(
                provider="hkexnews",
                source_id="HK1",
                source_url="https://www1.hkexnews.hk/listedco/listconews/sehk/2025/0408/hk1.pdf",
                title="ANNUAL REPORT 2024",
                language="en",
                filing_date="2025-04-08",
                fiscal_year=2024,
                fiscal_period="FY",
                amended=False,
                content_length=len(_PDF_BYTES),
                etag='"hk-v1"',
                last_modified="Tue, 08 Apr 2025 00:00:00 GMT",
            ),
        )

    def download_report_pdf(self, candidate: CnReportCandidate) -> DownloadedReportAsset:
        """返回本地临时 PDF 资产。

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


def test_download_runs_cn_workflow_with_injected_discovery_client(tmp_path: Path) -> None:
    """验证同步 `download` wrapper 会调用真实 CN workflow 且不访问网络。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    discovery = _PipelineDownloadFakeDiscoveryClient(temp_dir=tmp_path)
    converter = _PipelineDownloadFakeConverter()
    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
        cn_discovery_client=discovery,
        convert_pdf_to_docling_json=converter,
    )

    result = pipeline.download(
        ticker="000001",
        form_type="FY",
        start_date="2025-01-01",
        end_date="2026-12-31",
        overwrite=True,
    )

    assert result["pipeline"] == "cn"
    assert result["action"] == "download"
    assert result["status"] == "ok"
    assert result["ticker"] == "000001"
    assert result["summary"]["downloaded"] == 1
    assert discovery.download_calls == 1
    assert converter.calls == 1


def test_default_hk_discovery_client_is_hkexnews(tmp_path: Path) -> None:
    """验证 CnPipeline 默认 HK discovery client 已接入披露易。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )

    assert isinstance(pipeline.hk_discovery_client, HkexnewsDiscoveryClient)


def test_download_runs_hk_workflow_with_injected_discovery_client(tmp_path: Path) -> None:
    """验证 HK ticker 会经同一 CN/HK workflow 完成下载闭环。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    discovery = _PipelineDownloadFakeHkDiscoveryClient(temp_dir=tmp_path)
    converter = _PipelineDownloadFakeConverter()
    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
        hk_discovery_client=discovery,
        convert_pdf_to_docling_json=converter,
    )

    result = pipeline.download(
        ticker="0700",
        form_type="FY",
        start_date="2024-01-01",
        end_date="2025-12-31",
        overwrite=True,
    )

    assert result["pipeline"] == "cn"
    assert result["action"] == "download"
    assert result["status"] == "ok"
    assert result["ticker"] == "0700"
    assert result["company_info"]["company_id"] == "0700_HKEX"
    assert result["summary"]["downloaded"] == 1
    assert discovery.download_calls == 1
    assert converter.calls == 1


@pytest.mark.asyncio
async def test_download_stream_runs_cn_workflow_with_injected_discovery_client(
    tmp_path: Path,
) -> None:
    """验证 `download_stream` wrapper 会产出真实 CN download 事件。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    discovery = _PipelineDownloadFakeDiscoveryClient(temp_dir=tmp_path)
    converter = _PipelineDownloadFakeConverter()
    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
        cn_discovery_client=discovery,
        convert_pdf_to_docling_json=converter,
    )

    events = [
        event
        async for event in pipeline.download_stream(
            ticker="000001",
            form_type="FY",
            start_date="2025-01-01",
            end_date="2026-12-31",
            overwrite=False,
        )
    ]

    assert [event.event_type for event in events] == [
        DownloadEventType.PIPELINE_STARTED,
        DownloadEventType.COMPANY_RESOLVED,
        DownloadEventType.FILING_STARTED,
        DownloadEventType.FILE_DOWNLOADED,
        DownloadEventType.FILING_COMPLETED,
        DownloadEventType.PIPELINE_COMPLETED,
    ]
    assert events[-1].payload["result"]["status"] == "ok"
    assert events[-1].payload["result"]["summary"]["downloaded"] == 1
    assert discovery.download_calls == 1
    assert converter.calls == 1


@pytest.mark.asyncio
async def test_upload_filing_stream_uploads_files_with_docling(tmp_path: Path) -> None:
    """验证 `upload_filing_stream` 可完成上传并生成 docling 主文件。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    pipeline._upload_service._convert_with_docling = lambda raw_data, stream_name: {  # type: ignore[attr-defined]
        "name": stream_name,
        "format": "docling",
    }
    sample_file = tmp_path / "sample.pdf"
    sample_file.write_text("demo", encoding="utf-8")

    events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="000001",
            action="create",
            files=[sample_file],
            fiscal_year=2025,
            fiscal_period="FY",
            amended=False,
            filing_date="2025-01-01",
            report_date="2024-12-31",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]

    assert len(events) == 5
    assert events[0].event_type == UploadFilingEventType.UPLOAD_STARTED
    assert events[1].event_type == UploadFilingEventType.CONVERSION_STARTED
    assert events[1].payload["name"] == "sample.pdf"
    assert events[1].payload["message"] == "正在 convert"
    assert events[2].event_type == UploadFilingEventType.FILE_UPLOADED
    assert events[2].payload["name"] == "sample.pdf"
    assert events[2].payload["source"] == "original"
    assert events[3].event_type == UploadFilingEventType.FILE_UPLOADED
    assert events[3].payload["name"] == "sample_docling.json"
    assert events[3].payload["source"] == "docling"
    assert events[4].event_type == UploadFilingEventType.UPLOAD_COMPLETED
    result = events[4].payload["result"]
    assert result["status"] == "ok"
    assert str(result["document_id"]).startswith("fil_cn_")
    assert str(result["internal_document_id"]).startswith("cn_")
    assert result["ticker_aliases"] is None
    company_meta = pipeline._company_repository.get_company_meta("000001")  # type: ignore[attr-defined]
    assert company_meta.company_id == "000001_SZSE"
    assert company_meta.company_name == "平安银行"
    meta = pipeline._source_repository.get_source_meta("000001", result["document_id"], SourceKind.FILING)  # type: ignore[attr-defined]
    assert str(meta["primary_document"]).endswith("_docling.json")


@pytest.mark.asyncio
async def test_upload_filing_failed_result_uses_normalized_period_and_aliases(
    tmp_path: Path,
) -> None:
    """CN upload_filing 失败结果应与 SEC 一样回填归一化 fiscal_period 与 aliases。"""

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )

    def fail_convert(raw_data: bytes, stream_name: str) -> dict[str, str]:
        """模拟 Docling 转换失败。"""

        del raw_data, stream_name
        raise RuntimeError("convert failed")

    pipeline._upload_service._convert_with_docling = fail_convert  # type: ignore[attr-defined]
    sample_file = tmp_path / "sample.pdf"
    sample_file.write_text("demo", encoding="utf-8")

    events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="000001",
            action="create",
            files=[sample_file],
            fiscal_year=2025,
            fiscal_period="fy",
            company_id="000001",
            company_name="平安银行",
            ticker_aliases=["000001.SZ"],
            overwrite=False,
        )
    ]

    assert events[-1].event_type == UploadFilingEventType.UPLOAD_FAILED
    result = events[-1].payload["result"]
    assert result["status"] == "failed"
    assert result["fiscal_period"] == "FY"
    assert result["ticker_aliases"] == ["000001.SZ"]


@pytest.mark.asyncio
async def test_upload_material_stream_uploads_files_with_docling(tmp_path: Path) -> None:
    """验证 `upload_material_stream` 可完成上传并生成 docling 主文件。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    pipeline._upload_service._convert_with_docling = lambda raw_data, stream_name: {  # type: ignore[attr-defined]
        "name": stream_name,
        "format": "docling",
    }
    sample_file = tmp_path / "material.pdf"
    sample_file.write_text("demo", encoding="utf-8")

    events = [
        event
        async for event in pipeline.upload_material_stream(
            ticker="000001",
            action="create",
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=[sample_file],
            filing_date="2025-01-01",
            report_date="2024-12-31",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]

    assert len(events) == 5
    assert events[0].event_type == UploadMaterialEventType.UPLOAD_STARTED
    assert events[1].event_type == UploadMaterialEventType.CONVERSION_STARTED
    assert events[1].payload["name"] == "material.pdf"
    assert events[1].payload["message"] == "正在 convert"
    assert events[2].event_type == UploadMaterialEventType.FILE_UPLOADED
    assert events[2].payload["name"] == "material.pdf"
    assert events[2].payload["source"] == "original"
    assert events[3].event_type == UploadMaterialEventType.FILE_UPLOADED
    assert events[3].payload["name"] == "material_docling.json"
    assert events[3].payload["source"] == "docling"
    assert events[4].event_type == UploadMaterialEventType.UPLOAD_COMPLETED
    result = events[4].payload["result"]
    assert result["status"] == "ok"
    assert str(result["document_id"]).startswith("mat_")
    company_meta = pipeline._company_repository.get_company_meta("000001")  # type: ignore[attr-defined]
    assert company_meta.company_id == "000001_SZSE"
    assert company_meta.company_name == "平安银行"
    meta = pipeline._source_repository.get_source_meta("000001", result["document_id"], SourceKind.MATERIAL)  # type: ignore[attr-defined]
    assert str(meta["primary_document"]).endswith("_docling.json")


@pytest.mark.asyncio
async def test_upload_filing_stream_auto_resolves_create_update_skip(tmp_path: Path) -> None:
    """验证 upload_filing_stream 在未显式传 action 时会自动 create/update/skip。"""

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    pipeline._upload_service._convert_with_docling = lambda raw_data, stream_name: {  # type: ignore[attr-defined]
        "name": stream_name,
        "format": "docling",
    }
    sample_file = tmp_path / "sample.pdf"
    sample_file.write_text("demo-v1", encoding="utf-8")

    create_events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="000001",
            action=None,
            files=[sample_file],
            fiscal_year=2025,
            fiscal_period="FY",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]
    create_result = create_events[-1].payload["result"]
    assert create_result["status"] == "ok"
    assert create_result["filing_action"] == "create"

    skip_events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="000001",
            action=None,
            files=[sample_file],
            fiscal_year=2025,
            fiscal_period="FY",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]
    skip_result = skip_events[-1].payload["result"]
    assert skip_result["status"] == "skipped"
    assert skip_result["filing_action"] == "update"
    assert skip_result["skip_reason"] == "already_uploaded"
    assert [event.event_type.value for event in skip_events] == [
        "upload_started",
        "file_skipped",
        "upload_completed",
    ]

    sample_file.write_text("demo-v2", encoding="utf-8")
    update_events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="000001",
            action=None,
            files=[sample_file],
            fiscal_year=2025,
            fiscal_period="FY",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]
    update_result = update_events[-1].payload["result"]
    assert update_result["status"] == "ok"
    assert update_result["filing_action"] == "update"
    assert update_result["document_version"] == "v2"


@pytest.mark.asyncio
async def test_upload_material_stream_overwrite_resets_single_document(tmp_path: Path) -> None:
    """验证 upload_material_stream 的 overwrite 会重置当前 material 文档而非保留旧文件。"""

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    pipeline._upload_service._convert_with_docling = lambda raw_data, stream_name: {  # type: ignore[attr-defined]
        "name": stream_name,
        "format": "docling",
    }
    old_file = tmp_path / "deck_old.pdf"
    new_file = tmp_path / "deck_new.pdf"
    old_file.write_text("old", encoding="utf-8")
    new_file.write_text("new", encoding="utf-8")

    first_events = [
        event
        async for event in pipeline.upload_material_stream(
            ticker="000001",
            action=None,
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=[old_file],
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]
    document_id = str(first_events[-1].payload["result"]["document_id"])

    second_events = [
        event
        async for event in pipeline.upload_material_stream(
            ticker="000001",
            action=None,
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=[new_file],
            company_id="000001",
            company_name="平安银行",
            overwrite=True,
        )
    ]
    second_result = second_events[-1].payload["result"]
    assert second_result["status"] == "ok"
    assert second_result["material_action"] == "update"
    assert second_result["document_id"] == document_id

    handle = pipeline._source_repository.get_source_handle("000001", document_id, SourceKind.MATERIAL)  # type: ignore[attr-defined]
    file_names = sorted(meta.uri.split("/")[-1] for meta in pipeline._blob_repository.list_files(handle))  # type: ignore[attr-defined]
    assert file_names == ["deck_new.pdf", "deck_new_docling.json"]
