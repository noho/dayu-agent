"""港A股财报管线实现。

本模块负责 CN/HK 市场的 pipeline 装配、上传、下载与离线处理入口。下载链路
通过 ``cn_download_workflow`` 编排，文档存取统一经 ``dayu.fins.storage`` 仓储。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from dayu.contracts.cancellation import CancelledError
from dayu.log import Log
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.fins.docling_export import PdfToDoclingJsonBytes, convert_pdf_bytes_to_docling_json_bytes
from dayu.fins.domain.document_models import ProcessedHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.downloaders.cninfo_downloader import CninfoDiscoveryClient
from dayu.fins.downloaders.hkexnews_downloader import HkexnewsDiscoveryClient
from dayu.fins.ingestion.pipeline_backends import PipelineIngestionBackend
from dayu.fins.ingestion.process_events import ProcessEvent, ProcessEventType
from dayu.fins.ingestion.service import FinsIngestionService
from dayu.fins.pipelines.cn_download_pdf_gate import (
    CnDownloadPdfGateProtocol,
    NoopCnDownloadPdfGate,
)
from dayu.fins.pipelines.cn_download_protocols import CnReportDiscoveryClientProtocol
from dayu.fins.pipelines.cn_download_workflow import run_cn_download_stream_impl
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    DocumentBlobRepositoryProtocol,
    FsCompanyMetaRepository,
    FsDocumentBlobRepository,
    FsFilingMaintenanceRepository,
    FsProcessedDocumentRepository,
    FsSourceDocumentRepository,
    FilingMaintenanceRepositoryProtocol,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from dayu.fins.ticker_normalization import try_normalize_ticker
from .download_events import DownloadEvent, DownloadEventType
from .base import PipelineProtocol
from .docling_upload_service import (
    DoclingUploadService,
    build_cn_filing_ids,
    build_material_ids,
    derive_report_kind,
    normalize_cn_fiscal_period,
    reset_upload_target_for_overwrite,
    resolve_upload_action,
    validate_material_upload_ids,
)
from .processing_helpers import (
    filter_requested_document_ids as _filter_requested_document_ids_common,
    extract_process_identity_fields as _extract_process_identity_fields_common,
    log_process_document_result as _log_process_document_result_common,
    resolve_expected_parser_version as _resolve_expected_parser_version_common,
)
from .processed_snapshot_helpers import (
    cleanup_processed_snapshot_dir as _cleanup_processed_snapshot_dir_common,
    clear_processed_documents as _clear_processed_documents_common,
    match_snapshot_files as _match_snapshot_files_common,
    safe_read_snapshot_meta as _safe_read_snapshot_meta_common,
)
from .tool_snapshot_export import (
    TOOL_SNAPSHOT_SCHEMA_VERSION,
    build_snapshot_file_names,
    export_tool_snapshot,
)
from .upload_progress_helpers import (
    map_upload_file_event_to_filing_event_type as _map_upload_file_event_to_filing_event_type,
    map_upload_file_event_to_material_event_type as _map_upload_file_event_to_material_event_type,
)
from .upload_filing_events import UploadFilingEvent, UploadFilingEventType
from .upload_material_events import UploadMaterialEvent, UploadMaterialEventType
from .upload_company_meta import build_upload_company_id, upsert_company_meta_for_upload


def _raise_if_cancelled(
    *,
    module: str,
    ticker: str,
    document_id: str,
    cancel_checker: Callable[[], bool] | None,
) -> None:
    """在 CN 同步单文档处理阶段边界检查取消请求。"""

    if cancel_checker is None or not cancel_checker():
        return
    Log.info(
        "单文档预处理收到取消请求，停止执行: "
        f"ticker={ticker} document_id={document_id}",
        module=module,
    )
    raise CancelledError("操作已被取消")


class CnPipeline(PipelineProtocol):
    """港A股管线骨架实现。"""

    PIPELINE_NAME = "cn"
    MODULE = "FINS.CN_PIPELINE"

    def __init__(
        self,
        *,
        processor_registry: ProcessorRegistry,
        company_repository: CompanyMetaRepositoryProtocol | None = None,
        source_repository: SourceDocumentRepositoryProtocol | None = None,
        processed_repository: ProcessedDocumentRepositoryProtocol | None = None,
        blob_repository: DocumentBlobRepositoryProtocol | None = None,
        filing_maintenance_repository: FilingMaintenanceRepositoryProtocol | None = None,
        cn_discovery_client: CnReportDiscoveryClientProtocol | None = None,
        hk_discovery_client: CnReportDiscoveryClientProtocol | None = None,
        pdf_download_gate: CnDownloadPdfGateProtocol | None = None,
        convert_pdf_to_docling_json: PdfToDoclingJsonBytes | None = None,
        workspace_root: Optional[Path] = None,
    ) -> None:
        """初始化港A股管线。

        Args:
            processor_registry: 处理器注册表（调用方显式注入）。
            company_repository: 可选公司元数据仓储实现。
            source_repository: 可选源文档仓储实现。
            processed_repository: 可选 processed 文档仓储实现。
            blob_repository: 可选文件对象仓储实现。
            filing_maintenance_repository: 可选 filing 维护治理仓储实现。
            cn_discovery_client: 可选 CN 巨潮 discovery client。
            hk_discovery_client: 可选 HK 披露易 discovery client。
            pdf_download_gate: 可选 PDF 下载段 gate。
            convert_pdf_to_docling_json: 可选 PDF 到 Docling JSON 转换函数。
            workspace_root: 工作区根目录。
        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        if processor_registry is None:
            raise ValueError("processor_registry 必须由调用方显式传入")
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._processor_registry = processor_registry
        repository_set = build_fs_repository_set(workspace_root=self._workspace_root)
        self._company_repository = company_repository or FsCompanyMetaRepository(
            self._workspace_root,
            repository_set=repository_set,
        )
        self._source_repository = source_repository or FsSourceDocumentRepository(
            self._workspace_root,
            repository_set=repository_set,
        )
        self._processed_repository = processed_repository or FsProcessedDocumentRepository(
            self._workspace_root,
            repository_set=repository_set,
        )
        self._blob_repository = blob_repository or FsDocumentBlobRepository(
            self._workspace_root,
            repository_set=repository_set,
        )
        self._filing_maintenance_repository = (
            filing_maintenance_repository
            or FsFilingMaintenanceRepository(
                self._workspace_root,
                repository_set=repository_set,
            )
        )
        self._cn_discovery_client = cn_discovery_client or CninfoDiscoveryClient()
        self._hk_discovery_client = hk_discovery_client or HkexnewsDiscoveryClient()
        self._pdf_download_gate = pdf_download_gate or NoopCnDownloadPdfGate()
        self._convert_pdf_to_docling_json = (
            convert_pdf_to_docling_json or convert_pdf_bytes_to_docling_json_bytes
        )
        self._user_agent: Optional[str] = None
        self._sleep_seconds = 0.0
        self._max_retries = 3
        self._upload_service = DoclingUploadService(
            source_repository=self._source_repository,
            blob_repository=self._blob_repository,
        )
        self._ingestion_service = FinsIngestionService(
            backend=PipelineIngestionBackend(self),
        )
        Log.debug(
            f"初始化港A股管线: workspace_root={self._workspace_root}",
            module=self.MODULE,
        )

    @property
    def company_meta_repository(self) -> CompanyMetaRepositoryProtocol:
        """返回公司元数据仓储。"""

        return self._company_repository

    @property
    def source_repository(self) -> SourceDocumentRepositoryProtocol:
        """返回 source 文档仓储。"""

        return self._source_repository

    @property
    def blob_repository(self) -> DocumentBlobRepositoryProtocol:
        """返回 blob 仓储。"""

        return self._blob_repository

    @property
    def processed_repository(self) -> ProcessedDocumentRepositoryProtocol:
        """返回 processed 仓储。"""

        return self._processed_repository

    @property
    def filing_maintenance_repository(self) -> FilingMaintenanceRepositoryProtocol:
        """返回 filing 维护仓储。"""

        return self._filing_maintenance_repository

    @property
    def cn_discovery_client(self) -> CnReportDiscoveryClientProtocol:
        """返回 CN 巨潮 discovery client。"""

        return self._cn_discovery_client

    @property
    def hk_discovery_client(self) -> CnReportDiscoveryClientProtocol:
        """返回 HK 披露易 discovery client。"""

        return self._hk_discovery_client

    @property
    def pdf_download_gate(self) -> CnDownloadPdfGateProtocol:
        """返回 PDF 下载段 gate。"""

        return self._pdf_download_gate

    @property
    def convert_pdf_to_docling_json(self) -> PdfToDoclingJsonBytes:
        """返回 PDF 到 Docling JSON 转换函数。"""

        return self._convert_pdf_to_docling_json

    @property
    def user_agent(self) -> Optional[str]:
        """返回下载 User-Agent。"""

        return self._user_agent

    @property
    def sleep_seconds(self) -> float:
        """返回下载请求间隔。"""

        return self._sleep_seconds

    @property
    def max_retries(self) -> int:
        """返回下载最大重试次数。"""

        return self._max_retries

    @property
    def ingestion_service(self) -> FinsIngestionService:
        """返回共享长事务服务。

        Args:
            无。

        Returns:
            当前 pipeline 绑定的共享长事务服务。

        Raises:
            无。
        """

        return self._ingestion_service

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
        """执行 CN/HK 下载同步入口。

        Args:
            ticker: 股票代码。
            form_type: 可选文档类型。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否强制覆盖。
            rebuild: 是否仅基于本地已下载数据重建 `meta/manifest`。
            ticker_aliases: 可选公司 alias 列表；写入公司级 meta 时合并。
            cancel_checker: 可选取消检查函数。

        Returns:
            下载结果字典。

        Raises:
            无。
        """

        result = self._ingestion_service.download(
            ticker=ticker,
            form_type=form_type,
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
            rebuild=rebuild,
            ticker_aliases=ticker_aliases,
            cancel_checker=cancel_checker,
        )
        return result

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
    ) -> AsyncIterator[DownloadEvent]:
        """执行流式下载（共享服务包装器）。

        Args:
            ticker: 股票代码。
            form_type: 可选文档类型。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否强制覆盖。
            rebuild: 是否仅基于本地已下载数据重建 `meta/manifest`。
            ticker_aliases: 可选公司 alias 列表；写入公司级 meta 时合并。
            cancel_checker: 可选取消检查函数。

        Yields:
            下载流程事件。

        Raises:
            无。
        """

        async for event in self._ingestion_service.download_stream(
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

    async def download_stream_impl(
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
    ) -> AsyncIterator[DownloadEvent]:
        """执行流式下载。

        Args:
            ticker: 股票代码。
            form_type: 可选文档类型。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否强制覆盖。
            rebuild: 是否仅基于本地已下载数据重建 `meta/manifest`。
            ticker_aliases: 可选公司 alias 列表。
            cancel_checker: 可选取消检查函数。

        Yields:
            下载流程事件。

        Raises:
            无。
        """

        async for event in run_cn_download_stream_impl(
            self,
            ticker=ticker,
            form_type=form_type,
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
            rebuild=rebuild,
            ticker_aliases=ticker_aliases,
            cancel_checker=cancel_checker,
            module=self.MODULE,
            pipeline_name=self.PIPELINE_NAME,
        ):
            yield event

    def upload_filing(
        self,
        ticker: str,
        action: Optional[str],
        files: list[Path],
        fiscal_year: int,
        fiscal_period: str,
        amended: bool = False,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """执行财报上传（同步包装器）。

        Args:
            ticker: 股票代码。
            action: 可选动作类型（create/update/delete）；为空时自动判定。
            files: 上传文件列表（create/update 需要）。
            fiscal_year: 财年。
            fiscal_period: 财季或年度标识。
            amended: 是否修订版。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 可选兼容字段；公司 ID 会由 ticker 归一化结果自动生成。
            company_name: 公司名称（create/update 在公司级 meta 缺失时必填，可由 infer 补齐）。
            ticker_aliases: 可选 ticker alias 列表；用于初始化公司级 meta。
            overwrite: 是否强制覆盖。

        Returns:
            上传结果字典。

        Raises:
            RuntimeError: 当前线程存在运行中的事件循环时抛出。
        """

        return _run_async_pipeline_sync(
            _collect_upload_result_from_events(
                self.upload_filing_stream(
                    ticker=ticker,
                    action=action,
                    files=files,
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    amended=amended,
                    filing_date=filing_date,
                    report_date=report_date,
                    company_id=company_id,
                    company_name=company_name,
                    ticker_aliases=ticker_aliases,
                    overwrite=overwrite,
                ),
                stream_name="upload_filing_stream",
            )
        )

    async def upload_filing_stream(
        self,
        ticker: str,
        action: Optional[str],
        files: list[Path],
        fiscal_year: int,
        fiscal_period: str,
        amended: bool = False,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> AsyncIterator[UploadFilingEvent]:
        """执行流式财报上传。

        Args:
            ticker: 股票代码。
            action: 可选动作类型（create/update/delete）；为空时自动判定。
            files: 上传文件列表（create/update 需要）。
            fiscal_year: 财年。
            fiscal_period: 财季或年度标识。
            amended: 是否修订版。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 可选兼容字段；公司 ID 会由 ticker 归一化结果自动生成。
            company_name: 公司名称（create/update 在公司级 meta 缺失时必填，可由 infer 补齐）。
            ticker_aliases: 可选 ticker alias 列表；用于初始化公司级 meta。
            overwrite: 是否强制覆盖。

        Yields:
            上传过程事件流。

        Raises:
            RuntimeError: 上传执行失败时抛出。
        """

        normalized_ticker = _normalize_ticker(ticker)
        normalized_company_id = build_upload_company_id(normalized_ticker)
        normalized_period = normalize_cn_fiscal_period(fiscal_period)
        form_type = normalized_period
        requested_action = str(action or "").strip().lower() or None
        document_id, internal_document_id = build_cn_filing_ids(
            ticker=normalized_ticker,
            form_type=form_type,
            fiscal_year=fiscal_year,
            fiscal_period=normalized_period,
            amended=amended,
        )
        previous_meta = self._safe_get_document_meta(
            normalized_ticker,
            document_id,
            SourceKind.FILING,
        )
        resolved_action = resolve_upload_action(action, previous_meta)
        yield UploadFilingEvent(
            event_type=UploadFilingEventType.UPLOAD_STARTED,
            ticker=normalized_ticker,
            document_id=document_id,
            payload={
                "action": resolved_action,
                "requested_action": requested_action,
                "resolved_action": resolved_action,
                "fiscal_year": fiscal_year,
                "fiscal_period": normalized_period,
                "amended": amended,
                "filing_date": filing_date,
                "report_date": report_date,
                "company_id": normalized_company_id,
                "company_name": company_name,
                "ticker_aliases": ticker_aliases,
                "overwrite": overwrite,
                "file_count": len(files or []),
            },
        )
        try:
            upsert_company_meta_for_upload(
                repository=self._company_repository,
                ticker=normalized_ticker,
                action=resolved_action,
                company_id=company_id,
                company_name=company_name,
                ticker_aliases=ticker_aliases,
            )
            reset_upload_target_for_overwrite(
                source_repository=self._source_repository,
                ticker=normalized_ticker,
                document_id=document_id,
                source_kind=SourceKind.FILING,
                action=resolved_action,
                overwrite=overwrite,
                previous_meta=previous_meta,
            )
            upload_result = self._upload_service.execute_upload(
                ticker=normalized_ticker,
                source_kind=SourceKind.FILING,
                action=resolved_action,
                document_id=document_id,
                internal_document_id=internal_document_id,
                form_type=form_type,
                files=files,
                overwrite=overwrite,
                meta={
                    "company_id": normalized_company_id,
                    "ingest_method": "upload",
                    "fiscal_year": fiscal_year,
                    "fiscal_period": normalized_period,
                    "report_kind": derive_report_kind(normalized_period),
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "amended": amended,
                },
            )
            for file_event in upload_result.file_events:
                yield UploadFilingEvent(
                    event_type=_map_upload_file_event_to_filing_event_type(file_event),
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={"name": file_event.name, **file_event.payload},
                )
            final_result = self._build_result(
                action="upload_filing",
                ticker=normalized_ticker,
                filing_action=resolved_action,
                requested_action=requested_action,
                resolved_action=resolved_action,
                files=[str(path) for path in files],
                fiscal_year=fiscal_year,
                fiscal_period=normalized_period,
                amended=amended,
                filing_date=filing_date,
                report_date=report_date,
                company_id=normalized_company_id,
                company_name=company_name,
                ticker_aliases=ticker_aliases,
                overwrite=overwrite,
                **upload_result.payload,
                status=_resolve_upload_status(upload_result.status),
            )
            yield UploadFilingEvent(
                event_type=UploadFilingEventType.UPLOAD_COMPLETED,
                ticker=normalized_ticker,
                document_id=document_id,
                payload={"result": final_result},
            )
        except Exception as exc:
            failed_result = self._build_result(
                action="upload_filing",
                ticker=normalized_ticker,
                filing_action=resolved_action,
                requested_action=requested_action,
                resolved_action=resolved_action,
                files=[str(path) for path in files],
                fiscal_year=fiscal_year,
                fiscal_period=normalized_period,
                amended=amended,
                filing_date=filing_date,
                report_date=report_date,
                company_id=normalized_company_id,
                company_name=company_name,
                ticker_aliases=ticker_aliases,
                overwrite=overwrite,
                document_id=document_id,
                status="failed",
                message=str(exc),
            )
            yield UploadFilingEvent(
                event_type=UploadFilingEventType.UPLOAD_FAILED,
                ticker=normalized_ticker,
                document_id=document_id,
                payload={"error": str(exc), "result": failed_result},
            )

    def upload_material(
        self,
        ticker: str,
        action: Optional[str],
        form_type: str,
        material_name: str,
        files: Optional[list[Path]] = None,
        document_id: Optional[str] = None,
        internal_document_id: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """执行材料上传（同步包装器）。

        Args:
            ticker: 股票代码。
            action: 可选动作类型（create/update/delete）；为空时自动判定。
            form_type: 材料类型。
            material_name: 材料名称。
            files: 可选上传文件列表。
            document_id: 可选文档 ID。
            internal_document_id: 可选内部文档 ID。
            fiscal_year: 可选财年；提供时参与稳定 document_id 生成。
            fiscal_period: 可选财期；提供时参与稳定 document_id 生成。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 可选兼容字段；公司 ID 会由 ticker 归一化结果自动生成。
            company_name: 公司名称（create/update 在公司级 meta 缺失时必填，可由 infer 补齐）。
            ticker_aliases: 可选 ticker alias 列表；用于初始化公司级 meta。
            overwrite: 是否强制覆盖。

        Returns:
            上传结果字典。

        Raises:
            RuntimeError: 当前线程存在运行中的事件循环时抛出。
        """

        return _run_async_pipeline_sync(
            _collect_upload_result_from_events(
                self.upload_material_stream(
                    ticker=ticker,
                    action=action,
                    form_type=form_type,
                    material_name=material_name,
                    files=files,
                    document_id=document_id,
                    internal_document_id=internal_document_id,
                    fiscal_year=fiscal_year,
                    fiscal_period=fiscal_period,
                    filing_date=filing_date,
                    report_date=report_date,
                    company_id=company_id,
                    company_name=company_name,
                    ticker_aliases=ticker_aliases,
                    overwrite=overwrite,
                ),
                stream_name="upload_material_stream",
            )
        )

    async def upload_material_stream(
        self,
        ticker: str,
        action: Optional[str],
        form_type: str,
        material_name: str,
        files: Optional[list[Path]] = None,
        document_id: Optional[str] = None,
        internal_document_id: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> AsyncIterator[UploadMaterialEvent]:
        """执行流式材料上传。

        Args:
            ticker: 股票代码。
            action: 可选动作类型（create/update/delete）；为空时自动判定。
            form_type: 材料类型。
            material_name: 材料名称。
            files: 可选上传文件列表。
            document_id: 可选文档 ID。
            internal_document_id: 可选内部文档 ID。
            fiscal_year: 可选财年；提供时参与稳定 document_id 生成。
            fiscal_period: 可选财期；提供时参与稳定 document_id 生成。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 可选兼容字段；公司 ID 会由 ticker 归一化结果自动生成。
            company_name: 公司名称（create/update 在公司级 meta 缺失时必填，可由 infer 补齐）。
            ticker_aliases: 可选 ticker alias 列表；用于初始化公司级 meta。
            overwrite: 是否强制覆盖。

        Yields:
            上传过程事件流。

        Raises:
            RuntimeError: 上传执行失败时抛出。
        """

        file_list = files or []
        normalized_ticker = _normalize_ticker(ticker)
        normalized_company_id = build_upload_company_id(normalized_ticker)
        normalized_fiscal_period = str(fiscal_period or "").strip().upper() or None
        stable_document_id, stable_internal_document_id = build_material_ids(
            form_type=form_type,
            material_name=material_name,
            fiscal_year=fiscal_year,
            fiscal_period=normalized_fiscal_period,
        )
        resolved_document_id, resolved_internal_id = validate_material_upload_ids(
            stable_document_id=stable_document_id,
            stable_internal_document_id=stable_internal_document_id,
            document_id=document_id,
            internal_document_id=internal_document_id,
        )
        previous_meta = self._safe_get_document_meta(
            normalized_ticker,
            resolved_document_id,
            SourceKind.MATERIAL,
        )
        requested_action = str(action or "").strip().lower() or None
        resolved_action = resolve_upload_action(action, previous_meta)
        yield UploadMaterialEvent(
            event_type=UploadMaterialEventType.UPLOAD_STARTED,
            ticker=normalized_ticker,
            document_id=resolved_document_id,
            payload={
                "action": resolved_action,
                "requested_action": requested_action,
                "resolved_action": resolved_action,
                "form_type": form_type,
                "material_name": material_name,
                "internal_document_id": resolved_internal_id,
                "fiscal_year": fiscal_year,
                "fiscal_period": normalized_fiscal_period,
                "filing_date": filing_date,
                "report_date": report_date,
                "company_id": normalized_company_id,
                "company_name": company_name,
                "ticker_aliases": ticker_aliases,
                "overwrite": overwrite,
                "file_count": len(file_list),
            },
        )
        try:
            upsert_company_meta_for_upload(
                repository=self._company_repository,
                ticker=normalized_ticker,
                action=resolved_action,
                company_id=company_id,
                company_name=company_name,
                ticker_aliases=ticker_aliases,
            )
            reset_upload_target_for_overwrite(
                source_repository=self._source_repository,
                ticker=normalized_ticker,
                document_id=resolved_document_id,
                source_kind=SourceKind.MATERIAL,
                action=resolved_action,
                overwrite=overwrite,
                previous_meta=previous_meta,
            )
            upload_result = self._upload_service.execute_upload(
                ticker=normalized_ticker,
                source_kind=SourceKind.MATERIAL,
                action=resolved_action,
                document_id=resolved_document_id,
                internal_document_id=resolved_internal_id,
                form_type=form_type,
                files=file_list,
                overwrite=overwrite,
                meta={
                    "company_id": normalized_company_id,
                    "ingest_method": "upload",
                    "material_name": material_name,
                    "fiscal_year": fiscal_year,
                    "fiscal_period": normalized_fiscal_period,
                    "filing_date": filing_date,
                    "report_date": report_date,
                },
            )
            for file_event in upload_result.file_events:
                yield UploadMaterialEvent(
                    event_type=_map_upload_file_event_to_material_event_type(file_event),
                    ticker=normalized_ticker,
                    document_id=resolved_document_id,
                    payload={"name": file_event.name, **file_event.payload},
                )
            final_result = self._build_result(
                action="upload_material",
                ticker=normalized_ticker,
                material_action=resolved_action,
                requested_action=requested_action,
                resolved_action=resolved_action,
                form_type=form_type,
                material_name=material_name,
                files=[str(path) for path in file_list],
                fiscal_year=fiscal_year,
                fiscal_period=normalized_fiscal_period,
                filing_date=filing_date,
                report_date=report_date,
                company_id=normalized_company_id,
                company_name=company_name,
                overwrite=overwrite,
                **upload_result.payload,
                status=_resolve_upload_status(upload_result.status),
            )
            yield UploadMaterialEvent(
                event_type=UploadMaterialEventType.UPLOAD_COMPLETED,
                ticker=normalized_ticker,
                document_id=resolved_document_id,
                payload={"result": final_result},
            )
        except Exception as exc:
            failed_result = self._build_result(
                action="upload_material",
                ticker=normalized_ticker,
                material_action=resolved_action,
                requested_action=requested_action,
                resolved_action=resolved_action,
                form_type=form_type,
                material_name=material_name,
                files=[str(path) for path in file_list],
                document_id=resolved_document_id,
                internal_document_id=resolved_internal_id,
                fiscal_year=fiscal_year,
                fiscal_period=normalized_fiscal_period,
                filing_date=filing_date,
                report_date=report_date,
                company_id=normalized_company_id,
                company_name=company_name,
                overwrite=overwrite,
                status="failed",
                message=str(exc),
            )
            yield UploadMaterialEvent(
                event_type=UploadMaterialEventType.UPLOAD_FAILED,
                ticker=normalized_ticker,
                document_id=resolved_document_id,
                payload={"error": str(exc), "result": failed_result},
            )

    def process(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """执行全量离线快照导出（共享服务包装器）。

        Args:
            ticker: 股票代码。
            overwrite: 是否强制覆盖。
            ci: 是否追加导出 `search_document/query_xbrl_facts` 快照。
            document_ids: 可选文档 ID 列表；传入时仅处理这些文档。

        Returns:
            处理结果字典。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        return self._ingestion_service.process(
            ticker=ticker,
            overwrite=overwrite,
            ci=ci,
            document_ids=document_ids,
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
        """执行流式全量离线快照导出（共享服务包装器）。

        Args:
            ticker: 股票代码。
            overwrite: 是否强制覆盖。
            ci: 是否追加导出 `search_document/query_xbrl_facts` 快照。
            document_ids: 可选文档 ID 列表；传入时仅处理这些文档。
            cancel_checker: 可选取消检查函数。

        Yields:
            预处理事件流。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        async for event in self._ingestion_service.process_stream(
            ticker=ticker,
            overwrite=overwrite,
            ci=ci,
            document_ids=document_ids,
            cancel_checker=cancel_checker,
        ):
            yield event

    async def process_stream_impl(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: Optional[list[str]] = None,
        *,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> AsyncIterator[ProcessEvent]:
        """执行流式全量离线快照导出。

        Args:
            ticker: 股票代码。
            overwrite: 是否强制覆盖。
            ci: 是否追加导出 `search_document/query_xbrl_facts` 快照。
            document_ids: 可选文档 ID 列表；传入时仅处理这些文档。
            cancel_checker: 可选取消检查函数，仅在文档边界生效。

        Yields:
            预处理事件流。

        Raises:
            RuntimeError: 执行失败时抛出。
        """
        normalized_ticker = _normalize_ticker(ticker)

        # 定向 process 只应重建指定文档，不能清空整个 processed 目录。
        if overwrite and document_ids is None:
            _clear_processed_dir(self._processed_repository, normalized_ticker)

        filing_ids = _filter_requested_document_ids_common(
            self._source_repository.list_source_document_ids(normalized_ticker, SourceKind.FILING),
            document_ids,
        )
        filing_results: list[dict[str, Any]] = []
        material_ids = _filter_requested_document_ids_common(
            self._source_repository.list_source_document_ids(normalized_ticker, SourceKind.MATERIAL),
            document_ids,
        )
        yield ProcessEvent(
            event_type=ProcessEventType.PIPELINE_STARTED,
            ticker=normalized_ticker,
            payload={
                "overwrite": overwrite,
                "ci": ci,
                "requested_document_ids": list(document_ids) if document_ids is not None else None,
                "filing_total": len(filing_ids),
                "material_total": len(material_ids),
                "total_documents": len(filing_ids) + len(material_ids),
            },
        )
        for document_id in filing_ids:
            if cancel_checker is not None and cancel_checker():
                Log.info(
                    f"预处理任务收到取消请求，文档边界停止: ticker={normalized_ticker}",
                    module=self.MODULE,
                )
                break
            source_meta = self._safe_get_document_meta(normalized_ticker, document_id, SourceKind.FILING)
            if source_meta is None:
                item = {
                    "document_id": document_id,
                    "status": "skipped",
                    "reason": "missing_meta",
                }
                filing_results.append(item)
                _log_process_document_result_common(
                    module=self.MODULE,
                    ticker=normalized_ticker,
                    source_kind=SourceKind.FILING,
                    result=item,
                )
                yield ProcessEvent(
                    event_type=ProcessEventType.DOCUMENT_SKIPPED,
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={"source_kind": SourceKind.FILING.value, "reason": "missing_meta", "result_summary": item},
                )
                continue
            if not bool(source_meta.get("ingest_complete", False)):
                item = {
                    "document_id": document_id,
                    "status": "skipped",
                    "reason": "ingest_incomplete",
                    **_extract_process_identity_fields_common(source_meta=source_meta),
                }
                filing_results.append(item)
                _log_process_document_result_common(
                    module=self.MODULE,
                    ticker=normalized_ticker,
                    source_kind=SourceKind.FILING,
                    result=item,
                )
                yield ProcessEvent(
                    event_type=ProcessEventType.DOCUMENT_SKIPPED,
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={"source_kind": SourceKind.FILING.value, "reason": "ingest_incomplete", "result_summary": item},
                )
                continue
            if bool(source_meta.get("is_deleted", False)):
                item = {
                    "document_id": document_id,
                    "status": "skipped",
                    "reason": "deleted",
                    **_extract_process_identity_fields_common(source_meta=source_meta),
                }
                filing_results.append(item)
                _log_process_document_result_common(
                    module=self.MODULE,
                    ticker=normalized_ticker,
                    source_kind=SourceKind.FILING,
                    result=item,
                )
                yield ProcessEvent(
                    event_type=ProcessEventType.DOCUMENT_SKIPPED,
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={"source_kind": SourceKind.FILING.value, "reason": "deleted", "result_summary": item},
                )
                continue
            yield ProcessEvent(
                event_type=ProcessEventType.DOCUMENT_STARTED,
                ticker=normalized_ticker,
                document_id=document_id,
                payload={"source_kind": SourceKind.FILING.value},
            )
            try:
                item = self.process_filing(
                    ticker=normalized_ticker,
                    document_id=document_id,
                    overwrite=overwrite,
                    ci=ci,
                )
                filing_results.append(item)
                _log_process_document_result_common(
                    module=self.MODULE,
                    ticker=normalized_ticker,
                    source_kind=SourceKind.FILING,
                    result=item,
                )
                event_type = (
                    ProcessEventType.DOCUMENT_SKIPPED
                    if item.get("status") == "skipped"
                    else ProcessEventType.DOCUMENT_COMPLETED
                )
                yield ProcessEvent(
                    event_type=event_type,
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={
                        "source_kind": SourceKind.FILING.value,
                        "reason": item.get("reason"),
                        "result_summary": item,
                    },
                )
            except Exception as exc:
                item = {
                    "document_id": document_id,
                    "status": "failed",
                    "reason": str(exc),
                    **_extract_process_identity_fields_common(source_meta=source_meta),
                }
                filing_results.append(item)
                _log_process_document_result_common(
                    module=self.MODULE,
                    ticker=normalized_ticker,
                    source_kind=SourceKind.FILING,
                    result=item,
                )
                yield ProcessEvent(
                    event_type=ProcessEventType.DOCUMENT_FAILED,
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={
                        "source_kind": SourceKind.FILING.value,
                        "reason": str(exc),
                        "result_summary": item,
                    },
                )

        material_results: list[dict[str, Any]] = []
        for document_id in material_ids:
            if cancel_checker is not None and cancel_checker():
                Log.info(
                    f"预处理任务收到取消请求，文档边界停止: ticker={normalized_ticker}",
                    module=self.MODULE,
                )
                break
            source_meta = self._safe_get_document_meta(normalized_ticker, document_id, SourceKind.MATERIAL)
            if source_meta is None:
                item = {
                    "document_id": document_id,
                    "status": "skipped",
                    "reason": "missing_meta",
                }
                material_results.append(item)
                _log_process_document_result_common(
                    module=self.MODULE,
                    ticker=normalized_ticker,
                    source_kind=SourceKind.MATERIAL,
                    result=item,
                )
                yield ProcessEvent(
                    event_type=ProcessEventType.DOCUMENT_SKIPPED,
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={"source_kind": SourceKind.MATERIAL.value, "reason": "missing_meta", "result_summary": item},
                )
                continue
            if not bool(source_meta.get("ingest_complete", False)):
                item = {
                    "document_id": document_id,
                    "status": "skipped",
                    "reason": "ingest_incomplete",
                    **_extract_process_identity_fields_common(source_meta=source_meta),
                }
                material_results.append(item)
                _log_process_document_result_common(
                    module=self.MODULE,
                    ticker=normalized_ticker,
                    source_kind=SourceKind.MATERIAL,
                    result=item,
                )
                yield ProcessEvent(
                    event_type=ProcessEventType.DOCUMENT_SKIPPED,
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={"source_kind": SourceKind.MATERIAL.value, "reason": "ingest_incomplete", "result_summary": item},
                )
                continue
            if bool(source_meta.get("is_deleted", False)):
                item = {
                    "document_id": document_id,
                    "status": "skipped",
                    "reason": "deleted",
                    **_extract_process_identity_fields_common(source_meta=source_meta),
                }
                material_results.append(item)
                _log_process_document_result_common(
                    module=self.MODULE,
                    ticker=normalized_ticker,
                    source_kind=SourceKind.MATERIAL,
                    result=item,
                )
                yield ProcessEvent(
                    event_type=ProcessEventType.DOCUMENT_SKIPPED,
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={"source_kind": SourceKind.MATERIAL.value, "reason": "deleted", "result_summary": item},
                )
                continue
            yield ProcessEvent(
                event_type=ProcessEventType.DOCUMENT_STARTED,
                ticker=normalized_ticker,
                document_id=document_id,
                payload={"source_kind": SourceKind.MATERIAL.value},
            )
            try:
                item = self.process_material(
                    ticker=normalized_ticker,
                    document_id=document_id,
                    overwrite=overwrite,
                    ci=ci,
                )
                material_results.append(item)
                _log_process_document_result_common(
                    module=self.MODULE,
                    ticker=normalized_ticker,
                    source_kind=SourceKind.MATERIAL,
                    result=item,
                )
                event_type = (
                    ProcessEventType.DOCUMENT_SKIPPED
                    if item.get("status") == "skipped"
                    else ProcessEventType.DOCUMENT_COMPLETED
                )
                yield ProcessEvent(
                    event_type=event_type,
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={
                        "source_kind": SourceKind.MATERIAL.value,
                        "reason": item.get("reason"),
                        "result_summary": item,
                    },
                )
            except Exception as exc:
                item = {
                    "document_id": document_id,
                    "status": "failed",
                    "reason": str(exc),
                    **_extract_process_identity_fields_common(source_meta=source_meta),
                }
                material_results.append(item)
                _log_process_document_result_common(
                    module=self.MODULE,
                    ticker=normalized_ticker,
                    source_kind=SourceKind.MATERIAL,
                    result=item,
                )
                yield ProcessEvent(
                    event_type=ProcessEventType.DOCUMENT_FAILED,
                    ticker=normalized_ticker,
                    document_id=document_id,
                    payload={
                        "source_kind": SourceKind.MATERIAL.value,
                        "reason": str(exc),
                        "result_summary": item,
                    },
                )

        filing_summary = {
            "total": len(filing_results),
            "processed": sum(1 for item in filing_results if item.get("status") == "processed"),
            "skipped": sum(1 for item in filing_results if item.get("status") == "skipped"),
            "failed": sum(1 for item in filing_results if item.get("status") == "failed"),
        }
        material_summary = {
            "total": len(material_results),
            "processed": sum(1 for item in material_results if item.get("status") == "processed"),
            "skipped": sum(1 for item in material_results if item.get("status") == "skipped"),
            "failed": sum(1 for item in material_results if item.get("status") == "failed"),
        }
        final_result = self._build_result(
            action="process",
            ticker=normalized_ticker,
            overwrite=overwrite,
            ci=ci,
            filings=filing_results,
            filing_summary=filing_summary,
            materials=material_results,
            material_summary=material_summary,
            status="cancelled" if cancel_checker is not None and cancel_checker() else "ok",
        )
        yield ProcessEvent(
            event_type=ProcessEventType.PIPELINE_COMPLETED,
            ticker=normalized_ticker,
            payload={"result": final_result},
        )

    def process_filing(
        self,
        ticker: str,
        document_id: str,
        overwrite: bool = False,
        ci: bool = False,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """执行单个财报离线快照导出。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            overwrite: 是否强制重处理。
            ci: 是否追加导出 `search_document/query_xbrl_facts` 快照。
            cancel_checker: 可选取消检查函数，仅在同步单文档处理阶段边界生效。

        Returns:
            单文档处理结果字典。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        return self._process_single_source_document(
            ticker=ticker,
            document_id=document_id,
            source_kind=SourceKind.FILING,
            overwrite=overwrite,
            ci=ci,
            action="process_filing",
            cancel_checker=cancel_checker,
        )

    def process_material(
        self,
        ticker: str,
        document_id: str,
        overwrite: bool = False,
        ci: bool = False,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """执行单个材料离线快照导出。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            overwrite: 是否强制重处理。
            ci: 是否追加导出 `search_document/query_xbrl_facts` 快照。
            cancel_checker: 可选取消检查函数，仅在同步单文档处理阶段边界生效。

        Returns:
            单材料处理结果字典。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        return self._process_single_source_document(
            ticker=ticker,
            document_id=document_id,
            source_kind=SourceKind.MATERIAL,
            overwrite=overwrite,
            ci=ci,
            action="process_material",
            cancel_checker=cancel_checker,
        )

    def _process_single_source_document(
        self,
        *,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
        overwrite: bool,
        ci: bool,
        action: str,
        cancel_checker: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        """处理单个源文档并导出工具快照。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 源类型（filing/material）。
            overwrite: 是否强制重处理。
            ci: 是否追加导出 `search_document/query_xbrl_facts` 快照。
            action: 返回结果中的动作名称。
            cancel_checker: 可选取消检查函数，仅在同步单文档处理阶段边界生效。

        Returns:
            处理结果字典。

        Raises:
            RuntimeError: 处理失败时抛出。
        """

        normalized_ticker = _normalize_ticker(ticker)
        _raise_if_cancelled(
            module=self.MODULE,
            ticker=normalized_ticker,
            document_id=document_id,
            cancel_checker=cancel_checker,
        )
        source_meta = self._source_repository.get_source_meta(
            ticker=normalized_ticker,
            document_id=document_id,
            source_kind=source_kind,
        )
        if not bool(source_meta.get("ingest_complete", False)):
            return self._build_result(
                action=action,
                ticker=normalized_ticker,
                document_id=document_id,
                overwrite=overwrite,
                ci=ci,
                status="skipped",
                reason="ingest_incomplete",
                **_extract_process_identity_fields_common(source_meta=source_meta),
            )
        if bool(source_meta.get("is_deleted", False)):
            return self._build_result(
                action=action,
                ticker=normalized_ticker,
                document_id=document_id,
                overwrite=overwrite,
                ci=ci,
                status="skipped",
                reason="deleted",
                **_extract_process_identity_fields_common(source_meta=source_meta),
            )

        source = self._source_repository.get_primary_source(
            ticker=normalized_ticker,
            document_id=document_id,
            source_kind=source_kind,
        )
        normalized_form_type = str(source_meta.get("form_type") or "") or None
        expected_parser_signature = _resolve_expected_parser_version_common(
            processor_registry=self._processor_registry,
            source=source,
            form_type=normalized_form_type,
        )
        snapshot_meta = self._safe_read_snapshot_meta(
            ticker=normalized_ticker,
            document_id=document_id,
        )
        if self._can_skip_snapshot_export(
            source_meta=source_meta,
            snapshot_meta=snapshot_meta,
            overwrite=overwrite,
            expected_parser_signature=expected_parser_signature,
            ci=ci,
            ticker=normalized_ticker,
            document_id=document_id,
        ):
            return self._build_result(
                action=action,
                ticker=normalized_ticker,
                document_id=document_id,
                overwrite=overwrite,
                ci=ci,
                status="skipped",
                reason="version_matched",
                **_extract_process_identity_fields_common(source_meta=source_meta),
            )

        _raise_if_cancelled(
            module=self.MODULE,
            ticker=normalized_ticker,
            document_id=document_id,
            cancel_checker=cancel_checker,
        )
        self._export_tool_snapshot_for_document(
            ticker=normalized_ticker,
            document_id=document_id,
            source_kind=source_kind,
            source_meta=source_meta,
            ci=ci,
            expected_parser_signature=expected_parser_signature,
            cancel_checker=cancel_checker,
        )
        _raise_if_cancelled(
            module=self.MODULE,
            ticker=normalized_ticker,
            document_id=document_id,
            cancel_checker=cancel_checker,
        )
        return self._build_result(
            action=action,
            ticker=normalized_ticker,
            document_id=document_id,
            overwrite=overwrite,
            ci=ci,
            status="processed",
            **_extract_process_identity_fields_common(source_meta=source_meta),
        )

    def _export_tool_snapshot_for_document(
        self,
        *,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
        source_meta: dict[str, Any],
        ci: bool,
        expected_parser_signature: str,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        """导出单文档工具快照文件。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 源文档类型。
            source_meta: 源文档元数据。
            ci: 是否追加导出 CI 专用快照。
            expected_parser_signature: 预期解析器签名。
            cancel_checker: 可选取消检查函数，用于快照导出阶段边界取消。

        Returns:
            无。

        Raises:
            OSError: 文件写入失败时抛出。
            RuntimeError: 处理器调用失败时抛出。
        """
        allowed_files = set(build_snapshot_file_names(ci=ci))
        self._cleanup_processed_snapshot_dir(
            ticker=ticker,
            document_id=document_id,
            allowed_files=allowed_files,
        )
        processed_handle = ProcessedHandle(ticker=ticker, document_id=document_id)
        export_tool_snapshot(
            company_repository=self._company_repository,
            source_repository=self._source_repository,
            processed_repository=self._processed_repository,
            blob_repository=self._blob_repository,
            processor_registry=self._processor_registry,
            processed_handle=processed_handle,
            ticker=ticker,
            document_id=document_id,
            source_kind=source_kind,
            source_meta=source_meta,
            ci=ci,
            expected_parser_signature=expected_parser_signature,
            cancel_checker=cancel_checker,
        )

    def _can_skip_snapshot_export(
        self,
        *,
        source_meta: dict[str, Any],
        snapshot_meta: Optional[dict[str, Any]],
        overwrite: bool,
        expected_parser_signature: str,
        ci: bool,
        ticker: str,
        document_id: str,
    ) -> bool:
        """判断当前文档是否可跳过快照导出。"""

        if overwrite:
            return False
        if snapshot_meta is None:
            return False
        if str(snapshot_meta.get("snapshot_schema_version", "")) != TOOL_SNAPSHOT_SCHEMA_VERSION:
            return False
        if str(snapshot_meta.get("source_document_version", "")) != str(source_meta.get("document_version", "")):
            return False
        if str(snapshot_meta.get("source_fingerprint", "")) != str(source_meta.get("source_fingerprint", "")):
            return False
        expected_from_snapshot = str(
            snapshot_meta.get(
                "expected_parser_signature",
                snapshot_meta.get("parser_signature", ""),
            )
            or ""
        )
        if expected_from_snapshot != expected_parser_signature:
            return False
        return self._match_snapshot_files(
            ticker=ticker,
            document_id=document_id,
            ci=ci,
        )

    def _match_snapshot_files(
        self,
        *,
        ticker: str,
        document_id: str,
        ci: bool,
    ) -> bool:
        """校验快照目录文件集合是否与当前模式严格一致。"""
        return _match_snapshot_files_common(
            repository=self._blob_repository,
            ticker=ticker,
            document_id=document_id,
            ci=ci,
        )

    def _cleanup_processed_snapshot_dir(
        self,
        *,
        ticker: str,
        document_id: str,
        allowed_files: set[str],
    ) -> None:
        """清理 `processed/{document_id}` 中非目标快照文件。"""
        _cleanup_processed_snapshot_dir_common(
            repository=self._blob_repository,
            ticker=ticker,
            document_id=document_id,
            allowed_files=allowed_files,
        )

    def _safe_read_snapshot_meta(
        self,
        *,
        ticker: str,
        document_id: str,
    ) -> Optional[dict[str, Any]]:
        """安全读取 `tool_snapshot_meta.json`。"""
        return _safe_read_snapshot_meta_common(
            repository=self._blob_repository,
            ticker=ticker,
            document_id=document_id,
        )

    def _build_result(self, action: str, **payload: Any) -> dict[str, Any]:
        """构建统一结果。

        Args:
            action: 动作名称。
            **payload: 结果负载字段。

        Returns:
            统一结构的结果字典。

        Raises:
            无。
        """

        return {
            "pipeline": self.PIPELINE_NAME,
            "action": action,
            "status": payload.pop("status", "placeholder"),
            **payload,
        }

    def _safe_get_document_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> Optional[dict[str, Any]]:
        """安全读取文档元数据。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 源文档类型。

        Returns:
            元数据字典；不存在时返回 `None`。

        Raises:
            ValueError: 元数据格式非法时抛出。
        """

        try:
            return self._source_repository.get_source_meta(
                ticker=ticker,
                document_id=document_id,
                source_kind=source_kind,
            )
        except FileNotFoundError:
            return None

    def _safe_get_processed_meta(self, ticker: str, document_id: str) -> Optional[dict[str, Any]]:
        """安全读取 processed 元数据。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。

        Returns:
            processed 元数据；不存在时返回 `None`。

        Raises:
            ValueError: 元数据格式非法时抛出。
        """

        try:
            return self._processed_repository.get_processed_meta(ticker=ticker, document_id=document_id)
        except FileNotFoundError:
            return None


def _normalize_ticker(ticker: str) -> str:
    """标准化 ticker。

    代理到 ``dayu.fins.ticker_normalization`` 真源；真源识别失败时
    回退到 ``strip().upper()``，确保公司名等异常输入仍能触发空值校验。

    Args:
        ticker: 原始股票代码。

    Returns:
        标准化后 ticker。

    Raises:
        ValueError: ticker 为空时抛出。
    """

    normalized_source = try_normalize_ticker(ticker)
    if normalized_source is not None:
        return normalized_source.canonical
    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("ticker 不能为空")
    return normalized


def _clear_processed_dir(repository: ProcessedDocumentRepositoryProtocol, ticker: str) -> None:
    """清空 processed 目录（用于 `--overwrite` 全量重建）。

    Args:
        repository: processed 文档仓储。
        ticker: 股票代码。

    Returns:
        无。

    Raises:
        OSError: 仓储清理失败时抛出。
    """

    _clear_processed_documents_common(repository=repository, ticker=ticker)


def _resolve_upload_status(upload_status: str) -> str:
    """映射上传结果状态到统一输出状态。

    Args:
        upload_status: 上传服务状态。

    Returns:
        Pipeline 对外状态值。

    Raises:
        无。
    """

    if upload_status == "uploaded":
        return "ok"
    return upload_status


async def _collect_upload_result_from_events(
    stream: AsyncIterator[UploadFilingEvent | UploadMaterialEvent],
    *,
    stream_name: str,
) -> dict[str, Any]:
    """从上传事件流中提取最终结果。

    Args:
        stream: 上传事件流。
        stream_name: 事件流名称（用于错误提示）。

    Returns:
        事件流中的最终结果字典。

    Raises:
        RuntimeError: 事件流未返回有效最终结果时抛出。
    """

    result: Optional[dict[str, Any]] = None
    async for event in stream:
        if event.event_type not in {"upload_completed", "upload_failed"}:
            continue
        payload_result = event.payload.get("result")
        if isinstance(payload_result, dict):
            result = payload_result
    if result is None:
        raise RuntimeError(f"{stream_name} 未返回最终结果")
    return result


def _run_async_pipeline_sync(coro: Any) -> dict[str, Any]:
    """在同步上下文执行协程。

    Args:
        coro: 协程对象。

    Returns:
        协程结果字典。

    Raises:
        RuntimeError: 当前线程已有事件循环时抛出。
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("检测到正在运行的事件循环，请改用 stream 异步接口")
