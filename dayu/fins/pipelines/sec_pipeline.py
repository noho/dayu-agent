"""美股管线实现。

维护说明(不拆分本模块):
    本模块约 1930 行, 核心是 SecPipeline 类(1624 行, 29 个方法),
    方法围绕 download -> upload -> process 生命周期, 共享 __init__
    注入的十余个依赖. 没有方法超过 80 行, 粒度已经足够细. 拆分类
    方法到子模块需要透传大量 self 依赖, 收益极低.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import inspect
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, BinaryIO, Callable, Optional, TypeVar, cast

from dayu.log import Log
from dayu.fins.domain.document_models import (
    CompanyMeta,
    FileObjectMeta,
    FilingCreateRequest,
    FilingManifestItem,
    FilingUpdateRequest,
    ProcessedHandle,
    SourceFileEntry,
    SourceHandle,
    now_iso8601,
)
from dayu.fins.ingestion.pipeline_backends import PipelineIngestionBackend
from dayu.fins.ingestion.process_events import ProcessEvent, ProcessEventType
from dayu.fins.ingestion.service import FinsIngestionService
from dayu.fins.downloaders.sec_downloader import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_SLEEP_SECONDS,
    DownloaderEvent,
    RemoteFileDescriptor,
    Sc13PartyRoles,
    SecDownloader,
    accession_to_no_dash,
    build_source_fingerprint,
)
from dayu.fins.resolver.market_resolver import MarketResolver
from dayu.fins.domain.enums import SourceKind
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    DocumentBlobRepositoryProtocol,
    FilingMaintenanceRepositoryProtocol,
    FsCompanyMetaRepository,
    FsDocumentBlobRepository,
    FsFilingMaintenanceRepository,
    FsProcessedDocumentRepository,
    FsSourceDocumentRepository,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from dayu.engine.processors.processor_registry import ProcessorRegistry

from .base import PipelineProtocol
from .docling_upload_service import DoclingUploadService
from .download_events import DownloadEvent, DownloadEventType
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
    build_conversion_started_events as _build_conversion_started_events,
    build_original_file_uploaded_events as _build_original_file_uploaded_events,
    map_upload_file_event_to_filing_event_type as _map_upload_file_event_to_filing_event_type,
    map_upload_file_event_to_material_event_type as _map_upload_file_event_to_material_event_type,
    should_emit_upload_file_event as _should_emit_upload_file_event,
)
from .upload_filing_events import UploadFilingEvent, UploadFilingEventType
from .upload_material_events import UploadMaterialEvent, UploadMaterialEventType
from .upload_company_meta import upsert_company_meta_for_upload
from .sec_6k_rules import (
    _SixKCandidateDiagnosis,
    _has_6k_exhibit_candidate,
    _has_6k_xbrl_instance,
    _is_positive_6k_classification,
    _select_6k_target_name,
    _select_best_positive_6k_candidate,
)
from .sec_company_meta import (
    extract_sec_ticker_aliases,
    merge_ticker_aliases,
    upsert_company_meta as _upsert_company_meta_impl,
)
from .sec_download_diagnostics import (
    warn_insufficient_filings,
    warn_xbrl_missing_filings,
)
from .sec_download_event_mapping import (
    build_download_filing_event_payload,
    build_file_result_from_downloader_event,
    map_file_status_to_event_type,
    normalize_download_file_result,
    summarize_failed_download_file_reasons,
)
from .sec_download_persistence import (
    build_file_entries as _build_file_entries_impl,
    build_rejected_store_file as _build_rejected_store_file_impl,
    build_store_file as _build_store_file_impl,
    mark_processed_reprocess_required as _mark_processed_reprocess_required_impl,
    persist_rejected_filing_artifact as _persist_rejected_filing_artifact_impl,
)
from .sec_download_state import (
    _has_same_file_name_set,
    _index_file_entries,
    _is_rejected as _is_rejected_impl,
    _load_rejection_registry as _load_rejection_registry_impl,
    _read_sec_cache_async,
    _record_rejection as _record_rejection_impl,
    _remote_files_equivalent_to_previous_meta,
    _save_rejection_registry as _save_rejection_registry_impl,
    _write_sec_cache_async,
)
from .sec_download_workflow import run_download_stream_impl as _run_download_stream_impl
from .sec_download_filing_workflow import run_download_single_filing_stream as _run_download_single_filing_stream
from .sec_filing_collection import (
    FilingRecord,
    classify_6k_remote_candidates,
    collect_filenums_from_table,
    collect_filings_from_table,
)
from .sec_fiscal_fields import (
    _resolve_download_fiscal_fields,
    _should_skip_financial_extraction,
)
from .sec_form_utils import (
    DEFAULT_FORMS_US,
    LOOKBACK_GRACE_DAYS,
    LOOKBACK_YEARS_BY_FORM,
    expand_form_aliases,
    increment_document_version,
    normalize_form,
    parse_date,
    split_form_input,
    subtract_years,
)
from .sec_process_workflow import run_process_single_document as _run_process_single_document
from .sec_process_workflow import run_process_stream_impl as _run_process_stream_impl
from .sec_rebuild_workflow import (
    SecRebuildWorkflowHost as _SecRebuildWorkflowHost,
    overwrite_rebuilt_meta as _overwrite_rebuilt_meta_impl,
    rebuild_download_artifacts as _rebuild_download_artifacts_impl,
)
from .sec_safe_meta_access import (
    resolve_document_version as _resolve_document_version_impl,
    safe_get_company_meta as _safe_get_company_meta_impl,
    safe_get_document_meta as _safe_get_document_meta_impl,
    safe_get_filing_source_meta as _safe_get_filing_source_meta_impl,
    safe_get_processed_meta as _safe_get_processed_meta_impl,
)
from .sec_sc13_filtering import (
    SecSc13WorkflowHost as _SecSc13WorkflowHost,
    extend_with_browse_edgar_sc13 as _extend_with_browse_edgar_sc13_impl,
    filter_sc13_by_direction as _filter_sc13_by_direction_impl,
    keep_latest_sc13_per_filer as _keep_latest_sc13_per_filer_impl,
    retry_sc13_if_empty as _retry_sc13_if_empty_impl,
    should_keep_sc13_direction as _should_keep_sc13_direction_impl,
    should_warn_missing_sc13,
)
from .sec_upload_workflow import (
    collect_upload_result_from_events as _collect_upload_result_from_events,
    run_upload_filing_stream as _run_upload_filing_stream,
    run_upload_material_stream as _run_upload_material_stream,
)

SEC_PIPELINE_DOWNLOAD_VERSION = "sec_pipeline_download_v1.2.0"
SEC_PIPELINE_PROCESS_SCHEMA_VERSION = "sec_pipeline_process_v1.0.0"


# ---------- 版本注入 wrappers ----------


def _is_rejected(
    registry: dict[str, dict[str, str]],
    document_id: str,
    overwrite: bool,
) -> bool:
    """判断 document_id 是否命中拒绝注册表。

    Args:
        registry: 拒绝注册表。
        document_id: 文档 ID。
        overwrite: 是否覆盖模式。

    Returns:
        命中返回 True。
    """

    return _is_rejected_impl(
        registry=registry,
        document_id=document_id,
        overwrite=overwrite,
        download_version=SEC_PIPELINE_DOWNLOAD_VERSION,
    )


def _record_rejection(
    registry: dict[str, dict[str, str]],
    document_id: str,
    reason: str,
    category: str,
    form_type: str,
    filing_date: str,
) -> None:
    """向拒绝注册表写入一条拒绝记录。

    Args:
        registry: 拒绝注册表（就地修改）。
        document_id: 文档 ID。
        reason: 拒绝原因标识。
        category: 筛选分类标签。
        form_type: 表单类型。
        filing_date: 申报日期。

    Returns:
        无。
    """

    _record_rejection_impl(
        registry=registry,
        document_id=document_id,
        reason=reason,
        category=category,
        form_type=form_type,
        filing_date=filing_date,
        download_version=SEC_PIPELINE_DOWNLOAD_VERSION,
    )


# ---------- 辅助 ----------


_AwaitableResult = TypeVar("_AwaitableResult")


async def _maybe_await(value: Awaitable[_AwaitableResult] | _AwaitableResult) -> _AwaitableResult:
    """按需等待可等待对象。"""

    if inspect.isawaitable(value):
        return await value
    return value


def _run_async_upload_sync(coro: Any) -> dict[str, Any]:
    """在同步上下文运行上传协程并返回结果。

    Args:
        coro: 协程对象。

    Returns:
        协程返回的结果字典。

    Raises:
        RuntimeError: 当前线程已有事件循环时抛出。
    """

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("检测到正在运行的事件循环，请改用 stream 异步接口")


# ---------- SecPipeline ----------


class SecPipeline(PipelineProtocol):
    """美股管线实现。"""

    PIPELINE_NAME = "sec"
    MODULE = "FINS.SEC_PIPELINE"

    def __init__(
        self,
        *,
        processor_registry: ProcessorRegistry,
        workspace_root: Optional[Path] = None,
        resolver_cls: type[MarketResolver] = MarketResolver,
        downloader: Optional[SecDownloader] = None,
        company_repository: CompanyMetaRepositoryProtocol | None = None,
        source_repository: SourceDocumentRepositoryProtocol | None = None,
        processed_repository: ProcessedDocumentRepositoryProtocol | None = None,
        blob_repository: DocumentBlobRepositoryProtocol | None = None,
        filing_maintenance_repository: FilingMaintenanceRepositoryProtocol | None = None,
        user_agent: Optional[str] = None,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        """初始化美股管线。

        Args:
            workspace_root: 工作区根目录。
            resolver_cls: 市场解析器类型。
            downloader: 可选下载器实例。
            company_repository: 可选公司元数据仓储实例。
            source_repository: 可选源文档仓储实例。
            processed_repository: 可选 processed 文档仓储实例。
            blob_repository: 可选文件对象仓储实例。
            filing_maintenance_repository: 可选 filing 维护治理仓储实例。
            processor_registry: 处理器注册表（必须显式传入）。
            user_agent: SEC User-Agent。
            sleep_seconds: 请求间隔秒数。
            max_retries: 下载重试次数。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        if processor_registry is None:
            raise ValueError("processor_registry 必须由调用方显式传入")
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._resolver_cls = resolver_cls
        self._downloader = downloader or SecDownloader(
            workspace_root=self._workspace_root,
        )
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
        self._filing_maintenance_repository = filing_maintenance_repository or FsFilingMaintenanceRepository(
            self._workspace_root,
            repository_set=repository_set,
        )
        self._processor_registry = processor_registry
        self._user_agent = user_agent
        self._sleep_seconds = sleep_seconds
        self._max_retries = max_retries
        self._upload_service = DoclingUploadService(
            source_repository=self._source_repository,
            blob_repository=self._blob_repository,
        )
        self._ingestion_service = FinsIngestionService(
            backend=PipelineIngestionBackend(self),
        )

    @property
    def ingestion_service(self) -> FinsIngestionService:
        """返回共享长事务服务。"""

        return self._ingestion_service

    # ========== download ==========

    def download(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """执行下载（同步聚合包装器）。

        Args:
            ticker: 股票代码。
            form_type: 可选文档类型。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否强制覆盖。
            rebuild: 是否仅基于本地已下载数据重建 `meta/manifest`。
            ticker_aliases: CLI 侧传入的 alias 列表；会与 SEC submissions alias 合并。

        Returns:
            下载结果字典。

        Raises:
            ValueError: ticker 不合法或市场不匹配时抛出。
        """

        return self._ingestion_service.download(
            ticker=ticker,
            form_type=form_type,
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
            rebuild=rebuild,
            ticker_aliases=ticker_aliases,
        )

    async def download_stream(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: Optional[list[str]] = None,
    ) -> AsyncIterator[DownloadEvent]:
        """执行流式下载（共享服务包装器）。

        Args:
            ticker: 股票代码。
            form_type: 可选文档类型。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否强制覆盖。
            rebuild: 是否仅基于本地已下载数据重建 `meta/manifest`。
            ticker_aliases: CLI 侧传入的 alias 列表。

        Yields:
            下载流程事件。

        Raises:
            ValueError: ticker 不合法或市场不匹配时抛出。
        """

        async for event in self._ingestion_service.download_stream(
            ticker=ticker,
            form_type=form_type,
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
            rebuild=rebuild,
            ticker_aliases=ticker_aliases,
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
            rebuild: 是否仅基于本地已下载数据重建。
            ticker_aliases: CLI 侧传入的 alias 列表。
            cancel_checker: 可选取消检查函数。

        Yields:
            下载流程事件。

        Raises:
            ValueError: ticker 不合法或市场不匹配时抛出。
        """

        async for event in _run_download_stream_impl(
            self,
            ticker=ticker,
            form_type=form_type,
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
            rebuild=rebuild,
            ticker_aliases=ticker_aliases,
            cancel_checker=cancel_checker,
            parse_date=parse_date,
            extract_sec_ticker_aliases=extract_sec_ticker_aliases,
            merge_ticker_aliases=merge_ticker_aliases,
            clear_filings_dir=lambda repo, ticker: repo.clear_filing_documents(ticker),
            load_rejection_registry=_load_rejection_registry_impl,
            save_rejection_registry=_save_rejection_registry_impl,
            should_warn_missing_sc13=should_warn_missing_sc13,
            warn_insufficient_filings=warn_insufficient_filings,
            warn_xbrl_missing_filings=warn_xbrl_missing_filings,
            cleanup_stale_filing_dirs=_cleanup_stale_filing_dirs,
            build_download_filing_event_payload=build_download_filing_event_payload,
        ):
            yield event

    def _rebuild_download_artifacts(
        self,
        *,
        ticker: str,
        form_type: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        overwrite: bool,
    ) -> dict[str, Any]:
        """基于本地已下载 filings 重建 meta/manifest。

        Args:
            ticker: 标准化后的股票代码。
            form_type: 可选 form 过滤。
            start_date: 可选起始日期过滤。
            end_date: 可选结束日期过滤。
            overwrite: 是否覆盖。

        Returns:
            与 download 一致结构的结果字典。

        Raises:
            ValueError: 过滤参数非法时抛出。
        """

        return _rebuild_download_artifacts_impl(
            cast(_SecRebuildWorkflowHost, self),
            ticker=ticker,
            form_type=form_type,
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
            pipeline_download_version=SEC_PIPELINE_DOWNLOAD_VERSION,
            expand_form_aliases=expand_form_aliases,
            split_form_input=split_form_input,
            parse_date=parse_date,
            normalize_form=normalize_form,
            overwrite_rebuilt_meta=_overwrite_rebuilt_meta_impl,
        )

    def _log_filing_download_result(self, ticker: str, filing_result: dict[str, Any]) -> None:
        """输出单个 filing 下载完成日志。

        Args:
            ticker: 股票代码。
            filing_result: 单个 filing 的下载结果字典。

        Returns:
            无。
        """

        document_id = str(filing_result.get("document_id", ""))
        status = str(filing_result.get("status", "unknown"))
        form_type = str(filing_result.get("form_type", ""))
        filing_date = filing_result.get("filing_date")
        report_date = filing_result.get("report_date")
        downloaded_files = int(filing_result.get("downloaded_files", 0) or 0)
        skipped_files = int(filing_result.get("skipped_files", 0) or 0)
        failed_files = filing_result.get("failed_files")
        failed_count = len(failed_files) if isinstance(failed_files, list) else 0
        skip_reason = filing_result.get("skip_reason")
        reason_code = filing_result.get("reason_code")
        reason_message = filing_result.get("reason_message")
        filter_category = filing_result.get("filter_category")
        Log.info(
            (
                "filing 下载完成: "
                f"ticker={ticker} document_id={document_id} status={status} form={form_type} "
                f"filing_date={filing_date} report_date={report_date} "
                f"downloaded_files={downloaded_files} skipped_files={skipped_files} "
                f"failed_files={failed_count} skip_reason={skip_reason} "
                f"reason_code={reason_code} reason_message={reason_message} "
                f"filter_category={filter_category}"
            ),
            module=self.MODULE,
        )

    async def _download_single_filing_stream(
        self,
        ticker: str,
        cik: str,
        filing: FilingRecord,
        overwrite: bool,
        rejection_registry: Optional[dict[str, dict[str, str]]] = None,
    ) -> AsyncIterator[DownloadEvent]:
        """下载单个 filing 并流式产出事件。

        Args:
            ticker: 股票代码。
            cik: CIK（无前导零）。
            filing: filing 记录。
            overwrite: 是否覆盖。
            rejection_registry: 拒绝注册表（可选）。

        Yields:
            文件级与 filing 级事件。

        Raises:
            RuntimeError: 关键路径异常时抛出。
        """

        async for event in _run_download_single_filing_stream(
            self,
            ticker=ticker,
            cik=cik,
            filing=filing,
            overwrite=overwrite,
            rejection_registry=rejection_registry,
            is_rejected=_is_rejected,
            record_rejection=_record_rejection,
            build_download_filing_event_payload=build_download_filing_event_payload,
            build_file_result_from_downloader_event=build_file_result_from_downloader_event,
            normalize_download_file_result=normalize_download_file_result,
            summarize_failed_download_file_reasons=summarize_failed_download_file_reasons,
            map_file_status_to_event_type=map_file_status_to_event_type,
            has_same_file_name_set=lambda remote_files, existing_files: _has_same_file_name_set(
                remote_files=remote_files,
                existing_files=existing_files,
            ),
            resolve_download_fiscal_fields=_resolve_download_fiscal_fields,
            index_file_entries=_index_file_entries,
            download_version=SEC_PIPELINE_DOWNLOAD_VERSION,
        ):
            yield event

    # ========== download: filtering ==========

    def _resolve_form_windows(
        self,
        form_type: Optional[str],
        start_date: Optional[str],
        end_date: dt.date,
    ) -> dict[str, dt.date]:
        """计算 form 到起始日期的映射。

        Args:
            form_type: 可选输入 form。
            start_date: 可选输入开始日期。
            end_date: 已确定结束日期。

        Returns:
            ``{form_type: start_date}`` 映射。

        Raises:
            ValueError: 输入参数不合法时抛出。
        """

        if form_type:
            explicit_forms = expand_form_aliases(split_form_input(form_type))
        else:
            explicit_forms = expand_form_aliases(list(DEFAULT_FORMS_US))
        if start_date:
            lower_bound = parse_date(start_date, is_end=False)
            return {item: lower_bound for item in explicit_forms}
        grace = dt.timedelta(days=LOOKBACK_GRACE_DAYS)
        result: dict[str, dt.date] = {}
        for item in explicit_forms:
            years = LOOKBACK_YEARS_BY_FORM.get(item, 1)
            result[item] = subtract_years(end_date, years) - grace
        return result

    async def _filter_filings(
        self,
        ticker: str,
        submissions: dict[str, Any],
        form_windows: dict[str, dt.date],
        end_date: dt.date,
        target_cik: str,
        sc13_direction_cache: Optional[dict[str, Optional[bool]]] = None,
        rejection_registry: Optional[dict[str, dict[str, str]]] = None,
        overwrite: bool = False,
    ) -> tuple[list[FilingRecord], set[str]]:
        """过滤 filings 并收集 filenum。

        Args:
            ticker: 股票代码。
            submissions: submissions API 结果。
            form_windows: form 到起始日期映射。
            end_date: 结束日期。
            target_cik: 目标 CIK。
            sc13_direction_cache: SC13 方向判定缓存。
            rejection_registry: 拒绝注册表。
            overwrite: 是否覆盖模式。

        Returns:
            (过滤后的 filing 列表, filenum 集合)。

        Raises:
            ValueError: submissions 结构非法时抛出。
        """

        records: dict[str, FilingRecord] = {}
        filenums: set[str] = set()
        recent = submissions.get("filings", {}).get("recent", {})
        collect_filings_from_table(
            records=records,
            table=recent,
            form_windows=form_windows,
            end_date=end_date,
        )
        collect_filenums_from_table(filenums, recent)
        history_files = submissions.get("filings", {}).get("files", [])
        for history_file in history_files:
            if not isinstance(history_file, dict):
                continue
            filename = str(history_file.get("name", "")).strip()
            if not filename:
                continue
            # 磁盘缓存：优先读取本地 .dayu/sec_cache/submissions/ 下的缓存
            cache_key = filename.replace(".json", "")
            cached_data = await _read_sec_cache_async(
                self._workspace_root, "submissions", cache_key,
            )
            if cached_data is not None:
                history_json = cached_data
            else:
                history_url = f"https://data.sec.gov/submissions/{filename}"
                try:
                    history_json = await _maybe_await(self._downloader.fetch_json(history_url))
                except RuntimeError as exc:
                    Log.warn(f"历史 filings 文件抓取失败: {history_url} error={exc}", module=self.MODULE)
                    continue
                # 写入缓存供后续 retry / 下次运行复用
                await _write_sec_cache_async(
                    self._workspace_root, "submissions", cache_key, history_json,
                )
            collect_filings_from_table(
                records=records,
                table=history_json,
                form_windows=form_windows,
                end_date=end_date,
            )
            collect_filenums_from_table(filenums, history_json)
        sorted_records = sorted(
            records.values(),
            key=lambda item: (item.filing_date, item.form_type, item.accession_number),
        )
        # SC13 方向过滤与去重
        direction_filtered_records = cast(
            list[FilingRecord],
            await _filter_sc13_by_direction_impl(
                cast(_SecSc13WorkflowHost, self),
                ticker=ticker,
                filings=sorted_records,
                target_cik=target_cik,
                archive_cik=target_cik,
                sc13_direction_cache=sc13_direction_cache,
                rejection_registry=rejection_registry,
                overwrite=overwrite,
            ),
        )
        deduplicated_records = cast(
            list[FilingRecord],
            _keep_latest_sc13_per_filer_impl(tuple(direction_filtered_records)),
        )
        Log.debug(
            (
                "过滤后 filings 数量: "
                f"原始={len(sorted_records)} 方向过滤后={len(direction_filtered_records)} "
                f"去重后={len(deduplicated_records)}"
            ),
            module=self.MODULE,
        )
        return deduplicated_records, filenums

    # ========== download: SC13 ==========

    async def _extend_with_browse_edgar_sc13(
        self,
        ticker: str,
        filings: list[FilingRecord],
        filenums: set[str],
        form_windows: dict[str, dt.date],
        end_date: dt.date,
        target_cik: str,
        sc13_direction_cache: Optional[dict[str, Optional[bool]]] = None,
        rejection_registry: Optional[dict[str, dict[str, str]]] = None,
        overwrite: bool = False,
    ) -> list[FilingRecord]:
        """通过 browse-edgar 补齐 SC 13D/G。

        Args:
            ticker: 股票代码。
            filings: 已有 filings。
            filenums: submissions 中收集的 filenum。
            form_windows: form 到起始日期映射。
            end_date: 结束日期。
            target_cik: 目标 CIK。
            sc13_direction_cache: SC13 方向判定缓存。
            rejection_registry: 拒绝注册表。
            overwrite: 是否覆盖模式。

        Returns:
            合并后的 filings 列表。
        """

        return cast(
            list[FilingRecord],
            await _extend_with_browse_edgar_sc13_impl(
                cast(_SecSc13WorkflowHost, self),
                ticker=ticker,
                filings=filings,
                filenums=filenums,
                form_windows=form_windows,
                end_date=end_date,
                target_cik=target_cik,
                parse_date=parse_date,
                create_filing_record=lambda form_type, filing_date, report_date, accession_number, primary_document, filer_key: FilingRecord(
                    form_type=form_type,
                    filing_date=filing_date,
                    report_date=report_date,
                    accession_number=accession_number,
                    primary_document=primary_document,
                    filer_key=filer_key,
                ),
                sc13_direction_cache=sc13_direction_cache,
                rejection_registry=rejection_registry,
                overwrite=overwrite,
            ),
        )

    async def _retry_sc13_if_empty(
        self,
        ticker: str,
        filings: list[FilingRecord],
        filenums: set[str],
        submissions: dict[str, Any],
        form_windows: dict[str, dt.date],
        end_date: dt.date,
        target_cik: str,
        sc13_direction_cache: Optional[dict[str, Optional[bool]]] = None,
        rejection_registry: Optional[dict[str, dict[str, str]]] = None,
        overwrite: bool = False,
    ) -> list[FilingRecord]:
        """SC 13 渐进式回溯：初始窗口内无结果时逐次扩大窗口重试。

        Args:
            ticker: 股票代码。
            filings: 当前已过滤的 filings 列表。
            filenums: filenum 集合。
            submissions: SEC submissions 响应。
            form_windows: form 到起始日期映射。
            end_date: 结束日期。
            target_cik: 目标 CIK。
            sc13_direction_cache: SC13 方向判定缓存。
            rejection_registry: 拒绝注册表。
            overwrite: 是否覆盖模式。

        Returns:
            可能扩充了 SC 13 filings 的最终列表。
        """

        return cast(
            list[FilingRecord],
            await _retry_sc13_if_empty_impl(
                cast(_SecSc13WorkflowHost, self),
                ticker=ticker,
                filings=filings,
                filenums=filenums,
                submissions=submissions,
                form_windows=form_windows,
                end_date=end_date,
                target_cik=target_cik,
                sc13_direction_cache=sc13_direction_cache,
                rejection_registry=rejection_registry,
                overwrite=overwrite,
            ),
        )

    async def _should_keep_sc13_direction(
        self,
        ticker: str,
        filing: FilingRecord,
        archive_cik: str,
        target_cik: str,
        sc13_direction_cache: Optional[dict[str, Optional[bool]]] = None,
        rejection_registry: Optional[dict[str, dict[str, str]]] = None,
        overwrite: bool = False,
    ) -> bool:
        """判断单条 SC13 是否满足"别人持股我"方向。

        Args:
            ticker: 股票代码。
            filing: filing 记录。
            archive_cik: 归档路径 CIK。
            target_cik: 目标 CIK。
            sc13_direction_cache: SC13 方向判定缓存。
            rejection_registry: 拒绝注册表。
            overwrite: 是否覆盖模式。

        Returns:
            满足业务方向返回 True。
        """

        return await _should_keep_sc13_direction_impl(
            cast(_SecSc13WorkflowHost, self),
            ticker=ticker,
            filing=filing,
            archive_cik=archive_cik,
            target_cik=target_cik,
            download_version=SEC_PIPELINE_DOWNLOAD_VERSION,
            sc13_direction_cache=sc13_direction_cache,
            rejection_registry=rejection_registry,
            overwrite=overwrite,
        )

    # ========== download: skip & version ==========

    def _can_skip_fast(
        self,
        previous_meta: Optional[dict[str, Any]],
        overwrite: bool,
    ) -> Optional[str]:
        """快速预检：仅依据本地 meta 判断是否可跳过。

        Args:
            previous_meta: 旧 meta 数据。
            overwrite: 用户是否要求覆盖。

        Returns:
            命中跳过时返回原因码，否则返回 None。
        """

        if overwrite or previous_meta is None:
            return None
        if not bool(previous_meta.get("ingest_complete", False)):
            return None
        if str(previous_meta.get("download_version", "")) != SEC_PIPELINE_DOWNLOAD_VERSION:
            return None
        previous_fingerprint = str(previous_meta.get("source_fingerprint", "")).strip()
        return "already_downloaded_complete" if previous_fingerprint else None

    def _can_skip(
        self,
        previous_meta: Optional[dict[str, Any]],
        source_fingerprint: str,
        overwrite: bool,
        remote_files: Optional[list[RemoteFileDescriptor]] = None,
    ) -> Optional[str]:
        """判断是否可跳过下载。

        Args:
            previous_meta: 旧 meta 数据。
            source_fingerprint: 本次远端指纹。
            overwrite: 用户是否要求覆盖。
            remote_files: 本次远端文件列表。

        Returns:
            命中跳过时返回原因码，否则返回 None。
        """

        if overwrite or previous_meta is None:
            return None
        if not bool(previous_meta.get("ingest_complete", False)):
            return None
        if str(previous_meta.get("download_version", "")) != SEC_PIPELINE_DOWNLOAD_VERSION:
            return None
        previous_fingerprint = str(previous_meta.get("source_fingerprint", "")).strip()
        if previous_fingerprint and previous_fingerprint == source_fingerprint:
            return "source_fingerprint_matched"
        if _remote_files_equivalent_to_previous_meta(
            previous_meta=previous_meta,
            remote_files=remote_files,
        ):
            return "remote_files_equivalent"
        return None

    def _resolve_document_version(
        self,
        previous_meta: Optional[dict[str, Any]],
        source_fingerprint: str,
    ) -> str:
        """计算文档版本号。

        Args:
            previous_meta: 旧 meta。
            source_fingerprint: 新指纹。

        Returns:
            文档版本号。
        """

        return _resolve_document_version_impl(
            previous_meta,
            source_fingerprint,
            increment_document_version=increment_document_version,
        )

    # ========== download: safe meta access ==========

    def _safe_get_company_meta(self, ticker: str) -> Optional[CompanyMeta]:
        """安全读取公司元数据。

        Args:
            ticker: 股票代码。

        Returns:
            公司元数据；不存在时返回 None。
        """

        return _safe_get_company_meta_impl(self._company_repository, ticker=ticker)

    def _safe_get_document_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> Optional[dict[str, Any]]:
        """安全读取文档 meta。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 源文档类型。

        Returns:
            文档 meta；不存在时返回 None。
        """

        return _safe_get_document_meta_impl(
            self._source_repository,
            ticker=ticker,
            document_id=document_id,
            source_kind=source_kind,
        )

    def _safe_get_filing_source_meta(
        self, ticker: str, document_id: str
    ) -> Optional[dict[str, Any]]:
        """安全读取 filing source meta。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。

        Returns:
            filing meta 字典；不存在时返回 None。
        """

        return _safe_get_filing_source_meta_impl(
            self._source_repository,
            ticker=ticker,
            document_id=document_id,
        )

    def _safe_get_processed_meta(self, ticker: str, document_id: str) -> Optional[dict[str, Any]]:
        """安全读取 processed meta。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。

        Returns:
            processed meta；不存在时返回 None。
        """

        return _safe_get_processed_meta_impl(
            self._processed_repository,
            ticker=ticker,
            document_id=document_id,
        )

    # ========== download: persistence ==========

    def _build_file_entries(
        self,
        file_results: list[dict[str, Any]],
        previous_files: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """构建 meta.json 的 files 列表。

        Args:
            file_results: 下载结果列表。
            previous_files: 旧文件条目映射。

        Returns:
            文件条目列表。
        """

        return _build_file_entries_impl(
            file_results=file_results,
            previous_files=previous_files,
        )

    def _build_store_file(self, source_handle: SourceHandle) -> Callable[[str, BinaryIO], Any]:
        """构建 store_file 回调。

        Args:
            source_handle: 源文档句柄。

        Returns:
            store_file 回调。
        """

        return _build_store_file_impl(self._blob_repository, source_handle)

    def _build_rejected_store_file(
        self,
        *,
        ticker: str,
        document_id: str,
    ) -> Callable[[str, BinaryIO], FileObjectMeta]:
        """构建 rejected filing 的 store_file 回调。

        Args:
            ticker: 股票代码。
            document_id: rejected filing 文档 ID。

        Returns:
            写入 rejected artifact 文件的回调。
        """

        return _build_rejected_store_file_impl(
            self._filing_maintenance_repository,
            ticker=ticker,
            document_id=document_id,
        )

    async def _persist_rejected_filing_artifact(
        self,
        *,
        ticker: str,
        cik: str,
        filing: FilingRecord,
        remote_files: list[RemoteFileDescriptor],
        overwrite: bool,
        rejection_reason: str,
        rejection_category: str,
        selected_primary_document: str,
        source_fingerprint: str,
    ) -> tuple[bool, Optional[str]]:
        """下载并保存 rejected filing artifact。

        Args:
            ticker: 股票代码。
            cik: 公司 CIK。
            filing: filing 记录。
            remote_files: 远端文件列表。
            overwrite: 是否覆盖。
            rejection_reason: 拒绝原因。
            rejection_category: 拒绝分类。
            selected_primary_document: 当前规则选中的主文件。
            source_fingerprint: 远端文件指纹。

        Returns:
            ``(成功标记, 失败原因)``。
        """

        download_stream_func = getattr(self._downloader, "download_files_stream", None)
        normalized_download_stream = (
            cast(Callable[..., AsyncIterator[DownloaderEvent]], download_stream_func)
            if callable(download_stream_func)
            else None
        )
        return await _persist_rejected_filing_artifact_impl(
            ticker=ticker,
            cik=cik,
            filing=filing,
            remote_files=remote_files,
            overwrite=overwrite,
            rejection_reason=rejection_reason,
            rejection_category=rejection_category,
            selected_primary_document=selected_primary_document,
            source_fingerprint=source_fingerprint,
            classification_version=SEC_PIPELINE_DOWNLOAD_VERSION,
            filing_maintenance_repository=self._filing_maintenance_repository,
            download_files_stream=normalized_download_stream,
            download_files=self._downloader.download_files,
            build_file_result_from_downloader_event=build_file_result_from_downloader_event,
            normalize_download_file_result=normalize_download_file_result,
            summarize_failed_download_file_reasons=summarize_failed_download_file_reasons,
        )

    def _mark_processed_reprocess_required(self, ticker: str, document_id: str) -> None:
        """标记 processed 产物需要重处理。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
        """

        _mark_processed_reprocess_required_impl(
            self._processed_repository,
            ticker=ticker,
            document_id=document_id,
        )

    # ========== download: 6-K precheck ==========

    async def _precheck_6k_filter(
        self,
        remote_files: list[RemoteFileDescriptor],
        primary_document: str,
        ticker: str,
        document_id: str,
    ) -> tuple[bool, str, str]:
        """预先应用 6-K 筛选规则（未落盘前）。

        Args:
            remote_files: 远端文件描述列表。
            primary_document: 主文件名（6-K 封面）。
            ticker: 股票代码。
            document_id: 文档 ID。

        Returns:
            (是否保留, 分类标签, 选中 exhibit 名)。
        """

        has_xbrl_instance = _has_6k_xbrl_instance(remote_files)
        if has_xbrl_instance:
            try:
                selected_name = _select_6k_target_name(
                    [
                        {"name": item.name, "sec_document_type": item.sec_document_type}
                        for item in remote_files
                    ],
                    primary_document,
                )
            except ValueError:
                return True, "XBRL_AVAILABLE", primary_document
            return True, "XBRL_AVAILABLE", selected_name

        has_exhibit_candidate = _has_6k_exhibit_candidate(remote_files)
        selected_name = primary_document

        if has_exhibit_candidate:
            try:
                selected_name = _select_6k_target_name(
                    [
                        {"name": item.name, "sec_document_type": item.sec_document_type}
                        for item in remote_files
                    ],
                    primary_document,
                )
            except ValueError:
                return False, "NO_MATCH", ""

        if not primary_document and all(item.name != selected_name for item in remote_files):
            return False, "NO_MATCH", selected_name

        try:
            candidate_diagnoses = await classify_6k_remote_candidates(
                remote_files,
                primary_document,
                self._downloader,
                max_lines=120,
            )
        except RuntimeError as exc:
            Log.warn(
                f"6-K 预下载失败: ticker={ticker} document_id={document_id} error={exc}",
                module=self.MODULE,
            )
            return False, "DOWNLOAD_FAILED", selected_name

        if not candidate_diagnoses:
            return False, "NO_MATCH", selected_name

        positive_candidate = _select_best_positive_6k_candidate(candidate_diagnoses)
        if positive_candidate is not None:
            return True, positive_candidate.classification, positive_candidate.filename

        if not has_exhibit_candidate:
            return False, "NO_EX99_OR_XBRL", selected_name

        primary_diagnosis = next(
            (item for item in candidate_diagnoses if item.is_primary_document),
            None,
        )
        if primary_diagnosis is not None and primary_diagnosis.classification == "EXCLUDE_NON_QUARTERLY":
            return False, "EXCLUDE_NON_QUARTERLY", selected_name

        if any(item.classification == "EXCLUDE_NON_QUARTERLY" for item in candidate_diagnoses):
            return False, "EXCLUDE_NON_QUARTERLY", selected_name

        return False, "NO_MATCH", selected_name

    # ========== download: company meta ==========

    def _upsert_company_meta(
        self,
        ticker: str,
        company_id: str,
        company_name: str,
        ticker_aliases: Optional[list[str]] = None,
    ) -> None:
        """写入公司级元数据。

        Args:
            ticker: 股票代码。
            company_id: 公司 ID（CIK）。
            company_name: 公司名称。
            ticker_aliases: 可选 ticker alias 列表。
        """

        _upsert_company_meta_impl(
            repository=self._company_repository,
            ticker=ticker,
            company_id=company_id,
            company_name=company_name,
            ticker_aliases=ticker_aliases,
        )

    async def _load_upload_company_ticker_aliases(
        self,
        *,
        ticker: str,
        company_id: Optional[str],
    ) -> Optional[list[str]]:
        """在 upload 初始化时补充 SEC ticker alias。

        Args:
            ticker: 当前工作区规范 ticker。
            company_id: 用户传入的公司 ID（CIK）。

        Returns:
            解析成功时返回 alias 列表；否则返回 None。
        """

        try:
            self._company_repository.get_company_meta(ticker)
            return None
        except FileNotFoundError:
            pass
        normalized_company_id = str(company_id or "").strip()
        if not normalized_company_id.isdigit():
            return None
        cik10 = normalized_company_id.zfill(10)
        try:
            submissions = await _maybe_await(self._downloader.fetch_submissions(cik10))
        except Exception as exc:
            Log.warn(
                f"upload 初始化公司元数据时获取 SEC ticker alias 失败，"
                f"ticker={ticker} company_id={normalized_company_id} error={exc}",
                module=self.MODULE,
            )
            return None
        return extract_sec_ticker_aliases(
            submissions=submissions,
            primary_ticker=ticker,
        )

    # ========== upload ==========

    def upload_filing(
        self,
        ticker: str,
        action: str,
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
            action: 动作类型。
            files: 上传文件列表。
            fiscal_year: 财年。
            fiscal_period: 财季或年度标识。
            amended: 是否修订版。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID。
            company_name: 公司名称。
            ticker_aliases: 可选 ticker alias 列表。
            overwrite: 是否强制覆盖。

        Returns:
            上传结果字典。

        Raises:
            RuntimeError: 当前线程存在运行中的事件循环时抛出。
        """

        return _run_async_upload_sync(
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
        action: str,
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
            action: 动作类型。
            files: 上传文件列表。
            fiscal_year: 财年。
            fiscal_period: 财季或年度标识。
            amended: 是否修订版。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID。
            company_name: 公司名称。
            ticker_aliases: 可选 ticker alias 列表。
            overwrite: 是否强制覆盖。

        Yields:
            上传流程事件。

        Raises:
            RuntimeError: 上传执行失败时抛出。
        """

        async for event in _run_upload_filing_stream(
            self,
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
        ):
            yield event

    def upload_material(
        self,
        ticker: str,
        action: str,
        form_type: str,
        material_name: str,
        files: Optional[list[Path]] = None,
        document_id: Optional[str] = None,
        internal_document_id: Optional[str] = None,
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
            action: 动作类型。
            form_type: 材料类型。
            material_name: 材料名称。
            files: 可选上传文件列表。
            document_id: 可选文档 ID。
            internal_document_id: 可选内部文档 ID。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID。
            company_name: 公司名称。
            ticker_aliases: 可选 ticker alias 列表。
            overwrite: 是否强制覆盖。

        Returns:
            上传结果字典。

        Raises:
            RuntimeError: 当前线程存在运行中的事件循环时抛出。
        """

        return _run_async_upload_sync(
            _collect_upload_result_from_events(
                self.upload_material_stream(
                    ticker=ticker,
                    action=action,
                    form_type=form_type,
                    material_name=material_name,
                    files=files,
                    document_id=document_id,
                    internal_document_id=internal_document_id,
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
        action: str,
        form_type: str,
        material_name: str,
        files: Optional[list[Path]] = None,
        document_id: Optional[str] = None,
        internal_document_id: Optional[str] = None,
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
            action: 动作类型。
            form_type: 材料类型。
            material_name: 材料名称。
            files: 可选上传文件列表。
            document_id: 可选文档 ID。
            internal_document_id: 可选内部文档 ID。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID。
            company_name: 公司名称。
            ticker_aliases: 可选 ticker alias 列表。
            overwrite: 是否强制覆盖。

        Yields:
            上传流程事件。

        Raises:
            RuntimeError: 上传执行失败时抛出。
        """

        async for event in _run_upload_material_stream(
            self,
            ticker=ticker,
            action=action,
            form_type=form_type,
            material_name=material_name,
            files=files,
            document_id=document_id,
            internal_document_id=internal_document_id,
            filing_date=filing_date,
            report_date=report_date,
            company_id=company_id,
            company_name=company_name,
            ticker_aliases=ticker_aliases,
            overwrite=overwrite,
        ):
            yield event

    # ========== process ==========

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
            ci: 是否追加导出 CI 快照。
            document_ids: 可选文档 ID 列表。

        Returns:
            处理结果字典。
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
    ) -> AsyncIterator[ProcessEvent]:
        """执行流式全量离线快照导出（共享服务包装器）。

        Args:
            ticker: 股票代码。
            overwrite: 是否强制覆盖。
            ci: 是否追加导出 CI 快照。
            document_ids: 可选文档 ID 列表。

        Yields:
            预处理事件流。
        """

        async for event in self._ingestion_service.process_stream(
            ticker=ticker,
            overwrite=overwrite,
            ci=ci,
            document_ids=document_ids,
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
            ci: 是否追加导出 CI 快照。
            document_ids: 可选文档 ID 列表。
            cancel_checker: 可选取消检查函数。

        Yields:
            预处理事件流。
        """

        async for event in _run_process_stream_impl(
            self,
            ticker=ticker,
            overwrite=overwrite,
            ci=ci,
            document_ids=document_ids,
            cancel_checker=cancel_checker,
        ):
            yield event

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
            ci: 是否追加导出 CI 快照。
            cancel_checker: 可选取消检查函数。

        Returns:
            结果字典。
        """

        return _run_process_single_document(
            self,
            ticker=ticker,
            document_id=document_id,
            overwrite=overwrite,
            ci=ci,
            source_kind=SourceKind.FILING,
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
            ci: 是否追加导出 CI 快照。
            cancel_checker: 可选取消检查函数。

        Returns:
            结果字典。
        """

        return _run_process_single_document(
            self,
            ticker=ticker,
            document_id=document_id,
            overwrite=overwrite,
            ci=ci,
            source_kind=SourceKind.MATERIAL,
            action="process_material",
            cancel_checker=cancel_checker,
        )

    # ========== process: snapshot ==========

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
            ci: 是否追加导出 CI 快照。
            expected_parser_signature: 预期解析器签名。
            cancel_checker: 可选取消检查函数。
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
            market_override="US",
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
        """判断当前文档是否可跳过快照导出。

        Args:
            source_meta: 源文档元数据。
            snapshot_meta: 快照元数据。
            overwrite: 是否强制覆盖。
            expected_parser_signature: 预期解析器签名。
            ci: 是否 CI 模式。
            ticker: 股票代码。
            document_id: 文档 ID。

        Returns:
            可跳过返回 True。
        """

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
        return self._match_snapshot_files(ticker=ticker, document_id=document_id, ci=ci)

    def _match_snapshot_files(self, *, ticker: str, document_id: str, ci: bool) -> bool:
        """校验快照目录文件集合是否与当前模式严格一致。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            ci: 是否 CI 模式。

        Returns:
            文件集合一致返回 True。
        """

        return _match_snapshot_files_common(
            repository=self._blob_repository,
            ticker=ticker,
            document_id=document_id,
            ci=ci,
        )

    def _cleanup_processed_snapshot_dir(
        self, *, ticker: str, document_id: str, allowed_files: set[str]
    ) -> None:
        """清理 processed/{document_id} 中非目标快照文件。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            allowed_files: 允许保留的文件名集合。
        """

        _cleanup_processed_snapshot_dir_common(
            repository=self._blob_repository,
            ticker=ticker,
            document_id=document_id,
            allowed_files=allowed_files,
        )

    def _safe_read_snapshot_meta(
        self, *, ticker: str, document_id: str
    ) -> Optional[dict[str, Any]]:
        """安全读取 tool_snapshot_meta.json。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。

        Returns:
            快照 meta；不存在时返回 None。
        """

        return _safe_read_snapshot_meta_common(
            repository=self._blob_repository,
            ticker=ticker,
            document_id=document_id,
        )

    # ========== common ==========

    def _build_result(self, action: str, **payload: Any) -> dict[str, Any]:
        """构建统一结果。

        Args:
            action: 动作名称。
            **payload: 动作负载。

        Returns:
            结果字典。
        """

        return {
            "pipeline": self.PIPELINE_NAME,
            "action": action,
            "status": payload.pop("status", "ok"),
            **payload,
        }


# ---------- 仅供 DI 传递的薄封装 ----------


def _cleanup_stale_filing_dirs(
    repository: FilingMaintenanceRepositoryProtocol,
    ticker: str,
    form_windows: dict[str, Any],
    filing_results: list[dict[str, Any]],
) -> int:
    """删除 filings 目录中多余的文档目录。

    Args:
        repository: filing 维护治理仓储。
        ticker: 股票代码。
        form_windows: form_type → 开始日期 的下载窗口字典。
        filing_results: 本次 download_stream 产生的 filing 结果列表。

    Returns:
        被清理的目录数量。
    """

    valid_doc_ids: set[str] = {
        r["document_id"]
        for r in filing_results
        if r.get("status") in {"downloaded", "skipped"}
    }
    return repository.cleanup_stale_filing_documents(
        ticker,
        active_form_types=set(form_windows.keys()),
        valid_document_ids=valid_doc_ids,
    )
