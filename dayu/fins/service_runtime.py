"""Fins 运行时适配器。"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional, Protocol, runtime_checkable
from threading import Lock

from dayu.log import Log
from dayu.fins._converters import int_or_zero, optional_int
from dayu.contracts.fins import (
    DownloadCommandPayload,
    DownloadCompanyInfo,
    DownloadFailedFile,
    DownloadFilingResultItem,
    DownloadFilingResultStatus,
    DownloadFilterWindow,
    DownloadFilters,
    DownloadProgressPayload,
    DownloadResultData,
    DownloadSummary,
    FinsCommand,
    FinsCommandName,
    FinsCommandPayload,
    FinsEvent,
    FinsEventType,
    FinsProgressEventName,
    FinsResult,
    FinsResultData,
    ProcessCommandPayload,
    ProcessDocumentResultItem,
    ProcessFilingCommandPayload,
    ProcessMaterialCommandPayload,
    ProcessProgressPayload,
    ProcessResultData,
    ProcessSingleResultData,
    ProcessSummary,
    UploadFileResultItem,
    UploadFilingCommandPayload,
    UploadFilingProgressPayload,
    UploadFilingResultData,
    UploadFilingsFromCommandPayload,
    UploadFilingsFromMaterialItem,
    UploadFilingsFromRecognizedItem,
    UploadFilingsFromResultData,
    UploadFilingsFromSkippedItem,
    UploadMaterialCommandPayload,
    UploadMaterialProgressPayload,
    UploadMaterialResultData,
)
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.fins.cli_support import (
    _coerce_document_ids_input as coerce_document_ids_input,
    _coerce_forms_input as coerce_forms_input,
    _generate_upload_filings_script as generate_upload_filings_script,
    _prepare_cli_args as prepare_cli_args,
    _validate_upload_filing_args as validate_upload_filing_args,
    _validate_upload_material_args as validate_upload_material_args,
)
from dayu.fins.ingestion.process_events import ProcessEvent
from dayu.fins.domain.document_models import CompanyMeta, FilingSummary
from dayu.fins.domain.enums import SourceKind
from dayu.fins.ingestion.factory import (
    IngestionServiceFactory,
    build_ingestion_manager_key,
    build_ingestion_service_factory,
)
from dayu.fins.pipelines import PipelineProtocol, get_pipeline_from_normalized_ticker
from dayu.fins.pipelines.download_events import DownloadEvent
from dayu.fins.pipelines.upload_filing_events import UploadFilingEvent
from dayu.fins.pipelines.upload_material_events import UploadMaterialEvent
from dayu.fins.processors.registry import build_fins_processor_registry
from dayu.fins.ticker_normalization import normalize_ticker
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
from dayu.fins.tools.service import FinsToolService

_LOG_MODULE = "FINS.RUNTIME"


@runtime_checkable
class CompanyMetaProviderProtocol(Protocol):
    """提供公司名称与公司元信息摘要的窄协议。"""

    def get_company_name(self, ticker: str) -> str:
        """返回公司名称。"""

        ...

    def get_company_meta_summary(self, ticker: str) -> dict[str, str]:
        """返回公司基础 meta 摘要。"""

        ...


@runtime_checkable
class FinsRuntimeProtocol(CompanyMetaProviderProtocol, Protocol):
    """Fins 运行时协议。"""

    def validate_command(self, command: FinsCommand) -> None:
        """在执行前同步校验命令是否可被受理。"""

        ...

    def execute(
        self,
        command: FinsCommand,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> FinsResult | AsyncIterator[FinsEvent]:
        """执行财报命令。"""

        ...

    def get_processor_registry(self) -> ProcessorRegistry:
        """返回处理器注册表。"""

        ...

    def get_tool_service(self, *, processor_cache_max_entries: int = 128) -> FinsToolService:
        """返回共享的 FinsToolService 实例。

        首次调用时按指定参数创建并缓存，后续调用返回已有实例。

        Args:
            processor_cache_max_entries: Processor 缓存最大条目数（仅首次创建时生效）。

        Returns:
            共享的 FinsToolService 实例。
        """

        ...

    def build_ingestion_service_factory(self) -> IngestionServiceFactory:
        """构建按 ticker 路由的长事务服务工厂。"""

        ...

    def get_ingestion_manager_key(self) -> str:
        """返回长事务 job 管理器 key。"""

        ...

    def list_source_filings(self, ticker: str) -> list[FilingSummary]:
        """列出指定股票的已下载财报源文件摘要。

        Args:
            ticker: 股票代码。

        Returns:
            财报源文件摘要列表。
        """

        ...

def _coerce_forms_input(value: Any) -> Optional[str]:
    """标准化 `forms` 参数。"""

    if value is None:
        return None
    return coerce_forms_input(value)


def _coerce_document_ids_input(value: Any) -> Optional[list[str]]:
    """标准化 `document_ids` 参数。"""

    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return None
    return coerce_document_ids_input(value)


def _build_pipeline(
    *,
    ticker: str,
    workspace_root: Path,
    company_repository: CompanyMetaRepositoryProtocol,
    source_repository: SourceDocumentRepositoryProtocol,
    processed_repository: ProcessedDocumentRepositoryProtocol,
    blob_repository: DocumentBlobRepositoryProtocol,
    filing_maintenance_repository: FilingMaintenanceRepositoryProtocol,
    processor_registry: ProcessorRegistry,
) -> PipelineProtocol:
    """按 ticker 构建 pipeline。"""

    normalized_ticker = normalize_ticker(ticker)
    return get_pipeline_from_normalized_ticker(
        normalized_ticker=normalized_ticker,
        workspace_root=workspace_root,
        company_repository=company_repository,
        source_repository=source_repository,
        processed_repository=processed_repository,
        blob_repository=blob_repository,
        filing_maintenance_repository=filing_maintenance_repository,
        processor_registry=processor_registry,
    )


@dataclass(frozen=True)
class _PreparedDownloadArgs:
    """下载命令的运行时准备参数。"""

    ticker: str
    form_type: str | list[str] | None
    start_date: str | None
    end_date: str | None
    overwrite: bool
    rebuild: bool
    ticker_aliases: list[str] | None


@dataclass(frozen=True)
class _PreparedProcessArgs:
    """处理命令的运行时准备参数。"""

    ticker: str
    document_ids: list[str] | None
    overwrite: bool
    ci: bool


@dataclass(frozen=True)
class _PreparedUploadFilingArgs:
    """上传财报命令的运行时准备参数。"""

    ticker: str
    action: str | None
    fiscal_year: int
    fiscal_period: str
    amended: bool
    filing_date: str | None
    report_date: str | None
    company_id: str | None
    company_name: str | None
    ticker_aliases: list[str] | None
    overwrite: bool


@dataclass(frozen=True)
class _PreparedUploadMaterialArgs:
    """上传材料命令的运行时准备参数。"""

    ticker: str
    action: str | None
    form_type: str
    material_name: str
    document_id: str | None
    internal_document_id: str | None
    fiscal_year: int | None
    fiscal_period: str | None
    filing_date: str | None
    report_date: str | None
    company_id: str | None
    company_name: str | None
    ticker_aliases: list[str] | None
    overwrite: bool


@dataclass(frozen=True)
class _PreparedUploadFilingsFromArgs:
    """批量上传脚本生成命令的运行时准备参数。"""

    ticker: str
    original_ticker: str
    base: str
    source_dir: str
    action: str | None
    output_script: str | None
    recursive: bool
    amended: bool
    filing_date: str | None
    report_date: str | None
    company_id: str | None
    company_name: str | None
    original_company_name: str | None
    infer: bool
    overwrite: bool
    material_forms: str | None
    verbose: bool
    debug: bool
    info: bool
    quiet: bool
    log_level: str | None
    ticker_aliases: list[str] | None
    generated_ticker_csv: str


def _optional_string_list(value: object) -> list[str] | None:
    """把 Namespace 中的可选字符串序列收窄为 `list[str] | None`。

    Args:
        value: 原始属性值。

    Returns:
        规范化后的字符串列表；空值时返回 `None`。

    Raises:
        无。
    """

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return normalized or None
    normalized_value = str(value).strip()
    return [normalized_value] if normalized_value else None


def _build_upload_filing_args(
    payload: UploadFilingCommandPayload,
    workspace_root: Path,
) -> _PreparedUploadFilingArgs:
    """为上传财报命令构建强类型准备参数。"""

    namespace = _build_upload_filing_namespace(payload, workspace_root)
    validate_upload_filing_args(namespace)
    return _PreparedUploadFilingArgs(
        ticker=str(namespace.ticker),
        action=str(namespace.action).strip() or None if namespace.action is not None else None,
        fiscal_year=int(namespace.fiscal_year),
        fiscal_period=str(namespace.fiscal_period),
        amended=bool(namespace.amended),
        filing_date=str(namespace.filing_date).strip() or None if namespace.filing_date is not None else None,
        report_date=str(namespace.report_date).strip() or None if namespace.report_date is not None else None,
        company_id=str(namespace.company_id).strip() or None if namespace.company_id is not None else None,
        company_name=str(namespace.company_name).strip() or None if namespace.company_name is not None else None,
        ticker_aliases=_optional_string_list(getattr(namespace, "ticker_aliases", None)),
        overwrite=bool(namespace.overwrite),
    )


def _build_download_args(
    payload: DownloadCommandPayload,
    workspace_root: Path,
) -> _PreparedDownloadArgs:
    """为下载命令构建强类型准备参数。"""

    namespace = _build_download_namespace(payload, workspace_root)
    return _PreparedDownloadArgs(
        ticker=str(namespace.ticker),
        form_type=namespace.form_type,
        start_date=str(namespace.start_date).strip() or None if namespace.start_date is not None else None,
        end_date=str(namespace.end_date).strip() or None if namespace.end_date is not None else None,
        overwrite=bool(namespace.overwrite),
        rebuild=bool(namespace.rebuild),
        ticker_aliases=_optional_string_list(getattr(namespace, "ticker_aliases", None)),
    )


def _build_process_args(
    payload: ProcessCommandPayload,
    workspace_root: Path,
) -> _PreparedProcessArgs:
    """为处理命令构建强类型准备参数。"""

    namespace = _build_process_namespace(payload, workspace_root)
    return _PreparedProcessArgs(
        ticker=str(namespace.ticker),
        document_ids=_optional_string_list(namespace.document_ids),
        overwrite=bool(namespace.overwrite),
        ci=bool(namespace.ci),
    )


def _build_upload_material_args(
    payload: UploadMaterialCommandPayload,
    workspace_root: Path,
) -> _PreparedUploadMaterialArgs:
    """为上传材料命令构建强类型准备参数。"""

    namespace = _build_upload_material_namespace(payload, workspace_root)
    validate_upload_material_args(namespace)
    return _PreparedUploadMaterialArgs(
        ticker=str(namespace.ticker),
        action=str(namespace.action).strip() or None if namespace.action is not None else None,
        form_type=str(namespace.form_type),
        material_name=str(namespace.material_name),
        document_id=str(namespace.document_id).strip() or None if namespace.document_id is not None else None,
        internal_document_id=(
            str(namespace.internal_document_id).strip() or None
            if namespace.internal_document_id is not None
            else None
        ),
        fiscal_year=int(namespace.fiscal_year) if namespace.fiscal_year is not None else None,
        fiscal_period=str(namespace.fiscal_period).strip() or None if namespace.fiscal_period is not None else None,
        filing_date=str(namespace.filing_date).strip() or None if namespace.filing_date is not None else None,
        report_date=str(namespace.report_date).strip() or None if namespace.report_date is not None else None,
        company_id=str(namespace.company_id).strip() or None if namespace.company_id is not None else None,
        company_name=str(namespace.company_name).strip() or None if namespace.company_name is not None else None,
        ticker_aliases=_optional_string_list(getattr(namespace, "ticker_aliases", None)),
        overwrite=bool(namespace.overwrite),
    )


def _build_upload_filings_from_args(
    payload: UploadFilingsFromCommandPayload,
    workspace_root: Path,
) -> _PreparedUploadFilingsFromArgs:
    """为批量上传脚本命令构建强类型准备参数。"""

    namespace = _build_upload_filings_from_namespace(payload, workspace_root)
    generated_ticker_csv = str(getattr(namespace, "generated_ticker_csv", "")).strip()
    if not generated_ticker_csv:
        raise ValueError("upload_filings_from 缺少 generated_ticker_csv")
    return _PreparedUploadFilingsFromArgs(
        ticker=str(namespace.ticker),
        original_ticker=str(getattr(namespace, "original_ticker", namespace.ticker)),
        base=str(namespace.base),
        source_dir=str(namespace.source_dir),
        action=str(namespace.action).strip() or None if namespace.action is not None else None,
        output_script=str(namespace.output_script).strip() or None if namespace.output_script is not None else None,
        recursive=bool(namespace.recursive),
        amended=bool(namespace.amended),
        filing_date=str(namespace.filing_date).strip() or None if namespace.filing_date is not None else None,
        report_date=str(namespace.report_date).strip() or None if namespace.report_date is not None else None,
        company_id=str(namespace.company_id).strip() or None if namespace.company_id is not None else None,
        company_name=str(namespace.company_name).strip() or None if namespace.company_name is not None else None,
        original_company_name=(
            str(getattr(namespace, "original_company_name", None)).strip() or None
            if getattr(namespace, "original_company_name", None) is not None
            else None
        ),
        infer=bool(namespace.infer),
        overwrite=bool(namespace.overwrite),
        material_forms=str(namespace.material_forms).strip() or None if namespace.material_forms is not None else None,
        verbose=bool(namespace.verbose),
        debug=bool(namespace.debug),
        info=bool(namespace.info),
        quiet=bool(namespace.quiet),
        log_level=str(namespace.log_level).strip() or None if namespace.log_level is not None else None,
        ticker_aliases=_optional_string_list(getattr(namespace, "ticker_aliases", None)),
        generated_ticker_csv=generated_ticker_csv,
    )


def _load_company_meta_best_effort(
    repository: CompanyMetaRepositoryProtocol,
    *,
    ticker: str,
) -> CompanyMeta | None:
    """以 best-effort 方式读取公司元数据。

    Args:
        repository: 公司元数据仓储。
        ticker: 股票代码。

    Returns:
        公司元数据；不存在或底层读取失败时返回 `None`。

    Raises:
        无。
    """

    try:
        return repository.get_company_meta(ticker)
    except (FileNotFoundError, ValueError, OSError):
        return None


def _build_upload_filing_namespace(payload: UploadFilingCommandPayload, workspace_root: Path) -> Namespace:
    """为上传财报校验构建命名空间。"""

    namespace = Namespace(
        command="upload_filing",
        ticker=payload.ticker,
        base=str(workspace_root),
        action=payload.action,
        files=[str(path) for path in payload.files],
        fiscal_year=payload.fiscal_year,
        fiscal_period=payload.fiscal_period,
        amended=payload.amended,
        filing_date=payload.filing_date,
        report_date=payload.report_date,
        company_id=payload.company_id,
        company_name=payload.company_name,
        infer=payload.infer,
        ticker_aliases=list(payload.ticker_aliases),
        overwrite=payload.overwrite,
    )
    prepare_cli_args(namespace)
    return namespace


def _build_download_namespace(payload: DownloadCommandPayload, workspace_root: Path) -> Namespace:
    """为下载命令构建并规范化命名空间。"""

    namespace = Namespace(
        command="download",
        ticker=payload.ticker,
        base=str(workspace_root),
        form_type=list(payload.form_type) if payload.form_type else None,
        start_date=payload.start_date,
        end_date=payload.end_date,
        overwrite=payload.overwrite,
        rebuild=payload.rebuild,
        infer=payload.infer,
        ticker_aliases=list(payload.ticker_aliases),
    )
    prepare_cli_args(namespace)
    return namespace


def _build_process_namespace(payload: ProcessCommandPayload, workspace_root: Path) -> Namespace:
    """为处理命令构建并规范化命名空间。"""

    namespace = Namespace(
        command="process",
        ticker=payload.ticker,
        base=str(workspace_root),
        document_ids=list(payload.document_ids) if payload.document_ids else None,
        overwrite=payload.overwrite,
        ci=payload.ci,
    )
    prepare_cli_args(namespace)
    return namespace


def _build_upload_material_namespace(payload: UploadMaterialCommandPayload, workspace_root: Path) -> Namespace:
    """为上传材料校验构建命名空间。"""

    namespace = Namespace(
        command="upload_material",
        ticker=payload.ticker,
        base=str(workspace_root),
        action=payload.action,
        form_type=payload.form_type,
        material_name=payload.material_name,
        files=[str(path) for path in payload.files],
        document_id=payload.document_id,
        internal_document_id=payload.internal_document_id,
        fiscal_year=payload.fiscal_year,
        fiscal_period=payload.fiscal_period,
        filing_date=payload.filing_date,
        report_date=payload.report_date,
        company_id=payload.company_id,
        company_name=payload.company_name,
        infer=payload.infer,
        ticker_aliases=list(payload.ticker_aliases),
        overwrite=payload.overwrite,
    )
    prepare_cli_args(namespace)
    return namespace


def _build_upload_filings_from_namespace(
    payload: UploadFilingsFromCommandPayload,
    workspace_root: Path,
) -> Namespace:
    """为批量上传脚本生成构建命名空间。"""

    namespace = Namespace(
        command="upload_filings_from",
        ticker=payload.ticker,
        base=str(workspace_root),
        source_dir=str(payload.source_dir),
        action=payload.action,
        output_script=str(payload.output_script) if payload.output_script is not None else None,
        recursive=payload.recursive,
        amended=payload.amended,
        filing_date=payload.filing_date,
        report_date=payload.report_date,
        company_id=payload.company_id,
        company_name=payload.company_name,
        infer=payload.infer,
        overwrite=payload.overwrite,
        material_forms=list(payload.material_forms),
        verbose=payload.verbose,
        debug=payload.debug,
        info=payload.info,
        quiet=payload.quiet,
        log_level=payload.log_level,
    )
    prepare_cli_args(namespace)
    return namespace


def _build_upload_file_items(values: object) -> tuple[UploadFileResultItem, ...]:
    """把结果中的文件列表规范化为强类型条目。"""

    if not isinstance(values, list):
        return ()
    items: list[UploadFileResultItem] = []
    for value in values:
        normalized = str(value).strip()
        if normalized:
            items.append(UploadFileResultItem(path=normalized))
    return tuple(items)


def _build_download_failed_files(values: object) -> tuple[DownloadFailedFile, ...]:
    """规范化失败下载文件列表。"""

    if not isinstance(values, list):
        return ()
    items: list[DownloadFailedFile] = []
    for value in values:
        if isinstance(value, dict):
            items.append(
                DownloadFailedFile(
                    file_name=_optional_text(value.get("name")) or _optional_text(value.get("file_name")),
                    source=_optional_text(value.get("source")),
                    reason_code=_optional_text(value.get("reason_code")),
                    reason_message=(
                        _optional_text(value.get("reason_message"))
                        or _optional_text(value.get("message"))
                        or _optional_text(value.get("error"))
                    ),
                )
            )
            continue
        normalized = str(value).strip()
        if normalized:
            items.append(DownloadFailedFile(file_name=normalized))
    return tuple(items)


def _optional_text(value: object) -> str | None:
    """提取可选字符串。"""

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _require_download_payload(payload: FinsCommandPayload) -> DownloadCommandPayload:
    """收窄 download 命令载荷类型。

    Args:
        payload: 宽命令载荷。

    Returns:
        ``DownloadCommandPayload`` 强类型载荷。

    Raises:
        TypeError: 当载荷类型与 ``download`` 命令不匹配时抛出。
    """

    if not isinstance(payload, DownloadCommandPayload):
        raise TypeError("download 命令必须使用 DownloadCommandPayload")
    return payload


def _require_upload_filing_payload(payload: FinsCommandPayload) -> UploadFilingCommandPayload:
    """收窄 upload_filing 命令载荷类型。

    Args:
        payload: 宽命令载荷。

    Returns:
        ``UploadFilingCommandPayload`` 强类型载荷。

    Raises:
        TypeError: 当载荷类型与 ``upload_filing`` 命令不匹配时抛出。
    """

    if not isinstance(payload, UploadFilingCommandPayload):
        raise TypeError("upload_filing 命令必须使用 UploadFilingCommandPayload")
    return payload


def _require_upload_filings_from_payload(payload: FinsCommandPayload) -> UploadFilingsFromCommandPayload:
    """收窄 upload_filings_from 命令载荷类型。

    Args:
        payload: 宽命令载荷。

    Returns:
        ``UploadFilingsFromCommandPayload`` 强类型载荷。

    Raises:
        TypeError: 当载荷类型与 ``upload_filings_from`` 命令不匹配时抛出。
    """

    if not isinstance(payload, UploadFilingsFromCommandPayload):
        raise TypeError("upload_filings_from 命令必须使用 UploadFilingsFromCommandPayload")
    return payload


def _require_upload_material_payload(payload: FinsCommandPayload) -> UploadMaterialCommandPayload:
    """收窄 upload_material 命令载荷类型。

    Args:
        payload: 宽命令载荷。

    Returns:
        ``UploadMaterialCommandPayload`` 强类型载荷。

    Raises:
        TypeError: 当载荷类型与 ``upload_material`` 命令不匹配时抛出。
    """

    if not isinstance(payload, UploadMaterialCommandPayload):
        raise TypeError("upload_material 命令必须使用 UploadMaterialCommandPayload")
    return payload


def _require_process_payload(payload: FinsCommandPayload) -> ProcessCommandPayload:
    """收窄 process 命令载荷类型。

    Args:
        payload: 宽命令载荷。

    Returns:
        ``ProcessCommandPayload`` 强类型载荷。

    Raises:
        TypeError: 当载荷类型与 ``process`` 命令不匹配时抛出。
    """

    if not isinstance(payload, ProcessCommandPayload):
        raise TypeError("process 命令必须使用 ProcessCommandPayload")
    return payload


def _require_process_filing_payload(payload: FinsCommandPayload) -> ProcessFilingCommandPayload:
    """收窄 process_filing 命令载荷类型。

    Args:
        payload: 宽命令载荷。

    Returns:
        ``ProcessFilingCommandPayload`` 强类型载荷。

    Raises:
        TypeError: 当载荷类型与 ``process_filing`` 命令不匹配时抛出。
    """

    if not isinstance(payload, ProcessFilingCommandPayload):
        raise TypeError("process_filing 命令必须使用 ProcessFilingCommandPayload")
    return payload


def _require_process_material_payload(payload: FinsCommandPayload) -> ProcessMaterialCommandPayload:
    """收窄 process_material 命令载荷类型。

    Args:
        payload: 宽命令载荷。

    Returns:
        ``ProcessMaterialCommandPayload`` 强类型载荷。

    Raises:
        TypeError: 当载荷类型与 ``process_material`` 命令不匹配时抛出。
    """

    if not isinstance(payload, ProcessMaterialCommandPayload):
        raise TypeError("process_material 命令必须使用 ProcessMaterialCommandPayload")
    return payload


def _require_download_event(event: DownloadEvent | ProcessEvent | UploadFilingEvent | UploadMaterialEvent) -> DownloadEvent:
    """收窄 download 流事件类型。

    Args:
        event: 宽事件对象。

    Returns:
        ``DownloadEvent`` 强类型事件。

    Raises:
        TypeError: 当事件类型与 ``download`` 流不匹配时抛出。
    """

    if not isinstance(event, DownloadEvent):
        raise TypeError("download 事件流必须产出 DownloadEvent")
    return event


def _require_process_event(event: DownloadEvent | ProcessEvent | UploadFilingEvent | UploadMaterialEvent) -> ProcessEvent:
    """收窄 process 流事件类型。

    Args:
        event: 宽事件对象。

    Returns:
        ``ProcessEvent`` 强类型事件。

    Raises:
        TypeError: 当事件类型与 ``process`` 流不匹配时抛出。
    """

    if not isinstance(event, ProcessEvent):
        raise TypeError("process 事件流必须产出 ProcessEvent")
    return event


def _require_upload_filing_event(
    event: DownloadEvent | ProcessEvent | UploadFilingEvent | UploadMaterialEvent,
) -> UploadFilingEvent:
    """收窄 upload_filing 流事件类型。

    Args:
        event: 宽事件对象。

    Returns:
        ``UploadFilingEvent`` 强类型事件。

    Raises:
        TypeError: 当事件类型与 ``upload_filing`` 流不匹配时抛出。
    """

    if not isinstance(event, UploadFilingEvent):
        raise TypeError("upload_filing 事件流必须产出 UploadFilingEvent")
    return event


def _require_upload_material_event(
    event: DownloadEvent | ProcessEvent | UploadFilingEvent | UploadMaterialEvent,
) -> UploadMaterialEvent:
    """收窄 upload_material 流事件类型。

    Args:
        event: 宽事件对象。

    Returns:
        ``UploadMaterialEvent`` 强类型事件。

    Raises:
        TypeError: 当事件类型与 ``upload_material`` 流不匹配时抛出。
    """

    if not isinstance(event, UploadMaterialEvent):
        raise TypeError("upload_material 事件流必须产出 UploadMaterialEvent")
    return event


def _build_download_filing_result_item(payload: dict[str, Any]) -> DownloadFilingResultItem:
    """构建单个 filing 下载结果。"""

    return DownloadFilingResultItem(
        document_id=str(payload.get("document_id", "")).strip(),
        status=DownloadFilingResultStatus.from_raw(str(payload.get("status", ""))),
        form_type=_optional_text(payload.get("form_type")),
        filing_date=_optional_text(payload.get("filing_date")),
        report_date=_optional_text(payload.get("report_date")),
        downloaded_files=int_or_zero(payload.get("downloaded_files")),
        skipped_files=int_or_zero(payload.get("skipped_files")),
        failed_files=_build_download_failed_files(payload.get("failed_files")),
        has_xbrl=payload.get("has_xbrl") if isinstance(payload.get("has_xbrl"), bool) else None,
        reason_code=(
            _optional_text(payload.get("reason_code"))
            or _optional_text(payload.get("skip_reason"))
            or _optional_text(payload.get("reason"))
        ),
        reason_message=(
            _optional_text(payload.get("reason_message"))
            or _optional_text(payload.get("message"))
            or _optional_text(payload.get("error"))
        ),
        skip_reason=_optional_text(payload.get("skip_reason")),
        filter_category=_optional_text(payload.get("filter_category")),
    )


def _build_download_result_data(result: dict[str, Any]) -> DownloadResultData:
    """把 pipeline download 结果规范化为公共契约。"""

    company_info_payload = result.get("company_info")
    company_info = DownloadCompanyInfo()
    if isinstance(company_info_payload, dict):
        company_info = DownloadCompanyInfo(
            company_id=_optional_text(company_info_payload.get("company_id")),
            company_name=_optional_text(company_info_payload.get("company_name")),
            market=_optional_text(company_info_payload.get("market")),
        )
    filters_payload = result.get("filters")
    filters = DownloadFilters()
    if isinstance(filters_payload, dict):
        forms_raw = filters_payload.get("forms")
        start_dates_raw = filters_payload.get("start_dates")
        start_dates: list[DownloadFilterWindow] = []
        if isinstance(start_dates_raw, dict):
            for form_type, start_date in sorted(start_dates_raw.items()):
                normalized_form = str(form_type).strip()
                normalized_start = str(start_date).strip()
                if normalized_form and normalized_start:
                    start_dates.append(DownloadFilterWindow(form_type=normalized_form, start_date=normalized_start))
        filters = DownloadFilters(
            forms=tuple(str(item).strip() for item in forms_raw if str(item).strip()) if isinstance(forms_raw, list) else (),
            start_dates=tuple(start_dates),
            end_date=_optional_text(filters_payload.get("end_date")),
            overwrite=bool(filters_payload.get("overwrite", False)),
        )
    warnings_raw = result.get("warnings")
    filings_raw = result.get("filings")
    summary_raw = result.get("summary")
    filings = tuple(
        _build_download_filing_result_item(item)
        for item in filings_raw
        if isinstance(item, dict)
    ) if isinstance(filings_raw, list) else ()
    summary = DownloadSummary(0, 0, 0, 0, 0)
    if isinstance(summary_raw, dict):
        summary = DownloadSummary(
            total=int_or_zero(summary_raw.get("total")),
            downloaded=int_or_zero(summary_raw.get("downloaded")),
            skipped=int_or_zero(summary_raw.get("skipped")),
            failed=int_or_zero(summary_raw.get("failed")),
            elapsed_ms=int_or_zero(summary_raw.get("elapsed_ms")),
        )
    return DownloadResultData(
        pipeline=str(result.get("pipeline", "")).strip(),
        status=str(result.get("status", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        company_info=company_info,
        filters=filters,
        warnings=tuple(str(item).strip() for item in warnings_raw if str(item).strip()) if isinstance(warnings_raw, list) else (),
        filings=filings,
        summary=summary,
    )


def _build_upload_filing_result_data(result: dict[str, Any]) -> UploadFilingResultData:
    """把 upload_filing 结果规范化为公共契约。"""

    return UploadFilingResultData(
        pipeline=str(result.get("pipeline", "")).strip(),
        status=str(result.get("status", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        filing_action=str(result.get("filing_action", "")).strip(),
        files=_build_upload_file_items(result.get("files")),
        form_type=_optional_text(result.get("form_type")),
        fiscal_year=optional_int(result.get("fiscal_year")),
        fiscal_period=_optional_text(result.get("fiscal_period")),
        amended=result.get("amended") if isinstance(result.get("amended"), bool) else None,
        company_id=_optional_text(result.get("company_id")),
        company_name=_optional_text(result.get("company_name")),
        document_id=_optional_text(result.get("document_id")),
        primary_document=_optional_text(result.get("primary_document")),
        uploaded_files=optional_int(result.get("uploaded_files")),
        document_version=_optional_text(result.get("document_version")),
        source_fingerprint=_optional_text(result.get("source_fingerprint")),
        filing_date=_optional_text(result.get("filing_date")),
        report_date=_optional_text(result.get("report_date")),
        overwrite=result.get("overwrite") if isinstance(result.get("overwrite"), bool) else None,
        skip_reason=_optional_text(result.get("skip_reason")),
        message=_optional_text(result.get("message")),
    )


def _build_upload_material_result_data(result: dict[str, Any]) -> UploadMaterialResultData:
    """把 upload_material 结果规范化为公共契约。"""

    return UploadMaterialResultData(
        pipeline=str(result.get("pipeline", "")).strip(),
        status=str(result.get("status", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        material_action=str(result.get("material_action", "")).strip(),
        files=_build_upload_file_items(result.get("files")),
        form_type=_optional_text(result.get("form_type")),
        material_name=_optional_text(result.get("material_name")),
        fiscal_year=optional_int(result.get("fiscal_year")),
        fiscal_period=_optional_text(result.get("fiscal_period")),
        company_id=_optional_text(result.get("company_id")),
        company_name=_optional_text(result.get("company_name")),
        document_id=_optional_text(result.get("document_id")),
        internal_document_id=_optional_text(result.get("internal_document_id")),
        primary_document=_optional_text(result.get("primary_document")),
        uploaded_files=optional_int(result.get("uploaded_files")),
        document_version=_optional_text(result.get("document_version")),
        source_fingerprint=_optional_text(result.get("source_fingerprint")),
        filing_date=_optional_text(result.get("filing_date")),
        report_date=_optional_text(result.get("report_date")),
        overwrite=result.get("overwrite") if isinstance(result.get("overwrite"), bool) else None,
        skip_reason=_optional_text(result.get("skip_reason")),
        message=_optional_text(result.get("message")),
    )


def _build_upload_filings_from_result_data(result: dict[str, Any]) -> UploadFilingsFromResultData:
    """把 upload_filings_from 结果规范化为公共契约。"""

    recognized_raw = result.get("recognized")
    material_raw = result.get("material")
    skipped_raw = result.get("skipped")
    recognized = tuple(
        UploadFilingsFromRecognizedItem(
            file=str(item.get("file", "")).strip(),
            fiscal_year=optional_int(item.get("fiscal_year")),
            fiscal_period=_optional_text(item.get("fiscal_period")),
        )
        for item in recognized_raw
        if isinstance(item, dict)
    ) if isinstance(recognized_raw, list) else ()
    material = tuple(
        UploadFilingsFromMaterialItem(
            file=str(item.get("file", "")).strip(),
            material_name=_optional_text(item.get("material_name")),
        )
        for item in material_raw
        if isinstance(item, dict)
    ) if isinstance(material_raw, list) else ()
    skipped = tuple(
        UploadFilingsFromSkippedItem(
            file=str(item.get("file", "")).strip(),
            reason=_optional_text(item.get("reason")),
        )
        for item in skipped_raw
        if isinstance(item, dict)
    ) if isinstance(skipped_raw, list) else ()
    return UploadFilingsFromResultData(
        script_path=str(result.get("script_path", "")).strip(),
        script_platform=str(result.get("script_platform", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        source_dir=str(result.get("source_dir", "")).strip(),
        total_files=int_or_zero(result.get("total_files")),
        recognized_count=int_or_zero(result.get("recognized_count")),
        material_count=int_or_zero(result.get("material_count")),
        skipped_count=int_or_zero(result.get("skipped_count")),
        recognized=recognized,
        material=material,
        skipped=skipped,
    )


def _build_process_summary(payload: object) -> ProcessSummary:
    """构建 process 汇总。"""

    if not isinstance(payload, dict):
        return ProcessSummary(0, 0, 0, 0, False)
    return ProcessSummary(
        total=int_or_zero(payload.get("total")),
        processed=int_or_zero(payload.get("processed")),
        skipped=int_or_zero(payload.get("skipped")),
        failed=int_or_zero(payload.get("failed")),
        todo=bool(payload.get("todo", False)),
    )


def _build_process_document_result_item(payload: dict[str, Any]) -> ProcessDocumentResultItem:
    """构建单个 process 文档结果。"""

    return ProcessDocumentResultItem(
        document_id=str(payload.get("document_id", "")).strip(),
        status=str(payload.get("status", "")).strip(),
        reason=_optional_text(payload.get("reason")),
        form_type=_optional_text(payload.get("form_type")),
        fiscal_year=optional_int(payload.get("fiscal_year")),
        quality=_optional_text(payload.get("quality")),
        has_xbrl=payload.get("has_xbrl") if isinstance(payload.get("has_xbrl"), bool) else None,
        section_count=optional_int(payload.get("section_count")),
        table_count=optional_int(payload.get("table_count")),
        skip_reason=_optional_text(payload.get("skip_reason")),
        source_kind=_optional_text(payload.get("source_kind")),
    )


def _build_process_result_data(result: dict[str, Any]) -> ProcessResultData:
    """把 process 结果规范化为公共契约。"""

    filings_raw = result.get("filings")
    materials_raw = result.get("materials")
    filings = tuple(
        _build_process_document_result_item(item)
        for item in filings_raw
        if isinstance(item, dict)
    ) if isinstance(filings_raw, list) else ()
    materials = tuple(
        _build_process_document_result_item(item)
        for item in materials_raw
        if isinstance(item, dict)
    ) if isinstance(materials_raw, list) else ()
    return ProcessResultData(
        pipeline=str(result.get("pipeline", "")).strip(),
        status=str(result.get("status", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        overwrite=bool(result.get("overwrite", False)),
        ci=bool(result.get("ci", False)),
        filings=filings,
        filing_summary=_build_process_summary(result.get("filing_summary")),
        materials=materials,
        material_summary=_build_process_summary(result.get("material_summary")),
    )


def _build_process_single_result_data(result: dict[str, Any]) -> ProcessSingleResultData:
    """把 process_filing/process_material 结果规范化为公共契约。"""

    return ProcessSingleResultData(
        pipeline=str(result.get("pipeline", "")).strip(),
        action=str(result.get("action", "")).strip(),
        status=str(result.get("status", "")).strip(),
        ticker=str(result.get("ticker", "")).strip(),
        document_id=str(result.get("document_id", "")).strip(),
        overwrite=bool(result.get("overwrite", False)),
        ci=bool(result.get("ci", False)),
        reason=_optional_text(result.get("reason")),
        form_type=_optional_text(result.get("form_type")),
        fiscal_year=optional_int(result.get("fiscal_year")),
        quality=_optional_text(result.get("quality")),
        has_xbrl=result.get("has_xbrl") if isinstance(result.get("has_xbrl"), bool) else None,
        section_count=optional_int(result.get("section_count")),
        table_count=optional_int(result.get("table_count")),
        skip_reason=_optional_text(result.get("skip_reason")),
        message=_optional_text(result.get("message")),
    )


def _build_result_data(command_name: FinsCommandName, result: dict[str, Any]) -> FinsResultData:
    """按命令名把结果字典规范化为强类型结果。"""

    if command_name == FinsCommandName.DOWNLOAD:
        return _build_download_result_data(result)
    if command_name == FinsCommandName.UPLOAD_FILING:
        return _build_upload_filing_result_data(result)
    if command_name == FinsCommandName.UPLOAD_FILINGS_FROM:
        return _build_upload_filings_from_result_data(result)
    if command_name == FinsCommandName.UPLOAD_MATERIAL:
        return _build_upload_material_result_data(result)
    if command_name == FinsCommandName.PROCESS:
        return _build_process_result_data(result)
    return _build_process_single_result_data(result)


def _extract_result_data_from_event(
    *,
    command_name: FinsCommandName,
    event_payload: dict[str, Any],
) -> FinsResultData | None:
    """从内部 pipeline 事件中提取最终强类型结果。"""

    result = event_payload.get("result")
    if not isinstance(result, dict):
        return None
    return _build_result_data(command_name, result)


def _build_download_progress_payload(event: DownloadEvent) -> DownloadProgressPayload:
    """把下载 pipeline 事件映射为跨层 progress 负载。"""

    payload = event.payload if isinstance(event.payload, dict) else {}
    filing_result_payload = payload.get("filing_result")
    return DownloadProgressPayload(
        event_type=FinsProgressEventName(str(event.event_type)),
        ticker=event.ticker,
        document_id=event.document_id,
        action=_optional_text(payload.get("action")),
        name=_optional_text(payload.get("name")),
        form_type=_optional_text(payload.get("form_type")),
        file_count=optional_int(payload.get("file_count")),
        size=optional_int(payload.get("size")),
        message=_optional_text(payload.get("message")),
        reason=_optional_text(payload.get("reason")) or _optional_text(payload.get("skip_reason")),
        filing_result=(
            _build_download_filing_result_item(filing_result_payload)
            if isinstance(filing_result_payload, dict)
            else None
        ),
    )


def _build_process_progress_payload(event: ProcessEvent) -> ProcessProgressPayload:
    """把 process pipeline 事件映射为跨层 progress 负载。"""

    payload = event.payload if isinstance(event.payload, dict) else {}
    summary_payload = payload.get("result_summary")
    return ProcessProgressPayload(
        event_type=FinsProgressEventName(str(event.event_type)),
        ticker=event.ticker,
        document_id=event.document_id,
        source_kind=_optional_text(payload.get("source_kind")),
        total_documents=optional_int(payload.get("total_documents")),
        overwrite=payload.get("overwrite") if isinstance(payload.get("overwrite"), bool) else None,
        ci=payload.get("ci") if isinstance(payload.get("ci"), bool) else None,
        reason=_optional_text(payload.get("reason")),
        result_summary=(
            _build_process_document_result_item(summary_payload)
            if isinstance(summary_payload, dict)
            else None
        ),
    )


def _build_upload_filing_progress_payload(event: UploadFilingEvent) -> UploadFilingProgressPayload:
    """把上传财报事件映射为跨层 progress 负载。"""

    payload = event.payload if isinstance(event.payload, dict) else {}
    return UploadFilingProgressPayload(
        event_type=FinsProgressEventName(str(event.event_type)),
        ticker=event.ticker,
        document_id=event.document_id,
        action=_optional_text(payload.get("action")),
        name=_optional_text(payload.get("name")),
        file_count=optional_int(payload.get("file_count")),
        size=optional_int(payload.get("size")),
        message=_optional_text(payload.get("message")),
        error=_optional_text(payload.get("error")),
    )


def _build_upload_material_progress_payload(event: UploadMaterialEvent) -> UploadMaterialProgressPayload:
    """把上传材料事件映射为跨层 progress 负载。"""

    payload = event.payload if isinstance(event.payload, dict) else {}
    return UploadMaterialProgressPayload(
        event_type=FinsProgressEventName(str(event.event_type)),
        ticker=event.ticker,
        document_id=event.document_id,
        action=_optional_text(payload.get("action")),
        name=_optional_text(payload.get("name")),
        file_count=optional_int(payload.get("file_count")),
        size=optional_int(payload.get("size")),
        message=_optional_text(payload.get("message")),
        error=_optional_text(payload.get("error")),
    )


@dataclass
class DefaultFinsRuntime(FinsRuntimeProtocol):
    """默认 Fins 运行时实现。"""

    workspace_root: Path
    company_repository: CompanyMetaRepositoryProtocol
    source_repository: SourceDocumentRepositoryProtocol
    processed_repository: ProcessedDocumentRepositoryProtocol
    blob_repository: DocumentBlobRepositoryProtocol
    filing_maintenance_repository: FilingMaintenanceRepositoryProtocol
    processor_registry: ProcessorRegistry
    _tool_service: Optional[FinsToolService] = field(init=False, default=None, repr=False)
    _tool_service_lock: Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """初始化内部状态。"""

        # 复杂逻辑说明：_tool_service 使用懒创建 + 锁保证线程安全，
        # 所有 scene agent 共享同一个 FinsToolService 实例及其 processor cache。
        self._tool_service_lock = Lock()

    def _build_pipeline_for_ticker(self, ticker: str):
        """按 ticker 构建 direct operation 所需 pipeline。"""

        return _build_pipeline(
            ticker=ticker,
            workspace_root=self.workspace_root,
            company_repository=self.company_repository,
            source_repository=self.source_repository,
            processed_repository=self.processed_repository,
            blob_repository=self.blob_repository,
            filing_maintenance_repository=self.filing_maintenance_repository,
            processor_registry=self.processor_registry,
        )

    @classmethod
    def create(cls, *, workspace_root: Path) -> "DefaultFinsRuntime":
        """创建默认 Fins 运行时。"""

        repository_set = build_fs_repository_set(workspace_root=workspace_root)
        return cls(
            workspace_root=workspace_root,
            company_repository=FsCompanyMetaRepository(
                workspace_root,
                repository_set=repository_set,
            ),
            source_repository=FsSourceDocumentRepository(
                workspace_root,
                repository_set=repository_set,
            ),
            processed_repository=FsProcessedDocumentRepository(
                workspace_root,
                repository_set=repository_set,
            ),
            blob_repository=FsDocumentBlobRepository(
                workspace_root,
                repository_set=repository_set,
            ),
            filing_maintenance_repository=FsFilingMaintenanceRepository(
                workspace_root,
                repository_set=repository_set,
            ),
            processor_registry=build_fins_processor_registry(),
        )

    def get_processor_registry(self) -> ProcessorRegistry:
        """返回处理器注册表。"""

        return self.processor_registry

    def get_tool_service(self, *, processor_cache_max_entries: int = 128) -> FinsToolService:
        """返回共享的 FinsToolService 实例。

        首次调用时按指定参数创建并缓存，后续调用返回已有实例。
        同一进程内所有 scene agent 共享同一个实例及其 processor cache。

        Args:
            processor_cache_max_entries: Processor 缓存最大条目数（仅首次创建时生效）。

        Returns:
            共享的 FinsToolService 实例。

        Raises:
            无。
        """

        if self._tool_service is not None:
            return self._tool_service
        with self._tool_service_lock:
            if self._tool_service is not None:
                return self._tool_service
            service = FinsToolService(
                company_repository=self.company_repository,
                source_repository=self.source_repository,
                processed_repository=self.processed_repository,
                processor_registry=self.processor_registry,
                processor_cache_max_entries=processor_cache_max_entries,
            )
            self._tool_service = service
            return service

    def build_ingestion_service_factory(self) -> IngestionServiceFactory:
        """构建按 ticker 路由的长事务服务工厂。"""

        return build_ingestion_service_factory(
            workspace_root=self.workspace_root,
            company_repository=self.company_repository,
            source_repository=self.source_repository,
            processed_repository=self.processed_repository,
            blob_repository=self.blob_repository,
            filing_maintenance_repository=self.filing_maintenance_repository,
            processor_registry=self.processor_registry,
        )

    def get_ingestion_manager_key(self) -> str:
        """返回长事务 job 管理器 key。"""

        return build_ingestion_manager_key(workspace_root=self.workspace_root)

    def list_source_filings(self, ticker: str) -> list[FilingSummary]:
        """列出指定股票的已下载财报源文件摘要。

        Args:
            ticker: 股票代码。

        Returns:
            财报源文件摘要列表。

        Raises:
            OSError: 列举源文档 ID 或读取单文档数据时底层存储异常。
            ValueError: 存储数据损坏或格式不合法时抛出。
        """

        result: list[FilingSummary] = []
        try:
            document_ids = self.source_repository.list_source_document_ids(ticker, SourceKind.FILING)
        except (OSError, ValueError):
            Log.error(
                f"列出财报源文档 ID 失败: ticker={ticker}",
                exc_info=True,
                module=_LOG_MODULE,
            )
            raise

        skipped_document_count = 0
        for doc_id in document_ids:
            try:
                meta = self.source_repository.get_source_meta(ticker, doc_id, SourceKind.FILING)
                fiscal_year_raw = meta.get("fiscal_year")
                primary_file_name: Optional[str] = None
                primary_file_path: Optional[str] = None
                try:
                    primary_source = self.source_repository.get_primary_source(
                        ticker, doc_id, SourceKind.FILING
                    )
                    materialized_path = primary_source.materialize().resolve()
                    primary_file_name = materialized_path.name or None
                    primary_file_path = str(materialized_path)
                except (OSError, ValueError):
                    Log.warning(
                        f"读取财报主文件失败，降级返回元信息: ticker={ticker}, document_id={doc_id}",
                        module=_LOG_MODULE,
                    )

                result.append(FilingSummary(
                    document_id=doc_id,
                    form_type=meta.get("form_type"),
                    filing_date=meta.get("filing_date"),
                    report_date=meta.get("report_date"),
                    fiscal_year=fiscal_year_raw if isinstance(fiscal_year_raw, int) else None,
                    fiscal_period=meta.get("fiscal_period"),
                    is_deleted=bool(meta.get("is_deleted", False)),
                    primary_file_name=primary_file_name,
                    primary_file_path=primary_file_path,
                ))
            except (OSError, ValueError):
                skipped_document_count += 1
                Log.warning(
                    f"读取单条财报元信息失败，已跳过: ticker={ticker}, document_id={doc_id}",
                    module=_LOG_MODULE,
                )
                continue

        if skipped_document_count > 0:
            Log.warning(
                "财报列表存在降级结果: "
                f"ticker={ticker}, total={len(document_ids)}, skipped={skipped_document_count}",
                module=_LOG_MODULE,
            )
        result.sort(key=lambda x: x.filing_date or "", reverse=True)
        return result

    def get_company_name(self, ticker: str) -> str:
        """返回公司名称。"""

        meta = _load_company_meta_best_effort(self.company_repository, ticker=ticker)
        if meta is None:
            return ""
        return str(getattr(meta, "company_name", "")).strip()

    def get_company_meta_summary(self, ticker: str) -> dict[str, str]:
        """返回公司基础 meta 摘要。

        Args:
            ticker: 股票代码。

        Returns:
            仅包含当前写作链路需要的基础 meta 字段。

        Raises:
            无。
        """

        meta = _load_company_meta_best_effort(self.company_repository, ticker=ticker)
        if meta is None:
            return {"ticker": ticker}
        return {
            "ticker": str(getattr(meta, "ticker", "") or ticker).strip(),
            "company_name": str(getattr(meta, "company_name", "") or "").strip(),
            "market": str(getattr(meta, "market", "") or "").strip(),
            "company_id": str(getattr(meta, "company_id", "") or "").strip(),
        }

    def validate_command(self, command: FinsCommand) -> None:
        """在创建 Host run 前同步校验命令是否可被受理。

        对每个命令执行参数规范化和业务规则校验，确保请求级错误在返回
        执行句柄前被同步拒绝。

        Args:
            command: 待校验的财报命令。

        Returns:
            无。

        Raises:
            ValueError: 命令参数非法或不支持流式执行时抛出。
        """

        # 流式兼容性检查：仅 download/process/upload_filing/upload_material 支持流式
        if command.stream and command.name not in {
            FinsCommandName.DOWNLOAD,
            FinsCommandName.PROCESS,
            FinsCommandName.UPLOAD_FILING,
            FinsCommandName.UPLOAD_MATERIAL,
        }:
            raise ValueError(f"不支持流式执行的命令: {command.name}")

        if command.name == FinsCommandName.DOWNLOAD:
            typed_payload = _require_download_payload(command.payload)
            self._prepare_download_execution(typed_payload)
            return
        if command.name == FinsCommandName.PROCESS:
            typed_payload = _require_process_payload(command.payload)
            self._prepare_process_execution(typed_payload)
            return
        if command.name == FinsCommandName.UPLOAD_FILING:
            typed_payload = _require_upload_filing_payload(command.payload)
            self._prepare_upload_filing_execution(typed_payload)
            return
        if command.name == FinsCommandName.UPLOAD_MATERIAL:
            typed_payload = _require_upload_material_payload(command.payload)
            self._prepare_upload_material_execution(typed_payload)
            return
        if command.name == FinsCommandName.UPLOAD_FILINGS_FROM:
            typed_payload = _require_upload_filings_from_payload(command.payload)
            prepared_args = _build_upload_filings_from_args(typed_payload, self.workspace_root)
            # namespace 构建仅做 ticker / CLI 规范化；source_dir 存在性在
            # generate_upload_filings_script 内才检查，必须在 preflight 提前卡住
            source_dir = Path(prepared_args.source_dir).expanduser().resolve()
            if not source_dir.exists() or not source_dir.is_dir():
                raise FileNotFoundError(f"source_dir 不存在或不是目录: {source_dir}")
            return
        if command.name == FinsCommandName.PROCESS_FILING:
            typed_payload = _require_process_filing_payload(command.payload)
            if not typed_payload.document_id.strip():
                raise ValueError("process_filing 的 document_id 不能为空")
            self._build_pipeline_for_ticker(typed_payload.ticker)
            return
        if command.name == FinsCommandName.PROCESS_MATERIAL:
            typed_payload = _require_process_material_payload(command.payload)
            if not typed_payload.document_id.strip():
                raise ValueError("process_material 的 document_id 不能为空")
            self._build_pipeline_for_ticker(typed_payload.ticker)
            return

    def _prepare_download_execution(
        self,
        payload: DownloadCommandPayload,
    ) -> tuple[_PreparedDownloadArgs, PipelineProtocol]:
        """准备 download 命令的已校验参数与 pipeline。"""

        prepared_args = _build_download_args(payload, self.workspace_root)
        pipeline = self._build_pipeline_for_ticker(prepared_args.ticker)
        return prepared_args, pipeline

    def _prepare_process_execution(
        self,
        payload: ProcessCommandPayload,
    ) -> tuple[_PreparedProcessArgs, PipelineProtocol]:
        """准备 process 命令的已校验参数与 pipeline。"""

        prepared_args = _build_process_args(payload, self.workspace_root)
        pipeline = self._build_pipeline_for_ticker(prepared_args.ticker)
        return prepared_args, pipeline

    def _prepare_upload_filing_execution(
        self,
        payload: UploadFilingCommandPayload,
    ) -> tuple[_PreparedUploadFilingArgs, PipelineProtocol]:
        """准备 upload_filing 命令的已校验参数与 pipeline。

        Args:
            payload: 上传财报命令载荷。

        Returns:
            校验通过的命名空间与 pipeline。

        Raises:
            ValueError: 参数非法时抛出。
        """

        prepared_args = _build_upload_filing_args(payload, self.workspace_root)
        pipeline = self._build_pipeline_for_ticker(prepared_args.ticker)
        return prepared_args, pipeline

    def _prepare_upload_material_execution(
        self,
        payload: UploadMaterialCommandPayload,
    ) -> tuple[_PreparedUploadMaterialArgs, PipelineProtocol]:
        """准备 upload_material 命令的已校验参数与 pipeline。

        Args:
            payload: 上传材料命令载荷。

        Returns:
            校验通过的命名空间与 pipeline。

        Raises:
            ValueError: 参数非法时抛出。
        """

        prepared_args = _build_upload_material_args(payload, self.workspace_root)
        pipeline = self._build_pipeline_for_ticker(prepared_args.ticker)
        return prepared_args, pipeline

    def execute(
        self,
        command: FinsCommand,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> FinsResult | AsyncIterator[FinsEvent]:
        """执行财报命令。"""

        if command.stream:
            return self._execute_stream(command, cancel_checker=cancel_checker)
        result = self._execute_sync(command, cancel_checker=cancel_checker)
        return FinsResult(command=command.name, data=result)

    def _execute_sync(
        self,
        command: FinsCommand,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> FinsResultData:
        """执行同步命令。"""

        name = command.name
        payload = command.payload
        if name == FinsCommandName.UPLOAD_FILINGS_FROM:
            typed_payload = _require_upload_filings_from_payload(payload)
            prepared_args = _build_upload_filings_from_args(typed_payload, self.workspace_root)
            return _build_upload_filings_from_result_data(generate_upload_filings_script(prepared_args))

        if name == FinsCommandName.DOWNLOAD:
            typed_payload = _require_download_payload(payload)
            prepared_args, pipeline = self._prepare_download_execution(typed_payload)
            raw_result = pipeline.download(
                ticker=prepared_args.ticker,
                form_type=_coerce_forms_input(prepared_args.form_type),
                start_date=prepared_args.start_date,
                end_date=prepared_args.end_date,
                overwrite=prepared_args.overwrite,
                rebuild=prepared_args.rebuild,
                ticker_aliases=prepared_args.ticker_aliases,
            )
            return _build_download_result_data(raw_result)
        if name == FinsCommandName.UPLOAD_FILING:
            typed_payload = _require_upload_filing_payload(payload)
            prepared_args, pipeline = self._prepare_upload_filing_execution(typed_payload)
            raw_result = pipeline.upload_filing(
                ticker=prepared_args.ticker,
                action=prepared_args.action,
                files=list(typed_payload.files),
                fiscal_year=prepared_args.fiscal_year,
                fiscal_period=prepared_args.fiscal_period,
                amended=prepared_args.amended,
                filing_date=prepared_args.filing_date,
                report_date=prepared_args.report_date,
                company_id=prepared_args.company_id,
                company_name=prepared_args.company_name,
                ticker_aliases=prepared_args.ticker_aliases,
                overwrite=prepared_args.overwrite,
            )
            return _build_upload_filing_result_data(raw_result)
        if name == FinsCommandName.UPLOAD_MATERIAL:
            typed_payload = _require_upload_material_payload(payload)
            prepared_args, pipeline = self._prepare_upload_material_execution(typed_payload)
            raw_result = pipeline.upload_material(
                ticker=prepared_args.ticker,
                action=prepared_args.action,
                form_type=prepared_args.form_type,
                material_name=prepared_args.material_name,
                files=list(typed_payload.files),
                document_id=prepared_args.document_id,
                internal_document_id=prepared_args.internal_document_id,
                fiscal_year=prepared_args.fiscal_year,
                fiscal_period=prepared_args.fiscal_period,
                filing_date=prepared_args.filing_date,
                report_date=prepared_args.report_date,
                company_id=prepared_args.company_id,
                company_name=prepared_args.company_name,
                ticker_aliases=prepared_args.ticker_aliases,
                overwrite=prepared_args.overwrite,
            )
            return _build_upload_material_result_data(raw_result)
        if name == FinsCommandName.PROCESS:
            typed_payload = _require_process_payload(payload)
            prepared_args, pipeline = self._prepare_process_execution(typed_payload)
            raw_result = pipeline.process(
                ticker=prepared_args.ticker,
                overwrite=prepared_args.overwrite,
                ci=prepared_args.ci,
                document_ids=_coerce_document_ids_input(prepared_args.document_ids),
            )
            return _build_process_result_data(raw_result)
        if name == FinsCommandName.PROCESS_FILING:
            typed_payload = _require_process_filing_payload(payload)
            pipeline = self._build_pipeline_for_ticker(typed_payload.ticker)
            raw_result = pipeline.process_filing(
                ticker=typed_payload.ticker,
                document_id=typed_payload.document_id,
                overwrite=typed_payload.overwrite,
                ci=typed_payload.ci,
                cancel_checker=cancel_checker,
            )
            return _build_process_single_result_data(raw_result)
        if name == FinsCommandName.PROCESS_MATERIAL:
            typed_payload = _require_process_material_payload(payload)
            pipeline = self._build_pipeline_for_ticker(typed_payload.ticker)
            raw_result = pipeline.process_material(
                ticker=typed_payload.ticker,
                document_id=typed_payload.document_id,
                overwrite=typed_payload.overwrite,
                ci=typed_payload.ci,
                cancel_checker=cancel_checker,
            )
            return _build_process_single_result_data(raw_result)
        raise ValueError(f"不支持的命令: {name}")

    async def _execute_stream(
        self,
        command: FinsCommand,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> AsyncIterator[FinsEvent]:
        """执行流式命令。

        Args:
            command: 财报命令。
            cancel_checker: 可选取消检查函数；当前仅在已支持的长事务流式链路中继续下传。

        Yields:
            统一的财报事件。

        Raises:
            ValueError: 命令不支持流式执行时抛出。
            RuntimeError: 流式事件缺少最终结果时抛出。
        """

        name = command.name
        payload = command.payload

        if name == FinsCommandName.DOWNLOAD:
            typed_payload = _require_download_payload(payload)
            prepared_args, pipeline = self._prepare_download_execution(typed_payload)
            stream = pipeline.download_stream(
                ticker=prepared_args.ticker,
                form_type=_coerce_forms_input(prepared_args.form_type),
                start_date=prepared_args.start_date,
                end_date=prepared_args.end_date,
                overwrite=prepared_args.overwrite,
                rebuild=prepared_args.rebuild,
                ticker_aliases=prepared_args.ticker_aliases,
                cancel_checker=cancel_checker,
            )
            async for event in self._iter_stream_events(command_name=name, stream=stream, final_event_types={"pipeline_completed"}):
                yield event
            return

        if name == FinsCommandName.PROCESS:
            typed_payload = _require_process_payload(payload)
            prepared_args, pipeline = self._prepare_process_execution(typed_payload)
            stream = pipeline.process_stream(
                ticker=prepared_args.ticker,
                overwrite=prepared_args.overwrite,
                ci=prepared_args.ci,
                document_ids=_coerce_document_ids_input(prepared_args.document_ids),
                cancel_checker=cancel_checker,
            )
            async for event in self._iter_stream_events(command_name=name, stream=stream, final_event_types={"pipeline_completed"}):
                yield event
            return

        if name == FinsCommandName.UPLOAD_FILING:
            typed_payload = _require_upload_filing_payload(payload)
            prepared_args, pipeline = self._prepare_upload_filing_execution(typed_payload)
            stream = pipeline.upload_filing_stream(
                ticker=prepared_args.ticker,
                action=prepared_args.action,
                files=list(typed_payload.files),
                fiscal_year=prepared_args.fiscal_year,
                fiscal_period=prepared_args.fiscal_period,
                amended=prepared_args.amended,
                filing_date=prepared_args.filing_date,
                report_date=prepared_args.report_date,
                company_id=prepared_args.company_id,
                company_name=prepared_args.company_name,
                ticker_aliases=prepared_args.ticker_aliases,
                overwrite=prepared_args.overwrite,
            )
            async for event in self._iter_stream_events(
                command_name=name,
                stream=stream,
                final_event_types={"upload_completed", "upload_failed"},
            ):
                yield event
            return

        if name == FinsCommandName.UPLOAD_MATERIAL:
            typed_payload = _require_upload_material_payload(payload)
            prepared_args, pipeline = self._prepare_upload_material_execution(typed_payload)
            stream = pipeline.upload_material_stream(
                ticker=prepared_args.ticker,
                action=prepared_args.action,
                form_type=prepared_args.form_type,
                material_name=prepared_args.material_name,
                files=list(typed_payload.files),
                document_id=prepared_args.document_id,
                internal_document_id=prepared_args.internal_document_id,
                fiscal_year=prepared_args.fiscal_year,
                fiscal_period=prepared_args.fiscal_period,
                filing_date=prepared_args.filing_date,
                report_date=prepared_args.report_date,
                company_id=prepared_args.company_id,
                company_name=prepared_args.company_name,
                ticker_aliases=prepared_args.ticker_aliases,
                overwrite=prepared_args.overwrite,
            )
            async for event in self._iter_stream_events(
                command_name=name,
                stream=stream,
                final_event_types={"upload_completed", "upload_failed"},
            ):
                yield event
            return

        raise ValueError(f"不支持流式执行的命令: {name}")

    async def _iter_stream_events(
        self,
        *,
        command_name: FinsCommandName,
        stream: AsyncIterator[DownloadEvent | ProcessEvent | UploadFilingEvent | UploadMaterialEvent],
        final_event_types: set[str],
    ) -> AsyncIterator[FinsEvent]:
        """将 pipeline 事件流转换为统一 FinsEvent 流。"""

        final_result: FinsResultData | None = None
        async for event in stream:
            if command_name == FinsCommandName.DOWNLOAD:
                progress_payload = _build_download_progress_payload(_require_download_event(event))
            elif command_name == FinsCommandName.PROCESS:
                progress_payload = _build_process_progress_payload(_require_process_event(event))
            elif command_name == FinsCommandName.UPLOAD_FILING:
                progress_payload = _build_upload_filing_progress_payload(_require_upload_filing_event(event))
            else:
                progress_payload = _build_upload_material_progress_payload(_require_upload_material_event(event))
            yield FinsEvent(
                type=FinsEventType.PROGRESS,
                command=command_name,
                payload=progress_payload,
            )
            if event.event_type in final_event_types:
                extracted = _extract_result_data_from_event(command_name=command_name, event_payload=event.payload)
                if extracted is not None:
                    final_result = extracted

        if final_result is None:
            raise RuntimeError(f"{command_name} 事件流未返回最终结果")

        yield FinsEvent(
            type=FinsEventType.RESULT,
            command=command_name,
            payload=final_result,
        )


__all__ = [
    "CompanyMetaProviderProtocol",
    "DefaultFinsRuntime",
    "FinsRuntimeProtocol",
]
