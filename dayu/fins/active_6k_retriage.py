"""active 6-K 复判与误收剔除工具。

该模块用于扫描当前 active filings 中的 `6-K`，并以 `_classify_6k_text()` 为真源
重新判定其是否属于季度结果披露。对当前规则明确判为非季度披露的样本，可以通过
仓储对称地写回 `.rejections/`，并把 active source 逻辑删除。

关键约束：
- 只通过 `dayu.fins.storage` 下的仓储协议操作 active filings 与 `.rejections/`。
- 当前分类真源仅为 `_classify_6k_text()`，不在该模块实现第二套规则。
- 对 active 文档的“剔除”采用逻辑删除 + rejected artifact 存档，而不是手改 manifest
  或直接删除目录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

from dayu.log import Log

from dayu.fins.domain.document_models import (
    CompanyMeta,
    DocumentMeta,
    ProcessedDeleteRequest,
    RejectedFilingArtifactUpsertRequest,
    SourceDocumentStateChangeRequest,
    SourceFileEntry,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins._converters import optional_int
from dayu.fins.pipelines.sec_6k_rules import _classify_6k_text, _extract_head_text
from dayu.fins.pipelines.sec_pipeline import SEC_PIPELINE_DOWNLOAD_VERSION
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

_RETRIAGE_KEEP_CLASSIFICATIONS = frozenset({"RESULTS_RELEASE", "IFRS_RECON"})
_MODULE = "FINS.ACTIVE_6K_RETRIAGE"


@dataclass(frozen=True, slots=True)
class Active6KRetriageCandidate:
    """待剔除的 active 6-K 候选。"""

    ticker: str
    document_id: str
    current_classification: str
    primary_document: str


@dataclass(frozen=True, slots=True)
class Active6KRetriageOutcome:
    """单个 active 6-K 复判结果。"""

    ticker: str
    document_id: str
    action: Literal["rejected", "skipped"]
    reason: str
    current_classification: str


@dataclass(frozen=True, slots=True)
class Active6KRetriageReport:
    """active 6-K 复判报告。"""

    workspace_root: str
    apply: bool
    candidates: list[Active6KRetriageCandidate] = field(default_factory=list)
    outcomes: list[Active6KRetriageOutcome] = field(default_factory=list)


def retriage_active_6k_filings(
    *,
    workspace_root: Path,
    apply: bool,
    target_tickers: Optional[list[str]] = None,
    target_document_ids: Optional[list[str]] = None,
    company_repository: Optional[CompanyMetaRepositoryProtocol] = None,
    source_repository: Optional[SourceDocumentRepositoryProtocol] = None,
    blob_repository: Optional[DocumentBlobRepositoryProtocol] = None,
    maintenance_repository: Optional[FilingMaintenanceRepositoryProtocol] = None,
    processed_repository: Optional[ProcessedDocumentRepositoryProtocol] = None,
) -> Active6KRetriageReport:
    """复判 active filings 中的 6-K，并把误收样本写回 `.rejections/`。

    Args:
        workspace_root: workspace 根目录。
        apply: 是否实际执行误收剔除；`False` 时仅 dry-run。
        target_tickers: 可选 ticker 子集。
        target_document_ids: 可选 document_id 子集。
        company_repository: 可选公司仓储，便于测试注入。
        source_repository: 可选 source 仓储，便于测试注入。
        blob_repository: 可选 blob 仓储，便于测试注入。
        maintenance_repository: 可选 filing maintenance 仓储，便于测试注入。
        processed_repository: 可选 processed 仓储，便于测试注入。

    Returns:
        复判报告。

    Raises:
        OSError: 仓储读写失败时抛出。
        ValueError: 元数据非法时抛出。
    """

    resolved_workspace_root = workspace_root.resolve()
    normalized_target_tickers = _normalize_targets(target_tickers, uppercase=True)
    normalized_document_ids = _normalize_targets(target_document_ids, uppercase=False)
    document_id_filter = set(normalized_document_ids or [])

    effective_company_repository = company_repository or FsCompanyMetaRepository(resolved_workspace_root)
    effective_source_repository = source_repository or FsSourceDocumentRepository(resolved_workspace_root)
    effective_blob_repository = blob_repository or FsDocumentBlobRepository(resolved_workspace_root)
    effective_maintenance_repository = maintenance_repository or FsFilingMaintenanceRepository(
        resolved_workspace_root
    )
    effective_processed_repository = processed_repository or FsProcessedDocumentRepository(
        resolved_workspace_root
    )

    tickers = _resolve_target_tickers(
        company_repository=effective_company_repository,
        target_tickers=normalized_target_tickers,
    )

    candidates: list[Active6KRetriageCandidate] = []
    outcomes: list[Active6KRetriageOutcome] = []
    for ticker in tickers:
        rejection_registry = effective_maintenance_repository.load_download_rejection_registry(ticker)
        registry_changed = False
        company_meta = _get_company_meta_if_present(effective_company_repository, ticker)
        for document_id in effective_source_repository.list_source_document_ids(ticker, SourceKind.FILING):
            if document_id_filter and document_id not in document_id_filter:
                continue
            meta = _get_source_meta_if_present(
                source_repository=effective_source_repository,
                ticker=ticker,
                document_id=document_id,
            )
            if meta is None:
                continue
            if bool(meta.get("is_deleted", False)):
                continue
            if str(meta.get("form_type", "")).strip().upper() != "6-K":
                continue
            primary_document = str(meta.get("primary_document", "")).strip()
            if not primary_document:
                outcomes.append(
                    Active6KRetriageOutcome(
                        ticker=ticker,
                        document_id=document_id,
                        action="skipped",
                        reason="missing_primary_document",
                        current_classification="NO_MATCH",
                    )
                )
                continue
            current_classification = _classify_active_source_document(
                source_repository=effective_source_repository,
                blob_repository=effective_blob_repository,
                ticker=ticker,
                document_id=document_id,
                primary_document=primary_document,
            )
            if current_classification in _RETRIAGE_KEEP_CLASSIFICATIONS:
                continue
            candidates.append(
                Active6KRetriageCandidate(
                    ticker=ticker,
                    document_id=document_id,
                    current_classification=current_classification,
                    primary_document=primary_document,
                )
            )
            if not apply:
                outcomes.append(
                    Active6KRetriageOutcome(
                        ticker=ticker,
                        document_id=document_id,
                        action="skipped",
                        reason="dry_run",
                        current_classification=current_classification,
                    )
                )
                continue

            _archive_active_filing_as_rejected(
                source_repository=effective_source_repository,
                blob_repository=effective_blob_repository,
                maintenance_repository=effective_maintenance_repository,
                company_meta=company_meta,
                ticker=ticker,
                document_id=document_id,
                meta=meta,
                rejection_category=current_classification,
            )
            _record_rejection(
                registry=rejection_registry,
                document_id=document_id,
                reason="6k_filtered",
                category=current_classification,
                form_type="6-K",
                filing_date=str(meta.get("filing_date", "")),
            )
            registry_changed = True
            effective_source_repository.delete_source_document(
                SourceDocumentStateChangeRequest(
                    ticker=ticker,
                    document_id=document_id,
                    source_kind=SourceKind.FILING.value,
                )
            )
            _delete_processed_if_present(effective_processed_repository, ticker, document_id)
            outcomes.append(
                Active6KRetriageOutcome(
                    ticker=ticker,
                    document_id=document_id,
                    action="rejected",
                    reason="moved_to_rejections",
                    current_classification=current_classification,
                )
            )

        if apply and registry_changed:
            effective_maintenance_repository.save_download_rejection_registry(ticker, rejection_registry)

    return Active6KRetriageReport(
        workspace_root=str(resolved_workspace_root),
        apply=apply,
        candidates=candidates,
        outcomes=outcomes,
    )


def _normalize_targets(raw_items: Optional[list[str]], *, uppercase: bool) -> Optional[list[str]]:
    """规范化可选目标列表。

    Args:
        raw_items: 原始目标列表。
        uppercase: 是否统一转换为大写。

    Returns:
        去空、去重后的目标列表；若为空则返回 `None`。

    Raises:
        无。
    """

    if raw_items is None:
        return None
    normalized: list[str] = []
    for item in raw_items:
        cleaned = str(item).strip()
        if not cleaned:
            continue
        if uppercase:
            cleaned = cleaned.upper()
        if cleaned in normalized:
            continue
        normalized.append(cleaned)
    return normalized or None


def _resolve_target_tickers(
    *,
    company_repository: CompanyMetaRepositoryProtocol,
    target_tickers: Optional[list[str]],
) -> list[str]:
    """解析本次需要扫描的 ticker 列表。

    Args:
        company_repository: 公司元数据仓储。
        target_tickers: 显式指定的 ticker 子集。

    Returns:
        需要扫描的规范 ticker 列表。

    Raises:
        OSError: 扫描公司元数据失败时抛出。
        ValueError: 公司元数据非法时抛出。
    """

    if target_tickers is not None:
        return list(target_tickers)
    tickers: list[str] = []
    for entry in company_repository.scan_company_meta_inventory():
        if entry.status != "available" or entry.company_meta is None:
            continue
        ticker = entry.company_meta.ticker.strip().upper()
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def _get_company_meta_if_present(
    company_repository: CompanyMetaRepositoryProtocol,
    ticker: str,
) -> Optional[CompanyMeta]:
    """安全读取公司元数据。

    Args:
        company_repository: 公司元数据仓储。
        ticker: 股票代码。

    Returns:
        命中时返回公司元数据，否则返回 `None`。

    Raises:
        OSError: 仓储读取失败时抛出。
        ValueError: 元数据非法时抛出。
    """

    try:
        return company_repository.get_company_meta(ticker)
    except FileNotFoundError:
        return None


def _get_source_meta_if_present(
    *,
    source_repository: SourceDocumentRepositoryProtocol,
    ticker: str,
    document_id: str,
) -> Optional[DocumentMeta]:
    """安全读取 active source meta。

    Args:
        source_repository: source 仓储。
        ticker: 股票代码。
        document_id: 文档 ID。

    Returns:
        命中时返回 source meta；若 active source 不存在或元数据损坏则返回 `None`。

    Raises:
        OSError: 仓储读取失败时抛出。
    """

    try:
        return source_repository.get_source_meta(ticker, document_id, SourceKind.FILING)
    except FileNotFoundError:
        return None
    except ValueError as exc:
        Log.warn(
            (
                "跳过损坏的 active filing meta: "
                f"ticker={ticker} document_id={document_id} error={exc}"
            ),
            module=_MODULE,
        )
        return None


def _classify_active_source_document(
    *,
    source_repository: SourceDocumentRepositoryProtocol,
    blob_repository: DocumentBlobRepositoryProtocol,
    ticker: str,
    document_id: str,
    primary_document: str,
) -> str:
    """读取 active 6-K 主文件并以 `_classify_6k_text()` 重新分类。

    Args:
        source_repository: source 仓储。
        blob_repository: blob 仓储。
        ticker: 股票代码。
        document_id: 文档 ID。
        primary_document: 当前 active source 主文件名。

    Returns:
        当前规则下的 6-K 分类标签。

    Raises:
        FileNotFoundError: 主文件不存在时抛出。
        OSError: 文件读取失败时抛出。
    """

    handle = source_repository.get_source_handle(ticker, document_id, SourceKind.FILING)
    payload = blob_repository.read_file_bytes(handle, primary_document)
    head_text = _extract_head_text(payload, max_lines=120)
    return _classify_6k_text(head_text)


def _archive_active_filing_as_rejected(
    *,
    source_repository: SourceDocumentRepositoryProtocol,
    blob_repository: DocumentBlobRepositoryProtocol,
    maintenance_repository: FilingMaintenanceRepositoryProtocol,
    company_meta: Optional[CompanyMeta],
    ticker: str,
    document_id: str,
    meta: DocumentMeta,
    rejection_category: str,
) -> None:
    """把 active filing 对称写回 `.rejections/`。

    Args:
        source_repository: source 仓储。
        blob_repository: blob 仓储。
        maintenance_repository: filing maintenance 仓储。
        company_meta: 公司元数据，可为空。
        ticker: 股票代码。
        document_id: 文档 ID。
        meta: active source meta。
        rejection_category: 当前规则分类结果。

    Returns:
        无。

    Raises:
        OSError: 仓储写入失败时抛出。
        ValueError: 元数据非法时抛出。
    """

    typed_entries = _extract_active_source_file_entries(meta)
    source_handle = source_repository.get_source_handle(ticker, document_id, SourceKind.FILING)
    rejected_entries: list[SourceFileEntry] = []
    for entry in typed_entries:
        payload = blob_repository.read_file_bytes(source_handle, entry.name)
        stored_meta = maintenance_repository.store_rejected_filing_file(
            ticker=ticker,
            document_id=document_id,
            filename=entry.name,
            data=BytesIO(payload),
            content_type=entry.content_type,
        )
        rejected_entries.append(
            SourceFileEntry(
                name=entry.name,
                uri=stored_meta.uri,
                etag=stored_meta.etag,
                last_modified=stored_meta.last_modified,
                size=stored_meta.size,
                content_type=stored_meta.content_type,
                sha256=stored_meta.sha256,
                source_url=entry.source_url,
                http_etag=entry.http_etag,
                http_last_modified=entry.http_last_modified,
                ingested_at=entry.ingested_at,
            )
        )

    internal_document_id = str(meta.get("internal_document_id", document_id.removeprefix("fil_"))).strip()
    accession_number = str(meta.get("accession_number", internal_document_id)).strip()
    company_id = str(meta.get("company_id", company_meta.company_id if company_meta is not None else "")).strip()
    primary_document = str(meta.get("primary_document", "")).strip()

    maintenance_repository.upsert_rejected_filing_artifact(
        RejectedFilingArtifactUpsertRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=internal_document_id,
            accession_number=accession_number,
            company_id=company_id,
            form_type=str(meta.get("form_type", "6-K")),
            filing_date=str(meta.get("filing_date", "")),
            report_date=_optional_string(meta.get("report_date")),
            primary_document=primary_document,
            selected_primary_document=primary_document,
            rejection_reason="6k_filtered",
            rejection_category=rejection_category,
            classification_version=SEC_PIPELINE_DOWNLOAD_VERSION,
            source_fingerprint=str(meta.get("source_fingerprint", "")),
            files=rejected_entries,
            fiscal_year=optional_int(meta.get("fiscal_year")),
            fiscal_period=_optional_string(meta.get("fiscal_period")),
            report_kind=_optional_string(meta.get("report_kind")),
            amended=bool(meta.get("amended", False)),
            has_xbrl=_optional_bool(meta.get("has_xbrl")),
            ingest_method=str(meta.get("ingest_method", "download")) or "download",
        )
    )


def _extract_active_source_file_entries(meta: DocumentMeta) -> list[SourceFileEntry]:
    """从 active source meta 提取 typed 文件条目。

    Args:
        meta: active source meta。

    Returns:
        typed 文件条目列表。

    Raises:
        ValueError: 文件条目非法时抛出。
    """

    typed_entries: list[SourceFileEntry] = []
    for item in meta.get("files", []):
        if not isinstance(item, dict):
            continue
        typed_entries.append(SourceFileEntry.from_dict(item))
    return typed_entries


def _record_rejection(
    *,
    registry: dict[str, dict[str, str]],
    document_id: str,
    reason: str,
    category: str,
    form_type: str,
    filing_date: str,
) -> None:
    """向下载拒绝注册表写入一条 6-K 拒绝记录。

    Args:
        registry: 下载拒绝注册表（就地修改）。
        document_id: 文档 ID。
        reason: 拒绝原因。
        category: 当前分类标签。
        form_type: 表单类型。
        filing_date: 申报日期。

    Returns:
        无。

    Raises:
        无。
    """

    registry[document_id] = {
        "reason": reason,
        "category": category,
        "form_type": form_type,
        "filing_date": filing_date,
        "download_version": SEC_PIPELINE_DOWNLOAD_VERSION,
    }


def _delete_processed_if_present(
    processed_repository: ProcessedDocumentRepositoryProtocol,
    ticker: str,
    document_id: str,
) -> None:
    """尽力删除已存在的 processed 产物，避免误收样本继续污染下游统计。

    Args:
        processed_repository: processed 仓储。
        ticker: 股票代码。
        document_id: 文档 ID。

    Returns:
        无。

    Raises:
        OSError: 删除失败时抛出。
    """

    try:
        processed_repository.delete_processed(
            ProcessedDeleteRequest(ticker=ticker, document_id=document_id)
        )
    except FileNotFoundError:
        return


def _optional_string(value: object) -> Optional[str]:
    """把可选值安全转换为字符串。

    Args:
        value: 原始值。

    Returns:
        清洗后的字符串；空值返回 `None`。

    Raises:
        无。
    """

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_bool(value: object) -> Optional[bool]:
    """把可选值安全转换为布尔值。

    Args:
        value: 原始值。

    Returns:
        合法布尔值时返回该值，否则返回 `None`。

    Raises:
        无。
    """

    if isinstance(value, bool):
        return value
    return None