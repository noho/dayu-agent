"""SEC 下载重建工作流真源。"""

from __future__ import annotations

import datetime as dt
import time
from typing import Any, Callable, Optional, Protocol

from dayu.fins.domain.document_models import CompanyMeta, FilingUpdateRequest, SourceHandle, now_iso8601
from dayu.fins.domain.enums import SourceKind
from dayu.fins.storage import SourceDocumentRepositoryProtocol

from .sec_download_state import _normalize_rebuild_file_entries, _resolve_rebuild_source_fingerprint
from .sec_fiscal_fields import (
    _coerce_optional_int,
    _normalize_form_for_fiscal,
    _normalize_optional_period,
    _resolve_download_fiscal_fields,
)


class SecRebuildWorkflowHost(Protocol):
    """重建工作流所需的最小宿主边界。"""

    @property
    def _source_repository(self) -> SourceDocumentRepositoryProtocol:
        """返回 source 仓储。"""

        ...

    def _safe_get_company_meta(self, ticker: str) -> Optional[CompanyMeta]:
        """安全读取公司元数据。"""

        ...

    def _safe_get_document_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> Optional[dict[str, Any]]:
        """安全读取文档 meta。"""

        ...

    def _log_filing_download_result(
        self,
        ticker: str,
        filing_result: dict[str, Any],
    ) -> None:
        """记录单 filing 结果。"""

        ...

    def _build_result(self, action: str, **payload: Any) -> dict[str, Any]:
        """构建统一结果。"""

        ...


def _should_preserve_previous_rebuild_fiscal_fields(form_type: str) -> bool:
    """判断重建时是否允许沿用 previous_meta 中的 fiscal 字段。

    Args:
        form_type: source meta 中记录的表单类型。

    Returns:
        若允许在重建推断为空时沿用 previous_meta fiscal 字段则返回 `True`。

    Raises:
        无。
    """

    normalized_form = _normalize_form_for_fiscal(form_type)
    return normalized_form not in {"6-K", "6-K/A"}


def rebuild_download_artifacts(
    host: SecRebuildWorkflowHost,
    *,
    ticker: str,
    form_type: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    overwrite: bool,
    pipeline_download_version: str,
    expand_form_aliases: Callable[[list[str]], list[str]],
    split_form_input: Callable[[str], list[str]],
    parse_date: Callable[[str, bool], dt.date],
    normalize_form: Callable[[str], str],
    overwrite_rebuilt_meta: Callable[[SourceDocumentRepositoryProtocol, str, str, str, list[dict[str, Any]], dict[str, Any]], None],
) -> dict[str, Any]:
    """基于本地已下载 filings 重建 meta/manifest。"""

    target_forms, start_bound, end_bound = build_rebuild_filter_spec(
        form_type=form_type,
        start_date=start_date,
        end_date=end_date,
        expand_form_aliases=expand_form_aliases,
        split_form_input=split_form_input,
        parse_date=parse_date,
    )
    company_meta = host._safe_get_company_meta(ticker=ticker)
    document_ids = host._source_repository.list_source_document_ids(
        ticker=ticker,
        source_kind=SourceKind.FILING,
    )
    warnings: list[str] = []
    filing_results: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for document_id in document_ids:
        previous_meta = host._safe_get_document_meta(
            ticker=ticker,
            document_id=document_id,
            source_kind=SourceKind.FILING,
        )
        if previous_meta is None:
            continue
        if str(previous_meta.get("ingest_method", "")).strip().lower() != "download":
            continue
        if not passes_rebuild_filters(
            meta=previous_meta,
            target_forms=target_forms,
            start_bound=start_bound,
            end_bound=end_bound,
            normalize_form=normalize_form,
            parse_date=parse_date,
        ):
            continue
        filing_result = rebuild_single_local_filing(
            source_repository=host._source_repository,
            ticker=ticker,
            document_id=document_id,
            previous_meta=previous_meta,
            company_meta=company_meta,
            pipeline_download_version=pipeline_download_version,
            overwrite_rebuilt_meta=overwrite_rebuilt_meta,
        )
        filing_results.append(filing_result)
        host._log_filing_download_result(ticker=ticker, filing_result=filing_result)

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    if not filing_results:
        warnings.append("未匹配到可重建的已下载 filings")
    summary = {
        "total": len(filing_results),
        "downloaded": sum(1 for item in filing_results if item.get("status") == "downloaded"),
        "skipped": sum(1 for item in filing_results if item.get("status") == "skipped"),
        "failed": sum(1 for item in filing_results if item.get("status") == "failed"),
        "elapsed_ms": elapsed_ms,
        "reused_downloads": 0,
        "converted": 0,
    }
    return host._build_result(
        action="download",
        ticker=ticker,
        market_profile={"market": "US"},
        filters={
            "forms": sorted(target_forms) if target_forms is not None else None,
            "start_date": start_date,
            "end_date": end_date,
            "overwrite": overwrite,
            "rebuild": True,
        },
        warnings=warnings,
        filings=filing_results,
        summary=summary,
    )


def build_rebuild_filter_spec(
    *,
    form_type: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    expand_form_aliases: Callable[[list[str]], list[str]],
    split_form_input: Callable[[str], list[str]],
    parse_date: Callable[[str, bool], dt.date],
) -> tuple[Optional[set[str]], Optional[dt.date], Optional[dt.date]]:
    """构建重建模式本地过滤条件。"""

    target_forms: Optional[set[str]] = None
    if form_type:
        target_forms = set(expand_form_aliases(split_form_input(form_type)))
    start_bound = parse_date(start_date, False) if start_date else None
    end_bound = parse_date(end_date, True) if end_date else None
    if start_bound is not None and end_bound is not None and start_bound > end_bound:
        raise ValueError("start_date 不能晚于 end_date")
    return target_forms, start_bound, end_bound


def passes_rebuild_filters(
    *,
    meta: dict[str, Any],
    target_forms: Optional[set[str]],
    start_bound: Optional[dt.date],
    end_bound: Optional[dt.date],
    normalize_form: Callable[[str], str],
    parse_date: Callable[[str, bool], dt.date],
) -> bool:
    """判断本地 meta 是否满足重建过滤条件。"""

    if target_forms is not None:
        raw_form_type = str(meta.get("form_type", "")).strip()
        if not raw_form_type:
            return False
        try:
            normalized_form = normalize_form(raw_form_type)
        except ValueError:
            return False
        if normalized_form not in target_forms:
            return False
    if start_bound is None and end_bound is None:
        return True
    raw_filing_date = str(meta.get("filing_date", "")).strip()
    if not raw_filing_date:
        return False
    try:
        filing_date = parse_date(raw_filing_date, False)
    except ValueError:
        return False
    if start_bound is not None and filing_date < start_bound:
        return False
    if end_bound is not None and filing_date > end_bound:
        return False
    return True


def rebuild_single_local_filing(
    *,
    source_repository: SourceDocumentRepositoryProtocol,
    ticker: str,
    document_id: str,
    previous_meta: dict[str, Any],
    company_meta: Optional[CompanyMeta],
    pipeline_download_version: str,
    overwrite_rebuilt_meta: Callable[[SourceDocumentRepositoryProtocol, str, str, str, list[dict[str, Any]], dict[str, Any]], None],
) -> dict[str, Any]:
    """重建单个本地 filing 的 meta/manifest。"""

    raw_internal_document_id = str(previous_meta.get("internal_document_id", "")).strip()
    internal_document_id = (
        raw_internal_document_id
        if raw_internal_document_id
        else (document_id[4:] if document_id.startswith("fil_") else document_id)
    )
    accession_number = str(previous_meta.get("accession_number", "")).strip() or internal_document_id
    form_type = str(previous_meta.get("form_type", "")).strip()
    filing_date = str(previous_meta.get("filing_date", "")).strip() or None
    report_date = str(previous_meta.get("report_date", "")).strip() or None
    if not form_type:
        return {
            "document_id": document_id,
            "internal_document_id": internal_document_id,
            "status": "failed",
            "error": "missing_form_type",
            "reason_code": "missing_form_type",
            "reason_message": "重建失败：meta.json 缺少 form_type",
            "rebuild": True,
        }

    file_entries = _normalize_rebuild_file_entries(previous_meta=previous_meta)
    if not file_entries:
        return {
            "document_id": document_id,
            "internal_document_id": internal_document_id,
            "status": "failed",
            "form_type": form_type,
            "filing_date": filing_date,
            "report_date": report_date,
            "error": "missing_files",
            "reason_code": "missing_files",
            "reason_message": "重建失败：meta.json 缺少可用 files 列表",
            "rebuild": True,
        }

    source_fingerprint = _resolve_rebuild_source_fingerprint(
        previous_meta=previous_meta,
        file_entries=file_entries,
    )
    primary_document = str(previous_meta.get("primary_document", "")).strip() or file_entries[0]["name"]
    source_handle = SourceHandle(
        ticker=ticker,
        document_id=document_id,
        source_kind=SourceKind.FILING.value,
    )
    inferred_fiscal_year, inferred_fiscal_period = _resolve_download_fiscal_fields(
        source_handle=source_handle,
        source_repository=source_repository,
        file_entries=file_entries,
        form_type=form_type,
        report_date=report_date,
    )
    if inferred_fiscal_year is None and _should_preserve_previous_rebuild_fiscal_fields(form_type):
        inferred_fiscal_year = _coerce_optional_int(previous_meta.get("fiscal_year"))
    if inferred_fiscal_period is None and _should_preserve_previous_rebuild_fiscal_fields(form_type):
        inferred_fiscal_period = _normalize_optional_period(previous_meta.get("fiscal_period"))

    first_ingested_at = str(previous_meta.get("first_ingested_at", "")).strip() or now_iso8601()
    created_at = str(previous_meta.get("created_at", "")).strip() or first_ingested_at
    document_version = str(previous_meta.get("document_version", "v1")).strip() or "v1"
    company_id = str(previous_meta.get("company_id", "")).strip()
    if not company_id and company_meta is not None:
        company_id = str(company_meta.company_id).strip()
    amended = bool(previous_meta.get("amended", form_type.endswith("/A")))
    meta_payload = {
        "document_id": document_id,
        "internal_document_id": internal_document_id,
        "accession_number": accession_number,
        "ingest_method": "download",
        "ticker": ticker,
        "company_id": company_id,
        "form_type": form_type,
        "fiscal_year": inferred_fiscal_year,
        "fiscal_period": inferred_fiscal_period,
        "report_kind": None,
        "report_date": report_date,
        "filing_date": filing_date,
        "first_ingested_at": first_ingested_at,
        "ingest_complete": bool(previous_meta.get("ingest_complete", True)),
        "is_deleted": bool(previous_meta.get("is_deleted", False)),
        "deleted_at": previous_meta.get("deleted_at"),
        "document_version": document_version,
        "source_fingerprint": source_fingerprint,
        "amended": amended,
        "download_version": pipeline_download_version,
        "created_at": created_at,
        "updated_at": now_iso8601(),
    }
    try:
        has_xbrl = source_repository.has_filing_xbrl_instance(
            ticker=ticker,
            document_id=document_id,
        )
    except (FileNotFoundError, OSError):
        has_xbrl = None
    meta_payload["has_xbrl"] = has_xbrl
    try:
        source_repository.update_source_document(
            FilingUpdateRequest(
                ticker=ticker,
                document_id=document_id,
                internal_document_id=internal_document_id,
                form_type=form_type,
                primary_document=primary_document,
                file_entries=file_entries,
                meta=meta_payload,
            ),
            source_kind=SourceKind.FILING,
        )
        overwrite_rebuilt_meta(
            source_repository,
            ticker,
            document_id,
            primary_document,
            file_entries,
            meta_payload,
        )
    except Exception as exc:
        return {
            "document_id": document_id,
            "internal_document_id": internal_document_id,
            "status": "failed",
            "form_type": form_type,
            "filing_date": filing_date,
            "report_date": report_date,
            "error": str(exc),
            "reason_code": "rebuild_write_failed",
            "reason_message": str(exc),
            "rebuild": True,
        }

    return {
        "document_id": document_id,
        "internal_document_id": internal_document_id,
        "status": "downloaded",
        "form_type": form_type,
        "filing_date": filing_date,
        "report_date": report_date,
        "downloaded_files": 0,
        "skipped_files": len(file_entries),
        "rebuild": True,
    }


def overwrite_rebuilt_meta(
    repository: SourceDocumentRepositoryProtocol,
    ticker: str,
    document_id: str,
    primary_document: str,
    file_entries: list[dict[str, Any]],
    canonical_meta: dict[str, Any],
) -> None:
    """通过 repository 精确覆盖重建后的 source meta。"""

    payload = dict(canonical_meta)
    payload["primary_document"] = primary_document
    payload["files"] = file_entries
    repository.replace_source_meta(
        ticker=ticker,
        document_id=document_id,
        source_kind=SourceKind.FILING,
        meta=payload,
    )


__all__ = [
    "_should_preserve_previous_rebuild_fiscal_fields",
    "SecRebuildWorkflowHost",
    "build_rebuild_filter_spec",
    "overwrite_rebuilt_meta",
    "passes_rebuild_filters",
    "rebuild_download_artifacts",
    "rebuild_single_local_filing",
]
