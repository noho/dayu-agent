"""CN/HK 下载成功后的 source document commit 真源。

本模块负责把已经落盘的 PDF 与 Docling JSON 提交为完成态 source meta：
``ingest_complete=True``、``primary_document`` 指向 ``_docling.json``、
写入 ``download_version`` / ``remote_fingerprint`` / ``pdf_sha256`` /
``source_fingerprint``，并在必要时标记 processed 需要重处理。
"""

from __future__ import annotations

import hashlib
from typing import TypeAlias

from dayu.fins.domain.document_models import (
    FileObjectMeta,
    FilingCreateRequest,
    FilingUpdateRequest,
    SourceHandle,
    now_iso8601,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.cn_download_models import (
    CN_PIPELINE_DOWNLOAD_VERSION,
    CnCompanyProfile,
    CnReportCandidate,
)
from dayu.fins.storage import (
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)
from dayu.fins.ticker_normalization import normalize_ticker, ticker_to_company_id

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

_INGEST_METHOD_DOWNLOAD = "download"
_FISCAL_YEAR_SOURCE = "title_or_category_inferred"
_REPORT_DATE_SOURCE = "period_inferred"


def build_cn_file_entry(
    *,
    filename: str,
    file_meta: FileObjectMeta,
    source_label: str,
) -> JsonObject:
    """构建 source meta ``files[]`` 条目。

    Args:
        filename: 文件名。
        file_meta: blob 仓储返回的文件对象元数据。
        source_label: 文件来源标签，例如 ``"original"`` / ``"docling"``。

    Returns:
        可写入 ``file_entries`` 的 JSON 对象。

    Raises:
        ValueError: ``filename`` 或 ``source_label`` 为空时抛出。
    """

    name = filename.strip()
    label = source_label.strip()
    if not name:
        raise ValueError("filename 不能为空")
    if not label:
        raise ValueError("source_label 不能为空")
    return {
        "name": name,
        "uri": file_meta.uri,
        "etag": file_meta.etag,
        "last_modified": file_meta.last_modified,
        "size": file_meta.size,
        "content_type": file_meta.content_type,
        "sha256": file_meta.sha256,
        "source": label,
        "ingested_at": now_iso8601(),
    }


def build_remote_fingerprint(candidate: CnReportCandidate) -> str:
    """计算下载前可得的远端 fingerprint。

    Args:
        candidate: 远端候选报告。

    Returns:
        稳定 fingerprint 字符串。

    Raises:
        无。
    """

    parts = (
        candidate.provider,
        candidate.source_id,
        candidate.source_url,
        str(candidate.content_length or ""),
        candidate.etag or "",
        candidate.last_modified or "",
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def build_content_fingerprint(*, pdf_bytes: bytes, docling_json_bytes: bytes) -> str:
    """计算完成态 source fingerprint。

    Args:
        pdf_bytes: PDF 字节。
        docling_json_bytes: Docling JSON 字节。

    Returns:
        PDF SHA-256 与 Docling JSON SHA-256 组合后的 fingerprint。

    Raises:
        无。
    """

    pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
    docling_sha = hashlib.sha256(docling_json_bytes).hexdigest()
    return hashlib.sha256(f"{pdf_sha}|{docling_sha}".encode("utf-8")).hexdigest()


def update_cn_staging_source_document(
    *,
    source_repository: SourceDocumentRepositoryProtocol,
    ticker: str,
    document_id: str,
    internal_document_id: str,
    form_type: str,
    primary_document: str,
    file_entries: list[JsonObject],
    candidate: CnReportCandidate,
    profile: CnCompanyProfile,
    pdf_sha256: str | None,
    remote_fingerprint: str,
    previous_meta_exists: bool,
) -> None:
    """写入 CN/HK 下载中间态 source meta。

    Args:
        source_repository: source 文档仓储。
        ticker: ticker。
        document_id: 文档 ID。
        internal_document_id: 内部文档 ID。
        form_type: form type。
        primary_document: 当前主文件名。
        file_entries: 已落盘文件条目。
        candidate: 远端候选。
        profile: 公司基础元数据。
        pdf_sha256: PDF SHA-256；PDF 未落盘时为 ``None``。
        remote_fingerprint: 远端 fingerprint。
        previous_meta_exists: source document 是否已经存在。

    Returns:
        无。

    Raises:
        OSError: 仓储写入失败时抛出。
    """

    meta = _build_base_meta(
        ticker=ticker,
        document_id=document_id,
        internal_document_id=internal_document_id,
        form_type=form_type,
        candidate=candidate,
        profile=profile,
        ingest_complete=False,
    )
    meta["download_version"] = CN_PIPELINE_DOWNLOAD_VERSION
    meta["remote_fingerprint"] = remote_fingerprint
    meta["staging_remote_fingerprint"] = remote_fingerprint
    meta["staging_pdf_sha256"] = pdf_sha256
    request = _build_upsert_request(
        ticker=ticker,
        document_id=document_id,
        internal_document_id=internal_document_id,
        form_type=form_type,
        primary_document=primary_document,
        file_entries=file_entries,
        meta=meta,
        previous_meta_exists=previous_meta_exists,
    )
    if previous_meta_exists:
        source_repository.update_source_document(request, source_kind=SourceKind.FILING)
    else:
        source_repository.create_source_document(request, source_kind=SourceKind.FILING)


def commit_cn_filing_source_document(
    *,
    source_repository: SourceDocumentRepositoryProtocol,
    processed_repository: ProcessedDocumentRepositoryProtocol,
    ticker: str,
    document_id: str,
    internal_document_id: str,
    form_type: str,
    primary_document: str,
    file_entries: list[JsonObject],
    candidate: CnReportCandidate,
    profile: CnCompanyProfile,
    pdf_sha256: str,
    remote_fingerprint: str,
    source_fingerprint: str,
    previous_completed_meta: JsonObject | None,
    source_meta_exists: bool,
) -> None:
    """提交 CN/HK filing source document 完成态。

    Args:
        source_repository: source 文档仓储。
        processed_repository: processed 文档仓储。
        ticker: ticker。
        document_id: 文档 ID。
        internal_document_id: 内部文档 ID。
        form_type: form type。
        primary_document: 必须指向 Docling JSON 文件。
        file_entries: PDF 与 Docling JSON 文件条目。
        candidate: 远端候选报告。
        profile: 公司基础元数据。
        pdf_sha256: PDF SHA-256。
        remote_fingerprint: 远端 fingerprint。
        source_fingerprint: 内容 fingerprint。
        previous_completed_meta: 写入前上一版完成态 source meta；不存在时为
            ``None``。中间态 staging meta 不应传入此字段。
        source_meta_exists: 当前 source meta 是否已经存在；仅用于选择
            create/update，不参与版本计算。

    Returns:
        无。

    Raises:
        OSError: 仓储写入失败时抛出。
        ValueError: ``primary_document`` 未指向 Docling JSON 时抛出。
    """

    if not primary_document.endswith("_docling.json"):
        raise ValueError("CN download 完成态 primary_document 必须指向 _docling.json")
    meta = _build_base_meta(
        ticker=ticker,
        document_id=document_id,
        internal_document_id=internal_document_id,
        form_type=form_type,
        candidate=candidate,
        profile=profile,
        ingest_complete=True,
        previous_completed_meta=previous_completed_meta,
    )
    meta["download_version"] = CN_PIPELINE_DOWNLOAD_VERSION
    meta["remote_fingerprint"] = remote_fingerprint
    meta["source_fingerprint"] = source_fingerprint
    meta["pdf_sha256"] = pdf_sha256
    meta["staging_remote_fingerprint"] = None
    meta["staging_pdf_sha256"] = None
    meta["document_version"] = _resolve_document_version(previous_completed_meta, source_fingerprint)
    request = _build_upsert_request(
        ticker=ticker,
        document_id=document_id,
        internal_document_id=internal_document_id,
        form_type=form_type,
        primary_document=primary_document,
        file_entries=file_entries,
        meta=meta,
        previous_meta_exists=source_meta_exists,
    )
    if source_meta_exists:
        source_repository.update_source_document(request, source_kind=SourceKind.FILING)
    else:
        source_repository.create_source_document(request, source_kind=SourceKind.FILING)
    if _should_mark_processed_reprocess_required(
        processed_repository=processed_repository,
        ticker=ticker,
        document_id=document_id,
        previous_meta=previous_completed_meta,
        source_fingerprint=source_fingerprint,
    ):
        processed_repository.mark_processed_reprocess_required(ticker, document_id, True)


def _build_base_meta(
    *,
    ticker: str,
    document_id: str,
    internal_document_id: str,
    form_type: str,
    candidate: CnReportCandidate,
    profile: CnCompanyProfile,
    ingest_complete: bool,
    previous_completed_meta: JsonObject | None = None,
) -> JsonObject:
    """构建 CN/HK source meta 公共字段。"""

    company_id = ticker_to_company_id(normalize_ticker(ticker))
    now = now_iso8601()
    return {
        "document_id": document_id,
        "internal_document_id": internal_document_id,
        "ingest_method": _INGEST_METHOD_DOWNLOAD,
        "ticker": ticker,
        "company_id": company_id,
        "provider_company_id": profile.company_id,
        "company_name": profile.company_name,
        "form_type": form_type,
        "fiscal_year": candidate.fiscal_year,
        "fiscal_period": candidate.fiscal_period,
        "fiscal_year_source": _FISCAL_YEAR_SOURCE,
        "report_date_source": _REPORT_DATE_SOURCE,
        "report_kind": candidate.fiscal_period,
        "report_date": None,
        "filing_date": candidate.filing_date,
        "first_ingested_at": _preserve_text_meta(previous_completed_meta, "first_ingested_at", now),
        "ingest_complete": ingest_complete,
        "is_deleted": False,
        "deleted_at": None,
        "source_provider": candidate.provider,
        "source_id": candidate.source_id,
        "source_url": candidate.source_url,
        "source_language": candidate.language,
        "source_title": candidate.title,
        "amended": candidate.amended,
        "created_at": _preserve_text_meta(previous_completed_meta, "created_at", now),
        "updated_at": now,
        "has_xbrl": False,
    }


def _preserve_text_meta(previous_meta: JsonObject | None, key: str, fallback: str) -> str:
    """从上一版完成态 meta 保留文本字段。

    Args:
        previous_meta: 上一版完成态 meta；不存在时为 ``None``。
        key: 字段名。
        fallback: 字段缺失时使用的新值。

    Returns:
        保留后的文本字段值。

    Raises:
        无。
    """

    if previous_meta is None:
        return fallback
    value = previous_meta.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _build_upsert_request(
    *,
    ticker: str,
    document_id: str,
    internal_document_id: str,
    form_type: str,
    primary_document: str,
    file_entries: list[JsonObject],
    meta: JsonObject,
    previous_meta_exists: bool,
) -> FilingCreateRequest | FilingUpdateRequest:
    """构建 source document create/update 请求。"""

    kwargs = {
        "ticker": ticker,
        "document_id": document_id,
        "internal_document_id": internal_document_id,
        "form_type": form_type,
        "primary_document": primary_document,
        "file_entries": file_entries,
        "meta": meta,
    }
    if previous_meta_exists:
        return FilingUpdateRequest(**kwargs)
    return FilingCreateRequest(**kwargs)


def _resolve_document_version(previous_meta: JsonObject | None, source_fingerprint: str) -> str:
    """按 source fingerprint 计算文档版本。"""

    if previous_meta is None:
        return "v1"
    previous_fingerprint = str(previous_meta.get("source_fingerprint") or "").strip()
    previous_version = str(previous_meta.get("document_version") or "v1").strip() or "v1"
    if previous_fingerprint == source_fingerprint:
        return previous_version
    if previous_version.startswith("v") and previous_version[1:].isdigit():
        return f"v{int(previous_version[1:]) + 1}"
    return "v2"


def _should_mark_processed_reprocess_required(
    *,
    processed_repository: ProcessedDocumentRepositoryProtocol,
    ticker: str,
    document_id: str,
    previous_meta: JsonObject | None,
    source_fingerprint: str,
) -> bool:
    """判断 commit 后是否需要标记 processed 重处理。"""

    try:
        processed_repository.get_processed_meta(ticker, document_id)
    except FileNotFoundError:
        return False
    if previous_meta is None:
        return True
    previous_fingerprint = str(previous_meta.get("source_fingerprint") or "").strip()
    return bool(previous_fingerprint) and previous_fingerprint != source_fingerprint


__all__ = [
    "JsonObject",
    "JsonValue",
    "build_cn_file_entry",
    "build_content_fingerprint",
    "build_remote_fingerprint",
    "commit_cn_filing_source_document",
    "update_cn_staging_source_document",
]
