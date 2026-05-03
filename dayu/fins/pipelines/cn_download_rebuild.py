"""CN/HK 下载本地重建工作流。

本模块只基于已经落盘的 source meta 与文件条目重建 ``meta.json`` /
``filing_manifest.json``，不访问巨潮、披露易或 Docling。文档存取统一经
``dayu.fins.storage`` 仓储协议完成。
"""

from __future__ import annotations

import hashlib
import time

from dayu.fins.domain.document_models import FilingUpdateRequest, now_iso8601
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.cn_download_models import (
    CN_PIPELINE_DOWNLOAD_VERSION,
    CnFiscalPeriod,
    CnMarketKind,
)
from dayu.fins.pipelines.cn_download_protocols import CnDownloadWorkflowHost
from dayu.fins.pipelines.cn_form_utils import (
    PeriodDownloadWindow,
    resolve_period_windows,
    resolve_target_periods,
)

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]

_INGEST_METHOD_DOWNLOAD = "download"
_DOCLING_SUFFIX = "_docling.json"
_PDF_SUFFIX = ".pdf"


def rebuild_cn_download_artifacts(
    *,
    host: CnDownloadWorkflowHost,
    ticker: str,
    market: CnMarketKind,
    form_type: str | None,
    start_date: str | None,
    end_date: str | None,
    overwrite: bool,
    pipeline_name: str,
) -> JsonObject:
    """基于本地 CN/HK 下载结果重建 source meta 与 manifest。

    Args:
        host: CN/HK 下载 workflow 宿主协议。
        ticker: 已归一化 ticker。
        market: 已归一化市场。
        form_type: 可选 form 输入。
        start_date: 可选窗口起点。
        end_date: 可选窗口终点。
        overwrite: 是否覆盖；rebuild 不下载远端文件，仅回填 filters。
        pipeline_name: pipeline 名称。

    Returns:
        download 结果字典。

    Raises:
        ValueError: form/date 参数非法时抛出。
        OSError: 仓储写入失败时抛出。
    """

    periods = resolve_target_periods(form_type, "HK" if market == "HK" else "CN")
    period_windows = resolve_period_windows(
        target_periods=periods.target_periods,
        start_date=start_date,
        end_date=end_date,
    )
    started_at = time.perf_counter()
    filings: list[JsonObject] = []
    document_ids = host.source_repository.list_source_document_ids(ticker, SourceKind.FILING)
    for document_id in document_ids:
        previous_meta = host.source_repository.get_source_meta(ticker, document_id, SourceKind.FILING)
        meta = dict(previous_meta)
        if not _should_rebuild_meta(meta=meta, period_windows=period_windows):
            continue
        filings.append(
            _rebuild_single_cn_download_document(
                host=host,
                ticker=ticker,
                document_id=document_id,
                previous_meta=meta,
            )
        )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    warnings: list[str] = []
    if not filings:
        warnings.append("未匹配到可重建的已下载 CN/HK filings")
    form_values: list[JsonValue] = [period for period in periods.target_periods]
    warning_values: list[JsonValue] = [warning for warning in warnings]
    note_values: list[JsonValue] = [note for note in periods.notes]
    filing_values: list[JsonValue] = [filing for filing in filings]
    result: JsonObject = {
        "pipeline": pipeline_name,
        "action": "download",
        "status": "ok",
        "ticker": ticker,
        "company_info": {},
        "filters": {
            "forms": form_values,
            "start_dates": {item.fiscal_period: item.start_date for item in period_windows},
            "end_date": period_windows[0].end_date if period_windows else end_date,
            "overwrite": overwrite,
            "rebuild": True,
        },
        "warnings": warning_values,
        "notes": note_values,
        "filings": filing_values,
        "summary": _build_rebuild_summary(filings=filings, elapsed_ms=elapsed_ms),
    }
    return result


def _should_rebuild_meta(
    *,
    meta: JsonObject,
    period_windows: tuple[PeriodDownloadWindow, ...],
) -> bool:
    """判断 source meta 是否属于本次 CN/HK rebuild 范围。"""

    if str(meta.get("ingest_method") or "").strip().lower() != _INGEST_METHOD_DOWNLOAD:
        return False
    if bool(meta.get("is_deleted", False)):
        return False
    period = _optional_period(meta.get("fiscal_period"))
    if period is None:
        return False
    matched_window = next((item for item in period_windows if item.fiscal_period == period), None)
    if matched_window is None:
        return False
    filing_date = _optional_text(meta.get("filing_date"))
    if filing_date is None:
        return False
    return matched_window.start_date <= filing_date <= matched_window.end_date


def _rebuild_single_cn_download_document(
    *,
    host: CnDownloadWorkflowHost,
    ticker: str,
    document_id: str,
    previous_meta: JsonObject,
) -> JsonObject:
    """重建单个 CN/HK 本地下载文档。"""

    internal_document_id = _required_text(previous_meta, "internal_document_id", document_id)
    form_type = _required_text(previous_meta, "form_type", "")
    filing_date = _optional_text(previous_meta.get("filing_date"))
    report_date = _optional_text(previous_meta.get("report_date"))
    file_entries = _extract_file_entries(previous_meta)
    if not form_type:
        return _failed_rebuild_result(
            document_id=document_id,
            internal_document_id=internal_document_id,
            reason_code="missing_form_type",
            reason_message="重建失败：meta.json 缺少 form_type",
        )
    if not _has_docling_file(file_entries):
        return _failed_rebuild_result(
            document_id=document_id,
            internal_document_id=internal_document_id,
            form_type=form_type,
            filing_date=filing_date,
            report_date=report_date,
            reason_code="missing_docling_json",
            reason_message="重建失败：CN/HK 下载完成态缺少 Docling JSON",
        )
    if not _has_pdf_file(file_entries):
        return _failed_rebuild_result(
            document_id=document_id,
            internal_document_id=internal_document_id,
            form_type=form_type,
            filing_date=filing_date,
            report_date=report_date,
            reason_code="missing_pdf",
            reason_message="重建失败：CN/HK 下载完成态缺少 PDF",
        )
    primary_document = _resolve_primary_document(previous_meta=previous_meta, file_entries=file_entries)
    source_fingerprint = _resolve_source_fingerprint(previous_meta=previous_meta, file_entries=file_entries)
    meta_payload = dict(previous_meta)
    file_values: list[JsonValue] = [item for item in file_entries]
    update_payload: JsonObject = {
        "document_id": document_id,
        "internal_document_id": internal_document_id,
        "ingest_method": _INGEST_METHOD_DOWNLOAD,
        "ticker": ticker,
        "form_type": form_type,
        "primary_document": primary_document,
        "files": file_values,
        "ingest_complete": True,
        "download_version": CN_PIPELINE_DOWNLOAD_VERSION,
        "source_fingerprint": source_fingerprint,
        "staging_remote_fingerprint": None,
        "staging_pdf_sha256": None,
        "updated_at": now_iso8601(),
    }
    meta_payload.update(update_payload)
    host.source_repository.update_source_document(
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
    host.source_repository.replace_source_meta(
        ticker=ticker,
        document_id=document_id,
        source_kind=SourceKind.FILING,
        meta=meta_payload,
    )
    return {
        "document_id": document_id,
        "internal_document_id": internal_document_id,
        "status": "downloaded",
        "form_type": form_type,
        "filing_date": filing_date,
        "report_date": report_date,
        "downloaded_files": 0,
        "skipped_files": len(file_entries),
        "failed_files": [],
        "has_xbrl": False,
        "rebuild": True,
    }


def _extract_file_entries(meta: JsonObject) -> list[JsonObject]:
    """从 source meta 提取文件条目。"""

    raw_files = meta.get("files")
    if not isinstance(raw_files, list):
        return []
    return [dict(item) for item in raw_files if isinstance(item, dict)]


def _resolve_primary_document(*, previous_meta: JsonObject, file_entries: list[JsonObject]) -> str:
    """解析 CN/HK rebuild 完成态主文件名。"""

    raw_primary = _optional_text(previous_meta.get("primary_document"))
    if raw_primary is not None and raw_primary.endswith(_DOCLING_SUFFIX):
        return raw_primary
    for item in file_entries:
        name = _optional_text(item.get("name"))
        if name is not None and name.endswith(_DOCLING_SUFFIX):
            return name
    return raw_primary or _optional_text(file_entries[0].get("name")) or ""


def _resolve_source_fingerprint(*, previous_meta: JsonObject, file_entries: list[JsonObject]) -> str:
    """解析或重建 source fingerprint。"""

    previous_fingerprint = _optional_text(previous_meta.get("source_fingerprint"))
    if previous_fingerprint is not None:
        return previous_fingerprint
    pdf_sha = _find_entry_sha(file_entries, _PDF_SUFFIX)
    docling_sha = _find_entry_sha(file_entries, _DOCLING_SUFFIX)
    if pdf_sha is None or docling_sha is None:
        return ""
    return hashlib.sha256(f"{pdf_sha}|{docling_sha}".encode("utf-8")).hexdigest()


def _find_entry_sha(file_entries: list[JsonObject], suffix: str) -> str | None:
    """按文件后缀查找文件 SHA-256。"""

    for item in file_entries:
        name = _optional_text(item.get("name"))
        sha = _optional_text(item.get("sha256"))
        if name is not None and name.endswith(suffix) and sha is not None:
            return sha
    return None


def _has_docling_file(file_entries: list[JsonObject]) -> bool:
    """判断文件条目是否包含 Docling JSON。"""

    return any((_optional_text(item.get("name")) or "").endswith(_DOCLING_SUFFIX) for item in file_entries)


def _has_pdf_file(file_entries: list[JsonObject]) -> bool:
    """判断文件条目是否包含 PDF。"""

    return any((_optional_text(item.get("name")) or "").endswith(_PDF_SUFFIX) for item in file_entries)


def _failed_rebuild_result(
    *,
    document_id: str,
    internal_document_id: str,
    reason_code: str,
    reason_message: str,
    form_type: str | None = None,
    filing_date: str | None = None,
    report_date: str | None = None,
) -> JsonObject:
    """构建单文档 rebuild 失败结果。"""

    return {
        "document_id": document_id,
        "internal_document_id": internal_document_id,
        "status": "failed",
        "form_type": form_type,
        "filing_date": filing_date,
        "report_date": report_date,
        "downloaded_files": 0,
        "skipped_files": 0,
        "failed_files": [],
        "has_xbrl": False,
        "reason_code": reason_code,
        "reason_message": reason_message,
        "rebuild": True,
    }


def _build_rebuild_summary(*, filings: list[JsonObject], elapsed_ms: int) -> JsonObject:
    """构建 CN/HK rebuild summary。"""

    return {
        "total": len(filings),
        "downloaded": sum(1 for item in filings if item.get("status") == "downloaded"),
        "skipped": sum(1 for item in filings if item.get("status") == "skipped"),
        "failed": sum(1 for item in filings if item.get("status") == "failed"),
        "elapsed_ms": elapsed_ms,
        "reused_downloads": 0,
        "converted": 0,
    }


def _optional_period(value: JsonValue | None) -> CnFiscalPeriod | None:
    """把 JSON 字段收窄为 CN/HK 财期。"""

    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if normalized == "FY":
        return "FY"
    if normalized == "H1":
        return "H1"
    if normalized == "Q1":
        return "Q1"
    if normalized == "Q2":
        return "Q2"
    if normalized == "Q3":
        return "Q3"
    return None


def _optional_text(value: JsonValue | None) -> str | None:
    """把 JSON 字段转换为非空字符串。"""

    if not isinstance(value, (str, int, float, bool)):
        return None
    text = str(value).strip()
    return text or None


def _required_text(meta: JsonObject, key: str, fallback: str) -> str:
    """读取必需文本字段，缺失时使用 fallback。"""

    return _optional_text(meta.get(key)) or fallback


__all__ = ["rebuild_cn_download_artifacts"]
