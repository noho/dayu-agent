"""CN/HK 单份财报下载阶段机。

阶段机负责单个 :class:`CnReportCandidate` 的 skip、PDF 下载 / 复用、Docling
转换 / 复用以及 source commit。所有持久化动作都经 ``dayu.fins.storage`` 的
窄仓储协议完成；本模块不直接拼 workspace 路径。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from io import BytesIO
from pathlib import Path

from dayu.contracts.cancellation import CancelledError
from dayu.fins.domain.document_models import FileObjectMeta, SourceHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.cn_download_models import (
    CN_PIPELINE_DOWNLOAD_VERSION,
    CnCompanyProfile,
    CnReportCandidate,
)
from dayu.fins.pipelines.cn_download_protocols import (
    CnReportDiscoveryClientProtocol,
)
from dayu.fins.pipelines.cn_download_source_upsert import (
    JsonObject,
    JsonValue,
    build_cn_file_entry,
    build_content_fingerprint,
    build_remote_fingerprint,
    commit_cn_filing_source_document,
    update_cn_staging_source_document,
)
from dayu.fins.pipelines.cn_download_staging import has_blob_file, inspect_staged_blobs
from dayu.fins.pipelines.docling_upload_service import build_cn_filing_ids
from dayu.fins.pipelines.download_events import DownloadEvent, DownloadEventType
from dayu.fins.storage import (
    DocumentBlobRepositoryProtocol,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)
from dayu.log import Log

_PDF_CONTENT_TYPE = "application/pdf"
_JSON_CONTENT_TYPE = "application/json"
_SOURCE_LABEL_ORIGINAL = "original"
_SOURCE_LABEL_DOCLING = "docling"


class CnDownloadFilingError(RuntimeError):
    """CN/HK 单 filing 下载失败。"""


async def run_cn_download_single_filing_stream(
    *,
    source_repository: SourceDocumentRepositoryProtocol,
    blob_repository: DocumentBlobRepositoryProtocol,
    processed_repository: ProcessedDocumentRepositoryProtocol,
    discovery_client: CnReportDiscoveryClientProtocol,
    convert_pdf_to_docling_json: Callable[[bytes, str], bytes],
    ticker: str,
    profile: CnCompanyProfile,
    candidate: CnReportCandidate,
    overwrite: bool,
    cancel_checker: Callable[[], bool] | None,
    module: str,
) -> AsyncIterator[DownloadEvent]:
    """执行单个 CN/HK filing 下载阶段机。

    Args:
        source_repository: source 文档仓储。
        blob_repository: 文件对象仓储。
        processed_repository: processed 文档仓储。
        discovery_client: 当前市场 downloader。
        convert_pdf_to_docling_json: PDF -> Docling JSON 转换函数。
        ticker: 已归一化 ticker。
        profile: 公司基础元数据。
        candidate: 远端候选报告。
        overwrite: 是否强制覆盖；为 ``True`` 时禁止复用和 skip。
        cancel_checker: 可选取消检查函数。
        module: 日志模块名。

    Yields:
        单 filing 的文件级与终态下载事件。``FILING_STARTED`` 由上层 workflow
        统一发出，本函数只发后续事件。

    Raises:
        CnDownloadFilingError: 仓储、下载或转换失败时抛出。
        CancelledError: 取消检查命中时抛出。
    """

    _raise_if_cancelled(module=module, ticker=ticker, document_id="", cancel_checker=cancel_checker)
    document_id, internal_document_id = build_cn_filing_ids(
        ticker=ticker,
        form_type=candidate.fiscal_period,
        fiscal_year=candidate.fiscal_year,
        fiscal_period=candidate.fiscal_period,
        amended=candidate.amended,
    )
    pdf_filename = f"{document_id}.pdf"
    docling_filename = f"{document_id}_docling.json"
    previous_meta = _safe_get_source_meta(
        source_repository=source_repository,
        ticker=ticker,
        document_id=document_id,
    )
    previous_completed_meta = _resolve_previous_completed_meta(
        previous_meta=previous_meta,
        overwrite=overwrite,
    )
    source_meta_exists = previous_meta is not None
    remote_fingerprint = build_remote_fingerprint(candidate)
    skip_result = _resolve_fast_skip_result(
        previous_meta=previous_meta,
        remote_fingerprint=remote_fingerprint,
        overwrite=overwrite,
    )
    if skip_result is not None:
        yield DownloadEvent(
            event_type=DownloadEventType.FILING_COMPLETED,
            ticker=ticker,
            document_id=document_id,
            payload={"filing_result": skip_result, **skip_result},
        )
        return

    if _should_reset_before_download(previous_meta=previous_meta, remote_fingerprint=remote_fingerprint, overwrite=overwrite):
        source_repository.reset_source_document(ticker, document_id, SourceKind.FILING)
        previous_meta = None
        previous_completed_meta = None
        source_meta_exists = False

    if previous_meta is None:
        update_cn_staging_source_document(
            source_repository=source_repository,
            ticker=ticker,
            document_id=document_id,
            internal_document_id=internal_document_id,
            form_type=candidate.fiscal_period,
            primary_document=pdf_filename,
            file_entries=[],
            candidate=candidate,
            profile=profile,
            pdf_sha256=None,
            remote_fingerprint=remote_fingerprint,
            previous_meta_exists=False,
        )
        source_meta_exists = True
    handle = source_repository.get_source_handle(ticker, document_id, SourceKind.FILING)

    reusable_pdf = _resolve_reusable_pdf(
        blob_repository=blob_repository,
        handle=handle,
        pdf_filename=pdf_filename,
        docling_filename=docling_filename,
        previous_meta=previous_meta,
        remote_fingerprint=remote_fingerprint,
        overwrite=overwrite,
    )
    if reusable_pdf is None:
        try:
            asset = discovery_client.download_report_pdf(candidate)
        except CancelledError:
            raise
        except Exception as exc:
            failed = _build_filing_result(
                document_id=document_id,
                status="failed",
                candidate=candidate,
                reason_code="pdf_download_failed",
                reason_message=str(exc),
                downloaded_files=0,
                skipped_files=0,
            )
            yield DownloadEvent(
                event_type=DownloadEventType.FILING_FAILED,
                ticker=ticker,
                document_id=document_id,
                payload={"filing_result": failed, **failed},
            )
            return
        pdf_path = asset.pdf_path
        pdf_bytes = pdf_path.read_bytes()
        _unlink_temp_pdf(pdf_path, module=module)
        pdf_sha256 = asset.sha256
        reused_pdf = False
    else:
        pdf_bytes = reusable_pdf
        pdf_sha256 = _read_required_text(previous_meta, "staging_pdf_sha256")
        reused_pdf = True

    if _can_skip_by_pdf_sha(
        previous_meta=previous_meta,
        overwrite=overwrite,
        pdf_sha256=pdf_sha256,
        blob_repository=blob_repository,
        handle=handle,
        docling_filename=docling_filename,
    ):
        commit_cn_filing_source_document(
            source_repository=source_repository,
            processed_repository=processed_repository,
            ticker=ticker,
            document_id=document_id,
            internal_document_id=internal_document_id,
            form_type=candidate.fiscal_period,
            primary_document=_read_required_text(previous_meta, "primary_document"),
            file_entries=_read_file_entries(previous_meta),
            candidate=candidate,
            profile=profile,
            pdf_sha256=pdf_sha256,
            remote_fingerprint=remote_fingerprint,
            source_fingerprint=_read_required_text(previous_meta, "source_fingerprint"),
            previous_completed_meta=previous_completed_meta,
            source_meta_exists=True,
        )
        skipped = _build_filing_result(
            document_id=document_id,
            status="skipped",
            candidate=candidate,
            reason_code="pdf_sha256_matched",
            reason_message="PDF 内容与完成态一致且 Docling JSON 存在，跳过重新处理",
            downloaded_files=0,
            skipped_files=2,
        )
        yield DownloadEvent(
            event_type=DownloadEventType.FILING_COMPLETED,
            ticker=ticker,
            document_id=document_id,
            payload={"filing_result": skipped, **skipped},
        )
        return

    if previous_meta is not None and previous_meta.get("ingest_complete") is True and not overwrite:
        source_repository.reset_source_document(ticker, document_id, SourceKind.FILING)
        update_cn_staging_source_document(
            source_repository=source_repository,
            ticker=ticker,
            document_id=document_id,
            internal_document_id=internal_document_id,
            form_type=candidate.fiscal_period,
            primary_document=pdf_filename,
            file_entries=[],
            candidate=candidate,
            profile=profile,
            pdf_sha256=None,
            remote_fingerprint=remote_fingerprint,
            previous_meta_exists=False,
        )
        previous_meta = None
        source_meta_exists = True
        handle = source_repository.get_source_handle(ticker, document_id, SourceKind.FILING)

    if reused_pdf:
        pdf_entry_meta = _find_file_meta(blob_repository=blob_repository, handle=handle, filename=pdf_filename)
    else:
        pdf_entry_meta = blob_repository.store_file(
            handle,
            pdf_filename,
            BytesIO(pdf_bytes),
            content_type=_PDF_CONTENT_TYPE,
            metadata={"source": _SOURCE_LABEL_ORIGINAL},
        )
    pdf_entry = build_cn_file_entry(
        filename=pdf_filename,
        file_meta=pdf_entry_meta,
        source_label=_SOURCE_LABEL_ORIGINAL,
    )
    reusable_docling = _resolve_reusable_docling(
        blob_repository=blob_repository,
        handle=handle,
        docling_filename=docling_filename,
        previous_meta=previous_meta,
        remote_fingerprint=remote_fingerprint,
        pdf_sha256=pdf_sha256,
        overwrite=overwrite,
    )
    staged_docling_meta: FileObjectMeta | None = None
    staged_docling_entry: JsonObject | None = None
    if reusable_docling is not None:
        try:
            staged_docling_meta = _find_file_meta(
                blob_repository=blob_repository,
                handle=handle,
                filename=docling_filename,
            )
        except FileNotFoundError:
            staged_docling_meta = None
        if staged_docling_meta is not None:
            staged_docling_entry = build_cn_file_entry(
                filename=docling_filename,
                file_meta=staged_docling_meta,
                source_label=_SOURCE_LABEL_DOCLING,
            )
    staging_entries = [pdf_entry]
    if staged_docling_entry is not None:
        staging_entries.append(staged_docling_entry)
    update_cn_staging_source_document(
        source_repository=source_repository,
        ticker=ticker,
        document_id=document_id,
        internal_document_id=internal_document_id,
        form_type=candidate.fiscal_period,
        primary_document=pdf_filename,
        file_entries=staging_entries,
        candidate=candidate,
        profile=profile,
        pdf_sha256=pdf_sha256,
        remote_fingerprint=remote_fingerprint,
        previous_meta_exists=True,
    )
    yield DownloadEvent(
        event_type=DownloadEventType.FILE_DOWNLOADED,
        ticker=ticker,
        document_id=document_id,
        payload={
            "name": pdf_filename,
            "stage": "pdf_downloaded",
            "status": "skipped" if reused_pdf else "downloaded",
            "reused": reused_pdf,
            "reason_code": "local_pdf_reused" if reused_pdf else None,
        },
    )

    _raise_if_cancelled(module=module, ticker=ticker, document_id=document_id, cancel_checker=cancel_checker)
    if reusable_docling is None:
        try:
            Log.info(
                f"开始 Docling 转换: ticker={ticker} document_id={document_id} "
                f"form={candidate.fiscal_period} filing_date={candidate.filing_date} "
                f"source_file={pdf_filename} reused_pdf={reused_pdf}",
                module=module,
            )
            docling_json_bytes = convert_pdf_to_docling_json(pdf_bytes, pdf_filename)
        except Exception as exc:
            failed = _build_filing_result(
                document_id=document_id,
                status="failed",
                candidate=candidate,
                reason_code="docling_convert_failed",
                reason_message=str(exc),
                downloaded_files=0 if reused_pdf else 1,
                skipped_files=1 if reused_pdf else 0,
            )
            yield DownloadEvent(
                event_type=DownloadEventType.FILING_FAILED,
                ticker=ticker,
                document_id=document_id,
                payload={"filing_result": failed, **failed},
            )
            return
        docling_meta = blob_repository.store_file(
            handle,
            docling_filename,
            BytesIO(docling_json_bytes),
            content_type=_JSON_CONTENT_TYPE,
            metadata={"source": _SOURCE_LABEL_DOCLING, "pdf_sha256": pdf_sha256},
        )
        reused_docling = False
        converted = True
    else:
        docling_json_bytes = reusable_docling
        docling_meta = (
            staged_docling_meta
            if staged_docling_meta is not None
            else blob_repository.store_file(
                handle,
                docling_filename,
                BytesIO(docling_json_bytes),
                content_type=_JSON_CONTENT_TYPE,
                metadata={"source": _SOURCE_LABEL_DOCLING, "pdf_sha256": pdf_sha256},
            )
        )
        reused_docling = True
        converted = False
    docling_entry = build_cn_file_entry(
        filename=docling_filename,
        file_meta=docling_meta,
        source_label=_SOURCE_LABEL_DOCLING,
    )
    source_fingerprint = build_content_fingerprint(
        pdf_bytes=pdf_bytes,
        docling_json_bytes=docling_json_bytes,
    )
    commit_cn_filing_source_document(
        source_repository=source_repository,
        processed_repository=processed_repository,
        ticker=ticker,
        document_id=document_id,
        internal_document_id=internal_document_id,
        form_type=candidate.fiscal_period,
        primary_document=docling_filename,
        file_entries=[pdf_entry, docling_entry],
        candidate=candidate,
        profile=profile,
        pdf_sha256=pdf_sha256,
        remote_fingerprint=remote_fingerprint,
        source_fingerprint=source_fingerprint,
        previous_completed_meta=previous_completed_meta,
        source_meta_exists=source_meta_exists,
    )
    downloaded = _build_filing_result(
        document_id=document_id,
        status="downloaded",
        candidate=candidate,
        reason_code="download_committed",
        reason_message="PDF 与 Docling JSON 已完成落盘并提交 source meta",
        downloaded_files=(0 if reused_pdf else 1) + (0 if reused_docling else 1),
        skipped_files=(1 if reused_pdf else 0) + (1 if reused_docling else 0),
    )
    downloaded["reused_pdf"] = reused_pdf
    downloaded["reused_docling"] = reused_docling
    downloaded["converted"] = converted
    yield DownloadEvent(
        event_type=DownloadEventType.FILING_COMPLETED,
        ticker=ticker,
        document_id=document_id,
        payload={"filing_result": downloaded, **downloaded},
    )


def _safe_get_source_meta(
    *,
    source_repository: SourceDocumentRepositoryProtocol,
    ticker: str,
    document_id: str,
) -> JsonObject | None:
    """安全读取 source meta。"""

    try:
        meta = source_repository.get_source_meta(ticker, document_id, SourceKind.FILING)
    except FileNotFoundError:
        return None
    return {str(key): _coerce_json_value(value) for key, value in meta.items()}


def _resolve_fast_skip_result(
    *,
    previous_meta: JsonObject | None,
    remote_fingerprint: str,
    overwrite: bool,
) -> JsonObject | None:
    """判断是否可在下载 PDF 前 fast skip。"""

    if overwrite or previous_meta is None:
        return None
    if previous_meta.get("ingest_complete") is not True:
        return None
    if previous_meta.get("download_version") != CN_PIPELINE_DOWNLOAD_VERSION:
        return None
    if previous_meta.get("remote_fingerprint") != remote_fingerprint:
        return None
    return {
        "document_id": str(previous_meta.get("document_id") or ""),
        "status": "skipped",
        "form_type": str(previous_meta.get("form_type") or ""),
        "filing_date": str(previous_meta.get("filing_date") or ""),
        "report_date": None,
        "downloaded_files": 0,
        "skipped_files": 2,
        "reason_code": "remote_fingerprint_matched",
        "reason_message": "远端 fingerprint 与本地完成态一致，跳过下载",
        "skip_reason": "remote_fingerprint_matched",
    }


def _resolve_previous_completed_meta(
    *,
    previous_meta: JsonObject | None,
    overwrite: bool,
) -> JsonObject | None:
    """解析可用于版本计算的上一版完成态 meta。

    Args:
        previous_meta: 当前 source meta。
        overwrite: 是否强制覆盖。

    Returns:
        可用于版本计算和审计字段保留的完成态 meta；不存在时返回 ``None``。

    Raises:
        无。
    """

    if overwrite or previous_meta is None:
        return None
    if previous_meta.get("ingest_complete") is not True:
        return None
    return previous_meta


def _should_reset_before_download(
    *,
    previous_meta: JsonObject | None,
    remote_fingerprint: str,
    overwrite: bool,
) -> bool:
    """判断进入下载分支前是否应 reset 单个 source document。"""

    if overwrite or previous_meta is None:
        return False
    if previous_meta.get("ingest_complete") is True:
        return False
    return previous_meta.get("staging_remote_fingerprint") != remote_fingerprint


def _resolve_reusable_pdf(
    *,
    blob_repository: DocumentBlobRepositoryProtocol,
    handle: SourceHandle,
    pdf_filename: str,
    docling_filename: str,
    previous_meta: JsonObject | None,
    remote_fingerprint: str,
    overwrite: bool,
) -> bytes | None:
    """判断中间态 PDF 是否可复用。"""

    if overwrite or previous_meta is None:
        return None
    if previous_meta.get("ingest_complete") is not False:
        return None
    if previous_meta.get("staging_remote_fingerprint") != remote_fingerprint:
        return None
    expected_sha = _optional_text(previous_meta.get("staging_pdf_sha256"))
    staged = inspect_staged_blobs(
        blob_repository=blob_repository,
        handle=handle,
        pdf_filename=pdf_filename,
        docling_filename=docling_filename,
        expected_pdf_sha256=expected_sha,
    )
    return staged.pdf_bytes if staged.pdf_sha256_matched else None


def _resolve_reusable_docling(
    *,
    blob_repository: DocumentBlobRepositoryProtocol,
    handle: SourceHandle,
    docling_filename: str,
    previous_meta: JsonObject | None,
    remote_fingerprint: str,
    pdf_sha256: str,
    overwrite: bool,
) -> bytes | None:
    """判断中间态 Docling JSON 是否可复用。

    Args:
        blob_repository: blob 仓储。
        handle: source document 句柄。
        docling_filename: Docling JSON 文件名。
        previous_meta: 当前 source meta。
        remote_fingerprint: 当前候选远端 fingerprint。
        pdf_sha256: 当前 PDF 字节 SHA-256。
        overwrite: 是否强制覆盖。

    Returns:
        可复用 Docling JSON 字节；不可复用时返回 ``None``。

    Raises:
        OSError: 底层 blob 读取失败时抛出。
    """

    if overwrite or previous_meta is None:
        return None
    if previous_meta.get("ingest_complete") is not False:
        return None
    if previous_meta.get("staging_remote_fingerprint") != remote_fingerprint:
        return None
    if previous_meta.get("staging_pdf_sha256") != pdf_sha256:
        return None
    try:
        return blob_repository.read_file_bytes(handle, docling_filename)
    except FileNotFoundError:
        return None


def _can_skip_by_pdf_sha(
    *,
    previous_meta: JsonObject | None,
    overwrite: bool,
    pdf_sha256: str,
    blob_repository: DocumentBlobRepositoryProtocol,
    handle: SourceHandle,
    docling_filename: str,
) -> bool:
    """判断完成态 PDF 内容未变时是否可跳过。"""

    if overwrite or previous_meta is None:
        return False
    if previous_meta.get("ingest_complete") is not True:
        return False
    if previous_meta.get("download_version") != CN_PIPELINE_DOWNLOAD_VERSION:
        return False
    if previous_meta.get("pdf_sha256") != pdf_sha256:
        return False
    return has_blob_file(blob_repository=blob_repository, handle=handle, filename=docling_filename)


def _find_file_meta(
    *,
    blob_repository: DocumentBlobRepositoryProtocol,
    handle: SourceHandle,
    filename: str,
) -> FileObjectMeta:
    """从 blob 仓储中查找指定文件元数据。"""

    for item in blob_repository.list_files(handle):
        if item.uri.rsplit("/", 1)[-1] == filename:
            return item
    raise FileNotFoundError(f"文件不存在: {filename}")


def _read_file_entries(meta: JsonObject | None) -> list[JsonObject]:
    """读取完成态 meta 中的文件条目。

    Args:
        meta: source meta。

    Returns:
        可传给 source upsert 的文件条目列表。

    Raises:
        CnDownloadFilingError: meta 缺失或 ``files`` 字段不是对象列表时抛出。
    """

    if meta is None:
        raise CnDownloadFilingError("缺少 source meta，无法读取 files")
    raw_files = meta.get("files")
    if not isinstance(raw_files, list):
        raise CnDownloadFilingError("source meta.files 必须为 list")
    entries: list[JsonObject] = []
    for raw_item in raw_files:
        if not isinstance(raw_item, dict):
            raise CnDownloadFilingError("source meta.files 条目必须为 object")
        entries.append({str(key): _coerce_json_value(value) for key, value in raw_item.items()})
    return entries


def _build_filing_result(
    *,
    document_id: str,
    status: str,
    candidate: CnReportCandidate,
    reason_code: str,
    reason_message: str,
    downloaded_files: int,
    skipped_files: int,
) -> JsonObject:
    """构建单 filing 结果 payload。"""

    payload: JsonObject = {
        "document_id": document_id,
        "status": status,
        "form_type": candidate.fiscal_period,
        "filing_date": candidate.filing_date,
        "report_date": None,
        "fiscal_year": candidate.fiscal_year,
        "fiscal_period": candidate.fiscal_period,
        "downloaded_files": downloaded_files,
        "skipped_files": skipped_files,
        "failed_files": [],
        "has_xbrl": False,
        "reason_code": reason_code,
        "reason_message": reason_message,
    }
    if status == "skipped":
        payload["skip_reason"] = reason_code
    return payload


def _read_required_text(meta: JsonObject | None, key: str) -> str:
    """读取必填文本 meta 字段。"""

    if meta is None:
        raise CnDownloadFilingError(f"缺少 source meta，无法读取 {key}")
    value = _optional_text(meta.get(key))
    if value is None:
        raise CnDownloadFilingError(f"source meta 缺少 {key}")
    return value


def _optional_text(value: JsonValue) -> str | None:
    """把值收窄为非空字符串。"""

    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coerce_json_value(value: JsonValue) -> JsonValue:
    """把仓储 meta 值收窄到 JSON 值。

    Args:
        value: 仓储 meta 中的单个值。

    Returns:
        JSON 值；非 JSON 类型按字符串保存。

    Raises:
        无。
    """

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _coerce_json_value(item) for key, item in value.items()}
    return str(value)


def _unlink_temp_pdf(path: Path, *, module: str) -> None:
    """删除 downloader 暂存 PDF。"""

    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        Log.warn(f"删除临时 PDF 失败: path={path} error={exc}", module=module)


def _raise_if_cancelled(
    *,
    module: str,
    ticker: str,
    document_id: str,
    cancel_checker: Callable[[], bool] | None,
) -> None:
    """在阶段边界检查取消请求。"""

    if cancel_checker is None or not cancel_checker():
        return
    Log.info(
        f"CN/HK 下载收到取消请求: ticker={ticker} document_id={document_id}",
        module=module,
    )
    raise CancelledError("操作已被取消")


__all__ = ["CnDownloadFilingError", "run_cn_download_single_filing_stream"]
