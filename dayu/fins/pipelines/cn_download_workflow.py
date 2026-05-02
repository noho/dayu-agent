"""CN/HK ticker 级下载主工作流。

本模块负责 ticker 归一化、form/window 解析、company meta 写入、候选发现、
overwrite ticker 级清理、单 filing 阶段机调度和 summary 聚合。单文件落盘细节由
``cn_download_filing_workflow`` 承担。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable

from dayu.contracts.cancellation import CancelledError
from dayu.fins.pipelines.cn_download_company_meta import upsert_company_meta_for_cn_download
from dayu.fins.pipelines.cn_download_filing_workflow import (
    CnDownloadFilingError,
    run_cn_download_single_filing_stream,
)
from dayu.fins.pipelines.cn_download_models import CnMarketKind, CnReportCandidate, CnReportQuery
from dayu.fins.pipelines.cn_download_protocols import (
    CnDownloadWorkflowHost,
    CnReportDiscoveryClientProtocol,
)
from dayu.fins.pipelines.cn_form_utils import resolve_target_periods, resolve_window
from dayu.fins.pipelines.docling_upload_service import build_cn_filing_ids
from dayu.fins.pipelines.download_events import DownloadEvent, DownloadEventType
from dayu.fins.ticker_normalization import try_normalize_ticker
from dayu.log import Log

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


async def run_cn_download_stream_impl(
    host: CnDownloadWorkflowHost,
    *,
    ticker: str,
    form_type: str | None,
    start_date: str | None,
    end_date: str | None,
    overwrite: bool,
    rebuild: bool,
    ticker_aliases: list[str] | None,
    cancel_checker: Callable[[], bool] | None,
    module: str,
    pipeline_name: str,
) -> AsyncIterator[DownloadEvent]:
    """执行 CN/HK ticker 级下载工作流。

    Args:
        host: workflow 所需宿主协议。
        ticker: 原始 ticker。
        form_type: 可选 form 输入。
        start_date: 可选窗口起点。
        end_date: 可选窗口终点。
        overwrite: 是否强制覆盖。
        rebuild: 是否执行 rebuild；CN/HK 当前不支持。
        ticker_aliases: 可选 ticker alias。
        cancel_checker: 可选取消检查函数。
        module: 日志模块名。
        pipeline_name: pipeline 名称。

    Yields:
        下载事件流。

    Raises:
        无。请求级错误会收口为 ``PIPELINE_COMPLETED.status="failed"``。
    """

    started_at = time.perf_counter()
    normalized = try_normalize_ticker(ticker)
    if normalized is None or normalized.market not in {"CN", "HK"}:
        result = _build_result(
            pipeline_name=pipeline_name,
            status="failed",
            ticker=ticker,
            reason_code="unsupported_market",
            message=f"CN/HK download 不支持 ticker={ticker!r}",
        )
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_COMPLETED,
            ticker=ticker,
            payload={"result": result},
        )
        return
    market = _coerce_market(normalized.market)
    normalized_ticker = normalized.canonical
    try:
        if rebuild:
            raise ValueError("CN/HK download 当前不支持 rebuild")
        periods = resolve_target_periods(form_type, market)
        window = resolve_window(start_date, end_date)
    except ValueError as exc:
        result = _build_result(
            pipeline_name=pipeline_name,
            status="failed",
            ticker=normalized_ticker,
            reason_code="invalid_download_request",
            message=str(exc),
        )
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_COMPLETED,
            ticker=normalized_ticker,
            payload={"result": result},
        )
        return

    yield DownloadEvent(
        event_type=DownloadEventType.PIPELINE_STARTED,
        ticker=normalized_ticker,
        payload={
            "form_type": form_type,
            "start_date": window.start_date,
            "end_date": window.end_date,
            "overwrite": overwrite,
            "rebuild": rebuild,
        },
    )
    discovery = _select_discovery_client(host=host, market=market)
    query = CnReportQuery(
        market=market,
        normalized_ticker=normalized_ticker,
        start_date=window.start_date,
        end_date=window.end_date,
        target_periods=periods.target_periods,
    )
    filings: list[JsonObject] = []
    warnings: list[str] = []
    notes = list(periods.notes)
    try:
        profile = discovery.resolve_company(query)
        company_meta = upsert_company_meta_for_cn_download(
            repository=host.company_meta_repository,
            profile=profile,
            normalized_ticker=normalized_ticker,
            ticker_aliases=ticker_aliases,
        )
        yield DownloadEvent(
            event_type=DownloadEventType.COMPANY_RESOLVED,
            ticker=normalized_ticker,
            payload={
                "company_id": company_meta.company_id,
                "provider_company_id": profile.company_id,
                "company_name": profile.company_name,
                "market": market,
            },
        )
        if overwrite:
            host.filing_maintenance_repository.clear_filing_documents(normalized_ticker)
        candidates = discovery.list_report_candidates(query, profile)
        selected = _select_candidates_for_a4(candidates)
        missing_periods = _resolve_missing_periods(periods.target_periods, selected)
        for period in missing_periods:
            skipped = _build_missing_period_result(period=period)
            filings.append(skipped)
            yield DownloadEvent(
                event_type=DownloadEventType.FILING_COMPLETED,
                ticker=normalized_ticker,
                payload={"filing_result": skipped, **skipped},
            )
        cancelled = False
        for candidate in selected:
            if cancel_checker is not None and cancel_checker():
                notes.append("cancelled")
                cancelled = True
                break
            document_id = _candidate_document_id(normalized_ticker, candidate)
            yield DownloadEvent(
                event_type=DownloadEventType.FILING_STARTED,
                ticker=normalized_ticker,
                document_id=document_id,
                payload={
                    "form_type": candidate.fiscal_period,
                    "filing_date": candidate.filing_date,
                    "fiscal_year": candidate.fiscal_year,
                    "fiscal_period": candidate.fiscal_period,
                    "source_id": candidate.source_id,
                },
            )
            try:
                async for event in run_cn_download_single_filing_stream(
                    source_repository=host.source_repository,
                    blob_repository=host.blob_repository,
                    processed_repository=host.processed_repository,
                    discovery_client=discovery,
                    convert_pdf_to_docling_json=host.convert_pdf_to_docling_json,
                    ticker=normalized_ticker,
                    profile=profile,
                    candidate=candidate,
                    overwrite=overwrite,
                    cancel_checker=cancel_checker,
                    module=module,
                ):
                    item = event.payload.get("filing_result")
                    if isinstance(item, dict):
                        filings.append(dict(item))
                    yield event
            except CancelledError:
                notes.append("cancelled")
                cancelled = True
                break
            except Exception as exc:
                failed_item = _build_candidate_failed_result(
                    ticker=normalized_ticker,
                    candidate=candidate,
                    reason_code=_reason_code_from_exception(exc),
                    reason_message=str(exc),
                )
                filings.append(failed_item)
                yield DownloadEvent(
                    event_type=DownloadEventType.FILING_FAILED,
                    ticker=normalized_ticker,
                    document_id=str(failed_item["document_id"]),
                    payload={"filing_result": failed_item, **failed_item},
                )
    except Exception as exc:
        failed = _build_result(
            pipeline_name=pipeline_name,
            status="failed",
            ticker=normalized_ticker,
            reason_code=_reason_code_from_exception(exc),
            message=str(exc),
            filings=filings,
        )
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_COMPLETED,
            ticker=normalized_ticker,
            payload={"result": failed},
        )
        return

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    summary = _build_summary(filings=filings, elapsed_ms=elapsed_ms)
    result = _build_result(
        pipeline_name=pipeline_name,
        status="cancelled" if cancelled or (cancel_checker is not None and cancel_checker()) else "ok",
        ticker=normalized_ticker,
        company_info={
            "company_id": company_meta.company_id,
            "provider_company_id": profile.company_id,
            "company_name": profile.company_name,
            "market": market,
        },
        filters={
            "forms": list(periods.target_periods),
            "start_dates": {period: window.start_date for period in periods.target_periods},
            "end_date": window.end_date,
            "overwrite": overwrite,
        },
        warnings=warnings,
        notes=notes,
        filings=filings,
        summary=summary,
    )
    Log.info(
        (
            "CN/HK 下载完成: "
            f"ticker={normalized_ticker} total={summary['total']} "
            f"downloaded={summary['downloaded']} skipped={summary['skipped']} "
            f"failed={summary['failed']}"
        ),
        module=module,
    )
    yield DownloadEvent(
        event_type=DownloadEventType.PIPELINE_COMPLETED,
        ticker=normalized_ticker,
        payload={"result": result},
    )


def _select_discovery_client(
    *,
    host: CnDownloadWorkflowHost,
    market: CnMarketKind,
) -> CnReportDiscoveryClientProtocol:
    """按市场选择 discovery client。"""

    return host.cn_discovery_client if market == "CN" else host.hk_discovery_client


def _coerce_market(raw: str) -> CnMarketKind:
    """把 ticker_normalization 市场收窄为 CN/HK 字面量。"""

    if raw == "CN":
        return "CN"
    if raw == "HK":
        return "HK"
    raise ValueError(f"不支持的 market: {raw}")


def _select_candidates_for_a4(
    candidates: tuple[CnReportCandidate, ...],
) -> tuple[CnReportCandidate, ...]:
    """返回 downloader 在窗口内选出的全部候选。

    Args:
        candidates: downloader 已按 ``(fiscal_year, fiscal_period)`` 去重后的候选。

    Returns:
        原候选 tuple，不在 workflow 层再次截断。

    Raises:
        无。
    """

    return candidates


def _resolve_missing_periods(
    requested: tuple[str, ...],
    selected: tuple[CnReportCandidate, ...],
) -> tuple[str, ...]:
    """计算无候选的请求 period。"""

    found = {item.fiscal_period for item in selected}
    return tuple(period for period in requested if period not in found)


def _build_missing_period_result(*, period: str) -> JsonObject:
    """构建 period 缺失 skipped 结果。"""

    return {
        "document_id": "",
        "status": "skipped",
        "form_type": period,
        "filing_date": None,
        "report_date": None,
        "downloaded_files": 0,
        "skipped_files": 0,
        "failed_files": [],
        "has_xbrl": False,
        "reason_code": "candidate_not_found",
        "reason_message": "主源未返回对应财期报告",
        "skip_reason": "candidate_not_found",
    }


def _build_candidate_failed_result(
    *,
    ticker: str,
    candidate: CnReportCandidate,
    reason_code: str,
    reason_message: str,
) -> JsonObject:
    """构建单候选异常失败结果。

    Args:
        ticker: ticker。
        candidate: 远端候选。
        reason_code: 稳定原因码。
        reason_message: 失败说明。

    Returns:
        单 filing 失败结果。

    Raises:
        无。
    """

    return {
        "document_id": _candidate_document_id(ticker, candidate),
        "status": "failed",
        "form_type": candidate.fiscal_period,
        "filing_date": candidate.filing_date,
        "report_date": None,
        "fiscal_year": candidate.fiscal_year,
        "fiscal_period": candidate.fiscal_period,
        "downloaded_files": 0,
        "skipped_files": 0,
        "failed_files": [],
        "has_xbrl": False,
        "reason_code": reason_code,
        "reason_message": reason_message,
    }


def _build_summary(*, filings: list[JsonObject], elapsed_ms: int) -> JsonObject:
    """构建下载 summary。"""

    return {
        "total": len(filings),
        "downloaded": sum(1 for item in filings if item.get("status") == "downloaded"),
        "skipped": sum(1 for item in filings if item.get("status") == "skipped"),
        "failed": sum(1 for item in filings if item.get("status") == "failed"),
        "elapsed_ms": elapsed_ms,
        "reused_downloads": sum(1 for item in filings if item.get("reused_pdf") is True),
        "converted": sum(1 for item in filings if item.get("converted") is True),
    }


def _build_result(
    *,
    pipeline_name: str,
    status: str,
    ticker: str,
    reason_code: str | None = None,
    message: str | None = None,
    company_info: JsonObject | None = None,
    filters: JsonObject | None = None,
    warnings: list[str] | None = None,
    notes: list[str] | None = None,
    filings: list[JsonObject] | None = None,
    summary: JsonObject | None = None,
) -> JsonObject:
    """构建 pipeline download 结果。"""

    warning_values: list[JsonValue] = list(warnings or [])
    note_values: list[JsonValue] = list(notes or [])
    filing_values: list[JsonValue] = list(filings or [])
    return {
        "pipeline": pipeline_name,
        "action": "download",
        "status": status,
        "ticker": ticker,
        "reason_code": reason_code,
        "message": message,
        "company_info": company_info or {},
        "filters": filters or {},
        "warnings": warning_values,
        "notes": note_values,
        "filings": filing_values,
        "summary": summary or {
            "total": 0,
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "elapsed_ms": 0,
        },
    }


def _candidate_document_id(ticker: str, candidate: CnReportCandidate) -> str:
    """构建单候选真实 document_id。

    Args:
        ticker: 已归一化 ticker。
        candidate: 远端候选。

    Returns:
        与单 filing 阶段机一致的 source document ID。

    Raises:
        无。
    """

    document_id, _ = build_cn_filing_ids(
        ticker=ticker,
        form_type=candidate.fiscal_period,
        fiscal_year=candidate.fiscal_year,
        fiscal_period=candidate.fiscal_period,
        amended=candidate.amended,
    )
    return document_id


def _reason_code_from_exception(exc: Exception) -> str:
    """把异常映射为稳定 reason code。"""

    if isinstance(exc, CnDownloadFilingError):
        return "filing_download_failed"
    return "cn_download_failed"


__all__ = ["run_cn_download_stream_impl"]
