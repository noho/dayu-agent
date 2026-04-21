"""SecPipeline 预处理工作流模块。"""

from __future__ import annotations

from typing import Any, AsyncIterator, Callable, Optional, Protocol, Sequence

from dayu.contracts.cancellation import CancelledError
from dayu.log import Log
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.fins.domain.enums import SourceKind
from dayu.fins.ingestion.process_events import ProcessEvent, ProcessEventType
from dayu.fins.ticker_normalization import normalize_ticker
from dayu.fins.storage import ProcessedDocumentRepositoryProtocol, SourceDocumentRepositoryProtocol

from .processing_helpers import (
    extract_process_identity_fields as _extract_process_identity_fields_common,
    filter_requested_document_ids as _filter_requested_document_ids_common,
    log_process_document_result as _log_process_document_result_common,
    register_processed_snapshot_document as _register_processed_snapshot_document_common,
    resolve_expected_parser_version as _resolve_expected_parser_version_common,
)


class _TickerNormalizer(Protocol):
    """最小 ticker 标准化边界。"""

    def normalize_ticker(self, ticker: str) -> str:
        """标准化 ticker。

        Args:
            ticker: 股票代码。

        Returns:
            标准化后的 ticker。

        Raises:
            ValueError: ticker 非法时抛出。
        """

        ...


class SecProcessWorkflowHost(Protocol):
    """Sec process 工作流所需的最小宿主边界。"""

    @property
    def MODULE(self) -> str:
        """返回日志模块名。"""

        ...

    @property
    def _downloader(self) -> _TickerNormalizer:
        """返回具备 ticker 标准化能力的下载器。"""

        ...

    @property
    def _processor_registry(self) -> ProcessorRegistry:
        """返回处理器注册表。"""

        ...

    @property
    def _source_repository(self) -> SourceDocumentRepositoryProtocol:
        """返回 source 仓储。"""

        ...

    @property
    def _processed_repository(self) -> ProcessedDocumentRepositoryProtocol:
        """返回 processed 仓储。"""

        ...

    def _safe_get_document_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> Optional[dict[str, Any]]:
        """安全读取 source meta。"""

        ...

    def _safe_read_snapshot_meta(
        self,
        *,
        ticker: str,
        document_id: str,
    ) -> Optional[dict[str, Any]]:
        """安全读取快照 meta。"""

        ...

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
        """判断是否可跳过快照导出。"""

        ...

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
        """导出单文档快照。"""

        ...

    def _build_result(self, action: str, **payload: Any) -> dict[str, Any]:
        """构建统一结果。"""

        ...


def run_process_single_document(
    host: SecProcessWorkflowHost,
    *,
    ticker: str,
    document_id: str,
    overwrite: bool,
    ci: bool,
    source_kind: SourceKind,
    action: str,
    cancel_checker: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """执行单个 source 文档的快照导出决策。

    Args:
        host: `SecPipeline` facade 暴露出的最小宿主边界。
        ticker: 股票代码。
        document_id: 文档 ID。
        overwrite: 是否强制重处理。
        ci: 是否导出 CI 扩展快照。
        source_kind: 源文档类型。
        action: 结果中的动作名。
        cancel_checker: 可选取消检查函数，仅在同步单文档处理阶段边界生效。

    Returns:
        单文档处理结果字典。

    Raises:
        RuntimeError: 处理器解析或快照导出失败时抛出。
        ValueError: 市场类型非法时抛出。
    """

    normalized = normalize_ticker(ticker)
    if normalized.market != "US":
        raise ValueError(f"SecPipeline 仅支持 US，当前 market={normalized.market}")
    normalized_ticker = host._downloader.normalize_ticker(ticker)
    _raise_if_cancelled(
        host=host,
        ticker=normalized_ticker,
        document_id=document_id,
        cancel_checker=cancel_checker,
    )

    source_meta = host._source_repository.get_source_meta(
        ticker=normalized_ticker,
        document_id=document_id,
        source_kind=source_kind,
    )
    if not bool(source_meta.get("ingest_complete", False)):
        return host._build_result(
            action=action,
            ticker=normalized_ticker,
            document_id=document_id,
            overwrite=overwrite,
            status="skipped",
            reason="ingest_incomplete",
            **_extract_process_identity_fields_common(source_meta=source_meta),
        )
    if bool(source_meta.get("is_deleted", False)):
        return host._build_result(
            action=action,
            ticker=normalized_ticker,
            document_id=document_id,
            overwrite=overwrite,
            status="skipped",
            reason="deleted",
            **_extract_process_identity_fields_common(source_meta=source_meta),
        )

    source = host._source_repository.get_primary_source(
        ticker=normalized_ticker,
        document_id=document_id,
        source_kind=source_kind,
    )
    normalized_form_type = str(source_meta.get("form_type") or "") or None
    expected_parser_signature = _resolve_expected_parser_version_common(
        processor_registry=host._processor_registry,
        source=source,
        form_type=normalized_form_type,
    )
    snapshot_meta = host._safe_read_snapshot_meta(
        ticker=normalized_ticker,
        document_id=document_id,
    )
    if host._can_skip_snapshot_export(
        source_meta=source_meta,
        snapshot_meta=snapshot_meta,
        overwrite=overwrite,
        expected_parser_signature=expected_parser_signature,
        ci=ci,
        ticker=normalized_ticker,
        document_id=document_id,
    ):
        _register_processed_snapshot_document_common(
            repository=host._processed_repository,
            ticker=normalized_ticker,
            document_id=document_id,
            source_kind=source_kind,
            source_meta=source_meta,
            parser_signature=expected_parser_signature,
            has_xbrl=bool(snapshot_meta and snapshot_meta.get("has_xbrl", False)),
        )
        return host._build_result(
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
        host=host,
        ticker=normalized_ticker,
        document_id=document_id,
        cancel_checker=cancel_checker,
    )
    host._export_tool_snapshot_for_document(
        ticker=normalized_ticker,
        document_id=document_id,
        source_kind=source_kind,
        source_meta=source_meta,
        ci=ci,
        expected_parser_signature=expected_parser_signature,
        cancel_checker=cancel_checker,
    )
    _raise_if_cancelled(
        host=host,
        ticker=normalized_ticker,
        document_id=document_id,
        cancel_checker=cancel_checker,
    )
    return host._build_result(
        action=action,
        ticker=normalized_ticker,
        document_id=document_id,
        overwrite=overwrite,
        ci=ci,
        status="processed",
        **_extract_process_identity_fields_common(source_meta=source_meta),
    )


def _raise_if_cancelled(
    *,
    host: SecProcessWorkflowHost,
    ticker: str,
    document_id: str,
    cancel_checker: Callable[[], bool] | None,
) -> None:
    """在同步单文档处理阶段边界检查取消请求。"""

    if cancel_checker is None or not cancel_checker():
        return
    Log.info(
        "单文档预处理收到取消请求，停止执行: "
        f"ticker={ticker} document_id={document_id}",
        module=host.MODULE,
    )
    raise CancelledError("操作已被取消")


async def run_process_stream_impl(
    host: SecProcessWorkflowHost,
    *,
    ticker: str,
    overwrite: bool = False,
    ci: bool = False,
    document_ids: Optional[Sequence[str]] = None,
    cancel_checker: Optional[Callable[[], bool]] = None,
) -> AsyncIterator[ProcessEvent]:
    """执行流式全量离线快照导出。

    Args:
        host: `SecPipeline` facade 暴露出的最小宿主边界。
        ticker: 股票代码。
        overwrite: 是否强制覆盖。
        ci: 是否导出 CI 扩展快照。
        document_ids: 可选文档 ID 过滤列表。
        cancel_checker: 可选取消检查函数，仅在文档边界生效。

    Yields:
        预处理事件流。

    Raises:
        ValueError: 市场类型非法时抛出。
        RuntimeError: 单文档处理失败时向上传递异常并转为 failed 事件。
    """

    normalized = normalize_ticker(ticker)
    if normalized.market != "US":
        raise ValueError(f"SecPipeline 仅支持 US，当前 market={normalized.market}")
    normalized_ticker = host._downloader.normalize_ticker(ticker)

    if overwrite and document_ids is None:
        host._processed_repository.clear_processed_documents(normalized_ticker)

    filing_ids = _filter_requested_document_ids_common(
        host._source_repository.list_source_document_ids(normalized_ticker, SourceKind.FILING),
        document_ids,
    )
    material_ids = _filter_requested_document_ids_common(
        host._source_repository.list_source_document_ids(normalized_ticker, SourceKind.MATERIAL),
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

    filing_results: list[dict[str, Any]] = []
    for document_id in filing_ids:
        if cancel_checker is not None and cancel_checker():
            Log.info(
                f"预处理任务收到取消请求，文档边界停止: ticker={normalized_ticker}",
                module=host.MODULE,
            )
            break
        source_meta = host._safe_get_document_meta(normalized_ticker, document_id, SourceKind.FILING)
        if source_meta is None:
            item = {
                "document_id": document_id,
                "status": "skipped",
                "reason": "missing_meta",
            }
            filing_results.append(item)
            _log_process_document_result_common(
                module=host.MODULE,
                ticker=normalized_ticker,
                source_kind=SourceKind.FILING,
                result=item,
            )
            yield ProcessEvent(
                event_type=ProcessEventType.DOCUMENT_SKIPPED,
                ticker=normalized_ticker,
                document_id=document_id,
                payload={
                    "source_kind": SourceKind.FILING.value,
                    "reason": "missing_meta",
                    "result_summary": item,
                },
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
                module=host.MODULE,
                ticker=normalized_ticker,
                source_kind=SourceKind.FILING,
                result=item,
            )
            yield ProcessEvent(
                event_type=ProcessEventType.DOCUMENT_SKIPPED,
                ticker=normalized_ticker,
                document_id=document_id,
                payload={
                    "source_kind": SourceKind.FILING.value,
                    "reason": "ingest_incomplete",
                    "result_summary": item,
                },
            )
            continue
        yield ProcessEvent(
            event_type=ProcessEventType.DOCUMENT_STARTED,
            ticker=normalized_ticker,
            document_id=document_id,
            payload={"source_kind": SourceKind.FILING.value},
        )
        try:
            item = run_process_single_document(
                host,
                ticker=normalized_ticker,
                document_id=document_id,
                overwrite=overwrite,
                ci=ci,
                source_kind=SourceKind.FILING,
                action="process_filing",
            )
            filing_results.append(item)
            _log_process_document_result_common(
                module=host.MODULE,
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
                module=host.MODULE,
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

    filing_summary = _build_process_summary(filing_results)

    material_results: list[dict[str, Any]] = []
    for document_id in material_ids:
        if cancel_checker is not None and cancel_checker():
            Log.info(
                f"预处理任务收到取消请求，文档边界停止: ticker={normalized_ticker}",
                module=host.MODULE,
            )
            break
        source_meta = host._safe_get_document_meta(normalized_ticker, document_id, SourceKind.MATERIAL)
        if source_meta is None:
            item = {
                "document_id": document_id,
                "status": "skipped",
                "reason": "missing_meta",
            }
            material_results.append(item)
            _log_process_document_result_common(
                module=host.MODULE,
                ticker=normalized_ticker,
                source_kind=SourceKind.MATERIAL,
                result=item,
            )
            yield ProcessEvent(
                event_type=ProcessEventType.DOCUMENT_SKIPPED,
                ticker=normalized_ticker,
                document_id=document_id,
                payload={
                    "source_kind": SourceKind.MATERIAL.value,
                    "reason": "missing_meta",
                    "result_summary": item,
                },
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
                module=host.MODULE,
                ticker=normalized_ticker,
                source_kind=SourceKind.MATERIAL,
                result=item,
            )
            yield ProcessEvent(
                event_type=ProcessEventType.DOCUMENT_SKIPPED,
                ticker=normalized_ticker,
                document_id=document_id,
                payload={
                    "source_kind": SourceKind.MATERIAL.value,
                    "reason": "ingest_incomplete",
                    "result_summary": item,
                },
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
                module=host.MODULE,
                ticker=normalized_ticker,
                source_kind=SourceKind.MATERIAL,
                result=item,
            )
            yield ProcessEvent(
                event_type=ProcessEventType.DOCUMENT_SKIPPED,
                ticker=normalized_ticker,
                document_id=document_id,
                payload={
                    "source_kind": SourceKind.MATERIAL.value,
                    "reason": "deleted",
                    "result_summary": item,
                },
            )
            continue
        yield ProcessEvent(
            event_type=ProcessEventType.DOCUMENT_STARTED,
            ticker=normalized_ticker,
            document_id=document_id,
            payload={"source_kind": SourceKind.MATERIAL.value},
        )
        try:
            item = run_process_single_document(
                host,
                ticker=normalized_ticker,
                document_id=document_id,
                overwrite=overwrite,
                ci=ci,
                source_kind=SourceKind.MATERIAL,
                action="process_material",
            )
            material_results.append(item)
            _log_process_document_result_common(
                module=host.MODULE,
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
                module=host.MODULE,
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

    material_summary = _build_process_summary(material_results)
    final_result = host._build_result(
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


def _build_process_summary(results: list[dict[str, Any]]) -> dict[str, int]:
    """汇总批量 process 结果统计。

    Args:
        results: 单文档结果列表。

    Returns:
        `total/processed/skipped/failed` 统计字典。

    Raises:
        无。
    """

    return {
        "total": len(results),
        "processed": sum(1 for item in results if item.get("status") == "processed"),
        "skipped": sum(1 for item in results if item.get("status") == "skipped"),
        "failed": sum(1 for item in results if item.get("status") == "failed"),
    }
