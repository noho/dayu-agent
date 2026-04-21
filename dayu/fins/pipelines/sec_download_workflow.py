"""SecPipeline 下载工作流模块。"""

from __future__ import annotations

import datetime as dt
import inspect
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Protocol, TypeVar

from dayu.fins.domain.enums import SourceKind
from dayu.fins.ingestion.process_events import ProcessEvent
from dayu.fins.pipelines.download_events import DownloadEvent, DownloadEventType
from dayu.fins.ticker_normalization import normalize_ticker
from dayu.fins.storage import FilingMaintenanceRepositoryProtocol, SourceDocumentRepositoryProtocol
from dayu.log import Log


class _DownloadWorkflowDownloader(Protocol):
    """下载工作流所需的最小下载器边界。"""

    def configure(self, user_agent: Optional[str], sleep_seconds: float, max_retries: int) -> None:
        """配置下载器。"""

        ...

    def normalize_ticker(self, ticker: str) -> str:
        """标准化 ticker。"""

        ...

    def resolve_company(self, ticker: str) -> Awaitable[tuple[str, str, str]] | tuple[str, str, str]:
        """解析公司信息。"""

        ...

    def fetch_submissions(self, cik10: str) -> Awaitable[dict[str, Any]] | dict[str, Any]:
        """拉取 submissions。"""

        ...


class SecDownloadWorkflowHost(Protocol):
    """Sec download 工作流所需的最小宿主边界。"""

    @property
    def MODULE(self) -> str:
        """返回日志模块名。"""

        ...

    @property
    def _downloader(self) -> _DownloadWorkflowDownloader:
        """返回下载器实例。"""

        ...

    @property
    def _user_agent(self) -> Optional[str]:
        """返回 User-Agent。"""

        ...

    @property
    def _sleep_seconds(self) -> float:
        """返回下载间隔秒数。"""

        ...

    @property
    def _max_retries(self) -> int:
        """返回最大重试次数。"""

        ...

    @property
    def _filing_maintenance_repository(self) -> FilingMaintenanceRepositoryProtocol:
        """返回 filing 维护仓储。"""

        ...

    @property
    def _source_repository(self) -> SourceDocumentRepositoryProtocol:
        """返回 source 仓储。"""

        ...

    def _rebuild_download_artifacts(
        self,
        *,
        ticker: str,
        form_type: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        overwrite: bool,
    ) -> dict[str, Any]:
        """基于本地已下载 filings 重建 meta/manifest。"""

        ...

    def _resolve_form_windows(
        self,
        form_type: Optional[str],
        start_date: Optional[str],
        end_date: dt.date,
    ) -> dict[str, dt.date]:
        """计算 form 到起始日期映射。"""

        ...

    def _upsert_company_meta(
        self,
        ticker: str,
        company_id: str,
        company_name: str,
        ticker_aliases: Optional[list[str]],
    ) -> None:
        """写入公司元数据。"""

        ...

    def _build_result(self, action: str, **payload: Any) -> dict[str, Any]:
        """构建统一结果。"""

        ...

    def _log_filing_download_result(self, ticker: str, filing_result: dict[str, Any]) -> None:
        """记录单个 filing 下载结果。"""

        ...

    def _download_single_filing_stream(
        self,
        *,
        ticker: str,
        cik: str,
        filing: Any,
        overwrite: bool,
        rejection_registry: dict[str, dict[str, str]],
    ) -> AsyncIterator[DownloadEvent]:
        """执行单 filing 下载流。"""

        ...

    def _filter_filings(
        self,
        *,
        ticker: str,
        submissions: dict[str, Any],
        form_windows: dict[str, dt.date],
        end_date: dt.date,
        target_cik: str,
        sc13_direction_cache: Optional[dict[str, Optional[bool]]] = None,
        rejection_registry: Optional[dict[str, dict[str, str]]] = None,
        overwrite: bool = False,
    ) -> Awaitable[tuple[list[Any], set[str]]]:
        """过滤 filings 并收集 filenum。"""

        ...

    def _extend_with_browse_edgar_sc13(
        self,
        *,
        ticker: str,
        filings: list[Any],
        filenums: set[str],
        form_windows: dict[str, dt.date],
        end_date: dt.date,
        target_cik: str,
        sc13_direction_cache: Optional[dict[str, Optional[bool]]] = None,
        rejection_registry: Optional[dict[str, dict[str, str]]] = None,
        overwrite: bool = False,
    ) -> Awaitable[list[Any]]:
        """补充 browse-edgar SC13 filings。"""

        ...

    def _retry_sc13_if_empty(
        self,
        *,
        ticker: str,
        filings: list[Any],
        filenums: set[str],
        submissions: dict[str, Any],
        form_windows: dict[str, dt.date],
        end_date: dt.date,
        target_cik: str,
        sc13_direction_cache: Optional[dict[str, Optional[bool]]] = None,
        rejection_registry: Optional[dict[str, dict[str, str]]] = None,
        overwrite: bool = False,
    ) -> Awaitable[list[Any]]:
        """在 SC13 为空时执行渐进式回溯。"""

        ...


_AwaitableResult = TypeVar("_AwaitableResult")


async def _maybe_await(value: Awaitable[_AwaitableResult] | _AwaitableResult) -> _AwaitableResult:
    """按需等待可等待对象。"""

    if inspect.isawaitable(value):
        return await value
    return value


async def run_download_stream_impl(
    host: SecDownloadWorkflowHost,
    *,
    ticker: str,
    form_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    overwrite: bool = False,
    rebuild: bool = False,
    ticker_aliases: Optional[list[str]] = None,
    cancel_checker: Optional[Callable[[], bool]] = None,
    parse_date: Callable[[str, bool], dt.date],
    extract_sec_ticker_aliases: Callable[..., list[str]],
    merge_ticker_aliases: Callable[..., list[str]],
    clear_filings_dir: Callable[[FilingMaintenanceRepositoryProtocol, str], None],
    load_rejection_registry: Callable[
        [FilingMaintenanceRepositoryProtocol, str],
        dict[str, dict[str, str]],
    ],
    save_rejection_registry: Callable[
        [FilingMaintenanceRepositoryProtocol, str, dict[str, dict[str, str]]],
        None,
    ],
    should_warn_missing_sc13: Callable[[dict[str, dt.date], list[Any]], bool],
    warn_insufficient_filings: Callable[
        [dict[str, Any], list[dict[str, Any]], dict[str, dict[str, str]]],
        list[str],
    ],
    warn_xbrl_missing_filings: Callable[[list[dict[str, Any]]], list[str]],
    cleanup_stale_filing_dirs: Callable[..., int],
    build_download_filing_event_payload: Callable[[dict[str, Any]], dict[str, Any]],
) -> AsyncIterator[DownloadEvent]:
    """执行 SecPipeline 下载主工作流。

    Args:
        host: `SecPipeline` facade 暴露出的最小宿主边界。
        ticker: 股票代码。
        form_type: 可选文档类型。
        start_date: 可选开始日期。
        end_date: 可选结束日期。
        overwrite: 是否强制覆盖。
        rebuild: 是否仅基于本地已下载数据重建 `meta/manifest`。
        ticker_aliases: CLI 侧传入的 alias 列表。
        cancel_checker: 可选取消检查函数，仅在文档边界生效。
        parse_date: 日期解析 helper。
        extract_sec_ticker_aliases: SEC alias 提取 helper。
        merge_ticker_aliases: alias 合并 helper。
        clear_filings_dir: 清空 filings 目录 helper。
        load_rejection_registry: 加载拒绝注册表 helper。
        save_rejection_registry: 保存拒绝注册表 helper。
        should_warn_missing_sc13: SC13 缺失 warning helper。
        warn_insufficient_filings: form 数量检查 helper。
        warn_xbrl_missing_filings: XBRL 缺失检查 helper。
        cleanup_stale_filing_dirs: 清理过期 filing 目录 helper。
        build_download_filing_event_payload: 构建 filing 事件 payload helper。

    Yields:
        下载流程事件流。

    Raises:
        ValueError: ticker 不合法或市场不匹配时抛出。
        RuntimeError: 下载执行失败时抛出。
    """

    normalized = normalize_ticker(ticker)
    if normalized.market != "US":
        raise ValueError(f"SecPipeline 仅支持 US，当前 market={normalized.market}")
    normalized_ticker = host._downloader.normalize_ticker(ticker)
    if rebuild:
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_STARTED,
            ticker=normalized_ticker,
            payload={
                "form_type": form_type,
                "start_date": start_date,
                "end_date": end_date,
                "overwrite": overwrite,
                "rebuild": True,
            },
        )
        rebuild_result = host._rebuild_download_artifacts(
            ticker=normalized_ticker,
            form_type=form_type,
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
        )
        for filing_result in rebuild_result.get("filings", []):
            if not isinstance(filing_result, dict):
                continue
            status = str(filing_result.get("status", "failed"))
            event_type = (
                DownloadEventType.FILING_FAILED
                if status == "failed"
                else DownloadEventType.FILING_COMPLETED
            )
            document_id = str(filing_result.get("document_id", ""))
            yield DownloadEvent(
                event_type=event_type,
                ticker=normalized_ticker,
                document_id=document_id,
                payload=build_download_filing_event_payload(filing_result),
            )
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_COMPLETED,
            ticker=normalized_ticker,
            payload={"result": rebuild_result},
        )
        return

    download_end_date = parse_date(end_date, True) if end_date else dt.date.today()
    host._downloader.configure(
        user_agent=host._user_agent,
        sleep_seconds=host._sleep_seconds,
        max_retries=host._max_retries,
    )
    yield DownloadEvent(
        event_type=DownloadEventType.PIPELINE_STARTED,
        ticker=normalized_ticker,
        payload={
            "form_type": form_type,
            "start_date": start_date,
            "end_date": end_date,
            "overwrite": overwrite,
            "rebuild": False,
        },
    )

    cik, company_name, cik10 = await _maybe_await(host._downloader.resolve_company(normalized_ticker))
    submissions = await _maybe_await(host._downloader.fetch_submissions(cik10))
    sec_ticker_aliases = extract_sec_ticker_aliases(
        submissions=submissions,
        primary_ticker=normalized_ticker,
    )
    merged_ticker_aliases = merge_ticker_aliases(
        primary_ticker=normalized_ticker,
        alias_groups=[sec_ticker_aliases, ticker_aliases],
    )
    yield DownloadEvent(
        event_type=DownloadEventType.COMPANY_RESOLVED,
        ticker=normalized_ticker,
        payload={
            "cik": cik,
            "company_name": company_name,
            "cik10": cik10,
        },
    )
    form_windows = host._resolve_form_windows(
        form_type=form_type,
        start_date=start_date,
        end_date=download_end_date,
    )
    Log.verbose(
        f"下载窗口详情: { {key: value.isoformat() for key, value in form_windows.items()} }",
        module=host.MODULE,
    )
    Log.info(
        (
            "进入美股下载流程: "
            f"ticker={normalized_ticker} form_type={form_type} start={start_date} end={end_date} "
            f"overwrite={overwrite}"
        ),
        module=host.MODULE,
    )
    host._upsert_company_meta(
        ticker=normalized_ticker,
        company_id=cik,
        company_name=company_name,
        ticker_aliases=merged_ticker_aliases,
    )
    sc13_direction_cache: dict[str, Optional[bool]] = {}
    if overwrite:
        clear_filings_dir(host._filing_maintenance_repository, normalized_ticker)
    rejection_registry = load_rejection_registry(host._filing_maintenance_repository, normalized_ticker)
    filings, filenums = await host._filter_filings(
        ticker=normalized_ticker,
        submissions=submissions,
        form_windows=form_windows,
        end_date=download_end_date,
        target_cik=cik,
        sc13_direction_cache=sc13_direction_cache,
        rejection_registry=rejection_registry,
        overwrite=overwrite,
    )
    filings = await host._extend_with_browse_edgar_sc13(
        ticker=normalized_ticker,
        filings=filings,
        filenums=filenums,
        form_windows=form_windows,
        end_date=download_end_date,
        target_cik=cik,
        sc13_direction_cache=sc13_direction_cache,
        rejection_registry=rejection_registry,
        overwrite=overwrite,
    )
    filings = await host._retry_sc13_if_empty(
        ticker=normalized_ticker,
        filings=filings,
        filenums=filenums,
        submissions=submissions,
        form_windows=form_windows,
        end_date=download_end_date,
        target_cik=cik,
        sc13_direction_cache=sc13_direction_cache,
        rejection_registry=rejection_registry,
        overwrite=overwrite,
    )
    warnings: list[str] = []
    if should_warn_missing_sc13(form_windows, filings):
        warning = (
            "未在 issuer 的 submissions/browse-edgar 中发现 SC 13D/G；"
            "13D/G 往往由申报人提交，需要申报人 CIK 维度或反查补齐。"
        )
        warnings.append(warning)
        Log.warn(warning, module=host.MODULE)

    filing_results: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for filing in filings:
        if cancel_checker is not None and cancel_checker():
            Log.info(
                f"下载任务收到取消请求，文档边界停止: ticker={normalized_ticker}",
                module=host.MODULE,
            )
            break
        document_id = f"fil_{filing.accession_number}"
        yield DownloadEvent(
            event_type=DownloadEventType.FILING_STARTED,
            ticker=normalized_ticker,
            document_id=document_id,
            payload={
                "form_type": filing.form_type,
                "filing_date": filing.filing_date,
                "report_date": filing.report_date,
                "accession_number": filing.accession_number,
                "total_filings": len(filings),
            },
        )
        async for event in host._download_single_filing_stream(
            ticker=normalized_ticker,
            cik=cik,
            filing=filing,
            overwrite=overwrite,
            rejection_registry=rejection_registry,
        ):
            event_result = event.payload.get("filing_result")
            if event.event_type in {
                DownloadEventType.FILING_COMPLETED,
                DownloadEventType.FILING_FAILED,
            } and isinstance(event_result, dict):
                filing_results.append(event_result)
                host._log_filing_download_result(
                    ticker=normalized_ticker,
                    filing_result=event_result,
                )
            yield event

    save_rejection_registry(host._filing_maintenance_repository, normalized_ticker, rejection_registry)
    for warning in warn_insufficient_filings(
        form_windows,
        filing_results,
        rejection_registry,
    ):
        warnings.append(warning)
        Log.warn(warning, module=host.MODULE)
    for warning in warn_xbrl_missing_filings(filing_results):
        warnings.append(warning)
        Log.warn(warning, module=host.MODULE)
    cleaned = cleanup_stale_filing_dirs(
        repository=host._filing_maintenance_repository,
        ticker=normalized_ticker,
        form_windows=form_windows,
        filing_results=filing_results,
    )
    if cleaned:
        Log.info(
            f"清理过期 filing 目录: ticker={normalized_ticker} cleaned={cleaned}",
            module=host.MODULE,
        )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    summary = {
        "total": len(filing_results),
        "downloaded": sum(1 for item in filing_results if item["status"] == "downloaded"),
        "skipped": sum(1 for item in filing_results if item["status"] == "skipped"),
        "failed": sum(1 for item in filing_results if item["status"] == "failed"),
        "elapsed_ms": elapsed_ms,
    }
    Log.info(
        (
            "美股下载完成: "
            f"ticker={normalized_ticker} total={summary['total']} downloaded={summary['downloaded']} "
            f"skipped={summary['skipped']} failed={summary['failed']} elapsed_ms={summary['elapsed_ms']}"
        ),
        module=host.MODULE,
    )
    final_result = host._build_result(
        action="download",
        ticker=normalized_ticker,
        market_profile={
            "market": normalized.market,
        },
        filters={
            "forms": sorted(form_windows.keys()),
            "start_dates": {key: value.isoformat() for key, value in sorted(form_windows.items())},
            "end_date": download_end_date.isoformat(),
            "overwrite": overwrite,
        },
        warnings=warnings,
        filings=filing_results,
        summary=summary,
        status="cancelled" if cancel_checker is not None and cancel_checker() else "ok",
    )
    yield DownloadEvent(
        event_type=DownloadEventType.PIPELINE_COMPLETED,
        ticker=normalized_ticker,
        payload={"result": final_result},
    )