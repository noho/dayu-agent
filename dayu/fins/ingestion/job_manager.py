"""长事务 Job 管理器。"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Optional

from dayu.log import Log
from dayu.fins._converters import int_or_zero, normalize_optional_text
from dayu.fins.domain.document_models import now_iso8601

from ..pipelines.download_events import DownloadEventType
from .process_events import ProcessEvent, ProcessEventType
from .service import FinsIngestionService

MODULE = "FINS.INGESTION.JOB_MANAGER"
_TERMINAL_JOB_TTL = timedelta(minutes=60)

JobType = Literal["download", "process"]
JobStatus = Literal["queued", "running", "cancelling", "succeeded", "failed", "cancelled"]
IngestionServiceFactory = Callable[[str], FinsIngestionService]


@dataclass
class _IngestionJob:
    """内存中的 job 运行态。"""

    job_id: str
    job_type: JobType
    ticker: str
    request_payload: dict[str, Any]
    request_fingerprint: str
    progress_unit: str
    status: JobStatus = "queued"
    stage: str = "queued"
    created_at: str = field(default_factory=now_iso8601)
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    progress_completed: int = 0
    progress_total: Optional[int] = None
    result_summary: Optional[dict[str, Any]] = None
    failure: Optional[dict[str, Any]] = None
    final_result: Optional[dict[str, Any]] = None
    recent_issues: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False


class IngestionJobManager:
    """进程内长事务 job 管理器。

    设计约束：
    - 生命周期独立于 `AsyncAgent`。
    - 仅依赖 repo 已落盘状态做恢复/跳过。
    - `download/process` 各自串行执行一个 job。
    """

    def __init__(
        self,
        *,
        service_factory: IngestionServiceFactory,
        manager_key: str,
    ) -> None:
        """初始化 job 管理器。

        Args:
            service_factory: 按 ticker 构建共享事务服务的工厂。
            manager_key: 管理器标识，用于全局复用。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        if service_factory is None:
            raise ValueError("service_factory 不能为空")
        self._service_factory = service_factory
        self._manager_key = manager_key
        self._jobs: dict[str, _IngestionJob] = {}
        self._lock = threading.RLock()
        self._queues: dict[JobType, queue.Queue[str]] = {
            "download": queue.Queue(),
            "process": queue.Queue(),
        }
        self._workers: dict[JobType, threading.Thread] = {}

    def start_download_job(
        self,
        *,
        ticker: str,
        form_types: Optional[list[str]],
        filed_date_from: Optional[str],
        filed_date_to: Optional[str],
        overwrite: bool,
    ) -> tuple[str, dict[str, Any]]:
        """启动下载 job。

        Args:
            ticker: 股票代码。
            form_types: 可选表单数组。
            filed_date_from: 可选开始日期。
            filed_date_to: 可选结束日期。
            overwrite: 是否覆盖。

        Returns:
            `(request_outcome, snapshot)`。
        """

        normalized_form_types = sorted({item.strip() for item in form_types or [] if item and item.strip()})
        request_payload = {
            "ticker": str(ticker).strip().upper(),
            "form_types": normalized_form_types or None,
            "form_type": " ".join(normalized_form_types) if normalized_form_types else None,
            "filed_date_from": normalize_optional_text(filed_date_from),
            "filed_date_to": normalize_optional_text(filed_date_to),
            "overwrite": bool(overwrite),
            "rebuild": False,
        }
        return self._start_job(job_type="download", request_payload=request_payload, progress_unit="filing")

    def start_process_job(
        self,
        *,
        ticker: str,
        overwrite: bool,
        document_ids: Optional[list[str]] = None,
    ) -> tuple[str, dict[str, Any]]:
        """启动预处理 job。

        Args:
            ticker: 股票代码。
            overwrite: 是否覆盖。
            document_ids: 可选文档 ID 列表；为空时处理该 ticker 下当前可见文档。

        Returns:
            `(request_outcome, snapshot)`。
        """

        normalized_document_ids = _coerce_optional_document_ids(document_ids)
        request_payload = {
            "ticker": str(ticker).strip().upper(),
            "overwrite": bool(overwrite),
            "ci": False,
            "document_ids": normalized_document_ids,
        }
        return self._start_job(job_type="process", request_payload=request_payload, progress_unit="document")

    def get_job_snapshot(self, job_id: str) -> Optional[dict[str, Any]]:
        """读取 job 快照。

        Args:
            job_id: job 标识。

        Returns:
            job 快照；不存在时返回 `None`。
        """

        with self._lock:
            self._cleanup_expired_jobs_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return self._build_snapshot_locked(job)

    def cancel_job(self, job_id: str) -> tuple[str, Optional[dict[str, Any]]]:
        """请求取消 job。

        Args:
            job_id: job 标识。

        Returns:
            `(cancellation_outcome, snapshot)`；job 不存在时 snapshot 为 `None`。
        """

        with self._lock:
            self._cleanup_expired_jobs_locked()
            job = self._jobs.get(job_id)
            if job is None:
                return "not_found", None
            if job.status in {"succeeded", "failed", "cancelled"}:
                return "already_terminal", self._build_snapshot_locked(job)
            job.cancel_requested = True
            if job.status in {"queued", "running"}:
                job.status = "cancelling"
            return "cancellation_requested", self._build_snapshot_locked(job)

    def _start_job(
        self,
        *,
        job_type: JobType,
        request_payload: dict[str, Any],
        progress_unit: str,
    ) -> tuple[str, dict[str, Any]]:
        """创建或复用 active job。

        Args:
            job_type: job 类型。
            request_payload: 规范化请求载荷。
            progress_unit: 进度单位。

        Returns:
            `(request_outcome, snapshot)`。
        """

        with self._lock:
            self._cleanup_expired_jobs_locked()
            fingerprint = _make_request_fingerprint(job_type=job_type, request_payload=request_payload)
            active_job = self._find_active_job_locked(job_type=job_type, fingerprint=fingerprint)
            if active_job is not None:
                return "reused_active_job", self._build_snapshot_locked(active_job)

            job = _IngestionJob(
                job_id=f"job_{uuid.uuid4().hex}",
                job_type=job_type,
                ticker=str(request_payload["ticker"]),
                request_payload=dict(request_payload),
                request_fingerprint=fingerprint,
                progress_unit=progress_unit,
            )
            self._jobs[job.job_id] = job
            self._ensure_worker_started_locked(job_type)
            self._queues[job_type].put(job.job_id)
            return "started", self._build_snapshot_locked(job)

    def _find_active_job_locked(self, *, job_type: JobType, fingerprint: str) -> Optional[_IngestionJob]:
        """查找相同请求的 active job。"""

        for job in self._jobs.values():
            if job.job_type != job_type:
                continue
            if job.request_fingerprint != fingerprint:
                continue
            if job.status not in {"queued", "running", "cancelling"}:
                continue
            return job
        return None

    def _ensure_worker_started_locked(self, job_type: JobType) -> None:
        """确保指定类型的 worker 已启动。"""

        if job_type in self._workers and self._workers[job_type].is_alive():
            return
        worker = threading.Thread(
            target=self._worker_loop,
            name=f"fins-{self._manager_key}-{job_type}-worker",
            args=(job_type,),
            daemon=True,
        )
        worker.start()
        self._workers[job_type] = worker

    def _worker_loop(self, job_type: JobType) -> None:
        """串行执行指定类型的 job。"""

        queue_obj = self._queues[job_type]
        while True:
            job_id = queue_obj.get()
            try:
                self._run_job(job_type=job_type, job_id=job_id)
            except Exception as exc:
                Log.error(
                    f"worker 执行 job 失败: job_type={job_type} job_id={job_id} error={exc}",
                    module=MODULE,
                )
                self._mark_job_failed(
                    job_id=job_id,
                    code="execution_error",
                    message=str(exc),
                )
            finally:
                queue_obj.task_done()

    def _run_job(self, *, job_type: JobType, job_id: str) -> None:
        """执行单个 job。"""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = now_iso8601()
                return
            job.status = "running"
            job.started_at = now_iso8601()

        if job_type == "download":
            asyncio.run(self._run_download_job(job_id))
            return
        asyncio.run(self._run_process_job(job_id))

    async def _run_download_job(self, job_id: str) -> None:
        """执行下载 job。"""

        job = self._get_job_or_raise(job_id)
        request_payload = dict(job.request_payload)
        service = self._service_factory(job.ticker)
        async for event in service.download_stream(
            ticker=job.ticker,
            form_type=request_payload.get("form_type"),
            start_date=request_payload.get("filed_date_from"),
            end_date=request_payload.get("filed_date_to"),
            overwrite=bool(request_payload.get("overwrite", False)),
            rebuild=False,
            cancel_checker=lambda: self._is_cancel_requested(job_id),
        ):
            self._update_download_job_from_event(job_id=job_id, event=event)

    async def _run_process_job(self, job_id: str) -> None:
        """执行预处理 job。"""

        job = self._get_job_or_raise(job_id)
        request_payload = dict(job.request_payload)
        service = self._service_factory(job.ticker)
        async for event in service.process_stream(
            ticker=job.ticker,
            overwrite=bool(request_payload.get("overwrite", False)),
            ci=False,
            document_ids=_coerce_optional_document_ids(request_payload.get("document_ids")),
            cancel_checker=lambda: self._is_cancel_requested(job_id),
        ):
            self._update_process_job_from_event(job_id=job_id, event=event)

    def _get_job_or_raise(self, job_id: str) -> _IngestionJob:
        """读取 job；不存在时抛出异常。"""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"job 不存在: {job_id}")
            return job

    def _is_cancel_requested(self, job_id: str) -> bool:
        """判断 job 是否请求取消。"""

        with self._lock:
            job = self._jobs.get(job_id)
            return bool(job and job.cancel_requested)

    def _update_download_job_from_event(self, *, job_id: str, event: Any) -> None:
        """根据下载事件更新 job。"""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            payload = event.payload if isinstance(event.payload, dict) else {}
            if event.event_type == DownloadEventType.PIPELINE_STARTED:
                job.stage = "resolving_company"
                return
            if event.event_type == DownloadEventType.COMPANY_RESOLVED:
                job.stage = "scanning_filings"
                return
            if event.event_type == DownloadEventType.FILING_STARTED:
                job.stage = "downloading_filings"
                total_filings = payload.get("total_filings")
                if isinstance(total_filings, int):
                    job.progress_total = total_filings
                return
            if event.event_type in {
                DownloadEventType.FILING_COMPLETED,
                DownloadEventType.FILING_FAILED,
            }:
                job.stage = "downloading_filings"
                issue = _build_download_recent_issue(
                    document_id=event.document_id,
                    payload=payload,
                )
                if issue is not None:
                    _append_recent_issue_locked(job, issue)
                job.progress_completed += 1
                return
            if event.event_type == DownloadEventType.PIPELINE_COMPLETED:
                job.stage = "finalizing"
                result = payload.get("result")
                if not isinstance(result, dict):
                    self._finalize_job_locked(
                        job=job,
                        status="failed",
                        failure={"code": "execution_error", "message": "download 缺少最终结果", "retryable": True},
                        result_summary=None,
                        final_result=None,
                    )
                    return
                job.progress_total = int_or_zero(
                    result.get("summary", {}).get("total") if isinstance(result.get("summary"), dict) else None
                )
                status, failure = _resolve_terminal_status(
                    final_result=result,
                    cancel_requested=job.cancel_requested,
                )
                self._finalize_job_locked(
                    job=job,
                    status=status,
                    failure=failure,
                    result_summary=_build_download_result_summary(result),
                    final_result=result,
                )

    def _update_process_job_from_event(self, *, job_id: str, event: ProcessEvent) -> None:
        """根据预处理事件更新 job。"""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            payload = event.payload if isinstance(event.payload, dict) else {}
            if event.event_type == ProcessEventType.PIPELINE_STARTED:
                job.stage = "scanning_documents"
                total_documents = payload.get("total_documents")
                if isinstance(total_documents, int):
                    job.progress_total = total_documents
                return
            if event.event_type == ProcessEventType.DOCUMENT_STARTED:
                source_kind = str(payload.get("source_kind", "")).strip().lower()
                job.stage = "processing_materials" if source_kind == "material" else "processing_filings"
                return
            if event.event_type in {
                ProcessEventType.DOCUMENT_COMPLETED,
                ProcessEventType.DOCUMENT_FAILED,
                ProcessEventType.DOCUMENT_SKIPPED,
            }:
                source_kind = str(payload.get("source_kind", "")).strip().lower()
                job.stage = "processing_materials" if source_kind == "material" else "processing_filings"
                job.progress_completed += 1
                return
            if event.event_type == ProcessEventType.PIPELINE_COMPLETED:
                job.stage = "finalizing"
                result = payload.get("result")
                if not isinstance(result, dict):
                    self._finalize_job_locked(
                        job=job,
                        status="failed",
                        failure={"code": "execution_error", "message": "process 缺少最终结果", "retryable": True},
                        result_summary=None,
                        final_result=None,
                    )
                    return
                job.progress_total = _resolve_process_total(result)
                status, failure = _resolve_terminal_status(
                    final_result=result,
                    cancel_requested=job.cancel_requested,
                )
                self._finalize_job_locked(
                    job=job,
                    status=status,
                    failure=failure,
                    result_summary=_build_process_result_summary(result),
                    final_result=result,
                )

    def _finalize_job_locked(
        self,
        *,
        job: _IngestionJob,
        status: JobStatus,
        failure: Optional[dict[str, Any]],
        result_summary: Optional[dict[str, Any]],
        final_result: Optional[dict[str, Any]],
    ) -> None:
        """写入 job 终态。"""

        job.status = status
        job.failure = failure
        job.result_summary = result_summary
        job.final_result = final_result
        job.finished_at = now_iso8601()

    def _mark_job_failed(self, *, job_id: str, code: str, message: str) -> None:
        """将 job 标记为失败。"""

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            self._finalize_job_locked(
                job=job,
                status="failed",
                failure={"code": code, "message": message, "retryable": True},
                result_summary=job.result_summary,
                final_result=job.final_result,
            )

    def _build_snapshot_locked(self, job: _IngestionJob) -> dict[str, Any]:
        """构建 job 快照。"""

        percent: Optional[int] = None
        if job.progress_total is not None:
            if job.progress_total <= 0:
                percent = 100 if job.status in {"succeeded", "cancelled"} else 0
            else:
                percent = min(100, int((job.progress_completed / job.progress_total) * 100))
        return {
            "job": {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "ticker": job.ticker,
                "status": job.status,
                "stage": job.stage,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
            },
            "progress": {
                "unit": job.progress_unit,
                "completed": job.progress_completed,
                "total": job.progress_total,
                "percent": percent,
            },
            "result_summary": job.result_summary,
            "failure": job.failure,
            "recent_issues": list(job.recent_issues),
        }

    def _cleanup_expired_jobs_locked(self) -> None:
        """清理超时终态 job。"""

        now = datetime.now(timezone.utc)
        expired_ids: list[str] = []
        for job_id, job in self._jobs.items():
            if job.status not in {"succeeded", "failed", "cancelled"}:
                continue
            if not job.finished_at:
                continue
            finished_at = _parse_iso8601(job.finished_at)
            if finished_at is None:
                continue
            if now - finished_at >= _TERMINAL_JOB_TTL:
                expired_ids.append(job_id)
        for job_id in expired_ids:
            self._jobs.pop(job_id, None)


_MANAGER_REGISTRY: dict[str, IngestionJobManager] = {}
_MANAGER_REGISTRY_LOCK = threading.Lock()


def get_or_create_ingestion_job_manager(
    *,
    manager_key: str,
    service_factory: IngestionServiceFactory,
) -> IngestionJobManager:
    """获取或创建全局 job 管理器。

    Args:
        manager_key: 管理器标识。
        service_factory: 服务工厂。

    Returns:
        全局共享的 job 管理器。
    """

    with _MANAGER_REGISTRY_LOCK:
        manager = _MANAGER_REGISTRY.get(manager_key)
        if manager is not None:
            return manager
        manager = IngestionJobManager(service_factory=service_factory, manager_key=manager_key)
        _MANAGER_REGISTRY[manager_key] = manager
        return manager


def _make_request_fingerprint(*, job_type: JobType, request_payload: dict[str, Any]) -> str:
    """构建规范化请求指纹。"""

    return json.dumps(
        {
            "job_type": job_type,
            "request": request_payload,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _coerce_optional_document_ids(value: object) -> Optional[list[str]]:
    """把请求载荷中的 document_ids 规范化为字符串列表。

    Args:
        value: 原始请求载荷中的 document_ids 字段。

    Returns:
        规范化后的文档 ID 列表；会去空、去重并排序；缺失时返回 `None`。

    Raises:
        无。
    """

    if value is None:
        return None
    if not isinstance(value, list):
        return None
    normalized_set = {str(item).strip() for item in value if str(item).strip()}
    if not normalized_set:
        return None
    return sorted(normalized_set)


def _build_download_recent_issue(
    *,
    document_id: Optional[str],
    payload: dict[str, Any],
) -> Optional[dict[str, Any]]:
    """从 download 事件中提取面向 LLM 的最近问题摘要。

    Args:
        document_id: 事件上的文档 ID。
        payload: 事件负载。

    Returns:
        需要暴露时返回极简 issue 字典，否则返回 `None`。

    Raises:
        无。
    """

    filing_result = payload.get("filing_result")
    if isinstance(filing_result, dict):
        source = filing_result
    else:
        source = payload
    status = _first_non_empty_text(source.get("status"))
    reason_code = _first_non_empty_text(
        source.get("reason_code"),
        source.get("skip_reason"),
    )
    reason_message = _first_non_empty_text(
        source.get("reason_message"),
        source.get("error"),
    )
    if reason_message is None:
        failed_files = source.get("failed_files")
        if isinstance(failed_files, list):
            reason_message = _summarize_failed_file_errors(failed_files)
    if status not in {"skipped", "failed"} and reason_code is None and reason_message is None:
        return None
    resolved_document_id = _first_non_empty_text(source.get("document_id"), document_id)
    return {
        "document_id": resolved_document_id,
        "status": status or "unknown",
        "reason_code": reason_code,
        "reason_message": reason_message,
    }


def _append_recent_issue_locked(job: _IngestionJob, issue: dict[str, Any]) -> None:
    """向 job 追加最近问题，并限制保留窗口。

    Args:
        job: 目标 job。
        issue: 已规范化的问题摘要。

    Returns:
        无。

    Raises:
        无。
    """

    job.recent_issues.append(issue)
    if len(job.recent_issues) > 5:
        job.recent_issues = job.recent_issues[-5:]


def _summarize_failed_file_errors(failed_files: list[dict[str, Any]]) -> Optional[str]:
    """汇总文件级失败消息，供 job 摘要使用。

    Args:
        failed_files: 失败文件列表。

    Returns:
        简短失败说明；无可用信息时返回 `None`。

    Raises:
        无。
    """

    messages: list[str] = []
    for item in failed_files:
        if not isinstance(item, dict):
            continue
        message = _first_non_empty_text(
            item.get("reason_message"),
            item.get("message"),
            item.get("error"),
        )
        if message is None or message in messages:
            continue
        messages.append(message)
    if not messages:
        return None
    if len(messages) == 1:
        return messages[0]
    preview = "；".join(messages[:2])
    if len(messages) <= 2:
        return preview
    return f"{preview} 等{len(messages)}项"


def _first_non_empty_text(*values: Any) -> Optional[str]:
    """返回第一个非空白文本值。

    Args:
        *values: 候选文本值。

    Returns:
        第一个非空白字符串；均为空时返回 `None`。

    Raises:
        无。
    """

    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            return normalized
    return None


def _resolve_terminal_status(
    *,
    final_result: dict[str, Any],
    cancel_requested: bool,
) -> tuple[JobStatus, Optional[dict[str, Any]]]:
    """根据最终结果解析终态。"""

    if cancel_requested:
        return "cancelled", None
    result_status = str(final_result.get("status", "")).strip().lower()
    if result_status == "not_implemented":
        message = str(final_result.get("message", "")).strip() or "当前市场/管线不支持该操作"
        return "failed", {"code": "not_supported", "message": message, "retryable": False}
    if result_status == "failed":
        message = str(final_result.get("reason", "") or final_result.get("message", "")).strip() or "执行失败"
        return "failed", {"code": "execution_error", "message": message, "retryable": True}
    return "succeeded", None


def _build_download_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    """构建下载最小结果摘要。"""

    summary = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
    filings = result.get("filings", [])
    filing_list = filings if isinstance(filings, list) else []
    return {
        "filings_total": int_or_zero(summary.get("total")),
        "filings_completed": sum(1 for item in filing_list if str(item.get("status", "")).strip() in {"downloaded", "skipped"}),
        "filings_failed": sum(1 for item in filing_list if str(item.get("status", "")).strip() == "failed"),
        "files_downloaded": sum(int(item.get("downloaded_files", 0) or 0) for item in filing_list if isinstance(item, dict)),
    }


def _build_process_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    """构建预处理最小结果摘要。"""

    filing_summary = result.get("filing_summary", {}) if isinstance(result.get("filing_summary"), dict) else {}
    material_summary = result.get("material_summary", {}) if isinstance(result.get("material_summary"), dict) else {}
    return {
        "filings_total": int_or_zero(filing_summary.get("total")),
        "filings_processed": int_or_zero(filing_summary.get("processed")),
        "filings_skipped": int_or_zero(filing_summary.get("skipped")),
        "filings_failed": int_or_zero(filing_summary.get("failed")),
        "materials_total": int_or_zero(material_summary.get("total")),
        "materials_processed": int_or_zero(material_summary.get("processed")),
        "materials_skipped": int_or_zero(material_summary.get("skipped")),
        "materials_failed": int_or_zero(material_summary.get("failed")),
    }


def _resolve_process_total(result: dict[str, Any]) -> Optional[int]:
    """解析预处理总文档数。"""

    filing_summary = result.get("filing_summary", {}) if isinstance(result.get("filing_summary"), dict) else {}
    material_summary = result.get("material_summary", {}) if isinstance(result.get("material_summary"), dict) else {}
    filing_total = int_or_zero(filing_summary.get("total"))
    material_total = int_or_zero(material_summary.get("total"))
    return filing_total + material_total


def _parse_iso8601(value: str) -> Optional[datetime]:
    """解析 ISO8601 文本。"""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None
