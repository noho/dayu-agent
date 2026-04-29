"""download_progress 模块测试。"""

from __future__ import annotations

from queue import Queue

import pytest

from dayu.contracts.fins import (
    DownloadFilingResultItem,
    DownloadFilingResultStatus,
    DownloadProgressPayload,
    DownloadResultData,
    FinsCommandName,
    FinsEvent,
    FinsEventType,
    FinsProgressEventName,
)
from dayu.services.contracts import FinsSubmission
from dayu.web.streamlit.pages.filing.download_progress import (
    DownloadQueueEvent,
    DownloadStatus,
    DownloadTaskState,
    apply_download_completion,
    apply_download_progress,
    create_download_task,
    run_download_stream_worker,
)


# ── DownloadStatus ────────────────────────────────────────────────────


@pytest.mark.unit
def test_download_status_values() -> None:
    """DownloadStatus 应包含四种状态枚举值。"""
    assert DownloadStatus.PENDING == "pending"
    assert DownloadStatus.RUNNING == "running"
    assert DownloadStatus.COMPLETED == "completed"
    assert DownloadStatus.FAILED == "failed"


# ── DownloadQueueEvent ────────────────────────────────────────────────


@pytest.mark.unit
def test_download_queue_event_defaults() -> None:
    """DownloadQueueEvent payload 和 message 默认值应为 None / 空字符串。"""
    event = DownloadQueueEvent(kind="done")
    assert event.kind == "done"
    assert event.payload is None
    assert event.message == ""


@pytest.mark.unit
def test_download_queue_event_with_payload() -> None:
    """DownloadQueueEvent 可携带进度负载。"""
    payload = DownloadProgressPayload(
        event_type=FinsProgressEventName.PIPELINE_STARTED,
        ticker="AAPL",
    )
    event = DownloadQueueEvent(kind="progress", payload=payload, message="开始")
    assert event.kind == "progress"
    assert event.payload is payload
    assert event.message == "开始"


@pytest.mark.unit
def test_download_queue_event_frozen() -> None:
    """DownloadQueueEvent 应为不可变数据类。"""
    event = DownloadQueueEvent(kind="done")
    with pytest.raises(Exception):
        event.kind = "progress"  # type: ignore[misc]


# ── DownloadTaskState ─────────────────────────────────────────────────


@pytest.mark.unit
def test_download_task_state_defaults() -> None:
    """DownloadTaskState 新建实例的默认字段应与定义一致。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    assert task.session_id == "s1"
    assert task.ticker == "AAPL"
    assert task.status == DownloadStatus.PENDING
    assert task.progress == 0.0
    assert task.current_form_type is None
    assert task.current_document_id is None
    assert task.message == "等待开始..."
    assert task.downloaded_count == 0
    assert task.downloaded_filing_count == 0
    assert task.total_count is None
    assert task.errors == []
    assert task.logs == []
    assert task.started_at is None
    assert task.completed_at is None


# ── create_download_task ──────────────────────────────────────────────


@pytest.mark.unit
def test_create_download_task_initializes_running() -> None:
    """create_download_task 应将状态设为 RUNNING 并记录启动时间。"""
    task = create_download_task(session_id="s1", ticker="AAPL")
    assert task.session_id == "s1"
    assert task.ticker == "AAPL"
    assert task.status == DownloadStatus.RUNNING
    assert task.started_at is not None
    assert len(task.logs) == 1
    assert "下载任务已创建" in task.logs[0]["message"]


# ── apply_download_progress ───────────────────────────────────────────


def _make_progress(
    event_type: FinsProgressEventName,
    ticker: str = "AAPL",
    **kwargs: object,
) -> DownloadProgressPayload:
    """构造 DownloadProgressPayload 测试辅助。"""
    return DownloadProgressPayload(
        event_type=event_type,
        ticker=ticker,
        **{k: v for k, v in kwargs.items() if v is not None},  # type: ignore[misc]
    )


@pytest.mark.unit
def test_apply_pipeline_started() -> None:
    """PIPELINE_STARTED 应设置状态为 RUNNING 并追加日志。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(FinsProgressEventName.PIPELINE_STARTED)
    apply_download_progress(task, payload)
    assert task.status == DownloadStatus.RUNNING
    assert "开始下载任务" in task.message


@pytest.mark.unit
def test_apply_company_resolved() -> None:
    """COMPANY_RESOLVED 应记录已解析公司信息日志。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(FinsProgressEventName.COMPANY_RESOLVED)
    apply_download_progress(task, payload)
    assert "已解析公司信息" in task.message


@pytest.mark.unit
def test_apply_filing_started() -> None:
    """FILING_STARTED 应更新 form_type 和 document_id。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(
        FinsProgressEventName.FILING_STARTED,
        form_type="10-K",
        document_id="doc-1",
    )
    apply_download_progress(task, payload)
    assert task.current_form_type == "10-K"
    assert task.current_document_id == "doc-1"
    assert "开始下载: 10-K" in task.logs[-1]["message"]


@pytest.mark.unit
def test_apply_file_downloaded() -> None:
    """FILE_DOWNLOADED 应递增已下载计数并更新进度。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL", total_count=10)
    payload = _make_progress(
        FinsProgressEventName.FILE_DOWNLOADED,
        name="report.pdf",
        size=1024,
    )
    apply_download_progress(task, payload)
    assert task.downloaded_count == 1
    assert "report.pdf" in task.message
    assert "1024" in task.message


@pytest.mark.unit
def test_apply_file_downloaded_no_size() -> None:
    """FILE_DOWNLOADED 在无文件大小时仅显示文件名。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(
        FinsProgressEventName.FILE_DOWNLOADED,
        name="report.pdf",
    )
    apply_download_progress(task, payload)
    assert "report.pdf" in task.message
    assert "字节" not in task.message


@pytest.mark.unit
def test_apply_file_skipped() -> None:
    """FILE_SKIPPED 应记录警告级别日志。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(
        FinsProgressEventName.FILE_SKIPPED,
        name="existing.pdf",
    )
    apply_download_progress(task, payload)
    assert task.logs[-1]["level"] == "warning"
    assert "existing.pdf" in task.logs[-1]["message"]


@pytest.mark.unit
def test_apply_file_failed() -> None:
    """FILE_FAILED 应记录错误日志但不标记任务失败。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(
        FinsProgressEventName.FILE_FAILED,
        name="bad.pdf",
        reason="连接超时",
    )
    apply_download_progress(task, payload)
    assert len(task.errors) == 1
    assert "bad.pdf" in task.errors[0]
    assert "连接超时" in task.errors[0]
    assert task.logs[-1]["level"] == "error"
    # FILE_FAILED 不应标记任务失败
    assert task.status != DownloadStatus.FAILED


@pytest.mark.unit
def test_apply_filing_completed_success() -> None:
    """FILING_COMPLETED 成功时应递增 filing 计数并更新进度。"""
    filing_result = DownloadFilingResultItem(
        document_id="doc-1",
        status=DownloadFilingResultStatus.DOWNLOADED,
        downloaded_files=3,
    )
    task = DownloadTaskState(session_id="s1", ticker="AAPL", total_count=10)
    payload = DownloadProgressPayload(
        event_type=FinsProgressEventName.FILING_COMPLETED,
        ticker="AAPL",
        form_type="10-K",
        filing_result=filing_result,
        file_count=10,
    )
    apply_download_progress(task, payload)
    assert task.downloaded_filing_count == 1
    assert "完成下载 10-K" in task.logs[-1]["message"]


@pytest.mark.unit
def test_apply_filing_completed_skipped() -> None:
    """FILING_COMPLETED 跳过时应记录警告日志。"""
    filing_result = DownloadFilingResultItem(
        document_id="doc-1",
        status=DownloadFilingResultStatus.SKIPPED,
        reason_message="已存在",
    )
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = DownloadProgressPayload(
        event_type=FinsProgressEventName.FILING_COMPLETED,
        ticker="AAPL",
        form_type="10-K",
        filing_result=filing_result,
    )
    apply_download_progress(task, payload)
    assert task.logs[-1]["level"] == "warning"
    assert "跳过下载 10-K" in task.logs[-1]["message"]
    assert "已存在" in task.logs[-1]["message"]


@pytest.mark.unit
def test_apply_filing_completed_failed() -> None:
    """FILING_COMPLETED 失败时应记录错误日志。"""
    filing_result = DownloadFilingResultItem(
        document_id="doc-1",
        status=DownloadFilingResultStatus.FAILED,
        reason_message="网络错误",
    )
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = DownloadProgressPayload(
        event_type=FinsProgressEventName.FILING_COMPLETED,
        ticker="AAPL",
        form_type="10-Q",
        filing_result=filing_result,
    )
    apply_download_progress(task, payload)
    assert task.logs[-1]["level"] == "error"
    assert "下载失败 10-Q" in task.logs[-1]["message"]


@pytest.mark.unit
def test_apply_filing_failed() -> None:
    """FILING_FAILED 应记录错误并标记任务失败。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(
        FinsProgressEventName.FILING_FAILED,
        form_type="10-K",
        reason="文件损坏",
    )
    apply_download_progress(task, payload)
    assert task.status == DownloadStatus.FAILED
    assert len(task.errors) == 1
    assert "文件损坏" in task.errors[0]


@pytest.mark.unit
def test_apply_pipeline_completed() -> None:
    """PIPELINE_COMPLETED 应将进度设为 100% 并标记完成。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(FinsProgressEventName.PIPELINE_COMPLETED)
    apply_download_progress(task, payload)
    assert task.status == DownloadStatus.COMPLETED
    assert task.progress == 100.0
    assert task.completed_at is not None
    assert "下载任务完成" in task.logs[-1]["message"]


@pytest.mark.unit
def test_apply_progress_updates_form_type() -> None:
    """apply_download_progress 应更新 current_form_type。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(
        FinsProgressEventName.FILING_STARTED,
        form_type="10-Q",
    )
    apply_download_progress(task, payload)
    assert task.current_form_type == "10-Q"


@pytest.mark.unit
def test_apply_progress_updates_document_id() -> None:
    """apply_download_progress 应更新 current_document_id。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(
        FinsProgressEventName.FILING_STARTED,
        document_id="doc-42",
        form_type="10-K",
    )
    apply_download_progress(task, payload)
    assert task.current_document_id == "doc-42"


# ── apply_download_completion ─────────────────────────────────────────


@pytest.mark.unit
def test_apply_download_completion_success() -> None:
    """成功终态应设置 COMPLETED、100% 进度和 info 日志。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    apply_download_completion(task, success=True, message="下载完成")
    assert task.status == DownloadStatus.COMPLETED
    assert task.progress == 100.0
    assert task.completed_at is not None
    assert task.message == "下载完成"
    assert task.logs[-1]["level"] == "info"


@pytest.mark.unit
def test_apply_download_completion_failure() -> None:
    """失败终态应设置 FAILED、保持当前进度、记录 error 日志。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL", progress=45.0)
    apply_download_completion(task, success=False, message="任务异常")
    assert task.status == DownloadStatus.FAILED
    assert task.progress == 45.0
    assert task.completed_at is not None
    assert task.message == "任务异常"
    assert task.logs[-1]["level"] == "error"


@pytest.mark.unit
def test_apply_download_completion_default_message() -> None:
    """无 message 时使用任务当前 message。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    apply_download_completion(task, success=True)
    assert task.logs[-1]["level"] == "info"


# ── _update_download_progress_ratio (通过 FILE_DOWNLOADED 间接测试) ──


@pytest.mark.unit
def test_progress_ratio_no_total_count() -> None:
    """total_count 为 None 时不应更新进度百分比。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL")
    payload = _make_progress(
        FinsProgressEventName.FILE_DOWNLOADED,
        name="f1.pdf",
    )
    apply_download_progress(task, payload)
    assert task.progress == 0.0


@pytest.mark.unit
def test_progress_ratio_with_total() -> None:
    """有 total_count 时应正确计算并更新进度百分比。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL", total_count=4)
    payload = _make_progress(
        FinsProgressEventName.FILE_DOWNLOADED,
        name="f1.pdf",
    )
    apply_download_progress(task, payload)
    assert task.progress == 25.0

    payload2 = _make_progress(
        FinsProgressEventName.FILE_DOWNLOADED,
        name="f2.pdf",
    )
    apply_download_progress(task, payload2)
    assert task.progress == 50.0


@pytest.mark.unit
def test_progress_ratio_capped_at_100() -> None:
    """进度百分比不应超过 100。"""
    task = DownloadTaskState(session_id="s1", ticker="AAPL", total_count=2)
    for i in range(5):
        payload = _make_progress(
            FinsProgressEventName.FILE_DOWNLOADED,
            name=f"f{i}.pdf",
        )
        apply_download_progress(task, payload)
    assert task.progress == 100.0
    assert task.downloaded_count == 5


# ── run_download_stream_worker ────────────────────────────────────────


@pytest.mark.unit
def test_run_download_stream_worker_error_sends_error_event() -> None:
    """worker 异常时应向队列写入 error 事件，并最终写入 done 事件。"""

    class _FailingExecution:
        def __aiter__(self) -> _FailingExecution:
            return self

        async def __anext__(self) -> FinsEvent:
            raise RuntimeError("模拟下载异常")

    submission = FinsSubmission(
        session_id="s1",
        execution=_FailingExecution(),  # type: ignore[arg-type]
    )
    queue: Queue[DownloadQueueEvent] = Queue()
    run_download_stream_worker(submission, queue)

    events: list[DownloadQueueEvent] = []
    while not queue.empty():
        events.append(queue.get_nowait())

    error_events = [e for e in events if e.kind == "error"]
    done_events = [e for e in events if e.kind == "done"]
    assert len(error_events) == 1
    assert "模拟下载异常" in error_events[0].message
    assert len(done_events) == 1


@pytest.mark.unit
def test_run_download_stream_worker_sync_result() -> None:
    """同步结果（非 AsyncIterator）应直接写入 result 事件。"""
    from dayu.contracts.fins import (
        DownloadResultData,
        FinsCommandName,
        FinsResult,
    )

    submission = FinsSubmission(
        session_id="s1",
        execution=FinsResult(
            command=FinsCommandName.DOWNLOAD,
            data=DownloadResultData(
                pipeline="sec",
                status="completed",
                ticker="AAPL",
            ),
        ),
    )
    queue: Queue[DownloadQueueEvent] = Queue()
    run_download_stream_worker(submission, queue)

    events: list[DownloadQueueEvent] = []
    while not queue.empty():
        events.append(queue.get_nowait())

    result_events = [e for e in events if e.kind == "result"]
    assert len(result_events) == 1
    assert "同步模式" in result_events[0].message


# ── consume_download_stream_events_to_queue (via worker) ──────────────


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consume_stream_events_forward_progress() -> None:
    """消费流时应将 PROGRESS 事件转发为 DownloadQueueEvent。"""
    from dayu.web.streamlit.pages.filing.download_progress import (
        consume_download_stream_events_to_queue,
    )

    events_to_emit = [
        FinsEvent(
            type=FinsEventType.PROGRESS,
            command=FinsCommandName.DOWNLOAD,
            payload=DownloadProgressPayload(
                event_type=FinsProgressEventName.PIPELINE_STARTED,
                ticker="AAPL",
            ),
        ),
        FinsEvent(
            type=FinsEventType.RESULT,
            command=FinsCommandName.DOWNLOAD,
            payload=DownloadResultData(
                pipeline="sec",
                status="completed",
                ticker="AAPL",
            ),
        ),
    ]

    async def _async_gen():
        for event in events_to_emit:
            yield event

    submission = FinsSubmission(
        session_id="s1",
        execution=_async_gen(),
    )
    queue: Queue[DownloadQueueEvent] = Queue()
    await consume_download_stream_events_to_queue(submission, queue)

    events: list[DownloadQueueEvent] = []
    while not queue.empty():
        events.append(queue.get_nowait())

    progress_events = [e for e in events if e.kind == "progress"]
    result_events = [e for e in events if e.kind == "result"]
    assert len(progress_events) == 1
    assert progress_events[0].payload is not None
    assert len(result_events) == 1
    assert "下载完成" in result_events[0].message


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consume_stream_skips_non_fins_event() -> None:
    """消费流时非 FinsEvent 对象应被跳过。"""
    from dayu.web.streamlit.pages.filing.download_progress import (
        consume_download_stream_events_to_queue,
    )

    fins_event = FinsEvent(
        type=FinsEventType.RESULT,
        command=FinsCommandName.DOWNLOAD,
        payload=DownloadResultData(
            pipeline="sec",
            status="completed",
            ticker="AAPL",
        ),
    )

    async def _async_gen():
        yield "非FinsEvent对象"  # type: ignore[misc]
        yield fins_event

    submission = FinsSubmission(
        session_id="s1",
        execution=_async_gen(),  # type: ignore[arg-type]
    )
    queue: Queue[DownloadQueueEvent] = Queue()
    await consume_download_stream_events_to_queue(submission, queue)

    events: list[DownloadQueueEvent] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    # 只有 RESULT 事件，非 FinsEvent 被跳过
    assert len(events) == 1
    assert events[0].kind == "result"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_consume_stream_progress_non_download_payload() -> None:
    """PROGRESS 事件但 payload 非 DownloadProgressPayload 类型时应跳过。

    使用 PROCESS 命令的 ProcessProgressPayload 模拟：在 DOWNLOAD 流的消费逻辑中，
    isinstance(payload, DownloadProgressPayload) 检查应返回 False，从而跳过该事件。
    """
    from dayu.contracts.fins import ProcessProgressPayload
    from dayu.web.streamlit.pages.filing.download_progress import (
        consume_download_stream_events_to_queue,
    )

    # 构造一个 PROGRESS + DOWNLOAD 命令但 payload 类型不匹配的边界场景
    # 使用 frozen dataclass 的 object.__setattr__ 绕过 __post_init__ 校验
    fins_event = FinsEvent.__new__(FinsEvent)
    object.__setattr__(fins_event, "type", FinsEventType.PROGRESS)
    object.__setattr__(fins_event, "command", FinsCommandName.DOWNLOAD)
    object.__setattr__(
        fins_event,
        "payload",
        ProcessProgressPayload(
            event_type=FinsProgressEventName.PIPELINE_STARTED,
            ticker="AAPL",
        ),
    )

    async def _async_gen():
        yield fins_event
        yield FinsEvent(
            type=FinsEventType.RESULT,
            command=FinsCommandName.DOWNLOAD,
            payload=DownloadResultData(
                pipeline="sec",
                status="completed",
                ticker="AAPL",
            ),
        )

    submission = FinsSubmission(
        session_id="s1",
        execution=_async_gen(),
    )
    queue: Queue[DownloadQueueEvent] = Queue()
    await consume_download_stream_events_to_queue(submission, queue)

    events: list[DownloadQueueEvent] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    # 非 download progress payload 被跳过，只有 result
    assert len(events) == 1
    assert events[0].kind == "result"
