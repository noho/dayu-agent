"""FinsService 测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Callable, cast

import pytest

from dayu.contracts.cancellation import CancellationToken
from dayu.contracts.fins import (
    DownloadCommandPayload,
    DownloadProgressPayload,
    DownloadResultData,
    DownloadSummary,
    FinsCommand,
    FinsCommandName,
    FinsEvent,
    FinsEventType,
    FinsProgressEventName,
    FinsResult,
    ProcessCommandPayload,
    ProcessFilingCommandPayload,
    ProcessMaterialCommandPayload,
    ProcessSingleResultData,
    UploadFilingCommandPayload,
    UploadFilingsFromCommandPayload,
    UploadMaterialCommandPayload,
)
from dayu.contracts.session import SessionSource
from dayu.fins.domain.document_models import FilingSummary
from dayu.fins.service_runtime import DefaultFinsRuntime
from dayu.host.host import Host
from dayu.host.host_execution import HostedRunContext, HostedRunSpec
from dayu.fins.ingestion.factory import IngestionServiceFactory
from dayu.fins.processors.registry import build_fins_processor_registry
from dayu.fins.tools.service import FinsToolService
from dayu.services.contracts import FinsSubmitRequest
from dayu.services.concurrency_lanes import LANE_CN_DOWNLOAD, LANE_HK_DOWNLOAD, LANE_SEC_DOWNLOAD
from dayu.services.fins_service import FinsService
from dayu.services.protocols import FinsServiceProtocol
from tests.application.conftest import StubHostExecutor, StubSessionRegistry


class _FakeFinsRuntime:
    """测试用 FinsRuntime。"""

    def validate_command(self, command: FinsCommand) -> None:
        """测试桩默认接受命令。"""

        del command

    def get_processor_registry(self):
        """返回测试用 processor registry。"""

        return build_fins_processor_registry()

    def get_tool_service(self, *, processor_cache_max_entries: int = 128) -> FinsToolService:
        """测试中不应调用该分支。"""

        del processor_cache_max_entries
        raise AssertionError("当前测试不应调用 get_tool_service")

    def build_ingestion_service_factory(self) -> IngestionServiceFactory:
        """测试中不应调用该分支。"""

        raise AssertionError("当前测试不应调用 build_ingestion_service_factory")

    def get_ingestion_manager_key(self) -> str:
        """返回稳定测试 key。"""

        return "test-ingestion-manager"

    def get_company_name(self, ticker: str) -> str:
        """返回测试公司名。"""

        return ticker.strip().upper()

    def get_company_meta_summary(self, ticker: str) -> dict[str, str]:
        """返回测试公司摘要。"""

        normalized = ticker.strip().upper()
        return {"ticker": normalized, "company_name": normalized}

    def list_source_filings(self, ticker: str) -> list[FilingSummary]:
        """返回测试用财报摘要列表。"""

        del ticker
        return []

    def execute(
        self,
        command: FinsCommand,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> FinsResult | AsyncIterator[FinsEvent]:
        """执行测试命令。"""

        del cancel_checker
        if command.stream:
            return self._execute_stream(command)
        return FinsResult(
            command=command.name,
            data=DownloadResultData(
                pipeline="fake",
                status="ok",
                ticker=command.payload.ticker,
                summary=DownloadSummary(total=0, downloaded=0, skipped=0, failed=0),
            ),
        )

    async def _execute_stream(self, command: FinsCommand) -> AsyncIterator[FinsEvent]:
        """返回固定事件流。"""

        yield FinsEvent(
            type=FinsEventType.PROGRESS,
            command=command.name,
            payload=DownloadProgressPayload(
                event_type=FinsProgressEventName.PIPELINE_STARTED,
                ticker=command.payload.ticker,
            ),
        )
        yield FinsEvent(
            type=FinsEventType.RESULT,
            command=command.name,
            payload=DownloadResultData(
                pipeline="fake",
                status="ok",
                ticker=command.payload.ticker,
                summary=DownloadSummary(total=0, downloaded=0, skipped=0, failed=0),
            ),
        )


class _CancelAwareFinsRuntime(_FakeFinsRuntime):
    """用于验证取消检查函数透传的 runtime。"""

    def __init__(self) -> None:
        """初始化测试状态。"""

        self.received_cancel_checker: Callable[[], bool] | None = None
        self.observed_cancelled = False

    def execute(
        self,
        command: FinsCommand,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> FinsResult | AsyncIterator[FinsEvent]:
        """记录 cancel_checker，并在同步命令中观察取消状态。"""

        self.received_cancel_checker = cancel_checker
        self.observed_cancelled = bool(cancel_checker is not None and cancel_checker())
        if command.stream:
            return self._execute_stream(command)
        return FinsResult(
            command=command.name,
            data=ProcessSingleResultData(
                pipeline="fake",
                action="process_filing",
                status="cancelled" if self.observed_cancelled else "ok",
                ticker=command.payload.ticker,
                document_id="fil_0001",
            ),
        )


class _CancelledSyncExecutor:
    """在同步执行前预先取消 token 的 HostExecutor。"""

    def __init__(self) -> None:
        """初始化测试状态。"""

        self.last_spec: HostedRunSpec | None = None
        self.sync_call_count = 0

    def run_operation_sync(
        self,
        *,
        spec: HostedRunSpec,
        operation: Callable[[HostedRunContext], FinsResult],
        on_cancel: Callable[[], FinsResult] | None = None,
    ) -> FinsResult:
        """构造已取消上下文并执行同步操作。"""

        del on_cancel
        self.last_spec = spec
        self.sync_call_count += 1
        token = CancellationToken()
        token.cancel()
        context = HostedRunContext(run_id="run_test", cancellation_token=token)
        return operation(context)


class _CancelledStreamExecutor:
    """在流式执行前预先取消 token 的 HostExecutor。"""

    def __init__(self) -> None:
        """初始化测试状态。"""

        self.last_spec: HostedRunSpec | None = None
        self.stream_call_count = 0

    async def run_operation_stream(
        self,
        *,
        spec: HostedRunSpec,
        event_stream_factory: Callable[[HostedRunContext], AsyncIterator[FinsEvent]],
    ) -> AsyncIterator[FinsEvent]:
        """构造已取消上下文并执行流式操作。"""

        self.last_spec = spec
        self.stream_call_count += 1
        token = CancellationToken()
        token.cancel()
        context = HostedRunContext(run_id="run_test", cancellation_token=token)
        async for event in event_stream_factory(context):
            yield event



@pytest.mark.unit
def test_list_filings_delegates_to_runtime() -> None:
    """验证 list_filings 会透传到 runtime.list_source_filings。"""

    class _RecordingFinsRuntime(_FakeFinsRuntime):
        """记录 list_source_filings 调用。"""

        def __init__(self) -> None:
            super().__init__()
            self.received_ticker: str | None = None

        def list_source_filings(self, ticker: str) -> list[FilingSummary]:
            self.received_ticker = ticker
            return [
                FilingSummary(
                    document_id="fil_0001",
                    form_type="10-K",
                    filing_date="2024-01-31",
                    report_date="2023-12-31",
                    fiscal_year=2023,
                    fiscal_period="FY",
                    is_deleted=False,
                    primary_file_name="primary.pdf",
                    primary_file_path="/ws/AAPL/filings/fil_0001/primary.pdf",
                ),
            ]

    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=StubSessionRegistry(),  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    runtime = _RecordingFinsRuntime()
    service = FinsService(
        host=host,
        fins_runtime=runtime,
    )
    result = service.list_filings("AAPL")

    assert runtime.received_ticker == "AAPL"
    assert len(result) == 1
    assert result[0].document_id == "fil_0001"
    assert result[0].form_type == "10-K"


@pytest.mark.unit
def test_list_filings_empty_ticker_returns_empty() -> None:
    """验证 list_filings 对空 ticker 返回空列表。"""

    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=StubSessionRegistry(),  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=_FakeFinsRuntime(),
    )
    result = service.list_filings("")
    assert result == []
@pytest.mark.unit
def test_execute_sync_download() -> None:
    """验证同步路径会透传到 runtime。"""

    def _executor(host_obj: Host) -> StubHostExecutor:
        """把 Host 内部 executor 收窄到测试 stub。"""

        return cast(StubHostExecutor, host_obj._executor)

    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=StubSessionRegistry(),  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=_FakeFinsRuntime(),
    )
    result = service.execute(
        FinsCommand(
            name=FinsCommandName.DOWNLOAD,
            payload=DownloadCommandPayload(ticker="AAPL"),
            stream=False,
        )
    )

    assert isinstance(result, FinsResult)
    assert result.data.ticker == "AAPL"
    last_spec = _executor(host).last_spec
    assert last_spec is not None
    assert last_spec.metadata == {}
    assert last_spec.business_concurrency_lane == LANE_SEC_DOWNLOAD


@pytest.mark.unit
def test_execute_cn_download_uses_cn_download_hosted_lane() -> None:
    """CN download 应在 HostedRun 外层占用独立 CN 下载 lane。"""

    def _executor(host_obj: Host) -> StubHostExecutor:
        """把 Host 内部 executor 收窄到测试 stub。"""

        return cast(StubHostExecutor, host_obj._executor)

    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=StubSessionRegistry(),  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=_FakeFinsRuntime(),
    )
    result = service.execute(
        FinsCommand(
            name=FinsCommandName.DOWNLOAD,
            payload=DownloadCommandPayload(ticker="600519"),
            stream=False,
        )
    )

    assert isinstance(result, FinsResult)
    last_spec = _executor(host).last_spec
    assert last_spec is not None
    assert last_spec.business_concurrency_lane == LANE_CN_DOWNLOAD


@pytest.mark.unit
def test_execute_hk_download_uses_hk_download_hosted_lane() -> None:
    """HK download 应在 HostedRun 外层占用独立 HK 下载 lane。"""

    def _executor(host_obj: Host) -> StubHostExecutor:
        """把 Host 内部 executor 收窄到测试 stub。"""

        return cast(StubHostExecutor, host_obj._executor)

    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=StubSessionRegistry(),  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=_FakeFinsRuntime(),
    )
    result = service.execute(
        FinsCommand(
            name=FinsCommandName.DOWNLOAD,
            payload=DownloadCommandPayload(ticker="0700"),
            stream=False,
        )
    )

    assert isinstance(result, FinsResult)
    last_spec = _executor(host).last_spec
    assert last_spec is not None
    assert last_spec.business_concurrency_lane == LANE_HK_DOWNLOAD


@pytest.mark.unit
def test_execute_stream_download() -> None:
    """验证流式路径会透传到 runtime。"""

    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=StubSessionRegistry(),  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=_FakeFinsRuntime(),
    )

    async def _collect() -> list[FinsEvent]:
        events: list[FinsEvent] = []
        stream = service.execute(
            FinsCommand(
                name=FinsCommandName.DOWNLOAD,
                payload=DownloadCommandPayload(ticker="AAPL"),
                stream=True,
            )
        )
        async for event in cast(AsyncIterator[FinsEvent], stream):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    assert events[0].type == FinsEventType.PROGRESS
    assert events[-1].type == FinsEventType.RESULT
    assert isinstance(events[-1].payload, DownloadResultData)
    assert events[-1].payload.ticker == "AAPL"


@pytest.mark.unit
def test_fins_service_implements_protocol() -> None:
    """验证 FinsService 满足 FinsServiceProtocol 协议。"""

    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=StubSessionRegistry(),  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=_FakeFinsRuntime(),
    )
    assert isinstance(service, FinsServiceProtocol)


@pytest.mark.unit
def test_execute_reuses_explicit_session_id() -> None:
    """显式 session_id 时应复用既有 session。"""

    session_registry = StubSessionRegistry()
    session = session_registry.create_session(SessionSource.WEB, session_id="session_fins")
    host_executor = StubHostExecutor()
    host = Host(
        executor=host_executor,  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=_FakeFinsRuntime(),
    )

    result = service.execute(
        FinsCommand(
            name=FinsCommandName.DOWNLOAD,
            payload=DownloadCommandPayload(ticker="AAPL"),
            stream=False,
            session_id=session.session_id,
        ),
    )

    assert isinstance(result, FinsResult)
    assert host_executor.last_spec is not None
    assert host_executor.last_spec.session_id == session.session_id


@pytest.mark.unit
def test_submit_rejects_download_empty_ticker_before_creating_session(tmp_path: Path) -> None:
    """验证 submit 会在创建 session 前同步拒绝空 download ticker。"""

    session_registry = StubSessionRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=DefaultFinsRuntime.create(workspace_root=tmp_path),
    )
    before_sessions = session_registry.list_sessions()

    with pytest.raises(ValueError, match="ticker 不能为空"):
        service.submit(
            FinsSubmitRequest(
                command=FinsCommand(
                    name=FinsCommandName.DOWNLOAD,
                    payload=DownloadCommandPayload(ticker="   "),
                    stream=True,
                )
            )
        )

    after_sessions = session_registry.list_sessions()
    assert [session.session_id for session in after_sessions] == [session.session_id for session in before_sessions]


@pytest.mark.unit
def test_submit_rejects_process_empty_ticker_before_creating_session(tmp_path: Path) -> None:
    """验证 submit 会在创建 session 前同步拒绝空 process ticker。"""

    session_registry = StubSessionRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=DefaultFinsRuntime.create(workspace_root=tmp_path),
    )
    before_sessions = session_registry.list_sessions()

    with pytest.raises(ValueError, match="ticker 不能为空"):
        service.submit(
            FinsSubmitRequest(
                command=FinsCommand(
                    name=FinsCommandName.PROCESS,
                    payload=ProcessCommandPayload(ticker="   "),
                    stream=True,
                )
            )
        )

    after_sessions = session_registry.list_sessions()
    assert [session.session_id for session in after_sessions] == [session.session_id for session in before_sessions]


@pytest.mark.unit
def test_submit_rejects_upload_filing_without_files_before_creating_session(tmp_path: Path) -> None:
    """验证 submit 会在创建 session 前同步拒绝缺少 files 的 upload_filing。"""

    session_registry = StubSessionRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=DefaultFinsRuntime.create(workspace_root=tmp_path),
    )
    before_sessions = session_registry.list_sessions()

    with pytest.raises(ValueError, match="必须提供 --files"):
        service.submit(
            FinsSubmitRequest(
                command=FinsCommand(
                    name=FinsCommandName.UPLOAD_FILING,
                    payload=UploadFilingCommandPayload(
                        ticker="AAPL",
                        files=(),
                        fiscal_year=2024,
                        action="create",
                        company_id="0000320193",
                        company_name="Apple Inc",
                    ),
                )
            )
        )

    after_sessions = session_registry.list_sessions()
    assert [s.session_id for s in after_sessions] == [s.session_id for s in before_sessions]


@pytest.mark.unit
def test_submit_rejects_upload_material_invalid_form_type_before_creating_session(tmp_path: Path) -> None:
    """验证 submit 会在创建 session 前同步拒绝非法 form_type 的 upload_material。"""

    session_registry = StubSessionRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=DefaultFinsRuntime.create(workspace_root=tmp_path),
    )
    before_sessions = session_registry.list_sessions()

    with pytest.raises(ValueError, match="不是合法的 material form_type"):
        service.submit(
            FinsSubmitRequest(
                command=FinsCommand(
                    name=FinsCommandName.UPLOAD_MATERIAL,
                    payload=UploadMaterialCommandPayload(
                        ticker="AAPL",
                        files=(Path("/tmp/test.pdf"),),
                        form_type="INVALID_TYPE",
                        action="create",
                        company_id="0000320193",
                        company_name="Apple Inc",
                    ),
                )
            )
        )

    after_sessions = session_registry.list_sessions()
    assert [s.session_id for s in after_sessions] == [s.session_id for s in before_sessions]


@pytest.mark.unit
def test_submit_rejects_unsupported_stream_command_before_creating_session(tmp_path: Path) -> None:
    """验证 submit 会在创建 session 前同步拒绝不支持流式的命令。"""

    session_registry = StubSessionRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=DefaultFinsRuntime.create(workspace_root=tmp_path),
    )
    before_sessions = session_registry.list_sessions()

    with pytest.raises(ValueError, match="不支持流式执行的命令"):
        service.submit(
            FinsSubmitRequest(
                command=FinsCommand(
                    name=FinsCommandName.PROCESS_FILING,
                    payload=ProcessFilingCommandPayload(ticker="AAPL", document_id="doc_001"),
                    stream=True,
                )
            )
        )

    after_sessions = session_registry.list_sessions()
    assert [s.session_id for s in after_sessions] == [s.session_id for s in before_sessions]


@pytest.mark.unit
def test_submit_rejects_process_filing_empty_ticker_before_creating_session(tmp_path: Path) -> None:
    """验证 submit 会在创建 session 前同步拒绝空 process_filing ticker。"""

    session_registry = StubSessionRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=DefaultFinsRuntime.create(workspace_root=tmp_path),
    )
    before_sessions = session_registry.list_sessions()

    with pytest.raises(ValueError):
        service.submit(
            FinsSubmitRequest(
                command=FinsCommand(
                    name=FinsCommandName.PROCESS_FILING,
                    payload=ProcessFilingCommandPayload(ticker="   ", document_id="doc_001"),
                )
            )
        )

    after_sessions = session_registry.list_sessions()
    assert [s.session_id for s in after_sessions] == [s.session_id for s in before_sessions]


@pytest.mark.unit
def test_submit_rejects_process_material_empty_ticker_before_creating_session(tmp_path: Path) -> None:
    """验证 submit 会在创建 session 前同步拒绝空 process_material ticker。"""

    session_registry = StubSessionRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=DefaultFinsRuntime.create(workspace_root=tmp_path),
    )
    before_sessions = session_registry.list_sessions()

    with pytest.raises(ValueError):
        service.submit(
            FinsSubmitRequest(
                command=FinsCommand(
                    name=FinsCommandName.PROCESS_MATERIAL,
                    payload=ProcessMaterialCommandPayload(ticker="   ", document_id="doc_001"),
                )
            )
        )

    after_sessions = session_registry.list_sessions()
    assert [s.session_id for s in after_sessions] == [s.session_id for s in before_sessions]


@pytest.mark.unit
def test_submit_rejects_upload_filings_from_nonexistent_source_dir(tmp_path: Path) -> None:
    """验证 submit 会在创建 session 前同步拒绝不存在的 source_dir。"""

    session_registry = StubSessionRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=DefaultFinsRuntime.create(workspace_root=tmp_path),
    )
    before_sessions = session_registry.list_sessions()

    with pytest.raises(FileNotFoundError, match="source_dir 不存在或不是目录"):
        service.submit(
            FinsSubmitRequest(
                command=FinsCommand(
                    name=FinsCommandName.UPLOAD_FILINGS_FROM,
                    payload=UploadFilingsFromCommandPayload(
                        ticker="AAPL",
                        source_dir=tmp_path / "nonexistent",
                        company_id="0000320193",
                        company_name="Apple Inc",
                    ),
                )
            )
        )

    after_sessions = session_registry.list_sessions()
    assert [s.session_id for s in after_sessions] == [s.session_id for s in before_sessions]


@pytest.mark.unit
def test_submit_rejects_process_filing_empty_document_id(tmp_path: Path) -> None:
    """验证 submit 会在创建 session 前同步拒绝空 document_id。"""

    session_registry = StubSessionRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=DefaultFinsRuntime.create(workspace_root=tmp_path),
    )
    before_sessions = session_registry.list_sessions()

    with pytest.raises(ValueError, match="document_id 不能为空"):
        service.submit(
            FinsSubmitRequest(
                command=FinsCommand(
                    name=FinsCommandName.PROCESS_FILING,
                    payload=ProcessFilingCommandPayload(ticker="AAPL", document_id="   "),
                )
            )
        )

    after_sessions = session_registry.list_sessions()
    assert [s.session_id for s in after_sessions] == [s.session_id for s in before_sessions]


@pytest.mark.unit
def test_submit_rejects_process_material_empty_document_id(tmp_path: Path) -> None:
    """验证 submit 会在创建 session 前同步拒绝空 document_id。"""

    session_registry = StubSessionRegistry()
    host = Host(
        executor=StubHostExecutor(),  # type: ignore[arg-type]
        session_registry=session_registry,  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=DefaultFinsRuntime.create(workspace_root=tmp_path),
    )
    before_sessions = session_registry.list_sessions()

    with pytest.raises(ValueError, match="document_id 不能为空"):
        service.submit(
            FinsSubmitRequest(
                command=FinsCommand(
                    name=FinsCommandName.PROCESS_MATERIAL,
                    payload=ProcessMaterialCommandPayload(ticker="AAPL", document_id=""),
                )
            )
        )

    after_sessions = session_registry.list_sessions()
    assert [s.session_id for s in after_sessions] == [s.session_id for s in before_sessions]


@pytest.mark.unit
def test_execute_sync_process_filing_passes_host_cancel_checker() -> None:
    """验证同步单文档处理会把 Host 取消状态透传给 runtime。"""

    runtime = _CancelAwareFinsRuntime()
    host = Host(
        executor=_CancelledSyncExecutor(),  # type: ignore[arg-type]
        session_registry=StubSessionRegistry(),  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=runtime,
    )

    result = service.execute(
        FinsCommand(
            name=FinsCommandName.PROCESS_FILING,
            payload=ProcessFilingCommandPayload(ticker="AAPL", document_id="fil_0001"),
            stream=False,
        )
    )

    assert isinstance(result, FinsResult)
    assert isinstance(result.data, ProcessSingleResultData)
    assert runtime.received_cancel_checker is not None
    assert runtime.observed_cancelled is True
    assert result.data.status == "cancelled"


@pytest.mark.unit
def test_execute_stream_download_passes_host_cancel_checker() -> None:
    """验证流式 direct operation 也会把 Host 取消状态透传给 runtime。"""

    runtime = _CancelAwareFinsRuntime()
    host = Host(
        executor=_CancelledStreamExecutor(),  # type: ignore[arg-type]
        session_registry=StubSessionRegistry(),  # type: ignore[arg-type]
        run_registry=object(),  # type: ignore[arg-type]
    )
    service = FinsService(
        host=host,
        fins_runtime=runtime,
    )

    async def _collect() -> list[FinsEvent]:
        events: list[FinsEvent] = []
        stream = service.execute(
            FinsCommand(
                name=FinsCommandName.DOWNLOAD,
                payload=DownloadCommandPayload(ticker="AAPL"),
                stream=True,
            )
        )
        async for event in cast(AsyncIterator[FinsEvent], stream):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert runtime.received_cancel_checker is not None
    assert runtime.observed_cancelled is True
    assert [event.type for event in events] == [FinsEventType.PROGRESS, FinsEventType.RESULT]
