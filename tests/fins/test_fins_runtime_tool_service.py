"""DefaultFinsRuntime.get_tool_service 共享行为测试。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from threading import Barrier, Thread
from typing import Any, AsyncIterator, Optional, cast

import pytest

from dayu.contracts.fins import (
    DownloadCommandPayload,
    FinsCommand,
    FinsCommandName,
    FinsEvent,
    FinsEventType,
    FinsProgressEventName,
    FinsResult,
    ProcessCommandPayload,
    DownloadResultData,
    ProcessResultData,
    UploadFilingCommandPayload,
    UploadFilingsFromCommandPayload,
    UploadMaterialCommandPayload,
    UploadFilingProgressPayload,
    UploadFilingResultData,
    UploadMaterialProgressPayload,
)
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.fins.domain.document_models import CompanyMeta, SourceHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.ingestion.process_events import ProcessEvent, ProcessEventType
from dayu.fins.pipelines.download_events import DownloadEvent, DownloadEventType
from dayu.fins.pipelines.upload_filing_events import UploadFilingEvent, UploadFilingEventType
from dayu.fins.pipelines.upload_material_events import UploadMaterialEvent, UploadMaterialEventType
import dayu.fins.service_runtime as service_runtime_module
from dayu.fins.tools.service import FinsToolService
from dayu.fins.service_runtime import DefaultFinsRuntime
from dayu.fins.storage import DocumentBlobRepositoryProtocol, FilingMaintenanceRepositoryProtocol
from tests.fins.legacy_repository_adapters import build_legacy_repository_adapters


class _StubRepository:
    """仓储桩。"""

    def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
        """返回首个候选 ticker。"""

        return candidates[0] if candidates else None

    def list_document_ids(self, ticker: str, source_kind: Optional[SourceKind] = None) -> list[str]:
        """返回空列表。"""

        del ticker, source_kind
        return []

    def get_document_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
        """抛出异常。"""

        raise FileNotFoundError(document_id)

    def get_source_handle(self, ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
        """抛出异常。"""

        raise FileNotFoundError(document_id)

    def get_primary_source(self, ticker: str, document_id: str, source_kind: SourceKind) -> Any:
        """抛出异常。"""

        raise FileNotFoundError(document_id)

    def get_company_meta(self, ticker: str) -> CompanyMeta:
        """返回公司信息。"""

        return CompanyMeta(
            company_id="1",
            company_name="Test",
            ticker=ticker,
            market="US",
            resolver_version="test",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    def get_processed_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
        """返回空字典。"""

        del ticker, document_id
        return {}


class _StubProcessorRegistry:
    """处理器注册表桩。"""

    def create(self, source: Any, *, form_type: Optional[str] = None, media_type: Optional[str] = None) -> Any:
        """抛出异常。"""

        raise NotImplementedError

    def create_with_fallback(self, source: Any, *, form_type: Optional[str] = None, media_type: Optional[str] = None, on_fallback: Optional[Any] = None) -> Any:
        """抛出异常。"""

        raise NotImplementedError


class _StubBlobRepository:
    """blob 仓储桩。"""


class _StubFilingMaintenanceRepository:
    """filing 维护仓储桩。"""


def _require_sync_result(result: FinsResult | AsyncIterator[FinsEvent]) -> FinsResult:
    """把 runtime.execute 的同步返回值收窄为 FinsResult。

    Args:
        result: runtime.execute 原始返回值。

    Returns:
        同步执行结果对象。

    Raises:
        AssertionError: 当调用方意外拿到流式返回值时抛出。
    """

    if not isinstance(result, FinsResult):
        raise AssertionError("预期同步执行返回 FinsResult")
    return result


def _require_stream_result(result: FinsResult | AsyncIterator[FinsEvent]) -> AsyncIterator[FinsEvent]:
    """把 runtime.execute 的流式返回值收窄为事件流。

    Args:
        result: runtime.execute 原始返回值。

    Returns:
        流式事件迭代器。

    Raises:
        AssertionError: 当调用方意外拿到同步结果时抛出。
    """

    if isinstance(result, FinsResult):
        raise AssertionError("预期流式执行返回 AsyncIterator[FinsEvent]")
    return result


def _make_runtime(tmp_path: Path) -> DefaultFinsRuntime:
    """构建测试用 runtime。"""

    adapters = build_legacy_repository_adapters(_StubRepository())
    return DefaultFinsRuntime(
        workspace_root=tmp_path,
        company_repository=adapters.company_repository,
        source_repository=adapters.source_repository,
        processed_repository=adapters.processed_repository,
        blob_repository=cast(DocumentBlobRepositoryProtocol, _StubBlobRepository()),
        filing_maintenance_repository=cast(FilingMaintenanceRepositoryProtocol, _StubFilingMaintenanceRepository()),
        processor_registry=cast(ProcessorRegistry, _StubProcessorRegistry()),
    )


@pytest.mark.unit
def test_get_tool_service_returns_same_instance(tmp_path: Path) -> None:
    """验证多次调用 get_tool_service 返回同一实例。"""

    runtime = _make_runtime(tmp_path)
    svc1 = runtime.get_tool_service()
    svc2 = runtime.get_tool_service()
    assert svc1 is svc2


@pytest.mark.unit
def test_get_tool_service_returns_fins_tool_service(tmp_path: Path) -> None:
    """验证返回类型为 FinsToolService。"""

    runtime = _make_runtime(tmp_path)
    svc = runtime.get_tool_service()
    assert isinstance(svc, FinsToolService)


@pytest.mark.unit
def test_get_tool_service_uses_first_call_cache_size(tmp_path: Path) -> None:
    """验证首次调用的 processor_cache_max_entries 生效，后续调用忽略。"""

    runtime = _make_runtime(tmp_path)
    svc1 = runtime.get_tool_service(processor_cache_max_entries=16)
    svc2 = runtime.get_tool_service(processor_cache_max_entries=64)
    assert svc1 is svc2
    assert svc1._processor_cache.max_entries == 16


@pytest.mark.unit
def test_fins_tool_service_initializes_document_meta_cache_in_constructor(tmp_path: Path) -> None:
    """验证 FinsToolService 在构造期声明文档元数据缓存。"""

    adapters = build_legacy_repository_adapters(_StubRepository())
    service = FinsToolService(
        company_repository=adapters.company_repository,
        source_repository=adapters.source_repository,
        processed_repository=adapters.processed_repository,
        processor_registry=cast(ProcessorRegistry, _StubProcessorRegistry()),
    )

    assert service._meta_cache == {}


@pytest.mark.unit
def test_get_tool_service_thread_safe(tmp_path: Path) -> None:
    """验证并发调用 get_tool_service 只产生一个实例。"""

    runtime = _make_runtime(tmp_path)
    results: list[FinsToolService] = []
    thread_count = 8
    barrier = Barrier(thread_count)

    def _worker() -> None:
        barrier.wait()
        results.append(runtime.get_tool_service())

    threads = [Thread(target=_worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == thread_count
    assert all(r is results[0] for r in results)


@pytest.mark.unit
def test_different_runtimes_have_different_services(tmp_path: Path) -> None:
    """验证不同 runtime 实例各有独立的 FinsToolService。"""

    rt1 = _make_runtime(tmp_path)
    rt2 = _make_runtime(tmp_path)
    assert rt1.get_tool_service() is not rt2.get_tool_service()


@pytest.mark.unit
def test_service_runtime_helper_functions_cover_result_builders_and_event_narrowing() -> None:
    """验证 service runtime 的结果构造与事件收窄 helper。"""

    download_event = DownloadEvent(
        event_type=DownloadEventType.FILING_COMPLETED,
        ticker="AAPL",
        document_id="fil_1",
        payload={
            "action": "download",
            "message": "done",
            "skip_reason": "cached",
            "filing_result": {
                "document_id": "fil_1",
                "status": "downloaded",
                "downloaded_files": 2,
                "failed_files": [{"name": "x.htm", "error": "bad"}],
            },
        },
    )
    process_event = ProcessEvent(
        event_type=ProcessEventType.DOCUMENT_COMPLETED,
        ticker="AAPL",
        document_id="fil_1",
        payload={
            "source_kind": "filing",
            "overwrite": True,
            "ci": False,
            "result_summary": {"document_id": "fil_1", "status": "processed", "form_type": "10-K"},
        },
    )
    upload_material_event = UploadMaterialEvent(
        event_type=UploadMaterialEventType.FILE_UPLOADED,
        ticker="AAPL",
        document_id="mat_1",
        payload={"action": "upload", "name": "deck.pdf", "file_count": 1, "size": 123, "message": "ok"},
    )

    assert service_runtime_module._require_download_event(download_event) is download_event
    assert service_runtime_module._require_process_event(process_event) is process_event
    assert service_runtime_module._require_upload_filing_event(
        UploadFilingEvent(event_type=UploadFilingEventType.FILE_UPLOADED, ticker="AAPL")
    ).ticker == "AAPL"
    assert service_runtime_module._require_upload_material_event(upload_material_event) is upload_material_event
    with pytest.raises(TypeError, match="DownloadEvent"):
        service_runtime_module._require_download_event(process_event)
    with pytest.raises(TypeError, match="ProcessEvent"):
        service_runtime_module._require_process_event(download_event)
    with pytest.raises(TypeError, match="upload_material"):
        service_runtime_module._require_upload_material_event(download_event)

    download_item = service_runtime_module._build_download_filing_result_item(
        {
            "document_id": "fil_1",
            "status": "downloaded",
            "downloaded_files": 2,
            "failed_files": [{"name": "x.htm", "error": "bad"}],
            "message": "done",
        }
    )
    upload_material_result = service_runtime_module._build_upload_material_result_data(
        {
            "pipeline": "sec",
            "status": "ok",
            "ticker": "AAPL",
            "material_action": "upload",
            "files": [{"name": "deck.pdf", "status": "uploaded"}],
            "material_name": "deck",
            "message": "done",
        }
    )
    upload_filings_from_result = service_runtime_module._build_upload_filings_from_result_data(
        {
            "script_path": "/tmp/upload.py",
            "script_platform": "sec",
            "ticker": "AAPL",
            "source_dir": "/tmp/source",
            "total_files": 3,
            "recognized_count": 1,
            "material_count": 1,
            "skipped_count": 1,
            "recognized": [{"file": "10k.html", "fiscal_year": 2024, "fiscal_period": "FY"}],
            "material": [{"file": "deck.pdf", "material_name": "deck"}],
            "skipped": [{"file": "bad.txt", "reason": "unsupported"}],
        }
    )
    process_single_result = service_runtime_module._build_process_single_result_data(
        {
            "pipeline": "sec",
            "action": "process_filing",
            "status": "ok",
            "ticker": "AAPL",
            "document_id": "fil_1",
            "overwrite": True,
            "ci": False,
            "message": "done",
        }
    )
    extracted = service_runtime_module._extract_result_data_from_event(
        command_name=FinsCommandName.DOWNLOAD,
        event_payload={
            "result": {
                "pipeline": "sec",
                "status": "ok",
                "ticker": "AAPL",
                "company_info": {"company_id": "1", "company_name": "Test", "market": "US"},
                "summary": {"total": 1, "downloaded": 1, "skipped": 0, "failed": 0, "elapsed_ms": 9},
            }
        },
    )
    full_download_result = service_runtime_module._build_download_result_data(
        {
            "pipeline": "sec",
            "status": "ok",
            "ticker": "AAPL",
            "company_info": {"company_id": "1", "company_name": "Test", "market": "US"},
            "filters": {
                "forms": ["10-K", "10-Q"],
                "start_dates": {"10-K": "2024-01-01"},
                "end_date": "2024-12-31",
                "overwrite": True,
            },
            "warnings": ["warn"],
            "filings": [{"document_id": "fil_1", "status": "downloaded"}],
            "summary": {"total": 1, "downloaded": 1, "skipped": 0, "failed": 0, "elapsed_ms": 9},
        }
    )
    download_progress = service_runtime_module._build_download_progress_payload(download_event)
    process_progress = service_runtime_module._build_process_progress_payload(process_event)
    upload_material_progress = service_runtime_module._build_upload_material_progress_payload(upload_material_event)

    assert service_runtime_module._coerce_forms_input(None) is None
    from dayu.fins._converters import optional_int
    assert optional_int(bytearray(b"7")) == 7
    assert optional_int(object()) is None
    assert download_item.failed_files[0].file_name == "x.htm"
    assert upload_material_result.material_name == "deck"
    assert upload_filings_from_result.recognized[0].file == "10k.html"
    assert process_single_result.document_id == "fil_1"
    assert isinstance(extracted, DownloadResultData)
    assert full_download_result.filters.forms == ("10-K", "10-Q")
    assert full_download_result.filters.start_dates[0].form_type == "10-K"
    assert full_download_result.warnings == ("warn",)
    assert isinstance(
        service_runtime_module._build_result_data(FinsCommandName.UPLOAD_FILINGS_FROM, {"ticker": "AAPL"}),
        type(upload_filings_from_result),
    )
    assert isinstance(
        service_runtime_module._build_result_data(FinsCommandName.PROCESS, {"ticker": "AAPL"}),
        ProcessResultData,
    )
    assert download_progress.reason == "cached"
    assert download_progress.filing_result is not None
    assert process_progress.result_summary is not None
    assert process_progress.result_summary.form_type == "10-K"
    assert upload_material_progress.event_type == FinsProgressEventName.FILE_UPLOADED
    assert upload_material_progress.message == "ok"


@pytest.mark.unit
def test_default_fins_runtime_helper_methods_cover_registry_and_company_meta_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 runtime 的轻量 helper：processor registry、公司名和 meta 摘要。"""

    runtime = _make_runtime(tmp_path)

    assert runtime.get_processor_registry() is runtime.processor_registry
    assert runtime.get_company_name("AAPL") == "Test"
    assert runtime.get_company_meta_summary("AAPL") == {
        "ticker": "AAPL",
        "company_name": "Test",
        "market": "US",
        "company_id": "1",
    }

    monkeypatch.setattr(runtime.company_repository, "get_company_meta", lambda _ticker: (_ for _ in ()).throw(RuntimeError("missing")))

    assert runtime.get_company_name("MSFT") == ""
    assert runtime.get_company_meta_summary("MSFT") == {"ticker": "MSFT"}


@pytest.mark.unit
def test_execute_sync_upload_filings_from_uses_script_generator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `_execute_sync` 会走 upload_filings_from 分支并构造结果数据。"""

    runtime = _make_runtime(tmp_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    monkeypatch.setattr(
        service_runtime_module,
        "generate_upload_filings_script",
        lambda namespace: {
            "script_path": "/tmp/upload.py",
            "script_platform": "sec",
            "ticker": namespace.ticker,
            "source_dir": namespace.source_dir,
            "total_files": 1,
            "recognized_count": 1,
            "material_count": 0,
            "skipped_count": 0,
            "recognized": [{"file": "10k.html", "fiscal_year": 2024, "fiscal_period": "FY"}],
        },
    )

    result = _require_sync_result(
        runtime.execute(
            FinsCommand(
                name=FinsCommandName.UPLOAD_FILINGS_FROM,
                payload=UploadFilingsFromCommandPayload(
                    ticker="AAPL",
                    source_dir=source_dir,
                    company_id="1",
                    company_name="Apple",
                ),
                stream=False,
            )
        )
    )
    upload_filings_from_data = cast(Any, result.data)

    assert upload_filings_from_data.ticker == "AAPL"
    assert upload_filings_from_data.source_dir == str(source_dir)
    assert upload_filings_from_data.recognized[0].file == "10k.html"


@pytest.mark.unit
def test_execute_stream_upload_material_yields_progress_and_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `_execute_stream` 会走 upload_material 分支并产出统一事件。"""

    runtime = _make_runtime(tmp_path)
    material_file = tmp_path / "deck.pdf"
    material_file.write_text("demo", encoding="utf-8")

    class _FakePipeline:
        async def upload_material_stream(self, **kwargs) -> AsyncIterator[UploadMaterialEvent]:
            assert kwargs["ticker"] == "AAPL"
            yield UploadMaterialEvent(
                event_type=UploadMaterialEventType.FILE_UPLOADED,
                ticker="AAPL",
                document_id="mat_1",
                payload={"action": "upload", "name": "deck.pdf", "message": "uploaded"},
            )
            yield UploadMaterialEvent(
                event_type=UploadMaterialEventType.UPLOAD_COMPLETED,
                ticker="AAPL",
                document_id="mat_1",
                payload={
                    "result": {
                        "pipeline": "sec",
                        "status": "ok",
                        "ticker": "AAPL",
                        "material_action": "upload",
                        "files": ["deck.pdf"],
                        "material_name": "deck",
                    }
                },
            )

    monkeypatch.setattr(runtime, "_build_pipeline_for_ticker", lambda _ticker: _FakePipeline())
    monkeypatch.setattr(service_runtime_module, "validate_upload_material_args", lambda _namespace: None)

    async def _collect() -> list[FinsEvent]:
        result = runtime.execute(
            FinsCommand(
                name=FinsCommandName.UPLOAD_MATERIAL,
                payload=UploadMaterialCommandPayload(
                    ticker="AAPL",
                    files=(material_file,),
                    company_id="1",
                    company_name="Apple",
                ),
                stream=True,
            )
        )
        events: list[FinsEvent] = []
        async for event in _require_stream_result(result):
            events.append(event)
        return events

    events = asyncio.run(_collect())

    assert events[0].type == FinsEventType.PROGRESS
    assert isinstance(events[0].payload, UploadMaterialProgressPayload)
    assert events[0].payload.message == "uploaded"
    assert events[-1].type == FinsEventType.RESULT
    assert events[-1].payload.ticker == "AAPL"


@pytest.mark.unit
def test_execute_uses_runtime_repository_and_processor_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 direct operation 与工具链复用同一 repository/processor_registry 真源。"""

    runtime = _make_runtime(tmp_path)
    captured: dict[str, Any] = {}

    class _FakePipeline:
        def process(
            self,
            *,
            ticker: str,
            overwrite: bool,
            ci: bool,
            document_ids: list[str] | None = None,
        ) -> dict[str, Any]:
            captured["document_ids"] = document_ids
            return {
                "ticker": ticker,
                "overwrite": overwrite,
                "ci": ci,
                "document_ids": document_ids,
            }

    def _fake_factory(
        *,
        market_profile: Any,
        workspace_root: Path,
        company_repository: Any = None,
        source_repository: Any = None,
        processed_repository: Any = None,
        blob_repository: Any = None,
        filing_maintenance_repository: Any = None,
        processor_registry: Any = None,
        **_kwargs: Any,
    ) -> Any:
        captured["market"] = market_profile.market
        captured["workspace_root"] = workspace_root
        captured["company_repository"] = company_repository
        captured["source_repository"] = source_repository
        captured["processed_repository"] = processed_repository
        captured["blob_repository"] = blob_repository
        captured["filing_maintenance_repository"] = filing_maintenance_repository
        captured["processor_registry"] = processor_registry
        return _FakePipeline()

    monkeypatch.setattr("dayu.fins.service_runtime.get_pipeline_from_market_profile", _fake_factory)

    result = _require_sync_result(
        runtime.execute(
        FinsCommand(
            name=FinsCommandName.PROCESS,
            payload=ProcessCommandPayload(ticker="AAPL", overwrite=True, ci=False, document_ids=("fil_001",)),
            stream=False,
        )
    ))

    assert captured["market"] == "US"
    assert captured["workspace_root"] == tmp_path
    assert captured["company_repository"] is runtime.company_repository
    assert captured["source_repository"] is runtime.source_repository
    assert captured["processed_repository"] is runtime.processed_repository
    assert captured["blob_repository"] is runtime.blob_repository
    assert captured["filing_maintenance_repository"] is runtime.filing_maintenance_repository
    assert captured["processor_registry"] is runtime.processor_registry
    assert captured["document_ids"] == ["fil_001"]
    assert isinstance(result.data, ProcessResultData)
    assert result.data.ticker == "AAPL"
    assert result.data.overwrite is True
    assert result.data.ci is False


@pytest.mark.unit
def test_execute_process_treats_empty_document_ids_as_unspecified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证默认空 document_ids 不会在 runtime 桥接层报错。"""

    runtime = _make_runtime(tmp_path)
    captured: dict[str, Any] = {}

    class _FakePipeline:
        def process(
            self,
            *,
            ticker: str,
            overwrite: bool,
            ci: bool,
            document_ids: list[str] | None = None,
        ) -> dict[str, Any]:
            captured["ticker"] = ticker
            captured["overwrite"] = overwrite
            captured["ci"] = ci
            captured["document_ids"] = document_ids
            return {
                "ticker": ticker,
                "overwrite": overwrite,
                "ci": ci,
                "document_ids": document_ids,
            }

    monkeypatch.setattr(
        runtime,
        "_build_pipeline_for_ticker",
        lambda ticker: _FakePipeline(),
    )

    result = _require_sync_result(
        runtime.execute(
        FinsCommand(
            name=FinsCommandName.PROCESS,
            payload=ProcessCommandPayload(ticker="AAPL"),
            stream=False,
        )
    ))

    assert captured["ticker"] == "AAPL"
    assert captured["overwrite"] is False
    assert captured["ci"] is False
    assert captured["document_ids"] is None
    assert isinstance(result.data, ProcessResultData)
    assert result.data.ticker == "AAPL"


@pytest.mark.unit
def test_execute_download_uses_normalized_namespace_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 runtime 二次 prepare 时仍保留 payload 内的 ticker alias。"""

    runtime = _make_runtime(tmp_path)
    captured: dict[str, Any] = {}

    class _FakePipeline:
        def download(
            self,
            *,
            ticker: str,
            form_type: str | None,
            start_date: str | None,
            end_date: str | None,
            overwrite: bool,
            rebuild: bool,
            ticker_aliases: list[str] | None,
        ) -> dict[str, Any]:
            captured["download_ticker"] = ticker
            captured["download_form_type"] = form_type
            captured["download_ticker_aliases"] = ticker_aliases
            captured["download_overwrite"] = overwrite
            captured["download_rebuild"] = rebuild
            captured["download_start_date"] = start_date
            captured["download_end_date"] = end_date
            return {
                "pipeline": "fake",
                "status": "ok",
                "ticker": ticker,
                "summary": {"total": 0, "downloaded": 0, "skipped": 0, "failed": 0},
            }

    def _fake_factory(
        *,
        market_profile: Any,
        **_kwargs: Any,
    ) -> Any:
        captured["pipeline_ticker"] = market_profile.ticker
        return _FakePipeline()

    monkeypatch.setattr("dayu.fins.service_runtime.get_pipeline_from_market_profile", _fake_factory)

    result = _require_sync_result(
        runtime.execute(
        FinsCommand(
            name=FinsCommandName.DOWNLOAD,
            payload=DownloadCommandPayload(
                ticker="BABA",
                form_type=("10-K",),
                ticker_aliases=("BABA", "9988", "9988.HK"),
                overwrite=True,
                rebuild=True,
            ),
            stream=False,
        )
    ))

    assert captured["pipeline_ticker"] == "BABA"
    assert captured["download_ticker"] == "BABA"
    assert captured["download_form_type"] == "10-K"
    assert captured["download_ticker_aliases"] == ["BABA", "9988", "9988.HK"]
    assert isinstance(result.data, DownloadResultData)
    assert result.data.ticker == "BABA"


@pytest.mark.unit
def test_execute_upload_filing_uses_normalized_namespace_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 runtime 二次 prepare 不会丢失上传命令的 alias 与公司名。"""

    runtime = _make_runtime(tmp_path)
    captured: dict[str, Any] = {}

    class _FakePipeline:
        def upload_filing(
            self,
            *,
            ticker: str,
            action: str,
            files: list[Path],
            fiscal_year: int,
            fiscal_period: str,
            amended: bool,
            filing_date: str | None,
            report_date: str | None,
            company_id: str | None,
            company_name: str | None,
            ticker_aliases: list[str] | None,
            overwrite: bool,
        ) -> dict[str, Any]:
            captured["upload_ticker"] = ticker
            captured["upload_company_name"] = company_name
            captured["upload_ticker_aliases"] = ticker_aliases
            captured["upload_files"] = files
            captured["upload_fiscal_year"] = fiscal_year
            captured["upload_fiscal_period"] = fiscal_period
            return {
                "pipeline": "fake",
                "status": "ok",
                "ticker": ticker,
                "filing_action": action,
                "company_name": company_name,
            }

    def _fake_factory(
        *,
        market_profile: Any,
        **_kwargs: Any,
    ) -> Any:
        captured["pipeline_ticker"] = market_profile.ticker
        return _FakePipeline()

    monkeypatch.setattr("dayu.fins.service_runtime.get_pipeline_from_market_profile", _fake_factory)
    monkeypatch.setattr(
        "dayu.fins.cli.infer_company_aliases_from_fmp",
        lambda _ticker: (_ for _ in ()).throw(RuntimeError("should not be required for preserved payload data")),
    )

    result = _require_sync_result(
        runtime.execute(
        FinsCommand(
            name=FinsCommandName.UPLOAD_FILING,
            payload=UploadFilingCommandPayload(
                ticker="AAPL",
                files=(Path("report.pdf"),),
                fiscal_year=2024,
                fiscal_period="FY",
                action="create",
                company_id="1",
                company_name="Apple Inc.",
                infer=True,
                ticker_aliases=("AAPL", "AAPL.US"),
            ),
            stream=False,
        )
    ))

    assert captured["pipeline_ticker"] == "AAPL"
    assert captured["upload_ticker"] == "AAPL"
    assert captured["upload_company_name"] == "Apple Inc."
    assert captured["upload_ticker_aliases"] == ["AAPL", "AAPL.US"]
    assert isinstance(result.data, UploadFilingResultData)
    assert result.data.ticker == "AAPL"
    assert result.data.company_name == "Apple Inc."


@pytest.mark.unit
def test_execute_upload_filing_stream_accepts_file_uploaded_progress_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 runtime 能把 file_uploaded 事件映射到稳定 progress 契约。"""

    runtime = _make_runtime(tmp_path)

    class _FakePipeline:
        async def upload_filing_stream(
            self,
            *,
            ticker: str,
            action: str,
            files: list[Path],
            fiscal_year: int,
            fiscal_period: str,
            amended: bool,
            filing_date: str | None,
            report_date: str | None,
            company_id: str | None,
            company_name: str | None,
            ticker_aliases: list[str] | None,
            overwrite: bool,
        ):
            del files, fiscal_year, fiscal_period, amended, filing_date, report_date, company_id, company_name, ticker_aliases, overwrite
            yield UploadFilingEvent(
                event_type=UploadFilingEventType.FILE_UPLOADED,
                ticker=ticker,
                payload={"action": action, "name": "report.pdf", "size": 123},
            )
            yield UploadFilingEvent(
                event_type=UploadFilingEventType.UPLOAD_COMPLETED,
                ticker=ticker,
                payload={
                    "result": {
                        "pipeline": "fake",
                        "status": "ok",
                        "ticker": ticker,
                        "filing_action": action,
                    }
                },
            )

    def _fake_factory(**_kwargs: Any) -> Any:
        return _FakePipeline()

    monkeypatch.setattr("dayu.fins.service_runtime.get_pipeline_from_market_profile", _fake_factory)

    async def _collect() -> list[Any]:
        stream = _require_stream_result(runtime.execute(
            FinsCommand(
                name=FinsCommandName.UPLOAD_FILING,
                payload=UploadFilingCommandPayload(
                    ticker="AAPL",
                    files=(Path("report.pdf"),),
                    fiscal_year=2024,
                    fiscal_period="FY",
                    action="create",
                    company_id="1",
                    company_name="Apple Inc.",
                ),
                stream=True,
            )
        ))
        return [event async for event in stream]

    events = asyncio.run(_collect())

    assert events[0].type == FinsEventType.PROGRESS
    assert isinstance(events[0].payload, UploadFilingProgressPayload)
    assert events[0].payload.event_type == FinsProgressEventName.FILE_UPLOADED
    assert events[0].payload.name == "report.pdf"
    assert events[-1].type == FinsEventType.RESULT
