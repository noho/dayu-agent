"""Sec process workflow 真源测试。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, cast

import pytest

from dayu.fins.domain.enums import SourceKind
from dayu.fins.ingestion.process_events import ProcessEventType
from dayu.fins.pipelines import sec_process_workflow as workflow_module
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.fins.storage import ProcessedDocumentRepositoryProtocol, SourceDocumentRepositoryProtocol


class _FakeDownloader:
    """测试用 ticker 标准化器。"""

    def normalize_ticker(self, ticker: str) -> str:
        """返回大写 ticker。

        Args:
            ticker: 原始 ticker。

        Returns:
            大写 ticker。

        Raises:
            无。
        """

        return ticker.upper()


@dataclass
class _FakeProcessedRepository:
    """测试用 processed 仓储。"""

    cleared_tickers: list[str] = field(default_factory=list)

    def clear_processed_documents(self, ticker: str) -> None:
        """记录 clear 调用。

        Args:
            ticker: 股票代码。

        Returns:
            无。

        Raises:
            无。
        """

        self.cleared_tickers.append(ticker)


@dataclass
class _FakeSourceRepository:
    """测试用 source 仓储。"""

    filing_ids: list[str]
    material_ids: list[str]
    meta_map: dict[tuple[SourceKind, str], Optional[dict[str, object]]]

    def list_source_document_ids(self, ticker: str, source_kind: SourceKind) -> list[str]:
        """返回指定 source_kind 的文档列表。

        Args:
            ticker: 股票代码。
            source_kind: 文档类型。

        Returns:
            文档 ID 列表。

        Raises:
            无。
        """

        del ticker
        if source_kind == SourceKind.FILING:
            return list(self.filing_ids)
        return list(self.material_ids)

    def get_source_meta(
        self,
        *,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> dict[str, object]:
        """返回 source meta；测试里仅供单文档分支使用。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 文档类型。

        Returns:
            对应 source meta。

        Raises:
            KeyError: 当 meta 未准备时抛出。
        """

        del ticker
        meta = self.meta_map[(source_kind, document_id)]
        if meta is None:
            raise KeyError(document_id)
        return dict(meta)

    def get_primary_source(
        self,
        *,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> object:
        """本测试不应命中 primary source 读取。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 文档类型。

        Returns:
            无。

        Raises:
            AssertionError: 一旦调用说明测试准备不正确。
        """

        del ticker
        del document_id
        del source_kind
        raise AssertionError("本测试不应读取 primary source")


@dataclass
class _FakeWorkflowHost:
    """测试用 workflow host。"""

    source_repository: _FakeSourceRepository
    processed_repository: _FakeProcessedRepository = field(default_factory=_FakeProcessedRepository)
    built_results: list[dict[str, object]] = field(default_factory=list)

    MODULE = "TEST.SEC_PROCESS"
    _downloader = _FakeDownloader()
    _processor_registry = cast(ProcessorRegistry, object())

    @property
    def _source_repository(self) -> SourceDocumentRepositoryProtocol:
        """暴露 source 仓储。"""

        return cast(SourceDocumentRepositoryProtocol, self.source_repository)

    @property
    def _processed_repository(self) -> ProcessedDocumentRepositoryProtocol:
        """暴露 processed 仓储。"""

        return cast(ProcessedDocumentRepositoryProtocol, self.processed_repository)

    def _safe_get_document_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> Optional[dict[str, object]]:
        """安全读取测试 meta。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 文档类型。

        Returns:
            对应 meta；不存在时返回 `None`。

        Raises:
            无。
        """

        del ticker
        meta = self.source_repository.meta_map.get((source_kind, document_id))
        return None if meta is None else dict(meta)

    def _safe_read_snapshot_meta(
        self,
        *,
        ticker: str,
        document_id: str,
    ) -> Optional[dict[str, object]]:
        """测试中不使用 snapshot meta。"""

        del ticker
        del document_id
        return None

    def _can_skip_snapshot_export(
        self,
        *,
        source_meta: dict[str, object],
        snapshot_meta: Optional[dict[str, object]],
        overwrite: bool,
        expected_parser_signature: str,
        ci: bool,
        ticker: str,
        document_id: str,
    ) -> bool:
        """测试中默认不跳过导出。"""

        del source_meta
        del snapshot_meta
        del overwrite
        del expected_parser_signature
        del ci
        del ticker
        del document_id
        return False

    def _export_tool_snapshot_for_document(
        self,
        *,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
        source_meta: dict[str, object],
        ci: bool,
        expected_parser_signature: str,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        """本测试不直接走导出实现。"""

        del ticker
        del document_id
        del source_kind
        del source_meta
        del ci
        del expected_parser_signature
        del cancel_checker
        raise AssertionError("本测试不应直接导出快照")

    def _build_result(self, action: str, **payload: object) -> dict[str, object]:
        """构建并记录统一结果。

        Args:
            action: 动作名。
            payload: 结果载荷。

        Returns:
            统一结果字典。

        Raises:
            无。
        """

        result = {"action": action, **payload}
        self.built_results.append(result)
        return result


@pytest.mark.asyncio
async def test_run_process_stream_impl_covers_filing_material_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证流式 process 真源会覆盖 filing/material 的跳过、失败与汇总路径。"""

    host = _FakeWorkflowHost(
        source_repository=_FakeSourceRepository(
            filing_ids=["fil_missing", "fil_incomplete", "fil_ok", "fil_fail"],
            material_ids=["mat_deleted", "mat_ok"],
            meta_map={
                (SourceKind.FILING, "fil_missing"): None,
                (SourceKind.FILING, "fil_incomplete"): {"document_id": "fil_incomplete", "ingest_complete": False},
                (SourceKind.FILING, "fil_ok"): {"document_id": "fil_ok", "ingest_complete": True},
                (SourceKind.FILING, "fil_fail"): {"document_id": "fil_fail", "ingest_complete": True},
                (SourceKind.MATERIAL, "mat_deleted"): {
                    "document_id": "mat_deleted",
                    "ingest_complete": True,
                    "is_deleted": True,
                },
                (SourceKind.MATERIAL, "mat_ok"): {"document_id": "mat_ok", "ingest_complete": True},
            },
        )
    )

    def _fake_run_process_single_document(
        current_host: _FakeWorkflowHost,
        *,
        ticker: str,
        document_id: str,
        overwrite: bool,
        ci: bool,
        source_kind: SourceKind,
        action: str,
        cancel_checker: object = None,
    ) -> dict[str, object]:
        """按文档 ID 返回不同结果，覆盖 stream_impl 分支。"""

        del current_host
        del source_kind
        del cancel_checker
        if document_id == "fil_fail":
            raise RuntimeError("boom")
        if document_id == "mat_ok":
            return host._build_result(
                action=action,
                ticker=ticker,
                document_id=document_id,
                overwrite=overwrite,
                ci=ci,
                status="skipped",
                reason="version_matched",
            )
        return host._build_result(
            action=action,
            ticker=ticker,
            document_id=document_id,
            overwrite=overwrite,
            ci=ci,
            status="processed",
        )

    monkeypatch.setattr(workflow_module, "run_process_single_document", _fake_run_process_single_document)

    events = [
        event
        async for event in workflow_module.run_process_stream_impl(
            host,
            ticker="aapl",
            overwrite=True,
            ci=True,
        )
    ]

    assert host.processed_repository.cleared_tickers == ["AAPL"]
    assert events[0].event_type == ProcessEventType.PIPELINE_STARTED
    assert events[0].payload == {
        "overwrite": True,
        "ci": True,
        "requested_document_ids": None,
        "filing_total": 4,
        "material_total": 2,
        "total_documents": 6,
    }
    assert any(
        event.event_type == ProcessEventType.DOCUMENT_FAILED and event.document_id == "fil_fail"
        for event in events
    )
    assert any(
        event.event_type == ProcessEventType.DOCUMENT_SKIPPED
        and event.document_id == "mat_deleted"
        and event.payload["reason"] == "deleted"
        for event in events
    )
    assert any(
        event.event_type == ProcessEventType.DOCUMENT_SKIPPED
        and event.document_id == "mat_ok"
        and event.payload["reason"] == "version_matched"
        for event in events
    )

    final_result = events[-1].payload["result"]
    assert final_result["status"] == "ok"
    assert final_result["filing_summary"] == {"total": 4, "processed": 1, "skipped": 2, "failed": 1}
    assert final_result["material_summary"] == {"total": 2, "processed": 0, "skipped": 2, "failed": 0}


@pytest.mark.asyncio
async def test_run_process_stream_impl_marks_cancelled_at_material_boundary() -> None:
    """验证流式 process 在 material 文档边界收到取消请求时会停止并标记 cancelled。"""

    host = _FakeWorkflowHost(
        source_repository=_FakeSourceRepository(
            filing_ids=[],
            material_ids=["mat_001"],
            meta_map={(SourceKind.MATERIAL, "mat_001"): {"document_id": "mat_001", "ingest_complete": True}},
        )
    )

    events = [
        event
        async for event in workflow_module.run_process_stream_impl(
            host,
            ticker="aapl",
            cancel_checker=lambda: True,
        )
    ]

    assert [event.event_type for event in events] == [
        ProcessEventType.PIPELINE_STARTED,
        ProcessEventType.PIPELINE_COMPLETED,
    ]
    assert events[-1].payload["result"]["status"] == "cancelled"
    assert events[-1].payload["result"]["material_summary"] == {"total": 0, "processed": 0, "skipped": 0, "failed": 0}


@pytest.mark.unit
def test_run_process_single_document_skips_deleted_source() -> None:
    """验证单文档 process 真源会在 source 已删除时直接返回 skipped。"""

    host = _FakeWorkflowHost(
        source_repository=_FakeSourceRepository(
            filing_ids=["fil_deleted"],
            material_ids=[],
            meta_map={
                (SourceKind.FILING, "fil_deleted"): {
                    "document_id": "fil_deleted",
                    "ingest_complete": True,
                    "is_deleted": True,
                }
            },
        )
    )

    result = workflow_module.run_process_single_document(
        host,
        ticker="aapl",
        document_id="fil_deleted",
        overwrite=False,
        ci=False,
        source_kind=SourceKind.FILING,
        action="process_filing",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "deleted"
    assert result["ticker"] == "AAPL"