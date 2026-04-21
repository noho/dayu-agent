"""CnPipeline 辅助分支测试。"""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import pytest

from dayu.fins.domain.document_models import FilingCreateRequest, MaterialCreateRequest
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines import cn_pipeline
from dayu.fins.pipelines.cn_pipeline import CnPipeline
from dayu.fins.processors.registry import build_fins_processor_registry
from dayu.fins.storage.local_file_store import LocalFileStore
from tests.fins.storage_testkit import build_storage_core


class _SourceRepositoryLike(Protocol):
    """CnPipeline helper 测试所需的最小仓储边界。"""

    portfolio_root: Path

    def create_filing(self, req: FilingCreateRequest) -> object:
        """创建 filing 源文档。"""

    def create_material(self, req: MaterialCreateRequest) -> object:
        """创建 material 源文档。"""


def _repository(repository: _SourceRepositoryLike) -> _SourceRepositoryLike:
    """显式收窄 build_storage_core 返回的仓储类型。"""

    return repository


async def _collect_upload_result(stream: Any, stream_name: str) -> dict[str, Any]:
    """包装调用上传事件流聚合器。

    Args:
        stream: 事件流。
        stream_name: 流名称。

    Returns:
        最终结果字典。

    Raises:
        RuntimeError: 缺失最终结果时抛出。
    """

    return await cn_pipeline._collect_upload_result_from_events(stream, stream_name=stream_name)


def _prepare_filing(
    repository: _SourceRepositoryLike,
    *,
    ticker: str,
    document_id: str,
    ingest_complete: bool,
) -> None:
    """构造 filing 测试数据。

    Args:
        repository: 仓储实例。
        ticker: 股票代码。
        document_id: 文档 ID。
        ingest_complete: 是否完成入库。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    store = LocalFileStore(root=repository.portfolio_root, scheme="local")
    file_meta = store.put_object(f"{ticker}/filings/{document_id}/filing_docling.json", BytesIO(b"{}"))
    repository.create_filing(
        FilingCreateRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=document_id,
            form_type="FY",
            primary_document="filing_docling.json",
            files=[file_meta],
            meta={
                "ingest_complete": ingest_complete,
                "is_deleted": False,
                "document_version": "v1",
                "source_fingerprint": "fp",
                "fiscal_year": 2025,
                "fiscal_period": "FY",
            },
        )
    )


def _prepare_material(
    repository: _SourceRepositoryLike,
    *,
    ticker: str,
    document_id: str,
    is_deleted: bool,
) -> None:
    """构造 material 测试数据。

    Args:
        repository: 仓储实例。
        ticker: 股票代码。
        document_id: 文档 ID。
        is_deleted: 是否逻辑删除。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    store = LocalFileStore(root=repository.portfolio_root, scheme="local")
    file_meta = store.put_object(f"{ticker}/materials/{document_id}/material_docling.json", BytesIO(b"{}"))
    repository.create_material(
        MaterialCreateRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=document_id,
            form_type="MATERIAL_OTHER",
            primary_document="material_docling.json",
            files=[file_meta],
            meta={
                "ingest_complete": True,
                "is_deleted": is_deleted,
                "document_version": "v1",
                "source_fingerprint": "fp",
            },
        )
    )


@pytest.mark.unit
def test_helper_normalize_collect_and_sync_wrapper() -> None:
    """验证 ticker 标准化、事件聚合与同步包装器分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert cn_pipeline._normalize_ticker(" 000001 ") == "000001"
    with pytest.raises(ValueError, match="ticker 不能为空"):
        cn_pipeline._normalize_ticker("   ")

    assert cn_pipeline._resolve_upload_status("uploaded") == "ok"
    assert cn_pipeline._resolve_upload_status("failed") == "failed"

    async def _ok_stream():
        """返回带最终结果的事件流。

        Args:
            无。

        Returns:
            事件流。

        Raises:
            RuntimeError: 迭代失败时抛出。
        """

        yield SimpleNamespace(event_type="upload_started", payload={})
        yield SimpleNamespace(event_type="upload_completed", payload={"result": {"status": "ok"}})

    async def _bad_stream():
        """返回无最终结果的事件流。

        Args:
            无。

        Returns:
            事件流。

        Raises:
            RuntimeError: 迭代失败时抛出。
        """

        yield SimpleNamespace(event_type="upload_started", payload={})

    assert asyncio.run(_collect_upload_result(_ok_stream(), "ok_stream"))["status"] == "ok"
    with pytest.raises(RuntimeError, match="未返回最终结果"):
        asyncio.run(_collect_upload_result(_bad_stream(), "bad_stream"))

    async def _coro_ok() -> dict[str, Any]:
        """返回固定结果。

        Args:
            无。

        Returns:
            结果字典。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        return {"ok": True}

    assert cn_pipeline._run_async_pipeline_sync(_coro_ok())["ok"] is True

    async def _inside_loop() -> None:
        """在运行 loop 下触发异常分支。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        coro = _coro_ok()
        with pytest.raises(RuntimeError, match="stream 异步接口"):
            cn_pipeline._run_async_pipeline_sync(coro)
        coro.close()

    asyncio.run(_inside_loop())


@pytest.mark.asyncio
async def test_upload_filing_stream_emits_upload_failed_when_execute_upload_errors(tmp_path: Path) -> None:
    """验证 upload_filing_stream 异常路径会产出 upload_failed。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pipeline = CnPipeline(workspace_root=tmp_path, processor_registry=build_fins_processor_registry())

    def _raise_upload(**kwargs: Any) -> Any:
        """模拟上传失败。

        Args:
            **kwargs: 上传参数。

        Returns:
            无。

        Raises:
            RuntimeError: 始终抛出。
        """

        del kwargs
        raise RuntimeError("upload boom")

    pipeline._upload_service.execute_upload = _raise_upload  # type: ignore[attr-defined]
    sample_file = tmp_path / "cn_filing.pdf"
    sample_file.write_text("demo", encoding="utf-8")

    events = [
        event
        async for event in pipeline.upload_filing_stream(
            ticker="000001",
            action="create",
            files=[sample_file],
            fiscal_year=2025,
            fiscal_period="FY",
            amended=False,
            filing_date="2025-01-01",
            report_date="2024-12-31",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]

    assert events[-1].event_type == "upload_failed"
    result = events[-1].payload["result"]
    assert result["status"] == "failed"
    assert "upload boom" in result["message"]


@pytest.mark.asyncio
async def test_upload_material_stream_rejects_mismatched_explicit_ids(tmp_path: Path) -> None:
    """验证 upload_material_stream 会拒绝与稳定 ID 不一致的显式 document_id。"""

    pipeline = CnPipeline(workspace_root=tmp_path, processor_registry=build_fins_processor_registry())

    with pytest.raises(ValueError, match="显式 --document-id"):
        async for _ in pipeline.upload_material_stream(
            ticker="000001",
            action="update",
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=None,
            document_id="mat_legacy_1",
            internal_document_id=None,
            filing_date="2025-01-01",
            report_date="2024-12-31",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        ):
            pass


@pytest.mark.asyncio
async def test_upload_material_stream_passes_stable_ids_and_auto_action(tmp_path: Path) -> None:
    """验证 upload_material_stream 会把稳定 ID 与自动解析动作传给上传服务。"""

    pipeline = CnPipeline(workspace_root=tmp_path, processor_registry=build_fins_processor_registry())

    called_args: dict[str, Any] = {}

    def _execute_upload(**kwargs: Any) -> Any:
        """记录上传调用并返回固定成功结果。"""

        called_args.clear()
        called_args.update(kwargs)
        return SimpleNamespace(status="uploaded", payload={}, file_events=[])

    pipeline._upload_service.execute_upload = _execute_upload  # type: ignore[attr-defined]
    pipeline._safe_get_document_meta = (  # type: ignore[assignment]
        lambda ticker, document_id, source_kind: {"document_id": document_id}
    )

    events = [
        event
        async for event in pipeline.upload_material_stream(
            ticker="000001",
            action=None,
            form_type="MATERIAL_OTHER",
            material_name="Deck",
            files=[],
            fiscal_year=2025,
            fiscal_period="q1",
            filing_date="2025-01-01",
            report_date="2024-12-31",
            company_id="000001",
            company_name="平安银行",
            overwrite=False,
        )
    ]
    assert events[-1].event_type == "upload_completed"
    assert called_args["action"] == "update"
    assert str(called_args["document_id"]).startswith("mat_")
    assert called_args["document_id"] == called_args["internal_document_id"]


@pytest.mark.unit
def test_process_single_source_skip_and_safe_get_meta(tmp_path: Path) -> None:
    """验证单文档处理 skip 分支与 safe_get 返回 None 分支。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = _repository(build_storage_core(tmp_path))
    _prepare_filing(repository, ticker="000001", document_id="fil_skip_ingest", ingest_complete=False)
    _prepare_material(repository, ticker="000001", document_id="mat_skip_deleted", is_deleted=True)

    pipeline = CnPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )

    filing_result = pipeline.process_filing("000001", "fil_skip_ingest", overwrite=False)
    material_result = pipeline.process_material("000001", "mat_skip_deleted", overwrite=False)

    assert filing_result["status"] == "skipped"
    assert filing_result["reason"] == "ingest_incomplete"
    assert material_result["status"] == "skipped"
    assert material_result["reason"] == "deleted"

    assert pipeline._safe_get_document_meta("000001", "not_exists", SourceKind.FILING) is None
    assert pipeline._safe_get_processed_meta("000001", "not_exists") is None
