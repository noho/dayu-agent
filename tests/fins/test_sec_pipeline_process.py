"""SecPipeline 离线快照导出测试。"""

from __future__ import annotations

import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pytest

from dayu.fins.domain.document_models import FilingCreateRequest
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.sec_pipeline import SecPipeline
from dayu.fins.processors.registry import build_fins_processor_registry
from tests.fins.storage_testkit import build_storage_core
from dayu.fins.storage.local_file_store import LocalFileStore


def _portfolio_root(repository: object) -> Path:
    """读取测试仓储 core 的 portfolio_root。"""

    return cast(Any, repository).portfolio_root


def _create_filing(repository: object, request: FilingCreateRequest) -> None:
    """调用测试仓储 core 的 create_filing。"""

    cast(Any, repository).create_filing(request)


def _prepare_filing(
    repository: object,
    ticker: str,
    document_id: str,
    filename: str,
    content: bytes,
    ingest_complete: bool = True,
) -> None:
    """准备 filings 文档与本地文件。"""

    store = LocalFileStore(root=_portfolio_root(repository), scheme="local")
    key = f"{ticker}/filings/{document_id}/{filename}"
    file_meta = store.put_object(key, BytesIO(content))
    _create_filing(
        repository,
        FilingCreateRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=document_id,
            form_type="DEF 14A",
            primary_document=filename,
            files=[file_meta],
            meta={
                "ingest_complete": ingest_complete,
                "document_version": "v1",
                "source_fingerprint": "hash_v1",
                "fiscal_year": 2025,
                "fiscal_period": "FY",
            },
        )
    )


@pytest.mark.unit
def test_process_filing_batch(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证 process 会遍历 filings 并导出离线快照。"""

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository,
        ticker="AAPL",
        document_id="fil_001",
        filename="sample.html",
        content=b"<html><h1>Section</h1><p>Text</p></html>",
        ingest_complete=True,
    )
    _prepare_filing(
        repository,
        ticker="AAPL",
        document_id="fil_002",
        filename="skip.html",
        content=b"<html><h1>Skip</h1></html>",
        ingest_complete=False,
    )

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    with caplog.at_level(logging.INFO):
        result = pipeline.process("AAPL")

    summary = result["filing_summary"]
    assert summary["total"] == 2
    assert summary["processed"] == 1
    assert summary["skipped"] == 1
    material_summary = result["material_summary"]
    assert material_summary["total"] == 0
    assert material_summary["processed"] == 0
    assert material_summary["skipped"] == 0
    assert material_summary["failed"] == 0
    assert "document_id=fil_001 status=processed form_type=DEF 14A fiscal_year=2025" in caplog.text
    assert "document_id=fil_002 status=skipped form_type=DEF 14A fiscal_year=2025 reason=ingest_incomplete" in caplog.text
    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / "fil_001"
    assert (processed_dir / "tool_snapshot_meta.json").exists()
    assert not (processed_dir / "tool_snapshot_search_document.json").exists()
    assert not (processed_dir / "tool_snapshot_query_xbrl_facts.json").exists()


@pytest.mark.unit
def test_process_batch_can_limit_to_requested_document_ids(tmp_path: Path) -> None:
    """验证 `process(document_ids=...)` 只处理指定文档。"""

    repository = build_storage_core(tmp_path)
    _prepare_filing(
        repository,
        ticker="AAPL",
        document_id="fil_001",
        filename="sample.html",
        content=b"<html><h1>Section</h1><p>Text</p></html>",
        ingest_complete=True,
    )
    _prepare_filing(
        repository,
        ticker="AAPL",
        document_id="fil_002",
        filename="skip.html",
        content=b"<html><h1>Skip</h1></html>",
        ingest_complete=True,
    )
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )

    result = pipeline.process("AAPL", document_ids=["fil_002"])

    assert result["filing_summary"]["total"] == 1
    assert result["filing_summary"]["processed"] == 1
    assert (tmp_path / "portfolio" / "AAPL" / "processed" / "fil_002").exists()
    assert not (tmp_path / "portfolio" / "AAPL" / "processed" / "fil_001").exists()


@pytest.mark.unit
def test_process_filing_ci_exports_full_snapshot_files(tmp_path: Path) -> None:
    """验证 SEC `process_filing(ci=True)` 会导出 CI 全量快照文件。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    document_id = "fil_truth"
    _prepare_filing(
        repository,
        ticker="AAPL",
        document_id=document_id,
        filename="truth.html",
        content=b"<html><h1>Section</h1><p>Text</p></html>",
        ingest_complete=True,
    )
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )

    result = pipeline.process_filing("AAPL", document_id, overwrite=False, ci=True)

    assert result["status"] == "processed"
    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / document_id
    expected_files = [
        "tool_snapshot_list_documents.json",
        "tool_snapshot_get_document_sections.json",
        "tool_snapshot_read_section.json",
        "tool_snapshot_search_document.json",
        "tool_snapshot_list_tables.json",
        "tool_snapshot_get_table.json",
        "tool_snapshot_get_page_content.json",
        "tool_snapshot_get_financial_statement.json",
        "tool_snapshot_query_xbrl_facts.json",
        "tool_snapshot_meta.json",
    ]
    for file_name in expected_files:
        assert (processed_dir / file_name).exists()

    search_payload = json.loads((processed_dir / "tool_snapshot_search_document.json").read_text(encoding="utf-8"))
    search_queries = [item["request"]["query"] for item in search_payload["calls"]]
    first_request = search_payload["calls"][0]["request"]
    assert len(search_queries) == 20
    assert "board of directors" in search_queries
    assert "executive compensation" in search_queries
    assert "beneficial ownership" in search_queries
    assert "say on pay" in search_queries
    assert first_request["query"] == first_request["query_text"]
    assert first_request["query_id"].startswith("governance_pack.q")
    assert first_request["query_intent"]
    assert first_request["query_weight"] == 1.0


@pytest.mark.unit
def test_process_filing_ci_rebuilds_snapshot_meta_on_version_skip(tmp_path: Path) -> None:
    """验证 SEC `process_filing(ci=True)` 命中版本跳过时会补齐快照文件。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    repository = build_storage_core(tmp_path)
    document_id = "fil_truth_skip"
    _prepare_filing(
        repository,
        ticker="AAPL",
        document_id=document_id,
        filename="truth_skip.html",
        content=b"<html><h1>Section</h1><p>Text</p></html>",
        ingest_complete=True,
    )
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )

    first = pipeline.process_filing("AAPL", document_id, overwrite=False, ci=True)
    assert first["status"] == "processed"
    processed_dir = tmp_path / "portfolio" / "AAPL" / "processed" / document_id
    snapshot_meta_file = processed_dir / "tool_snapshot_meta.json"
    snapshot_meta_file.unlink()

    second = pipeline.process_filing("AAPL", document_id, overwrite=False, ci=True)
    assert second["status"] == "processed"
    assert snapshot_meta_file.exists()
