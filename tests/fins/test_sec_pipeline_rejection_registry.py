"""拒绝注册表（Rejection Registry）单元测试。

覆盖 sec_pipeline 模块中的拒绝注册表辅助函数：
- _load_rejection_registry
- _save_rejection_registry
- _is_rejected
- _record_rejection

以及与下载流程集成的路径：
- _download_single_filing_stream 中 rejection_registry 跳过路径
- _filter_sc13_by_direction 中 rejection_registry 跳过路径
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, cast

import pytest

from dayu.fins.downloaders.sec_downloader import (
    DownloaderEvent,
    RemoteFileDescriptor,
    Sc13PartyRoles,
)
from dayu.fins.pipelines import sec_pipeline
from dayu.fins.pipelines.sec_download_state import (
    _load_rejection_registry,
    _save_rejection_registry,
)
from dayu.fins.pipelines.sec_filing_collection import FilingRecord
from dayu.fins.pipelines.sec_pipeline import (
    SecPipeline,
    _is_rejected,
    _record_rejection,
)
from dayu.fins.pipelines.sec_sc13_filtering import (
    SecSc13WorkflowHost as _SecSc13WorkflowHost,
    filter_sc13_by_direction as _filter_sc13_by_direction_impl,
)
from dayu.fins.processors.registry import build_fins_processor_registry
from tests.fins.storage_testkit import build_fs_storage_test_context


# ---------------------------------------------------------------------------
# 测试用桩
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _load_rejection_registry / _save_rejection_registry
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_load_empty_when_file_not_exists(tmp_path: Path) -> None:
    """文件不存在时返回空字典。"""
    repository = build_fs_storage_test_context(tmp_path).filing_maintenance_repository
    registry = _load_rejection_registry(repository, "AAPL")
    assert registry == {}


@pytest.mark.unit
def test_load_empty_when_file_is_invalid_json(tmp_path: Path) -> None:
    """文件包含非法 JSON 时返回空字典。"""
    repository = build_fs_storage_test_context(tmp_path).filing_maintenance_repository
    path = tmp_path / "portfolio" / "AAPL" / "filings" / "_download_rejections.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {{{", encoding="utf-8")
    registry = _load_rejection_registry(repository, "AAPL")
    assert registry == {}


@pytest.mark.unit
def test_load_empty_when_file_is_json_array(tmp_path: Path) -> None:
    """文件包含 JSON 数组（非字典）时返回空字典。"""
    repository = build_fs_storage_test_context(tmp_path).filing_maintenance_repository
    path = tmp_path / "portfolio" / "AAPL" / "filings" / "_download_rejections.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")
    registry = _load_rejection_registry(repository, "AAPL")
    assert registry == {}


@pytest.mark.unit
def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    """保存后加载应返回相同内容。"""
    repository = build_fs_storage_test_context(tmp_path).filing_maintenance_repository
    registry = {
        "fil_0001234567-25-000001": {
            "reason": "6k_filtered",
            "category": "6k_filter",
            "form_type": "6-K",
            "filing_date": "2025-01-15",
            "download_version": sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION,
        }
    }
    _save_rejection_registry(repository, "ABEV", registry)
    loaded = _load_rejection_registry(repository, "ABEV")
    assert loaded == registry


@pytest.mark.unit
def test_save_creates_parent_directories(tmp_path: Path) -> None:
    """保存时自动创建父目录。"""
    repository = build_fs_storage_test_context(tmp_path).filing_maintenance_repository
    registry = {"fil_test": {"reason": "test", "download_version": "v1"}}
    _save_rejection_registry(repository, "NEWCO", registry)
    path = tmp_path / "portfolio" / "NEWCO" / "filings" / "_download_rejections.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == registry


# ---------------------------------------------------------------------------
# _is_rejected
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_rejected_returns_false_when_overwrite() -> None:
    """overwrite=True 时始终返回 False。"""
    registry = {
        "fil_test": {
            "download_version": sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION,
        }
    }
    assert _is_rejected(registry, "fil_test", overwrite=True) is False


@pytest.mark.unit
def test_is_rejected_returns_false_when_not_in_registry() -> None:
    """document_id 不在注册表中返回 False。"""
    assert _is_rejected({}, "fil_test", overwrite=False) is False


@pytest.mark.unit
def test_is_rejected_returns_false_when_version_mismatch() -> None:
    """download_version 不匹配时返回 False（版本升级使旧拒绝失效）。"""
    registry = {
        "fil_test": {
            "download_version": "old_version_v0.9.0",
        }
    }
    assert _is_rejected(registry, "fil_test", overwrite=False) is False


@pytest.mark.unit
def test_is_rejected_returns_true_when_matching() -> None:
    """document_id 在注册表中且版本匹配时返回 True。"""
    registry = {
        "fil_test": {
            "download_version": sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION,
        }
    }
    assert _is_rejected(registry, "fil_test", overwrite=False) is True


@pytest.mark.unit
def test_is_rejected_handles_missing_version_field() -> None:
    """注册表条目缺少 download_version 字段时返回 False。"""
    registry = {
        "fil_test": {
            "reason": "6k_filtered",
        }
    }
    assert _is_rejected(registry, "fil_test", overwrite=False) is False


# ---------------------------------------------------------------------------
# _record_rejection
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_record_rejection_adds_entry() -> None:
    """记录拒绝后注册表应包含条目。"""
    registry: dict[str, dict[str, str]] = {}
    _record_rejection(
        registry,
        document_id="fil_0001234567-25-000001",
        reason="6k_filtered",
        category="6k_filter",
        form_type="6-K",
        filing_date="2025-01-15",
    )
    assert "fil_0001234567-25-000001" in registry
    entry = registry["fil_0001234567-25-000001"]
    assert entry["reason"] == "6k_filtered"
    assert entry["category"] == "6k_filter"
    assert entry["form_type"] == "6-K"
    assert entry["filing_date"] == "2025-01-15"
    assert entry["download_version"] == sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION


@pytest.mark.unit
def test_record_rejection_overwrites_existing() -> None:
    """重复记录同一 document_id 时覆盖旧条目。"""
    registry: dict[str, dict[str, str]] = {
        "fil_test": {
            "reason": "old_reason",
            "category": "old_cat",
            "form_type": "6-K",
            "filing_date": "2024-01-01",
            "download_version": "old_version",
        }
    }
    _record_rejection(
        registry,
        document_id="fil_test",
        reason="sc13_direction_rejected",
        category="sc13_direction",
        form_type="SC 13D",
        filing_date="2025-06-01",
    )
    entry = registry["fil_test"]
    assert entry["reason"] == "sc13_direction_rejected"
    assert entry["download_version"] == sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION


# ---------------------------------------------------------------------------
# _filter_sc13_by_direction 集成（rejection_registry 跳过路径）
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_path: Path) -> SecPipeline:
    """创建最小 SecPipeline 用于方法测试。"""
    return SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )


def _make_filing(
    form_type: str = "SC 13D",
    accession: str = "0001234567-25-000001",
    filing_date: str = "2025-01-15",
) -> FilingRecord:
    """创建测试用 FilingRecord。"""
    return FilingRecord(
        form_type=form_type,
        filing_date=filing_date,
        report_date=None,
        accession_number=accession,
        primary_document="sc13d.htm",
        filer_key="005-12345",
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_filter_sc13_by_direction_skips_rejected(tmp_path: Path) -> None:
    """rejection_registry 中的 SC13 应被跳过（不调用远程接口）。"""
    pipeline = _make_pipeline(tmp_path)
    filing = _make_filing()
    doc_id = f"fil_{filing.accession_number}"
    # 预填充拒绝注册表
    registry: dict[str, dict[str, str]] = {
        doc_id: {
            "reason": "sc13_direction_rejected",
            "category": "sc13_direction",
            "form_type": "SC 13D",
            "filing_date": "2025-01-15",
            "download_version": sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION,
        }
    }
    result = await _filter_sc13_by_direction_impl(
        cast(_SecSc13WorkflowHost, pipeline),
        ticker="AAPL",
        filings=[filing],
        target_cik="320193",
        archive_cik="320193",
        rejection_registry=registry,
        overwrite=False,
    )
    # SC13 被拒绝注册表过滤掉
    assert len(result) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_filter_sc13_by_direction_ignores_registry_in_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """overwrite=True 时应忽略 rejection_registry；结果取决于实际方向判定。"""
    pipeline = _make_pipeline(tmp_path)
    filing = _make_filing()
    doc_id = f"fil_{filing.accession_number}"
    registry: dict[str, dict[str, str]] = {
        doc_id: {
            "reason": "sc13_direction_rejected",
            "category": "sc13_direction",
            "form_type": "SC 13D",
            "filing_date": "2025-01-15",
            "download_version": sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION,
        }
    }
    # mock _should_keep_sc13_direction 使其返回 True（方向匹配）
    async def fake_keep(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(pipeline, "_should_keep_sc13_direction", fake_keep)
    result = await _filter_sc13_by_direction_impl(
        cast(_SecSc13WorkflowHost, pipeline),
        ticker="AAPL",
        filings=[filing],
        target_cik="320193",
        archive_cik="320193",
        rejection_registry=registry,
        overwrite=True,
    )
    # overwrite 跳过注册表，方向匹配 → 保留
    assert len(result) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_filter_sc13_non_sc13_filings_pass_through(tmp_path: Path) -> None:
    """非 SC13 filings 不受 rejection_registry 影响。"""
    pipeline = _make_pipeline(tmp_path)
    ten_k = FilingRecord(
        form_type="10-K",
        filing_date="2025-01-15",
        report_date=None,
        accession_number="0001234567-25-000010",
        primary_document="10k.htm",
    )
    result = await _filter_sc13_by_direction_impl(
        cast(_SecSc13WorkflowHost, pipeline),
        ticker="AAPL",
        filings=[ten_k],
        target_cik="320193",
        archive_cik="320193",
        rejection_registry={},
        overwrite=False,
    )
    assert len(result) == 1
    assert result[0].form_type == "10-K"


# ---------------------------------------------------------------------------
# _should_keep_sc13_direction 中 rejection_registry 预检查
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_should_keep_sc13_direction_skips_via_registry(tmp_path: Path) -> None:
    """已在 rejection_registry 中的 SC13 直接返回 False，不发起远程调用。"""
    pipeline = _make_pipeline(tmp_path)
    filing = _make_filing()
    doc_id = f"fil_{filing.accession_number}"
    registry: dict[str, dict[str, str]] = {
        doc_id: {
            "reason": "sc13_direction_rejected",
            "category": "sc13_direction",
            "form_type": "SC 13D",
            "filing_date": "2025-01-15",
            "download_version": sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION,
        }
    }
    # 不 mock fetch_sc13_party_roles — 如果被调用会抛出（因为没有真正的下载器）
    result = await pipeline._should_keep_sc13_direction(
        ticker="AAPL",
        filing=filing,
        archive_cik="320193",
        target_cik="320193",
        rejection_registry=registry,
        overwrite=False,
    )
    assert result is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_should_keep_sc13_direction_records_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """方向不匹配时应写入 rejection_registry。"""
    pipeline = _make_pipeline(tmp_path)
    filing = _make_filing()

    # mock fetch_sc13_party_roles → 返回"我持股别人"（方向不匹配）
    async def fake_fetch(*args: Any, **kwargs: Any) -> Sc13PartyRoles:
        return Sc13PartyRoles(
            subject_cik="999999",  # 非目标 CIK
            filed_by_cik="320193",  # 目标 CIK（=我方持股）
        )

    monkeypatch.setattr(pipeline._downloader, "fetch_sc13_party_roles", fake_fetch)
    monkeypatch.setattr(
        pipeline._downloader,
        "list_filing_files",
        lambda **kwargs: [
            RemoteFileDescriptor(
                name="sc13d.htm",
                source_url="https://example.com/sc13d.htm",
                http_etag="etag-1",
                http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
                remote_size=10,
                http_status=200,
            )
        ],
    )

    def fake_download_files(
        remote_files: list[RemoteFileDescriptor],
        overwrite: bool,
        store_file: Any,
        existing_files: Optional[dict[str, dict[str, Any]]] = None,
        primary_document: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        del overwrite, existing_files, primary_document
        results: list[dict[str, Any]] = []
        for item in remote_files:
            file_meta = store_file(item.name, BytesIO(b"<html><body>SC13</body></html>"))
            results.append(
                {
                    "name": item.name,
                    "status": "downloaded",
                    "file_meta": file_meta,
                    "source_url": item.source_url,
                    "http_etag": item.http_etag,
                    "http_last_modified": item.http_last_modified,
                }
            )
        return results

    async def fake_download_files_stream(
        remote_files: list[RemoteFileDescriptor],
        overwrite: bool,
        store_file: Any,
        existing_files: Optional[dict[str, dict[str, Any]]] = None,
        primary_document: Optional[str] = None,
    ) -> Any:
        """模拟流式下载 rejected artifact。

        Args:
            remote_files: 远端文件列表。
            overwrite: 是否覆盖。
            store_file: 文件落盘回调。
            existing_files: 已存在文件索引。
            primary_document: 主文档文件名。

        Yields:
            与下载器一致的下载事件字典。

        Raises:
            无。
        """

        del overwrite, existing_files, primary_document
        for item in remote_files:
            file_meta = store_file(item.name, BytesIO(b"<html><body>SC13</body></html>"))
            yield DownloaderEvent(
                event_type="file_downloaded",
                name=item.name,
                source_url=item.source_url,
                http_etag=item.http_etag,
                http_last_modified=item.http_last_modified,
                http_status=item.http_status,
                file_meta=file_meta,
            )

    monkeypatch.setattr(pipeline._downloader, "download_files", fake_download_files)
    monkeypatch.setattr(
        pipeline._downloader,
        "download_files_stream",
        fake_download_files_stream,
    )

    registry: dict[str, dict[str, str]] = {}
    result = await pipeline._should_keep_sc13_direction(
        ticker="AAPL",
        filing=filing,
        archive_cik="320193",
        target_cik="320193",
        rejection_registry=registry,
        overwrite=False,
    )
    assert result is False
    doc_id = f"fil_{filing.accession_number}"
    assert doc_id in registry
    assert registry[doc_id]["reason"] == "sc13_direction_rejected"
    artifact_path = (
        tmp_path
        / "portfolio"
        / "AAPL"
        / "filings"
        / ".rejections"
        / doc_id
        / "meta.json"
    )
    assert artifact_path.exists()


# ---------------------------------------------------------------------------
# 端到端：save → load → is_rejected 完整流程
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_full_registry_lifecycle(tmp_path: Path) -> None:
    """完整生命周期：记录 → 保存 → 加载 → 验证拒绝 → 版本升级失效。"""
    ticker = "ABEV"
    repository = build_fs_storage_test_context(tmp_path).filing_maintenance_repository
    registry: dict[str, dict[str, str]] = {}

    # 记录两条拒绝
    _record_rejection(registry, "fil_001", "6k_filtered", "6k_filter", "6-K", "2025-01-01")
    _record_rejection(registry, "fil_002", "sc13_direction_rejected", "sc13_direction", "SC 13D", "2025-02-01")

    # 保存并加载
    _save_rejection_registry(repository, ticker, registry)
    loaded = _load_rejection_registry(repository, ticker)

    # 验证拒绝
    assert _is_rejected(loaded, "fil_001", overwrite=False) is True
    assert _is_rejected(loaded, "fil_002", overwrite=False) is True
    assert _is_rejected(loaded, "fil_003", overwrite=False) is False

    # overwrite 旁路
    assert _is_rejected(loaded, "fil_001", overwrite=True) is False

    # 模拟版本升级 → 旧拒绝失效
    original_version = sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION
    try:
        sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION = "sec_pipeline_download_v2.0.0"  # type: ignore[assignment]
        assert _is_rejected(loaded, "fil_001", overwrite=False) is False
    finally:
        sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION = original_version  # type: ignore[assignment]
