"""窄仓储拆分后的文件系统实现测试。"""

from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path

import pytest

from dayu.fins.domain.document_models import (
    CompanyMeta,
    DocumentQuery,
    ProcessedCreateRequest,
    RejectedFilingArtifactUpsertRequest,
    SourceFileEntry,
    SourceDocumentStateChangeRequest,
    SourceDocumentUpsertRequest,
    now_iso8601,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from dayu.fins.storage.fs_company_meta_repository import FsCompanyMetaRepository
from dayu.fins.storage.fs_document_blob_repository import FsDocumentBlobRepository
from dayu.fins.storage.fs_filing_maintenance_repository import FsFilingMaintenanceRepository
from dayu.fins.storage.fs_processed_document_repository import FsProcessedDocumentRepository
from dayu.fins.storage.fs_source_document_repository import FsSourceDocumentRepository


@pytest.fixture
def fs_repositories(tmp_path: Path) -> tuple[
    FsCompanyMetaRepository,
    FsSourceDocumentRepository,
    FsProcessedDocumentRepository,
    FsDocumentBlobRepository,
    FsFilingMaintenanceRepository,
]:
    """构建共享 core 的窄仓储集合。"""

    repository_set = build_fs_repository_set(workspace_root=tmp_path)
    return (
        FsCompanyMetaRepository(tmp_path, repository_set=repository_set),
        FsSourceDocumentRepository(tmp_path, repository_set=repository_set),
        FsProcessedDocumentRepository(tmp_path, repository_set=repository_set),
        FsDocumentBlobRepository(tmp_path, repository_set=repository_set),
        FsFilingMaintenanceRepository(tmp_path, repository_set=repository_set),
    )


def test_company_meta_repository_roundtrip(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证公司元数据窄仓储可独立读写。"""

    company_repository, _, _, _, _ = fs_repositories
    meta = CompanyMeta(
        company_id="320193",
        company_name="Apple Inc.",
        ticker="AAPL",
        market="US",
        resolver_version="test",
        updated_at=now_iso8601(),
        ticker_aliases=["AAPL"],
    )
    company_repository.upsert_company_meta(meta)
    loaded = company_repository.get_company_meta("AAPL")
    assert loaded.company_id == "320193"
    assert loaded.company_name == "Apple Inc."


def test_company_meta_repository_scan_inventory_records_skipped_directories(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证公司目录扫描会记录隐藏目录、缺失 meta 与可用公司。

    Args:
        fs_repositories: 共享 core 的窄仓储集合。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    company_repository, _, _, _, _ = fs_repositories
    workspace_root = company_repository._repository_set.core.workspace_root
    (workspace_root / ".dayu" / "repo_batches").mkdir(parents=True, exist_ok=True)
    (workspace_root / "portfolio" / "NO_META").mkdir(parents=True, exist_ok=True)
    company_repository.upsert_company_meta(
        CompanyMeta(
            company_id="320193",
            company_name="Apple Inc.",
            ticker="AAPL",
            market="US",
            resolver_version="test",
            updated_at=now_iso8601(),
        )
    )

    entries = company_repository.scan_company_meta_inventory()
    by_name = {entry.directory_name: entry for entry in entries}

    assert by_name[".dayu"].status == "hidden_directory"
    assert by_name["NO_META"].status == "missing_meta"
    assert by_name["AAPL"].status == "available"
    assert by_name["AAPL"].company_meta is not None


def test_company_meta_repository_resolves_existing_ticker_via_alias_and_detects_conflicts(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证公司元数据仓储会按 alias 解析 ticker，并在 alias 冲突时显式失败。"""

    company_repository, _, _, _, _ = fs_repositories
    company_repository.upsert_company_meta(
        CompanyMeta(
            company_id="320193",
            company_name="Apple Inc.",
            ticker="AAPL",
            market="US",
            resolver_version="test",
            updated_at=now_iso8601(),
            ticker_aliases=["AAPL", "APC"],
        )
    )
    company_repository.upsert_company_meta(
        CompanyMeta(
            company_id="0002969",
            company_name="Air Products",
            ticker="APD",
            market="US",
            resolver_version="test",
            updated_at=now_iso8601(),
            ticker_aliases=["APD", "APC"],
        )
    )

    assert company_repository.resolve_existing_ticker(["aapl"]) == "AAPL"
    with pytest.raises(ValueError, match="命中多个公司目录"):
        company_repository.resolve_existing_ticker(["apc"])


def test_company_meta_core_private_helpers_cover_cache_meta_scan_and_active_batch_view(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证公司元数据 core helper 的缓存、alias 索引与 batch 读视图。"""

    company_repository, _, _, _, _ = fs_repositories
    core = company_repository._repository_set.core
    company_repository.upsert_company_meta(
        CompanyMeta(
            company_id="320193",
            company_name="Apple Inc.",
            ticker="AAPL",
            market="US",
            resolver_version="test",
            updated_at=now_iso8601(),
            ticker_aliases=["AAPL", "APPLE"],
        )
    )
    company_repository.upsert_company_meta(
        CompanyMeta(
            company_id="9988",
            company_name="Alibaba Group",
            ticker="BABA",
            market="US",
            resolver_version="test",
            updated_at=now_iso8601(),
            ticker_aliases=["BABA", "9988.HK"],
        )
    )

    alias_index = core._build_company_alias_index()
    cached_meta = core._get_cached_company_meta("AAPL")
    built_alias_index = core._build_company_alias_index_from_meta(
        {
            "AAPL": CompanyMeta(
                company_id="320193",
                company_name="Apple Inc.",
                ticker="AAPL",
                market="US",
                resolver_version="test",
                updated_at=now_iso8601(),
                ticker_aliases=["APPLE"],
            )
        }
    )
    scanned_meta = core._scan_company_meta_by_ticker()

    token = core.begin_batch("MSFT")
    company_repository.upsert_company_meta(
        CompanyMeta(
            company_id="789019",
            company_name="Microsoft",
            ticker="MSFT",
            market="US",
            resolver_version="test",
            updated_at=now_iso8601(),
            ticker_aliases=["MSFT", "MICROSOFT"],
        )
    )
    readable_dirs = core._collect_readable_ticker_dirs()
    scanned_with_batch = core._scan_company_meta_by_ticker()
    core.rollback_batch(token)

    assert alias_index["APPLE"] == ["AAPL"]
    assert alias_index["9988"] == ["BABA"]
    assert cached_meta is not None
    assert cached_meta.company_name == "Apple Inc."
    assert built_alias_index == {"AAPL": ["AAPL"], "APPLE": ["AAPL"]}
    assert sorted(scanned_meta) == ["AAPL", "BABA"]
    assert readable_dirs["MSFT"] == token.staging_ticker_dir
    assert scanned_with_batch["MSFT"].company_name == "Microsoft"


def test_source_document_repository_supports_generic_create_delete_restore(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证源文档窄仓储支持统一的 create/delete/restore 接口。"""

    _, source_repository, _, _, _ = fs_repositories
    source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="AAPL",
            document_id="fil_1",
            internal_document_id="0001",
            form_type="10-K",
            primary_document="main.html",
            file_entries=[{"name": "main.html", "uri": "local://AAPL/fil_1/main.html"}],
            meta={"fiscal_year": 2024, "fiscal_period": "FY"},
        ),
        source_kind=SourceKind.FILING,
    )
    meta = source_repository.get_source_meta("AAPL", "fil_1", SourceKind.FILING)
    assert meta["internal_document_id"] == "0001"

    source_repository.delete_source_document(
        SourceDocumentStateChangeRequest(
            ticker="AAPL",
            document_id="fil_1",
            source_kind=SourceKind.FILING.value,
        )
    )
    deleted_meta = source_repository.get_source_meta("AAPL", "fil_1", SourceKind.FILING)
    assert deleted_meta["is_deleted"] is True

    source_repository.restore_source_document(
        SourceDocumentStateChangeRequest(
            ticker="AAPL",
            document_id="fil_1",
            source_kind=SourceKind.FILING.value,
        )
    )
    restored_meta = source_repository.get_source_meta("AAPL", "fil_1", SourceKind.FILING)
    assert restored_meta["is_deleted"] is False


def test_document_blob_repository_can_store_and_read_source_file(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证文件对象窄仓储可对源文档读写二进制内容。"""

    _, source_repository, _, blob_repository, _ = fs_repositories
    source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="AAPL",
            document_id="fil_blob",
            internal_document_id="blob-1",
            form_type="10-Q",
            primary_document="main.txt",
            meta={},
        ),
        source_kind=SourceKind.FILING,
    )
    handle = source_repository.get_source_handle("AAPL", "fil_blob", SourceKind.FILING)
    blob_repository.store_file(
        handle=handle,
        filename="main.txt",
        data=BytesIO(b"hello world"),
        content_type="text/plain",
    )
    assert blob_repository.read_file_bytes(handle, "main.txt") == b"hello world"


def test_source_document_repository_can_resolve_primary_file_and_source(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证源文档仓储可解析主文件元数据并返回指定文件 Source。

    Args:
        fs_repositories: 共享 core 的窄仓储集合。

    Returns:
        无。

    Raises:
        无。
    """

    _, source_repository, _, blob_repository, _ = fs_repositories
    source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="AAPL",
            document_id="fil_source",
            internal_document_id="source-1",
            form_type="10-K",
            primary_document="main.txt",
            meta={},
        ),
        source_kind=SourceKind.FILING,
    )
    handle = source_repository.get_source_handle("AAPL", "fil_source", SourceKind.FILING)
    stored_file = blob_repository.store_file(
        handle=handle,
        filename="main.txt",
        data=BytesIO(b"primary source body"),
        content_type="text/plain",
    )
    source_repository.update_source_document(
        SourceDocumentUpsertRequest(
            ticker="AAPL",
            document_id="fil_source",
            internal_document_id="source-1",
            form_type="10-K",
            primary_document="main.txt",
            meta={},
            files=[stored_file],
        ),
        source_kind=SourceKind.FILING,
    )

    primary_file = source_repository.get_primary_file("AAPL", "fil_source", SourceKind.FILING)
    assert primary_file.uri.endswith("main.txt")

    source = source_repository.get_source("AAPL", "fil_source", SourceKind.FILING, "main.txt")
    with source.open() as stream:
        assert stream.read() == b"primary source body"


def test_source_document_repository_exposes_storage_root_and_xbrl_instance_fact(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证源文档仓储统一承载 filing 根目录与 XBRL instance 存在性事实。"""

    _, source_repository, _, blob_repository, _ = fs_repositories
    assert source_repository.has_source_storage_root("AAPL", SourceKind.FILING) is False

    source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="AAPL",
            document_id="fil_xbrl",
            internal_document_id="xbrl-1",
            form_type="10-K",
            primary_document="main.htm",
            meta={},
        ),
        source_kind=SourceKind.FILING,
    )
    handle = source_repository.get_source_handle("AAPL", "fil_xbrl", SourceKind.FILING)
    blob_repository.store_file(
        handle=handle,
        filename="sample_htm.xml",
        data=BytesIO(b"<xbrl></xbrl>"),
        content_type="application/xml",
    )

    assert source_repository.has_source_storage_root("AAPL", SourceKind.FILING) is True
    assert source_repository.has_filing_xbrl_instance("AAPL", "fil_xbrl") is True


def test_processed_document_repository_uses_explicit_processed_listing(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证 processed 窄仓储通过显式 list_processed_documents 查询。"""

    _, _, processed_repository, _, _ = fs_repositories
    processed_repository.create_processed(
        ProcessedCreateRequest(
            ticker="AAPL",
            document_id="fil_proc_1",
            internal_document_id="proc-1",
            source_kind=SourceKind.FILING.value,
            form_type="10-K",
            meta={"fiscal_year": 2024, "fiscal_period": "FY"},
            sections=[{"ref": "s1", "title": "Overview"}],
            tables=[],
            financials=None,
        )
    )
    summaries = processed_repository.list_processed_documents("AAPL", DocumentQuery())
    assert [summary.document_id for summary in summaries] == ["fil_proc_1"]

    processed_repository.mark_processed_reprocess_required("AAPL", "fil_proc_1", True)
    processed_meta = processed_repository.get_processed_meta("AAPL", "fil_proc_1")
    assert processed_meta["reprocess_required"] is True


def test_filing_maintenance_repository_persists_rejection_registry(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证 filing 维护仓储可保存与读取 rejection registry。"""

    _, _, _, _, maintenance_repository = fs_repositories
    registry = {
        "fil_1": {
            "reason": "filtered",
            "category": "test",
            "form_type": "6-K",
            "filing_date": "2024-01-01",
        }
    }
    maintenance_repository.save_download_rejection_registry("AAPL", registry)
    assert maintenance_repository.load_download_rejection_registry("AAPL") == registry


def test_filing_maintenance_repository_roundtrip_rejected_artifact(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证 filing 维护仓储可独立保存 rejected artifact。"""

    company_repository, _, _, _, maintenance_repository = fs_repositories
    maintenance_repository.upsert_rejected_filing_artifact(
        RejectedFilingArtifactUpsertRequest(
            ticker="AAPL",
            document_id="fil_rej_1",
            internal_document_id="0001",
            accession_number="0001",
            company_id="320193",
            form_type="6-K",
            filing_date="2025-01-01",
            report_date="2024-12-31",
            primary_document="cover.htm",
            selected_primary_document="ex99.htm",
            rejection_reason="6k_filtered",
            rejection_category="NO_MATCH",
            classification_version="sec_pipeline_download_v1.1.0",
            source_fingerprint="fp-1",
            files=[
                SourceFileEntry(
                    name="cover.htm",
                    uri="local://AAPL/filings/.rejections/fil_rej_1/cover.htm",
                )
            ],
        )
    )

    artifact = maintenance_repository.get_rejected_filing_artifact("AAPL", "fil_rej_1")
    listed = maintenance_repository.list_rejected_filing_artifacts("AAPL")

    assert artifact.document_id == "fil_rej_1"
    assert artifact.rejection_reason == "6k_filtered"
    assert artifact.files[0].name == "cover.htm"
    assert [item.document_id for item in listed] == ["fil_rej_1"]
    manifest_path = company_repository._repository_set.core.workspace_root / "portfolio" / "AAPL" / "filings" / "filing_manifest.json"
    assert manifest_path.exists() is False


def test_source_document_repository_meta_and_listing_error_paths(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证源文档仓储的缺失 meta、替换与过滤列表分支。"""

    _, source_repository, processed_repository, _, _ = fs_repositories
    source_core = source_repository._repository_set.core
    source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="AAPL",
            document_id="mat_1",
            internal_document_id="mat-1",
            form_type="PRESENTATION",
            primary_document="deck.pdf",
            meta={},
        ),
        source_kind=SourceKind.MATERIAL,
    )
    processed_repository.create_processed(
        ProcessedCreateRequest(
            ticker="AAPL",
            document_id="fil_proc_filtered",
            internal_document_id="proc-filtered",
            source_kind=SourceKind.FILING.value,
            form_type="10-Q",
            meta={"fiscal_year": 2024, "fiscal_period": "Q2"},
            sections=[],
            tables=[],
            financials=None,
        )
    )

    material_meta = source_core.get_document_meta("AAPL", "mat_1")
    source_repository.replace_source_meta(
        "AAPL",
        "mat_1",
        SourceKind.MATERIAL,
        {**material_meta, "custom": "updated"},
    )
    replaced_meta = source_repository.get_source_meta("AAPL", "mat_1", SourceKind.MATERIAL)
    filtered_documents = source_core.list_documents(
        "AAPL",
        DocumentQuery(source_kind=SourceKind.FILING, form_type="10-Q", fiscal_years=[2024], fiscal_periods=["Q2"]),
    )

    assert replaced_meta["custom"] == "updated"
    assert [item.document_id for item in filtered_documents] == ["fil_proc_filtered"]

    with pytest.raises(FileNotFoundError, match="meta.json"):
        source_core.get_document_meta("AAPL", "missing_doc")
    with pytest.raises(FileNotFoundError, match="material"):
        source_repository.get_source_meta("AAPL", "missing_doc", SourceKind.MATERIAL)
    with pytest.raises(FileNotFoundError, match="material"):
        source_repository.replace_source_meta("AAPL", "missing_doc", SourceKind.MATERIAL, {})


def test_source_document_repository_directory_and_filing_checks_cover_errors(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证根目录和 filing 目录的异常分支。"""

    _, source_repository, _, _, _ = fs_repositories
    workspace_root = source_repository._repository_set.core.workspace_root
    filing_root = workspace_root / "portfolio" / "AAPL" / "filings"
    filing_root.parent.mkdir(parents=True, exist_ok=True)
    filing_root.write_text("not-a-directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="source root"):
        source_repository.has_source_storage_root("AAPL", SourceKind.FILING)

    filing_root.unlink()
    filing_dir = filing_root / "fil_missing"
    filing_dir.parent.mkdir(parents=True, exist_ok=True)
    filing_dir.write_text("not-a-directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError, match="filing 路径不是目录"):
        source_repository.has_filing_xbrl_instance("AAPL", "fil_missing")
    filing_dir.unlink()
    with pytest.raises(FileNotFoundError, match="filing 目录不存在"):
        source_repository.has_filing_xbrl_instance("AAPL", "fil_missing")


def test_source_document_repository_reset_source_document_removes_directory_and_manifest_entry(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证单文档重置会删除目录并同步清理 manifest 条目。

    Args:
        fs_repositories: 共享 core 的窄仓储集合。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    _, source_repository, _, _, _ = fs_repositories
    source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="AAPL",
            document_id="fil_reset_dir",
            internal_document_id="reset-dir-1",
            form_type="10-K",
            primary_document="main.pdf",
            meta={},
        ),
        source_kind=SourceKind.FILING,
    )
    workspace_root = source_repository._repository_set.core.workspace_root
    document_dir = workspace_root / "portfolio" / "AAPL" / "filings" / "fil_reset_dir"

    source_repository.reset_source_document("AAPL", "fil_reset_dir", SourceKind.FILING)

    assert document_dir.exists() is False
    assert source_repository.list_source_document_ids("AAPL", SourceKind.FILING) == []
    with pytest.raises(FileNotFoundError, match="filing"):
        source_repository.get_source_meta("AAPL", "fil_reset_dir", SourceKind.FILING)


def test_source_document_repository_reset_source_document_unlinks_file_target(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证单文档重置在目标路径是文件时会直接 unlink 并清理 manifest。

    Args:
        fs_repositories: 共享 core 的窄仓储集合。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    _, source_repository, _, _, _ = fs_repositories
    source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="AAPL",
            document_id="fil_reset_file",
            internal_document_id="reset-file-1",
            form_type="10-Q",
            primary_document="main.pdf",
            meta={},
        ),
        source_kind=SourceKind.FILING,
    )
    workspace_root = source_repository._repository_set.core.workspace_root
    document_path = workspace_root / "portfolio" / "AAPL" / "filings" / "fil_reset_file"
    shutil.rmtree(document_path)
    document_path.write_text("not-a-directory", encoding="utf-8")

    source_repository.reset_source_document("AAPL", "fil_reset_file", SourceKind.FILING)

    assert document_path.exists() is False
    assert source_repository.list_source_document_ids("AAPL", SourceKind.FILING) == []


def test_source_document_repository_reset_source_document_tolerates_missing_target(
    fs_repositories: tuple[
        FsCompanyMetaRepository,
        FsSourceDocumentRepository,
        FsProcessedDocumentRepository,
        FsDocumentBlobRepository,
        FsFilingMaintenanceRepository,
    ],
) -> None:
    """验证单文档重置在目录与 manifest 都不存在时保持幂等。

    Args:
        fs_repositories: 共享 core 的窄仓储集合。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    _, source_repository, _, _, _ = fs_repositories

    source_repository.reset_source_document("AAPL", "fil_missing_reset", SourceKind.FILING)

    assert source_repository.list_source_document_ids("AAPL", SourceKind.FILING) == []
