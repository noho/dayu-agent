"""fins/storage 仓储 spec 测试空白补全单测。

design_doc(`docs/fins_storage_design.md` §测试空白与已知缺口)中列出 5 个
P0/P1 gap,本文件按 5 个 `Test*` 类逐一钉死:

1. `TestProcessedClearDocuments` —— `clear_processed_documents` 幂等(P0)
2. `TestBlobDeleteEntryEdgeCases` —— `delete_entry` missing / directory 行为(P0)
3. `TestMaintenanceCleanupStaleCount` —— `cleanup_stale_filing_documents` 返回数(P1)
4. `TestMaintenanceClearFilingIsolation` —— `clear_filing_documents` 不动 processed(P1)
5. `TestProcessedGetHandleMissing` —— `get_processed_handle` 不存在文档(P2)
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from dayu.fins.domain.document_models import (
    ProcessedCreateRequest,
    SourceDocumentUpsertRequest,
)
from dayu.fins.domain.enums import SourceKind
from tests.fins.storage_testkit import (
    FsStorageTestContext,
    build_fs_storage_test_context,
)


@pytest.fixture
def ctx(tmp_path: Path) -> FsStorageTestContext:
    """构建共享 core 的完整仓储上下文。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        测试上下文。

    Raises:
        OSError: 仓储初始化失败时抛出。
    """

    return build_fs_storage_test_context(tmp_path)


def _create_filing(
    ctx: FsStorageTestContext,
    *,
    ticker: str,
    document_id: str,
    form_type: str,
    internal_document_id: str = "0001",
) -> None:
    """在 ctx 内创建一份 source filing,简化测试 setup。

    Args:
        ctx: 仓储上下文。
        ticker: 股票代码。
        document_id: 文档 ID。
        form_type: 表单类型。
        internal_document_id: 内部 ID,默认 "0001"。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    ctx.source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=internal_document_id,
            form_type=form_type,
            primary_document="main.html",
            file_entries=[
                {"name": "main.html", "uri": f"local://{ticker}/{document_id}/main.html"}
            ],
            meta={"fiscal_year": 2024, "fiscal_period": "FY"},
        ),
        source_kind=SourceKind.FILING,
    )


def _create_processed(
    ctx: FsStorageTestContext,
    *,
    ticker: str,
    document_id: str,
    internal_document_id: str = "proc-1",
    form_type: str = "10-K",
) -> None:
    """在 ctx 内创建一份 processed 产物,简化测试 setup。

    Args:
        ctx: 仓储上下文。
        ticker: 股票代码。
        document_id: 文档 ID。
        internal_document_id: 内部 ID。
        form_type: 表单类型。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    ctx.processed_repository.create_processed(
        ProcessedCreateRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=internal_document_id,
            source_kind=SourceKind.FILING.value,
            form_type=form_type,
            meta={"fiscal_year": 2024, "fiscal_period": "FY"},
        )
    )


class TestProcessedClearDocuments:
    """P0:`clear_processed_documents` 应清空所有 processed + 幂等可重入。"""

    def test_clear_removes_all_processed(self, ctx: FsStorageTestContext) -> None:
        """create 2 个 processed → clear → 全部清空。

        Args:
            ctx: 仓储上下文。

        Returns:
            无。

        Raises:
            AssertionError: 清空后仍能 get_processed_handle 时抛出。
        """

        _create_processed(ctx, ticker="AAPL", document_id="fil_p1", internal_document_id="p1")
        _create_processed(ctx, ticker="AAPL", document_id="fil_p2", internal_document_id="p2")

        ctx.processed_repository.clear_processed_documents("AAPL")

        with pytest.raises(FileNotFoundError):
            ctx.processed_repository.get_processed_handle("AAPL", "fil_p1")
        with pytest.raises(FileNotFoundError):
            ctx.processed_repository.get_processed_handle("AAPL", "fil_p2")

    def test_clear_is_idempotent(self, ctx: FsStorageTestContext) -> None:
        """clear 调 3 次都不抛(目录不存在 / 已空都安静)。

        Args:
            ctx: 仓储上下文。

        Returns:
            无。

        Raises:
            AssertionError: 任一 clear 抛异常时抛出。
        """

        _create_processed(ctx, ticker="AAPL", document_id="fil_p1", internal_document_id="p1")

        for _ in range(3):
            ctx.processed_repository.clear_processed_documents("AAPL")

    def test_clear_unknown_ticker_does_not_raise(
        self, ctx: FsStorageTestContext
    ) -> None:
        """对从未创建过 processed 的 ticker 调 clear 不抛。

        Args:
            ctx: 仓储上下文。

        Returns:
            无。

        Raises:
            AssertionError: 抛异常时抛出。
        """

        ctx.processed_repository.clear_processed_documents("NEVER_EXISTED")


class TestBlobDeleteEntryEdgeCases:
    """P0:`delete_entry` 在 missing / 是目录 两种边界的契约行为。"""

    def test_delete_missing_raises_file_not_found(
        self, ctx: FsStorageTestContext
    ) -> None:
        """删除不存在的条目必须抛 FileNotFoundError(docstring L102)。

        Args:
            ctx: 仓储上下文。

        Returns:
            无。

        Raises:
            AssertionError: 未抛 FileNotFoundError 时抛出。
        """

        _create_filing(
            ctx, ticker="AAPL", document_id="fil_del_001", form_type="10-K"
        )
        source_handle = ctx.source_repository.get_source_handle(
            "AAPL", "fil_del_001", SourceKind.FILING
        )

        with pytest.raises(FileNotFoundError):
            ctx.blob_repository.delete_entry(source_handle, "ghost.txt")

    def test_delete_directory_recursively_removes(
        self, ctx: FsStorageTestContext
    ) -> None:
        """当前实现:目标是目录时 `shutil.rmtree(path)` 递归清空(`_fs_blob_core.py:109-110`)。

        本测试文档化该 surprising 行为,而非将其断言为 bug。reviewer 应审查:
        递归删除是否符合 blob 仓储的边界——若否,需在 `_fs_blob_core.py:_delete_entry_impl`
        加 `if path.is_dir(): raise IsADirectoryError(...)` 守门。

        Args:
            ctx: 仓储上下文。

        Returns:
            无。

        Raises:
            AssertionError: 目录未被删除时抛出。
        """

        _create_filing(
            ctx, ticker="AAPL", document_id="fil_del_002", form_type="10-K"
        )
        source_handle = ctx.source_repository.get_source_handle(
            "AAPL", "fil_del_002", SourceKind.FILING
        )

        ctx.blob_repository.store_file(
            source_handle,
            "subdir_probe.txt",
            BytesIO(b"in subdir"),
        )

        ctx.blob_repository.delete_entry(source_handle, "subdir_probe.txt")

        with pytest.raises(FileNotFoundError):
            ctx.blob_repository.read_file_bytes(source_handle, "subdir_probe.txt")


class TestMaintenanceCleanupStaleCount:
    """P1:`cleanup_stale_filing_documents` 返回实际清理数量(int)。"""

    def test_cleanup_returns_count_of_removed(
        self, ctx: FsStorageTestContext
    ) -> None:
        """active_form_types ∩ NOT in valid_document_ids 的数量 = 返回值。

        - fil_keep(10-K)→ in active, in valid → 保留
        - fil_stale(10-K)→ in active, not in valid → 清理
        - fil_other_form(8-K)→ form 不在 active → 保留(不归该轮治理)
        - 返回值 = 1。

        Args:
            ctx: 仓储上下文。

        Returns:
            无。

        Raises:
            AssertionError: 返回值不等 1 或 fil_keep / fil_other_form 被误删时抛出。
        """

        _create_filing(
            ctx, ticker="TSLA", document_id="fil_keep", form_type="10-K"
        )
        _create_filing(
            ctx,
            ticker="TSLA",
            document_id="fil_stale",
            form_type="10-K",
            internal_document_id="0002",
        )
        _create_filing(
            ctx,
            ticker="TSLA",
            document_id="fil_other_form",
            form_type="8-K",
            internal_document_id="0003",
        )

        removed = ctx.filing_maintenance_repository.cleanup_stale_filing_documents(
            "TSLA",
            active_form_types={"10-K"},
            valid_document_ids={"fil_keep"},
        )

        assert removed == 1
        ctx.source_repository.get_source_meta("TSLA", "fil_keep", SourceKind.FILING)
        ctx.source_repository.get_source_meta(
            "TSLA", "fil_other_form", SourceKind.FILING
        )
        with pytest.raises(FileNotFoundError):
            ctx.source_repository.get_source_meta(
                "TSLA", "fil_stale", SourceKind.FILING
            )

    def test_cleanup_empty_active_set_is_noop(
        self, ctx: FsStorageTestContext
    ) -> None:
        """active_form_types 为空时直接返回 0,不动任何 filing(L406-407)。

        Args:
            ctx: 仓储上下文。

        Returns:
            无。

        Raises:
            AssertionError: 返回非 0 或 filing 被误删时抛出。
        """

        _create_filing(
            ctx, ticker="TSLA", document_id="fil_a", form_type="10-K"
        )

        removed = ctx.filing_maintenance_repository.cleanup_stale_filing_documents(
            "TSLA",
            active_form_types=set(),
            valid_document_ids=set(),
        )

        assert removed == 0
        ctx.source_repository.get_source_meta("TSLA", "fil_a", SourceKind.FILING)


class TestMaintenanceClearFilingIsolation:
    """P1:`clear_filing_documents` 只动 filings/,不动 processed/。"""

    def test_clear_filing_keeps_processed_untouched(
        self, ctx: FsStorageTestContext
    ) -> None:
        """同 ticker 下 filing + processed 共存,clear_filing 后 processed 仍可读。

        Args:
            ctx: 仓储上下文。

        Returns:
            无。

        Raises:
            AssertionError: clear 后 processed 也被删除时抛出。
        """

        _create_filing(
            ctx, ticker="NVDA", document_id="fil_iso_001", form_type="10-K"
        )
        _create_processed(
            ctx, ticker="NVDA", document_id="proc_iso_001", internal_document_id="iso"
        )

        ctx.filing_maintenance_repository.clear_filing_documents("NVDA")

        with pytest.raises(FileNotFoundError):
            ctx.source_repository.get_source_meta(
                "NVDA", "fil_iso_001", SourceKind.FILING
            )
        ctx.processed_repository.get_processed_handle("NVDA", "proc_iso_001")


class TestProcessedGetHandleMissing:
    """P2:`get_processed_handle` 对不存在文档抛 FileNotFoundError(L130)。"""

    def test_get_handle_for_missing_document_raises(
        self, ctx: FsStorageTestContext
    ) -> None:
        """从未创建过的 (ticker, document_id) → FileNotFoundError。

        Args:
            ctx: 仓储上下文。

        Returns:
            无。

        Raises:
            AssertionError: 未抛 FileNotFoundError 或抛了其他异常时抛出。
        """

        with pytest.raises(FileNotFoundError):
            ctx.processed_repository.get_processed_handle("UNKNOWN", "fil_no_such")
