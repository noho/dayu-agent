"""财报仓储窄协议定义。

该模块按真实职责簇拆分财报仓储协议，避免单一仓储同时承担：
- 批处理事务
- 公司级元数据
- 源文档 CRUD
- processed 产物 CRUD
- 文件对象读写
- filing 维护治理
"""

from __future__ import annotations

from typing import BinaryIO, Optional, Protocol

from dayu.engine.processors.source import Source
from dayu.fins.domain.document_models import (
    BatchToken,
    CompanyMeta,
    CompanyMetaInventoryEntry,
    DocumentEntry,
    DocumentHandle,
    DocumentMeta,
    DocumentQuery,
    DocumentSummary,
    FileObjectMeta,
    ProcessedCreateRequest,
    ProcessedDeleteRequest,
    ProcessedHandle,
    RejectedFilingArtifact,
    RejectedFilingArtifactUpsertRequest,
    ProcessedUpdateRequest,
    SourceDocumentStateChangeRequest,
    SourceDocumentUpsertRequest,
    SourceHandle,
)
from dayu.fins.domain.enums import SourceKind


class BatchingRepositoryProtocol(Protocol):
    """批处理事务仓储协议。"""

    def begin_batch(self, ticker: str) -> BatchToken:
        """开启批处理事务。"""
        ...

    def commit_batch(self, token: BatchToken) -> None:
        """提交批处理事务。"""
        ...

    def rollback_batch(self, token: BatchToken) -> None:
        """回滚批处理事务。"""
        ...

    def recover_orphan_batches(self, *, dry_run: bool = False) -> tuple[str, ...]:
        """恢复异常退出后遗留的孤儿 batch/backup。"""
        ...


class CompanyMetaRepositoryProtocol(Protocol):
    """公司级元数据仓储协议。"""

    def scan_company_meta_inventory(self) -> list[CompanyMetaInventoryEntry]:
        """扫描公司目录并返回元数据盘点结果。"""
        ...

    def get_company_meta(self, ticker: str) -> CompanyMeta:
        """读取公司级元数据。"""
        ...

    def upsert_company_meta(self, meta: CompanyMeta) -> None:
        """写入公司级元数据。"""
        ...

    def resolve_existing_ticker(self, ticker_candidates: list[str]) -> Optional[str]:
        """在候选 ticker 中解析工作区内已存在的规范 ticker。"""
        ...


class SourceDocumentRepositoryProtocol(Protocol):
    """源文档仓储协议。"""

    def has_source_storage_root(self, ticker: str, source_kind: SourceKind) -> bool:
        """判断某类源文档根目录是否存在且为目录。"""
        ...

    def has_filing_xbrl_instance(self, ticker: str, document_id: str) -> bool:
        """判断某个 filing 目录下是否已落盘 XBRL instance 文件。"""
        ...

    def create_source_document(
        self,
        req: SourceDocumentUpsertRequest,
        source_kind: SourceKind,
    ) -> DocumentHandle:
        """创建源文档。"""
        ...

    def update_source_document(
        self,
        req: SourceDocumentUpsertRequest,
        source_kind: SourceKind,
    ) -> DocumentHandle:
        """更新源文档。"""
        ...

    def delete_source_document(self, req: SourceDocumentStateChangeRequest) -> None:
        """逻辑删除源文档。"""
        ...

    def restore_source_document(self, req: SourceDocumentStateChangeRequest) -> DocumentHandle:
        """恢复逻辑删除的源文档。"""
        ...

    def get_source_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> DocumentMeta:
        """读取源文档 meta。"""
        ...

    def replace_source_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
        meta: DocumentMeta,
    ) -> None:
        """整体替换源文档 meta。"""
        ...

    def list_source_document_ids(self, ticker: str, source_kind: SourceKind) -> list[str]:
        """按来源列出源文档 ID。"""
        ...

    def get_source_handle(self, ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
        """构造源文档句柄。"""
        ...

    def get_primary_file(self, ticker: str, document_id: str, source_kind: SourceKind) -> FileObjectMeta:
        """读取源文档主文件对象元数据。"""
        ...

    def get_source(self, ticker: str, document_id: str, source_kind: SourceKind, filename: str) -> Source:
        """读取源文档指定文件 source。"""
        ...

    def get_primary_source(self, ticker: str, document_id: str, source_kind: SourceKind) -> Source:
        """读取源文档主文件 source。"""
        ...


class ProcessedDocumentRepositoryProtocol(Protocol):
    """processed 产物仓储协议。"""

    def create_processed(self, req: ProcessedCreateRequest) -> DocumentHandle:
        """创建 processed 文档。"""
        ...

    def update_processed(self, req: ProcessedUpdateRequest) -> DocumentHandle:
        """更新 processed 文档。"""
        ...

    def delete_processed(self, req: ProcessedDeleteRequest) -> None:
        """删除 processed 文档。"""
        ...

    def get_processed_handle(self, ticker: str, document_id: str) -> ProcessedHandle:
        """构造 processed 句柄。"""
        ...

    def get_processed_meta(self, ticker: str, document_id: str) -> DocumentMeta:
        """读取 processed meta。"""
        ...

    def list_processed_documents(self, ticker: str, query: DocumentQuery) -> list[DocumentSummary]:
        """按查询条件列出 processed 文档摘要。"""
        ...

    def clear_processed_documents(self, ticker: str) -> None:
        """清空某个 ticker 的全部 processed 产物。"""
        ...

    def mark_processed_reprocess_required(self, ticker: str, document_id: str, required: bool) -> None:
        """标记 processed 文档是否需要重处理。"""
        ...


class DocumentBlobRepositoryProtocol(Protocol):
    """文档文件对象仓储协议。"""

    def list_entries(self, handle: SourceHandle | ProcessedHandle) -> list[DocumentEntry]:
        """列出文档目录直系条目。"""
        ...

    def read_file_bytes(self, handle: SourceHandle | ProcessedHandle, name: str) -> bytes:
        """读取文件字节内容。"""
        ...

    def delete_entry(self, handle: SourceHandle | ProcessedHandle, name: str) -> None:
        """删除直系条目。"""
        ...

    def store_file(
        self,
        handle: SourceHandle | ProcessedHandle,
        filename: str,
        data: BinaryIO,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> FileObjectMeta:
        """写入文件对象。"""
        ...

    def list_files(self, handle: SourceHandle | ProcessedHandle) -> list[FileObjectMeta]:
        """列出目录中的文件对象元数据。"""
        ...


class FilingMaintenanceRepositoryProtocol(Protocol):
    """filing 维护治理仓储协议。"""

    def clear_filing_documents(self, ticker: str) -> None:
        """清空某个 ticker 下的全部 filing 文档。"""
        ...

    def load_download_rejection_registry(self, ticker: str) -> dict[str, dict[str, str]]:
        """读取下载拒绝注册表。"""
        ...

    def save_download_rejection_registry(
        self,
        ticker: str,
        registry: dict[str, dict[str, str]],
    ) -> None:
        """保存下载拒绝注册表。"""
        ...

    def store_rejected_filing_file(
        self,
        ticker: str,
        document_id: str,
        filename: str,
        data: BinaryIO,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> FileObjectMeta:
        """写入 rejected filing 文件对象。"""
        ...

    def upsert_rejected_filing_artifact(
        self,
        req: RejectedFilingArtifactUpsertRequest,
    ) -> RejectedFilingArtifact:
        """写入或更新 rejected filing artifact。"""
        ...

    def get_rejected_filing_artifact(
        self,
        ticker: str,
        document_id: str,
    ) -> RejectedFilingArtifact:
        """读取 rejected filing artifact。"""
        ...

    def list_rejected_filing_artifacts(
        self,
        ticker: str,
    ) -> list[RejectedFilingArtifact]:
        """列出某个 ticker 下的 rejected filing artifacts。"""
        ...

    def read_rejected_filing_file_bytes(
        self,
        ticker: str,
        document_id: str,
        filename: str,
    ) -> bytes:
        """读取 rejected filing 文件内容。"""
        ...

    def cleanup_stale_filing_documents(
        self,
        ticker: str,
        *,
        active_form_types: set[str],
        valid_document_ids: set[str],
    ) -> int:
        """清理不在有效集合中的 filing 文档。"""
        ...


__all__ = [
    "BatchingRepositoryProtocol",
    "CompanyMetaRepositoryProtocol",
    "SourceDocumentRepositoryProtocol",
    "ProcessedDocumentRepositoryProtocol",
    "DocumentBlobRepositoryProtocol",
    "FilingMaintenanceRepositoryProtocol",
]
