"""Fins 测试用旧仓储适配器。

该模块只服务于测试代码，目的是把仍按“单总仓储”组织的 fake repository
适配为当前生产代码要求的 company/source/processed/blob 窄仓储接口，
避免在每个测试文件里重复书写同样的适配胶水。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, BinaryIO, Callable, Literal, Optional, Protocol, cast

from dayu.engine.processors.source import Source
from dayu.engine.tool_registry import ToolRegistry
from dayu.fins.domain.document_models import (
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
    ProcessedUpdateRequest,
    SourceDocumentStateChangeRequest,
    SourceDocumentUpsertRequest,
    SourceHandle,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.tool_snapshot_export import export_tool_snapshot as _export_tool_snapshot
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    DocumentBlobRepositoryProtocol,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)
from dayu.fins.tools import FinsToolLimits
from dayu.fins.tools.fins_tools import register_fins_read_tools as _register_fins_read_tools
from dayu.fins.tools.service import FinsToolService as _FinsToolService


def _forward_legacy_processed_write(
    *,
    repository: LegacyReadRepositoryProtocol,
    method_name: Literal["create_processed", "update_processed"],
    request: ProcessedCreateRequest | ProcessedUpdateRequest,
) -> DocumentHandle:
    """把 processed 写请求尽量透传给底层旧仓储。

    Args:
        repository: 旧测试仓储对象。
        method_name: 目标写方法名。
        request: processed 写请求。

    Returns:
        底层仓储返回的文档句柄。

    Raises:
        NotImplementedError: 底层旧仓储未实现目标写方法时抛出。
    """

    method = getattr(repository, method_name, None)
    if callable(method):
        return cast(DocumentHandle, method(request))
    raise NotImplementedError(f"测试适配器未实现 {method_name}")


class LegacyReadRepositoryProtocol(Protocol):
    """旧测试 fake repository 的最小读取协议。"""

    def get_company_meta(self, ticker: str) -> CompanyMeta:
        """读取公司元数据。"""

        ...

    def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
        """解析已存在 ticker。"""

        ...

    def list_document_ids(self, ticker: str, source_kind: Optional[SourceKind] = None) -> list[str]:
        """列出旧仓储中的文档 ID。"""

        ...

    def get_document_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
        """读取旧仓储中的文档元数据。"""

        ...

    def get_source_handle(self, ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
        """构造源文档句柄。"""

        ...

    def get_primary_source(self, ticker: str, document_id: str, source_kind: SourceKind) -> Source:
        """读取源文档主文件。"""

        ...

    def get_processed_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
        """读取 processed 元数据。"""

        ...


class LegacySnapshotRepositoryProtocol(LegacyReadRepositoryProtocol, Protocol):
    """支持 tool snapshot 导出的旧测试仓储协议。"""

    def get_processed_handle(self, ticker: str, document_id: str) -> ProcessedHandle:
        """构造 processed 句柄。"""

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


class LegacyProcessorRegistryProtocol(Protocol):
    """旧测试兼容层可接受的最小 processor registry 协议。"""

    def create(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> object:
        """根据 source 创建处理器实例。"""

    def create_with_fallback(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
        on_fallback: Optional[Callable[[type[object], Exception, int, int], None]] = None,
    ) -> object:
        """根据 source 创建处理器实例，并在必要时触发候选回退。"""


@dataclass(frozen=True)
class LegacyRepositoryAdapters:
    """旧测试仓储拆分后的窄仓储集合。"""

    company_repository: CompanyMetaRepositoryProtocol
    source_repository: SourceDocumentRepositoryProtocol
    processed_repository: ProcessedDocumentRepositoryProtocol


@dataclass(frozen=True)
class LegacySnapshotRepositoryAdapters(LegacyRepositoryAdapters):
    """支持 tool snapshot 的旧测试仓储拆分集合。"""

    blob_repository: DocumentBlobRepositoryProtocol


class _LegacyCompanyRepositoryAdapter:
    """公司仓储测试适配器。"""

    def __init__(self, repository: LegacyReadRepositoryProtocol) -> None:
        """保存底层旧仓储引用。"""

        self._repository = repository

    def get_company_meta(self, ticker: str) -> CompanyMeta:
        """转发公司元数据读取。"""

        return self._repository.get_company_meta(ticker)

    def scan_company_meta_inventory(self) -> list[CompanyMetaInventoryEntry]:
        """旧测试适配器不提供公司元数据盘点。"""

        return []

    def upsert_company_meta(self, meta: CompanyMeta) -> None:
        """测试适配器不负责写路径。"""

        raise NotImplementedError("测试适配器未实现 upsert_company_meta")

    def resolve_existing_ticker(self, ticker_candidates: list[str]) -> Optional[str]:
        """转发 ticker 解析。"""

        return self._repository.resolve_existing_ticker(ticker_candidates)


class _LegacySourceRepositoryAdapter:
    """源文档仓储测试适配器。"""

    def __init__(self, repository: LegacyReadRepositoryProtocol) -> None:
        """保存底层旧仓储引用。"""

        self._repository = repository

    def create_source_document(
        self,
        req: SourceDocumentUpsertRequest,
        source_kind: SourceKind,
    ) -> DocumentHandle:
        """测试适配器不负责写路径。"""

        raise NotImplementedError("测试适配器未实现 create_source_document")

    def update_source_document(
        self,
        req: SourceDocumentUpsertRequest,
        source_kind: SourceKind,
    ) -> DocumentHandle:
        """测试适配器不负责写路径。"""

        raise NotImplementedError("测试适配器未实现 update_source_document")

    def delete_source_document(self, req: SourceDocumentStateChangeRequest) -> None:
        """测试适配器不负责写路径。"""

        raise NotImplementedError("测试适配器未实现 delete_source_document")

    def restore_source_document(self, req: SourceDocumentStateChangeRequest) -> DocumentHandle:
        """测试适配器不负责写路径。"""

        raise NotImplementedError("测试适配器未实现 restore_source_document")

    def get_source_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> DocumentMeta:
        """转发源文档元数据读取。"""

        del source_kind
        return cast(DocumentMeta, self._repository.get_document_meta(ticker, document_id))

    def replace_source_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
        meta: DocumentMeta,
    ) -> None:
        """测试适配器不负责写路径。"""

        raise NotImplementedError("测试适配器未实现 replace_source_meta")

    def list_source_document_ids(self, ticker: str, source_kind: SourceKind) -> list[str]:
        """转发源文档 ID 列举。"""

        return self._repository.list_document_ids(ticker, source_kind)

    def has_source_storage_root(self, ticker: str, source_kind: SourceKind) -> bool:
        """按旧仓储可见文档集合近似回答根目录是否存在。"""

        return bool(self._repository.list_document_ids(ticker, source_kind))

    def has_filing_xbrl_instance(self, ticker: str, document_id: str) -> bool:
        """旧测试适配器不提供 XBRL instance 存在性判断。"""

        raise NotImplementedError("测试适配器未实现 has_filing_xbrl_instance")

    def get_source_handle(self, ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
        """转发源文档句柄构造。"""

        return self._repository.get_source_handle(ticker, document_id, source_kind)

    def get_primary_file(self, ticker: str, document_id: str, source_kind: SourceKind) -> FileObjectMeta:
        """测试适配器不提供文件对象元数据读取。"""

        raise NotImplementedError("测试适配器未实现 get_primary_file")

    def get_source(self, ticker: str, document_id: str, source_kind: SourceKind, filename: str) -> Source:
        """旧 fake repository 只有 primary source，测试中等价复用。"""

        del filename
        return self._repository.get_primary_source(ticker, document_id, source_kind)

    def get_primary_source(self, ticker: str, document_id: str, source_kind: SourceKind) -> Source:
        """转发主文件读取。"""

        return self._repository.get_primary_source(ticker, document_id, source_kind)


class _LegacyProcessedRepositoryAdapter:
    """processed 仓储测试适配器。"""

    def __init__(self, repository: LegacyReadRepositoryProtocol) -> None:
        """保存底层旧仓储引用。"""

        self._repository = repository

    def create_processed(self, req: ProcessedCreateRequest) -> DocumentHandle:
        """尽量透传 processed 创建请求。"""

        return _forward_legacy_processed_write(
            repository=self._repository,
            method_name="create_processed",
            request=req,
        )

    def update_processed(self, req: ProcessedUpdateRequest) -> DocumentHandle:
        """尽量透传 processed 更新请求。"""

        return _forward_legacy_processed_write(
            repository=self._repository,
            method_name="update_processed",
            request=req,
        )

    def delete_processed(self, req: ProcessedDeleteRequest) -> None:
        """测试适配器不负责写路径。"""

        raise NotImplementedError("测试适配器未实现 delete_processed")

    def get_processed_handle(self, ticker: str, document_id: str) -> ProcessedHandle:
        """尽量转发旧仓储的 processed 句柄；否则回退到最小句柄。"""

        getter = getattr(self._repository, "get_processed_handle", None)
        if callable(getter):
            return cast(ProcessedHandle, getter(ticker, document_id))
        return ProcessedHandle(ticker=ticker, document_id=document_id)

    def get_processed_meta(self, ticker: str, document_id: str) -> DocumentMeta:
        """转发 processed 元数据读取。"""

        return cast(DocumentMeta, self._repository.get_processed_meta(ticker, document_id))

    def list_processed_documents(self, ticker: str, query: DocumentQuery) -> list[DocumentSummary]:
        """测试适配器不提供列表查询。"""

        raise NotImplementedError("测试适配器未实现 list_processed_documents")

    def clear_processed_documents(self, ticker: str) -> None:
        """测试适配器不负责写路径。"""

        raise NotImplementedError("测试适配器未实现 clear_processed_documents")

    def mark_processed_reprocess_required(self, ticker: str, document_id: str, required: bool) -> None:
        """测试适配器不负责写路径。"""

        raise NotImplementedError("测试适配器未实现 mark_processed_reprocess_required")


class _LegacyBlobRepositoryAdapter:
    """blob 仓储测试适配器。"""

    def __init__(self, repository: LegacySnapshotRepositoryProtocol) -> None:
        """保存底层旧仓储引用。"""

        self._repository = repository

    def list_entries(self, handle: SourceHandle | ProcessedHandle) -> list[DocumentEntry]:
        """测试适配器不提供目录枚举。"""

        raise NotImplementedError("测试适配器未实现 list_entries")

    def read_file_bytes(self, handle: SourceHandle | ProcessedHandle, name: str) -> bytes:
        """测试适配器不提供文件字节读取。"""

        raise NotImplementedError("测试适配器未实现 read_file_bytes")

    def delete_entry(self, handle: SourceHandle | ProcessedHandle, name: str) -> None:
        """测试适配器不负责删除文件。"""

        raise NotImplementedError("测试适配器未实现 delete_entry")

    def store_file(
        self,
        handle: SourceHandle | ProcessedHandle,
        filename: str,
        data: BinaryIO,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> FileObjectMeta:
        """转发快照文件写入。"""

        return self._repository.store_file(
            handle,
            filename,
            data,
            content_type=content_type,
            metadata=metadata,
        )

    def list_files(self, handle: SourceHandle | ProcessedHandle) -> list[FileObjectMeta]:
        """测试适配器不提供文件列表。"""

        raise NotImplementedError("测试适配器未实现 list_files")


def build_legacy_repository_adapters(repository: LegacyReadRepositoryProtocol) -> LegacyRepositoryAdapters:
    """把旧测试仓储拆分为读取链路所需的窄仓储集合。"""

    return LegacyRepositoryAdapters(
        company_repository=_LegacyCompanyRepositoryAdapter(repository),
        source_repository=_LegacySourceRepositoryAdapter(repository),
        processed_repository=_LegacyProcessedRepositoryAdapter(repository),
    )


def build_legacy_snapshot_repository_adapters(
    repository: LegacySnapshotRepositoryProtocol,
) -> LegacySnapshotRepositoryAdapters:
    """把旧测试仓储拆分为 tool snapshot 所需的窄仓储集合。"""

    base = build_legacy_repository_adapters(repository)
    return LegacySnapshotRepositoryAdapters(
        company_repository=base.company_repository,
        source_repository=base.source_repository,
        processed_repository=base.processed_repository,
        blob_repository=_LegacyBlobRepositoryAdapter(repository),
    )


class LegacyCompatibleFinsToolService(_FinsToolService):
    """测试专用兼容服务。

    允许旧测试继续传入 `repository=`，内部自动拆分为窄仓储。
    """

    def __init__(
        self,
        *,
        repository: LegacyReadRepositoryProtocol | None = None,
        company_repository: CompanyMetaRepositoryProtocol | None = None,
        source_repository: SourceDocumentRepositoryProtocol | None = None,
        processed_repository: ProcessedDocumentRepositoryProtocol | None = None,
        processor_registry: LegacyProcessorRegistryProtocol,
        processor_cache_max_entries: int = 128,
    ) -> None:
        """初始化测试兼容服务。"""

        if repository is not None:
            if any(item is not None for item in (company_repository, source_repository, processed_repository)):
                raise ValueError("repository 与拆分仓储参数不能同时传入")
            adapters = build_legacy_repository_adapters(repository)
            company_repository = adapters.company_repository
            source_repository = adapters.source_repository
            processed_repository = adapters.processed_repository
        if company_repository is None or source_repository is None or processed_repository is None:
            raise ValueError("必须提供 repository 或完整的拆分仓储参数")
        super().__init__(
            company_repository=company_repository,
            source_repository=source_repository,
            processed_repository=processed_repository,
            processor_registry=cast(Any, processor_registry),
            processor_cache_max_entries=processor_cache_max_entries,
        )


def register_fins_read_tools_with_legacy_repository(
    registry: ToolRegistry,
    *,
    repository: LegacyReadRepositoryProtocol | None = None,
    service: _FinsToolService | None = None,
    processor_registry: LegacyProcessorRegistryProtocol | None = None,
    limits: FinsToolLimits | None = None,
    timeout_budget: float | None = None,
) -> None:
    """测试专用读取工具注册入口。

    兼容旧测试使用 `repository=` 直接完成最小装配。
    """

    resolved_service = service
    if resolved_service is None:
        if repository is None or processor_registry is None:
            raise ValueError("未提供 service 时，必须同时提供 repository 和 processor_registry")
        resolved_service = LegacyCompatibleFinsToolService(
            repository=repository,
            processor_registry=processor_registry,
        )
    _register_fins_read_tools(
        registry,
        service=resolved_service,
        limits=limits,
        timeout_budget=timeout_budget,
    )


def export_tool_snapshot_with_legacy_repository(
    *,
    repository: LegacySnapshotRepositoryProtocol | None = None,
    company_repository: CompanyMetaRepositoryProtocol | None = None,
    source_repository: SourceDocumentRepositoryProtocol | None = None,
    processed_repository: ProcessedDocumentRepositoryProtocol | None = None,
    blob_repository: DocumentBlobRepositoryProtocol | None = None,
    processor_registry: LegacyProcessorRegistryProtocol,
    processed_handle: ProcessedHandle,
    ticker: str,
    document_id: str,
    source_kind: SourceKind,
    source_meta: dict[str, Any],
    ci: bool,
    expected_parser_signature: str,
    market_override: str | None = None,
    processor_cache_max_entries: int = 128,
    cancel_checker: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """测试专用 tool snapshot 导出入口。

    允许旧测试继续传入 `repository=`，内部自动拆分为窄仓储。
    """

    if repository is not None:
        if any(item is not None for item in (company_repository, source_repository, processed_repository, blob_repository)):
            raise ValueError("repository 与拆分仓储参数不能同时传入")
        adapters = build_legacy_snapshot_repository_adapters(repository)
        company_repository = adapters.company_repository
        source_repository = adapters.source_repository
        processed_repository = adapters.processed_repository
        blob_repository = adapters.blob_repository
    if company_repository is None or source_repository is None or processed_repository is None or blob_repository is None:
        raise ValueError("必须提供 repository 或完整的拆分仓储参数")
    return _export_tool_snapshot(
        company_repository=company_repository,
        source_repository=source_repository,
        processed_repository=processed_repository,
        blob_repository=blob_repository,
        processor_registry=cast(Any, processor_registry),
        processed_handle=processed_handle,
        ticker=ticker,
        document_id=document_id,
        source_kind=source_kind,
        source_meta=source_meta,
        ci=ci,
        expected_parser_signature=expected_parser_signature,
        market_override=market_override,
        processor_cache_max_entries=processor_cache_max_entries,
        cancel_checker=cancel_checker,
    )


__all__ = [
    "LegacyCompatibleFinsToolService",
    "LegacyRepositoryAdapters",
    "LegacySnapshotRepositoryAdapters",
    "build_legacy_repository_adapters",
    "build_legacy_snapshot_repository_adapters",
    "export_tool_snapshot_with_legacy_repository",
    "register_fins_read_tools_with_legacy_repository",
]
