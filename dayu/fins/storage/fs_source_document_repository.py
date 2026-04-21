"""文件系统源文档仓储实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dayu.engine.processors.source import Source
from dayu.fins.domain.document_models import (
    DocumentHandle,
    DocumentMeta,
    FilingCreateRequest,
    FilingDeleteRequest,
    FilingRestoreRequest,
    FilingUpdateRequest,
    FileObjectMeta,
    MaterialCreateRequest,
    MaterialDeleteRequest,
    MaterialRestoreRequest,
    MaterialUpdateRequest,
    SourceDocumentStateChangeRequest,
    SourceDocumentUpsertRequest,
    SourceHandle,
)
from dayu.fins.domain.enums import SourceKind

from ._fs_repository_factory import _FsRepositorySet, build_fs_repository_set
from .file_store import FileStore
from .repository_protocols import SourceDocumentRepositoryProtocol


def _build_source_handle(ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
    """构造源文档句柄。

    Args:
        ticker: 股票代码。
        document_id: 文档 ID。
        source_kind: 源文档类型。

    Returns:
        源文档句柄。

    Raises:
        无。
    """

    return SourceHandle(
        ticker=ticker,
        document_id=document_id,
        source_kind=source_kind.value,
    )


def _build_filing_create_request(req: SourceDocumentUpsertRequest) -> FilingCreateRequest:
    """将通用写入请求收敛为 filing 创建请求。

    Args:
        req: 通用源文档写入请求。

    Returns:
        filing 创建请求。

    Raises:
        无。
    """

    return FilingCreateRequest(
        ticker=req.ticker,
        document_id=req.document_id,
        internal_document_id=req.internal_document_id,
        form_type=req.form_type,
        primary_document=req.primary_document,
        meta=req.meta,
        files=req.files,
        file_entries=req.file_entries,
    )


def _build_material_create_request(req: SourceDocumentUpsertRequest) -> MaterialCreateRequest:
    """将通用写入请求收敛为 material 创建请求。

    Args:
        req: 通用源文档写入请求。

    Returns:
        material 创建请求。

    Raises:
        无。
    """

    return MaterialCreateRequest(
        ticker=req.ticker,
        document_id=req.document_id,
        internal_document_id=req.internal_document_id,
        form_type=req.form_type,
        primary_document=req.primary_document,
        meta=req.meta,
        files=req.files,
        file_entries=req.file_entries,
    )


def _build_filing_update_request(req: SourceDocumentUpsertRequest) -> FilingUpdateRequest:
    """将通用写入请求收敛为 filing 更新请求。

    Args:
        req: 通用源文档写入请求。

    Returns:
        filing 更新请求。

    Raises:
        无。
    """

    return FilingUpdateRequest(
        ticker=req.ticker,
        document_id=req.document_id,
        internal_document_id=req.internal_document_id,
        form_type=req.form_type,
        primary_document=req.primary_document,
        meta=req.meta,
        files=req.files,
        file_entries=req.file_entries,
    )


def _build_material_update_request(req: SourceDocumentUpsertRequest) -> MaterialUpdateRequest:
    """将通用写入请求收敛为 material 更新请求。

    Args:
        req: 通用源文档写入请求。

    Returns:
        material 更新请求。

    Raises:
        无。
    """

    return MaterialUpdateRequest(
        ticker=req.ticker,
        document_id=req.document_id,
        internal_document_id=req.internal_document_id,
        form_type=req.form_type,
        primary_document=req.primary_document,
        meta=req.meta,
        files=req.files,
        file_entries=req.file_entries,
    )


def _build_filing_delete_request(req: SourceDocumentStateChangeRequest) -> FilingDeleteRequest:
    """将通用状态变更请求收敛为 filing 删除请求。

    Args:
        req: 通用源文档状态变更请求。

    Returns:
        filing 删除请求。

    Raises:
        无。
    """

    return FilingDeleteRequest(ticker=req.ticker, document_id=req.document_id)


def _build_material_delete_request(req: SourceDocumentStateChangeRequest) -> MaterialDeleteRequest:
    """将通用状态变更请求收敛为 material 删除请求。

    Args:
        req: 通用源文档状态变更请求。

    Returns:
        material 删除请求。

    Raises:
        无。
    """

    return MaterialDeleteRequest(ticker=req.ticker, document_id=req.document_id)


def _build_filing_restore_request(req: SourceDocumentStateChangeRequest) -> FilingRestoreRequest:
    """将通用状态变更请求收敛为 filing 恢复请求。

    Args:
        req: 通用源文档状态变更请求。

    Returns:
        filing 恢复请求。

    Raises:
        无。
    """

    return FilingRestoreRequest(ticker=req.ticker, document_id=req.document_id)


def _build_material_restore_request(req: SourceDocumentStateChangeRequest) -> MaterialRestoreRequest:
    """将通用状态变更请求收敛为 material 恢复请求。

    Args:
        req: 通用源文档状态变更请求。

    Returns:
        material 恢复请求。

    Raises:
        无。
    """

    return MaterialRestoreRequest(ticker=req.ticker, document_id=req.document_id)


def _infer_filename_from_uri(uri: str) -> str:
    """从文件 URI 推断文件名。

    Args:
        uri: 文件 URI。

    Returns:
        文件名；无法解析时返回空字符串。

    Raises:
        无。
    """

    raw_uri = str(uri).strip()
    if not raw_uri:
        return ""
    if "://" in raw_uri:
        raw_uri = raw_uri.split("://", 1)[1]
    raw_uri = raw_uri.rstrip("/")
    if not raw_uri:
        return ""
    return Path(raw_uri).name or raw_uri.split("/")[-1]


def _find_file_meta_by_filename(file_metas: list[FileObjectMeta], filename: str) -> FileObjectMeta:
    """按文件名定位文件元数据。

    Args:
        file_metas: 可选文件元数据列表。
        filename: 目标文件名。

    Returns:
        命中的文件元数据。

    Raises:
        FileNotFoundError: 未找到目标文件时抛出。
    """

    normalized_filename = filename.strip()
    if not normalized_filename:
        raise FileNotFoundError("filename 不能为空")
    for file_meta in file_metas:
        if _infer_filename_from_uri(file_meta.uri) == normalized_filename:
            return file_meta
    raise FileNotFoundError(f"未找到文件: {normalized_filename}")


class FsSourceDocumentRepository(SourceDocumentRepositoryProtocol):
    """基于文件系统的源文档仓储实现。"""

    def __init__(
        self,
        workspace_root: Path,
        *,
        file_store: Optional[FileStore] = None,
        repository_set: Optional[_FsRepositorySet] = None,
        create_directories: bool = True,
    ) -> None:
        """初始化源文档仓储。

        Args:
            workspace_root: 工作区根目录。
            file_store: 可选文件存储实现。
            repository_set: 可选共享仓储 core 集合。
            create_directories: 是否在初始化时创建仓储根目录。

        Returns:
            无。

        Raises:
            OSError: 底层仓储初始化失败时抛出。
        """

        self._repository_set = build_fs_repository_set(
            workspace_root=workspace_root,
            file_store=file_store,
            repository_set=repository_set,
            create_directories=create_directories,
        )

    def has_source_storage_root(self, ticker: str, source_kind: SourceKind) -> bool:
        """判断某类源文档根目录是否存在且为目录。"""

        return self._repository_set.core.has_source_storage_root(ticker, source_kind)

    def has_filing_xbrl_instance(self, ticker: str, document_id: str) -> bool:
        """判断某个 filing 目录下是否已落盘 XBRL instance 文件。"""

        return self._repository_set.core.has_filing_xbrl_instance(ticker, document_id)

    def create_source_document(
        self,
        req: SourceDocumentUpsertRequest,
        source_kind: SourceKind,
    ) -> DocumentHandle:
        """创建源文档。"""

        if source_kind == SourceKind.FILING:
            return self._repository_set.core.create_filing(_build_filing_create_request(req))
        return self._repository_set.core.create_material(_build_material_create_request(req))

    def update_source_document(
        self,
        req: SourceDocumentUpsertRequest,
        source_kind: SourceKind,
    ) -> DocumentHandle:
        """更新源文档。"""

        if source_kind == SourceKind.FILING:
            return self._repository_set.core.update_filing(_build_filing_update_request(req))
        return self._repository_set.core.update_material(_build_material_update_request(req))

    def delete_source_document(self, req: SourceDocumentStateChangeRequest) -> None:
        """逻辑删除源文档。"""

        source_kind = SourceKind(str(req.source_kind))
        if source_kind == SourceKind.FILING:
            self._repository_set.core.delete_filing(_build_filing_delete_request(req))
            return
        self._repository_set.core.delete_material(_build_material_delete_request(req))

    def reset_source_document(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> None:
        """重置单个源文档的完整存储。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 来源类型。

        Returns:
            无。

        Raises:
            OSError: 重置底层存储失败时抛出。
        """

        self._repository_set.core.reset_source_document(ticker, document_id, source_kind)

    def restore_source_document(self, req: SourceDocumentStateChangeRequest) -> DocumentHandle:
        """恢复逻辑删除的源文档。"""

        source_kind = SourceKind(str(req.source_kind))
        if source_kind == SourceKind.FILING:
            return self._repository_set.core.restore_filing(_build_filing_restore_request(req))
        return self._repository_set.core.restore_material(_build_material_restore_request(req))

    def get_source_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
    ) -> DocumentMeta:
        """读取源文档 meta。"""

        return self._repository_set.core.get_source_meta(ticker, document_id, source_kind)

    def replace_source_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
        meta: DocumentMeta,
    ) -> None:
        """整体替换源文档 meta。"""

        self._repository_set.core.replace_source_meta(ticker, document_id, source_kind, meta)

    def list_source_document_ids(self, ticker: str, source_kind: SourceKind) -> list[str]:
        """按来源列出源文档 ID。"""

        return self._repository_set.core.list_document_ids(ticker, source_kind)

    def get_source_handle(self, ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
        """构造源文档句柄。"""

        return self._repository_set.core.get_source_handle(ticker, document_id, source_kind)

    def get_primary_file(self, ticker: str, document_id: str, source_kind: SourceKind) -> FileObjectMeta:
        """读取源文档主文件对象元数据。"""

        handle = _build_source_handle(ticker, document_id, source_kind)
        return self._repository_set.core.get_primary_file(handle)

    def get_source(self, ticker: str, document_id: str, source_kind: SourceKind, filename: str) -> Source:
        """读取源文档指定文件 source。"""

        handle = _build_source_handle(ticker, document_id, source_kind)
        file_metas = self._repository_set.core.list_files(handle)
        file_meta = _find_file_meta_by_filename(file_metas, filename)
        return self._repository_set.core.get_source(handle, file_meta)

    def get_primary_source(self, ticker: str, document_id: str, source_kind: SourceKind) -> Source:
        """读取源文档主文件 source。"""

        return self._repository_set.core.get_primary_source(ticker, document_id, source_kind)
