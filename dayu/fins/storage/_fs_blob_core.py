"""文件系统仓储 — Blob / 文件条目操作 mixin。"""

from __future__ import annotations

import shutil
from typing import BinaryIO, Optional

from dayu.fins.domain.document_models import (
    DocumentEntry,
    FileObjectMeta,
    ProcessedHandle,
    SourceHandle,
)

from ._fs_storage_infra import _FsStorageInfra
from ._fs_storage_utils import (
    _file_object_meta_from_dict,
    _normalize_entry_name,
    _normalize_ticker,
)


class _FsBlobMixin(_FsStorageInfra):
    """Blob / 文件条目操作 mixin。"""

    def list_entries(self, handle: SourceHandle | ProcessedHandle) -> list[DocumentEntry]:
        """列出文档目录下的直系条目。

        Args:
            handle: 源文档/解析产物句柄。

        Returns:
            直系条目列表；目录不存在时返回空列表。

        Raises:
            OSError: 读取目录失败时抛出。
        """

        directory = self._handle_dir_path(handle)
        if not directory.exists() or not directory.is_dir():
            return []
        return [
            DocumentEntry(name=child.name, is_file=child.is_file())
            for child in sorted(directory.iterdir(), key=lambda item: item.name)
        ]

    def read_file_bytes(self, handle: SourceHandle | ProcessedHandle, filename: str) -> bytes:
        """读取文档目录下的单个文件内容。

        Args:
            handle: 源文档/解析产物句柄。
            filename: 直系文件名。

        Returns:
            文件二进制内容。

        Raises:
            FileNotFoundError: 文件不存在时抛出。
            IsADirectoryError: 目标为目录时抛出。
            OSError: 读取失败时抛出。
        """

        path = self._resolve_handle_child_path(handle, filename)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        if path.is_dir():
            raise IsADirectoryError(f"目标是目录，无法按文件读取: {path}")
        return path.read_bytes()

    def delete_entry(self, handle: SourceHandle | ProcessedHandle, name: str) -> None:
        """删除文档目录下的单个直系条目。

        Args:
            handle: 源文档/解析产物句柄。
            name: 直系条目名称。

        Returns:
            无。

        Raises:
            FileNotFoundError: 条目不存在时抛出。
            OSError: 删除失败时抛出。
        """

        self._execute_with_auto_batch(
            handle.ticker,
            self._delete_entry_impl,
            handle,
            name,
        )

    def _delete_entry_impl(self, handle: SourceHandle | ProcessedHandle, name: str) -> None:
        """执行单个直系条目删除（内部实现）。

        Args:
            handle: 源文档/解析产物句柄。
            name: 直系条目名称。

        Returns:
            无。

        Raises:
            FileNotFoundError: 条目不存在时抛出。
            OSError: 删除失败时抛出。
        """

        path = self._resolve_handle_child_path(handle, name)
        if not path.exists():
            raise FileNotFoundError(f"条目不存在: {path}")
        if path.is_dir():
            shutil.rmtree(path)
            return
        path.unlink()

    def store_file(
        self,
        handle: SourceHandle | ProcessedHandle,
        filename: str,
        data: BinaryIO,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> FileObjectMeta:
        """存储文件并返回文件元数据。

        Args:
            handle: 源文档/解析产物句柄。
            filename: 文件名。
            data: 文件二进制流。
            content_type: 可选内容类型。
            metadata: 可选扩展元数据。

        Returns:
            文件对象元数据。

        Raises:
            FileNotFoundError: 句柄对应文档不存在时抛出。
            ValueError: filename 为空、为 ``.`` / ``..``、或包含路径分隔符时抛出。
            OSError: 写入失败时抛出。
        """

        normalized_filename = _normalize_entry_name(filename)
        normalized_ticker = _normalize_ticker(handle.ticker)
        key = self._build_store_key(handle, normalized_filename)
        file_store = self._build_file_store(normalized_ticker)
        return file_store.put_object(
            key,
            data,
            content_type=content_type,
            metadata=metadata,
        )

    def list_files(self, handle: SourceHandle | ProcessedHandle) -> list[FileObjectMeta]:
        """列出文档关联的文件元数据列表。

        Args:
            handle: 源文档/解析产物句柄。

        Returns:
            文件元数据列表。

        Raises:
            FileNotFoundError: 文档不存在时抛出。
            ValueError: 元数据格式非法时抛出。
        """

        meta = self._get_handle_meta(handle)
        files = meta.get("files", [])
        if not isinstance(files, list):
            raise ValueError("meta.files 必须为 list")
        result: list[FileObjectMeta] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            result.append(_file_object_meta_from_dict(item))
        return result
