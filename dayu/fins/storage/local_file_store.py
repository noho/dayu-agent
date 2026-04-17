"""本地文件系统对象存储实现。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Optional

from dayu.fins.domain.document_models import FileObjectMeta

from .file_store import FileStore


class LocalFileStore(FileStore):
    """基于本地文件系统的对象存储实现。"""

    def __init__(self, root: Path, scheme: str = "local") -> None:
        """初始化本地对象存储。

        Args:
            root: 存储根目录。
            scheme: URI scheme（默认 local）。

        Returns:
            无。

        Raises:
            ValueError: root 为空或 scheme 非法时抛出。
            OSError: 目录创建失败时抛出。
        """

        if not scheme or not scheme.strip():
            raise ValueError("scheme 不能为空")
        self._scheme = scheme.strip()
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def put_object(
        self,
        key: str,
        data: BinaryIO,
        *,
        content_type: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> FileObjectMeta:
        """写入对象内容并返回元数据。

        Args:
            key: 对象键。
            data: 二进制流。
            content_type: 可选内容类型。
            metadata: 可选扩展元数据。

        Returns:
            文件对象元数据。

        Raises:
            ValueError: key 非法时抛出。
            OSError: 写入失败时抛出。
        """

        path = self._resolve_key(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".part")
        if temp_path.exists():
            temp_path.unlink()
        sha256 = hashlib.sha256()
        size = 0
        with temp_path.open("wb") as stream:
            while True:
                chunk = data.read(1024 * 64)
                if not chunk:
                    break
                stream.write(chunk)
                sha256.update(chunk)
                size += len(chunk)
        temp_path.replace(path)
        return FileObjectMeta(
            uri=self._build_uri(key),
            etag=sha256.hexdigest(),
            last_modified=_iso_now(),
            size=size,
            content_type=content_type,
            sha256=sha256.hexdigest(),
        )

    def get_object(self, key: str) -> BinaryIO:
        """读取对象内容。

        Args:
            key: 对象键。

        Returns:
            二进制流。

        Raises:
            FileNotFoundError: 对象不存在时抛出。
        """

        path = self._resolve_key(key)
        if not path.exists():
            raise FileNotFoundError(f"对象不存在: {path}")
        return path.open("rb")

    def stat_object(self, key: str) -> FileObjectMeta:
        """查询对象元数据。

        Args:
            key: 对象键。

        Returns:
            文件对象元数据。

        Raises:
            FileNotFoundError: 对象不存在时抛出。
        """

        path = self._resolve_key(key)
        if not path.exists():
            raise FileNotFoundError(f"对象不存在: {path}")
        sha256 = _hash_file_sha256(path)
        stat = path.stat()
        return FileObjectMeta(
            uri=self._build_uri(key),
            etag=sha256,
            last_modified=_iso_from_timestamp(stat.st_mtime),
            size=stat.st_size,
            sha256=sha256,
        )

    def delete_object(self, key: str) -> None:
        """删除对象。

        Args:
            key: 对象键。

        Returns:
            无。

        Raises:
            FileNotFoundError: 对象不存在时抛出。
        """

        path = self._resolve_key(key)
        if not path.exists():
            raise FileNotFoundError(f"对象不存在: {path}")
        path.unlink()

    def get_presigned_url(self, key: str, expires_in: int) -> str:
        """获取预签名 URL（本地实现直接返回 URI）。

        Args:
            key: 对象键。
            expires_in: 过期秒数。

        Returns:
            预签名 URL。

        Raises:
            NotImplementedError: 本地实现不支持预签名时抛出。
        """

        raise NotImplementedError("本地文件存储不支持预签名 URL")

    def list_objects(self, prefix: str) -> list[FileObjectMeta]:
        """按前缀列出对象。

        Args:
            prefix: 对象前缀。

        Returns:
            文件对象元数据列表。

        Raises:
            OSError: 读取失败时抛出。
        """

        root = self._resolve_key(prefix)
        if not root.exists():
            return []
        items: list[FileObjectMeta] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            key = path.relative_to(self._root).as_posix()
            items.append(self.stat_object(key))
        return items

    def _resolve_key(self, key: str) -> Path:
        """将对象 key 解析为本地路径。

        Args:
            key: 对象键。

        Returns:
            本地路径。

        Raises:
            ValueError: key 非法或路径越界时抛出。
        """

        normalized = key.strip().lstrip("/")
        if not normalized:
            raise ValueError("key 不能为空")
        path = (self._root / Path(*normalized.split("/"))).resolve()
        if self._root not in path.parents and path != self._root:
            raise ValueError("key 越界，禁止访问根目录外路径")
        return path

    def _build_uri(self, key: str) -> str:
        """构造对象 URI。

        Args:
            key: 对象键。

        Returns:
            对象 URI。

        Raises:
            ValueError: key 为空时抛出。
        """

        normalized = key.strip().lstrip("/")
        if not normalized:
            raise ValueError("key 不能为空")
        return f"{self._scheme}://{normalized}"


def _hash_file_sha256(path: Path) -> str:
    """计算文件 sha256。

    Args:
        path: 文件路径。

    Returns:
        sha256 字符串。

    Raises:
        OSError: 文件读取失败时抛出。
    """

    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 64)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def _iso_from_timestamp(timestamp: float) -> str:
    """将时间戳转换为 ISO8601。

    Args:
        timestamp: 时间戳。

    Returns:
        ISO8601 字符串。

    Raises:
        无。
    """

    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def _iso_now() -> str:
    """获取当前 UTC ISO8601 时间。

    Args:
        无。

    Returns:
        ISO8601 字符串。

    Raises:
        无。
    """

    return datetime.now(UTC).isoformat()
