"""CN/HK 下载链路的中断恢复探测。

本模块只通过 ``DocumentBlobRepositoryProtocol`` 读取已落盘文件状态，不写入
任何仓储。workflow 根据返回结果决定是否复用本地 PDF / Docling JSON，或重置
单个 source document 后重新下载。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from dayu.fins.domain.document_models import SourceHandle
from dayu.fins.storage import DocumentBlobRepositoryProtocol


@dataclass(frozen=True)
class CnStagedBlobState:
    """单 filing 已落盘 blob 状态。

    Attributes:
        has_pdf: PDF 文件是否存在。
        has_docling_json: Docling JSON 文件是否存在。
        pdf_sha256_matched: PDF 文件存在且内容 SHA-256 与期望值一致。
        pdf_bytes: 可复用的 PDF 字节；未命中时为 ``None``。
        docling_json_bytes: 可复用的 Docling JSON 字节；未命中时为 ``None``。
    """

    has_pdf: bool
    has_docling_json: bool
    pdf_sha256_matched: bool
    pdf_bytes: bytes | None
    docling_json_bytes: bytes | None


def inspect_staged_blobs(
    *,
    blob_repository: DocumentBlobRepositoryProtocol,
    handle: SourceHandle,
    pdf_filename: str,
    docling_filename: str,
    expected_pdf_sha256: str | None,
) -> CnStagedBlobState:
    """探测 source document 下已落盘的 PDF / Docling JSON。

    Args:
        blob_repository: 文件对象仓储。
        handle: source document 句柄。
        pdf_filename: PDF 文件名。
        docling_filename: Docling JSON 文件名。
        expected_pdf_sha256: 期望 PDF SHA-256；为空时只判断文件存在，不判定命中。

    Returns:
        ``CnStagedBlobState``。

    Raises:
        OSError: 底层文件读取失败时抛出。
    """

    names = {item.uri.rsplit("/", 1)[-1] for item in blob_repository.list_files(handle)}
    has_pdf = pdf_filename in names
    has_docling = docling_filename in names
    pdf_bytes: bytes | None = None
    docling_bytes: bytes | None = None
    pdf_matched = False
    if has_pdf:
        pdf_bytes = blob_repository.read_file_bytes(handle, pdf_filename)
        pdf_sha = _sha256_hex(pdf_bytes)
        pdf_matched = expected_pdf_sha256 is not None and pdf_sha == expected_pdf_sha256
    if has_docling:
        docling_bytes = blob_repository.read_file_bytes(handle, docling_filename)
    return CnStagedBlobState(
        has_pdf=has_pdf,
        has_docling_json=has_docling,
        pdf_sha256_matched=pdf_matched,
        pdf_bytes=pdf_bytes if pdf_matched else None,
        docling_json_bytes=docling_bytes if has_docling else None,
    )


def has_blob_file(
    *,
    blob_repository: DocumentBlobRepositoryProtocol,
    handle: SourceHandle,
    filename: str,
) -> bool:
    """判断 source document 下是否存在指定文件。

    Args:
        blob_repository: 文件对象仓储。
        handle: source document 句柄。
        filename: 文件名。

    Returns:
        存在返回 ``True``，否则返回 ``False``。

    Raises:
        OSError: 底层文件枚举失败时抛出。
    """

    return any(item.uri.rsplit("/", 1)[-1] == filename for item in blob_repository.list_files(handle))


def _sha256_hex(payload: bytes) -> str:
    """计算字节内容 SHA-256。

    Args:
        payload: 原始字节。

    Returns:
        小写 hex SHA-256。

    Raises:
        无。
    """

    return hashlib.sha256(payload).hexdigest()


__all__ = ["CnStagedBlobState", "has_blob_file", "inspect_staged_blobs"]
