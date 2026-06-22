"""文档文件对象仓储句柄越界拒绝单测。

实测剧本 E 组的可执行版本(docs/fins_storage_test.md):验证 6 类常见 path
traversal 攻击在 ``_resolve_handle_child_path`` / ``_normalize_entry_name``
两层校验下全部以 ``ValueError`` 拒绝;读、写、删三个写口同样必须拒绝。

任何越界 attack 经 normalize 后落到合法但不存在路径并抛 ``FileNotFoundError``
都属于 P0 安全 bug(校验缺失,绕过两层防护)。
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from dayu.fins.domain.document_models import SourceHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from dayu.fins.storage.fs_document_blob_repository import FsDocumentBlobRepository


_PATH_TRAVERSAL_ATTACKS: tuple[str, ...] = (
    "../etc/passwd",
    "/etc/passwd",
    "..\\windows\\system32\\drivers\\etc\\hosts",
    "",
    ".",
    "..",
)


@pytest.fixture
def blob_repository(tmp_path: Path) -> FsDocumentBlobRepository:
    """构建独立 blob 仓储,共享 core 但隔离工作区。

    Args:
        tmp_path: pytest 提供的临时目录。

    Returns:
        测试用 blob 仓储实例。

    Raises:
        OSError: 仓储初始化失败时抛出。
    """

    repository_set = build_fs_repository_set(workspace_root=tmp_path)
    return FsDocumentBlobRepository(tmp_path, repository_set=repository_set)


@pytest.fixture
def source_handle() -> SourceHandle:
    """构造合法 source 句柄(指向尚不存在的文档目录)。

    Returns:
        指向 ``NVDA / filing / fil_test_001`` 的合法句柄;真实目录不必存在,
        因为越界校验在 ``_resolve_handle_child_path`` 内早于 ``path.exists()``
        检查执行。
    """

    return SourceHandle(
        ticker="NVDA",
        document_id="fil_test_001",
        source_kind=SourceKind.FILING.value,
    )


@pytest.mark.parametrize("attack", _PATH_TRAVERSAL_ATTACKS)
def test_blob_read_file_bytes_rejects_path_traversal(
    blob_repository: FsDocumentBlobRepository,
    source_handle: SourceHandle,
    attack: str,
) -> None:
    """读路径必须以 ValueError 拒绝 6 类越界 attack。

    Args:
        blob_repository: 测试 blob 仓储。
        source_handle: 测试 source 句柄。
        attack: 越界 attack 字符串。

    Returns:
        无。

    Raises:
        AssertionError: 任一 attack 未被 ValueError 拒绝时抛出。
    """

    with pytest.raises(ValueError):
        blob_repository.read_file_bytes(source_handle, attack)


@pytest.mark.parametrize("attack", _PATH_TRAVERSAL_ATTACKS)
def test_blob_store_file_rejects_path_traversal(
    blob_repository: FsDocumentBlobRepository,
    source_handle: SourceHandle,
    attack: str,
) -> None:
    """写路径必须以 ValueError 拒绝 6 类越界 attack。

    Args:
        blob_repository: 测试 blob 仓储。
        source_handle: 测试 source 句柄。
        attack: 越界 attack 字符串。

    Returns:
        无。

    Raises:
        AssertionError: 任一 attack 未被 ValueError 拒绝时抛出。
    """

    with pytest.raises(ValueError):
        blob_repository.store_file(
            source_handle,
            attack,
            BytesIO(b"path traversal probe"),
        )


@pytest.mark.parametrize("attack", _PATH_TRAVERSAL_ATTACKS)
def test_blob_delete_entry_rejects_path_traversal(
    blob_repository: FsDocumentBlobRepository,
    source_handle: SourceHandle,
    attack: str,
) -> None:
    """删除路径必须以 ValueError 拒绝 6 类越界 attack。

    Args:
        blob_repository: 测试 blob 仓储。
        source_handle: 测试 source 句柄。
        attack: 越界 attack 字符串。

    Returns:
        无。

    Raises:
        AssertionError: 任一 attack 未被 ValueError 拒绝时抛出。
    """

    with pytest.raises(ValueError):
        blob_repository.delete_entry(source_handle, attack)


def test_blob_path_traversal_does_not_degrade_to_file_not_found(
    blob_repository: FsDocumentBlobRepository,
    source_handle: SourceHandle,
) -> None:
    """越界 attack 绝不能经 normalize 后退化为 FileNotFoundError。

    若任一 attack 在两层校验下未被 ``ValueError`` 拒绝,而是穿透到
    ``path.exists()`` 阶段抛 ``FileNotFoundError``,说明 normalize 把越界变成
    "合法但找不到",**这是 P0 安全 bug**。

    Args:
        blob_repository: 测试 blob 仓储。
        source_handle: 测试 source 句柄。

    Returns:
        无。

    Raises:
        AssertionError: 任一 attack 引发 FileNotFoundError 时抛出。
    """

    for attack in _PATH_TRAVERSAL_ATTACKS:
        try:
            blob_repository.read_file_bytes(source_handle, attack)
        except ValueError:
            continue
        except FileNotFoundError as exc:  # pragma: no cover - 失败分支
            raise AssertionError(
                f"越界 attack {attack!r} 经 normalize 后退化为 FileNotFoundError: {exc}"
            ) from exc
        else:  # pragma: no cover - 失败分支
            raise AssertionError(f"越界 attack {attack!r} 未被拒绝")
