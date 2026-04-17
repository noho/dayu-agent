"""跨平台文本文件锁辅助。

统一封装 POSIX `fcntl.flock()` 与 Windows `msvcrt.locking()`，
供需要跨进程互斥的模块复用。

模块级 `_FCNTL` / `_MSVCRT` 作为平台后端的唯一入口，允许测试在非宿主平台
通过 `monkeypatch.setattr` 注入 fake 后端以验证跨平台分支行为；生产运行时
这两个变量由 `sys.platform` narrowing 初始化，保持 pyright 类型分析的精确性。
"""

from __future__ import annotations

import errno
import os
import sys
import time
from typing import Protocol, TextIO, cast

_WINDOWS_LOCK_RETRY_INTERVAL_SEC = 0.1


class _MsvcrtLockingModule(Protocol):
    """Windows `msvcrt` 锁接口的最小协议。"""

    LK_NBLCK: int
    LK_UNLCK: int

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        """锁定或解锁给定字节区间。"""

        ...


class _FcntlLockingModule(Protocol):
    """POSIX `fcntl` 锁接口的最小协议。"""

    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, fd: int, operation: int) -> None:
        """对文件描述符执行 flock 操作。"""

        ...


if sys.platform == "win32":
    import msvcrt as _msvcrt_native

    _FCNTL: _FcntlLockingModule | None = None
    _MSVCRT: _MsvcrtLockingModule | None = cast(_MsvcrtLockingModule, _msvcrt_native)
else:
    import fcntl as _fcntl_native

    _FCNTL: _FcntlLockingModule | None = cast(_FcntlLockingModule, _fcntl_native)
    _MSVCRT: _MsvcrtLockingModule | None = None


def ensure_lock_region(stream: TextIO, *, region_bytes: int) -> None:
    """确保锁文件存在可锁定的固定字节区间。

    Args:
        stream: 已打开的锁文件流。
        region_bytes: 需要锁定的字节数。

    Returns:
        无。

    Raises:
        OSError: 当写入或同步失败时抛出。
        ValueError: `region_bytes` 非正数时抛出。
    """

    if region_bytes <= 0:
        raise ValueError("region_bytes 必须大于 0")
    stream.seek(0, os.SEEK_END)
    if stream.tell() >= region_bytes:
        stream.seek(0)
        return
    stream.write("\0" * region_bytes)
    stream.flush()
    os.fsync(stream.fileno())
    stream.seek(0)


def is_lock_contention_error(exc: OSError) -> bool:
    """判断是否为跨进程文件锁竞争错误。

    Args:
        exc: 捕获到的底层 `OSError`。

    Returns:
        若错误由锁竞争触发则返回 `True`，否则返回 `False`。

    Raises:
        无。
    """

    return exc.errno in {errno.EACCES, errno.EAGAIN} or getattr(exc, "winerror", None) == 33


def acquire_text_file_lock(
    stream: TextIO,
    *,
    blocking: bool,
    region_bytes: int = 1,
    lock_name: str,
) -> None:
    """获取文本文件流上的跨平台排他锁。

    Args:
        stream: 已打开的文本流。
        blocking: 是否阻塞等待锁。
        region_bytes: Windows 下需要锁定的字节数。
        lock_name: 锁用途描述，用于错误提示。

    Returns:
        无。

    Raises:
        OSError: 当前平台没有可用锁实现，或底层加锁失败时抛出。
        ValueError: `region_bytes` 非法时抛出。
    """

    fcntl_backend = _FCNTL
    if fcntl_backend is not None:
        lock_flags = fcntl_backend.LOCK_EX
        if not blocking:
            lock_flags |= fcntl_backend.LOCK_NB
        fcntl_backend.flock(stream.fileno(), lock_flags)
        return
    msvcrt_backend = _MSVCRT
    if msvcrt_backend is not None:
        ensure_lock_region(stream, region_bytes=region_bytes)
        while True:
            stream.seek(0)
            try:
                msvcrt_backend.locking(stream.fileno(), msvcrt_backend.LK_NBLCK, region_bytes)
                return
            except OSError as exc:
                if not blocking or not is_lock_contention_error(exc):
                    raise
                # Windows 的 LK_LOCK 最多重试 10 次，不符合 blocking=True 的语义；
                # 这里改为显式轮询 LK_NBLCK，直到真正拿到锁。
                time.sleep(_WINDOWS_LOCK_RETRY_INTERVAL_SEC)
    raise OSError(f"当前平台不支持 {lock_name}")


def release_text_file_lock(
    stream: TextIO,
    *,
    region_bytes: int = 1,
    lock_name: str,
) -> None:
    """释放文本文件流上的跨平台排他锁。

    Args:
        stream: 已打开且已持锁的文本流。
        region_bytes: Windows 下需要解锁的字节数。
        lock_name: 锁用途描述，用于错误提示。

    Returns:
        无。

    Raises:
        OSError: 当前平台没有可用锁实现，或底层解锁失败时抛出。
        ValueError: `region_bytes` 非法时抛出。
    """

    fcntl_backend = _FCNTL
    if fcntl_backend is not None:
        fcntl_backend.flock(stream.fileno(), fcntl_backend.LOCK_UN)
        return
    msvcrt_backend = _MSVCRT
    if msvcrt_backend is not None:
        stream.seek(0)
        msvcrt_backend.locking(stream.fileno(), msvcrt_backend.LK_UNLCK, region_bytes)
        return
    raise OSError(f"当前平台不支持 {lock_name}")
