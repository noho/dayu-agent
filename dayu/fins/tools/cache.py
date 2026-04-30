"""财报工具缓存组件。

本模块仅提供 Processor 实例缓存能力：
- 只做进程内缓存。
- 只做 LRU 淘汰（无 TTL）。
- 线程安全，适配多线程工具并发调用。
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Generic, Optional, TypeVar


ProcessorT = TypeVar("ProcessorT")


@dataclass(frozen=True)
class ProcessorCacheKey:
    """Processor 缓存键。

    Attributes:
        ticker: 股票代码（标准化后）。
        document_id: 文档唯一标识。
    """

    ticker: str
    document_id: str


class ProcessorLRUCache(Generic[ProcessorT]):
    """线程安全的 Processor LRU 缓存。

    设计说明：
    - 本缓存只按访问顺序做 LRU 淘汰，不做时间失效。
    - 读命中会刷新到最近使用位置。
    - 容量上限触发时淘汰最久未使用条目。
    """

    def __init__(self, max_entries: int = 128) -> None:
        """初始化缓存。

        Args:
            max_entries: 最大缓存条目数，必须大于 0。

        Returns:
            无。

        Raises:
            ValueError: 当 `max_entries <= 0` 时抛出。
        """

        if max_entries <= 0:
            raise ValueError("max_entries 必须大于 0")
        self._max_entries = int(max_entries)
        self._store: OrderedDict[ProcessorCacheKey, ProcessorT] = OrderedDict()
        self._lock = RLock()

    @property
    def max_entries(self) -> int:
        """返回缓存容量上限。

        Args:
            无。

        Returns:
            缓存容量上限。

        Raises:
            无。
        """

        return self._max_entries

    def get(self, key: ProcessorCacheKey) -> Optional[ProcessorT]:
        """读取缓存并刷新 LRU 顺序。

        Args:
            key: 缓存键。

        Returns:
            命中时返回 Processor 实例；未命中返回 `None`。

        Raises:
            RuntimeError: 内部存储异常时抛出。
        """

        with self._lock:
            value = self._store.get(key)
            if value is None:
                return None
            # 复杂逻辑说明：命中后需要提升优先级，避免被误淘汰。
            self._store.move_to_end(key, last=True)
            return value

    def peek(self, key: ProcessorCacheKey) -> Optional[ProcessorT]:
        """读取缓存但不改变 LRU 顺序。

        诊断、统计、只读巡检等不应将候选条目"算作一次访问"的场景使用此方法；
        与 ``get`` 不同，``peek`` 不会把命中条目移动到 LRU 末尾，因此不会污染
        缓存的真实使用画像。

        Args:
            key: 缓存键。

        Returns:
            命中时返回 Processor 实例；未命中返回 ``None``。

        Raises:
            RuntimeError: 内部存储异常时抛出。
        """

        with self._lock:
            return self._store.get(key)

    def put(self, key: ProcessorCacheKey, value: ProcessorT) -> None:
        """写入缓存并按 LRU 规则淘汰。

        Args:
            key: 缓存键。
            value: Processor 实例。

        Returns:
            无。

        Raises:
            RuntimeError: 内部存储异常时抛出。
        """

        with self._lock:
            if key in self._store:
                self._store[key] = value
                self._store.move_to_end(key, last=True)
                return
            self._store[key] = value
            # 复杂逻辑说明：超过容量时持续淘汰最旧条目，确保容量严格受控。
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def evict(self, key: ProcessorCacheKey) -> bool:
        """移除指定缓存键。

        Args:
            key: 缓存键。

        Returns:
            当键存在并被移除时返回 `True`，否则返回 `False`。

        Raises:
            RuntimeError: 内部存储异常时抛出。
        """

        with self._lock:
            if key not in self._store:
                return False
            self._store.pop(key, None)
            return True

    def clear(self) -> None:
        """清空缓存。

        Args:
            无。

        Returns:
            无。

        Raises:
            RuntimeError: 内部存储异常时抛出。
        """

        with self._lock:
            self._store.clear()

    def size(self) -> int:
        """返回当前缓存条目数。

        Args:
            无。

        Returns:
            当前缓存条目数。

        Raises:
            RuntimeError: 内部存储异常时抛出。
        """

        with self._lock:
            return len(self._store)

    def keys_snapshot(self) -> tuple[ProcessorCacheKey, ...]:
        """返回当前缓存键的只读快照。

        快照用于上层做"已缓存条目枚举"等只读诊断；本方法不会改变 LRU 顺序，
        也不会暴露内部存储引用，调用方拿到的是不可变 tuple。

        Args:
            无。

        Returns:
            按当前 LRU 顺序（从最久未使用到最近使用）返回的 ``ProcessorCacheKey`` 元组。

        Raises:
            RuntimeError: 内部存储异常时抛出。
        """

        with self._lock:
            return tuple(self._store.keys())
