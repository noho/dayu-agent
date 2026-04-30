"""fins tools 缓存测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from dayu.fins.tools.cache import ProcessorCacheKey, ProcessorLRUCache


@pytest.mark.unit
def test_processor_lru_cache_get_put_and_size() -> None:
    """验证缓存基础读写与条目计数。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    cache: ProcessorLRUCache[str] = ProcessorLRUCache(max_entries=2)
    key = ProcessorCacheKey(ticker="AAPL", document_id="fil_1")

    assert cache.get(key) is None
    cache.put(key, "processor-a")

    assert cache.get(key) == "processor-a"
    assert cache.size() == 1


@pytest.mark.unit
def test_processor_lru_cache_evicts_least_recently_used() -> None:
    """验证 LRU 淘汰策略。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    cache: ProcessorLRUCache[str] = ProcessorLRUCache(max_entries=2)
    key_a = ProcessorCacheKey(ticker="AAPL", document_id="fil_1")
    key_b = ProcessorCacheKey(ticker="AAPL", document_id="fil_2")
    key_c = ProcessorCacheKey(ticker="AAPL", document_id="fil_3")

    cache.put(key_a, "a")
    cache.put(key_b, "b")
    # 复杂逻辑说明：主动访问 key_a 使其成为“最近使用”，确保淘汰 key_b。
    assert cache.get(key_a) == "a"
    cache.put(key_c, "c")

    assert cache.get(key_a) == "a"
    assert cache.get(key_b) is None
    assert cache.get(key_c) == "c"


@pytest.mark.unit
def test_processor_lru_cache_thread_safe_for_read_write() -> None:
    """验证多线程读写不会破坏缓存状态。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    cache: ProcessorLRUCache[str] = ProcessorLRUCache(max_entries=8)

    def worker(index: int) -> str:
        """执行一次写入并读取。

        Args:
            index: 任务序号。

        Returns:
            缓存读取值。

        Raises:
            RuntimeError: 缓存异常时抛出。
        """

        key = ProcessorCacheKey(ticker="AAPL", document_id=f"fil_{index % 4}")
        value = f"processor-{index}"
        cache.put(key, value)
        current = cache.get(key)
        return "" if current is None else current

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(worker, range(64)))

    assert all(result.startswith("processor-") for result in results)
    assert cache.size() <= 8


@pytest.mark.unit
def test_processor_lru_cache_rejects_invalid_capacity() -> None:
    """验证非法容量会触发异常。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    with pytest.raises(ValueError):
        ProcessorLRUCache(max_entries=0)


@pytest.mark.unit
def test_processor_lru_cache_property_and_evict_clear_paths() -> None:
    """验证容量属性、evict 未命中/命中与 clear 分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    cache: ProcessorLRUCache[str] = ProcessorLRUCache(max_entries=2)
    key = ProcessorCacheKey(ticker="AAPL", document_id="fil_1")

    assert cache.max_entries == 2
    assert cache.evict(key) is False

    cache.put(key, "processor-a")
    assert cache.evict(key) is True
    assert cache.get(key) is None

    cache.put(key, "processor-b")
    assert cache.size() == 1
    cache.clear()
    assert cache.size() == 0


@pytest.mark.unit
def test_processor_lru_cache_keys_snapshot_returns_lru_order() -> None:
    """验证 keys_snapshot 返回当前 LRU 顺序的只读快照。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    cache: ProcessorLRUCache[str] = ProcessorLRUCache(max_entries=4)
    key_a = ProcessorCacheKey(ticker="AAPL", document_id="fil_1")
    key_b = ProcessorCacheKey(ticker="AAPL", document_id="fil_2")
    key_c = ProcessorCacheKey(ticker="AAPL", document_id="fil_3")

    assert cache.keys_snapshot() == ()

    cache.put(key_a, "a")
    cache.put(key_b, "b")
    cache.put(key_c, "c")
    # 复杂逻辑说明：访问 key_a 使其成为最近使用，验证快照顺序随之刷新。
    assert cache.get(key_a) == "a"

    snapshot = cache.keys_snapshot()
    assert snapshot == (key_b, key_c, key_a)
    # 快照必须是只读元组，调用方拿不到内部存储引用。
    assert isinstance(snapshot, tuple)


@pytest.mark.unit
def test_processor_lru_cache_keys_snapshot_reflects_eviction_and_clear() -> None:
    """验证淘汰与清空后 keys_snapshot 同步收敛。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    cache: ProcessorLRUCache[str] = ProcessorLRUCache(max_entries=2)
    key_a = ProcessorCacheKey(ticker="AAPL", document_id="fil_1")
    key_b = ProcessorCacheKey(ticker="AAPL", document_id="fil_2")
    key_c = ProcessorCacheKey(ticker="AAPL", document_id="fil_3")

    cache.put(key_a, "a")
    cache.put(key_b, "b")
    cache.put(key_c, "c")  # 触发 key_a 被淘汰
    assert cache.keys_snapshot() == (key_b, key_c)

    cache.clear()
    assert cache.keys_snapshot() == ()


@pytest.mark.unit
def test_processor_lru_cache_peek_does_not_change_lru_order() -> None:
    """验证 peek 命中不会改变 LRU 顺序。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    cache: ProcessorLRUCache[str] = ProcessorLRUCache(max_entries=4)
    key_a = ProcessorCacheKey(ticker="AAPL", document_id="fil_1")
    key_b = ProcessorCacheKey(ticker="AAPL", document_id="fil_2")
    key_c = ProcessorCacheKey(ticker="AAPL", document_id="fil_3")

    cache.put(key_a, "a")
    cache.put(key_b, "b")
    cache.put(key_c, "c")

    # 复杂逻辑说明：peek 命中应保持原 LRU 顺序，不把候选条目"算作一次访问"。
    assert cache.peek(key_a) == "a"
    assert cache.keys_snapshot() == (key_a, key_b, key_c)

    # 未命中时返回 None，且不影响顺序。
    assert cache.peek(ProcessorCacheKey(ticker="AAPL", document_id="fil_x")) is None
    assert cache.keys_snapshot() == (key_a, key_b, key_c)
