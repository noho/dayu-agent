"""CancellationToken 取消原语测试。"""

from __future__ import annotations

import threading
import time

import pytest

from dayu.contracts.cancellation import CancellationToken, CancelledError


class TestBasicBehavior:
    """基本取消行为测试。"""

    @pytest.mark.unit
    def test_initial_state_not_cancelled(self) -> None:
        """新建 token 未取消。"""
        token = CancellationToken()
        assert not token.is_cancelled()

    @pytest.mark.unit
    def test_cancel_sets_state(self) -> None:
        """cancel() 后 is_cancelled 返回 True。"""
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled()

    @pytest.mark.unit
    def test_cancel_is_idempotent(self) -> None:
        """多次 cancel() 不报错。"""
        token = CancellationToken()
        token.cancel()
        token.cancel()
        assert token.is_cancelled()

    @pytest.mark.unit
    def test_raise_if_cancelled_before_cancel(self) -> None:
        """未取消时 raise_if_cancelled 不抛异常。"""
        token = CancellationToken()
        token.raise_if_cancelled()  # 不应抛出

    @pytest.mark.unit
    def test_raise_if_cancelled_after_cancel(self) -> None:
        """取消后 raise_if_cancelled 抛出 CancelledError。"""
        token = CancellationToken()
        token.cancel()
        with pytest.raises(CancelledError):
            token.raise_if_cancelled()

    @pytest.mark.unit
    def test_wait_returns_true_when_cancelled(self) -> None:
        """已取消时 wait() 立即返回 True。"""
        token = CancellationToken()
        token.cancel()
        assert token.wait(timeout=0.01) is True

    @pytest.mark.unit
    def test_wait_returns_false_on_timeout(self) -> None:
        """未取消时 wait() 超时返回 False。"""
        token = CancellationToken()
        assert token.wait(timeout=0.01) is False


class TestCallbacks:
    """取消回调测试。"""

    @pytest.mark.unit
    def test_on_cancel_fires_on_cancel(self) -> None:
        """注册的回调在 cancel() 时触发。"""
        token = CancellationToken()
        called = []
        token.on_cancel(lambda: called.append(True))

        token.cancel()
        assert called == [True]

    @pytest.mark.unit
    def test_on_cancel_fires_immediately_if_already_cancelled(self) -> None:
        """在已取消的 token 上注册回调会立即触发。"""
        token = CancellationToken()
        token.cancel()

        called = []
        token.on_cancel(lambda: called.append(True))
        assert called == [True]

    @pytest.mark.unit
    def test_on_cancel_unregister_removes_callback_before_cancel(self) -> None:
        """注销函数应能在取消前移除已注册回调。"""

        token = CancellationToken()
        called: list[bool] = []

        unregister = token.on_cancel(lambda: called.append(True))
        unregister()

        token.cancel()
        assert called == []
        assert token._callbacks == []

    @pytest.mark.unit
    def test_multiple_callbacks_all_fire(self) -> None:
        """多个回调全部触发。"""
        token = CancellationToken()
        results = []
        token.on_cancel(lambda: results.append("a"))
        token.on_cancel(lambda: results.append("b"))

        token.cancel()
        assert sorted(results) == ["a", "b"]

    @pytest.mark.unit
    def test_callback_exception_does_not_block_others(self) -> None:
        """一个回调异常不阻止其他回调执行。"""
        token = CancellationToken()
        results = []
        token.on_cancel(lambda: results.append("before"))
        token.on_cancel(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        token.on_cancel(lambda: results.append("after"))

        token.cancel()
        # "before" 一定在，"after" 也应在（异常的回调不影响后续）
        assert "before" in results
        assert "after" in results

    @pytest.mark.unit
    def test_callbacks_cleared_after_cancel(self) -> None:
        """cancel() 后内部回调列表被清空（避免内存泄漏）。"""
        token = CancellationToken()
        token.on_cancel(lambda: None)
        token.cancel()
        # 第二次 cancel 不应再触发任何回调（也不报错）
        token.cancel()


class TestLinkedToken:
    """级联取消令牌测试。"""

    @pytest.mark.unit
    def test_linked_token_cancelled_by_parent(self) -> None:
        """父 token 取消时子 token 自动取消。"""
        parent = CancellationToken()
        child = CancellationToken.create_linked(parent)

        assert not child.is_cancelled()
        parent.cancel()
        assert child.is_cancelled()

    @pytest.mark.unit
    def test_linked_token_multiple_parents(self) -> None:
        """多个父 token，任一取消即触发子 token。"""
        parent_a = CancellationToken()
        parent_b = CancellationToken()
        child = CancellationToken.create_linked(parent_a, parent_b)

        parent_b.cancel()
        assert child.is_cancelled()

    @pytest.mark.unit
    def test_child_cancel_does_not_affect_parent(self) -> None:
        """子 token 取消不影响父 token。"""
        parent = CancellationToken()
        child = CancellationToken.create_linked(parent)

        child.cancel()
        assert child.is_cancelled()
        assert not parent.is_cancelled()

    @pytest.mark.unit
    def test_linked_from_already_cancelled_parent(self) -> None:
        """从已取消的父 token 创建 linked token，子 token 立即取消。"""
        parent = CancellationToken()
        parent.cancel()

        child = CancellationToken.create_linked(parent)
        assert child.is_cancelled()


class TestThreadSafety:
    """线程安全测试。"""

    @pytest.mark.unit
    def test_concurrent_cancel_and_check(self) -> None:
        """多线程同时 cancel + is_cancelled 不出错。"""
        token = CancellationToken()
        errors: list[Exception] = []

        def cancel_worker() -> None:
            try:
                time.sleep(0.01)
                token.cancel()
            except Exception as e:
                errors.append(e)

        def check_worker() -> None:
            try:
                for _ in range(100):
                    token.is_cancelled()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=cancel_worker),
            threading.Thread(target=check_worker),
            threading.Thread(target=check_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert errors == []
        assert token.is_cancelled()

    @pytest.mark.unit
    def test_concurrent_on_cancel_registration(self) -> None:
        """多线程同时注册回调不出错。"""
        token = CancellationToken()
        results: list[int] = []
        lock = threading.Lock()

        def register_worker(worker_id: int) -> None:
            def callback() -> None:
                with lock:
                    results.append(worker_id)
            token.on_cancel(callback)

        threads = [threading.Thread(target=register_worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        token.cancel()
        assert len(results) == 10
