"""Playwright backend 真源测试。"""

from __future__ import annotations

from multiprocessing.process import BaseProcess
import time
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
import sys
from typing import cast

import pytest

from dayu.engine.tools import web_playwright_backend as backend_mod


class _FakeRoute:
    """测试用 Playwright route。"""

    def __init__(self, resource_type: str) -> None:
        """初始化 route。

        Args:
            resource_type: 资源类型。

        Returns:
            无。

        Raises:
            无。
        """

        self.request = SimpleNamespace(resource_type=resource_type)
        self.aborted = False
        self.continued = False

    def abort(self) -> None:
        """记录 abort 调用。"""

        self.aborted = True

    def continue_(self) -> None:
        """记录 continue 调用。"""

        self.continued = True


class _FakeProcess:
    """测试用子进程对象。"""

    def __init__(self, *, alive: bool, can_kill: bool = True) -> None:
        """初始化进程状态。

        Args:
            alive: 初始存活状态。
            can_kill: 是否暴露 kill 方法。

        Returns:
            无。

        Raises:
            无。
        """

        self._alive = alive
        self.join_calls: list[float] = []
        self.terminate_calls = 0
        self.kill_calls = 0
        if not can_kill:
            delattr(self, "kill")

    def is_alive(self) -> bool:
        """返回当前存活状态。"""

        return self._alive

    def join(self, timeout: float = 0) -> None:
        """记录 join 调用。"""

        self.join_calls.append(timeout)

    def terminate(self) -> None:
        """记录 terminate 调用。"""

        self.terminate_calls += 1
        self._alive = True

    def kill(self) -> None:
        """记录 kill 调用。"""

        self.kill_calls += 1
        self._alive = False


class _FakeClosable:
    """测试用可关闭对象。"""

    def __init__(self, *, method_name: str, should_raise: bool = False) -> None:
        """初始化对象。

        Args:
            method_name: 要暴露的方法名。
            should_raise: 是否在调用时抛异常。

        Returns:
            无。

        Raises:
            无。
        """

        self.calls = 0
        self.should_raise = should_raise
        setattr(self, method_name, self._call)

    def _call(self) -> None:
        """记录调用并按需抛异常。"""

        self.calls += 1
        if self.should_raise:
            raise RuntimeError("boom")


@pytest.mark.unit
def test_playwright_process_entry_puts_result_or_error() -> None:
    """验证子进程入口会回传结果或结构化错误。"""

    queue = SimpleNamespace(items=[])
    queue.put = queue.items.append

    backend_mod._playwright_process_entry(
        queue,
        lambda *, value: {"ok": True, "value": value},
        {"value": 1},
    )
    backend_mod._playwright_process_entry(
        queue,
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError(f"bad:{kwargs['value']}")),
        {"value": 2},
    )

    assert queue.items[0] == {"kind": "result", "payload": {"ok": True, "value": 1}}
    assert queue.items[1]["kind"] == "error"
    assert queue.items[1]["error_type"] == "RuntimeError"
    assert queue.items[1]["message"] == "bad:2"


@pytest.mark.unit
def test_playwright_process_helpers_cover_terminate_and_route_branching() -> None:
    """验证进程终止 helper 与资源路由 helper 的剩余分支。"""

    finished_process = _FakeProcess(alive=False)
    backend_mod._terminate_playwright_process(cast(BaseProcess, finished_process))
    assert finished_process.join_calls == [0]

    alive_process = _FakeProcess(alive=True)
    backend_mod._terminate_playwright_process(cast(BaseProcess, alive_process))
    assert alive_process.terminate_calls == 1
    assert alive_process.kill_calls == 1
    assert alive_process.join_calls == [backend_mod._PW_PROCESS_TERMINATE_GRACE_SECONDS, backend_mod._PW_PROCESS_TERMINATE_GRACE_SECONDS]

    image_route = _FakeRoute("image")
    document_route = _FakeRoute("document")
    backend_mod._route_handler_abort_resources(image_route)
    backend_mod._route_handler_abort_resources(document_route)
    assert image_route.aborted is True and image_route.continued is False
    assert document_route.aborted is False and document_route.continued is True
    assert backend_mod._normalize_playwright_channel(None) is None
    assert backend_mod._normalize_playwright_channel("  chrome  ") == "chrome"


@pytest.mark.unit
def test_close_playwright_browser_resets_globals_even_if_close_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证关闭浏览器 helper 会吞掉 close/stop 异常并重置单例。"""

    fake_browser = _FakeClosable(method_name="close", should_raise=True)
    fake_instance = _FakeClosable(method_name="stop", should_raise=True)
    monkeypatch.setattr(backend_mod, "_PW_BROWSER", fake_browser)
    monkeypatch.setattr(backend_mod, "_PW_INSTANCE", fake_instance)
    monkeypatch.setattr(backend_mod, "_PW_BROWSER_KEY", ("chrome", True))

    backend_mod._close_playwright_browser()

    assert fake_browser.calls == 1
    assert fake_instance.calls == 1
    assert backend_mod._PW_BROWSER is None
    assert backend_mod._PW_INSTANCE is None
    assert backend_mod._PW_BROWSER_KEY is None


@pytest.mark.unit
def test_get_playwright_browser_reuses_browser_and_handles_launch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证浏览器单例会复用同 key 实例，并在初始化失败时返回 None。"""

    launches: list[dict[str, object]] = []
    fake_browser = SimpleNamespace(close=lambda: None)

    def _fake_launch(**kwargs: bool | str) -> object:
        launches.append(dict(kwargs))
        return fake_browser

    fake_pw = SimpleNamespace(chromium=SimpleNamespace(launch=_fake_launch), stop=lambda: None)
    fake_sync_module = SimpleNamespace(sync_playwright=lambda: SimpleNamespace(start=lambda: fake_pw))
    monkeypatch.setitem(sys.modules, "playwright.sync_api", fake_sync_module)
    monkeypatch.setattr(backend_mod, "_PW_BROWSER", None)
    monkeypatch.setattr(backend_mod, "_PW_INSTANCE", None)
    monkeypatch.setattr(backend_mod, "_PW_BROWSER_KEY", None)

    browser = backend_mod._get_playwright_browser(playwright_channel=" chrome ", headless=False)
    reused_browser = backend_mod._get_playwright_browser(playwright_channel="chrome", headless=False)

    assert browser is fake_browser
    assert reused_browser is fake_browser
    assert launches == [
        {
            "headless": False,
            "channel": "chrome",
            "args": ["--disable-blink-features=AutomationControlled"],
        }
    ]

    warnings: list[str] = []
    failing_sync_module = SimpleNamespace(
        sync_playwright=lambda: SimpleNamespace(start=lambda: (_ for _ in ()).throw(RuntimeError("no browser")))
    )
    monkeypatch.setitem(sys.modules, "playwright.sync_api", failing_sync_module)
    monkeypatch.setattr(backend_mod, "_PW_BROWSER", None)
    monkeypatch.setattr(backend_mod, "_PW_INSTANCE", None)
    monkeypatch.setattr(backend_mod, "_PW_BROWSER_KEY", None)
    monkeypatch.setattr(backend_mod.Log, "warning", lambda message, *, module: warnings.append(f"{module}:{message}"))

    assert backend_mod._get_playwright_browser(playwright_channel=None, headless=True) is None
    assert any("Playwright 浏览器初始化失败" in message for message in warnings)


@pytest.mark.unit
def test_is_picklable_worker_detects_unpicklable() -> None:
    """不可 pickle 的 callable 应返回 False。"""

    class _Unpicklable:
        def __call__(self) -> dict[str, int]:
            return {}

        def __reduce__(self) -> tuple[type, tuple[int, ...]]:
            raise TypeError("cannot pickle")

    # 函数级 lambda 无法 pickle（捕获了本地作用域）
    assert backend_mod._is_picklable_worker(lambda: {}) is False
    assert backend_mod._is_picklable_worker(_Unpicklable()) is False
    # 模块级函数可以 pickle
    assert backend_mod._is_picklable_worker(_noop_callable) is True


def _noop_callable() -> dict[str, int]:
    """测试用可 pickle callable。"""

    return {}


@pytest.mark.unit
def test_poll_playwright_result_queue_returns_none_on_timeout() -> None:
    """空队列在超时后应返回 None。"""

    class _EmptyQueue:
        def get(self, *, timeout: float = 0) -> None:
            raise Empty()

        def get_nowait(self) -> None:
            raise Empty()

    result = backend_mod._poll_playwright_result_queue(result_queue=_EmptyQueue(), timeout=0.1)
    assert result is None

    result_nowait = backend_mod._poll_playwright_result_queue(result_queue=_EmptyQueue(), timeout=0)
    assert result_nowait is None


@pytest.mark.unit
def test_poll_playwright_result_queue_returns_item() -> None:
    """有结果的队列应返回结果。"""

    class _QueueWithItem:
        def get(self, *, timeout: float = 0) -> dict[str, object]:
            return {"ok": True}

        def get_nowait(self) -> dict[str, object]:
            return {"ok": True}

    result = backend_mod._poll_playwright_result_queue(result_queue=_QueueWithItem(), timeout=0.1)
    assert result == {"ok": True}

    result_nowait = backend_mod._poll_playwright_result_queue(result_queue=_QueueWithItem(), timeout=0)
    assert result_nowait == {"ok": True}


@pytest.mark.unit
def test_close_playwright_result_queue_handles_exceptions() -> None:
    """close 和 join_thread 抛异常时不应影响函数返回。"""

    class _BrokenQueue:
        def close(self) -> None:
            raise RuntimeError("close failed")

        def join_thread(self) -> None:
            raise RuntimeError("join failed")

    backend_mod._close_playwright_result_queue(_BrokenQueue())


@pytest.mark.unit
def test_close_playwright_result_queue_handles_missing_methods() -> None:
    """队列无 close/join_thread 方法时应安全跳过。"""

    backend_mod._close_playwright_result_queue(SimpleNamespace())


@pytest.mark.unit
def test_normalize_playwright_storage_state_dir_variants() -> None:
    """测试 storage state 目录规范化各分支。"""

    assert backend_mod._normalize_playwright_storage_state_dir(None) is None
    assert backend_mod._normalize_playwright_storage_state_dir("") is None
    assert backend_mod._normalize_playwright_storage_state_dir("  ") is None
    assert backend_mod._normalize_playwright_storage_state_dir("/tmp/states") == "/tmp/states"
    assert backend_mod._normalize_playwright_storage_state_dir("  /tmp/states  ") == "/tmp/states"


@pytest.mark.unit
def test_resolve_playwright_storage_state_path_variants(tmp_path: Path) -> None:
    """测试 storage state 路径解析各分支。"""

    # dir 为 None → 空字符串
    assert backend_mod._resolve_playwright_storage_state_path(url="https://example.com", playwright_storage_state_dir=None) == ""

    # url 无 host → 空字符串
    assert backend_mod._resolve_playwright_storage_state_path(url="", playwright_storage_state_dir=str(tmp_path)) == ""

    # 文件不存在 → 空字符串
    assert backend_mod._resolve_playwright_storage_state_path(url="https://example.com", playwright_storage_state_dir=str(tmp_path)) == ""

    # 文件存在 → 返回路径
    state_file = tmp_path / "example.com.json"
    state_file.write_text("{}")
    result = backend_mod._resolve_playwright_storage_state_path(url="https://example.com", playwright_storage_state_dir=str(tmp_path))
    assert result == str(state_file)

    # www. 前缀 host 应 fallback 到无 www 版本
    www_file = tmp_path / "www.example.org.json"
    www_file.write_text("{}")
    result2 = backend_mod._resolve_playwright_storage_state_path(url="https://www.example.org", playwright_storage_state_dir=str(tmp_path))
    assert result2 == str(www_file)


@pytest.mark.unit
def test_get_remaining_playwright_timeout_ms() -> None:
    """测试剩余超时毫秒数计算。"""

    fake_time = 100.0
    assert backend_mod._get_remaining_playwright_timeout_ms(105.0, time_monotonic=lambda: fake_time) == 5000
    assert backend_mod._get_remaining_playwright_timeout_ms(100.0, time_monotonic=lambda: fake_time) == 0
    assert backend_mod._get_remaining_playwright_timeout_ms(99.0, time_monotonic=lambda: fake_time) == 0
    # 小数进位
    assert backend_mod._get_remaining_playwright_timeout_ms(100.002, time_monotonic=lambda: 100.0) == 2


@pytest.mark.unit
def test_require_playwright_timeout_ms_raises_on_expired() -> None:
    """deadline 已过时应抛 RuntimeError。"""

    with pytest.raises(RuntimeError, match="Playwright 页面加载超时"):
        backend_mod._require_playwright_timeout_ms(0.0, time_monotonic=lambda: 1.0)


@pytest.mark.unit
def test_require_playwright_timeout_ms_returns_value() -> None:
    """deadline 未过时应返回剩余毫秒。"""

    result = backend_mod._require_playwright_timeout_ms(10.0, time_monotonic=lambda: 0.0)
    assert result == 10000


@pytest.mark.unit
def test_close_playwright_browser_noop_when_already_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """浏览器实例为 None 时关闭应为 no-op。"""

    monkeypatch.setattr(backend_mod, "_PW_BROWSER", None)
    monkeypatch.setattr(backend_mod, "_PW_INSTANCE", None)
    monkeypatch.setattr(backend_mod, "_PW_BROWSER_KEY", ("chrome", True))
    backend_mod._close_playwright_browser()
    assert backend_mod._PW_BROWSER is None
    assert backend_mod._PW_INSTANCE is None
    assert backend_mod._PW_BROWSER_KEY is None


def _fake_resolve_timeout(_timeout: float, **_kw: float | str | None) -> float:
    """测试用 timeout 预算解析。"""

    return 10.0


def _fake_detect_bot_challenge(**_kw: bool | str) -> SimpleNamespace:
    """测试用 bot challenge 检测。"""

    return SimpleNamespace(challenge_detected=False, challenge_signals=[])


@pytest.mark.unit
def test_fetch_and_convert_with_playwright_playwright_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """playwright 未安装时应返回 unavailable 结果。"""

    # 模拟 playwright 包不存在：清除 sys.modules 中的 playwright 相关模块
    for key in list(sys.modules):
        if key.startswith("playwright"):
            monkeypatch.delitem(sys.modules, key, raising=False)
    # 阻止 import playwright 重新加载
    monkeypatch.setitem(sys.modules, "playwright", None)
    result = backend_mod._fetch_and_convert_with_playwright(
        url="https://example.com",
        timeout_seconds=10.0,
        headers={},
        timeout_budget=None,
        deadline_monotonic=time.monotonic() + 60,
        cancellation_token=None,
        playwright_channel=None,
        playwright_storage_state_path="",
        resolve_timeout_budget=_fake_resolve_timeout,
        playwright_sync_worker=lambda **_kw: {},
        detect_bot_challenge=_fake_detect_bot_challenge,
    )
    assert result["ok"] is False
    assert result["reason"] == "playwright_not_installed"


@pytest.mark.unit
def test_fetch_and_convert_with_playwright_timeout_on_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_timeout_budget 抛 TimeoutError 时应返回 timeout 结果。"""

    from requests import Timeout

    def _raise_timeout(_timeout: float, **_kw: float | str | None) -> float:
        raise Timeout()

    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=lambda: None))
    result = backend_mod._fetch_and_convert_with_playwright(
        url="https://example.com",
        timeout_seconds=10.0,
        headers={},
        timeout_budget=None,
        deadline_monotonic=time.monotonic() + 60,
        cancellation_token=None,
        playwright_channel=None,
        playwright_storage_state_path="",
        resolve_timeout_budget=_raise_timeout,
        playwright_sync_worker=lambda **_kw: {},
        detect_bot_challenge=_fake_detect_bot_challenge,
    )
    assert result["ok"] is False
    assert result["reason"] == "playwright_timeout"


@pytest.mark.unit
def test_fetch_and_convert_with_playwright_nonpicklable_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """worker 不可 pickle 时走内联执行路径（非子进程）。"""

    fake_result = {"ok": True, "content": "hello", "http_status": 200, "response_headers": {}}
    monkeypatch.setitem(sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=lambda: None))

    result = backend_mod._fetch_and_convert_with_playwright(
        url="https://example.com",
        timeout_seconds=10.0,
        headers={},
        timeout_budget=None,
        deadline_monotonic=time.monotonic() + 60,
        cancellation_token=None,
        playwright_channel=None,
        playwright_storage_state_path="",
        resolve_timeout_budget=_fake_resolve_timeout,
        playwright_sync_worker=lambda **_kw: fake_result,
        detect_bot_challenge=_fake_detect_bot_challenge,
    )
    assert result == fake_result