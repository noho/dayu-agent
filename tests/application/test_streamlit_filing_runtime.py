"""filing 下载 runtime 并发语义测试。"""

from __future__ import annotations

import threading
from queue import Queue
from typing import cast

import pytest

from dayu.web.streamlit.pages.filing import download_panel as panel_module
from dayu.web.streamlit.pages.filing.download_progress import (
    DownloadQueueEvent,
    DownloadStatus,
    DownloadTaskState,
    create_download_task,
)
from dayu.web.streamlit.pages.filing.download_panel import DownloadRuntimeState


class _FakeSessionState(dict[str, object]):
    """支持属性访问的简化 session_state。"""

    def __getattr__(self, name: str) -> object:
        if name in self:
            return self[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: object) -> None:
        self[name] = value


class _FakeStreamlit:
    """最小化 Streamlit 替身。"""

    def __init__(self) -> None:
        self.session_state = _FakeSessionState()


class _NeverEndingWorker:
    """永远处于存活态的线程替身。"""

    def __init__(self) -> None:
        self.join_timeouts: list[float | None] = []

    def is_alive(self) -> bool:
        return True

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)


@pytest.fixture
def fake_filing_st(monkeypatch: pytest.MonkeyPatch) -> _FakeStreamlit:
    """将下载状态模块内 st 替换为可控假对象。"""

    fake = _FakeStreamlit()
    monkeypatch.setattr(panel_module, "st", fake)
    return fake


def _set_active_download_task(
    fake_st: _FakeStreamlit,
    *,
    session_id: str,
    ticker: str,
) -> DownloadTaskState:
    """初始化 active_downloads 并返回任务状态对象。"""

    task = create_download_task(session_id=session_id, ticker=ticker)
    fake_st.session_state.active_downloads = {session_id: task}
    return task


@pytest.mark.unit
def test_poll_download_runtime_events_worker_exit_without_done_keeps_runtime(
    fake_filing_st: _FakeStreamlit,
) -> None:
    """未收到 done 信号时，即使线程已退出也不能提前清理 runtime。"""

    task = _set_active_download_task(fake_filing_st, session_id="session-a", ticker="AAPL")

    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()

    runtime_state = panel_module._get_download_runtime_state()
    runtime_state["session-a"] = {
        "worker": worker,
        "event_queue": Queue(),
        "done": False,
    }

    panel_module.poll_download_runtime_events()

    assert "session-a" in runtime_state
    assert task.status == DownloadStatus.RUNNING


@pytest.mark.unit
def test_poll_download_runtime_events_done_signal_triggers_cleanup(
    fake_filing_st: _FakeStreamlit,
) -> None:
    """收到 done 且队列清空后，应执行清理并移除 runtime 句柄。"""

    task = _set_active_download_task(fake_filing_st, session_id="session-b", ticker="MSFT")

    event_queue: Queue[DownloadQueueEvent] = Queue()
    event_queue.put(DownloadQueueEvent(kind="done"))
    worker = threading.Thread(target=lambda: None)
    worker.start()
    worker.join()

    runtime_state = panel_module._get_download_runtime_state()
    runtime_state["session-b"] = {
        "worker": worker,
        "event_queue": event_queue,
        "done": False,
    }

    panel_module.poll_download_runtime_events()

    assert "session-b" not in runtime_state
    assert task.status == DownloadStatus.FAILED


@pytest.mark.unit
def test_poll_download_runtime_events_join_timeout_keeps_runtime(
    fake_filing_st: _FakeStreamlit,
) -> None:
    """done 后若 join 超时线程未结束，应保留 runtime 防止事件丢失。"""

    task = _set_active_download_task(fake_filing_st, session_id="session-c", ticker="NVDA")

    event_queue: Queue[DownloadQueueEvent] = Queue()
    event_queue.put(DownloadQueueEvent(kind="done"))
    never_ending_worker = _NeverEndingWorker()

    runtime_state = panel_module._get_download_runtime_state()
    runtime_state["session-c"] = cast(
        DownloadRuntimeState,
        {
            "worker": never_ending_worker,
            "event_queue": event_queue,
            "done": False,
        },
    )

    panel_module.poll_download_runtime_events()

    assert "session-c" in runtime_state
    assert never_ending_worker.join_timeouts == [0.1]
    assert task.status == DownloadStatus.RUNNING


@pytest.mark.unit
def test_dispatch_download_runtime_event_unknown_kind_raises(
    fake_filing_st: _FakeStreamlit,
) -> None:
    """未知队列事件类型必须显式抛错，避免静默丢失。"""

    _set_active_download_task(fake_filing_st, session_id="session-d", ticker="TSLA")

    with pytest.raises(ValueError, match="未知下载运行时事件类型"):
        panel_module._dispatch_download_runtime_event(
            "session-d",
            DownloadQueueEvent(kind="unexpected"),  # type: ignore[arg-type]
        )
