"""Streamlit 下载状态组件单元测试。"""

from __future__ import annotations

import importlib

import pytest

filing_download_progress_module = importlib.import_module(
    "dayu.web.streamlit.components.filing_download_progress"
)
# 显式导入类型，便于测试中标注
DownloadTaskState = filing_download_progress_module.DownloadTaskState
LogEntry = filing_download_progress_module.LogEntry


@pytest.mark.unit
def test_download_task_state_round_trip() -> None:
    """验证下载任务状态可序列化并无损恢复。"""
    state = DownloadTaskState(
        session_id="session-1",
        ticker="AAPL",
        status="running",
        progress=25.0,
        message="下载中",
        downloaded_count=2,
        errors=["err1"],
    )

    restored = DownloadTaskState.from_dict(state.to_dict())

    assert restored.session_id == "session-1"
    assert restored.ticker == "AAPL"
    assert restored.status == "running"
    assert restored.progress == 25.0
    assert restored.downloaded_count == 2
    assert restored.errors == ["err1"]


@pytest.mark.unit
def test_download_task_state_from_dict_defaults() -> None:
    """验证 from_dict 对缺失字段使用默认值。"""
    restored = DownloadTaskState.from_dict({})

    assert restored.session_id == ""
    assert restored.ticker == ""
    assert restored.status == filing_download_progress_module._DOWNLOAD_STATUS_PENDING
    assert restored.progress == 0.0
    assert restored.message == "等待开始..."
    assert restored.downloaded_count == 0
    assert restored.downloaded_filing_count == 0
    assert restored.total_count is None
    assert restored.errors == []
    assert restored.logs == []
    assert restored.started_at is None
    assert restored.completed_at is None


@pytest.mark.unit
def test_build_download_logs_html_renders_level_and_message() -> None:
    """验证日志 HTML 渲染包含级别与消息文本。"""
    logs: list[LogEntry] = [
        {
            "timestamp": "2026-01-01T10:00:00",
            "message": "开始下载",
            "level": "info",
        },
        {
            "timestamp": "2026-01-01T10:00:01",
            "message": "下载失败",
            "level": "error",
        },
    ]

    html = filing_download_progress_module.build_download_logs_html(logs)

    assert "INFO" in html
    assert "ERROR" in html
    assert "开始下载" in html
    assert "下载失败" in html


@pytest.mark.unit
def test_build_download_logs_html_truncates_to_max() -> None:
    """验证日志超过最大条数时仅展示最近条目。"""
    max_count = filing_download_progress_module._STATUS_CONTAINER_MAX_LOG_ITEMS
    all_logs: list[LogEntry] = [
        {"timestamp": "2026-01-01T10:00:00", "message": f"msg-{i}", "level": "info"}
        for i in range(max_count + 30)
    ]

    html = filing_download_progress_module.build_download_logs_html(all_logs)

    # 最早的消息不会被渲染
    assert "msg-0" not in html
    assert "msg-29" not in html
    # 最近的消息会被渲染
    assert f"msg-{max_count + 29}" in html


@pytest.mark.unit
def test_build_download_logs_html_empty_list() -> None:
    """验证空日志列表返回占位文本。"""
    html = filing_download_progress_module.build_download_logs_html([])
    assert "暂无日志" in html


@pytest.mark.unit
def test_build_download_logs_html_warning_level() -> None:
    """验证警告级别日志渲染为 WARN 标记。"""
    logs: list[LogEntry] = [
        {"timestamp": "2026-01-01T10:00:00", "message": "警告消息", "level": "warning"},
    ]
    html = filing_download_progress_module.build_download_logs_html(logs)
    assert "WARN" in html
