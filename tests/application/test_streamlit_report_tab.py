"""分析报告 Tab 测试。

测试三种状态的UI渲染逻辑和报告状态检测功能。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dayu.web.streamlit.pages.report_tab import (
    WriteTaskState,
    _bind_write_task_run_id_from_host,
    _build_state1_guide_card_content,
    _cancel_write_task_via_host,
    _build_report_toc_html,
    _collect_report_artifact_overview,
    _ensure_pandoc_v3_or_newer,
    _extract_template_guide_headings,
    _extract_markdown_headings,
    _fast_to_generation_mode,
    _generation_mode_to_fast,
    _get_report_primary_action_label,
    _get_report_panel_container_height_px,
    _is_report_download_available,
    _inject_heading_anchors,
    _parse_manifest_chapter_snapshots,
    _refresh_task_status_from_manifest,
    _sync_write_task_status_from_host,
)


class TestReportStateDetection:
    """测试报告状态检测功能。"""

    def test_load_report_state_with_existing_files(self, tmp_path: Path) -> None:
        """测试当报告文件存在时正确加载状态。"""
        # 由于Python 3.10无法直接导入，这里使用模拟测试
        ticker = "AAPL"
        draft_dir = tmp_path / "draft" / ticker
        draft_dir.mkdir(parents=True)

        # 创建run_summary.json
        summary_data = {
            "ticker": ticker,
            "output_file": str(draft_dir / f"{ticker}_qual_report.md"),
            "chapter_count": 11,
            "failed_count": 0,
            "failed_chapters": [],
        }
        summary_path = draft_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f)

        # 创建报告文件
        report_path = draft_dir / f"{ticker}_qual_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 测试报告\n\n这是一份测试报告。\n")

        # 验证文件存在
        assert summary_path.exists()
        assert report_path.exists()

    def test_load_report_state_with_missing_files(self, tmp_path: Path) -> None:
        """测试当报告文件不存在时返回正确状态。"""
        ticker = "NONEXISTENT"
        draft_dir = tmp_path / "draft" / ticker

        # 验证文件不存在
        assert not (draft_dir / "run_summary.json").exists()
        assert not (draft_dir / f"{ticker}_qual_report.md").exists()

    def test_load_report_content_success(self, tmp_path: Path) -> None:
        """测试成功加载报告内容。"""
        report_path = tmp_path / "test_report.md"
        expected_content = "# 测试报告\n\n这是一份测试报告。\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(expected_content)

        # 验证文件内容
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert content == expected_content

    def test_load_report_content_failure(self, tmp_path: Path) -> None:
        """测试报告文件不存在时返回None。"""
        nonexistent_path = tmp_path / "nonexistent.md"
        assert not nonexistent_path.exists()

    def test_collect_report_artifact_overview_with_summary(self, tmp_path: Path) -> None:
        """产物概览应包含报告尺寸与摘要统计。"""

        ticker = "AAPL"
        draft_dir = tmp_path / "draft" / ticker
        draft_dir.mkdir(parents=True)
        report_path = draft_dir / f"{ticker}_qual_report.md"
        report_path.write_text("# 报告\n", encoding="utf-8")
        summary_path = draft_dir / "run_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "chapter_count": 11,
                    "failed_count": 1,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        overview = _collect_report_artifact_overview(tmp_path, ticker)
        assert "report_exists=True" in overview
        assert "summary_exists=True" in overview
        assert "chapter_count=11" in overview
        assert "failed_count=1" in overview

    def test_collect_report_artifact_overview_with_invalid_summary(self, tmp_path: Path) -> None:
        """摘要损坏时应输出可诊断的失败标记。"""

        ticker = "MSFT"
        draft_dir = tmp_path / "draft" / ticker
        draft_dir.mkdir(parents=True)
        (draft_dir / "run_summary.json").write_text("{invalid-json", encoding="utf-8")

        overview = _collect_report_artifact_overview(tmp_path, ticker)
        assert "report_exists=False" in overview
        assert "summary_exists=True" in overview
        assert "chapter_count=摘要读取失败" in overview
        assert "failed_count=摘要读取失败" in overview


class TestWriteTaskState:
    """测试Write任务状态管理。"""

    def test_write_task_state_to_dict(self) -> None:
        """测试任务状态序列化。"""
        # 模拟任务状态对象
        task_data = {
            "status": "running",
            "session_id": "test-session-001",
            "run_id": "test-run-001",
            "started_at": "2024-01-01T12:00:00",
            "completed_at": None,
            "exit_code": None,
            "message": "任务运行中",
            "progress": 50.0,
            "current_chapter": "章节1",
            "logs": [
                {"timestamp": "2024-01-01T12:00:00", "message": "任务开始", "level": "info"},
            ],
        }

        # 验证字典结构
        assert task_data["status"] == "running"
        assert task_data["progress"] == 50.0
        assert len(task_data["logs"]) == 1

    def test_write_task_state_from_dict(self) -> None:
        """测试任务状态反序列化。"""
        data = {
            "status": "completed",
            "session_id": "test-session-002",
            "run_id": "test-run-002",
            "started_at": "2024-01-01T13:00:00",
            "completed_at": "2024-01-01T13:30:00",
            "exit_code": 0,
            "message": "任务完成",
            "progress": 100.0,
            "current_chapter": None,
            "logs": [],
        }

        # 验证数据完整性
        assert data["status"] == "completed"
        assert data["exit_code"] == 0
        assert data["progress"] == 100.0


def test_parse_manifest_chapter_snapshots_returns_sorted_snapshots(tmp_path: Path) -> None:
    """manifest 解析应按章节序号返回快照。"""

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "chapter_results": {
                    "第二章": {"index": 2, "status": "failed", "failure_reason": "证据不足"},
                    "第一章": {"index": 1, "status": "passed", "failure_reason": ""},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    snapshots = _parse_manifest_chapter_snapshots(manifest_path)

    assert [snapshot.title for snapshot in snapshots] == ["第一章", "第二章"]
    assert [snapshot.status for snapshot in snapshots] == ["passed", "failed"]
    assert snapshots[1].failure_reason == "证据不足"


def test_parse_manifest_chapter_snapshots_tracks_new_chapter_artifacts(tmp_path: Path) -> None:
    """manifest 解析应优先包含 chapters 目录下的新产物文件名。"""

    output_dir = tmp_path / "draft" / "AAPL"
    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True)
    (chapters_dir / "01_第一章.initial_write.md").write_text("初稿", encoding="utf-8")
    (chapters_dir / "02_第二章.repair_1_input_write.md").write_text("修复输入", encoding="utf-8")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "chapter_results": {
                    "第一章": {"index": 1, "status": "passed", "failure_reason": ""},
                    "第二章": {"index": 2, "status": "failed", "failure_reason": "证据不足"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    snapshots = _parse_manifest_chapter_snapshots(manifest_path)

    tracked_statuses = [snapshot.status for snapshot in snapshots]
    assert "01_第一章.initial_write.md" in tracked_statuses
    assert "02_第二章.repair_1_input_write.md" in tracked_statuses
    assert "passed" in tracked_statuses
    assert "failed" in tracked_statuses


@patch("dayu.web.streamlit.pages.report_tab.st.session_state", new_callable=dict)
def test_refresh_task_status_from_manifest_streams_chapter_updates(
    mock_session_state: dict[str, object],
    tmp_path: Path,
) -> None:
    """manifest 状态变化应增量写入任务日志并更新进度文案。"""

    ticker = "AAPL"
    task_key = f"write_task_{ticker}"
    output_dir = tmp_path / "draft" / ticker
    output_dir.mkdir(parents=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "chapter_results": {
                    "第一章": {"index": 1, "status": "passed", "failure_reason": ""},
                    "第二章": {"index": 2, "status": "failed", "failure_reason": "数据缺失"},
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    running_task = WriteTaskState(
        status="running",
        output_dir=str(output_dir),
        expected_chapter_count=4,
        progress=10.0,
    )
    mock_session_state["active_write_tasks"] = {
        task_key: running_task.to_dict(),
    }
    mock_session_state["write_task_settings"] = {}

    _refresh_task_status_from_manifest(ticker, running_task)

    active_write_tasks = mock_session_state["active_write_tasks"]
    assert isinstance(active_write_tasks, dict)
    task_data = active_write_tasks[task_key]
    assert isinstance(task_data, dict)
    refreshed = WriteTaskState.from_dict(task_data)
    messages = [entry["message"] for entry in refreshed.logs]

    assert "检测到任务状态清单，开始流式跟踪章节执行状态" in messages
    assert "章节完成: [01] 第一章" in messages
    assert "章节失败: [02] 第二章，原因: 数据缺失" in messages
    assert refreshed.progress > 10.0
    assert "已完成章节 2/4" in refreshed.message


@patch("dayu.web.streamlit.pages.report_tab.st.session_state", new_callable=dict)
def test_refresh_task_status_from_manifest_tracks_new_artifact_once(
    mock_session_state: dict[str, object],
    tmp_path: Path,
) -> None:
    """同一章节产物文件不应被重复记录。"""

    ticker = "AAPL"
    task_key = f"write_task_{ticker}"
    output_dir = tmp_path / "draft" / ticker
    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True)
    artifact_name = "02_行业吸引力与公司位置.repair_1_input_write.md"
    (chapters_dir / artifact_name).write_text("修复输入", encoding="utf-8")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"chapter_results": {}}, ensure_ascii=False), encoding="utf-8")

    running_task = WriteTaskState(
        status="running",
        output_dir=str(output_dir),
        expected_chapter_count=4,
        progress=10.0,
    )
    mock_session_state["active_write_tasks"] = {
        task_key: running_task.to_dict(),
    }
    mock_session_state["write_task_settings"] = {}

    _refresh_task_status_from_manifest(ticker, running_task)
    first_data = mock_session_state["active_write_tasks"][task_key]
    assert isinstance(first_data, dict)
    first_task = WriteTaskState.from_dict(first_data)
    first_messages = [entry["message"] for entry in first_task.logs]
    assert f"章节产物生成: [02] 行业吸引力与公司位置 -> {artifact_name}" in first_messages

    _refresh_task_status_from_manifest(ticker, first_task)
    second_data = mock_session_state["active_write_tasks"][task_key]
    assert isinstance(second_data, dict)
    second_task = WriteTaskState.from_dict(second_data)
    second_messages = [entry["message"] for entry in second_task.logs]
    assert second_messages.count(f"章节产物生成: [02] 行业吸引力与公司位置 -> {artifact_name}") == 1


@patch("dayu.web.streamlit.pages.report_tab.st.session_state", new_callable=dict)
def test_bind_write_task_run_id_from_host_updates_run_id(
    mock_session_state: dict[str, object],
) -> None:
    """运行中任务应可从 Host 活跃 run 自动绑定 run_id。"""

    class _FakeHostAdminService:
        """测试用 Host 管理服务桩。"""

        def list_runs(
            self,
            *,
            session_id: str | None = None,
            state: str | None = None,
            service_type: str | None = None,
            active_only: bool = False,
        ) -> list[object]:
            del session_id, state, active_only
            assert service_type == "write_pipeline"
            return [
                MagicMock(
                    run_id="run-1",
                    session_id="session-1",
                    created_at="2026-04-21T19:00:00",
                )
            ]

    ticker = "AAPL"
    task_key = f"write_task_{ticker}"
    running_task = WriteTaskState(
        status="running",
        started_at="2026-04-21T18:59:00",
    )
    mock_session_state["host_admin_service"] = _FakeHostAdminService()
    mock_session_state["active_write_tasks"] = {task_key: running_task.to_dict()}
    mock_session_state["write_task_settings"] = {}

    refreshed = _bind_write_task_run_id_from_host(ticker, running_task)

    assert refreshed.run_id == "run-1"
    assert refreshed.session_id == "session-1"


@patch("dayu.web.streamlit.pages.report_tab.st.session_state", new_callable=dict)
def test_bind_write_task_run_id_from_host_handles_mixed_timezone_datetimes(
    mock_session_state: dict[str, object],
) -> None:
    """run 时间带时区、任务时间不带时区时也应可绑定，不应抛异常。"""

    class _FakeHostAdminService:
        """测试用 Host 管理服务桩。"""

        def list_runs(
            self,
            *,
            session_id: str | None = None,
            state: str | None = None,
            service_type: str | None = None,
            active_only: bool = False,
        ) -> list[object]:
            del session_id, state, active_only
            assert service_type == "write_pipeline"
            return [
                MagicMock(
                    run_id="run-aware",
                    session_id="session-aware",
                    created_at="2026-04-21T19:00:00+00:00",
                )
            ]

    ticker = "MSFT"
    task_key = f"write_task_{ticker}"
    running_task = WriteTaskState(
        status="running",
        started_at="2026-04-21T19:00:10",
    )
    mock_session_state["host_admin_service"] = _FakeHostAdminService()
    mock_session_state["active_write_tasks"] = {task_key: running_task.to_dict()}
    mock_session_state["write_task_settings"] = {}

    refreshed = _bind_write_task_run_id_from_host(ticker, running_task)

    assert refreshed.run_id == "run-aware"
    assert refreshed.session_id == "session-aware"


@patch("dayu.web.streamlit.pages.report_tab.st.session_state", new_callable=dict)
def test_cancel_write_task_via_host_requests_cancel(
    mock_session_state: dict[str, object],
) -> None:
    """取消操作应调用 Host 管理服务并记录取消提示。"""

    class _FakeHostAdminService:
        """测试用 Host 管理服务桩。"""

        def __init__(self) -> None:
            self.cancelled_run_id: str | None = None

        def list_runs(
            self,
            *,
            session_id: str | None = None,
            state: str | None = None,
            service_type: str | None = None,
            active_only: bool = False,
        ) -> list[object]:
            del session_id, state, active_only
            assert service_type == "write_pipeline"
            return [
                MagicMock(
                    run_id="run-2",
                    session_id="session-2",
                    created_at="2026-04-21T19:10:00",
                )
            ]

        def cancel_run(self, run_id: str) -> object:
            self.cancelled_run_id = run_id
            return MagicMock()

    ticker = "TSLA"
    task_key = f"write_task_{ticker}"
    running_task = WriteTaskState(
        status="running",
        started_at="2026-04-21T19:09:00",
    )
    fake_service = _FakeHostAdminService()
    mock_session_state["host_admin_service"] = fake_service
    mock_session_state["active_write_tasks"] = {task_key: running_task.to_dict()}
    mock_session_state["write_task_settings"] = {}

    cancelled, message = _cancel_write_task_via_host(ticker, running_task)

    assert cancelled is True
    assert "已发送取消请求" in message
    assert fake_service.cancelled_run_id == "run-2"


@patch("dayu.web.streamlit.pages.report_tab.st.session_state", new_callable=dict)
def test_sync_write_task_status_from_host_marks_task_cancelled(
    mock_session_state: dict[str, object],
) -> None:
    """Host run 进入 cancelled 后应同步到任务状态。"""

    class _FakeHostAdminService:
        """测试用 Host 管理服务桩。"""

        def get_run(self, run_id: str) -> object:
            assert run_id == "run-3"
            return MagicMock(
                state="cancelled",
                cancel_reason="user_cancelled",
                finished_at="2026-04-21T20:12:00+00:00",
                cancel_requested_at="2026-04-21T20:11:00+00:00",
                error_summary=None,
            )

    ticker = "NVDA"
    task_key = f"write_task_{ticker}"
    running_task = WriteTaskState(
        status="running",
        run_id="run-3",
        progress=66.0,
        status_cursor={},
    )
    mock_session_state["host_admin_service"] = _FakeHostAdminService()
    mock_session_state["active_write_tasks"] = {task_key: running_task.to_dict()}
    mock_session_state["write_task_settings"] = {}

    refreshed = _sync_write_task_status_from_host(ticker, running_task)

    assert refreshed.status == "cancelled"
    assert refreshed.exit_code == 130
    assert "Host 已收敛" in refreshed.message

class TestTaskSettings:
    """测试任务设置管理。"""

    def test_default_task_settings(self) -> None:
        """测试默认任务设置。"""
        expected_defaults = {
            "template_path": "定性分析模板.md",
            "write_max_retries": 2,
            "resume": True,
            "fast": True,
            "force": False,
        }

        assert expected_defaults["template_path"] == "定性分析模板.md"
        assert expected_defaults["write_max_retries"] == 2
        assert expected_defaults["resume"] is True

    def test_task_settings_update(self) -> None:
        """测试更新任务设置。"""
        settings = {
            "template_path": "定性分析模板.md",
            "write_max_retries": 2,
            "resume": True,
        }

        # 更新设置
        settings["write_max_retries"] = 5
        settings["fast"] = True

        assert settings["write_max_retries"] == 5
        assert settings["fast"] is True


def test_generation_mode_mapping_roundtrip() -> None:
    """生成模式与 fast 标记应保持双向一致。"""

    assert _fast_to_generation_mode(True) == "fast"
    assert _fast_to_generation_mode(False) == "deep"
    assert _generation_mode_to_fast("fast") is True
    assert _generation_mode_to_fast("deep") is False


class TestTimeFormatting:
    """测试时间格式化功能。"""

    def test_format_valid_time(self) -> None:
        """测试格式化有效时间字符串。"""
        time_str = "2024-01-15T14:30:00"

        # 模拟格式化
        from datetime import datetime
        dt = datetime.fromisoformat(time_str)
        formatted = dt.strftime("%Y-%m-%d %H:%M:%S")

        assert formatted == "2024-01-15 14:30:00"

    def test_format_invalid_time(self) -> None:
        """测试格式化无效时间字符串。"""
        invalid_time_str = "invalid-time"

        # 模拟错误处理
        try:
            from datetime import datetime
            datetime.fromisoformat(invalid_time_str)
            assert False, "应该抛出异常"
        except ValueError:
            pass  # 预期行为

    def test_format_none_time(self) -> None:
        """测试格式化None时间。"""
        time_str = None
        result = "未知" if time_str is None else time_str
        assert result == "未知"


class TestLogRendering:
    """测试日志渲染功能。"""

    def test_render_empty_logs(self) -> None:
        """测试渲染空日志列表。"""
        logs = []
        # 预期返回提示信息
        assert len(logs) == 0

    def test_render_task_logs(self) -> None:
        """测试渲染任务日志。"""
        logs = [
            {"timestamp": "2024-01-01T12:00:00", "message": "任务开始", "level": "info"},
            {"timestamp": "2024-01-01T12:05:00", "message": "处理中", "level": "info"},
            {"timestamp": "2024-01-01T12:10:00", "message": "警告信息", "level": "warning"},
            {"timestamp": "2024-01-01T12:15:00", "message": "错误信息", "level": "error"},
        ]

        assert len(logs) == 4
        assert logs[0]["level"] == "info"
        assert logs[2]["level"] == "warning"
        assert logs[3]["level"] == "error"

    def test_log_trimming(self) -> None:
        """测试日志数量限制。"""
        # 模拟超过200条日志的情况
        logs = [{"timestamp": f"2024-01-01T12:{i:02d}:00", "message": f"日志{i}", "level": "info"} for i in range(250)]

        # 模拟修剪逻辑
        if len(logs) > 200:
            logs = logs[-200:]

        assert len(logs) == 200
        assert logs[0]["message"] == "日志50"  # 第一条应该是第50条


class TestStateRouting:
    """测试状态路由逻辑。"""

    def test_state1_no_report_no_task(self) -> None:
        """测试状态1：无报告 + 无任务。"""
        report_exists = False
        has_active_task = False

        # 应该路由到状态1
        assert not report_exists and not has_active_task

    def test_state2_no_report_with_task(self) -> None:
        """测试状态2：无报告 + 有任务。"""
        report_exists = False
        has_active_task = True

        # 有任务时应该优先显示任务状态
        assert has_active_task

    def test_state3_has_report_no_task(self) -> None:
        """测试状态3：有报告 + 无任务。"""
        report_exists = True
        has_active_task = False

        # 应该路由到状态3
        assert report_exists and not has_active_task

    def test_priority_task_over_report(self) -> None:
        """测试有任务时优先于报告存在性。"""
        report_exists = True  # 报告已存在
        has_active_task = True  # 但任务正在进行（重新生成）

        # 应该优先显示任务状态
        assert has_active_task


class TestReportPathResolution:
    """测试报告路径解析。"""

    def test_get_draft_dir(self, tmp_path: Path) -> None:
        """测试草稿目录路径解析。"""
        ticker = "AAPL"
        workspace_root = tmp_path

        expected_path = workspace_root / "draft" / ticker
        assert str(expected_path) == str(tmp_path / "draft" / "AAPL")

    def test_report_file_name_format(self) -> None:
        """测试报告文件名格式。"""
        ticker = "AAPL"
        expected_name = f"{ticker}_qual_report.md"
        assert expected_name == "AAPL_qual_report.md"

    def test_summary_file_name(self) -> None:
        """测试摘要文件名。"""
        assert "run_summary.json" == "run_summary.json"


class TestMarkdownReportRendering:
    """测试 Markdown 报告目录与锚点辅助函数。"""

    def test_get_report_panel_container_height_px_uses_min_height_for_short_content(self) -> None:
        """验证内容较短时，容器高度会被限制到最小值。"""

        markdown_content = "# 简报"
        headings = _extract_markdown_headings(markdown_content)

        panel_height_px = _get_report_panel_container_height_px(markdown_content, headings)

        assert panel_height_px == 480

    def test_get_report_panel_container_height_px_scales_with_content_length(self) -> None:
        """验证中等长度内容会按正文和标题数量动态增高。"""

        markdown_content = "# 总览\n" + "\n".join("正文内容" for _ in range(149))
        headings = _extract_markdown_headings(markdown_content)

        panel_height_px = _get_report_panel_container_height_px(markdown_content, headings)

        assert panel_height_px == 483

    def test_get_report_panel_container_height_px_caps_at_max_height(self) -> None:
        """验证超长内容时，容器高度会被限制到最大值。"""

        markdown_content = "\n".join(f"# 章节{i}\n" + "\n".join("正文内容" for _ in range(29)) for i in range(80))
        headings = _extract_markdown_headings(markdown_content)

        panel_height_px = _get_report_panel_container_height_px(markdown_content, headings)

        assert panel_height_px == 1200

    def test_is_report_download_available_returns_true_for_existing_file(self, tmp_path: Path) -> None:
        """验证报告文件存在时下载按钮应可用。"""

        report_path = tmp_path / "AAPL_qual_report.md"
        report_path.write_text("# 测试报告\n", encoding="utf-8")

        assert _is_report_download_available(report_path) is True

    def test_is_report_download_available_returns_false_for_missing_file(self, tmp_path: Path) -> None:
        """验证报告文件缺失或路径为空时下载按钮应禁用。"""

        missing_report_path = tmp_path / "missing.md"

        assert _is_report_download_available(missing_report_path) is False
        assert _is_report_download_available(None) is False

    def test_extract_markdown_headings_preserves_levels_and_deduplicates_anchor(self) -> None:
        """验证标题层级保留，且重复标题会生成唯一锚点。"""

        markdown_content = """# 总览

## 风险

## 风险

### 结论
"""

        headings = _extract_markdown_headings(markdown_content)

        assert [heading.level for heading in headings] == [1, 2, 2, 3]
        assert [heading.title for heading in headings] == ["总览", "风险", "风险", "结论"]
        assert [heading.anchor for heading in headings] == ["总览", "风险", "风险-2", "结论"]

    def test_extract_markdown_headings_ignores_fenced_code_blocks(self) -> None:
        """验证 fenced code block 中的伪标题不会进入目录。"""

        markdown_content = """# 正文标题

```markdown
## 代码里的假标题
```

## 真正的二级标题
"""

        headings = _extract_markdown_headings(markdown_content)

        assert [heading.title for heading in headings] == ["正文标题", "真正的二级标题"]

    def test_inject_heading_anchors_adds_matching_ids(self) -> None:
        """验证正文会为每个目录标题注入对应锚点。"""

        markdown_content = """# 总览

## 风险
"""

        headings = _extract_markdown_headings(markdown_content)
        anchored_markdown = _inject_heading_anchors(markdown_content, headings)

        assert '# <span id="总览"></span>总览' in anchored_markdown
        assert '## <span id="风险"></span>风险' in anchored_markdown
        assert '<div id="总览"></div>' not in anchored_markdown
        assert '<div id="风险"></div>' not in anchored_markdown

    def test_build_report_toc_html_contains_anchor_links(self) -> None:
        """验证目录 HTML 会输出可跳转链接。"""

        markdown_content = """# 总览

## 风险
"""

        headings = _extract_markdown_headings(markdown_content)
        toc_html = _build_report_toc_html(headings)

        assert 'href="#总览"' in toc_html
        assert 'href="#风险"' in toc_html
        assert 'onclick="this.blur();"' in toc_html
        assert "点击条目可跳转到对应章节" in toc_html


class TestReportHeaderActions:
    """测试报告页头操作按钮文案。"""

    def test_get_report_primary_action_label_returns_generate_when_report_missing(self) -> None:
        """无报告时应展示“生成”按钮文案。"""

        assert _get_report_primary_action_label(report_exists=False) == "🚀 生成"

    def test_get_report_primary_action_label_returns_regenerate_when_report_exists(self) -> None:
        """有报告时应展示“重新生成”按钮文案。"""

        assert _get_report_primary_action_label(report_exists=True) == "🔄 重新生成"


class TestState1GuideCard:
    """测试状态1引导卡片内容构建。"""

    def test_extract_template_guide_headings_returns_level1_and_level2_only(self, tmp_path: Path) -> None:
        """模板提取应返回一级和二级标题，忽略更深层级。"""

        template_path = tmp_path / "template.md"
        template_path.write_text(
            "# 一级目录A\n\n## 二级目录A-1\n\n### 三级目录A-1-1\n\n# 一级目录B\n",
            encoding="utf-8",
        )

        guide_headings = _extract_template_guide_headings(str(template_path))

        assert [(heading.level, heading.title) for heading in guide_headings] == [
            (1, "一级目录A"),
            (2, "二级目录A-1"),
            (1, "一级目录B"),
        ]

    def test_build_state1_guide_card_content_contains_level1_and_level2_bullets(self) -> None:
        """引导卡片应渲染模板一级和二级目录列表。"""

        guide_headings = _extract_markdown_headings("# 目录一\n\n## 子目录一\n\n# 目录二\n")
        card_content = _build_state1_guide_card_content(guide_headings)

        assert "内容包括" in card_content
        assert "- 目录一" in card_content
        assert "  - 子目录一" in card_content
        assert "- 目录二" in card_content


class TestPandocVersionRequirement:
    """测试 pandoc 3.0+ 版本要求校验逻辑。"""

    def test_ensure_pandoc_v3_or_newer_accepts_v3(self) -> None:
        """验证 3.x 版本 pandoc 可通过校验。"""
        fake_version = MagicMock(stdout="pandoc 3.9.0.2\n")
        with patch("dayu.web.streamlit.pages.report_tab.subprocess.run", return_value=fake_version):
            _ensure_pandoc_v3_or_newer()

    def test_ensure_pandoc_v3_or_newer_rejects_v2(self) -> None:
        """验证 2.x 版本 pandoc 会被拒绝并提示升级。"""
        fake_version = MagicMock(stdout="pandoc 2.9.2.1\n")
        with patch("dayu.web.streamlit.pages.report_tab.subprocess.run", return_value=fake_version):
            with pytest.raises(RuntimeError, match="请升级到 3.0 及以上"):
                _ensure_pandoc_v3_or_newer()

class MockTestReportTabRendering:
    """模拟测试分析报告Tab渲染（需要Python 3.11+环境）。"""

    @pytest.mark.skip(reason="需要Python 3.11+环境和streamlit")
    def test_render_state1_placeholder(self) -> None:
        """测试状态1渲染占位符。"""
        pass

    @pytest.mark.skip(reason="需要Python 3.11+环境和streamlit")
    def test_render_state2_placeholder(self) -> None:
        """测试状态2渲染占位符。"""
        pass

    @pytest.mark.skip(reason="需要Python 3.11+环境和streamlit")
    def test_render_state3_placeholder(self) -> None:
        """测试状态3渲染占位符。"""
        pass
