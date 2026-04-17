"""ingestion_tools 工具契约测试。"""

from __future__ import annotations

from typing import Any

import pytest

from dayu.engine.tool_registry import ToolRegistry
from dayu.fins.resolver.market_resolver import MarketProfile
from dayu.fins.tools import register_fins_ingestion_tools


class _StubRepository:
    """工具注册用仓储桩。"""


class _StubProcessorRegistry:
    """工具注册用处理器注册表桩。"""


class _FakeJobManager:
    """长事务 job 管理器桩。"""

    def __init__(self) -> None:
        """初始化固定快照。"""

        self.start_download_calls: list[dict[str, Any]] = []
        self.start_process_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[str] = []
        self.snapshots = {
            "job_download_1": {
                "job": {
                    "job_id": "job_download_1",
                    "job_type": "download",
                    "ticker": "AAPL",
                    "status": "running",
                    "stage": "downloading_filings",
                    "created_at": "2026-03-11T00:00:00+00:00",
                    "started_at": "2026-03-11T00:00:01+00:00",
                    "finished_at": None,
                },
                "progress": {"unit": "filing", "completed": 1, "total": 3, "percent": 33},
                "result_summary": None,
                "failure": None,
                "recent_issues": [
                    {
                        "document_id": "fil_1",
                        "status": "skipped",
                        "reason_code": "not_modified",
                        "reason_message": "所有文件均未修改，跳过重新下载",
                    }
                ],
            },
            "job_process_done": {
                "job": {
                    "job_id": "job_process_done",
                    "job_type": "process",
                    "ticker": "AAPL",
                    "status": "succeeded",
                    "stage": "finalizing",
                    "created_at": "2026-03-11T00:00:00+00:00",
                    "started_at": "2026-03-11T00:00:01+00:00",
                    "finished_at": "2026-03-11T00:00:05+00:00",
                },
                "progress": {"unit": "document", "completed": 2, "total": 2, "percent": 100},
                "result_summary": {
                    "filings_total": 2,
                    "filings_processed": 1,
                    "filings_skipped": 1,
                    "filings_failed": 0,
                    "materials_total": 0,
                    "materials_processed": 0,
                    "materials_skipped": 0,
                    "materials_failed": 0,
                },
                "failure": None,
                "recent_issues": [],
            },
            "job_failed": {
                "job": {
                    "job_id": "job_failed",
                    "job_type": "download",
                    "ticker": "AAPL",
                    "status": "failed",
                    "stage": "finalizing",
                    "created_at": "2026-03-11T00:00:00+00:00",
                    "started_at": "2026-03-11T00:00:01+00:00",
                    "finished_at": "2026-03-11T00:00:05+00:00",
                },
                "progress": {"unit": "filing", "completed": 1, "total": 3, "percent": 33},
                "result_summary": {"filings_total": 3, "filings_completed": 1, "filings_failed": 2, "files_downloaded": 2},
                "failure": {"code": "execution_error", "message": "boom", "retryable": True},
                "recent_issues": [
                    {
                        "document_id": "fil_2",
                        "status": "failed",
                        "reason_code": "file_download_failed",
                        "reason_message": "network down",
                    }
                ],
            },
        }

    def start_download_job(
        self,
        *,
        ticker: str,
        form_types: Any,
        filed_date_from: Any,
        filed_date_to: Any,
        overwrite: bool,
    ) -> tuple[str, dict[str, Any]]:
        """记录下载启动参数并返回固定运行中快照。"""

        self.start_download_calls.append(
            {
                "ticker": ticker,
                "form_types": form_types,
                "filed_date_from": filed_date_from,
                "filed_date_to": filed_date_to,
                "overwrite": overwrite,
            }
        )
        return "started", self.snapshots["job_download_1"]

    def start_process_job(
        self,
        *,
        ticker: str,
        overwrite: bool,
        document_ids: Any = None,
    ) -> tuple[str, dict[str, Any]]:
        """记录预处理启动参数并返回固定终态快照。"""

        self.start_process_calls.append(
            {"ticker": ticker, "overwrite": overwrite, "document_ids": document_ids}
        )
        return "reused_active_job", self.snapshots["job_process_done"]

    def get_job_snapshot(self, job_id: str) -> dict[str, Any] | None:
        """按 job_id 返回快照。"""

        return self.snapshots.get(job_id)

    def cancel_job(self, job_id: str) -> tuple[str, dict[str, Any] | None]:
        """记录取消请求并返回固定结果。"""

        self.cancel_calls.append(job_id)
        if job_id == "missing":
            return "not_found", None
        return "already_terminal", self.snapshots["job_failed"]


@pytest.mark.unit
def test_ingestion_tool_schema_hides_internal_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证长事务工具 schema 不暴露内部开关和文档级过滤参数。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    download_schema = registry.schemas["start_financial_filing_download_job"]["function"]["parameters"]["properties"]

    assert "rebuild" not in download_schema
    assert "start_financial_document_preprocess_job" not in registry.schemas


@pytest.mark.unit
def test_ingestion_tool_schema_descriptions_follow_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 ingestion 工具 schema 文案按工作流解释参数来源与后续动作。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    download_schema = registry.schemas["start_financial_filing_download_job"]["function"]
    status_schema = registry.schemas["get_financial_filing_download_job_status"]["function"]
    cancel_schema = registry.schemas["cancel_financial_filing_download_job"]["function"]

    assert "最自然的写法" in download_schema["parameters"]["properties"]["ticker"]["description"]
    assert "只在你明确要限制时间范围时填写" in download_schema["parameters"]["properties"]["filed_date_from"]["description"]
    assert "直接使用启动工具返回的 job.job_id" in status_schema["parameters"]["properties"]["job_id"]["description"]
    assert "下一步只用状态工具轮询" in download_schema["description"]
    assert "优先按 next_step.action 决定" in status_schema["description"]
    assert "取消不是立即完成的" in cancel_schema["description"]


@pytest.mark.unit
def test_start_download_job_tool_returns_low_cognitive_load_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证下载启动工具返回最小决策信息。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    response = _execute_tool(
        registry,
        "start_financial_filing_download_job",
        {
            "ticker": " aapl ",
            "form_types": ["10-Q", "10-K"],
            "filed_date_from": "2024-01-01",
            "overwrite": True,
        },
    )

    assert response["request_outcome"] == "started"
    assert response["job"]["job_type"] == "filing_download"
    assert response["job"]["status"] == "running"
    assert response["failure"] is None
    assert response["next_step"]["action"] == "poll_status"
    assert response["next_step"]["tool_name"] == "get_financial_filing_download_job_status"
    assert manager.start_download_calls[0]["ticker"] == "aapl"
    assert manager.start_download_calls[0]["form_types"] == ["10-K", "10-Q"]


@pytest.mark.unit
def test_status_tool_distinguishes_job_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证状态工具会显式返回 `job_not_found`。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    response = _execute_tool(
        registry,
        "get_financial_filing_download_job_status",
        {"job_id": "missing"},
    )

    assert response["job"] is None
    assert response["recent_issues"] is None
    assert response["failure"]["code"] == "job_not_found"
    assert response["next_step"]["action"] == "stop"


@pytest.mark.unit
def test_start_download_job_tool_returns_not_implemented_for_non_us_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证非 US ticker 不创建 job，直接返回 `not_implemented`。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    from dayu.fins.tools import ingestion_tools as module

    monkeypatch.setattr(
        module.MarketResolver,
        "resolve",
        lambda ticker: MarketProfile(ticker=ticker.upper(), market="CN"),
    )

    response = _execute_tool(
        registry,
        "start_financial_filing_download_job",
        {"ticker": "000333"},
    )

    assert response["request_outcome"] == "not_implemented"
    assert response["job"] is None
    assert response["failure"]["code"] == "not_implemented"
    assert "当前市场暂不支持下载任务" in response["failure"]["message"]
    assert response["next_step"]["action"] == "stop"
    assert manager.start_download_calls == []


@pytest.mark.unit
def test_cancel_and_download_status_tools_return_machine_actionable_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证取消与下载状态返回机器可判别字段。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    cancel_response = _execute_tool(
        registry,
        "cancel_financial_filing_download_job",
        {"job_id": "job_failed"},
    )
    download_status_response = _execute_tool(
        registry,
        "get_financial_filing_download_job_status",
        {"job_id": "job_download_1"},
    )

    assert cancel_response["cancellation_outcome"] == "already_terminal"
    assert cancel_response["failure"]["code"] == "execution_error"
    assert cancel_response["recent_issues"][0]["reason_code"] == "file_download_failed"
    assert cancel_response["next_step"]["action"] == "stop_or_retry"
    assert cancel_response["next_step"]["tool_name"] == "start_financial_filing_download_job"
    assert download_status_response["recent_issues"][0]["reason_code"] == "not_modified"
    assert download_status_response["next_step"]["action"] == "poll_status"
    assert download_status_response["next_step"]["tool_name"] == "get_financial_filing_download_job_status"


@pytest.mark.unit
def test_status_tools_register_polling_dup_call_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证轮询型 status 工具会注册 DupCallSpec。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    download_spec = registry.get_dup_call_spec("get_financial_filing_download_job_status")
    start_spec = registry.get_dup_call_spec("start_financial_filing_download_job")

    assert download_spec is not None
    assert download_spec.mode == "poll_until_terminal"
    assert download_spec.status_path == "job.status"
    assert download_spec.terminal_values == ["succeeded", "failed", "cancelled"]
    assert start_spec is None


def _register_tools(
    *,
    monkeypatch: pytest.MonkeyPatch,
    manager: _FakeJobManager,
) -> ToolRegistry:
    """注册带 fake manager 的 ingestion 工具集合。"""

    from dayu.fins.tools import ingestion_tools as module

    monkeypatch.setattr(module, "get_or_create_ingestion_job_manager", lambda **kwargs: manager)

    registry = ToolRegistry()
    register_fins_ingestion_tools(
        registry,
        service_factory=lambda _ticker: None,
        manager_key="test-key",
    )
    return registry


def _execute_tool(registry: ToolRegistry, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """执行工具并提取 JSON 值。"""

    result = registry.execute(name, arguments)
    assert result["ok"] is True
    return result["value"]
