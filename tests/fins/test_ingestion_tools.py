"""ingestion_tools 工具契约测试。"""

# pyright: reportIndexIssue=false, reportArgumentType=false, reportOptionalSubscript=false, reportCallIssue=false
# 原因：工具执行结果为嵌套 JSON 结构，逐层 subscript 产生的 union 类型在 pyright 下
# 无法自动收窄；测试仅做断言，不需要严格的 subscript 类型推导。

from __future__ import annotations

import pytest

from dayu.engine.tool_registry import ToolRegistry
from dayu.fins.ticker_normalization import NormalizedTicker
from dayu.fins.tools import register_fins_ingestion_tools


class _FakeJobManager:
    """长事务 job 管理器桩。"""

    def __init__(self) -> None:
        """初始化固定快照。"""

        self.start_download_calls: list[dict[str, str | list[str] | bool | None]] = []
        self.start_process_calls: list[dict[str, str | bool | list[str] | None]] = []
        self.cancel_calls: list[str] = []
        self.snapshots: dict[str, dict[str, str | int | float | bool | None | list | dict]] = {
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
        form_types: list[str] | None,
        filed_date_from: str | None,
        filed_date_to: str | None,
        overwrite: bool,
    ) -> tuple[str, dict[str, str | int | float | bool | None | list | dict]]:
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
        document_ids: list[str] | None = None,
    ) -> tuple[str, dict[str, str | int | float | bool | None | list | dict]]:
        """记录预处理启动参数并返回固定终态快照。"""

        self.start_process_calls.append(
            {"ticker": ticker, "overwrite": overwrite, "document_ids": document_ids}
        )
        return "reused_active_job", self.snapshots["job_process_done"]

    def get_job_snapshot(self, job_id: str) -> dict[str, str | int | float | bool | None | list | dict] | None:
        """按 job_id 返回快照。"""

        return self.snapshots.get(job_id)

    def cancel_job(self, job_id: str) -> tuple[str, dict[str, str | int | float | bool | None | list | dict] | None]:
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
        module,
        "normalize_ticker",
        lambda ticker: NormalizedTicker(canonical=ticker.upper(), market="CN", exchange=None, raw=ticker),
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

    monkeypatch.setattr(module, "get_or_create_ingestion_job_manager", lambda **_kwargs: manager)

    registry = ToolRegistry()
    register_fins_ingestion_tools(
        registry,
        service_factory=lambda _ticker: None,
        manager_key="test-key",
    )
    return registry


def _execute_tool(registry: ToolRegistry, name: str, arguments: dict[str, str | list[str] | bool]) -> dict[str, str | int | float | bool | None | list | dict]:
    """执行工具并提取 JSON 值。"""

    result = registry.execute(name, arguments)
    assert result["ok"] is True
    value: dict[str, str | int | float | bool | None | list | dict] = result["value"]
    return value


# ── register_ingestion_tools 参数校验 ──


@pytest.mark.unit
def test_register_ingestion_tools_raises_when_registry_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 registry 为 None 时抛出 ValueError。"""

    from dayu.fins.tools import ingestion_tools as module

    monkeypatch.setattr(module, "get_or_create_ingestion_job_manager", lambda **_kwargs: None)

    with pytest.raises(ValueError, match="registry 不能为空"):
        module.register_ingestion_tools(None, service_factory=lambda _t: None, manager_key="k")  # type: ignore[arg-type]


@pytest.mark.unit
def test_register_ingestion_tools_raises_when_service_factory_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 service_factory 为 None 时抛出 ValueError。"""

    from dayu.fins.tools import ingestion_tools as module

    monkeypatch.setattr(module, "get_or_create_ingestion_job_manager", lambda **_kwargs: None)

    with pytest.raises(ValueError, match="service_factory 不能为空"):
        module.register_ingestion_tools(ToolRegistry(), service_factory=None, manager_key="k")  # type: ignore[arg-type]


@pytest.mark.unit
def test_register_ingestion_tools_raises_when_manager_key_is_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 manager_key 为空白时抛出 ValueError。"""

    from dayu.fins.tools import ingestion_tools as module

    monkeypatch.setattr(module, "get_or_create_ingestion_job_manager", lambda **_kwargs: None)

    with pytest.raises(ValueError, match="manager_key 不能为空"):
        module.register_ingestion_tools(ToolRegistry(), service_factory=lambda _t: None, manager_key="  ")  # type: ignore[arg-type]


# ── require_non_empty_text 空值 ──


@pytest.mark.unit
def test_require_non_empty_text_raises_on_empty() -> None:
    """验证空值触发 ToolArgumentError。"""

    from dayu.engine.exceptions import ToolArgumentError
    from dayu.fins._converters import require_non_empty_text

    with pytest.raises(ToolArgumentError):
        require_non_empty_text("", empty_error=ToolArgumentError("t", "a", "", "不能为空"))


@pytest.mark.unit
def test_require_non_empty_text_raises_on_none() -> None:
    """验证 None 值触发 ToolArgumentError。"""

    from dayu.engine.exceptions import ToolArgumentError
    from dayu.fins._converters import require_non_empty_text

    with pytest.raises(ToolArgumentError):
        require_non_empty_text(None, empty_error=ToolArgumentError("t", "a", None, "不能为空"))


# ── _normalize_form_types 分支 ──


@pytest.mark.unit
def test_normalize_form_types_returns_none_for_none_input() -> None:
    """验证 None 输入返回 None。"""

    from dayu.fins.tools.ingestion_tools import _normalize_form_types

    assert _normalize_form_types(None) is None


@pytest.mark.unit
def test_normalize_form_types_raises_on_blank_item() -> None:
    """验证表单数组包含空白项时抛出 ToolArgumentError。"""

    from dayu.engine.exceptions import ToolArgumentError
    from dayu.fins.tools.ingestion_tools import _normalize_form_types

    with pytest.raises(ToolArgumentError, match="不能包含空白表单类型"):
        _normalize_form_types(["10-K", "  "])


@pytest.mark.unit
def test_normalize_form_types_returns_none_for_empty_list() -> None:
    """验证空列表返回 None。"""

    from dayu.fins.tools.ingestion_tools import _normalize_form_types

    assert _normalize_form_types([]) is None


@pytest.mark.unit
def test_normalize_form_types_deduplicates_and_sorts() -> None:
    """验证去重与排序。"""

    from dayu.fins.tools.ingestion_tools import _normalize_form_types

    assert _normalize_form_types(["10-Q", "10-K", "10-K"]) == ["10-K", "10-Q"]


# ── process job 工具（start / status / cancel）──


@pytest.mark.unit
def test_start_process_job_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证预处理启动工具正常流程。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    from dayu.fins.tools import ingestion_tools as module

    name, func, schema = module._create_start_process_job_tool(registry=registry, manager=manager)  # type: ignore[arg-type]
    registry.register(name, func, schema)

    response = _execute_tool(registry, "start_financial_document_preprocess_job", {"ticker": "AAPL"})

    assert response["request_outcome"] == "reused_active_job"
    assert response["job"]["job_type"] == "document_preprocess"
    assert response["next_step"]["action"] == "stop"


@pytest.mark.unit
def test_get_process_job_status_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证预处理状态查询工具。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    from dayu.fins.tools import ingestion_tools as module

    name, func, schema = module._create_get_process_job_status_tool(registry=registry, manager=manager)  # type: ignore[arg-type]  # type: ignore[arg-type]
    registry.register(name, func, schema)

    response = _execute_tool(registry, "get_financial_document_preprocess_job_status", {"job_id": "job_process_done"})

    assert response["job"]["job_type"] == "document_preprocess"
    assert response["job"]["status"] == "succeeded"
    assert response["next_step"]["action"] == "stop"


@pytest.mark.unit
def test_get_process_job_status_tool_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证预处理状态查询工具处理 job_not_found。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    from dayu.fins.tools import ingestion_tools as module

    name, func, schema = module._create_get_process_job_status_tool(registry=registry, manager=manager)  # type: ignore[arg-type]
    registry.register(name, func, schema)

    response = _execute_tool(registry, "get_financial_document_preprocess_job_status", {"job_id": "missing"})

    assert response["job"] is None
    assert response["failure"]["code"] == "job_not_found"


@pytest.mark.unit
def test_cancel_process_job_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证预处理取消工具正常流程。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    from dayu.fins.tools import ingestion_tools as module

    name, func, schema = module._create_cancel_process_job_tool(registry=registry, manager=manager)  # type: ignore[arg-type]
    registry.register(name, func, schema)

    response = _execute_tool(registry, "cancel_financial_document_preprocess_job", {"job_id": "job_failed"})

    assert response["cancellation_outcome"] == "already_terminal"
    assert response["failure"]["code"] == "execution_error"
    assert response["next_step"]["action"] == "stop_or_retry"


@pytest.mark.unit
def test_cancel_process_job_tool_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证预处理取消工具处理 job_not_found。"""

    manager = _FakeJobManager()
    registry = _register_tools(monkeypatch=monkeypatch, manager=manager)

    from dayu.fins.tools import ingestion_tools as module

    name, func, schema = module._create_cancel_process_job_tool(registry=registry, manager=manager)  # type: ignore[arg-type]
    registry.register(name, func, schema)

    response = _execute_tool(registry, "cancel_financial_document_preprocess_job", {"job_id": "missing"})

    assert response["cancellation_outcome"] == "job_not_found"
    assert response["job"] is None
    assert response["failure"]["code"] == "job_not_found"


# ── _build_cancel_response snapshot 为 None ──


@pytest.mark.unit
def test_build_cancel_response_snapshot_is_none() -> None:
    """验证 snapshot 为 None 时 cancellation_outcome 映射为 job_not_found。"""

    from dayu.fins.tools.ingestion_tools import _build_cancel_response

    result = _build_cancel_response(
        cancellation_outcome="some_outcome",
        snapshot=None,
        requested_job_id="x",
        status_tool_name="s",
        start_tool_name="st",
    )
    assert result["cancellation_outcome"] == "job_not_found"
    assert result["job"] is None
    assert result["failure"]["code"] == "job_not_found"


# ── _build_public_job 缺失 job 字段 ──


@pytest.mark.unit
def test_build_public_job_raises_when_job_missing() -> None:
    """验证 snapshot.job 缺失时抛出 ValueError。"""

    from dayu.fins.tools.ingestion_tools import _build_public_job

    with pytest.raises(ValueError, match="snapshot.job 缺失"):
        _build_public_job({"progress": {}})


# ── _build_public_progress 非 dict ──


@pytest.mark.unit
def test_build_public_progress_returns_none_for_non_dict() -> None:
    """验证 progress 非 dict 时返回 None。"""

    from dayu.fins.tools.ingestion_tools import _build_public_progress

    assert _build_public_progress({"progress": "not_a_dict"}) is None
    assert _build_public_progress({}) is None


# ── _map_public_job_type 映射 ──


@pytest.mark.unit
def test_map_public_job_type_process() -> None:
    """验证 process 类型映射为 document_preprocess。"""

    from dayu.fins.tools.ingestion_tools import _map_public_job_type

    assert _map_public_job_type("process") == "document_preprocess"


@pytest.mark.unit
def test_map_public_job_type_unknown() -> None:
    """验证未知类型映射为 unknown。"""

    from dayu.fins.tools.ingestion_tools import _map_public_job_type

    assert _map_public_job_type("something_else") == "unknown"
    assert _map_public_job_type(None) == "unknown"


# ── _has_failed_units ──


@pytest.mark.unit
def test_has_failed_units_true() -> None:
    """验证存在失败计数时返回 True。"""

    from dayu.fins.tools.ingestion_tools import _has_failed_units

    assert _has_failed_units({"filings_failed": 2, "filings_total": 5}) is True


@pytest.mark.unit
def test_has_failed_units_false_when_zero() -> None:
    """验证失败计数为 0 时返回 False。"""

    from dayu.fins.tools.ingestion_tools import _has_failed_units

    assert _has_failed_units({"filings_failed": 0}) is False


@pytest.mark.unit
def test_has_failed_units_false_for_non_dict() -> None:
    """验证非 dict 输入返回 False。"""

    from dayu.fins.tools.ingestion_tools import _has_failed_units

    assert _has_failed_units(None) is False
    assert _has_failed_units("string") is False


@pytest.mark.unit
def test_has_failed_units_skips_non_int_values() -> None:
    """验证非整数值被跳过。"""

    from dayu.fins.tools.ingestion_tools import _has_failed_units

    assert _has_failed_units({"materials_failed": "abc"}) is False


# ── _build_next_step _has_failed_units 分支 ──


@pytest.mark.unit
def test_build_next_step_stop_or_retry_when_succeeded_with_failed_units() -> None:
    """验证 succeeded 状态但有失败单元时返回 stop_or_retry。"""

    from dayu.fins.tools.ingestion_tools import _build_next_step

    snapshot = {
        "job": {"job_id": "j1", "job_type": "download", "ticker": "AAPL", "status": "succeeded", "stage": "done", "created_at": None, "started_at": None, "finished_at": None},
        "result_summary": {"filings_failed": 1},
    }
    result = _build_next_step(snapshot=snapshot, status_tool_name="get_status", start_tool_name="start_job")

    assert result["action"] == "stop_or_retry"
    assert result["tool_name"] == "start_job"


@pytest.mark.unit
def test_build_next_step_stop_when_succeeded_no_failed_units() -> None:
    """验证 succeeded 状态且无失败单元时返回 stop。"""

    from dayu.fins.tools.ingestion_tools import _build_next_step

    snapshot = {
        "job": {"job_id": "j1", "job_type": "download", "ticker": "AAPL", "status": "succeeded", "stage": "done", "created_at": None, "started_at": None, "finished_at": None},
        "result_summary": {"filings_failed": 0},
    }
    result = _build_next_step(snapshot=snapshot, status_tool_name="get_status", start_tool_name="start_job")

    assert result["action"] == "stop"


@pytest.mark.unit
def test_build_next_step_poll_status_when_running() -> None:
    """验证 running 状态返回 poll_status。"""

    from dayu.fins.tools.ingestion_tools import _build_next_step

    snapshot = {
        "job": {"job_id": "j1", "job_type": "download", "ticker": "AAPL", "status": "running", "stage": "downloading", "created_at": None, "started_at": None, "finished_at": None},
        "result_summary": None,
    }
    result = _build_next_step(snapshot=snapshot, status_tool_name="get_status", start_tool_name="start_job")

    assert result["action"] == "poll_status"
    assert result["tool_name"] == "get_status"


@pytest.mark.unit
def test_build_next_step_stop_or_retry_when_failed() -> None:
    """验证 failed 状态返回 stop_or_retry。"""

    from dayu.fins.tools.ingestion_tools import _build_next_step

    snapshot = {
        "job": {"job_id": "j1", "job_type": "download", "ticker": "AAPL", "status": "failed", "stage": "done", "created_at": None, "started_at": None, "finished_at": None},
        "result_summary": None,
    }
    result = _build_next_step(snapshot=snapshot, status_tool_name="get_status", start_tool_name="start_job")

    assert result["action"] == "stop_or_retry"
    assert result["tool_name"] == "start_job"


# ── _normalize_optional_text ──


@pytest.mark.unit
def test_normalize_optional_text_none() -> None:
    """验证 None 返回 None。"""

    from dayu.fins._converters import normalize_optional_text

    assert normalize_optional_text(None) is None


@pytest.mark.unit
def test_normalize_optional_text_blank() -> None:
    """验证空白字符串返回 None。"""

    from dayu.fins._converters import normalize_optional_text

    assert normalize_optional_text("  ") is None


@pytest.mark.unit
def test_normalize_optional_text_strips() -> None:
    """验证去空白。"""

    from dayu.fins._converters import normalize_optional_text

    assert normalize_optional_text("  hello  ") == "hello"


@pytest.mark.unit
def test_require_non_empty_text_preserves_falsy_scalars() -> None:
    """验证必填文本收口不会把 0/False 误判为空。"""

    from dayu.fins._converters import require_non_empty_text

    assert require_non_empty_text(0, empty_error=ValueError("bad")) == "0"
    assert require_non_empty_text(False, empty_error=ValueError("bad")) == "False"
