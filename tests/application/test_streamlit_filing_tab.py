"""filing_tab 纯函数测试。

仅测试不依赖 st.session_state 与 Streamlit 渲染的纯逻辑函数。
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from dayu.contracts.fins import (
    DownloadCommandPayload,
    FinsCommandName,
)
from dayu.fins.domain.document_models import FilingSummary
from dayu.services.contracts import FinsSubmitRequest
from dayu.web.streamlit.pages.filing.download_panel import (
    _DownloadFormValues,
    _build_download_log_lines,
    _build_download_submit_request,
    _format_log_time,
)
from dayu.web.streamlit.pages.filing.download_progress import LogEntry
from dayu.web.streamlit.pages.filing_tab import (
    _calculate_dataframe_height,
    _get_filing_list,
)


# ── _DownloadFormValues ───────────────────────────────────────────────


@pytest.mark.unit
def test_download_form_values_creation() -> None:
    """_DownloadFormValues 应正确存储所有字段。"""
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2024, 12, 31)
    form = _DownloadFormValues(
        form_types=("10-K", "10-Q"),
        start_date=start,
        end_date=end,
        overwrite=True,
    )
    assert form.form_types == ("10-K", "10-Q")
    assert form.start_date == start
    assert form.end_date == end
    assert form.overwrite is True


@pytest.mark.unit
def test_download_form_values_defaults() -> None:
    """_DownloadFormValues 空日期 None 值应保留。"""
    form = _DownloadFormValues(
        form_types=(),
        start_date=None,
        end_date=None,
        overwrite=False,
    )
    assert form.start_date is None
    assert form.end_date is None
    assert form.overwrite is False


@pytest.mark.unit
def test_download_form_values_frozen() -> None:
    """_DownloadFormValues 应为不可变数据类。"""
    form = _DownloadFormValues(
        form_types=("10-K",),
        start_date=None,
        end_date=None,
        overwrite=False,
    )
    with pytest.raises(Exception):
        form.overwrite = True  # type: ignore[misc]


# ── _calculate_dataframe_height ───────────────────────────────────────


@pytest.mark.unit
def test_calculate_dataframe_height_one_row() -> None:
    """1 行数据应返回表头高 + 1 行行高。"""
    result = _calculate_dataframe_height(1)
    assert result == 38 + 35  # _DATAFRAME_HEADER_HEIGHT_PX + _DATAFRAME_ROW_HEIGHT_PX


@pytest.mark.unit
def test_calculate_dataframe_height_multiple_rows() -> None:
    """多行数据应正确计算总高度。"""
    result = _calculate_dataframe_height(5)
    assert result == 38 + 5 * 35


@pytest.mark.unit
def test_calculate_dataframe_height_zero_raises() -> None:
    """visible_rows=0 应抛出 ValueError。"""
    with pytest.raises(ValueError, match="visible_rows"):
        _calculate_dataframe_height(0)


@pytest.mark.unit
def test_calculate_dataframe_height_negative_raises() -> None:
    """visible_rows 负数应抛出 ValueError。"""
    with pytest.raises(ValueError, match="visible_rows"):
        _calculate_dataframe_height(-3)


# ── _format_log_time ──────────────────────────────────────────────────


@pytest.mark.unit
def test_format_log_time_valid_iso() -> None:
    """合法 ISO 时间戳应返回 HH:MM:SS 格式。"""
    result = _format_log_time("2026-01-15T14:30:45")
    assert result == "14:30:45"


@pytest.mark.unit
def test_format_log_time_empty_string() -> None:
    """空字符串应返回空字符串。"""
    result = _format_log_time("")
    assert result == ""


@pytest.mark.unit
def test_format_log_time_invalid_format() -> None:
    """非法时间戳格式应返回原字符串。"""
    result = _format_log_time("invalid-timestamp")
    assert result == "invalid-timestamp"


@pytest.mark.unit
def test_format_log_time_with_microseconds() -> None:
    """带微秒的 ISO 时间戳也应正确格式化。"""
    result = _format_log_time("2026-01-15T14:30:45.123456")
    assert result == "14:30:45"


# ── _build_download_log_lines ─────────────────────────────────────────


@pytest.mark.unit
def test_build_log_lines_empty() -> None:
    """空日志列表应返回空列表。"""
    result = _build_download_log_lines([])
    assert result == []


@pytest.mark.unit
def test_build_log_lines_single_entry() -> None:
    """单条 info 日志应格式化为 [时间] INFO 消息。"""
    logs: list[LogEntry] = [
        {"timestamp": "2026-01-15T10:00:00", "message": "开始下载", "level": "info"},
    ]
    result = _build_download_log_lines(logs)
    assert len(result) == 1
    assert "[10:00:00] INFO 开始下载" in result[0]


@pytest.mark.unit
def test_build_log_lines_error_level() -> None:
    """error 级别日志应显示 ERROR 标签。"""
    logs: list[LogEntry] = [
        {"timestamp": "2026-01-15T10:00:01", "message": "下载失败", "level": "error"},
    ]
    result = _build_download_log_lines(logs)
    assert "ERROR" in result[0]


@pytest.mark.unit
def test_build_log_lines_warning_level() -> None:
    """warning 级别日志应显示 WARN 标签。"""
    logs: list[LogEntry] = [
        {"timestamp": "2026-01-15T10:00:02", "message": "文件跳过", "level": "warning"},
    ]
    result = _build_download_log_lines(logs)
    assert "WARN" in result[0]


@pytest.mark.unit
def test_build_log_lines_no_timestamp() -> None:
    """无时间戳时应省略时间前缀。"""
    logs: list[LogEntry] = [
        {"timestamp": "", "message": "启动完成", "level": "info"},
    ]
    result = _build_download_log_lines(logs)
    assert result[0] == "INFO 启动完成"


@pytest.mark.unit
def test_build_log_lines_truncates_to_max() -> None:
    """日志超过 120 条时应截断，仅保留最近 120 条。"""
    logs: list[LogEntry] = [
        {"timestamp": "2026-01-15T10:00:00", "message": f"msg-{i}", "level": "info"}
        for i in range(200)
    ]
    result = _build_download_log_lines(logs)
    assert len(result) == 120
    # 最早的条目被截断
    assert "msg-0" not in "\n".join(result)
    # 最近的条目保留
    assert "msg-199" in result[-1]


@pytest.mark.unit
def test_build_log_lines_unknown_level() -> None:
    """未知日志级别默认显示 INFO。"""
    logs: list[LogEntry] = [
        {"timestamp": "2026-01-15T10:00:00", "message": "未知级别消息", "level": "debug"},
    ]
    result = _build_download_log_lines(logs)
    assert "INFO" in result[0]


# ── _build_download_submit_request ────────────────────────────────────


@pytest.mark.unit
def test_build_download_submit_request_full() -> None:
    """有日期和覆盖参数时应完整构建请求。"""
    start = datetime.date(2023, 1, 1)
    end = datetime.date(2024, 12, 31)
    form_values = _DownloadFormValues(
        form_types=("10-K", "10-Q"),
        start_date=start,
        end_date=end,
        overwrite=True,
    )
    result = _build_download_submit_request("AAPL", form_values)
    assert isinstance(result, FinsSubmitRequest)
    assert result.command.name == FinsCommandName.DOWNLOAD
    assert result.command.stream is True
    payload = result.command.payload
    assert isinstance(payload, DownloadCommandPayload)
    assert payload.ticker == "AAPL"
    assert payload.form_type == ("10-K", "10-Q")
    assert payload.start_date == "2023-01-01"
    assert payload.end_date == "2024-12-31"
    assert payload.overwrite is True


@pytest.mark.unit
def test_build_download_submit_request_no_dates() -> None:
    """无日期时应为 None 值。"""
    form_values = _DownloadFormValues(
        form_types=("10-K",),
        start_date=None,
        end_date=None,
        overwrite=False,
    )
    result = _build_download_submit_request("MSFT", form_values)
    payload = result.command.payload
    assert isinstance(payload, DownloadCommandPayload)
    assert payload.start_date is None
    assert payload.end_date is None


@pytest.mark.unit
def test_build_download_submit_request_empty_form_types() -> None:
    """空的 form_types 元组也应正确传递。"""
    form_values = _DownloadFormValues(
        form_types=(),
        start_date=None,
        end_date=None,
        overwrite=False,
    )
    result = _build_download_submit_request("GOOG", form_values)
    payload = result.command.payload
    assert isinstance(payload, DownloadCommandPayload)
    assert payload.form_type == ()


# ── _get_filing_list ──────────────────────────────────────────────────


class _FakeFinsService:
    """模拟财报服务，返回固定 FilingSummary 列表。"""

    def __init__(self, filings: list[FilingSummary] | None = None) -> None:
        self._filings = filings or []

    def list_filings(self, ticker: str) -> list[FilingSummary]:
        return self._filings


@pytest.mark.unit
def test_get_filing_list_empty() -> None:
    """无财报时应返回空列表。"""
    service = _FakeFinsService([])
    result = _get_filing_list(Path("/workspace"), "AAPL", service)  # type: ignore[arg-type]
    assert result == []


@pytest.mark.unit
def test_get_filing_list_returns_filing_info() -> None:
    """有财报时应返回包含正确字段的 _FilingInfo 列表。"""
    summary = FilingSummary(
        document_id="doc-1",
        form_type="10-K",
        filing_date="2024-02-15",
        report_date="2023-12-31",
        fiscal_year=2023,
        fiscal_period="FY",
        primary_file_name="report.pdf",
        primary_file_path="/workspace/filings/AAPL/doc-1/report.pdf",
    )
    service = _FakeFinsService([summary])
    result = _get_filing_list(Path("/workspace"), "AAPL", service)  # type: ignore[arg-type]
    assert len(result) == 1
    info = result[0]
    assert info["document_id"] == "doc-1"
    assert info["file_name"] == "report.pdf"
    assert info["form_type"] == "10-K"
    assert info["filing_date"] == "2024-02-15"
    assert info["report_date"] == "2023-12-31"
    assert info["fiscal_year"] == "2023"
    assert info["fiscal_period"] == "FY"
    assert info["status"] == "可用"


@pytest.mark.unit
def test_get_filing_list_deleted_item() -> None:
    """已删除财报的状态应显示为'已删除'。"""
    summary = FilingSummary(
        document_id="doc-2",
        form_type="10-Q",
        filing_date="2024-05-01",
        report_date="2024-03-31",
        primary_file_name="q.pdf",
        primary_file_path="/workspace/filings/AAPL/doc-2/q.pdf",
        is_deleted=True,
    )
    service = _FakeFinsService([summary])
    result = _get_filing_list(Path("/workspace"), "AAPL", service)  # type: ignore[arg-type]
    assert len(result) == 1
    assert result[0]["status"] == "已删除"


@pytest.mark.unit
def test_get_filing_list_missing_fields() -> None:
    """字段缺失时应显示'未知'占位。"""
    summary = FilingSummary(document_id="doc-3")
    service = _FakeFinsService([summary])
    result = _get_filing_list(Path("/workspace"), "AAPL", service)  # type: ignore[arg-type]
    info = result[0]
    assert info["file_name"] == "未知"
    assert info["form_type"] == "未知"
    assert info["filing_date"] == "未知"
    assert info["report_date"] == "未知"
    assert info["fiscal_year"] == "未知"
    assert info["fiscal_period"] == "未知"


@pytest.mark.unit
def test_get_filing_list_computes_relative_path() -> None:
    """文件路径应计算为相对于 workspace_root 的相对路径。"""
    workspace = Path("/workspace")
    summary = FilingSummary(
        document_id="doc-4",
        primary_file_name="annual.pdf",
        primary_file_path="/workspace/filings/AAPL/doc-4/annual.pdf",
    )
    service = _FakeFinsService([summary])
    result = _get_filing_list(workspace, "AAPL", service)  # type: ignore[arg-type]
    assert result[0]["file_path"] == "filings/AAPL/doc-4/annual.pdf"


@pytest.mark.unit
def test_get_filing_list_path_outside_workspace() -> None:
    """文件路径不在 workspace_root 下时，保留原始绝对路径。"""
    workspace = Path("/workspace")
    summary = FilingSummary(
        document_id="doc-5",
        primary_file_name="external.pdf",
        primary_file_path="/other/path/external.pdf",
    )
    service = _FakeFinsService([summary])
    result = _get_filing_list(workspace, "AAPL", service)  # type: ignore[arg-type]
    assert result[0]["file_path"] == "/other/path/external.pdf"


@pytest.mark.unit
def test_get_filing_list_none_service_returns_empty() -> None:
    """fins_service 为 None 时应返回空列表。"""
    result = _get_filing_list(Path("/workspace"), "AAPL", None)
    assert result == []


@pytest.mark.unit
def test_get_filing_list_service_raises_oserror() -> None:
    """service 抛出 OSError 时，_get_filing_list 应保持纯查询并上抛异常。"""

    class _ErrorService:
        def list_filings(self, ticker: str) -> list[FilingSummary]:
            raise OSError("磁盘错误")

    with pytest.raises(OSError, match="磁盘错误"):
        _get_filing_list(Path("/workspace"), "AAPL", _ErrorService())  # type: ignore[arg-type]
