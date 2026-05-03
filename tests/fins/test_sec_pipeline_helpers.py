"""SecPipeline 辅助函数测试。"""

from __future__ import annotations

import asyncio
import datetime as dt
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import AsyncMock

import pytest

from dayu.fins.domain.document_models import FileObjectMeta, FilingCreateRequest, SourceHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.downloaders.sec_downloader import DownloaderEvent, RemoteFileDescriptor
from dayu.fins.pipelines.download_events import DownloadEventType
from dayu.fins.pipelines import sec_pipeline
from dayu.fins.pipelines import sec_6k_rules as _sec_6k_rules
from dayu.fins.pipelines import sec_download_diagnostics as _sec_download_diagnostics
from dayu.fins.pipelines import sec_download_event_mapping as _sec_download_event_mapping
from dayu.fins.pipelines import sec_download_state as _sec_download_state
from dayu.fins.pipelines import sec_filing_collection as _sec_filing_collection
from dayu.fins.pipelines import sec_fiscal_fields as _sec_fiscal_fields
from dayu.fins.pipelines import sec_form_utils as _sec_form_utils
from dayu.fins.pipelines import sec_sc13_filtering as _sec_sc13_filtering
from dayu.fins.pipelines import sec_upload_workflow as _sec_upload_workflow
from dayu.fins.pipelines import processed_snapshot_helpers as _processed_snapshot_helpers
from dayu.fins.pipelines.sec_pipeline import SecPipeline
from dayu.fins.processors.registry import build_fins_processor_registry
from tests.fins.storage_testkit import build_fs_storage_test_context, build_storage_core


class _NoFinancialProcessor:
    """不提供财务接口的处理器桩。"""


class _PartialFinancialProcessor:
    """仅返回 partial 财务结果的处理器桩。"""

    def get_financial_statement(self, statement_type: str) -> dict[str, Any]:
        """返回无 XBRL 结果。

        Args:
            statement_type: 报表类型。

        Returns:
            财务结果字典。

        Raises:
            RuntimeError: 生成失败时抛出。
        """

        return {
            "statement_type": statement_type,
            "periods": [],
            "rows": [],
            "currency": None,
            "units": None,
            "data_quality": "partial",
        }


class _MixedFinancialProcessor:
    """混合返回值的处理器桩。"""

    def get_financial_statement(self, statement_type: str) -> Any:
        """返回混合结果用于覆盖分支。

        Args:
            statement_type: 报表类型。

        Returns:
            混合类型结果。

        Raises:
            RuntimeError: 指定分支触发时抛出。
        """

        if statement_type == "income":
            raise RuntimeError("boom")
        if statement_type == "balance_sheet":
            return "invalid"
        return {
            "statement_type": statement_type,
            "periods": [{"period_end": "2024-12-31", "fiscal_year": 2024, "fiscal_period": "FY"}],
            "rows": [{"concept": "assets", "values": {"2024-12-31": 1}}],
            "currency": "USD",
            "units": ["USD"],
            "data_quality": "xbrl",
        }


class _QueryProcessor:
    """提供 query_xbrl_facts 的处理器桩。"""

    def __init__(self, result: Any, raise_error: bool = False) -> None:
        """初始化查询处理器。

        Args:
            result: 预设返回值。
            raise_error: 是否抛出异常。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
        """

        self._result = result
        self._raise_error = raise_error

    def query_xbrl_facts(self, concepts: list[str]) -> Any:
        """返回预设查询结果。

        Args:
            concepts: 概念列表。

        Returns:
            查询结果。

        Raises:
            RuntimeError: 配置为抛错时抛出。
        """

        del concepts
        if self._raise_error:
            raise RuntimeError("query error")
        return self._result


async def _collect_result_or_none(stream: Any) -> dict[str, Any]:
    """包装调用异步结果收集函数。

    Args:
        stream: 事件流。

    Returns:
        结果字典。

    Raises:
        RuntimeError: 事件流无最终结果时抛出。
    """

    return await _sec_upload_workflow.collect_upload_result_from_events(stream, stream_name="test_stream")


@pytest.mark.unit
def test_map_file_status_and_event_payload_mapping() -> None:
    """验证下载文件状态与事件负载映射。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _sec_download_event_mapping.map_file_status_to_event_type("downloaded") == DownloadEventType.FILE_DOWNLOADED
    assert _sec_download_event_mapping.map_file_status_to_event_type("skipped") == DownloadEventType.FILE_SKIPPED
    assert _sec_download_event_mapping.map_file_status_to_event_type("failed") == DownloadEventType.FILE_FAILED
    with pytest.raises(ValueError, match="未知文件状态"):
        _sec_download_event_mapping.map_file_status_to_event_type("unknown")

    downloaded_event = DownloaderEvent(
        event_type="file_downloaded",
        name="a.htm",
        source_url="https://x/a.htm",
        http_etag="e1",
        http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
        http_status=200,
        file_meta=FileObjectMeta(uri="local://a.htm"),
    )
    skipped_event = DownloaderEvent(
        event_type="file_skipped",
        name="b.htm",
        source_url="https://x/b.htm",
        http_etag="e2",
        http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
        http_status=304,
        reason_code="not_modified",
        reason_message="远端文件未修改，跳过重新下载",
    )
    failed_event = DownloaderEvent(
        event_type="file_failed",
        name="c.htm",
        source_url="https://x/c.htm",
        http_etag="e3",
        http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
        http_status=500,
        reason_code="download_error",
        reason_message="network",
        error="network",
    )

    assert _sec_download_event_mapping.build_file_result_from_downloader_event(downloaded_event)["status"] == "downloaded"
    skipped_result = _sec_download_event_mapping.build_file_result_from_downloader_event(skipped_event)
    failed_result = _sec_download_event_mapping.build_file_result_from_downloader_event(failed_event)
    assert skipped_result["status"] == "skipped"
    assert skipped_result["reason_code"] == "not_modified"
    assert failed_result["status"] == "failed"
    assert failed_result["reason_code"] == "download_error"
    assert failed_result["reason_message"] == "network"

    with pytest.raises(ValueError, match="不支持的下载器事件类型"):
        _sec_download_event_mapping.build_file_result_from_downloader_event(
            cast(
                DownloaderEvent,
                SimpleNamespace(
                    event_type="other",
                    name="x",
                    source_url="u",
                    http_etag=None,
                    http_last_modified=None,
                    http_status=None,
                    file_meta=None,
                    error=None,
                ),
            )
        )


@pytest.mark.unit
def test_collect_upload_result_from_events_success_and_failure() -> None:
    """验证上传事件聚合器的成功与失败分支。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    async def _ok_stream():
        """返回包含最终结果的事件流。

        Args:
            无。

        Returns:
            事件流。

        Raises:
            RuntimeError: 迭代失败时抛出。
        """

        yield SimpleNamespace(event_type="upload_started", payload={})
        yield SimpleNamespace(event_type="upload_completed", payload={"result": {"status": "ok"}})

    async def _bad_stream():
        """返回无最终结果的事件流。

        Args:
            无。

        Returns:
            事件流。

        Raises:
            RuntimeError: 迭代失败时抛出。
        """

        yield SimpleNamespace(event_type="upload_started", payload={})

    assert asyncio.run(_collect_result_or_none(_ok_stream()))["status"] == "ok"
    with pytest.raises(RuntimeError, match="未返回最终结果"):
        asyncio.run(_collect_result_or_none(_bad_stream()))


@pytest.mark.unit
def test_async_sync_wrappers_cover_running_loop_branch() -> None:
    """验证同步包装器在有无事件循环时的行为。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    async def _coro_ok() -> dict[str, Any]:
        """返回固定结果。

        Args:
            无。

        Returns:
            结果字典。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        return {"ok": True}

    assert sec_pipeline._run_async_upload_sync(_coro_ok())["ok"] is True

    async def _inside_loop() -> None:
        """在运行中的 loop 内触发错误分支。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        upload_coro = _coro_ok()
        with pytest.raises(RuntimeError, match="stream 异步接口"):
            sec_pipeline._run_async_upload_sync(upload_coro)
        upload_coro.close()

    asyncio.run(_inside_loop())


@pytest.mark.unit
def test_date_version_and_sc13_helpers() -> None:
    """验证日期、版本与 SC13 判定辅助函数。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    leap_day = dt.date(2024, 2, 29)
    assert _sec_form_utils.subtract_years(leap_day, 1) == dt.date(2023, 2, 28)
    assert _sec_form_utils.subtract_years(dt.date(2025, 3, 1), 2) == dt.date(2023, 3, 1)
    with pytest.raises(ValueError, match="years 必须大于 0"):
        _sec_form_utils.subtract_years(dt.date(2025, 1, 1), 0)

    assert _sec_form_utils.increment_document_version("v1") == "v2"
    assert _sec_form_utils.increment_document_version("invalid") == "v2"

    assert _sec_sc13_filtering.should_warn_missing_sc13(
        {"SC 13D": dt.date(2025, 1, 1)},
        [],
    ) is True
    assert _sec_sc13_filtering.should_warn_missing_sc13(
        {"10-K": dt.date(2025, 1, 1)},
        [],
    ) is False


@pytest.mark.unit
def test_6k_selection_and_filenum_index_helpers() -> None:
    """验证 6-K 目标文件与 filenum/文件索引相关辅助函数。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    remote_files = [
        RemoteFileDescriptor(
            name="a.htm",
            source_url="https://x/a.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
        ),
        RemoteFileDescriptor(
            name="q12025pressrelease.htm",
            source_url="https://x/d123dex991.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
            sec_document_type="EX-99.1",
        ),
    ]
    assert _sec_6k_rules._score_6k_filename("d123dex992.htm", "a.htm")[0] == 1
    assert _sec_6k_rules._score_6k_filename("unknown.htm", "a.htm")[0] == 4
    # Edgar Filing Services 格式: ex99-1, ex99_1
    assert _sec_6k_rules._score_6k_filename("tm257183d1_ex99-1.htm", "a.htm")[0] == 0
    assert _sec_6k_rules._score_6k_filename("tm2531299d1_ex99-2.htm", "a.htm")[0] == 1
    assert _sec_6k_rules._score_6k_filename("tm2531299d1_ex99-3.htm", "a.htm")[0] == 1
    assert _sec_6k_rules._score_6k_filename("file_ex99_1.htm", "a.htm")[0] == 0
    assert _sec_6k_rules._score_6k_filename("file_ex994.htm", "a.htm")[0] == 2
    # 优先使用 SEC 文档类型（即使文件名不含 ex99）
    assert (
        _sec_6k_rules._score_6k_filename(
            "q12025pressrelease.htm",
            "a.htm",
            "EX-99.1",
        )[0]
        == 0
    )

    filenums: set[str] = set()
    _sec_filing_collection.collect_filenums_from_table(filenums, {"fileNumber": ["005-10000", "", "005-10000"]})
    assert filenums == {"005-10000"}

    _sec_filing_collection.collect_filenums_from_table(filenums, {"fileNumber": "invalid"})
    assert filenums == {"005-10000"}

    indexed = _sec_download_state._index_file_entries(
        {
            "files": [
                {"name": "a.htm", "uri": "local://x/a.htm"},
                {"uri": "local://x/b.htm"},
                "bad",
            ]
        }
    )
    assert set(indexed.keys()) == {"a.htm", "b.htm"}
    assert _sec_download_state._index_file_entries(None) == {}
    assert _sec_download_state._index_file_entries({"files": "bad"}) == {}

    assert _sec_6k_rules._infer_filename_from_uri("local://AAPL/filings/a.htm") == "a.htm"
    with pytest.raises(ValueError, match="文件列表为空"):
        _sec_6k_rules._select_6k_target_name([], "a.htm")


@pytest.mark.unit
def test_classify_6k_text_and_match_any() -> None:
    """验证 6-K 文本分类规则。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _sec_6k_rules._classify_6k_text("") == "NO_MATCH"
    assert _sec_6k_rules._classify_6k_text("Financial Results and Business Updates") == "RESULTS_RELEASE"
    assert _sec_6k_rules._classify_6k_text("UNAUDITED INTERIM CONDENSED CONSOLIDATED") == "IFRS_RECON"
    assert _sec_6k_rules._classify_6k_text("ANNUAL GENERAL MEETING") == "EXCLUDE_NON_QUARTERLY"
    # BABA 格式: "Announces December Quarter 2024 Results"
    assert _sec_6k_rules._classify_6k_text("Announces December Quarter 2024 Results") == "RESULTS_RELEASE"
    assert (
        _sec_6k_rules._classify_6k_text("Copa Holdings Reports Fourth-Quarter Financial Results")
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text("Copa Holdings Reports Third-Quarter Financial Results")
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Copa Holdings Reports Fourth-Quarter and Full-Year 2025 Financial Results"
        )
        == "RESULTS_RELEASE"
    )
    # BABA 格式: "announced its financial results for the quarter ended"
    assert _sec_6k_rules._classify_6k_text(
        "today announced its financial results for the quarter ended December 31, 2024"
    ) == "RESULTS_RELEASE"
    # BABA 格式: "QUARTER 2025 RESULTS AND"
    assert _sec_6k_rules._classify_6k_text("QUARTER 2025 RESULTS AND") == "RESULTS_RELEASE"
    # 时间段短语必须与财务语义共同出现，避免误命中董事会通知
    assert _sec_6k_rules._classify_6k_text(
        "Financial results for the six months ended September 30, 2025"
    ) == "RESULTS_RELEASE"
    # 通用季报财务报表: "unaudited consolidated results"
    assert _sec_6k_rules._classify_6k_text("unaudited consolidated results of the company") == "IFRS_RECON"
    assert _sec_6k_rules._classify_6k_text(
        "unaudited consolidated financial statements have been prepared"
    ) == "IFRS_RECON"
    assert _sec_6k_rules._classify_6k_text("Announcement from Trip.com Group") == "NO_MATCH"
    assert (
        _sec_6k_rules._classify_6k_text("Announcement from Example Holdings")
        == "NO_MATCH"
    )
    # 预告型新闻稿：应排除
    assert (
        _sec_6k_rules._classify_6k_text(
            "Ascentage Pharma to Report 2025 Six Month Interim Results and provide updates"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Arbe to Announce Q1 2025 Financial Results and Hold a Conference Call on May 20, 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "The Bank has made the below announcement to the stock exchanges: the meeting of the Board of Directors is scheduled to be held on Saturday, April 19, 2025 to, inter alia, consider and approve the audited financial results (standalone and consolidated) for the quarter and year ending March 31, 2025 and recommendation of dividend, if any. The trading window will remain closed."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Sub: Intimation regarding Board meeting and Trading window closure. A meeting of the Board of Directors of the Company will be held on Friday, May 9, 2025, inter alia, to consider and approve the Audited Standalone and Consolidated Financial Results of the Company for the quarter and financial year ending on March 31, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "The Company will host a live conference call on Wednesday, March 5, 2025 at 8:30 a.m. Eastern Time to discuss its financial results for the fourth quarter and year ended December 31, 2024. On February 12, 2025, the Company issued a press release announcing the conference call."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Atour Lifestyle Holdings Limited today announced that it will report its unaudited financial results for the first quarter 2025 on Thursday, May 22, 2025, before the U.S. markets open. The Company will host a conference call at 7:00 AM U.S. Eastern time on the same day."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "GROUP REPORTING CHANGES. In advance of the publication of the 1Q 2025 Earnings Release, the attached data pack presents the impact on the previously reported financial information for each quarter in 2023 and 2024 of the organisational changes to reporting segments."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "NOTICE OF BOARD MEETING for considering and approving the unaudited interim results of the Group for the six months ended June 30, 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "See https://www.merck.com/news/merck-announces-fourth-quarter-and-full-year-2024-financial-results/ for reference"
        )
        == "NO_MATCH"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Interim results from the open-label extension ADHERE+ further build upon the clinical dataset"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Sibanye-Stillwater is pleased to provide an operating update for the quarter ended 31 March 2025. The Group's financial results are only provided on a six-monthly basis."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "2025 Annual Report including the Annual Financial Statements for the year ended December 31, 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "2025 Interim Report Report on Review of Interim Financial Information Interim condensed consolidated statement of comprehensive income These condensed interim financial information have been prepared based on the annual report of the Company dated on April 24, 2025."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "2025 INTERIM REPORT Unaudited Condensed Consolidated Balance Sheets Unaudited Condensed Consolidated Statement of Comprehensive Loss Certain information and note disclosures normally included in the annual financial statements prepared in accordance with U.S. GAAP have been condensed or omitted."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "MINISO Group Announces 2025 June Quarter and Interim Unaudited Financial Results The company also references its annual report to shareholders and other written materials."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "The company hosts quarterly results presentations and investor days for stakeholders"
        )
        == "NO_MATCH"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "A Phase 3 clinical trial is expected to initiate in the third quarter of 2026, following positive results from the Phase 1 trial"
        )
        == "NO_MATCH"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Invitation to the Annual General Meeting 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "2025 Interim Report Unaudited Condensed Consolidated Balance Sheets Record date for the 2025 annual general meeting will be announced separately."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Transcript of the Earnings call conducted on January 23, 2025 for the quarter ended December 31, 2024"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Sub: Audio Recording of the Earnings call conducted on May 9, 2025 for the quarter ended March 31, 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Investor presentation referred during the earnings call with analysts and investors for the financial results of the Bank for the quarter and financial year ended March 31, 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Kyivstar will host an investor meeting and provide an update on second quarter 2025 results"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Tropicana Analyst Visit 2025 includes forward-looking statements regarding financial reports and financial condition"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Interim financial results for the six months ended June 30, 2025"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "GFL Environmental Inc. Sets Date for Q1 2025 Earnings Release and will release its 2025 first quarter financial results after the market closes on Wednesday April 30, 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Copa Holdings Announces Third Quarter Financial Results Release Schedule Earnings Release – Third Quarter 2025 Date: November 19, 2025 Time: After US market close"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Copa Holdings Announces Third Quarter Financial Results Release SchedulePANAMA CITY, October 7, 2025 Earnings Release – Third Quarter 2025 Date:November 19, 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Alvotech Appoints Linda Jonsdottir as Chief Financial Officer following Q2 2025 financial results"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "CAPITALIZATION AND INDEBTEDNESS The following table sets forth our capitalization and indebtedness as of June 30, 2025. You should read this table in conjunction with Management Discussion and Analysis for the six months ended June 30, 2025 included in our Report on Form 6-K."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "INTERIM CONSOLIDATED FINANCIAL STATEMENTS For the periods ending on June 30, 2025 and 2024 and December 31, 2024 INTERIM CONSOLIDATED STATEMENTS OF FINANCIAL POSITION"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Alvotech Q4 2025 and Full Year 2025 Financial Results Financial Highlights Q4 2025 Highlights Total revenues were $173 million"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Q4 2024 Update Table of Contents Key Highlights Financial Summary Outlook Financial Statements Executive Summary"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Financial Report Grupo Financiero Galicia S.A. 2024 4th. quarter This report is a summary analysis of Grupo Galicia's financial condition and results of operations"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "DATE OF AUDIT COMMITTEE MEETING AND 2025 FIRST QUARTER FINANCIAL RESULTS ANNOUNCEMENT DATE"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Hesai Group DATE OF BOARD MEETING The board of directors announces that a meeting of the Board will be held to approve the unaudited financial results for the third quarter of 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Golar LNG Limited – Q2 2025 results presentation Golar LNG's 2nd Quarter 2025 results will be released before the NASDAQ opens on Thursday, August 14, 2025. In connection with this a webcast presentation will be held at 1:00 P.M (London Time)."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "OneConnect Financial Technology Co., Ltd. PROFIT WARNING The board wishes to inform shareholders and potential investors that the group is expected to record a loss for the six months ended June 30, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Sibanye-Stillwater Operating update for the quarter ended 31 March 2025. The Group's financial results are only provided on a six-monthly basis."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Trading Statement Second quarter 2025 trading statement The following Trading Statement provides a summary of current estimates and expectations for the second quarter of 2025. The information presented is not an estimate of those results. Group results for the second quarter 2025 are expected to be published on 5 August 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "TRADING STATEMENT AND OPERATIONAL PERFORMANCE UPDATE FOR THE SIX MONTHS ENDED 30 JUNE 2025 Further detail will be provided as part of the H1 2025 financial and operational results to be released on Friday, 22 August 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Results for Announcement to the Market James Hardie announced today its results for the 3rd quarter and nine months ended 31 December 2024 and has filed the following documents with the ASX: Earnings Release, Management's Analysis of Results, Earnings Presentation and Condensed Consolidated Financial Statements."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Response to ASX Aware Letter. James Hardie Industries plc refers to ASX Compliance's aware letter and provides the attached response."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Announcement on 2025/07/10: Chunghwa Telecom announced its operating results for June 2025. Date of events: 2025/07/10. Contents: June 2025 sales."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "ANNUAL REPORT AND FORM 20-F 2025 & NOTICE OF AGM. AGM statements and voting results will be published after the annual general meeting."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "ArcelorMittal announces updated financial calendar for 2025. The Q1 2025 earnings results initially scheduled on 2 May 2025 will now be published on 30 April 2025. The remainder of the financial calendar for 2025 remains unchanged."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Elbit Systems To Report Second Quarter 2025 Financial Results on August 13, 2025. The Company will host a Conference Call to discuss its financial results on August 13, 2025 at 9:00am ET."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "BHP CEO will present at the Bank of America 2025 Global Metals, Mining & Steel Conference. The presentation slides are attached and a transcript of the presentation will be available shortly after the presentation."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "VEON and Kyivstar to Host Investor Meeting on August 28, 2025. The meeting will provide an update on Kyivstar's strategic initiatives, financial performance, and market outlook."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "CyberArk announced, ahead of its scheduled Investor Day, that the Company is affirming its previously issued Business Outlook that was included in the press release entitled CyberArk Announces Record Fourth Quarter and Full Year 2024 Results."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Preliminary Results for the Three Months Ended March 31, 2025. We have not yet completed our closing procedures for the three months ended March 31, 2025 and presented below are certain estimated preliminary unaudited financial results."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Preliminary Results for the Year Ended December 31, 2025. Ranges have been provided because our unaudited consolidated financial statements for the year ended December 31, 2025 are not yet available and our financial closing procedures are not yet complete."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "James Hardie Announces Preliminary Second Quarter Results and sees improved outlook for siding and trim. The company today announced preliminary expected financial results for the fiscal second quarter ended September 30, 2025. These estimates are preliminary and subject to the Company's detailed quarter-end closing procedures."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "JinkoSolar's Subsidiary Announces Certain Preliminary Unaudited Financial Results for Full Year 2024. Preliminary unaudited financial results included in this press release are unaudited and are subject to change upon the completion of the audit process for full year 2024 and should not be viewed as a substitute for the full financial statements."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "JinkoSolar's Subsidiary Announces Estimates of Certain Preliminary Unaudited Financial Results for Full Year 2025. Preliminary unaudited net loss attributable to shareholders is estimated to be in the range of RMB5,900 million to RMB6,900 million."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Sportradar Announces Preliminary First Quarter 2025 Financial Results. Release of full first quarter results and earnings call to take place on May 12, 2025. The following preliminary unaudited first quarter 2025 results are based on preliminary internal data available as of the date of this announcement, and the Company's independent registered accounting firm has not reviewed or audited the preliminary financial information."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "GRAVITY Reports Preliminary Unaudited 4Q 2025 Results and Business Updates. Tentative consolidated revenue for the fourth quarter of 2025 is KRW 113 billion. 2025 figures are unaudited and subject to revision. Final result for the fourth quarter and year ended December 31, 2025 will be provided by our annual report for the fiscal year ended December 31, 2025 on Form 20-F."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Elbit Systems Reports Second Quarter 2025 Financial Results. Financial Highlights are set forth below and the unaudited condensed consolidated financial statements are included in this release. A replay of the conference call will be available on the company website."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Reports Unaudited First Quarter 2025 Financial Results. Unaudited condensed consolidated financial statements and financial highlights are included below."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "MINUTES OF THE BOARD OF DIRECTORS’ MEETING HELD ON MAY 5TH, 2025 To acknowledge on the Company’s Quarterly Financial Report for the 1st quarter of 2025 and to approve the financial statements."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Shell second quarter 2025 update note Outlooks presented may vary from the actual second quarter 2025 results and are subject to finalisation of those results, which are scheduled to be published on July 31, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "AngloGold Ashanti Operating Statistics for the Three Months and Year Ended 31 December 2024 FULL YEAR 2024 OPERATING STATISTICS Reporting method Non-GAAP financial measures all-in sustaining costs and production."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Golar LNG Limited announces strategic review to maximize stakeholder value and appoints Goldman Sachs as financial advisor. As previewed on the 4th Quarter 2025 results, this review aims to accelerate the FLNG growth pipeline."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Zhihu Inc. Announces Change of Independent Director. The Company announced the appointment of Dr. Li-Lan Cheng as an independent director and the resignation of another independent director."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "DISCLOSEABLE TRANSACTION IN RESPECT OF SUBSCRIPTION IN A PRIVATE CREDIT DIGITAL YIELD FUND"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Next Day Disclosure Return (Equity issuer - changes in issued shares or treasury shares)"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "ADJUSTMENT TO EXERCISE PRICE OF EQUITY LINKED SECURITIES AND CALL SPREAD Further reference is made to the interim cash dividend for the six months ended June 30, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Creating shared value Our strategy Summary and conclusions Agenda Q&A FORWARD LOOKING STATEMENTS These forward-looking statements relate to future financial position and business strategies."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Trading statement and production update for the six months ended 30 June 2025 The Company will publish the Group operating and financial results for the six months ended 30 June 2025 on Thursday, 28 August 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Silicom's first quarter 2026 results release scheduled for April 30, 2026. The Company will be releasing its first quarter 2026 results on that date and will host a conference call."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "KB Financial Group Inc. will hold its 2025 First Quarter Earnings Conference on Thursday, April 24, 2025. A live webcast of the conference and the presentation materials will be available on its IR website."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Notification of Q1 2025 earnings release. The earnings release materials will be posted on the website after the disclosure."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "LG Display Purpose of IR : Attending the Conference in Q3 2025. Summary of key topics to be covered: Q2 2025 financial results and Q&A."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "ICICI Bank wishes to inform that the audio recording of the call with media on the financial results for the quarter ended June 30, 2025 has been uploaded on the website of the Bank. Further, the audio recording of earnings call with analysts and investors has also been uploaded."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Please find attached the investor presentation which will be referred during the earnings call with analysts and investors for the financial results of the Bank for the quarter ended June 30, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Dr. Reddy's Laboratories Ltd. we are enclosing herewith the presentation on the unaudited financial results of the Company for the quarter ended June 30, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Copies of newspaper advertisement regarding audited financial results for the quarter and year ended March 31, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "This Form 6-K consists of the presentation materials related to the second quarter 2025 financial results of UBS and the prepared remarks of Sergio Ermotti for the media conference."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Numbers for slides refer to the third quarter 2025 results presentation. Materials and a webcast replay are available at www.ubs.com/investors. Including analyst Q&A session transcript."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "We will present our 3Q25 results in an interactive meeting, with a Q&A session at the end. Results will be published on the investor relations website on November 4th, after trading hours."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Announcement to the Market 3rd Quarter 2025 Results. The Condensed Financial Statements for the year ended September 30, 2025 are already available on the Investor Relations website. Additionally, we forward the press presentation and the presentation of the interactive meeting on the quarterly results."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Alcon Agrees to Acquire STAAR Surgical. As previously announced, STAAR will release financial results for its second quarter that ended June 27, 2025, on Wednesday, August 6, 2025, after the market close."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Elbit Systems To Report First Quarter 2025 Financial Results on May 20, 2025. The Company will host a conference call to discuss its financial results on May 20, 2025 at 9:00am ET."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Elbit Systems To Report Second Quarter 2025 Financial Results on August 13, 2025. The Company will host a Conference Call to discuss its financial results on August 13, 2025 at 9:00am ET. Haifa, Israel, July 30, 2025 – Elbit Systems Ltd. announced today that it will publish its Second Quarter 2025 financial results on Wednesday, August 13, 2025. Results Conference Call. About Elbit Systems: the company provides advanced defense technology and generated revenues for the three months ended March 31, 2025 in its prior quarterly release."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Shinhan Financial Group 1Q2025 Earnings Release Conference and Blackout Period. Shinhan Financial Group will be holding its 1Q2025 Earnings Release Conference on Friday, April 25, 2025. Investors are welcome to participate during the Q&A session, which will follow the presentation."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Nebius Group announces date of first quarter 2025 results and conference call. Nebius Group will release its first quarter 2025 financial results on Tuesday, May 20, 2025, before market open."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Prior Notice on Disclosure of Final Earnings. Objective: To notice 2025 Q1 earnings disclosure date."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "We will present our 4Q25 results in an interactive meeting, with a Q&A session at the end. Results will be published on the investor relations website on February 4th, after trading hours."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "SK Telecom Co., Ltd. plans to hold a conference call for 2025 1Q earnings results as follows: agenda, conference schedule and Q&A session."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "On May 13, 2025, a webcast will be held to present the company's results for the first quarter of 2025. The event will be presented in Portuguese and will have simultaneous translation into English."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "HUTCHMED (China) Limited Form 6-K Exhibit 99.1 Announcement relating to notice of announcement of 2025 interim results."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "DETAILS OF EARNING CALL. Participant Dial in: To join: +1-888-506-0062. Live webcast: https://example.com/webcast. Results will be published on the investor relations website after trading hours."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Elbit Systems To Report First Quarter 2025 Financial Results on May 20, 2025. The Company will host a conference call to discuss its financial results on May 20, 2025. As of December 31, 2024, the Company reported $6.8 billion in revenues and an order backlog of $22.6 billion."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "The attached material contains Tenaris's announcement regarding Tenaris 2024 Fourth Quarter and Annual Results Conference Call. On 20th February 2025, senior management participated in an audio conference with investors and analysts to discuss the Company's results, market background and outlook."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "3Q25 performance report dates. 3Q25 production and sales report: Date: October 21, 2025. 3Q25 financial performance report: Date: October 30, 2025. Conference call/webcast: Date: October 31, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "The filing of the Company's 2024 Financial Statements, scheduled for February 26, 2025, will be postponed to March 10, 2025, as the work from independent auditors has not yet been completed."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Cemex announced today that it will report its first quarter 2025 results on Monday, April 28, 2025, and host a conference call and a live audio webcast presentation to discuss the results on that same date."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "KT Corporation 2025 Q2 Earnings Release Conference Call. Purpose: 2025 Q2 Earnings Release. Method: Conference Call. Agenda: Earnings results for 2025 Q2, Q&A Session."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Attached are the presentation slides for BHP's HY2026 Results Presentation. The presentation slides and a video of this presentation are available at bhp.com/financial-results."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Enclosure: AngloGold Ashanti Q1 2025 Earnings Release Investor Presentation. Exhibits to Form 6-K Exhibit 99.1 AngloGold Ashanti Q1 2025 Earnings Release Investor Presentation."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "AGA Q4 Dec 2024 ER Investor PresentationUNITED STATESSECURITIES AND EXCHANGE COMMISSION Form 6-K Enclosure: AngloGold Ashanti Q4 2024 Earnings Release Investor PresentationExhibits to Form 6-K Exhibit 99.1 AngloGold Ashanti Q4 2024 Earnings Release Investor Presentation"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "ICL GROUP LTD. 1. Q1 2025 Investor Presentation 2025 First Quarter Financial Results. Important legal notes disclaimer and safe harbor for forward-looking statements."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "AVINO DELIVERS STRONG Q1 2025 PRODUCTION RESULTS; ANNOUNCES DATE OF Q1 EARNINGS CALL The Company’s unaudited condensed consolidated interim financial statements for the First Quarter 2025 will be released after the market closes on Tuesday, May 13, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Exhibit 99.1 Nova Minerals Quarterly Activities and Cashflow Report – 30 June 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "ASML reports €7.7 billion total net sales and €2.4 billion net income in Q1 2025. Today, ASML has published its 2025 first-quarter results. This quarterly update also references the annual report for background information."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "ASUR ANNOUNCES 4Q24 RESULTS Passenger Traffic Declined 0.3%, with Puerto Rico and Colombia Up 9.6% and 14.1%, Respectively while Traffic in Mexico Declined 8.0%"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Grupo Aeroportuario del Centro Norte reported its unaudited, consolidated financial and operating results for the third quarter 2025. 3Q25 summary passenger traffic increased 7.7% and adjusted EBITDA increased 9.0%. OMA will hold its 3Q25 earnings conference call on October 24, 2025."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "1Q 2025 Results April 28, 2025 HIGHLIGHTS Monterrey, Mexico"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Alcon Reports Full-Year 2024 Results, with Strong Top-line and Earnings Growth. Alcon reported its financial results for the three and twelve month periods ending December 31, 2024. For the fourth quarter of 2024, sales were $2.5 billion."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Alcon Delivers Strong Fourth-Quarter 2025 Topline Growth as New Product Launches Accelerate Sales. Alcon reported its financial results for the three and twelve month periods ending December 31, 2025. For the fourth quarter of 2025, sales were $2.7 billion."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "AVINO'S Q3 RESULTS DEMONSTRATE PRODUCTION CONSISTENCY AND ADVANCEMENT OF LA PRECIOSA AHEAD OF SCHEDULE. Overall results continue to support the Company's original production estimate."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "AVINO ANNOUNCES Q2 2025 FINANCIAL RESULTS. Record Revenues of $24.4 million, Record Gross Profit of $10.5 million and Record Cash Flow Generation. The Company will also host a conference call and webcast for investors."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Central Puerto reports its financial results for the fourth quarter 2025 and full year 2025. Quarterly results include adjusted EBITDA, net income and a conference call to discuss the results of this quarter and full year will be held tomorrow."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "PRESS RELEASE Financial Management Review 4Q24 February 2025 Quarterly YTD Report About this Report Basis for Presentation This financial report, which accompanies our quarterly financial statements, has been prepared as requested by the Chilean Financial Market Commission."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Exhibit 99.1 Brookfield Wealth Solutions Ltd.'s interim report for the quarter ended March 31, 2025"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Exhibit 99.1 UNAUDITED CONDENSED CONSOLIDATED FINANCIAL STATEMENTS OF BROOKFIELD WEALTH SOLUTIONS LTD. AS OF JUNE 30, 2025 AND DECEMBER 31, 2024 AND FOR THE THREE AND SIX MONTHS ENDED JUNE 30, 2025 AND 2024"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "bnt-20250630_d2Exhibit 99.1UNAUDITED CONDENSED CONSOLIDATED FINANCIAL STATEMENTS OF BROOKFIELD WEALTH SOLUTIONS LTD. AS OF JUNE 30, 2025 AND DECEMBER 31, 2024AND FOR THE THREE AND SIX MONTHS ENDED JUNE 30, 2025 AND 2024INDEXPage"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "EX-99.1 DocumentExhibit 99.1UNAUDITED CONDENSED CONSOLIDATED FINANCIAL STATEMENTS OF BROOKFIELD WEALTH SOLUTIONS LTD. AS OF SEPTEMBER 30, 2025 AND DECEMBER 31, 2024AND FOR THE THREE AND NINE MONTHS ENDED SEPTEMBER 30, 2025 AND 2024INDEXPage"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "argenx Reports Full Year 2024 Financial Results and Provides Fourth Quarter Business Update February 27, 2025 Amsterdam, the Netherlands argenx today reported financial results for the full year 2024 and provided a fourth quarter business update."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "argenx Reports Half Year 2025 Financial Results and Provides Second Quarter Business Update July 31, 2025 Amsterdam, the Netherlands argenx today announced its half year 2025 results and provided a second quarter business update."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "arm-202506300001973239ARM HOLDINGS PLC /UK03-312025-06-302026Q1false"
            "iso4217:USDiso4217:USDxbrli:sharesxbrli:pure"
            "00019732392025-04-012025-06-300001973239us-gaap:NonrelatedPartyMember"
            "2024-04-012024-06-300001973239"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "au-20250630False30 June 20252025Q2000197383231 Decemberiso4217:USDiso4217:USDxbrli:sharesxbrli:pure00019738322025-01-012025-06-300001973832ifrs-full:RetainedEarningsMember"
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "annual-report00001234562025-03-312025Q4false"
            "iso4217:USDxbrli:shares"
            "00001234562024-04-012025-03-310000123456"
        )
        == "NO_MATCH"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "avino_6k.htm0000316888false--12-312024-12-31FY2024"
            "iso4217:USDifrs-full:Level3OfFairValueHierarchyMember"
            "2024-01-012024-12-310000316888"
        )
        == "NO_MATCH"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "NOTICE AND INFORMATION CIRCULAR NOTICE OF MEETING AND INFORMATION CIRCULAR FOR THE ANNUAL GENERAL MEETING OF SHAREHOLDERS TO BE HELD ON Tuesday, May 27, 2025"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "AVINO SILVER & GOLD MINES LTD. Consolidated Financial Statements For the years ended December 31, 2024 and 2023 Management's Responsibility for Financial Reporting Report of Independent Registered Public Accounting Firm"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Arm Everywhere Investor Conference management provided an update on the Company's business. The Company reaffirmed the Company's fourth quarter 2026 guidance and the presentation materials will be available on the Company's website."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "ANGLOGOLD ASHANTI 2025 MINING FORUM AMERICAS PRESENTATION Certain statements contained in this document are forward-looking statements regarding AngloGold Ashanti's financial reports, operations, economic performance and financial condition."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Bilibili Inc. Announces First Quarter 2025 Financial Results today announced its unaudited financial results for the first quarter ended March 31, 2025. The Company also discusses its Convertible Senior Notes in a later section."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "TRANSACTION IN OWN SHARES GSK plc announces today acting through its corporate stockbroker that it has purchased ordinary shares as part of the Company's existing buyback programme."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Pearson plc Announcement of share repurchase programme Pearson plc announces that it has entered into an engagement in connection with the second and final tranche of its share buyback programme ahead of the closed period prior to its interim results announcement."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Shell announces commencement of a share buyback programme Shell plc today announces the commencement of a $3.5 billion share buyback programme. It is intended that, subject to market conditions, the programme will be completed prior to the Company's Q3 2025 results announcement."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Adjustment to Cash Dividend Per Share TSMC Board of Directors approved to distribute cash dividend for the first quarter of 2025, and the dividend will be paid on October 9, 2025. As a result of the reclamation of shares, the number of common shares outstanding has changed slightly."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "ITAU UNIBANCO HOLDING S.A. Stockholder Remuneration Policy (Dividends and Interest on Capital) This Policy aims to establish the guidelines for determining the remuneration to the shareholders of Itaú Unibanco."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Banco Santander, S.A. hereby communicates the following inside information. The board approved the payment of an interim cash dividend against 2025 results of 11.5 euro cents per share. This distribution, together with the on-going share buyback programme, represents the interim shareholder remuneration."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "CERTIFIED TRUE COPY OF THE RESOLUTION PASSED BY THE BOARD OF DIRECTORS OF INFOSYS LIMITED Subject: To consider and approve Proposal for buyback of equity shares RESOLVED THAT the Company may proceed with the buyback."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "PUBLIC ANNOUNCEMENT POST-BUYBACK This post buyback public announcement is being made in accordance with the Buy-back Regulations. Infosys Limited had announced the Offer to buyback equity shares."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Canaan Inc. Renews Its US$30 Million Share Buyback Program Canaan announced that its board of directors has authorized a share repurchase program under which the Company may buy back its ADSs over the next 12 months."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Elbit Systems To Report First Quarter 2025 Financial Results on May 20, 2025 The Company will host a Conference Call to discuss its financial results on May 20, 2025 at 9:00am ET."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "EARNINGS RELEASE DATE Melco Announces Earnings Release Date and today announces that it will release its unaudited financial results for the third quarter of 2025 on Thursday, November 6, 2025 to be followed by a conference call."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text("Q3 2025 RESULTS CALL")
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "MESOBLAST FINANCIAL RESULTS AND CORPORATE UPDATE WEBCAST Mesoblast Limited will host a webcast to discuss operational highlights and financial results for the half year ended December 31, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "The Bank will host an earnings call with analysts and investors regarding the audited standalone and consolidated financial results of the Bank for the quarter/year ended March 31, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "1Q 2025 EARNINGS RELEASE ZOOM MEETING HSBC will be holding a Zoom meeting today for investors and analysts. A copy of the presentation to investors and analysts is attached."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "OVERSEAS REGULATORY ANNOUNCEMENT - BOARD MEETING HSBC HOLDINGS PLC INTERIM RESULTS FOR 2025 Pursuant to Rule 13.43, notice is given that a meeting of a committee of the Board of Directors will be held to consider the announcement of the interim results for the six month period ended 30 June 2025 and to consider the payment of a second interim dividend."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Announcement Regarding 4th Quarter 2024 Financial Results Calendar Our Company's financial results for the period January 1, 2024 – December 31, 2024 are planned to be publicly announced on February 27, 2025 after Borsa Istanbul trading hours."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "2025 First Quarter Earnings Release Conference Woori Financial Group Inc. will hold its 2025 First Quarter Earnings Release Conference on Friday, April 25, 2025. Details of the Earnings Release Conference are as follows: Agenda : 2025 First Quarter Earnings Release and Q&A Date : Friday, April 25, 2025 Time : 16:00 (Korea Time) Format : Live Audio Webcast and Conference Call."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "VinFast Announces 4Q24 Global Deliveries, January 2025 Domestic Deliveries, 2025 Guidance and Sets Date for the Release of Full Year 2024 Results. The Company today announced its preliminary vehicle deliveries for the fourth quarter of 2024. The Company also announced that it will release its 4Q24 and full year 2024 financial results before the market opens on April 24, 2025. On the same day, management will hold a live webcast to discuss the Company’s business performance and strategy. Details for the call are below."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "MINUTES OF 170th MEETING OF THE AUDIT AND CONTROL COMMITEE OF TELEFÔNICA BRASIL S.A., HELD ON FEBRUARY 19th, 2025. Appraisal of the Company’s Financial Statements accompanied by the Independent Auditors’ Report and the Annual Management Report related to the fiscal year ended on December 31st, 2024. The Committee decided to recommend to the Company’s Board of Directors the approval of said documents."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Buenaventura Announces First Quarter 2025 Results for Production and Volume Sold per Metal. Production per Metal Three Months Ended March 31, 2025 2025 Guidance Gold ounces produced Volume Sold per Metal."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "QUARTERLY ACTIVITIES REPORT For the period ending 31 March 2025 Highlights Mineral Resource Estimate Planning for a Strategic Partnering Process has begun. Current Mining Projects."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Pearson Q1 2025 Trading Update (Unaudited) Pearson on track to deliver 2025 guidance with expected Q1 result and momentum building for the second half. Highlights Underlying Group sales up 1%, with growth expected to accelerate in the second half of the year. Financial and operating performance in our smallest quarter was in line with our plans."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Shell first quarter 2025 update noteThe following is an update to the first quarter 2025 outlook and gives an overview of our current expectations for the first quarter. Outlooks presented may vary from the actual first quarter 2025 results and are subject to finalisation of those results, which are scheduled to be published on May 2, 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "TIM S.A. MINUTES OF THE BOARD OF DIRECTORS’ MEETING HELD ON MAY 5TH, 2025. Agenda: resolve on the payment proposal of the Company’s interest on shareholders’ equity and acknowledge on the Company’s Quarterly Financial Report (ITRs) for the 1st quarter of 2025."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Telefônica Brasil S.A. MINUTES OF THE 241st FISCAL COUNCIL’S MEETING. Financial Statements accompanied by the Independent Auditors’ Report and the Annual Management Report related to the fiscal year ended on December 31st, 2024."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Autonomous City of Buenos Aires, May 7, 2025. To the BOLSAS Y MERCADOS ARGENTINOS S.A. and COMISIÓN NACIONAL DE VALORES. Dear Sirs: the Board of Directors approved the condensed interim financial statements for the three-month period ended March 31, 2025. Relevant information of the condensed consolidated interim financial statements follows: net profit for the period, other comprehensive income for the period, detail of shareholders’ equity."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "City of Buenos Aires, May 9, 2025. Messrs. BOLSAS Y MERCADOS ARGENTINOS, Messrs. NATIONAL SECURITIES COMMISSION, Dear Sirs, at the Board of Directors meeting held today, the following documents were approved: Condensed Interim Financial Statements, Statement of Financial Position, Statement of Comprehensive Income, Statement of Changes in Equity and Statement of Cash Flows for the three-month period ended March 31, 2025."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "To: Brazilian Securities and Exchange Commission (CVM) Subject: Request for clarification – News published in the media – Official Letter No. 222/2025/CVM/SEP/GEA-2. Dear Sirs, in response to the Official Letter referenced above, Vale S.A. hereby provides the following clarifications regarding the news article. During a conference call with analysts regarding the third quarter results, management commented on extraordinary dividends."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "OMA Announces Fourth Quarter 2024 Operating and Financial Results. The company today reported its unaudited, consolidated financial and operating results for the fourth quarter 2024 (4Q24). Adjusted EBITDA reached Ps.2,433 million."
        )
        == "RESULTS_RELEASE"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Appendix 3A.1 - Notification of dividend / distribution Announcement Summary Entity name RIO TINTO LIMITED Security on which the Distribution will be paid RIO - ORDINARY FULLY PAID"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Notice to ASX/LSE 2026 Key Dates Date Event 2025 Final dividend 19 February 2026 2025 full year results announcement 5 March 2026 Ex-dividend date for Rio Tinto plc and Rio Tinto Limited ordinary shares"
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert (
        _sec_6k_rules._classify_6k_text(
            "Exhibit 99.1 ASML Annual Report 2025 STRATEGIC REPORT CORPORATE GOVERNANCE SUSTAINABILITY FINANCIALS "
            + ("A" * 5000)
            + " Today, ASML has published its 2025 fourth-quarter and full-year results. Q4 total net sales were strong."
        )
        == "EXCLUDE_NON_QUARTERLY"
    )
    assert _sec_6k_rules._match_any("hello world", [r"world"]) is True


@pytest.mark.unit
def test_financial_payload_and_fiscal_extract_helpers() -> None:
    """验证财务载荷构建与 fiscal 提取辅助函数。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    payload, has_xbrl = _sec_fiscal_fields._build_financials_payload(_NoFinancialProcessor())
    assert payload is None
    assert has_xbrl is False

    payload, has_xbrl = _sec_fiscal_fields._build_financials_payload(_PartialFinancialProcessor())
    assert payload is None
    assert has_xbrl is False

    payload, has_xbrl = _sec_fiscal_fields._build_financials_payload(_MixedFinancialProcessor())
    assert isinstance(payload, dict)
    assert has_xbrl is True

    assert _sec_fiscal_fields._extract_fiscal_from_financials(None) == (None, None)
    assert _sec_fiscal_fields._extract_fiscal_from_financials({"statements": "bad"}) == (None, None)
    financials_payload = {
        "statements": {
            "income": {
                "periods": [
                    {"period_end": "2024-12-31", "fiscal_year": None, "fiscal_period": "Q4"},
                ]
            }
        }
    }
    assert _sec_fiscal_fields._extract_fiscal_from_financials(financials_payload) == (2024, "Q4")

    assert _sec_fiscal_fields._extract_fiscal_from_xbrl_query(object()) == (None, None)
    assert _sec_fiscal_fields._extract_fiscal_from_xbrl_query(_QueryProcessor(result={}, raise_error=True)) == (None, None)
    assert _sec_fiscal_fields._extract_fiscal_from_xbrl_query(_QueryProcessor(result=[])) == (None, None)
    assert _sec_fiscal_fields._extract_fiscal_from_xbrl_query(_QueryProcessor(result={"facts": "bad"})) == (None, None)

    only_year = _sec_fiscal_fields._extract_fiscal_from_xbrl_query(
        _QueryProcessor(result={"facts": [{"period_end": "2023-12-31"}]})
    )
    assert only_year == (2023, None)

    full_value = _sec_fiscal_fields._extract_fiscal_from_xbrl_query(
        _QueryProcessor(result={"facts": [{"fiscal_year": 2024, "fiscal_period": "FY"}]})
    )
    assert full_value == (2024, "FY")


@pytest.mark.unit
def test_resolve_fiscal_fields_and_normalize_helpers() -> None:
    """验证 fiscal 字段推断与标准化辅助函数。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_meta = {
        "fiscal_year": None,
        "fiscal_period": None,
        "form_type": "10-Q",
        "report_date": "2025-03-31",
    }
    resolved_year, resolved_period = _sec_fiscal_fields._resolve_processed_fiscal_fields(
        source_meta=source_meta,
        financials_payload=None,
        processor=_QueryProcessor(result={"facts": [{"fiscal_year": 2024, "fiscal_period": "FY"}]}),
        allow_xbrl_query=True,
    )
    assert resolved_year == 2024
    assert resolved_period is None

    source_meta_6k = {
        "fiscal_year": None,
        "fiscal_period": None,
        "form_type": "6-K",
        "report_date": "2025-06-30",
    }
    resolved_6k = _sec_fiscal_fields._resolve_processed_fiscal_fields(
        source_meta=source_meta_6k,
        financials_payload=None,
        processor=object(),
        allow_xbrl_query=False,
    )
    assert resolved_6k == (2025, None)

    source_meta_10k = {
        "fiscal_year": None,
        "fiscal_period": None,
        "form_type": "10-K",
        "report_date": "2024-12-31",
    }
    resolved_10k = _sec_fiscal_fields._resolve_processed_fiscal_fields(
        source_meta=source_meta_10k,
        financials_payload=None,
        processor=object(),
        allow_xbrl_query=False,
    )
    assert resolved_10k == (2024, "FY")

    assert _sec_fiscal_fields._coerce_optional_int(None) is None
    assert _sec_fiscal_fields._coerce_optional_int(True) is None
    assert _sec_fiscal_fields._coerce_optional_int("42") == 42
    assert _sec_fiscal_fields._coerce_optional_int("x42") is None
    assert _sec_fiscal_fields._normalize_optional_string("  x  ") == "x"
    assert _sec_fiscal_fields._normalize_optional_string("   ") is None
    assert _sec_fiscal_fields._normalize_optional_period(" q1 ") == "Q1"
    assert _sec_fiscal_fields._coerce_year_from_date("2024-09-30") == 2024
    assert _sec_fiscal_fields._coerce_year_from_date("2024/09/30") is None
    assert _sec_fiscal_fields._normalize_form_for_fiscal(" 10q ") == "10-Q"
    assert _sec_fiscal_fields._normalize_form_for_fiscal("6k/a") == "6-K/A"
    assert _sec_fiscal_fields._is_6k_family_form("6-K") is True
    assert _sec_fiscal_fields._is_6k_family_form("6-K/A") is True
    assert _sec_fiscal_fields._is_6k_family_form("8-K") is False

    assert _sec_fiscal_fields._should_skip_financial_extraction("8-K") is True
    assert _sec_fiscal_fields._should_skip_financial_extraction("SC13D") is True
    assert _sec_fiscal_fields._should_skip_financial_extraction(None) is False

    assert _sec_fiscal_fields._sanitize_fiscal_period_by_form("10-K", "FY") == "FY"
    assert _sec_fiscal_fields._sanitize_fiscal_period_by_form("10-K", "Q1") is None
    assert _sec_fiscal_fields._sanitize_fiscal_period_by_form("10-Q", "Q2") == "Q2"
    assert _sec_fiscal_fields._sanitize_fiscal_period_by_form("10-Q", "FY") is None
    assert _sec_fiscal_fields._sanitize_fiscal_period_by_form("6-K", "cy") == "CY"


@pytest.mark.unit
def test_pipeline_internal_form_windows(tmp_path: Path) -> None:
    """验证管线内部 form 窗口辅助逻辑（统一 DEFAULT_FORMS_US 策略）。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )

    # --- 验证 grace period 生效：窗口起点 = end_date - N年 - LOOKBACK_GRACE_DAYS ---
    grace = dt.timedelta(days=sec_pipeline.LOOKBACK_GRACE_DAYS)
    end = dt.date(2026, 3, 1)

    # 默认 forms：同时包含年报（10-K/20-F）和季报（10-Q/6-K）及其他表单
    default_windows = pipeline._resolve_form_windows(
        form_type=None,
        start_date=None,
        end_date=end,
    )
    # 年报回溯 5 年
    assert default_windows["10-K"] == dt.date(2021, 3, 1) - grace
    assert default_windows["20-F"] == dt.date(2021, 3, 1) - grace
    # 季报回溯 2 年；当期报告仍回溯 1 年
    assert default_windows["10-Q"] == dt.date(2024, 3, 1) - grace
    assert default_windows["6-K"] == dt.date(2024, 3, 1) - grace
    assert default_windows["8-K"] == dt.date(2025, 3, 1) - grace
    # DEF 14A 回溯 3 年
    assert default_windows["DEF 14A"] == dt.date(2023, 3, 1) - grace
    # SC 13 系列回溯 1 年
    assert default_windows["SC 13G"] == dt.date(2025, 3, 1) - grace

    # 手动指定 start_date 时不叠加 grace period
    explicit_windows = pipeline._resolve_form_windows(
        form_type="10-K",
        start_date="2020-01-01",
        end_date=end,
    )
    assert explicit_windows["10-K"] == dt.date(2020, 1, 1)


@pytest.mark.unit
def test_resolve_upload_status_and_extract_head_text() -> None:
    """验证上传状态映射与头部文本提取。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    assert _sec_upload_workflow._resolve_upload_status("uploaded") == "ok"
    assert _sec_upload_workflow._resolve_upload_status("failed") == "failed"
    payload = b"line1\nline2\nline3\n"
    assert _sec_6k_rules._extract_head_text(payload, max_lines=2) == "line1\nline2"


class _HistoryDownloader:
    """用于历史 filings 拉取测试的下载器桩。"""

    def fetch_json(self, url: str) -> dict[str, Any]:
        """返回历史 submissions 内容。

        Args:
            url: 历史文件 URL。

        Returns:
            历史表结构。

        Raises:
            RuntimeError: 指定 URL 用于覆盖异常分支时抛出。
        """

        if url.endswith("bad.json"):
            raise RuntimeError("history fetch failed")
        return {
            "form": ["10-K", "10-K"],
            "filingDate": ["2025-02-01", "2025-02-02"],
            "reportDate": ["2024-12-31", "2024-12-31"],
            "accessionNumber": ["0002", "0003"],
            "primaryDocument": ["hist_1.htm", "hist_2.htm"],
            "fileNumber": ["005-30000"],
        }

    def fetch_sc13_party_roles(self, archive_cik: str, accession_number: str) -> SimpleNamespace:
        """返回默认可保留的 SC13 方向角色。

        Args:
            archive_cik: archive CIK。
            accession_number: accession。

        Returns:
            方向角色。

        Raises:
            无。
        """

        del archive_cik, accession_number
        return SimpleNamespace(filed_by_cik="111111", subject_cik="320193")


class _BrowseDownloader:
    """用于 browse-edgar 补齐测试的下载器桩。"""

    def fetch_browse_edgar_filenum(self, filenum: str) -> list[SimpleNamespace]:
        """返回 browse-edgar 结果。

        Args:
            filenum: 文件编号。

        Returns:
            条目列表。

        Raises:
            RuntimeError: 指定 filenum 用于覆盖异常分支时抛出。
        """

        if filenum == "005-11111":
            raise RuntimeError("browse failed")
        return [
            SimpleNamespace(
                form_type="10-K",
                filing_date="2025-01-02",
                accession_number="0001-25-000001",
                cik="1",
            ),
            SimpleNamespace(
                form_type="SC 13G",
                filing_date="bad-date",
                accession_number="0002-25-000002",
                cik="1",
            ),
            SimpleNamespace(
                form_type="SC 13G",
                filing_date="2025-01-03",
                accession_number="0001-25-000001",
                cik="1",
            ),
            SimpleNamespace(
                form_type="SC 13G",
                filing_date="2025-01-04",
                accession_number="0003-25-ERR001",
                cik="1",
            ),
            SimpleNamespace(
                form_type="SC 13G",
                filing_date="2025-01-05",
                accession_number="0004-25-000004",
                cik="1",
            ),
        ]

    def resolve_primary_document(self, cik: str, accession_no_dash: str, form_type: str) -> str:
        """解析主文件名。

        Args:
            cik: CIK。
            accession_no_dash: 无连接符 accession。
            form_type: 表单类型。

        Returns:
            主文件名。

        Raises:
            RuntimeError: 用于覆盖失败分支时抛出。
        """

        del cik, form_type
        if "ERR001" in accession_no_dash:
            raise RuntimeError("resolve primary failed")
        return "primary_from_browse.htm"

    def fetch_sc13_party_roles(self, archive_cik: str, accession_number: str) -> Optional[SimpleNamespace]:
        """返回 browse 场景的 SC13 方向角色。

        Args:
            archive_cik: archive CIK。
            accession_number: accession。

        Returns:
            方向角色；无法判定时返回 `None`。

        Raises:
            无。
        """

        del archive_cik
        mapping: dict[str, Optional[SimpleNamespace]] = {
            "0001-25-000001": SimpleNamespace(filed_by_cik="111111", subject_cik="320193"),
            "0002-25-000002": SimpleNamespace(filed_by_cik="111111", subject_cik="320193"),
            "0003-25-ERR001": SimpleNamespace(filed_by_cik="111111", subject_cik="320193"),
            "0004-25-000004": SimpleNamespace(filed_by_cik="222222", subject_cik="320193"),
        }
        return mapping.get(accession_number)


class _Sc13DirectionDownloader:
    """用于 SC13 方向过滤测试的下载器桩。"""

    def fetch_sc13_party_roles(self, archive_cik: str, accession_number: str) -> Optional[SimpleNamespace]:
        """返回混合方向角色数据。

        Args:
            archive_cik: archive CIK。
            accession_number: accession。

        Returns:
            方向角色；无法判定时返回 `None`。

        Raises:
            无。
        """

        del archive_cik
        mapping: dict[str, Optional[SimpleNamespace]] = {
            # ticker 持股别人，应过滤
            "A-0001": SimpleNamespace(filed_by_cik="320193", subject_cik="999999"),
            # 别人持股 ticker，应保留
            "A-0002": SimpleNamespace(filed_by_cik="999999", subject_cik="320193"),
            # 无法判定，应过滤
            "A-0003": None,
        }
        return mapping.get(accession_number)


class _PrecheckDownloader:
    """用于 6-K 预筛选测试的下载器桩。"""

    def fetch_file_bytes(self, source_url: str) -> bytes:
        """返回远端文件字节内容。

        Args:
            source_url: 文件 URL。

        Returns:
            文件字节流。

        Raises:
            RuntimeError: 用于覆盖下载失败分支时抛出。
        """

        raise RuntimeError(f"download failed: {source_url}")


@pytest.mark.unit
def test_filter_filings_and_collect_filenums_from_history(tmp_path: Path) -> None:
    """覆盖 `_filter_filings` 的历史文件过滤与异常分支。"""

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=_HistoryDownloader(),  # type: ignore[arg-type]
        processor_registry=build_fins_processor_registry(),
    )
    submissions = {
        "filings": {
            "recent": {
                "form": ["10-K", "10-K", "10-K"],
                "filingDate": ["2025-01-10", "2023-01-10", "2025-01-15"],
                "reportDate": ["2024-12-31", "2022-12-31", "2024-12-31"],
                "accessionNumber": ["0001", "too_old", ""],
                "primaryDocument": ["a.htm", "b.htm", "c.htm"],
                "fileNumber": ["005-10000", "", "005-20000"],
            },
            "files": ["bad-entry", {"name": ""}, {"name": "bad.json"}, {"name": "hist.json"}],
        }
    }
    form_windows = {"10-K": dt.date(2024, 1, 1)}
    records, filenums = asyncio.run(
        pipeline._filter_filings(
            ticker="AAA",
            submissions=submissions,
            form_windows=form_windows,
            end_date=dt.date(2025, 12, 31),
            target_cik="320193",
            sc13_direction_cache={},
        )
    )
    assert [item.accession_number for item in records] == ["0001", "0002", "0003"]
    assert filenums == {"005-10000", "005-20000", "005-30000"}


@pytest.mark.unit
def test_filter_filings_keeps_latest_sc13_per_filer(tmp_path: Path) -> None:
    """覆盖 SC 13 在 `_filter_filings` 阶段按申报主体仅保留最新 1 份。"""

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=_HistoryDownloader(),  # type: ignore[arg-type]
        processor_registry=build_fins_processor_registry(),
    )
    submissions = {
        "filings": {
            "recent": {
                "form": ["SC 13G", "SC 13G/A", "SC 13D", "8-K"],
                "filingDate": ["2025-01-10", "2025-02-10", "2025-01-15", "2025-01-20"],
                "reportDate": ["", "", "", ""],
                "accessionNumber": ["0001", "0002", "0003", "0004"],
                "primaryDocument": ["sc13g.htm", "sc13ga.htm", "sc13d.htm", "8k.htm"],
                "fileNumber": ["005-10000", "005-10000", "005-20000", "001-99999"],
            },
            "files": [],
        }
    }
    form_windows = {
        "SC 13G": dt.date(2024, 1, 1),
        "SC 13G/A": dt.date(2024, 1, 1),
        "SC 13D": dt.date(2024, 1, 1),
        "8-K": dt.date(2024, 1, 1),
    }
    records, _filenums = asyncio.run(
        pipeline._filter_filings(
            ticker="AAA",
            submissions=submissions,
            form_windows=form_windows,
            end_date=dt.date(2025, 12, 31),
            target_cik="320193",
            sc13_direction_cache={},
        )
    )
    accession_set = {item.accession_number for item in records}
    # 005-10000 主体应仅保留较新的 0002（替换 0001）。
    assert accession_set == {"0002", "0003", "0004"}
    assert len(records) == 3


@pytest.mark.unit
def test_filter_sc13_by_direction_keeps_only_subject_ticker(tmp_path: Path) -> None:
    """验证 SC13 方向过滤仅保留“别人持股 ticker”记录。"""

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=_Sc13DirectionDownloader(),  # type: ignore[arg-type]
        processor_registry=build_fins_processor_registry(),
    )
    filings = [
        _sec_filing_collection.FilingRecord(
            form_type="SC 13G",
            filing_date="2025-01-01",
            report_date=None,
            accession_number="A-0001",
            primary_document="a.htm",
            filer_key="005-1",
        ),
        _sec_filing_collection.FilingRecord(
            form_type="SC 13G",
            filing_date="2025-01-02",
            report_date=None,
            accession_number="A-0002",
            primary_document="b.htm",
            filer_key="005-2",
        ),
        _sec_filing_collection.FilingRecord(
            form_type="SC 13D",
            filing_date="2025-01-03",
            report_date=None,
            accession_number="A-0003",
            primary_document="c.htm",
            filer_key="005-3",
        ),
        _sec_filing_collection.FilingRecord(
            form_type="8-K",
            filing_date="2025-01-04",
            report_date=None,
            accession_number="A-0004",
            primary_document="d.htm",
            filer_key="001-1",
        ),
    ]
    from dayu.fins.pipelines.sec_sc13_filtering import (
        SecSc13WorkflowHost as _SecSc13WorkflowHost,
        filter_sc13_by_direction as _filter_sc13_by_direction_impl,
    )
    from typing import cast as _cast

    filtered = asyncio.run(
        _filter_sc13_by_direction_impl(
            _cast(_SecSc13WorkflowHost, pipeline),
            ticker="AAA",
            filings=filings,
            target_cik="320193",
            archive_cik="320193",
            sc13_direction_cache={},
        )
    )
    assert [item.accession_number for item in filtered] == ["A-0002", "A-0004"]


@pytest.mark.unit
def test_extend_with_browse_edgar_sc13_covers_error_and_skip_paths(tmp_path: Path) -> None:
    """覆盖 browse-edgar 补齐中的异常、去重与有效补录分支。"""

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=_BrowseDownloader(),  # type: ignore[arg-type]
        processor_registry=build_fins_processor_registry(),
    )
    existing = [
        _sec_filing_collection.FilingRecord(
            form_type="SC 13G",
            filing_date="2025-01-03",
            report_date=None,
            accession_number="0001-25-000001",
            primary_document="existing.htm",
            filer_key="005-22222",
        )
    ]
    merged = asyncio.run(
        pipeline._extend_with_browse_edgar_sc13(
            ticker="AAA",
            filings=existing,
            filenums={"005-11111", "005-22222"},
            form_windows={"SC 13G": dt.date(2024, 1, 1)},
            end_date=dt.date(2025, 12, 31),
            target_cik="320193",
            sc13_direction_cache={},
        )
    )
    accession_set = {item.accession_number for item in merged}
    # 同一 filenum（005-22222）仅保留最新的 0004。
    assert "0001-25-000001" not in accession_set
    assert "0004-25-000004" in accession_set
    assert len(merged) == 1


@pytest.mark.unit
def test_precheck_6k_filter_error_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """覆盖 6-K 预筛选的异常路径。"""

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=_PrecheckDownloader(),  # type: ignore[arg-type]
        processor_registry=build_fins_processor_registry(),
    )
    original_classify_remote_candidates = _sec_filing_collection.classify_6k_remote_candidates

    keep, category, selected = asyncio.run(
        pipeline._precheck_6k_filter(
            remote_files=[],
            primary_document="a.htm",
            ticker="AAA",
            document_id="fil_1",
        )
    )
    assert (keep, category, selected) == (False, "NO_MATCH", "a.htm")

    remote_files = [
        RemoteFileDescriptor(
            name="a.htm",
            source_url="https://x/a.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
        )
    ]
    remote_files_with_xbrl = [
        RemoteFileDescriptor(
            name="sample-6k_htm.xml",
            source_url="https://x/sample-6k_htm.xml",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
        )
    ]
    keep, category, selected = asyncio.run(
        pipeline._precheck_6k_filter(
            remote_files=remote_files_with_xbrl,
            primary_document="a.htm",
            ticker="AAA",
            document_id="fil_xbrl",
        )
    )
    assert keep is True
    assert category == "XBRL_AVAILABLE"

    monkeypatch.setattr(
        sec_pipeline,
        "classify_6k_remote_candidates",
        AsyncMock(
            return_value=[
                _sec_6k_rules._SixKCandidateDiagnosis(
                    filename="a.htm",
                    filename_priority=0,
                    classification="NO_MATCH",
                    is_primary_document=True,
                )
            ]
        ),
    )

    # 无 EX-99/XBRL 且主文件未命中季度结果时，仍应拒绝
    keep, category, selected = asyncio.run(
        pipeline._precheck_6k_filter(
            remote_files=remote_files,
            primary_document="a.htm",
            ticker="AAA",
            document_id="fil_2",
        )
    )
    assert (keep, category, selected) == (False, "NO_EX99_OR_XBRL", "a.htm")

    remote_files_with_exhibit = [
        RemoteFileDescriptor(
            name="a.htm",
            source_url="https://x/a.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
            sec_document_type="EX-99.1",
        )
    ]
    # primary_document 为空 且 selected_name 在 remote_files 里也找不到 → NO_MATCH
    monkeypatch.setattr(sec_pipeline, "_select_6k_target_name", lambda *_args, **_kwargs: "missing.htm")
    keep, category, selected = asyncio.run(
        pipeline._precheck_6k_filter(
            remote_files=remote_files_with_exhibit,
            primary_document="",  # 无封面文件，退化为 selected_name；selected_name 也不在列表 → NO_MATCH
            ticker="AAA",
            document_id="fil_3",
        )
    )
    assert (keep, category, selected) == (False, "NO_MATCH", "missing.htm")

    monkeypatch.setattr(sec_pipeline, "_select_6k_target_name", lambda *_args, **_kwargs: "a.htm")
    monkeypatch.setattr(
        sec_pipeline,
        "classify_6k_remote_candidates",
        original_classify_remote_candidates,
    )
    keep, category, selected = asyncio.run(
        pipeline._precheck_6k_filter(
            remote_files=remote_files_with_exhibit,
            primary_document="a.htm",
            ticker="AAA",
            document_id="fil_4",
        )
    )
    assert (keep, category, selected) == (False, "DOWNLOAD_FAILED", "a.htm")


@pytest.mark.unit
def test_precheck_6k_filter_promotes_positive_candidate_even_when_primary_is_excluded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """6-K 旧 primary 被排除时，仍应提升真正命中季度结果的候选。"""

    class _ConflictDownloader:
        def fetch_file_bytes(self, source_url: str) -> bytes:
            payloads = {
                "https://x/cover.htm": b"cover",
                "https://x/ex99.htm": b"selected",
            }
            return payloads[source_url]

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=_ConflictDownloader(),  # type: ignore[arg-type]
        processor_registry=build_fins_processor_registry(),
    )
    remote_files = [
        RemoteFileDescriptor(
            name="cover.htm",
            source_url="https://x/cover.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
        ),
        RemoteFileDescriptor(
            name="ex99.htm",
            source_url="https://x/ex99.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
            sec_document_type="EX-99.1",
        ),
    ]
    monkeypatch.setattr(sec_pipeline, "_select_6k_target_name", lambda *_args, **_kwargs: "ex99.htm")
    monkeypatch.setattr(
        _sec_filing_collection,
        "_classify_6k_text",
        lambda content: "EXCLUDE_NON_QUARTERLY" if content == "cover" else "RESULTS_RELEASE",
    )

    keep, category, selected = asyncio.run(
        pipeline._precheck_6k_filter(
            remote_files=remote_files,
            primary_document="cover.htm",
            ticker="AAA",
            document_id="fil_conflict",
        )
    )

    assert keep is True
    assert category == "RESULTS_RELEASE"
    assert selected == "ex99.htm"


@pytest.mark.unit
def test_precheck_6k_filter_excludes_when_primary_is_excluded_and_no_positive_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """6-K 主文件被排除且无任何季度正文候选时，应维持排除。"""

    class _ConflictDownloader:
        def fetch_file_bytes(self, source_url: str) -> bytes:
            payloads = {
                "https://x/cover.htm": b"cover",
                "https://x/ex99.htm": b"selected",
            }
            return payloads[source_url]

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=_ConflictDownloader(),  # type: ignore[arg-type]
        processor_registry=build_fins_processor_registry(),
    )
    remote_files = [
        RemoteFileDescriptor(
            name="cover.htm",
            source_url="https://x/cover.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
        ),
        RemoteFileDescriptor(
            name="ex99.htm",
            source_url="https://x/ex99.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
            sec_document_type="EX-99.1",
        ),
    ]
    monkeypatch.setattr(sec_pipeline, "_select_6k_target_name", lambda *_args, **_kwargs: "ex99.htm")
    monkeypatch.setattr(
        _sec_filing_collection,
        "_classify_6k_text",
        lambda content: "EXCLUDE_NON_QUARTERLY" if content == "cover" else "NO_MATCH",
    )

    keep, category, selected = asyncio.run(
        pipeline._precheck_6k_filter(
            remote_files=remote_files,
            primary_document="cover.htm",
            ticker="AAA",
            document_id="fil_exclude_without_positive",
        )
    )

    assert keep is False
    assert category == "EXCLUDE_NON_QUARTERLY"
    assert selected == "ex99.htm"


@pytest.mark.unit
def test_precheck_6k_filter_keeps_positive_primary_without_exhibit_or_xbrl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """6-K 主文件自身命中季度结果时，不应因缺少 EX-99/XBRL 被提前拒绝。"""

    class _PrimaryOnlyDownloader:
        def fetch_file_bytes(self, source_url: str) -> bytes:
            payloads = {
                "https://x/cover.htm": b"cover",
            }
            return payloads[source_url]

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=_PrimaryOnlyDownloader(),  # type: ignore[arg-type]
        processor_registry=build_fins_processor_registry(),
    )
    remote_files = [
        RemoteFileDescriptor(
            name="cover.htm",
            source_url="https://x/cover.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
        )
    ]
    monkeypatch.setattr(
        _sec_filing_collection,
        "_classify_6k_text",
        lambda content: "RESULTS_RELEASE" if content == "cover" else "NO_MATCH",
    )

    keep, category, selected = asyncio.run(
        pipeline._precheck_6k_filter(
            remote_files=remote_files,
            primary_document="cover.htm",
            ticker="AAA",
            document_id="fil_primary_only_positive",
        )
    )

    assert keep is True
    assert category == "RESULTS_RELEASE"
    assert selected == "cover.htm"


@pytest.mark.unit
def test_precheck_6k_filter_promotes_positive_non_primary_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """6-K 预筛选应能把真正命中季度结果的非主候选提升为 primary。"""

    class _CandidateDownloader:
        def fetch_file_bytes(self, source_url: str) -> bytes:
            payloads = {
                "https://x/index.htm": b"cover",
                "https://x/ex99w01.htm": b"board",
                "https://x/ex99w03.htm": b"results",
            }
            return payloads[source_url]

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=_CandidateDownloader(),  # type: ignore[arg-type]
        processor_registry=build_fins_processor_registry(),
    )
    remote_files = [
        RemoteFileDescriptor(
            name="index.htm",
            source_url="https://x/index.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
        ),
        RemoteFileDescriptor(
            name="ex99w01.htm",
            source_url="https://x/ex99w01.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
            sec_document_type="EX-99.1",
        ),
        RemoteFileDescriptor(
            name="ex99w03.htm",
            source_url="https://x/ex99w03.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
            sec_document_type="EX-99.3",
        ),
    ]
    monkeypatch.setattr(sec_pipeline, "_select_6k_target_name", lambda *_args, **_kwargs: "ex99w01.htm")
    monkeypatch.setattr(
        _sec_filing_collection,
        "_classify_6k_text",
        lambda content: {
            "cover": "NO_MATCH",
            "board": "NO_MATCH",
            "results": "RESULTS_RELEASE",
        }[content],
    )

    keep, category, selected = asyncio.run(
        pipeline._precheck_6k_filter(
            remote_files=remote_files,
            primary_document="index.htm",
            ticker="INFY",
            document_id="fil_promote_candidate",
        )
    )

    assert keep is True
    assert category == "RESULTS_RELEASE"
    assert selected == "ex99w03.htm"


@pytest.mark.unit
def test_precheck_6k_filter_keeps_positive_primary_when_exhibits_mislead(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """6-K 预筛选在只有主文件命中季度结果时，应保留主文件自身。"""

    class _PrimaryDownloader:
        def fetch_file_bytes(self, source_url: str) -> bytes:
            payloads = {
                "https://x/cover.htm": b"cover",
                "https://x/ex99w01.htm": b"financial_report",
                "https://x/ex99w03.htm": b"presentation",
            }
            return payloads[source_url]

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=_PrimaryDownloader(),  # type: ignore[arg-type]
        processor_registry=build_fins_processor_registry(),
    )
    remote_files = [
        RemoteFileDescriptor(
            name="cover.htm",
            source_url="https://x/cover.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
        ),
        RemoteFileDescriptor(
            name="ex99w01.htm",
            source_url="https://x/ex99w01.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
            sec_document_type="EX-99.1",
        ),
        RemoteFileDescriptor(
            name="ex99w03.htm",
            source_url="https://x/ex99w03.htm",
            http_etag=None,
            http_last_modified=None,
            remote_size=None,
            http_status=None,
            sec_document_type="EX-99.3",
        ),
    ]
    monkeypatch.setattr(sec_pipeline, "_select_6k_target_name", lambda *_args, **_kwargs: "ex99w01.htm")
    monkeypatch.setattr(
        _sec_filing_collection,
        "_classify_6k_text",
        lambda content: {
            "cover": "RESULTS_RELEASE",
            "financial_report": "EXCLUDE_NON_QUARTERLY",
            "presentation": "NO_MATCH",
        }[content],
    )

    keep, category, selected = asyncio.run(
        pipeline._precheck_6k_filter(
            remote_files=remote_files,
            primary_document="cover.htm",
            ticker="FER",
            document_id="fil_primary_cover",
        )
    )

    assert keep is True
    assert category == "RESULTS_RELEASE"
    assert selected == "cover.htm"


@pytest.mark.unit
def test_meta_skip_and_file_entries_helpers(tmp_path: Path) -> None:
    """覆盖 meta 读取、skip 判断与文件条目构建辅助逻辑。"""

    pipeline = SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )
    assert pipeline._safe_get_processed_meta("AAA", "fil_1") is None

    assert pipeline._can_skip(None, "fp", overwrite=False) is None
    assert pipeline._can_skip({"ingest_complete": False}, "fp", overwrite=False) is None
    assert pipeline._can_skip(
        {
            "ingest_complete": True,
            "download_version": "legacy",
            "source_fingerprint": "fp",
        },
        "fp",
        overwrite=False,
    ) is None
    assert pipeline._can_skip(
        {
            "ingest_complete": True,
            "download_version": sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION,
            "source_fingerprint": " ",
        },
        "fp",
        overwrite=False,
    ) is None
    assert pipeline._can_skip(
        {
            "ingest_complete": True,
            "download_version": sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION,
            "source_fingerprint": "fp",
        },
        "fp",
        overwrite=False,
    ) == "source_fingerprint_matched"
    assert pipeline._can_skip(
        {
            "ingest_complete": True,
            "download_version": sec_pipeline.SEC_PIPELINE_DOWNLOAD_VERSION,
            "source_fingerprint": "legacy-fp",
            "files": [
                {
                    "name": "a.htm",
                    "http_etag": "\"etag-core\"",
                    "http_last_modified": "Mon, 01 Jan 2025 00:00:00 GMT",
                    "size": 123,
                }
            ],
        },
        "new-fp",
        overwrite=False,
        remote_files=[
            RemoteFileDescriptor(
                name="a.htm",
                source_url="https://x/a.htm",
                http_etag="W/\"etag-core-gzip\"",
                http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
                remote_size=20,
                http_status=200,
            )
        ],
    ) == "remote_files_equivalent"

    file_meta = SimpleNamespace(
        uri="local://AAA/a.htm",
        etag="e1",
        last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
        size=123,
        content_type="text/html",
        sha256="abc",
    )
    entries = pipeline._build_file_entries(
        file_results=[
            {"status": "downloaded", "name": "bad.htm", "file_meta": None},
            {
                "status": "downloaded",
                "name": "a.htm",
                "file_meta": file_meta,
                "source_url": "https://x/a.htm",
                "http_etag": "etag",
                "http_last_modified": "Mon, 01 Jan 2025 00:00:00 GMT",
            },
            {"status": "skipped", "name": "a_prev.htm"},
            {"status": "skipped", "name": "a_keep.htm"},
        ],
        previous_files={"a_keep.htm": {"name": "a_keep.htm", "uri": "local://AAA/a_keep.htm"}},
    )
    assert len(entries) == 2
    assert {item["name"] for item in entries} == {"a.htm", "a_keep.htm"}


@pytest.mark.unit
def test_safe_get_filing_source_meta_reads_active_batch_staging(tmp_path: Path) -> None:
    """验证 `_safe_get_filing_source_meta` 能读取活动 batch 的 staging 视图。"""

    context = build_fs_storage_test_context(tmp_path)
    repository = context.core
    token = repository.begin_batch("AAPL")
    try:
        handle = SourceHandle(ticker="AAPL", document_id="fil_batch_meta", source_kind=SourceKind.FILING.value)
        stored = repository.store_file(handle, "primary.htm", BytesIO(b"<html>batch</html>"))
        repository.create_filing(
            FilingCreateRequest(
                ticker="AAPL",
                document_id="fil_batch_meta",
                internal_document_id="batch_meta",
                form_type="10-K",
                primary_document="primary.htm",
                files=[stored],
                meta={"source_fingerprint": "batch-stage-fp"},
            )
        )

        pipeline = SecPipeline(
            workspace_root=tmp_path,
            company_repository=context.company_repository,
            source_repository=context.source_repository,
            processed_repository=context.processed_repository,
            blob_repository=context.blob_repository,
            filing_maintenance_repository=context.filing_maintenance_repository,
            processor_registry=build_fins_processor_registry(),
        )
        result = pipeline._safe_get_filing_source_meta("AAPL", "fil_batch_meta")
        assert result is not None
        assert result["source_fingerprint"] == "batch-stage-fp"
    finally:
        repository.rollback_batch(token)


@pytest.mark.unit
def test_fiscal_and_misc_helper_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    """覆盖 fiscal 相关辅助函数的剩余边界分支。"""

    assert _sec_fiscal_fields._resolve_processed_quality(False, True, "DEF 14A") == "partial"
    assert _sec_fiscal_fields._resolve_processed_quality(True, False, "10-K") == "full"
    assert _sec_fiscal_fields._resolve_processed_quality(False, True, "10-K") == "partial"
    assert _sec_fiscal_fields._resolve_processed_quality(False, False, None) == "fallback"

    early_year, early_period = _sec_fiscal_fields._resolve_processed_fiscal_fields(
        source_meta={"fiscal_year": "2024", "fiscal_period": "Q2", "form_type": "10-Q", "report_date": "2024-06-30"},
        financials_payload=None,
        processor=object(),
        allow_xbrl_query=False,
    )
    assert (early_year, early_period) == (2024, "Q2")

    sanitized_year, sanitized_period = _sec_fiscal_fields._resolve_processed_fiscal_fields(
        source_meta={"fiscal_year": "2024", "fiscal_period": "FY", "form_type": "10-Q", "report_date": "2024-06-30"},
        financials_payload={
            "statements": {
                "income": {"periods": [{"fiscal_year": 2023, "fiscal_period": "FY"}]},
            }
        },
        processor=object(),
        allow_xbrl_query=False,
    )
    assert (sanitized_year, sanitized_period) == (2024, None)

    complex_payload = {
        "statements": {
            "income": "bad",
            "balance_sheet": {"periods": "bad"},
            "cash_flow": {"periods": [123, {"period_end": "2024-12-31"}]},
        }
    }
    assert _sec_fiscal_fields._extract_fiscal_from_financials(complex_payload) == (2024, None)

    assert _sec_fiscal_fields._coerce_optional_int("   ") is None

    assert _sec_fiscal_fields._coerce_optional_int("42") == 42
    assert _sec_form_utils.parse_date("2025-12", is_end=True) == dt.date(2025, 12, 31)
    assert _sec_form_utils.parse_date("2025-12", is_end=False) == dt.date(2025, 12, 1)
    assert _sec_fiscal_fields._infer_download_fiscal_fields("10-K", "2024-09-30") == (2024, "FY")
    assert _sec_fiscal_fields._infer_download_fiscal_fields("6-K", "2024-12-31") == (None, None)
    assert _sec_fiscal_fields._infer_download_fiscal_fields("6-K/A", "2024-12-31") == (None, None)
    assert _sec_fiscal_fields._infer_download_fiscal_fields("8-K", "2024-12-31") == (2024, None)
    assert _sec_fiscal_fields._infer_download_fiscal_fields("10-Q", "bad-date") == (None, None)
    monkeypatch.setattr(_sec_fiscal_fields, "_extract_download_fiscal_from_xbrl", lambda **_kwargs: (2023, "FY"))
    assert _sec_fiscal_fields._resolve_download_fiscal_fields(
        source_handle=object(),  # type: ignore[arg-type]
        source_repository=object(),  # type: ignore[arg-type]
        file_entries=[],
        form_type="10-K",
        report_date="2024-12-31",
    ) == (2023, "FY")
    monkeypatch.setattr(_sec_fiscal_fields, "_extract_download_fiscal_from_xbrl", lambda **_kwargs: (2023, None))
    assert _sec_fiscal_fields._resolve_download_fiscal_fields(
        source_handle=object(),  # type: ignore[arg-type]
        source_repository=object(),  # type: ignore[arg-type]
        file_entries=[],
        form_type="6-K",
        report_date="2024-12-31",
    ) == (2023, None)
    monkeypatch.setattr(_sec_fiscal_fields, "_extract_download_fiscal_from_xbrl", lambda **_kwargs: (None, None))
    assert _sec_fiscal_fields._resolve_download_fiscal_fields(
        source_handle=object(),  # type: ignore[arg-type]
        source_repository=object(),  # type: ignore[arg-type]
        file_entries=[],
        form_type="6-K",
        report_date="2024-12-31",
    ) == (None, None)
    assert _sec_fiscal_fields._resolve_download_fiscal_fields(
        source_handle=object(),  # type: ignore[arg-type]
        source_repository=object(),  # type: ignore[arg-type]
        file_entries=[],
        form_type="10-K",
        report_date="2024-12-31",
    ) == (2024, "FY")

    assert _sec_6k_rules._infer_filename_from_uri("") == ""
    assert _sec_6k_rules._infer_filename_from_uri("local://") == ""
    assert _sec_fiscal_fields._sanitize_fiscal_period_by_form("10-Q", "   ") is None

    # _normalize_form_for_fiscal 对无法识别的 form 返回 None，_should_skip 应返回 False
    assert _sec_fiscal_fields._should_skip_financial_extraction("BAD-FORM") is False


@pytest.mark.unit
def test_warn_insufficient_filings() -> None:
    """验证 _warn_insufficient_filings 按年报/季报分组检查落盘数量。

    分组规则：
    - 年报组：10-K + 20-F 合并计数，最低期望 5。
    - 季报组：10-Q + 6-K 合并计数，最低期望 3。
    - DEF 14A：单独检查，最低期望 2。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """


@pytest.mark.unit
def test_resolve_download_fiscal_fields_covers_period_only_and_20f_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证下载 fiscal 解析覆盖 period-only 与 20-F 回退分支。"""

    monkeypatch.setattr(_sec_fiscal_fields, "_extract_download_fiscal_from_xbrl", lambda **_kwargs: (None, "Q3"))
    period_only = _sec_fiscal_fields._resolve_download_fiscal_fields(
        source_handle=object(),  # type: ignore[arg-type]
        source_repository=object(),  # type: ignore[arg-type]
        file_entries=[],
        form_type="10-Q",
        report_date="2024-09-30",
    )

    monkeypatch.setattr(_sec_fiscal_fields, "_extract_download_fiscal_from_xbrl", lambda **_kwargs: (2023, None))
    annual_fallback = _sec_fiscal_fields._resolve_download_fiscal_fields(
        source_handle=object(),  # type: ignore[arg-type]
        source_repository=object(),  # type: ignore[arg-type]
        file_entries=[],
        form_type="20-F",
        report_date="2024-12-31",
    )

    assert period_only == (2024, "Q3")
    assert annual_fallback == (2023, "FY")

    def _result(form: str, status: str) -> dict:
        return {"form_type": form, "status": status}

    # ── 数量足够，不产生 warning ──
    form_windows_domestic = {
        "10-K": dt.date(2021, 1, 1),
        "10-Q": dt.date(2025, 1, 1),
        "DEF 14A": dt.date(2023, 1, 1),
    }
    enough_results = (
        [_result("10-K", "downloaded")] * 5
        + [_result("10-Q", "downloaded")] * 3
        + [_result("DEF 14A", "skipped")] * 2
    )
    assert _sec_download_diagnostics.warn_insufficient_filings(
        form_windows=form_windows_domestic,
        filing_results=enough_results,
        rejection_registry={},
    ) == []

    # ── FUTU 外国公司情景：form_windows 含 10-K/20-F/10-Q/6-K，
    #    实际只有 20-F 和 6-K 落盘 → 年报组+季报组都满足，无 warning ──
    form_windows_futu = {
        "10-K": dt.date(2021, 1, 1),
        "20-F": dt.date(2021, 1, 1),
        "10-Q": dt.date(2025, 1, 1),
        "6-K": dt.date(2025, 1, 1),
        "DEF 14A": dt.date(2023, 1, 1),
    }
    futu_results = (
        [_result("20-F", "downloaded")] * 5
        + [_result("6-K", "downloaded")] * 3
        + [_result("DEF 14A", "downloaded")] * 3
    )
    assert _sec_download_diagnostics.warn_insufficient_filings(
        form_windows=form_windows_futu,
        filing_results=futu_results,
        rejection_registry={},
    ) == []

    # ── 联合补足情景：10-K 3 份 + 20-F 2 份 = 5 → 年报组满足，无 warning ──
    assert _sec_download_diagnostics.warn_insufficient_filings(
        form_windows={"10-K": dt.date(2021, 1, 1), "20-F": dt.date(2021, 1, 1)},
        filing_results=[_result("10-K", "downloaded")] * 3
        + [_result("20-F", "skipped")] * 2,
        rejection_registry={},
    ) == []

    # ── 年报组：form_windows 只含 10-K，季报组未请求 ──
    # 10-K=0 < 5 → 年报 warning；10-Q 未请求 → 无季报 warning
    warns = _sec_download_diagnostics.warn_insufficient_filings(
        form_windows={"10-K": dt.date(2021, 1, 1)},
        filing_results=[],
        rejection_registry={},
    )
    assert len(warns) == 1
    assert "年报" in warns[0]
    assert "10-K" in warns[0]
    assert "季报" not in warns[0]

    # ── failed 不算有效落盘 ──
    warns = _sec_download_diagnostics.warn_insufficient_filings(
        form_windows={"10-Q": dt.date(2025, 1, 1)},
        filing_results=[_result("10-Q", "failed")] * 5,
        rejection_registry={},
    )
    assert len(warns) == 1
    assert "季报" in warns[0]
    assert "10-Q" in warns[0]

    # ── 6-K：有效数不足 + rejection_registry 有记录 → 提示 _classify_6k_text ──
    registry_with_6k = {
        "fil_001": {"form_type": "6-K", "reason": "6k_filtered", "category": "NO_MATCH",
                    "filing_date": "2025-01-01", "download_version": "v1"},
        "fil_002": {"form_type": "6-K", "reason": "6k_filtered", "category": "EXCLUDE_NON_QUARTERLY",
                    "filing_date": "2025-03-01", "download_version": "v1"},
    }
    warns = _sec_download_diagnostics.warn_insufficient_filings(
        form_windows={"6-K": dt.date(2025, 1, 1)},
        filing_results=[_result("6-K", "downloaded")],  # 1 < 3
        rejection_registry=registry_with_6k,
    )
    assert len(warns) == 1
    assert "6-K" in warns[0]
    assert "rejection_registry" in warns[0]
    assert "2" in warns[0]  # rejected_count

    # ── 6-K：有效数不足 + rejection_registry 无 6-K 记录 → 提示无季报 6-K ──
    warns = _sec_download_diagnostics.warn_insufficient_filings(
        form_windows={"6-K": dt.date(2025, 1, 1)},
        filing_results=[],
        rejection_registry={},
    )
    assert len(warns) == 1
    assert "6-K" in warns[0]
    assert "rejection_registry" not in warns[0]

    # ── 多个组同时不足 ──
    warns = _sec_download_diagnostics.warn_insufficient_filings(
        form_windows={
            "20-F": dt.date(2021, 1, 1),
            "6-K": dt.date(2025, 1, 1),
            "DEF 14A": dt.date(2023, 1, 1),
        },
        filing_results=[_result("20-F", "downloaded")] * 2,  # 2 < 5
        rejection_registry={},
    )
    # 年报组(20-F)、季报组(6-K)、DEF 14A 都不足
    assert len(warns) == 3
    all_text = " ".join(warns)
    assert "年报" in all_text
    assert "20-F" in all_text
    assert "季报" in all_text
    assert "6-K" in all_text
    assert "DEF 14A" in all_text


def test_clear_filings_dir(tmp_path: Path) -> None:
    """验证 _clear_filings_dir 清空 filings 目录下所有内容。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    ticker = "AAPL"
    filings_dir = tmp_path / "portfolio" / ticker / "filings"
    filings_dir.mkdir(parents=True)

    # 创建若干子目录和文件
    (filings_dir / "fil_001").mkdir()
    (filings_dir / "fil_001" / "meta.json").write_text("{}")
    (filings_dir / "fil_002").mkdir()
    (filings_dir / "fil_002" / "meta.json").write_text("{}")
    (filings_dir / "filing_manifest.json").write_text('{"documents":[]}')
    (filings_dir / "_download_rejections.json").write_text("{}")
    repository = build_fs_storage_test_context(tmp_path).filing_maintenance_repository

    repository.clear_filing_documents(ticker)

    # filings 目录本身保留
    assert filings_dir.exists()
    # 所有子目录和文件已被清除
    remaining = list(filings_dir.iterdir())
    assert remaining == [], f"filings 目录应为空，实际残留: {remaining}"

    # ── 目录不存在时不报错 ──
    repository.clear_filing_documents("NONEXISTENT")


def test_clear_processed_dir(tmp_path: Path) -> None:
    """验证 _clear_processed_dir 清空 processed 目录下所有内容。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    ticker = "AAPL"
    processed_dir = tmp_path / "portfolio" / ticker / "processed"
    processed_dir.mkdir(parents=True)

    # 创建子目录（filing 快照）和顶层文件
    (processed_dir / "fil_001").mkdir()
    (processed_dir / "fil_001" / "tool_snapshot_meta.json").write_text("{}")
    (processed_dir / "mat_001").mkdir()
    (processed_dir / "mat_001" / "tool_snapshot_meta.json").write_text("{}")
    (processed_dir / "processed_manifest.json").write_text('{"documents":[]}')

    repository = build_fs_storage_test_context(tmp_path).processed_repository

    _processed_snapshot_helpers.clear_processed_documents(repository=repository, ticker=ticker)

    # processed 目录本身保留
    assert processed_dir.exists()
    # 所有子目录和文件已被清除
    remaining = list(processed_dir.iterdir())
    assert remaining == [], f"processed 目录应为空，实际残留: {remaining}"

    # ── 目录不存在时不报错 ──
    _processed_snapshot_helpers.clear_processed_documents(repository=repository, ticker="NONEXISTENT")


def test_cleanup_stale_filing_dirs(tmp_path: Path) -> None:
    """验证 _cleanup_stale_filing_dirs 删除过期目录并同步更新 manifest。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    import json

    ticker = "AAPL"
    filings_dir = tmp_path / "portfolio" / ticker / "filings"
    filings_dir.mkdir(parents=True)
    repository = build_fs_storage_test_context(tmp_path).filing_maintenance_repository

    def _mk_filing(doc_id: str, form_type: str) -> None:
        d = filings_dir / doc_id
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"form_type": form_type}))

    _mk_filing("fil_001", "10-K")   # 本次返回（valid）
    _mk_filing("fil_002", "10-K")   # 本次未返回（stale）
    _mk_filing("fil_003", "10-Q")   # form_type 不在 form_windows，跳过
    _mk_filing("fil_004", "10-K")   # 本次 downloaded（valid）

    # 写入 manifest（包含 4 条）
    manifest = {
        "ticker": ticker,
        "updated_at": "2026-01-01T00:00:00",
        "documents": [
            {"document_id": "fil_001", "form_type": "10-K"},
            {"document_id": "fil_002", "form_type": "10-K"},
            {"document_id": "fil_003", "form_type": "10-Q"},
            {"document_id": "fil_004", "form_type": "10-K"},
        ],
    }
    manifest_path = filings_dir / "filing_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    form_windows = {"10-K": dt.date(2021, 1, 1)}  # 只覆盖 10-K
    filing_results = [
        {"document_id": "fil_001", "form_type": "10-K", "status": "skipped"},
        {"document_id": "fil_004", "form_type": "10-K", "status": "downloaded"},
    ]

    cleaned = sec_pipeline._cleanup_stale_filing_dirs(
        repository=repository,
        ticker=ticker,
        form_windows=form_windows,
        filing_results=filing_results,
    )

    assert cleaned == 1  # fil_002 被删除
    assert not (filings_dir / "fil_002").exists()
    assert (filings_dir / "fil_001").exists()   # valid，保留
    assert (filings_dir / "fil_003").exists()   # form 不在 form_windows，保留
    assert (filings_dir / "fil_004").exists()   # valid，保留

    # manifest 中 fil_002 已移除，其余 3 条保留
    updated = json.loads(manifest_path.read_text())
    doc_ids = {d["document_id"] for d in updated["documents"]}
    assert doc_ids == {"fil_001", "fil_003", "fil_004"}

    # ── failed 状态不视为 valid，应被清理 ──
    _mk_filing("fil_005", "10-K")
    filing_results_2 = [
        {"document_id": "fil_001", "form_type": "10-K", "status": "skipped"},
        {"document_id": "fil_004", "form_type": "10-K", "status": "downloaded"},
        {"document_id": "fil_005", "form_type": "10-K", "status": "failed"},  # failed 不保护
    ]
    # 写入包含 fil_005 的 manifest（fil_001/fil_004 已在前一轮被移除，此处补回验证）
    manifest2 = {
        "ticker": ticker,
        "updated_at": "2026-01-01T00:00:00",
        "documents": [{"document_id": "fil_005", "form_type": "10-K"}],
    }
    manifest_path.write_text(json.dumps(manifest2))

    cleaned2 = sec_pipeline._cleanup_stale_filing_dirs(
        repository=repository,
        ticker=ticker,
        form_windows=form_windows,
        filing_results=filing_results_2,
    )
    assert cleaned2 == 1
    assert not (filings_dir / "fil_005").exists()

    # ── 目录不存在时返回 0 ──
    assert sec_pipeline._cleanup_stale_filing_dirs(
        repository=repository,
        ticker="NONEXISTENT",
        form_windows=form_windows,
        filing_results=[],
    ) == 0


@pytest.mark.unit
def test_split_form_input_preserves_forms_with_spaces() -> None:
    """``split_form_input`` 仅按逗号拆分，不按空格拆，确保 SC 13G 等不被破坏。"""
    assert _sec_form_utils.split_form_input("10-K,10-Q") == ["10-K", "10-Q"]
    assert _sec_form_utils.split_form_input("SC 13G,10-K,DEF 14A") == ["SC 13G", "10-K", "DEF 14A"]
    assert _sec_form_utils.split_form_input("SC 13G") == ["SC 13G"]
    assert _sec_form_utils.split_form_input("10-K, 10-Q") == ["10-K", "10-Q"]

    with pytest.raises(ValueError, match="form_type 不能为空"):
        _sec_form_utils.split_form_input("")
    with pytest.raises(ValueError, match="form_type 不能为空"):
        _sec_form_utils.split_form_input(" , ")
