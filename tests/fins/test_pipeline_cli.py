"""Pipeline 工厂与 CLI 路由测试。"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional, cast

import pytest

from dayu.fins import cli_support as cli
from dayu.log import Log, LogLevel
from dayu.fins.domain.document_models import CompanyMeta
from dayu.fins.downloaders.sec_downloader import RemoteFileDescriptor, SecDownloader
from dayu.fins.ingestion.process_events import ProcessEvent, ProcessEventType
from dayu.fins.pipelines import CnPipeline, SecPipeline, get_pipeline_from_normalized_ticker
from dayu.fins.pipelines.base import PipelineProtocol
from dayu.fins.pipelines.download_events import DownloadEvent, DownloadEventType
from dayu.fins.pipelines.upload_filing_events import UploadFilingEvent, UploadFilingEventType
from dayu.fins.pipelines.upload_material_events import UploadMaterialEvent, UploadMaterialEventType
from dayu.fins.processors.registry import build_fins_processor_registry
from dayu.fins.resolver.fmp_company_alias_resolver import FmpAliasInferenceResult
from dayu.fins.ticker_normalization import NormalizedTicker


class FakePipeline(PipelineProtocol):
    """用于验证 CLI 路由的假 pipeline。"""

    def __init__(self) -> None:
        """初始化调用记录容器。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.last_call: Optional[tuple[str, dict[str, Any]]] = None

    def download(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """记录 download 调用。

        Args:
            ticker: 股票代码。
            form_type: 可选 form 类型。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否覆盖。
            rebuild: 是否重建本地 meta/manifest。
            ticker_aliases: 可选 alias 列表。

        Returns:
            占位结果。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "form_type": form_type,
            "start_date": start_date,
            "end_date": end_date,
            "overwrite": overwrite,
            "rebuild": rebuild,
            "ticker_aliases": ticker_aliases,
        }
        self.last_call = ("download", payload)
        return {"action": "download", **payload}

    async def download_stream(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: Optional[list[str]] = None,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> AsyncIterator[DownloadEvent]:
        """记录 download_stream 调用并返回最小事件流。

        Args:
            ticker: 股票代码。
            form_type: 可选 form 类型。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否覆盖。
            rebuild: 是否重建本地 meta/manifest。
            ticker_aliases: 可选 alias 列表。
            cancel_checker: 可选取消检查函数。

        Yields:
            最小下载事件流。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "form_type": form_type,
            "start_date": start_date,
            "end_date": end_date,
            "overwrite": overwrite,
            "rebuild": rebuild,
            "ticker_aliases": ticker_aliases,
            "cancel_checker": cancel_checker,
        }
        self.last_call = ("download_stream", payload)
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_STARTED,
            ticker=ticker,
            payload={"form_type": form_type, "overwrite": overwrite, "rebuild": rebuild},
        )
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_COMPLETED,
            ticker=ticker,
            payload={
                "result": _mock_download_result(
                    ticker,
                    form_type,
                    start_date,
                    end_date,
                    overwrite,
                    rebuild,
                )
            },
        )

    def upload_filing(
        self,
        ticker: str,
        action: str | None,
        files: list[Path],
        fiscal_year: int,
        fiscal_period: str,
        amended: bool = False,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """记录 upload_filing 调用。

        Args:
            ticker: 股票代码。
            action: 上传动作。
            files: 文件列表。
            fiscal_year: 财年。
            fiscal_period: 财季/财年标记。
            amended: 是否修订版。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID。
            company_name: 公司名称。
            ticker_aliases: 可选 alias 列表。
            overwrite: 是否覆盖。

        Returns:
            占位结果。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "action": action,
            "files": [str(item) for item in files],
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "amended": amended,
            "filing_date": filing_date,
            "report_date": report_date,
            "company_id": company_id,
            "company_name": company_name,
            "ticker_aliases": ticker_aliases,
            "overwrite": overwrite,
        }
        self.last_call = ("upload_filing", payload)
        return {
            "action": "upload_filing",
            "ticker": ticker,
            "filing_action": action,
            "files": [str(item) for item in files],
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "amended": amended,
            "filing_date": filing_date,
            "report_date": report_date,
            "company_id": company_id,
            "company_name": company_name,
            "ticker_aliases": ticker_aliases,
            "overwrite": overwrite,
        }

    async def upload_filing_stream(
        self,
        ticker: str,
        action: str | None,
        files: list[Path],
        fiscal_year: int,
        fiscal_period: str,
        amended: bool = False,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> AsyncIterator[UploadFilingEvent]:
        """记录 upload_filing_stream 调用并返回最小事件流。

        Args:
            ticker: 股票代码。
            action: 上传动作。
            files: 文件列表。
            fiscal_year: 财年。
            fiscal_period: 财季/财年标记。
            amended: 是否修订版。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID。
            company_name: 公司名称。
            ticker_aliases: 可选 alias 列表。
            overwrite: 是否覆盖。

        Yields:
            最小上传事件流。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "action": action,
            "files": [str(item) for item in files],
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "amended": amended,
            "filing_date": filing_date,
            "report_date": report_date,
            "company_id": company_id,
            "company_name": company_name,
            "ticker_aliases": ticker_aliases,
            "overwrite": overwrite,
        }
        self.last_call = ("upload_filing_stream", payload)
        yield UploadFilingEvent(
            event_type=UploadFilingEventType.UPLOAD_STARTED,
            ticker=ticker,
            payload={"file_count": len(files)},
        )
        yield UploadFilingEvent(
            event_type=UploadFilingEventType.UPLOAD_COMPLETED,
            ticker=ticker,
            payload={
                "result": {
                    "action": "upload_filing",
                    "ticker": ticker,
                    "filing_action": action,
                    "files": [str(item) for item in files],
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "amended": amended,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "company_id": company_id,
                    "company_name": company_name,
                    "ticker_aliases": ticker_aliases,
                    "overwrite": overwrite,
                }
            },
        )

    def upload_material(
        self,
        ticker: str,
        action: str | None,
        form_type: str,
        material_name: str,
        files: Optional[list[Path]] = None,
        document_id: Optional[str] = None,
        internal_document_id: Optional[str] = None,
        fiscal_year: int | None = None,
        fiscal_period: str | None = None,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """记录 upload_material 调用。

        Args:
            ticker: 股票代码。
            action: 上传动作。
            form_type: 材料类型。
            material_name: 材料名称。
            files: 文件列表。
            document_id: 文档 ID。
            internal_document_id: 内部文档 ID。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID。
            company_name: 公司名称。
            ticker_aliases: 可选 alias 列表。
            overwrite: 是否覆盖。

        Returns:
            占位结果。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "action": action,
            "form_type": form_type,
            "material_name": material_name,
            "files": [str(item) for item in files] if files else [],
            "document_id": document_id,
            "internal_document_id": internal_document_id,
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "filing_date": filing_date,
            "report_date": report_date,
            "company_id": company_id,
            "company_name": company_name,
            "ticker_aliases": ticker_aliases,
            "overwrite": overwrite,
        }
        self.last_call = ("upload_material", payload)
        return {
            "action": "upload_material",
            "ticker": ticker,
            "material_action": action,
            "form_type": form_type,
            "material_name": material_name,
            "files": [str(item) for item in files] if files else [],
            "document_id": document_id,
            "internal_document_id": internal_document_id,
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "filing_date": filing_date,
            "report_date": report_date,
            "company_id": company_id,
            "company_name": company_name,
            "ticker_aliases": ticker_aliases,
            "overwrite": overwrite,
        }

    async def upload_material_stream(
        self,
        ticker: str,
        action: str | None,
        form_type: str,
        material_name: str,
        files: Optional[list[Path]] = None,
        document_id: Optional[str] = None,
        internal_document_id: Optional[str] = None,
        fiscal_year: int | None = None,
        fiscal_period: str | None = None,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> AsyncIterator[UploadMaterialEvent]:
        """记录 upload_material_stream 调用并返回最小事件流。

        Args:
            ticker: 股票代码。
            action: 上传动作。
            form_type: 材料类型。
            material_name: 材料名称。
            files: 文件列表。
            document_id: 文档 ID。
            internal_document_id: 内部文档 ID。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID。
            company_name: 公司名称。
            ticker_aliases: 可选 alias 列表。
            overwrite: 是否覆盖。

        Yields:
            最小上传事件流。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "action": action,
            "form_type": form_type,
            "material_name": material_name,
            "files": [str(item) for item in files] if files else [],
            "document_id": document_id,
            "internal_document_id": internal_document_id,
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "filing_date": filing_date,
            "report_date": report_date,
            "company_id": company_id,
            "company_name": company_name,
            "ticker_aliases": ticker_aliases,
            "overwrite": overwrite,
        }
        self.last_call = ("upload_material_stream", payload)
        yield UploadMaterialEvent(
            event_type=UploadMaterialEventType.UPLOAD_STARTED,
            ticker=ticker,
            document_id=document_id,
            payload={"file_count": len(files or [])},
        )
        yield UploadMaterialEvent(
            event_type=UploadMaterialEventType.UPLOAD_COMPLETED,
            ticker=ticker,
            document_id=document_id,
            payload={
                "result": {
                    "action": "upload_material",
                    "ticker": ticker,
                    "material_action": action,
                    "form_type": form_type,
                    "material_name": material_name,
                    "files": [str(item) for item in files] if files else [],
                    "document_id": document_id,
                    "internal_document_id": internal_document_id,
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "company_id": company_id,
                    "company_name": company_name,
                    "ticker_aliases": ticker_aliases,
                    "overwrite": overwrite,
                }
            },
        )

    def process(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """记录 process 调用。

        Args:
            ticker: 股票代码。
            overwrite: 是否覆盖。
            ci: 是否追加导出 CI 快照。
            document_ids: 可选文档 ID 列表。

        Returns:
            占位结果。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "overwrite": overwrite,
            "ci": ci,
            "document_ids": document_ids,
        }
        self.last_call = ("process", payload)
        return {"action": "process", **payload}

    async def process_stream(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: list[str] | None = None,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> AsyncIterator[ProcessEvent]:
        """记录 process_stream 调用并返回最小事件流。

        Args:
            ticker: 股票代码。
            overwrite: 是否覆盖。
            ci: 是否追加导出 CI 快照。
            document_ids: 可选文档 ID 列表。
            cancel_checker: 可选取消检查函数。

        Yields:
            最小预处理事件流。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "overwrite": overwrite,
            "ci": ci,
            "document_ids": document_ids,
            "cancel_checker": cancel_checker,
        }
        self.last_call = ("process_stream", payload)
        yield ProcessEvent(
            event_type=ProcessEventType.PIPELINE_STARTED,
            ticker=ticker,
            payload={"overwrite": overwrite, "ci": ci, "total_documents": 3},
        )
        yield ProcessEvent(
            event_type=ProcessEventType.PIPELINE_COMPLETED,
            ticker=ticker,
            payload={
                "result": _mock_process_result(
                    ticker=ticker,
                    overwrite=overwrite,
                    ci=ci,
                    document_ids=document_ids,
                )
            },
        )

    def process_filing(
        self,
        ticker: str,
        document_id: str,
        overwrite: bool = False,
        ci: bool = False,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """记录 process_filing 调用。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            overwrite: 是否覆盖。
            ci: 是否追加导出 CI 快照。

        Returns:
            占位结果。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "document_id": document_id,
            "overwrite": overwrite,
            "ci": ci,
            "cancel_checker": cancel_checker,
        }
        self.last_call = ("process_filing", payload)
        return {"action": "process_filing", **payload}

    def process_material(
        self,
        ticker: str,
        document_id: str,
        overwrite: bool = False,
        ci: bool = False,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """记录 process_material 调用。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            overwrite: 是否覆盖。
            ci: 是否追加导出 CI 快照。

        Returns:
            占位结果。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "document_id": document_id,
            "overwrite": overwrite,
            "ci": ci,
            "cancel_checker": cancel_checker,
        }
        self.last_call = ("process_material", payload)
        return {"action": "process_material", **payload}

class FakePipelineWithUploadFileEvents(FakePipeline):
    """用于验证上传逐文件回显的假 pipeline。"""

    async def upload_filing_stream(
        self,
        ticker: str,
        action: str | None,
        files: list[Path],
        fiscal_year: int,
        fiscal_period: str,
        amended: bool = False,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> AsyncIterator[UploadFilingEvent]:
        """返回包含文件级事件的财报上传事件流。

        Args:
            ticker: 股票代码。
            action: 上传动作。
            files: 文件列表。
            fiscal_year: 财年。
            fiscal_period: 财季/财年标记。
            amended: 是否修订版。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID。
            company_name: 公司名称。
            ticker_aliases: 可选 alias 列表。
            overwrite: 是否覆盖。

        Yields:
            上传事件。

        Raises:
            无。
        """

        payload = {
            "ticker": ticker,
            "action": action,
            "files": [str(item) for item in files],
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "amended": amended,
            "filing_date": filing_date,
            "report_date": report_date,
            "company_id": company_id,
            "company_name": company_name,
            "ticker_aliases": ticker_aliases,
            "overwrite": overwrite,
        }
        self.last_call = ("upload_filing_stream", payload)
        yield UploadFilingEvent(
            event_type=UploadFilingEventType.UPLOAD_STARTED,
            ticker=ticker,
            payload={"action": action, "file_count": len(files)},
        )
        for file_path in files:
            yield UploadFilingEvent(
                event_type=UploadFilingEventType.FILE_UPLOADED,
                ticker=ticker,
                payload={"name": file_path.name, "size": 1},
            )
        yield UploadFilingEvent(
            event_type=UploadFilingEventType.UPLOAD_COMPLETED,
            ticker=ticker,
            payload={
                "result": {
                    "pipeline": "sec",
                    "action": "upload_filing",
                    "status": "processed",
                    "ticker": ticker,
                    "filing_action": action,
                    "files": [str(item) for item in files],
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "amended": amended,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "company_id": company_id,
                    "company_name": company_name,
                    "ticker_aliases": ticker_aliases,
                    "overwrite": overwrite,
                }
            },
        )

    async def upload_material_stream(
        self,
        ticker: str,
        action: str | None,
        form_type: str,
        material_name: str,
        files: Optional[list[Path]] = None,
        document_id: Optional[str] = None,
        internal_document_id: Optional[str] = None,
        fiscal_year: int | None = None,
        fiscal_period: str | None = None,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> AsyncIterator[UploadMaterialEvent]:
        """返回包含文件级事件的材料上传事件流。

        Args:
            ticker: 股票代码。
            action: 上传动作。
            form_type: 材料类型。
            material_name: 材料名称。
            files: 文件列表。
            document_id: 文档 ID。
            internal_document_id: 内部文档 ID。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID。
            company_name: 公司名称。
            ticker_aliases: 可选 alias 列表。
            overwrite: 是否覆盖。

        Yields:
            上传事件。

        Raises:
            无。
        """

        file_list = files or []
        payload = {
            "ticker": ticker,
            "action": action,
            "form_type": form_type,
            "material_name": material_name,
            "files": [str(item) for item in file_list],
            "document_id": document_id,
            "internal_document_id": internal_document_id,
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "filing_date": filing_date,
            "report_date": report_date,
            "company_id": company_id,
            "company_name": company_name,
            "ticker_aliases": ticker_aliases,
            "overwrite": overwrite,
        }
        self.last_call = ("upload_material_stream", payload)
        yield UploadMaterialEvent(
            event_type=UploadMaterialEventType.UPLOAD_STARTED,
            ticker=ticker,
            document_id=document_id,
            payload={"action": action, "file_count": len(file_list)},
        )
        for file_path in file_list:
            yield UploadMaterialEvent(
                event_type=UploadMaterialEventType.FILE_UPLOADED,
                ticker=ticker,
                document_id=document_id,
                payload={"name": file_path.name, "size": 1},
            )
        yield UploadMaterialEvent(
            event_type=UploadMaterialEventType.UPLOAD_COMPLETED,
            ticker=ticker,
            document_id=document_id,
            payload={
                "result": {
                    "pipeline": "sec",
                    "action": "upload_material",
                    "status": "processed",
                    "ticker": ticker,
                    "material_action": action,
                    "form_type": form_type,
                    "material_name": material_name,
                    "files": [str(item) for item in file_list],
                    "document_id": document_id,
                    "internal_document_id": internal_document_id,
                    "fiscal_year": fiscal_year,
                    "fiscal_period": fiscal_period,
                    "filing_date": filing_date,
                    "report_date": report_date,
                    "company_id": company_id,
                    "company_name": company_name,
                    "ticker_aliases": ticker_aliases,
                    "overwrite": overwrite,
                }
            },
        )


class DummyDownloader:
    """用于测试 SecPipeline.download 的下载器桩。"""

    def __init__(self) -> None:
        """初始化调用记录。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.configure_called = False
        self.download_files_called = False

    def configure(self, user_agent: Optional[str], sleep_seconds: float, max_retries: int) -> None:
        """记录配置调用。

        Args:
            user_agent: User-Agent。
            sleep_seconds: 间隔秒数。
            max_retries: 重试次数。

        Returns:
            无。

        Raises:
            无。
        """

        self.configure_called = True

    def normalize_ticker(self, ticker: str) -> str:
        """标准化 ticker。

        Args:
            ticker: 股票代码。

        Returns:
            标准化 ticker。

        Raises:
            ValueError: ticker 为空时抛出。
        """

        normalized = ticker.strip().upper()
        if not normalized:
            raise ValueError("ticker 不能为空")
        return normalized

    async def resolve_company(self, ticker: str) -> tuple[str, str, str]:
        """返回固定公司信息。

        Args:
            ticker: 股票代码。

        Returns:
            `(cik, company_name, cik10)`。

        Raises:
            无。
        """

        return ("320193", "Test Inc.", "0000320193")

    async def fetch_submissions(self, cik10: str) -> dict[str, Any]:
        """返回固定 submissions 数据。

        Args:
            cik10: 10 位 CIK。

        Returns:
            submissions JSON。

        Raises:
            无。
        """

        return {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2025-02-01"],
                    "reportDate": ["2024-12-31"],
                    "accessionNumber": ["0000000000-25-000001"],
                    "primaryDocument": ["sample-10k.htm"],
                },
                "files": [],
            }
        }

    async def list_filing_files(
        self,
        cik: str,
        accession_no_dash: str,
        primary_document: str,
        form_type: str,
        include_xbrl: bool = True,
        include_exhibits: bool = True,
        include_http_metadata: bool = True,
    ) -> list[RemoteFileDescriptor]:
        """返回固定远端文件列表。

        Args:
            cik: CIK。
            accession_no_dash: accession。
            primary_document: primary 文件名。
            form_type: form。
            include_xbrl: 是否含 XBRL。
            include_exhibits: 是否含 exhibits。

        Returns:
            远端文件描述列表。

        Raises:
            无。
        """

        return [
            RemoteFileDescriptor(
                name="sample-10k.htm",
                source_url="https://example.com/sample-10k.htm",
                http_etag="etag",
                http_last_modified="Mon, 01 Jan 2025 00:00:00 GMT",
                remote_size=100,
                http_status=200,
            )
        ]

    async def download_files(
        self,
        remote_files: list[RemoteFileDescriptor],
        overwrite: bool,
        store_file: Any,
        existing_files: Optional[dict[str, dict[str, Any]]] = None,
        primary_document: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """返回下载结果并写入文件存储。

        Args:
            remote_files: 远端文件列表。
            overwrite: 是否覆盖。
            store_file: 文件存储回调。
            existing_files: 既有文件映射。

        Returns:
            下载结果列表。

        Raises:
            无。
        """

        self.download_files_called = True
        del overwrite, existing_files, primary_document
        file_meta = store_file(remote_files[0].name, BytesIO(b"dummy"))
        return [
            {
                "name": remote_files[0].name,
                "status": "downloaded",
                "file_meta": file_meta,
                "source_url": remote_files[0].source_url,
                "http_etag": remote_files[0].http_etag,
                "http_last_modified": remote_files[0].http_last_modified,
            }
        ]


def _mock_download_result(
    ticker: str,
    form_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    overwrite: bool = False,
    rebuild: bool = False,
) -> dict[str, Any]:
    """返回固定下载结果用于 CLI 输出测试。

    Args:
        ticker: 股票代码。
        form_type: 可选 form。
        start_date: 可选开始日期。
        end_date: 可选结束日期。
        overwrite: 是否覆盖。
        rebuild: 是否重建本地 meta/manifest。

    Returns:
        固定下载结果字典。

    Raises:
        无。
    """

    del form_type, start_date, end_date, overwrite, rebuild
    return {
        "pipeline": "sec",
        "action": "download",
        "ticker": ticker,
        "summary": {"total": 3, "downloaded": 1, "skipped": 1, "failed": 1, "elapsed_ms": 100},
        "filings": [
            {"document_id": "fil_a", "status": "downloaded", "form_type": "10-K"},
            {
                "document_id": "fil_b",
                "status": "skipped",
                "form_type": "10-Q",
                "reason_code": "not_modified",
                "reason_message": "所有文件均未修改，跳过重新下载",
            },
            {
                "document_id": "fil_c",
                "status": "failed",
                "form_type": "8-K",
                "reason_code": "file_download_failed",
                "reason_message": "network down",
            },
        ],
        "warnings": [],
    }


def _mock_process_result(
    *,
    ticker: str,
    overwrite: bool = False,
    ci: bool = False,
    document_ids: list[str] | None = None,
) -> dict[str, Any]:
    """返回固定预处理结果用于 CLI 输出测试。

    Args:
        ticker: 股票代码。
        overwrite: 是否覆盖。
        ci: 是否追加导出 CI 快照。
        document_ids: 可选文档 ID 列表。

    Returns:
        固定预处理结果字典。

    Raises:
        无。
    """

    return {
        "pipeline": "sec",
        "action": "process",
        "status": "ok",
        "ticker": ticker,
        "overwrite": overwrite,
        "ci": ci,
        "document_ids": document_ids,
        "filings": [
            {
                "document_id": "fil_1",
                "status": "processed",
                "form_type": "10-K",
                "fiscal_year": 2025,
                "section_count": 12,
            },
            {
                "document_id": "fil_2",
                "status": "skipped",
                "reason": "version_matched",
                "form_type": "10-Q",
                "fiscal_year": 2025,
            },
            {"document_id": "fil_3", "status": "failed", "reason": "parse_error"},
        ],
        "filing_summary": {"total": 3, "processed": 1, "skipped": 1, "failed": 1},
        "materials": [],
        "material_summary": {"total": 0, "processed": 0, "skipped": 0, "failed": 0, "todo": True},
    }


def test_factory_returns_sec_pipeline_for_us_market(tmp_path: Path) -> None:
    """验证 US 市场返回 `SecPipeline`。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    normalized = NormalizedTicker(canonical="AAPL", market="US", exchange=None, raw="AAPL")
    pipeline = get_pipeline_from_normalized_ticker(
        normalized_ticker=normalized,
        workspace_root=tmp_path,
    )
    assert isinstance(pipeline, SecPipeline)


def test_factory_returns_cn_pipeline_for_cn_market(tmp_path: Path) -> None:
    """验证 CN 市场返回 `CnPipeline`。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    normalized = NormalizedTicker(canonical="000333", market="CN", exchange="SZSE", raw="000333")
    pipeline = get_pipeline_from_normalized_ticker(
        normalized_ticker=normalized,
        workspace_root=tmp_path,
    )
    assert isinstance(pipeline, CnPipeline)


def test_factory_raises_for_unsupported_market(tmp_path: Path) -> None:
    """验证不支持的 market 会抛出错误。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    normalized = NormalizedTicker(canonical="ABC", market="US", exchange=None, raw="ABC")
    # 通过动态赋值模拟异常市场，验证工厂兜底分支。
    object.__setattr__(normalized, "market", "JP")
    with pytest.raises(ValueError, match="不支持的 market"):
        get_pipeline_from_normalized_ticker(
            normalized_ticker=normalized,
            workspace_root=tmp_path,
        )


def test_sec_pipeline_download_calls_sec_downloader(tmp_path: Path) -> None:
    """验证 `SecPipeline.download` 会调用 downloader 低层接口。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    dummy_downloader = DummyDownloader()
    pipeline = SecPipeline(
        workspace_root=tmp_path,
        downloader=cast(SecDownloader, dummy_downloader),
        processor_registry=build_fins_processor_registry(),
    )
    result = pipeline.download(
        ticker="AAPL",
        form_type="10-K",
        start_date="2024-01-01",
        end_date="2025-12-31",
        overwrite=True,
    )

    assert dummy_downloader.configure_called is True
    assert dummy_downloader.download_files_called is True
    assert result["action"] == "download"


def test_dispatch_download_routes_to_pipeline_download() -> None:
    """验证 `download` 子命令路由到 `Pipeline.download`。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    parser = cli._create_parser()
    args = parser.parse_args(["download", "--ticker", "AAPL", "--forms", "10-K"])
    fake = FakePipeline()
    result = cli._dispatch_action(fake, args)

    assert fake.last_call is not None
    assert fake.last_call[0] == "download"
    assert result["action"] == "download"
    assert result["form_type"] == "10-K"


def test_dispatch_download_passes_rebuild_flag() -> None:
    """验证 `download --rebuild` 会透传到 `Pipeline.download`。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    parser = cli._create_parser()
    args = parser.parse_args(["download", "--ticker", "AAPL", "--rebuild"])
    fake = FakePipeline()
    result = cli._dispatch_action(fake, args)

    assert result["action"] == "download"
    assert result["rebuild"] is True
    assert fake.last_call is not None
    assert fake.last_call[0] == "download"
    assert fake.last_call[1]["rebuild"] is True


def test_dispatch_download_supports_multiple_forms() -> None:
    """验证 `download` 子命令支持多 form 输入。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    parser = cli._create_parser()
    args = parser.parse_args(
        ["download", "--ticker", "AAPL", "--forms", "10Q", "10K", "DEF14A"]
    )
    fake = FakePipeline()
    result = cli._dispatch_action(fake, args)

    assert result["action"] == "download"
    assert result["form_type"] == "10Q,10K,DEF14A"


def test_dispatch_process_passes_ci_flag() -> None:
    """验证 `process` 子命令会透传 `--ci`。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    parser = cli._create_parser()
    args = parser.parse_args(["process", "--ticker", "AAPL", "--ci"])
    fake = FakePipeline()
    result = cli._dispatch_action(fake, args)

    assert result["action"] == "process"
    assert result["ci"] is True
    assert fake.last_call is not None
    assert fake.last_call[0] == "process"
    assert fake.last_call[1]["ci"] is True


def test_dispatch_process_filing_passes_ci_flag() -> None:
    """验证 `process_filing` 子命令会透传 `--ci`。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    parser = cli._create_parser()
    args = parser.parse_args(
        ["process_filing", "--ticker", "AAPL", "--document-id", "fil_1", "--ci"]
    )
    fake = FakePipeline()
    result = cli._dispatch_action(fake, args)

    assert result["action"] == "process_filing"
    assert result["ci"] is True
    assert fake.last_call is not None
    assert fake.last_call[0] == "process_filing"
    assert fake.last_call[1]["ci"] is True


def test_dispatch_process_material_passes_ci_flag() -> None:
    """验证 `process_material` 子命令会透传 `--ci`。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    parser = cli._create_parser()
    args = parser.parse_args(
        ["process_material", "--ticker", "AAPL", "--document-id", "mat_1", "--ci"]
    )
    fake = FakePipeline()
    result = cli._dispatch_action(fake, args)

    assert result["action"] == "process_material"
    assert result["ci"] is True
    assert fake.last_call is not None
    assert fake.last_call[0] == "process_material"
    assert fake.last_call[1]["ci"] is True


def test_parser_rejects_removed_processor_hint() -> None:
    """验证 `fins process` 不再接受 `--processor-hint`。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    parser = cli._create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["process", "--ticker", "AAPL", "--processor-hint", "bs"])


def test_dispatch_upload_filing_uses_default_action_create(tmp_path: Path) -> None:
    """验证 `upload_filing` 默认 action 为 create。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    filing_path = tmp_path / "filing.pdf"
    filing_path.write_text("content", encoding="utf-8")
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filing",
            "--ticker",
            "AAPL",
            "--files",
            str(filing_path),
            "--fiscal-year",
            "2025",
            "--fiscal-period",
            "FY",
            "--company-name",
            "Apple Inc.",
        ]
    )
    fake = FakePipeline()
    result = cli._dispatch_action(fake, args)

    assert result["action"] == "upload_filing"
    assert result["ticker"] == "AAPL"
    assert result["fiscal_period"] == "FY"
    assert fake.last_call is not None
    assert fake.last_call[0] == "upload_filing"
    assert fake.last_call[1]["action"] is None
    assert fake.last_call[1]["company_id"] is None
    assert fake.last_call[1]["company_name"] == "Apple Inc."


def test_dispatch_upload_filing_delete_does_not_require_files() -> None:
    """验证 `upload_filing` 在 delete 动作下不要求 --files。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filing",
            "--ticker",
            "AAPL",
            "--action",
            "delete",
            "--fiscal-year",
            "2025",
            "--fiscal-period",
            "FY",
        ]
    )
    result = cli._dispatch_action(FakePipeline(), args)
    assert result["action"] == "upload_filing"


def test_dispatch_download_passes_csv_aliases_to_pipeline() -> None:
    """验证 `download` 会把 ticker CSV 解析为 canonical ticker 与 alias。"""

    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "download",
            "--ticker",
            "BABA,9988,9988.HK",
        ]
    )

    fake = FakePipeline()
    result = cli._dispatch_action(fake, args)

    assert result["ticker"] == "BABA"
    assert fake.last_call is not None
    assert fake.last_call[1]["ticker"] == "BABA"
    # CSV 每个 token 都归一化后去重：`9988.HK` canonical=`9988` 与前者重复，被去掉。
    assert fake.last_call[1]["ticker_aliases"] == ["BABA", "9988"]


def test_dispatch_download_infer_merges_explicit_and_fmp_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `download --infer` 成功时合并显式 CSV alias 与 FMP alias。"""

    monkeypatch.setattr(
        cli,
        "infer_company_aliases_from_fmp",
        lambda ticker: FmpAliasInferenceResult(
            canonical_ticker=ticker,
            company_name="Alibaba Group Holding Limited",
            ticker_aliases=["BABA", "BABAF", "9988.HK"],
        ),
    )
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "download",
            "--ticker",
            "BABA,9988,9988.HK",
            "--infer",
        ]
    )

    fake = FakePipeline()
    cli._dispatch_action(fake, args)

    assert fake.last_call is not None
    # CSV token 与 FMP alias 都归一化后整体去重：`9988.HK`→`9988`，`BABAF` 不是合法 ticker 回退大写。
    assert fake.last_call[1]["ticker_aliases"] == ["BABA", "9988", "BABAF"]


def test_dispatch_download_infer_failure_falls_back_to_explicit_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `download --infer` 失败时回退到显式 CSV alias。"""

    def _raise_infer_error(ticker: str) -> FmpAliasInferenceResult:
        del ticker
        raise cli.FmpAliasInferenceError("boom")

    monkeypatch.setattr(cli, "infer_company_aliases_from_fmp", _raise_infer_error)
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "download",
            "--ticker",
            "BABA,9988,9988.HK",
            "--infer",
        ]
    )

    fake = FakePipeline()
    cli._dispatch_action(fake, args)

    assert fake.last_call is not None
    assert fake.last_call[1]["ticker_aliases"] == ["BABA", "9988"]


def test_dispatch_upload_filing_create_requires_company_meta_when_meta_missing(
    tmp_path: Path,
) -> None:
    """验证 `upload_filing` 在缺少现有 meta 且未 infer 成功时会强制要求 company meta。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    filing_path = tmp_path / "filing.pdf"
    filing_path.write_text("content", encoding="utf-8")
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filing",
            "--ticker",
            "AAPL",
            "--base",
            str(tmp_path),
            "--action",
            "create",
            "--files",
            str(filing_path),
            "--fiscal-year",
            "2025",
            "--fiscal-period",
            "FY",
        ]
    )
    with pytest.raises(ValueError, match="company-name"):
        cli._dispatch_action(FakePipeline(), args)


def test_dispatch_upload_filing_infer_merges_aliases_and_preserves_explicit_company_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `upload_filing --infer` 成功时合并 alias，且显式 company_name 优先。"""

    monkeypatch.setattr(
        cli,
        "infer_company_aliases_from_fmp",
        lambda ticker: FmpAliasInferenceResult(
            canonical_ticker=ticker,
            company_name="Alibaba Group Holding Limited",
            ticker_aliases=["BABA", "BABAF", "9988.HK"],
        ),
    )
    filing_path = tmp_path / "filing.pdf"
    filing_path.write_text("content", encoding="utf-8")
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filing",
            "--ticker",
            "BABA,9988",
            "--base",
            str(tmp_path),
            "--infer",
            "--files",
            str(filing_path),
            "--fiscal-year",
            "2025",
            "--fiscal-period",
            "FY",
            "--company-name",
            "阿里巴巴",
        ]
    )

    fake = FakePipeline()
    result = cli._dispatch_action(fake, args)

    assert result["ticker"] == "BABA"
    assert result["company_name"] == "阿里巴巴"
    assert fake.last_call is not None
    assert fake.last_call[1]["ticker_aliases"] == ["BABA", "9988", "BABAF"]
    assert fake.last_call[1]["company_name"] == "阿里巴巴"


def test_dispatch_upload_filing_infer_fills_missing_company_name_and_merges_aliases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `upload_filing --infer` 在未传 company_name 时会补齐名称并合并 alias。"""

    monkeypatch.setattr(
        cli,
        "infer_company_aliases_from_fmp",
        lambda ticker: FmpAliasInferenceResult(
            canonical_ticker=ticker,
            company_name="Alibaba Group Holding Limited",
            ticker_aliases=["BABA", "BABAF", "9988.HK"],
        ),
    )
    filing_path = tmp_path / "filing.pdf"
    filing_path.write_text("content", encoding="utf-8")
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filing",
            "--ticker",
            "BABA,9988",
            "--base",
            str(tmp_path),
            "--infer",
            "--files",
            str(filing_path),
            "--fiscal-year",
            "2025",
            "--fiscal-period",
            "FY",
        ]
    )

    fake = FakePipeline()
    result = cli._dispatch_action(fake, args)

    assert result["company_name"] == "Alibaba Group Holding Limited"
    assert fake.last_call is not None
    assert fake.last_call[1]["ticker_aliases"] == ["BABA", "9988", "BABAF"]
    assert fake.last_call[1]["company_name"] == "Alibaba Group Holding Limited"


def test_dispatch_upload_filing_infer_failure_without_company_name_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `upload_filing --infer` 失败且缺少 company_name 时直接失败。"""

    def _raise_infer_error(ticker: str) -> FmpAliasInferenceResult:
        del ticker
        raise cli.FmpAliasInferenceError("boom")

    monkeypatch.setattr(cli, "infer_company_aliases_from_fmp", _raise_infer_error)
    filing_path = tmp_path / "filing.pdf"
    filing_path.write_text("content", encoding="utf-8")
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filing",
            "--ticker",
            "BABA,9988",
            "--base",
            str(tmp_path),
            "--infer",
            "--files",
            str(filing_path),
            "--fiscal-year",
            "2025",
            "--fiscal-period",
            "FY",
        ]
    )

    with pytest.raises(ValueError, match="company-name"):
        cli._dispatch_action(FakePipeline(), args)


def test_dispatch_upload_filings_from_generates_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `upload_filings_from` 可识别文件名并生成脚本。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(cli, "_get_current_upload_script_platform", lambda: "linux")

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "2025Q1美的集团一季度报告.pdf").write_text("a", encoding="utf-8")
    (source_dir / "2023Q4美的集团年度报告.pdf").write_text("b", encoding="utf-8")
    (source_dir / "2025Q2美的集团半年度报告.pdf").write_text("c", encoding="utf-8")
    (source_dir / "说明文档.txt").write_text("d", encoding="utf-8")
    output_script = tmp_path / "upload_000333.sh"

    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker",
            "000333",
            "--from",
            str(source_dir),
            "--base",
            str(tmp_path),
            "--output",
            str(output_script),
            "--company-name",
            "美的集团",
        ]
    )
    result = cli._dispatch_action(FakePipeline(), args)

    assert result["action"] == "upload_filings_from"
    assert result["recognized_count"] == 3
    assert result["skipped_count"] == 1
    assert output_script.exists()
    script_text = output_script.read_text(encoding="utf-8")
    command_lines = [line for line in script_text.splitlines() if line.startswith("python -m dayu.cli upload_")]
    assert "--fiscal-year 2025 --fiscal-period Q1" in script_text
    assert "--fiscal-year 2023 --fiscal-period FY" in script_text
    assert "--fiscal-year 2025 --fiscal-period H1" in script_text
    assert "--action create" not in script_text
    assert "--company-id" not in "\n".join(command_lines)
    assert "\n".join(command_lines).count("--company-name") == 1
    assert all(line.endswith('"$@"') for line in command_lines)
    assert "# python -m dayu.cli upload_filings_from" in script_text


def test_dispatch_upload_filings_from_output_directory_writes_default_script_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `upload_filings_from` 在 --output 传目录时会写入默认脚本名。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr(cli, "_get_current_upload_script_platform", lambda: "linux")

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "2025Q1美的集团一季度报告.pdf").write_text("a", encoding="utf-8")
    output_dir = tmp_path / "workspace"
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker",
            "0300",
            "--from",
            str(source_dir),
            "--base",
            str(tmp_path),
            "--output",
            str(output_dir),
            "--company-name",
            "美的集团",
        ]
    )
    result = cli._dispatch_action(FakePipeline(), args)
    expected_script = output_dir / "upload_filings_0300.sh"

    assert result["script_path"] == str(expected_script.resolve())
    assert expected_script.exists()
    script_text = expected_script.read_text(encoding="utf-8")
    command_lines = [line for line in script_text.splitlines() if line.startswith("python -m dayu.cli upload_")]
    assert "--ticker 0300" in script_text
    assert "--company-id" not in "\n".join(command_lines)
    assert "\n".join(command_lines).count("--company-name") == 1


def test_dispatch_upload_filings_from_without_output_writes_default_script_to_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `upload_filings_from` 未传 --output 时默认写到 workspace 根目录。"""

    monkeypatch.setattr(cli, "_get_current_upload_script_platform", lambda: "linux")

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "2025Q1美的集团一季度报告.pdf").write_text("a", encoding="utf-8")

    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker",
            "0300",
            "--from",
            str(source_dir),
            "--base",
            str(tmp_path),
            "--company-name",
            "美的集团",
        ]
    )
    result = cli._dispatch_action(FakePipeline(), args)
    expected_script = tmp_path / "upload_filings_0300.sh"

    assert result["script_path"] == str(expected_script.resolve())
    assert expected_script.exists()
    assert not (source_dir / "upload_filings_0300.sh").exists()


def test_dispatch_upload_filings_from_windows_output_directory_writes_cmd_script(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `upload_filings_from` 在 Windows 平台生成 `.cmd` 脚本。"""

    monkeypatch.setattr(cli, "_get_current_upload_script_platform", lambda: "windows")

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "2025Q1美的集团一季度报告.pdf").write_text("a", encoding="utf-8")
    output_dir = tmp_path / "workspace"
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker",
            "0300",
            "--from",
            str(source_dir),
            "--base",
            str(tmp_path),
            "--output",
            str(output_dir),
            "--company-name",
            "美的集团",
        ]
    )
    result = cli._dispatch_action(FakePipeline(), args)
    expected_script = output_dir / "upload_filings_0300.cmd"

    assert result["script_path"] == str(expected_script.resolve())
    assert result["script_platform"] == "windows"
    assert expected_script.exists()
    script_text = expected_script.read_text(encoding="utf-8")
    assert script_text.startswith("@echo off")
    assert "REM python -m dayu.cli upload_filings_from" in script_text


def test_dispatch_upload_filings_from_requires_company_meta(tmp_path: Path) -> None:
    """验证 `upload_filings_from` 在 create/update 动作下强制要求 company meta。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "2025Q1示例报告.pdf").write_text("demo", encoding="utf-8")
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker",
            "000333",
            "--from",
            str(source_dir),
            "--base",
            str(tmp_path),
        ]
    )
    with pytest.raises(ValueError, match="company-name"):
        cli._dispatch_action(FakePipeline(), args)


def test_dispatch_upload_filings_from_infer_bakes_result_into_generated_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `upload_filings_from --infer` 只推断一次，并合并 alias 后 bake 到脚本命令。"""

    infer_calls: list[str] = []

    def _fake_infer(ticker: str) -> FmpAliasInferenceResult:
        infer_calls.append(ticker)
        return FmpAliasInferenceResult(
            canonical_ticker=ticker,
            company_name="Alibaba Group Holding Limited",
            ticker_aliases=["BABA", "9988.HK", "89988.HK"],
        )

    monkeypatch.setattr(cli, "infer_company_aliases_from_fmp", _fake_infer)
    monkeypatch.setattr(cli, "_get_current_upload_script_platform", lambda: "linux")

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "2025Q1阿里巴巴一季度报告.pdf").write_text("demo", encoding="utf-8")
    output_script = tmp_path / "upload_baba.sh"

    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker",
            "BABA,9988",
            "--from",
            str(source_dir),
            "--base",
            str(tmp_path),
            "--output",
            str(output_script),
            "--company-name",
            "阿里巴巴",
            "--infer",
        ]
    )

    result = cli._dispatch_action(FakePipeline(), args)

    assert infer_calls == ["BABA"]
    assert result["generated_ticker_csv"] == "BABA,9988,89988"
    script_text = output_script.read_text(encoding="utf-8")
    assert "# python -m dayu.cli upload_filings_from --ticker BABA,9988" in script_text
    assert "--infer" in script_text
    assert "upload_filing" in script_text
    assert "--ticker BABA,9988,89988" in script_text
    assert "--company-name '阿里巴巴'" in script_text
    assert script_text.count("--infer") == 1


def test_dispatch_upload_filings_from_infer_failure_without_company_name_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 `upload_filings_from --infer` 失败且缺少 company_name 时直接失败。"""

    def _raise_infer_error(ticker: str) -> FmpAliasInferenceResult:
        del ticker
        raise cli.FmpAliasInferenceError("boom")

    monkeypatch.setattr(cli, "infer_company_aliases_from_fmp", _raise_infer_error)
    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "2025Q1阿里巴巴一季度报告.pdf").write_text("demo", encoding="utf-8")
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker",
            "BABA,9988",
            "--from",
            str(source_dir),
            "--base",
            str(tmp_path),
            "--infer",
        ]
    )

    with pytest.raises(ValueError, match="company-name"):
        cli._dispatch_action(FakePipeline(), args)


def test_dispatch_upload_filings_from_omits_company_meta_when_existing_meta_present(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证 `upload_filings_from` 在已有 meta 时忽略传入 company meta，且脚本不再附带该参数。"""

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "2025Q1示例报告.pdf").write_text("demo", encoding="utf-8")
    repo = cli.FsCompanyMetaRepository(tmp_path)
    repo.upsert_company_meta(
        CompanyMeta(
            company_id="000333",
            company_name="美的集团",
            ticker="000333",
            market="CN",
            resolver_version="market_resolver_v1",
            updated_at="2026-03-13T00:00:00+00:00",
        )
    )
    output_script = tmp_path / "upload_000333.sh"

    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker",
            "000333",
            "--from",
            str(source_dir),
            "--base",
            str(tmp_path),
            "--output",
            str(output_script),
            "--company-name",
            "错误名称",
        ]
    )

    with caplog.at_level(logging.WARNING):
        result = cli._dispatch_action(FakePipeline(), args)

    assert result["recognized_count"] == 1
    script_text = output_script.read_text(encoding="utf-8")
    assert "--company-id" not in script_text
    assert "--company-name" not in script_text
    assert "将忽略本次传入的 company_id/company_name" in caplog.text


def test_main_upload_filings_from_does_not_require_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 `upload_filings_from` 走独立分支，不构建 pipeline。

    Args:
        monkeypatch: monkeypatch fixture。
        tmp_path: 临时目录。
        capsys: 输出捕获 fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_dir = tmp_path / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "2025Q3美的集团三季度报告.pdf").write_text("x", encoding="utf-8")
    output_script = tmp_path / "upload_000333.sh"
    monkeypatch.setattr(
        cli,
        "_build_pipeline_for_ticker",
        lambda ticker, workspace_root, **kwargs: (_ for _ in ()).throw(AssertionError("不应构建 pipeline")),
    )

    exit_code = cli.main(
        [
            "upload_filings_from",
            "--ticker",
            "000333",
            "--from",
            str(source_dir),
            "--base",
            str(tmp_path),
            "--output",
            str(output_script),
            "--company-name",
            "美的集团",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "批量上传脚本生成结果" in output
    assert "recognized=1" in output
    assert "识别成功的文件 (upload_filing):" in output
    assert "跳过的文件:" in output
    assert output_script.exists()


def test_filter_upload_entries_annual_limit(tmp_path: Path) -> None:
    """验证年报超过5份时只保留最新5份。

    Args:
        tmp_path: 临时目录（占位，保持 fixture 签名一致）。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    entries = [
        {"file": f"fy{y}.pdf", "fiscal_year": y, "fiscal_period": "FY", "command": f"cmd{y}"}
        for y in range(2018, 2026)  # 8份年报
    ]
    kept, dropped = cli._filter_upload_entries(entries)
    kept_years = {e["fiscal_year"] for e in kept}
    assert len(kept) == 5
    assert kept_years == {2025, 2024, 2023, 2022, 2021}
    assert len(dropped) == 3


def test_filter_upload_entries_periodic_only_latest_year(tmp_path: Path) -> None:
    """验证季报/半年报只取最新一年内最多3份，且按财期升序排列。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    entries = [
        {"file": "2024Q1.pdf", "fiscal_year": 2024, "fiscal_period": "Q1", "command": "c1"},
        {"file": "2024H1.pdf", "fiscal_year": 2024, "fiscal_period": "H1", "command": "c2"},
        {"file": "2024Q3.pdf", "fiscal_year": 2024, "fiscal_period": "Q3", "command": "c3"},
        {"file": "2023Q1.pdf", "fiscal_year": 2023, "fiscal_period": "Q1", "command": "c4"},
        {"file": "2023H1.pdf", "fiscal_year": 2023, "fiscal_period": "H1", "command": "c5"},
    ]
    kept, dropped = cli._filter_upload_entries(entries)
    kept_periods = [(e["fiscal_year"], e["fiscal_period"]) for e in kept]
    # 只保留 2024 年的3份，按财期升序
    assert kept_periods == [(2024, "Q1"), (2024, "H1"), (2024, "Q3")]
    assert len(dropped) == 2
    assert all(e["fiscal_year"] == 2023 for e in dropped)


def test_filter_upload_entries_mixed(tmp_path: Path) -> None:
    """验证年报+周期性报告同时存在时，各自独立过滤。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    entries = [
        # 6份年报，应保留最新5份
        *[{"file": f"fy{y}.pdf", "fiscal_year": y, "fiscal_period": "FY", "command": f"a{y}"} for y in range(2019, 2025)],
        # 2024年内2份季报
        {"file": "2024Q1.pdf", "fiscal_year": 2024, "fiscal_period": "Q1", "command": "q1"},
        {"file": "2024H1.pdf", "fiscal_year": 2024, "fiscal_period": "H1", "command": "h1"},
    ]
    kept, dropped = cli._filter_upload_entries(entries)
    annual_kept = [e for e in kept if e["fiscal_period"] == "FY"]
    periodic_kept = [e for e in kept if e["fiscal_period"] != "FY"]
    assert len(annual_kept) == 5
    assert {e["fiscal_year"] for e in annual_kept} == {2020, 2021, 2022, 2023, 2024}
    assert len(periodic_kept) == 2
    assert len(dropped) == 1  # 2019年报被截断


def test_dispatch_upload_filings_from_applies_filter(tmp_path: Path) -> None:
    """验证 upload_filings_from 会对超限条目写入 skipped 并只生成保留条目的命令。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    # 创建6份年报（超过5份上限）
    for y in range(2019, 2025):
        (source_dir / f"{y}年报.pdf").write_text("x", encoding="utf-8")
    output_script = tmp_path / "upload.sh"

    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker", "TEST",
            "--from", str(source_dir),
            "--base", str(tmp_path),
            "--output", str(output_script),
            "--company-name", "测试公司",
        ]
    )
    result = cli._generate_upload_filings_script(cast(cli.UploadFilingsScriptArgsProtocol, args))

    assert result["recognized_count"] == 5
    assert result["skipped_count"] == 1
    # 被截断的那份应有超出上限的 reason
    skipped_reasons = [s["reason"] for s in result["skipped"]]
    assert any("超出收集上限" in r for r in skipped_reasons)


def test_detect_year_subdir_layout_true(tmp_path: Path) -> None:
    """验证含 20XX 子目录时检测结果为 True。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    (tmp_path / "2024").mkdir()
    (tmp_path / "2025").mkdir()
    (tmp_path / "old").mkdir()
    assert cli._detect_year_subdir_layout(tmp_path) is True


def test_detect_year_subdir_layout_true_with_quarter_subdir(tmp_path: Path) -> None:
    """验证含 20XXQ1 风格子目录时检测结果为 True。

    港股典型布局：2025Q1/、2025Q2/ 而非 2025/，应同样触发自动递归扫描。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    (tmp_path / "2025Q1").mkdir()
    (tmp_path / "2025Q2").mkdir()
    (tmp_path / "old").mkdir()
    assert cli._detect_year_subdir_layout(tmp_path) is True


def test_detect_year_subdir_layout_false(tmp_path: Path) -> None:
    """验证不含 20XX 子目录时检测结果为 False。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    (tmp_path / "old").mkdir()
    (tmp_path / "docs").mkdir()
    assert cli._detect_year_subdir_layout(tmp_path) is False


def test_collect_upload_from_files_auto_recursive_on_year_subdirs(tmp_path: Path) -> None:
    """验证年份子目录布局时自动启用递归扫描，无需传入 --recursive。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    year_dir = tmp_path / "2024"
    year_dir.mkdir()
    pdf = year_dir / "2024Q2 业绩报告.pdf"
    pdf.write_text("x", encoding="utf-8")
    # 未传 recursive=True，但因为有年份子目录应自动扫描
    found = cli._collect_upload_from_files(source_dir=tmp_path, recursive=False)
    assert len(found) == 1


def test_collect_upload_from_files_auto_recursive_on_year_quarter_subdirs(tmp_path: Path) -> None:
    """验证年季子目录（20XXQ1/）布局时自动启用递归扫描。

    港股布局：_0700腾讯/ 下有 2025Q1/、2025Q2/、2025Q3/ 等目录，
    应自动触发递归扫描，无需 --recursive。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    q1_dir = tmp_path / "2025Q1"
    q1_dir.mkdir()
    (q1_dir / "2025Q1业绩公告.pdf").write_text("x", encoding="utf-8")
    (q1_dir / "2025Q1业绩演示.pdf").write_text("x", encoding="utf-8")
    q2_dir = tmp_path / "2025Q2"
    q2_dir.mkdir()
    (q2_dir / "2025Q2业绩公告.pdf").write_text("x", encoding="utf-8")
    found = cli._collect_upload_from_files(source_dir=tmp_path, recursive=False)
    assert len(found) == 3


def test_dispatch_upload_filings_from_year_subdir_layout(tmp_path: Path) -> None:
    """验证 upload_filings_from 对年份子目录布局自动识别并生成正确命令。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 模拟 _0388港交所 目录结构
    for year, period, title in [
        ("2024", "Q2", "业绩报告"),
        ("2024", "Q3", "业绩公告"),
        ("2025", "Q1", "业绩公告"),
        ("2025", "Q2", "业绩报告"),
    ]:
        d = tmp_path / year
        d.mkdir(exist_ok=True)
        (d / f"{year}{period} {title}.pdf").write_text("x", encoding="utf-8")

    output_script = tmp_path / "upload.sh"
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker", "0388",
            "--from", str(tmp_path),
            "--base", str(tmp_path),
            "--output", str(output_script),
            "--company-name", "香港交易所",
        ]
    )
    result = cli._generate_upload_filings_script(cast(cli.UploadFilingsScriptArgsProtocol, args))

    # 因 _UPLOAD_MAX_PERIODIC=6，2025年内有2条记录，全部保留
    recognized_periods = [(e["fiscal_year"], e["fiscal_period"]) for e in result["recognized"]]
    assert (2025, "Q1") in recognized_periods
    assert (2025, "Q2") in recognized_periods
    assert result["recognized_count"] == 2
    assert output_script.exists()


@pytest.mark.unit
def test_validate_upload_material_rejects_invalid_form_type(tmp_path: Path) -> None:
    """验证 `_validate_upload_material_args` 拒绝不合法的 form_type。

    背景：早期 --forms 不做合法值检查，导致 typo（如 earning_calls）被悄悄写入
    manifest，使 document_type 映射错误。本测试确保不合法值在 CLI 入口被拒绝。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    file_path = tmp_path / "material.pdf"
    file_path.write_text("content", encoding="utf-8")
    parser = cli._create_parser()
    # earning_calls 是典型 typo（少了 's'，且 _EARNINGS_CALL 键需大写）
    args = parser.parse_args(
        [
            "upload_material",
            "--ticker", "FUTU",
            "--action", "create",
            "--forms", "earning_calls",
            "--material-name", "Q1 Call",
            "--files", str(file_path),
            "--company-name", "Futu",
        ]
    )
    with pytest.raises(ValueError, match="form_type"):
        cli._validate_upload_material_args(args)


@pytest.mark.unit
def test_validate_upload_material_normalizes_form_type_to_upper(tmp_path: Path) -> None:
    """验证 `_validate_upload_material_args` 将合法的小写 form_type 规范化为大写。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    file_path = tmp_path / "material.pdf"
    file_path.write_text("content", encoding="utf-8")
    parser = cli._create_parser()
    # 合法值小写传入
    args = parser.parse_args(
        [
            "upload_material",
            "--ticker", "FUTU",
            "--action", "create",
            "--forms", "earnings_call",
            "--material-name", "Q1 Call",
            "--files", str(file_path),
            "--company-name", "Futu",
        ]
    )
    cli._validate_upload_material_args(args)
    # 规范化后的 form_type 应为大写
    assert args.form_type == "EARNINGS_CALL"


def test_dispatch_upload_material_allows_update_without_explicit_document_id(tmp_path: Path) -> None:
    """验证 `upload_material update` 可仅凭稳定规则定位目标文档。"""

    file_path = tmp_path / "material.pdf"
    file_path.write_text("content", encoding="utf-8")
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_material",
            "--ticker",
            "MSFT",
            "--action",
            "update",
            "--forms",
            "MATERIAL_OTHER",
            "--material-name",
            "Deck",
            "--files",
            str(file_path),
            "--company-name",
            "Apple Inc.",
        ]
    )
    result = cli._dispatch_action(FakePipeline(), args)
    assert result["action"] == "upload_material"
    assert result["material_action"] == "update"


def test_dispatch_upload_material_passes_optional_fiscal_fields(tmp_path: Path) -> None:
    """验证 `upload_material` 会把可选 fiscal 字段传给 pipeline。"""

    file_path = tmp_path / "material.pdf"
    file_path.write_text("content", encoding="utf-8")
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_material",
            "--ticker",
            "MSFT",
            "--forms",
            "MATERIAL_OTHER",
            "--material-name",
            "Deck",
            "--files",
            str(file_path),
            "--fiscal-year",
            "2025",
            "--fiscal-period",
            "q1",
            "--company-name",
            "Apple Inc.",
        ]
    )
    fake = FakePipeline()
    cli._dispatch_action(fake, args)
    assert fake.last_call is not None
    assert fake.last_call[0] == "upload_material"
    assert fake.last_call[1]["action"] is None
    assert fake.last_call[1]["fiscal_year"] == 2025
    assert fake.last_call[1]["fiscal_period"] == "Q1"


def test_dispatch_upload_material_create_requires_company_meta_when_meta_missing(
    tmp_path: Path,
) -> None:
    """验证 `upload_material` 在缺少现有 meta 且未 infer 成功时会强制要求 company meta。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    file_path = tmp_path / "material.pdf"
    file_path.write_text("content", encoding="utf-8")
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_material",
            "--ticker",
            "AAPL",
            "--base",
            str(tmp_path),
            "--action",
            "create",
            "--forms",
            "MATERIAL_OTHER",
            "--material-name",
            "Deck",
            "--files",
            str(file_path),
        ]
    )
    with pytest.raises(ValueError, match="company-name"):
        cli._dispatch_action(FakePipeline(), args)


# ---------------------------------------------------------------------------
# Q4 / FY 消歧 + 主报告优先级 + 财务报表 material 测试
# ---------------------------------------------------------------------------


class TestInferFiscalPeriodQ4Disambiguation:
    """验证 Q4 文件名与 FY 的消歧逻辑。"""

    def test_q4_without_marker_becomes_fy(self) -> None:
        """Q4 + 无「季报」关键词 → FY。"""
        period = cli._infer_fiscal_period_from_filename("2024Q4 业绩报告.pdf")
        assert period == "FY"

    def test_q4_with_quarterly_marker_stays_q4(self) -> None:
        """Q4 + 含「季报」→ Q4。"""
        period = cli._infer_fiscal_period_from_filename("2024Q4季报.pdf")
        assert period == "Q4"

    def test_q4_with_annual_keyword_already_fy_via_fy_pattern(self) -> None:
        """文件名同时含 Q4 和「年报」→ FY（由 FY_PATTERN 优先）。"""
        period = cli._infer_fiscal_period_from_filename("2024Q4年报.pdf")
        assert period == "FY"

    def test_q4_业绩公告_becomes_fy(self) -> None:
        """Q4 + 业绩公告（无季报）→ FY。"""
        period = cli._infer_fiscal_period_from_filename("2024Q4 HKEX业绩公告.pdf")
        assert period == "FY"

    def test_non_q4_period_unaffected(self) -> None:
        """非 Q4 文件保持原有期间。"""
        assert cli._infer_fiscal_period_from_filename("2024Q3 三季报.pdf") == "Q3"
        assert cli._infer_fiscal_period_from_filename("2024H1 中期报告.pdf") == "H1"

    def test_standalone_中期报告_recognized_as_h1(self) -> None:
        """无财期前缀的独立「中期报告」文件（如腾讯根目录）应识别为 H1。"""
        period = cli._infer_fiscal_period_from_filename("2023 中期报告.PDF")
        assert period == "H1"

    def test_q2_中期报告_recognized_as_h1_via_h1_pattern(self) -> None:
        """Q2 前缀的中期报告：H1_PATTERN（含中期报告）先于 Q2_PATTERN 检查，返回 H1。"""
        period = cli._infer_fiscal_period_from_filename("2018Q2 0700 中期报告.PDF")
        assert period == "H1"

    def test_year_digit_does_not_pollute_q_pattern_2022q4(self) -> None:
        """2022Q4 文件名中「2Q」不应误匹配为 Q2。Q4_PATTERN 正确返回 FY（无季报标记）。"""
        period = cli._infer_fiscal_period_from_filename("2022Q4 腾讯 业绩公告.PDF")
        assert period == "FY"

    def test_year_digit_does_not_pollute_q_pattern_2023q4(self) -> None:
        """2023Q4 文件名中「3Q」不应误匹配为 Q3。Q4_PATTERN 正确返回 FY（无季报标记）。"""
        period = cli._infer_fiscal_period_from_filename("2023Q4 腾讯 业绩公告.pdf")
        assert period == "FY"

    def test_year_digit_does_not_pollute_q_pattern_2021q4(self) -> None:
        """2021Q4 文件名中「1Q」不应误匹配为 Q1。Q4_PATTERN 正确返回 Q4（含季报标记）。"""
        period = cli._infer_fiscal_period_from_filename("2021Q4 0700季报.pdf")
        assert period == "Q4"


class TestComputeMainReportPriority:
    """验证自适应主报告优先级计算。"""

    def test_年度报告_is_tier0(self) -> None:
        """「年度报告」属于长覆盖正式报告层（层 0）。"""
        assert cli._compute_main_report_priority("2024年度报告.pdf") == 0

    def test_中期报告_is_tier0(self) -> None:
        """「中期报告」属于长覆盖正式报告层（层 0），与年度报告同级。"""
        assert cli._compute_main_report_priority("2024中期报告.pdf") == 0

    def test_年报_is_tier0(self) -> None:
        """「年报」为年度报告简称，属于层 0。"""
        assert cli._compute_main_report_priority("2024年报.pdf") == 0

    def test_中报_is_tier0(self) -> None:
        """「中报」为中期报告简称，属于层 0。"""
        assert cli._compute_main_report_priority("2024中报.pdf") == 0

    def test_季报_is_tier1(self) -> None:
        """「季报」属于季度正式报告层（层 1）。"""
        assert cli._compute_main_report_priority("2024Q3 季报.pdf") == 1

    def test_业绩报告_is_tier2(self) -> None:
        """「业绩报告」含"报告"但无年度/中期/季度 scope，属于通用报告层（层 2）。"""
        assert cli._compute_main_report_priority("2024业绩报告.pdf") == 2

    def test_业绩公告_is_tier3(self) -> None:
        """「业绩公告」属于公告/通告层（层 3）。"""
        assert cli._compute_main_report_priority("2024Q4 业绩公告.pdf") == 3

    def test_unknown_is_tier4(self) -> None:
        """无任何识别关键词的文件属于未知层（层 4）。"""
        assert cli._compute_main_report_priority("2024其他文件.pdf") == 4

    def test_演示_is_tier5(self) -> None:
        """「业绩演示」属于补充材料层（层 5）。"""
        assert cli._compute_main_report_priority("2024Q1 业绩演示.pdf") == 5

    def test_tier0_beats_tier3(self) -> None:
        """长覆盖正式报告（层 0）优先级高于公告类（层 3）。"""
        assert (
            cli._compute_main_report_priority("2024年度报告.pdf")
            < cli._compute_main_report_priority("2024业绩公告.pdf")
        )

    def test_中期报告_beats_季报(self) -> None:
        """「中期报告」（层 0）优先级高于「季报」（层 1）——腾讯 Q2 场景。"""
        assert (
            cli._compute_main_report_priority("2024Q2 中期报告.pdf")
            < cli._compute_main_report_priority("2024Q2 季报.pdf")
        )

    def test_中期报告_beats_业绩公告(self) -> None:
        """「中期报告」（层 0）优先级高于「业绩公告」（层 3）——腾讯 Q2 场景。"""
        assert (
            cli._compute_main_report_priority("2024Q2 中期报告.PDF")
            < cli._compute_main_report_priority("2024Q2 业绩公告.PDF")
        )


# ---------------------------------------------------------------------------
# _infer_fiscal_from_path 父目录回退测试
# ---------------------------------------------------------------------------


class TestInferFiscalFromPath:
    """验证 _infer_fiscal_from_path 在文件名无年份时从父目录回退推断。"""

    def test_filename_with_year_period_wins(self, tmp_path: Path) -> None:
        """文件名已含年份与财期时直接返回，不依赖父目录。"""
        p = tmp_path / "2025Q1" / "2024Q3 三季报.pdf"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        result = cli._infer_fiscal_from_path(p)
        assert result == (2024, "Q3")

    def test_bare_filename_q1_dir(self, tmp_path: Path) -> None:
        """「业绩公告.pdf」在 2025Q1/ 下 → (2025, Q1)。"""
        p = tmp_path / "2025Q1" / "业绩公告.pdf"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        result = cli._infer_fiscal_from_path(p)
        assert result == (2025, "Q1")

    def test_bare_filename_q2_dir(self, tmp_path: Path) -> None:
        """「业绩公告.pdf」在 2025Q2/ 下 → (2025, Q2)。"""
        p = tmp_path / "2025Q2" / "业绩公告.pdf"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        result = cli._infer_fiscal_from_path(p)
        assert result == (2025, "Q2")

    def test_bare_filename_q3_dir(self, tmp_path: Path) -> None:
        """「业绩公告.pdf」在 2025Q3/ 下 → (2025, Q3)。"""
        p = tmp_path / "2025Q3" / "业绩公告.pdf"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        result = cli._infer_fiscal_from_path(p)
        assert result == (2025, "Q3")

    def test_bare_filename_h1_dir(self, tmp_path: Path) -> None:
        """「业绩公告.pdf」在 2025H1/ 下 → (2025, H1)。"""
        p = tmp_path / "2025H1" / "业绩公告.pdf"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        result = cli._infer_fiscal_from_path(p)
        assert result == (2025, "H1")

    def test_bare_filename_q4_dir_no_marker_becomes_fy(self, tmp_path: Path) -> None:
        """「业绩公告.pdf」在 2021Q4/ 下且文件名无「季报」→ (2021, FY)。"""
        p = tmp_path / "2021Q4" / "业绩公告.pdf"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        result = cli._infer_fiscal_from_path(p)
        assert result == (2021, "FY")

    def test_bare_filename_q4_dir_with_marker_stays_q4(self, tmp_path: Path) -> None:
        """「季报.pdf」在 2021Q4/ 下且文件名含「季报」→ (2021, Q4)。"""
        p = tmp_path / "2021Q4" / "季报.pdf"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        result = cli._infer_fiscal_from_path(p)
        assert result == (2021, "Q4")

    def test_bare_filename_year_only_dir_returns_none(self, tmp_path: Path) -> None:
        """纯年份目录（2025/）缺失财期信息 → None。"""
        p = tmp_path / "2025" / "业绩公告.pdf"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        result = cli._infer_fiscal_from_path(p)
        assert result is None

    def test_bare_filename_no_year_subdir_returns_none(self, tmp_path: Path) -> None:
        """父目录不是年份子目录（普通命名）→ None。"""
        p = tmp_path / "业绩公告.pdf"
        p.write_text("x")
        result = cli._infer_fiscal_from_path(p)
        assert result is None


def test_upload_filings_from_bare_filename_in_quarter_subdir(tmp_path: Path) -> None:
    """验证 upload_filings_from 正确识别父目录携带财期的「业绩公告.pdf」。

    场景：_3690美团/ 下有 2025Q1/业绩公告.pdf、2025Q2/业绩公告.pdf、
    2025Q3/业绩公告.pdf，文件名本身无年份，年份与财期信息来自父目录。

    Args:
        tmp_path: 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """
    for quarter in ("2025Q1", "2025Q2", "2025Q3"):
        d = tmp_path / quarter
        d.mkdir()
        (d / "业绩公告.pdf").write_text("x", encoding="utf-8")

    output_script = tmp_path / "upload.sh"
    parser = cli._create_parser()
    args = parser.parse_args(
        [
            "upload_filings_from",
            "--ticker", "3690",
            "--from", str(tmp_path),
            "--base", str(tmp_path),
            "--output", str(output_script),
            "--company-name", "美团",
        ]
    )
    result = cli._generate_upload_filings_script(cast(cli.UploadFilingsScriptArgsProtocol, args))

    recognized_periods = [(e["fiscal_year"], e["fiscal_period"]) for e in result["recognized"]]
    assert (2025, "Q1") in recognized_periods
    assert (2025, "Q2") in recognized_periods
    assert (2025, "Q3") in recognized_periods
    assert result["recognized_count"] == 3


class TestPickBestPerPeriod:
    """验证同期最优主报告去重逻辑。"""

    def _make_entry(self, filename: str, year: int, period: str) -> dict:
        """构造测试 entry。"""
        return {"file": f"/x/{filename}", "fiscal_year": year, "fiscal_period": period, "command": f"cmd_{filename}"}

    def test_single_entry_not_deduped(self) -> None:
        """单条 entry 直接保留，无去重。"""
        entries = [self._make_entry("2024年报.pdf", 2024, "FY")]
        kept, duped = cli._pick_best_per_period(entries)
        assert len(kept) == 1
        assert len(duped) == 0

    def test_same_period_picks_highest_priority(self) -> None:
        """同期保留优先级最高文件，其余标为去重。"""
        entries = [
            self._make_entry("2024业绩报告.pdf", 2024, "FY"),
            self._make_entry("2024年度报告.pdf", 2024, "FY"),
            self._make_entry("2024业绩简报.pdf", 2024, "FY"),
        ]
        kept, duped = cli._pick_best_per_period(entries)
        assert len(kept) == 1
        assert len(duped) == 2
        assert kept[0]["file"].endswith("2024年度报告.pdf")

    def test_different_periods_all_kept(self) -> None:
        """不同期间各自保留，互不影响。"""
        entries = [
            self._make_entry("2024年报.pdf", 2024, "FY"),
            self._make_entry("2024H1.pdf", 2024, "H1"),
            self._make_entry("2025年报.pdf", 2025, "FY"),
        ]
        kept, duped = cli._pick_best_per_period(entries)
        assert len(kept) == 3
        assert len(duped) == 0


class TestDeriveMaterialName:
    """验证 material_name 推导逻辑。"""

    def test_keep_year_period_prefix(self) -> None:
        """保留年份+财期前缀，让 LLM 能感知财期。"""
        assert cli._derive_material_name("2024Q2 财务报表.pdf") == "2024Q2 财务报表"

    def test_strip_hkex_but_keep_date_prefix(self) -> None:
        """去除 HKEX 冗余前缀，但保留日期前缀。"""
        assert cli._derive_material_name("2024Q4 HKEX财务报表.pdf") == "2024Q4 财务报表"

    def test_year_only_prefix_kept(self) -> None:
        """仅含年份前缀时也保留。"""
        assert cli._derive_material_name("2024 财务报表.pdf") == "2024 财务报表"

    def test_quarter_prefix_no_space(self) -> None:
        """前缀与文件名无空格时也能正确分离并保留日期。"""
        assert cli._derive_material_name("2025Q1业绩演示.pdf") == "2025Q1 业绩演示"

    def test_no_prefix_returns_stem(self) -> None:
        """无前缀时返回完整 stem。"""
        assert cli._derive_material_name("财务报表.pdf") == "财务报表"

    def test_embedded_date_not_prefix_unchanged(self) -> None:
        """日期不在开头（嵌入文件名中）时不干预，原样保留。"""
        assert cli._derive_material_name("腾讯2025Q3 业绩演示.pdf") == "腾讯2025Q3 业绩演示"

    def test_bare_filename_uses_parent_dir_name(self) -> None:
        """文件名无日期前缀时从 parent_dir_name 补全（2025Q3/财报电话会议.pdf 场景）。"""
        assert cli._derive_material_name("财报电话会议.pdf", parent_dir_name="2025Q3") == "2025Q3 财报电话会议"

    def test_bare_filename_parent_dir_year_only(self) -> None:
        """父目录为纯年份（2025/）时也补全前缀。"""
        assert cli._derive_material_name("业绩演示.pdf", parent_dir_name="2025") == "2025 业绩演示"

    def test_bare_filename_parent_dir_not_year_subdir_unchanged(self) -> None:
        """父目录不是年份子目录（普通命名）时不补全，原样返回。"""
        assert cli._derive_material_name("财报电话会议.pdf", parent_dir_name="美团材料") == "财报电话会议"

    def test_filename_with_date_ignores_parent_dir_name(self) -> None:
        """文件名已有日期前缀时忽略 parent_dir_name。"""
        assert cli._derive_material_name("2024Q2 财务报表.pdf", parent_dir_name="2025Q3") == "2024Q2 财务报表"


class TestMatchMaterialFormType:
    """验证 _match_material_form_type 的路由表匹配逻辑。"""

    def test_财务报表_returns_financial_statements(self) -> None:
        """「财务报表」→ FINANCIAL_STATEMENTS。"""
        assert cli._match_material_form_type("2024Q4 财务报表.pdf") == "FINANCIAL_STATEMENTS"

    def test_业绩演示_returns_earnings_presentation(self) -> None:
        """「业绩演示」→ EARNINGS_PRESENTATION。"""
        assert cli._match_material_form_type("2025Q1业绩演示.pdf") == "EARNINGS_PRESENTATION"

    def test_slide_returns_earnings_presentation(self) -> None:
        """英文 Slide → EARNINGS_PRESENTATION。"""
        assert cli._match_material_form_type("2024Q1 0700 Earning Slide.pdf") == "EARNINGS_PRESENTATION"

    def test_presentation_returns_earnings_presentation(self) -> None:
        """英文 Presentation → EARNINGS_PRESENTATION。"""
        assert cli._match_material_form_type("2025Q1 腾讯 Results Presentation.pdf") == "EARNINGS_PRESENTATION"

    def test_filing_returns_none(self) -> None:
        """普通报告（年度报告、业绩公告）→ None（走 filing 流程）。"""
        assert cli._match_material_form_type("2024年度报告.pdf") is None
        assert cli._match_material_form_type("2024Q4 业绩公告.pdf") is None

    def test_routing_order_财务报表_first(self) -> None:
        """路由顺序：财务报表 pattern 先于演示 pattern。"""
        # 假设文件名同时含「演示」「报表」，先匹配 FINANCIAL_STATEMENTS
        result = cli._match_material_form_type("2024年度财务报表演示.pdf")
        assert result == "FINANCIAL_STATEMENTS"

    def test_电话会议_returns_earnings_call(self) -> None:
        """「电话会议」→ EARNINGS_CALL。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        assert cli._match_material_form_type("2025Q1业绩电话会议.pdf") == "EARNINGS_CALL"

    def test_earnings_call_english_returns_earnings_call(self) -> None:
        """英文 Earnings Call → EARNINGS_CALL。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        assert cli._match_material_form_type("2025Q1 Earnings Call Transcript.pdf") == "EARNINGS_CALL"

    def test_transcript_returns_earnings_call(self) -> None:
        """Transcript → EARNINGS_CALL。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        assert cli._match_material_form_type("2024FY Transcript.pdf") == "EARNINGS_CALL"

    def test_财报会议纪要_returns_earnings_call(self) -> None:
        """「财报会议纪要」→ EARNINGS_CALL。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        assert cli._match_material_form_type("腾讯2025Q4财报会议纪要.pdf") == "EARNINGS_CALL"

    def test_earnings_call_before_presentation_in_routing(self) -> None:
        """路由顺序：EARNINGS_CALL pattern 先于 EARNINGS_PRESENTATION pattern。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        # 含「Earnings Call」的 Presentation 文件，EARNINGS_CALL 先命中
        result = cli._match_material_form_type("2024Q4 Earnings Call Presentation.pdf")
        assert result == "EARNINGS_CALL"


class TestGenerateUploadFilingsScriptWithMaterial:
    """验证 _generate_upload_filings_script 的 material/filing 分流逻辑。"""

    def test_financial_statement_goes_to_material(self, tmp_path: Path) -> None:
        """「财务报表」文件识别为 material，不出现在 recognized 中。"""
        source = tmp_path / "source"
        source.mkdir()
        (source / "2024Q4 年度报告.pdf").write_text("x", encoding="utf-8")
        (source / "2024Q4 财务报表.pdf").write_text("x", encoding="utf-8")
        output_script = tmp_path / "upload.sh"
        parser = cli._create_parser()
        args = parser.parse_args(
            [
                "upload_filings_from",
                "--ticker", "9999",
                "--from", str(source),
                "--base", str(tmp_path),
                "--output", str(output_script),
                "--company-name", "测试公司",
            ]
        )
        result = cli._generate_upload_filings_script(cast(cli.UploadFilingsScriptArgsProtocol, args))

        assert result["recognized_count"] == 1
        assert result["material_count"] == 1
        assert result["skipped_count"] == 0
        assert not (tmp_path / "portfolio" / "9999").exists()
        # 主报告只有年度报告
        assert result["recognized"][0]["fiscal_period"] == "FY"
        # material_name 保留日期前缀 + 语义名称
        assert result["material"][0]["material_name"] == "2024Q4 财务报表"
        # 脚本生成成功
        assert output_script.exists()
        script_text = output_script.read_text(encoding="utf-8")
        assert "upload_material" in script_text
        assert "upload_filing" in script_text

    def test_per_period_dedup_in_script_generation(self, tmp_path: Path) -> None:
        """同期只保留优先级最高文件，低优先级入 skipped。"""
        source = tmp_path / "source"
        source.mkdir()
        (source / "2024年度报告.pdf").write_text("x", encoding="utf-8")
        (source / "2024Q4 业绩公告.pdf").write_text("x", encoding="utf-8")  # Q4 + 无季报 → FY，但优先级低
        output_script = tmp_path / "upload.sh"
        parser = cli._create_parser()
        args = parser.parse_args(
            [
                "upload_filings_from",
                "--ticker", "1234",
                "--from", str(source),
                "--base", str(tmp_path),
                "--output", str(output_script),
                "--company-name", "测试",
            ]
        )
        result = cli._generate_upload_filings_script(cast(cli.UploadFilingsScriptArgsProtocol, args))

        assert result["recognized_count"] == 1
        assert result["skipped_count"] == 1
        # 保留「年度报告」（层 0）而不是「业绩公告」（层 3）
        assert result["recognized"][0]["file"].endswith("2024年度报告.pdf")
        # 跳过「业绩公告」，原因包含去重说明
        assert any("去重" in s["reason"] for s in result["skipped"])

    def test_q4_no_marker_treated_as_fy(self, tmp_path: Path) -> None:
        """Q4 + 无「季报」关键词的文件归为 FY。"""
        source = tmp_path / "source"
        source.mkdir()
        (source / "2024Q4 业绩公告.pdf").write_text("x", encoding="utf-8")
        output_script = tmp_path / "upload.sh"
        parser = cli._create_parser()
        args = parser.parse_args(
            [
                "upload_filings_from",
                "--ticker", "8888",
                "--from", str(source),
                "--base", str(tmp_path),
                "--output", str(output_script),
                "--company-name", "测试",
            ]
        )
        result = cli._generate_upload_filings_script(cast(cli.UploadFilingsScriptArgsProtocol, args))

        assert result["recognized_count"] == 1
        assert result["recognized"][0]["fiscal_period"] == "FY"

    def test_演示文件_goes_to_material_as_earnings_presentation(self, tmp_path: Path) -> None:
        """「业绩演示」文件识别为 EARNINGS_PRESENTATION material，不出现在 recognized 中。

        Args:
            tmp_path: pytest 临时目录。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        source = tmp_path / "source"
        source.mkdir()
        (source / "2025Q1业绩公告.pdf").write_text("x", encoding="utf-8")
        (source / "2025Q1业绩演示.pdf").write_text("x", encoding="utf-8")
        output_script = tmp_path / "upload.sh"
        parser = cli._create_parser()
        args = parser.parse_args(
            [
                "upload_filings_from",
                "--ticker", "0700",
                "--from", str(source),
                "--base", str(tmp_path),
                "--output", str(output_script),
                "--company-name", "腾讯控股",
            ]
        )
        result = cli._generate_upload_filings_script(cast(cli.UploadFilingsScriptArgsProtocol, args))

        # 业绩公告 → recognized；业绩演示 → material
        assert result["recognized_count"] == 1
        assert result["material_count"] == 1
        assert result["material"][0]["material_forms"] == "EARNINGS_PRESENTATION"
        assert result["material"][0]["material_name"] == "2025Q1 业绩演示"
        script_text = output_script.read_text(encoding="utf-8")
        assert "EARNINGS_PRESENTATION" in script_text
        assert "upload_material" in script_text

    def test_earnings_call_goes_to_material(self, tmp_path: Path) -> None:
        """「电话会议」文件识别为 EARNINGS_CALL material。

        Args:
            tmp_path: pytest 临时目录。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        source = tmp_path / "source"
        source.mkdir()
        (source / "2025Q1业绩公告.pdf").write_text("x", encoding="utf-8")
        (source / "2025Q1电话会议.pdf").write_text("x", encoding="utf-8")
        output_script = tmp_path / "upload.sh"
        parser = cli._create_parser()
        args = parser.parse_args(
            [
                "upload_filings_from",
                "--ticker", "0700",
                "--from", str(source),
                "--base", str(tmp_path),
                "--output", str(output_script),
                "--company-name", "腾讯控股",
            ]
        )
        result = cli._generate_upload_filings_script(cast(cli.UploadFilingsScriptArgsProtocol, args))

        assert result["recognized_count"] == 1
        assert result["material_count"] == 1
        assert result["material"][0]["material_forms"] == "EARNINGS_CALL"
        script_text = output_script.read_text(encoding="utf-8")
        assert "EARNINGS_CALL" in script_text

    def test_财报会议纪要_goes_to_material(self, tmp_path: Path) -> None:
        """「财报会议纪要」文件识别为 EARNINGS_CALL material。

        Args:
            tmp_path: pytest 临时目录。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        source = tmp_path / "source"
        (source / "2025Q4").mkdir(parents=True)
        (source / "2025Q4" / "腾讯2025Q4业绩公告.pdf").write_text("x", encoding="utf-8")
        (source / "2025Q4" / "腾讯2025Q4财报会议纪要.pdf").write_text("x", encoding="utf-8")
        output_script = tmp_path / "upload.sh"
        parser = cli._create_parser()
        args = parser.parse_args(
            [
                "upload_filings_from",
                "--ticker", "0700",
                "--from", str(source),
                "--base", str(tmp_path),
                "--output", str(output_script),
                "--company-name", "腾讯控股",
            ]
        )
        result = cli._generate_upload_filings_script(cast(cli.UploadFilingsScriptArgsProtocol, args))

        assert result["recognized_count"] == 1
        assert result["material_count"] == 1
        assert result["skipped_count"] == 0
        assert result["material"][0]["file"].endswith("腾讯2025Q4财报会议纪要.pdf")
        assert result["material"][0]["material_forms"] == "EARNINGS_CALL"
        script_text = output_script.read_text(encoding="utf-8")
        assert "腾讯2025Q4财报会议纪要.pdf" in script_text
        assert "upload_material" in script_text


class TestFilterMaterialEntries:
    """验证 _filter_material_entries 的 form_type 分组上限逻辑。"""

    def _make_entry(self, filename: str, form_type: str) -> dict:
        """创建测试用 material 条目。

        Args:
            filename: 文件名（含年份，用于年份排序键）。
            form_type: material form_type。

        Returns:
            条目字典。

        Raises:
            无。
        """
        return {"file": f"/tmp/{filename}", "material_forms": form_type, "material_name": "test", "command": ""}

    def test_no_cap_keeps_all(self) -> None:
        """未设置上限的 form_type 全部保留。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        entries = [
            self._make_entry("2025Q1财务报表.pdf", "FINANCIAL_STATEMENTS"),
            self._make_entry("2024Q4财务报表.pdf", "FINANCIAL_STATEMENTS"),
            self._make_entry("2024Q2财务报表.pdf", "FINANCIAL_STATEMENTS"),
        ]
        kept, dropped = cli._filter_material_entries(entries, {})
        assert len(kept) == 3
        assert len(dropped) == 0

    def test_cap_keeps_most_recent(self) -> None:
        """超出上限时保留最新年份的条目。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        entries = [
            self._make_entry("2025Q1业绩演示.pdf", "EARNINGS_PRESENTATION"),
            self._make_entry("2024Q4业绩演示.pdf", "EARNINGS_PRESENTATION"),
            self._make_entry("2023Q4业绩演示.pdf", "EARNINGS_PRESENTATION"),
        ]
        kept, dropped = cli._filter_material_entries(entries, {"EARNINGS_PRESENTATION": 2})
        assert len(kept) == 2
        assert len(dropped) == 1
        # 最旧的 2023 应被丢弃
        dropped_file = dropped[0]["file"]
        assert "2023" in dropped_file

    def test_earnings_call_cap_equals_zero_drops_all(self) -> None:
        """EARNINGS_CALL 上限为 0 时全部丢弃。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        entries = [
            self._make_entry("2025Q1电话会议.pdf", "EARNINGS_CALL"),
            self._make_entry("2024FY电话会议.pdf", "EARNINGS_CALL"),
        ]
        kept, dropped = cli._filter_material_entries(entries, {"EARNINGS_CALL": 0})
        assert len(kept) == 0
        assert len(dropped) == 2

    def test_mixed_form_types_independent_caps(self) -> None:
        """不同 form_type 的上限独立计算。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        entries = [
            self._make_entry("2025Q1业绩演示.pdf", "EARNINGS_PRESENTATION"),
            self._make_entry("2024Q4业绩演示.pdf", "EARNINGS_PRESENTATION"),
            self._make_entry("2025Q1电话会议.pdf", "EARNINGS_CALL"),
            self._make_entry("2024FY电话会议.pdf", "EARNINGS_CALL"),
            self._make_entry("2025Q1财务报表.pdf", "FINANCIAL_STATEMENTS"),
        ]
        kept, dropped = cli._filter_material_entries(
            entries,
            {"EARNINGS_PRESENTATION": 1, "EARNINGS_CALL": 1},
        )
        # EARNINGS_PRESENTATION: 保留 2025（最新），丢弃 2024
        # EARNINGS_CALL: 保留 2025（最新），丢弃 2024
        # FINANCIAL_STATEMENTS: 无上限，全部保留
        assert len(kept) == 3
        assert len(dropped) == 2


def test_main_prints_human_readable_process_filing_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 `process_filing` 子命令输出可读文本。

    Args:
        monkeypatch: monkeypatch fixture。
        capsys: 输出捕获 fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake = FakePipeline()
    monkeypatch.setattr(cli, "_build_pipeline_for_ticker", lambda ticker, workspace_root, **kwargs: fake)
    exit_code = cli.main(
        [
            "process_filing",
            "--ticker",
            "AAPL",
            "--document-id",
            "fil_1",
        ]
    )
    output = capsys.readouterr().out.strip()

    assert exit_code == 0
    assert "process_filing 结果" in output
    assert "document_id: fil_1" in output
    assert not output.strip().startswith("{")


def test_main_prints_human_readable_download_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 `download` 子命令输出可读摘要而非 JSON。

    Args:
        monkeypatch: monkeypatch fixture。
        capsys: 输出捕获 fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake = FakePipeline()
    emitted_lines: list[str] = []

    def _capture_info(message: str, *, module: str = "APP") -> None:
        """捕获 CLI 事件回显。"""

        del module
        emitted_lines.append(message)

    monkeypatch.setattr(cli, "_build_pipeline_for_ticker", lambda ticker, workspace_root, **kwargs: fake)
    monkeypatch.setattr(cli.Log, "verbose", _capture_info)
    async def _download_stream(
        ticker: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: Optional[list[str]] = None,
    ) -> AsyncIterator[DownloadEvent]:
        del ticker_aliases
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_STARTED,
            ticker=ticker,
            payload={"form_type": form_type, "overwrite": overwrite, "rebuild": rebuild},
        )
        yield DownloadEvent(
            event_type=DownloadEventType.FILING_COMPLETED,
            ticker=ticker,
            document_id="fil_b",
            payload={
                "status": "skipped",
                "reason_code": "not_modified",
                "reason_message": "所有文件均未修改，跳过重新下载",
                "filing_result": {
                    "document_id": "fil_b",
                    "status": "skipped",
                    "reason_code": "not_modified",
                    "reason_message": "所有文件均未修改，跳过重新下载",
                },
            },
        )
        yield DownloadEvent(
            event_type=DownloadEventType.PIPELINE_COMPLETED,
            ticker=ticker,
            payload={"result": _mock_download_result(ticker, form_type, start_date, end_date, overwrite, rebuild)},
        )

    monkeypatch.setattr(fake, "download_stream", _download_stream)
    exit_code = cli.main(["download", "--ticker", "AAPL"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert any("[download] started ticker=AAPL" in line for line in emitted_lines)
    assert any("reason=not_modified" in line for line in emitted_lines)
    assert any("[download] completed ticker=AAPL" in line for line in emitted_lines)
    assert "下载结果" in output
    assert "成功下载的 filings:" in output
    assert "跳过的 filings:" in output
    assert "失败的 filings:" in output
    assert any("message=所有文件均未修改，跳过重新下载" in line for line in emitted_lines)
    assert not output.strip().startswith("{")


def test_main_prints_human_readable_process_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 `process` 子命令输出可读摘要。

    Args:
        monkeypatch: monkeypatch fixture。
        capsys: 输出捕获 fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake = FakePipeline()
    emitted_lines: list[str] = []
    captured_document_ids: list[str] | None = None

    def _capture_info(message: str, *, module: str = "APP") -> None:
        """捕获 CLI 事件回显。"""

        del module
        emitted_lines.append(message)

    monkeypatch.setattr(cli, "_build_pipeline_for_ticker", lambda ticker, workspace_root, **kwargs: fake)
    monkeypatch.setattr(cli.Log, "verbose", _capture_info)
    async def _process_stream(
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: list[str] | None = None,
    ) -> AsyncIterator[ProcessEvent]:
        nonlocal captured_document_ids
        captured_document_ids = document_ids
        yield ProcessEvent(
            event_type=ProcessEventType.PIPELINE_STARTED,
            ticker=ticker,
            payload={"overwrite": overwrite, "ci": ci, "total_documents": 3},
        )
        yield ProcessEvent(
            event_type=ProcessEventType.DOCUMENT_COMPLETED,
            ticker=ticker,
            document_id="fil_1",
            payload={"source_kind": "filing", "result_summary": {"document_id": "fil_1", "status": "processed"}},
        )
        yield ProcessEvent(
            event_type=ProcessEventType.PIPELINE_COMPLETED,
            ticker=ticker,
            payload={"result": _mock_process_result(ticker=ticker, overwrite=overwrite, ci=ci)},
        )

    monkeypatch.setattr(fake, "process_stream", _process_stream)
    exit_code = cli.main(["process", "--ticker", "AAPL", "--document-id", "fil_1", "--document-id", "fil_2"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert captured_document_ids == ["fil_1", "fil_2"]
    assert any("[process] started ticker=AAPL" in line for line in emitted_lines)
    assert any("[process] document_completed source_kind=filing document_id=fil_1" in line for line in emitted_lines)
    assert any("[process] completed ticker=AAPL" in line for line in emitted_lines)
    assert "全量处理结果" in output
    assert "filings 汇总:" in output
    assert "成功处理的 filings:" in output
    assert "form_type=10-K" in output
    assert "fiscal_year=2025" in output
    assert "materials 处理: 未实现（TODO）" in output
    assert not output.strip().startswith("{")


def test_main_prints_human_readable_upload_material_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """验证 `upload_material` 子命令输出可读文本。

    Args:
        monkeypatch: monkeypatch fixture。
        tmp_path: 临时目录。
        capsys: 输出捕获 fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake = FakePipeline()
    monkeypatch.setattr(cli, "_build_pipeline_for_ticker", lambda ticker, workspace_root, **kwargs: fake)
    file_path = tmp_path / "deck.pdf"
    file_path.write_text("deck", encoding="utf-8")
    exit_code = cli.main(
        [
            "upload_material",
            "--ticker",
            "AAPL",
            "--action",
            "create",
            "--forms",
            "MATERIAL_OTHER",
            "--material-name",
            "Deck",
            "--files",
            str(file_path),
            "--company-name",
            "Apple Inc.",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "上传材料结果" in output
    assert "material_action: create" in output
    assert "files:" in output
    assert not output.strip().startswith("{")


def test_main_upload_material_prints_file_level_progress_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证 `upload_material` 会输出逐文件回显。

    Args:
        monkeypatch: monkeypatch fixture。
        tmp_path: 临时目录。
        capsys: 输出捕获 fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake = FakePipelineWithUploadFileEvents()
    monkeypatch.setattr(cli, "_build_pipeline_for_ticker", lambda ticker, workspace_root, **kwargs: fake)
    file_a = tmp_path / "deck_a.pdf"
    file_b = tmp_path / "deck_b.pdf"
    file_a.write_text("a", encoding="utf-8")
    file_b.write_text("b", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        exit_code = cli.main(
            [
                "upload_material",
                "--ticker",
                "AAPL",
                "--action",
                "create",
                "--forms",
                "MATERIAL_OTHER",
                "--material-name",
                "Deck",
                "--files",
                str(file_a),
                str(file_b),
                "--company-name",
                "Apple Inc.",
            ]
        )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "[upload] started" in caplog.text
    assert "[upload] file_uploaded name=deck_a.pdf" in caplog.text
    assert "[upload] file_uploaded name=deck_b.pdf" in caplog.text
    assert "[upload] completed" in caplog.text
    assert "上传材料结果" in output


def test_main_upload_filing_prints_file_level_progress_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证 `upload_filing` 会输出逐文件回显。

    Args:
        monkeypatch: monkeypatch fixture。
        tmp_path: 临时目录。
        capsys: 输出捕获 fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    fake = FakePipelineWithUploadFileEvents()
    monkeypatch.setattr(cli, "_build_pipeline_for_ticker", lambda ticker, workspace_root, **kwargs: fake)
    file_path = tmp_path / "report.pdf"
    file_path.write_text("r", encoding="utf-8")

    with caplog.at_level(logging.INFO):
        exit_code = cli.main(
            [
                "upload_filing",
                "--ticker",
                "AAPL",
                "--action",
                "create",
                "--files",
                str(file_path),
                "--fiscal-year",
                "2025",
                "--fiscal-period",
                "FY",
                "--company-name",
                "Apple Inc.",
            ]
        )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "[upload] started" in caplog.text
    assert "[upload] file_uploaded name=report.pdf" in caplog.text
    assert "[upload] completed" in caplog.text
    assert "上传财报结果" in output


def test_configure_third_party_http_logging_levels() -> None:
    """验证第三方日志级别配置。

    第三方日志抑制逻辑已整合到 ``Log.set_level``，这里通过 set_level 验证一致性。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 默认（INFO）： httpx/edgar 均抺制到 WARNING
    Log.set_level(LogLevel.INFO)
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("edgar").level == logging.WARNING

    # verbose（VERBOSE）： httpx/edgar 均抺制到 WARNING（仅 DEBUG 时放开）
    Log.set_level(LogLevel.VERBOSE)
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("edgar").level == logging.WARNING

    # debug（DEBUG）：httpx 仍保持常驻抑制，edgar 放开到 INFO
    Log.set_level(LogLevel.DEBUG)
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("edgar").level == logging.INFO
