"""Pipeline 协议定义。

该模块对齐 `copilot/design.md` 的 `1.3 流程设计`，定义：
1. `download_stream` 流式下载
2. `download` 下载
3. `upload_filing_stream` 流式上传财报
4. `upload_filing` 上传财报
5. `upload_material_stream` 流式上传材料
6. `upload_material` 上传材料
7. `process_stream` 流式全量预处理
8. `process` 全量预处理
9. `process_filing` 预处理单个 filing
10. `process_material` 预处理单个 material
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional, Protocol

from dayu.fins.ingestion.process_events import ProcessEvent

from .download_events import DownloadEvent
from .upload_filing_events import UploadFilingEvent
from .upload_material_events import UploadMaterialEvent


class PipelineProtocol(Protocol):
    """财报管线协议。"""

    def download_stream(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: Optional[list[str]] = None,
    ) -> AsyncIterator[DownloadEvent]:
        """执行流式下载。

        Args:
            ticker: 股票代码。
            form_type: 可选文档类型。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否强制覆盖。
            rebuild: 是否仅基于本地已下载数据重建 `meta/manifest`。
            ticker_aliases: 可选公司 alias 列表；download 场景下会与市场侧 alias 合并。

        Returns:
            异步事件流，逐步产出下载过程事件。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...

    def download(
        self,
        ticker: str,
        form_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        overwrite: bool = False,
        rebuild: bool = False,
        ticker_aliases: Optional[list[str]] = None,
    ) -> dict[str, Any]: # pyright: ignore[reportReturnType]
        """执行下载。

        Args:
            ticker: 股票代码。
            form_type: 可选文档类型。
            start_date: 可选开始日期。
            end_date: 可选结束日期。
            overwrite: 是否强制覆盖。
            rebuild: 是否仅基于本地已下载数据重建 `meta/manifest`。
            ticker_aliases: 可选公司 alias 列表；download 场景下会与市场侧 alias 合并。

        Returns:
            下载结果。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...

    def upload_filing(
        self,
        ticker: str,
        action: Optional[str],
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
    ) -> dict[str, Any]: # pyright: ignore[reportReturnType]
        """执行财报上传。

        Args:
            ticker: 股票代码。
            action: 可选动作类型（create/update/delete）；为空时自动判定。
            files: 上传文件列表。
            fiscal_year: 财年。
            fiscal_period: 财季或年度标识（Q1/Q2/Q3/Q4/FY/H1）。
            amended: 是否修订版。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID（create/update 必填）。
            company_name: 公司名称（create/update 必填）。
            ticker_aliases: 可选 ticker alias 列表；用于初始化公司级 meta。
            overwrite: 是否强制覆盖。

        Returns:
            上传结果。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...

    def upload_filing_stream(
        self,
        ticker: str,
        action: Optional[str],
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
        """执行流式财报上传。

        Args:
            ticker: 股票代码。
            action: 可选动作类型（create/update/delete）；为空时自动判定。
            files: 上传文件列表。
            fiscal_year: 财年。
            fiscal_period: 财季或年度标识（Q1/Q2/Q3/Q4/FY/H1）。
            amended: 是否修订版。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID（create/update 必填）。
            company_name: 公司名称（create/update 必填）。
            ticker_aliases: 可选 ticker alias 列表；用于初始化公司级 meta。
            overwrite: 是否强制覆盖。

        Returns:
            异步事件流，逐步产出上传过程事件。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...

    def upload_material(
        self,
        ticker: str,
        action: Optional[str],
        form_type: str,
        material_name: str,
        files: Optional[list[Path]] = None,
        document_id: Optional[str] = None,
        internal_document_id: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """执行材料上传。

        Args:
            ticker: 股票代码。
            action: 可选动作类型（create/update/delete）；为空时自动判定。
            form_type: 材料类型。
            material_name: 材料名称。
            files: 上传文件列表（create/update 需要）。
            document_id: 可选文档 ID（update/delete 需要）。
            internal_document_id: 可选内部文档 ID（update/delete 可用）。
            fiscal_year: 可选财年；提供时参与稳定 document_id 生成。
            fiscal_period: 可选财期；提供时参与稳定 document_id 生成。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID（create/update 必填）。
            company_name: 公司名称（create/update 必填）。
            ticker_aliases: 可选 ticker alias 列表；用于初始化公司级 meta。
            overwrite: 是否强制覆盖。

        Returns:
            上传结果。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...

    def upload_material_stream(
        self,
        ticker: str,
        action: Optional[str],
        form_type: str,
        material_name: str,
        files: Optional[list[Path]] = None,
        document_id: Optional[str] = None,
        internal_document_id: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        filing_date: Optional[str] = None,
        report_date: Optional[str] = None,
        company_id: Optional[str] = None,
        company_name: Optional[str] = None,
        ticker_aliases: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> AsyncIterator[UploadMaterialEvent]:
        """执行流式材料上传。

        Args:
            ticker: 股票代码。
            action: 可选动作类型（create/update/delete）；为空时自动判定。
            form_type: 材料类型。
            material_name: 材料名称。
            files: 上传文件列表（create/update 需要）。
            document_id: 可选文档 ID（update/delete 需要）。
            internal_document_id: 可选内部文档 ID（update/delete 可用）。
            fiscal_year: 可选财年；提供时参与稳定 document_id 生成。
            fiscal_period: 可选财期；提供时参与稳定 document_id 生成。
            filing_date: 可选披露日期。
            report_date: 可选报告日期。
            company_id: 公司 ID（create/update 必填）。
            company_name: 公司名称（create/update 必填）。
            ticker_aliases: 可选 ticker alias 列表；用于初始化公司级 meta。
            overwrite: 是否强制覆盖。

        Returns:
            异步事件流，逐步产出上传过程事件。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...

    def process_stream(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: Optional[list[str]] = None,
    ) -> AsyncIterator[ProcessEvent]:
        """执行流式全量预处理。

        Args:
            ticker: 股票代码。
            overwrite: 是否强制覆盖。
            ci: 是否追加导出 `search_document/query_xbrl_facts` 快照。
            document_ids: 可选文档 ID 列表；传入时仅处理这些文档。

        Returns:
            异步事件流，逐步产出预处理过程事件。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...

    def process(
        self,
        ticker: str,
        overwrite: bool = False,
        ci: bool = False,
        document_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """执行全量预处理。

        Args:
            ticker: 股票代码。
            overwrite: 是否强制覆盖。
            ci: 是否追加导出 `search_document/query_xbrl_facts` 快照。
            document_ids: 可选文档 ID 列表；传入时仅处理这些文档。

        Returns:
            预处理结果。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...

    def process_filing(
        self,
        ticker: str,
        document_id: str,
        overwrite: bool = False,
        ci: bool = False,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """预处理单个 filing。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            overwrite: 是否强制重处理。
            ci: 是否追加导出 `search_document/query_xbrl_facts` 快照。
            cancel_checker: 可选取消检查函数，仅在单文档处理阶段边界生效。

        Returns:
            预处理结果。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...

    def process_material(
        self,
        ticker: str,
        document_id: str,
        overwrite: bool = False,
        ci: bool = False,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """预处理单个 material。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            overwrite: 是否强制重处理。
            ci: 是否追加导出 `search_document/query_xbrl_facts` 快照。
            cancel_checker: 可选取消检查函数，仅在单文档处理阶段边界生效。

        Returns:
            预处理结果。

        Raises:
            RuntimeError: 执行失败时抛出。
        """

        ...
