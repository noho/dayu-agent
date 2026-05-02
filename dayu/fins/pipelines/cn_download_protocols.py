"""CN/HK 下载链路对外 Protocol 边界。

本模块定义两类协议：

- :class:`CnReportDiscoveryClientProtocol`：discovery / 下载器对外稳定边界。
  巨潮 / 披露易 / 未来其他 CN/HK 实现均按此协议接入；workflow 层只依赖此协议，
  不假设具体实现细节。
- :class:`CnDownloadWorkflowHost`：CN download workflow 所需的最小宿主边界，
  仅暴露所需仓储以及 docling 注入点。pipeline 通过实现该 Protocol 把仓储、
  docling 函数透传给 workflow，避免 workflow 反向依赖 ``CnPipeline``。

设计要点：

- 所有协议方法签名禁止 ``Any`` / ``object``；仅在与第三方 SDK 必要边界处用
  类型化结构（如 ``CnReportCandidate``）。
- ``Protocol`` 用 ``runtime_checkable`` 修饰仅在确实有运行期 ``isinstance``
  检查时才使用；本模块仅做静态契约，不开 runtime check。
- 不在协议内塞入"取消检查"/"日志模块名"等横切关注；横切由 workflow 层显式
  接收 ``cancel_checker`` 参数管理。
"""

from __future__ import annotations

from typing import Optional, Protocol

from dayu.fins.docling_export import PdfToDoclingJsonBytes
from dayu.fins.pipelines.cn_download_models import (
    CnCompanyProfile,
    CnReportCandidate,
    CnReportQuery,
    DownloadedReportAsset,
)
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    DocumentBlobRepositoryProtocol,
    FilingMaintenanceRepositoryProtocol,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)


class CnReportDiscoveryClientProtocol(Protocol):
    """CN/HK 报告发现 / 下载 Protocol。

    一个 ticker 一次 download 流程会先调用 ``resolve_company`` 拿
    :class:`CnCompanyProfile`，再按 ``target_periods`` 调
    ``list_report_candidates`` 拿候选列表，最后按 candidate 调
    ``download_report_pdf`` 落 PDF。

    实现约束：

    - 不写 workspace、不依赖 pipeline、不调 docling、不生成 ``document_id``。
    - HEAD/GET 失败、PDF magic bytes 校验失败等候选层失败仅影响该 candidate，
      不能让整个 ticker 流程崩。
    - HK Q1/Q3 主板查无视为返回空列表（不抛异常），由 workflow 标 skipped。
    """

    def resolve_company(self, query: CnReportQuery) -> CnCompanyProfile:
        """根据 ``CnReportQuery.normalized_ticker`` 解析公司基础元数据。

        Args:
            query: 单次 download 的查询参数。

        Returns:
            :class:`CnCompanyProfile`，``company_id`` 必须遵循 ``CNINFO:{orgId}``
            或 ``HKEX:{stockId}`` 前缀约定。

        Raises:
            ValueError: 解析失败（无法在主源公司映射中定位公司）时抛出。
        """

        ...

    def list_report_candidates(
        self,
        query: CnReportQuery,
        profile: CnCompanyProfile,
    ) -> tuple[CnReportCandidate, ...]:
        """列出符合 ``target_periods`` 与窗口约束的候选报告。

        实现层负责按白/黑名单与类别过滤、按 fiscal_period 去重；多版本仅保留
        最新有效全文，amended 优先。HK Q1/Q3 主板查无返回空 tuple，**不**抛
        异常。

        Args:
            query: 单次 download 的查询参数。
            profile: ``resolve_company`` 返回的公司元数据。

        Returns:
            候选报告 tuple；候选已经按 fiscal_period 收敛、amended 优先。

        Raises:
            ValueError: 主源响应无法解析时抛出。
        """

        ...

    def download_report_pdf(self, candidate: CnReportCandidate) -> DownloadedReportAsset:
        """下载单份候选 PDF 并返回强类型资产对象。

        实现层负责 ``%PDF-`` magic bytes 校验、最小长度校验、Content-Type
        校验、必要的 retry / sleep。返回值 ``pdf_path`` 指向本地暂存路径，
        workflow 层取字节后会自行 ``unlink``。

        Args:
            candidate: 单份候选远端元数据。

        Returns:
            :class:`DownloadedReportAsset`。

        Raises:
            RuntimeError: 下载失败、PDF 校验失败、HTTP 状态码异常时抛出。
        """

        ...


class CnDownloadWorkflowHost(Protocol):
    """CN download workflow 所需的最小宿主边界。

    实现侧典型为 :class:`dayu.fins.pipelines.cn_pipeline.CnPipeline`，但
    workflow 不直接 import ``CnPipeline``，仅按本协议消费。

    协议只暴露：

    - 5 个仓储（company / source / blob / processed / filing_maintenance）。
    - CN/HK 两个 discovery client 注入点。
    - docling 转换函数注入点（``Callable[[bytes, str], bytes]``）。

    其它横切（取消、日志）由 workflow 层显式接收 ``cancel_checker`` 等参数
    管理，不放进 host。
    """

    @property
    def company_meta_repository(self) -> CompanyMetaRepositoryProtocol:
        """公司元数据仓储。"""

        ...

    @property
    def source_repository(self) -> SourceDocumentRepositoryProtocol:
        """源文档仓储。"""

        ...

    @property
    def blob_repository(self) -> DocumentBlobRepositoryProtocol:
        """文件对象仓储。"""

        ...

    @property
    def processed_repository(self) -> ProcessedDocumentRepositoryProtocol:
        """processed 仓储；download 链路在 commit 后按规则标 reprocess。"""

        ...

    @property
    def filing_maintenance_repository(self) -> FilingMaintenanceRepositoryProtocol:
        """filing 维护仓储；``overwrite=True`` 时做 ticker 级 ``clear_filing_documents``。"""

        ...

    @property
    def cn_discovery_client(self) -> CnReportDiscoveryClientProtocol:
        """A 股巨潮 discovery client。"""

        ...

    @property
    def hk_discovery_client(self) -> CnReportDiscoveryClientProtocol:
        """港股披露易 discovery client。"""

        ...

    @property
    def convert_pdf_to_docling_json(self) -> PdfToDoclingJsonBytes:
        """docling 转换函数注入点；签名 ``(bytes, str) -> bytes``。"""

        ...

    @property
    def user_agent(self) -> Optional[str]:
        """HTTP User-Agent，用于 downloader 透传。"""

        ...

    @property
    def sleep_seconds(self) -> float:
        """连续 HTTP 请求间隔秒数。"""

        ...

    @property
    def max_retries(self) -> int:
        """最大重试次数。"""

        ...


__all__ = [
    "CnDownloadWorkflowHost",
    "CnReportDiscoveryClientProtocol",
]
