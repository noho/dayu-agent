"""长事务服务工厂。"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.fins.ticker_normalization import normalize_ticker
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    DocumentBlobRepositoryProtocol,
    FilingMaintenanceRepositoryProtocol,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)

from .service import FinsIngestionService
from ..pipelines.cn_pipeline import CnPipeline
from ..pipelines.sec_pipeline import SecPipeline

IngestionServiceFactory = Callable[[str], FinsIngestionService]


def build_ingestion_service_factory(
    *,
    workspace_root: Path,
    company_repository: CompanyMetaRepositoryProtocol,
    source_repository: SourceDocumentRepositoryProtocol,
    processed_repository: ProcessedDocumentRepositoryProtocol,
    blob_repository: DocumentBlobRepositoryProtocol,
    filing_maintenance_repository: FilingMaintenanceRepositoryProtocol,
    processor_registry: ProcessorRegistry,
) -> IngestionServiceFactory:
    """构建按 ticker 路由的长事务服务工厂。

    Args:
        workspace_root: 工作区根目录。
        company_repository: 公司元数据仓储。
        source_repository: 源文档仓储。
        processed_repository: processed 文档仓储。
        blob_repository: 文件对象仓储。
        filing_maintenance_repository: filing 维护治理仓储。
        processor_registry: 处理器注册表。

    Returns:
        `ticker -> FinsIngestionService` 的工厂函数。

    Raises:
        ValueError: 参数非法时抛出。
    """

    if workspace_root is None:
        raise ValueError("workspace_root 不能为空")
    if company_repository is None:
        raise ValueError("company_repository 不能为空")
    if source_repository is None:
        raise ValueError("source_repository 不能为空")
    if processed_repository is None:
        raise ValueError("processed_repository 不能为空")
    if blob_repository is None:
        raise ValueError("blob_repository 不能为空")
    if filing_maintenance_repository is None:
        raise ValueError("filing_maintenance_repository 不能为空")
    if processor_registry is None:
        raise ValueError("processor_registry 不能为空")

    def factory(ticker: str) -> FinsIngestionService:
        """按 ticker 创建共享长事务服务。"""

        normalized_ticker = normalize_ticker(ticker)
        if normalized_ticker.market == "US":
            pipeline = SecPipeline(
                workspace_root=workspace_root,
                company_repository=company_repository,
                source_repository=source_repository,
                processed_repository=processed_repository,
                blob_repository=blob_repository,
                filing_maintenance_repository=filing_maintenance_repository,
                processor_registry=processor_registry,
            )
            return pipeline.ingestion_service
        if normalized_ticker.market in {"CN", "HK"}:
            pipeline = CnPipeline(
                workspace_root=workspace_root,
                company_repository=company_repository,
                source_repository=source_repository,
                processed_repository=processed_repository,
                blob_repository=blob_repository,
                processor_registry=processor_registry,
            )
            return pipeline.ingestion_service
        raise ValueError(f"不支持的 market: {normalized_ticker.market}")

    return factory


def build_ingestion_manager_key(*, workspace_root: Path) -> str:
    """构建全局 job 管理器 key。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        稳定的管理器标识。
    """

    return f"workspace:{workspace_root}"
