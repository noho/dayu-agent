"""Pipeline 工厂。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.log import Log
from dayu.fins.ingestion.service import FinsIngestionService
from dayu.fins.processors.registry import (
    build_bs_experiment_registry,
    build_fins_processor_registry,
)
from dayu.fins.ticker_normalization import NormalizedTicker
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    DocumentBlobRepositoryProtocol,
    FilingMaintenanceRepositoryProtocol,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)

from .base import PipelineProtocol
from .cn_pipeline import CnPipeline
from .sec_pipeline import SecPipeline

MODULE = "FINS.PIPELINE_FACTORY"

# 可选的处理器路线提示：匹配 ``--processor-hint`` 参数
_PROCESSOR_HINT_BS = "bs"


def _build_registry_for_hint(processor_hint: Optional[str]) -> ProcessorRegistry:
    """根据处理器路线提示构建注册表。

    Args:
        processor_hint: 可选路线提示（``"bs"`` 使用 BeautifulSoup 路线）。

    Returns:
        处理器注册表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    if processor_hint == _PROCESSOR_HINT_BS:
        Log.info("使用 BS 实验路线处理器注册表", module=MODULE)
        return build_bs_experiment_registry()
    return build_fins_processor_registry()


def get_pipeline_from_normalized_ticker(
    normalized_ticker: NormalizedTicker,
    workspace_root: Path,
    processor_hint: Optional[str] = None,
    company_repository: CompanyMetaRepositoryProtocol | None = None,
    source_repository: SourceDocumentRepositoryProtocol | None = None,
    processed_repository: ProcessedDocumentRepositoryProtocol | None = None,
    blob_repository: DocumentBlobRepositoryProtocol | None = None,
    filing_maintenance_repository: FilingMaintenanceRepositoryProtocol | None = None,
    processor_registry: ProcessorRegistry | None = None,
) -> PipelineProtocol:
    """根据 ``NormalizedTicker`` 构建对应 Pipeline。

    Args:
        normalized_ticker: 规范化 ticker。
        workspace_root: 工作区根目录。
        processor_hint: 可选处理器路线提示（``"bs"`` 使用 BeautifulSoup 路线）。
        company_repository: 可选共享公司元数据仓储实例。
        source_repository: 可选共享源文档仓储实例。
        processed_repository: 可选共享 processed 文档仓储实例。
        blob_repository: 可选共享文件对象仓储实例。
        filing_maintenance_repository: 可选共享 filing 维护治理仓储实例。
        processor_registry: 可选共享处理器注册表；传入后优先于 `processor_hint`。

    Returns:
        对应的 pipeline 实例。

    Raises:
        ValueError: 市场类型不支持时抛出。
    """

    Log.debug(
        f"准备创建 pipeline: market={normalized_ticker.market}",
        module=MODULE,
    )
    registry = processor_registry if processor_registry is not None else _build_registry_for_hint(processor_hint)
    if normalized_ticker.market == "US":
        return SecPipeline(
            workspace_root=workspace_root,
            processor_registry=registry,
            company_repository=company_repository,
            source_repository=source_repository,
            processed_repository=processed_repository,
            blob_repository=blob_repository,
            filing_maintenance_repository=filing_maintenance_repository,
        )
    if normalized_ticker.market in {"HK", "CN"}:
        return CnPipeline(
            workspace_root=workspace_root,
            processor_registry=registry,
            company_repository=company_repository,
            source_repository=source_repository,
            processed_repository=processed_repository,
            blob_repository=blob_repository,
            filing_maintenance_repository=filing_maintenance_repository,
        )
    raise ValueError(f"不支持的 market: {normalized_ticker.market}")


def build_ingestion_service_from_normalized_ticker(
    normalized_ticker: NormalizedTicker,
    workspace_root: Path,
    *,
    company_repository: CompanyMetaRepositoryProtocol,
    source_repository: SourceDocumentRepositoryProtocol,
    processed_repository: ProcessedDocumentRepositoryProtocol,
    blob_repository: DocumentBlobRepositoryProtocol,
    filing_maintenance_repository: FilingMaintenanceRepositoryProtocol,
    processor_registry: ProcessorRegistry,
) -> FinsIngestionService:
    """按 ``NormalizedTicker`` 返回对应 pipeline 的长事务服务。

    该 helper 是 pipelines 包对外暴露的稳定入口，替代上层直接访问具体
    pipeline 实现类的 ``ingestion_service`` 属性；同时把 market→Pipeline
    的分发职责统一收敛到 ``get_pipeline_from_normalized_ticker`` 真源。

    Args:
        normalized_ticker: 已归一化的 ticker。
        workspace_root: 工作区根目录。
        company_repository: 共享公司元数据仓储。
        source_repository: 共享源文档仓储。
        processed_repository: 共享 processed 文档仓储。
        blob_repository: 共享文件对象仓储。
        filing_maintenance_repository: 共享 filing 维护治理仓储；US 与 CN/HK
            download overwrite 都通过它执行 ticker 级清理。
        processor_registry: 共享处理器注册表。

    Returns:
        目标 pipeline 的 ``FinsIngestionService`` 实例。

    Raises:
        ValueError: 市场类型不支持时抛出。
    """

    pipeline = get_pipeline_from_normalized_ticker(
        normalized_ticker=normalized_ticker,
        workspace_root=workspace_root,
        company_repository=company_repository,
        source_repository=source_repository,
        processed_repository=processed_repository,
        blob_repository=blob_repository,
        filing_maintenance_repository=filing_maintenance_repository,
        processor_registry=processor_registry,
    )
    if isinstance(pipeline, (SecPipeline, CnPipeline)):
        return pipeline.ingestion_service
    raise ValueError(f"pipeline 未暴露 ingestion_service: {type(pipeline).__name__}")
