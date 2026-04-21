"""ingestion.factory 模块测试。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, cast

import pytest

import dayu.fins.ingestion.factory as module
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.fins.ingestion.process_events import ProcessEvent, ProcessEventType
from dayu.fins.ingestion.service import FinsIngestionService
from dayu.fins.pipelines.download_events import DownloadEvent, DownloadEventType
from dayu.fins.ticker_normalization import NormalizedTicker
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    DocumentBlobRepositoryProtocol,
    FilingMaintenanceRepositoryProtocol,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)


class _StubProcessorRegistry(ProcessorRegistry):
    """处理器注册表桩。"""


@dataclass
class _CompanyRepositoryStub:
    """公司元数据仓储桩。"""


@dataclass
class _SourceRepositoryStub:
    """源文档仓储桩。"""


@dataclass
class _ProcessedRepositoryStub:
    """processed 仓储桩。"""


@dataclass
class _BlobRepositoryStub:
    """文件对象仓储桩。"""


@dataclass
class _FilingMaintenanceRepositoryStub:
    """filing 维护治理仓储桩。"""


class _BackendStub:
    """最小化 ingestion backend 桩。"""

    def download_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[DownloadEvent]:
        """测试桩不执行下载。"""

        del args, kwargs
        return _empty_download_stream()

    def process_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ProcessEvent]:
        """测试桩不执行预处理。"""

        del args, kwargs
        return _empty_process_stream()


async def _empty_download_stream() -> AsyncIterator[DownloadEvent]:
    """返回空下载事件流。"""

    if False:
        yield DownloadEvent(event_type=DownloadEventType.PIPELINE_COMPLETED, ticker="TEST")


async def _empty_process_stream() -> AsyncIterator[ProcessEvent]:
    """返回空预处理事件流。"""

    if False:
        yield ProcessEvent(event_type=ProcessEventType.PIPELINE_COMPLETED, ticker="TEST")


@dataclass
class _PipelineSpy:
    """记录构造参数的 pipeline 替身。"""

    workspace_root: Path
    company_repository: Any
    source_repository: Any
    processed_repository: Any
    blob_repository: Any
    processor_registry: Any
    filing_maintenance_repository: Any | None = None

    def __post_init__(self) -> None:
        """初始化固定 ingestion_service。"""

        self.ingestion_service = FinsIngestionService(backend=_BackendStub())


@dataclass(frozen=True)
class _RepositoryArgs:
    """测试用仓储参数包。"""

    company_repository: CompanyMetaRepositoryProtocol
    source_repository: SourceDocumentRepositoryProtocol
    processed_repository: ProcessedDocumentRepositoryProtocol
    blob_repository: DocumentBlobRepositoryProtocol
    filing_maintenance_repository: FilingMaintenanceRepositoryProtocol


def _build_repository_args() -> _RepositoryArgs:
    """构建公共仓储桩参数。

    Args:
        无。

    Returns:
        构造 ingestion factory 所需的仓储桩参数。

    Raises:
        无。
    """

    return _RepositoryArgs(
        company_repository=cast(CompanyMetaRepositoryProtocol, _CompanyRepositoryStub()),
        source_repository=cast(SourceDocumentRepositoryProtocol, _SourceRepositoryStub()),
        processed_repository=cast(ProcessedDocumentRepositoryProtocol, _ProcessedRepositoryStub()),
        blob_repository=cast(DocumentBlobRepositoryProtocol, _BlobRepositoryStub()),
        filing_maintenance_repository=cast(
            FilingMaintenanceRepositoryProtocol,
            _FilingMaintenanceRepositoryStub(),
        ),
    )


@pytest.mark.unit
def test_build_ingestion_service_factory_rejects_missing_dependencies(tmp_path: Path) -> None:
    """验证工厂函数会拒绝缺失依赖。"""

    base_args = _build_repository_args()

    with pytest.raises(ValueError, match="workspace_root 不能为空"):
        module.build_ingestion_service_factory(
            workspace_root=cast(Path, None),
            processor_registry=_StubProcessorRegistry(),
            company_repository=base_args.company_repository,
            source_repository=base_args.source_repository,
            processed_repository=base_args.processed_repository,
            blob_repository=base_args.blob_repository,
            filing_maintenance_repository=base_args.filing_maintenance_repository,
        )

    with pytest.raises(ValueError, match="company_repository 不能为空"):
        module.build_ingestion_service_factory(
            workspace_root=tmp_path,
            company_repository=cast(CompanyMetaRepositoryProtocol, None),
            source_repository=base_args.source_repository,
            processed_repository=base_args.processed_repository,
            blob_repository=base_args.blob_repository,
            filing_maintenance_repository=base_args.filing_maintenance_repository,
            processor_registry=_StubProcessorRegistry(),
        )

    with pytest.raises(ValueError, match="processor_registry 不能为空"):
        module.build_ingestion_service_factory(
            workspace_root=tmp_path,
            processor_registry=cast(ProcessorRegistry, None),
            company_repository=base_args.company_repository,
            source_repository=base_args.source_repository,
            processed_repository=base_args.processed_repository,
            blob_repository=base_args.blob_repository,
            filing_maintenance_repository=base_args.filing_maintenance_repository,
        )


@pytest.mark.unit
def test_build_ingestion_service_factory_routes_us_market_to_sec_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 US ticker 会路由到 SecPipeline。"""

    captured: dict[str, Any] = {}

    def _fake_resolve(ticker: str) -> NormalizedTicker:
        """返回固定 US 市场画像。"""

        return NormalizedTicker(canonical=ticker.upper(), market="US", exchange=None, raw=ticker)

    class _SecPipelineSpy(_PipelineSpy):
        """SecPipeline 替身。"""

        def __post_init__(self) -> None:
            """记录被调用的管线类型。"""

            captured["kind"] = "sec"
            captured["workspace_root"] = self.workspace_root
            captured["company_repository"] = self.company_repository
            captured["source_repository"] = self.source_repository
            captured["processed_repository"] = self.processed_repository
            captured["blob_repository"] = self.blob_repository
            captured["filing_maintenance_repository"] = self.filing_maintenance_repository
            captured["processor_registry"] = self.processor_registry
            super().__post_init__()

    monkeypatch.setattr(module, "normalize_ticker", _fake_resolve)
    monkeypatch.setattr(module, "SecPipeline", _SecPipelineSpy)

    repository_args = _build_repository_args()
    processor_registry = _StubProcessorRegistry()
    factory = module.build_ingestion_service_factory(
        workspace_root=tmp_path,
        processor_registry=processor_registry,
        company_repository=repository_args.company_repository,
        source_repository=repository_args.source_repository,
        processed_repository=repository_args.processed_repository,
        blob_repository=repository_args.blob_repository,
        filing_maintenance_repository=repository_args.filing_maintenance_repository,
    )

    service = factory("aapl")

    assert isinstance(service, FinsIngestionService)
    assert captured["kind"] == "sec"
    assert captured["workspace_root"] == tmp_path
    assert captured["company_repository"] is repository_args.company_repository
    assert captured["source_repository"] is repository_args.source_repository
    assert captured["processed_repository"] is repository_args.processed_repository
    assert captured["blob_repository"] is repository_args.blob_repository
    assert captured["filing_maintenance_repository"] is repository_args.filing_maintenance_repository
    assert captured["processor_registry"] is processor_registry


@pytest.mark.unit
@pytest.mark.parametrize("market", ["CN", "HK"])
def test_build_ingestion_service_factory_routes_cn_and_hk_to_cn_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    market: str,
) -> None:
    """验证 CN/HK ticker 会复用 CnPipeline。"""

    captured: dict[str, Any] = {}

    def _fake_resolve(ticker: str) -> NormalizedTicker:
        """返回固定市场画像。"""

        return NormalizedTicker(canonical=ticker.upper(), market=cast(Any, market), exchange=None, raw=ticker)

    class _CnPipelineSpy(_PipelineSpy):
        """CnPipeline 替身。"""

        def __post_init__(self) -> None:
            """记录被调用的管线类型。"""

            captured["kind"] = "cn"
            captured["workspace_root"] = self.workspace_root
            captured["company_repository"] = self.company_repository
            captured["source_repository"] = self.source_repository
            captured["processed_repository"] = self.processed_repository
            captured["blob_repository"] = self.blob_repository
            captured["processor_registry"] = self.processor_registry
            super().__post_init__()

    monkeypatch.setattr(module, "normalize_ticker", _fake_resolve)
    monkeypatch.setattr(module, "CnPipeline", _CnPipelineSpy)

    repository_args = _build_repository_args()
    factory = module.build_ingestion_service_factory(
        workspace_root=tmp_path,
        processor_registry=_StubProcessorRegistry(),
        company_repository=repository_args.company_repository,
        source_repository=repository_args.source_repository,
        processed_repository=repository_args.processed_repository,
        blob_repository=repository_args.blob_repository,
        filing_maintenance_repository=repository_args.filing_maintenance_repository,
    )

    service = factory("000001")

    assert isinstance(service, FinsIngestionService)
    assert captured["kind"] == "cn"
    assert captured["workspace_root"] == tmp_path
    assert captured["company_repository"] is repository_args.company_repository
    assert captured["source_repository"] is repository_args.source_repository
    assert captured["processed_repository"] is repository_args.processed_repository
    assert captured["blob_repository"] is repository_args.blob_repository


@pytest.mark.unit
def test_build_ingestion_service_factory_rejects_unsupported_market(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证未知市场会显式报错。"""

    monkeypatch.setattr(
        module,
        "normalize_ticker",
        lambda ticker: cast(Any, SimpleNamespace(canonical=ticker.upper(), market="JP", exchange=None, raw=ticker)),
    )

    factory = module.build_ingestion_service_factory(
        workspace_root=tmp_path,
        processor_registry=_StubProcessorRegistry(),
        company_repository=_build_repository_args().company_repository,
        source_repository=_build_repository_args().source_repository,
        processed_repository=_build_repository_args().processed_repository,
        blob_repository=_build_repository_args().blob_repository,
        filing_maintenance_repository=_build_repository_args().filing_maintenance_repository,
    )

    with pytest.raises(ValueError, match="不支持的 market: JP"):
        factory("sony")


@pytest.mark.unit
def test_build_ingestion_manager_key_uses_workspace_root_literal(tmp_path: Path) -> None:
    """验证管理器 key 基于工作区根目录生成。"""

    result = module.build_ingestion_manager_key(workspace_root=tmp_path)

    assert result == f"workspace:{tmp_path}"
