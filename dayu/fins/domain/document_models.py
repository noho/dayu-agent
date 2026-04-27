"""财报领域模型定义。

该模块集中定义仓储层与管线层通用的数据对象，包含：
- 批处理事务 token
- 公司级元数据
- 文档 CRUD 请求对象
- 文档查询对象与摘要对象
- manifest item 对象

说明：
- 这些模型用于财报仓储窄协议与具体文件系统仓储实现的方法签名。
- 所有对象均采用 dataclass，便于类型检查、测试和序列化。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Optional


DocumentMeta = dict[str, Any]
"""文档元数据字典类型别名。"""


@dataclass(frozen=True)
class FileObjectMeta:
    """文件对象元数据。"""

    uri: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    size: Optional[int] = None
    content_type: Optional[str] = None
    sha256: Optional[str] = None


@dataclass(frozen=True)
class SourceFileEntry:
    """源文档文件条目。

    该模型对应 `filings/*/meta.json` 中的 `files[]` 条目，用于在不依赖
    宽泛字典的前提下表达文件级元数据。
    """

    name: str
    uri: str
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    size: Optional[int] = None
    content_type: Optional[str] = None
    sha256: Optional[str] = None
    source_url: Optional[str] = None
    http_etag: Optional[str] = None
    http_last_modified: Optional[str] = None
    ingested_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """将条目转换为可序列化字典。

        Args:
            无。

        Returns:
            JSON 可序列化字典。

        Raises:
            无。
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceFileEntry":
        """从字典构建源文档文件条目。

        Args:
            data: 原始字典。

        Returns:
            `SourceFileEntry` 实例。

        Raises:
            KeyError: 缺少必填字段时抛出。
            ValueError: 必填字段为空时抛出。
        """

        name = str(data["name"]).strip()
        uri = str(data["uri"]).strip()
        if not name:
            raise ValueError("SourceFileEntry.name 不能为空")
        if not uri:
            raise ValueError("SourceFileEntry.uri 不能为空")
        raw_size = data.get("size")
        size = int(raw_size) if isinstance(raw_size, int) else None
        return cls(
            name=name,
            uri=uri,
            etag=_optional_str(data.get("etag")),
            last_modified=_optional_str(data.get("last_modified")),
            size=size,
            content_type=_optional_str(data.get("content_type")),
            sha256=_optional_str(data.get("sha256")),
            source_url=_optional_str(data.get("source_url")),
            http_etag=_optional_str(data.get("http_etag")),
            http_last_modified=_optional_str(data.get("http_last_modified")),
            ingested_at=_optional_str(data.get("ingested_at")),
        )


@dataclass(frozen=True)
class DocumentEntry:
    """文档目录直系条目。"""

    name: str
    is_file: bool


@dataclass(frozen=True)
class BatchToken:
    """批处理事务 token。

    Attributes:
        token_id: 批处理唯一标识。
        ticker: 对应股票代码。
        target_ticker_dir: 正式 `portfolio/{ticker}` 目录。
        staging_root_dir: 批处理暂存根目录。
        staging_ticker_dir: 批处理暂存目录。
        backup_dir: 提交阶段的备份目录。
        journal_path: 事务 journal 路径。
        ticker_lock_path: ticker 事务锁路径。
        created_at: token 创建时间（ISO8601）。
    """

    token_id: str
    ticker: str
    target_ticker_dir: Path
    staging_root_dir: Path
    staging_ticker_dir: Path
    backup_dir: Path
    journal_path: Path
    ticker_lock_path: Path
    created_at: str


@dataclass(frozen=True)
class CompanyMeta:
    """公司级元数据模型。"""

    company_id: str
    company_name: str
    ticker: str
    market: str
    resolver_version: str
    updated_at: str
    ticker_aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """将对象转换为字典。

        Args:
            无。

        Returns:
            可序列化字典。

        Raises:
            无。
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompanyMeta":
        """从字典构建 `CompanyMeta`。

        Args:
            data: 原始字典数据。

        Returns:
            `CompanyMeta` 实例。

        Raises:
            KeyError: 缺少必填字段时抛出。
        """

        raw_ticker_aliases = data.get("ticker_aliases")
        ticker_aliases = raw_ticker_aliases if isinstance(raw_ticker_aliases, list) else []
        return cls(
            company_id=str(data["company_id"]),
            company_name=str(data["company_name"]),
            ticker=str(data["ticker"]),
            ticker_aliases=[
                str(item).strip()
                for item in ticker_aliases
                if str(item).strip()
            ],
            market=str(data["market"]),
            resolver_version=str(data["resolver_version"]),
            updated_at=str(data["updated_at"]),
        )


CompanyMetaInventoryStatus = Literal[
    "available",
    "hidden_directory",
    "missing_meta",
    "invalid_meta",
]
"""公司目录扫描状态。"""


@dataclass(frozen=True)
class CompanyMetaInventoryEntry:
    """公司目录扫描结果。

    Attributes:
        directory_name: 公司目录名。
        status: 扫描状态。
        company_meta: 当状态为 ``available`` 时的公司元数据。
        detail: 附加说明或错误信息。
    """

    directory_name: str
    status: CompanyMetaInventoryStatus
    company_meta: Optional[CompanyMeta] = None
    detail: str = ""


@dataclass(frozen=True)
class DocumentHandle:
    """文档句柄。"""

    ticker: str
    document_id: str
    form_type: Optional[str] = None
    primary_file_uri: Optional[str] = None
    file_uris: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SourceHandle:
    """源文档句柄。"""

    ticker: str
    document_id: str
    source_kind: str


@dataclass(frozen=True)
class ProcessedHandle:
    """解析产物句柄。"""

    ticker: str
    document_id: str


@dataclass(frozen=True)
class SourceDocumentUpsertRequest:
    """源文档（filings/materials）写入请求基类。"""

    ticker: str
    document_id: str
    internal_document_id: str
    form_type: Optional[str] = None
    primary_document: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    files: list[FileObjectMeta] = field(default_factory=list)
    file_entries: Optional[list[dict[str, Any]]] = None


@dataclass(frozen=True)
class SourceDocumentStateChangeRequest:
    """源文档状态变更请求。

    用于统一 filings / materials 的逻辑删除与恢复操作，避免 public
    仓储协议继续暴露成对重复的 filing/material 方法。
    """

    ticker: str
    document_id: str
    source_kind: str


@dataclass(frozen=True)
class MaterialCreateRequest(SourceDocumentUpsertRequest):
    """材料创建请求。"""


@dataclass(frozen=True)
class MaterialUpdateRequest(SourceDocumentUpsertRequest):
    """材料更新请求。"""


@dataclass(frozen=True)
class MaterialDeleteRequest:
    """材料删除请求。"""

    ticker: str
    document_id: str


@dataclass(frozen=True)
class MaterialRestoreRequest:
    """材料恢复请求。"""

    ticker: str
    document_id: str


@dataclass(frozen=True)
class FilingCreateRequest(SourceDocumentUpsertRequest):
    """财报创建请求。"""


@dataclass(frozen=True)
class FilingUpdateRequest(SourceDocumentUpsertRequest):
    """财报更新请求。"""


@dataclass(frozen=True)
class RejectedFilingArtifactUpsertRequest:
    """rejected filing artifact 写入请求。

    该请求用于将 policy reject 的 filing 以完整 source artifact 形态保存到
    `.rejections/`，但不进入 active filings manifest。
    """

    ticker: str
    document_id: str
    internal_document_id: str
    accession_number: str
    company_id: str
    form_type: str
    filing_date: str
    report_date: Optional[str]
    primary_document: str
    selected_primary_document: str
    rejection_reason: str
    rejection_category: str
    classification_version: str
    source_fingerprint: str
    files: list[SourceFileEntry] = field(default_factory=list)
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    report_kind: Optional[str] = None
    amended: bool = False
    has_xbrl: Optional[bool] = None
    ingest_method: str = "download"


@dataclass(frozen=True)
class RejectedFilingArtifact:
    """rejected filing artifact 读取结果。"""

    ticker: str
    document_id: str
    internal_document_id: str
    accession_number: str
    company_id: str
    form_type: str
    filing_date: str
    report_date: Optional[str]
    primary_document: str
    selected_primary_document: str
    rejection_reason: str
    rejection_category: str
    classification_version: str
    source_fingerprint: str
    files: list[SourceFileEntry] = field(default_factory=list)
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    report_kind: Optional[str] = None
    amended: bool = False
    has_xbrl: Optional[bool] = None
    ingest_method: str = "download"
    rejected_at: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_meta_dict(cls, data: dict[str, Any]) -> "RejectedFilingArtifact":
        """从 rejected artifact meta 构建对象。

        Args:
            data: meta.json 字典。

        Returns:
            `RejectedFilingArtifact` 实例。

        Raises:
            KeyError: 缺少必填字段时抛出。
            ValueError: 必填字段非法时抛出。
        """

        return cls(
            ticker=str(data["ticker"]).strip(),
            document_id=str(data["document_id"]).strip(),
            internal_document_id=str(data["internal_document_id"]).strip(),
            accession_number=str(data["accession_number"]).strip(),
            company_id=str(data["company_id"]).strip(),
            form_type=str(data["form_type"]).strip(),
            filing_date=str(data["filing_date"]).strip(),
            report_date=_optional_str(data.get("report_date")),
            primary_document=str(data["primary_document"]).strip(),
            selected_primary_document=str(data["selected_primary_document"]).strip(),
            rejection_reason=str(data["rejection_reason"]).strip(),
            rejection_category=str(data["rejection_category"]).strip(),
            classification_version=str(data["classification_version"]).strip(),
            source_fingerprint=str(data.get("source_fingerprint", "")).strip(),
            files=[
                SourceFileEntry.from_dict(item)
                for item in data.get("files", [])
                if isinstance(item, dict)
            ],
            fiscal_year=int(data["fiscal_year"]) if isinstance(data.get("fiscal_year"), int) else None,
            fiscal_period=_optional_str(data.get("fiscal_period")),
            report_kind=_optional_str(data.get("report_kind")),
            amended=bool(data.get("amended", False)),
            has_xbrl=data.get("has_xbrl") if isinstance(data.get("has_xbrl"), bool) else None,
            ingest_method=str(data.get("ingest_method", "download")).strip() or "download",
            rejected_at=str(data.get("rejected_at", "")).strip(),
            created_at=str(data.get("created_at", "")).strip(),
            updated_at=str(data.get("updated_at", "")).strip(),
        )

    def to_meta_dict(self) -> dict[str, Any]:
        """将对象转换为 rejected artifact meta 字典。

        Args:
            无。

        Returns:
            meta.json 字典。

        Raises:
            无。
        """

        return {
            "ticker": self.ticker,
            "document_id": self.document_id,
            "internal_document_id": self.internal_document_id,
            "accession_number": self.accession_number,
            "company_id": self.company_id,
            "form_type": self.form_type,
            "filing_date": self.filing_date,
            "report_date": self.report_date,
            "primary_document": self.primary_document,
            "selected_primary_document": self.selected_primary_document,
            "rejection_reason": self.rejection_reason,
            "rejection_category": self.rejection_category,
            "classification_version": self.classification_version,
            "source_fingerprint": self.source_fingerprint,
            "files": [item.to_dict() for item in self.files],
            "fiscal_year": self.fiscal_year,
            "fiscal_period": self.fiscal_period,
            "report_kind": self.report_kind,
            "amended": self.amended,
            "has_xbrl": self.has_xbrl,
            "ingest_method": self.ingest_method,
            "rejected_at": self.rejected_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class FilingDeleteRequest:
    """财报删除请求。"""

    ticker: str
    document_id: str


@dataclass(frozen=True)
class FilingRestoreRequest:
    """财报恢复请求。"""

    ticker: str
    document_id: str


@dataclass(frozen=True)
class ProcessedUpsertRequest:
    """解析产物写入请求基类。"""

    ticker: str
    document_id: str
    internal_document_id: str
    source_kind: str
    form_type: Optional[str] = None
    meta: dict[str, Any] = field(default_factory=dict)
    sections: Optional[list[dict[str, Any]]] = None
    tables: Optional[list[dict[str, Any]]] = None
    financials: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ProcessedCreateRequest(ProcessedUpsertRequest):
    """解析产物创建请求。"""


@dataclass(frozen=True)
class ProcessedUpdateRequest(ProcessedUpsertRequest):
    """解析产物更新请求。"""


@dataclass(frozen=True)
class ProcessedDeleteRequest:
    """解析产物删除请求。"""

    ticker: str
    document_id: str


@dataclass(frozen=True)
class DocumentQuery:
    """文档查询条件。"""

    form_type: Optional[str] = None
    fiscal_years: Optional[list[int]] = None
    fiscal_periods: Optional[list[str]] = None
    source_kind: Optional[str] = None
    include_deleted: bool = False


@dataclass(frozen=True)
class DocumentSummary:
    """文档摘要对象。"""

    document_id: str
    internal_document_id: str
    source_kind: str
    form_type: Optional[str] = None
    material_name: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    report_date: Optional[str] = None
    filing_date: Optional[str] = None
    amended: bool = False
    is_deleted: bool = False
    document_version: str = "v1"
    quality: str = "full"
    has_financials: bool = False
    section_count: int = 0
    table_count: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentSummary":
        """从字典创建 `DocumentSummary`。

        Args:
            data: 摘要字典。

        Returns:
            文档摘要对象。

        Raises:
            KeyError: 缺失必要字段时抛出。
        """

        return cls(
            document_id=str(data["document_id"]),
            internal_document_id=str(data.get("internal_document_id", "")),
            source_kind=str(data.get("source_kind", "filing")),
            form_type=data.get("form_type"),
            material_name=data.get("material_name"),
            fiscal_year=data.get("fiscal_year"),
            fiscal_period=data.get("fiscal_period"),
            report_date=data.get("report_date"),
            filing_date=data.get("filing_date"),
            amended=bool(data.get("amended", False)),
            is_deleted=bool(data.get("is_deleted", False)),
            document_version=str(data.get("document_version", "v1")),
            quality=str(data.get("quality", "full")),
            has_financials=bool(data.get("has_financials", False)),
            section_count=int(data.get("section_count", 0)),
            table_count=int(data.get("table_count", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """将对象转换为字典。

        Args:
            无。

        Returns:
            可序列化字典。

        Raises:
            无。
        """

        return asdict(self)


@dataclass(frozen=True)
class FilingSummary:
    """财报源文件摘要。

    用于服务层向 UI 层暴露已下载财报文件的基本信息，包含文档标识、表单类型、
    申报/报告日期、财年财期、主文件路径等展示所需字段。
    """

    document_id: str
    form_type: Optional[str] = None
    filing_date: Optional[str] = None
    report_date: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    is_deleted: bool = False
    primary_file_name: Optional[str] = None
    primary_file_path: Optional[str] = None


@dataclass(frozen=True)
class FilingManifestItem:
    """`filings/filing_manifest.json` 项目。"""

    document_id: str
    internal_document_id: str
    form_type: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    report_date: Optional[str] = None
    filing_date: Optional[str] = None
    amended: bool = False
    ingest_method: str = "download"
    ingest_complete: bool = True
    is_deleted: bool = False
    deleted_at: Optional[str] = None
    document_version: str = "v1"
    source_fingerprint: str = ""
    has_xbrl: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        """将对象转换为 manifest 字典。

        Args:
            无。

        Returns:
            项目字典。

        Raises:
            无。
        """

        return asdict(self)


@dataclass(frozen=True)
class MaterialManifestItem:
    """`materials/material_manifest.json` 项目。"""

    document_id: str
    internal_document_id: str
    form_type: Optional[str] = None
    material_name: Optional[str] = None
    filing_date: Optional[str] = None
    report_date: Optional[str] = None
    ingest_complete: bool = True
    is_deleted: bool = False
    deleted_at: Optional[str] = None
    document_version: str = "v1"
    source_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        """将对象转换为 manifest 字典。

        Args:
            无。

        Returns:
            项目字典。

        Raises:
            无。
        """

        return asdict(self)


@dataclass(frozen=True)
class ProcessedManifestItem:
    """`processed/manifest.json` 项目。"""

    document_id: str
    internal_document_id: str
    source_kind: str
    form_type: Optional[str] = None
    material_name: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    report_date: Optional[str] = None
    filing_date: Optional[str] = None
    amended: bool = False
    is_deleted: bool = False
    document_version: str = "v1"
    quality: str = "full"
    has_financials: bool = False
    section_count: int = 0
    table_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """将对象转换为 manifest 字典。

        Args:
            无。

        Returns:
            项目字典。

        Raises:
            无。
        """

        return asdict(self)


def now_iso8601() -> str:
    """返回当前 UTC 时间的 ISO8601 字符串。

    Args:
        无。

    Returns:
        ISO8601 时间字符串。

    Raises:
        无。
    """

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _optional_str(value: Any) -> Optional[str]:
    """将任意值标准化为可选字符串。

    Args:
        value: 原始值。

    Returns:
        去空白后的字符串；若为空则返回 `None`。

    Raises:
        无。
    """

    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
