"""财报领域公共契约。

该模块定义 UI、Service、Runtime 共享的财报命令、事件与结果模型，
避免 direct operation 的跨层数据结构退化为无约束字典。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import TypeAlias, cast


class FinsCommandName(StrEnum):
    """财报命令名称。"""

    DOWNLOAD = "download"
    UPLOAD_FILING = "upload_filing"
    UPLOAD_FILINGS_FROM = "upload_filings_from"
    UPLOAD_MATERIAL = "upload_material"
    PROCESS = "process"
    PROCESS_FILING = "process_filing"
    PROCESS_MATERIAL = "process_material"


class FinsEventType(Enum):
    """财报流式事件类型。"""

    PROGRESS = "progress"
    RESULT = "result"


class FinsProgressEventName(StrEnum):
    """跨层稳定 progress 事件名称。"""

    PIPELINE_STARTED = "pipeline_started"
    COMPANY_RESOLVED = "company_resolved"
    FILING_STARTED = "filing_started"
    FILE_DOWNLOADED = "file_downloaded"
    FILE_UPLOADED = "file_uploaded"
    FILE_SKIPPED = "file_skipped"
    FILE_FAILED = "file_failed"
    FILING_COMPLETED = "filing_completed"
    FILING_FAILED = "filing_failed"
    DOCUMENT_STARTED = "document_started"
    DOCUMENT_SKIPPED = "document_skipped"
    DOCUMENT_COMPLETED = "document_completed"
    DOCUMENT_FAILED = "document_failed"
    UPLOAD_STARTED = "upload_started"
    CONVERSION_STARTED = "conversion_started"
    UPLOAD_COMPLETED = "upload_completed"
    UPLOAD_FAILED = "upload_failed"
    PIPELINE_COMPLETED = "pipeline_completed"


class DownloadFilingResultStatus(StrEnum):
    """单个财报下载结果状态。"""

    DOWNLOADED = "downloaded"
    SKIPPED = "skipped"
    FAILED = "failed"

    @classmethod
    def from_raw(cls, raw_status: str) -> DownloadFilingResultStatus:
        """将原始状态字符串解析为下载结果状态枚举。

        参数:
            raw_status: 外部输入的状态字符串，允许包含大小写和首尾空白。

        返回值:
            DownloadFilingResultStatus: 标准化后的下载结果状态。

        异常:
            ValueError: 当输入无法映射到已定义状态时抛出。
        """

        normalized_status = raw_status.strip().lower()
        try:
            return cls(normalized_status)
        except ValueError as exception:
            raise ValueError(f"未知下载结果状态: {raw_status}") from exception


@dataclass(frozen=True)
class DownloadCommandPayload:
    """`download` 命令载荷。

    Attributes:
        ticker: 股票代码。
        form_type: 可选表单集合；为空表示使用 pipeline 默认表单。
        start_date: 可选起始日期。
        end_date: 可选结束日期。
        overwrite: 是否允许覆盖已有结果。
        rebuild: 是否走本地重建路径。
        ticker_aliases: 可选 ticker alias 集合。
    """

    ticker: str
    form_type: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None
    overwrite: bool = False
    rebuild: bool = False
    infer: bool = False
    ticker_aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class UploadFilingCommandPayload:
    """`upload_filing` 命令载荷。"""

    ticker: str
    files: tuple[Path, ...]
    fiscal_year: int
    action: str | None = None
    fiscal_period: str = ""
    amended: bool = False
    filing_date: str | None = None
    report_date: str | None = None
    company_id: str | None = None
    company_name: str | None = None
    infer: bool = False
    ticker_aliases: tuple[str, ...] = ()
    overwrite: bool = False


@dataclass(frozen=True)
class UploadFilingsFromCommandPayload:
    """`upload_filings_from` 命令载荷。"""

    ticker: str
    source_dir: Path
    action: str | None = None
    output_script: Path | None = None
    recursive: bool = False
    amended: bool = False
    filing_date: str | None = None
    report_date: str | None = None
    company_id: str | None = None
    company_name: str | None = None
    infer: bool = False
    overwrite: bool = False
    material_forms: tuple[str, ...] = ()
    verbose: bool = False
    debug: bool = False
    info: bool = False
    quiet: bool = False
    log_level: str | None = None


@dataclass(frozen=True)
class UploadMaterialCommandPayload:
    """`upload_material` 命令载荷。"""

    ticker: str
    files: tuple[Path, ...]
    action: str | None = None
    form_type: str = ""
    material_name: str = ""
    document_id: str | None = None
    internal_document_id: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    filing_date: str | None = None
    report_date: str | None = None
    company_id: str | None = None
    company_name: str | None = None
    infer: bool = False
    ticker_aliases: tuple[str, ...] = ()
    overwrite: bool = False


@dataclass(frozen=True)
class ProcessCommandPayload:
    """`process` 命令载荷。"""

    ticker: str
    document_ids: tuple[str, ...] = ()
    overwrite: bool = False
    ci: bool = False


@dataclass(frozen=True)
class ProcessFilingCommandPayload:
    """`process_filing` 命令载荷。"""

    ticker: str
    document_id: str
    overwrite: bool = False
    ci: bool = False


@dataclass(frozen=True)
class ProcessMaterialCommandPayload:
    """`process_material` 命令载荷。"""

    ticker: str
    document_id: str
    overwrite: bool = False
    ci: bool = False


FinsCommandPayload: TypeAlias = (
    DownloadCommandPayload
    | UploadFilingCommandPayload
    | UploadFilingsFromCommandPayload
    | UploadMaterialCommandPayload
    | ProcessCommandPayload
    | ProcessFilingCommandPayload
    | ProcessMaterialCommandPayload
)


@dataclass(frozen=True)
class DownloadFilterWindow:
    """download 过滤窗口。"""

    form_type: str
    start_date: str


@dataclass(frozen=True)
class DownloadFilters:
    """download 过滤条件快照。"""

    forms: tuple[str, ...] = ()
    start_dates: tuple[DownloadFilterWindow, ...] = ()
    end_date: str | None = None
    overwrite: bool = False


@dataclass(frozen=True)
class DownloadCompanyInfo:
    """download 结果中的公司信息摘要。"""

    company_id: str | None = None
    company_name: str | None = None
    market: str | None = None


@dataclass(frozen=True)
class DownloadFailedFile:
    """单个失败下载文件摘要。"""

    file_name: str | None = None
    source: str | None = None
    reason_code: str | None = None
    reason_message: str | None = None


@dataclass(frozen=True)
class DownloadFilingResultItem:
    """单个 filing 下载结果。"""

    document_id: str
    status: DownloadFilingResultStatus
    form_type: str | None = None
    filing_date: str | None = None
    report_date: str | None = None
    downloaded_files: int = 0
    skipped_files: int = 0
    failed_files: tuple[DownloadFailedFile, ...] = ()
    has_xbrl: bool | None = None
    reason_code: str | None = None
    reason_message: str | None = None
    skip_reason: str | None = None
    filter_category: str | None = None


@dataclass(frozen=True)
class DownloadSummary:
    """download 汇总。"""

    total: int
    downloaded: int
    skipped: int
    failed: int
    elapsed_ms: int = 0


@dataclass(frozen=True)
class DownloadResultData:
    """`download` 命令结果。"""

    pipeline: str
    status: str
    ticker: str
    company_info: DownloadCompanyInfo = field(default_factory=DownloadCompanyInfo)
    filters: DownloadFilters = field(default_factory=DownloadFilters)
    warnings: tuple[str, ...] = ()
    filings: tuple[DownloadFilingResultItem, ...] = ()
    summary: DownloadSummary = field(default_factory=lambda: DownloadSummary(0, 0, 0, 0, 0))


@dataclass(frozen=True)
class UploadFileResultItem:
    """上传结果中的文件条目。"""

    path: str


@dataclass(frozen=True)
class UploadFilingResultData:
    """`upload_filing` 命令结果。"""

    pipeline: str
    status: str
    ticker: str
    filing_action: str
    files: tuple[UploadFileResultItem, ...] = ()
    form_type: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    amended: bool | None = None
    company_id: str | None = None
    company_name: str | None = None
    document_id: str | None = None
    primary_document: str | None = None
    uploaded_files: int | None = None
    document_version: str | None = None
    source_fingerprint: str | None = None
    filing_date: str | None = None
    report_date: str | None = None
    overwrite: bool | None = None
    skip_reason: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class UploadMaterialResultData:
    """`upload_material` 命令结果。"""

    pipeline: str
    status: str
    ticker: str
    material_action: str
    files: tuple[UploadFileResultItem, ...] = ()
    form_type: str | None = None
    material_name: str | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    company_id: str | None = None
    company_name: str | None = None
    document_id: str | None = None
    internal_document_id: str | None = None
    primary_document: str | None = None
    uploaded_files: int | None = None
    document_version: str | None = None
    source_fingerprint: str | None = None
    filing_date: str | None = None
    report_date: str | None = None
    overwrite: bool | None = None
    skip_reason: str | None = None
    message: str | None = None


@dataclass(frozen=True)
class UploadFilingsFromRecognizedItem:
    """`upload_filings_from` 识别成功条目。"""

    file: str
    fiscal_year: int | None = None
    fiscal_period: str | None = None


@dataclass(frozen=True)
class UploadFilingsFromMaterialItem:
    """`upload_filings_from` 材料条目。"""

    file: str
    material_name: str | None = None


@dataclass(frozen=True)
class UploadFilingsFromSkippedItem:
    """`upload_filings_from` 跳过条目。"""

    file: str
    reason: str | None = None


@dataclass(frozen=True)
class UploadFilingsFromResultData:
    """`upload_filings_from` 命令结果。"""

    script_path: str
    script_platform: str
    ticker: str
    source_dir: str
    total_files: int
    recognized_count: int
    material_count: int
    skipped_count: int
    recognized: tuple[UploadFilingsFromRecognizedItem, ...] = ()
    material: tuple[UploadFilingsFromMaterialItem, ...] = ()
    skipped: tuple[UploadFilingsFromSkippedItem, ...] = ()


@dataclass(frozen=True)
class ProcessSummary:
    """process 汇总。"""

    total: int
    processed: int
    skipped: int
    failed: int
    todo: bool = False


@dataclass(frozen=True)
class ProcessDocumentResultItem:
    """单个 process 文档结果。"""

    document_id: str
    status: str
    reason: str | None = None
    form_type: str | None = None
    fiscal_year: int | None = None
    quality: str | None = None
    has_xbrl: bool | None = None
    section_count: int | None = None
    table_count: int | None = None
    skip_reason: str | None = None
    source_kind: str | None = None


@dataclass(frozen=True)
class ProcessResultData:
    """`process` 命令结果。"""

    pipeline: str
    status: str
    ticker: str
    overwrite: bool = False
    ci: bool = False
    filings: tuple[ProcessDocumentResultItem, ...] = ()
    filing_summary: ProcessSummary = field(default_factory=lambda: ProcessSummary(0, 0, 0, 0, False))
    materials: tuple[ProcessDocumentResultItem, ...] = ()
    material_summary: ProcessSummary = field(default_factory=lambda: ProcessSummary(0, 0, 0, 0, False))


@dataclass(frozen=True)
class ProcessSingleResultData:
    """`process_filing/process_material` 命令结果。"""

    pipeline: str
    action: str
    status: str
    ticker: str
    document_id: str
    overwrite: bool = False
    ci: bool = False
    reason: str | None = None
    form_type: str | None = None
    fiscal_year: int | None = None
    quality: str | None = None
    has_xbrl: bool | None = None
    section_count: int | None = None
    table_count: int | None = None
    skip_reason: str | None = None
    message: str | None = None


FinsResultData: TypeAlias = (
    DownloadResultData
    | UploadFilingResultData
    | UploadMaterialResultData
    | UploadFilingsFromResultData
    | ProcessResultData
    | ProcessSingleResultData
)


@dataclass(frozen=True)
class DownloadProgressPayload:
    """download 进度负载。"""

    event_type: FinsProgressEventName
    ticker: str
    document_id: str | None = None
    action: str | None = None
    name: str | None = None
    form_type: str | None = None
    file_count: int | None = None
    size: int | None = None
    message: str | None = None
    reason: str | None = None
    filing_result: DownloadFilingResultItem | None = None


@dataclass(frozen=True)
class ProcessProgressPayload:
    """process 进度负载。"""

    event_type: FinsProgressEventName
    ticker: str
    document_id: str | None = None
    source_kind: str | None = None
    total_documents: int | None = None
    overwrite: bool | None = None
    ci: bool | None = None
    reason: str | None = None
    result_summary: ProcessDocumentResultItem | None = None


@dataclass(frozen=True)
class UploadFilingProgressPayload:
    """upload_filing 进度负载。"""

    event_type: FinsProgressEventName
    ticker: str
    document_id: str | None = None
    action: str | None = None
    name: str | None = None
    file_count: int | None = None
    size: int | None = None
    message: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class UploadMaterialProgressPayload:
    """upload_material 进度负载。"""

    event_type: FinsProgressEventName
    ticker: str
    document_id: str | None = None
    action: str | None = None
    name: str | None = None
    file_count: int | None = None
    size: int | None = None
    message: str | None = None
    error: str | None = None


FinsProgressPayload: TypeAlias = (
    DownloadProgressPayload
    | ProcessProgressPayload
    | UploadFilingProgressPayload
    | UploadMaterialProgressPayload
)


@dataclass(frozen=True)
class FinsCommand:
    """财报操作命令。"""

    name: FinsCommandName
    payload: FinsCommandPayload
    stream: bool = False
    session_id: str | None = None

    def __post_init__(self) -> None:
        """校验命令名与载荷类型的一致性。"""

        if self.name == FinsCommandName.DOWNLOAD and not isinstance(self.payload, DownloadCommandPayload):
            raise TypeError("download 命令必须使用 DownloadCommandPayload")
        if self.name == FinsCommandName.UPLOAD_FILING and not isinstance(self.payload, UploadFilingCommandPayload):
            raise TypeError("upload_filing 命令必须使用 UploadFilingCommandPayload")
        if self.name == FinsCommandName.UPLOAD_FILINGS_FROM and not isinstance(self.payload, UploadFilingsFromCommandPayload):
            raise TypeError("upload_filings_from 命令必须使用 UploadFilingsFromCommandPayload")
        if self.name == FinsCommandName.UPLOAD_MATERIAL and not isinstance(self.payload, UploadMaterialCommandPayload):
            raise TypeError("upload_material 命令必须使用 UploadMaterialCommandPayload")
        if self.name == FinsCommandName.PROCESS and not isinstance(self.payload, ProcessCommandPayload):
            raise TypeError("process 命令必须使用 ProcessCommandPayload")
        if self.name == FinsCommandName.PROCESS_FILING and not isinstance(self.payload, ProcessFilingCommandPayload):
            raise TypeError("process_filing 命令必须使用 ProcessFilingCommandPayload")
        if self.name == FinsCommandName.PROCESS_MATERIAL and not isinstance(self.payload, ProcessMaterialCommandPayload):
            raise TypeError("process_material 命令必须使用 ProcessMaterialCommandPayload")


@dataclass(frozen=True)
class FinsResult:
    """财报操作结果。"""

    command: FinsCommandName
    data: FinsResultData

    def __post_init__(self) -> None:
        """校验命令与结果类型的一致性。"""

        if self.command == FinsCommandName.DOWNLOAD and not isinstance(self.data, DownloadResultData):
            raise TypeError("download 结果必须使用 DownloadResultData")
        if self.command == FinsCommandName.UPLOAD_FILING and not isinstance(self.data, UploadFilingResultData):
            raise TypeError("upload_filing 结果必须使用 UploadFilingResultData")
        if self.command == FinsCommandName.UPLOAD_FILINGS_FROM and not isinstance(self.data, UploadFilingsFromResultData):
            raise TypeError("upload_filings_from 结果必须使用 UploadFilingsFromResultData")
        if self.command == FinsCommandName.UPLOAD_MATERIAL and not isinstance(self.data, UploadMaterialResultData):
            raise TypeError("upload_material 结果必须使用 UploadMaterialResultData")
        if self.command == FinsCommandName.PROCESS and not isinstance(self.data, ProcessResultData):
            raise TypeError("process 结果必须使用 ProcessResultData")
        if self.command in {FinsCommandName.PROCESS_FILING, FinsCommandName.PROCESS_MATERIAL} and not isinstance(
            self.data,
            ProcessSingleResultData,
        ):
            raise TypeError("process_filing/process_material 结果必须使用 ProcessSingleResultData")


@dataclass(frozen=True)
class FinsEvent:
    """财报流式事件。"""

    type: FinsEventType
    command: FinsCommandName
    payload: FinsProgressPayload | FinsResultData

    def __post_init__(self) -> None:
        """校验事件类型、命令和载荷类型的一致性。"""

        if self.type == FinsEventType.PROGRESS:
            if self.command == FinsCommandName.DOWNLOAD and not isinstance(self.payload, DownloadProgressPayload):
                raise TypeError("download progress 事件必须使用 DownloadProgressPayload")
            if self.command == FinsCommandName.PROCESS and not isinstance(self.payload, ProcessProgressPayload):
                raise TypeError("process progress 事件必须使用 ProcessProgressPayload")
            if self.command == FinsCommandName.UPLOAD_FILING and not isinstance(self.payload, UploadFilingProgressPayload):
                raise TypeError("upload_filing progress 事件必须使用 UploadFilingProgressPayload")
            if self.command == FinsCommandName.UPLOAD_MATERIAL and not isinstance(self.payload, UploadMaterialProgressPayload):
                raise TypeError("upload_material progress 事件必须使用 UploadMaterialProgressPayload")
            if self.command in {FinsCommandName.UPLOAD_FILINGS_FROM, FinsCommandName.PROCESS_FILING, FinsCommandName.PROCESS_MATERIAL}:
                raise TypeError(f"{self.command} 不支持 progress 事件")
            return
        result = FinsResult(command=self.command, data=cast(FinsResultData, self.payload))
        del result


__all__ = [
    "DownloadCommandPayload",
    "DownloadCompanyInfo",
    "DownloadFailedFile",
    "DownloadFilingResultItem",
    "DownloadFilingResultStatus",
    "DownloadFilterWindow",
    "DownloadFilters",
    "DownloadProgressPayload",
    "DownloadResultData",
    "DownloadSummary",
    "FinsCommand",
    "FinsCommandName",
    "FinsCommandPayload",
    "FinsEvent",
    "FinsEventType",
    "FinsProgressEventName",
    "FinsProgressPayload",
    "FinsResult",
    "FinsResultData",
    "ProcessCommandPayload",
    "ProcessDocumentResultItem",
    "ProcessFilingCommandPayload",
    "ProcessMaterialCommandPayload",
    "ProcessProgressPayload",
    "ProcessResultData",
    "ProcessSingleResultData",
    "ProcessSummary",
    "UploadFileResultItem",
    "UploadFilingCommandPayload",
    "UploadFilingProgressPayload",
    "UploadFilingResultData",
    "UploadFilingsFromCommandPayload",
    "UploadFilingsFromMaterialItem",
    "UploadFilingsFromRecognizedItem",
    "UploadFilingsFromResultData",
    "UploadFilingsFromSkippedItem",
    "UploadMaterialCommandPayload",
    "UploadMaterialProgressPayload",
    "UploadMaterialResultData",
]
