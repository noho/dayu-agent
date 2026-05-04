"""CN/HK Docling 文档 LLM 可喂性 CI 评分入口。

本模块面向 A 股 / 港股 Docling JSON 路线，只读取 ``process --ci`` 已导出的
``tool_snapshot_*`` 快照文件，并按 CN/HK Docling profile 输出 JSON 与 Markdown
报告。评分规则独立于 SEC ``score_sec_ci``，不使用 SEC Item 覆盖率，也不修改
processor 或工具 schema。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
import unicodedata
from typing import Mapping, Optional, Sequence, TypeAlias, cast

from dayu.fins.domain.document_models import (
    CompanyMetaInventoryEntry,
    DocumentQuery,
    ProcessedHandle,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    DocumentBlobRepositoryProtocol,
    FsCompanyMetaRepository,
    FsDocumentBlobRepository,
    FsProcessedDocumentRepository,
    FsSourceDocumentRepository,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

DEFAULT_TICKERS: tuple[str, ...] = ()
REQUIRED_SNAPSHOT_SCHEMA_VERSION = "tool_snapshot_v1.0.0"
REPORT_PROFILE_ID = "cn_hk_docling_v1"

REPORT_KIND_ANNUAL = "annual"
REPORT_KIND_SEMIANNUAL = "semiannual"
REPORT_KIND_QUARTERLY = "quarterly"
REPORT_KIND_MATERIAL = "material"
REPORT_KIND_ALL = "all"
REPORT_KIND_CHOICES = (
    REPORT_KIND_ALL,
    REPORT_KIND_ANNUAL,
    REPORT_KIND_SEMIANNUAL,
    REPORT_KIND_QUARTERLY,
    REPORT_KIND_MATERIAL,
)

SOURCE_KIND_ALL = "all"
SOURCE_KIND_CHOICES = (SOURCE_KIND_ALL, SourceKind.FILING.value, SourceKind.MATERIAL.value)

TOOL_SNAPSHOT_META_FILE_NAME = "tool_snapshot_meta.json"
TOOL_SNAPSHOT_SECTIONS_FILE_NAME = "tool_snapshot_get_document_sections.json"
TOOL_SNAPSHOT_READ_SECTION_FILE_NAME = "tool_snapshot_read_section.json"
TOOL_SNAPSHOT_SEARCH_FILE_NAME = "tool_snapshot_search_document.json"
TOOL_SNAPSHOT_LIST_TABLES_FILE_NAME = "tool_snapshot_list_tables.json"
TOOL_SNAPSHOT_GET_TABLE_FILE_NAME = "tool_snapshot_get_table.json"
TOOL_SNAPSHOT_PAGE_FILE_NAME = "tool_snapshot_get_page_content.json"

DIMENSION_A_STRUCTURE = "A_structure"
DIMENSION_B_TEXT = "B_text"
DIMENSION_C_SEARCH = "C_search"
DIMENSION_D_TABLE = "D_table"
DIMENSION_E_TRACEABILITY = "E_traceability"
DIMENSION_F_PAGE = "F_page"

MAX_A_STRUCTURE = 20.0
MAX_B_TEXT = 15.0
MAX_C_SEARCH = 15.0
MAX_D_TABLE = 20.0
MAX_E_TRACEABILITY = 20.0
MAX_F_PAGE = 10.0
MAX_TOTAL_SCORE = 100.0

MIN_DOC_PASS = 85.0
MIN_DOC_WARN = 75.0
MIN_BATCH_AVG = 85.0
MIN_BATCH_P10 = 78.0
HUGE_SECTION_FAIL_CHARS = 300_000
NEAR_EMPTY_SECTION_CHARS = 80

SEARCH_STRATEGY_EXACT = "exact"
TABLE_PLACEHOLDER_RE = re.compile(r"\[\[(t_\d{4})\]\]")
MOJIBAKE_RE = re.compile(r"Ã.|Â.|â€.|ï¿½|\uFFFD|[\x80-\x9f]")
CJK_CHAR_RE = re.compile(r"[\u3400-\u9fff]")
CJK_SPACED_RE = re.compile(r"[\u3400-\u9fff]\s+[\u3400-\u9fff]")
TOC_LINE_RE = re.compile(r"(?m)^\s*.{2,80}(?:\.{2,}|\s{2,})\d{1,4}\s*$")
DEFAULT_HEADER_RE = re.compile(r"(?i)^(?:unnamed.*|\d+|column\s*\d+|col\s*\d+)$")


@dataclass(frozen=True, slots=True)
class KeywordGroup:
    """CN/HK 章节或表格关键词组。

    Attributes:
        label: 标准化标签。
        keywords: 可匹配关键词。
        required: 是否属于 hard gate 关键组。
    """

    label: str
    keywords: tuple[str, ...]
    required: bool = False


@dataclass(frozen=True, slots=True)
class DoclingProfile:
    """CN/HK Docling report kind 评分参数。

    Attributes:
        report_kind: 标准 report kind。
        section_count_low: 合理 section 数下界。
        section_count_high: 合理 section 数上界。
        search_t5: 搜索覆盖率 5 分阈值。
        search_t7: 搜索覆盖率 7 分阈值。
        search_t9: 搜索覆盖率 9 分阈值。
        key_groups: 关键章节关键词组。
        financial_groups: 财务结构关键词组。
        require_page_evidence: 是否要求关键 Docling 页码可复核。
    """

    report_kind: str
    section_count_low: int
    section_count_high: int
    search_t5: float
    search_t7: float
    search_t9: float
    key_groups: tuple[KeywordGroup, ...]
    financial_groups: tuple[KeywordGroup, ...]
    require_page_evidence: bool


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    """评分阈值配置。

    Attributes:
        min_doc_pass: 单文档通过阈值。
        min_doc_warn: 单文档警告阈值。
        min_batch_avg: 批量平均分阈值。
        min_batch_p10: 批量 P10 阈值。
        huge_section_fail: 单 section 过大 hard gate 阈值。
    """

    min_doc_pass: float = MIN_DOC_PASS
    min_doc_warn: float = MIN_DOC_WARN
    min_batch_avg: float = MIN_BATCH_AVG
    min_batch_p10: float = MIN_BATCH_P10
    huge_section_fail: int = HUGE_SECTION_FAIL_CHARS


@dataclass(slots=True)
class DimensionScore:
    """单维度评分结果。"""

    points: float
    max_points: float
    details: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class HardGateResult:
    """单文档 hard gate 结果。"""

    passed: bool
    reasons: list[str]


@dataclass(frozen=True, slots=True)
class CompletenessFailure:
    """样本完整性 hard gate 失败记录。"""

    ticker: str
    document_id: str
    source_kind: str
    reason: str


@dataclass(slots=True)
class DocumentScore:
    """单文档评分结果。"""

    ticker: str
    document_id: str
    source_kind: str
    report_kind: str
    market: str
    total_score: float
    grade: str
    hard_gate: HardGateResult
    dimensions: dict[str, DimensionScore]


@dataclass(slots=True)
class BatchScore:
    """批量评分汇总结果。"""

    documents: list[DocumentScore]
    average_score: float
    p10_score: float
    hard_gate_failures: int
    passed: bool
    failed_reasons: list[str]
    completeness_failures: list[CompletenessFailure] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ProcessedSnapshotDocument:
    """待评分 processed 快照访问对象。"""

    ticker: str
    document_id: str
    source_kind: SourceKind
    report_kind: str
    source_meta: JsonObject
    handle: ProcessedHandle


@dataclass(frozen=True, slots=True)
class SnapshotLoadResult:
    """单个工具快照加载结果。"""

    payload: JsonObject
    exists: bool
    valid: bool


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """CN/HK Docling 样本发现结果。"""

    snapshots: list[ProcessedSnapshotDocument]
    completeness_failures: list[CompletenessFailure]


ANNUAL_KEY_GROUPS = (
    KeywordGroup("company_profile", ("公司简介", "公司資料", "公司资料")),
    KeywordGroup("main_business", ("主营业务", "主營業務", "业务概要", "業務概覽")),
    KeywordGroup("mda", ("管理层讨论", "管理層討論", "董事会报告", "董事會報告", "业务回顾", "業務回顧", "业务表现", "業務表現", "主营业务分析", "主營業務分析", "财务回顾", "財務回顧", "财务表现", "財務表現", "财务及营运回顾", "財務及營運回顧", "集团回顾", "集團回顧", "经营亮点", "經營亮點", "业绩综述", "業績綜述", "主席报告书", "主席報告書", "行政总裁报告书", "行政總裁報告書", "行政總裁之回顧", "首席财务总监回顾", "首席財務總監回顧"), True),
    KeywordGroup("key_financials", ("主要会计数据", "主要財務資料", "财务指标", "財務指標")),
    KeywordGroup("governance", ("公司治理", "企業管治")),
    KeywordGroup("shareholders", ("股东信息", "股東資料", "主要股东", "主要股東")),
    KeywordGroup("risk", ("重大事项", "重大事項", "风险提示", "風險提示", "风险因素")),
    KeywordGroup("audit", ("审计意见", "审计报告", "審計報告", "財務報表審計報告", "财务报表审计报告", "核數師報告", "独立审计", "獨立核數師", "關鍵審計事項", "关键审计事项", "我們已審計的內容", "我们已审计的内容", "已审财务报表", "已審財務報表", "注册会计师对财务报表审计", "註冊會計師對財務報表審計"), True),
    KeywordGroup("notes", ("附注", "附註", "財務報表附註", "财务报表附注", "綜合財務報表附註", "综合财务报表附注", "会计报表注释", "會計報表註釋", "财务报表注释", "財務報表註釋", "财务报表项目注释", "会计报表主要项目注释", "會計報表主要項目註釋", "合并财务报表项目注释", "合并财务报表主要项目注释", "合併會計報表註釋", "合併會計報表主要項目註釋", "母公司财务报表主要项目注释", "綜合財務報表說明", "综合财务报表说明"), True),
)
SEMIANNUAL_KEY_GROUPS = (
    KeywordGroup("main_business", ("主营业务", "主營業務", "业务概要", "業務概覽")),
    KeywordGroup("mda", ("管理层讨论", "管理層討論", "董事会报告", "董事會報告", "业务回顾", "業務回顧", "业务表现", "業務表現", "主营业务分析", "主營業務分析", "财务回顾", "財務回顧", "财务表现", "財務表現", "财务及营运回顾", "財務及營運回顧", "集团回顾", "集團回顧", "经营亮点", "經營亮點", "业绩综述", "業績綜述", "主席报告书", "主席報告書", "行政总裁报告书", "行政總裁報告書", "行政總裁之回顧", "首席财务总监回顾", "首席財務總監回顧"), True),
    KeywordGroup("key_financials", ("主要会计数据", "主要會計數據", "主要財務資料", "主要财务数据", "主要財務數據", "财务指标", "財務指標", "财务资料", "財務資料", "财务摘要", "財務摘要", "财务概要", "財務概要", "财务概览", "財務概覽", "财务撮要", "財務撮要", "财务要点", "財務要點", "财务表现", "財務表現", "财务表现摘要", "財務表現摘要", "财务表现指标", "財務表現指標", "业绩摘要", "業績摘要", "业绩概览", "業績概覽", "经营亮点", "經營亮點", "中期业绩", "中期業績", "HIGHLIGHTS"), True),
    KeywordGroup("risk", ("重大事项", "重大事項", "风险提示", "風險提示")),
    KeywordGroup("notes", ("附注", "附註", "簡明附註", "简明附注", "綜合財務報表附註", "综合财务报表附注", "财务报表注释", "財務報表註釋", "财务报表项目注释", "合并财务报表项目注释", "合并财务报表主要项目注释", "母公司财务报表主要项目注释", "綜合財務報表說明", "综合财务报表说明"), True),
)
QUARTERLY_KEY_GROUPS = (
    KeywordGroup("key_financials", ("主要会计数据", "主要會計數據", "主要財務資料", "主要财务数据", "主要財務數據", "财务指标", "財務指標", "财务资料", "財務資料", "财务概要", "財務概要", "财务表现", "財務表現", "财务表现摘要", "財務表現摘要", "主要财务衡量指标", "主要財務衡量指標", "关键数据摘要", "關鍵數據摘要", "主要经营业绩", "主要經營業績"), True),
    KeywordGroup("operations", ("经营情况", "經營情況", "管理层讨论", "管理層討論", "业务回顾", "業務回顧", "主营业务分析", "主營業務分析", "财务回顾", "財務回顧")),
    KeywordGroup("risk", ("重大事项", "重大事項", "风险提示", "風險提示")),
)
MATERIAL_KEY_GROUPS = (
    KeywordGroup("title", ("公告", "通函", "材料", "业绩", "業績", "风险提示", "風險提示"), True),
    KeywordGroup("body", ("事项", "交易", "影响", "風險", "风险", "董事会", "董事會"), True),
)
FINANCIAL_GROUPS_FULL = (
    KeywordGroup("balance_sheet", ("资产负债表", "資產負債表", "財務狀況表", "财务状况表", "財務狀況報表", "财务状况报表"), True),
    KeywordGroup("income_statement", ("利润表", "利潤表", "合併利潤表", "損益表", "合併損益表", "收益表", "全面收益表", "綜合收益表", "综合收益表", "經營狀況表", "经营状况表", "合併經營狀況表", "合并经营状况表", "綜合虧損表", "综合亏损表", "合併綜合虧損表", "合并综合亏损表", "損益及其他全面收入", "損益及其他全面收益"), True),
    KeywordGroup("cash_flow", ("现金流量表", "現金流量表", "現金流動表", "现金流动表", "綜合現金流動表", "综合现金流动表"), True),
)
FINANCIAL_GROUPS_QUARTERLY = (
    KeywordGroup("financial_data", ("主要财务数据", "主要会计数据", "主要財務資料"), True),
    *FINANCIAL_GROUPS_FULL,
)
FINANCIAL_GROUPS_MATERIAL = (
    KeywordGroup("financial_summary", ("业绩快报", "業績快報", "业绩预告", "盈利預警", "财务摘要", "財務摘要")),
)
MATERIAL_FINANCIAL_KEYWORDS = (
    "业绩快报",
    "業績快報",
    "业绩预告",
    "業績預告",
    "盈利预警",
    "盈利預警",
    "财务摘要",
    "財務摘要",
    "经营业绩",
    "經營業績",
    "营业收入",
    "營業收入",
    "净利润",
    "淨利潤",
    "每股收益",
)
TABLE_REQUIREMENT_REQUIRED = "required"
TABLE_REQUIREMENT_NOT_APPLICABLE = "not_applicable"
SEARCH_CALL_COUNT_ERROR = "search_call_count"

PROFILES: dict[str, DoclingProfile] = {
    REPORT_KIND_ANNUAL: DoclingProfile(
        report_kind=REPORT_KIND_ANNUAL,
        section_count_low=20,
        section_count_high=220,
        search_t5=0.55,
        search_t7=0.70,
        search_t9=0.85,
        key_groups=ANNUAL_KEY_GROUPS,
        financial_groups=FINANCIAL_GROUPS_FULL,
        require_page_evidence=True,
    ),
    REPORT_KIND_SEMIANNUAL: DoclingProfile(
        report_kind=REPORT_KIND_SEMIANNUAL,
        section_count_low=12,
        section_count_high=140,
        search_t5=0.55,
        search_t7=0.70,
        search_t9=0.85,
        key_groups=SEMIANNUAL_KEY_GROUPS,
        financial_groups=FINANCIAL_GROUPS_FULL,
        require_page_evidence=True,
    ),
    REPORT_KIND_QUARTERLY: DoclingProfile(
        report_kind=REPORT_KIND_QUARTERLY,
        section_count_low=6,
        section_count_high=90,
        search_t5=0.45,
        search_t7=0.60,
        search_t9=0.75,
        key_groups=QUARTERLY_KEY_GROUPS,
        financial_groups=FINANCIAL_GROUPS_QUARTERLY,
        require_page_evidence=False,
    ),
    REPORT_KIND_MATERIAL: DoclingProfile(
        report_kind=REPORT_KIND_MATERIAL,
        section_count_low=1,
        section_count_high=50,
        search_t5=0.25,
        search_t7=0.40,
        search_t9=0.55,
        key_groups=MATERIAL_KEY_GROUPS,
        financial_groups=FINANCIAL_GROUPS_MATERIAL,
        require_page_evidence=False,
    ),
}


def _resolve_workspace_root(base: str) -> Path:
    """将 CLI ``--base`` 解析为 workspace 根目录。

    Args:
        base: workspace 根目录或 ``portfolio`` 目录。

    Returns:
        workspace 根目录。

    Raises:
        无。
    """

    normalized = Path(str(base)).resolve()
    if normalized.name == "portfolio":
        return normalized.parent
    return normalized


def _as_json_object(value: JsonValue) -> JsonObject:
    """把 JSON 值收窄为对象。

    Args:
        value: JSON 值。

    Returns:
        JSON 对象；非对象返回空字典。

    Raises:
        无。
    """

    if isinstance(value, dict):
        return cast(JsonObject, value)
    return {}


def _as_json_list(value: JsonValue) -> list[JsonValue]:
    """把 JSON 值收窄为列表。

    Args:
        value: JSON 值。

    Returns:
        JSON 列表；非列表返回空列表。

    Raises:
        无。
    """

    if isinstance(value, list):
        return cast(list[JsonValue], value)
    return []


def _as_json_objects(value: JsonValue) -> list[JsonObject]:
    """把 JSON 值收窄为对象列表。

    Args:
        value: JSON 值。

    Returns:
        对象列表；非对象元素会被跳过。

    Raises:
        无。
    """

    return [
        cast(JsonObject, item)
        for item in _as_json_list(value)
        if isinstance(item, dict)
    ]


def _string_value(data: JsonObject, key: str) -> str:
    """读取 JSON 对象中的字符串值。

    Args:
        data: JSON 对象。
        key: 字段名。

    Returns:
        去空白字符串；非标量返回空字符串。

    Raises:
        无。
    """

    value = data.get(key)
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return ""
    return str(value).strip()


def _bool_value(data: JsonObject, key: str) -> bool:
    """读取 JSON 对象中的布尔值。

    Args:
        data: JSON 对象。
        key: 字段名。

    Returns:
        布尔值；缺失或非布尔时返回 ``False``。

    Raises:
        无。
    """

    value = data.get(key)
    return bool(value) if isinstance(value, bool) else False


def _int_value(data: JsonObject, key: str) -> Optional[int]:
    """读取 JSON 对象中的整数值。

    Args:
        data: JSON 对象。
        key: 字段名。

    Returns:
        整数；无法读取时返回 ``None``。

    Raises:
        无。
    """

    value = data.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _float_value(data: JsonObject, key: str, default: float) -> float:
    """读取 JSON 对象中的浮点值。

    Args:
        data: JSON 对象。
        key: 字段名。
        default: 缺失或非法时的默认值。

    Returns:
        浮点值。

    Raises:
        无。
    """

    value = data.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _json_text(value: JsonValue) -> str:
    """把 JSON 标量转换为文本。

    Args:
        value: JSON 值。

    Returns:
        字符串；列表和对象返回空字符串。

    Raises:
        无。
    """

    if value is None or isinstance(value, (list, dict)):
        return ""
    return str(value)


def _load_json_snapshot(
    *,
    snapshot: ProcessedSnapshotDocument,
    blob_repository: DocumentBlobRepositoryProtocol,
    file_name: str,
) -> SnapshotLoadResult:
    """读取单个工具快照 JSON。

    Args:
        snapshot: processed 快照访问对象。
        blob_repository: blob 仓储。
        file_name: 快照文件名。

    Returns:
        加载结果。

    Raises:
        OSError: 底层仓储读取失败时可能抛出。
    """

    try:
        payload_bytes = blob_repository.read_file_bytes(snapshot.handle, file_name)
    except FileNotFoundError:
        return SnapshotLoadResult(payload={}, exists=False, valid=False)
    try:
        parsed: JsonValue = cast(JsonValue, json.loads(payload_bytes.decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return SnapshotLoadResult(payload={}, exists=True, valid=False)
    if not isinstance(parsed, dict):
        return SnapshotLoadResult(payload={}, exists=True, valid=False)
    return SnapshotLoadResult(payload=cast(JsonObject, parsed), exists=True, valid=True)


def _load_snapshot_meta_required(
    *,
    snapshot: ProcessedSnapshotDocument,
    blob_repository: DocumentBlobRepositoryProtocol,
) -> JsonObject:
    """读取并校验 ``tool_snapshot_meta.json``。

    Args:
        snapshot: processed 快照访问对象。
        blob_repository: blob 仓储。

    Returns:
        已校验的 meta 对象。

    Raises:
        ValueError: meta 缺失或字段不满足当前 CI 契约时抛出。
        OSError: 底层仓储读取失败时可能抛出。
    """

    load = _load_json_snapshot(
        snapshot=snapshot,
        blob_repository=blob_repository,
        file_name=TOOL_SNAPSHOT_META_FILE_NAME,
    )
    if not load.exists:
        raise ValueError(f"缺少 {TOOL_SNAPSHOT_META_FILE_NAME}")
    if not load.valid:
        raise ValueError(f"{TOOL_SNAPSHOT_META_FILE_NAME} 不是合法 JSON 对象")
    meta = load.payload
    if _string_value(meta, "snapshot_schema_version") != REQUIRED_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("tool_snapshot_meta.snapshot_schema_version 不匹配")
    if _string_value(meta, "source_kind") != snapshot.source_kind.value:
        raise ValueError("tool_snapshot_meta.source_kind 与 source manifest 不一致")
    if _string_value(meta, "document_id") and _string_value(meta, "document_id") != snapshot.document_id:
        raise ValueError("tool_snapshot_meta.document_id 与 source manifest 不一致")
    market = _string_value(meta, "market").upper()
    if market not in {"CN", "HK"}:
        raise ValueError("tool_snapshot_meta.market 必须为 CN 或 HK")
    if not _string_value(meta, "document_type"):
        raise ValueError("tool_snapshot_meta.document_type 缺失")
    pack_name = _string_value(meta, "search_query_pack_name")
    pack_version = _string_value(meta, "search_query_pack_version")
    search_query_count = _int_value(meta, "search_query_count")
    search_queries = _as_json_list(meta.get("search_queries", []))
    if not pack_name:
        raise ValueError("tool_snapshot_meta.search_query_pack_name 缺失")
    if not pack_version:
        raise ValueError("tool_snapshot_meta.search_query_pack_version 缺失")
    if search_query_count is None or search_query_count <= 0:
        raise ValueError("tool_snapshot_meta.search_query_count 必须为正整数")
    if search_query_count != len(search_queries):
        raise ValueError("tool_snapshot_meta.search_query_count 与 search_queries 数量不一致")
    return meta


def _calls_from_payload(payload: JsonObject) -> list[JsonObject]:
    """从工具快照载荷中读取 calls。

    Args:
        payload: 工具快照 JSON 对象。

    Returns:
        调用对象列表。

    Raises:
        无。
    """

    return _as_json_objects(payload.get("calls", []))


def _first_response(calls: list[JsonObject]) -> JsonObject:
    """读取第一条工具调用响应。

    Args:
        calls: 工具调用列表。

    Returns:
        第一条响应对象；缺失时返回空对象。

    Raises:
        无。
    """

    if not calls:
        return {}
    return _as_json_object(calls[0].get("response", {}))


def _sections_from_calls(calls: list[JsonObject]) -> list[JsonObject]:
    """从 ``get_document_sections`` calls 提取 sections。

    Args:
        calls: 工具调用列表。

    Returns:
        section 对象列表。

    Raises:
        无。
    """

    response = _first_response(calls)
    return _as_json_objects(response.get("sections", []))


def _read_map_from_calls(calls: list[JsonObject]) -> dict[str, JsonObject]:
    """从 ``read_section`` calls 构建 ref -> response 映射。

    Args:
        calls: 工具调用列表。

    Returns:
        section ref 到响应对象的映射。

    Raises:
        无。
    """

    result: dict[str, JsonObject] = {}
    for call in calls:
        response = _as_json_object(call.get("response", {}))
        ref = _string_value(response, "ref")
        if ref:
            result[ref] = response
    return result


def _tables_from_list_calls(calls: list[JsonObject]) -> list[JsonObject]:
    """从 ``list_tables`` calls 提取 tables。

    Args:
        calls: 工具调用列表。

    Returns:
        表格摘要列表。

    Raises:
        无。
    """

    response = _first_response(calls)
    return _as_json_objects(response.get("tables", []))


def _get_table_map_from_calls(calls: list[JsonObject]) -> dict[str, JsonObject]:
    """从 ``get_table`` calls 构建 table_ref -> response 映射。

    Args:
        calls: 工具调用列表。

    Returns:
        table_ref 到响应对象的映射。

    Raises:
        无。
    """

    result: dict[str, JsonObject] = {}
    for call in calls:
        response = _as_json_object(call.get("response", {}))
        table_ref = _string_value(response, "table_ref")
        if table_ref:
            result[table_ref] = response
    return result


def _normalize_report_kind(raw: str, source_kind: SourceKind) -> str:
    """归一化 CN/HK report kind。

    Args:
        raw: source meta 中的 report kind / form_type / fiscal_period。
        source_kind: 源文档类型。

    Returns:
        标准 report kind。

    Raises:
        无。
    """

    if source_kind == SourceKind.MATERIAL:
        return REPORT_KIND_MATERIAL
    normalized = raw.strip().lower().replace("-", "_")
    if normalized in {"annual", "fy", "year", "yearly"}:
        return REPORT_KIND_ANNUAL
    if normalized in {"semi_annual", "semiannual", "h1", "half_year", "interim"}:
        return REPORT_KIND_SEMIANNUAL
    if normalized in {"quarterly", "quarter", "q1", "q2", "q3", "q4"}:
        return REPORT_KIND_QUARTERLY
    return REPORT_KIND_QUARTERLY if normalized.startswith("q") else REPORT_KIND_ANNUAL


def _resolve_report_kind(source_meta: JsonObject, source_kind: SourceKind) -> str:
    """从 source meta 解析 report kind。

    Args:
        source_meta: source meta。
        source_kind: 源文档类型。

    Returns:
        标准 report kind。

    Raises:
        无。
    """

    for key in ("report_kind", "fiscal_period", "form_type"):
        raw = _string_value(source_meta, key)
        if raw:
            return _normalize_report_kind(raw, source_kind)
    return _normalize_report_kind("", source_kind)


def _source_kinds_from_filter(source_kind_filter: str) -> tuple[SourceKind, ...]:
    """解析 source kind 过滤条件。

    Args:
        source_kind_filter: CLI 过滤值。

    Returns:
        需要扫描的 source kind 元组。

    Raises:
        ValueError: 过滤值非法时抛出。
    """

    normalized = source_kind_filter.strip().lower()
    if normalized == SOURCE_KIND_ALL:
        return (SourceKind.FILING, SourceKind.MATERIAL)
    if normalized == SourceKind.FILING.value:
        return (SourceKind.FILING,)
    if normalized == SourceKind.MATERIAL.value:
        return (SourceKind.MATERIAL,)
    raise ValueError(f"不支持的 source_kind: {source_kind_filter}")


def _is_active_source(source_meta: JsonObject) -> bool:
    """判断 source 文档是否应纳入评分样本。

    Args:
        source_meta: source meta。

    Returns:
        是否为 active source。

    Raises:
        无。
    """

    if _bool_value(source_meta, "is_deleted"):
        return False
    ingest_complete = source_meta.get("ingest_complete")
    return not isinstance(ingest_complete, bool) or ingest_complete


def _resolve_tickers(
    *,
    tickers: list[str],
    company_repository: CompanyMetaRepositoryProtocol,
) -> list[str]:
    """解析评分 ticker 集合。

    Args:
        tickers: CLI 显式传入的 ticker 列表。
        company_repository: 公司元数据仓储。

    Returns:
        规范 ticker 列表。

    Raises:
        OSError: 扫描公司元数据失败时可能抛出。
    """

    if tickers:
        return sorted({ticker.strip().upper() for ticker in tickers if ticker.strip()})
    entries: list[CompanyMetaInventoryEntry] = company_repository.scan_company_meta_inventory()
    result: list[str] = []
    for entry in entries:
        if entry.status != "available" or entry.company_meta is None:
            continue
        result.append(entry.company_meta.ticker.strip().upper())
    return sorted({ticker for ticker in result if ticker})


def _discover_docling_snapshots(
    *,
    base: str,
    tickers: list[str],
    report_kind_filter: str,
    source_kind_filter: str,
    company_repository: Optional[CompanyMetaRepositoryProtocol] = None,
    source_repository: Optional[SourceDocumentRepositoryProtocol] = None,
    processed_repository: Optional[ProcessedDocumentRepositoryProtocol] = None,
) -> DiscoveryResult:
    """通过仓储发现 CN/HK Docling 评分样本。

    Args:
        base: workspace 根目录或 ``portfolio`` 目录。
        tickers: 显式 ticker 列表；为空时扫描公司元数据。
        report_kind_filter: report kind 过滤条件。
        source_kind_filter: source kind 过滤条件。
        company_repository: 可选公司元数据仓储。
        source_repository: 可选 source 仓储。
        processed_repository: 可选 processed 仓储。

    Returns:
        样本发现结果。

    Raises:
        ValueError: 过滤条件非法时抛出。
        OSError: 仓储读取失败时可能抛出。
    """

    workspace_root = _resolve_workspace_root(base)
    company_repo = company_repository or FsCompanyMetaRepository(workspace_root)
    source_repo = source_repository or FsSourceDocumentRepository(workspace_root)
    processed_repo = processed_repository or FsProcessedDocumentRepository(workspace_root)
    resolved_tickers = _resolve_tickers(tickers=tickers, company_repository=company_repo)
    source_kinds = _source_kinds_from_filter(source_kind_filter)
    normalized_report_filter = report_kind_filter.strip().lower()

    snapshots: list[ProcessedSnapshotDocument] = []
    failures: list[CompletenessFailure] = []
    for ticker in resolved_tickers:
        for source_kind in source_kinds:
            source_document_ids = source_repo.list_source_document_ids(ticker, source_kind)
            processed_summaries = processed_repo.list_processed_documents(
                ticker,
                DocumentQuery(source_kind=source_kind.value),
            )
            summary_by_document_id = {
                summary.document_id: summary
                for summary in processed_summaries
                if not summary.is_deleted
            }
            for document_id in sorted(source_document_ids):
                try:
                    source_meta_raw = source_repo.get_source_meta(ticker, document_id, source_kind)
                except FileNotFoundError:
                    continue
                source_meta = cast(JsonObject, source_meta_raw)
                if not _is_active_source(source_meta):
                    continue
                report_kind = _resolve_report_kind(source_meta, source_kind)
                if normalized_report_filter != REPORT_KIND_ALL and report_kind != normalized_report_filter:
                    continue
                summary = summary_by_document_id.get(document_id)
                if summary is None:
                    failures.append(
                        CompletenessFailure(
                            ticker=ticker,
                            document_id=document_id,
                            source_kind=source_kind.value,
                            reason="缺少 processed 快照",
                        )
                    )
                    continue
                snapshots.append(
                    ProcessedSnapshotDocument(
                        ticker=ticker,
                        document_id=document_id,
                        source_kind=source_kind,
                        report_kind=report_kind,
                        source_meta=source_meta,
                        handle=processed_repo.get_processed_handle(ticker, document_id),
                    )
                )
    return DiscoveryResult(snapshots=snapshots, completeness_failures=failures)


def _text_contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """判断文本是否包含任一关键词。

    Args:
        text: 待匹配文本。
        keywords: 关键词列表。

    Returns:
        是否命中。

    Raises:
        无。
    """

    normalized = _normalize_keyword_match_text(text)
    compact_normalized = _remove_match_whitespace(normalized)
    for keyword in keywords:
        normalized_keyword = _normalize_keyword_match_text(keyword)
        if normalized_keyword in normalized:
            return True
        compact_keyword = _remove_match_whitespace(normalized_keyword)
        if compact_keyword and compact_keyword in compact_normalized:
            return True
    return False


def _normalize_keyword_match_text(text: str) -> str:
    """归一化 profile 关键词匹配文本。

    Args:
        text: 原始文本。

    Returns:
        NFKC 与大小写折叠后的文本。

    Raises:
        无。
    """

    return unicodedata.normalize("NFKC", text).casefold()


def _remove_match_whitespace(text: str) -> str:
    """移除关键词匹配中的空白噪声。

    Args:
        text: 已归一化或待归一化文本。

    Returns:
        去除所有空白字符后的文本。

    Raises:
        无。
    """

    return re.sub(r"\s+", "", text)


def _matched_group_labels(text: str, groups: tuple[KeywordGroup, ...]) -> list[str]:
    """返回命中的关键词组标签。

    Args:
        text: 待匹配文本。
        groups: 关键词组。

    Returns:
        命中的标签列表。

    Raises:
        无。
    """

    return [
        group.label
        for group in groups
        if _text_contains_any(text, group.keywords)
    ]


def _section_titles_text(sections: list[JsonObject]) -> str:
    """拼接章节标题文本。

    Args:
        sections: section 对象列表。

    Returns:
        标题拼接文本。

    Raises:
        无。
    """

    return "\n".join(_string_value(section, "title") for section in sections)


def _all_read_content(read_map: dict[str, JsonObject]) -> str:
    """拼接 ``read_section.content`` 文本。

    Args:
        read_map: section ref 到 read_section 响应的映射。

    Returns:
        全部正文拼接文本。

    Raises:
        无。
    """

    return "\n".join(_string_value(section, "content") for section in read_map.values())


def _table_text(tables: list[JsonObject], get_table_map: dict[str, JsonObject]) -> str:
    """拼接表格标题、表头和所属章节文本。

    Args:
        tables: ``list_tables`` 表格摘要。
        get_table_map: ``get_table`` 响应映射。

    Returns:
        表格相关文本。

    Raises:
        无。
    """

    parts: list[str] = []
    for table in tables:
        parts.append(_string_value(table, "caption"))
        within_section = _as_json_object(table.get("within_section", {}))
        parts.append(_string_value(within_section, "title"))
        for header in _as_json_list(table.get("headers", [])):
            parts.append(_json_text(header))
        table_ref = _string_value(table, "table_ref")
        detail = get_table_map.get(table_ref, {})
        parts.append(_string_value(detail, "caption"))
        detail_within = _as_json_object(detail.get("within_section", {}))
        parts.append(_string_value(detail_within, "title"))
    return "\n".join(part for part in parts if part)


def _material_financial_evidence_text(
    *,
    snapshot: ProcessedSnapshotDocument,
    meta: JsonObject,
    sections: list[JsonObject],
    read_map: dict[str, JsonObject],
    tables: list[JsonObject],
    get_table_map: dict[str, JsonObject],
) -> str:
    """拼接判断 material 是否业绩类所需的现有证据。

    Args:
        snapshot: processed 快照访问对象。
        meta: snapshot meta。
        sections: section 列表。
        read_map: ``read_section`` 映射。
        tables: ``list_tables`` 表格摘要。
        get_table_map: ``get_table`` 响应映射。

    Returns:
        用于关键词判断的文本。

    Raises:
        无。
    """

    parts = [
        _string_value(meta, "document_type"),
        _string_value(meta, "form_type"),
        _string_value(snapshot.source_meta, "document_type"),
        _string_value(snapshot.source_meta, "form_type"),
        _section_titles_text(sections),
        _all_read_content(read_map),
        _table_text(tables, get_table_map),
    ]
    return "\n".join(part for part in parts if part)


def _material_requires_financial_tables(
    *,
    snapshot: ProcessedSnapshotDocument,
    meta: JsonObject,
    sections: list[JsonObject],
    read_map: dict[str, JsonObject],
    tables: list[JsonObject],
    get_table_map: dict[str, JsonObject],
) -> bool:
    """判断 material 是否应启用财务表覆盖评分。

    Args:
        snapshot: processed 快照访问对象。
        meta: snapshot meta。
        sections: section 列表。
        read_map: ``read_section`` 映射。
        tables: ``list_tables`` 表格摘要。
        get_table_map: ``get_table`` 响应映射。

    Returns:
        业绩类材料返回 ``True``，普通公告返回 ``False``。

    Raises:
        无。
    """

    if snapshot.report_kind != REPORT_KIND_MATERIAL:
        return True
    evidence_text = _material_financial_evidence_text(
        snapshot=snapshot,
        meta=meta,
        sections=sections,
        read_map=read_map,
        tables=tables,
        get_table_map=get_table_map,
    )
    return _text_contains_any(evidence_text, MATERIAL_FINANCIAL_KEYWORDS)


def _effective_financial_groups(
    *,
    profile: DoclingProfile,
    material_requires_financial_tables: bool,
) -> tuple[KeywordGroup, ...]:
    """按文档类型返回实际参与评分的财务关键词组。

    Args:
        profile: report kind profile。
        material_requires_financial_tables: material 是否属于业绩类材料。

    Returns:
        参与 A/D 维评分的财务关键词组。

    Raises:
        无。
    """

    if profile.report_kind == REPORT_KIND_MATERIAL and not material_requires_financial_tables:
        return ()
    return profile.financial_groups


def _extract_table_placeholders(read_map: dict[str, JsonObject]) -> set[str]:
    """从 ``read_section.content`` 提取表格占位符。

    Args:
        read_map: section ref 到 read_section 响应的映射。

    Returns:
        表格 ref 集合。

    Raises:
        无。
    """

    result: set[str] = set()
    for section in read_map.values():
        content = _string_value(section, "content")
        result.update(TABLE_PLACEHOLDER_RE.findall(content))
    return result


def _table_refs_from_list_tables(tables: list[JsonObject]) -> set[str]:
    """从 ``list_tables`` 提取 table ref 集合。

    Args:
        tables: 表格摘要列表。

    Returns:
        table ref 集合。

    Raises:
        无。
    """

    return {
        _string_value(table, "table_ref")
        for table in tables
        if _string_value(table, "table_ref")
    }


def _section_refs(sections: list[JsonObject]) -> set[str]:
    """提取 section ref 集合。

    Args:
        sections: section 对象列表。

    Returns:
        section ref 集合。

    Raises:
        无。
    """

    return {
        _string_value(section, "ref")
        for section in sections
        if _string_value(section, "ref")
    }


def _table_section_refs(tables: list[JsonObject], get_table_map: dict[str, JsonObject]) -> set[str]:
    """提取表格所属 section ref 集合。

    Args:
        tables: ``list_tables`` 表格摘要。
        get_table_map: ``get_table`` 响应映射。

    Returns:
        section ref 集合。

    Raises:
        无。
    """

    refs: set[str] = set()
    for table in tables:
        direct = _string_value(table, "section_ref")
        within = _as_json_object(table.get("within_section", {}))
        within_ref = _string_value(within, "ref")
        if direct:
            refs.add(direct)
        if within_ref:
            refs.add(within_ref)
        detail = get_table_map.get(_string_value(table, "table_ref"), {})
        detail_direct = _string_value(detail, "section_ref")
        detail_within = _as_json_object(detail.get("within_section", {}))
        detail_within_ref = _string_value(detail_within, "ref")
        if detail_direct:
            refs.add(detail_direct)
        if detail_within_ref:
            refs.add(detail_within_ref)
    return refs


def _valid_page_range(section: JsonObject) -> bool:
    """判断 section page_range 是否有效。

    Args:
        section: section 对象。

    Returns:
        page_range 是否为正整数闭区间。

    Raises:
        无。
    """

    raw = _as_json_list(section.get("page_range", []))
    if len(raw) != 2:
        return False
    start = raw[0]
    end = raw[1]
    if not isinstance(start, int) or isinstance(start, bool):
        return False
    if not isinstance(end, int) or isinstance(end, bool):
        return False
    return start > 0 and end >= start


def _table_page_no(table: JsonObject, get_table_map: dict[str, JsonObject]) -> Optional[int]:
    """读取表格页码。

    Args:
        table: ``list_tables`` 表格摘要。
        get_table_map: ``get_table`` 响应映射。

    Returns:
        正整数页码；无法读取时返回 ``None``。

    Raises:
        无。
    """

    page_no = _int_value(table, "page_no")
    if page_no is not None and page_no > 0:
        return page_no
    detail = get_table_map.get(_string_value(table, "table_ref"), {})
    detail_page_no = _int_value(detail, "page_no")
    if detail_page_no is not None and detail_page_no > 0:
        return detail_page_no
    return None


def _score_ratio_points(ratio: float, max_points: float) -> float:
    """按比例给分。

    Args:
        ratio: 比例，按 ``0..1`` 处理。
        max_points: 最大分。

    Returns:
        得分。

    Raises:
        无。
    """

    bounded = max(0.0, min(1.0, ratio))
    return bounded * max_points


def _score_structure(
    *,
    profile: DoclingProfile,
    financial_groups: tuple[KeywordGroup, ...],
    sections: list[JsonObject],
    tables: list[JsonObject],
    get_table_map: dict[str, JsonObject],
) -> DimensionScore:
    """评分 A：结构可导航性。

    Args:
        profile: report kind profile。
        financial_groups: 实际参与评分的财务关键词组。
        sections: section 列表。
        tables: 表格摘要列表。
        get_table_map: ``get_table`` 响应映射。

    Returns:
        维度评分。

    Raises:
        无。
    """

    section_count = len(sections)
    if profile.section_count_low <= section_count <= profile.section_count_high:
        count_points = 4.0
    elif section_count > 0:
        count_points = 2.0
    else:
        count_points = 0.0

    titled_sections = [
        section for section in sections
        if _string_value(section, "title") and len(_string_value(section, "title")) <= 120
    ]
    title_ratio = len(titled_sections) / section_count if section_count else 0.0
    title_points = _score_ratio_points(title_ratio, 4.0)

    title_text = _section_titles_text(sections)
    table_context = _table_text(tables, get_table_map)
    matched_key_groups = _matched_group_labels(title_text, profile.key_groups)
    matched_financial_groups = _matched_group_labels(f"{title_text}\n{table_context}", financial_groups)
    key_points = _score_ratio_points(
        len(matched_key_groups) / len(profile.key_groups) if profile.key_groups else 1.0,
        8.0,
    )
    financial_points = _score_ratio_points(
        len(matched_financial_groups) / len(financial_groups) if financial_groups else 1.0,
        3.0,
    )

    child_sections = [
        section for section in sections
        if _string_value(section, "parent_ref") or (_int_value(section, "level") or 1) > 1
    ]
    hierarchy_points = 1.0 if child_sections or profile.report_kind in {REPORT_KIND_MATERIAL, REPORT_KIND_QUARTERLY} else 0.0

    points = count_points + title_points + key_points + financial_points + hierarchy_points
    return DimensionScore(
        points=round(points, 2),
        max_points=MAX_A_STRUCTURE,
        details={
            "section_count": section_count,
            "section_count_low": profile.section_count_low,
            "section_count_high": profile.section_count_high,
            "title_ratio": round(title_ratio, 4),
            "matched_key_groups": matched_key_groups,
            "missing_key_groups": [
                group.label for group in profile.key_groups if group.label not in matched_key_groups
            ],
            "matched_financial_groups": matched_financial_groups,
            "missing_financial_groups": [
                group.label for group in financial_groups if group.label not in matched_financial_groups
            ],
            "financial_table_requirement": (
                TABLE_REQUIREMENT_REQUIRED if financial_groups else TABLE_REQUIREMENT_NOT_APPLICABLE
            ),
            "hierarchy_section_count": len(child_sections),
        },
    )


def _score_text(
    *,
    read_map: dict[str, JsonObject],
    table_refs: set[str],
) -> DimensionScore:
    """评分 B：文本可读性。

    Args:
        read_map: section ref 到 ``read_section`` 响应映射。
        table_refs: ``list_tables`` 暴露的 table refs。

    Returns:
        维度评分。

    Raises:
        无。
    """

    sections = list(read_map.values())
    nonempty = [
        section for section in sections
        if len(_string_value(section, "content").strip()) > NEAR_EMPTY_SECTION_CHARS
    ]
    nonempty_ratio = len(nonempty) / len(sections) if sections else 0.0
    nonempty_points = _score_ratio_points(nonempty_ratio, 4.0)

    full_text = _all_read_content(read_map)
    mojibake_count = len(MOJIBAKE_RE.findall(full_text))
    mojibake_points = 3.0 if mojibake_count == 0 else 0.0

    placeholders = _extract_table_placeholders(read_map)
    dangling = sorted(ref for ref in placeholders if ref not in table_refs)
    placeholder_points = 3.0 if not dangling else 0.0

    toc_line_count = len(TOC_LINE_RE.findall(full_text))
    toc_points = 3.0 if toc_line_count <= 4 else 0.0

    cjk_count = len(CJK_CHAR_RE.findall(full_text))
    spaced_cjk_count = len(CJK_SPACED_RE.findall(full_text))
    spaced_ratio = spaced_cjk_count / cjk_count if cjk_count else 0.0
    cjk_points = 2.0 if spaced_ratio <= 0.02 else 0.0

    points = nonempty_points + mojibake_points + placeholder_points + toc_points + cjk_points
    return DimensionScore(
        points=round(points, 2),
        max_points=MAX_B_TEXT,
        details={
            "section_count": len(sections),
            "nonempty_section_count": len(nonempty),
            "nonempty_ratio": round(nonempty_ratio, 4),
            "mojibake_count": mojibake_count,
            "placeholder_count": len(placeholders),
            "dangling_placeholders": dangling,
            "toc_line_count": toc_line_count,
            "cjk_spaced_ratio": round(spaced_ratio, 4),
        },
    )


def _search_matches(response: JsonObject) -> list[JsonObject]:
    """读取搜索响应中的 matches。

    Args:
        response: ``search_document`` 响应。

    Returns:
        match 对象列表。

    Raises:
        无。
    """

    return _as_json_objects(response.get("matches", []))


def _match_section_ref(match: JsonObject) -> str:
    """读取搜索命中的 section ref。

    Args:
        match: 搜索命中对象。

    Returns:
        section ref；缺失时为空字符串。

    Raises:
        无。
    """

    section = _as_json_object(match.get("section", {}))
    return _string_value(section, "ref")


def _match_context(match: JsonObject) -> str:
    """读取搜索命中的证据上下文。

    Args:
        match: 搜索命中对象。

    Returns:
        evidence.context 或 snippet。

    Raises:
        无。
    """

    evidence = _as_json_object(match.get("evidence", {}))
    context = _string_value(evidence, "context")
    if context:
        return context
    return _string_value(match, "snippet")


def _score_search(
    *,
    profile: DoclingProfile,
    meta: JsonObject,
    search_calls: list[JsonObject],
    search_exists: bool,
) -> DimensionScore:
    """评分 C：搜索可用性。

    Args:
        profile: report kind profile。
        meta: snapshot meta。
        search_calls: ``search_document`` calls。
        search_exists: 搜索快照是否存在。

    Returns:
        维度评分。

    Raises:
        无。
    """

    expected_query_count = _int_value(meta, "search_query_count") or 0
    if not search_exists:
        return DimensionScore(
            points=0.0,
            max_points=MAX_C_SEARCH,
            details={
                "missing_snapshots": [TOOL_SNAPSHOT_SEARCH_FILE_NAME],
                "expected_search_query_count": expected_query_count,
                "call_count": 0,
                "missing_search_call_count": expected_query_count,
                "extra_search_call_count": 0,
                "search_call_count_matches_meta": expected_query_count == 0,
            },
        )

    total_weight = 0.0
    hit_weight = 0.0
    hit_queries = 0
    evidence_good = 0
    exact_hit_queries = 0
    for call in search_calls:
        request = _as_json_object(call.get("request", {}))
        response = _as_json_object(call.get("response", {}))
        weight = _float_value(request, "query_weight", 1.0)
        if weight <= 0.0:
            weight = 1.0
        total_weight += weight
        matches = _search_matches(response)
        if not matches:
            continue
        hit_weight += weight
        hit_queries += 1
        if any(_match_section_ref(match) and 20 <= len(_match_context(match)) <= 1500 for match in matches):
            evidence_good += 1
        diagnostics = _as_json_object(response.get("diagnostics", {}))
        strategy_counts = _as_json_object(diagnostics.get("strategy_hit_counts", {}))
        exact_count = _int_value(strategy_counts, SEARCH_STRATEGY_EXACT) or 0
        if exact_count > 0 or any(_bool_value(match, "is_exact_phrase") for match in matches):
            exact_hit_queries += 1

    missing_search_calls = max(0, expected_query_count - len(search_calls))
    extra_search_calls = max(0, len(search_calls) - expected_query_count) if expected_query_count else 0
    total_weight += float(missing_search_calls)
    coverage_rate = hit_weight / total_weight if total_weight > 0.0 else 0.0
    if coverage_rate >= profile.search_t9:
        coverage_points = 9.0
    elif coverage_rate >= profile.search_t7:
        coverage_points = 7.0
    elif coverage_rate >= profile.search_t5:
        coverage_points = 5.0
    else:
        coverage_points = 0.0

    evidence_quality_rate = evidence_good / hit_queries if hit_queries else 0.0
    if evidence_quality_rate >= 0.90:
        evidence_points = 4.0
    elif evidence_quality_rate >= 0.75:
        evidence_points = 2.0
    else:
        evidence_points = 0.0

    efficiency_rate = exact_hit_queries / hit_queries if hit_queries else 0.0
    if efficiency_rate >= 0.70:
        efficiency_points = 2.0
    elif efficiency_rate >= 0.50:
        efficiency_points = 1.0
    else:
        efficiency_points = 0.0

    count_alignment_penalty = 3.0 if expected_query_count and len(search_calls) != expected_query_count else 0.0
    points = max(0.0, coverage_points + evidence_points + efficiency_points - count_alignment_penalty)
    return DimensionScore(
        points=round(points, 2),
        max_points=MAX_C_SEARCH,
        details={
            "search_query_pack_name": _string_value(meta, "search_query_pack_name"),
            "search_query_pack_version": _string_value(meta, "search_query_pack_version"),
            "search_query_count": expected_query_count,
            "call_count": len(search_calls),
            "missing_search_call_count": missing_search_calls,
            "extra_search_call_count": extra_search_calls,
            "search_call_count_matches_meta": len(search_calls) == expected_query_count,
            "count_alignment_penalty": count_alignment_penalty,
            "coverage_rate_weighted": round(coverage_rate, 4),
            "hit_query_count": hit_queries,
            "evidence_quality_rate": round(evidence_quality_rate, 4),
            "efficiency_rate": round(efficiency_rate, 4),
            "thresholds": {
                "coverage_t5": profile.search_t5,
                "coverage_t7": profile.search_t7,
                "coverage_t9": profile.search_t9,
            },
        },
    )


def _table_has_meaningful_context(table: JsonObject, detail: JsonObject) -> bool:
    """判断表格是否有可解释上下文。

    Args:
        table: ``list_tables`` 表格摘要。
        detail: ``get_table`` 表格详情。

    Returns:
        是否具备 caption、within_section 或 headers。

    Raises:
        无。
    """

    if _string_value(table, "caption") or _string_value(detail, "caption"):
        return True
    within = _as_json_object(table.get("within_section", {}))
    detail_within = _as_json_object(detail.get("within_section", {}))
    if _string_value(within, "title") or _string_value(detail_within, "title"):
        return True
    headers = _as_json_list(table.get("headers", []))
    return any(_json_text(header).strip() for header in headers)


def _table_data_has_content(detail: JsonObject) -> bool:
    """判断 ``get_table`` 数据是否包含可消费内容。

    Args:
        detail: ``get_table`` 响应。

    Returns:
        是否有内容。

    Raises:
        无。
    """

    data = _as_json_object(detail.get("data", {}))
    markdown = _string_value(data, "markdown")
    if markdown and "|" in markdown:
        return True
    records = _as_json_list(data.get("records", []))
    if records:
        return True
    raw_text = _string_value(data, "raw_text")
    return bool(raw_text.strip())


def _headers_are_readable(table: JsonObject) -> bool:
    """判断表头是否可读。

    Args:
        table: 表格摘要。

    Returns:
        表头是否可读。

    Raises:
        无。
    """

    headers = [_json_text(header).strip() for header in _as_json_list(table.get("headers", []))]
    meaningful = [header for header in headers if header]
    if not meaningful:
        return False
    return not any(DEFAULT_HEADER_RE.match(header) for header in meaningful)


def _table_has_null_noise(detail: JsonObject) -> bool:
    """判断表格详情是否包含明显空洞噪声。

    Args:
        detail: ``get_table`` 响应。

    Returns:
        是否存在 NaN/null 字符噪声。

    Raises:
        无。
    """

    rendered = json.dumps(detail, ensure_ascii=False)
    lowered = rendered.lower()
    return "nan" in lowered or "null" in lowered


def _score_table(
    *,
    profile: DoclingProfile,
    financial_groups: tuple[KeywordGroup, ...],
    tables: list[JsonObject],
    get_table_map: dict[str, JsonObject],
    placeholder_refs: set[str],
    tables_snapshot_valid: bool,
) -> DimensionScore:
    """评分 D：表格可消费性。

    Args:
        profile: report kind profile。
        financial_groups: 实际参与评分的财务关键词组。
        tables: ``list_tables`` 表格摘要。
        get_table_map: ``get_table`` 响应映射。
        placeholder_refs: 正文中出现的表格占位符。
        tables_snapshot_valid: ``list_tables`` 快照是否存在且合法。

    Returns:
        维度评分。

    Raises:
        无。
    """

    if not tables_snapshot_valid:
        return DimensionScore(
            points=0.0,
            max_points=MAX_D_TABLE,
            details={
                "missing_snapshots": [TOOL_SNAPSHOT_LIST_TABLES_FILE_NAME],
                "table_count": 0,
                "placeholder_ref_count": len(placeholder_refs),
                "financial_table_requirement": (
                    TABLE_REQUIREMENT_REQUIRED if financial_groups else TABLE_REQUIREMENT_NOT_APPLICABLE
                ),
            },
        )

    table_refs = _table_refs_from_list_tables(tables)
    missing_details = sorted(ref for ref in table_refs if ref not in get_table_map)
    dangling_placeholders = sorted(ref for ref in placeholder_refs if ref not in table_refs)

    ref_points = 5.0 if not missing_details and not dangling_placeholders else 0.0

    table_context = _table_text(tables, get_table_map)
    matched_financial = _matched_group_labels(table_context, financial_groups)
    financial_points = _score_ratio_points(
        len(matched_financial) / len(financial_groups) if financial_groups else 1.0,
        4.0,
    )

    context_good = 0
    readable_headers = 0
    data_good = 0
    null_noise = 0
    for table in tables:
        table_ref = _string_value(table, "table_ref")
        detail = get_table_map.get(table_ref, {})
        if _table_has_meaningful_context(table, detail):
            context_good += 1
        if _headers_are_readable(table):
            readable_headers += 1
        if _table_data_has_content(detail):
            data_good += 1
        if _table_has_null_noise(detail):
            null_noise += 1

    table_count = len(tables)
    if profile.report_kind == REPORT_KIND_MATERIAL and table_count == 0 and not placeholder_refs and not financial_groups:
        return DimensionScore(
            points=MAX_D_TABLE,
            max_points=MAX_D_TABLE,
            details={
                "table_count": table_count,
                "placeholder_ref_count": len(placeholder_refs),
                "missing_get_table_refs": missing_details,
                "dangling_placeholders": dangling_placeholders,
                "matched_financial_groups": matched_financial,
                "missing_financial_groups": [],
                "financial_table_requirement": TABLE_REQUIREMENT_NOT_APPLICABLE,
                "context_good_count": 0,
                "readable_header_count": 0,
                "data_good_count": 0,
                "null_noise_table_count": 0,
            },
        )

    context_points = _score_ratio_points(context_good / table_count if table_count else 0.0, 3.0)
    header_points = _score_ratio_points(readable_headers / table_count if table_count else 0.0, 3.0)
    data_points = _score_ratio_points(data_good / table_count if table_count else 0.0, 4.0)
    null_points = 1.0 if null_noise == 0 else 0.0
    points = ref_points + financial_points + context_points + header_points + data_points + null_points
    return DimensionScore(
        points=round(points, 2),
        max_points=MAX_D_TABLE,
        details={
            "table_count": table_count,
            "placeholder_ref_count": len(placeholder_refs),
            "missing_get_table_refs": missing_details,
            "dangling_placeholders": dangling_placeholders,
            "matched_financial_groups": matched_financial,
            "missing_financial_groups": [
                group.label for group in financial_groups if group.label not in matched_financial
            ],
            "financial_table_requirement": (
                TABLE_REQUIREMENT_REQUIRED if financial_groups else TABLE_REQUIREMENT_NOT_APPLICABLE
            ),
            "context_good_count": context_good,
            "readable_header_count": readable_headers,
            "data_good_count": data_good,
            "null_noise_table_count": null_noise,
        },
    )


def _meta_consistency_errors(meta: JsonObject, snapshot: ProcessedSnapshotDocument) -> list[str]:
    """检查 snapshot meta 与 source 样本是否一致。

    Args:
        meta: snapshot meta。
        snapshot: processed 快照访问对象。

    Returns:
        错误列表。

    Raises:
        无。
    """

    errors: list[str] = []
    if _string_value(meta, "ticker") and _string_value(meta, "ticker") != snapshot.ticker:
        errors.append("ticker")
    if _string_value(meta, "document_id") and _string_value(meta, "document_id") != snapshot.document_id:
        errors.append("document_id")
    if _string_value(meta, "source_kind") != snapshot.source_kind.value:
        errors.append("source_kind")
    count = _int_value(meta, "search_query_count")
    queries = _as_json_list(meta.get("search_queries", []))
    if count is None or count != len(queries):
        errors.append("search_query_count")
    if not _string_value(meta, "search_query_pack_name"):
        errors.append("search_query_pack_name")
    if not _string_value(meta, "search_query_pack_version"):
        errors.append("search_query_pack_version")
    return errors


def _score_traceability(
    *,
    snapshot: ProcessedSnapshotDocument,
    meta: JsonObject,
    sections: list[JsonObject],
    read_map: dict[str, JsonObject],
    tables: list[JsonObject],
    get_table_map: dict[str, JsonObject],
    placeholder_refs: set[str],
    missing_snapshots: list[str],
    search_call_count: int,
) -> DimensionScore:
    """评分 E：一致性与可追溯性。

    Args:
        snapshot: processed 快照访问对象。
        meta: snapshot meta。
        sections: section 列表。
        read_map: ``read_section`` 映射。
        tables: ``list_tables`` 表格摘要。
        get_table_map: ``get_table`` 响应映射。
        placeholder_refs: 正文表格占位符。
        missing_snapshots: 缺失或损坏的非 meta 快照文件。
        search_call_count: 实际导出的 ``search_document`` call 数。

    Returns:
        维度评分。

    Raises:
        无。
    """

    section_refs = _section_refs(sections)
    read_refs = set(read_map.keys())
    missing_read_refs = sorted(ref for ref in section_refs if ref not in read_refs)
    extra_read_refs = sorted(ref for ref in read_refs if ref not in section_refs)
    section_points = 5.0 if not missing_read_refs and not extra_read_refs else 0.0

    table_refs = _table_refs_from_list_tables(tables)
    get_table_refs = set(get_table_map.keys())
    missing_get_table_refs = sorted(ref for ref in table_refs if ref not in get_table_refs)
    dangling_placeholders = sorted(ref for ref in placeholder_refs if ref not in table_refs)
    table_section_dangling = sorted(ref for ref in _table_section_refs(tables, get_table_map) if ref not in section_refs)
    table_points = 5.0 if not missing_get_table_refs and not dangling_placeholders and not table_section_dangling else 0.0

    meta_errors = _meta_consistency_errors(meta, snapshot)
    expected_search_query_count = _int_value(meta, "search_query_count") or 0
    search_call_errors: list[str] = []
    if expected_search_query_count and search_call_count != expected_search_query_count:
        search_call_errors.append(SEARCH_CALL_COUNT_ERROR)
    meta_points = 5.0 if not meta_errors and not search_call_errors else 0.0
    missing_points = 5.0 if not missing_snapshots else max(0.0, 5.0 - float(len(missing_snapshots)))
    points = section_points + table_points + meta_points + missing_points
    return DimensionScore(
        points=round(points, 2),
        max_points=MAX_E_TRACEABILITY,
        details={
            "missing_read_section_refs": missing_read_refs,
            "extra_read_section_refs": extra_read_refs,
            "missing_get_table_refs": missing_get_table_refs,
            "dangling_placeholders": dangling_placeholders,
            "dangling_table_section_refs": table_section_dangling,
            "meta_errors": meta_errors,
            "search_call_errors": search_call_errors,
            "expected_search_query_count": expected_search_query_count,
            "search_call_count": search_call_count,
            "missing_search_call_count": max(0, expected_search_query_count - search_call_count),
            "extra_search_call_count": max(0, search_call_count - expected_search_query_count)
            if expected_search_query_count else 0,
            "missing_snapshots": missing_snapshots,
        },
    )


def _score_page(
    *,
    profile: DoclingProfile,
    sections: list[JsonObject],
    tables: list[JsonObject],
    get_table_map: dict[str, JsonObject],
    page_calls: list[JsonObject],
    page_exists: bool,
) -> DimensionScore:
    """评分 F：Docling 页面定位。

    Args:
        profile: report kind profile。
        sections: section 列表。
        tables: ``list_tables`` 表格摘要。
        get_table_map: ``get_table`` 响应映射。
        page_calls: ``get_page_content`` calls。
        page_exists: 页面快照是否存在。

    Returns:
        维度评分。

    Raises:
        无。
    """

    valid_section_pages = sum(1 for section in sections if _valid_page_range(section))
    section_page_ratio = valid_section_pages / len(sections) if sections else 0.0
    section_points = _score_ratio_points(section_page_ratio, 4.0)

    table_pages = sum(1 for table in tables if _table_page_no(table, get_table_map) is not None)
    table_page_ratio = table_pages / len(tables) if tables else (1.0 if profile.report_kind == REPORT_KIND_MATERIAL else 0.0)
    table_points = _score_ratio_points(table_page_ratio, 3.0)

    if not page_exists:
        page_content_points = 0.0
        supported_pages = 0
        useful_pages = 0
    else:
        supported_pages = 0
        useful_pages = 0
        for call in page_calls:
            response = _as_json_object(call.get("response", {}))
            if _bool_value(response, "supported"):
                supported_pages += 1
            if _bool_value(response, "has_content"):
                useful_pages += 1
        page_content_points = _score_ratio_points(useful_pages / len(page_calls) if page_calls else 0.0, 3.0)

    points = section_points + table_points + page_content_points
    return DimensionScore(
        points=round(points, 2),
        max_points=MAX_F_PAGE,
        details={
            "valid_section_page_count": valid_section_pages,
            "section_count": len(sections),
            "section_page_ratio": round(section_page_ratio, 4),
            "table_page_count": table_pages,
            "table_count": len(tables),
            "table_page_ratio": round(table_page_ratio, 4),
            "page_snapshot_exists": page_exists,
            "page_call_count": len(page_calls),
            "supported_page_count": supported_pages,
            "useful_page_count": useful_pages,
        },
    )


def _evaluate_hard_gate(
    *,
    profile: DoclingProfile,
    read_map: dict[str, JsonObject],
    dim_a: DimensionScore,
    dim_b: DimensionScore,
    dim_d: DimensionScore,
    dim_e: DimensionScore,
    dim_f: DimensionScore,
    cfg: ScoreConfig,
) -> HardGateResult:
    """评估单文档 hard gate。

    Args:
        profile: report kind profile。
        read_map: ``read_section`` 映射。
        dim_a: A 维评分。
        dim_b: B 维评分。
        dim_d: D 维评分。
        dim_e: E 维评分。
        dim_f: F 维评分。
        cfg: 评分配置。

    Returns:
        hard gate 结果。

    Raises:
        无。
    """

    reasons: list[str] = []
    required_key_labels = [group.label for group in profile.key_groups if group.required]
    missing_key_groups = [
        label for label in _as_json_list(dim_a.details.get("missing_key_groups", []))
        if _json_text(label) in required_key_labels
    ]
    if missing_key_groups:
        reasons.append("缺少关键章节: " + ", ".join(_json_text(label) for label in missing_key_groups))

    required_financial_labels = [group.label for group in profile.financial_groups if group.required]
    missing_financial_groups = [
        label for label in _as_json_list(dim_d.details.get("missing_financial_groups", []))
        if _json_text(label) in required_financial_labels
    ]
    if missing_financial_groups and profile.report_kind != REPORT_KIND_MATERIAL:
        reasons.append("缺少关键财务表: " + ", ".join(_json_text(label) for label in missing_financial_groups))

    for section in read_map.values():
        content_len = len(_string_value(section, "content"))
        if content_len > cfg.huge_section_fail:
            reasons.append(f"超大 section: {content_len} chars")
            break

    dangling_text = _as_json_list(dim_e.details.get("dangling_placeholders", []))
    if dangling_text:
        reasons.append("悬挂 table placeholder: " + ", ".join(_json_text(item) for item in dangling_text))

    missing_read = _as_json_list(dim_e.details.get("missing_read_section_refs", []))
    extra_read = _as_json_list(dim_e.details.get("extra_read_section_refs", []))
    if missing_read or extra_read:
        reasons.append("section ref 不一致")

    meta_errors = _as_json_list(dim_e.details.get("meta_errors", []))
    if meta_errors:
        reasons.append("snapshot 元信息不一致: " + ", ".join(_json_text(item) for item in meta_errors))

    if _int_value(dim_b.details, "toc_line_count") is not None and (_int_value(dim_b.details, "toc_line_count") or 0) > 20:
        reasons.append("目录污染过高")

    if profile.require_page_evidence:
        section_page_ratio = _float_value(dim_f.details, "section_page_ratio", 0.0)
        table_page_ratio = _float_value(dim_f.details, "table_page_ratio", 0.0)
        if section_page_ratio == 0.0 and table_page_ratio == 0.0:
            reasons.append("关键章节和表格均无可复核页码")

    return HardGateResult(passed=not reasons, reasons=reasons)


def score_document(
    snapshot: ProcessedSnapshotDocument,
    blob_repository: DocumentBlobRepositoryProtocol,
    cfg: ScoreConfig,
) -> DocumentScore:
    """对单个 CN/HK Docling snapshot 评分。

    Args:
        snapshot: processed 快照访问对象。
        blob_repository: blob 仓储。
        cfg: 评分配置。

    Returns:
        单文档评分。

    Raises:
        ValueError: ``tool_snapshot_meta.json`` 缺失或非法时抛出。
        OSError: 底层仓储读取失败时可能抛出。
    """

    meta = _load_snapshot_meta_required(snapshot=snapshot, blob_repository=blob_repository)
    profile = PROFILES[snapshot.report_kind]

    sections_load = _load_json_snapshot(
        snapshot=snapshot,
        blob_repository=blob_repository,
        file_name=TOOL_SNAPSHOT_SECTIONS_FILE_NAME,
    )
    read_load = _load_json_snapshot(
        snapshot=snapshot,
        blob_repository=blob_repository,
        file_name=TOOL_SNAPSHOT_READ_SECTION_FILE_NAME,
    )
    search_load = _load_json_snapshot(
        snapshot=snapshot,
        blob_repository=blob_repository,
        file_name=TOOL_SNAPSHOT_SEARCH_FILE_NAME,
    )
    list_tables_load = _load_json_snapshot(
        snapshot=snapshot,
        blob_repository=blob_repository,
        file_name=TOOL_SNAPSHOT_LIST_TABLES_FILE_NAME,
    )
    get_table_load = _load_json_snapshot(
        snapshot=snapshot,
        blob_repository=blob_repository,
        file_name=TOOL_SNAPSHOT_GET_TABLE_FILE_NAME,
    )
    page_load = _load_json_snapshot(
        snapshot=snapshot,
        blob_repository=blob_repository,
        file_name=TOOL_SNAPSHOT_PAGE_FILE_NAME,
    )
    missing_snapshots = [
        file_name for file_name, load in (
            (TOOL_SNAPSHOT_SECTIONS_FILE_NAME, sections_load),
            (TOOL_SNAPSHOT_READ_SECTION_FILE_NAME, read_load),
            (TOOL_SNAPSHOT_SEARCH_FILE_NAME, search_load),
            (TOOL_SNAPSHOT_LIST_TABLES_FILE_NAME, list_tables_load),
            (TOOL_SNAPSHOT_GET_TABLE_FILE_NAME, get_table_load),
            (TOOL_SNAPSHOT_PAGE_FILE_NAME, page_load),
        )
        if not load.exists or not load.valid
    ]

    sections = _sections_from_calls(_calls_from_payload(sections_load.payload)) if sections_load.valid else []
    read_map = _read_map_from_calls(_calls_from_payload(read_load.payload)) if read_load.valid else {}
    tables = _tables_from_list_calls(_calls_from_payload(list_tables_load.payload)) if list_tables_load.valid else []
    get_table_map = _get_table_map_from_calls(_calls_from_payload(get_table_load.payload)) if get_table_load.valid else {}
    search_calls = _calls_from_payload(search_load.payload) if search_load.valid else []
    page_calls = _calls_from_payload(page_load.payload) if page_load.valid else []
    table_refs = _table_refs_from_list_tables(tables)
    placeholder_refs = _extract_table_placeholders(read_map)
    material_requires_financial_tables = _material_requires_financial_tables(
        snapshot=snapshot,
        meta=meta,
        sections=sections,
        read_map=read_map,
        tables=tables,
        get_table_map=get_table_map,
    )
    financial_groups = _effective_financial_groups(
        profile=profile,
        material_requires_financial_tables=material_requires_financial_tables,
    )

    dim_a = _score_structure(
        profile=profile,
        financial_groups=financial_groups,
        sections=sections,
        tables=tables,
        get_table_map=get_table_map,
    )
    dim_b = _score_text(read_map=read_map, table_refs=table_refs)
    dim_c = _score_search(
        profile=profile,
        meta=meta,
        search_calls=search_calls,
        search_exists=search_load.exists and search_load.valid,
    )
    dim_d = _score_table(
        profile=profile,
        financial_groups=financial_groups,
        tables=tables,
        get_table_map=get_table_map,
        placeholder_refs=placeholder_refs,
        tables_snapshot_valid=list_tables_load.exists and list_tables_load.valid,
    )
    dim_e = _score_traceability(
        snapshot=snapshot,
        meta=meta,
        sections=sections,
        read_map=read_map,
        tables=tables,
        get_table_map=get_table_map,
        placeholder_refs=placeholder_refs,
        missing_snapshots=missing_snapshots,
        search_call_count=len(search_calls),
    )
    dim_f = _score_page(
        profile=profile,
        sections=sections,
        tables=tables,
        get_table_map=get_table_map,
        page_calls=page_calls,
        page_exists=page_load.exists and page_load.valid,
    )
    dimensions = {
        DIMENSION_A_STRUCTURE: dim_a,
        DIMENSION_B_TEXT: dim_b,
        DIMENSION_C_SEARCH: dim_c,
        DIMENSION_D_TABLE: dim_d,
        DIMENSION_E_TRACEABILITY: dim_e,
        DIMENSION_F_PAGE: dim_f,
    }
    total_score = round(sum(dimension.points for dimension in dimensions.values()), 2)
    hard_gate = _evaluate_hard_gate(
        profile=profile,
        read_map=read_map,
        dim_a=dim_a,
        dim_b=dim_b,
        dim_d=dim_d,
        dim_e=dim_e,
        dim_f=dim_f,
        cfg=cfg,
    )
    if total_score >= cfg.min_doc_pass:
        grade = "pass"
    elif total_score >= cfg.min_doc_warn:
        grade = "warn"
    else:
        grade = "fail"
    return DocumentScore(
        ticker=snapshot.ticker,
        document_id=snapshot.document_id,
        source_kind=snapshot.source_kind.value,
        report_kind=snapshot.report_kind,
        market=_string_value(meta, "market").upper(),
        total_score=total_score,
        grade=grade,
        hard_gate=hard_gate,
        dimensions=dimensions,
    )


def _percentile_p10(values: list[float]) -> float:
    """计算 P10 分位值。

    Args:
        values: 分数列表。

    Returns:
        P10；空列表返回 0。

    Raises:
        无。
    """

    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = 0.1 * (len(sorted_values) - 1)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(sorted_values[lower])
    ratio = pos - lower
    return float(sorted_values[lower] * (1 - ratio) + sorted_values[upper] * ratio)


def score_batch(
    *,
    base: str,
    tickers: list[str],
    cfg: ScoreConfig,
    report_kind: str = REPORT_KIND_ALL,
    source_kind: str = SOURCE_KIND_ALL,
) -> BatchScore:
    """批量评分 CN/HK Docling snapshot。

    Args:
        base: workspace 根目录或 ``portfolio`` 目录。
        tickers: ticker 列表；为空时扫描公司元数据。
        cfg: 评分配置。
        report_kind: report kind 过滤。
        source_kind: source kind 过滤。

    Returns:
        批量评分结果。

    Raises:
        ValueError: 过滤参数非法时抛出。
        OSError: 仓储读取失败时可能抛出。
    """

    normalized_report_kind = report_kind.strip().lower()
    if normalized_report_kind not in REPORT_KIND_CHOICES:
        raise ValueError(f"不支持的 report_kind: {report_kind}")
    workspace_root = _resolve_workspace_root(base)
    discovery = _discover_docling_snapshots(
        base=base,
        tickers=tickers,
        report_kind_filter=normalized_report_kind,
        source_kind_filter=source_kind,
    )
    blob_repository = FsDocumentBlobRepository(workspace_root)
    docs: list[DocumentScore] = []
    completeness_failures = list(discovery.completeness_failures)
    for snapshot in discovery.snapshots:
        try:
            docs.append(score_document(snapshot, blob_repository, cfg))
        except ValueError as exc:
            completeness_failures.append(
                CompletenessFailure(
                    ticker=snapshot.ticker,
                    document_id=snapshot.document_id,
                    source_kind=snapshot.source_kind.value,
                    reason=str(exc),
                )
            )

    scores = [doc.total_score for doc in docs]
    average_score = round(float(sum(scores) / len(scores)) if scores else 0.0, 2)
    p10_score = round(_percentile_p10(scores), 2)
    hard_gate_failures = sum(1 for doc in docs if not doc.hard_gate.passed)
    failed_reasons: list[str] = []
    if completeness_failures:
        failed_reasons.append(f"completeness hard gate 失败文档数={len(completeness_failures)}")
    if hard_gate_failures:
        failed_reasons.append(f"硬门禁失败文档数={hard_gate_failures}")
    if average_score < cfg.min_batch_avg:
        failed_reasons.append(f"批量平均分 {average_score:.2f} < {cfg.min_batch_avg:.2f}")
    if p10_score < cfg.min_batch_p10:
        failed_reasons.append(f"批量 P10 分位 {p10_score:.2f} < {cfg.min_batch_p10:.2f}")
    if any(doc.total_score < cfg.min_doc_warn for doc in docs):
        failed_reasons.append("存在单文档分数低于 warn 阈值")
    return BatchScore(
        documents=docs,
        average_score=average_score,
        p10_score=p10_score,
        hard_gate_failures=hard_gate_failures,
        passed=not failed_reasons,
        failed_reasons=failed_reasons,
        completeness_failures=completeness_failures,
    )


def _serialize_dimension(dimension: DimensionScore) -> JsonObject:
    """序列化维度评分。

    Args:
        dimension: 维度评分。

    Returns:
        JSON 对象。

    Raises:
        无。
    """

    return {
        "points": dimension.points,
        "max_points": dimension.max_points,
        "details": dimension.details,
    }


def _serialize_document(doc: DocumentScore) -> JsonObject:
    """序列化单文档评分。

    Args:
        doc: 单文档评分。

    Returns:
        JSON 对象。

    Raises:
        无。
    """

    return {
        "ticker": doc.ticker,
        "document_id": doc.document_id,
        "source_kind": doc.source_kind,
        "report_kind": doc.report_kind,
        "market": doc.market,
        "total_score": doc.total_score,
        "grade": doc.grade,
        "hard_gate": {
            "passed": doc.hard_gate.passed,
            "reasons": doc.hard_gate.reasons,
        },
        "dimensions": {
            name: _serialize_dimension(dimension)
            for name, dimension in doc.dimensions.items()
        },
    }


def _serialize_failure(failure: CompletenessFailure) -> JsonObject:
    """序列化完整性失败记录。

    Args:
        failure: 完整性失败记录。

    Returns:
        JSON 对象。

    Raises:
        无。
    """

    return {
        "ticker": failure.ticker,
        "document_id": failure.document_id,
        "source_kind": failure.source_kind,
        "reason": failure.reason,
    }


def _config_to_json(cfg: ScoreConfig) -> JsonObject:
    """序列化评分配置。

    Args:
        cfg: 评分配置。

    Returns:
        JSON 对象。

    Raises:
        无。
    """

    return {
        "min_doc_pass": cfg.min_doc_pass,
        "min_doc_warn": cfg.min_doc_warn,
        "min_batch_avg": cfg.min_batch_avg,
        "min_batch_p10": cfg.min_batch_p10,
        "huge_section_fail": cfg.huge_section_fail,
        "max_total_score": MAX_TOTAL_SCORE,
    }


def write_json_report(
    *,
    path: str,
    batch: BatchScore,
    cfg: ScoreConfig,
    report_kind: str,
    source_kind: str,
) -> None:
    """写出 JSON 报告。

    Args:
        path: 输出路径。
        batch: 批量评分。
        cfg: 评分配置。
        report_kind: report kind 过滤。
        source_kind: source kind 过滤。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload: JsonObject = {
        "profile_id": REPORT_PROFILE_ID,
        "report_kind": report_kind,
        "source_kind": source_kind,
        "config": _config_to_json(cfg),
        "summary": {
            "average_score": batch.average_score,
            "p10_score": batch.p10_score,
            "hard_gate_failures": batch.hard_gate_failures,
            "completeness_failure_count": len(batch.completeness_failures),
            "expected_document_count": len(batch.documents) + len(batch.completeness_failures),
            "passed": batch.passed,
            "failed_reasons": batch.failed_reasons,
            "document_count": len(batch.documents),
        },
        "documents": [_serialize_document(doc) for doc in batch.documents],
        "completeness_failures": [_serialize_failure(failure) for failure in batch.completeness_failures],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _format_markdown_table(rows: list[list[str]]) -> str:
    """构造 Markdown 表格。

    Args:
        rows: 表格行。

    Returns:
        Markdown 表格。

    Raises:
        无。
    """

    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def _dimension_overview_rows(batch: BatchScore) -> list[list[str]]:
    """构建维度概览表。

    Args:
        batch: 批量评分。

    Returns:
        Markdown 行。

    Raises:
        无。
    """

    if not batch.documents:
        return []
    dim_names = list(batch.documents[0].dimensions.keys())
    rows = [["Ticker", "Document", "Kind", *dim_names, "Total"]]
    for doc in batch.documents:
        row = [doc.ticker, doc.document_id, doc.report_kind]
        for name in dim_names:
            dimension = doc.dimensions[name]
            row.append(f"{dimension.points:.1f}/{dimension.max_points:.0f}")
        row.append(f"{doc.total_score:.2f}")
        rows.append(row)
    return rows


def write_markdown_report(
    *,
    path: str,
    batch: BatchScore,
    cfg: ScoreConfig,
    report_kind: str,
    source_kind: str,
) -> None:
    """写出 Markdown 报告。

    Args:
        path: 输出路径。
        batch: 批量评分。
        cfg: 评分配置。
        report_kind: report kind 过滤。
        source_kind: source kind 过滤。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    rows = [["Ticker", "Document", "SourceKind", "ReportKind", "Market", "Score", "Grade", "HardGate"]]
    for doc in batch.documents:
        rows.append([
            doc.ticker,
            doc.document_id,
            doc.source_kind,
            doc.report_kind,
            doc.market,
            f"{doc.total_score:.2f}",
            doc.grade,
            "PASS" if doc.hard_gate.passed else "FAIL",
        ])

    lines = [
        "# CN/HK Docling CI 评分报告",
        "",
        "## 批量结果",
        "",
        f"- Profile: **{REPORT_PROFILE_ID}**",
        f"- Report kind: **{report_kind}**",
        f"- Source kind: **{source_kind}**",
        f"- 期望文档数: **{len(batch.documents) + len(batch.completeness_failures)}**",
        f"- 成功评分文档数: **{len(batch.documents)}**",
        f"- Completeness hard gate 失败数: **{len(batch.completeness_failures)}**",
        f"- 平均分: **{batch.average_score:.2f}**（阈值 {cfg.min_batch_avg:.2f}）",
        f"- P10 分位: **{batch.p10_score:.2f}**（阈值 {cfg.min_batch_p10:.2f}）",
        f"- 硬门禁失败数: **{batch.hard_gate_failures}**",
        f"- CI 判定: **{'PASS' if batch.passed else 'FAIL'}**",
        "",
        "## 文档明细",
        "",
        _format_markdown_table(rows),
        "",
    ]
    if batch.completeness_failures:
        lines.extend(["## Completeness Hard Gate", ""])
        for failure in batch.completeness_failures:
            lines.append(f"- {failure.ticker}/{failure.document_id} ({failure.source_kind}): {failure.reason}")
        lines.append("")
    failed_docs = [doc for doc in batch.documents if not doc.hard_gate.passed]
    if failed_docs:
        lines.extend(["## 硬门禁详情", ""])
        for doc in failed_docs:
            lines.append(f"- {doc.ticker}/{doc.document_id}: {', '.join(doc.hard_gate.reasons)}")
        lines.append("")
    overview_rows = _dimension_overview_rows(batch)
    if overview_rows:
        lines.extend(["## 维度概览", "", _format_markdown_table(overview_rows), ""])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _parse_tickers(raw: str) -> list[str]:
    """解析 CLI ticker 参数。

    Args:
        raw: 逗号分隔 ticker 字符串。

    Returns:
        ticker 列表。

    Raises:
        无。
    """

    return sorted({token.strip().upper() for token in raw.split(",") if token.strip()})


def _default_report_path(report_kind: str, source_kind: str, ext: str) -> str:
    """构造默认报告路径。

    Args:
        report_kind: report kind 过滤。
        source_kind: source kind 过滤。
        ext: 扩展名。

    Returns:
        默认报告路径。

    Raises:
        无。
    """

    slug = f"{source_kind}_{report_kind}".replace("_", "-")
    return f"workspace/reports/score_docling_{slug}_ci.{ext}"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 命令行参数列表；``None`` 使用 ``sys.argv``。

    Returns:
        argparse 结果。

    Raises:
        SystemExit: 参数非法时由 argparse 抛出。
    """

    parser = argparse.ArgumentParser(description="CN/HK Docling LLM 可喂性 CI 评分")
    parser.add_argument("--base", default="workspace", help="workspace 根目录或 portfolio 目录")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="逗号分隔 ticker；为空时扫描公司 meta")
    parser.add_argument("--report-kind", default=REPORT_KIND_ALL, choices=REPORT_KIND_CHOICES)
    parser.add_argument("--source-kind", default=SOURCE_KIND_ALL, choices=SOURCE_KIND_CHOICES)
    parser.add_argument("--output-json", default=None, help="JSON 报告输出路径")
    parser.add_argument("--output-md", default=None, help="Markdown 报告输出路径")
    parser.add_argument("--min-doc-pass", type=float, default=MIN_DOC_PASS)
    parser.add_argument("--min-doc-warn", type=float, default=MIN_DOC_WARN)
    parser.add_argument("--min-batch-avg", type=float, default=MIN_BATCH_AVG)
    parser.add_argument("--min-batch-p10", type=float, default=MIN_BATCH_P10)
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ScoreConfig:
    """从 CLI 参数构造评分配置。

    Args:
        args: argparse 结果。

    Returns:
        评分配置。

    Raises:
        无。
    """

    return ScoreConfig(
        min_doc_pass=float(args.min_doc_pass),
        min_doc_warn=float(args.min_doc_warn),
        min_batch_avg=float(args.min_batch_avg),
        min_batch_p10=float(args.min_batch_p10),
    )


def _print_console_summary(batch: BatchScore) -> None:
    """打印控制台摘要。

    Args:
        batch: 批量评分。

    Returns:
        无。

    Raises:
        OSError: stdout 写入失败时可能抛出。
    """

    print("=" * 80)
    print("CN/HK Docling CI 评分结果")
    print("=" * 80)
    for doc in batch.documents:
        gate = "PASS" if doc.hard_gate.passed else "FAIL"
        print(
            f"- {doc.ticker:8s} {doc.document_id}: "
            f"kind={doc.report_kind} source={doc.source_kind} "
            f"score={doc.total_score:6.2f}, grade={doc.grade}, gate={gate}"
        )
    print("-" * 80)
    print(
        f"average={batch.average_score:.2f}, "
        f"p10={batch.p10_score:.2f}, "
        f"hard_gate_failures={batch.hard_gate_failures}, "
        f"completeness_failures={len(batch.completeness_failures)}"
    )
    for failure in batch.completeness_failures:
        print(f"  ! completeness {failure.ticker}/{failure.document_id}: {failure.reason}")
    print(f"CI: {'PASS' if batch.passed else 'FAIL'}")
    for reason in batch.failed_reasons:
        print(f"  * {reason}")


def main(argv: Optional[list[str]] = None) -> int:
    """脚本入口。

    Args:
        argv: 命令行参数列表；``None`` 使用 ``sys.argv``。

    Returns:
        退出码；CI 通过返回 0，否则返回 1。

    Raises:
        OSError: 仓储读取或报告写入失败时可能抛出。
    """

    args = parse_args(argv)
    report_kind = str(args.report_kind).strip().lower()
    source_kind = str(args.source_kind).strip().lower()
    cfg = build_config(args)
    tickers = _parse_tickers(str(args.tickers))
    base = str(args.base).strip() or "workspace"
    output_json = args.output_json or _default_report_path(report_kind, source_kind, "json")
    output_md = args.output_md or _default_report_path(report_kind, source_kind, "md")
    batch = score_batch(
        base=base,
        tickers=tickers,
        cfg=cfg,
        report_kind=report_kind,
        source_kind=source_kind,
    )
    write_json_report(
        path=str(output_json),
        batch=batch,
        cfg=cfg,
        report_kind=report_kind,
        source_kind=source_kind,
    )
    write_markdown_report(
        path=str(output_md),
        batch=batch,
        cfg=cfg,
        report_kind=report_kind,
        source_kind=source_kind,
    )
    _print_console_summary(batch)
    return 0 if batch.passed else 1


if __name__ == "__main__":
    sys.exit(main())
