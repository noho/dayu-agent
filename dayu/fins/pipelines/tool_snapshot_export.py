"""财报工具快照导出模块。

该模块用于在 `process/process_filing/process_material` 阶段导出工具调用快照，
供后续离线回放（OfflineFinsToolService）与 CI 回归使用。

设计约束：
1. 快照响应必须统一通过 `FinsToolService` 生成，避免与在线工具链路漂移。
2. 默认离线模式不导出 `search_document` 与 `query_xbrl_facts`。
3. CI 模式在离线模式基础上追加 `search_document` 与 `query_xbrl_facts`。
4. 文件命名统一为 `tool_snapshot_*.json`。
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from io import BytesIO
from typing import Any, Callable, Mapping, Optional

from dayu.contracts.cancellation import CancelledError
from dayu.fins._converters import normalize_optional_text, require_non_empty_text
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.log import Log
from dayu.fins.domain.document_models import ProcessedHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.pipelines.processing_helpers import (
    register_processed_snapshot_document,
    resolve_processor_parser_version,
)
from dayu.fins.processors.form_type_utils import normalize_form_type
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    DocumentBlobRepositoryProtocol,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)
from dayu.fins.tools.result_types import DocumentSectionsResult, TablesListResult
from dayu.fins.tools.service import FinsToolService
from dayu.fins.tools.service_helpers import resolve_document_type_for_source

TOOL_SNAPSHOT_SCHEMA_VERSION = "tool_snapshot_v1.0.0"
TOOL_SNAPSHOT_FILE_PREFIX = "tool_snapshot_"
TOOL_SNAPSHOT_META_FILE_NAME = "tool_snapshot_meta.json"
TOOL_SNAPSHOT_OFFLINE_TOOLS = (
    "list_documents",
    "get_document_sections",
    "read_section",
    "list_tables",
    "get_table",
    "get_page_content",
    "get_financial_statement",
)
TOOL_SNAPSHOT_CI_EXTRA_TOOLS = (
    "search_document",
    "query_xbrl_facts",
)
FINANCIAL_STATEMENT_TYPES = (
    "income",
    "balance_sheet",
    "cash_flow",
    "equity",
    "comprehensive_income",
)
SC13_FORM_TYPES = frozenset({"SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A"})
EVENT_FORM_TYPES = frozenset({"8-K", "8-K/A", "6-K"})
GOVERNANCE_FORM_TYPES = frozenset({"DEF 14A"})

TOOL_SNAPSHOT_SEARCH_QUERY_PACK_VERSION = "search_query_pack_v1.0.0"
OFFLINE_SEARCH_QUERY_PACK_NAME = "offline_disabled"
DEFAULT_SEARCH_QUERY_WEIGHT = 1.0
SEARCH_QUERY_PACK_ANNUAL_QUARTER_CORE40 = "annual_quarter_core40"
SEARCH_QUERY_PACK_EVENT = "event_pack"
SEARCH_QUERY_PACK_GOVERNANCE = "governance_pack"
SEARCH_QUERY_PACK_OWNERSHIP = "ownership_pack"
SEARCH_QUERY_PACK_DEFAULT_MARKET = "CN"
SEARCH_QUERY_PACK_MARKETS = frozenset({"US", "CN", "HK"})

US_ANNUAL_QUARTER_CORE40_QUERIES = (
    "company history",
    "founded",
    "incorporated",
    "reorganization",
    "business overview",
    "segment",
    "product",
    "pricing",
    "customer concentration",
    "distribution channel",
    "competition",
    "market share",
    "moat",
    "strategy",
    "capital allocation",
    "dividend",
    "share repurchase",
    "debt",
    "liquidity",
    "cash flow",
    "capital expenditures",
    "revenue",
    "net income",
    "gross margin",
    "operating margin",
    "risk factors",
    "legal proceedings",
    "cybersecurity",
    "internal control",
    "accounting policies",
    "goodwill impairment",
    "management",
    "executive officers",
    "board of directors",
    "compensation",
    "related party transactions",
    "major shareholders",
    "subsidiaries",
    "acquisitions",
    "guidance",
)
CN_ANNUAL_QUARTER_CORE40_QUERIES = (
    "公司沿革",
    "成立时间",
    "注册地",
    "重组",
    "主营业务",
    "业务分部",
    "核心产品",
    "定价模式",
    "客户集中度",
    "渠道结构",
    "竞争格局",
    "市场份额",
    "护城河",
    "经营战略",
    "资本配置",
    "分红政策",
    "回购计划",
    "有息负债",
    "流动性",
    "现金流",
    "资本开支",
    "营业收入",
    "净利润",
    "毛利率",
    "营业利润率",
    "风险因素",
    "诉讼",
    "网络安全",
    "内部控制",
    "会计政策",
    "商誉减值",
    "管理层",
    "核心团队",
    "董事会",
    "高管薪酬",
    "关联交易",
    "主要股东",
    "子公司",
    "并购",
    "业绩指引",
)
HK_ANNUAL_QUARTER_CORE40_QUERIES = (
    "公司沿革",
    "成立時間",
    "註冊地",
    "重組",
    "主營業務",
    "業務分部",
    "核心產品",
    "定價模式",
    "客戶集中度",
    "渠道結構",
    "競爭格局",
    "市場份額",
    "護城河",
    "經營戰略",
    "資本配置",
    "分紅政策",
    "回購計劃",
    "有息負債",
    "流動性",
    "現金流",
    "资本开支",
    "營業收入",
    "淨利潤",
    "毛利率",
    "營業利潤率",
    "風險因素",
    "訴訟",
    "網絡安全",
    "內部控制",
    "會計政策",
    "商譽減值",
    "管理層",
    "核心團隊",
    "董事會",
    "高管薪酬",
    "關聯交易",
    "主要股東",
    "子公司",
    "併購",
    "回購",
)

US_EVENT_PACK_QUERIES = (
    "material agreement",
    "acquisition",
    "disposition",
    "financing",
    "earnings release",
    "guidance",
    "leadership change",
    "director appointment",
    "auditor change",
    "legal proceedings",
    "investigation",
    "cybersecurity incident",
    "bankruptcy",
    "share repurchase",
    "dividend",
    "impairment",
    "restructuring",
    "segment change",
    "internal control",
    "risk factors",
)
CN_EVENT_PACK_QUERIES = (
    "重大合同",
    "重大交易",
    "收购",
    "处置",
    "融资",
    "业绩预告",
    "业绩快报",
    "利润预警",
    "管理层变动",
    "审计师变更",
    "诉讼仲裁",
    "监管调查",
    "网络安全事件",
    "债务违约",
    "破产重整",
    "回购",
    "分红",
    "资产减值",
    "重组",
    "风险提示",
)
HK_EVENT_PACK_QUERIES = (
    "重大合同",
    "重大交易",
    "收購",
    "處置",
    "融資",
    "業績預告",
    "業績快報",
    "盈利預警",
    "管理層變動",
    "核數師變更",
    "訴訟仲裁",
    "監管調查",
    "網絡安全事件",
    "債務違約",
    "破產重整",
    "回購",
    "回購計劃",
    "分紅",
    "資產減值",
    "風險提示",
)

US_GOVERNANCE_PACK_QUERIES = (
    "board of directors",
    "board committee",
    "independent director",
    "audit committee",
    "compensation committee",
    "nomination committee",
    "executive compensation",
    "equity incentive plan",
    "related party transactions",
    "beneficial ownership",
    "voting power",
    "dual class",
    "shareholder proposal",
    "governance",
    "internal control",
    "auditor",
    "risk oversight",
    "director election",
    "say on pay",
    "succession planning",
)
CN_GOVERNANCE_PACK_QUERIES = (
    "董事会",
    "董事会委员会",
    "独立董事",
    "审计委员会",
    "薪酬委员会",
    "提名委员会",
    "管理层",
    "高管薪酬",
    "股权激励",
    "员工持股计划",
    "关联交易",
    "同业竞争",
    "控制权",
    "投票权",
    "主要股东",
    "公司治理",
    "内部控制",
    "审计师",
    "风险监督",
    "换届",
)
HK_GOVERNANCE_PACK_QUERIES = (
    "董事會",
    "董事會委員會",
    "獨立董事",
    "審計委員會",
    "薪酬委員會",
    "提名委員會",
    "管理層",
    "高管薪酬",
    "股權激勵",
    "員工持股計劃",
    "關聯交易",
    "同業競爭",
    "控制權",
    "投票權",
    "主要股東",
    "公司治理",
    "內部控制",
    "核數師",
    "風險監督",
    "換屆",
)

US_OWNERSHIP_PACK_QUERIES = (
    "beneficial ownership",
    "shares",
    "percent of class",
    "voting power",
    "sole voting power",
    "shared voting power",
    "sole dispositive power",
    "shared dispositive power",
    "source of funds",
    "purpose of transaction",
    "group",
    "cusip",
    "item 4",
    "item 5",
    "item 6",
    "item 7",
    "item 8",
    "item 9",
    "item 10",
    "security ownership",
)
CN_OWNERSHIP_PACK_QUERIES = (
    "受益所有权",
    "持股数量",
    "持股比例",
    "投票权",
    "单独投票权",
    "共同投票权",
    "处置权",
    "资金来源",
    "交易目的",
    "一致行动",
    "证券类别",
    "主要股东",
    "权益变动",
    "披露义务",
    "减持计划",
    "增持计划",
    "股权结构",
    "控制权",
    "股份来源",
    "信息披露",
)
HK_OWNERSHIP_PACK_QUERIES = (
    "受益所有權",
    "持股數量",
    "持股比例",
    "投票權",
    "單獨投票權",
    "共同投票權",
    "處置權",
    "資金來源",
    "交易目的",
    "一致行動",
    "證券類別",
    "主要股東",
    "權益變動",
    "披露義務",
    "減持計劃",
    "增持計劃",
    "股權結構",
    "控制權",
    "股份來源",
    "信息披露",
)

SEARCH_QUERY_PACKS: dict[str, dict[str, tuple[str, ...]]] = {
    SEARCH_QUERY_PACK_ANNUAL_QUARTER_CORE40: {
        "US": US_ANNUAL_QUARTER_CORE40_QUERIES,
        "CN": CN_ANNUAL_QUARTER_CORE40_QUERIES,
        "HK": HK_ANNUAL_QUARTER_CORE40_QUERIES,
    },
    SEARCH_QUERY_PACK_EVENT: {
        "US": US_EVENT_PACK_QUERIES,
        "CN": CN_EVENT_PACK_QUERIES,
        "HK": HK_EVENT_PACK_QUERIES,
    },
    SEARCH_QUERY_PACK_GOVERNANCE: {
        "US": US_GOVERNANCE_PACK_QUERIES,
        "CN": CN_GOVERNANCE_PACK_QUERIES,
        "HK": HK_GOVERNANCE_PACK_QUERIES,
    },
    SEARCH_QUERY_PACK_OWNERSHIP: {
        "US": US_OWNERSHIP_PACK_QUERIES,
        "CN": CN_OWNERSHIP_PACK_QUERIES,
        "HK": HK_OWNERSHIP_PACK_QUERIES,
    },
}

MODULE = "FINS.TOOL_SNAPSHOT"
XBRL_DEFAULT_CONCEPTS = ["Revenues", "NetIncomeLoss", "Assets"]


def _raise_if_cancelled(*, ticker: str, document_id: str, stage: str, cancel_checker: Callable[[], bool] | None) -> None:
    """在工具快照导出阶段边界检查取消请求。

    Args:
        ticker: 股票代码。
        document_id: 文档 ID。
        stage: 当前执行阶段。
        cancel_checker: 可选取消检查函数。

    Returns:
        无。

    Raises:
        CancelledError: 当前执行已被取消时抛出。
    """

    if cancel_checker is None or not cancel_checker():
        return
    Log.info(
        "工具快照导出收到取消请求，停止执行: "
        f"ticker={ticker} document_id={document_id} stage={stage}",
        module=MODULE,
    )
    raise CancelledError("操作已被取消")


def build_snapshot_tool_names(*, ci: bool) -> tuple[str, ...]:
    """构建当前模式需要导出的工具列表。

    Args:
        ci: 是否为 CI 导出模式。

    Returns:
        工具名称元组。

    Raises:
        无。
    """

    if not ci:
        return TOOL_SNAPSHOT_OFFLINE_TOOLS
    return TOOL_SNAPSHOT_OFFLINE_TOOLS + TOOL_SNAPSHOT_CI_EXTRA_TOOLS


def build_snapshot_file_names(*, ci: bool) -> list[str]:
    """构建当前模式应存在的快照文件名列表（含 meta）。

    Args:
        ci: 是否为 CI 导出模式。

    Returns:
        文件名列表。

    Raises:
        无。
    """

    tool_names = build_snapshot_tool_names(ci=ci)
    result = [f"{TOOL_SNAPSHOT_FILE_PREFIX}{tool_name}.json" for tool_name in tool_names]
    result.append(TOOL_SNAPSHOT_META_FILE_NAME)
    return result


def is_snapshot_file_name(file_name: str) -> bool:
    """判断文件名是否属于快照文件。

    Args:
        file_name: 文件名。

    Returns:
        是否为快照文件。

    Raises:
        无。
    """

    normalized_file_name = str(file_name or "").strip()
    if not normalized_file_name:
        return False
    return normalized_file_name.startswith(TOOL_SNAPSHOT_FILE_PREFIX)


def export_tool_snapshot(
    *,
    company_repository: CompanyMetaRepositoryProtocol,
    source_repository: SourceDocumentRepositoryProtocol,
    processed_repository: ProcessedDocumentRepositoryProtocol,
    blob_repository: DocumentBlobRepositoryProtocol,
    processor_registry: ProcessorRegistry,
    processed_handle: ProcessedHandle,
    ticker: str,
    document_id: str,
    source_kind: SourceKind,
    source_meta: dict[str, Any],
    ci: bool,
    expected_parser_signature: str,
    market_override: Optional[str] = None,
    processor_cache_max_entries: int = 128,
    cancel_checker: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """导出单文档工具快照文件。

    Args:
        company_repository: 公司元数据仓储实现。
        source_repository: 源文档仓储实现。
        processed_repository: processed 文档仓储实现。
        blob_repository: 文档文件对象仓储实现。
        processor_registry: 处理器注册表。
        processed_handle: `processed/{document_id}` 对应仓储句柄。
        ticker: 股票代码。
        document_id: 文档 ID。
        source_kind: 源文档类型。
        source_meta: 源文档元数据。
        ci: 是否开启 CI 导出（追加 `search_document/query_xbrl_facts`）。
        expected_parser_signature: 当前文档预期解析器签名（用于增量判定）。
        market_override: 可选市场覆盖值（如 `US/CN/HK`）。
        processor_cache_max_entries: Processor LRU 缓存容量。
        cancel_checker: 可选取消检查函数，用于导出阶段边界取消。

    Returns:
        导出摘要，包含写入文件列表。

    Raises:
        OSError: 文件写入失败时抛出。
        RuntimeError: 工具调用失败时抛出。
        ValueError: 入参不合法时抛出。
    """

    normalized_ticker = require_non_empty_text(ticker, empty_error=ValueError("必填文本不能为空"))
    normalized_document_id = require_non_empty_text(document_id, empty_error=ValueError("必填文本不能为空"))
    normalized_expected_parser_signature = require_non_empty_text(
        expected_parser_signature,
        empty_error=ValueError("必填文本不能为空"),
    )
    _raise_if_cancelled(
        ticker=normalized_ticker,
        document_id=normalized_document_id,
        stage="before_prepare",
        cancel_checker=cancel_checker,
    )
    Log.debug(
        f"开始导出工具快照: ticker={normalized_ticker} document_id={document_id} mode={'ci' if ci else 'offline'}",
        module=MODULE,
    )
    _validate_processed_handle(
        processed_handle=processed_handle,
        ticker=normalized_ticker,
        document_id=normalized_document_id,
    )

    _, market = _read_company_info(repository=company_repository, ticker=normalized_ticker)
    resolved_market = normalize_optional_text(market_override) or market
    resolved_form_type = _normalize_form_type(source_meta.get("form_type", ""))
    resolved_document_type = resolve_document_type_for_source(
        form_type=source_meta.get("form_type", ""),
        source_kind=source_kind.value,
    )
    search_query_pack = _build_search_query_pack(
        market=resolved_market,
        form_type=resolved_form_type,
    )
    search_query_pack_name = str(search_query_pack["name"])
    search_query_pack_version = str(search_query_pack["version"])
    search_queries = list(search_query_pack["queries"])
    search_query_specs = _build_search_query_specs(
        pack_name=search_query_pack_name,
        queries=search_queries,
    )
    xbrl_concepts = _build_xbrl_concepts()
    tool_names = build_snapshot_tool_names(ci=ci)
    mode_name = "ci" if ci else "offline"
    if ci:
        Log.debug(
            "CI 搜索词包已加载: "
            f"ticker={normalized_ticker} document_id={normalized_document_id} "
            f"pack={search_query_pack_name} version={search_query_pack_version} "
            f"query_count={len(search_queries)}",
            module=MODULE,
        )

    service = FinsToolService(
        company_repository=company_repository,
        source_repository=source_repository,
        processed_repository=processed_repository,
        processor_registry=processor_registry,
        processor_cache_max_entries=processor_cache_max_entries,
    )
    actual_processor = service._get_or_create_processor(  # noqa: SLF001
        ticker=normalized_ticker,
        document_id=normalized_document_id,
    )
    actual_parser_signature = resolve_processor_parser_version(actual_processor)
    _raise_if_cancelled(
        ticker=normalized_ticker,
        document_id=normalized_document_id,
        stage="before_prefetch",
        cancel_checker=cancel_checker,
    )
    list_documents_response = service.list_documents(ticker=normalized_ticker)
    sections_response = service.get_document_sections(
        ticker=normalized_ticker,
        document_id=normalized_document_id,
    )
    list_tables_response = service.list_tables(
        ticker=normalized_ticker,
        document_id=normalized_document_id,
        financial_only=False,
        within_section_ref=None,
    )
    _raise_if_cancelled(
        ticker=normalized_ticker,
        document_id=normalized_document_id,
        stage="before_build_payloads",
        cancel_checker=cancel_checker,
    )

    tool_payloads: dict[str, dict[str, Any]] = {
        "list_documents": _build_tool_snapshot_payload(
            tool="list_documents",
            mode=mode_name,
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            source_kind=source_kind,
            market=resolved_market,
            calls=[
                {
                    "request": {"ticker": normalized_ticker},
                    "response": list_documents_response,
                }
            ],
        ),
        "get_document_sections": _build_tool_snapshot_payload(
            tool="get_document_sections",
            mode=mode_name,
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            source_kind=source_kind,
            market=resolved_market,
            calls=[
                {
                    "request": {
                        "ticker": normalized_ticker,
                        "document_id": normalized_document_id,
                    },
                    "response": sections_response,
                }
            ],
        ),
        "read_section": _build_tool_snapshot_payload(
            tool="read_section",
            mode=mode_name,
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            source_kind=source_kind,
            market=resolved_market,
            calls=_build_read_section_calls(
                service=service,
                ticker=normalized_ticker,
                document_id=normalized_document_id,
                sections_response=sections_response,
            ),
        ),
        "list_tables": _build_tool_snapshot_payload(
            tool="list_tables",
            mode=mode_name,
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            source_kind=source_kind,
            market=resolved_market,
            calls=[
                {
                    "request": {
                        "ticker": normalized_ticker,
                        "document_id": normalized_document_id,
                        "financial_only": False,
                        "within_section_ref": None,
                    },
                    "response": list_tables_response,
                }
            ],
        ),
        "get_table": _build_tool_snapshot_payload(
            tool="get_table",
            mode=mode_name,
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            source_kind=source_kind,
            market=resolved_market,
            calls=_build_get_table_calls(
                service=service,
                ticker=normalized_ticker,
                document_id=normalized_document_id,
                list_tables_response=list_tables_response,
            ),
        ),
        "get_page_content": _build_tool_snapshot_payload(
            tool="get_page_content",
            mode=mode_name,
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            source_kind=source_kind,
            market=resolved_market,
            calls=_build_get_page_content_calls(
                service=service,
                ticker=normalized_ticker,
                document_id=normalized_document_id,
                sections_response=sections_response,
                list_tables_response=list_tables_response,
            ),
        ),
        "get_financial_statement": _build_tool_snapshot_payload(
            tool="get_financial_statement",
            mode=mode_name,
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            source_kind=source_kind,
            market=resolved_market,
            calls=_build_get_financial_statement_calls(
                service=service,
                ticker=normalized_ticker,
                document_id=normalized_document_id,
            ),
        ),
    }
    if ci:
        tool_payloads["search_document"] = _build_tool_snapshot_payload(
            tool="search_document",
            mode=mode_name,
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            source_kind=source_kind,
            market=resolved_market,
            calls=_build_search_document_calls(
                service=service,
                ticker=normalized_ticker,
                document_id=normalized_document_id,
                search_query_specs=search_query_specs,
            ),
        )
        tool_payloads["query_xbrl_facts"] = _build_tool_snapshot_payload(
            tool="query_xbrl_facts",
            mode=mode_name,
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            source_kind=source_kind,
            market=resolved_market,
            calls=[
                _build_query_xbrl_facts_call(
                    service=service,
                    ticker=normalized_ticker,
                    document_id=normalized_document_id,
                    source_meta=source_meta,
                    concepts=xbrl_concepts,
                )
            ],
        )

    financial_calls = tool_payloads["get_financial_statement"]["calls"]
    (
        has_financial_statement,
        has_xbrl,
        financial_statement_availability,
        has_structured_financial_statements,
        has_financial_statement_sections,
        has_financial_data,
    ) = _analyze_financial_statement_capability(
        financial_statement_calls=financial_calls
    )
    written_files: list[str] = []
    for tool_name in tool_names:
        _raise_if_cancelled(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            stage=f"before_write_{tool_name}",
            cancel_checker=cancel_checker,
        )
        payload = tool_payloads[tool_name]
        file_name = f"{TOOL_SNAPSHOT_FILE_PREFIX}{tool_name}.json"
        _write_tool_snapshot_file(
            repository=blob_repository,
            processed_handle=processed_handle,
            file_name=file_name,
            payload=payload,
        )
        written_files.append(file_name)

    _raise_if_cancelled(
        ticker=normalized_ticker,
        document_id=normalized_document_id,
        stage="before_write_meta",
        cancel_checker=cancel_checker,
    )
    meta_search_query_pack_name = search_query_pack_name if ci else OFFLINE_SEARCH_QUERY_PACK_NAME
    meta_search_query_pack_version = search_query_pack_version if ci else TOOL_SNAPSHOT_SEARCH_QUERY_PACK_VERSION
    meta_search_queries = search_queries if ci else []
    meta_search_query_count = len(meta_search_queries)
    meta_payload = _build_tool_snapshot_meta(
        ticker=normalized_ticker,
        document_id=normalized_document_id,
        source_kind=source_kind,
        market=resolved_market,
        mode=mode_name,
        parser_signature=actual_parser_signature,
        expected_parser_signature=normalized_expected_parser_signature,
        source_document_version=str(source_meta.get("document_version", "")),
        source_fingerprint=str(source_meta.get("source_fingerprint", "")),
        form_type=resolved_form_type,
        document_type=resolved_document_type,
        has_financial_statement=has_financial_statement,
        has_xbrl=has_xbrl,
        financial_statement_availability=financial_statement_availability,
        has_structured_financial_statements=has_structured_financial_statements,
        has_financial_statement_sections=has_financial_statement_sections,
        has_financial_data=has_financial_data,
        search_queries=meta_search_queries,
        search_query_pack_name=meta_search_query_pack_name,
        search_query_pack_version=meta_search_query_pack_version,
        search_query_count=meta_search_query_count,
        xbrl_concepts=xbrl_concepts if ci else [],
        statement_types=list(FINANCIAL_STATEMENT_TYPES),
        tools=list(tool_names),
        written_files=written_files,
    )
    _write_tool_snapshot_file(
        repository=blob_repository,
        processed_handle=processed_handle,
        file_name=TOOL_SNAPSHOT_META_FILE_NAME,
        payload=meta_payload,
    )
    written_files.append(TOOL_SNAPSHOT_META_FILE_NAME)
    register_processed_snapshot_document(
        repository=processed_repository,
        ticker=normalized_ticker,
        document_id=normalized_document_id,
        source_kind=source_kind,
        source_meta=source_meta,
        parser_signature=actual_parser_signature,
        has_xbrl=has_xbrl,
    )
    result = {
        "ticker": normalized_ticker,
        "document_id": normalized_document_id,
        "mode": mode_name,
        "written_files": written_files,
    }
    Log.info(
        f"工具快照导出完成: ticker={normalized_ticker} document_id={normalized_document_id} mode={mode_name} files={len(written_files)}",
        module=MODULE,
    )
    return result


def _build_tool_snapshot_payload(
    *,
    tool: str,
    mode: str,
    ticker: str,
    document_id: str,
    source_kind: SourceKind,
    market: str,
    calls: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建单工具快照载荷。

    Args:
        tool: 工具名称。
        mode: 导出模式（`offline/ci`）。
        ticker: 股票代码。
        document_id: 文档 ID。
        source_kind: 源文档类型。
        market: 市场编码。
        calls: 调用列表。

    Returns:
        单工具快照字典。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    return {
        "snapshot_type": "tool_snapshot",
        "schema_version": TOOL_SNAPSHOT_SCHEMA_VERSION,
        "mode": mode,
        "tool": tool,
        "ticker": ticker,
        "document_id": document_id,
        "source_kind": source_kind.value,
        "market": market,
        "generated_at": datetime.now(UTC).isoformat(),
        "calls": calls,
    }


def _build_read_section_calls(
    *,
    service: FinsToolService,
    ticker: str,
    document_id: str,
    sections_response: DocumentSectionsResult,
) -> list[dict[str, Any]]:
    """构建 `read_section` 调用集。"""

    section_refs = _collect_section_refs(sections_response=sections_response)
    calls: list[dict[str, Any]] = []
    for ref in section_refs:
        request = {
            "ticker": ticker,
            "document_id": document_id,
            "ref": ref,
        }
        calls.append({"request": request, "response": service.read_section(**request)})
    return calls


def _build_search_document_calls(
    *,
    service: FinsToolService,
    ticker: str,
    document_id: str,
    search_query_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """构建 `search_document` 调用集。

    Args:
        service: 工具服务实例。
        ticker: 股票代码。
        document_id: 文档 ID。
        search_query_specs: 搜索查询规格列表。

    Returns:
        `search_document` 调用快照列表。

    Raises:
        ValueError: 查询规格缺少必填字段时抛出。
    """

    calls: list[dict[str, Any]] = []
    for query_spec in search_query_specs:
        query_id = require_non_empty_text(query_spec.get("query_id"), empty_error=ValueError("必填文本不能为空"))
        query_text = require_non_empty_text(query_spec.get("query_text"), empty_error=ValueError("必填文本不能为空"))
        query_intent = require_non_empty_text(
            query_spec.get("query_intent"),
            empty_error=ValueError("必填文本不能为空"),
        )
        query_weight = float(query_spec.get("query_weight", DEFAULT_SEARCH_QUERY_WEIGHT))
        if query_weight <= 0:
            raise ValueError("search_query_specs.query_weight 必须大于 0")
        request = {
            "ticker": ticker,
            "document_id": document_id,
            "query": query_text,
            "query_id": query_id,
            "query_text": query_text,
            "query_intent": query_intent,
            "query_weight": query_weight,
            "within_section_ref": None,
        }
        response = service.search_document(
            ticker=ticker,
            document_id=document_id,
            query=query_text,
            within_section_ref=None,
        )
        calls.append({"request": request, "response": response})
    return calls


def _build_get_table_calls(
    *,
    service: FinsToolService,
    ticker: str,
    document_id: str,
    list_tables_response: TablesListResult,
) -> list[dict[str, Any]]:
    """构建 `get_table` 调用集。"""

    table_refs = _collect_table_refs(list_tables_response=list_tables_response)
    calls: list[dict[str, Any]] = []
    for table_ref in table_refs:
        request = {
            "ticker": ticker,
            "document_id": document_id,
            "table_ref": table_ref,
        }
        calls.append({"request": request, "response": service.get_table(**request)})
    return calls


def _build_get_page_content_calls(
    *,
    service: FinsToolService,
    ticker: str,
    document_id: str,
    sections_response: DocumentSectionsResult,
    list_tables_response: TablesListResult,
) -> list[dict[str, Any]]:
    """构建 `get_page_content` 调用集。"""

    page_nos = _collect_page_candidates(
        sections_response=sections_response,
        list_tables_response=list_tables_response,
    )
    calls: list[dict[str, Any]] = []
    for page_no in page_nos:
        request = {
            "ticker": ticker,
            "document_id": document_id,
            "page_no": page_no,
        }
        calls.append({"request": request, "response": service.get_page_content(**request)})
    return calls


def _build_get_financial_statement_calls(
    *,
    service: FinsToolService,
    ticker: str,
    document_id: str,
) -> list[dict[str, Any]]:
    """构建 `get_financial_statement` 调用集。"""

    calls: list[dict[str, Any]] = []
    for statement_type in FINANCIAL_STATEMENT_TYPES:
        request = {
            "ticker": ticker,
            "document_id": document_id,
            "statement_type": statement_type,
        }
        calls.append({"request": request, "response": service.get_financial_statement(**request)})
    return calls


def _build_query_xbrl_facts_call(
    *,
    service: FinsToolService,
    ticker: str,
    document_id: str,
    source_meta: dict[str, Any],
    concepts: list[str],
) -> dict[str, Any]:
    """构建 `query_xbrl_facts` 单次调用。"""

    request: dict[str, Any] = {
        "ticker": ticker,
        "document_id": document_id,
        "concepts": concepts,
    }
    fiscal_year = source_meta.get("fiscal_year")
    if isinstance(fiscal_year, int):
        request["fiscal_year"] = fiscal_year
    fiscal_period = normalize_optional_text(source_meta.get("fiscal_period"))
    if fiscal_period is not None:
        request["fiscal_period"] = fiscal_period
    period_end = normalize_optional_text(source_meta.get("report_date"))
    if period_end is not None:
        request["period_end"] = period_end
    return {"request": request, "response": service.query_xbrl_facts(**request)}


def _read_company_info(*, repository: CompanyMetaRepositoryProtocol, ticker: str) -> tuple[str, str]:
    """读取公司基础信息。"""

    try:
        company_meta = repository.get_company_meta(ticker)
    except FileNotFoundError:
        return ticker, "unknown"
    return company_meta.company_name, company_meta.market


def _build_search_queries(market: str, form_type: str = "") -> list[str]:
    """按市场和表单类型构建搜索词包（仅返回查询词列表）。

    Args:
        market: 市场编码。
        form_type: 文档表单类型。

    Returns:
        查询词列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    pack = _build_search_query_pack(market=market, form_type=form_type)
    return list(pack["queries"])


def _build_search_query_pack(market: str, form_type: str = "") -> dict[str, Any]:
    """按市场和表单类型构建搜索词包。

    Args:
        market: 市场编码（`US/CN/HK`）。
        form_type: 文档表单类型。

    Returns:
        词包信息字典，包含 `name/version/queries`。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    normalized_form = _normalize_form_type(form_type)
    if normalized_form in SC13_FORM_TYPES:
        pack_name = SEARCH_QUERY_PACK_OWNERSHIP
    elif normalized_form in GOVERNANCE_FORM_TYPES:
        pack_name = SEARCH_QUERY_PACK_GOVERNANCE
    elif normalized_form in EVENT_FORM_TYPES:
        pack_name = SEARCH_QUERY_PACK_EVENT
    else:
        pack_name = SEARCH_QUERY_PACK_ANNUAL_QUARTER_CORE40

    normalized_market = _normalize_market(market)
    queries_by_market = SEARCH_QUERY_PACKS[pack_name]
    queries = list(queries_by_market[normalized_market])
    return {
        "name": pack_name,
        "version": TOOL_SNAPSHOT_SEARCH_QUERY_PACK_VERSION,
        "queries": queries,
    }


def _build_search_query_specs(
    *,
    pack_name: str,
    queries: list[str],
) -> list[dict[str, Any]]:
    """构建结构化查询规格列表。

    Args:
        pack_name: 搜索词包名称。
        queries: 查询文本列表。

    Returns:
        结构化查询规格列表，包含 `query_id/query_text/query_intent/query_weight`。

    Raises:
        ValueError: 查询文本无效时抛出。
    """

    normalized_pack_name = require_non_empty_text(pack_name, empty_error=ValueError("必填文本不能为空"))
    specs: list[dict[str, Any]] = []
    for index, raw_query in enumerate(queries, start=1):
        query_text = require_non_empty_text(raw_query, empty_error=ValueError("必填文本不能为空"))
        query_id = f"{normalized_pack_name}.q{index:03d}"
        query_intent = _build_query_intent(query_text=query_text, index=index)
        specs.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "query_intent": query_intent,
                "query_weight": DEFAULT_SEARCH_QUERY_WEIGHT,
            }
        )
    return specs


def _build_query_intent(*, query_text: str, index: int) -> str:
    """构建查询意图标签。

    Args:
        query_text: 查询文本。
        index: 查询序号（从 1 开始）。

    Returns:
        语义稳定的查询意图标签。

    Raises:
        ValueError: 参数非法时抛出。
    """

    _ = require_non_empty_text(query_text, empty_error=ValueError("必填文本不能为空"))
    if index <= 0:
        raise ValueError("index 必须大于 0")
    normalized = re.sub(r"[^a-z0-9]+", "_", query_text.lower()).strip("_")
    if normalized:
        return normalized[:64]
    return f"query_{index:03d}"


def _normalize_market(value: Any) -> str:
    """标准化市场编码。

    Args:
        value: 原始市场值。

    Returns:
        标准化市场编码，未知值回退 `CN`。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized = normalize_optional_text(value)
    if normalized is None:
        return SEARCH_QUERY_PACK_DEFAULT_MARKET
    upper_value = normalized.upper()
    if upper_value not in SEARCH_QUERY_PACK_MARKETS:
        return SEARCH_QUERY_PACK_DEFAULT_MARKET
    return upper_value


def _normalize_form_type(value: Any) -> str:
    """标准化表单类型。

    Args:
        value: 原始表单类型。

    Returns:
        标准化后表单类型；空值返回空字符串。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized = normalize_optional_text(value)
    if normalized is None:
        return ""
    normalized_form = normalize_form_type(normalized)
    return normalize_optional_text(normalized_form) or ""


def _build_xbrl_concepts() -> list[str]:
    """构建固定 XBRL 概念包。"""

    return list(XBRL_DEFAULT_CONCEPTS)


def _collect_page_candidates(
    *,
    sections_response: Mapping[str, Any],
    list_tables_response: Mapping[str, Any],
) -> list[int]:
    """采集 `get_page_content` 候选页码。"""

    candidates: set[int] = set()
    tables = list_tables_response.get("tables")
    if isinstance(tables, list):
        for table in tables:
            if not isinstance(table, Mapping):
                continue
            page_no = table.get("page_no")
            if isinstance(page_no, int) and page_no > 0:
                candidates.add(page_no)

    sections = sections_response.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, Mapping):
                continue
            page_range = _extract_page_range(section)
            if page_range is None:
                continue
            start, end = page_range
            if start > end:
                continue
            for page_no in range(start, end + 1):
                candidates.add(page_no)

    if not candidates:
        candidates.add(1)
    return sorted(candidates)


def _collect_section_refs(*, sections_response: Mapping[str, Any]) -> list[str]:
    """提取章节 ref 列表。"""

    section_refs: set[str] = set()
    sections = sections_response.get("sections")
    if not isinstance(sections, list):
        return []
    for item in sections:
        if not isinstance(item, Mapping):
            continue
        raw_ref = normalize_optional_text(item.get("ref"))
        if raw_ref is not None:
            section_refs.add(raw_ref)
    return sorted(section_refs)


def _collect_table_refs(*, list_tables_response: Mapping[str, Any]) -> list[str]:
    """提取表格 ref 列表。"""

    table_refs: set[str] = set()
    tables = list_tables_response.get("tables")
    if not isinstance(tables, list):
        return []
    for item in tables:
        if not isinstance(item, Mapping):
            continue
        table_ref = normalize_optional_text(item.get("table_ref"))
        if table_ref is not None:
            table_refs.add(table_ref)
    return sorted(table_refs)


def _build_tool_snapshot_meta(
    *,
    ticker: str,
    document_id: str,
    source_kind: SourceKind,
    market: str,
    mode: str,
    parser_signature: str,
    expected_parser_signature: str,
    source_document_version: str,
    source_fingerprint: str,
    form_type: Optional[str],
    document_type: Optional[str],
    has_financial_statement: bool,
    has_xbrl: bool,
    financial_statement_availability: Optional[str],
    has_structured_financial_statements: bool,
    has_financial_statement_sections: bool,
    has_financial_data: bool,
    search_queries: list[str],
    search_query_pack_name: str,
    search_query_pack_version: str,
    search_query_count: int,
    xbrl_concepts: list[str],
    statement_types: list[str],
    tools: list[str],
    written_files: list[str],
) -> dict[str, Any]:
    """构建 `tool_snapshot_meta.json` 载荷。

    Args:
        ticker: 股票代码。
        document_id: 文档 ID。
        source_kind: 文档来源类型。
        market: 市场编码。
        mode: 导出模式。
        parser_signature: 实际成功实例化的解析器签名。
        expected_parser_signature: 预期主路由解析器签名。
        source_document_version: 源文档版本。
        source_fingerprint: 源文档指纹。
        form_type: 表单类型。
        document_type: 面向 LLM 的语义文档类型。
        has_financial_statement: 是否具备财报能力。
        has_xbrl: 是否具备 XBRL 能力。
        financial_statement_availability: 面向 LLM 的财报可用性语义。
        has_structured_financial_statements: 是否具备结构化财报能力。
        has_financial_statement_sections: 是否具备章节级财报能力。
        has_financial_data: 面向 LLM 的财务数据可用性布尔。
        search_queries: 搜索词列表。
        search_query_pack_name: 搜索词包名称（必填）。
        search_query_pack_version: 搜索词包版本（必填）。
        search_query_count: 搜索词数量。
        xbrl_concepts: XBRL 概念列表。
        statement_types: 财报类型列表。
        tools: 导出的工具名列表。
        written_files: 已写入文件列表。

    Returns:
        `tool_snapshot_meta.json` 字典。

    Raises:
        ValueError: 搜索词包元信息不满足契约时抛出。
    """

    normalized_pack_name = require_non_empty_text(
        search_query_pack_name,
        empty_error=ValueError("必填文本不能为空"),
    )
    normalized_pack_version = require_non_empty_text(
        search_query_pack_version,
        empty_error=ValueError("必填文本不能为空"),
    )
    if search_query_count < 0:
        raise ValueError("search_query_count 不能为负数")
    if search_query_count != len(search_queries):
        raise ValueError("search_query_count 与 search_queries 实际数量不一致")

    return {
        "snapshot_type": "snapshot_meta",
        "schema_version": TOOL_SNAPSHOT_SCHEMA_VERSION,
        "mode": mode,
        "ticker": ticker,
        "document_id": document_id,
        "source_kind": source_kind.value,
        "market": market,
        "parser_signature": parser_signature,
        "expected_parser_signature": expected_parser_signature,
        "snapshot_schema_version": TOOL_SNAPSHOT_SCHEMA_VERSION,
        "source_document_version": source_document_version,
        "source_fingerprint": source_fingerprint,
        "form_type": form_type,
        "document_type": document_type,
        "has_financial_statement": has_financial_statement,
        "has_xbrl": has_xbrl,
        "financial_statement_availability": financial_statement_availability,
        "has_structured_financial_statements": has_structured_financial_statements,
        "has_financial_statement_sections": has_financial_statement_sections,
        "has_financial_data": has_financial_data,
        "search_queries": search_queries,
        "search_query_pack_name": normalized_pack_name,
        "search_query_pack_version": normalized_pack_version,
        "search_query_count": search_query_count,
        "xbrl_concepts": xbrl_concepts,
        "statement_types": statement_types,
        "tools": tools,
        "written_files": written_files,
        "generated_at": datetime.now(UTC).isoformat(),
    }


def _analyze_financial_statement_capability(
    *,
    financial_statement_calls: list[dict[str, Any]],
) -> tuple[bool, bool, Optional[str], bool, bool, bool]:
    """分析财报快照中财务报表能力与 XBRL 可用性。

    判定规则：
    1. `has_financial_statement`：任一调用成功返回“有效报表数据”即为 `True`；
       有效数据指 `rows` 或 `periods` 至少一项非空。
    2. `has_xbrl`：任一成功调用的 `data_quality == "xbrl"` 即为 `True`。
    3. `has_financial_data`：面向 LLM 的单一布尔，等价于 has_financial_statement。

    Args:
        financial_statement_calls: `get_financial_statement` 的调用快照列表。

    Returns:
        `(has_financial_statement, has_xbrl, financial_statement_availability,
        has_structured_financial_statements, has_financial_statement_sections,
        has_financial_data)`。

    Raises:
        无。
    """

    has_financial_statement = False
    has_xbrl = False
    has_structured_financial_statements = False
    has_financial_statement_sections = False
    for call in financial_statement_calls:
        if not isinstance(call, Mapping):
            continue
        response = call.get("response")
        if not isinstance(response, Mapping):
            continue
        rows = response.get("rows")
        periods = response.get("periods")
        has_rows = isinstance(rows, list) and len(rows) > 0
        has_periods = isinstance(periods, list) and len(periods) > 0
        if has_rows or has_periods:
            has_financial_statement = True
            has_financial_statement_sections = True
        data_quality = normalize_optional_text(response.get("data_quality"))
        normalized_quality = data_quality.lower() if data_quality is not None else None
        if normalized_quality in {"xbrl", "extracted"} and (has_rows or has_periods):
            has_structured_financial_statements = True
        if normalized_quality == "xbrl":
            has_xbrl = True
    if has_structured_financial_statements:
        availability = "structured_data_available"
    elif has_financial_statement_sections:
        availability = "statement_sections_available"
    else:
        availability = "not_available"
    return (
        has_financial_statement,
        has_xbrl,
        availability,
        has_structured_financial_statements,
        has_financial_statement_sections,
        has_financial_statement,
    )


def _write_tool_snapshot_file(
    *,
    repository: DocumentBlobRepositoryProtocol,
    processed_handle: ProcessedHandle,
    file_name: str,
    payload: dict[str, Any],
) -> None:
    """将快照文件写入仓储。"""

    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    repository.store_file(
        processed_handle,
        file_name,
        BytesIO(data),
        content_type="application/json",
        metadata={"encoding": "utf-8"},
    )


def _validate_processed_handle(
    *,
    processed_handle: ProcessedHandle,
    ticker: str,
    document_id: str,
) -> None:
    """校验处理句柄与目标文档一致。"""

    if processed_handle.ticker != ticker:
        raise ValueError(
            "processed_handle.ticker 与导出 ticker 不一致: "
            f"{processed_handle.ticker!r} != {ticker!r}"
        )
    if processed_handle.document_id != document_id:
        raise ValueError(
            "processed_handle.document_id 与导出 document_id 不一致: "
            f"{processed_handle.document_id!r} != {document_id!r}"
        )


def _extract_page_range(section: Mapping[str, Any]) -> Optional[list[int]]:
    """提取章节页码范围。"""

    raw = section.get("page_range")
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    start, end = raw
    if isinstance(start, int) and isinstance(end, int) and start > 0 and end > 0:
        return [start, end]
    return None

