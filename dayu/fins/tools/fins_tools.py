"""财报工具注册模块。"""

from __future__ import annotations

from typing import Any, Optional

from dayu.log import Log
from dayu.contracts.tool_configs import FinsToolLimits
from dayu.engine.tool_contracts import ToolTruncateSpec
from dayu.engine.tool_registry import ToolRegistry
from dayu.engine.tools.base import tool

from .ingestion_tools import register_ingestion_tools
from .result_types import (
    DocumentSectionsResult,
    FinancialStatementResult,
    ListDocumentsResult,
    NotSupportedResult,
    PageContentResult,
    SearchDocumentResult,
    SectionContentResult,
    TableDetailResult,
    TablesListResult,
    XbrlQueryResult,
)
from .service import FinsToolService

MODULE = "FINS.FINS_TOOLS"
FINS_TOOL_TAGS = frozenset({"fins"})


def _resolve_service(
    *,
    service: Optional[FinsToolService],
) -> FinsToolService:
    """解析或新建 FinsToolService 实例。

    只允许复用预构建的 `FinsToolService`，避免工具注册阶段继续感知仓储装配细节。

    Args:
        service: 预构建的 FinsToolService 实例。
    Returns:
        可用的 FinsToolService 实例。

    Raises:
        ValueError: 无法解析出可用 service 时抛出。
    """

    if service is not None:
        return service
    raise ValueError("需要提供预构建的 FinsToolService 实例")


def register_fins_read_tools(
    registry: ToolRegistry,
    *,
    service: FinsToolService,
    limits: Optional[FinsToolLimits] = None,
    timeout_budget: float | None = None,
) -> None:
    """注册财报读取工具集合。

    Args:
        registry: ToolRegistry 实例。
        service: 预构建的 FinsToolService 实例。
        limits: 可选工具限制配置。
        timeout_budget: Runner 为单次 tool call 提供的预算秒数；当前 fins 读工具预留该参数，
            暂未消费。

    Returns:
        无。

    Raises:
        ValueError: 配置非法时抛出。
    """

    resolved_limits = limits or FinsToolLimits()
    resolved_service = _resolve_service(
        service=service,
    )

    # 读取工具工厂函数列表（按注册顺序）
    read_tool_factories = [
        _create_list_documents_tool,
        _create_get_document_sections_tool,
        _create_read_section_tool,
        _create_search_document_tool,
        _create_list_tables_tool,
        _create_get_table_tool,
        _create_get_page_content_tool,
        _create_get_financial_statement_tool,
        _create_query_xbrl_facts_tool,
    ]

    del timeout_budget

    # 批量注册读取工具。读工具已统一收口为单一 fins 标签。
    for factory in read_tool_factories:
        name, func, schema = factory(registry, resolved_service, resolved_limits)
        registry.register(name, func, schema)

    Log.verbose(f"已注册 {len(read_tool_factories)} 个财报读取工具", module=MODULE)


def register_fins_ingestion_tools(
    registry: ToolRegistry,
    *,
    service_factory: Any,
    manager_key: str,
    timeout_budget: float | None = None,
) -> int:
    """注册财报下载与预处理长事务工具。

    Args:
        registry: ToolRegistry 实例。
        service_factory: `ticker -> FinsIngestionService` 工厂。
        manager_key: 长事务 job 管理器 key。
        timeout_budget: Runner 为单次 tool call 提供的预算秒数；当前 ingestion 工具预留该参数，
            暂未消费。

    Returns:
        新注册的 ingestion 工具数量。

    Raises:
        ValueError: 工厂或管理器 key 缺失时抛出。
    """

    tool_count = register_ingestion_tools(
        registry,
        service_factory=service_factory,
        manager_key=manager_key,
        timeout_budget=timeout_budget,
    )
    Log.verbose(f"已注册 {tool_count} 个财报下载/处理工具", module=MODULE)
    return tool_count


def _create_list_documents_tool(
    registry: ToolRegistry,
    service: FinsToolService,
    limits: FinsToolLimits,
) -> tuple[str, Any, Any]:
    """创建 `list_documents` 工具。

    Args:
        registry: 工具注册表实例。
        service: 财报工具服务实例。
        limits: 财报工具限制配置。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: 工具 schema 非法时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "直接传最自然的写法即可，不要手工穷举变体。",
            },
            "document_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "annual_report",
                        "semi_annual_report",
                        "quarterly_report",
                        "current_report",
                        "proxy",
                        "ownership",
                        "earnings_call",
                        "earnings_presentation",
                        "corporate_governance",
                        "material",
                    ],
                },
                "description": "可选文档类型过滤。只在你已明确要看哪类文档时填写；否则留空先看推荐文档。",
            },
            "fiscal_years": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "可选财年过滤。只在你已明确年份时填写，例如 [2024, 2025]。",
            },
            "fiscal_periods": {
                "type": "array",
                "items": {"type": "string", "enum": ["FY", "H1", "Q1", "Q2", "Q3", "Q4"]},
                "description": "可选财期过滤。只在你已明确财期时填写，例如 FY、Q1、Q2。",
            },
        },
        "required": ["ticker"],
    }

    @tool(
        registry,
        name="list_documents",
        description=(
            "列出公司可用文档。先用本工具拿到 document_id，再继续读章节、表格或财务数据。"
        ),
        parameters=parameters,
        tags=FINS_TOOL_TAGS,
        display_name="列出文档",
        summary_params=["ticker"],
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="list_items",
            limits={"max_items": limits.list_documents_max_items},
            target_field="documents",
        ),
    )
    def list_documents(
        ticker: str,
        document_types: Optional[list[str]] = None,
        fiscal_years: Optional[list[int]] = None,
        fiscal_periods: Optional[list[str]] = None,
    ) -> ListDocumentsResult:
        """列出可用文档。

        Args:
            ticker: 股票代码。
            document_types: 可选文档类型过滤（枚举数组）。
            fiscal_years: 可选财年过滤。
            fiscal_periods: 可选财期过滤。

        Returns:
            文档列表结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        return service.list_documents(
            ticker=ticker,
            document_types=document_types,
            fiscal_years=fiscal_years,
            fiscal_periods=fiscal_periods,
        )

    return list_documents.__tool_name__, list_documents, list_documents.__tool_schema__


def _create_get_document_sections_tool(
    registry: ToolRegistry,
    service: FinsToolService,
    limits: FinsToolLimits,
) -> tuple[str, Any, Any]:
    """创建 `get_document_sections` 工具。

    Args:
        registry: 工具注册表实例。
        service: 财报工具服务实例。
        limits: 财报工具限制配置。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: 工具 schema 非法时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
            },
            "document_id": {
                "type": "string",
            },
        },
        "required": ["ticker", "document_id"],
    }

    @tool(
        registry,
        name="get_document_sections",
        description="读取文档章节结构，返回可定位的章节 ref 列表。",
        parameters=parameters,
        tags=FINS_TOOL_TAGS,
        display_name="浏览财报结构",
        summary_params=["ticker"],
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="list_items",
            limits={"max_items": limits.get_document_sections_max_items},
            target_field="sections",
        ),
    )
    def get_document_sections(ticker: str, document_id: str) -> DocumentSectionsResult:
        """获取文档章节结构。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。

        Returns:
            章节结构结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        return service.get_document_sections(ticker=ticker, document_id=document_id)

    return get_document_sections.__tool_name__, get_document_sections, get_document_sections.__tool_schema__


def _create_read_section_tool(
    registry: ToolRegistry,
    service: FinsToolService,
    limits: FinsToolLimits,
) -> tuple[str, Any, Any]:
    """创建 `read_section` 工具。

    Args:
        registry: 工具注册表实例。
        service: 财报工具服务实例。
        limits: 财报工具限制配置。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: 工具 schema 非法时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
            },
            "document_id": {
                "type": "string",
            },
            "ref": {
                "type": "string",
                "description": "章节ref。只能来自于 `get_document_sections` 的 `sections[].ref`，`search_document` 的 `next_section_to_read.section.ref`，或 `search_document` 的 `next_section_by_query[*].section.ref`。",
            },
        },
        "required": ["ticker", "document_id", "ref"],
    }

    @tool(
        registry,
        name="read_section",
        description="读取章节全文。若正文里出现 [[t_XXXX]]，可用 get_table(t_XXXX) 读取对应表格。",
        parameters=parameters,
        tags=FINS_TOOL_TAGS,
        display_name="读取财报章节",
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="text_chars",
            limits={"max_chars": limits.read_section_max_chars},
            target_field="content",
        ),
    )
    def read_section(
        ticker: str,
        document_id: str,
        ref: str,
        **_kwargs,
    ) -> SectionContentResult:
        """读取章节正文。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            ref: 章节引用。
            **_kwargs: 历史兼容参数（如 within_section_ref），均被忽略。

        Returns:
            章节内容结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """
        _ = _kwargs

        return service.read_section(ticker=ticker, document_id=document_id, ref=ref)

    return read_section.__tool_name__, read_section, read_section.__tool_schema__


def _create_search_document_tool(
    registry: ToolRegistry,
    service: FinsToolService,
    limits: FinsToolLimits,
) -> tuple[str, Any, Any]:
    """创建 `search_document` 工具。

    Args:
        registry: 工具注册表实例。
        service: 财报工具服务实例。
        limits: 财报工具限制配置。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: 工具 schema 非法时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
            },
            "document_id": {
                "type": "string",
            },
            "query": {"type": "string", "description": "单个搜索词。只搜一个概念时使用；避免裸数字、裸百分比或过于宽泛的词。"},
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 20,
                "description": "多关键词搜索时使用，一次最多 20 个；与 query 互斥。只有这些词都服务同一主题时再一起传。",
            },
            "within_section_ref": {
                "type": "string",
                "description": "章节ref。结果太多时用它收窄范围。",
            },
            "mode": {
                "type": "string",
                "enum": ["auto", "exact", "keyword", "semantic"],
                "description": "搜索模式。通常用 auto；只有你明确要精确短语匹配或关键词匹配时再手动指定。",
            },
        },
        "required": ["ticker", "document_id"],
    }

    @tool(
        registry,
        name="search_document",
        description=(
            "在文档内搜索定位相关章节。先找最相关命中，再优先 read_section(top_match.ref) 精读；不要靠翻页继续猜。"
        ),
        parameters=parameters,
        tags=FINS_TOOL_TAGS,
        display_name="检索文档",
        summary_params=["query"],
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="list_items",
            limits={"max_items": limits.search_document_max_items},
            target_field="matches",
            continuation_hint={
                "continuation_required": False,
                "continuation_priority": None,
                "next_action": "read_section on matched section.ref to get full context, or narrow with within_section_ref",
            },
        ),
    )
    def search_document(
        ticker: str,
        document_id: str,
        query: Optional[str] = None,
        queries: Optional[list[str]] = None,
        within_section_ref: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> SearchDocumentResult:
        """搜索文档。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            query: 单条搜索词（与 queries 互斥）。
            queries: 批量搜索词（与 query 互斥，上限 20 条）。
            within_section_ref: 可选章节范围。
            mode: 搜索模式（auto/exact/keyword/semantic）。

        Returns:
            搜索结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        result = service.search_document(
            ticker=ticker,
            document_id=document_id,
            query=query,
            queries=queries,
            within_section_ref=within_section_ref,
            mode=mode,
            display_budget=limits.search_document_max_items,
        )
        # 剥离内部诊断信息，不暴露给 LLM
        result.pop("diagnostics", None)
        return result

    return search_document.__tool_name__, search_document, search_document.__tool_schema__


def _create_list_tables_tool(
    registry: ToolRegistry,
    service: FinsToolService,
    limits: FinsToolLimits,
) -> tuple[str, Any, Any]:
    """创建 `list_tables` 工具。

    Args:
        registry: 工具注册表实例。
        service: 财报工具服务实例。
        limits: 财报工具限制配置。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: 工具 schema 非法时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
            },
            "document_id": {
                "type": "string",
            },
            "financial_only": {
                "type": "boolean",
                "description": "只在你明确只看财务报表类表格时设为 true；否则保持默认 false。",
                "default": False,
            },
            "within_section_ref": {
                "type": "string",
                "description": "章节ref。想只看某一章里的表格时填写。",
            },
        },
        "required": ["ticker", "document_id"],
    }

    @tool(
        registry,
        name="list_tables",
        description="列出文档内表格，返回可定位的 table_ref 列表。",
        parameters=parameters,
        tags=FINS_TOOL_TAGS,
        display_name="列出表格",
        summary_params=["ticker"],
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="list_items",
            limits={"max_items": limits.list_tables_max_items},
            target_field="tables",
            continuation_hint={
                "continuation_required": False,
                "continuation_priority": None,
                "next_action": "use within_section_ref to narrow scope, or get_table for a specific table",
            },
        ),
    )
    def list_tables(
        ticker: str,
        document_id: str,
        financial_only: bool = False,
        within_section_ref: Optional[str] = None,
    ) -> TablesListResult:
        """列出表格。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            financial_only: 是否仅返回财务表。
            within_section_ref: 可选章节范围。

        Returns:
            表格列表结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        return service.list_tables(
            ticker=ticker,
            document_id=document_id,
            financial_only=financial_only,
            within_section_ref=within_section_ref,
        )

    return list_tables.__tool_name__, list_tables, list_tables.__tool_schema__


def _create_get_table_tool(
    registry: ToolRegistry,
    service: FinsToolService,
    limits: FinsToolLimits,
) -> tuple[str, Any, Any]:
    """创建 `get_table` 工具。

    Args:
        registry: 工具注册表实例。
        service: 财报工具服务实例。
        limits: 财报工具限制配置。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: 工具 schema 非法时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
            },
            "document_id": {
                "type": "string",
            },
            "table_ref": {
                "type": "string",
                "description": "表格ref。只能来自于`list_tables` 的 `tables[].table_ref` 或 `read_section` 正文里的 `[[t_XXXX]]`",
            },
        },
        "required": ["ticker", "document_id", "table_ref"],
    }

    @tool(
        registry,
        name="get_table",
        description="按 table_ref 读取单个表格。",
        parameters=parameters,
        tags=FINS_TOOL_TAGS,
        display_name="查看表格",
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="list_items",
            limits={"max_items": limits.get_table_max_items},
        ),
    )
    def get_table(ticker: str, document_id: str, table_ref: str) -> TableDetailResult:
        """读取表格。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            table_ref: 表格引用。

        Returns:
            表格结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        return service.get_table(ticker=ticker, document_id=document_id, table_ref=table_ref)

    return get_table.__tool_name__, get_table, get_table.__tool_schema__


def _create_get_page_content_tool(
    registry: ToolRegistry,
    service: FinsToolService,
    limits: FinsToolLimits,
) -> tuple[str, Any, Any]:
    """创建 `get_page_content` 工具。

    Args:
        registry: 工具注册表实例。
        service: 财报工具服务实例。
        limits: 财报工具限制配置。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: 工具 schema 非法时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
            },
            "document_id": {
                "type": "string",
            },
            "page_no": {"type": "integer", "description": "页码，从 1 开始。", "minimum": 1},
        },
        "required": ["ticker", "document_id", "page_no"],
    }

    @tool(
        registry,
        name="get_page_content",
        description="按页码读取同页内容。只有已有 page_range 且需要补同页上下文时才使用。",
        parameters=parameters,
        tags=FINS_TOOL_TAGS,
        display_name="读取页面",
        summary_params=["ticker"],
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="text_chars",
            limits={"max_chars": limits.get_page_content_max_chars},
            target_field="text_preview",
        ),
    )
    def get_page_content(ticker: str, document_id: str, page_no: int) -> PageContentResult | NotSupportedResult:
        """读取页面内容。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            page_no: 页码。

        Returns:
            页面内容结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        return service.get_page_content(ticker=ticker, document_id=document_id, page_no=page_no)

    return get_page_content.__tool_name__, get_page_content, get_page_content.__tool_schema__


def _create_get_financial_statement_tool(
    registry: ToolRegistry,
    service: FinsToolService,
    limits: FinsToolLimits,
) -> tuple[str, Any, Any]:
    """创建 `get_financial_statement` 工具。

    Args:
        registry: 工具注册表实例。
        service: 财报工具服务实例。
        limits: 财报工具限制配置。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: 工具 schema 非法时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
            },
            "document_id": {
                "type": "string",
            },
            "statement_type": {
                "type": "string",
                "description": "报表类型。通常先看 income、balance_sheet、cash_flow；只有明确需要时再看 equity 或 comprehensive_income。",
                "enum": ["income", "balance_sheet", "cash_flow", "equity", "comprehensive_income"],
            },
        },
        "required": ["ticker", "document_id", "statement_type"],
    }

    @tool(
        registry,
        name="get_financial_statement",
        description=(
            "读取标准财务报表。"
        ),
        parameters=parameters,
        tags=FINS_TOOL_TAGS,
        display_name="查看财务报表",
        summary_params=["statement_type"],
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="list_items",
            limits={"max_items": limits.get_financial_statement_max_items},
            target_field="rows",
        ),
    )
    def get_financial_statement(ticker: str, document_id: str, statement_type: str) -> FinancialStatementResult | NotSupportedResult:
        """读取财务报表。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            statement_type: 报表类型。

        Returns:
            财务报表结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        return service.get_financial_statement(
            ticker=ticker,
            document_id=document_id,
            statement_type=statement_type,
        )

    return get_financial_statement.__tool_name__, get_financial_statement, get_financial_statement.__tool_schema__


def _create_query_xbrl_facts_tool(
    registry: ToolRegistry,
    service: FinsToolService,
    limits: FinsToolLimits,
) -> tuple[str, Any, Any]:
    """创建 `query_xbrl_facts` 工具。

    Args:
        registry: 工具注册表实例。
        service: 财报工具服务实例。
        limits: 财报工具限制配置。

    Returns:
        `(tool_name, tool_callable, tool_schema)` 三元组。

    Raises:
        ValueError: 工具 schema 非法时抛出。
    """

    parameters = {
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
            },
            "document_id": {
                "type": "string",
            },
            "concepts": {
                "type": "array",
                "items": {"type": "string"},
                "description": "可选 XBRL 概念列表。已明确要找某个概念时填写；不确定时可留空，先看默认概念集。",
                "minItems": 1,
            },
            "statement_type": {"type": "string", "description": "可选报表类型过滤。想把结果收窄到某一类报表时再填。"},
            "period_end": {"type": "string", "description": "可选期末日期过滤，格式 YYYY-MM-DD。"},
            "fiscal_year": {"type": "integer", "description": "可选财年过滤。只在你已明确年份时填写。"},
            "fiscal_period": {"type": "string", "description": "可选财期过滤，例如 FY、Q1、Q2。"},
            "min_value": {"type": "number", "description": "可选最小值过滤。只在你明确要排除过小数值时填写。"},
            "max_value": {"type": "number", "description": "可选最大值过滤。只在你明确要排除过大数值时填写。"},
        },
        "required": ["ticker", "document_id"],
    }

    @tool(
        registry,
        name="query_xbrl_facts",
        description="查询结构化 XBRL 数值 facts。",
        parameters=parameters,
        tags=FINS_TOOL_TAGS,
        display_name="查询财务数据",
        summary_params=["concepts"],  # list[str]，_build_param_preview 展开为逗号分隔
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="list_items",
            limits={"max_items": limits.query_xbrl_facts_max_items},
            target_field="facts",
        ),
    )
    def query_xbrl_facts(
        ticker: str,
        document_id: str,
        concepts: Optional[list[str]] = None,
        statement_type: Optional[str] = None,
        period_end: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> XbrlQueryResult | NotSupportedResult:
        """查询 XBRL facts。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            concepts: 可选概念列表。
            statement_type: 可选报表类型。
            period_end: 可选期末日期。
            fiscal_year: 可选财年。
            fiscal_period: 可选财期。
            min_value: 可选最小值。
            max_value: 可选最大值。

        Returns:
            查询结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
        """

        return service.query_xbrl_facts(
            ticker=ticker,
            document_id=document_id,
            concepts=concepts,
            statement_type=statement_type,
            period_end=period_end,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            min_value=min_value,
            max_value=max_value,
        )

    return query_xbrl_facts.__tool_name__, query_xbrl_facts, query_xbrl_facts.__tool_schema__
