"""fins 财报工具服务层测试。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional, cast

import pytest

from dayu.engine.processors.base import DocumentProcessor
from dayu.engine.exceptions import ToolArgumentError
from dayu.engine.tool_errors import ToolBusinessError
from dayu.engine.processors.source import Source
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.fins.domain.document_models import CompanyMeta, SourceHandle
from dayu.fins.domain.enums import SourceKind
from dayu.fins.tools.result_types import DocumentSectionsResult
from tests.fins.legacy_repository_adapters import LegacyCompatibleFinsToolService as FinsToolService


class DummySource:
    """测试用 Source。"""

    def __init__(self, uri: str, media_type: Optional[str] = "text/html") -> None:
        """初始化 Source。

        Args:
            uri: 资源 URI。
            media_type: 媒体类型。

        Returns:
            无。

        Raises:
            ValueError: URI 为空时抛出。
        """

        if not uri:
            raise ValueError("uri 不能为空")
        self.uri = uri
        self.media_type = media_type
        self.content_length = None
        self.etag = None

    def open(self) -> Any:
        """打开流（测试桩）。

        Args:
            无。

        Returns:
            无。

        Raises:
            OSError: 测试桩不提供该能力。
        """

        raise OSError("dummy source 不支持 open")

    def materialize(self, suffix: Optional[str] = None) -> Any:
        """物化路径（测试桩）。

        Args:
            suffix: 可选后缀。

        Returns:
            无。

        Raises:
            OSError: 测试桩不提供该能力。
        """

        raise OSError("dummy source 不支持 materialize")


@dataclass
class FakeProcessor:
    """全能力处理器桩。"""

    document_id: str

    def list_sections(self) -> list[dict[str, Any]]:
        """返回章节列表。"""

        return [
            {
                "ref": "s_0001",
                "title": f"章节-{self.document_id}",
                "level": 1,
                "parent_ref": None,
                "preview": "preview",
                "page_range": [1, 2],
            }
        ]

    def get_section_title(self, ref: str) -> Optional[str]:
        """根据 section ref 获取章节标题。"""
        for sec in self.list_sections():
            if sec.get("ref") == ref:
                return sec.get("title")
        return None

    def read_section(self, ref: str) -> dict[str, Any]:
        """读取章节。"""

        return {
            "ref": ref,
            "title": "章节",
            "content": "这是章节正文 [[t_0001]]",
            "tables": ["t_0001"],
            "word_count": 4,
            "contains_full_text": False,
            "page_range": [1, 2],
        }

    def search(self, query: str, within_ref: Optional[str] = None) -> list[dict[str, Any]]:
        """返回搜索命中。"""

        return [
            {
                "section_ref": within_ref or "s_0001",
                "section_title": "章节",
                "snippet": f"命中: {query}",
                "page_no": 1,
            }
        ]

    def list_tables(self) -> list[dict[str, Any]]:
        """返回表格列表。"""

        return [
            {
                "table_ref": "t_0001",
                "caption": "表1",
                "row_count": 2,
                "col_count": 2,
                "is_financial": True,
                "table_type": "financial",
                "headers": ["A", "B"],
                "section_ref": "s_0001",
                "page_no": 2,
            }
        ]

    def read_table(self, table_ref: str) -> dict[str, Any]:
        """读取表格。"""

        return {
            "table_ref": table_ref,
            "caption": "表1",
            "data_format": "records",
            "data": [{"A": 1, "B": 2}],
            "columns": ["A", "B"],
            "row_count": 1,
            "col_count": 2,
            "is_financial": True,
            "table_type": "financial",
            "section_ref": "s_0001",
            "page_no": 2,
        }

    def get_page_content(self, page_no: int) -> dict[str, Any]:
        """返回页面内容。"""

        return {
            "sections": [
                {
                    "ref": "s_0001",
                    "title": "章节",
                    "content_preview": "预览",
                    "is_partial": False,
                    "start_on_page": True,
                    "end_on_page": True,
                }
            ],
            "tables": [
                {
                    "table_ref": "t_0001",
                    "caption": "表1",
                    "context_before": "上下文",
                    "row_count": 2,
                    "col_count": 2,
                    "is_financial": True,
                }
            ],
            "text_preview": f"第{page_no}页文本",
            "has_content": True,
            "total_items": 2,
            "supported": True,
        }

    def get_financial_statement(self, statement_type: str) -> dict[str, Any]:
        """返回财务报表。"""

        return {
            "statement_type": statement_type,
            "periods": [],
            "rows": [],
            "currency": "USD",
            "units": "millions",
            "data_quality": "xbrl",
            "statement_locator": {
                "statement_type": statement_type,
                "statement_title": "Income Statement",
                "period_labels": ["FY2025"],
                "row_labels": ["Revenue"],
            },
        }

    def query_xbrl_facts(
        self,
        concepts: Optional[list[str]] = None,
        statement_type: Optional[str] = None,
        period_end: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> dict[str, Any]:
        """返回 XBRL 查询结果。"""

        del statement_type, period_end, fiscal_year, fiscal_period, min_value, max_value
        resolved_concepts = concepts or []
        concept = resolved_concepts[0] if resolved_concepts else "Revenues"
        return {
            "query_params": {"concepts": resolved_concepts},
            "facts": [
                {
                    "concept": concept,
                    "label": "概念",
                    "numeric_value": 1.0,
                    "text_value": None,
                    "content_type": None,
                    "unit": "usd",
                    "period_end": "2024-09-28",
                    "fiscal_year": 2024,
                    "fiscal_period": "FY",
                    "statement_type": "IncomeStatement",
                }
            ],
            "total": 1,
        }


@dataclass
class BasicProcessor:
    """无扩展能力处理器桩。"""

    document_id: str

    def list_sections(self) -> list[dict[str, Any]]:
        """返回章节列表。"""

        return [{"ref": "s_0001", "title": "章节", "level": 1, "parent_ref": None, "preview": "x"}]

    def read_section(self, ref: str) -> dict[str, Any]:
        """读取章节。"""

        return {"ref": ref, "content": "正文", "tables": [], "contains_full_text": True}

    def search(self, query: str, within_ref: Optional[str] = None) -> list[dict[str, Any]]:
        """搜索。"""

        del within_ref
        return [{"section_ref": "s_0001", "section_title": "章节", "snippet": query}]

    def list_tables(self) -> list[dict[str, Any]]:
        """列表表格。"""

        return []

    def read_table(self, table_ref: str) -> dict[str, Any]:
        """读取表格。"""

        return {
            "table_ref": table_ref,
            "data_format": "markdown",
            "data": "|A|\\n|---|",
            "columns": None,
            "row_count": 1,
            "col_count": 1,
            "is_financial": False,
        }


@dataclass
class SearchEnhancedProcessor(FakeProcessor):
    """用于验证 search 输出直通行为的处理器桩。"""

    def search(self, query: str, within_ref: Optional[str] = None) -> list[dict[str, Any]]:
        """返回已去重且增强后的命中。

        Args:
            query: 查询词。
            within_ref: 可选章节限制。

        Returns:
            命中列表。

        Raises:
            RuntimeError: 搜索失败时抛出。
        """

        return [
            {
                "section_ref": within_ref or "s_0001",
                "section_title": "章节",
                "snippet": f"{query} appears in agreement terms and closing conditions.",
                "page_no": 1,
            },
            {
                "section_ref": within_ref or "s_0001",
                "section_title": "章节",
                "snippet": f"The board approved {query} and related disclosure requirements.",
                "page_no": 1,
            },
        ]


@dataclass
class SynonymFallbackProcessor(BasicProcessor):
    """用于验证同义词回退检索的处理器桩。"""

    def search(self, query: str, within_ref: Optional[str] = None) -> list[dict[str, Any]]:
        """按查询词返回可控命中。

        Args:
            query: 查询词。
            within_ref: 可选章节范围。

        Returns:
            命中列表。

        Raises:
            RuntimeError: 搜索失败时抛出。
        """

        del within_ref
        normalized = str(query).strip().lower()
        if normalized == "share repurchase":
            return [
                {
                    "section_ref": "s_0001",
                    "section_title": "章节",
                    "snippet": "The board approved share repurchase authorization in 2025.",
                    "page_no": 1,
                }
            ]
        return []


@dataclass
class Bm25fOrderingProcessor(FakeProcessor):
    """用于验证 BM25F 桶内排序的处理器桩。"""

    def list_sections(self) -> list[dict[str, Any]]:
        """返回带有明显字段差异的章节摘要。

        Args:
            无。

        Returns:
            章节摘要列表。

        Raises:
            RuntimeError: 无。
        """

        return [
            {
                "ref": "s_0001",
                "title": "Item 5. Share Repurchase Program",
                "level": 1,
                "parent_ref": None,
                "preview": "The company expanded its share repurchase program and buyback authorization.",
                "page_range": [1, 2],
            },
            {
                "ref": "s_0002",
                "title": "Item 7. Liquidity and Capital Resources",
                "level": 1,
                "parent_ref": None,
                "preview": "Liquidity discussion with a brief mention of repurchase authorization.",
                "page_range": [3, 4],
            },
        ]

    def search(self, query: str, within_ref: Optional[str] = None) -> list[dict[str, Any]]:
        """返回相同策略桶中的两条命中。

        Args:
            query: 查询词。
            within_ref: 可选章节范围。

        Returns:
            命中列表。

        Raises:
            RuntimeError: 无。
        """

        del within_ref
        normalized = str(query).strip().lower()
        if normalized != "share repurchase":
            return []
        return [
            {
                "section_ref": "s_0002",
                "section_title": "Item 7. Liquidity and Capital Resources",
                "snippet": "The company referenced the share repurchase authorization while discussing liquidity.",
                "page_no": 3,
            },
            {
                "section_ref": "s_0001",
                "section_title": "Item 5. Share Repurchase Program",
                "snippet": "The share repurchase program remained a core capital allocation priority.",
                "page_no": 1,
            },
        ]


@dataclass
class RawTextTableProcessor(BasicProcessor):
    """返回非 Markdown 文本表格的处理器桩。"""

    def read_table(self, table_ref: str) -> dict[str, Any]:
        """读取表格并返回原始文本内容。

        Args:
            table_ref: 表格引用。

        Returns:
            原始文本形态的表格内容字典。

        Raises:
            KeyError: 表格不存在时抛出。
        """

        return {
            "table_ref": table_ref,
            "caption": None,
            "data_format": "markdown",
            "data": "{'type': 'GENERAL', 'headers': [], 'data': [['', '']]}",
            "columns": None,
            "row_count": 2,
            "col_count": 2,
            "is_financial": False,
        }


class SearchEnhancedProcessorRegistry:
    """返回固定 SearchEnhancedProcessor 的注册表桩。"""

    def __init__(self) -> None:
        """初始化注册表桩。"""

        self.create_call_count = 0

    def create(self, source: Source, *, form_type: Optional[str] = None, media_type: Optional[str] = None) -> Any:
        """创建处理器实例。

        Args:
            source: 文档来源。
            form_type: 可选 form type。
            media_type: 可选 media type。

        Returns:
            SearchEnhancedProcessor 实例。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        del form_type, media_type
        document_id = str(source.uri).split("/")[-1].split(".")[0]
        self.create_call_count += 1
        return SearchEnhancedProcessor(document_id=document_id)

    def create_with_fallback(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
        on_fallback: Optional[Any] = None,
    ) -> Any:
        """兼容统一回退接口并复用 create。

        Args:
            source: 文档来源。
            form_type: 可选 form type。
            media_type: 可选 media type。
            on_fallback: 回退回调（本桩不触发）。

        Returns:
            SearchEnhancedProcessor 实例。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        del on_fallback
        return self.create(source, form_type=form_type, media_type=media_type)


class RawTextProcessorRegistry:
    """返回固定 RawTextTableProcessor 的注册表桩。"""

    def create(self, source: Source, *, form_type: Optional[str] = None, media_type: Optional[str] = None) -> Any:
        """创建处理器实例。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            RawTextTableProcessor 实例。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        del form_type, media_type
        document_id = str(source.uri).split("/")[-1].split(".")[0]
        return RawTextTableProcessor(document_id=document_id)

    def create_with_fallback(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
        on_fallback: Optional[Any] = None,
    ) -> Any:
        """兼容统一回退接口并复用 create。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。
            on_fallback: 回退回调（本桩不触发）。

        Returns:
            RawTextTableProcessor 实例。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        del on_fallback
        return self.create(source, form_type=form_type, media_type=media_type)


class SynonymFallbackProcessorRegistry:
    """返回固定 SynonymFallbackProcessor 的注册表桩。"""

    def create(self, source: Source, *, form_type: Optional[str] = None, media_type: Optional[str] = None) -> Any:
        """创建处理器实例。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            SynonymFallbackProcessor 实例。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        del source, form_type, media_type
        return SynonymFallbackProcessor(document_id="fil_1")

    def create_with_fallback(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
        on_fallback: Optional[Any] = None,
    ) -> Any:
        """兼容统一回退接口并复用 create。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。
            on_fallback: 回退回调（本桩不触发）。

        Returns:
            SynonymFallbackProcessor 实例。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        del on_fallback
        return self.create(source, form_type=form_type, media_type=media_type)


class Bm25fOrderingProcessorRegistry:
    """返回固定 Bm25fOrderingProcessor 的注册表桩。"""

    def create(self, source: Source, *, form_type: Optional[str] = None, media_type: Optional[str] = None) -> Any:
        """创建处理器实例。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            Bm25fOrderingProcessor 实例。

        Raises:
            RuntimeError: 无。
        """

        del source, form_type, media_type
        return Bm25fOrderingProcessor(document_id="fil_1")

    def create_with_fallback(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
        on_fallback: Optional[Any] = None,
    ) -> Any:
        """兼容统一回退接口并复用 create。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。
            on_fallback: 回退回调（本桩不触发）。

        Returns:
            Bm25fOrderingProcessor 实例。

        Raises:
            RuntimeError: 无。
        """

        del on_fallback
        return self.create(source, form_type=form_type, media_type=media_type)


@dataclass
class TaxonomyAwareFactProcessor(FakeProcessor):
    """用于验证默认 concept 映射与 fact 去重/清洗的处理器桩。"""

    taxonomy: str = "us-gaap"
    last_concepts: Optional[list[str]] = None

    def get_xbrl_taxonomy(self) -> str:
        """返回固定 taxonomy。

        Args:
            无。

        Returns:
            taxonomy 名称。

        Raises:
            RuntimeError: 无。
        """

        return self.taxonomy

    def query_xbrl_facts(
        self,
        concepts: Optional[list[str]] = None,
        statement_type: Optional[str] = None,
        period_end: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> dict[str, Any]:
        """返回包含重复与 XHTML 文本的事实集合。

        Args:
            concepts: 概念列表。
            statement_type: 可选报表类型。
            period_end: 可选期末。
            fiscal_year: 可选财年。
            fiscal_period: 可选财期。
            min_value: 可选最小值。
            max_value: 可选最大值。

        Returns:
            包含重复与文本事实的查询结果。

        Raises:
            RuntimeError: 无。
        """

        del statement_type, period_end, fiscal_year, fiscal_period, min_value, max_value
        self.last_concepts = list(concepts or [])
        return {
            "query_params": {"concepts": list(concepts or [])},
            "facts": [
                {
                    "concept": "us-gaap:Revenues",
                    "label": "Revenue",
                    "numeric_value": 100.0,
                    "unit": "USD",
                    "period_end": "2024-09-28",
                    "fiscal_year": 2024,
                    "fiscal_period": "FY",
                },
                {
                    "concept": "us-gaap:Revenues",
                    "label": "RevenueText",
                    "text_value": "<div>100</div>",
                    "unit": "USD",
                    "period_end": "2024-09-28",
                    "fiscal_year": 2024,
                    "fiscal_period": "FY",
                },
                {
                    "concept": "us-gaap:Assets",
                    "label": "Assets",
                    "text_value": "<div>Assets<br>line</div>",
                    "unit": "USD",
                    "period_end": "2024-09-28",
                    "fiscal_year": 2024,
                    "fiscal_period": "FY",
                },
            ],
            "total": 3,
        }


class TaxonomyAwareProcessorRegistry:
    """返回固定 TaxonomyAwareFactProcessor 的注册表桩。"""

    def __init__(self, *, taxonomy: str = "us-gaap") -> None:
        """初始化注册表桩。

        Args:
            taxonomy: 处理器返回的 taxonomy。

        Returns:
            无。

        Raises:
            RuntimeError: 无。
        """

        self.taxonomy = taxonomy
        self.last_processor: Optional[TaxonomyAwareFactProcessor] = None

    def create(self, source: Source, *, form_type: Optional[str] = None, media_type: Optional[str] = None) -> Any:
        """创建处理器实例。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            TaxonomyAwareFactProcessor 实例。

        Raises:
            RuntimeError: 无。
        """

        del source, form_type, media_type
        processor = TaxonomyAwareFactProcessor(document_id="fil_1", taxonomy=self.taxonomy)
        self.last_processor = processor
        return processor

    def create_with_fallback(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
        on_fallback: Optional[Any] = None,
    ) -> Any:
        """兼容统一回退接口并复用 create。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。
            on_fallback: 回退回调（本桩不触发）。

        Returns:
            TaxonomyAwareFactProcessor 实例。

        Raises:
            RuntimeError: 无。
        """

        del on_fallback
        return self.create(source, form_type=form_type, media_type=media_type)


@dataclass
class FiscalPeriodPreferenceFactProcessor(FakeProcessor):
    """用于验证 period_end 场景下 fiscal_period 优先保留的处理器桩。"""

    def query_xbrl_facts(
        self,
        concepts: Optional[list[str]] = None,
        statement_type: Optional[str] = None,
        period_end: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        fiscal_period: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> dict[str, Any]:
        """返回同一期末存在 fiscal_period 差异的重复事实。

        Args:
            concepts: 概念列表。
            statement_type: 可选报表类型。
            period_end: 可选期末。
            fiscal_year: 可选财年。
            fiscal_period: 可选财期。
            min_value: 可选最小值。
            max_value: 可选最大值。

        Returns:
            重复事实查询结果。

        Raises:
            RuntimeError: 无。
        """

        del concepts, statement_type, period_end, fiscal_year, fiscal_period, min_value, max_value
        return {
            "query_params": {"concepts": ["NetIncomeLoss"]},
            "facts": [
                {
                    "concept": "us-gaap:NetIncomeLoss",
                    "label": "Net income",
                    "numeric_value": 24780000000.0,
                    "unit": "usd",
                    "period_end": "2025-03-29",
                    "fiscal_year": 2025,
                    "fiscal_period": None,
                    "statement_type": None,
                    "decimals": "-6",
                },
                {
                    "concept": "us-gaap:NetIncomeLoss",
                    "label": "Net income",
                    "numeric_value": 24780000000.0,
                    "unit": "usd",
                    "period_end": "2025-03-29",
                    "fiscal_year": 2025,
                    "fiscal_period": "Q2",
                    "statement_type": "IncomeStatement",
                    "decimals": "-6",
                },
            ],
            "total": 2,
        }


class FiscalPeriodPreferenceProcessorRegistry:
    """返回固定 FiscalPeriodPreferenceFactProcessor 的注册表桩。"""

    def create(self, source: Source, *, form_type: Optional[str] = None, media_type: Optional[str] = None) -> Any:
        """创建处理器实例。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            FiscalPeriodPreferenceFactProcessor 实例。

        Raises:
            RuntimeError: 无。
        """

        del source, form_type, media_type
        return FiscalPeriodPreferenceFactProcessor(document_id="fil_1")

    def create_with_fallback(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
        on_fallback: Optional[Any] = None,
    ) -> Any:
        """兼容统一回退接口并复用 create。

        Args:
            source: 文档来源。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。
            on_fallback: 回退回调（本桩不触发）。

        Returns:
            FiscalPeriodPreferenceFactProcessor 实例。

        Raises:
            RuntimeError: 无。
        """

        del on_fallback
        return self.create(source, form_type=form_type, media_type=media_type)


class FakeProcessorRegistry:
    """处理器注册表桩。"""

    def __init__(self, *, basic_mode: bool = False) -> None:
        """初始化处理器注册表桩。

        Args:
            basic_mode: 是否返回基础处理器。

        Returns:
            无。

        Raises:
            无。
        """

        self.basic_mode = basic_mode
        self.create_call_count = 0
        self.create_call_count_by_document: dict[str, int] = {}
        self._lock = Lock()

    def create(self, source: Source, *, form_type: Optional[str] = None, media_type: Optional[str] = None) -> Any:
        """创建处理器实例。

        Args:
            source: Source。
            form_type: 表单类型。
            media_type: 媒体类型。

        Returns:
            处理器实例。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        del form_type, media_type
        document_id = str(source.uri).split("/")[-1].split(".")[0]
        with self._lock:
            self.create_call_count += 1
            self.create_call_count_by_document[document_id] = (
                self.create_call_count_by_document.get(document_id, 0) + 1
            )
        if self.basic_mode:
            return BasicProcessor(document_id=document_id)
        return FakeProcessor(document_id=document_id)

    def create_with_fallback(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
        on_fallback: Optional[Any] = None,
    ) -> Any:
        """兼容统一回退接口并复用 create。

        Args:
            source: Source。
            form_type: 表单类型。
            media_type: 媒体类型。
            on_fallback: 回退回调（本桩不触发）。

        Returns:
            处理器实例。

        Raises:
            RuntimeError: 创建失败时抛出。
        """

        del on_fallback
        return self.create(source, form_type=form_type, media_type=media_type)


class FakeRepository:
    """文档仓储桩。"""

    def __init__(self) -> None:
        """初始化仓储桩。"""

        self._filing_meta: dict[str, dict[str, Any]] = {
            "fil_1": {
                "document_id": "fil_1",
                "form_type": "10-K",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "report_date": "2024-09-28",
                "filing_date": "2024-11-01",
                "amended": False,
                "is_deleted": False,
                "ingest_complete": True,
            },
            "fil_2": {
                "document_id": "fil_2",
                "form_type": "10-Q",
                "fiscal_year": 2025,
                "fiscal_period": "Q1",
                "report_date": "2025-03-29",
                "filing_date": "2025-05-02",
                "amended": False,
                "is_deleted": False,
                "ingest_complete": True,
            },
        }
        self._material_meta: dict[str, dict[str, Any]] = {
            "mat_1": {
                "document_id": "mat_1",
                "form_type": "MATERIAL_OTHER",
                "material_name": "Deck",
                "report_date": "2025-06-01",
                "filing_date": "2025-06-01",
                "amended": False,
                "is_deleted": False,
                "ingest_complete": True,
            }
        }
        self.get_processed_meta_calls = 0

    def list_document_ids(self, ticker: str, source_kind: Optional[SourceKind] = None) -> list[str]:
        """列出文档 ID。"""

        del ticker
        if source_kind == SourceKind.FILING:
            return sorted(self._filing_meta.keys())
        if source_kind == SourceKind.MATERIAL:
            return sorted(self._material_meta.keys())
        return sorted(list(self._filing_meta.keys()) + list(self._material_meta.keys()))

    def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
        """解析已存在的 ticker。"""

        if "AAPL" in candidates:
            return "AAPL"
        return None

    def get_document_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
        """读取文档 meta。"""

        del ticker
        if document_id in self._filing_meta:
            return dict(self._filing_meta[document_id])
        if document_id in self._material_meta:
            return dict(self._material_meta[document_id])
        raise FileNotFoundError(document_id)

    def get_source_handle(self, ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
        """获取 source handle。"""

        del ticker
        if source_kind == SourceKind.FILING and document_id in self._filing_meta:
            return SourceHandle(ticker="AAPL", document_id=document_id, source_kind=source_kind.value)
        if source_kind == SourceKind.MATERIAL and document_id in self._material_meta:
            return SourceHandle(ticker="AAPL", document_id=document_id, source_kind=source_kind.value)
        raise FileNotFoundError(document_id)

    def get_primary_source(self, ticker: str, document_id: str, source_kind: SourceKind) -> Source:
        """获取 source。"""

        del ticker
        return DummySource(uri=f"local://{source_kind.value}/{document_id}.html", media_type="text/html")

    def get_company_meta(self, ticker: str) -> CompanyMeta:
        """获取公司信息。"""

        return CompanyMeta(
            company_id="320193",
            company_name="Apple Inc.",
            ticker=ticker,
            market="US",
            resolver_version="test",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    def get_processed_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
        """读取 processed 元数据（用于能力标志）。"""

        del ticker
        self.get_processed_meta_calls += 1
        # 所有 filing 都有财务数据；material 均无
        if document_id.startswith("fil_"):
            return {"has_financial_data": True}
        return {"has_financial_data": False}


@pytest.mark.unit
def test_list_documents_uses_new_contract_without_processed_fields() -> None:
    """验证 list_documents 新契约字段。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(ticker="AAPL")

    assert result["total"] == 3
    assert result["matched"] == 3
    # company 对象包含市场信息
    company = result["company"]
    assert company["ticker"] == "AAPL"
    assert company["name"] == "Apple Inc."
    assert company["market"] == "US"
    assert "issuer_type" not in company
    # match_status 正常
    assert result["match_status"] == "ok"
    assert "suggestion" not in result
    # 文档级别包含 document_type 和能力标志
    first_doc = result["documents"][0]
    assert "section_count" not in first_doc
    assert "table_count" not in first_doc
    assert "quality" not in first_doc
    assert "document_type" in first_doc
    assert first_doc["has_financial_data"] is not None
    filing_doc = next(item for item in result["documents"] if item["document_id"] == "fil_1")
    material_doc = next(item for item in result["documents"] if item["document_id"] == "mat_1")
    assert filing_doc["has_financial_data"] is True
    assert material_doc["has_financial_data"] is False
    # 文档中不再冗余 ticker/company
    assert "ticker" not in first_doc
    assert "company" not in first_doc
    recommended = result["recommended_documents"]
    assert recommended["latest_document_id"] == "mat_1"
    assert "latest_financial_update_document_id" not in recommended
    assert "recommended_for_structured_financial_statements_document_id" not in recommended
    assert recommended["latest_annual_report_document_id"] == "fil_1"
    assert recommended["latest_quarterly_report_document_id"] == "fil_2"
    assert recommended["latest_earnings_call_document_id"] is None
    assert recommended["latest_earnings_presentation_document_id"] is None
    assert recommended["latest_material_document_id"] == "mat_1"
    assert recommended["recommended_for_company_overview_document_id"] == "fil_1"


@pytest.mark.unit
def test_list_documents_normalizes_material_form_type_and_sets_earnings_recommendations() -> None:
    """验证 material form_type 归一化后可稳定映射到 earnings document_type。"""

    class EarningsMaterialRepository(FakeRepository):
        """用于覆盖 material form_type 归一化分支的仓储桩。"""

        def __init__(self) -> None:
            """初始化仓储桩。"""

            super().__init__()
            self._material_meta = {
                "mat_call": {
                    "document_id": "mat_call",
                    "form_type": "earning_calls",
                    "material_name": "Call",
                    "report_date": "2025-06-02",
                    "filing_date": "2025-06-02",
                    "amended": False,
                    "is_deleted": False,
                    "ingest_complete": True,
                },
                "mat_presentation": {
                    "document_id": "mat_presentation",
                    "form_type": "earning_presentations",
                    "material_name": "Presentation",
                    "report_date": "2025-06-01",
                    "filing_date": "2025-06-01",
                    "amended": False,
                    "is_deleted": False,
                    "ingest_complete": True,
                },
            }

    repository = EarningsMaterialRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(
        ticker="AAPL",
        document_types=["earnings_call", "earnings_presentation"],
    )

    assert result["matched"] == 2
    document_types = {item["document_id"]: item["document_type"] for item in result["documents"]}
    assert document_types["mat_call"] == "earnings_call"
    assert document_types["mat_presentation"] == "earnings_presentation"
    recommended = result["recommended_documents"]
    assert recommended["latest_earnings_call_document_id"] == "mat_call"
    assert recommended["latest_earnings_presentation_document_id"] == "mat_presentation"
    assert recommended["latest_material_document_id"] is None
    assert "latest_financial_update_document_id" not in recommended
    assert "recommended_for_structured_financial_statements_document_id" not in recommended
    assert recommended["recommended_for_company_overview_document_id"] == "fil_1"


@pytest.mark.unit
def test_list_documents_recommended_company_overview_ignores_document_type_filter() -> None:
    """验证 company overview 推荐槽位基于全量文档而非当前过滤结果。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(
        ticker="AAPL",
        document_types=["quarterly_report", "earnings_call"],
    )

    assert result["matched"] == 1
    assert {item["document_type"] for item in result["documents"]} == {"quarterly_report"}
    recommended = result["recommended_documents"]
    assert recommended["latest_quarterly_report_document_id"] == "fil_2"
    assert recommended["latest_annual_report_document_id"] == "fil_1"
    assert "latest_financial_update_document_id" not in recommended
    assert "recommended_for_structured_financial_statements_document_id" not in recommended
    assert recommended["recommended_for_company_overview_document_id"] == "fil_1"


@pytest.mark.unit
def test_processor_cache_hit_by_ticker_document_id() -> None:
    """验证同文档重复调用只创建一次处理器。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    first_result = service.get_document_sections(ticker="AAPL", document_id="fil_1")
    service.get_document_sections(ticker="AAPL", document_id="fil_1")

    assert "has_page_info" not in first_result
    assert processor_registry.create_call_count == 1
    assert repository.get_processed_meta_calls == 0


@pytest.mark.unit
def test_get_document_sections_citation_covers_sec_edgar_supplementary_and_uploaded() -> None:
    """验证 citation 会根据元数据推断不同 source_type。"""

    class CitationRepository(FakeRepository):
        """用于覆盖 citation 来源类型分支的仓储桩。"""

        def __init__(self) -> None:
            """初始化仓储桩。"""

            super().__init__()
            self._filing_meta["fil_1"]["source_kind"] = SourceKind.FILING.value
            self._filing_meta["fil_1"]["ingest_method"] = "download"
            self._filing_meta["fil_1"]["accession_number"] = "0000320193-24-000123"
            self._material_meta["mat_1"]["source_kind"] = SourceKind.MATERIAL.value
            self._filing_meta["custom_1"] = {
                "document_id": "custom_1",
                "form_type": "6-K",
                "report_date": "2025-01-01",
                "filing_date": "2025-01-02",
                "amended": False,
                "is_deleted": False,
                "ingest_complete": True,
            }

        def get_source_handle(self, ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
            """为 custom_1 提供 filing source handle。"""

            if document_id == "custom_1" and source_kind == SourceKind.FILING:
                return SourceHandle(ticker=ticker, document_id=document_id, source_kind=source_kind.value)
            return super().get_source_handle(ticker, document_id, source_kind)

    service = FinsToolService(repository=CitationRepository(), processor_registry=FakeProcessorRegistry())

    filing_result = service.get_document_sections(ticker="AAPL", document_id="fil_1")
    material_result = service.get_document_sections(ticker="AAPL", document_id="mat_1")
    uploaded_result = service.get_document_sections(ticker="AAPL", document_id="custom_1")

    assert filing_result["citation"]["source_type"] == "SEC_EDGAR"
    assert filing_result["citation"]["accession_no"] == "0000320193-24-000123"
    assert material_result["citation"]["source_type"] == "SUPPLEMENTARY"
    assert uploaded_result["citation"]["source_type"] == "UPLOADED"


@pytest.mark.unit
def test_processor_cache_lru_evicts_old_entry() -> None:
    """验证 Processor LRU 淘汰行为。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(
        repository=repository,
        processor_registry=processor_registry,
        processor_cache_max_entries=1,
    )

    service.get_document_sections(ticker="AAPL", document_id="fil_1")
    service.get_document_sections(ticker="AAPL", document_id="fil_2")
    service.get_document_sections(ticker="AAPL", document_id="fil_1")

    assert processor_registry.create_call_count_by_document["fil_1"] == 2


@pytest.mark.unit
def test_processor_creation_is_singleflight_under_concurrency() -> None:
    """验证并发下同文档仅创建一次处理器。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    def run_once(_: int) -> DocumentSectionsResult:
        """并发执行一次章节读取。"""

        return service.get_document_sections(ticker="AAPL", document_id="fil_1")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(run_once, range(32)))

    assert len(results) == 32
    assert processor_registry.create_call_count_by_document["fil_1"] == 1


@pytest.mark.unit
def test_service_fallback_to_next_processor_when_primary_creation_fails() -> None:
    """验证服务层在首候选创建失败时会回退到下一候选。"""

    class BrokenProcessor:
        """首候选处理器：构造阶段故意失败。"""

        def __init__(
            self,
            source: Source,
            *,
            form_type: Optional[str] = None,
            media_type: Optional[str] = None,
        ) -> None:
            """初始化并抛出异常。"""

            del source, form_type, media_type
            raise RuntimeError("broken")

        @classmethod
        def supports(
            cls,
            source: Source,
            *,
            form_type: Optional[str] = None,
            media_type: Optional[str] = None,
        ) -> bool:
            """始终声明支持。"""

            del source, form_type, media_type
            return True

    class RecoveredProcessor:
        """次候选处理器：用于验证回退成功。"""

        def __init__(
            self,
            source: Source,
            *,
            form_type: Optional[str] = None,
            media_type: Optional[str] = None,
        ) -> None:
            """初始化处理器。"""

            del form_type, media_type
            self.document_id = str(source.uri).split("/")[-1].split(".")[0]

        @classmethod
        def supports(
            cls,
            source: Source,
            *,
            form_type: Optional[str] = None,
            media_type: Optional[str] = None,
        ) -> bool:
            """始终声明支持。"""

            del source, form_type, media_type
            return True

        def list_sections(self) -> list[dict[str, Any]]:
            """返回最小章节列表。"""

            return [{"ref": "sec_1", "title": f"{self.document_id}", "level": 1, "parent_ref": None}]

    repository = FakeRepository()
    registry = ProcessorRegistry()
    registry.register(cast(type[DocumentProcessor], BrokenProcessor), name="broken", priority=10)
    registry.register(cast(type[DocumentProcessor], RecoveredProcessor), name="recovered", priority=5)
    service = FinsToolService(repository=repository, processor_registry=registry)

    processor = service._get_or_create_processor(ticker="AAPL", document_id="fil_1")

    assert isinstance(processor, RecoveredProcessor)


@pytest.mark.unit
def test_not_supported_for_page_and_financial_capabilities() -> None:
    """验证能力降级返回 not_supported。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry(basic_mode=True)
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    page_result = service.get_page_content(ticker="AAPL", document_id="fil_1", page_no=1)
    statement_result = service.get_financial_statement(
        ticker="AAPL",
        document_id="fil_1",
        statement_type="income",
    )

    assert page_result["supported"] is False
    assert "error" in page_result
    assert page_result["error"]["code"] == "not_supported"
    assert "error" in statement_result
    assert statement_result["supported"] is False
    assert statement_result["error"]["code"] == "not_supported"


@pytest.mark.unit
def test_normalize_document_identity_accepts_internal_and_accession_aliases() -> None:
    """验证文档身份归一化支持 internal_document_id 与 accession 别名。"""

    repository = FakeRepository()
    repository._filing_meta["fil_1"]["internal_document_id"] = "0000320193-25-000079"
    repository._filing_meta["fil_1"]["accession_number"] = "0000320193-25-000079"
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    normalized = service._normalize_document_identity(
        ticker="AAPL",
        document_id="000032019325000079",
        tool_name="get_financial_statement",
    )

    assert normalized == ("AAPL", "fil_1")


@pytest.mark.unit
def test_query_xbrl_facts_returns_structured_numeric_payload() -> None:
    """验证 `query_xbrl_facts` 返回结构化数值字段。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.query_xbrl_facts(
        ticker="AAPL",
        document_id="fil_1",
        concepts=["Revenues"],
    )

    assert "error" not in result
    assert result["total"] == 1
    fact = result["facts"][0]
    assert fact["numeric_value"] == 1.0
    assert fact["text_value"] is None
    assert fact["content_type"] is None
    assert "value" not in fact


@pytest.mark.unit
def test_query_xbrl_facts_uses_form_taxonomy_default_concepts_when_omitted() -> None:
    """验证省略 concepts 时会命中 `(form_type, taxonomy)` 默认 concept 包。"""

    repository = FakeRepository()
    processor_registry = TaxonomyAwareProcessorRegistry(taxonomy="us-gaap")
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.query_xbrl_facts(
        ticker="AAPL",
        document_id="fil_1",
    )

    assert processor_registry.last_processor is not None
    expected = [
        "Revenues",
        "NetIncomeLoss",
        "Assets",
        "Liabilities",
        "StockholdersEquity",
        "NetCashProvidedByUsedInOperatingActivities",
    ]
    assert processor_registry.last_processor.last_concepts == expected
    assert "error" not in result
    assert result["query_params"]["concepts"] == expected


@pytest.mark.unit
def test_query_xbrl_facts_falls_back_to_global_default_concepts() -> None:
    """验证 taxonomy 不可识别时会回退全局默认概念包。"""

    repository = FakeRepository()
    processor_registry = TaxonomyAwareProcessorRegistry(taxonomy="unknown")
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.query_xbrl_facts(
        ticker="AAPL",
        document_id="fil_1",
        concepts=[],
    )

    expected = ["Revenues", "NetIncomeLoss", "Assets"]
    assert processor_registry.last_processor is not None
    assert processor_registry.last_processor.last_concepts == expected
    assert "error" not in result
    assert result["query_params"]["concepts"] == expected


@pytest.mark.unit
def test_query_xbrl_facts_deduplicates_and_sanitizes_text_values() -> None:
    """验证 XBRL facts 会按优先级去重并清洗 XHTML 文本。"""

    repository = FakeRepository()
    processor_registry = TaxonomyAwareProcessorRegistry(taxonomy="us-gaap")
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.query_xbrl_facts(
        ticker="AAPL",
        document_id="fil_1",
        concepts=["Revenues", "Assets"],
    )

    assert "error" not in result
    assert result["total"] == 2
    revenues = next(item for item in result["facts"] if item["concept"] == "us-gaap:Revenues")
    assets = next(item for item in result["facts"] if item["concept"] == "us-gaap:Assets")
    assert revenues["numeric_value"] == 100.0
    assert revenues["text_value"] is None
    assert assets["numeric_value"] is None
    assert assets["text_value"] == "Assets line"
    assert assets["content_type"] == "xhtml"


@pytest.mark.unit
def test_query_xbrl_facts_prefers_non_empty_fiscal_period_for_same_period_end() -> None:
    """验证同一期末重复 facts 会优先保留 fiscal_period 非空记录。"""

    repository = FakeRepository()
    processor_registry = FiscalPeriodPreferenceProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.query_xbrl_facts(
        ticker="AAPL",
        document_id="fil_1",
        concepts=["NetIncomeLoss"],
    )

    assert "error" not in result
    assert result["total"] == 1
    fact = result["facts"][0]
    assert fact["concept"] == "us-gaap:NetIncomeLoss"
    assert fact["fiscal_period"] == "Q2"
    assert fact["statement_type"] == "IncomeStatement"


@pytest.mark.unit
def test_list_tables_contract_without_has_page_info() -> None:
    """验证 list_tables 新契约：无 has_page_info，section_ref 替换为 within_section{ref, title}。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_tables(ticker="AAPL", document_id="fil_1")

    assert result["total"] == 1
    assert result["financial_count"] == 1
    assert "has_page_info" not in result
    table = result["tables"][0]
    assert table["table_type"] == "financial"
    # within_section 替代了旧 section_ref 字段
    assert "section_ref" not in table
    assert table["within_section"]["ref"] == "s_0001"
    assert isinstance(table["within_section"]["title"], str)


@pytest.mark.unit
def test_list_tables_applies_filters_and_omits_empty_optional_fields() -> None:
    """验证 list_tables 会应用过滤条件，并省略空的可选字段。"""

    @dataclass
    class FilteredTableProcessor(BasicProcessor):
        """提供多种表格形态以覆盖过滤分支。"""

        def list_tables(self) -> list[dict[str, Any]]:
            """返回用于过滤测试的表格摘要。"""

            return [
                {
                    "table_ref": "tbl_skip_non_financial",
                    "row_count": 1,
                    "col_count": 1,
                    "is_financial": False,
                    "table_type": "operational",
                    "headers": ["skip"],
                    "section_ref": "sec_target",
                    "page_no": 1,
                },
                {
                    "table_ref": "tbl_skip_other_section",
                    "row_count": 1,
                    "col_count": 1,
                    "is_financial": True,
                    "table_type": "financial",
                    "headers": ["skip"],
                    "section_ref": "sec_other",
                    "page_no": 2,
                },
                {
                    "table_ref": "tbl_keep",
                    "row_count": 2,
                    "col_count": 3,
                    "is_financial": True,
                    "table_type": None,
                    "headers": "not_a_list",
                    "section_ref": "sec_target",
                    "page_no": 0,
                },
            ]

        def get_section_title(self, ref: str) -> Optional[str]:
            """仅返回空标题，覆盖 within_section 无 title 分支。"""

            del ref
            return None

    class FilteredTableRegistry:
        """返回固定 FilteredTableProcessor 的注册表桩。"""

        def create(self, source: Source, **kwargs: Any) -> Any:
            """创建处理器。"""

            del source, kwargs
            return FilteredTableProcessor(document_id="fil_1")

        def create_with_fallback(self, source: Source, **kwargs: Any) -> Any:
            """兼容统一回退接口并复用 create。"""

            return self.create(source, **kwargs)

    service = FinsToolService(repository=FakeRepository(), processor_registry=FilteredTableRegistry())

    result = service.list_tables(
        ticker="AAPL",
        document_id="fil_1",
        financial_only=True,
        within_section_ref="sec_target",
    )

    assert result["total"] == 1
    table = result["tables"][0]
    assert table["table_ref"] == "tbl_keep"
    assert table["headers"] is None
    assert table["within_section"] == {"ref": "sec_target"}
    assert "caption" not in table
    assert "page_no" not in table


@pytest.mark.unit
def test_search_document_keeps_processor_schema_without_raw_fields() -> None:
    """验证 search_document 仅直通 Processor 输出，不增加 raw 统计字段。"""

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        query="repurchase",
    )

    assert result["total_matches"] == 2
    assert len(result["matches"]) == 2
    assert "next_section_to_read" in result
    assert result["next_section_to_read"] is not None
    assert result["next_section_to_read"]["section"]["ref"] == "s_0001"
    assert result["next_section_to_read"]["evidence_hit_count"] == 2
    assert "matched_queries" not in result["next_section_to_read"]
    assert "next_section_by_query" not in result
    assert "raw_match_count" not in result
    assert all("repurchase" in str(item["evidence"]["context"]).lower() for item in result["matches"])
    # 默认 searched_in 应为英文 "full text"，不含中文
    assert result["searched_in"] == "full text"
    assert "diagnostics" in result
    diagnostics = result["diagnostics"]
    assert diagnostics["used_expansion"] is False
    assert diagnostics["expansion_query_count"] == 0
    assert diagnostics["strategy_hit_counts"]["exact"] == 2
    assert diagnostics["strategy_hit_counts"]["synonym"] == 0


@pytest.mark.unit
def test_search_document_falls_back_to_synonym_query_when_exact_miss() -> None:
    """验证 search_document 在精确无命中时会触发同义词回退。"""

    repository = FakeRepository()
    processor_registry = SynonymFallbackProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        query="buyback",
    )

    assert result["total_matches"] == 1
    assert "share repurchase" in str(result["matches"][0]["evidence"]["context"]).lower()
    assert "diagnostics" in result
    diagnostics = result["diagnostics"]
    assert diagnostics["used_expansion"] is True
    assert diagnostics["expansion_query_count"] > 0
    assert diagnostics["strategy_hit_counts"]["exact"] == 0
    assert diagnostics["strategy_hit_counts"]["synonym"] == 1


@pytest.mark.unit
def test_search_document_prefers_title_and_item_matches_within_exact_bucket() -> None:
    """验证 BM25F 在 exact 桶内优先排序标题和 Item 更匹配的章节。"""

    repository = FakeRepository()
    processor_registry = Bm25fOrderingProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        query="share repurchase",
    )

    assert result["matches"][0]["section"]["ref"] == "s_0001"
    assert result["matches"][1]["section"]["ref"] == "s_0002"
    assert "diagnostics" in result
    assert result["diagnostics"]["ranking_version"] == "adaptive_bm25f_v1.0.0"


# ---- search_document 批量查询测试 ----


@pytest.mark.unit
def test_search_document_multi_queries_aggregates_results() -> None:
    """验证 queries 批量查询聚合去重排序。"""

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        queries=["repurchase", "board"],
    )

    # 多查询路径：query=None, queries 非空
    assert result["query"] is None
    assert "queries" in result
    assert result["queries"] == ["repurchase", "board"]
    assert result["total_matches"] > 0
    # diagnostics 包含 per_query_stats
    assert "diagnostics" in result
    diag = result["diagnostics"]
    assert "per_query_stats" in diag
    assert diag["query_count"] == 2
    assert len(diag["per_query_stats"]) == 2
    assert "next_section_to_read" not in result
    assert "next_section_by_query" in result
    rep_next = result["next_section_by_query"]["repurchase"]
    board_next = result["next_section_by_query"]["board"]
    assert rep_next is not None
    assert rep_next["section"]["ref"] == "s_0001"
    assert board_next is not None
    assert board_next["section"]["ref"] == "s_0001"
    assert rep_next["evidence_hit_count"] == 2
    assert "matched_queries" not in rep_next


@pytest.mark.unit
def test_search_document_single_query_unchanged() -> None:
    """验证 单 query 路径行为不变（回归）。"""

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        query="repurchase",
    )

    assert result["query"] == "repurchase"
    assert "queries" not in result
    assert result["total_matches"] == 2
    assert "diagnostics" in result
    assert "input_query" in result["diagnostics"]


@pytest.mark.unit
def test_search_document_rejects_both_query_and_queries() -> None:
    """验证同时传 query 和 queries 时报错。"""

    from dayu.engine.exceptions import ToolArgumentError

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    with pytest.raises(ToolArgumentError, match="Cannot specify both"):
        service.search_document(
            ticker="AAPL",
            document_id="fil_1",
            query="repurchase",
            queries=["board"],
        )


@pytest.mark.unit
def test_search_document_rejects_neither_query_nor_queries() -> None:
    """验证不传 query 也不传 queries 时报错。"""

    from dayu.engine.exceptions import ToolArgumentError

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    with pytest.raises(ToolArgumentError, match="Must specify either"):
        service.search_document(
            ticker="AAPL",
            document_id="fil_1",
        )


@pytest.mark.unit
def test_search_document_queries_deduplicates_input() -> None:
    """验证 queries 自动去重。"""

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        queries=["repurchase", "Repurchase", "REPURCHASE"],
    )

    # 三个重复查询去重后只有 1 个，走单查询路径
    assert result["query"] == "repurchase"
    assert "queries" not in result


@pytest.mark.unit
def test_search_document_queries_accepts_twenty_items() -> None:
    """验证 queries 在 20 条上限内可正常执行。"""

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    queries = [f"q{i}" for i in range(20)]
    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        queries=queries,
    )

    assert result["query"] is None
    assert "queries" in result
    assert result["queries"] == queries
    assert "diagnostics" in result
    assert result["diagnostics"]["query_count"] == 20


@pytest.mark.unit
def test_search_document_queries_rejects_more_than_twenty_items() -> None:
    """验证 queries 去重后超过 20 条时返回参数错误。"""

    from dayu.engine.exceptions import ToolArgumentError

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    with pytest.raises(ToolArgumentError, match="exceeds maximum of 20"):
        service.search_document(
            ticker="AAPL",
            document_id="fil_1",
            queries=[f"q{i}" for i in range(21)],
        )


@pytest.mark.unit
def test_search_document_queries_deduplicates_before_limit_check() -> None:
    """验证 queries 会先去重，再按 20 条上限校验。"""

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    unique_queries = [f"q{i}" for i in range(20)]
    duplicated_queries = unique_queries + ["Q0", "q1", "Q2", "q3", "Q4"]
    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        queries=duplicated_queries,
    )

    assert "queries" in result
    assert result["queries"] == unique_queries
    assert "diagnostics" in result
    assert result["diagnostics"]["query_count"] == 20


# ---- 同义词扩展测试 ----


@pytest.mark.unit
def test_synonym_expansion_covers_competition_group() -> None:
    """验证 competitor 同义词组覆盖主要竞争相关词。"""

    from dayu.fins.tools.search_engine import _build_synonym_queries

    synonyms = _build_synonym_queries("competitor")
    synonyms_lower = {s.lower() for s in synonyms}
    assert "competitive" in synonyms_lower or "competition" in synonyms_lower
    assert "compete" in synonyms_lower


@pytest.mark.unit
def test_synonym_expansion_covers_acquisition_group() -> None:
    """验证 acquisition 同义词组覆盖并购相关词。"""

    from dayu.fins.tools.search_engine import _build_synonym_queries

    synonyms = _build_synonym_queries("acquisition")
    synonyms_lower = {s.lower() for s in synonyms}
    assert "merger" in synonyms_lower
    assert "takeover" in synonyms_lower


# ---- 邻近度排序测试 ----


@pytest.mark.unit
def test_proximity_score_ranks_closer_keywords_higher() -> None:
    """验证多词匹配中关键词邻近的条目排在远距条目之前。"""

    from dayu.fins.tools.search_engine import (
        _compute_keyword_proximity_score,
        _sort_ranked_search_entries,
    )

    # 条目 A: "market share" 中 market 和 share 相邻（距离 1）
    entry_close = {
        "_priority": 0,
        "_strategy": "exact",
        "section_ref": "s_0001",
        "page_no": 1,
        "evidence": {
            "matched_text": "market share",
            "context": "The company gained significant market share in 2025.",
        },
    }
    # 条目 B: market 和 share 距离很远
    entry_far = {
        "_priority": 0,
        "_strategy": "exact",
        "section_ref": "s_0002",
        "page_no": 1,
        "evidence": {
            "matched_text": "market share",
            "context": "The market for semiconductors continued to expand and the outstanding share count increased.",
        },
    }

    score_close = _compute_keyword_proximity_score(entry_close)
    score_far = _compute_keyword_proximity_score(entry_far)
    # 邻近的分数应更低
    assert score_close < score_far

    # 排序验证：close 条目排在 far 之前
    sorted_entries = _sort_ranked_search_entries([entry_far, entry_close])
    assert sorted_entries[0]["section_ref"] == "s_0001"


@pytest.mark.unit
def test_proximity_score_single_word_returns_zero() -> None:
    """验证单词查询邻近度分数为 0。"""

    from dayu.fins.tools.search_engine import _compute_keyword_proximity_score

    entry = {
        "evidence": {
            "matched_text": "revenue",
            "context": "Total revenue grew by 15% year over year.",
        },
    }
    assert _compute_keyword_proximity_score(entry) == 0


@pytest.mark.unit
def test_sort_ranked_search_entries_keeps_strategy_bucket_ahead_of_bm25f_score() -> None:
    """验证 strategy bucket 优先于 BM25F 分数，避免 exact 被非 exact 覆盖。"""

    from dayu.fins.tools.bm25f_scorer import build_section_bm25f_index
    from dayu.fins.tools.search_engine import _sort_ranked_search_entries

    bm25f_index = build_section_bm25f_index([
        {
            "ref": "s_0001",
            "title": "Item 5. Share Repurchase Program",
            "item": "Item 5",
            "topic": "market_for_equity",
            "path": "10-K > Item 5 > Share Repurchase Program",
            "preview": "Share repurchase share repurchase share repurchase.",
        },
        {
            "ref": "s_0002",
            "title": "General Corporate Overview",
            "item": None,
            "topic": None,
            "path": None,
            "preview": "General overview.",
        },
    ])
    synonym_high_score = {
        "_priority": 2,
        "_strategy": "synonym",
        "_query": "share repurchase",
        "section_ref": "s_0001",
        "page_no": 1,
        "evidence": {
            "matched_text": "share repurchase",
            "context": "share repurchase share repurchase share repurchase share repurchase",
        },
    }
    exact_lower_score = {
        "_priority": 0,
        "_strategy": "exact",
        "_query": "share repurchase",
        "section_ref": "s_0002",
        "page_no": 1,
        "evidence": {
            "matched_text": "share repurchase",
            "context": "The company discussed repurchase once.",
        },
    }

    sorted_entries = _sort_ranked_search_entries(
        [synonym_high_score, exact_lower_score],
        bm25f_index=bm25f_index,
    )

    assert sorted_entries[0]["_strategy"] == "exact"
    assert sorted_entries[0]["section_ref"] == "s_0002"


@pytest.mark.unit
def test_intent_alignment_prefers_expected_semantic_bucket() -> None:
    """验证排序在同优先级下优先选择意图一致章节。"""

    from dayu.fins.tools.search_engine import _sort_ranked_search_entries
    from dayu.fins.tools.search_models import QueryDiagnosis, SectionSemanticProfile

    diagnosis = QueryDiagnosis(
        query="competitor",
        tokens=("competitor",),
        token_count=1,
        ambiguity_score=0.8,
        is_high_ambiguity=True,
        intent="business_competition",
        allow_direct_token_fallback=False,
    )
    semantic_profiles = {
        "sec_comp": SectionSemanticProfile(
            section_ref="sec_comp",
            title="Item 1. Competition",
            item="Item 1",
            topic="business",
            path="10-K > Item 1 > Competition",
            bucket="business",
            lexical_tokens=("competitor",),
        ),
        "sec_noise": SectionSemanticProfile(
            section_ref="sec_noise",
            title="Item 6. Human Capital",
            item="Item 6",
            topic="directors_employees",
            path="20-F > Item 6 > Human Capital",
            bucket="people",
            lexical_tokens=("employee",),
        ),
    }

    entries = [
        {
            "_priority": 2,
            "_query": "competitor",
            "section_ref": "sec_noise",
            "page_no": 1,
            "evidence": {"matched_text": "competitor", "context": "employer brand and talent competition"},
        },
        {
            "_priority": 2,
            "_query": "competitor",
            "section_ref": "sec_comp",
            "page_no": 1,
            "evidence": {"matched_text": "competitor", "context": "principal competitors in cloud software markets"},
        },
    ]

    sorted_entries = _sort_ranked_search_entries(
        entries,
        diagnosis=diagnosis,
        semantic_profiles=semantic_profiles,
    )

    assert sorted_entries[0]["section_ref"] == "sec_comp"


@pytest.mark.unit
def test_context_noise_penalty_demotes_hr_competition_context() -> None:
    """验证竞争查询下 HR 语境噪音会被惩罚并后置。"""

    from dayu.fins.tools.search_engine import _sort_ranked_search_entries
    from dayu.fins.tools.search_models import QueryDiagnosis, SectionSemanticProfile

    diagnosis = QueryDiagnosis(
        query="competition",
        tokens=("competition",),
        token_count=1,
        ambiguity_score=0.9,
        is_high_ambiguity=True,
        intent="business_competition",
        allow_direct_token_fallback=False,
    )
    semantic_profiles = {
        "sec_a": SectionSemanticProfile(
            section_ref="sec_a",
            title="Item 1. Competition Overview",
            item="Item 1",
            topic="business",
            path="10-K > Item 1 > Competition Overview",
            bucket="business",
            lexical_tokens=("competition", "competitor", "market"),
        ),
        "sec_b": SectionSemanticProfile(
            section_ref="sec_b",
            title="Item 1. Human Capital",
            item="Item 1",
            topic="business",
            path="10-K > Item 1 > Human Capital",
            bucket="business",
            lexical_tokens=("competition", "employee", "hiring"),
        ),
    }

    entries = [
        {
            "_priority": 2,
            "_query": "competition",
            "section_ref": "sec_b",
            "page_no": 1,
            "evidence": {
                "matched_text": "competition",
                "context": "we participate in employer branding and universum students league competition activities",
            },
        },
        {
            "_priority": 2,
            "_query": "competition",
            "section_ref": "sec_a",
            "page_no": 1,
            "evidence": {
                "matched_text": "competition",
                "context": "the company competes in highly competitive market segments",
            },
        },
    ]

    sorted_entries = _sort_ranked_search_entries(
        entries,
        diagnosis=diagnosis,
        semantic_profiles=semantic_profiles,
    )

    assert sorted_entries[0]["section_ref"] == "sec_a"
    assert float(sorted_entries[0]["_context_noise_penalty"]) < float(
        sorted_entries[1]["_context_noise_penalty"]
    )


@pytest.mark.unit
class TestResolveSemanticBucket:
    """验证 _resolve_semantic_bucket 自适应两级策略。"""

    def test_topic_direct_mapping_covers_all_section_types(self) -> None:
        """一级策略：已知 topic 直接映射，无需关键词评分。"""

        from dayu.fins.tools.search_engine import _resolve_semantic_bucket

        cases = [
            ("business", "business"),
            ("company_information", "business"),
            ("operating_review", "business"),
            ("risk_factors", "risk"),
            ("market_risk", "risk"),
            ("cybersecurity", "risk"),
            ("mda", "financial"),
            ("financial_statements", "financial"),
            ("market_for_equity", "financial"),
            ("directors", "governance"),
            ("executive_compensation", "governance"),
            ("controls_procedures", "governance"),
            ("directors_employees", "people"),
            ("legal_proceedings", "legal"),
            ("exhibits", "other"),
            ("signature", "other"),
        ]
        for topic, expected_bucket in cases:
            result = _resolve_semantic_bucket(topic=topic, path="", title="", item="")
            assert result == expected_bucket, f"topic={topic}: got {result}, expected {expected_bucket}"

    def test_keyword_fallback_when_topic_unknown(self) -> None:
        """二级策略：topic 未知时基于 title/path 关键词评分。"""

        from dayu.fins.tools.search_engine import _resolve_semantic_bucket

        assert _resolve_semantic_bucket(
            topic="", path="", title="Business Overview", item=""
        ) == "business"
        assert _resolve_semantic_bucket(
            topic="", path="", title="Risk Factors and Uncertainty", item=""
        ) == "risk"
        assert _resolve_semantic_bucket(
            topic="", path="", title="Employee Workforce Summary", item=""
        ) == "people"

    def test_fallback_returns_other_when_no_signal(self) -> None:
        """无任何信号时返回 other。"""

        from dayu.fins.tools.search_engine import _resolve_semantic_bucket

        assert _resolve_semantic_bucket(topic="", path="", title="", item="") == "other"
        assert _resolve_semantic_bucket(
            topic="unknown_custom_section", path="", title="Misc", item=""
        ) == "other"

    def test_directors_employees_is_people_not_governance(self) -> None:
        """directors_employees 必须归入 people 而非 governance。"""

        from dayu.fins.tools.search_engine import _resolve_semantic_bucket

        assert _resolve_semantic_bucket(
            topic="directors_employees", path="20-F > Item 6", title="Directors and Employees", item="Item 6"
        ) == "people"

    def test_keyword_scoring_prefers_highest_overlap(self) -> None:
        """多关键词命中时取交集最大的桶。"""

        from dayu.fins.tools.search_engine import _resolve_semantic_bucket

        # title 含 "financial" + "income" + "assets" → financial(3) > 其他
        assert _resolve_semantic_bucket(
            topic="", path="", title="Financial Income and Assets Report", item=""
        ) == "financial"


@pytest.mark.unit
def test_get_table_returns_self_descriptive_records_payload() -> None:
    """验证 `get_table` 在 records 场景返回自解释 `data` 结构。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.get_table(
        ticker="AAPL",
        document_id="fil_1",
        table_ref="t_0001",
    )

    assert "data_schema_version" not in result
    assert "data_kind" not in result
    assert "data_format" not in result
    assert "columns" not in result
    assert result["data"]["kind"] == "records"
    assert result["data"]["columns"] == ["A", "B"]
    assert result["data"]["rows"] == [{"A": 1, "B": 2}]


@pytest.mark.unit
def test_get_table_demotes_non_markdown_text_to_raw_text_payload() -> None:
    """验证 `get_table` 对伪 markdown 文本降级为 `raw_text` 结构。"""

    repository = FakeRepository()
    processor_registry = RawTextProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.get_table(
        ticker="AAPL",
        document_id="fil_1",
        table_ref="t_0001",
    )

    assert "data_kind" not in result
    assert result["data"]["kind"] == "raw_text"
    assert "GENERAL" in result["data"]["text"]


@pytest.mark.unit
def test_get_table_returns_within_section_and_table_type() -> None:
    """验证 get_table 返回 within_section{ref, title}、table_type，caption 条件化。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.get_table(
        ticker="AAPL",
        document_id="fil_1",
        table_ref="t_0001",
    )

    # within_section 替代了旧 caption 始终输出
    assert "section_ref" not in result
    assert "within_section" in result
    assert result["within_section"]["ref"] == "s_0001"
    assert isinstance(result["within_section"]["title"], str)
    # table_type 新增
    assert result["table_type"] is not None
    # caption 条件输出（FakeProcessor caption="表1" 非空，应存在）
    assert "caption" in result
    assert result["caption"] == "表1"
    # page_no 条件输出（FakeProcessor page_no=2，应存在）
    assert "page_no" in result
    assert result["page_no"] == 2


@pytest.mark.unit
def test_service_init_rejects_invalid_cache_size() -> None:
    """验证 service 初始化时拒绝非法缓存容量。"""

    import pytest

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()

    with pytest.raises(ValueError, match="processor_cache_max_entries must be greater than 0"):
        FinsToolService(
            repository=repository,
            processor_registry=processor_registry,
            processor_cache_max_entries=0,
        )

    with pytest.raises(ValueError, match="processor_cache_max_entries must be greater than 0"):
        FinsToolService(
            repository=repository,
            processor_registry=processor_registry,
            processor_cache_max_entries=-1,
        )


@pytest.mark.unit
def test_list_documents_filters_by_document_type() -> None:
    """验证 list_documents 按 document_types 过滤。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    # 只返回年报
    result = service.list_documents(ticker="AAPL", document_types=["annual_report"])

    assert result["filters"]["document_types"] == ["annual_report"]
    assert result["matched"] <= result["total"]
    # 如果有多个不同类型的文档，matched 应该少于 total
    if result["total"] > 1:
        assert all(doc.get("document_type") == "annual_report" for doc in result["documents"])


@pytest.mark.unit
def test_list_documents_filters_by_fiscal_year() -> None:
    """验证 list_documents 按 fiscal_year 过滤。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(ticker="AAPL", fiscal_years=[2023])

    assert result["filters"]["fiscal_years"] == [2023]
    if result["documents"]:
        assert all(
            doc.get("fiscal_year") == 2023 for doc in result["documents"] if doc.get("fiscal_year")
        )


@pytest.mark.unit
def test_list_documents_filters_by_fiscal_period() -> None:
    """验证 list_documents 按 fiscal_period 过滤。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(ticker="AAPL", fiscal_periods=["FY"])

    assert result["filters"]["fiscal_periods"] == ["FY"]
    if result["documents"]:
        assert all(doc.get("fiscal_period") in ["FY", None] for doc in result["documents"])


@pytest.mark.unit
def test_list_documents_only_keeps_intrinsic_fiscal_period_fallback_when_meta_is_null() -> None:
    """验证 list_documents 仅保留不依赖日期猜测的 FY 回退，不再回填 fiscal_year。"""

    repository = FakeRepository()
    repository._filing_meta["fil_1"]["fiscal_year"] = None
    repository._filing_meta["fil_1"]["fiscal_period"] = None
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(
        ticker="AAPL",
        document_types=["annual_report"],
        fiscal_periods=["FY"],
    )

    doc_by_id = {str(item.get("document_id")): item for item in result["documents"]}
    assert "fil_1" in doc_by_id
    assert doc_by_id["fil_1"]["fiscal_year"] is None
    assert doc_by_id["fil_1"]["fiscal_period"] == "FY"


@pytest.mark.unit
def test_list_documents_does_not_infer_6k_fiscal_year_from_report_date() -> None:
    """验证 list_documents 不会把 6-K 的 report_date 回填为 fiscal_year。"""

    class SixKRepository(FakeRepository):
        """用于覆盖 6-K fiscal_year 消费侧回退的仓储桩。"""

        def __init__(self) -> None:
            """初始化仓储桩。"""

            super().__init__()
            self._filing_meta["fil_6k"] = {
                "document_id": "fil_6k",
                "form_type": "6-K",
                "fiscal_year": None,
                "fiscal_period": None,
                "report_date": "2026-03-12",
                "filing_date": "2026-03-12",
                "amended": False,
                "is_deleted": False,
                "ingest_complete": True,
            }

    repository = SixKRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    filtered = service.list_documents(
        ticker="AAPL",
        document_types=["quarterly_report"],
        fiscal_years=[2026],
    )
    full = service.list_documents(ticker="AAPL")

    assert filtered["matched"] == 0
    doc_by_id = {str(item.get("document_id")): item for item in full["documents"]}
    assert doc_by_id["fil_6k"]["fiscal_year"] is None
    assert doc_by_id["fil_6k"]["fiscal_period"] is None


@pytest.mark.unit
def test_list_documents_does_not_infer_fiscal_fields_from_report_date() -> None:
    """验证 list_documents 不会再用 report_date 伪造 10-Q/10-K 的 fiscal 字段。"""

    class MissingFiscalRepository(FakeRepository):
        """用于覆盖 source fiscal 回退边界的仓储桩。"""

        def __init__(self) -> None:
            """初始化仓储桩。"""

            super().__init__()
            self._filing_meta["fil_q_missing"] = {
                "document_id": "fil_q_missing",
                "form_type": "10-Q",
                "fiscal_year": None,
                "fiscal_period": None,
                "report_date": "2025-03-29",
                "filing_date": "2025-05-02",
                "amended": False,
                "is_deleted": False,
                "ingest_complete": True,
            }
            self._filing_meta["fil_k_missing"] = {
                "document_id": "fil_k_missing",
                "form_type": "10-K",
                "fiscal_year": None,
                "fiscal_period": None,
                "report_date": "2025-09-28",
                "filing_date": "2025-11-01",
                "amended": False,
                "is_deleted": False,
                "ingest_complete": True,
            }

    repository = MissingFiscalRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    filtered_quarter = service.list_documents(
        ticker="AAPL",
        document_types=["quarterly_report"],
        fiscal_years=[2025],
        fiscal_periods=["Q1"],
    )
    filtered_annual_year = service.list_documents(
        ticker="AAPL",
        document_types=["annual_report"],
        fiscal_years=[2025],
    )
    filtered_annual_period = service.list_documents(
        ticker="AAPL",
        document_types=["annual_report"],
        fiscal_periods=["FY"],
    )
    full = service.list_documents(ticker="AAPL")

    doc_by_id = {str(item.get("document_id")): item for item in full["documents"]}
    assert filtered_quarter["matched"] == 1
    assert {str(item.get("document_id")) for item in filtered_quarter["documents"]} == {"fil_2"}
    assert filtered_annual_year["matched"] == 0
    assert filtered_annual_period["matched"] == 2
    assert {str(item.get("document_id")) for item in filtered_annual_period["documents"]} == {"fil_1", "fil_k_missing"}
    assert doc_by_id["fil_q_missing"]["fiscal_year"] is None
    assert doc_by_id["fil_q_missing"]["fiscal_period"] is None
    assert doc_by_id["fil_k_missing"]["fiscal_year"] is None
    assert doc_by_id["fil_k_missing"]["fiscal_period"] == "FY"


# ---- list_documents fallback 兜底测试 ----


@pytest.mark.unit
def test_list_documents_fallback_when_document_type_has_no_match() -> None:
    """验证 document_types 无命中时返回 match_status=no_match 和 suggestion。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    # FakeRepository 只有 annual_report / quarterly_report / material，搜 earnings_call 必定无命中
    result = service.list_documents(ticker="AAPL", document_types=["earnings_call"])

    assert result["matched"] == 0
    assert result["documents"] == []
    # 新契约：match_status + suggestion
    assert result["match_status"] == "no_match"
    assert "suggestion" in result
    suggestion = result["suggestion"]
    assert suggestion["action"] == "broaden_filter"
    assert suggestion["reason"] == "no_documents_matched_document_types"
    assert isinstance(suggestion["available_document_types"], list)
    assert len(suggestion["available_document_types"]) > 0
    # 旧 fallback 字段不再存在
    assert "fallback_results" not in result
    assert "fallback_note" not in result
    assert "fallback_recommended_documents" not in result


@pytest.mark.unit
def test_list_documents_no_fallback_when_document_type_has_match() -> None:
    """验证 document_types 有命中时 match_status=ok 且无 suggestion。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(ticker="AAPL", document_types=["annual_report"])

    assert result["matched"] > 0
    assert result["match_status"] == "ok"
    assert "suggestion" not in result


@pytest.mark.unit
def test_list_documents_no_fallback_when_no_form_type_specified() -> None:
    """验证不传 form_types 时 match_status=ok。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(ticker="AAPL")

    assert result["matched"] == result["total"]
    assert result["match_status"] == "ok"
    assert "suggestion" not in result


@pytest.mark.unit
def test_list_documents_fallback_respects_fiscal_year_filter() -> None:
    """验证 no_match 时 suggestion.available_document_types 仅来自全量文档。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    # 搜 earnings_call + fiscal_year=2024，必定无命中
    result = service.list_documents(ticker="AAPL", document_types=["earnings_call"], fiscal_years=[2024])

    assert result["matched"] == 0
    assert result["match_status"] == "no_match"
    assert "suggestion" in result
    suggestion = result["suggestion"]
    assert "annual_report" in suggestion["available_document_types"]


@pytest.mark.unit
def test_list_documents_document_types_multi_filter() -> None:
    """验证 document_types 同时传多个值的过滤行为。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(ticker="AAPL", document_types=["annual_report", "quarterly_report"])

    assert result["matched"] == 2
    doc_types_in_result = {doc["document_type"] for doc in result["documents"]}
    assert doc_types_in_result == {"annual_report", "quarterly_report"}
    assert result["match_status"] == "ok"


@pytest.mark.unit
def test_list_documents_document_type_field() -> None:
    """验证每个文档都包含正确的 document_type 字段。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(ticker="AAPL")

    doc_types = {doc["document_id"]: doc["document_type"] for doc in result["documents"]}
    assert doc_types["fil_1"] == "annual_report"
    assert doc_types["fil_2"] == "quarterly_report"
    assert doc_types["mat_1"] == "material"


@pytest.mark.unit
def test_list_documents_maps_cn_fiscal_period_form_type_to_semantic_document_types() -> None:
    """验证港股/A 股上传产生的 FY/H1/Q* form_type 会映射到稳定 document_type。"""

    class CnFiscalRepository(FakeRepository):
        """用于覆盖 CN/HK fiscal_period form_type 语义映射的仓储桩。"""

        def __init__(self) -> None:
            """初始化仓储桩。"""

            super().__init__()
            self._filing_meta = {
                "fil_fy": {
                    "document_id": "fil_fy",
                    "form_type": "FY",
                    "fiscal_year": 2025,
                    "fiscal_period": "FY",
                    "report_date": None,
                    "filing_date": None,
                    "amended": False,
                    "is_deleted": False,
                    "ingest_complete": True,
                },
                "fil_h1": {
                    "document_id": "fil_h1",
                    "form_type": "H1",
                    "fiscal_year": 2025,
                    "fiscal_period": "H1",
                    "report_date": None,
                    "filing_date": None,
                    "amended": False,
                    "is_deleted": False,
                    "ingest_complete": True,
                },
                "fil_q2": {
                    "document_id": "fil_q2",
                    "form_type": "Q2",
                    "fiscal_year": 2024,
                    "fiscal_period": "Q2",
                    "report_date": None,
                    "filing_date": None,
                    "amended": False,
                    "is_deleted": False,
                    "ingest_complete": True,
                },
            }
            self._material_meta = {}

        def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
            """返回仓储中已存在的港股 ticker。"""

            if "9992" in candidates:
                return "9992"
            return None

        def get_company_meta(self, ticker: str) -> CompanyMeta:
            """返回港股公司信息。"""

            return CompanyMeta(
                company_id="01009992",
                company_name="泡泡玛特",
                ticker=ticker,
                market="HK",
                resolver_version="test",
                updated_at="2026-01-01T00:00:00+00:00",
            )

    service = FinsToolService(
        repository=CnFiscalRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    result = service.list_documents(ticker="9992")

    doc_types = {doc["document_id"]: doc["document_type"] for doc in result["documents"]}
    assert doc_types["fil_fy"] == "annual_report"
    assert doc_types["fil_h1"] == "semi_annual_report"
    assert doc_types["fil_q2"] == "quarterly_report"
    recommended = result["recommended_documents"]
    assert recommended["latest_annual_report_document_id"] == "fil_fy"
    assert recommended["latest_quarterly_report_document_id"] == "fil_h1"
    assert recommended["recommended_for_company_overview_document_id"] == "fil_fy"


@pytest.mark.unit
def test_list_documents_recency_falls_back_to_fiscal_year_and_period_when_dates_missing() -> None:
    """验证缺失日期时按 fiscal_year/fiscal_period 回退排序，而非让无日期 material 抢到最前。"""

    class MissingDateRecencyRepository(FakeRepository):
        """用于覆盖 list_documents 时间排序回退分支的仓储桩。"""

        def __init__(self) -> None:
            """初始化仓储桩。"""

            super().__init__()
            self._filing_meta = {
                "fil_fy_2025": {
                    "document_id": "fil_fy_2025",
                    "form_type": "FY",
                    "fiscal_year": 2025,
                    "fiscal_period": "FY",
                    "report_date": None,
                    "filing_date": None,
                    "amended": False,
                    "is_deleted": False,
                    "ingest_complete": True,
                },
                "fil_h1_2025": {
                    "document_id": "fil_h1_2025",
                    "form_type": "H1",
                    "fiscal_year": 2025,
                    "fiscal_period": "H1",
                    "report_date": None,
                    "filing_date": None,
                    "amended": False,
                    "is_deleted": False,
                    "ingest_complete": True,
                },
                "fil_fy_2024": {
                    "document_id": "fil_fy_2024",
                    "form_type": "FY",
                    "fiscal_year": 2024,
                    "fiscal_period": "FY",
                    "report_date": None,
                    "filing_date": None,
                    "amended": False,
                    "is_deleted": False,
                    "ingest_complete": True,
                },
            }
            self._material_meta = {
                "mat_governance": {
                    "document_id": "mat_governance",
                    "form_type": "CORPORATE_GOVERNANCE",
                    "material_name": "董事会成员名单与其角色和职能",
                    "report_date": None,
                    "filing_date": None,
                    "amended": False,
                    "is_deleted": False,
                    "ingest_complete": True,
                }
            }

        def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
            """返回仓储中已存在的港股 ticker。"""

            if "9992" in candidates:
                return "9992"
            return None

        def get_company_meta(self, ticker: str) -> CompanyMeta:
            """返回港股公司信息。"""

            return CompanyMeta(
                company_id="01009992",
                company_name="泡泡玛特",
                ticker=ticker,
                market="HK",
                resolver_version="test",
                updated_at="2026-01-01T00:00:00+00:00",
            )

    service = FinsToolService(
        repository=MissingDateRecencyRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    result = service.list_documents(ticker="9992")

    ordered_ids = [doc["document_id"] for doc in result["documents"]]
    assert ordered_ids[:4] == ["fil_fy_2025", "fil_h1_2025", "fil_fy_2024", "mat_governance"]
    recommended = result["recommended_documents"]
    assert recommended["latest_document_id"] == "fil_fy_2025"
    assert recommended["latest_annual_report_document_id"] == "fil_fy_2025"
    assert recommended["recommended_for_company_overview_document_id"] == "fil_fy_2025"


@pytest.mark.unit
def test_list_documents_capability_flags() -> None:
    """验证文档包含 has_financial_data 能力标志。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(ticker="AAPL")

    for doc in result["documents"]:
        assert "has_financial_data" in doc
        assert "has_financial_statements" not in doc
        assert "has_xbrl" not in doc
        assert "financial_statement_availability" not in doc
    # filing 应有财务数据
    fil_docs = [d for d in result["documents"] if d["document_id"].startswith("fil_")]
    assert all(d["has_financial_data"] is True for d in fil_docs)
    # material 无财务数据
    mat_docs = [d for d in result["documents"] if d["document_id"].startswith("mat_")]
    assert all(d["has_financial_data"] is False for d in mat_docs)


@pytest.mark.unit
def test_list_documents_has_financial_data_prefers_null_over_ambiguous() -> None:
    """验证旧布尔字段只能说明“可能有财务内容”时，has_financial_data 保守返回 null。"""

    class AmbiguousCapabilityRepository(FakeRepository):
        """返回旧能力字段但不提供 has_financial_data 的仓储桩。"""

        def get_processed_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
            del ticker
            if document_id == "fil_1":
                return {"has_financial_statement": True, "has_xbrl": False}
            if document_id.startswith("fil_"):
                return {"has_financial_statement": True, "has_xbrl": True}
            return {"has_financial_statement": False, "has_xbrl": False}

    service = FinsToolService(
        repository=AmbiguousCapabilityRepository(),
        processor_registry=FakeProcessorRegistry(),
    )

    result = service.list_documents(ticker="AAPL")
    fil_1 = next(doc for doc in result["documents"] if doc["document_id"] == "fil_1")
    # has_financial_statement=True + has_xbrl=False 且无其它字段 → 无法判定
    assert fil_1["has_financial_data"] is None


@pytest.mark.unit
def test_list_documents_skips_missing_deleted_incomplete_and_uses_fallbacks() -> None:
    """验证 list_documents 会跳过异常文档，并在公司/能力信息缺失时回退。"""

    class SparseRepository(FakeRepository):
        """用于覆盖缺失元数据与回退分支的仓储桩。"""

        def __init__(self) -> None:
            """初始化仓储桩。"""

            super().__init__()
            self._filing_meta["fil_deleted"] = {
                "document_id": "fil_deleted",
                "form_type": "10-K",
                "fiscal_year": 2024,
                "fiscal_period": "FY",
                "report_date": "2024-09-28",
                "filing_date": "2024-11-01",
                "amended": False,
                "is_deleted": True,
                "ingest_complete": True,
            }
            self._filing_meta["fil_incomplete"] = {
                "document_id": "fil_incomplete",
                "form_type": "10-Q",
                "fiscal_year": 2025,
                "fiscal_period": "Q2",
                "report_date": "2025-06-28",
                "filing_date": "2025-08-01",
                "amended": False,
                "is_deleted": False,
                "ingest_complete": False,
            }

        def list_document_ids(self, ticker: str, source_kind: Optional[SourceKind] = None) -> list[str]:
            """返回包含缺失元数据 ID 的文档列表。"""

            document_ids = super().list_document_ids(ticker, source_kind)
            if source_kind in (None, SourceKind.FILING):
                return sorted([*document_ids, "fil_missing"])
            return document_ids

        def get_document_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
            """对指定文档模拟元数据缺失。"""

            if document_id == "fil_missing":
                raise FileNotFoundError(document_id)
            return super().get_document_meta(ticker, document_id)

        def get_company_meta(self, ticker: str) -> CompanyMeta:
            """模拟公司信息缺失。"""

            raise FileNotFoundError(ticker)

        def get_processed_meta(self, ticker: str, document_id: str) -> dict[str, Any]:
            """模拟能力标志缺失。"""

            raise FileNotFoundError(f"{ticker}:{document_id}")

    repository = SparseRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.list_documents(ticker="AAPL")

    assert result["company"]["name"] == "AAPL"
    assert result["company"]["market"] == "unknown"
    document_ids = {item["document_id"] for item in result["documents"]}
    assert "fil_missing" not in document_ids
    assert "fil_deleted" not in document_ids
    assert "fil_incomplete" not in document_ids
    filing = next(item for item in result["documents"] if item["document_id"] == "fil_1")
    # processed meta 缺失时 has_financial_data 应为 None
    assert filing["has_financial_data"] is None


@pytest.mark.unit
def test_list_documents_normalizes_hk_alias_to_existing_canonical_ticker() -> None:
    """验证港股 alias 会归一化到仓储中已存在的 canonical ticker。"""

    class HkAliasRepository(FakeRepository):
        """用于覆盖港股 alias 收敛分支的仓储桩。"""

        def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
            """返回仓储中已存在的港股 canonical ticker。"""

            if "0300" in candidates:
                return "0300"
            return None

        def get_company_meta(self, ticker: str) -> CompanyMeta:
            """返回港股公司信息。"""

            return CompanyMeta(
                company_id="0300",
                company_name="HK Sample",
                ticker=ticker,
                market="HK",
                resolver_version="test",
                updated_at="2026-01-01T00:00:00+00:00",
            )

    service = FinsToolService(repository=HkAliasRepository(), processor_registry=FakeProcessorRegistry())

    result = service.list_documents(ticker="00300.HK")

    assert result["company"]["ticker"] == "0300"
    assert result["company"]["market"] == "HK"


@pytest.mark.unit
def test_list_documents_normalizes_three_digit_hk_alias_to_existing_canonical_ticker() -> None:
    """验证三位港股 `.HK` 写法也会归一化到仓储 canonical ticker。"""

    class HkAliasRepository(FakeRepository):
        """用于覆盖三位港股 alias 收敛分支的仓储桩。"""

        def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
            """返回仓储中已存在的四位港股 canonical ticker。"""

            if "0300" in candidates:
                return "0300"
            return None

        def get_company_meta(self, ticker: str) -> CompanyMeta:
            """返回港股公司信息。"""

            return CompanyMeta(
                company_id="0300",
                company_name="HK Sample",
                ticker=ticker,
                market="HK",
                resolver_version="test",
                updated_at="2026-01-01T00:00:00+00:00",
            )

    service = FinsToolService(repository=HkAliasRepository(), processor_registry=FakeProcessorRegistry())

    result = service.list_documents(ticker="300.HK")

    assert result["company"]["ticker"] == "0300"
    assert result["company"]["market"] == "HK"


@pytest.mark.unit
def test_document_tools_reuse_hk_alias_resolution_across_calls() -> None:
    """验证其余文档读取工具也复用 ticker alias 归一化。"""

    class HkAliasRepository(FakeRepository):
        """返回仓储中实际存在的 4 位港股目录。"""

        def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
            """返回仓储中的 canonical ticker。"""

            if "9992" in candidates:
                return "9992"
            return None

        def get_company_meta(self, ticker: str) -> CompanyMeta:
            """返回港股公司信息。"""

            return CompanyMeta(
                company_id="9992",
                company_name="Pop Mart",
                ticker=ticker,
                market="HK",
                resolver_version="test",
                updated_at="2026-01-01T00:00:00+00:00",
            )

    service = FinsToolService(repository=HkAliasRepository(), processor_registry=FakeProcessorRegistry())

    documents = service.list_documents(ticker="9992.HK")
    sections = service.get_document_sections(ticker="9992.HK", document_id="fil_1")

    assert documents["company"]["ticker"] == "9992"
    assert sections["ticker"] == "9992"


@pytest.mark.unit
def test_list_documents_raises_not_found_for_uningested_ticker() -> None:
    """验证未收录 ticker 走显式 not_found，而非返回空结果。"""

    class MissingTickerRepository(FakeRepository):
        """始终模拟 ticker 未收录。"""

        def resolve_existing_ticker(self, candidates: list[str]) -> Optional[str]:
            """返回未命中。"""

            del candidates
            return None

    service = FinsToolService(repository=MissingTickerRepository(), processor_registry=FakeProcessorRegistry())

    with pytest.raises(ToolBusinessError) as exc_info:
        service.list_documents(ticker="09992.HK")

    assert exc_info.value.code == "not_found"
    assert "Financial Document Tools do not have this company" in exc_info.value.message
    assert "不允许：继续穷举 ticker 变体" in exc_info.value.hint


@pytest.mark.unit
def test_get_page_content_validates_page_no() -> None:
    """验证 get_page_content 拒绝非法 page_no。"""

    import pytest
    from dayu.engine.exceptions import ToolArgumentError

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    # page_no <= 0
    with pytest.raises(ToolArgumentError, match="page_no must be a positive integer"):
        service.get_page_content(ticker="AAPL", document_id="fil_1", page_no=0)

    with pytest.raises(ToolArgumentError, match="page_no must be a positive integer"):
        service.get_page_content(ticker="AAPL", document_id="fil_1", page_no=-1)

    # page_no 不是整数  
    with pytest.raises(ToolArgumentError, match="page_no must be a positive integer"):
        service.get_page_content(ticker="AAPL", document_id="fil_1", page_no="1")  # type: ignore


@pytest.mark.unit
def test_get_page_content_returns_supported_payload() -> None:
    """验证 get_page_content 支持路径会返回标准化页面内容与 citation。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.get_page_content(ticker="AAPL", document_id="fil_1", page_no=2)

    assert "error" not in result
    assert result["page_no"] == 2
    assert result["supported"] is True
    assert result["has_content"] is True
    assert result["text_preview"] == "第2页文本"
    assert result["sections"][0]["ref"] == "s_0001"
    assert result["tables"][0]["table_ref"] == "t_0001"
    assert result["citation"]["document_id"] == "fil_1"


@pytest.mark.unit
def test_read_section_handles_non_list_tables() -> None:
    """验证 read_section 不再输出 tables 字段（content 中 [[t_XXXX]] 占位符已携带 ref + 位置上下文）。"""

    @dataclass
    class NonListTablesProcessor(BasicProcessor):
        """返回非列表 tables 字段的处理器。"""

        def read_section(self, ref: str) -> dict[str, Any]:
            """返回 tables=None 的 section。"""
            return {
                "ref": ref,
                "title": "Section",
                "content": "Content",
                "tables": None,  # 不是列表
                "word_count": None,
                "contains_full_text": True,
            }

    repository = FakeRepository()

    class NonListTablesRegistry:
        """注册表桩。"""

        def create(self, source: Source, **kwargs: Any) -> Any:
            """创建处理器。"""
            del source, kwargs
            return NonListTablesProcessor(document_id="test")

        def create_with_fallback(self, source: Source, **kwargs: Any) -> Any:
            """兼容统一回退接口并复用 create。"""

            return self.create(source, **kwargs)

    processor_registry = NonListTablesRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.read_section(ticker="AAPL", document_id="fil_1", ref="sec_001")

    # tables 字段不再输出给 LLM
    assert "tables" not in result
    # content_word_count 应该从 content 计算
    assert result["content_word_count"] > 0
    # children 非法时应回退为空列表
    assert result["children"] == []


@pytest.mark.unit
def test_read_section_normalizes_children_and_content_word_count() -> None:
    """验证 read_section 返回 children 与 content_word_count 新契约。

    children 只保留 ref + title（level/preview 已去除）；
    path 字段不输出给 LLM。
    """

    @dataclass
    class ChildrenProcessor(BasicProcessor):
        """返回 children 字段的处理器。"""

        def read_section(self, ref: str) -> dict[str, Any]:
            """读取章节。"""

            return {
                "ref": ref,
                "title": "Parent",
                "content": "directory content",
                "tables": [],
                "children": [
                    {"ref": "s_0001_c01", "title": "Child A", "level": 2, "preview": "p1"},
                    {"ref": "s_0001_c02", "title": "Child B", "level": 2, "preview": "p2"},
                ],
                "word_count": 2,
                "contains_full_text": False,
            }

    repository = FakeRepository()

    class ChildrenRegistry:
        """注册表桩。"""

        def create(self, source: Source, **kwargs: Any) -> Any:
            """创建处理器。"""

            del source, kwargs
            return ChildrenProcessor(document_id="test")

        def create_with_fallback(self, source: Source, **kwargs: Any) -> Any:
            """回退创建。"""

            return self.create(source, **kwargs)

    service = FinsToolService(repository=repository, processor_registry=ChildrenRegistry())
    result = service.read_section(ticker="AAPL", document_id="fil_1", ref="s_0001")

    assert result["content_word_count"] == 2
    assert "word_count" not in result
    # children 只保留 ref + title
    assert len(result["children"]) == 2
    assert result["children"][0] == {"ref": "s_0001_c01", "title": "Child A"}
    assert result["children"][1] == {"ref": "s_0001_c02", "title": "Child B"}
    assert "level" not in result["children"][0]
    assert "preview" not in result["children"][0]
    # path 不输出给 LLM
    assert "path" not in result
    # tables 不输出给 LLM（content 中 [[t_XXXX]] 占位符已携带 ref + 位置上下文）
    assert "tables" not in result


@pytest.mark.unit
def test_read_section_invalid_ref_raises_tool_argument_error_with_hint() -> None:
    """验证 read_section 在 ref 不存在时返回可执行的参数错误提示。"""

    @dataclass
    class MissingRefProcessor(BasicProcessor):
        """始终抛出缺失章节的处理器。"""

        def read_section(self, ref: str) -> dict[str, Any]:
            """读取章节。"""

            raise KeyError(f"章节不存在: {ref}")

    class MissingRefRegistry:
        """返回固定 MissingRefProcessor 的注册表桩。"""

        def create(self, source: Source, **kwargs: Any) -> Any:
            """创建处理器。"""

            del source, kwargs
            return MissingRefProcessor(document_id="fil_1")

        def create_with_fallback(self, source: Source, **kwargs: Any) -> Any:
            """兼容统一回退接口并复用 create。"""

            return self.create(source, **kwargs)

    service = FinsToolService(repository=FakeRepository(), processor_registry=MissingRefRegistry())

    with pytest.raises(ToolArgumentError, match="原样复制返回的 ref"):
        service.read_section(ticker="AAPL", document_id="fil_1", ref="s_3")


@pytest.mark.unit
def test_get_table_invalid_table_ref_raises_tool_argument_error_with_hint() -> None:
    """验证 get_table 在 table_ref 不存在时返回可执行的参数错误提示。"""

    @dataclass
    class MissingTableRefProcessor(BasicProcessor):
        """始终抛出缺失表格的处理器。"""

        def read_table(self, table_ref: str) -> dict[str, Any]:
            """读取表格。"""

            raise KeyError(f"表格不存在: {table_ref}")

    class MissingTableRefRegistry:
        """返回固定 MissingTableRefProcessor 的注册表桩。"""

        def create(self, source: Source, **kwargs: Any) -> Any:
            """创建处理器。"""

            del source, kwargs
            return MissingTableRefProcessor(document_id="fil_1")

        def create_with_fallback(self, source: Source, **kwargs: Any) -> Any:
            """兼容统一回退接口并复用 create。"""

            return self.create(source, **kwargs)

    service = FinsToolService(repository=FakeRepository(), processor_registry=MissingTableRefRegistry())

    with pytest.raises(ToolArgumentError, match="原样复制返回的 table_ref"):
        service.get_table(ticker="AAPL", document_id="fil_1", table_ref="t_9999")


def _build_per_document_registry(section_refs: dict[str, set[str]], table_refs: dict[str, set[str]]) -> Any:
    """构造按 document_id 分流的注册表桩。

    Args:
        section_refs: ``document_id -> 已知章节 ref 集合`` 的映射。
        table_refs: ``document_id -> 已知 table_ref 集合`` 的映射。

    Returns:
        实现 ``create`` / ``create_with_fallback`` 协议的注册表实例。

    Raises:
        无。
    """

    @dataclass
    class _PerDocumentProcessor:
        """根据所属 document_id 决定 ref/table_ref 是否存在的处理器桩。"""

        document_id: str
        known_section_refs: set[str]
        known_table_refs: set[str]

        def list_sections(self) -> list[dict[str, Any]]:
            """返回章节列表。"""

            return [
                {"ref": ref, "title": "章节", "level": 1, "parent_ref": None, "preview": "x"}
                for ref in sorted(self.known_section_refs)
            ]

        def get_section_title(self, ref: str) -> Optional[str]:
            """获取章节标题。"""

            return "章节" if ref in self.known_section_refs else None

        def read_section(self, ref: str) -> dict[str, Any]:
            """读取章节，未命中抛 KeyError。"""

            if ref not in self.known_section_refs:
                raise KeyError(f"章节不存在: ref={ref} document_id={self.document_id}")
            return {
                "ref": ref,
                "title": "章节",
                "content": "正文",
                "tables": [],
                "contains_full_text": True,
            }

        def list_tables(self) -> list[dict[str, Any]]:
            """列表表格。"""

            return [{"table_ref": tref, "section_ref": "s_0001"} for tref in sorted(self.known_table_refs)]

        def read_table(self, table_ref: str) -> dict[str, Any]:
            """读取表格，未命中抛 KeyError。"""

            if table_ref not in self.known_table_refs:
                raise KeyError(f"表格不存在: table_ref={table_ref} document_id={self.document_id}")
            return {
                "table_ref": table_ref,
                "data_format": "markdown",
                "data": "|A|\\n|---|",
                "columns": None,
                "row_count": 1,
                "col_count": 1,
                "is_financial": False,
                "section_ref": "s_0001",
            }

    class _PerDocumentRegistry:
        """根据 source URI 路由到对应 document 的处理器桩。"""

        def _resolve_document_id(self, source: Source) -> str:
            """从 source URI 中解析 document_id。

            URI 形如 ``local://filing/fil_1.html``。
            """

            uri = source.uri
            tail = uri.rsplit("/", 1)[-1]
            return tail.rsplit(".", 1)[0]

        def create(self, source: Source, **kwargs: Any) -> Any:
            """创建按 document_id 分流的处理器。"""

            del kwargs
            document_id = self._resolve_document_id(source)
            return _PerDocumentProcessor(
                document_id=document_id,
                known_section_refs=set(section_refs.get(document_id, set())),
                known_table_refs=set(table_refs.get(document_id, set())),
            )

        def create_with_fallback(self, source: Source, **kwargs: Any) -> Any:
            """兼容统一回退接口并复用 create。"""

            return self.create(source, **kwargs)

    return _PerDocumentRegistry()


@pytest.mark.unit
def test_read_section_diagnoses_cross_document_ref_reuse() -> None:
    """验证 read_section 在 ref 命中其他已缓存 processor 时给出跨文档诊断。

    覆盖 issue #102：locator 是 document-local，跨文档复用应在错误消息中显式区分，
    避免与"模型自造 ref"混为一谈。
    """

    registry = _build_per_document_registry(
        section_refs={"fil_1": {"s_0011_c01"}, "fil_2": set()},
        table_refs={"fil_1": set(), "fil_2": set()},
    )
    service = FinsToolService(repository=FakeRepository(), processor_registry=registry)

    # 先把 fil_1 的 processor 拉进缓存：成功读取 s_0011_c01
    service.read_section(ticker="AAPL", document_id="fil_1", ref="s_0011_c01")

    # 再用同一 ref 调 fil_2，应被诊断为跨文档复用
    with pytest.raises(ToolArgumentError) as excinfo:
        service.read_section(ticker="AAPL", document_id="fil_2", ref="s_0011_c01")

    message = str(excinfo.value)
    assert "疑似跨文档" in message
    assert "fil_2" in message  # 当前 document_id
    assert "fil_1" in message  # 疑似来源 document_id


@pytest.mark.unit
def test_get_table_diagnoses_cross_document_table_ref_reuse() -> None:
    """验证 get_table 对 table_ref 同样能识别跨文档复用。

    覆盖 issue #102 的表格路径。
    """

    registry = _build_per_document_registry(
        section_refs={"fil_1": set(), "fil_2": set()},
        table_refs={"fil_1": {"t_0007"}, "fil_2": set()},
    )
    service = FinsToolService(repository=FakeRepository(), processor_registry=registry)

    service.get_table(ticker="AAPL", document_id="fil_1", table_ref="t_0007")

    with pytest.raises(ToolArgumentError) as excinfo:
        service.get_table(ticker="AAPL", document_id="fil_2", table_ref="t_0007")

    message = str(excinfo.value)
    assert "疑似跨文档" in message
    assert "fil_2" in message
    assert "fil_1" in message


@pytest.mark.unit
def test_read_section_invalid_ref_falls_back_when_no_cached_match() -> None:
    """验证未命中任何已缓存 processor 时回退到原"请先调用 get_document_sections"提示。

    当不存在任何已缓存的同 ticker 文档可佐证跨文档复用时，错误文案应保持现状，
    避免误把"模型自造 ref"标成"疑似跨文档"。
    """

    registry = _build_per_document_registry(
        section_refs={"fil_1": set()},
        table_refs={"fil_1": set()},
    )
    service = FinsToolService(repository=FakeRepository(), processor_registry=registry)

    with pytest.raises(ToolArgumentError) as excinfo:
        service.read_section(ticker="AAPL", document_id="fil_1", ref="s_unknown")

    message = str(excinfo.value)
    assert "疑似跨文档" not in message
    assert "原样复制返回的 ref" in message


@pytest.mark.unit
def test_read_section_inherits_item_and_topic_from_parent_title() -> None:
    """验证 read_section 会从父章节标题继承 item 与 topic。"""

    from dayu.fins.domain.tool_models import SectionType

    @dataclass
    class ParentSemanticProcessor(BasicProcessor):
        """返回父子章节结构的处理器。"""

        def get_section_title(self, ref: str) -> Optional[str]:
            """返回父章节标题，供子章节继承语义。"""

            if ref == "sec_parent":
                return "Item 7. Management's Discussion and Analysis"
            return None

        def read_section(self, ref: str) -> dict[str, Any]:
            """返回无法自解析 item 的子章节。"""

            return {
                "ref": ref,
                "title": "Overview",
                "parent_ref": "sec_parent",
                "content": "discussion overview",
                "children": [],
                "contains_full_text": False,
            }

    class ParentSemanticRegistry:
        """返回固定 ParentSemanticProcessor 的注册表桩。"""

        def create(self, source: Source, **kwargs: Any) -> Any:
            """创建处理器。"""

            del source, kwargs
            return ParentSemanticProcessor(document_id="fil_1")

        def create_with_fallback(self, source: Source, **kwargs: Any) -> Any:
            """兼容统一回退接口并复用 create。"""

            return self.create(source, **kwargs)

    service = FinsToolService(repository=FakeRepository(), processor_registry=ParentSemanticRegistry())

    result = service.read_section(ticker="AAPL", document_id="fil_1", ref="sec_child")

    assert result["item"] == "Item 7"
    assert result["topic"] == SectionType.MDA.value
    assert result["citation"]["item"] == "Item 7"
    assert result["citation"]["heading"] == "Overview"


@pytest.mark.unit
def test_read_section_uses_get_section_title_without_listing_sections() -> None:
    """验证 read_section 获取父标题时优先走 `get_section_title()` 轻量接口。"""

    @dataclass
    class ParentTitleProcessor(BasicProcessor):
        """为父标题查询提供最小实现的处理器。"""

        list_sections_called: int = 0

        def list_sections(self) -> list[dict[str, Any]]:
            """若被调用，记录调用次数以防止回退到全量扫描。"""

            self.list_sections_called += 1
            raise AssertionError("read_section 不应为父标题再次调用 list_sections")

        def get_section_title(self, ref: str) -> Optional[str]:
            """按 ref 返回父章节标题。"""

            if ref == "sec_parent":
                return "Item 7. Management's Discussion and Analysis"
            return None

        def read_section(self, ref: str) -> dict[str, Any]:
            """返回带父章节引用的子章节。"""

            return {
                "ref": ref,
                "title": "Overview",
                "parent_ref": "sec_parent",
                "content": "discussion overview",
                "tables": [],
                "contains_full_text": False,
            }

    class ParentTitleRegistry:
        """返回固定 `ParentTitleProcessor` 的注册表桩。"""

        def __init__(self) -> None:
            """初始化处理器实例。"""

            self.processor = ParentTitleProcessor(document_id="fil_1")

        def create(self, source: Source, **kwargs: Any) -> Any:
            """创建处理器。"""

            del source, kwargs
            return self.processor

        def create_with_fallback(self, source: Source, **kwargs: Any) -> Any:
            """兼容统一回退接口并复用 create。"""

            return self.create(source, **kwargs)

    registry = ParentTitleRegistry()
    service = FinsToolService(repository=FakeRepository(), processor_registry=registry)

    result = service.read_section(ticker="AAPL", document_id="fil_1", ref="sec_child")

    assert result["item"] == "Item 7"
    assert result["citation"]["item"] == "Item 7"
    assert registry.processor.list_sections_called == 0


@pytest.mark.unit
def test_query_xbrl_facts_validates_concepts_type() -> None:
    """验证 query_xbrl_facts 拒绝非列表的 concepts。"""

    import pytest
    from dayu.engine.exceptions import ToolArgumentError

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    with pytest.raises(ToolArgumentError, match="concepts must be a string array or omitted"):
        service.query_xbrl_facts(
            ticker="AAPL",
            document_id="fil_1",
            concepts="not_a_list",  # type: ignore
        )


@pytest.mark.unit
def test_query_xbrl_facts_returns_not_supported_when_processor_lacks_capability() -> None:
    """验证 query_xbrl_facts 在处理器不支持时返回 not_supported 契约。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry(basic_mode=True)
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.query_xbrl_facts(
        ticker="AAPL",
        document_id="fil_1",
        concepts=["Revenues"],
    )

    assert "error" in result
    assert result["supported"] is False
    assert result["error"]["code"] == "not_supported"
    assert "concepts" in result
    assert result["concepts"] == ["Revenues"]


@pytest.mark.unit
def test_get_financial_statement_returns_payload_with_citation() -> None:
    """验证 get_financial_statement 支持路径会透传报表载荷、定位信息并附带 citation。"""

    repository = FakeRepository()
    processor_registry = FakeProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.get_financial_statement(
        ticker="AAPL",
        document_id="fil_1",
        statement_type="income",
    )

    assert "error" not in result
    assert result["statement_type"] == "income"
    assert result["currency"] == "USD"
    assert result["units"] == "millions"
    assert "statement_locator" in result
    assert "statement_type" in result["statement_locator"]
    assert result["statement_locator"]["statement_type"] == "income"
    assert "period_labels" in result["statement_locator"]
    assert result["statement_locator"]["period_labels"] == ["FY2025"]
    assert "row_labels" in result["statement_locator"]
    assert result["statement_locator"]["row_labels"] == ["Revenue"]
    assert result["citation"]["document_id"] == "fil_1"


# ---- _build_match_quality 单元测试 ----


def _make_match(is_exact: bool) -> dict[str, Any]:
    """构造最小化的 match 字典，仅含 is_exact_phrase 字段。"""
    return {"section_ref": "s1", "is_exact_phrase": is_exact}


@pytest.mark.unit
class TestBuildMatchQuality:
    """_build_match_quality 基于 post-cap matches 的四种 primary_source 测试。"""

    def test_exact_only(self) -> None:
        """仅精确命中时 primary_source 应为 exact。"""

        from dayu.fins.tools.service_helpers import _build_match_quality

        matches = [_make_match(True) for _ in range(5)]
        result = _build_match_quality(matches)
        assert result == {
            "exact_phrase_matches": 5,
            "expansion_matches": 0,
            "primary_source": "exact",
        }

    def test_expansion_only(self) -> None:
        """仅扩展命中时 primary_source 应为 expansion_only。"""

        from dayu.fins.tools.service_helpers import _build_match_quality

        matches = [_make_match(False) for _ in range(10)]
        result = _build_match_quality(matches)
        assert result == {
            "exact_phrase_matches": 0,
            "expansion_matches": 10,
            "primary_source": "expansion_only",
        }

    def test_mixed(self) -> None:
        """精确与扩展皆有命中时 primary_source 应为 mixed。"""

        from dayu.fins.tools.service_helpers import _build_match_quality

        matches = [_make_match(True) for _ in range(3)] + [_make_match(False) for _ in range(7)]
        result = _build_match_quality(matches)
        assert result == {
            "exact_phrase_matches": 3,
            "expansion_matches": 7,
            "primary_source": "mixed",
        }

    def test_none(self) -> None:
        """零命中时 primary_source 应为 none。"""

        from dayu.fins.tools.service_helpers import _build_match_quality

        result = _build_match_quality([])
        assert result == {
            "exact_phrase_matches": 0,
            "expansion_matches": 0,
            "primary_source": "none",
        }

    def test_counts_equal_total(self) -> None:
        """exact + expansion 应始终等于 len(matches)，消除语义层级歧义。"""

        from dayu.fins.tools.service_helpers import _build_match_quality

        matches = [_make_match(True), _make_match(False), _make_match(False)]
        result = _build_match_quality(matches)
        assert result["exact_phrase_matches"] + result["expansion_matches"] == len(matches)


# ---- _build_search_hint 单元测试 ----


@pytest.mark.unit
class TestBuildSearchHint:
    """_build_search_hint 面向 LLM 操作引导提示的测试。"""

    def test_none_source_returns_none(self) -> None:
        """零命中时不生成提示。"""

        from dayu.fins.tools.service_helpers import _build_search_hint

        assert _build_search_hint([], "none") is None

    def test_exact_source_returns_none(self) -> None:
        """纯精确命中时不生成提示。"""

        from dayu.fins.tools.service_helpers import _build_search_hint

        matches = [_make_match(True) for _ in range(5)]
        assert _build_search_hint(matches, "exact") is None

    def test_expansion_only_warns_noise(self) -> None:
        """纯扩展命中时应警告噪声风险并建议替代操作。"""

        from dayu.fins.tools.service_helpers import _build_search_hint

        matches = [_make_match(False) for _ in range(20)]
        hint = _build_search_hint(matches, "expansion_only")
        assert hint is not None
        assert "目标：" in hint
        assert "允许动作：" in hint
        assert "不允许：" in hint
        assert "下一步：" in hint
        assert "read_section" in hint

    def test_mixed_shows_exact_count_and_fetch_more_guidance(self) -> None:
        """混合命中时应报告精确数并引导避免 fetch_more。"""

        from dayu.fins.tools.service_helpers import _build_search_hint

        matches = [_make_match(True) for _ in range(3)] + [_make_match(False) for _ in range(17)]
        hint = _build_search_hint(matches, "mixed")
        assert hint is not None
        assert "目标：" in hint
        assert "fetch_more" in hint
        assert "不允许" in hint
        assert "fetch_more" in hint


# ---- match_quality 集成测试 ----


@pytest.mark.unit
def test_search_document_match_quality_exact_only() -> None:
    """验证 search_document 单查询精确命中时返回 match_quality.primary_source=exact。"""

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        query="repurchase",
    )

    mq = result["match_quality"]
    assert mq["primary_source"] == "exact"
    assert mq["exact_phrase_matches"] == 2
    assert mq["expansion_matches"] == 0


@pytest.mark.unit
def test_search_document_match_quality_expansion_only() -> None:
    """验证 search_document 同义词回退时返回 match_quality.primary_source=expansion_only。"""

    repository = FakeRepository()
    processor_registry = SynonymFallbackProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        query="buyback",
    )

    mq = result["match_quality"]
    assert mq["primary_source"] == "expansion_only"
    assert mq["exact_phrase_matches"] == 0
    assert mq["expansion_matches"] >= 1


@pytest.mark.unit
def test_search_document_match_quality_none() -> None:
    """验证 search_document 零命中时返回 match_quality.primary_source=none。"""

    repository = FakeRepository()
    processor_registry = SynonymFallbackProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        query="nonexistent_xyz_gibberish",
    )

    mq = result["match_quality"]
    assert mq["primary_source"] == "none"
    assert mq["exact_phrase_matches"] == 0
    assert mq["expansion_matches"] == 0
    assert result["total_matches"] == 0


@pytest.mark.unit
def test_search_document_multi_queries_has_match_quality() -> None:
    """验证 queries 批量查询路径也返回 match_quality。"""

    repository = FakeRepository()
    processor_registry = SearchEnhancedProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        queries=["repurchase", "board"],
    )

    mq = result["match_quality"]
    assert "primary_source" in mq
    assert "exact_phrase_matches" in mq
    assert "expansion_matches" in mq
    # SearchEnhancedProcessor 全是精确命中
    assert mq["primary_source"] == "exact"
    assert mq["exact_phrase_matches"] > 0


# ===========================================================================
# 中文查询零结果 hint（S1）
# ===========================================================================


@pytest.mark.unit
def test_search_document_chinese_query_no_results_generates_hint() -> None:
    """中文查询无命中时，search_document 应生成中文动作引导 hint（单 query 路径）。

    SynonymFallbackProcessor 对非 'share repurchase' 的词返回空，模拟中文词在英文文档零命中。
    期望：total_matches=0 且 hint 含中文操作引导。
    """

    repository = FakeRepository()
    processor_registry = SynonymFallbackProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        query="未命中的词汇",
    )

    assert result["total_matches"] == 0
    hint = result.get("hint", "")
    assert hint, "中文查询无结果时应生成 hint"
    assert "把中文词换成英文关键词再搜" in hint


@pytest.mark.unit
def test_search_document_chinese_queries_batch_no_results_generates_hint() -> None:
    """批量中文查询无命中时，多查询路径同样生成中文动作引导 hint。"""

    repository = FakeRepository()
    processor_registry = SynonymFallbackProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        queries=["未知词甲", "未知词乙"],
    )

    assert result["total_matches"] == 0
    hint = result.get("hint", "")
    assert hint, "批量中文查询无结果时应生成 hint"
    assert "把中文词换成英文关键词再搜" in hint


@pytest.mark.unit
def test_search_document_english_query_no_results_no_chinese_hint() -> None:
    """英文查询无命中时，不应生成中文相关 hint（防误触）。"""

    repository = FakeRepository()
    processor_registry = SynonymFallbackProcessorRegistry()
    service = FinsToolService(repository=repository, processor_registry=processor_registry)

    result = service.search_document(
        ticker="AAPL",
        document_id="fil_1",
        query="nonexistent term",
    )

    assert result["total_matches"] == 0
    hint = result.get("hint", "")
    assert "把中文词换成英文关键词再搜" not in hint
