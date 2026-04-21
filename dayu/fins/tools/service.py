"""财报工具服务层。

该模块是财报工具与底层仓储/处理器之间的中间调用层，职责包括：
- 参数校验与标准化。
- `document_id -> source_kind -> source -> processor` 路由。
- 统一能力降级（`not_supported`）。
- 仅缓存 Processor 实例（key=`ticker + document_id`，仅 LRU）。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from threading import Lock, RLock
from typing import Any, Optional, cast

from dayu.engine.exceptions import ToolArgumentError
from dayu.log import Log
from dayu.engine.tool_errors import ToolBusinessError
from dayu.engine.processors.base import (
    DocumentProcessor,
    SectionContent,
    SectionSummary,
    TableContent,
    TableSummary,
)
from dayu.engine.processors.processor_registry import ProcessorRegistry
from dayu.engine.tools.error_contract import ErrorCode
from dayu.fins.domain.enums import SourceKind
from dayu.fins.domain.tool_models import Citation, SourceType
from dayu.fins.storage import (
    CompanyMetaRepositoryProtocol,
    ProcessedDocumentRepositoryProtocol,
    SourceDocumentRepositoryProtocol,
)
from .bm25f_scorer import BM25FSectionIndex, build_section_bm25f_index
from .section_semantic import (
    build_section_path,
    resolve_section_semantic,
)
from .cache import ProcessorCacheKey, ProcessorLRUCache

# 从拆分模块导入（FinsToolService 直接使用）
from .search_models import (
    QueryDiagnosis,
    SectionSemanticProfile,
    SEARCH_MODE_AUTO,
    _SEARCH_RANKING_VERSION,
)
from .search_engine import (
    _resolve_search_queries,
    _resolve_search_mode,
    _diagnose_search_query,
    _execute_query_search,
    _build_empty_search_strategy_hit_counts,
    _deduplicate_ranked_search_entries,
    _sort_ranked_search_entries,
    _build_evidence_matches,
    _build_section_semantic_profiles,
    _cap_entries_with_exact_priority,
)
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
from dayu.fins._converters import normalize_optional_text, require_non_empty_text
from dayu.fins.ticker_normalization import try_normalize_ticker
from .service_helpers import (
    _collect_parent_titles,
    _normalize_form_type_for_matching,
    _normalize_document_types,
    _build_recommended_documents,
    _normalize_section_children,
    _normalize_periods,
    _build_not_supported_result,
    _extract_page_range,
    _infer_fiscal_period,
    _infer_fiscal_year,
    _resolve_fiscal_year_with_fallback,
    _resolve_fiscal_period_with_fallback,
    _build_table_data_payload,
    _normalize_table_type,
    _resolve_processor_taxonomy,
    _resolve_default_xbrl_concepts,
    _normalize_xbrl_query_payload,
    _collect_available_document_types,
    _build_match_quality,
    _build_search_hint,
    build_search_next_section_fields,
    build_document_recency_sort_key,
    resolve_has_financial_data,
    resolve_document_type_for_source,
)
# 匹配 CJK 统一汉字（基本区 + 扩展A），用于检测查询词是否含中文
_CN_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
def _any_query_has_chinese(queries: list[str]) -> bool:
    """检测查询列表中是否存在含中文字符的查询词。"""
    return any(_CN_CHAR_RE.search(q) for q in queries)


# 中文查询在未命中时给出的操作引导提示（通用，不假设文档语言）
_CHINESE_QUERY_NO_RESULTS_HINT = (
    "目标：把查询改成文档更可能命中的写法。允许动作：把中文词换成英文关键词再搜。"
    "不允许：继续用中文词反复重试。下一步：例如把“年度经常性收入”改成“annual recurring revenue”后再搜。"
)
_MISSING_TICKER_HINT = (
    "目标：先确认这家公司是否已被当前财报工具收录。允许动作：切到公司或网页来源确认公司标识。"
    "不允许：继续穷举 ticker 变体。下一步：先确认公司标识，再回到财报工具。"
)


class FinsToolService:
    """财报工具服务。

    设计约束：
    - 不依赖 `processed/*.json` 产物。
    - 所有读取均通过实时 Processor 能力完成。
    - 缓存仅保留 Processor 实例。
    """

    MODULE = "FINS.TOOL_SERVICE"
    def __init__(
        self,
        *,
        company_repository: CompanyMetaRepositoryProtocol,
        source_repository: SourceDocumentRepositoryProtocol,
        processed_repository: ProcessedDocumentRepositoryProtocol,
        processor_registry: ProcessorRegistry,
        processor_cache_max_entries: int = 128,
    ) -> None:
        """初始化服务。

        Args:
            company_repository: 公司元数据仓储实现。
            source_repository: 源文档仓储实现。
            processed_repository: processed 文档仓储实现。
            processor_registry: 处理器注册表。
            processor_cache_max_entries: Processor LRU 缓存容量。

        Returns:
            无。

        Raises:
            ValueError: 当缓存容量非法时抛出。
        """

        if processor_cache_max_entries <= 0:
            raise ValueError("processor_cache_max_entries must be greater than 0")
        self._company_repository = company_repository
        self._source_repository = source_repository
        self._processed_repository = processed_repository
        self._processor_registry = processor_registry
        self._processor_cache: ProcessorLRUCache[DocumentProcessor] = ProcessorLRUCache(
            max_entries=processor_cache_max_entries,
        )
        self._meta_cache: dict[tuple[str, str], Optional[dict[str, Any]]] = {}
        self._creation_locks: dict[ProcessorCacheKey, Lock] = {}
        self._creation_locks_guard = RLock()

    def list_documents(
        self,
        *,
        ticker: str,
        document_types: Optional[list[str]] = None,
        fiscal_years: Optional[list[int]] = None,
        fiscal_periods: Optional[list[str]] = None,
    ) -> ListDocumentsResult:
        """列出可用文档。

        Args:
            ticker: 股票代码。
            document_types: 可选文档类型过滤（枚举数组，如 ["annual_report", "quarterly_report"]）。
            fiscal_years: 可选财年过滤。
            fiscal_periods: 可选财期过滤。

        Returns:
            文档列表结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
            RuntimeError: 仓储读取失败时抛出。
        """

        normalized_ticker = self._resolve_canonical_ticker(
            ticker=ticker,
            tool_name="list_documents",
        )
        normalized_document_types = _normalize_document_types(document_types)
        normalized_fiscal_periods = _normalize_periods(fiscal_periods)

        company_name, market = self._read_company_info(normalized_ticker)
        base_documents = self._collect_source_documents(normalized_ticker)

        # 先为全量文档附加 document_type，供推荐槽位与过滤逻辑共享。
        documents_with_type: list[dict[str, Any]] = []
        for item in base_documents:
            output = dict(item)
            output["document_type"] = resolve_document_type_for_source(
                form_type=item.get("form_type"),
                source_kind=item.get("source_kind"),
            )
            documents_with_type.append(output)

        # 主过滤逻辑：按类型 / 财年 / 财期筛选；推荐槽位仍基于全量文档构建。
        filtered_documents: list[dict[str, Any]] = []
        for item in documents_with_type:
            doc_type = item["document_type"]
            if normalized_document_types is not None and doc_type not in normalized_document_types:
                continue
            fiscal_year = item.get("fiscal_year")
            if fiscal_years and fiscal_year not in fiscal_years:
                continue
            fiscal_period = item.get("fiscal_period")
            if normalized_fiscal_periods and fiscal_period not in normalized_fiscal_periods:
                continue
            output = dict(item)
            # 屏蔽底层 SEC 表单名，不对 LLM 暴露
            output.pop("form_type", None)
            filtered_documents.append(output)
        recommended_documents = _build_recommended_documents(documents_with_type)

        # 判定匹配状态并构建 suggestion
        if normalized_document_types is not None and len(filtered_documents) == 0:
            available = _collect_available_document_types(base_documents)
            match_status = "no_match"
            suggestion: Optional[dict[str, Any]] = {
                "action": "broaden_filter",
                "available_document_types": available,
                "reason": "no_documents_matched_document_types",
            }
        else:
            match_status = "ok"
            suggestion = None

        result: ListDocumentsResult = {
            "company": {
                "ticker": normalized_ticker,
                "name": company_name,
                "market": market,
            },
            "filters": {
                "document_types": normalized_document_types,
                "fiscal_years": fiscal_years,
                "fiscal_periods": normalized_fiscal_periods,
            },
            "recommended_documents": recommended_documents,
            "documents": filtered_documents,
            "total": len(base_documents),
            "matched": len(filtered_documents),
            "match_status": match_status,
        }
        if suggestion is not None:
            result["suggestion"] = suggestion

        return result

    def get_document_sections(self, *, ticker: str, document_id: str) -> DocumentSectionsResult:
        """获取文档章节结构。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。

        Returns:
            章节结构结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
            FileNotFoundError: 文档不存在时抛出。
        """

        normalized_ticker, normalized_document_id = self._normalize_document_identity(
            ticker=ticker,
            document_id=document_id,
            tool_name="get_document_sections",
        )
        processor = self._get_or_create_processor(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        sections_raw: list[SectionSummary] = processor.list_sections()
        form_type = self._resolve_document_form_type(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        enriched_sections = self._enrich_sections_with_semantic(sections_raw, form_type)
        citation = self._build_citation(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        return {
            "ticker": normalized_ticker,
            "document_id": normalized_document_id,
            "sections": enriched_sections,
            "citation": citation,
        }

    def read_section(self, *, ticker: str, document_id: str, ref: str) -> SectionContentResult:
        """读取章节正文。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            ref: 章节引用。

        Returns:
            章节正文结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
            KeyError: 章节不存在时抛出。
        """

        normalized_ticker, normalized_document_id = self._normalize_document_identity(
            ticker=ticker,
            document_id=document_id,
            tool_name="read_section",
        )
        normalized_ref = require_non_empty_text(
            ref,
            empty_error=ToolArgumentError("read_section", "ref", ref, "Argument must not be empty"),
        )
        processor = self._get_or_create_processor(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        try:
            section_raw: SectionContent = processor.read_section(normalized_ref)
        except KeyError as exc:
            raise ToolArgumentError(
                "read_section",
                "ref",
                normalized_ref,
                "章节不存在；请先调用 get_document_sections，并原样复制返回的 ref，不要简写、重编号或自造 ref",
            ) from exc
        content = str(section_raw.get("content", ""))
        # tables 字段不输出给 LLM——content 中 [[t_XXXX]] 占位符已携带 ref + 位置上下文，
        # 纯 ref 列表无选择线索（与 children 同理：ref=入参，需有线索才有决策价值），
        # content 截断时 LLM 应走 list_tables(within_section_ref) 获取完整表格元数据。
        normalized_children = _normalize_section_children(section_raw.get("children"))
        content_word_count = int(
            section_raw.get("content_word_count") or section_raw.get("word_count") or len(content.split())
        )
        # 语义增强：解析 item/topic/path
        form_type = self._resolve_document_form_type(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        title = section_raw.get("title")
        # read_section 没有直接的 parent_ref 上下文，从 list_sections 获取
        parent_ref = section_raw.get("parent_ref")
        parent_title = None
        if parent_ref:
            # 直接走处理器的 O(1) 标题查询，避免为父标题再扫一遍全量 sections。
            try:
                parent_title = processor.get_section_title(str(parent_ref))
            except Exception:
                parent_title = None

        item_number, canonical_title, topic = resolve_section_semantic(
            title=title,
            form_type=form_type,
            parent_title=parent_title,
        )
        # 子章节无法自解析时，尝试从父章节标题继承 item/topic
        if (item_number is None or topic is None) and parent_title:
            parent_item, _, parent_topic = resolve_section_semantic(
                title=parent_title,
                form_type=form_type,
            )
            if item_number is None:
                item_number = parent_item
            if topic is None:
                topic = parent_topic
        parent_titles: list[str] = []
        if parent_title:
            parent_titles.append(parent_title)
        # path 计算保留供内部诊断与未来评估，但不输出给 LLM——
        # item + topic + title 已充分表达语义位置，path 是冗余拼合，
        # 与 get_document_sections 去 path 的 T1 决策保持一致。
        _path = build_section_path(
            form_type=form_type,
            item_number=item_number,
            canonical_title=canonical_title,
            section_title=title,
            parent_titles=parent_titles,
        )
        del _path  # 显式丢弃，静默 linter unused-variable 警告
        item_label = f"Item {item_number}" if item_number else None
        citation = self._build_citation(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
            item=item_label,
            heading=str(title) if title else None,
        )
        return {
            "ticker": normalized_ticker,
            "document_id": normalized_document_id,
            "ref": normalized_ref,
            "title": title,
            "item": item_label,
            "topic": topic,
            "content": content,
            "children": normalized_children,
            "page_range": _extract_page_range(section_raw),
            "content_word_count": content_word_count,
            "citation": citation,
        }

    def search_document(
        self,
        *,
        ticker: str,
        document_id: str,
        query: Optional[str] = None,
        queries: Optional[list[str]] = None,
        within_section_ref: Optional[str] = None,
        mode: Optional[str] = None,
        display_budget: Optional[int] = None,
    ) -> SearchDocumentResult:
        """在文档内搜索关键词，支持单查询和批量查询。

        ``query`` 与 ``queries`` 互斥，必须提供其一。
        批量查询时逐条执行搜索后聚合去重排序。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            query: 单条搜索关键词（与 queries 互斥）。
            queries: 批量搜索关键词列表（与 query 互斥，上限 20 条）。
            within_section_ref: 可选章节范围。
            mode: 搜索模式，可选值：
                - ``auto``（默认）：先精确匹配，无命中时自动扩展。
                - ``exact``：仅精确匹配。
                - ``keyword``：仅关键词拆分搜索。
                - ``semantic``：语义扩展（短语变体 + 同义词 + 关键词）。
            display_budget: 可选展示预算上限，传递给 exact 优先限流，
                避免裁剪后条目数超出下游 truncation max_items 引发信号冲突。

        Returns:
            搜索结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
        """

        _QUERIES_MAX = 20

        normalized_ticker, normalized_document_id = self._normalize_document_identity(
            ticker=ticker,
            document_id=document_id,
            tool_name="search_document",
        )
        # 校验互斥：query 与 queries 必须提供其一
        resolved_queries = _resolve_search_queries(
            query=query, queries=queries, max_queries=_QUERIES_MAX,
        )
        # 保存查询词副本，供中文无结果 hint 检测使用
        original_queries = resolved_queries
        normalized_within_ref = normalize_optional_text(within_section_ref)
        resolved_mode = _resolve_search_mode(mode)

        processor = self._get_or_create_processor(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        # 预构建证据化所需的 form_type / ref_to_topic
        form_type = self._resolve_document_form_type(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        ref_to_topic: dict[str, Optional[str]] = {}
        semantic_profiles: dict[str, SectionSemanticProfile] = {}
        query_term_df: dict[str, int] = {}
        bm25f_index = BM25FSectionIndex(
            profiles={},
            document_frequency={},
            avg_field_lengths={},
            avg_content_length=0.0,
            document_count=0,
        )
        try:
            all_secs = processor.list_sections()
            enriched_for_search = self._enrich_sections_with_semantic(
                sections=all_secs, form_type=form_type
            )
            bm25f_index = build_section_bm25f_index(enriched_for_search)
            semantic_profiles, query_term_df = _build_section_semantic_profiles(enriched_for_search)
            for sec in enriched_for_search:
                ref = sec.get("ref")
                if ref:
                    ref_to_topic[ref] = sec.get("topic")
        except Exception:
            pass

        is_multi = len(resolved_queries) > 1

        if is_multi:
            # ---- 批量查询聚合路径 ----
            return self._search_document_multi(
                normalized_ticker=normalized_ticker,
                normalized_document_id=normalized_document_id,
                resolved_queries=resolved_queries,
                original_queries=original_queries,
                normalized_within_ref=normalized_within_ref,
                resolved_mode=resolved_mode,
                processor=processor,
                form_type=form_type,
                ref_to_topic=ref_to_topic,
                bm25f_index=bm25f_index,
                semantic_profiles=semantic_profiles,
                query_term_df=query_term_df,
                display_budget=display_budget,
            )

        # ---- 单查询路径 ----
        normalized_query = resolved_queries[0]
        diagnosis = _diagnose_search_query(
            query=normalized_query,
            term_document_frequency=query_term_df,
            document_count=max(1, len(semantic_profiles)),
            mode=resolved_mode,
        )
        ranked_entries, strategy_hit_counts, exact_matches, expansion_queries = (
            _execute_query_search(
                processor=processor,
                query=normalized_query,
                within_ref=normalized_within_ref,
                mode=resolved_mode,
                diagnosis=diagnosis,
                semantic_profiles=semantic_profiles,
            )
        )

        deduplicated_entries = _deduplicate_ranked_search_entries(ranked_entries)
        sorted_entries = _sort_ranked_search_entries(
            deduplicated_entries,
            bm25f_index=bm25f_index,
            diagnosis=diagnosis,
            semantic_profiles=semantic_profiles,
        )
        # exact 优先限流：当精确命中存在时，压缩扩展结果占比
        capped_entries = _cap_entries_with_exact_priority(sorted_entries, display_budget=display_budget)
        matches = _build_evidence_matches(capped_entries, form_type, ref_to_topic)
        fallback_opened = any(bool(item.get("_token_fallback_opened", False)) for item in sorted_entries)
        noise_penalty_applied_count = sum(
            1 for item in sorted_entries if float(item.get("_context_noise_penalty", 0.0)) > 0.0
        )
        diagnostics = {
            "input_query": normalized_query,
            "mode": resolved_mode,
            "used_expansion": not bool(exact_matches) and bool(expansion_queries),
            "expanded_queries": expansion_queries,
            "expansion_query_count": len(expansion_queries),
            "strategy_hit_counts": strategy_hit_counts,
            "ranking_version": _SEARCH_RANKING_VERSION,
            "diagnosis_summary": {
                "intent": diagnosis.intent,
                "token_count": diagnosis.token_count,
                "ambiguity_score": diagnosis.ambiguity_score,
                "is_high_ambiguity": diagnosis.is_high_ambiguity,
            },
            "search_plan": {
                "fallback_gated": diagnosis.is_high_ambiguity and not diagnosis.allow_direct_token_fallback,
                "scoped_before_token": diagnosis.is_high_ambiguity,
            },
            "fallback_gated": diagnosis.is_high_ambiguity and not diagnosis.allow_direct_token_fallback,
            "fallback_opened": fallback_opened,
            "noise_penalty_applied_count": noise_penalty_applied_count,
        }
        Log.debug(
            "search_document 检索完成: "
            f"ticker={normalized_ticker} document_id={normalized_document_id} "
            f"query={normalized_query!r} mode={resolved_mode} "
            f"searched_in={normalized_within_ref or 'full text'} "
            f"exact_hits={len(exact_matches)} expansion_count={len(expansion_queries)} "
            f"total_matches={len(matches)} strategy_hits={strategy_hit_counts}",
            module=self.MODULE,
        )

        match_quality = _build_match_quality(matches)
        hint = _build_search_hint(matches, match_quality["primary_source"])
        # 中文查询无结果时补充操作引导提示
        if not hint and not matches and _any_query_has_chinese(original_queries):
            hint = _CHINESE_QUERY_NO_RESULTS_HINT
        next_section_to_read, next_section_by_query = build_search_next_section_fields(matches=matches)

        result: SearchDocumentResult = {
            "ticker": normalized_ticker,
            "document_id": normalized_document_id,
            "query": normalized_query,
            "mode": resolved_mode,
            "searched_in": normalized_within_ref or "full text",
            "match_quality": match_quality,
            "matches": matches,
            "next_section_to_read": next_section_to_read,
            "total_matches": len(matches),
            "diagnostics": diagnostics,
            "citation": self._build_citation(
                ticker=normalized_ticker,
                document_id=normalized_document_id,
            ),
        }
        if hint:
            result["hint"] = hint
        return result

    def _search_document_multi(
        self,
        *,
        normalized_ticker: str,
        normalized_document_id: str,
        resolved_queries: list[str],
        original_queries: list[str],
        normalized_within_ref: Optional[str],
        resolved_mode: str,
        processor: "DocumentProcessor",
        form_type: Optional[str],
        ref_to_topic: dict[str, Optional[str]],
        bm25f_index: BM25FSectionIndex,
        semantic_profiles: dict[str, SectionSemanticProfile],
        query_term_df: dict[str, int],
        display_budget: Optional[int] = None,
    ) -> SearchDocumentResult:
        """批量查询聚合路径。

        逐条执行搜索后汇总 ranked_entries，统一去重排序构建结果。

        Args:
            normalized_ticker: 标准化 ticker。
            normalized_document_id: 标准化 document_id。
            resolved_queries: 已翻译并校验的查询列表。
            original_queries: 翻译前的原始查询列表，用于中文无结果 hint 检测。
            normalized_within_ref: 可选章节范围。
            resolved_mode: 搜索模式。
            processor: 文档处理器。
            form_type: 文档 form_type。
            ref_to_topic: ref → topic 映射。
            bm25f_index: BM25F 索引。
            semantic_profiles: 章节语义画像映射。
            query_term_df: 查询词 document frequency。
            display_budget: 可选展示预算上限，传递给 exact 优先限流。

        Returns:
            聚合搜索结果。
        """

        all_ranked: list[dict[str, Any]] = []
        per_query_stats: list[dict[str, Any]] = []
        merged_strategy_hits = _build_empty_search_strategy_hit_counts()

        for q in resolved_queries:
            query_diagnosis = _diagnose_search_query(
                query=q,
                term_document_frequency=query_term_df,
                document_count=max(1, len(semantic_profiles)),
                mode=resolved_mode,
            )
            ranked, strategy_hits, exact_matches, expansion_queries = (
                _execute_query_search(
                    processor=processor, query=q,
                    within_ref=normalized_within_ref, mode=resolved_mode,
                    diagnosis=query_diagnosis,
                    semantic_profiles=semantic_profiles,
                )
            )
            all_ranked.extend(ranked)
            # 合并策略命中计数
            for strat, cnt in strategy_hits.items():
                merged_strategy_hits[strat] = merged_strategy_hits.get(strat, 0) + cnt
            per_query_stats.append({
                "query": q,
                "hits": len(ranked),
                "exact_hits": len(exact_matches),
                "expansion_count": len(expansion_queries),
                "is_high_ambiguity": query_diagnosis.is_high_ambiguity,
                "intent": query_diagnosis.intent,
            })

        deduplicated = _deduplicate_ranked_search_entries(all_ranked)
        sorted_entries = _sort_ranked_search_entries(
            deduplicated,
            bm25f_index=bm25f_index,
            diagnosis=None,
            semantic_profiles=semantic_profiles,
        )
        # exact 优先限流：当精确命中存在时，压缩扩展结果占比
        capped_entries = _cap_entries_with_exact_priority(sorted_entries, display_budget=display_budget)
        matches = _build_evidence_matches(capped_entries, form_type, ref_to_topic)

        diagnostics = {
            "input_queries": resolved_queries,
            "mode": resolved_mode,
            "query_count": len(resolved_queries),
            "per_query_stats": per_query_stats,
            "strategy_hit_counts": merged_strategy_hits,
            "ranking_version": _SEARCH_RANKING_VERSION,
            "fallback_gated": any(bool(item.get("is_high_ambiguity")) for item in per_query_stats),
            "noise_penalty_applied_count": sum(
                1 for item in sorted_entries if float(item.get("_context_noise_penalty", 0.0)) > 0.0
            ),
        }
        Log.debug(
            "search_document(multi) 检索完成: "
            f"ticker={normalized_ticker} document_id={normalized_document_id} "
            f"queries={resolved_queries!r} mode={resolved_mode} "
            f"searched_in={normalized_within_ref or 'full text'} "
            f"total_matches={len(matches)} per_query_stats={per_query_stats}",
            module=self.MODULE,
        )

        match_quality = _build_match_quality(matches)
        hint = _build_search_hint(matches, match_quality["primary_source"])
        # 中文查询无结果时补充操作引导提示
        if not hint and not matches and _any_query_has_chinese(original_queries):
            hint = _CHINESE_QUERY_NO_RESULTS_HINT
        _next_section_to_read, next_section_by_query = build_search_next_section_fields(
            matches=matches,
            queries=resolved_queries,
        )
        del _next_section_to_read
        # 批量查询路径下 queries 非空，next_section_by_query 必有值
        assert next_section_by_query is not None

        result: SearchDocumentResult = {
            "ticker": normalized_ticker,
            "document_id": normalized_document_id,
            "query": None,
            "queries": resolved_queries,
            "mode": resolved_mode,
            "searched_in": normalized_within_ref or "full text",
            "match_quality": match_quality,
            "matches": matches,
            "next_section_by_query": next_section_by_query,
            "total_matches": len(matches),
            "diagnostics": diagnostics,
            "citation": self._build_citation(
                ticker=normalized_ticker,
                document_id=normalized_document_id,
            ),
        }
        if hint:
            result["hint"] = hint
        return result

    def list_tables(
        self,
        *,
        ticker: str,
        document_id: str,
        financial_only: bool = False,
        within_section_ref: Optional[str] = None,
    ) -> TablesListResult:
        """列出文档表格元数据。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            financial_only: 是否仅返回财务表格。
            within_section_ref: 可选章节范围。

        Returns:
            表格列表结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
        """

        normalized_ticker, normalized_document_id = self._normalize_document_identity(
            ticker=ticker,
            document_id=document_id,
            tool_name="list_tables",
        )
        normalized_within_ref = normalize_optional_text(within_section_ref)

        processor = self._get_or_create_processor(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        tables_raw: list[TableSummary] = processor.list_tables()

        filtered_tables: list[dict[str, Any]] = []
        for item in tables_raw:
            is_financial = bool(item.get("is_financial", False))
            section_ref = item.get("section_ref")
            if financial_only and not is_financial:
                continue
            if normalized_within_ref is not None and section_ref != normalized_within_ref:
                continue
            page_no = item.get("page_no")
            # 构建表格条目：headers 截断至 80 chars/条，省略 null 可选字段减少序列化开销
            # context_before 已移除——753cd7a 后 caption 推断覆盖了其信息，
            # caption + headers + table_type 足够 LLM 判断表格相关性。
            raw_headers = item.get("headers")
            entry: dict[str, Any] = {
                "table_ref": item.get("table_ref"),
                "row_count": int(item.get("row_count", 0) or 0),
                "col_count": int(item.get("col_count", 0) or 0),
                "is_financial": is_financial,
                "table_type": _normalize_table_type(item.get("table_type")),
                "headers": (
                    [str(h)[:80] for h in raw_headers if h]
                    if isinstance(raw_headers, list)
                    else None
                ),
            }
            # within_section：与请求参数 within_section_ref 语义同源，表达 table 所属 section
            if section_ref:
                ws: dict[str, str] = {"ref": section_ref}
                sec_title = processor.get_section_title(section_ref)
                if sec_title:
                    ws["title"] = sec_title
                entry["within_section"] = ws
            caption = item.get("caption")
            if caption:
                entry["caption"] = caption
            if isinstance(page_no, int) and page_no > 0:
                entry["page_no"] = page_no
            filtered_tables.append(entry)

        # 复杂逻辑说明：先按财务优先排序，再按 table_ref 稳定排序，确保返回结果可复现。
        filtered_tables.sort(
            key=lambda item: (
                0 if bool(item.get("is_financial", False)) else 1,
                str(item.get("table_ref", "")),
            )
        )
        financial_count = sum(1 for item in filtered_tables if bool(item.get("is_financial", False)))
        return {
            "ticker": normalized_ticker,
            "document_id": normalized_document_id,
            "tables": filtered_tables,
            "total": len(filtered_tables),
            "financial_count": financial_count,
            "citation": self._build_citation(
                ticker=normalized_ticker,
                document_id=normalized_document_id,
            ),
        }

    def get_table(self, *, ticker: str, document_id: str, table_ref: str) -> TableDetailResult:
        """读取指定表格。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            table_ref: 表格引用。

        Returns:
            表格数据结果。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
            KeyError: 表格不存在时抛出。
        """

        normalized_ticker, normalized_document_id = self._normalize_document_identity(
            ticker=ticker,
            document_id=document_id,
            tool_name="get_table",
        )
        normalized_table_ref = require_non_empty_text(
            table_ref,
            empty_error=ToolArgumentError("get_table", "table_ref", table_ref, "Argument must not be empty"),
        )
        processor = self._get_or_create_processor(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        try:
            table_raw: TableContent = processor.read_table(normalized_table_ref)
        except KeyError as exc:
            raise ToolArgumentError(
                "get_table",
                "table_ref",
                normalized_table_ref,
                "表格不存在；请先调用 list_tables，并原样复制返回的 table_ref，不要简写、重编号或自造 table_ref",
            ) from exc
        data_payload = _build_table_data_payload(table_raw)

        # within_section：通过 get_section_title O(1) 获取所属章节信息
        section_ref = table_raw.get("section_ref")
        within_section: dict[str, str] | None = None
        if section_ref:
            within_section = {"ref": section_ref}
            sec_title = processor.get_section_title(section_ref)
            if sec_title:
                within_section["title"] = sec_title

        page_no = table_raw.get("page_no")
        caption = table_raw.get("caption")
        result: TableDetailResult = {
            "ticker": normalized_ticker,
            "document_id": normalized_document_id,
            "table_ref": normalized_table_ref,
            "data": data_payload,
            "row_count": int(table_raw.get("row_count", 0) or 0),
            "col_count": int(table_raw.get("col_count", 0) or 0),
            "is_financial": bool(table_raw.get("is_financial", False)),
            "table_type": _normalize_table_type(table_raw.get("table_type")),
            "citation": self._build_citation(
                ticker=normalized_ticker,
                document_id=normalized_document_id,
            ),
        }
        # 条件字段：省略 null 值减少序列化噪声
        if within_section:
            result["within_section"] = within_section
        if caption:
            result["caption"] = caption
        if isinstance(page_no, int) and page_no > 0:
            result["page_no"] = page_no
        return result

    def get_page_content(self, *, ticker: str, document_id: str, page_no: int) -> PageContentResult | NotSupportedResult:
        """读取页面上下文。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            page_no: 目标页码（1-based）。

        Returns:
            页面内容结果；不支持时返回 `not_supported` 结构。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
        """

        normalized_ticker, normalized_document_id = self._normalize_document_identity(
            ticker=ticker,
            document_id=document_id,
            tool_name="get_page_content",
        )
        if not isinstance(page_no, int) or page_no <= 0:
            raise ToolArgumentError(
                "get_page_content",
                "page_no",
                page_no,
                "page_no must be a positive integer",
            )

        processor = self._get_or_create_processor(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        page_method = getattr(processor, "get_page_content", None)
        if not callable(page_method):
            return _build_not_supported_result(
                ticker=normalized_ticker,
                document_id=normalized_document_id,
                feature="get_page_content",
                payload={"page_no": page_no, "supported": False},
            )

        page_payload = cast(dict[str, Any], page_method(page_no))
        # processor 贡献的子字段通过 .get() 提取；已知字段由 PageContentResult 声明。
        result: PageContentResult = {
            "ticker": normalized_ticker,
            "document_id": normalized_document_id,
            "page_no": page_no,
            "sections": list(page_payload.get("sections") or []),
            "tables": list(page_payload.get("tables") or []),
            "text_preview": str(page_payload.get("text_preview", "")),
            "has_content": bool(page_payload.get("has_content", False)),
            "total_items": int(page_payload.get("total_items", 0) or 0),
            "supported": bool(page_payload.get("supported", True)),
            "citation": self._build_citation(
                ticker=normalized_ticker,
                document_id=normalized_document_id,
            ),
        }
        return result

    def get_financial_statement(
        self,
        *,
        ticker: str,
        document_id: str,
        statement_type: str,
    ) -> FinancialStatementResult | NotSupportedResult:
        """读取标准财务报表。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            statement_type: 报表类型。

        Returns:
            财务报表结果；成功时除标准报表数据外，还包含 `statement_locator`
            结构化定位信息，供写作链路生成可复核的“证据与出处”锚点；
            不支持时返回 `not_supported` 结构。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
        """

        normalized_ticker, normalized_document_id = self._normalize_document_identity(
            ticker=ticker,
            document_id=document_id,
            tool_name="get_financial_statement",
        )
        normalized_statement_type = require_non_empty_text(
            statement_type,
            empty_error=ToolArgumentError(
                "get_financial_statement",
                "statement_type",
                statement_type,
                "Argument must not be empty",
            ),
        )

        processor = self._get_or_create_processor(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        statement_method = getattr(processor, "get_financial_statement", None)
        if not callable(statement_method):
            return _build_not_supported_result(
                ticker=normalized_ticker,
                document_id=normalized_document_id,
                feature="get_financial_statement",
                payload={"statement_type": normalized_statement_type},
            )

        statement_payload = cast(dict[str, Any], statement_method(normalized_statement_type))
        citation = self._build_citation(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        # processor 贡献的字段（statement_type, rows, currency 等）通过 spread 合入；
        # 已知字段由 FinancialStatementResult TypedDict 声明，运行时由 processor 保证。
        result: dict[str, Any] = {
            "ticker": normalized_ticker,
            "document_id": normalized_document_id,
            **statement_payload,
            "citation": citation,
        }
        return cast(FinancialStatementResult, result)

    def query_xbrl_facts(
        self,
        *,
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
            concepts: 可选 XBRL 概念列表。为空时按文档 form/taxonomy 选择默认概念包。
            statement_type: 可选报表类型。
            period_end: 可选期末日期。
            fiscal_year: 可选财年。
            fiscal_period: 可选财期。
            min_value: 可选最小值。
            max_value: 可选最大值。

        Returns:
            XBRL 数值 facts 查询结果；不支持时返回 `not_supported` 结构。

        Raises:
            ToolArgumentError: 参数非法时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
        """

        normalized_ticker, normalized_document_id = self._normalize_document_identity(
            ticker=ticker,
            document_id=document_id,
            tool_name="query_xbrl_facts",
        )
        if concepts is not None and not isinstance(concepts, list):
            raise ToolArgumentError(
                "query_xbrl_facts",
                "concepts",
                concepts,
                "concepts must be a string array or omitted",
            )
        normalized_concepts = [
            item
            for item in (normalize_optional_text(concept) for concept in (concepts or []))
            if item is not None
        ]

        form_type = self._resolve_document_form_type(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        processor = self._get_or_create_processor(
            ticker=normalized_ticker,
            document_id=normalized_document_id,
        )
        taxonomy = _resolve_processor_taxonomy(processor)
        resolved_concepts = (
            normalized_concepts
            if normalized_concepts
            else _resolve_default_xbrl_concepts(form_type=form_type, taxonomy=taxonomy)
        )
        query_method = getattr(processor, "query_xbrl_facts", None)
        if not callable(query_method):
            return _build_not_supported_result(
                ticker=normalized_ticker,
                document_id=normalized_document_id,
                feature="query_xbrl_facts",
                payload={"concepts": resolved_concepts},
            )

        payload = cast(
            dict[str, Any],
            query_method(
                concepts=resolved_concepts,
                statement_type=normalize_optional_text(statement_type),
                period_end=normalize_optional_text(period_end),
                fiscal_year=fiscal_year,
                fiscal_period=normalize_optional_text(fiscal_period),
                min_value=min_value,
                max_value=max_value,
            ),
        )
        normalized_payload = _normalize_xbrl_query_payload(
            payload=payload,
            default_concepts=resolved_concepts,
        )
        # processor 贡献的字段（query_params, facts, total 等）通过 spread 合入；
        # 已知字段由 XbrlQueryResult TypedDict 声明，运行时由 normalizer 保证。
        result: dict[str, Any] = {
            "ticker": normalized_ticker,
            "document_id": normalized_document_id,
            **normalized_payload,
            "citation": self._build_citation(
                ticker=normalized_ticker,
                document_id=normalized_document_id,
            ),
        }
        return cast(XbrlQueryResult, result)

    def _normalize_document_identity(
        self,
        *,
        ticker: str,
        document_id: str,
        tool_name: str,
    ) -> tuple[str, str]:
        """标准化文档身份参数。

        Args:
            ticker: 原始股票代码。
            document_id: 原始文档 ID。
            tool_name: 调用工具名。

        Returns:
            `(normalized_ticker, normalized_document_id)`。其中 ``normalized_document_id``
            始终是仓储可识别的规范 `document_id`。

        Raises:
            ToolArgumentError: 参数为空时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
        """

        normalized_ticker = self._resolve_canonical_ticker(ticker=ticker, tool_name=tool_name)
        normalized_document_id = require_non_empty_text(
            document_id,
            empty_error=ToolArgumentError(
                tool_name,
                "document_id",
                document_id,
                "Argument must not be empty",
            ),
        )
        resolved_document_id = self._resolve_canonical_document_id(
            ticker=normalized_ticker,
            raw_document_id=normalized_document_id,
            tool_name=tool_name,
        )
        return normalized_ticker, resolved_document_id

    def _resolve_canonical_ticker(self, *, ticker: str, tool_name: str) -> str:
        """将外部 ticker 归一化为可用 ticker。

        解析顺序：
        1. ``require_non_empty_text`` 拒绝空输入。
        2. 走 ``try_normalize_ticker`` 真源把 ``0700.HK`` / ``600519.SH`` 等
           常见变形归一化为 canonical；作为唯一查询候选。
        3. 若真源识别失败（例如用户传了 ``"Apple Inc."`` 这种公司名），回退到
           ``strip().upper()`` 作为候选；保留"公司名可当 ticker 传"的既有行为。
        4. 仓储 ``resolve_existing_ticker`` 在 canonical 未命中时会走公司级
           ``ticker_aliases`` 索引反查；alias 已全部归一化，无需再构造变体。

        Args:
            ticker: 原始 ticker。
            tool_name: 当前调用工具名。

        Returns:
            当前财报工具可用的 ticker。

        Raises:
            ToolArgumentError: ticker 为空时抛出。
            ToolBusinessError: ticker 未收录于当前工作区时抛出。
        """

        normalized_ticker = require_non_empty_text(
            ticker,
            empty_error=ToolArgumentError(
                tool_name,
                "ticker",
                ticker,
                "Argument must not be empty",
            ),
        )
        normalized_source = try_normalize_ticker(normalized_ticker)
        if normalized_source is not None:
            probe_ticker = normalized_source.canonical
        else:
            probe_ticker = normalized_ticker.strip().upper()
        resolved_ticker = self._company_repository.resolve_existing_ticker([probe_ticker])
        if resolved_ticker is None:
            raise ToolBusinessError(
                code=ErrorCode.NOT_FOUND.value,
                message=f"Financial Document Tools do not have this company: ticker='{normalized_ticker}'.",
                hint=_MISSING_TICKER_HINT,
            )
        if resolved_ticker != normalized_ticker:
            Log.debug(
                f"ticker 已归一化: tool={tool_name} raw={normalized_ticker!r} "
                f"probe={probe_ticker!r} canonical={resolved_ticker!r}",
                module=self.MODULE,
            )
        return resolved_ticker

    def _resolve_canonical_document_id(
        self,
        *,
        ticker: str,
        raw_document_id: str,
        tool_name: str,
    ) -> str:
        """将外部传入的文档标识归一化为仓储 `document_id`。

        这里仅依赖仓储公开元数据做最小归一化，不依赖 processor 内部实现。
        支持以下几类输入：
        - 已经是仓储 `document_id`
        - `meta.json` 中的 `internal_document_id`
        - `meta.json` 中的 `accession_number`
        - 去掉连字符后的 accession

        Args:
            ticker: 标准化股票代码。
            raw_document_id: 外部传入的文档标识。
            tool_name: 当前工具名，仅用于日志。

        Returns:
            仓储规范 `document_id`。

        Raises:
            无。
        """

        direct_meta = self._get_document_meta_cached(ticker, raw_document_id)
        if direct_meta is not None:
            return raw_document_id

        normalized_alias = re.sub(r"\s+", "", raw_document_id).strip()
        for source_kind in (SourceKind.FILING, SourceKind.MATERIAL):
            for candidate_document_id in self._source_repository.list_source_document_ids(ticker, source_kind):
                candidate_meta = self._get_document_meta_cached(ticker, candidate_document_id)
                if not candidate_meta:
                    continue
                alias_fields = self._build_document_identity_aliases(
                    candidate_document_id=candidate_document_id,
                    meta=candidate_meta,
                )
                if normalized_alias not in alias_fields:
                    continue
                matched_field = alias_fields[normalized_alias]
                if matched_field != "document_id":
                    Log.debug(
                        f"文档标识已归一化: tool={tool_name} ticker={ticker} raw={raw_document_id!r} "
                        f"matched_field={matched_field} canonical={candidate_document_id!r}",
                        module=self.MODULE,
                    )
                return candidate_document_id

        Log.debug(
            f"文档标识未命中归一化映射: tool={tool_name} ticker={ticker} raw={raw_document_id!r}",
            module=self.MODULE,
        )
        return raw_document_id

    def _build_document_identity_aliases(
        self,
        *,
        candidate_document_id: str,
        meta: Mapping[str, Any],
    ) -> dict[str, str]:
        """构建单个文档可接受的身份别名集合。

        Args:
            candidate_document_id: 仓储规范 `document_id`。
            meta: 对应 `meta.json` 内容。

        Returns:
            `alias -> matched_field` 映射。

        Raises:
            无。
        """

        aliases: dict[str, str] = {
            re.sub(r"\s+", "", candidate_document_id).strip(): "document_id",
        }
        for field_name in ("internal_document_id", "accession_number"):
            raw_value = meta.get(field_name)
            normalized_value = normalize_optional_text(raw_value)
            if not normalized_value:
                continue
            aliases[re.sub(r"\s+", "", normalized_value).strip()] = field_name
            aliases[normalized_value.replace("-", "")] = field_name
        return aliases

    def _collect_source_documents(self, ticker: str) -> list[dict[str, Any]]:
        """汇总 source 层文档摘要。

        Args:
            ticker: 标准化股票代码。

        Returns:
            文档摘要列表。

        Raises:
            RuntimeError: 仓储读取失败时抛出。
        """

        documents: list[dict[str, Any]] = []
        documents.extend(self._collect_source_documents_by_kind(ticker, SourceKind.FILING))
        documents.extend(self._collect_source_documents_by_kind(ticker, SourceKind.MATERIAL))
        documents.sort(key=build_document_recency_sort_key, reverse=True)
        return documents

    def _collect_source_documents_by_kind(
        self,
        ticker: str,
        source_kind: SourceKind,
    ) -> list[dict[str, Any]]:
        """按来源类型采集文档摘要。

        Args:
            ticker: 标准化股票代码。
            source_kind: 文档来源。

        Returns:
            文档摘要列表。

        Raises:
            RuntimeError: 仓储读取失败时抛出。
        """

        document_ids = self._source_repository.list_source_document_ids(ticker, source_kind)
        results: list[dict[str, Any]] = []
        for document_id in document_ids:
            try:
                meta = self._source_repository.get_source_meta(ticker, document_id, source_kind)
            except FileNotFoundError:
                continue
            if bool(meta.get("is_deleted", False)):
                continue
            if not bool(meta.get("ingest_complete", True)):
                continue
            inferred_period = _infer_fiscal_period(meta)
            inferred_year = _infer_fiscal_year(meta, inferred_period)
            resolved_fiscal_year = _resolve_fiscal_year_with_fallback(
                raw_value=meta.get("fiscal_year"),
                inferred_year=inferred_year,
            )
            resolved_fiscal_period = _resolve_fiscal_period_with_fallback(
                raw_value=meta.get("fiscal_period"),
                inferred_period=inferred_period,
            )
            # 从 processed meta 读取能力标志（轻量 JSON），处理缺失的情况
            has_financial_data = self._read_capability_flags(
                ticker, document_id,
            )
            results.append(
                {
                    "document_id": document_id,
                    "source_kind": source_kind.value,
                    "form_type": _normalize_form_type_for_matching(meta.get("form_type")),
                    "material_name": meta.get("material_name"),
                    "fiscal_year": resolved_fiscal_year,
                    "fiscal_period": resolved_fiscal_period,
                    "report_date": meta.get("report_date"),
                    "filing_date": meta.get("filing_date"),
                    "amended": bool(meta.get("amended", False)),
                    "has_financial_data": has_financial_data,
                }
            )
        return results

    def _build_citation(
        self,
        *,
        ticker: str,
        document_id: str,
        item: Optional[str] = None,
        heading: Optional[str] = None,
    ) -> dict[str, Any]:
        """构建统一 citation 对象。

        从 meta.json 读取文档元数据，构建可序列化的 citation 字典。
        同一 (ticker, document_id) 的 meta 读取会被 _get_document_meta_cached 缓存。

        Args:
            ticker: 标准化股票代码。
            document_id: 标准化文档 ID。
            item: 可选 Item 编号（如 "Item 1A"）。
            heading: 可选章节标题。

        Returns:
            citation 字典（值为 None 的键已移除）。
        """
        meta = self._get_document_meta_cached(ticker, document_id)
        source_kind = normalize_optional_text(meta.get("source_kind")) if meta else None
        # 推断来源类型
        if source_kind == SourceKind.MATERIAL.value:
            source_type = SourceType.SUPPLEMENTARY.value
        elif document_id.startswith("fil_"):
            # 美股 filing: document_id = fil_{accession_number}
            ingest_method = meta.get("ingest_method") if meta else None
            source_type = SourceType.SEC_EDGAR.value if ingest_method == "download" else SourceType.UPLOADED.value
        else:
            source_type = SourceType.UPLOADED.value

        form_type = _normalize_form_type_for_matching(meta.get("form_type")) if meta else None
        # 美股 filing 的 accession_number 存储在 meta.json 中
        accession_no = normalize_optional_text(meta.get("accession_number")) if meta else None

        citation = Citation(
            source_type=source_type,
            document_id=document_id,
            ticker=ticker,
            form_type=form_type,
            filing_date=normalize_optional_text(meta.get("filing_date")) if meta else None,
            accession_no=accession_no,
            fiscal_year=meta.get("fiscal_year") if meta else None,
            fiscal_period=normalize_optional_text(meta.get("fiscal_period")) if meta else None,
            item=item,
            heading=heading,
        )
        return citation.to_dict()

    def _get_document_meta_cached(self, ticker: str, document_id: str) -> Optional[dict[str, Any]]:
        """读取文档元数据（带实例级缓存）。

        同一 FinsToolService 实例内，对相同 (ticker, document_id) 的
        meta.json 读取做内存缓存，避免同一次工具调用链中重复 IO。

        Args:
            ticker: 标准化股票代码。
            document_id: 标准化文档 ID。

        Returns:
            meta 字典；文档不存在时返回 None。
        """
        cache_key = (ticker, document_id)
        if cache_key in self._meta_cache:
            return self._meta_cache[cache_key]
        try:
            source_kind = self._resolve_source_kind(ticker=ticker, document_id=document_id)
            meta = self._source_repository.get_source_meta(ticker, document_id, source_kind)
        except FileNotFoundError:
            meta = None
        self._meta_cache[cache_key] = meta
        return meta

    def _enrich_sections_with_semantic(
        self,
        sections: list[SectionSummary],
        form_type: Optional[str],
    ) -> list[dict[str, Any]]:
        """为章节列表注入语义层字段。

        遍历 sections，为每个章节解析 item/topic/path，
        并构建 ref → section 索引以便通过 parent_ref 追溯路径。

        Args:
            sections: processor 返回的章节摘要列表。
            form_type: 文档的 form_type。

        Returns:
            增强后的章节字典列表。
        """
        # 构建 ref → section 索引，用于 parent_ref 追溯
        ref_to_section: dict[str, SectionSummary] = {}
        for sec in sections:
            ref = sec.get("ref")
            if ref:
                ref_to_section[ref] = sec

        enriched: list[dict[str, Any]] = []
        # 记录已解析的 ref → (item_number, topic)，供子章节继承使用
        ref_to_resolved: dict[str, tuple[Optional[str], Optional[str]]] = {}
        for sec in sections:
            entry = dict(sec)
            # 移除 preview 字段：与 title 高度重复，LLM 需要详情时用 read_section
            entry.pop("preview", None)
            title = sec.get("title")
            parent_ref = sec.get("parent_ref")

            # 获取父章节标题（用于 10-Q Part 消歧）
            parent_title = None
            if parent_ref and parent_ref in ref_to_section:
                parent_title = ref_to_section[parent_ref].get("title")

            item_number, canonical_title, topic = resolve_section_semantic(
                title=title,
                form_type=form_type,
                parent_title=parent_title,
            )

            # 子章节无法自解析时，从父章节继承 item/topic（支持多级辭传）
            if (item_number is None or topic is None) and parent_ref and parent_ref in ref_to_resolved:
                parent_item_number, parent_topic = ref_to_resolved[parent_ref]
                if item_number is None:
                    item_number = parent_item_number
                if topic is None:
                    topic = parent_topic

            # 记录当前解析结果，供下级子章节查表
            ref = sec.get("ref")
            if ref:
                ref_to_resolved[ref] = (item_number, topic)

            # 构建层级路径：上溯 parent_ref 链收集父标题
            parent_titles = _collect_parent_titles(sec, ref_to_section)
            path = build_section_path(
                form_type=form_type,
                item_number=item_number,
                canonical_title=canonical_title,
                section_title=title,
                parent_titles=parent_titles,
            )

            entry["item"] = f"Item {item_number}" if item_number else None
            entry["topic"] = topic
            # 只为顶层章节保留路径（子章节层级关系已由 parent_ref 表达）
            if sec.get("level", 0) <= 1:
                entry["path"] = path if path else None
            enriched.append(entry)
        return enriched

    def _resolve_document_form_type(self, *, ticker: str, document_id: str) -> Optional[str]:
        """读取文档 form_type。

        Args:
            ticker: 标准化股票代码。
            document_id: 标准化文档 ID。

        Returns:
            标准化后的 form_type；读取失败时返回 `None`。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        meta = self._get_document_meta_cached(ticker, document_id)
        if meta is None:
            return None
        return _normalize_form_type_for_matching(meta.get("form_type"))

    def _read_company_info(self, ticker: str) -> tuple[str, str]:
        """读取公司信息。

        Args:
            ticker: 标准化股票代码。

        Returns:
            `(company_name, market)`。

        Raises:
            RuntimeError: 仓储读取失败时抛出。
        """

        try:
            company_meta = self._company_repository.get_company_meta(ticker)
        except FileNotFoundError:
            return ticker, "unknown"
        return company_meta.company_name, company_meta.market

    def _read_capability_flags(
        self,
        ticker: str,
        document_id: str,
    ) -> Optional[bool]:
        """从 processed meta 读取文档财务数据能力标志。

        Args:
            ticker: 标准化股票代码。
            document_id: 文档 ID。

        Returns:
            `has_financial_data`：`True`（可调用 get_financial_statement）/
            `False`（无数据）/ `None`（未处理或无法判定）。

        Raises:
            无（内部异常已捕获）。
        """

        try:
            processed_meta = self._processed_repository.get_processed_meta(ticker, document_id)
        except (FileNotFoundError, ValueError):
            return None
        return resolve_has_financial_data(
            has_financial_data=processed_meta.get("has_financial_data"),
            availability=processed_meta.get("financial_statement_availability"),
            has_financial_statement=processed_meta.get("has_financial_statement"),
            has_xbrl=processed_meta.get("has_xbrl"),
            has_structured_financial_statements=processed_meta.get("has_structured_financial_statements"),
            has_financial_statement_sections=processed_meta.get("has_financial_statement_sections"),
        )

    def _get_or_create_processor(self, *, ticker: str, document_id: str) -> DocumentProcessor:
        """读取或创建 Processor 实例。

        Args:
            ticker: 标准化股票代码。
            document_id: 标准化文档 ID。

        Returns:
            Processor 实例。

        Raises:
            FileNotFoundError: 文档不存在时抛出。
            ValueError: 未匹配处理器时抛出。
        """

        cache_key = ProcessorCacheKey(ticker=ticker, document_id=document_id)
        cached = self._processor_cache.get(cache_key)
        if cached is not None:
            return cached

        lock = self._get_creation_lock(cache_key)
        with lock:
            # 复杂逻辑说明：并发线程在锁内二次检查，避免重复构建 Processor。
            cached = self._processor_cache.get(cache_key)
            if cached is not None:
                return cached
            processor = self._create_processor(ticker=ticker, document_id=document_id)
            self._processor_cache.put(cache_key, processor)
            Log.debug(
                f"processor 已创建并缓存: ticker={ticker} document_id={document_id} type={type(processor).__name__}",
                module=self.MODULE,
            )
            return processor

    def _create_processor(self, *, ticker: str, document_id: str) -> DocumentProcessor:
        """创建 Processor 实例。

        Args:
            ticker: 标准化股票代码。
            document_id: 标准化文档 ID。

        Returns:
            Processor 实例。

        Raises:
            FileNotFoundError: 文档不存在时抛出。
            ValueError: 未匹配处理器时抛出。
            RuntimeError: 候选处理器全部创建失败时抛出。
        """

        source_kind = self._resolve_source_kind(ticker=ticker, document_id=document_id)
        source = self._source_repository.get_primary_source(
            ticker=ticker,
            document_id=document_id,
            source_kind=source_kind,
        )
        source_meta = self._source_repository.get_source_meta(ticker, document_id, source_kind)
        form_type = normalize_optional_text(source_meta.get("form_type"))
        return self._processor_registry.create_with_fallback(
            source=source,
            form_type=form_type,
            media_type=getattr(source, "media_type", None),
        )

    def _resolve_source_kind(self, *, ticker: str, document_id: str) -> SourceKind:
        """解析文档来源类型。

        Args:
            ticker: 标准化股票代码。
            document_id: 标准化文档 ID。

        Returns:
            来源类型。

        Raises:
            FileNotFoundError: 当文档既不在 filing 也不在 material 中时抛出。
        """

        try:
            self._source_repository.get_source_handle(ticker, document_id, SourceKind.FILING)
            return SourceKind.FILING
        except FileNotFoundError:
            pass
        try:
            self._source_repository.get_source_handle(ticker, document_id, SourceKind.MATERIAL)
            return SourceKind.MATERIAL
        except FileNotFoundError:
            pass
        raise FileNotFoundError(f"Document not found: ticker={ticker}, document_id={document_id}")

    def _get_creation_lock(self, cache_key: ProcessorCacheKey) -> Lock:
        """读取或创建文档级构建锁。

        Args:
            cache_key: Processor 缓存键。

        Returns:
            文档级互斥锁。

        Raises:
            RuntimeError: 锁表访问失败时抛出。
        """

        with self._creation_locks_guard:
            lock = self._creation_locks.get(cache_key)
            if lock is not None:
                return lock
            created = Lock()
            self._creation_locks[cache_key] = created
            return created
