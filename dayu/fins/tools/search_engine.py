"""搜索引擎核心逻辑。

该模块实现文档搜索的完整流水线：
- 查询诊断（歧义度、意图分类）
- 自适应搜索计划生成
- 查询扩展（短语变体、同义词、token 回退）
- 意图过滤与语义桶匹配
- 排序（策略优先级 → 意图一致性 → 噪音惩罚 → BM25F → 邻近度）
- 去重与证据化结构构建

所有函数均为模块级私有函数，由 FinsToolService 调用。
"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, Optional

from dayu.engine.exceptions import ToolArgumentError
from dayu.engine.processors.base import (
    DocumentProcessor,
    SearchHit,
)
from .bm25f_scorer import BM25FSectionIndex, score_search_entry_bm25f
from .search_models import (
    QueryDiagnosis,
    SectionSemanticProfile,
    SearchPlan,
    SEARCH_MODE_AUTO,
    SEARCH_MODE_EXACT,
    SEARCH_MODE_KEYWORD,
    SEARCH_MODE_SEMANTIC,
    _VALID_SEARCH_MODES,
    _TOKEN_STOP_WORDS,
    _SEARCH_SYNONYM_GROUPS,
    _GENERIC_AMBIGUOUS_TOKENS,
    _INTENT_KEYWORDS,
    _NOISE_CONTEXT_TOKENS_BY_INTENT,
    _SUPPORT_CONTEXT_TOKENS_BY_INTENT,
    _SEARCH_STRATEGY_EXACT,
    _SEARCH_STRATEGY_PHRASE_VARIANT,
    _SEARCH_STRATEGY_SYNONYM,
    _SEARCH_STRATEGY_TOKEN,
    _SEARCH_STRATEGY_PRIORITY,
    _WORD_SPLIT_PATTERN,
    _SPACE_NORMALIZE_PATTERN,
    _TOPIC_TO_BUCKET,
    _BUCKET_KEYWORD_SIGNALS,
    _EXPECTED_BUCKETS_BY_INTENT,
)
from .section_semantic import resolve_section_semantic
from dayu.fins._converters import normalize_optional_text, require_non_empty_text


# =====================================================================
# 搜索匹配标准化
# =====================================================================

def _normalize_search_matches(matches_raw: list[SearchHit]) -> list[dict[str, Any]]:
    """标准化搜索命中结构。

    支持两种命中模式：
    - 传统 snippet 模式：命中包含 snippet 字段。
    - 证据模式：命中包含 evidence 字段。

    Args:
        matches_raw: 处理器原始命中列表。

    Returns:
        标准化命中列表。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized_matches: list[dict[str, Any]] = []
    for hit in matches_raw:
        page_no = hit.get("page_no")
        entry: dict[str, Any] = {
            "section_ref": hit.get("section_ref"),
            "section_title": hit.get("section_title"),
            "page_no": page_no if isinstance(page_no, int) and page_no > 0 else None,
        }
        # 透传 processor 层的 token_fallback 标记，供上游区分精确/回退命中
        if hit.get("_token_fallback"):
            entry["_token_fallback"] = True
        # 保留 evidence 结构（如果处理器返回了证据化命中）
        evidence = hit.get("evidence")
        if isinstance(evidence, dict):
            entry["evidence"] = evidence
        else:
            entry["snippet"] = str(hit.get("snippet", ""))
        normalized_matches.append(entry)
    return normalized_matches


# =====================================================================
# 搜索模式与查询参数校验
# =====================================================================

def _resolve_search_mode(mode: Optional[str]) -> str:
    """校验并规范化搜索模式参数。

    Args:
        mode: 原始模式参数，None 时默认为 ``auto``。

    Returns:
        规范化后的搜索模式字符串。

    Raises:
        ToolArgumentError: 模式值无效时抛出。
    """

    if mode is None:
        return SEARCH_MODE_AUTO
    normalized = str(mode).strip().lower()
    if not normalized:
        return SEARCH_MODE_AUTO
    if normalized not in _VALID_SEARCH_MODES:
        raise ToolArgumentError(
            "search_document",
            "mode",
            mode,
            f"无效值，允许值为 {sorted(_VALID_SEARCH_MODES)}",
        )
    return normalized


def _build_empty_search_strategy_hit_counts() -> dict[str, int]:
    """构建搜索策略命中统计字典。

    Args:
        无。

    Returns:
        初始化后的策略命中计数字典。

    Raises:
        无。
    """

    return {
        _SEARCH_STRATEGY_EXACT: 0,
        _SEARCH_STRATEGY_PHRASE_VARIANT: 0,
        _SEARCH_STRATEGY_SYNONYM: 0,
        _SEARCH_STRATEGY_TOKEN: 0,
    }


def _resolve_search_queries(
    *,
    query: Optional[str],
    queries: Optional[list[str]],
    max_queries: int,
) -> list[str]:
    """校验并解析 query / queries 互斥参数，返回标准化查询列表。

    Args:
        query: 单条查询。
        queries: 批量查询。
        max_queries: 批量查询上限。

    Returns:
        非空的标准化查询列表。

    Raises:
        ToolArgumentError: 两项同时指定、均未指定或超上限时抛出。
    """

    has_query = query is not None and str(query).strip() != ""
    has_queries = queries is not None and len(queries) > 0

    if has_query and has_queries:
        raise ToolArgumentError(
            "search_document",
            "query/queries",
            None,
            "Cannot specify both 'query' and 'queries'. Use one or the other.",
        )
    if not has_query and not has_queries:
        raise ToolArgumentError(
            "search_document",
            "query/queries",
            None,
            "Must specify either 'query' or 'queries'.",
        )

    if has_query:
        normalized = require_non_empty_text(
            query,
            empty_error=ToolArgumentError(
                "search_document",
                "query",
                query,
                "Argument must not be empty",
            ),
        )
        return [normalized]

    # 批量路径：去空值、去重、校验上限
    assert queries is not None
    seen: set[str] = set()
    result: list[str] = []
    for item in queries:
        normalized = normalize_optional_text(item)
        if normalized is None:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)

    if not result:
        raise ToolArgumentError(
            "search_document",
            "queries",
            queries,
            "'queries' must contain at least one non-empty string.",
        )
    if len(result) > max_queries:
        raise ToolArgumentError(
            "search_document",
            "queries",
            None,
            f"'queries' exceeds maximum of {max_queries} items (got {len(result)}).",
        )
    return result


# =====================================================================
# 章节语义画像
# =====================================================================

def _build_section_semantic_profiles(
    sections: list[dict[str, Any]],
) -> tuple[dict[str, SectionSemanticProfile], dict[str, int]]:
    """构建章节语义画像与查询词文档频次。

    Args:
        sections: 语义增强后的章节列表。

    Returns:
        ``(semantic_profiles, term_document_frequency)``。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    profiles: dict[str, SectionSemanticProfile] = {}
    term_df: Counter[str] = Counter()
    for section in sections:
        section_ref = str(section.get("ref") or "").strip()
        if not section_ref:
            continue
        topic = str(section.get("topic") or "").strip().lower()
        path = str(section.get("path") or "").strip()
        title = str(section.get("title") or "").strip()
        item = str(section.get("item") or "").strip()
        preview = str(section.get("preview") or "").strip()
        bucket = _resolve_semantic_bucket(topic=topic, path=path, title=title, item=item)
        lexical_text = " ".join([title, item, topic, path, preview]).lower()
        lexical_tokens = tuple(_extract_ascii_tokens(lexical_text))
        profiles[section_ref] = SectionSemanticProfile(
            section_ref=section_ref,
            topic=topic,
            path=path,
            title=title,
            item=item,
            bucket=bucket,
            lexical_tokens=lexical_tokens,
        )
        term_df.update(set(lexical_tokens))
    return profiles, dict(term_df)


def _resolve_semantic_bucket(*, topic: str, path: str, title: str, item: str) -> str:
    """根据章节语义字段归一化语义桶（自适应方案）。

    采用两级决策：

    1. **Topic 直连**（一级）：topic 在 ``_TOPIC_TO_BUCKET`` 映射表中直接返回。
       覆盖所有 ``SectionType`` 值，新增 SectionType 只需加一行映射。
    2. **关键词评分**（二级 fallback）：topic 未命中时，将 path / title / item
       拆词后与 ``_BUCKET_KEYWORD_SIGNALS`` 各桶关键词集求交集计分，
       取最高分桶。全部为零时返回 ``"other"``。

    该设计避免了 if-else 链的顺序依赖和子串误匹配问题。

    Args:
        topic: 章节 topic（来自 ``section_semantic.py`` 的 ``SectionType.value``）。
        path: 章节层级路径。
        title: 章节标题。
        item: 章节 Item 编号。

    Returns:
        语义桶名称（business / risk / financial / governance / people / legal / other）。
    """

    # 一级：topic 直连（最可靠信号，O(1) 查表）
    bucket = _TOPIC_TO_BUCKET.get(topic.lower().strip())
    if bucket is not None:
        return bucket

    # 二级：关键词评分 fallback（适用于无 topic 或自定义 topic 的章节）
    text = f"{path} {title} {item}".lower()
    words = frozenset(_WORD_SPLIT_PATTERN.findall(text))
    best_bucket = "other"
    best_score = 0
    for candidate, keywords in _BUCKET_KEYWORD_SIGNALS.items():
        score = len(words & keywords)
        if score > best_score:
            best_score = score
            best_bucket = candidate
    return best_bucket


# =====================================================================
# 查询诊断与意图分类
# =====================================================================

def _diagnose_search_query(
    *,
    query: str,
    term_document_frequency: dict[str, int],
    document_count: int,
    mode: str,
) -> QueryDiagnosis:
    """诊断查询的歧义度和意图，用于自适应检索。

    Args:
        query: 标准化查询词。
        term_document_frequency: 词项文档频次。
        document_count: 文档章节数。
        mode: 搜索模式。

    Returns:
        QueryDiagnosis 结构体。

    Raises:
        RuntimeError: 诊断失败时抛出。
    """

    tokens = tuple(_extract_ascii_tokens(query.lower()))
    token_count = len(tokens)
    if token_count == 0:
        return QueryDiagnosis(
            query=query,
            tokens=tokens,
            token_count=0,
            ambiguity_score=0.0,
            is_high_ambiguity=False,
            intent="general",
            allow_direct_token_fallback=True,
        )

    generic_hits = sum(1 for token in tokens if token in _GENERIC_AMBIGUOUS_TOKENS)
    generic_ratio = generic_hits / token_count
    df_ratio = 0.0
    for token in tokens:
        token_df = term_document_frequency.get(token, 0)
        if document_count > 0:
            df_ratio += min(1.0, token_df / document_count)
    df_ratio = df_ratio / token_count
    short_query_factor = 1.0 if token_count <= 2 else 0.0
    ambiguity_score = round((generic_ratio + df_ratio + short_query_factor) / 3.0, 4)
    is_high_ambiguity = ambiguity_score >= 0.62
    intent = _classify_query_intent(tokens)
    allow_direct_token_fallback = not (mode == SEARCH_MODE_AUTO and is_high_ambiguity)

    return QueryDiagnosis(
        query=query,
        tokens=tokens,
        token_count=token_count,
        ambiguity_score=ambiguity_score,
        is_high_ambiguity=is_high_ambiguity,
        intent=intent,
        allow_direct_token_fallback=allow_direct_token_fallback,
    )


def _classify_query_intent(tokens: tuple[str, ...]) -> str:
    """根据 token 估计 query 意图。

    Args:
        tokens: 查询 token 列表。

    Returns:
        意图名称。

    Raises:
        RuntimeError: 分类失败时抛出。
    """

    if not tokens:
        return "general"
    scored: dict[str, int] = {}
    token_set = set(tokens)
    for intent, keywords in _INTENT_KEYWORDS.items():
        scored[intent] = len(token_set.intersection(keywords))
    best_intent = "general"
    best_score = 0
    for intent, score in scored.items():
        if score > best_score:
            best_intent = intent
            best_score = score
    return best_intent


# =====================================================================
# 自适应搜索计划
# =====================================================================

def _build_adaptive_search_plan(
    *,
    query: str,
    mode: str,
    diagnosis: QueryDiagnosis,
) -> SearchPlan:
    """根据 query 诊断生成搜索执行计划。

    Args:
        query: 标准化查询词。
        mode: 搜索模式。
        diagnosis: 查询诊断结果。

    Returns:
        SearchPlan。

    Raises:
        RuntimeError: 生成失败时抛出。
    """

    run_exact = mode in (SEARCH_MODE_AUTO, SEARCH_MODE_EXACT)
    run_expansion = mode in (SEARCH_MODE_AUTO, SEARCH_MODE_KEYWORD, SEARCH_MODE_SEMANTIC)
    if not run_expansion:
        return SearchPlan(
            run_exact=run_exact,
            expansion_phases=(),
            fallback_gated=False,
            scoped_before_token=False,
        )

    expansions = _build_search_query_expansions(query, mode=mode)
    if mode == SEARCH_MODE_KEYWORD:
        return SearchPlan(
            run_exact=False,
            expansion_phases=(tuple(expansions),),
            fallback_gated=False,
            scoped_before_token=False,
        )

    if mode == SEARCH_MODE_AUTO and diagnosis.is_high_ambiguity:
        non_token = [item for item in expansions if item.get("strategy") != _SEARCH_STRATEGY_TOKEN]
        token_only = [item for item in expansions if item.get("strategy") == _SEARCH_STRATEGY_TOKEN]
        phases: list[tuple[dict[str, str], ...]] = []
        if non_token:
            phases.append(tuple(non_token))
        if token_only:
            phases.append(tuple(token_only))
        return SearchPlan(
            run_exact=True,
            expansion_phases=tuple(phases),
            fallback_gated=True,
            scoped_before_token=True,
        )

    return SearchPlan(
        run_exact=run_exact,
        expansion_phases=(tuple(expansions),),
        fallback_gated=False,
        scoped_before_token=False,
    )


# =====================================================================
# 意图过滤
# =====================================================================

def _filter_matches_by_intent(
    *,
    matches: list[dict[str, Any]],
    diagnosis: QueryDiagnosis,
    semantic_profiles: dict[str, SectionSemanticProfile],
) -> list[dict[str, Any]]:
    """按 query 意图过滤命中集合。

    Args:
        matches: 原始命中列表。
        diagnosis: 查询诊断结果。
        semantic_profiles: 章节语义画像。

    Returns:
        过滤后的命中列表，若无匹配则返回空列表。

    Raises:
        RuntimeError: 过滤失败时抛出。
    """

    expected_buckets = _expected_buckets_for_intent(diagnosis.intent)
    if not expected_buckets:
        return matches
    filtered: list[dict[str, Any]] = []
    for match in matches:
        section_ref = str(match.get("section_ref") or "").strip()
        profile = semantic_profiles.get(section_ref)
        if profile is None:
            continue
        if profile.bucket in expected_buckets:
            filtered.append(match)
    return filtered


def _expected_buckets_for_intent(intent: str) -> set[str]:
    """返回 query intent 对应的语义桶优先集合。

    数据驱动：由 ``_EXPECTED_BUCKETS_BY_INTENT`` 表提供映射。
    新增意图只需扩展该表，无需修改此函数。
    """

    return set(_EXPECTED_BUCKETS_BY_INTENT.get(intent, frozenset()))


# =====================================================================
# 查询执行
# =====================================================================

def _execute_query_search(
    *,
    processor: "DocumentProcessor",
    query: str,
    within_ref: Optional[str],
    mode: str,
    diagnosis: QueryDiagnosis,
    semantic_profiles: dict[str, SectionSemanticProfile],
) -> tuple[list[dict[str, Any]], dict[str, int], list[dict[str, Any]], list[dict[str, str]]]:
    """对单条查询执行搜索策略，返回原始 ranked_entries 及诊断数据。

    Args:
        processor: 文档处理器。
        query: 标准化查询词。
        within_ref: 可选章节范围。
        mode: 搜索模式。
        diagnosis: 查询诊断结果。
        semantic_profiles: 章节语义画像映射。

    Returns:
        ``(ranked_entries, strategy_hit_counts, exact_matches, expansion_queries)`` 四元组。
    """

    exact_matches: list[dict[str, Any]] = []
    expansion_queries: list[dict[str, str]] = []
    strategy_hit_counts = _build_empty_search_strategy_hit_counts()
    ranked_entries: list[dict[str, Any]] = []

    search_plan = _build_adaptive_search_plan(
        query=query,
        mode=mode,
        diagnosis=diagnosis,
    )

    if search_plan.run_exact:
        # 剥离引号：LLM 常用引号表达精确意图（仿搜索引擎语法），
        # run_exact 路径已隐含精确匹配语义，字面引号会导致假阴性。
        # 适用于 auto 和 exact 两种 mode（keyword/semantic 不走此路径）。
        exact_query = query.replace('"', '').strip()
        exact_matches_raw: list[SearchHit] = processor.search(exact_query or query, within_ref)
        all_normalized = _normalize_search_matches(exact_matches_raw)
        # 分离真正的精确命中与 processor 层 token fallback 命中。
        # 带 _token_fallback 标记的命中来自 processor 内部的 token OR 回退，
        # 应归入 token 策略而非 exact 策略。
        exact_matches = [m for m in all_normalized if not m.get("_token_fallback")]
        token_fallback_matches = [m for m in all_normalized if m.get("_token_fallback")]
        if exact_matches:
            strategy_hit_counts[_SEARCH_STRATEGY_EXACT] = len(exact_matches)
            ranked_entries = _build_ranked_search_entries(
                matches=exact_matches,
                strategy=_SEARCH_STRATEGY_EXACT,
                query=query,
            )
        if token_fallback_matches:
            strategy_hit_counts[_SEARCH_STRATEGY_TOKEN] = (
                strategy_hit_counts.get(_SEARCH_STRATEGY_TOKEN, 0) + len(token_fallback_matches)
            )
            ranked_entries.extend(
                _build_ranked_search_entries(
                    matches=token_fallback_matches,
                    strategy=_SEARCH_STRATEGY_TOKEN,
                    query=query,
                )
            )

    should_expand = bool(search_plan.expansion_phases) and (
        mode != SEARCH_MODE_AUTO or not exact_matches
    )
    if should_expand:
        for phase_index, phase in enumerate(search_plan.expansion_phases, start=1):
            for expansion in phase:
                expanded_query = expansion["query"]
                strategy = expansion["strategy"]
                expansion_queries.append({"query": expanded_query, "strategy": strategy})
                matches_raw = processor.search(expanded_query, within_ref)
                normalized_matches = _normalize_search_matches(matches_raw)
                if not normalized_matches:
                    continue

                token_phase = strategy == _SEARCH_STRATEGY_TOKEN
                if diagnosis.intent != "general" and search_plan.scoped_before_token:
                    strict_scope = token_phase and search_plan.fallback_gated and phase_index > 1
                    scoped_matches = _filter_matches_by_intent(
                        matches=normalized_matches,
                        diagnosis=diagnosis,
                        semantic_profiles=semantic_profiles,
                    )
                    if scoped_matches:
                        normalized_matches = scoped_matches
                    elif strict_scope:
                        for item in normalized_matches:
                            item["_token_fallback_opened"] = True

                if not normalized_matches:
                    continue
                strategy_hit_counts[strategy] = strategy_hit_counts.get(strategy, 0) + len(normalized_matches)
                ranked_entries.extend(
                    _build_ranked_search_entries(
                        matches=normalized_matches,
                        strategy=strategy,
                        query=expanded_query,
                    )
                )

    return ranked_entries, strategy_hit_counts, exact_matches, expansion_queries


# =====================================================================
# 查询扩展
# =====================================================================

def _build_search_query_expansions(
    query: str,
    *,
    mode: str = SEARCH_MODE_AUTO,
) -> list[dict[str, str]]:
    """构建搜索扩展查询集合。

    扩展顺序固定为：
    1. ``phrase_variant``（词形与分隔符变体）
    2. ``synonym``（同义词/术语映射）
    3. ``token``（关键词拆分回退）

    当 ``mode`` 为 ``keyword`` 时只生成 token 拆分查询；
    其他模式生成全部扩展。

    Args:
        query: 原始查询词。
        mode: 搜索模式，影响生成的扩展策略子集。

    Returns:
        扩展查询列表，每项包含 ``query`` 与 ``strategy``。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    expansions: list[dict[str, str]] = []
    seen: set[str] = {_normalize_search_query_for_key(query)}

    # keyword 模式仅做 token 拆分，跳过短语变体和同义词
    include_phrase_variant = mode != SEARCH_MODE_KEYWORD
    include_synonym = mode != SEARCH_MODE_KEYWORD

    if include_phrase_variant:
        phrase_variants = _build_phrase_variant_queries(query)
        for variant in phrase_variants:
            _append_search_expansion(
                expansions=expansions,
                seen=seen,
                query=variant,
                strategy=_SEARCH_STRATEGY_PHRASE_VARIANT,
            )

    if include_synonym:
        synonym_queries = _build_synonym_queries(query)
        for synonym_query in synonym_queries:
            _append_search_expansion(
                expansions=expansions,
                seen=seen,
                query=synonym_query,
                strategy=_SEARCH_STRATEGY_SYNONYM,
            )

    token_queries = _build_token_queries(query)
    for token_query in token_queries:
        _append_search_expansion(
            expansions=expansions,
            seen=seen,
            query=token_query,
            strategy=_SEARCH_STRATEGY_TOKEN,
        )
    return expansions


def _append_search_expansion(
    *,
    expansions: list[dict[str, str]],
    seen: set[str],
    query: str,
    strategy: str,
) -> None:
    """向扩展查询列表追加唯一查询项。

    Args:
        expansions: 目标扩展列表。
        seen: 已出现查询集合（归一化后）。
        query: 候选查询。
        strategy: 查询策略名。

    Returns:
        无。

    Raises:
        RuntimeError: 追加失败时抛出。
    """

    normalized_query = normalize_optional_text(query)
    if normalized_query is None:
        return
    normalized_key = _normalize_search_query_for_key(normalized_query)
    if normalized_key in seen:
        return
    seen.add(normalized_key)
    expansions.append({"query": normalized_query, "strategy": strategy})


def _normalize_search_query_for_key(query: str) -> str:
    """标准化查询词用于去重键。

    Args:
        query: 原始查询词。

    Returns:
        归一化后的键字符串。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    normalized = normalize_optional_text(query) or ""
    lowered = normalized.lower()
    return _SPACE_NORMALIZE_PATTERN.sub(" ", lowered).strip()


def _build_phrase_variant_queries(query: str) -> list[str]:
    """生成短语变体查询。

    Args:
        query: 原始查询词。

    Returns:
        变体查询列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    normalized = normalize_optional_text(query)
    if normalized is None:
        return []

    variants: set[str] = set()
    if "-" in normalized:
        variants.add(normalized.replace("-", " "))
    if "/" in normalized:
        variants.add(normalized.replace("/", " "))

    lowered = normalized.lower()
    ascii_tokens = _extract_ascii_tokens(lowered)
    if ascii_tokens:
        for index, token in enumerate(ascii_tokens):
            for inflection in _expand_ascii_token_inflections(token):
                replaced = list(ascii_tokens)
                replaced[index] = inflection
                variants.add(" ".join(replaced))

    normalized_key = _normalize_search_query_for_key(normalized)
    ordered_variants: list[str] = []
    for candidate in variants:
        candidate_key = _normalize_search_query_for_key(candidate)
        if not candidate_key or candidate_key == normalized_key:
            continue
        ordered_variants.append(candidate)
    ordered_variants.sort()
    return ordered_variants


def _build_synonym_queries(query: str) -> list[str]:
    """生成同义词扩展查询。

    Args:
        query: 原始查询词。

    Returns:
        同义词查询列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    normalized_key = _normalize_search_query_for_key(query)
    if not normalized_key:
        return []
    synonyms: set[str] = set()
    for group in _SEARCH_SYNONYM_GROUPS:
        group_keys = {_normalize_search_query_for_key(item): item for item in group}
        if normalized_key not in group_keys:
            continue
        for key, value in group_keys.items():
            if key == normalized_key:
                continue
            synonyms.add(value)
    ordered_synonyms = sorted(synonyms, key=lambda item: _normalize_search_query_for_key(item))
    return ordered_synonyms


def _build_token_queries(query: str) -> list[str]:
    """生成 token 回退查询。

    Args:
        query: 原始查询词。

    Returns:
        token 查询列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    tokens = _extract_ascii_tokens(query.lower())
    result: list[str] = []
    for token in tokens:
        if len(token) < 3 or token in _TOKEN_STOP_WORDS:
            continue
        result.append(token)
    return result


def _extract_ascii_tokens(query: str) -> list[str]:
    """提取查询中的英文/数字 token。

    Args:
        query: 原始查询词。

    Returns:
        token 列表。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    return _WORD_SPLIT_PATTERN.findall(query or "")


def _expand_ascii_token_inflections(token: str) -> list[str]:
    """生成英文 token 的简单词形变体。

    Args:
        token: 原始 token。

    Returns:
        词形变体列表（不含原 token）。

    Raises:
        RuntimeError: 生成失败时抛出。
    """

    normalized = token.strip().lower()
    if len(normalized) < 3:
        return []
    variants: set[str] = set()
    if normalized.endswith("ies") and len(normalized) > 4:
        variants.add(f"{normalized[:-3]}y")
    if normalized.endswith("es") and len(normalized) > 3:
        variants.add(normalized[:-2])
    if normalized.endswith("s") and len(normalized) > 3:
        variants.add(normalized[:-1])
    else:
        variants.add(f"{normalized}s")
    variants.discard(normalized)
    ordered = sorted(variants)
    return ordered


# =====================================================================
# 命中排序与去重
# =====================================================================

def _build_ranked_search_entries(
    *,
    matches: list[dict[str, Any]],
    strategy: str,
    query: str,
) -> list[dict[str, Any]]:
    """构建带策略优先级的搜索命中条目。

    Args:
        matches: 标准化命中列表。
        strategy: 命中策略名。
        query: 生成该命中的原始查询词。

    Returns:
        带排序权重的命中条目列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    priority = _SEARCH_STRATEGY_PRIORITY.get(strategy, 999)
    result: list[dict[str, Any]] = []
    for match in matches:
        entry = dict(match)
        entry["_strategy"] = strategy
        entry["_priority"] = priority
        entry["_query"] = query
        result.append(entry)
    return result


def _deduplicate_ranked_search_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按命中内容去重并保留更高优先级条目。

    Args:
        entries: 原始带权重命中列表。

    Returns:
        去重后的命中列表。

    Raises:
        RuntimeError: 去重失败时抛出。
    """

    selected: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for entry in entries:
        section_ref = str(entry.get("section_ref") or "")
        section_title = str(entry.get("section_title") or "")
        # 支持 evidence 和传统 snippet 两种模式的去重键
        evidence = entry.get("evidence")
        if isinstance(evidence, dict):
            content_key = str(evidence.get("context") or evidence.get("matched_text") or "")
        else:
            content_key = str(entry.get("snippet") or "")
        page_no = str(entry.get("page_no") or "")
        key = (section_ref, section_title, content_key, page_no)
        current = selected.get(key)
        if current is None:
            selected[key] = entry
            continue
        if int(entry.get("_priority", 999)) < int(current.get("_priority", 999)):
            selected[key] = entry
    return list(selected.values())


def _compute_keyword_proximity_score(entry: dict[str, Any]) -> int:
    """计算命中条目中关键词的邻近度分数。

    多词查询时，命中文本中关键词距离越近分数越低（排序越靠前）。
    单词查询或无法计算时返回 0（不影响排序）。

    算法：提取 evidence.context 的所有 token 位置，计算查询 token
    在文本中最小包围窗口的长度作为邻近度分数。

    Args:
        entry: 带 evidence 的搜索命中条目。

    Returns:
        邻近度分数（越小越好），无法计算时返回 0。
    """

    evidence = entry.get("evidence")
    if not isinstance(evidence, dict):
        return 0
    context = str(evidence.get("context") or "").lower()
    if not context:
        return 0
    # 从 entry 的原始查询信息中提取 token
    # 使用 evidence.matched_text 作为查询词近似
    matched_text = str(evidence.get("matched_text") or "").lower()
    query_tokens = _WORD_SPLIT_PATTERN.findall(matched_text)
    # 移除停用词和短 token
    query_tokens = [t for t in query_tokens if len(t) >= 3 and t not in _TOKEN_STOP_WORDS]
    if len(query_tokens) < 2:
        return 0

    # 在 context 中查找每个 query token 的所有出现位置
    context_tokens = _WORD_SPLIT_PATTERN.findall(context)
    token_positions: dict[str, list[int]] = {}
    for pos, ct in enumerate(context_tokens):
        for qt in query_tokens:
            if ct == qt or ct.startswith(qt) or qt.startswith(ct):
                token_positions.setdefault(qt, []).append(pos)

    # 只有全部 query token 都出现时才计算窗口
    if len(token_positions) < 2:
        return 0

    # 最小包围窗口：各 token 取第一次出现的位置，计算 max-min 范围
    min_window = 999999
    first_positions = [positions[0] for positions in token_positions.values()]
    window = max(first_positions) - min(first_positions)
    min_window = min(min_window, window)

    return min_window


def _sort_ranked_search_entries(
    entries: list[dict[str, Any]],
    *,
    bm25f_index: Optional[BM25FSectionIndex] = None,
    diagnosis: Optional[QueryDiagnosis] = None,
    semantic_profiles: Optional[dict[str, SectionSemanticProfile]] = None,
) -> list[dict[str, Any]]:
    """对命中条目进行稳定排序。

    排序轴：策略优先级 → 意图一致性（降序）→ 噪音惩罚（升序）
    → BM25F 分数（降序）→ 关键词邻近度 → 章节 ref → 页码 → 内容文本。
    邻近度分数越小（关键词越近）排序越靠前。

    Args:
        entries: 去重后的命中条目列表。
        bm25f_index: 可选 BM25F 索引；为空时跳过该排序信号。
        diagnosis: 可选查询诊断结果。
        semantic_profiles: 可选章节语义画像索引。

    Returns:
        排序后的命中条目列表。

    Raises:
        RuntimeError: 排序失败时抛出。
    """

    for item in entries:
        query = str(item.get("_query") or "").strip()
        if bm25f_index is None or not query:
            item["_bm25f_score"] = 0.0
        else:
            item["_bm25f_score"] = score_search_entry_bm25f(
                entry=item,
                query=query,
                index=bm25f_index,
            )
        item["_intent_alignment_score"] = _compute_intent_alignment_score(
            entry=item,
            diagnosis=diagnosis,
            semantic_profiles=semantic_profiles,
        )
        item["_context_noise_penalty"] = _compute_context_noise_penalty(
            entry=item,
            diagnosis=diagnosis,
        )

    return sorted(
        entries,
        key=lambda item: (
            int(item.get("_priority", 999)),
            -float(item.get("_intent_alignment_score", 0.0)),
            float(item.get("_context_noise_penalty", 0.0)),
            -float(item.get("_bm25f_score", 0.0)),
            _compute_keyword_proximity_score(item),
            str(item.get("section_ref") or ""),
            int(item.get("page_no") or 0),
            str(item.get("snippet") or item.get("evidence", {}).get("context", "") or ""),
        ),
    )


def _compute_intent_alignment_score(
    *,
    entry: dict[str, Any],
    diagnosis: Optional[QueryDiagnosis],
    semantic_profiles: Optional[dict[str, SectionSemanticProfile]],
) -> float:
    """计算命中与查询意图的一致性得分。

    Args:
        entry: 搜索命中条目。
        diagnosis: 查询诊断结果。
        semantic_profiles: 章节语义画像索引。

    Returns:
        一致性得分，范围 ``0~1``。

    Raises:
        RuntimeError: 计算失败时抛出。
    """

    if diagnosis is None or semantic_profiles is None:
        return 0.0
    if diagnosis.intent == "general":
        return 0.0
    section_ref = str(entry.get("section_ref") or "").strip()
    if not section_ref:
        return 0.0
    profile = semantic_profiles.get(section_ref)
    if profile is None:
        return 0.0
    expected_buckets = _expected_buckets_for_intent(diagnosis.intent)
    if not expected_buckets:
        return 0.0
    return 1.0 if profile.bucket in expected_buckets else 0.0


def _compute_context_noise_penalty(
    *,
    entry: dict[str, Any],
    diagnosis: Optional[QueryDiagnosis],
) -> float:
    """计算命中上下文噪音惩罚分。

    Args:
        entry: 搜索命中条目。
        diagnosis: 查询诊断结果。

    Returns:
        惩罚分（越大表示噪音越多）。

    Raises:
        RuntimeError: 计算失败时抛出。
    """

    if diagnosis is None:
        return 0.0
    noise_terms = _NOISE_CONTEXT_TOKENS_BY_INTENT.get(diagnosis.intent)
    support_terms = _SUPPORT_CONTEXT_TOKENS_BY_INTENT.get(diagnosis.intent)
    if not noise_terms:
        return 0.0

    evidence = entry.get("evidence")
    context = ""
    if isinstance(evidence, dict):
        context = str(evidence.get("context") or evidence.get("matched_text") or "")
    if not context:
        context = str(entry.get("snippet") or "")
    tokens = _extract_ascii_tokens(context.lower())
    if not tokens:
        return 0.0

    token_set = set(tokens)
    noise_hits = len(token_set.intersection(noise_terms))
    if noise_hits <= 0:
        return 0.0
    support_hits = len(token_set.intersection(support_terms or frozenset()))
    # 当上下文同时出现行业支持词时，降低惩罚，避免误伤真实业务竞争描述。
    return 0.8 if support_hits > 0 else min(2.0, 0.6 + (0.25 * noise_hits))


# =====================================================================
# exact 优先限流
# =====================================================================

# 当 exact + expansion 共存时，expansion 最多占用的比例上限
_EXPANSION_RATIO_WHEN_EXACT_EXISTS: float = 0.3
# 限流生效的最小条目阈值（低于此数量不触发裁剪）
_CAP_MIN_TRIGGER: int = 8


def _cap_entries_with_exact_priority(
    sorted_entries: list[dict[str, Any]],
    display_budget: Optional[int] = None,
) -> list[dict[str, Any]]:
    """exact 优先限流：当精确命中存在时，压缩扩展结果占比。

    精确命中全部保留；扩展结果最多占总容量的 30%（至少保留 2 条）。
    条目总数低于 _CAP_MIN_TRIGGER 时不触发裁剪。

    当 ``display_budget`` 存在且所有 exact 条目能装入 budget 时，
    进一步收紧 expansion 配额，使裁剪后总数 ≤ display_budget，
    避免下游 truncation_manager 产生 cursor 与 hint 信号冲突。

    Args:
        sorted_entries: 已排序的搜索条目（exact 在前）。
        display_budget: 可选的展示预算上限（对应 truncation max_items）。

    Returns:
        裁剪后的条目列表。
    """
    total = len(sorted_entries)
    if total < _CAP_MIN_TRIGGER:
        return sorted_entries

    exact_entries: list[dict[str, Any]] = []
    expansion_entries: list[dict[str, Any]] = []
    for entry in sorted_entries:
        if entry.get("_strategy") == _SEARCH_STRATEGY_EXACT:
            exact_entries.append(entry)
        else:
            expansion_entries.append(entry)

    # 无 exact 命中时不裁剪 expansion
    if not exact_entries:
        return sorted_entries

    # expansion 配额 = 总量 × 比例上限，至少保留 2 条
    expansion_cap = max(2, int(total * _EXPANSION_RATIO_WHEN_EXACT_EXISTS))

    # 当 display_budget 存在且 exact 全部可以装入时，
    # 收紧 expansion 配额使总数不超过 display_budget
    if display_budget and len(exact_entries) <= display_budget:
        budget_remaining = display_budget - len(exact_entries)
        expansion_cap = min(expansion_cap, max(2, budget_remaining))

    capped_expansion = expansion_entries[:expansion_cap]

    return exact_entries + capped_expansion


# =====================================================================
# 证据化结构构建
# =====================================================================

# matched_text 的最大字符数——用于从 snippet 中裁剪查询命中句
_MATCHED_TEXT_MAX_CHARS: int = 120


def _center_matched_text(snippet: str, query: str, max_chars: int = _MATCHED_TEXT_MAX_CHARS) -> str:
    """从 snippet 中提取围绕 query 命中位置的摘要文本。

    找到 query 在 snippet 中的位置后，以该位置为中心截取 max_chars；
    若 query 不存在于 snippet 中，则回退到 snippet 头部截取。

    Args:
        snippet: 完整 snippet 文本。
        query: 查询词。
        max_chars: 最大字符数。

    Returns:
        围绕查询命中位置的摘要文本。
    """
    if not snippet:
        return ""
    if len(snippet) <= max_chars:
        return snippet
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return snippet[:max_chars]
    # 在 snippet 中查找 query 出现位置
    try:
        pattern = re.compile(re.escape(normalized_query), flags=re.IGNORECASE)
    except re.error:
        return snippet[:max_chars]
    match = pattern.search(snippet)
    if match is None:
        return snippet[:max_chars]
    # 左预算 = max_chars 的一半，使命中点居中
    left_budget = max(1, max_chars // 2)
    start = max(0, match.start() - left_budget)
    end = min(len(snippet), start + max_chars)
    start = max(0, end - max_chars)
    return snippet[start:end]


def _build_evidence_matches(
    sorted_entries: list[dict[str, Any]],
    form_type: Optional[str],
    ref_to_topic: Optional[dict[str, Optional[str]]] = None,
) -> list[dict[str, Any]]:
    """将排序后的搜索条目转换为证据化返回格式。

    每个命中包含：
    - evidence 复合结构（matched_text + context + match_position）
    - matched_query: 产生此命中的原始查询词
    - is_exact_phrase: 是否为原文精确短语匹配
    - topic 语义标注

    Args:
        sorted_entries: 排序去重后的搜索条目。
        form_type: 文档 form_type（用于语义解析）。
        ref_to_topic: 可选的 section_ref → topic 预构建索引，
            用于子章节命中的 topic 回退查表。

    Returns:
        证据化命中列表。
    """
    matches: list[dict[str, Any]] = []
    for entry in sorted_entries:
        section_ref = entry.get("section_ref")
        section_title = entry.get("section_title")
        # 解析章节语义
        item_number, _, topic = resolve_section_semantic(
            title=section_title,
            form_type=form_type,
        )
        # 子章节无法自解析时，从预构建索引回退查表
        if topic is None and ref_to_topic and section_ref:
            topic = ref_to_topic.get(section_ref)
        # 查询归属与精确度
        matched_query = str(entry.get("_query") or "")
        strategy = str(entry.get("_strategy") or "")
        is_exact = strategy == _SEARCH_STRATEGY_EXACT
        # 构建 evidence 结构
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict):
            # 兼容传统 snippet 命中：升级为 evidence 结构，matched_text 围绕查询居中
            snippet_text = str(entry.get("snippet") or "")
            evidence = {
                "matched_text": _center_matched_text(snippet_text, matched_query),
                "context": snippet_text,
            }
        # section 对象：与 get_document_sections / read_section 统一结构
        item_label = f"Item {item_number}" if item_number else None
        match_entry: dict[str, Any] = {
            "section": {
                "ref": section_ref,
                "title": section_title,
                "item": item_label,
                "topic": topic,
            },
            "matched_query": matched_query,
            "is_exact_phrase": is_exact,
            "evidence": evidence,
            "page_no": entry.get("page_no"),
        }
        matches.append(match_entry)
    return matches
