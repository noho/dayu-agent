"""搜索结果后处理工具。

该模块为多个 Processor 提供统一的搜索片段抽取与去重能力，目标是：
1. 以查询词为锚点生成更可读的抽取式 snippet（非生成式摘要）。
2. 在 section 内去除近重复片段，避免同类命中刷屏。
3. 以确定性规则输出，保证可复现与可测试。
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Any, Callable, Protocol, TypeVar

from .base import SearchEvidence, SearchHit, build_search_hit
from .text_utils import normalize_whitespace as _normalize_whitespace

_SENTENCE_END_PUNCT = {"。", "！", "？", "!", "?", "；", ";"}
_NON_WORD_PATTERN = re.compile(r"[\W_]+", flags=re.UNICODE)
# 句末标点正则，用于 _split_sentence_spans 的高效切分
_SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?；;]")

# ---------------------------------------------------------------------------
# 搜索结果配置常量
# ---------------------------------------------------------------------------
# 单个 section 内最多返回的命中条数
SEARCH_PER_SECTION_LIMIT: int = 2
# snippet 最大字符数（抽取式摘要截断长度）
SEARCH_SNIPPET_MAX_CHARS: int = 360


def extract_query_anchored_snippets(
    content: str,
    query: str,
    max_chars: int = SEARCH_SNIPPET_MAX_CHARS,
    max_per_section: int = SEARCH_PER_SECTION_LIMIT,
) -> list[str]:
    """按查询词抽取并去重片段。

    Args:
        content: section 文本内容。
        query: 查询词。
        max_chars: 单条 snippet 最大字符数。
        max_per_section: 每个 section 最多保留条数。

    Returns:
        去重并限流后的 snippet 列表。

    Raises:
        RuntimeError: 处理失败时抛出。
    """

    normalized_content = _normalize_whitespace(content)
    normalized_query = str(query or "").strip()
    if not normalized_content or not normalized_query:
        return []

    sentence_spans = _split_sentence_spans(normalized_content)
    if not sentence_spans:
        return []

    query_pattern = re.compile(re.escape(normalized_query), flags=re.IGNORECASE)
    match_starts = [match.start() for match in query_pattern.finditer(normalized_content)]
    if not match_starts:
        return []

    sentences = [span["sentence"] for span in sentence_spans]
    snippets_raw: list[str] = []
    for match_start in match_starts:
        sentence_index = _locate_sentence_index(sentence_spans, match_start)
        if sentence_index is None:
            continue
        snippet = build_snippet_from_sentence_window(
            sentences=sentences,
            hit_index=sentence_index,
            query=normalized_query,
            max_chars=max_chars,
        )
        if not snippet:
            continue
        if query_pattern.search(snippet) is None:
            # 复杂逻辑说明：理论上命中片段一定包含 query；这里额外防御，避免分句异常导致锚点丢失。
            continue
        snippets_raw.append(snippet)

    if not snippets_raw:
        # 复杂逻辑说明：极端情况下分句可能失败，回退到按字符窗口截取，保证搜索有结果。
        snippets_raw = _fallback_char_window_snippets(
            content=normalized_content,
            query=normalized_query,
            max_chars=max_chars,
        )

    deduped = dedup_snippets(snippets_raw)
    return cap_per_section(deduped, max_per_section)


def split_sentences(text: str) -> list[str]:
    """按中英文句末标点切分句子。

    Args:
        text: 输入文本。

    Returns:
        句子列表。

    Raises:
        RuntimeError: 处理失败时抛出。
    """

    spans = _split_sentence_spans(_normalize_whitespace(text))
    return [span["sentence"] for span in spans]


def build_snippet_from_sentence_window(
    sentences: list[str],
    hit_index: int,
    query: str,
    max_chars: int,
) -> str:
    """以命中句为中心构建 snippet。

    Args:
        sentences: 句子列表。
        hit_index: 命中句索引。
        query: 查询词。
        max_chars: snippet 最大字符数。

    Returns:
        构建后的 snippet。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    normalized_query = str(query or "").strip()
    if not sentences:
        return ""
    if hit_index < 0 or hit_index >= len(sentences):
        return ""
    if max_chars <= 0:
        return ""

    left = hit_index
    right = hit_index
    snippet = _join_sentence_window(sentences, left, right)
    if len(snippet) > max_chars:
        return _truncate_around_query(snippet, normalized_query, max_chars)

    while True:
        expanded = False
        if left > 0:
            candidate_left = _join_sentence_window(sentences, left - 1, right)
            if len(candidate_left) <= max_chars:
                left -= 1
                snippet = candidate_left
                expanded = True
        if right < len(sentences) - 1:
            candidate_right = _join_sentence_window(sentences, left, right + 1)
            if len(candidate_right) <= max_chars:
                right += 1
                snippet = candidate_right
                expanded = True
        if not expanded:
            break

    if re.search(re.escape(normalized_query), snippet, flags=re.IGNORECASE) is None:
        return _truncate_around_query(snippet, normalized_query, max_chars)
    return snippet


def normalize_for_dedup(text: str) -> str:
    """规范化文本用于去重比较。

    Args:
        text: 原始文本。

    Returns:
        规范化字符串。

    Raises:
        RuntimeError: 处理失败时抛出。
    """

    lowered = _normalize_whitespace(text).lower()
    return _NON_WORD_PATTERN.sub("", lowered)


def dedup_snippets(snippets: list[str]) -> list[str]:
    """对片段做稳定去重。

    去重规则：
    1. 规范化后完全相同视为重复。
    2. 规范化后存在包含关系时，保留信息量更高（更长）的片段。

    Args:
        snippets: 原始片段列表。

    Returns:
        去重后片段。

    Raises:
        RuntimeError: 去重失败时抛出。
    """

    deduped: list[str] = []
    normalized_values: list[str] = []

    for snippet in snippets:
        current = _normalize_whitespace(snippet)
        if not current:
            continue
        normalized = normalize_for_dedup(current)
        if not normalized:
            continue

        duplicated = False
        for index, existing in enumerate(normalized_values):
            if normalized == existing or normalized in existing:
                duplicated = True
                break
            if existing in normalized:
                deduped[index] = current
                normalized_values[index] = normalized
                duplicated = True
                break

        if not duplicated:
            deduped.append(current)
            normalized_values.append(normalized)

    return deduped


def cap_per_section(snippets: list[str], limit: int = 2) -> list[str]:
    """按 section 限流片段数量。

    Args:
        snippets: 片段列表。
        limit: 保留上限。

    Returns:
        限流后的片段列表。

    Raises:
        RuntimeError: 限流失败时抛出。
    """

    if limit <= 0:
        return []
    return list(snippets[:limit])


def enrich_hits_by_section(
    hits_raw: list[SearchHit],
    section_content_map: dict[str, str],
    query: str,
    per_section_limit: int = SEARCH_PER_SECTION_LIMIT,
    snippet_max_chars: int = SEARCH_SNIPPET_MAX_CHARS,
) -> list[SearchHit]:
    """按 section 聚合并增强搜索命中。

    Args:
        hits_raw: Processor 原始命中（至少包含 `section_ref`）。
        section_content_map: `section_ref -> section content` 映射。
        query: 查询词。
        per_section_limit: 每个 section 的最大返回条数。
        snippet_max_chars: snippet 最大长度。

    Returns:
        增强后的命中列表。

    Raises:
        RuntimeError: 增强失败时抛出。
    """

    grouped: "OrderedDict[str, list[SearchHit]]" = OrderedDict()
    for hit in hits_raw:
        section_ref = str(hit.get("section_ref", "")).strip()
        if not section_ref:
            continue
        grouped.setdefault(section_ref, []).append(hit)

    enriched_hits: list[SearchHit] = []
    for section_ref, section_hits in grouped.items():
        title = section_hits[0].get("section_title")
        page_no = _pick_first_positive_page_no(section_hits)
        section_content = section_content_map.get(section_ref, "")

        snippets = extract_query_anchored_snippets(
            content=section_content,
            query=query,
            max_chars=snippet_max_chars,
            max_per_section=per_section_limit,
        )
        if not snippets:
            fallback_raw = [str(hit.get("snippet", "")).strip() for hit in section_hits]
            snippets = cap_per_section(dedup_snippets(fallback_raw), per_section_limit)

        for snippet in snippets:
            hit = build_search_hit(
                section_ref=section_ref,
                section_title=title,
                snippet=snippet,
                page_no=page_no,
            )
            enriched_hits.append(hit)

    return enriched_hits


def _split_sentence_spans(text: str) -> list[dict[str, Any]]:
    """切分句子并返回原文位置区间。

    使用预编译正则 ``_SENTENCE_SPLIT_PATTERN`` 按句末标点切分，
    替代逐字符迭代，性能更优。

    Args:
        text: 输入文本。

    Returns:
        句子区间列表，每项包含 `start/end/sentence`。

    Raises:
        RuntimeError: 切分失败时抛出。
    """

    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    spans: list[dict[str, Any]] = []
    current_start = 0
    for match in _SENTENCE_SPLIT_PATTERN.finditer(normalized):
        end = match.end()
        sentence = normalized[current_start:end].strip()
        if sentence:
            spans.append({"start": current_start, "end": end, "sentence": sentence})
        current_start = end

    tail = normalized[current_start:].strip()
    if tail:
        spans.append({"start": current_start, "end": len(normalized), "sentence": tail})
    return spans


def _locate_sentence_index(sentence_spans: list[dict[str, Any]], position: int) -> int | None:
    """根据字符位置定位命中句索引。

    Args:
        sentence_spans: 句子区间列表。
        position: 命中起始位置。

    Returns:
        命中句索引，未找到时返回 `None`。

    Raises:
        RuntimeError: 定位失败时抛出。
    """

    for index, span in enumerate(sentence_spans):
        start = int(span["start"])
        end = int(span["end"])
        if start <= position < end:
            return index
    return None


def _join_sentence_window(sentences: list[str], left: int, right: int) -> str:
    """拼接句子窗口文本。

    Args:
        sentences: 句子列表。
        left: 左边界（含）。
        right: 右边界（含）。

    Returns:
        拼接后的文本。

    Raises:
        RuntimeError: 拼接失败时抛出。
    """

    if left < 0 or right >= len(sentences) or left > right:
        return ""
    return _normalize_whitespace(" ".join(sentences[left:right + 1]))


def _truncate_around_query(text: str, query: str, max_chars: int) -> str:
    """围绕 query 对超长文本截断。

    Args:
        text: 原始文本。
        query: 查询词。
        max_chars: 最大长度。

    Returns:
        截断后的文本。

    Raises:
        RuntimeError: 截断失败时抛出。
    """

    normalized = _normalize_whitespace(text)
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 0:
        return ""

    normalized_query = str(query or "").strip()
    if not normalized_query:
        return normalized[:max_chars]

    # 使用 re.compile 预编译，避免每次调用重新编译 escape 后的模式
    try:
        pattern = re.compile(re.escape(normalized_query), flags=re.IGNORECASE)
    except re.error:
        return normalized[:max_chars]
    match = pattern.search(normalized)
    if match is None:
        return normalized[:max_chars]

    left_budget = max(1, max_chars // 2)
    start = max(0, match.start() - left_budget)
    end = min(len(normalized), start + max_chars)
    start = max(0, end - max_chars)
    return normalized[start:end]


def _fallback_char_window_snippets(content: str, query: str, max_chars: int) -> list[str]:
    """字符窗口回退提取。

    以查询命中位置为中心截取字符窗口，并自适应对齐到最近的单词边界，
    避免 snippet 在单词中间截断。

    Args:
        content: 文本内容。
        query: 查询词。
        max_chars: 窗口长度上限。

    Returns:
        回退片段列表。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    snippets: list[str] = []
    if not content or not query:
        return snippets

    pattern = re.compile(re.escape(query), flags=re.IGNORECASE)
    for match in pattern.finditer(content):
        start = max(0, match.start() - max_chars // 2)
        end = min(len(content), start + max_chars)
        start = max(0, end - max_chars)
        # Step 14: 自适应对齐到最近的单词边界
        start = _snap_to_word_boundary_left(content, start)
        end = _snap_to_word_boundary_right(content, end)
        snippet = _normalize_whitespace(content[start:end])
        if snippet:
            snippets.append(snippet)
    return snippets


def _snap_to_word_boundary_left(text: str, pos: int) -> int:
    """将位置向左对齐到最近的单词边界（空白处）。

    从 pos 向右搜索不超过 20 字符，找到第一个空白后的位置。
    若 pos 已在单词开头或文本开头，则原样返回。

    Args:
        text: 原始文本。
        pos: 起始位置。

    Returns:
        对齐后的位置。

    Raises:
        RuntimeError: 处理失败时抛出。
    """
    if pos <= 0:
        return 0
    # 如果 pos 前面是空白，已在边界
    if text[pos - 1].isspace():
        return pos
    # 向右查找最近空白（不超过 20 字符）
    for i in range(pos, min(pos + 20, len(text))):
        if text[i].isspace():
            return i + 1
    return pos


def _snap_to_word_boundary_right(text: str, pos: int) -> int:
    """将位置向右对齐到最近的单词边界（空白处）。

    从 pos 向右搜索不超过 20 字符，找到第一个空白处截断。
    若 pos 已在单词结尾或文本结尾，则原样返回。

    Args:
        text: 原始文本。
        pos: 结束位置。

    Returns:
        对齐后的位置。

    Raises:
        RuntimeError: 处理失败时抛出。
    """
    if pos >= len(text):
        return len(text)
    # 如果 pos 处是空白，已在边界
    if text[pos].isspace():
        return pos
    # 向右查找最近空白（不超过 20 字符）
    for i in range(pos, min(pos + 20, len(text))):
        if text[i].isspace():
            return i
    return pos


# ---------------------------------------------------------------------------
# Token 共现 snippet 提取（token fallback 搜索专用）
# ---------------------------------------------------------------------------


def extract_token_cooccurrence_snippets(
    content: str,
    tokens: list[str],
    original_query: str,
    max_chars: int = SEARCH_SNIPPET_MAX_CHARS,
    max_per_section: int = SEARCH_PER_SECTION_LIMIT,
) -> list[str]:
    """基于 token 共现密度提取最优 snippet。

    用于 token fallback 搜索场景：原始多词查询在文本中无精确匹配，
    但各 token 独立存在。本函数找到 token 共现密度最高的文本窗口，
    优先返回多 token 共现的区域而非单 token 命中。

    Args:
        content: section 文本内容。
        tokens: 查询拆分后的 token 列表。
        original_query: 原始查询词（用于优先精确匹配回退）。
        max_chars: 单条 snippet 最大字符数。
        max_per_section: 每个 section 最多保留条数。

    Returns:
        去重并限流后的 snippet 列表。
    """
    normalized = _normalize_whitespace(content)
    if not normalized or not tokens:
        return []

    # 优先尝试精确匹配（极少情况下 token fallback 路径中仍有精确匹配）
    exact_snippets = extract_query_anchored_snippets(
        content=normalized,
        query=original_query,
        max_chars=max_chars,
        max_per_section=max_per_section,
    )
    if exact_snippets:
        return exact_snippets

    # 收集每个 token 的所有出现位置
    token_positions: list[tuple[int, int]] = []  # (position, token_index)
    for token_idx, token in enumerate(tokens):
        pattern = re.compile(re.escape(token), flags=re.IGNORECASE)
        for m in pattern.finditer(normalized):
            token_positions.append((m.start(), token_idx))

    if not token_positions:
        return []

    # 按位置排序
    token_positions.sort()

    # 滑动窗口找 token 多样性最高的区域
    best_start = token_positions[0][0]
    best_diversity = 0
    best_center = best_start

    for i, (pos_i, _) in enumerate(token_positions):
        # 窗口：从 pos_i 开始，max_chars 范围内的所有 token 出现
        seen_tokens: set[int] = set()
        window_end = pos_i + max_chars
        for j in range(i, len(token_positions)):
            pos_j, tok_j = token_positions[j]
            if pos_j > window_end:
                break
            seen_tokens.add(tok_j)

        diversity = len(seen_tokens)
        if diversity > best_diversity:
            best_diversity = diversity
            # 窗口中心为区域中的最佳锚点
            best_center = pos_i
            best_start = pos_i

    # 围绕 best_center 截取 snippet
    half = max_chars // 2
    start = max(0, best_center - half)
    end = min(len(normalized), start + max_chars)
    start = max(0, end - max_chars)
    start = _snap_to_word_boundary_left(normalized, start)
    end = _snap_to_word_boundary_right(normalized, end)
    snippet = _normalize_whitespace(normalized[start:end])

    if not snippet:
        return []

    return cap_per_section(dedup_snippets([snippet]), max_per_section)


def enrich_hits_by_section_token_or(
    hits_raw: list[SearchHit],
    section_content_map: dict[str, str],
    tokens: list[str],
    original_query: str,
    per_section_limit: int = SEARCH_PER_SECTION_LIMIT,
    snippet_max_chars: int = SEARCH_SNIPPET_MAX_CHARS,
) -> list[SearchHit]:
    """为 token OR 回退搜索的命中生成 snippet。

    与 ``enrich_hits_by_section`` 类似，但使用 token 共现窗口提取 snippet，
    而非基于精确短语锚点。返回的每个 hit 带 ``_token_fallback: True`` 标记，
    供上游区分 token 回退命中与精确命中。

    Args:
        hits_raw: Processor 原始命中。
        section_content_map: section_ref → section content 映射。
        tokens: 查询拆分后的 token 列表。
        original_query: 原始查询词。
        per_section_limit: 每个 section 的最大返回条数。
        snippet_max_chars: snippet 最大长度。

    Returns:
        增强后的命中列表，每个 hit 带 ``_token_fallback: True``。
    """
    grouped: "OrderedDict[str, list[SearchHit]]" = OrderedDict()
    for hit in hits_raw:
        section_ref = str(hit.get("section_ref", "")).strip()
        if not section_ref:
            continue
        grouped.setdefault(section_ref, []).append(hit)

    enriched_hits: list[SearchHit] = []
    for section_ref, section_hits in grouped.items():
        title = section_hits[0].get("section_title")
        page_no = _pick_first_positive_page_no(section_hits)
        section_content = section_content_map.get(section_ref, "")

        snippets = extract_token_cooccurrence_snippets(
            content=section_content,
            tokens=tokens,
            original_query=original_query,
            max_chars=snippet_max_chars,
            max_per_section=per_section_limit,
        )
        if not snippets:
            # 回退：使用 raw snippet
            fallback_raw = [str(hit.get("snippet", "")).strip() for hit in section_hits]
            snippets = cap_per_section(dedup_snippets(fallback_raw), per_section_limit)

        for snippet in snippets:
            hit = build_search_hit(
                section_ref=section_ref,
                section_title=title,
                snippet=snippet,
                page_no=page_no,
                token_fallback=True,
            )
            enriched_hits.append(hit)

    return enriched_hits


def _pick_first_positive_page_no(hits: list[SearchHit]) -> int | None:
    """从命中列表中选择首个有效页码。

    Args:
        hits: 命中列表。

    Returns:
        首个正整数页码；不存在时返回 `None`。

    Raises:
        RuntimeError: 选择失败时抛出。
    """

    for hit in hits:
        page_no = hit.get("page_no")
        if isinstance(page_no, int) and page_no > 0:
            return page_no
    return None


# ---------------------------------------------------------------------------
# 带 title 的 section 搜索循环（bs / docling / markdown 三处理器共享）
# ---------------------------------------------------------------------------


class _TitledSection(Protocol):
    """带 title/ref 的 section 协议，用于 ``run_titled_section_search``。

    仅声明共享的属性字段，不约束具体 dataclass 类型，避免 processor 之间耦合。
    """

    ref: str
    title: str | None


_TitledSectionT = TypeVar("_TitledSectionT", bound=_TitledSection)


def run_titled_section_search(
    sections: list[_TitledSectionT],
    normalized_query: str,
    get_text: Callable[[_TitledSectionT], str],
    page_no_of: Callable[[_TitledSectionT], int | None] | None = None,
) -> tuple[list[SearchHit], dict[str, str]]:
    """按 title + content 双锚点在 sections 列表上做关键词搜索。

    原 `bs_processor` / `docling_processor` / `markdown_processor` 三处几乎同构的
    搜索循环被抽取到此处，核心行为保持一致：

    - 预编译 query 正则（避免循环内重复编译）。
    - 同时检测 `title` 与 `content`；
    - 若 title 命中但 content 无命中，将 title 前置拼入搜索文本，确保下游 snippet
      能定位到匹配词。
    - 命中后构建 `SearchHit`，snippet 暂存为 `normalized_query`，由下游
      `enrich_hits_by_section` 替换为锚点化 snippet。

    Args:
        sections: 待搜索的 section 列表。
        normalized_query: 已 strip 的查询词；调用方负责前置过滤空值。
        get_text: 从 section 读取正文的回调。
        page_no_of: 从 section 读取 page_no 的回调；不适用时传 None。

    Returns:
        二元组：`(hits_raw, section_content_map)`。`hits_raw` 作为入参传入
        `enrich_hits_by_section`；`section_content_map` 提供可搜索正文，供
        snippet 抽取时使用（若 title 前置则以拼接后的文本为准）。
    """

    query_pattern = re.compile(re.escape(normalized_query), flags=re.IGNORECASE)
    hits_raw: list[SearchHit] = []
    section_content_map: dict[str, str] = {}

    for section in sections:
        text = get_text(section)
        title_text = section.title or ""
        title_hit = bool(title_text) and query_pattern.search(title_text) is not None
        content_hit = query_pattern.search(text) is not None
        if not title_hit and not content_hit:
            continue
        # 若 title 命中而 content 无命中，将 title 前置进搜索文本，确保 snippet 能定位到匹配词。
        searchable_text = (
            (title_text + "\n" + text).strip()
            if title_hit and not content_hit
            else text
        )
        section_content_map[section.ref] = searchable_text
        hits_raw.append(
            build_search_hit(
                section_ref=section.ref,
                section_title=section.title,
                snippet=normalized_query,
                page_no=page_no_of(section) if page_no_of is not None else None,
            )
        )
    return hits_raw, section_content_map


# ---------------------------------------------------------------------------
# 证据化返回（Evidence Mode）
# ---------------------------------------------------------------------------
# 较 snippet 更丰富的结构，包含精确命中文本与扩展上下文。
EVIDENCE_CONTEXT_MAX_CHARS: int = 600
"""证据上下文最大字符数（比 snippet 更大的窗口）。"""


def extract_evidence_items(
    content: str,
    query: str,
    context_max_chars: int = EVIDENCE_CONTEXT_MAX_CHARS,
    max_per_section: int = SEARCH_PER_SECTION_LIMIT,
) -> list[SearchEvidence]:
    """按查询词抽取证据化命中条目。

    与 extract_query_anchored_snippets 类似，但返回结构化的 evidence 对象：
    - matched_text: 精确命中的原文片段
    - context: 包含命中位置的句子窗口上下文（更大范围）

    Args:
        content: section 文本内容。
        query: 查询词。
        context_max_chars: 上下文窗口最大字符数。
        max_per_section: 每个 section 最多保留条数。

    Returns:
        证据条目列表。
    """
    normalized_content = _normalize_whitespace(content)
    normalized_query = str(query or "").strip()
    if not normalized_content or not normalized_query:
        return []

    sentence_spans = _split_sentence_spans(normalized_content)
    if not sentence_spans:
        return []

    query_pattern = re.compile(re.escape(normalized_query), flags=re.IGNORECASE)
    matches_iter = list(query_pattern.finditer(normalized_content))
    if not matches_iter:
        return []

    sentences = [span["sentence"] for span in sentence_spans]
    evidence_items: list[SearchEvidence] = []
    seen_normalized: set[str] = set()

    for match in matches_iter:
        matched_text = normalized_content[match.start():match.end()]
        sentence_index = _locate_sentence_index(sentence_spans, match.start())
        if sentence_index is None:
            continue

        # 构建更大的上下文窗口
        context = build_snippet_from_sentence_window(
            sentences=sentences,
            hit_index=sentence_index,
            query=normalized_query,
            max_chars=context_max_chars,
        )
        if not context:
            continue

        # 去重：基于 context 的规范化形式
        norm_key = normalize_for_dedup(context)
        if norm_key in seen_normalized:
            continue
        # 检查包含关系
        skip = False
        for existing_key in list(seen_normalized):
            if norm_key in existing_key or existing_key in norm_key:
                skip = True
                break
        if skip:
            continue
        seen_normalized.add(norm_key)

        evidence_items.append(
            {
                "matched_text": matched_text,
                "context": context,
            }
        )

        if len(evidence_items) >= max_per_section:
            break

    return evidence_items


def enrich_hits_with_evidence(
    hits_raw: list[SearchHit],
    section_content_map: dict[str, str],
    query: str,
    per_section_limit: int = SEARCH_PER_SECTION_LIMIT,
    context_max_chars: int = EVIDENCE_CONTEXT_MAX_CHARS,
) -> list[SearchHit]:
    """按 section 聚合并生成证据化搜索命中。

    与 enrich_hits_by_section 类似但返回 evidence 结构。

    Args:
        hits_raw: Processor 原始命中。
        section_content_map: section_ref → section content 映射。
        query: 查询词。
        per_section_limit: 每个 section 的最大返回条数。
        context_max_chars: 证据上下文最大字符数。

    Returns:
        增强后的命中列表，包含 evidence 字段。
    """
    grouped: "OrderedDict[str, list[SearchHit]]" = OrderedDict()
    for hit in hits_raw:
        section_ref = str(hit.get("section_ref", "")).strip()
        if not section_ref:
            continue
        grouped.setdefault(section_ref, []).append(hit)

    enriched_hits: list[SearchHit] = []
    for section_ref, section_hits in grouped.items():
        title = section_hits[0].get("section_title")
        page_no = _pick_first_positive_page_no(section_hits)
        section_content = section_content_map.get(section_ref, "")

        evidence_items = extract_evidence_items(
            content=section_content,
            query=query,
            context_max_chars=context_max_chars,
            max_per_section=per_section_limit,
        )

        if not evidence_items:
            # 回退：使用传统 snippet 方式
            snippets = extract_query_anchored_snippets(
                content=section_content,
                query=query,
                max_chars=SEARCH_SNIPPET_MAX_CHARS,
                max_per_section=per_section_limit,
            )
            if not snippets:
                fallback_raw = [str(hit.get("snippet", "")).strip() for hit in section_hits]
                snippets = cap_per_section(dedup_snippets(fallback_raw), per_section_limit)
            for snippet in snippets:
                evidence: SearchEvidence = {
                    "matched_text": str(query),
                    "context": snippet,
                }
                hit = build_search_hit(
                    section_ref=section_ref,
                    section_title=title,
                    page_no=page_no,
                    evidence=evidence,
                )
                enriched_hits.append(hit)
            continue

        for item in evidence_items:
            hit = build_search_hit(
                section_ref=section_ref,
                section_title=title,
                page_no=page_no,
                evidence=item,
            )
            enriched_hits.append(hit)

    return enriched_hits


