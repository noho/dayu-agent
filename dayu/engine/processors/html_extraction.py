"""HTML 主体内容抽取工具。

本模块提供面向网页 HTML 的主体抽取原语，供 `web_tools` 与未来的
`HTMLProcessor` 复用。设计目标不是“完整保留 DOM”，而是稳定产出
低噪音、可继续规范化与 Markdown 渲染的正文片段。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Optional

from bs4 import BeautifulSoup, Tag

from .text_utils import normalize_whitespace as _normalize_whitespace

_HTML_FALLBACK_MIN_TEXT_CHARS = 60
_HTML_FALLBACK_MAX_BLOCKS = 80
_HTML_FALLBACK_CANDIDATE_SELECTORS: tuple[tuple[str, int], ...] = (
    ("article", 240),
    ("main", 220),
    ("[role='main']", 220),
    (".article-content", 230),
    (".entry-content", 230),
    (".post-content", 230),
    (".news-content", 220),
    (".g-articl-text", 220),
    (".content", 180),
    ("[class*='article']", 170),
    ("[class*='content']", 140),
    ("[id*='article']", 170),
    ("[id*='content']", 140),
)
_HTML_FALLBACK_TEXT_TAGS = ("h1", "h2", "h3", "h4", "p", "li", "blockquote")
_HTML_FALLBACK_REMOVE_TAGS = ("script", "style", "noscript", "svg", "canvas", "template", "iframe")
_HTML_FALLBACK_REMOVE_SELECTORS = ("nav", "footer", "aside", "form", "button", "[aria-hidden='true']")
_QUALITY_MIN_TEXT_CHARS = 80
_QUALITY_MIN_BLOCK_COUNT = 2
_QUALITY_MAX_LINK_DENSITY = 0.45
_NAVIGATION_TOKENS = (
    "subscribe",
    "sign in",
    "sign up",
    "newsletter",
    "advertisement",
    "cookie",
    "privacy policy",
    "terms of use",
    "all rights reserved",
)
_CHALLENGE_TOKENS = (
    "just a moment",
    "attention required",
    "verify you are human",
    "checking your browser",
    "access denied",
    "bot challenge",
    "captcha-delivery",
    "please enable js and disable any ad blocker",
)


@dataclass(frozen=True)
class ExtractionQualityReport:
    """主体抽取质量报告。"""

    is_usable: bool
    quality_flags: tuple[str, ...]
    content_stats: dict[str, Any]


@dataclass(frozen=True)
class ExtractedHtmlContent:
    """统一的 HTML 主体抽取结果。"""

    title: str
    html: str
    text: str
    extractor_source: str
    quality_report: ExtractionQualityReport


def extract_main_content(html: str, *, url: str = "") -> ExtractedHtmlContent:
    """抽取 HTML 主体内容。

    按 `trafilatura -> readability -> bs_fallback` 顺序尝试，并在前一路
    质量不足时自动回退。

    Args:
        html: 原始 HTML 文本。
        url: 可选原始 URL，仅用于提升第三方抽取器效果。

    Returns:
        统一抽取结果。

    Raises:
        RuntimeError: 当所有抽取路径都无法产出可用正文时抛出。
    """

    # 逐个尝试抽取器，首个可用即返回，避免后续抽取器的无谓开销
    extractors: list[Callable[[], Optional[ExtractedHtmlContent]]] = [
        lambda: extract_with_trafilatura(html, url=url),
        lambda: extract_with_readability(html, url=url),
        lambda: extract_with_bs_fallback(html),
    ]
    last_candidate: Optional[ExtractedHtmlContent] = None
    for extractor in extractors:
        candidate = extractor()
        if candidate is None:
            continue
        last_candidate = candidate
        if candidate.quality_report.is_usable:
            return candidate

    if last_candidate is None:
        raise RuntimeError("HTML 主体抽取失败：所有抽取器均未产出结果")
    if last_candidate.html.strip() or last_candidate.text.strip():
        return last_candidate
    raise RuntimeError("HTML 主体抽取失败：正文为空")


def extract_with_trafilatura(html: str, *, url: str = "") -> Optional[ExtractedHtmlContent]:
    """使用 trafilatura 抽取主体内容。"""

    try:
        import trafilatura
    except ImportError:
        return None

    try:
        extracted_html = trafilatura.extract(
            html,
            output_format="html",
            include_links=True,
            include_formatting=True,
            include_images=False,
            url=url or None,
        )
    except Exception:
        return None

    if not extracted_html:
        return None
    return _build_extracted_content(
        raw_html=html,
        extracted_html=str(extracted_html),
        extractor_source="trafilatura",
        extracted_title="",
    )


def extract_with_readability(html: str, *, url: str = "") -> Optional[ExtractedHtmlContent]:
    """使用 readability-lxml 抽取主体内容。"""

    del url
    try:
        from readability import Document
    except ImportError:
        return None

    try:
        document = Document(html)
        extracted_html = document.summary(html_partial=True)
        extracted_title = _normalize_whitespace(document.short_title() or "")
    except Exception:
        return None
    if not extracted_html:
        return None
    return _build_extracted_content(
        raw_html=html,
        extracted_html=str(extracted_html),
        extractor_source="readability",
        extracted_title=extracted_title,
    )


def extract_with_bs_fallback(html: str) -> Optional[ExtractedHtmlContent]:
    """使用 BeautifulSoup 规则化回退抽取主体内容。"""

    soup = BeautifulSoup(html, "html.parser")
    _remove_html_noise(soup)
    title = _extract_html_title(soup)
    candidate = _select_html_fallback_candidate(soup)
    if candidate is None:
        body = soup.body
        if body is None:
            return None
        candidate_html = body.decode_contents().strip()
    else:
        candidate_html = candidate.decode().strip()
    if not candidate_html:
        return None
    return _build_extracted_content(
        raw_html=html,
        extracted_html=candidate_html,
        extractor_source="bs_fallback",
        extracted_title=title,
    )


def assess_extraction_quality(*, title: str, html: str, text: str) -> ExtractionQualityReport:
    """评估主体抽取质量。"""

    soup = BeautifulSoup(html or "", "html.parser")
    paragraph_count = _count_text_blocks(soup)
    raw_text = _normalize_whitespace(text or "")
    raw_length = len(raw_text)
    link_text = _normalize_whitespace(" ".join(link.get_text(" ", strip=True) for link in soup.find_all("a")))
    link_density = (len(link_text) / raw_length) if raw_length else 0.0
    quality_flags: list[str] = []

    if raw_length < _QUALITY_MIN_TEXT_CHARS:
        quality_flags.append("too_short")
    if paragraph_count < _QUALITY_MIN_BLOCK_COUNT:
        quality_flags.append("too_few_blocks")
    if link_density > _QUALITY_MAX_LINK_DENSITY:
        quality_flags.append("high_link_density")
    if _looks_like_navigation_page(title=title, text=raw_text):
        quality_flags.append("navigation_like")
    if _looks_like_challenge_page(title=title, text=raw_text, html=html):
        quality_flags.append("challenge_like")

    return ExtractionQualityReport(
        is_usable=not quality_flags,
        quality_flags=tuple(quality_flags),
        content_stats={
            "text_length": raw_length,
            "paragraph_count": paragraph_count,
            "link_density": round(link_density, 4),
            "has_title": bool(title.strip()),
        },
    )


def _build_extracted_content(
    *,
    raw_html: str,
    extracted_html: str,
    extractor_source: str,
    extracted_title: str,
) -> ExtractedHtmlContent:
    """构造统一抽取结果对象。"""

    fallback_title = _extract_html_title(BeautifulSoup(raw_html, "html.parser"))
    soup = BeautifulSoup(extracted_html, "html.parser")
    text = _normalize_whitespace(soup.get_text("\n", strip=True))
    title = _normalize_whitespace(extracted_title or fallback_title)
    quality_report = assess_extraction_quality(
        title=title,
        html=extracted_html,
        text=text,
    )
    return ExtractedHtmlContent(
        title=title,
        html=extracted_html.strip(),
        text=text,
        extractor_source=extractor_source,
        quality_report=quality_report,
    )


def _remove_html_noise(soup: BeautifulSoup) -> None:
    """移除明显的模板噪音节点。"""

    for tag_name in _HTML_FALLBACK_REMOVE_TAGS:
        for node in soup.find_all(tag_name):
            node.decompose()
    for selector in _HTML_FALLBACK_REMOVE_SELECTORS:
        for node in soup.select(selector):
            node.decompose()


def _extract_html_title(soup: BeautifulSoup) -> str:
    """从 HTML 中提取较可信的标题。"""

    for attrs in (
        {"property": "og:title"},
        {"name": "twitter:title"},
        {"name": "title"},
    ):
        if "property" in attrs:
            node = soup.select_one(f'meta[property="{attrs["property"]}"]')
        else:
            node = soup.select_one(f'meta[name="{attrs["name"]}"]')
        if node is None:
            continue
        content = _normalize_whitespace(str(node.get("content", "") or ""))
        if content:
            return content
    if soup.title is None:
        return ""
    return _normalize_whitespace(soup.title.get_text(" ", strip=True))


def _extract_html_text_blocks(node: Any) -> list[str]:
    """从候选 HTML 节点提取正文文本块。"""

    blocks: list[str] = []
    seen: set[str] = set()
    for child in node.find_all(_HTML_FALLBACK_TEXT_TAGS):
        text = _normalize_whitespace(child.get_text(" ", strip=True))
        if len(text) < 8 or text in seen:
            continue
        blocks.append(text)
        seen.add(text)
        if len(blocks) >= _HTML_FALLBACK_MAX_BLOCKS:
            break
    if blocks:
        return blocks
    fallback_text = _normalize_whitespace(node.get_text("\n", strip=True))
    if fallback_text:
        return [fallback_text]
    return []


def _score_html_candidate(node: Any, *, selector_bonus: int, text: str, block_count: int) -> int:
    """给 HTML 正文候选节点打分。"""

    raw_text = _normalize_whitespace(node.get_text(" ", strip=True))
    raw_length = len(raw_text)
    if raw_length == 0:
        return -10_000
    link_text = _normalize_whitespace(" ".join(link.get_text(" ", strip=True) for link in node.find_all("a")))
    link_density_penalty = int((len(link_text) / raw_length) * 300) if link_text else 0
    return min(len(text), 6_000) + block_count * 35 + selector_bonus - link_density_penalty


def _select_html_fallback_candidate(soup: BeautifulSoup) -> Optional[Tag]:
    """从 HTML 文档中选择最可能的正文节点。"""

    best_node: Optional[Tag] = None
    best_score = -10_000
    visited_nodes: set[int] = set()
    for selector, selector_bonus in _HTML_FALLBACK_CANDIDATE_SELECTORS:
        for node in soup.select(selector):
            if not isinstance(node, Tag):
                continue
            node_key = id(node)
            if node_key in visited_nodes:
                continue
            visited_nodes.add(node_key)
            blocks = _extract_html_text_blocks(node)
            if not blocks:
                continue
            text = "\n\n".join(blocks).strip()
            if len(text) < _HTML_FALLBACK_MIN_TEXT_CHARS:
                continue
            score = _score_html_candidate(
                node,
                selector_bonus=selector_bonus,
                text=text,
                block_count=len(blocks),
            )
            if score > best_score:
                best_score = score
                best_node = node
    return best_node


def _count_text_blocks(soup: BeautifulSoup) -> int:
    """统计正文块数量。"""

    count = 0
    for tag_name in _HTML_FALLBACK_TEXT_TAGS:
        count += len(soup.find_all(tag_name))
    return count


def _looks_like_navigation_page(*, title: str, text: str) -> bool:
    """判断正文是否更像导航/壳页面。"""

    lower_text = f"{title} {text}".lower()
    matched = sum(1 for token in _NAVIGATION_TOKENS if token in lower_text)
    return matched >= 2


def _looks_like_challenge_page(*, title: str, text: str, html: str) -> bool:
    """判断正文是否更像 challenge/access gate。"""

    lower_text = f"{title} {text} {html}".lower()
    return any(token in lower_text for token in _CHALLENGE_TOKENS)
