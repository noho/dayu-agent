"""审计 Markdown 文本操作与内容提取纯函数集合。

本模块提供章节正文的 Markdown 格式化、标题结构操作、证据行规范化、
文本匹配与内容提取等无状态纯函数，供审计规则、修复执行器与下游
协调器调用。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from dayu.services.internal.write_pipeline.enums import RepairTargetKind
from dayu.services.internal.write_pipeline.models import ChapterResult, ChapterTask
from dayu.services.internal.write_pipeline.source_list_builder import (
    looks_like_evidence_item,
)

# 模块级正则常量，避免每次函数调用重复编译。
# 注意：代码块正则使用贪婪 `[\s\S]*`，以便正确处理章节正文中嵌套的 ```json``` 等
# 子代码块（外层 ```markdown ... ``` 需整体保留）。该贪婪模式的已知副作用是：
# 当输入包含多个平级的 ```markdown fence 时，会把第一个 ``` 到最后一个 ``` 之间
# 的所有内容（包括中间的说明文字）当成一个代码块提取。如需精确处理这种平级
# 多 fence 场景，需改为 fence 计数/行级解析，不能单纯把 `*` 换成 `*?`
# （会截断合法的嵌套子代码块）。
_MARKDOWN_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:markdown)?\s*([\s\S]*)```", re.IGNORECASE
)
_MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_EVIDENCE_SECTION_PATTERN = re.compile(
    r"\n###\s+证据与出处\b.*?(?=\n##[^#]|\Z)",
    re.DOTALL,
)

_HEADING_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "【": "[",
        "】": "]",
        "〔": "[",
        "〕": "]",
        "｛": "{",
        "｝": "}",
        "“": '"',
        "”": '"',
        "「": '"',
        "」": '"',
        "『": '"',
        "』": '"',
        "《": '"',
        "》": '"',
        "〈": '"',
        "〉": '"',
        "‘": "'",
        "’": "'",
        "、": ",",
        "；": ";",
        "。": ".",
        "—": "-",
        "–": "-",
        "―": "-",
        "－": "-",
    }
)


_PLACEHOLDER_PATTERNS = [
    r"【\s*占位符[^】]*】",           # 标准格式：【占位符：xxx】
    r"\[[^\]]*未[^\]]*披露[^\]]*\]",  # 非标：[...未...披露...]
    r"\[[^\]]*未在[^\]]*\]",          # 非标：[...未在...]
    r"\[[^\]]*待[^\]]*\]",            # 非标：[待补充]、[待确认]
    r"\{\{[^{}]+\}\}",               # 双花括号模板变量
    r"\bTODO\b",
    r"\bTBD\b",
    r"待补充",
    r"待确认",
]


def _extract_markdown_content(raw_text: str) -> str:
    """从模型返回中提取 Markdown 代码块正文。

    Args:
        raw_text: 原始模型输出。

    Returns:
        提取后的正文文本。

    Raises:
        无。
    """

    match = _MARKDOWN_CODE_BLOCK_PATTERN.search(raw_text)
    if match is None:
        return raw_text.strip()
    return match.group(1).strip()


def _extract_json_text(text: str) -> str:
    """从文本中抽取 JSON 子串。

    Args:
        text: 原始文本。

    Returns:
        JSON 字符串，失败返回空字符串。

    Raises:
        无。
    """

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return ""
    return text[start : end + 1]


def _normalize_heading_text(text: str) -> str:
    """归一化标题文本，消除常见中英文标点差异。

    同时剥离误带的 Markdown heading 标记前缀（如 ``### ``）。

    Args:
        text: 原始标题文本。

    Returns:
        归一化后的标题文本。
    """

    # 剥离误带的 markdown heading 标记（如 "### 证据与出处" → "证据与出处"）
    normalized = re.sub(r"^#{1,6}\s+", "", text)
    # 先做 Unicode 兼容归一化，统一全角英数与常见全角 ASCII 标点。
    normalized = unicodedata.normalize("NFKC", normalized)
    # 再补齐 NFKC 不会处理的中文标点，避免结构标题仅因中英文标点差异被误判。
    normalized = normalized.translate(_HEADING_PUNCTUATION_TRANSLATION)
    # 合并连续空白
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_structure_heading(text: str) -> str:
    """归一化结构比对用标题文本。

    Args:
        text: 原始标题文本。

    Returns:
        仅保留结构相关差异的归一化标题文本。

    Raises:
        无。
    """

    return re.sub(r"\s+", "", _normalize_heading_text(text))


def _extract_markdown_headings(markdown_text: str) -> list[tuple[int, str]]:
    """提取 Markdown 中的可见标题序列。

    Args:
        markdown_text: Markdown 文本。

    Returns:
        标题序列，每项为 ``(level, title)``。

    Raises:
        无。
    """

    headings: list[tuple[int, str]] = []
    for match in _MARKDOWN_HEADING_PATTERN.finditer(markdown_text):
        headings.append((len(match.group(1)), match.group(2).strip()))
    return headings


def _matches_skeleton_structure(
    content: str,
    skeleton: str,
    *,
    allowed_conditional_headings: set[str] | None = None,
) -> bool:
    """检查正文是否按骨架标题顺序输出。

    Args:
        content: 章节正文。
        skeleton: 章节骨架。
        allowed_conditional_headings: 允许出现的条件型可见小节标题集合。

    Returns:
        结构匹配返回 ``True``，否则返回 ``False``。

    Raises:
        无。
    """

    if not skeleton.strip():
        return True
    skeleton_headings = _extract_markdown_headings(skeleton)
    # 若骨架只提供章节标题而未提供可见子标题，则不做严格结构比对，
    # 避免把极简测试骨架或占位骨架误判为结构性失败。
    if len(skeleton_headings) <= 1:
        return True
    content_headings = _extract_markdown_headings(content)
    normalized_skeleton = [_normalize_structure_heading(title) for _level, title in skeleton_headings]
    normalized_allowed = {
        _normalize_structure_heading(title)
        for title in (allowed_conditional_headings or set())
        if title.strip()
    }
    skeleton_index = 0
    for _level, title in content_headings:
        normalized_title = _normalize_structure_heading(title)
        if skeleton_index < len(normalized_skeleton) and normalized_title == normalized_skeleton[skeleton_index]:
            skeleton_index += 1
            continue
        if normalized_title in normalized_allowed:
            continue
        return False
    return skeleton_index == len(normalized_skeleton)


def _find_enclosing_heading_section(
    *,
    markdown_text: str,
    label_text: str,
    headings: list[dict[str, Any]],
) -> tuple[int, int] | None:
    """当 label_text 不是 heading 而是 bullet 标签时，查找其所在 heading section 的字符区间。

    在 markdown_text 中搜索包含 label_text 的行（通常是骨架 ``- xxx：`` 格式的 slot），
    然后回退到该行所属的最近上级 heading section。

    Args:
        markdown_text: 章节正文 Markdown。
        label_text: 经归一化后的标签文本。
        headings: 已解析的 heading 列表（含 level/title/start/end）。

    Returns:
        `(start, end)` 字符区间；若未找到唯一匹配行或无上级 heading，返回 `None`。
    """

    if not headings:
        return None
    # 在正文中搜索包含 label_text 的行
    # 使用去空格比较，消除 model 输出与骨架之间的空格差异（如 "2-3 条" vs "2-3条"）
    normalized_label = re.sub(r"\s+", "", _normalize_heading_text(label_text))
    line_positions: list[int] = []
    for line_match in re.finditer(r"^.+$", markdown_text, re.MULTILINE):
        normalized_line = re.sub(r"\s+", "", _normalize_heading_text(line_match.group()))
        if normalized_label in normalized_line:
            line_positions.append(line_match.start())
    if len(line_positions) != 1:
        # 未找到或多处匹配，无法确定唯一归属
        return None
    label_pos = line_positions[0]
    # 找 label_pos 所属的最近上级 heading
    enclosing_heading = None
    for heading in headings:
        if heading["start"] <= label_pos:
            enclosing_heading = heading
        else:
            break
    if enclosing_heading is None:
        return None
    enclosing_index = headings.index(enclosing_heading)
    section_end = len(markdown_text)
    for next_heading in headings[enclosing_index + 1 :]:
        if next_heading["level"] <= enclosing_heading["level"]:
            section_end = next_heading["start"]
            break
    return enclosing_heading["start"], section_end


def _find_markdown_section_span(*, markdown_text: str, heading_text: str) -> tuple[int, int] | None:
    """查找指定 Markdown 标题对应 section 的字符区间。

    匹配策略按优先级依次尝试：
    1. 精确匹配 heading 标题。
    2. 归一化匹配（剥离 ``###`` 前缀、全半角括号等符号差异）。
    3. 当 heading_text 实际是骨架 bullet 标签而非 heading 时，回退到查找其所在 heading section。

    Args:
        markdown_text: 章节正文 Markdown。
        heading_text: 标题文本（可能是真实 heading，也可能是模型误传的 bullet 标签）。

    Returns:
        `(start, end)` 字符区间；若标题不存在或不唯一，返回 `None`。

    Raises:
        无。
    """

    headings: list[dict[str, Any]] = []
    for match in _MARKDOWN_HEADING_PATTERN.finditer(markdown_text):
        headings.append(
            {
                "level": len(match.group(1)),
                "title": match.group(2).strip(),
                "start": match.start(),
                "end": match.end(),
            }
        )
    # 策略 1: 精确匹配
    matches = [item for item in headings if item["title"] == heading_text]
    if len(matches) != 1:
        # 策略 2: 归一化匹配（剥离 ### 前缀、全半角括号等）
        normalized_target = _normalize_heading_text(heading_text)
        matches = [item for item in headings if _normalize_heading_text(item["title"]) == normalized_target]
    if len(matches) == 1:
        current_heading = matches[0]
        current_index = headings.index(current_heading)
        section_end = len(markdown_text)
        for next_heading in headings[current_index + 1 :]:
            if next_heading["level"] <= current_heading["level"]:
                section_end = next_heading["start"]
                break
        return current_heading["start"], section_end
    # 策略 3: heading_text 可能是骨架 bullet 标签，回退到查找其所在 heading section
    return _find_enclosing_heading_section(
        markdown_text=markdown_text,
        label_text=heading_text,
        headings=headings,
    )


def _heading_exists_in_markdown(*, markdown_text: str, heading_text: str) -> bool:
    """判断标题是否真实存在于当前 Markdown 标题集合中。

    Args:
        markdown_text: 当前章节正文。
        heading_text: 待校验标题文本。

    Returns:
        若标题存在则返回 ``True``，否则返回 ``False``。

    Raises:
        无。
    """

    normalized_target = _normalize_heading_text(heading_text)
    return any(
        _normalize_heading_text(title) == normalized_target
        for _level, title in _extract_markdown_headings(markdown_text)
    )


def _build_current_visible_headings_block(markdown_text: str) -> str:
    """构建当前正文中的真实可见标题列表。

    Args:
        markdown_text: 当前章节正文。

    Returns:
        供 repair prompt 使用的标题列表 Markdown。

    Raises:
        无。
    """

    headings = _extract_markdown_headings(markdown_text)
    if not headings:
        return "- （当前正文中没有可见标题）"
    return "\n".join(f"- {'#' * level} {title}" for level, title in headings)


def _build_allowed_conditional_headings(task: ChapterTask) -> set[str]:
    """构建章节允许出现的条件型可见标题集合。

    `ITEM_RULE` 当前语义是“满足 when 时，可作为条件小节或条件条目显式出现”。
    因此程序审计在做 `P1` 结构校验时，必须把当前章节的 `item_rules.item`
    视为允许的条件标题，而不是一律当成结构错位。

    Args:
        task: 当前章节任务。

    Returns:
        允许的条件型可见标题集合。

    Raises:
        无。
    """

    return {rule.item.strip() for rule in task.item_rules if rule.item.strip()}


def _has_evidence_section(content: str) -> bool:
    """检测章节正文是否包含"### 证据与出处"小节。

    Args:
        content: 章节正文。

    Returns:
        包含则返回 True，否则 False。

    Raises:
        无。
    """

    for line in content.splitlines():
        if line.strip() == "### 证据与出处":
            return True
    return False


def _strip_generated_parenthetical_summary(location: str) -> str:
    """删除定位段末尾显然由模型生成的摘要型括号说明。

    Args:
        location: evidence 定位段。

    Returns:
        删除摘要型括号后的定位段。

    Raises:
        无。
    """

    match = re.match(r"^(?P<head>.*?)(?P<suffix>\s*\((?P<inner>[^()]*)\))$", location)
    if match is None:
        return location
    inner = match.group("inner").strip()
    if not inner:
        return match.group("head").strip()
    tokens = [token.strip() for token in inner.split(",") if token.strip()]
    if not tokens:
        return match.group("head").strip()
    if all(re.fullmatch(r"\d{4}", token) for token in tokens):
        return location
    if len(tokens) >= 2:
        return match.group("head").strip()
    return location


def _normalize_evidence_location_segment(location: str) -> str:
    """规范化 evidence line 的定位段。

    Args:
        location: 定位段原文。

    Returns:
        清理后的定位段。

    Raises:
        无。
    """

    normalized = re.sub(r"\s+", " ", location).strip()
    normalized = _strip_generated_parenthetical_summary(normalized)
    return normalized


def _normalize_evidence_line(line: str) -> str:
    """规范化单条 evidence line。

    Args:
        line: 原始 evidence line。

    Returns:
        规范化后的 evidence line。

    Raises:
        无。
    """

    prefix = line[: len(line) - len(line.lstrip())]
    body = line.strip()
    if not body.startswith("- "):
        return line
    text = body[2:].strip()
    parts = [part.strip() for part in text.split(" | ")]
    if len(parts) < 4:
        return line
    parts[-1] = _normalize_evidence_location_segment(parts[-1])
    return f"{prefix}- {' | '.join(parts)}"


def _normalize_chapter_markdown_for_audit(content: str) -> str:
    """对章节正文做送审前的确定性预处理。

    该预处理仅修正可机械判定的证据格式噪音，不改变正文事实语义。

    Args:
        content: 原始章节正文。

    Returns:
        规范化后的章节正文。

    Raises:
        无。
    """

    lines = content.splitlines()
    normalized_lines: list[str] = []
    in_evidence_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == "### 证据与出处":
            in_evidence_section = True
            normalized_lines.append(line)
            continue
        if in_evidence_section and stripped.startswith("#"):
            in_evidence_section = False
        if in_evidence_section:
            if stripped.startswith("```"):
                continue
            if stripped.startswith("- "):
                normalized_lines.append(_normalize_evidence_line(line))
                continue
            if looks_like_evidence_item(stripped):
                indent = line[: len(line) - len(line.lstrip())]
                normalized_lines.append(_normalize_evidence_line(f"{indent}- {stripped}"))
                continue
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _collect_lines_between_heading(lines: list[str], heading: str) -> list[str]:
    """收集指定三级标题到下一个三级标题之间的文本。

    Args:
        lines: 行文本列表。
        heading: 目标三级标题。

    Returns:
        文本行列表。

    Raises:
        无。
    """

    start_index = -1
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start_index = index
            break
    if start_index < 0:
        return []

    collected: list[str] = []
    for line in lines[start_index + 1 :]:
        stripped = line.strip()
        if stripped.startswith("### "):
            break
        if stripped:
            collected.append(line)
    return collected


def _extract_evidence_section_block(chapter_markdown: str) -> str:
    """提取章节中的“证据与出处”小节块。

    Args:
        chapter_markdown: 章节 Markdown 全文。

    Returns:
        `### 证据与出处` 小节块；若不存在则返回空字符串。

    Raises:
        无。
    """

    lines = chapter_markdown.splitlines()
    evidence_lines = _collect_lines_between_heading(lines, "### 证据与出处")
    if not evidence_lines:
        return ""
    blocks = ["### 证据与出处", ""]
    blocks.extend(evidence_lines)
    return "\n".join(blocks).strip()


def _replace_evidence_section_block(*, chapter_markdown: str, evidence_section: str) -> str:
    """替换章节中的“证据与出处”小节块。

    Args:
        chapter_markdown: 章节 Markdown 全文。
        evidence_section: 新的 evidence section 文本，必须包含三级标题。

    Returns:
        替换后的章节 Markdown。

    Raises:
        ValueError: 当正文缺少 evidence section 或新块不合法时抛出。
    """

    if not evidence_section.strip().startswith("### 证据与出处"):
        raise ValueError("新的 evidence section 必须以“### 证据与出处”开头")
    span = _find_markdown_section_span(markdown_text=chapter_markdown, heading_text="证据与出处")
    if span is None:
        raise ValueError("当前正文缺少“### 证据与出处”小节，无法替换")
    section_start, section_end = span
    replaced = (
        chapter_markdown[:section_start].rstrip()
        + "\n\n"
        + evidence_section.strip()
        + chapter_markdown[section_end:]
    )
    return replaced.strip() + "\n"


def _strip_evidence_section(content: str) -> str:
    """剥除章节正文中的"证据与出处"小节。

    Args:
        content: 章节 Markdown 全文。

    Returns:
        删除"### 证据与出处"及其后续内容后的文本；若无该小节则原样返回。

    Raises:
        无。
    """

    # 匹配 ### 证据与出处 直到下一个同级或更高级标题，或文末
    return _EVIDENCE_SECTION_PATTERN.sub("", content).rstrip()


def _normalize_patch_match_text(text: str) -> str:
    """归一化 repair patch 目标文本，降低轻微格式漂移的影响。

    Args:
        text: 原始文本。

    Returns:
        归一化后的文本。

    Raises:
        无。
    """

    normalized = text.strip()
    normalized = re.sub(r"[\"'“”‘’`]", "", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _find_normalized_line_like_spans(text: str, target: str, *, bullet_only: bool) -> list[tuple[int, int]]:
    """按行或 bullet 单元查找规范化匹配区间。

    Args:
        text: 待搜索文本。
        target: 目标片段原文。
        bullet_only: 为 True 时仅匹配 bullet 行。

    Returns:
        命中的 `(start, end)` 区间列表。

    Raises:
        无。
    """

    normalized_target = _normalize_patch_match_text(target)
    if not normalized_target:
        return []
    spans: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped:
            offset += len(line)
            continue
        if bullet_only and not stripped.startswith(("- ", "* ")):
            offset += len(line)
            continue
        normalized_line = _normalize_patch_match_text(stripped)
        if normalized_line and (
            normalized_line == normalized_target
            or normalized_line.startswith(normalized_target)
            or normalized_target.startswith(normalized_line)
        ):
            spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def _find_normalized_paragraph_spans(text: str, target: str) -> list[tuple[int, int]]:
    """按段落单元查找规范化匹配区间。

    Args:
        text: 待搜索文本。
        target: 目标片段原文。

    Returns:
        命中的 `(start, end)` 区间列表。

    Raises:
        无。
    """

    normalized_target = _normalize_patch_match_text(target)
    if not normalized_target:
        return []
    spans: list[tuple[int, int]] = []
    paragraph_start = 0
    current_start = 0
    lines = text.splitlines(keepends=True)
    buffer: list[str] = []
    for line in lines + ["\n"]:
        if line.strip():
            if not buffer:
                current_start = paragraph_start
            buffer.append(line)
        else:
            if buffer:
                paragraph_text = "".join(buffer)
                normalized_paragraph = _normalize_patch_match_text(paragraph_text)
                if normalized_paragraph and (
                    normalized_paragraph == normalized_target
                    or normalized_paragraph.startswith(normalized_target)
                    or normalized_target.startswith(normalized_paragraph)
                ):
                    spans.append((current_start, current_start + len(paragraph_text)))
                buffer = []
            paragraph_start += len(line)
            continue
        paragraph_start += len(line)
    return spans


def _find_normalized_match_spans(text: str, target: str, target_kind: str) -> list[tuple[int, int]]:
    """按更稳定的文本单元查找规范化匹配区间。

    Args:
        text: 待搜索文本。
        target: 目标片段原文。
        target_kind: 目标片段类型，支持 substring/line/bullet/paragraph。

    Returns:
        规范化后命中的 `(start, end)` 区间列表。

    Raises:
        无。
    """

    if not target:
        return []
    if target_kind == RepairTargetKind.SUBSTRING:
        return _find_normalized_line_like_spans(text=text, target=target, bullet_only=False)
    if target_kind == RepairTargetKind.LINE:
        return _find_normalized_line_like_spans(text=text, target=target, bullet_only=False)
    if target_kind == RepairTargetKind.BULLET:
        return _find_normalized_line_like_spans(text=text, target=target, bullet_only=True)
    return _find_normalized_paragraph_spans(text=text, target=target)


def _find_all_occurrences(text: str, target: str) -> list[int]:
    """返回目标子串在文本中的所有非重叠命中起点。

    Args:
        text: 待搜索文本。
        target: 目标子串。

    Returns:
        所有命中起点的升序列表。

    Raises:
        无。
    """

    if not target:
        return []
    positions: list[int] = []
    start = 0
    while True:
        found = text.find(target, start)
        if found < 0:
            break
        positions.append(found)
        start = found + len(target)
    return positions


def _normalize_line_for_match(text: str) -> str:
    """归一化整行文本，供去重与稳定匹配使用。

    Args:
        text: 原始行文本。

    Returns:
        去空白后的归一化结果。

    Raises:
        无。
    """

    return re.sub(r"\s+", "", text).strip()


def _should_run_fix_placeholders(chapter_markdown: str) -> bool:
    """判断是否需要执行占位符补强步骤。

    Args:
        chapter_markdown: 当前章节正文。

    Returns:
        命中占位符特征时返回 `True`。

    Raises:
        无。
    """

    if not chapter_markdown.strip():
        return False
    for pattern in _PLACEHOLDER_PATTERNS:
        if re.search(pattern, chapter_markdown, flags=re.IGNORECASE):
            return True
    return False


def _extract_overview_summary(chapter_markdown: str) -> str:
    """从章节中提取第0章可消费的结构化输入。

    当前默认只抽取：
    - `### 结论要点`

    第0章的目标是把前文章节判断链压缩成封面页，而不是重新消费
    “证据与出处”或把整章全文再次塞给模型。这里故意只保留最小判断
    链，降低第0章的认知负担。

    Args:
        chapter_markdown: 章节正文。

    Returns:
        结构化摘要文本。

    Raises:
        无。
    """

    lines = chapter_markdown.splitlines()
    summary_lines = _collect_lines_between_heading(lines, "### 结论要点")
    blocks: list[str] = []
    if summary_lines:
        blocks.append("### 结论要点")
        blocks.extend(summary_lines)
    return "\n".join(blocks).strip()


def _extract_research_decision_summary(chapter_markdown: str) -> str:
    """从章节中提取第10章可消费的结构化摘要。

    当前默认只抽取：
    - `### 结论要点`
    - `### 证据与出处`

    这样既能保留章节判断链，又避免把整章全文再次喂给第10章。后续若要改成
    “结论+证据+未决项”或“章节全文输入”，只需替换这层提取逻辑。

    Args:
        chapter_markdown: 章节正文。

    Returns:
        结构化摘要文本。

    Raises:
        无。
    """

    lines = chapter_markdown.splitlines()
    summary_lines = _collect_lines_between_heading(lines, "### 结论要点")
    evidence_lines = _collect_lines_between_heading(lines, "### 证据与出处")

    blocks: list[str] = []
    if summary_lines:
        blocks.append("### 结论要点")
        blocks.extend(summary_lines)
    if evidence_lines:
        blocks.append("### 证据与出处")
        blocks.extend(evidence_lines)
    return "\n".join(blocks).strip()


def _collect_all_evidence_items(
    chapter_results: dict[str, ChapterResult],
    *,
    ordered_titles: list[str] | None = None,
) -> list[str]:
    """汇总全部章节证据条目。

    Args:
        chapter_results: 章节结果映射。
        ordered_titles: 可选的章节标题顺序；传入时按该顺序聚合证据。

    Returns:
        聚合证据条目列表。

    Raises:
        无。
    """

    items: list[str] = []
    if ordered_titles is None:
        results_iterable = chapter_results.values()
    else:
        results_iterable = [
            chapter_results[title]
            for title in ordered_titles
            if title in chapter_results and chapter_results[title].content
        ]
    for result in results_iterable:
        items.extend(result.evidence_items)
    return items

