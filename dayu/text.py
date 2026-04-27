"""跨层通用文本处理工具。

本模块提供各层（UI / Service / Host / Agent）共享的文本处理函数，
位于包根级别，任何层均可干净导入，不违反分层架构约束。
"""

from __future__ import annotations


def strip_markdown_fence(text: str | None) -> str:
    """剥离模型输出外层的 Markdown 代码块围栏。

    仅匹配**整个文本**被单一 ````` `` 围栏包裹的场景：
    首行以 ````` `` 开头、末行以 ````` `` 结尾、中间至少一行正文。
    不处理嵌套围栏、不处理文本中间出现的围栏片段。

    Args:
        text: 原始输出文本；接受 ``None``（视为空字符串）。

    Returns:
        剥离围栏后的正文；无围栏时返回去空白后的原文。
    """

    stripped = (text or "").strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


__all__ = ["strip_markdown_fence"]
