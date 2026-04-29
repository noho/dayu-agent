"""SC 13 系列表单专项章节处理器。"""

from __future__ import annotations

from typing import Optional

from dayu.engine.processors.source import Source
from dayu.fins.processors.sec_report_form_common import _BaseSecReportFormProcessor

from .sc13_form_common import (
    _SC13_FORMS,
    _build_sc13_markers,
    _has_sufficient_sc13_markers,
)
from .sec_form_section_common import (
    _is_table_placeholder_dominant_text,
    _normalize_whitespace,
    _safe_virtual_document_text,
)


class Sc13FormProcessor(_BaseSecReportFormProcessor):
    """SC 13 系列表单专项处理器。"""

    PARSER_VERSION = "sc13_section_processor_v1.0.0"
    _SUPPORTED_FORMS = _SC13_FORMS

    def __init__(
        self,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> None:
        """初始化处理器。

        Args:
            source: 文档来源抽象。
            form_type: 可选表单类型。
            media_type: 可选媒体类型。

        Returns:
            无。

        Raises:
            ValueError: 参数非法时抛出。
            RuntimeError: 解析失败时抛出。
        """

        super().__init__(source=source, form_type=form_type, media_type=media_type)

    def _build_markers(self, full_text: str) -> list[tuple[int, Optional[str]]]:
        """构建 SC13 专项边界。

        Args:
            full_text: 文档全文。

        Returns:
            `(start_index, title)` 列表。

        Raises:
            RuntimeError: 构建失败时抛出。
        """

        return _build_sc13_markers(full_text)

    def _collect_full_text_from_base(self) -> str:
        """收集 SC13 切分输入文本。

        优先使用父类“章节拼接文本”；当 base 文本无法切出有效 SC13 标记时，
        自动尝试 `document.text()`，避免 `Item 1..7` 被表格占位符或结构化抽取策略吞掉。

        Args:
            无。

        Returns:
            用于 SC13 专项切分的全文字符串。

        Raises:
            RuntimeError: 文本收集失败时抛出。
        """

        base_text = super()._collect_full_text_from_base()
        if _has_sufficient_sc13_markers(base_text):
            return base_text

        # base 文本缺 marker 时，优先尝试 document.text() 补全完整语义。
        document_text = _safe_virtual_document_text(self)
        if _has_sufficient_sc13_markers(document_text):
            return document_text

        # 若 base 主要由表格占位符构成，则优先使用更长的 document 文本。
        if _is_table_placeholder_dominant_text(base_text):
            if len(document_text) > len(_normalize_whitespace(base_text)):
                return document_text
        return base_text or document_text


__all__ = ["Sc13FormProcessor"]
