"""DEF 14A 表单专项章节处理器。"""

from __future__ import annotations

from typing import Optional

from dayu.engine.processors.source import Source

from .def14a_form_common import (
    _DEF14A_FORMS,
    _build_def14a_markers,
)
from .sec_report_form_common import _BaseSecReportFormProcessor


class Def14AFormProcessor(_BaseSecReportFormProcessor):
    """DEF 14A 表单专项处理器。"""

    PARSER_VERSION = "def14a_section_processor_v1.0.0"
    _SUPPORTED_FORMS = _DEF14A_FORMS

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
        """构建 DEF 14A 专项边界。

        Args:
            full_text: 文档全文。

        Returns:
            `(start_index, title)` 列表。

        Raises:
            RuntimeError: 构建失败时抛出。
        """

        return _build_def14a_markers(full_text)


__all__ = ["Def14AFormProcessor"]
