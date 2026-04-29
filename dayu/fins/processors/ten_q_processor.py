"""10-Q 表单专项处理器。

本模块实现 10-Q 的专项切分策略：
- 基于 SEC Form 10-Q 法定结构（Part I Items 1-4, Part II Items 1-6+1A）；
- 使用 SEC 法定 Part 标题锚定内容区边界，避免 TOC 缓冲区过宽导致误判；
- 两阶段有序选取：Phase 1 在 Part I 区域选取 Items 1-4，
  Phase 2 在 Part II 区域选取 Items 1-6+1A；
- 在尾段补充 ``SIGNATURE`` 章节。
"""

from __future__ import annotations

from typing import Optional

from dayu.engine.processors.source import Source

from .ten_q_form_common import (
    _build_ten_q_markers,
    expand_ten_q_virtual_sections_content,
)
from .sec_report_form_common import _BaseSecReportFormProcessor


class TenQFormProcessor(_BaseSecReportFormProcessor):
    """10-Q 表单专项处理器。

    基于 edgartools/SecProcessor 技术路线的 10-Q 处理器。
    当 BsTenQFormProcessor（BS 路线）不可用时作为回退。
    """

    PARSER_VERSION = "ten_q_section_processor_v2.0.0"
    _SUPPORTED_FORMS = frozenset({"10-Q"})

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
        # 在父类初始化完成后再次执行一次 10-Q 专项正文修复，确保最终暴露给
        # FinsToolService 的虚拟章节已应用最新边界收敛逻辑。
        self._postprocess_virtual_sections(self._collect_document_text())

    def _build_markers(self, full_text: str) -> list[tuple[int, Optional[str]]]:
        """构建 10-Q 专项边界。

        Args:
            full_text: 文档全文。

        Returns:
            `(start_index, title)` 列表。

        Raises:
            RuntimeError: 构建失败时抛出。
        """

        return _build_ten_q_markers(full_text)

    def _postprocess_virtual_sections(self, full_text: str) -> None:
        """对 10-Q 虚拟章节应用专项正文修复。

        Args:
            full_text: 用于切分的完整文本。

        Returns:
            无。

        Raises:
            RuntimeError: 修复失败时抛出。
        """

        expand_ten_q_virtual_sections_content(
            full_text=full_text,
            virtual_sections=self._virtual_sections,
        )
__all__ = ["TenQFormProcessor"]
