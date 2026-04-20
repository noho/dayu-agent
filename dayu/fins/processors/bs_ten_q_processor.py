"""基于 BeautifulSoup 的 10-Q 表单专项处理器。

本模块实现基于 BSProcessor（BeautifulSoup）的 10-Q 切分策略，
与 ``ten_q_processor.py``（基于 edgartools/SecProcessor）平行：
- 共享同一套两阶段有序 Item 选取逻辑（``_build_ten_q_markers``）；
- HTML 解析完全由 BeautifulSoup 驱动，无 edgartools 黑箱；
- XBRL 通过独立文件发现加载，不依赖 edgartools 文档对象。

设计意图：
- 为 10-Q 提供 BS 路线主处理器（priority 200），
  ``TenQFormProcessor`` 降级为回退（priority 190）；
- BS 路线独立提供 XBRL 财务报表能力，提升 D_consistency 维度评分。
"""

from __future__ import annotations

from typing import Optional

from dayu.engine.processors.source import Source

from .bs_report_form_common import _BaseBsReportFormProcessor
from .ten_q_form_common import (
    _build_ten_q_markers,
    expand_ten_q_virtual_sections_content,
)


class BsTenQFormProcessor(_BaseBsReportFormProcessor):
    """基于 BeautifulSoup 的 10-Q 表单专项处理器。

    继承链：
    ``BsTenQFormProcessor → _BaseBsReportFormProcessor
    → _VirtualSectionProcessorMixin → FinsBSProcessor → BSProcessor``

    与 ``TenQFormProcessor``（基于 SecProcessor）平行，
    共享 ``_build_ten_q_markers()`` marker 策略。
    """

    PARSER_VERSION = "bs_ten_q_processor_v1.0.0"
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
        """构建 10-Q 专项边界标记。

        复用 ``_build_ten_q_markers()``，该函数仅依赖纯文本正则扫描，
        与底层 HTML 解析引擎无关。

        Args:
            full_text: 文档全文。

        Returns:
            ``(start_index, title)`` 列表。

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


__all__ = ["BsTenQFormProcessor"]
