"""20-F 表单专项处理器。

本模块实现 20-F（外国私人发行人年报）的专项切分策略：
- 以 ``Item`` 为主轴切分章节（SEC Form 20-F 法定 Item 1–19）；
- 基于 SEC 法定 Part→Item 映射为每个 Item 标注所属 Part；
- 为关键 Item 附加 SEC 标准描述，提升 LLM 章节定位效率；
- 在尾段补充 ``SIGNATURE`` 章节。

SEC 20-F 结构参考：
- SEC Form 20-F General Instructions
- 17 CFR Part 249 §249.220f

与 10-K 的核心差异：
- Item 编号全局唯一（Part 前缀为信息性，非必需用于消歧）；
- Item 16 为 [Reserved]，16A–16J 为独立治理披露子项；
- 多数 FPI 采用 IFRS，Item 18 为主财务报表章节。
"""

from __future__ import annotations

from typing import Optional

from dayu.engine.processors.source import Source

from .twenty_f_form_common import (
    _build_twenty_f_markers,
    _select_preferred_twenty_f_text,
    _trim_twenty_f_source_text,
)
from .sec_report_form_common import (
    _BaseSecReportFormProcessor,
    _extract_source_text_preserving_lines,
)

class TwentyFFormProcessor(_BaseSecReportFormProcessor):
    """20-F 表单专项处理器（edgartools 路线）。

    当 ``BsTwentyFFormProcessor``（BS 路线）不可用时作为回退。
    """

    PARSER_VERSION = "twenty_f_section_processor_v2.1.5"
    _SUPPORTED_FORMS = frozenset({"20-F"})
    _ENABLE_FAST_SECTION_BUILD = True
    _FAST_SECTION_BUILD_SINGLE_FULL_TEXT = True

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
        """构建 20-F 专项边界。

        Args:
            full_text: 文档全文。

        Returns:
            `(start_index, title)` 列表。

        Raises:
            RuntimeError: 构建失败时抛出。
        """

        return _build_twenty_f_markers(full_text)

    def _collect_document_text(self) -> str:
        """提取尽量保留行边界的 20-F 全文文本。

        20-F 的 Item 标题经常以表格行或独立块呈现。edgartools 的
        ``document.text()`` 在部分文档上会把这些边界压平，导致 marker
        构建失败并退回到错位的章节重建。这里优先直接从源 HTML 提取
        带换行的文本，失败时再回退父类实现。

        Args:
            无。

        Returns:
            尽量保留标题换行结构的全文文本；失败时回退父类结果。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        parsed_text = super()._collect_document_text()
        extracted = _trim_twenty_f_source_text(
            _extract_source_text_preserving_lines(self._source)
        )
        return _select_preferred_twenty_f_text(
            source_text=extracted,
            parsed_text=parsed_text,
        )



__all__ = ["TwentyFFormProcessor"]
