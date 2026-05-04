"""fins 业务域 Docling 处理器。

该处理器复用 engine `DoclingProcessor` 的通用解析能力，
并在业务域层补充表格金融语义标注。
"""

from __future__ import annotations

from typing import Optional

from dayu.engine.processors.docling_processor import DoclingProcessor
from dayu.engine.processors.source import Source

from .financial_enhancer import FinsProcessorMixin, relabel_tables


class FinsDoclingProcessor(FinsProcessorMixin, DoclingProcessor):
    """fins 业务域 Docling 处理器。"""

    PARSER_VERSION = "fins_docling_processor_v1.0.0"

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
            ValueError: 源文件不存在或参数非法时抛出。
            RuntimeError: 解析失败时抛出。
        """

        super().__init__(source=source, form_type=form_type, media_type=media_type)
        relabel_tables(self._tables, docling_document=self._document)
