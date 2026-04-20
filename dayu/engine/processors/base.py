"""处理器协议定义。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, Optional, Protocol, TypedDict

from .source import Source


class SectionSummary(TypedDict):
    """章节摘要结构。"""

    ref: str
    title: str | None
    level: int
    parent_ref: str | None
    preview: str
    page_range: NotRequired[list[int] | None]
    internal_ref: NotRequired[str | None]


class TableSummary(TypedDict):
    """表格摘要结构。"""

    table_ref: str
    caption: str | None
    context_before: str
    row_count: int
    col_count: int
    table_type: str
    headers: list[str] | None
    section_ref: str | None
    page_no: NotRequired[int | None]
    internal_ref: NotRequired[str | None]
    is_financial: NotRequired[bool]


class SectionContent(TypedDict):
    """章节内容结构。"""

    ref: str
    title: str | None
    content: str
    tables: list[str]
    word_count: int
    contains_full_text: bool
    children: NotRequired[list[SectionSummary]]
    page_range: NotRequired[list[int] | None]
    internal_ref: NotRequired[str | None]


class TableContent(TypedDict):
    """表格内容结构。"""

    table_ref: str
    caption: str | None
    data_format: str
    data: Sequence[Mapping[str, object]] | str
    columns: list[str] | None
    row_count: int
    col_count: int
    section_ref: str | None
    table_type: str
    page_no: NotRequired[int | None]
    internal_ref: NotRequired[str | None]
    is_financial: NotRequired[bool]


class SearchEvidence(TypedDict):
    """搜索命中证据结构。"""

    matched_text: str
    context: str


class SearchHit(TypedDict, total=False):
    """搜索命中结构。"""

    section_ref: str
    section_title: str | None
    snippet: str
    page_no: int
    evidence: SearchEvidence
    _token_fallback: bool


class PageContentResult(TypedDict):
    """页面内容结构。"""

    page_no: int
    sections: list[SectionSummary]
    tables: list[TableSummary]
    text_preview: str
    has_content: bool
    total_items: int
    supported: bool


def build_section_summary(
    *,
    ref: str,
    title: str | None,
    level: int,
    parent_ref: str | None,
    preview: str,
    page_range: list[int] | None = None,
    internal_ref: str | None = None,
) -> SectionSummary:
    """构建章节摘要。"""

    result: SectionSummary = {
        "ref": ref,
        "title": title,
        "level": level,
        "parent_ref": parent_ref,
        "preview": preview,
    }
    if page_range is not None:
        result["page_range"] = page_range
    if internal_ref is not None:
        result["internal_ref"] = internal_ref
    return result


def build_table_summary(
    *,
    table_ref: str,
    caption: str | None,
    context_before: str,
    row_count: int,
    col_count: int,
    table_type: str,
    headers: list[str] | None,
    section_ref: str | None,
    page_no: int | None = None,
    internal_ref: str | None = None,
    is_financial: bool | None = None,
) -> TableSummary:
    """构建表格摘要。"""

    result: TableSummary = {
        "table_ref": table_ref,
        "caption": caption,
        "context_before": context_before,
        "row_count": row_count,
        "col_count": col_count,
        "table_type": table_type,
        "headers": headers,
        "section_ref": section_ref,
    }
    if page_no is not None:
        result["page_no"] = page_no
    if internal_ref is not None:
        result["internal_ref"] = internal_ref
    if is_financial is not None:
        result["is_financial"] = is_financial
    return result


def build_section_content(
    *,
    ref: str,
    title: str | None,
    content: str,
    tables: list[str],
    word_count: int,
    contains_full_text: bool,
    page_range: list[int] | None = None,
    internal_ref: str | None = None,
) -> SectionContent:
    """构建章节内容。"""

    result: SectionContent = {
        "ref": ref,
        "title": title,
        "content": content,
        "tables": tables,
        "word_count": word_count,
        "contains_full_text": contains_full_text,
    }
    if page_range is not None:
        result["page_range"] = page_range
    if internal_ref is not None:
        result["internal_ref"] = internal_ref
    return result


def build_table_content(
    *,
    table_ref: str,
    caption: Optional[str],
    data_format: str,
    data: Sequence[Mapping[str, object]] | str,
    columns: Optional[list[str]],
    row_count: int,
    col_count: int,
    section_ref: Optional[str],
    table_type: str,
    **extra: Any,
) -> TableContent:
    """构建 TableContent 字典的工厂函数。

    将 read_table() 在三个 processor 中重复的 9 个公共字段集中定义，
    各 processor 仅需传入渲染结果和差异字段（page_no、internal_ref、
    is_financial 等）通过 ``**extra``。

    Args:
        table_ref: 表格引用标识。
        caption: 表格标题。
        data_format: 数据格式（"records" / "markdown"）。
        data: 渲染后的表格数据。
        columns: 列名列表。
        row_count: 行数。
        col_count: 列数。
        section_ref: 所属章节引用。
        table_type: 表格类型。
        **extra: 处理器特有的额外字段（page_no、internal_ref、is_financial 等）。

    Returns:
        TableContent 字典。
    """
    result: TableContent = {
        "table_ref": table_ref,
        "caption": caption,
        "data_format": data_format,
        "data": data,
        "columns": columns,
        "row_count": row_count,
        "col_count": col_count,
        "section_ref": section_ref,
        "table_type": table_type,
    }
    page_no = extra.get("page_no")
    if isinstance(page_no, int):
        result["page_no"] = page_no
    internal_ref = extra.get("internal_ref")
    if isinstance(internal_ref, str):
        result["internal_ref"] = internal_ref
    is_financial = extra.get("is_financial")
    if isinstance(is_financial, bool):
        result["is_financial"] = is_financial
    return result


def build_search_hit(
    *,
    section_ref: str,
    section_title: str | None,
    snippet: str | None = None,
    page_no: int | None = None,
    evidence: SearchEvidence | None = None,
    token_fallback: bool = False,
) -> SearchHit:
    """构建搜索命中。"""

    result: SearchHit = {
        "section_ref": section_ref,
        "section_title": section_title,
    }
    if snippet is not None:
        result["snippet"] = snippet
    if page_no is not None:
        result["page_no"] = page_no
    if evidence is not None:
        result["evidence"] = evidence
    if token_fallback:
        result["_token_fallback"] = True
    return result


def build_page_content_result(
    *,
    page_no: int,
    sections: list[SectionSummary],
    tables: list[TableSummary],
    text_preview: str,
    has_content: bool,
    total_items: int,
    supported: bool,
) -> PageContentResult:
    """构建页面内容结果。"""

    return {
        "page_no": page_no,
        "sections": sections,
        "tables": tables,
        "text_preview": text_preview,
        "has_content": has_content,
        "total_items": total_items,
        "supported": supported,
    }





class DocumentProcessor(Protocol):
    """文档处理器协议。"""

    @classmethod
    def get_parser_version(cls) -> str:
        """返回处理器 parser version。"""

        ...

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
            form_type: 文档类型。
            media_type: 媒体类型。

        Returns:
            None。

        Raises:
            ValueError: 当参数非法时抛出。
        """

    @classmethod
    def supports(
        cls,
        source: Source,
        *,
        form_type: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> bool:
        """判断是否支持处理该文件。

        Args:
            source: 文档来源抽象。
            form_type: 文档类型。
            media_type: 媒体类型。

        Returns:
            是否支持。

        Raises:
            OSError: 访问文件失败时可能抛出。
        """

        ...

    def list_sections(
        self,
    ) -> list[SectionSummary]:
        """读取章节列表。

        Args:
            无。

        Returns:
            章节摘要列表。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        ...

    def list_tables(
        self,
    ) -> list[TableSummary]:
        """读取表格列表。

        Args:
            无。

        Returns:
            表格摘要列表。

        Raises:
            RuntimeError: 读取失败时抛出。
        """

        ...

    def read_section(self, ref: str) -> SectionContent:
        """按 ref 读取章节内容。

        Args:
            ref: 章节引用。

        Returns:
            章节内容。

        Raises:
            KeyError: 找不到章节时抛出。
        """

        ...

    def read_table(self, table_ref: str) -> TableContent:
        """按 ref 读取表格内容。

        Args:
            table_ref: 表格引用。

        Returns:
            表格内容。

        Raises:
            KeyError: 找不到表格时抛出。
        """

        ...

    def get_section_title(self, ref: str) -> Optional[str]:
        """根据 section ref 获取章节标题。

        O(1) 查询，适用于 service 层为表格等附加 section 上下文。
        比 read_section() 更轻量——不需要构建完整 SectionContent。

        Args:
            ref: 章节引用。

        Returns:
            章节标题字符串；ref 不存在时返回 None。
        """

        ...

    def search(
        self,
        query: str,
        within_ref: Optional[str] = None,
    ) -> list[SearchHit]:
        """在文档中搜索。

        Args:
            query: 搜索词。
            within_ref: 可选章节范围。

        Returns:
            命中列表。

        Raises:
            RuntimeError: 搜索失败时抛出。
        """

        ...

    def get_full_text(self) -> str:
        """获取文档的完整纯文本内容（包含表格内文本）。

        返回文档全文的纯文本形式，保留表格内所有文本内容，
        不做占位符替换。适用于需要全文分析的场景（如章节切分
        marker 检测）。

        Args:
            无。

        Returns:
            文档完整纯文本字符串。

        Raises:
            RuntimeError: 提取失败时抛出。
        """

        ...

    def get_full_text_with_table_markers(self) -> str:
        """获取文档全文，非 layout 表格用 ``[[t_XXXX]]`` 占位符替代。

        与 ``get_full_text()`` 不同，本方法将每个非 layout 表格
        替换为对应的 ``[[t_XXXX]]`` 占位符后再提取文本。占位符
        编号与 ``list_tables()`` 返回的 ``table_ref`` 一致。

        用途：虚拟章节处理器在全文切分后，通过解析占位符确定每个表格
        落入哪个虚拟章节，从而建立 table→virtual_section 映射。

        不支持 DOM 级表格标记注入的处理器应返回空字符串，表示
        不具备此能力，上层会安全降级。

        Args:
            无。

        Returns:
            带 ``[[t_XXXX]]`` 占位符的文档全文字符串；
            不支持时返回空字符串。

        Raises:
            RuntimeError: 提取失败时抛出。
        """

        ...


class PageAwareProcessor(Protocol):
    """分页能力协议。"""

    def get_page_content(self, page_no: int) -> PageContentResult:
        """读取页面内容。

        Args:
            page_no: 页码，从 1 开始。

        Returns:
            页面内容结果。

        Raises:
            ValueError: 页码非法时抛出。
        """

        ...



