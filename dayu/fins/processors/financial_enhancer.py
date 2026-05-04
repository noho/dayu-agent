"""fins 表格金融语义增强工具。

本模块负责在业务域层为通用处理器产出的表格补充金融语义：
- 统一关键词库；
- 统一判定规则；
- 统一表格重标注流程。
- 金融语义额外字段提取（``extra_financial_table_fields``）。
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any, Optional, Protocol

from dayu.engine.processors.text_utils import (
    normalize_optional_string as _normalize_optional_string,
    normalize_whitespace as _normalize_whitespace,
)


def extra_financial_table_fields(table: Any) -> dict[str, Any]:
    """提取金融语义额外字段（跨 FinsBSProcessor / FinsDoclingProcessor / FinsMarkdownProcessor 共享）。

    ``relabel_tables`` 会通过 ``setattr`` 为每个表格对象动态添加
    ``is_financial`` 属性，此函数将其填充到输出字典中。

    Args:
        table: 内部表格对象。

    Returns:
        包含 ``is_financial`` 的字段字典。
    """
    return {"is_financial": getattr(table, "is_financial", False)}

_FINANCIAL_KEYWORDS = (
    "balance sheet",
    "income statement",
    "cash flow",
    "statement of operations",
    "statement of cash flows",
    "financial position",
    "financial results",
    "total assets",
    "total liabilities",
    "net income",
    "net earnings",
    "revenue",
    "revenues",
    "earnings",
    "profit",
    "loss",
    "资产负债表",
    "利润表",
    "现金流量表",
    "現金流量表",
    "财务状况表",
    "財務狀況表",
    "財務狀況報表",
    "主要财务数据",
    "主要財務數據",
    "主要財務資料",
    "综合收益",
    "綜合收益",
    "营业收入",
    "營業收入",
    "净利润",
    "淨利潤",
)

_FINANCIAL_STATEMENT_EVIDENCE_GROUPS = (
    (
        "主要财务数据",
        (
            "营业收入",
            "營業收入",
            "收入及其他收益",
            "主要業務收入",
            "经营收入",
            "經營收入",
            "收入",
            "收益",
            "EBITDA",
            "股東應佔溢利",
            "股东应占溢利",
            "股東應佔盈利",
            "普通股股東應佔",
            "普通股股东应占",
            "汽車銷售收入",
            "汽车销售收入",
            "毛利率",
            "淨收益",
            "净收益",
            "淨虧損",
            "净亏损",
            "期內利潤",
            "期内利润",
            "經調整利潤淨額",
            "经调整利润净额",
            "經調整 EBITDA",
            "经调整 EBITDA",
            "基本每股盈利",
            "資本開支",
            "除税前利润",
            "除稅前利潤",
            "除税后利润",
            "除稅後利潤",
            "净利润",
            "淨利潤",
            "期内盈利",
            "期內盈利",
            "本公司权益持有人应占盈利",
            "本公司權益持有人應佔盈利",
            "归属于上市公司股东",
            "歸屬於本行股東",
            "歸屬於",
            "每股收益",
            "每股盈利",
            "基本每股收益",
            "基本和稀释每股收益",
            "资产总额",
            "資產總額",
            "总资产",
            "總資產",
            "经营活动产生的现金流量净额",
            "经营活动的现金流量净额",
            "經營活動產生的現金流量淨額",
            "經營活動產生的現金流 量淨額",
            "现金流量净额",
            "現金流 量淨額",
            "加权平均净资产收益率",
            "年化加權平均淨資產收益率",
        ),
        4,
    ),
    (
        "资产负债表",
        (
            "流动资产",
            "非流动资产",
            "资产总计",
            "资产总额",
            "资产总值",
            "流动负债",
            "非流动负债",
            "负债合计",
            "权益及负债",
            "所有者权益",
            "股东权益",
            "归属于母公司股东权益",
            "归属于上市公司股东的所有者权益",
            "负债和股东权益",
            "负债及股东权益",
            "现金及存放中央银行款项",
            "发放贷款和垫款",
            "客户贷款及垫款",
            "客户贷款和垫款",
            "吸收存款",
            "客户存款",
            "流動資產",
            "非流動資產",
            "資產總計",
            "資產總額",
            "資產總值",
            "流動負債",
            "非流動負債",
            "負債合計",
            "權益及負債",
            "所有者權益",
            "股東權益",
            "本公司權益持有人應佔權益",
            "負債總額",
            "總資產",
            "總權益",
            "權益總額",
            "權益及負債總額",
            "基金單位持有人應佔資產淨值",
            "非控制性權益",
            "負債和股東權益",
            "負債及股東權益",
            "現金及存放中央銀行款項",
            "發放貸款和墊款",
            "客戶貸款及墊款",
            "客戶貸款和墊款",
            "吸收存款",
            "客戶存款",
        ),
        4,
    ),
    (
        "利润表",
        (
            "营业收入",
            "营业利润",
            "利润总额",
            "净利润",
            "净亏损",
            "综合收益总额",
            "总收入",
            "总销售成本",
            "销售成本",
            "毛利",
            "毛亏损",
            "经营开支",
            "经营亏损",
            "销售费用",
            "行政费用",
            "除所得税",
            "税后利润",
            "利息收入",
            "利息支出",
            "净利息收入",
            "手续费及佣金收入",
            "營業收入",
            "營業利潤",
            "利潤總額",
            "淨利潤",
            "淨虧損",
            "綜合收益總額",
            "總收入",
            "總銷售成本",
            "銷售成本",
            "毛利",
            "毛虧損",
            "毛 （虧損） 溢利",
            "經營開支",
            "經營開⽀",
            "經營虧損",
            "銷售費用",
            "銷售費⽤",
            "分銷及銷售費用",
            "行政費用",
            "行政費⽤",
            "除所得稅",
            "稅後利潤",
            "利息收入",
            "利息支出",
            "利息⽀出",
            "淨利息收入",
            "手續費及佣金收入",
            "綜合虧損總額",
            "保險收益",
            "保險服務開支",
            "保險服務開⽀",
            "保險服務業績",
            "投資回報",
            "投資業績淨額",
            "稅後營運溢利",
            "純利",
        ),
        3,
    ),
    (
        "现金流量表",
        (
            "经营活动产生的现金流量",
            "经营活动现金流量",
            "经营活动现金流入",
            "经营活动现金流出",
            "投资活动产生的现金流量",
            "投资活动现金流量",
            "筹资活动产生的现金流量",
            "筹资活动现金流量",
            "现金及现金等价物",
            "經營活動產生的現金流量",
            "經營活動產生╱ （使用） 的現金流量",
            "經營活動產生╱（使用）的現金流量",
            "經營活動現金流量",
            "經營活動所得現金流量",
            "經營活動所得現金流量淨額",
            "經營活動產生的現金流量淨額",
            "經營業務之現金流量",
            "營運活動所得之現金",
            "經營活動現金流入",
            "經營活動現金流出",
            "投資活動產生的現金流量",
            "投資活動 （使用） ╱產生的現金流量",
            "投資活動（使用）╱產生的現金流量",
            "投資活動 （使用） ╱產生的現金流量淨額",
            "投資活動（使用）╱產生的現金流量淨額",
            "投資活動現金流量",
            "投資活動所用現金流量",
            "投資業務之現金流量",
            "籌資活動產生的現金流量",
            "籌資活動現金流量",
            "融資活動產生的現金流量",
            "融資活動產生的現金流量淨額",
            "融資活動現金流量",
            "融資活動所用現金流量",
            "現金及現金等價物增加",
            "現金及現金等價物減少",
            "現金及現金等價物",
            "主要業務活動之現金流量",
            "主要業務活動之現金流入淨額",
            "業務活動之現金流入淨額",
            "營業活動產生之現金淨額",
            "營業活動產生的現金淨額",
            "營業活動產生之現金净额",
            "投資活動之現金流量",
            "投資活動產生之現金淨額",
            "投資活動產生的現金淨額",
            "投資活動產生之現金净额",
            "財務活動之現金流量",
            "融資活動產生之現金淨額",
            "融資活動產生的現金淨額",
            "融資活動產生之現金净额",
            "現金流動",
        ),
        2,
    ),
)

_MAX_TABLE_BODY_EVIDENCE_CHARS = 6000


class _DoclingDocumentProtocol(Protocol):
    """Docling 文档导出上下文协议。

    fins 层只把文档对象原样传回 Docling 表格导出方法，不读取任何
    Docling 内部字段；因此该协议刻意保持不声明成员。
    """


class _DoclingTableItemProtocol(Protocol):
    """Docling 表格对象协议。

    表格导出能力由第三方 Docling 对象提供，fins 层通过运行时方法探测
    安全调用，类型上只表达“这是 Docling 表格对象”的边界。
    """


def is_financial_table(
    caption: Optional[str],
    headers: Optional[list[str]],
    context_before: str,
) -> bool:
    """判断表格是否为财务表。

    Args:
        caption: 表格标题。
        headers: 表头列表。
        context_before: 表格前文。

    Returns:
        命中金融关键词时返回 `True`，否则返回 `False`。

    Raises:
        RuntimeError: 判定失败时抛出。
    """

    parts = [str(caption or ""), str(context_before or "")]
    if headers:
        parts.extend(str(item or "") for item in headers)
    normalized_text = _normalize_whitespace(" ".join(parts)).lower()
    if not normalized_text:
        return False
    return any(keyword in normalized_text for keyword in _FINANCIAL_KEYWORDS)


def relabel_tables(tables: Iterable[Any], *, docling_document: _DoclingDocumentProtocol | None = None) -> None:
    """批量重标注表格金融语义。

    Args:
        tables: 表格对象可迭代序列。
        docling_document: 可选 Docling 文档对象；传入时可从表体补充财报语义 caption。

    Returns:
        无。

    Raises:
        RuntimeError: 重标注失败时抛出。
    """

    for table in tables:
        relabel_single_table(table, docling_document=docling_document)


def relabel_single_table(table: Any, *, docling_document: _DoclingDocumentProtocol | None = None) -> None:
    """重标注单个表格的金融语义。

    Args:
        table: 表格对象（需具备 `caption/headers/context_before/is_financial/table_type` 字段）。
        docling_document: 可选 Docling 文档对象；传入时可从表体补充财报语义 caption。

    Returns:
        无。

    Raises:
        RuntimeError: 重标注失败时抛出。
    """

    caption = _normalize_optional_string(getattr(table, "caption", None))
    headers_value = getattr(table, "headers", None)
    headers = headers_value if isinstance(headers_value, list) else None
    context_before = str(getattr(table, "context_before", "") or "")
    semantic_caption = _derive_financial_statement_caption(
        table=table,
        docling_document=docling_document,
    )
    detection_headers = _merge_headers(headers, [semantic_caption] if semantic_caption else [])
    detection_caption = caption or semantic_caption
    is_financial = is_financial_table(
        caption=detection_caption,
        headers=detection_headers,
        context_before=context_before,
    )

    setattr(table, "is_financial", is_financial)
    if is_financial:
        if caption is None and semantic_caption is not None:
            setattr(table, "caption", semantic_caption)
        setattr(table, "table_type", "financial")
        return

    raw_type = str(getattr(table, "table_type", "") or "").strip().lower()
    if raw_type not in {"data", "layout"}:
        setattr(table, "table_type", "data")


def _derive_financial_statement_caption(
    *,
    table: Any,
    docling_document: _DoclingDocumentProtocol | None,
) -> str | None:
    """从表体推断三大财报表语义 caption。

    Args:
        table: 内部表格对象。
        docling_document: 可选 Docling 文档对象。

    Returns:
        推断出的语义 caption；无法可靠推断时返回 None。

    Raises:
        无。
    """

    table_item = getattr(table, "table_item", None)
    if table_item is None:
        return None
    evidence_text = _extract_table_body_text(
        table_item=table_item,
        docling_document=docling_document,
    )
    labels = _match_financial_statement_labels(evidence_text)
    if not labels:
        return None
    return " / ".join(labels)


def _extract_table_body_text(
    *,
    table_item: _DoclingTableItemProtocol,
    docling_document: _DoclingDocumentProtocol | None,
) -> str:
    """提取表体文本作为 fins 财报语义增强证据。

    Args:
        table_item: Docling 表格对象。
        docling_document: 可选 Docling 文档对象。

    Returns:
        表格正文文本，最多保留 `_MAX_TABLE_BODY_EVIDENCE_CHARS` 个字符。

    Raises:
        无。
    """

    markdown = _export_table_markdown(table_item=table_item, docling_document=docling_document)
    if markdown:
        return markdown[:_MAX_TABLE_BODY_EVIDENCE_CHARS]
    dataframe_text = _export_table_dataframe_text(table_item=table_item, docling_document=docling_document)
    return dataframe_text[:_MAX_TABLE_BODY_EVIDENCE_CHARS]


def _export_table_markdown(
    *,
    table_item: _DoclingTableItemProtocol,
    docling_document: _DoclingDocumentProtocol | None,
) -> str:
    """安全导出 Docling 表格 markdown。

    Args:
        table_item: Docling 表格对象。
        docling_document: 可选 Docling 文档对象。

    Returns:
        markdown 文本；导出失败时返回空字符串。

    Raises:
        无。
    """

    # Docling TableItem 是第三方对象，导出能力只能通过鸭子类型探测。
    exporter = getattr(table_item, "export_to_markdown", None)
    if not callable(exporter):
        return ""
    try:
        if docling_document is not None:
            payload = exporter(doc=docling_document)
        else:
            payload = exporter()
    except TypeError:
        try:
            payload = exporter()
        except Exception:
            return ""
    except Exception:
        return ""
    return payload if isinstance(payload, str) else ""


def _export_table_dataframe_text(
    *,
    table_item: _DoclingTableItemProtocol,
    docling_document: _DoclingDocumentProtocol | None,
) -> str:
    """安全导出 Docling 表格 dataframe 文本。

    Args:
        table_item: Docling 表格对象。
        docling_document: 可选 Docling 文档对象。

    Returns:
        dataframe 文本；导出失败时返回空字符串。

    Raises:
        无。
    """

    # Docling TableItem 是第三方对象，导出能力只能通过鸭子类型探测。
    exporter = getattr(table_item, "export_to_dataframe", None)
    if not callable(exporter):
        return ""
    try:
        if docling_document is not None:
            dataframe = exporter(doc=docling_document)
        else:
            dataframe = exporter()
    except TypeError:
        try:
            dataframe = exporter()
        except Exception:
            return ""
    except Exception:
        return ""
    to_dict = getattr(dataframe, "to_dict", None)
    if not callable(to_dict):
        return ""
    try:
        records = to_dict(orient="records")
    except Exception:
        return ""
    if not isinstance(records, list):
        return ""
    parts: list[str] = []
    for row in records[:80]:
        if not isinstance(row, Mapping):
            continue
        for key, value in row.items():
            parts.append(str(key))
            parts.append(str(value))
    return " ".join(parts)


def _match_financial_statement_labels(text: str) -> list[str]:
    """根据表体证据匹配三大财报表标签。

    Args:
        text: 表体文本。

    Returns:
        命中的财报表标签列表。

    Raises:
        无。
    """

    normalized_text = _normalize_for_financial_match(text)
    if not normalized_text:
        return []
    labels: list[str] = []
    for label, keywords, minimum_hits in _FINANCIAL_STATEMENT_EVIDENCE_GROUPS:
        hit_count = sum(1 for keyword in keywords if _normalize_for_financial_match(keyword) in normalized_text)
        if hit_count >= minimum_hits:
            labels.append(label)
    return labels


def _merge_headers(headers: Optional[list[str]], inferred_headers: list[str]) -> Optional[list[str]]:
    """合并原始表头与推断语义标签。

    Args:
        headers: 原始表头。
        inferred_headers: 推断语义标签。

    Returns:
        合并后的表头；两者都为空时返回 None。

    Raises:
        无。
    """

    result: list[str] = []
    for item in headers or []:
        normalized = _normalize_optional_string(str(item or ""))
        if normalized and normalized not in result:
            result.append(normalized)
    for item in inferred_headers:
        normalized = _normalize_optional_string(item)
        if normalized and normalized not in result:
            result.append(normalized)
    return result or None


def _normalize_for_financial_match(text: str) -> str:
    """归一化用于金融语义匹配的文本。

    Args:
        text: 原始文本。

    Returns:
        NFKC 与空白归一化后的大小写折叠文本。

    Raises:
        无。
    """

    return _normalize_whitespace(unicodedata.normalize("NFKC", str(text or ""))).casefold()


class FinsProcessorMixin:
    """为 fins 处理器提供金融语义扩展的通用 Mixin。

    三个 fins 处理器子类（FinsBSProcessor / FinsDoclingProcessor /
    FinsMarkdownProcessor）均需覆盖 ``_extra_table_fields``，且实现完全
    相同。将此方法提升到本 Mixin，避免三处重复定义。

    MRO 约定：本 Mixin 须置于具体基类（BSProcessor / DoclingProcessor /
    MarkdownProcessor）之前，即：class FinsXxxProcessor(FinsProcessorMixin, XxxProcessor)。
    """

    def _extra_table_fields(self, table: Any) -> dict[str, Any]:
        """注入金融语义字段，委托 ``extra_financial_table_fields``。

        Args:
            table: 内部表格对象。

        Returns:
            包含 ``is_financial`` 的字段字典。
        """
        return extra_financial_table_fields(table)
