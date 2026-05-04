"""FinancialEnhancer 覆盖率补充测试（提升到 90%+）。

本测试文件补充财务表格语义增强模块的边界情况、异常处理和特殊场景，
覆盖所有关键字判定、表格重标注流程和文本规范化。
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock
from typing import Any, Optional, cast

import pytest

from dayu.fins.processors.financial_enhancer import (
    FinsProcessorMixin,
    extra_financial_table_fields,
    is_financial_table,
    relabel_single_table,
    relabel_tables,
    _normalize_whitespace,
    _normalize_optional_string,
    _FINANCIAL_KEYWORDS,
)


@pytest.mark.unit
class TestNormalizeWhitespace:
    """_normalize_whitespace 函数单元测试。"""

    def test_normalize_whitespace_multiple_spaces(self) -> None:
        """验证多个空格被合并为一个。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_whitespace("hello    world")
        assert result == "hello world"

    def test_normalize_whitespace_tabs_and_newlines(self) -> None:
        """验证制表符和换行符被转换为空格。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_whitespace("hello\t\tworld\ntest\n")
        assert result == "hello world test"

    def test_normalize_whitespace_leading_trailing(self) -> None:
        """验证去除首尾空白。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_whitespace("   hello world   ")
        assert result == "hello world"

    def test_normalize_whitespace_empty_string(self) -> None:
        """验证空字符串返回空字符串。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_whitespace("")
        assert result == ""

    def test_normalize_whitespace_only_spaces(self) -> None:
        """验证仅空白字符返回空字符串。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_whitespace("    \t\n  ")
        assert result == ""

    def test_normalize_whitespace_mixed_unicode(self) -> None:
        """验证处理 Unicode 字符。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_whitespace("资产负债表\n\n利润表")
        assert result == "资产负债表 利润表"


@pytest.mark.unit
class TestNormalizeOptionalString:
    """_normalize_optional_string 函数单元测试。"""

    def test_normalize_optional_string_valid(self) -> None:
        """验证有效字符串正常处理。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_optional_string("  hello  world  ")
        assert result == "hello world"

    def test_normalize_optional_string_none(self) -> None:
        """验证 None 返回 None。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_optional_string(None)
        assert result is None

    def test_normalize_optional_string_empty(self) -> None:
        """验证空字符串返回 None。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_optional_string("")
        assert result is None

    def test_normalize_optional_string_whitespace_only(self) -> None:
        """验证仅空白返回 None。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_optional_string("   \t\n  ")
        assert result is None

    def test_normalize_optional_string_numeric_input(self) -> None:
        """验证数字转换为字符串。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = _normalize_optional_string(123)
        assert result == "123"

    def test_normalize_optional_string_non_string_object(self) -> None:
        """验证对象转换为字符串。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        obj = Mock()
        obj.__str__ = Mock(return_value="mock object")
        result = _normalize_optional_string(obj)
        assert result == "mock object"


@pytest.mark.unit
class TestIsFinancialTable:
    """is_financial_table 函数单元测试。"""

    def test_is_financial_table_with_caption_balance_sheet(self) -> None:
        """验证通过标题识别资产负债表。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption="Balance Sheet December 31, 2023",
            headers=None,
            context_before=""
        )
        assert result is True

    def test_is_financial_table_with_caption_income_statement(self) -> None:
        """验证通过标题识别利润表。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption="Statement of Operations",
            headers=None,
            context_before=""
        )
        assert result is True

    def test_is_financial_table_with_chinese_keywords(self) -> None:
        """验证中文关键词识别。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption="资产负债表",
            headers=None,
            context_before=""
        )
        assert result is True

    def test_is_financial_table_with_context_revenue(self) -> None:
        """验证通过上下文识别收入相关表。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption=None,
            headers=None,
            context_before="Following shows the annual revenues"
        )
        assert result is True

    def test_is_financial_table_with_headers_net_income(self) -> None:
        """验证通过表头识别净利润信息。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption=None,
            headers=["Year", "Net Income", "Total Assets"],
            context_before=""
        )
        assert result is True

    def test_is_financial_table_combined_context(self) -> None:
        """验证组合多个信息来源的识别。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption="Financial Summary",
            headers=["Profit", "Assets"],
            context_before="As shown in the financial reports"
        )
        assert result is True

    def test_is_financial_table_no_keywords(self) -> None:
        """验证不含关键词返回 False。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption="Product Features",
            headers=["Feature One", "Feature Two"],
            context_before="This table shows product details"
        )
        assert result is False

    def test_is_financial_table_empty_inputs(self) -> None:
        """验证全为空值返回 False。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption=None,
            headers=None,
            context_before=""
        )
        assert result is False

    def test_is_financial_table_cash_flow(self) -> None:
        """验证现金流表识别（英文）。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption="Statement of Cash Flows",
            headers=None,
            context_before=""
        )
        assert result is True

    def test_is_financial_table_chinese_cash_flow(self) -> None:
        """验证现金流表识别（中文）。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption="现金流量表",
            headers=None,
            context_before=""
        )
        assert result is True

    def test_is_financial_table_case_insensitive(self) -> None:
        """验证大小写不敏感。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption="BALANCE SHEET",
            headers=None,
            context_before=""
        )
        assert result is True

    def test_is_financial_table_with_empty_headers_list(self) -> None:
        """验证空头列表处理。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption=None,
            headers=[],
            context_before="earnings are crucial"
        )
        assert result is True

    def test_is_financial_table_header_with_none_values(self) -> None:
        """验证包含 None 值的表头处理。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        result = is_financial_table(
            caption=None,
            headers=cast(Any, ["Name", None, "Revenue"]),
            context_before=""
        )
        assert result is True


@pytest.mark.unit
class TestRelabelSingleTable:
    """relabel_single_table 函数单元测试。"""

    def test_relabel_single_table_financial_keyword_match(self) -> None:
        """验证财务表标注为财务类型。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        table = MagicMock()
        table.caption = "Balance Sheet"
        table.headers = None
        table.context_before = ""
        
        relabel_single_table(table)
        
        assert table.is_financial is True
        assert table.table_type == "financial"

    def test_relabel_single_table_non_financial(self) -> None:
        """验证非财务表标注为数据类型。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        table = MagicMock()
        table.caption = "Product List"
        table.headers = ["Name", "Category"]
        table.context_before = "See our products"
        
        relabel_single_table(table)
        
        assert table.is_financial is False
        assert table.table_type == "data"

    def test_relabel_single_table_with_existing_table_type(self) -> None:
        """验证保留有效的现有表格类型。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        table = MagicMock()
        table.caption = "Product List"
        table.headers = None
        table.context_before = ""
        table.table_type = "layout"
        
        relabel_single_table(table)
        
        assert table.is_financial is False
        assert table.table_type == "layout"

    def test_relabel_single_table_invalid_table_type_normalization(self) -> None:
        """验证无效表格类型被标准化为 data。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        table = MagicMock()
        table.caption = "Generic Table"
        table.headers = None
        table.context_before = ""
        table.table_type = "INVALID_TYPE"
        
        relabel_single_table(table)
        
        assert table.is_financial is False
        assert table.table_type == "data"

    def test_relabel_single_table_missing_caption(self) -> None:
        """验证处理缺失标题属性。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        table = MagicMock()
        del table.caption
        table.headers = None
        table.context_before = "revenues increased"
        
        relabel_single_table(table)
        
        assert table.is_financial is True
        assert table.table_type == "financial"

    def test_relabel_single_table_missing_headers(self) -> None:
        """验证处理缺失表头属性。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        table = MagicMock()
        table.caption = None
        del table.headers
        table.context_before = "balance sheet"
        
        relabel_single_table(table)
        
        assert table.is_financial is True


@pytest.mark.unit
class TestRelabelTables:
    """relabel_tables 函数单元测试。"""

    def test_relabel_tables_multiple_tables(self) -> None:
        """验证批量重标注多个表格。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        tables = [
            MagicMock(caption="Balance Sheet", headers=None, context_before=""),
            MagicMock(caption="Product List", headers=["Name"], context_before=""),
            MagicMock(caption="Cash Flow", headers=None, context_before=""),
        ]
        
        relabel_tables(tables)
        
        assert tables[0].is_financial is True
        assert tables[1].is_financial is False
        assert tables[2].is_financial is True

    def test_relabel_tables_empty_iterable(self) -> None:
        """验证空迭代器不抛出异常。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        # 应该不拋出异常
        relabel_tables([])

    def test_relabel_tables_generator(self) -> None:
        """验证支持生成器迭代。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        def table_generator():
            yield MagicMock(caption="Balance Sheet", headers=None, context_before="")
            yield MagicMock(caption="Regular Table", headers=None, context_before="")
        
        tables = list(table_generator())
        relabel_tables(tables)
        
        assert tables[0].is_financial is True
        assert tables[1].is_financial is False

    def test_relabel_tables_all_financial(self) -> None:
        """验证处理全是财务表的情况。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        tables = [
            MagicMock(caption="Balance Sheet", headers=None, context_before=""),
            MagicMock(caption="利润表", headers=None, context_before=""),
            MagicMock(caption=None, headers=None, context_before="net income"),
        ]
        
        relabel_tables(tables)
        
        assert all(t.is_financial for t in tables)

    def test_relabel_tables_with_special_characters_in_caption(self) -> None:
        """验证处理特殊字符的标题。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        tables = [
            MagicMock(caption="Balance Sheet (FY 2023)", headers=None, context_before=""),
            MagicMock(
                caption="资产负债表（2023年12月31日）",
                headers=None,
                context_before=""
            ),
        ]
        
        relabel_tables(tables)
        
        assert tables[0].is_financial is True
        assert tables[1].is_financial is True

    def test_relabel_tables_derives_caption_from_docling_cash_flow_body(self) -> None:
        """验证 Docling 表体中的现金流量表语义会提升为表格 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 2021年度 | 人民币千元 |\n"
            "| --- | --- |\n"
            "| 一、经营活动产生的现金流量 | |\n"
            "| 销售商品、提供劳务收到的现金 | 120976285 |\n"
            "| 二、投资活动产生的现金流量 | |\n"
            "| 三、筹资活动产生的现金流量 | |"
        )
        table = MagicMock(caption=None, headers=["一、", "二、"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "现金流量表"

    def test_relabel_tables_derives_caption_from_hk_cash_flow_body(self) -> None:
        """验证港股现金流量表体表达会提升为表格 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 截至年度 | 人民幣百萬元 |\n"
            "| --- | --- |\n"
            "| 經營活動現金流量 | |\n"
            "| 經營活動所得現金淨額 | 29787 |\n"
            "| 投資活動現金流量 | |\n"
            "| 融資活動現金流量 | |"
        )
        table = MagicMock(caption=None, headers=["截至年度"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "现金流量表"

    def test_relabel_tables_derives_caption_from_hk_cash_flow_net_body(self) -> None:
        """验证港股现金流量淨額表体表达会提升为现金流量表 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 本年度 |\n"
            "| --- | --- |\n"
            "| 經營活動所得現金流量淨額 | 106676 |\n"
            "| 投資活動所用現金流量淨額 | (32852) |\n"
            "| 融資活動所用現金流量淨額 | (17459) |\n"
            "| 現金及現金等價物增加淨額 | 56365 |"
        )
        table = MagicMock(caption=None, headers=["項目", "本年度"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "现金流量表"

    def test_relabel_tables_derives_caption_from_hk_loss_statement_body(self) -> None:
        """验证港股亏损企业损益表体会提升为利润表 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 本年度 |\n"
            "| --- | --- |\n"
            "| 總收入 | 30676067 |\n"
            "| 總銷售成本 | (27872710) |\n"
            "| 毛 （虧損） 溢利 | 2803357 |\n"
            "| 經營開支總額 | (6911460) |\n"
            "| 經營虧損 | (4108103) |\n"
            "| 淨虧損 | (4856850) |"
        )
        table = MagicMock(caption=None, headers=["項目", "本年度"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "利润表"

    def test_relabel_tables_derives_caption_from_hk_insurance_income_body(self) -> None:
        """验证港股保险损益表体会提升为利润表 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 本年度 |\n"
            "| --- | --- |\n"
            "| 保險收益 | 165420 |\n"
            "| 保險服務開支 | (140103) |\n"
            "| 保險服務業績 | 25317 |\n"
            "| 投資回報 | 14302 |\n"
            "| 稅後營運溢利 | 6765 |\n"
            "| 純利 | 3764 |"
        )
        table = MagicMock(caption=None, headers=["項目", "本年度"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "利润表"

    def test_relabel_tables_derives_caption_from_hk_cash_flow_statement_body(self) -> None:
        """验证港股現金流動表表体会提升为现金流量表 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 本年度 |\n"
            "| --- | --- |\n"
            "| 主要業務活動之現金流量 | |\n"
            "| 主要業務活動之現金流入淨額 | 12895 |\n"
            "| 投資活動之現金流量 | |\n"
            "| 財務活動之現金流量 | |"
        )
        table = MagicMock(caption=None, headers=["項目", "本年度"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "现金流量表"

    def test_relabel_tables_derives_caption_from_hk_bank_cash_net_body(self) -> None:
        """验证港股银行現金淨額表体会提升为现金流量表 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 本年度 |\n"
            "| --- | --- |\n"
            "| 營業活動產生之現金淨額 | 27160 |\n"
            "| 購入金融投資 | (29812) |\n"
            "| 投資活動產生之現金淨額 | (2517) |\n"
            "| 融資活動產生之現金淨額 | (8420) |"
        )
        table = MagicMock(caption=None, headers=["項目", "本年度"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "现金流量表"

    def test_relabel_tables_derives_caption_from_hk_slash_cash_flow_body(self) -> None:
        """验证港股斜线格式现金流量表体会提升为现金流量表 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 本期间 |\n"
            "| --- | --- |\n"
            "| 經營活動產生╱ （使用） 的現金流量 | |\n"
            "| 經營活動產生╱ （使用） 的現金流量淨額 | 1810 |\n"
            "| 投資活動 （使用） ╱產生的現金流量 | |\n"
            "| 投資活動 （使用） ╱產生的現金流量淨額 | (380) |\n"
            "| 融資活動產生的現金流量 | |\n"
            "| 融資活動產生的現金流量淨額 | 42 |"
        )
        table = MagicMock(caption=None, headers=["項目", "本期间"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "现金流量表"

    def test_relabel_tables_derives_caption_from_quarterly_key_metrics_body(self) -> None:
        """验证季报关键财务指标表体会提升为主要财务数据 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 本季度 |\n"
            "| --- | --- |\n"
            "| 經營收入 | 174376 |\n"
            "| 淨利潤 | 95808 |\n"
            "| 歸屬於本行股東的淨利潤 | 95284 |\n"
            "| 經營活動產生的現金流 量淨額 | 1817380 |\n"
            "| 基本和稀釋每股收益 | 0.35 |\n"
            "| 資產總額 | 4200000 |"
        )
        table = MagicMock(caption=None, headers=["經營收入", "淨利潤"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "主要财务数据"

    def test_relabel_tables_derives_caption_from_bank_balance_sheet_body(self) -> None:
        """验证银行资产负债表体会提升为资产负债表 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 期末 |\n"
            "| --- | --- |\n"
            "| 資產 | |\n"
            "| 現金及存放中央銀行款項 | 103400 |\n"
            "| 客戶貸款及墊款 | 826500 |\n"
            "| 負債及股東權益 | |\n"
            "| 客戶存款 | 915300 |\n"
            "| 股東權益 | 80200 |"
        )
        table = MagicMock(caption=None, headers=["項目", "期末"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "资产负债表"

    def test_relabel_tables_derives_caption_from_reit_financial_position_body(self) -> None:
        """验证港股 REIT 财务状况表体会提升为资产负债表 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 期末 |\n"
            "| --- | --- |\n"
            "| 投資物業 | 832900 |\n"
            "| 資產總值 | 880100 |\n"
            "| 負債總額（不包括基金單位持有人應佔資產淨值） | 303200 |\n"
            "| 基金單位持有人應佔資產淨值 | 576300 |\n"
            "| 非控制性權益 | 600 |"
        )
        table = MagicMock(caption=None, headers=["項目", "期末"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "资产负债表"

    def test_relabel_tables_derives_caption_from_hk_equity_balance_sheet_body(self) -> None:
        """验证港股权益口径资产负债表体会提升为资产负债表 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 期末 |\n"
            "| --- | --- |\n"
            "| 本公司權益持有人應佔權益 | |\n"
            "| 權益總額 | 1122938 |\n"
            "| 負債總額 | 795342 |\n"
            "| 權益及負債總額 | 1918280 |\n"
            "| 總資產 | 1918280 |"
        )
        table = MagicMock(caption=None, headers=["項目", "期末"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "资产负债表"

    def test_relabel_tables_derives_caption_from_hk_market_key_metrics_body(self) -> None:
        """验证港股市场运营类关键财务数据表体会提升为主要财务数据 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 本期 |\n"
            "| --- | --- |\n"
            "| 收入及其他收益 | 13295 |\n"
            "| 主要業務收入 | 12301 |\n"
            "| EBITDA | 9800 |\n"
            "| 股東應佔溢利 | 6500 |\n"
            "| 基本每股盈利 | 4.20 |\n"
            "| 資本開支 | 500 |"
        )
        table = MagicMock(caption=None, headers=["項目", "本期"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "主要财务数据"

    def test_relabel_tables_derives_caption_from_hk_new_economy_key_metrics_body(self) -> None:
        """验证港股新经济公司财务概要表体会提升为主要财务数据 caption。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        table_item = MagicMock()
        table_item.export_to_markdown.return_value = (
            "| 項目 | 本期 |\n"
            "| --- | --- |\n"
            "| 汽車銷售收入 | 1000 |\n"
            "| 總收入 | 1200 |\n"
            "| 毛利率 | 12% |\n"
            "| 淨虧損 | (300) |\n"
            "| 普通股股東應佔淨虧損 | (310) |\n"
            "| 經調整 EBITDA | 50 |"
        )
        table = MagicMock(caption=None, headers=["項目", "本期"], context_before="")
        table.table_item = table_item

        relabel_tables([table], docling_document=object())

        assert table.is_financial is True
        assert table.table_type == "financial"
        assert table.caption == "主要财务数据 / 利润表"


@pytest.mark.unit
class TestFinancialKeywordsCompleteness:
    """验证金融关键词库的完整性。"""

    def test_financial_keywords_contains_expected_english(self) -> None:
        """验证包含主要英文关键词。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        expected = [
            "balance sheet",
            "income statement",
            "cash flow",
            "net income",
            "revenue",
            "earnings",
        ]
        for keyword in expected:
            assert keyword in _FINANCIAL_KEYWORDS

    def test_financial_keywords_contains_expected_chinese(self) -> None:
        """验证包含主要中文关键词。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        expected = [
            "资产负债表",
            "利润表",
            "现金流量表",
            "营业收入",
            "净利润",
        ]
        for keyword in expected:
            assert keyword in _FINANCIAL_KEYWORDS

    def test_financial_keywords_is_tuple(self) -> None:
        """验证关键词库是元组类型。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        assert isinstance(_FINANCIAL_KEYWORDS, tuple)

    def test_financial_keywords_not_empty(self) -> None:
        """验证关键词库非空。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        assert len(_FINANCIAL_KEYWORDS) > 0


@pytest.mark.unit
class TestFinsProcessorMixin:
    """FinsProcessorMixin 单元测试。"""

    def test_extra_table_fields_returns_is_financial_false(self) -> None:
        """验证 is_financial=False 时字段正确返回。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        class FakeTable:
            is_financial = False

        mixin = FinsProcessorMixin()
        result = mixin._extra_table_fields(FakeTable())
        assert result == {"is_financial": False}

    def test_extra_table_fields_returns_is_financial_true(self) -> None:
        """验证 is_financial=True 时字段正确返回。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        class FakeTable:
            is_financial = True

        mixin = FinsProcessorMixin()
        result = mixin._extra_table_fields(FakeTable())
        assert result == {"is_financial": True}

    def test_extra_table_fields_missing_attribute_defaults_false(self) -> None:
        """验证表格对象缺少 is_financial 属性时默认返回 False。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        class FakeTable:
            pass

        mixin = FinsProcessorMixin()
        result = mixin._extra_table_fields(FakeTable())
        assert result == {"is_financial": False}

    def test_mixin_delegates_to_extra_financial_table_fields(self) -> None:
        """验证 Mixin 方法委托 extra_financial_table_fields。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """

        class FakeTable:
            is_financial = True

        table = FakeTable()
        expected = extra_financial_table_fields(table)
        result = FinsProcessorMixin()._extra_table_fields(table)
        assert result == expected

    def test_fins_bs_processor_inherits_mixin(self) -> None:
        """验证 FinsBSProcessor 通过 MRO 继承 FinsProcessorMixin。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        from dayu.fins.processors.fins_bs_processor import FinsBSProcessor
        assert issubclass(FinsBSProcessor, FinsProcessorMixin)
        mro_names = [c.__name__ for c in FinsBSProcessor.__mro__]
        mixin_idx = mro_names.index("FinsProcessorMixin")
        bs_idx = mro_names.index("BSProcessor")
        assert mixin_idx < bs_idx, "FinsProcessorMixin 必须在 BSProcessor 之前"

    def test_fins_docling_processor_inherits_mixin(self) -> None:
        """验证 FinsDoclingProcessor 通过 MRO 继承 FinsProcessorMixin。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        from dayu.fins.processors.fins_docling_processor import FinsDoclingProcessor
        assert issubclass(FinsDoclingProcessor, FinsProcessorMixin)

    def test_fins_markdown_processor_inherits_mixin(self) -> None:
        """验证 FinsMarkdownProcessor 通过 MRO 继承 FinsProcessorMixin。

        Args:
            无。

        Returns:
            无。

        Raises:
            AssertionError: 断言失败时抛出。
        """
        from dayu.fins.processors.fins_markdown_processor import FinsMarkdownProcessor
        assert issubclass(FinsMarkdownProcessor, FinsProcessorMixin)
