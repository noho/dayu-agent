"""Web 下载表单市场分类与选项单测。"""

from __future__ import annotations

import pytest

from dayu.web.streamlit.pages.filing.download_form_profile import (
    classify_fins_download_form_market,
    fins_download_default_form_selection,
    fins_download_form_options,
)


@pytest.mark.unit
def test_classify_hk_ticker_uses_cn_hk_forms() -> None:
    """港股代码应走 A 股/港股财期表单。"""

    assert classify_fins_download_form_market("0700") == "cn_hk"
    assert classify_fins_download_form_market("00700.HK") == "cn_hk"


@pytest.mark.unit
def test_classify_cn_ticker_uses_cn_hk_forms() -> None:
    """沪深代码应走 A 股/港股财期表单。"""

    assert classify_fins_download_form_market("600519") == "cn_hk"
    assert classify_fins_download_form_market("000333") == "cn_hk"


@pytest.mark.unit
def test_classify_us_ticker_uses_sec_forms() -> None:
    """美股代码应走 SEC 表单。"""

    assert classify_fins_download_form_market("AAPL") == "sec"
    assert classify_fins_download_form_market("BRK-B") == "sec"


@pytest.mark.unit
def test_cn_hk_options_are_fiscal_periods() -> None:
    """cn_hk 市场选项为 FY/H1/Q1–Q4。"""

    opts = fins_download_form_options("cn_hk")
    assert opts == ("FY", "H1", "Q1", "Q2", "Q3", "Q4")
    assert fins_download_default_form_selection("cn_hk") == ("FY", "H1")


@pytest.mark.unit
def test_sec_options_include_ten_k() -> None:
    """sec 市场选项包含常见 SEC 表单。"""

    opts = fins_download_form_options("sec")
    assert "10-K" in opts
    assert "10-Q" in opts
    assert fins_download_default_form_selection("sec") == ("10-K", "10-Q")
