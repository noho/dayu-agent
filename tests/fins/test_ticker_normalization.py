"""ticker_normalization 真源的单元测试。

覆盖港 / 沪 / 深 / 美股各常见变形，以及非法输入与回退路径。
"""

from __future__ import annotations

import pytest

from dayu.fins.ticker_normalization import (
    NormalizedTicker,
    normalize_ticker,
    ticker_to_company_id,
    try_normalize_ticker,
)


# ---------- 港股 ----------


@pytest.mark.parametrize(
    "raw, canonical",
    [
        ("0700", "0700"),
        ("700", "0700"),
        ("00700", "0700"),
        ("0700.HK", "0700"),
        ("700.HK", "0700"),
        ("00700.HK", "0700"),
        ("HK.00700", "0700"),
        ("HK:0700", "0700"),
        ("HK-700", "0700"),
        ("0700HK", "0700"),
        ("hk.0700", "0700"),
        ("  0700  ", "0700"),
        ("HKEX.0700", "0700"),
    ],
)
def test_hk_variants_collapse_to_four_digits(raw: str, canonical: str) -> None:
    """港股各种变形都归一到 4 位补零。"""
    result = normalize_ticker(raw)
    assert result.canonical == canonical
    assert result.market == "HK"
    assert result.exchange == "HKEX"
    assert result.raw == raw


def test_hk_five_digit_keeps_original() -> None:
    """港股 5 位代码（如 89988）不补零、保留原值。"""
    result = normalize_ticker("89988")
    assert result.canonical == "89988"
    assert result.market == "HK"
    assert result.exchange == "HKEX"


def test_hk_five_digit_with_suffix() -> None:
    """港股 5 位代码带后缀也能识别。"""
    result = normalize_ticker("89988.HK")
    assert result.canonical == "89988"
    assert result.market == "HK"


# ---------- 沪股 ----------


@pytest.mark.parametrize(
    "raw",
    [
        "600519",
        "600519.SH",
        "600519.SS",
        "SH.600519",
        "sh600519",
        "SH:600519",
        "SSE.600519",
        "600519.SSE",
    ],
)
def test_sh_variants(raw: str) -> None:
    """沪股主板代码各种变形都归一到 6 位数字。"""
    result = normalize_ticker(raw)
    assert result.canonical == "600519"
    assert result.market == "CN"
    assert result.exchange == "SSE"


def test_sh_star_market() -> None:
    """沪股科创板（68 开头）识别正确。"""
    result = normalize_ticker("688981.SH")
    assert result.canonical == "688981"
    assert result.market == "CN"
    assert result.exchange == "SSE"


# ---------- 深股 ----------


@pytest.mark.parametrize(
    "raw, canonical",
    [
        ("000333", "000333"),
        ("000333.SZ", "000333"),
        ("SZ.000333", "000333"),
        ("sz000333", "000333"),
        ("300750", "300750"),
        ("300750.SZ", "300750"),
        ("SZSE.000333", "000333"),
    ],
)
def test_sz_variants(raw: str, canonical: str) -> None:
    """深股主板 / 创业板代码归一到 6 位数字。"""
    result = normalize_ticker(raw)
    assert result.canonical == canonical
    assert result.market == "CN"
    assert result.exchange == "SZSE"


# ---------- 美股 ----------


@pytest.mark.parametrize(
    "raw, canonical",
    [
        ("AAPL", "AAPL"),
        ("aapl", "AAPL"),
        ("AAPL.US", "AAPL"),
        ("US.AAPL", "AAPL"),
        ("US:AAPL", "AAPL"),
        ("AAPL.O", "AAPL"),
        ("AAPL.N", "AAPL"),
        ("AAPL.OQ", "AAPL"),
        ("AAPL.PK", "AAPL"),
        ("NASDAQ.AAPL", "AAPL"),
        ("NYSE.BRK", "BRK"),
        ("BRK.B", "BRK-B"),
        ("BRK-A", "BRK-A"),
        ("BRK.A", "BRK-A"),
        ("BF.B", "BF-B"),
        ("BF-B", "BF-B"),
        ("SHEL", "SHEL"),
        ("SHOP", "SHOP"),
    ],
)
def test_us_variants(raw: str, canonical: str) -> None:
    """美股各种变形，exchange 一律为 None。"""
    result = normalize_ticker(raw)
    assert result.canonical == canonical
    assert result.market == "US"
    assert result.exchange is None


# ---------- 非法输入 ----------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "中国平安",
        "Apple Inc.",
        "123456789",
        "199999",  # 6 位但首位非 0/3/6
        "700.UN",  # 未知后缀
        "AAPL.SW",  # 外部交易所后缀，不是美股类股分隔符
        "!!!",
        "A" * 9,  # 超过 US 长度上限
    ],
)
def test_invalid_inputs_raise_value_error(raw: str) -> None:
    """无法识别的输入抛 ValueError。"""
    with pytest.raises(ValueError):
        normalize_ticker(raw)


# ---------- try_normalize_ticker ----------


def test_try_normalize_success() -> None:
    """成功路径返回 NormalizedTicker。"""
    result = try_normalize_ticker("0700.HK")
    assert result is not None
    assert result.canonical == "0700"


def test_try_normalize_failure_returns_none() -> None:
    """失败路径返回 None，不抛错。"""
    assert try_normalize_ticker("Apple Inc.") is None
    assert try_normalize_ticker("") is None
    assert try_normalize_ticker("中国平安") is None


# ---------- ticker_to_company_id ----------


def test_ticker_to_company_id_returns_canonical_with_exchange_suffix() -> None:
    """公司 ID 由 canonical ticker 与交易所/市场后缀组成。"""

    ticker = NormalizedTicker(
        canonical="AAPL", market="US", exchange=None, raw="AAPL.US"
    )
    assert ticker_to_company_id(ticker) == "AAPL_US"
    assert ticker_to_company_id(normalize_ticker("600519.SH")) == "600519_SSE"
    assert ticker_to_company_id(normalize_ticker("000333.SZ")) == "000333_SZSE"
    assert ticker_to_company_id(normalize_ticker("0700.HK")) == "0700_HKEX"


# ---------- 回退路径 ----------


def test_us_ticker_starting_with_sh_not_misparsed() -> None:
    """SHEL/SHOP 这类字母开头美股不能被误拆为 SH 前缀 + 非法主体。"""
    result = normalize_ticker("SHEL")
    assert result.market == "US"
    assert result.canonical == "SHEL"


@pytest.mark.parametrize("raw", ["ALVO", "NVO", "TPVG", "NVGO"])
def test_us_ticker_ending_in_single_letter_not_misparsed(raw: str) -> None:
    """无分隔符的单字母尾缀（N/O）不能被误剥。"""
    result = normalize_ticker(raw)
    assert result.market == "US"
    assert result.canonical == raw


def test_raw_field_preserves_original() -> None:
    """raw 字段保留未大写、未去首尾空白的原始输入。"""
    result = normalize_ticker("  aapl  ")
    assert result.raw == "  aapl  "
    assert result.canonical == "AAPL"
