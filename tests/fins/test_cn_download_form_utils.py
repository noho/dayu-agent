"""``dayu/fins/pipelines/cn_form_utils.py`` 单元测试。

覆盖：

- 默认 forms（CN/HK 一致）；
- ``Q2`` 归一为 ``H1`` 并落 summary 标记；
- 中文输入与拼写多样性；
- 非法 token 抛 ``ValueError``；
- 窗口默认值（``today=fixture`` 注入）；
- 按财期区分的默认业务窗口；
- 窗口日期串解析（``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD``）；
- ``start > end`` 抛 ``ValueError``。
"""

from __future__ import annotations

import datetime as dt

import pytest

from dayu.fins.pipelines.cn_form_utils import (
    DEFAULT_FORMS_CN,
    DEFAULT_FORMS_HK,
    DownloadWindow,
    PeriodDownloadWindow,
    TargetPeriodResolution,
    resolve_period_windows,
    resolve_target_periods,
    resolve_window,
    split_cn_form_input,
)


def test_default_forms_cn_and_hk_are_identical_quad() -> None:
    """CN/HK 默认 forms 一致：``FY/H1/Q1/Q3``。"""

    assert DEFAULT_FORMS_CN == ("FY", "H1", "Q1", "Q3")
    assert DEFAULT_FORMS_HK == ("FY", "H1", "Q1", "Q3")


def test_resolve_target_periods_empty_input_returns_default_for_cn() -> None:
    """空输入对 CN 返回 :data:`DEFAULT_FORMS_CN`。"""

    result = resolve_target_periods((), "CN")

    assert result.target_periods == DEFAULT_FORMS_CN
    assert result.notes == ()


def test_resolve_target_periods_none_input_returns_default_for_cn() -> None:
    """``None`` 输入对 CN 返回默认 forms（与 ``service_runtime`` 透传 ``None`` 一致）。"""

    result = resolve_target_periods(None, "CN")

    assert result.target_periods == DEFAULT_FORMS_CN
    assert result.notes == ()


def test_resolve_target_periods_csv_string_input_split_by_comma() -> None:
    """CSV 字符串输入按英文逗号切分（``service_runtime._coerce_forms_input`` 形态）。"""

    result = resolve_target_periods("FY,H1", "CN")

    assert result.target_periods == ("FY", "H1")
    assert result.notes == ()


def test_resolve_target_periods_whitespace_string_input_split_by_space() -> None:
    """空白分隔字符串输入也能解析（CLI 直接拼接形态）。"""

    result = resolve_target_periods("FY H1 Q1", "CN")

    assert result.target_periods == ("FY", "H1", "Q1")
    assert result.notes == ()


def test_resolve_target_periods_chinese_comma_string_input_split() -> None:
    """中文全角逗号分隔同样能解析，避免用户输入习惯踩坑。"""

    result = resolve_target_periods("FY，H1", "CN")

    assert result.target_periods == ("FY", "H1")


def test_split_cn_form_input_none_returns_empty_tuple() -> None:
    """``None`` -> 空 tuple；调用方走默认 forms。"""

    assert split_cn_form_input(None) == ()


def test_split_cn_form_input_string_handles_mixed_separators() -> None:
    """字符串输入支持英文逗号 / 中文逗号 / 空白混合分隔，连续分隔符过滤为空。"""

    assert split_cn_form_input("FY, H1，Q1  Q3") == ("FY", "H1", "Q1", "Q3")


def test_split_cn_form_input_tuple_passes_through() -> None:
    """tuple 输入原样返回，保留顺序与重复 token。"""

    assert split_cn_form_input(("FY", "Q1", "FY")) == ("FY", "Q1", "FY")


def test_split_cn_form_input_blank_string_returns_empty() -> None:
    """全空白字符串解析后为空 tuple。"""

    assert split_cn_form_input("   ") == ()


def test_resolve_target_periods_empty_input_returns_default_for_hk() -> None:
    """空输入对 HK 返回 :data:`DEFAULT_FORMS_HK`。"""

    result = resolve_target_periods((), "HK")

    assert result.target_periods == DEFAULT_FORMS_HK
    assert result.notes == ()


def test_resolve_target_periods_explicit_forms_preserved_canonical_order() -> None:
    """显式 form 输入按字面量稳定顺序去重，不被默认值覆盖。"""

    result = resolve_target_periods(("Q3", "FY", "Q1"), "CN")

    assert result.target_periods == ("FY", "Q1", "Q3")
    assert result.notes == ()


def test_resolve_target_periods_q2_normalized_to_h1_with_summary_note() -> None:
    """``Q2`` 输入归一为 ``H1`` 并产出 ``cn_q2_normalized_to_h1`` 标记。"""

    result = resolve_target_periods(("Q2",), "CN")

    assert result.target_periods == ("H1",)
    assert "cn_q2_normalized_to_h1" in result.notes


def test_resolve_target_periods_chinese_q2_normalized_to_h1() -> None:
    """中文 ``二季报`` 同样归一为 ``H1`` 并产出标记。"""

    result = resolve_target_periods(("二季报",), "CN")

    assert result.target_periods == ("H1",)
    assert "cn_q2_normalized_to_h1" in result.notes


def test_resolve_target_periods_q2_note_emitted_only_once() -> None:
    """多次出现 Q2-style 输入只产出一次标记。"""

    result = resolve_target_periods(("Q2", "2Q", "二季报", "H1"), "CN")

    assert result.target_periods == ("H1",)
    assert result.notes.count("cn_q2_normalized_to_h1") == 1


def test_resolve_target_periods_chinese_inputs_supported() -> None:
    """中文 ``年报``/``半年报``/``一季报``/``三季报`` 全部能识别。"""

    result = resolve_target_periods(("年报", "半年报", "一季报", "三季报"), "CN")

    assert result.target_periods == ("FY", "H1", "Q1", "Q3")
    assert result.notes == ()


def test_resolve_target_periods_invalid_token_raises_value_error() -> None:
    """无法识别的 token 必须抛 ``ValueError``，由调用方升级为 failed。"""

    with pytest.raises(ValueError) as exc_info:
        resolve_target_periods(("FY", "10-K"), "CN")

    assert "10-K" in str(exc_info.value)


def test_resolve_target_periods_blank_tokens_filtered() -> None:
    """空白 token 被忽略；剩余必须解析成功。"""

    result = resolve_target_periods(("", "  ", "FY"), "CN")

    assert result.target_periods == ("FY",)


def test_resolve_target_periods_all_blank_raises_value_error() -> None:
    """全是空白且非空列表时抛错（与"完全空 -> 默认值"区分）。"""

    with pytest.raises(ValueError):
        resolve_target_periods(("", " "), "CN")


def test_resolve_target_periods_returns_target_period_resolution_dataclass() -> None:
    """返回值类型是 :class:`TargetPeriodResolution`。"""

    result = resolve_target_periods(("FY",), "HK")

    assert isinstance(result, TargetPeriodResolution)


def test_resolve_window_defaults_to_five_year_lookback_with_grace() -> None:
    """``start_date`` 缺省 -> ``end - 5 年 - 60 天``，``end`` 缺省 -> 注入 today。"""

    today = dt.date(2025, 5, 1)

    window = resolve_window(None, None, today=today)

    assert window == DownloadWindow(
        start_date="2020-03-02",
        end_date="2025-05-01",
    )


def test_resolve_period_windows_defaults_annual_five_years_interim_two_years() -> None:
    """默认业务窗口应为年报 5 年、半年报/季报 2 年。"""

    windows = resolve_period_windows(
        target_periods=("FY", "H1", "Q1", "Q3"),
        start_date=None,
        end_date=None,
        today=dt.date(2025, 5, 1),
    )

    assert windows == (
        PeriodDownloadWindow("FY", "2020-03-02", "2025-05-01"),
        PeriodDownloadWindow("H1", "2023-03-02", "2025-05-01"),
        PeriodDownloadWindow("Q1", "2023-03-02", "2025-05-01"),
        PeriodDownloadWindow("Q3", "2023-03-02", "2025-05-01"),
    )


def test_resolve_period_windows_explicit_start_applies_to_all_periods() -> None:
    """显式 start_date 应覆盖各财期默认回溯年限。"""

    windows = resolve_period_windows(
        target_periods=("FY", "H1"),
        start_date="2024",
        end_date="2025",
        today=dt.date(2026, 5, 1),
    )

    assert windows == (
        PeriodDownloadWindow("FY", "2024-01-01", "2025-12-31"),
        PeriodDownloadWindow("H1", "2024-01-01", "2025-12-31"),
    )


def test_resolve_window_year_only_full_year_window() -> None:
    """``YYYY`` 输入展开为整年窗口。"""

    window = resolve_window("2024", "2024", today=dt.date(2025, 5, 1))

    assert window == DownloadWindow(start_date="2024-01-01", end_date="2024-12-31")


def test_resolve_window_year_month_input_expanded_to_full_month() -> None:
    """``YYYY-MM`` 输入展开为整月窗口（end 取月末）。"""

    window = resolve_window("2024-02", "2024-02", today=dt.date(2025, 5, 1))

    assert window == DownloadWindow(start_date="2024-02-01", end_date="2024-02-29")


def test_resolve_window_full_date_input_round_trips() -> None:
    """``YYYY-MM-DD`` 输入直接采用。"""

    window = resolve_window("2024-03-15", "2024-09-30", today=dt.date(2025, 5, 1))

    assert window == DownloadWindow(start_date="2024-03-15", end_date="2024-09-30")


def test_resolve_window_invalid_date_format_raises() -> None:
    """非法日期格式抛 ``ValueError``。"""

    with pytest.raises(ValueError):
        resolve_window("2024/03/15", None)


def test_resolve_window_start_after_end_raises() -> None:
    """``start > end`` 抛 ``ValueError``。"""

    with pytest.raises(ValueError):
        resolve_window("2025-01-01", "2024-12-31")


def test_resolve_window_today_default_uses_real_today_when_not_injected() -> None:
    """不注入 ``today`` 时使用真实当天日期（仅校验返回格式与不抛异常）。"""

    window = resolve_window(None, None)

    assert len(window.start_date) == 10
    assert len(window.end_date) == 10
    # ISO 8601 形态。
    dt.date.fromisoformat(window.start_date)
    dt.date.fromisoformat(window.end_date)
