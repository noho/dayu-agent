"""CN/HK 下载链路的 form / 窗口纯函数工具集。

本模块仅提供无副作用纯函数：

- :func:`split_cn_form_input`：把 CLI / service 透传过来的 form 输入
  （``None`` / CSV 字符串 / 已切分 tuple）规范化为 ``tuple[str, ...]``，
  同时支持英文逗号 ``,``、中文全角逗号 ``，`` 与空白分隔。
- :func:`resolve_target_periods`：把 form 输入解析成
  :data:`CnFiscalPeriod` 字面量集合，CN ``Q2``/``二季报`` 归一为 ``H1``。
- :func:`resolve_window`：解析 ``start_date`` / ``end_date``，生成远端查询用的
  最大窗口。
- :func:`resolve_period_windows`：生成按财期区分的业务窗口；年报默认 5 年，
  半年报/季报默认 2 年。
- 默认 forms 常量：CN/HK 默认均为 ``(FY, H1, Q1, Q3)``。

设计要点：

- 不依赖仓储 / downloader / docling，可被 workflow 与 pipeline 共享。
- 解析失败抛 ``ValueError``，调用方决定是否升级为 ``failed`` 事件。
- ``Q2`` / ``2Q`` / ``二季报`` 在 CN 链路归一为 ``H1`` 时，调用方应在
  ``summary.notes`` 标记 ``cn_q2_normalized_to_h1``；本模块通过返回结构
  显式暴露归一化事实，**不**在内部静默丢弃。
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Final

from dayu.fins.pipelines.cn_download_models import CnFiscalPeriod, CnMarketKind

DEFAULT_FORMS_CN: Final[tuple[CnFiscalPeriod, ...]] = ("FY", "H1", "Q1", "Q3")
"""A 股默认下载 form 集合。"""

DEFAULT_FORMS_HK: Final[tuple[CnFiscalPeriod, ...]] = ("FY", "H1", "Q1", "Q3")
"""港股默认下载 form 集合。HK 主板 Q1/Q3 缺失视为 skipped 而非 failed。"""

# 窗口默认值与 SEC 链路的业务意图对齐：年报 5 年，季报/半年报 2 年。
_ANNUAL_LOOKBACK_YEARS: Final[int] = 5
_INTERIM_LOOKBACK_YEARS: Final[int] = 2
_LOOKBACK_GRACE_DAYS: Final[int] = 60

# CLI ``--forms`` 输入的 token 拼写到 ``CnFiscalPeriod`` 字面量的归一化映射。
# 源覆盖：英文大写、拼音首字母、纯数字 + Q 后缀、中文"X 季报/年报/半年报"。
# Q2 / 2Q / 二季报 故意映射到 ``H1``，由调用方在 summary 标 ``cn_q2_normalized_to_h1``。
_TOKEN_TO_PERIOD: Final[dict[str, CnFiscalPeriod]] = {
    "FY": "FY",
    "ANNUAL": "FY",
    "年报": "FY",
    "年度报告": "FY",
    "H1": "H1",
    "1H": "H1",
    "半年报": "H1",
    "中报": "H1",
    "Q1": "Q1",
    "1Q": "Q1",
    "一季报": "Q1",
    "一季度报告": "Q1",
    "Q2": "H1",
    "2Q": "H1",
    "二季报": "H1",
    "Q3": "Q3",
    "3Q": "Q3",
    "三季报": "Q3",
    "三季度报告": "Q3",
}

# 触发 ``cn_q2_normalized_to_h1`` summary 标记的输入 token 集合。
_Q2_TOKENS: Final[frozenset[str]] = frozenset({"Q2", "2Q", "二季报"})

# form 输入分隔符：英文逗号 / 中文全角逗号 / 任意空白。
_FORM_INPUT_SEPARATOR_PATTERN: Final[re.Pattern[str]] = re.compile(r"[,，\s]+")

_DATE_FULL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_DATE_YEAR_MONTH_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{1,2}$")
_DATE_YEAR_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\d{4}$")


@dataclass(frozen=True)
class TargetPeriodResolution:
    """``resolve_target_periods`` 的强类型返回。

    Attributes:
        target_periods: 已去重并按 ``CnFiscalPeriod`` 字面量归一的 form tuple。
        notes: 解析过程产生的 summary 标记字符串集合。例如出现 ``Q2`` 输入时
            会包含 ``"cn_q2_normalized_to_h1"``；调用方应将其合并到
            ``DownloadResultData.notes``。
    """

    target_periods: tuple[CnFiscalPeriod, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class DownloadWindow:
    """``resolve_window`` 的强类型返回。

    Attributes:
        start_date: 已规范为 ``YYYY-MM-DD`` 的窗口起点（含）。
        end_date: 已规范为 ``YYYY-MM-DD`` 的窗口终点（含）。
    """

    start_date: str
    end_date: str


@dataclass(frozen=True)
class PeriodDownloadWindow:
    """单个财期的下载窗口。

    Attributes:
        fiscal_period: 财期字面量。
        start_date: 已规范为 ``YYYY-MM-DD`` 的窗口起点（含）。
        end_date: 已规范为 ``YYYY-MM-DD`` 的窗口终点（含）。
    """

    fiscal_period: CnFiscalPeriod
    start_date: str
    end_date: str


def split_cn_form_input(form_type: str | tuple[str, ...] | None) -> tuple[str, ...]:
    """把 CLI / service 透传的 form 输入规范化为 ``tuple[str, ...]``。

    支持三种形态，均不在此处做合法性校验，仅做切分与去空白：

    - ``None`` -> 空 tuple；调用方据此走默认 forms 分支。
    - ``str``：按英文逗号 ``,``、中文全角逗号 ``，`` 与任意空白拆分；
      连续分隔符与首尾分隔符产生的空 token 被过滤。
    - ``tuple[str, ...]``：原样返回，保留 token 顺序与重复。

    Args:
        form_type: 原始 form 输入。``service_runtime`` 经
            ``_coerce_forms_input`` 转成 CSV 字符串后调用 download_stream，
            CSV 串与 CLI 直接拼接的 ``"FY,H1"`` / ``"FY H1"`` 都能解析。

    Returns:
        归一化后的 token tuple；调用方再交给 :func:`resolve_target_periods`
        做语义校验。
    """

    if form_type is None:
        return ()
    if isinstance(form_type, tuple):
        return form_type
    tokens = tuple(token for token in _FORM_INPUT_SEPARATOR_PATTERN.split(form_type) if token)
    return tokens


def resolve_target_periods(
    raw_forms: str | tuple[str, ...] | None,
    market: CnMarketKind,
) -> TargetPeriodResolution:
    """把 form 输入解析成 :class:`CnFiscalPeriod` 集合。

    解析规则：

    - 输入为空 / ``None`` / 全空白 -> 返回 :data:`DEFAULT_FORMS_CN` 或
      :data:`DEFAULT_FORMS_HK`。
    - 字符串输入按 :func:`split_cn_form_input` 规则切分；tuple 输入直接消费。
    - 输入 token 经 :data:`_TOKEN_TO_PERIOD` 归一，``Q2``/``2Q``/``二季报``
      归一为 ``H1`` 并在 notes 标记 ``cn_q2_normalized_to_h1``。
    - 输出按字面量稳定顺序去重：``FY`` / ``H1`` / ``Q1`` / ``Q3``，**不**保留
      ``Q2`` 字面量（已被归一为 ``H1``）。

    Args:
        raw_forms: 原始 form 输入；接受 ``None`` / CSV 字符串 / 已切分 tuple。
        market: 市场标识，决定空输入时使用哪个默认集合。

    Returns:
        :class:`TargetPeriodResolution`。

    Raises:
        ValueError: 出现无法识别的 token、或全部 token 解析后为空时抛出，
            调用方据此升级为 ``PIPELINE_COMPLETED.status="failed"``。
    """

    tokens = split_cn_form_input(raw_forms)
    if not tokens:
        defaults = DEFAULT_FORMS_CN if market == "CN" else DEFAULT_FORMS_HK
        return TargetPeriodResolution(target_periods=defaults, notes=())

    seen: set[CnFiscalPeriod] = set()
    notes: list[str] = []
    invalid: list[str] = []
    for raw in tokens:
        token = raw.strip().upper()
        if not token:
            continue
        period = _TOKEN_TO_PERIOD.get(token)
        if period is None:
            invalid.append(raw)
            continue
        seen.add(period)
        if token in _Q2_TOKENS and "cn_q2_normalized_to_h1" not in notes:
            notes.append("cn_q2_normalized_to_h1")
    if invalid:
        raise ValueError(f"不支持的 form 输入: {invalid!r}")
    if not seen:
        raise ValueError("form 输入解析后为空")

    canonical_order: tuple[CnFiscalPeriod, ...] = ("FY", "H1", "Q1", "Q3")
    target_periods: tuple[CnFiscalPeriod, ...] = tuple(
        period for period in canonical_order if period in seen
    )
    return TargetPeriodResolution(target_periods=target_periods, notes=tuple(notes))


def resolve_window(
    start_date: str | None,
    end_date: str | None,
    today: dt.date | None = None,
) -> DownloadWindow:
    """解析远端查询用的最大 ``start_date`` / ``end_date``。

    解析规则：

    - ``end_date`` 缺省 -> ``today``。
    - ``start_date`` 缺省 -> ``end_date`` 回退 5 年再减 60 天宽限；这是远端
      查询最大窗口，workflow 会再按财期应用 :func:`resolve_period_windows`。
    - ``YYYY`` -> ``YYYY-01-01`` / ``YYYY-12-31``（``end`` 语义补尾）。
    - ``YYYY-MM`` -> 月初 / 月末。
    - ``YYYY-MM-DD`` -> 直接采用。

    Args:
        start_date: 原始起点字符串；``None`` 表示缺省。
        end_date: 原始终点字符串；``None`` 表示缺省。
        today: 用于注入测试。生产调用传 ``None`` 即取当天 UTC 日期。

    Returns:
        :class:`DownloadWindow`，字段已规范为 ``YYYY-MM-DD``。

    Raises:
        ValueError: 输入格式非法、或 ``start_date > end_date``。
    """

    anchor_today = today if today is not None else dt.date.today()
    end = _parse_date(end_date, is_end=True) if end_date else anchor_today
    if start_date:
        start = _parse_date(start_date, is_end=False)
    else:
        start = _subtract_years(end, _ANNUAL_LOOKBACK_YEARS) - dt.timedelta(
            days=_LOOKBACK_GRACE_DAYS
        )
    if start > end:
        raise ValueError(f"start_date 不能晚于 end_date: {start.isoformat()} > {end.isoformat()}")
    return DownloadWindow(start_date=start.isoformat(), end_date=end.isoformat())


def resolve_period_windows(
    *,
    target_periods: tuple[CnFiscalPeriod, ...],
    start_date: str | None,
    end_date: str | None,
    today: dt.date | None = None,
) -> tuple[PeriodDownloadWindow, ...]:
    """解析各财期的业务下载窗口。

    Args:
        target_periods: 已归一化目标财期。
        start_date: 用户显式起点；提供时所有财期共用该起点。
        end_date: 用户显式终点；缺省时使用 ``today``。
        today: 测试注入日期；生产传 ``None``。

    Returns:
        按 ``target_periods`` 顺序返回的窗口 tuple。默认窗口为年报 5 年、
        半年报/季报 2 年，均加 60 天披露宽限。

    Raises:
        ValueError: 日期非法或起点晚于终点时抛出。
    """

    anchor_today = today if today is not None else dt.date.today()
    end = _parse_date(end_date, is_end=True) if end_date else anchor_today
    explicit_start = _parse_date(start_date, is_end=False) if start_date else None
    windows: list[PeriodDownloadWindow] = []
    for period in target_periods:
        lookback_years = _ANNUAL_LOOKBACK_YEARS if period == "FY" else _INTERIM_LOOKBACK_YEARS
        start = explicit_start or (
            _subtract_years(end, lookback_years) - dt.timedelta(days=_LOOKBACK_GRACE_DAYS)
        )
        if start > end:
            raise ValueError(f"start_date 不能晚于 end_date: {start.isoformat()} > {end.isoformat()}")
        windows.append(
            PeriodDownloadWindow(
                fiscal_period=period,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
            )
        )
    return tuple(windows)


# ---------- 模块级私有辅助 ----------


def _parse_date(value: str, *, is_end: bool) -> dt.date:
    """解析 ``YYYY`` / ``YYYY-MM`` / ``YYYY-MM-DD`` 字符串。"""

    raw = value.strip()
    if _DATE_YEAR_PATTERN.fullmatch(raw):
        year = int(raw)
        return dt.date(year, 12, 31) if is_end else dt.date(year, 1, 1)
    if _DATE_YEAR_MONTH_PATTERN.fullmatch(raw):
        year_str, month_str = raw.split("-")
        year = int(year_str)
        month = int(month_str)
        if is_end:
            next_month = (
                dt.date(year + 1, 1, 1) if month == 12 else dt.date(year, month + 1, 1)
            )
            return next_month - dt.timedelta(days=1)
        return dt.date(year, month, 1)
    if _DATE_FULL_PATTERN.fullmatch(raw):
        return dt.datetime.strptime(raw, "%Y-%m-%d").date()
    raise ValueError(f"日期格式非法: {value!r}")


def _subtract_years(anchor_date: dt.date, years: int) -> dt.date:
    """从 ``anchor_date`` 回退 ``years`` 年；闰日 2 月 29 取 28。"""

    target_year = anchor_date.year - years
    try:
        return anchor_date.replace(year=target_year)
    except ValueError:
        return anchor_date.replace(year=target_year, day=28)


__all__ = [
    "DEFAULT_FORMS_CN",
    "DEFAULT_FORMS_HK",
    "DownloadWindow",
    "PeriodDownloadWindow",
    "TargetPeriodResolution",
    "resolve_period_windows",
    "resolve_target_periods",
    "resolve_window",
    "split_cn_form_input",
]
