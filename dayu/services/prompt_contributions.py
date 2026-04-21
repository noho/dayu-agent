"""Service 共享 Prompt Contributions 构造函数。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from dayu.fins.ticker_normalization import try_normalize_ticker


def build_fins_default_subject_contribution(
    *,
    ticker: str,
    company_name: str | None = None,
) -> str:
    """构造财报默认分析对象 prompt contribution。

    优先走 ``try_normalize_ticker`` 真源把 ``0700.HK`` / ``600519.SH`` 等变形
    归一化为 canonical ticker；真源识别失败（例如用户传了公司名字符串）时
    回退到 ``strip().upper()``，保留 prompt 文本稳定。

    Args:
        ticker: 股票代码。
        company_name: 公司名称。

    Returns:
        对应 ``fins_default_subject`` slot 的文本；缺少 ticker 时返回空字符串。

    Raises:
        无。
    """

    normalized_source = try_normalize_ticker(ticker)
    if normalized_source is not None:
        normalized_ticker = normalized_source.canonical
    else:
        normalized_ticker = str(ticker or "").strip().upper()
    if not normalized_ticker:
        return ""
    normalized_company_name = str(company_name or "").strip()
    lines = ["# 当前分析对象"]
    if normalized_company_name:
        lines.append(f"你正在分析的是 {normalized_ticker}（{normalized_company_name}）。")
    else:
        lines.append(f"你正在分析的是 {normalized_ticker}。")
    return "\n".join(lines)


def build_base_user_contribution(*, now: datetime | None = None) -> str:
    """构造通用用户与运行时上下文 prompt contribution。

    Args:
        now: 可选当前时间，测试时注入。

    Returns:
        对应 ``base_user`` slot 的文本。

    Raises:
        无。
    """

    current = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    return "\n".join(
        [
            "# 用户与运行时上下文",
            f"当前时间：{current:%Y}年{current:%m}月{current:%d}日。",
        ]
    )


__all__ = [
    "build_base_user_contribution",
    "build_fins_default_subject_contribution",
]
