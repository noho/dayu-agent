"""Service 共享 Prompt Contributions 构造函数。"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


def build_fins_default_subject_contribution(
    *,
    ticker: str,
    company_name: str | None = None,
) -> str:
    """构造财报默认分析对象 prompt contribution。

    Args:
        ticker: 股票代码。
        company_name: 公司名称。

    Returns:
        对应 ``fins_default_subject`` slot 的文本；缺少 ticker 时返回空字符串。

    Raises:
        无。
    """

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
