"""财报 Web 下载表单：按 ticker 市场区分 SEC 与 A 股/港股财期选项。

纯函数模块，不依赖 Streamlit，供 ``download_panel`` 与单测复用。

设计要点：市场判定与 ``dayu-cli download`` / Fins CN 链路一致，依赖
``try_normalize_ticker`` 的 ``market`` 字段；港股与 A 股共用
``DEFAULT_FORMS_CN`` 所列财期字面量（与 ``cn_form_utils`` 默认集合一致）。
"""

from __future__ import annotations

from typing import Final, Literal

from dayu.fins.pipelines.cn_form_utils import DEFAULT_FORMS_CN
from dayu.fins.score_sec_ci import FORM_PROFILES
from dayu.fins.ticker_normalization import try_normalize_ticker

DownloadFormMarketKind = Literal["sec", "cn_hk"]
"""下载表单所服务的市场分类：美股 SEC 或 A 股/港股财期。"""

_SEC_FORM_OPTIONS: Final[tuple[str, ...]] = tuple(FORM_PROFILES.keys())
"""SEC 表单多选固定选项（与 ``FORM_PROFILES`` 键顺序一致）。"""

_CN_HK_FORM_OPTIONS: Final[tuple[str, ...]] = DEFAULT_FORMS_CN
"""A 股与港股下载链路支持的财期字面量（与 ``cn_form_utils`` 默认集合一致）。"""

_DEFAULT_SEC_FORMS: Final[tuple[str, ...]] = ("10-K", "10-Q")
_DEFAULT_CN_HK_FORMS: Final[tuple[str, ...]] = ("FY", "H1")


def classify_fins_download_form_market(ticker: str) -> DownloadFormMarketKind:
    """根据 ticker 判定下载表单应使用 SEC 类型还是 A 股/港股财期类型。

    使用 ``try_normalize_ticker`` 解析市场；无法识别时回退为 ``sec``，与
    历史 Web 默认（美股表单）一致。

    参数:
        ticker: 股票代码或自选展示用代码。

    返回值:
        ``\"sec\"`` 或 ``\"cn_hk\"``。

    异常:
        无。
    """

    stripped = ticker.strip()
    if not stripped:
        return "sec"
    normalized = try_normalize_ticker(stripped)
    if normalized is None:
        return "sec"
    if normalized.market in ("CN", "HK"):
        return "cn_hk"
    return "sec"


def fins_download_form_options(market: DownloadFormMarketKind) -> tuple[str, ...]:
    """返回指定市场下多选控件可用的表单/财期 token 列表。

    参数:
        market: ``sec`` 或 ``cn_hk``。

    返回值:
        选项元组。

    异常:
        无。
    """

    if market == "cn_hk":
        return _CN_HK_FORM_OPTIONS
    return _SEC_FORM_OPTIONS


def fins_download_default_form_selection(market: DownloadFormMarketKind) -> tuple[str, ...]:
    """返回多选控件在未记忆用户选择时的默认勾选集合。

    参数:
        market: ``sec`` 或 ``cn_hk``。

    返回值:
        默认勾选的表单或财期 token 元组。

    异常:
        无。
    """

    if market == "cn_hk":
        return _DEFAULT_CN_HK_FORMS
    return _DEFAULT_SEC_FORMS


def fins_download_form_help_text(market: DownloadFormMarketKind) -> str:
    """返回多选控件旁的帮助文案。

    参数:
        market: ``sec`` 或 ``cn_hk``。

    返回值:
        帮助字符串。

    异常:
        无。
    """

    if market == "cn_hk":
        return (
            "A 股 / 港股使用财期代号：FY 年报、H1 半年报、Q1–Q4 季报；"
            "与美股 SEC 的 10-K/10-Q 等不是同一套标签。"
        )
    return "美股等 SEC 注册发行人使用的表单类型（如 10-K 年报、10-Q 季报）。"
