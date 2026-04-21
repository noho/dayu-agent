"""Ticker 归一化唯一真源。

该模块提供港/沪/深/美股 ticker 各种常见变形到 canonical 形态的统一归一化逻辑。

设计要点：
- 仅暴露 ``NormalizedTicker`` 与 ``normalize_ticker`` / ``try_normalize_ticker``
  / ``ticker_to_company_id`` 作为公共 API；其它均为模块级私有辅助。
- Canonical 形态：港股 4 位补零（``0700``）或保留原 5 位（``89988``），沪股 6 位
  （``600519``），深股 6 位（``000333`` / ``300750``），美股保留字母（``AAPL``、``BRK.B``）。
- 美股在无明确交易所后缀时，``exchange`` 返回 ``None``，当前不区分 NYSE/NASDAQ。
- 无法识别的输入：``normalize_ticker`` 抛 ``ValueError``；``try_normalize_ticker``
  返回 ``None``。
- 市场前后缀识别失败时（例如 ``SHEL`` 被误剥为 ``SH`` + ``EL``），会回退到
  整体自适应判定，避免把合法美股误判为非法沪股。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Final, Literal, Optional

Market = Literal["US", "HK", "CN"]
Exchange = Literal["HKEX", "SSE", "SZSE"]


@dataclass(frozen=True)
class NormalizedTicker:
    """规范化 ticker 结果。

    Attributes:
        canonical: 规范裸码；港/沪/深为纯数字字符串，美股保留字母（可能含
            一个 ``.`` 或 ``-`` 分节，如 ``BRK.B``）。
        market: 市场标识，取值 ``"US"`` / ``"HK"`` / ``"CN"``。
        exchange: 交易所标识；港股 ``"HKEX"``、沪股 ``"SSE"``、深股 ``"SZSE"``；
            美股无后缀时为 ``None``。
        raw: 原始输入（未大写、未去空白），便于日志与诊断。
    """

    canonical: str
    market: Market
    exchange: Optional[Exchange]
    raw: str


# ---- 市场前后缀识别常量（仅在本模块内使用） ----
# 前缀形如 ``HK.00700`` / ``SH:600519`` / ``NASDAQ-AAPL``；分隔符可选。
# 单字符交易所后缀（``N``/``O``）不纳入前缀，避免把诸如 ``OPEN`` 误拆分为 ``O`` + ``PEN``。
_HK_TOKENS: Final[frozenset[str]] = frozenset({"HK", "HKEX"})
_SH_TOKENS: Final[frozenset[str]] = frozenset({"SH", "SS", "SSE"})
_SZ_TOKENS: Final[frozenset[str]] = frozenset({"SZ", "SZSE"})
_US_TOKENS: Final[frozenset[str]] = frozenset({"US", "N", "O", "OQ", "PK", "NASDAQ", "NYSE"})

_PREFIX_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(HKEX|HK|SSE|SH|SS|SZSE|SZ|NASDAQ|NYSE|US)[.:\-_]?(.+)$"
)
# 带分隔符的后缀：支持所有 token（含单字母 N/O）。
_SUFFIX_WITH_SEP_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(.+?)[.\-_](HKEX|HK|SSE|SH|SS|SZSE|SZ|NASDAQ|NYSE|OQ|PK|US|N|O)$"
)
# 无分隔符的后缀：仅识别多字符 token。排除 N/O，避免把 ``ALVO``/``NVO``
# 这类字母美股拆成 ``ALV``+``O``。
_SUFFIX_NO_SEP_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(.+?)(HKEX|HK|SSE|SH|SS|SZSE|SZ|NASDAQ|NYSE|OQ|PK|US)$"
)
_US_SYMBOL_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Z]+(?:[.\-][A-Z0-9]+)?$")
_MAX_US_SYMBOL_LENGTH: Final[int] = 8


def normalize_ticker(raw: str) -> NormalizedTicker:
    """把 ticker 的各种常见变形归一化为 canonical 形态。

    支持（覆盖例子，非穷举）：

    - 港股：``0700`` / ``700`` / ``00700`` / ``0700.HK`` / ``700.HK`` /
      ``HK.00700`` / ``HK:0700`` / ``0700HK`` / ``89988`` →
      canonical=``0700`` 或 ``89988``，market=``HK``，exchange=``HKEX``。
    - 沪股：``600519`` / ``600519.SH`` / ``600519.SS`` / ``SH.600519`` /
      ``sh600519`` → canonical=同值，market=``CN``，exchange=``SSE``。
    - 深股：``000333`` / ``300750`` / ``000333.SZ`` / ``SZ.000333`` /
      ``sz000333`` → canonical=同值，market=``CN``，exchange=``SZSE``。
    - 美股：``AAPL`` / ``AAPL.US`` / ``AAPL.O`` / ``US.AAPL`` / ``BRK.B`` /
      ``BF.B`` / ``SHEL`` / ``SHOP`` → canonical 保留原字母形态，market=``US``，
      exchange=``None``。

    Args:
        raw: 原始输入字符串。

    Returns:
        ``NormalizedTicker``。

    Raises:
        ValueError: 输入为空、或无法识别为任何支持市场的 ticker 形态时抛出。
    """

    text = unicodedata.normalize("NFKC", raw).strip()
    if not text:
        raise ValueError("ticker 不能为空")
    upper = text.upper()

    # 先尝试按市场 hint 剥离前后缀；构造失败时回退到整体自适应判定，
    # 以避免把 ``SHEL``/``SHOP`` 这类字母开头的美股当作非法沪股。
    body, market_token = _split_market_token(upper)
    if market_token is not None:
        hinted = _build_by_token(body=body, token=market_token, raw=raw)
        if hinted is not None:
            return hinted
    auto = _build_auto(upper=upper, raw=raw)
    if auto is not None:
        return auto
    raise ValueError(f"无法识别的 ticker 形态: {raw!r}")


def try_normalize_ticker(raw: str) -> Optional[NormalizedTicker]:
    """``normalize_ticker`` 的非抛错版本。

    用于 service 层"ticker 可能是公司名"的分支：识别失败时由上层回退到
    公司 alias 查表，而不是直接报错。

    Args:
        raw: 原始输入。

    Returns:
        ``NormalizedTicker``；输入非字符串、为空、或无法识别时返回 ``None``。

    Raises:
        无。
    """

    try:
        return normalize_ticker(raw)
    except (TypeError, ValueError):
        return None


def ticker_to_company_id(ticker: NormalizedTicker) -> str:
    """由 ``NormalizedTicker`` 推导公司 ID。

    当前实现直接返回 ``ticker.canonical``；保留该接口以便后续接入更精细的
    公司主体映射（跨市场上市折叠、CIK、统一社会信用代码等），属稳定契约、
    实现可演进。

    Args:
        ticker: 已归一化的 ticker。

    Returns:
        公司 ID 字符串。

    Raises:
        无。
    """

    return ticker.canonical


# ---------- 模块级私有辅助 ----------


def _split_market_token(upper: str) -> tuple[str, Optional[str]]:
    """尝试剥离 ticker 中的市场前后缀。

    Args:
        upper: 已 ``upper()`` 后的 ticker 文本。

    Returns:
        ``(body, market_token)``；未匹配到市场标识时 ``market_token`` 为
        ``None``，``body`` 为整体 ``upper``。

    Raises:
        无。
    """

    prefix_match = _PREFIX_PATTERN.match(upper)
    if prefix_match is not None:
        return prefix_match.group(2), prefix_match.group(1)
    suffix_match = _SUFFIX_WITH_SEP_PATTERN.match(upper)
    if suffix_match is not None:
        return suffix_match.group(1), suffix_match.group(2)
    suffix_match = _SUFFIX_NO_SEP_PATTERN.match(upper)
    if suffix_match is not None:
        return suffix_match.group(1), suffix_match.group(2)
    return upper, None


def _build_by_token(*, body: str, token: str, raw: str) -> Optional[NormalizedTicker]:
    """根据市场 hint 构造 ``NormalizedTicker``。

    Args:
        body: 剥离市场标识后的主体字符串。
        token: 识别出的市场标识。
        raw: 原始输入。

    Returns:
        构造成功的 ``NormalizedTicker``；格式与 hint 不匹配时返回 ``None``，
        上层据此回退到自适应判定。

    Raises:
        无。
    """

    if token in _HK_TOKENS:
        return _build_hk(body, raw)
    if token in _SH_TOKENS:
        return _build_sh(body, raw)
    if token in _SZ_TOKENS:
        return _build_sz(body, raw)
    if token in _US_TOKENS:
        return _build_us(body, raw)
    return None


def _build_auto(*, upper: str, raw: str) -> Optional[NormalizedTicker]:
    """无市场 hint 时的自适应判定。

    Args:
        upper: 已 ``upper()`` 的 ticker 文本。
        raw: 原始输入。

    Returns:
        识别成功的 ``NormalizedTicker``；失败返回 ``None``。

    Raises:
        无。
    """

    if upper.isdigit():
        return _classify_pure_digits(upper, raw)
    return _build_us(upper, raw)


def _classify_pure_digits(body: str, raw: str) -> Optional[NormalizedTicker]:
    """纯数字无市场 hint 时的市场判定。

    Args:
        body: 纯数字主体。
        raw: 原始输入。

    Returns:
        识别成功的 ``NormalizedTicker``；失败返回 ``None``。

    Raises:
        无。
    """

    length = len(body)
    if 1 <= length <= 5:
        return _build_hk(body, raw)
    if length == 6:
        head = body[0]
        if head == "6":
            return NormalizedTicker(canonical=body, market="CN", exchange="SSE", raw=raw)
        if head in ("0", "3"):
            return NormalizedTicker(canonical=body, market="CN", exchange="SZSE", raw=raw)
    return None


def _build_hk(body: str, raw: str) -> Optional[NormalizedTicker]:
    """构造港股 ``NormalizedTicker``。

    规则：
    - 主体必须是纯数字。
    - 去除前导零后长度需在 1–5 之间（覆盖经典 4 位与新发 5 位代码如 ``89988``）。
    - 长度 ≤ 4 时补零到 4 位；长度 5 时保留原样。

    Args:
        body: ticker 主体。
        raw: 原始输入。

    Returns:
        港股 ``NormalizedTicker``；格式非法返回 ``None``。

    Raises:
        无。
    """

    if not body.isdigit():
        return None
    stripped = body.lstrip("0") or "0"
    length = len(stripped)
    if length > 5:
        return None
    canonical = stripped.zfill(4) if length <= 4 else stripped
    return NormalizedTicker(canonical=canonical, market="HK", exchange="HKEX", raw=raw)


def _build_sh(body: str, raw: str) -> Optional[NormalizedTicker]:
    """构造沪股 ``NormalizedTicker``。

    规则：主体必须是 6 位纯数字、首位为 ``6``（主板 ``60xxxx``、科创板 ``68xxxx``）。

    Args:
        body: ticker 主体。
        raw: 原始输入。

    Returns:
        沪股 ``NormalizedTicker``；格式非法返回 ``None``。

    Raises:
        无。
    """

    if not body.isdigit() or len(body) != 6 or body[0] != "6":
        return None
    return NormalizedTicker(canonical=body, market="CN", exchange="SSE", raw=raw)


def _build_sz(body: str, raw: str) -> Optional[NormalizedTicker]:
    """构造深股 ``NormalizedTicker``。

    规则：主体必须是 6 位纯数字、首位为 ``0``（主板 / 中小板）或 ``3``（创业板）。

    Args:
        body: ticker 主体。
        raw: 原始输入。

    Returns:
        深股 ``NormalizedTicker``；格式非法返回 ``None``。

    Raises:
        无。
    """

    if not body.isdigit() or len(body) != 6 or body[0] not in ("0", "3"):
        return None
    return NormalizedTicker(canonical=body, market="CN", exchange="SZSE", raw=raw)


def _build_us(body: str, raw: str) -> Optional[NormalizedTicker]:
    """构造美股 ``NormalizedTicker``。

    规则：首字符字母、仅含 ``A-Z`` 以及可选 ``.`` / ``-`` 分节（如 ``BRK.B``），
    长度不超过 ``_MAX_US_SYMBOL_LENGTH``。

    Args:
        body: ticker 主体。
        raw: 原始输入。

    Returns:
        美股 ``NormalizedTicker``；格式非法返回 ``None``。

    Raises:
        无。
    """

    if not body or len(body) > _MAX_US_SYMBOL_LENGTH:
        return None
    if _US_SYMBOL_PATTERN.fullmatch(body) is None:
        return None
    return NormalizedTicker(canonical=body, market="US", exchange=None, raw=raw)
