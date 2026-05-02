"""上传场景公司元数据写入助手。

该模块聚合 upload 相关的 company meta 写入逻辑，目标是：
1. 统一 create/update 场景下的 company meta 解析规则。
2. 已存在 `meta.json` 时，将其视为唯一真相源，忽略重复传入的 company meta。
3. 统一 `CompanyMeta` 的构建规则，避免 CN/SEC pipeline 重复实现。
4. 保持 delete 场景不触发公司级元数据写入。
"""

from __future__ import annotations

from typing import Optional

from dayu.log import Log
from dayu.fins.domain.document_models import CompanyMeta, now_iso8601
from dayu.fins.ticker_normalization import normalize_ticker, ticker_to_company_id
from dayu.fins.storage import CompanyMetaRepositoryProtocol

UPLOAD_ACTIONS_REQUIRING_COMPANY_META = frozenset({"create", "update"})
RESOLVER_VERSION = "market_resolver_v1.0.0"
MODULE = "FINS.UPLOAD_COMPANY_META"


def upsert_company_meta_for_upload(
    *,
    repository: CompanyMetaRepositoryProtocol,
    ticker: str,
    action: str,
    company_id: Optional[str],
    company_name: Optional[str],
    ticker_aliases: Optional[list[str]] = None,
) -> None:
    """在上传链路中按规则写入公司级元数据。

    Args:
        repository: 公司元数据仓储实现。
        ticker: 股票代码。
        action: 上传动作。
        company_id: 兼容既有调用方的可选字段；上传链路不会把它作为身份真源。
        company_name: 公司名称。
        ticker_aliases: 可选 ticker alias 列表；缺失时仅使用规范 ticker。

    Returns:
        无。

    Raises:
        ValueError: create/update 场景在缺少公司名称时抛出。
        OSError: 仓储写入失败时抛出。
    """

    normalized_action = action.strip().lower()
    if normalized_action not in UPLOAD_ACTIONS_REQUIRING_COMPANY_META:
        return

    existing_meta = _load_existing_company_meta(repository=repository, ticker=ticker)
    if existing_meta is not None:
        _warn_ignored_company_meta_args(
            ticker=existing_meta.ticker,
            company_id=company_id,
            company_name=company_name,
        )
        return

    profile = normalize_ticker(ticker)
    normalized_company_id = ticker_to_company_id(profile)
    normalized_company_name = _require_company_meta_field(
        value=company_name,
        option_name="--company-name",
    )
    normalized_ticker_aliases = _normalize_ticker_aliases(
        canonical_ticker=profile.canonical,
        ticker_aliases=ticker_aliases,
    )
    repository.upsert_company_meta(
        CompanyMeta(
            company_id=normalized_company_id,
            company_name=normalized_company_name,
            ticker=profile.canonical,
            market=profile.market,
            resolver_version=RESOLVER_VERSION,
            updated_at=now_iso8601(),
            ticker_aliases=normalized_ticker_aliases,
        )
    )


def build_upload_company_id(ticker: str) -> str:
    """按上传链路稳定规则生成公司 ID。

    Args:
        ticker: 股票代码，可为原始输入或 canonical ticker。

    Returns:
        由 ``ticker_to_company_id`` 推导出的公司 ID。

    Raises:
        ValueError: ticker 为空或无法归一化时抛出。
    """

    return ticker_to_company_id(normalize_ticker(ticker))


def _require_company_meta_field(*, value: Optional[str], option_name: str) -> str:
    """校验并返回 company meta 字段值。

    Args:
        value: 原始字段值。
        option_name: CLI 参数名（用于构造错误信息）。

    Returns:
        去除首尾空白后的字段值。

    Raises:
        ValueError: 字段为空时抛出。
    """

    normalized_value = str(value or "").strip()
    if not normalized_value:
        raise ValueError(f"create/update 时必须提供 {option_name}")
    return normalized_value


def _load_existing_company_meta(
    *,
    repository: CompanyMetaRepositoryProtocol,
    ticker: str,
) -> Optional[CompanyMeta]:
    """读取现有公司元数据。

    Args:
        repository: 公司元数据仓储实现。
        ticker: 股票代码。

    Returns:
        若仓储中已存在公司元数据则返回该对象，否则返回 `None`。

    Raises:
        ValueError: 现有元数据格式非法时抛出。
        OSError: 仓储读取失败时抛出。
    """

    try:
        return repository.get_company_meta(ticker)
    except FileNotFoundError:
        return None


def _normalize_ticker_aliases(
    *,
    canonical_ticker: str,
    ticker_aliases: Optional[list[str]],
) -> list[str]:
    """标准化公司级 ticker alias 列表。

    Args:
        canonical_ticker: 规范 ticker。
        ticker_aliases: 原始 alias 列表。

    Returns:
        去重后的大写 ticker 列表，且首项始终为规范 ticker。

    Raises:
        无。
    """

    normalized_canonical = str(canonical_ticker).strip().upper()
    normalized_aliases: list[str] = []
    for raw_alias in [normalized_canonical, *(ticker_aliases or [])]:
        normalized_alias = str(raw_alias).strip().upper()
        if not normalized_alias:
            continue
        if normalized_alias in normalized_aliases:
            continue
        normalized_aliases.append(normalized_alias)
    return normalized_aliases


def _warn_ignored_company_meta_args(
    *,
    ticker: str,
    company_id: Optional[str],
    company_name: Optional[str],
) -> None:
    """在现有 meta 已存在时记录 company meta 参数忽略告警。

    Args:
        ticker: 股票代码。
        company_id: 传入的公司 ID。
        company_name: 传入的公司名称。

    Returns:
        无。

    Raises:
        无。
    """

    normalized_company_id = str(company_id or "").strip()
    normalized_company_name = str(company_name or "").strip()
    if not normalized_company_id and not normalized_company_name:
        return
    Log.warn(
        (
            f"ticker={ticker} 已存在公司元数据，"
            "将忽略本次上传传入的 company_id/company_name，继续使用现有 meta.json"
        ),
        module=MODULE,
    )
