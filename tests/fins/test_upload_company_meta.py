"""upload_company_meta 模块测试。"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import pytest

from dayu.fins.pipelines import upload_company_meta as module
from dayu.fins.domain.document_models import CompanyMeta, CompanyMetaInventoryEntry
from dayu.fins.ticker_normalization import NormalizedTicker


@dataclass
class _RepositoryStub:
    """company meta 仓储桩。"""

    captured: Optional[CompanyMeta] = None
    existing: Optional[CompanyMeta] = None

    def scan_company_meta_inventory(self) -> list[CompanyMetaInventoryEntry]:
        """返回空盘点结果。"""

        return []

    def resolve_existing_ticker(self, ticker_candidates: list[str]) -> Optional[str]:
        """测试桩不做 ticker 候选解析。"""

        del ticker_candidates
        return None

    def get_company_meta(self, ticker: str) -> CompanyMeta:
        """返回预置的公司元数据。

        Args:
            ticker: 股票代码。

        Returns:
            预置公司元数据。

        Raises:
            FileNotFoundError: 未预置公司元数据时抛出。
        """

        if self.existing is None:
            raise FileNotFoundError(ticker)
        return self.existing

    def upsert_company_meta(self, meta: CompanyMeta) -> None:
        """记录写入请求。

        Args:
            meta: 写入对象。

        Returns:
            无。

        Raises:
            无。
        """

        self.captured = meta


@pytest.mark.unit
def test_upsert_company_meta_skips_non_create_update() -> None:
    """验证 delete 等动作不会触发公司元数据写入。"""

    repo = _RepositoryStub()
    module.upsert_company_meta_for_upload(
        repository=repo,
        ticker="AAPL",
        action="delete",
        company_id=None,
        company_name=None,
    )

    assert repo.captured is None


@pytest.mark.unit
def test_upsert_company_meta_requires_fields_when_meta_missing() -> None:
    """验证仓储中缺少 meta 时，create/update 仍要求公司名称。"""

    repo = _RepositoryStub()

    with pytest.raises(ValueError, match="--company-name"):
        module.upsert_company_meta_for_upload(
            repository=repo,
            ticker="AAPL",
            action="update",
            company_id="320193",
            company_name="",
        )


@pytest.mark.unit
def test_upsert_company_meta_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证仓储中缺少 meta 时，create/update 会写入新的 company meta。"""

    repo = _RepositoryStub()
    monkeypatch.setattr(
        module,
        "normalize_ticker",
        lambda ticker: NormalizedTicker(canonical=ticker.strip().upper(), market="US", exchange=None, raw=ticker),
    )
    monkeypatch.setattr(module, "now_iso8601", lambda: "2026-03-02T00:00:00+00:00")

    module.upsert_company_meta_for_upload(
        repository=repo,
        ticker=" aapl ",
        action=" Update ",
        company_id=" 320193 ",
        company_name=" Apple Inc. ",
        ticker_aliases=["AAPL", "APC"],
    )

    assert repo.captured is not None
    assert repo.captured.company_id == "AAPL_US"
    assert repo.captured.company_name == "Apple Inc."
    assert repo.captured.ticker == "AAPL"
    assert repo.captured.market == "US"
    assert repo.captured.resolver_version == module.RESOLVER_VERSION
    assert repo.captured.ticker_aliases == ["AAPL", "APC"]


@pytest.mark.unit
def test_upsert_company_meta_ignores_passed_fields_when_meta_exists(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """验证仓储中已存在 meta 时，重复传入的 company meta 会被忽略并告警。"""

    repo = _RepositoryStub(
        existing=CompanyMeta(
            company_id="1754581",
            company_name="Futu Holdings Ltd",
            ticker="FUTU",
            market="US",
            resolver_version="market_resolver_v1",
            updated_at="2026-03-13T00:00:00+00:00",
        )
    )

    with caplog.at_level(logging.WARNING):
        module.upsert_company_meta_for_upload(
            repository=repo,
            ticker="FUTU",
            action="create",
            company_id="1754582",
            company_name="Futu Holdings Changed",
            ticker_aliases=["FUTU", "3588.HK"],
        )

    assert repo.captured is None
    assert "忽略本次上传传入的 company_id/company_name" in caplog.text
