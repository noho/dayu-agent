"""FMP 公司 alias 推断器测试。"""

from __future__ import annotations

import pytest

from dayu.fins.resolver import fmp_company_alias_resolver as module


def test_infer_company_aliases_from_fmp_combines_symbol_and_name_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证两跳式 infer 会合并 symbol 搜索与 name 搜索的同名结果。"""

    monkeypatch.setenv("FMP_API_KEY", "demo")

    def _fake_fetch(
        *,
        endpoint: str,
        query: str,
        limit: int,
        api_key: str,
    ) -> list[dict[str, str]]:
        del limit, api_key
        if endpoint == "search-symbol":
            assert query == "BABA"
            return [
                {"symbol": "BABA", "name": "Alibaba Group Holding Limited"},
                {"symbol": "BABAF", "name": "Alibaba Group Holding Limited"},
                {"symbol": "BABA.BO", "name": "Baba Arts Limited"},
            ]
        assert endpoint == "search-name"
        assert query == "Alibaba Group Holding Limited"
        return [
            {"symbol": "9988.HK", "name": "Alibaba Group Holding Limited"},
            {"symbol": "89988.HK", "name": "Alibaba Group Holding Limited"},
            {"symbol": "BABA.BO", "name": "Baba Arts Limited"},
        ]

    monkeypatch.setattr(module, "_fetch_fmp_search_results", _fake_fetch)

    result = module.infer_company_aliases_from_fmp("baba")

    assert result.canonical_ticker == "BABA"
    assert result.company_name == "Alibaba Group Holding Limited"
    assert result.ticker_aliases == ["BABA", "BABAF", "9988", "89988"]


def test_infer_company_aliases_from_fmp_uses_first_result_when_no_exact_symbol_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 search-symbol 无精确 symbol 时会回退到第一条结果。"""

    monkeypatch.setenv("FMP_API_KEY", "demo")

    def _fake_fetch(
        *,
        endpoint: str,
        query: str,
        limit: int,
        api_key: str,
    ) -> list[dict[str, str]]:
        del query, limit, api_key
        if endpoint == "search-symbol":
            return [{"symbol": "9988.HK", "name": "Alibaba Group Holding Limited"}]
        return [{"symbol": "89988.HK", "name": "Alibaba Group Holding Limited"}]

    monkeypatch.setattr(module, "_fetch_fmp_search_results", _fake_fetch)

    result = module.infer_company_aliases_from_fmp("9988")

    assert result.company_name == "Alibaba Group Holding Limited"
    assert result.ticker_aliases == ["9988", "89988"]


def test_infer_company_aliases_from_fmp_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证缺少 `FMP_API_KEY` 时抛出明确异常。"""

    monkeypatch.delenv("FMP_API_KEY", raising=False)

    with pytest.raises(module.FmpAliasInferenceError, match="FMP_API_KEY"):
        module.infer_company_aliases_from_fmp("BABA")
