"""``dayu/fins/pipelines/cn_download_models.py`` 单元测试。

仅验证 typed model / 字面量 / 版本常量层面的契约；本模块禁止 IO。
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest

from dayu.fins.pipelines.cn_download_models import (
    CN_PIPELINE_DOWNLOAD_VERSION,
    CnCompanyProfile,
    CnFilingStage,
    CnFiscalPeriod,
    CnLanguage,
    CnMarketKind,
    CnReportCandidate,
    CnReportQuery,
    CnSourceProvider,
    DownloadedReportAsset,
)


def test_cn_market_kind_only_covers_cn_and_hk() -> None:
    """``CnMarketKind`` 必须只覆盖 CN/HK，不允许 US 漏入下载链路。"""

    assert set(get_args(CnMarketKind)) == {"CN", "HK"}


def test_cn_fiscal_period_includes_q2_for_input_normalization() -> None:
    """``CnFiscalPeriod`` 字面量集合需包含 ``Q2``，供 form 解析阶段输入归一。

    虽然 Q2 在 form 解析阶段会被归一为 H1，但保留 Q2 字面量可让"输入侧 Q2"与
    "candidate 侧 Q2"在静态类型层面共享同一字面量集合。
    """

    expected = {"FY", "H1", "Q1", "Q2", "Q3"}
    assert set(get_args(CnFiscalPeriod)) == expected


def test_cn_source_provider_locked_to_cninfo_and_hkexnews() -> None:
    """provider 字面量被锁定为 cninfo / hkexnews，禁止漂移。"""

    assert set(get_args(CnSourceProvider)) == {"cninfo", "hkexnews"}


def test_cn_filing_stage_state_machine_is_strict() -> None:
    """``CnFilingStage`` 字面量必须严格覆盖 4 个阶段。"""

    assert set(get_args(CnFilingStage)) == {
        "remote_selected",
        "pdf_downloaded",
        "docling_converted",
        "source_committed",
    }


def test_cn_language_locked_to_zh_and_en() -> None:
    """语言字面量锁定为 ``zh`` / ``en``。"""

    assert set(get_args(CnLanguage)) == {"zh", "en"}


def test_download_version_constant_format() -> None:
    """版本常量需符合 ``cn_pipeline_download_v{semver}`` 形态。"""

    assert CN_PIPELINE_DOWNLOAD_VERSION.startswith("cn_pipeline_download_v")
    suffix = CN_PIPELINE_DOWNLOAD_VERSION[len("cn_pipeline_download_v") :]
    parts = suffix.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit(), f"semver 段必须为数字: {suffix!r}"


def test_cn_company_profile_is_frozen() -> None:
    """``CnCompanyProfile`` 是 frozen dataclass，禁止运行期改字段。"""

    profile = CnCompanyProfile(
        provider="cninfo",
        company_id="CNINFO:9900007792",
        company_name="贵州茅台",
        ticker="600519",
    )
    with pytest.raises(Exception):  # FrozenInstanceError 是 Exception 子类
        profile.company_id = "CNINFO:other"  # pyright: ignore[reportAttributeAccessIssue]


def test_cn_report_query_carries_target_periods_tuple() -> None:
    """``CnReportQuery.target_periods`` 必须是 tuple，不接受 list。"""

    query = CnReportQuery(
        market="CN",
        normalized_ticker="600519",
        start_date="2020-01-01",
        end_date="2025-12-31",
        target_periods=("FY", "H1"),
    )
    assert isinstance(query.target_periods, tuple)
    assert query.target_periods == ("FY", "H1")


def test_cn_report_candidate_records_remote_fingerprint_inputs() -> None:
    """candidate 必须携带 fingerprint 计算所需的 ``content_length``/``etag``/``last_modified``。"""

    candidate = CnReportCandidate(
        provider="cninfo",
        source_id="1219470830",
        source_url="https://static.cninfo.com.cn/example.PDF",
        title="贵州茅台2024年年度报告",
        language="zh",
        filing_date="2025-04-03",
        fiscal_year=2024,
        fiscal_period="FY",
        amended=False,
        content_length=12345,
        etag='"abc"',
        last_modified="Wed, 03 Apr 2025 02:00:00 GMT",
    )
    assert candidate.fiscal_period == "FY"
    assert candidate.content_length == 12345
    assert candidate.etag == '"abc"'
    assert candidate.last_modified is not None


def test_downloaded_report_asset_carries_pdf_path_and_sha256() -> None:
    """``DownloadedReportAsset`` 字段对齐 PDF skip / staging 复用所需。"""

    candidate = CnReportCandidate(
        provider="hkexnews",
        source_id="2024090100123",
        source_url="https://www1.hkexnews.hk/listedco/listconews/sehk/2024/example.pdf",
        title="2024 Interim Report",
        language="en",
        filing_date="2024-09-01",
        fiscal_year=2024,
        fiscal_period="H1",
        amended=False,
        content_length=99,
        etag=None,
        last_modified=None,
    )
    asset = DownloadedReportAsset(
        candidate=candidate,
        pdf_path=Path("/tmp/example.pdf"),
        sha256="0" * 64,
        content_length=99,
        downloaded_at="2025-05-02T00:00:00+00:00",
    )
    assert asset.candidate is candidate
    assert asset.sha256 == "0" * 64
    assert asset.content_length == 99


def test_cn_company_profile_supports_hk_prefix() -> None:
    """``company_id`` 必须支持 ``HKEX:`` 前缀（不在 model 内强校验，但允许）。"""

    profile = CnCompanyProfile(
        provider="hkexnews",
        company_id="HKEX:7609",
        company_name="腾讯控股",
        ticker="0700",
    )
    assert profile.company_id.startswith("HKEX:")
    assert profile.provider == "hkexnews"
