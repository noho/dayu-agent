"""Deterministic research-template routing from local company facets."""

from __future__ import annotations

import json
from pathlib import Path

from dayu.services.internal.write_pipeline.models import CompanyFacetProfile

TEMPLATE_PRIORITY = {
    "financial": 0,
    "consumer": 1,
    "cyclical": 2,
    "technology": 3,
    "common": 4,
}

TEMPLATE_FACET_RULES: dict[str, frozenset[str]] = {
    "consumer": frozenset(
        {
            "消费品牌",
            "零售渠道/连锁",
            "品牌效应明显",
            "规模效应显著",
            "有明显定价权",
            "高营销费用驱动",
            "硬件/消费电子",
            "酒店/旅游服务",
        }
    ),
    "cyclical": frozenset(
        {
            "上游资源/勘探开发",
            "能源设备/服务",
            "大宗材料/基础化工",
            "商品价格敏感",
            "周期性强",
            "航空/航运/出行服务",
            "物流网络/快递",
            "公用事业",
            "REIT/基础设施",
            "利用率敏感",
            "高资本开支",
        }
    ),
    "technology": frozenset(
        {
            "平台互联网",
            "电商/交易平台",
            "广告媒体",
            "内容/娱乐平台",
            "游戏/互动娱乐",
            "企业软件",
            "垂直软件/创意软件",
            "数据基础设施/数据中心",
            "半导体设计",
            "半导体设备/制造",
            "整车制造",
            "汽车零部件",
            "动力电池/关键部件",
            "工业制造/关键部件",
            "高SBC",
            "高研发驱动",
        }
    ),
    "financial": frozenset(
        {
            "支付/金融基础设施",
            "交易所/市场基础设施",
            "资产管理/财富管理",
            "银行",
            "消费金融/信贷",
            "保险",
            "利率敏感",
            "监管敏感",
            "许可/牌照依赖",
            "高负债/融资依赖",
        }
    ),
}


def load_company_facets_from_manifest(path: Path) -> CompanyFacetProfile:
    """Load a company facet profile from a write manifest or facet object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must contain an object: {path}")
    raw_facets = payload.get("company_facets", payload)
    if not isinstance(raw_facets, dict):
        raise ValueError(f"manifest does not contain company_facets object: {path}")
    try:
        return CompanyFacetProfile.from_dict(raw_facets)
    except TypeError as exc:
        raise ValueError(f"manifest contains invalid company_facets: {path}: {exc}") from exc


def recommend_research_template_name(company_facets: CompanyFacetProfile) -> str:
    """Return the highest-scoring deterministic research-template name."""

    scores: list[tuple[int, int, str]] = []
    primary_facets = tuple(company_facets.primary_facets)
    constraint_facets = tuple(company_facets.cross_cutting_facets)
    for name, rule_facets in TEMPLATE_FACET_RULES.items():
        score = sum(3 for facet in primary_facets if facet in rule_facets)
        score += sum(1 for facet in constraint_facets if facet in rule_facets)
        if score > 0:
            scores.append((-score, TEMPLATE_PRIORITY.get(name, 99), name))
    if not scores:
        return "common"
    scores.sort()
    return scores[0][2]
