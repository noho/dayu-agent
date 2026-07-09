"""CLI helpers for local research template assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Iterable
from pathlib import Path

from dayu.cli.dependency_setup import setup_loglevel
from dayu.cli.research_template_assets import build_composed_research_template_text
from dayu.cli.research_template_routing import (
    TEMPLATE_FACET_RULES,
    TEMPLATE_PRIORITY,
    load_company_facets_from_manifest,
)
from dayu.cli.commands.research_workbook import (
    build_research_workbook_payload,
    build_research_workbook_report,
    build_research_workbook_report_status_snapshot,
    build_research_workbook_rollback_preview,
    build_research_workbook_status_snapshot,
    build_research_workbook_update_preview,
    discover_research_workbook_reports,
    discover_research_workbooks,
    inspect_research_workbook,
    inspect_research_workbook_report,
    validate_research_workbook_payload,
    write_research_workbook_payload,
    write_research_workbook_report,
    write_research_workbook_report_status_snapshot,
    write_research_workbook_rollback,
    write_research_workbook_status_snapshot,
    write_research_workbook_update,
)
from dayu.services.internal.write_pipeline.models import CompanyFacetProfile
from dayu.startup.config_file_resolver import resolve_package_assets_path

_TEMPLATE_DIR_NAME = "research_templates"
_TEMPLATE_SUFFIX = ".md"
_FALLBACK_TEMPLATE_NAME = "common"
_BUNDLE_ARTIFACT_KEYS = (
    "write_template",
    "research_workbook",
    "research_progress_report",
    "monitoring_rules",
    "source_map",
    "package_manifest",
    "usage_guide",
)
_COMMON_DATA_SOURCE_CANDIDATES = (
    "financial_statements",
    "company_filings",
    "market_data",
)
_TEMPLATE_DATA_SOURCE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "common": _COMMON_DATA_SOURCE_CANDIDATES,
    "consumer": (
        "financial_statements",
        "company_filings",
        "channel_checks",
        "market_data",
    ),
    "cyclical": (
        "financial_statements",
        "industry_price_data",
        "inventory_supply_data",
        "market_data",
    ),
    "technology": (
        "financial_statements",
        "operating_metrics",
        "product_release_notes",
        "market_data",
    ),
    "financial": (
        "financial_statements",
        "regulatory_filings",
        "capital_adequacy_data",
        "market_data",
    ),
}
_DATA_SOURCE_BINDING_CANDIDATES: dict[str, dict[str, object]] = {
    "financial_statements": {
        "provider_type": "dayu_fins_tool",
        "candidate_tools": ["get_financial_statement", "query_xbrl_facts", "get_table"],
        "candidate_fields": [
            "revenue",
            "gross_profit",
            "operating_profit",
            "net_income",
            "operating_cash_flow",
        ],
    },
    "company_filings": {
        "provider_type": "dayu_fins_tool",
        "candidate_tools": ["list_documents", "search_document", "read_section"],
        "candidate_fields": ["document_type", "section_title", "filing_date", "matched_text"],
    },
    "market_data": {
        "provider_type": "external_market_data_placeholder",
        "candidate_tools": [],
        "candidate_fields": ["price", "market_cap", "pe_ttm", "pb", "dividend_yield", "turnover"],
    },
    "channel_checks": {
        "provider_type": "manual_or_external_placeholder",
        "candidate_tools": [],
        "candidate_fields": ["same_store_sales", "channel_inventory", "store_count", "customer_frequency"],
    },
    "industry_price_data": {
        "provider_type": "external_industry_data_placeholder",
        "candidate_tools": [],
        "candidate_fields": ["spot_price", "spread", "freight_rate", "utilization_rate"],
    },
    "inventory_supply_data": {
        "provider_type": "external_industry_data_placeholder",
        "candidate_tools": [],
        "candidate_fields": ["inventory_days", "orderbook", "capacity_additions", "operating_rate"],
    },
    "operating_metrics": {
        "provider_type": "company_filings_or_manual_placeholder",
        "candidate_tools": ["search_document", "read_section"],
        "candidate_fields": ["mau", "dau", "arr", "nrr", "churn", "gross_margin"],
    },
    "product_release_notes": {
        "provider_type": "company_filings_or_web_placeholder",
        "candidate_tools": ["search_document", "read_section"],
        "candidate_fields": ["product_launch", "customer_case", "release_date", "management_commentary"],
    },
    "regulatory_filings": {
        "provider_type": "dayu_fins_tool",
        "candidate_tools": ["list_documents", "search_document", "read_section"],
        "candidate_fields": ["capital_ratio", "npl_ratio", "solvency_ratio", "regulatory_disclosure"],
    },
    "capital_adequacy_data": {
        "provider_type": "dayu_fins_tool",
        "candidate_tools": ["get_financial_statement", "search_document", "read_section"],
        "candidate_fields": ["cet1_ratio", "capital_adequacy_ratio", "leverage_ratio", "provision_coverage"],
    },
}


@dataclass(frozen=True)
class ResearchTemplate:
    """Stable view of a packaged research template."""

    name: str
    title: str
    path: Path


@dataclass(frozen=True)
class ResearchTemplateRecommendation:
    """Deterministic routing result for one research template."""

    name: str
    title: str
    score: int
    matched_facets: tuple[str, ...]
    reason: str


def run_research_template_command(args: argparse.Namespace) -> int:
    """Dispatch ``research-template`` subcommands."""

    setup_loglevel(args)
    action = str(getattr(args, "research_template_action", "") or "").strip().lower()
    try:
        if action == "list":
            return _run_list(args)
        if action == "show":
            return _run_show(args)
        if action == "copy":
            return _run_copy(args)
        if action == "recommend":
            return _run_recommend(args)
        if action == "compose":
            return _run_compose(args)
        if action == "monitoring-rules":
            return _run_monitoring_rules(args)
        if action == "research-workbook":
            return _run_research_workbook(args)
        if action == "validate-research-workbook":
            return _run_validate_research_workbook(args)
        if action == "update-research-workbook":
            return _run_update_research_workbook(args)
        if action == "rollback-research-workbook":
            return _run_rollback_research_workbook(args)
        if action == "source-map":
            return _run_source_map(args)
        if action == "validate-source-map":
            return _run_validate_source_map(args)
        if action == "package-manifest":
            return _run_package_manifest(args)
        if action == "materialize":
            return _run_materialize(args)
        if action == "refresh-workspace":
            return _run_refresh_workspace(args)
        if action == "list-bundles":
            return _run_list_bundles(args)
        if action == "validate-bundle":
            return _run_validate_bundle(args)
        if action == "rebind-bundle":
            return _run_rebind_bundle(args)
        if action == "rollback-bundle-rebind":
            return _run_rollback_bundle_rebind(args)
        if action == "monitoring-plan":
            return _run_monitoring_plan(args)
        if action == "validate-monitoring-plan":
            return _run_validate_monitoring_plan(args)
        if action == "list-monitoring-plans":
            return _run_list_monitoring_plans(args)
        if action == "monitoring-status":
            return _run_monitoring_status(args)
        if action == "workbook-status":
            return _run_workbook_status(args)
        if action == "workbook-report":
            return _run_workbook_report(args)
        if action == "validate-workbook-report":
            return _run_validate_workbook_report(args)
        if action == "workbook-report-status":
            return _run_workbook_report_status(args)
        if action == "materialize-portfolio":
            return _run_materialize_portfolio(args)
        if action == "preview-portfolio":
            return _run_preview_portfolio(args)
        if action == "scheduler-manifest":
            return _run_scheduler_manifest(args)
        if action == "validate-scheduler-manifest":
            return _run_validate_scheduler_manifest(args)
        if action == "source-bindings":
            return _run_source_bindings(args)
        if action == "rollback-source-bindings":
            return _run_rollback_source_bindings(args)
        if action == "source-binding-history":
            return _run_source_binding_history(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"research-template error: {exc}", file=sys.stderr)
        return 1
    return 1


def list_research_templates() -> tuple[ResearchTemplate, ...]:
    """Return all packaged research templates except the directory README."""

    template_dir = _resolve_template_dir()
    templates = []
    for path in sorted(template_dir.glob(f"*{_TEMPLATE_SUFFIX}")):
        name = path.stem
        if name.lower() == "readme":
            continue
        templates.append(ResearchTemplate(name=name, title=_read_template_title(path), path=path))
    return tuple(templates)


def load_research_template(name: str) -> str:
    """Load a packaged research template by stable name."""

    return _resolve_template_path(name).read_text(encoding="utf-8")


def copy_research_template(
    name: str,
    *,
    workspace_root: Path,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Copy a packaged template into the workspace assets directory."""

    source_path = _resolve_template_path(name)
    target_path = output_path or workspace_root / "assets" / _TEMPLATE_DIR_NAME / source_path.name
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    return target_path


def compose_research_template(
    name: str,
    *,
    workspace_root: Path,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Compose the common template with one industry template."""

    normalized = _normalize_template_name(name)
    composed_text = build_composed_research_template_text(normalized)
    default_name = "common.write.md" if normalized == _FALLBACK_TEMPLATE_NAME else f"common-plus-{normalized}.md"
    target_path = output_path or workspace_root / "assets" / _TEMPLATE_DIR_NAME / default_name
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(composed_text, encoding="utf-8", newline="\n")
    return target_path


def recommend_research_templates(
    company_facets: CompanyFacetProfile | None,
    *,
    limit: int | None = None,
) -> tuple[ResearchTemplateRecommendation, ...]:
    """Recommend research templates from company facet tags."""

    templates_by_name = {template.name: template for template in list_research_templates()}
    primary_facets = tuple(company_facets.primary_facets) if company_facets is not None else ()
    constraint_facets = tuple(company_facets.cross_cutting_facets) if company_facets is not None else ()
    recommendations: list[ResearchTemplateRecommendation] = []
    for name, rule_facets in TEMPLATE_FACET_RULES.items():
        template = templates_by_name.get(name)
        if template is None:
            continue
        primary_matches = tuple(facet for facet in primary_facets if facet in rule_facets)
        constraint_matches = tuple(facet for facet in constraint_facets if facet in rule_facets)
        score = len(primary_matches) * 3 + len(constraint_matches)
        matched = _dedupe([*primary_matches, *constraint_matches])
        if score <= 0:
            continue
        recommendations.append(
            ResearchTemplateRecommendation(
                name=name,
                title=template.title,
                score=score,
                matched_facets=matched,
                reason=f"matched facets: {', '.join(matched)}",
            )
        )
    if not recommendations:
        fallback = templates_by_name[_FALLBACK_TEMPLATE_NAME]
        recommendations.append(
            ResearchTemplateRecommendation(
                name=fallback.name,
                title=fallback.title,
                score=0,
                matched_facets=(),
                reason="no industry-specific facet matched; using common template",
            )
        )
    recommendations.sort(key=lambda item: (-item.score, TEMPLATE_PRIORITY.get(item.name, 99), item.name))
    if limit is not None:
        return tuple(recommendations[: max(limit, 0)])
    return tuple(recommendations)


def extract_monitoring_variables(name: str) -> tuple[str, ...]:
    """Extract bullet items from the template monitoring-variable section."""

    content = load_research_template(name)
    variables: list[str] = []
    in_section = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") and "监控变量" in stripped:
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if not in_section or not stripped.startswith("- "):
            continue
        variables.append(stripped.removeprefix("- ").strip())
    return _dedupe(variables)


def build_monitoring_rules_payload(name: str) -> dict[str, object]:
    """Build a local monitoring-rule draft from one research template."""

    normalized = _normalize_template_name(name)
    template = _resolve_template_path(normalized)
    variables = extract_monitoring_variables(normalized)
    data_source_candidates = list(get_monitoring_data_source_candidates(normalized))
    return {
        "schema_version": 1,
        "template": normalized,
        "source_template_file": str(template),
        "rule_type": "manual_review",
        "cadence": "quarterly",
        "data_source_candidates": data_source_candidates,
        "variables": [
            {
                "name": variable,
                "source_section": "监控变量",
                "evidence_required": True,
                "data_source_candidates": data_source_candidates,
                "binding_status": "unbound",
                "notes": "Draft rule extracted from research template; bind to concrete data source before automation.",
            }
            for variable in variables
        ],
    }


def build_monitoring_source_map_payload(name: str) -> dict[str, object]:
    """Build a local source-map draft for monitoring-rule data bindings."""

    normalized = _normalize_template_name(name)
    candidates = get_monitoring_data_source_candidates(normalized)
    return {
        "schema_version": 1,
        "template": normalized,
        "source_map_type": "monitoring_data_binding_draft",
        "binding_status": "unbound",
        "data_sources": [
            {
                "source": source,
                "binding_status": "unbound",
                **_DATA_SOURCE_BINDING_CANDIDATES.get(
                    source,
                    {
                        "provider_type": "unknown_placeholder",
                        "candidate_tools": [],
                        "candidate_fields": [],
                    },
                ),
                "notes": "Draft source binding only; verify concrete provider fields before automation.",
            }
            for source in candidates
        ],
    }


def validate_monitoring_source_map_payload(
    rules_payload: dict[str, object],
    source_map_payload: dict[str, object],
) -> dict[str, object]:
    """Validate consistency between monitoring rules and source-map drafts."""

    errors: list[str] = []
    warnings: list[str] = []
    rules_template = str(rules_payload.get("template", "") or "")
    source_map_template = str(source_map_payload.get("template", "") or "")
    if rules_template != source_map_template:
        errors.append(f"template mismatch: rules={rules_template!r} source_map={source_map_template!r}")

    rules_sources = set(_as_str_list(rules_payload.get("data_source_candidates")))
    variable_sources: set[str] = set()
    variables = rules_payload.get("variables", [])
    if isinstance(variables, list):
        for index, variable in enumerate(variables):
            if not isinstance(variable, dict):
                errors.append(f"variables[{index}] must be an object")
                continue
            variable_sources.update(_as_str_list(variable.get("data_source_candidates")))
            binding_status = str(variable.get("binding_status", "") or "")
            if binding_status not in {"unbound", "bound"}:
                errors.append(f"variables[{index}].binding_status must be unbound or bound")
    else:
        errors.append("variables must be a list")

    data_sources = source_map_payload.get("data_sources", [])
    mapped_sources: set[str] = set()
    if isinstance(data_sources, list):
        for index, source in enumerate(data_sources):
            if not isinstance(source, dict):
                errors.append(f"data_sources[{index}] must be an object")
                continue
            source_name = str(source.get("source", "") or "")
            if source_name:
                mapped_sources.add(source_name)
            binding_status = str(source.get("binding_status", "") or "")
            if binding_status not in {"unbound", "bound"}:
                errors.append(f"data_sources[{index}].binding_status must be unbound or bound")
    else:
        errors.append("data_sources must be a list")

    required_sources = rules_sources | variable_sources
    missing_sources = sorted(required_sources - mapped_sources)
    extra_sources = sorted(mapped_sources - required_sources)
    if missing_sources:
        errors.append(f"missing source-map entries: {', '.join(missing_sources)}")
    if extra_sources:
        warnings.append(f"unused source-map entries: {', '.join(extra_sources)}")
    return {
        "ok": not errors,
        "template": rules_template or source_map_template,
        "errors": errors,
        "warnings": warnings,
        "rules_source_count": len(required_sources),
        "source_map_entry_count": len(mapped_sources),
    }


def build_monitoring_source_binding_preview(
    source_map_payload: dict[str, object],
    approval_payload: dict[str, object],
) -> dict[str, object]:
    """Validate and preview explicit bindings for supported Dayu data sources."""

    if approval_payload.get("schema_version") != 1:
        raise ValueError("binding approval schema_version must be 1")
    if approval_payload.get("approval_type") != "research_monitoring_source_binding":
        raise ValueError("approval_type must be research_monitoring_source_binding")
    source_template = str(source_map_payload.get("template", "") or "")
    approval_template = str(approval_payload.get("template", "") or "")
    if not approval_template or approval_template != source_template:
        raise ValueError(
            f"binding approval template mismatch: source_map={source_template!r} approval={approval_template!r}"
        )
    approved_by = str(approval_payload.get("approved_by", "") or "").strip()
    approval_reference = str(approval_payload.get("approval_reference", "") or "").strip()
    if not approved_by:
        raise ValueError("binding approval approved_by is required")
    if not approval_reference:
        raise ValueError("binding approval approval_reference is required")
    bindings = approval_payload.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("binding approval bindings must be a non-empty list")

    updated = deepcopy(source_map_payload)
    data_sources = updated.get("data_sources")
    if not isinstance(data_sources, list):
        raise ValueError("source-map data_sources must be a list")
    sources_by_name = {
        str(source.get("source", "") or ""): source
        for source in data_sources
        if isinstance(source, dict) and str(source.get("source", "") or "")
    }
    changed_sources: list[str] = []
    seen_sources: set[str] = set()
    for index, binding in enumerate(bindings):
        if not isinstance(binding, dict):
            raise ValueError(f"binding approval bindings[{index}] must be an object")
        source_name = str(binding.get("source", "") or "").strip()
        if not source_name or source_name not in sources_by_name:
            raise ValueError(f"binding approval bindings[{index}].source is unknown: {source_name!r}")
        if source_name in seen_sources:
            raise ValueError(f"duplicate binding approval source: {source_name}")
        seen_sources.add(source_name)
        source = sources_by_name[source_name]
        if source.get("provider_type") != "dayu_fins_tool":
            raise ValueError(f"source {source_name!r} is not an implemented dayu_fins_tool provider")
        selected_tool = str(binding.get("selected_tool", "") or "").strip()
        candidate_tools = _as_str_list(source.get("candidate_tools"))
        if selected_tool not in candidate_tools:
            raise ValueError(f"source {source_name!r} selected_tool is not a declared candidate: {selected_tool!r}")
        selected_fields = _dedupe(_as_str_list(binding.get("selected_fields")))
        if not selected_fields:
            raise ValueError(f"source {source_name!r} selected_fields must be non-empty")
        candidate_fields = set(_as_str_list(source.get("candidate_fields")))
        unknown_fields = [field for field in selected_fields if field not in candidate_fields]
        if unknown_fields:
            raise ValueError(f"source {source_name!r} selected_fields are not declared candidates: {unknown_fields}")
        source["binding_status"] = "bound"
        source["selected_tool"] = selected_tool
        source["selected_fields"] = list(selected_fields)
        source["binding_approval"] = {
            "approved_by": approved_by,
            "approval_reference": approval_reference,
        }
        changed_sources.append(source_name)

    bound_count = sum(isinstance(source, dict) and source.get("binding_status") == "bound" for source in data_sources)
    if bound_count == len(data_sources):
        overall_status = "bound"
    elif bound_count:
        overall_status = "partially_bound"
    else:
        overall_status = "unbound"
    updated["binding_status"] = overall_status
    return {
        "schema_version": 1,
        "preview_type": "research_monitoring_source_binding",
        "template": source_template,
        "approved_by": approved_by,
        "approval_reference": approval_reference,
        "changed_source_count": len(changed_sources),
        "changed_sources": changed_sources,
        "resulting_binding_status": overall_status,
        "source_map": updated,
    }


def write_monitoring_source_binding_approval(
    source_map_path: Path,
    approval_path: Path,
) -> dict[str, object]:
    """Apply approved Dayu source bindings in place after writing an immutable backup."""

    resolved_source_map = source_map_path.resolve()
    resolved_approval = approval_path.resolve()
    source_map_payload = _load_json_object(resolved_source_map)
    approval_payload = _load_json_object(resolved_approval)
    preview = build_monitoring_source_binding_preview(source_map_payload, approval_payload)
    original_fingerprint = _sha256_file(resolved_source_map)
    backup_path = resolved_source_map.with_name(
        f"{resolved_source_map.stem}.before-bindings.{original_fingerprint[:12]}.json"
    )
    if backup_path.exists():
        if not backup_path.is_file() or _sha256_file(backup_path) != original_fingerprint:
            raise FileExistsError(f"binding backup path is occupied by different content: {backup_path}")
    else:
        backup_path.write_bytes(resolved_source_map.read_bytes())
    updated_source_map = preview.get("source_map")
    if not isinstance(updated_source_map, dict):
        raise ValueError("binding preview did not produce a source-map object")
    resolved_source_map.write_text(
        json.dumps(updated_source_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "source_map_file": str(resolved_source_map),
        "backup_file": str(backup_path),
        "approval_file": str(resolved_approval),
        "source_map_fingerprint_before": original_fingerprint,
        "source_map_fingerprint_after": _sha256_file(resolved_source_map),
        "preview": preview,
    }


def build_monitoring_source_binding_rollback_preview(
    source_map_path: Path,
    backup_path: Path,
) -> dict[str, object]:
    """Validate and preview restoring one content-addressed binding backup."""

    resolved_source_map = source_map_path.resolve()
    resolved_backup = backup_path.resolve()
    if resolved_source_map == resolved_backup:
        raise ValueError("source-map and binding backup must be different files")
    if resolved_source_map.parent != resolved_backup.parent:
        raise ValueError("binding backup must be in the same directory as source-map")

    current_payload = _load_json_object(resolved_source_map)
    backup_payload = _load_json_object(resolved_backup)
    current_template = str(current_payload.get("template", "") or "").strip()
    backup_template = str(backup_payload.get("template", "") or "").strip()
    if not current_template or current_template != backup_template:
        raise ValueError(
            f"binding rollback template mismatch: source_map={current_template!r} backup={backup_template!r}"
        )

    backup_fingerprint = _sha256_file(resolved_backup)
    trusted_snapshot_names = {
        "before_bindings": f"{resolved_source_map.stem}.before-bindings.{backup_fingerprint[:12]}.json",
        "before_rollback": f"{resolved_source_map.stem}.before-rollback.{backup_fingerprint[:12]}.json",
    }
    matching_snapshot_types = [
        snapshot_type
        for snapshot_type, expected_name in trusted_snapshot_names.items()
        if resolved_backup.name == expected_name
    ]
    if not matching_snapshot_types:
        raise ValueError(
            "binding backup filename does not match a trusted source-map snapshot and content fingerprint: "
            f"expected one of {sorted(trusted_snapshot_names.values())!r}"
        )
    snapshot_type = matching_snapshot_types[0]

    current_sources = _source_map_sources_by_name(current_payload, label="source-map")
    backup_sources = _source_map_sources_by_name(backup_payload, label="binding backup")
    current_source_names = set(current_sources)
    backup_source_names = set(backup_sources)
    if current_source_names != backup_source_names:
        missing = sorted(current_source_names - backup_source_names)
        extra = sorted(backup_source_names - current_source_names)
        raise ValueError(f"binding rollback source set mismatch: missing={missing} extra={extra}")

    changed_sources = sorted(
        source_name
        for source_name in current_source_names
        if current_sources[source_name] != backup_sources[source_name]
    )
    return {
        "schema_version": 1,
        "preview_type": "research_monitoring_source_binding_rollback",
        "template": current_template,
        "source_map_file": str(resolved_source_map),
        "binding_backup_file": str(resolved_backup),
        "snapshot_type": snapshot_type,
        "source_map_fingerprint_before": _sha256_file(resolved_source_map),
        "source_map_fingerprint_after": backup_fingerprint,
        "binding_status_before": str(current_payload.get("binding_status", "") or ""),
        "binding_status_after": str(backup_payload.get("binding_status", "") or ""),
        "changed_source_count": len(changed_sources),
        "changed_sources": changed_sources,
        "write_required": True,
    }


def write_monitoring_source_binding_rollback(
    source_map_path: Path,
    backup_path: Path,
) -> dict[str, object]:
    """Restore a binding backup after preserving the current source-map immutably."""

    preview = build_monitoring_source_binding_rollback_preview(source_map_path, backup_path)
    resolved_source_map = Path(str(preview["source_map_file"]))
    resolved_backup = Path(str(preview["binding_backup_file"]))
    current_fingerprint = str(preview["source_map_fingerprint_before"])
    rollback_backup_path = resolved_source_map.with_name(
        f"{resolved_source_map.stem}.before-rollback.{current_fingerprint[:12]}.json"
    )
    if rollback_backup_path.exists():
        if not rollback_backup_path.is_file() or _sha256_file(rollback_backup_path) != current_fingerprint:
            raise FileExistsError(f"rollback backup path is occupied by different content: {rollback_backup_path}")
    else:
        rollback_backup_path.write_bytes(resolved_source_map.read_bytes())

    resolved_source_map.write_bytes(resolved_backup.read_bytes())
    restored_fingerprint = _sha256_file(resolved_source_map)
    expected_fingerprint = str(preview["source_map_fingerprint_after"])
    if restored_fingerprint != expected_fingerprint:
        raise ValueError("restored source-map fingerprint does not match binding backup")
    return {
        "source_map_file": str(resolved_source_map),
        "binding_backup_file": str(resolved_backup),
        "rollback_backup_file": str(rollback_backup_path),
        "source_map_fingerprint_before": current_fingerprint,
        "source_map_fingerprint_after": restored_fingerprint,
        "preview": preview,
    }


def inspect_monitoring_source_binding_history(source_map_path: Path) -> dict[str, object]:
    """Inspect content-addressed binding snapshots beside one source-map."""

    resolved_source_map = source_map_path.resolve()
    current_payload = _load_json_object(resolved_source_map)
    template = str(current_payload.get("template", "") or "").strip()
    if not template:
        raise ValueError("source-map template is required")
    current_sources = _source_map_sources_by_name(current_payload, label="source-map")
    snapshot_specs = (
        ("before_bindings", "before-bindings"),
        ("before_rollback", "before-rollback"),
    )
    snapshots: list[dict[str, object]] = []
    for snapshot_type, filename_marker in snapshot_specs:
        pattern = f"{resolved_source_map.stem}.{filename_marker}.*.json"
        for snapshot_path in sorted(resolved_source_map.parent.glob(pattern), key=str):
            snapshots.append(
                _inspect_source_binding_snapshot(
                    snapshot_path.resolve(),
                    source_map_path=resolved_source_map,
                    current_template=template,
                    current_source_names=set(current_sources),
                    snapshot_type=snapshot_type,
                    filename_marker=filename_marker,
                )
            )

    invalid_snapshots = [snapshot for snapshot in snapshots if snapshot.get("ok") is not True]
    before_bindings_count = sum(snapshot.get("snapshot_type") == "before_bindings" for snapshot in snapshots)
    before_rollback_count = sum(snapshot.get("snapshot_type") == "before_rollback" for snapshot in snapshots)
    bound_sources = sorted(
        source_name for source_name, source in current_sources.items() if source.get("binding_status") == "bound"
    )
    return {
        "schema_version": 1,
        "inspection_type": "research_monitoring_source_binding_history",
        "source_map_file": str(resolved_source_map),
        "template": template,
        "current": {
            "fingerprint": _sha256_file(resolved_source_map),
            "binding_status": str(current_payload.get("binding_status", "") or ""),
            "bound_sources": bound_sources,
        },
        "summary": {
            "snapshot_count": len(snapshots),
            "valid_snapshot_count": len(snapshots) - len(invalid_snapshots),
            "invalid_snapshot_count": len(invalid_snapshots),
            "before_bindings_count": before_bindings_count,
            "before_rollback_count": before_rollback_count,
        },
        "snapshots": snapshots,
        "validation": {
            "ok": not invalid_snapshots,
            "errors": [f"invalid binding snapshot: {snapshot.get('snapshot_file')}" for snapshot in invalid_snapshots],
        },
    }


def build_research_template_package_manifest() -> dict[str, object]:
    """Build a local package manifest for all research templates."""

    templates = []
    for template in list_research_templates():
        rules_payload = build_monitoring_rules_payload(template.name)
        source_map_payload = build_monitoring_source_map_payload(template.name)
        validation = validate_monitoring_source_map_payload(rules_payload, source_map_payload)
        variables = rules_payload.get("variables", [])
        data_sources = source_map_payload.get("data_sources", [])
        templates.append(
            {
                "name": template.name,
                "title": template.title,
                "template_file": str(template.path),
                "monitoring_variable_count": len(variables) if isinstance(variables, list) else 0,
                "data_source_count": len(data_sources) if isinstance(data_sources, list) else 0,
                "validation": validation,
            }
        )
    return {
        "schema_version": 1,
        "manifest_type": "research_template_package",
        "templates": templates,
    }


def get_monitoring_data_source_candidates(name: str) -> tuple[str, ...]:
    """Return placeholder data sources for a template monitoring draft."""

    normalized = _normalize_template_name(name)
    return _TEMPLATE_DATA_SOURCE_CANDIDATES.get(normalized, _COMMON_DATA_SOURCE_CANDIDATES)


def write_monitoring_rules_payload(
    name: str,
    *,
    workspace_root: Path,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a monitoring-rule draft JSON file for one research template."""

    normalized = _normalize_template_name(name)
    payload = build_monitoring_rules_payload(normalized)
    target_path = output_path or workspace_root / "assets" / _TEMPLATE_DIR_NAME / f"{normalized}.monitoring-rules.json"
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return target_path


def write_monitoring_source_map_payload(
    name: str,
    *,
    workspace_root: Path,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a monitoring source-map draft JSON file for one research template."""

    normalized = _normalize_template_name(name)
    payload = build_monitoring_source_map_payload(normalized)
    target_path = output_path or workspace_root / "assets" / _TEMPLATE_DIR_NAME / f"{normalized}.source-map.json"
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return target_path


def write_research_template_package_manifest(
    *,
    workspace_root: Path,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a research-template package manifest JSON file."""

    payload = build_research_template_package_manifest()
    target_path = output_path or workspace_root / "assets" / _TEMPLATE_DIR_NAME / "research-template.manifest.json"
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return target_path


def build_research_template_usage_guide(
    name: str,
    *,
    template_file: Path | None = None,
    workbook_file: Path | None = None,
    progress_report_file: Path | None = None,
    rules_file: Path | None = None,
    source_map_file: Path | None = None,
    manifest_file: Path | None = None,
    monitoring_plan_file: Path | None = None,
    monitoring_status_file: Path | None = None,
    workbook_status_file: Path | None = None,
    report_status_file: Path | None = None,
    ticker: str = "",
    company: str = "",
) -> str:
    """Build a local guide for using one materialized research template."""

    normalized = _normalize_template_name(name)
    template = _resolve_template_path(normalized)
    variables = extract_monitoring_variables(normalized)
    data_sources = get_monitoring_data_source_candidates(normalized)
    artifact_lines = []
    for label, path in (
        ("template", template_file),
        ("research_workbook", workbook_file),
        ("research_progress_report", progress_report_file),
        ("monitoring_rules", rules_file),
        ("source_map", source_map_file),
        ("package_manifest", manifest_file),
        ("monitoring_plan", monitoring_plan_file),
        ("monitoring_status", monitoring_status_file),
        ("research_workbook_status", workbook_status_file),
        ("research_progress_report_status", report_status_file),
    ):
        if path is not None:
            artifact_lines.append(f"- `{label}`: `{path}`")
    if not artifact_lines:
        artifact_lines.append(
            "- Run `dayu-cli research-template materialize <name> --base ./workspace` to create local files."
        )

    variable_lines = [f"- {variable}" for variable in variables] or ["- No monitoring variables found."]
    source_lines = [f"- `{source}`" for source in data_sources] or ["- No data source candidates found."]
    template_arg = (
        str(template_file)
        if template_file is not None
        else f"./workspace/assets/research_templates/common-plus-{normalized}.md"
    )
    if normalized == _FALLBACK_TEMPLATE_NAME and template_file is None:
        template_arg = "./workspace/assets/research_templates/common.md"
    research_target = _normalize_research_target(ticker=ticker, company=company)
    ticker_arg = research_target["ticker"] or "<TICKER>"
    target_lines = [
        f"- Ticker: `{research_target['ticker'] or '(not set)'}`",
        f"- Company: `{research_target['company'] or '(not set)'}`",
    ]

    return "\n".join(
        [
            f"# Research Template Usage Guide: {normalized}",
            "",
            f"- Packaged template: `{template}`",
            f"- Template title: {_read_template_title(template)}",
            "",
            "## Research Target",
            "",
            *target_lines,
            "",
            "## Local Artifacts",
            "",
            *artifact_lines,
            "",
            "## Next Write Command",
            "",
            "```bash",
            f'dayu-cli write --ticker {ticker_arg} --template "{template_arg}"',
            "```",
            "",
            "## Monitoring Variables",
            "",
            *variable_lines,
            "",
            "## Data Source Candidates",
            "",
            *source_lines,
            "",
            "## Notes",
            "",
            "- Generated monitoring rules are draft/manual-review rules until their source-map bindings are reviewed.",
            "- Keep this guide with the materialized template files so a future UI, scheduler, or report workflow can discover the bundle.",
            "",
        ]
    )


def write_research_template_usage_guide(
    name: str,
    *,
    workspace_root: Path,
    template_file: Path | None = None,
    workbook_file: Path | None = None,
    progress_report_file: Path | None = None,
    rules_file: Path | None = None,
    source_map_file: Path | None = None,
    manifest_file: Path | None = None,
    monitoring_plan_file: Path | None = None,
    monitoring_status_file: Path | None = None,
    workbook_status_file: Path | None = None,
    report_status_file: Path | None = None,
    ticker: str = "",
    company: str = "",
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a local usage guide for one materialized research template."""

    normalized = _normalize_template_name(name)
    guide = build_research_template_usage_guide(
        normalized,
        template_file=template_file,
        workbook_file=workbook_file,
        progress_report_file=progress_report_file,
        rules_file=rules_file,
        source_map_file=source_map_file,
        manifest_file=manifest_file,
        monitoring_plan_file=monitoring_plan_file,
        monitoring_status_file=monitoring_status_file,
        workbook_status_file=workbook_status_file,
        report_status_file=report_status_file,
        ticker=ticker,
        company=company,
    )
    target_path = output_path or workspace_root / "assets" / _TEMPLATE_DIR_NAME / f"{normalized}.research-guide.md"
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(guide, encoding="utf-8", newline="\n")
    return target_path


def build_research_template_bundle_descriptor(
    name: str,
    *,
    template_file: Path,
    workbook_file: Path,
    progress_report_file: Path | None = None,
    rules_file: Path,
    source_map_file: Path,
    manifest_file: Path,
    guide_file: Path,
    monitoring_validation: dict[str, object],
    research_target: dict[str, str] | None = None,
    source_write_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build a machine-readable descriptor for one materialized bundle."""

    normalized = _normalize_template_name(name)
    template = _resolve_template_path(normalized)
    payload: dict[str, object] = {
        "schema_version": 1,
        "bundle_type": "research_template_bundle",
        "template": normalized,
        "template_title": _read_template_title(template),
        "research_target": _normalize_research_target(
            ticker=(research_target or {}).get("ticker", ""),
            company=(research_target or {}).get("company", ""),
        ),
        "automation_status": "manual_review",
        "artifacts": {
            "write_template": str(template_file.resolve()),
            "research_workbook": str(workbook_file.resolve()),
            "monitoring_rules": str(rules_file.resolve()),
            "source_map": str(source_map_file.resolve()),
            "package_manifest": str(manifest_file.resolve()),
            "usage_guide": str(guide_file.resolve()),
        },
        "capabilities": {
            "write_report": True,
            "track_research_evidence": True,
            "review_monitoring_rules": True,
            "review_source_bindings": True,
            "automated_monitoring": False,
        },
        "monitoring_validation": monitoring_validation,
    }
    if progress_report_file is not None:
        artifacts = payload["artifacts"]
        assert isinstance(artifacts, dict)
        artifacts["research_progress_report"] = str(progress_report_file.resolve())
    if source_write_manifest is not None:
        payload["source_write_manifest"] = deepcopy(source_write_manifest)
    return payload


def _recompute_bundle_monitoring_integrity(
    template: str,
    rules_path_raw: object,
    source_map_path_raw: object,
) -> list[str]:
    """Recompute monitoring rules/source-map integrity from the referenced files.

    The descriptor embeds a ``monitoring_validation`` snapshot captured at
    materialize time, but trusting it lets a tampered ``source-map.json`` (for
    example an ``unbound`` source hand-flipped to ``bound``) keep reporting the
    bundle healthy. This recomputes consistency from the current file contents
    and additionally requires that any ``bound`` source carry the approval
    provenance the sanctioned binding flow writes, so an unauthorized flip is
    detected without touching the intentionally-mutable source-map itself.
    """

    errors: list[str] = []
    if not isinstance(rules_path_raw, str) or not rules_path_raw.strip():
        return errors
    if not isinstance(source_map_path_raw, str) or not source_map_path_raw.strip():
        return errors
    rules_path = Path(rules_path_raw)
    source_map_path = Path(source_map_path_raw)
    if not rules_path.is_file() or not source_map_path.is_file():
        return errors
    try:
        rules_payload = _load_json_object(rules_path)
        source_map_payload = _load_json_object(source_map_path)
    except (OSError, ValueError) as exc:
        errors.append(f"monitoring integrity could not be recomputed: {exc}")
        return errors
    recomputed = validate_monitoring_source_map_payload(rules_payload, source_map_payload)
    recomputed_errors = recomputed.get("errors")
    if isinstance(recomputed_errors, list):
        errors.extend(f"monitoring integrity: {error}" for error in recomputed_errors)
    recomputed_template = str(recomputed.get("template", "") or "")
    if template and recomputed_template and recomputed_template != template:
        errors.append(
            "monitoring integrity: source-map template mismatch: "
            f"bundle={template!r} source_map={recomputed_template!r}"
        )
    data_sources = source_map_payload.get("data_sources")
    if isinstance(data_sources, list):
        for index, source in enumerate(data_sources):
            if not isinstance(source, dict):
                continue
            if str(source.get("binding_status", "") or "") != "bound":
                continue
            source_name = str(source.get("source", "") or "") or f"index {index}"
            approval = source.get("binding_approval")
            approved_by = ""
            approval_reference = ""
            if isinstance(approval, dict):
                approved_by = str(approval.get("approved_by", "") or "").strip()
                approval_reference = str(approval.get("approval_reference", "") or "").strip()
            if not approved_by or not approval_reference:
                errors.append(
                    "monitoring integrity: bound source lacks binding_approval provenance: "
                    f"{source_name}"
                )
    return errors


def validate_research_template_bundle_descriptor(payload: dict[str, object]) -> dict[str, object]:
    """Validate bundle schema shape and referenced local artifacts."""

    errors: list[str] = []
    warnings: list[str] = []
    workbook_validation: dict[str, object] | None = None
    workbook_report_validation: dict[str, object] | None = None
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("bundle_type") != "research_template_bundle":
        errors.append("bundle_type must be research_template_bundle")

    template = str(payload.get("template", "") or "")
    if not template:
        errors.append("template is required")
    if payload.get("automation_status") != "manual_review":
        errors.append("automation_status must be manual_review")

    research_target = payload.get("research_target")
    if not isinstance(research_target, dict):
        errors.append("research_target must be an object")
    else:
        for key in ("ticker", "company"):
            if not isinstance(research_target.get(key), str):
                errors.append(f"research_target.{key} must be a string")

    source_write_manifest = payload.get("source_write_manifest")
    if source_write_manifest is not None:
        if not isinstance(source_write_manifest, dict):
            errors.append("source_write_manifest must be an object")
        else:
            source_path_raw = source_write_manifest.get("path")
            source_file_fingerprint = str(source_write_manifest.get("file_fingerprint", "") or "")
            source_semantic_fingerprint = str(source_write_manifest.get("semantic_fingerprint", "") or "")
            selected_template = str(source_write_manifest.get("selected_template", "") or "")
            if not isinstance(source_path_raw, str) or not source_path_raw.strip():
                errors.append("source_write_manifest.path must be a non-empty path")
            else:
                source_path = Path(source_path_raw)
                if not source_path.is_file():
                    errors.append(f"source_write_manifest.path does not exist: {source_path}")
                else:
                    try:
                        current_semantics = _build_write_manifest_binding_semantics(source_path.resolve())
                        semantic_matches = _sha256_json_object(current_semantics) == source_semantic_fingerprint
                        if not semantic_matches:
                            errors.append("source_write_manifest semantic fingerprint is stale")
                        if semantic_matches and _sha256_file(source_path) != source_file_fingerprint:
                            warnings.append("source_write_manifest file changed without selection drift")
                        current_template, _current_selection = _resolve_template_selection_from_write_manifest(
                            source_path.resolve()
                        )
                        if current_template != selected_template or current_template != template:
                            errors.append(
                                "source_write_manifest template mismatch: "
                                f"bundle={template!r} source={current_template!r}"
                            )
                    except (OSError, ValueError) as exc:
                        errors.append(f"source_write_manifest is invalid: {exc}")

    artifacts = payload.get("artifacts")
    artifact_count = 0
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
    else:
        artifact_count = len(artifacts)
        for key in (item for item in _BUNDLE_ARTIFACT_KEYS if item != "research_progress_report"):
            path_raw = artifacts.get(key)
            if not isinstance(path_raw, str) or not path_raw.strip():
                errors.append(f"artifacts.{key} must be a non-empty path")
            elif not Path(path_raw).is_file():
                errors.append(f"artifacts.{key} does not exist: {path_raw}")
        extra_keys = sorted(str(key) for key in artifacts if key not in _BUNDLE_ARTIFACT_KEYS)
        if extra_keys:
            warnings.append(f"unknown artifact entries: {', '.join(extra_keys)}")
        errors.extend(
            _recompute_bundle_monitoring_integrity(
                template,
                artifacts.get("monitoring_rules"),
                artifacts.get("source_map"),
            )
        )
        workbook_raw = artifacts.get("research_workbook")
        if isinstance(workbook_raw, str) and workbook_raw.strip() and Path(workbook_raw).is_file():
            try:
                workbook_payload = _load_json_object(Path(workbook_raw))
                workbook_validation = validate_research_workbook_payload(workbook_payload)
                workbook_errors = workbook_validation.get("errors")
                if isinstance(workbook_errors, list):
                    errors.extend(f"artifacts.research_workbook: {error}" for error in workbook_errors)
                workbook_warnings = workbook_validation.get("warnings")
                if isinstance(workbook_warnings, list):
                    warnings.extend(f"artifacts.research_workbook: {warning}" for warning in workbook_warnings)
                workbook_template = str(workbook_payload.get("template", "") or "")
                if workbook_template != template:
                    errors.append(
                        "artifacts.research_workbook template mismatch: "
                        f"bundle={template!r} workbook={workbook_template!r}"
                    )
                workbook_target = workbook_payload.get("research_target")
                if isinstance(research_target, dict) and workbook_target != research_target:
                    errors.append("artifacts.research_workbook research_target does not match bundle")
            except (OSError, ValueError) as exc:
                errors.append(f"artifacts.research_workbook is invalid JSON: {exc}")
        report_raw = artifacts.get("research_progress_report")
        if report_raw is not None:
            if not isinstance(report_raw, str) or not report_raw.strip():
                errors.append("artifacts.research_progress_report must be a non-empty path")
            elif not Path(report_raw).is_file():
                errors.append(f"artifacts.research_progress_report does not exist: {report_raw}")
            elif isinstance(workbook_raw, str) and workbook_raw.strip():
                workbook_report_validation = inspect_research_workbook_report(
                    Path(report_raw),
                    Path(workbook_raw),
                )
                report_validation = workbook_report_validation.get("validation")
                if isinstance(report_validation, dict):
                    report_errors = report_validation.get("errors")
                    if isinstance(report_errors, list):
                        errors.extend(f"artifacts.research_progress_report: {error}" for error in report_errors)
                    report_warnings = report_validation.get("warnings")
                    if isinstance(report_warnings, list):
                        warnings.extend(f"artifacts.research_progress_report: {warning}" for warning in report_warnings)

    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict):
        errors.append("capabilities must be an object")
    else:
        for key in (
            "write_report",
            "track_research_evidence",
            "review_monitoring_rules",
            "review_source_bindings",
            "automated_monitoring",
        ):
            if not isinstance(capabilities.get(key), bool):
                errors.append(f"capabilities.{key} must be a boolean")

    monitoring_validation = payload.get("monitoring_validation")
    if not isinstance(monitoring_validation, dict):
        errors.append("monitoring_validation must be an object")
    elif monitoring_validation.get("ok") is not True:
        errors.append("monitoring_validation.ok must be true")

    return {
        "ok": not errors,
        "template": template,
        "errors": errors,
        "warnings": warnings,
        "artifact_count": artifact_count,
        "workbook_validation": workbook_validation,
        "workbook_report_validation": workbook_report_validation,
    }


def inspect_research_template_bundle(bundle_path: Path) -> dict[str, object]:
    """Load and validate one materialized research-template bundle."""

    resolved_path = bundle_path.resolve()
    try:
        payload = _load_json_object(resolved_path)
    except (OSError, ValueError) as exc:
        return {
            "descriptor_file": str(resolved_path),
            "template": "",
            "template_title": "",
            "research_target": {},
            "automation_status": "",
            "validation": {
                "ok": False,
                "template": "",
                "errors": [f"unable to load bundle descriptor: {exc}"],
                "warnings": [],
                "artifact_count": 0,
            },
        }

    return {
        "descriptor_file": str(resolved_path),
        "template": str(payload.get("template", "") or ""),
        "template_title": str(payload.get("template_title", "") or ""),
        "research_target": payload.get("research_target", {}),
        "automation_status": str(payload.get("automation_status", "") or ""),
        "validation": validate_research_template_bundle_descriptor(payload),
    }


def build_research_template_bundle_rebind_preview(bundle_path: Path) -> dict[str, object]:
    """Build a validated no-write preview for refreshing a bundle source binding."""

    _candidate, preview = _prepare_research_template_bundle_rebind(bundle_path)
    return preview


def write_research_template_bundle_rebind(bundle_path: Path) -> dict[str, object]:
    """Refresh a bundle source binding while preserving exact prior bytes."""

    candidate, preview = _prepare_research_template_bundle_rebind(bundle_path)
    resolved_bundle = Path(str(preview["bundle_file"]))
    if preview["changed"] is not True:
        return {**preview, "applied": False, "backup_file": None}

    original_fingerprint = _sha256_file(resolved_bundle)
    backup_path = resolved_bundle.with_name(f"{resolved_bundle.stem}.before-rebind.{original_fingerprint[:12]}.json")
    if backup_path.exists():
        if not backup_path.is_file() or _sha256_file(backup_path) != original_fingerprint:
            raise ValueError(f"bundle rebind backup collision: {backup_path}")
    else:
        backup_path.write_bytes(resolved_bundle.read_bytes())
    resolved_bundle.write_text(
        json.dumps(candidate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    final_validation = validate_research_template_bundle_descriptor(_load_json_object(resolved_bundle))
    if final_validation.get("ok") is not True:
        raise ValueError(f"rebound bundle failed validation: {final_validation.get('errors', [])}")
    return {
        **preview,
        "applied": True,
        "backup_file": str(backup_path),
        "bundle_fingerprint_after": _sha256_file(resolved_bundle),
        "validation": final_validation,
    }


def _prepare_research_template_bundle_rebind(
    bundle_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    resolved_bundle = bundle_path.resolve()
    payload = _load_json_object(resolved_bundle)
    template = str(payload.get("template", "") or "")
    if not template:
        raise ValueError("bundle template is required")
    source_binding = payload.get("source_write_manifest")
    if not isinstance(source_binding, dict):
        raise ValueError("bundle does not contain a source_write_manifest binding")
    source_path_raw = source_binding.get("path")
    if not isinstance(source_path_raw, str) or not source_path_raw.strip():
        raise ValueError("bundle source_write_manifest.path is required")
    source_path = Path(source_path_raw).resolve()
    refreshed_binding = _build_source_write_manifest_binding(source_path, expected_template=template)
    candidate = deepcopy(payload)
    candidate["source_write_manifest"] = refreshed_binding
    validation = validate_research_template_bundle_descriptor(candidate)
    if validation.get("ok") is not True:
        raise ValueError(f"bundle rebind candidate is unhealthy: {validation.get('errors', [])}")
    return candidate, {
        "bundle_file": str(resolved_bundle),
        "template": template,
        "source_manifest_file": str(source_path),
        "changed": refreshed_binding != source_binding,
        "semantic_fingerprint_before": str(source_binding.get("semantic_fingerprint", "") or ""),
        "semantic_fingerprint_after": str(refreshed_binding["semantic_fingerprint"]),
        "file_fingerprint_before": str(source_binding.get("file_fingerprint", "") or ""),
        "file_fingerprint_after": str(refreshed_binding["file_fingerprint"]),
        "validation": validation,
    }


def build_research_template_bundle_rebind_rollback_preview(
    bundle_path: Path,
    backup_path: Path,
) -> dict[str, object]:
    """Validate an exact descriptor backup before a rebind rollback."""

    resolved_bundle = bundle_path.resolve()
    resolved_backup = backup_path.resolve()
    if resolved_backup.parent != resolved_bundle.parent:
        raise ValueError("bundle rebind backup must be in the same directory as the bundle")
    if not resolved_bundle.is_file() or not resolved_backup.is_file():
        raise FileNotFoundError("bundle and rebind backup must both exist")
    backup_fingerprint = _sha256_file(resolved_backup)
    expected_name = f"{resolved_bundle.stem}.before-rebind.{backup_fingerprint[:12]}.json"
    if resolved_backup.name != expected_name:
        raise ValueError(f"bundle rebind backup filename mismatch: expected {expected_name!r}")
    current_payload = _load_json_object(resolved_bundle)
    backup_payload = _load_json_object(resolved_backup)
    for key in ("template", "research_target", "artifacts"):
        if backup_payload.get(key) != current_payload.get(key):
            raise ValueError(f"bundle rebind backup {key} does not match current bundle")
    current_binding = current_payload.get("source_write_manifest")
    backup_binding = backup_payload.get("source_write_manifest")
    if not isinstance(current_binding, dict) or not isinstance(backup_binding, dict):
        raise ValueError("bundle and rebind backup must contain source_write_manifest bindings")
    if backup_binding.get("path") != current_binding.get("path"):
        raise ValueError("bundle rebind backup source manifest path does not match current bundle")
    current_fingerprint = _sha256_file(resolved_bundle)
    restored_validation = validate_research_template_bundle_descriptor(backup_payload)
    return {
        "bundle_file": str(resolved_bundle),
        "backup_file": str(resolved_backup),
        "changed": current_fingerprint != backup_fingerprint,
        "bundle_fingerprint_before": current_fingerprint,
        "bundle_fingerprint_after": backup_fingerprint,
        "restored_validation": restored_validation,
        "restored_will_be_healthy": restored_validation.get("ok") is True,
    }


def write_research_template_bundle_rebind_rollback(
    bundle_path: Path,
    backup_path: Path,
) -> dict[str, object]:
    """Restore exact descriptor bytes and preserve the current state for redo."""

    preview = build_research_template_bundle_rebind_rollback_preview(bundle_path, backup_path)
    resolved_bundle = Path(str(preview["bundle_file"]))
    resolved_backup = Path(str(preview["backup_file"]))
    if preview["changed"] is not True:
        return {**preview, "applied": False, "redo_backup_file": None}
    current_fingerprint = str(preview["bundle_fingerprint_before"])
    redo_backup_path = resolved_bundle.with_name(
        f"{resolved_bundle.stem}.before-rebind.{current_fingerprint[:12]}.json"
    )
    if redo_backup_path.exists():
        if not redo_backup_path.is_file() or _sha256_file(redo_backup_path) != current_fingerprint:
            raise ValueError(f"bundle rebind redo backup collision: {redo_backup_path}")
    else:
        redo_backup_path.write_bytes(resolved_bundle.read_bytes())
    resolved_bundle.write_bytes(resolved_backup.read_bytes())
    restored_fingerprint = _sha256_file(resolved_bundle)
    if restored_fingerprint != preview["bundle_fingerprint_after"]:
        raise ValueError("bundle rebind rollback did not restore the expected bytes")
    restored_validation = validate_research_template_bundle_descriptor(_load_json_object(resolved_bundle))
    return {
        **preview,
        "applied": True,
        "redo_backup_file": str(redo_backup_path),
        "restored_validation": restored_validation,
        "restored_will_be_healthy": restored_validation.get("ok") is True,
    }


def discover_research_template_bundles(
    workspace_root: Path,
    *,
    recursive: bool = False,
) -> tuple[dict[str, object], ...]:
    """Discover and inspect bundle descriptors in the standard workspace directory."""

    paths = _discover_research_artifact_paths(workspace_root, "*.bundle.json", recursive=recursive)
    return tuple(inspect_research_template_bundle(path) for path in paths)


def build_monitoring_execution_plan(bundle_path: Path) -> dict[str, object]:
    """Build a scheduler-readable dry-run plan from one healthy bundle."""

    resolved_bundle_path = bundle_path.resolve()
    inspection = inspect_research_template_bundle(resolved_bundle_path)
    bundle_validation = inspection.get("validation")
    if not isinstance(bundle_validation, dict) or bundle_validation.get("ok") is not True:
        errors = bundle_validation.get("errors", []) if isinstance(bundle_validation, dict) else []
        detail = "; ".join(_as_str_list(errors)) or "bundle validation failed"
        raise ValueError(f"cannot build monitoring plan from unhealthy bundle: {detail}")

    bundle_payload = _load_json_object(resolved_bundle_path)
    artifacts = bundle_payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("bundle artifacts must be an object")
    rules_path = Path(str(artifacts["monitoring_rules"])).resolve()
    source_map_path = Path(str(artifacts["source_map"])).resolve()
    rules_payload = _load_json_object(rules_path)
    source_map_payload = _load_json_object(source_map_path)
    source_validation = validate_monitoring_source_map_payload(rules_payload, source_map_payload)
    if source_validation.get("ok") is not True:
        detail = "; ".join(_as_str_list(source_validation.get("errors"))) or "source-map validation failed"
        raise ValueError(f"cannot build monitoring plan: {detail}")

    sources_by_name: dict[str, dict[str, object]] = {}
    data_sources = source_map_payload.get("data_sources", [])
    if isinstance(data_sources, list):
        for source in data_sources:
            if not isinstance(source, dict):
                continue
            source_name = str(source.get("source", "") or "")
            if source_name:
                sources_by_name[source_name] = source

    template = str(bundle_payload.get("template", "") or "")
    research_target = bundle_payload.get("research_target")
    if not isinstance(research_target, dict):
        research_target = {}
    target = _normalize_research_target(
        ticker=str(research_target.get("ticker", "") or ""),
        company=str(research_target.get("company", "") or ""),
    )
    task_prefix = _identifier_component(target["ticker"])
    task_namespace = f"{task_prefix}-{template}" if task_prefix else template
    tasks: list[dict[str, object]] = []
    variables = rules_payload.get("variables", [])
    if isinstance(variables, list):
        for index, variable in enumerate(variables, start=1):
            if not isinstance(variable, dict):
                continue
            candidates = _as_str_list(variable.get("data_source_candidates"))
            bound_sources = [
                source_name
                for source_name in candidates
                if sources_by_name.get(source_name, {}).get("binding_status") == "bound"
            ]
            task_status = "ready_for_review" if bound_sources else "blocked_unbound_sources"
            tasks.append(
                {
                    "task_id": f"{task_namespace}-monitor-{index:03d}",
                    "variable": str(variable.get("name", "") or ""),
                    "status": task_status,
                    "evidence_required": bool(variable.get("evidence_required", True)),
                    "data_source_candidates": candidates,
                    "bound_data_sources": bound_sources,
                    "blocking_reasons": [] if bound_sources else ["no candidate data source is bound"],
                }
            )

    ready_task_count = sum(task.get("status") == "ready_for_review" for task in tasks)
    blocked_task_count = len(tasks) - ready_task_count
    if not tasks:
        readiness_status = "blocked_no_tasks"
        blocking_reasons = ["monitoring rules contain no variables"]
    elif blocked_task_count:
        readiness_status = "blocked_unbound_sources"
        blocking_reasons = [f"{blocked_task_count} monitoring tasks have no bound data source"]
    else:
        readiness_status = "ready_for_review"
        blocking_reasons = []

    return {
        "schema_version": 1,
        "plan_type": "research_monitoring_execution_plan",
        "template": template,
        "research_target": target,
        "bundle_file": str(resolved_bundle_path),
        "execution_mode": "dry_run",
        "cadence": str(rules_payload.get("cadence", "") or ""),
        "automated_execution_allowed": False,
        "input_files": {
            "monitoring_rules": str(rules_path),
            "source_map": str(source_map_path),
        },
        "input_fingerprints": {
            "monitoring_rules": _sha256_file(rules_path),
            "source_map": _sha256_file(source_map_path),
        },
        "readiness": {
            "status": readiness_status,
            "task_count": len(tasks),
            "ready_task_count": ready_task_count,
            "blocked_task_count": blocked_task_count,
            "blocking_reasons": blocking_reasons,
        },
        "source_validation": source_validation,
        "tasks": tasks,
    }


def validate_monitoring_execution_plan(payload: dict[str, object]) -> dict[str, object]:
    """Validate a persisted dry-run monitoring plan and detect stale inputs."""

    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("plan_type") != "research_monitoring_execution_plan":
        errors.append("plan_type must be research_monitoring_execution_plan")
    if payload.get("execution_mode") != "dry_run":
        errors.append("execution_mode must be dry_run")
    if payload.get("automated_execution_allowed") is not False:
        errors.append("automated_execution_allowed must be false")

    template = str(payload.get("template", "") or "")
    if not template:
        errors.append("template is required")
    research_target = payload.get("research_target")
    if not isinstance(research_target, dict):
        errors.append("research_target must be an object")
    else:
        for key in ("ticker", "company"):
            if not isinstance(research_target.get(key), str):
                errors.append(f"research_target.{key} must be a string")
    bundle_file = str(payload.get("bundle_file", "") or "")
    if not bundle_file:
        errors.append("bundle_file is required")
    elif not Path(bundle_file).is_file():
        errors.append(f"bundle_file does not exist: {bundle_file}")

    input_files = payload.get("input_files")
    fingerprints = payload.get("input_fingerprints")
    if not isinstance(input_files, dict):
        errors.append("input_files must be an object")
        input_files = {}
    if not isinstance(fingerprints, dict):
        errors.append("input_fingerprints must be an object")
        fingerprints = {}
    for key in ("monitoring_rules", "source_map"):
        path_raw = input_files.get(key)
        fingerprint_raw = fingerprints.get(key)
        if not isinstance(path_raw, str) or not path_raw.strip():
            errors.append(f"input_files.{key} must be a non-empty path")
            continue
        path = Path(path_raw)
        if not path.is_file():
            errors.append(f"input_files.{key} does not exist: {path_raw}")
            continue
        if not isinstance(fingerprint_raw, str) or len(fingerprint_raw) != 64:
            errors.append(f"input_fingerprints.{key} must be a SHA-256 hex digest")
        elif _sha256_file(path) != fingerprint_raw:
            errors.append(f"input_files.{key} changed after plan generation")

    # Recompute the genuinely-bound source names from the authentic source-map so
    # a plan cannot claim a task is ready_for_review (or that an unbindable
    # external placeholder is bound) that the source-map does not actually
    # support. Only enforced when the source-map is present and its fingerprint
    # matches (i.e. the plan's declared inputs are authentic); a
    # missing/changed source-map is already reported above.
    genuinely_bound_sources: set[str] | None = None
    source_map_raw = input_files.get("source_map")
    source_map_fp = fingerprints.get("source_map")
    if (
        isinstance(source_map_raw, str)
        and source_map_raw.strip()
        and Path(source_map_raw).is_file()
        and isinstance(source_map_fp, str)
        and len(source_map_fp) == 64
        and _sha256_file(Path(source_map_raw)) == source_map_fp
    ):
        try:
            source_map_payload = _load_json_object(Path(source_map_raw))
        except (OSError, ValueError):
            source_map_payload = {}
        genuinely_bound_sources = set()
        data_sources = source_map_payload.get("data_sources")
        if isinstance(data_sources, list):
            for source in data_sources:
                if not isinstance(source, dict):
                    continue
                if str(source.get("binding_status", "") or "") != "bound":
                    continue
                source_name = str(source.get("source", "") or "")
                if source_name:
                    genuinely_bound_sources.add(source_name)

    task_statuses: list[str] = []
    task_ids: set[str] = set()
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        errors.append("tasks must be a list")
        tasks = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"tasks[{index}] must be an object")
            continue
        task_id = str(task.get("task_id", "") or "")
        if not task_id:
            errors.append(f"tasks[{index}].task_id is required")
        elif task_id in task_ids:
            errors.append(f"duplicate task_id: {task_id}")
        task_ids.add(task_id)
        status = str(task.get("status", "") or "")
        if status not in {"ready_for_review", "blocked_unbound_sources"}:
            errors.append(f"tasks[{index}].status is invalid: {status!r}")
        task_statuses.append(status)
        bound_sources = _as_str_list(task.get("bound_data_sources"))
        if status == "ready_for_review" and not bound_sources:
            errors.append(f"tasks[{index}] is ready_for_review without a bound data source")
        if status == "blocked_unbound_sources" and bound_sources:
            errors.append(f"tasks[{index}] is blocked despite having a bound data source")
        if genuinely_bound_sources is not None:
            forged = [source for source in bound_sources if source not in genuinely_bound_sources]
            if forged:
                errors.append(
                    f"tasks[{index}] claims bound data sources not bound in the source-map: "
                    f"{', '.join(sorted(forged))}"
                )
            candidates = _as_str_list(task.get("data_source_candidates"))
            expected_bound = sorted(source for source in candidates if source in genuinely_bound_sources)
            if status == "ready_for_review" and not expected_bound:
                errors.append(
                    f"tasks[{index}] is ready_for_review but no candidate data source is bound in the source-map"
                )

    ready_count = sum(status == "ready_for_review" for status in task_statuses)
    blocked_count = sum(status == "blocked_unbound_sources" for status in task_statuses)
    readiness = payload.get("readiness")
    if not isinstance(readiness, dict):
        errors.append("readiness must be an object")
    else:
        expected_status = (
            "blocked_no_tasks" if not tasks else "blocked_unbound_sources" if blocked_count else "ready_for_review"
        )
        if readiness.get("status") != expected_status:
            errors.append(f"readiness.status must be {expected_status}")
        expected_counts = {
            "task_count": len(tasks),
            "ready_task_count": ready_count,
            "blocked_task_count": blocked_count,
        }
        for key, expected in expected_counts.items():
            if readiness.get(key) != expected:
                errors.append(f"readiness.{key} must be {expected}")

    source_validation = payload.get("source_validation")
    if not isinstance(source_validation, dict) or source_validation.get("ok") is not True:
        errors.append("source_validation.ok must be true")

    return {
        "ok": not errors,
        "template": template,
        "errors": errors,
        "warnings": warnings,
        "task_count": len(tasks),
        "ready_task_count": ready_count,
        "blocked_task_count": blocked_count,
    }


def inspect_monitoring_execution_plan(plan_path: Path) -> dict[str, object]:
    """Load and validate one persisted monitoring execution plan."""

    resolved_path = plan_path.resolve()
    try:
        payload = _load_json_object(resolved_path)
    except (OSError, ValueError) as exc:
        return {
            "monitoring_plan_file": str(resolved_path),
            "validation": {
                "ok": False,
                "template": "",
                "errors": [f"unable to load monitoring plan: {exc}"],
                "warnings": [],
                "task_count": 0,
                "ready_task_count": 0,
                "blocked_task_count": 0,
            },
        }
    return {
        "monitoring_plan_file": str(resolved_path),
        "template": str(payload.get("template", "") or ""),
        "research_target": payload.get("research_target", {}),
        "cadence": str(payload.get("cadence", "") or ""),
        "execution_mode": str(payload.get("execution_mode", "") or ""),
        "readiness": payload.get("readiness", {}),
        "validation": validate_monitoring_execution_plan(payload),
    }


def discover_monitoring_execution_plans(
    workspace_root: Path,
    *,
    recursive: bool = False,
) -> tuple[dict[str, object], ...]:
    """Discover monitoring plans in the standard workspace directory."""

    paths = _discover_research_artifact_paths(workspace_root, "*.monitoring-plan.json", recursive=recursive)
    return tuple(inspect_monitoring_execution_plan(path) for path in paths)


def build_monitoring_status_snapshot(workspace_root: Path, *, recursive: bool = False) -> dict[str, object]:
    """Build a deterministic workspace-level monitoring status snapshot."""

    resolved_workspace = workspace_root.resolve()
    plans = discover_monitoring_execution_plans(resolved_workspace, recursive=recursive)
    valid_count = 0
    invalid_count = 0
    ready_plan_count = 0
    blocked_plan_count = 0
    task_count = 0
    ready_task_count = 0
    blocked_task_count = 0
    targeted_plan_count = 0
    untargeted_plan_count = 0
    target_companies: dict[str, str] = {}
    target_counters: dict[str, dict[str, int]] = {}
    for plan in plans:
        research_target = plan.get("research_target")
        ticker = str(research_target.get("ticker", "") or "") if isinstance(research_target, dict) else ""
        company = str(research_target.get("company", "") or "") if isinstance(research_target, dict) else ""
        target_counter: dict[str, int] | None = None
        if ticker:
            targeted_plan_count += 1
            target_companies.setdefault(ticker, company)
            target_counter = target_counters.setdefault(
                ticker,
                {
                    "plan_count": 0,
                    "valid_plan_count": 0,
                    "invalid_plan_count": 0,
                    "ready_for_review_plan_count": 0,
                    "blocked_plan_count": 0,
                },
            )
            target_counter["plan_count"] += 1
        else:
            untargeted_plan_count += 1
        validation = plan.get("validation")
        if not isinstance(validation, dict) or validation.get("ok") is not True:
            invalid_count += 1
            if target_counter is not None:
                target_counter["invalid_plan_count"] += 1
            continue
        valid_count += 1
        if target_counter is not None:
            target_counter["valid_plan_count"] += 1
        readiness = plan.get("readiness")
        readiness_status = str(readiness.get("status", "") or "") if isinstance(readiness, dict) else ""
        if readiness_status == "ready_for_review":
            ready_plan_count += 1
            if target_counter is not None:
                target_counter["ready_for_review_plan_count"] += 1
        else:
            blocked_plan_count += 1
            if target_counter is not None:
                target_counter["blocked_plan_count"] += 1
        task_count += int(validation.get("task_count", 0) or 0)
        ready_task_count += int(validation.get("ready_task_count", 0) or 0)
        blocked_task_count += int(validation.get("blocked_task_count", 0) or 0)

    if not plans:
        overall_status = "no_plans"
    elif invalid_count:
        overall_status = "unhealthy"
    elif blocked_plan_count:
        overall_status = "blocked"
    else:
        overall_status = "ready_for_review"
    targets = []
    for ticker in sorted(target_counters):
        counters = target_counters[ticker]
        if counters["invalid_plan_count"]:
            target_status = "unhealthy"
        elif counters["blocked_plan_count"]:
            target_status = "blocked"
        else:
            target_status = "ready_for_review"
        targets.append(
            {
                "ticker": ticker,
                "company": target_companies[ticker],
                "overall_status": target_status,
                **counters,
            }
        )
    return {
        "schema_version": 1,
        "snapshot_type": "research_monitoring_status",
        "workspace_root": str(resolved_workspace),
        "scan_scope": "recursive" if recursive else "workspace",
        "overall_status": overall_status,
        "summary": {
            "plan_count": len(plans),
            "valid_plan_count": valid_count,
            "invalid_plan_count": invalid_count,
            "ready_for_review_plan_count": ready_plan_count,
            "blocked_plan_count": blocked_plan_count,
            "task_count": task_count,
            "ready_task_count": ready_task_count,
            "blocked_task_count": blocked_task_count,
            "targeted_plan_count": targeted_plan_count,
            "untargeted_plan_count": untargeted_plan_count,
            "target_count": len(targets),
        },
        "targets": targets,
        "plans": list(plans),
    }


def write_monitoring_status_snapshot(
    workspace_root: Path,
    *,
    output_path: Path | None = None,
    recursive: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write a workspace-level monitoring status snapshot."""

    resolved_workspace = workspace_root.resolve()
    payload = build_monitoring_status_snapshot(resolved_workspace, recursive=recursive)
    target_path = output_path or resolved_workspace / "assets" / _TEMPLATE_DIR_NAME / "monitoring-status.json"
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return target_path


def build_monitoring_scheduler_manifest(
    workspace_root: Path,
    *,
    recursive: bool = False,
    timezone: str = "UTC",
) -> dict[str, object]:
    """Build a platform-neutral, disabled-by-default scheduler job manifest."""

    resolved_workspace = workspace_root.resolve()
    normalized_timezone = timezone.strip()
    if not normalized_timezone:
        raise ValueError("scheduler timezone is required")
    plans = discover_monitoring_execution_plans(resolved_workspace, recursive=recursive)
    jobs: list[dict[str, object]] = []
    seen_job_ids: set[str] = set()
    invalid_count = 0
    blocked_count = 0
    ready_count = 0
    for plan in plans:
        plan_path = Path(str(plan["monitoring_plan_file"])).resolve()
        validation = plan.get("validation")
        is_valid = isinstance(validation, dict) and validation.get("ok") is True
        readiness = plan.get("readiness")
        readiness_status = str(readiness.get("status", "") or "") if isinstance(readiness, dict) else ""
        if not is_valid:
            state = "invalid_plan"
            invalid_count += 1
        elif readiness_status == "ready_for_review":
            state = "ready_for_review"
            ready_count += 1
        else:
            state = readiness_status or "blocked_unknown"
            blocked_count += 1
        research_target = plan.get("research_target")
        target = (
            _normalize_research_target(
                ticker=str(research_target.get("ticker", "") or ""),
                company=str(research_target.get("company", "") or ""),
            )
            if isinstance(research_target, dict)
            else {"ticker": "", "company": ""}
        )
        template = str(plan.get("template", "") or "unknown")
        namespace = _identifier_component(target["ticker"] or template) or "unknown"
        candidate_job_id = f"{namespace}-{template}-monitoring"
        job_id = candidate_job_id
        if job_id in seen_job_ids:
            path_suffix = hashlib.sha256(str(plan_path).encode("utf-8")).hexdigest()[:8]
            job_id = f"{candidate_job_id}-{path_suffix}"
        seen_job_ids.add(job_id)
        cadence = str(plan.get("cadence", "") or "manual")
        jobs.append(
            {
                "job_id": job_id,
                "enabled": False,
                "eligible_for_manual_activation": state == "ready_for_review",
                "state": state,
                "research_target": target,
                "template": template,
                "monitoring_plan_file": str(plan_path),
                "monitoring_plan_fingerprint": _sha256_file(plan_path) if plan_path.is_file() else "",
                "trigger": {
                    "type": "cadence",
                    "cadence": cadence,
                    "timezone": normalized_timezone,
                    "binding_status": "unbound",
                },
                "action": {
                    "type": "command_argv",
                    "argv": [
                        "dayu-cli",
                        "research-template",
                        "validate-monitoring-plan",
                        "--plan",
                        str(plan_path),
                    ],
                },
            }
        )
    return {
        "schema_version": 1,
        "manifest_type": "research_monitoring_scheduler",
        "workspace_root": str(resolved_workspace),
        "scan_scope": "recursive" if recursive else "workspace",
        "timezone": normalized_timezone,
        "activation_policy": {
            "manual_approval_required": True,
            "automated_provider_execution_allowed": False,
            "scheduler_binding_status": "unbound",
        },
        "summary": {
            "job_count": len(jobs),
            "enabled_job_count": 0,
            "manual_activation_candidate_count": ready_count,
            "ready_for_review_job_count": ready_count,
            "blocked_job_count": blocked_count,
            "invalid_job_count": invalid_count,
        },
        "jobs": jobs,
    }


def validate_monitoring_scheduler_manifest(payload: dict[str, object]) -> dict[str, object]:
    """Validate scheduler safety invariants and referenced plan freshness."""

    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("manifest_type") != "research_monitoring_scheduler":
        errors.append("manifest_type must be research_monitoring_scheduler")
    timezone = str(payload.get("timezone", "") or "")
    if not timezone:
        errors.append("timezone is required")

    policy = payload.get("activation_policy")
    if not isinstance(policy, dict):
        errors.append("activation_policy must be an object")
    else:
        if policy.get("manual_approval_required") is not True:
            errors.append("activation_policy.manual_approval_required must be true")
        if policy.get("automated_provider_execution_allowed") is not False:
            errors.append("activation_policy.automated_provider_execution_allowed must be false")
        if policy.get("scheduler_binding_status") != "unbound":
            errors.append("activation_policy.scheduler_binding_status must be unbound")

    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        errors.append("jobs must be a list")
        jobs = []
    seen_job_ids: set[str] = set()
    enabled_count = 0
    ready_count = 0
    blocked_count = 0
    invalid_count = 0
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            errors.append(f"jobs[{index}] must be an object")
            continue
        job_id = str(job.get("job_id", "") or "")
        if not job_id:
            errors.append(f"jobs[{index}].job_id is required")
        elif job_id in seen_job_ids:
            errors.append(f"duplicate job_id: {job_id}")
        seen_job_ids.add(job_id)
        if job.get("enabled") is not False:
            errors.append(f"jobs[{index}].enabled must be false")
            enabled_count += 1

        state = str(job.get("state", "") or "")
        if state == "ready_for_review":
            ready_count += 1
        elif state == "invalid_plan":
            invalid_count += 1
        else:
            blocked_count += 1
        expected_eligibility = state == "ready_for_review"
        if job.get("eligible_for_manual_activation") is not expected_eligibility:
            errors.append(f"jobs[{index}].eligible_for_manual_activation must be {str(expected_eligibility).lower()}")

        plan_raw = job.get("monitoring_plan_file")
        plan_path: Path | None = None
        if not isinstance(plan_raw, str) or not plan_raw.strip():
            errors.append(f"jobs[{index}].monitoring_plan_file must be a non-empty path")
        else:
            plan_path = Path(plan_raw)
            if not plan_path.is_file():
                errors.append(f"jobs[{index}].monitoring_plan_file does not exist: {plan_raw}")
            else:
                fingerprint = job.get("monitoring_plan_fingerprint")
                if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                    errors.append(f"jobs[{index}].monitoring_plan_fingerprint must be a SHA-256 hex digest")
                elif _sha256_file(plan_path) != fingerprint:
                    errors.append(f"jobs[{index}].monitoring_plan_file changed after scheduler export")
                inspection = inspect_monitoring_execution_plan(plan_path)
                expected_state = _scheduler_state_from_plan_inspection(inspection)
                if state != expected_state:
                    errors.append(f"jobs[{index}].state must be {expected_state}")
                if str(job.get("template", "") or "") != str(inspection.get("template", "") or ""):
                    errors.append(f"jobs[{index}].template does not match monitoring plan")

        trigger = job.get("trigger")
        if not isinstance(trigger, dict):
            errors.append(f"jobs[{index}].trigger must be an object")
        else:
            if trigger.get("type") != "cadence":
                errors.append(f"jobs[{index}].trigger.type must be cadence")
            if not str(trigger.get("cadence", "") or ""):
                errors.append(f"jobs[{index}].trigger.cadence is required")
            if trigger.get("timezone") != timezone:
                errors.append(f"jobs[{index}].trigger.timezone must match manifest timezone")
            if trigger.get("binding_status") != "unbound":
                errors.append(f"jobs[{index}].trigger.binding_status must be unbound")

        action = job.get("action")
        if not isinstance(action, dict):
            errors.append(f"jobs[{index}].action must be an object")
        else:
            expected_argv = (
                [
                    "dayu-cli",
                    "research-template",
                    "validate-monitoring-plan",
                    "--plan",
                    str(plan_path),
                ]
                if plan_path is not None
                else []
            )
            if action.get("type") != "command_argv":
                errors.append(f"jobs[{index}].action.type must be command_argv")
            if action.get("argv") != expected_argv:
                errors.append(f"jobs[{index}].action.argv must match the safe validation command")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
    else:
        expected_summary = {
            "job_count": len(jobs),
            "enabled_job_count": enabled_count,
            "manual_activation_candidate_count": ready_count,
            "ready_for_review_job_count": ready_count,
            "blocked_job_count": blocked_count,
            "invalid_job_count": invalid_count,
        }
        for key, expected in expected_summary.items():
            if summary.get(key) != expected:
                errors.append(f"summary.{key} must be {expected}")
    if enabled_count:
        errors.append("scheduler manifest must not contain enabled jobs")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "job_count": len(jobs),
        "ready_for_review_job_count": ready_count,
        "blocked_job_count": blocked_count,
        "invalid_job_count": invalid_count,
    }


def inspect_monitoring_scheduler_manifest(manifest_path: Path) -> dict[str, object]:
    """Load and validate one persisted scheduler manifest."""

    resolved_path = manifest_path.resolve()
    try:
        payload = _load_json_object(resolved_path)
    except (OSError, ValueError) as exc:
        return {
            "scheduler_manifest_file": str(resolved_path),
            "validation": {
                "ok": False,
                "errors": [f"unable to load scheduler manifest: {exc}"],
                "warnings": [],
                "job_count": 0,
                "ready_for_review_job_count": 0,
                "blocked_job_count": 0,
                "invalid_job_count": 0,
            },
        }
    return {
        "scheduler_manifest_file": str(resolved_path),
        "timezone": str(payload.get("timezone", "") or ""),
        "validation": validate_monitoring_scheduler_manifest(payload),
    }


def write_monitoring_scheduler_manifest(
    workspace_root: Path,
    *,
    recursive: bool = False,
    timezone: str = "UTC",
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a platform-neutral monitoring scheduler manifest."""

    resolved_workspace = workspace_root.resolve()
    payload = build_monitoring_scheduler_manifest(
        resolved_workspace,
        recursive=recursive,
        timezone=timezone,
    )
    target_path = output_path or resolved_workspace / "assets" / _TEMPLATE_DIR_NAME / "monitoring-scheduler.json"
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return target_path


def write_monitoring_execution_plan(
    bundle_path: Path,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a dry-run monitoring execution plan next to its bundle."""

    resolved_bundle_path = bundle_path.resolve()
    payload = build_monitoring_execution_plan(resolved_bundle_path)
    template = str(payload["template"])
    target_path = output_path or resolved_bundle_path.parent / f"{template}.monitoring-plan.json"
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return target_path


def write_research_template_bundle_descriptor(
    name: str,
    *,
    workspace_root: Path,
    template_file: Path,
    workbook_file: Path,
    progress_report_file: Path | None = None,
    rules_file: Path,
    source_map_file: Path,
    manifest_file: Path,
    guide_file: Path,
    monitoring_validation: dict[str, object],
    research_target: dict[str, str] | None = None,
    source_write_manifest: dict[str, object] | None = None,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a machine-readable descriptor for one materialized bundle."""

    normalized = _normalize_template_name(name)
    payload = build_research_template_bundle_descriptor(
        normalized,
        template_file=template_file,
        workbook_file=workbook_file,
        progress_report_file=progress_report_file,
        rules_file=rules_file,
        source_map_file=source_map_file,
        manifest_file=manifest_file,
        guide_file=guide_file,
        monitoring_validation=monitoring_validation,
        research_target=research_target,
        source_write_manifest=source_write_manifest,
    )
    target_path = output_path or workspace_root / "assets" / _TEMPLATE_DIR_NAME / f"{normalized}.bundle.json"
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return target_path


def _materialization_artifact_paths(workspace_root: Path, template: str) -> tuple[Path, ...]:
    artifact_dir = workspace_root.resolve() / "assets" / _TEMPLATE_DIR_NAME
    template_file = (
        artifact_dir / "common.md"
        if template == _FALLBACK_TEMPLATE_NAME
        else artifact_dir / f"common-plus-{template}.md"
    )
    return (
        template_file,
        artifact_dir / f"{template}.research-workbook.json",
        artifact_dir / f"{template}.research-progress.md",
        artifact_dir / f"{template}.monitoring-rules.json",
        artifact_dir / f"{template}.source-map.json",
        artifact_dir / "research-template.manifest.json",
        artifact_dir / f"{template}.research-guide.md",
        artifact_dir / f"{template}.bundle.json",
        artifact_dir / f"{template}.monitoring-plan.json",
        artifact_dir / "monitoring-status.json",
        artifact_dir / "research-workbook-status.json",
        artifact_dir / "research-workbook-report-status.json",
    )


def _snapshot_materialization_artifacts(paths: Iterable[Path]) -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in paths}


def _rollback_materialization_artifacts(snapshot: dict[Path, bytes | None]) -> list[str]:
    errors: list[str] = []
    for path, original_bytes in reversed(snapshot.items()):
        try:
            if original_bytes is None:
                if path.exists():
                    path.unlink()
                continue
            if not path.is_file() or path.read_bytes() != original_bytes:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(original_bytes)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    return errors


def materialize_research_template_bundle(
    name: str,
    *,
    workspace_root: Path,
    ticker: str = "",
    company: str = "",
    write_manifest_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Materialize all local artifacts needed to use one research template."""

    normalized = _normalize_template_name(name)
    workspace_root = workspace_root.resolve()
    research_target = _normalize_research_target(ticker=ticker, company=company)
    snapshot = _snapshot_materialization_artifacts(_materialization_artifact_paths(workspace_root, normalized))
    try:
        if normalized == _FALLBACK_TEMPLATE_NAME:
            template_path = copy_research_template(normalized, workspace_root=workspace_root, overwrite=overwrite)
        else:
            template_path = compose_research_template(normalized, workspace_root=workspace_root, overwrite=overwrite)
        workbook_path = write_research_workbook_payload(
            normalized,
            workspace_root=workspace_root,
            ticker=research_target["ticker"],
            company=research_target["company"],
            overwrite=overwrite,
        )
        progress_report_path = write_research_workbook_report(
            workbook_path,
            overwrite=overwrite,
        )
        rules_path = write_monitoring_rules_payload(normalized, workspace_root=workspace_root, overwrite=overwrite)
        source_map_path = write_monitoring_source_map_payload(
            normalized, workspace_root=workspace_root, overwrite=overwrite
        )
        manifest_path = write_research_template_package_manifest(workspace_root=workspace_root, overwrite=True)
        guide_path = write_research_template_usage_guide(
            normalized,
            workspace_root=workspace_root,
            template_file=template_path,
            workbook_file=workbook_path,
            progress_report_file=progress_report_path,
            rules_file=rules_path,
            source_map_file=source_map_path,
            manifest_file=manifest_path,
            ticker=research_target["ticker"],
            company=research_target["company"],
            overwrite=overwrite,
        )
        validation = validate_monitoring_source_map_payload(
            _load_json_object(rules_path),
            _load_json_object(source_map_path),
        )
        source_write_manifest = (
            _build_source_write_manifest_binding(
                write_manifest_path.resolve(),
                expected_template=normalized,
            )
            if write_manifest_path is not None
            else None
        )
        bundle_path = write_research_template_bundle_descriptor(
            normalized,
            workspace_root=workspace_root,
            template_file=template_path,
            workbook_file=workbook_path,
            progress_report_file=progress_report_path,
            rules_file=rules_path,
            source_map_file=source_map_path,
            manifest_file=manifest_path,
            guide_file=guide_path,
            monitoring_validation=validation,
            research_target=research_target,
            source_write_manifest=source_write_manifest,
            overwrite=overwrite,
        )
        bundle_validation = validate_research_template_bundle_descriptor(_load_json_object(bundle_path))
        if not bool(bundle_validation.get("ok")):
            raise ValueError(f"materialized bundle failed validation: {bundle_validation['errors']}")
    except BaseException as exc:
        rollback_errors = _rollback_materialization_artifacts(snapshot)
        if rollback_errors:
            raise RuntimeError(
                f"research materialization failed ({exc}); rollback also failed: {rollback_errors}"
            ) from exc
        raise
    return {
        "template": normalized,
        "research_target": research_target,
        "template_file": str(template_path),
        "workbook_file": str(workbook_path),
        "progress_report_file": str(progress_report_path),
        "rules_file": str(rules_path),
        "source_map_file": str(source_map_path),
        "manifest_file": str(manifest_path),
        "guide_file": str(guide_path),
        "bundle_file": str(bundle_path),
        "validation": validation,
        "bundle_validation": bundle_validation,
        "source_write_manifest": source_write_manifest,
    }


def materialize_research_workspace(
    name: str,
    *,
    workspace_root: Path,
    ticker: str = "",
    company: str = "",
    write_manifest_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Materialize a complete local research workspace with a dry-run plan."""

    normalized = _normalize_template_name(name)
    resolved_workspace = workspace_root.resolve()
    snapshot = _snapshot_materialization_artifacts(_materialization_artifact_paths(resolved_workspace, normalized))
    try:
        payload = materialize_research_template_bundle(
            normalized,
            workspace_root=resolved_workspace,
            ticker=ticker,
            company=company,
            write_manifest_path=write_manifest_path,
            overwrite=overwrite,
        )
        plan_path = write_monitoring_execution_plan(
            Path(str(payload["bundle_file"])),
            overwrite=overwrite,
        )
        plan_inspection = inspect_monitoring_execution_plan(plan_path)
        plan_validation = plan_inspection.get("validation")
        if not isinstance(plan_validation, dict) or plan_validation.get("ok") is not True:
            raise ValueError(f"materialized monitoring plan failed validation: {plan_validation}")
        monitoring_status_path = write_monitoring_status_snapshot(
            resolved_workspace,
            overwrite=True,
        )
        workbook_status_path = write_research_workbook_status_snapshot(
            resolved_workspace,
            overwrite=True,
        )
        report_status_path = write_research_workbook_report_status_snapshot(
            resolved_workspace,
            overwrite=True,
        )
        monitoring_status = _load_json_object(monitoring_status_path)
        workbook_status = _load_json_object(workbook_status_path)
        report_status = _load_json_object(report_status_path)
        status_failures = {
            "monitoring": monitoring_status.get("overall_status"),
            "workbook": workbook_status.get("overall_status"),
            "report": report_status.get("overall_status"),
        }
        if any(
            status in {"unhealthy", "no_plans", "no_workbooks", "no_reports"} for status in status_failures.values()
        ):
            raise ValueError(f"materialized workspace status is incomplete or unhealthy: {status_failures}")
        target = payload["research_target"]
        assert isinstance(target, dict)
        guide_path = write_research_template_usage_guide(
            normalized,
            workspace_root=resolved_workspace,
            template_file=Path(str(payload["template_file"])),
            workbook_file=Path(str(payload["workbook_file"])),
            progress_report_file=Path(str(payload["progress_report_file"])),
            rules_file=Path(str(payload["rules_file"])),
            source_map_file=Path(str(payload["source_map_file"])),
            manifest_file=Path(str(payload["manifest_file"])),
            monitoring_plan_file=plan_path,
            monitoring_status_file=monitoring_status_path,
            workbook_status_file=workbook_status_path,
            report_status_file=report_status_path,
            ticker=str(target.get("ticker", "")),
            company=str(target.get("company", "")),
            overwrite=True,
        )
        bundle_validation = validate_research_template_bundle_descriptor(
            _load_json_object(Path(str(payload["bundle_file"])))
        )
        if bundle_validation.get("ok") is not True:
            raise ValueError(f"materialized workspace bundle failed validation: {bundle_validation['errors']}")
        payload.update(
            {
                "guide_file": str(guide_path),
                "monitoring_plan_file": str(plan_path),
                "monitoring_plan_validation": plan_validation,
                "monitoring_status_file": str(monitoring_status_path),
                "monitoring_status": monitoring_status,
                "workbook_status_file": str(workbook_status_path),
                "workbook_status": workbook_status,
                "report_status_file": str(report_status_path),
                "report_status": report_status,
                "bundle_validation": bundle_validation,
            }
        )
        return payload
    except BaseException as exc:
        rollback_errors = _rollback_materialization_artifacts(snapshot)
        if rollback_errors:
            raise RuntimeError(
                f"research workspace materialization failed ({exc}); rollback also failed: {rollback_errors}"
            ) from exc
        raise


def build_research_workspace_refresh_preview(bundle_path: Path) -> dict[str, object]:
    """Preview whether derived workspace artifacts can be safely refreshed."""

    resolved_bundle = bundle_path.resolve()
    inspection = inspect_research_template_bundle(resolved_bundle)
    payload = _load_json_object(resolved_bundle)
    template = str(payload.get("template", "") or "")
    artifacts = payload.get("artifacts")
    blockers: list[str] = []
    if not template:
        blockers.append("bundle template is missing")
    if not isinstance(artifacts, dict):
        blockers.append("bundle artifacts must be an object")
        artifacts = {}
    workbook_raw = artifacts.get("research_workbook")
    if not isinstance(workbook_raw, str) or not Path(workbook_raw).is_file():
        blockers.append("bundle research workbook is missing")
    else:
        try:
            workbook_validation = validate_research_workbook_payload(_load_json_object(Path(workbook_raw)))
            if workbook_validation.get("ok") is not True:
                blockers.append(f"bundle research workbook is invalid: {workbook_validation.get('errors', [])}")
        except (OSError, ValueError) as exc:
            blockers.append(f"bundle research workbook is invalid: {exc}")
    validation = inspection.get("validation")
    validation_errors = validation.get("errors", []) if isinstance(validation, dict) else []
    if isinstance(validation_errors, list):
        blockers.extend(
            str(error) for error in validation_errors if not str(error).startswith("artifacts.research_progress_report")
        )
    artifact_dir = resolved_bundle.parent
    derived_paths = {
        "research_progress_report": artifact_dir / f"{template}.research-progress.md",
        "monitoring_plan": artifact_dir / f"{template}.monitoring-plan.json",
        "monitoring_status": artifact_dir / "monitoring-status.json",
        "workbook_status": artifact_dir / "research-workbook-status.json",
        "report_status": artifact_dir / "research-workbook-report-status.json",
        "usage_guide": Path(str(artifacts.get("usage_guide", artifact_dir / f"{template}.research-guide.md"))),
    }
    return {
        "schema_version": 1,
        "preview_type": "research_workspace_refresh",
        "bundle_file": str(resolved_bundle),
        "template": template,
        "can_refresh": not blockers,
        "blockers": blockers,
        "current_bundle_validation": validation,
        "outputs": {
            key: {"path": str(path.resolve()), "action": "refresh" if path.exists() else "create"}
            for key, path in derived_paths.items()
        },
    }


def write_research_workspace_refresh(bundle_path: Path) -> dict[str, object]:
    """Refresh all derived workspace artifacts under one rollback boundary."""

    preview = build_research_workspace_refresh_preview(bundle_path)
    if preview.get("can_refresh") is not True:
        raise ValueError(f"research workspace cannot be refreshed: {preview['blockers']}")
    resolved_bundle = Path(str(preview["bundle_file"]))
    payload = _load_json_object(resolved_bundle)
    template = str(payload["template"])
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    outputs = preview["outputs"]
    assert isinstance(outputs, dict)
    output_paths = {key: Path(str(value["path"])) for key, value in outputs.items() if isinstance(value, dict)}
    snapshot = _snapshot_materialization_artifacts(output_paths.values())
    try:
        workbook_path = Path(str(artifacts["research_workbook"]))
        report_path = write_research_workbook_report(
            workbook_path,
            output_path=output_paths["research_progress_report"],
            overwrite=True,
        )
        bundle_validation = validate_research_template_bundle_descriptor(payload)
        if bundle_validation.get("ok") is not True:
            raise ValueError(f"bundle remains unhealthy after report refresh: {bundle_validation['errors']}")
        plan_path = write_monitoring_execution_plan(
            resolved_bundle,
            output_path=output_paths["monitoring_plan"],
            overwrite=True,
        )
        plan_inspection = inspect_monitoring_execution_plan(plan_path)
        plan_validation = plan_inspection.get("validation")
        if not isinstance(plan_validation, dict) or plan_validation.get("ok") is not True:
            raise ValueError(f"refreshed monitoring plan failed validation: {plan_validation}")
        workspace_root = resolved_bundle.parent.parent.parent
        # Write the status snapshots to the exact paths the preview computed
        # (derived from resolved_bundle.parent), so they stay inside the snapshot
        # rollback boundary even when the bundle sits at a non-canonical depth.
        # workspace_root still drives the scan for aggregation.
        monitoring_status_path = write_monitoring_status_snapshot(
            workspace_root,
            output_path=output_paths["monitoring_status"],
            overwrite=True,
        )
        workbook_status_path = write_research_workbook_status_snapshot(
            workspace_root,
            output_path=output_paths["workbook_status"],
            overwrite=True,
        )
        report_status_path = write_research_workbook_report_status_snapshot(
            workspace_root,
            output_path=output_paths["report_status"],
            overwrite=True,
        )
        monitoring_status = _load_json_object(monitoring_status_path)
        workbook_status = _load_json_object(workbook_status_path)
        report_status = _load_json_object(report_status_path)
        statuses = {
            "monitoring": monitoring_status,
            "workbook": workbook_status,
            "report": report_status,
        }
        if any(
            status.get("overall_status") in {"unhealthy", "no_plans", "no_workbooks", "no_reports"}
            for status in statuses.values()
        ):
            raise ValueError("refreshed workspace status is incomplete or unhealthy")
        target = payload.get("research_target")
        target = target if isinstance(target, dict) else {}
        guide_path = write_research_template_usage_guide(
            template,
            workspace_root=workspace_root,
            template_file=Path(str(artifacts["write_template"])),
            workbook_file=workbook_path,
            progress_report_file=report_path,
            rules_file=Path(str(artifacts["monitoring_rules"])),
            source_map_file=Path(str(artifacts["source_map"])),
            manifest_file=Path(str(artifacts["package_manifest"])),
            monitoring_plan_file=plan_path,
            monitoring_status_file=monitoring_status_path,
            workbook_status_file=workbook_status_path,
            report_status_file=report_status_path,
            ticker=str(target.get("ticker", "")),
            company=str(target.get("company", "")),
            output_path=output_paths["usage_guide"],
            overwrite=True,
        )
        return {
            **preview,
            "applied": True,
            "bundle_validation": bundle_validation,
            "monitoring_plan_validation": plan_validation,
            "monitoring_status": monitoring_status,
            "workbook_status": workbook_status,
            "report_status": report_status,
            "guide_file": str(guide_path),
        }
    except BaseException as exc:
        rollback_errors = _rollback_materialization_artifacts(snapshot)
        if rollback_errors:
            raise RuntimeError(
                f"research workspace refresh failed ({exc}); rollback also failed: {rollback_errors}"
            ) from exc
        raise


def materialize_research_bundle_from_write_manifest(
    manifest_path: Path,
    *,
    workspace_root: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Materialize a research bundle from the completed write-manifest truth."""

    resolved_manifest = manifest_path.resolve()
    template, selection = _resolve_template_selection_from_write_manifest(resolved_manifest)
    target = _resolve_materialize_research_target(
        ticker_raw=None,
        company_raw=None,
        manifest_raw=str(resolved_manifest),
    )
    payload = materialize_research_workspace(
        template,
        workspace_root=workspace_root.resolve(),
        ticker=target["ticker"],
        company=target["company"],
        write_manifest_path=resolved_manifest,
        overwrite=overwrite,
    )
    payload["selection"] = selection
    return payload


def materialize_research_portfolio(
    portfolio_path: Path,
    *,
    workspace_root: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Materialize isolated research bundles and dry-run plans for a portfolio."""

    resolved_portfolio_path = portfolio_path.resolve()
    resolved_workspace = workspace_root.resolve()
    targets = _prepare_research_portfolio_targets(resolved_portfolio_path)
    preview = _build_research_portfolio_preview(
        resolved_portfolio_path,
        resolved_workspace,
        targets,
        overwrite=overwrite,
    )
    preview_targets = preview.get("targets")
    preview_by_ticker: dict[str, dict[str, object]] = {}
    if isinstance(preview_targets, list):
        for item in preview_targets:
            if isinstance(item, dict):
                preview_by_ticker[str(item.get("ticker", ""))] = item
    results: list[dict[str, object]] = []
    for target in targets:
        ticker = str(target["ticker"])
        company = str(target["company"])
        template = str(target["template"])
        target_workspace = resolved_workspace / str(target["workspace_key"])
        result: dict[str, object] = {
            "ticker": ticker,
            "company": company,
            "template": template,
            "workspace_root": str(target_workspace),
            "selection": target["selection"],
            "write_manifest": target["write_manifest"],
        }
        target_preview = preview_by_ticker[ticker]
        if target_preview.get("action") == "blocked_existing_files":
            existing_files = target_preview.get("existing_files", [])
            result.update(
                {
                    "status": "failed",
                    "error": "existing generated files require --overwrite",
                    "existing_files": existing_files,
                }
            )
            results.append(result)
            continue
        try:
            bundle = materialize_research_workspace(
                template,
                workspace_root=target_workspace,
                ticker=ticker,
                company=company,
                write_manifest_path=(
                    Path(str(target["write_manifest"]))
                    if target["write_manifest"] is not None
                    and isinstance(target["selection"], dict)
                    and target["selection"].get("selection_mode") != "explicit"
                    else None
                ),
                overwrite=overwrite,
            )
            result.update(
                {
                    "status": "success",
                    "bundle_file": bundle["bundle_file"],
                    "workbook_file": bundle["workbook_file"],
                    "monitoring_plan_file": bundle["monitoring_plan_file"],
                    "monitoring_status_file": bundle["monitoring_status_file"],
                    "workbook_status_file": bundle["workbook_status_file"],
                    "report_status_file": bundle["report_status_file"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - isolate one target's failure into the batch report
            # materialize_research_workspace raises OSError/ValueError on normal
            # failures but RuntimeError (rollback-also-failed) or AssertionError
            # on degenerate paths. Recording any of them keeps the batch report
            # honest instead of aborting the whole portfolio with a traceback.
            result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        results.append(result)

    status_path = write_monitoring_status_snapshot(
        resolved_workspace,
        recursive=True,
        overwrite=True,
    )
    status_payload = _load_json_object(status_path)
    workbook_status_path = write_research_workbook_status_snapshot(
        resolved_workspace,
        recursive=True,
        overwrite=True,
    )
    workbook_status_payload = _load_json_object(workbook_status_path)
    report_status_path = write_research_workbook_report_status_snapshot(
        resolved_workspace,
        recursive=True,
        overwrite=True,
    )
    report_status_payload = _load_json_object(report_status_path)
    success_count = sum(result.get("status") == "success" for result in results)
    failure_count = len(results) - success_count
    report_path = resolved_workspace / "research-portfolio.materialization.json"
    report = {
        "schema_version": 1,
        "report_type": "research_portfolio_materialization",
        "portfolio_file": str(resolved_portfolio_path),
        "portfolio_fingerprint": _sha256_file(resolved_portfolio_path),
        "workspace_root": str(resolved_workspace),
        "report_file": str(report_path.resolve()),
        "ok": failure_count == 0,
        "target_count": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "monitoring_status_file": str(status_path),
        "monitoring_status": status_payload,
        "workbook_status_file": str(workbook_status_path),
        "workbook_status": workbook_status_payload,
        "report_status_file": str(report_status_path),
        "report_status": report_status_payload,
        "preview": preview,
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def build_research_portfolio_preview(
    portfolio_path: Path,
    *,
    workspace_root: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Preview portfolio materialization without creating or modifying files."""

    resolved_portfolio_path = portfolio_path.resolve()
    targets = _prepare_research_portfolio_targets(resolved_portfolio_path)
    return _build_research_portfolio_preview(
        resolved_portfolio_path,
        workspace_root.resolve(),
        targets,
        overwrite=overwrite,
    )


def _run_list(args: argparse.Namespace) -> int:
    templates = list_research_templates()
    if bool(getattr(args, "json", False)):
        payload = [{"name": template.name, "title": template.title} for template in templates]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for template in templates:
        print(f"{template.name}\t{template.title}")
    return 0


def _run_show(args: argparse.Namespace) -> int:
    print(load_research_template(str(getattr(args, "name"))))
    return 0


def _run_copy(args: argparse.Namespace) -> int:
    output_raw = getattr(args, "output", None)
    copied_path = copy_research_template(
        str(getattr(args, "name")),
        workspace_root=Path(str(getattr(args, "base", "./workspace"))).resolve(),
        output_path=Path(str(output_raw)).resolve() if output_raw else None,
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps({"template_file": str(copied_path)}, ensure_ascii=False, indent=2))
    else:
        print(f"template_file: {copied_path}")
    return 0


def _run_recommend(args: argparse.Namespace) -> int:
    company_facets = _company_facets_from_args(args)
    recommendations = recommend_research_templates(
        company_facets,
        limit=int(getattr(args, "limit", 3)),
    )
    if bool(getattr(args, "json", False)):
        payload = [_recommendation_payload(recommendation) for recommendation in recommendations]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for recommendation in recommendations:
        matched = ", ".join(recommendation.matched_facets) if recommendation.matched_facets else "-"
        print(f"{recommendation.name}\tscore={recommendation.score}\tmatched={matched}\t{recommendation.title}")
    return 0


def _run_compose(args: argparse.Namespace) -> int:
    output_raw = getattr(args, "output", None)
    composed_path = compose_research_template(
        str(getattr(args, "name")),
        workspace_root=Path(str(getattr(args, "base", "./workspace"))).resolve(),
        output_path=Path(str(output_raw)).resolve() if output_raw else None,
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps({"template_file": str(composed_path)}, ensure_ascii=False, indent=2))
    else:
        print(f"template_file: {composed_path}")
    return 0


def _run_monitoring_rules(args: argparse.Namespace) -> int:
    output_raw = getattr(args, "output", None)
    should_write = bool(output_raw) or bool(getattr(args, "write", False))
    if should_write:
        rules_path = write_monitoring_rules_payload(
            str(getattr(args, "name")),
            workspace_root=Path(str(getattr(args, "base", "./workspace"))).resolve(),
            output_path=Path(str(output_raw)).resolve() if output_raw else None,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        print(json.dumps({"rules_file": str(rules_path)}, ensure_ascii=False, indent=2))
        return 0
    payload = build_monitoring_rules_payload(str(getattr(args, "name")))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_research_workbook(args: argparse.Namespace) -> int:
    name = str(getattr(args, "name"))
    ticker = str(getattr(args, "ticker", "") or "")
    company = str(getattr(args, "company", "") or "")
    output_raw = getattr(args, "output", None)
    should_write = bool(output_raw) or bool(getattr(args, "write", False))
    if should_write:
        workbook_path = write_research_workbook_payload(
            name,
            workspace_root=Path(str(getattr(args, "base", "./workspace"))).resolve(),
            ticker=ticker,
            company=company,
            output_path=Path(str(output_raw)).resolve() if output_raw else None,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        print(json.dumps({"workbook_file": str(workbook_path)}, ensure_ascii=False, indent=2))
        return 0
    payload = build_research_workbook_payload(name, ticker=ticker, company=company)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_validate_research_workbook(args: argparse.Namespace) -> int:
    workbook_path = Path(str(getattr(args, "workbook"))).resolve()
    result = validate_research_workbook_payload(_load_json_object(workbook_path))
    print(json.dumps({"workbook_file": str(workbook_path), "validation": result}, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") is True else 1


def _run_update_research_workbook(args: argparse.Namespace) -> int:
    workbook_path = Path(str(getattr(args, "workbook"))).resolve()
    evidence_raw = getattr(args, "evidence_file", None)
    evidence_records = _load_workbook_evidence_records(Path(str(evidence_raw)).resolve()) if evidence_raw else None
    update_kwargs = {
        "item_id": str(getattr(args, "item_id")),
        "status": getattr(args, "status", None),
        "response": getattr(args, "response", None),
        "analyst_notes": getattr(args, "analyst_notes", None),
        "evidence_records": evidence_records,
    }
    if bool(getattr(args, "write", False)):
        result = write_research_workbook_update(workbook_path, **update_kwargs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    preview = build_research_workbook_update_preview(
        _load_json_object(workbook_path),
        **update_kwargs,
    )
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


def _run_rollback_research_workbook(args: argparse.Namespace) -> int:
    workbook_path = Path(str(getattr(args, "workbook"))).resolve()
    backup_path = Path(str(getattr(args, "backup"))).resolve()
    if bool(getattr(args, "write", False)):
        result = write_research_workbook_rollback(workbook_path, backup_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    preview = build_research_workbook_rollback_preview(workbook_path, backup_path)
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


def _run_source_map(args: argparse.Namespace) -> int:
    output_raw = getattr(args, "output", None)
    should_write = bool(output_raw) or bool(getattr(args, "write", False))
    if should_write:
        source_map_path = write_monitoring_source_map_payload(
            str(getattr(args, "name")),
            workspace_root=Path(str(getattr(args, "base", "./workspace"))).resolve(),
            output_path=Path(str(output_raw)).resolve() if output_raw else None,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        print(json.dumps({"source_map_file": str(source_map_path)}, ensure_ascii=False, indent=2))
        return 0
    payload = build_monitoring_source_map_payload(str(getattr(args, "name")))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_validate_source_map(args: argparse.Namespace) -> int:
    rules_payload = _load_json_object(Path(str(getattr(args, "rules"))).resolve())
    source_map_payload = _load_json_object(Path(str(getattr(args, "source_map"))).resolve())
    result = validate_monitoring_source_map_payload(rules_payload, source_map_payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_package_manifest(args: argparse.Namespace) -> int:
    output_raw = getattr(args, "output", None)
    should_write = bool(output_raw) or bool(getattr(args, "write", False))
    if should_write:
        manifest_path = write_research_template_package_manifest(
            workspace_root=Path(str(getattr(args, "base", "./workspace"))).resolve(),
            output_path=Path(str(output_raw)).resolve() if output_raw else None,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        print(json.dumps({"manifest_file": str(manifest_path)}, ensure_ascii=False, indent=2))
        return 0
    payload = build_research_template_package_manifest()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_materialize(args: argparse.Namespace) -> int:
    template_name, selection_payload = _resolve_materialize_template_selection(
        getattr(args, "name", None),
        getattr(args, "manifest", None),
    )
    research_target = _resolve_materialize_research_target(
        ticker_raw=getattr(args, "ticker", None),
        company_raw=getattr(args, "company", None),
        manifest_raw=getattr(args, "manifest", None),
    )
    payload = materialize_research_workspace(
        template_name,
        workspace_root=Path(str(getattr(args, "base", "./workspace"))).resolve(),
        ticker=research_target["ticker"],
        company=research_target["company"],
        write_manifest_path=(
            Path(str(getattr(args, "manifest"))).resolve()
            if getattr(args, "manifest", None) and selection_payload.get("selection_mode") != "explicit"
            else None
        ),
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    payload["selection"] = selection_payload
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_refresh_workspace(args: argparse.Namespace) -> int:
    bundle_path = Path(str(getattr(args, "bundle"))).resolve()
    if bool(getattr(args, "write", False)):
        result = write_research_workspace_refresh(bundle_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    preview = build_research_workspace_refresh_preview(bundle_path)
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0 if preview.get("can_refresh") is True else 1


def _run_list_bundles(args: argparse.Namespace) -> int:
    bundles = discover_research_template_bundles(
        Path(str(getattr(args, "base", "./workspace"))).resolve(),
        recursive=bool(getattr(args, "recursive", False)),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(bundles, ensure_ascii=False, indent=2))
        return 0
    for bundle in bundles:
        validation = bundle.get("validation")
        is_valid = isinstance(validation, dict) and validation.get("ok") is True
        template = str(bundle.get("template", "") or "<invalid>")
        research_target = bundle.get("research_target")
        ticker = str(research_target.get("ticker", "") or "-") if isinstance(research_target, dict) else "-"
        print(f"{ticker}\t{template}\t{'valid' if is_valid else 'invalid'}\t{bundle['descriptor_file']}")
    return 0


def _run_validate_bundle(args: argparse.Namespace) -> int:
    result = inspect_research_template_bundle(Path(str(getattr(args, "bundle"))).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    validation = result.get("validation")
    return 0 if isinstance(validation, dict) and validation.get("ok") is True else 1


def _run_rebind_bundle(args: argparse.Namespace) -> int:
    bundle_path = Path(str(getattr(args, "bundle"))).resolve()
    result = (
        write_research_template_bundle_rebind(bundle_path)
        if bool(getattr(args, "write", False))
        else build_research_template_bundle_rebind_preview(bundle_path)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_rollback_bundle_rebind(args: argparse.Namespace) -> int:
    bundle_path = Path(str(getattr(args, "bundle"))).resolve()
    backup_path = Path(str(getattr(args, "backup"))).resolve()
    result = (
        write_research_template_bundle_rebind_rollback(bundle_path, backup_path)
        if bool(getattr(args, "write", False))
        else build_research_template_bundle_rebind_rollback_preview(bundle_path, backup_path)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_monitoring_plan(args: argparse.Namespace) -> int:
    bundle_path = Path(str(getattr(args, "bundle"))).resolve()
    output_raw = getattr(args, "output", None)
    should_write = bool(output_raw) or bool(getattr(args, "write", False))
    if should_write:
        plan_path = write_monitoring_execution_plan(
            bundle_path,
            output_path=Path(str(output_raw)).resolve() if output_raw else None,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        payload = _load_json_object(plan_path)
        print(json.dumps({"monitoring_plan_file": str(plan_path), "plan": payload}, ensure_ascii=False, indent=2))
        return 0
    payload = build_monitoring_execution_plan(bundle_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _run_validate_monitoring_plan(args: argparse.Namespace) -> int:
    result = inspect_monitoring_execution_plan(Path(str(getattr(args, "plan"))).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    validation = result.get("validation")
    return 0 if isinstance(validation, dict) and validation.get("ok") is True else 1


def _run_list_monitoring_plans(args: argparse.Namespace) -> int:
    plans = discover_monitoring_execution_plans(
        Path(str(getattr(args, "base", "./workspace"))).resolve(),
        recursive=bool(getattr(args, "recursive", False)),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(plans, ensure_ascii=False, indent=2))
        return 0
    for plan in plans:
        validation = plan.get("validation")
        is_valid = isinstance(validation, dict) and validation.get("ok") is True
        template = str(plan.get("template", "") or "<invalid>")
        research_target = plan.get("research_target")
        ticker = str(research_target.get("ticker", "") or "-") if isinstance(research_target, dict) else "-"
        readiness = plan.get("readiness")
        status = str(readiness.get("status", "") or "unknown") if isinstance(readiness, dict) else "unknown"
        print(f"{ticker}\t{template}\t{'valid' if is_valid else 'invalid'}\t{status}\t{plan['monitoring_plan_file']}")
    return 0


def _run_monitoring_status(args: argparse.Namespace) -> int:
    workspace_root = Path(str(getattr(args, "base", "./workspace"))).resolve()
    recursive = bool(getattr(args, "recursive", False))
    output_raw = getattr(args, "output", None)
    should_write = bool(output_raw) or bool(getattr(args, "write", False))
    if should_write:
        status_path = write_monitoring_status_snapshot(
            workspace_root,
            output_path=Path(str(output_raw)).resolve() if output_raw else None,
            recursive=recursive,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        payload = _load_json_object(status_path)
        print(json.dumps({"monitoring_status_file": str(status_path), "status": payload}, ensure_ascii=False, indent=2))
        return 0
    print(
        json.dumps(build_monitoring_status_snapshot(workspace_root, recursive=recursive), ensure_ascii=False, indent=2)
    )
    return 0


def _run_workbook_status(args: argparse.Namespace) -> int:
    workspace_root = Path(str(getattr(args, "base", "./workspace"))).resolve()
    recursive = bool(getattr(args, "recursive", False))
    output_raw = getattr(args, "output", None)
    should_write = bool(output_raw) or bool(getattr(args, "write", False))
    if should_write:
        status_path = write_research_workbook_status_snapshot(
            workspace_root,
            output_path=Path(str(output_raw)).resolve() if output_raw else None,
            recursive=recursive,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        payload = _load_json_object(status_path)
        print(json.dumps({"workbook_status_file": str(status_path), "status": payload}, ensure_ascii=False, indent=2))
        return 0
    print(
        json.dumps(
            build_research_workbook_status_snapshot(workspace_root, recursive=recursive),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_workbook_report(args: argparse.Namespace) -> int:
    workbook_path = Path(str(getattr(args, "workbook"))).resolve()
    output_raw = getattr(args, "output", None)
    should_write = bool(output_raw) or bool(getattr(args, "write", False))
    if should_write:
        report_path = write_research_workbook_report(
            workbook_path,
            output_path=Path(str(output_raw)).resolve() if output_raw else None,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        print(json.dumps({"workbook_report_file": str(report_path)}, ensure_ascii=False, indent=2))
        return 0
    print(build_research_workbook_report(_load_json_object(workbook_path)), end="")
    return 0


def _run_validate_workbook_report(args: argparse.Namespace) -> int:
    result = inspect_research_workbook_report(
        Path(str(getattr(args, "report"))).resolve(),
        Path(str(getattr(args, "workbook"))).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    validation = result.get("validation")
    return 0 if isinstance(validation, dict) and validation.get("ok") is True else 1


def _run_workbook_report_status(args: argparse.Namespace) -> int:
    workspace_root = Path(str(getattr(args, "base", "./workspace"))).resolve()
    recursive = bool(getattr(args, "recursive", False))
    output_raw = getattr(args, "output", None)
    should_write = bool(output_raw) or bool(getattr(args, "write", False))
    if should_write:
        status_path = write_research_workbook_report_status_snapshot(
            workspace_root,
            output_path=Path(str(output_raw)).resolve() if output_raw else None,
            recursive=recursive,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        payload = _load_json_object(status_path)
        print(
            json.dumps(
                {"workbook_report_status_file": str(status_path), "status": payload},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    print(
        json.dumps(
            build_research_workbook_report_status_snapshot(workspace_root, recursive=recursive),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_materialize_portfolio(args: argparse.Namespace) -> int:
    report = materialize_research_portfolio(
        Path(str(getattr(args, "portfolio"))).resolve(),
        workspace_root=Path(str(getattr(args, "base", "./workspace"))).resolve(),
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok") is True else 1


def _run_preview_portfolio(args: argparse.Namespace) -> int:
    preview = build_research_portfolio_preview(
        Path(str(getattr(args, "portfolio"))).resolve(),
        workspace_root=Path(str(getattr(args, "base", "./workspace"))).resolve(),
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0 if preview.get("can_materialize") is True else 1


def _run_scheduler_manifest(args: argparse.Namespace) -> int:
    workspace_root = Path(str(getattr(args, "base", "./workspace"))).resolve()
    recursive = bool(getattr(args, "recursive", False))
    timezone = str(getattr(args, "timezone", "UTC"))
    output_raw = getattr(args, "output", None)
    should_write = bool(output_raw) or bool(getattr(args, "write", False))
    if should_write:
        schedule_path = write_monitoring_scheduler_manifest(
            workspace_root,
            recursive=recursive,
            timezone=timezone,
            output_path=Path(str(output_raw)).resolve() if output_raw else None,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
        payload = _load_json_object(schedule_path)
        print(
            json.dumps(
                {"scheduler_manifest_file": str(schedule_path), "manifest": payload}, ensure_ascii=False, indent=2
            )
        )
        return 0
    print(
        json.dumps(
            build_monitoring_scheduler_manifest(workspace_root, recursive=recursive, timezone=timezone),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_validate_scheduler_manifest(args: argparse.Namespace) -> int:
    result = inspect_monitoring_scheduler_manifest(Path(str(getattr(args, "manifest"))).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    validation = result.get("validation")
    return 0 if isinstance(validation, dict) and validation.get("ok") is True else 1


def _run_source_bindings(args: argparse.Namespace) -> int:
    source_map_path = Path(str(getattr(args, "source_map"))).resolve()
    approval_path = Path(str(getattr(args, "approval"))).resolve()
    if bool(getattr(args, "write", False)):
        result = write_monitoring_source_binding_approval(source_map_path, approval_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    preview = build_monitoring_source_binding_preview(
        _load_json_object(source_map_path),
        _load_json_object(approval_path),
    )
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


def _run_rollback_source_bindings(args: argparse.Namespace) -> int:
    source_map_path = Path(str(getattr(args, "source_map"))).resolve()
    backup_path = Path(str(getattr(args, "backup"))).resolve()
    if bool(getattr(args, "write", False)):
        result = write_monitoring_source_binding_rollback(source_map_path, backup_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    preview = build_monitoring_source_binding_rollback_preview(source_map_path, backup_path)
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


def _run_source_binding_history(args: argparse.Namespace) -> int:
    result = inspect_monitoring_source_binding_history(Path(str(getattr(args, "source_map"))).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    validation = result.get("validation")
    return 0 if isinstance(validation, dict) and validation.get("ok") is True else 1


def _resolve_materialize_template_selection(name_raw: object, manifest_raw: object) -> tuple[str, dict[str, object]]:
    explicit_name = str(name_raw).strip() if name_raw is not None else ""
    if explicit_name:
        template_name = _normalize_template_name(explicit_name)
        return template_name, {
            "selection_mode": "explicit",
            "selected_template": template_name,
        }

    if manifest_raw is None or not str(manifest_raw).strip():
        raise ValueError("materialize requires a template name or --manifest")

    manifest_path = Path(str(manifest_raw)).resolve()
    return _resolve_template_selection_from_write_manifest(manifest_path)


def _resolve_template_selection_from_write_manifest(manifest_path: Path) -> tuple[str, dict[str, object]]:
    payload = _load_json_object(manifest_path)
    config = payload.get("config")
    config_payload = config if isinstance(config, dict) else {}
    requested_name = str(config_payload.get("research_template_requested_name", "") or "").strip().lower()
    resolved_name = str(config_payload.get("research_template_resolved_name", "") or "").strip().lower()
    selection_mode = str(config_payload.get("research_template_selection_mode", "") or "").strip().lower()
    provenance_values = (requested_name, resolved_name, selection_mode)
    if any(provenance_values):
        if not all(provenance_values):
            raise ValueError(f"write manifest contains incomplete research-template provenance: {manifest_path}")
        resolved_name = _normalize_template_name(resolved_name)
        _resolve_template_path(resolved_name)
        if selection_mode == "named" and requested_name != resolved_name:
            raise ValueError(f"write manifest named research-template provenance is inconsistent: {manifest_path}")
        if selection_mode == "auto" and requested_name != "auto":
            raise ValueError(f"write manifest auto research-template provenance is inconsistent: {manifest_path}")
        if selection_mode not in {"named", "auto"}:
            raise ValueError(f"write manifest research-template selection mode is unsupported: {selection_mode!r}")
        return resolved_name, {
            "selection_mode": "manifest_provenance",
            "selected_template": resolved_name,
            "manifest_file": str(manifest_path),
            "write_selection": {
                "requested_name": requested_name,
                "resolved_name": resolved_name,
                "selection_mode": selection_mode,
            },
        }

    company_facets = _load_company_facets_from_manifest(manifest_path)
    recommendation = recommend_research_templates(company_facets, limit=1)[0]
    return recommendation.name, {
        "selection_mode": "manifest_recommendation",
        "selected_template": recommendation.name,
        "manifest_file": str(manifest_path),
        "recommendation": _recommendation_payload(recommendation),
    }


def _build_source_write_manifest_binding(
    manifest_path: Path,
    *,
    expected_template: str,
) -> dict[str, object]:
    selected_template, selection = _resolve_template_selection_from_write_manifest(manifest_path)
    if selected_template != expected_template:
        raise ValueError(
            "write manifest selected template does not match materialized bundle: "
            f"source={selected_template!r} bundle={expected_template!r}"
        )
    source_payload = _load_json_object(manifest_path)
    source_config = source_payload.get("config")
    source_target = {"ticker": "", "company": ""}
    if isinstance(source_config, dict):
        source_target = _normalize_research_target(
            ticker=str(source_config.get("ticker", "") or ""),
            company=str(source_config.get("company", "") or ""),
        )
    return {
        "path": str(manifest_path),
        "file_fingerprint": _sha256_file(manifest_path),
        "semantic_fingerprint": _sha256_json_object(_build_write_manifest_binding_semantics(manifest_path)),
        "selected_template": selected_template,
        "selection": selection,
        "research_target": source_target,
    }


def _build_write_manifest_binding_semantics(manifest_path: Path) -> dict[str, object]:
    selected_template, selection = _resolve_template_selection_from_write_manifest(manifest_path)
    source_payload = _load_json_object(manifest_path)
    source_config = source_payload.get("config")
    source_target = {"ticker": "", "company": ""}
    if isinstance(source_config, dict):
        source_target = _normalize_research_target(
            ticker=str(source_config.get("ticker", "") or ""),
            company=str(source_config.get("company", "") or ""),
        )
    company_facets = _load_company_facets_from_manifest(manifest_path)
    return {
        "schema_version": 1,
        "selected_template": selected_template,
        "selection_mode": str(selection.get("selection_mode", "") or ""),
        "research_target": source_target,
        "company_facets": company_facets.to_dict(),
    }


def _resolve_materialize_research_target(
    *,
    ticker_raw: object,
    company_raw: object,
    manifest_raw: object,
) -> dict[str, str]:
    manifest_target = {"ticker": "", "company": ""}
    if manifest_raw is not None and str(manifest_raw).strip():
        manifest_payload = _load_json_object(Path(str(manifest_raw)).resolve())
        config = manifest_payload.get("config")
        if isinstance(config, dict):
            manifest_target = _normalize_research_target(
                ticker=str(config.get("ticker", "") or ""),
                company=str(config.get("company", "") or ""),
            )
    explicit_ticker = str(ticker_raw).strip() if ticker_raw is not None else ""
    explicit_company = str(company_raw).strip() if company_raw is not None else ""
    return _normalize_research_target(
        ticker=explicit_ticker or manifest_target["ticker"],
        company=explicit_company or manifest_target["company"],
    )


def _prepare_research_portfolio_targets(portfolio_path: Path) -> list[dict[str, object]]:
    payload = _load_json_object(portfolio_path)
    if payload.get("schema_version") != 1:
        raise ValueError("portfolio schema_version must be 1")
    if payload.get("portfolio_type") != "research_monitoring_portfolio":
        raise ValueError("portfolio_type must be research_monitoring_portfolio")
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("portfolio targets must be a non-empty list")

    prepared: list[dict[str, object]] = []
    seen_tickers: set[str] = set()
    seen_workspace_keys: set[str] = set()
    for index, raw_target in enumerate(raw_targets):
        if not isinstance(raw_target, dict):
            raise ValueError(f"portfolio targets[{index}] must be an object")
        manifest_raw = str(raw_target.get("write_manifest", "") or "").strip()
        manifest_path: Path | None = None
        if manifest_raw:
            candidate = Path(manifest_raw)
            manifest_path = (
                (portfolio_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
            )
            if not manifest_path.is_file():
                raise ValueError(f"portfolio targets[{index}].write_manifest does not exist: {manifest_path}")

        target = _resolve_materialize_research_target(
            ticker_raw=raw_target.get("ticker"),
            company_raw=raw_target.get("company"),
            manifest_raw=str(manifest_path) if manifest_path is not None else None,
        )
        ticker = target["ticker"]
        if not ticker:
            raise ValueError(f"portfolio targets[{index}] requires ticker or write_manifest config.ticker")
        if ticker in seen_tickers:
            raise ValueError(f"duplicate portfolio ticker: {ticker}")
        workspace_key = _identifier_component(ticker).upper()
        if not workspace_key:
            raise ValueError(f"portfolio targets[{index}] ticker cannot form a workspace directory")
        if workspace_key in seen_workspace_keys:
            raise ValueError(f"portfolio workspace directory collision: {workspace_key}")

        template_raw = str(raw_target.get("template", "") or "").strip()
        if template_raw:
            template = _normalize_template_name(template_raw)
            _resolve_template_path(template)
            selection: dict[str, object] = {
                "selection_mode": "explicit",
                "selected_template": template,
            }
        elif manifest_path is not None:
            template, selection = _resolve_template_selection_from_write_manifest(manifest_path)
        else:
            raise ValueError(f"portfolio targets[{index}] requires template or write_manifest")

        seen_tickers.add(ticker)
        seen_workspace_keys.add(workspace_key)
        prepared.append(
            {
                "ticker": ticker,
                "company": target["company"],
                "template": template,
                "workspace_key": workspace_key,
                "write_manifest": str(manifest_path) if manifest_path is not None else None,
                "selection": selection,
            }
        )
    return prepared


def _build_research_portfolio_preview(
    portfolio_path: Path,
    workspace_root: Path,
    targets: list[dict[str, object]],
    *,
    overwrite: bool,
) -> dict[str, object]:
    target_previews: list[dict[str, object]] = []
    blocked_count = 0
    overwrite_count = 0
    for target in targets:
        template = str(target["template"])
        target_workspace = workspace_root / str(target["workspace_key"])
        artifacts = _portfolio_target_artifact_paths(target_workspace, template)
        existing_files = [str(path) for path in artifacts.values() if path.exists()]
        if existing_files and not overwrite:
            action = "blocked_existing_files"
            blocked_count += 1
        elif existing_files:
            action = "overwrite"
            overwrite_count += 1
        else:
            action = "create"
        target_previews.append(
            {
                "ticker": target["ticker"],
                "company": target["company"],
                "template": template,
                "workspace_root": str(target_workspace),
                "selection": target["selection"],
                "write_manifest": target["write_manifest"],
                "action": action,
                "existing_files": existing_files,
                "artifacts": {key: str(path) for key, path in artifacts.items()},
                "regenerated_files": [
                    str(target_workspace / "assets" / _TEMPLATE_DIR_NAME / "research-template.manifest.json")
                ],
            }
        )
    return {
        "schema_version": 1,
        "preview_type": "research_portfolio_materialization",
        "portfolio_file": str(portfolio_path),
        "portfolio_fingerprint": _sha256_file(portfolio_path),
        "workspace_root": str(workspace_root),
        "overwrite": overwrite,
        "can_materialize": blocked_count == 0,
        "summary": {
            "target_count": len(target_previews),
            "create_count": len(target_previews) - blocked_count - overwrite_count,
            "overwrite_count": overwrite_count,
            "blocked_count": blocked_count,
        },
        "derived_outputs": {
            "materialization_report": str(workspace_root / "research-portfolio.materialization.json"),
            "monitoring_status": str(workspace_root / "assets" / _TEMPLATE_DIR_NAME / "monitoring-status.json"),
            "workbook_status": str(workspace_root / "assets" / _TEMPLATE_DIR_NAME / "research-workbook-status.json"),
            "report_status": str(
                workspace_root / "assets" / _TEMPLATE_DIR_NAME / "research-workbook-report-status.json"
            ),
        },
        "targets": target_previews,
    }


def _portfolio_target_artifact_paths(workspace_root: Path, template: str) -> dict[str, Path]:
    artifact_dir = workspace_root / "assets" / _TEMPLATE_DIR_NAME
    template_file_name = "common.md" if template == _FALLBACK_TEMPLATE_NAME else f"common-plus-{template}.md"
    return {
        "write_template": artifact_dir / template_file_name,
        "research_workbook": artifact_dir / f"{template}.research-workbook.json",
        "research_progress_report": artifact_dir / f"{template}.research-progress.md",
        "monitoring_rules": artifact_dir / f"{template}.monitoring-rules.json",
        "source_map": artifact_dir / f"{template}.source-map.json",
        "usage_guide": artifact_dir / f"{template}.research-guide.md",
        "bundle": artifact_dir / f"{template}.bundle.json",
        "monitoring_plan": artifact_dir / f"{template}.monitoring-plan.json",
        "monitoring_status": artifact_dir / "monitoring-status.json",
        "workbook_status": artifact_dir / "research-workbook-status.json",
        "report_status": artifact_dir / "research-workbook-report-status.json",
    }


def _recommendation_payload(recommendation: ResearchTemplateRecommendation) -> dict[str, object]:
    return {
        "name": recommendation.name,
        "title": recommendation.title,
        "score": recommendation.score,
        "matched_facets": list(recommendation.matched_facets),
        "reason": recommendation.reason,
    }


def _company_facets_from_args(args: argparse.Namespace) -> CompanyFacetProfile:
    manifest_raw = getattr(args, "manifest", None)
    profile = _load_company_facets_from_manifest(Path(str(manifest_raw)).resolve()) if manifest_raw else None
    primary_facets = list(profile.primary_facets) if profile is not None else []
    constraint_facets = list(profile.cross_cutting_facets) if profile is not None else []
    primary_facets.extend(_as_str_list(getattr(args, "business_model_tags", ())))
    constraint_facets.extend(_as_str_list(getattr(args, "constraint_tags", ())))
    return CompanyFacetProfile(
        primary_facets=list(_dedupe(primary_facets)),
        cross_cutting_facets=list(_dedupe(constraint_facets)),
        confidence_notes=profile.confidence_notes if profile is not None else "",
    )


def _load_company_facets_from_manifest(path: Path) -> CompanyFacetProfile:
    return load_company_facets_from_manifest(path)


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _load_workbook_evidence_records(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"evidence file must contain an object or list: {path}")
    if not records or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"evidence file must contain one or more objects: {path}")
    return records


def _resolve_template_dir() -> Path:
    template_dir = resolve_package_assets_path() / _TEMPLATE_DIR_NAME
    if not template_dir.is_dir():
        raise FileNotFoundError(f"research template directory not found: {template_dir}")
    return template_dir


def _resolve_template_path(name: str) -> Path:
    normalized = _normalize_template_name(name)
    template_path = _resolve_template_dir() / f"{normalized}{_TEMPLATE_SUFFIX}"
    if not template_path.is_file():
        available = ", ".join(template.name for template in list_research_templates()) or "(none)"
        raise FileNotFoundError(f"unknown research template {name!r}; available: {available}")
    return template_path


def _normalize_template_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("template name is required")
    if any(char in normalized for char in ("\\", "/", ":", "..")):
        raise ValueError(f"invalid template name: {name!r}")
    return normalized


def _read_template_title(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped.removeprefix("# ").strip()
    return path.stem


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    deduped: list[str] = []
    for value in values:
        item = value.strip()
        if item and item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def _normalize_research_target(*, ticker: str, company: str) -> dict[str, str]:
    return {
        "ticker": ticker.strip().upper(),
        "company": company.strip(),
    }


def _identifier_component(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    return "-".join(part for part in normalized.split("-") if part)


def _scheduler_state_from_plan_inspection(inspection: dict[str, object]) -> str:
    validation = inspection.get("validation")
    if not isinstance(validation, dict) or validation.get("ok") is not True:
        return "invalid_plan"
    readiness = inspection.get("readiness")
    if not isinstance(readiness, dict):
        return "blocked_unknown"
    return str(readiness.get("status", "") or "blocked_unknown")


def _discover_research_artifact_paths(
    workspace_root: Path,
    filename_glob: str,
    *,
    recursive: bool,
) -> tuple[Path, ...]:
    resolved_workspace = workspace_root.resolve()
    direct_dir = resolved_workspace / "assets" / _TEMPLATE_DIR_NAME
    paths = set(direct_dir.glob(filename_glob)) if direct_dir.is_dir() else set()
    if recursive and resolved_workspace.is_dir():
        paths.update(resolved_workspace.glob(f"**/assets/{_TEMPLATE_DIR_NAME}/{filename_glob}"))
    return tuple(sorted((path.resolve() for path in paths), key=str))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json_object(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _source_map_sources_by_name(
    payload: dict[str, object],
    *,
    label: str,
) -> dict[str, dict[str, object]]:
    data_sources = payload.get("data_sources")
    if not isinstance(data_sources, list):
        raise ValueError(f"{label} data_sources must be a list")
    sources: dict[str, dict[str, object]] = {}
    for index, source in enumerate(data_sources):
        if not isinstance(source, dict):
            raise ValueError(f"{label} data_sources[{index}] must be an object")
        source_name = str(source.get("source", "") or "").strip()
        if not source_name:
            raise ValueError(f"{label} data_sources[{index}].source is required")
        if source_name in sources:
            raise ValueError(f"{label} contains duplicate source: {source_name}")
        binding_status = str(source.get("binding_status", "") or "")
        if binding_status not in {"unbound", "bound"}:
            raise ValueError(f"{label} data_sources[{index}].binding_status must be unbound or bound")
        sources[source_name] = source
    return sources


def _inspect_source_binding_snapshot(
    snapshot_path: Path,
    *,
    source_map_path: Path,
    current_template: str,
    current_source_names: set[str],
    snapshot_type: str,
    filename_marker: str,
) -> dict[str, object]:
    fingerprint = _sha256_file(snapshot_path)
    expected_name = f"{source_map_path.stem}.{filename_marker}.{fingerprint[:12]}.json"
    errors: list[str] = []
    if snapshot_path.name != expected_name:
        errors.append(f"filename fingerprint mismatch: expected {expected_name!r}")
    binding_status = ""
    bound_sources: list[str] = []
    try:
        payload = _load_json_object(snapshot_path)
        snapshot_template = str(payload.get("template", "") or "").strip()
        if snapshot_template != current_template:
            errors.append(f"template mismatch: source_map={current_template!r} snapshot={snapshot_template!r}")
        snapshot_sources = _source_map_sources_by_name(payload, label="binding snapshot")
        snapshot_source_names = set(snapshot_sources)
        if snapshot_source_names != current_source_names:
            missing = sorted(current_source_names - snapshot_source_names)
            extra = sorted(snapshot_source_names - current_source_names)
            errors.append(f"source set mismatch: missing={missing} extra={extra}")
        binding_status = str(payload.get("binding_status", "") or "")
        bound_sources = sorted(
            source_name for source_name, source in snapshot_sources.items() if source.get("binding_status") == "bound"
        )
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    return {
        "snapshot_file": str(snapshot_path),
        "snapshot_type": snapshot_type,
        "fingerprint": fingerprint,
        "binding_status": binding_status,
        "bound_sources": bound_sources,
        "ok": not errors,
        "errors": errors,
    }


__all__ = [
    "ResearchTemplate",
    "ResearchTemplateRecommendation",
    "build_monitoring_execution_plan",
    "build_monitoring_status_snapshot",
    "build_monitoring_scheduler_manifest",
    "build_research_portfolio_preview",
    "build_monitoring_rules_payload",
    "build_monitoring_source_map_payload",
    "build_monitoring_source_binding_preview",
    "build_monitoring_source_binding_rollback_preview",
    "build_research_workbook_payload",
    "build_research_workbook_report",
    "build_research_workbook_report_status_snapshot",
    "build_research_workbook_status_snapshot",
    "build_research_workbook_rollback_preview",
    "build_research_workbook_update_preview",
    "build_research_template_bundle_descriptor",
    "build_research_template_bundle_rebind_preview",
    "build_research_template_bundle_rebind_rollback_preview",
    "build_research_template_package_manifest",
    "build_research_template_usage_guide",
    "build_research_workspace_refresh_preview",
    "compose_research_template",
    "copy_research_template",
    "discover_research_template_bundles",
    "discover_research_workbook_reports",
    "discover_research_workbooks",
    "discover_monitoring_execution_plans",
    "extract_monitoring_variables",
    "get_monitoring_data_source_candidates",
    "inspect_monitoring_execution_plan",
    "inspect_monitoring_scheduler_manifest",
    "inspect_monitoring_source_binding_history",
    "inspect_research_template_bundle",
    "inspect_research_workbook",
    "inspect_research_workbook_report",
    "list_research_templates",
    "load_research_template",
    "materialize_research_template_bundle",
    "materialize_research_workspace",
    "materialize_research_bundle_from_write_manifest",
    "materialize_research_portfolio",
    "recommend_research_templates",
    "run_research_template_command",
    "validate_research_template_bundle_descriptor",
    "validate_monitoring_source_map_payload",
    "validate_research_workbook_payload",
    "validate_monitoring_execution_plan",
    "validate_monitoring_scheduler_manifest",
    "write_monitoring_rules_payload",
    "write_monitoring_execution_plan",
    "write_monitoring_status_snapshot",
    "write_monitoring_scheduler_manifest",
    "write_monitoring_source_map_payload",
    "write_monitoring_source_binding_approval",
    "write_monitoring_source_binding_rollback",
    "write_research_workbook_payload",
    "write_research_workbook_rollback",
    "write_research_workbook_report",
    "write_research_workbook_report_status_snapshot",
    "write_research_workbook_status_snapshot",
    "write_research_workbook_update",
    "write_research_template_package_manifest",
    "write_research_template_bundle_descriptor",
    "write_research_template_bundle_rebind",
    "write_research_template_bundle_rebind_rollback",
    "write_research_template_usage_guide",
    "write_research_workspace_refresh",
]
