"""Research workbook, evidence, progress-report, and status workflows."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from dayu.startup.config_file_resolver import resolve_package_assets_path

_TEMPLATE_DIR_NAME = "research_templates"
_TEMPLATE_SUFFIX = ".md"
_WORKBOOK_CATEGORIES = frozenset(
    {
        "research_question",
        "business_analysis",
        "evidence_requirement",
        "monitoring_variable",
        "falsifier",
        "synthesis",
        "valuation",
        "catalyst",
        "management_governance",
        "portfolio_decision",
    }
)
_WORKBOOK_ITEM_STATUSES = frozenset(
    {"open", "in_progress", "answered", "blocked", "not_applicable"}
)
_WORKBOOK_TERMINAL_STATUSES = frozenset({"answered", "not_applicable"})
_WORKBOOK_REPORT_METADATA_PREFIX = "<!-- DAYU_RESEARCH_WORKBOOK_REPORT "
_WORKBOOK_REPORT_METADATA_SUFFIX = " -->"

def build_research_workbook_payload(
    name: str,
    *,
    ticker: str = "",
    company: str = "",
) -> dict[str, object]:
    """Convert one research template into a trackable manual-review workbook."""

    normalized = _normalize_template_name(name)
    template_path = _resolve_template_path(normalized)
    sections: list[dict[str, object]] = []
    current_title = ""
    current_category = ""
    current_items: list[str] = []

    def append_current_section() -> None:
        if not current_category or not current_items:
            return
        section_key = hashlib.sha256(
            f"{normalized}\0{current_category}\0{current_title}".encode()
        ).hexdigest()[:10]
        items = []
        for prompt in current_items:
            item_key = hashlib.sha256(
                f"{normalized}\0{current_category}\0{current_title}\0{prompt}".encode()
            ).hexdigest()[:12]
            items.append(
                {
                    "item_id": f"item-{item_key}",
                    "prompt": prompt,
                    "status": "open",
                    "response": "",
                    "evidence": [],
                    "analyst_notes": "",
                    "evidence_required": True,
                }
            )
        sections.append(
            {
                "section_id": f"section-{section_key}",
                "title": current_title,
                "category": current_category,
                "items": items,
            }
        )

    for line in template_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            append_current_section()
            current_title = stripped.removeprefix("## ").strip()
            current_category = _research_workbook_section_category(current_title)
            current_items = []
            continue
        if current_category and stripped.startswith("- "):
            prompt = stripped.removeprefix("- ").strip()
            if prompt:
                current_items.append(prompt)
            continue
        if (
            current_category
            and stripped
            and not stripped.startswith("#")
            and not stripped.startswith("<!--")
            and stripped != "---"
        ):
            current_items.append(stripped)
    append_current_section()

    if not sections:
        raise ValueError(f"research template {normalized!r} has no actionable workbook sections")
    category_counts: dict[str, int] = {}
    item_count = 0
    for section in sections:
        category = str(section["category"])
        items = section["items"]
        count = len(items) if isinstance(items, list) else 0
        category_counts[category] = category_counts.get(category, 0) + count
        item_count += count
    return {
        "schema_version": 1,
        "workbook_type": "research_evidence_workbook",
        "template": normalized,
        "source_template_file": str(template_path),
        "source_template_fingerprint": _sha256_file(template_path),
        "research_target": _normalize_research_target(ticker=ticker, company=company),
        "completion_status": "not_started",
        "automation_status": "manual_review",
        "summary": {
            "section_count": len(sections),
            "item_count": item_count,
            "open_item_count": item_count,
            "category_counts": category_counts,
        },
        "sections": sections,
    }


def validate_research_workbook_payload(payload: dict[str, object]) -> dict[str, object]:
    """Validate a research workbook and derive live completion statistics."""

    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if payload.get("workbook_type") != "research_evidence_workbook":
        errors.append("workbook_type must be research_evidence_workbook")
    if payload.get("automation_status") != "manual_review":
        errors.append("automation_status must be manual_review")

    template = str(payload.get("template", "") or "").strip()
    source_fingerprint = str(payload.get("source_template_fingerprint", "") or "").strip()
    if not template:
        errors.append("template is required")
    else:
        try:
            current_template_path = _resolve_template_path(template)
            current_fingerprint = _sha256_file(current_template_path)
            if source_fingerprint != current_fingerprint:
                errors.append("source_template_fingerprint does not match the current packaged template")
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))
    source_template_file = payload.get("source_template_file")
    if not isinstance(source_template_file, str) or not source_template_file.strip():
        errors.append("source_template_file must be a non-empty path")

    research_target = payload.get("research_target")
    if not isinstance(research_target, dict):
        errors.append("research_target must be an object")
    else:
        for key in ("ticker", "company"):
            if not isinstance(research_target.get(key), str):
                errors.append(f"research_target.{key} must be a string")

    completion_status = str(payload.get("completion_status", "") or "")
    if completion_status not in {"not_started", "in_progress", "complete"}:
        errors.append("completion_status must be not_started, in_progress, or complete")

    sections = payload.get("sections")
    section_ids: set[str] = set()
    item_ids: set[str] = set()
    category_counts: dict[str, int] = {}
    status_counts = {status: 0 for status in sorted(_WORKBOOK_ITEM_STATUSES)}
    section_count = 0
    item_count = 0
    if not isinstance(sections, list) or not sections:
        errors.append("sections must be a non-empty list")
    else:
        section_count = len(sections)
        for section_index, section in enumerate(sections):
            if not isinstance(section, dict):
                errors.append(f"sections[{section_index}] must be an object")
                continue
            section_id = str(section.get("section_id", "") or "").strip()
            if not section_id:
                errors.append(f"sections[{section_index}].section_id is required")
            elif section_id in section_ids:
                errors.append(f"duplicate section_id: {section_id}")
            else:
                section_ids.add(section_id)
            if not str(section.get("title", "") or "").strip():
                errors.append(f"sections[{section_index}].title is required")
            category = str(section.get("category", "") or "")
            if category not in _WORKBOOK_CATEGORIES:
                errors.append(f"sections[{section_index}].category is invalid: {category!r}")
            items = section.get("items")
            if not isinstance(items, list) or not items:
                errors.append(f"sections[{section_index}].items must be a non-empty list")
                continue
            for item_index, item in enumerate(items):
                item_count += 1
                if category in _WORKBOOK_CATEGORIES:
                    category_counts[category] = category_counts.get(category, 0) + 1
                item_label = f"sections[{section_index}].items[{item_index}]"
                if not isinstance(item, dict):
                    errors.append(f"{item_label} must be an object")
                    continue
                item_id = str(item.get("item_id", "") or "").strip()
                if not item_id:
                    errors.append(f"{item_label}.item_id is required")
                elif item_id in item_ids:
                    errors.append(f"duplicate item_id: {item_id}")
                else:
                    item_ids.add(item_id)
                if not str(item.get("prompt", "") or "").strip():
                    errors.append(f"{item_label}.prompt is required")
                status = str(item.get("status", "") or "")
                if status not in _WORKBOOK_ITEM_STATUSES:
                    errors.append(f"{item_label}.status is invalid: {status!r}")
                else:
                    status_counts[status] += 1
                response = item.get("response")
                if not isinstance(response, str):
                    errors.append(f"{item_label}.response must be a string")
                analyst_notes = item.get("analyst_notes")
                if not isinstance(analyst_notes, str):
                    errors.append(f"{item_label}.analyst_notes must be a string")
                evidence_required = item.get("evidence_required")
                if not isinstance(evidence_required, bool):
                    errors.append(f"{item_label}.evidence_required must be a boolean")
                evidence = item.get("evidence")
                if not isinstance(evidence, list):
                    errors.append(f"{item_label}.evidence must be a list")
                    evidence = []
                for evidence_index, record in enumerate(evidence):
                    record_label = f"{item_label}.evidence[{evidence_index}]"
                    if not isinstance(record, dict):
                        errors.append(f"{record_label} must be an object")
                        continue
                    for key in ("source", "reference", "finding"):
                        if not str(record.get(key, "") or "").strip():
                            errors.append(f"{record_label}.{key} is required")
                if status == "answered":
                    if not isinstance(response, str) or not response.strip():
                        errors.append(f"{item_label}.response is required when status is answered")
                    if evidence_required is True and not evidence:
                        errors.append(f"{item_label}.evidence is required when status is answered")

    if item_count and status_counts["open"] == item_count:
        derived_completion_status = "not_started"
    elif item_count and sum(status_counts[status] for status in _WORKBOOK_TERMINAL_STATUSES) == item_count:
        derived_completion_status = "complete"
    else:
        derived_completion_status = "in_progress"
    live_summary = {
        "section_count": section_count,
        "item_count": item_count,
        "status_counts": status_counts,
        "category_counts": category_counts,
    }
    stored_summary = payload.get("summary")
    if not isinstance(stored_summary, dict):
        errors.append("summary must be an object")
    else:
        expected_stored_values = {
            "section_count": section_count,
            "item_count": item_count,
            "open_item_count": status_counts["open"],
            "category_counts": category_counts,
        }
        for key, expected in expected_stored_values.items():
            if stored_summary.get(key) != expected:
                warnings.append(f"summary.{key} is stale; expected {expected!r}")
    if completion_status in {"not_started", "in_progress", "complete"} and completion_status != derived_completion_status:
        warnings.append(
            f"completion_status is stale; expected {derived_completion_status!r}"
        )
    return {
        "ok": not errors,
        "template": template,
        "errors": errors,
        "warnings": warnings,
        "derived_completion_status": derived_completion_status,
        "live_summary": live_summary,
    }


def build_research_workbook_report(workbook_payload: dict[str, object]) -> str:
    """Render one valid workbook as a deterministic Markdown progress report."""

    validation = validate_research_workbook_payload(workbook_payload)
    validation_errors = validation.get("errors")
    if validation.get("ok") is not True:
        errors = validation_errors if isinstance(validation_errors, list) else []
        raise ValueError("research workbook is invalid: " + "; ".join(str(error) for error in errors))
    target = workbook_payload.get("research_target")
    research_target = (
        _normalize_research_target(
            ticker=str(target.get("ticker", "") or ""),
            company=str(target.get("company", "") or ""),
        )
        if isinstance(target, dict)
        else {"ticker": "", "company": ""}
    )
    template = str(workbook_payload.get("template", "") or "")
    subject = research_target["company"] or research_target["ticker"] or template
    live_summary = validation.get("live_summary")
    summary = live_summary if isinstance(live_summary, dict) else {}
    status_counts = summary.get("status_counts")
    status_counts = status_counts if isinstance(status_counts, dict) else {}
    lines = [
        f"# Research Workbook Progress: {subject}",
        "",
        f"- Template: `{template}`",
        f"- Ticker: `{research_target['ticker'] or '(not set)'}`",
        f"- Company: `{research_target['company'] or '(not set)'}`",
        f"- Derived completion: `{validation.get('derived_completion_status', '')}`",
        f"- Source template fingerprint: `{workbook_payload.get('source_template_fingerprint', '')}`",
        f"- Items: `{summary.get('item_count', 0)}`",
        f"- Status counts: `{json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    open_items: list[tuple[str, str, str]] = []
    sections = workbook_payload.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            title = str(section.get("title", "") or "")
            category = str(section.get("category", "") or "")
            lines.extend([f"## {title}", "", f"Category: `{category}`", ""])
            items = section.get("items")
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get("item_id", "") or "")
                status = str(item.get("status", "") or "")
                prompt = str(item.get("prompt", "") or "")
                response = str(item.get("response", "") or "").strip()
                analyst_notes = str(item.get("analyst_notes", "") or "").strip()
                lines.extend(
                    [
                        f"### [{status}] `{item_id}`",
                        "",
                        f"**Question:** {prompt}",
                        "",
                        f"**Response:** {response or '(not answered)'}",
                        "",
                        "**Evidence:**",
                        "",
                    ]
                )
                evidence = item.get("evidence")
                if isinstance(evidence, list) and evidence:
                    for record in evidence:
                        if not isinstance(record, dict):
                            continue
                        source = str(record.get("source", "") or "")
                        reference = str(record.get("reference", "") or "")
                        finding = str(record.get("finding", "") or "")
                        lines.append(f"- `{source}` | {reference} | {finding}")
                else:
                    lines.append("- (none)")
                lines.extend(
                    [
                        "",
                        f"**Analyst notes:** {analyst_notes or '(none)'}",
                        "",
                    ]
                )
                if status not in _WORKBOOK_TERMINAL_STATUSES:
                    open_items.append((item_id, status, prompt))
    lines.extend(["## Open Research Gaps", ""])
    if open_items:
        lines.extend(f"- [{status}] `{item_id}`: {prompt}" for item_id, status, prompt in open_items)
    else:
        lines.append("- None. All workbook items are terminal.")
    lines.append("")
    body = "\n".join(lines)
    metadata = {
        "schema_version": 1,
        "workbook_semantic_fingerprint": _sha256_json_object(workbook_payload),
        "report_body_fingerprint": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    metadata_json = json.dumps(metadata, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return f"{_WORKBOOK_REPORT_METADATA_PREFIX}{metadata_json}{_WORKBOOK_REPORT_METADATA_SUFFIX}\n\n{body}"


def write_research_workbook_report(
    workbook_path: Path,
    *,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a Markdown progress report next to one valid research workbook."""

    resolved_workbook = workbook_path.resolve()
    report = build_research_workbook_report(_load_json_object(resolved_workbook))
    default_name = resolved_workbook.name.removesuffix(".research-workbook.json") + ".research-progress.md"
    target_path = (output_path or resolved_workbook.with_name(default_name)).resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(report, encoding="utf-8", newline="\n")
    return target_path


def inspect_research_workbook_report(
    report_path: Path,
    workbook_path: Path,
) -> dict[str, object]:
    """Validate report integrity and freshness against its research workbook."""

    resolved_report = report_path.resolve()
    resolved_workbook = workbook_path.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, object] = {}
    report_body_fingerprint = ""
    current_workbook_fingerprint = ""
    stale = False
    report_tampered = False
    try:
        report_text = resolved_report.read_text(encoding="utf-8")
        metadata_line, separator, body = report_text.partition("\n\n")
        if not separator:
            errors.append("report metadata separator is missing")
        if not (
            metadata_line.startswith(_WORKBOOK_REPORT_METADATA_PREFIX)
            and metadata_line.endswith(_WORKBOOK_REPORT_METADATA_SUFFIX)
        ):
            errors.append("report metadata header is missing or malformed")
        else:
            metadata_json = metadata_line[
                len(_WORKBOOK_REPORT_METADATA_PREFIX) : -len(_WORKBOOK_REPORT_METADATA_SUFFIX)
            ]
            parsed_metadata = json.loads(metadata_json)
            if not isinstance(parsed_metadata, dict):
                errors.append("report metadata must be a JSON object")
            else:
                metadata = parsed_metadata
        report_body_fingerprint = hashlib.sha256(body.encode("utf-8")).hexdigest()
    except (OSError, ValueError) as exc:
        errors.append(f"report could not be read: {exc}")

    try:
        workbook_payload = _load_json_object(resolved_workbook)
        workbook_validation = validate_research_workbook_payload(workbook_payload)
        workbook_errors = workbook_validation.get("errors")
        if workbook_validation.get("ok") is not True:
            if isinstance(workbook_errors, list):
                errors.extend(f"workbook: {error}" for error in workbook_errors)
            else:
                errors.append("workbook validation failed")
        current_workbook_fingerprint = _sha256_json_object(workbook_payload)
    except (OSError, ValueError) as exc:
        errors.append(f"workbook could not be read: {exc}")

    if metadata:
        if metadata.get("schema_version") != 1:
            errors.append("report metadata schema_version must be 1")
        recorded_workbook_fingerprint = metadata.get("workbook_semantic_fingerprint")
        if not isinstance(recorded_workbook_fingerprint, str) or len(recorded_workbook_fingerprint) != 64:
            errors.append("report metadata workbook_semantic_fingerprint must be a SHA-256 digest")
        elif current_workbook_fingerprint and recorded_workbook_fingerprint != current_workbook_fingerprint:
            stale = True
            errors.append("report is stale because workbook content changed")
        recorded_body_fingerprint = metadata.get("report_body_fingerprint")
        if not isinstance(recorded_body_fingerprint, str) or len(recorded_body_fingerprint) != 64:
            errors.append("report metadata report_body_fingerprint must be a SHA-256 digest")
        elif recorded_body_fingerprint != report_body_fingerprint:
            report_tampered = True
            errors.append("report body fingerprint does not match metadata")
    return {
        "report_file": str(resolved_report),
        "workbook_file": str(resolved_workbook),
        "metadata": metadata,
        "current_workbook_semantic_fingerprint": current_workbook_fingerprint,
        "current_report_body_fingerprint": report_body_fingerprint,
        "stale": stale,
        "report_tampered": report_tampered,
        "validation": {
            "ok": not errors,
            "errors": errors,
            "warnings": warnings,
        },
    }


def discover_research_workbook_reports(
    workspace_root: Path,
    *,
    recursive: bool = False,
) -> tuple[dict[str, object], ...]:
    """Discover progress reports and validate each against its sibling workbook."""

    report_paths = _discover_research_artifact_paths(
        workspace_root,
        "*.research-progress.md",
        recursive=recursive,
    )
    inspections = []
    for report_path in report_paths:
        prefix = report_path.name.removesuffix(".research-progress.md")
        workbook_path = report_path.with_name(f"{prefix}.research-workbook.json")
        inspections.append(inspect_research_workbook_report(report_path, workbook_path))
    return tuple(inspections)


def build_research_workbook_report_status_snapshot(
    workspace_root: Path,
    *,
    recursive: bool = False,
) -> dict[str, object]:
    """Aggregate report freshness and integrity across a workspace or portfolio."""

    resolved_workspace = workspace_root.resolve()
    reports = discover_research_workbook_reports(resolved_workspace, recursive=recursive)
    valid_count = 0
    invalid_count = 0
    stale_count = 0
    tampered_count = 0
    missing_workbook_count = 0
    for report in reports:
        validation = report.get("validation")
        if isinstance(validation, dict) and validation.get("ok") is True:
            valid_count += 1
        else:
            invalid_count += 1
        if report.get("stale") is True:
            stale_count += 1
        if report.get("report_tampered") is True:
            tampered_count += 1
        workbook_path = Path(str(report.get("workbook_file", "")))
        if not workbook_path.is_file():
            missing_workbook_count += 1
    if not reports:
        overall_status = "no_reports"
    elif invalid_count:
        overall_status = "unhealthy"
    else:
        overall_status = "current"
    return {
        "schema_version": 1,
        "snapshot_type": "research_workbook_report_status",
        "workspace_root": str(resolved_workspace),
        "scan_scope": "recursive" if recursive else "workspace",
        "overall_status": overall_status,
        "summary": {
            "report_count": len(reports),
            "current_report_count": valid_count,
            "invalid_report_count": invalid_count,
            "stale_report_count": stale_count,
            "tampered_report_count": tampered_count,
            "missing_workbook_count": missing_workbook_count,
        },
        "reports": list(reports),
    }


def write_research_workbook_report_status_snapshot(
    workspace_root: Path,
    *,
    output_path: Path | None = None,
    recursive: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write a workbook-report health snapshot for UI or portfolio consumers."""

    resolved_workspace = workspace_root.resolve()
    payload = build_research_workbook_report_status_snapshot(
        resolved_workspace,
        recursive=recursive,
    )
    target_path = output_path or (
        resolved_workspace / "assets" / _TEMPLATE_DIR_NAME / "research-workbook-report-status.json"
    )
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target_path


def build_research_workbook_update_preview(
    workbook_payload: dict[str, object],
    *,
    item_id: str,
    status: str | None = None,
    response: str | None = None,
    analyst_notes: str | None = None,
    evidence_records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Preview one validated workbook item update and refreshed counters."""

    normalized_item_id = item_id.strip()
    if not normalized_item_id:
        raise ValueError("item_id is required")
    if status is None and response is None and analyst_notes is None and not evidence_records:
        raise ValueError("at least one workbook item update is required")
    updated = deepcopy(workbook_payload)
    sections = updated.get("sections")
    if not isinstance(sections, list):
        raise ValueError("workbook sections must be a list")
    matches: list[dict[str, object]] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        items = section.get("items")
        if not isinstance(items, list):
            continue
        matches.extend(
            item
            for item in items
            if isinstance(item, dict) and str(item.get("item_id", "") or "") == normalized_item_id
        )
    if not matches:
        raise ValueError(f"workbook item_id was not found: {normalized_item_id}")
    if len(matches) != 1:
        raise ValueError(f"workbook item_id is not unique: {normalized_item_id}")
    item = matches[0]
    changed_fields: list[str] = []
    if status is not None:
        item["status"] = status
        changed_fields.append("status")
    if response is not None:
        item["response"] = response
        changed_fields.append("response")
    if analyst_notes is not None:
        item["analyst_notes"] = analyst_notes
        changed_fields.append("analyst_notes")
    if evidence_records:
        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            raise ValueError(f"workbook item {normalized_item_id} evidence must be a list")
        evidence.extend(deepcopy(evidence_records))
        changed_fields.append("evidence")

    validation = validate_research_workbook_payload(updated)
    validation_errors = validation.get("errors")
    if isinstance(validation_errors, list) and validation_errors:
        raise ValueError("workbook update is invalid: " + "; ".join(str(error) for error in validation_errors))
    live_summary = validation.get("live_summary")
    derived_completion_status = validation.get("derived_completion_status")
    if not isinstance(live_summary, dict) or not isinstance(derived_completion_status, str):
        raise ValueError("workbook validation did not produce live progress")
    status_counts = live_summary.get("status_counts")
    category_counts = live_summary.get("category_counts")
    if not isinstance(status_counts, dict) or not isinstance(category_counts, dict):
        raise ValueError("workbook validation did not produce live counts")
    updated["summary"] = {
        "section_count": live_summary.get("section_count", 0),
        "item_count": live_summary.get("item_count", 0),
        "open_item_count": status_counts.get("open", 0),
        "category_counts": category_counts,
    }
    updated["completion_status"] = derived_completion_status
    final_validation = validate_research_workbook_payload(updated)
    if final_validation.get("ok") is not True:
        raise ValueError("workbook update failed final validation")
    return {
        "schema_version": 1,
        "preview_type": "research_workbook_item_update",
        "template": str(updated.get("template", "") or ""),
        "item_id": normalized_item_id,
        "changed_fields": changed_fields,
        "resulting_status": str(item.get("status", "") or ""),
        "derived_completion_status": derived_completion_status,
        "validation": final_validation,
        "workbook": updated,
    }


def write_research_workbook_update(
    workbook_path: Path,
    *,
    item_id: str,
    status: str | None = None,
    response: str | None = None,
    analyst_notes: str | None = None,
    evidence_records: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Apply one workbook update after writing an immutable content backup."""

    resolved_workbook = workbook_path.resolve()
    preview = build_research_workbook_update_preview(
        _load_json_object(resolved_workbook),
        item_id=item_id,
        status=status,
        response=response,
        analyst_notes=analyst_notes,
        evidence_records=evidence_records,
    )
    original_fingerprint = _sha256_file(resolved_workbook)
    backup_path = resolved_workbook.with_name(
        f"{resolved_workbook.stem}.before-update.{original_fingerprint[:12]}.json"
    )
    if backup_path.exists():
        if not backup_path.is_file() or _sha256_file(backup_path) != original_fingerprint:
            raise FileExistsError(f"workbook update backup path is occupied by different content: {backup_path}")
    else:
        backup_path.write_bytes(resolved_workbook.read_bytes())
    updated = preview.get("workbook")
    if not isinstance(updated, dict):
        raise ValueError("workbook update preview did not produce a workbook object")
    resolved_workbook.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "workbook_file": str(resolved_workbook),
        "backup_file": str(backup_path),
        "workbook_fingerprint_before": original_fingerprint,
        "workbook_fingerprint_after": _sha256_file(resolved_workbook),
        "preview": preview,
    }


def build_research_workbook_rollback_preview(
    workbook_path: Path,
    backup_path: Path,
) -> dict[str, object]:
    """Validate and preview restoring one content-addressed workbook backup."""

    resolved_workbook = workbook_path.resolve()
    resolved_backup = backup_path.resolve()
    if resolved_workbook == resolved_backup:
        raise ValueError("workbook and backup must be different files")
    if resolved_workbook.parent != resolved_backup.parent:
        raise ValueError("workbook backup must be in the same directory as workbook")
    current_payload = _load_json_object(resolved_workbook)
    backup_payload = _load_json_object(resolved_backup)
    current_template = str(current_payload.get("template", "") or "")
    backup_template = str(backup_payload.get("template", "") or "")
    if not current_template or current_template != backup_template:
        raise ValueError(
            f"workbook rollback template mismatch: workbook={current_template!r} backup={backup_template!r}"
        )
    if current_payload.get("research_target") != backup_payload.get("research_target"):
        raise ValueError("workbook rollback research_target mismatch")
    backup_fingerprint = _sha256_file(resolved_backup)
    expected_name = f"{resolved_workbook.stem}.before-update.{backup_fingerprint[:12]}.json"
    if resolved_backup.name != expected_name:
        raise ValueError(
            "workbook backup filename does not match workbook and content fingerprint: "
            f"expected {expected_name!r}"
        )
    backup_validation = validate_research_workbook_payload(backup_payload)
    if backup_validation.get("ok") is not True:
        raise ValueError("workbook rollback backup is not a valid research workbook")
    current_validation = validate_research_workbook_payload(current_payload)
    return {
        "schema_version": 1,
        "preview_type": "research_workbook_rollback",
        "template": current_template,
        "workbook_file": str(resolved_workbook),
        "backup_file": str(resolved_backup),
        "workbook_fingerprint_before": _sha256_file(resolved_workbook),
        "workbook_fingerprint_after": backup_fingerprint,
        "completion_status_before": str(current_payload.get("completion_status", "") or ""),
        "completion_status_after": str(backup_payload.get("completion_status", "") or ""),
        "current_validation": current_validation,
        "backup_validation": backup_validation,
        "write_required": True,
    }


def write_research_workbook_rollback(
    workbook_path: Path,
    backup_path: Path,
) -> dict[str, object]:
    """Restore a workbook backup after preserving the current bytes for redo."""

    preview = build_research_workbook_rollback_preview(workbook_path, backup_path)
    resolved_workbook = Path(str(preview["workbook_file"]))
    resolved_backup = Path(str(preview["backup_file"]))
    current_fingerprint = str(preview["workbook_fingerprint_before"])
    redo_backup_path = resolved_workbook.with_name(
        f"{resolved_workbook.stem}.before-update.{current_fingerprint[:12]}.json"
    )
    if redo_backup_path.exists():
        if not redo_backup_path.is_file() or _sha256_file(redo_backup_path) != current_fingerprint:
            raise FileExistsError(f"workbook redo backup path is occupied by different content: {redo_backup_path}")
    else:
        redo_backup_path.write_bytes(resolved_workbook.read_bytes())
    resolved_workbook.write_bytes(resolved_backup.read_bytes())
    restored_fingerprint = _sha256_file(resolved_workbook)
    expected_fingerprint = str(preview["workbook_fingerprint_after"])
    if restored_fingerprint != expected_fingerprint:
        raise ValueError("restored workbook fingerprint does not match backup")
    return {
        "workbook_file": str(resolved_workbook),
        "restored_backup_file": str(resolved_backup),
        "redo_backup_file": str(redo_backup_path),
        "workbook_fingerprint_before": current_fingerprint,
        "workbook_fingerprint_after": restored_fingerprint,
        "preview": preview,
    }


def write_research_workbook_payload(
    name: str,
    *,
    workspace_root: Path,
    ticker: str = "",
    company: str = "",
    output_path: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Write a manual-review research workbook for one template and target."""

    normalized = _normalize_template_name(name)
    payload = build_research_workbook_payload(normalized, ticker=ticker, company=company)
    target_path = output_path or workspace_root / "assets" / _TEMPLATE_DIR_NAME / f"{normalized}.research-workbook.json"
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target_path


def inspect_research_workbook(workbook_path: Path) -> dict[str, object]:
    """Load one workbook and retain diagnostics for discovery/status consumers."""

    resolved_path = workbook_path.resolve()
    try:
        payload = _load_json_object(resolved_path)
        validation = validate_research_workbook_payload(payload)
        research_target = payload.get("research_target")
        target = (
            _normalize_research_target(
                ticker=str(research_target.get("ticker", "") or ""),
                company=str(research_target.get("company", "") or ""),
            )
            if isinstance(research_target, dict)
            else {"ticker": "", "company": ""}
        )
        return {
            "workbook_file": str(resolved_path),
            "template": str(payload.get("template", "") or ""),
            "research_target": target,
            "completion_status": validation.get("derived_completion_status", ""),
            "live_summary": validation.get("live_summary", {}),
            "validation": validation,
        }
    except (OSError, ValueError) as exc:
        return {
            "workbook_file": str(resolved_path),
            "template": "",
            "research_target": {"ticker": "", "company": ""},
            "completion_status": "invalid",
            "live_summary": {},
            "validation": {
                "ok": False,
                "template": "",
                "errors": [str(exc)],
                "warnings": [],
            },
        }


def discover_research_workbooks(
    workspace_root: Path,
    *,
    recursive: bool = False,
) -> tuple[dict[str, object], ...]:
    """Discover standard research workbooks with diagnostics."""

    paths = _discover_research_artifact_paths(
        workspace_root,
        "*.research-workbook.json",
        recursive=recursive,
    )
    return tuple(inspect_research_workbook(path) for path in paths)


def build_research_workbook_status_snapshot(
    workspace_root: Path,
    *,
    recursive: bool = False,
) -> dict[str, object]:
    """Aggregate validated workbook progress across one workspace or portfolio."""

    resolved_workspace = workspace_root.resolve()
    workbooks = discover_research_workbooks(resolved_workspace, recursive=recursive)
    valid_count = 0
    invalid_count = 0
    completion_counts = {status: 0 for status in ("not_started", "in_progress", "complete")}
    item_status_counts = {status: 0 for status in sorted(_WORKBOOK_ITEM_STATUSES)}
    item_count = 0
    for workbook in workbooks:
        validation = workbook.get("validation")
        if not isinstance(validation, dict) or validation.get("ok") is not True:
            invalid_count += 1
            continue
        valid_count += 1
        completion_status = str(workbook.get("completion_status", "") or "")
        if completion_status in completion_counts:
            completion_counts[completion_status] += 1
        live_summary = workbook.get("live_summary")
        if not isinstance(live_summary, dict):
            continue
        item_count += int(live_summary.get("item_count", 0) or 0)
        status_counts = live_summary.get("status_counts")
        if isinstance(status_counts, dict):
            for status in item_status_counts:
                item_status_counts[status] += int(status_counts.get(status, 0) or 0)
    if not workbooks:
        overall_status = "no_workbooks"
    elif invalid_count:
        overall_status = "unhealthy"
    elif completion_counts["complete"] == valid_count:
        overall_status = "complete"
    elif completion_counts["not_started"] == valid_count:
        overall_status = "not_started"
    else:
        overall_status = "in_progress"
    return {
        "schema_version": 1,
        "snapshot_type": "research_workbook_status",
        "workspace_root": str(resolved_workspace),
        "scan_scope": "recursive" if recursive else "workspace",
        "overall_status": overall_status,
        "summary": {
            "workbook_count": len(workbooks),
            "valid_workbook_count": valid_count,
            "invalid_workbook_count": invalid_count,
            "completion_counts": completion_counts,
            "item_count": item_count,
            "item_status_counts": item_status_counts,
        },
        "workbooks": list(workbooks),
    }


def write_research_workbook_status_snapshot(
    workspace_root: Path,
    *,
    output_path: Path | None = None,
    recursive: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write a workspace- or portfolio-level research workbook status snapshot."""

    resolved_workspace = workspace_root.resolve()
    payload = build_research_workbook_status_snapshot(resolved_workspace, recursive=recursive)
    target_path = output_path or resolved_workspace / "assets" / _TEMPLATE_DIR_NAME / "research-workbook-status.json"
    target_path = target_path.resolve()
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"{target_path} already exists; pass --overwrite to replace it")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target_path


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _resolve_template_path(name: str) -> Path:
    normalized = _normalize_template_name(name)
    template_dir = resolve_package_assets_path() / _TEMPLATE_DIR_NAME
    template_path = template_dir / f"{normalized}{_TEMPLATE_SUFFIX}"
    if not template_path.is_file():
        available = ", ".join(
            path.stem
            for path in sorted(template_dir.glob(f"*{_TEMPLATE_SUFFIX}"))
            if path.stem.lower() != "readme"
        ) or "(none)"
        raise FileNotFoundError(f"unknown research template {name!r}; available: {available}")
    return template_path


def _normalize_template_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("template name is required")
    if any(char in normalized for char in ("\\", "/", ":", "..")):
        raise ValueError(f"invalid template name: {name!r}")
    return normalized


def _normalize_research_target(*, ticker: str, company: str) -> dict[str, str]:
    return {
        "ticker": ticker.strip().upper(),
        "company": company.strip(),
    }


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
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _research_workbook_section_category(title: str) -> str:
    if "买方问题" in title or "研究对象" in title:
        return "research_question"
    if "经营拆解" in title or "赚钱机制" in title:
        return "business_analysis"
    if "证据" in title:
        return "evidence_requirement"
    if "监控变量" in title:
        return "monitoring_variable"
    if "否决" in title or "反证" in title:
        return "falsifier"
    if "结论" in title or "输出" in title:
        return "synthesis"
    if "估值" in title or "预期差" in title:
        return "valuation"
    if "催化" in title or "时间轴" in title:
        return "catalyst"
    if "管理层" in title or "治理" in title or "资本配置" in title:
        return "management_governance"
    if "组合决策" in title or "风险预算" in title:
        return "portfolio_decision"
    return ""


__all__ = [
    "build_research_workbook_payload",
    "build_research_workbook_report",
    "build_research_workbook_report_status_snapshot",
    "build_research_workbook_rollback_preview",
    "build_research_workbook_status_snapshot",
    "build_research_workbook_update_preview",
    "discover_research_workbook_reports",
    "discover_research_workbooks",
    "inspect_research_workbook",
    "inspect_research_workbook_report",
    "validate_research_workbook_payload",
    "write_research_workbook_payload",
    "write_research_workbook_report",
    "write_research_workbook_report_status_snapshot",
    "write_research_workbook_rollback",
    "write_research_workbook_status_snapshot",
    "write_research_workbook_update",
]
