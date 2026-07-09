from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from dayu.cli.arg_parsing import parse_arguments
from dayu.cli.main import main
from dayu.cli.commands.research_template import (
    build_monitoring_rules_payload,
    build_monitoring_execution_plan,
    build_monitoring_status_snapshot,
    build_monitoring_scheduler_manifest,
    build_monitoring_source_binding_preview,
    build_monitoring_source_binding_rollback_preview,
    build_monitoring_source_map_payload,
    build_research_template_bundle_rebind_preview,
    build_research_template_bundle_rebind_rollback_preview,
    build_research_template_bundle_descriptor,
    build_research_template_package_manifest,
    build_research_portfolio_preview,
    build_research_workbook_payload,
    build_research_workbook_report,
    build_research_workbook_report_status_snapshot,
    build_research_workbook_status_snapshot,
    build_research_workbook_rollback_preview,
    build_research_workbook_update_preview,
    build_research_workspace_refresh_preview,
    build_research_template_usage_guide,
    compose_research_template,
    copy_research_template,
    discover_research_template_bundles,
    discover_monitoring_execution_plans,
    extract_monitoring_variables,
    get_monitoring_data_source_candidates,
    inspect_research_template_bundle,
    inspect_research_workbook_report,
    inspect_monitoring_execution_plan,
    inspect_monitoring_scheduler_manifest,
    inspect_monitoring_source_binding_history,
    list_research_templates,
    load_research_template,
    materialize_research_bundle_from_write_manifest,
    materialize_research_template_bundle,
    materialize_research_workspace,
    materialize_research_portfolio,
    recommend_research_templates,
    run_research_template_command,
    validate_monitoring_source_map_payload,
    validate_monitoring_execution_plan,
    validate_monitoring_scheduler_manifest,
    validate_research_workbook_payload,
    validate_research_template_bundle_descriptor,
    write_monitoring_rules_payload,
    write_monitoring_execution_plan,
    write_monitoring_status_snapshot,
    write_monitoring_scheduler_manifest,
    write_monitoring_source_binding_approval,
    write_monitoring_source_binding_rollback,
    write_monitoring_source_map_payload,
    write_research_template_package_manifest,
    write_research_template_bundle_rebind,
    write_research_template_bundle_rebind_rollback,
    write_research_template_usage_guide,
    write_research_workbook_payload,
    write_research_workbook_rollback,
    write_research_workbook_report,
    write_research_workbook_report_status_snapshot,
    write_research_workbook_status_snapshot,
    write_research_workbook_update,
    write_research_workspace_refresh,
)
from dayu.cli.commands.research_workbook import (
    build_research_workbook_payload as direct_build_research_workbook_payload,
)
from dayu.cli.research_template_assets import resolve_research_template_for_write
from dayu.services.internal.write_pipeline.models import CompanyFacetProfile
from dayu.services.internal.write_pipeline.template_parser import parse_template_layout


@pytest.mark.unit
def test_list_research_templates_includes_industry_templates() -> None:
    names = {template.name for template in list_research_templates()}

    assert {"common", "consumer", "cyclical", "technology", "financial"} <= names


@pytest.mark.unit
def test_load_research_template_returns_template_body() -> None:
    content = load_research_template("consumer")

    assert "DAYU_RESEARCH_TEMPLATE" in content
    assert "监控变量" in content


@pytest.mark.unit
def test_copy_research_template_defaults_to_workspace_assets(tmp_path: Path) -> None:
    copied_path = copy_research_template("technology", workspace_root=tmp_path)

    assert copied_path == (tmp_path / "assets" / "research_templates" / "technology.md").resolve()
    assert "DAYU_RESEARCH_TEMPLATE" in copied_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_copy_research_template_requires_overwrite(tmp_path: Path) -> None:
    copied_path = copy_research_template("financial", workspace_root=tmp_path)

    with pytest.raises(FileExistsError):
        copy_research_template("financial", workspace_root=tmp_path)

    copied_path.write_text("custom", encoding="utf-8")
    copy_research_template("financial", workspace_root=tmp_path, overwrite=True)

    assert "custom" not in copied_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_compose_research_template_combines_common_and_industry(tmp_path: Path) -> None:
    composed_path = compose_research_template("consumer", workspace_root=tmp_path)
    content = composed_path.read_text(encoding="utf-8")

    assert composed_path == (tmp_path / "assets" / "research_templates" / "common-plus-consumer.md").resolve()
    assert "name=common" in content
    assert "name=consumer" in content
    assert "\n---\n" in content
    layout = parse_template_layout(content)
    titles = [chapter.title for chapter in layout.chapters]
    assert titles[0] == "投资要点概览"
    assert "深化研究框架" in titles


@pytest.mark.unit
def test_resolve_research_template_for_write_creates_and_reuses_composition(tmp_path: Path) -> None:
    template_path = resolve_research_template_for_write("consumer", workspace_root=tmp_path)

    assert template_path == (tmp_path / "assets" / "research_templates" / "common-plus-consumer.md").resolve()
    original_content = template_path.read_text(encoding="utf-8")
    comparison_path = compose_research_template(
        "consumer",
        workspace_root=tmp_path,
        output_path=tmp_path / "comparison.md",
    )
    assert original_content == comparison_path.read_text(encoding="utf-8")
    assert resolve_research_template_for_write("CONSUMER", workspace_root=tmp_path) == template_path


@pytest.mark.unit
def test_resolve_research_template_for_write_rejects_drifted_composition(tmp_path: Path) -> None:
    template_path = resolve_research_template_for_write("technology", workspace_root=tmp_path)
    template_path.write_text("# analyst customization\n", encoding="utf-8")

    with pytest.raises(ValueError, match="differs from packaged composition"):
        resolve_research_template_for_write("technology", workspace_root=tmp_path)


@pytest.mark.unit
def test_resolve_research_template_for_write_auto_routes_from_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "company_facets": {
                    "primary_facets": ["银行"],
                    "cross_cutting_facets": ["利率敏感"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    template_path = resolve_research_template_for_write(
        "auto",
        workspace_root=tmp_path,
        manifest_path=manifest_path,
    )

    assert template_path.name == "common-plus-financial.md"
    assert "name=financial" in template_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_resolve_research_template_for_write_auto_falls_back_to_common(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"company_facets": {"primary_facets": ["未知业务"]}}, ensure_ascii=False),
        encoding="utf-8",
    )

    template_path = resolve_research_template_for_write(
        "auto",
        workspace_root=tmp_path,
        manifest_path=manifest_path,
    )

    assert template_path.name == "common.write.md"
    titles = [chapter.title for chapter in parse_template_layout(template_path.read_text(encoding="utf-8")).chapters]
    assert titles[0] == "投资要点概览"
    assert "深化研究框架" in titles


@pytest.mark.unit
def test_resolve_research_template_for_write_auto_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run write --infer first"):
        resolve_research_template_for_write(
            "auto",
            workspace_root=tmp_path,
            manifest_path=tmp_path / "missing.json",
        )


@pytest.mark.unit
def test_compose_research_template_builds_write_compatible_common(tmp_path: Path) -> None:
    composed_path = compose_research_template("common", workspace_root=tmp_path)

    assert composed_path.name == "common.write.md"
    titles = [chapter.title for chapter in parse_template_layout(composed_path.read_text(encoding="utf-8")).chapters]
    assert titles[0] == "投资要点概览"
    assert "深化研究框架" in titles


@pytest.mark.unit
def test_extract_monitoring_variables_reads_template_section() -> None:
    variables = extract_monitoring_variables("consumer")

    assert len(variables) >= 4
    assert all(variable for variable in variables)


@pytest.mark.unit
def test_build_monitoring_rules_payload_uses_extracted_variables() -> None:
    payload = build_monitoring_rules_payload("cyclical")

    assert payload["schema_version"] == 1
    assert payload["template"] == "cyclical"
    assert payload["rule_type"] == "manual_review"
    data_source_candidates = payload["data_source_candidates"]
    assert isinstance(data_source_candidates, list)
    assert "industry_price_data" in data_source_candidates
    variables = payload["variables"]
    assert isinstance(variables, list)
    assert len(variables) >= 4
    assert all(item["evidence_required"] is True for item in variables)
    assert all(item["binding_status"] == "unbound" for item in variables)
    assert all("data_source_candidates" in item for item in variables)


@pytest.mark.unit
def test_get_monitoring_data_source_candidates_is_template_specific() -> None:
    candidates = get_monitoring_data_source_candidates("financial")

    assert "regulatory_filings" in candidates
    assert "capital_adequacy_data" in candidates


@pytest.mark.unit
def test_write_monitoring_rules_payload_defaults_to_workspace_assets(tmp_path: Path) -> None:
    rules_path = write_monitoring_rules_payload("technology", workspace_root=tmp_path)

    assert rules_path == (tmp_path / "assets" / "research_templates" / "technology.monitoring-rules.json").resolve()
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    assert payload["template"] == "technology"
    assert len(payload["variables"]) >= 4


@pytest.mark.unit
def test_research_template_reexports_workbook_api() -> None:
    assert build_research_workbook_payload is direct_build_research_workbook_payload


@pytest.mark.unit
def test_build_research_workbook_payload_extracts_actionable_consumer_sections() -> None:
    payload = build_research_workbook_payload(
        "consumer",
        ticker=" 600519 ",
        company="贵州茅台",
    )

    assert payload["workbook_type"] == "research_evidence_workbook"
    assert payload["completion_status"] == "not_started"
    assert payload["automation_status"] == "manual_review"
    assert payload["research_target"] == {"ticker": "600519", "company": "贵州茅台"}
    sections = payload["sections"]
    assert isinstance(sections, list)
    categories = {section["category"] for section in sections}
    assert categories == {
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
    assert all(not str(section["title"]).startswith("0.") for section in sections)
    items = [item for section in sections for item in section["items"]]
    item_ids = [item["item_id"] for item in items]
    assert len(item_ids) == len(set(item_ids))
    assert all(item["status"] == "open" for item in items)
    assert all(item["response"] == "" and item["evidence"] == [] for item in items)
    summary = payload["summary"]
    assert isinstance(summary, dict)
    assert summary["item_count"] == len(items)
    assert summary["open_item_count"] == len(items)


@pytest.mark.unit
def test_build_research_workbook_payload_uses_stable_ids() -> None:
    first = build_research_workbook_payload("common")
    second = build_research_workbook_payload("common")

    assert first == second
    sections = first["sections"]
    assert isinstance(sections, list)
    assert {section["category"] for section in sections} == {
        "research_question",
        "business_analysis",
        "evidence_requirement",
        "falsifier",
        "synthesis",
        "valuation",
        "catalyst",
        "management_governance",
        "portfolio_decision",
    }


@pytest.mark.unit
def test_build_research_workbook_payload_keeps_ids_unique_for_duplicate_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dayu.cli.commands import research_workbook as research_workbook_module

    template = tmp_path / "dup.md"
    # Repeat a heading and a bullet: without positional indexing these collide.
    template.write_text(
        "## 买方问题\n- 同样的问题\n- 同样的问题\n\n## 买方问题\n- 同样的问题\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        research_workbook_module,
        "_resolve_template_path",
        lambda _name: template,
    )

    payload = build_research_workbook_payload("consumer")

    sections = payload["sections"]
    assert isinstance(sections, list)
    section_ids = [section["section_id"] for section in sections]
    item_ids = [item["item_id"] for section in sections for item in section["items"]]
    assert len(section_ids) == len(set(section_ids))
    assert len(item_ids) == len(set(item_ids))
    # The freshly-built workbook must validate against its own contract.
    assert validate_research_workbook_payload(payload)["ok"] is True


@pytest.mark.unit
def test_build_research_workbook_payload_tolerates_template_bom(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dayu.cli.commands import research_workbook as research_workbook_module

    template = tmp_path / "bom.md"
    # A BOM-prefixed template must not silently drop its first section.
    template.write_bytes("﻿## 买方问题\n- 这家公司是做什么生意的？\n".encode("utf-8"))
    monkeypatch.setattr(
        research_workbook_module,
        "_resolve_template_path",
        lambda _name: template,
    )

    payload = build_research_workbook_payload("consumer")

    sections = payload["sections"]
    assert isinstance(sections, list)
    assert [section["category"] for section in sections] == ["research_question"]
    assert sections[0]["items"][0]["prompt"] == "这家公司是做什么生意的？"


@pytest.mark.unit
def test_validate_research_workbook_payload_excludes_non_dict_items_from_counts() -> None:
    payload = build_research_workbook_payload("common")
    sections = payload["sections"]
    assert isinstance(sections, list)
    baseline = validate_research_workbook_payload(payload)["live_summary"]["item_count"]
    # Inject a malformed (non-dict) item; it must not inflate live_summary.
    sections[0]["items"].append("not-an-object")

    result = validate_research_workbook_payload(payload)

    assert result["ok"] is False
    assert result["live_summary"]["item_count"] == baseline


@pytest.mark.unit
@pytest.mark.parametrize("template", ["common", "consumer", "cyclical", "technology", "financial"])
def test_research_workbook_includes_decision_and_governance_questions(template: str) -> None:
    payload = build_research_workbook_payload(template)

    summary = payload["summary"]
    assert isinstance(summary, dict)
    category_counts = summary["category_counts"]
    assert isinstance(category_counts, dict)
    assert category_counts["valuation"] == 4
    assert category_counts["catalyst"] == 4
    assert category_counts["management_governance"] == 4
    assert category_counts["portfolio_decision"] == 4


@pytest.mark.unit
def test_write_research_workbook_payload_requires_overwrite(tmp_path: Path) -> None:
    workbook_path = write_research_workbook_payload(
        "technology",
        workspace_root=tmp_path,
        ticker="AAPL",
        company="Apple Inc.",
    )

    assert workbook_path == (tmp_path / "assets" / "research_templates" / "technology.research-workbook.json").resolve()
    payload = json.loads(workbook_path.read_text(encoding="utf-8"))
    assert payload["research_target"]["ticker"] == "AAPL"
    with pytest.raises(FileExistsError):
        write_research_workbook_payload("technology", workspace_root=tmp_path)


@pytest.mark.unit
def test_validate_research_workbook_payload_accepts_generated_workbook() -> None:
    payload = build_research_workbook_payload("financial", ticker="0005.HK")

    result = validate_research_workbook_payload(payload)

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["warnings"] == []
    assert result["derived_completion_status"] == "not_started"
    live_summary = result["live_summary"]
    stored_summary = payload["summary"]
    assert isinstance(live_summary, dict)
    assert isinstance(stored_summary, dict)
    status_counts = live_summary["status_counts"]
    assert isinstance(status_counts, dict)
    assert status_counts["open"] == stored_summary["item_count"]


@pytest.mark.unit
def test_validate_research_workbook_payload_derives_progress_from_answered_item() -> None:
    payload = build_research_workbook_payload("consumer")
    sections = payload["sections"]
    assert isinstance(sections, list)
    first_item = sections[0]["items"][0]
    first_item["status"] = "answered"
    first_item["response"] = "收入增长主要来自同店销售改善。"
    first_item["evidence"] = [
        {
            "source": "annual_report",
            "reference": "2025 annual report p.42",
            "finding": "同店销售同比增长 8%。",
        }
    ]

    result = validate_research_workbook_payload(payload)

    assert result["ok"] is True
    assert result["derived_completion_status"] == "in_progress"
    warnings = result["warnings"]
    assert isinstance(warnings, list)
    assert any("summary.open_item_count is stale" in warning for warning in warnings)
    assert any("completion_status is stale" in warning for warning in warnings)


@pytest.mark.unit
def test_validate_research_workbook_payload_rejects_duplicate_and_unsupported_answer() -> None:
    payload = build_research_workbook_payload("technology")
    sections = payload["sections"]
    assert isinstance(sections, list)
    first_item = sections[0]["items"][0]
    second_item = sections[0]["items"][1]
    second_item["item_id"] = first_item["item_id"]
    first_item["status"] = "answered"
    payload["source_template_fingerprint"] = "0" * 64

    result = validate_research_workbook_payload(payload)

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert any("source_template_fingerprint" in error for error in errors)
    assert any("duplicate item_id" in error for error in errors)
    assert any("response is required" in error for error in errors)
    assert any("evidence is required" in error for error in errors)


@pytest.mark.unit
def test_build_research_workbook_update_preview_answers_item_and_refreshes_summary() -> None:
    payload = build_research_workbook_payload("consumer")
    original = json.loads(json.dumps(payload))
    sections = payload["sections"]
    assert isinstance(sections, list)
    item_id = str(sections[0]["items"][0]["item_id"])

    preview = build_research_workbook_update_preview(
        payload,
        item_id=item_id,
        status="answered",
        response="需求增长由同店销售改善驱动。",
        analyst_notes="下一季继续跟踪库存。",
        evidence_records=[
            {
                "source": "annual_report",
                "reference": "2025 annual report p.42",
                "finding": "同店销售同比增长 8%。",
            }
        ],
    )

    assert payload == original
    assert preview["changed_fields"] == ["status", "response", "analyst_notes", "evidence"]
    assert preview["resulting_status"] == "answered"
    assert preview["derived_completion_status"] == "in_progress"
    validation = preview["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is True
    assert validation["warnings"] == []
    workbook = preview["workbook"]
    assert isinstance(workbook, dict)
    summary = workbook["summary"]
    assert isinstance(summary, dict)
    assert summary["open_item_count"] == 36


@pytest.mark.unit
def test_build_research_workbook_update_preview_rejects_answer_without_evidence() -> None:
    payload = build_research_workbook_payload("technology")
    sections = payload["sections"]
    assert isinstance(sections, list)
    item_id = str(sections[0]["items"][0]["item_id"])

    with pytest.raises(ValueError, match="evidence is required"):
        build_research_workbook_update_preview(
            payload,
            item_id=item_id,
            status="answered",
            response="产品留存改善。",
        )


@pytest.mark.unit
def test_write_research_workbook_update_creates_immutable_backup(tmp_path: Path) -> None:
    workbook_path = write_research_workbook_payload("financial", workspace_root=tmp_path)
    original = workbook_path.read_bytes()
    payload = json.loads(original)
    item_id = payload["sections"][0]["items"][0]["item_id"]

    result = write_research_workbook_update(
        workbook_path,
        item_id=item_id,
        status="in_progress",
        analyst_notes="等待下一期资本充足率披露。",
    )

    backup_path = Path(str(result["backup_file"]))
    assert backup_path.read_bytes() == original
    assert workbook_path.read_bytes() != original
    updated = json.loads(workbook_path.read_text(encoding="utf-8"))
    assert updated["completion_status"] == "in_progress"
    assert updated["summary"]["open_item_count"] == 36


@pytest.mark.unit
def test_research_workbook_rollback_is_previewable_and_reversible(tmp_path: Path) -> None:
    workbook_path = write_research_workbook_payload("financial", workspace_root=tmp_path)
    original = workbook_path.read_bytes()
    payload = json.loads(original)
    item_id = payload["sections"][0]["items"][0]["item_id"]
    update_result = write_research_workbook_update(
        workbook_path,
        item_id=item_id,
        status="in_progress",
        analyst_notes="等待下一期资本充足率披露。",
    )
    updated = workbook_path.read_bytes()
    backup_path = Path(str(update_result["backup_file"]))

    preview = build_research_workbook_rollback_preview(workbook_path, backup_path)

    assert preview["completion_status_before"] == "in_progress"
    assert preview["completion_status_after"] == "not_started"
    assert workbook_path.read_bytes() == updated

    rollback_result = write_research_workbook_rollback(workbook_path, backup_path)

    redo_backup_path = Path(str(rollback_result["redo_backup_file"]))
    assert redo_backup_path.read_bytes() == updated
    assert workbook_path.read_bytes() == original

    redo_result = write_research_workbook_rollback(workbook_path, redo_backup_path)

    assert Path(str(redo_result["redo_backup_file"])).read_bytes() == original
    assert workbook_path.read_bytes() == updated


@pytest.mark.unit
def test_research_workbook_rollback_rejects_forged_backup_filename(tmp_path: Path) -> None:
    workbook_path = write_research_workbook_payload("consumer", workspace_root=tmp_path)
    forged = workbook_path.with_name(f"{workbook_path.stem}.before-update.000000000000.json")
    forged.write_bytes(workbook_path.read_bytes())

    with pytest.raises(ValueError, match="filename does not match"):
        build_research_workbook_rollback_preview(workbook_path, forged)


@pytest.mark.unit
def test_research_workbook_rollback_recovers_when_current_file_is_corrupt(tmp_path: Path) -> None:
    workbook_path = write_research_workbook_payload("financial", workspace_root=tmp_path)
    original = workbook_path.read_bytes()
    item_id = json.loads(original)["sections"][0]["items"][0]["item_id"]
    update_result = write_research_workbook_update(
        workbook_path,
        item_id=item_id,
        status="in_progress",
        analyst_notes="等待披露。",
    )
    backup_path = Path(str(update_result["backup_file"]))

    # Corrupt the current workbook: rollback (the recovery path) must still run.
    workbook_path.write_text("{ this is not valid json", encoding="utf-8")

    preview = build_research_workbook_rollback_preview(workbook_path, backup_path)
    assert preview["backup_validation"]["ok"] is True
    assert preview["current_validation"]["ok"] is False
    assert preview["current_restorable"] is False

    rollback_result = write_research_workbook_rollback(workbook_path, backup_path)
    assert workbook_path.read_bytes() == original
    # A corrupt current file must not be saved as a junk redo backup that could
    # never be rolled back to.
    assert rollback_result["redo_backup_file"] is None


@pytest.mark.unit
def test_research_workbook_rollback_rejects_unreadable_backup_cleanly(tmp_path: Path) -> None:
    workbook_path = write_research_workbook_payload("consumer", workspace_root=tmp_path)
    corrupt_bytes = b"{ not valid json"
    fingerprint = hashlib.sha256(corrupt_bytes).hexdigest()[:12]
    backup = workbook_path.with_name(f"{workbook_path.stem}.before-update.{fingerprint}.json")
    backup.write_bytes(corrupt_bytes)

    with pytest.raises(ValueError, match="not readable JSON"):
        build_research_workbook_rollback_preview(workbook_path, backup)


@pytest.mark.unit
def test_build_research_workbook_status_snapshot_aggregates_recursive_progress(tmp_path: Path) -> None:
    apple = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path / "AAPL",
        ticker="AAPL",
        company="Apple Inc.",
    )
    hsbc = materialize_research_template_bundle(
        "financial",
        workspace_root=tmp_path / "0005.HK",
        ticker="0005.HK",
        company="HSBC Holdings",
    )
    apple_workbook = Path(str(apple["workbook_file"]))
    apple_payload = json.loads(apple_workbook.read_text(encoding="utf-8"))
    apple_item_id = apple_payload["sections"][0]["items"][0]["item_id"]
    write_research_workbook_update(
        apple_workbook,
        item_id=apple_item_id,
        status="in_progress",
        analyst_notes="等待下一季留存数据。",
    )

    direct = build_research_workbook_status_snapshot(tmp_path)
    recursive = build_research_workbook_status_snapshot(tmp_path, recursive=True)

    assert direct["overall_status"] == "no_workbooks"
    assert recursive["overall_status"] == "in_progress"
    summary = recursive["summary"]
    assert isinstance(summary, dict)
    assert summary["workbook_count"] == 2
    assert summary["valid_workbook_count"] == 2
    assert summary["item_count"] == 74
    completion_counts = summary["completion_counts"]
    assert isinstance(completion_counts, dict)
    assert completion_counts == {"not_started": 1, "in_progress": 1, "complete": 0}
    workbooks = recursive["workbooks"]
    assert isinstance(workbooks, list)
    assert {item["research_target"]["ticker"] for item in workbooks} == {"AAPL", "0005.HK"}
    assert Path(str(hsbc["workbook_file"])).exists()


@pytest.mark.unit
def test_build_research_workbook_status_snapshot_exposes_invalid_workbook(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("consumer", workspace_root=tmp_path)
    workbook_path = Path(str(materialized["workbook_file"]))
    workbook_path.write_text("{broken", encoding="utf-8")

    snapshot = build_research_workbook_status_snapshot(tmp_path)

    assert snapshot["overall_status"] == "unhealthy"
    summary = snapshot["summary"]
    assert isinstance(summary, dict)
    assert summary["workbook_count"] == 1
    assert summary["valid_workbook_count"] == 0
    assert summary["invalid_workbook_count"] == 1
    assert summary["item_count"] == 0


@pytest.mark.unit
def test_write_research_workbook_status_snapshot_requires_overwrite(tmp_path: Path) -> None:
    materialize_research_template_bundle("common", workspace_root=tmp_path)

    status_path = write_research_workbook_status_snapshot(tmp_path)

    assert status_path == (tmp_path / "assets" / "research_templates" / "research-workbook-status.json").resolve()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "not_started"
    with pytest.raises(FileExistsError):
        write_research_workbook_status_snapshot(tmp_path)


@pytest.mark.unit
def test_build_research_workbook_report_renders_answer_evidence_and_open_gaps() -> None:
    payload = build_research_workbook_payload("consumer", ticker="600519", company="贵州茅台")
    sections = payload["sections"]
    assert isinstance(sections, list)
    item_id = str(sections[0]["items"][0]["item_id"])
    preview = build_research_workbook_update_preview(
        payload,
        item_id=item_id,
        status="answered",
        response="需求增长由同店销售改善驱动。",
        analyst_notes="下一季继续验证库存。",
        evidence_records=[
            {
                "source": "annual_report",
                "reference": "2025 annual report p.42",
                "finding": "同店销售同比增长 8%。",
            }
        ],
    )
    workbook = preview["workbook"]
    assert isinstance(workbook, dict)

    report = build_research_workbook_report(workbook)

    assert "# Research Workbook Progress: 贵州茅台" in report
    assert f"### [answered] `{item_id}`" in report
    assert "**Response:** 需求增长由同店销售改善驱动。" in report
    assert "`annual_report` | 2025 annual report p.42 | 同店销售同比增长 8%。" in report
    assert "**Analyst notes:** 下一季继续验证库存。" in report
    assert "## Open Research Gaps" in report
    assert "[open]" in report


@pytest.mark.unit
def test_build_research_workbook_report_rejects_invalid_workbook() -> None:
    payload = build_research_workbook_payload("technology")
    payload["source_template_fingerprint"] = "0" * 64

    with pytest.raises(ValueError, match="research workbook is invalid"):
        build_research_workbook_report(payload)


@pytest.mark.unit
def test_write_research_workbook_report_requires_overwrite(tmp_path: Path) -> None:
    workbook_path = write_research_workbook_payload("financial", workspace_root=tmp_path)

    report_path = write_research_workbook_report(workbook_path)

    assert report_path == workbook_path.with_name("financial.research-progress.md")
    assert "Research Workbook Progress" in report_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_research_workbook_report(workbook_path)


@pytest.mark.unit
def test_inspect_workbook_report_detects_stale_source_and_refreshes(tmp_path: Path) -> None:
    workbook_path = write_research_workbook_payload("consumer", workspace_root=tmp_path)
    report_path = write_research_workbook_report(workbook_path)

    initial = inspect_research_workbook_report(report_path, workbook_path)

    assert initial["stale"] is False
    assert initial["report_tampered"] is False
    initial_validation = initial["validation"]
    assert isinstance(initial_validation, dict)
    assert initial_validation["ok"] is True
    metadata = initial["metadata"]
    assert isinstance(metadata, dict)
    assert len(str(metadata["workbook_semantic_fingerprint"])) == 64
    assert len(str(metadata["report_body_fingerprint"])) == 64

    payload = json.loads(workbook_path.read_text(encoding="utf-8"))
    workbook_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    reformatted = inspect_research_workbook_report(report_path, workbook_path)
    reformatted_validation = reformatted["validation"]
    assert isinstance(reformatted_validation, dict)
    assert reformatted_validation["ok"] is True

    item_id = payload["sections"][0]["items"][0]["item_id"]
    write_research_workbook_update(
        workbook_path,
        item_id=item_id,
        status="in_progress",
        analyst_notes="等待新证据。",
    )
    stale = inspect_research_workbook_report(report_path, workbook_path)

    assert stale["stale"] is True
    stale_validation = stale["validation"]
    assert isinstance(stale_validation, dict)
    assert stale_validation["ok"] is False
    assert any("report is stale" in error for error in stale_validation["errors"])

    write_research_workbook_report(workbook_path, overwrite=True)
    refreshed = inspect_research_workbook_report(report_path, workbook_path)
    refreshed_validation = refreshed["validation"]
    assert isinstance(refreshed_validation, dict)
    assert refreshed_validation["ok"] is True
    assert refreshed["stale"] is False


@pytest.mark.unit
def test_inspect_workbook_report_detects_body_tampering(tmp_path: Path) -> None:
    workbook_path = write_research_workbook_payload("financial", workspace_root=tmp_path)
    report_path = write_research_workbook_report(workbook_path)
    report_path.write_text(
        report_path.read_text(encoding="utf-8") + "\nmanual edit\n",
        encoding="utf-8",
    )

    inspected = inspect_research_workbook_report(report_path, workbook_path)

    assert inspected["report_tampered"] is True
    validation = inspected["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is False
    assert any("body fingerprint" in error for error in validation["errors"])


@pytest.mark.unit
def test_workbook_report_status_aggregates_recursive_health_states(tmp_path: Path) -> None:
    materialized: dict[str, dict[str, object]] = {}
    for ticker, template in (
        ("AAPL", "technology"),
        ("MSFT", "technology"),
        ("0005.HK", "financial"),
        ("600519", "consumer"),
    ):
        bundle = materialize_research_template_bundle(
            template,
            workspace_root=tmp_path / ticker,
            ticker=ticker,
        )
        materialized[ticker] = bundle

    msft_workbook = Path(str(materialized["MSFT"]["workbook_file"]))
    msft_payload = json.loads(msft_workbook.read_text(encoding="utf-8"))
    msft_item_id = msft_payload["sections"][0]["items"][0]["item_id"]
    write_research_workbook_update(msft_workbook, item_id=msft_item_id, status="in_progress")

    hsbc_report = Path(str(materialized["0005.HK"]["workbook_file"])).with_name("financial.research-progress.md")
    hsbc_report.write_text(hsbc_report.read_text(encoding="utf-8") + "\nmanual edit\n", encoding="utf-8")
    Path(str(materialized["600519"]["workbook_file"])).unlink()

    direct = build_research_workbook_report_status_snapshot(tmp_path)
    recursive = build_research_workbook_report_status_snapshot(tmp_path, recursive=True)

    assert direct["overall_status"] == "no_reports"
    assert recursive["overall_status"] == "unhealthy"
    summary = recursive["summary"]
    assert isinstance(summary, dict)
    assert summary == {
        "report_count": 4,
        "current_report_count": 1,
        "invalid_report_count": 3,
        "stale_report_count": 1,
        "tampered_report_count": 1,
        "missing_workbook_count": 1,
    }


@pytest.mark.unit
def test_write_workbook_report_status_snapshot_requires_overwrite(tmp_path: Path) -> None:
    materialize_research_template_bundle("common", workspace_root=tmp_path)

    status_path = write_research_workbook_report_status_snapshot(tmp_path)

    assert (
        status_path == (tmp_path / "assets" / "research_templates" / "research-workbook-report-status.json").resolve()
    )
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "current"
    with pytest.raises(FileExistsError):
        write_research_workbook_report_status_snapshot(tmp_path)


@pytest.mark.unit
def test_build_monitoring_source_map_payload_uses_candidate_sources() -> None:
    payload = build_monitoring_source_map_payload("financial")

    assert payload["template"] == "financial"
    assert payload["source_map_type"] == "monitoring_data_binding_draft"
    data_sources = payload["data_sources"]
    assert isinstance(data_sources, list)
    regulatory = next(item for item in data_sources if item["source"] == "regulatory_filings")
    assert regulatory["provider_type"] == "dayu_fins_tool"
    assert "search_document" in regulatory["candidate_tools"]
    assert "regulatory_disclosure" in regulatory["candidate_fields"]
    assert regulatory["binding_status"] == "unbound"


@pytest.mark.unit
def test_write_monitoring_source_map_payload_defaults_to_workspace_assets(tmp_path: Path) -> None:
    source_map_path = write_monitoring_source_map_payload("consumer", workspace_root=tmp_path)

    assert source_map_path == (tmp_path / "assets" / "research_templates" / "consumer.source-map.json").resolve()
    payload = json.loads(source_map_path.read_text(encoding="utf-8"))
    assert payload["template"] == "consumer"
    assert len(payload["data_sources"]) >= 4


@pytest.mark.unit
def test_validate_monitoring_source_map_payload_accepts_matching_payloads() -> None:
    rules_payload = build_monitoring_rules_payload("financial")
    source_map_payload = build_monitoring_source_map_payload("financial")

    result = validate_monitoring_source_map_payload(rules_payload, source_map_payload)

    assert result["ok"] is True
    assert result["errors"] == []


@pytest.mark.unit
def test_validate_monitoring_source_map_payload_reports_missing_source() -> None:
    rules_payload = build_monitoring_rules_payload("financial")
    source_map_payload = build_monitoring_source_map_payload("financial")
    data_sources = source_map_payload["data_sources"]
    assert isinstance(data_sources, list)
    source_map_payload["data_sources"] = [item for item in data_sources if item["source"] != "market_data"]

    result = validate_monitoring_source_map_payload(rules_payload, source_map_payload)

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert any("market_data" in error for error in errors)


@pytest.mark.unit
def test_validate_monitoring_source_map_payload_reports_template_mismatch() -> None:
    rules_payload = build_monitoring_rules_payload("financial")
    source_map_payload = build_monitoring_source_map_payload("consumer")

    result = validate_monitoring_source_map_payload(rules_payload, source_map_payload)

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert any("template mismatch" in error for error in errors)


@pytest.mark.unit
def test_build_monitoring_source_binding_preview_binds_declared_dayu_source() -> None:
    source_map = build_monitoring_source_map_payload("consumer")
    approval = {
        "schema_version": 1,
        "approval_type": "research_monitoring_source_binding",
        "template": "consumer",
        "approved_by": "research-owner",
        "approval_reference": "review-2026-001",
        "bindings": [
            {
                "source": "financial_statements",
                "selected_tool": "get_financial_statement",
                "selected_fields": ["revenue", "gross_profit"],
            }
        ],
    }

    preview = build_monitoring_source_binding_preview(source_map, approval)

    assert preview["changed_sources"] == ["financial_statements"]
    assert preview["resulting_binding_status"] == "partially_bound"
    updated = preview["source_map"]
    assert isinstance(updated, dict)
    sources = updated["data_sources"]
    assert isinstance(sources, list)
    financials = next(source for source in sources if source["source"] == "financial_statements")
    assert financials["binding_status"] == "bound"
    assert financials["selected_tool"] == "get_financial_statement"
    assert financials["binding_approval"]["approved_by"] == "research-owner"


@pytest.mark.unit
def test_build_monitoring_source_binding_preview_rejects_external_placeholder() -> None:
    source_map = build_monitoring_source_map_payload("consumer")
    approval = {
        "schema_version": 1,
        "approval_type": "research_monitoring_source_binding",
        "template": "consumer",
        "approved_by": "research-owner",
        "approval_reference": "review-2026-002",
        "bindings": [
            {
                "source": "channel_checks",
                "selected_tool": "manual_check",
                "selected_fields": ["store_count"],
            }
        ],
    }

    with pytest.raises(ValueError, match="not an implemented dayu_fins_tool provider"):
        build_monitoring_source_binding_preview(source_map, approval)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("selected_tool", "selected_fields", "message"),
    [
        ("unknown_tool", ["revenue"], "selected_tool is not a declared candidate"),
        ("get_financial_statement", ["unknown_field"], "selected_fields are not declared candidates"),
    ],
)
def test_build_monitoring_source_binding_preview_rejects_undeclared_choices(
    selected_tool: str,
    selected_fields: list[str],
    message: str,
) -> None:
    source_map = build_monitoring_source_map_payload("consumer")
    approval = {
        "schema_version": 1,
        "approval_type": "research_monitoring_source_binding",
        "template": "consumer",
        "approved_by": "research-owner",
        "approval_reference": "review-2026-003",
        "bindings": [
            {
                "source": "financial_statements",
                "selected_tool": selected_tool,
                "selected_fields": selected_fields,
            }
        ],
    }

    with pytest.raises(ValueError, match=message):
        build_monitoring_source_binding_preview(source_map, approval)


@pytest.mark.unit
def test_write_monitoring_source_binding_approval_backs_up_and_invalidates_old_plan(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle(
        "consumer",
        workspace_root=tmp_path,
        ticker="600519",
    )
    source_map_path = Path(str(materialized["source_map_file"]))
    original_bytes = source_map_path.read_bytes()
    plan_path = write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "approval_type": "research_monitoring_source_binding",
                "template": "consumer",
                "approved_by": "research-owner",
                "approval_reference": "review-2026-004",
                "bindings": [
                    {
                        "source": "financial_statements",
                        "selected_tool": "get_financial_statement",
                        "selected_fields": ["revenue", "gross_profit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = write_monitoring_source_binding_approval(source_map_path, approval_path)

    backup_path = Path(str(result["backup_file"]))
    assert backup_path.exists()
    assert backup_path.read_bytes() == original_bytes
    assert result["source_map_fingerprint_before"] != result["source_map_fingerprint_after"]
    old_plan = inspect_monitoring_execution_plan(plan_path)
    old_validation = old_plan["validation"]
    assert isinstance(old_validation, dict)
    assert old_validation["ok"] is False

    write_monitoring_execution_plan(Path(str(materialized["bundle_file"])), overwrite=True)
    refreshed_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert refreshed_plan["readiness"]["status"] == "ready_for_review"


@pytest.mark.unit
def test_build_source_binding_rollback_preview_does_not_write(tmp_path: Path) -> None:
    source_map_path = write_monitoring_source_map_payload("consumer", workspace_root=tmp_path)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "approval_type": "research_monitoring_source_binding",
                "template": "consumer",
                "approved_by": "research-owner",
                "approval_reference": "review-2026-006",
                "bindings": [
                    {
                        "source": "financial_statements",
                        "selected_tool": "get_financial_statement",
                        "selected_fields": ["revenue"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    binding_result = write_monitoring_source_binding_approval(source_map_path, approval_path)
    backup_path = Path(str(binding_result["backup_file"]))
    bound_bytes = source_map_path.read_bytes()

    preview = build_monitoring_source_binding_rollback_preview(source_map_path, backup_path)

    assert preview["preview_type"] == "research_monitoring_source_binding_rollback"
    assert preview["snapshot_type"] == "before_bindings"
    assert preview["binding_status_before"] == "partially_bound"
    assert preview["binding_status_after"] == "unbound"
    assert preview["changed_sources"] == ["financial_statements"]
    assert source_map_path.read_bytes() == bound_bytes
    assert not list(source_map_path.parent.glob("*.before-rollback.*.json"))


@pytest.mark.unit
@pytest.mark.parametrize("mutation", ["template", "source_set"])
def test_build_source_binding_rollback_preview_rejects_mismatched_backup(
    tmp_path: Path,
    mutation: str,
) -> None:
    source_map_path = write_monitoring_source_map_payload("consumer", workspace_root=tmp_path)
    backup_payload = json.loads(source_map_path.read_text(encoding="utf-8"))
    if mutation == "template":
        backup_payload["template"] = "financial"
        expected_error = "template mismatch"
    else:
        backup_payload["data_sources"] = backup_payload["data_sources"][:-1]
        expected_error = "source set mismatch"
    backup_bytes = (json.dumps(backup_payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    fingerprint = hashlib.sha256(backup_bytes).hexdigest()
    backup_path = source_map_path.with_name(f"{source_map_path.stem}.before-bindings.{fingerprint[:12]}.json")
    backup_path.write_bytes(backup_bytes)

    with pytest.raises(ValueError, match=expected_error):
        build_monitoring_source_binding_rollback_preview(source_map_path, backup_path)


@pytest.mark.unit
def test_build_source_binding_rollback_preview_rejects_forged_filename(tmp_path: Path) -> None:
    source_map_path = write_monitoring_source_map_payload("consumer", workspace_root=tmp_path)
    forged_backup = source_map_path.with_name(f"{source_map_path.stem}.before-bindings.000000000000.json")
    forged_backup.write_bytes(source_map_path.read_bytes())

    with pytest.raises(ValueError, match="filename does not match"):
        build_monitoring_source_binding_rollback_preview(source_map_path, forged_backup)


@pytest.mark.unit
def test_write_source_binding_rollback_restores_and_invalidates_ready_plan(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle(
        "consumer",
        workspace_root=tmp_path,
        ticker="600519",
    )
    source_map_path = Path(str(materialized["source_map_file"]))
    original_bytes = source_map_path.read_bytes()
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "approval_type": "research_monitoring_source_binding",
                "template": "consumer",
                "approved_by": "research-owner",
                "approval_reference": "review-2026-007",
                "bindings": [
                    {
                        "source": "financial_statements",
                        "selected_tool": "get_financial_statement",
                        "selected_fields": ["revenue", "gross_profit"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    binding_result = write_monitoring_source_binding_approval(source_map_path, approval_path)
    binding_backup_path = Path(str(binding_result["backup_file"]))
    bound_bytes = source_map_path.read_bytes()
    bundle_path = Path(str(materialized["bundle_file"]))
    plan_path = write_monitoring_execution_plan(bundle_path)
    ready_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert ready_plan["readiness"]["status"] == "ready_for_review"

    rollback_result = write_monitoring_source_binding_rollback(source_map_path, binding_backup_path)

    rollback_backup_path = Path(str(rollback_result["rollback_backup_file"]))
    assert rollback_backup_path.read_bytes() == bound_bytes
    assert source_map_path.read_bytes() == original_bytes
    stale_plan = inspect_monitoring_execution_plan(plan_path)
    stale_validation = stale_plan["validation"]
    assert isinstance(stale_validation, dict)
    assert stale_validation["ok"] is False

    write_monitoring_execution_plan(bundle_path, overwrite=True)
    refreshed_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert refreshed_plan["readiness"]["status"] == "blocked_unbound_sources"

    restore_result = write_monitoring_source_binding_rollback(source_map_path, rollback_backup_path)

    restore_preview = restore_result["preview"]
    assert isinstance(restore_preview, dict)
    assert restore_preview["snapshot_type"] == "before_rollback"
    assert source_map_path.read_bytes() == bound_bytes
    stale_blocked_plan = inspect_monitoring_execution_plan(plan_path)
    stale_blocked_validation = stale_blocked_plan["validation"]
    assert isinstance(stale_blocked_validation, dict)
    assert stale_blocked_validation["ok"] is False

    write_monitoring_execution_plan(bundle_path, overwrite=True)
    restored_plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert restored_plan["readiness"]["status"] == "ready_for_review"


@pytest.mark.unit
def test_inspect_source_binding_history_reports_binding_and_rollback_snapshots(tmp_path: Path) -> None:
    source_map_path = write_monitoring_source_map_payload("consumer", workspace_root=tmp_path)
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "approval_type": "research_monitoring_source_binding",
                "template": "consumer",
                "approved_by": "research-owner",
                "approval_reference": "review-2026-009",
                "bindings": [
                    {
                        "source": "financial_statements",
                        "selected_tool": "get_financial_statement",
                        "selected_fields": ["revenue"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    binding_result = write_monitoring_source_binding_approval(source_map_path, approval_path)
    write_monitoring_source_binding_rollback(source_map_path, Path(str(binding_result["backup_file"])))

    history = inspect_monitoring_source_binding_history(source_map_path)

    validation = history["validation"]
    current = history["current"]
    summary = history["summary"]
    assert isinstance(validation, dict)
    assert isinstance(current, dict)
    assert isinstance(summary, dict)
    assert validation["ok"] is True
    assert current["binding_status"] == "unbound"
    assert summary == {
        "snapshot_count": 2,
        "valid_snapshot_count": 2,
        "invalid_snapshot_count": 0,
        "before_bindings_count": 1,
        "before_rollback_count": 1,
    }
    snapshots = history["snapshots"]
    assert isinstance(snapshots, list)
    snapshots_by_type = {snapshot["snapshot_type"]: snapshot for snapshot in snapshots}
    assert snapshots_by_type["before_bindings"]["binding_status"] == "unbound"
    assert snapshots_by_type["before_rollback"]["binding_status"] == "partially_bound"
    assert snapshots_by_type["before_rollback"]["bound_sources"] == ["financial_statements"]


@pytest.mark.unit
def test_inspect_source_binding_history_exposes_forged_snapshot(tmp_path: Path) -> None:
    source_map_path = write_monitoring_source_map_payload("consumer", workspace_root=tmp_path)
    forged_snapshot = source_map_path.with_name(f"{source_map_path.stem}.before-bindings.000000000000.json")
    forged_snapshot.write_bytes(source_map_path.read_bytes())

    history = inspect_monitoring_source_binding_history(source_map_path)

    validation = history["validation"]
    summary = history["summary"]
    snapshots = history["snapshots"]
    assert isinstance(validation, dict)
    assert isinstance(summary, dict)
    assert isinstance(snapshots, list)
    assert validation["ok"] is False
    assert summary["invalid_snapshot_count"] == 1
    snapshot = snapshots[0]
    assert snapshot["ok"] is False
    assert any("filename fingerprint mismatch" in error for error in snapshot["errors"])


@pytest.mark.unit
def test_build_research_template_package_manifest_summarizes_all_templates() -> None:
    payload = build_research_template_package_manifest()

    assert payload["manifest_type"] == "research_template_package"
    templates = payload["templates"]
    assert isinstance(templates, list)
    names = {item["name"] for item in templates}
    assert {"common", "consumer", "cyclical", "technology", "financial"} <= names
    assert all(item["validation"]["ok"] is True for item in templates)


@pytest.mark.unit
def test_write_research_template_package_manifest_defaults_to_workspace_assets(tmp_path: Path) -> None:
    manifest_path = write_research_template_package_manifest(workspace_root=tmp_path)

    assert manifest_path == (tmp_path / "assets" / "research_templates" / "research-template.manifest.json").resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["manifest_type"] == "research_template_package"


@pytest.mark.unit
def test_build_research_template_usage_guide_includes_write_command() -> None:
    guide = build_research_template_usage_guide("consumer")

    assert "# Research Template Usage Guide: consumer" in guide
    assert "dayu-cli write --ticker <TICKER> --template" in guide
    assert "## Monitoring Variables" in guide
    assert "## Data Source Candidates" in guide
    assert '--template "./workspace/assets/research_templates/common-plus-consumer.md"' in guide


@pytest.mark.unit
def test_write_research_template_usage_guide_defaults_to_workspace_assets(tmp_path: Path) -> None:
    guide_path = write_research_template_usage_guide("cyclical", workspace_root=tmp_path)

    assert guide_path == (tmp_path / "assets" / "research_templates" / "cyclical.research-guide.md").resolve()
    guide = guide_path.read_text(encoding="utf-8")
    assert "Research Template Usage Guide: cyclical" in guide
    assert "dayu-cli write --ticker <TICKER> --template" in guide


@pytest.mark.unit
def test_build_and_validate_research_template_bundle_descriptor(tmp_path: Path) -> None:
    artifacts = {}
    for name in ("template", "workbook", "rules", "source-map", "manifest", "guide"):
        path = tmp_path / name
        path.write_text("fixture", encoding="utf-8")
        artifacts[name] = path
    artifacts["workbook"].write_text(
        json.dumps(build_research_workbook_payload("consumer"), ensure_ascii=False),
        encoding="utf-8",
    )
    artifacts["rules"].write_text(
        json.dumps(build_monitoring_rules_payload("consumer"), ensure_ascii=False),
        encoding="utf-8",
    )
    artifacts["source-map"].write_text(
        json.dumps(build_monitoring_source_map_payload("consumer"), ensure_ascii=False),
        encoding="utf-8",
    )
    monitoring_validation: dict[str, object] = {"ok": True, "errors": [], "warnings": []}

    payload = build_research_template_bundle_descriptor(
        "consumer",
        template_file=artifacts["template"],
        workbook_file=artifacts["workbook"],
        rules_file=artifacts["rules"],
        source_map_file=artifacts["source-map"],
        manifest_file=artifacts["manifest"],
        guide_file=artifacts["guide"],
        monitoring_validation=monitoring_validation,
    )
    result = validate_research_template_bundle_descriptor(payload)

    assert payload["bundle_type"] == "research_template_bundle"
    assert payload["automation_status"] == "manual_review"
    assert result["ok"] is True
    assert result["artifact_count"] == 6
    workbook_validation = result["workbook_validation"]
    assert isinstance(workbook_validation, dict)
    assert workbook_validation["ok"] is True


@pytest.mark.unit
def test_validate_research_template_bundle_descriptor_reports_missing_artifact(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "bundle_type": "research_template_bundle",
        "template": "consumer",
        "research_target": {"ticker": "", "company": ""},
        "automation_status": "manual_review",
        "artifacts": {
            key: str(tmp_path / key)
            for key in (
                "write_template",
                "research_workbook",
                "monitoring_rules",
                "source_map",
                "package_manifest",
                "usage_guide",
            )
        },
        "capabilities": {
            "write_report": True,
            "track_research_evidence": True,
            "review_monitoring_rules": True,
            "review_source_bindings": True,
            "automated_monitoring": False,
        },
        "monitoring_validation": {"ok": True},
    }

    result = validate_research_template_bundle_descriptor(payload)

    assert result["ok"] is False
    errors = result["errors"]
    assert isinstance(errors, list)
    assert any("does not exist" in str(error) for error in errors)


@pytest.mark.unit
def test_bundle_validation_rejects_invalid_or_mismatched_workbook(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle(
        "consumer",
        workspace_root=tmp_path,
        ticker="600519",
        company="贵州茅台",
    )
    workbook_path = Path(str(materialized["workbook_file"]))
    workbook = json.loads(workbook_path.read_text(encoding="utf-8"))
    workbook["sections"][0]["items"][0]["status"] = "answered"
    workbook["research_target"]["ticker"] = "000001"
    workbook_path.write_text(json.dumps(workbook, ensure_ascii=False), encoding="utf-8")

    inspected = inspect_research_template_bundle(Path(str(materialized["bundle_file"])))

    validation = inspected["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is False
    errors = validation["errors"]
    assert isinstance(errors, list)
    assert any("response is required" in error for error in errors)
    assert any("evidence is required" in error for error in errors)
    assert any("research_target does not match bundle" in error for error in errors)


@pytest.mark.unit
def test_bundle_validation_requires_progress_report_refresh_after_workbook_update(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("financial", workspace_root=tmp_path)
    workbook_path = Path(str(materialized["workbook_file"]))
    workbook = json.loads(workbook_path.read_text(encoding="utf-8"))
    item = workbook["sections"][0]["items"][0]
    item["status"] = "answered"
    item["response"] = "净息差仍是主要利润驱动。"
    item["evidence"] = [
        {
            "source": "annual_report",
            "reference": "2025 annual report p.18",
            "finding": "净利息收入占营业收入 68%。",
        }
    ]
    workbook_path.write_text(json.dumps(workbook, ensure_ascii=False), encoding="utf-8")

    stale = inspect_research_template_bundle(Path(str(materialized["bundle_file"])))

    stale_validation = stale["validation"]
    assert isinstance(stale_validation, dict)
    assert stale_validation["ok"] is False
    assert any("report is stale" in error for error in stale_validation["errors"])

    write_research_workbook_report(workbook_path, overwrite=True)
    inspected = inspect_research_template_bundle(Path(str(materialized["bundle_file"])))

    validation = inspected["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is True
    warnings = validation["warnings"]
    assert isinstance(warnings, list)
    assert any("summary.open_item_count is stale" in warning for warning in warnings)


@pytest.mark.unit
def test_bundle_validation_detects_unauthorized_source_binding_flip(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("financial", workspace_root=tmp_path)
    source_map_path = Path(str(materialized["source_map_file"]))
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    # Flip an unbound source to bound WITHOUT the approval provenance the
    # sanctioned binding flow writes; keep the embedded monitoring_validation
    # snapshot untouched (a tamper would not rewrite the bundle descriptor).
    source_map["data_sources"][0]["binding_status"] = "bound"
    source_map_path.write_text(json.dumps(source_map, ensure_ascii=False), encoding="utf-8")

    inspected = inspect_research_template_bundle(Path(str(materialized["bundle_file"])))

    validation = inspected["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is False
    errors = validation["errors"]
    assert isinstance(errors, list)
    assert any("bound source lacks binding_approval provenance" in error for error in errors)


@pytest.mark.unit
def test_bundle_validation_detects_source_map_template_tamper(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("financial", workspace_root=tmp_path)
    source_map_path = Path(str(materialized["source_map_file"]))
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    source_map["template"] = "consumer"
    source_map_path.write_text(json.dumps(source_map, ensure_ascii=False), encoding="utf-8")

    inspected = inspect_research_template_bundle(Path(str(materialized["bundle_file"])))

    validation = inspected["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is False
    errors = validation["errors"]
    assert isinstance(errors, list)
    assert any("monitoring integrity" in error for error in errors)


@pytest.mark.unit
def test_write_research_template_bundle_descriptor_defaults_to_workspace_assets(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("common", workspace_root=tmp_path)

    bundle_path = Path(str(materialized["bundle_file"]))
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle_path == (tmp_path / "assets" / "research_templates" / "common.bundle.json").resolve()
    assert payload["artifacts"]["write_template"].endswith("common.md")


@pytest.mark.unit
def test_materialized_research_target_propagates_to_guide_bundle_plan_and_status(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path,
        ticker="0700.hk",
        company="Tencent Holdings",
    )

    assert materialized["research_target"] == {"ticker": "0700.HK", "company": "Tencent Holdings"}
    guide = Path(str(materialized["guide_file"])).read_text(encoding="utf-8")
    assert "Ticker: `0700.HK`" in guide
    assert "Company: `Tencent Holdings`" in guide
    assert "dayu-cli write --ticker 0700.HK" in guide
    assert "research_progress_report" in guide
    bundle = json.loads(Path(str(materialized["bundle_file"])).read_text(encoding="utf-8"))
    assert bundle["research_target"]["ticker"] == "0700.HK"
    workbook = json.loads(Path(str(materialized["workbook_file"])).read_text(encoding="utf-8"))
    assert workbook["research_target"] == {"ticker": "0700.HK", "company": "Tencent Holdings"}
    assert bundle["artifacts"]["research_workbook"] == str(Path(str(materialized["workbook_file"])))
    assert bundle["artifacts"]["research_progress_report"] == str(Path(str(materialized["progress_report_file"])))

    plan_path = write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["research_target"]["company"] == "Tencent Holdings"
    assert plan["tasks"][0]["task_id"].startswith("0700-hk-technology-monitor-")
    snapshot = build_monitoring_status_snapshot(tmp_path)
    summary = snapshot["summary"]
    assert isinstance(summary, dict)
    assert summary["targeted_plan_count"] == 1
    assert summary["untargeted_plan_count"] == 0


@pytest.mark.unit
def test_recursive_discovery_and_status_aggregate_ticker_workspaces(tmp_path: Path) -> None:
    apple = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path / "AAPL",
        ticker="AAPL",
        company="Apple Inc.",
    )
    write_monitoring_execution_plan(Path(str(apple["bundle_file"])))
    microsoft = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path / "MSFT",
        ticker="MSFT",
        company="Microsoft Corp.",
    )
    write_monitoring_execution_plan(Path(str(microsoft["bundle_file"])))

    assert discover_research_template_bundles(tmp_path) == ()
    assert discover_monitoring_execution_plans(tmp_path) == ()
    assert len(discover_research_template_bundles(tmp_path, recursive=True)) == 2
    assert len(discover_monitoring_execution_plans(tmp_path, recursive=True)) == 2

    snapshot = build_monitoring_status_snapshot(tmp_path, recursive=True)
    assert snapshot["scan_scope"] == "recursive"
    assert snapshot["overall_status"] == "blocked"
    summary = snapshot["summary"]
    assert isinstance(summary, dict)
    assert summary["plan_count"] == 2
    assert summary["target_count"] == 2
    targets = snapshot["targets"]
    assert isinstance(targets, list)
    assert [target["ticker"] for target in targets] == ["AAPL", "MSFT"]

    microsoft_rules = Path(str(microsoft["rules_file"]))
    microsoft_rules.write_text(microsoft_rules.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    stale_snapshot = build_monitoring_status_snapshot(tmp_path, recursive=True)
    assert stale_snapshot["overall_status"] == "unhealthy"
    stale_targets = stale_snapshot["targets"]
    assert isinstance(stale_targets, list)
    statuses = {target["ticker"]: target["overall_status"] for target in stale_targets}
    assert statuses == {"AAPL": "blocked", "MSFT": "unhealthy"}


@pytest.mark.unit
def test_materialize_research_portfolio_creates_isolated_targets_and_report(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portfolio_type": "research_monitoring_portfolio",
                "targets": [
                    {"ticker": "AAPL", "company": "Apple Inc.", "template": "technology"},
                    {"ticker": "600519", "company": "Kweichow Moutai", "template": "consumer"},
                ],
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"

    report = materialize_research_portfolio(portfolio_path, workspace_root=workspace)

    assert report["ok"] is True
    assert report["success_count"] == 2
    assert report["failure_count"] == 0
    assert Path(str(report["report_file"])).exists()
    assert Path(str(report["monitoring_status_file"])).exists()
    assert Path(str(report["workbook_status_file"])).exists()
    assert Path(str(report["report_status_file"])).exists()
    results = report["results"]
    assert isinstance(results, list)
    assert {result["ticker"] for result in results} == {"AAPL", "600519"}
    assert all(Path(str(result["bundle_file"])).exists() for result in results)
    assert all(Path(str(result["workbook_file"])).exists() for result in results)
    assert all(Path(str(result["monitoring_plan_file"])).exists() for result in results)
    assert all(Path(str(result["monitoring_status_file"])).exists() for result in results)
    assert all(Path(str(result["workbook_status_file"])).exists() for result in results)
    assert all(Path(str(result["report_status_file"])).exists() for result in results)
    monitoring_status = report["monitoring_status"]
    assert isinstance(monitoring_status, dict)
    assert monitoring_status["summary"]["target_count"] == 2
    workbook_status = report["workbook_status"]
    assert isinstance(workbook_status, dict)
    assert workbook_status["overall_status"] == "not_started"
    workbook_summary = workbook_status["summary"]
    assert isinstance(workbook_summary, dict)
    assert workbook_summary["workbook_count"] == 2
    assert workbook_summary["completion_counts"] == {
        "not_started": 2,
        "in_progress": 0,
        "complete": 0,
    }
    report_status = report["report_status"]
    assert isinstance(report_status, dict)
    assert report_status["overall_status"] == "current"
    assert report_status["summary"]["current_report_count"] == 2


@pytest.mark.unit
def test_materialize_research_portfolio_resolves_relative_write_manifest(tmp_path: Path) -> None:
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    write_manifest = manifests_dir / "0005.json"
    write_manifest.write_text(
        json.dumps(
            {
                "config": {
                    "ticker": "0005.hk",
                    "company": "HSBC Holdings",
                    "research_template_requested_name": "auto",
                    "research_template_resolved_name": "consumer",
                    "research_template_selection_mode": "auto",
                },
                "company_facets": {"business_model_tags": ["银行"], "constraint_tags": ["利率敏感"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portfolio_type": "research_monitoring_portfolio",
                "targets": [{"write_manifest": "manifests/0005.json"}],
            }
        ),
        encoding="utf-8",
    )

    report = materialize_research_portfolio(portfolio_path, workspace_root=tmp_path / "workspace")

    results = report["results"]
    assert isinstance(results, list)
    assert results[0]["ticker"] == "0005.HK"
    assert results[0]["template"] == "consumer"
    assert results[0]["selection"]["selection_mode"] == "manifest_provenance"
    assert results[0]["selection"]["write_selection"]["resolved_name"] == "consumer"


@pytest.mark.unit
def test_materialize_research_portfolio_preflight_rejects_duplicates_before_writes(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portfolio_type": "research_monitoring_portfolio",
                "targets": [
                    {"ticker": "AAPL", "template": "technology"},
                    {"ticker": "aapl", "template": "consumer"},
                ],
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"

    with pytest.raises(ValueError, match="duplicate portfolio ticker"):
        materialize_research_portfolio(portfolio_path, workspace_root=workspace)

    assert not workspace.exists()


@pytest.mark.unit
def test_build_research_portfolio_preview_is_no_write_and_reports_artifacts(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portfolio_type": "research_monitoring_portfolio",
                "targets": [{"ticker": "AAPL", "company": "Apple Inc.", "template": "technology"}],
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"

    preview = build_research_portfolio_preview(portfolio_path, workspace_root=workspace)

    assert preview["can_materialize"] is True
    assert not workspace.exists()
    summary = preview["summary"]
    assert isinstance(summary, dict)
    assert summary == {"target_count": 1, "create_count": 1, "overwrite_count": 0, "blocked_count": 0}
    targets = preview["targets"]
    assert isinstance(targets, list)
    assert targets[0]["action"] == "create"
    assert str(targets[0]["artifacts"]["bundle"]).endswith("technology.bundle.json")
    assert str(targets[0]["artifacts"]["research_workbook"]).endswith("technology.research-workbook.json")
    derived_outputs = preview["derived_outputs"]
    assert isinstance(derived_outputs, dict)
    assert str(derived_outputs["workbook_status"]).endswith("research-workbook-status.json")
    assert str(derived_outputs["report_status"]).endswith("research-workbook-report-status.json")


@pytest.mark.unit
def test_build_research_portfolio_preview_classifies_conflict_and_overwrite(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portfolio_type": "research_monitoring_portfolio",
                "targets": [{"ticker": "AAPL", "template": "technology"}],
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    conflict = workspace / "AAPL" / "assets" / "research_templates" / "technology.monitoring-rules.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("existing", encoding="utf-8")

    blocked = build_research_portfolio_preview(portfolio_path, workspace_root=workspace)
    overwrite = build_research_portfolio_preview(portfolio_path, workspace_root=workspace, overwrite=True)

    assert blocked["can_materialize"] is False
    blocked_targets = blocked["targets"]
    assert isinstance(blocked_targets, list)
    assert blocked_targets[0]["action"] == "blocked_existing_files"
    assert blocked_targets[0]["existing_files"] == [str(conflict)]
    assert overwrite["can_materialize"] is True
    overwrite_targets = overwrite["targets"]
    assert isinstance(overwrite_targets, list)
    assert overwrite_targets[0]["action"] == "overwrite"


@pytest.mark.unit
def test_materialize_research_portfolio_records_partial_failure_and_can_overwrite(tmp_path: Path) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portfolio_type": "research_monitoring_portfolio",
                "targets": [
                    {"ticker": "AAPL", "template": "technology"},
                    {"ticker": "MSFT", "template": "technology"},
                ],
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    conflict = workspace / "AAPL" / "assets" / "research_templates" / "common-plus-technology.md"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("conflict", encoding="utf-8")

    partial = materialize_research_portfolio(portfolio_path, workspace_root=workspace)

    assert partial["ok"] is False
    assert partial["success_count"] == 1
    assert partial["failure_count"] == 1
    results = partial["results"]
    assert isinstance(results, list)
    statuses = {result["ticker"]: result["status"] for result in results}
    assert statuses == {"AAPL": "failed", "MSFT": "success"}
    assert not (workspace / "AAPL" / "assets" / "research_templates" / "technology.source-map.json").exists()
    partial_workbook_status = partial["workbook_status"]
    assert isinstance(partial_workbook_status, dict)
    partial_workbooks = partial_workbook_status["workbooks"]
    assert isinstance(partial_workbooks, list)
    assert [item["research_target"]["ticker"] for item in partial_workbooks] == ["MSFT"]

    recovered = materialize_research_portfolio(portfolio_path, workspace_root=workspace, overwrite=True)
    assert recovered["ok"] is True
    assert recovered["success_count"] == 2
    recovered_workbook_status = recovered["workbook_status"]
    assert isinstance(recovered_workbook_status, dict)
    recovered_summary = recovered_workbook_status["summary"]
    assert isinstance(recovered_summary, dict)
    assert recovered_summary["workbook_count"] == 2


@pytest.mark.unit
def test_materialize_research_portfolio_records_runtime_error_without_aborting_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dayu.cli.commands import research_template as research_template_module

    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portfolio_type": "research_monitoring_portfolio",
                "targets": [
                    {"ticker": "AAPL", "template": "technology"},
                    {"ticker": "MSFT", "template": "technology"},
                ],
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    real_materialize = research_template_module.materialize_research_workspace

    def _fake_materialize(name: str, **kwargs: object) -> dict[str, object]:
        # Simulate a degenerate per-target failure (e.g. rollback-also-failed)
        # that raises outside the old (OSError, ValueError) catch tuple.
        if str(kwargs.get("ticker")) == "AAPL":
            raise RuntimeError("materialization failed; rollback also failed")
        return real_materialize(name, **kwargs)

    monkeypatch.setattr(research_template_module, "materialize_research_workspace", _fake_materialize)

    report = materialize_research_portfolio(portfolio_path, workspace_root=workspace)

    assert report["ok"] is False
    assert report["success_count"] == 1
    assert report["failure_count"] == 1
    assert Path(str(report["report_file"])).exists()
    results = report["results"]
    assert isinstance(results, list)
    by_ticker = {result["ticker"]: result for result in results}
    assert by_ticker["AAPL"]["status"] == "failed"
    assert "RuntimeError" in str(by_ticker["AAPL"]["error"])
    assert by_ticker["MSFT"]["status"] == "success"


@pytest.mark.unit
def test_inspect_research_template_bundle_detects_deleted_artifact(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("consumer", workspace_root=tmp_path)
    Path(str(materialized["source_map_file"])).unlink()

    inspected = inspect_research_template_bundle(Path(str(materialized["bundle_file"])))

    validation = inspected["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is False
    errors = validation["errors"]
    assert isinstance(errors, list)
    assert any("artifacts.source_map does not exist" in str(error) for error in errors)


@pytest.mark.unit
def test_discover_research_template_bundles_keeps_malformed_descriptors(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("consumer", workspace_root=tmp_path)
    bundle_dir = Path(str(materialized["bundle_file"])).parent
    malformed_path = bundle_dir / "broken.bundle.json"
    malformed_path.write_text("{", encoding="utf-8")

    bundles = discover_research_template_bundles(tmp_path)

    assert len(bundles) == 2
    by_name = {Path(str(bundle["descriptor_file"])).name: bundle for bundle in bundles}
    assert by_name["consumer.bundle.json"]["template"] == "consumer"
    malformed_validation = by_name["broken.bundle.json"]["validation"]
    assert isinstance(malformed_validation, dict)
    assert malformed_validation["ok"] is False


@pytest.mark.unit
def test_build_monitoring_execution_plan_blocks_unbound_sources(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("consumer", workspace_root=tmp_path)

    plan = build_monitoring_execution_plan(Path(str(materialized["bundle_file"])))

    assert plan["plan_type"] == "research_monitoring_execution_plan"
    assert plan["execution_mode"] == "dry_run"
    assert plan["automated_execution_allowed"] is False
    readiness = plan["readiness"]
    assert isinstance(readiness, dict)
    assert readiness["status"] == "blocked_unbound_sources"
    assert readiness["blocked_task_count"] == readiness["task_count"]
    tasks = plan["tasks"]
    assert isinstance(tasks, list)
    assert tasks
    assert all(task["status"] == "blocked_unbound_sources" for task in tasks)


def _approve_all_internal_source_bindings(
    materialized: dict[str, object],
    *,
    template: str,
    tmp_path: Path,
) -> None:
    """Bind every internal dayu_fins_tool source through the sanctioned approval flow.

    Hand-flipping ``binding_status`` to ``bound`` in the source-map is exactly the
    unauthorized tamper that bundle validation now rejects (a real binding also
    writes a ``binding_approval`` provenance block). Tests that need a genuinely
    bound, healthy workspace must go through ``write_monitoring_source_binding_approval``.
    """

    source_map_path = Path(str(materialized["source_map_file"]))
    source_map = json.loads(source_map_path.read_text(encoding="utf-8"))
    bindings = []
    for source in source_map["data_sources"]:
        if source.get("provider_type") != "dayu_fins_tool":
            continue
        candidate_tools = source.get("candidate_tools") or []
        candidate_fields = source.get("candidate_fields") or []
        bindings.append(
            {
                "source": source["source"],
                "selected_tool": candidate_tools[0],
                "selected_fields": list(candidate_fields[:2]) or list(candidate_fields),
            }
        )
    approval_path = tmp_path / f"{template}.approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "approval_type": "research_monitoring_source_binding",
                "template": template,
                "approved_by": "research-owner",
                "approval_reference": "review-2026-approval",
                "bindings": bindings,
            }
        ),
        encoding="utf-8",
    )
    write_monitoring_source_binding_approval(source_map_path, approval_path)


@pytest.mark.unit
def test_build_monitoring_execution_plan_marks_bound_sources_ready_for_review(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("financial", workspace_root=tmp_path)
    _approve_all_internal_source_bindings(materialized, template="financial", tmp_path=tmp_path)

    plan = build_monitoring_execution_plan(Path(str(materialized["bundle_file"])))

    readiness = plan["readiness"]
    assert isinstance(readiness, dict)
    assert readiness["status"] == "ready_for_review"
    assert readiness["blocked_task_count"] == 0
    assert plan["automated_execution_allowed"] is False


@pytest.mark.unit
def test_write_monitoring_execution_plan_defaults_next_to_bundle(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("technology", workspace_root=tmp_path)
    bundle_path = Path(str(materialized["bundle_file"]))

    plan_path = write_monitoring_execution_plan(bundle_path)

    assert plan_path == bundle_path.parent / "technology.monitoring-plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["template"] == "technology"
    assert payload["execution_mode"] == "dry_run"


@pytest.mark.unit
def test_build_monitoring_execution_plan_rejects_unhealthy_bundle(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("cyclical", workspace_root=tmp_path)
    Path(str(materialized["rules_file"])).unlink()

    with pytest.raises(ValueError, match="unhealthy bundle"):
        build_monitoring_execution_plan(Path(str(materialized["bundle_file"])))


@pytest.mark.unit
def test_validate_monitoring_execution_plan_accepts_blocked_dry_run_plan(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("consumer", workspace_root=tmp_path)
    plan = build_monitoring_execution_plan(Path(str(materialized["bundle_file"])))

    validation = validate_monitoring_execution_plan(plan)

    assert validation["ok"] is True
    assert validation["blocked_task_count"] == validation["task_count"]
    fingerprints = plan["input_fingerprints"]
    assert isinstance(fingerprints, dict)
    assert len(str(fingerprints["monitoring_rules"])) == 64


@pytest.mark.unit
def test_validate_monitoring_execution_plan_rejects_forged_ready_task(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("financial", workspace_root=tmp_path)
    plan_path = write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    # Forge ready_for_review with a bound source the source-map does not actually
    # bind (market_data is an unbindable external placeholder). Inputs are left
    # authentic so the fingerprint check passes.
    for task in plan["tasks"]:
        task["status"] = "ready_for_review"
        task["bound_data_sources"] = ["market_data"]
        task["blocking_reasons"] = []
    plan["readiness"]["status"] = "ready_for_review"
    plan["readiness"]["blocked_task_count"] = 0
    plan["readiness"]["ready_task_count"] = len(plan["tasks"])
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    inspected = inspect_monitoring_execution_plan(plan_path)

    validation = inspected["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is False
    errors = validation["errors"]
    assert isinstance(errors, list)
    assert any("not bound in the source-map" in str(error) for error in errors)


@pytest.mark.unit
def test_inspect_monitoring_execution_plan_detects_stale_input(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("technology", workspace_root=tmp_path)
    plan_path = write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    rules_path = Path(str(materialized["rules_file"]))
    rules_path.write_text(rules_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    inspected = inspect_monitoring_execution_plan(plan_path)

    validation = inspected["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is False
    errors = validation["errors"]
    assert isinstance(errors, list)
    assert any("monitoring_rules changed after plan generation" in str(error) for error in errors)


@pytest.mark.unit
def test_discover_monitoring_execution_plans_keeps_malformed_plans(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("consumer", workspace_root=tmp_path)
    plan_path = write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    malformed_path = plan_path.parent / "broken.monitoring-plan.json"
    malformed_path.write_text("{", encoding="utf-8")

    plans = discover_monitoring_execution_plans(tmp_path)

    assert len(plans) == 2
    by_name = {Path(str(plan["monitoring_plan_file"])).name: plan for plan in plans}
    assert by_name["consumer.monitoring-plan.json"]["template"] == "consumer"
    malformed_validation = by_name["broken.monitoring-plan.json"]["validation"]
    assert isinstance(malformed_validation, dict)
    assert malformed_validation["ok"] is False


@pytest.mark.unit
def test_build_monitoring_status_snapshot_handles_empty_workspace(tmp_path: Path) -> None:
    snapshot = build_monitoring_status_snapshot(tmp_path)

    assert snapshot["overall_status"] == "no_plans"
    summary = snapshot["summary"]
    assert isinstance(summary, dict)
    assert summary["plan_count"] == 0


@pytest.mark.unit
def test_build_monitoring_status_snapshot_prioritizes_unhealthy_plans(tmp_path: Path) -> None:
    consumer = materialize_research_template_bundle("consumer", workspace_root=tmp_path)
    write_monitoring_execution_plan(Path(str(consumer["bundle_file"])))
    technology = materialize_research_template_bundle("technology", workspace_root=tmp_path)
    write_monitoring_execution_plan(Path(str(technology["bundle_file"])))
    rules_path = Path(str(technology["rules_file"]))
    rules_path.write_text(rules_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    snapshot = build_monitoring_status_snapshot(tmp_path)

    assert snapshot["overall_status"] == "unhealthy"
    summary = snapshot["summary"]
    assert isinstance(summary, dict)
    assert summary["plan_count"] == 2
    assert summary["valid_plan_count"] == 1
    assert summary["invalid_plan_count"] == 1
    assert summary["blocked_plan_count"] == 1


@pytest.mark.unit
def test_build_monitoring_status_snapshot_reports_ready_for_review(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle("financial", workspace_root=tmp_path)
    _approve_all_internal_source_bindings(materialized, template="financial", tmp_path=tmp_path)
    write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))

    snapshot = build_monitoring_status_snapshot(tmp_path)

    assert snapshot["overall_status"] == "ready_for_review"
    summary = snapshot["summary"]
    assert isinstance(summary, dict)
    assert summary["ready_for_review_plan_count"] == 1
    assert summary["blocked_task_count"] == 0


@pytest.mark.unit
def test_write_monitoring_status_snapshot_defaults_to_workspace_assets(tmp_path: Path) -> None:
    status_path = write_monitoring_status_snapshot(tmp_path)

    assert status_path == (tmp_path / "assets" / "research_templates" / "monitoring-status.json").resolve()
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["snapshot_type"] == "research_monitoring_status"


@pytest.mark.unit
def test_build_monitoring_scheduler_manifest_handles_empty_workspace(tmp_path: Path) -> None:
    manifest = build_monitoring_scheduler_manifest(tmp_path)

    assert manifest["manifest_type"] == "research_monitoring_scheduler"
    assert manifest["timezone"] == "UTC"
    summary = manifest["summary"]
    assert isinstance(summary, dict)
    assert summary["job_count"] == 0
    assert summary["enabled_job_count"] == 0


@pytest.mark.unit
def test_build_monitoring_scheduler_manifest_exports_disabled_blocked_job(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle(
        "consumer",
        workspace_root=tmp_path,
        ticker="600519",
        company="Kweichow Moutai",
    )
    plan_path = write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))

    manifest = build_monitoring_scheduler_manifest(tmp_path, timezone="Asia/Shanghai")

    jobs = manifest["jobs"]
    assert isinstance(jobs, list)
    assert len(jobs) == 1
    job = jobs[0]
    assert job["job_id"] == "600519-consumer-monitoring"
    assert job["enabled"] is False
    assert job["eligible_for_manual_activation"] is False
    assert job["state"] == "blocked_unbound_sources"
    assert job["trigger"] == {
        "type": "cadence",
        "cadence": "quarterly",
        "timezone": "Asia/Shanghai",
        "binding_status": "unbound",
    }
    assert job["action"]["argv"][-1] == str(plan_path)
    assert len(job["monitoring_plan_fingerprint"]) == 64


@pytest.mark.unit
def test_build_monitoring_scheduler_manifest_marks_ready_job_as_manual_candidate(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle(
        "financial",
        workspace_root=tmp_path,
        ticker="0005.HK",
    )
    _approve_all_internal_source_bindings(materialized, template="financial", tmp_path=tmp_path)
    write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))

    manifest = build_monitoring_scheduler_manifest(tmp_path)

    jobs = manifest["jobs"]
    assert isinstance(jobs, list)
    assert jobs[0]["state"] == "ready_for_review"
    assert jobs[0]["eligible_for_manual_activation"] is True
    assert jobs[0]["enabled"] is False
    summary = manifest["summary"]
    assert isinstance(summary, dict)
    assert summary["manual_activation_candidate_count"] == 1
    assert summary["enabled_job_count"] == 0


@pytest.mark.unit
def test_build_monitoring_scheduler_manifest_includes_invalid_recursive_plan(tmp_path: Path) -> None:
    apple = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path / "AAPL",
        ticker="AAPL",
    )
    write_monitoring_execution_plan(Path(str(apple["bundle_file"])))
    microsoft = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path / "MSFT",
        ticker="MSFT",
    )
    write_monitoring_execution_plan(Path(str(microsoft["bundle_file"])))
    rules_path = Path(str(microsoft["rules_file"]))
    rules_path.write_text(rules_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    manifest = build_monitoring_scheduler_manifest(tmp_path, recursive=True)

    summary = manifest["summary"]
    assert isinstance(summary, dict)
    assert summary["job_count"] == 2
    assert summary["blocked_job_count"] == 1
    assert summary["invalid_job_count"] == 1
    jobs = manifest["jobs"]
    assert isinstance(jobs, list)
    assert all(job["enabled"] is False for job in jobs)


@pytest.mark.unit
def test_write_monitoring_scheduler_manifest_defaults_to_workspace_assets(tmp_path: Path) -> None:
    schedule_path = write_monitoring_scheduler_manifest(tmp_path)

    assert schedule_path == (tmp_path / "assets" / "research_templates" / "monitoring-scheduler.json").resolve()
    payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    assert payload["activation_policy"]["scheduler_binding_status"] == "unbound"


@pytest.mark.unit
def test_validate_monitoring_scheduler_manifest_accepts_safe_blocked_jobs(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle(
        "consumer",
        workspace_root=tmp_path,
        ticker="600519",
    )
    write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    manifest = build_monitoring_scheduler_manifest(tmp_path)

    validation = validate_monitoring_scheduler_manifest(manifest)

    assert validation["ok"] is True
    assert validation["blocked_job_count"] == 1
    assert validation["ready_for_review_job_count"] == 0


@pytest.mark.unit
def test_inspect_monitoring_scheduler_manifest_detects_stale_plan(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path,
        ticker="AAPL",
    )
    plan_path = write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    schedule_path = write_monitoring_scheduler_manifest(tmp_path)
    plan_path.write_text(plan_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    inspected = inspect_monitoring_scheduler_manifest(schedule_path)

    validation = inspected["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is False
    errors = validation["errors"]
    assert isinstance(errors, list)
    assert any("changed after scheduler export" in str(error) for error in errors)


@pytest.mark.unit
def test_validate_monitoring_scheduler_manifest_rejects_enabled_or_rewritten_job(tmp_path: Path) -> None:
    materialized = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path,
        ticker="AAPL",
    )
    write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    manifest = build_monitoring_scheduler_manifest(tmp_path)
    jobs = manifest["jobs"]
    assert isinstance(jobs, list)
    jobs[0]["enabled"] = True
    jobs[0]["action"]["argv"] = ["dangerous-command"]

    validation = validate_monitoring_scheduler_manifest(manifest)

    assert validation["ok"] is False
    errors = validation["errors"]
    assert isinstance(errors, list)
    assert any("enabled must be false" in str(error) for error in errors)
    assert any("safe validation command" in str(error) for error in errors)


@pytest.mark.unit
def test_inspect_monitoring_scheduler_manifest_handles_malformed_json(tmp_path: Path) -> None:
    schedule_path = tmp_path / "monitoring-scheduler.json"
    schedule_path.write_text("{", encoding="utf-8")

    inspected = inspect_monitoring_scheduler_manifest(schedule_path)

    validation = inspected["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is False


@pytest.mark.unit
def test_materialize_research_template_bundle_writes_all_artifacts(tmp_path: Path) -> None:
    payload = materialize_research_template_bundle("consumer", workspace_root=tmp_path)

    assert payload["template"] == "consumer"
    validation = payload["validation"]
    assert isinstance(validation, dict)
    assert validation["ok"] is True
    for key in (
        "template_file",
        "workbook_file",
        "rules_file",
        "source_map_file",
        "manifest_file",
        "guide_file",
        "bundle_file",
    ):
        assert Path(str(payload[key])).exists()
    bundle_validation = payload["bundle_validation"]
    assert isinstance(bundle_validation, dict)
    assert bundle_validation["ok"] is True
    assert str(payload["template_file"]).endswith("common-plus-consumer.md")
    guide = Path(str(payload["guide_file"])).read_text(encoding="utf-8")
    assert "common-plus-consumer.md" in guide
    assert "consumer.research-workbook.json" in guide


@pytest.mark.unit
def test_materialize_research_template_bundle_copies_common_template(tmp_path: Path) -> None:
    payload = materialize_research_template_bundle("common", workspace_root=tmp_path)

    assert str(payload["template_file"]).endswith("common.md")
    assert Path(str(payload["workbook_file"])).exists()
    assert Path(str(payload["rules_file"])).exists()
    assert Path(str(payload["guide_file"])).exists()
    assert Path(str(payload["bundle_file"])).exists()


@pytest.mark.unit
def test_recommend_research_templates_uses_company_facets() -> None:
    profile = CompanyFacetProfile(
        primary_facets=["消费品牌", "零售渠道/连锁"],
        cross_cutting_facets=["高营销费用驱动"],
    )

    recommendations = recommend_research_templates(profile, limit=1)

    assert recommendations[0].name == "consumer"
    assert recommendations[0].score == 7
    assert "消费品牌" in recommendations[0].matched_facets


@pytest.mark.unit
def test_recommend_research_templates_falls_back_to_common() -> None:
    recommendations = recommend_research_templates(CompanyFacetProfile(primary_facets=["未知业务"]), limit=1)

    assert recommendations[0].name == "common"
    assert recommendations[0].score == 0


@pytest.mark.unit
def test_run_list_command_can_emit_json(capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(research_template_action="list", json=True)

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(item["name"] == "consumer" for item in payload)


@pytest.mark.unit
def test_run_recommend_command_can_read_manifest_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        "\ufeff"
        + json.dumps(
            {
                "company_facets": {
                    "business_model_tags": ["银行"],
                    "constraint_tags": ["利率敏感"],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        research_template_action="recommend",
        manifest=str(manifest_path),
        business_model_tags=[],
        constraint_tags=[],
        limit=1,
        json=True,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["name"] == "financial"
    assert payload[0]["score"] == 4


@pytest.mark.unit
def test_run_compose_command_can_emit_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        research_template_action="compose",
        name="technology",
        base=str(tmp_path),
        output=None,
        overwrite=False,
        json=True,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["template_file"].endswith("common-plus-technology.md")


@pytest.mark.unit
def test_run_monitoring_rules_command_can_write_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        research_template_action="monitoring-rules",
        name="financial",
        base=str(tmp_path),
        output=None,
        write=True,
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rules_file"].endswith("financial.monitoring-rules.json")


@pytest.mark.unit
def test_run_research_workbook_command_previews_then_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(
        research_template_action="research-workbook",
        name="cyclical",
        ticker="601919",
        company="中远海控",
        base=str(tmp_path),
        output=None,
        write=False,
        overwrite=False,
    )

    preview_result = run_research_template_command(args)

    assert preview_result == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["template"] == "cyclical"
    assert preview["research_target"]["ticker"] == "601919"
    assert not (tmp_path / "assets" / "research_templates" / "cyclical.research-workbook.json").exists()

    args.write = True
    write_result = run_research_template_command(args)

    assert write_result == 0
    written = json.loads(capsys.readouterr().out)
    workbook_path = Path(written["workbook_file"])
    assert workbook_path.exists()
    assert json.loads(workbook_path.read_text(encoding="utf-8"))["research_target"]["company"] == "中远海控"


@pytest.mark.unit
def test_run_validate_research_workbook_command_returns_nonzero_for_invalid_status(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook_path = write_research_workbook_payload("consumer", workspace_root=tmp_path)
    args = argparse.Namespace(
        research_template_action="validate-research-workbook",
        workbook=str(workbook_path),
    )

    valid_result = run_research_template_command(args)

    assert valid_result == 0
    valid = json.loads(capsys.readouterr().out)
    assert valid["validation"]["ok"] is True

    payload = json.loads(workbook_path.read_text(encoding="utf-8"))
    payload["sections"][0]["items"][0]["status"] = "invented"
    workbook_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    invalid_result = run_research_template_command(args)

    assert invalid_result == 1
    invalid = json.loads(capsys.readouterr().out)
    assert invalid["validation"]["ok"] is False
    assert any("status is invalid" in error for error in invalid["validation"]["errors"])


@pytest.mark.unit
def test_run_update_research_workbook_command_previews_then_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook_path = write_research_workbook_payload("consumer", workspace_root=tmp_path)
    original = workbook_path.read_bytes()
    workbook = json.loads(original)
    item_id = workbook["sections"][0]["items"][0]["item_id"]
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "source": "annual_report",
                "reference": "2025 annual report p.42",
                "finding": "同店销售同比增长 8%。",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        research_template_action="update-research-workbook",
        workbook=str(workbook_path),
        item_id=item_id,
        status="answered",
        response="需求增长由同店销售改善驱动。",
        analyst_notes=None,
        evidence_file=str(evidence_path),
        write=False,
    )

    preview_result = run_research_template_command(args)

    assert preview_result == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["resulting_status"] == "answered"
    assert workbook_path.read_bytes() == original

    args.write = True
    write_result = run_research_template_command(args)

    assert write_result == 0
    written = json.loads(capsys.readouterr().out)
    assert Path(written["backup_file"]).read_bytes() == original
    updated = json.loads(workbook_path.read_text(encoding="utf-8"))
    assert updated["sections"][0]["items"][0]["evidence"][0]["source"] == "annual_report"


@pytest.mark.unit
def test_run_rollback_research_workbook_command_previews_then_restores(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook_path = write_research_workbook_payload("cyclical", workspace_root=tmp_path)
    original = workbook_path.read_bytes()
    payload = json.loads(original)
    item_id = payload["sections"][0]["items"][0]["item_id"]
    update_result = write_research_workbook_update(
        workbook_path,
        item_id=item_id,
        status="in_progress",
    )
    updated = workbook_path.read_bytes()
    args = argparse.Namespace(
        research_template_action="rollback-research-workbook",
        workbook=str(workbook_path),
        backup=str(update_result["backup_file"]),
        write=False,
    )

    preview_result = run_research_template_command(args)

    assert preview_result == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["preview_type"] == "research_workbook_rollback"
    assert workbook_path.read_bytes() == updated

    args.write = True
    write_result = run_research_template_command(args)

    assert write_result == 0
    written = json.loads(capsys.readouterr().out)
    assert Path(written["redo_backup_file"]).read_bytes() == updated
    assert workbook_path.read_bytes() == original


@pytest.mark.unit
def test_run_workbook_status_command_previews_then_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materialize_research_template_bundle(
        "consumer",
        workspace_root=tmp_path / "600519",
        ticker="600519",
    )
    args = argparse.Namespace(
        research_template_action="workbook-status",
        base=str(tmp_path),
        recursive=True,
        output=None,
        write=False,
        overwrite=False,
    )

    preview_result = run_research_template_command(args)

    assert preview_result == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["overall_status"] == "not_started"
    assert preview["summary"]["workbook_count"] == 1

    args.write = True
    write_result = run_research_template_command(args)

    assert write_result == 0
    written = json.loads(capsys.readouterr().out)
    status_path = Path(written["workbook_status_file"])
    assert status_path.exists()
    assert written["status"]["scan_scope"] == "recursive"


@pytest.mark.unit
def test_run_workbook_report_command_previews_then_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook_path = write_research_workbook_payload(
        "cyclical",
        workspace_root=tmp_path,
        ticker="601919",
        company="中远海控",
    )
    args = argparse.Namespace(
        research_template_action="workbook-report",
        workbook=str(workbook_path),
        output=None,
        write=False,
        overwrite=False,
    )

    preview_result = run_research_template_command(args)

    assert preview_result == 0
    preview = capsys.readouterr().out
    assert "# Research Workbook Progress: 中远海控" in preview
    assert not workbook_path.with_name("cyclical.research-progress.md").exists()

    args.write = True
    write_result = run_research_template_command(args)

    assert write_result == 0
    written = json.loads(capsys.readouterr().out)
    report_path = Path(written["workbook_report_file"])
    assert report_path.exists()
    assert "## Open Research Gaps" in report_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_run_validate_workbook_report_command_returns_nonzero_when_stale(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workbook_path = write_research_workbook_payload("technology", workspace_root=tmp_path)
    report_path = write_research_workbook_report(workbook_path)
    args = argparse.Namespace(
        research_template_action="validate-workbook-report",
        report=str(report_path),
        workbook=str(workbook_path),
    )

    valid_result = run_research_template_command(args)

    assert valid_result == 0
    valid = json.loads(capsys.readouterr().out)
    assert valid["validation"]["ok"] is True

    payload = json.loads(workbook_path.read_text(encoding="utf-8"))
    item_id = payload["sections"][0]["items"][0]["item_id"]
    write_research_workbook_update(workbook_path, item_id=item_id, status="in_progress")
    stale_result = run_research_template_command(args)

    assert stale_result == 1
    stale = json.loads(capsys.readouterr().out)
    assert stale["stale"] is True
    assert stale["validation"]["ok"] is False


@pytest.mark.unit
def test_run_workbook_report_status_command_previews_then_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materialize_research_template_bundle("consumer", workspace_root=tmp_path)
    args = argparse.Namespace(
        research_template_action="workbook-report-status",
        base=str(tmp_path),
        recursive=False,
        output=None,
        write=False,
        overwrite=False,
    )

    preview_result = run_research_template_command(args)

    assert preview_result == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["overall_status"] == "current"
    assert preview["summary"]["report_count"] == 1

    args.write = True
    write_result = run_research_template_command(args)

    assert write_result == 0
    written = json.loads(capsys.readouterr().out)
    status_path = Path(written["workbook_report_status_file"])
    assert status_path.exists()
    assert written["status"]["overall_status"] == "current"


@pytest.mark.unit
def test_run_source_map_command_can_write_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        research_template_action="source-map",
        name="cyclical",
        base=str(tmp_path),
        output=None,
        write=True,
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_map_file"].endswith("cyclical.source-map.json")


@pytest.mark.unit
def test_run_source_bindings_previews_then_writes_with_backup(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_map_path = write_monitoring_source_map_payload("financial", workspace_root=tmp_path)
    original = source_map_path.read_bytes()
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "approval_type": "research_monitoring_source_binding",
                "template": "financial",
                "approved_by": "research-owner",
                "approval_reference": "review-2026-005",
                "bindings": [
                    {
                        "source": "regulatory_filings",
                        "selected_tool": "search_document",
                        "selected_fields": ["regulatory_disclosure"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        research_template_action="source-bindings",
        source_map=str(source_map_path),
        approval=str(approval_path),
        write=False,
    )

    preview_result = run_research_template_command(args)

    assert preview_result == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["changed_sources"] == ["regulatory_filings"]
    assert source_map_path.read_bytes() == original

    args.write = True
    write_result = run_research_template_command(args)

    assert write_result == 0
    written = json.loads(capsys.readouterr().out)
    assert Path(written["backup_file"]).read_bytes() == original
    assert source_map_path.read_bytes() != original


@pytest.mark.unit
def test_run_rollback_source_bindings_previews_then_restores(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_map_path = write_monitoring_source_map_payload("financial", workspace_root=tmp_path)
    original = source_map_path.read_bytes()
    approval_path = tmp_path / "approval.json"
    approval_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "approval_type": "research_monitoring_source_binding",
                "template": "financial",
                "approved_by": "research-owner",
                "approval_reference": "review-2026-008",
                "bindings": [
                    {
                        "source": "regulatory_filings",
                        "selected_tool": "search_document",
                        "selected_fields": ["regulatory_disclosure"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    binding_result = write_monitoring_source_binding_approval(source_map_path, approval_path)
    backup_path = Path(str(binding_result["backup_file"]))
    bound = source_map_path.read_bytes()
    args = argparse.Namespace(
        research_template_action="rollback-source-bindings",
        source_map=str(source_map_path),
        backup=str(backup_path),
        write=False,
    )

    preview_result = run_research_template_command(args)

    assert preview_result == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["changed_sources"] == ["regulatory_filings"]
    assert source_map_path.read_bytes() == bound

    args.write = True
    write_result = run_research_template_command(args)

    assert write_result == 0
    written = json.loads(capsys.readouterr().out)
    assert Path(written["rollback_backup_file"]).read_bytes() == bound
    assert source_map_path.read_bytes() == original


@pytest.mark.unit
def test_run_source_binding_history_returns_nonzero_for_invalid_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_map_path = write_monitoring_source_map_payload("financial", workspace_root=tmp_path)
    args = argparse.Namespace(
        research_template_action="source-binding-history",
        source_map=str(source_map_path),
    )

    healthy_result = run_research_template_command(args)

    assert healthy_result == 0
    healthy = json.loads(capsys.readouterr().out)
    assert healthy["summary"]["snapshot_count"] == 0

    forged_snapshot = source_map_path.with_name(f"{source_map_path.stem}.before-rollback.000000000000.json")
    forged_snapshot.write_bytes(source_map_path.read_bytes())
    invalid_result = run_research_template_command(args)

    assert invalid_result == 1
    invalid = json.loads(capsys.readouterr().out)
    assert invalid["summary"]["invalid_snapshot_count"] == 1


@pytest.mark.unit
def test_run_validate_source_map_command_outputs_validation_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rules_path = write_monitoring_rules_payload("consumer", workspace_root=tmp_path)
    source_map_path = write_monitoring_source_map_payload("consumer", workspace_root=tmp_path)
    args = argparse.Namespace(
        research_template_action="validate-source-map",
        rules=str(rules_path),
        source_map=str(source_map_path),
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


@pytest.mark.unit
def test_run_package_manifest_command_can_write_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        research_template_action="package-manifest",
        base=str(tmp_path),
        output=None,
        write=True,
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_file"].endswith("research-template.manifest.json")


@pytest.mark.unit
def test_run_materialize_command_outputs_artifact_paths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        research_template_action="materialize",
        name="technology",
        manifest=None,
        base=str(tmp_path),
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["template"] == "technology"
    assert payload["selection"]["selection_mode"] == "explicit"
    assert payload["selection"]["selected_template"] == "technology"
    assert payload["validation"]["ok"] is True
    assert payload["template_file"].endswith("common-plus-technology.md")
    assert Path(payload["monitoring_status_file"]).is_file()
    assert Path(payload["workbook_status_file"]).is_file()
    assert Path(payload["report_status_file"]).is_file()
    assert payload["monitoring_status"]["overall_status"] == "blocked"
    assert payload["workbook_status"]["overall_status"] == "not_started"
    assert payload["report_status"]["overall_status"] == "current"


@pytest.mark.unit
def test_run_refresh_workspace_command_previews_then_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materialized = materialize_research_workspace("technology", workspace_root=tmp_path)
    workbook_path = Path(str(materialized["workbook_file"]))
    workbook = json.loads(workbook_path.read_text(encoding="utf-8"))
    workbook["sections"][0]["items"][0]["status"] = "in_progress"
    workbook_path.write_text(json.dumps(workbook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        research_template_action="refresh-workspace",
        bundle=str(materialized["bundle_file"]),
        write=False,
    )

    assert run_research_template_command(args) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["can_refresh"] is True

    args.write = True
    assert run_research_template_command(args) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["applied"] is True
    assert applied["report_status"]["overall_status"] == "current"


@pytest.mark.unit
def test_run_materialize_command_can_select_template_from_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest_path = tmp_path / "write-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": {"ticker": "0005.hk", "company": "HSBC Holdings"},
                "company_facets": {
                    "business_model_tags": ["银行"],
                    "constraint_tags": ["利率敏感"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8-sig",
    )
    output_root = tmp_path / "workspace"
    args = argparse.Namespace(
        research_template_action="materialize",
        name=None,
        manifest=str(manifest_path),
        base=str(output_root),
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["template"] == "financial"
    assert payload["research_target"] == {"ticker": "0005.HK", "company": "HSBC Holdings"}
    assert payload["selection"]["selection_mode"] == "manifest_recommendation"
    assert payload["selection"]["selected_template"] == "financial"
    assert payload["selection"]["recommendation"]["matched_facets"] == ["银行", "利率敏感"]
    assert payload["template_file"].endswith("common-plus-financial.md")
    assert Path(payload["template_file"]).exists()
    assert Path(payload["workbook_file"]).exists()
    assert Path(payload["progress_report_file"]).exists()
    assert Path(payload["rules_file"]).exists()
    assert Path(payload["source_map_file"]).exists()
    assert Path(payload["guide_file"]).exists()
    assert Path(payload["bundle_file"]).exists()
    assert Path(payload["monitoring_plan_file"]).exists()
    assert payload["bundle_validation"]["ok"] is True
    assert payload["bundle_validation"]["artifact_count"] == 7
    assert payload["bundle_validation"]["workbook_report_validation"]["validation"]["ok"] is True
    assert payload["validation"]["ok"] is True
    assert payload["monitoring_plan_validation"]["ok"] is True


@pytest.mark.unit
def test_run_materialize_command_prefers_confirmed_manifest_provenance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": {
                    "ticker": "AAPL",
                    "company": "Apple Inc.",
                    "research_template_requested_name": "auto",
                    "research_template_resolved_name": "technology",
                    "research_template_selection_mode": "auto",
                },
                "company_facets": {
                    "business_model_tags": ["银行"],
                    "constraint_tags": ["利率敏感"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        research_template_action="materialize",
        name=None,
        manifest=str(manifest_path),
        base=str(tmp_path / "workspace"),
        overwrite=False,
    )

    assert run_research_template_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["template"] == "technology"
    assert payload["selection"]["selection_mode"] == "manifest_provenance"
    assert payload["selection"]["write_selection"] == {
        "requested_name": "auto",
        "resolved_name": "technology",
        "selection_mode": "auto",
    }
    bundle_path = Path(payload["bundle_file"])
    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    source_binding = bundle_payload["source_write_manifest"]
    assert source_binding["path"] == str(manifest_path.resolve())
    assert source_binding["selected_template"] == "technology"
    assert source_binding["file_fingerprint"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert len(source_binding["semantic_fingerprint"]) == 64

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["audit_note"] = "write progress changed"
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")
    progress_only = inspect_research_template_bundle(bundle_path)
    assert progress_only["validation"]["ok"] is True
    assert "source_write_manifest file changed without selection drift" in progress_only["validation"]["warnings"]

    manifest_payload["company_facets"]["constraint_tags"].append("高资本开支")
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")
    drifted = inspect_research_template_bundle(bundle_path)
    assert drifted["validation"]["ok"] is False
    assert "source_write_manifest semantic fingerprint is stale" in drifted["validation"]["errors"]


@pytest.mark.unit
def test_materialize_research_bundle_from_completed_write_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "draft" / "AAPL" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "config": {
                    "ticker": "AAPL",
                    "company": "Apple Inc.",
                    "research_template_requested_name": "auto",
                    "research_template_resolved_name": "technology",
                    "research_template_selection_mode": "auto",
                },
                "company_facets": {
                    "primary_facets": ["semiconductor design"],
                    "cross_cutting_facets": ["high R&D intensity"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "research"

    payload = materialize_research_bundle_from_write_manifest(
        manifest_path,
        workspace_root=workspace,
    )

    assert payload["template"] == "technology"
    assert payload["research_target"] == {"ticker": "AAPL", "company": "Apple Inc."}
    assert payload["selection"]["selection_mode"] == "manifest_provenance"
    assert payload["bundle_validation"]["ok"] is True
    assert Path(str(payload["bundle_file"])).is_file()
    assert Path(str(payload["workbook_file"])).is_file()
    assert payload["source_write_manifest"]["path"] == str(manifest_path.resolve())


@pytest.mark.unit
def test_materialize_rolls_back_new_artifacts_after_mid_write_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fail_guide(*_args: object, **_kwargs: object) -> Path:
        raise OSError("injected guide failure")

    monkeypatch.setattr(
        "dayu.cli.commands.research_template.write_research_template_usage_guide",
        _fail_guide,
    )

    with pytest.raises(OSError, match="injected guide failure"):
        materialize_research_template_bundle("technology", workspace_root=tmp_path)

    artifact_dir = tmp_path / "assets" / "research_templates"
    assert list(artifact_dir.glob("*")) == []


@pytest.mark.unit
def test_materialize_restores_existing_artifacts_after_overwrite_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path,
        ticker="AAPL",
        company="Apple Inc.",
    )
    artifact_dir = tmp_path / "assets" / "research_templates"
    original_files = {path: path.read_bytes() for path in artifact_dir.iterdir() if path.is_file()}

    def _fail_guide(*_args: object, **_kwargs: object) -> Path:
        raise OSError("injected guide failure")

    monkeypatch.setattr(
        "dayu.cli.commands.research_template.write_research_template_usage_guide",
        _fail_guide,
    )

    with pytest.raises(OSError, match="injected guide failure"):
        materialize_research_template_bundle(
            "technology",
            workspace_root=tmp_path,
            ticker="MSFT",
            company="Microsoft Corp.",
            overwrite=True,
        )

    restored_files = {path: path.read_bytes() for path in artifact_dir.iterdir() if path.is_file()}
    assert restored_files == original_files


@pytest.mark.unit
def test_workspace_materialize_rolls_back_bundle_and_plan_after_late_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dayu.cli.commands import research_template as research_template_module

    original_write_guide = research_template_module.write_research_template_usage_guide
    call_count = 0

    def _fail_second_guide(*args: object, **kwargs: object) -> Path:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("injected final guide failure")
        return original_write_guide(*args, **kwargs)

    monkeypatch.setattr(
        research_template_module,
        "write_research_template_usage_guide",
        _fail_second_guide,
    )

    with pytest.raises(OSError, match="injected final guide failure"):
        materialize_research_workspace("technology", workspace_root=tmp_path)

    artifact_dir = tmp_path / "assets" / "research_templates"
    assert list(artifact_dir.glob("*")) == []


@pytest.mark.unit
def test_refresh_workspace_previews_then_refreshes_all_derived_artifacts(tmp_path: Path) -> None:
    materialized = materialize_research_workspace(
        "technology",
        workspace_root=tmp_path,
        ticker="AAPL",
        company="Apple Inc.",
    )
    workbook_path = Path(str(materialized["workbook_file"]))
    report_path = Path(str(materialized["progress_report_file"]))
    plan_path = Path(str(materialized["monitoring_plan_file"]))
    workbook = json.loads(workbook_path.read_text(encoding="utf-8"))
    workbook["sections"][0]["items"][0]["status"] = "in_progress"
    workbook_path.write_text(json.dumps(workbook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    old_report = report_path.read_bytes()
    old_plan = plan_path.read_bytes()

    preview = build_research_workspace_refresh_preview(Path(str(materialized["bundle_file"])))

    assert preview["can_refresh"] is True
    assert preview["outputs"]["research_progress_report"]["action"] == "refresh"
    assert report_path.read_bytes() == old_report
    assert plan_path.read_bytes() == old_plan

    refreshed = write_research_workspace_refresh(Path(str(materialized["bundle_file"])))

    assert refreshed["applied"] is True
    assert refreshed["bundle_validation"]["ok"] is True
    assert refreshed["monitoring_plan_validation"]["ok"] is True
    assert refreshed["monitoring_status"]["overall_status"] == "blocked"
    assert refreshed["workbook_status"]["overall_status"] == "in_progress"
    assert refreshed["report_status"]["overall_status"] == "current"
    assert report_path.read_bytes() != old_report


@pytest.mark.unit
def test_refresh_workspace_writes_status_inside_snapshot_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dayu.cli.commands import research_template as research_template_module

    materialized = materialize_research_workspace("consumer", workspace_root=tmp_path)
    bundle_path = Path(str(materialized["bundle_file"]))
    bundle_dir = bundle_path.parent

    captured: dict[str, Path | None] = {}

    def _spy(name: str, real):
        def _wrapper(_workspace_root: Path, *, output_path: Path | None = None, **kwargs: object) -> Path:
            captured[name] = output_path
            return real(_workspace_root, output_path=output_path, **kwargs)

        return _wrapper

    monkeypatch.setattr(
        research_template_module,
        "write_monitoring_status_snapshot",
        _spy("monitoring", research_template_module.write_monitoring_status_snapshot),
    )
    monkeypatch.setattr(
        research_template_module,
        "write_research_workbook_status_snapshot",
        _spy("workbook", research_template_module.write_research_workbook_status_snapshot),
    )
    monkeypatch.setattr(
        research_template_module,
        "write_research_workbook_report_status_snapshot",
        _spy("report", research_template_module.write_research_workbook_report_status_snapshot),
    )

    write_research_workspace_refresh(bundle_path)

    # Every status snapshot must be written to an explicit path inside the bundle
    # directory (the snapshot rollback boundary), not re-derived from a hardcoded
    # parent.parent.parent that can diverge for non-canonical bundle depths.
    assert set(captured) == {"monitoring", "workbook", "report"}
    for name, output_path in captured.items():
        assert output_path is not None, name
        assert output_path.resolve().parent == bundle_dir.resolve(), name


@pytest.mark.unit
def test_refresh_workspace_rejects_invalid_workbook(tmp_path: Path) -> None:
    materialized = materialize_research_workspace("consumer", workspace_root=tmp_path)
    workbook_path = Path(str(materialized["workbook_file"]))
    workbook_path.write_text("{}", encoding="utf-8")

    preview = build_research_workspace_refresh_preview(Path(str(materialized["bundle_file"])))

    assert preview["can_refresh"] is False
    assert any("workbook is invalid" in blocker for blocker in preview["blockers"])
    with pytest.raises(ValueError, match="cannot be refreshed"):
        write_research_workspace_refresh(Path(str(materialized["bundle_file"])))


@pytest.mark.unit
def test_refresh_workspace_restores_derived_files_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    materialized = materialize_research_workspace("financial", workspace_root=tmp_path)
    workbook_path = Path(str(materialized["workbook_file"]))
    workbook = json.loads(workbook_path.read_text(encoding="utf-8"))
    workbook["sections"][0]["items"][0]["status"] = "in_progress"
    workbook_path.write_text(json.dumps(workbook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifact_dir = Path(str(materialized["bundle_file"])).parent
    derived_files = (
        artifact_dir / "financial.research-progress.md",
        artifact_dir / "financial.monitoring-plan.json",
        artifact_dir / "monitoring-status.json",
        artifact_dir / "research-workbook-status.json",
        artifact_dir / "research-workbook-report-status.json",
        artifact_dir / "financial.research-guide.md",
    )
    original_bytes = {path: path.read_bytes() for path in derived_files}

    def _fail_status(*_args: object, **_kwargs: object) -> Path:
        raise OSError("injected status refresh failure")

    monkeypatch.setattr(
        "dayu.cli.commands.research_template.write_monitoring_status_snapshot",
        _fail_status,
    )

    with pytest.raises(OSError, match="injected status refresh failure"):
        write_research_workspace_refresh(Path(str(materialized["bundle_file"])))

    assert {path: path.read_bytes() for path in derived_files} == original_bytes


@pytest.mark.unit
def test_rebind_bundle_refreshes_only_descriptor_and_preserves_backup(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_payload = {
        "config": {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "research_template_requested_name": "auto",
            "research_template_resolved_name": "technology",
            "research_template_selection_mode": "auto",
        },
        "company_facets": {"primary_facets": ["半导体设计"]},
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")
    materialized = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path / "workspace",
        ticker="AAPL",
        company="Apple Inc.",
        write_manifest_path=manifest_path,
    )
    bundle_path = Path(str(materialized["bundle_file"]))
    workbook_path = Path(str(materialized["workbook_file"]))
    original_bundle_bytes = bundle_path.read_bytes()
    original_workbook_bytes = workbook_path.read_bytes()
    manifest_payload["audit_note"] = "progress changed"
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")

    preview = build_research_template_bundle_rebind_preview(bundle_path)

    assert preview["changed"] is True
    assert bundle_path.read_bytes() == original_bundle_bytes
    applied = write_research_template_bundle_rebind(bundle_path)
    assert applied["applied"] is True
    backup_path = Path(str(applied["backup_file"]))
    assert backup_path.read_bytes() == original_bundle_bytes
    assert workbook_path.read_bytes() == original_workbook_bytes
    assert inspect_research_template_bundle(bundle_path)["validation"]["ok"] is True
    second = write_research_template_bundle_rebind(bundle_path)
    assert second["applied"] is False
    assert second["backup_file"] is None
    rebound_bundle_bytes = bundle_path.read_bytes()

    rollback_preview = build_research_template_bundle_rebind_rollback_preview(bundle_path, backup_path)
    assert rollback_preview["changed"] is True
    assert bundle_path.read_bytes() == rebound_bundle_bytes
    rollback = write_research_template_bundle_rebind_rollback(bundle_path, backup_path)
    assert rollback["applied"] is True
    assert bundle_path.read_bytes() == original_bundle_bytes
    redo_backup_path = Path(str(rollback["redo_backup_file"]))
    assert redo_backup_path.read_bytes() == rebound_bundle_bytes
    redo = write_research_template_bundle_rebind_rollback(bundle_path, redo_backup_path)
    assert redo["applied"] is True
    assert bundle_path.read_bytes() == rebound_bundle_bytes
    assert workbook_path.read_bytes() == original_workbook_bytes


@pytest.mark.unit
def test_bundle_rebind_rollback_rejects_foreign_backup(tmp_path: Path) -> None:
    first = materialize_research_template_bundle("technology", workspace_root=tmp_path / "first")
    second = materialize_research_template_bundle("financial", workspace_root=tmp_path / "second")
    bundle_path = Path(str(first["bundle_file"]))
    foreign_bytes = Path(str(second["bundle_file"])).read_bytes()
    foreign_fingerprint = hashlib.sha256(foreign_bytes).hexdigest()
    foreign_path = bundle_path.with_name(f"{bundle_path.stem}.before-rebind.{foreign_fingerprint[:12]}.json")
    foreign_path.write_bytes(foreign_bytes)

    with pytest.raises(ValueError, match="template does not match"):
        build_research_template_bundle_rebind_rollback_preview(bundle_path, foreign_path)


@pytest.mark.unit
def test_rebind_bundle_rejects_source_template_reroute(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_payload = {
        "config": {
            "ticker": "AAPL",
            "company": "Apple Inc.",
            "research_template_requested_name": "auto",
            "research_template_resolved_name": "technology",
            "research_template_selection_mode": "auto",
        },
        "company_facets": {"primary_facets": ["半导体设计"]},
    }
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")
    materialized = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path / "workspace",
        ticker="AAPL",
        company="Apple Inc.",
        write_manifest_path=manifest_path,
    )
    manifest_payload["config"]["research_template_resolved_name"] = "financial"
    manifest_payload["company_facets"] = {"primary_facets": ["银行"]}
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="selected template does not match"):
        build_research_template_bundle_rebind_preview(Path(str(materialized["bundle_file"])))


@pytest.mark.unit
def test_run_materialize_command_rejects_incomplete_manifest_provenance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": {"research_template_resolved_name": "technology"},
                "company_facets": {"business_model_tags": ["银行"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        research_template_action="materialize",
        name=None,
        manifest=str(manifest_path),
        base=str(tmp_path / "workspace"),
        overwrite=False,
    )

    assert run_research_template_command(args) == 1
    assert "incomplete research-template provenance" in capsys.readouterr().err


@pytest.mark.unit
def test_run_materialize_command_explicit_target_overrides_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "write-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "config": {"ticker": "OLD", "company": "Old Company"},
                "company_facets": {"business_model_tags": ["閾惰"], "constraint_tags": []},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        research_template_action="materialize",
        name="financial",
        manifest=str(manifest_path),
        ticker="new.hk",
        company="New Company",
        base=str(tmp_path / "workspace"),
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection"]["selection_mode"] == "explicit"
    assert payload["research_target"] == {"ticker": "NEW.HK", "company": "New Company"}


@pytest.mark.unit
def test_run_materialize_command_requires_name_or_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    args = argparse.Namespace(
        research_template_action="materialize",
        name=None,
        manifest=None,
        base="./workspace",
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 1
    assert "materialize requires a template name or --manifest" in capsys.readouterr().err


@pytest.mark.unit
def test_run_list_bundles_command_can_emit_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    materialize_research_template_bundle("technology", workspace_root=tmp_path)
    args = argparse.Namespace(research_template_action="list-bundles", base=str(tmp_path), json=True)

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["template"] == "technology"
    assert payload[0]["validation"]["ok"] is True


@pytest.mark.unit
def test_run_validate_bundle_command_returns_nonzero_for_broken_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materialized = materialize_research_template_bundle("financial", workspace_root=tmp_path)
    Path(str(materialized["guide_file"])).unlink()
    args = argparse.Namespace(
        research_template_action="validate-bundle",
        bundle=str(materialized["bundle_file"]),
    )

    result = run_research_template_command(args)

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["template"] == "financial"
    assert payload["validation"]["ok"] is False


@pytest.mark.unit
def test_run_monitoring_plan_command_can_write_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    materialized = materialize_research_template_bundle("consumer", workspace_root=tmp_path)
    args = argparse.Namespace(
        research_template_action="monitoring-plan",
        bundle=str(materialized["bundle_file"]),
        output=None,
        write=True,
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["monitoring_plan_file"]).exists()
    assert payload["plan"]["readiness"]["status"] == "blocked_unbound_sources"


@pytest.mark.unit
def test_run_validate_monitoring_plan_returns_nonzero_for_stale_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materialized = materialize_research_template_bundle("financial", workspace_root=tmp_path)
    plan_path = write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    source_map_path = Path(str(materialized["source_map_file"]))
    source_map_path.write_text(source_map_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    args = argparse.Namespace(research_template_action="validate-monitoring-plan", plan=str(plan_path))

    result = run_research_template_command(args)

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["validation"]["ok"] is False


@pytest.mark.unit
def test_run_list_monitoring_plans_command_can_emit_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materialized = materialize_research_template_bundle("consumer", workspace_root=tmp_path)
    write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    args = argparse.Namespace(research_template_action="list-monitoring-plans", base=str(tmp_path), json=True)

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["template"] == "consumer"
    assert payload[0]["validation"]["ok"] is True


@pytest.mark.unit
def test_run_monitoring_status_command_can_write_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = argparse.Namespace(
        research_template_action="monitoring-status",
        base=str(tmp_path),
        output=None,
        write=True,
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["monitoring_status_file"]).exists()
    assert payload["status"]["overall_status"] == "no_plans"


@pytest.mark.unit
def test_run_monitoring_status_command_can_recursively_aggregate_targets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materialized = materialize_research_template_bundle(
        "consumer",
        workspace_root=tmp_path / "600519",
        ticker="600519",
        company="Kweichow Moutai",
    )
    write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    args = argparse.Namespace(
        research_template_action="monitoring-status",
        base=str(tmp_path),
        recursive=True,
        output=None,
        write=False,
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scan_scope"] == "recursive"
    assert payload["summary"]["target_count"] == 1
    assert payload["targets"][0]["ticker"] == "600519"


@pytest.mark.unit
def test_run_materialize_portfolio_returns_nonzero_for_partial_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portfolio_type": "research_monitoring_portfolio",
                "targets": [{"ticker": "AAPL", "template": "technology"}],
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    conflict = workspace / "AAPL" / "assets" / "research_templates" / "common-plus-technology.md"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("conflict", encoding="utf-8")
    args = argparse.Namespace(
        research_template_action="materialize-portfolio",
        portfolio=str(portfolio_path),
        base=str(workspace),
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["failure_count"] == 1


@pytest.mark.unit
def test_run_preview_portfolio_returns_nonzero_for_conflicts_without_writes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    portfolio_path = tmp_path / "portfolio.json"
    portfolio_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "portfolio_type": "research_monitoring_portfolio",
                "targets": [{"ticker": "AAPL", "template": "technology"}],
            }
        ),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    conflict = workspace / "AAPL" / "assets" / "research_templates" / "technology.bundle.json"
    conflict.parent.mkdir(parents=True)
    conflict.write_text("existing", encoding="utf-8")
    args = argparse.Namespace(
        research_template_action="preview-portfolio",
        portfolio=str(portfolio_path),
        base=str(workspace),
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["can_materialize"] is False
    assert payload["summary"]["blocked_count"] == 1
    assert not (workspace / "research-portfolio.materialization.json").exists()


@pytest.mark.unit
def test_run_scheduler_manifest_command_can_write_recursive_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materialized = materialize_research_template_bundle(
        "technology",
        workspace_root=tmp_path / "AAPL",
        ticker="AAPL",
    )
    write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    args = argparse.Namespace(
        research_template_action="scheduler-manifest",
        base=str(tmp_path),
        recursive=True,
        timezone="Asia/Shanghai",
        output=None,
        write=True,
        overwrite=False,
    )

    result = run_research_template_command(args)

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["scheduler_manifest_file"]).exists()
    assert payload["manifest"]["scan_scope"] == "recursive"
    assert payload["manifest"]["timezone"] == "Asia/Shanghai"
    assert payload["manifest"]["summary"]["enabled_job_count"] == 0


@pytest.mark.unit
def test_run_validate_scheduler_manifest_returns_nonzero_for_stale_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    materialized = materialize_research_template_bundle(
        "consumer",
        workspace_root=tmp_path,
        ticker="600519",
    )
    plan_path = write_monitoring_execution_plan(Path(str(materialized["bundle_file"])))
    schedule_path = write_monitoring_scheduler_manifest(tmp_path)
    plan_path.write_text(plan_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    args = argparse.Namespace(
        research_template_action="validate-scheduler-manifest",
        manifest=str(schedule_path),
    )

    result = run_research_template_command(args)

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["validation"]["ok"] is False


@pytest.mark.unit
def test_parse_research_template_copy_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "copy",
            "consumer",
            "--base",
            "workspace",
            "--overwrite",
            "--json",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "copy"
    assert args.name == "consumer"
    assert args.overwrite is True
    assert args.json is True


@pytest.mark.unit
def test_parse_research_template_compose_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "compose",
            "consumer",
            "--base",
            "workspace",
            "--overwrite",
            "--json",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "compose"
    assert args.name == "consumer"
    assert args.overwrite is True
    assert args.json is True


@pytest.mark.unit
def test_parse_research_template_monitoring_rules_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "monitoring-rules",
            "consumer",
            "--base",
            "workspace",
            "--write",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "monitoring-rules"
    assert args.name == "consumer"
    assert args.write is True
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_research_workbook_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "research-workbook",
            "consumer",
            "--ticker",
            "600519",
            "--company",
            "贵州茅台",
            "--base",
            "workspace",
            "--write",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "research-workbook"
    assert args.name == "consumer"
    assert args.ticker == "600519"
    assert args.company == "贵州茅台"
    assert args.write is True
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_validate_research_workbook_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "validate-research-workbook",
            "--workbook",
            "consumer.research-workbook.json",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "validate-research-workbook"
    assert args.workbook == "consumer.research-workbook.json"


@pytest.mark.unit
def test_parse_research_template_update_research_workbook_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "update-research-workbook",
            "--workbook",
            "consumer.research-workbook.json",
            "--item-id",
            "item-abc123",
            "--status",
            "answered",
            "--response",
            "结论",
            "--evidence-file",
            "evidence.json",
            "--write",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "update-research-workbook"
    assert args.item_id == "item-abc123"
    assert args.status == "answered"
    assert args.response == "结论"
    assert args.evidence_file == "evidence.json"
    assert args.write is True


@pytest.mark.unit
def test_parse_research_template_rollback_research_workbook_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "rollback-research-workbook",
            "--workbook",
            "consumer.research-workbook.json",
            "--backup",
            "consumer.research-workbook.before-update.abc123.json",
            "--write",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "rollback-research-workbook"
    assert args.workbook == "consumer.research-workbook.json"
    assert args.backup == "consumer.research-workbook.before-update.abc123.json"
    assert args.write is True


@pytest.mark.unit
def test_parse_research_template_workbook_status_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "workbook-status",
            "--base",
            "workspace",
            "--recursive",
            "--write",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "workbook-status"
    assert args.recursive is True
    assert args.write is True
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_workbook_report_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "workbook-report",
            "--workbook",
            "consumer.research-workbook.json",
            "--write",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "workbook-report"
    assert args.workbook == "consumer.research-workbook.json"
    assert args.write is True
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_validate_workbook_report_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "validate-workbook-report",
            "--report",
            "consumer.research-progress.md",
            "--workbook",
            "consumer.research-workbook.json",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "validate-workbook-report"
    assert args.report == "consumer.research-progress.md"
    assert args.workbook == "consumer.research-workbook.json"


@pytest.mark.unit
def test_parse_research_template_workbook_report_status_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "workbook-report-status",
            "--base",
            "workspace",
            "--recursive",
            "--write",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "workbook-report-status"
    assert args.recursive is True
    assert args.write is True
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_source_map_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "source-map",
            "financial",
            "--base",
            "workspace",
            "--write",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "source-map"
    assert args.name == "financial"
    assert args.write is True
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_source_bindings_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "source-bindings",
            "--source-map",
            "consumer.source-map.json",
            "--approval",
            "approval.json",
            "--write",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "source-bindings"
    assert args.source_map == "consumer.source-map.json"
    assert args.approval == "approval.json"
    assert args.write is True


@pytest.mark.unit
def test_parse_research_template_rollback_source_bindings_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "rollback-source-bindings",
            "--source-map",
            "consumer.source-map.json",
            "--backup",
            "consumer.source-map.before-bindings.abc123.json",
            "--write",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "rollback-source-bindings"
    assert args.source_map == "consumer.source-map.json"
    assert args.backup == "consumer.source-map.before-bindings.abc123.json"
    assert args.write is True


@pytest.mark.unit
def test_parse_research_template_source_binding_history_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "source-binding-history",
            "--source-map",
            "consumer.source-map.json",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "source-binding-history"
    assert args.source_map == "consumer.source-map.json"


@pytest.mark.unit
def test_parse_research_template_validate_source_map_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "validate-source-map",
            "--rules",
            "rules.json",
            "--source-map",
            "source-map.json",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "validate-source-map"
    assert args.rules == "rules.json"
    assert args.source_map == "source-map.json"


@pytest.mark.unit
def test_parse_research_template_package_manifest_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "package-manifest",
            "--base",
            "workspace",
            "--write",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "package-manifest"
    assert args.write is True
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_materialize_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "materialize",
            "consumer",
            "--base",
            "workspace",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "materialize"
    assert args.name == "consumer"
    assert args.manifest is None
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_materialize_manifest_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "materialize",
            "--manifest",
            "write-manifest.json",
            "--base",
            "workspace",
            "--ticker",
            "0700.hk",
            "--company",
            "Tencent Holdings",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "materialize"
    assert args.name is None
    assert args.manifest == "write-manifest.json"
    assert args.ticker == "0700.hk"
    assert args.company == "Tencent Holdings"
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_refresh_workspace_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "cli.py",
            "research-template",
            "refresh-workspace",
            "--bundle",
            "./workspace/AAPL/assets/research_templates/technology.bundle.json",
            "--write",
        ],
    )

    args = parse_arguments()

    assert args.research_template_action == "refresh-workspace"
    assert args.bundle.endswith("technology.bundle.json")
    assert args.write is True


@pytest.mark.unit
def test_parse_research_template_list_bundles_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["dayu-cli", "research-template", "list-bundles", "--base", "workspace", "--recursive", "--json"],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "list-bundles"
    assert args.base == "workspace"
    assert args.recursive is True
    assert args.json is True


@pytest.mark.unit
def test_parse_research_template_validate_bundle_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["dayu-cli", "research-template", "validate-bundle", "--bundle", "consumer.bundle.json"],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "validate-bundle"
    assert args.bundle == "consumer.bundle.json"


@pytest.mark.unit
def test_parse_research_template_rebind_bundle_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "rebind-bundle",
            "--bundle",
            "consumer.bundle.json",
            "--write",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "rebind-bundle"
    assert args.bundle == "consumer.bundle.json"
    assert args.write is True


@pytest.mark.unit
def test_parse_research_template_rollback_bundle_rebind_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "rollback-bundle-rebind",
            "--bundle",
            "consumer.bundle.json",
            "--backup",
            "consumer.bundle.before-rebind.abc123.json",
            "--write",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "rollback-bundle-rebind"
    assert args.bundle == "consumer.bundle.json"
    assert args.backup == "consumer.bundle.before-rebind.abc123.json"
    assert args.write is True


@pytest.mark.unit
def test_parse_research_template_monitoring_plan_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "monitoring-plan",
            "--bundle",
            "consumer.bundle.json",
            "--write",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "monitoring-plan"
    assert args.bundle == "consumer.bundle.json"
    assert args.write is True
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_validate_monitoring_plan_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "validate-monitoring-plan",
            "--plan",
            "consumer.monitoring-plan.json",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "validate-monitoring-plan"
    assert args.plan == "consumer.monitoring-plan.json"


@pytest.mark.unit
def test_parse_research_template_list_monitoring_plans_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "list-monitoring-plans",
            "--base",
            "workspace",
            "--recursive",
            "--json",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "list-monitoring-plans"
    assert args.base == "workspace"
    assert args.recursive is True
    assert args.json is True


@pytest.mark.unit
def test_parse_research_template_monitoring_status_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "monitoring-status",
            "--base",
            "workspace",
            "--recursive",
            "--write",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "monitoring-status"
    assert args.base == "workspace"
    assert args.recursive is True
    assert args.write is True


@pytest.mark.unit
def test_parse_research_template_materialize_portfolio_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "materialize-portfolio",
            "--portfolio",
            "portfolio.json",
            "--base",
            "workspace",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "materialize-portfolio"
    assert args.portfolio == "portfolio.json"
    assert args.base == "workspace"
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_preview_portfolio_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "preview-portfolio",
            "--portfolio",
            "portfolio.json",
            "--base",
            "workspace",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "preview-portfolio"
    assert args.portfolio == "portfolio.json"
    assert args.base == "workspace"
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_scheduler_manifest_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "scheduler-manifest",
            "--base",
            "workspace",
            "--recursive",
            "--timezone",
            "Asia/Shanghai",
            "--write",
            "--overwrite",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "scheduler-manifest"
    assert args.base == "workspace"
    assert args.recursive is True
    assert args.timezone == "Asia/Shanghai"
    assert args.write is True
    assert args.overwrite is True


@pytest.mark.unit
def test_parse_research_template_validate_scheduler_manifest_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "validate-scheduler-manifest",
            "--manifest",
            "monitoring-scheduler.json",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "validate-scheduler-manifest"
    assert args.manifest == "monitoring-scheduler.json"


@pytest.mark.unit
def test_parse_research_template_recommend_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "dayu-cli",
            "research-template",
            "recommend",
            "--business-model-tag",
            "半导体设计",
            "--constraint-tag",
            "高研发驱动",
            "--limit",
            "2",
            "--json",
        ],
    )

    args = parse_arguments()

    assert args.command == "research-template"
    assert args.research_template_action == "recommend"
    assert args.business_model_tags == ["半导体设计"]
    assert args.constraint_tags == ["高研发驱动"]
    assert args.limit == 2
    assert args.json is True


@pytest.mark.unit
def test_main_dispatches_research_template_command() -> None:
    args = argparse.Namespace(command="research-template", research_template_action="list")

    with (
        patch("dayu.cli.main.parse_arguments", return_value=args),
        patch("dayu.cli.main.configure_standard_streams_for_console_output"),
        patch("dayu.cli.commands.research_template.run_research_template_command", return_value=0) as run_command,
    ):
        result = main()

    assert result == 0
    run_command.assert_called_once_with(args)
