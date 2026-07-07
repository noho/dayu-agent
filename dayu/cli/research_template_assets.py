"""Shared research-template asset resolution for CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dayu.cli.research_template_routing import (
    load_company_facets_from_manifest,
    recommend_research_template_name,
)
from dayu.startup.config_file_resolver import resolve_package_assets_path

_TEMPLATE_DIR_NAME = "research_templates"
_TEMPLATE_SUFFIX = ".md"
_COMMON_TEMPLATE_NAME = "common"


@dataclass(frozen=True)
class ResearchTemplateSelection:
    """Resolved provenance for one named or automatic research template."""

    requested_name: str
    resolved_name: str
    selection_mode: str
    path: Path
    manifest_path: Path | None = None


def normalize_research_template_name(name: str) -> str:
    """Normalize and validate a packaged research-template name."""

    normalized = str(name or "").strip().lower()
    if not normalized:
        raise ValueError("research template name is required")
    if any(char in normalized for char in ("\\", "/", ":", "..")):
        raise ValueError(f"invalid research template name: {name!r}")
    return normalized


def resolve_packaged_research_template(name: str) -> Path:
    """Resolve one packaged research template by stable name."""

    normalized = normalize_research_template_name(name)
    template_dir = resolve_package_assets_path() / _TEMPLATE_DIR_NAME
    template_path = template_dir / f"{normalized}{_TEMPLATE_SUFFIX}"
    if not template_path.is_file():
        available = ", ".join(
            sorted(
                path.stem
                for path in template_dir.glob(f"*{_TEMPLATE_SUFFIX}")
                if path.stem.lower() != "readme"
            )
        )
        raise FileNotFoundError(f"unknown research template {name!r}; available: {available or '(none)'}")
    return template_path.resolve()


def build_composed_research_template_text(name: str) -> str:
    """Build a write-compatible template with common and industry research lenses."""

    normalized = normalize_research_template_name(name)
    common_text = resolve_packaged_research_template(_COMMON_TEMPLATE_NAME).read_text(encoding="utf-8").rstrip()
    if normalized == _COMMON_TEMPLATE_NAME:
        research_text = common_text
    else:
        industry_text = resolve_packaged_research_template(normalized).read_text(encoding="utf-8").rstrip()
        research_text = f"{common_text}\n\n---\n\n{industry_text}"
    base_template_path = resolve_package_assets_path() / "定性分析模板.md"
    if not base_template_path.is_file():
        raise FileNotFoundError(f"default write template not found: {base_template_path}")
    base_text = base_template_path.read_text(encoding="utf-8").rstrip()
    source_heading = "\n## 来源清单"
    source_index = base_text.rfind(source_heading)
    research_lens = _demote_research_template_headings(research_text)
    research_chapter = (
        "\n\n## 深化研究框架\n\n"
        "以下问题用于约束本章研究路径；结论仍须绑定当前公司的可验证证据。\n\n"
        f"{research_lens}\n"
    )
    if source_index < 0:
        return f"{base_text}{research_chapter}\n"
    return f"{base_text[:source_index]}{research_chapter}{base_text[source_index:]}\n"


def resolve_research_template_for_write(
    name: str,
    *,
    workspace_root: Path,
    manifest_path: Path | None = None,
) -> Path:
    """Resolve a named research template into a safe write-pipeline input."""

    return resolve_research_template_selection(
        name,
        workspace_root=workspace_root,
        manifest_path=manifest_path,
    ).path


def resolve_research_template_selection(
    name: str,
    *,
    workspace_root: Path,
    manifest_path: Path | None = None,
) -> ResearchTemplateSelection:
    """Resolve a template path together with auditable routing provenance."""

    requested_name = normalize_research_template_name(name)
    resolved_name = requested_name
    selection_mode = "named"
    source_manifest_path: Path | None = None
    if requested_name == "auto":
        if manifest_path is None or not manifest_path.is_file():
            raise FileNotFoundError(
                f"auto research-template selection requires an existing write manifest: {manifest_path}; "
                "run write --infer first"
            )
        source_manifest_path = manifest_path.resolve()
        resolved_name = recommend_research_template_name(load_company_facets_from_manifest(source_manifest_path))
        selection_mode = "auto"
    composed_text = build_composed_research_template_text(resolved_name)
    target_name = (
        f"{_COMMON_TEMPLATE_NAME}.write{_TEMPLATE_SUFFIX}"
        if resolved_name == _COMMON_TEMPLATE_NAME
        else f"common-plus-{resolved_name}{_TEMPLATE_SUFFIX}"
    )
    target_path = (
        workspace_root / "assets" / _TEMPLATE_DIR_NAME / target_name
    ).resolve()
    if target_path.exists():
        if not target_path.is_file():
            raise ValueError(f"research template output is not a file: {target_path}")
        if target_path.read_text(encoding="utf-8") != composed_text:
            raise ValueError(
                f"existing research template differs from packaged composition: {target_path}; "
                f"review it and run 'research-template compose {resolved_name} --overwrite' to replace it"
            )
        return ResearchTemplateSelection(
            requested_name=requested_name,
            resolved_name=resolved_name,
            selection_mode=selection_mode,
            path=target_path,
            manifest_path=source_manifest_path,
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target_path.open("x", encoding="utf-8", newline="\n") as file_handle:
            file_handle.write(composed_text)
    except FileExistsError:
        if not target_path.is_file() or target_path.read_text(encoding="utf-8") != composed_text:
            raise ValueError(f"research template was concurrently replaced with different content: {target_path}") from None
    return ResearchTemplateSelection(
        requested_name=requested_name,
        resolved_name=resolved_name,
        selection_mode=selection_mode,
        path=target_path,
        manifest_path=source_manifest_path,
    )


def _demote_research_template_headings(content: str) -> str:
    lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("## "):
            lines.append(f"### {line[3:]}")
        elif line.startswith("# "):
            lines.append(f"### {line[2:]}")
        else:
            lines.append(line)
    return "\n".join(lines).strip()
