# Codex Review: Research Template Library

## Review Focus

- Templates are packaged with the project and remain available after installation.
- CLI operations are local-only, deterministic, and safe by default.
- Copy command does not overwrite user workspace files unless `--overwrite` is passed.
- Recommend command maps existing company facet tags to templates without calling an LLM or external data provider.
- Compose command creates a combined common + industry template without changing write pipeline semantics.
- Monitoring-rules command extracts draft manual-review variables from template Markdown without enabling automated alerts prematurely.
- Monitoring-rule drafts include template-level data source candidates and explicit `binding_status=unbound`.
- Source-map command maps candidate data sources to tool/provider placeholders while keeping all bindings unbound.
- Validate-source-map command checks template alignment, missing mapped sources, extra mapped sources, and binding status shape.
- Package-manifest command summarizes all templates with validation status for future UI or scheduler discovery.
- Materialize command writes the usable local bundle for one template in a single command.
- Materialize command can select a template from write manifest `company_facets` while keeping explicit names authoritative.
- Materialize command writes a local research guide that points to generated artifacts and the next `write --template` command.
- Materialize command writes and validates a machine-readable bundle descriptor for UI and scheduler discovery.
- Research-guide write commands quote template paths so workspace paths containing spaces remain usable.
- List-bundles discovers only the standard workspace bundle directory and preserves malformed entries as unhealthy results.
- Validate-bundle rechecks descriptor shape and referenced artifacts, returning nonzero for unhealthy bundles.
- Monitoring-plan requires a healthy bundle and consistent source-map, exposes binding blockers per variable, and never enables automated execution.
- Validate-monitoring-plan checks dry-run invariants, task/readiness counts, unique IDs, referenced files, and SHA-256 input freshness.
- Monitoring-status aggregates only validated task counts and reports deterministic `no_plans`, `unhealthy`, `blocked`, or `ready_for_review` states.
- Materialize resolves research target identity with explicit CLI values taking precedence over write-manifest config, and preserves it through guide, bundle, plan, and status outputs.
- Recursive discovery is opt-in, preserves default workspace isolation, and produces ticker-level portfolio rollups without hiding stale target plans.
- Materialize-portfolio preflights the complete manifest before writes, resolves relative write manifests, isolates ticker workspaces, records partial failures, and supports overwrite recovery.
- Preview-portfolio performs no writes, reports exact artifact conflicts, and shares its target-level gate with materialize-portfolio.
- Scheduler-manifest exports every plan state without enabling jobs, keeps provider execution forbidden, and marks only ready plans as manual activation candidates.
- Validate-scheduler-manifest treats blocked jobs as valid but rejects unsafe activation, command rewriting, stale plans, state drift, and count mismatches.
- Source-bindings requires reviewer metadata, permits only declared Dayu tools/fields, rejects external placeholders, writes immutable backups, and relies on plan fingerprints to force regeneration.
- Rollback-source-bindings validates backup provenance, template and source identity, preserves the pre-rollback state, restores exact bytes, and returns monitoring plans to stale then blocked-unbound state after regeneration.
- Source-binding-history discovers both immutable snapshot classes, reports bound sources and health per file, and fails visibly when snapshot provenance or identity validation fails.
- The copied template can be passed directly to `dayu-cli write --template`.

## Verification Checklist

- `pytest tests/cli/test_research_template_command.py`
- `pyright dayu/cli/commands/research_template.py dayu/cli/arg_parsing.py dayu/cli/main.py tests/cli/test_research_template_command.py`
- `dayu-cli research-template list`
- `dayu-cli research-template show consumer`
- `dayu-cli research-template copy consumer --base <tmp> --json`
- `dayu-cli research-template recommend --business-model-tag 消费品牌 --constraint-tag 高营销费用驱动 --json`
- `dayu-cli research-template compose consumer --base <tmp> --json`
- `dayu-cli research-template monitoring-rules consumer --write --base <tmp>`
- `dayu-cli research-template source-map consumer --write --base <tmp>`
- `dayu-cli research-template validate-source-map --rules <rules.json> --source-map <source-map.json>`
- `dayu-cli research-template package-manifest --write --base <tmp>`
- `dayu-cli research-template materialize consumer --base <tmp>`
- `dayu-cli research-template materialize --manifest <write-manifest.json> --base <tmp>`
- `dayu-cli research-template list-bundles --base <tmp> --json`
- `dayu-cli research-template validate-bundle --bundle <bundle.json>`
- `dayu-cli research-template monitoring-plan --bundle <bundle.json> --write`
- `dayu-cli research-template validate-monitoring-plan --plan <monitoring-plan.json>`
- `dayu-cli research-template list-monitoring-plans --base <tmp> --json`
- `dayu-cli research-template monitoring-status --base <tmp> --write`
- `dayu-cli research-template rollback-source-bindings --source-map <source-map.json> --backup <before-bindings.json> --write`
- `dayu-cli research-template source-binding-history --source-map <source-map.json>`

## Phase 7.24 Review

- Added no-write binding rollback preview with strict same-directory, filename fingerprint, template, source-set, duplicate-source, and binding-status validation.
- Applied rollback first writes an immutable content-addressed `before-rollback` snapshot, then restores the selected `before-bindings` bytes exactly.
- Lifecycle coverage proves a bound plan is ready for review, rollback makes it stale, and regeneration returns it to `blocked_unbound_sources` without enabling provider execution.

## Phase 7.25 Review

- Added deterministic discovery for `before-bindings` and `before-rollback` snapshots beside one source-map.
- History inspection validates actual SHA-256 against filenames plus template, source-set, duplicate-source, and binding-status invariants without modifying files.
- Invalid snapshots remain in machine-readable output and force a nonzero CLI exit instead of disappearing from audit history.

## Phase 7.26 Review

- `rollback-source-bindings` now accepts either a valid `before-bindings` or `before-rollback` content-addressed snapshot without relaxing directory, fingerprint, template, or source-set checks.
- Every restore preserves the current state first, making bind rollback and rollback reversal exact byte-for-byte operations.
- Lifecycle coverage proves plan readiness changes from ready to stale to blocked, then stale to ready again when the bound snapshot is restored.

## Phase 8.1 Review

- Added deterministic extraction of research questions, business analysis, evidence requirements, monitoring variables, falsifiers, and synthesis prompts from all packaged templates.
- Workbook items have stable content-derived IDs, open status, response/evidence/note slots, target identity, and source-template fingerprint while remaining manual-review only.
- CLI preview is no-write; explicit writes use the standard workspace asset path and reject replacement without `--overwrite`.

## Phase 8.2 Review

- Materialization now writes a target-aware research workbook before guide and bundle creation, returning its path as a first-class artifact.
- Bundle descriptors require the workbook, advertise `track_research_evidence=true`, and guide files link analysts to the generated workbook.
- Portfolio previews include workbook paths in create/overwrite/conflict classification so analyst state is not silently replaced.

## Phase 8.3 Review

- Added independent workbook validation for schema, target shape, current packaged-template fingerprint, stable identity, item state, response, notes, and evidence records.
- Answered evidence-required items fail without both a written response and at least one source/reference/finding record.
- Live status/category counts and completion state are derived from items; stale cached summary fields remain visible as warnings rather than blocking otherwise valid analyst edits.

## Phase 8.4 Review

- Bundle validation now loads and deep-validates the referenced research workbook rather than checking file existence only.
- Workbook template and research target must match the bundle descriptor, preventing cross-company or cross-template workbook substitution.
- Valid partial progress keeps the bundle healthy with warnings; missing responses/evidence, invalid structure, or identity mismatch makes the bundle unhealthy.

## Phase 8.5 Review

- Added item-level workbook updates for status, response, analyst notes, and structured evidence append using stable item IDs.
- Preview applies full validation without writing; accepted updates refresh cached summary/completion fields and pass a second final validation.
- Applied updates preserve exact prior bytes in immutable `before-update.<sha256>.json` backups before replacing the workbook.

## Phase 8.6 Review

- Added workbook rollback preview with same-directory, content-addressed filename, template, target, and full backup-health validation.
- Applied rollback preserves the current workbook under its own content fingerprint before restoring exact backup bytes.
- The same command can restore the generated redo backup, making item updates byte-for-byte reversible in both directions.

## Phase 8.7 Review

- Added standard-path workbook discovery with opt-in recursive portfolio scanning and per-file validation diagnostics.
- Status snapshots derive deterministic `no_workbooks`, `unhealthy`, `not_started`, `in_progress`, or `complete` states and aggregate only validated item counts.
- Persisted `research-workbook-status.json` snapshots are excluded from workbook discovery and protected from replacement without `--overwrite`.

## Phase 8.8 Review

- Portfolio materialization now writes recursive workbook status beside monitoring status and embeds both path and payload in the durable report.
- Successful target records expose their workbook path, and portfolio preview announces the workbook status derived output without writing it.
- Partial-failure reports aggregate only workbooks that were actually materialized, while preserving failed target diagnostics separately.

## Phase 9.1 Review

- Added four valuation/expectation-gap and four catalyst/timeline prompts to the common template and every industry template.
- Industry prompts encode the relevant valuation mechanics, sensitivity variables, downside risks, observable milestones, and thesis invalidation points instead of generic multiple comparisons.
- Workbook extraction recognizes `valuation` and `catalyst` as first-class validated categories; existing sections remain unchanged so their stable item IDs are preserved.

## Phase 9.2 Review

- Added four management/governance/capital-allocation prompts to the common template and every industry template without renaming existing sections.
- Prompts focus on observable guidance accuracy, incentives, reinvestment outcomes, channel or stakeholder conflicts, risk culture, dilution, and minority-shareholder exposure.
- Workbook extraction and validation treat these prompts as first-class `management_governance` items, preserving the same evidence and completion controls as other research categories.

## Phase 9.3 Review

- Added four portfolio-decision/risk-budget prompts to the common template and every industry template without changing prior section IDs.
- Prompts convert evidence strength, downside/tail risk, liquidity, factor correlation, and opportunity cost into initial/max sizing plus explicit add, reduce, and exit rules.
- Workbook extraction and validation treat these prompts as `portfolio_decision` items, so sizing claims require the same response and evidence discipline as operating conclusions.

## Phase 10.1 Review

- Added deterministic Markdown rendering for validated workbooks with target identity, source fingerprint, live completion counts, and every section/item.
- Reports preserve written responses, structured evidence references/findings, analyst notes, and a final list of all non-terminal research gaps without generating conclusions.
- CLI preview performs no writes; persisted `{template}.research-progress.md` output rejects replacement without explicit `--overwrite`.

## Phase 10.2 Review

- Added machine-readable report metadata with canonical workbook semantic SHA-256 and independent Markdown body SHA-256.
- `validate-workbook-report` revalidates the source workbook, ignores formatting-only JSON changes, fails stale reports after semantic updates, and detects body tampering separately.
- Regenerating with explicit overwrite refreshes both fingerprints and returns the report to a valid state.

## Phase 10.3 Review

- Added standard-path report discovery with automatic sibling-workbook pairing and opt-in recursive portfolio scans.
- Status aggregation distinguishes current, stale, tampered, and missing-workbook reports while retaining full validation diagnostics per file.
- Persisted `research-workbook-report-status.json` is protected by overwrite controls and cannot be rediscovered as a Markdown report.

## Phase 10.4 Review

- Extracted workbook construction, validation, mutation, rollback, reporting, and status aggregation into `research_workbook.py`.
- Preserved the existing `research_template` imports and CLI routing through explicit re-exports plus a compatibility identity test.
- Verified the boundary with Ruff, Pyright, 412 CLI tests, and a real materialize-to-report status workflow.

## Phase 10.5 Review

- Added `write --research-template <name>` as an explicit, opt-in bridge from the reusable research library into the production write configuration.
- Industry selections generate the exact same write-compatible `common-plus-<industry>.md` content as `research-template compose`: the official write contract plus one deep-research chapter containing the common and industry lenses; common-only selection generates the equivalent `common.write.md` contract.
- Existing identical compositions are reused, while analyst-edited or stale compositions fail closed instead of being silently overwritten; `--template` remains available for deliberate custom files.

## Phase 10.6 Review

- Added `--research-template auto` routing from the existing `draft/<ticker>/manifest.json` company facets without making a model call during configuration.
- Extracted facet rules and manifest parsing into a shared routing module so recommend, materialize, and write auto-selection use one deterministic source of truth.
- Same-ticker manifest signature changes now preserve company facets while still purging stale chapter/source artifacts, preventing the selected research template from triggering a redundant inference pass.

## Phase 10.7 Review

- `write --research-template auto` now handles an empty workspace in one invocation: default-contract facet inference first, deterministic research-template routing second, normal writing third.
- Both stages reuse the already prepared host, Fins runtime, and write service; no duplicate dependency assembly or hidden external integration was introduced.
- Bootstrap nonzero/cancelled results stop immediately, and `auto --infer` exits after the bootstrap stage, preserving the established infer-only contract.

## Phase 10.8 Review

- Added a first-class research-template selection result instead of inferring provenance from generated filenames.
- Named and auto requests now persist requested name, resolved template name, and selection mode in the write manifest and runtime configuration summary.
- New fields default to empty strings, so older write manifests deserialize unchanged; round-trip tests verify new provenance survives persistence.

## Phase 10.9 Review

- Single-target and portfolio materialization now share one write-manifest template-selection resolver.
- Confirmed write provenance is authoritative even if current facet routing rules would recommend another template, preventing report and workbook/bundle divergence.
- Legacy manifests still use deterministic facet recommendation; incomplete, unknown, unsupported, or inconsistent provenance fails closed instead of silently rerouting.

## Phase 10.10 Review

- Bundles materialized from a write manifest now retain a source binding rather than only a one-time template decision.
- `validate-bundle` reloads and reroutes the current source manifest, detecting deletion, invalid provenance, and template mismatch before monitoring plans can be regenerated.
- Source target identity is recorded for audit, but explicit `--ticker/--company` overrides remain authoritative for the materialized research target.

## Phase 10.11 Review

- Split source binding into a canonical research-semantic fingerprint and an informational full-file fingerprint.
- Template provenance, source target, or normalized company-facet changes invalidate the bundle; unrelated chapter results and audit metadata no longer create false stale failures.
- Full-file-only changes remain visible as warnings, retaining auditability without blocking monitoring-plan consumers.

## Phase 10.12 Review

- Added `rebind-bundle` preview/write flow for accepting reviewed source-manifest changes without regenerating analyst workbooks.
- Applied rebinds preserve exact prior descriptor bytes in `before-rebind.<sha>.json`; workbook, template, rules, source-map, and guide bytes remain untouched.
- Rebind refuses template reroutes, validates the candidate before writing and the persisted descriptor afterward, and returns a no-op when fingerprints are already current.

## Phase 10.13 Review

- Added `rollback-bundle-rebind` preview/write flow with strict same-directory, filename fingerprint, template, target, artifact, and source-path identity checks.
- Rollback restores exact prior bytes and content-addresses the displaced current descriptor as a redo backup, enabling exact backward and forward transitions.
- Restored validation is always returned and may be unhealthy when the source manifest has not itself been reverted; no research artifact bytes are modified.

## Phase 10.14 Review

- Added opt-in `write --materialize-research` orchestration so a successful report can produce its provenance-bound bundle and workbook without a second command.
- Related root/overwrite controls fail closed unless materialization is enabled; infer-only and summary modes are rejected because neither establishes a newly completed write result.
- Nonzero or cancelled writes never materialize. Post-write conflicts or invalid manifests return `2` while retaining the already completed report and logging the partial-success boundary explicitly.

## Phase 10.15 Review

- Wrapped single-target materialization in a byte-snapshot rollback boundary covering the composed write template, workbook, rules, source map, package manifest, guide, and bundle descriptor.
- Any generation exception or unhealthy final bundle restores pre-existing files exactly and removes files created by the failed attempt; overwrite mode no longer leaves mixed old/new research state after ordinary process exceptions.
- Fault-injection tests cover both a fresh workspace failure and a late overwrite failure with byte-for-byte restoration. The scope is deliberately documented as in-process exception safety, not power-loss atomicity.

## Phase 10.16 Review

- Standard materialization now writes `{template}.research-progress.md` immediately after the workbook and exposes its path in the return payload, usage guide, bundle descriptor, and portfolio conflict preview.
- Bundle validation deeply verifies report metadata, workbook semantic freshness, and Markdown body integrity; workbook changes make the bundle unhealthy until the report is explicitly refreshed.
- The report participates in materialization rollback, while descriptors created before this phase remain valid when the optional report artifact is absent.

## Phase 10.17 Review

- Added a complete research-workspace materializer above the existing bundle-only primitive, avoiding hidden plan side effects for low-level callers.
- CLI materialize, `write --materialize-research`, and portfolio materialization now all return a validated dry-run monitoring plan and a guide that links it.
- The outer snapshot spans bundle and plan creation: a late plan/guide/validation failure restores or removes every protected artifact, while all monitoring jobs remain unbound and automation-disabled.

## Phase 10.18 Review

- Complete workspace materialization now writes `monitoring-status.json`, `research-workbook-status.json`, and `research-workbook-report-status.json` after the bundle and plan pass validation.
- Missing or unhealthy initial snapshots fail the transaction; valid initial states remain explicit as blocked monitoring, not-started research, and current report rendering.
- Paths and payloads are returned and linked from the guide. Portfolio reports now include recursive report health alongside existing monitoring and workbook rollups.

## Phase 10.19 Review

- Added `refresh-workspace --bundle` as a no-write preview and `--write` as the explicit mutation gate for all derived workspace artifacts.
- Refresh accepts report staleness/missing derived outputs but refuses invalid workbooks, missing core artifacts, source-manifest semantic drift, or any bundle error that report regeneration cannot solve.
- Applied refresh regenerates the report, validates the bundle, regenerates and validates the dry-run plan, rebuilds all statuses and the guide, and restores every prior derived file byte-for-byte if a late step fails.

## Phase 10.20 Completion Audit

- The original packaged-template, lightweight CLI, local-only, and `write --template` compatibility requirements are implemented without enabling external providers or automated monitoring.
- Later phases close the two historical deferred items: ticker-driven first-run auto routing and direct opt-in write-pipeline composition.
- Final acceptance requires isolated built-wheel asset discovery, real CLI materialize/refresh validation, zero Pyright errors in affected modules, full repository runtime tests, clean diff checks, and no temporary artifacts. Repository-wide Pyright is also probed, but unrelated optional-dependency and baseline findings are reported separately rather than folded into this feature.
- Built-wheel audit found all six packaged Markdown assets; an isolated target installation discovered `common`, `consumer`, `cyclical`, `financial`, and `technology` and loaded template content successfully.
- Final affected-source Ruff and Pyright checks report zero errors. Full repository runtime verification reports 5,397 passed and 83 skipped after installing the declared `web` and `browser` extras into the local test environment.
- Repository-wide Pyright remains nonzero on pre-existing optional UI typings and unrelated test typing debt; the affected research/write source set is clean, and this limitation is recorded rather than presented as a green global static check.

## Opus 4.8 Post-Merge Review (commit 7579e0d)

Independent read-only review of the merged feature by Claude Opus 4.8. No P0.
Findings verified against source, then fixed in three follow-up commits with
regression tests and real-CLI smokes. Fixes are backward compatible: no existing
test required changing, and the full repository suite passes (5407 passed, 83
skipped).

### P1 — `validate-bundle` trusted a tampered source-map (commit 5715f96)

- Bundle health trusted the descriptor's embedded `monitoring_validation.ok` and
  checked `monitoring_rules` / `source_map` / `write_template` / `package_manifest`
  / `usage_guide` for existence only. A hand-flipped `unbound` -> `bound`
  source-map kept reporting the bundle healthy, and the flipped source then
  produced `ready_for_review` tasks.
- Fix: `validate_research_template_bundle_descriptor` now recomputes monitoring
  integrity from the current rules/source-map files and requires a
  `binding_approval` provenance block on any `bound` source (which the sanctioned
  approval flow always writes). The intentionally-mutable source-map is not
  fingerprinted, so the `source-bindings` approval flow and `refresh-workspace`
  stay healthy.

### P2 — `write --research-template auto` hard-failed on null facets (commit 5715f96)

- The infer bootstrap gate keyed off `manifest.json` existence only. A prior
  write that persisted `company_facets: null` (transient infer failure) made the
  next `auto` run skip the bootstrap and then raise `SystemExit(2)` at template
  resolution.
- Fix: the gate now keys off company-facet presence via
  `_manifest_has_usable_company_facets`, so a null-facets manifest re-infers as
  documented.

### P2 — Portfolio batch aborted on degenerate target failures (commit e3d9b5c)

- `materialize_research_portfolio` caught only `(OSError, ValueError)` per target,
  but `materialize_research_workspace` can raise `RuntimeError`
  (rollback-also-failed) or `AssertionError`. Those escaped the loop, aborting the
  whole batch with an uncaught traceback and skipping the report/status snapshots.
- Fix: widen the per-target catch to `Exception` (still excludes
  `KeyboardInterrupt`/`SystemExit`) so a single failure is recorded and the batch
  report is still written. The same widening applies to `write --materialize-research`,
  which now returns the documented exit `2` for those degenerate failures instead
  of a traceback-exit-1.

### P3 (x4) — research_workbook hardening (commit 31dd9d6)

- ID uniqueness: section/item positions are now indexed into the ID hash, so a
  template that repeats a heading or bullet no longer collides and fails the
  freshly-built workbook's own validation.
- BOM tolerance: templates and progress reports are read with `utf-8-sig`
  (`str.strip()` does not remove a BOM, which previously dropped the first section
  or made a report look malformed).
- Counts: `live_summary` counts only well-formed dict items, so a malformed item
  no longer inflates `item_count` / `category_counts`.
- Corrupt-file rollback: workbook rollback authenticates against the backup, not
  the file being recovered, so it runs even when the current workbook is corrupt
  (unreadable JSON / blanked template). Filename<->content fingerprint and
  same-directory checks remain the safety guarantees; the redo backup is skipped
  when no current file exists.

### Verification

- Full repository suite: 5407 passed, 83 skipped (baseline 7579e0d was 5397).
- Affected-source Ruff and Pyright: zero errors.
- Real-CLI / single-process smokes confirmed: tampered bundle rejected while a
  legitimately approved bundle stays healthy; null-facets manifest re-infers;
  portfolio partial failure writes an honest batch report; corrupt-current-file
  workbook rollback restores exact original bytes.

### P2 follow-ups from self-review (commits 399e350, 48d3658)

Skeptical re-review of the fix commits surfaced two more real defects:

- `write_research_workbook_rollback` saved the current bytes as a redo backup
  even when the current workbook was corrupt -- a junk file that could never be
  rolled back to, and whose later use raised a raw parse traceback. Fixed by
  only creating a redo backup when the current file is a valid restorable
  workbook (new `current_restorable` preview flag) and wrapping the backup read
  to raise a clean `ValueError` (399e350).

- `write_research_workspace_refresh` wrote the three status snapshots to
  `bundle.parent.parent.parent/assets/research_templates`, while the snapshot
  rollback boundary covered `bundle.parent`-derived paths. For a bundle at a
  non-canonical depth these diverge, orphaning status files outside the boundary
  on failure. Fixed by writing each status snapshot to the preview-computed
  output path inside the boundary; the scan root is unchanged (48d3658).

Both remaining agent-flagged items are now resolved. The other reported items
(scheduler `enabled=false` invariant, backup directory confinement, fingerprint
drift, byte-snapshot rollback core) were verified correct and required no change.

### P2 — plan validation trusted self-reported task binding (commit 3861ca6)

Correction to an earlier "bounded" assessment: this was a real, reachable gap.
`validate_monitoring_execution_plan` checked each task only against itself
(`status in {ready, blocked}`, `ready => non-empty bound list`) and never
cross-checked `bound_data_sources` against the source-map. A plan file with
authentic input fingerprints but hand-edited `ready_for_review` tasks claiming an
unbindable external placeholder (`market_data`) is bound validated as `ok`.
`build_monitoring_execution_plan` gates on bundle health, but that does not
protect a plan already on disk, which is exactly what `validate-monitoring-plan`
inspects. Fixed by recomputing the genuinely-bound source names from the
fingerprint-matched source-map and rejecting any task that claims bound sources
the source-map does not bind. Verified: a forged plan is rejected while a
legitimately approved-then-regenerated plan still validates.
