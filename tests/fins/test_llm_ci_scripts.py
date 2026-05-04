"""`utils/llm_ci_*` 脚本的轻量回归测试。"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from subprocess import CompletedProcess
import sys
from types import ModuleType
from typing import Sequence

import pytest

from dayu.fins.domain.document_models import CompanyMeta, SourceDocumentUpsertRequest
from dayu.fins.domain.enums import SourceKind
from tests.fins.storage_testkit import build_fs_storage_test_context


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESS_SCRIPT_PATH = REPO_ROOT / "utils" / "llm_ci_process.py"
SCORE_SCRIPT_PATH = REPO_ROOT / "utils" / "llm_ci_score.py"
DOCLING_SCORE_SCRIPT_PATH = REPO_ROOT / "utils" / "llm_docling_ci_score.py"


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    """按路径加载测试目标模块。

    Args:
        name: 模块名。
        path: 脚本路径。

    Returns:
        已加载模块对象。

    Raises:
        ImportError: 模块 spec 不存在时抛出。
    """

    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {path}")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
def test_llm_ci_process_aggregate_document_ids_by_ticker_dedupes_and_preserves_order() -> None:
    """验证文档选择项会按 ticker 聚合并稳定去重。"""

    module = _load_module_from_path("workspace_tmp_llm_ci_process", PROCESS_SCRIPT_PATH)
    entries = [
        module.DocumentSelectorEntry(ticker="TSM", document_id="fil_2"),
        module.DocumentSelectorEntry(ticker="TSM", document_id="fil_1"),
        module.DocumentSelectorEntry(ticker="TSM", document_id="fil_2"),
        module.DocumentSelectorEntry(ticker="BILI", document_id="fil_a"),
    ]

    grouped = module._aggregate_document_ids_by_ticker(entries)

    assert grouped == {
        "BILI": ("fil_a",),
        "TSM": ("fil_2", "fil_1"),
    }


@pytest.mark.unit
def test_llm_ci_process_split_document_ids_for_job_uses_stable_chunks() -> None:
    """验证大批文档会按稳定顺序切成多个子批次。"""

    module = _load_module_from_path("workspace_tmp_llm_ci_process_split", PROCESS_SCRIPT_PATH)

    batches = module._split_document_ids_for_job(
        ("fil_1", "fil_2", "fil_3", "fil_4", "fil_5"),
        max_documents_per_job=2,
    )

    assert batches == (
        ("fil_1", "fil_2"),
        ("fil_3", "fil_4"),
        ("fil_5",),
    )


@pytest.mark.unit
def test_llm_ci_process_build_jobs_prefers_document_mapping() -> None:
    """验证存在文档映射时只按映射构造作业。"""

    module = _load_module_from_path("workspace_tmp_llm_ci_process_jobs", PROCESS_SCRIPT_PATH)

    jobs = module._build_jobs(
        tickers=["AAPL", "TSM"],
        document_ids_by_ticker={"TSM": ("fil_1", "fil_2")},
    )

    assert jobs == [module.ProcessJob(ticker="TSM", document_ids=("fil_1", "fil_2"), batch_index=1)]


@pytest.mark.unit
def test_llm_ci_process_build_jobs_splits_large_document_batches() -> None:
    """验证单个 ticker 的超大文档集合会拆成多个 process 作业。"""

    module = _load_module_from_path("workspace_tmp_llm_ci_process_large_jobs", PROCESS_SCRIPT_PATH)
    jobs = module._build_jobs(
        tickers=["TSM"],
        document_ids_by_ticker={"TSM": ("fil_1", "fil_2", "fil_3")},
        max_documents_per_job=2,
    )

    assert jobs == [
        module.ProcessJob(ticker="TSM", document_ids=("fil_1", "fil_2"), batch_index=1),
        module.ProcessJob(ticker="TSM", document_ids=("fil_3",), batch_index=2),
    ]


@pytest.mark.unit
def test_llm_ci_process_group_jobs_by_ticker_keeps_same_ticker_batches_serialized() -> None:
    """验证同一 ticker 的批次会被分到同一串行组内。"""

    module = _load_module_from_path("workspace_tmp_llm_ci_process_grouped_jobs", PROCESS_SCRIPT_PATH)
    jobs = [
        module.ProcessJob(ticker="NVS", document_ids=("fil_3",), batch_index=2),
        module.ProcessJob(ticker="AAPL", document_ids=("fil_a",), batch_index=1),
        module.ProcessJob(ticker="NVS", document_ids=("fil_1", "fil_2"), batch_index=1),
    ]

    grouped_jobs = module._group_jobs_by_ticker(jobs)

    assert grouped_jobs == (
        (module.ProcessJob(ticker="AAPL", document_ids=("fil_a",), batch_index=1),),
        (
            module.ProcessJob(ticker="NVS", document_ids=("fil_1", "fil_2"), batch_index=1),
            module.ProcessJob(ticker="NVS", document_ids=("fil_3",), batch_index=2),
        ),
    )


@pytest.mark.unit
def test_llm_ci_process_detects_failed_documents_from_cli_summary() -> None:
    """验证脚本会把 CLI 日志中的 failed filings 识别为真实失败。"""

    module = _load_module_from_path("workspace_tmp_llm_ci_process_failed_log", PROCESS_SCRIPT_PATH)
    log_text = (
        "全量处理结果\n"
        "- ticker: NVS\n"
        "失败的 filings:\n"
        "  - fil_0001370368-25-000004 | status=failed | reason=commit_batch failed\n"
    )

    assert module._count_reported_failed_documents(log_text) == 1

    result = module.ProcessRunResult(
        ticker="NVS",
        document_ids=("fil_0001370368-25-000004",),
        command=("python", "-m", "dayu.cli"),
        return_code=0,
        duration_seconds=1.0,
        timed_out=False,
        log_path="/tmp/NVS.log",
        batch_index=1,
        reported_failed_documents=1,
    )

    assert module._is_successful_result(result) is False


def test_llm_ci_score_resolve_forms_normalizes_amendment_suffix() -> None:
    """验证 `SC 13G/A` 会归一化为 `SC 13G`。"""

    module = _load_module_from_path("workspace_tmp_llm_ci_score_forms", SCORE_SCRIPT_PATH)

    forms = module._resolve_forms("SC 13G/A,20-F")

    assert forms == ["SC 13G", "20-F"]


@pytest.mark.unit
def test_llm_ci_score_build_form_summary_uses_probe_for_missing_documents() -> None:
    """验证 form 摘要会把未进 score JSON 的文档映射回探针原因。"""

    module = _load_module_from_path("workspace_tmp_llm_ci_score_summary", SCORE_SCRIPT_PATH)
    universe = [
        module.FilingUniverseDocument(ticker="TSM", document_id="fil_1", form_type="20-F"),
        module.FilingUniverseDocument(ticker="TSM", document_id="fil_2", form_type="20-F"),
    ]
    probe_results = [
        module.ProbeResult(
            ticker="TSM",
            document_id="fil_1",
            form_type="20-F",
            status=module.PROBE_READY,
            detail="可进入 score_sec_ci 评分",
        ),
        module.ProbeResult(
            ticker="TSM",
            document_id="fil_2",
            form_type="20-F",
            status=module.PROBE_MISSING_PROCESSED,
            detail="processed manifest 中不存在该文档",
        ),
    ]
    payload = module.LoadedScorePayload(
        summary=module.ScoreSummaryRecord(
            average_score=97.0,
            p10_score=97.0,
            hard_gate_failures=0,
            document_count=1,
        ),
        documents=(
            module.ScoreDocumentRecord(
                ticker="TSM",
                document_id="fil_1",
                total_score=97.0,
            ),
        ),
    )

    summary = module._build_form_summary(
        form_type="20-F",
        score_payload=payload,
        probe_results=probe_results,
        universe_documents=universe,
        return_code=0,
    )

    assert summary.missing_from_score_count == 1
    assert summary.missing_processed_count == 1
    assert summary.missing_from_score == (
        module.MissingDocumentGap(
            ticker="TSM",
            document_id="fil_2",
            status=module.PROBE_MISSING_PROCESSED,
            detail="processed manifest 中不存在该文档",
        ),
    )


@pytest.mark.unit
def test_llm_ci_score_build_overall_summary_aggregates_document_scores() -> None:
    """验证 overall 摘要按文档分数聚合而不是按 form 平均值二次平均。"""

    module = _load_module_from_path("workspace_tmp_llm_ci_score_overall", SCORE_SCRIPT_PATH)
    form_summaries = [
        module.FormSummary(
            form_type="20-F",
            avg=98.0,
            p10=98.0,
            hard_gate_failures=0,
            document_count=1,
            universe_document_count=2,
            missing_from_score_count=1,
            missing_processed_count=1,
            missing_snapshot_count=0,
            invalid_snapshot_count=0,
            score_return_code=0,
            missing_from_score=tuple(),
        ),
        module.FormSummary(
            form_type="10-K",
            avg=90.0,
            p10=88.0,
            hard_gate_failures=2,
            document_count=2,
            universe_document_count=2,
            missing_from_score_count=0,
            missing_processed_count=0,
            missing_snapshot_count=0,
            invalid_snapshot_count=0,
            score_return_code=1,
            missing_from_score=tuple(),
        ),
    ]
    payloads = [
        module.LoadedScorePayload(
            summary=None,
            documents=(
                module.ScoreDocumentRecord(ticker="TSM", document_id="fil_1", total_score=98.0),
            ),
        ),
        module.LoadedScorePayload(
            summary=None,
            documents=(
                module.ScoreDocumentRecord(ticker="AAPL", document_id="fil_a", total_score=92.0),
                module.ScoreDocumentRecord(ticker="AAPL", document_id="fil_b", total_score=88.0),
            ),
        ),
    ]

    summary = module._build_overall_summary(
        form_summaries=form_summaries,
        form_payloads=payloads,
    )

    assert summary.overall_avg == 92.67
    assert summary.overall_hard_gate_failures == 2
    assert summary.overall_document_count == 3
    assert summary.overall_universe_document_count == 4
    assert summary.overall_missing_from_score_count == 1
    assert summary.forms_included == ("20-F", "10-K")


@pytest.mark.unit
def test_llm_docling_ci_score_build_targets_uses_cn_hk_kind_pairs() -> None:
    """验证 CN/HK Docling score runner 只生成合法 by-kind 组合。"""

    module = _load_module_from_path("workspace_tmp_llm_docling_ci_score_targets", DOCLING_SCORE_SCRIPT_PATH)

    targets = module._build_score_targets(
        source_kinds=("filing", "material"),
        report_kinds=("annual", "semiannual", "quarterly", "material"),
    )

    assert targets == (
        module.ScoreTarget(source_kind="all", report_kind="all", output_stem="score", by_kind=False),
        module.ScoreTarget(source_kind="filing", report_kind="annual", output_stem="score_filing_annual", by_kind=True),
        module.ScoreTarget(
            source_kind="filing",
            report_kind="semiannual",
            output_stem="score_filing_semiannual",
            by_kind=True,
        ),
        module.ScoreTarget(
            source_kind="filing",
            report_kind="quarterly",
            output_stem="score_filing_quarterly",
            by_kind=True,
        ),
        module.ScoreTarget(
            source_kind="material",
            report_kind="material",
            output_stem="score_material_material",
            by_kind=True,
        ),
    )


@pytest.mark.unit
def test_llm_docling_ci_score_command_calls_docling_scorer() -> None:
    """验证 runner 固定调用 score_docling_ci 而不是 SEC scorer。"""

    module = _load_module_from_path("workspace_tmp_llm_docling_ci_score_command", DOCLING_SCORE_SCRIPT_PATH)
    target = module.ScoreTarget(
        source_kind="filing",
        report_kind="annual",
        output_stem="score_filing_annual",
        by_kind=True,
    )

    command = module._build_score_command(
        workspace_root=Path("/tmp/workspace"),
        tickers=["000001", "00700"],
        target=target,
        output_json=Path("/tmp/score.json"),
        output_md=Path("/tmp/score.md"),
    )

    assert "dayu.fins.score_docling_ci" in command
    assert "dayu.fins.score_sec_ci" not in command
    assert "--source-kind" in command
    assert "--report-kind" in command
    assert "--tickers" in command
    assert "000001,00700" in command


@pytest.mark.unit
def test_llm_docling_ci_score_discovers_cn_hk_active_tickers(tmp_path: Path) -> None:
    """验证空 tickers 时 runner 只扫描 CN/HK active source ticker。"""

    module = _load_module_from_path("workspace_tmp_llm_docling_ci_score_discovery", DOCLING_SCORE_SCRIPT_PATH)
    context = build_fs_storage_test_context(tmp_path)
    for ticker, market in (("000001", "CN"), ("00700", "HK"), ("AAPL", "US")):
        context.company_repository.upsert_company_meta(
            CompanyMeta(
                company_id=f"company_{ticker}",
                company_name=f"公司 {ticker}",
                ticker=ticker,
                market=market,
                resolver_version="test",
                updated_at="2026-01-01T00:00:00Z",
            )
        )
    context.source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="000001",
            document_id="fil_cn",
            internal_document_id="fil_cn",
            form_type="FY",
            primary_document="fil_cn.json",
            meta={"ingest_complete": True, "is_deleted": False, "form_type": "FY"},
        ),
        SourceKind.FILING,
    )
    context.source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="00700",
            document_id="mat_hk_deleted",
            internal_document_id="mat_hk_deleted",
            form_type="MATERIAL_OTHER",
            primary_document="mat_hk_deleted.json",
            meta={"ingest_complete": True, "is_deleted": True, "form_type": "MATERIAL_OTHER"},
        ),
        SourceKind.MATERIAL,
    )
    context.source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="AAPL",
            document_id="fil_us",
            internal_document_id="fil_us",
            form_type="10-K",
            primary_document="fil_us.html",
            meta={"ingest_complete": True, "is_deleted": False, "form_type": "10-K"},
        ),
        SourceKind.FILING,
    )

    tickers = module._discover_cn_hk_tickers(tmp_path)

    assert tickers == ["000001"]


@pytest.mark.unit
def test_llm_docling_ci_score_main_empty_tickers_writes_fail_fast_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """验证空 CN/HK ticker 集不会退化为 scorer 全量扫描。"""

    module = _load_module_from_path("workspace_tmp_llm_docling_ci_score_empty_main", DOCLING_SCORE_SCRIPT_PATH)
    subprocess_called = False

    def fake_run(
        command: Sequence[str],
        cwd: Path | None = None,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
    ) -> CompletedProcess[str]:
        """空 ticker 测试中不应调用 subprocess.run。"""

        nonlocal subprocess_called
        subprocess_called = True
        return CompletedProcess(args=(), returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code = module.main(["--base", str(tmp_path), "--tag", "empty"])

    score_dir = tmp_path / "tmp" / "docling_ci_score" / "empty"
    assert exit_code == 1
    assert subprocess_called is False
    assert (score_dir / "score.json").exists()
    assert (score_dir / "score.md").exists()
    assert (score_dir / "score.txt").exists()
    assert (score_dir / "summary.json").exists()
    assert (score_dir / "overall_summary.json").exists()
    overall = json.loads((score_dir / "overall_summary.json").read_text(encoding="utf-8"))
    assert overall["overall_document_count"] == 0
    assert overall["score_return_code"] == 1


@pytest.mark.unit
def test_llm_docling_ci_score_main_writes_outputs_and_preserves_failure_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 main 落盘 score/summary，并在 scorer 非 0 时返回非 0。"""

    module = _load_module_from_path("workspace_tmp_llm_docling_ci_score_e2e", DOCLING_SCORE_SCRIPT_PATH)

    def fake_run(
        command: Sequence[str],
        cwd: Path | None = None,
        capture_output: bool = False,
        text: bool = False,
        check: bool = False,
    ) -> CompletedProcess[str]:
        """伪造 score_docling_ci 子进程输出。"""

        parts = [str(item) for item in command]
        output_json = Path(parts[parts.index("--output-json") + 1])
        output_md = Path(parts[parts.index("--output-md") + 1])
        source_kind = parts[parts.index("--source-kind") + 1]
        report_kind = parts[parts.index("--report-kind") + 1]
        is_material = source_kind == "material" and report_kind == "material"
        score = 80.0 if is_material else 92.0
        return_code = 1 if is_material else 0
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(
                {
                    "summary": {
                        "average_score": score,
                        "p10_score": score,
                        "hard_gate_failures": return_code,
                        "completeness_failure_count": 0,
                        "expected_document_count": 1,
                        "document_count": 1,
                    },
                    "documents": [
                        {
                            "ticker": "000001",
                            "document_id": f"{source_kind}_{report_kind}",
                            "market": "CN",
                            "total_score": score,
                        }
                    ],
                    "completeness_failures": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text("# score\n", encoding="utf-8")
        return CompletedProcess(args=parts, returncode=return_code, stdout="STDOUT\n", stderr="STDERR\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    exit_code = module.main(
        [
            "--base",
            str(tmp_path),
            "--tickers",
            "000001",
            "--tag",
            "e2e",
            "--source-kinds",
            "filing,material",
            "--report-kinds",
            "annual,material",
        ]
    )

    score_dir = tmp_path / "tmp" / "docling_ci_score" / "e2e"
    assert exit_code == 1
    assert (score_dir / "score.json").exists()
    assert (score_dir / "score.md").exists()
    assert "STDOUT" in (score_dir / "score.txt").read_text(encoding="utf-8")
    assert (score_dir / "by_kind" / "score_filing_annual.json").exists()
    assert (score_dir / "by_kind" / "score_material_material.json").exists()
    summary = json.loads((score_dir / "summary.json").read_text(encoding="utf-8"))
    overall = json.loads((score_dir / "overall_summary.json").read_text(encoding="utf-8"))
    assert summary["material_material"]["score_return_code"] == 1
    assert overall["overall_document_count"] == 1
    assert overall["markets_included"] == ["CN"]


@pytest.mark.unit
def test_llm_docling_ci_score_summary_payload_uses_scorer_summary() -> None:
    """验证 summary.json 从 scorer summary 和输出路径聚合。"""

    module = _load_module_from_path("workspace_tmp_llm_docling_ci_score_summary", DOCLING_SCORE_SCRIPT_PATH)
    target = module.ScoreTarget(source_kind="material", report_kind="material", output_stem="score_material_material", by_kind=True)
    result = module.ScoreRunResult(
        target=target,
        return_code=1,
        output_json=Path("/tmp/score_material_material.json"),
        output_md=Path("/tmp/score_material_material.md"),
        output_txt=Path("/tmp/score_material_material.txt"),
        summary=module.ScoreSummaryRecord(
            avg=86.0,
            p10=80.0,
            hard_gate_failures=1,
            completeness_failure_count=2,
            document_count=3,
            expected_document_count=5,
        ),
        markets=("CN", "HK"),
    )

    payload = module._summary_payload((result,))

    assert payload["material_material"]["avg"] == 86.0
    assert payload["material_material"]["score_return_code"] == 1
    assert payload["material_material"]["markets"] == ["CN", "HK"]


@pytest.mark.unit
def test_llm_docling_ci_score_overall_payload_aggregates_documents_and_markets() -> None:
    """验证 overall_summary.json 从 all/all score JSON 聚合 market 与分数。"""

    module = _load_module_from_path("workspace_tmp_llm_docling_ci_score_overall", DOCLING_SCORE_SCRIPT_PATH)
    target = module.ScoreTarget(source_kind="all", report_kind="all", output_stem="score", by_kind=False)
    result = module.ScoreRunResult(
        target=target,
        return_code=0,
        output_json=Path("/tmp/score.json"),
        output_md=Path("/tmp/score.md"),
        output_txt=Path("/tmp/score.txt"),
        summary=module.ScoreSummaryRecord(
            avg=90.0,
            p10=88.0,
            hard_gate_failures=1,
            completeness_failure_count=0,
            document_count=2,
            expected_document_count=2,
        ),
        markets=("CN",),
    )
    score_payload = {
        "documents": [
            {"ticker": "000001", "document_id": "fil_1", "market": "CN", "total_score": 92.0},
            {"ticker": "00700", "document_id": "fil_2", "market": "HK", "total_score": 88.0},
        ]
    }

    payload = module._overall_payload(
        overall_result=result,
        overall_score_payload=score_payload,
        results=(result,),
    )

    assert payload["overall_avg"] == 90.0
    assert payload["overall_p10"] == 88.4
    assert payload["markets_included"] == ["CN", "HK"]
