"""Ground truth 入库脚本测试。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dayu.fins._converters import require_non_empty_text
from dayu.fins.domain.document_models import ProcessedCreateRequest
from dayu.fins.domain.enums import SourceKind
from dayu.fins.ground_truth_baseline import (
    GROUND_TRUTH_REQUIRED_FILES,
    _assert_required_truth_files,
    _load_or_init_manifest,
    _upsert_manifest_sample,
    main,
    parse_args,
    promote_ground_truth_sample,
)
from tests.fins.storage_testkit import build_fs_storage_test_context


def _prepare_source_truth_dir(*, workspace_root: Path, ticker: str, document_id: str) -> Path:
    """构造测试用 source ground truth 目录与文件。

    Args:
        workspace_root: 工作区根目录。
        ticker: 股票代码。
        document_id: 文档 ID。

    Returns:
        已构造好的 source 目录路径。

    Raises:
        OSError: 文件写入失败时抛出。
    """

    source_dir = workspace_root / "portfolio" / ticker / "processed" / document_id
    context = build_fs_storage_test_context(workspace_root)
    context.processed_repository.create_processed(
        ProcessedCreateRequest(
            ticker=ticker,
            document_id=document_id,
            internal_document_id=document_id.replace("fil_", ""),
            source_kind=SourceKind.FILING.value,
            form_type="10-K",
            meta={"form_type": "10-K", "is_deleted": False},
            sections=[],
            tables=[],
            financials=None,
        )
    )
    source_dir.mkdir(parents=True, exist_ok=True)
    for file_name in GROUND_TRUTH_REQUIRED_FILES:
        file_path = source_dir / file_name
        payload: dict[str, Any] = {"file": file_name}
        if file_name == "tool_snapshot_meta.json":
            payload = {
                "schema_version": "tool_snapshot_v1",
                "ticker": ticker,
                "document_id": document_id,
                "source_kind": "filing",
                "market": "US",
                "generated_at": "2026-02-25T00:00:00+00:00",
            }
        file_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return source_dir


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的字典。

    Raises:
        ValueError: JSON 非法时抛出。
        OSError: 读取失败时抛出。
    """

    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_promote_ground_truth_sample_copies_files_and_updates_manifest(tmp_path: Path) -> None:
    """验证入库会复制文件并更新 manifest。"""

    workspace_root = tmp_path / "workspace"
    baseline_root = tmp_path / "baseline"
    _prepare_source_truth_dir(
        workspace_root=workspace_root,
        ticker="AAPL",
        document_id="fil_001",
    )

    result = promote_ground_truth_sample(
        ticker="AAPL",
        document_id="fil_001",
        workspace_root=workspace_root,
        baseline_root=baseline_root,
        reviewed_by="leo",
        review_note="人工核对通过",
        overwrite=False,
    )

    assert result["ticker"] == "AAPL"
    assert result["document_id"] == "fil_001"
    target_dir = baseline_root / "AAPL" / "fil_001"
    for file_name in GROUND_TRUTH_REQUIRED_FILES:
        assert (target_dir / file_name).exists()

    manifest = _read_json(baseline_root / "manifest.json")
    assert manifest["schema_version"] == "fins_ground_truth_manifest_v1.0.0"
    assert isinstance(manifest["samples"], list)
    assert len(manifest["samples"]) == 1
    sample = manifest["samples"][0]
    assert sample["ticker"] == "AAPL"
    assert sample["document_id"] == "fil_001"
    assert sample["reviewed_by"] == "leo"
    assert sample["review_note"] == "人工核对通过"


@pytest.mark.unit
def test_promote_ground_truth_sample_requires_overwrite_for_existing_target(tmp_path: Path) -> None:
    """验证目标已存在时必须显式覆盖。"""

    workspace_root = tmp_path / "workspace"
    baseline_root = tmp_path / "baseline"
    _prepare_source_truth_dir(
        workspace_root=workspace_root,
        ticker="AAPL",
        document_id="fil_001",
    )

    promote_ground_truth_sample(
        ticker="AAPL",
        document_id="fil_001",
        workspace_root=workspace_root,
        baseline_root=baseline_root,
        reviewed_by="leo",
        review_note="首次入库",
        overwrite=False,
    )

    with pytest.raises(FileExistsError):
        promote_ground_truth_sample(
            ticker="AAPL",
            document_id="fil_001",
            workspace_root=workspace_root,
            baseline_root=baseline_root,
            reviewed_by="leo",
            review_note="重复入库",
            overwrite=False,
        )


@pytest.mark.unit
def test_promote_ground_truth_sample_upserts_manifest_by_ticker(tmp_path: Path) -> None:
    """验证同 ticker 再入库会覆盖 manifest 记录。"""

    workspace_root = tmp_path / "workspace"
    baseline_root = tmp_path / "baseline"
    _prepare_source_truth_dir(
        workspace_root=workspace_root,
        ticker="AAPL",
        document_id="fil_old",
    )
    _prepare_source_truth_dir(
        workspace_root=workspace_root,
        ticker="AAPL",
        document_id="fil_new",
    )

    promote_ground_truth_sample(
        ticker="AAPL",
        document_id="fil_old",
        workspace_root=workspace_root,
        baseline_root=baseline_root,
        reviewed_by="leo",
        review_note="旧样本",
        overwrite=False,
    )
    promote_ground_truth_sample(
        ticker="AAPL",
        document_id="fil_new",
        workspace_root=workspace_root,
        baseline_root=baseline_root,
        reviewed_by="leo",
        review_note="新样本",
        overwrite=False,
    )

    manifest = _read_json(baseline_root / "manifest.json")
    samples = manifest["samples"]
    assert len(samples) == 1
    assert samples[0]["ticker"] == "AAPL"
    assert samples[0]["document_id"] == "fil_new"
    assert samples[0]["review_note"] == "新样本"


@pytest.mark.unit
def test_parse_args_defaults_and_required_fields() -> None:
    """验证命令行参数解析默认值与必填字段。"""

    args = parse_args(["--ticker", "AAPL", "--document-id", "fil_1", "--reviewed-by", "leo"])
    assert args.workspace_root == "workspace"
    assert args.baseline_root == "tests/fixtures/fins/ground_truth"
    assert args.overwrite is False


@pytest.mark.unit
def test_normalize_required_text_and_required_files_errors(tmp_path: Path) -> None:
    """验证必填文本与必需文件校验的异常路径。"""

    with pytest.raises(ValueError, match="ticker 不能为空"):
        require_non_empty_text("  ", empty_error=ValueError("ticker 不能为空"))

    workspace_root = tmp_path / "workspace"
    source_dir = workspace_root / "portfolio" / "AAPL" / "processed" / "fil_missing"
    context = build_fs_storage_test_context(workspace_root)
    context.processed_repository.create_processed(
        ProcessedCreateRequest(
            ticker="AAPL",
            document_id="fil_missing",
            internal_document_id="missing",
            source_kind=SourceKind.FILING.value,
            form_type="10-K",
            meta={"form_type": "10-K", "is_deleted": False},
            sections=[],
            tables=[],
            financials=None,
        )
    )
    with pytest.raises(FileNotFoundError, match="缺少必需快照文件"):
        _assert_required_truth_files(
            blob_repository=context.blob_repository,
            source_handle=context.processed_repository.get_processed_handle("AAPL", "fil_missing"),
        )


@pytest.mark.unit
def test_load_or_init_manifest_and_upsert_filters_invalid_sample(tmp_path: Path) -> None:
    """验证 manifest 初始化、非法结构与 upsert 过滤逻辑。"""

    manifest_path = tmp_path / "manifest.json"
    initialized = _load_or_init_manifest(manifest_path=manifest_path)
    assert initialized["schema_version"] == "fins_ground_truth_manifest_v1.0.0"
    assert initialized["samples"] == []

    manifest_path.write_text(json.dumps({"samples": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest.samples 非法"):
        _load_or_init_manifest(manifest_path=manifest_path)

    payload = {
        "samples": [
            "bad-entry",
            {"ticker": "AAPL", "document_id": "old"},
            {"ticker": "MSFT", "document_id": "keep"},
        ]
    }
    updated = _upsert_manifest_sample(
        manifest_payload=payload,
        ticker="AAPL",
        document_id="new",
        reviewed_by="leo",
        review_note="ok",
        source_locator="processed:AAPL/new",
        source_truth_meta={"source_kind": "filing", "market": "US", "generated_at": "2026-03-02T00:00:00+00:00"},
        copied_files=["a.json"],
    )
    assert len(updated["samples"]) == 2
    assert updated["samples"][0]["ticker"] == "AAPL"
    assert updated["samples"][0]["document_id"] == "new"
    assert updated["samples"][1]["ticker"] == "MSFT"


@pytest.mark.unit
def test_main_runs_and_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """验证脚本入口 main 成功执行并输出摘要。"""

    workspace_root = tmp_path / "workspace"
    baseline_root = tmp_path / "baseline"
    _prepare_source_truth_dir(
        workspace_root=workspace_root,
        ticker="AAPL",
        document_id="fil_100",
    )

    exit_code = main(
        [
            "--ticker",
            "AAPL",
            "--document-id",
            "fil_100",
            "--workspace-root",
            str(workspace_root),
            "--baseline-root",
            str(baseline_root),
            "--reviewed-by",
            "leo",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "tool_snapshot 基线入库完成" in captured.out
    assert "fil_100" in captured.out
