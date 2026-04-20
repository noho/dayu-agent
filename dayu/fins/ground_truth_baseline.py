"""快照基线入库脚本。

该模块提供“审核后入库”能力：
1. 通过 `dayu.fins.storage` 的 processed/blob 仓储读取 `tool_snapshot_*.json`。
2. 复制到仓库内 `tests/fixtures/fins/ground_truth/{ticker}/{document_id}`。
3. 维护 `tests/fixtures/fins/ground_truth/manifest.json`，记录样本与审核信息。

设计目标：
- 固化人工审核后的基线，避免回归测试读取可变工作区产物。
- 保持脚本幂等且可追溯（支持覆盖更新与审核人信息记录）。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Optional

from dayu.fins.domain.document_models import ProcessedHandle, now_iso8601
from dayu.fins._converters import require_non_empty_text
from dayu.fins.storage import FsDocumentBlobRepository, FsProcessedDocumentRepository

GROUND_TRUTH_MANIFEST_SCHEMA_VERSION = "fins_ground_truth_manifest_v1.0.0"
GROUND_TRUTH_REQUIRED_FILES = [
    "tool_snapshot_list_documents.json",
    "tool_snapshot_get_document_sections.json",
    "tool_snapshot_read_section.json",
    "tool_snapshot_search_document.json",
    "tool_snapshot_list_tables.json",
    "tool_snapshot_get_table.json",
    "tool_snapshot_get_page_content.json",
    "tool_snapshot_get_financial_statement.json",
    "tool_snapshot_query_xbrl_facts.json",
    "tool_snapshot_meta.json",
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 可选参数列表；为 `None` 时读取进程命令行。

    Returns:
        解析完成的参数对象。

    Raises:
        SystemExit: 参数非法时由 argparse 抛出。
    """

    parser = argparse.ArgumentParser(description="将人工审核通过的 tool_snapshot 入库为固定基线。")
    parser.add_argument("--ticker", required=True, help="样本 ticker，例如 AAPL/TCOM/0300。")
    parser.add_argument("--document-id", required=True, help="样本文档 ID。")
    parser.add_argument(
        "--workspace-root",
        default="workspace",
        help="工作区根目录，默认 ./workspace。",
    )
    parser.add_argument(
        "--baseline-root",
        default="tests/fixtures/fins/ground_truth",
        help="基线目录，默认 ./tests/fixtures/fins/ground_truth。",
    )
    parser.add_argument("--reviewed-by", required=True, help="审核人标识（姓名或工号）。")
    parser.add_argument("--review-note", default="", help="审核备注，可选。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若目标目录已存在，允许覆盖更新。",
    )
    return parser.parse_args(argv)


def promote_ground_truth_sample(
    *,
    ticker: str,
    document_id: str,
    workspace_root: Path,
    baseline_root: Path,
    reviewed_by: str,
    review_note: str,
    overwrite: bool,
) -> dict[str, Any]:
    """执行单个样本的快照基线入库。

    Args:
        ticker: 股票代码。
        document_id: 文档 ID。
        workspace_root: 工作区根目录。
        baseline_root: 基线根目录。
        reviewed_by: 审核人标识。
        review_note: 审核备注。
        overwrite: 目标目录已存在时是否允许覆盖。

    Returns:
        入库结果字典，包含源目录、目标目录与写入文件列表。

    Raises:
        FileNotFoundError: 源目录或必需文件缺失时抛出。
        FileExistsError: 目标目录已存在且未启用覆盖时抛出。
        ValueError: 参数为空或 manifest 内容非法时抛出。
        OSError: 文件读写失败时抛出。
    """

    normalized_ticker = require_non_empty_text(ticker, empty_error=ValueError("ticker 不能为空"))
    normalized_document_id = require_non_empty_text(
        document_id,
        empty_error=ValueError("document_id 不能为空"),
    )
    normalized_reviewer = require_non_empty_text(
        reviewed_by,
        empty_error=ValueError("reviewed_by 不能为空"),
    )
    normalized_note = str(review_note or "").strip()

    processed_repository = FsProcessedDocumentRepository(workspace_root)
    blob_repository = FsDocumentBlobRepository(workspace_root)
    source_handle = _build_processed_handle(
        repository=processed_repository,
        ticker=normalized_ticker,
        document_id=normalized_document_id,
    )
    _assert_required_truth_files(
        blob_repository=blob_repository,
        source_handle=source_handle,
    )

    target_dir = baseline_root / normalized_ticker / normalized_document_id
    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"目标目录已存在，请加 --overwrite: {target_dir}")
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    copied_files = _copy_truth_files(
        blob_repository=blob_repository,
        source_handle=source_handle,
        target_dir=target_dir,
    )
    source_truth_meta = _read_processed_json(
        blob_repository=blob_repository,
        source_handle=source_handle,
        file_name="tool_snapshot_meta.json",
    )
    manifest_path = baseline_root / "manifest.json"
    manifest_payload = _load_or_init_manifest(manifest_path=manifest_path)
    updated_manifest = _upsert_manifest_sample(
        manifest_payload=manifest_payload,
        ticker=normalized_ticker,
        document_id=normalized_document_id,
        reviewed_by=normalized_reviewer,
        review_note=normalized_note,
        source_locator=_format_processed_locator(source_handle),
        source_truth_meta=source_truth_meta,
        copied_files=copied_files,
    )
    _write_json(path=manifest_path, payload=updated_manifest)

    return {
        "ticker": normalized_ticker,
        "document_id": normalized_document_id,
        "source_dir": _format_processed_locator(source_handle),
        "target_dir": str(target_dir),
        "copied_files": copied_files,
        "manifest_path": str(manifest_path),
    }


def _build_processed_handle(
    *,
    repository: FsProcessedDocumentRepository,
    ticker: str,
    document_id: str,
) -> ProcessedHandle:
    """解析 processed 文档句柄。

    Args:
        repository: processed 文档仓储。
        ticker: 股票代码。
        document_id: 文档 ID。

    Returns:
        已存在的 processed 文档句柄。

    Raises:
        FileNotFoundError: processed 文档不存在时抛出。
        OSError: 仓储读取失败时抛出。
    """

    try:
        return repository.get_processed_handle(ticker, document_id)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"源 processed 文档不存在: processed:{ticker}/{document_id}"
        ) from exc


def _format_processed_locator(source_handle: ProcessedHandle) -> str:
    """格式化 processed 文档逻辑定位符。

    Args:
        source_handle: processed 文档句柄。

    Returns:
        逻辑定位符字符串。

    Raises:
        无。
    """

    return f"processed:{source_handle.ticker}/{source_handle.document_id}"

def _assert_required_truth_files(
    *,
    blob_repository: FsDocumentBlobRepository,
    source_handle: ProcessedHandle,
) -> None:
    """校验源文档包含完整快照文件集。

    Args:
        blob_repository: 文档文件仓储。
        source_handle: processed 文档句柄。

    Returns:
        无。

    Raises:
        FileNotFoundError: 存在缺失文件时抛出。
    """

    missing_files: list[str] = []
    existing_entries = {
        entry.name
        for entry in blob_repository.list_entries(source_handle)
        if entry.is_file
    }
    for file_name in GROUND_TRUTH_REQUIRED_FILES:
        if file_name not in existing_entries:
            missing_files.append(file_name)
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(f"缺少必需快照文件: {missing_text}")


def _copy_truth_files(
    *,
    blob_repository: FsDocumentBlobRepository,
    source_handle: ProcessedHandle,
    target_dir: Path,
) -> list[str]:
    """复制快照文件到目标目录。

    Args:
        blob_repository: 文档文件仓储。
        source_handle: processed 文档句柄。
        target_dir: 目标目录。

    Returns:
        已复制文件名列表（按常量顺序）。

    Raises:
        OSError: 文件复制失败时抛出。
    """

    copied_files: list[str] = []
    for file_name in GROUND_TRUTH_REQUIRED_FILES:
        target_file = target_dir / file_name
        file_bytes = blob_repository.read_file_bytes(source_handle, file_name)
        target_file.write_bytes(file_bytes)
        copied_files.append(file_name)
    return copied_files


def _load_or_init_manifest(*, manifest_path: Path) -> dict[str, Any]:
    """读取 manifest；若不存在则返回默认结构。

    Args:
        manifest_path: manifest 文件路径。

    Returns:
        manifest 字典。

    Raises:
        ValueError: manifest 顶层结构非法时抛出。
        OSError: 文件读取失败时抛出。
    """

    if not manifest_path.exists():
        return {
            "schema_version": GROUND_TRUTH_MANIFEST_SCHEMA_VERSION,
            "updated_at": now_iso8601(),
            "samples": [],
        }
    payload = _read_json(manifest_path)
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"manifest.samples 非法: {manifest_path}")
    payload["schema_version"] = GROUND_TRUTH_MANIFEST_SCHEMA_VERSION
    return payload


def _upsert_manifest_sample(
    *,
    manifest_payload: dict[str, Any],
    ticker: str,
    document_id: str,
    reviewed_by: str,
    review_note: str,
    source_locator: str,
    source_truth_meta: dict[str, Any],
    copied_files: list[str],
) -> dict[str, Any]:
    """向 manifest 写入或更新指定 ticker 的样本记录。

    Args:
        manifest_payload: 已加载 manifest 数据。
        ticker: 股票代码。
        document_id: 文档 ID。
        reviewed_by: 审核人。
        review_note: 审核备注。
        source_locator: 源文档逻辑定位符。
        source_truth_meta: 源 `tool_snapshot_meta.json`。
        copied_files: 已复制文件列表。

    Returns:
        更新后的 manifest 字典。

    Raises:
        ValueError: manifest 结构非法时抛出。
    """

    samples = manifest_payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("manifest.samples 必须为 list")

    reviewed_at = now_iso8601()
    entry = {
        "ticker": ticker,
        "document_id": document_id,
        "source_kind": str(source_truth_meta.get("source_kind", "")),
        "market": str(source_truth_meta.get("market", "")),
        "tool_snapshot_generated_at": str(source_truth_meta.get("generated_at", "")),
        "reviewed_by": reviewed_by,
        "review_note": review_note,
        "reviewed_at": reviewed_at,
        "promoted_from": source_locator,
        "files": copied_files,
    }

    filtered_samples: list[dict[str, Any]] = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_ticker = str(sample.get("ticker", "")).strip().upper()
        if sample_ticker == ticker.upper():
            continue
        filtered_samples.append(sample)
    filtered_samples.append(entry)
    filtered_samples.sort(key=lambda item: str(item.get("ticker", "")))

    return {
        "schema_version": GROUND_TRUTH_MANIFEST_SCHEMA_VERSION,
        "updated_at": reviewed_at,
        "samples": filtered_samples,
    }


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 文件并返回对象。

    Args:
        path: JSON 文件路径。

    Returns:
        解析后的字典对象。

    Raises:
        ValueError: JSON 非法时抛出。
        OSError: 读取失败时抛出。
    """

    return json.loads(path.read_text(encoding="utf-8"))


def _read_processed_json(
    *,
    blob_repository: FsDocumentBlobRepository,
    source_handle: ProcessedHandle,
    file_name: str,
) -> dict[str, Any]:
    """读取 processed 文档内的 JSON 文件。

    Args:
        blob_repository: 文档文件仓储。
        source_handle: processed 文档句柄。
        file_name: 目标文件名。

    Returns:
        解析后的字典对象。

    Raises:
        ValueError: JSON 非法时抛出。
        FileNotFoundError: 文件不存在时抛出。
        OSError: 仓储读取失败时抛出。
    """

    file_bytes = blob_repository.read_file_bytes(source_handle, file_name)
    payload = json.loads(file_bytes.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 顶层必须为对象: {file_name}")
    return payload


def _write_json(*, path: Path, payload: dict[str, Any]) -> None:
    """写入 JSON 文件（UTF-8，缩进 2）。

    Args:
        path: 目标路径。
        payload: 待写入 JSON 对象。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[list[str]] = None) -> int:
    """脚本入口。

    Args:
        argv: 可选参数列表；为 `None` 时读取进程命令行。

    Returns:
        进程退出码：`0` 表示成功。

    Raises:
        FileNotFoundError: 源目录或文件缺失时抛出。
        FileExistsError: 目标目录冲突且未覆盖时抛出。
        ValueError: 参数非法或 manifest 非法时抛出。
        OSError: 文件读写失败时抛出。
    """

    args = parse_args(argv)
    workspace_root = Path(args.workspace_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    result = promote_ground_truth_sample(
        ticker=args.ticker,
        document_id=args.document_id,
        workspace_root=workspace_root,
        baseline_root=baseline_root,
        reviewed_by=args.reviewed_by,
        review_note=args.review_note,
        overwrite=bool(args.overwrite),
    )
    print("tool_snapshot 基线入库完成")
    print(f"- ticker: {result['ticker']}")
    print(f"- document_id: {result['document_id']}")
    print(f"- source_dir: {result['source_dir']}")
    print(f"- target_dir: {result['target_dir']}")
    print(f"- manifest: {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
