"""分析报告 manifest 解析辅助模块。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_CHAPTERS_DIR_NAME = "chapters"
_FINAL_CHAPTER_STATUSES = frozenset({"passed", "failed"})
_CHAPTER_FILE_PREFIX_PATTERN = re.compile(r"^(?P<index>\d+)_(?P<title>.+)$")


@dataclass(frozen=True)
class ManifestChapterSnapshot:
    """manifest 中的单章节状态快照。"""

    title: str
    index: int
    status: str
    failure_reason: str


def parse_manifest_chapter_snapshots(manifest_path: Path) -> list[ManifestChapterSnapshot]:
    """解析 manifest 中的章节状态快照。"""

    snapshots: list[ManifestChapterSnapshot] = []
    chapters_dir = manifest_path.parent / _CHAPTERS_DIR_NAME
    if chapters_dir.exists() and chapters_dir.is_dir():
        chapter_artifacts = sorted(
            (path for path in chapters_dir.iterdir() if path.is_file()),
            key=lambda path: (path.stat().st_mtime_ns, path.name),
        )
        for chapter_artifact in chapter_artifacts:
            chapter_stem = chapter_artifact.stem
            chapter_identity, separator, _ = chapter_stem.partition(".")
            if not separator:
                chapter_identity = chapter_stem
            match = _CHAPTER_FILE_PREFIX_PATTERN.match(chapter_identity)
            if match is None:
                continue
            raw_index = match.group("index")
            chapter_title = match.group("title")
            try:
                chapter_index = int(raw_index)
            except ValueError:
                continue
            snapshots.append(
                ManifestChapterSnapshot(
                    title=chapter_title,
                    index=chapter_index,
                    status=chapter_artifact.name,
                    failure_reason="",
                )
            )

    raw_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        return _sort_manifest_snapshots(snapshots)
    raw_results = raw_payload.get("chapter_results")
    if not isinstance(raw_results, dict):
        return _sort_manifest_snapshots(snapshots)

    for chapter_title, raw_result in raw_results.items():
        if not isinstance(chapter_title, str) or not isinstance(raw_result, dict):
            continue
        raw_index = raw_result.get("index", 0)
        raw_status = raw_result.get("status", "unknown")
        raw_failure_reason = raw_result.get("failure_reason", "")
        index = raw_index if isinstance(raw_index, int) else 0
        status = raw_status if isinstance(raw_status, str) else "unknown"
        failure_reason = raw_failure_reason if isinstance(raw_failure_reason, str) else ""
        snapshots.append(
            ManifestChapterSnapshot(
                title=chapter_title,
                index=index,
                status=status,
                failure_reason=failure_reason,
            )
        )

    return _sort_manifest_snapshots(snapshots)


def _sort_manifest_snapshots(snapshots: list[ManifestChapterSnapshot]) -> list[ManifestChapterSnapshot]:
    """按章节序号和状态稳定排序。"""

    snapshots.sort(
        key=lambda item: (
            item.index,
            1 if item.status in _FINAL_CHAPTER_STATUSES else 0,
            item.status,
        )
    )
    return snapshots
