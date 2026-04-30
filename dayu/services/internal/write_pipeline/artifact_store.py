"""写作流水线产物存储管理。

本模块从 pipeline.py 中抽离，封装 manifest CRUD、
章节产物落盘与加载、报告组装、路径计算等 I/O 操作。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Optional, TextIO

import dayu.file_lock as file_lock_module

from dayu.log import Log
from dayu.services.internal.write_pipeline.audit_rules import (
    RepairPlanApplyResult,
    _build_confirm_artifact_payload,
)
from dayu.services.internal.write_pipeline.models import (
    AuditDecision,
    ChapterResult,
    ChapterTask,
    EvidenceConfirmationResult,
    RunManifest,
    SourceEntry,
    WriteRunConfig,
)


MODULE = "APP.WRITE_PIPELINE"

_MANIFEST_FILE_NAME = "manifest.json"

_MANIFEST_LOCK_FILE_NAME = ".manifest.lock"

_MANIFEST_LOCK_REGION_BYTES = 1

_CHAPTERS_DIR_NAME = "chapters"

_OVERVIEW_CHAPTER_TITLE = "投资要点概览"
_SOURCE_CHAPTER_TITLE = "来源清单"


def _build_phase_artifact_name(*, phase: str, kind: str) -> str:
    """构建阶段产物名。

    Args:
        phase: 阶段名，如 ``initial``、``repair_1``、``regenerate_1``。
        kind: 产物类型，如 ``write``、``audit``、``audit_suspect``、``confirm``。

    Returns:
        统一的阶段产物名。

    Raises:
        ValueError: 当 ``phase`` 或 ``kind`` 为空时抛出。
    """

    normalized_phase = phase.strip()
    normalized_kind = kind.strip()
    if not normalized_phase or not normalized_kind:
        raise ValueError("阶段名和产物类型不能为空")
    return f"{normalized_phase}_{normalized_kind}"


def _fsync_parent_directory(path: Path) -> None:
    """尽力 fsync 父目录，降低 rename 后目录元数据丢失风险。

    Args:
        path: 目标文件路径。

    Returns:
        无。

    Raises:
        无。
    """

    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        return
    finally:
        os.close(dir_fd)


def _atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """通过临时文件 + 原子替换写入文本，避免目标文件出现半写状态。

    Args:
        path: 目标文件路径。
        content: 待写入文本。
        encoding: 文本编码。

    Returns:
        无。

    Raises:
        OSError: 写入或替换失败时抛出。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path_text = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    temp_path = Path(temp_path_text)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
        _fsync_parent_directory(path)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
def _acquire_manifest_stream_lock(stream: TextIO) -> None:
    """获取 manifest 锁文件的跨进程排他锁。

    Args:
        stream: 已打开的锁文件流。

    Returns:
        无。

    Raises:
        OSError: 当前平台没有可用锁实现，或底层加锁失败时抛出。
    """

    file_lock_module.acquire_text_file_lock(
        stream,
        blocking=True,
        region_bytes=_MANIFEST_LOCK_REGION_BYTES,
        lock_name="manifest 文件锁",
    )


def _release_manifest_stream_lock(stream: TextIO) -> None:
    """释放 manifest 锁文件的跨进程排他锁。

    Args:
        stream: 已打开且已持锁的锁文件流。

    Returns:
        无。

    Raises:
        OSError: 底层解锁失败时抛出。
    """

    file_lock_module.release_text_file_lock(
        stream,
        region_bytes=_MANIFEST_LOCK_REGION_BYTES,
        lock_name="manifest 文件锁",
    )


@contextmanager
def _manifest_file_lock(output_dir: Path) -> Iterator[TextIO]:
    """对写作输出目录内的 manifest 相关读写加跨进程文件锁。

    Args:
        output_dir: 写作输出目录。

    Yields:
        已打开并持有排他锁的锁文件对象。

    Raises:
        OSError: 锁文件打开或加锁失败时抛出。
    """

    lock_path = output_dir / _MANIFEST_LOCK_FILE_NAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        _acquire_manifest_stream_lock(stream)
        try:
            yield stream
        finally:
            _release_manifest_stream_lock(stream)


# 章节 write 阶段产物文件名形如 `{idx}_{slug}.{stage}_write.md`；
# 这里给每个 stage 指定单调递增的优先级，后面补上阶段重试序号作为次序键，
# 保证排序语义只依赖文件名本身，不被跨平台 mtime 粒度（Windows NTFS 可能 ~15ms）干扰。
_CHAPTER_WRITE_STAGE_ORDER: dict[str, int] = {
    "initial": 0,
    "initial_fix_placeholders": 1,
    "repair": 2,
    "regenerate": 3,
}
_CHAPTER_WRITE_STAGE_PATTERN = re.compile(
    r"^(?P<stage>initial_fix_placeholders|initial|repair|regenerate)(?:_(?P<retry>\d+))?_write$"
)


def _chapter_write_artifact_sort_key(path: Path) -> tuple[int, int, int, str]:
    """按轮次 → 阶段优先级为 write 中间稿排序。

    轮次（retry）是时间主轴，阶段（stage）仅在同一轮次内决定先后。
    这样跨轮次切换策略（如先 regenerate 再 repair）时，较新轮次的产物
    始终排在较旧轮次之后，避免回退到更旧中间稿。

    Args:
        path: 形如 ``{idx}_{slug}.{stage}[_{retry}]_write.md`` 的 write 产物路径。

    Returns:
        `(retry, stage_order, 0, full_stem)` 四元组；未知格式退化到
        `(-1, -1, 0, stem)` 排到最前。

    Raises:
        无。
    """

    stem_parts = path.stem.split(".", 1)
    if len(stem_parts) != 2:
        return (-1, -1, 0, path.stem)
    stage_token = stem_parts[1]
    match = _CHAPTER_WRITE_STAGE_PATTERN.match(stage_token)
    if match is None:
        return (-1, -1, 0, path.stem)
    stage = match.group("stage")
    retry = int(match.group("retry") or 0)
    return (retry, _CHAPTER_WRITE_STAGE_ORDER[stage], 0, path.stem)


def _slugify_title(title: str) -> str:
    """将章节标题转为文件名 slug。

    Args:
        title: 原始标题。

    Returns:
        slug 字符串。

    Raises:
        无。
    """

    slug = re.sub(r"\s+", "_", title.strip())
    slug = re.sub(r"[^\w\u4e00-\u9fff\-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "chapter"


def _read_manifest_from_dir(output_dir: Path) -> Optional[RunManifest]:
    """从输出目录读取 manifest.json。

    Args:
        output_dir: 写作输出目录路径。

    Returns:
        RunManifest 实例；若文件不存在或解析失败则返回 None。

    Raises:
        无。
    """

    try:
        with _manifest_file_lock(output_dir):
            manifest_path = output_dir / _MANIFEST_FILE_NAME
            if not manifest_path.exists():
                return None
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            return RunManifest.from_dict(raw)
    except Exception as exc:  # noqa: BLE001
        Log.error(f"读取 manifest.json 失败: {exc}", module=MODULE)
        return None



class ArtifactStore:
    """写作流水线产物存储管理器。

    封装 manifest CRUD、章节产物落盘/加载、报告组装、
    路径计算等所有文件 I/O 操作。

    Attributes:
        _output_dir: 执行输出根目录。
        _chapters_dir: 章节产物目录。
        _manifest_path: manifest 文件路径。
        _write_config: 写作运行配置。
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        chapters_dir: Path,
        manifest_path: Path,
        write_config: WriteRunConfig,
    ) -> None:
        """初始化产物存储管理器。

        Args:
            output_dir: 执行输出根目录。
            chapters_dir: 章节产物目录。
            manifest_path: manifest 文件路径。
            write_config: 写作运行配置。

        Returns:
            无。

        Raises:
            无。
        """

        self._output_dir = output_dir
        self._chapters_dir = chapters_dir
        self._manifest_path = manifest_path
        self._write_config = write_config

    def load_chapter_content_for_decision(self, result: ChapterResult) -> str:
        """公开读取第10章决策输入所需的章节正文。

        Args:
            result: 章节执行结果。

        Returns:
            可用于决策输入的章节正文。

        Raises:
            无。
        """

        return self._load_chapter_content_for_decision(result)

    def load_final_audit_payload(self, result: ChapterResult) -> dict[str, Any] | None:
        """公开读取章节最终 audit 产物。

        Args:
            result: 章节执行结果。

        Returns:
            最终 audit JSON；不存在时返回 ``None``。

        Raises:
            无。
        """

        return self._load_final_audit_payload(result)

    def remove_sources_json_if_exists(self) -> None:
        """公开删除来源清单 JSON 产物。"""

        self._remove_sources_json_if_exists()

    def load_or_create_manifest(self, signature: str) -> RunManifest:
        """公开加载或创建运行清单。"""

        return self._load_or_create_manifest(signature)

    def persist_manifest(self, *, manifest: RunManifest, chapter_results: dict[str, ChapterResult]) -> None:
        """公开持久化运行清单。"""

        self._persist_manifest(manifest=manifest, chapter_results=chapter_results)

    def read_manifest_from_disk_unlocked(self) -> RunManifest | None:
        """公开在已持锁前提下读取磁盘 manifest。"""

        return self._read_manifest_from_disk_unlocked()

    def persist_chapter_artifacts(self, result: ChapterResult) -> None:
        """公开落盘章节最终正文。"""

        self._persist_chapter_artifacts(result)

    def purge_chapter_artifacts(self, task: ChapterTask) -> None:
        """清除指定章节在 ``chapters/`` 目录下的全部历史产物。

        触发时机：
        - 单章重写（``cli write --chapter``）、并行中段章节重跑、决策章重跑、
          概览章重跑前，先把旧产物删除，避免后续 fallback 路径读到陈旧最终正文
          或阶段中间稿。

        清理口径：
        - 章节相关产物全部以 ``{display_index:02d}_{slug}.`` 为文件名前缀
          平铺在 ``chapters/`` 目录下（最终正文 ``.md``、阶段产物
          ``.{phase}.{ext}``）；按前缀一次性清理，与 ``_chapter_file_path`` /
          ``_chapter_phase_artifact_path`` 的写入口径保持一致。

        Args:
            task: 待清理章节任务。

        Returns:
            无。

        Raises:
            无。
        """

        self._purge_chapter_artifacts(task)

    def load_latest_failed_chapter_content(self, task: ChapterTask) -> str:
        """公开加载章节失败时的最新中间稿。"""

        return self._load_latest_failed_chapter_content(task)

    def persist_write_artifact(self, *, task: ChapterTask, artifact_name: str, content: str) -> None:
        """公开按阶段落盘写作输出。"""

        self._persist_write_artifact(task=task, artifact_name=artifact_name, content=content)

    def persist_repair_input_artifacts(
        self,
        *,
        task: ChapterTask,
        retry_count: int,
        current_content: str,
        prompt_inputs: dict[str, Any],
    ) -> None:
        """公开落盘 repair 前输入上下文。"""

        self._persist_repair_input_artifacts(
            task=task,
            retry_count=retry_count,
            current_content=current_content,
            prompt_inputs=prompt_inputs,
        )

    def persist_repair_plan_artifact(self, *, task: ChapterTask, retry_count: int, content: str) -> None:
        """公开落盘 repair 计划原文。"""

        self._persist_repair_plan_artifact(task=task, retry_count=retry_count, content=content)

    def persist_repair_apply_result_artifact(
        self,
        *,
        task: ChapterTask,
        retry_count: int,
        apply_result: RepairPlanApplyResult,
    ) -> None:
        """公开落盘 repair 应用结果。"""

        self._persist_repair_apply_result_artifact(
            task=task,
            retry_count=retry_count,
            apply_result=apply_result,
        )

    def persist_phase_audit_suspect_artifact(
        self,
        *,
        task: ChapterTask,
        phase: str,
        audit_decision: AuditDecision,
    ) -> None:
        """公开落盘阶段疑似审计结果。"""

        self._persist_phase_audit_suspect_artifact(task=task, phase=phase, audit_decision=audit_decision)

    def persist_phase_audit_artifact(
        self,
        *,
        task: ChapterTask,
        phase: str,
        audit_decision: AuditDecision,
    ) -> None:
        """公开落盘阶段最终审计结果。"""

        self._persist_phase_audit_artifact(task=task, phase=phase, audit_decision=audit_decision)

    def persist_final_audit_artifact(self, *, task: ChapterTask, audit_decision: AuditDecision) -> None:
        """公开落盘章节最终 audit 产物。

        Args:
            task: 章节任务。
            audit_decision: 最终审计结果。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        path = self._chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name="audit",
            extension="json",
        )
        _atomic_write_text(
            path,
            json.dumps(_build_audit_artifact_payload(audit_decision), ensure_ascii=False, indent=2),
        )

    def persist_phase_confirm_artifact(
        self,
        *,
        task: ChapterTask,
        phase: str,
        confirmation_result: EvidenceConfirmationResult,
    ) -> None:
        """公开落盘阶段证据复核结果。"""

        self._persist_phase_confirm_artifact(
            task=task,
            phase=phase,
            confirmation_result=confirmation_result,
        )

    def persist_phase_confirm_raw_artifact(self, *, task: ChapterTask, phase: str, raw_text: str) -> None:
        """公开落盘 confirm 原始输出。"""

        self._persist_phase_confirm_raw_artifact(task=task, phase=phase, raw_text=raw_text)

    def persist_phase_confirm_parse_error_artifact(
        self,
        *,
        task: ChapterTask,
        phase: str,
        parse_error: str,
    ) -> None:
        """公开落盘 confirm 解析错误。"""

        self._persist_phase_confirm_parse_error_artifact(
            task=task,
            phase=phase,
            parse_error=parse_error,
        )

    def persist_sources_json(self, source_entries: list[SourceEntry]) -> None:
        """公开落盘来源去重清单。"""

        self._persist_sources_json(source_entries)

    def chapter_file_path(self, index: int, title: str) -> Path:
        """公开获取章节正文输出路径。"""

        return self._chapter_file_path(index, title)

    def chapter_phase_artifact_path(
        self,
        *,
        index: int,
        title: str,
        artifact_name: str,
        extension: str,
    ) -> Path:
        """公开获取章节阶段产物路径。"""

        return self._chapter_phase_artifact_path(
            index=index,
            title=title,
            artifact_name=artifact_name,
            extension=extension,
        )

    def _load_chapter_content_for_decision(self, result: ChapterResult) -> str:
        """为第10章优先读取已落盘章节正文。

        Args:
            result: 章节执行结果。

        Returns:
            可用于构建决策输入的章节正文。

        Raises:
            无。
        """

        chapter_path = self._chapter_file_path(result.index, result.title)
        try:
            if chapter_path.exists():
                disk_content = chapter_path.read_text(encoding="utf-8").strip()
                if disk_content:
                    return disk_content
        except OSError as exc:
            Log.warn(f"读取第10章前文章节正文失败，回退 manifest 内容: {chapter_path}, error={exc}", module=MODULE)
        return result.content

    def _load_final_audit_payload(self, result: ChapterResult) -> dict[str, Any] | None:
        """读取章节最终 audit 产物，用于构建第10章未决问题摘要。

        Args:
            result: 章节执行结果。

        Returns:
            最终 audit JSON；若不存在或读取失败则返回 ``None``。

        Raises:
            无。
        """

        audit_path = self._chapter_phase_artifact_path(
            index=result.index,
            title=result.title,
            artifact_name="audit",
            extension="json",
        )
        try:
            if not audit_path.exists():
                return None
            raw = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            Log.warn(f"读取章节最终 audit 产物失败: {audit_path}, error={exc}", module=MODULE)
            return None
        return raw if isinstance(raw, dict) else None

    def _remove_sources_json_if_exists(self) -> None:
        """删除陈旧的来源清单 JSON 产物。

        当模板不包含“来源清单”时，不应继续保留上一次运行遗留的
        `sources_dedup.json`，否则会造成输出目录语义不一致。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        sources_path = self._output_dir / "sources_dedup.json"
        if sources_path.exists():
            sources_path.unlink()

    def _load_or_create_manifest(self, signature: str) -> RunManifest:
        """加载或创建运行清单。

        Args:
            signature: 当前运行签名。

        Returns:
            运行清单对象。

        Raises:
            ValueError: 清单文件格式损坏时抛出。
        """

        with _manifest_file_lock(self._output_dir):
            current = self._read_manifest_from_disk_unlocked()
        if self._write_config.resume and current is not None:
            if current.signature == signature:
                return current
            Log.warn("检测到 manifest 签名变更，将重新开始从头写作", module=MODULE)
            self._purge_stale_run_artifacts()

        return RunManifest(
            version="write_manifest_v1",
            signature=signature,
            config=self._write_config,
            chapter_results={},
            company_facets=None,
        )

    def _purge_stale_run_artifacts(self) -> None:
        """清空 chapters/ 目录下的全部历史产物以及 sources_dedup.json。

        当 manifest 签名变更时，旧的章节正文/中间稿/sources 落盘文件不再与
        新的运行匹配，必须主动清理；否则单章 fallback 与决策章节恢复路径
        会读到陈旧文件，污染本次报告。

        清理失败被降级为 warning 日志，不阻塞主流程：清理是恢复保证，
        即便残留也由后续覆盖写入或人工介入处置。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        try:
            if self._chapters_dir.exists():
                for entry in self._chapters_dir.iterdir():
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink(missing_ok=True)
                Log.info(
                    f"已清理 chapters 目录下陈旧产物: {self._chapters_dir}",
                    module=MODULE,
                )
        except OSError as exc:
            Log.warning(
                f"清理 chapters 目录失败，可能残留陈旧文件: dir={self._chapters_dir} error={exc}",
                module=MODULE,
            )
        sources_path = self._output_dir / "sources_dedup.json"
        try:
            if sources_path.exists():
                sources_path.unlink()
                Log.info(f"已清理陈旧 sources_dedup.json: {sources_path}", module=MODULE)
        except OSError as exc:
            Log.warning(
                f"清理 sources_dedup.json 失败: path={sources_path} error={exc}",
                module=MODULE,
            )

    def _purge_chapter_artifacts(self, task: ChapterTask) -> None:
        """清除指定章节在 ``chapters/`` 目录下的所有平铺产物。

        清理策略：
        - 章节最终正文与所有阶段产物均以 ``{display_index:02d}_{slug}.``
          为文件名前缀直接平铺在 ``chapters/`` 目录下；按前缀枚举该目录、
          删除所有匹配文件，覆盖最终正文 ``.md``、阶段中间稿
          (``initial_write.md``、``repair_N_*.md``/``json``、
          ``regenerate_N_*.md``/``json`` 等)。
        - 单文件删除失败降级为 warning，不阻塞主流程：清理是恢复保证，
          失败时由后续阶段继续覆盖写入。

        Args:
            task: 待清理章节任务。

        Returns:
            无。

        Raises:
            无。
        """

        if not self._chapters_dir.exists():
            return
        display_index = self._chapter_display_index(index=task.index, title=task.title)
        slug = _slugify_title(task.title)
        prefix = f"{display_index:02d}_{slug}."
        try:
            entries = list(self._chapters_dir.iterdir())
        except OSError as exc:
            Log.warning(
                f"枚举 chapters 目录失败，跳过章节产物清理: dir={self._chapters_dir} error={exc}",
                module=MODULE,
            )
            return
        for entry in entries:
            if not entry.is_file() or not entry.name.startswith(prefix):
                continue
            try:
                entry.unlink(missing_ok=True)
            except OSError as exc:
                Log.warning(
                    f"清理章节产物失败: path={entry} error={exc}",
                    module=MODULE,
                )

    def _persist_manifest(self, *, manifest: RunManifest, chapter_results: dict[str, ChapterResult]) -> None:
        """持久化运行清单。

        Args:
            manifest: 当前 manifest。
            chapter_results: 最新章节结果。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        with _manifest_file_lock(self._output_dir):
            persisted_manifest = self._read_manifest_from_disk_unlocked()
            merged_results = dict(chapter_results)
            if (
                persisted_manifest is not None
                and persisted_manifest.signature == manifest.signature
                and persisted_manifest.config.ticker == manifest.config.ticker
            ):
                merged_results = dict(persisted_manifest.chapter_results)
                merged_results.update(chapter_results)
            to_write = RunManifest(
                version=manifest.version,
                signature=manifest.signature,
                config=manifest.config,
                chapter_results=merged_results,
                company_facets=manifest.company_facets,
            )
            _atomic_write_text(
                self._manifest_path,
                json.dumps(to_write.to_dict(), ensure_ascii=False, indent=2),
            )

    def _read_manifest_from_disk_unlocked(self) -> RunManifest | None:
        """在已持有 manifest 文件锁时读取当前磁盘 manifest。

        Args:
            无。

        Returns:
            解析成功的 manifest；若文件不存在则返回 `None`。

        Raises:
            ValueError: 当 manifest 内容非法时抛出。
            OSError: 当 manifest 读取失败时抛出。
        """

        if not self._manifest_path.exists():
            return None
        raw = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return RunManifest.from_dict(raw)

    def _persist_chapter_artifacts(self, result: ChapterResult) -> None:
        """落盘章节最终正文。

        Args:
            result: 章节执行结果。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        if result.status == "failed" and not result.content.strip():
            return
        chapter_path = self._chapter_file_path(result.index, result.title)
        _atomic_write_text(chapter_path, result.content)

    def _load_latest_failed_chapter_content(self, task: ChapterTask) -> str:
        """加载章节失败时可保留的最新中间稿。

        设计目标：
        - 运行中若已产出 `initial_write.md`、`initial_fix_placeholders.md`、
          `repair_N_write.md` 或 `regenerate_N_write.md`，异常兜底时优先保留最新中间稿。
        - 若没有任何中间稿，则返回空字符串，不再把章节骨架写成最终章节正文。

        Args:
            task: 当前章节任务。

        Returns:
            最新可用中间稿正文；若不存在则返回空字符串。

        Raises:
            无。
        """

        chapter_dir = self._chapter_file_path(task.index, task.title).parent
        if not chapter_dir.exists():
            return ""
        display_index = self._chapter_display_index(index=task.index, title=task.title)
        slug = _slugify_title(task.title)
        candidates = sorted(
            chapter_dir.glob(f"{display_index:02d}_{slug}.*_write.md"),
            key=_chapter_write_artifact_sort_key,
            reverse=True,
        )
        for candidate in candidates:
            try:
                content = candidate.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if content:
                return content
        return ""

    def _persist_write_artifact(self, *, task: ChapterTask, artifact_name: str, content: str) -> None:
        """按阶段落盘写作输出。

        Args:
            task: 章节任务。
            artifact_name: 产物名，如 ``initial_write``、``repair_1_write``。
            content: 写作输出正文。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        path = self._chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name=artifact_name,
            extension="md",
        )
        _atomic_write_text(path, content)

    def _persist_rewrite_artifact(self, *, task: ChapterTask, retry_count: int, content: str) -> None:
        """落盘重试阶段中间稿。

        Args:
            task: 章节任务。
            retry_count: 重试轮次。
            content: 中间稿内容。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        self._persist_write_artifact(
            task=task,
            artifact_name=_build_phase_artifact_name(phase=f"repair_{retry_count}", kind="write"),
            content=content,
        )

    def _persist_repair_input_artifacts(
        self,
        *,
        task: ChapterTask,
        retry_count: int,
        current_content: str,
        prompt_inputs: dict[str, Any],
    ) -> None:
        """落盘 repair 执行前的关键输入上下文。

        Args:
            task: 章节任务。
            retry_count: repair 轮次。
            current_content: 进入 repair 前的章节正文。
            prompt_inputs: repair prompt 的结构化输入字段。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        self._persist_write_artifact(
            task=task,
            artifact_name=_build_phase_artifact_name(phase=f"repair_{retry_count}", kind="input_write"),
            content=current_content,
        )
        context_payload = {
            "chapter": prompt_inputs.get("chapter", ""),
            "company": prompt_inputs.get("company", ""),
            "ticker": prompt_inputs.get("ticker", ""),
            "allow_new_facts": bool(prompt_inputs.get("allow_new_facts", False)),
            "retry_scope": str(prompt_inputs.get("retry_scope", "")),
            "current_visible_headings": str(prompt_inputs.get("current_visible_headings", "")),
            "last_repair_contract": prompt_inputs.get("last_repair_contract", {}),
            "chapter_contract": prompt_inputs.get("chapter_contract", {}),
            "input_write_artifact": _build_phase_artifact_name(phase=f"repair_{retry_count}", kind="input_write") + ".md",
        }
        path = self._chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name=_build_phase_artifact_name(phase=f"repair_{retry_count}", kind="context"),
            extension="json",
        )
        _atomic_write_text(path, json.dumps(context_payload, ensure_ascii=False, indent=2))

    def _persist_repair_plan_artifact(self, *, task: ChapterTask, retry_count: int, content: str) -> None:
        """落盘 repair patch 计划原文。

        Args:
            task: 章节任务。
            retry_count: repair 轮次。
            content: patch 计划原始 JSON 文本。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        path = self._chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name=_build_phase_artifact_name(phase=f"repair_{retry_count}", kind="repair_plan"),
            extension="json",
        )
        _atomic_write_text(path, content)

    def _persist_repair_apply_result_artifact(
        self,
        *,
        task: ChapterTask,
        retry_count: int,
        apply_result: RepairPlanApplyResult,
    ) -> None:
        """落盘 repair patch 的逐条应用结果。

        Args:
            task: 章节任务。
            retry_count: repair 轮次。
            apply_result: patch 应用结果。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        path = self._chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name=_build_phase_artifact_name(phase=f"repair_{retry_count}", kind="apply_result"),
            extension="json",
        )
        _atomic_write_text(path, json.dumps(apply_result.to_dict(), ensure_ascii=False, indent=2))

    def _persist_phase_audit_suspect_artifact(
        self,
        *,
        task: ChapterTask,
        phase: str,
        audit_decision: AuditDecision,
    ) -> None:
        """按阶段落盘疑似审计输出。

        Args:
            task: 章节任务。
            phase: 审计阶段名，如 ``initial``、``repair_1``、``regenerate_1``。
            audit_decision: 疑似审计结果。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        path = self._chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name=_build_phase_artifact_name(phase=phase, kind="audit_suspect"),
            extension="json",
        )
        _atomic_write_text(
            path,
            json.dumps(_build_audit_artifact_payload(audit_decision), ensure_ascii=False, indent=2),
        )

    def _persist_phase_audit_artifact(
        self,
        *,
        task: ChapterTask,
        phase: str,
        audit_decision: AuditDecision,
    ) -> None:
        """按阶段落盘合并后的最终审计输出。

        Args:
            task: 章节任务。
            phase: 审计阶段名，如 ``initial``、``repair_1``、``regenerate_1``。
            audit_decision: 合并后的最终审计结果。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        path = self._chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name=_build_phase_artifact_name(phase=phase, kind="audit"),
            extension="json",
        )
        _atomic_write_text(
            path,
            json.dumps(_build_audit_artifact_payload(audit_decision), ensure_ascii=False, indent=2),
        )

    def _persist_phase_confirm_artifact(
        self,
        *,
        task: ChapterTask,
        phase: str,
        confirmation_result: EvidenceConfirmationResult,
    ) -> None:
        """按阶段落盘证据复核输出。

        Args:
            task: 章节任务。
            phase: 阶段名，如 ``initial``、``repair_1``。
            confirmation_result: 证据复核结果。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        path = self._chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name=_build_phase_artifact_name(phase=phase, kind="confirm"),
            extension="json",
        )
        _atomic_write_text(
            path,
            json.dumps(_build_confirm_artifact_payload(confirmation_result), ensure_ascii=False, indent=2),
        )

    def _persist_phase_confirm_raw_artifact(
        self,
        *,
        task: ChapterTask,
        phase: str,
        raw_text: str,
    ) -> None:
        """按阶段落盘 confirm 原始输出。

        Args:
            task: 章节任务。
            phase: 阶段名，如 ``initial``、``repair_1``。
            raw_text: confirm 原始输出文本。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        path = self._chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name=_build_phase_artifact_name(phase=phase, kind="confirm_raw"),
            extension="txt",
        )
        _atomic_write_text(path, raw_text)

    def _persist_phase_confirm_parse_error_artifact(
        self,
        *,
        task: ChapterTask,
        phase: str,
        parse_error: str,
    ) -> None:
        """按阶段落盘 confirm 解析错误信息。

        Args:
            task: 章节任务。
            phase: 阶段名，如 ``initial``、``repair_1``。
            parse_error: 解析异常说明。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        path = self._chapter_phase_artifact_path(
            index=task.index,
            title=task.title,
            artifact_name=_build_phase_artifact_name(phase=phase, kind="confirm_parse_error"),
            extension="json",
        )
        payload = {
            "phase": phase,
            "chapter": task.title,
            "ticker": self._write_config.ticker,
            "parse_error": parse_error,
        }
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _persist_sources_json(self, source_entries: list[SourceEntry]) -> None:
        """落盘来源去重清单。

        Args:
            source_entries: 去重来源条目。

        Returns:
            无。

        Raises:
            OSError: 写文件失败时抛出。
        """

        payload = [asdict(entry) for entry in source_entries]
        (self._output_dir / "sources_dedup.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _chapter_file_path(self, index: int, title: str) -> Path:
        """获取章节正文输出文件路径。

        Args:
            index: 章节序号。
            title: 章节标题。

        Returns:
            文件路径。

        Raises:
            无。
        """

        display_index = self._chapter_display_index(index=index, title=title)
        slug = _slugify_title(title)
        return self._chapters_dir / f"{display_index:02d}_{slug}.md"

    def _chapter_phase_artifact_path(
        self,
        *,
        index: int,
        title: str,
        artifact_name: str,
        extension: str,
    ) -> Path:
        """获取章节阶段产物路径。

        Args:
            index: 章节序号。
            title: 章节标题。
            artifact_name: 产物名，如 ``initial_write``、``repair_1_audit``。
            extension: 文件扩展名，不含前导点。

        Returns:
            阶段产物路径。

        Raises:
            无。
        """

        normalized_artifact_name = _normalize_artifact_name(artifact_name)
        normalized_extension = extension.lstrip(".")
        display_index = self._chapter_display_index(index=index, title=title)
        slug = _slugify_title(title)
        return self._chapters_dir / f"{display_index:02d}_{slug}.{normalized_artifact_name}.{normalized_extension}"

    def _chapter_display_index(self, *, index: int, title: str) -> int:
        """返回章节文件命名使用的显示序号。

        设计口径：
        - 第0章为“投资要点概览”。
        - 第1章为“公司做的是什么生意”。
        - 内部任务仍保留模板顺序索引；文件命名时再做显示层映射。

        Args:
            index: 模板顺序索引。
            title: 章节标题。

        Returns:
            文件命名使用的显示序号。

        Raises:
            无。
        """

        if title == _OVERVIEW_CHAPTER_TITLE:
            return 0
        if index <= 0:
            return index
        return index - 1

def _normalize_artifact_name(artifact_name: str) -> str:
    """标准化章节阶段产物名。

    Args:
        artifact_name: 原始产物名。

    Returns:
        仅保留字母、数字、下划线、中划线和点号的安全文件名片段。

    Raises:
        无。
    """

    collapsed = re.sub(r"\s+", "_", artifact_name.strip())
    normalized = re.sub(r"[^0-9A-Za-z_.-]+", "_", collapsed)
    return normalized.strip("._") or "artifact"


def _build_audit_artifact_payload(audit_decision: AuditDecision) -> dict[str, Any]:
    """构建审计产物 JSON 负载。

    Args:
        audit_decision: 审计结果。

    Returns:
        可直接序列化为 JSON 的字典。

    Raises:
        无。
    """

    return {
        "pass": audit_decision.passed,
        "class": audit_decision.category,
        "violations": [asdict(v) for v in audit_decision.violations],
        "notes": audit_decision.notes,
        "repair_contract": asdict(audit_decision.repair_contract),
        "raw": audit_decision.raw,
    }
