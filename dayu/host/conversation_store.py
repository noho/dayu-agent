"""会话 transcript 数据类与 archive store 抽象 / 默认实现。

本模块同时承载三类职责：

1. ``ConversationTranscript`` / ``ConversationTurnRecord`` /
   ``ConversationPinnedState`` / ``ConversationEpisodeSummary``：运行态送模真源
   的数据结构。
2. ``ConversationSessionArchiveStore`` Protocol：archive 聚合根的存储抽象。
3. ``FileConversationSessionArchiveStore``：基于文件系统的默认实现，封装文件
   锁、原子写、fsync 等 IO 工具。

把 store 抽象与运行态数据类放在同一模块，是因为
``ConversationSessionArchive``（定义于 ``conversation_session_archive``）依赖
``ConversationTranscript``；如果再让 store 反过来依赖 archive 模块，会形成
循环导入。当前组织把 transcript 数据类与 store 抽象同住、archive 聚合根
单独成模块，是务实的边界划分。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Protocol, TextIO

if TYPE_CHECKING:
    from dayu.host.conversation_session_archive import (
        ConversationHistoryTurnRecord,
        ConversationSessionArchive,
    )

import dayu.file_lock as file_lock_module

from dayu.log import Log
from dayu.host._coercion import (
    _coerce_string_tuple,
    _normalize_session_id,
    _utc_now_iso,
)

_SESSION_FILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
MODULE = "HOST.CONVERSATION_STORE"
_CONVERSATION_LOCK_REGION_BYTES = 1
_LOCK_FILE_SUFFIX = ".lock"


def _coerce_non_negative_int(value: object, *, default: int = 0) -> int:
    """将对象规范化为非负整数。"""

    if value is None:
        return max(0, default)
    if isinstance(value, bool):
        return max(0, int(value))
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        return max(0, int(value))
    return max(0, default)


def _normalize_session_file_name(session_id: str) -> str:
    """将 session_id 规范化为安全文件名。

    Args:
        session_id: 原始会话 ID。

    Returns:
        可安全映射为 transcript 文件名的会话 ID。

    Raises:
        ValueError: 当会话 ID 为空或包含不安全字符时抛出。
    """

    normalized = _normalize_session_id(session_id)
    if not _SESSION_FILE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "session_id 只能包含字母、数字、点、下划线和中划线，且必须以字母或数字开头"
        )
    return normalized


def _serialize_transcript(transcript: "ConversationTranscript") -> dict[str, object]:
    """将 transcript 序列化为 JSON 对象。"""

    return {
        "session_id": transcript.session_id,
        "revision": transcript.revision,
        "created_at": transcript.created_at,
        "updated_at": transcript.updated_at,
        "last_scene_name": transcript.last_scene_name,
        "compacted_turn_count": transcript.compacted_turn_count,
        "pinned_state": asdict(transcript.pinned_state),
        "episodes": [asdict(item) for item in transcript.episodes],
        "turns": [asdict(turn) for turn in transcript.turns],
    }


def _fsync_parent_directory(path: Path) -> None:
    """尽力 fsync 父目录，降低原子替换后的目录元数据丢失风险。

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
    """通过临时文件 + 原子替换写入文本。

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


def _acquire_transcript_stream_lock(stream: TextIO) -> None:
    """获取 transcript 锁文件的跨进程排他锁。

    Args:
        stream: 已打开的锁文件流。

    Returns:
        无。

    Raises:
        OSError: 底层加锁失败时抛出。
    """

    file_lock_module.acquire_text_file_lock(
        stream,
        blocking=True,
        region_bytes=_CONVERSATION_LOCK_REGION_BYTES,
        lock_name="conversation transcript 文件锁",
    )


def _release_transcript_stream_lock(stream: TextIO) -> None:
    """释放 transcript 锁文件的跨进程排他锁。

    Args:
        stream: 已打开且已持锁的锁文件流。

    Returns:
        无。

    Raises:
        OSError: 底层解锁失败时抛出。
    """

    file_lock_module.release_text_file_lock(
        stream,
        region_bytes=_CONVERSATION_LOCK_REGION_BYTES,
        lock_name="conversation transcript 文件锁",
    )


@dataclass(frozen=True)
class ConversationToolUseSummary:
    """会话 turn 中的工具调用摘要。"""

    name: str
    arguments: dict[str, object] = field(default_factory=dict)
    result_summary: str = ""


@dataclass(frozen=True)
class ConversationTurnRecord:
    """规范化 transcript 的单轮记录。"""

    turn_id: str
    scene_name: str
    user_text: str
    assistant_final: str
    assistant_degraded: bool = False
    tool_uses: tuple[ConversationToolUseSummary, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=_utc_now_iso)


@dataclass(frozen=True)
class ConversationPinnedState:
    """会话中不可压缩的最小状态槽。"""

    current_goal: str = ""
    confirmed_subjects: tuple[str, ...] = field(default_factory=tuple)
    user_constraints: tuple[str, ...] = field(default_factory=tuple)
    open_questions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ConversationEpisodeSummary:
    """会话阶段摘要。"""

    episode_id: str
    start_turn_id: str
    end_turn_id: str
    title: str
    goal: str = ""
    completed_actions: tuple[str, ...] = field(default_factory=tuple)
    confirmed_facts: tuple[str, ...] = field(default_factory=tuple)
    user_constraints: tuple[str, ...] = field(default_factory=tuple)
    open_questions: tuple[str, ...] = field(default_factory=tuple)
    next_step: str = ""
    tool_findings: tuple[str, ...] = field(default_factory=tuple)
    created_at: str = field(default_factory=_utc_now_iso)


@dataclass(frozen=True)
class ConversationTranscript:
    """通用多轮会话 transcript。"""

    session_id: str
    revision: str
    created_at: str
    updated_at: str
    last_scene_name: str = ""
    compacted_turn_count: int = 0
    pinned_state: ConversationPinnedState = field(default_factory=ConversationPinnedState)
    episodes: tuple[ConversationEpisodeSummary, ...] = field(default_factory=tuple)
    turns: tuple[ConversationTurnRecord, ...] = field(default_factory=tuple)

    @classmethod
    def create_empty(cls, session_id: str) -> "ConversationTranscript":
        """创建空 transcript。

        Args:
            session_id: 会话 ID。

        Returns:
            空 transcript。

        Raises:
            ValueError: 当会话 ID 为空时抛出。
        """

        normalized_session_id = _normalize_session_id(session_id)
        now = _utc_now_iso()
        return cls(
            session_id=normalized_session_id,
            revision=uuid.uuid4().hex,
            created_at=now,
            updated_at=now,
            last_scene_name="",
            compacted_turn_count=0,
            pinned_state=ConversationPinnedState(),
            episodes=(),
            turns=(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ConversationTranscript":
        """从 JSON 对象反序列化 transcript。

        Args:
            data: 原始 JSON 对象。

        Returns:
            transcript 实例。

        Raises:
            ValueError: 当核心字段非法时抛出。
        """

        session_id = _normalize_session_id(str(data.get("session_id") or ""))
        revision = str(data.get("revision") or "").strip() or uuid.uuid4().hex
        created_at = str(data.get("created_at") or "").strip() or _utc_now_iso()
        updated_at = str(data.get("updated_at") or "").strip() or created_at
        last_scene_name = str(data.get("last_scene_name") or "").strip()
        compacted_turn_count = _coerce_non_negative_int(data.get("compacted_turn_count"), default=0)
        pinned_state = _parse_pinned_state(data.get("pinned_state"))
        episodes = _parse_episode_list(data.get("episodes"))
        turns = _parse_turn_list(data.get("turns"))
        return cls(
            session_id=session_id,
            revision=revision,
            created_at=created_at,
            updated_at=updated_at,
            last_scene_name=last_scene_name,
            compacted_turn_count=min(compacted_turn_count, len(turns)),
            pinned_state=pinned_state,
            episodes=episodes,
            turns=turns,
        )

    def append_turn(self, turn: ConversationTurnRecord) -> "ConversationTranscript":
        """追加单轮记录并生成新 revision。

        Args:
            turn: 新增 turn。

        Returns:
            追加后的 transcript。

        Raises:
            无。
        """

        return ConversationTranscript(
            session_id=self.session_id,
            revision=uuid.uuid4().hex,
            created_at=self.created_at,
            updated_at=_utc_now_iso(),
            last_scene_name=turn.scene_name,
            compacted_turn_count=self.compacted_turn_count,
            pinned_state=self.pinned_state,
            episodes=self.episodes,
            turns=(*self.turns, turn),
        )

    def replace_memory(
        self,
        *,
        pinned_state: ConversationPinnedState,
        episodes: tuple[ConversationEpisodeSummary, ...],
        compacted_turn_count: int,
    ) -> "ConversationTranscript":
        """返回替换 derived memory 后的新 transcript。

        Args:
            pinned_state: 新的 pinned state。
            episodes: 新的 episode 列表。
            compacted_turn_count: 新的已压缩 turn 数量。

        Returns:
            更新后的 transcript。

        Raises:
            无。
        """

        return ConversationTranscript(
            session_id=self.session_id,
            revision=uuid.uuid4().hex,
            created_at=self.created_at,
            updated_at=_utc_now_iso(),
            last_scene_name=self.last_scene_name,
            compacted_turn_count=max(0, min(compacted_turn_count, len(self.turns))),
            pinned_state=pinned_state,
            episodes=episodes,
            turns=self.turns,
        )


def _parse_pinned_state(raw: object) -> ConversationPinnedState:
    """解析 pinned state。

    Args:
        raw: 原始对象。

    Returns:
        解析后的 pinned state。

    Raises:
        无。
    """

    if not isinstance(raw, dict):
        return ConversationPinnedState()
    return ConversationPinnedState(
        current_goal=str(raw.get("current_goal") or "").strip(),
        confirmed_subjects=_coerce_string_tuple(raw.get("confirmed_subjects")),
        user_constraints=_coerce_string_tuple(raw.get("user_constraints")),
        open_questions=_coerce_string_tuple(raw.get("open_questions")),
    )


def _parse_episode_list(raw: object) -> tuple[ConversationEpisodeSummary, ...]:
    """解析 episode 列表。

    Args:
        raw: 原始对象。

    Returns:
        episode 元组。

    Raises:
        无。
    """

    if not isinstance(raw, list):
        return ()
    episodes: list[ConversationEpisodeSummary] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        episodes.append(
            ConversationEpisodeSummary(
                episode_id=str(item.get("episode_id") or "").strip() or uuid.uuid4().hex,
                start_turn_id=str(item.get("start_turn_id") or "").strip(),
                end_turn_id=str(item.get("end_turn_id") or "").strip(),
                title=str(item.get("title") or "").strip(),
                goal=str(item.get("goal") or "").strip(),
                completed_actions=_coerce_string_tuple(item.get("completed_actions")),
                confirmed_facts=_coerce_string_tuple(item.get("confirmed_facts")),
                user_constraints=_coerce_string_tuple(item.get("user_constraints")),
                open_questions=_coerce_string_tuple(item.get("open_questions")),
                next_step=str(item.get("next_step") or "").strip(),
                tool_findings=_coerce_string_tuple(item.get("tool_findings")),
                created_at=str(item.get("created_at") or "").strip() or _utc_now_iso(),
            )
        )
    return tuple(episodes)


def _parse_turn_list(raw: object) -> tuple[ConversationTurnRecord, ...]:
    """解析 turn 列表。

    Args:
        raw: 原始对象。

    Returns:
        turn 元组。

    Raises:
        无。
    """

    if not isinstance(raw, list):
        return ()
    turns: list[ConversationTurnRecord] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_tool_uses = item.get("tool_uses") or []
        tool_uses: list[ConversationToolUseSummary] = []
        if isinstance(raw_tool_uses, list):
            for raw_tool_use in raw_tool_uses:
                if not isinstance(raw_tool_use, dict):
                    continue
                tool_uses.append(
                    ConversationToolUseSummary(
                        name=str(raw_tool_use.get("name") or "").strip(),
                        arguments=dict(raw_tool_use.get("arguments") or {}),
                        result_summary=str(raw_tool_use.get("result_summary") or ""),
                    )
                )
        turns.append(
            ConversationTurnRecord(
                turn_id=str(item.get("turn_id") or "").strip() or uuid.uuid4().hex,
                scene_name=str(item.get("scene_name") or "").strip(),
                user_text=str(item.get("user_text") or ""),
                assistant_final=str(item.get("assistant_final") or ""),
                assistant_degraded=bool(item.get("assistant_degraded", False)),
                tool_uses=tuple(tool_uses),
                warnings=tuple(str(message) for message in (item.get("warnings") or [])),
                errors=tuple(str(message) for message in (item.get("errors") or [])),
                created_at=str(item.get("created_at") or "").strip() or _utc_now_iso(),
            )
        )
    return tuple(turns)


class ConversationSessionArchiveStore(Protocol):
    """会话 archive 存储协议。

    存储对象升级为 ``ConversationSessionArchive`` 聚合根；``runtime_transcript``
    与 ``history_archive`` 子视图通过聚合根在单文件下原子提交。
    """

    def load(self, session_id: str) -> "ConversationSessionArchive | None":
        """读取指定 session 的 archive。

        Args:
            session_id: 会话 ID。

        Returns:
            archive；不存在时返回 ``None``。

        Raises:
            ValueError: session_id 非法时抛出。
            RuntimeError: 磁盘内容是旧 transcript schema 时抛出。
        """
        ...

    def save(
        self,
        archive: "ConversationSessionArchive",
        *,
        expected_revision: str | None = None,
    ) -> "ConversationSessionArchive":
        """保存 archive。

        Args:
            archive: 待保存 archive。
            expected_revision: 预期旧 archive revision；用于乐观锁控制。

        Returns:
            实际保存后的 archive。

        Raises:
            RuntimeError: revision 冲突时抛出。
        """
        ...

    def load_or_create(self, session_id: str) -> "ConversationSessionArchive":
        """读取 archive；若磁盘缺失则在锁内原子创建空 archive。

        ``prepare`` 首轮装配真源专用：消除 ``load -> save(expected_revision=None)``
        之间的 TOCTOU 窗口，避免并发 prepare 把另一进程已经写好的 live archive
        覆盖成空文件。

        语义：取 session 文件锁 -> ``load`` live -> 若 ``None`` 则**在锁内**
        原子写入 ``ConversationSessionArchive.create_empty(session_id)`` -> 返回
        live archive 或新建 archive。任何并发写入只会让先到的胜出，后到的拿到
        live archive 而不会回写空版本。

        Args:
            session_id: 会话 ID。

        Returns:
            读到或新建的 archive。

        Raises:
            ValueError: session_id 非法时抛出。
            RuntimeError: 磁盘内容是旧 transcript schema 时抛出。
        """
        ...

    def append_turn(
        self,
        session_id: str,
        *,
        turn_record: ConversationTurnRecord,
        history_record: "ConversationHistoryTurnRecord",
    ) -> "ConversationSessionArchive":
        """resume 落盘专用：在 live archive 上 reconcile 写入新一轮。

        语义：取 session 文件锁 -> ``load`` live -> 若 ``None`` 则抛
        ``ConversationArchiveMissingError`` -> 在 live 上 ``with_next_turn`` 推进 ->
        单文件原子写。**不**预设"自动 create_empty"策略。

        Args:
            session_id: 会话 ID。
            turn_record: 新增运行态 turn。
            history_record: 与之一一对应的历史展示记录。

        Returns:
            落盘后的 archive。

        Raises:
            ConversationArchiveMissingError: live archive 缺失时抛出。
            ValueError: 当 ``turn_id`` 不一致时抛出。
        """
        ...

    def delete(self, session_id: str) -> bool:
        """删除指定 session 的 archive。

        Args:
            session_id: 会话 ID。

        Returns:
            archive 文件存在并被删除返回 ``True``，否则 ``False``。

        Raises:
            ValueError: session_id 非法时抛出。
        """
        ...


class FileConversationSessionArchiveStore:
    """基于文件系统的 archive 存储。"""

    def __init__(self, root_dir: Path) -> None:
        """初始化存储实现。

        Args:
            root_dir: archive 根目录。

        Returns:
            无。

        Raises:
            无。
        """

        self._root_dir = root_dir.resolve()
        self._root_dir.mkdir(parents=True, exist_ok=True)
        self._lock_dir = self._root_dir / ".locks"
        self._lock_dir.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str) -> "ConversationSessionArchive | None":
        """读取指定 session 的 archive。

        Args:
            session_id: 会话 ID。

        Returns:
            archive；文件不存在时返回 ``None``。

        Raises:
            ValueError: session_id 为空时抛出。
            RuntimeError: 磁盘内容是旧 transcript schema 时抛出。
        """

        from dayu.host.conversation_session_archive import ConversationSessionArchive

        file_path = self._resolve_file_path(session_id)
        if not file_path.exists():
            Log.debug(f"archive 不存在: session_id={session_id}", module=MODULE)
            return None
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"conversation session archive 非法: {file_path}")
        if "runtime_transcript" not in payload and "turns" in payload:
            raise RuntimeError(
                "conversation transcript 已升级为 archive schema，请运行 dayu-cli init 完成迁移："
                f"session_id={session_id}, file={file_path}"
            )
        Log.debug(f"加载 archive: session_id={session_id}, file={file_path.name}", module=MODULE)
        return ConversationSessionArchive.from_dict(payload)

    def save(
        self,
        archive: "ConversationSessionArchive",
        *,
        expected_revision: str | None = None,
    ) -> "ConversationSessionArchive":
        """保存 archive。

        Args:
            archive: 待保存 archive。
            expected_revision: 预期旧 archive revision；用于乐观锁控制。

        Returns:
            实际保存后的 archive。

        Raises:
            RuntimeError: revision 冲突时抛出。
        """

        file_path = self._resolve_file_path(archive.session_id)
        with self._transcript_file_lock(archive.session_id):
            if file_path.exists():
                existing = self.load(archive.session_id)
                if (
                    existing is not None
                    and expected_revision is not None
                    and existing.revision != expected_revision
                ):
                    Log.warning(
                        "conversation session archive revision 冲突: "
                        f"session_id={archive.session_id}, expected={expected_revision}, actual={existing.revision}",
                        module=MODULE,
                    )
                    raise RuntimeError(
                        "conversation session archive revision 冲突："
                        f"session_id={archive.session_id}, expected={expected_revision}, actual={existing.revision}"
                    )
            # 文件不存在时直接写入：首轮 archive 仅存在于内存，不构成并发冲突
            _atomic_write_text(
                file_path,
                json.dumps(archive.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        Log.debug(
            "保存 archive: "
            f"session_id={archive.session_id}, revision={archive.revision}, "
            f"turns={len(archive.runtime_transcript.turns)}, history_turns={len(archive.history_archive.turns)}",
            module=MODULE,
        )
        return archive

    def load_or_create(self, session_id: str) -> "ConversationSessionArchive":
        """读取 archive；磁盘缺失时在锁内原子创建空 archive。

        消除 ``prepare`` 首轮装配真源的 TOCTOU 窗口：``load`` 与 ``save`` 之间
        若另一进程已经写好真实 archive，普通的
        ``save(create_empty, expected_revision=None)`` 会把对方刚落下的历史
        覆盖成空文件。本方法在文件锁保护下做 load-or-create，先到的胜出，
        后到的拿到 live archive。

        Args:
            session_id: 会话 ID。

        Returns:
            读到或新建的 archive。

        Raises:
            ValueError: session_id 非法时抛出。
            RuntimeError: 磁盘内容是旧 transcript schema 时抛出。
        """

        from dayu.host.conversation_session_archive import ConversationSessionArchive

        normalized_session_id = _normalize_session_id(session_id)
        file_path = self._resolve_file_path(normalized_session_id)
        with self._transcript_file_lock(normalized_session_id):
            existing = self.load(normalized_session_id)
            if existing is not None:
                return existing
            new_archive = ConversationSessionArchive.create_empty(normalized_session_id)
            _atomic_write_text(
                file_path,
                json.dumps(new_archive.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        Log.debug(
            f"创建空 archive: session_id={normalized_session_id}, revision={new_archive.revision}",
            module=MODULE,
        )
        return new_archive

    def append_turn(
        self,
        session_id: str,
        *,
        turn_record: ConversationTurnRecord,
        history_record: "ConversationHistoryTurnRecord",
    ) -> "ConversationSessionArchive":
        """resume 落盘专用：在 live archive 上 reconcile 写入新一轮。

        Args:
            session_id: 会话 ID。
            turn_record: 新增运行态 turn。
            history_record: 与之一一对应的历史展示记录。

        Returns:
            落盘后的 archive。

        Raises:
            ConversationArchiveMissingError: live archive 缺失时抛出。
            ValueError: 当 ``turn_id`` 不一致时抛出。
        """

        from dayu.host.conversation_session_archive import ConversationArchiveMissingError

        normalized_session_id = _normalize_session_id(session_id)
        file_path = self._resolve_file_path(normalized_session_id)
        with self._transcript_file_lock(normalized_session_id):
            live = self.load(normalized_session_id)
            if live is None:
                raise ConversationArchiveMissingError(
                    "conversation session archive 缺失，无法 reconcile-on-write："
                    f"session_id={normalized_session_id}"
                )
            next_archive = live.with_next_turn(turn_record, history_record)
            _atomic_write_text(
                file_path,
                json.dumps(next_archive.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        Log.debug(
            "reconcile 写入 archive: "
            f"session_id={normalized_session_id}, revision={next_archive.revision}, "
            f"turns={len(next_archive.runtime_transcript.turns)}",
            module=MODULE,
        )
        return next_archive

    def delete(self, session_id: str) -> bool:
        """删除指定 session 的 archive。

        Args:
            session_id: 会话 ID。

        Returns:
            archive 文件存在并被删除返回 ``True``，否则 ``False``。

        Raises:
            ValueError: session_id 非法时抛出。
        """

        normalized_session_id = _normalize_session_id(session_id)
        file_path = self._resolve_file_path(normalized_session_id)
        with self._transcript_file_lock(normalized_session_id):
            if not file_path.exists():
                return False
            file_path.unlink()
            _fsync_parent_directory(file_path)
        Log.debug(f"删除 archive: session_id={normalized_session_id}", module=MODULE)
        return True

    def _resolve_file_path(self, session_id: str) -> Path:
        """解析指定 session 的文件路径。

        Args:
            session_id: 会话 ID。

        Returns:
            transcript 文件绝对路径。

        Raises:
            ValueError: 当会话 ID 非法时抛出。
        """

        normalized_session_id = _normalize_session_file_name(session_id)
        return self._root_dir / f"{normalized_session_id}.json"

    def _resolve_lock_file_path(self, session_id: str) -> Path:
        """解析指定 session 的锁文件路径。

        Args:
            session_id: 会话 ID。

        Returns:
            锁文件绝对路径。

        Raises:
            ValueError: 当会话 ID 非法时抛出。
        """

        normalized_session_id = _normalize_session_file_name(session_id)
        return self._lock_dir / f"{normalized_session_id}{_LOCK_FILE_SUFFIX}"

    @contextmanager
    def _transcript_file_lock(self, session_id: str) -> Iterator[TextIO]:
        """对单个 session transcript 的读改写流程加跨进程排他锁。

        Args:
            session_id: 会话 ID。

        Yields:
            已打开并持有排他锁的锁文件流。

        Raises:
            OSError: 锁文件打开或加锁失败时抛出。
        """

        lock_path = self._resolve_lock_file_path(session_id)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as stream:
            _acquire_transcript_stream_lock(stream)
            try:
                yield stream
            finally:
                _release_transcript_stream_lock(stream)


__all__ = [
    "ConversationEpisodeSummary",
    "ConversationPinnedState",
    "ConversationSessionArchiveStore",
    "ConversationToolUseSummary",
    "ConversationTranscript",
    "ConversationTurnRecord",
    "FileConversationSessionArchiveStore",
]
