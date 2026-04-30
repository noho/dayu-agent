"""会话存储聚合根 ``ConversationSessionArchive`` 与历史展示子视图。

本模块定义 Host 落盘的会话存储聚合根：

- ``runtime_transcript``：运行态送模视图，沿用 ``ConversationTranscript``，
  字段集合不变；conversation memory / compaction / 送模 prompt 装配
  仅消费此子视图。
- ``history_archive``：历史展示子视图，承载 ``assistant_reasoning`` 等
  仅供 UI 历史回放使用、**绝不**进入运行态决策的字段。

聚合根硬约束：

- 单聚合、单 revision、单文件原子提交。
- ``runtime_transcript`` 与 ``history_archive`` 必须在同一次写入下原子推进。
- ``with_*`` 在推进 ``archive.revision`` 时同步刷新
  ``runtime_transcript.revision``，保持两者一致以兼容现有读运行态字段访问；
  乐观锁参与方仍是 ``archive.revision``。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace

from dayu.host._coercion import _normalize_session_id, _utc_now_iso
from dayu.host.conversation_store import (
    ConversationTranscript,
    ConversationTurnRecord,
)
from dayu.log import Log

_MODULE = "HOST.CONVERSATION_SESSION_ARCHIVE"


class ConversationArchiveMissingError(RuntimeError):
    """live archive 缺失时由 archive store reconcile 路径抛出。

    显式语义：``append_turn`` 在 reconcile-on-write 路径下要求 live archive
    存在；缺失时不预设默认行为（例如自动 ``create_empty``），而是直接报错，
    把"清空 vs pending turn 并发"的处置策略留给上层（``#117``）决定。
    """


@dataclass(frozen=True)
class ConversationHistoryTurnRecord:
    """历史展示子视图中的单轮记录。

    Attributes:
        turn_id: 与同步写入的 ``ConversationTurnRecord.turn_id`` 一一对应。
        scene_name: 触发该轮的 scene 名称。
        user_text: 用户输入文本。
        assistant_text: 助手最终回复文本。
        assistant_reasoning: 助手 reasoning 文本，**仅展示**，禁止参与运行态。
        created_at: 创建时间 ISO 字符串。
    """

    turn_id: str
    scene_name: str
    user_text: str
    assistant_text: str
    assistant_reasoning: str
    created_at: str


@dataclass(frozen=True)
class ConversationHistoryArchive:
    """历史展示子视图聚合。

    Attributes:
        session_id: 会话 ID。
        turns: 按时间从旧到新排列的历史轮次。
    """

    session_id: str
    turns: tuple[ConversationHistoryTurnRecord, ...] = field(default_factory=tuple)

    @classmethod
    def create_empty(cls, session_id: str) -> "ConversationHistoryArchive":
        """创建空 history archive。

        Args:
            session_id: 会话 ID。

        Returns:
            空 history archive。

        Raises:
            ValueError: 当会话 ID 为空时抛出。
        """

        return cls(session_id=_normalize_session_id(session_id), turns=())

    def append_turn(self, record: ConversationHistoryTurnRecord) -> "ConversationHistoryArchive":
        """追加一条历史记录。

        Args:
            record: 新增历史记录。

        Returns:
            追加后的 history archive。

        Raises:
            无。
        """

        return ConversationHistoryArchive(
            session_id=self.session_id,
            turns=(*self.turns, record),
        )

    def to_dict(self) -> dict[str, object]:
        """序列化为 JSON 对象。

        Args:
            无。

        Returns:
            JSON 可序列化字典。

        Raises:
            无。
        """

        return {
            "session_id": self.session_id,
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "scene_name": turn.scene_name,
                    "user_text": turn.user_text,
                    "assistant_text": turn.assistant_text,
                    "assistant_reasoning": turn.assistant_reasoning,
                    "created_at": turn.created_at,
                }
                for turn in self.turns
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ConversationHistoryArchive":
        """从 JSON 对象反序列化。

        Args:
            data: 原始 JSON 对象。

        Returns:
            ``ConversationHistoryArchive`` 实例。

        Raises:
            ValueError: 当核心字段非法时抛出。
        """

        session_id = _normalize_session_id(str(data.get("session_id") or ""))
        raw_turns = data.get("turns")
        turns: list[ConversationHistoryTurnRecord] = []
        if isinstance(raw_turns, list):
            for raw_turn in raw_turns:
                if not isinstance(raw_turn, dict):
                    Log.warning(
                        f"history_archive 跳过非法 turn 元素（非对象）: session_id={session_id}",
                        module=_MODULE,
                    )
                    continue
                turns.append(
                    ConversationHistoryTurnRecord(
                        turn_id=str(raw_turn.get("turn_id") or "").strip() or uuid.uuid4().hex,
                        scene_name=str(raw_turn.get("scene_name") or "").strip(),
                        user_text=str(raw_turn.get("user_text") or ""),
                        assistant_text=str(raw_turn.get("assistant_text") or ""),
                        assistant_reasoning=str(raw_turn.get("assistant_reasoning") or ""),
                        created_at=str(raw_turn.get("created_at") or "").strip() or _utc_now_iso(),
                    )
                )
        return cls(session_id=session_id, turns=tuple(turns))


@dataclass(frozen=True)
class ConversationSessionArchive:
    """会话存储聚合根。

    单 revision、单文件原子提交；``runtime_transcript`` 与
    ``history_archive`` 任一推进都必须经过聚合根的 ``with_*`` 接口。

    Attributes:
        session_id: 会话 ID。
        revision: 聚合根 revision，作为乐观锁参与方。
        created_at: 创建时间 ISO 字符串。
        updated_at: 更新时间 ISO 字符串。
        runtime_transcript: 运行态送模视图。
        history_archive: 历史展示子视图。
    """

    session_id: str
    revision: str
    created_at: str
    updated_at: str
    runtime_transcript: ConversationTranscript
    history_archive: ConversationHistoryArchive

    @classmethod
    def create_empty(cls, session_id: str) -> "ConversationSessionArchive":
        """创建空聚合根。

        Args:
            session_id: 会话 ID。

        Returns:
            空 ``ConversationSessionArchive``。

        Raises:
            ValueError: 当会话 ID 为空时抛出。
        """

        normalized_session_id = _normalize_session_id(session_id)
        now = _utc_now_iso()
        archive_revision = uuid.uuid4().hex
        runtime_transcript = ConversationTranscript.create_empty(normalized_session_id)
        runtime_transcript = replace(runtime_transcript, revision=archive_revision)
        history_archive = ConversationHistoryArchive.create_empty(normalized_session_id)
        return cls(
            session_id=normalized_session_id,
            revision=archive_revision,
            created_at=now,
            updated_at=now,
            runtime_transcript=runtime_transcript,
            history_archive=history_archive,
        )

    def with_next_turn(
        self,
        turn_record: ConversationTurnRecord,
        history_record: ConversationHistoryTurnRecord,
    ) -> "ConversationSessionArchive":
        """同步推进运行态与历史展示子视图。

        Args:
            turn_record: 新增运行态 turn。
            history_record: 与之一一对应的历史展示记录；``turn_id`` 必须一致。

        Returns:
            推进后的聚合根。

        Raises:
            ValueError: 当 ``turn_id`` 不一致时抛出。
        """

        if turn_record.turn_id != history_record.turn_id:
            raise ValueError(
                "ConversationSessionArchive.with_next_turn 要求 turn_record 与 history_record "
                f"的 turn_id 一致：runtime={turn_record.turn_id!r}, history={history_record.turn_id!r}"
            )
        next_runtime_transcript = self.runtime_transcript.append_turn(turn_record)
        next_history_archive = self.history_archive.append_turn(history_record)
        return self._with_synchronized_revision(
            runtime_transcript=next_runtime_transcript,
            history_archive=next_history_archive,
        )

    def with_runtime_transcript(
        self,
        runtime_transcript: ConversationTranscript,
    ) -> "ConversationSessionArchive":
        """仅替换运行态子视图。

        compaction 写回路径专用：``history_archive`` 原样保留。

        Args:
            runtime_transcript: 新的运行态子视图。

        Returns:
            替换后的聚合根。

        Raises:
            无。
        """

        return self._with_synchronized_revision(
            runtime_transcript=runtime_transcript,
            history_archive=self.history_archive,
        )

    def _with_synchronized_revision(
        self,
        *,
        runtime_transcript: ConversationTranscript,
        history_archive: ConversationHistoryArchive,
    ) -> "ConversationSessionArchive":
        """推进聚合根 revision 并同步刷新 runtime_transcript.revision。

        Args:
            runtime_transcript: 新的运行态子视图。
            history_archive: 新的历史展示子视图。

        Returns:
            推进后的聚合根。

        Raises:
            无。
        """

        next_archive_revision = uuid.uuid4().hex
        synchronized_runtime_transcript = replace(runtime_transcript, revision=next_archive_revision)
        return ConversationSessionArchive(
            session_id=self.session_id,
            revision=next_archive_revision,
            created_at=self.created_at,
            updated_at=_utc_now_iso(),
            runtime_transcript=synchronized_runtime_transcript,
            history_archive=history_archive,
        )

    def to_dict(self) -> dict[str, object]:
        """序列化聚合根为 JSON 对象。

        Args:
            无。

        Returns:
            JSON 可序列化字典。

        Raises:
            无。
        """

        from dayu.host.conversation_store import _serialize_transcript  # 避免循环导入语义膨胀

        return {
            "schema": "conversation_session_archive/v1",
            "session_id": self.session_id,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "runtime_transcript": _serialize_transcript(self.runtime_transcript),
            "history_archive": self.history_archive.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ConversationSessionArchive":
        """从 JSON 对象反序列化聚合根。

        Args:
            data: 原始 JSON 对象。

        Returns:
            ``ConversationSessionArchive`` 实例。

        Raises:
            ValueError: 当核心字段非法或缺失运行态子视图时抛出。
        """

        if "runtime_transcript" not in data:
            raise ValueError(
                "conversation session archive 缺少 runtime_transcript 字段，"
                "可能仍是旧 schema，请运行 dayu-cli init 完成迁移"
            )
        session_id = _normalize_session_id(str(data.get("session_id") or ""))
        raw_runtime_transcript = data.get("runtime_transcript")
        if not isinstance(raw_runtime_transcript, dict):
            raise ValueError("conversation session archive runtime_transcript 必须是对象")
        runtime_transcript = ConversationTranscript.from_dict(raw_runtime_transcript)
        if "history_archive" not in data:
            raise ValueError(
                "conversation session archive 缺少 history_archive 字段；"
                "新 schema 要求显式承载历史展示子视图。请通过 dayu-cli init "
                "或 repair 流程修复，禁止静默降级为空历史以避免覆盖丢失 reasoning。"
            )
        raw_history_archive = data.get("history_archive")
        if not isinstance(raw_history_archive, dict):
            raise ValueError(
                "conversation session archive history_archive 必须是对象，"
                "禁止静默降级为空历史。"
            )
        history_archive = ConversationHistoryArchive.from_dict(raw_history_archive)
        revision = str(data.get("revision") or "").strip() or runtime_transcript.revision
        created_at = str(data.get("created_at") or "").strip() or runtime_transcript.created_at
        updated_at = str(data.get("updated_at") or "").strip() or created_at
        return cls(
            session_id=session_id,
            revision=revision,
            created_at=created_at,
            updated_at=updated_at,
            runtime_transcript=runtime_transcript,
            history_archive=history_archive,
        )


__all__ = [
    "ConversationArchiveMissingError",
    "ConversationHistoryArchive",
    "ConversationHistoryTurnRecord",
    "ConversationSessionArchive",
]
