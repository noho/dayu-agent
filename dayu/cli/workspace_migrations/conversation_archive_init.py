"""conversation transcript → ConversationSessionArchive 原地迁移。

旧 conversation 落盘 schema 是 ``ConversationTranscript`` 的 JSON
（顶层有 ``turns`` key、无 ``runtime_transcript`` key）。``#118``
引入 ``ConversationSessionArchive`` 聚合根后，落盘 schema 升级为
``conversation_session_archive/v1``：

- ``runtime_transcript``：保留旧 transcript 全量字段，作为运行态送模子视图。
- ``history_archive``：从 ``runtime_transcript.turns`` 全量投影出
  ``user_text``/``assistant_text``/``created_at`` 等展示字段，
  ``assistant_reasoning`` 缺省为空字符串（老会话当时没有 reasoning，
  这与"老会话无可展示历史"的退化是两回事）。

本迁移按硬约束执行：

- **原地** 覆盖旧文件路径（``CONVERSATION_STORE_RELATIVE_DIR``），
  不换目录、不双写、不留兼容读取路径。
- **幂等**：识别到顶层已有 ``runtime_transcript`` 的新 schema 直接跳过。
- 单文件失败仅 stderr warning 跳过，不阻塞整体迁移。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dayu.host.conversation_session_archive import (
    ConversationHistoryArchive,
    ConversationHistoryTurnRecord,
    ConversationSessionArchive,
)
from dayu.host.conversation_store import ConversationTranscript
from dayu.workspace_paths import CONVERSATION_STORE_RELATIVE_DIR


_NEW_SCHEMA_KEY = "runtime_transcript"
_LEGACY_TURNS_KEY = "turns"


def migrate_conversation_archive_init(workspace_root: Path) -> int:
    """将旧 transcript JSON 全量包装成 ``ConversationSessionArchive``。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        实际被改写的文件数量。

    Raises:
        无：单文件失败仅 stderr warning 跳过；目录不存在则返回 0。
    """

    target_dir = workspace_root / CONVERSATION_STORE_RELATIVE_DIR
    if not target_dir.is_dir():
        return 0

    rewritten = 0
    for json_path in sorted(target_dir.glob("*.json")):
        if _migrate_single_file(json_path):
            rewritten += 1
    return rewritten


def _migrate_single_file(json_path: Path) -> bool:
    """迁移单个 transcript JSON 文件。

    Args:
        json_path: 目标文件绝对路径。

    Returns:
        True 表示实际重写了文件；False 表示已是新 schema 或异常跳过。

    Raises:
        无：所有异常都被吞掉并 stderr warning。
    """

    try:
        raw_text = json_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(
            f"⚠ 工作区迁移: 跳过 {json_path.name}（读取失败：{exc}）",
            file=sys.stderr,
        )
        return False

    try:
        payload: object = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(
            f"⚠ 工作区迁移: 跳过 {json_path.name}（JSON 解析失败：{exc}）",
            file=sys.stderr,
        )
        return False
    if not isinstance(payload, dict):
        print(
            f"⚠ 工作区迁移: 跳过 {json_path.name}（顶层不是对象）",
            file=sys.stderr,
        )
        return False

    if _NEW_SCHEMA_KEY in payload:
        return False
    if _LEGACY_TURNS_KEY not in payload:
        return False

    try:
        runtime_transcript = ConversationTranscript.from_dict(payload)
    except (ValueError, KeyError, TypeError) as exc:
        print(
            f"⚠ 工作区迁移: 跳过 {json_path.name}（旧 transcript 反序列化失败：{exc}）",
            file=sys.stderr,
        )
        return False

    history_records = tuple(
        ConversationHistoryTurnRecord(
            turn_id=turn.turn_id,
            scene_name=turn.scene_name,
            user_text=turn.user_text,
            assistant_text=turn.assistant_final,
            assistant_reasoning="",
            created_at=turn.created_at,
        )
        for turn in runtime_transcript.turns
    )
    history_archive = ConversationHistoryArchive(
        session_id=runtime_transcript.session_id,
        turns=history_records,
    )

    archive = ConversationSessionArchive(
        session_id=runtime_transcript.session_id,
        revision=runtime_transcript.revision,
        created_at=runtime_transcript.created_at,
        updated_at=runtime_transcript.created_at,
        runtime_transcript=runtime_transcript,
        history_archive=history_archive,
    )

    new_text = json.dumps(archive.to_dict(), ensure_ascii=False, indent=2) + "\n"
    try:
        json_path.write_text(new_text, encoding="utf-8")
    except OSError as exc:
        print(
            f"⚠ 工作区迁移: 跳过 {json_path.name}（写回失败：{exc}）",
            file=sys.stderr,
        )
        return False
    return True


__all__ = ["migrate_conversation_archive_init"]
