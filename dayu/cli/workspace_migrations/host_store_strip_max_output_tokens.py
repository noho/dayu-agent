"""Host SQLite 迁移：从 pending turn 快照中剥离 ``max_output_tokens`` 字段。

2026-04 架构调整把 ``max_output_tokens`` 从 ``AgentCreateArgs`` 与
``AgentRunningConfigSnapshot`` 两层契约中彻底删除（该字段从未被 runner
payload 消费，属于历史噪声）。项目规则"schema 变更一律按全新 schema 起库"
决定了运行时代码不做旧字段兼容读取；但旧工作区里已有的
``.dayu/host/dayu_host.db`` 需要一次性把 ``resume_source_json`` 中残留的
``max_output_tokens`` 键原地 pop 掉，否则重启后旧快照反序列化会因多余键
或 schema 不匹配而失败。

本迁移只动 ``pending_conversation_turns.resume_source_json`` 这一列，
逐行读取、递归 pop 指定键、写回；不动 schema、不动其他表。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


_STRIP_KEY = "max_output_tokens"
# 作为 JSON 文本里的"键"出现时必然带双引号；加引号短路才能避免误匹配。
_STRIP_KEY_JSON_TOKEN = f'"{_STRIP_KEY}"'
_TABLE_NAME = "pending_conversation_turns"
_JSON_COLUMN = "resume_source_json"
_ID_COLUMN = "pending_turn_id"


def migrate_host_store_strip_max_output_tokens(host_db_path: Path) -> int:
    """把 Host SQLite pending turn 快照里的 ``max_output_tokens`` 键剥离。

    Args:
        host_db_path: Host SQLite 数据库文件路径，
            通常由 :func:`dayu.workspace_paths.build_host_store_default_path` 解析。

    Returns:
        实际被改写的行数；数据库不存在、表不存在或没有该 key 时返回 0。

    Raises:
        无：打开失败、SQL 异常由本函数吞掉并返回 0，避免阻塞 init 流程。
    """

    if not host_db_path.exists():
        return 0

    try:
        conn = sqlite3.connect(str(host_db_path))
    except sqlite3.Error:
        return 0

    try:
        conn.row_factory = sqlite3.Row
        if not _table_exists(conn, _TABLE_NAME):
            return 0

        rewritten = 0
        rows = conn.execute(
            f"SELECT {_ID_COLUMN}, {_JSON_COLUMN} FROM {_TABLE_NAME}"  # noqa: S608
        ).fetchall()
        for row in rows:
            raw = row[_JSON_COLUMN]
            if not isinstance(raw, str) or not raw:
                continue
            if _STRIP_KEY_JSON_TOKEN not in raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not _strip_key_in_place(payload):
                continue
            new_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            conn.execute(
                f"UPDATE {_TABLE_NAME} SET {_JSON_COLUMN} = ? WHERE {_ID_COLUMN} = ?",  # noqa: S608
                (new_text, row[_ID_COLUMN]),
            )
            rewritten += 1
        if rewritten:
            conn.commit()
        return rewritten
    except sqlite3.Error:
        try:
            conn.rollback()
        except sqlite3.Error:
            pass
        return 0
    finally:
        conn.close()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """判断指定表是否存在。"""

    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _strip_key_in_place(node: Any) -> bool:
    """递归地从 dict 里 pop 掉 ``max_output_tokens`` 键。

    Args:
        node: 任意 JSON 解析后的 Python 对象。

    Returns:
        True 表示至少改写了一处。

    Raises:
        无。
    """

    changed = False
    if isinstance(node, dict):
        if _STRIP_KEY in node:
            node.pop(_STRIP_KEY)
            changed = True
        for value in node.values():
            if _strip_key_in_place(value):
                changed = True
    elif isinstance(node, list):
        for item in node:
            if _strip_key_in_place(item):
                changed = True
    return changed
