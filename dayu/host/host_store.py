"""宿主层 SQLite 存储基础设施。

管理宿主 SQLite 数据库文件的连接与 schema 初始化。
所有宿主级 registry/governor 的 SQLite 操作都通过 HostStore 获取连接。

设计要点：
- SQLite WAL 模式，支持并发读 + 单写
- per-thread connection（threading.local），避免跨线程共享连接
- 所有表使用 CREATE TABLE IF NOT EXISTS，schema 初始化幂等
- owner_pid 字段用于死进程清理
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterable

from dayu.log import Log

MODULE = "HOST.STORE"

# ── Schema 定义 ──────────────────────────────────────────────────────

_SCHEMA_SQL = """
-- sessions 表：宿主级会话元数据索引
CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,
    source           TEXT NOT NULL,
    state            TEXT NOT NULL DEFAULT 'active',
    scene_name       TEXT,
    created_at       TEXT NOT NULL,
    last_activity_at TEXT NOT NULL,
    metadata_json    TEXT NOT NULL DEFAULT '{}'
);

-- runs 表：Agent 运行记录
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    session_id    TEXT,
    service_type  TEXT NOT NULL,
    scene_name    TEXT,
    state         TEXT NOT NULL DEFAULT 'created',
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    completed_at  TEXT,
    error_summary TEXT,
    cancel_requested_at TEXT,
    cancel_requested_reason TEXT,
    cancel_reason TEXT,
    owner_pid     INTEGER NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_runs_session ON runs(session_id);
CREATE INDEX IF NOT EXISTS idx_runs_state   ON runs(state);

-- pending_conversation_turns 表：resume V1 真源
CREATE TABLE IF NOT EXISTS pending_conversation_turns (
    pending_turn_id TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    scene_name      TEXT NOT NULL,
    user_text       TEXT NOT NULL,
    source_run_id   TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    resumable       INTEGER NOT NULL,
    state           TEXT NOT NULL,
    resume_source_json TEXT NOT NULL DEFAULT '',
    resume_attempt_count INTEGER NOT NULL DEFAULT 0,
    last_resume_error_message TEXT,
    pre_resume_state TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_pending_turns_session_scene
    ON pending_conversation_turns(session_id, scene_name);
CREATE INDEX IF NOT EXISTS idx_pending_turns_scene_state
    ON pending_conversation_turns(scene_name, state, updated_at);
CREATE INDEX IF NOT EXISTS idx_pending_turns_source_run
    ON pending_conversation_turns(source_run_id);

-- reply_outbox 表：出站交付真源
CREATE TABLE IF NOT EXISTS reply_outbox (
    delivery_id TEXT PRIMARY KEY,
    delivery_key TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    scene_name TEXT NOT NULL,
    source_run_id TEXT NOT NULL,
    reply_content TEXT NOT NULL,
    state TEXT NOT NULL,
    delivery_attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_reply_outbox_session_scene
    ON reply_outbox(session_id, scene_name);
CREATE INDEX IF NOT EXISTS idx_reply_outbox_state_updated
    ON reply_outbox(state, updated_at);
CREATE INDEX IF NOT EXISTS idx_reply_outbox_source_run
    ON reply_outbox(source_run_id);

-- permits 表：并发治理许可
CREATE TABLE IF NOT EXISTS permits (
    permit_id   TEXT PRIMARY KEY,
    lane        TEXT NOT NULL,
    owner_pid   INTEGER NOT NULL,
    acquired_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_permits_lane ON permits(lane);
"""


def _build_connection_debug_message(*, db_path: Path, conn: sqlite3.Connection) -> str:
    """构造 HostStore 首次建连调试日志。

    Args:
        db_path: SQLite 数据库路径。
        conn: 当前线程首次创建的连接对象。

    Returns:
        统一格式的调试日志文本。

    Raises:
        无。
    """

    current_thread = threading.current_thread()
    thread_ident = current_thread.ident
    thread_name = current_thread.name
    return (
        "HostStore 创建线程连接: "
        f"db_path={db_path}, "
        f"thread_ident={thread_ident if thread_ident is not None else 'unknown'}, "
        f"thread_name={thread_name}, conn_id=0x{id(conn):x}"
    )


class HostStore:
    """宿主层 SQLite 存储。线程安全，支持多进程并发访问。

    每个线程维护独立的 SQLite 连接，避免跨线程共享连接引发的问题。
    数据库使用 WAL 模式以获得更好的并发读写性能。

    Attributes:
        db_path: 数据库文件路径。
    """

    def __init__(self, db_path: Path) -> None:
        """初始化 HostStore。

        Args:
            db_path: SQLite 数据库文件路径（如 workspace/.dayu/host/dayu_host.db）。
        """
        self._db_path = db_path
        self._local = threading.local()
        self._closed = False

        # 确保目录存在
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        """数据库文件路径。"""
        return self._db_path

    def get_connection(self) -> sqlite3.Connection:
        """获取当前线程的 SQLite 连接。

        每个线程首次调用时创建新连接；后续复用同一连接。
        连接启用 WAL 模式和外键约束。

        Returns:
            当前线程的 SQLite 连接。

        Raises:
            RuntimeError: HostStore 已关闭。
        """
        if self._closed:
            raise RuntimeError("HostStore 已关闭")

        conn = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        conn = _create_connection(self._db_path)
        Log.debug(
            _build_connection_debug_message(db_path=self._db_path, conn=conn),
            module=MODULE,
        )
        self._local.conn = conn
        return conn

    def initialize_schema(self) -> None:
        """初始化数据库 schema。

        幂等操作：所有表和索引使用 IF NOT EXISTS。
        应在应用启动时调用一次。
        """
        conn = self.get_connection()
        conn.executescript(_SCHEMA_SQL)
        _ensure_runs_cancel_intent_columns(conn)
        _require_table_columns(
            conn,
            table_name="pending_conversation_turns",
            required_columns=(
                "pending_turn_id",
                "session_id",
                "scene_name",
                "user_text",
                "source_run_id",
                "created_at",
                "updated_at",
                "resumable",
                "state",
                "resume_source_json",
                "resume_attempt_count",
                "last_resume_error_message",
                "metadata_json",
            ),
            db_path=self._db_path,
        )
        Log.verbose(f"HostStore schema 初始化完成: {self._db_path}", module=MODULE)

    def close(self) -> None:
        """关闭 HostStore。

        关闭当前线程持有的连接。其他线程的连接将在各自线程结束时由 GC 清理。
        调用后 get_connection() 将抛出 RuntimeError。
        """
        self._closed = True
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None


def _create_connection(db_path: Path) -> sqlite3.Connection:
    """创建一个配置好的 SQLite 连接。

    Args:
        db_path: 数据库文件路径。

    Returns:
        配置好 WAL 模式和外键约束的连接。
    """
    conn = sqlite3.connect(
        str(db_path),
        timeout=10.0,  # 写锁等待超时
        check_same_thread=False,  # per-thread 模式下由 threading.local 保证线程安全
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")  # 写冲突重试等待 5 秒
    return conn


def _ensure_runs_cancel_intent_columns(conn: sqlite3.Connection) -> None:
    """幂等确保 runs 表存在取消相关列。"""

    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(runs)").fetchall()
    }
    if "cancel_requested_at" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN cancel_requested_at TEXT")
    if "cancel_requested_reason" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN cancel_requested_reason TEXT")
    if "cancel_reason" not in columns:
        conn.execute("ALTER TABLE runs ADD COLUMN cancel_reason TEXT")
    conn.commit()


def _require_table_columns(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    required_columns: Iterable[str],
    db_path: Path,
) -> None:
    """校验既有表结构是否满足当前 schema 要求。

    Args:
        conn: SQLite 连接。
        table_name: 待校验的表名。
        required_columns: 当前实现要求存在的列名集合。
        db_path: 当前 HostStore 对应的数据库文件路径。

    Returns:
        无。

    Raises:
        RuntimeError: 既有表缺少当前 schema 所需列时抛出，提示删除旧库重建。
    """

    actual_columns = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()  # noqa: S608
    }
    missing_columns = sorted({str(name) for name in required_columns} - actual_columns)
    if not missing_columns:
        return
    missing_columns_text = ", ".join(missing_columns)
    raise RuntimeError(
        "HostStore 检测到旧版 SQLite schema 与当前实现不兼容: "
        f"table={table_name}, missing_columns=[{missing_columns_text}], db_path={db_path}. "
        "当前项目默认不做数据库 schema 升级兼容，请删除该数据库后重建。"
    )
