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
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

from dayu.log import Log

MODULE = "HOST.STORE"
_SQLITE_LOCK_TIMEOUT_SEC = 30.0
_SQLITE_BUSY_TIMEOUT_MS = 30_000
_SQLITE_JOURNAL_MODE = "WAL"
_SQLITE_SYNCHRONOUS_MODE = "FULL"

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
    owner_process_start_time REAL,
    owner_boot_id TEXT,
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
    resume_lease_id TEXT,
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
    metadata_json TEXT NOT NULL DEFAULT '{}',
    lease_id TEXT
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
    owner_process_start_time REAL,
    owner_boot_id TEXT,
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
                "pre_resume_state",
                "resume_lease_id",
                "metadata_json",
            ),
            db_path=self._db_path,
        )
        _require_table_columns(
            conn,
            table_name="reply_outbox",
            required_columns=(
                "delivery_id",
                "delivery_key",
                "session_id",
                "scene_name",
                "source_run_id",
                "reply_content",
                "state",
                "delivery_attempt_count",
                "last_error_message",
                "created_at",
                "updated_at",
                "metadata_json",
                "lease_id",
            ),
            db_path=self._db_path,
        )
        _require_table_columns(
            conn,
            table_name="runs",
            required_columns=(
                "run_id",
                "session_id",
                "service_type",
                "scene_name",
                "state",
                "created_at",
                "started_at",
                "completed_at",
                "error_summary",
                "cancel_requested_at",
                "cancel_requested_reason",
                "cancel_reason",
                "owner_pid",
                "owner_process_start_time",
                "owner_boot_id",
                "metadata_json",
            ),
            db_path=self._db_path,
        )
        _require_table_columns(
            conn,
            table_name="permits",
            required_columns=(
                "permit_id",
                "lane",
                "owner_pid",
                "owner_process_start_time",
                "owner_boot_id",
                "acquired_at",
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
        配置好 WAL 模式、autocommit 与外键约束的连接。写事务由
        ``write_transaction`` 作为真源负责开启 / 提交 / 回滚。

    Raises:
        sqlite3.Error: SQLite 建连或 PRAGMA 配置失败时抛出。
    """
    conn = sqlite3.connect(
        str(db_path),
        timeout=_SQLITE_LOCK_TIMEOUT_SEC,
        check_same_thread=False,  # per-thread 模式下由 threading.local 保证线程安全
        isolation_level=None,  # autocommit；写事务统一经 write_transaction 显式开启
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA journal_mode={_SQLITE_JOURNAL_MODE}")
    # WAL + FULL：锁竞争的根因已经由 `write_transaction`(BEGIN IMMEDIATE) 承担，
    # 同步级别保持 FULL 以守住宿主真源（runs / pending_conversation_turns /
    # reply_outbox / sessions）在崩溃/掉电后的持久化语义，不借放松 fsync 来换并发。
    conn.execute(f"PRAGMA synchronous={_SQLITE_SYNCHRONOUS_MODE}")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}")
    return conn


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """在给定 SQLite 连接上开启一个显式写事务。

    作为 host 层所有写路径的唯一事务入口：通过 ``BEGIN IMMEDIATE`` 在
    事务开始时立刻获取 RESERVED 锁，使并发 writer 在入口排队而非在
    ``COMMIT`` 时相互冲撞，从根上避免 Python sqlite3 默认
    ``BEGIN DEFERRED`` 下的 upgrade deadlock（并发升级到 RESERVED
    失败会直接返回 ``SQLITE_BUSY`` 且不走 busy_handler 重试）。

    ``BEGIN IMMEDIATE`` / ``COMMIT`` 均可能失败（``SQLITE_BUSY`` / I/O 等）；
    因此事务体、``COMMIT`` 自身的任何异常都会触发 ``ROLLBACK`` 清理，
    避免 thread-local 共享连接被卡在"事务未收口"状态污染后续写入。
    ``ROLLBACK`` 本身若再失败，只静默吞掉，保留原始异常。

    用法示例::

        with write_transaction(conn):
            conn.execute("INSERT ...", params)

    Args:
        conn: 已配置为 autocommit（``isolation_level=None``）的连接。

    Yields:
        None。调用方在 ``with`` 块内执行任意写语句，块结束时自动提交。

    Raises:
        sqlite3.Error: ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK`` 本身
            抛出的 SQLite 异常（``COMMIT`` 失败时已经先触发 ``ROLLBACK``）。
        BaseException: 事务体中出现的任何异常将触发回滚后原样再抛出。
    """

    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - 回滚失败仅记录，不掩盖原始异常
            pass
        raise
    try:
        conn.execute("COMMIT")
    except BaseException:
        # COMMIT 自身失败（SQLITE_BUSY / I/O 等）时，连接仍处于事务中；
        # 必须显式 ROLLBACK 回到干净态，避免同线程后续写撞上 "cannot start
        # a transaction within a transaction"。
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - 清理失败不掩盖原始 COMMIT 异常
            pass
        raise


def _ensure_runs_cancel_intent_columns(conn: sqlite3.Connection) -> None:
    """幂等确保 runs 表存在取消相关列。"""

    columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(runs)").fetchall()
    }
    missing_ddl: list[str] = []
    if "cancel_requested_at" not in columns:
        missing_ddl.append("ALTER TABLE runs ADD COLUMN cancel_requested_at TEXT")
    if "cancel_requested_reason" not in columns:
        missing_ddl.append("ALTER TABLE runs ADD COLUMN cancel_requested_reason TEXT")
    if "cancel_reason" not in columns:
        missing_ddl.append("ALTER TABLE runs ADD COLUMN cancel_reason TEXT")
    if not missing_ddl:
        return
    with write_transaction(conn):
        for ddl in missing_ddl:
            conn.execute(ddl)


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
