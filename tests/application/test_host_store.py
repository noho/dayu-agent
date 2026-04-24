"""HostStore SQLite 存储基础设施测试。"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from dayu.host.host_store import HostStore, write_transaction
from dayu.log import Log


class TestSchemaInitialization:
    """Schema 初始化测试。"""

    @pytest.mark.unit
    def test_initialize_creates_tables(self, tmp_path: Path) -> None:
        """初始化后三张表都存在。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()

        conn = store.get_connection()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "sessions" in tables
        assert "runs" in tables
        assert "permits" in tables
        store.close()

    @pytest.mark.unit
    def test_initialize_is_idempotent(self, tmp_path: Path) -> None:
        """多次初始化不报错。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()
        store.initialize_schema()  # 第二次不应报错
        store.close()

    @pytest.mark.unit
    def test_initialize_logs_with_project_log(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """schema 初始化通过项目统一日志入口输出 verbose。"""
        store = HostStore(tmp_path / "test.db")
        verbose_mock = Mock()
        monkeypatch.setattr(Log, "verbose", verbose_mock)

        store.initialize_schema()

        verbose_mock.assert_called_once_with(
            f"HostStore schema 初始化完成: {tmp_path / 'test.db'}",
            module="HOST.STORE",
        )
        store.close()

    @pytest.mark.unit
    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """自动创建父目录。"""
        db_path = tmp_path / "sub" / "dir" / "test.db"
        store = HostStore(db_path)
        store.initialize_schema()
        assert db_path.exists()
        store.close()

    @pytest.mark.unit
    def test_initialize_rejects_legacy_pending_turn_table_without_resume_source_json(self, tmp_path: Path) -> None:
        """旧 pending turn 表缺少当前必需列时应在初始化阶段直接失败。"""

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE pending_conversation_turns (
                pending_turn_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                scene_name TEXT NOT NULL,
                user_text TEXT NOT NULL,
                source_run_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resumable INTEGER NOT NULL,
                state TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        conn.commit()
        conn.close()

        store = HostStore(db_path)

        with pytest.raises(RuntimeError, match="pending_conversation_turns"):
            store.initialize_schema()

        with pytest.raises(RuntimeError, match="resume_source_json"):
            store.initialize_schema()

        store.close()


class TestConnection:
    """连接管理测试。"""

    @pytest.mark.unit
    def test_get_connection_returns_same_per_thread(self, tmp_path: Path) -> None:
        """同一线程获取同一连接。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()

        conn1 = store.get_connection()
        conn2 = store.get_connection()
        assert conn1 is conn2
        store.close()

    @pytest.mark.unit
    def test_get_connection_logs_only_on_first_connection_creation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """首次建连会输出调试日志，后续同线程复用不重复输出。"""

        debug_mock = Mock()
        monkeypatch.setattr(Log, "debug", debug_mock)
        store = HostStore(tmp_path / "test.db")

        store.initialize_schema()
        store.get_connection()
        store.get_connection()

        debug_messages = [call.args[0] for call in debug_mock.call_args_list]
        assert len(debug_messages) == 1
        assert "HostStore 创建线程连接" in debug_messages[0]
        assert f"db_path={tmp_path / 'test.db'}" in debug_messages[0]
        assert "thread_ident=" in debug_messages[0]
        assert "conn_id=" in debug_messages[0]
        store.close()

    @pytest.mark.unit
    def test_get_connection_different_per_thread(self, tmp_path: Path) -> None:
        """不同线程获取不同连接。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()

        conns: list[sqlite3.Connection] = []

        def worker() -> None:
            conns.append(store.get_connection())

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=5)

        main_conn = store.get_connection()
        assert len(conns) == 1
        assert conns[0] is not main_conn
        store.close()

    @pytest.mark.unit
    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        """确认 WAL 模式已启用。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()

        conn = store.get_connection()
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        store.close()

    @pytest.mark.unit
    def test_row_factory_set(self, tmp_path: Path) -> None:
        """确认 row_factory 为 sqlite3.Row。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()

        conn = store.get_connection()
        assert conn.row_factory is sqlite3.Row
        store.close()

    @pytest.mark.unit
    def test_connection_uses_autocommit(self, tmp_path: Path) -> None:
        """默认连接应启用 autocommit，缩短单语句写事务持锁时间。"""

        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()

        conn = store.get_connection()
        assert conn.isolation_level is None
        store.close()

    @pytest.mark.unit
    def test_busy_timeout_extended_for_cross_thread_writes(self, tmp_path: Path) -> None:
        """默认连接应为跨线程写竞争配置更长的 busy timeout。"""

        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()

        conn = store.get_connection()
        busy_timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert busy_timeout_ms == 30000
        store.close()


class TestClose:
    """关闭行为测试。"""

    @pytest.mark.unit
    def test_close_prevents_get_connection(self, tmp_path: Path) -> None:
        """关闭后 get_connection 抛出 RuntimeError。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()
        store.close()

        with pytest.raises(RuntimeError, match="已关闭"):
            store.get_connection()

    @pytest.mark.unit
    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        """多次关闭不报错。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()
        store.close()
        store.close()  # 第二次不应报错


class TestConcurrentReadWrite:
    """多线程并发读写测试。"""

    @pytest.mark.unit
    def test_concurrent_inserts(self, tmp_path: Path) -> None:
        """多线程并发插入不丢数据。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()
        errors: list[Exception] = []

        def insert_worker(worker_id: int) -> None:
            try:
                conn = store.get_connection()
                for i in range(10):
                    with write_transaction(conn):
                        conn.execute(
                            "INSERT INTO sessions (session_id, source, state, created_at, last_activity_at) "
                            "VALUES (?, 'cli', 'active', datetime('now'), datetime('now'))",
                            (f"sess_{worker_id}_{i}",),
                        )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=insert_worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"并发写入出错: {errors}"

        # 验证数据完整性
        conn = store.get_connection()
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert count == 50  # 5 workers × 10 rows
        store.close()

    @pytest.mark.unit
    def test_concurrent_read_write(self, tmp_path: Path) -> None:
        """一写多读不阻塞。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()
        errors: list[Exception] = []

        # 预插入一些数据
        conn = store.get_connection()
        with write_transaction(conn):
            for i in range(10):
                conn.execute(
                    "INSERT INTO sessions (session_id, source, state, created_at, last_activity_at) "
                    "VALUES (?, 'cli', 'active', datetime('now'), datetime('now'))",
                    (f"sess_pre_{i}",),
                )

        def writer() -> None:
            try:
                c = store.get_connection()
                for i in range(20):
                    with write_transaction(c):
                        c.execute(
                            "INSERT INTO sessions (session_id, source, state, created_at, last_activity_at) "
                            "VALUES (?, 'web', 'active', datetime('now'), datetime('now'))",
                            (f"sess_w_{i}",),
                        )
            except Exception as e:
                errors.append(e)

        def reader() -> None:
            try:
                c = store.get_connection()
                for _ in range(50):
                    c.execute("SELECT COUNT(*) FROM sessions").fetchone()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"并发读写出错: {errors}"
        store.close()

    @pytest.mark.unit
    def test_concurrent_writers_serialize_without_lock_error(self, tmp_path: Path) -> None:
        """两个写线程经 write_transaction 序列化后不会触发 upgrade deadlock。"""

        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()
        errors: list[Exception] = []

        def writer(worker_id: int) -> None:
            try:
                c = store.get_connection()
                for i in range(30):
                    with write_transaction(c):
                        c.execute(
                            "INSERT INTO sessions (session_id, source, state, created_at, last_activity_at) "
                            "VALUES (?, 'cli', 'active', datetime('now'), datetime('now'))",
                            (f"sess_cs_{worker_id}_{i}",),
                        )
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(wid,)) for wid in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"序列化写事务意外报错: {errors}"
        conn = store.get_connection()
        total = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert total == 60
        store.close()

    @pytest.mark.unit
    def test_write_transaction_rolls_back_on_exception(self, tmp_path: Path) -> None:
        """事务体抛异常时已写入内容被回滚，后续事务可正常继续。"""

        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()
        conn = store.get_connection()

        class _InjectedError(RuntimeError):
            """测试专用注入异常。"""

        with pytest.raises(_InjectedError):
            with write_transaction(conn):
                conn.execute(
                    "INSERT INTO sessions (session_id, source, state, created_at, last_activity_at) "
                    "VALUES (?, 'cli', 'active', datetime('now'), datetime('now'))",
                    ("sess_rollback",),
                )
                raise _InjectedError("boom")

        count_after_rollback = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
            ("sess_rollback",),
        ).fetchone()[0]
        assert count_after_rollback == 0

        # 证明连接未被卡在事务里：再开一次事务 INSERT 能成功提交。
        with write_transaction(conn):
            conn.execute(
                "INSERT INTO sessions (session_id, source, state, created_at, last_activity_at) "
                "VALUES (?, 'cli', 'active', datetime('now'), datetime('now'))",
                ("sess_after_rollback",),
            )
        count_after_success = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
            ("sess_after_rollback",),
        ).fetchone()[0]
        assert count_after_success == 1
        store.close()

    @pytest.mark.unit
    def test_write_transaction_rolls_back_on_commit_failure(self, tmp_path: Path) -> None:
        """COMMIT 自身失败时必须触发 ROLLBACK，连接不能被卡在事务中。"""

        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()
        conn = store.get_connection()

        class _CommitBoom(sqlite3.OperationalError):
            """测试专用：模拟 SQLite COMMIT 阶段抛 OperationalError。"""

        class _CommitFailingProxy:
            """只拦截 COMMIT 的连接代理；其余调用透传给真实连接。"""

            def __init__(self, real: sqlite3.Connection) -> None:
                self._real = real

            def execute(self, sql: str, *args: object, **kwargs: object) -> sqlite3.Cursor:
                if sql.strip().upper() == "COMMIT":
                    raise _CommitBoom("simulated commit failure")
                return self._real.execute(sql, *args, **kwargs)  # type: ignore[arg-type]

        proxy = cast(sqlite3.Connection, _CommitFailingProxy(conn))
        with pytest.raises(_CommitBoom):
            with write_transaction(proxy):
                conn.execute(
                    "INSERT INTO sessions (session_id, source, state, created_at, last_activity_at) "
                    "VALUES (?, 'cli', 'active', datetime('now'), datetime('now'))",
                    ("sess_commit_fail",),
                )

        # COMMIT 失败必须触发 ROLLBACK：未提交的 INSERT 不应落库。
        count_after_failure = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
            ("sess_commit_fail",),
        ).fetchone()[0]
        assert count_after_failure == 0

        # 连接不能被卡在事务里：下一个 write_transaction 仍能正常开启与提交。
        with write_transaction(conn):
            conn.execute(
                "INSERT INTO sessions (session_id, source, state, created_at, last_activity_at) "
                "VALUES (?, 'cli', 'active', datetime('now'), datetime('now'))",
                ("sess_after_commit_fail",),
            )
        count_after_success = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?",
            ("sess_after_commit_fail",),
        ).fetchone()[0]
        assert count_after_success == 1
        store.close()


class TestSchemaContent:
    """Schema 细节验证。"""

    @pytest.mark.unit
    def test_runs_has_owner_pid_column(self, tmp_path: Path) -> None:
        """runs 表包含 owner_pid 列。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()

        conn = store.get_connection()
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(runs)").fetchall()
        }
        assert "owner_pid" in columns
        store.close()

    @pytest.mark.unit
    def test_permits_has_lane_column(self, tmp_path: Path) -> None:
        """permits 表包含 lane 列。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()

        conn = store.get_connection()
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(permits)").fetchall()
        }
        assert "lane" in columns
        assert "owner_pid" in columns
        store.close()

    @pytest.mark.unit
    def test_indexes_created(self, tmp_path: Path) -> None:
        """索引正确创建。"""
        store = HostStore(tmp_path / "test.db")
        store.initialize_schema()

        conn = store.get_connection()
        indexes = {
            row[1]
            for row in conn.execute(
                "SELECT * FROM sqlite_master WHERE type='index'"
            ).fetchall()
            if row[1] is not None
        }
        assert "idx_runs_session" in indexes
        assert "idx_runs_state" in indexes
        assert "idx_permits_lane" in indexes
        store.close()
