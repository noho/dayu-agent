"""PR-2 (#107) 遗留：``cleanup_stale_resuming`` 跨进程冒烟。

PR-2 在 store 事务层完成了 fence token 强一致改造，回归测试覆盖到了
**线程级双连接**（同一进程内 ``threading.Thread`` 共享 SQLite 文件）。
真正的"两个 dayu 进程同 workspace 同 pending turn"场景需要等到 PR-5
落地 advisory lock 之后再补跨进程冒烟，本测试就是这条遗留断言。

测试结构：
  * 主进程预置一条 pending turn 并完成首次 ``record_resume_attempt``
    （持 lease ``A``）。
  * 启动一个 ``subprocess.Popen`` worker，在另一进程内对同一个 SQLite 文件
    打开第二把连接并执行 ``cleanup_stale_resuming``，模拟另一台 host 进程
    抢占 stale RESUMING。
  * 等 worker 干净退出后，主进程持旧 lease ``A`` 调用
    ``release_resume_lease`` / ``rebind_source_run_id_for_resume`` /
    ``record_resume_failure``，全部必须抛 ``LeaseExpiredError``。
  * 反向路径：主进程重新 ``record_resume_attempt`` 拿到 lease ``B``；启动
    第二个 worker 用主进程已经"过期"的 ``A`` lease 去触发 release/cleanup
    必须 no-op，不会抢走 ``B`` 的合法 lease。

不依赖外部 fixture：通过 ``subprocess.run`` 起 ``python -c '<inline>'`` 子
进程，子进程在自身 ``sys.path`` 里 import 真实 dayu 模块，对同一个
SQLite 文件做 ``cleanup_stale_resuming``；事件同步走 stdout 行（worker
干完写一行 ``DONE``）。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dayu.host.host_store import HostStore
from dayu.host.lease import LeaseExpiredError
from dayu.host.pending_turn_store import (
    PendingConversationTurnState,
    SQLitePendingConversationTurnStore,
)


# 用 ``__file__`` 反推项目根而不是 ``Path.cwd()``：CI / 运维从任意目录
# 直接跑 ``pytest /abs/path/to/tests/...`` 时 cwd 不一定是项目根，子进程
# inline 脚本必须显式 import dayu，所以把项目根钉在编译期已知的常量上。
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_worker_script(db_path: Path, pending_turn_id: str, expected_updated_at_iso: str) -> str:
    """构造 worker 子进程要 ``python -c`` 执行的源码字符串。

    Args:
        db_path: 主进程 HostStore 落地的 SQLite 数据库文件路径。
        pending_turn_id: 目标 pending turn ID。
        expected_updated_at_iso: ``cleanup_stale_resuming`` 所需的 stale
            判定时间戳，按 ``datetime.isoformat()`` 序列化。

    Returns:
        worker 子进程的 inline Python 源码。
    """

    return (
        "import sys\n"
        f"sys.path.insert(0, {str(_PROJECT_ROOT)!r})\n"
        "from datetime import datetime\n"
        "from pathlib import Path\n"
        "from dayu.host.host_store import HostStore\n"
        "from dayu.host.pending_turn_store import SQLitePendingConversationTurnStore\n"
        f"host_store = HostStore(Path({str(db_path)!r}))\n"
        "host_store.initialize_schema()\n"
        "store = SQLitePendingConversationTurnStore(host_store)\n"
        "store.cleanup_stale_resuming(\n"
        f"    {pending_turn_id!r},\n"
        f"    expected_updated_at=datetime.fromisoformat({expected_updated_at_iso!r}),\n"
        ")\n"
        "print('DONE', flush=True)\n"
    )


def _run_worker(db_path: Path, pending_turn_id: str, expected_updated_at_iso: str) -> None:
    """启动 worker 子进程并等待其完成。

    Args:
        db_path: SQLite 数据库路径。
        pending_turn_id: 目标 pending turn ID。
        expected_updated_at_iso: stale 判定时间戳的 ISO 序列化形式。

    Returns:
        无。

    Raises:
        AssertionError: worker 退出码非 0 或未输出 ``DONE`` 标记。
    """

    script = _build_worker_script(db_path, pending_turn_id, expected_updated_at_iso)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, (
        f"worker 退出码={completed.returncode}, stderr={completed.stderr}"
    )
    assert "DONE" in completed.stdout


@pytest.mark.unit
def test_cleanup_stale_resuming_cross_process_invalidates_lease(tmp_path: Path) -> None:
    """跨进程：主进程持旧 lease，worker 进程 cleanup 后旧 lease 操作必抛 LeaseExpiredError。"""

    db_path = tmp_path / ".host" / "dayu_host.db"
    host_store = HostStore(db_path)
    host_store.initialize_schema()

    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    a_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    a_lease = a_record.resume_lease_id or ""
    assert a_lease

    # 关闭主进程持有的连接，避免 SQLite 在不同进程间因 WAL shm 状态产生干扰。
    host_store.close()

    _run_worker(
        db_path,
        created.pending_turn_id,
        a_record.updated_at.isoformat(),
    )

    # 主进程重新打开 store；旧 lease 走任一 fence token CAS 写路径都必须 LeaseExpiredError。
    host_store = HostStore(db_path)
    store = SQLitePendingConversationTurnStore(host_store)

    with pytest.raises(LeaseExpiredError):
        store.release_resume_lease(created.pending_turn_id, lease_id=a_lease)
    with pytest.raises(LeaseExpiredError):
        store.rebind_source_run_id_for_resume(
            created.pending_turn_id,
            new_source_run_id="run_a_late",
            lease_id=a_lease,
        )
    with pytest.raises(LeaseExpiredError):
        store.record_resume_failure(
            created.pending_turn_id,
            error_message="late",
            lease_id=a_lease,
        )


@pytest.mark.unit
def test_cleanup_stale_resuming_cross_process_does_not_steal_fresh_lease(tmp_path: Path) -> None:
    """反向路径：主进程已重新 acquire 拿到新 lease，worker 用旧 stale 时间戳 cleanup 必须 no-op。"""

    db_path = tmp_path / ".host" / "dayu_host.db"
    host_store = HostStore(db_path)
    host_store.initialize_schema()

    store = SQLitePendingConversationTurnStore(host_store)
    created = store.upsert_pending_turn(
        session_id="s1",
        scene_name="wechat",
        user_text="问题",
        source_run_id="run_1",
        resumable=True,
        state=PendingConversationTurnState.ACCEPTED_BY_HOST,
    )
    a_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    a_stale_snapshot_iso = a_record.updated_at.isoformat()

    # 主线程内合法 release，再次 acquire 拿到 fresh lease。
    store.release_resume_lease(
        created.pending_turn_id, lease_id=a_record.resume_lease_id or ""
    )
    b_record = store.record_resume_attempt(created.pending_turn_id, max_attempts=5)
    b_lease = b_record.resume_lease_id or ""
    assert b_lease and b_lease != (a_record.resume_lease_id or "")

    host_store.close()

    # worker 用 a_record 的 stale 时间戳触发 cleanup：expected_updated_at 与
    # 当前 b_record.updated_at 不一致 → cleanup 应识别为非 stale，no-op。
    _run_worker(
        db_path,
        created.pending_turn_id,
        a_stale_snapshot_iso,
    )

    # 主进程重开 store，B 的 fresh lease 仍然合法可用。
    host_store = HostStore(db_path)
    store = SQLitePendingConversationTurnStore(host_store)
    released = store.release_resume_lease(
        created.pending_turn_id, lease_id=b_lease
    )
    assert released is not None
    assert released.state is PendingConversationTurnState.ACCEPTED_BY_HOST
