"""Fins 文件系统 batch recovery 真源测试。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time
from typing import TypedDict

import pytest

import dayu.fins.storage._fs_storage_infra as fs_storage_infra_module
from dayu.fins.storage.fs_company_meta_repository import FsCompanyMetaRepository
from tests.fins.storage_testkit import build_fs_storage_test_context


class _LockHolderState(TypedDict):
    """跨进程持锁子进程回传的状态。"""

    ticker: str
    staging_root_dir: str
    ticker_lock_path: str


def _write_text(path: Path, content: str) -> None:
    """向测试路径写入文本文件。

    Args:
        path: 目标文件路径。
        content: 文件内容。

    Returns:
        无。

    Raises:
        OSError: 写入失败时抛出。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _repo_root() -> Path:
    """返回仓库根目录。

    Args:
        无。

    Returns:
        当前测试所在仓库根目录。

    Raises:
        无。
    """

    return Path(__file__).resolve().parents[2]


def _build_lock_holder_script() -> str:
    """构造跨进程持锁子脚本文本。

    Args:
        无。

    Returns:
        供 `python -c` 执行的脚本文本。

    Raises:
        无。
    """

    return textwrap.dedent(
        """
        import json
        import sys
        import time
        from pathlib import Path

        from dayu.fins.storage._fs_repository_factory import build_fs_repository_set

        workspace_root = Path(sys.argv[1])
        ticker = sys.argv[2]
        ready_path = Path(sys.argv[3])
        release_path = Path(sys.argv[4])

        repository_set = build_fs_repository_set(workspace_root=workspace_root)
        core = repository_set.core
        token = core.begin_batch(ticker)
        ready_path.write_text(
            json.dumps(
                {
                    "ticker": token.ticker,
                    "staging_root_dir": str(token.staging_root_dir),
                    "ticker_lock_path": str(token.ticker_lock_path),
                }
            ),
            encoding="utf-8",
        )
        try:
            while not release_path.exists():
                time.sleep(0.05)
        finally:
            core.rollback_batch(token)
        """
    )


def _spawn_lock_holder(
    workspace_root: Path,
    ticker: str,
    ready_path: Path,
    release_path: Path,
) -> subprocess.Popen[str]:
    """启动一个持有指定 ticker batch 锁的子进程。

    Args:
        workspace_root: 测试工作区根目录。
        ticker: 需要持有的股票代码。
        ready_path: 子进程回传状态文件路径。
        release_path: 主进程发出释放信号的文件路径。

    Returns:
        已启动的子进程句柄。

    Raises:
        OSError: 子进程启动失败时抛出。
    """

    environment = dict(os.environ)
    python_path_parts = [str(_repo_root())]
    existing_python_path = environment.get("PYTHONPATH")
    if existing_python_path:
        python_path_parts.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            _build_lock_holder_script(),
            str(workspace_root),
            ticker,
            str(ready_path),
            str(release_path),
        ],
        cwd=_repo_root(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_lock_holder_state(process: subprocess.Popen[str], ready_path: Path) -> _LockHolderState:
    """等待持锁子进程写出状态文件。

    Args:
        process: 持锁子进程。
        ready_path: 子进程状态文件路径。

    Returns:
        子进程回传的持锁状态。

    Raises:
        AssertionError: 子进程提前退出或超时未就绪时抛出。
    """

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if ready_path.exists():
            raw_payload = json.loads(ready_path.read_text(encoding="utf-8"))
            assert isinstance(raw_payload, dict)
            ticker = raw_payload.get("ticker")
            staging_root_dir = raw_payload.get("staging_root_dir")
            ticker_lock_path = raw_payload.get("ticker_lock_path")
            assert isinstance(ticker, str)
            assert isinstance(staging_root_dir, str)
            assert isinstance(ticker_lock_path, str)
            return {
                "ticker": ticker,
                "staging_root_dir": staging_root_dir,
                "ticker_lock_path": ticker_lock_path,
            }
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                "持锁子进程在写出状态前提前退出:\n"
                f"stdout={stdout}\n"
                f"stderr={stderr}"
            )
        time.sleep(0.05)
    _stop_lock_holder(process, ready_path.parent / "holder-release-on-timeout")
    raise AssertionError("等待持锁子进程状态超时")


def _stop_lock_holder(process: subprocess.Popen[str], release_path: Path) -> None:
    """通知持锁子进程退出并等待收口。

    Args:
        process: 持锁子进程。
        release_path: 释放信号文件路径。

    Returns:
        无。

    Raises:
        AssertionError: 子进程异常退出时抛出。
    """

    if process.poll() is None:
        release_path.touch()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise AssertionError(
                "持锁子进程未能在超时内退出:\n"
                f"stdout={stdout}\n"
                f"stderr={stderr}"
            )
        if process.returncode != 0:
            raise AssertionError(
                "持锁子进程异常退出:\n"
                f"stdout={stdout}\n"
                f"stderr={stderr}"
            )


@pytest.mark.unit
def test_batch_paths_move_under_dayu_root(tmp_path: Path) -> None:
    """验证 batch 与 backup 目录已迁移到 `.dayu/` 下。"""

    context = build_fs_storage_test_context(tmp_path)
    core = context.core

    assert core.dayu_root == tmp_path / ".dayu"
    assert core.batch_root == tmp_path / ".dayu" / "repo_batches"
    assert core.backup_root == tmp_path / ".dayu" / "repo_backups"


@pytest.mark.unit
def test_recover_started_batch_cleans_orphan_staging(tmp_path: Path) -> None:
    """验证 started 阶段遗留的 staging 会在下次构造仓储时自动清理。"""

    context = build_fs_storage_test_context(tmp_path)
    core = context.core
    token = core.begin_batch("AAPL")
    staged_marker = token.staging_ticker_dir / "filings" / "marker.txt"
    _write_text(staged_marker, "staged")

    core._active_batches.clear()
    core._release_ticker_lock(token.ticker)

    assert token.staging_root_dir.exists()

    FsCompanyMetaRepository(tmp_path)

    assert not token.staging_root_dir.exists()


@pytest.mark.unit
def test_recover_backed_up_target_restores_backup(tmp_path: Path) -> None:
    """验证 backed_up_target 阶段会把 backup 恢复回正式目录。"""

    context = build_fs_storage_test_context(tmp_path)
    core = context.core
    target_marker = core._target_ticker_dir("AAPL") / "filings" / "before.txt"
    _write_text(target_marker, "before")

    token = core.begin_batch("AAPL")
    core._write_batch_journal(token, "backed_up_target")
    token.target_ticker_dir.parent.mkdir(parents=True, exist_ok=True)
    target_dir = token.target_ticker_dir
    backup_dir = token.backup_dir
    target_dir.rename(backup_dir)

    core._active_batches.clear()
    core._release_ticker_lock(token.ticker)

    actions = core.recover_orphan_batches()

    assert any("restore backup ticker=AAPL" in action for action in actions)
    assert target_dir.exists()
    assert not backup_dir.exists()
    assert not token.staging_root_dir.exists()
    assert (target_dir / "filings" / "before.txt").read_text(encoding="utf-8") == "before"


@pytest.mark.unit
def test_recover_swapped_target_deletes_leftover_backup(tmp_path: Path) -> None:
    """验证 swapped_target 阶段会保留新 target 并清理遗留 backup。"""

    context = build_fs_storage_test_context(tmp_path)
    core = context.core
    old_marker = core._target_ticker_dir("AAPL") / "filings" / "version.txt"
    _write_text(old_marker, "old")

    token = core.begin_batch("AAPL")
    _write_text(token.staging_ticker_dir / "filings" / "version.txt", "new")
    core._write_batch_journal(token, "backed_up_target")
    token.target_ticker_dir.rename(token.backup_dir)
    token.staging_ticker_dir.rename(token.target_ticker_dir)
    core._write_batch_journal(token, "swapped_target")

    core._active_batches.clear()
    core._release_ticker_lock(token.ticker)

    actions = core.recover_orphan_batches()

    assert any("delete backup ticker=AAPL" in action for action in actions)
    assert token.target_ticker_dir.exists()
    assert not token.backup_dir.exists()
    assert not token.staging_root_dir.exists()
    assert (token.target_ticker_dir / "filings" / "version.txt").read_text(encoding="utf-8") == "new"


@pytest.mark.unit
def test_recover_orphan_batches_dry_run_is_non_destructive(tmp_path: Path) -> None:
    """验证 dry-run 只返回动作，不修改文件系统。"""

    context = build_fs_storage_test_context(tmp_path)
    core = context.core
    token = core.begin_batch("AAPL")
    _write_text(token.staging_ticker_dir / "filings" / "marker.txt", "staged")

    core._active_batches.clear()
    core._release_ticker_lock(token.ticker)

    actions = core.recover_orphan_batches(dry_run=True)

    assert any("cleanup batch ticker=AAPL" in action for action in actions)
    assert token.staging_root_dir.exists()


@pytest.mark.unit
def test_recover_missing_journal_without_ticker_keeps_unknown_token_dir(tmp_path: Path) -> None:
    """验证缺少 journal 且无法推断 ticker 的 token 目录会保守跳过。"""

    context = build_fs_storage_test_context(tmp_path)
    core = context.core
    token_dir = core.batch_root / "token-without-journal"
    token_dir.mkdir(parents=True, exist_ok=True)

    actions = core.recover_orphan_batches()

    assert actions == (
        "skip batch token=token-without-journal phase=unknown reason=missing journal",
    )
    assert token_dir.exists()


@pytest.mark.unit
def test_recover_orphan_batches_cleans_live_owner_without_lock(tmp_path: Path) -> None:
    """验证 recovery 在 ticker 锁已释放时会清理 orphan，即使 owner 进程仍存活。"""

    context = build_fs_storage_test_context(tmp_path)
    core = context.core
    token = core.begin_batch("AAPL")
    _write_text(token.staging_ticker_dir / "filings" / "marker.txt", "staged")

    core._active_batches.clear()
    core._release_ticker_lock(token.ticker)

    actions = core.recover_orphan_batches()

    assert any("cleanup batch ticker=AAPL" in action for action in actions)
    assert not token.staging_root_dir.exists()


@pytest.mark.unit
def test_recover_orphan_batches_tolerates_token_dir_disappearing_during_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 recovery 扫描 live batch 时，token 目录中途消失不会中断整批恢复。

    复现场景：
    - recovery 进程先枚举到 `repo_batches/<token>`。
    - owner 进程随后完成提交并删掉该 token 根目录。
    - 当前恢复流程继续进入 `_infer_batch_ticker()` 时，`token_dir.iterdir()`
      可能直接抛出 `FileNotFoundError`。

    期望行为：
    - recovery 把该 token 视为已消失的 live batch，安静跳过。
    - 不因单个竞态样本中断整个 `recover_orphan_batches()`。
    """

    context = build_fs_storage_test_context(tmp_path)
    core = context.core
    token = core.begin_batch("AAPL")
    _write_text(token.staging_ticker_dir / "filings" / "marker.txt", "staged")

    core._active_batches.clear()
    core._release_ticker_lock(token.ticker)
    token.journal_path.unlink()

    original_infer_batch_ticker = core._infer_batch_ticker

    def _delete_token_dir_before_infer(token_dir: Path) -> str:
        """在推断 ticker 前模拟 owner 进程已删除 token 根目录。"""

        if token_dir == token.staging_root_dir:
            shutil.rmtree(token_dir, ignore_errors=True)
        return original_infer_batch_ticker(token_dir)

    monkeypatch.setattr(core, "_infer_batch_ticker", _delete_token_dir_before_infer)

    actions = core.recover_orphan_batches()

    assert actions == (
        f"skip batch token={token.token_id} phase=unknown reason=missing journal",
    )
    assert not token.staging_root_dir.exists()


@pytest.mark.unit
def test_commit_batch_keeps_new_target_when_swapped_target_journal_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证新 ticker 在换入目标后 journal 写失败时仍保留已落盘目标。"""

    context = build_fs_storage_test_context(tmp_path)
    core = context.core
    token = core.begin_batch("AAPL")
    _write_text(token.staging_ticker_dir / "filings" / "version.txt", "new")
    original_write_batch_journal = core._write_batch_journal

    def _failing_write_batch_journal(batch_token: object, phase: str) -> None:
        """在 swapped_target 阶段模拟 journal 写入失败。"""

        if phase == "swapped_target":
            raise OSError("disk full")
        original_write_batch_journal(batch_token, phase)  # type: ignore[arg-type]

    monkeypatch.setattr(core, "_write_batch_journal", _failing_write_batch_journal)

    with pytest.raises(OSError, match="disk full"):
        core.commit_batch(token)

    assert token.target_ticker_dir.exists()
    assert not token.backup_dir.exists()
    assert not token.staging_root_dir.exists()
    assert token.ticker not in core._active_batches
    assert (token.target_ticker_dir / "filings" / "version.txt").read_text(encoding="utf-8") == "new"


@pytest.mark.unit
def test_rollback_batch_cleans_and_unlocks_when_rolled_back_journal_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 rollback journal 写失败时仍会清理 staging 并释放 ticker 锁。"""

    context = build_fs_storage_test_context(tmp_path)
    core = context.core
    token = core.begin_batch("AAPL")
    _write_text(token.staging_ticker_dir / "filings" / "version.txt", "draft")
    original_write_batch_journal = core._write_batch_journal

    def _failing_write_batch_journal(batch_token: object, phase: str) -> None:
        """在 rolled_back 阶段模拟 journal 写入失败。"""

        if phase == "rolled_back":
            raise OSError("disk full")
        original_write_batch_journal(batch_token, phase)  # type: ignore[arg-type]

    monkeypatch.setattr(core, "_write_batch_journal", _failing_write_batch_journal)

    with pytest.raises(OSError, match="disk full"):
        core.rollback_batch(token)

    assert not token.staging_root_dir.exists()
    assert token.ticker not in core._active_batches

    monkeypatch.setattr(core, "_write_batch_journal", original_write_batch_journal)
    next_token = core.begin_batch("AAPL")
    try:
        assert next_token.ticker == "AAPL"
    finally:
        core.rollback_batch(next_token)


@pytest.mark.unit
def test_execute_with_auto_batch_preserves_original_error_when_rollback_journal_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 auto batch 回滚失败时仍优先抛出原始写路径异常。"""

    context = build_fs_storage_test_context(tmp_path)
    core = context.core
    original_write_batch_journal = core._write_batch_journal

    def _failing_write_batch_journal(batch_token: object, phase: str) -> None:
        """在 rolled_back 阶段模拟 journal 写入失败。"""

        if phase == "rolled_back":
            raise OSError("disk full")
        original_write_batch_journal(batch_token, phase)  # type: ignore[arg-type]

    def _failing_operation() -> None:
        """模拟原始写路径失败。"""

        raise ValueError("write failed")

    monkeypatch.setattr(core, "_write_batch_journal", _failing_write_batch_journal)

    with pytest.raises(ValueError, match="write failed") as exc_info:
        core._execute_with_auto_batch("AAPL", _failing_operation)

    notes = getattr(exc_info.value, "__notes__", [])
    assert any("rollback_batch failed: disk full" in note for note in notes)

    monkeypatch.setattr(core, "_write_batch_journal", original_write_batch_journal)
    next_token = core.begin_batch("AAPL")
    try:
        assert next_token.ticker == "AAPL"
    finally:
        core.rollback_batch(next_token)


@pytest.mark.unit
def test_begin_batch_rejects_cross_process_same_ticker(tmp_path: Path) -> None:
    """验证同一 ticker 的 batch 锁不会被两个进程同时拿到。"""

    ready_path = tmp_path / "holder-ready.json"
    release_path = tmp_path / "holder-release"
    process = _spawn_lock_holder(tmp_path, "AAPL", ready_path, release_path)
    try:
        holder_state = _wait_for_lock_holder_state(process, ready_path)
        assert holder_state["ticker"] == "AAPL"
        assert Path(holder_state["ticker_lock_path"]).exists()

        contender_core = build_fs_storage_test_context(tmp_path).core
        with pytest.raises(RuntimeError, match="跨进程活动 batch"):
            contender_core.begin_batch("AAPL")
    finally:
        _stop_lock_holder(process, release_path)


@pytest.mark.unit
def test_recover_orphan_batches_skips_live_locked_batch(tmp_path: Path) -> None:
    """验证 recovery 遇到活跃 ticker 锁时会跳过，而不会误删 staging。"""

    ready_path = tmp_path / "holder-ready.json"
    release_path = tmp_path / "holder-release"
    process = _spawn_lock_holder(tmp_path, "AAPL", ready_path, release_path)
    try:
        holder_state = _wait_for_lock_holder_state(process, ready_path)
        assert holder_state["ticker"] == "AAPL"
        staging_root_dir = Path(holder_state["staging_root_dir"])
        assert staging_root_dir.exists()

        recovery_core = build_fs_storage_test_context(tmp_path).core
        actions = recovery_core.recover_orphan_batches()

        assert actions == ()
        assert staging_root_dir.exists()
    finally:
        _stop_lock_holder(process, release_path)


@pytest.mark.unit
def test_begin_batch_allows_different_ticker_while_other_process_holds_lock(tmp_path: Path) -> None:
    """验证 batch 锁按 ticker 粒度隔离，不同 ticker 可并发开启。"""

    ready_path = tmp_path / "holder-ready.json"
    release_path = tmp_path / "holder-release"
    process = _spawn_lock_holder(tmp_path, "AAPL", ready_path, release_path)
    try:
        holder_state = _wait_for_lock_holder_state(process, ready_path)
        assert holder_state["ticker"] == "AAPL"

        contender_core = build_fs_storage_test_context(tmp_path).core
        token = contender_core.begin_batch("MSFT")
        try:
            assert token.ticker == "MSFT"
            assert token.staging_root_dir.exists()
            assert token.ticker_lock_path.name == "MSFT.lock"
        finally:
            contender_core.rollback_batch(token)
    finally:
        _stop_lock_holder(process, release_path)


@pytest.mark.unit
def test_fins_batch_lock_uses_msvcrt_when_fcntl_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Windows 分支应改用 msvcrt 提供真正的非阻塞 batch 锁。"""

    class _FakeMsvcrt:
        """记录 locking 调用的 Windows 锁实现桩。"""

        LK_NBLCK = 2
        LK_UNLCK = 3

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def locking(self, fd: int, mode: int, size: int) -> None:
            del fd
            self.calls.append((mode, size))

    core = build_fs_storage_test_context(tmp_path).core
    fake_msvcrt = _FakeMsvcrt()
    monkeypatch.setattr(fs_storage_infra_module.file_lock_module, "_FCNTL", None)
    monkeypatch.setattr(fs_storage_infra_module.file_lock_module, "_MSVCRT", fake_msvcrt)
    fake_msvcrt.calls.clear()
    stream = core._open_and_lock_stream(tmp_path / ".dayu" / "batch_locks" / "AAPL.lock", blocking=False)
    try:
        assert fake_msvcrt.calls == [(fake_msvcrt.LK_NBLCK, 1)]
    finally:
        core._release_lock_stream(stream)

    assert fake_msvcrt.calls == [
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_UNLCK, 1),
    ]


@pytest.mark.unit
def test_recovery_lock_retries_until_windows_lock_is_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 recovery 阻塞锁在 Windows 上会持续重试直到成功。"""

    class _RetryingMsvcrt:
        """前两次竞争失败、随后成功的 Windows 锁实现桩。"""

        LK_NBLCK = 2
        LK_UNLCK = 3

        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []
            self._remaining_contention_failures = 2

        def locking(self, fd: int, mode: int, size: int) -> None:
            del fd
            self.calls.append((mode, size))
            if mode == self.LK_NBLCK and self._remaining_contention_failures > 0:
                self._remaining_contention_failures -= 1
                raise BlockingIOError(
                    fs_storage_infra_module.file_lock_module.errno.EAGAIN,
                    "locked",
                )

    core = build_fs_storage_test_context(tmp_path).core
    fake_msvcrt = _RetryingMsvcrt()
    sleep_calls: list[float] = []
    monkeypatch.setattr(fs_storage_infra_module.file_lock_module, "_FCNTL", None)
    monkeypatch.setattr(fs_storage_infra_module.file_lock_module, "_MSVCRT", fake_msvcrt)
    monkeypatch.setattr(fs_storage_infra_module.file_lock_module.time, "sleep", sleep_calls.append)

    stream = core._acquire_recovery_lock()
    try:
        assert fake_msvcrt.calls == [
            (fake_msvcrt.LK_NBLCK, 1),
            (fake_msvcrt.LK_NBLCK, 1),
            (fake_msvcrt.LK_NBLCK, 1),
        ]
        assert sleep_calls == [
            fs_storage_infra_module.file_lock_module._WINDOWS_LOCK_RETRY_INTERVAL_SEC,
            fs_storage_infra_module.file_lock_module._WINDOWS_LOCK_RETRY_INTERVAL_SEC,
        ]
    finally:
        core._release_lock_stream(stream)

    assert fake_msvcrt.calls == [
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_NBLCK, 1),
        (fake_msvcrt.LK_UNLCK, 1),
    ]
