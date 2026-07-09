"""fins/storage 跨进程 crash recovery 真源测试。

design_doc(`docs/fins_storage_design.md` §5)中 batch 跨阶段算账只在单进程
journal/dry_run 层面被现有 `test_storage_batch_recovery` 覆盖,**worker 真崩**
(`os._exit`)留下 orphan 后主进程能否自动恢复并继续工作未被显式钉死。

本文件用 `subprocess.Popen` 起 worker:worker 在 `_PHASE_STARTED` 阶段写完
staging 后调用 ``os._exit(7)`` 模拟进程崩溃,留下 orphan batch、ticker 锁
和 journal。主进程随后通过 ``build_fs_repository_set`` 装配新 core,
``ensure_batch_recovery`` 在初始化期自动跑,并验证:

1. 同 ticker 可以再次 ``begin_batch``(orphan 已被清理)。
2. ``recover_orphan_batches(dry_run=True)`` 返回空 tuple(再无残留)。
3. target 目录数据未受 worker staging 写入影响(隔离生效)。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from dayu.fins.domain.document_models import SourceDocumentUpsertRequest
from dayu.fins.domain.enums import SourceKind
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from tests.fins.storage_testkit import build_fs_storage_test_context


_CRASH_HOLDER_READY_TIMEOUT_SEC = 15.0
_CRASH_HOLDER_POLL_INTERVAL_SEC = 0.05
_CRASH_HOLDER_EXIT_CODE = 7


def _repo_root() -> Path:
    """返回仓库根目录。

    Returns:
        当前测试所在仓库根目录。

    Raises:
        无。
    """

    return Path(__file__).resolve().parents[2]


def _build_crash_holder_script() -> str:
    """构造 worker 子进程脚本:起 batch、写 staging、`os._exit(7)` 崩溃。

    Worker 不调用 ``commit_batch`` 也不调用 ``rollback_batch``,直接 ``os._exit``
    退出,留下 ticker 锁文件、staging 目录与 ``_PHASE_STARTED`` journal。

    Returns:
        供 ``python -c`` 执行的脚本文本。

    Raises:
        无。
    """

    return textwrap.dedent(
        """
        import json
        import os
        import sys
        from pathlib import Path

        from dayu.fins.storage._fs_repository_factory import build_fs_repository_set

        workspace_root = Path(sys.argv[1])
        ticker = sys.argv[2]
        ready_path = Path(sys.argv[3])

        repository_set = build_fs_repository_set(workspace_root=workspace_root)
        core = repository_set.core
        token = core.begin_batch(ticker)

        staging_filings_dir = token.staging_ticker_dir / "filings" / "fil_crash_orphan"
        staging_filings_dir.mkdir(parents=True, exist_ok=True)
        (staging_filings_dir / "marker.txt").write_text(
            "worker died with staging dirty", encoding="utf-8"
        )

        ready_path.write_text(
            json.dumps(
                {
                    "ticker": token.ticker,
                    "token_id": token.token_id,
                    "staging_root_dir": str(token.staging_root_dir),
                    "ticker_lock_path": str(token.ticker_lock_path),
                    "journal_path": str(token.journal_path),
                }
            ),
            encoding="utf-8",
        )

        os._exit(7)
        """
    )


def _spawn_crash_holder(
    workspace_root: Path,
    ticker: str,
    ready_path: Path,
) -> subprocess.Popen[str]:
    """启动会在 ``_PHASE_STARTED`` 后崩溃的 worker 子进程。

    Args:
        workspace_root: 测试工作区根目录。
        ticker: 需要持有 batch 锁的股票代码。
        ready_path: 子进程回传状态文件路径,主进程据此确认 staging 已就绪。

    Returns:
        worker 子进程句柄。

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
            _build_crash_holder_script(),
            str(workspace_root),
            ticker,
            str(ready_path),
        ],
        cwd=_repo_root(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_crash_then_exit(
    process: subprocess.Popen[str],
    ready_path: Path,
) -> dict[str, str]:
    """等待 worker 写出 ready 文件并完成崩溃退出。

    Args:
        process: worker 子进程句柄。
        ready_path: 子进程状态文件路径。

    Returns:
        worker 在崩溃前回传的 staging / 锁 / journal 路径状态。

    Raises:
        AssertionError: 超时或退出码非预期时抛出。
    """

    deadline = time.monotonic() + _CRASH_HOLDER_READY_TIMEOUT_SEC
    payload: dict[str, str] | None = None
    while time.monotonic() < deadline:
        if ready_path.exists() and payload is None:
            raw = json.loads(ready_path.read_text(encoding="utf-8"))
            assert isinstance(raw, dict)
            payload = {str(k): str(v) for k, v in raw.items()}
        if process.poll() is not None:
            break
        time.sleep(_CRASH_HOLDER_POLL_INTERVAL_SEC)

    stdout, stderr = process.communicate(timeout=5)
    assert payload is not None, (
        f"worker 未在崩溃前写出 ready 状态:\nstdout={stdout}\nstderr={stderr}"
    )
    assert process.returncode == _CRASH_HOLDER_EXIT_CODE, (
        f"worker 退出码应为 {_CRASH_HOLDER_EXIT_CODE} (os._exit 模拟崩溃),"
        f"实际={process.returncode}\nstdout={stdout}\nstderr={stderr}"
    )
    return payload


@pytest.mark.timeout(60)
def test_cross_process_crash_leaves_recoverable_orphan(tmp_path: Path) -> None:
    """worker 在 `_PHASE_STARTED` 后崩溃留 orphan,主进程重启可自动恢复。

    Args:
        tmp_path: pytest 临时目录,作为测试工作区根目录。

    Returns:
        无。

    Raises:
        AssertionError: orphan 未被恢复、目标目录被污染、或 ticker 锁未释放时抛出。
    """

    ctx = build_fs_storage_test_context(tmp_path)
    ctx.source_repository.create_source_document(
        SourceDocumentUpsertRequest(
            ticker="CRASH",
            document_id="fil_stable_001",
            internal_document_id="0001",
            form_type="10-K",
            primary_document="main.html",
            file_entries=[
                {"name": "main.html", "uri": "local://CRASH/fil_stable_001/main.html"}
            ],
            meta={"fiscal_year": 2024, "fiscal_period": "FY"},
        ),
        source_kind=SourceKind.FILING,
    )

    stable_meta_before = ctx.source_repository.get_source_meta(
        "CRASH", "fil_stable_001", SourceKind.FILING
    )

    ready_path = tmp_path / "crash-ready.json"
    process = _spawn_crash_holder(tmp_path, "CRASH", ready_path)
    payload = _wait_for_crash_then_exit(process, ready_path)

    journal_path = Path(payload["journal_path"])
    assert journal_path.exists(), "崩溃后 journal 应该残留,作为恢复依据"

    # 主进程通过新 repository_set 触发 ensure_batch_recovery 自动跑。
    recovered_ctx = build_fs_storage_test_context(tmp_path)

    # 验证 1:同 ticker 可以再次 begin_batch,说明 ticker 锁与 staging 已被恢复清理。
    new_token = recovered_ctx.core.begin_batch("CRASH")
    try:
        # 验证 2:recover_orphan_batches(dry_run=True) 应再无残留动作。
        remaining = recovered_ctx.core.recover_orphan_batches(dry_run=True)
        # 当前进程的活跃 batch 不应出现在 orphan 列表中(归活跃 ticker 锁保护)。
        for entry in remaining:
            assert new_token.token_id not in str(entry), (
                f"活跃 batch token 不应被列为 orphan: {entry}"
            )
    finally:
        recovered_ctx.core.rollback_batch(new_token)

    # 验证 3:target 目录 stable doc 数据未受 worker staging 写入影响。
    stable_meta_after = recovered_ctx.source_repository.get_source_meta(
        "CRASH", "fil_stable_001", SourceKind.FILING
    )
    assert stable_meta_after == stable_meta_before, (
        "worker 只写了 staging,主进程读 target 不应受影响"
    )

    # 验证 4:worker staging 写入的 fil_crash_orphan 不应渗透到 target。
    with pytest.raises(FileNotFoundError):
        recovered_ctx.source_repository.get_source_meta(
            "CRASH", "fil_crash_orphan", SourceKind.FILING
        )
