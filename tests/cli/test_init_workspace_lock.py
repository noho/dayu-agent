"""``dayu-cli init`` workspace 临界区互斥测试。

本模块覆盖 PR-5 (#108) 引入的 workspace advisory lock：
``_copy_config + _copy_assets + apply_all_workspace_migrations`` 整段必须由
``StateDirSingleInstanceLock`` 独占持有，以避免双 ``dayu-cli init`` 进程在同一
workspace 上并发改动 SQLite 与 ``run.json`` 时互相踩踏。

测试不真正起子进程：直接复用 ``StateDirSingleInstanceLock`` 的真实文件锁
（POSIX ``fcntl.flock`` / Windows ``msvcrt.locking``）模拟"另一进程已经持锁"，
然后调用 ``dayu.cli.commands.init`` 入口，断言：

1. 检测到锁竞争时返回非 0 退出码；
2. 不会进入 ``_copy_config`` / ``apply_all_workspace_migrations`` 临界区；
3. 锁释放后再次 init 可以正常进入临界区。
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from dayu.cli.commands import init as init_module
from dayu.state_dir_lock import StateDirSingleInstanceLock


@pytest.fixture()
def workspace_root(tmp_path: Path) -> Path:
    """构造一个干净的 workspace 根目录。"""

    root = tmp_path / "workspace"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _build_args(workspace_root: Path) -> Namespace:
    """构造 ``run_init_command`` 所需的 argparse Namespace。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        Namespace：模拟 ``dayu-cli init -d <workspace>`` 的解析结果。
    """

    return Namespace(base=str(workspace_root), overwrite=False, reset=False)


def _build_reset_args(workspace_root: Path) -> Namespace:
    """构造 ``dayu-cli init --reset`` 的 Namespace。"""

    return Namespace(base=str(workspace_root), overwrite=False, reset=True)


@pytest.mark.unit
def test_init_aborts_when_workspace_lock_already_held(
    workspace_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """另一进程持有 workspace lock 时 init 必须立即放弃。"""

    # 临界区不可被进入。任一函数被调用即说明锁互斥失效。
    def _critical_region_must_not_run(*_args: Any, **_kwargs: Any) -> Path:
        raise AssertionError("锁竞争失败：critical region 在锁被占用时仍被进入")

    monkeypatch.setattr(init_module, "_copy_config", _critical_region_must_not_run)
    monkeypatch.setattr(init_module, "_copy_assets", _critical_region_must_not_run)
    monkeypatch.setattr(
        init_module,
        "apply_all_workspace_migrations",
        _critical_region_must_not_run,
    )

    holder = StateDirSingleInstanceLock(
        state_dir=workspace_root,
        lock_file_name=init_module._INIT_WORKSPACE_LOCK_FILE_NAME,
        lock_name=init_module._INIT_WORKSPACE_LOCK_NAME,
    )
    holder.acquire()
    try:
        exit_code = init_module.run_init_command(_build_args(workspace_root))
    finally:
        holder.release()

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "dayu workspace init" in captured.out
    assert "另一个 dayu-cli init" in captured.out


@pytest.mark.unit
def test_init_proceeds_after_lock_released(
    workspace_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """先竞争失败、再成功获得锁时，init 必须能正常进入临界区。"""

    entered: dict[str, bool] = {"copy_config": False, "migrations": False}

    config_dir = workspace_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = workspace_root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    def _fake_copy_config(_base_dir: Path, *, overwrite: bool) -> Path:
        del overwrite
        entered["copy_config"] = True
        return config_dir

    def _fake_copy_assets(_base_dir: Path, *, overwrite: bool) -> Path:
        del overwrite
        return assets_dir

    def _fake_migrations(*, base_dir: Path, config_dir: Path) -> None:
        del base_dir, config_dir
        entered["migrations"] = True

    monkeypatch.setattr(init_module, "_copy_config", _fake_copy_config)
    monkeypatch.setattr(init_module, "_copy_assets", _fake_copy_assets)
    monkeypatch.setattr(init_module, "apply_all_workspace_migrations", _fake_migrations)

    # 临界区已经被记录后，立即让 _prompt_provider_selection 抛 SystemExit
    # 让 init 早退，避免后续 prompt 交互。
    def _short_circuit() -> str:
        raise SystemExit(0)

    monkeypatch.setattr(init_module, "_prompt_provider_selection", _short_circuit)

    with pytest.raises(SystemExit):
        init_module.run_init_command(_build_args(workspace_root))

    assert entered["copy_config"] is True
    assert entered["migrations"] is True

    # 锁文件应该已经被释放：再起一把 lock 不应抛 RuntimeError。
    second_holder = StateDirSingleInstanceLock(
        state_dir=workspace_root,
        lock_file_name=init_module._INIT_WORKSPACE_LOCK_FILE_NAME,
        lock_name=init_module._INIT_WORKSPACE_LOCK_NAME,
    )
    second_holder.acquire()
    second_holder.release()


@pytest.mark.unit
def test_init_reset_does_not_run_when_lock_already_held(
    workspace_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """锁竞争失败时 ``--reset`` 删除目标的动作必须 **不** 被执行。

    ``--reset`` 删除的 ``.dayu/`` / ``config/`` / ``assets/`` 与临界区操作
    的是同一批 workspace 产物。如果 reset 在锁外发生，另一进程仍会在
    "我们删 → 我们复制" 的中间窗口冲进来踩踏，因此 reset 必须落在锁内。
    """

    # 预置工作区里几个 reset 目标，让删除动作可观测
    (workspace_root / "config").mkdir(parents=True, exist_ok=True)
    (workspace_root / "config" / "marker.txt").write_text("alive", encoding="utf-8")
    (workspace_root / "assets").mkdir(parents=True, exist_ok=True)
    (workspace_root / "assets" / "marker.txt").write_text("alive", encoding="utf-8")

    # 用户已确认 reset
    monkeypatch.setattr(init_module, "_confirm_workspace_reset", lambda _base: True)

    def _critical_region_must_not_run(*_args: Any, **_kwargs: Any) -> Path:
        raise AssertionError("锁竞争失败：critical region 在锁被占用时仍被进入")

    monkeypatch.setattr(init_module, "_copy_config", _critical_region_must_not_run)
    monkeypatch.setattr(init_module, "_copy_assets", _critical_region_must_not_run)
    monkeypatch.setattr(
        init_module,
        "apply_all_workspace_migrations",
        _critical_region_must_not_run,
    )

    holder = StateDirSingleInstanceLock(
        state_dir=workspace_root,
        lock_file_name=init_module._INIT_WORKSPACE_LOCK_FILE_NAME,
        lock_name=init_module._INIT_WORKSPACE_LOCK_NAME,
    )
    holder.acquire()
    try:
        exit_code = init_module.run_init_command(_build_reset_args(workspace_root))
    finally:
        holder.release()

    assert exit_code == 1
    # reset 目标必须仍然在原位 —— 锁外没有被偷偷删掉
    assert (workspace_root / "config" / "marker.txt").read_text(encoding="utf-8") == "alive"
    assert (workspace_root / "assets" / "marker.txt").read_text(encoding="utf-8") == "alive"
    captured = capsys.readouterr()
    assert "另一个 dayu-cli init" in captured.out


@pytest.mark.unit
def test_init_reset_runs_inside_lock_when_acquired(
    workspace_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """成功拿到锁后，``--reset`` 删除动作必须在锁内执行（critical region 之前）。"""

    (workspace_root / "config").mkdir(parents=True, exist_ok=True)
    (workspace_root / "config" / "marker.txt").write_text("alive", encoding="utf-8")
    (workspace_root / "assets").mkdir(parents=True, exist_ok=True)
    (workspace_root / "assets" / "marker.txt").write_text("alive", encoding="utf-8")

    monkeypatch.setattr(init_module, "_confirm_workspace_reset", lambda _base: True)

    # 临界区内：要求 reset 先发生（config / assets 已被删除），否则 _copy_config
    # 会看到旧的 marker.txt 仍在 → 立即 fail。
    def _fake_copy_config(base_dir: Path, *, overwrite: bool) -> Path:
        del overwrite
        # reset 必须已经把旧 config 清掉
        assert not (base_dir / "config" / "marker.txt").exists(), (
            "reset 应该在 _copy_config 之前在锁内执行，但旧 marker 仍在"
        )
        config_dir = base_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir

    def _fake_copy_assets(base_dir: Path, *, overwrite: bool) -> Path:
        del overwrite
        assets_dir = base_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        return assets_dir

    def _fake_migrations(*, base_dir: Path, config_dir: Path) -> None:
        del base_dir, config_dir

    monkeypatch.setattr(init_module, "_copy_config", _fake_copy_config)
    monkeypatch.setattr(init_module, "_copy_assets", _fake_copy_assets)
    monkeypatch.setattr(init_module, "apply_all_workspace_migrations", _fake_migrations)

    def _short_circuit() -> str:
        raise SystemExit(0)

    monkeypatch.setattr(init_module, "_prompt_provider_selection", _short_circuit)

    with pytest.raises(SystemExit):
        init_module.run_init_command(_build_reset_args(workspace_root))
