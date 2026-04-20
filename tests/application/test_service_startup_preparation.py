"""Service 启动准备公共 API 测试。"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from dayu.execution.options import ExecutionOptions, ResolvedExecutionOptions
from dayu.fins.service_runtime import DefaultFinsRuntime
from dayu.host import Host
from dayu.services.scene_execution_acceptance import SceneExecutionAcceptancePreparer
from dayu.services.startup_preparation import prepare_host_runtime_dependencies
from dayu.startup.workspace import WorkspaceResources


@pytest.mark.unit
def test_prepare_host_runtime_dependencies_runs_unified_startup_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """共享 Host 启动准备应在装配完成后执行统一 startup recovery。"""

    fake_paths = SimpleNamespace(
        workspace_root=tmp_path,
        config_root=tmp_path / "config",
        output_dir=tmp_path / "output",
    )
    fake_workspace = cast(WorkspaceResources, object())
    fake_model_catalog = object()
    fake_default_execution_options = cast(ResolvedExecutionOptions, object())
    fake_scene_preparer = cast(SceneExecutionAcceptancePreparer, object())
    fake_fins_runtime = cast(DefaultFinsRuntime, object())
    fake_host = cast(Host, object())
    recover_calls: list[tuple[Host, str, str]] = []

    monkeypatch.setattr(
        "dayu.services.startup_preparation.prepare_startup_paths",
        lambda **_kwargs: fake_paths,
    )
    monkeypatch.setattr(
        "dayu.services.startup_preparation.prepare_config_file_resolver",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "dayu.services.startup_preparation.prepare_config_loader",
        lambda **_kwargs: SimpleNamespace(load_run_config=lambda: SimpleNamespace()),
    )
    monkeypatch.setattr(
        "dayu.services.startup_preparation.prepare_prompt_asset_store",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "dayu.services.startup_preparation.prepare_workspace_resources",
        lambda **_kwargs: fake_workspace,
    )
    monkeypatch.setattr(
        "dayu.services.startup_preparation.prepare_model_catalog",
        lambda **_kwargs: fake_model_catalog,
    )
    monkeypatch.setattr(
        "dayu.services.startup_preparation.prepare_default_execution_options",
        lambda **_kwargs: fake_default_execution_options,
    )
    monkeypatch.setattr(
        "dayu.services.startup_preparation.prepare_scene_execution_acceptance_preparer",
        lambda **_kwargs: fake_scene_preparer,
    )
    monkeypatch.setattr(
        "dayu.services.startup_preparation.prepare_fins_runtime",
        lambda **_kwargs: fake_fins_runtime,
    )
    monkeypatch.setattr(
        "dayu.services.startup_preparation.resolve_host_config",
        lambda **_kwargs: SimpleNamespace(
            store_path=tmp_path / "host.sqlite3",
            lane_config={"llm_api": 1},
            pending_turn_resume_max_attempts=3,
        ),
    )
    monkeypatch.setattr("dayu.services.startup_preparation.Host", lambda **_kwargs: fake_host)
    monkeypatch.setattr(
        "dayu.services.startup_preparation.recover_host_startup_state",
        lambda host_admin_service, *, runtime_label, log_module: recover_calls.append(
            (cast(Host, host_admin_service.host), runtime_label, log_module)
        ),
    )

    prepared = prepare_host_runtime_dependencies(
        workspace_root=tmp_path,
        config_root=tmp_path / "config",
        execution_options=ExecutionOptions(),
        runtime_label="Shared Host runtime",
        log_module="APP.TEST",
    )

    assert prepared.workspace is fake_workspace
    assert prepared.default_execution_options is fake_default_execution_options
    assert prepared.scene_execution_acceptance_preparer is fake_scene_preparer
    assert prepared.host is fake_host
    assert prepared.fins_runtime is fake_fins_runtime
    assert recover_calls == [(fake_host, "Shared Host runtime", "APP.TEST")]
