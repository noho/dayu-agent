"""Scene Prompt replay 行为测试。

聚焦 helper 双路语义：

- 业务错误（``AppResult.errors`` 非空）→ 走「无历史重发」，不触发 replay；
- 脏数据（``success_parser`` 抛 ``ValueError``）→ 在 Host 端带历史回放兜底；
- 一旦切到 replay 路径，下一 attempt 仍以 replay 起手，避免 zigzag；
- 取消事件在 replay 路径上同样必须透传。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

# 复用主测试文件中的 fixture / 构造器，避免重复装配。
sys.path.insert(0, str(Path(__file__).parent))
from test_write_pipeline import (  # noqa: E402  pyright: ignore[reportMissingImports]
    _build_runner,
    _make_app_result,
)

from dayu.contracts.agent_execution import ExecutionContract, ReplayHandle  # noqa: E402
from dayu.contracts.cancellation import CancelledError  # noqa: E402
from dayu.contracts.events import AppResult  # noqa: E402
from dayu.services.internal.write_pipeline.audit_rules import (  # noqa: E402
    ConfirmOutputError,
    EmptyOutputError,
    RepairOutputError,
)


_VALID_MARKDOWN = "```markdown\n## 标题\n\n足够长的合法 Markdown 正文，用以通过脏数据校验。\n```"
_DIRTY_MARKDOWN = "x"  # strip 后长度 1，且无代码块
_VALID_FACET_JSON = json.dumps(
    {
        "business_model_tags": ["平台互联网"],
        "constraint_tags": ["监管敏感"],
        "judgement_notes": "ok",
    },
    ensure_ascii=False,
)
_FACET_CATALOG = {
    "business_model_candidates": ["平台互联网"],
    "constraint_candidates": ["监管敏感"],
}
_VALID_AUDIT_JSON = json.dumps(
    {"pass": True, "violations": [], "notes": []}, ensure_ascii=False
)
_VALID_REPAIR_JSON = json.dumps(
    {
        "patches": [
            {
                "target_excerpt": "原文片段",
                "target_kind": "substring",
                "target_section_heading": "## 标题",
                "occurrence_index": 1,
                "replacement": "替换后片段",
                "reason": "占位符替换",
            }
        ],
        "notes": [],
        "resolution_mode": "patch",
    },
    ensure_ascii=False,
)
_VALID_CONFIRM_JSON = json.dumps(
    {
        "results": [
            {
                "violation_id": "v1",
                "rule": "E2",
                "excerpt": "x",
                "status": "supported",
                "reason": "ok",
                "rewrite_hint": "",
            }
        ],
        "notes": [],
    },
    ensure_ascii=False,
)


class _FakeContractExecutor:
    """按序列返回 ``run_replayable`` / ``replay`` 结果的测试桩。

    Args:
        run_results: 首发执行返回序列。
        replay_results: 回放执行返回序列。
    """

    def __init__(
        self,
        *,
        run_results: list[AppResult] | None = None,
        replay_results: list[AppResult] | None = None,
        cancel_on_replay: bool = False,
    ) -> None:
        self._run_results = list(run_results or [])
        self._replay_results = list(replay_results or [])
        self.run_calls: list[ExecutionContract] = []
        self.replay_calls: list[tuple[ReplayHandle, ExecutionContract]] = []
        self.discard_calls: list[ReplayHandle] = []
        self._handle_seq = 0
        self._cancel_on_replay = cancel_on_replay

    def _next_handle(self) -> ReplayHandle:
        self._handle_seq += 1
        return ReplayHandle(handle_id=f"handle-{self._handle_seq}")

    async def run_replayable(
        self, execution_contract: ExecutionContract
    ) -> tuple[AppResult, ReplayHandle]:
        self.run_calls.append(execution_contract)
        if not self._run_results:
            raise IndexError("run_replayable 序列耗尽")
        return self._run_results.pop(0), self._next_handle()

    async def replay(
        self, handle: ReplayHandle, execution_contract: ExecutionContract
    ) -> tuple[AppResult, ReplayHandle]:
        self.replay_calls.append((handle, execution_contract))
        if self._cancel_on_replay:
            raise CancelledError("replay 取消")
        if not self._replay_results:
            raise IndexError("replay 序列耗尽")
        return self._replay_results.pop(0), self._next_handle()

    def discard(self, handle: ReplayHandle) -> None:
        self.discard_calls.append(handle)


def _install_fake_executor(
    runner: Any, executor: _FakeContractExecutor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """切换 ScenePromptRunner 的执行入口为测试桩，并解除测试缝拦截。"""

    monkeypatch.setattr(runner._prompt_runner, "_contract_executor", executor)
    monkeypatch.setattr(runner._prompt_runner, "_prompt_agent", None)
    monkeypatch.setattr(
        "dayu.services.internal.write_pipeline.scene_executor.time.sleep",
        lambda _seconds: None,
    )


@pytest.mark.unit
def test_run_infer_prompt_replays_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """infer 首轮脏数据 → replay 成功解析。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="not json")],
        replay_results=[_make_app_result(content=_VALID_FACET_JSON)],
    )
    _install_fake_executor(runner, executor, monkeypatch)
    monkeypatch.setattr(
        runner._prompt_runner._preparer,
        "get_company_facet_catalog",
        lambda: _FACET_CATALOG,
    )

    result = runner._prompt_runner.run_infer_prompt("prompt")

    assert result.primary_facets == ["平台互联网"]
    assert len(executor.run_calls) == 1
    assert len(executor.replay_calls) == 1


@pytest.mark.unit
def test_run_infer_prompt_replay_failure_advances_to_next_attempt_via_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """replay 仍失败 → 下一 attempt 仍以 replay 起手；总失败则抛出。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="not json")],
        replay_results=[
            _make_app_result(content="still bad"),
            _make_app_result(content="still bad 2"),
            _make_app_result(content="still bad 3"),
        ],
    )
    _install_fake_executor(runner, executor, monkeypatch)
    monkeypatch.setattr(
        runner._prompt_runner._preparer,
        "get_company_facet_catalog",
        lambda: _FACET_CATALOG,
    )

    with pytest.raises(RuntimeError, match="公司级 Facet 归因输出非法"):
        runner._prompt_runner.run_infer_prompt("prompt")

    # 一次首发 + attempt0 内 1 次 replay + attempt1 起手 replay + attempt1 内再 1 次 replay = 3 次 replay
    assert len(executor.run_calls) == 1
    assert len(executor.replay_calls) == 3


@pytest.mark.unit
def test_run_audit_prompt_business_error_does_not_trigger_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """audit 业务错误 → 走「无历史重发」，不触发 replay。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[
            _make_app_result(errors=[{"error": "biz fail"}]),
            _make_app_result(content=_VALID_AUDIT_JSON),
        ],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    decision = runner._prompt_runner.run_audit_prompt("prompt")

    assert decision.passed is True
    assert len(executor.run_calls) == 2
    assert executor.replay_calls == []


@pytest.mark.unit
def test_run_repair_prompt_replays_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """repair 脏数据 → replay 成功解析。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="not json")],
        replay_results=[_make_app_result(content=_VALID_REPAIR_JSON)],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    plan, raw = runner._prompt_runner.run_repair_prompt("prompt")

    assert isinstance(plan, dict)
    assert raw == _VALID_REPAIR_JSON
    assert len(executor.replay_calls) == 1


@pytest.mark.unit
def test_run_repair_prompt_replay_exhausted_raises_repair_output_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """repair replay 仍失败 → 抛 RepairOutputError。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="bad-1")],
        replay_results=[
            _make_app_result(content="bad-2"),
            _make_app_result(content="bad-3"),
            _make_app_result(content="bad-4"),
        ],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    with pytest.raises(RepairOutputError):
        runner._prompt_runner.run_repair_prompt("prompt")


@pytest.mark.unit
def test_run_confirm_prompt_replays_on_parse_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """confirm 脏数据 → replay 成功解析。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="not json")],
        replay_results=[_make_app_result(content=_VALID_CONFIRM_JSON)],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    result = runner._prompt_runner.run_confirm_prompt("prompt")

    assert len(result.entries) == 1
    assert len(executor.replay_calls) == 1


@pytest.mark.unit
def test_run_confirm_prompt_replay_exhausted_raises_confirm_output_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """confirm replay 仍失败 → 抛 ConfirmOutputError。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="bad-1")],
        replay_results=[
            _make_app_result(content="bad-2"),
            _make_app_result(content="bad-3"),
            _make_app_result(content="bad-4"),
        ],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    with pytest.raises(ConfirmOutputError):
        runner._prompt_runner.run_confirm_prompt("prompt")


@pytest.mark.unit
@pytest.mark.parametrize(
    "method_name",
    [
        "run_write_prompt",
        "run_overview_prompt",
        "run_decision_prompt",
        "run_fix_prompt",
        "run_regenerate_prompt",
    ],
)
def test_markdown_scene_replays_on_empty_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, method_name: str
) -> None:
    """5 个 markdown scene：脏数据 → replay 成功提取正文。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content=_DIRTY_MARKDOWN)],
        replay_results=[_make_app_result(content=_VALID_MARKDOWN)],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    result = getattr(runner._prompt_runner, method_name)("prompt")

    assert "足够长的合法 Markdown 正文" in result
    assert len(executor.replay_calls) == 1


@pytest.mark.unit
def test_markdown_scene_replay_exhausted_raises_empty_output_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """markdown replay 仍空白 → 抛 EmptyOutputError。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content=_DIRTY_MARKDOWN)],
        replay_results=[
            _make_app_result(content=_DIRTY_MARKDOWN),
            _make_app_result(content=_DIRTY_MARKDOWN),
            _make_app_result(content=_DIRTY_MARKDOWN),
        ],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    with pytest.raises(EmptyOutputError):
        runner._prompt_runner.run_write_prompt("prompt")


@pytest.mark.unit
def test_markdown_scene_valid_first_shot_does_not_trigger_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """首轮即输出合法 markdown → 不触发 replay。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content=_VALID_MARKDOWN)],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    result = runner._prompt_runner.run_write_prompt("prompt")

    assert "足够长的合法 Markdown 正文" in result
    assert executor.replay_calls == []


@pytest.mark.unit
def test_replay_path_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """replay 路径触发取消时，CancelledError 必须透传。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="not json")],
        cancel_on_replay=True,
    )
    _install_fake_executor(runner, executor, monkeypatch)

    with pytest.raises(CancelledError):
        runner._prompt_runner.run_repair_prompt("prompt")


@pytest.mark.unit
def test_run_audit_prompt_replays_on_dirty_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """audit success_parser 在拿不到合法 JSON 时必须抛 ValueError，
    helper 借此触发 Host.replay 兜底；replay 返回合法 JSON 后正常解析为审计结果。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="not json")],
        replay_results=[
            _make_app_result(
                content='{"pass": true, "class": "ok", "violations": [], "notes": []}'
            )
        ],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    decision = runner._prompt_runner.run_audit_prompt("prompt")

    assert decision.passed is True
    assert decision.violations == []
    # 关键：audit 路径已接入 Host.replay 兜底，replay 必须被调用一次
    assert len(executor.replay_calls) == 1


@pytest.mark.unit
def test_run_audit_prompt_replay_failure_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """audit replay 仍返回脏数据时，helper 必须把解析失败上抛为 RuntimeError，
    而不是再用伪装 STYLE_VIOLATION 决策吞掉错误。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="not json")],
        replay_results=[
            _make_app_result(content="still not json"),
            _make_app_result(content="still bad"),
            _make_app_result(content="still bad too"),
        ],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    with pytest.raises(RuntimeError, match="审计输出解析失败"):
        runner._prompt_runner.run_audit_prompt("prompt")
    assert len(executor.replay_calls) >= 1


# 防止 lint 抱怨未使用的 cast 导入；保留以便将来按需断言。
_ = cast
_ = asyncio


@pytest.mark.unit
def test_first_shot_success_discards_unused_replay_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """首轮成功解析 → 必须释放未消费的 replay 句柄，避免 Host stash 泄漏。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content=_VALID_MARKDOWN)],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    runner._prompt_runner.run_write_prompt("prompt")

    assert len(executor.discard_calls) == 1
    # 释放的句柄就是首发返回的句柄
    assert executor.discard_calls[0].handle_id == "handle-1"


@pytest.mark.unit
def test_replay_success_discards_new_replay_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """replay 成功 → 释放 replay 颁发的新句柄。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content=_DIRTY_MARKDOWN)],
        replay_results=[_make_app_result(content=_VALID_MARKDOWN)],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    runner._prompt_runner.run_write_prompt("prompt")

    # 首发返回 handle-1，被消费做 replay；replay 返回 handle-2，本次成功后释放
    assert len(executor.discard_calls) == 1
    assert executor.discard_calls[0].handle_id == "handle-2"


@pytest.mark.unit
def test_replay_contract_disables_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """replay 路径必须把 ``replay_disable_tools=True`` 写入契约，强制收口。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="not json")],
        replay_results=[_make_app_result(content=_VALID_REPAIR_JSON)],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    runner._prompt_runner.run_repair_prompt("prompt")

    assert len(executor.replay_calls) == 1
    _handle, replay_contract = executor.replay_calls[0]
    assert replay_contract.message_inputs.replay_disable_tools is True
    assert replay_contract.message_inputs.replay_from is not None


@pytest.mark.unit
def test_business_error_attempt_discards_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """业务错误 attempt → 新 handle 不会被消费，必须立即 discard 释放 stash。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[
            _make_app_result(errors=[{"error": "biz fail"}]),
            _make_app_result(content=_VALID_AUDIT_JSON),
        ],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    runner._prompt_runner.run_audit_prompt("prompt")

    # 第一次失败 attempt 释放 handle-1，第二次成功 attempt 释放 handle-2
    assert [h.handle_id for h in executor.discard_calls] == ["handle-1", "handle-2"]


@pytest.mark.unit
def test_replay_exhausted_discards_final_new_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """replay 预算耗尽抛错前，最后一次 attempt 的 new_handle 必须释放。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content="bad-1")],
        replay_results=[
            _make_app_result(content="bad-2"),
            _make_app_result(content="bad-3"),
            _make_app_result(content="bad-4"),
        ],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    with pytest.raises(RepairOutputError):
        runner._prompt_runner.run_repair_prompt("prompt")

    # 中间 attempt 的 new_handle 由下一 attempt 的 replay() 消费出 stash；
    # 仅最后一次 attempt 的 new_handle 没有任何消费方，必须由 helper 显式 discard。
    # handle 序列：1=首发, 2/3/4=三次 replay；最后释放的是 handle-4。
    assert any(h.handle_id == "handle-4" for h in executor.discard_calls)


@pytest.mark.unit
def test_run_prepared_scene_prompt_sync_entry_discards_handle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """同步入口 ``run_prepared_scene_prompt`` 退出前必须释放 stash 句柄。"""

    runner = _build_runner(tmp_path)
    executor = _FakeContractExecutor(
        run_results=[_make_app_result(content=_VALID_MARKDOWN)],
    )
    _install_fake_executor(runner, executor, monkeypatch)

    prepared_scene = runner._prompt_runner._preparer.get_or_create_write_scene()
    runner._prompt_runner.run_prepared_scene_prompt(
        prepared_scene=prepared_scene, prompt_text="prompt"
    )

    assert len(executor.discard_calls) == 1
    assert executor.discard_calls[0].handle_id == "handle-1"
