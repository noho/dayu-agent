"""conversation_memory 测试。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from typing import cast

import pytest

from dayu.contracts.agent_execution import AgentCreateArgs
from dayu.contracts.agent_types import AgentMessage
from dayu.contracts.infrastructure import PromptAssetStoreProtocol
from dayu.contracts.model_config import ModelConfig
from dayu.contracts.protocols import PromptToolExecutorProtocol
from dayu.contracts.prompt_assets import SceneManifestAsset, TaskPromptContractAsset
from dayu.contracts.tool_configs import DocToolLimits, FinsToolLimits, WebToolsConfig
from dayu.contracts.toolset_config import ToolsetConfigSnapshot, build_toolset_config_snapshot
from dayu.engine.events import EventType, StreamEvent
from dayu.engine.tool_registry import ToolRegistry
from dayu.execution.runtime_config import AgentRuntimeConfig, OpenAIRunnerRuntimeConfig
from dayu.execution.options import (
    ConversationMemorySettings,
    ExecutionOptions,
    ResolvedExecutionOptions,
    TraceSettings,
    build_base_execution_options,
    merge_execution_options,
    resolve_conversation_memory_settings,
)
from dayu.host.conversation_memory import (
    ConversationCompactionResult,
    ConversationPinnedStatePatch,
    DefaultConversationMemoryManager,
    DefaultEpisodicMemoryCompressor,
    DefaultWorkingMemoryPolicy,
    _estimate_tokens,
    _truncate_text_to_token_budget,
)
from dayu.host.conversation_runtime import (
    ConversationCompactionAgentProtocol,
    ConversationCompactionAgentHandle,
    ConversationCompactionRequest,
    ConversationCompactionSceneProtocol,
)
from dayu.host.conversation_store import (
    ConversationEpisodeSummary,
    ConversationPinnedState,
    ConversationToolUseSummary,
    ConversationTranscript,
    ConversationTurnRecord,
    FileConversationStore,
)


def _build_toolset_configs(
    *,
    doc_tool_limits: DocToolLimits | None = None,
    fins_tool_limits: FinsToolLimits | None = None,
    web_tools_config: WebToolsConfig | None = None,
) -> tuple[ToolsetConfigSnapshot, ...]:
    """构造测试用通用 toolset 配置快照。"""

    return tuple(
        snapshot
        for snapshot in (
            build_toolset_config_snapshot("doc", doc_tool_limits),
            build_toolset_config_snapshot("fins", fins_tool_limits),
            build_toolset_config_snapshot("web", web_tools_config),
        )
        if snapshot is not None
    )
from dayu.host.scene_preparer import PreparedSceneState
from dayu.prompting import SceneDefinition, SceneModelDefinition
from dayu.log import Log


class _FakeAsyncAgent:
    """测试用底层 AsyncAgent。"""

    def __init__(self, events_per_call: list[list[StreamEvent]]) -> None:
        self._events_per_call = list(events_per_call)
        self.calls: list[list[AgentMessage]] = []

    async def run_messages(self, messages: list[AgentMessage], *, session_id: str | None = None):
        """模拟 run_messages。

        Args:
            messages: 送模消息。
            session_id: 会话 ID。

        Returns:
            异步事件流。

        Raises:
            无。
        """

        del session_id
        self.calls.append(list(messages))
        events = self._events_per_call.pop(0)
        for event in events:
            yield event


class _FakeRuntime:
    """测试用 Runtime。"""

    def __init__(
        self,
        *,
        resolved_options: ResolvedExecutionOptions,
        compaction_agent: _FakeAsyncAgent | None = None,
    ) -> None:
        self._resolved_options = resolved_options
        self._compaction_agent = compaction_agent
        self.compaction_requests: list[ConversationCompactionRequest] = []

    def resolve_options(self, execution_options=None) -> ResolvedExecutionOptions:
        """返回固定执行选项。"""

        del execution_options
        return self._resolved_options

    def prepare_compaction_scene(
        self,
        scene_name: str,
        execution_options: ExecutionOptions | None = None,
        web_tools_config: WebToolsConfig | None = None,
    ) -> PreparedSceneState:
        """返回测试用静态 scene 状态。"""

        del execution_options
        del web_tools_config
        return _build_prepared_scene(scene_name=scene_name, settings=self._resolved_options.conversation_memory_settings)

    def prepare_compaction_agent(
        self,
        prepared_scene: ConversationCompactionSceneProtocol,
        request: ConversationCompactionRequest,
    ) -> ConversationCompactionAgentHandle:
        """返回测试用 Agent。"""

        self.compaction_requests.append(request)
        if self._compaction_agent is None:
            raise AssertionError("当前测试未提供 compaction agent")
        return ConversationCompactionAgentHandle(
            agent=cast(ConversationCompactionAgentProtocol, self._compaction_agent),
            system_prompt=f"SYS:{prepared_scene.scene_name}",
        )


class _FakeCompressor:
    """测试用异步压缩器。"""

    def __init__(self, *, delay: float = 0.0, title_prefix: str = "压缩阶段") -> None:
        self.delay = delay
        self.title_prefix = title_prefix
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    async def compress(self, *, session_id: str, transcript, turns, settings):
        """返回固定压缩结果。"""

        del transcript
        del settings
        self.calls.append((session_id, tuple(turn.turn_id for turn in turns)))
        if self.delay:
            await asyncio.sleep(self.delay)
        return ConversationCompactionResult(
            episode_summary=ConversationEpisodeSummary(
                episode_id=f"ep_{len(self.calls)}",
                start_turn_id=turns[0].turn_id,
                end_turn_id=turns[-1].turn_id,
                title=f"{self.title_prefix}{len(self.calls)}",
                goal="阶段目标",
                confirmed_facts=("已确认事实",),
            ),
            pinned_state_patch=ConversationPinnedStatePatch(
                current_goal="跟踪公司最新变化",
                open_questions=("下季度指引",),
            ),
        )


def _build_resolved_options(settings: ConversationMemorySettings | None = None) -> ResolvedExecutionOptions:
    """构建测试用执行选项。"""

    return ResolvedExecutionOptions(
        model_name="test-model",
        temperature=0.1,
        runner_running_config=OpenAIRunnerRuntimeConfig(),
        agent_running_config=AgentRuntimeConfig(),
        toolset_configs=_build_toolset_configs(
            doc_tool_limits=DocToolLimits(),
            fins_tool_limits=FinsToolLimits(),
            web_tools_config=WebToolsConfig(),
        ),
        trace_settings=TraceSettings(enabled=False, output_dir=Path("/tmp")),
        conversation_memory_settings=settings or ConversationMemorySettings(),
    )


def _build_prompt_asset_store() -> PromptAssetStoreProtocol:
    """构造最小 PromptAssetStoreProtocol 测试桩。"""

    scene_manifest = cast(SceneManifestAsset, {"scene": "interactive", "fragments": []})
    task_contract = cast(TaskPromptContractAsset, {})
    return cast(
        PromptAssetStoreProtocol,
        SimpleNamespace(
            load_scene_manifest=lambda scene_name: {**scene_manifest, "scene": scene_name},
            load_fragment_template=lambda fragment_path, required=True: "",
            load_task_prompt=lambda task_name: "",
            load_task_prompt_contract=lambda task_name: task_contract,
        ),
    )


def _message_role(message: AgentMessage) -> str:
    """安全读取 AgentMessage.role。"""

    return str(cast(dict[str, object], message).get("role") or "")


def _message_content(message: AgentMessage) -> str:
    """安全读取 AgentMessage.content。"""

    return str(cast(dict[str, object], message).get("content") or "")


def _build_prepared_scene(
    *,
    scene_name: str = "interactive",
    settings: ConversationMemorySettings | None = None,
    max_context_tokens: int = 10000,
) -> PreparedSceneState:
    """构建测试用静态 scene 状态。"""

    resolved_options = _build_resolved_options(settings)
    model_config = cast(
        ModelConfig,
        {
            "runner_type": "openai_compatible",
            "name": "test-model",
            "max_context_tokens": max_context_tokens,
        },
    )
    return PreparedSceneState(
        scene_name=scene_name,
        scene_definition=SceneDefinition(
            name=scene_name,
            model=SceneModelDefinition(default_name="test-model"),
            version="v1",
            description="test",
        ),
        resolved_options=resolved_options,
        model_config=model_config,
        prompt_asset_store=_build_prompt_asset_store(),
        tool_registry=cast(PromptToolExecutorProtocol, ToolRegistry()),
        agent_create_args=AgentCreateArgs(
            runner_type="openai_compatible",
            model_name="test-model",
            max_context_tokens=max_context_tokens,
        ),
        conversation_memory_settings=settings or ConversationMemorySettings(),
    )


def _build_turn(index: int) -> ConversationTurnRecord:
    """构建测试用 turn。"""

    return ConversationTurnRecord(
        turn_id=f"turn_{index}",
        scene_name="interactive",
        user_text=f"问题 {index}",
        assistant_final=f"回答 {index}",
        tool_uses=(
            ConversationToolUseSummary(
                name="search_web",
                arguments={"query": f"q{index}"},
                result_summary=f"result {index}",
            ),
        ),
    )


@pytest.mark.unit
def test_estimate_tokens_counts_wide_chars_more_conservatively() -> None:
    """验证宽字符 token 估算不会继续按半字符低估。"""

    assert _estimate_tokens("abcd") == 2
    assert _estimate_tokens("中文测试") == 4
    assert _estimate_tokens("ab中文c") == 4


@pytest.mark.unit
def test_truncate_text_to_token_budget_respects_mixed_width_budget() -> None:
    """验证 mixed-width 文本截断后仍维持保守预算语义。"""

    truncated = _truncate_text_to_token_budget("ab中文cd中文ef中文gh", 9)

    assert truncated.endswith("...<truncated>")
    assert _estimate_tokens(truncated) <= 9


@pytest.mark.unit
def test_build_base_execution_options_keeps_default_conversation_memory_settings(
    tmp_path: Path,
) -> None:
    """验证 base execution options 仅保留 conversation_memory default。"""

    base_options = build_base_execution_options(
        workspace_dir=tmp_path,
        run_config={
            "conversation_memory": {
                "default": {
                    "working_memory_token_budget_cap": 7000,
                    "episodic_memory_token_budget_floor": 2100,
                    "episodic_memory_token_budget_cap": 2100,
                },
            }
        },
    )

    assert base_options.conversation_memory_settings.working_memory_token_budget_cap == 7000
    assert base_options.conversation_memory_config.default.episodic_memory_token_budget_floor == 2100
    assert base_options.conversation_memory_config.default.episodic_memory_token_budget_cap == 2100


@pytest.mark.unit
def test_build_base_execution_options_keeps_typed_default_run_config_values(tmp_path: Path) -> None:
    """验证强类型默认配置真源不会改变基础运行默认值。"""

    base_options = build_base_execution_options(
        workspace_dir=tmp_path,
        run_config={},
    )

    assert cast(OpenAIRunnerRuntimeConfig, base_options.runner_running_config).tool_timeout_seconds == 90.0
    assert base_options.agent_running_config.max_iterations == 16
    assert base_options.conversation_memory_settings.working_memory_token_budget_cap == 12000
    assert base_options.conversation_memory_settings.episodic_memory_token_budget_cap == 12000


@pytest.mark.unit
def test_merge_execution_options_keeps_default_conversation_memory_settings(
    tmp_path: Path,
) -> None:
    """验证 merge_execution_options 不再按模型名切换 conversation memory 配置。"""

    base_options = build_base_execution_options(
        workspace_dir=tmp_path,
        run_config={
            "conversation_memory": {
                "default": {
                    "working_memory_token_budget_cap": 6500,
                    "episodic_memory_token_budget_floor": 2600,
                    "episodic_memory_token_budget_cap": 2600,
                },
            }
        },
    )

    resolved = merge_execution_options(
        base_options=base_options,
        workspace_dir=tmp_path,
        execution_options=ExecutionOptions(model_name="deepseek-thinking"),
    )

    assert resolved.model_name == "deepseek-thinking"
    assert resolved.conversation_memory_settings.working_memory_token_budget_cap == 6500
    assert resolved.conversation_memory_settings.episodic_memory_token_budget_floor == 2600
    assert resolved.conversation_memory_settings.episodic_memory_token_budget_cap == 2600


@pytest.mark.unit
def test_build_base_execution_options_uses_default_when_model_has_no_runtime_hints(tmp_path: Path) -> None:
    """验证默认 conversation memory 配置会保留到合并结果。"""

    base_options = build_base_execution_options(
        workspace_dir=tmp_path,
        run_config={
            "conversation_memory": {
                "default": {
                    "working_memory_token_budget_cap": 9000,
                    "episodic_memory_token_budget_floor": 2600,
                    "episodic_memory_token_budget_cap": 2600,
                },
            }
        },
    )

    resolved = merge_execution_options(
        base_options=base_options,
        workspace_dir=tmp_path,
        execution_options=ExecutionOptions(model_name="qwen3-thinking"),
    )

    assert resolved.model_name == "qwen3-thinking"
    assert resolved.conversation_memory_settings.working_memory_token_budget_cap == 9000
    assert resolved.conversation_memory_settings.episodic_memory_token_budget_floor == 2600
    assert resolved.conversation_memory_settings.episodic_memory_token_budget_cap == 2600


@pytest.mark.unit
def test_resolve_conversation_memory_settings_merges_model_runtime_hints(tmp_path: Path) -> None:
    """验证模型 runtime hints 可覆盖 conversation memory 默认公式。"""

    base_options = build_base_execution_options(
        workspace_dir=tmp_path,
        run_config={
            "conversation_memory": {
                "default": {
                    "working_memory_token_budget_cap": 9000,
                    "episodic_memory_token_budget_floor": 2600,
                    "episodic_memory_token_budget_cap": 2600,
                }
            }
        },
    )

    settings = resolve_conversation_memory_settings(
        conversation_memory_config=base_options.conversation_memory_config,
        model_config={
            "runtime_hints": {
                "conversation_memory": {
                    "working_memory_token_budget_cap": 20000,
                    "episodic_memory_token_budget_floor": 4200,
                    "episodic_memory_token_budget_cap": 4200,
                }
            }
        },
    )

    assert settings.working_memory_token_budget_cap == 20000
    assert settings.episodic_memory_token_budget_floor == 4200
    assert settings.episodic_memory_token_budget_cap == 4200


@pytest.mark.unit
def test_working_memory_policy_uses_uncompacted_tail_and_budget() -> None:
    """验证 working memory 只从未压缩 tail 中按预算选 turn。"""

    policy = DefaultWorkingMemoryPolicy()
    transcript = ConversationTranscript.create_empty("sess_1")
    for index in range(1, 8):
        transcript = transcript.append_turn(_build_turn(index))
    transcript = transcript.replace_memory(
        pinned_state=ConversationPinnedState(),
        episodes=(),
        compacted_turn_count=3,
    )
    settings = ConversationMemorySettings(
        working_memory_max_turns=2,
        working_memory_token_budget_ratio=0.01,
        working_memory_token_budget_floor=120,
        working_memory_token_budget_cap=120,
    )

    selected = policy.select_turns(transcript, settings=settings, max_context_tokens=1000)

    assert tuple(turn.turn_id for turn in selected) == ("turn_6", "turn_7")


@pytest.mark.unit
def test_working_memory_policy_does_not_overrun_small_context_budget() -> None:
    """验证小上下文模型下最新 turn 仍会以最小保真视图保留。"""

    policy = DefaultWorkingMemoryPolicy()
    transcript = ConversationTranscript.create_empty("sess_1").append_turn(
        ConversationTurnRecord(
            turn_id="turn_1",
            scene_name="interactive",
            user_text="x" * 4000,
            assistant_final="",
        )
    )

    selected = policy.select_turns(transcript, settings=ConversationMemorySettings(), max_context_tokens=1024)

    assert len(selected) == 1
    assert selected[0].turn_id == "turn_1"
    assert selected[0].user_text == "x" * 4000
    assert selected[0].assistant_text == ""


@pytest.mark.unit
def test_working_memory_policy_keeps_latest_turn_as_truncated_view_when_single_turn_exceeds_budget() -> None:
    """验证最新 turn 即使超预算也会以裁剪视图保留，而不是整轮消失。"""

    policy = DefaultWorkingMemoryPolicy()
    transcript = ConversationTranscript.create_empty("sess_1").append_turn(
        ConversationTurnRecord(
            turn_id="turn_1",
            scene_name="interactive",
            user_text="泡泡玛特做小家电是什么情况？介绍一下。",
            assistant_final=(
                "2025年3月25日，泡泡玛特首席运营官司德在2025年业绩发布会上宣布，"
                "家电产品将于4月正式面市。" * 50
            ),
            tool_uses=(
                ConversationToolUseSummary(
                    name="search_web",
                    arguments={"query": "泡泡玛特 小家电"},
                    result_summary="result " * 400,
                ),
            ),
        )
    )
    settings = ConversationMemorySettings(
        working_memory_max_turns=6,
        working_memory_token_budget_ratio=0.01,
        working_memory_token_budget_floor=120,
        working_memory_token_budget_cap=120,
    )

    selected = policy.select_turns(transcript, settings=settings, max_context_tokens=1000)

    assert len(selected) == 1
    assert selected[0].turn_id == "turn_1"
    assert selected[0].user_text == "泡泡玛特做小家电是什么情况？介绍一下。"
    assert "2025年3月25日" in selected[0].assistant_text
    assert "历史工具摘要" not in selected[0].assistant_text
    assert selected[0].assistant_text.endswith("...<truncated>")


@pytest.mark.unit
def test_default_conversation_memory_manager_builds_memory_block_and_raw_tail(tmp_path: Path) -> None:
    """验证 memory manager 会把 pinned state、episodes 和 raw tail 一起编译进消息。"""

    settings = ConversationMemorySettings(
        episodic_memory_token_budget_ratio=0.0,
        episodic_memory_token_budget_floor=400,
        episodic_memory_token_budget_cap=400,
        working_memory_max_turns=2,
    )
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    manager = DefaultConversationMemoryManager(
        runtime,
        conversation_store=FileConversationStore(tmp_path / "conversations"),
    )
    transcript = ConversationTranscript.create_empty("sess_1")
    for index in range(1, 4):
        transcript = transcript.append_turn(_build_turn(index))
    transcript = transcript.replace_memory(
        pinned_state=ConversationPinnedState(
            current_goal="跟踪公司最新变化",
            confirmed_subjects=("PDD",),
        ),
        episodes=(
            ConversationEpisodeSummary(
                episode_id="ep_1",
                start_turn_id="turn_1",
                end_turn_id="turn_1",
                title="确认对象",
                goal="确认当前分析主题",
                confirmed_facts=("当前对象是 PDD",),
            ),
        ),
        compacted_turn_count=1,
    )

    messages = manager.build_messages(
        prepared_scene=_build_prepared_scene(settings=settings),
        transcript=transcript,
        system_prompt="SYS:interactive",
        user_text="最新信息更新",
    )

    assert messages[0] == {"role": "system", "content": "SYS:interactive"}
    assert _message_role(messages[1]) == "system"
    assert "[Conversation Memory]" in _message_content(messages[1])
    assert "Pinned State" in _message_content(messages[1])
    assert "Episode Summaries" in _message_content(messages[1])
    assert messages[-1] == {"role": "user", "content": "最新信息更新"}
    assert messages[2] == {"role": "user", "content": "问题 2"}


@pytest.mark.unit
def test_compaction_candidate_uses_current_scene_context_budget(tmp_path: Path) -> None:
    """验证 compaction 触发阈值与当前 scene 的上下文预算一致。"""

    settings = ConversationMemorySettings()
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    manager = DefaultConversationMemoryManager(
        runtime,
        conversation_store=FileConversationStore(tmp_path / "conversations"),
    )
    transcript = ConversationTranscript.create_empty("sess_1")
    for index in range(1, 7):
        transcript = transcript.append_turn(
            ConversationTurnRecord(
                turn_id=f"turn_{index}",
                scene_name="interactive",
                user_text=f"turn-{index}-" + ("x" * 794),
                assistant_final="",
            )
        )

    candidate_turns = manager._select_compaction_candidate(
        transcript=transcript,
        settings=settings,
        max_context_tokens=2048,
    )

    assert tuple(turn.turn_id for turn in candidate_turns) == ("turn_1", "turn_2")


@pytest.mark.unit
def test_default_episodic_memory_compressor_parses_json_payload() -> None:
    """验证默认压缩器可从 compaction scene 的最终回答中解析结构化 JSON。"""

    fake_agent = _FakeAsyncAgent(
        [[
            StreamEvent(
                type=EventType.FINAL_ANSWER,
                data={
                    "content": json.dumps(
                        {
                            "episode_summary": {
                                "title": "跟踪最新动态",
                                "goal": "更新公司信息",
                                "completed_actions": ["搜索最近新闻"],
                                "confirmed_facts": ["公司已发布最新财报"],
                                "user_constraints": ["优先使用公开来源"],
                                "open_questions": ["管理层指引变化"],
                                "next_step": "补充财报细节",
                                "tool_findings": ["找到财报发布日期"],
                            },
                            "pinned_state_patch": {
                                "current_goal": "跟踪公司最新变化",
                                "confirmed_subjects": ["PDD"],
                                "user_constraints": ["优先使用公开来源"],
                                "open_questions": ["管理层指引变化"],
                            },
                        },
                        ensure_ascii=False,
                    )
                },
                metadata={},
            ),
        ]]
    )
    runtime = _FakeRuntime(
        resolved_options=_build_resolved_options(),
        compaction_agent=fake_agent,
    )
    compressor = DefaultEpisodicMemoryCompressor(runtime)
    transcript = ConversationTranscript.create_empty("sess_1").append_turn(_build_turn(1))

    result = asyncio.run(
        compressor.compress(
            session_id="sess_1",
            transcript=transcript,
            turns=(transcript.turns[0],),
            settings=ConversationMemorySettings(),
        )
    )

    assert result is not None
    assert result.episode_summary.title == "跟踪最新动态"
    assert result.pinned_state_patch.current_goal == "跟踪公司最新变化"
    assert fake_agent.calls
    assert _message_role(fake_agent.calls[0][0]) == "system"
    assert runtime.compaction_requests == [ConversationCompactionRequest(session_id="sess_1.compaction")]


@pytest.mark.unit
def test_default_episodic_memory_compressor_rejects_empty_summary_title() -> None:
    """验证空标题 episode summary 会被视为无效压缩结果。"""

    fake_agent = _FakeAsyncAgent(
        [[
            StreamEvent(
                type=EventType.FINAL_ANSWER,
                data={
                    "content": json.dumps(
                        {
                            "episode_summary": {
                                "title": "   ",
                                "goal": "更新公司信息",
                            },
                            "pinned_state_patch": {
                                "current_goal": "跟踪公司最新变化",
                            },
                        },
                        ensure_ascii=False,
                    )
                },
                metadata={},
            ),
        ]]
    )
    runtime = _FakeRuntime(
        resolved_options=_build_resolved_options(),
        compaction_agent=fake_agent,
    )
    compressor = DefaultEpisodicMemoryCompressor(runtime)
    transcript = ConversationTranscript.create_empty("sess_1").append_turn(_build_turn(1))

    result = asyncio.run(
        compressor.compress(
            session_id="sess_1",
            transcript=transcript,
            turns=(transcript.turns[0],),
            settings=ConversationMemorySettings(),
        )
    )

    assert result is None


@pytest.mark.unit
def test_conversation_memory_manager_compacts_in_background_and_replans(tmp_path: Path) -> None:
    """验证后台 compaction 会取消旧任务并按最新 transcript 重排。"""

    settings = ConversationMemorySettings(compaction_trigger_turn_count=2, compaction_tail_preserve_turns=1)
    store = FileConversationStore(tmp_path / "conversations")
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    compressor = _FakeCompressor(delay=0.02)
    manager = DefaultConversationMemoryManager(
        runtime,
        conversation_store=store,
        episodic_memory_compressor=compressor,
    )
    transcript = ConversationTranscript.create_empty("sess_1")
    for index in range(1, 5):
        transcript = transcript.append_turn(_build_turn(index))
    store.save(transcript)

    async def _run() -> None:
        manager.schedule_compaction(
            session_id="sess_1",
            prepared_scene=_build_prepared_scene(settings=settings),
            transcript=transcript,
        )
        await asyncio.sleep(0)
        current = store.load("sess_1")
        assert current is not None
        updated = current.append_turn(_build_turn(5))
        store.save(updated, expected_revision=current.revision)
        await manager.cancel_pending_compaction("sess_1")
        manager.schedule_compaction(
            session_id="sess_1",
            prepared_scene=_build_prepared_scene(settings=settings),
            transcript=updated,
        )
        await manager.wait_for_session("sess_1")

    asyncio.run(_run())

    loaded = store.load("sess_1")

    assert loaded is not None
    assert loaded.compacted_turn_count >= 4
    assert loaded.episodes
    assert loaded.pinned_state.current_goal == "跟踪公司最新变化"
    assert compressor.calls[-1][1][-1] == "turn_4"


@pytest.mark.unit
def test_prepare_transcript_compacts_persisted_tail_before_resumed_turn(tmp_path: Path) -> None:
    """验证进程重启后恢复同 session 的首轮会先同步压缩已落盘 transcript。"""

    settings = ConversationMemorySettings()
    store = FileConversationStore(tmp_path / "conversations")
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    compressor = _FakeCompressor()
    manager = DefaultConversationMemoryManager(
        runtime,
        conversation_store=store,
        episodic_memory_compressor=compressor,
    )
    transcript = ConversationTranscript.create_empty("sess_1")
    for index in range(1, 7):
        transcript = transcript.append_turn(
            ConversationTurnRecord(
                turn_id=f"turn_{index}",
                scene_name="interactive",
                user_text=f"turn-{index}-" + ("x" * 794),
                assistant_final="",
            )
        )
    store.save(transcript)

    prepared = asyncio.run(
        manager.prepare_transcript(
            session_id="sess_1",
            prepared_scene=_build_prepared_scene(settings=settings, max_context_tokens=2048),
            transcript=transcript,
        )
    )

    persisted = store.load("sess_1")

    assert prepared.compacted_turn_count == 2
    assert prepared.episodes
    assert compressor.calls == [("sess_1", ("turn_1", "turn_2"))]
    assert persisted is not None
    assert persisted.compacted_turn_count == 2


@pytest.mark.unit
def test_conversation_memory_emits_compaction_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ConversationMemory 关键 compaction 动作应输出 verbose/debug 日志。"""

    settings = ConversationMemorySettings()
    store = FileConversationStore(tmp_path / "conversations")
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    compressor = _FakeCompressor()
    manager = DefaultConversationMemoryManager(
        runtime,
        conversation_store=store,
        episodic_memory_compressor=compressor,
    )
    transcript = ConversationTranscript.create_empty("sess_log")
    for index in range(1, 7):
        transcript = transcript.append_turn(
            ConversationTurnRecord(
                turn_id=f"turn_{index}",
                scene_name="interactive",
                user_text=f"turn-{index}-" + ("x" * 794),
                assistant_final="",
            )
        )
    store.save(transcript)

    verbose_mock = Mock()
    debug_mock = Mock()
    monkeypatch.setattr(Log, "verbose", verbose_mock)
    monkeypatch.setattr(Log, "debug", debug_mock)

    prepared = asyncio.run(
        manager.prepare_transcript(
            session_id="sess_log",
            prepared_scene=_build_prepared_scene(settings=settings, max_context_tokens=2048),
            transcript=transcript,
        )
    )
    manager.schedule_compaction(
        session_id="sess_log",
        prepared_scene=_build_prepared_scene(settings=settings, max_context_tokens=100000),
        transcript=prepared,
    )

    verbose_messages = [call.args[0] for call in verbose_mock.call_args_list]
    debug_messages = [call.args[0] for call in debug_mock.call_args_list]

    assert any("准备同步压缩 transcript" in message for message in verbose_messages)
    assert any("同步压缩写回 transcript" in message for message in verbose_messages)
    assert any("无需调度 compaction" in message for message in debug_messages)
