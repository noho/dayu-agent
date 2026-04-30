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
from dayu.host.conversation_session_archive import (
    ConversationHistoryArchive,
    ConversationSessionArchive,
)
from dayu.host.conversation_store import (
    ConversationEpisodeSummary,
    ConversationPinnedState,
    ConversationToolUseSummary,
    ConversationTranscript,
    ConversationTurnRecord,
    FileConversationSessionArchiveStore,
)


def _archive_from_transcript(transcript: ConversationTranscript) -> ConversationSessionArchive:
    """把测试用 transcript 包装成 archive，便于直接 store.save 预置。"""

    from dataclasses import replace

    base = ConversationSessionArchive.create_empty(transcript.session_id)
    runtime = replace(transcript, revision=base.revision)
    return ConversationSessionArchive(
        session_id=base.session_id,
        revision=base.revision,
        created_at=base.created_at,
        updated_at=base.updated_at,
        runtime_transcript=runtime,
        history_archive=ConversationHistoryArchive.create_empty(transcript.session_id),
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
                    "memory_token_budget_ratio": 0.10,
                    "memory_token_budget_floor": 4000,
                    "memory_token_budget_cap": 7000,
                    "recent_turns_floor": 2,
                    "compaction_trigger_context_ratio": 0.60,
                },
            }
        },
    )

    assert base_options.conversation_memory_settings.memory_token_budget_cap == 7000
    assert base_options.conversation_memory_config.default.memory_token_budget_floor == 4000
    assert base_options.conversation_memory_config.default.memory_token_budget_cap == 7000


@pytest.mark.unit
def test_build_base_execution_options_keeps_typed_default_run_config_values(tmp_path: Path) -> None:
    """验证强类型默认配置真源不会改变基础运行默认值。"""

    base_options = build_base_execution_options(
        workspace_dir=tmp_path,
        run_config={},
    )

    assert cast(OpenAIRunnerRuntimeConfig, base_options.runner_running_config).tool_timeout_seconds == 90.0
    assert base_options.agent_running_config.max_iterations == 16
    assert base_options.conversation_memory_settings.memory_token_budget_cap == 60000
    assert base_options.conversation_memory_settings.memory_token_budget_floor == 4000


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
                    "memory_token_budget_ratio": 0.10,
                    "memory_token_budget_floor": 4000,
                    "memory_token_budget_cap": 6500,
                    "recent_turns_floor": 2,
                    "compaction_trigger_context_ratio": 0.60,
                },
            }
        },
    )

    resolved = merge_execution_options(
        base_options=base_options,
        workspace_dir=tmp_path,
        execution_options=ExecutionOptions(model_name="deepseek-v4-flash-thinking"),
    )

    assert resolved.model_name == "deepseek-v4-flash-thinking"
    assert resolved.conversation_memory_settings.memory_token_budget_cap == 6500
    assert resolved.conversation_memory_settings.memory_token_budget_floor == 4000


@pytest.mark.unit
def test_build_base_execution_options_uses_default_when_model_has_no_runtime_hints(tmp_path: Path) -> None:
    """验证默认 conversation memory 配置会保留到合并结果。"""

    base_options = build_base_execution_options(
        workspace_dir=tmp_path,
        run_config={
            "conversation_memory": {
                "default": {
                    "memory_token_budget_ratio": 0.10,
                    "memory_token_budget_floor": 4000,
                    "memory_token_budget_cap": 9000,
                    "recent_turns_floor": 2,
                    "compaction_trigger_context_ratio": 0.60,
                },
            }
        },
    )

    resolved = merge_execution_options(
        base_options=base_options,
        workspace_dir=tmp_path,
        execution_options=ExecutionOptions(model_name="qwen-plus-thinking"),
    )

    assert resolved.model_name == "qwen-plus-thinking"
    assert resolved.conversation_memory_settings.memory_token_budget_cap == 9000


@pytest.mark.unit
def test_resolve_conversation_memory_settings_merges_model_runtime_hints(tmp_path: Path) -> None:
    """验证模型 runtime hints 可覆盖 conversation memory 默认公式。"""

    base_options = build_base_execution_options(
        workspace_dir=tmp_path,
        run_config={
            "conversation_memory": {
                "default": {
                    "memory_token_budget_ratio": 0.10,
                    "memory_token_budget_floor": 4000,
                    "memory_token_budget_cap": 9000,
                    "recent_turns_floor": 2,
                    "compaction_trigger_context_ratio": 0.60,
                }
            }
        },
    )

    settings = resolve_conversation_memory_settings(
        conversation_memory_config=base_options.conversation_memory_config,
        model_config={
            "runtime_hints": {
                "conversation_memory": {
                    "memory_token_budget_cap": 20000,
                    "recent_turns_floor": 3,
                }
            }
        },
    )

    assert settings.memory_token_budget_cap == 20000
    assert settings.recent_turns_floor == 3


@pytest.mark.unit
def test_working_memory_policy_uses_uncompacted_tail_and_budget() -> None:
    """验证 working memory 强制保留最近 N 轮，并按预算回放更老 raw turn。"""

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
        memory_token_budget_ratio=0.01,
        memory_token_budget_floor=120,
        memory_token_budget_cap=120,
        recent_turns_floor=2,
    )

    # 总预算 120 tokens 仅够最近 2 轮 forced 保留，更老 raw 不进。
    selected = policy.select_turns(transcript, settings=settings, available_token_budget=0, max_context_tokens=200_000)

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

    selected = policy.select_turns(
        transcript,
        settings=ConversationMemorySettings(),
        available_token_budget=1024,
        max_context_tokens=200_000,
    )

    assert len(selected) == 1
    assert selected[0].turn_id == "turn_1"
    assert selected[0].user_text == "x" * 4000
    assert selected[0].assistant_text == ""


@pytest.mark.unit
def test_working_memory_policy_keeps_latest_turn_as_truncated_view_when_single_turn_exceeds_budget() -> None:
    """验证最新 turn 即使超 cap 也会以裁剪视图保留，而不是整轮消失。"""

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
        memory_token_budget_ratio=0.01,
        memory_token_budget_floor=120,
        memory_token_budget_cap=120,
        recent_turns_floor=1,
    )

    selected = policy.select_turns(transcript, settings=settings, available_token_budget=0, max_context_tokens=240)

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
        memory_token_budget_ratio=0.0,
        memory_token_budget_floor=400,
        memory_token_budget_cap=400,
        recent_turns_floor=2,
    )
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=FileConversationSessionArchiveStore(tmp_path / "conversations"),
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
        archive_store=FileConversationSessionArchiveStore(tmp_path / "conversations"),
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

    settings = ConversationMemorySettings(
        compaction_trigger_context_ratio=0.001,
        compaction_tail_preserve_turns=1,
    )
    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    compressor = _FakeCompressor(delay=0.02)
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=store,
        episodic_memory_compressor=compressor,
    )
    transcript = ConversationTranscript.create_empty("sess_1")
    for index in range(1, 5):
        transcript = transcript.append_turn(_build_turn(index))
    store.save(_archive_from_transcript(transcript), expected_revision=None)

    async def _run() -> None:
        manager.schedule_compaction(
            session_id="sess_1",
            prepared_scene=_build_prepared_scene(settings=settings),
            transcript=transcript,
            system_prompt="sys",
        )
        await asyncio.sleep(0)
        current_archive = store.load("sess_1")
        assert current_archive is not None
        current_runtime = current_archive.runtime_transcript
        from dataclasses import replace as _dc_replace
        updated_runtime = current_runtime.append_turn(_build_turn(5))
        store.save(
            _dc_replace(current_archive, runtime_transcript=updated_runtime),
            expected_revision=current_archive.revision,
        )
        await manager.cancel_pending_compaction("sess_1")
        manager.schedule_compaction(
            session_id="sess_1",
            prepared_scene=_build_prepared_scene(settings=settings),
            transcript=updated_runtime,
            system_prompt="sys",
        )
        await manager.wait_for_session("sess_1")

    asyncio.run(_run())

    loaded_archive = store.load("sess_1")

    assert loaded_archive is not None
    loaded = loaded_archive.runtime_transcript
    assert loaded.compacted_turn_count >= 4
    assert loaded.episodes
    assert loaded.pinned_state.current_goal == "跟踪公司最新变化"
    assert compressor.calls[-1][1][-1] == "turn_4"


@pytest.mark.unit
def test_prepare_transcript_compacts_persisted_tail_before_resumed_turn(tmp_path: Path) -> None:
    """验证进程重启后恢复同 session 的首轮会先同步压缩已落盘 transcript。"""

    settings = ConversationMemorySettings()
    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    compressor = _FakeCompressor()
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=store,
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
    store.save(_archive_from_transcript(transcript), expected_revision=None)
    seeded_archive = store.load("sess_1")
    assert seeded_archive is not None
    seeded_transcript = seeded_archive.runtime_transcript

    prepared = asyncio.run(
        manager.prepare_transcript(
            session_id="sess_1",
            prepared_scene=_build_prepared_scene(settings=settings, max_context_tokens=2048),
            transcript=seeded_transcript,
            system_prompt="sys",
            user_text="hi",
        )
    )

    persisted_archive = store.load("sess_1")

    assert prepared.compacted_turn_count == 2
    assert prepared.episodes
    assert compressor.calls == [("sess_1", ("turn_1", "turn_2"))]
    assert persisted_archive is not None
    assert persisted_archive.runtime_transcript.compacted_turn_count == 2


@pytest.mark.unit
def test_conversation_memory_emits_compaction_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ConversationMemory 关键 compaction 动作应输出 verbose/debug 日志。"""

    settings = ConversationMemorySettings()
    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    compressor = _FakeCompressor()
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=store,
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
    store.save(_archive_from_transcript(transcript), expected_revision=None)
    seeded_archive = store.load("sess_log")
    assert seeded_archive is not None
    transcript = seeded_archive.runtime_transcript

    verbose_mock = Mock()
    debug_mock = Mock()
    monkeypatch.setattr(Log, "verbose", verbose_mock)
    monkeypatch.setattr(Log, "debug", debug_mock)

    prepared = asyncio.run(
        manager.prepare_transcript(
            session_id="sess_log",
            prepared_scene=_build_prepared_scene(settings=settings, max_context_tokens=2048),
            transcript=transcript,
            system_prompt="sys",
            user_text="hi",
        )
    )
    manager.schedule_compaction(
        session_id="sess_log",
        prepared_scene=_build_prepared_scene(settings=settings, max_context_tokens=100000),
        transcript=prepared,
        system_prompt="sys",
    )

    verbose_messages = [call.args[0] for call in verbose_mock.call_args_list]
    debug_messages = [call.args[0] for call in debug_mock.call_args_list]

    assert any("准备同步压缩 transcript" in message for message in verbose_messages)
    assert any("同步压缩写回 transcript" in message for message in verbose_messages)
    assert any("无需调度 compaction" in message for message in debug_messages)


@pytest.mark.unit
def test_resolve_memory_total_budget_clamped_by_floor_and_cap() -> None:
    """验证总池预算公式 ``clamp(window * ratio, floor, cap)`` 在跨档位都生效。"""

    from dayu.host.conversation_memory import _resolve_memory_total_budget

    settings = ConversationMemorySettings(
        memory_token_budget_ratio=0.10,
        memory_token_budget_floor=4000,
        memory_token_budget_cap=32000,
    )
    # 1M 档：100K 算出被 cap 截到 32K
    assert _resolve_memory_total_budget(settings=settings, max_context_tokens=1_000_000) == 32000
    # 256K 档：自然落到 25.6K
    assert _resolve_memory_total_budget(settings=settings, max_context_tokens=256_000) == 25_600
    # 8K 档：800 太小被 floor 兜底
    assert _resolve_memory_total_budget(settings=settings, max_context_tokens=8_000) == 4000


@pytest.mark.unit
def test_recent_turns_floor_forced_preserved_when_budget_zero() -> None:
    """``recent_turns_floor`` 在 budget=0 时仍强制保留最近 N 轮。"""

    policy = DefaultWorkingMemoryPolicy()
    transcript = ConversationTranscript.create_empty("sess_floor")
    for index in range(1, 6):
        transcript = transcript.append_turn(_build_turn(index))
    settings = ConversationMemorySettings(
        memory_token_budget_ratio=0.10,
        memory_token_budget_floor=4000,
        memory_token_budget_cap=32000,
        recent_turns_floor=2,
    )

    selected = policy.select_turns(transcript, settings=settings, available_token_budget=0, max_context_tokens=200_000)

    assert tuple(t.turn_id for t in selected) == ("turn_4", "turn_5")


@pytest.mark.unit
def test_compaction_triggered_by_window_ratio(tmp_path: Path) -> None:
    """当 window_used 超过 ``max_context_tokens * trigger_context_ratio`` 时触发 compaction。"""

    settings = ConversationMemorySettings(
        compaction_trigger_context_ratio=0.50,
        compaction_tail_preserve_turns=1,
    )
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=FileConversationSessionArchiveStore(tmp_path / "conversations"),
    )
    transcript = ConversationTranscript.create_empty("sess_trig")
    for index in range(1, 5):
        transcript = transcript.append_turn(
            ConversationTurnRecord(
                turn_id=f"turn_{index}",
                scene_name="interactive",
                user_text="x" * 800,
                assistant_final="",
            )
        )

    candidates = manager._select_compaction_candidate(
        transcript=transcript,
        settings=settings,
        max_context_tokens=2048,
    )
    assert len(candidates) > 0


@pytest.mark.unit
def test_compaction_not_triggered_below_ratio(tmp_path: Path) -> None:
    """当 window_used 远小于阈值时不触发 compaction。"""

    settings = ConversationMemorySettings(
        compaction_trigger_context_ratio=0.99,
        compaction_tail_preserve_turns=1,
    )
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=FileConversationSessionArchiveStore(tmp_path / "conversations"),
    )
    transcript = ConversationTranscript.create_empty("sess_low")
    for index in range(1, 4):
        transcript = transcript.append_turn(_build_turn(index))

    candidates = manager._select_compaction_candidate(
        transcript=transcript,
        settings=settings,
        max_context_tokens=1_000_000,
    )
    assert candidates == ()


@pytest.mark.unit
def test_forced_turn_overflow_threshold_decoupled_from_cap() -> None:
    """Fix #1：单轮兜底阈值由 ``max_context_tokens`` 派生，与 ``memory_token_budget_cap`` 解耦。

    构造场景：单轮 token 数 < cap 但 > 当前模型窗口。旧逻辑会按 cap 兜底，
    把整轮原样回放从而撑爆窗口；新逻辑必须走 ``_build_minimum_preserved_turn_view``。
    """

    policy = DefaultWorkingMemoryPolicy()
    transcript = ConversationTranscript.create_empty("sess_decouple").append_turn(
        ConversationTurnRecord(
            turn_id="turn_1",
            scene_name="interactive",
            user_text="问题",
            assistant_final="x" * 4000,  # ~1000 tokens 远小于 cap=60000
        )
    )
    settings = ConversationMemorySettings(
        memory_token_budget_cap=60000,  # 故意调高 cap
        recent_turns_floor=1,
    )
    # 模型窗口仅 200，单轮就超出窗口；阈值应由 max_context_tokens 派生，触发 minimum_preserve。
    selected = policy.select_turns(
        transcript,
        settings=settings,
        available_token_budget=0,
        max_context_tokens=200,
    )
    assert len(selected) == 1
    assert selected[0].assistant_text.endswith("...<truncated>")


@pytest.mark.unit
def test_forced_turn_overflow_threshold_uses_actual_forced_count() -> None:
    """Round3 #1：单轮兜底阈值除数取**当前实际 forced 轮数**而非 ``settings.recent_turns_floor``。

    场景：``recent_turns_floor=5`` 但当前 transcript 仅 1 轮 raw turn，``max_context_tokens=300``。
    旧实现按 ``floor + 1 = 6`` 派生阈值 ``300/6 = 50``，单轮（约 75 tokens）会被错误判定溢出
    走 minimum_preserve；新实现按 ``actual_forced_count + 1 = 2`` 派生阈值 ``300/2 = 150``，
    单轮可完整保留。
    """

    policy = DefaultWorkingMemoryPolicy()
    transcript = ConversationTranscript.create_empty("sess_actual_floor").append_turn(
        ConversationTurnRecord(
            turn_id="turn_only",
            scene_name="interactive",
            user_text="新对话第一轮",
            assistant_final="x" * 200,  # ~50 tokens
        )
    )
    settings = ConversationMemorySettings(
        memory_token_budget_cap=60000,
        recent_turns_floor=5,  # 配置高 floor，但实际只有 1 轮
    )
    selected = policy.select_turns(
        transcript,
        settings=settings,
        available_token_budget=0,
        max_context_tokens=300,
    )
    assert len(selected) == 1
    # 实际 forced=1，阈值=300/2=150，单轮 ~50 tokens 完整保留，不应被截断。
    assert not selected[0].assistant_text.endswith("...<truncated>")


@pytest.mark.unit
def test_compaction_trigger_does_not_count_episodes(tmp_path: Path) -> None:
    """Round 1 Fix #2 + Round 2 #1：``transcript.episodes`` 仅按 ``_build_memory_block``
    裁切后真实进入 prompt 的部分计入 ``window_used``；超大 episode 在剩余预算放不下时
    应被丢弃，不计入触发判定，避免压缩抖动。"""

    settings = ConversationMemorySettings(
        compaction_trigger_context_ratio=0.50,
        compaction_tail_preserve_turns=1,
    )
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=FileConversationSessionArchiveStore(tmp_path / "conversations"),
    )
    transcript = ConversationTranscript.create_empty("sess_no_eps")
    transcript = transcript.append_turn(_build_turn(1))
    transcript = transcript.append_turn(_build_turn(2))
    # 注入一个超大 episode；若错误计入 window_used，会无谓触发压缩。
    huge_episode = ConversationEpisodeSummary(
        episode_id="ep_huge",
        start_turn_id="turn_1",
        end_turn_id="turn_1",
        title="历史摘要",
        goal="x" * 50_000,
        confirmed_facts=(),
    )
    transcript = transcript.replace_memory(
        pinned_state=transcript.pinned_state,
        episodes=(huge_episode,),
        compacted_turn_count=transcript.compacted_turn_count,
    )

    # raw turn 体量很小，加 system/user 远不到 1M*0.5=500K 阈值；episode 不应让公式越线。
    candidates = manager._select_compaction_candidate(
        transcript=transcript,
        settings=settings,
        max_context_tokens=1_000_000,
        system_prompt_tokens=200,
        user_text_tokens=50,
    )
    assert candidates == ()


@pytest.mark.unit
def test_schedule_compaction_passes_system_prompt_token_estimate(tmp_path: Path) -> None:
    """``schedule_compaction`` 必须把 system_prompt 估算 token 接到触发公式。

    Round2 #2：``schedule_compaction`` 在 ``persist_turn`` 之后调用，当前轮 user_text
    已落入 ``transcript.turns`` 最末位，自然出现在 ``uncompressed_tokens`` 中，
    本入口不再接收 ``user_text`` 避免重复计数。``prepare_transcript`` 仍显式传
    ``user_text``（开局尚未 persist）。
    """

    settings = ConversationMemorySettings(
        compaction_trigger_context_ratio=0.50,
        compaction_tail_preserve_turns=1,
    )
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=FileConversationSessionArchiveStore(tmp_path / "conversations"),
    )
    transcript = ConversationTranscript.create_empty("sess_wired")
    for index in range(1, 4):
        transcript = transcript.append_turn(
            ConversationTurnRecord(
                turn_id=f"turn_{index}",
                scene_name="interactive",
                user_text="x" * 100,
                assistant_final="",
            )
        )

    captured: dict[str, int] = {}
    original = manager._select_compaction_candidate

    def _spy(
        *,
        transcript: ConversationTranscript,
        settings: ConversationMemorySettings,
        max_context_tokens: int,
        system_prompt_tokens: int = 0,
        user_text_tokens: int = 0,
    ) -> tuple[ConversationTurnRecord, ...]:
        captured["system_prompt_tokens"] = system_prompt_tokens
        captured["user_text_tokens"] = user_text_tokens
        return original(
            transcript=transcript,
            settings=settings,
            max_context_tokens=max_context_tokens,
            system_prompt_tokens=system_prompt_tokens,
            user_text_tokens=user_text_tokens,
        )

    manager._select_compaction_candidate = _spy  # type: ignore[method-assign]

    manager.schedule_compaction(
        session_id="sess_wired",
        prepared_scene=_build_prepared_scene(settings=settings),
        transcript=transcript,
        system_prompt="SYS:" + "y" * 1000,
    )

    assert captured["system_prompt_tokens"] > 100
    # Round2 #2 回归：schedule_compaction 不再把 user_text 重复计入触发公式。
    assert captured["user_text_tokens"] == 0


@pytest.mark.unit
def test_compaction_trigger_counts_actual_episode_in_prompt(tmp_path: Path) -> None:
    """Round2 #1：触发公式必须计入 ``_build_memory_block`` 的兜底分支保留的 episode。

    场景：单条 episode 远大于 ``total_budget``。``_build_memory_block`` 仍会保留这条
    最新 episode（避免 episodic 层完全失效）。如果 ``_select_compaction_candidate``
    把 ``transcript.episodes`` 完全踢出 ``window_used``，会低估真实 prompt 体积、
    在该压缩时不压缩。
    """

    settings = ConversationMemorySettings(
        memory_token_budget_ratio=0.10,
        memory_token_budget_floor=2000,
        memory_token_budget_cap=2000,  # 把 budget 锁到 2000，让大 episode 必然超 budget
        compaction_trigger_context_ratio=0.50,
        compaction_tail_preserve_turns=1,
    )
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=FileConversationSessionArchiveStore(tmp_path / "conversations"),
    )
    transcript = ConversationTranscript.create_empty("sess_round2_1")
    # raw turns 体量很小，单独不会触发。
    transcript = transcript.append_turn(_build_turn(1))
    transcript = transcript.append_turn(_build_turn(2))
    huge_summary = "x" * 100_000  # ~25K tokens，远超 budget 2000
    transcript = transcript.replace_memory(
        pinned_state=transcript.pinned_state,
        episodes=(
            ConversationEpisodeSummary(
                episode_id="ep_huge",
                start_turn_id="turn_0",
                end_turn_id="turn_0",
                title="历史摘要",
                goal=huge_summary,
            ),
        ),
        compacted_turn_count=transcript.compacted_turn_count,
    )

    # 模型窗口 30K，阈值 = 30K * 0.5 = 15K。raw + system + user 单独 < 15K，
    # 但 _build_memory_block 兜底保留的大 episode 会让 actual_episodic > 15K，触发公式必须包含。
    candidates = manager._select_compaction_candidate(
        transcript=transcript,
        settings=settings,
        max_context_tokens=30_000,
        system_prompt_tokens=200,
        user_text_tokens=50,
    )
    assert len(candidates) > 0


@pytest.mark.unit
def test_default_conversation_memory_settings_match_runtime_default() -> None:
    """Round2 #3：``ConversationMemorySettings()`` dataclass 默认值必须与 run.json
    / ``options.py`` 运行时默认一致，避免快照恢复 / dataclass 直构 / run.json
    三条路径得到不同 memory policy。"""

    from dayu.execution.options import _DEFAULT_RUN_CONFIG  # type: ignore[attr-defined]

    runtime_default = _DEFAULT_RUN_CONFIG.conversation_memory.default
    dataclass_default = ConversationMemorySettings()
    assert dataclass_default.memory_token_budget_ratio == runtime_default.memory_token_budget_ratio
    assert dataclass_default.memory_token_budget_floor == runtime_default.memory_token_budget_floor
    assert dataclass_default.memory_token_budget_cap == runtime_default.memory_token_budget_cap
    assert dataclass_default.recent_turns_floor == runtime_default.recent_turns_floor
    assert (
        dataclass_default.compaction_trigger_context_ratio
        == runtime_default.compaction_trigger_context_ratio
    )
    assert (
        dataclass_default.compaction_tail_preserve_turns
        == runtime_default.compaction_tail_preserve_turns
    )
    assert (
        dataclass_default.compaction_context_episode_window
        == runtime_default.compaction_context_episode_window
    )
    assert dataclass_default.compaction_scene_name == runtime_default.compaction_scene_name


@pytest.mark.unit
def test_prepare_transcript_skips_compaction_write_when_live_runtime_revision_diverged(
    tmp_path: Path,
) -> None:
    """业务层 stale-check：``live.runtime_transcript.revision`` 与
    ``compaction_input_transcript.revision`` 不一致时丢弃 compaction 结果，
    既不调用 ``archive_store.save``，也不触碰 ``history_archive``。
    """

    from unittest.mock import MagicMock

    settings = ConversationMemorySettings()
    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    compressor = _FakeCompressor()
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=store,
        episodic_memory_compressor=compressor,
    )
    transcript = ConversationTranscript.create_empty("sess_stale")
    for index in range(1, 7):
        transcript = transcript.append_turn(
            ConversationTurnRecord(
                turn_id=f"turn_{index}",
                scene_name="interactive",
                user_text=f"问题-{index}-" + ("x" * 794),
                assistant_final="",
            )
        )
    store.save(_archive_from_transcript(transcript), expected_revision=None)
    seeded_archive = store.load("sess_stale")
    assert seeded_archive is not None

    # 模拟 prepare_transcript 期间外部并发推进 runtime_transcript.revision：
    # 通过 monkeypatch store.load 让 manager 读到的 live archive 带新 revision
    # 同时保留原 history_archive，使写入断言能区分两者。
    from dataclasses import replace

    bumped_runtime = replace(
        seeded_archive.runtime_transcript, revision="bumped-runtime-rev"
    )
    bumped_archive = replace(
        seeded_archive, runtime_transcript=bumped_runtime, revision="bumped-archive-rev"
    )

    save_spy = MagicMock(side_effect=AssertionError("save 不应被调用"))
    store.save = save_spy  # type: ignore[method-assign]
    store.load = lambda session_id: (  # type: ignore[method-assign]
        bumped_archive if session_id == "sess_stale" else None
    )

    asyncio.run(
        manager.prepare_transcript(
            session_id="sess_stale",
            prepared_scene=_build_prepared_scene(
                settings=settings, max_context_tokens=2048
            ),
            transcript=seeded_archive.runtime_transcript,
            system_prompt="sys",
            user_text="hi",
        )
    )

    # 业务层 stale-check 应触发：save 一次也不应被调用
    save_spy.assert_not_called()
    # compaction 仍尝试压缩（_FakeCompressor 被调用），只是结果被丢弃
    assert compressor.calls, "compaction 应该执行了，但写回被 stale-check 丢弃"


@pytest.mark.unit
def test_compaction_only_replaces_runtime_transcript_subview(tmp_path: Path) -> None:
    """compaction 写回路径仅替换 runtime_transcript 子视图，``history_archive`` 原样保留。"""

    settings = ConversationMemorySettings()
    store = FileConversationSessionArchiveStore(tmp_path / "conversations")
    runtime = _FakeRuntime(resolved_options=_build_resolved_options(settings))
    compressor = _FakeCompressor()
    manager = DefaultConversationMemoryManager(
        runtime,
        archive_store=store,
        episodic_memory_compressor=compressor,
    )
    transcript = ConversationTranscript.create_empty("sess_keep_history")
    for index in range(1, 7):
        transcript = transcript.append_turn(
            ConversationTurnRecord(
                turn_id=f"turn_{index}",
                scene_name="interactive",
                user_text=f"问题-{index}-" + ("x" * 794),
                assistant_final="",
            )
        )
    store.save(_archive_from_transcript(transcript), expected_revision=None)
    seeded_archive = store.load("sess_keep_history")
    assert seeded_archive is not None

    # 在 archive 中预置一条 history（模拟"已有展示历史"）
    from dayu.host.conversation_session_archive import ConversationHistoryTurnRecord

    history_turn = ConversationHistoryTurnRecord(
        turn_id="hist_turn_1",
        scene_name="interactive",
        user_text="历史问题",
        assistant_text="历史回答",
        assistant_reasoning="历史思考片段",
        created_at=seeded_archive.runtime_transcript.created_at,
    )
    archive_with_history = _dc_replace_archive_history(
        seeded_archive, history_turns=(history_turn,)
    )
    store.save(archive_with_history, expected_revision=seeded_archive.revision)

    seeded_archive_with_history = store.load("sess_keep_history")
    assert seeded_archive_with_history is not None

    asyncio.run(
        manager.prepare_transcript(
            session_id="sess_keep_history",
            prepared_scene=_build_prepared_scene(
                settings=settings, max_context_tokens=2048
            ),
            transcript=seeded_archive_with_history.runtime_transcript,
            system_prompt="sys",
            user_text="hi",
        )
    )

    persisted = store.load("sess_keep_history")
    assert persisted is not None
    # runtime_transcript 已被压缩推进
    assert persisted.runtime_transcript.compacted_turn_count >= 1
    # history_archive 完整保留：内容、reasoning 都未被 compaction 触碰
    assert len(persisted.history_archive.turns) == 1
    assert persisted.history_archive.turns[0].assistant_reasoning == "历史思考片段"
    assert persisted.history_archive.turns[0].assistant_text == "历史回答"


def _dc_replace_archive_history(
    archive,
    *,
    history_turns,
):
    """构造仅替换 history_archive 的 archive 副本，并推进 archive.revision 与 runtime.revision 同步。"""

    from dataclasses import replace
    import uuid

    from dayu.host.conversation_session_archive import ConversationHistoryArchive

    new_history = ConversationHistoryArchive(
        session_id=archive.session_id, turns=tuple(history_turns)
    )
    new_revision = f"rev_{uuid.uuid4().hex[:8]}"
    runtime = replace(archive.runtime_transcript, revision=new_revision)
    return replace(
        archive,
        revision=new_revision,
        runtime_transcript=runtime,
        history_archive=new_history,
    )
