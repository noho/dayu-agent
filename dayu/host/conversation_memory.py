"""多轮会话分层记忆实现。"""

from __future__ import annotations

import asyncio
from functools import lru_cache
import json
import math
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Protocol

from dayu.contracts.agent_types import (
    AgentMessage,
    build_assistant_chat_message,
    build_system_chat_message,
    build_user_chat_message,
)
from dayu.engine.events import EventType
from dayu.log import Log
from dayu.text import strip_markdown_fence
from dayu.host._coercion import _coerce_string_tuple
from dayu.host.conversation_store import (
    ConversationEpisodeSummary,
    ConversationPinnedState,
    ConversationSessionArchiveStore,
    ConversationToolUseSummary,
    ConversationTranscript,
    ConversationTurnRecord,
)
from dayu.host.conversation_runtime import (
    ConversationCompactionRequest,
    ConversationPreparedSceneProtocol,
    ConversationRuntimeProtocol,
)
from dayu.execution.options import ConversationMemorySettings

MODULE = "RUNTIME.CONVERSATION_MEMORY"
_COMPACTION_USER_PROMPT_HEADER = "请基于以下会话片段生成结构化阶段摘要。仅输出严格 JSON。"
_WORKING_MEMORY_TRUNCATION_SUFFIX = "...<truncated>"
_HALF_WIDTH_TOKEN_UNITS = 1
_FULL_WIDTH_TOKEN_UNITS = 2
_TOKEN_UNITS_PER_ESTIMATED_TOKEN = 2


@lru_cache(maxsize=256)
def _token_units_for_char(char: str) -> int:
    """返回单个字符对应的 token 估算单位。

    Args:
        char: 单个字符。

    Returns:
        宽字符返回 ``2``，其余字符返回 ``1``。

    Raises:
        无。
    """

    if unicodedata.east_asian_width(char) in {"W", "F"}:
        return _FULL_WIDTH_TOKEN_UNITS
    return _HALF_WIDTH_TOKEN_UNITS


def _estimate_token_units(text: str) -> int:
    """估算文本对应的原始 token 单位数。

    Args:
        text: 原始文本。

    Returns:
        估算得到的 token 单位总数。

    Raises:
        无。
    """

    return sum(_token_units_for_char(char) for char in text)


def _token_units_to_estimated_tokens(token_units: int) -> int:
    """将 token 单位数转换为保守 token 估算值。

    Args:
        token_units: token 单位总数。

    Returns:
        保守 token 估算值。

    Raises:
        无。
    """

    if token_units <= 0:
        return 0
    return max(1, math.ceil(token_units / _TOKEN_UNITS_PER_ESTIMATED_TOKEN))


def _estimate_tokens(text: str) -> int:
    """用保守字符近似估算 token 数。

    Args:
        text: 原始文本。

    Returns:
        估算 token 数。

    Raises:
        无。
    """

    normalized = str(text or "")
    if not normalized:
        return 0
    return _token_units_to_estimated_tokens(_estimate_token_units(normalized))


def _estimate_turn_tokens(turn: ConversationTurnRecord) -> int:
    """估算单个 turn 的 token 占用。

    Args:
        turn: 单轮记录。

    Returns:
        token 估算值。

    Raises:
        无。
    """

    pieces = [turn.user_text, turn.assistant_final]
    for tool_use in turn.tool_uses:
        pieces.append(tool_use.name)
        pieces.append(json.dumps(tool_use.arguments, ensure_ascii=False, sort_keys=True))
        pieces.append(tool_use.result_summary)
    return sum(_estimate_tokens(piece) for piece in pieces)


def _resolve_memory_total_budget(
    *,
    settings: ConversationMemorySettings,
    max_context_tokens: int,
) -> int:
    """解析单总池的 token 预算。

    单总池供"更老 raw turn + episode summaries"共同消费；最近 N 轮 raw turn 由
    ``recent_turns_floor`` 强制保留，不计入此预算。

    Args:
        settings: 分层记忆配置。
        max_context_tokens: 当前模型最大上下文 token。

    Returns:
        单总池 token 预算。

    Raises:
        无。
    """

    if max_context_tokens <= 0:
        return settings.memory_token_budget_cap
    computed = int(max_context_tokens * settings.memory_token_budget_ratio)
    bounded = max(
        settings.memory_token_budget_floor,
        min(settings.memory_token_budget_cap, computed),
    )
    return max(1, min(max_context_tokens, bounded))


def _resolve_prepared_scene_max_context_tokens(
    prepared_scene: ConversationPreparedSceneProtocol,
) -> int:
    """解析当前 scene 的最大上下文 token。

    Args:
        prepared_scene: 当前轮静态 scene 计划。

    Returns:
        最大上下文 token；未知时返回 ``0``。

    Raises:
        无。
    """

    max_context_tokens = int(prepared_scene.agent_create_args.max_context_tokens or 0)
    if max_context_tokens > 0:
        return max_context_tokens
    return int(prepared_scene.model_config.get("max_context_tokens") or 0)


def _has_any_pinned_state(pinned_state: ConversationPinnedState) -> bool:
    """判断 pinned state 是否存在有效内容。

    Args:
        pinned_state: 当前 pinned state。

    Returns:
        若至少有一项非空则返回 ``True``。

    Raises:
        无。
    """

    return bool(
        pinned_state.current_goal
        or pinned_state.confirmed_subjects
        or pinned_state.user_constraints
        or pinned_state.open_questions
    )


def _render_pinned_state_block(pinned_state: ConversationPinnedState) -> str:
    """渲染 pinned state 文本块。

    Args:
        pinned_state: 当前 pinned state。

    Returns:
        文本块；无内容时返回空字符串。

    Raises:
        无。
    """

    if not _has_any_pinned_state(pinned_state):
        return ""
    lines = ["Pinned State:"]
    if pinned_state.current_goal:
        lines.append(f"- 当前主任务：{pinned_state.current_goal}")
    if pinned_state.confirmed_subjects:
        lines.append(f"- 已确认对象：{'；'.join(pinned_state.confirmed_subjects)}")
    if pinned_state.user_constraints:
        lines.append(f"- 用户约束：{'；'.join(pinned_state.user_constraints)}")
    if pinned_state.open_questions:
        lines.append(f"- 未决问题：{'；'.join(pinned_state.open_questions)}")
    return "\n".join(lines)


def _render_episode_summary(episode: ConversationEpisodeSummary) -> str:
    """渲染单个 episode 摘要块。

    Args:
        episode: 单个阶段摘要。

    Returns:
        文本块。

    Raises:
        无。
    """

    lines = [f"Episode {episode.episode_id}: {episode.title or '未命名阶段'}"]
    if episode.goal:
        lines.append(f"- 阶段目标：{episode.goal}")
    if episode.completed_actions:
        lines.append(f"- 已完成动作：{'；'.join(episode.completed_actions)}")
    if episode.confirmed_facts:
        lines.append(f"- 已确认事实：{'；'.join(episode.confirmed_facts)}")
    if episode.user_constraints:
        lines.append(f"- 用户约束：{'；'.join(episode.user_constraints)}")
    if episode.open_questions:
        lines.append(f"- 未决问题：{'；'.join(episode.open_questions)}")
    if episode.next_step:
        lines.append(f"- 建议下一步：{episode.next_step}")
    if episode.tool_findings:
        lines.append(f"- 工具发现：{'；'.join(episode.tool_findings)}")
    return "\n".join(lines)


def _estimate_working_turn_view_tokens(turn_view: WorkingMemoryTurnView) -> int:
    """估算 working memory turn view 的 token 占用。

    Args:
        turn_view: 待估算的 turn 视图。

    Returns:
        token 估算值。

    Raises:
        无。
    """

    return _estimate_tokens(turn_view.user_text) + _estimate_tokens(turn_view.assistant_text)


def _truncate_text_to_token_budget(text: str, token_budget: int) -> str:
    """按 token 预算截断文本并附加显式截断标记。

    Args:
        text: 原始文本。
        token_budget: 可用 token 预算。

    Returns:
        截断后的文本。若原文已在预算内，则原样返回。

    Raises:
        无。
    """

    normalized = str(text or "").strip()
    if not normalized:
        return ""
    if token_budget <= 0:
        return _WORKING_MEMORY_TRUNCATION_SUFFIX
    if _estimate_tokens(normalized) <= token_budget:
        return normalized

    suffix = _WORKING_MEMORY_TRUNCATION_SUFFIX
    suffix_tokens = _estimate_tokens(suffix)
    if token_budget <= suffix_tokens:
        return suffix

    kept_chars: list[str] = []
    used_token_units = 0
    max_content_token_units = (
        token_budget * _TOKEN_UNITS_PER_ESTIMATED_TOKEN
    ) - _estimate_token_units(suffix)
    for char in normalized:
        char_token_units = _token_units_for_char(char)
        if used_token_units + char_token_units > max_content_token_units:
            break
        kept_chars.append(char)
        used_token_units += char_token_units
    kept_text = "".join(kept_chars).rstrip()
    if not kept_text:
        return suffix
    return kept_text + suffix


def _render_tool_summary_block(tool_uses: tuple[ConversationToolUseSummary, ...]) -> str:
    """渲染工具摘要文本块。

    Args:
        tool_uses: 当前 turn 内的工具调用摘要。

    Returns:
        拼接后的工具摘要文本；无工具时返回空字符串。

    Raises:
        无。
    """

    if not tool_uses:
        return ""
    lines = ["历史工具摘要："]
    for tool_use in tool_uses:
        args_text = json.dumps(tool_use.arguments, ensure_ascii=False, sort_keys=True)
        lines.append(f"- {tool_use.name} args={args_text} result={tool_use.result_summary}")
    return "\n".join(lines)


def _build_full_working_turn_view(turn: ConversationTurnRecord) -> WorkingMemoryTurnView:
    """基于原始 transcript turn 构建完整的 working memory 视图。

    Args:
        turn: 原始 turn。

    Returns:
        完整 turn 视图。

    Raises:
        无。
    """

    assistant_text = str(turn.assistant_final or "").strip()
    tool_block = _render_tool_summary_block(turn.tool_uses)
    if assistant_text and tool_block:
        assistant_text = f"{assistant_text}\n\n{tool_block}"
    elif tool_block:
        assistant_text = tool_block
    return WorkingMemoryTurnView(
        turn_id=turn.turn_id,
        user_text=turn.user_text,
        assistant_text=assistant_text,
    )


def _build_minimum_preserved_turn_view(
    turn: ConversationTurnRecord,
    *,
    token_budget: int,
) -> WorkingMemoryTurnView:
    """为最新 turn 构建不会整轮消失的最小保真视图。

    裁剪顺序固定为：
    1. 保留完整 `user_text`
    2. 若存在 `assistant_final`：预算足够则保留完整正文，否则按预算截断正文
       （工具摘要在该分支下不会进入视图，属于隐式丢弃）
    3. 若没有 `assistant_final` 但存在工具摘要，则对工具摘要做截断降级

    Args:
        turn: 原始 turn。
        token_budget: 当前 working memory 可用预算。

    Returns:
        最小保真视图。

    Raises:
        无。
    """

    user_text = turn.user_text
    user_tokens = _estimate_tokens(user_text)
    remaining_budget = max(0, token_budget - user_tokens)
    assistant_text = str(turn.assistant_final or "").strip()
    tool_block = _render_tool_summary_block(turn.tool_uses)

    if assistant_text:
        assistant_tokens = _estimate_tokens(assistant_text)
        if remaining_budget >= assistant_tokens:
            return WorkingMemoryTurnView(
                turn_id=turn.turn_id,
                user_text=user_text,
                assistant_text=assistant_text,
            )
        return WorkingMemoryTurnView(
            turn_id=turn.turn_id,
            user_text=user_text,
            assistant_text=_truncate_text_to_token_budget(assistant_text, remaining_budget),
        )

    if tool_block:
        return WorkingMemoryTurnView(
            turn_id=turn.turn_id,
            user_text=user_text,
            assistant_text=_truncate_text_to_token_budget(tool_block, remaining_budget),
        )

    return WorkingMemoryTurnView(
        turn_id=turn.turn_id,
        user_text=user_text,
        assistant_text="",
    )


@dataclass(frozen=True)
class ConversationPinnedStatePatch:
    """pinned state 的增量 patch。

    每个字段三态语义（稳定契约）：

    - ``None``：本次不修改该字段，保留 ``base`` 中的旧值。
    - 空值（``""`` 或 ``()``）：显式清空该字段。
    - 非空值：覆盖该字段为新值。

    `_parse_result` 通过 ``key in patch_dict`` 区分"LLM 未输出该字段"
    （映射为 ``None``）与"LLM 显式输出空值"（映射为 ``""`` / ``()``）。
    LLM 提示词需保证：仅在确实要清空时输出空值，否则省略键。
    """

    current_goal: str | None = None
    confirmed_subjects: tuple[str, ...] | None = None
    user_constraints: tuple[str, ...] | None = None
    open_questions: tuple[str, ...] | None = None

    def apply_to(self, base: ConversationPinnedState) -> ConversationPinnedState:
        """将 patch 应用到现有 pinned state。

        按字段三态语义（见类 docstring）逐项合并：``None`` 保留旧值，
        其它值（含空字符串、空 tuple）一律覆盖。

        Args:
            base: 旧状态。

        Returns:
            更新后的 pinned state。

        Raises:
            无。
        """

        return ConversationPinnedState(
            current_goal=base.current_goal if self.current_goal is None else self.current_goal,
            confirmed_subjects=(
                base.confirmed_subjects if self.confirmed_subjects is None else self.confirmed_subjects
            ),
            user_constraints=(
                base.user_constraints if self.user_constraints is None else self.user_constraints
            ),
            open_questions=base.open_questions if self.open_questions is None else self.open_questions,
        )


@dataclass(frozen=True)
class ConversationCompactionResult:
    """单次会话压缩的结构化结果。"""

    episode_summary: ConversationEpisodeSummary
    pinned_state_patch: ConversationPinnedStatePatch


@dataclass(frozen=True)
class WorkingMemoryTurnView:
    """Working memory 中供送模使用的 turn 视图。"""

    turn_id: str
    user_text: str
    assistant_text: str


class WorkingMemoryPolicyProtocol(Protocol):
    """Working memory 选择策略协议。"""

    def select_turns(
        self,
        transcript: ConversationTranscript,
        *,
        settings: ConversationMemorySettings,
        available_token_budget: int,
        max_context_tokens: int,
    ) -> tuple[WorkingMemoryTurnView, ...]:
        """选择需要作为 working memory 回放的 raw turns。"""
        ...


class EpisodicMemoryCompressorProtocol(Protocol):
    """阶段摘要压缩协议。"""

    async def compress(
        self,
        *,
        session_id: str,
        transcript: ConversationTranscript,
        turns: tuple[ConversationTurnRecord, ...],
        settings: ConversationMemorySettings,
    ) -> ConversationCompactionResult | None:
        """将一段 raw turns 压缩为结构化 episode summary。"""
        ...


class DurableMemoryStoreProtocol(Protocol):
    """Durable memory 存储协议。"""

    def read(self, session_id: str) -> dict[str, object]:
        """读取 durable memory。"""
        ...

    def write(self, session_id: str, payload: dict[str, object]) -> None:
        """写入 durable memory。"""
        ...


class ConversationRetrievalIndexProtocol(Protocol):
    """历史检索索引协议。"""

    def index_transcript(self, transcript: ConversationTranscript) -> None:
        """索引当前 transcript。"""
        ...

    def search(self, session_id: str, query: str, *, limit: int = 5) -> tuple[str, ...]:
        """检索历史片段。"""
        ...


class ConversationMemoryManagerProtocol(Protocol):
    """会话分层记忆统一协调协议。"""

    async def prepare_transcript(
        self,
        *,
        session_id: str,
        prepared_scene: ConversationPreparedSceneProtocol,
        transcript: ConversationTranscript,
        system_prompt: str,
        user_text: str,
    ) -> ConversationTranscript:
        """在当前轮开始前补齐必要的同步压缩。"""
        ...

    def build_messages(
        self,
        *,
        prepared_scene: ConversationPreparedSceneProtocol,
        transcript: ConversationTranscript,
        system_prompt: str,
        user_text: str,
    ) -> list[AgentMessage]:
        """构建当前轮送模消息。"""
        ...

    async def cancel_pending_compaction(self, session_id: str) -> None:
        """取消并等待指定 session 的压缩任务结束。"""
        ...

    def schedule_compaction(
        self,
        *,
        session_id: str,
        prepared_scene: ConversationPreparedSceneProtocol,
        transcript: ConversationTranscript,
        system_prompt: str,
    ) -> None:
        """基于最新 transcript 调度后台压缩。"""
        ...


class NullDurableMemoryStore:
    """Durable memory 的默认空实现。"""

    def read(self, session_id: str) -> dict[str, object]:
        """读取 durable memory。

        Args:
            session_id: 会话 ID。

        Returns:
            空字典。

        Raises:
            无。
        """

        del session_id
        return {}

    def write(self, session_id: str, payload: dict[str, object]) -> None:
        """写入 durable memory。

        Args:
            session_id: 会话 ID。
            payload: 待写入负载。

        Returns:
            无。

        Raises:
            无。
        """

        del session_id
        del payload


class NullConversationRetrievalIndex:
    """历史检索索引的默认空实现。"""

    def index_transcript(self, transcript: ConversationTranscript) -> None:
        """索引当前 transcript。

        Args:
            transcript: 当前 transcript。

        Returns:
            无。

        Raises:
            无。
        """

        del transcript

    def search(self, session_id: str, query: str, *, limit: int = 5) -> tuple[str, ...]:
        """检索历史片段。

        Args:
            session_id: 会话 ID。
            query: 查询语句。
            limit: 返回上限。

        Returns:
            空元组。

        Raises:
            无。
        """

        del session_id
        del query
        del limit
        return ()


class DefaultWorkingMemoryPolicy:
    """基于单总池预算的默认 working memory 策略。

    回放策略：
    - **强制保留** 最近 ``recent_turns_floor`` 轮 raw turn（不消耗 budget）；
      若最新一轮单轮即超出预算，走 ``_build_minimum_preserved_turn_view`` 兜底。
    - **按预算回放** 更老的 raw turn：从新到老消费 ``available_token_budget``，
      预算耗尽即停（更早历史已由 compaction 压缩为 episode summary）。
    """

    def select_turns(
        self,
        transcript: ConversationTranscript,
        *,
        settings: ConversationMemorySettings,
        available_token_budget: int,
        max_context_tokens: int,
    ) -> tuple[WorkingMemoryTurnView, ...]:
        """选择当前轮要回放的 raw turns。

        Args:
            transcript: 当前 transcript。
            settings: 分层记忆配置。
            available_token_budget: 单总池中分配给 raw turn 回放的剩余预算
                （pinned_state 与 recent_turns_floor 不计入此值）。
            max_context_tokens: 当前模型最大上下文 token；用于派生
                ``recent_turns_floor`` 的单轮溢出兜底阈值。

        Returns:
            选中的 working memory turn 视图，按时间从旧到新排列。

        Raises:
            无。
        """

        raw_tail = transcript.turns[transcript.compacted_turn_count :]
        if not raw_tail:
            return ()

        recent_floor = max(0, settings.recent_turns_floor)
        if recent_floor > 0:
            forced_turns = raw_tail[-recent_floor:]
            older_turns = raw_tail[:-recent_floor]
        else:
            forced_turns = ()
            older_turns = raw_tail

        forced_views = self._render_forced_turns(
            forced_turns,
            settings=settings,
            max_context_tokens=max_context_tokens,
            actual_forced_count=len(forced_turns),
        )

        extra_pool = max(0, available_token_budget)
        extra_views: list[WorkingMemoryTurnView] = []
        for turn in reversed(older_turns):
            full_view = _build_full_working_turn_view(turn)
            cost = _estimate_working_turn_view_tokens(full_view)
            if cost > extra_pool:
                break
            extra_views.append(full_view)
            extra_pool -= cost
        extra_views.reverse()
        return tuple([*extra_views, *forced_views])

    def _render_forced_turns(
        self,
        forced_turns: tuple[ConversationTurnRecord, ...],
        *,
        settings: ConversationMemorySettings,
        max_context_tokens: int,
        actual_forced_count: int,
    ) -> tuple[WorkingMemoryTurnView, ...]:
        """渲染强制保留的最近 N 轮 raw turn。

        默认走完整视图（forced floor 不计入 budget）；只有当**单轮 token 量超过
        模型当前窗口派生的兜底阈值** 这种极端情况才退到
        ``_build_minimum_preserved_turn_view`` 兜底，避免上下文窗口被一轮极长输入
        撑爆。阈值与 ``memory_token_budget_cap`` **解耦**：必须基于当前模型
        ``max_context_tokens`` 推导，否则在小窗口模型上可能出现"单轮 < cap 但
        > 当前窗口"被原样回放、撑爆 prompt 的失败模式。

        阈值公式：``max_context_tokens / max(2, actual_forced_count + 1)``。
        除数取**当前实际 forced 轮数**而不是 ``settings.recent_turns_floor``：新会话或
        刚 compaction 后只剩 1-2 轮 raw turn 时，若仍按配置 floor=4/5 计算会把单轮上限
        压得过低，触发不必要的 minimum_preserve 裁剪。语义上为"模型窗口给单轮
        forced turn 的上限"，按实际轮数+1 平摊，使 forced 队列加 system/pinned/当前
        user 共用窗口时单轮不会独吞。

        Args:
            forced_turns: 最近 N 轮强制保留的原始 turn。
            settings: 分层记忆配置；提供 ``memory_token_budget_cap`` 作为窗口未知
                时的兜底阈值。
            max_context_tokens: 当前模型最大上下文 token。``<= 0`` 时回落
                到 ``settings.memory_token_budget_cap`` 作为最后兜底。
            actual_forced_count: 当前实际 forced turn 数量，用于派生溢出阈值。

        Returns:
            渲染后的视图，按时间从旧到新。

        Raises:
            无。
        """

        if max_context_tokens > 0:
            divisor = max(2, actual_forced_count + 1)
            overflow_threshold = max(1, max_context_tokens // divisor)
        else:
            # 防御性兜底：模型窗口未知时回到 cap，保持旧行为不破坏。
            overflow_threshold = max(1, settings.memory_token_budget_cap)
        rendered: list[WorkingMemoryTurnView] = []
        for turn in forced_turns:
            full_view = _build_full_working_turn_view(turn)
            if _estimate_working_turn_view_tokens(full_view) <= overflow_threshold:
                rendered.append(full_view)
            else:
                rendered.append(
                    _build_minimum_preserved_turn_view(turn, token_budget=overflow_threshold)
                )
        return tuple(rendered)


class DefaultEpisodicMemoryCompressor:
    """基于 LLM compaction scene 的默认阶段摘要压缩器。"""

    def __init__(self, runtime: ConversationRuntimeProtocol) -> None:
        """初始化压缩器。

        Args:
            runtime: 默认 Runtime 实现。

        Returns:
            无。

        Raises:
            无。
        """

        self._runtime = runtime

    async def compress(
        self,
        *,
        session_id: str,
        transcript: ConversationTranscript,
        turns: tuple[ConversationTurnRecord, ...],
        settings: ConversationMemorySettings,
    ) -> ConversationCompactionResult | None:
        """将一段 raw turns 压缩为结构化阶段摘要。

        Args:
            session_id: 会话 ID。
            transcript: 当前 transcript。
            turns: 待压缩的 raw turn 片段。
            settings: 分层记忆配置。

        Returns:
            压缩结果；失败时返回 ``None``。

        Raises:
            无。解析失败等错误会被吞掉并记日志。
        """

        if not turns:
            return None
        prepared_scene = self._runtime.prepare_compaction_scene(settings.compaction_scene_name)
        request = ConversationCompactionRequest(session_id=f"{session_id}.compaction")
        prepared_agent = self._runtime.prepare_compaction_agent(prepared_scene, request)
        payload = self._build_user_payload(transcript=transcript, turns=turns, settings=settings)
        model_messages: list[AgentMessage] = [
            build_system_chat_message(prepared_agent.system_prompt),
            build_user_chat_message(payload),
        ]
        final_answer = ""
        async for event in prepared_agent.agent.run_messages(model_messages, session_id=f"{session_id}.compaction"):
            if event.type == EventType.FINAL_ANSWER and isinstance(event.data, dict):
                final_answer = str(event.data.get("content") or "")
        return self._parse_result(
            final_answer=final_answer,
            turns=turns,
        )

    def _build_user_payload(
        self,
        *,
        transcript: ConversationTranscript,
        turns: tuple[ConversationTurnRecord, ...],
        settings: ConversationMemorySettings,
    ) -> str:
        """构建压缩轮的用户输入 JSON。

        Args:
            transcript: 当前 transcript。
            turns: 待压缩 turn。
            settings: 分层记忆配置。

        Returns:
            送给压缩 scene 的用户输入文本。

        Raises:
            无。
        """

        payload = {
            "task": _COMPACTION_USER_PROMPT_HEADER,
            "pinned_state": {
                "current_goal": transcript.pinned_state.current_goal,
                "confirmed_subjects": list(transcript.pinned_state.confirmed_subjects),
                "user_constraints": list(transcript.pinned_state.user_constraints),
                "open_questions": list(transcript.pinned_state.open_questions),
            },
            "recent_episodes": [
                {
                    "episode_id": item.episode_id,
                    "title": item.title,
                    "goal": item.goal,
                    "confirmed_facts": list(item.confirmed_facts),
                    "user_constraints": list(item.user_constraints),
                    "open_questions": list(item.open_questions),
                    "next_step": item.next_step,
                }
                for item in transcript.episodes[-settings.compaction_context_episode_window :]
            ],
            "turns": [
                {
                    "turn_id": turn.turn_id,
                    "scene_name": turn.scene_name,
                    "user_text": turn.user_text,
                    "assistant_final": turn.assistant_final,
                    "tool_uses": [
                        {
                            "name": tool_use.name,
                            "arguments": tool_use.arguments,
                            "result_summary": tool_use.result_summary,
                        }
                        for tool_use in turn.tool_uses
                    ],
                    "warnings": list(turn.warnings),
                    "errors": list(turn.errors),
                }
                for turn in turns
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _parse_result(
        self,
        *,
        final_answer: str,
        turns: tuple[ConversationTurnRecord, ...],
    ) -> ConversationCompactionResult | None:
        """解析压缩 scene 返回的 JSON。

        ``pinned_state_patch`` 各字段按 ``ConversationPinnedStatePatch`` 三态契约解析：
        通过 ``"key" in patch_dict`` 区分"LLM 未输出该字段"（映射为 ``None``，
        语义=不动）与"LLM 显式输出空值"（映射为 ``""`` 或 ``()``，语义=清空）。
        非空值原样转换后覆盖。

        Args:
            final_answer: 模型最终回答文本。
            turns: 本次压缩对应的原始 turn 片段。

        Returns:
            压缩结果；非法时返回 ``None``。

        Raises:
            无。
        """

        normalized = strip_markdown_fence(final_answer)
        try:
            payload = json.loads(normalized)
        except json.JSONDecodeError:
            Log.warning("conversation compaction 返回非法 JSON，已跳过本次压缩", module=MODULE)
            return None
        if not isinstance(payload, dict):
            Log.warning("conversation compaction 返回非对象 JSON，已跳过本次压缩", module=MODULE)
            return None
        raw_summary = payload.get("episode_summary")
        if not isinstance(raw_summary, dict):
            Log.warning("conversation compaction 缺少 episode_summary，已跳过本次压缩", module=MODULE)
            return None
        raw_patch = payload.get("pinned_state_patch")
        summary_title = str(raw_summary.get("title") or "").strip()
        if not summary_title:
            Log.warning(
                "conversation compaction episode_summary.title 为空，已跳过本次压缩",
                module=MODULE,
            )
            return None
        summary = ConversationEpisodeSummary(
            episode_id=f"ep_{uuid.uuid4().hex[:8]}",
            start_turn_id=turns[0].turn_id,
            end_turn_id=turns[-1].turn_id,
            title=summary_title,
            goal=str(raw_summary.get("goal") or "").strip(),
            completed_actions=_coerce_string_tuple(raw_summary.get("completed_actions")),
            confirmed_facts=_coerce_string_tuple(raw_summary.get("confirmed_facts")),
            user_constraints=_coerce_string_tuple(raw_summary.get("user_constraints")),
            open_questions=_coerce_string_tuple(raw_summary.get("open_questions")),
            next_step=str(raw_summary.get("next_step") or "").strip(),
            tool_findings=_coerce_string_tuple(raw_summary.get("tool_findings")),
        )
        patch_dict = raw_patch if isinstance(raw_patch, dict) else {}
        return ConversationCompactionResult(
            episode_summary=summary,
            pinned_state_patch=ConversationPinnedStatePatch(
                current_goal=(
                    str(patch_dict.get("current_goal") or "").strip() if "current_goal" in patch_dict else None
                ),
                confirmed_subjects=(
                    _coerce_string_tuple(patch_dict.get("confirmed_subjects"))
                    if "confirmed_subjects" in patch_dict
                    else None
                ),
                user_constraints=(
                    _coerce_string_tuple(patch_dict.get("user_constraints"))
                    if "user_constraints" in patch_dict
                    else None
                ),
                open_questions=(
                    _coerce_string_tuple(patch_dict.get("open_questions"))
                    if "open_questions" in patch_dict
                    else None
                ),
            ),
        )


@dataclass
class _CompactionTaskState:
    """单个 session 的后台压缩任务状态。"""

    revision: str
    task: asyncio.Task[None]


class ConversationCompactionCoordinator:
    """后台压缩任务协调器。"""

    def __init__(self) -> None:
        """初始化协调器。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self._tasks: dict[str, _CompactionTaskState] = {}

    async def cancel(self, session_id: str) -> None:
        """取消并等待指定 session 的后台任务结束。

        Args:
            session_id: 会话 ID。

        Returns:
            无。

        Raises:
            无。
        """

        state = self._tasks.pop(session_id, None)
        if state is None:
            return
        Log.debug(f"取消后台 compaction: session_id={session_id}", module=MODULE)
        state.task.cancel()
        try:
            await state.task
        except asyncio.CancelledError:
            pass

    def schedule(self, session_id: str, *, revision: str, task_coro) -> None:
        """调度新的后台压缩任务。

        Args:
            session_id: 会话 ID。
            revision: 调度时对应的 transcript revision。
            task_coro: 协程对象。

        Returns:
            无。

        Raises:
            无。
        """

        existing = self._tasks.get(session_id)
        if existing is not None:
            existing.task.cancel()
            Log.verbose(
                f"重排后台 compaction 任务: session_id={session_id}, revision={revision}",
                module=MODULE,
            )
        loop = asyncio.get_running_loop()
        task = loop.create_task(task_coro, name=f"conversation_compaction:{session_id}")
        self._tasks[session_id] = _CompactionTaskState(revision=revision, task=task)
        Log.verbose(
            f"调度后台 compaction: session_id={session_id}, revision={revision}",
            module=MODULE,
        )
        task.add_done_callback(lambda _task, sid=session_id, rev=revision: self._cleanup(sid, rev))

    async def wait_for_session(self, session_id: str) -> None:
        """等待指定 session 当前后台任务完成。

        Args:
            session_id: 会话 ID。

        Returns:
            无。

        Raises:
            无。
        """

        state = self._tasks.get(session_id)
        if state is None:
            return
        Log.debug(f"等待后台 compaction 完成: session_id={session_id}", module=MODULE)
        try:
            await state.task
        except asyncio.CancelledError:
            return

    def _cleanup(self, session_id: str, revision: str) -> None:
        """清理已完成的后台任务句柄。

        Args:
            session_id: 会话 ID。
            revision: 调度时对应的 revision。

        Returns:
            无。

        Raises:
            无。
        """

        state = self._tasks.get(session_id)
        if state is None or state.revision != revision:
            return
        self._tasks.pop(session_id, None)
        Log.debug(f"清理后台 compaction 任务句柄: session_id={session_id}, revision={revision}", module=MODULE)


class DefaultConversationMemoryManager:
    """默认会话分层记忆协调器。"""

    def __init__(
        self,
        runtime: ConversationRuntimeProtocol,
        *,
        archive_store: ConversationSessionArchiveStore,
        working_memory_policy: WorkingMemoryPolicyProtocol | None = None,
        episodic_memory_compressor: EpisodicMemoryCompressorProtocol | None = None,
        durable_memory_store: DurableMemoryStoreProtocol | None = None,
        retrieval_index: ConversationRetrievalIndexProtocol | None = None,
        compaction_coordinator: ConversationCompactionCoordinator | None = None,
    ) -> None:
        """初始化会话分层记忆协调器。

        Args:
            runtime: 默认 Runtime 实现。
            archive_store: 会话 archive 存储。
            working_memory_policy: working memory 策略。
            episodic_memory_compressor: episode 摘要压缩器。
            durable_memory_store: durable memory 存储。
            retrieval_index: 历史检索索引。
            compaction_coordinator: 后台任务协调器。

        Returns:
            无。

        Raises:
            无。
        """

        self._runtime = runtime
        self._archive_store = archive_store
        self._working_memory_policy = working_memory_policy or DefaultWorkingMemoryPolicy()
        self._episodic_memory_compressor = episodic_memory_compressor or DefaultEpisodicMemoryCompressor(runtime)
        self._durable_memory_store = durable_memory_store or NullDurableMemoryStore()
        self._retrieval_index = retrieval_index or NullConversationRetrievalIndex()
        self._compaction_coordinator = compaction_coordinator or ConversationCompactionCoordinator()

    async def prepare_transcript(
        self,
        *,
        session_id: str,
        prepared_scene: ConversationPreparedSceneProtocol,
        transcript: ConversationTranscript,
        system_prompt: str,
        user_text: str,
    ) -> ConversationTranscript:
        """在当前轮开始前补齐必要的同步压缩。

        Args:
            session_id: 会话 ID。
            prepared_scene: 当前轮静态 scene 计划。
            transcript: 当前已落盘 transcript。
            system_prompt: 当前轮 system prompt，用于触发公式估算 token。
            user_text: 当前轮用户输入，用于触发公式估算 token。

        Returns:
            已补齐同步压缩后的 transcript。

        Raises:
            RuntimeError: 保存压缩结果时 revision 冲突会继续向上抛出。
        """

        self._retrieval_index.index_transcript(transcript)
        current = transcript
        settings = prepared_scene.conversation_memory_settings
        max_context_tokens = _resolve_prepared_scene_max_context_tokens(prepared_scene)
        system_prompt_tokens = _estimate_tokens(str(system_prompt or ""))
        user_text_tokens = _estimate_tokens(str(user_text or ""))
        while True:
            candidate_turns = self._select_compaction_candidate(
                transcript=current,
                settings=settings,
                max_context_tokens=max_context_tokens,
                system_prompt_tokens=system_prompt_tokens,
                user_text_tokens=user_text_tokens,
            )
            if not candidate_turns:
                return current
            Log.verbose(
                f"准备同步压缩 transcript: session_id={session_id}, revision={current.revision}, turns={len(candidate_turns)}",
                module=MODULE,
            )
            result = await self._episodic_memory_compressor.compress(
                session_id=session_id,
                transcript=current,
                turns=candidate_turns,
                settings=settings,
            )
            if result is None:
                Log.verbose(
                    f"同步压缩返回空结果，保持 transcript 不变: session_id={session_id}, revision={current.revision}",
                    module=MODULE,
                )
                return current
            next_transcript = current.replace_memory(
                pinned_state=result.pinned_state_patch.apply_to(current.pinned_state),
                episodes=(*current.episodes, result.episode_summary),
                compacted_turn_count=current.compacted_turn_count + len(candidate_turns),
            )
            # archive 形态下：只在聚合根上替换运行态子视图，``history_archive`` 原样保留。
            live_archive = self._archive_store.load(session_id)
            if live_archive is None or live_archive.runtime_transcript.revision != current.revision:
                # 同步 compaction 罕见冲突路径（外部并发推进 archive）：放弃本次写回，
                # 与既有 _run_compaction 自愈链路一致——下一轮 turn 会重新触发判定。
                Log.verbose(
                    f"同步 compaction 写回 archive 冲突，跳过本次：session_id={session_id}",
                    module=MODULE,
                )
                return current
            next_archive = live_archive.with_runtime_transcript(next_transcript)
            persisted_archive = self._archive_store.save(
                next_archive,
                expected_revision=live_archive.revision,
            )
            current = persisted_archive.runtime_transcript
            self._retrieval_index.index_transcript(current)
            Log.verbose(
                f"同步压缩写回 transcript: session_id={session_id}, revision={current.revision}, compacted_turn_count={current.compacted_turn_count}",
                module=MODULE,
            )

    def build_messages(
        self,
        *,
        prepared_scene: ConversationPreparedSceneProtocol,
        transcript: ConversationTranscript,
        system_prompt: str,
        user_text: str,
    ) -> list[AgentMessage]:
        """构建当前轮送模消息。

        Args:
            prepared_scene: 当前轮静态 scene 计划。
            transcript: 当前 transcript。
            system_prompt: 当前轮 system prompt。
            user_text: 当前用户输入。

        Returns:
            最终送模消息列表。

        Raises:
            无。
        """

        settings = prepared_scene.conversation_memory_settings
        max_context_tokens = _resolve_prepared_scene_max_context_tokens(prepared_scene)
        total_budget = _resolve_memory_total_budget(
            settings=settings,
            max_context_tokens=max_context_tokens,
        )
        # 单总池消费顺序：先扣 episode summaries，再把剩余预算交给"更老 raw turn"。
        # pinned_state / 最近 N 轮 raw turn 永远独立、不计入此预算。
        memory_block, episodic_used_tokens = self._build_memory_block(
            transcript=transcript,
            total_budget=total_budget,
        )
        remaining_for_working = max(0, total_budget - episodic_used_tokens)
        raw_turns = self._working_memory_policy.select_turns(
            transcript,
            settings=settings,
            available_token_budget=remaining_for_working,
            max_context_tokens=max_context_tokens,
        )
        messages: list[AgentMessage] = []
        normalized_system_prompt = str(system_prompt or "").strip()
        if normalized_system_prompt:
            messages.append(build_system_chat_message(normalized_system_prompt))
        if memory_block:
            messages.append(build_system_chat_message(memory_block))
        messages.extend(self._compile_raw_turns(raw_turns))
        messages.append(build_user_chat_message(user_text))
        return messages

    async def cancel_pending_compaction(self, session_id: str) -> None:
        """取消并等待指定 session 的后台压缩任务结束。

        Args:
            session_id: 会话 ID。

        Returns:
            无。

        Raises:
            无。
        """

        await self._compaction_coordinator.cancel(session_id)

    def schedule_compaction(
        self,
        *,
        session_id: str,
        prepared_scene: ConversationPreparedSceneProtocol,
        transcript: ConversationTranscript,
        system_prompt: str,
    ) -> None:
        """基于最新 transcript 调度后台压缩。

        本入口在 ``persist_turn`` 之后被调用，当前轮 ``user_text`` 已经写进
        ``transcript.turns`` 最末位、自然出现在 ``uncompressed_tokens`` 中，
        因此**不接收** ``user_text`` 参数避免重复计数。开局尚未 persist 的同步压缩
        判定走 ``prepare_transcript``，由它显式传 ``user_text``。

        Args:
            session_id: 会话 ID。
            prepared_scene: 当前轮静态 scene 计划。
            transcript: 最新 transcript（已包含本轮 user/assistant）。
            system_prompt: 当前轮 system prompt，用于触发公式估算 token。

        Returns:
            无。

        Raises:
            无。
        """

        settings = prepared_scene.conversation_memory_settings
        max_context_tokens = _resolve_prepared_scene_max_context_tokens(prepared_scene)
        candidate_turns = self._select_compaction_candidate(
            transcript=transcript,
            settings=settings,
            max_context_tokens=max_context_tokens,
            system_prompt_tokens=_estimate_tokens(str(system_prompt or "")),
            user_text_tokens=0,
        )
        self._retrieval_index.index_transcript(transcript)
        if not candidate_turns:
            Log.debug(f"无需调度 compaction: session_id={session_id}, revision={transcript.revision}", module=MODULE)
            return
        Log.verbose(
            f"检测到 compaction 候选: session_id={session_id}, revision={transcript.revision}, turns={len(candidate_turns)}",
            module=MODULE,
        )
        self._compaction_coordinator.schedule(
            session_id,
            revision=transcript.revision,
            task_coro=self._run_compaction(
                session_id=session_id,
                transcript=transcript,
                turns=candidate_turns,
                settings=settings,
            ),
        )

    async def wait_for_session(self, session_id: str) -> None:
        """等待指定 session 的后台任务完成。

        Args:
            session_id: 会话 ID。

        Returns:
            无。

        Raises:
            无。
        """

        await self._compaction_coordinator.wait_for_session(session_id)

    def _build_memory_block(
        self,
        *,
        transcript: ConversationTranscript,
        total_budget: int,
    ) -> tuple[str, int]:
        """构建 ``[Conversation Memory]`` system block。

        ``pinned_state`` 永远完整渲染、不消耗 ``total_budget``；``episode summaries``
        从最新到最旧消费 ``total_budget``，预算耗尽即停。

        Args:
            transcript: 当前 transcript。
            total_budget: 单总池预算（pinned_state 与最近 N 轮 raw turn 不在此池）。

        Returns:
            ``(memory_block, episodic_used_tokens)``：渲染后的 block；以及 episode
            summaries 实际占用的 token 数（用于把剩余预算让给 raw turn 回放）。
            block 无内容时返回 ``("", 0)``。

        Raises:
            无。
        """

        pieces: list[str] = []
        pinned_block = _render_pinned_state_block(transcript.pinned_state)
        if pinned_block:
            pieces.append(pinned_block)

        episodic_used_tokens = 0
        episode_pieces: list[str] = []
        budget_remaining = max(0, total_budget)
        for episode in reversed(transcript.episodes):
            rendered = _render_episode_summary(episode)
            rendered_tokens = _estimate_tokens(rendered)
            if rendered_tokens > budget_remaining:
                if not episode_pieces:
                    # 单条 episode 即超预算时仍保留至少一条最新 episode，避免 episodic
                    # 层完全失效；token 仍按真实占用记账供调用方核账。
                    episode_pieces.append(rendered)
                    episodic_used_tokens += rendered_tokens
                break
            episode_pieces.append(rendered)
            episodic_used_tokens += rendered_tokens
            budget_remaining -= rendered_tokens
        if episode_pieces:
            episode_pieces.reverse()
            pieces.append("Episode Summaries:\n" + "\n\n".join(episode_pieces))
        if not pieces:
            return "", 0
        return "[Conversation Memory]\n" + "\n\n".join(pieces), episodic_used_tokens

    def _compile_raw_turns(self, turns: tuple[WorkingMemoryTurnView, ...]) -> list[AgentMessage]:
        """将 working memory raw turns 编译为消息列表。

        Args:
            turns: 已选中的 working memory 视图。

        Returns:
            历史消息列表。

        Raises:
            无。
        """

        messages: list[AgentMessage] = []
        for turn in turns:
            messages.append(build_user_chat_message(turn.user_text))
            messages.append(build_assistant_chat_message(content=turn.assistant_text))
        return messages

    def _select_compaction_candidate(
        self,
        *,
        transcript: ConversationTranscript,
        settings: ConversationMemorySettings,
        max_context_tokens: int,
        system_prompt_tokens: int = 0,
        user_text_tokens: int = 0,
    ) -> tuple[ConversationTurnRecord, ...]:
        """选择本次可压缩的 raw turn 片段。

        触发公式（占模型窗口百分比，跨档位自动伸缩）::

            window_used = system_prompt + pinned_state
                         + actual_episodic_in_prompt
                         + uncompressed_raw_turns + current_user_text
            should_compact = window_used > max_context_tokens
                             * compaction_trigger_context_ratio

        说明：``actual_episodic_in_prompt`` 通过复用 ``_build_memory_block`` 的渲染逻辑
        在当前 ``total_budget`` 下计算"本轮会真正送进 prompt 的 episode token"，从而：
        - 既避免把全量 ``transcript.episodes`` 计入触发公式带来的抖动（fix #2 原意）；
        - 也避免无视 ``_build_memory_block`` 的"最新 episode 即使超 budget 也保留一条"
          兜底分支造成的真实 prompt 体积低估（round2 review #1）。

        Args:
            transcript: 当前 transcript。
            settings: 分层记忆配置。
            max_context_tokens: 当前模型最大上下文 token；``<= 0`` 时跳过 token 触发。
            system_prompt_tokens: 当前轮 system prompt 估算 token 数。
            user_text_tokens: 当前轮用户输入估算 token 数。

        Returns:
            待压缩 turn 元组；不满足阈值时返回空元组。

        Raises:
            无。
        """

        uncompressed_turns = transcript.turns[transcript.compacted_turn_count :]
        if len(uncompressed_turns) <= settings.compaction_tail_preserve_turns:
            return ()
        if max_context_tokens <= 0:
            # 模型窗口未知时无法判定占比，直接放弃 token 触发；防御性兜底。
            return ()

        pinned_tokens = _estimate_tokens(_render_pinned_state_block(transcript.pinned_state))
        uncompressed_tokens = sum(_estimate_turn_tokens(turn) for turn in uncompressed_turns)

        # 用 build_messages 同款逻辑估算本轮实际会送的 episode token：
        # 复用 _build_memory_block，使触发判定与真实 prompt 体积保持一致。
        total_budget = _resolve_memory_total_budget(
            settings=settings,
            max_context_tokens=max_context_tokens,
        )
        _, episodic_in_prompt_tokens = self._build_memory_block(
            transcript=transcript,
            total_budget=total_budget,
        )

        window_used = (
            max(0, system_prompt_tokens)
            + pinned_tokens
            + episodic_in_prompt_tokens
            + uncompressed_tokens
            + max(0, user_text_tokens)
        )
        threshold = int(max_context_tokens * settings.compaction_trigger_context_ratio)
        if window_used <= threshold:
            return ()

        cutoff = len(uncompressed_turns) - settings.compaction_tail_preserve_turns
        if cutoff <= 0:
            return ()
        return tuple(uncompressed_turns[:cutoff])

    async def _run_compaction(
        self,
        *,
        session_id: str,
        transcript: ConversationTranscript,
        turns: tuple[ConversationTurnRecord, ...],
        settings: ConversationMemorySettings,
    ) -> None:
        """执行单次后台压缩并尝试写回 transcript。

        Args:
            session_id: 会话 ID。
            transcript: 调度时的 transcript 快照。
            turns: 待压缩的 turn 片段。
            settings: 分层记忆配置。

        Returns:
            无。

        Raises:
            无。异常会被记录并吞掉。
        """

        try:
            Log.verbose(
                f"开始后台 compaction: session_id={session_id}, revision={transcript.revision}, turns={len(turns)}",
                module=MODULE,
            )
            result = await self._episodic_memory_compressor.compress(
                session_id=session_id,
                transcript=transcript,
                turns=turns,
                settings=settings,
            )
            if result is None:
                Log.verbose(
                    f"后台 compaction 返回空结果，跳过写回: session_id={session_id}, revision={transcript.revision}",
                    module=MODULE,
                )
                return
            live_archive = self._archive_store.load(session_id)
            if live_archive is None:
                Log.debug(f"后台 compaction 写回前 archive 已不存在: session_id={session_id}", module=MODULE)
                return
            current_transcript = live_archive.runtime_transcript
            if current_transcript.revision != transcript.revision:
                # REVIEW NOTE (finding 078, 驳回留观):
                # revision 冲突时本次后台压缩结果被丢弃且**不在此处退避重试**。
                # 这是刻意的选择，不是遗漏：
                # 1. 自愈链路存在 —— 写回失败时 compacted_turn_count 不前进；
                #    下一次 turn 的 schedule_compaction 会基于相同阈值重新选出
                #    相同候选 turns 并再跑一次 _run_compaction，相当于"下一轮自动重试"。
                # 2. 在此处加重试会把 Coordinator 从"单任务句柄管理"升级为"退避/
                #    抢占调度器"，与新到来的 schedule 产生竞态窗口（cancel vs 重入）。
                # 3. 失效场景只有"session 永久闲置"，此时 compaction 的价值本身也消失。
                # 下一位 reviewer：若要改动此分支，请先核对上述自愈假设是否仍成立。
                #
                # archive 形态硬规则：业务层 stale-check 比较的是
                # ``live.runtime_transcript.revision``（与运行态语义对齐），不是
                # ``live.revision``（那是物理层 archive 乐观锁的参与方）。两层语义
                # 必须保持独立——禁止偷换为 archive.revision 比较，否则在
                # history_archive 单独被推进、runtime_transcript 未变的场景下会
                # 误判 stale。
                Log.verbose(
                    f"conversation compaction 结果已过期，跳过写回：session_id={session_id}",
                    module=MODULE,
                )
                return
            next_transcript = current_transcript.replace_memory(
                pinned_state=result.pinned_state_patch.apply_to(current_transcript.pinned_state),
                episodes=(*current_transcript.episodes, result.episode_summary),
                compacted_turn_count=current_transcript.compacted_turn_count + len(turns),
            )
            # archive 形态：只替换运行态子视图，``history_archive`` 原样保留。
            next_archive = live_archive.with_runtime_transcript(next_transcript)
            self._archive_store.save(next_archive, expected_revision=live_archive.revision)
            self._retrieval_index.index_transcript(next_transcript)
            Log.verbose(
                f"后台 compaction 写回成功: session_id={session_id}, revision={next_transcript.revision}, compacted_turn_count={next_transcript.compacted_turn_count}",
                module=MODULE,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - 防御性日志
            Log.warning(f"conversation compaction 失败：{exc}", module=MODULE)


__all__ = [
    "ConversationCompactionCoordinator",
    "ConversationMemoryManagerProtocol",
    "ConversationRetrievalIndexProtocol",
    "ConversationPinnedStatePatch",
    "DefaultConversationMemoryManager",
    "DefaultEpisodicMemoryCompressor",
    "DefaultWorkingMemoryPolicy",
    "DurableMemoryStoreProtocol",
    "EpisodicMemoryCompressorProtocol",
    "NullConversationRetrievalIndex",
    "NullDurableMemoryStore",
    "WorkingMemoryPolicyProtocol",
]
