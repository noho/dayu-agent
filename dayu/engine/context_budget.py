"""Agent 侧上下文预算治理原语。

该模块只承载 Agent 在推理循环中做全局预算决策所需的纯原语：
- 运行时预算状态 `ContextBudgetState`
- 工具结果注入前的字符预算估算
- 工具结果按预算公平裁剪

注意：这里不处理工具 schema 驱动的截断与 `fetch_more`，那部分仍属于
`ToolRegistry`/`TruncationManager` 的职责。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

CHARS_PER_TOKEN = 4
"""字符到 token 的保守换算比例。"""

MIN_RESULT_CHARS = 4000
"""预测性预算裁剪后，单个工具结果最少保留的字符数。

取 4000 字符是因为财报表格和段落的有效信息密度较高，
低于此阈值时 LLM 容易因信息不足而产生幻觉或要求重新调用工具。
"""

PREDICTIVE_OVERHEAD_TOKENS = 4096
"""估算工具结果注入消息时额外保留的结构开销。

4096 token 覆盖 JSON 消息结构、role/tool_call_id 元数据以及
tokenizer 对中文/特殊字符的膨胀余量（实测结构开销在 2k-3k token，
取 4096 留出安全裕度）。
"""


def _coerce_usage_token_count(raw_value: object) -> int:
    """把 usage 字段中的原始 token 值收敛为整数。

    Args:
        raw_value: usage 字段中的原始值。

    Returns:
        合法的整数 token 数；无法解析时返回 0。

    Raises:
        无。
    """

    if raw_value is None:
        return 0
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return int(raw_value)
    if isinstance(raw_value, str):
        normalized_value = raw_value.strip()
        if not normalized_value:
            return 0
        return int(normalized_value)
    return 0


@dataclass
class ContextBudgetState:
    """上下文预算运行时状态。

    Args:
        max_context_tokens: 模型最大上下文 token 数；0 表示预算治理未启用。
        max_output_tokens: 模型最大输出 token 数；0 表示未知。
        soft_limit_ratio: 软阈值比例，超过后主动压缩。
        hard_limit_ratio: 硬阈值比例，超过后紧急治理。
        current_prompt_tokens: 最近一轮 prompt token 数。
        latest_completion_tokens: 最近一轮 completion token 数。
        total_prompt_tokens: 累计 prompt token 数。
        total_completion_tokens: 累计 completion token 数。
        iteration_count: 已完成的 agent iteration 计数。
        compaction_count: 已执行压缩次数。
        continuation_count: 已执行续写次数。

    Returns:
        无。

    Raises:
        无。
    """

    max_context_tokens: int = 0
    max_output_tokens: int = 0
    soft_limit_ratio: float = 0.75
    hard_limit_ratio: float = 0.90
    current_prompt_tokens: int = 0
    latest_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    iteration_count: int = 0
    compaction_count: int = 0
    continuation_count: int = 0

    @property
    def is_budget_enabled(self) -> bool:
        """预算治理是否已启用。

        Args:
            无。

        Returns:
            当 `max_context_tokens > 0` 时返回 `True`。

        Raises:
            无。
        """

        return self.max_context_tokens > 0

    @property
    def soft_limit_tokens(self) -> int:
        """返回软阈值 token 数。

        Args:
            无。

        Returns:
            软阈值 token 数；预算未启用时返回 0。

        Raises:
            无。
        """

        if not self.is_budget_enabled:
            return 0
        return int(self.max_context_tokens * self.soft_limit_ratio)

    @property
    def hard_limit_tokens(self) -> int:
        """返回硬阈值 token 数。

        Args:
            无。

        Returns:
            硬阈值 token 数；预算未启用时返回 0。

        Raises:
            无。
        """

        if not self.is_budget_enabled:
            return 0
        return int(self.max_context_tokens * self.hard_limit_ratio)

    @property
    def is_over_soft_limit(self) -> bool:
        """当前 prompt 是否超过软阈值。

        Args:
            无。

        Returns:
            是否超过软阈值。

        Raises:
            无。
        """

        return self.is_budget_enabled and self.current_prompt_tokens >= self.soft_limit_tokens

    @property
    def is_over_hard_limit(self) -> bool:
        """当前 prompt 是否超过硬阈值。

        Args:
            无。

        Returns:
            是否超过硬阈值。

        Raises:
            无。
        """

        return self.is_budget_enabled and self.current_prompt_tokens >= self.hard_limit_tokens

    def record_usage(self, usage: Mapping[str, object]) -> None:
        """根据 Runner usage 更新预算状态。

        Args:
            usage: usage 字典，支持 `prompt_tokens` 与 `completion_tokens`。

        Returns:
            无。

        Raises:
            无。
        """

        prompt_tokens = _coerce_usage_token_count(usage.get("prompt_tokens"))
        completion_tokens = _coerce_usage_token_count(usage.get("completion_tokens"))
        self.current_prompt_tokens = prompt_tokens
        self.latest_completion_tokens = completion_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.iteration_count += 1


class ToolResultBudgetCapper:
    """工具结果预算裁剪器。

    该协作者只处理一件事：在工具结果已经序列化为字符串之后、注入下一轮
    `tool` message 之前，按当前 Agent 的剩余预算进行公平裁剪。
    """

    @staticmethod
    def estimate_chars_to_tokens(chars: int) -> int:
        """将字符数转换为 token 数的保守估计。

        Args:
            chars: 字符数。

        Returns:
            估计 token 数。

        Raises:
            无。
        """

        return chars // CHARS_PER_TOKEN

    @staticmethod
    def truncate_result_str(result_str: str, max_chars: int) -> str:
        """截断单个过大的工具结果字符串。

        Args:
            result_str: 原始结果字符串。
            max_chars: 最大允许保留字符数。

        Returns:
            截断后的字符串；若未超限则原样返回。

        Raises:
            无。
        """

        if len(result_str) <= max_chars:
            return result_str
        original_len = len(result_str)
        truncated = result_str[:max_chars]
        note = (
            f'\n\n[CONTEXT_BUDGET_TRUNCATED: '
            f'original={original_len} chars, kept={max_chars} chars. '
            f'Please narrow your query scope or request a smaller result set.]'
        )
        return truncated + note

    @classmethod
    def cap_results_for_budget(
        cls,
        serialized_pairs: list[tuple[dict[str, object], str]],
        budget_state: ContextBudgetState,
    ) -> tuple[list[tuple[dict[str, object], str]], bool]:
        """按上下文预算截断过大的工具结果。

        Args:
            serialized_pairs: `(tool_call_dict, serialized_result_str)` 列表。
            budget_state: 当前上下文预算状态。

        Returns:
            `(capped_pairs, was_capped)`，分别表示裁剪后的结果和是否发生裁剪。

        Raises:
            无。
        """

        available_tokens = max(
            0,
            budget_state.soft_limit_tokens
            - budget_state.current_prompt_tokens
            - budget_state.latest_completion_tokens,
        )
        available_chars = available_tokens * CHARS_PER_TOKEN

        indexed_sizes: list[tuple[int, int]] = [
            (index, len(result_str))
            for index, (_, result_str) in enumerate(serialized_pairs)
            if result_str
        ]
        total_chars = sum(size for _, size in indexed_sizes)
        if total_chars <= available_chars:
            return serialized_pairs, False

        indexed_sizes.sort(key=lambda item: item[1])
        caps: dict[int, int] = {}
        remaining_budget = available_chars
        remaining_count = len(indexed_sizes)
        for index, size in indexed_sizes:
            # 这里采用升序公平分配：小结果优先完整保留，剩余预算让给大结果，
            # 这样比平均裁剪更能保住整体信息量。
            fair_share = max(MIN_RESULT_CHARS, remaining_budget // max(1, remaining_count))
            actual = min(size, fair_share)
            caps[index] = actual
            remaining_budget = max(0, remaining_budget - actual)
            remaining_count -= 1

        capped = False
        new_pairs: list[tuple[dict[str, object], str]] = []
        for index, (tool_call, result_str) in enumerate(serialized_pairs):
            if index in caps and len(result_str) > caps[index]:
                result_str = cls.truncate_result_str(result_str, caps[index])
                capped = True
            new_pairs.append((tool_call, result_str))
        return new_pairs, capped


__all__ = [
    "CHARS_PER_TOKEN",
    "MIN_RESULT_CHARS",
    "PREDICTIVE_OVERHEAD_TOKENS",
    "ContextBudgetState",
    "ToolResultBudgetCapper",
]
