"""``base/fact_rules.md`` 共享 prompt 资产的关键词回归测试。

该测试只对资产做最小回归保护：确保"时间表达"小节及其关键词没有被误删或漂移。
不做完整文本快照，避免文案微调即破坏测试。
"""

from __future__ import annotations

from pathlib import Path

import pytest

_FACT_RULES_PATH = (
    Path(__file__).resolve().parents[2]
    / "dayu"
    / "config"
    / "prompts"
    / "base"
    / "fact_rules.md"
)


@pytest.mark.unit
def test_fact_rules_contains_time_expression_section() -> None:
    """验证 fact_rules.md 含有"时间表达"小节及关键约束词。

    覆盖 issue #125：相对时间归一化规则必须存在于共享 SOUL 层 prompt 中，
    保证所有引用 base/fact_rules.md 的 scene 都受到约束。
    """

    content = _FACT_RULES_PATH.read_text(encoding="utf-8")

    assert "## 时间表达" in content, "fact_rules.md 必须保留'时间表达'小节"

    required_keywords = (
        "绝对日期",
        "今天",
    )
    missing = [keyword for keyword in required_keywords if keyword not in content]
    assert not missing, f"fact_rules.md '时间表达'小节缺少关键约束词: {missing}"
