"""处理器性能打点工具。

本模块提供可选性能打点能力，用于在不改变业务行为的前提下输出
处理器各阶段耗时。打点默认关闭，通过环境变量启用：

- `FINS_PROCESSOR_PROFILE=1`：开启打点日志；
- 其他值或未设置：关闭打点日志。
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from dayu.contracts.env_keys import FINS_PROCESSOR_PROFILE_ENV
from dayu.log import Log

MODULE = "ENGINE.PERF_UTILS"
_PROFILE_ENABLED_VALUES = {"1", "true", "yes", "on"}


def is_processor_profile_enabled() -> bool:
    """判断是否启用处理器性能打点。

    Args:
        无。

    Returns:
        若环境变量启用则返回 `True`，否则返回 `False`。

    Raises:
        RuntimeError: 环境变量读取失败时抛出。
    """

    raw_value = os.getenv(FINS_PROCESSOR_PROFILE_ENV, "")
    normalized = str(raw_value).strip().lower()
    return normalized in _PROFILE_ENABLED_VALUES


@dataclass
class ProcessorStageProfiler:
    """处理器阶段耗时统计器。

    Attributes:
        component: 组件名称（通常为处理器类名）。
        enabled: 是否启用打点。
        records: 阶段耗时记录，单位毫秒。
    """

    component: str
    enabled: bool = False
    records: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, stage_name: str) -> Iterator[None]:
        """统计单个阶段耗时。

        Args:
            stage_name: 阶段名称。

        Returns:
            上下文管理器迭代器。

        Raises:
            RuntimeError: 耗时计算失败时抛出。
        """

        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.records[stage_name] = self.records.get(stage_name, 0.0) + elapsed_ms

    def log_summary(self, extra: str = "") -> None:
        """输出阶段耗时汇总日志。

        Args:
            extra: 附加说明文本。

        Returns:
            无。

        Raises:
            RuntimeError: 日志序列化失败时抛出。
        """

        if not self.enabled or not self.records:
            return
        ordered = sorted(self.records.items(), key=lambda item: item[0])
        summary = ", ".join(f"{name}={value:.2f}ms" for name, value in ordered)
        suffix = f" | {extra}" if extra else ""
        Log.info(f"processor_profile[{self.component}] {summary}{suffix}", module=MODULE)
