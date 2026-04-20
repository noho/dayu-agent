"""运行摘要构建模块。

该模块只负责从章节结果聚合最终运行摘要，
不承担任何文件读写职责。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from dayu.services.contracts import WriteRunConfig
from dayu.services.internal.write_pipeline.models import ChapterResult


class ExecutionSummaryBuilder:
    """运行摘要构建器。"""

    def __init__(self, *, write_config: WriteRunConfig) -> None:
        """初始化运行摘要构建器。

        Args:
            write_config: 写作运行配置。

        Returns:
            无。

        Raises:
            无。
        """

        self._write_config = write_config

    def build_summary(
        self,
        chapter_results: dict[str, ChapterResult],
        *,
        output_file: Path,
        success_predicate: Callable[[ChapterResult | None], bool],
    ) -> dict[str, Any]:
        """生成运行摘要。

        Args:
            chapter_results: 章节结果映射。
            output_file: 最终报告路径。
            success_predicate: 判断章节是否成功的谓词函数。

        Returns:
            摘要字典。

        Raises:
            无。
        """

        failed = [
            {
                "title": result.title,
                "reason": result.failure_reason,
                "retry_count": result.retry_count,
            }
            for result in chapter_results.values()
            if not success_predicate(result)
        ]
        return {
            "ticker": self._write_config.ticker,
            "output_file": str(output_file),
            "chapter_count": len(chapter_results),
            "failed_count": len(failed),
            "failed_chapters": failed,
        }
