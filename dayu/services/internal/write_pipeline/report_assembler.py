"""最终报告拼装模块。

该模块只负责按模板顺序拼装最终报告 Markdown，
不承担任何文件读写职责。
"""

from __future__ import annotations

from dayu.services.internal.write_pipeline.audit_formatting import _strip_evidence_section
from dayu.services.contracts import WriteRunConfig
from dayu.services.internal.write_pipeline.models import ChapterResult
from dayu.services.internal.write_pipeline.template_parser import TemplateLayout, build_report_markdown

_OVERVIEW_CHAPTER_TITLE = "投资要点概览"
_SOURCE_CHAPTER_TITLE = "来源清单"


class ReportAssembler:
    """最终报告拼装器。"""

    def __init__(self, *, write_config: WriteRunConfig) -> None:
        """初始化报告拼装器。

        Args:
            write_config: 写作运行配置。

        Returns:
            无。

        Raises:
            无。
        """

        self._write_config = write_config

    def assemble_report(
        self,
        layout: TemplateLayout,
        chapter_results: dict[str, ChapterResult],
        source_chapter_markdown: str | None,
        *,
        company_name: str,
    ) -> str:
        """按模板顺序拼装最终报告。

        Args:
            layout: 模板布局对象。
            chapter_results: 章节结果映射。
            source_chapter_markdown: 来源清单章节；模板未声明该章节时可为 ``None``。
            company_name: 公司名称，用于替换 preface 中的占位符。

        Returns:
            最终报告 Markdown。

        Raises:
            KeyError: 章节缺失时抛出。
        """

        ticker = self._write_config.ticker
        preface = (
            layout.preface_skeleton.replace("[公司名称]", company_name)
            .replace("(TICKER)", f"({ticker})")
            .rstrip()
        )
        while preface.endswith("---"):
            preface = preface[:-3].rstrip()

        ordered_chapters: list[str] = []
        for chapter in layout.chapters:
            if chapter.title == _SOURCE_CHAPTER_TITLE:
                if source_chapter_markdown:
                    ordered_chapters.append(source_chapter_markdown)
                else:
                    ordered_chapters.append(chapter.skeleton)
                continue
            result = chapter_results.get(chapter.title)
            if result is None or not result.content:
                ordered_chapters.append(chapter.skeleton)
            elif chapter.title == _OVERVIEW_CHAPTER_TITLE:
                ordered_chapters.append(_strip_evidence_section(result.content))
            else:
                ordered_chapters.append(result.content)
        return build_report_markdown(preface, ordered_chapters)
