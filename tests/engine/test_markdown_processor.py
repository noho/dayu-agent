"""MarkdownProcessor 单元测试。"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Mapping, Optional
from unittest.mock import Mock

import pytest

from dayu.engine.processors import markdown_processor
from dayu.engine.processors.markdown_processor import MarkdownProcessor
from dayu.engine.processors.source import Source


def _hit_section_ref(hit: Mapping[str, object]) -> str | None:
    """安全读取搜索命中的 section_ref。"""

    value = hit.get("section_ref")
    return value if isinstance(value, str) else None


def _hit_snippet(hit: Mapping[str, object]) -> str:
    """安全读取搜索命中的 snippet。"""

    value = hit.get("snippet")
    return value if isinstance(value, str) else ""


class DummySource:
    """测试用 Source。"""

    def __init__(self, path: Path, *, uri: Optional[str] = None, media_type: Optional[str] = None) -> None:
        """初始化测试 Source。

        Args:
            path: 本地文件路径。
            uri: 可选 URI。
            media_type: 可选媒体类型。

        Returns:
            无。

        Raises:
            ValueError: 路径为空时抛出。
        """

        if not path:
            raise ValueError("path 不能为空")
        self._path = path
        self.uri = uri or str(path)
        self.media_type = media_type
        self.content_length = None
        self.etag = None

    def open(self) -> BinaryIO:
        """打开文件流。"""

        return self._path.open("rb")

    def materialize(self, suffix: Optional[str] = None) -> Path:
        """返回本地路径。"""

        del suffix
        return self._path


@pytest.mark.unit
def test_markdown_processor_supports_md_suffix_and_media_type(tmp_path: Path) -> None:
    """验证 supports 判定逻辑。"""

    md_path = tmp_path / "sample.md"
    md_path.write_text("# title\n", encoding="utf-8")

    source_by_suffix = DummySource(md_path, uri="local://sample.md", media_type=None)
    source_by_type = DummySource(md_path, uri="local://sample.txt", media_type="text/markdown")

    assert MarkdownProcessor.supports(source_by_suffix) is True
    assert MarkdownProcessor.supports(source_by_type) is True


@pytest.mark.unit
def test_markdown_processor_section_table_and_search_flow(tmp_path: Path) -> None:
    """验证章节、表格、章节占位与搜索主流程。"""

    md_path = tmp_path / "report.md"
    md_path.write_text(
        "\n".join(
            [
                "# 总览",
                "收入增长明显。",
                "| 项目 | 金额 |",
                "| --- | --- |",
                "| Revenue | 100 |",
                "## 风险",
                "需要关注原材料价格波动、汇率波动以及需求波动。",
            ]
        ),
        encoding="utf-8",
    )

    processor = MarkdownProcessor(DummySource(md_path, media_type="text/markdown"))

    sections = processor.list_sections()
    assert len(sections) == 2
    assert sections[0]["ref"] == "s_0001"
    assert sections[1]["parent_ref"] == "s_0001"

    tables = processor.list_tables()
    assert len(tables) == 1
    assert tables[0]["table_ref"] == "t_0001"
    assert tables[0]["table_type"] == "layout"

    section_content = processor.read_section("s_0001")
    assert "[[t_0001]]" in section_content["content"]
    assert section_content["tables"] == ["t_0001"]

    table_content = processor.read_table("t_0001")
    assert table_content["data_format"] == "records"
    assert table_content["columns"] == ["项目", "金额"]
    assert isinstance(table_content["data"], list)
    assert table_content["data"][0]["项目"] == "Revenue"

    hits = processor.search("波动", within_ref="s_0002")
    assert len(hits) <= 2
    assert _hit_section_ref(hits[0]) == "s_0002"
    assert all("波动" in _hit_snippet(hit) for hit in hits)
    assert all(len(_hit_snippet(hit)) <= 360 for hit in hits)


@pytest.mark.unit
def test_markdown_processor_fallback_full_text_when_no_heading(tmp_path: Path) -> None:
    """验证无标题场景会返回全文章节。"""

    md_path = tmp_path / "plain.md"
    md_path.write_text("纯文本内容\n无标题", encoding="utf-8")

    processor = MarkdownProcessor(DummySource(md_path))
    sections = processor.list_sections()

    assert len(sections) == 1
    assert sections[0]["ref"] == "s_0001"
    content = processor.read_section("s_0001")
    assert content["contains_full_text"] is True
    assert "纯文本内容" in content["content"]


@pytest.mark.unit
def test_markdown_processor_search_reuses_section_render_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证重复 search 不会重复渲染相同章节。

    Args:
        tmp_path: pytest 临时目录。
        monkeypatch: pytest monkeypatch。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    md_path = tmp_path / "cache_search.md"
    md_path.write_text(
        "\n".join(
            [
                "# Section A",
                "Revenue keeps growing.",
                "## Section B",
                "Growth remains strong for revenue and margin.",
            ]
        ),
        encoding="utf-8",
    )
    processor = MarkdownProcessor(DummySource(md_path, media_type="text/markdown"))

    original_render = markdown_processor._render_section_content
    tracked_render = Mock(side_effect=original_render)
    monkeypatch.setattr(markdown_processor, "_render_section_content", tracked_render)

    hits_first = processor.search("revenue")
    hits_second = processor.search("growth")

    assert hits_first
    assert hits_second
    # 两个章节在第一次 search 已渲染，第二次应复用缓存。
    assert tracked_render.call_count == 2
