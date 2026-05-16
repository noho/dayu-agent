"""MinerUProcessor 单元测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dayu.engine.processors.mineru_processor import MineruProcessor
from dayu.fins.storage.local_file_source import LocalFileSource


def _make_source(path: Path, *, media_type: str = "application/json") -> LocalFileSource:
    """构建本地 Source。

    Args:
        path: 文件路径。
        media_type: 媒体类型。

    Returns:
        Source 实例。

    Raises:
        OSError: 构建失败时抛出。
    """

    return LocalFileSource(
        path=path,
        uri=f"local://{path.name}",
        media_type=media_type,
        content_length=path.stat().st_size,
        etag=None,
    )


def _write_mineru_json(
    tmp_path: Path,
    name: str,
    *,
    raw_markdown: str = "",
    sections: list[dict[str, object]] | None = None,
    tables: list[dict[str, object]] | None = None,
) -> Path:
    """创建 MinerU JSON 测试文件。

    Args:
        tmp_path: pytest 临时目录。
        name: 文件名（必须包含 _mineru.json 后缀）。
        raw_markdown: 原始 markdown 内容。
        sections: 章节列表。
        tables: 表格列表。

    Returns:
        创建的文件路径。
    """

    data: dict[str, object] = {"backend": "mineru_local"}
    if raw_markdown:
        data["raw_markdown"] = raw_markdown
    if sections is not None:
        data["sections"] = sections
    if tables is not None:
        data["tables"] = tables

    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


# ============================================================================
# 正常路径
# ============================================================================


@pytest.mark.unit
def test_get_parser_version() -> None:
    """验证 get_parser_version 返回正确版本字符串。

    Args:
        无。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    version = MineruProcessor.get_parser_version()
    assert version == "mineru_processor_v1.0.0"
    assert isinstance(version, str)


@pytest.mark.unit
def test_supports_matching_suffix(tmp_path: Path) -> None:
    """验证 supports 匹配以 _mineru.json 结尾的文件。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "annual_report_mineru.json"
    json_path.write_text("{}", encoding="utf-8")
    source = _make_source(json_path)

    assert MineruProcessor.supports(source) is True


@pytest.mark.unit
def test_supports_non_matching_suffix(tmp_path: Path) -> None:
    """验证 supports 不匹配非 _mineru.json 结尾的文件。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "plain_docling.json"
    json_path.write_text("{}", encoding="utf-8")
    source = _make_source(json_path)

    assert MineruProcessor.supports(source) is False


@pytest.mark.unit
def test_supports_case_insensitive(tmp_path: Path) -> None:
    """验证 supports 不区分 URI 大小写。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "REPORT_mineru.JSON"
    json_path.write_text("{}", encoding="utf-8")
    # 使用大写 URI（supports 内部会 lower()）
    source = LocalFileSource(
        path=json_path,
        uri="local://REPORT_mineru.JSON",
        media_type="application/json",
        content_length=json_path.stat().st_size,
        etag=None,
    )

    assert MineruProcessor.supports(source) is True


@pytest.mark.unit
def test_init_loads_json(tmp_path: Path) -> None:
    """验证 __init__ 正常加载 JSON 文件。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "sample_mineru.json",
        raw_markdown="# Doc Title",
        sections=[
            {"title": "Section 1", "level": 1, "content": "Content 1", "page_idx": 0},
        ],
        tables=[
            {"caption": "Table 1", "html": "<table><tr><td>A</td></tr></table>", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    assert processor._source_path == json_path
    assert processor._data["backend"] == "mineru_local"


@pytest.mark.unit
def test_list_sections_returns_summaries(tmp_path: Path) -> None:
    """验证 list_sections 返回正确的章节摘要列表。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "sec_mineru.json",
        sections=[
            {"title": "概述", "level": 1, "content": "这是概述内容。", "page_idx": 0},
            {"title": "财务数据", "level": 2, "content": "以下是财务表格。", "page_idx": 1},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    summaries = processor.list_sections()
    assert len(summaries) == 2
    assert summaries[0]["ref"] == "s_0001"
    assert summaries[0]["title"] == "概述"
    assert summaries[0]["level"] == 1
    assert summaries[0]["preview"] == "这是概述内容。"

    assert summaries[1]["ref"] == "s_0002"
    assert summaries[1]["title"] == "财务数据"
    assert summaries[1]["level"] == 2


@pytest.mark.unit
def test_list_tables_returns_summaries(tmp_path: Path) -> None:
    """验证 list_tables 返回正确的表格摘要列表。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "tbl_mineru.json",
        tables=[
            {"caption": "资产负债表", "html": "<table><tr><td>现金</td><td>100</td></tr></table>", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    summaries = processor.list_tables()
    assert len(summaries) == 1
    assert summaries[0]["table_ref"] == "t_0001"
    assert summaries[0]["caption"] == "资产负债表"
    assert summaries[0]["row_count"] == 1
    assert summaries[0]["col_count"] == 2


@pytest.mark.unit
def test_read_section_valid_ref(tmp_path: Path) -> None:
    """验证 read_section 返回有效 ref 的章节内容。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "read_sec_mineru.json",
        sections=[
            {"title": "第一章", "level": 1, "content": "第一章正文内容。", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    content = processor.read_section("s_0001")
    assert content["ref"] == "s_0001"
    assert content["title"] == "第一章"
    assert "第一章正文内容" in content["content"]
    assert content["word_count"] > 0
    assert content["contains_full_text"] is True


@pytest.mark.unit
def test_read_section_with_tables_appends_placeholders(tmp_path: Path) -> None:
    """验证 read_section 在关联表格时追加占位符。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "sec_tbl_mineru.json",
        sections=[
            {"title": "财报", "level": 1, "content": "以下是财务摘要", "page_idx": 0},
        ],
        tables=[
            {"caption": "利润表", "html": "<table><tr><td>收入</td><td>500</td></tr></table>", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    content = processor.read_section("s_0001")
    # 占位符应追加到内容末尾
    assert "[[t_0001]]" in content["content"]
    assert content["tables"] == ["t_0001"]


@pytest.mark.unit
def test_read_table_valid_ref(tmp_path: Path) -> None:
    """验证 read_table 返回有效 ref 的表格内容。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "read_tbl_mineru.json",
        tables=[
            {
                "caption": "利润表",
                "html": "<table><tr><th>项目</th><th>金额</th></tr><tr><td>收入</td><td>1000</td></tr></table>",
                "page_idx": 0,
            },
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    table_content = processor.read_table("t_0001")
    assert table_content["table_ref"] == "t_0001"
    assert table_content["caption"] == "利润表"
    assert table_content["data_format"] == "markdown"
    assert table_content["row_count"] == 2
    assert table_content["col_count"] == 2


@pytest.mark.unit
def test_search_finds_match(tmp_path: Path) -> None:
    """验证 search 能匹配到关键词。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "search_mineru.json",
        sections=[
            {"title": "收入", "level": 1, "content": "公司2024年营收增长20%。", "page_idx": 0},
            {"title": "成本", "level": 2, "content": "运营成本有所下降。", "page_idx": 1},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    hits = processor.search("营收", within_ref=None)
    assert len(hits) >= 1
    assert any("营收" in hit["snippet"] or "营收" in hit["section_title"] for hit in hits)


@pytest.mark.unit
def test_search_no_match_returns_empty(tmp_path: Path) -> None:
    """验证 search 无匹配时返回空列表。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "no_match_mineru.json",
        sections=[
            {"title": "报告", "level": 1, "content": "一些无关内容", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    hits = processor.search("不存在的关键词", within_ref=None)
    assert hits == []


@pytest.mark.unit
def test_search_empty_query_returns_empty(tmp_path: Path) -> None:
    """验证 search 空查询时返回空列表。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "empty_q_mineru.json",
        sections=[
            {"title": "报告", "level": 1, "content": "内容", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    hits = processor.search("", within_ref=None)
    assert hits == []


@pytest.mark.unit
def test_get_full_text_concatenates(tmp_path: Path) -> None:
    """验证 get_full_text 拼接 raw_markdown 与章节内容。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "fulltext_mineru.json",
        raw_markdown="# 年度报告\n\n这是一份财务报告。",
        sections=[
            {"title": "概述", "level": 1, "content": "公司表现良好。", "page_idx": 0},
            {"title": "财务", "level": 2, "content": "净利润增长。", "page_idx": 1},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    full_text = processor.get_full_text()
    assert "年度报告" in full_text
    assert "公司表现良好。" in full_text
    assert "净利润增长。" in full_text


@pytest.mark.unit
def test_get_full_text_caches_result(tmp_path: Path) -> None:
    """验证 get_full_text 使用缓存，多次调用返回相同结果。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "cache_mineru.json",
        raw_markdown="# 缓存的全文",
        sections=[
            {"title": "章节", "level": 1, "content": "缓存内容", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    first = processor.get_full_text()
    second = processor.get_full_text()
    assert first == second
    assert processor._full_text_cache is not None
    assert processor._full_text_cache == first


@pytest.mark.unit
def test_get_full_text_with_table_markers(tmp_path: Path) -> None:
    """验证 get_full_text_with_table_markers 返回空字符串。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "markers_mineru.json",
        sections=[
            {"title": "章节", "level": 1, "content": "内容", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    result = processor.get_full_text_with_table_markers()
    assert result == ""


@pytest.mark.unit
def test_page_content_valid_page(tmp_path: Path) -> None:
    """验证 get_page_content 返回指定页码的内容。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "page_mineru.json",
        sections=[
            {"title": "第零页", "level": 1, "content": "第零页内容", "page_idx": 0},
            {"title": "第一页", "level": 1, "content": "第一页内容", "page_idx": 1},
        ],
        tables=[
            {"caption": "表格1", "html": "<table><tr><td>A</td></tr></table>", "page_idx": 1},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    page_0 = processor.get_page_content(0)
    assert page_0["page_no"] == 0
    assert page_0["has_content"] is True
    assert len(page_0["sections"]) == 1
    assert page_0["sections"][0]["ref"] == "s_0001"
    assert len(page_0["tables"]) == 0

    page_1 = processor.get_page_content(1)
    assert page_1["page_no"] == 1
    assert len(page_1["sections"]) == 1
    assert page_1["sections"][0]["ref"] == "s_0002"
    assert len(page_1["tables"]) == 1
    assert page_1["tables"][0]["table_ref"] == "t_0001"


# ============================================================================
# 异常 / 边界
# ============================================================================


@pytest.mark.unit
def test_init_file_not_found(tmp_path: Path) -> None:
    """验证文件不存在时 __init__ 抛出 ValueError。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    missing_path = tmp_path / "nonexistent_mineru.json"
    # 使用手动构造的 Source 避免 stat() 调用
    source = LocalFileSource(
        path=missing_path,
        uri="local://nonexistent_mineru.json",
        media_type="application/json",
        content_length=0,
        etag=None,
    )

    with pytest.raises(ValueError, match="MinerU JSON 文件不存在"):
        MineruProcessor(source)


@pytest.mark.unit
def test_init_invalid_json(tmp_path: Path) -> None:
    """验证非法 JSON 时 __init__ 抛出 RuntimeError。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "bad_mineru.json"
    json_path.write_text("不是有效的 JSON {{{{", encoding="utf-8")
    source = _make_source(json_path)

    with pytest.raises(RuntimeError, match="MinerU JSON 解析失败"):
        MineruProcessor(source)


@pytest.mark.unit
def test_init_json_not_dict(tmp_path: Path) -> None:
    """验证 JSON 顶层非字典时 __init__ 抛出 ValueError。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = tmp_path / "array_mineru.json"
    json_path.write_text("[1, 2, 3]", encoding="utf-8")
    source = _make_source(json_path)

    with pytest.raises(ValueError, match="MinerU JSON 必须是顶层字典"):
        MineruProcessor(source)


@pytest.mark.unit
def test_read_section_not_found(tmp_path: Path) -> None:
    """验证 read_section 访问不存在的 ref 时抛出 KeyError。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "no_sec_mineru.json",
        sections=[
            {"title": "存在", "level": 1, "content": "内容", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    with pytest.raises(KeyError, match="Section not found"):
        processor.read_section("s_9999")


@pytest.mark.unit
def test_read_table_not_found(tmp_path: Path) -> None:
    """验证 read_table 访问不存在的 ref 时抛出 KeyError。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "no_tbl_mineru.json",
        tables=[
            {"caption": "表", "html": "<table><tr><td>A</td></tr></table>", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    with pytest.raises(KeyError, match="Table not found"):
        processor.read_table("t_9999")


@pytest.mark.unit
def test_page_content_negative_page(tmp_path: Path) -> None:
    """验证 page_no 为负数时抛出 ValueError。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "neg_mineru.json",
        sections=[
            {"title": "章节", "level": 1, "content": "内容", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    with pytest.raises(ValueError, match="page_no must be a non-negative integer"):
        processor.get_page_content(-1)


@pytest.mark.unit
def test_page_content_out_of_range(tmp_path: Path) -> None:
    """验证 page_no 超出范围时返回无内容的结果。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "oor_mineru.json",
        sections=[
            {"title": "唯一页", "level": 1, "content": "唯一内容", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    result = processor.get_page_content(99)
    assert result["page_no"] == 99
    assert result["has_content"] is False
    assert result["total_items"] == 0
    assert result["sections"] == []
    assert result["tables"] == []


@pytest.mark.unit
def test_empty_document(tmp_path: Path) -> None:
    """验证无 sections 和 tables 的空文档能正常初始化并返回空列表。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(tmp_path, "empty_mineru.json")
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    assert processor.list_sections() == []
    assert processor.list_tables() == []
    assert processor.get_full_text() == ""
    assert processor.get_full_text_with_table_markers() == ""

    page = processor.get_page_content(0)
    assert page["has_content"] is False


@pytest.mark.unit
def test_page_content_no_sections_only_raw_markdown(tmp_path: Path) -> None:
    """验证仅有 raw_markdown 无 sections 时能正常初始化。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "rawonly_mineru.json",
        raw_markdown="# 纯 Markdown 文档\n\n无需结构化章节。",
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    assert processor.list_sections() == []
    assert processor.get_full_text() == "# 纯 Markdown 文档\n\n无需结构化章节。"


@pytest.mark.unit
def test_search_within_ref_finds_in_section(tmp_path: Path) -> None:
    """验证 search 在指定章节范围内搜索。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "within_mineru.json",
        sections=[
            {"title": "第一章", "level": 1, "content": "第一章有一些关键词。", "page_idx": 0},
            {"title": "第二章", "level": 1, "content": "第二章没有。", "page_idx": 1},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    hits = processor.search("关键词", within_ref="s_0001")
    assert len(hits) >= 1
    assert any("关键词" in hit["snippet"] for hit in hits)


@pytest.mark.unit
def test_search_within_ref_not_found_returns_empty(tmp_path: Path) -> None:
    """验证 search 在无效 within_ref 时返回空列表。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "invalid_within_mineru.json",
        sections=[
            {"title": "报告", "level": 1, "content": "关于关键词的内容", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    hits = processor.search("关键词", within_ref="s_9999")
    assert hits == []


@pytest.mark.unit
def test_section_content_lru_eviction(tmp_path: Path) -> None:
    """验证 _section_content_cache 超过 256 条目时驱逐最早条目。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    # 生成 300 个章节
    sections = []
    for i in range(300):
        sections.append(
            {
                "title": f"章节{i + 1}",
                "level": 1,
                "content": f"第{i + 1}章内容。",
                "page_idx": i,
            }
        )

    json_path = _write_mineru_json(
        tmp_path,
        "lru_mineru.json",
        sections=sections,
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    # 按序读取所有章节（缓存满 256 后开始驱逐）
    for i in range(300):
        ref = f"s_{i + 1:04d}"
        processor.read_section(ref)

    # 验证缓存不超过 256
    assert len(processor._section_content_cache) <= 256

    # 最先被驱逐的应该是 s_0001（LRU 尾部）
    cache_keys = list(processor._section_content_cache.keys())
    assert "s_0001" not in cache_keys
    assert "s_0002" not in cache_keys

    # 最后读取的 s_0300 应该还在缓存中
    assert "s_0300" in cache_keys or len(cache_keys) == 256


@pytest.mark.unit
def test_get_section_title_valid(tmp_path: Path) -> None:
    """验证 get_section_title 返回正确标题。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "title_mineru.json",
        sections=[
            {"title": "重要章节", "level": 2, "content": "内容", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    title = processor.get_section_title("s_0001")
    assert title == "重要章节"


@pytest.mark.unit
def test_get_section_title_not_found(tmp_path: Path) -> None:
    """验证 get_section_title 对不存在的 ref 返回 None。

    Args:
        tmp_path: pytest 临时目录。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    json_path = _write_mineru_json(
        tmp_path,
        "no_title_mineru.json",
        sections=[
            {"title": "章节", "level": 1, "content": "内容", "page_idx": 0},
        ],
    )
    source = _make_source(json_path)
    processor = MineruProcessor(source)

    title = processor.get_section_title("s_9999")
    assert title is None
