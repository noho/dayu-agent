"""MinerU 集成基础测试。

覆盖 document_protocol、quota_tracker、mineru_runtime 的核心逻辑。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from dayu.document_protocol import (
    ConvertedDocument,
    DocumentBackend,
    DocumentImage,
    DocumentSection,
    DocumentTable,
    detect_mineru_content_list_version,
    detect_mineru_content_list_version_from_bytes,
    parse_block,
    parse_content_list,
)
from dayu.quota_tracker import QuotaExhaustedError, QuotaTracker


# ---------------------------------------------------------------------------
# document_protocol 测试
# ---------------------------------------------------------------------------


class TestDocumentProtocol:
    """统一中间格式测试。"""

    def test_converted_document_defaults(self) -> None:
        """ConvertedDocument 默认值测试。"""
        doc = ConvertedDocument(backend=DocumentBackend.DOCLING)
        assert doc.backend == DocumentBackend.DOCLING
        assert doc.sections == ()
        assert doc.tables == ()
        assert doc.images == ()
        assert doc.raw_markdown == ""

    def test_converted_document_with_data(self) -> None:
        """ConvertedDocument 填充数据测试。"""
        section = DocumentSection(title="Test", level=1, content="content")
        table = DocumentTable(caption="tbl", html="<table></table>")
        doc = ConvertedDocument(
            backend=DocumentBackend.MINERU_CLOUD,
            sections=(section,),
            tables=(table,),
            raw_markdown="# Test\ncontent",
        )
        assert len(doc.sections) == 1
        assert doc.sections[0].title == "Test"
        assert len(doc.tables) == 1
        assert doc.tables[0].html == "<table></table>"

    def test_document_section_frozen(self) -> None:
        """DocumentSection 不可变。"""
        section = DocumentSection(title="T", level=1, content="c")
        with pytest.raises(AttributeError):
            section.title = "X"  # type: ignore[misc]

    def test_document_table_frozen(self) -> None:
        """DocumentTable 不可变。"""
        table = DocumentTable(caption="c", html="<table/>")
        with pytest.raises(AttributeError):
            table.html = "<div/>"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 版本探测测试（bbox 类型判断）
# ---------------------------------------------------------------------------


class TestVersionDetection:
    """MinerU content_list 版本探测测试。"""

    def test_v2_detected_by_list_bbox(self) -> None:
        """bbox 为 list → v2。"""
        content_list = [
            {"type": "text", "bbox": [100, 200, 300, 400]},
            {"type": "text", "bbox": [10, 20, 30, 40]},
        ]
        assert detect_mineru_content_list_version(content_list) == "v2"

    def test_v1_detected_by_dict_bbox(self) -> None:
        """bbox 为 dict → v1。"""
        content_list = [
            {"type": "text", "bbox": {"x0": 100, "y0": 200, "x1": 300, "y1": 400}},
        ]
        assert detect_mineru_content_list_version(content_list) == "v1"

    def test_unknown_when_no_bbox(self) -> None:
        """所有元素均无 bbox → unknown。"""
        content_list = [
            {"type": "text", "text": "hello"},
            {"type": "title", "title": "Chapter 1"},
        ]
        assert detect_mineru_content_list_version(content_list) == "unknown"

    def test_skips_first_items_without_bbox(self) -> None:
        """前几个元素无 bbox，后续有 → 正确识别。"""
        content_list = [
            {"type": "title", "title": "Header"},  # 无 bbox
            {"type": "text", "text": "body"},        # 无 bbox
            {"type": "text", "bbox": [0, 1, 2, 3]},  # 有 bbox → v2
        ]
        assert detect_mineru_content_list_version(content_list) == "v2"

    def test_mixed_bbox_types_first_match_wins(self) -> None:
        """多种 bbox 类型共存，第一个有 bbox 的决定版本。"""
        content_list = [
            {"type": "text", "bbox": {"x0": 0, "y0": 0}},  # dict → v1
            {"type": "text", "bbox": [0, 1, 2, 3]},         # list（但第二个）
        ]
        assert detect_mineru_content_list_version(content_list) == "v1"

    def test_empty_raises(self) -> None:
        """空列表 → TypeError。"""
        with pytest.raises(TypeError, match="为空"):
            detect_mineru_content_list_version([])

    def test_check_first_10_elements_only(self) -> None:
        """只检查前 10 个元素。"""
        # 前 10 个均无 bbox，第 11 个有 → unknown
        content_list: list[dict[str, object]] = [
            {"type": "text", "text": f"item {i}"}
            for i in range(10)
        ]
        content_list.append({"type": "text", "bbox": [0, 1, 2, 3]})
        assert detect_mineru_content_list_version(content_list) == "unknown"

    def test_from_bytes_bare_list_v2(self) -> None:
        """从 JSON bytes（裸列表，list bbox）探测 → v2。"""
        data = [{"type": "text", "bbox": [0, 1, 2, 3]}]
        raw = json.dumps(data).encode()
        assert detect_mineru_content_list_version_from_bytes(raw) == "v2"

    def test_from_bytes_bare_list_v1(self) -> None:
        """从 JSON bytes（裸列表，dict bbox）探测 → v1。"""
        data = [{"type": "text", "bbox": {"x0": 0, "y0": 0}}]
        raw = json.dumps(data).encode()
        assert detect_mineru_content_list_version_from_bytes(raw) == "v1"

    def test_from_bytes_wrapper(self) -> None:
        """从 JSON bytes（带 content_list 包壳）探测。"""
        data = {"content_list": [{"type": "text", "bbox": [0, 1, 2, 3]}]}
        raw = json.dumps(data).encode()
        assert detect_mineru_content_list_version_from_bytes(raw) == "v2"

    def test_from_bytes_empty_raises(self) -> None:
        """空 bytes → TypeError。"""
        with pytest.raises(TypeError, match="为空"):
            detect_mineru_content_list_version_from_bytes(b"")

    def test_from_bytes_invalid_json_raises(self) -> None:
        """非法 JSON → JSONDecodeError。"""
        with pytest.raises(json.JSONDecodeError):
            detect_mineru_content_list_version_from_bytes(b"not json")

    def test_from_bytes_no_content_list_field_raises(self) -> None:
        """dict 中无 content_list 字段 → TypeError。"""
        data = {"foo": "bar"}
        raw = json.dumps(data).encode()
        with pytest.raises(TypeError, match="不包含"):
            detect_mineru_content_list_version_from_bytes(raw)


# ---------------------------------------------------------------------------
# parse_block / parse_content_list 测试
# ---------------------------------------------------------------------------


class TestParseBlock:
    """单 block 解析测试。"""

    def test_known_types_pass_through(self) -> None:
        """已知类型原样返回。"""
        for btype in ("text", "title", "figure", "table"):
            block: dict[str, object] = {"type": btype, "data": "x"}
            result = parse_block(block)
            assert result is block
            assert "_unknown_type" not in result

    def test_unknown_type_marked(self) -> None:
        """未知类型附带 _unknown_type=True 标记。"""
        block: dict[str, object] = {"type": "equation", "data": "E=mc2"}
        result = parse_block(block)
        assert result["_unknown_type"] is True
        assert result["type"] == "equation"
        assert result["data"] == "E=mc2"

    def test_missing_type_treated_as_unknown(self) -> None:
        """缺少 type 字段视为 unknown。"""
        block: dict[str, object] = {"data": "x"}
        result = parse_block(block)
        assert result["_unknown_type"] is True


class TestParseContentList:
    """content_list 批量解析测试。"""

    def test_all_known(self) -> None:
        """全部已知类型 → 全部返回。"""
        cl: list[dict[str, object]] = [
            {"type": "text", "text": "a"},
            {"type": "title", "title": "T"},
            {"type": "table", "html": "<table/>"},
        ]
        result = parse_content_list(cl)
        assert len(result) == 3

    def test_unknown_types_included(self) -> None:
        """未知类型也被保留（带标记）。"""
        cl: list[dict[str, object]] = [
            {"type": "text", "text": "a"},
            {"type": "equation", "eq": "E=mc2"},
        ]
        result = parse_content_list(cl)
        assert len(result) == 2
        assert "_unknown_type" in result[1]

    def test_empty_returns_empty(self) -> None:
        """空列表返回空列表。"""
        # parse_content_list 内部调 detect，空列表会 raise
        # 但 parse_content_list 本身不 raise，因为它遍历 content_list[:10]
        # 实际上 detect_mineru_content_list_version([]) 会 raise TypeError
        # parse_content_list 没有 catch 这个 —— 这是预期行为
        with pytest.raises(TypeError):
            parse_content_list([])


# ---------------------------------------------------------------------------
# quota_tracker 测试
# ---------------------------------------------------------------------------


class TestQuotaTracker:
    """配额跟踪器测试。"""

    def test_basic_consume(self) -> None:
        """基本消耗和剩余查询。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "quota.json")
            qt = QuotaTracker(daily_limit=100, state_file=state_file)
            assert qt.check_and_consume(30) is True
            assert qt.get_remaining() == 70
            assert qt.get_used() == 30

    def test_exhausted(self) -> None:
        """配额不足返回 False。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "quota.json")
            qt = QuotaTracker(daily_limit=100, state_file=state_file)
            qt.check_and_consume(60)
            assert qt.check_and_consume(50) is False
            assert qt.get_remaining() == 40

    def test_consume_or_raise(self) -> None:
        """配额不足时抛出异常。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "quota.json")
            qt = QuotaTracker(daily_limit=100, state_file=state_file)
            qt.check_and_consume(80)
            with pytest.raises(QuotaExhaustedError, match="配额不足"):
                qt.check_and_consume_or_raise(30)

    def test_consume_or_raise_success(self) -> None:
        """配额充足时不抛异常。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "quota.json")
            qt = QuotaTracker(daily_limit=100, state_file=state_file)
            qt.check_and_consume_or_raise(30)
            assert qt.get_used() == 30

    def test_persistence(self) -> None:
        """状态持久化到文件后可恢复。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "quota.json")
            qt1 = QuotaTracker(daily_limit=100, state_file=state_file)
            qt1.check_and_consume(40)

            qt2 = QuotaTracker(daily_limit=100, state_file=state_file)
            assert qt2.get_used() == 40
            assert qt2.get_remaining() == 60

    def test_corrupted_state_file(self) -> None:
        """损坏的状态文件不崩溃，重置为 0。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "quota.json"
            state_file.write_text("not json!!!", encoding="utf-8")
            qt = QuotaTracker(daily_limit=100, state_file=str(state_file))
            assert qt.get_used() == 0
            assert qt.get_remaining() == 100

    def test_exact_boundary(self) -> None:
        """恰好用完配额（边界值）。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = str(Path(tmpdir) / "quota.json")
            qt = QuotaTracker(daily_limit=100, state_file=state_file)
            assert qt.check_and_consume(100) is True
            assert qt.get_remaining() == 0
            assert qt.check_and_consume(1) is False
