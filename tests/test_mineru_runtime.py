"""MinerU 运行时核心逻辑测试。

覆盖 page_ranges 计算、zip 解析、结果转换、合并、回退链等逻辑。
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from dayu.document_protocol import (
    ConvertedDocument,
    DocumentBackend,
    DocumentSection,
    DocumentTable,
    DocumentImage,
)
from dayu.mineru_runtime import (
    MinerUAPIError,
    MinerUTaskFailedError,
    MinerUTimeoutError,
    _build_page_ranges,
    _convert_mineru_result,
    _download_and_parse_zip,
    _estimate_pages,
    _merge_chunk_results,
    parse_pdf_bytes_with_mineru,
)


# ---------------------------------------------------------------------------
# 辅助工厂
# ---------------------------------------------------------------------------


def _make_zip_bytes(
    markdown: str = "# Test\nContent",
    content_list: list[list[dict[str, object]]] | None = None,
) -> bytes:
    """构造测试用 zip 字节流，模拟 MinerU 结果包。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("full.md", markdown)
        if content_list is not None:
            zf.writestr(
                "xxx_content_list_v2.json",
                json.dumps(content_list, ensure_ascii=False),
            )
    return buf.getvalue()


def _make_mineru_result(
    zip_url: str = "https://example.com/result.zip",
) -> dict[str, object]:
    """构造模拟的 MinerU API 查询结果。"""
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "task_id": "test-task-001",
            "state": "done",
            "full_zip_url": zip_url,
        },
    }


# ---------------------------------------------------------------------------
# _build_page_ranges 测试
# ---------------------------------------------------------------------------


class TestBuildPageRanges:
    """page_ranges 服务端分页计算测试。"""

    def test_single_batch(self) -> None:
        """≤200 页单批次。"""
        assert _build_page_ranges(100) == ["1-100"]
        assert _build_page_ranges(200) == ["1-200"]
        assert _build_page_ranges(1) == ["1-1"]

    def test_two_batches(self) -> None:
        """201-400 页双批次。"""
        assert _build_page_ranges(201) == ["1-200", "201-201"]
        assert _build_page_ranges(374) == ["1-200", "201-374"]
        assert _build_page_ranges(400) == ["1-200", "201-400"]

    def test_three_batches(self) -> None:
        """401-600 页三批次。"""
        assert _build_page_ranges(500) == ["1-200", "201-400", "401-500"]
        assert _build_page_ranges(600) == ["1-200", "201-400", "401-600"]

    def test_custom_batch_size(self) -> None:
        """自定义批次大小。"""
        assert _build_page_ranges(100, max_per_batch=50) == ["1-50", "51-100"]
        assert _build_page_ranges(150, max_per_batch=100) == ["1-100", "101-150"]

    def test_boundary_exact(self) -> None:
        """恰好整除边界。"""
        assert _build_page_ranges(400, max_per_batch=200) == [
            "1-200", "201-400"
        ]
        assert _build_page_ranges(600, max_per_batch=200) == [
            "1-200", "201-400", "401-600"
        ]


# ---------------------------------------------------------------------------
# _estimate_pages 测试
# ---------------------------------------------------------------------------


class TestEstimatePages:
    """页数估算测试。"""

    def test_small_file(self) -> None:
        """小文件至少 1 页。"""
        assert _estimate_pages(b"x" * 1000) == 1

    def test_typical_page(self) -> None:
        """~50KB 大约 1 页。"""
        assert _estimate_pages(b"x" * 50000) == 1

    def test_large_file(self) -> None:
        """~500KB 大约 10 页。"""
        assert _estimate_pages(b"x" * 500000) == 10


# ---------------------------------------------------------------------------
# _download_and_parse_zip 测试（mock httpx.get）
# ---------------------------------------------------------------------------


class TestDownloadAndParseZip:
    """zip 结果下载解析测试。"""

    def test_normal_zip(self) -> None:
        """正常 zip 包解析：markdown + content_list。"""
        cl = [
            [  # page 0
                {"type": "title", "content": "Chapter 1"},
                {"type": "text", "content": "Body text"},
            ],
            [  # page 1
                {"type": "table", "content": "<table/>"},
            ],
        ]
        zip_bytes = _make_zip_bytes(markdown="# Chapter 1\nBody text", content_list=cl)

        mock_resp = MagicMock()
        mock_resp.content = zip_bytes
        mock_resp.raise_for_status = MagicMock()

        with patch("dayu.mineru_runtime.httpx.get", return_value=mock_resp):
            md, blocks = _download_and_parse_zip("https://example.com/result.zip")

        assert md == "# Chapter 1\nBody text"
        assert len(blocks) == 3
        # page_idx 应该被附加
        assert blocks[0]["page_idx"] == 0
        assert blocks[1]["page_idx"] == 0
        assert blocks[2]["page_idx"] == 1

    def test_zip_without_content_list(self) -> None:
        """zip 中无 content_list → blocks 为空。"""
        zip_bytes = _make_zip_bytes(markdown="# Only md", content_list=None)

        mock_resp = MagicMock()
        mock_resp.content = zip_bytes
        mock_resp.raise_for_status = MagicMock()

        with patch("dayu.mineru_runtime.httpx.get", return_value=mock_resp):
            md, blocks = _download_and_parse_zip("https://example.com/result.zip")

        assert md == "# Only md"
        assert blocks == []

    def test_zip_without_markdown(self) -> None:
        """zip 中无 .md 文件 → markdown 为空。"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("data.json", '{"key": "value"}')
        zip_bytes = buf.getvalue()

        mock_resp = MagicMock()
        mock_resp.content = zip_bytes
        mock_resp.raise_for_status = MagicMock()

        with patch("dayu.mineru_runtime.httpx.get", return_value=mock_resp):
            md, blocks = _download_and_parse_zip("https://example.com/result.zip")

        assert md == ""
        assert blocks == []

    def test_bad_zip_raises(self) -> None:
        """损坏的 zip → MinerUAPIError。"""
        mock_resp = MagicMock()
        mock_resp.content = b"not a zip file"
        mock_resp.raise_for_status = MagicMock()

        with patch("dayu.mineru_runtime.httpx.get", return_value=mock_resp):
            with pytest.raises(MinerUAPIError, match="损坏"):
                _download_and_parse_zip("https://example.com/result.zip")

    def test_download_failure_raises(self) -> None:
        """下载失败 → MinerUAPIError。"""
        import httpx as _httpx

        with patch(
            "dayu.mineru_runtime.httpx.get",
            side_effect=_httpx.ConnectError("connection refused"),
        ):
            with pytest.raises(MinerUAPIError, match="下载失败"):
                _download_and_parse_zip("https://example.com/result.zip")

    def test_content_list_flattening(self) -> None:
        """content_list 多页展平 + page_idx 正确。"""
        cl = [
            [{"type": "title", "content": "P0"}],  # page 0: 1 block
            [],                                       # page 1: empty
            [{"type": "text", "content": "P2"},       # page 2: 2 blocks
             {"type": "table", "content": "<t/>"}],
        ]
        zip_bytes = _make_zip_bytes(content_list=cl)
        mock_resp = MagicMock()
        mock_resp.content = zip_bytes
        mock_resp.raise_for_status = MagicMock()

        with patch("dayu.mineru_runtime.httpx.get", return_value=mock_resp):
            _, blocks = _download_and_parse_zip("https://example.com/result.zip")

        assert len(blocks) == 3
        assert blocks[0]["page_idx"] == 0
        assert blocks[1]["page_idx"] == 2
        assert blocks[2]["page_idx"] == 2

    def test_dict_item_in_content_list(self) -> None:
        """content_list 中出现 dict（非 list）→ 也正确处理。"""
        cl: list[object] = [
            [{"type": "title", "content": "A"}],
            {"type": "text", "content": "B"},  # dict instead of list
        ]
        zip_bytes = _make_zip_bytes(content_list=cl)
        mock_resp = MagicMock()
        mock_resp.content = zip_bytes
        mock_resp.raise_for_status = MagicMock()

        with patch("dayu.mineru_runtime.httpx.get", return_value=mock_resp):
            _, blocks = _download_and_parse_zip("https://example.com/result.zip")

        assert len(blocks) == 2
        assert blocks[0]["page_idx"] == 0
        assert blocks[1]["page_idx"] == 1


# ---------------------------------------------------------------------------
# _convert_mineru_result 测试
# ---------------------------------------------------------------------------


class TestConvertMineruResult:
    """MinerU 结果转 ConvertedDocument 测试。"""

    def test_basic_conversion(self) -> None:
        """正常结果 → ConvertedDocument。"""
        cl = [
            [{"type": "title", "content": "Chapter 1"}],
            [{"type": "text", "content": "Body"},
             {"type": "table", "html": "<table/>"},
             {"type": "image", "image_path": "/img/1.jpg", "caption": "fig1"}],
        ]
        zip_bytes = _make_zip_bytes(markdown="# Chapter 1\nBody", content_list=cl)
        mock_resp = MagicMock()
        mock_resp.content = zip_bytes
        mock_resp.raise_for_status = MagicMock()

        result = _make_mineru_result()
        with patch("dayu.mineru_runtime.httpx.get", return_value=mock_resp):
            doc = _convert_mineru_result(result, DocumentBackend.MINERU_CLOUD)

        assert doc.backend == DocumentBackend.MINERU_CLOUD
        assert doc.raw_markdown == "# Chapter 1\nBody"
        assert len(doc.sections) == 2  # title + text
        assert len(doc.tables) == 1
        assert len(doc.images) == 1

    def test_missing_zip_url(self) -> None:
        """无 full_zip_url → 空内容。"""
        result = {"code": 0, "data": {"task_id": "x", "state": "done"}}
        doc = _convert_mineru_result(result, DocumentBackend.MINERU_CLOUD)
        assert doc.raw_markdown == ""
        assert len(doc.sections) == 0

    def test_zip_download_failure_graceful(self) -> None:
        """zip 下载失败 → 空内容（不抛异常）。"""
        result = _make_mineru_result()
        with patch(
            "dayu.mineru_runtime.httpx.get",
            side_effect=Exception("network error"),
        ):
            doc = _convert_mineru_result(result, DocumentBackend.MINERU_CLOUD)
        assert doc.raw_markdown == ""

    def test_unknown_block_type_skipped(self) -> None:
        """未知 block type → 跳过，不崩溃。"""
        cl = [
            [{"type": "equation", "content": "E=mc2"},
             {"type": "text", "content": "normal"}],
        ]
        zip_bytes = _make_zip_bytes(content_list=cl)
        mock_resp = MagicMock()
        mock_resp.content = zip_bytes
        mock_resp.raise_for_status = MagicMock()

        result = _make_mineru_result()
        with patch("dayu.mineru_runtime.httpx.get", return_value=mock_resp):
            doc = _convert_mineru_result(result, DocumentBackend.MINERU_CLOUD)

        # equation 被跳过，只有 text 被保留
        assert len(doc.sections) == 1
        assert len(doc.tables) == 0


# ---------------------------------------------------------------------------
# _merge_chunk_results 测试
# ---------------------------------------------------------------------------


class TestMergeChunkResults:
    """多批次结果合并测试。"""

    def test_merge_two_results(self) -> None:
        """两个结果合并。"""
        cl1 = [[{"type": "title", "content": "Part 1"}]]
        cl2 = [[{"type": "title", "content": "Part 2"}]]
        zip1 = _make_zip_bytes(markdown="# Part 1", content_list=cl1)
        zip2 = _make_zip_bytes(markdown="# Part 2", content_list=cl2)

        mock1 = MagicMock()
        mock1.content = zip1
        mock1.raise_for_status = MagicMock()
        mock2 = MagicMock()
        mock2.content = zip2
        mock2.raise_for_status = MagicMock()

        results = [_make_mineru_result("https://a.com/z1.zip"),
                    _make_mineru_result("https://a.com/z2.zip")]

        with patch("dayu.mineru_runtime.httpx.get", side_effect=[mock1, mock2]):
            doc = _merge_chunk_results(results, DocumentBackend.MINERU_CLOUD)

        assert doc.backend == DocumentBackend.MINERU_CLOUD
        assert len(doc.sections) == 2
        assert "# Part 1" in doc.raw_markdown
        assert "# Part 2" in doc.raw_markdown
        assert doc.metadata["chunk_count"] == "2"

    def test_merge_empty(self) -> None:
        """空结果列表 → 空文档。"""
        doc = _merge_chunk_results([], DocumentBackend.MINERU_CLOUD)
        assert doc.raw_markdown == ""
        assert len(doc.sections) == 0
        assert doc.metadata["chunk_count"] == "0"

    def test_merge_preserves_order(self) -> None:
        """合并保持输入顺序。"""
        cl_a = [[{"type": "text", "content": "A"}]]
        cl_b = [[{"type": "text", "content": "B"}]]
        zip_a = _make_zip_bytes(markdown="A", content_list=cl_a)
        zip_b = _make_zip_bytes(markdown="B", content_list=cl_b)

        mock_a = MagicMock(content=zip_a, raise_for_status=MagicMock())
        mock_b = MagicMock(content=zip_b, raise_for_status=MagicMock())

        results = [_make_mineru_result("https://a.com/z.zip"),
                    _make_mineru_result("https://b.com/z.zip")]

        with patch("dayu.mineru_runtime.httpx.get", side_effect=[mock_a, mock_b]):
            doc = _merge_chunk_results(results, DocumentBackend.MINERU_CLOUD)

        assert doc.sections[0].content == "A"
        assert doc.sections[1].content == "B"


# ---------------------------------------------------------------------------
# parse_pdf_bytes_with_mineru 回退链测试
# ---------------------------------------------------------------------------


class TestParsePdfBytesFallback:
    """五层回退链集成测试。"""

    def test_no_token_falls_to_docling(self) -> None:
        """无 Token → 跳过云 API → 回退 Docling。"""
        mock_docling = ConvertedDocument(
            backend=DocumentBackend.DOCLING,
            raw_markdown="docling result",
        )
        with (
            patch("dayu.mineru_runtime._MINERU_API_TOKEN", ""),
            patch("dayu.mineru_runtime._parse_with_docling", return_value=mock_docling),
            patch("dayu.mineru_runtime._try_parse_with_mineru_cli", return_value=None),
        ):
            doc = parse_pdf_bytes_with_mineru(b"fake pdf", total_pages=10)
        assert doc.backend == DocumentBackend.DOCLING
        assert doc.raw_markdown == "docling result"

    def test_quota_exhausted_falls_to_docling(self) -> None:
        """配额不足 → 跳过云 API → 回退 Docling。"""
        mock_docling = ConvertedDocument(
            backend=DocumentBackend.DOCLING,
            raw_markdown="fallback",
        )
        tracker = MagicMock()
        tracker.check_and_consume.return_value = False
        with (
            patch("dayu.mineru_runtime._MINERU_API_TOKEN", "tk"),
            patch("dayu.mineru_runtime._get_quota_tracker", return_value=tracker),
            patch("dayu.mineru_runtime._parse_with_docling", return_value=mock_docling),
            patch("dayu.mineru_runtime._try_parse_with_mineru_cli", return_value=None),
        ):
            doc = parse_pdf_bytes_with_mineru(b"fake pdf", total_pages=10)
        assert doc.backend == DocumentBackend.DOCLING

    def test_cloud_api_failure_falls_to_docling(self) -> None:
        """云 API 失败 → 回退 Docling。"""
        mock_docling = ConvertedDocument(
            backend=DocumentBackend.DOCLING,
            raw_markdown="fallback",
        )
        tracker = MagicMock()
        tracker.check_and_consume.return_value = True
        with (
            patch("dayu.mineru_runtime._MINERU_API_TOKEN", "tk"),
            patch("dayu.mineru_runtime._get_quota_tracker", return_value=tracker),
            patch("dayu.mineru_runtime._parse_with_cloud_api", side_effect=MinerUAPIError("fail")),
            patch("dayu.mineru_runtime._parse_with_docling", return_value=mock_docling),
            patch("dayu.mineru_runtime._try_parse_with_mineru_cli", return_value=None),
        ):
            doc = parse_pdf_bytes_with_mineru(b"fake pdf", total_pages=10)
        assert doc.backend == DocumentBackend.DOCLING

    def test_cloud_api_success_no_fallback(self) -> None:
        """云 API 成功 → 不走回退。"""
        mock_cloud = ConvertedDocument(
            backend=DocumentBackend.MINERU_CLOUD,
            raw_markdown="cloud result",
        )
        tracker = MagicMock()
        tracker.check_and_consume.return_value = True
        with (
            patch("dayu.mineru_runtime._MINERU_API_TOKEN", "tk"),
            patch("dayu.mineru_runtime._get_quota_tracker", return_value=tracker),
            patch("dayu.mineru_runtime._parse_with_cloud_api", return_value=mock_cloud),
        ):
            doc = parse_pdf_bytes_with_mineru(b"fake pdf", total_pages=10)
        assert doc.backend == DocumentBackend.MINERU_CLOUD
        assert doc.raw_markdown == "cloud result"


# ---------------------------------------------------------------------------
# 层3 CLI 解析测试
# ---------------------------------------------------------------------------


class TestLayer3CLI:
    """层3 CLI 解析测试。"""

    def test_cli_not_found_returns_none(self) -> None:
        """mock shutil.which 返回 None → return None。"""
        from dayu.mineru_runtime import _try_parse_with_mineru_cli

        with patch("dayu.mineru_runtime.shutil.which", return_value=None):
            result = _try_parse_with_mineru_cli(b"fake pdf")
        assert result is None

    def test_cli_subprocess_error_returns_none(self) -> None:
        """mock subprocess.run 抛 CalledProcessError → return None。"""
        from dayu.mineru_runtime import _try_parse_with_mineru_cli

        with (
            patch(
                "dayu.mineru_runtime.shutil.which", return_value="/usr/bin/magic-pdf"
            ),
            patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["magic-pdf"]),
            ),
            patch("tempfile.NamedTemporaryFile") as mock_tmp,
            patch("tempfile.mkdtemp", return_value="/tmp/mineru_cli_test"),
        ):
            mock_f = MagicMock()
            mock_f.name = "/tmp/test.pdf"
            mock_tmp.return_value.__enter__.return_value = mock_f

            result = _try_parse_with_mineru_cli(b"fake pdf")
        assert result is None

    def test_cli_timeout_returns_none(self) -> None:
        """mock subprocess.run 抛 TimeoutExpired → return None。"""
        from dayu.mineru_runtime import _try_parse_with_mineru_cli

        with (
            patch(
                "dayu.mineru_runtime.shutil.which", return_value="/usr/bin/magic-pdf"
            ),
            patch(
                "subprocess.run",
                side_effect=subprocess.TimeoutExpired(
                    cmd=["magic-pdf"], timeout=300
                ),
            ),
            patch("tempfile.NamedTemporaryFile") as mock_tmp,
            patch("tempfile.mkdtemp", return_value="/tmp/mineru_cli_test"),
        ):
            mock_f = MagicMock()
            mock_f.name = "/tmp/test.pdf"
            mock_tmp.return_value.__enter__.return_value = mock_f

            result = _try_parse_with_mineru_cli(b"fake pdf")
        assert result is None

    def test_cli_parse_success(self) -> None:
        """mock subprocess + mock 输出文件 → ConvertedDocument(MINERU_LOCAL)。"""
        from dayu.mineru_runtime import _try_parse_with_mineru_cli

        mock_md_path = MagicMock()
        mock_md_path.exists.return_value = True
        mock_md_path.read_text.return_value = "# Valid markdown"

        mock_cl_path = MagicMock()
        mock_cl_path.exists.return_value = True
        mock_cl_path.read_text.return_value = "[]"

        with (
            patch(
                "dayu.mineru_runtime.shutil.which", return_value="/usr/bin/magic-pdf"
            ),
            patch("subprocess.run"),
            patch("tempfile.NamedTemporaryFile") as mock_tmp,
            patch("tempfile.mkdtemp", return_value="/tmp/mineru_cli_test"),
        ):
            mock_f = MagicMock()
            mock_f.name = "/tmp/test.pdf"
            mock_tmp.return_value.__enter__.return_value = mock_f

            # Mock Path so that Path(output_dir)/pdf_filename/"auto" returns our mock dir
            mock_result_dir = MagicMock()
            mock_result_dir.__truediv__.side_effect = lambda x: {
                "input.pdf.md": mock_md_path,
                "input.pdf_content_list.json": mock_cl_path,
            }.get(x, MagicMock())

            mock_pdf_dir = MagicMock()
            mock_pdf_dir.__truediv__.return_value = mock_result_dir

            mock_root = MagicMock()
            mock_root.__truediv__.return_value = mock_pdf_dir

            with patch("pathlib.Path", return_value=mock_root):
                result = _try_parse_with_mineru_cli(b"fake pdf")

        assert result is not None
        assert result.backend == DocumentBackend.MINERU_LOCAL
        assert result.raw_markdown == "# Valid markdown"
        assert len(result.sections) == 0

    def test_cli_parse_with_content_list(self) -> None:
        """mock 返回 table/image blocks → sections/tables/images 正确。"""
        from dayu.mineru_runtime import _try_parse_with_mineru_cli

        content_list = [
            [
                {"type": "text", "content": "Some paragraph text"},
                {"type": "title", "content": "Section 1", "level": 2},
                {
                    "type": "table",
                    "caption": "Table 1",
                    "html": "<table><tr><td>data</td></tr></table>",
                },
                {
                    "type": "figure",
                    "image_path": "/img/test.png",
                    "caption": "Figure 1",
                },
            ]
        ]

        mock_md_path = MagicMock()
        mock_md_path.exists.return_value = True
        mock_md_path.read_text.return_value = "# Document with content list"

        mock_cl_path = MagicMock()
        mock_cl_path.exists.return_value = True
        mock_cl_path.read_text.return_value = json.dumps(content_list)

        with (
            patch(
                "dayu.mineru_runtime.shutil.which", return_value="/usr/bin/magic-pdf"
            ),
            patch("subprocess.run"),
            patch("tempfile.NamedTemporaryFile") as mock_tmp,
            patch("tempfile.mkdtemp", return_value="/tmp/mineru_cli_test"),
        ):
            mock_f = MagicMock()
            mock_f.name = "/tmp/test.pdf"
            mock_tmp.return_value.__enter__.return_value = mock_f

            mock_result_dir = MagicMock()
            mock_result_dir.__truediv__.side_effect = lambda x: {
                "input.pdf.md": mock_md_path,
                "input.pdf_content_list.json": mock_cl_path,
            }.get(x, MagicMock())

            mock_pdf_dir = MagicMock()
            mock_pdf_dir.__truediv__.return_value = mock_result_dir

            mock_root = MagicMock()
            mock_root.__truediv__.return_value = mock_pdf_dir

            with patch("pathlib.Path", return_value=mock_root):
                result = _try_parse_with_mineru_cli(b"fake pdf")

        assert result is not None
        assert result.backend == DocumentBackend.MINERU_LOCAL
        # 1 paragraph + 1 title
        assert len(result.sections) == 2
        assert result.sections[0].title == ""
        assert result.sections[0].content == "Some paragraph text"
        assert result.sections[1].title == "Section 1"
        assert result.sections[1].level == 2
        # 1 table
        assert len(result.tables) == 1
        assert result.tables[0].caption == "Table 1"
        # 1 figure
        assert len(result.images) == 1
        assert result.images[0].path == "/img/test.png"
        assert result.images[0].caption == "Figure 1"

    def test_cli_parse_no_content_list(self) -> None:
        """mock 只有 .md → raw_markdown 有值。"""
        from dayu.mineru_runtime import _try_parse_with_mineru_cli

        mock_md_path = MagicMock()
        mock_md_path.exists.return_value = True
        mock_md_path.read_text.return_value = "# Just markdown\nNo content list."

        mock_cl_path = MagicMock()
        mock_cl_path.exists.return_value = False

        with (
            patch(
                "dayu.mineru_runtime.shutil.which", return_value="/usr/bin/magic-pdf"
            ),
            patch("subprocess.run"),
            patch("tempfile.NamedTemporaryFile") as mock_tmp,
            patch("tempfile.mkdtemp", return_value="/tmp/mineru_cli_test"),
        ):
            mock_f = MagicMock()
            mock_f.name = "/tmp/test.pdf"
            mock_tmp.return_value.__enter__.return_value = mock_f

            mock_result_dir = MagicMock()
            mock_result_dir.__truediv__.side_effect = lambda x: {
                "input.pdf.md": mock_md_path,
                "input.pdf_content_list.json": mock_cl_path,
            }.get(x, MagicMock())

            mock_pdf_dir = MagicMock()
            mock_pdf_dir.__truediv__.return_value = mock_result_dir

            mock_root = MagicMock()
            mock_root.__truediv__.return_value = mock_pdf_dir

            with patch("pathlib.Path", return_value=mock_root):
                result = _try_parse_with_mineru_cli(b"fake pdf")

        assert result is not None
        assert result.backend == DocumentBackend.MINERU_LOCAL
        assert result.raw_markdown == "# Just markdown\nNo content list."
        assert len(result.sections) == 0
        assert len(result.tables) == 0
        assert len(result.images) == 0

    def test_cli_cleanup_temp_files(self) -> None:
        """验证 os.unlink 和 shutil.rmtree 被调用。"""
        from dayu.mineru_runtime import _try_parse_with_mineru_cli

        with (
            patch(
                "dayu.mineru_runtime.shutil.which", return_value="/usr/bin/magic-pdf"
            ),
            patch("subprocess.run"),
            patch("tempfile.NamedTemporaryFile") as mock_tmp,
            patch("tempfile.mkdtemp", return_value="/tmp/mineru_cli_test"),
            patch("os.unlink") as mock_unlink,
            patch("shutil.rmtree") as mock_rmtree,
            patch("os.path.exists", return_value=True),
            patch("os.path.isdir", return_value=True),
        ):
            mock_f = MagicMock()
            mock_f.name = "/tmp/test.pdf"
            mock_tmp.return_value.__enter__.return_value = mock_f

            mock_md_path = MagicMock()
            mock_md_path.exists.return_value = True
            mock_md_path.read_text.return_value = "# test"
            mock_cl_path = MagicMock()
            mock_cl_path.exists.return_value = True
            mock_cl_path.read_text.return_value = "[]"

            mock_result_dir = MagicMock()
            mock_result_dir.__truediv__.side_effect = lambda x: {
                "input.pdf.md": mock_md_path,
                "input.pdf_content_list.json": mock_cl_path,
            }.get(x, MagicMock())

            mock_pdf_dir = MagicMock()
            mock_pdf_dir.__truediv__.return_value = mock_result_dir

            mock_root = MagicMock()
            mock_root.__truediv__.return_value = mock_pdf_dir

            with patch("pathlib.Path", return_value=mock_root):
                result = _try_parse_with_mineru_cli(b"fake pdf")

        assert result is not None
        mock_unlink.assert_called_once_with("/tmp/test.pdf")
        mock_rmtree.assert_called_once_with("/tmp/mineru_cli_test", ignore_errors=True)

    def test_cli_exception_does_not_propagate(self) -> None:
        """任何异常 → return None。"""
        from dayu.mineru_runtime import _try_parse_with_mineru_cli

        with (
            patch(
                "dayu.mineru_runtime.shutil.which", return_value="/usr/bin/magic-pdf"
            ),
            patch(
                "subprocess.run",
                side_effect=RuntimeError("unexpected error"),
            ),
            patch("tempfile.NamedTemporaryFile") as mock_tmp,
            patch("tempfile.mkdtemp", return_value="/tmp/mineru_cli_test"),
            patch("os.path.exists", return_value=True),
            patch("os.unlink"),
            patch("shutil.rmtree"),
        ):
            mock_f = MagicMock()
            mock_f.name = "/tmp/test.pdf"
            mock_tmp.return_value.__enter__.return_value = mock_f

            result = _try_parse_with_mineru_cli(b"fake pdf")
        assert result is None

    @pytest.mark.skipif(
        not shutil.which("magic-pdf"),
        reason="magic-pdf CLI 未安装，跳过集成测试",
    )
    def test_cli_integration_smoke(self) -> None:
        """集成冒烟测试（需要 magic-pdf 已安装）。"""
        from dayu.mineru_runtime import _try_parse_with_mineru_cli

        # TXT-based PDF（纯文本）
        pdf_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 100 700 Td (Hello World) Tj ET\nendstream\nendobj\n5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\nxref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000260 00000 n \n0000000360 00000 n \ntrailer<</Size 6/Root 1 0 R>>\nstartxref\n437\n%%EOF"
        result = _try_parse_with_mineru_cli(pdf_content)
        assert result is not None
        assert result.backend == DocumentBackend.MINERU_LOCAL
