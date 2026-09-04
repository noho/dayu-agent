"""Docling 降级替代方案。

当 Docling（依赖 PyTorch/transformers）因内存不足无法初始化时，
使用轻量级 PDF 解析器提取文本与表格。

引擎策略（按优先级尝试）：
1. ``pdf-inspector``（Firecrawl 开源的纯 Rust 引擎，MIT 许可，无 ML 依赖）：
   - 速度极快（264 页年报约 3 秒）
   - 支持结构化表格提取（Markdown 表格 → docling TableData）
   - 已知缺陷：CJK 字体子集 ToUnicode 为 full-range identity bfrange 时
     部分中文映射为 ``U+FFFD`` 替换符（firecrawl/pdf-inspector issue #246）
2. ``poppler``（``pdftotext -bbox-layout``，poppler-utils 系统工具）：
   - 依赖字体的 CMap 解码，对 pdf-inspector/pypdf 都解不了的
     CID 子集字库 PDF 能输出正确中文（此类 PDF pypdf 会整页乱码）
   - 从词级坐标重建可读文本行 + 逐行分列表格
3. ``pypdf``（纯 Python）：最终兜底，对常规 CJK 字体解码正确率较高，
   但无表格结构，且对 CID 子集字库会乱码。

当 pdf-inspector 失败/为空、或其文本层（pypdf）检测到乱码时，
优先改用 poppler 重建文本与表格；poppler 也不可用时才回退 pypdf。

本模块通过 docling-core 的 Python API 构建 DoclingDocument 实例，
再序列化为 JSON，确保与 DoclingProcessor 完全兼容。
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr
from typing import TYPE_CHECKING, Any

from dayu.log import Log

if TYPE_CHECKING:
    from docling_core.types.doc.document import (
        DoclingDocument,
    )
    from docling_core.types.doc.items.table.table_data import (
        TableCell,
        TableData,
    )

_MODULE = "FINS.DOCLING_FALLBACK"

# 每页最多提取的文本行数（防止超大 PDF 导致 JSON 过大）
_MAX_TEXT_ITEMS_PER_PAGE = 500
# 一行最少的字符数（过滤空白/噪声行）
_MIN_LINE_LENGTH = 2
# 表格单元格文本长度上限（防止异常长文本撑爆 JSON）
_MAX_CELL_TEXT_LEN = 500


def convert_pdf_bytes_to_docling_payload(
    raw_data: bytes,
    *,
    stream_name: str,
) -> dict[str, Any]:
    """将 PDF 字节流转换为 DoclingDocument 兼容的结构化字典。

    在低内存环境（1.6GB 以下）下，Docling 加载 PyTorch/transformers
    时会因内存不足而挂起。本函数作为降级方案，优先使用 pdf-inspector
    （纯 Rust，无 ML 依赖）快速提取文本与表格，失败时回退 pypdf。

    Args:
        raw_data: PDF 原始字节内容。
        stream_name: 流名称，建议直接传文件名以保留扩展名。

    Returns:
        DoclingDocument.model_dump_json() 输出字典。
    """
    # 引擎一：pdf-inspector（快 + 表格结构；文本层 pypdf）
    try:
        return _convert_via_pdf_inspector(raw_data, stream_name)
    except Exception as exc:  # noqa: BLE001 - 缺失/空/乱码都继续降级
        Log.warning(
            f"pdf-inspector 提取失败 ({type(exc).__name__}: {exc})，"
            f"尝试 poppler: {stream_name}",
            module=_MODULE,
        )

    # 引擎二：poppler（pdftotext -bbox-layout）——处理 CID 子集字库导致的乱码
    try:
        return _convert_via_poppler(raw_data, stream_name)
    except Exception as exc:  # noqa: BLE001 - poppler 不可用/解析失败则回退 pypdf
        Log.warning(
            f"poppler 提取失败 ({type(exc).__name__}: {exc})，"
            f"回退 pypdf: {stream_name}",
            module=_MODULE,
        )

    # 引擎三：pypdf（最终兜底，可能对特殊字库乱码但已是可用方案）
    return _convert_via_pypdf(raw_data, stream_name)


def convert_pdf_bytes_to_docling_json_bytes(
    raw_data: bytes,
    stream_name: str,
) -> bytes:
    """将 PDF 字节流转换为序列化的 DoclingDocument JSON 字节内容。

    Args:
        raw_data: PDF 原始字节内容。
        stream_name: 流名称。

    Returns:
        已编码为 UTF-8 的 Docling JSON 字节内容。
    """
    payload = convert_pdf_bytes_to_docling_payload(
        raw_data, stream_name=stream_name
    )
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


# ══════════════════════════════════════════════════════════════════
# 引擎一：pdf-inspector
# ══════════════════════════════════════════════════════════════════


def _convert_via_pdf_inspector(
    raw_data: bytes, stream_name: str
) -> dict[str, Any]:
    """使用 pdf-inspector 提取文本与结构化表格。

    Args:
        raw_data: PDF 原始字节内容。
        stream_name: 流名称。

    Returns:
        DoclingDocument JSON 字典。

    Raises:
        ImportError: pdf-inspector 未安装。
        RuntimeError: 提取结果为空（可能为扫描件或引擎缺陷）。
    """
    try:
        import pdf_inspector
    except ImportError:
        raise RuntimeError(
            "pdf-inspector 未安装，无法执行 PDF 提取。"
            "请执行: pip install pdf-inspector"
        )

    Log.info(
        f"使用 pdf-inspector 提取 PDF: {stream_name} ({len(raw_data)} bytes)",
        module=_MODULE,
    )

    result = pdf_inspector.process_pdf_bytes(raw_data)
    markdown = result.markdown or ""
    if not markdown.strip():
        raise RuntimeError(
            f"pdf-inspector 提取为空: type={result.pdf_type} "
            f"encoding_issues={result.has_encoding_issues}"
        )

    # 解析 markdown → 文本块 + 表格
    # 文本块仅用于兜底；表格是 pdf-inspector 的核心价值（结构化提取）
    _text_blocks, tables = _parse_markdown(markdown)

    # 文本层使用 pypdf 提取（中文解码正确率高于 pdf-inspector，
    # 后者对 CJK 子集字体存在 issue #246 的乱码缺陷）。
    # 表格层使用 pdf-inspector（结构化 Markdown 表格）。
    text_blocks = _extract_pypdf_text_blocks(raw_data)
    if _is_text_garbled(text_blocks):
        # pypdf 对该 PDF 的字库解码成乱码（CID 子集字库），
        # 交给引擎二 poppler 重建（poppler 依赖字体 CMap 解码）。
        raise RuntimeError(
            "pypdf 文本层乱码（CID 子集字库），转交 poppler 重建"
        )

    # 构建 docling 文档
    doc = _build_docling_document(
        stream_name, text_blocks, tables
    )

    with open(os.devnull, "w") as f:
        with redirect_stderr(f):
            return json.loads(doc.model_dump_json(exclude_none=True))


def _parse_markdown(
    markdown: str,
) -> tuple[list[str], list[list[list[str]]]]:
    """解析 pdf-inspector 的 Markdown 输出。

    将 Markdown 拆分为两类内容：
    - 普通文本块（非表格行）
    - 表格（连续的 ``|`` 分隔行），每个表格为二维字符串数组

    Args:
        markdown: pdf-inspector 输出的 Markdown 文本。

    Returns:
        (文本块列表, 表格列表)。表格为 ``list[list[list[str]]]``，
        即每个表格是行列表，每行是单元格列表。
    """
    text_blocks: list[str] = []
    tables: list[list[list[str]]] = []
    current_table: list[list[str]] | None = None

    for raw_line in markdown.split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()

        # 表格行：以 | 开头或结尾，且含多个 |（至少 2 个单元格）
        is_table_row = (
            stripped.startswith("|")
            and stripped.count("|") >= 2
        )
        # Markdown 表格分隔行（|---|---|）
        is_separator = re.fullmatch(r"\|[\s:\-|]+\|", stripped)

        if is_table_row and not is_separator:
            cells = _split_table_row(stripped)
            # 过滤伪表格：单元格过长（>150 字符）说明是正文被误判
            if any(len(c) > 150 for c in cells):
                _flush_table(tables, current_table)
                current_table = None
                if stripped:
                    text_blocks.append(line)
                continue
            if current_table is None:
                current_table = []
            current_table.append(cells)
        else:
            # 表格结束
            _flush_table(tables, current_table)
            current_table = None
            if stripped:
                text_blocks.append(line)

    _flush_table(tables, current_table)

    return text_blocks, tables


def _flush_table(
    tables: list[list[list[str]]], current_table: list[list[str]] | None
) -> None:
    """结束并归一化一个表格。

    统一所有行到最大列数（空单元格补空串），过滤过短表格
    （少于 2 行 或 少于 2 列）。

    Args:
        tables: 表格列表（原地追加）。
        current_table: 刚结束的表格；None 表示无待处理表格。
    """
    if current_table is None or len(current_table) < 2:
        return
    ncols = max((len(row) for row in current_table), default=0)
    if ncols < 2:
        return
    normalized = [
        row + [""] * (ncols - len(row)) for row in current_table
    ]
    tables.append(normalized)


def _split_table_row(row: str) -> list[str]:
    """拆分 Markdown 表格行（处理转义管道符）。

    Args:
        row: 形如 ``|a|b|c|`` 的表格行。

    Returns:
        单元格文本列表。
    """
    inner = row.strip().strip("|")
    cells: list[str] = []
    buf: list[str] = []
    for ch in inner:
        if ch == "|":
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    cells.append("".join(buf).strip())
    return cells


def _extract_pypdf_text_blocks(raw_data: bytes) -> list[str]:
    """使用 pypdf 提取全部文本行（作为文档文本层）。

    pdf-inspector 对 CJK 子集字体的解码存在缺陷（issue #246），
    文本层统一使用 pypdf 以保证中文正确性；表格层保留
    pdf-inspector 的结构化输出。

    Args:
        raw_data: PDF 原始字节内容。

    Returns:
        文本行列表。
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return []

    try:
        reader = PdfReader(io.BytesIO(raw_data))
    except Exception as exc:  # noqa: BLE001
        Log.warning(
            f"pypdf 打开 PDF 失败: {exc}", module=_MODULE
        )
        return []

    lines: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        for raw_line in page_text.split("\n"):
            line = raw_line.strip()
            if len(line) >= _MIN_LINE_LENGTH:
                lines.append(line)
    return lines


def _build_docling_document(
    stream_name: str,
    text_blocks: list[str],
    tables: list[list[list[str]]],
) -> Any:
    """使用 docling-core Python API 构建 DoclingDocument。

    Args:
        stream_name: PDF 文件名。
        text_blocks: 文本块列表（已修复乱码）。
        tables: 解析出的表格（二维单元格数组）。

    Returns:
        DoclingDocument 实例。
    """
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.items.table.table_data import (
        TableCell,
        TableData,
    )
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name=stream_name)

    text_item_count = 0
    for block in text_blocks:
        line = block.strip()
        if not line or len(line) < _MIN_LINE_LENGTH:
            continue
        label = _infer_label(line, text_item_count)
        doc.add_text(label=label, text=line, prov=None)
        text_item_count += 1

    for table_cells in tables:
        try:
            data = _build_table_data(table_cells)
            doc.add_table(data=data, prov=None)
        except Exception as exc:  # noqa: BLE001 - 单表失败不影响整体
            Log.warning(
                f"表格构建失败，跳过: {exc}", module=_MODULE
            )

    Log.info(
        f"pdf-inspector 降级完成: {stream_name} — "
        f"{text_item_count} 个文本项, {len(tables)} 个表格",
        module=_MODULE,
    )
    return doc


def _build_table_data(table_cells: list[list[str]]) -> TableData:
    """将二维单元格数组转换为 docling TableData。

    Args:
        table_cells: 表格单元格（行×列）。

    Returns:
        TableData 实例。
    """
    from docling_core.types.doc.items.table.table_data import (
        TableCell,
        TableData,
    )

    num_rows = len(table_cells)
    num_cols = max(len(row) for row in table_cells)

    cells: list[TableCell] = []
    for r, row in enumerate(table_cells):
        for c in range(num_cols):
            text = row[c] if c < len(row) else ""
            text = text[: _MAX_CELL_TEXT_LEN]
            is_header = r == 0
            cells.append(
                TableCell(
                    text=text,
                    start_row_offset_idx=r,
                    end_row_offset_idx=r + 1,
                    start_col_offset_idx=c,
                    end_col_offset_idx=c + 1,
                    column_header=is_header,
                    row_header=False,
                    row_section=False,
                    fillable=False,
                )
            )

    return TableData(table_cells=cells, num_rows=num_rows, num_cols=num_cols)


# ══════════════════════════════════════════════════════════════════
# 引擎二：pypdf（兜底）
# ══════════════════════════════════════════════════════════════════


def _convert_via_pypdf(
    raw_data: bytes, stream_name: str
) -> dict[str, Any]:
    """使用 pypdf 将 PDF 字节流转换为 DoclingDocument 兼容字典。

    Args:
        raw_data: PDF 原始字节内容。
        stream_name: 流名称。

    Returns:
        DoclingDocument JSON 字典。
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError(
            "pypdf 未安装，无法执行降级 PDF 文本提取。"
            "请执行: pip install pypdf"
        )

    Log.info(
        f"使用 pypdf 降级方案处理 PDF: {stream_name} "
        f"({len(raw_data)} bytes)",
        module=_MODULE,
    )

    reader = PdfReader(io.BytesIO(raw_data))
    num_pages = len(reader.pages)
    doc = _build_pypdf_docling_document(reader, stream_name, num_pages)

    with open(os.devnull, "w") as f:
        with redirect_stderr(f):
            return json.loads(doc.model_dump_json(exclude_none=True))


def _build_pypdf_docling_document(
    reader: Any, stream_name: str, num_pages: int
) -> Any:
    """使用 pypdf 页面文本构建 DoclingDocument。

    Args:
        reader: pypdf.PdfReader 实例。
        stream_name: PDF 文件名。
        num_pages: 总页数。

    Returns:
        DoclingDocument 实例。
    """
    from docling_core.types.doc.document import DoclingDocument
    from docling_core.types.doc.labels import DocItemLabel

    doc = DoclingDocument(name=stream_name)

    text_item_count = 0
    for page_no in range(num_pages):
        page = reader.pages[page_no]
        page_text = page.extract_text() or ""

        lines = [l.strip() for l in page_text.split("\n") if l.strip()]
        lines = [l for l in lines if len(l) >= _MIN_LINE_LENGTH]

        for line in lines[: _MAX_TEXT_ITEMS_PER_PAGE]:
            label = _infer_label(line, text_item_count)
            doc.add_text(label=label, text=line, prov=None)
            text_item_count += 1

    Log.info(
        f"pypdf 降级完成: {stream_name} — {num_pages} 页, "
        f"{text_item_count} 个文本项",
        module=_MODULE,
    )
    return doc


# ══════════════════════════════════════════════════════════════════
# 公共工具
# ══════════════════════════════════════════════════════════════════


def _infer_label(line: str, index: int) -> Any:
    """根据行内容和位置推断 DocItemLabel。

    Args:
        line: 文本行。
        index: 文本行序号。

    Returns:
        DocItemLabel 枚举值。
    """
    from docling_core.types.doc.labels import DocItemLabel

    if index == 0:
        return DocItemLabel("title")

    upper = line.strip().upper()

    if upper.startswith("目") and "录" in upper[:5]:
        return DocItemLabel("section_header")

    if any(kw in line for kw in ["第", "节", "章", "篇"]) and len(line) < 80:
        return DocItemLabel("section_header")

    stripped = line.strip()
    if stripped.isdigit() or (
        "/" in stripped
        and all(p.strip().isdigit() for p in stripped.split("/") if p.strip())
    ):
        return DocItemLabel("page_footer")

    if "|" in line:
        return DocItemLabel("text")

    if len(line) > 100:
        return DocItemLabel("paragraph")
    if len(line) > 40:
        return DocItemLabel("text")

    return DocItemLabel("text")


# ══════════════════════════════════════════════════════════════════
# 引擎二：poppler（pdftotext -bbox-layout）
# ══════════════════════════════════════════════════════════════════


# 乱码特征字符的 Unicode 区段：对 CID 子集字库解码失败时，
# 常见输出为这些区段的字符（[U+FFFD] 或非 CJK 高位字符）。
_GARBLED_BLOCKS: tuple[tuple[int, int], ...] = (
    (0x0250, 0x02AF),  # 国际音标扩展
    (0x0370, 0x03FF),  # 希腊文及科普特文
    (0x0530, 0x058F),  # 亚美尼亚文
    (0x0590, 0x05FF),  # 希伯来文
    (0x0600, 0x06FF),  # 阿拉伯文
    (0x0900, 0x097F),  # 天城文
    (0x0980, 0x09FF),  # 孟加拉文
    (0x0A00, 0x0A7F),  # 古木基文
    (0x0A80, 0x0AFF),  # 古吉拉特文
    (0x0E00, 0x0E7F),  # 泰文
    (0x0E80, 0x0EFF),  # 老挝文
    (0x10A0, 0x10FF),  # 乔治亚文
    (0x1E00, 0x1EFF),  # 拉丁文扩展附加
    (0xFFFD, 0xFFFD),  # U+FFFD 替换符
)


def _is_in_garbled_blocks(ch: str) -> bool:
    """判断字符是否落在乱码特征区段。

    Args:
        ch: 单个字符。

    Returns:
        是否落在乱码特征区段。
    """
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _GARBLED_BLOCKS)


def _is_text_garbled(blocks: list[str], *, min_len: int = 200) -> bool:
    """启发式判断文本块是否被乱码污染。

    当非常用高位字符（乱码特征区段）数量显著多于汉字数量时判为乱码。
    英文 PDF（近零汉字、近零特征字符）不会被误判；
    正常中文 PDF（汉字占主导）不会被误判。

    Args:
        blocks: 文本块列表。
        min_len: 参与判定的最低总字符数，过短不判定以避免误判。

    Returns:
        是否为乱码。
    """
    text = "".join(blocks)
    if len(text) < min_len:
        return False
    hanzi = sum(1 for c in text if 0x4E00 <= ord(c) <= 0x9FFF)
    unusual = sum(1 for c in text if _is_in_garbled_blocks(c))
    return unusual > 0 and unusual > hanzi * 2


def _parse_poppler_bbox(
    xml_text: str,
) -> tuple[list[str], list[list[list[str]]]]:
    """解析 ``pdftotext -bbox-layout`` 输出的 XML，重建文本块与表格。

    返回 ``(text_blocks, tables)``：文本块为按行合并的可读中文，
    表格为逐行 gap 分列的二维单元格数组（非跨行对齐）。

    Args:
        xml_text: ``pdftotext -bbox-layout`` 的标准输出。

    Returns:
        文本块列表与表格列表。

    Raises:
        RuntimeError: XML 无法解析或没有任何可读文本。
    """
    cleaned = "".join(
        ch for ch in xml_text if ch in "\t\n\r" or ord(ch) >= 0x20
    )
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError as exc:
        raise RuntimeError(f"poppler bbox XML 解析失败: {exc}") from exc

    text_blocks: list[str] = []
    tables: list[list[list[str]]] = []
    for page in root.iter():
        if page.tag.split("}")[-1] != "page":
            continue
        words: list[tuple[float, float, float, float, str]] = []
        for word in page.iter():
            if word.tag.split("}")[-1] != "word":
                continue
            txt = (word.text or "").strip()
            if not txt:
                continue
            words.append(
                (
                    float(word.get("xMin", "0")),
                    float(word.get("yMin", "0")),
                    float(word.get("xMax", "0")),
                    float(word.get("yMax", "0")),
                    txt,
                )
            )
        if not words:
            continue
        _append_poppler_page(text_blocks, tables, words)
    if not text_blocks:
        raise RuntimeError("poppler 提取为空（无可用文本）")
    return text_blocks, tables


def _append_poppler_page(
    text_blocks: list[str],
    tables: list[list[list[str]]],
    words: list[tuple[float, float, float, float, str]],
) -> None:
    """把一页的词坐标并入文本块与表格结果。

    Args:
        text_blocks: 累积文本块列表（就地追加）。
        tables: 累积表格列表（就地追加）。
        words: 本页词坐标（xMin, yMin, xMax, yMax, 文本）。
    """
    y_tol = 3.0
    gap_tol = 4.0
    rows: list[list[tuple[float, float, str]]] = []
    ordered = sorted(words, key=lambda w: (w[1], w[0]))
    cur_ymin: float | None = None
    for w in ordered:
        if cur_ymin is None or w[1] - cur_ymin > y_tol:
            cur_ymin = w[1]
            rows.append([(w[0], w[2], w[4])])
        else:
            rows[-1].append((w[0], w[2], w[4]))

    for row in rows:
        row.sort(key=lambda item: item[0])
        text_blocks.append(" ".join(item[2] for item in row).strip())

    # 逐行 gap 分列表格（表格页判定：多数行有 >=2 格）
    grid: list[list[str]] = []
    for row in rows:
        cells: list[list[tuple[float, float, str]]] = []
        cur: list[tuple[float, float, str]] = []
        prev_xmax: float | None = None
        for xmin, xmax, txt in row:
            if cur and xmin - (prev_xmax or 0) > gap_tol:
                cells.append(cur)
                cur = []
            cur.append((xmin, xmax, txt))
            prev_xmax = max(prev_xmax or 0, xmax)
        if cur:
            cells.append(cur)
        line = [" ".join(item[2] for item in cell) for cell in cells]
        if any(line):
            grid.append(line)

    ncols = [len(r) for r in grid]
    multi = sum(1 for n in ncols if n >= 2)
    if max(ncols, default=0) >= 2 and multi >= max(2, int(len(grid) * 0.4)):
        maxc = max(ncols)
        for r in grid:
            r += [""] * (maxc - len(r))
        tables.append(grid)


def _convert_via_poppler(raw_data: bytes, stream_name: str) -> dict[str, Any]:
    """使用 poppler（pdftotext -bbox-layout）重建文本与表格。

    依赖 poppler-utils 提供 ``pdftotext`` 命令行工具。对 pypdf 无法
    解码的 CID 子集字库 PDF 能输出正确中文，并从词级坐标恢复表格。

    Args:
        raw_data: PDF 原始字节内容。
        stream_name: 流名称。

    Returns:
        DoclingDocument JSON 字典。

    Raises:
        RuntimeError: poppler-utils 未安装 / pdftotext 失败 / 输出为空。
    """
    binary = shutil.which("pdftotext")
    if not binary:
        raise RuntimeError(
            "poppler-utils 未安装（缺少 pdftotext），无法执行 poppler 提取"
        )

    Log.info(
        f"使用 poppler 提取 PDF: {stream_name} ({len(raw_data)} bytes)",
        module=_MODULE,
    )

    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as fh:
            fh.write(raw_data)
        proc = subprocess.run(
            [binary, "-bbox-layout", tmp_path, "-"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"pdftotext 失败 rc={proc.returncode}: {proc.stderr[:200]}"
            )
        xml_text = proc.stdout
    finally:
        os.unlink(tmp_path)

    text_blocks, tables = _parse_poppler_bbox(xml_text)
    doc = _build_docling_document(stream_name, text_blocks, tables)
    with open(os.devnull, "w") as f:
        with redirect_stderr(f):
            return json.loads(doc.model_dump_json(exclude_none=True))
