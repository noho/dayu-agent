"""MinerU 集成方案的统一中间格式与格式版本探测。

本模块定义 MinerU / Docling 多后端的输出收敛格式 ``ConvertedDocument``，
以及 MinerU content_list_v2 输出结构的版本探测工具。

设计目标：

- ``ConvertedDocument`` 是各个 PDF 解析后端（MinerU 云 API、MinerU 本地、Docling）
  输出的**唯一中间格式**，上层只关心此 dataclass，不感知后端差异。
- ``DocumentBackend`` 枚举提供所有受支持后端的可识别标识。
- ``detect_mineru_content_list_version`` 在 MinerU content_list_v2 输出结构
  发生产出格式变更时（设计文档 risk item），通过字段指纹差异感知并告警。

依赖方向：本模块不依赖任何运行时模块（不 import ``docling_runtime`` 或
``mineru_runtime``），只定义纯数据结构和探测函数，是所有 PDF 后端模块的
**依赖目标**（它们 import 本模块）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from dayu.log import Log

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_MODULE = __name__

# ---------------------------------------------------------------------------
# 后端标识
# ---------------------------------------------------------------------------


class DocumentBackend(str, Enum):
    """受支持的 PDF 解析后端标识。

    ``mineru_cloud`` 调用 MinerU 云 API（异步提交 + 轮询结果）。
    ``mineru_local`` 调用 MinerU 本地 Python API 或 CLI。
    ``docling`` 调用 Dayu 现有 Docling 后端（含 docling-parse / pypdfium2 回退）。
    """

    MINERU_CLOUD = "mineru_cloud"
    MINERU_LOCAL = "mineru_local"
    DOCLING = "docling"


# ---------------------------------------------------------------------------
# 统一中间格式
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentSection:
    """文档中的一个章节。

    Attributes:
        title: 章节标题文字。
        level: 标题层级（1 为最高级）。
        content: 章节正文 Markdown 内容。
        page_idx: 章节开始的零基页码；不确定时可为 ``None``。
    """

    title: str
    level: int
    content: str
    page_idx: int | None = None


@dataclass(frozen=True)
class DocumentTable:
    """文档中的一个表格。

    Attributes:
        caption: 表格标题；无标题时可为空字符串。
        html: 表格的 HTML 表示（统一用 HTML）。
        page_idx: 表格所在的零基页码；不确定时可为 ``None``。
    """

    caption: str
    html: str
    page_idx: int | None = None


@dataclass(frozen=True)
class DocumentImage:
    """文档中的一张图片。

    Attributes:
        path: 图片路径（可以是本地文件路径或 base64 data URI）。
        caption: 图片标题或 alt 文字。
        page_idx: 图片所在的零基页码；不确定时可为 ``None``。
    """

    path: str
    caption: str
    page_idx: int | None = None


@dataclass(frozen=True)
class ConvertedDocument:
    """PDF 解析结果统一中间格式。

    所有 PDF 解析后端（MinerU 云 API / MinerU 本地 / Docling）的输出
    均收敛为此 dataclass，上层流程只依赖此结构。

    Attributes:
        backend: 产生该结果的后端标识。
        sections: 文档章节列表，按出现顺序排列。
        tables: 文档表格列表，按出现顺序排列。
        images: 文档图片列表，按出现顺序排列。
        raw_markdown: 全文原始 Markdown 内容。
        metadata: 与后端无关的元信息（总页数、文件大小等）。
    """

    backend: DocumentBackend
    sections: tuple[DocumentSection, ...] = field(default_factory=tuple)
    tables: tuple[DocumentTable, ...] = field(default_factory=tuple)
    images: tuple[DocumentImage, ...] = field(default_factory=tuple)
    raw_markdown: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MinerU content_list_v2 格式指纹与版本探测
# ---------------------------------------------------------------------------

#: 已知稳定 block 类型集合（用于类型 fallback）。
#: 注意：需要与 mineru_runtime.py 中 _convert_mineru_result 的类型分发保持一致。
_KNOWN_BLOCK_TYPES: frozenset[str] = frozenset({
    "text",
    "paragraph",
    "title",
    "table",
    "figure",
    "image",
})


def detect_mineru_content_list_version(
    raw_content_list: Sequence[Mapping[str, object]],
) -> str:
    """检测 MinerU content_list 输出结构的格式版本。

    通过检查前 10 个包含 ``bbox`` 字段的元素的 bbox 类型来判断版本：

    - ``"v1"``：bbox 是 dict 格式（``{"x0": ..., "y0": ...}``）。
    - ``"v2"``：bbox 是 list 格式（``[x0, y0, x1, y1]``）。
    - ``"unknown"``：前 10 个元素均无 bbox 字段，无法判断。

    Args:
        raw_content_list: MinerU 返回的 ``content_list`` 列表。

    Returns:
        格式版本标识字符串：``"v1"``、``"v2"`` 或 ``"unknown"``。

    Raises:
        TypeError: ``raw_content_list`` 为空或非序列时抛出。
    """
    if not raw_content_list:
        raise TypeError("MinerU content_list 为空或不可迭代，无法检测版本")

    for item in raw_content_list[:10]:
        bbox = item.get("bbox")
        if bbox is None:
            continue
        if isinstance(bbox, list):
            return "v2"
        if isinstance(bbox, dict):
            return "v1"

    return "unknown"


def parse_content_list(
    content_list: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    """解析 MinerU content_list，对未知格式做 best-effort fallback。

    遍历每个 block 并尝试解析；单个 block 解析失败时记录警告并跳过，
    而非让整个流程崩溃。

    Args:
        content_list: MinerU 返回的 ``content_list`` 列表。

    Returns:
        解析后的 block 列表（可能少于输入数量）。
    """
    version = detect_mineru_content_list_version(content_list)
    if version == "unknown":
        Log.warn("MinerU 格式版本未知，尝试 best-effort 解析", module=_MODULE)

    parsed: list[Mapping[str, object]] = []
    for block in content_list:
        try:
            parsed.append(parse_block(block))
        except Exception as exc:
            Log.warn(
                f"跳过解析失败的 block: error={exc}, block_keys={list(block.keys())}",
                module=_MODULE,
            )
            continue
    return parsed


def parse_block(block: Mapping[str, object]) -> Mapping[str, object]:
    """解析单个 MinerU block，对未知类型做标记而非崩溃。

    已知类型（text/title/figure/table）原样返回；未知类型附加
    ``_unknown_type=True`` 标记，让上层自行决定处理方式。

    Args:
        block: MinerU content_list 中的单个 block 字典。

    Returns:
        解析后的 block（可能附带 ``_unknown_type`` 标记）。
    """
    block_type = str(block.get("type", "unknown"))
    if block_type in _KNOWN_BLOCK_TYPES:
        return block
    # 未知类型：原样返回，附加标记
    Log.info(f"未知 MinerU block type: {block_type}", module=_MODULE)
    return {**block, "_unknown_type": True}


def detect_mineru_content_list_version_from_bytes(
    raw_json_bytes: bytes,
) -> str:
    """从 MinerU 返回的 JSON 字节流中检测 content_list 版本指纹。

    Args:
        raw_json_bytes: MinerU API 返回的原始 JSON 字节内容。

    Returns:
        格式版本标识字符串，同 ``detect_mineru_content_list_version``。

    Raises:
        TypeError: 字节流为空或 JSON 解析结果不符合预期时抛出。
        json.JSONDecodeError: JSON 格式非法时抛出。
    """
    if not raw_json_bytes:
        raise TypeError("MinerU content_list JSON 字节流为空")

    data = json.loads(raw_json_bytes)

    if not isinstance(data, list):
        # 若 JSON 根是字典（旧格式或包壳），尝试抽取 content_list 字段
        if isinstance(data, dict):
            content_list = data.get("content_list")
            if content_list is None:
                raise TypeError(
                    "MinerU 返回的 JSON 既不是 content_list 数组，"
                    "也不包含 'content_list' 字段"
                )
            raw_content_list: Sequence[Mapping[str, object]] = content_list
        else:
            raise TypeError(
                f"MinerU 返回的 JSON 根元素类型为 {type(data).__name__}，"
                f"期望 list 或 dict"
            )
    else:
        raw_content_list = data

    return detect_mineru_content_list_version(raw_content_list)


# ---------------------------------------------------------------------------
# 辅助工厂函数
# ---------------------------------------------------------------------------


def _build_section(
    *,
    title: str,
    level: int,
    content: str,
    page_idx: int | None = None,
) -> DocumentSection:
    """构造一个 ``DocumentSection``，封装合法性校验。

    Args:
        title: 章节标题。
        level: 标题层级，必须 >= 1。
        content: 章节正文。
        page_idx: 零基页码。

    Returns:
        构造完成的 ``DocumentSection``。

    Raises:
        ValueError: level 小于 1 时抛出。
    """
    if level < 1:
        raise ValueError(f"章节层级必须 >= 1，实际为 {level}")
    return DocumentSection(
        title=title,
        level=level,
        content=content,
        page_idx=page_idx,
    )


def _build_table(
    *,
    caption: str,
    html: str,
    page_idx: int | None = None,
) -> DocumentTable:
    """构造一个 ``DocumentTable``。

    Args:
        caption: 表格标题。
        html: 表格 HTML 内容。
        page_idx: 零基页码。

    Returns:
        构造完成的 ``DocumentTable``。
    """
    return DocumentTable(
        caption=caption,
        html=html,
        page_idx=page_idx,
    )


def _build_image(
    *,
    path: str,
    caption: str,
    page_idx: int | None = None,
) -> DocumentImage:
    """构造一个 ``DocumentImage``。

    Args:
        path: 图片路径。
        caption: 图片标题。
        page_idx: 零基页码。

    Returns:
        构造完成的 ``DocumentImage``。
    """
    return DocumentImage(
        path=path,
        caption=caption,
        page_idx=page_idx,
    )


__all__ = [
    "ConvertedDocument",
    "DocumentBackend",
    "DocumentImage",
    "DocumentSection",
    "DocumentTable",
    "detect_mineru_content_list_version",
    "detect_mineru_content_list_version_from_bytes",
    "parse_content_list",
    "parse_block",
]
