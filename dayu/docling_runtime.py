"""Docling 运行时装配辅助。

本模块集中收敛 Dayu 对 Docling PDF 转换器的稳定装配策略，避免不同
调用点各自依赖 Docling 默认的 ``auto`` 设备选择。

当前策略：

- 若显式设置环境变量 ``DAYU_DOCLING_DEVICE``，则以该值为准。
- 若未显式设置且运行在 macOS，则默认固定为 ``cpu``，避免 Apple
  Silicon 上 Docling 自动选用 ``mps`` 后触发高概率 OOM。
- 其他平台保持 ``auto``，继续让 Docling 自己选择最优设备。
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import PipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter

DOCLING_DEVICE_ENV = "DAYU_DOCLING_DEVICE"
_SUPPORTED_DOCLING_DEVICES = frozenset({"auto", "cpu", "cuda", "mps", "xpu"})
_TABLE_MODE_ACCURATE = "accurate"
_TABLE_MODE_FAST = "fast"


class _DoclingTableStructureOptionsProtocol(Protocol):
    """Docling 表格结构选项最小协议。"""

    mode: "TableFormerMode"
    do_cell_matching: bool


class _DoclingPdfPipelineOptionsProtocol(Protocol):
    """Docling PDF pipeline 选项最小协议。"""

    do_ocr: bool
    do_table_structure: bool
    accelerator_options: "AcceleratorOptions | None"
    table_structure_options: _DoclingTableStructureOptionsProtocol


def resolve_docling_device_name() -> str:
    """解析当前 Docling PDF 转换应使用的设备名。

    Args:
        无。

    Returns:
        Docling 设备名，取值为 ``auto/cpu/cuda/mps/xpu`` 之一。

    Raises:
        RuntimeError: 当 ``DAYU_DOCLING_DEVICE`` 配置了不支持的值时抛出。
    """

    configured_device = str(os.environ.get(DOCLING_DEVICE_ENV, "") or "").strip().lower()
    if configured_device:
        if configured_device not in _SUPPORTED_DOCLING_DEVICES:
            supported = ", ".join(sorted(_SUPPORTED_DOCLING_DEVICES))
            raise RuntimeError(
                f"{DOCLING_DEVICE_ENV} 不支持 {configured_device!r}；"
                f"允许值: {supported}"
            )
        return configured_device

    if sys.platform == "darwin":
        # macOS 上 Docling 默认会优先选用 MPS。对当前项目的 PDF/表格链路，
        # 这会显著提高集成测试和本地运行的 OOM 概率，因此默认固定到 CPU。
        return "cpu"

    return "auto"


def build_docling_pdf_converter(
    *,
    do_ocr: bool = True,
    do_table_structure: bool = True,
    table_mode: str = _TABLE_MODE_ACCURATE,
    do_cell_matching: bool = True,
) -> "DocumentConverter":
    """构造带稳定设备策略的 Docling PDF 转换器。

    Args:
        do_ocr: 是否开启 OCR。
        do_table_structure: 是否开启表格结构识别。
        table_mode: 表格结构模式，仅支持 ``accurate`` 或 ``fast``。
        do_cell_matching: 是否开启表格单元格匹配。

    Returns:
        配置完成的 Docling `DocumentConverter`。

    Raises:
        RuntimeError: Docling 依赖未安装或设备环境变量非法时抛出。
        ValueError: `table_mode` 非法时抛出。
    """

    pipeline_options = build_docling_pdf_pipeline_options(
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
        table_mode=table_mode,
        do_cell_matching=do_cell_matching,
    )

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - 依赖缺失保护
        raise RuntimeError("Docling 未安装，无法构造 PDF 转换器") from exc

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=cast("PipelineOptions", pipeline_options)
            ),
        }
    )


def build_docling_pdf_pipeline_options(
    *,
    do_ocr: bool = True,
    do_table_structure: bool = True,
    table_mode: str = _TABLE_MODE_ACCURATE,
    do_cell_matching: bool = True,
) -> _DoclingPdfPipelineOptionsProtocol:
    """构造带稳定设备策略的 Docling PDF pipeline 选项。

    Args:
        do_ocr: 是否开启 OCR。
        do_table_structure: 是否开启表格结构识别。
        table_mode: 表格结构模式，仅支持 ``accurate`` 或 ``fast``。
        do_cell_matching: 是否开启表格单元格匹配。

    Returns:
        配置完成的 Docling PDF pipeline 选项对象。

    Raises:
        RuntimeError: Docling 依赖未安装或设备环境变量非法时抛出。
        ValueError: `table_mode` 非法时抛出。
    """

    normalized_table_mode = table_mode.strip().lower()
    if normalized_table_mode not in {_TABLE_MODE_ACCURATE, _TABLE_MODE_FAST}:
        raise ValueError(f"不支持的 Docling table_mode: {table_mode}")

    try:
        from docling.datamodel.accelerator_options import (
            AcceleratorOptions,
            AcceleratorDevice,
        )
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableFormerMode,
        )
    except ImportError as exc:  # pragma: no cover - 依赖缺失保护
        raise RuntimeError("Docling 未安装，无法构造 PDF pipeline 选项") from exc

    pipeline_options = cast(_DoclingPdfPipelineOptionsProtocol, PdfPipelineOptions())
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice(resolve_docling_device_name())
    )

    if do_table_structure:
        table_structure_options = cast(
            _DoclingTableStructureOptionsProtocol,
            pipeline_options.table_structure_options,
        )
        table_structure_options.mode = (
            TableFormerMode.ACCURATE
            if normalized_table_mode == _TABLE_MODE_ACCURATE
            else TableFormerMode.FAST
        )
        table_structure_options.do_cell_matching = do_cell_matching

    return pipeline_options
