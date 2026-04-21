"""Docling 运行时装配辅助。

本模块是 Dayu 所有 Docling PDF 转换入口的总控真源，统一负责：

1. 解析稳定的设备策略；
2. 构造带统一参数的 Docling `DocumentConverter`；
3. 在 `auto` 设备转换失败时，回退到 `cpu` 重试一次。

当前策略：

- 若显式设置环境变量 ``DAYU_DOCLING_DEVICE``，则以该值为准。
- 若未显式设置，则默认使用 ``auto``。
- 若有效设备为 ``auto`` 且转换阶段失败，则统一记录告警并回退到
  ``cpu`` 重试一次。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from dayu.log import Log

if TYPE_CHECKING:
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import PipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter

DOCLING_DEVICE_ENV = "DAYU_DOCLING_DEVICE"
_SUPPORTED_DOCLING_DEVICES = frozenset({"auto", "cpu", "cuda", "mps", "xpu"})
_AUTO_DEVICE_NAME = "auto"
_CPU_DEVICE_NAME = "cpu"
_TABLE_MODE_ACCURATE = "accurate"
_TABLE_MODE_FAST = "fast"
_MODULE = __name__
_TResult = TypeVar("_TResult")
# Protocol 返回值需要协变，才能让更具体的转换结果回调安全替换更宽的调用点。
_TResultCovariant = TypeVar("_TResultCovariant", covariant=True)


class DoclingRuntimeInitializationError(RuntimeError):
    """Docling 运行时初始化错误。"""


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


class _DoclingPdfConvertOperation(Protocol[_TResultCovariant]):
    """Docling PDF 转换执行回调协议。"""

    def __call__(self, converter: "DocumentConverter") -> _TResultCovariant:
        """使用已构造的转换器执行一次转换。"""

        ...


def _normalize_docling_device_name(device_name: str) -> str:
    """规范化并校验 Docling 设备名。

    Args:
        device_name: 候选设备名。

    Returns:
        规范化后的设备名。

    Raises:
        DoclingRuntimeInitializationError: 设备名不在允许列表时抛出。
    """

    normalized_device_name = device_name.strip().lower()
    if normalized_device_name not in _SUPPORTED_DOCLING_DEVICES:
        supported = ", ".join(sorted(_SUPPORTED_DOCLING_DEVICES))
        raise DoclingRuntimeInitializationError(
            f"{DOCLING_DEVICE_ENV} 不支持 {normalized_device_name!r}；"
            f"允许值: {supported}"
        )
    return normalized_device_name


def resolve_docling_device_name() -> str:
    """解析当前 Docling PDF 转换应使用的设备名。

    Args:
        无。

    Returns:
        Docling 设备名，取值为 ``auto/cpu/cuda/mps/xpu`` 之一。

    Raises:
        DoclingRuntimeInitializationError: 当 ``DAYU_DOCLING_DEVICE`` 配置了不支持的值时抛出。
    """

    configured_device = str(os.environ.get(DOCLING_DEVICE_ENV, "") or "").strip()
    if configured_device:
        return _normalize_docling_device_name(configured_device)

    return _AUTO_DEVICE_NAME


def build_docling_pdf_converter(
    *,
    do_ocr: bool = True,
    do_table_structure: bool = True,
    table_mode: str = _TABLE_MODE_ACCURATE,
    do_cell_matching: bool = True,
    device_name: str | None = None,
) -> "DocumentConverter":
    """构造带稳定设备策略的 Docling PDF 转换器。

    Args:
        do_ocr: 是否开启 OCR。
        do_table_structure: 是否开启表格结构识别。
        table_mode: 表格结构模式，仅支持 ``accurate`` 或 ``fast``。
        do_cell_matching: 是否开启表格单元格匹配。
        device_name: 显式设备名；为空时按 `resolve_docling_device_name()` 解析。

    Returns:
        配置完成的 Docling `DocumentConverter`。

    Raises:
        DoclingRuntimeInitializationError: Docling 依赖未安装或设备环境变量非法时抛出。
        ValueError: `table_mode` 非法时抛出。
    """

    pipeline_options = build_docling_pdf_pipeline_options(
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
        table_mode=table_mode,
        do_cell_matching=do_cell_matching,
        device_name=device_name,
    )

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - 依赖缺失保护
        raise DoclingRuntimeInitializationError("Docling 未安装，无法构造 PDF 转换器") from exc

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
    device_name: str | None = None,
) -> _DoclingPdfPipelineOptionsProtocol:
    """构造带稳定设备策略的 Docling PDF pipeline 选项。

    Args:
        do_ocr: 是否开启 OCR。
        do_table_structure: 是否开启表格结构识别。
        table_mode: 表格结构模式，仅支持 ``accurate`` 或 ``fast``。
        do_cell_matching: 是否开启表格单元格匹配。
        device_name: 显式设备名；为空时按 `resolve_docling_device_name()` 解析。

    Returns:
        配置完成的 Docling PDF pipeline 选项对象。

    Raises:
        DoclingRuntimeInitializationError: Docling 依赖未安装或设备环境变量非法时抛出。
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
        raise DoclingRuntimeInitializationError("Docling 未安装，无法构造 PDF pipeline 选项") from exc

    normalized_device_name = (
        resolve_docling_device_name()
        if device_name is None
        else _normalize_docling_device_name(device_name)
    )

    pipeline_options = cast(_DoclingPdfPipelineOptionsProtocol, PdfPipelineOptions())
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice(normalized_device_name)
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


def run_docling_pdf_conversion(
    convert_operation: _DoclingPdfConvertOperation[_TResult],
    *,
    do_ocr: bool = True,
    do_table_structure: bool = True,
    table_mode: str = _TABLE_MODE_ACCURATE,
    do_cell_matching: bool = True,
) -> _TResult:
    """执行带统一设备策略的 Docling PDF 转换。

    Args:
        convert_operation: 接收 `DocumentConverter` 并执行具体转换的回调。
        do_ocr: 是否开启 OCR。
        do_table_structure: 是否开启表格结构识别。
        table_mode: 表格结构模式，仅支持 ``accurate`` 或 ``fast``。
        do_cell_matching: 是否开启表格单元格匹配。

    Returns:
        由 `convert_operation` 返回的转换结果。

    Raises:
        DoclingRuntimeInitializationError: Docling 依赖缺失或设备配置非法时抛出。
        ValueError: `table_mode` 非法时抛出。
    """

    resolved_device_name = resolve_docling_device_name()
    try:
        converter = build_docling_pdf_converter(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            table_mode=table_mode,
            do_cell_matching=do_cell_matching,
            device_name=resolved_device_name,
        )
    except DoclingRuntimeInitializationError:
        raise
    except Exception as exc:
        raise DoclingRuntimeInitializationError(f"Docling 转换器初始化失败: {exc}") from exc
    auto_convert_error: Exception | None = None
    try:
        return convert_operation(converter)
    except Exception as exc:
        if resolved_device_name != _AUTO_DEVICE_NAME:
            raise
        auto_convert_error = exc
        Log.warn(
            (
                "Docling auto 设备转换失败，准备回退 CPU 重试一次: "
                f"error_type={type(exc).__name__} error={exc}"
            ),
            module=_MODULE,
        )
    try:
        retry_converter = build_docling_pdf_converter(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            table_mode=table_mode,
            do_cell_matching=do_cell_matching,
            device_name=_CPU_DEVICE_NAME,
        )
    except DoclingRuntimeInitializationError:
        raise
    except Exception as retry_init_exc:
        raise DoclingRuntimeInitializationError(
            f"Docling CPU 回退初始化失败: {retry_init_exc}"
        ) from retry_init_exc
    try:
        return convert_operation(retry_converter)
    except Exception as retry_exc:
        # CPU 重试若仍失败，保留第一次 auto 转换失败作为异常因果链，便于排查真实退化路径。
        raise retry_exc from auto_convert_error
