"""Docling 运行时装配辅助。

本模块是 Dayu 所有 Docling PDF 转换入口的总控真源，统一负责：

1. 解析稳定的设备策略；
2. 构造带统一参数的 Docling `DocumentConverter`；
3. 维护一条二维（backend × device）的有序回退尝试链，自动绕开
   docling-parse 后端在某些上市公司年报 PDF 上把合法文档判定为
   ``not valid`` 的情况，并兼容 GPU/MPS 推理崩溃后的 CPU 兜底。

当前策略：

- 若显式设置环境变量 ``DAYU_DOCLING_DEVICE``，则以该值为准。
- 若未显式设置，则默认使用 ``auto``。
- 转换尝试链按下列顺序展开：
  1. ``backend=docling-parse, device=resolved``：默认行为，保留现状。
  2. ``backend=pypdfium2, device=resolved``：专治 docling-parse 解析
     阶段把合法 PDF 判 invalid 的故障。
  3. ``backend=docling-parse, device=cpu``：仅当 ``resolved == auto``
     时追加，专治加速器栈在 ``auto`` 阶段崩溃的故障。
- 任意一档成功即返回；全部失败时抛出最后一次异常，并以**首次**失败
  作为 ``__cause__``，便于排查首因。
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from dayu.log import Log

if TYPE_CHECKING:
    from docling.backend.abstract_backend import AbstractDocumentBackend
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.pipeline_options import PipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter

DOCLING_DEVICE_ENV = "DAYU_DOCLING_DEVICE"
_SUPPORTED_DOCLING_DEVICES = frozenset({"auto", "cpu", "cuda", "mps", "xpu"})
_AUTO_DEVICE_NAME = "auto"
_CPU_DEVICE_NAME = "cpu"
_DOCLING_PARSE_BACKEND_NAME = "docling-parse"
_PYPDFIUM2_BACKEND_NAME = "pypdfium2"
_SUPPORTED_DOCLING_BACKENDS = frozenset({_DOCLING_PARSE_BACKEND_NAME, _PYPDFIUM2_BACKEND_NAME})
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


@dataclass(frozen=True)
class _DoclingConversionAttempt:
    """一次 Docling PDF 转换尝试的描述。

    每个尝试由 (backend_name, device_name) 二元组唯一标识：

    - ``backend_name``：决定 PDF 解析后端，可选 ``docling-parse`` 或 ``pypdfium2``。
    - ``device_name``：决定加速器设备，与 ``DAYU_DOCLING_DEVICE`` 同空间。
    """

    backend_name: str
    device_name: str


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


def _normalize_docling_backend_name(backend_name: str) -> str:
    """规范化并校验 Docling backend 名。

    Args:
        backend_name: 候选 backend 名。

    Returns:
        规范化后的 backend 名。

    Raises:
        DoclingRuntimeInitializationError: backend 名不在允许列表时抛出。
    """

    normalized_backend_name = backend_name.strip().lower()
    if normalized_backend_name not in _SUPPORTED_DOCLING_BACKENDS:
        supported = ", ".join(sorted(_SUPPORTED_DOCLING_BACKENDS))
        raise DoclingRuntimeInitializationError(
            f"不支持的 Docling backend {normalized_backend_name!r}；"
            f"允许值: {supported}"
        )
    return normalized_backend_name


def _resolve_backend_class(backend_name: str) -> type["AbstractDocumentBackend"]:
    """把 backend 名映射到 Docling 后端实现类。

    集中此处的 lazy import 是为了将 Docling 第三方依赖收口到真源单点，
    便于错误信息统一与依赖缺失的兜底。

    Args:
        backend_name: 已规范化的 backend 名。

    Returns:
        Docling 后端实现类。

    Raises:
        DoclingRuntimeInitializationError: Docling 依赖缺失或 backend 名非法时抛出。
    """

    normalized_backend_name = _normalize_docling_backend_name(backend_name)
    try:
        if normalized_backend_name == _DOCLING_PARSE_BACKEND_NAME:
            from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

            return DoclingParseDocumentBackend
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

        return PyPdfiumDocumentBackend
    except ImportError as exc:  # pragma: no cover - 依赖缺失保护
        raise DoclingRuntimeInitializationError(
            f"Docling 未安装，无法解析 backend {normalized_backend_name!r}"
        ) from exc


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


def _plan_conversion_attempts(resolved_device_name: str) -> list[_DoclingConversionAttempt]:
    """按二维回退策略生成 Docling 转换尝试链。

    顺序设计：

    1. ``(docling-parse, resolved)``：保留现状，覆盖正常路径。
    2. ``(pypdfium2, resolved)``：救 docling-parse 解析失败（已实证）。
    3. ``(docling-parse, cpu)``：仅当 ``resolved == auto`` 时追加，
       救加速器栈在 ``auto`` 阶段崩溃的故障。

    Args:
        resolved_device_name: 已规范化的 Docling 设备名。

    Returns:
        有序的尝试链，至少包含一项。

    Raises:
        无。
    """

    attempts: list[_DoclingConversionAttempt] = [
        _DoclingConversionAttempt(
            backend_name=_DOCLING_PARSE_BACKEND_NAME,
            device_name=resolved_device_name,
        ),
        _DoclingConversionAttempt(
            backend_name=_PYPDFIUM2_BACKEND_NAME,
            device_name=resolved_device_name,
        ),
    ]
    if resolved_device_name == _AUTO_DEVICE_NAME:
        attempts.append(
            _DoclingConversionAttempt(
                backend_name=_DOCLING_PARSE_BACKEND_NAME,
                device_name=_CPU_DEVICE_NAME,
            )
        )
    return attempts


def build_docling_pdf_converter(
    *,
    do_ocr: bool = True,
    do_table_structure: bool = True,
    table_mode: str = _TABLE_MODE_ACCURATE,
    do_cell_matching: bool = True,
    device_name: str | None = None,
    backend_name: str = _DOCLING_PARSE_BACKEND_NAME,
) -> "DocumentConverter":
    """构造带稳定设备与 backend 策略的 Docling PDF 转换器。

    Args:
        do_ocr: 是否开启 OCR。
        do_table_structure: 是否开启表格结构识别。
        table_mode: 表格结构模式，仅支持 ``accurate`` 或 ``fast``。
        do_cell_matching: 是否开启表格单元格匹配。
        device_name: 显式设备名；为空时按 `resolve_docling_device_name()` 解析。
        backend_name: 显式 PDF backend 名；默认 ``docling-parse``。

    Returns:
        配置完成的 Docling `DocumentConverter`。

    Raises:
        DoclingRuntimeInitializationError: Docling 依赖未安装、设备或 backend 配置非法时抛出。
        ValueError: `table_mode` 非法时抛出。
    """

    pipeline_options = build_docling_pdf_pipeline_options(
        do_ocr=do_ocr,
        do_table_structure=do_table_structure,
        table_mode=table_mode,
        do_cell_matching=do_cell_matching,
        device_name=device_name,
    )

    backend_class = _resolve_backend_class(backend_name)

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - 依赖缺失保护
        raise DoclingRuntimeInitializationError("Docling 未安装，无法构造 PDF 转换器") from exc

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=cast("PipelineOptions", pipeline_options),
                backend=backend_class,
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


def _build_attempt_converter(
    attempt: _DoclingConversionAttempt,
    *,
    do_ocr: bool,
    do_table_structure: bool,
    table_mode: str,
    do_cell_matching: bool,
    attempt_index: int,
    total_attempts: int,
) -> "DocumentConverter":
    """为指定尝试构造 Docling 转换器，并把初始化异常包装成统一类型。

    Args:
        attempt: 当前尝试描述。
        do_ocr: 是否开启 OCR。
        do_table_structure: 是否开启表格结构识别。
        table_mode: 表格结构模式。
        do_cell_matching: 是否开启表格单元格匹配。
        attempt_index: 当前尝试在尝试链中的 0 基序号。
        total_attempts: 尝试链总长度。

    Returns:
        Docling 转换器实例。

    Raises:
        DoclingRuntimeInitializationError: 初始化失败时抛出。
    """

    try:
        return build_docling_pdf_converter(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            table_mode=table_mode,
            do_cell_matching=do_cell_matching,
            device_name=attempt.device_name,
            backend_name=attempt.backend_name,
        )
    except DoclingRuntimeInitializationError:
        raise
    except Exception as exc:
        raise DoclingRuntimeInitializationError(
            f"Docling 转换器初始化失败 (attempt {attempt_index + 1}/{total_attempts}: "
            f"backend={attempt.backend_name}, device={attempt.device_name}): {exc}"
        ) from exc


def run_docling_pdf_conversion(
    convert_operation: _DoclingPdfConvertOperation[_TResult],
    *,
    do_ocr: bool = True,
    do_table_structure: bool = True,
    table_mode: str = _TABLE_MODE_ACCURATE,
    do_cell_matching: bool = True,
) -> _TResult:
    """执行带二维（backend × device）回退的 Docling PDF 转换。

    尝试链由 `_plan_conversion_attempts` 给出，命中即返回；全链失败时抛出
    最后一次异常，并以首次失败作为 ``__cause__`` 保留首因。

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
    attempts = _plan_conversion_attempts(resolved_device_name)
    total_attempts = len(attempts)
    first_failure: Exception | None = None
    last_failure: Exception | None = None
    for attempt_index, attempt in enumerate(attempts):
        converter = _build_attempt_converter(
            attempt,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            table_mode=table_mode,
            do_cell_matching=do_cell_matching,
            attempt_index=attempt_index,
            total_attempts=total_attempts,
        )
        try:
            return convert_operation(converter)
        except Exception as exc:
            last_failure = exc
            if first_failure is None:
                first_failure = exc
            if attempt_index + 1 < total_attempts:
                next_attempt = attempts[attempt_index + 1]
                Log.warn(
                    (
                        "Docling 转换失败，准备按尝试链回退: "
                        f"attempt={attempt_index + 1}/{total_attempts} "
                        f"failed_backend={attempt.backend_name} "
                        f"failed_device={attempt.device_name} "
                        f"next_backend={next_attempt.backend_name} "
                        f"next_device={next_attempt.device_name} "
                        f"error_type={type(exc).__name__} error={exc}"
                    ),
                    module=_MODULE,
                )
                continue
    # 全链失败：保留首次失败为 __cause__，便于排查首因。
    assert last_failure is not None
    if first_failure is not None and first_failure is not last_failure:
        raise last_failure from first_failure
    raise last_failure
