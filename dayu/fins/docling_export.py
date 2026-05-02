"""Docling 转换公共出口。

该模块是仓库内调用 :mod:`dayu.docling_runtime` 的**唯一**收敛点，向上层提供
两种稳定签名：

- :func:`convert_pdf_bytes_to_docling_payload`：返回 Docling SDK 导出的结构化字典。
  这是与第三方 docling SDK 交互**不可避免**的 ``dict[str, Any]`` 边界，仅在本
  模块内部允许出现。上层（upload service）若历史上以字典形态使用，可继续使用
  这个出口。
- :func:`convert_pdf_bytes_to_docling_json_bytes`：返回已 JSON 序列化的 ``bytes``。
  下载链路按 ``Callable[[bytes, str], bytes]`` 的强类型协议注入此函数，避免把
  ``Any`` 透传到上层 workflow。

两个函数共用 :func:`dayu.docling_runtime.convert_pdf_bytes_with_docling` 调用；
任何参数策略调整集中在本模块完成，避免 docling-runtime 调用点散落到 upload /
download 双链路造成漂移。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from dayu.docling_runtime import (
    DoclingRuntimeInitializationError,
    convert_pdf_bytes_with_docling,
)

# 下载链路注入点的稳定签名：``(raw_bytes, stream_name) -> json_bytes``。
# 显式以位置参数风格暴露，避免 keyword-only 与 ``Callable[[bytes, str], bytes]``
# 协议不兼容。download workflow 与 filing workflow 应直接引用此别名。
PdfToDoclingJsonBytes = Callable[[bytes, str], bytes]

__all__ = [
    "PdfToDoclingJsonBytes",
    "convert_pdf_bytes_to_docling_payload",
    "convert_pdf_bytes_to_docling_json_bytes",
]


def convert_pdf_bytes_to_docling_payload(
    raw_data: bytes,
    *,
    stream_name: str,
) -> dict[str, Any]:
    """将 PDF 字节流转换为 Docling 导出字典。

    Args:
        raw_data: PDF 原始字节内容。
        stream_name: 流名称，建议直接传文件名以保留扩展名。

    Returns:
        Docling 导出的结构化字典（``result.document.export_to_dict()``）。

    Raises:
        DoclingRuntimeInitializationError: Docling 依赖缺失或装配失败时抛出。
        RuntimeError: Docling 转换失败时抛出。
    """

    try:
        result = convert_pdf_bytes_with_docling(
            raw_data,
            stream_name=stream_name,
            do_ocr=True,
            do_table_structure=True,
            table_mode="accurate",
            do_cell_matching=True,
        )
    except DoclingRuntimeInitializationError:
        raise
    except Exception as exc:  # pragma: no cover - 第三方异常兜底
        raise RuntimeError(f"Docling 转换失败: {stream_name}") from exc
    return result.document.export_to_dict()


def convert_pdf_bytes_to_docling_json_bytes(
    raw_data: bytes,
    stream_name: str,
) -> bytes:
    """将 PDF 字节流转换为序列化后的 Docling JSON 字节内容。

    用于下载链路：下游 workflow 注入签名 ``Callable[[bytes, str], bytes]``，
    本函数即该协议的默认实现，因此两个参数均为位置参数，调用方既可以
    ``convert_pdf_bytes_to_docling_json_bytes(raw, name)`` 也可以使用关键字。
    序列化策略与 SEC 下载链路保持一致：UTF-8 编码、``ensure_ascii=False``
    保留中文字符、两空格缩进便于诊断。

    Args:
        raw_data: PDF 原始字节内容。
        stream_name: 流名称，建议直接传文件名以保留扩展名。

    Returns:
        已编码为 UTF-8 的 Docling JSON 字节内容。

    Raises:
        DoclingRuntimeInitializationError: Docling 依赖缺失或装配失败时抛出。
        RuntimeError: Docling 转换失败时抛出。
    """

    payload = convert_pdf_bytes_to_docling_payload(raw_data, stream_name=stream_name)
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
