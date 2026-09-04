"""Docling 转换公共出口。

该模块是仓库内调用 :mod:`dayu.docling_runtime` 的**唯一**收敛点，向上层提供
两种稳定签名：

- :func:`convert_pdf_bytes_to_docling_payload`：返回 Docling SDK 导出的结构化字典。
- :func:`convert_pdf_bytes_to_docling_json_bytes`：返回已 JSON 序列化的 ``bytes``。

== 降级机制 ==

当 ``DAYU_DOCLING_DISABLE=true``（或环境内存不足导致 Docling 挂起）时，
自动降级为 pypdf 轻量级 PDF 文本提取，无需 PyTorch/transformers 依赖。

判断逻辑：
1. 若 ``DAYU_DOCLING_DISABLE=true`` → 直接使用 pypdf 降级
2. 否则尝试 Docling → 若 DoclingRuntimeInitializationError → 使用 pypdf 降级
3. 若低内存环境（< 2GB 可用 RAM）且之前 Docling 初始化失败 → 自动禁用

设置任意环境变量为 ``1`` / ``true`` 开启，``0`` / ``false`` 关闭（不区分大小写）。
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from dayu.docling_runtime import (
    DoclingRuntimeInitializationError,
    convert_pdf_bytes_with_docling,
)

from dayu.docling_fallback import (
    convert_pdf_bytes_to_docling_payload,
    convert_pdf_bytes_to_docling_json_bytes,
)

# 旧别名：带 _fallback 后缀的导入路径兼容
from dayu.docling_fallback import (
    convert_pdf_bytes_to_docling_payload as convert_pdf_bytes_to_docling_payload_fallback,
    convert_pdf_bytes_to_docling_json_bytes as convert_pdf_bytes_to_docling_json_bytes_fallback,
)
from dayu.log import Log

_MODULE = "FINS.DOCLING_EXPORT"

# 下载链路注入点的稳定签名：``(raw_bytes, stream_name) -> json_bytes``。
PdfToDoclingJsonBytes = Callable[[bytes, str], bytes]


def _env_bool(name: str, default: bool) -> bool:
    """从环境变量读取布尔值。"""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes")


def _is_docling_disabled() -> bool:
    """检查是否强制禁用 Docling。"""
    return _env_bool("DAYU_DOCLING_DISABLE", False)


def _check_low_memory() -> bool:
    """检查是否为低内存环境（< 2GB 可用 RAM）。"""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        mb = kb / 1024
                        return mb < 2048  # < 2GB
    except (OSError, ValueError):
        pass
    return False


def convert_pdf_bytes_to_docling_payload(
    raw_data: bytes,
    *,
    stream_name: str,
) -> dict[str, Any]:
    """将 PDF 字节流转换为结构化字典。

    转换参数优先从环境变量读取，其次使用保守的资源友好默认值。

    Args:
        raw_data: PDF 原始字节内容。
        stream_name: 流名称，建议直接传文件名以保留扩展名。

    Returns:
        结构化字典（docling 导出格式或简化降级格式）。
    """
    # 强制禁用检查
    if _is_docling_disabled():
        Log.info(
            f"DAYU_DOCLING_DISABLE=true，使用 pypdf 降级: {stream_name}",
            module=_MODULE,
        )
        return convert_pdf_bytes_to_docling_payload_fallback(
            raw_data, stream_name=stream_name
        )

    # 低内存环境检测
    if _check_low_memory():
        Log.warning(
            f"低内存环境（<2GB 可用），自动降级为 pypdf: {stream_name}",
            module=_MODULE,
        )
        return convert_pdf_bytes_to_docling_payload_fallback(
            raw_data, stream_name=stream_name
        )

    # 尝试 Docling 正常路径
    do_ocr = _env_bool("DAYU_DOCLING_DO_OCR", False)
    do_table_structure = _env_bool("DAYU_DOCLING_DO_TABLE_STRUCTURE", True)
    table_mode = os.environ.get("DAYU_DOCLING_TABLE_MODE", "fast").strip().lower()
    if table_mode not in ("fast", "accurate"):
        table_mode = "fast"
    do_cell_matching = _env_bool("DAYU_DOCLING_DO_CELL_MATCHING", False)

    try:
        result = convert_pdf_bytes_with_docling(
            raw_data,
            stream_name=stream_name,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            table_mode=table_mode,
            do_cell_matching=do_cell_matching,
        )
        return result.document.export_to_dict()
    except DoclingRuntimeInitializationError:
        # Docling 不可用，降级
        Log.warning(
            f"Docling 初始化失败，降级为 pypdf: {stream_name}",
            module=_MODULE,
        )
        return convert_pdf_bytes_to_docling_payload_fallback(
            raw_data, stream_name=stream_name
        )
    except Exception as exc:
        # 其他异常也尝试降级
        Log.warning(
            f"Docling 转换失败 ({type(exc).__name__})，降级为 pypdf: {stream_name}",
            module=_MODULE,
        )
        return convert_pdf_bytes_to_docling_payload_fallback(
            raw_data, stream_name=stream_name
        )


def convert_pdf_bytes_to_docling_json_bytes(
    raw_data: bytes,
    stream_name: str,
) -> bytes:
    """将 PDF 字节流转换为序列化后的 JSON 字节内容。

    Args:
        raw_data: PDF 原始字节内容。
        stream_name: 流名称。

    Returns:
        已编码为 UTF-8 的 JSON 字节内容。
    """
    payload = convert_pdf_bytes_to_docling_payload(raw_data, stream_name=stream_name)
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
