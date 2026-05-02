"""``dayu/fins/docling_export.py`` 单元测试。

仅验证公共出口的契约：
- ``convert_pdf_bytes_to_docling_payload`` 返回 SDK 字典；
- ``convert_pdf_bytes_to_docling_json_bytes`` 序列化为 UTF-8 JSON 字节，且与
  payload 内容一致；
- ``convert_pdf_bytes_to_docling_json_bytes`` 满足 ``Callable[[bytes, str], bytes]``
  协议（位置参数调用必须能跑通）；
- ``stream_name`` 透传到底层 docling-runtime；
- 底层异常按文档分类透出。

不真实跑 docling；通过 monkeypatch 注入 fake 实现。

测试内部禁用 ``Any``：用 ``JsonValue`` 递归别名表达 fake payload 的 JSON 结构，
``_CapturedKwargs`` dataclass 表达底层调用入参。docling-runtime 真源 ``export_to_dict``
返回 ``dict[str, Any]`` 是第三方 SDK 边界，仅在 ``dayu/fins/docling_export.py`` 内
允许出现，不外泄到测试。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Union

import pytest

from dayu import docling_runtime
from dayu.docling_runtime import DoclingRuntimeInitializationError
from dayu.fins import docling_export

# 测试用 JSON 递归类型别名；fake payload 仅使用合法 JSON 结构，不引入 ``Any``。
JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    list["JsonValue"],
    dict[str, "JsonValue"],
]
JsonObject = dict[str, JsonValue]


@dataclass
class _CapturedKwargs:
    """记录底层 ``convert_pdf_bytes_with_docling`` 入参的强类型容器。"""

    raw: bytes = b""
    stream_name: str = ""
    do_ocr: bool = False
    do_table_structure: bool = False
    table_mode: str = ""
    do_cell_matching: bool = False
    called: bool = field(default=False)


class _FakeDocument:
    """Docling document fake，只暴露 ``export_to_dict``。"""

    def __init__(self, payload: JsonObject) -> None:
        self._payload = payload

    def export_to_dict(self) -> JsonObject:
        return self._payload


class _FakeConversionResult:
    """``convert_pdf_bytes_with_docling`` 返回值 fake。"""

    def __init__(self, payload: JsonObject) -> None:
        self.document = _FakeDocument(payload)


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: JsonObject,
    captured: _CapturedKwargs,
) -> None:
    """安装一个 fake ``convert_pdf_bytes_with_docling``，记录入参。"""

    def _fake(
        raw_bytes: bytes,
        *,
        stream_name: str,
        do_ocr: bool,
        do_table_structure: bool,
        table_mode: str,
        do_cell_matching: bool,
    ) -> _FakeConversionResult:
        captured.raw = raw_bytes
        captured.stream_name = stream_name
        captured.do_ocr = do_ocr
        captured.do_table_structure = do_table_structure
        captured.table_mode = table_mode
        captured.do_cell_matching = do_cell_matching
        captured.called = True
        return _FakeConversionResult(payload)

    monkeypatch.setattr(docling_export, "convert_pdf_bytes_with_docling", _fake)


def test_convert_pdf_bytes_to_docling_payload_returns_sdk_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """payload 出口应直接返回 ``export_to_dict`` 的结果，并按强参数策略调用底层。"""

    payload: JsonObject = {"name": "doc", "pages": [{"index": 0}]}
    captured = _CapturedKwargs()
    _install_fake_runtime(monkeypatch, payload=payload, captured=captured)

    result = docling_export.convert_pdf_bytes_to_docling_payload(
        b"%PDF-1.4 fake bytes",
        stream_name="report.pdf",
    )

    assert result is payload
    assert captured.called
    assert captured.raw == b"%PDF-1.4 fake bytes"
    assert captured.stream_name == "report.pdf"
    assert captured.do_ocr is True
    assert captured.do_table_structure is True
    assert captured.table_mode == "accurate"
    assert captured.do_cell_matching is True


def test_convert_pdf_bytes_to_docling_json_bytes_is_utf8_json_of_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """json_bytes 出口应把 payload 序列化为 UTF-8 JSON 字节。"""

    payload: JsonObject = {
        "title": "贵州茅台 2024 年年度报告",
        "tables": [{"name": "利润表"}],
    }
    captured = _CapturedKwargs()
    _install_fake_runtime(monkeypatch, payload=payload, captured=captured)

    raw_json = docling_export.convert_pdf_bytes_to_docling_json_bytes(
        b"%PDF-1.4",
        stream_name="茅台.pdf",
    )

    assert isinstance(raw_json, bytes)
    decoded = json.loads(raw_json.decode("utf-8"))
    assert decoded == payload
    # ensure_ascii=False -> 中文字符直接保留
    assert "贵州茅台" in raw_json.decode("utf-8")
    assert captured.stream_name == "茅台.pdf"


def test_convert_pdf_bytes_to_docling_json_bytes_supports_positional_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """json_bytes 出口必须满足 ``Callable[[bytes, str], bytes]`` 协议。

    下游 download workflow 会按位置参数注入此函数，因此必须显式覆盖位置参数
    调用：去掉 keyword-only 标记后，``f(raw, name)`` 必须可用。
    """

    payload: JsonObject = {"title": "positional"}
    captured = _CapturedKwargs()
    _install_fake_runtime(monkeypatch, payload=payload, captured=captured)

    # 通过 ``Callable[[bytes, str], bytes]`` 别名取出函数后再调用，
    # 既验证签名兼容也避免类型检查器把 ``stream_name`` 推断为 keyword-only。
    fn: docling_export.PdfToDoclingJsonBytes = (
        docling_export.convert_pdf_bytes_to_docling_json_bytes
    )
    raw_json = fn(b"%PDF-1.4", "positional.pdf")

    assert isinstance(raw_json, bytes)
    decoded = json.loads(raw_json.decode("utf-8"))
    assert decoded == payload
    assert captured.stream_name == "positional.pdf"


def test_convert_pdf_bytes_to_docling_payload_propagates_initialization_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``DoclingRuntimeInitializationError`` 应原样向上抛出。"""

    def _raise(
        raw_bytes: bytes,
        *,
        stream_name: str,
        do_ocr: bool,
        do_table_structure: bool,
        table_mode: str,
        do_cell_matching: bool,
    ) -> _FakeConversionResult:
        del raw_bytes, stream_name, do_ocr, do_table_structure, table_mode, do_cell_matching
        raise DoclingRuntimeInitializationError("docling missing")

    monkeypatch.setattr(docling_export, "convert_pdf_bytes_with_docling", _raise)

    with pytest.raises(DoclingRuntimeInitializationError):
        docling_export.convert_pdf_bytes_to_docling_payload(
            b"%PDF-1.4",
            stream_name="x.pdf",
        )


def test_convert_pdf_bytes_to_docling_payload_wraps_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """非初始化异常应被包成 ``RuntimeError`` 并保留 ``__cause__``。"""

    def _raise(
        raw_bytes: bytes,
        *,
        stream_name: str,
        do_ocr: bool,
        do_table_structure: bool,
        table_mode: str,
        do_cell_matching: bool,
    ) -> _FakeConversionResult:
        del raw_bytes, stream_name, do_ocr, do_table_structure, table_mode, do_cell_matching
        raise ValueError("bad pdf")

    monkeypatch.setattr(docling_export, "convert_pdf_bytes_with_docling", _raise)

    with pytest.raises(RuntimeError) as exc_info:
        docling_export.convert_pdf_bytes_to_docling_payload(
            b"not a pdf",
            stream_name="bad.pdf",
        )

    assert "bad.pdf" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_convert_pdf_bytes_to_docling_json_bytes_propagates_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """json_bytes 出口在底层失败时也应抛 ``RuntimeError``。"""

    def _raise(
        raw_bytes: bytes,
        *,
        stream_name: str,
        do_ocr: bool,
        do_table_structure: bool,
        table_mode: str,
        do_cell_matching: bool,
    ) -> _FakeConversionResult:
        del raw_bytes, stream_name, do_ocr, do_table_structure, table_mode, do_cell_matching
        raise ValueError("oops")

    monkeypatch.setattr(docling_export, "convert_pdf_bytes_with_docling", _raise)

    with pytest.raises(RuntimeError):
        docling_export.convert_pdf_bytes_to_docling_json_bytes(
            b"x",
            "y.pdf",
        )


def test_docling_runtime_dependency_uses_real_runtime_symbol() -> None:
    """``docling_export`` 必须直接复用 ``dayu.docling_runtime`` 的真源类型。

    通过对比模块内被 monkeypatch 替换前的原始绑定，确保 ``except`` 分支拦截的
    是 docling-runtime 真源 ``DoclingRuntimeInitializationError``，避免被静默
    替换为同名子类导致拦截失效。
    """

    # 此处不读 __all__；用 getattr 显式从模块内取出现有绑定，确保它就是
    # docling_runtime 上的同一对象。
    bound = getattr(docling_export, "DoclingRuntimeInitializationError")
    assert bound is docling_runtime.DoclingRuntimeInitializationError
