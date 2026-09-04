"""``dayu/docling_fallback.py`` 引擎降级链与 poppler 引擎单元测试。

覆盖：
- 三引擎降级链：pdf-inspector → poppler → pypdf 的优先级与兜底行为；
- ``_is_text_garbled`` 乱码启发式判定；
- poppler bbox XML 解析（文本行 + 逐行分列表格）；
- ``_convert_via_poppler`` 缺 pdftotext 时报错 / 解析出 docling JSON。

poppler 引擎测试通过 monkeypatch ``subprocess.run`` 注入假 bbox XML，
不真实调用系统 pdftotext，保持测试无关外部依赖。

测试内部禁用 ``Any``：用 ``JsonValue`` 递归别名表达 JSON 结构。
"""

from __future__ import annotations

import json
from typing import Union

import pytest

from dayu import docling_fallback as fb

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


class _FakeCompletedProcess:
    """subprocess.run 返回值 fake（text=True 时 stdout 为 str）。"""

    def __init__(self, stdout: str, *, returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


def _canned_bbox_xml() -> str:
    """一段含中文标题与 2 列表格的 ``pdftotext -bbox-layout`` 输出。"""
    return (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body><doc>'
        '<page width="595" height="842">'
        '<flow><block><line>'
        '<word xMin="42" yMin="10" xMax="100" yMax="20">财务摘要</word>'
        '<word xMin="200" yMin="10" xMax="250" yMax="20">一元</word>'
        "</line><line>"
        '<word xMin="42" yMin="30" xMax="80" yMax="40">收入</word>'
        '<word xMin="200" yMin="30" xMax="240" yMax="40">116,573</word>'
        '<word xMin="300" yMin="30" xMax="340" yMax="40">30,429</word>'
        "</line></block></flow></page></doc></body></html>"
    )


def test_dispatch_pdf_inspector_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pdf-inspector 成功时应直接用其结果，不降级。"""

    def _pi(raw: bytes, name: str) -> JsonObject:
        del raw, name
        return {"engine": "pdf_inspector"}

    def _pp(raw: bytes, name: str) -> JsonObject:
        del raw, name
        raise AssertionError("poppler 不应被调用")

    def _py(raw: bytes, name: str) -> JsonObject:
        del raw, name
        raise AssertionError("pypdf 不应被调用")

    monkeypatch.setattr(fb, "_convert_via_pdf_inspector", _pi)
    monkeypatch.setattr(fb, "_convert_via_poppler", _pp)
    monkeypatch.setattr(fb, "_convert_via_pypdf", _py)

    result = fb.convert_pdf_bytes_to_docling_payload(
        b"%PDF-1.4", stream_name="a.pdf"
    )
    assert result == {"engine": "pdf_inspector"}


def test_dispatch_poppler_on_pdfinspector_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pdf-inspector 失败时，poppler 接管（而非直接 pypdf）。"""

    def _pi(raw: bytes, name: str) -> JsonObject:
        del raw, name
        raise RuntimeError("pi empty")

    def _pp(raw: bytes, name: str) -> JsonObject:
        del raw, name
        return {"engine": "poppler"}

    def _py(raw: bytes, name: str) -> JsonObject:
        del raw, name
        raise AssertionError("pypdf 不应在 poppler 成功后调用")

    monkeypatch.setattr(fb, "_convert_via_pdf_inspector", _pi)
    monkeypatch.setattr(fb, "_convert_via_poppler", _pp)
    monkeypatch.setattr(fb, "_convert_via_pypdf", _py)

    result = fb.convert_pdf_bytes_to_docling_payload(
        b"%PDF-1.4", stream_name="a.pdf"
    )
    assert result == {"engine": "poppler"}


def test_dispatch_pypdf_last_resort(monkeypatch: pytest.MonkeyPatch) -> None:
    """pdf-inspector 与 poppler 都失败时，pypdf 兜底。"""

    def _pi(raw: bytes, name: str) -> JsonObject:
        del raw, name
        raise RuntimeError("pi empty")

    def _pp(raw: bytes, name: str) -> JsonObject:
        del raw, name
        raise RuntimeError("poppler missing")

    def _py(raw: bytes, name: str) -> JsonObject:
        del raw, name
        return {"engine": "pypdf"}

    monkeypatch.setattr(fb, "_convert_via_pdf_inspector", _pi)
    monkeypatch.setattr(fb, "_convert_via_poppler", _pp)
    monkeypatch.setattr(fb, "_convert_via_pypdf", _py)

    result = fb.convert_pdf_bytes_to_docling_payload(
        b"%PDF-1.4", stream_name="a.pdf"
    )
    assert result == {"engine": "pypdf"}


def test_is_text_garbled_matrix() -> None:
    """乱码判定：乱码中文字符 > 汉字*2 → True；正常中文/英文/短文 → False。"""
    # 乱码样本文本（CID 子集字库解码失败的典型字符）
    garbled = "ʕಂజѓ" * 60  # 亚美尼亚/孟加拉/泰文区段，无汉字
    assert len(garbled) >= 200
    assert fb._is_text_garbled([garbled]) is True

    normal_cn = "财务摘要收入毛利净利润" * 40
    assert len(normal_cn) >= 200
    assert fb._is_text_garbled([normal_cn]) is False

    normal_en = "This is an English annual report narrative text. " * 20
    assert len(normal_en) >= 200
    assert fb._is_text_garbled([normal_en]) is False

    # 文本过短不判定
    assert fb._is_text_garbled(["ab cd ef"]) is False


def test_parse_poppler_bbox_builds_text_and_table() -> None:
    """bbox XML 应解析出可读中文文本行与独立表格。"""
    text_blocks, tables = fb._parse_poppler_bbox(_canned_bbox_xml())

    assert any("财务摘要" in t for t in text_blocks)
    assert any("收入" in t for t in text_blocks)
    assert any("116,573" in t for t in text_blocks)
    assert len(tables) == 1
    # 表格应包含标签+数值单元格
    flat = [cell for row in tables[0] for cell in row]
    assert "收入" in flat and "116,573" in flat and "30,429" in flat


def test_parse_poppler_bbox_empty_raises() -> None:
    """无任何文本的 XML 应抛 RuntimeError（让更上层降级）。"""
    empty_xml = '<html><body><doc><page width="595" height="842"/>'
    empty_xml += "</doc></body></html>"
    with pytest.raises(RuntimeError):
        fb._parse_poppler_bbox(empty_xml)


def test_parse_poppler_bbox_bad_xml_raises() -> None:
    """非法 XML 应包装为 RuntimeError。"""
    with pytest.raises(RuntimeError, match="XML 解析失败"):
        fb._parse_poppler_bbox("<<not xml>>")


def test_convert_via_poppler_missing_binary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """缺 pdftotext 时 poppler 引擎应抛 RuntimeError。"""
    monkeypatch.setattr(fb.shutil, "which", lambda _name: None)
    with pytest.raises(RuntimeError, match="poppler-utils 未安装"):
        fb._convert_via_poppler(b"x", "a.pdf")


def test_convert_via_poppler_produces_docling_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """poppler 引擎在注入 bbox XML 后应产出含文本与表格的 docling JSON。"""

    def _which(name: str) -> str | None:
        del name
        return "/usr/bin/pdftotext"

    def _run(*_args: object, **_kwargs: object) -> _FakeCompletedProcess:
        return _FakeCompletedProcess(_canned_bbox_xml())

    monkeypatch.setattr(fb.shutil, "which", _which)
    monkeypatch.setattr(fb.subprocess, "run", _run)

    payload = fb._convert_via_poppler(b"%PDF-1.4 fake", "a.pdf")

    assert isinstance(payload, dict)
    texts = " ".join(item["text"] for item in payload.get("texts", []))
    assert "财务摘要" in texts
    tables = payload.get("tables", [])
    assert len(tables) >= 1
    # docling JSON 可被反序列化
    json.dumps(payload, ensure_ascii=False)