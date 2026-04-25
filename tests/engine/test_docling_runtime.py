"""Docling 运行时装配策略测试。"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from dayu.docling_runtime import (
    DOCLING_DEVICE_ENV,
    DoclingRuntimeInitializationError,
    build_docling_pdf_pipeline_options,
    convert_pdf_bytes_with_docling,
    resolve_docling_device_name,
    run_docling_pdf_conversion,
)

pytestmark = pytest.mark.unit


def test_resolve_docling_device_name_defaults_to_auto_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 macOS 默认走 auto。

    Args:
        monkeypatch: pytest 环境变量与平台 monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.delenv(DOCLING_DEVICE_ENV, raising=False)

    assert resolve_docling_device_name() == "auto"


def test_resolve_docling_device_name_defaults_to_auto_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证非 macOS 平台默认保留 Docling 的 auto 选择。

    Args:
        monkeypatch: pytest 环境变量与平台 monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.delenv(DOCLING_DEVICE_ENV, raising=False)

    assert resolve_docling_device_name() == "auto"


def test_resolve_docling_device_name_respects_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证显式环境变量会覆盖默认设备策略。

    Args:
        monkeypatch: pytest 环境变量 monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setenv(DOCLING_DEVICE_ENV, "mps")

    assert resolve_docling_device_name() == "mps"


def test_resolve_docling_device_name_rejects_invalid_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证非法设备配置会 fail fast，而不是静默回退。

    Args:
        monkeypatch: pytest 环境变量 monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setenv(DOCLING_DEVICE_ENV, "bad-device")

    with pytest.raises(RuntimeError, match=DOCLING_DEVICE_ENV):
        resolve_docling_device_name()


def test_build_docling_pdf_pipeline_options_uses_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 PDF pipeline 选项会写入解析后的设备策略。

    Args:
        monkeypatch: pytest 环境变量与平台 monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.delenv(DOCLING_DEVICE_ENV, raising=False)

    pipeline_options = build_docling_pdf_pipeline_options()

    assert pipeline_options.accelerator_options is not None
    assert str(pipeline_options.accelerator_options.device).endswith("AUTO")
    assert pipeline_options.do_ocr is True
    assert pipeline_options.do_table_structure is True
    assert pipeline_options.table_structure_options.do_cell_matching is True


class _FakeConverter:
    """携带 backend 与 device 名的假转换器。

    Attributes:
        backend_name: 当前转换器绑定的 backend 名。
        device_name: 当前转换器绑定的设备名。
    """

    def __init__(self, *, backend_name: str, device_name: str) -> None:
        """初始化假转换器。

        Args:
            backend_name: 当前转换器绑定的 backend 名。
            device_name: 当前转换器绑定的设备名。

        Returns:
            无。

        Raises:
            无。
        """

        self.backend_name = backend_name
        self.device_name = device_name


def _install_recording_builder(
    monkeypatch: pytest.MonkeyPatch,
    build_log: list[tuple[str, str]],
) -> None:
    """安装一个把每次构造调用记录到日志列表的假 builder。

    Args:
        monkeypatch: pytest monkeypatch fixture。
        build_log: 接收 (backend_name, device_name) 元组的列表。

    Returns:
        无。

    Raises:
        无。
    """

    def _build_converter(
        *,
        do_ocr: bool,
        do_table_structure: bool,
        table_mode: str,
        do_cell_matching: bool,
        device_name: str,
        backend_name: str,
    ) -> _FakeConverter:
        """记录构造参数并返回假转换器。

        Args:
            do_ocr: OCR 开关。
            do_table_structure: 表格结构开关。
            table_mode: 表格模式。
            do_cell_matching: 单元格匹配开关。
            device_name: 指定设备名。
            backend_name: 指定 backend 名。

        Returns:
            假转换器。

        Raises:
            无。
        """

        _ = (do_ocr, do_table_structure, table_mode, do_cell_matching)
        build_log.append((backend_name, device_name))
        return _FakeConverter(backend_name=backend_name, device_name=device_name)

    monkeypatch.setattr("dayu.docling_runtime.build_docling_pdf_converter", _build_converter)


def test_run_docling_pdf_conversion_recovers_via_pypdfium2_after_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 docling-parse 解析失败时会立即切到 pypdfium2 救回。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    build_log: list[tuple[str, str]] = []
    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "auto")
    _install_recording_builder(monkeypatch, build_log)

    def _convert(converter: object) -> str:
        """模拟 docling-parse 解析失败、pypdfium2 成功。

        Args:
            converter: Docling 转换器对象。

        Returns:
            成功时返回固定字符串。

        Raises:
            RuntimeError: 当 backend 为 docling-parse 时固定抛出。
        """

        fake_converter = cast(_FakeConverter, converter)
        if fake_converter.backend_name == "docling-parse":
            raise RuntimeError("Inconsistent number of pages: 238!=-1")
        return "ok"

    result = run_docling_pdf_conversion(_convert)

    assert result == "ok"
    assert build_log == [("docling-parse", "auto"), ("pypdfium2", "auto")]


def test_run_docling_pdf_conversion_walks_full_attempt_chain_when_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 auto 设备时尝试链按 (parse,auto)->(pypdfium2,auto)->(parse,cpu) 展开。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    build_log: list[tuple[str, str]] = []
    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "auto")
    _install_recording_builder(monkeypatch, build_log)

    def _convert(converter: object) -> str:
        """前两档失败，第三档 (parse, cpu) 成功。

        Args:
            converter: Docling 转换器对象。

        Returns:
            成功时返回固定字符串。

        Raises:
            RuntimeError: 在 cpu 之前固定抛出。
        """

        fake_converter = cast(_FakeConverter, converter)
        if fake_converter.device_name == "cpu":
            return "ok"
        if fake_converter.backend_name == "docling-parse":
            raise RuntimeError("auto failed")
        raise RuntimeError("pypdfium2 also failed")

    result = run_docling_pdf_conversion(_convert)

    assert result == "ok"
    assert build_log == [
        ("docling-parse", "auto"),
        ("pypdfium2", "auto"),
        ("docling-parse", "cpu"),
    ]


def test_run_docling_pdf_conversion_skips_cpu_when_device_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证显式非 auto 设备时尝试链不追加 (parse, cpu)。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    build_log: list[tuple[str, str]] = []
    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "mps")
    _install_recording_builder(monkeypatch, build_log)

    def _convert(converter: object) -> str:
        """两档全失败。

        Args:
            converter: Docling 转换器对象。

        Returns:
            无。

        Raises:
            RuntimeError: 固定抛出，区分 backend。
        """

        fake_converter = cast(_FakeConverter, converter)
        if fake_converter.backend_name == "docling-parse":
            raise RuntimeError("mps parse failed")
        raise RuntimeError("mps pypdfium2 failed")

    with pytest.raises(RuntimeError, match="mps pypdfium2 failed"):
        run_docling_pdf_conversion(_convert)

    assert build_log == [("docling-parse", "mps"), ("pypdfium2", "mps")]


def test_run_docling_pdf_conversion_keeps_first_failure_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证全链失败时，最后一次异常的 ``__cause__`` 指向首次失败。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    build_log: list[tuple[str, str]] = []
    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "auto")
    _install_recording_builder(monkeypatch, build_log)

    def _convert(converter: object) -> str:
        """按尝试链顺序抛出区分文案的异常。

        Args:
            converter: Docling 转换器对象。

        Returns:
            无。

        Raises:
            RuntimeError: 固定抛出。
        """

        fake_converter = cast(_FakeConverter, converter)
        marker = f"{fake_converter.backend_name}/{fake_converter.device_name}"
        raise RuntimeError(f"failure@{marker}")

    with pytest.raises(RuntimeError, match=r"failure@docling-parse/cpu") as exc_info:
        run_docling_pdf_conversion(_convert)

    cause = exc_info.value.__cause__
    assert isinstance(cause, RuntimeError)
    assert str(cause) == "failure@docling-parse/auto"
    assert build_log == [
        ("docling-parse", "auto"),
        ("pypdfium2", "auto"),
        ("docling-parse", "cpu"),
    ]


def test_run_docling_pdf_conversion_first_attempt_success_returns_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证首档成功时不会进入后续尝试。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    build_log: list[tuple[str, str]] = []
    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "auto")
    _install_recording_builder(monkeypatch, build_log)

    def _convert(converter: object) -> str:
        """首档直接返回。

        Args:
            converter: Docling 转换器对象。

        Returns:
            固定字符串。

        Raises:
            无。
        """

        _ = converter
        return "ok"

    result = run_docling_pdf_conversion(_convert)

    assert result == "ok"
    assert build_log == [("docling-parse", "auto")]


def test_run_docling_pdf_conversion_wraps_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证首轮初始化的未知异常会被包装为统一运行时错误。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "auto")

    def _raise_unexpected_init_error(**_: str | bool) -> object:
        """模拟初始化阶段的未知异常。

        Args:
            _: 占位参数。

        Returns:
            无。

        Raises:
            RuntimeError: 固定抛出。
        """

        raise RuntimeError("boom")

    monkeypatch.setattr(
        "dayu.docling_runtime.build_docling_pdf_converter",
        _raise_unexpected_init_error,
    )

    def _convert(converter: object) -> str:
        """占位转换回调；初始化失败时不应真正执行。

        Args:
            converter: Docling 转换器对象。

        Returns:
            固定字符串。

        Raises:
            无。
        """

        _ = converter
        return "never"

    with pytest.raises(
        DoclingRuntimeInitializationError,
        match=r"Docling 转换器初始化失败 \(attempt 1/3:.*backend=docling-parse.*device=auto\): boom",
    ) as exc_info:
        run_docling_pdf_conversion(_convert)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom"


def test_run_docling_pdf_conversion_wraps_followup_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证回退档初始化失败时仍抛 ``DoclingRuntimeInitializationError``。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "auto")

    def _build_converter(**kwargs: str | bool) -> _FakeConverter:
        """模拟首档成功、第二档（pypdfium2/auto）初始化失败。

        Args:
            kwargs: 转换器构造参数。

        Returns:
            首档时返回假转换器。

        Raises:
            RuntimeError: 第二档初始化阶段固定抛出。
        """

        backend_name = str(kwargs["backend_name"])
        device_name = str(kwargs["device_name"])
        if backend_name == "pypdfium2":
            raise RuntimeError("pypdfium2 init boom")
        return _FakeConverter(backend_name=backend_name, device_name=device_name)

    monkeypatch.setattr("dayu.docling_runtime.build_docling_pdf_converter", _build_converter)

    def _convert(converter: object) -> str:
        """模拟首档转换失败，触发第二档初始化。

        Args:
            converter: Docling 转换器对象。

        Returns:
            无。

        Raises:
            RuntimeError: 当 backend 为 docling-parse 时固定抛出。
        """

        fake_converter = cast(_FakeConverter, converter)
        if fake_converter.backend_name == "docling-parse":
            raise RuntimeError("auto failed")
        return "ok"

    with pytest.raises(
        DoclingRuntimeInitializationError,
        match=r"Docling 转换器初始化失败 \(attempt 2/3:.*backend=pypdfium2.*device=auto\): pypdfium2 init boom",
    ) as exc_info:
        run_docling_pdf_conversion(_convert)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "pypdfium2 init boom"


def test_convert_pdf_bytes_with_docling_uses_document_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 ``convert_pdf_bytes_with_docling`` 走 ``DocumentStream`` 路径。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from io import BytesIO

    from docling.datamodel.base_models import DocumentStream

    captured: dict[str, object] = {}

    class _StubConverter:
        """伪转换器，仅记录 ``convert`` 入参。"""

        def convert(self, source: object) -> str:
            """记录入参并返回固定哨兵值。

            Args:
                source: ``DocumentStream`` 输入。

            Returns:
                固定字符串哨兵。

            Raises:
                无。
            """

            captured["source"] = source
            return "ok-sentinel"

    def _fake_run(
        convert_operation: Callable[[object], str],
        *,
        do_ocr: bool,
        do_table_structure: bool,
        table_mode: str,
        do_cell_matching: bool,
    ) -> str:
        """直接调用 convert_operation 模拟成功路径。

        Args:
            convert_operation: 真实转换回调。
            do_ocr: OCR 开关。
            do_table_structure: 表格结构开关。
            table_mode: 表格模式。
            do_cell_matching: 单元格匹配开关。

        Returns:
            转换回调返回值。

        Raises:
            无。
        """

        captured["pipeline"] = (do_ocr, do_table_structure, table_mode, do_cell_matching)
        return convert_operation(_StubConverter())

    monkeypatch.setattr("dayu.docling_runtime.run_docling_pdf_conversion", _fake_run)

    result = cast(
        str,
        convert_pdf_bytes_with_docling(b"hello-bytes", stream_name="page.pdf"),
    )

    assert result == "ok-sentinel"
    captured_source = cast(DocumentStream, captured["source"])
    assert isinstance(captured_source, DocumentStream)
    assert captured_source.name == "page.pdf"
    assert isinstance(captured_source.stream, BytesIO)
    assert captured_source.stream.getvalue() == b"hello-bytes"
    assert captured["pipeline"] == (True, True, "accurate", True)
