"""Docling 运行时装配策略测试。"""

from __future__ import annotations

from typing import cast

import pytest

from dayu.docling_runtime import (
    DOCLING_DEVICE_ENV,
    DoclingRuntimeInitializationError,
    build_docling_pdf_pipeline_options,
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


def test_run_docling_pdf_conversion_retries_on_cpu_after_auto_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 auto 设备转换失败后会回退到 CPU 重试一次。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    build_devices: list[str] = []

    class _FakeConverter:
        """携带设备名的假转换器。"""

        def __init__(self, device_name: str) -> None:
            """初始化假转换器。

            Args:
                device_name: 当前转换器绑定的设备名。

            Returns:
                无。

            Raises:
                无。
            """

            self.device_name = device_name

    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "auto")

    def _build_converter(
        *,
        do_ocr: bool,
        do_table_structure: bool,
        table_mode: str,
        do_cell_matching: bool,
        device_name: str,
    ) -> _FakeConverter:
        """记录设备名并返回假转换器。

        Args:
            do_ocr: OCR 开关。
            do_table_structure: 表格结构开关。
            table_mode: 表格模式。
            do_cell_matching: 单元格匹配开关。
            device_name: 指定设备名。

        Returns:
            假转换器。

        Raises:
            无。
        """

        _ = (do_ocr, do_table_structure, table_mode, do_cell_matching)
        build_devices.append(device_name)
        return _FakeConverter(device_name)

    monkeypatch.setattr("dayu.docling_runtime.build_docling_pdf_converter", _build_converter)

    def _convert(converter: object) -> str:
        """按设备名模拟转换行为。

        Args:
            converter: Docling 转换器对象。

        Returns:
            成功时返回固定字符串。

        Raises:
            RuntimeError: `auto` 阶段固定抛错。
        """

        fake_converter = cast(_FakeConverter, converter)
        if fake_converter.device_name == "auto":
            raise RuntimeError("auto failed")
        return "ok"

    result = run_docling_pdf_conversion(_convert)

    assert result == "ok"
    assert build_devices == ["auto", "cpu"]


def test_run_docling_pdf_conversion_does_not_retry_non_auto_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证显式非 auto 设备失败时不会偷偷回退 CPU。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    build_devices: list[str] = []

    class _FakeConverter:
        """携带设备名的假转换器。"""

        def __init__(self, device_name: str) -> None:
            """初始化假转换器。

            Args:
                device_name: 当前转换器绑定的设备名。

            Returns:
                无。

            Raises:
                无。
            """

            self.device_name = device_name

    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "mps")

    def _build_converter(
        *,
        do_ocr: bool,
        do_table_structure: bool,
        table_mode: str,
        do_cell_matching: bool,
        device_name: str,
    ) -> _FakeConverter:
        """记录设备名并返回假转换器。

        Args:
            do_ocr: OCR 开关。
            do_table_structure: 表格结构开关。
            table_mode: 表格模式。
            do_cell_matching: 单元格匹配开关。
            device_name: 指定设备名。

        Returns:
            假转换器。

        Raises:
            无。
        """

        _ = (do_ocr, do_table_structure, table_mode, do_cell_matching)
        build_devices.append(device_name)
        return _FakeConverter(device_name)

    monkeypatch.setattr("dayu.docling_runtime.build_docling_pdf_converter", _build_converter)

    def _convert(converter: object) -> str:
        """固定抛出转换失败。

        Args:
            converter: Docling 转换器对象。

        Returns:
            无。

        Raises:
            RuntimeError: 固定抛出。
        """

        _ = cast(_FakeConverter, converter)
        raise RuntimeError("mps failed")

    with pytest.raises(RuntimeError, match="mps failed"):
        run_docling_pdf_conversion(_convert)

    assert build_devices == ["mps"]


def test_run_docling_pdf_conversion_keeps_auto_failure_as_retry_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 CPU 重试仍失败时，会保留首次 auto 失败作为异常因果链。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    class _FakeConverter:
        """携带设备名的假转换器。"""

        def __init__(self, device_name: str) -> None:
            """初始化假转换器。

            Args:
                device_name: 当前转换器绑定的设备名。

            Returns:
                无。

            Raises:
                无。
            """

            self.device_name = device_name

    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "auto")
    monkeypatch.setattr(
        "dayu.docling_runtime.build_docling_pdf_converter",
        lambda **kwargs: _FakeConverter(str(kwargs["device_name"])),
    )

    def _convert(converter: object) -> str:
        """按设备名模拟 auto 与 CPU 都失败。

        Args:
            converter: Docling 转换器对象。

        Returns:
            无。

        Raises:
            RuntimeError: 固定抛出，且区分 auto 与 CPU。
        """

        fake_converter = cast(_FakeConverter, converter)
        if fake_converter.device_name == "auto":
            raise RuntimeError("auto failed")
        raise RuntimeError("cpu failed")

    with pytest.raises(RuntimeError, match="cpu failed") as exc_info:
        run_docling_pdf_conversion(_convert)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "auto failed"


def test_run_docling_pdf_conversion_wraps_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证首轮转换器初始化的未知异常会被包装成统一运行时错误。

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

    with pytest.raises(DoclingRuntimeInitializationError, match="Docling 转换器初始化失败: boom") as exc_info:
        run_docling_pdf_conversion(_convert)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "boom"


def test_run_docling_pdf_conversion_wraps_cpu_retry_initialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 CPU 回退初始化失败时会抛出统一运行时错误。

    Args:
        monkeypatch: pytest monkeypatch fixture。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    class _FakeConverter:
        """携带设备名的假转换器。"""

        def __init__(self, device_name: str) -> None:
            """初始化假转换器。

            Args:
                device_name: 当前转换器绑定的设备名。

            Returns:
                无。

            Raises:
                无。
            """

            self.device_name = device_name

    monkeypatch.setattr("dayu.docling_runtime.resolve_docling_device_name", lambda: "auto")

    def _build_converter(**kwargs: str | bool) -> _FakeConverter:
        """模拟首轮成功、CPU 回退初始化失败。

        Args:
            kwargs: 转换器构造参数。

        Returns:
            首轮转换时返回假转换器。

        Raises:
            RuntimeError: CPU 回退初始化阶段固定抛出。
        """

        device_name = str(kwargs["device_name"])
        if device_name == "cpu":
            raise RuntimeError("cpu init boom")
        return _FakeConverter(device_name)

    monkeypatch.setattr("dayu.docling_runtime.build_docling_pdf_converter", _build_converter)

    def _convert(converter: object) -> str:
        """模拟首轮 auto 转换失败，触发 CPU 回退。

        Args:
            converter: Docling 转换器对象。

        Returns:
            无。

        Raises:
            RuntimeError: 当设备为 auto 时固定抛出。
        """

        fake_converter = cast(_FakeConverter, converter)
        if fake_converter.device_name == "auto":
            raise RuntimeError("auto failed")
        return "ok"

    with pytest.raises(
        DoclingRuntimeInitializationError,
        match="Docling CPU 回退初始化失败: cpu init boom",
    ) as exc_info:
        run_docling_pdf_conversion(_convert)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "cpu init boom"
