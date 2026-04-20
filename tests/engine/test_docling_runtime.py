"""Docling 运行时装配策略测试。"""

from __future__ import annotations

import pytest

from dayu.docling_runtime import (
    DOCLING_DEVICE_ENV,
    build_docling_pdf_pipeline_options,
    resolve_docling_device_name,
)

pytestmark = pytest.mark.unit


def test_resolve_docling_device_name_defaults_to_cpu_on_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 macOS 默认固定走 CPU，避免 MPS OOM。

    Args:
        monkeypatch: pytest 环境变量与平台 monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    monkeypatch.delenv(DOCLING_DEVICE_ENV, raising=False)
    monkeypatch.setattr("dayu.docling_runtime.sys.platform", "darwin")

    assert resolve_docling_device_name() == "cpu"


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
    monkeypatch.setattr("dayu.docling_runtime.sys.platform", "linux")

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
    monkeypatch.setattr("dayu.docling_runtime.sys.platform", "darwin")

    pipeline_options = build_docling_pdf_pipeline_options()

    assert pipeline_options.accelerator_options is not None
    assert str(pipeline_options.accelerator_options.device).endswith("CPU")
    assert pipeline_options.do_ocr is True
    assert pipeline_options.do_table_structure is True
    assert pipeline_options.table_structure_options.do_cell_matching is True
