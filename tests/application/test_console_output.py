"""终端输出容错与 CLI 入口边界测试。"""

from __future__ import annotations

import argparse
import importlib

import pytest

from dayu.console_output import configure_standard_streams_for_console_output
from dayu.render import render as render_module


class _FakeReconfigurableStream:
    """可记录 `reconfigure` 调用的假文本流。"""

    def __init__(self) -> None:
        """初始化记录状态。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self.observed_errors: list[str | None] = []

    def reconfigure(self, *, errors: str | None = None) -> None:
        """记录编码错误策略。

        Args:
            errors: 编码错误处理策略。

        Returns:
            无。

        Raises:
            无。
        """

        self.observed_errors.append(errors)


class _NonReconfigurableStream:
    """不支持 `reconfigure` 的假流对象。"""


@pytest.mark.unit
def test_configure_standard_streams_for_console_output_uses_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证标准流会统一收口到 `replace` 容错策略。

    Args:
        monkeypatch: pytest monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    stdout_stream = _FakeReconfigurableStream()
    stderr_stream = _FakeReconfigurableStream()
    monkeypatch.setattr("dayu.console_output.sys.stdout", stdout_stream)
    monkeypatch.setattr("dayu.console_output.sys.stderr", stderr_stream)

    configure_standard_streams_for_console_output()

    assert stdout_stream.observed_errors == ["replace"]
    assert stderr_stream.observed_errors == ["replace"]


@pytest.mark.unit
def test_configure_standard_streams_for_console_output_ignores_plain_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证不支持 `reconfigure` 的流不会导致配置阶段崩溃。

    Args:
        monkeypatch: pytest monkeypatch 工具。

    Returns:
        无。

    Raises:
        无。
    """

    monkeypatch.setattr(
        "dayu.console_output.sys.stdout",
        _NonReconfigurableStream(),
    )
    monkeypatch.setattr(
        "dayu.console_output.sys.stderr",
        _NonReconfigurableStream(),
    )

    configure_standard_streams_for_console_output()


@pytest.mark.unit
def test_cli_main_configures_console_output_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `dayu.cli.main` 会先配置标准流再解析参数。

    Args:
        monkeypatch: pytest monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    from dayu.cli import main as cli_main_module

    observed: list[str] = []
    monkeypatch.setattr(
        "dayu.cli.main.configure_standard_streams_for_console_output",
        lambda: observed.append("configured"),
    )
    monkeypatch.setattr(
        "dayu.cli.main.parse_arguments",
        lambda: argparse.Namespace(command=""),
    )

    assert cli_main_module.main() == 0
    assert observed == ["configured"]


@pytest.mark.unit
def test_wechat_main_configures_console_output_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `dayu.wechat.main` 会先配置标准流再解析参数。

    Args:
        monkeypatch: pytest monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    wechat_main_module = importlib.import_module("dayu.wechat.main")

    observed: list[str] = []
    monkeypatch.setattr(
        wechat_main_module,
        "configure_standard_streams_for_console_output",
        lambda: observed.append("configured"),
    )
    monkeypatch.setattr(
        wechat_main_module,
        "parse_arguments",
        lambda _argv=None: argparse.Namespace(command="login"),
    )
    monkeypatch.setattr(wechat_main_module, "setup_loglevel", lambda _args: None)
    monkeypatch.setattr(wechat_main_module, "_dispatch_command", lambda _args: 0)

    assert wechat_main_module.main() == 0
    assert observed == ["configured"]


@pytest.mark.unit
def test_render_main_configures_console_output_before_help(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证 `dayu-render --help` 会先配置标准流容错。

    Args:
        monkeypatch: pytest monkeypatch 工具。

    Returns:
        无。

    Raises:
        AssertionError: 断言失败时抛出。
    """

    observed: list[str] = []
    monkeypatch.setattr(
        "dayu.render.render.configure_standard_streams_for_console_output",
        lambda: observed.append("configured"),
    )
    monkeypatch.setattr("dayu.render.render._print_help", lambda: observed.append("help"))
    monkeypatch.setattr("dayu.render.render.sys.argv", ["dayu-render", "--help"])

    assert render_module.main() == 0
    assert observed == ["configured", "help"]
