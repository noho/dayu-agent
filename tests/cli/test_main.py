"""``dayu.cli.main`` 顶层 KeyboardInterrupt 收口测试。"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from dayu.cli.main import main
from dayu.process_lifecycle.exit_codes import EXIT_CODE_SIGINT


@pytest.mark.unit
def test_main_returns_exit_code_sigint_on_keyboard_interrupt() -> None:
    """非交互式命令触发 KeyboardInterrupt 时，``main`` 应收口并返回退出码 130。

    信号 handler 在 ``run_full_shutdown_sequence`` 后 raise KeyboardInterrupt，
    非交互式命令（fins / download / write）的调用栈不捕获该异常，
    ``main`` 顶层统一收口避免 traceback 泄漏到 stderr。
    """

    def _fake_parse() -> argparse.Namespace:
        return argparse.Namespace(command="download", ticker="MCO")

    with (
        patch("dayu.cli.main.parse_arguments", side_effect=_fake_parse),
        patch("dayu.cli.main.configure_standard_streams_for_console_output"),
        patch(
            "dayu.cli.commands.fins.run_fins_command",
            side_effect=KeyboardInterrupt,
        ),
    ):
        result = main()

    assert result == EXIT_CODE_SIGINT
