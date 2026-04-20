"""终端标准流容错辅助。

本模块负责为 CLI 入口统一配置标准输出/错误流的编码错误策略，
避免帮助信息或日志中包含 Unicode 文本时，在非 UTF-8 终端里直接抛出
``UnicodeEncodeError``。
"""

from __future__ import annotations

import sys
from typing import Protocol, cast


class _ReconfigurableTextStreamProtocol(Protocol):
    """支持重新配置编码错误策略的最小文本流协议。"""

    def reconfigure(self, *, errors: str | None = None) -> None:
        """重新配置文本流的错误处理策略。

        Args:
            errors: 文本编码错误处理策略。

        Returns:
            无。

        Raises:
            OSError: 底层流不支持重配置时抛出。
            ValueError: 参数不合法时抛出。
        """


def _configure_stream_encoding_errors(
    stream: _ReconfigurableTextStreamProtocol | None,
) -> None:
    """为单个文本流设置安全的编码错误处理策略。

    Args:
        stream: 待配置的文本流。

    Returns:
        无。

    Raises:
        无。内部会吞掉“不支持重配置”类异常。
    """

    if stream is None:
        return
    try:
        stream.reconfigure(errors="replace")
    except (AttributeError, OSError, TypeError, ValueError):
        return


def configure_standard_streams_for_console_output() -> None:
    """为标准输出与标准错误启用编码容错。

    该函数不会强行改变当前终端编码，只会把编码错误策略收敛到
    ``replace``，让不支持目标字符集的终端至少能完成输出，而不是在
    打印帮助信息或日志时直接崩溃。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """

    _configure_stream_encoding_errors(
        cast(_ReconfigurableTextStreamProtocol | None, sys.stdout)
    )
    _configure_stream_encoding_errors(
        cast(_ReconfigurableTextStreamProtocol | None, sys.stderr)
    )
