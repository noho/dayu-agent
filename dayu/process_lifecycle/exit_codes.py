"""统一退出码常量与信号映射。"""

from __future__ import annotations


EXIT_CODE_SIGINT: int = 130
"""SIGINT / KeyboardInterrupt 的标准退出码（128 + 2）。"""

EXIT_CODE_SIGTERM: int = 0
"""SIGTERM 视为正常关停。"""

_SIGNAL_TO_EXIT_CODE: dict[str, int] = {
    "SIGINT": EXIT_CODE_SIGINT,
    "SIGTERM": EXIT_CODE_SIGTERM,
    "SIGHUP": EXIT_CODE_SIGINT,
}


def map_signal_to_exit_code(signal_name: str) -> int:
    """把信号名称映射成统一退出码。

    Args:
        signal_name: 信号名（如 ``"SIGINT"``）。

    Returns:
        统一退出码；未知信号默认归到 ``EXIT_CODE_SIGINT``。

    Raises:
        无。
    """

    return _SIGNAL_TO_EXIT_CODE.get(signal_name, EXIT_CODE_SIGINT)


__all__ = [
    "EXIT_CODE_SIGINT",
    "EXIT_CODE_SIGTERM",
    "map_signal_to_exit_code",
]
