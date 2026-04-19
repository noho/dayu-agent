"""统一 CLI 入口包。"""

import dayu.cli.main as main


def run_main() -> int:
    """运行统一 CLI 主入口。

    Args:
        无。

    Returns:
        主入口退出码。

    Raises:
        无。
    """

    return main.main()


__all__ = [
    "main",
    "run_main",
]
