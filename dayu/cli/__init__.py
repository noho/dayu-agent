"""统一 CLI 入口包。"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dayu.cli import main as main


def __getattr__(name: str) -> ModuleType:
    """按需暴露 CLI 主入口模块。

    Args:
        name: 属性名。

    Returns:
        `dayu.cli.main` 模块对象。

    Raises:
        AttributeError: 当属性名不受支持时抛出。
    """

    if name != "main":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return import_module("dayu.cli.main")


def run_main() -> int:
    """运行统一 CLI 主入口。

    Args:
        无。

    Returns:
        主入口退出码。

    Raises:
        无。
    """

    return import_module("dayu.cli.main").main()


__all__ = [
    "main",
    "run_main",
]
