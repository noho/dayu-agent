"""WeChat UI 包入口。"""

from dayu.wechat.main import main


def run_main() -> int:
    """执行 WeChat CLI 主入口。

    Args:
        无。

    Returns:
        主入口退出码。

    Raises:
        无。
    """

    return main()


__all__ = ["main", "run_main"]
