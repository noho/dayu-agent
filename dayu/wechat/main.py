"""WeChat UI 启动入口。"""

from __future__ import annotations

import argparse

from dayu.console_output import configure_standard_streams_for_console_output
from dayu.log import Log
from dayu.wechat.arg_parsing import MODULE, parse_arguments, setup_loglevel


def _dispatch_command(args: argparse.Namespace) -> int:
    """分发 WeChat CLI 子命令。

    Args:
        args: argparse 解析结果。

    Returns:
        退出码。

    Raises:
        ValueError: 当主命令非法时抛出。
    """

    command = str(getattr(args, "command", "") or "").strip()
    if command == "login":
        # 登录命令会导入 daemon/runtime，仅在命中该命令时按需加载。
        from dayu.wechat.commands.login import run_login_command

        return run_login_command(args)
    if command == "run":
        # 运行命令会装配 Host/Fins/daemon 依赖，保持主入口冷启动轻量。
        from dayu.wechat.commands.run import run_run_command

        return run_run_command(args)
    if command == "service":
        # service 命令只在用户显式操作托管服务时导入实现模块。
        from dayu.wechat.commands.service import run_service_command

        return run_service_command(args)
    raise ValueError(f"未知命令: {command}")


def main(argv: list[str] | None = None) -> int:
    """WeChat CLI 主入口。

    Args:
        argv: 可选参数列表；为空时读取命令行。

    Returns:
        退出码。

    Raises:
        无。
    """

    # 先收口标准流容错，避免非 UTF-8 终端在打印中文帮助/错误文案时崩溃。
    configure_standard_streams_for_console_output()
    args = parse_arguments(argv)
    setup_loglevel(args)
    try:
        return _dispatch_command(args)
    except KeyboardInterrupt:
        Log.info("收到中断信号，WeChat daemon 正在退出", module=MODULE)
        return 130
    except Exception as exc:
        Log.error(f"WeChat 命令失败: {exc}", module=MODULE)
        return 1


__all__ = ["main"]
