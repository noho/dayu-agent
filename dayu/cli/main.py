#!/usr/bin/env python3
"""`dayu.cli` 轻量主入口。

模块职责：
- 只负责解析顶层参数并把控制权分发给具体命令模块。
- 不直接导入运行时装配、服务实现或业务命令逻辑。
"""

from __future__ import annotations

from dayu.cli.arg_parsing import parse_arguments

_FINS_COMMANDS = frozenset(
    {
        "download",
        "upload_filing",
        "upload_filings_from",
        "upload_material",
        "process",
        "process_filing",
        "process_material",
    }
)
_HOST_COMMANDS = frozenset({"sessions", "runs", "cancel", "host"})


def main() -> int:
    """解析顶层参数并分发到具体命令模块。

    Args:
        无。

    Returns:
        CLI 退出码。

    Raises:
        无。
    """

    args = parse_arguments()
    if args.command == "init":
        from dayu.cli.commands.init import run_init_command

        return run_init_command(args)
    if args.command in _FINS_COMMANDS:
        from dayu.cli.commands.fins import run_fins_command

        return run_fins_command(args)
    if args.command in _HOST_COMMANDS:
        from dayu.cli.commands.host import run_host_command

        return run_host_command(args)
    if args.command == "interactive":
        from dayu.cli.commands.interactive import run_interactive_command

        return run_interactive_command(args)
    if args.command == "prompt":
        from dayu.cli.commands.prompt import run_prompt_command

        return run_prompt_command(args)
    if args.command == "write":
        from dayu.cli.commands.write import run_write_command

        return run_write_command(args)
    return 0

