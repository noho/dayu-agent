#!/usr/bin/env python3
"""`dayu.cli` 轻量主入口。

模块职责：
- 只负责解析顶层参数并把控制权分发给具体命令模块。
- 不直接导入运行时装配、服务实现或业务命令逻辑。
"""

from __future__ import annotations

from dayu.cli.command_names import FINS_COMMANDS, HOST_COMMANDS
from dayu.cli.arg_parsing import parse_arguments
from dayu.console_output import configure_standard_streams_for_console_output
from dayu.process_lifecycle.exit_codes import EXIT_CODE_SIGINT


def main() -> int:
    """解析顶层参数并分发到具体命令模块。

    Args:
        无。

    Returns:
        CLI 退出码。

    Raises:
        无。
    """

    # Windows 等非 UTF-8 终端下，中文帮助信息可能因输出编码不支持而崩溃。
    # 入口层先收口标准流容错，确保 `--help` 至少不会直接抛出编码异常。
    configure_standard_streams_for_console_output()
    args = parse_arguments()
    try:
        if args.command == "init":
            # `init` 必须保持冷启动轻量，只在命中该命令时再导入实现模块。
            from dayu.cli.commands.init import run_init_command

            return run_init_command(args)
        if args.command in FINS_COMMANDS:
            # 财报命令会装配 fins/runtime 依赖，避免在 `--help` 阶段抢先导入。
            from dayu.cli.commands.fins import run_fins_command

            return run_fins_command(args)
        if args.command in HOST_COMMANDS:
            # 宿主管理命令需要 Host 运行时，按需导入保持主入口轻量。
            from dayu.cli.commands.host import run_host_command

            return run_host_command(args)
        if args.command == "interactive":
            # interactive 会拉起完整 CLI 运行时，延迟到命中命令时导入。
            from dayu.cli.commands.interactive import run_interactive_command

            return run_interactive_command(args)
        if args.command == "prompt":
            # prompt 会构建 Service/Host 依赖，避免在帮助路径提前导入。
            from dayu.cli.commands.prompt import run_prompt_command

            return run_prompt_command(args)
        if args.command == "conv":
            # conv 需要读取 CLI label registry 与 Host 管理面，按需导入保持主入口轻量。
            from dayu.cli.commands.conv import run_conv_command

            return run_conv_command(args)
        if args.command == "write":
            # write 依赖写作 pipeline 与 Host 运行时，仅在需要时导入。
            from dayu.cli.commands.write import run_write_command

            return run_write_command(args)
        return 0
    except KeyboardInterrupt:
        # 信号 handler（sync_signals._handler）在收到 SIGINT 时先执行
        # `coordinator.run_full_shutdown_sequence`（cooperative cancel + 强收敛），
        # 再 raise KeyboardInterrupt 打断阻塞调用（asyncio.run /
        # concurrent.futures.wait）。
        # 非交互式命令（fins / download / write）的调用栈不捕获
        # KeyboardInterrupt，这里统一收口，避免 traceback 泄漏到 stderr。
        # 交互式命令（interactive / prompt）已在各自 REPL 循环内捕获
        # KeyboardInterrupt，不会逃逸到这里。
        return EXIT_CODE_SIGINT
