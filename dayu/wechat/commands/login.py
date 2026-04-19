"""WeChat `login` 子命令实现。"""

from __future__ import annotations

import argparse
import asyncio

from dayu.log import Log
from dayu.wechat.arg_parsing import MODULE, _resolve_command_context
from dayu.wechat.runtime import _create_login_daemon


async def _run_login_command(args: argparse.Namespace) -> int:
    """执行 `login` 子命令。"""

    context = _resolve_command_context(args)
    daemon = _create_login_daemon(args, context)
    try:
        await daemon.ensure_authenticated(force_relogin=bool(getattr(args, "relogin", False)))
    finally:
        await daemon.aclose()
    Log.info("WeChat 登录态已就绪", module=MODULE)
    return 0


def run_login_command(args: argparse.Namespace) -> int:
    """以同步入口执行 `login` 子命令。"""

    return asyncio.run(_run_login_command(args))


__all__ = ["_run_login_command", "run_login_command"]
