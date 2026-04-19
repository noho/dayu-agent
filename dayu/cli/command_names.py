"""CLI 顶层命令名常量。

该模块只承载轻量命令名集合，供 `arg_parsing.py`、`main.py` 与
具体命令模块共享；不得在此处引入任何运行时装配或业务逻辑依赖。
"""

from __future__ import annotations

FINS_COMMANDS = frozenset(
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

HOST_COMMANDS = frozenset({"sessions", "runs", "cancel", "host"})

__all__ = ["FINS_COMMANDS", "HOST_COMMANDS"]
