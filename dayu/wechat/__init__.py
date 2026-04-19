"""WeChat UI 包入口。"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dayu.wechat.daemon import WeChatDaemon, WeChatDaemonConfig, WeChatReply, WeChatReplyBuilder
    from dayu.wechat.ilink_client import IlinkApiClient, IlinkApiError
    from dayu.wechat.main import main
    from dayu.wechat.state_store import FileWeChatStateStore, WeChatDaemonState


_LAZY_EXPORTS = {
    "FileWeChatStateStore": ("dayu.wechat.state_store", "FileWeChatStateStore"),
    "IlinkApiClient": ("dayu.wechat.ilink_client", "IlinkApiClient"),
    "IlinkApiError": ("dayu.wechat.ilink_client", "IlinkApiError"),
    "WeChatDaemon": ("dayu.wechat.daemon", "WeChatDaemon"),
    "WeChatDaemonConfig": ("dayu.wechat.daemon", "WeChatDaemonConfig"),
    "WeChatDaemonState": ("dayu.wechat.state_store", "WeChatDaemonState"),
    "WeChatReply": ("dayu.wechat.daemon", "WeChatReply"),
    "WeChatReplyBuilder": ("dayu.wechat.daemon", "WeChatReplyBuilder"),
    "main": ("dayu.wechat.main", "main"),
}


def __getattr__(name: str) -> object:
    """按需暴露 WeChat 包的稳定导出符号。

    Args:
        name: 导出符号名。

    Returns:
        对应的运行时对象。

    Raises:
        AttributeError: 当符号名不存在时抛出。
    """

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    return getattr(import_module(module_name), attr_name)

__all__ = [
    "FileWeChatStateStore",
    "IlinkApiClient",
    "IlinkApiError",
    "WeChatDaemon",
    "WeChatDaemonConfig",
    "WeChatDaemonState",
    "WeChatReply",
    "WeChatReplyBuilder",
    "main",
]
