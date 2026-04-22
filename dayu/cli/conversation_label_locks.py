"""CLI labeled conversation 独占锁。

本模块负责 `prompt --label`、`interactive --label` 与
`conv remove --label` 共用的 label 独占语义。

同一个 label 在任意时刻只能被一个 CLI 进程占用：
- `prompt --label` 在本轮输出完成前持有
- `interactive --label` 在整个 REPL 生命周期内持有
- `conv remove --label` 需要先成功获取独占锁，才能释放该 label
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from dayu.cli.conversation_labels import validate_conversation_label
from dayu.state_dir_lock import StateDirSingleInstanceLock
from dayu.workspace_paths import build_cli_conversation_label_lock_dir


_LABEL_LOCK_FILE_NAME = ".label.lock"


def build_label_busy_message(label: str) -> str:
    """构造 label 被占用时的稳定提示文案。

    Args:
        label: 已规范化的 conversation label。

    Returns:
        面向 CLI 用户的占用提示文案。

    Raises:
        无。
    """

    return f"label 正在使用中: {label}。请等待当前对话结束后重试，或使用新的 --label"


class ConversationLabelLease:
    """基于工作区文件锁的 labeled conversation 独占租约。"""

    def __init__(self, workspace_root: Path, label: str) -> None:
        """初始化 label 独占租约。

        Args:
            workspace_root: 工作区根目录。
            label: 待占用的 conversation label。

        Returns:
            无。

        Raises:
            ValueError: 当 label 非法时抛出。
        """

        normalized_label = validate_conversation_label(label)
        self._label = normalized_label
        self._lock = StateDirSingleInstanceLock(
            state_dir=build_cli_conversation_label_lock_dir(workspace_root, normalized_label),
            lock_file_name=_LABEL_LOCK_FILE_NAME,
            lock_name=f"label 对话占用锁({normalized_label})",
        )

    @property
    def label(self) -> str:
        """返回当前租约对应的 label。"""

        return self._label

    def acquire(self) -> None:
        """获取当前 label 的独占租约。

        Args:
            无。

        Returns:
            无。

        Raises:
            RuntimeError: 当 label 已被其他进程占用时抛出。
        """

        try:
            self._lock.acquire()
        except RuntimeError as exc:
            raise RuntimeError(build_label_busy_message(self._label)) from exc

    def release(self) -> None:
        """释放当前 label 的独占租约。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        self._lock.release()

    def __enter__(self) -> "ConversationLabelLease":
        """进入租约上下文并获取独占锁。"""

        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """退出租约上下文并释放独占锁。

        Args:
            exc_type: 异常类型。
            exc: 异常对象。
            traceback: traceback 对象。

        Returns:
            无。

        Raises:
            无。
        """

        del exc_type
        del exc
        del traceback
        self.release()


__all__ = [
    "ConversationLabelLease",
    "build_label_busy_message",
]
