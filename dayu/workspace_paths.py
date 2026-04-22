"""工作区内部隐藏状态路径真源。

本模块统一声明工作区下 `.dayu/` 目录的稳定布局，
供 Host、CLI、Fins、WeChat 共同复用，避免各入口各自拼接隐藏目录。
"""

from __future__ import annotations

from pathlib import Path


DAYU_INTERNAL_ROOT_RELATIVE_DIR = Path(".dayu")
HOST_STORE_RELATIVE_PATH = DAYU_INTERNAL_ROOT_RELATIVE_DIR / "host" / "dayu_host.db"
CONVERSATION_STORE_RELATIVE_DIR = DAYU_INTERNAL_ROOT_RELATIVE_DIR / "session"
INTERACTIVE_STATE_RELATIVE_DIR = DAYU_INTERNAL_ROOT_RELATIVE_DIR / "interactive"
CLI_CONVERSATION_REGISTRY_RELATIVE_DIR = DAYU_INTERNAL_ROOT_RELATIVE_DIR / "cli-conversations"
CLI_CONVERSATION_LOCKS_RELATIVE_DIR = CLI_CONVERSATION_REGISTRY_RELATIVE_DIR / "locks"
SEC_CACHE_RELATIVE_DIR = DAYU_INTERNAL_ROOT_RELATIVE_DIR / "sec_cache"
SEC_THROTTLE_RELATIVE_DIR = DAYU_INTERNAL_ROOT_RELATIVE_DIR / "sec_throttle"
DEFAULT_WECHAT_INSTANCE_LABEL = "default"
WECHAT_STATE_DIR_PREFIX = "wechat-"


def build_dayu_root_path(workspace_root: Path) -> Path:
    """构造 `.dayu` 根目录路径。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        工作区下 `.dayu` 根目录路径。

    Raises:
        无。
    """

    return _build_workspace_relative_path(workspace_root, DAYU_INTERNAL_ROOT_RELATIVE_DIR)


def build_host_store_default_path(workspace_root: Path) -> Path:
    """构造 Host SQLite 默认存储路径。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        默认 Host SQLite 数据库文件路径。

    Raises:
        无。
    """

    return _build_workspace_relative_path(workspace_root, HOST_STORE_RELATIVE_PATH)


def build_conversation_store_dir(workspace_root: Path) -> Path:
    """构造 conversation transcript 默认目录。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        Host conversation transcript 默认目录。

    Raises:
        无。
    """

    return _build_workspace_relative_path(workspace_root, CONVERSATION_STORE_RELATIVE_DIR)


def build_interactive_state_dir(workspace_root: Path) -> Path:
    """构造 interactive 本地状态默认目录。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        interactive 本地状态默认目录。

    Raises:
        无。
    """

    return _build_workspace_relative_path(workspace_root, INTERACTIVE_STATE_RELATIVE_DIR)


def build_cli_conversation_registry_dir(workspace_root: Path) -> Path:
    """构造 CLI label conversation registry 目录。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        CLI label conversation registry 目录路径。

    Raises:
        无。
    """

    return _build_workspace_relative_path(workspace_root, CLI_CONVERSATION_REGISTRY_RELATIVE_DIR)


def build_cli_conversation_label_record_path(workspace_root: Path, label: str) -> Path:
    """构造指定 label 的 CLI conversation record 文件路径。

    Args:
        workspace_root: 工作区根目录。
        label: conversation label。

    Returns:
        指定 label 对应的 record 文件路径。

    Raises:
        无。
    """

    return build_cli_conversation_registry_dir(workspace_root) / f"{label}.json"


def build_cli_conversation_label_lock_dir(workspace_root: Path, label: str) -> Path:
    """构造指定 label 的 CLI conversation 锁目录。

    Args:
        workspace_root: 工作区根目录。
        label: conversation label。

    Returns:
        指定 label 对应的锁目录路径。

    Raises:
        无。
    """

    return _build_workspace_relative_path(
        workspace_root,
        CLI_CONVERSATION_LOCKS_RELATIVE_DIR / label,
    )


def build_sec_cache_dir(workspace_root: Path) -> Path:
    """构造 SEC HTTP 缓存默认目录。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        SEC HTTP 缓存默认目录。

    Raises:
        无。
    """

    return _build_workspace_relative_path(workspace_root, SEC_CACHE_RELATIVE_DIR)


def build_sec_throttle_dir(workspace_root: Path) -> Path:
    """构造 SEC 全局限流状态默认目录。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        SEC 全局限流状态默认目录。

    Raises:
        无。
    """

    return _build_workspace_relative_path(workspace_root, SEC_THROTTLE_RELATIVE_DIR)


def build_wechat_state_dir(
    workspace_root: Path,
    *,
    label: str = DEFAULT_WECHAT_INSTANCE_LABEL,
) -> Path:
    """构造指定 WeChat 实例的状态目录。

    Args:
        workspace_root: 工作区根目录。
        label: WeChat 实例标签。

    Returns:
        WeChat 实例状态目录。

    Raises:
        无。
    """

    return _build_workspace_relative_path(
        workspace_root,
        DAYU_INTERNAL_ROOT_RELATIVE_DIR / f"{WECHAT_STATE_DIR_PREFIX}{label}",
    )


def list_wechat_state_dirs(workspace_root: Path) -> tuple[Path, ...]:
    """枚举工作区下已有的 WeChat 实例状态目录。

    Args:
        workspace_root: 工作区根目录。

    Returns:
        按目录名排序后的 WeChat 状态目录元组。

    Raises:
        无。
    """

    dayu_root = build_dayu_root_path(workspace_root)
    if not dayu_root.is_dir():
        return ()
    state_dirs: list[Path] = []
    for candidate in sorted(dayu_root.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir():
            continue
        if extract_wechat_instance_label(candidate.name) is None:
            continue
        state_dirs.append(candidate)
    return tuple(state_dirs)


def extract_wechat_instance_label(state_dir_name: str) -> str | None:
    """从 WeChat 状态目录名中提取实例标签。

    Args:
        state_dir_name: 状态目录名。

    Returns:
        实例标签；目录名不匹配时返回 `None`。

    Raises:
        无。
    """

    normalized_name = state_dir_name.strip()
    if not normalized_name.startswith(WECHAT_STATE_DIR_PREFIX):
        return None
    label = normalized_name.removeprefix(WECHAT_STATE_DIR_PREFIX)
    if not label:
        return None
    return label


def _build_workspace_relative_path(workspace_root: Path, relative_path: Path) -> Path:
    """把工作区相对路径映射为绝对路径。

    Args:
        workspace_root: 工作区根目录。
        relative_path: 相对于工作区的稳定路径。

    Returns:
        拼接后的绝对路径。

    Raises:
        无。
    """

    return workspace_root / relative_path


__all__ = [
    "DAYU_INTERNAL_ROOT_RELATIVE_DIR",
    "HOST_STORE_RELATIVE_PATH",
    "CONVERSATION_STORE_RELATIVE_DIR",
    "INTERACTIVE_STATE_RELATIVE_DIR",
    "CLI_CONVERSATION_REGISTRY_RELATIVE_DIR",
    "CLI_CONVERSATION_LOCKS_RELATIVE_DIR",
    "SEC_CACHE_RELATIVE_DIR",
    "SEC_THROTTLE_RELATIVE_DIR",
    "DEFAULT_WECHAT_INSTANCE_LABEL",
    "WECHAT_STATE_DIR_PREFIX",
    "build_dayu_root_path",
    "build_host_store_default_path",
    "build_conversation_store_dir",
    "build_interactive_state_dir",
    "build_cli_conversation_registry_dir",
    "build_cli_conversation_label_record_path",
    "build_cli_conversation_label_lock_dir",
    "build_sec_cache_dir",
    "build_sec_throttle_dir",
    "build_wechat_state_dir",
    "extract_wechat_instance_label",
    "list_wechat_state_dirs",
]
