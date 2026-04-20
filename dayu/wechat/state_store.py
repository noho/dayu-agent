"""WeChat UI 状态持久化。

该模块负责持久化以下运行时状态：
- iLink 登录态（bot_token / base_url）
- get_updates_buf 游标
- typing_ticket 缓存

注意：微信会话对应的 Dayu `session_id` 不再持久化映射表。
当前实现改为基于 `chat_key` 通过 `build_wechat_session_id(...)`
确定性生成稳定 `session_id`，再由 `SessionRegistry.ensure_session(...)`
负责幂等登记与生命周期管理。
- typing_ticket 缓存
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dayu.wechat.ilink_client import DEFAULT_ILINK_BASE_URL


def build_wechat_session_id(chat_key: str) -> str:
    """为微信会话生成稳定的 Dayu session_id。

    Args:
        chat_key: 微信侧会话键；当前首版使用 `group_id or from_user_id`。

    Returns:
        稳定的 session_id。

    Raises:
        ValueError: 当 `chat_key` 为空时抛出。
    """

    normalized = str(chat_key or "").strip()
    if not normalized:
        raise ValueError("chat_key 不能为空")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"wechat_{digest}"


def build_wechat_runtime_identity(state_dir: Path) -> str:
    """根据状态目录生成稳定的 WeChat runtime identity。

    Args:
        state_dir: 当前 daemon 使用的状态目录。

    Returns:
        基于状态目录的稳定 runtime identity。

    Raises:
        ValueError: 当 `state_dir` 为空时抛出。
    """

    normalized = str(Path(state_dir).expanduser().resolve()).strip()
    if not normalized:
        raise ValueError("state_dir 不能为空")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"wechat_runtime_{digest}"


def _guess_binary_extension(data: bytes) -> str:
    """根据字节内容猜测二维码文件扩展名。

    Args:
        data: 文件字节。

    Returns:
        扩展名，包含前导点。

    Raises:
        无。
    """

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return ".gif"
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if data.lstrip().startswith(b"<svg"):
        return ".svg"
    return ".bin"


def _extract_base64_payload(content: str) -> tuple[str | None, str]:
    """从 data URI 或裸 base64 字符串中提取 payload。

    Args:
        content: 原始二维码内容。

    Returns:
        `(mime_type, payload)` 元组；mime_type 可能为空。

    Raises:
        无。
    """

    stripped = str(content or "").strip()
    if stripped.startswith("data:") and ";base64," in stripped:
        header, payload = stripped.split(",", 1)
        mime_type = header[5:].split(";", 1)[0].strip() or None
        return mime_type, payload
    return None, stripped


def _looks_like_http_url(value: str | None) -> bool:
    """判断字符串是否看起来是 HTTP/HTTPS URL。

    Args:
        value: 待判断字符串。

    Returns:
        `True` 表示像 URL，否则返回 `False`。

    Raises:
        无。
    """

    normalized = str(value or "").strip().lower()
    return normalized.startswith("http://") or normalized.startswith("https://")


def _write_text_atomic(target: Path, content: str) -> None:
    """以原子替换方式写入 UTF-8 文本文件。

    Args:
        target: 目标文件路径。
        content: 待写入文本。

    Returns:
        无。

    Raises:
        OSError: 当文件写入或替换失败时抛出。
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_path_str = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temp_path = Path(temp_path_str)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temp_file:
            temp_file.write(content)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        os.replace(temp_path, target)
    except Exception:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise


@dataclass
class WeChatDaemonState:
    """WeChat daemon 持久化状态。"""

    base_url: str = DEFAULT_ILINK_BASE_URL
    bot_token: str | None = None
    get_updates_buf: str = ""
    typing_ticket: str | None = None


class FileWeChatStateStore:
    """基于文件系统的 WeChat 状态仓储。"""

    def __init__(self, state_dir: Path) -> None:
        """初始化状态仓储。

        Args:
            state_dir: 状态目录。

        Returns:
            无。

        Raises:
            无。
        """

        self._state_dir = Path(state_dir).expanduser().resolve()
        self._state_file = self._state_dir / "state.json"

    @property
    def state_dir(self) -> Path:
        """返回状态目录。"""

        return self._state_dir

    def load(self) -> WeChatDaemonState:
        """加载状态。

        Args:
            无。

        Returns:
            状态对象；文件不存在时返回默认状态。

        Raises:
            ValueError: 当状态文件不是合法对象时抛出。
        """

        if not self._state_file.exists():
            return WeChatDaemonState()
        raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("WeChat 状态文件必须是 JSON 对象")
        return WeChatDaemonState(
            base_url=str(raw.get("base_url") or DEFAULT_ILINK_BASE_URL),
            bot_token=str(raw.get("bot_token") or "").strip() or None,
            get_updates_buf=str(raw.get("get_updates_buf") or ""),
            typing_ticket=str(raw.get("typing_ticket") or "").strip() or None,
        )

    def save(self, state: WeChatDaemonState) -> None:
        """保存状态。

        Args:
            state: 待保存状态。

        Returns:
            无。

        Raises:
            无。
        """

        _write_text_atomic(
            self._state_file,
            json.dumps(asdict(state), ensure_ascii=False, indent=2, sort_keys=True),
        )

    def clear_auth(self) -> None:
        """清除登录态。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        state = self.load()
        state.bot_token = None
        state.base_url = DEFAULT_ILINK_BASE_URL
        state.typing_ticket = None
        self.save(state)

    def write_qrcode_artifact(self, qrcode_img_content: str | None) -> Path | None:
        """把二维码图片内容写到状态目录。

        Args:
            qrcode_img_content: 服务端返回的二维码内容。

        Returns:
            生成的文件路径；无内容时返回 `None`。

        Raises:
            无。
        """

        if not qrcode_img_content:
            return None
        if _looks_like_http_url(qrcode_img_content):
            self._state_dir.mkdir(parents=True, exist_ok=True)
            target = self._state_dir / "login_qrcode.html"
            escaped_url = html.escape(str(qrcode_img_content), quote=True)
            target.write_text(
                (
                    "<!doctype html>\n"
                    "<html lang=\"zh-CN\">\n"
                    "<head>\n"
                    "  <meta charset=\"utf-8\">\n"
                    "  <meta http-equiv=\"refresh\" content=\"0; url="
                    f"{escaped_url}"
                    "\">\n"
                    "  <title>WeChat Login QR</title>\n"
                    "</head>\n"
                    "<body>\n"
                    "  <p>如果没有自动跳转，请打开下面的链接查看微信登录二维码：</p>\n"
                    f"  <p><a href=\"{escaped_url}\">{escaped_url}</a></p>\n"
                    "</body>\n"
                    "</html>\n"
                ),
                encoding="utf-8",
            )
            return target
        mime_type, payload = _extract_base64_payload(qrcode_img_content)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        try:
            binary = base64.b64decode(payload, validate=True)
        except Exception:
            target = self._state_dir / "login_qrcode.txt"
            target.write_text(str(qrcode_img_content), encoding="utf-8")
            return target
        extension = mimetypes.guess_extension(mime_type or "") if mime_type else None
        if not extension:
            extension = _guess_binary_extension(binary)
        target = self._state_dir / f"login_qrcode{extension}"
        target.write_bytes(binary)
        return target


_TRACKED_SESSIONS_FILENAME = "tracked_sessions.json"


def record_tracked_session_id(state_dir: Path, session_id: str) -> None:
    """将 session_id 追加到 state_dir 下的追踪列表（去重）。

    Args:
        state_dir: 当前 daemon 使用的状态目录。
        session_id: 需要追踪的 session_id。

    Returns:
        无。

    Raises:
        OSError: 文件写入失败时抛出。
    """

    existing = load_tracked_session_ids(state_dir)
    if session_id in existing:
        return
    existing.append(session_id)
    target = state_dir / _TRACKED_SESSIONS_FILENAME
    _write_text_atomic(target, json.dumps(existing, ensure_ascii=False))


def load_tracked_session_ids(state_dir: Path) -> list[str]:
    """读取 state_dir 下已追踪的 session_id 列表。

    Args:
        state_dir: 当前 daemon 使用的状态目录。

    Returns:
        session_id 列表；文件不存在时返回空列表。

    Raises:
        无。
    """

    target = state_dir / _TRACKED_SESSIONS_FILENAME
    if not target.exists():
        return []
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if isinstance(item, str) and item.strip()]


__all__ = [
    "FileWeChatStateStore",
    "WeChatDaemonState",
    "build_wechat_runtime_identity",
    "build_wechat_session_id",
    "load_tracked_session_ids",
    "record_tracked_session_id",
]