"""WeChat 状态仓储测试。"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from dayu.wechat import state_store as state_store_module
from dayu.wechat.state_store import (
    FileWeChatStateStore,
    WeChatDaemonState,
    build_wechat_runtime_identity,
    build_wechat_session_id,
)


@pytest.mark.unit
def test_build_wechat_session_id_is_stable() -> None:
    """验证同一 chat_key 总是生成同一 session_id。"""

    first = build_wechat_session_id("user@im.wechat")
    second = build_wechat_session_id("user@im.wechat")

    assert first == second
    assert first.startswith("wechat_")


@pytest.mark.unit
def test_build_wechat_session_id_rejects_blank_key() -> None:
    """验证空 chat_key 会被拒绝。"""

    with pytest.raises(ValueError, match="chat_key"):
        build_wechat_session_id("   ")


@pytest.mark.unit
def test_build_wechat_runtime_identity_is_stable(tmp_path: Path) -> None:
    """验证 runtime identity 基于状态目录稳定生成。"""

    state_dir = tmp_path / ".wechat"

    assert build_wechat_runtime_identity(state_dir) == build_wechat_runtime_identity(state_dir)
    assert build_wechat_runtime_identity(state_dir).startswith("wechat_runtime_")


@pytest.mark.unit
def test_state_store_round_trip_and_preserve_session_map(tmp_path: Path) -> None:
    """验证状态文件可往返持久化。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    state = WeChatDaemonState(
        base_url="https://example.com",
        bot_token="token-1",
        get_updates_buf="cursor-1",
        typing_ticket="typing-1",
    )

    store.save(state)
    loaded = store.load()

    assert loaded == state


@pytest.mark.unit
def test_state_store_load_returns_default_when_missing_file(tmp_path: Path) -> None:
    """验证状态文件缺失时返回默认状态。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")

    assert store.load() == WeChatDaemonState()


@pytest.mark.unit
def test_state_store_load_rejects_non_object_json(tmp_path: Path) -> None:
    """验证状态文件顶层不是 JSON 对象时抛错。"""

    state_dir = tmp_path / ".wechat"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "state.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON 对象"):
        FileWeChatStateStore(state_dir).load()


@pytest.mark.unit
def test_state_store_save_does_not_leave_temp_files(tmp_path: Path) -> None:
    """验证原子写保存后不会残留临时文件。"""

    state_dir = tmp_path / ".wechat"
    store = FileWeChatStateStore(state_dir)
    store.save(WeChatDaemonState(bot_token="token-1", get_updates_buf="cursor-1"))

    child_names = sorted(path.name for path in state_dir.iterdir())

    assert child_names == ["state.json"]


@pytest.mark.unit
def test_clear_auth_preserves_existing_session_map(tmp_path: Path) -> None:
    """验证清理登录态时状态文件正常。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    store.save(
        WeChatDaemonState(
            bot_token="token-1",
            base_url="https://example.com",
            typing_ticket="typing-1",
        )
    )

    store.clear_auth()
    loaded = store.load()

    assert loaded.bot_token is None
    assert loaded.typing_ticket is None
    assert loaded.base_url == "https://ilinkai.weixin.qq.com"


@pytest.mark.unit
def test_write_qrcode_artifact_supports_base64_binary(tmp_path: Path) -> None:
    """验证二维码二进制内容会落到文件。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    png_bytes = b"\x89PNG\r\n\x1a\nrest"
    raw = base64.b64encode(png_bytes).decode("utf-8")

    artifact = store.write_qrcode_artifact(raw)

    assert artifact is not None
    assert artifact.suffix == ".png"
    assert artifact.read_bytes() == png_bytes


@pytest.mark.unit
def test_write_qrcode_artifact_supports_url_payload(tmp_path: Path) -> None:
    """验证二维码链接会落成可双击打开的 html 文件。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")

    artifact = store.write_qrcode_artifact("https://liteapp.weixin.qq.com/q/demo")

    assert artifact is not None
    assert artifact.suffix == ".html"
    assert "https://liteapp.weixin.qq.com/q/demo" in artifact.read_text(encoding="utf-8")


@pytest.mark.unit
def test_write_qrcode_artifact_falls_back_to_plain_text_for_invalid_base64(tmp_path: Path) -> None:
    """验证无法解码的二维码内容会回退保存为文本。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")

    artifact = store.write_qrcode_artifact("not-valid-base64@@")

    assert artifact is not None
    assert artifact.suffix == ".txt"
    assert artifact.read_text(encoding="utf-8") == "not-valid-base64@@"


@pytest.mark.unit
def test_write_qrcode_artifact_prefers_data_uri_mime_extension(tmp_path: Path) -> None:
    """验证 data URI 会优先使用 MIME 推断扩展名。"""

    store = FileWeChatStateStore(tmp_path / ".wechat")
    payload = base64.b64encode(b"<svg xmlns='http://www.w3.org/2000/svg'></svg>").decode("utf-8")

    artifact = store.write_qrcode_artifact(f"data:image/svg+xml;base64,{payload}")

    assert artifact is not None
    assert artifact.suffix == ".svg"


@pytest.mark.unit
def test_state_store_private_helpers_cover_url_base64_and_extension_detection() -> None:
    """验证状态仓储私有 helper 的基础分支。"""

    assert state_store_module._looks_like_http_url(" https://example.com ") is True
    assert state_store_module._looks_like_http_url("ftp://example.com") is False
    assert state_store_module._extract_base64_payload("data:image/png;base64,AAA=") == ("image/png", "AAA=")
    assert state_store_module._extract_base64_payload("BBB=") == (None, "BBB=")
    assert state_store_module._guess_binary_extension(b"GIF89atest") == ".gif"
    assert state_store_module._guess_binary_extension(b"\xff\xd8\xffrest") == ".jpg"
    assert state_store_module._guess_binary_extension(b"   <svg viewBox='0 0 1 1'>") == ".svg"
    assert state_store_module._guess_binary_extension(b"unknown") == ".bin"

@pytest.mark.unit
def test_write_text_atomic_closes_fd_when_fdopen_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 ``_write_text_atomic`` 在 fdopen 失败时也会关闭底层 fd，避免泄漏。"""

    import os as os_mod

    closed_fds: list[int] = []
    original_close = os_mod.close

    def _capturing_close(fd: int) -> None:
        closed_fds.append(fd)
        original_close(fd)

    def _failing_fdopen(*_args: object, **_kwargs: object) -> object:
        raise OSError("simulated fdopen failure")

    monkeypatch.setattr(state_store_module.os, "fdopen", _failing_fdopen)
    monkeypatch.setattr(state_store_module.os, "close", _capturing_close)

    target = tmp_path / "ledger.json"
    with pytest.raises(OSError, match="simulated fdopen failure"):
        state_store_module._write_text_atomic(target, "{}")

    assert closed_fds, "fd 应在 fdopen 失败路径下被显式关闭"
    # 临时文件应已被清理
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
