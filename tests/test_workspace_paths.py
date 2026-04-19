"""dayu.workspace_paths 模块测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from dayu.workspace_paths import (
    build_conversation_store_dir,
    build_dayu_root_path,
    build_host_store_default_path,
    build_interactive_state_dir,
    build_sec_cache_dir,
    build_sec_throttle_dir,
    build_wechat_state_dir,
    extract_wechat_instance_label,
    list_wechat_state_dirs,
)

pytestmark = pytest.mark.unit


class TestBuildDayuRootPath:
    """build_dayu_root_path 测试。"""

    def test_returns_dot_dayu_under_workspace(self, tmp_path: Path) -> None:
        """应返回 workspace 下的 .dayu 路径。"""
        result = build_dayu_root_path(tmp_path)
        assert result == tmp_path / ".dayu"


class TestBuildHostStoreDefaultPath:
    """build_host_store_default_path 测试。"""

    def test_returns_db_under_dayu_host(self, tmp_path: Path) -> None:
        """应返回 .dayu/host/dayu_host.db 路径。"""
        result = build_host_store_default_path(tmp_path)
        assert result == tmp_path / ".dayu" / "host" / "dayu_host.db"


class TestBuildConversationStoreDir:
    """build_conversation_store_dir 测试。"""

    def test_returns_session_dir(self, tmp_path: Path) -> None:
        """应返回 .dayu/session 路径。"""
        result = build_conversation_store_dir(tmp_path)
        assert result == tmp_path / ".dayu" / "session"


class TestBuildInteractiveStateDir:
    """build_interactive_state_dir 测试。"""

    def test_returns_interactive_dir(self, tmp_path: Path) -> None:
        """应返回 .dayu/interactive 路径。"""
        result = build_interactive_state_dir(tmp_path)
        assert result == tmp_path / ".dayu" / "interactive"


class TestBuildSecCacheDir:
    """build_sec_cache_dir 测试。"""

    def test_returns_sec_cache_dir(self, tmp_path: Path) -> None:
        """应返回 .dayu/sec_cache 路径。"""
        result = build_sec_cache_dir(tmp_path)
        assert result == tmp_path / ".dayu" / "sec_cache"


class TestBuildSecThrottleDir:
    """build_sec_throttle_dir 测试。"""

    def test_returns_sec_throttle_dir(self, tmp_path: Path) -> None:
        """应返回 .dayu/sec_throttle 路径。"""
        result = build_sec_throttle_dir(tmp_path)
        assert result == tmp_path / ".dayu" / "sec_throttle"


class TestBuildWechatStateDir:
    """build_wechat_state_dir 测试。"""

    def test_default_label(self, tmp_path: Path) -> None:
        """默认标签应生成 wechat-default 目录。"""
        result = build_wechat_state_dir(tmp_path)
        assert result == tmp_path / ".dayu" / "wechat-default"

    def test_custom_label(self, tmp_path: Path) -> None:
        """自定义标签应生成对应目录。"""
        result = build_wechat_state_dir(tmp_path, label="alice")
        assert result == tmp_path / ".dayu" / "wechat-alice"


class TestExtractWechatInstanceLabel:
    """extract_wechat_instance_label 测试。"""

    def test_prefix_match_returns_label(self) -> None:
        """前缀匹配成功应返回标签。"""
        result = extract_wechat_instance_label("wechat-default")
        assert result == "default"

    def test_prefix_match_custom_label(self) -> None:
        """自定义标签应正确提取。"""
        result = extract_wechat_instance_label("wechat-alice")
        assert result == "alice"

    def test_no_prefix_returns_none(self) -> None:
        """不以 wechat- 开头应返回 None。"""
        result = extract_wechat_instance_label("other-dir")
        assert result is None

    def test_empty_label_returns_none(self) -> None:
        """仅有前缀无标签应返回 None。"""
        result = extract_wechat_instance_label("wechat-")
        assert result is None

    def test_whitespace_stripped(self) -> None:
        """前后空白应被 strip 后再匹配。"""
        result = extract_wechat_instance_label("  wechat-foo  ")
        assert result == "foo"

    def test_empty_string_returns_none(self) -> None:
        """空字符串应返回 None。"""
        result = extract_wechat_instance_label("")
        assert result is None

    def test_whitespace_only_returns_none(self) -> None:
        """仅空白字符应 strip 后返回 None。"""
        result = extract_wechat_instance_label("   ")
        assert result is None


class TestListWechatStateDirs:
    """list_wechat_state_dirs 测试。"""

    def test_dayu_root_not_exists_returns_empty(self, tmp_path: Path) -> None:
        """.dayu 目录不存在时应返回空元组。"""
        result = list_wechat_state_dirs(tmp_path)
        assert result == ()

    def test_dayu_root_is_file_returns_empty(self, tmp_path: Path) -> None:
        """.dayu 是文件而非目录时应返回空元组。"""
        dayu_file = tmp_path / ".dayu"
        dayu_file.write_text("not a dir")
        result = list_wechat_state_dirs(tmp_path)
        assert result == ()

    def test_skips_non_directory_entries(self, tmp_path: Path) -> None:
        """应跳过非目录的条目。"""
        dayu_root = tmp_path / ".dayu"
        dayu_root.mkdir()
        # 创建一个文件，名称虽然匹配前缀但不是目录
        (dayu_root / "wechat-file").write_text("x")
        result = list_wechat_state_dirs(tmp_path)
        assert result == ()

    def test_skips_dirs_with_invalid_label(self, tmp_path: Path) -> None:
        """应跳过目录名无法提取标签的目录。"""
        dayu_root = tmp_path / ".dayu"
        dayu_root.mkdir()
        (dayu_root / "other-dir").mkdir()
        # wechat- 无标签
        (dayu_root / "wechat-").mkdir()
        result = list_wechat_state_dirs(tmp_path)
        assert result == ()

    def test_returns_matching_dirs_sorted(self, tmp_path: Path) -> None:
        """应返回匹配的目录并按名称排序。"""
        dayu_root = tmp_path / ".dayu"
        dayu_root.mkdir()
        # 乱序创建
        (dayu_root / "wechat-charlie").mkdir()
        (dayu_root / "wechat-alice").mkdir()
        (dayu_root / "wechat-bob").mkdir()

        result = list_wechat_state_dirs(tmp_path)
        assert result == (
            dayu_root / "wechat-alice",
            dayu_root / "wechat-bob",
            dayu_root / "wechat-charlie",
        )

    def test_mixed_entries_filters_correctly(self, tmp_path: Path) -> None:
        """混合目录与文件时应只返回有效 wechat 状态目录。"""
        dayu_root = tmp_path / ".dayu"
        dayu_root.mkdir()
        (dayu_root / "wechat-apple").mkdir()
        (dayu_root / "random-file").write_text("x")
        (dayu_root / "other-dir").mkdir()
        (dayu_root / "wechat-").mkdir()

        result = list_wechat_state_dirs(tmp_path)
        assert result == (dayu_root / "wechat-apple",)
