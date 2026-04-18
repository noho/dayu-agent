"""``dayu-cli init`` 子命令测试。"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from dayu.cli.init_command import (
    _INIT_ROLE_KEY,
    _ROLE_NON_THINKING,
    _ROLE_THINKING,
    _classify_model_role,
    _copy_config,
    _detect_shell_profile,
    _update_manifest_default_models,
    _write_env_to_shell_profile,
    run_init,
)


# --------------------------------------------------------------------------- #
#  _detect_shell_profile
# --------------------------------------------------------------------------- #


class TestDetectShellProfile:
    """shell profile 检测测试。"""

    def test_zsh_shell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """$SHELL 含 zsh 时返回 ~/.zshrc，兼容标记为 True。"""
        monkeypatch.setenv("SHELL", "/bin/zsh")
        profile, compatible = _detect_shell_profile()
        assert profile == Path.home() / ".zshrc"
        assert compatible is True

    def test_bash_shell_bashrc(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """$SHELL 含 bash 且无 .bash_profile 时返回 ~/.bashrc。"""
        monkeypatch.setenv("SHELL", "/bin/bash")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        profile, compatible = _detect_shell_profile()
        assert profile == tmp_path / ".bashrc"
        assert compatible is True

    def test_bash_shell_bash_profile(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """$SHELL 含 bash 且 .bash_profile 存在时返回 .bash_profile。"""
        monkeypatch.setenv("SHELL", "/bin/bash")
        (tmp_path / ".bash_profile").touch()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        profile, compatible = _detect_shell_profile()
        assert profile == tmp_path / ".bash_profile"
        assert compatible is True

    def test_unknown_shell_returns_incompatible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未知 shell（如 fish）返回 ~/.profile，兼容标记为 False。"""
        monkeypatch.setenv("SHELL", "/bin/fish")
        profile, compatible = _detect_shell_profile()
        assert profile == Path.home() / ".profile"
        assert compatible is False


# --------------------------------------------------------------------------- #
#  _write_env_to_shell_profile
# --------------------------------------------------------------------------- #


class TestWriteEnvToShellProfile:
    """shell profile 写入测试。"""

    def test_append_new_key(self, tmp_path: Path) -> None:
        """新 key 追加到文件末尾。"""
        profile = tmp_path / ".zshrc"
        profile.write_text("# existing\n", encoding="utf-8")

        result = _write_env_to_shell_profile("MY_KEY", "my_value", profile)

        assert result is True
        content = profile.read_text(encoding="utf-8")
        assert 'export MY_KEY="my_value"' in content

    def test_replace_existing_key(self, tmp_path: Path) -> None:
        """已存在的 key 被替换。"""
        profile = tmp_path / ".zshrc"
        profile.write_text('export MY_KEY="old_value"\n', encoding="utf-8")

        result = _write_env_to_shell_profile("MY_KEY", "new_value", profile)

        assert result is True
        content = profile.read_text(encoding="utf-8")
        assert 'export MY_KEY="new_value"' in content
        assert "old_value" not in content

    def test_same_value_returns_false(self, tmp_path: Path) -> None:
        """值相同时返回 False。"""
        profile = tmp_path / ".zshrc"
        profile.write_text('export MY_KEY="same"\n', encoding="utf-8")

        result = _write_env_to_shell_profile("MY_KEY", "same", profile)
        assert result is False

    def test_create_new_file(self, tmp_path: Path) -> None:
        """文件不存在时创建。"""
        profile = tmp_path / ".zshrc"
        result = _write_env_to_shell_profile("MY_KEY", "val", profile)
        assert result is True
        assert profile.exists()
        assert 'export MY_KEY="val"' in profile.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
#  _copy_config
# --------------------------------------------------------------------------- #


class TestCopyConfig:
    """配置复制测试。"""

    def test_copy_creates_config_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """复制应创建 config 目录。"""
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "run.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.init_command._resolve_package_config_path",
            lambda: src,
        )

        base = tmp_path / "workspace"
        base.mkdir()
        result = _copy_config(base, overwrite=False)

        assert result.exists()
        assert (result / "run.json").exists()

    def test_skip_existing_without_overwrite(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """已存在时不覆盖。"""
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "new.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.init_command._resolve_package_config_path",
            lambda: src,
        )

        base = tmp_path / "workspace"
        config_dir = base / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "old.json").write_text("{}", encoding="utf-8")

        result = _copy_config(base, overwrite=False)

        assert (result / "old.json").exists()
        assert not (result / "new.json").exists()

    def test_overwrite_replaces(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--overwrite 时替换。"""
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "new.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.init_command._resolve_package_config_path",
            lambda: src,
        )

        base = tmp_path / "workspace"
        config_dir = base / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "old.json").write_text("{}", encoding="utf-8")

        result = _copy_config(base, overwrite=True)

        assert (result / "new.json").exists()
        assert not (result / "old.json").exists()


# --------------------------------------------------------------------------- #
#  _update_manifest_default_models
# --------------------------------------------------------------------------- #


class TestUpdateManifestDefaultModels:
    """Manifest 模型替换测试。"""

    def test_replaces_both_models(self, tmp_path: Path) -> None:
        """应替换 mimo-v2-pro-plan 和 mimo-v2-pro-thinking-plan。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        write_manifest = {"model": {"default_name": "mimo-v2-pro-plan"}}
        audit_manifest = {"model": {"default_name": "mimo-v2-pro-thinking-plan"}}

        (manifests / "write.json").write_text(json.dumps(write_manifest), encoding="utf-8")
        (manifests / "audit.json").write_text(json.dumps(audit_manifest), encoding="utf-8")

        count = _update_manifest_default_models(tmp_path, "deepseek-chat", "deepseek-thinking")

        assert count == 2

        write_data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        assert write_data["model"]["default_name"] == "deepseek-chat"

        audit_data = json.loads((manifests / "audit.json").read_text(encoding="utf-8"))
        assert audit_data["model"]["default_name"] == "deepseek-thinking"

    def test_skips_when_name_and_role_match(self, tmp_path: Path) -> None:
        """模型名和角色标记均已匹配时不计入更新。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        data = {"model": {"default_name": "mimo-v2-pro-plan", _INIT_ROLE_KEY: _ROLE_NON_THINKING}}
        (manifests / "a.json").write_text(json.dumps(data), encoding="utf-8")

        count = _update_manifest_default_models(tmp_path, "mimo-v2-pro-plan", "mimo-v2-pro-thinking-plan")
        assert count == 0

    def test_no_manifests_dir(self, tmp_path: Path) -> None:
        """manifests 目录不存在时返回 0。"""
        count = _update_manifest_default_models(tmp_path, "a", "b")
        assert count == 0


# --------------------------------------------------------------------------- #
#  run_init 集成测试
# --------------------------------------------------------------------------- #


class TestRunInit:
    """run_init 集成测试（mock 交互输入）。"""

    def test_full_flow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """完整流程：复制配置 → 选供应商 → 输入 Key → 更新 manifest。"""
        # 准备 mock 包内配置
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "run.json").write_text("{}", encoding="utf-8")
        manifests = src / "prompts" / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "mimo-v2-pro-plan"}}),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "dayu.cli.init_command._resolve_package_config_path",
            lambda: src,
        )

        # 确保环境变量中没有任何模型 key 和搜索 key
        for k in ("DEEPSEEK_API_KEY", "TAVILY_API_KEY", "SERPER_API_KEY", "FMP_API_KEY"):
            monkeypatch.delenv(k, raising=False)

        # Mock 交互输入: 选择 3 (DeepSeek)，输入 key，跳过可选 key
        inputs = iter(["3", "sk-test-key-123", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

        # Mock 环境变量持久化
        monkeypatch.setattr(
            "dayu.cli.init_command._persist_env_var",
            lambda _k, _v: ("~/.zshrc", True),
        )

        base = tmp_path / "workspace"
        base.mkdir()
        args = Namespace(base=str(base), overwrite=False)

        exit_code = run_init(args)
        assert exit_code == 0

        # 验证 manifest 被更新
        result_manifest = json.loads(
            (base / "config" / "prompts" / "manifests" / "write.json").read_text(encoding="utf-8")
        )
        assert result_manifest["model"]["default_name"] == "deepseek-chat"

    def test_skip_api_key_when_already_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """环境变量中已有 API Key 时跳过输入，不调用 _persist_env_var。"""
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "run.json").write_text("{}", encoding="utf-8")
        manifests = src / "prompts" / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "mimo-v2-pro-plan"}}),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "dayu.cli.init_command._resolve_package_config_path",
            lambda: src,
        )

        # 预设 DEEPSEEK_API_KEY 和所有搜索 key
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-already-set-123456")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")
        monkeypatch.setenv("SERPER_API_KEY", "sp-xxx")
        monkeypatch.setenv("FMP_API_KEY", "fmp-xxx")

        # 只需要选供应商（选 3 = DeepSeek），不需要输入 key
        inputs = iter(["3"])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

        # _persist_env_var 不应被调用
        persist_calls: list[str] = []

        def _mock_persist(k: str, v: str) -> tuple[str, bool]:
            persist_calls.append(k)
            return "~/.zshrc", True

        monkeypatch.setattr("dayu.cli.init_command._persist_env_var", _mock_persist)

        base = tmp_path / "workspace"
        base.mkdir()
        args = Namespace(base=str(base), overwrite=False)

        exit_code = run_init(args)
        assert exit_code == 0
        # 没有调用 _persist_env_var（key 已存在，搜索 key 也已存在）
        assert persist_calls == []

        # manifest 仍然被更新
        result_manifest = json.loads(
            (base / "config" / "prompts" / "manifests" / "write.json").read_text(encoding="utf-8")
        )
        assert result_manifest["model"]["default_name"] == "deepseek-chat"


class TestUpdateManifestSecondInit:
    """二次 init 换供应商时 manifest 应被正确替换。"""

    def test_switch_from_deepseek_to_qwen(self, tmp_path: Path) -> None:
        """已经是 deepseek-chat 的 manifest，换成 qwen3 应成功。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "deepseek-chat", _INIT_ROLE_KEY: _ROLE_NON_THINKING}}),
            encoding="utf-8",
        )
        (manifests / "audit.json").write_text(
            json.dumps({"model": {"default_name": "deepseek-thinking", _INIT_ROLE_KEY: _ROLE_THINKING}}),
            encoding="utf-8",
        )

        count = _update_manifest_default_models(tmp_path, "qwen3", "qwen3-thinking")
        assert count == 2

        write_data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        assert write_data["model"]["default_name"] == "qwen3"

        audit_data = json.loads((manifests / "audit.json").read_text(encoding="utf-8"))
        assert audit_data["model"]["default_name"] == "qwen3-thinking"

    def test_ambiguous_model_uses_role_marker(self, tmp_path: Path) -> None:
        """thinking/non-thinking 同名模型（如 gpt-5.4）通过角色标记正确分类。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        # gpt-5.4 同时在 non-thinking 和 thinking 集合中
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "gpt-5.4", _INIT_ROLE_KEY: _ROLE_NON_THINKING}}),
            encoding="utf-8",
        )
        (manifests / "audit.json").write_text(
            json.dumps({"model": {"default_name": "gpt-5.4", _INIT_ROLE_KEY: _ROLE_THINKING}}),
            encoding="utf-8",
        )

        count = _update_manifest_default_models(tmp_path, "deepseek-chat", "deepseek-thinking")
        assert count == 2

        write_data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        assert write_data["model"]["default_name"] == "deepseek-chat"

        audit_data = json.loads((manifests / "audit.json").read_text(encoding="utf-8"))
        assert audit_data["model"]["default_name"] == "deepseek-thinking"

    def test_ambiguous_model_without_marker_uses_package_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """歧义模型名无角色标记时，回退到包内原始 manifest 推断角色。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        # 工作区 manifest：gpt-5.4 无标记，无法直接判断角色
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "gpt-5.4"}}),
            encoding="utf-8",
        )

        # 包内原始 manifest：write.json 的 default_name 是无歧义的 non-thinking 模型
        pkg_manifests = tmp_path / "pkg_config" / "prompts" / "manifests"
        pkg_manifests.mkdir(parents=True)
        (pkg_manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "mimo-v2-pro-plan"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "dayu.cli.init_command._resolve_package_config_path",
            lambda: tmp_path / "pkg_config",
        )

        count = _update_manifest_default_models(tmp_path, "deepseek-chat", "deepseek-thinking")
        assert count == 1

        data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        assert data["model"]["default_name"] == "deepseek-chat"
        assert data["model"][_INIT_ROLE_KEY] == _ROLE_NON_THINKING

    def test_ambiguous_model_without_marker_no_package_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """歧义模型名无标记且包内无对应 manifest 时跳过。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "gpt-5.4"}}),
            encoding="utf-8",
        )

        # 包内无 manifests 目录
        pkg_config = tmp_path / "pkg_config"
        pkg_config.mkdir()
        monkeypatch.setattr(
            "dayu.cli.init_command._resolve_package_config_path",
            lambda: pkg_config,
        )

        count = _update_manifest_default_models(tmp_path, "deepseek-chat", "deepseek-thinking")
        assert count == 0

        data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        assert data["model"]["default_name"] == "gpt-5.4"

    def test_unknown_model_name_not_touched(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """不在已知供应商模型列表中的 default_name 不应被替换。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        (manifests / "custom.json").write_text(
            json.dumps({"model": {"default_name": "my-custom-model"}}),
            encoding="utf-8",
        )

        # 包内也无此 manifest，fallback 同样返回 None
        pkg_config = tmp_path / "pkg_config"
        pkg_config.mkdir()
        monkeypatch.setattr(
            "dayu.cli.init_command._resolve_package_config_path",
            lambda: pkg_config,
        )

        count = _update_manifest_default_models(tmp_path, "qwen3", "qwen3-thinking")
        assert count == 0

        data = json.loads((manifests / "custom.json").read_text(encoding="utf-8"))
        assert data["model"]["default_name"] == "my-custom-model"


class TestClassifyModelRole:
    """_classify_model_role 测试。"""

    def test_stored_role_takes_precedence(self) -> None:
        """有标记时优先使用标记。"""
        assert _classify_model_role("gpt-5.4", _ROLE_THINKING) == _ROLE_THINKING
        assert _classify_model_role("gpt-5.4", _ROLE_NON_THINKING) == _ROLE_NON_THINKING

    def test_unambiguous_non_thinking(self) -> None:
        """仅在 non-thinking 集合中的模型名正确分类。"""
        assert _classify_model_role("deepseek-chat", "") == _ROLE_NON_THINKING

    def test_unambiguous_thinking(self) -> None:
        """仅在 thinking 集合中的模型名正确分类。"""
        assert _classify_model_role("deepseek-thinking", "") == _ROLE_THINKING

    def test_ambiguous_without_marker_returns_none(self) -> None:
        """歧义模型名无标记时返回 None。"""
        assert _classify_model_role("gpt-5.4", "") is None

    def test_unknown_model_returns_none(self) -> None:
        """完全未知的模型名返回 None。"""
        assert _classify_model_role("my-custom-model", "") is None
