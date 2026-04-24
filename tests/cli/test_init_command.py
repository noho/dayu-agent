"""``dayu-cli init`` 子命令测试。"""

from __future__ import annotations

import json
import urllib.error
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from dayu.cli.commands.init import (
    _CUSTOM_CATALOG_KEY,
    _CustomOpenAIConfig,
    _HF_MIRROR_URL,
    _INIT_ROLE_KEY,
    _PROVIDER_OPTION_CUSTOM_OPENAI,
    _PROVIDER_OPTION_DEEPSEEK_FLASH,
    _PROVIDER_OPTION_DEEPSEEK_PRO,
    _PROVIDER_OPTION_MIMO_PRO,
    _ROLE_NON_THINKING,
    _ROLE_THINKING,
    _classify_model_role,
    _confirm_workspace_reset,
    _copy_assets,
    _copy_config,
    _detect_shell_profile,
    _is_hf_hub_reachable,
    _persist_env_var,
    _reset_workspace_init_targets,
    _prompt_api_key,
    _prompt_custom_openai_config,
    _prompt_huggingface_config,
    _prompt_optional_search_keys,
    _prompt_provider_selection,
    _prompt_sec_user_agent,
    _resolve_role_from_package_manifest,
    _run_init_prewarm,
    _should_run_init_prewarm,
    _update_manifest_default_models,
    _write_custom_openai_catalog_entries,
    _write_env_to_shell_profile,
    _write_env_windows,
    SEC_USER_AGENT_ENV,
    run_init_command,
)

_PROVIDER_ENV_KEYS: tuple[str, ...] = (
    "MIMO_PLAN_API_KEY",
    "MIMO_PLAN_SG_API_KEY",
    "MIMO_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "QWEN_API_KEY",
    "CUSTOM_OPENAI_API_KEY",
)


def _clear_provider_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """清理 `dayu-cli init` 主模型供应商相关环境变量。"""

    for key in _PROVIDER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


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

    def test_fish_shell_returns_incompatible(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """明确识别为 fish 时返回 ~/.profile，兼容标记为 False。"""
        monkeypatch.setenv("SHELL", "/bin/fish")
        profile, compatible = _detect_shell_profile()
        assert profile == Path.home() / ".profile"
        assert compatible is False

    def test_missing_shell_falls_back_to_existing_bashrc(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """SHELL 缺失时，回退到 HOME 下现有 .bashrc。"""
        monkeypatch.delenv("SHELL", raising=False)
        (tmp_path / ".bashrc").touch()
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        profile, compatible = _detect_shell_profile()
        assert profile == tmp_path / ".bashrc"
        assert compatible is True

    def test_missing_shell_without_known_profile_uses_profile(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """SHELL 缺失且无现有 profile 时，回退到 ~/.profile 并视为兼容。"""
        monkeypatch.delenv("SHELL", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        profile, compatible = _detect_shell_profile()
        assert profile == tmp_path / ".profile"
        assert compatible is True


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
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: src,
        )

        base = tmp_path / "workspace"
        base.mkdir()
        result = _copy_config(base, overwrite=False)

        assert result.exists()
        assert (result / "run.json").exists()

    def test_skip_existing_without_overwrite(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """已存在时不覆盖非 prompt 配置。"""
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "new.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: src,
        )

        base = tmp_path / "workspace"
        config_dir = base / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "old.json").write_text("{}", encoding="utf-8")

        result = _copy_config(base, overwrite=False)

        assert (result / "old.json").exists()
        assert not (result / "new.json").exists()

    def test_backfills_missing_prompt_assets_without_overwrite(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """已有 config 时补齐缺失 prompt 资产，但不覆盖已有 prompt 文件。"""

        src = tmp_path / "pkg_config"
        pkg_manifests = src / "prompts" / "manifests"
        pkg_scenes = src / "prompts" / "scenes"
        pkg_manifests.mkdir(parents=True)
        pkg_scenes.mkdir(parents=True)
        (pkg_manifests / "prompt.json").write_text(
            json.dumps({"scene": "prompt", "model": {"default_name": "mimo-v2.5-pro-thinking-plan"}}),
            encoding="utf-8",
        )
        (pkg_manifests / "prompt_mt.json").write_text(
            json.dumps({"scene": "prompt_mt", "model": {"default_name": "mimo-v2.5-pro-thinking-plan"}}),
            encoding="utf-8",
        )
        (pkg_scenes / "prompt.md").write_text("# prompt old", encoding="utf-8")
        (pkg_scenes / "prompt_mt.md").write_text("# prompt_mt new", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: src,
        )

        base = tmp_path / "workspace"
        config_dir = base / "config"
        manifests_dir = config_dir / "prompts" / "manifests"
        scenes_dir = config_dir / "prompts" / "scenes"
        manifests_dir.mkdir(parents=True)
        scenes_dir.mkdir(parents=True)
        (manifests_dir / "prompt.json").write_text(
            json.dumps({"scene": "prompt", "model": {"default_name": "kimi-k2.5"}}),
            encoding="utf-8",
        )
        (scenes_dir / "prompt.md").write_text("# prompt customized", encoding="utf-8")

        result = _copy_config(base, overwrite=False)

        assert result == config_dir
        assert json.loads((manifests_dir / "prompt.json").read_text(encoding="utf-8"))["model"][
            "default_name"
        ] == "kimi-k2.5"
        assert (manifests_dir / "prompt_mt.json").exists()
        assert (scenes_dir / "prompt.md").read_text(encoding="utf-8") == "# prompt customized"
        assert (scenes_dir / "prompt_mt.md").read_text(encoding="utf-8") == "# prompt_mt new"

    def test_overwrite_replaces(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--overwrite 时替换。"""
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "new.json").write_text("{}", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
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
#  _copy_assets
# --------------------------------------------------------------------------- #


class TestCopyAssets:
    """assets 复制测试。"""

    def test_copy_creates_assets_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """复制应创建 assets 目录。"""
        src = tmp_path / "pkg_assets"
        src.mkdir()
        (src / "定性分析模板.md").write_text("# 模板", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_assets_path",
            lambda: src,
        )

        base = tmp_path / "workspace"
        base.mkdir()
        result = _copy_assets(base, overwrite=False)

        assert result.exists()
        assert (result / "定性分析模板.md").exists()

    def test_skip_existing_without_overwrite(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """已存在时不覆盖。"""
        src = tmp_path / "pkg_assets"
        src.mkdir()
        (src / "定性分析模板.md").write_text("# 新模板", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_assets_path",
            lambda: src,
        )

        base = tmp_path / "workspace"
        assets_dir = base / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "定性分析模板.md").write_text("# 旧模板", encoding="utf-8")

        result = _copy_assets(base, overwrite=False)

        assert (result / "定性分析模板.md").read_text(encoding="utf-8") == "# 旧模板"

    def test_overwrite_replaces(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--overwrite 时替换。"""
        src = tmp_path / "pkg_assets"
        src.mkdir()
        (src / "定性分析模板.md").write_text("# 新模板", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_assets_path",
            lambda: src,
        )

        base = tmp_path / "workspace"
        assets_dir = base / "assets"
        assets_dir.mkdir(parents=True)
        (assets_dir / "定性分析模板.md").write_text("# 旧模板", encoding="utf-8")

        result = _copy_assets(base, overwrite=True)

        assert (result / "定性分析模板.md").read_text(encoding="utf-8") == "# 新模板"


# --------------------------------------------------------------------------- #
#  _confirm_workspace_reset
# --------------------------------------------------------------------------- #


class TestConfirmWorkspaceReset:
    """工作区 reset 二次确认测试。"""

    def test_prompt_lists_targets_from_reset_definition(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """确认提示应列出 reset 真源中的全部目标。"""

        captured_prompt: list[str] = []

        def _capture_input(prompt: str = "") -> str:
            captured_prompt.append(prompt)
            return "n"

        base_dir = tmp_path / "workspace"
        monkeypatch.setattr("builtins.input", _capture_input)

        assert _confirm_workspace_reset(base_dir) is False
        assert len(captured_prompt) == 1
        assert str(base_dir / ".dayu") in captured_prompt[0]
        assert str(base_dir / "config") in captured_prompt[0]
        assert str(base_dir / "assets") in captured_prompt[0]

    def test_yes_confirms_reset(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """输入 y 时返回 True。"""
        monkeypatch.setattr("builtins.input", lambda *_args: "y")

        assert _confirm_workspace_reset(tmp_path / "workspace") is True

    def test_enter_defaults_to_no(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """直接回车时按 N 处理。"""
        monkeypatch.setattr("builtins.input", lambda *_args: "")

        assert _confirm_workspace_reset(tmp_path / "workspace") is False

    def test_keyboard_interrupt_defaults_to_no(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """中断输入时按未确认处理。"""

        def _raise_keyboard_interrupt(*_args: str) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", _raise_keyboard_interrupt)

        assert _confirm_workspace_reset(tmp_path / "workspace") is False


# --------------------------------------------------------------------------- #
#  _reset_workspace_init_targets
# --------------------------------------------------------------------------- #


class TestResetWorkspaceInitTargets:
    """工作区 reset 目标删除测试。"""

    def test_reset_removes_dayu_config_and_assets_dirs(self, tmp_path: Path) -> None:
        """应删除 `.dayu`、`config`、`assets` 三个目录。"""
        base = tmp_path / "workspace"
        dayu_dir = base / ".dayu"
        config_dir = base / "config"
        assets_dir = base / "assets"
        (dayu_dir / "host").mkdir(parents=True)
        config_dir.mkdir(parents=True)
        assets_dir.mkdir(parents=True)

        removed = _reset_workspace_init_targets(base)

        assert removed == (dayu_dir, config_dir, assets_dir)
        assert not dayu_dir.exists()
        assert not config_dir.exists()
        assert not assets_dir.exists()

    def test_reset_also_removes_file_targets(self, tmp_path: Path) -> None:
        """若目标被错误占成文件，也应删除以便重新初始化。"""
        base = tmp_path / "workspace"
        dayu_file = base / ".dayu"
        config_file = base / "config"
        assets_file = base / "assets"
        base.mkdir()
        dayu_file.write_text("state", encoding="utf-8")
        config_file.write_text("config", encoding="utf-8")
        assets_file.write_text("assets", encoding="utf-8")

        removed = _reset_workspace_init_targets(base)

        assert removed == (dayu_file, config_file, assets_file)
        assert not dayu_file.exists()
        assert not config_file.exists()
        assert not assets_file.exists()


# --------------------------------------------------------------------------- #
#  _update_manifest_default_models
# --------------------------------------------------------------------------- #


class TestUpdateManifestDefaultModels:
    """Manifest 模型替换测试。"""

    def test_replaces_both_models(self, tmp_path: Path) -> None:
        """应替换 mimo-v2.5-pro-plan 和 mimo-v2.5-pro-thinking-plan。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        write_manifest = {"model": {"default_name": "mimo-v2.5-pro-plan"}}
        audit_manifest = {"model": {"default_name": "mimo-v2.5-pro-thinking-plan"}}

        (manifests / "write.json").write_text(json.dumps(write_manifest), encoding="utf-8")
        (manifests / "audit.json").write_text(json.dumps(audit_manifest), encoding="utf-8")

        count = _update_manifest_default_models(tmp_path, "deepseek-v4-flash", "deepseek-v4-flash-thinking")

        assert count == 2

        write_data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        assert write_data["model"]["default_name"] == "deepseek-v4-flash"

        audit_data = json.loads((manifests / "audit.json").read_text(encoding="utf-8"))
        assert audit_data["model"]["default_name"] == "deepseek-v4-flash-thinking"

    def test_skips_when_name_and_role_match(self, tmp_path: Path) -> None:
        """模型名和角色标记均已匹配时不计入更新。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        data = {"model": {"default_name": "mimo-v2.5-pro-plan", _INIT_ROLE_KEY: _ROLE_NON_THINKING}}
        (manifests / "a.json").write_text(json.dumps(data), encoding="utf-8")

        count = _update_manifest_default_models(tmp_path, "mimo-v2.5-pro-plan", "mimo-v2.5-pro-thinking-plan")
        assert count == 0

    def test_no_manifests_dir(self, tmp_path: Path) -> None:
        """manifests 目录不存在时返回 0。"""
        count = _update_manifest_default_models(tmp_path, "a", "b")
        assert count == 0

    def test_appends_custom_openai_to_allowed_names(self, tmp_path: Path) -> None:
        """切到 custom-openai 时应把模型名追加到 allowed_names。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "deepseek-v4-flash", "allowed_names": ["deepseek-v4-flash"]}}),
            encoding="utf-8",
        )
        (manifests / "audit.json").write_text(
            json.dumps({"model": {"default_name": "deepseek-v4-flash-thinking", "allowed_names": ["deepseek-v4-flash-thinking"]}}),
            encoding="utf-8",
        )

        count = _update_manifest_default_models(
            tmp_path,
            _CUSTOM_CATALOG_KEY,
            _CUSTOM_CATALOG_KEY,
        )

        assert count == 2
        write_data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        audit_data = json.loads((manifests / "audit.json").read_text(encoding="utf-8"))
        assert write_data["model"]["allowed_names"] == ["deepseek-v4-flash", _CUSTOM_CATALOG_KEY]
        assert audit_data["model"]["allowed_names"] == ["deepseek-v4-flash-thinking", _CUSTOM_CATALOG_KEY]


class TestPromptCustomOpenAIConfig:
    """自定义 OpenAI 兼容 API 交互测试。"""

    def test_collects_values_and_normalizes_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """正常输入时返回收集结果，并去掉 base URL 末尾斜杠。"""
        monkeypatch.delenv("CUSTOM_OPENAI_API_KEY", raising=False)
        inputs = iter(["https://openrouter.ai/api/v1/", "sk-custom", "openai/gpt-4o"])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

        config = _prompt_custom_openai_config("CUSTOM_OPENAI_API_KEY")

        assert config == _CustomOpenAIConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key_value="sk-custom",
            model_id="openai/gpt-4o",
        )

    def test_reuses_existing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """环境变量已存在时复用现有 API Key，不再重复输入。"""
        monkeypatch.setenv("CUSTOM_OPENAI_API_KEY", "sk-existing-123456")
        inputs = iter(["https://openrouter.ai/api/v1", "anthropic/claude-sonnet-4"])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

        config = _prompt_custom_openai_config("CUSTOM_OPENAI_API_KEY")

        assert config.api_key_value == "sk-existing-123456"
        assert config.model_id == "anthropic/claude-sonnet-4"

    def test_empty_base_url_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Base URL 为空时退出。"""
        monkeypatch.setattr("builtins.input", lambda *_args: "")
        with pytest.raises(SystemExit) as exc_info:
            _prompt_custom_openai_config("CUSTOM_OPENAI_API_KEY")
        assert exc_info.value.code == 1

    def test_eof_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EOF 时退出。"""
        monkeypatch.setattr("builtins.input", lambda *_args: (_ for _ in ()).throw(EOFError))
        with pytest.raises(SystemExit) as exc_info:
            _prompt_custom_openai_config("CUSTOM_OPENAI_API_KEY")
        assert exc_info.value.code == 1


class TestWriteCustomOpenAICatalogEntries:
    """自定义 OpenAI 兼容 catalog 写入测试。"""

    def test_merges_into_existing_catalog(self, tmp_path: Path) -> None:
        """写入 custom-openai 时保留其他供应商条目。"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        catalog_path = config_dir / "llm_models.json"
        catalog_path.write_text(
            json.dumps(
                {
                    "deepseek-v4-flash": {
                        "runner_type": "openai_compatible",
                        "name": "deepseek-v4-flash",
                    }
                }
            ),
            encoding="utf-8",
        )

        _write_custom_openai_catalog_entries(
            config_dir,
            _CustomOpenAIConfig(
                base_url="https://openrouter.ai/api/v1",
                api_key_value="sk-custom",
                model_id="openai/gpt-4o",
            ),
            api_key_name="CUSTOM_OPENAI_API_KEY",
        )

        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        assert "deepseek-v4-flash" in data
        assert data[_CUSTOM_CATALOG_KEY]["endpoint_url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert data[_CUSTOM_CATALOG_KEY]["model"] == "openai/gpt-4o"
        assert data[_CUSTOM_CATALOG_KEY]["headers"]["Authorization"] == "Bearer {{CUSTOM_OPENAI_API_KEY}}"
        assert data[_CUSTOM_CATALOG_KEY]["runtime_hints"]["temperature_profiles"]["write"]["temperature"] == 1.0

    def test_invalid_json_raises_value_error(self, tmp_path: Path) -> None:
        """损坏的 llm_models.json 应抛出用户可理解的 ValueError。"""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "llm_models.json").write_text("{broken json", encoding="utf-8")

        with pytest.raises(ValueError, match="不是合法 JSON"):
            _write_custom_openai_catalog_entries(
                config_dir,
                _CustomOpenAIConfig(
                    base_url="https://openrouter.ai/api/v1",
                    api_key_value="sk-custom",
                    model_id="openai/gpt-4o",
                ),
                api_key_name="CUSTOM_OPENAI_API_KEY",
            )


# --------------------------------------------------------------------------- #
#  run_init_command 集成测试
# --------------------------------------------------------------------------- #


class TestRunInit:
    """run_init_command 集成测试（mock 交互输入）。"""

    def test_full_flow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """完整流程：复制配置 → 选供应商 → 输入 Key → 更新 manifest。"""
        # 准备 mock 包内配置
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "run.json").write_text("{}", encoding="utf-8")
        manifests = src / "prompts" / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "mimo-v2.5-pro-plan"}}),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: src,
        )

        # 准备 mock 包内 assets
        pkg_assets = tmp_path / "pkg_assets"
        pkg_assets.mkdir()
        (pkg_assets / "定性分析模板.md").write_text("# 模板", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_assets_path",
            lambda: pkg_assets,
        )

        # 确保环境变量中没有任何模型 key 和搜索 key
        _clear_provider_env_vars(monkeypatch)
        for k in ("TAVILY_API_KEY", "SERPER_API_KEY", "FMP_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv(SEC_USER_AGENT_ENV, raising=False)

        # 清除 HF 环境变量
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        # Mock HF Hub 不可达 → 默认 Y
        monkeypatch.setattr("dayu.cli.commands.init._is_hf_hub_reachable", lambda: False)

        # Mock 交互输入: 输入 key，跳过可选 key x3，HF 镜像回车(默认Y)，HF_TOKEN 跳过
        inputs = iter(["sk-test-key-123", "", "", "", "", "", ""])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)

        # Mock 环境变量持久化
        monkeypatch.setattr(
            "dayu.cli.commands.init._persist_env_var",
            lambda _k, _v: ("~/.zshrc", True),
        )
        monkeypatch.setattr(
            "dayu.cli.commands.init._run_init_prewarm",
            lambda **_kwargs: (True, ""),
        )

        base = tmp_path / "workspace"
        base.mkdir()
        args = Namespace(base=str(base), overwrite=False)

        exit_code = run_init_command(args)
        assert exit_code == 0

        # 验证 manifest 被更新
        result_manifest = json.loads(
            (base / "config" / "prompts" / "manifests" / "write.json").read_text(encoding="utf-8")
        )
        assert result_manifest["model"]["default_name"] == "deepseek-v4-flash"

    def test_skip_api_key_when_already_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """环境变量中已有 API Key 时跳过输入，不调用 _persist_env_var。"""
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "run.json").write_text("{}", encoding="utf-8")
        manifests = src / "prompts" / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "mimo-v2.5-pro-plan"}}),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: src,
        )

        # 准备 mock 包内 assets
        pkg_assets = tmp_path / "pkg_assets"
        pkg_assets.mkdir()
        (pkg_assets / "定性分析模板.md").write_text("# 模板", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_assets_path",
            lambda: pkg_assets,
        )

        _clear_provider_env_vars(monkeypatch)
        # 预设 DEEPSEEK_API_KEY 和所有搜索 key
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-already-set-123456")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")
        monkeypatch.setenv("SERPER_API_KEY", "sp-xxx")
        monkeypatch.setenv("FMP_API_KEY", "fmp-xxx")
        monkeypatch.setenv(SEC_USER_AGENT_ENV, "Demo demo@example.com")

        # 预设 HF 环境变量
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        monkeypatch.setenv("HF_TOKEN", "hf-test-xxx")

        # _persist_env_var 不应被调用
        persist_calls: list[str] = []

        def _mock_persist(k: str, v: str) -> tuple[str, bool]:
            persist_calls.append(k)
            return "~/.zshrc", True

        monkeypatch.setattr("dayu.cli.commands.init._persist_env_var", _mock_persist)
        monkeypatch.setattr(
            "dayu.cli.commands.init._run_init_prewarm",
            lambda **_kwargs: (True, ""),
        )
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)

        base = tmp_path / "workspace"
        base.mkdir()
        args = Namespace(base=str(base), overwrite=False)

        exit_code = run_init_command(args)
        assert exit_code == 0
        # 除了 DAYU_INIT_PROVIDER_OPTION（总是 persist 以记住用户选择）之外，
        # 其余 key 都已存在，不应再触发 _persist_env_var。
        assert persist_calls == ["DAYU_INIT_PROVIDER_OPTION"]

        # manifest 仍然被更新
        result_manifest = json.loads(
            (base / "config" / "prompts" / "manifests" / "write.json").read_text(encoding="utf-8")
        )
        assert result_manifest["model"]["default_name"] == "deepseek-v4-flash"

    def test_custom_openai_flow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """选择 custom-openai 时应写入 catalog、持久化 key 并更新 manifest。"""
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "run.json").write_text("{}", encoding="utf-8")
        manifests = src / "prompts" / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "mimo-v2.5-pro-plan", "allowed_names": ["mimo-v2.5-pro-plan"]}}),
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: src,
        )

        pkg_assets = tmp_path / "pkg_assets"
        pkg_assets.mkdir()
        (pkg_assets / "定性分析模板.md").write_text("# 模板", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_assets_path",
            lambda: pkg_assets,
        )

        _clear_provider_env_vars(monkeypatch)
        for key in ("TAVILY_API_KEY", "SERPER_API_KEY", "FMP_API_KEY", "HF_ENDPOINT", "HF_TOKEN"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(SEC_USER_AGENT_ENV, raising=False)
        monkeypatch.setattr("dayu.cli.commands.init._is_hf_hub_reachable", lambda: False)
        monkeypatch.setattr(
            "dayu.cli.commands.init._run_init_prewarm",
            lambda **_kwargs: (True, ""),
        )
        monkeypatch.setattr(
            "dayu.cli.commands.init._prompt_provider_selection",
            lambda: _PROVIDER_OPTION_CUSTOM_OPENAI,
        )

        persist_calls: list[tuple[str, str]] = []

        def _mock_persist(key: str, value: str) -> tuple[str, bool]:
            persist_calls.append((key, value))
            return "~/.zshrc", True

        monkeypatch.setattr("dayu.cli.commands.init._persist_env_var", _mock_persist)
        inputs = iter(
            [
                "https://openrouter.ai/api/v1/",
                "sk-custom-openrouter",
                "openai/gpt-4o",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

        base = tmp_path / "workspace"
        base.mkdir()
        args = Namespace(base=str(base), overwrite=False)

        exit_code = run_init_command(args)

        assert exit_code == 0
        # persist_calls[0] 是 DAYU_INIT_PROVIDER_OPTION（记住用户选择），
        # 接下来才是具体 API Key 的写入。
        assert persist_calls[0] == ("DAYU_INIT_PROVIDER_OPTION", "custom_openai")
        assert persist_calls[1] == ("CUSTOM_OPENAI_API_KEY", "sk-custom-openrouter")
        result_manifest = json.loads(
            (base / "config" / "prompts" / "manifests" / "write.json").read_text(encoding="utf-8")
        )
        assert result_manifest["model"]["default_name"] == _CUSTOM_CATALOG_KEY
        assert result_manifest["model"]["allowed_names"] == ["mimo-v2.5-pro-plan", _CUSTOM_CATALOG_KEY]
        llm_models = json.loads((base / "config" / "llm_models.json").read_text(encoding="utf-8"))
        assert llm_models[_CUSTOM_CATALOG_KEY]["model"] == "openai/gpt-4o"
        assert llm_models[_CUSTOM_CATALOG_KEY]["endpoint_url"] == "https://openrouter.ai/api/v1/chat/completions"


class TestUpdateManifestSecondInit:
    """二次 init 换供应商时 manifest 应被正确替换。"""

    def test_switch_from_deepseek_to_qwen(self, tmp_path: Path) -> None:
        """已经是 deepseek-v4-flash 的 manifest，换成 qwen-plus 应成功。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "deepseek-v4-flash", _INIT_ROLE_KEY: _ROLE_NON_THINKING}}),
            encoding="utf-8",
        )
        (manifests / "audit.json").write_text(
            json.dumps({"model": {"default_name": "deepseek-v4-flash-thinking", _INIT_ROLE_KEY: _ROLE_THINKING}}),
            encoding="utf-8",
        )

        count = _update_manifest_default_models(tmp_path, "qwen-plus", "qwen-plus-thinking")
        assert count == 2

        write_data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        assert write_data["model"]["default_name"] == "qwen-plus"

        audit_data = json.loads((manifests / "audit.json").read_text(encoding="utf-8"))
        assert audit_data["model"]["default_name"] == "qwen-plus-thinking"

    def test_ambiguous_model_uses_role_marker(self, tmp_path: Path) -> None:
        """thinking/non-thinking 同名模型（如 custom-openai）通过角色标记正确分类。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        # custom-openai 同时在 non-thinking 和 thinking 集合中
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "custom-openai", _INIT_ROLE_KEY: _ROLE_NON_THINKING}}),
            encoding="utf-8",
        )
        (manifests / "audit.json").write_text(
            json.dumps({"model": {"default_name": "custom-openai", _INIT_ROLE_KEY: _ROLE_THINKING}}),
            encoding="utf-8",
        )

        count = _update_manifest_default_models(tmp_path, "deepseek-v4-flash", "deepseek-v4-flash-thinking")
        assert count == 2

        write_data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        assert write_data["model"]["default_name"] == "deepseek-v4-flash"

        audit_data = json.loads((manifests / "audit.json").read_text(encoding="utf-8"))
        assert audit_data["model"]["default_name"] == "deepseek-v4-flash-thinking"

    def test_ambiguous_model_without_marker_uses_package_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """歧义模型名无角色标记时，回退到包内原始 manifest 推断角色。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        # 工作区 manifest：custom-openai 无标记，无法直接判断角色
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "custom-openai"}}),
            encoding="utf-8",
        )

        # 包内原始 manifest：write.json 的 default_name 是无歧义的 non-thinking 模型
        pkg_manifests = tmp_path / "pkg_config" / "prompts" / "manifests"
        pkg_manifests.mkdir(parents=True)
        (pkg_manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "mimo-v2.5-pro-plan"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: tmp_path / "pkg_config",
        )

        count = _update_manifest_default_models(tmp_path, "deepseek-v4-flash", "deepseek-v4-flash-thinking")
        assert count == 1

        data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        assert data["model"]["default_name"] == "deepseek-v4-flash"
        assert data["model"][_INIT_ROLE_KEY] == _ROLE_NON_THINKING

    def test_ambiguous_model_without_marker_no_package_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """歧义模型名无标记且包内无对应 manifest 时跳过。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)

        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "custom-openai"}}),
            encoding="utf-8",
        )

        # 包内无 manifests 目录
        pkg_config = tmp_path / "pkg_config"
        pkg_config.mkdir()
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: pkg_config,
        )

        count = _update_manifest_default_models(tmp_path, "deepseek-v4-flash", "deepseek-v4-flash-thinking")
        assert count == 0

        data = json.loads((manifests / "write.json").read_text(encoding="utf-8"))
        assert data["model"]["default_name"] == "custom-openai"

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
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: pkg_config,
        )

        count = _update_manifest_default_models(tmp_path, "qwen-plus", "qwen-plus-thinking")
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
        assert _classify_model_role("deepseek-v4-flash", "") == _ROLE_NON_THINKING

    def test_unambiguous_thinking(self) -> None:
        """仅在 thinking 集合中的模型名正确分类。"""
        assert _classify_model_role("deepseek-v4-flash-thinking", "") == _ROLE_THINKING

    def test_ambiguous_without_marker_returns_none(self) -> None:
        """歧义模型名无标记时返回 None。"""
        assert _classify_model_role("custom-openai", "") is None

    def test_unknown_model_returns_none(self) -> None:
        """完全未知的模型名返回 None。"""
        assert _classify_model_role("my-custom-model", "") is None


# --------------------------------------------------------------------------- #
#  _prompt_huggingface_config
# --------------------------------------------------------------------------- #


class TestPromptHuggingfaceConfig:
    """HuggingFace 镜像与 Token 配置测试。"""

    def test_both_already_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """HF_ENDPOINT 和 HF_TOKEN 均已设置时返回空列表。"""
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        monkeypatch.setenv("HF_TOKEN", "hf-xxx")

        result = _prompt_huggingface_config()
        assert result == []

    def test_accept_mirror_and_input_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hub 不可达，用户按回车接受默认 Y 并输入 token 时返回两项。"""
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("dayu.cli.commands.init._is_hf_hub_reachable", lambda: False)

        inputs = iter(["", "hf_test_token_123"])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

        result = _prompt_huggingface_config()
        assert ("HF_ENDPOINT", _HF_MIRROR_URL) in result
        assert ("HF_TOKEN", "hf_test_token_123") in result

    def test_decline_mirror_and_skip_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """用户选 n 并跳过 token 时返回空列表。"""
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("dayu.cli.commands.init._is_hf_hub_reachable", lambda: False)

        inputs = iter(["n", ""])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

        result = _prompt_huggingface_config()
        assert result == []

    def test_hub_reachable_default_no_mirror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hub 可达时，回车默认不启用镜像。"""
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("dayu.cli.commands.init._is_hf_hub_reachable", lambda: True)

        inputs = iter(["", ""])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

        result = _prompt_huggingface_config()
        assert ("HF_ENDPOINT", _HF_MIRROR_URL) not in result

    def test_hub_reachable_explicit_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Hub 可达但用户显式输入 y 时仍启用镜像。"""
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr("dayu.cli.commands.init._is_hf_hub_reachable", lambda: True)

        inputs = iter(["y", ""])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

        result = _prompt_huggingface_config()
        assert ("HF_ENDPOINT", _HF_MIRROR_URL) in result


# --------------------------------------------------------------------------- #
#  _write_env_to_shell_profile — content 不以换行结尾
# --------------------------------------------------------------------------- #


class TestWriteEnvNoTrailingNewline:
    """content 不以换行结尾时先写换行再追加。"""

    def test_append_when_content_lacks_trailing_newline(self, tmp_path: Path) -> None:
        """文件内容不以换行结尾时，追加前应先写入换行。"""
        profile = tmp_path / ".zshrc"
        profile.write_text("# existing", encoding="utf-8")

        result = _write_env_to_shell_profile("MY_KEY", "val", profile)
        assert result is True

        content = profile.read_text(encoding="utf-8")
        assert 'export MY_KEY="val"' in content


# --------------------------------------------------------------------------- #
#  _write_env_windows
# --------------------------------------------------------------------------- #


class TestWriteEnvWindows:
    """Windows setx 环境变量写入测试。"""

    def test_setx_success(self) -> None:
        """setx 返回码为 0 时返回 True。"""
        with patch("dayu.cli.commands.init.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 0
            assert _write_env_windows("MY_KEY", "my_value") is True

    def test_setx_failure(self) -> None:
        """setx 返回码非 0 时返回 False。"""
        with patch("dayu.cli.commands.init.subprocess.run") as mock_run:
            mock_run.return_value.returncode = 1
            assert _write_env_windows("MY_KEY", "my_value") is False


# --------------------------------------------------------------------------- #
#  _persist_env_var — Windows 与非兼容 shell 分支
# --------------------------------------------------------------------------- #


class TestPersistEnvVar:
    """跨平台环境变量持久化测试。"""

    def test_windows_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows 平台 setx 成功时返回正确描述。"""
        monkeypatch.setattr("dayu.cli.commands.init.platform.system", lambda: "Windows")
        with patch("dayu.cli.commands.init._write_env_windows", return_value=True):
            target, ok = _persist_env_var("MY_KEY", "val")
        assert ok is True
        assert "setx" in target

    def test_windows_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows 平台 setx 失败时返回 False。"""
        monkeypatch.setattr("dayu.cli.commands.init.platform.system", lambda: "Windows")
        with patch("dayu.cli.commands.init._write_env_windows", return_value=False):
            target, ok = _persist_env_var("MY_KEY", "val")
        assert ok is False
        assert target == "setx"

    def test_non_compatible_shell(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """明确识别为不兼容 shell 时返回 False。"""
        monkeypatch.setattr("dayu.cli.commands.init.platform.system", lambda: "Linux")
        monkeypatch.setattr(
            "dayu.cli.commands.init._detect_shell_profile",
            lambda: (Path.home() / ".profile", False),
        )
        target, ok = _persist_env_var("MY_KEY", "val")
        assert ok is False

    def test_write_profile_failure_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """profile 文件写入失败时返回 False，而不是依赖 SHELL 误判。"""
        monkeypatch.setattr("dayu.cli.commands.init.platform.system", lambda: "Linux")
        monkeypatch.setattr(
            "dayu.cli.commands.init._detect_shell_profile",
            lambda: (Path.home() / ".profile", True),
        )

        def _raise_os_error(_key: str, _value: str, _profile: Path) -> bool:
            raise OSError("read-only file system")

        monkeypatch.setattr("dayu.cli.commands.init._write_env_to_shell_profile", _raise_os_error)
        target, ok = _persist_env_var("MY_KEY", "val")
        assert target.endswith(".profile")
        assert ok is False

    def test_sets_process_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """始终设置当前进程环境变量。"""
        monkeypatch.setattr("dayu.cli.commands.init.platform.system", lambda: "Windows")
        with patch("dayu.cli.commands.init._write_env_windows", return_value=True):
            _persist_env_var("TEST_PERSIST_KEY", "test_val")
        assert __import__("os").environ.get("TEST_PERSIST_KEY") == "test_val"


# --------------------------------------------------------------------------- #
#  _resolve_role_from_package_manifest — 错误分支
# --------------------------------------------------------------------------- #


class TestResolveRoleFromPackageManifest:
    """包内 manifest 角色推断测试。"""

    def test_manifest_not_exists(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """包内 manifest 不存在时返回 None。"""
        pkg_config = tmp_path / "pkg_config"
        pkg_config.mkdir()
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: pkg_config,
        )
        assert _resolve_role_from_package_manifest("nonexistent.json") is None

    def test_manifest_invalid_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """包内 manifest JSON 解析失败时返回 None。"""
        pkg_manifests = tmp_path / "pkg_config" / "prompts" / "manifests"
        pkg_manifests.mkdir(parents=True)
        (pkg_manifests / "broken.json").write_text("not json {{{", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: tmp_path / "pkg_config",
        )
        assert _resolve_role_from_package_manifest("broken.json") is None

    def test_manifest_model_not_dict(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """包内 manifest 的 model 字段非 dict 时返回 None。"""
        pkg_manifests = tmp_path / "pkg_config" / "prompts" / "manifests"
        pkg_manifests.mkdir(parents=True)
        (pkg_manifests / "weird.json").write_text(json.dumps({"model": "string_instead_of_dict"}), encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: tmp_path / "pkg_config",
        )
        assert _resolve_role_from_package_manifest("weird.json") is None


# --------------------------------------------------------------------------- #
#  _update_manifest_default_models — 额外分支
# --------------------------------------------------------------------------- #


class TestUpdateManifestExtraBranches:
    """manifest 更新的额外分支覆盖。"""

    def test_model_section_not_dict_skipped(self, tmp_path: Path) -> None:
        """model section 非 dict 时跳过该文件。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "bad.json").write_text(json.dumps({"model": "not_a_dict"}), encoding="utf-8")
        count = _update_manifest_default_models(tmp_path, "deepseek-v4-flash", "deepseek-v4-flash-thinking")
        assert count == 0

    def test_stored_role_not_string_reset(self, tmp_path: Path) -> None:
        """stored_role 非字符串时当作空字符串处理。"""
        manifests = tmp_path / "prompts" / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "a.json").write_text(
            json.dumps({"model": {"default_name": "mimo-v2.5-pro-plan", _INIT_ROLE_KEY: 42}}),
            encoding="utf-8",
        )
        count = _update_manifest_default_models(tmp_path, "deepseek-v4-flash", "deepseek-v4-flash-thinking")
        assert count == 1
        data = json.loads((manifests / "a.json").read_text(encoding="utf-8"))
        assert data["model"]["default_name"] == "deepseek-v4-flash"


# --------------------------------------------------------------------------- #
#  _prompt_provider_selection — 额外分支
# --------------------------------------------------------------------------- #


class TestPromptProviderSelection:
    """供应商选择交互测试。"""

    def test_eof_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EOFError 时调用 sys.exit(1)。"""
        monkeypatch.setattr("builtins.input", lambda *_args: (_ for _ in ()).throw(EOFError))
        with pytest.raises(SystemExit) as exc_info:
            _prompt_provider_selection()
        assert exc_info.value.code == 1

    def test_keyboard_interrupt_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KeyboardInterrupt 时调用 sys.exit(1)。"""

        def _raise_kb(*_args: str) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", _raise_kb)
        with pytest.raises(SystemExit) as exc_info:
            _prompt_provider_selection()
        assert exc_info.value.code == 1

    def test_empty_input_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """空输入时选择默认供应商（第一个已配置的）。"""
        monkeypatch.setenv("MIMO_API_KEY", "existing")
        for k in (
            "MIMO_PLAN_API_KEY",
            "MIMO_PLAN_SG_API_KEY",
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "QWEN_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setattr("builtins.input", lambda *_args: "")
        result = _prompt_provider_selection()
        assert result == _PROVIDER_OPTION_MIMO_PRO

    def test_select_deepseek_flash_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """显式选择 DeepSeek Flash 方案（菜单第 5 项）时返回对应方案 key。"""
        _clear_provider_env_vars(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda *_args: "5")
        result = _prompt_provider_selection()
        assert result == _PROVIDER_OPTION_DEEPSEEK_FLASH

    def test_invalid_non_integer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """非整数输入时调用 sys.exit(1)。"""
        _clear_provider_env_vars(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda *_args: "abc")
        with pytest.raises(SystemExit) as exc_info:
            _prompt_provider_selection()
        assert exc_info.value.code == 1

    def test_out_of_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """编号超出范围时调用 sys.exit(1)。"""
        _clear_provider_env_vars(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda *_args: "999")
        with pytest.raises(SystemExit) as exc_info:
            _prompt_provider_selection()
        assert exc_info.value.code == 1

    def test_first_configured_provider_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """首个已配置供应商触发 first_configured_idx 分支。"""
        _clear_provider_env_vars(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-already")
        monkeypatch.setattr("builtins.input", lambda *_args: "")
        result = _prompt_provider_selection()
        # DeepSeek Pro 在 Flash 之前声明，共享同一 DEEPSEEK_API_KEY 时 Pro 先命中。
        assert result == _PROVIDER_OPTION_DEEPSEEK_PRO

    def test_saved_option_overrides_first_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DAYU_INIT_PROVIDER_OPTION 环境变量优先于「第一个已配置」作为默认推荐。

        场景：用户上次显式选了 DeepSeek Flash（值被记在 DAYU_INIT_PROVIDER_OPTION），
        同时环境变量里已有 DEEPSEEK_API_KEY；按 Enter 不应被 DeepSeek Pro（声明顺序
        靠前）静默抢走，而应保留 Flash。
        """
        _clear_provider_env_vars(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-already")
        monkeypatch.setenv("DAYU_INIT_PROVIDER_OPTION", _PROVIDER_OPTION_DEEPSEEK_FLASH)
        monkeypatch.setattr("builtins.input", lambda *_args: "")
        result = _prompt_provider_selection()
        assert result == _PROVIDER_OPTION_DEEPSEEK_FLASH

    def test_stale_saved_option_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DAYU_INIT_PROVIDER_OPTION 为失效 option_key 时回退到第一个已配置方案。"""
        _clear_provider_env_vars(monkeypatch)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-already")
        monkeypatch.setenv("DAYU_INIT_PROVIDER_OPTION", "mimo_flash_removed")
        monkeypatch.setattr("builtins.input", lambda *_args: "")
        result = _prompt_provider_selection()
        assert result == _PROVIDER_OPTION_DEEPSEEK_PRO


# --------------------------------------------------------------------------- #
#  _prompt_api_key — 分支覆盖
# --------------------------------------------------------------------------- #


class TestPromptApiKey:
    """API Key 输入交互测试。"""

    def test_eof_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EOFError 时调用 sys.exit(1)。"""
        monkeypatch.setattr("builtins.input", lambda *_args: (_ for _ in ()).throw(EOFError))
        with pytest.raises(SystemExit) as exc_info:
            _prompt_api_key("MY_KEY")
        assert exc_info.value.code == 1

    def test_keyboard_interrupt_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """KeyboardInterrupt 时调用 sys.exit(1)。"""

        def _raise_kb(*_args: str) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", _raise_kb)
        with pytest.raises(SystemExit) as exc_info:
            _prompt_api_key("MY_KEY")
        assert exc_info.value.code == 1

    def test_empty_value_exits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """用户输入空白时调用 sys.exit(1)。"""
        monkeypatch.setattr("builtins.input", lambda *_args: "   ")
        with pytest.raises(SystemExit) as exc_info:
            _prompt_api_key("MY_KEY")
        assert exc_info.value.code == 1


# --------------------------------------------------------------------------- #
#  _prompt_optional_search_keys — EOF 分支
# --------------------------------------------------------------------------- #


class TestPromptOptionalSearchKeys:
    """可选搜索 Key 输入测试。"""

    def test_eof_interrupts_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EOFError 时中断输入，返回已有结果。"""
        for k in ("TAVILY_API_KEY", "SERPER_API_KEY", "FMP_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        inputs = iter(["sk-test"])
        call_count = 0

        def _mock_input(prompt: str = "") -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise EOFError
            return next(inputs)

        monkeypatch.setattr("builtins.input", _mock_input)
        result = _prompt_optional_search_keys()
        assert len(result) >= 1


# --------------------------------------------------------------------------- #
#  _is_hf_hub_reachable — 异常分支
# --------------------------------------------------------------------------- #


class TestIsHfHubReachable:
    """HuggingFace Hub 可达性探测测试。"""

    def test_url_error_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """URLError 时返回 False。"""

        def _raise_url_error(*_args: str, **_kw: int) -> None:
            raise urllib.error.URLError("network down")

        monkeypatch.setattr("urllib.request.urlopen", _raise_url_error)
        assert _is_hf_hub_reachable() is False

    def test_os_error_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OSError 时返回 False。"""

        def _raise_os_error(*_args: str, **_kw: int) -> None:
            raise OSError("connection refused")

        monkeypatch.setattr("urllib.request.urlopen", _raise_os_error)
        assert _is_hf_hub_reachable() is False

    def test_timeout_error_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TimeoutError 时返回 False。"""

        def _raise_timeout(*_args: str, **_kw: int) -> None:
            raise TimeoutError("timed out")

        monkeypatch.setattr("urllib.request.urlopen", _raise_timeout)
        assert _is_hf_hub_reachable() is False


# --------------------------------------------------------------------------- #
#  _prompt_sec_user_agent
# --------------------------------------------------------------------------- #


class TestPromptSecUserAgent:
    """SEC User-Agent 交互提示测试。"""

    def test_returns_none_when_env_already_exists(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """环境变量已存在时直接跳过交互。"""

        monkeypatch.setenv(SEC_USER_AGENT_ENV, "Demo demo@example.com")
        monkeypatch.setattr("builtins.input", lambda *_args: (_ for _ in ()).throw(AssertionError("should not prompt")))

        assert _prompt_sec_user_agent() is None

    def test_returns_value_when_user_enters_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """用户输入有效 User-Agent 时返回键值对。"""

        monkeypatch.delenv(SEC_USER_AGENT_ENV, raising=False)
        monkeypatch.setattr("builtins.input", lambda *_args: "Demo demo@example.com")

        assert _prompt_sec_user_agent() == (SEC_USER_AGENT_ENV, "Demo demo@example.com")

    def test_returns_none_when_user_skips(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """用户直接回车时返回 None。"""

        monkeypatch.delenv(SEC_USER_AGENT_ENV, raising=False)
        monkeypatch.setattr("builtins.input", lambda *_args: "")

        assert _prompt_sec_user_agent() is None


# --------------------------------------------------------------------------- #
#  run_init_command — 持久化失败路径与 Windows 平台分支
# --------------------------------------------------------------------------- #


class TestRunInitFailurePaths:
    """run_init_command 失败路径与平台分支测试。"""

    def _prepare_mocks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """准备通用 mock 环境，返回 base 目录。"""
        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "run.json").write_text("{}", encoding="utf-8")
        manifests = src / "prompts" / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "mimo-v2.5-pro-plan"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: src,
        )
        pkg_assets = tmp_path / "pkg_assets"
        pkg_assets.mkdir()
        (pkg_assets / "template.md").write_text("# t", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_assets_path",
            lambda: pkg_assets,
        )
        _clear_provider_env_vars(monkeypatch)
        for k in ("TAVILY_API_KEY", "SERPER_API_KEY", "FMP_API_KEY", "HF_ENDPOINT", "HF_TOKEN"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv(SEC_USER_AGENT_ENV, raising=False)
        monkeypatch.setattr("dayu.cli.commands.init._is_hf_hub_reachable", lambda: True)
        monkeypatch.setattr(
            "dayu.cli.commands.init._run_init_prewarm",
            lambda **_kwargs: (True, ""),
        )
        base = tmp_path / "workspace"
        base.mkdir()
        return base

    def test_main_key_persist_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """主 API Key 持久化失败时跳过 manifest 更新并返回 1。"""
        base = self._prepare_mocks(tmp_path, monkeypatch)
        prewarm_calls: list[tuple[Path, Path]] = []

        inputs = iter(["sk-test", "", "", "", "n", "", ""])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)
        monkeypatch.setattr(
            "dayu.cli.commands.init._persist_env_var",
            lambda _k, _v: ("setx", False),
        )

        def _capture_prewarm(*, base_dir: Path, config_dir: Path) -> tuple[bool, str]:
            prewarm_calls.append((base_dir, config_dir))
            return True, ""

        monkeypatch.setattr("dayu.cli.commands.init._run_init_prewarm", _capture_prewarm)

        args = Namespace(base=str(base), overwrite=False)
        exit_code = run_init_command(args)
        assert exit_code == 1
        assert prewarm_calls == []

    def test_search_key_persist_fails(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """搜索 Key 持久化失败时触发 search_persist_failed 分支。"""
        base = self._prepare_mocks(tmp_path, monkeypatch)

        inputs = iter(["sk-test", "tvly-test", "", "", "n", "", ""])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)

        call_count = 0

        def _mock_persist(k: str, v: str) -> tuple[str, bool]:
            nonlocal call_count
            call_count += 1
            # 第 1 次 persist 是 DAYU_INIT_PROVIDER_OPTION（记住选择），第 2 次是 DEEPSEEK_API_KEY，
            # 从第 3 次起（搜索 key）开始模拟失败。
            if call_count >= 3:
                return "setx", False
            return "~/.zshrc", True

        monkeypatch.setattr("dayu.cli.commands.init._persist_env_var", _mock_persist)

        args = Namespace(base=str(base), overwrite=False)
        exit_code = run_init_command(args)
        assert exit_code == 0

    def test_custom_openai_invalid_catalog_json_returns_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """custom-openai 遇到损坏的 llm_models.json 时应打印错误并返回 1。"""
        base = self._prepare_mocks(tmp_path, monkeypatch)
        config_dir = base / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "llm_models.json").write_text("{broken json", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init._prompt_provider_selection",
            lambda: _PROVIDER_OPTION_CUSTOM_OPENAI,
        )
        inputs = iter(["https://openrouter.ai/api/v1", "sk-custom", "openai/gpt-4o"])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))

        args = Namespace(base=str(base), overwrite=False)
        exit_code = run_init_command(args)

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "不是合法 JSON" in captured.out


class TestInitPrewarm:
    """`dayu-cli init` 首次安装 prewarm 行为测试。"""

    def _prepare_run_init_environment(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """准备 `run_init_command` 所需的最小包内配置与资产。"""

        src = tmp_path / "pkg_config"
        src.mkdir()
        (src / "run.json").write_text("{}", encoding="utf-8")
        manifests = src / "prompts" / "manifests"
        manifests.mkdir(parents=True)
        (manifests / "write.json").write_text(
            json.dumps({"model": {"default_name": "mimo-v2.5-pro-plan"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_config_path",
            lambda: src,
        )

        pkg_assets = tmp_path / "pkg_assets"
        pkg_assets.mkdir()
        (pkg_assets / "template.md").write_text("# t", encoding="utf-8")
        monkeypatch.setattr(
            "dayu.cli.commands.init.resolve_package_assets_path",
            lambda: pkg_assets,
        )

        _clear_provider_env_vars(monkeypatch)
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv(SEC_USER_AGENT_ENV, raising=False)
        monkeypatch.setattr("dayu.cli.commands.init._is_hf_hub_reachable", lambda: True)
        init_inputs = iter(["sk-test", "", "", "", "n", "", ""])
        monkeypatch.setattr("builtins.input", lambda *_args: next(init_inputs))
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)
        monkeypatch.setattr(
            "dayu.cli.commands.init._persist_env_var",
            lambda _k, _v: ("~/.zshrc", True),
        )

        base = tmp_path / "workspace"
        base.mkdir()
        return base

    def test_should_run_init_prewarm_only_for_first_init(self, tmp_path: Path) -> None:
        """仅首次初始化允许执行 prewarm。"""

        base = tmp_path / "workspace"
        base.mkdir()

        assert (
            _should_run_init_prewarm(
                is_first_workspace_init=True,
                overwrite=False,
                main_key_persist_failed=False,
            )
            is True
        )

        config_dir = base / "config"
        config_dir.mkdir()
        assert (
            _should_run_init_prewarm(
                is_first_workspace_init=False,
                overwrite=False,
                main_key_persist_failed=False,
            )
            is False
        )
        assert (
            _should_run_init_prewarm(
                is_first_workspace_init=False,
                overwrite=True,
                main_key_persist_failed=False,
            )
            is False
        )
        assert (
            _should_run_init_prewarm(
                is_first_workspace_init=False,
                overwrite=False,
                main_key_persist_failed=True,
            )
            is False
        )

    def test_run_init_prewarm_only_imports_runtime_modules(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """prewarm 只做模块导入预热，不执行运行时装配。"""

        imported_modules: list[str] = []

        def _capture_import(module_name: str) -> object:
            imported_modules.append(module_name)
            return object()

        monkeypatch.setattr("dayu.cli.commands.init.importlib.import_module", _capture_import)

        success, message = _run_init_prewarm(
            base_dir=tmp_path / "workspace",
            config_dir=tmp_path / "workspace" / "config",
        )

        assert success is True
        assert message == ""
        assert imported_modules == [
            "dayu.cli.dependency_setup",
            "dayu.cli.interactive_ui",
            "dayu.cli.commands.interactive",
            "dayu.cli.commands.prompt",
            "dayu.cli.commands.write",
        ]

    def test_run_init_runs_prewarm_only_on_first_install(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """首次安装时执行 prewarm，二次 init 不再重复执行。"""

        base = self._prepare_run_init_environment(tmp_path, monkeypatch)
        prewarm_calls: list[tuple[Path, Path]] = []

        def _capture_prewarm(*, base_dir: Path, config_dir: Path) -> tuple[bool, str]:
            prewarm_calls.append((base_dir, config_dir))
            return True, ""

        monkeypatch.setattr("dayu.cli.commands.init._run_init_prewarm", _capture_prewarm)

        assert run_init_command(Namespace(base=str(base), overwrite=False)) == 0
        assert prewarm_calls == [(base.resolve(), (base / "config").resolve())]

        monkeypatch.setattr("builtins.input", lambda *_args: "")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-already-set-123456")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")
        monkeypatch.setenv("SERPER_API_KEY", "sp-xxx")
        monkeypatch.setenv("FMP_API_KEY", "fmp-xxx")
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        monkeypatch.setenv("HF_TOKEN", "hf-test-xxx")
        monkeypatch.setenv(SEC_USER_AGENT_ENV, "Demo demo@example.com")
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)
        assert run_init_command(Namespace(base=str(base), overwrite=False)) == 0
        assert prewarm_calls == [(base.resolve(), (base / "config").resolve())]

    def test_run_init_skips_prewarm_when_overwrite_is_true(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`--overwrite` 不属于首次安装，不执行 prewarm。"""

        base = self._prepare_run_init_environment(tmp_path, monkeypatch)
        prewarm_calls: list[tuple[Path, Path]] = []

        def _capture_prewarm(*, base_dir: Path, config_dir: Path) -> tuple[bool, str]:
            prewarm_calls.append((base_dir, config_dir))
            return True, ""

        monkeypatch.setattr("dayu.cli.commands.init._run_init_prewarm", _capture_prewarm)

        assert run_init_command(Namespace(base=str(base), overwrite=False)) == 0
        assert len(prewarm_calls) == 1

        monkeypatch.setattr("builtins.input", lambda *_args: "")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-already-set-123456")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")
        monkeypatch.setenv("SERPER_API_KEY", "sp-xxx")
        monkeypatch.setenv("FMP_API_KEY", "fmp-xxx")
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        monkeypatch.setenv("HF_TOKEN", "hf-test-xxx")
        monkeypatch.setenv(SEC_USER_AGENT_ENV, "Demo demo@example.com")
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)
        assert run_init_command(Namespace(base=str(base), overwrite=True)) == 0
        assert len(prewarm_calls) == 1

    def test_run_init_reset_rebuilds_workspace_and_runs_prewarm(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`--reset` 应清空旧工作区后按首次初始化重新预热。"""

        base = self._prepare_run_init_environment(tmp_path, monkeypatch)
        assert run_init_command(Namespace(base=str(base), overwrite=False, reset=False)) == 0

        stale_config_file = base / "config" / "stale.json"
        stale_assets_file = base / "assets" / "stale.md"
        stale_dayu_file = base / ".dayu" / "interactive" / "state.json"
        stale_config_file.write_text("{}", encoding="utf-8")
        stale_assets_file.write_text("# stale", encoding="utf-8")
        stale_dayu_file.parent.mkdir(parents=True, exist_ok=True)
        stale_dayu_file.write_text("{}", encoding="utf-8")

        prewarm_calls: list[tuple[Path, Path]] = []

        def _capture_prewarm(*, base_dir: Path, config_dir: Path) -> tuple[bool, str]:
            prewarm_calls.append((base_dir, config_dir))
            return True, ""

        monkeypatch.setattr("dayu.cli.commands.init._run_init_prewarm", _capture_prewarm)
        monkeypatch.setattr("builtins.input", lambda *_args: "y")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-already-set-123456")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")
        monkeypatch.setenv("SERPER_API_KEY", "sp-xxx")
        monkeypatch.setenv("FMP_API_KEY", "fmp-xxx")
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        monkeypatch.setenv("HF_TOKEN", "hf-test-xxx")
        monkeypatch.setenv(SEC_USER_AGENT_ENV, "Demo demo@example.com")
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)

        assert run_init_command(Namespace(base=str(base), overwrite=False, reset=True)) == 0
        assert prewarm_calls == [(base.resolve(), (base / "config").resolve())]
        assert not stale_config_file.exists()
        assert not stale_assets_file.exists()
        assert not stale_dayu_file.exists()

    def test_run_init_reset_cancelled_by_default_no(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`--reset` 直接回车时应取消，不删除已有目录。"""

        base = self._prepare_run_init_environment(tmp_path, monkeypatch)
        (base / "config").mkdir(parents=True, exist_ok=True)
        (base / "assets").mkdir(parents=True, exist_ok=True)
        (base / ".dayu").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("builtins.input", lambda *_args: "")

        prewarm_calls: list[tuple[Path, Path]] = []

        def _capture_prewarm(*, base_dir: Path, config_dir: Path) -> tuple[bool, str]:
            prewarm_calls.append((base_dir, config_dir))
            return True, ""

        monkeypatch.setattr("dayu.cli.commands.init._run_init_prewarm", _capture_prewarm)

        assert run_init_command(Namespace(base=str(base), overwrite=False, reset=True)) == 1
        assert (base / "config").exists()
        assert (base / "assets").exists()
        assert (base / ".dayu").exists()
        assert prewarm_calls == []

    def test_run_init_reset_noop_message_uses_reset_targets(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """空 reset 时提示文案应基于 reset 目标列表生成。"""

        base = self._prepare_run_init_environment(tmp_path, monkeypatch)
        monkeypatch.setattr("builtins.input", lambda *_args: "y")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-already-set-123456")
        monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")
        monkeypatch.setenv("SERPER_API_KEY", "sp-xxx")
        monkeypatch.setenv("FMP_API_KEY", "fmp-xxx")
        monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com")
        monkeypatch.setenv("HF_TOKEN", "hf-test-xxx")
        monkeypatch.setenv(SEC_USER_AGENT_ENV, "Demo demo@example.com")
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)
        monkeypatch.setattr(
            "dayu.cli.commands.init._run_init_prewarm",
            lambda **_kwargs: (True, ""),
        )

        assert run_init_command(Namespace(base=str(base), overwrite=False, reset=True)) == 0

        captured = capsys.readouterr()
        assert "未发现需要删除的 .dayu、config、assets" in captured.out

    def test_run_init_prewarm_failure_only_warns(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """prewarm 失败只告警，不影响 init 主成功码。"""

        base = self._prepare_run_init_environment(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "dayu.cli.commands.init._run_init_prewarm",
            lambda **_kwargs: (False, "RuntimeError: prewarm failed"),
        )

        assert run_init_command(Namespace(base=str(base), overwrite=False)) == 0

        captured = capsys.readouterr()
        assert "正在预热 CLI 运行时，请稍候" in captured.out
        assert "CLI 运行时预热失败: RuntimeError: prewarm failed" in captured.out
        assert "不影响当前工作区初始化成功" in captured.out

    def test_run_init_skips_prewarm_when_main_key_persist_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """主 API Key 持久化失败时不执行 prewarm。"""

        base = self._prepare_run_init_environment(tmp_path, monkeypatch)
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)
        monkeypatch.setattr("dayu.cli.commands.init._prompt_api_key", lambda _key: "sk-test")
        monkeypatch.setattr("dayu.cli.commands.init._persist_env_var", lambda _k, _v: ("~/.zshrc", False))
        monkeypatch.setattr("dayu.cli.commands.init._prompt_optional_search_keys", lambda: [])
        monkeypatch.setattr("dayu.cli.commands.init._prompt_huggingface_config", lambda: [])

        prewarm_calls: list[tuple[Path, Path]] = []

        def _capture_prewarm(*, base_dir: Path, config_dir: Path) -> tuple[bool, str]:
            prewarm_calls.append((base_dir, config_dir))
            return True, ""

        monkeypatch.setattr("dayu.cli.commands.init._run_init_prewarm", _capture_prewarm)

        assert run_init_command(Namespace(base=str(base), overwrite=False)) == 1
        assert prewarm_calls == []

    def test_windows_setx_message(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Windows 平台持久化成功时打印 setx 提示。"""
        base = self._prepare_run_init_environment(tmp_path, monkeypatch)
        monkeypatch.setattr("dayu.cli.commands.init.platform.system", lambda: "Windows")
        monkeypatch.setattr(
            "dayu.cli.commands.init._run_init_prewarm",
            lambda **_kwargs: (True, ""),
        )

        inputs = iter(["sk-test", "", "", "", "n", "", ""])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)
        monkeypatch.setattr(
            "dayu.cli.commands.init._persist_env_var",
            lambda _k, _v: ("setx（重开终端生效）", True),
        )

        args = Namespace(base=str(base), overwrite=False)
        exit_code = run_init_command(args)
        assert exit_code == 0

    def test_hf_persist_fails_sets_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """HF 环境变量持久化失败时也触发 search_persist_failed。"""
        base = self._prepare_run_init_environment(tmp_path, monkeypatch)
        monkeypatch.setattr(
            "dayu.cli.commands.init._run_init_prewarm",
            lambda **_kwargs: (True, ""),
        )

        inputs = iter(["sk-test", "", "", "", "y", "hf-token-val", ""])
        monkeypatch.setattr("builtins.input", lambda *_args: next(inputs))
        monkeypatch.setattr("dayu.cli.commands.init._prompt_provider_selection", lambda: _PROVIDER_OPTION_DEEPSEEK_FLASH)

        call_count = 0

        def _mock_persist(k: str, v: str) -> tuple[str, bool]:
            nonlocal call_count
            call_count += 1
            # 第 1 次是 DAYU_INIT_PROVIDER_OPTION、第 2 次是 DEEPSEEK_API_KEY，
            # 从第 3 次起（HF 相关 key）开始模拟失败。
            if call_count >= 3:
                return "setx", False
            return "~/.zshrc", True

        monkeypatch.setattr("dayu.cli.commands.init._persist_env_var", _mock_persist)

        args = Namespace(base=str(base), overwrite=False)
        exit_code = run_init_command(args)
        assert exit_code == 0
