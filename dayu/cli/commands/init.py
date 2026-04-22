"""``dayu-cli init`` 子命令实现。

模块职责：
- 复制包内配置到工作区
- 交互式选择初始化模型方案 → 配置 API Key → 更新 manifest 默认模型
- 可选配置联网检索 API Key
- 跨平台环境变量持久化（macOS/Linux shell profile，Windows setx）
"""

from __future__ import annotations

import json
import importlib
import os
import platform
import re
import shutil
import subprocess
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
import urllib.error
import urllib.request
from dayu.cli.workspace_migrations import apply_all_workspace_migrations
from dayu.startup.config_file_resolver import resolve_package_assets_path, resolve_package_config_path
from dayu.contracts.env_keys import (
    FMP_API_KEY_ENV,
    SEC_USER_AGENT_ENV,
    SERPER_API_KEY_ENV,
    TAVILY_API_KEY_ENV,
)
from dayu.workspace_paths import build_dayu_root_path

MODULE = "CLI.INIT"

# --------------------------------------------------------------------------- #
#  供应商定义
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _ProviderOption:
    """`dayu-cli init` 中的单个初始化模型方案定义。"""

    option_key: str
    display_name: str
    api_key_name: str
    non_thinking_model: str
    thinking_model: str


_PROVIDER_OPTION_MIMO_PLAN = "mimo_plan"
_PROVIDER_OPTION_MIMO_PLAN_SG = "mimo_plan_sg"
_PROVIDER_OPTION_MIMO_PRO = "mimo_pro"
_PROVIDER_OPTION_MIMO_FLASH = "mimo_flash"
_PROVIDER_OPTION_DEEPSEEK = "deepseek"
_PROVIDER_OPTION_OPENAI = "openai"
_PROVIDER_OPTION_ANTHROPIC = "anthropic"
_PROVIDER_OPTION_GEMINI = "gemini"
_PROVIDER_OPTION_QWEN = "qwen"
_PROVIDER_OPTION_CUSTOM_OPENAI = "custom_openai"

# 自定义 OpenAI 兼容 API（OpenRouter 等）统一使用的目录键
_CUSTOM_CATALOG_KEY = "custom-openai"
_CUSTOM_OPENAI_TEMPERATURE_PROFILES: dict[str, dict[str, float]] = {
    "write": {"temperature": 1.0},
    "overview": {"temperature": 1.0},
    "audit": {"temperature": 0.8},
    "decision": {"temperature": 1.0},
    "interactive": {"temperature": 1.0},
    "prompt": {"temperature": 1.0},
    "infer": {"temperature": 0.5},
    "conversation_compaction": {"temperature": 0.4},
}
_CUSTOM_OPENAI_CONVERSATION_MEMORY: dict[str, int] = {
    "episodic_memory_token_budget_floor": 4000,
    "episodic_memory_token_budget_cap": 4000,
}

_PROVIDER_OPTIONS: tuple[_ProviderOption, ...] = (
    _ProviderOption(
        option_key=_PROVIDER_OPTION_MIMO_PLAN,
        display_name="Mimo Token Plan",
        api_key_name="MIMO_PLAN_API_KEY",
        non_thinking_model="mimo-v2-pro-plan",
        thinking_model="mimo-v2-pro-thinking-plan",
    ),
    _ProviderOption(
        option_key=_PROVIDER_OPTION_MIMO_PLAN_SG,
        display_name="Mimo Token Plan (海外)",
        api_key_name="MIMO_PLAN_SG_API_KEY",
        non_thinking_model="mimo-v2-pro-plan-sg",
        thinking_model="mimo-v2-pro-thinking-plan-sg",
    ),
    _ProviderOption(
        option_key=_PROVIDER_OPTION_MIMO_PRO,
        display_name="Mimo Pro（常规 API）",
        api_key_name="MIMO_API_KEY",
        non_thinking_model="mimo-v2-pro",
        thinking_model="mimo-v2-pro-thinking",
    ),
    _ProviderOption(
        option_key=_PROVIDER_OPTION_MIMO_FLASH,
        display_name="Mimo Flash（常规 API）",
        api_key_name="MIMO_API_KEY",
        non_thinking_model="mimo-v2-flash",
        thinking_model="mimo-v2-flash-thinking",
    ),
    _ProviderOption(
        option_key=_PROVIDER_OPTION_DEEPSEEK,
        display_name="DeepSeek",
        api_key_name="DEEPSEEK_API_KEY",
        non_thinking_model="deepseek-chat",
        thinking_model="deepseek-thinking",
    ),
    _ProviderOption(
        option_key=_PROVIDER_OPTION_OPENAI,
        display_name="OpenAI",
        api_key_name="OPENAI_API_KEY",
        non_thinking_model="gpt-5.4",
        thinking_model="gpt-5.4",
    ),
    _ProviderOption(
        option_key=_PROVIDER_OPTION_ANTHROPIC,
        display_name="Anthropic",
        api_key_name="ANTHROPIC_API_KEY",
        non_thinking_model="claude-sonnet-4-6",
        thinking_model="claude-sonnet-4-6",
    ),
    _ProviderOption(
        option_key=_PROVIDER_OPTION_GEMINI,
        display_name="Google Gemini",
        api_key_name="GEMINI_API_KEY",
        non_thinking_model="gemini-2.5-flash",
        thinking_model="gemini-2.5-flash",
    ),
    _ProviderOption(
        option_key=_PROVIDER_OPTION_QWEN,
        display_name="阿里通义千问",
        api_key_name="QWEN_API_KEY",
        non_thinking_model="qwen3",
        thinking_model="qwen3-thinking",
    ),
    _ProviderOption(
        option_key=_PROVIDER_OPTION_CUSTOM_OPENAI,
        display_name="自定义 OpenAI 兼容 API（OpenRouter 等）",
        api_key_name="CUSTOM_OPENAI_API_KEY",
        non_thinking_model=_CUSTOM_CATALOG_KEY,
        thinking_model=_CUSTOM_CATALOG_KEY,
    ),
)

_PROVIDER_OPTIONS_BY_KEY: dict[str, _ProviderOption] = {option.option_key: option for option in _PROVIDER_OPTIONS}

# 从初始化方案推导：所有可能出现在 manifest default_name 中的 non-thinking 模型名集合
_ALL_NON_THINKING_MODELS: frozenset[str] = frozenset(option.non_thinking_model for option in _PROVIDER_OPTIONS)

# 所有可能出现的 thinking 模型名集合
_ALL_THINKING_MODELS: frozenset[str] = frozenset(option.thinking_model for option in _PROVIDER_OPTIONS)

# 仅出现在 non-thinking 集合而不在 thinking 集合中的模型名
_ONLY_NON_THINKING: frozenset[str] = _ALL_NON_THINKING_MODELS - _ALL_THINKING_MODELS

# 仅出现在 thinking 集合而不在 non-thinking 集合中的模型名
_ONLY_THINKING: frozenset[str] = _ALL_THINKING_MODELS - _ALL_NON_THINKING_MODELS

# init 在 manifest 中写入的角色标记 key
_INIT_ROLE_KEY = "_init_model_role"

_ROLE_NON_THINKING = "non_thinking"
_ROLE_THINKING = "thinking"

_OPTIONAL_SEARCH_KEYS: list[str] = [
    TAVILY_API_KEY_ENV,
    SERPER_API_KEY_ENV,
    FMP_API_KEY_ENV,
]

_HF_MIRROR_URL = "https://hf-mirror.com"

_HF_PROBE_URL = "https://huggingface.co"
_HF_PROBE_TIMEOUT_SECONDS = 5

_PREWARM_MODULES: tuple[str, ...] = (
    "dayu.cli.dependency_setup",
    "dayu.cli.interactive_ui",
    "dayu.cli.commands.interactive",
    "dayu.cli.commands.prompt",
    "dayu.cli.commands.write",
)

# --------------------------------------------------------------------------- #
#  环境变量持久化
# --------------------------------------------------------------------------- #


def _detect_shell_profile() -> tuple[Path, bool]:
    """检测当前用户的 shell profile 路径。

    优先依据 ``SHELL`` 推断 profile 类型；若 ``SHELL`` 缺失，则回退到
    当前 ``HOME`` 下已有的 profile 文件。只有在明确识别为 ``fish`` / ``nu``
    这类不兼容 ``export KEY=value`` 语法的 shell 时，才返回不兼容标记。

    Returns:
        ``(profile_path, is_export_compatible)`` 元组。
        ``is_export_compatible`` 为 ``True`` 表示可以安全写入
        ``export KEY=value`` 形式的 profile 文件。
        为 ``False`` 表示 shell 明确不兼容该语法，调用方应提示用户手动配置。

    Raises:
        无。
    """

    shell = os.environ.get("SHELL", "").lower()
    home = Path.home()
    zshrc = home / ".zshrc"
    bashrc = home / ".bashrc"
    bash_profile = home / ".bash_profile"
    profile = home / ".profile"

    if "zsh" in shell:
        return zshrc, True
    if "bash" in shell:
        return (bash_profile if bash_profile.exists() else bashrc), True
    if "fish" in shell or "nushell" in shell or shell.endswith("/nu"):
        return profile, False

    if zshrc.exists():
        return zshrc, True
    if bash_profile.exists():
        return bash_profile, True
    if bashrc.exists():
        return bashrc, True
    return profile, True


def _write_env_to_shell_profile(key: str, value: str, profile: Path) -> bool:
    """将 ``export KEY=value`` 追加到 shell profile。

    如果 profile 中已存在同名 export 行，则替换。

    Args:
        key: 环境变量名。
        value: 环境变量值。
        profile: shell profile 路径。

    Returns:
        是否实际写入（``True`` 表示写入成功，``False`` 表示已存在相同值）。

    Raises:
        无。
    """

    export_line = f'export {key}="{value}"'
    pattern = re.compile(rf"^export\s+{re.escape(key)}=.*$", re.MULTILINE)

    if profile.exists():
        content = profile.read_text(encoding="utf-8")
        if pattern.search(content):
            old_match = pattern.search(content)
            if old_match and old_match.group(0) == export_line:
                return False
            new_content = pattern.sub(export_line, content)
            profile.write_text(new_content, encoding="utf-8")
            return True
    else:
        content = ""

    with profile.open("a", encoding="utf-8") as f:
        if content and not content.endswith("\n"):
            f.write("\n")
        f.write(f"\n# dayu-cli init 自动写入\n{export_line}\n")
    return True


def _write_env_windows(key: str, value: str) -> bool:
    """在 Windows 上通过 setx 持久化环境变量。

    Args:
        key: 环境变量名。
        value: 环境变量值。

    Returns:
        是否执行成功。

    Raises:
        无。
    """

    result = subprocess.run(
        ["setx", key, value],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _persist_env_var(key: str, value: str) -> tuple[str, bool]:
    """跨平台持久化环境变量。

    同时设置当前进程的环境变量。

    Args:
        key: 环境变量名。
        value: 环境变量值。

    Returns:
        ``(target_description, success)`` 元组。
        ``target_description`` 为写入目标描述文本（如 ``~/.zshrc``、``setx``）。
        ``success`` 为 ``False`` 表示持久化失败或 shell 不兼容，需要用户手动配置。

    Raises:
        无。
    """

    os.environ[key] = value

    if platform.system() == "Windows":
        ok = _write_env_windows(key, value)
        if not ok:
            return "setx", False
        return "setx（重开终端生效）", True

    profile, is_compatible = _detect_shell_profile()
    if not is_compatible:
        return str(profile), False

    try:
        _write_env_to_shell_profile(key, value, profile)
    except OSError:
        return str(profile), False
    return str(profile), True


# --------------------------------------------------------------------------- #
#  工作区重置
# --------------------------------------------------------------------------- #


def _build_workspace_reset_targets(base_dir: Path) -> tuple[Path, ...]:
    """构造 `init --reset` 需要处理的工作区目标路径列表。

    Args:
        base_dir: 工作区根目录。

    Returns:
        需要参与 reset 的目标路径元组。

    Raises:
        无。
    """

    return (
        build_dayu_root_path(base_dir),
        base_dir / "config",
        base_dir / "assets",
    )


def _confirm_workspace_reset(base_dir: Path) -> bool:
    """交互确认是否执行工作区重置。

    Args:
        base_dir: 工作区根目录。

    Returns:
        用户明确输入 ``y`` / ``yes`` 时返回 `True`；其余输入（含回车）返回 `False`。

    Raises:
        无。输入流中断时按未确认处理。
    """

    targets = _build_workspace_reset_targets(base_dir)
    target_lines = "\n".join(f"  - {target}" for target in targets)
    prompt = (
        "\n⚠️  --reset 将删除以下目录并重新初始化：\n"
        f"{target_lines}\n"
        "是否继续？(y/N): "
    )
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer in ("y", "yes")


def _remove_workspace_target(target: Path) -> bool:
    """删除工作区中的单个初始化目标路径。

    Args:
        target: 需要删除的目标路径，可能是目录、文件或符号链接。

    Returns:
        若目标存在且已删除则返回 `True`；若目标原本不存在则返回 `False`。

    Raises:
        OSError: 当文件系统删除失败时抛出。
    """

    if target.is_symlink() or target.is_file():
        target.unlink()
        return True
    if target.is_dir():
        shutil.rmtree(target)
        return True
    return False


def _reset_workspace_init_targets(base_dir: Path) -> tuple[Path, ...]:
    """删除 `init` 管理的工作区初始化产物与运行时状态。

    Args:
        base_dir: 工作区根目录。

    Returns:
        实际被删除的目标路径元组，顺序与删除顺序一致。

    Raises:
        OSError: 当任一目标删除失败时抛出。
    """

    targets = _build_workspace_reset_targets(base_dir)
    removed_targets: list[Path] = []
    for target in targets:
        if _remove_workspace_target(target):
            removed_targets.append(target)
    return tuple(removed_targets)


# --------------------------------------------------------------------------- #
#  配置复制
# --------------------------------------------------------------------------- #


def _copy_config(base_dir: Path, *, overwrite: bool) -> Path:
    """复制包内配置到工作区。

    Args:
        base_dir: 工作区根目录。
        overwrite: 是否覆盖已有文件。

    Returns:
        目标配置目录路径。

    Raises:
        无。
    """

    src = resolve_package_config_path()
    dst = (base_dir / "config").resolve()

    if dst.exists() and not overwrite:
        synced_count = _sync_missing_prompt_assets(src, dst)
        print(f"配置目录已存在: {dst}（使用 --overwrite 覆盖）")
        if synced_count > 0:
            print(f"已补齐 {synced_count} 个缺失 prompt 资产")
        return dst

    if dst.exists() and overwrite:
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    return dst


def _sync_missing_prompt_assets(package_config_dir: Path, workspace_config_dir: Path) -> int:
    """为已有工作区增量补齐缺失的 prompt 资产。

    仅同步 ``config/prompts`` 子树中包内新增、而工作区当前不存在的文件。
    已存在的工作区文件一律保留，不做覆盖，以避免破坏用户本地定制配置。

    Args:
        package_config_dir: 包内默认配置根目录。
        workspace_config_dir: 工作区配置根目录。

    Returns:
        实际补齐的文件数量。

    Raises:
        无。
    """

    package_prompts_dir = package_config_dir / "prompts"
    workspace_prompts_dir = workspace_config_dir / "prompts"
    if not package_prompts_dir.exists():
        return 0

    created_count = 0
    for source_file in sorted(package_prompts_dir.rglob("*")):
        if not source_file.is_file():
            continue
        relative_path = source_file.relative_to(package_prompts_dir)
        target_file = workspace_prompts_dir / relative_path
        if target_file.exists():
            continue
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        created_count += 1
    return created_count


def _copy_assets(base_dir: Path, *, overwrite: bool) -> Path:
    """复制包内 assets 到工作区。

    Args:
        base_dir: 工作区根目录。
        overwrite: 是否覆盖已有文件。

    Returns:
        目标 assets 目录路径。

    Raises:
        无。
    """

    src = resolve_package_assets_path()
    dst = (base_dir / "assets").resolve()

    if dst.exists() and not overwrite:
        print(f"assets 目录已存在: {dst}（使用 --overwrite 覆盖）")
        return dst

    if dst.exists() and overwrite:
        shutil.rmtree(dst)

    shutil.copytree(src, dst)
    return dst


def _should_run_init_prewarm(
    *,
    is_first_workspace_init: bool,
    overwrite: bool,
    main_key_persist_failed: bool,
) -> bool:
    """判断当前 `init` 是否应执行首次安装 prewarm。

    Args:
        is_first_workspace_init: 当前是否属于首次初始化。
        overwrite: 是否显式覆盖初始化。
        main_key_persist_failed: 主 API Key 是否持久化失败。

    Returns:
        若属于首次初始化且主 Key 可用，则返回 `True`；否则返回 `False`。

    Raises:
        无。
    """

    if overwrite or main_key_persist_failed:
        return False
    return is_first_workspace_init


def _run_init_prewarm(*, base_dir: Path, config_dir: Path) -> tuple[bool, str]:
    """执行 `dayu-cli init` 的首次安装 runtime prewarm。

    该预热只做无副作用的运行时装配，不执行 prompt / interactive / write
    的真实业务逻辑，用于把首次冷启动成本前移到安装阶段。

    Args:
        base_dir: 工作区根目录。
        config_dir: 已复制完成的配置目录。

    Returns:
        `(success, message)`。成功时 `message` 为空字符串；失败时返回可直接展示给用户的错误摘要。

    Raises:
        无。所有异常都会被内部捕获并转成失败结果。
    """

    del base_dir
    del config_dir
    try:
        for module_name in _PREWARM_MODULES:
            importlib.import_module(module_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


# --------------------------------------------------------------------------- #
#  Manifest 默认模型替换
# --------------------------------------------------------------------------- #


def _classify_model_role(current_name: str, stored_role: str) -> str | None:
    """判断 manifest 中当前模型属于 non-thinking 还是 thinking 角色。

    优先使用 ``_init_model_role`` 标记；标记不存在时根据模型名推断。
    对同时出现在两个集合中的歧义模型名（如 ``gpt-5.4``），无标记时无法判断，返回 ``None``。

    Args:
        current_name: 当前 ``model.default_name``。
        stored_role: manifest 中已记录的 ``_init_model_role``，空字符串表示无记录。

    Returns:
        ``"non_thinking"``、``"thinking"`` 或 ``None``（无法判断）。

    Raises:
        无。
    """

    if stored_role in (_ROLE_NON_THINKING, _ROLE_THINKING):
        return stored_role

    # 无标记时靠模型名推断——但歧义名无法判断
    if current_name in _ONLY_NON_THINKING:
        return _ROLE_NON_THINKING
    if current_name in _ONLY_THINKING:
        return _ROLE_THINKING
    # 歧义模型或完全未知模型
    return None


def _resolve_role_from_package_manifest(manifest_filename: str) -> str | None:
    """从包内原始 manifest 推断 scene 的模型角色。

    包内原始 manifest 使用无歧义的默认模型名（``mimo-v2-pro-plan`` / ``mimo-v2-pro-thinking-plan``），
    可作为 fallback 判断角色。

    Args:
        manifest_filename: manifest 文件名（如 ``write.json``）。

    Returns:
        ``"non_thinking"``、``"thinking"`` 或 ``None``（原始 manifest 不存在或无法判断）。

    Raises:
        无。
    """

    pkg_manifests = resolve_package_config_path() / "prompts" / "manifests"
    pkg_file = pkg_manifests / manifest_filename
    if not pkg_file.exists():
        return None

    try:
        pkg_data = json.loads(pkg_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    pkg_model = pkg_data.get("model", {})
    if not isinstance(pkg_model, dict):
        return None

    pkg_name = pkg_model.get("default_name", "")
    # 包内原始模型名一定是无歧义的，直接用 _classify_model_role（无标记）
    return _classify_model_role(pkg_name, "")


def _update_manifest_default_models(
    config_dir: Path,
    non_thinking_model: str,
    thinking_model: str,
) -> int:
    """更新 manifest 文件中的 ``model.default_name``。

    识别当前 ``default_name`` 的角色（non-thinking / thinking），
    替换为目标供应商对应角色的模型名，并写入 ``_init_model_role`` 标记
    以便后续 init 能正确识别角色。

    当工作区 manifest 中的模型名歧义且无角色标记时，回退到包内原始 manifest
    推断该 scene 的角色。

    Args:
        config_dir: 工作区配置目录。
        non_thinking_model: 替换后的非 thinking 模型名。
        thinking_model: 替换后的 thinking 模型名。

    Returns:
        被修改的 manifest 文件数量。

    Raises:
        无。
    """

    manifests_dir = config_dir / "prompts" / "manifests"
    if not manifests_dir.exists():
        return 0

    role_to_model = {
        _ROLE_NON_THINKING: non_thinking_model,
        _ROLE_THINKING: thinking_model,
    }

    updated = 0
    for manifest_file in sorted(manifests_dir.glob("*.json")):
        text = manifest_file.read_text(encoding="utf-8")
        data = json.loads(text)

        model_section = data.get("model")
        if not isinstance(model_section, dict):
            continue

        current = model_section.get("default_name", "")
        stored_role = model_section.get(_INIT_ROLE_KEY, "")
        if not isinstance(stored_role, str):
            stored_role = ""

        role = _classify_model_role(current, stored_role)
        if role is None:
            # 歧义模型名且无标记——回退到包内原始 manifest 推断角色
            role = _resolve_role_from_package_manifest(manifest_file.name)
        if role is None:
            continue

        new_name = role_to_model[role]
        changed = new_name != current or stored_role != role

        allowed = model_section.get("allowed_names")
        if isinstance(allowed, list) and new_name not in allowed:
            allowed.append(new_name)
            changed = True

        if changed:
            model_section["default_name"] = new_name
            model_section[_INIT_ROLE_KEY] = role
            manifest_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            updated += 1

    return updated


# --------------------------------------------------------------------------- #
#  交互式供应商选择
# --------------------------------------------------------------------------- #


def _prompt_provider_selection() -> str:
    """交互式让用户选择初始化模型方案。

    已在环境变量中设置了 API Key 的方案会标注「✓ 已配置」。

    Returns:
        选中的方案 key。

    Raises:
        SystemExit: 用户输入无效或 EOF 时退出。
    """

    # 找第一个已配置的方案作为默认推荐；共享同一 API Key 的方案按声明顺序优先。
    first_configured_idx = 0
    for i, option in enumerate(_PROVIDER_OPTIONS):
        if os.environ.get(option.api_key_name):
            first_configured_idx = i
            break

    print("\n请选择你要使用的初始化模型方案（输入编号）：\n")
    for i, option in enumerate(_PROVIDER_OPTIONS, 1):
        configured = "  ✓ 已配置" if os.environ.get(option.api_key_name) else ""
        default_marker = "（默认）" if (i - 1) == first_configured_idx else ""
        print(f"  {i}. {option.display_name}  — {option.api_key_name}{configured}{default_marker}")

    print()
    default_num = first_configured_idx + 1
    try:
        raw = input(f"选择 [{default_num}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)

    if raw == "":
        return _PROVIDER_OPTIONS[first_configured_idx].option_key

    try:
        idx = int(raw)
    except ValueError:
        print(f"无效输入: {raw}")
        sys.exit(1)

    if idx < 1 or idx > len(_PROVIDER_OPTIONS):
        print(f"编号超出范围: {idx}")
        sys.exit(1)

    return _PROVIDER_OPTIONS[idx - 1].option_key


def _prompt_api_key(api_key_name: str) -> str:
    """交互式获取 API Key 值。

    Args:
        api_key_name: API Key 环境变量名。

    Returns:
        用户输入的 API Key 值。

    Raises:
        SystemExit: 用户未输入或 EOF 时退出。
    """

    try:
        value = input(f"\n请输入 {api_key_name}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)

    if not value:
        print(f"错误: {api_key_name} 不能为空")
        sys.exit(1)

    return value


@dataclass(frozen=True)
class _CustomOpenAIConfig:
    """用户自定义 OpenAI 兼容 API 的参数。"""

    base_url: str
    api_key_value: str
    model_id: str


def _prompt_custom_openai_config(api_key_name: str) -> _CustomOpenAIConfig:
    """交互式收集自定义 OpenAI 兼容 API 参数。

    收集 base URL、API Key 值，以及单个模型 ID；thinking / 非 thinking
    两个角色共享该模型。

    Args:
        api_key_name: 当前供应商对应的 API Key 环境变量名。

    Returns:
        已收集完毕的 `_CustomOpenAIConfig`。

    Raises:
        SystemExit: 用户中断或必填项为空。
    """

    print("\n— 自定义 OpenAI 兼容 API 配置 —")
    print("  示例: OpenRouter 的 base URL 为 https://openrouter.ai/api/v1")
    try:
        base_url = input("  Base URL（如 https://openrouter.ai/api/v1）: ").strip()
        if not base_url:
            print("错误: Base URL 不能为空")
            sys.exit(1)
        base_url = base_url.rstrip("/")

        existing_value = os.environ.get(api_key_name, "")
        if existing_value:
            masked = existing_value[:4] + "***" + existing_value[-4:] if len(existing_value) > 8 else "***"
            print(f"  {api_key_name} 已在环境变量中配置（{masked}），将复用该值。")
            api_key_value = existing_value
        else:
            api_key_value = input(f"  请输入 {api_key_name}: ").strip()
            if not api_key_value:
                print(f"错误: {api_key_name} 不能为空")
                sys.exit(1)

        print("  请填写模型 ID（OpenRouter 示例：openai/gpt-4o、anthropic/claude-sonnet-4）")
        model_id = input("  模型 ID: ").strip()
        if not model_id:
            print("错误: 模型 ID 不能为空")
            sys.exit(1)
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)

    return _CustomOpenAIConfig(
        base_url=base_url,
        api_key_value=api_key_value,
        model_id=model_id,
    )


def _build_custom_openai_catalog_entry(
    *,
    endpoint_url: str,
    api_key_name: str,
    model_id: str,
    description: str,
) -> dict[str, object]:
    """构造自定义 OpenAI 兼容模型的 catalog 条目。

    Args:
        endpoint_url: OpenAI 兼容接口的 ``/chat/completions`` 地址。
        api_key_name: 对应 API Key 的环境变量名。
        model_id: 用户输入的目标模型 ID。
        description: 写入 catalog 的描述信息。

    Returns:
        可直接写入 ``llm_models.json`` 的模型条目字典。

    Raises:
        无。
    """

    headers = {
        "Authorization": f"Bearer {{{{{api_key_name}}}}}",
        "Content-Type": "application/json",
    }
    runtime_hints = {
        "temperature_profiles": _CUSTOM_OPENAI_TEMPERATURE_PROFILES,
        "conversation_memory": _CUSTOM_OPENAI_CONVERSATION_MEMORY,
    }
    return {
        "runner_type": "openai_compatible",
        "name": _CUSTOM_CATALOG_KEY,
        "endpoint_url": endpoint_url,
        "model": model_id,
        "headers": headers,
        "timeout": 3600,
        "stream_idle_timeout": 120.0,
        "stream_idle_heartbeat_sec": 10.0,
        "supports_stream": True,
        "supports_tool_calling": True,
        "supports_usage": True,
        "supports_stream_usage": True,
        "max_context_tokens": 131072,
        "max_output_tokens": 8192,
        "extra_payloads": {},
        "description": description,
        "runtime_hints": runtime_hints,
    }


def _write_custom_openai_catalog_entries(
    config_dir: Path,
    custom: _CustomOpenAIConfig,
    *,
    api_key_name: str,
) -> None:
    """将自定义 API 的单个条目写入工作区 ``llm_models.json``。

    thinking / 非 thinking 角色共享同一个 catalog key ``custom-openai``，
    因而这里只覆盖该单个条目，同时保留文件中的其他供应商配置。

    Args:
        config_dir: 工作区配置目录。
        custom: 用户填写的自定义 API 参数。
        api_key_name: 当前供应商对应的 API Key 环境变量名。

    Raises:
        ValueError: ``llm_models.json`` 非法或顶层结构不是对象时抛出。
        OSError: 文件读写失败时抛出。
    """

    catalog_path = config_dir / "llm_models.json"
    data: dict[str, object] = {}
    if catalog_path.exists():
        raw_text = catalog_path.read_text(encoding="utf-8")
        if raw_text.strip():
            try:
                parsed = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{catalog_path} 不是合法 JSON，请先修复该文件后再重试 init。") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{catalog_path} 顶层必须是 JSON 对象，请先修复该文件后再重试 init。")
            data = parsed

    endpoint_url = f"{custom.base_url}/chat/completions"
    data[_CUSTOM_CATALOG_KEY] = _build_custom_openai_catalog_entry(
        endpoint_url=endpoint_url,
        api_key_name=api_key_name,
        model_id=custom.model_id,
        description=f"自定义 OpenAI 兼容 API（{custom.base_url}）",
    )

    catalog_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _prompt_optional_search_keys() -> list[tuple[str, str]]:
    """交互式配置可选的联网检索 API Key。

    已在环境变量中存在的 key 自动跳过。

    Returns:
        ``(key_name, key_value)`` 列表，仅包含用户实际输入了值的项。

    Raises:
        无。
    """

    missing_keys = [k for k in _OPTIONAL_SEARCH_KEYS if not os.environ.get(k)]
    already_set = [k for k in _OPTIONAL_SEARCH_KEYS if os.environ.get(k)]

    if already_set:
        print(f"\n联网检索 API Key（已配置: {', '.join(already_set)}）")

    if not missing_keys:
        print("  所有联网检索 API Key 均已配置，跳过。")
        return []

    print("是否配置以下联网检索 API Key？（可选，直接回车跳过）")
    results: list[tuple[str, str]] = []

    for key in missing_keys:
        try:
            value = input(f"  {key}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if value:
            results.append((key, value))

    return results


# --------------------------------------------------------------------------- #
#  HuggingFace 镜像与 Token
# --------------------------------------------------------------------------- #


def _is_hf_hub_reachable() -> bool:
    """探测 HuggingFace 官方 Hub 是否可达。

    使用 HEAD 请求探测，超时时间较短，仅用于决定镜像配置的默认值。

    Returns:
        ``True`` 表示官方 Hub 可达，``False`` 表示不可达。

    Raises:
        无。
    """

    req = urllib.request.Request(_HF_PROBE_URL, method="HEAD")
    try:
        urllib.request.urlopen(req, timeout=_HF_PROBE_TIMEOUT_SECONDS)  # noqa: S310
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def _prompt_huggingface_config() -> list[tuple[str, str]]:
    """交互式配置 HuggingFace 镜像加速和认证 Token。

    先探测官方 Hub 是否可达：可达则默认不启用镜像（n），不可达则默认启用（Y）。
    已在环境变量中存在的项自动跳过。

    Returns:
        ``(key_name, key_value)`` 列表，仅包含需要写入的项。

    Raises:
        无。
    """

    results: list[tuple[str, str]] = []

    print("\n— HuggingFace 模型下载配置 —")

    # HF_ENDPOINT（镜像）
    if os.environ.get("HF_ENDPOINT"):
        print(f"  HF_ENDPOINT 已配置: {os.environ['HF_ENDPOINT']}，跳过。")
    else:
        print("  正在检测 HuggingFace Hub 连通性…", end="", flush=True)
        hub_ok = _is_hf_hub_reachable()
        if hub_ok:
            print(" 可达。")
            default_mirror = "n"
            prompt_text = f"  是否使用 HuggingFace 镜像加速（{_HF_MIRROR_URL}）？(y/N): "
        else:
            print(" 不可达，建议启用镜像。")
            default_mirror = "y"
            prompt_text = f"  是否使用 HuggingFace 镜像加速（{_HF_MIRROR_URL}）？(Y/n): "

        try:
            answer = input(prompt_text).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return results

        if answer == "":
            answer = default_mirror
        if answer in ("y", "yes"):
            results.append(("HF_ENDPOINT", _HF_MIRROR_URL))

    # HF_TOKEN（可选认证）
    if os.environ.get("HF_TOKEN"):
        existing = os.environ["HF_TOKEN"]
        masked = existing[:4] + "***" if len(existing) > 4 else "***"
        print(f"  HF_TOKEN 已配置（{masked}），跳过。")
    else:
        try:
            token = input("  HF_TOKEN（可选，直接回车跳过）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return results
        if token:
            results.append(("HF_TOKEN", token))

    return results


def _prompt_sec_user_agent() -> tuple[str, str] | None:
    """交互式配置 SEC User-Agent。

    SEC 要求所有爬虫请求在 User-Agent 中提供真实公司名称和联系邮箱，
    格式为 ``"CompanyName admin@company.com"``。

    已在环境变量中存在时自动跳过。

    Returns:
        ``(key_name, value)`` 元组；用户跳过时返回 ``None``。

    Raises:
        无。
    """

    print("\n— SEC 下载配置 —")

    existing = os.environ.get(SEC_USER_AGENT_ENV)
    if existing:
        print(f"  {SEC_USER_AGENT_ENV} 已配置: {existing}，跳过。")
        return None

    print("  SEC 要求爬虫请求提供真实 User-Agent（含邮箱），")
    print('  格式示例: "MyCompany admin@mycompany.com"')

    try:
        value = input(f"  {SEC_USER_AGENT_ENV}（直接回车跳过）: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not value:
        print("  已跳过。下载 SEC 文件时将使用通用 User-Agent（可能被限流）。")
        return None

    return SEC_USER_AGENT_ENV, value


# --------------------------------------------------------------------------- #
#  主入口
# --------------------------------------------------------------------------- #


def run_init_command(args: Namespace) -> int:
    """执行 ``dayu-cli init`` 子命令。

    Args:
        args: 解析后的命令行参数，包含 ``base``（工作区目录）、
            ``overwrite``（是否覆盖已有配置）与 ``reset``（是否先重置工作区状态）。

    Returns:
        退出码，0 表示成功。

    Raises:
        无。
    """

    base_dir = Path(args.base).resolve()
    overwrite: bool = bool(getattr(args, "overwrite", False))
    reset: bool = bool(getattr(args, "reset", False))
    if reset:
        if not _confirm_workspace_reset(base_dir):
            print("已取消工作区重置。")
            return 1
        removed_targets = _reset_workspace_init_targets(base_dir)
        if removed_targets:
            removed_names = "、".join(path.name for path in removed_targets)
            print(f"✓ 已重置工作区初始化目录: {removed_names}")
        else:
            target_names = "、".join(path.name for path in _build_workspace_reset_targets(base_dir))
            print(f"✓ 已执行工作区重置：未发现需要删除的 {target_names}")

    is_first_workspace_init = reset or not (base_dir / "config").exists()
    # 1. 复制配置
    config_dir = _copy_config(base_dir, overwrite=overwrite)
    print(f"✓ 配置已复制到: {config_dir}")

    # 1b. 复制 assets（定性分析模板等）
    assets_dir = _copy_assets(base_dir, overwrite=overwrite)
    print(f"✓ assets 已复制到: {assets_dir}")

    # 1c. 对旧工作区应用一次性迁移（run.json、Host SQLite）。
    # 具体规则集中在 dayu.cli.workspace_migrations，避免混入 init 常规流程。
    apply_all_workspace_migrations(base_dir=base_dir, config_dir=config_dir)

    # 2. 选择初始化模型方案 + 输入 API Key（已有则跳过）
    chosen_option_key = _prompt_provider_selection()
    chosen_option = _PROVIDER_OPTIONS_BY_KEY[chosen_option_key]
    env_vars_written = False
    main_key_persist_failed = False

    if chosen_option_key == _PROVIDER_OPTION_CUSTOM_OPENAI:
        effective_api_key_name = chosen_option.api_key_name
        custom = _prompt_custom_openai_config(effective_api_key_name)
        try:
            _write_custom_openai_catalog_entries(
                config_dir,
                custom,
                api_key_name=effective_api_key_name,
            )
        except ValueError as exc:
            print(f"\n❌ {exc}")
            return 1
        print(f"✓ 已写入自定义模型条目到 {config_dir / 'llm_models.json'}")
        if os.environ.get(effective_api_key_name) == custom.api_key_value:
            print(f"\n✓ {effective_api_key_name} 已在环境变量中配置，跳过写入。")
        else:
            _target, ok = _persist_env_var(effective_api_key_name, custom.api_key_value)
            env_vars_written = True
            if not ok:
                main_key_persist_failed = True
                print(f"\n❌ {effective_api_key_name} 无法持久化到系统环境变量。")
                print(f"   已为当前进程设置，但重开终端后会丢失。")
                print(f"   为避免切换模型后下次启动找不到 API Key，跳过 manifest 更新。")
                print(f"   请手动配置环境变量后重新运行 dayu-cli init。")
    else:
        effective_api_key_name = chosen_option.api_key_name
        existing_value = os.environ.get(chosen_option.api_key_name)
        if existing_value:
            masked = existing_value[:4] + "***" + existing_value[-4:] if len(existing_value) > 8 else "***"
            print(f"\n✓ {chosen_option.api_key_name} 已在环境变量中配置（{masked}），跳过写入。")
        else:
            api_key_value = _prompt_api_key(chosen_option.api_key_name)
            _target, ok = _persist_env_var(chosen_option.api_key_name, api_key_value)
            env_vars_written = True
            if not ok:
                main_key_persist_failed = True
                print(f"\n❌ {chosen_option.api_key_name} 无法持久化到系统环境变量。")
                print(f"   已为当前进程设置，但重开终端后会丢失。")
                print(f"   为避免切换模型后下次启动找不到 API Key，跳过 manifest 更新。")
                print(f"   请手动配置环境变量后重新运行 dayu-cli init。")

    # 3. 更新 manifest 默认模型（仅在主 key 持久化成功或已存在时执行）
    non_thinking = chosen_option.non_thinking_model
    thinking = chosen_option.thinking_model
    if main_key_persist_failed:
        print(f"\n⚠️  跳过 manifest 更新（{effective_api_key_name} 未持久化）")
    else:
        updated_count = _update_manifest_default_models(config_dir, non_thinking, thinking)
        print(f"✓ 默认模型已设置为: {non_thinking} / {thinking}（更新了 {updated_count} 个 manifest）")

    should_run_prewarm = _should_run_init_prewarm(
        is_first_workspace_init=is_first_workspace_init,
        overwrite=overwrite,
        main_key_persist_failed=main_key_persist_failed,
    )

    # 4. 可选联网检索 Key
    search_persist_failed = False
    search_keys = _prompt_optional_search_keys()
    for key, value in search_keys:
        _search_target, search_ok = _persist_env_var(key, value)
        env_vars_written = True
        if not search_ok:
            search_persist_failed = True
        print(f"✓ {key} 已配置")

    # 5. HuggingFace 镜像与 Token
    hf_keys = _prompt_huggingface_config()
    for key, value in hf_keys:
        _hf_target, hf_ok = _persist_env_var(key, value)
        env_vars_written = True
        if not hf_ok:
            search_persist_failed = True
        display = value if key == "HF_ENDPOINT" else f"{value[:4]}***"
        print(f"✓ {key} 已配置: {display}")

    # 6. SEC User-Agent
    sec_ua = _prompt_sec_user_agent()
    if sec_ua is not None:
        _sec_target, sec_ok = _persist_env_var(sec_ua[0], sec_ua[1])
        env_vars_written = True
        if not sec_ok:
            search_persist_failed = True
        print(f"✓ {sec_ua[0]} 已配置: {sec_ua[1]}")

    if should_run_prewarm and not main_key_persist_failed:
        print("\n正在预热 CLI 运行时，请稍候...")
        prewarm_success, prewarm_message = _run_init_prewarm(
            base_dir=base_dir,
            config_dir=config_dir,
        )
        if prewarm_success:
            print("✓ CLI 运行时预热完成")
        else:
            print(f"⚠️  CLI 运行时预热失败: {prewarm_message}")
            print("   不影响当前工作区初始化成功；后续首次运行 prompt / interactive / write 时可能更慢。")

    # 6. 完成提示
    print(f"\n✓ 工作区已初始化: {base_dir}")

    if env_vars_written:
        if main_key_persist_failed or search_persist_failed:
            shell_name = os.environ.get("SHELL", "").rsplit("/", 1)[-1] or "unknown"
            print(f"\n⚠️  当前环境（shell={shell_name}）不支持自动写入环境变量。")
            print(f"   已为当前进程设置环境变量，但重开终端后会丢失。")
            print(f"   请手动将 API Key 添加到你的 shell 配置文件中。\n")
        elif platform.system() != "Windows":
            profile, _ = _detect_shell_profile()
            print(f"\n⚠️  环境变量已写入 {profile}，但当前终端尚未生效。")
            print(f"   请立即执行以下命令，或重新打开终端：\n")
            print(f"   source {profile}\n")
        else:
            print(f"\n⚠️  环境变量已通过 setx 写入，但当前终端尚未生效。")
            print(f"   请关闭并重新打开终端。\n")

        print("环境变量生效后，可以开始使用：")
    else:
        print("\n可以开始使用：")
    print(f"  dayu-cli download --ticker AAPL")
    print(f'  dayu-cli prompt "总结苹果最新财报的主要风险"')

    return 1 if main_key_persist_failed else 0
