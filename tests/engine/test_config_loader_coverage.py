"""ConfigLoader 补充覆盖测试。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

from dayu.contracts.model_config import OpenAICompatibleModelConfig
from dayu.startup import config_loader as module
from dayu.startup.config_file_resolver import ConfigFileResolver, resolve_package_config_path


@pytest.mark.unit
def test_env_var_replacer_warns_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证环境变量缺失时返回原文并记录告警。"""

    captured: dict[str, Any] = {}

    def _fake_warn(message: str, *, module: str) -> None:
        """记录 warn 日志。

        Args:
            message: 日志文本。
            module: 模块名。

        Returns:
            无。

        Raises:
            无。
        """

        captured["message"] = message
        captured["module"] = module

    monkeypatch.setattr(module.Log, "warning", _fake_warn)
    match = re.search(r"\{\{([A-Z_][A-Z0-9_]*)\}\}", "{{MISSING_TOKEN}}")
    assert match is not None

    replaced = module._env_var_replacer(match)
    assert replaced == "{{MISSING_TOKEN}}"
    assert "未设置" in captured["message"]


@pytest.mark.unit
def test_config_loader_init_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """验证 ConfigLoader 初始化的三条分支。"""

    del monkeypatch

    loader_default = module.ConfigLoader(ConfigFileResolver())
    assert len(loader_default._resolver.config_dirs) == 1

    package_config_path = resolve_package_config_path()
    loader_same = module.ConfigLoader(ConfigFileResolver(package_config_path))
    assert loader_same._resolver.config_dirs == [package_config_path]

    workspace_config = tmp_path / "workspace_config"
    workspace_config.mkdir(parents=True, exist_ok=True)
    loader_fallback = module.ConfigLoader(ConfigFileResolver(workspace_config))
    assert loader_fallback._resolver.config_dirs[0] == workspace_config.resolve()
    assert loader_fallback._resolver.config_dirs[1] == package_config_path


@pytest.mark.unit
def test_config_file_resolver_read_text_required_and_optional_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """验证 ConfigFileResolver 的文本读取分支。"""

    warnings: list[str] = []
    errors: list[str] = []
    monkeypatch.setattr(module.Log, "warning", lambda message, **kwargs: warnings.append(str(message)))
    monkeypatch.setattr(module.Log, "error", lambda message, **kwargs: errors.append(str(message)))

    workspace_config = tmp_path / "workspace"
    workspace_config.mkdir(parents=True, exist_ok=True)
    resolver = ConfigFileResolver(workspace_config)

    with pytest.raises(FileNotFoundError, match="查找路径"):
        resolver.read_text("missing.json", required=True)
    assert errors

    assert resolver.read_text("optional_missing.json", required=False) is None
    assert warnings

    app_config_resolver = ConfigFileResolver(resolve_package_config_path())
    with pytest.raises(FileNotFoundError, match="查找路径"):
        app_config_resolver.read_text("also_missing.json", required=True)


@pytest.mark.unit
def test_config_file_resolver_read_json_optional_none_and_decode_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 ConfigFileResolver JSON 解析的 None 与失败分支。"""

    resolver = ConfigFileResolver(resolve_package_config_path())

    monkeypatch.setattr(resolver, "read_text", lambda filename, required=True: None)
    assert resolver.read_json("x.json", required=False) is None

    errors: list[str] = []
    monkeypatch.setattr(module.Log, "error", lambda message, **kwargs: errors.append(str(message)))
    monkeypatch.setattr(resolver, "read_text", lambda filename, required=True: "{bad-json")

    with pytest.raises(json.JSONDecodeError):
        resolver.read_json("bad.json")
    assert errors


@pytest.mark.unit
def test_collect_referenced_env_vars_prefers_workspace_override(tmp_path: Path) -> None:
    """验证环境变量收集遵循 workspace config 优先于包内 fallback。"""

    workspace_config = tmp_path / "config"
    workspace_config.mkdir(parents=True, exist_ok=True)
    (workspace_config / "llm_models.json").write_text(
        json.dumps(
            {
                "demo": {
                    "headers": {
                        "Authorization": "Bearer {{WORKSPACE_API_KEY}}",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (workspace_config / "run.json").write_text(
        json.dumps(
            {
                "tool_trace_config": {
                    "token": "{{TRACE_TOKEN}}",
                }
            }
        ),
        encoding="utf-8",
    )

    loader = module.ConfigLoader(ConfigFileResolver(workspace_config))

    assert loader.collect_referenced_env_vars() == ("TRACE_TOKEN", "WORKSPACE_API_KEY")


@pytest.mark.unit
def test_collect_referenced_env_vars_ignores_non_config_and_binary_files(tmp_path: Path) -> None:
    """验证环境变量收集会跳过 README 与二进制缓存文件。"""

    workspace_config = tmp_path / "config"
    workspace_config.mkdir(parents=True, exist_ok=True)
    (workspace_config / "llm_models.json").write_text(
        json.dumps(
            {
                "demo": {
                    "headers": {
                        "Authorization": "Bearer {{MODEL_API_KEY}}",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (workspace_config / "README.md").write_text("{{README_ONLY_API_KEY}}", encoding="utf-8")
    pycache_dir = workspace_config / "__pycache__"
    pycache_dir.mkdir(parents=True, exist_ok=True)
    (pycache_dir / "run.cpython-313.pyc").write_bytes(b"\xf3\x00\x00\x00binary")

    loader = module.ConfigLoader(ConfigFileResolver(workspace_config))

    assert loader.collect_referenced_env_vars() == ("MODEL_API_KEY",)


@pytest.mark.unit
def test_collect_model_referenced_env_vars_only_scans_selected_models(tmp_path: Path) -> None:
    """验证模型环境变量收集只分析指定模型，并聚合去重。"""

    workspace_config = tmp_path / "config"
    workspace_config.mkdir(parents=True, exist_ok=True)
    (workspace_config / "llm_models.json").write_text(
        json.dumps(
            {
                "write_model": {
                    "headers": {
                        "Authorization": "Bearer {{WRITE_API_KEY}}",
                    },
                    "endpoint_url": "{{WRITE_BASE_URL}}",
                },
                "audit_model": {
                    "headers": {
                        "Authorization": "Bearer {{WRITE_API_KEY}}",
                        "X-Audit-Key": "{{AUDIT_API_KEY}}",
                    }
                },
                "unused_model": {
                    "headers": {
                        "Authorization": "Bearer {{UNUSED_API_KEY}}",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    loader = module.ConfigLoader(ConfigFileResolver(workspace_config))

    assert loader.collect_model_referenced_env_vars(("write_model", "audit_model")) == (
        "AUDIT_API_KEY",
        "WRITE_API_KEY",
        "WRITE_BASE_URL",
    )


@pytest.mark.unit
def test_load_run_config_rejects_non_object(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 run.json 顶层不是对象时会显式失败。"""

    resolver = ConfigFileResolver(resolve_package_config_path())
    monkeypatch.setattr(resolver, "read_json", lambda filename, required=True: ["bad"])

    loader = module.ConfigLoader(resolver)

    with pytest.raises(TypeError, match="run.json 必须是对象"):
        loader.load_run_config()


@pytest.mark.unit
def test_load_llm_models_rejects_non_object_model_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 llm_models.json 中单个模型配置不是对象时会显式失败。"""

    resolver = ConfigFileResolver(resolve_package_config_path())
    monkeypatch.setattr(
        resolver,
        "read_json",
        lambda filename, required=True: {"bad_model": "not-an-object"},
    )

    loader = module.ConfigLoader(resolver)

    with pytest.raises(TypeError, match="llm_models.json.bad_model 必须是对象"):
        loader.load_llm_models()


@pytest.mark.unit
def test_load_llm_models_ignores_metadata_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 llm_models.json 顶层元信息键不会被当成模型配置。"""

    resolver = ConfigFileResolver(resolve_package_config_path())
    monkeypatch.setattr(
        resolver,
        "read_json",
        lambda filename, required=True: {
            "_comment": "metadata",
            "demo_model": {"runner_type": "openai_compatible", "model": "demo"},
        },
    )

    loader = module.ConfigLoader(resolver)

    assert tuple(loader.load_llm_models().keys()) == ("demo_model",)


@pytest.mark.unit
def test_extract_env_var_names_and_should_scan_helpers(tmp_path: Path) -> None:
    """环境变量辅助函数应提取占位符并过滤非配置文件。"""

    config_file = tmp_path / "demo.json"
    config_file.write_text('{"token":"{{API_KEY}}","url":"{{BASE_URL}}"}', encoding="utf-8")
    readme_file = tmp_path / "README.md"
    readme_file.write_text("{{IGNORED}}", encoding="utf-8")
    pycache_file = tmp_path / "__pycache__" / "demo.json"
    pycache_file.parent.mkdir(parents=True, exist_ok=True)
    pycache_file.write_text("{}", encoding="utf-8")

    names = module._extract_env_var_names_from_text(config_file.read_text(encoding="utf-8"))

    assert names == ("API_KEY", "BASE_URL")
    assert module._should_scan_env_var_file(config_file) is True
    assert module._should_scan_env_var_file(readme_file) is False
    assert module._should_scan_env_var_file(pycache_file) is False


@pytest.mark.unit
def test_collect_env_var_names_from_model_config_recurses_nested_values() -> None:
    """模型环境变量收集应递归遍历嵌套 dict/list，而非依赖 JSON 序列化。"""

    names = module._collect_env_var_names_from_model_config(
        cast(
            Any,
            {
                "endpoint_url": "{{BASE_URL}}",
                "headers": [
                    {"Authorization": "Bearer {{API_KEY}}"},
                    {"X-Tags": ["{{TAG_ONE}}", "{{TAG_TWO}}"]},
                ],
                "metadata": {
                    "notes": "use {{API_KEY}} and {{BASE_URL}}",
                },
            },
        )
    )

    assert names == ("API_KEY", "BASE_URL", "TAG_ONE", "TAG_TWO")


@pytest.mark.unit
def test_load_llm_models_returns_deep_copy_of_cached_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """返回给调用方的模型配置副本不应污染内部缓存。"""

    resolver = ConfigFileResolver(resolve_package_config_path())
    monkeypatch.setattr(
        resolver,
        "read_json",
        lambda filename, required=True: {
            "demo_model": {
                "runner_type": "openai_compatible",
                "endpoint_url": "http://example.com",
                "headers": {"Authorization": "Bearer token"},
            }
        },
    )
    loader = module.ConfigLoader(resolver)

    first = loader.load_llm_models()
    first_model = cast(OpenAICompatibleModelConfig, first["demo_model"])
    first_model["headers"] = {"Authorization": "Bearer mutated"}
    second = loader.load_llm_models()
    second_model = cast(OpenAICompatibleModelConfig, second["demo_model"])
    headers = second_model.get("headers")
    assert headers is not None
    assert headers["Authorization"] == "Bearer token"


@pytest.mark.unit
def test_load_llm_model_replaces_env_vars_and_reports_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """加载单模型时应替换环境变量，并在模型不存在时报错。"""

    resolver = ConfigFileResolver(resolve_package_config_path())
    monkeypatch.setenv("TEST_API_KEY", "secret-token")
    monkeypatch.setattr(
        resolver,
        "read_json",
        lambda filename, required=True: {
            "demo_model": {
                "runner_type": "openai_compatible",
                "endpoint_url": "http://example.com",
                "model": "demo",
                "headers": {"Authorization": "Bearer {{TEST_API_KEY}}"},
            }
        },
    )
    errors: list[str] = []
    monkeypatch.setattr(module.Log, "error", lambda message, **kwargs: errors.append(str(message)))

    loader = module.ConfigLoader(resolver)
    loaded = loader.load_llm_model("demo_model")

    headers = cast(dict[str, str] | None, loaded.get("headers"))
    assert headers is not None
    assert headers["Authorization"] == "Bearer secret-token"

    with pytest.raises(KeyError):
        loader.load_llm_model("missing_model")

    assert errors


@pytest.mark.unit
def test_load_toolset_registrars_caches_and_validates_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    """toolset registrar 加载应返回副本，并校验 key/value。"""

    resolver = ConfigFileResolver(resolve_package_config_path())
    payload = {"doc": "dayu.engine.registrar:register"}
    monkeypatch.setattr(resolver, "read_json", lambda filename, required=True: payload)

    loader = module.ConfigLoader(resolver)
    loaded = loader.load_toolset_registrars()
    loaded["doc"] = "mutated"

    assert loader.load_toolset_registrars() == payload

    monkeypatch.setattr(resolver, "read_json", lambda filename, required=True: {"": "x"})
    invalid_key_loader = module.ConfigLoader(resolver)
    with pytest.raises(TypeError):
        invalid_key_loader.load_toolset_registrars()

    monkeypatch.setattr(resolver, "read_json", lambda filename, required=True: {"doc": "  "})
    invalid_value_loader = module.ConfigLoader(resolver)
    with pytest.raises(TypeError):
        invalid_value_loader.load_toolset_registrars()
