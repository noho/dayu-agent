"""PromptComposer 纯装配测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from dayu.prompting import (
    PromptManifestError,
    PromptToolSnapshot,
    PromptComposeContext,
    PromptComposer,
    build_prompt_assembly_plan,
    load_scene_definition,
)
from dayu.prompting.prompt_plan import PromptAssemblyPlan, PromptFragmentPlan
from dayu.prompting.scene_definition import ToolSelectionMode, ToolSelectionPolicy, parse_scene_definition
from dayu.startup.config_file_resolver import ConfigFileResolver
from dayu.startup.prompt_assets import FilePromptAssetStore

_EXPECTED_THINKING_ALLOWED_NAMES: tuple[str, ...] = (
    "mimo-v2-flash-thinking",
    "mimo-v2-pro-thinking",
    "mimo-v2-pro-thinking-plan",
    "mimo-v2-pro-thinking-plan-sg",
    "deepseek-thinking",
    "qwen3-thinking",
    "qwen3:30b-thinking",
    "gpt-5.4",
    "claude-sonnet-4-6",
    "gemini-2.5-flash",
)


@pytest.mark.unit
def test_compose_interactive_scene_contains_base_and_scene_fragments() -> None:
    """验证 interactive scene 只装配静态 fragment，并支持尾部追加 Prompt Contributions。"""

    composer = PromptComposer()
    plan = build_prompt_assembly_plan(asset_store=FilePromptAssetStore(ConfigFileResolver()), scene_name="interactive")
    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver()), "interactive")
    composed = composer.compose(
        plan=plan,
        context=PromptComposeContext(
            values={
                "directories": "workspace/test",
            }
        ),
        tool_snapshot=PromptToolSnapshot(
            tool_tags=frozenset({"fins"}),
            tool_names=frozenset({"list_documents"}),
            allowed_paths=("workspace/test",),
            supports_tool_calling=True,
        ),
        prompt_contributions={"base_user": "# 用户与运行时上下文\n当前时间：2026年03月13日。"},
        context_slots=manifest.context_slots,
    )

    assert composed.fragment_ids == (
        "base_agents",
        "base_soul",
        "base_fact_rules",
        "base_tools",
        "interactive_scene",
    )
    assert "行为规范" in composed.system_message
    assert "交互执行契约" in composed.system_message
    assert "# 用户与运行时上下文" in composed.system_message


@pytest.mark.unit
def test_compose_audit_scene_only_uses_audit_fragment() -> None:
    """验证 audit scene 只装配静态 fragment，并支持尾部追加 Prompt Contributions。"""

    composer = PromptComposer()
    plan = build_prompt_assembly_plan(asset_store=FilePromptAssetStore(ConfigFileResolver()), scene_name="audit")
    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver()), "audit")
    composed = composer.compose(
        plan=plan,
        context=PromptComposeContext(values={}),
        tool_snapshot=PromptToolSnapshot(),
        prompt_contributions={"base_user": "# 用户与运行时上下文\n当前时间：2026年03月13日。"},
        context_slots=manifest.context_slots,
    )

    assert composed.fragment_ids == (
        "base_agents",
        "base_fact_rules",
        "audit_scene",
    )
    assert "审计执行契约" in composed.system_message
    assert "输出必须是严格可解析 JSON" in composed.system_message
    assert "你当前没有任何可调用工具" in composed.system_message
    assert "# 用户与运行时上下文" in composed.system_message


@pytest.mark.unit
def test_compose_write_scene_is_independent_from_interactive() -> None:
    """验证 write scene 独立声明自身静态 fragment，动态上下文改走 Prompt Contributions。"""

    composer = PromptComposer()
    plan = build_prompt_assembly_plan(asset_store=FilePromptAssetStore(ConfigFileResolver()), scene_name="write")
    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver()), "write")
    composed = composer.compose(
        plan=plan,
        context=PromptComposeContext(
            values={
                "directories": "workspace/test",
            }
        ),
        tool_snapshot=PromptToolSnapshot(
            tool_tags=frozenset({"fins"}),
            tool_names=frozenset({"list_documents"}),
            allowed_paths=("workspace/test",),
            supports_tool_calling=True,
        ),
        prompt_contributions={"base_user": "# 用户与运行时上下文\n当前时间：2026年03月13日。"},
        context_slots=manifest.context_slots,
    )

    assert composed.fragment_ids == (
        "base_agents",
        "base_soul",
        "base_fact_rules",
        "base_tools",
        "write_scene",
    )
    assert "写作执行契约" in composed.system_message
    assert "交互执行契约" not in composed.system_message
    assert "# 用户与运行时上下文" in composed.system_message


@pytest.mark.unit
def test_manifest_duplicate_order_raises(tmp_path: Path) -> None:
    """验证 manifest 中重复 order 会被拒绝。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (scenes_dir / "b.md").write_text("B", encoding="utf-8")
    (manifests_dir / "dup.json").write_text(
        """
        {
          "scene": "dup",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "dup"
          },
          "version": "v1",
          "description": "dup",
          "fragments": [
            {"id": "a", "type": "SCENE", "path": "scenes/a.md", "order": 100},
            {"id": "b", "type": "SCENE", "path": "scenes/b.md", "order": 100}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="order"):
        build_prompt_assembly_plan(asset_store=FilePromptAssetStore(ConfigFileResolver(config_dir)), scene_name="dup")


@pytest.mark.unit
def test_manifest_tool_selection_is_loaded_from_shared_parser() -> None:
    """验证 scene manifest 的 tool_selection 会被共享解析器读取。"""

    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver()), "audit")

    assert manifest.model.default_name == "mimo-v2-pro-thinking-plan"
    assert manifest.model.allowed_names == _EXPECTED_THINKING_ALLOWED_NAMES
    assert manifest.model.temperature_profile == "audit"
    assert manifest.runtime.agent.max_iterations == 24
    assert manifest.runtime.agent.max_consecutive_failed_tool_batches is None
    assert manifest.runtime.runner.tool_timeout_seconds is None
    assert manifest.tool_selection_policy.mode.value == "none"


@pytest.mark.unit
def test_conversation_compaction_manifest_is_tool_free() -> None:
    """验证 conversation_compaction scene 使用独立无工具 manifest。"""

    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver()), "conversation_compaction")

    assert manifest.model.default_name == "mimo-v2-pro-thinking-plan"
    assert manifest.model.temperature_profile == "conversation_compaction"
    assert manifest.tool_selection_policy.mode.value == "none"


@pytest.mark.unit
def test_compose_conversation_compaction_scene_uses_only_scene_fragment() -> None:
    """验证 conversation_compaction scene 只装配自身 scene fragment。"""

    composer = PromptComposer()
    plan = build_prompt_assembly_plan(asset_store=FilePromptAssetStore(ConfigFileResolver()), scene_name="conversation_compaction")
    composed = composer.compose(
        plan=plan,
        context=PromptComposeContext(values={}),
        tool_snapshot=PromptToolSnapshot(),
    )

    assert composed.fragment_ids == ("conversation_compaction_scene",)
    assert "会话压缩执行契约" in composed.system_message
    assert "严格可解析 JSON 对象" in composed.system_message


@pytest.mark.unit
def test_interactive_manifest_excludes_doc_tools_by_default() -> None:
    """验证 interactive scene 默认工具选择不包含 doc 工具。"""

    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver()), "interactive")

    assert manifest.model.default_name == "mimo-v2-pro-thinking-plan"
    assert manifest.model.temperature_profile == "interactive"
    assert manifest.conversation.enabled is True
    assert manifest.tool_selection_policy.mode.value == "select"
    assert manifest.tool_selection_policy.tool_tags_any == ("web", "fins", "ingestion")


@pytest.mark.unit
def test_prompt_scene_defaults_to_single_turn_conversation_mode() -> None:
    """验证未显式声明 conversation 的 scene 默认仍是单轮。"""

    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver()), "prompt")

    assert manifest.conversation.enabled is False


@pytest.mark.unit
def test_interactive_manifest_explicitly_declares_ticker_fragment() -> None:
    """验证 interactive scene 在 manifest 中显式声明 Prompt Contributions slots。"""

    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver()), "interactive")

    assert manifest.context_slots == ("fins_default_subject", "base_user")


@pytest.mark.unit
def test_compose_interactive_scene_renders_ticker_fragment_only_when_context_present() -> None:
    """验证显式声明的 ticker slot 只有在 Prompt Contributions 提供时才渲染。"""

    composer = PromptComposer()
    plan = build_prompt_assembly_plan(asset_store=FilePromptAssetStore(ConfigFileResolver()), scene_name="interactive")
    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver()), "interactive")
    snapshot = PromptToolSnapshot(
        tool_tags=frozenset({"fins"}),
        tool_names=frozenset({"list_documents"}),
        allowed_paths=("workspace/test",),
        supports_tool_calling=True,
    )

    without_ticker = composer.compose(
        plan=plan,
        context=PromptComposeContext(values={"directories": "workspace/test"}),
        tool_snapshot=snapshot,
        prompt_contributions={"base_user": "# 用户与运行时上下文\n当前时间：2026年03月13日。"},
        context_slots=manifest.context_slots,
    )
    with_ticker = composer.compose(
        plan=plan,
        context=PromptComposeContext(values={"directories": "workspace/test"}),
        tool_snapshot=snapshot,
        prompt_contributions={
            "fins_default_subject": "# 当前分析对象\n你正在分析的是 TEST。",
            "base_user": "# 用户与运行时上下文\n当前时间：2026年03月13日。",
        },
        context_slots=manifest.context_slots,
    )

    assert "fins_default_subject" not in without_ticker.fragment_ids
    assert "当前分析对象" not in without_ticker.system_message
    assert "fins_default_subject" not in with_ticker.fragment_ids
    assert "当前分析对象" in with_ticker.system_message
    assert "你正在分析的是 TEST。" in with_ticker.system_message
    assert with_ticker.system_message.endswith("# 当前分析对象\n你正在分析的是 TEST。\n\n# 用户与运行时上下文\n当前时间：2026年03月13日。")


@pytest.mark.unit
def test_compose_ignores_unknown_prompt_contributions_slots() -> None:
    """验证 PromptComposer 会忽略未声明 slot，而不是把它升级成装配失败。"""

    composer = PromptComposer()
    plan = build_prompt_assembly_plan(asset_store=FilePromptAssetStore(ConfigFileResolver()), scene_name="interactive")
    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver()), "interactive")
    composed = composer.compose(
        plan=plan,
        context=PromptComposeContext(values={"directories": "workspace/test"}),
        tool_snapshot=PromptToolSnapshot(
            tool_tags=frozenset({"fins"}),
            tool_names=frozenset({"list_documents"}),
            allowed_paths=("workspace/test",),
            supports_tool_calling=True,
        ),
        prompt_contributions={
            "unknown_slot": "不应进入 system prompt",
            "base_user": "# 用户与运行时上下文\n当前时间：2026年03月13日。",
        },
        context_slots=manifest.context_slots,
    )

    assert "不应进入 system prompt" not in composed.system_message
    assert composed.system_message.endswith("# 用户与运行时上下文\n当前时间：2026年03月13日。")


@pytest.mark.unit
def test_tools_fragment_tool_filters_is_rejected(tmp_path: Path) -> None:
    """验证 TOOLS fragment 上配置 tool_filters 会被拒绝。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    base_dir = config_dir / "prompts" / "base"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "tools.md").write_text("tools", encoding="utf-8")
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "invalid_tools.json").write_text(
        """
        {
          "scene": "invalid_tools",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "invalid_tools"
          },
          "version": "v1",
          "description": "invalid",
          "tool_selection": {"mode": "all"},
          "fragments": [
            {
              "id": "tools",
              "type": "TOOLS",
              "path": "base/tools.md",
              "order": 100,
              "tool_filters": {"tool_tags_any": ["fins"]}
            },
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="TOOLS fragment 不允许配置 tool_filters"):
        build_prompt_assembly_plan(asset_store=FilePromptAssetStore(ConfigFileResolver(config_dir)), scene_name="invalid_tools")


@pytest.mark.unit
def test_tool_selection_select_requires_non_empty_tags(tmp_path: Path) -> None:
    """验证 tool_selection.mode=select 时必须提供非空 tool_tags_any。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "invalid_select.json").write_text(
        """
        {
          "scene": "invalid_select",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "invalid_select"
          },
          "version": "v1",
          "description": "invalid",
          "tool_selection": {"mode": "select", "tool_tags_any": []},
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="tool_selection.mode=select"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "invalid_select")


@pytest.mark.unit
def test_scene_manifest_requires_model_object(tmp_path: Path) -> None:
    """验证 scene manifest 缺少 model 时直接报错。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "missing_model.json").write_text(
        """
        {
          "scene": "missing_model",
          "version": "v1",
          "description": "invalid",
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="model"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "missing_model")


@pytest.mark.unit
def test_scene_manifest_rejects_invalid_conversation_enabled_type(tmp_path: Path) -> None:
    """验证 conversation.enabled 必须是布尔值。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "invalid_conversation.json").write_text(
        """
        {
          "scene": "invalid_conversation",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "invalid_conversation"
          },
          "version": "v1",
          "description": "invalid",
          "conversation": {"enabled": "yes"},
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="conversation.enabled"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "invalid_conversation")


@pytest.mark.unit
def test_child_scene_inherits_parent_conversation_mode(tmp_path: Path) -> None:
    """验证子 scene 未显式覆盖时继承父 scene 的 conversation 策略。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "base.md").write_text("BASE", encoding="utf-8")
    (scenes_dir / "child.md").write_text("CHILD", encoding="utf-8")
    (manifests_dir / "base_scene.json").write_text(
        """
        {
          "scene": "base_scene",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "base_scene"
          },
          "version": "v1",
          "description": "base",
          "conversation": {"enabled": true},
          "fragments": [
            {"id": "base_scene", "type": "SCENE", "path": "scenes/base.md", "order": 100}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (manifests_dir / "child_scene.json").write_text(
        """
        {
          "scene": "child_scene",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "child_scene"
          },
          "version": "v1",
          "description": "child",
          "extends": ["base_scene"],
          "fragments": [
            {"id": "child_scene", "type": "SCENE", "path": "scenes/child.md", "order": 200}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    manifest = load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "child_scene")

    assert manifest.conversation.enabled is True


@pytest.mark.unit
def test_scene_manifest_rejects_cyclic_extends_chain(tmp_path: Path) -> None:
    """验证 scene extends 出现循环继承时抛出显式 manifest 错误。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (scenes_dir / "b.md").write_text("B", encoding="utf-8")
    (manifests_dir / "scene_a.json").write_text(
        """
        {
          "scene": "scene_a",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "scene_a"
          },
          "version": "v1",
          "description": "scene a",
          "extends": ["scene_b"],
          "fragments": [
            {"id": "scene_a", "type": "SCENE", "path": "scenes/a.md", "order": 100}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (manifests_dir / "scene_b.json").write_text(
        """
        {
          "scene": "scene_b",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "scene_b"
          },
          "version": "v1",
          "description": "scene b",
          "extends": ["scene_a"],
          "fragments": [
            {"id": "scene_b", "type": "SCENE", "path": "scenes/b.md", "order": 200}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="scene extends 存在循环继承: scene_a -> scene_b -> scene_a"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "scene_a")


@pytest.mark.unit
def test_scene_manifest_rejects_blank_default_model_name(tmp_path: Path) -> None:
    """验证 scene manifest 的空 model.default_name 会被拒绝。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "blank_model.json").write_text(
        """
        {
          "scene": "blank_model",
          "model": {
            "default_name": "   ",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "blank_model"
          },
          "version": "v1",
          "description": "invalid",
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="model.default_name"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "blank_model")


@pytest.mark.unit
def test_prompt_compose_context_select_and_has_any_cover_empty_values() -> None:
  """验证上下文选择与空值判断遵循白名单和非空语义。"""

  context = PromptComposeContext(
    values={
      "title": " Dayu ",
      "empty_text": "   ",
      "none_value": None,
    }
  )

  assert context.select(()) == context.values
  assert context.select(("title", "missing")) == {"title": " Dayu "}
  assert context.has_any(()) is True
  assert context.has_any(("missing", "empty_text", "none_value")) is False
  assert context.has_any(("missing", "title")) is True


@pytest.mark.unit
def test_compose_skips_mismatched_filters_missing_context_and_blank_render() -> None:
  """验证装配阶段会跳过工具不匹配、上下文缺失和空白渲染结果。"""

  plan = PromptAssemblyPlan(
    name="custom_scene",
    version="v1",
    fragments=(
      PromptFragmentPlan(id="late", template="Late", order=300),
      PromptFragmentPlan(id="blank", template="ignored", order=100),
      PromptFragmentPlan(
        id="requires_context",
        template="Need title",
        order=200,
        context_keys=("title",),
        skip_if_context_missing=True,
      ),
      PromptFragmentPlan(
        id="tool_only",
        template="Tool",
        order=250,
        tool_filters={"tool_names_any": ["allowed_tool"]},
      ),
    ),
  )

  class _Renderer:
    """返回固定渲染结果，方便命中跳过分支。"""

    def render(
      self,
      *,
      template: str,
      variables: dict[str, object] | None = None,
      tool_snapshot: PromptToolSnapshot | None = None,
    ) -> str:
      """渲染模板。"""

      del variables
      del tool_snapshot
      if template == "ignored":
        return ""
      return template

  composed = PromptComposer(renderer=cast(Any, _Renderer())).compose(
    plan=plan,
    context=PromptComposeContext(values={}),
    tool_snapshot=PromptToolSnapshot(tool_names=frozenset({"other_tool"})),
    prompt_contributions={
      "slot_b": "  second  ",
      "slot_a": "first",
      "slot_c": "   ",
    },
    context_slots=("slot_a", "slot_b", "slot_c"),
  )

  assert composed.fragment_ids == ("late",)
  assert composed.skipped_fragments == ("blank", "requires_context", "tool_only")
  assert composed.system_message == "Late\n\nfirst\n\nsecond"


@pytest.mark.unit
def test_compose_uses_default_tool_snapshot_when_omitted() -> None:
  """验证未传入 tool_snapshot 时仍可正常装配。"""

  plan = PromptAssemblyPlan(
    name="custom_scene",
    version="v1",
    fragments=(
      PromptFragmentPlan(
        id="base",
        template="Base",
        order=10,
        tool_filters={"tool_tags_all": ["missing"]},
      ),
    ),
  )

  composed = PromptComposer().compose(
    plan=plan,
    context=PromptComposeContext(values={}),
  )

  assert composed.system_message == ""
  assert composed.fragment_ids == ()
  assert composed.skipped_fragments == ("base",)


@pytest.mark.unit
def test_tool_selection_policy_allows_tool_tags_respects_mode() -> None:
  """验证工具选择策略按 mode 决定标签是否放行。"""

  assert ToolSelectionPolicy(mode=ToolSelectionMode.ALL).allows_tool_tags(()) is True
  assert ToolSelectionPolicy(mode=ToolSelectionMode.NONE).allows_tool_tags(("fins",)) is False
  assert ToolSelectionPolicy(mode=ToolSelectionMode.SELECT, tool_tags_any=("fins",)).allows_tool_tags(()) is False
  assert ToolSelectionPolicy(mode=ToolSelectionMode.SELECT, tool_tags_any=("fins",)).allows_tool_tags(("web",)) is False
  assert ToolSelectionPolicy(mode=ToolSelectionMode.SELECT, tool_tags_any=("fins",)).allows_tool_tags(("web", "fins")) is True


@pytest.mark.unit
def test_parse_scene_definition_rejects_invalid_defaults_and_context_slots() -> None:
  """验证 scene_definition 会拒绝非法 defaults 与 context_slots。"""

  with pytest.raises(PromptManifestError, match="defaults"):
    parse_scene_definition(
      {
        "scene": "invalid_defaults",
        "model": {
          "default_name": "mimo-v2-flash",
          "allowed_names": ["mimo-v2-flash"],
          "temperature_profile": "invalid_defaults",
        },
                "defaults": ["bad-defaults"],
        "fragments": [{"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 1}],
      }
    )

  with pytest.raises(PromptManifestError, match="context_slots"):
    parse_scene_definition(
      {
        "scene": "invalid_slots",
        "model": {
          "default_name": "mimo-v2-flash",
          "allowed_names": ["mimo-v2-flash"],
          "temperature_profile": "invalid_slots",
        },
        "context_slots": ["slot_a", " ", "slot_a"],
        "fragments": [{"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 1}],
      }
    )


@pytest.mark.unit
def test_parse_scene_definition_rejects_invalid_tool_selection_variants() -> None:
  """验证 scene_definition 会拒绝非法 tool_selection 配置。"""

  with pytest.raises(PromptManifestError, match="tool_selection.mode 非法"):
    parse_scene_definition(
      {
        "scene": "invalid_tool_mode",
        "model": {
          "default_name": "mimo-v2-flash",
          "allowed_names": ["mimo-v2-flash"],
          "temperature_profile": "invalid_tool_mode",
        },
        "tool_selection": {"mode": "bad-mode"},
        "fragments": [{"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 1}],
      }
    )

  with pytest.raises(PromptManifestError, match="tool_tags_any"):
    parse_scene_definition(
      {
        "scene": "invalid_tool_tags",
        "model": {
          "default_name": "mimo-v2-flash",
          "allowed_names": ["mimo-v2-flash"],
          "temperature_profile": "invalid_tool_tags",
        },
        "tool_selection": {"mode": "none", "tool_tags_any": ["fins"]},
        "fragments": [{"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 1}],
      }
    )


@pytest.mark.unit
def test_scene_manifest_rejects_non_positive_runtime_agent_max_iterations(tmp_path: Path) -> None:
    """验证 scene manifest 的非正数 runtime.agent.max_iterations 会被拒绝。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "invalid_iterations.json").write_text(
        """
        {
          "scene": "invalid_iterations",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "invalid_iterations"
          },
          "runtime": {
            "agent": {
              "max_iterations": 0
            }
          },
          "version": "v1",
          "description": "invalid",
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="runtime.agent.max_iterations"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "invalid_iterations")


@pytest.mark.unit
def test_scene_manifest_rejects_non_positive_runtime_failed_tool_batch_limit(tmp_path: Path) -> None:
    """验证 scene manifest 的非正数 runtime.agent failed tool batch 上限会被拒绝。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "invalid_failed_batches.json").write_text(
        """
        {
          "scene": "invalid_failed_batches",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "invalid_failed_batches"
          },
          "runtime": {
            "agent": {
              "max_consecutive_failed_tool_batches": 0
            }
          },
          "version": "v1",
          "description": "invalid",
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="runtime.agent.max_consecutive_failed_tool_batches"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "invalid_failed_batches")


@pytest.mark.unit
def test_scene_manifest_rejects_legacy_model_max_iterations_field(tmp_path: Path) -> None:
    """验证旧版 model.max_iterations 字段会被显式拒绝。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "legacy_iterations.json").write_text(
        """
        {
          "scene": "legacy_iterations",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "legacy_iterations",
            "max_iterations": 24
          },
          "version": "v1",
          "description": "invalid",
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="model.max_iterations 已迁移到 runtime.agent.max_iterations"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "legacy_iterations")


@pytest.mark.unit
def test_scene_manifest_rejects_legacy_failed_tool_batch_limit_field(tmp_path: Path) -> None:
    """验证旧版 model.max_consecutive_failed_tool_batches 字段会被显式拒绝。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "legacy_failed_batches.json").write_text(
        """
        {
          "scene": "legacy_failed_batches",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "legacy_failed_batches",
            "max_consecutive_failed_tool_batches": 3
          },
          "version": "v1",
          "description": "invalid",
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        PromptManifestError,
        match="model.max_consecutive_failed_tool_batches 已迁移到 runtime.agent.max_consecutive_failed_tool_batches",
    ):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "legacy_failed_batches")


@pytest.mark.unit
def test_child_scene_must_still_declare_model_when_extending(tmp_path: Path) -> None:
    """验证存在 extends 时子 scene 仍需显式声明 model。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "base.md").write_text("BASE", encoding="utf-8")
    (scenes_dir / "child.md").write_text("CHILD", encoding="utf-8")
    (manifests_dir / "base_scene.json").write_text(
        """
        {
          "scene": "base_scene",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "base_scene"
          },
          "version": "v1",
          "description": "base",
          "fragments": [
            {"id": "base_scene", "type": "SCENE", "path": "scenes/base.md", "order": 100}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )
    (manifests_dir / "child_scene.json").write_text(
        """
        {
          "scene": "child_scene",
          "version": "v1",
          "description": "child",
          "extends": ["base_scene"],
          "fragments": [
            {"id": "child_scene", "type": "SCENE", "path": "scenes/child.md", "order": 200}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="model"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "child_scene")


@pytest.mark.unit
def test_scene_manifest_rejects_blank_temperature_profile(tmp_path: Path) -> None:
    """验证 scene manifest 的空 temperature_profile 会被拒绝。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "invalid_temperature.json").write_text(
        """
        {
          "scene": "invalid_temperature",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "   "
          },
          "version": "v1",
          "description": "invalid",
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="temperature_profile"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "invalid_temperature")


@pytest.mark.unit
def test_scene_manifest_rejects_duplicate_allowed_names(tmp_path: Path) -> None:
    """验证 scene manifest 的重复 allowed_names 会被拒绝。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "boolean_temperature.json").write_text(
        """
        {
          "scene": "boolean_temperature",
          "model": {
            "default_name": "mimo-v2-flash",
            "allowed_names": ["mimo-v2-flash", "mimo-v2-flash"],
            "temperature_profile": "boolean_temperature"
          },
          "version": "v1",
          "description": "invalid",
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="allowed_names"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "boolean_temperature")


@pytest.mark.unit
def test_scene_manifest_requires_default_name_to_exist_in_allowed_names(tmp_path: Path) -> None:
    """验证 default_name 必须出现在 model.allowed_names 中。"""

    config_dir = tmp_path / "config"
    manifests_dir = config_dir / "prompts" / "manifests"
    scenes_dir = config_dir / "prompts" / "scenes"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    (scenes_dir / "a.md").write_text("A", encoding="utf-8")
    (manifests_dir / "invalid_default_name.json").write_text(
        """
        {
          "scene": "invalid_default_name",
          "model": {
            "default_name": "deepseek-chat",
            "allowed_names": ["mimo-v2-flash"],
            "temperature_profile": "invalid_default_name"
          },
          "version": "v1",
          "description": "invalid",
          "fragments": [
            {"id": "scene", "type": "SCENE", "path": "scenes/a.md", "order": 900}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(PromptManifestError, match="default_name"):
        load_scene_definition(FilePromptAssetStore(ConfigFileResolver(config_dir)), "invalid_default_name")
