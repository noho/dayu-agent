"""Scene 定义解析与工具选择策略。

该模块属于启动期 / Host 共用的 prompt 定义层，负责：
- 解析并校验 scene manifest。
- 处理 scene definition 的单继承扩展。
- 解析顶层 ``model.default_name`` / ``model.allowed_names`` / ``model.temperature_profile`` /
    ``runtime.agent.max_iterations`` / ``runtime.agent.max_consecutive_failed_tool_batches`` /
    ``runtime.runner.tool_timeout_seconds``、``conversation.enabled`` 与 ``tool_selection``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Collection, Protocol, cast

from dayu.contracts.prompt_assets import (
    SceneAgentRuntimeAsset,
    SceneConversationAsset,
    SceneFragmentAsset,
    SceneManifestAsset,
    SceneModelAsset,
    SceneRunnerRuntimeAsset,
    SceneRuntimeAsset,
    SceneToolSelectionAsset,
)


class PromptManifestError(ValueError):
    """Scene manifest 非法时抛出的异常。"""


class PromptFragmentType(str, Enum):
    """Prompt fragment 类型枚举。"""

    AGENTS = "AGENTS"
    SOUL = "SOUL"
    USER = "USER"
    TOOLS = "TOOLS"
    MEMORY = "MEMORY"
    SKILL = "SKILL"
    SCENE = "SCENE"


class ToolSelectionMode(str, Enum):
    """工具选择模式。"""

    ALL = "all"
    NONE = "none"
    SELECT = "select"


@dataclass(frozen=True)
class ToolSelectionPolicy:
    """工具注册选择策略。

    Args:
        mode: 工具选择模式。
        tool_tags_any: 当 ``mode=select`` 时允许注册的工具标签集合。

    Returns:
        无。

    Raises:
        无。
    """

    mode: ToolSelectionMode = ToolSelectionMode.ALL
    tool_tags_any: tuple[str, ...] = ()

    def allows_tool_tags(self, tool_tags: Collection[str]) -> bool:
        """判断指定标签集合是否允许注册。

        Args:
            tool_tags: 工具标签集合。

        Returns:
            若当前策略允许注册则返回 ``True``。

        Raises:
            无。
        """

        if self.mode == ToolSelectionMode.ALL:
            return True
        if self.mode == ToolSelectionMode.NONE:
            return False
        if not tool_tags:
            return False
        return any(tag in tool_tags for tag in self.tool_tags_any)


@dataclass(frozen=True)
class SceneFragmentDefinition:
    """单个 scene fragment 的定义。"""

    id: str
    type: PromptFragmentType
    path: str
    order: int
    required: bool = True
    context_keys: tuple[str, ...] = ()
    skip_if_context_missing: bool = False
    enabled: bool = True
    tool_filters: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneModelDefinition:
    """Scene 默认模型定义。"""

    default_name: str
    allowed_names: tuple[str, ...] = ()
    temperature_profile: str = ""


@dataclass(frozen=True)
class SceneAgentRuntimeDefinition:
    """Scene 的 Agent 运行时定义。"""

    max_iterations: int | None = None
    max_consecutive_failed_tool_batches: int | None = None


@dataclass(frozen=True)
class SceneRunnerRuntimeDefinition:
    """Scene 的 Runner 运行时定义。"""

    tool_timeout_seconds: float | None = None


@dataclass(frozen=True)
class SceneRuntimeDefinition:
    """Scene 的运行时预算定义。"""

    agent: SceneAgentRuntimeDefinition = field(default_factory=SceneAgentRuntimeDefinition)
    runner: SceneRunnerRuntimeDefinition = field(default_factory=SceneRunnerRuntimeDefinition)


@dataclass(frozen=True)
class SceneConversationDefinition:
    """Scene 对话模式定义。"""

    enabled: bool = False


@dataclass(frozen=True)
class SceneDefinition:
    """完整 scene 定义。"""

    name: str
    model: SceneModelDefinition
    version: str
    description: str
    runtime: SceneRuntimeDefinition = field(default_factory=SceneRuntimeDefinition)
    extends: tuple[str, ...] = ()
    missing_fragment_policy: str = "error"
    fragments: tuple[SceneFragmentDefinition, ...] = ()
    context_slots: tuple[str, ...] = ()
    conversation: SceneConversationDefinition = field(default_factory=SceneConversationDefinition)
    tool_selection_policy: ToolSelectionPolicy = field(default_factory=ToolSelectionPolicy)
    _runtime_explicit: bool = False
    _conversation_explicit: bool = False
    _tool_selection_explicit: bool = False


class ScenePromptAssetStoreProtocol(Protocol):
    """Scene prompt 资产仓储协议。"""

    def load_scene_manifest(self, scene_name: str) -> SceneManifestAsset:
        """读取指定 scene 的 manifest。"""

        ...


def parse_scene_definition(data: object) -> SceneDefinition:
    """将原始 manifest 解析为 scene 定义。

    Args:
        data: 原始 manifest JSON 对象。

    Returns:
        解析后的 ``SceneDefinition``。

    Raises:
        PromptManifestError: manifest 非法时抛出。
    """

    if not isinstance(data, dict):
        raise PromptManifestError("scene manifest 必须是 JSON 对象")
    raw_manifest = cast(SceneManifestAsset, data)
    name = str(raw_manifest.get("scene") or "").strip()
    raw_model = raw_manifest.get("model")
    raw_runtime = raw_manifest.get("runtime")
    version = str(raw_manifest.get("version") or "v1").strip()
    description = str(raw_manifest.get("description") or "").strip()
    if not name:
        raise PromptManifestError("scene manifest 缺少 scene")
    model = _parse_scene_model_definition(raw_model)
    runtime_explicit = "runtime" in raw_manifest
    runtime = _parse_scene_runtime_definition(raw_runtime)
    extends_raw = raw_manifest.get("extends") or []
    if not isinstance(extends_raw, list):
        raise PromptManifestError("extends 必须是数组")
    defaults = raw_manifest.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise PromptManifestError("defaults 必须是对象")
    fragments_raw = raw_manifest.get("fragments") or []
    if not isinstance(fragments_raw, list) or not fragments_raw:
        raise PromptManifestError("fragments 必须是非空数组")
    fragments = [_parse_fragment_definition(item) for item in fragments_raw]
    context_slots = _parse_context_slots(raw_manifest.get("context_slots"))
    conversation_explicit = "conversation" in raw_manifest
    conversation = _parse_scene_conversation_definition(raw_manifest.get("conversation"))
    orders = [fragment.order for fragment in fragments]
    if len(orders) != len(set(orders)):
        raise PromptManifestError("fragment.order 不允许重复")
    tool_selection_explicit = "tool_selection" in raw_manifest
    return SceneDefinition(
        name=name,
        model=model,
        version=version,
        description=description,
        runtime=runtime,
        extends=tuple(str(item) for item in extends_raw),
        missing_fragment_policy=str(defaults.get("missing_fragment_policy") or "error"),
        fragments=tuple(fragments),
        context_slots=context_slots,
        conversation=conversation,
        tool_selection_policy=_parse_tool_selection_policy(raw_manifest.get("tool_selection")),
        _runtime_explicit=runtime_explicit,
        _conversation_explicit=conversation_explicit,
        _tool_selection_explicit=tool_selection_explicit,
    )


def load_scene_definition(
    asset_store: ScenePromptAssetStoreProtocol,
    scene_name: str,
) -> SceneDefinition:
    """加载并解析指定 scene 的完整定义。

    Args:
        asset_store: scene prompt 资产仓储。
        scene_name: scene 名称。

    Returns:
        处理继承后的 ``SceneDefinition``。

    Raises:
        PromptManifestError: manifest 非法时抛出。
    """

    return _load_scene_definition(
        asset_store=asset_store,
        scene_name=scene_name,
        inheritance_chain=(),
    )


def _load_scene_definition(
    *,
    asset_store: ScenePromptAssetStoreProtocol,
    scene_name: str,
    inheritance_chain: tuple[str, ...],
) -> SceneDefinition:
    """加载并解析指定 scene 的完整定义。

    Args:
        asset_store: scene prompt 资产仓储。
        scene_name: scene 名称。
        inheritance_chain: 当前递归加载中的继承链。

    Returns:
        处理继承后的 ``SceneDefinition``。

    Raises:
        PromptManifestError: manifest 非法时抛出。
    """

    if scene_name in inheritance_chain:
        raise PromptManifestError(
            f"scene extends 存在循环继承: {' -> '.join((*inheritance_chain, scene_name))}"
        )
    raw_manifest = asset_store.load_scene_manifest(scene_name)
    definition = parse_scene_definition(raw_manifest)
    if not definition.extends:
        normalized = definition
        if not definition._tool_selection_explicit:
            normalized = _replace_tool_selection_policy(normalized, ToolSelectionPolicy())
        if not definition._conversation_explicit:
            normalized = _replace_conversation_definition(normalized, SceneConversationDefinition())
        return normalized
    if len(definition.extends) != 1:
        raise PromptManifestError("第一版仅支持单继承 extends")
    parent_definition = _load_scene_definition(
        asset_store=asset_store,
        scene_name=definition.extends[0],
        inheritance_chain=(*inheritance_chain, scene_name),
    )
    inherited_fragments = list(parent_definition.fragments)
    existing_ids = {fragment.id for fragment in inherited_fragments}
    existing_orders = {fragment.order for fragment in inherited_fragments}
    for fragment in definition.fragments:
        if fragment.id in existing_ids:
            raise PromptManifestError(f"fragment.id 重复: {fragment.id}")
        if fragment.order in existing_orders:
            raise PromptManifestError(f"fragment.order 重复: {fragment.order}")
        inherited_fragments.append(fragment)
    merged_context_slots = list(parent_definition.context_slots)
    for slot_name in definition.context_slots:
        if slot_name in merged_context_slots:
            continue
        merged_context_slots.append(slot_name)
    merged_tool_selection = (
        definition.tool_selection_policy
        if definition._tool_selection_explicit
        else parent_definition.tool_selection_policy
    )
    merged_runtime = (
        _merge_scene_runtime_definition(parent_definition.runtime, definition.runtime)
        if definition._runtime_explicit
        else parent_definition.runtime
    )
    merged_conversation = (
        definition.conversation
        if definition._conversation_explicit
        else parent_definition.conversation
    )
    return SceneDefinition(
        name=definition.name,
        model=definition.model,
        version=definition.version,
        description=definition.description,
        runtime=merged_runtime,
        extends=definition.extends,
        missing_fragment_policy=definition.missing_fragment_policy,
        fragments=tuple(inherited_fragments),
        context_slots=tuple(merged_context_slots),
        conversation=merged_conversation,
        tool_selection_policy=merged_tool_selection,
        _runtime_explicit=definition._runtime_explicit,
        _conversation_explicit=definition._conversation_explicit,
        _tool_selection_explicit=definition._tool_selection_explicit,
    )


def _merge_scene_runtime_definition(
    base_runtime: SceneRuntimeDefinition,
    override_runtime: SceneRuntimeDefinition,
) -> SceneRuntimeDefinition:
    """按 agent/runner 子结构对 scene runtime 做稀疏覆盖合并。"""

    return SceneRuntimeDefinition(
        agent=SceneAgentRuntimeDefinition(
            max_iterations=(
                override_runtime.agent.max_iterations
                if override_runtime.agent.max_iterations is not None
                else base_runtime.agent.max_iterations
            ),
            max_consecutive_failed_tool_batches=(
                override_runtime.agent.max_consecutive_failed_tool_batches
                if override_runtime.agent.max_consecutive_failed_tool_batches is not None
                else base_runtime.agent.max_consecutive_failed_tool_batches
            ),
        ),
        runner=SceneRunnerRuntimeDefinition(
            tool_timeout_seconds=(
                override_runtime.runner.tool_timeout_seconds
                if override_runtime.runner.tool_timeout_seconds is not None
                else base_runtime.runner.tool_timeout_seconds
            ),
        ),
    )


def _replace_tool_selection_policy(
    definition: SceneDefinition,
    tool_selection_policy: ToolSelectionPolicy,
) -> SceneDefinition:
    """返回替换了工具选择策略的 scene 定义副本。

    Args:
        definition: 原始 scene 定义。
        tool_selection_policy: 替换后的工具选择策略。

    Returns:
        新的 ``SceneDefinition``。

    Raises:
        无。
    """

    return SceneDefinition(
        name=definition.name,
        model=definition.model,
        version=definition.version,
        description=definition.description,
        runtime=definition.runtime,
        extends=definition.extends,
        missing_fragment_policy=definition.missing_fragment_policy,
        fragments=definition.fragments,
        context_slots=definition.context_slots,
        conversation=definition.conversation,
        tool_selection_policy=tool_selection_policy,
        _runtime_explicit=definition._runtime_explicit,
        _conversation_explicit=definition._conversation_explicit,
        _tool_selection_explicit=definition._tool_selection_explicit,
    )


def _replace_conversation_definition(
    definition: SceneDefinition,
    conversation: SceneConversationDefinition,
) -> SceneDefinition:
    """返回替换了对话模式定义的 scene 定义副本。"""

    return SceneDefinition(
        name=definition.name,
        model=definition.model,
        version=definition.version,
        description=definition.description,
        runtime=definition.runtime,
        extends=definition.extends,
        missing_fragment_policy=definition.missing_fragment_policy,
        fragments=definition.fragments,
        context_slots=definition.context_slots,
        conversation=conversation,
        tool_selection_policy=definition.tool_selection_policy,
        _runtime_explicit=definition._runtime_explicit,
        _conversation_explicit=definition._conversation_explicit,
        _tool_selection_explicit=definition._tool_selection_explicit,
    )


def _parse_context_slots(data: list[str] | object | None) -> tuple[str, ...]:
    """解析 scene 声明的 Prompt Contributions slot 列表。

    Args:
        data: 原始 context_slots 配置。

    Returns:
        过滤空白后的 slot 名元组。

    Raises:
        PromptManifestError: 当输入不是字符串数组时抛出。
    """

    if data is None:
        return ()
    if not isinstance(data, list):
        raise PromptManifestError("context_slots 必须是字符串数组")
    normalized: list[str] = []
    for item in data:
        if not isinstance(item, str):
            raise PromptManifestError("context_slots 必须是字符串数组")
        slot_name = item.strip()
        if not slot_name:
            raise PromptManifestError("context_slots 不允许空字符串")
        if slot_name in normalized:
            raise PromptManifestError(f"context_slots 不允许重复: {slot_name}")
        normalized.append(slot_name)
    return tuple(normalized)


def _parse_scene_conversation_definition(data: SceneConversationAsset | object | None) -> SceneConversationDefinition:
    """解析 scene 对话模式配置。"""

    if data is None:
        return SceneConversationDefinition()
    if not isinstance(data, dict):
        raise PromptManifestError("conversation 必须是对象")
    raw_conversation = cast(SceneConversationAsset, data)
    if "enabled" not in raw_conversation or not isinstance(raw_conversation.get("enabled"), bool):
        raise PromptManifestError("conversation.enabled 必须是布尔值")
    return SceneConversationDefinition(enabled=bool(raw_conversation["enabled"]))


def _parse_scene_model_definition(data: SceneModelAsset | object | None) -> SceneModelDefinition:
    """解析 scene 默认模型配置。

    Args:
        data: 原始 ``model`` 字段。

    Returns:
        标准化后的 ``SceneModelDefinition``。

    Raises:
        PromptManifestError: 字段非法时抛出。
    """

    if not isinstance(data, dict):
        raise PromptManifestError("scene manifest 缺少 model")
    raw_model = cast(SceneModelAsset, data)
    if "max_iterations" in raw_model:
        raise PromptManifestError(
            "scene manifest.model.max_iterations 已迁移到 runtime.agent.max_iterations"
        )
    if "max_consecutive_failed_tool_batches" in raw_model:
        raise PromptManifestError(
            "scene manifest.model.max_consecutive_failed_tool_batches 已迁移到 "
            "runtime.agent.max_consecutive_failed_tool_batches"
        )
    raw_default_name = raw_model.get("default_name")
    if not isinstance(raw_default_name, str):
        raise PromptManifestError("scene manifest.model 缺少 default_name")
    default_name = raw_default_name.strip()
    if not default_name:
        raise PromptManifestError("scene manifest.model.default_name 不能为空")
    raw_allowed_names = raw_model.get("allowed_names")
    if not isinstance(raw_allowed_names, list) or not raw_allowed_names:
        raise PromptManifestError("scene manifest.model.allowed_names 必须是非空数组")
    allowed_names: list[str] = []
    seen_names: set[str] = set()
    for raw_name in raw_allowed_names:
        model_name = str(raw_name).strip()
        if not model_name:
            raise PromptManifestError("scene manifest.model.allowed_names 不能为空字符串")
        if model_name in seen_names:
            raise PromptManifestError(f"scene manifest.model.allowed_names 模型重复: {model_name}")
        seen_names.add(model_name)
        allowed_names.append(model_name)
    temperature_profile = str(raw_model.get("temperature_profile") or "").strip()
    if not temperature_profile:
        raise PromptManifestError("scene manifest.model.temperature_profile 不能为空")
    if default_name not in seen_names:
        raise PromptManifestError("scene manifest.model.default_name 必须出现在 allowed_names 中")
    return SceneModelDefinition(
        default_name=default_name,
        allowed_names=tuple(allowed_names),
        temperature_profile=temperature_profile,
    )


def _parse_optional_scene_model_positive_int(raw_value: object, *, field_name: str) -> int | None:
    """解析 scene model 的可选正整数配置。

    Args:
        raw_value: manifest 中字段的原始值。
        field_name: 字段名，仅用于错误提示。

    Returns:
        标准化后的正整数；未配置时返回 ``None``。

    Raises:
        PromptManifestError: 当值不是正整数时抛出。
    """

    if raw_value is None:
        return None
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise PromptManifestError(f"{field_name} 必须是正整数")
    if raw_value <= 0:
        raise PromptManifestError(f"{field_name} 必须是正整数")
    return raw_value


def _parse_optional_scene_runtime_positive_float(raw_value: object, *, field_name: str) -> float | None:
    """解析 scene runtime 的可选正数配置。"""

    if raw_value is None:
        return None
    if isinstance(raw_value, bool) or not isinstance(raw_value, int | float | str):
        raise PromptManifestError(f"{field_name} 必须是正数")
    try:
        parsed_value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise PromptManifestError(f"{field_name} 必须是正数") from exc
    if parsed_value <= 0:
        raise PromptManifestError(f"{field_name} 必须是正数")
    return parsed_value


def _parse_scene_agent_runtime_definition(data: SceneAgentRuntimeAsset | object | None) -> SceneAgentRuntimeDefinition:
    """解析 scene runtime.agent 子结构。"""

    if data is None:
        return SceneAgentRuntimeDefinition()
    if not isinstance(data, dict):
        raise PromptManifestError("scene manifest.runtime.agent 必须是对象")
    raw_runtime = cast(SceneAgentRuntimeAsset, data)
    return SceneAgentRuntimeDefinition(
        max_iterations=_parse_optional_scene_model_positive_int(
            raw_runtime.get("max_iterations"),
            field_name="scene manifest.runtime.agent.max_iterations",
        ),
        max_consecutive_failed_tool_batches=_parse_optional_scene_model_positive_int(
            raw_runtime.get("max_consecutive_failed_tool_batches"),
            field_name="scene manifest.runtime.agent.max_consecutive_failed_tool_batches",
        ),
    )


def _parse_scene_runner_runtime_definition(data: SceneRunnerRuntimeAsset | object | None) -> SceneRunnerRuntimeDefinition:
    """解析 scene runtime.runner 子结构。"""

    if data is None:
        return SceneRunnerRuntimeDefinition()
    if not isinstance(data, dict):
        raise PromptManifestError("scene manifest.runtime.runner 必须是对象")
    raw_runtime = cast(SceneRunnerRuntimeAsset, data)
    return SceneRunnerRuntimeDefinition(
        tool_timeout_seconds=_parse_optional_scene_runtime_positive_float(
            raw_runtime.get("tool_timeout_seconds"),
            field_name="scene manifest.runtime.runner.tool_timeout_seconds",
        )
    )


def _parse_scene_runtime_definition(data: SceneRuntimeAsset | object | None) -> SceneRuntimeDefinition:
    """解析 scene runtime 子结构。"""

    if data is None:
        return SceneRuntimeDefinition()
    if not isinstance(data, dict):
        raise PromptManifestError("scene manifest.runtime 必须是对象")
    raw_runtime = cast(SceneRuntimeAsset, data)
    return SceneRuntimeDefinition(
        agent=_parse_scene_agent_runtime_definition(raw_runtime.get("agent")),
        runner=_parse_scene_runner_runtime_definition(raw_runtime.get("runner")),
    )


def _parse_tool_selection_policy(data: SceneToolSelectionAsset | object | None) -> ToolSelectionPolicy:
    """解析工具选择策略。

    Args:
        data: 原始 ``tool_selection`` 字段。

    Returns:
        标准化后的 ``ToolSelectionPolicy``。

    Raises:
        PromptManifestError: 配置非法时抛出。
    """

    if data is None:
        return ToolSelectionPolicy()
    if not isinstance(data, dict):
        raise PromptManifestError("tool_selection 必须是对象")
    raw_policy = cast(SceneToolSelectionAsset, data)
    raw_mode = str(raw_policy.get("mode") or "").strip().lower()
    if not raw_mode:
        raise PromptManifestError("tool_selection.mode 不能为空")
    try:
        mode = ToolSelectionMode(raw_mode)
    except ValueError as exc:
        raise PromptManifestError(f"tool_selection.mode 非法: {raw_mode}") from exc
    raw_tags = raw_policy.get("tool_tags_any") or []
    if not isinstance(raw_tags, list):
        raise PromptManifestError("tool_selection.tool_tags_any 必须是数组")
    normalized_tags = tuple(str(item).strip() for item in raw_tags if str(item).strip())
    if mode == ToolSelectionMode.SELECT and not normalized_tags:
        raise PromptManifestError("tool_selection.mode=select 时 tool_tags_any 必须为非空数组")
    if mode in {ToolSelectionMode.ALL, ToolSelectionMode.NONE} and normalized_tags:
        raise PromptManifestError("tool_selection.mode=all/none 时不得配置 tool_tags_any")
    return ToolSelectionPolicy(mode=mode, tool_tags_any=normalized_tags)


def _parse_fragment_definition(data: SceneFragmentAsset | object) -> SceneFragmentDefinition:
    """解析单个 fragment 定义。

    Args:
        data: 原始 fragment 对象。

    Returns:
        解析后的 ``SceneFragmentDefinition``。

    Raises:
        PromptManifestError: fragment 非法时抛出。
    """

    if not isinstance(data, dict):
        raise PromptManifestError("fragment 必须是对象")
    raw_fragment = cast(SceneFragmentAsset, data)
    fragment_id = str(raw_fragment.get("id") or "").strip()
    fragment_type = str(raw_fragment.get("type") or "").strip()
    path = str(raw_fragment.get("path") or "").strip()
    order = raw_fragment.get("order")
    if not fragment_id or not fragment_type or not path:
        raise PromptManifestError("fragment 缺少 id/type/path")
    if not isinstance(order, int):
        raise PromptManifestError(f"fragment.order 非法: {fragment_id}")
    try:
        parsed_type = PromptFragmentType(fragment_type)
    except ValueError as exc:
        raise PromptManifestError(f"不支持的 fragment.type: {fragment_type}") from exc
    context_keys = raw_fragment.get("context_keys") or []
    if not isinstance(context_keys, list):
        raise PromptManifestError(f"fragment.context_keys 必须是数组: {fragment_id}")
    skip_if_context_missing = bool(raw_fragment.get("skip_if_context_missing", False))
    has_tool_filters = "tool_filters" in raw_fragment
    if parsed_type == PromptFragmentType.TOOLS and has_tool_filters:
        raise PromptManifestError("TOOLS fragment 不允许配置 tool_filters")
    tool_filters = raw_fragment.get("tool_filters") or {}
    if not isinstance(tool_filters, dict):
        raise PromptManifestError(f"fragment.tool_filters 必须是对象: {fragment_id}")
    return SceneFragmentDefinition(
        id=fragment_id,
        type=parsed_type,
        path=path,
        order=order,
        required=bool(raw_fragment.get("required", True)),
        context_keys=tuple(str(item) for item in context_keys),
        skip_if_context_missing=skip_if_context_missing,
        enabled=bool(raw_fragment.get("enabled", True)),
        tool_filters={key: [str(item) for item in value] for key, value in tool_filters.items()},
    )


__all__ = [
    "PromptFragmentType",
    "PromptManifestError",
    "SceneConversationDefinition",
    "SceneDefinition",
    "SceneFragmentDefinition",
    "ToolSelectionMode",
    "ToolSelectionPolicy",
    "load_scene_definition",
    "parse_scene_definition",
]
