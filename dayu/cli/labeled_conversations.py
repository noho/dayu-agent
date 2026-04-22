"""CLI 带标签对话辅助逻辑。

本模块只承载 `prompt --label` / `interactive --label` 共用的
UI 层辅助逻辑：
- 清理已漂移或已关闭的 label registry record
- 校验带标签对话绑定的 scene 必须启用多轮会话
- 解析最终使用的 `(session_id, scene_name, created)`
"""

from __future__ import annotations

from dataclasses import dataclass

from dayu.cli.conversation_labels import FileConversationLabelRegistry
from dayu.contracts.infrastructure import PromptAssetStoreProtocol
from dayu.prompting.scene_definition import PromptManifestError, load_scene_definition
from dayu.services.protocols import HostAdminServiceProtocol


@dataclass(frozen=True)
class LabeledConversationTarget:
    """带标签对话解析结果。"""

    session_id: str
    scene_name: str
    created: bool
    recreated_from_closed: bool = False


def resolve_labeled_conversation_target(
    *,
    registry: FileConversationLabelRegistry,
    prompt_asset_store: PromptAssetStoreProtocol | None,
    label: str,
    default_scene_name: str,
    explicit_session_id: str | None,
    explicit_scene_name: str,
    host_admin_service: HostAdminServiceProtocol | None = None,
) -> LabeledConversationTarget:
    """解析带标签对话对应的稳定 session 与 scene。

    Args:
        registry: CLI label registry。
        prompt_asset_store: prompt 资产仓储；提供时会校验 scene 是否允许多轮会话。
        label: 已规范化的 conversation label。
        default_scene_name: 首次创建 label 时采用的默认 scene。
        explicit_session_id: 可选显式 session_id。
        explicit_scene_name: 显式 scene 名或调用方默认 scene。
        host_admin_service: 可选 HostAdmin service；提供时会清理不可恢复 record。

    Returns:
        解析后的带标签对话目标。

    Raises:
        ValueError: 当 scene 不存在、scene 未启用 conversation，或 registry record 非法时抛出。
        OSError: 清理不可恢复 record 失败时抛出。
    """

    if explicit_session_id is not None:
        _ensure_labeled_scene_is_conversational(prompt_asset_store, explicit_scene_name)
        return LabeledConversationTarget(
            session_id=explicit_session_id,
            scene_name=explicit_scene_name,
            created=False,
        )
    recreated_from_closed = prune_unavailable_label_record(
        registry=registry,
        host_admin_service=host_admin_service,
        label=label,
    )
    resolution = registry.get_or_create_record(
        label=label,
        scene_name=default_scene_name,
    )
    _ensure_labeled_scene_is_conversational(prompt_asset_store, resolution.record.scene_name)
    return LabeledConversationTarget(
        session_id=resolution.record.session_id,
        scene_name=resolution.record.scene_name,
        created=resolution.created,
        recreated_from_closed=recreated_from_closed and resolution.created,
    )


def prune_unavailable_label_record(
    *,
    registry: FileConversationLabelRegistry,
    host_admin_service: HostAdminServiceProtocol | None,
    label: str,
) -> bool:
    """删除已不可恢复的 label record。

    Args:
        registry: CLI label registry。
        host_admin_service: 可选 HostAdmin service；未提供时不做清理。
        label: 已规范化的 conversation label。

    Returns:
        若因命中 closed session 而清理返回 ``True``；其它情况返回 ``False``。

    Raises:
        ValueError: registry record 非法时抛出。
        OSError: 清理不可恢复 record 失败时抛出。
    """

    if host_admin_service is None:
        return False
    record = registry.get_record(label)
    if record is None:
        return False
    session = host_admin_service.get_session(record.session_id)
    if session is not None and session.state != "closed":
        return False
    registry.delete_record(label)
    return session is not None and session.state == "closed"


def _ensure_labeled_scene_is_conversational(
    prompt_asset_store: PromptAssetStoreProtocol | None,
    scene_name: str,
) -> None:
    """校验带标签对话使用的 scene 已开启多轮模式。

    Args:
        prompt_asset_store: prompt 资产仓储；未提供时跳过校验。
        scene_name: 待校验的 scene 名。

    Returns:
        无。

    Raises:
        ValueError: 当 scene 不存在或未启用 `conversation.enabled=true` 时抛出。
    """

    if prompt_asset_store is None:
        return
    try:
        scene_definition = load_scene_definition(prompt_asset_store, scene_name)
    except FileNotFoundError as exc:
        raise ValueError(f"scene 不存在: {scene_name}") from exc
    except PromptManifestError as exc:
        raise ValueError(f"scene 非法: {scene_name}") from exc
    if scene_definition.conversation.enabled:
        return
    raise ValueError(
        "带标签对话要求 scene 开启 conversation.enabled=true: "
        f"scene={scene_name}"
    )


__all__ = [
    "LabeledConversationTarget",
    "prune_unavailable_label_record",
    "resolve_labeled_conversation_target",
]
