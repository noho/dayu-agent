"""工作区初始化与默认模型切换公共能力。

该模块提供无交互、可复用的启动期能力，供 CLI 与 Web 共同调用：
- 初始化工作区配置与资产目录
- 执行工作区一次性迁移
- 更新 scene manifest 的默认模型
- 读取可选模型名列表
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from dayu.contracts.infrastructure import StructuredConfigValue
from dayu.startup.config_file_resolver import (
    ConfigFileResolver,
    resolve_package_assets_path,
    resolve_package_config_path,
)
from dayu.startup.config_loader import ConfigLoader
from dayu.workspace_paths import build_dayu_root_path, build_host_store_default_path

_RUN_JSON_FILENAME = "run.json"
_HOST_CONFIG_KEY = "host_config"
_LANE_KEY = "lane"
_WRITE_CHAPTER_LANE = "write_chapter"
_DEFAULT_WRITE_CHAPTER_CONCURRENCY = 5

_OLD_LANE_KEY = "concurrency_lane"
_NEW_LANE_KEY = "business_concurrency_lane"
_OLD_LANE_JSON_TOKEN = f'"{_OLD_LANE_KEY}"'
_PENDING_TURN_TABLE_NAME = "pending_conversation_turns"
_PENDING_TURN_JSON_COLUMN = "resume_source_json"
_PENDING_TURN_ID_COLUMN = "pending_turn_id"

_INIT_ROLE_KEY = "_init_model_role"
_ROLE_NON_THINKING = "non_thinking"
_ROLE_THINKING = "thinking"


@dataclass(frozen=True)
class WorkspaceMigrationResult:
    """工作区迁移执行结果。"""

    run_json_write_chapter_added: bool
    host_store_lane_rows_rewritten: int


@dataclass(frozen=True)
class WorkspaceInitializationResult:
    """工作区初始化结果。"""

    config_dir: Path
    assets_dir: Path
    migration_result: WorkspaceMigrationResult


def build_workspace_reset_targets(base_dir: Path) -> tuple[Path, ...]:
    """构造 reset 需要处理的工作区目标路径。

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


def reset_workspace_init_targets(base_dir: Path) -> tuple[Path, ...]:
    """删除初始化产物与运行时状态目录。

    Args:
        base_dir: 工作区根目录。

    Returns:
        实际被删除的目标路径元组。

    Raises:
        OSError: 删除文件或目录失败时抛出。
    """

    removed_targets: list[Path] = []
    for target in build_workspace_reset_targets(base_dir):
        if _remove_workspace_target(target):
            removed_targets.append(target)
    return tuple(removed_targets)


def initialize_workspace_configuration(base_dir: Path, *, overwrite: bool) -> WorkspaceInitializationResult:
    """初始化工作区配置、资产并执行迁移。

    Args:
        base_dir: 工作区根目录。
        overwrite: 是否覆盖已有 `config` / `assets` 目录。

    Returns:
        初始化结果，包含 config/assets 路径与迁移结果。

    Raises:
        OSError: 复制目录或写入迁移结果失败时抛出。
    """

    config_dir = _copy_config(base_dir=base_dir, overwrite=overwrite)
    assets_dir = _copy_assets(base_dir=base_dir, overwrite=overwrite)
    migration_result = apply_workspace_migrations(base_dir=base_dir, config_dir=config_dir)
    return WorkspaceInitializationResult(
        config_dir=config_dir,
        assets_dir=assets_dir,
        migration_result=migration_result,
    )


def apply_workspace_migrations(*, base_dir: Path, config_dir: Path) -> WorkspaceMigrationResult:
    """执行全部工作区迁移并返回结构化结果。

    Args:
        base_dir: 工作区根目录。
        config_dir: 工作区配置目录。

    Returns:
        迁移执行结果。

    Raises:
        无：迁移异常会被吞掉并体现在返回值中。
    """

    run_json_added = _migrate_run_json_add_write_chapter_lane(config_dir)
    host_store_rows = _migrate_host_store_rename_concurrency_lane(
        build_host_store_default_path(base_dir)
    )
    return WorkspaceMigrationResult(
        run_json_write_chapter_added=run_json_added,
        host_store_lane_rows_rewritten=host_store_rows,
    )


def update_manifest_default_models(
    config_dir: Path,
    *,
    non_thinking_model: str,
    thinking_model: str,
) -> int:
    """按 scene 角色更新 manifest 的默认模型。

    优先使用 manifest 内 `_init_model_role` 角色标记；若缺失，则回退到包内
    同名 manifest 的默认模型推断角色。

    Args:
        config_dir: 工作区配置目录。
        non_thinking_model: 目标 non-thinking 模型名。
        thinking_model: 目标 thinking 模型名。

    Returns:
        被更新的 manifest 文件数量。

    Raises:
        OSError: 读取或写入 manifest 文件失败时抛出。
        json.JSONDecodeError: manifest 文件 JSON 非法时抛出。
    """

    manifests_dir = config_dir / "prompts" / "manifests"
    if not manifests_dir.exists():
        return 0

    role_to_model = {
        _ROLE_NON_THINKING: non_thinking_model,
        _ROLE_THINKING: thinking_model,
    }

    updated_count = 0
    for manifest_file in sorted(manifests_dir.glob("*.json")):
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        model_section = data.get("model")
        if not isinstance(model_section, dict):
            continue

        current_model = model_section.get("default_name")
        if not isinstance(current_model, str):
            continue

        stored_role = model_section.get(_INIT_ROLE_KEY, "")
        role = _resolve_manifest_role(
            manifest_filename=manifest_file.name,
            stored_role=stored_role if isinstance(stored_role, str) else "",
        )
        if role is None:
            continue

        target_model = role_to_model[role]
        changed = target_model != current_model or stored_role != role

        allowed_names = model_section.get("allowed_names")
        if isinstance(allowed_names, list) and target_model not in allowed_names:
            allowed_names.append(target_model)
            changed = True

        if not changed:
            continue

        model_section["default_name"] = target_model
        model_section[_INIT_ROLE_KEY] = role
        manifest_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        updated_count += 1
    return updated_count


def load_available_model_names(config_dir: Path) -> tuple[str, ...]:
    """读取当前配置目录可用的模型名集合。

    Args:
        config_dir: 工作区配置目录。

    Returns:
        去重并排序后的模型名元组。

    Raises:
        FileNotFoundError: `llm_models.json` 缺失时抛出。
        TypeError: `llm_models.json` 结构非法时抛出。
        json.JSONDecodeError: `llm_models.json` 解析失败时抛出。
    """

    config_loader = ConfigLoader(ConfigFileResolver(config_dir))
    model_names = tuple(sorted(config_loader.load_llm_models().keys()))
    return model_names


def _remove_workspace_target(target: Path) -> bool:
    """删除单个目标路径。

    Args:
        target: 需要删除的路径。

    Returns:
        实际删除返回 `True`；不存在返回 `False`。

    Raises:
        OSError: 删除失败时抛出。
    """

    if target.is_symlink() or target.is_file():
        target.unlink()
        return True
    if target.is_dir():
        shutil.rmtree(target)
        return True
    return False


def _copy_config(*, base_dir: Path, overwrite: bool) -> Path:
    """复制包内 config 到工作区。

    Args:
        base_dir: 工作区根目录。
        overwrite: 是否覆盖已有目录。

    Returns:
        目标 config 目录路径。

    Raises:
        OSError: 复制目录失败时抛出。
    """

    source_dir = resolve_package_config_path()
    target_dir = (base_dir / "config").resolve()
    if target_dir.exists() and not overwrite:
        return target_dir
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    return target_dir


def _copy_assets(*, base_dir: Path, overwrite: bool) -> Path:
    """复制包内 assets 到工作区。

    Args:
        base_dir: 工作区根目录。
        overwrite: 是否覆盖已有目录。

    Returns:
        目标 assets 目录路径。

    Raises:
        OSError: 复制目录失败时抛出。
    """

    source_dir = resolve_package_assets_path()
    target_dir = (base_dir / "assets").resolve()
    if target_dir.exists() and not overwrite:
        return target_dir
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    return target_dir


def _resolve_manifest_role(manifest_filename: str, stored_role: str) -> str | None:
    """解析 manifest 的模型角色。

    Args:
        manifest_filename: manifest 文件名。
        stored_role: 当前 manifest 中记录的 `_init_model_role`。

    Returns:
        `non_thinking` / `thinking`；无法判断时返回 `None`。

    Raises:
        无。
    """

    if stored_role in (_ROLE_NON_THINKING, _ROLE_THINKING):
        return stored_role
    return _resolve_role_from_package_manifest(manifest_filename)


def _resolve_role_from_package_manifest(manifest_filename: str) -> str | None:
    """从包内原始 manifest 推断角色。

    Args:
        manifest_filename: manifest 文件名。

    Returns:
        `non_thinking` / `thinking`；无法判断时返回 `None`。

    Raises:
        无。
    """

    package_manifest_path = resolve_package_config_path() / "prompts" / "manifests" / manifest_filename
    if not package_manifest_path.exists():
        return None
    try:
        payload = json.loads(package_manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    model_section = payload.get("model")
    if not isinstance(model_section, dict):
        return None
    default_name = model_section.get("default_name")
    if not isinstance(default_name, str):
        return None
    if "thinking" in default_name:
        return _ROLE_THINKING
    return _ROLE_NON_THINKING


def _migrate_run_json_add_write_chapter_lane(config_dir: Path) -> bool:
    """为旧工作区 run.json 补齐 `write_chapter` lane。

    Args:
        config_dir: 工作区配置目录。

    Returns:
        发生写入返回 `True`，否则返回 `False`。

    Raises:
        无：异常被吞掉并返回 `False`。
    """

    run_json_path = config_dir / _RUN_JSON_FILENAME
    if not run_json_path.exists():
        return False
    try:
        payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    host_config = payload.get(_HOST_CONFIG_KEY)
    if not isinstance(host_config, dict):
        return False
    lane_config = host_config.get(_LANE_KEY)
    if not isinstance(lane_config, dict):
        return False
    if _WRITE_CHAPTER_LANE in lane_config:
        return False
    lane_config[_WRITE_CHAPTER_LANE] = _DEFAULT_WRITE_CHAPTER_CONCURRENCY
    run_json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def _migrate_host_store_rename_concurrency_lane(host_store_path: Path) -> int:
    """改写 Host SQLite 旧字段 `concurrency_lane`。

    Args:
        host_store_path: Host SQLite 文件路径。

    Returns:
        实际改写行数。

    Raises:
        无：异常被吞掉并返回 0。
    """

    if not host_store_path.exists():
        return 0
    try:
        connection = sqlite3.connect(str(host_store_path))
    except sqlite3.Error:
        return 0

    try:
        connection.row_factory = sqlite3.Row
        if not _table_exists(connection, _PENDING_TURN_TABLE_NAME):
            return 0
        rewritten_rows = 0
        rows = connection.execute(
            f"SELECT {_PENDING_TURN_ID_COLUMN}, {_PENDING_TURN_JSON_COLUMN} FROM {_PENDING_TURN_TABLE_NAME}"  # noqa: S608
        ).fetchall()
        for row in rows:
            raw_json = row[_PENDING_TURN_JSON_COLUMN]
            if not isinstance(raw_json, str) or not raw_json:
                continue
            if _OLD_LANE_JSON_TOKEN not in raw_json:
                continue
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            if not _rename_lane_key_in_place(payload):
                continue
            connection.execute(
                f"UPDATE {_PENDING_TURN_TABLE_NAME} SET {_PENDING_TURN_JSON_COLUMN} = ? WHERE {_PENDING_TURN_ID_COLUMN} = ?",  # noqa: S608
                (json.dumps(payload, ensure_ascii=False, sort_keys=True), row[_PENDING_TURN_ID_COLUMN]),
            )
            rewritten_rows += 1
        if rewritten_rows:
            connection.commit()
        return rewritten_rows
    except sqlite3.Error:
        try:
            connection.rollback()
        except sqlite3.Error:
            pass
        return 0
    finally:
        connection.close()


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    """判断 SQLite 表是否存在。

    Args:
        connection: SQLite 连接对象。
        table_name: 表名。

    Returns:
        表存在返回 `True`。

    Raises:
        sqlite3.Error: SQL 执行异常时抛出。
    """

    result = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return result is not None


def _rename_lane_key_in_place(value: StructuredConfigValue) -> bool:
    """递归重命名 JSON 结构中的并发 lane 字段。

    Args:
        value: 任意 JSON 结构值。

    Returns:
        至少有一处变更时返回 `True`。

    Raises:
        无。
    """

    changed = False
    if isinstance(value, dict):
        if _OLD_LANE_KEY in value and _NEW_LANE_KEY not in value:
            value[_NEW_LANE_KEY] = value.pop(_OLD_LANE_KEY)
            changed = True
        elif _OLD_LANE_KEY in value and _NEW_LANE_KEY in value:
            value.pop(_OLD_LANE_KEY)
            changed = True
        for child in value.values():
            if _rename_lane_key_in_place(child):
                changed = True
        return changed
    if isinstance(value, list):
        for child in value:
            if _rename_lane_key_in_place(child):
                changed = True
    return changed


__all__ = [
    "WorkspaceInitializationResult",
    "WorkspaceMigrationResult",
    "apply_workspace_migrations",
    "build_workspace_reset_targets",
    "initialize_workspace_configuration",
    "load_available_model_names",
    "reset_workspace_init_targets",
    "update_manifest_default_models",
]
