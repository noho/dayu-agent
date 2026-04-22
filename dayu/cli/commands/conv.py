"""CLI labeled conversation 管理命令。

提供 `conv list` 与 `conv status` 两个子命令，
用于读取 CLI label registry，并联查 Host session 的状态与摘要信息。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from dayu.cli.commands.host import (
    _build_host_runtime,
    _format_datetime_iso,
    _format_table_cell,
    _resolve_host_admin_service,
)
from dayu.cli.conversation_labels import (
    ConversationLabelRecord,
    FileConversationLabelRegistry,
    validate_conversation_label,
)
from dayu.cli.dependency_setup import setup_loglevel
from dayu.services.contracts import SessionAdminView
from dayu.services.protocols import HostAdminServiceProtocol


_LABEL_COLUMN_WIDTH = 20
_SESSION_ID_COLUMN_WIDTH = 36
_SOURCE_COLUMN_WIDTH = 10
_SCENE_COLUMN_WIDTH = 16
_STATE_COLUMN_WIDTH = 10
_LAST_ACTIVITY_COLUMN_WIDTH = 20
_OVERVIEW_COLUMN_WIDTH = 48
_MISSING_VALUE = "-"
_MISSING_SESSION_STATE = "missing"
_EMPTY_LIST_MESSAGE = "无 labeled conversation 记录"


@dataclass(frozen=True)
class ConversationRow:
    """`conv` 命令输出的一行稳定视图。"""

    label: str
    session_id: str
    source: str
    scene_name: str
    state: str
    last_activity_at: str
    overview: str


def run_conv_command(args: argparse.Namespace) -> int:
    """分发 `conv` 子命令。

    Args:
        args: 解析后的命令行参数。

    Returns:
        命令退出码。

    Raises:
        无。
    """

    setup_loglevel(args)
    action = str(getattr(args, "conv_action", "") or "").strip().lower()
    if action == "list":
        return _run_conv_list_command(args)
    if action == "status":
        return _run_conv_status_command(args)
    return 1


def _run_conv_list_command(args: argparse.Namespace) -> int:
    """执行 `conv list`。

    Args:
        args: 命令行参数。

    Returns:
        命令退出码。

    Raises:
        ValueError: registry record 非法时抛出。
    """

    registry, service = _build_conv_dependencies(args)
    records = registry.list_records()
    if not records:
        print(_EMPTY_LIST_MESSAGE)
        return 0

    sessions_by_id = {
        session.session_id: session
        for session in service.list_sessions(source="cli")
    }
    rows = tuple(
        _build_conversation_row(record=record, session=sessions_by_id.get(record.session_id))
        for record in records
    )
    _print_conversation_rows(rows)
    return 0


def _run_conv_status_command(args: argparse.Namespace) -> int:
    """执行 `conv status --label <label>`。

    Args:
        args: 命令行参数。

    Returns:
        命令退出码。

    Raises:
        ValueError: label 或 registry record 非法时抛出。
    """

    registry, service = _build_conv_dependencies(args)
    label = validate_conversation_label(str(getattr(args, "label", "") or ""))
    record = registry.get_record(label)
    if record is None:
        print(f"label 不存在: {label}", file=sys.stderr)
        return 1

    row = _build_conversation_row(
        record=record,
        session=service.get_session(record.session_id),
    )
    _print_conversation_rows((row,))
    return 0


def _build_conv_dependencies(
    args: argparse.Namespace,
) -> tuple[FileConversationLabelRegistry, HostAdminServiceProtocol]:
    """构造 `conv` 命令所需的 registry 与 HostAdmin service。

    Args:
        args: 命令行参数。

    Returns:
        二元组 `(registry, service)`。

    Raises:
        AttributeError: 运行时缺少 `host_admin_service` 时抛出。
    """

    runtime = _build_host_runtime(args)
    workspace_root = _resolve_workspace_root(runtime.paths.workspace_root)
    registry = FileConversationLabelRegistry(workspace_root)
    service = _resolve_host_admin_service(runtime)
    return registry, service


def _resolve_workspace_root(workspace_root: Path) -> Path:
    """规范化工作区根目录路径。

    Args:
        workspace_root: runtime 暴露的工作区目录。

    Returns:
        规范化后的工作区绝对路径。

    Raises:
        无。
    """

    return Path(workspace_root).expanduser().resolve()


def _build_conversation_row(
    *,
    record: ConversationLabelRecord,
    session: SessionAdminView | None,
) -> ConversationRow:
    """把 registry record 与可选 Host session 拼装为渲染行。

    Args:
        record: CLI label registry 记录。
        session: Host session 视图；缺失时为 `None`。

    Returns:
        可直接用于 CLI 表格输出的一行数据。

    Raises:
        无。
    """

    if session is None:
        return ConversationRow(
            label=record.label,
            session_id=record.session_id,
            source=record.source,
            scene_name=record.scene_name,
            state=_MISSING_SESSION_STATE,
            last_activity_at=_MISSING_VALUE,
            overview=_MISSING_VALUE,
        )
    return ConversationRow(
        label=record.label,
        session_id=record.session_id,
        source=record.source,
        scene_name=record.scene_name,
        state=session.state,
        last_activity_at=session.last_activity_at,
        overview=_resolve_conversation_overview(session),
    )


def _resolve_conversation_overview(session: SessionAdminView) -> str:
    """按文档约定解析会话概览文本。

    Args:
        session: Host session 摘要视图。

    Returns:
        依次取首问预览、末问预览，均为空时返回 `-`。

    Raises:
        无。
    """

    return session.first_question_preview or session.last_question_preview or _MISSING_VALUE


def _print_conversation_rows(rows: tuple[ConversationRow, ...]) -> None:
    """打印 `conv` 命令表格结果。

    Args:
        rows: 待打印的数据行。

    Returns:
        无。

    Raises:
        无。
    """

    header = (
        f"{_format_table_cell('LABEL', _LABEL_COLUMN_WIDTH)} "
        f"{'SESSION_ID':<{_SESSION_ID_COLUMN_WIDTH}} "
        f"{'SOURCE':<{_SOURCE_COLUMN_WIDTH}} "
        f"{_format_table_cell('SCENE', _SCENE_COLUMN_WIDTH)} "
        f"{'STATE':<{_STATE_COLUMN_WIDTH}} "
        f"{'LAST_ACTIVITY':<{_LAST_ACTIVITY_COLUMN_WIDTH}} "
        f"{_format_table_cell('OVERVIEW', _OVERVIEW_COLUMN_WIDTH)}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{_format_table_cell(row.label, _LABEL_COLUMN_WIDTH)} "
            f"{row.session_id:<{_SESSION_ID_COLUMN_WIDTH}} "
            f"{row.source:<{_SOURCE_COLUMN_WIDTH}} "
            f"{_format_table_cell(row.scene_name, _SCENE_COLUMN_WIDTH)} "
            f"{row.state:<{_STATE_COLUMN_WIDTH}} "
            f"{_format_datetime_iso(row.last_activity_at):<{_LAST_ACTIVITY_COLUMN_WIDTH}} "
            f"{_format_table_cell(row.overview, _OVERVIEW_COLUMN_WIDTH)}"
        )
