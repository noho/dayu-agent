"""跨层共享的工具配置契约。

该模块承载会在 ``Service / Contracts / Host / Engine / Fins`` 之间流动的稳定
工具配置对象，并提供从通用 ``ToolsetConfigSnapshot`` 恢复专用配置的辅助函数。
"""

from __future__ import annotations

from dataclasses import dataclass

from dayu.contracts.toolset_config import (
    ToolsetConfigSnapshot,
    coerce_toolset_config_float,
    coerce_toolset_config_int,
    find_toolset_config,
    replace_toolset_config,
    build_toolset_config_snapshot,
)


@dataclass(frozen=True)
class DocToolLimits:
    """文档工具限制配置。

    Args:
        list_files_max: ``list_files`` 最大返回文件数。
        get_sections_max: ``get_file_sections`` 最大返回章节数。
        search_files_max_results: ``search_files`` 最大返回命中数。
        read_file_max_chars: ``read_file`` 最大返回字符数。
        read_file_section_max_chars: ``read_file_section`` 最大返回字符数。

    Returns:
        无。

    Raises:
        无。
    """

    list_files_max: int = 200
    get_sections_max: int = 200
    search_files_max_results: int = 50
    read_file_max_chars: int = 80_000
    read_file_section_max_chars: int = 50_000


@dataclass(frozen=True)
class FinsToolLimits:
    """财报工具限制配置。

    Args:
        processor_cache_max_entries: Processor 缓存最大条目数。
        list_documents_max_items: ``list_documents`` 返回最大文档条目数。
        get_document_sections_max_items: ``get_document_sections`` 返回最大章节条目数。
        search_document_max_items: ``search_document`` 返回最大命中条目数。
        list_tables_max_items: ``list_tables`` 返回最大表格条目数。
        read_section_max_chars: ``read_section`` 文本最大字符数。
        get_page_content_max_chars: ``get_page_content`` 文本最大字符数。
        get_table_max_items: ``get_table`` 列表数据最大条目数。
        get_financial_statement_max_items: ``get_financial_statement`` 列表数据最大条目数。
        query_xbrl_facts_max_items: ``query_xbrl_facts`` 列表数据最大条目数。

    Returns:
        无。

    Raises:
        无。
    """

    processor_cache_max_entries: int = 128
    list_documents_max_items: int = 300
    get_document_sections_max_items: int = 1200
    search_document_max_items: int = 20
    list_tables_max_items: int = 50
    read_section_max_chars: int = 80000
    get_page_content_max_chars: int = 80000
    get_table_max_items: int = 800
    get_financial_statement_max_items: int = 1200
    query_xbrl_facts_max_items: int = 1200


@dataclass(frozen=True)
class WebToolsConfig:
    """联网工具配置。

    Args:
        provider: provider 策略。
        request_timeout_seconds: HTTP 请求超时秒数。
        max_search_results: ``search_web`` 结果数量上限。
        fetch_truncate_chars: ``fetch_web_page`` 正文截断字符上限。
        allow_private_network_url: 是否允许访问内网/本地网络 URL。
        playwright_channel: 浏览器回退使用的 Chromium channel。
        playwright_storage_state_dir: Playwright storage state 目录。

    Returns:
        无。

    Raises:
        无。
    """

    provider: str = "auto"
    request_timeout_seconds: float = 12.0
    max_search_results: int = 20
    fetch_truncate_chars: int = 80000
    allow_private_network_url: bool = False
    playwright_channel: str = "chrome"
    playwright_storage_state_dir: str = "output/web_diagnostics/storage_states"


def build_doc_tool_limits(snapshot: ToolsetConfigSnapshot | None) -> DocToolLimits:
    """从通用 toolset 快照恢复文档工具限制。

    Args:
        snapshot: 文档工具的通用配置快照。

    Returns:
        恢复后的文档工具限制配置。

    Raises:
        TypeError: 当快照字段类型非法时抛出。
    """

    payload = snapshot.payload if snapshot is not None else {}
    defaults = DocToolLimits()
    return DocToolLimits(
        list_files_max=coerce_toolset_config_int(
            payload.get("list_files_max"),
            field_name="doc.list_files_max",
            default=defaults.list_files_max,
        ),
        get_sections_max=coerce_toolset_config_int(
            payload.get("get_sections_max"),
            field_name="doc.get_sections_max",
            default=defaults.get_sections_max,
        ),
        search_files_max_results=coerce_toolset_config_int(
            payload.get("search_files_max_results"),
            field_name="doc.search_files_max_results",
            default=defaults.search_files_max_results,
        ),
        read_file_max_chars=coerce_toolset_config_int(
            payload.get("read_file_max_chars"),
            field_name="doc.read_file_max_chars",
            default=defaults.read_file_max_chars,
        ),
        read_file_section_max_chars=coerce_toolset_config_int(
            payload.get("read_file_section_max_chars"),
            field_name="doc.read_file_section_max_chars",
            default=defaults.read_file_section_max_chars,
        ),
    )


def build_fins_tool_limits(snapshot: ToolsetConfigSnapshot | None) -> FinsToolLimits:
    """从通用 toolset 快照恢复财报工具限制。

    Args:
        snapshot: 财报工具的通用配置快照。

    Returns:
        恢复后的财报工具限制配置。

    Raises:
        TypeError: 当快照字段类型非法时抛出。
    """

    payload = snapshot.payload if snapshot is not None else {}
    defaults = FinsToolLimits()
    return FinsToolLimits(
        processor_cache_max_entries=coerce_toolset_config_int(
            payload.get("processor_cache_max_entries"),
            field_name="fins.processor_cache_max_entries",
            default=defaults.processor_cache_max_entries,
        ),
        list_documents_max_items=coerce_toolset_config_int(
            payload.get("list_documents_max_items"),
            field_name="fins.list_documents_max_items",
            default=defaults.list_documents_max_items,
        ),
        get_document_sections_max_items=coerce_toolset_config_int(
            payload.get("get_document_sections_max_items"),
            field_name="fins.get_document_sections_max_items",
            default=defaults.get_document_sections_max_items,
        ),
        search_document_max_items=coerce_toolset_config_int(
            payload.get("search_document_max_items"),
            field_name="fins.search_document_max_items",
            default=defaults.search_document_max_items,
        ),
        list_tables_max_items=coerce_toolset_config_int(
            payload.get("list_tables_max_items"),
            field_name="fins.list_tables_max_items",
            default=defaults.list_tables_max_items,
        ),
        read_section_max_chars=coerce_toolset_config_int(
            payload.get("read_section_max_chars"),
            field_name="fins.read_section_max_chars",
            default=defaults.read_section_max_chars,
        ),
        get_page_content_max_chars=coerce_toolset_config_int(
            payload.get("get_page_content_max_chars"),
            field_name="fins.get_page_content_max_chars",
            default=defaults.get_page_content_max_chars,
        ),
        get_table_max_items=coerce_toolset_config_int(
            payload.get("get_table_max_items"),
            field_name="fins.get_table_max_items",
            default=defaults.get_table_max_items,
        ),
        get_financial_statement_max_items=coerce_toolset_config_int(
            payload.get("get_financial_statement_max_items"),
            field_name="fins.get_financial_statement_max_items",
            default=defaults.get_financial_statement_max_items,
        ),
        query_xbrl_facts_max_items=coerce_toolset_config_int(
            payload.get("query_xbrl_facts_max_items"),
            field_name="fins.query_xbrl_facts_max_items",
            default=defaults.query_xbrl_facts_max_items,
        ),
    )


def build_web_tools_config(snapshot: ToolsetConfigSnapshot | None) -> WebToolsConfig:
    """从通用 toolset 快照恢复联网工具配置。

    Args:
        snapshot: 联网工具的通用配置快照。

    Returns:
        恢复后的联网工具配置。

    Raises:
        TypeError: 当快照字段类型非法时抛出。
    """

    payload = snapshot.payload if snapshot is not None else {}
    defaults = WebToolsConfig()
    return WebToolsConfig(
        provider=str(payload.get("provider", defaults.provider)),
        request_timeout_seconds=coerce_toolset_config_float(
            payload.get("request_timeout_seconds"),
            field_name="web.request_timeout_seconds",
            default=defaults.request_timeout_seconds,
        ),
        max_search_results=coerce_toolset_config_int(
            payload.get("max_search_results"),
            field_name="web.max_search_results",
            default=defaults.max_search_results,
        ),
        fetch_truncate_chars=coerce_toolset_config_int(
            payload.get("fetch_truncate_chars"),
            field_name="web.fetch_truncate_chars",
            default=defaults.fetch_truncate_chars,
        ),
        allow_private_network_url=bool(
            payload.get("allow_private_network_url", defaults.allow_private_network_url)
        ),
        playwright_channel=str(payload.get("playwright_channel", defaults.playwright_channel)),
        playwright_storage_state_dir=str(
            payload.get("playwright_storage_state_dir", defaults.playwright_storage_state_dir)
        ),
    )


def resolve_doc_tool_limits_from_toolset_configs(
    toolset_configs: tuple[ToolsetConfigSnapshot, ...],
) -> DocToolLimits | None:
    """从 toolset 配置序列中解析文档工具限制。

    Args:
        toolset_configs: 通用 toolset 配置快照序列。

    Returns:
        命中的文档工具限制；不存在时返回 ``None``。

    Raises:
        TypeError: 当快照字段类型非法时抛出。
    """

    snapshot = find_toolset_config(toolset_configs, "doc")
    if snapshot is None:
        return None
    return build_doc_tool_limits(snapshot)


def resolve_fins_tool_limits_from_toolset_configs(
    toolset_configs: tuple[ToolsetConfigSnapshot, ...],
) -> FinsToolLimits | None:
    """从 toolset 配置序列中解析财报工具限制。

    Args:
        toolset_configs: 通用 toolset 配置快照序列。

    Returns:
        命中的财报工具限制；不存在时返回 ``None``。

    Raises:
        TypeError: 当快照字段类型非法时抛出。
    """

    snapshot = find_toolset_config(toolset_configs, "fins")
    if snapshot is None:
        return None
    return build_fins_tool_limits(snapshot)


def resolve_web_tools_config_from_toolset_configs(
    toolset_configs: tuple[ToolsetConfigSnapshot, ...],
) -> WebToolsConfig | None:
    """从 toolset 配置序列中解析联网工具配置。

    Args:
        toolset_configs: 通用 toolset 配置快照序列。

    Returns:
        命中的联网工具配置；不存在时返回 ``None``。

    Raises:
        TypeError: 当快照字段类型非法时抛出。
    """

    snapshot = find_toolset_config(toolset_configs, "web")
    if snapshot is None:
        return None
    return build_web_tools_config(snapshot)


def build_legacy_toolset_configs(
    *,
    doc_tool_limits: DocToolLimits | None,
    fins_tool_limits: FinsToolLimits | None,
    web_tools_config: WebToolsConfig | None,
) -> tuple[ToolsetConfigSnapshot, ...]:
    """把专用工具配置折叠为通用 toolset 配置快照。

    Args:
        doc_tool_limits: 文档工具限制配置。
        fins_tool_limits: 财报工具限制配置。
        web_tools_config: 联网工具配置。

    Returns:
        规范化后的通用 toolset 配置序列。

    Raises:
        TypeError: 当配置对象无法序列化时抛出。
        ValueError: 当 toolset 名称非法时抛出。
    """

    snapshots: tuple[ToolsetConfigSnapshot, ...] = ()
    for snapshot in (
        build_toolset_config_snapshot("doc", doc_tool_limits),
        build_toolset_config_snapshot("fins", fins_tool_limits),
        build_toolset_config_snapshot("web", web_tools_config),
    ):
        if snapshot is None:
            continue
        snapshots = replace_toolset_config(snapshots, snapshot)
    return snapshots


__all__ = [
    "DocToolLimits",
    "FinsToolLimits",
    "WebToolsConfig",
    "build_doc_tool_limits",
    "build_fins_tool_limits",
    "build_legacy_toolset_configs",
    "build_web_tools_config",
    "resolve_doc_tool_limits_from_toolset_configs",
    "resolve_fins_tool_limits_from_toolset_configs",
    "resolve_web_tools_config_from_toolset_configs",
]
