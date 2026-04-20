"""toolset registrar 数值收口测试。"""

from __future__ import annotations

import pytest

from dayu.contracts import tool_configs
from dayu.contracts.toolset_config import ToolsetConfigSnapshot


@pytest.mark.unit
def test_build_doc_tool_limits_accepts_string_numbers() -> None:
    """验证 doc toolset 快照中的字符串数字会被收敛为整数。"""

    limits = tool_configs.build_doc_tool_limits(
        ToolsetConfigSnapshot(
            toolset_name="doc",
            payload={
                "list_files_max": "12",
                "get_sections_max": "34",
                "search_files_max_results": "56",
                "read_file_max_chars": "78",
                "read_file_section_max_chars": "90",
            },
        )
    )

    assert limits.list_files_max == 12
    assert limits.get_sections_max == 34
    assert limits.search_files_max_results == 56
    assert limits.read_file_max_chars == 78
    assert limits.read_file_section_max_chars == 90


@pytest.mark.unit
def test_build_web_tool_limits_accepts_string_numbers() -> None:
    """验证 web toolset 快照中的字符串数字会被收敛为数值。"""

    config = tool_configs.build_web_tools_config(
        ToolsetConfigSnapshot(
            toolset_name="web",
            payload={
                "provider": "duckduckgo",
                "request_timeout_seconds": "9.5",
                "max_search_results": "18",
                "fetch_truncate_chars": "2048",
            },
        )
    )

    assert config.provider == "duckduckgo"
    assert config.request_timeout_seconds == pytest.approx(9.5)
    assert config.max_search_results == 18
    assert config.fetch_truncate_chars == 2048


@pytest.mark.unit
def test_build_fins_tool_limits_accepts_string_numbers() -> None:
    """验证 fins toolset 快照中的字符串数字会被收敛为整数。"""

    limits = tool_configs.build_fins_tool_limits(
        ToolsetConfigSnapshot(
            toolset_name="fins",
            payload={
                "processor_cache_max_entries": "11",
                "list_documents_max_items": "22",
                "get_document_sections_max_items": "33",
                "search_document_max_items": "44",
                "list_tables_max_items": "55",
                "read_section_max_chars": "66",
                "get_page_content_max_chars": "77",
                "get_table_max_items": "88",
                "get_financial_statement_max_items": "99",
                "query_xbrl_facts_max_items": "111",
            },
        )
    )

    assert limits.processor_cache_max_entries == 11
    assert limits.list_documents_max_items == 22
    assert limits.get_document_sections_max_items == 33
    assert limits.search_document_max_items == 44
    assert limits.list_tables_max_items == 55
    assert limits.read_section_max_chars == 66
    assert limits.get_page_content_max_chars == 77
    assert limits.get_table_max_items == 88
    assert limits.get_financial_statement_max_items == 99
    assert limits.query_xbrl_facts_max_items == 111
