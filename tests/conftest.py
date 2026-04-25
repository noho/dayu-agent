"""
共享 pytest fixtures

提供所有测试模块通用的 Mock 对象和测试数据
"""
import importlib.util
import json
import os
import sys
import tempfile
from typing import Any, cast
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest


def _detect_symlink_capability() -> tuple[bool, str]:
    """探测当前进程是否具备创建文件 symlink 的能力。

    Args:
        无。

    Returns:
        二元组 ``(能否创建 symlink, 说明)``。说明在不可创建时给出原因，便于
        skip 输出可读；可创建时为空串。

    Raises:
        无。
    """

    with tempfile.TemporaryDirectory(prefix="dayu_symlink_probe_") as probe_dir:
        probe_root = Path(probe_dir)
        target = probe_root / "target"
        target.write_text("probe", encoding="utf-8")
        link = probe_root / "link"
        try:
            os.symlink(target, link)
        except OSError as exc:
            return False, f"无 symlink 能力: {type(exc).__name__}: {exc}"
        return True, ""


_SYMLINK_AVAILABLE, _SYMLINK_UNAVAILABLE_REASON = _detect_symlink_capability()

requires_symlink = pytest.mark.skipif(
    not _SYMLINK_AVAILABLE,
    reason=(
        _SYMLINK_UNAVAILABLE_REASON
        or "当前环境无法创建 symlink（如 Windows 普通用户未启用开发者模式或缺少 SeCreateSymbolicLinkPrivilege）"
    ),
)

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def fixtures_dir():
    """返回 tests/fixtures 目录路径"""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def doc_tools_fixtures(fixtures_dir):
    """返回 doc_tools 测试数据目录路径"""
    return fixtures_dir / "doc_tools"


@pytest.fixture
def prompts_fixtures(fixtures_dir):
    """返回 prompts 测试数据目录路径"""
    return fixtures_dir / "prompts"


@pytest.fixture
def registry_fixtures(fixtures_dir):
    """返回 registry 测试数据目录路径"""
    return fixtures_dir / "registry"


@pytest.fixture
def config_fixtures(fixtures_dir):
    """返回 config 测试数据目录路径"""
    return fixtures_dir / "config"

# 预加载 protocols 并补齐 LLMRunner 别名，避免 utils.engine.__init__ 导入失败
protocols_path = Path(__file__).parent.parent / "dayu" / "engine" / "protocols.py"
spec = importlib.util.spec_from_file_location("utils.engine.protocols", protocols_path)
assert spec is not None and spec.loader is not None
protocols = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocols)
if not hasattr(protocols, "LLMRunner"):
    cast(Any, protocols).LLMRunner = protocols.AsyncRunner
sys.modules["utils.engine.protocols"] = protocols


# ============================================================================
# Mock LLM Runner Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_response_no_tools():
    """Mock LLM 响应：无工具调用，直接完成"""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Task completed successfully."
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }


@pytest.fixture
def mock_llm_response_with_tool_call():
    """Mock LLM 响应：包含工具调用"""
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_test_123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({
                            "directory": "test_dir",
                            "filename": "test.txt",
                            "start_line": 1,
                            "end_line": 10
                        })
                    }
                }]
            }
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 30,
            "total_tokens": 50
        }
    }


@pytest.fixture
def mock_runner(mock_llm_response_no_tools):
    """Mock LLM Runner（默认返回无工具调用响应）"""
    runner = Mock(spec=protocols.AsyncRunner)
    runner.call = Mock()
    runner.set_tools = Mock()
    return runner


# ============================================================================
# Mock Tool Registry Fixtures
# ============================================================================

@pytest.fixture
def mock_tool_schemas():
    """Mock 工具 Schema"""
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "读取文件内容",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {"type": "string"},
                        "filename": {"type": "string"},
                        "start_line": {"type": "integer", "default": 1},
                        "end_line": {"type": "integer", "default": -1}
                    },
                    "required": ["directory", "filename"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "获取当前时间",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "default": "Asia/Shanghai"}
                    }
                }
            }
        }
    ]


@pytest.fixture
def mock_tool_executor(mock_tool_schemas):
    """Mock Tool Executor"""
    executor = Mock(spec=protocols.ToolExecutor)
    executor.get_schemas = Mock(return_value=mock_tool_schemas)
    executor.execute = Mock(return_value='{"content": "File content", "total_lines": 100}')
    return executor


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_text_content():
    """示例文本内容"""
    return """Line 1: Introduction
Line 2: This is a test file
Line 3: With multiple lines
Line 4: For testing purposes
Line 5: End of file
"""


@pytest.fixture
def sample_json_content():
    """示例 JSON 内容（content_list.json 格式）"""
    return [
        {
            "page_idx": 1,
            "text_level": 1,
            "type": "title",
            "text": "Chapter 1: Introduction"
        },
        {
            "page_idx": 1,
            "text_level": 2,
            "type": "title",
            "text": "Section 1.1: Overview"
        },
        {
            "page_idx": 2,
            "text_level": 1,
            "type": "title",
            "text": "Chapter 2: Details"
        }
    ]


@pytest.fixture
def sample_html_content():
    """示例 HTML 内容"""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Test Document</title></head>
    <body>
        <h1>Main Title</h1>
        <p>Some content here.</p>
        <h2>Section 1</h2>
        <p>Section content.</p>
        <table>
            <tr><th>Header 1</th><th>Header 2</th></tr>
            <tr><td>Data 1</td><td>Data 2</td></tr>
        </table>
    </body>
    </html>
    """


@pytest.fixture
def sample_markdown_content():
    """示例 Markdown 内容"""
    return """# Main Title

## Section 1

This is the first section.

### Subsection 1.1

Some details here.

## Section 2

Another section.
"""


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture
def temp_test_files(tmp_path, sample_text_content, sample_json_content, 
                     sample_html_content, sample_markdown_content):
    """创建临时测试文件"""
    # 文本文件
    text_file = tmp_path / "test.txt"
    text_file.write_text(sample_text_content, encoding='utf-8')
    
    # JSON 文件
    json_file = tmp_path / "content_list.json"
    json_file.write_text(json.dumps(sample_json_content, ensure_ascii=False), encoding='utf-8')
    
    # HTML 文件
    html_file = tmp_path / "test.html"
    html_file.write_text(sample_html_content, encoding='utf-8')
    
    # Markdown 文件
    md_file = tmp_path / "test.md"
    md_file.write_text(sample_markdown_content, encoding='utf-8')
    
    # GBK 编码文件（测试编码识别）
    gbk_file = tmp_path / "test_gbk.txt"
    gbk_file.write_bytes("中文内容测试".encode('gbk'))
    
    # 大文件（测试行数限制）
    large_file = tmp_path / "large.txt"
    large_file.write_text("\n".join([f"Line {i}" for i in range(6000)]), encoding='utf-8')
    
    return {
        "dir": tmp_path,
        "text_file": text_file,
        "json_file": json_file,
        "html_file": html_file,
        "md_file": md_file,
        "gbk_file": gbk_file,
        "large_file": large_file
    }


# ============================================================================
# HTTP Error Fixtures
# ============================================================================

@pytest.fixture
def mock_http_error():
    """创建 Mock HTTP 错误的工厂函数"""
    def _create_http_error(status_code: int, response_text: str = "Error"):
        import requests
        response = Mock()
        response.status_code = status_code
        response.text = response_text
        error = requests.HTTPError()
        error.response = response
        return error
    return _create_http_error


# ============================================================================
# Agent Message Fixtures
# ============================================================================

@pytest.fixture
def sample_messages():
    """示例消息列表"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Read the file test.txt"}
    ]


@pytest.fixture
def sample_messages_with_tool_result():
    """包含工具结果的消息列表"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Read the file test.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_123",
                "function": {"name": "read_file", "arguments": '{"directory":"test","filename":"test.txt"}'}
            }]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": '{"content": "File content", "total_lines": 10}'
        }
    ]
