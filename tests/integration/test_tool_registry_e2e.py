"""ToolRegistry 高级功能端到端测试。

测试重点：
1. 路径白名单机制（register_allowed_paths）
2. 工具注册与执行流程
3. fetch_more 分页功能（continuation token）
4. 工具指导模板渲染
5. 路径安全验证
"""

import json
from pathlib import Path

import pytest

from dayu.contracts.tool_configs import DocToolLimits
from dayu.engine.exceptions import FileAccessError
from dayu.engine.tool_registry import ToolRegistry
from dayu.engine.tools.doc_tools import register_doc_tools
from dayu.prompting.prompt_renderer import load_prompt


@pytest.fixture
def tool_registry(registry_fixtures):
    """创建 ToolRegistry 实例并注册测试路径"""
    registry = ToolRegistry()

    # 注册测试目录
    registry.register_allowed_paths([registry_fixtures])

    # 注册 doc_tools
    limits = DocToolLimits(
        list_files_max=200,
        get_sections_max=200,
        search_files_max_results=50,
        read_file_max_chars=200000
    )
    register_doc_tools(registry, limits)

    return registry


def test_tool_registration(tool_registry):
    """测试工具注册"""
    tool_names = tool_registry.list_tools()
    
    # 验证 doc_tools 已注册
    assert "list_files" in tool_names
    assert "get_file_sections" in tool_names
    assert "search_files" in tool_names
    assert "read_file" in tool_names
    assert "fetch_more" in tool_names


def test_get_tool_schemas(tool_registry):
    """测试获取工具 schemas"""
    schemas = tool_registry.get_schemas()
    
    assert len(schemas) >= 5  # 至少 4 个 doc_tools + fetch_more
    
    # 验证 schema 格式
    for schema in schemas:
        assert "type" in schema
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "description" in schema["function"]


def test_list_files_in_registry_fixtures(tool_registry, registry_fixtures):
    """测试列出测试文件"""
    result = tool_registry.execute("list_files", {
        "directory": str(registry_fixtures)
    })
    
    # 验证 README 规范的返回格式
    assert result["ok"] is True
    
    # 提取实际工具返回值
    data = result["value"]
    
    assert "files" in data
    files = data["files"]
    
    # 验证测试文件存在
    file_names = [f["name"] for f in files]
    assert "long_text.txt" in file_names
    assert "large_json.json" in file_names


def test_read_long_text_with_pagination(tool_registry, registry_fixtures):
    """测试读取文件的分页功能（精确边界验证）"""
    # 使用 100 行的小文件进行精确验证
    pagination_test_path = registry_fixtures / "pagination_test.txt"
    
    # 读取第 1-50 行
    result1 = tool_registry.execute("read_file", {
        "file_path": str(pagination_test_path),
        "start_line": 1,
        "end_line": 50
    })
    
    # 验证 README 规范的返回格式
    assert result1["ok"] is True
    assert result1.get("truncation") is None  # 小文件不应截断
    
    data1 = result1["value"]
    assert "content" in data1
    content1 = data1["content"]
    
    # 验证第一页边界
    assert "Line 001:" in content1
    assert "Line 050:" in content1
    assert "Line 051:" not in content1
    
    # 读取第 51-100 行
    result2 = tool_registry.execute("read_file", {
        "file_path": str(pagination_test_path),
        "start_line": 51,
        "end_line": 100
    })
    
    # 验证 README 规范的返回格式
    assert result2["ok"] is True
    assert result2.get("truncation") is None
    
    data2 = result2["value"]
    assert "content" in data2
    content2 = data2["content"]
    
    # 验证第二页边界
    assert "Line 050:" not in content2
    assert "Line 051:" in content2
    assert "Line 100:" in content2
    
    # 验证两页内容不重叠
    assert "Line 051:" not in content1
    assert "Line 050:" not in content2


def test_read_json_file(tool_registry, registry_fixtures):
    """测试读取 JSON 文件"""
    json_path = registry_fixtures / "large_json.json"
    
    result = tool_registry.execute("read_file", {
        "file_path": str(json_path)
    })
    
    # 验证 README 规范的返回格式
    assert result["ok"] is True
    assert result.get("truncation") is None  # JSON 文件不大，不应截断
    
    # 提取实际工具返回值
    data_value = result["value"]
    
    assert "content" in data_value
    content_str = data_value["content"]
    
    # 解析 JSON 字符串
    parsed_json = json.loads(content_str)
    
    # 验证基本结构
    assert "level1" in parsed_json
    assert "users" in parsed_json
    assert "config" in parsed_json
    assert "large_array" in parsed_json
    
    # 验证嵌套结构
    assert "level2" in parsed_json["level1"]
    assert "app" in parsed_json["config"]
    
    # 验证 users 数组内容
    assert len(parsed_json["users"]) > 0
    assert "name" in parsed_json["users"][0]
    assert "profile" in parsed_json["users"][0]
    assert "email" in parsed_json["users"][0]["profile"]


def test_search_files_in_nested_dirs(tool_registry, registry_fixtures):
    """测试在嵌套目录中搜索文件"""
    nested_dir = registry_fixtures / "nested_dirs"
    
    result = tool_registry.execute("search_files", {
        "directory": str(nested_dir),
        "query": "level 1"
    })
    
    # 验证 README 规范的返回格式
    assert result["ok"] is True
    
    # 提取实际工具返回值
    data = result["value"]
    
    assert "matches" in data
    matches = data["matches"]
    
    # 应该能找到包含 "level 1" 的文件
    assert len(matches) > 0


def test_get_file_sections_for_nested_file(tool_registry, registry_fixtures):
    """测试获取嵌套目录文件的分节信息"""
    file_path = registry_fixtures / "nested_dirs" / "level1" / "file_level1.txt"
    
    result = tool_registry.execute("get_file_sections", {
        "file_path": str(file_path)
    })
    
    # 验证 README 规范的返回格式
    assert result["ok"] is True
    
    # 提取实际工具返回值
    data = result["value"]
    
    # 对于简单文本文件，应该返回基本信息
    assert "total_lines" in data or "sections" in data


def test_path_security_unauthorized_access(tool_registry):
    """测试路径安全：访问未授权路径应该失败"""
    # 尝试访问系统文件
    result = tool_registry.execute("read_file", {
        "file_path": "/etc/passwd"
    })
    
    # 应该返回 success=False 或包含错误信息
    assert result["ok"] is False or "error" in result


def test_tool_guidance_generation(tool_registry, prompts_fixtures):
    """测试工具指导模板渲染。"""
    guidance_file = prompts_fixtures / "tool_guidance_multi_tool.md"
    template = guidance_file.read_text(encoding="utf-8")

    guidance = load_prompt(
        template,
        tool_names=tool_registry.get_tool_names(),
        tag_names=tool_registry.get_tool_tags(),
    )
    
    # 验证指导包含已注册工具的说明
    assert "list_files" in guidance
    assert "get_file_sections" in guidance
    assert "read_file" in guidance
    assert "search_files" in guidance


def test_tool_execution_error_handling(tool_registry):
    """测试工具执行错误处理"""
    # 执行不存在的工具应该返回 success=False
    try:
        result = tool_registry.execute("non_existent_tool", {})
        # 如果没有抛出异常，应该返回 success=False
        assert result["ok"] is False
    except KeyError:
        # 如果抛出 KeyError 也是可接受的
        pass


def test_full_workflow_list_search_read(tool_registry, registry_fixtures):
    """测试完整工作流：列出 -> 搜索 -> 读取"""
    # 1. 列出文件
    list_result = tool_registry.execute("list_files", {
        "directory": str(registry_fixtures / "nested_dirs")
    })
    
    # 验证 README 规范
    assert list_result["ok"] is True
    
    list_data = list_result["value"]
    assert "files" in list_data
    assert len(list_data["files"]) > 0
    
    # 2. 搜索文件
    search_result = tool_registry.execute("search_files", {
        "directory": str(registry_fixtures / "nested_dirs"),
        "query": "allowed"
    })
    
    # 验证 README 规范
    assert search_result["ok"] is True
    
    search_data = search_result["value"]
    assert "matches" in search_data
    
    # 3. 读取找到的文件
    if len(search_data["matches"]) > 0:
        first_match = search_data["matches"][0]
        # 构建完整文件路径
        file_path = registry_fixtures / "nested_dirs" / first_match["file"]
        
        read_result = tool_registry.execute("read_file", {
            "file_path": str(file_path)
        })
        
        # 验证 README 规范
        assert read_result["ok"] is True
        
        read_data = read_result["value"]
        assert "content" in read_data
        assert "allowed" in read_data["content"].lower()


def test_continuation_token_management(tool_registry, registry_fixtures):
    """测试 continuation token 管理"""
    long_text_path = registry_fixtures / "long_text.txt"
    
    # 读取第一页
    result = tool_registry.execute("read_file", {
        "file_path": str(long_text_path),
        "start_line": 1,
        "end_line": 5000
    })
    
    if result.get("has_more"):
        token = result["continuation_token"]
        
        # 验证 token 是有效的
        assert isinstance(token, str)
        assert len(token) > 0
        
        # 使用 token 获取下一页
        scope_token = result["truncation"]["fetch_more_args"]["scope_token"]
        next_result = tool_registry.execute("fetch_more", {
            "cursor": token,
            "scope_token": scope_token,
        })
        
        assert "content" in next_result


def test_allowed_paths_registration(tool_registry, registry_fixtures):
    """测试路径白名单注册"""
    allowed_paths = tool_registry.get_allowed_paths()
    
    # 验证测试目录在白名单中
    allowed_path_strs = [str(p) for p in allowed_paths]
    registry_fixtures_str = str(registry_fixtures.resolve())
    
    # 检查是否包含注册的路径
    assert any(registry_fixtures_str in p for p in allowed_path_strs)


def test_truncation_boundary_over_limit(tool_registry, registry_fixtures):
    """测试超过 max_chars 限制时的截断行为（README 规范验证）"""
    # 使用 long_text.txt (15000 行)，超过 200000 字符限制
    long_text_path = registry_fixtures / "long_text.txt"
    
    result = tool_registry.execute("read_file", {
        "file_path": str(long_text_path)
    })
    
    # 验证 README 规范：超过限制时应截断
    assert result["ok"] is True
    assert result.get("truncation") is not None  # 应该被截断
    
    # 验证 truncation 字段存在（README L424-426）
    truncation = result["truncation"]
    
    # 验证 truncation 结构（README 明确定义）
    assert "reason" in truncation
    assert truncation["reason"] in ["max_chars", "max_bytes", "max_items", "max_depth", "max_time"]
    assert "limit" in truncation
    assert "unit" in truncation
    assert "cursor" in truncation  # opaque cursor
    assert "has_more" in truncation
    assert truncation["has_more"] is True  # 应该有更多内容
    
    # 验证 cursor 可用于 fetch_more
    cursor = truncation["cursor"]
    assert isinstance(cursor, str)
    assert len(cursor) > 0
    
    # 验证内容确实被截断
    content = result["value"]["content"]
    assert len(content) <= 200000  # 不应超过限制


def test_truncation_fetch_more_continuation(tool_registry, registry_fixtures):
    """测试 fetch_more 续读截断内容（README 规范验证）"""
    # 先读取一个大文件触发截断
    long_text_path = registry_fixtures / "long_text.txt"
    
    result = tool_registry.execute("read_file", {
        "file_path": str(long_text_path)
    })
    
    # 如果被截断，使用 fetch_more 续读
    if result.get("truncation") is not None:
        cursor = result["truncation"]["cursor"]
        
        # 使用 fetch_more 工具
        scope_token = result["truncation"]["fetch_more_args"]["scope_token"]
        next_result = tool_registry.execute("fetch_more", {
            "cursor": cursor,
            "scope_token": scope_token,
        })
        
        # 验证 README 规范：fetch_more 返回格式
        assert next_result["ok"] is True
            
        # 验证返回了后续内容
        next_content = next_result["value"]
        assert isinstance(next_content, (str, dict))


def test_data_type_field_consistency(tool_registry, registry_fixtures):
    """测试 data.type 字段一致性（README L250-263）"""
    # read_file 返回 dict，被包装为 json 类型
    text_result = tool_registry.execute("read_file", {
        "file_path": str(registry_fixtures / "pagination_test.txt")
    })
    
    assert text_result["ok"] is True
    assert isinstance(text_result["value"], dict)  # 工具返回结构化数据
    
    # list_files 应返回结构化类型（text 或其他）
    list_result = tool_registry.execute("list_files", {
        "directory": str(registry_fixtures)
    })
    
    assert list_result["ok"] is True
    assert isinstance(list_result["value"], dict)  # list_files 返回结构化数据


def test_error_return_format(tool_registry):
    """测试错误返回格式（README L266-267）"""
    # 尝试读取不存在的文件
    result = tool_registry.execute("read_file", {
        "file_path": "/nonexistent/path/to/file.txt"
    })
    
    # 验证 README 规范的错误格式
    assert result["ok"] is False
    assert "error" in result
    assert "message" in result
    assert isinstance(result["error"], str)
    assert isinstance(result["message"], str)
