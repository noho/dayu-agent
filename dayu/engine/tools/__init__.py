"""
Tools - 内置工具集与工具开发工具

提供：
- 工具装饰器和 schema 构建器
- 文档工具注册函数
- 通用工具注册函数
- 工具类型定义

推荐从顶层导入：
    from dayu.engine import register_doc_tools, tool
    
也支持从 tools 导入：
    from dayu.engine.tools import register_doc_tools, tool
    
注意：为避免循环导入，工具注册函数不在此导入，
请直接从对应模块导入：
    from dayu.engine.tools.doc_tools import register_doc_tools
    from dayu.engine.tools.utils_tools import register_utils_builtin_tools
"""

from ..tool_contracts import ToolSchema, ToolFunctionSchema, ToolTruncateSpec
from .base import tool, build_tool_schema
from .web_tools import register_web_tools
from dayu.contracts.tool_configs import WebToolsConfig

# 不在这里导入 doc_tools 和 utils_tools，避免循环依赖
# 它们依赖 ToolRegistry，而 ToolRegistry 又导入 tools.__init__
#
# web_recovery 内部常量（RECOVERY_CONTRACT_VERSION / ALLOWED_NEXT_ACTIONS / ... 等）
# 是 web 工具内部的重试策略实现细节，不暴露为 dayu.engine.tools 顶层稳定契约；
# 需要这些常量的调用方请直接 `from dayu.engine.tools.web_recovery import ...`。

__all__ = [
    # 类型定义
    "ToolSchema",
    "ToolFunctionSchema",
    "ToolTruncateSpec",

    # 工具装饰器
    "tool",
    "build_tool_schema",
    "register_web_tools",
    "WebToolsConfig",
]
