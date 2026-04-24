"""
通用工具模块 - 基础工具集合

提供与业务无关的通用辅助工具，便于快速接入：
- 时间获取（含时区与格式化）
- 当前日期也已通过 system prompt 的 {{current_date}} 占位符注入

主要入口:
1. create_get_current_time_tool()
2. register_utils_builtin_tools(registry)
"""
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Dict, Any, Optional, Tuple
from dayu.log import Log
from .base import tool

MODULE = "ENGINE.UTILS_BUILTIN_TOOLS"


def create_get_current_time_tool(registry=None) -> Tuple[str, Any, Any]:
    """
    创建 get_current_time 工具。

    Args:
        registry: ToolRegistry 实例（可选）

    Returns:
        (name, func, schema): 工具名称、函数与 schema
    """
    parameters = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "IANA timezone name",
                "enum": ["Asia/Shanghai"],
                "default": "Asia/Shanghai",
            }
        },
        "required": [],
    }

    @tool(
        registry,
        name="get_current_time",
        description=(
            "Get current date and time. "
            "Returns: time (formatted), timezone, weekday, iso (ISO 8601)."
        ),
        parameters=parameters,
        tags={"utils"},
        display_name="获取时间",
    )
    def get_current_time(timezone: str = "Asia/Shanghai") -> Dict[str, Any]:
        """
        获取当前时间
        
        Args:
            timezone: 时区名称（当前仅支持 Asia/Shanghai）
            
        Returns:
            Dict 包含:
                - time: 格式化时间字符串
                - timezone: 时区
                - weekday: 星期几
                - iso: ISO 8601 格式时间
        """
        if timezone != "Asia/Shanghai":
            raise ValueError(f"不支持的时区: {timezone}，当前仅支持 Asia/Shanghai")
        try:
            tzinfo = ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"无法加载时区: {timezone}") from exc
        now = datetime.now(tzinfo)
        
        weekday_names = {
            0: "星期一", 1: "星期二", 2: "星期三", 3: "星期四",
            4: "星期五", 5: "星期六", 6: "星期日"
        }
        
        return {
            "time": f"{now:%Y}年{now:%m}月{now:%d}日 {now:%H:%M:%S}",
            "timezone": timezone,
            "weekday": weekday_names[now.weekday()],
            "iso": now.isoformat()
        }

    return (
        get_current_time.__tool_name__,
        get_current_time,
        get_current_time.__tool_schema__,
    )


def register_utils_builtin_tools(
    registry,
    timeout_budget: Optional[float] = None,
):
    """
    将通用工具注册到 ToolRegistry。

    注意：当前日期已通过 system prompt 的 {{current_date}} 占位符预先注入，
    get_current_time 工具仅在尚需实时时间的场景下使用。

    Args:
        registry: ToolRegistry 实例。
        timeout_budget: Runner 为单次 tool call 提供的预算秒数；当前 utils 工具预留该参数，
            暂未消费。

    Returns:
        注册的工具数量。

    Raises:
        无。
    """
    del timeout_budget
    name, func, schema = create_get_current_time_tool(registry)
    registry.register(name, func, schema)
    Log.verbose("已注册 1 个通用工具", module=MODULE)
    return 1
