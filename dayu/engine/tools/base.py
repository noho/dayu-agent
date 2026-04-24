"""工具 decorator 与 schema 构建辅助。

该模块负责：
- 生成 ``ToolSchema``。
- 将运行时元数据（tags / truncate / dup_call / file_path_params）挂到函数对象上。
- 供 ``ToolRegistry.register()`` 在注册时统一读取。
"""

import copy
from dataclasses import dataclass
from typing import AbstractSet, Any, Callable, Dict, Optional, ParamSpec, Protocol, TypeVar, Union, cast

from ..tool_contracts import DupCallSpec, ToolFunctionSchema, ToolSchema, ToolTruncateSpec
from ..exceptions import ConfigError

P = ParamSpec("P")
R = TypeVar("R", covariant=True)


def _resolve_enum_values(
    enum_spec: Any,
    registry: Any,
) -> Optional[list]:
    """解析参数的枚举值。

    Args:
        enum_spec: 枚举值列表，或接收 registry 返回列表的可调用对象。
        registry: ``ToolRegistry`` 实例；当 ``enum_spec`` 为可调用对象时必填。

    Returns:
        解析后的枚举值列表；若 ``enum_spec`` 或解析结果为 ``None`` 则返回 ``None``。

    Raises:
        ConfigError: 可调用 ``enum_spec`` 未传入 ``registry``，或解析结果非列表。
    """
    if enum_spec is None:
        return None
    if callable(enum_spec):
        if registry is None:
            raise ConfigError("tool_schema", None, "enum resolver requires registry")
        enum_values = enum_spec(registry)
    else:
        enum_values = enum_spec

    if enum_values is None:
        return None
    if not isinstance(enum_values, list):
        raise ConfigError("tool_schema", None, "enum values must be a list")
    return enum_values


def build_tool_schema(
    *,
    name: str,
    description: str,
    parameters: Dict[str, Any],
    enums: Optional[Dict[str, Any]] = None,
    registry: Any = None,
) -> ToolSchema:
    """构建带可选枚举注入的 ``ToolSchema``。

    Args:
        name: 工具名称。
        description: 供 LLM 使用的工具描述。
        parameters: 参数的 JSON Schema dict。
        enums: 可选映射，字段名 -> 枚举值列表或 ``callable(registry) -> list``。
        registry: 用于解析动态枚举的 ``ToolRegistry`` 实例。

    Returns:
        构建好的 ``ToolSchema`` 实例。

    Raises:
        ConfigError: 参数结构非法，或枚举字段不存在于 parameters 中。
    """
    if not isinstance(parameters, dict):
        raise ConfigError("tool_schema", None, "parameters must be a dict")

    params_copy = copy.deepcopy(parameters)
    properties = params_copy.get("properties")
    if not isinstance(properties, dict):
        raise ConfigError("tool_schema", None, "parameters.properties must be a dict")

    if enums:
        for field_name, enum_spec in enums.items():
            if field_name not in properties:
                raise ConfigError("tool_schema", None, f"enum field not found in parameters: {field_name}")
            enum_values = _resolve_enum_values(enum_spec, registry)
            if enum_values:
                properties[field_name]["enum"] = enum_values
            else:
                properties[field_name].pop("enum", None)

    return ToolSchema(
        function=ToolFunctionSchema(
            name=name,
            description=description,
            parameters=params_copy,
        )
    )


@dataclass
class ToolExtra:
    """附加工具元数据（不参与 OpenAI schema）。"""

    __file_path_params__: list[str]
    __truncate__: ToolTruncateSpec
    __dup_call__: Optional[DupCallSpec]
    __execution_context_param_name__: str | None
    __display_name__: str | None = None
    __summary_params__: list[str] | None = None


class DecoratedToolCallable(Protocol[P, R]):
    """带工具元数据的可调用对象协议。"""

    __tool_name__: str
    __tool_schema__: ToolSchema
    __tool_tags__: set[str]
    __tool_extra__: ToolExtra

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """执行工具函数。"""

        ...


def tool(
    registry: Any,
    *,
    name: str,
    description: str,
    parameters: Union[Dict[str, Any], Callable[[Any], Dict[str, Any]]],
    enums: Optional[Dict[str, Any]] = None,
    tags: Optional[AbstractSet[str]] = None,
    truncate: Optional[Union[ToolTruncateSpec, Dict[str, Any]]] = None,
    dup_call: Optional[Union[DupCallSpec, Dict[str, Any]]] = None,
    file_path_params: Optional[list[str]] = None,
    execution_context_param_name: str | None = None,
    display_name: str | None = None,
    summary_params: list[str] | None = None,
) -> Callable[[Callable[P, R]], DecoratedToolCallable[P, R]]:
    """工具函数装饰器。

    该装饰器解析参数、注入枚举、构建 ``ToolSchema``，并把元数据挂到
    函数对象上，供 ``ToolRegistry.register()`` 在注册时统一读取。

    Args:
        registry: ``ToolRegistry`` 实例。
        name: 工具名称。
        description: 供 LLM 使用的工具描述。
        parameters: 参数的 JSON Schema dict，或返回该 dict 的可调用对象。
        enums: 可选字段名 -> 枚举值映射。
        tags: 可选的工具分组标签集合。
        truncate: 可选截断规格。
        dup_call: 可选重复调用规格。
        file_path_params: 可选的需要按文件路径校验的参数名列表
            （如 ``["file_path", "directory"]``）。
        execution_context_param_name: 工具函数中接收 execution context 的显式参数名；
            为 ``None`` 表示该工具不接收 execution context。
        display_name: 面向用户展示的工具名（中文），``None`` 时 fallback 到 ``name``。
        summary_params: 调用时摘要展示的参数名列表，``None`` 时只展示工具名。

    Returns:
        装饰器函数，它会返回挂有工具元数据的可调用对象。

    Raises:
        ConfigError: ``truncate`` / ``dup_call`` 参数类型非法。
    """

    def wrap(func: Callable[P, R]) -> DecoratedToolCallable[P, R]:
        resolved_parameters = parameters(registry) if callable(parameters) else parameters
        schema = build_tool_schema(
            name=name,
            description=description,
            parameters=resolved_parameters,
            enums=enums,
            registry=registry
        )
        if truncate is None:
            truncate_spec = ToolTruncateSpec()
        elif isinstance(truncate, ToolTruncateSpec):
            truncate_spec = truncate
        elif isinstance(truncate, dict):
            truncate_spec = ToolTruncateSpec(**truncate)
        else:
            raise ConfigError("tool_schema", None, "truncate must be ToolTruncateSpec or dict")

        if dup_call is None:
            dup_call_spec = None
        elif isinstance(dup_call, DupCallSpec):
            dup_call_spec = dup_call
        elif isinstance(dup_call, dict):
            dup_call_spec = DupCallSpec(**dup_call)
        else:
            raise ConfigError("tool_schema", None, "dup_call must be DupCallSpec or dict")

        decorated_func = cast(DecoratedToolCallable[P, R], func)
        decorated_func.__tool_name__ = name
        decorated_func.__tool_schema__ = schema
        decorated_func.__tool_tags__ = set(tags) if tags is not None else set()
        decorated_func.__tool_extra__ = ToolExtra(
            __file_path_params__=file_path_params or [],
            __truncate__=truncate_spec,
            __dup_call__=dup_call_spec,
            __execution_context_param_name__=(
                str(execution_context_param_name).strip()
                if execution_context_param_name is not None and str(execution_context_param_name).strip()
                else None
            ),
            __display_name__=display_name,
            __summary_params__=summary_params,
        )
        return decorated_func

    return wrap
