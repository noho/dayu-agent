"""工具注册表模块。

该模块负责：
- 工具注册与 schema 管理
- 路径白名单与安全校验
- 统一的工具执行入口与错误封装
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from dayu.contracts.protocols import ToolExecutionContext

from .argument_validator import ArgumentValidator
from .exceptions import ConfigError, FileAccessError
from dayu.log import Log
from .tool_contracts import DupCallSpec, ToolFunctionSchema, ToolSchema, ToolTruncateSpec
from .tool_errors import ToolBusinessError
from .tool_result import build_error, build_success, is_tool_success
from .truncation_manager import TruncationManager
from dayu.contracts.cancellation import CancelledError

MODULE = "ENGINE.TOOL_REGISTRY"


@dataclass
class ToolDescriptor:
    """已注册工具的元信息。"""

    name: str
    tags: set = field(default_factory=set)
    dup_call: Optional[DupCallSpec] = None
    execution_context_param_name: str | None = None


def _format_tool_business_error_log(name: str, error: ToolBusinessError) -> str:
    """格式化工具业务错误日志。

    Args:
        name: 工具名称。
        error: 工具业务异常。

    Returns:
        适合直接写入日志的单行文案。

    Raises:
        无。
    """

    url = str(error.extra.get("url", "") or "").strip()
    prefix = f"工具 {name} 业务错误: {error.code}"
    if url:
        prefix = f"{prefix} url={url}"
    return f"{prefix} - {error.message}"


class ToolRegistry:
    """工具注册表 - 管理和执行 Tool Calling 工具

    内部组合了 ArgumentValidator（参数校验）和 TruncationManager（截断分页）。
    """

    def __init__(self):
        """
        初始化工具注册表

        Example:
            registry = ToolRegistry()
            registry.register_allowed_paths([
                Path("workspace/config"),
                Path("workspace/prompts"),
                Path("output/logs/app.log")
            ])
        """
        self.tools: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.tool_schemas: Dict[str, ToolSchema] = {}
        self.tool_descriptors: Dict[str, ToolDescriptor] = {}
        self.allowed_paths: set[Path] = set()
        self._argument_validator = ArgumentValidator()
        self._truncation_manager = TruncationManager()
        self._response_middlewares: list[Callable[[str, Dict[str, Any], ToolExecutionContext | None], Dict[str, Any]]] = []

        Log.debug(
            f"工具注册表初始化，允许路径数: {len(self.allowed_paths)}",
            module=MODULE,
        )

    def clear_cursors(self) -> None:
        """清除所有截断游标，释放关联的数据引用。

        在新 run 开始前由 AsyncAgent 调用，避免上一轮残留游标占用内存。
        """
        self._truncation_manager.clear_cursors()

    def register_response_middleware(
        self,
        callback: Callable[[str, Dict[str, Any], ToolExecutionContext | None], Dict[str, Any]],
    ) -> None:
        """注册 response middleware，在工具执行成功后链式调用。

        Args:
            callback: ``(tool_name, result, context) -> result`` 签名的回调函数。
                ``context`` 含 ``run_id``、``iteration_id``、``tool_call_id``、
                ``index_in_iteration``
                等字段，middleware 可据此实现按轮次感知逻辑。
                多个 middleware 按注册顺序依次执行。

        Returns:
            无。
        """

        self._response_middlewares.append(callback)

    def register(
        self,
        name: str,
        func: Callable,
        schema: Any,
    ) -> None:
        """
        注册一个工具。

        Args:
            name: 工具名称（需与 schema 中的 function.name 一致）
            func: 工具函数
            schema: ToolSchema 或 OpenAI 工具定义 schema dict
        """
        if name != "fetch_more" and "fetch_more" not in self.tools:
            # 首个真实工具注册前再挂载框架级续读工具，避免空 registry 暴露 fetch_more。
            Log.verbose("自动挂载 fetch_more 续读工具", module=MODULE)
            self.register_fetch_more_tool()

        tool_schema = self._coerce_tool_schema(name, schema)
        openai_schema = tool_schema.to_openai()

        # 验证 schema 结构与名称
        self._validate_tool_schema(name, openai_schema)

        if name in self.tools:
            Log.warn(f"工具 '{name}' 已存在，将被覆盖", module=MODULE)

        self.tools[name] = func
        self.schemas[name] = openai_schema
        self.tool_schemas[name] = tool_schema
        raw_tags = getattr(func, "__tool_tags__", set())
        raw_extra = getattr(func, "__tool_extra__", None)
        self.tool_descriptors[name] = ToolDescriptor(
            name=name,
            tags=set(raw_tags) if raw_tags else set(),
            dup_call=getattr(raw_extra, "__dup_call__", None),
            execution_context_param_name=getattr(raw_extra, "__execution_context_param_name__", None),
        )

        Log.debug(f"注册工具: {name}", module=MODULE)

    def _coerce_tool_schema(self, name: str, schema: Any) -> ToolSchema:
        """
        Normalize schema input into ToolSchema.

        Args:
            name: Tool name for validation context.
            schema: ToolSchema or OpenAI schema dict.

        Returns:
            ToolSchema instance.
        """
        if isinstance(schema, ToolSchema):
            return schema
        if not isinstance(schema, dict):
            raise ConfigError("tool_schema", None, "schema 必须是 dict 或 ToolSchema")

        function = schema.get("function", {})
        return ToolSchema(
            function=ToolFunctionSchema(
                name=function.get("name", name),
                description=function.get("description", ""),
                parameters=function.get("parameters", {}),
            )
        )

    def _validate_tool_schema(self, name: str, schema: Dict[str, Any]) -> None:
        """
        校验工具 schema 是否符合最佳实践。
        """
        if not isinstance(schema, dict):
            raise ConfigError("tool_schema", None, "schema 必须是 dict")

        if schema.get("type") != "function":
            raise ConfigError("tool_schema", None, "schema.type 必须为 'function'")

        function = schema.get("function")
        if not isinstance(function, dict):
            raise ConfigError("tool_schema", None, "schema.function 必须是对象")

        schema_name = function.get("name")
        if not isinstance(schema_name, str) or not schema_name:
            raise ConfigError("tool_schema", None, "schema.function.name 必须是非空字符串")
        if schema_name != name:
            raise ConfigError(
                "tool_registration",
                None,
                f"工具名称不匹配: register(name='{name}') "
                f"但 schema.function.name='{schema_name}'"
            )

        description = function.get("description")
        if description is not None and not isinstance(description, str):
            raise ConfigError("tool_schema", None, "schema.function.description 必须是字符串")

        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            raise ConfigError("tool_schema", None, "schema.function.parameters 必须是对象")
        if parameters.get("type") != "object":
            raise ConfigError("tool_schema", None, "schema.function.parameters.type 必须是 'object'")

        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            raise ConfigError("tool_schema", None, "schema.function.parameters.properties 必须是对象")

        required = parameters.get("required")
        if required is not None:
            if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
                raise ConfigError("tool_schema", None, "schema.function.parameters.required 必须是字符串数组")

        additional_props = parameters.get("additionalProperties")
        if additional_props is not None and not isinstance(additional_props, bool):
            raise ConfigError(
                "tool_schema",
                None,
                "schema.function.parameters.additionalProperties 必须是布尔值",
            )
    
    def get_schemas(self) -> list:
        """获取所有工具的 schema 列表（用于传递给 LLM）"""
        return list(self.schemas.values())
    
    def list_tools(self) -> List[str]:
        """获取所有已注册工具的名称列表"""
        return list(self.tools.keys())
    
    def get_allowed_paths(self) -> List[str]:
        """获取允许访问的路径列表（规范化绝对路径）"""
        return sorted(str(p.resolve()) for p in self.allowed_paths)

    def register_fetch_more_tool(self) -> None:
        """
        注册统一续读工具 fetch_more(cursor, scope_token, limit)。

        该工具由 ToolRegistry.execute 处理，直接调用会抛出异常。
        """
        if "fetch_more" in self.tools:
            return
        fetch_more_schema = {
            "type": "function",
            "function": {
                "name": "fetch_more",
                "description": "继续读取上一条已截断的工具结果。只有当最新返回里的 truncation.next_action=\"fetch_more\" 时才调用；否则不要调用。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cursor": {
                            "type": "string",
                            "description": "单次有效的续读游标。直接使用最新一条截断结果里的 truncation.fetch_more_args.cursor；成功续读后会返回下一页的新 cursor，不要复用更早返回里的旧 cursor。",
                        },
                        "scope_token": {
                            "type": "string",
                            "description": "范围校验令牌。直接使用与当前 cursor 同一条返回里的 truncation.fetch_more_args.scope_token。",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "可选。本次续读最多返回多少项；不得超过上一轮工具原本允许的上限。",
                            "minimum": 1,
                        },
                    },
                    "required": ["cursor", "scope_token"],
                },
            },
        }

        self.register("fetch_more", self._fetch_more_placeholder, fetch_more_schema)

    def _fetch_more_placeholder(self, cursor: str, scope_token: str, limit: Optional[int] = None) -> None:
        """
        fetch_more 占位函数，避免被直接调用。
        """
        raise RuntimeError("fetch_more should be handled by ToolRegistry.execute")
    
    def register_allowed_paths(self, paths: List[Path]) -> None:
        """
        注册可访问的文件或目录（路径白名单安全机制）

        动态注册允许访问的文件和目录。工具执行时（通过 file_path_params 声明的
        路径参数）会自动校验是否在已注册的白名单内。

        Args:
            paths: 文件或目录的路径列表（支持 Path 对象或字符串）

        Raises:
            ConfigError: 路径不存在时抛出

        Example:
            >>> registry = ToolRegistry()
            >>> registry.register_allowed_paths([
            ...     Path("workspace/config"),          # 目录
            ...     Path("workspace/prompts"),         # 目录
            ...     Path("output/logs/app.log"),       # 单个文件
            ...     Path("manual/GUIDE.md")            # 单个文件
            ... ])
        """
        for path in paths:
            path_obj = Path(path).resolve()
            
            # 验证路径存在
            if not path_obj.exists():
                raise ConfigError(
                    "register_allowed_paths",
                    None,
                    f"路径不存在: {path} (resolved to {path_obj})"
                )
            
            # 添加到白名单
            self.allowed_paths.add(path_obj)
            
            Log.debug(f"注册路径: {path_obj}", module=MODULE)
        
        Log.debug(f"已注册 {len(paths)} 个路径，当前允许路径总数: {len(self.allowed_paths)}", module=MODULE)
    
    def _validate_path(self, path: str) -> Path:
        """
        验证路径是否在允许范围内（新的安全机制）
        
        安全检查机制：
        1. Symlink 逃逸防护：resolve() 解析所有符号链接后再检查
        2. 路径遍历防护：禁止 ../ 等相对路径逃逸
        3. 绝对路径规范化：统一转为绝对路径比较
        4. 存在性检查：确保路径存在
        5. 白名单验证：必须在 allowed_paths 范围内
        
        Args:
            path: 待验证的文件路径（可以是相对路径或绝对路径）
            
        Returns:
            Path: 验证通过的规范化绝对路径
            
        Raises:
            PermissionError: 路径不在允许范围内
            FileNotFoundError: 路径不存在
            
        Example:
            >>> registry = ToolRegistry()
            >>> registry.register_allowed_paths([Path("workspace/config")])
            >>> validated = registry._validate_path("workspace/config/llm_models.json")
            >>> # ✅ 允许：在 workspace/config 下
            >>>
            >>> validated = registry._validate_path("/etc/passwd")
            >>> # ❌ 拒绝：PermissionError
        """
        # 1. 转为绝对路径并解析所有符号链接
        try:
            resolved_path = Path(path).resolve(strict=False)
        except (OSError, RuntimeError) as e:
            raise PermissionError(f"无效路径: {path} ({e})")
        
        # 2. 检查路径是否存在
        if not resolved_path.exists():
            raise FileNotFoundError(f"路径不存在: {path} (resolved to {resolved_path})")
        
        # 3. 检查是否在允许的路径范围内
        for allowed_path in self.allowed_paths:
            # 同样解析 allowed_path 的符号链接
            resolved_allowed = allowed_path.resolve()
            
            try:
                # 检查 resolved_path 是否在 resolved_allowed 下
                # 如果是文件，检查文件本身是否匹配
                if resolved_allowed.is_file():
                    if resolved_path == resolved_allowed:
                        return resolved_path  # 精确匹配单个文件
                else:
                    # 如果是目录，检查是否在目录下
                    resolved_path.relative_to(resolved_allowed)
                    return resolved_path  # 在允许的目录范围内
            except ValueError:
                continue  # 不在当前 allowed_path 下，继续检查下一个
        
        # 4. 所有允许路径都不匹配
        raise PermissionError(
            f"访问被拒绝: {path} (resolved to {resolved_path}) 不在允许的路径范围内。\n"
            f"已注册路径: {[str(p) for p in self.allowed_paths]}"
        )
    
    def get_tool_names(self) -> Set[str]:
        """获取已注册工具的名称集合（不包含 fetch_more）"""
        names: Set[str] = set()
        for name in self.tool_descriptors.keys():
            if name == "fetch_more":
                continue
            names.add(name)
        return names

    def get_tool_tags(self) -> Set[str]:
        """获取已注册工具的 tag 集合（不包含 fetch_more）"""
        tags: Set[str] = set()
        for name, descriptor in self.tool_descriptors.items():
            if name == "fetch_more":
                continue
            if descriptor.tags:
                tags.update(descriptor.tags)
        return tags

    def get_dup_call_spec(self, name: str) -> Optional[DupCallSpec]:
        """按工具名读取重复调用策略声明。

        Args:
            name: 工具名称。

        Returns:
            ``DupCallSpec``；未声明或工具不存在时返回 ``None``。

        Raises:
            无。
        """

        descriptor = self.tool_descriptors.get(name)
        if descriptor is None:
            return None
        return descriptor.dup_call

    def get_execution_context_param_name(self, name: str) -> str | None:
        """按工具名读取 execution context 注入参数名。

        Args:
            name: 工具名称。

        Returns:
            参数名；未声明或工具不存在时返回 ``None``。

        Raises:
            无。
        """

        descriptor = self.tool_descriptors.get(name)
        if descriptor is None:
            return None
        return descriptor.execution_context_param_name
    

    def execute(
        self,
        name: str,
        arguments: Dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Dict[str, Any]:
        """执行工具并返回结构化结果。

        注意：该方法不抛异常；所有失败都通过返回值中的 ok/error 表达。

        Args:
            name: 工具名称。
            arguments: 工具参数。
            context: 可选的执行上下文。

        Returns:
            统一信封格式的字典：
            成功: ``{"ok": True, "value": <any>, "truncation": {...}|None, "meta": {...}|None}``
            失败: ``{"ok": False, "error": "<code>", "message": "..."}``
        """
        # Log.debug(f"执行工具: {name}, 参数: {arguments}", module=MODULE)
        
        try:
            # 检查工具是否存在
            if name not in self.tools:
                return build_error(
                    "tool_not_found",
                    f"工具 '{name}' 不存在",
                    available_tools=list(self.tools.keys()),
                )

            # 参数校验与规整
            schema = self.schemas.get(name, {})
            parameters = schema.get("function", {}).get("parameters")
            validation = self._argument_validator.validate_and_coerce(arguments, parameters)
            if not validation["ok"]:
                return validation

            if name == "fetch_more":
                return self._truncation_manager.execute_fetch_more(validation["arguments"], context)

            # 执行工具前：自动路径安全检查
            arguments = validation["arguments"]
            func = self.tools[name]

            # 检查工具是否声明了 file_path_params
            tool_extra = getattr(func, "__tool_extra__", None)
            file_path_params = getattr(tool_extra, "__file_path_params__", None)
            if file_path_params:
                if not self.allowed_paths:
                    Log.error(
                        (
                            f"工具 {name} 声明了 file_path_params={file_path_params}，"
                            "但未注册任何 allowed_paths，拒绝执行（fail-closed）"
                        ),
                        module=MODULE,
                    )
                    return build_error(
                        "permission_denied",
                        "未配置路径白名单，拒绝访问文件系统",
                        hint=f"工具 {name} 需要先通过 register_allowed_paths() 注册允许访问路径",
                    )
                for param_name in file_path_params:
                    if param_name in arguments:
                        param_value = arguments[param_name]
                        try:
                            # 自动验证路径并替换为规范化的绝对路径
                            validated_path = self._validate_path(param_value)
                            arguments[param_name] = str(validated_path)
                        except (PermissionError, FileNotFoundError) as e:
                            # 路径验证失败，返回错误
                            Log.warn(f"工具 {name} 路径验证失败: {param_name}={param_value} - {e}", module=MODULE)
                            return build_error(
                                "permission_denied" if isinstance(e, PermissionError) else "file_not_found",
                                str(e),
                                hint=f"参数 {param_name} 的路径验证失败",
                            )

            execution_context_param_name = self.get_execution_context_param_name(name)
            if execution_context_param_name is None:
                result = func(**arguments)
            else:
                call_arguments = dict(arguments)
                call_arguments[execution_context_param_name] = context
                result = func(**call_arguments)

            # 截断处理
            truncate_spec = getattr(tool_extra, "__truncate__", None)
            value, truncation = self._truncation_manager.apply_truncation(
                name=name,
                arguments=arguments,
                value=result,
                context=context,
                truncate_spec=truncate_spec,
            )
            result = build_success(value=value, truncation=truncation)
            # 链式执行 response middleware；透传 context 以支持按轮次感知（index_in_iteration）
            for middleware in self._response_middlewares:
                result = middleware(name, result, context)
            return result

        except ToolBusinessError as e:
            Log.warn(_format_tool_business_error_log(name, e), module=MODULE)
            return build_error(e.code, e.message, hint=e.hint, **e.extra)

        except CancelledError:
            Log.warn(f"工具 {name} 执行被取消", module=MODULE)
            return build_error("cancelled", "tool execution cancelled")
            
        except FileNotFoundError as e:
            Log.warn(f"工具 {name} 执行失败: 文件不存在 - {e}", module=MODULE)
            return build_error("file_not_found", "文件不存在", hint=str(e))
        except (PermissionError, FileAccessError) as e:
            Log.warn(f"工具 {name} 执行失败: 权限不足 - {e}", module=MODULE)
            return build_error("permission_denied", "权限不足或路径不在安全范围内", hint=str(e))
            
        except TypeError as e:
            Log.error(f"工具 {name} 执行失败: 参数错误 - {e}", module=MODULE)
            return build_error("invalid_argument", "参数类型或数量错误", hint=str(e))
            
        except Exception as e:
            Log.error(f"工具 {name} 执行失败: {type(e).__name__} - {e}", module=MODULE)
            return build_error("execution_error", type(e).__name__, hint=str(e))
