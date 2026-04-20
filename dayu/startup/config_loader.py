"""应用配置加载模块。

该模块只负责应用级配置语义：
- run.json
- llm_models.json

prompt 目录结构与 prompt 资产读取不再由本模块负责。
"""

from copy import deepcopy
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, cast

from dayu.contracts.infrastructure import StructuredConfigObject, StructuredConfigValue
from dayu.contracts.model_config import ModelConfig, ModelConfigJsonValue, ensure_runner_type_enabled
from dayu.startup.config_file_resolver import ConfigFileResolver
from dayu.log import Log

MODULE = "CONFIG"
_ENV_VAR_PATTERN = re.compile(r"\{\{([A-Z_][A-Z0-9_]*)\}\}")
_ENV_VAR_SCAN_FILE_SUFFIXES = frozenset({".json", ".yaml", ".yml"})


def _env_var_replacer(match: re.Match) -> str:
    """替换单个环境变量引用
    
    将 {{ENV_VAR_NAME}} 格式替换为对应的环境变量值。
    如果环境变量未设置，保持原始格式并输出警告日志。
    
    Args:
        match: 正则匹配对象，group(1) 为环境变量名
    
    Returns:
        环境变量值，或原始 {{VAR}} 格式（未设置时）
    """
    env_var = match.group(1)
    value = os.environ.get(env_var)
    if value is None:
        Log.warning(f"环境变量 {env_var} 未设置", module=MODULE)
        return match.group(0)  # 保持原始 {{VAR}} 格式
    return value


def _extract_env_var_names_from_text(content: str) -> tuple[str, ...]:
    """从文本中提取环境变量占位符名称。

    Args:
        content: 原始文本内容。

    Returns:
        文本中出现的环境变量名，按字典序去重后返回。

    Raises:
        无。
    """

    found_names = {match.group(1) for match in _ENV_VAR_PATTERN.finditer(str(content or ""))}
    return tuple(sorted(found_names))


def _collect_env_var_names_from_model_config(
    value: ModelConfig | ModelConfigJsonValue,
) -> tuple[str, ...]:
    """递归收集模型配置 JSON 中引用的环境变量名。

    Args:
        value: 模型配置 JSON 值。

    Returns:
        引用到的环境变量名，按字典序去重后返回。

    Raises:
        无。
    """

    collected_names: set[str] = set()
    _collect_env_var_names_from_model_config_value(value=value, collected_names=collected_names)
    return tuple(sorted(collected_names))


def _collect_env_var_names_from_model_config_value(
    *,
    value: ModelConfig | ModelConfigJsonValue,
    collected_names: set[str],
) -> None:
    """递归遍历模型配置值并收集环境变量名。

    Args:
        value: 当前遍历到的模型配置值。
        collected_names: 累积收集结果。

    Returns:
        无。

    Raises:
        无。
    """

    if isinstance(value, dict):
        for child in value.values():
            _collect_env_var_names_from_model_config_value(
                value=cast(ModelConfig | ModelConfigJsonValue, child),
                collected_names=collected_names,
            )
        return
    if isinstance(value, list):
        for child in value:
            _collect_env_var_names_from_model_config_value(
                value=cast(ModelConfig | ModelConfigJsonValue, child),
                collected_names=collected_names,
            )
        return
    if isinstance(value, str):
        collected_names.update(_extract_env_var_names_from_text(value))


def _should_scan_env_var_file(file_path: Path) -> bool:
    """判断某个文件是否应参与环境变量占位符扫描。

    当前只扫描结构化配置文件，避免把 ``README``、缓存产物或其它非配置文本
    误纳入结果，也避免把二进制缓存文件按 UTF-8 解码。

    Args:
        file_path: 待判断文件路径。

    Returns:
        ``True`` 表示应扫描；否则返回 ``False``。

    Raises:
        无。
    """

    if not file_path.is_file():
        return False
    if file_path.suffix.lower() not in _ENV_VAR_SCAN_FILE_SUFFIXES:
        return False
    return "__pycache__" not in file_path.parts


def _require_structured_config_object(
    payload: StructuredConfigValue | None,
    *,
    filename: str,
) -> StructuredConfigObject:
    """要求配置文件读取结果必须是 JSON object。

    Args:
        payload: 配置文件读取结果。
        filename: 配置文件名，用于报错。

    Returns:
        经过结构校验后的配置对象。

    Raises:
        RuntimeError: 当读取结果为空时抛出。
        TypeError: 当顶层不是对象时抛出。
    """

    if payload is None:
        raise RuntimeError(f"{filename} 读取结果为空")
    if not isinstance(payload, dict):
        raise TypeError(f"{filename} 必须是对象")
    return cast(StructuredConfigObject, payload)


def _replace_model_config_env_vars(value: ModelConfigJsonValue) -> ModelConfigJsonValue:
    """递归替换模型配置中的环境变量占位符。

    Args:
        value: 模型配置 JSON 值。

    Returns:
        完成环境变量替换后的模型配置 JSON 值。

    Raises:
        无。
    """

    if isinstance(value, dict):
        return {key: _replace_model_config_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_model_config_env_vars(item) for item in value]
    if isinstance(value, str):
        return re.sub(_ENV_VAR_PATTERN, _env_var_replacer, value)
    return value


class ConfigLoader:
    """应用配置加载器。"""
    
    @staticmethod
    def _replace_env_vars(obj: StructuredConfigValue) -> StructuredConfigValue:
        """递归替换结构化配置中的环境变量引用。

        Args:
            obj: 待处理的结构化配置值。

        Returns:
            完成环境变量替换后的结构化配置值。

        Raises:
            无。
        """

        if isinstance(obj, dict):
            return cast(
                StructuredConfigObject,
                {key: ConfigLoader._replace_env_vars(value) for key, value in obj.items()},
            )
        if isinstance(obj, list):
            return [ConfigLoader._replace_env_vars(item) for item in obj]
        if isinstance(obj, str):
            return re.sub(_ENV_VAR_PATTERN, _env_var_replacer, obj)
        return obj
    
    def __init__(self, resolver: ConfigFileResolver):
        """初始化配置加载器。

        Args:
            resolver: 配置文件解析器实例。

        Returns:
            无。

        Raises:
            无。
        """
        self._resolver = resolver
        self._run_config_cache: StructuredConfigObject | None = None
        self._llm_models_cache: dict[str, ModelConfig] | None = None
        self._toolset_registrars_cache: dict[str, str] | None = None
    
    def load_run_config(self) -> StructuredConfigObject:
        """加载运行配置（run.json）
        
        包含：
        - runner_running_config: Runner 调试与超时配置
        - agent_running_config: Agent 运行配置
        - doc_tool_limits: 文档工具参数上限
        - fins_tool_limits: 财报工具参数上限
        - web_tools_config: 联网工具配置
        - host_config: 宿主 SQLite 存储、并发 lane 与 pending turn resume 策略
        - tool_trace_config: 工具调用追踪配置
        - conversation_memory: 多轮会话记忆配置
        
        Returns:
            运行配置字典
        
        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        if self._run_config_cache is None:
            self._run_config_cache = _require_structured_config_object(
                self._resolver.read_json("run.json", required=True),
                filename="run.json",
            )
        run_config = self._run_config_cache
        if run_config is None:
            raise RuntimeError("run.json 缓存为空")
        return run_config
    
    def load_llm_models(self) -> dict[str, ModelConfig]:
        """加载所有 LLM 模型配置（llm_models.json）
        
        Returns:
            所有模型配置字典，key 为模型名称，value 为配置详情
        
        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON 格式错误
        """
        if self._llm_models_cache is None:
            loaded_models = _require_structured_config_object(
                self._resolver.read_json("llm_models.json", required=True),
                filename="llm_models.json",
            )
            normalized_models: dict[str, ModelConfig] = {}
            for raw_model_name, raw_model_config in loaded_models.items():
                model_name = str(raw_model_name or "").strip()
                if not model_name:
                    raise TypeError("llm_models.json 的 key 必须是非空字符串")
                if model_name.startswith("_"):
                    continue
                if not isinstance(raw_model_config, dict):
                    raise TypeError(f"llm_models.json.{model_name} 必须是对象")
                normalized_models[model_name] = cast(ModelConfig, raw_model_config)
            self._llm_models_cache = normalized_models
        llm_models = self._llm_models_cache
        if llm_models is None:
            raise RuntimeError("llm_models.json 缓存为空")
        return cast(dict[str, ModelConfig], deepcopy(llm_models))
    
    def load_llm_model(self, model_name: str) -> ModelConfig:
        """加载指定 LLM 模型的配置（包含环境变量替换）
        
        Args:
            model_name: 模型名称（如 "deepseek_chat"）
        
        Returns:
            指定模型的配置字典（已替换环境变量），包含：
            - runner_type: 运行方式
            - endpoint_url: API 端点
            - model: 模型名称
            - headers: HTTP 请求头（支持 {{ENV_VAR}} 占位符）
            - 等...
        
        Raises:
            FileNotFoundError: 配置文件不存在
            json.JSONDecodeError: JSON 格式错误
            KeyError: 模型名称不存在
        """
        models = self.load_llm_models()
        if model_name not in models:
            available_models = ", ".join(models.keys())
            error_msg = (
                f"模型 '{model_name}' 不存在\n"
                f"  可用模型: {available_models}"
            )
            Log.error(error_msg, module=MODULE)
            raise KeyError(error_msg)

        model_config = cast(
            ModelConfig,
            _replace_model_config_env_vars(cast(ModelConfigJsonValue, models[model_name])),
        )
        ensure_runner_type_enabled(model_config.get("runner_type"))
        return cast(ModelConfig, model_config)

    def load_toolset_registrars(self) -> dict[str, str]:
        """读取 toolset registrar 安装清单。

        Returns:
            ``toolset_name -> registrar_import_path`` 映射。

        Raises:
            TypeError: 当配置结构非法时抛出。
        """

        if self._toolset_registrars_cache is None:
            loaded_config = _require_structured_config_object(
                self._resolver.read_json("toolset_registrars.json", required=True),
                filename="toolset_registrars.json",
            )
            normalized: dict[str, str] = {}
            for raw_toolset_name, raw_import_path in loaded_config.items():
                toolset_name = str(raw_toolset_name or "").strip()
                import_path = str(raw_import_path or "").strip()
                if not toolset_name:
                    raise TypeError("toolset_registrars.json 的 key 必须是非空字符串")
                if not import_path:
                    raise TypeError(
                        f"toolset_registrars.json.{toolset_name} 必须是非空字符串"
                    )
                normalized[toolset_name] = import_path
            self._toolset_registrars_cache = normalized

        cached = self._toolset_registrars_cache
        if cached is None:
            raise RuntimeError("toolset_registrars.json 缓存为空")
        return dict(cached)

    def collect_referenced_env_vars(self) -> tuple[str, ...]:
        """收集当前生效配置中引用的环境变量名称。

        扫描顺序遵循配置 fallback 规则：
        - 若工作区配置存在某个相对路径，则只读取工作区版本；
        - 否则回退到包内默认配置版本。

        Args:
            无。

        Returns:
            当前生效配置里引用到的环境变量名，按字典序去重后返回。

        Raises:
            OSError: 当配置文件读取失败时抛出。
        """

        referenced_names: set[str] = set()
        seen_relative_paths: set[str] = set()
        for config_dir in self._resolver.config_dirs:
            for file_path in sorted(path for path in config_dir.rglob("*") if _should_scan_env_var_file(path)):
                relative_path = file_path.relative_to(config_dir).as_posix()
                if relative_path in seen_relative_paths:
                    continue
                seen_relative_paths.add(relative_path)
                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    Log.warning(f"跳过非 UTF-8 配置文件: {file_path}", module=MODULE)
                    continue
                referenced_names.update(_extract_env_var_names_from_text(content))
        return tuple(sorted(referenced_names))

    def collect_model_referenced_env_vars(self, model_names: Iterable[str]) -> tuple[str, ...]:
        """收集指定模型配置引用的环境变量名称。

        该方法只分析指定模型在 ``llm_models.json`` 中的原始配置，
        不执行环境变量替换，因此可用于启动前 fail-fast 校验，避免先触发
        缺失环境变量 warning 再在运行期失败。

        Args:
            model_names: 需要分析的模型名序列。

        Returns:
            指定模型引用到的环境变量名，按字典序去重后返回。

        Raises:
            KeyError: 当任一模型名在 ``llm_models.json`` 中不存在时抛出。
        """

        models = self.load_llm_models()
        normalized_model_names = tuple(
            dict.fromkeys(
                str(model_name or "").strip()
                for model_name in model_names
                if str(model_name or "").strip()
            )
        )
        if not normalized_model_names:
            return ()

        missing_model_names: list[str] = []
        referenced_names: set[str] = set()
        for model_name in normalized_model_names:
            raw_model_config = models.get(model_name)
            if raw_model_config is None:
                missing_model_names.append(model_name)
                continue
            referenced_names.update(_collect_env_var_names_from_model_config(raw_model_config))

        if missing_model_names:
            available_models = ", ".join(sorted(models.keys()))
            missing_models = ", ".join(missing_model_names)
            error_message = (
                f"模型不存在: {missing_models}\n"
                f"  可用模型: {available_models}"
            )
            Log.error(error_message, module=MODULE)
            raise KeyError(error_message)

        return tuple(sorted(referenced_names))
    
