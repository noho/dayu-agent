"""配置文件解析基础设施。

该模块负责：
- 统一处理「工作区配置目录 -> 包内默认配置目录」的 fallback 顺序
- 提供文本、JSON 与 YAML 文件的通用读取能力

模块不承载任何业务语义，不理解 run.json、llm_models.json 或 prompts 目录结构。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional, overload

import yaml

from dayu.contracts.infrastructure import StructuredConfigValue
from dayu.log import Log

MODULE = "CONFIG.FILE_RESOLVER"


def resolve_package_config_path() -> Path:
    """解析包内默认配置目录路径。

    Returns:
        ``dayu/config`` 的绝对路径。

    Raises:
        无。
    """

    return (Path(__file__).resolve().parent.parent / "config").resolve()


def resolve_package_assets_path() -> Path:
    """解析包内默认 assets 目录路径。

    Returns:
        ``dayu/assets`` 的绝对路径。

    Raises:
        无。
    """

    return (Path(__file__).resolve().parent.parent / "assets").resolve()


class ConfigFileResolver:
    """配置目录 fallback 解析器。"""

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """初始化解析器。

        Args:
            config_dir: 工作区配置目录；为空时默认使用包内 ``dayu/config``。

        Returns:
            无。

        Raises:
            无。
        """

        package_config_path = resolve_package_config_path()
        if config_dir is None:
            self.config_dirs: list[Path] = [package_config_path]
            Log.verbose(f"配置文件解析器初始化: config_dir={package_config_path}", module=MODULE)
            return

        resolved = Path(config_dir).resolve()
        if resolved == package_config_path:
            self.config_dirs = [package_config_path]
            Log.verbose(f"配置文件解析器初始化: config_dir={package_config_path}", module=MODULE)
            return

        self.config_dirs = [resolved, package_config_path]
        Log.verbose(
            f"配置文件解析器初始化: config_dir={resolved}, fallback={package_config_path}",
            module=MODULE,
        )

    @overload
    def read_text(self, relative_path: str, required: Literal[True] = True) -> str:
        ...

    @overload
    def read_text(self, relative_path: str, required: Literal[False]) -> Optional[str]:
        ...

    def read_text(self, relative_path: str, required: bool = True) -> Optional[str]:
        """读取文本文件。

        Args:
            relative_path: 相对 config 根目录的路径。
            required: 是否必须存在。

        Returns:
            文件文本；当 ``required=False`` 且文件不存在时返回 ``None``。

        Raises:
            FileNotFoundError: 当 ``required=True`` 且文件不存在时抛出。
        """

        normalized_path = relative_path.strip().lstrip("/")
        for config_path in self.config_dirs:
            file_path = config_path / normalized_path
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                Log.verbose(f"从 {config_path} 加载文件: {normalized_path}", module=MODULE)
                return content

        if required:
            if len(self.config_dirs) > 1:
                search_paths = "\n".join([f"  - {path / normalized_path}" for path in self.config_dirs])
                error_msg = f"文件 {normalized_path} 不存在\n查找路径:\n{search_paths}"
            else:
                error_msg = f"文件 {normalized_path} 不存在\n  查找路径: {self.config_dirs[0] / normalized_path}"
            Log.error(error_msg, module=MODULE)
            raise FileNotFoundError(error_msg)

        Log.warning(f"文件 {normalized_path} 不存在（可选）", module=MODULE)
        return None

    @overload
    def read_json(self, relative_path: str, required: Literal[True] = True) -> StructuredConfigValue:
        ...

    @overload
    def read_json(self, relative_path: str, required: Literal[False]) -> Optional[StructuredConfigValue]:
        ...

    def read_json(self, relative_path: str, required: bool = True) -> Optional[StructuredConfigValue]:
        """读取 JSON 文件。

        Args:
            relative_path: 相对 config 根目录的路径。
            required: 是否必须存在。

        Returns:
            解析后的 JSON 对象；当 ``required=False`` 且文件不存在时返回 ``None``。

        Raises:
            FileNotFoundError: 当 ``required=True`` 且文件不存在时抛出。
            json.JSONDecodeError: JSON 解析失败时抛出。
        """

        content = self.read_text(relative_path, required=required)
        if content is None:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            Log.error(f"解析 JSON 文件 {relative_path} 失败: {exc}", module=MODULE)
            raise

    @overload
    def read_yaml(self, relative_path: str, required: Literal[True] = True) -> StructuredConfigValue:
        ...

    @overload
    def read_yaml(self, relative_path: str, required: Literal[False]) -> Optional[StructuredConfigValue]:
        ...

    def read_yaml(self, relative_path: str, required: bool = True) -> Optional[StructuredConfigValue]:
        """读取 YAML 文件。

        Args:
            relative_path: 相对 config 根目录的路径。
            required: 是否必须存在。

        Returns:
            解析后的 YAML 对象；当 ``required=False`` 且文件不存在时返回 ``None``。

        Raises:
            FileNotFoundError: 当 ``required=True`` 且文件不存在时抛出。
            yaml.YAMLError: YAML 解析失败时抛出。
        """

        content = self.read_text(relative_path, required=required)
        if content is None:
            return None
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError as exc:
            Log.error(f"解析 YAML 文件 {relative_path} 失败: {exc}", module=MODULE)
            raise
