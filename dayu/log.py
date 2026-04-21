"""全局日志模块 - 基于 logging 的统一日志接口。

提供简单的分级日志能力，支持：
- LogLevel 分级过滤（映射到 logging 级别）
- 时间戳与统一格式输出
- 标准 logging 模块的所有特性
- 可选的异常堆栈打印

默认输出策略：
- `ERROR` 以下日志输出到 stdout，便于后台托管场景保留运行时间线
- `ERROR` 及以上只输出到 stderr，避免终端重复显示同一条失败日志

该模块是全局基础设施，不属于任何架构层，所有包均直接 import。
"""

import logging
import sys
from enum import Enum
from typing import TextIO


class LogLevel(Enum):
    """日志级别（映射到 logging 级别）。"""

    DEBUG = logging.DEBUG      # 10
    VERBOSE = 15               # 自定义级别，介于 DEBUG 和 INFO 之间
    INFO = logging.INFO        # 20
    WARN = logging.WARNING     # 30
    ERROR = logging.ERROR      # 40


# 注册自定义 VERBOSE 级别
logging.addLevelName(LogLevel.VERBOSE.value, "VERBOSE")

_DEFAULT_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DEFAULT_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_STDERR_MIN_LEVEL = logging.ERROR
_STDOUT_MAX_LEVEL = _STDERR_MIN_LEVEL - 1

# ------------------------------------------------------------------
# 第三方库日志抑制规则
# ------------------------------------------------------------------
_ALWAYS_WARNING_LOGGERS = ("RapidOCR", "httpx", "asyncio", "charset_normalizer", "PIL", "chardet.charsetprober", "matplotlib")
_DEBUG_MODE_INFO_LOGGERS = ("httpcore", "urllib3", "urllib3.connectionpool", "edgar", "docling", "readability.readability")


class _LevelRangeFilter(logging.Filter):
    """按日志级别区间过滤记录。

    Args:
        min_level: 允许通过的最小级别；为空时不限制下界。
        max_level: 允许通过的最大级别；为空时不限制上界。

    Returns:
        无。

    Raises:
        无。
    """

    def __init__(self, *, min_level: int | None = None, max_level: int | None = None) -> None:
        """初始化级别区间过滤器。

        Args:
            min_level: 允许通过的最小级别；为空时不限制下界。
            max_level: 允许通过的最大级别；为空时不限制上界。

        Returns:
            无。

        Raises:
            无。
        """

        super().__init__()
        self._min_level = min_level
        self._max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        """判断当前日志记录是否允许通过。

        Args:
            record: 待过滤的日志记录。

        Returns:
            若日志级别在允许区间内则返回 `True`，否则返回 `False`。

        Raises:
            无。
        """

        if self._min_level is not None and record.levelno < self._min_level:
            return False
        if self._max_level is not None and record.levelno > self._max_level:
            return False
        return True


def _build_stream_handler(
    *,
    stream: TextIO,
    min_level: int | None = None,
    max_level: int | None = None,
) -> logging.Handler:
    """构建默认日志输出 handler。

    Args:
        stream: 目标输出流。
        min_level: 允许通过的最小级别；为空时不限制下界。
        max_level: 允许通过的最大级别；为空时不限制上界。

    Returns:
        已绑定统一格式与级别过滤器的 `logging.Handler`。

    Raises:
        无。
    """

    handler = logging.StreamHandler(stream=stream)
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(logging.Formatter(fmt=_DEFAULT_LOG_FORMAT, datefmt=_DEFAULT_LOG_DATE_FORMAT))
    handler.addFilter(_LevelRangeFilter(min_level=min_level, max_level=max_level))
    return handler


class Log:
    """全局日志输出类（基于 logging 模块）。"""

    _loggers: dict[str, logging.Logger] = {}
    _configured: bool = False

    @classmethod
    def _ensure_configured(cls) -> None:
        """确保 logging 已配置（仅配置一次）。

        Args:
            无。

        Returns:
            无。

        Raises:
            无。
        """

        if not cls._configured:
            root_logger = logging.getLogger()
            if not root_logger.handlers:
                root_logger.setLevel(logging.INFO)
                root_logger.addHandler(
                    _build_stream_handler(stream=sys.stdout, max_level=_STDOUT_MAX_LEVEL)
                )
                root_logger.addHandler(_build_stream_handler(stream=sys.stderr, min_level=_STDERR_MIN_LEVEL))
            cls._configured = True

    @classmethod
    def _get_logger(cls, module: str) -> logging.Logger:
        """获取或创建指定模块的 logger。"""

        cls._ensure_configured()
        if module not in cls._loggers:
            cls._loggers[module] = logging.getLogger(module)
        return cls._loggers[module]

    @classmethod
    def set_level(cls, min_level: LogLevel) -> None:
        """设置全局最小日志级别。

        Args:
            min_level: 最小日志级别，低于此级别的日志不会输出。
        """

        cls._ensure_configured()
        logging.getLogger().setLevel(min_level.value)

        for name in _ALWAYS_WARNING_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

        verbose_level = logging.INFO if min_level.value <= LogLevel.DEBUG.value else logging.WARNING
        for name in _DEBUG_MODE_INFO_LOGGERS:
            logging.getLogger(name).setLevel(verbose_level)

    @classmethod
    def debug(cls, message: str, *, module: str = "APP") -> None:
        """输出调试信息。"""

        cls._get_logger(module).debug(message)

    @classmethod
    def verbose(cls, message: str, *, module: str = "APP") -> None:
        """输出详细信息（自定义级别）。"""

        cls._get_logger(module).log(LogLevel.VERBOSE.value, message)

    @classmethod
    def info(cls, message: str, *, module: str = "APP") -> None:
        """输出常规信息。"""

        cls._get_logger(module).info(message)

    @classmethod
    def warn(cls, message: str, *, module: str = "APP") -> None:
        """输出警告信息。"""

        cls._get_logger(module).warning(message)

    @classmethod
    def warning(cls, message: str, *, module: str = "APP") -> None:
        """输出警告信息（warn 的别名）。"""

        cls.warn(message, module=module)

    @classmethod
    def error(cls, message: str, exc_info: bool = False, *, module: str = "APP") -> None:
        """输出错误信息。

        Args:
            message: 错误消息。
            exc_info: 是否输出异常堆栈跟踪。
            module: 模块名。
        """

        cls._get_logger(module).error(message, exc_info=exc_info)


__all__ = ["Log", "LogLevel"]
