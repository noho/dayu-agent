"""MinerU 集成配置模块。

统一管理 MinerU 云 API / 本地运行时的所有配置项。
所有配置通过环境变量注入，提供类型安全的读取与合理默认值。
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 环境变量名常量
# ---------------------------------------------------------------------------

ENV_MINERU_API_BASE = "DAYU_MINERU_API_BASE"
ENV_MINERU_TOKEN = "DAYU_MINERU_TOKEN"
ENV_MINERU_CHUNK_SIZE = "DAYU_MINERU_CHUNK_SIZE"
ENV_MINERU_LANG = "DAYU_MINERU_LANG"
ENV_MINERU_OCR = "DAYU_MINERU_OCR"
ENV_MINERU_TIMEOUT = "DAYU_MINERU_TIMEOUT"
ENV_QUOTA_DAILY_LIMIT = "DAYU_QUOTA_DAILY_LIMIT"


def get_mineru_api_base() -> str:
    """获取 MinerU 云 API 基础地址。

    Returns:
        API 基础地址字符串。
    """
    return os.environ.get(ENV_MINERU_API_BASE, "https://mineru.net")


def get_mineru_token() -> str:
    """获取 MinerU 云 API Token。

    Returns:
        Token 字符串；未配置时为空字符串。
    """
    return os.environ.get(ENV_MINERU_TOKEN, "")


def get_mineru_chunk_size() -> int:
    """获取分批大小（每批最多页数）。

    Returns:
        分批大小，默认 200。
    """
    return int(os.environ.get(ENV_MINERU_CHUNK_SIZE, "200"))


def get_mineru_lang() -> str:
    """获取 MinerU 解析语言。

    Returns:
        语言标识，默认 ``"ch"``（中文）。
    """
    return os.environ.get(ENV_MINERU_LANG, "ch")


def get_mineru_ocr_enabled() -> bool:
    """获取 OCR 是否启用。

    Returns:
        ``True`` 表示启用，``False`` 表示禁用。
    """
    return os.environ.get(ENV_MINERU_OCR, "1") == "1"


def get_mineru_timeout() -> float:
    """获取单批轮询超时（秒）。

    Returns:
        超时秒数，默认 1800。
    """
    return float(os.environ.get(ENV_MINERU_TIMEOUT, "1800"))


def get_quota_daily_limit() -> int:
    """获取每日配额上限（页）。

    Returns:
        配额上限，默认 5000。
    """
    return int(os.environ.get(ENV_QUOTA_DAILY_LIMIT, "5000"))


def get_quota_state_file() -> str:
    """获取配额状态持久化文件路径。

    Returns:
        文件路径字符串，默认 ``"~/.dayu/quota_state.json"``。
    """
    return os.environ.get("DAYU_QUOTA_STATE_FILE", "~/.dayu/quota_state.json")


__all__ = [
    "ENV_MINERU_API_BASE",
    "ENV_MINERU_TOKEN",
    "ENV_MINERU_CHUNK_SIZE",
    "ENV_MINERU_LANG",
    "ENV_MINERU_OCR",
    "ENV_MINERU_TIMEOUT",
    "ENV_QUOTA_DAILY_LIMIT",
    "get_mineru_api_base",
    "get_mineru_token",
    "get_mineru_chunk_size",
    "get_mineru_lang",
    "get_mineru_ocr_enabled",
    "get_mineru_timeout",
    "get_quota_daily_limit",
    "get_quota_state_file",
]
