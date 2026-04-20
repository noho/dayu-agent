"""跨层共享的环境变量名常量。

该模块集中定义被多层（CLI、Service、Domain）引用的环境变量名，
避免同一字符串在多处重复硬编码。
"""

SEC_USER_AGENT_ENV = "SEC_USER_AGENT"
"""SEC 下载请求使用的 User-Agent 环境变量名。"""

TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
"""Tavily 联网检索 API Key 环境变量名。"""

SERPER_API_KEY_ENV = "SERPER_API_KEY"
"""Serper 联网检索 API Key 环境变量名。"""

FMP_API_KEY_ENV = "FMP_API_KEY"
"""Financial Modeling Prep API Key 环境变量名。"""

FINS_PROCESSOR_PROFILE_ENV = "FINS_PROCESSOR_PROFILE"
"""处理器性能 profiling 开关环境变量名。"""
