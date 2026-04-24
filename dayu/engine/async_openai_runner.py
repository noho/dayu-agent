"""
异步 OpenAI 兼容 API 运行器

支持通过 HTTP API 调用 OpenAI 兼容的服务（如 DeepSeek、OpenAI、Claude 等），
提供 SSE (Server-Sent Events) 流式响应和工具调用功能。

核心特性:
- 自动重试机制：网络超时、服务器错误、限流等可恢复错误
- 智能退避策略：支持 Retry-After 头和指数退避
- 流式和非流式响应处理（自动检测 Content-Type）
- 工具调用自动执行：Runner 内部完成工具执行，上层只需监听事件
- 完整的错误处理和事件流
- 两层 extra_payloads 优先级：实例默认 < 调用参数
- 环境变量替换：支持 {{ENV_VAR_NAME}} 格式
- 能力声明：supports_stream、supports_tool_calling 配置级控制

工具执行机制:
- 发起：tool_call_dispatched + tool_calls_batch_ready
- 执行：Runner 并发执行一批工具调用（可配置 tool_timeout_seconds）
 - 完成：逐个 tool_call_result（包含 ok/value/error/meta）+ tool_calls_batch_done
- 关键设计：工具失败封装为 tool_call_result（而非 error_event），上层可回填给 LLM

配置示例（llm_models.json）：
{
  "deepseek_chat": {
    "endpoint_url": "https://api.deepseek.com/v1/chat/completions",
    "model": "deepseek-v4-flash",
    "temperature": 0.7,
    "headers": {
      "Authorization": "Bearer {{DEEPSEEK_API_KEY}}",
      "Content-Type": "application/json"
    },
    "timeout": 3600,
    "supports_stream": true,
    "supports_tool_calling": true
  }
}

主要方法：

1. __init__(*, endpoint_url, model, headers, name=None, temperature=0.7,
            default_extra_payloads=None, timeout=3600, max_retries=3,
            supports_stream=True, supports_tool_calling=True,
            debug_sse=False, debug_tool_delta=False,
            debug_sse_sample_rate=1.0, debug_sse_throttle_sec=0.0)
   初始化 Runner 实例
   - endpoint_url: OpenAI 兼容的 API 端点 URL
   - model: 模型名称
   - headers: HTTP 请求头（包含 Authorization）
   - name: Runner 名称（用于日志，默认使用 model）
   - temperature: 温度参数 (0.0-2.0)
    - default_extra_payloads: 实例级默认请求参数（可被 call() 的参数覆盖，但不能覆盖保留字段）
   - timeout: 单次请求超时时间（秒）
   - max_retries: 最大重试次数
   - supports_stream: 是否支持流式响应
   - supports_tool_calling: 是否支持工具调用
   - debug_sse: 是否输出 SSE 相关高频调试日志
   - debug_tool_delta: 是否输出工具调用增量调试日志
   - debug_sse_sample_rate: SSE 调试日志采样率（0-1）
   - debug_sse_throttle_sec: SSE 调试日志节流窗口（秒）

2. set_default_extra_payloads(payloads: Dict[str, Any])
    设置实例级默认的额外请求参数（会被 call() 的参数覆盖，但不能覆盖保留字段）
   示例：runner.set_default_extra_payloads({"max_tokens": 2000})

3. set_tools(executor: Optional[ToolExecutor])
   设置工具执行器（工具定义从 executor.get_schemas() 获取）

4. async call(messages, *, stream=True, **extra_payloads) -> AsyncIterator[StreamEvent]
   调用 OpenAI 兼容 API 并返回事件流
   - messages: OpenAI 格式的消息列表
   - stream: 是否启用流式响应（模型不支持时自动降级）
    - **extra_payloads: 额外的请求参数（优先级高于 default_extra_payloads，但不能覆盖保留字段）
   
   事件类型：
   - content_delta: 内容增量
   - content_complete: 内容完成（总是触发，即使内容为空）
   - tool_call_start: 工具调用开始（可用于流式增量展示）
   - tool_call_delta: 工具调用参数增量（可用于流式增量展示）
   - tool_call_dispatched: 工具调用已发起执行
   - tool_calls_batch_ready: 工具调用批次已就绪
   - tool_call_result: 工具执行结果（包含 ok/value/error/meta）
   - tool_calls_batch_done: 工具调用批次完成
   - warning_event: 警告（如重试）
   - error_event: 错误（包含 error_type 和 recoverable 字段）
   - done_event: 完成
   
   重试机制：
   - 网络超时/连接错误：自动重试
   - HTTP 429 限流：根据 Retry-After 头或指数退避
   - HTTP 50x 服务器错误：自动重试
   - HTTP 40x 客户端错误：不重试，立即返回错误
"""

import asyncio
import json
import time
import uuid
from types import ModuleType
from typing import AsyncIterator, Callable, Dict, List, Optional, Any, TYPE_CHECKING, cast
from dataclasses import dataclass

from dayu.contracts.agent_types import AgentMessage
from dayu.contracts.protocols import ToolExecutionContext
from dayu.contracts.cancellation import CancelledError as EngineCancelledError, CancellationToken

# 可选依赖，导入失败不立即报错
aiohttp: ModuleType | None
try:
    import aiohttp as _aiohttp_module
    aiohttp = _aiohttp_module
except ImportError:
    aiohttp = None

# 类型检查时导入
if TYPE_CHECKING:
    from aiohttp import ClientResponse, ClientSession, ClientTimeout

from .events import (
    EventType,
    StreamEvent,
    content_delta,          ## """创建内容增量事件"""
    content_complete,       ## """创建内容完成事件"""
    reasoning_delta,        ## """创建推理增量事件（thinking 模式思维链片段）"""
    done_event,             ## """创建完成事件"""
    error_event,            ## """创建错误事件"""
    metadata_event,         ## """创建元数据事件"""
    warning_event,          ## """创建警告事件"""
    tool_call_dispatched,   ## """创建工具调用已发起执行事件"""
    tool_call_delta,        ## """创建工具调用参数增量事件"""
    tool_call_result,       ## """创建工具调用结果事件"""
    tool_call_start,        ## """创建工具调用开始事件"""
    tool_calls_batch_ready, ## """创建工具调用批次就绪事件"""
    tool_calls_batch_done,  ## """创建工具调用批次完成事件"""
)
from .protocols import ToolExecutor
from dayu.log import Log
from .tool_result import (
    build_error,
    get_error_code,
    get_error_message,
    get_value,
    is_tool_success,
    validate_tool_result_contract,
)
from .sse_parser import SSEStreamParser

MODULE = "ENGINE.ASYNC_OPENAI_RUNNER"

from dayu.engine.cancellation import await_or_cancel as _await_or_cancel
from dayu.engine.cancellation import cancel_task_and_wait
from dayu.engine.cancellation import create_cancellation_waiter as _create_cancellation_waiter


def _require_aiohttp_module() -> ModuleType:
    """返回已安装的 aiohttp 模块。

    Args:
        无。

    Returns:
        aiohttp 模块对象。

    Raises:
        RuntimeError: aiohttp 未安装时抛出。
    """

    if aiohttp is None:
        raise RuntimeError("aiohttp is required for AsyncOpenAIRunner")
    return aiohttp


def _require_tool_executor(executor: Optional[ToolExecutor]) -> ToolExecutor:
    """确保工具执行器存在。

    Args:
        executor: 当前工具执行器。

    Returns:
        非空工具执行器。

    Raises:
        RuntimeError: 工具执行器未设置时抛出。
    """

    if executor is None:
        raise RuntimeError("tool executor is not set")
    return executor

# HTTP 状态码常量
# 可重试状态码（瞬时故障或限流场景）
RETRIABLE_STATUS_CODES = {
    408,  # Request Timeout：请求超时
    429,  # Too Many Requests：请求过频/限流
    500,  # Internal Server Error：服务端内部错误
    502,  # Bad Gateway：网关错误
    503,  # Service Unavailable：服务繁忙/不可用
    504,  # Gateway Timeout：网关超时
}
# 不可重试状态码（请求本身或账号/权限/内容问题）
NON_RETRIABLE_STATUS_CODES = {
    400,  # Bad Request：请求格式错误
    401,  # Unauthorized：认证失败（API Key/鉴权头错误）
    402,  # Payment Required：余额不足
    403,  # Forbidden：拒绝访问（权限/风控/地区限制）
    404,  # Not Found：资源不存在
    421,  # Content Blocked：内容审核拦截
    422,  # Unprocessable Entity：请求参数错误
}

# 不可重试 HTTP 状态码 -> error_type 的固定映射，提升为模块级常量避免每次请求重建 dict。
_NON_RETRIABLE_ERROR_TYPE_MAP: dict[int, str] = {
    400: "invalid_request",
    401: "auth_error",
    402: "insufficient_quota",
    403: "auth_error",
    404: "resource_not_found",
    421: "content_blocked",
    422: "invalid_request",
}

# 工具调用日志辅助常量
_MAX_ARG_STR_LEN = 40   # 单个参数字符串值的最大显示长度
_MAX_COMPACT_ARGS_LEN = 120  # compact args 总长度上限

# Runner 默认超时与心跳常量。
# - DEFAULT_REQUEST_TIMEOUT_SEC：单次 HTTP 请求总超时（含连接/读取），1 小时给长上下文留余地。
# - DEFAULT_TOOL_TIMEOUT_SEC：单个工具执行超时；财报场景多数工具应在 90s 内完成。
# - DEFAULT_STREAM_IDLE_TIMEOUT_SEC：SSE 流空闲超时；超过 120s 无字节即视为服务端挂起。
# - DEFAULT_STREAM_IDLE_HEARTBEAT_SEC：每 10s 心跳触发一次空闲检查。
# - DEFAULT_ERROR_BODY_PREVIEW_CHARS：日志中的错误体短摘要长度。
# - DEFAULT_ERROR_BODY_DETAIL_CHARS：错误事件 metadata 中的错误体长摘要长度。
_DEFAULT_REQUEST_TIMEOUT_SEC: int = 3600
_DEFAULT_TOOL_TIMEOUT_SEC: float = 90.0
_DEFAULT_STREAM_IDLE_TIMEOUT_SEC: float = 120.0
_DEFAULT_STREAM_IDLE_HEARTBEAT_SEC: float = 10.0
_DEFAULT_ERROR_BODY_PREVIEW_CHARS: int = 200
_DEFAULT_ERROR_BODY_DETAIL_CHARS: int = 400
_RESERVED_EXTRA_PAYLOAD_KEYS = frozenset(
    {
        "messages",
        "model",
        "stream",
        "temperature",
        "tools",
    }
)
_RESERVED_DEFAULT_EXTRA_PAYLOAD_KEYS = frozenset({"trace_context"}).union(
    _RESERVED_EXTRA_PAYLOAD_KEYS
)


def _validate_extra_payload_keys(
    payloads: Dict[str, Any],
    *,
    source: str,
    reserved_keys: frozenset[str],
) -> None:
    """校验额外请求参数是否试图覆盖 Runner 保留字段。

    Args:
        payloads: 待校验的额外请求参数。
        source: 参数来源标识，用于错误提示。
        reserved_keys: 当前入口禁止出现的保留字段集合。

    Returns:
        None。

    Raises:
        ValueError: 当额外参数包含 Runner 保留字段时抛出。
    """

    duplicated_keys = sorted(reserved_keys.intersection(payloads.keys()))
    if duplicated_keys:
        duplicated = ", ".join(duplicated_keys)
        raise ValueError(
            f"{source} 包含 Runner 保留字段: {duplicated}；请改用显式结构化参数"
        )


def _detect_context_overflow(error_body: str) -> bool:
    """检测 HTTP 400 错误是否为上下文长度超限。

    同时检查结构化 JSON error code 与文本关键词，兼容不同服务商的错误格式。

    Args:
        error_body: 错误响应体文本。

    Returns:
        True 表示为上下文超限错误。
    """
    if not error_body:
        return False
    # 结构化检测：OpenAI 标准格式 {"error": {"code": "context_length_exceeded"}}
    try:
        err_obj = json.loads(error_body)
        code = err_obj.get("error", {}).get("code", "")
        if code == "context_length_exceeded":
            return True
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    # 文本兜底：不同服务商可能只在 message 中提及
    lowered = error_body.lower()
    overflow_signals = (
        "maximum context length is",
        "context length exceeded",
        "total message token length exceed model limit",
        "model's maximum context length",
        "range of input length should be",
        "model requires more context",
    )
    return any(signal in lowered for signal in overflow_signals)
_MAX_SUMMARY_FIELDS = 4  # result summary 最多展示字段数
_MAX_SUMMARY_STR_LEN = 40  # result summary 单个字符串值最大显示长度


def _compact_args(arguments: Any, max_len: int = _MAX_COMPACT_ARGS_LEN) -> str:
    """将工具调用参数格式化为紧凑单行字符串，过滤 None 值，便于日志阅读。

    Args:
        arguments: 工具调用参数（通常为 dict）。
        max_len: 输出最大字符数，超出时截断。

    Returns:
        格式化参数字符串，如 "ticker='V', ref='s_0001'"。
    """
    if not isinstance(arguments, dict):
        s = str(arguments)
        return s[:max_len] + "…" if len(s) > max_len else s
    parts = []
    for k, v in arguments.items():
        if v is None:
            continue
        if isinstance(v, str) and len(v) > _MAX_ARG_STR_LEN:
            v_str = repr(v[:_MAX_ARG_STR_LEN]) + "…"
        else:
            v_str = repr(v)
        parts.append(f"{k}={v_str}")
    result = ", ".join(parts)
    return result[:max_len] + "…" if len(result) > max_len else result


def _result_summary(tool_result: dict) -> str:
    """从工具调用结果中提取关键字段，格式化为紧凑摘要，便于日志阅读。

    成功时从 data.value 提取一级原始类型字段（str/int/bool，跳过 list/dict），
    并附加 truncated 标志。失败时返回 "error_code: message"。

    Args:
        tool_result: 工具调用结果 dict（含 ok/value/error/truncation 等字段）。

    Returns:
        格式化摘要字符串，如 "ticker='V' ref='s_0001' truncated=False"。
        失败时返回 "cursor_not_found: cursor not found"。
    """
    if not is_tool_success(tool_result):
        code = get_error_code(tool_result) or "UNKNOWN"
        msg = get_error_message(tool_result) or ""
        return f"{code}: {msg}" if msg else code
    parts = []
    value = get_value(tool_result)
    if isinstance(value, dict):
        for k, v in value.items():
            if len(parts) >= _MAX_SUMMARY_FIELDS:
                break
            if isinstance(v, bool):
                # bool 先判断，因为 bool 是 int 子类
                parts.append(f"{k}={v}")
            elif isinstance(v, (str, int)):
                if isinstance(v, str) and len(v) > _MAX_SUMMARY_STR_LEN:
                    parts.append(f"{k}={v[:_MAX_SUMMARY_STR_LEN]!r}…")
                else:
                    parts.append(f"{k}={v!r}")
    elif isinstance(value, (bytes, bytearray)):
        parts.append(f"binary={len(value)}b")
    elif isinstance(value, str):
        rendered = value[:_MAX_SUMMARY_STR_LEN]
        suffix = "…" if len(value) > _MAX_SUMMARY_STR_LEN else ""
        parts.append(f"content={rendered!r}{suffix}")
    elif isinstance(value, (int, float, bool)):
        parts.append(f"content={value!r}")
    truncation = tool_result.get("truncation")
    if truncation is not None:
        parts.append("truncated=True")
    return " ".join(parts) if parts else "ok"


def _extract_error_body_slices(error_body: str) -> tuple[str, str]:
    """从错误响应体中截取日志预览与事件 detail 两个切片。

    Args:
        error_body: HTTP 错误响应的原始文本，可能为空串。

    Returns:
        ``(error_preview, error_detail)`` 二元组。空响应统一降级为
        ``"(empty response)"``；非空则分别按 preview / detail 长度上限截断。

    Raises:
        无。
    """

    error_preview = error_body[:_DEFAULT_ERROR_BODY_PREVIEW_CHARS] if error_body else "(empty response)"
    error_detail = error_body[:_DEFAULT_ERROR_BODY_DETAIL_CHARS] if error_body else "(empty response)"
    return error_preview, error_detail


def _classify_non_retriable_error_type(status: int, error_body: str) -> str:
    """将不可重试 HTTP 状态码归类为稳定的 error_type 字符串。

    Args:
        status: HTTP 响应状态码。
        error_body: HTTP 错误响应的原始文本，用于 400 时检测 context_overflow。

    Returns:
        对应 ``error_event`` 的 ``error_type`` 字段文本；400 且检测到
        context_overflow 时返回 ``"context_overflow"``，其它按
        ``_NON_RETRIABLE_ERROR_TYPE_MAP`` 映射，缺省为 ``"client_error"``。

    Raises:
        无。
    """

    if status == 400 and _detect_context_overflow(error_body):
        return "context_overflow"
    return _NON_RETRIABLE_ERROR_TYPE_MAP.get(status, "client_error")


def _classify_retried_final_error_type(status: int) -> str:
    """将可重试 HTTP 状态码在重试耗尽后的 error_type 归类。

    Args:
        status: HTTP 响应状态码。

    Returns:
        429 归类为 ``"rate_limit_exceeded"``，其它归类为 ``"server_error"``。

    Raises:
        无。
    """

    return "rate_limit_exceeded" if status == 429 else "server_error"


@dataclass
class AsyncOpenAIRunnerRunningConfig:
    """
    AsyncOpenAIRunner 的运行时配置
    """
    debug_sse: bool = False
    debug_tool_delta: bool = False
    debug_sse_sample_rate: float = 1.0
    debug_sse_throttle_sec: float = 0.0
    tool_timeout_seconds: Optional[float] = None  # 工具执行超时（秒），None 表示使用默认值 90.0
    stream_idle_timeout: Optional[float] = None  # SSE 空闲读超时（秒），None 表示使用默认值 120.0
    stream_idle_heartbeat_sec: Optional[float] = None  # SSE 空闲心跳日志间隔（秒），None 表示使用默认值 10.0

class AsyncOpenAIRunner:
    """
    异步 OpenAI Compatible Runner，支持 streaming 和工具调用
    
    配置示例（llm_models.json）：
    {
      "deepseek_chat": {
        "runner_type": "openai_compatible",
        "endpoint_url": "https://api.deepseek.com/v1/chat/completions",
        "model": "deepseek-v4-flash",
        "temperature": 0.7,
        "headers": {
          "Authorization": "{{DEEPSEEK_API_KEY}}",
          "Content-Type": "application/json"
        },
        "timeout": 3600,
        "supports_stream": true,
        "supports_tool_calling": true
      }
    }
    """
    
    def __init__(
        self,
        *,
        endpoint_url: str,
        model: str,
        headers: Dict[str, str],
        name: Optional[str] = None,
        temperature: float = 0.7,
        default_extra_payloads: Optional[Dict[str, Any]] = None,
        timeout: int = _DEFAULT_REQUEST_TIMEOUT_SEC,
        max_retries: int = 3,
        supports_stream: bool = True,
        supports_tool_calling: bool = True,
        supports_stream_usage: bool = False,
        running_config: Optional[AsyncOpenAIRunnerRunningConfig] = None,
        cancellation_token: CancellationToken | None = None,
    ):
        """
        初始化 OpenAI 兼容 Runner。

        Args:
            endpoint_url: API 端点 URL
            model: 模型名称
            headers: HTTP 请求头（包含 Authorization）
            name: Runner 名称（用于日志，默认使用 model）
            temperature: 温度参数
            default_extra_payloads: 实例级默认请求参数，禁止覆盖 Runner 保留字段。
            timeout: 超时时间（秒）
            max_retries: 最大重试次数（网络/临时错误）
            supports_stream: 是否支持流式响应（默认 True）
            supports_tool_calling: 是否支持工具调用（默认 True）
            supports_stream_usage: 是否支持流式 usage 采集（默认 False）
            running_config: 运行时配置（包含 tool_timeout_seconds 等调试和运行参数）
            cancellation_token: 当前 runner 关联的取消令牌。

        Returns:
            None
        """
        if aiohttp is None:
            Log.error("aiohttp 库未安装，无法使用 AsyncOpenAIRunner", module=MODULE)
            raise ImportError("aiohttp is required for AsyncOpenAIRunner. Install with: pip install aiohttp")
        
        # 创建 running_config 实例（避免默认参数共享问题）
        if running_config is None:
            running_config = AsyncOpenAIRunnerRunningConfig()
        if running_config.tool_timeout_seconds is None:
            running_config.tool_timeout_seconds = _DEFAULT_TOOL_TIMEOUT_SEC
        if running_config.stream_idle_timeout is None:
            running_config.stream_idle_timeout = _DEFAULT_STREAM_IDLE_TIMEOUT_SEC
        if running_config.stream_idle_heartbeat_sec is None:
            running_config.stream_idle_heartbeat_sec = _DEFAULT_STREAM_IDLE_HEARTBEAT_SEC
        
        self.endpoint_url = endpoint_url
        self.model = model
        Log.verbose(f"初始化 AsyncOpenAIRunner: model={model}, endpoint_url={endpoint_url}", module=MODULE)
        self.name = name or model  # 默认使用 model 作为 name
        self.headers = headers
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.running_config = running_config
        self.tool_timeout_seconds = self.running_config.tool_timeout_seconds
        self.stream_idle_timeout = self.running_config.stream_idle_timeout
        self.stream_idle_heartbeat_sec = self.running_config.stream_idle_heartbeat_sec
        Log.verbose(
            "Runner 配置: "
            f"temperature={temperature}, timeout={timeout}, tool_timeout_seconds={self.tool_timeout_seconds}, "
            f"stream_idle_timeout={self.stream_idle_timeout}, "
            f"stream_idle_heartbeat_sec={self.stream_idle_heartbeat_sec}, max_retries={max_retries}",
            module=MODULE,
        )
        validated_default_payloads = dict(default_extra_payloads or {})
        _validate_extra_payload_keys(
            validated_default_payloads,
            source="default_extra_payloads",
            reserved_keys=_RESERVED_DEFAULT_EXTRA_PAYLOAD_KEYS,
        )
        self.default_extra_payloads = validated_default_payloads
        self.supports_stream = supports_stream
        self.supports_tool_calling = supports_tool_calling
        self.supports_stream_usage = supports_stream_usage
        self.cancellation_token = cancellation_token
        self._session: Optional[Any] = None
        self._tool_executor: Optional[ToolExecutor] = None

    def _raise_if_cancelled(self) -> None:
        """检查当前 runner 是否已收到取消信号。"""

        if self.cancellation_token is not None:
            self.cancellation_token.raise_if_cancelled()

    def _build_request_timeout(self, *, stream: bool) -> "ClientTimeout":
        """构造当前请求的 aiohttp timeout 配置。"""

        aiohttp_module = _require_aiohttp_module()
        return aiohttp_module.ClientTimeout(
            total=self.timeout,
            sock_read=self.stream_idle_timeout if stream else None,
        )

    def _ensure_session(self) -> "ClientSession":
        """确保当前 Runner 拥有可复用的 HTTP session。"""

        session = self._session
        if session is not None and not bool(getattr(session, "closed", False)):
            return session
        aiohttp_module = _require_aiohttp_module()
        new_session = aiohttp_module.ClientSession()
        self._session = new_session
        return new_session

    async def close(self) -> None:
        """关闭 Runner 级异步资源。"""

        session = self._session
        self._session = None
        if session is None:
            return
        if bool(getattr(session, "closed", False)):
            return
        await session.close()

    def _build_tool_execution_context(
        self,
        *,
        tool_call: Dict[str, Any],
        trace_meta: Dict[str, Any],
        cancellation_token: CancellationToken | None = None,
    ) -> ToolExecutionContext:
        """构造单次 tool call 的强类型执行上下文。

        Args:
            tool_call: 当前工具调用数据。
            trace_meta: 当前 trace 元数据。
            cancellation_token: 当前 tool call 绑定的取消令牌；为空时内部创建。

        Returns:
            规整后的工具执行上下文。

        Raises:
            无。
        """

        linked_token = cancellation_token
        if linked_token is None:
            if self.cancellation_token is not None:
                linked_token = CancellationToken.create_linked(self.cancellation_token)
            else:
                linked_token = CancellationToken()
        return ToolExecutionContext(
            run_id=str(trace_meta.get("run_id") or "").strip() or None,
            iteration_id=str(trace_meta.get("iteration_id") or "").strip() or None,
            tool_call_id=str(tool_call.get("id") or "").strip() or None,
            index_in_iteration=int(tool_call.get("index_in_iteration") or 0),
            timeout_seconds=self.tool_timeout_seconds,
            cancellation_token=linked_token,
        )

    def _tool_supports_execution_context(self, tool_name: str) -> bool:
        """判断指定工具是否已显式声明 execution context 注入。"""

        executor = self._tool_executor
        if executor is None:
            return False
        getter = getattr(executor, "get_execution_context_param_name", None)
        if not callable(getter):
            return False
        return bool(getter(tool_name))
    
    def set_default_extra_payloads(self, payloads: Dict[str, Any]) -> None:
        """
        设置实例级默认请求参数。

        Args:
            payloads: 额外请求参数字典，禁止覆盖 Runner 保留字段。

        Returns:
            None

        Raises:
            ValueError: 当参数包含 Runner 保留字段时抛出。
        """
        _validate_extra_payload_keys(
            payloads,
            source="default_extra_payloads",
            reserved_keys=_RESERVED_DEFAULT_EXTRA_PAYLOAD_KEYS,
        )
        Log.debug(f"[{self.name}] 设置实例级默认请求参数: {payloads}", module=MODULE)
        self.default_extra_payloads = dict(payloads or {})

    def set_tools(self, executor: Optional[ToolExecutor]) -> None:
        """
        设置工具执行器（工具定义从 executor.get_schemas() 获取）。

        Args:
            executor: 工具执行器实例；传 None 表示清空工具能力。

        Returns:
            None
        """
        self._tool_executor = executor

    def is_supports_tool_calling(self) -> bool:
        """
        返回是否支持工具调用。

        Args:
            None

        Returns:
            是否支持工具调用。
        """
        return self.supports_tool_calling

    async def _run_tool_call(
        self,
        tool_call: Dict[str, Any],
        request_id: str,
        trace_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        执行单个工具调用（支持并发与超时）。

        Args:
            tool_call: 标准化的工具调用数据。

        Returns:
            包含工具调用与结构化结果的字典。
        """
        start_time = time.monotonic()
        log_prefix = f"[{self.name}][{request_id}]"
        Log.debug(
            f"{log_prefix} ▶ {tool_call['name']}({_compact_args(tool_call['arguments'])}) "
            f"[id={tool_call.get('id')}, index={tool_call.get('index_in_iteration')}]",
            module=MODULE,
        )
        tool_cancellation_token = (
            CancellationToken.create_linked(self.cancellation_token)
            if self.cancellation_token is not None
            else CancellationToken()
        )
        try:
            tool_executor = _require_tool_executor(self._tool_executor)
            context = self._build_tool_execution_context(
                tool_call=tool_call,
                trace_meta=trace_meta,
                cancellation_token=tool_cancellation_token,
            )
            exec_coro = asyncio.to_thread(
                tool_executor.execute,
                name=tool_call["name"],
                arguments=tool_call["arguments"],
                context=context,
            )
            if self.tool_timeout_seconds is not None:
                tool_result = await asyncio.wait_for(exec_coro, timeout=self.tool_timeout_seconds)
            else:
                tool_result = await exec_coro
        except asyncio.TimeoutError:
            tool_cancellation_token.cancel()
            supports_execution_context = self._tool_supports_execution_context(tool_call["name"])
            timeout_meta: dict[str, Any] = {
                "execution_may_continue": not supports_execution_context,
            }
            timeout_hint = "tool execution timed out and cancellation was requested"
            timeout_log_suffix = "已触发协作式取消"
            if not supports_execution_context:
                timeout_meta["orphan_thread_warning"] = True
                timeout_hint = "tool execution may still be running; do not blindly retry"
                timeout_log_suffix = "后台线程可能仍在运行"
            tool_result = build_error(
                "tool_execution_timeout",
                "tool execution timeout",
                hint=timeout_hint,
                meta=timeout_meta,
            )
            Log.warn(
                f"{log_prefix} ⚠ 工具 {tool_call['name']} 执行超时（{self.tool_timeout_seconds}s），{timeout_log_suffix}",
                module=MODULE,
            )
        except asyncio.CancelledError:
            tool_cancellation_token.cancel()
            tool_result = build_error("cancelled", "tool execution cancelled")
        except Exception as exc:
            tool_result = build_error("execution_error", str(exc))
        latency_ms = int((time.monotonic() - start_time) * 1000)
        contract_error = validate_tool_result_contract(tool_result)
        if contract_error is not None:
            tool_result = build_error(
                "invalid_result",
                contract_error,
                hint=str(tool_result),
            )

        meta = tool_result.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta["tool"] = tool_call["name"]
        meta["latency_ms"] = latency_ms
        tool_result["meta"] = meta
        success = is_tool_success(tool_result)
        summary = _result_summary(tool_result)
        if success:
            # 仅在成功时计算响应体积，用于日志显示
            result_size = len(json.dumps(tool_result, ensure_ascii=False, default=str))
            Log.debug(
                f"{log_prefix} ✅ {tool_call['name']} → {summary} ({latency_ms}ms, {result_size}c)",
                module=MODULE,
            )
        else:
            Log.debug(
                f"{log_prefix} ❌ {tool_call['name']} → {summary} ({latency_ms}ms)",
                module=MODULE,
            )
        return {
            "id": tool_call["id"],
            "name": tool_call["name"],
            "arguments": tool_call["arguments"],
            "index_in_iteration": tool_call["index_in_iteration"],
            "result": tool_result,
        }

    async def _emit_tool_batch(
        self,
        tool_calls: List[Dict[str, Any]],
        request_id: str,
        trace_meta: Dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """
        发起并等待一批工具调用，按顺序产出事件（不做参数校验，依赖上游保证完整性）。

        Args:
            tool_calls: 工具调用列表。

        Returns:
            事件异步迭代器。
        """
        log_prefix = f"[{self.name}][{request_id}]"
        if not tool_calls:
            Log.debug(f"{log_prefix} 未检测到工具调用，跳过执行", module=MODULE)
            return
        if not self._tool_executor:
            Log.error(
                f"{log_prefix} 收到工具调用但未设置工具执行器，无法执行"
                f"（error_type=tool_executor_missing）",
                module=MODULE,
            )
            yield self._annotate_event(error_event(
                "tool_calls received but tool executor is not set",
                recoverable=False,
                error_type="tool_executor_missing",
            ), trace_meta)
            return

        ordered_calls = sorted(tool_calls, key=lambda tc: tc["index_in_iteration"])
        # Log.debug(f"{log_prefix} 准备执行 {len(ordered_calls)} 个工具调用", module=MODULE)
        call_ids = [tc["id"] for tc in ordered_calls]

        for tc in ordered_calls:
            display_name, summary_params = (
                self._tool_executor.get_tool_display_info(tc["name"])
                if self._tool_executor else (tc["name"], None)
            )
            event = tool_call_dispatched(         # ← 工具调用已发起事件
                tool_call_id=tc["id"],
                name=tc["name"],
                arguments=tc["arguments"],
                index_in_iteration=tc["index_in_iteration"],
                display_name=display_name,
                summary_params=summary_params,
            )
            yield self._annotate_event(event, trace_meta)

        yield self._annotate_event(tool_calls_batch_ready(call_ids), trace_meta)  # ← 批次工具调用就绪事件

        tasks = [
            asyncio.create_task(self._run_tool_call(tc, request_id, trace_meta))
            for tc in ordered_calls
        ]
        if not tasks:
            results: list[Dict[str, Any]] = []
        else:
            # 取消信号到达时，立刻取消未完成的工具任务并退出，
            # 不再等待 tool_timeout_seconds 触发逐个工具的超时回收。
            waiter, unregister_waiter = _create_cancellation_waiter(self.cancellation_token)
            try:
                if waiter is None:
                    results = await asyncio.gather(*tasks)
                else:
                    gather_task = asyncio.ensure_future(asyncio.gather(*tasks))
                    try:
                        done, _pending = await asyncio.wait(
                            {gather_task, waiter},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if (
                            waiter in done
                            and self.cancellation_token is not None
                            and self.cancellation_token.is_cancelled()
                            and not gather_task.done()
                        ):
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            await cancel_task_and_wait(gather_task)
                            Log.info(
                                f"{log_prefix} 工具批次因取消信号提前中止: {len(ordered_calls)} 个待完成",
                                module=MODULE,
                            )
                            raise EngineCancelledError("tool batch cancelled")
                        results = await gather_task
                    except asyncio.CancelledError:
                        await cancel_task_and_wait(gather_task)
                        raise
            finally:
                if unregister_waiter is not None:
                    unregister_waiter()
                if waiter is not None and not waiter.done():
                    waiter.cancel()

        counts = {"ok": 0, "error": 0, "timeout": 0, "cancelled": 0}
        for result in results:
            tool_result = result["result"]
            success = is_tool_success(tool_result)
            if success:
                counts["ok"] += 1
            else:
                code = get_error_code(tool_result)
                if code == "tool_execution_timeout":
                    counts["timeout"] += 1
                elif code == "cancelled":
                    counts["cancelled"] += 1
                else:
                    counts["error"] += 1
            result_display_name, _ = (
                self._tool_executor.get_tool_display_info(result["name"])
                if self._tool_executor else (result["name"], None)
            )
            event = tool_call_result(             # ← 工具调用结果事件
                tool_call_id=result["id"],
                name=result["name"],
                arguments=result["arguments"],
                index_in_iteration=result["index_in_iteration"],
                result=result["result"],
                display_name=result_display_name,
            )
            yield self._annotate_event(event, trace_meta)

        event = tool_calls_batch_done(            # ← 工具调用批次完成事件
            call_ids,
            ok=counts["ok"],
            error=counts["error"],
            timeout=counts["timeout"],
            cancelled=counts["cancelled"],
        )
        yield self._annotate_event(event, trace_meta)

    async def call(
        self,
        messages: List[AgentMessage],
        *,
        stream: bool = True,
        **extra_payloads,
    ) -> AsyncIterator[StreamEvent]:
        """
        调用 OpenAI 兼容 API 并返回 streaming 事件流
        
        支持自动重试：
        - 网络超时 → 重试
        - 50x 服务器错误 → 重试
        - 429 限流 → 根据 Retry-After 等待后重试
        - 40x 配置错误 → 不重试，立即返回
        
        Args:
            messages: 消息列表（OpenAI 格式）
            stream: 是否启用 streaming
            **extra_payloads: 额外的请求参数，禁止覆盖 Runner 保留字段。
        
        Returns:
            事件异步迭代器。

        Raises:
            ValueError: 当额外参数包含 Runner 保留字段时抛出。
        """
        _validate_extra_payload_keys(
            extra_payloads,
            source="extra_payloads",
            reserved_keys=_RESERVED_EXTRA_PAYLOAD_KEYS,
        )
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        trace_context = extra_payloads.pop("trace_context", None)
        if not isinstance(trace_context, dict):
            trace_context = {}
        run_id = trace_context.get("run_id") or f"run_{uuid.uuid4().hex[:8]}"
        iteration_id = trace_context.get("iteration_id") or f"{run_id}_iteration_{request_id}"
        trace_meta = {
            "run_id": run_id,
            "iteration_id": iteration_id,
            "request_id": request_id,
        }
        log_prefix = f"[{self.name}][{request_id}]"
        cancellation_waiter, unregister_cancellation_waiter = _create_cancellation_waiter(self.cancellation_token)

        # 配置级检查：模型不支持 streaming 时自动降级
        if stream and not self.supports_stream:
            Log.warn(f"{log_prefix} 模型配置不支持流式，自动降级为非流式模式", module=MODULE)
            stream = False
        
        # 两层合并 extra payloads：实例默认 < 调用参数
        merged_extra_payloads = {}
        merged_extra_payloads.update(self.default_extra_payloads)
        if extra_payloads:
            Log.debug(f"{log_prefix} 传入调用级别的额外请求参数: {extra_payloads}", module=MODULE)
            merged_extra_payloads.update(extra_payloads)
        
        # 1. 构建请求 payload
        payload = {
            "model": self.model,
            # messages 允许透传兼容 provider 的 assistant 扩展字段；当前已知依赖方
            # 包括 DeepSeek thinking tool loop、MiMo thinking tool loop，以及开启
            # preserve_thinking 的 Qwen。不要在这里擅自清洗 reasoning_content。
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
            **merged_extra_payloads,
        }

        # 流式 usage 采集：通过 stream_options 让服务端在最后一个 chunk 返回 usage
        if stream and self.supports_stream_usage:
            payload.setdefault("stream_options", {})["include_usage"] = True
        
        # 添加工具定义（如果有且模型支持）
        if self._tool_executor and self.supports_tool_calling:
            tools = self._tool_executor.get_schemas()
            if tools:
                payload["tools"] = [self._tool_to_openai_spec(t) for t in tools]
        elif self._tool_executor and not self.supports_tool_calling:
            Log.warn(f"{log_prefix} 模型配置不支持工具调用，已忽略工具定义", module=MODULE)
        
        # 检查 n 参数（多候选回答）
        n = payload.get("n", 1)
        if n > 1:
            Log.warn(
                f"{log_prefix} 检测到 'n' 参数为 {n}，已覆盖为 1，因 Runner 仅处理 choices[0]。"
                f"如需多候选回答，请在上层多次调用 run()。",
                module=MODULE,
            )
            payload["n"] = 1  # 强制覆盖为 1
        
        # 2. 重试循环（复用同一 Runner 实例的 session，利用连接池）
        aiohttp_module = _require_aiohttp_module()
        request_timeout = self._build_request_timeout(stream=stream)
        session = self._ensure_session()
        try:
            for attempt in range(self.max_retries + 1):
                self._raise_if_cancelled()
                post_context: Any = None
                response: Any = None
                response_entered = False
                try:
                    Log.debug(
                        f"{log_prefix} 发送请求到 {self.endpoint_url}（第 {attempt + 1} 次尝试）",
                        module=MODULE,
                    )
                    post_context = session.post(
                        self.endpoint_url,
                        json=payload,
                        headers=self.headers,
                        timeout=request_timeout,
                    )
                    response = await _await_or_cancel(
                        post_context.__aenter__(),
                        operation_name="http_request_enter",
                        cancellation_waiter=cancellation_waiter,
                        cancellation_token=self.cancellation_token,
                        raise_if_cancelled=self._raise_if_cancelled,
                        log_prefix=f"[{self.name}]",
                        log_module=MODULE,
                    )
                    response_entered = True

                    if response.status == 200:
                        Log.debug(f"{log_prefix} HTTP 200 成功（第 {attempt + 1} 次尝试）", module=MODULE)
                        content_type = response.headers.get("Content-Type", "")

                        if "text/event-stream" in content_type:
                            if not stream:
                                Log.debug(
                                    f"{log_prefix} 请求非流式模式但服务端返回 SSE 格式（Content-Type: {content_type}）",
                                    module=MODULE,
                                )
                            async for event in self._process_sse_stream(response, request_id, trace_meta):
                                yield event
                        elif "application/json" in content_type or not stream:
                            if stream and "application/json" in content_type:
                                Log.debug(
                                    f"{log_prefix} 请求流式模式但服务端返回 JSON 格式（Content-Type: {content_type}）",
                                    module=MODULE,
                                )
                            result = await _await_or_cancel(
                                response.json(),
                                operation_name="response_json",
                                cancellation_waiter=cancellation_waiter,
                                cancellation_token=self.cancellation_token,
                                raise_if_cancelled=self._raise_if_cancelled,
                                log_prefix=f"[{self.name}]",
                                log_module=MODULE,
                            )
                            async for event in self._process_non_stream(result, request_id, trace_meta):
                                yield event
                        else:
                            if stream:
                                async for event in self._process_sse_stream(response, request_id, trace_meta):
                                    yield event
                            else:
                                result = await _await_or_cancel(
                                    response.json(),
                                    operation_name="response_json_unknown_content_type",
                                    cancellation_waiter=cancellation_waiter,
                                    cancellation_token=self.cancellation_token,
                                    raise_if_cancelled=self._raise_if_cancelled,
                                    log_prefix=f"[{self.name}]",
                                    log_module=MODULE,
                                )
                                async for event in self._process_non_stream(result, request_id, trace_meta):
                                    yield event
                        return

                    error_body = await _await_or_cancel(
                        response.text(),
                        operation_name="response_error_body",
                        cancellation_waiter=cancellation_waiter,
                        cancellation_token=self.cancellation_token,
                        raise_if_cancelled=self._raise_if_cancelled,
                        log_prefix=f"[{self.name}]",
                        log_module=MODULE,
                    )
                    error_preview, error_detail = _extract_error_body_slices(error_body)

                    if response.status in NON_RETRIABLE_STATUS_CODES:
                        error_type = _classify_non_retriable_error_type(response.status, error_body)

                        Log.warn(
                            f"{log_prefix} HTTP {response.status}（不可重试，error_type={error_type}）: "
                            f"{error_preview}",
                            module=MODULE,
                        )
                        yield self._annotate_event(error_event(
                            f"HTTP {response.status}: {error_detail}",
                            recoverable=False,
                            error_type=error_type,
                            status=response.status,
                            body=error_body,
                        ), trace_meta)
                        return

                    if response.status in RETRIABLE_STATUS_CODES:
                        if attempt < self.max_retries:
                            delay = self._calculate_backoff(attempt, response)
                            Log.info(
                                f"{log_prefix} HTTP {response.status}，{delay:.1f}s 后重试 "
                                f"({attempt + 1}/{self.max_retries})",
                                module=MODULE,
                            )
                            yield self._annotate_event(warning_event(
                                f"HTTP {response.status}, retry {attempt + 1}/{self.max_retries} in {delay:.1f}s"
                            ), trace_meta)
                            await _await_or_cancel(
                                asyncio.sleep(delay),
                                operation_name="retry_backoff_after_http_error",
                                cancellation_waiter=cancellation_waiter,
                                cancellation_token=self.cancellation_token,
                                raise_if_cancelled=self._raise_if_cancelled,
                                log_prefix=f"[{self.name}]",
                                log_module=MODULE,
                            )
                            continue

                        error_type = _classify_retried_final_error_type(response.status)
                        Log.error(
                            f"{log_prefix} HTTP {response.status} 重试 {self.max_retries} 次后仍失败"
                            f"（error_type={error_type}）",
                            module=MODULE,
                        )
                        yield self._annotate_event(error_event(
                            f"HTTP {response.status} after {self.max_retries} retries: {error_detail}",
                            recoverable=False,
                            error_type=error_type,
                            status=response.status,
                            body=error_body,
                        ), trace_meta)
                        return

                    Log.warn(
                        f"{log_prefix} HTTP {response.status}（未知状态码，error_type=unknown_http_status）: "
                        f"{error_preview}",
                        module=MODULE,
                    )
                    yield self._annotate_event(error_event(
                        f"HTTP {response.status}: {error_detail}",
                        recoverable=False,
                        error_type="unknown_http_status",
                        status=response.status,
                        body=error_body,
                    ), trace_meta)
                    return

                except EngineCancelledError:
                    raise
                except asyncio.TimeoutError:
                    if attempt < self.max_retries:
                        delay = 2 ** attempt
                        Log.info(
                            f"{log_prefix} 请求超时，{delay}s 后重试（{attempt + 1}/{self.max_retries}）",
                            module=MODULE,
                        )
                        yield self._annotate_event(
                            warning_event(f"Timeout, retry {attempt + 1}/{self.max_retries} in {delay}s"),
                            trace_meta,
                        )
                        await _await_or_cancel(
                            asyncio.sleep(delay),
                            operation_name="retry_backoff_after_timeout",
                            cancellation_waiter=cancellation_waiter,
                            cancellation_token=self.cancellation_token,
                            raise_if_cancelled=self._raise_if_cancelled,
                            log_prefix=f"[{self.name}]",
                            log_module=MODULE,
                        )
                        continue

                    Log.error(
                        f"{log_prefix} 请求超时，重试 {self.max_retries} 次后仍失败（error_type=timeout）",
                        module=MODULE,
                    )
                    yield self._annotate_event(error_event(
                        f"Request timeout after {self.max_retries} retries ({self.timeout}s each)",
                        recoverable=False,
                        error_type="timeout",
                    ), trace_meta)
                    return

                except aiohttp_module.ClientError as exc:
                    if attempt < self.max_retries:
                        delay = 2 ** attempt
                        Log.info(
                            f"{log_prefix} 网络错误: {exc}，{delay}s 后重试（{attempt + 1}/{self.max_retries}）",
                            module=MODULE,
                        )
                        yield self._annotate_event(
                            warning_event(f"Network error, retry {attempt + 1}/{self.max_retries} in {delay}s"),
                            trace_meta,
                        )
                        await _await_or_cancel(
                            asyncio.sleep(delay),
                            operation_name="retry_backoff_after_network_error",
                            cancellation_waiter=cancellation_waiter,
                            cancellation_token=self.cancellation_token,
                            raise_if_cancelled=self._raise_if_cancelled,
                            log_prefix=f"[{self.name}]",
                            log_module=MODULE,
                        )
                        continue

                    Log.error(
                        f"{log_prefix} 网络错误，重试 {self.max_retries} 次后仍失败: {exc}"
                        f"（error_type=network_error）",
                        module=MODULE,
                    )
                    yield self._annotate_event(error_event(
                        f"Network error after {self.max_retries} retries: {exc}",
                        exception=exc,
                        recoverable=False,
                        error_type="network_error",
                    ), trace_meta)
                    return

                except Exception as exc:
                    Log.error(
                        f"{log_prefix} 未知异常: {exc}（error_type=unknown_error）",
                        exc_info=True,
                        module=MODULE,
                    )
                    yield self._annotate_event(error_event(
                        f"Unexpected error: {exc}",
                        exception=exc,
                        recoverable=False,
                        error_type="unknown_error",
                    ), trace_meta)
                    return
                finally:
                    if response_entered and post_context is not None:
                        try:
                            await post_context.__aexit__(None, None, None)
                        except (EngineCancelledError, asyncio.CancelledError):
                            # 取消语义必须穿透 cleanup，不能被吞；协作式取消依赖这条透传。
                            raise
                        except Exception as cleanup_exc:
                            # aiohttp release 失败 = 连接可能未归还或被归还为脏连接；
                            # 1) 把 cleanup 异常降级为 warn，避免覆盖已 yield 的业务事件与真正的业务异常；
                            # 2) 废弃当前 Runner 级 session，强制下一轮 / 下一次 call() 重建，防止
                            #    后续请求继续复用受污染的连接池。
                            Log.warn(
                                f"{log_prefix} HTTP 响应上下文释放异常"
                                f"（attempt={attempt + 1}，连接可能未归还，作废当前 HTTP session）: {cleanup_exc}",
                                module=MODULE,
                            )
                            stale_session = self._session
                            self._session = None
                            if stale_session is not None:
                                try:
                                    if not bool(getattr(stale_session, "closed", False)):
                                        await stale_session.close()
                                except (EngineCancelledError, asyncio.CancelledError):
                                    raise
                                except Exception as close_exc:
                                    # 脏 session 关闭再失败属于 aiohttp 内部问题，这里只记日志、不向上传播，
                                    # 让 Runner 继续使用后续新建的 session。
                                    Log.warn(
                                        f"{log_prefix} 作废 HTTP session 时关闭失败（忽略并丢弃）: {close_exc}",
                                        module=MODULE,
                                    )
                            # 若外层决策是 continue 去重试，下一轮 session.post(...) 必须看到新 session，
                            # 所以同步更新循环内的局部 session 绑定。
                            session = self._ensure_session()
        finally:
            if unregister_cancellation_waiter is not None:
                unregister_cancellation_waiter()
            if cancellation_waiter is not None and not cancellation_waiter.done():
                cancellation_waiter.cancel()
    
    async def _process_sse_stream(
        self,
        response: "ClientResponse",
        request_id: str,
        trace_meta: Dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """
        处理 SSE (Server-Sent Events) streaming 响应。

        委托 SSEStreamParser 完成行缓冲与 payload 解析，Runner 负责：
        - 事件注解（trace_meta）
        - 工具调用执行（_emit_tool_batch）
        - 完成/错误事件产出

        Args:
            response: aiohttp 响应对象。
            request_id: 请求 ID。
            trace_meta: 追踪元数据（run_id、iteration_id、request_id 等）。

        Returns:
            事件异步迭代器。
        """
        log_prefix = f"[{self.name}][{request_id}]"

        # 1) 使用 SSEStreamParser 解析流，透传增量事件
        parser = SSEStreamParser(
            name=self.name,
            request_id=request_id,
            running_config=self.running_config,
            cancellation_token=self.cancellation_token,
        )
        async for event in parser.parse_stream(response):
            yield self._annotate_event(event, trace_meta)

        self._raise_if_cancelled()

        # 2) 获取解析结果
        result = parser.get_result()
        full_content = result.content
        raw_tool_call_count = result.raw_tool_call_count

        if result.protocol_errors:
            error_item = result.protocol_errors[0]
            Log.error(
                f"{log_prefix} SSE 协议错误: {error_item.get('message', '')}"
                f"（error_type={error_item.get('error_type', 'response_error')}）",
                module=MODULE,
            )
            # 保持事件契约：即使错误路径也先发 content_complete 再发 error
            yield self._annotate_event(content_complete(full_content), trace_meta)
            yield self._annotate_event(
                error_event(
                    error_item.get("message", "Invalid SSE payload"),
                    recoverable=False,
                    error_type=error_item.get("error_type", "response_error"),
                    body=error_item.get("body"),
                ),
                trace_meta,
            )
            return

        # 3) 无 [DONE] 标记时警告
        if not result.done_received:
            Log.warn(f"{log_prefix} SSE 流结束但未收到 [DONE] 标记", module=MODULE)

        # 4) 校验输出：既没有内容也没有工具调用
        if not full_content and raw_tool_call_count == 0:
            if result.stream_state.get("saw_choice"):
                # 收到 choices 但 delta 为空（正常情况，例如 content:null）
                yield self._annotate_event(content_complete(full_content), trace_meta)
                empty_summary: Dict[str, Any] = {
                    "total_chars": len(full_content),
                    "tool_calls": 0,
                    "truncated": False,
                }
                if result.usage:
                    empty_summary["usage"] = result.usage
                yield self._annotate_event(done_event(summary=empty_summary), trace_meta)
                return
            Log.error(
                f"{log_prefix} SSE 流结束但没有任何有效输出（choices 字段始终为空）"
                f"（error_type=response_error）",
                module=MODULE,
            )
            yield self._annotate_event(error_event(
                "No valid output in SSE stream (all choices were empty)",
                recoverable=False,
                error_type="response_error",
            ), trace_meta)
            return

        # 5) 执行工具调用（如果有）
        if raw_tool_call_count > 0:
            if not result.stream_state.get("tool_calls_finished") and not result.done_received:
                Log.warn(f"{log_prefix} 工具调用结束但未收到 finish_reason 或 [DONE]", module=MODULE)

            if result.validation_errors:
                Log.error(
                    f"{log_prefix} 工具调用参数不完整或无效: {result.validation_errors}"
                    f"（error_type=tool_call_incomplete）",
                    module=MODULE,
                )
                # 保持事件契约：先发 content_complete 再发 error
                yield self._annotate_event(content_complete(full_content), trace_meta)
                yield self._annotate_event(error_event(
                    "Tool call arguments incomplete or invalid",
                    recoverable=False,
                    error_type="tool_call_incomplete",
                    body=json.dumps(result.validation_errors, ensure_ascii=False),
                ), trace_meta)
                return

            Log.debug(
                f"{log_prefix} 已聚合 {len(result.tool_calls)} 个工具调用，准备执行",
                module=MODULE,
            )

            async for event in self._emit_tool_batch(result.tool_calls, request_id, trace_meta):
                yield event
                if event.type == EventType.ERROR:
                    return

        # 6) 发送内容完成事件
        cc_kwargs: Dict[str, Any] = {}
        if result.reasoning_content:
            cc_kwargs["reasoning_content"] = result.reasoning_content
        yield self._annotate_event(content_complete(full_content, **cc_kwargs), trace_meta)

        # 7) 构建并发送 DONE 事件（含 usage）
        done_summary: Dict[str, Any] = {
            "total_chars": len(full_content),
            "tool_calls": len(result.tool_calls),
            "truncated": result.stream_state.get("finish_reason") == "length",
            "content_filtered": bool(result.stream_state.get("content_filtered", False)),
            "finish_reason": result.stream_state.get("finish_reason"),
        }
        if result.usage:
            done_summary["usage"] = result.usage
        yield self._annotate_event(done_event(summary=done_summary), trace_meta)

        # 8) 发送 token 遥测元数据事件（便于 Agent 维护预算状态）
        if result.usage:
            yield self._annotate_event(
                metadata_event("token_usage_summary", {
                    "prompt_tokens": result.usage.get("prompt_tokens", 0),
                    "completion_tokens": result.usage.get("completion_tokens", 0),
                    "total_tokens": result.usage.get("total_tokens", 0),
                    "cached_tokens": result.usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    if isinstance(result.usage.get("prompt_tokens_details"), dict) else 0,
                    "reasoning_tokens": result.usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                    if isinstance(result.usage.get("completion_tokens_details"), dict) else 0,
                }),
                trace_meta,
            )

    def _build_non_stream_error_event(
        self,
        *,
        trace_meta: Dict[str, Any],
        log_message: str,
        message: str,
        error_type: str,
        body: str | None = None,
    ) -> StreamEvent:
        """构造非流式响应的单个错误事件。

        参数:
            trace_meta: 当前 trace 元数据。
            log_message: 需要同步写入日志的错误文本。
            message: 返回给上游的错误摘要。
            error_type: 错误类型标签。
            body: 可选错误详情。

        返回值:
            已附带 trace 元数据的错误事件。

        异常:
            无。
        """

        Log.error(log_message, module=MODULE)
        return self._annotate_event(
            error_event(
                message,
                recoverable=False,
                error_type=error_type,
                body=body,
            ),
            trace_meta,
        )

    async def _process_non_stream(
        self,
        result: Dict,
        request_id: str,
        trace_meta: Dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """
        处理非 streaming 响应（完整 JSON）。

        Args:
            result: JSON 响应体。

        Returns:
            事件异步迭代器。
        """
        log_prefix = f"[{self.name}][{request_id}]"
        # 模型没有返回choices，这意味着服务端没有生成任何内容
        choices = result.get("choices", [])
        if not choices:
            yield self._build_non_stream_error_event(
                trace_meta=trace_meta,
                log_message=f"{log_prefix} 响应缺少 choices 字段，无法生成内容（error_type=response_error）",
                message="No choices in response",
                error_type="response_error",
            )
            return
        
        # 只处理第一个回答（如果 n > 1，其余候选回答会被忽略，请求时已警告）
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            yield self._build_non_stream_error_event(
                trace_meta=trace_meta,
                log_message=f"{log_prefix} choices[0] 非 dict（实际类型 {type(first_choice).__name__}），无法解析（error_type=response_error）",
                message="Invalid choices[0] structure",
                error_type="response_error",
            )
            return
        message = first_choice.get("message", {})
        if not isinstance(message, dict):
            yield self._build_non_stream_error_event(
                trace_meta=trace_meta,
                log_message=f"{log_prefix} choices[0].message 非 dict（实际类型 {type(message).__name__}），无法解析（error_type=response_error）",
                message="Invalid choices[0].message structure",
                error_type="response_error",
            )
            return
        
        # 内容
        content = message.get("content")
        if content is None:
            content = ""

        # 推理内容（thinking 模式思维链）
        reasoning_content_text = message.get("reasoning_content") or ""

        if reasoning_content_text:
            yield self._annotate_event(reasoning_delta(reasoning_content_text), trace_meta)

        if content:
            # 有别于流式响应，这里一次性返回完整内容
            yield self._annotate_event(content_delta(content), trace_meta)        # ← 内容增量事件
        
        # 工具调用
        # NOTE: legacy function_call is not supported; only tool_calls are processed.
        # NOTE: code review: dot NOT check for function_call support here.
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            tool_calls = []
        elif not isinstance(tool_calls, list):
            try:
                body = json.dumps({"tool_calls": tool_calls}, ensure_ascii=False, default=str)
            except TypeError:
                body = json.dumps({"tool_calls": str(tool_calls)}, ensure_ascii=False)
            yield self._build_non_stream_error_event(
                trace_meta=trace_meta,
                log_message=f"{log_prefix} tool_calls 字段不是列表，无法处理（error_type=tool_call_invalid）",
                message="tool_calls must be a list",
                error_type="tool_call_invalid",
                body=body,
            )
            return
        if tool_calls:
            Log.debug(f"{log_prefix} 非流式响应包含 {len(tool_calls)} 个工具调用", module=MODULE)
            tool_call_batch = []
            for idx, tc in enumerate(tool_calls):
                if not isinstance(tc, dict):
                    yield self._build_non_stream_error_event(
                        trace_meta=trace_meta,
                        log_message=f"{log_prefix} 工具调用条目不是对象: index={idx}（error_type=tool_call_incomplete）",
                        message="Tool call arguments incomplete or invalid",
                        error_type="tool_call_incomplete",
                        body=json.dumps([f"tool_index {idx}: tool call is not object"], ensure_ascii=False),
                    )
                    return

                # id 必须为非空字符串；null / 非字符串 / 空串 统一归一化为 "" 由下方校验拦截
                raw_tc_id = tc.get("id")
                tc_id = raw_tc_id if isinstance(raw_tc_id, str) else ""
                func = tc.get("function", {})
                if not isinstance(func, dict):
                    yield self._build_non_stream_error_event(
                        trace_meta=trace_meta,
                        log_message=f"{log_prefix} 工具调用缺少 function 对象: index={idx}（error_type=tool_call_incomplete）",
                        message="Tool call arguments incomplete or invalid",
                        error_type="tool_call_incomplete",
                        body=json.dumps([f"tool_index {idx}: missing function object"], ensure_ascii=False),
                    )
                    return
                name = func.get("name", "")
                arguments = func.get("arguments", "")

                if not tc_id or not name:
                    yield self._build_non_stream_error_event(
                        trace_meta=trace_meta,
                        log_message=f"{log_prefix} 工具调用缺少 id 或 name: index={idx}（error_type=tool_call_incomplete）",
                        message="Tool call arguments incomplete or invalid",
                        error_type="tool_call_incomplete",
                        body=json.dumps([f"tool_index {idx}: missing id or name"], ensure_ascii=False),
                    )
                    return

                if isinstance(arguments, dict):
                    args_obj = arguments
                elif isinstance(arguments, str):
                    try:
                        args_obj = json.loads(arguments)
                    except json.JSONDecodeError as exc:
                        yield self._build_non_stream_error_event(
                            trace_meta=trace_meta,
                            log_message=(
                                f"{log_prefix} 工具调用参数 JSON 无效: index={idx}, error={exc}"
                                "（error_type=tool_call_incomplete）"
                            ),
                            message="Tool call arguments incomplete or invalid",
                            error_type="tool_call_incomplete",
                            body=json.dumps([f"tool_index {idx}: invalid arguments JSON ({exc})"], ensure_ascii=False),
                        )
                        return
                else:
                    yield self._build_non_stream_error_event(
                        trace_meta=trace_meta,
                        log_message=(
                            f"{log_prefix} 工具调用参数类型非法: index={idx}, type={type(arguments).__name__}"
                            "（error_type=tool_call_incomplete）"
                        ),
                        message="Tool call arguments incomplete or invalid",
                        error_type="tool_call_incomplete",
                        body=json.dumps(
                            [f"tool_index {idx}: arguments type is {type(arguments).__name__}"],
                            ensure_ascii=False,
                        ),
                    )
                    return
                if not isinstance(args_obj, dict):
                    yield self._build_non_stream_error_event(
                        trace_meta=trace_meta,
                        log_message=f"{log_prefix} 工具调用参数不是对象: index={idx}（error_type=tool_call_incomplete）",
                        message="Tool call arguments incomplete or invalid",
                        error_type="tool_call_incomplete",
                        body=json.dumps([f"tool_index {idx}: arguments is not object"], ensure_ascii=False),
                    )
                    return
                
                # 有别于流式响应，这里一次性返回完整参数
                yield self._annotate_event(
                    tool_call_start(tool_name=name, tool_call_id=tc_id),
                    trace_meta,
                )   # ← 工具调用开始事件
                tool_call_batch.append({
                    "id": tc_id,
                    "name": name,
                    "arguments": args_obj,
                    "index_in_iteration": idx,
                })
            
            async for event in self._emit_tool_batch(tool_call_batch, request_id, trace_meta):
                yield event
                if event.type == EventType.ERROR:
                    return
        
        # 总是 yield content_complete，即使内容为空（保持与 SSE 流式一致）
        cc_kwargs_ns: Dict[str, Any] = {}
        if reasoning_content_text:
            cc_kwargs_ns["reasoning_content"] = reasoning_content_text
        yield self._annotate_event(content_complete(content, **cc_kwargs_ns), trace_meta)         # ← 内容完成事件
        # 非流式 finish_reason 截断检测
        non_stream_finish = choices[0].get("finish_reason")
        truncated = non_stream_finish == "length"
        content_filtered = non_stream_finish == "content_filter"
        if truncated:
            Log.warn(
                f"{log_prefix} 输出被截断 (finish_reason=length)，"
                f"模型可能因上下文窗口不足而无法生成完整内容",
                module=MODULE,
            )
        if content_filtered:
            Log.warn(
                f"{log_prefix} 输出命中内容过滤 (finish_reason=content_filter)，"
                f"本轮结果将标记为 filtered",
                module=MODULE,
            )
        # 读取非流式 usage
        non_stream_usage = result.get("usage")
        if non_stream_usage and not isinstance(non_stream_usage, dict):
            non_stream_usage = None
        # 构建并发送 DONE 事件（含 usage）
        done_summary: Dict[str, Any] = {
            "total_chars": len(content),
            "tool_calls": len(tool_calls),
            "truncated": truncated,
            "content_filtered": content_filtered,
            "finish_reason": non_stream_finish,
        }
        if non_stream_usage:
            done_summary["usage"] = non_stream_usage
        yield self._annotate_event(done_event(summary=done_summary), trace_meta)

        # 发送 token 遥测元数据事件
        if non_stream_usage:
            yield self._annotate_event(
                metadata_event("token_usage_summary", {
                    "prompt_tokens": non_stream_usage.get("prompt_tokens", 0),
                    "completion_tokens": non_stream_usage.get("completion_tokens", 0),
                    "total_tokens": non_stream_usage.get("total_tokens", 0),
                    "cached_tokens": non_stream_usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    if isinstance(non_stream_usage.get("prompt_tokens_details"), dict) else 0,
                    "reasoning_tokens": non_stream_usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                    if isinstance(non_stream_usage.get("completion_tokens_details"), dict) else 0,
                }),
                trace_meta,
            )
    
    def _calculate_backoff(self, attempt: int, response: "ClientResponse") -> float:
        """
        计算重试等待时间
        
        Args:
            attempt: 当前重试次数（0-based）
            response: HTTP 响应对象
        
        Returns:
            等待秒数
        """
        status = response.status
        
        # 429 限流：优先使用 Retry-After 头
        if status == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = int(retry_after)
                Log.debug(f"使用 Retry-After 头部: {delay}s", module=MODULE)
                return min(delay, 120)  # 最多等待 2 分钟
            
            # 没有 Retry-After，使用指数退避（429 用更长时间）
            delay = min(60, 4 * (2 ** attempt))  # 4s, 8s, 16s, 32s, 60s
            Log.debug(f"限流退避: {delay}s", module=MODULE)
            return delay
        
        # 其他可重试错误：标准指数退避
        delay = min(30, 2 ** attempt)  # 1s, 2s, 4s, 8s, 16s, 30s
        Log.debug(f"标准退避: {delay}s", module=MODULE)
        return delay

    def _annotate_event(self, event: StreamEvent, trace_meta: Dict[str, Any]) -> StreamEvent:
        metadata = dict(event.metadata) if event.metadata else {}
        for key in ("run_id", "iteration_id", "request_id"):
            if key in trace_meta:
                metadata.setdefault(key, trace_meta[key])
        if event.type in {
            EventType.TOOL_CALL_START,
            EventType.TOOL_CALL_DELTA,
            EventType.TOOL_CALL_DISPATCHED,
            EventType.TOOL_CALL_RESULT,
        }:
            tool_call_id = event.data.get("id") if isinstance(event.data, dict) else None
            if tool_call_id:
                metadata.setdefault("tool_call_id", tool_call_id)
        event.metadata = metadata
        return event
    
    def _tool_to_openai_spec(self, tool: Dict) -> Dict:
        """
        将工具字典转换为 OpenAI 工具规范。

        Args:
            tool: 工具定义。

        Returns:
            OpenAI 工具规范字典。
        """
        # 如果已经是 OpenAI 格式，直接返回
        if "type" in tool and "function" in tool:
            return tool
        
        # 否则从简化格式转换
        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }
