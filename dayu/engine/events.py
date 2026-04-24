"""
事件模型 - 统一的流式事件协议

定义 Engine 内部的事件类型与数据结构，用于 Runner/Agent 的异步事件流。

事件分类:
- 内容事件: content_delta / content_complete / reasoning_delta
- 工具事件: tool_call_start / tool_call_delta / tool_call_dispatched /
           tool_call_result / tool_calls_batch_ready / tool_calls_batch_done
- 控制事件: warning / error / done / final_answer

设计目标:
- 统一数据结构，便于透传与日志记录
- 保持事件顺序约束（由 Runner 保证）
- 支持结构化工具结果回填（tool_call_result.result 为 dict）
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """事件类型枚举"""
    
    # 内容生成事件
    CONTENT_DELTA = "content_delta"          # 内容增量（streaming 文本片段）
    CONTENT_COMPLETE = "content_complete"    # 内容生成完成
    REASONING_DELTA = "reasoning_delta"      # 推理增量（thinking 模式思维链片段）
    
    # 工具调用事件
    TOOL_CALL_START = "tool_call_start"              # 工具调用开始
    TOOL_CALL_DELTA = "tool_call_delta"              # 工具调用参数增量（streaming）
    TOOL_CALL_DISPATCHED = "tool_call_dispatched"    # 工具调用已发起执行
    TOOL_CALL_RESULT = "tool_call_result"            # 工具调用结果返回
    TOOL_CALLS_BATCH_READY = "tool_calls_batch_ready" # 工具调用批次已就绪
    TOOL_CALLS_BATCH_DONE = "tool_calls_batch_done"   # 工具调用批次完成
    
    # 迭代事件
    ITERATION_START = "iteration_start"      # Agent 开始新一轮迭代

    # 错误和完成事件
    ERROR = "error"                          # 错误
    WARNING = "warning"                      # 警告
    DONE = "done"                            # 完成
    
    # 元数据事件
    METADATA = "metadata"                    # 元数据更新（如 token 统计）
    FINAL_ANSWER = "final_answer"            # 最终答案事件（附带完整内容与降级标志）


@dataclass
class StreamEvent:
    """流式事件数据结构"""
    
    type: EventType
    """事件类型"""
    
    data: Any
    """事件数据（类型根据 type 不同而不同）"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """元数据（可选，用于携带额外信息）"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（方便序列化）"""
        return {
            "type": self.type.value,
            "data": self.data,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StreamEvent":
        """从字典创建事件"""
        return cls(
            type=EventType(d["type"]),
            data=d["data"],
            metadata=d.get("metadata", {}),
        )


# 便捷构造函数
def content_delta(text: str, **metadata) -> StreamEvent:
    """创建内容增量事件"""
    return StreamEvent(type=EventType.CONTENT_DELTA, data=text, metadata=metadata)


def content_complete(full_text: str, **metadata) -> StreamEvent:
    """创建内容完成事件"""
    return StreamEvent(type=EventType.CONTENT_COMPLETE, data=full_text, metadata=metadata)


def reasoning_delta(text: str, **metadata) -> StreamEvent:
    """创建推理增量事件（thinking 模式思维链片段）"""
    return StreamEvent(type=EventType.REASONING_DELTA, data=text, metadata=metadata)


def tool_call_start(tool_name: str, tool_call_id: str, **metadata) -> StreamEvent:
    """创建工具调用开始事件"""
    return StreamEvent(
        type=EventType.TOOL_CALL_START,
        data={"name": tool_name, "id": tool_call_id},
        metadata=metadata,
    )


def tool_call_delta(tool_call_id: str, name: str, arguments_delta: str, **metadata) -> StreamEvent:
    """创建工具调用参数增量事件"""
    return StreamEvent(
        type=EventType.TOOL_CALL_DELTA,
        data={"id": tool_call_id, "name": name, "arguments_delta": arguments_delta},
        metadata=metadata,
    )


_MAX_PARAM_PREVIEW_LEN = 40


def _build_param_preview(arguments: Any, summary_params: list[str] | None) -> str:
    """从工具调用参数中提取面向用户的摘要预览。

    Args:
        arguments: 工具调用参数（dict 或 JSON 字符串）。
        summary_params: 需要展示的参数名列表；None 时返回空字符串。

    Returns:
        逗号分隔的参数值预览，超长值截断。
    """
    if not summary_params:
        return ""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            return ""
    if not isinstance(arguments, dict):
        return ""
    values: list[str] = []
    for key in summary_params:
        if key not in arguments:
            continue
        val = arguments[key]
        if isinstance(val, list):
            text = ", ".join(str(item) for item in val)
        else:
            text = str(val)
        if len(text) > _MAX_PARAM_PREVIEW_LEN:
            text = text[:_MAX_PARAM_PREVIEW_LEN - 3] + "..."
        values.append(text)
    return ", ".join(values)


def tool_call_dispatched(
    tool_call_id: str,
    name: str,
    arguments: Any,
    *,
    index_in_iteration: int,
    display_name: str | None = None,
    summary_params: list[str] | None = None,
    **metadata,
) -> StreamEvent:
    """创建工具调用已发起执行事件"""
    return StreamEvent(
        type=EventType.TOOL_CALL_DISPATCHED,
        data={
            "id": tool_call_id,
            "name": name,
            "arguments": arguments,
            "index_in_iteration": index_in_iteration,
            "display_name": display_name or name,
            "param_preview": _build_param_preview(arguments, summary_params),
        },
        metadata=metadata,
    )


def tool_call_result(
    tool_call_id: str,
    result: Any,
    *,
    name: str,
    arguments: Any,
    index_in_iteration: int,
    display_name: str | None = None,
    **metadata,
) -> StreamEvent:
    """创建工具调用结果事件

    Args:
        tool_call_id: 工具调用ID
        result: 工具执行结果（结构化，包含 ok/value/error/meta）
        name: 工具名称
        arguments: 工具调用参数（与 tool_call_dispatched 保持一致）
        index_in_iteration: 工具调用在当前轮次的顺序
        display_name: 面向用户的展示名，None 时 fallback 到 name
        **metadata: 额外元数据（如 error_type）
    """
    data = {
        "id": tool_call_id,
        "name": name,
        "arguments": arguments,
        "index_in_iteration": index_in_iteration,
        "result": result,
        "display_name": display_name or name,
    }
    return StreamEvent(
        type=EventType.TOOL_CALL_RESULT,
        data=data,
        metadata=metadata,
    )


def tool_calls_batch_ready(call_ids: list[str], **metadata) -> StreamEvent:
    """创建工具调用批次就绪事件"""
    return StreamEvent(
        type=EventType.TOOL_CALLS_BATCH_READY,
        data={"call_ids": call_ids, "count": len(call_ids)},
        metadata=metadata,
    )


def tool_calls_batch_done(
    call_ids: list[str],
    *,
    ok: int,
    error: int,
    timeout: int,
    cancelled: int,
    **metadata,
) -> StreamEvent:
    """创建工具调用批次完成事件"""
    return StreamEvent(
        type=EventType.TOOL_CALLS_BATCH_DONE,
        data={
            "call_ids": call_ids,
            "ok": ok,
            "error": error,
            "timeout": timeout,
            "cancelled": cancelled,
        },
        metadata=metadata,
    )


def iteration_start_event(*, iteration: int, run_id: str, **metadata) -> StreamEvent:
    """创建 Agent 迭代开始事件。"""
    return StreamEvent(
        type=EventType.ITERATION_START,
        data={"iteration": iteration, "run_id": run_id},
        metadata=metadata,
    )


def error_event(message: str, exception: Optional[Exception] = None, recoverable: bool = False, **metadata) -> StreamEvent:
    """创建错误事件"""
    data = {"message": message, "recoverable": recoverable}
    if exception:
        data["exception_type"] = type(exception).__name__
        data["exception_str"] = str(exception)
    return StreamEvent(type=EventType.ERROR, data=data, metadata=metadata)


def warning_event(message: str, **metadata) -> StreamEvent:
    """创建警告事件"""
    return StreamEvent(type=EventType.WARNING, data={"message": message}, metadata=metadata)


def done_event(summary: Optional[Dict[str, Any]] = None, **metadata) -> StreamEvent:
    """创建完成事件"""
    return StreamEvent(type=EventType.DONE, data=summary or {}, metadata=metadata)


def metadata_event(key: str, value: Any, **extra_metadata) -> StreamEvent:
    """创建元数据事件"""
    return StreamEvent(
        type=EventType.METADATA,
        data={key: value},
        metadata=extra_metadata,
    )


def final_answer_event(
    content: str,
    degraded: bool = False,
    *,
    filtered: bool = False,
    finish_reason: str | None = None,
    **metadata,
) -> StreamEvent:
    """创建最终答案事件。"""

    payload: Dict[str, Any] = {"content": content, "degraded": degraded}
    if filtered:
        payload["filtered"] = True
    if finish_reason:
        payload["finish_reason"] = finish_reason
    return StreamEvent(
        type=EventType.FINAL_ANSWER,
        data=payload,
        metadata=metadata,
    )
