"""服务层公共 DTO 定义。

该模块只定义 UI -> Service 与 Service -> UI 的稳定契约，
不承载装配逻辑，也不暴露 Host 层内部记录类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Optional

from dayu.contracts.events import AppEvent
from dayu.contracts.execution_metadata import ExecutionDeliveryContext, empty_execution_delivery_context
from dayu.contracts.fins import FinsCommand, FinsEvent, FinsResult
from dayu.contracts.reply_outbox import ReplyOutboxState
from dayu.execution.options import ExecutionOptions


@dataclass
class SceneModelConfig:
    """写作流水线中单个 scene 的生效模型配置。

    Attributes:
        name: 生效模型名。
        temperature: 生效 temperature。
    """

    name: str
    temperature: float


@dataclass
class WriteRunConfig:
    """写作运行配置。

    Attributes:
        ticker: 公司股票代码。
        company: 公司名称。
        template_path: 模板文件绝对路径。
        output_dir: 输出目录绝对路径。
        write_max_retries: 章节重写最大次数。
        web_provider: 联网 provider 策略。
        resume: 是否启用断点恢复。
        write_model_override_name: 主写作场景模型覆盖名。
        audit_model_override_name: 审计场景模型覆盖名。
        scene_models: 各 scene 实际生效模型映射。
        chapter_filter: 章节过滤表达式。
        fast: 是否仅执行写作，不进入 audit/confirm/repair。
        force: 是否强制放宽第0章/第10章的 audit 前置门禁。
        infer: 是否仅执行一次公司级 facet 归因并写回 manifest。
    """

    ticker: str
    company: str
    template_path: str
    output_dir: str
    write_max_retries: int
    web_provider: str
    resume: bool
    write_model_override_name: str = ""
    audit_model_override_name: str = ""
    scene_models: dict[str, SceneModelConfig] = field(default_factory=dict)
    chapter_filter: str = ""
    fast: bool = False
    force: bool = False
    infer: bool = False


class SessionResolutionPolicy(str, Enum):
    """Service 层会话解析策略。

    UI 通过该策略声明“这次请求希望怎样解析 session”，
    Service 只理解会话生命周期，不再通过请求来源推断 UI 特例。
    """

    AUTO = "auto"
    CREATE_NEW = "create_new"
    REQUIRE_EXISTING = "require_existing"
    ENSURE_DETERMINISTIC = "ensure_deterministic"


@dataclass(frozen=True)
class ChatTurnRequest:
    """聊天单轮请求。

    Attributes:
        user_text: 用户输入文本。
        session_id: 可选会话 ID；首轮可不传，由 Service 创建。
        ticker: 可选股票代码。
        execution_options: 可选请求级执行参数覆盖。
        scene_name: 可选 scene 名称；未传时由 Service 使用默认值。
        session_resolution_policy: 会话解析策略。
        delivery_context: UI 侧交付上下文，用于 pending turn 恢复后重新投递回复。
    """

    user_text: str
    session_id: str | None = None
    ticker: Optional[str] = None
    execution_options: ExecutionOptions | None = None
    scene_name: str | None = None
    session_resolution_policy: SessionResolutionPolicy = SessionResolutionPolicy.AUTO
    delivery_context: ExecutionDeliveryContext | None = None


@dataclass(frozen=True)
class PromptRequest:
    """单轮 Prompt 请求。

    Attributes:
        user_text: 用户输入文本。
        ticker: 可选股票代码。
        session_id: 可选会话 ID；未传时由 Service 创建。
        execution_options: 可选请求级执行参数覆盖。
        session_resolution_policy: 会话解析策略。
    """

    user_text: str
    ticker: Optional[str] = None
    session_id: Optional[str] = None
    execution_options: ExecutionOptions | None = None
    session_resolution_policy: SessionResolutionPolicy = SessionResolutionPolicy.AUTO


@dataclass(frozen=True)
class FinsSubmitRequest:
    """财报服务提交请求。

    Attributes:
        command: 财报命令。
        session_resolution_policy: 会话解析策略。
    """

    command: FinsCommand
    session_resolution_policy: SessionResolutionPolicy = SessionResolutionPolicy.AUTO


@dataclass(frozen=True)
class ChatTurnSubmission:
    """聊天单轮提交句柄。

    Attributes:
        session_id: Service 解析后的会话 ID。
        event_stream: 事件流句柄。
    """

    session_id: str
    event_stream: AsyncIterator[AppEvent]


@dataclass(frozen=True)
class ChatPendingTurnView:
    """聊天服务暴露给 UI 的 pending turn 视图。"""

    pending_turn_id: str
    session_id: str
    scene_name: str
    user_text: str
    source_run_id: str
    resumable: bool
    state: str
    metadata: ExecutionDeliveryContext


@dataclass(frozen=True)
class ChatResumeRequest:
    """聊天恢复请求。

    Attributes:
        session_id: 当前请求所属会话 ID。
        pending_turn_id: 需要恢复的 pending turn ID。
    """

    session_id: str
    pending_turn_id: str


@dataclass(frozen=True)
class ReplyDeliverySubmitRequest:
    """reply delivery 提交请求。

    Attributes:
        delivery_key: 渠道层提供的稳定幂等键。
        session_id: 关联会话 ID。
        scene_name: 关联 scene 名。
        source_run_id: 产生该回复的执行 run ID。
        reply_content: 待交付的最终文本回复。
        metadata: 渠道交付上下文。
    """

    delivery_key: str
    session_id: str
    scene_name: str
    source_run_id: str
    reply_content: str
    metadata: ExecutionDeliveryContext = field(default_factory=empty_execution_delivery_context)


@dataclass(frozen=True)
class ReplyDeliveryFailureRequest:
    """reply delivery 失败回写请求。

    Attributes:
        delivery_id: 交付记录 ID。
        retryable: 是否允许后续重试。
        error_message: 失败说明。
    """

    delivery_id: str
    retryable: bool
    error_message: str


@dataclass(frozen=True)
class ReplyDeliveryView:
    """渠道层可见的交付视图。"""

    delivery_id: str
    delivery_key: str
    session_id: str
    scene_name: str
    source_run_id: str
    reply_content: str
    metadata: ExecutionDeliveryContext
    state: ReplyOutboxState
    created_at: str
    updated_at: str
    delivery_attempt_count: int
    last_error_message: str | None = None


@dataclass(frozen=True)
class PromptSubmission:
    """Prompt 提交句柄。

    Attributes:
        session_id: Service 解析后的会话 ID。
        event_stream: 事件流句柄。
    """

    session_id: str
    event_stream: AsyncIterator[AppEvent]


@dataclass(frozen=True)
class FinsSubmission:
    """财报服务提交句柄。

    Attributes:
        session_id: Service 解析后的会话 ID。
        execution: 同步结果或流式事件句柄。
    """

    session_id: str
    execution: FinsResult | AsyncIterator[FinsEvent]


@dataclass(frozen=True)
class SessionAdminView:
    """宿主管理面的会话视图。

    Attributes:
        session_id: 会话 ID。
        source: 会话来源。
        state: 会话状态。
        scene_name: 首次使用的 scene 名称。
        created_at: 创建时间的 ISO 文本。
        last_activity_at: 最后活跃时间的 ISO 文本。
    """

    session_id: str
    source: str
    state: str
    scene_name: str | None
    created_at: str
    last_activity_at: str


@dataclass(frozen=True)
class InteractiveSessionAdminView:
    """宿主管理面的 interactive 会话摘要视图。

    Attributes:
        session_id: 会话 ID。
        state: 会话状态。
        created_at: 创建时间的 ISO 文本。
        last_activity_at: 最后活跃时间的 ISO 文本。
        turn_count: 已持久化的 conversation turn 数量。
        first_question_preview: 第一轮用户问题预览。
        last_question_preview: 最后一轮用户问题预览。
        conversation_summary: 会话概览；当前为空，预留给后续一次性摘要。
    """

    session_id: str
    state: str
    created_at: str
    last_activity_at: str
    turn_count: int
    first_question_preview: str
    last_question_preview: str
    conversation_summary: str = ""


@dataclass(frozen=True)
class InteractiveSessionTurnView:
    """宿主管理面的 interactive 单轮对话视图。

    Attributes:
        user_text: 用户输入文本。
        assistant_text: 助手最终回答文本。
        created_at: 该轮对话创建时间的 ISO 文本。
    """

    user_text: str
    assistant_text: str
    created_at: str


@dataclass(frozen=True)
class RunAdminView:
    """宿主管理面的运行视图。

    Attributes:
        run_id: 运行 ID。
        session_id: 关联会话 ID。
        service_type: 服务类型。
        state: 运行状态。
        cancel_requested_at: 请求取消时间。
        cancel_requested_reason: 请求取消原因。
        cancel_reason: 取消原因。
        scene_name: scene 名称。
        created_at: 创建时间的 ISO 文本。
        started_at: 开始时间的 ISO 文本。
        finished_at: 结束时间的 ISO 文本。
        error_summary: 错误摘要。
    """

    run_id: str
    session_id: str | None
    service_type: str
    state: str
    cancel_requested_at: str | None
    cancel_requested_reason: str | None
    cancel_reason: str | None
    scene_name: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None
    error_summary: str | None


@dataclass(frozen=True)
class HostCleanupResult:
    """宿主清理结果。

    Attributes:
        orphan_run_ids: 被清理的孤儿 run ID。
        stale_permit_ids: 被清理的过期 permit ID。
    """

    orphan_run_ids: tuple[str, ...]
    stale_permit_ids: tuple[str, ...]


@dataclass(frozen=True)
class LaneStatusView:
    """并发通道状态视图。

    Attributes:
        lane: 通道名称。
        active: 当前活跃运行数。
        max_concurrent: 最大并发数。
    """

    lane: str
    active: int
    max_concurrent: int


@dataclass(frozen=True)
class HostStatusView:
    """宿主状态视图。

    Attributes:
        active_session_count: 活跃会话数量。
        total_session_count: 会话总数。
        active_run_count: 活跃运行数量。
        active_runs_by_type: 按服务类型聚合的活跃运行数。
        lane_statuses: 并发通道状态快照。
    """

    active_session_count: int
    total_session_count: int
    active_run_count: int
    active_runs_by_type: dict[str, int]
    lane_statuses: dict[str, LaneStatusView]


@dataclass(frozen=True)
class WriteRequest:
    """写作服务请求。"""

    write_config: WriteRunConfig
    execution_options: ExecutionOptions | None = None


__all__ = [
    "ChatPendingTurnView",
    "ChatResumeRequest",
    "ChatTurnSubmission",
    "ChatTurnRequest",
    "FinsSubmission",
    "FinsSubmitRequest",
    "HostCleanupResult",
    "HostStatusView",
    "LaneStatusView",
    "PromptRequest",
    "PromptSubmission",
    "ReplyDeliveryFailureRequest",
    "ReplyDeliverySubmitRequest",
    "ReplyDeliveryView",
    "RunAdminView",
    "SceneModelConfig",
    "SessionResolutionPolicy",
    "SessionAdminView",
    "WriteRequest",
    "WriteRunConfig",
]
