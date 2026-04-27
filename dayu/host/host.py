"""Host 聚合根。"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Callable, TypeVar, cast

from dayu.contracts.agent_execution import ExecutionContract, ReplayHandle
from dayu.contracts.agent_execution_serialization import deserialize_execution_contract_snapshot
from dayu.contracts.events import AppEvent, AppResult
from dayu.contracts.events import PublishedRunEventProtocol
from dayu.contracts.execution_metadata import ExecutionDeliveryContext
from dayu.contracts.infrastructure import ModelCatalogProtocol, WorkspaceResourcesProtocol
from dayu.contracts.reply_outbox import ReplyOutboxRecord, ReplyOutboxState, ReplyOutboxSubmitRequest
from dayu.contracts.run import RunCancelReason, RunRecord, RunState
from dayu.contracts.session import SessionRecord, SessionSource, SessionState
from dayu.execution.options import ResolvedExecutionOptions
from dayu.engine.tool_registry import ToolRegistry
from dayu.host.concurrency import SQLiteConcurrencyGovernor
from dayu.host.conversation_store import ConversationStore, FileConversationStore
from dayu.host.executor import DefaultHostExecutor, should_delete_pending_turn_after_terminal_run
from dayu.host.host_execution import HostExecutorProtocol, HostedRunContext, HostedRunSpec
from dayu.host._datetime_utils import now_utc as _now_utc
from dayu.host.host_store import HostStore
from dayu.host.pending_turn_store import (
    InMemoryPendingConversationTurnStore,
    PendingConversationTurn,
    PendingConversationTurnState,
    PendingTurnResumeConflictError,
    SQLitePendingConversationTurnStore,
)
from dayu.host.startup_preparation import (
    DEFAULT_PENDING_TURN_RESUME_MAX_ATTEMPTS,
    DEFAULT_PENDING_TURN_RETENTION_HOURS,
)
from dayu.host.prepared_turn import (
    PreparedAgentTurnSnapshot,
    deserialize_prepared_agent_turn_snapshot,
)
from dayu.host.reply_outbox_store import InMemoryReplyOutboxStore, SQLiteReplyOutboxStore
from dayu.host.protocols import (
    ConcurrencyGovernorProtocol,
    EventSubscription,
    ConversationSessionDigest,
    ConversationSessionTurnExcerpt,
    LaneStatus,
    PendingConversationTurnStoreProtocol,
    PendingTurnSummary,
    ReplyOutboxStoreProtocol,
    RunEventBusProtocol,
    RunRegistryProtocol,
    SessionRegistryProtocol,
    SessionClosedError,
)
from dayu.log import Log
from dayu.host.run_registry import SQLiteRunRegistry
from dayu.host.scene_preparer import DefaultScenePreparer
from dayu.host.session_registry import SQLiteSessionRegistry
from dayu.workspace_paths import build_conversation_store_dir


MODULE = "HOST"
# 会话预览截断长度：控制 session 摘要、日志输出等场景下的 user_message 预览宽度，
# 48 字符能覆盖常见一行标题，同时避免日志中占用过多宽度。
_CONVERSATION_PREVIEW_MAX_CHARS = 48
_CONVERSATION_PREVIEW_SUFFIX = "..."
# 回复 outbox 清理阈值：启动调和时用来识别"陈旧"未送达的 reply delivery 记录。
# 15 分钟覆盖典型外部信道（WeChat / Web SSE）的正常延迟，超过该阈值
# 判定为上次进程异常退出时遗留，可安全丢弃，避免重启后重复投递历史消息。
_STALE_REPLY_OUTBOX_MAX_AGE = timedelta(minutes=15)

# RESUMING pending turn 超过该阈值后视为 resumer 异常退出遗留，启动/维护
# 阶段会把 state 原子回退到 pre_resume_state，允许后续 resume 重新 acquire。
# 10 分钟覆盖"正常 resume 最长耗时"（长链路 Agent 工具调用 + LLM 回复）的
# 安全余量，同时避免迟到的 resumer 进程产生双执行窗口。
_STALE_RESUMING_PENDING_TURN_MAX_AGE = timedelta(minutes=10)


TStreamEvent = TypeVar("TStreamEvent", bound=PublishedRunEventProtocol)
TSyncResult = TypeVar("TSyncResult")


class _PermanentPendingTurnResumeError(ValueError):
    """表示当前 pending turn 的恢复真源已永久损坏或已永久失效。"""


def _to_pending_turn_summary(record: PendingConversationTurn) -> PendingTurnSummary:
    """把 Host 内部 pending turn 记录映射为公开摘要。

    Args:
        record: Host 内部 pending turn 仓储记录。

    Returns:
        供 Service / UI 依赖的稳定公开摘要。

    Raises:
        无。
    """

    return PendingTurnSummary(
        pending_turn_id=record.pending_turn_id,
        session_id=record.session_id,
        scene_name=record.scene_name,
        user_text=record.user_text,
        source_run_id=record.source_run_id,
        resumable=record.resumable,
        state=record.state.value,
        metadata=record.metadata,
    )


def _build_default_conversation_store(
    workspace: WorkspaceResourcesProtocol | None,
) -> ConversationStore | None:
    """根据工作区构造默认 conversation 存储。

    Args:
        workspace: Host 运行所需工作区稳定资源。

    Returns:
        conversation 存储；未提供工作区时返回 ``None``。

    Raises:
        无。
    """

    if workspace is None:
        return None
    return FileConversationStore(build_conversation_store_dir(workspace.workspace_dir))


def _empty_conversation_session_digest() -> ConversationSessionDigest:
    """构造空 conversation 摘要。

    Args:
        无。

    Returns:
        空 conversation 摘要。

    Raises:
        无。
    """

    return ConversationSessionDigest(
        turn_count=0,
        first_question_preview="",
        last_question_preview="",
    )


def _truncate_conversation_preview(text: str) -> str:
    """截断 conversation 列表中的问题预览文本。

    Args:
        text: 原始问题文本。

    Returns:
        适合 CLI 表格展示的预览文本。

    Raises:
        无。
    """

    normalized = " ".join(str(text or "").split())
    if len(normalized) <= _CONVERSATION_PREVIEW_MAX_CHARS:
        return normalized
    content_length = _CONVERSATION_PREVIEW_MAX_CHARS - len(_CONVERSATION_PREVIEW_SUFFIX)
    return normalized[:content_length].rstrip() + _CONVERSATION_PREVIEW_SUFFIX


class Host:
    """Host 聚合根。

    设计意图：
    - `UI` 持有 `Host`，但不需要理解 Host 默认子组件的实现细节。
    - 生产代码通过稳定输入构造 `Host`，由 `Host` 内部装配默认
      `SQLiteSessionRegistry` / `SQLiteRunRegistry` / `SQLiteConcurrencyGovernor` /
      `DefaultScenePreparer` / `DefaultHostExecutor`。
    - 测试代码仍可显式注入 executor / registries / governor 等替身实现。

    Args:
        workspace: Host 运行所需工作区稳定资源；使用默认 scene preparation 时必填。
        model_catalog: Host 运行所需模型目录；使用默认 scene preparation 时必填。
        default_execution_options: Host 默认执行基线；使用默认 scene preparation 时必填。
        host_store_path: Host SQLite 存储路径；使用默认 SQLite 子组件时必填。
        lane_config: Host 并发 lane 配置；使用默认并发治理器时可选。
        event_bus: Host 事件总线；默认不启用。
        executor: 显式注入的宿主执行器。
        session_registry: 显式注入的 Session 注册表。
        run_registry: 显式注入的 Run 注册表。
        concurrency_governor: 显式注入的并发治理器。
        pending_turn_store: 显式注入的 pending turn 仓储。
        reply_outbox_store: 显式注入的 reply outbox 仓储。
        conversation_store: 显式注入的 conversation transcript 存储。

    Returns:
        无。

    Raises:
        ValueError: 默认装配所需稳定输入缺失时抛出。
    """

    def __init__(
        self,
        *,
        workspace: WorkspaceResourcesProtocol | None = None,
        model_catalog: ModelCatalogProtocol | None = None,
        default_execution_options: ResolvedExecutionOptions | None = None,
        host_store_path: Path | None = None,
        lane_config: dict[str, int] | None = None,
        pending_turn_resume_max_attempts: int = DEFAULT_PENDING_TURN_RESUME_MAX_ATTEMPTS,
        pending_turn_retention_hours: int = DEFAULT_PENDING_TURN_RETENTION_HOURS,
        event_bus: RunEventBusProtocol | None = None,
        executor: HostExecutorProtocol | None = None,
        session_registry: SessionRegistryProtocol | None = None,
        run_registry: RunRegistryProtocol | None = None,
        concurrency_governor: ConcurrencyGovernorProtocol | None = None,
        pending_turn_store: PendingConversationTurnStoreProtocol | None = None,
        reply_outbox_store: ReplyOutboxStoreProtocol | None = None,
        conversation_store: ConversationStore | None = None,
    ) -> None:
        """初始化 Host。"""

        explicit_core_components = (
            executor is not None,
            session_registry is not None,
            run_registry is not None,
            concurrency_governor is not None,
        )
        if any(explicit_core_components) and not (
            executor is not None
            and session_registry is not None
            and run_registry is not None
        ):
            raise ValueError(
                "显式注入 Host 内部子组件时，必须同时提供 executor、session_registry、run_registry"
            )

        if executor is not None and session_registry is not None and run_registry is not None:
            self._executor = executor
            self._session_registry = session_registry
            self._run_registry = run_registry
            self._concurrency_governor = concurrency_governor
            # 显式注入路径：若调用方未提供 store，则装配携带 session 活性屏障的默认
            # 内存 store，让 cancel_session 的写入屏障语义与默认装配路径保持一致。
            self._pending_turn_store = (
                pending_turn_store
                if pending_turn_store is not None
                else InMemoryPendingConversationTurnStore(session_activity=session_registry)
            )
            self._reply_outbox_store = (
                reply_outbox_store
                if reply_outbox_store is not None
                else InMemoryReplyOutboxStore(session_activity=session_registry)
            )
            self._event_bus = event_bus
            self._pending_turn_resume_max_attempts = pending_turn_resume_max_attempts
            self._pending_turn_retention = timedelta(hours=pending_turn_retention_hours)
            self._conversation_store = conversation_store or _build_default_conversation_store(workspace)
            return

        if host_store_path is None:
            raise ValueError("默认 Host 装配缺少 host_store_path")

        # 默认装配路径下 Host 与内部 ScenePreparer 共享同一 conversation_store 实例，
        # 避免出现指向同一 workspace 目录的多实例（文件锁虽保证跨实例一致性，
        # 但内存缓存/订阅状态无法共享，后续若引入 transcript 缓存会触发 stale read）。
        shared_conversation_store = conversation_store or _build_default_conversation_store(workspace)
        default_components = _build_default_host_components(
            workspace=workspace,
            model_catalog=model_catalog,
            default_execution_options=default_execution_options,
            host_store_path=host_store_path,
            lane_config=lane_config,
            event_bus=event_bus,
            conversation_store=shared_conversation_store,
        )
        self._executor = executor or default_components._executor
        self._session_registry = session_registry or default_components._session_registry
        self._run_registry = run_registry or default_components._run_registry
        self._concurrency_governor = concurrency_governor or default_components._concurrency_governor
        self._pending_turn_store = pending_turn_store or default_components._pending_turn_store
        self._reply_outbox_store = reply_outbox_store or default_components._reply_outbox_store
        self._event_bus = event_bus
        self._pending_turn_resume_max_attempts = pending_turn_resume_max_attempts
        self._pending_turn_retention = timedelta(hours=pending_turn_retention_hours)
        self._conversation_store = shared_conversation_store

    def create_session(
        self,
        source: SessionSource,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """创建新的 Host Session。

        Args:
            source: 会话来源。
            session_id: 可选显式 session_id。
            scene_name: 首次使用的 scene。
            metadata: 会话级交付上下文元数据。

        Returns:
            新建 SessionRecord。

        Raises:
            无。
        """

        session = self._session_registry.create_session(
            source,
            session_id=session_id,
            scene_name=scene_name,
            metadata=metadata,
        )
        Log.debug(
            f"Host 创建 session: session_id={session.session_id}, source={session.source.value}, scene_name={session.scene_name or ''}",
            module=MODULE,
        )
        return session

    def ensure_session(
        self,
        session_id: str,
        source: SessionSource,
        *,
        scene_name: str | None = None,
        metadata: ExecutionDeliveryContext | None = None,
    ) -> SessionRecord:
        """幂等获取或创建 Host Session。

        Args:
            session_id: 确定性会话 ID。
            source: 会话来源。
            scene_name: 首次使用的 scene。
            metadata: 会话级交付上下文元数据。

        Returns:
            现有或新建 SessionRecord。

        Raises:
            无。
        """

        session = self._session_registry.ensure_session(
            session_id,
            source,
            scene_name=scene_name,
            metadata=metadata,
        )
        Log.debug(
            f"Host ensure session: session_id={session.session_id}, source={source.value}, scene_name={session.scene_name or ''}",
            module=MODULE,
        )
        return session

    def touch_session(self, session_id: str) -> None:
        """刷新 Host Session 活跃时间。

        Args:
            session_id: 目标 session_id。

        Returns:
            无。

        Raises:
            KeyError: session 不存在时抛出。
        """

        self._session_registry.touch_session(session_id)
        Log.debug(f"Host touch session: session_id={session_id}", module=MODULE)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """查询单个 Host Session。

        Args:
            session_id: 目标 session_id。

        Returns:
            匹配的 SessionRecord；不存在时返回 `None`。

        Raises:
            无。
        """

        return self._session_registry.get_session(session_id)

    def list_sessions(
        self,
        *,
        state: SessionState | None = None,
        source: SessionSource | None = None,
        scene_name: str | None = None,
    ) -> list[SessionRecord]:
        """列出 Host Session。

        Args:
            state: 可选状态过滤。
            source: 可选来源过滤。
            scene_name: 可选 scene 名称过滤。

        Returns:
            匹配的 SessionRecord 列表。

        Raises:
            无。
        """

        return self._session_registry.list_sessions(
            state=state,
            source=source,
            scene_name=scene_name,
        )

    def get_conversation_session_digest(self, session_id: str) -> ConversationSessionDigest:
        """读取指定 session 的 conversation 摘要。

        Args:
            session_id: 目标 session ID。

        Returns:
            conversation 摘要；若 transcript 不存在则返回空摘要。

        Raises:
            无。
        """

        if self._conversation_store is None:
            return _empty_conversation_session_digest()
        transcript = self._conversation_store.load(session_id)
        if transcript is None or not transcript.turns:
            return _empty_conversation_session_digest()
        first_turn = transcript.turns[0]
        last_turn = transcript.turns[-1]
        return ConversationSessionDigest(
            turn_count=len(transcript.turns),
            first_question_preview=_truncate_conversation_preview(first_turn.user_text),
            last_question_preview=_truncate_conversation_preview(last_turn.user_text),
        )

    def list_conversation_session_turn_excerpts(
        self,
        session_id: str,
        *,
        limit: int,
    ) -> list[ConversationSessionTurnExcerpt]:
        """读取指定 session 的最近 conversation 单轮摘录。

        Args:
            session_id: 目标 session ID。
            limit: 最多返回的 turn 数量。

        Returns:
            最近单轮摘录列表，按时间从旧到新排列；若 transcript 不存在则返回空列表。

        Raises:
            无。
        """

        if self._conversation_store is None or limit <= 0:
            return []
        transcript = self._conversation_store.load(session_id)
        if transcript is None or not transcript.turns:
            return []
        recent_turns = transcript.turns[-limit:]
        return [
            ConversationSessionTurnExcerpt(
                user_text=turn.user_text,
                assistant_text=turn.assistant_final,
                created_at=turn.created_at,
            )
            for turn in recent_turns
        ]

    def run_operation_stream(
        self,
        *,
        spec: HostedRunSpec,
        event_stream_factory: Callable[[HostedRunContext], AsyncIterator[TStreamEvent]],
    ) -> AsyncIterator[TStreamEvent]:
        """托管一次流式 direct operation 执行。

        Args:
            spec: 宿主执行描述。
            event_stream_factory: 业务事件流工厂。

        Returns:
            由 Host 托管的事件流。

        Raises:
            无。
        """

        Log.debug(
            f"Host 托管流式 direct operation: operation_name={spec.operation_name}, session_id={spec.session_id or ''}, scene_name={spec.scene_name or ''}",
            module=MODULE,
        )
        return self._executor.run_operation_stream(
            spec=spec,
            event_stream_factory=event_stream_factory,
        )

    def run_operation_sync(
        self,
        *,
        spec: HostedRunSpec,
        operation: Callable[[HostedRunContext], TSyncResult],
        on_cancel: Callable[[], TSyncResult] | None = None,
    ) -> TSyncResult:
        """托管一次同步 direct operation 执行。

        Args:
            spec: 宿主执行描述。
            operation: 同步业务处理函数。
            on_cancel: 可选取消回调。

        Returns:
            业务执行结果。

        Raises:
            无。
        """

        Log.debug(
            f"Host 托管同步 direct operation: operation_name={spec.operation_name}, session_id={spec.session_id or ''}, scene_name={spec.scene_name or ''}",
            module=MODULE,
        )
        return self._executor.run_operation_sync(
            spec=spec,
            operation=operation,
            on_cancel=on_cancel,
        )

    async def run_agent_stream(
        self,
        execution_contract: ExecutionContract,
        *,
        resumed_pending_turn_id: str | None = None,
    ) -> AsyncIterator[AppEvent]:
        """托管一次 Agent 子执行并返回应用层事件流。

        Args:
            execution_contract: 已准备好的执行契约。
            resumed_pending_turn_id: resume 路径下由 Host 端传入的 pending turn
                ID；executor 以此识别"不要再 upsert pending turn"。非 resume 路径
                保持 ``None``。

        Yields:
            应用层事件。

        Raises:
            无。
        """

        Log.debug(
            f"Host 启动 agent stream: service_name={execution_contract.service_name}, session_id={execution_contract.host_policy.session_key or ''}, scene_name={execution_contract.scene_name}",
            module=MODULE,
        )
        async for event in self._executor.run_agent_stream(
            execution_contract,
            resumed_pending_turn_id=resumed_pending_turn_id,
        ):
            yield event

    async def run_prepared_turn_stream(
        self,
        prepared_turn: PreparedAgentTurnSnapshot,
        *,
        resumed_pending_turn_id: str | None = None,
    ) -> AsyncIterator[AppEvent]:
        """托管一次已完成 scene preparation 的 Agent 子执行。

        Args:
            prepared_turn: Host 已准备完成的稳定 turn 快照。
            resumed_pending_turn_id: resume 路径下由 Host 端传入的 pending turn
                ID；executor 以此识别"不要再 upsert pending turn"。非 resume 路径
                保持 ``None``。

        Yields:
            应用层事件。

        Raises:
            无。
        """

        Log.debug(
            f"Host 启动 prepared turn stream: service_name={prepared_turn.service_name}, scene_name={prepared_turn.scene_name}",
            module=MODULE,
        )
        async for event in self._executor.run_prepared_turn_stream(
            prepared_turn,
            resumed_pending_turn_id=resumed_pending_turn_id,
        ):
            yield event

    async def run_agent_and_wait(
        self,
        execution_contract: ExecutionContract,
    ) -> AppResult:
        """托管一次 Agent 子执行并等待完整结果。

        Args:
            execution_contract: 已准备好的执行契约。

        Returns:
            Agent 最终结果。

        Raises:
            无。
        """

        Log.debug(
            f"Host 启动 agent sync wait: service_name={execution_contract.service_name}, session_id={execution_contract.host_policy.session_key or ''}, scene_name={execution_contract.scene_name}",
            module=MODULE,
        )
        return await self._executor.run_agent_and_wait(execution_contract)

    async def run_agent_and_wait_replayable(
        self,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        """托管一次 Agent 子执行，并颁发可用于带历史回放的不透明句柄。

        Args:
            execution_contract: 已准备好的执行契约。

        Returns:
            ``(AppResult, ReplayHandle)`` 二元组；``ReplayHandle`` 仅在颁发
            它的 Host 实例进程内有效，session 关闭或 Host 重启即失效。

        Raises:
            CancelledError: 执行被取消时抛出。
        """

        Log.debug(
            f"Host 启动 agent replayable wait: service_name={execution_contract.service_name}, "
            f"session_id={execution_contract.host_policy.session_key or ''}, "
            f"scene_name={execution_contract.scene_name}",
            module=MODULE,
        )
        return await self._executor.run_agent_and_wait_replayable(execution_contract)

    async def replay_agent_and_wait(
        self,
        handle: ReplayHandle,
        execution_contract: ExecutionContract,
    ) -> tuple[AppResult, ReplayHandle]:
        """基于上一次颁发的 ``ReplayHandle`` 带历史回放一次 Agent 执行。

        Host 取出 handle 关联的对话历史，把
        ``execution_contract.message_inputs.user_message`` 追加到末尾，再用
        本次新颁发的 ``CancellationToken`` 重建 ``AsyncAgent`` 跑一次。

        Args:
            handle: 上一次 ``run_agent_and_wait_replayable`` 颁发的句柄。
            execution_contract: 用于本次回放的执行契约。

        Returns:
            ``(AppResult, ReplayHandle)`` 二元组；新句柄可继续追加 replay。

        Raises:
            RuntimeError: handle 已失效或与目标 session 不匹配。
            CancelledError: 执行被取消时抛出。
        """

        Log.debug(
            f"Host 启动 agent replay: handle_id={handle.handle_id}, "
            f"service_name={execution_contract.service_name}, "
            f"session_id={execution_contract.host_policy.session_key or ''}, "
            f"scene_name={execution_contract.scene_name}",
            module=MODULE,
        )
        return await self._executor.replay_agent_and_wait(handle, execution_contract)

    def discard_replay_state_for_session(self, session_id: str) -> None:
        """清理指定 session 关联的所有 replay 句柄状态。

        Args:
            session_id: 目标 session_id。

        Returns:
            无。

        Raises:
            无。
        """

        self._executor.discard_replay_state_for_session(session_id)

    def discard_replay_state(self, handle: ReplayHandle) -> None:
        """释放单个未消费的 replay 句柄。

        Args:
            handle: 待释放的不透明句柄；若已不存在则静默跳过。

        Returns:
            无。

        Raises:
            无。
        """

        self._executor.discard_replay_state(handle)

    def cancel_run(self, run_id: str) -> RunRecord:
        """请求取消指定 run（异步语义）。

        仅写入 ``cancel_requested_at``，让 cancel-bridge / executor 在协作边界感知，
        不会推进 run 状态。需要"请求取消并立即推进到 CANCELLED 终态"的同步语义时，
        使用 ``cancel_run_and_settle``。

        Args:
            run_id: 目标 run_id。

        Returns:
            更新后的 RunRecord。

        Raises:
            KeyError: run 不存在时抛出。
        """

        self._run_registry.request_cancel(run_id)
        run = self._run_registry.get_run(run_id)
        if run is None:
            raise KeyError(f"run 不存在: {run_id}")
        Log.debug(f"Host 请求取消 run: run_id={run_id}", module=MODULE)
        return run

    def cancel_run_and_settle(self, run_id: str) -> RunRecord:
        """同步取消 run 并推进到 CANCELLED 终态，同时清理关联 pending turn。

        sync CLI 在 SIGINT/SIGTERM/atexit 路径上调用：``KeyboardInterrupt`` 会同步
        打断 ``asyncio.run()``，event loop 立即终止，cancel-bridge / executor 收尾
        路径没机会跑完；若仅调 ``cancel_run`` 写 ``cancel_requested_at``，run 会
        卡在 ``running``、关联 pending turn 卡在 ``prepared_by_host``，下一轮新
        prompt 会命中"活跃 pending turn 不能写入不同 user_text"。本方法做同步
        合成保证 run 与 pending turn 都收敛到稳定状态。

        步骤：
            1. ``request_cancel`` 写 ``cancel_requested_at``（跨进程可见）；
            2. ``mark_cancelled`` 同步推到 CANCELLED（已幂等，重复调用安全）；
            3. ``cleanup_stale_pending_turns(session_id=run.session_id)`` 复用
               既有调和规则清理同 session 的 stale pending turn。

        Args:
            run_id: 目标 run_id。

        Returns:
            收敛后的 RunRecord。

        Raises:
            KeyError: run 不存在时抛出。
        """

        self._run_registry.request_cancel(
            run_id,
            cancel_reason=RunCancelReason.USER_CANCELLED,
        )
        latest = self._run_registry.get_run(run_id)
        if latest is None:
            raise KeyError(f"run 不存在: {run_id}")
        if latest.state == RunState.CANCELLED:
            settled = latest
        else:
            settled = self._run_registry.mark_cancelled(
                run_id,
                cancel_reason=RunCancelReason.USER_CANCELLED,
            )
        if settled.session_id is not None:
            self.cleanup_stale_pending_turns(session_id=settled.session_id)
        Log.debug(
            f"Host 同步取消并收敛 run: run_id={run_id}, session_id={settled.session_id or ''}",
            module=MODULE,
        )
        return settled

    def cancel_session(self, session_id: str) -> tuple[SessionRecord, list[str]]:
        """关闭 session 并取消其下所有活跃 run。

        Args:
            session_id: 目标 session_id。

        Returns:
            `(更新后的 session, 被取消的 run_id 列表)`。

        Raises:
            KeyError: session 不存在时抛出。
        """

        session = self.get_session(session_id)
        if session is None:
            raise KeyError(f"session 不存在: {session_id}")
        # 先 close session，把仓储层的写入屏障立起来；即使 cancel 流程内有并发
        # executor 正在登记 pending turn / reply outbox，屏障会让其 SessionClosedError
        # 降级为 no-op，避免产生孤儿数据。
        self._session_registry.close_session(session_id)
        cancelled_ids = self.cancel_session_runs(session_id)
        # 再做一次幂等 delete sweep，兜底回收 close_session 之前已成功写入的记录。
        self._reply_outbox_store.delete_by_session_id(session_id)
        self._pending_turn_store.delete_by_session_id(session_id)
        # session 关闭后清理 replay stash 内还挂着的句柄状态，避免 AgentInput /
        # 历史 messages 在 Host 内泄漏。该入口已上 ``HostExecutorProtocol``，
        # 直接走协议方法即可，禁止再做 ``isinstance`` 具体实现判定。
        self._executor.discard_replay_state_for_session(session_id)
        updated = self.get_session(session_id)
        if updated is None:
            raise KeyError(f"session 不存在: {session_id}")
        Log.info(
            f"Host 关闭 session: session_id={session_id}, cancelled_runs={len(cancelled_ids)}",
            module=MODULE,
        )
        return updated, cancelled_ids

    def get_run(self, run_id: str) -> RunRecord | None:
        """查询单个 run。

        Args:
            run_id: 目标 run_id。

        Returns:
            匹配的 RunRecord；不存在时返回 `None`。

        Raises:
            无。
        """

        return self._run_registry.get_run(run_id)

    def list_runs(
        self,
        *,
        session_id: str | None = None,
        state: RunState | None = None,
        service_type: str | None = None,
    ) -> list[RunRecord]:
        """列出 Host run。

        Args:
            session_id: 可选 session 过滤。
            state: 可选状态过滤。
            service_type: 可选服务类型过滤。

        Returns:
            匹配的 RunRecord 列表。

        Raises:
            无。
        """

        return self._run_registry.list_runs(
            session_id=session_id,
            state=state,
            service_type=service_type,
        )

    def list_active_runs(self) -> list[RunRecord]:
        """列出全部活跃 run。

        Args:
            无。

        Returns:
            当前活跃的 RunRecord 列表。

        Raises:
            无。
        """

        return self._run_registry.list_active_runs()

    def list_active_run_ids_for_current_owner(self) -> list[str]:
        """列出当前进程持有的全部活跃 run_id。

        sync CLI 在 SIGINT/SIGTERM/atexit 路径上需要兜底覆盖那些尚未通过事件流
        登记到 ``ProcessShutdownCoordinator`` 的 run，例如：
        - ``FinsService`` 直接调 ``run_operation_sync`` / ``run_operation_stream``，
          没有 observer 登记路径；
        - interactive / prompt 在事件流首帧 ``meta["run_id"]`` 之前，run 已经被
          ``HostExecutor.run_agent_stream`` ``register_run + start_run`` 但尚未
          被 observer 登记。

        Args:
            无。

        Returns:
            owner_pid 等于当前进程 PID 的活跃 run_id 列表。

        Raises:
            无。
        """

        owner_pid = os.getpid()
        return [
            record.run_id
            for record in self._run_registry.list_active_runs_for_owner(owner_pid)
        ]

    def cancel_session_runs(self, session_id: str) -> list[str]:
        """取消指定 session 下全部活跃 run。

        Args:
            session_id: 目标 session_id。

        Returns:
            成功登记取消请求的 run_id 列表。

        Raises:
            无。
        """

        cancelled_ids: list[str] = []
        for run in self.list_runs(session_id=session_id):
            if run.is_terminal():
                continue
            if self._run_registry.request_cancel(run.run_id):
                cancelled_ids.append(run.run_id)
        if cancelled_ids:
            Log.debug(
                f"Host 批量取消 session 下活跃 runs: session_id={session_id}, run_ids={','.join(cancelled_ids)}",
                module=MODULE,
            )
        return cancelled_ids

    def cleanup_orphan_runs(self) -> list[str]:
        """清理 owner_pid 已死亡的活跃 run。

        Args:
            无。

        Returns:
            被清理的 run_id 列表。

        Raises:
            无。
        """

        orphan_ids = self._run_registry.cleanup_orphan_runs()
        if orphan_ids:
            Log.info(f"Host 清理 orphan runs: run_ids={','.join(orphan_ids)}", module=MODULE)
        return orphan_ids

    def cleanup_stale_permits(self) -> list[str]:
        """清理 owner_pid 已死亡的并发 permit。

        Args:
            无。

        Returns:
            被清理的 permit_id 列表。

        Raises:
            无。
        """

        if self._concurrency_governor is None:
            return []
        return self._concurrency_governor.cleanup_stale_permits()

    def cleanup_stale_reply_outbox_deliveries(
        self,
        *,
        max_age: timedelta = _STALE_REPLY_OUTBOX_MAX_AGE,
    ) -> list[str]:
        """清理卡住的 DELIVERY_IN_PROGRESS reply outbox 记录。

        Args:
            max_age: IN_PROGRESS 多久未收到终态视为 stale，默认 15 分钟。

        Returns:
            被回退的 delivery_id 列表。

        Raises:
            无。失败仅 Log.warn。
        """

        try:
            stale_ids = self._reply_outbox_store.cleanup_stale_in_progress_deliveries(max_age=max_age)
        except Exception as exc:  # pragma: no cover - 防御 DB 异常
            Log.warn(f"Host 清理 stale reply outbox 失败: {exc}", module=MODULE)
            return []
        if stale_ids:
            Log.debug(
                f"Host 清理 stale reply outbox: count={len(stale_ids)}, ids={','.join(stale_ids)}",
                module=MODULE,
            )
        return stale_ids

    def cleanup_stale_pending_turns(
        self,
        *,
        session_id: str | None = None,
    ) -> list[str]:
        """清理关联 run 已终态、且按调和规则应删除的 pending turn。

        进程崩溃或启动调和路径上，``_reconcile_pending_turn_after_terminal_run``
        可能未完成调用；而 Host 的 orphan run cleanup 只收敛 run 本身，
        pending turn 会残留至下一次 resume 流程发现。本方法在启动/维护阶段
        主动扫描 pending turn，按终态 run 调和规则做兜底清理。

        Args:
            session_id: 若提供，仅扫描该 session 下的 pending turn；
                为 ``None`` 时全量扫描。

        Returns:
            被清理的 pending_turn_id 列表。

        Raises:
            无。失败仅 Log.warn。
        """

        try:
            records = self._pending_turn_store.list_pending_turns(session_id=session_id)
        except Exception as exc:  # pragma: no cover - 防御 DB 异常
            Log.warn(f"Host 扫描 pending turn 失败: {exc}", module=MODULE)
            return []
        cleaned_ids: list[str] = []
        released_ids: list[str] = []
        expired_ids: list[str] = []
        now = _now_utc()
        for record in records:
            # 分支 A：RESUMING 状态超时 → 回退 lease，允许后续 resume 重新 acquire。
            if record.state is PendingConversationTurnState.RESUMING:
                if now - record.updated_at < _STALE_RESUMING_PENDING_TURN_MAX_AGE:
                    continue
                try:
                    released = self._pending_turn_store.release_resume_lease(record.pending_turn_id)
                except Exception as exc:
                    Log.warn(
                        "Host 释放 stale RESUMING pending turn lease 失败: "
                        f"pending_turn_id={record.pending_turn_id}, error={exc}",
                        module=MODULE,
                    )
                    continue
                if released is not None and released.state is not PendingConversationTurnState.RESUMING:
                    released_ids.append(record.pending_turn_id)
                continue
            # 分支 B：source run 已终态 → 按既有调和规则删除。
            try:
                run = self._run_registry.get_run(record.source_run_id)
            except Exception as exc:  # pragma: no cover - 防御 DB 异常
                Log.warn(
                    f"Host 查询 pending turn 源 run 失败: "
                    f"pending_turn_id={record.pending_turn_id}, "
                    f"source_run_id={record.source_run_id}, error={exc}",
                    module=MODULE,
                )
                continue
            # source_run 仍活跃时严格保留 pending turn：它是执行链路上的恢复真源，
            # 即便 updated_at 已超过 retention 也禁止删除，避免 run 随后失败/超时
            # 时本应可 resume 的记录已经丢失。
            if run is not None and not run.is_terminal():
                continue
            if should_delete_pending_turn_after_terminal_run(
                run=run,
                resumable=record.resumable,
            ):
                try:
                    self._pending_turn_store.delete_pending_turn(record.pending_turn_id)
                    cleaned_ids.append(record.pending_turn_id)
                except Exception as exc:
                    Log.warn(
                        f"Host 清理 stale pending turn 失败: "
                        f"pending_turn_id={record.pending_turn_id}, error={exc}",
                        module=MODULE,
                    )
                continue
            # 分支 C：source_run 已终态但分支 B 判为保留（FAILED/UNSETTLED+resumable、
            # CANCELLED+TIMEOUT+resumable）且 updated_at 超过保留期 → 视作 UI 已错过
            # 询问窗口，兜底删除避免库无限累积。白名单 state 防止 RESUMING / 未知态
            # 被误删；活跃 run 已在上方 continue 掉，不会到达此处。
            if (
                record.state
                in (
                    PendingConversationTurnState.ACCEPTED_BY_HOST,
                    PendingConversationTurnState.PREPARED_BY_HOST,
                )
                and now - record.updated_at >= self._pending_turn_retention
            ):
                try:
                    self._pending_turn_store.delete_pending_turn(record.pending_turn_id)
                except Exception as exc:
                    Log.warn(
                        "Host 清理超保留期 pending turn 失败: "
                        f"pending_turn_id={record.pending_turn_id}, error={exc}",
                        module=MODULE,
                    )
                    continue
                expired_ids.append(record.pending_turn_id)
        if cleaned_ids:
            Log.debug(
                f"Host 清理 stale pending turns: count={len(cleaned_ids)}, "
                f"ids={','.join(cleaned_ids)}",
                module=MODULE,
            )
        if released_ids:
            Log.debug(
                f"Host 回退 stale RESUMING pending turns: count={len(released_ids)}, "
                f"ids={','.join(released_ids)}",
                module=MODULE,
            )
        if expired_ids:
            Log.debug(
                f"Host 清理超保留期 pending turns: count={len(expired_ids)}, "
                f"ids={','.join(expired_ids)}",
                module=MODULE,
            )
        return cleaned_ids + expired_ids

    def get_all_lane_statuses(self) -> dict[str, LaneStatus]:
        """获取全部并发 lane 状态快照。

        Args:
            无。

        Returns:
            lane 名到状态快照的映射；未启用并发治理器时返回空映射。

        Raises:
            无。
        """

        if self._concurrency_governor is None:
            return {}
        return self._concurrency_governor.get_all_status()

    def subscribe_run_events(self, run_id: str) -> EventSubscription:
        """订阅指定 run 的事件流。

        Args:
            run_id: 目标 run_id。

        Returns:
            事件订阅句柄。

        Raises:
            RuntimeError: 未启用事件总线时抛出。
        """

        if self._event_bus is None:
            raise RuntimeError("event bus not enabled")
        return self._event_bus.subscribe(run_id=run_id)

    def subscribe_session_events(self, session_id: str) -> EventSubscription:
        """订阅指定 session 下全部 run 的事件流。

        Args:
            session_id: 目标 session_id。

        Returns:
            事件订阅句柄。

        Raises:
            RuntimeError: 未启用事件总线时抛出。
        """

        if self._event_bus is None:
            raise RuntimeError("event bus not enabled")
        return self._event_bus.subscribe(session_id=session_id)

    def get_session_pending_turn(
        self,
        *,
        session_id: str,
        scene_name: str,
    ) -> PendingTurnSummary | None:
        """查询指定 session/scene 的 pending turn。

        Args:
            session_id: 目标会话 ID。
            scene_name: 目标 scene 名。

        Returns:
            匹配的 pending turn 摘要；不存在时返回 ``None``。

        Raises:
            无。
        """

        record = self._pending_turn_store.get_session_pending_turn(
            session_id=session_id,
            scene_name=scene_name,
        )
        if record is None:
            return None
        return _to_pending_turn_summary(record)

    def _get_pending_turn_record(self, pending_turn_id: str) -> PendingConversationTurn | None:
        """按 ID 查询 Host 内部 pending turn 仓储记录。

        Args:
            pending_turn_id: 目标 pending turn ID。

        Returns:
            匹配的内部仓储记录；不存在时返回 ``None``。

        Raises:
            无。
        """

        return self._pending_turn_store.get_pending_turn(pending_turn_id)

    def _delete_pending_turn(self, pending_turn_id: str) -> None:
        """删除 Host 内部 pending turn 真源。

        Args:
            pending_turn_id: 目标 pending turn ID。

        Returns:
            无。

        Raises:
            无。
        """

        self._pending_turn_store.delete_pending_turn(pending_turn_id)

    def get_pending_turn(self, pending_turn_id: str) -> PendingTurnSummary | None:
        """按 ID 查询 pending turn。

        Args:
            pending_turn_id: 目标 pending turn ID。

        Returns:
            匹配的 pending turn 摘要；不存在时返回 ``None``。

        Raises:
            无。
        """

        record = self._get_pending_turn_record(pending_turn_id)
        if record is None:
            return None
        return _to_pending_turn_summary(record)

    async def resume_pending_turn_stream(
        self,
        pending_turn_id: str,
        *,
        session_id: str,
    ) -> AsyncIterator[AppEvent]:
        """校验 pending turn 是否允许恢复，并直接恢复执行。

        Args:
            pending_turn_id: 目标 pending turn ID。
            session_id: 请求方所属 session ID。

        Yields:
            恢复执行产生的应用层事件。

        Raises:
            KeyError: pending turn 或 source run 不存在时抛出。
            ValueError: pending turn 当前不可恢复时抛出；内部仓储屏障异常
                （如 :class:`SessionClosedError`）也会在此方法的边界被统一转换为
                ``ValueError``，对外不泄漏 ``RuntimeError`` 子类。
        """

        pending_turn_record = self._require_resume_pending_turn_record(
            pending_turn_id,
            session_id=session_id,
        )
        # source run 合法性必须在 acquire 之前校验：活跃 run 关联的 pending turn
        # 不允许被任何人 resume。真源反序列化则挪到 acquire 之后，避免并发第二个
        # resumer 读到 RESUMING + pre_resume_state=ACCEPTED_BY_HOST 的记录时，
        # 按当前 state 误把 accepted snapshot 当 prepared snapshot 解析并将其
        # 误判为 "永久损坏" 而删除。
        try:
            self._validate_source_run_for_resume(pending_turn_record)
        except _PermanentPendingTurnResumeError as exc:
            try:
                self._delete_pending_turn(pending_turn_id)
            except Exception as delete_exc:
                Log.warning(
                    "pending conversation turn 已永久不可恢复，但删除记录失败"
                    f" pending_turn_id={pending_turn_id}"
                    f" session_id={session_id}"
                    f" delete_error={delete_exc}",
                    module=MODULE,
                )
            raise ValueError(
                f"{exc}，已拒绝继续恢复: "
                f"pending_turn_id={pending_turn_id}, session_id={session_id}"
            ) from exc
        try:
            pending_turn_record = self._pending_turn_store.record_resume_attempt(
                pending_turn_id,
                max_attempts=self._pending_turn_resume_max_attempts,
            )
        except PendingTurnResumeConflictError as exc:
            # 多进程部署下，其他 resumer 已持有该 pending turn；转成明确的
            # ValueError，避免与"达上限"语义混淆。
            raise ValueError(
                "pending conversation turn 正被其他 resumer 持有，拒绝并发恢复: "
                f"pending_turn_id={pending_turn_id}, session_id={session_id}"
            ) from exc
        except SessionClosedError as exc:
            # acquire lease 阶段的仓储写入屏障：session 已在 acquire 前后一刻被关闭；
            # 此时 lease 尚未落地，直接按"session 已关闭"契约对外暴露 ValueError，
            # 避免把 RuntimeError 子类泄漏给 Service / UI。
            raise ValueError(
                "pending conversation turn 所属 session 已关闭，拒绝恢复: "
                f"pending_turn_id={pending_turn_id}, session_id={session_id}"
            ) from exc
        except ValueError as exc:
            # record_resume_attempt 内部已在达上限时原子删除记录；此处不再重复 DELETE，避免无意义 SQL 与模糊的代码意图。
            raise ValueError(
                "pending conversation turn 已达到最大恢复次数，已删除: "
                f"pending_turn_id={pending_turn_id}, session_id={session_id}"
            ) from exc
        # 已持有 RESUMING lease 后再解析真源；以 pre_resume_state 判目标类型，
        # 避免被 RESUMING 中间态干扰。任何解析失败从此刻起都只影响当前持有者，
        # 不会误伤并发 resumer 的合法记录。
        try:
            resume_target_kind, resume_target = self._prepare_resume_target_for_acquired_turn(
                pending_turn_record,
            )
        except _PermanentPendingTurnResumeError as exc:
            try:
                self._delete_pending_turn(pending_turn_id)
            except Exception as delete_exc:
                Log.warning(
                    "pending conversation turn 已永久不可恢复，但删除记录失败"
                    f" pending_turn_id={pending_turn_id}"
                    f" session_id={session_id}"
                    f" delete_error={delete_exc}",
                    module=MODULE,
                )
            raise ValueError(
                f"{exc}，已拒绝继续恢复: "
                f"pending_turn_id={pending_turn_id}, session_id={session_id}"
            ) from exc
        try:
            Log.verbose(
                "开始恢复 pending turn: "
                f"pending_turn_id={pending_turn_id}, session_id={session_id}, "
                f"state={pending_turn_record.state.value}, source_run_id={pending_turn_record.source_run_id}, "
                f"resume_attempt_count={pending_turn_record.resume_attempt_count}",
                module=MODULE,
            )
            if resume_target_kind == PendingConversationTurnState.ACCEPTED_BY_HOST.value:
                Log.verbose(
                    f"pending turn 按 accepted snapshot 恢复，将重新执行 scene preparation: "
                    f"pending_turn_id={pending_turn_id}",
                    module=MODULE,
                )
                execution_contract = cast(ExecutionContract, resume_target)
                async for event in self._executor.run_agent_stream(
                    execution_contract,
                    resumed_pending_turn_id=pending_turn_id,
                ):
                    yield event
                return
            Log.verbose(
                f"pending turn 按 prepared snapshot 恢复，将直接重放 prepared execution: "
                f"pending_turn_id={pending_turn_id}",
                module=MODULE,
            )
            prepared_turn = cast(PreparedAgentTurnSnapshot, resume_target)
            async for event in self._executor.run_prepared_turn_stream(
                prepared_turn,
                resumed_pending_turn_id=pending_turn_id,
            ):
                yield event
        except Exception as exc:
            current_record = self._get_pending_turn_record(pending_turn_id)
            if current_record is not None:
                if current_record.resume_attempt_count >= self._pending_turn_resume_max_attempts:
                    try:
                        self._delete_pending_turn(pending_turn_id)
                    except SessionClosedError as delete_exc:
                        Log.warning(
                            "pending conversation turn 达到最大恢复次数但删除被 session 屏障拒绝: "
                            f"pending_turn_id={pending_turn_id}, session_id={session_id}, "
                            f"error={delete_exc}",
                            module=MODULE,
                        )
                    raise ValueError(
                        "pending conversation turn 恢复失败且已达到最大恢复次数，已删除: "
                        f"pending_turn_id={pending_turn_id}, session_id={session_id}"
                    ) from exc
                try:
                    self._pending_turn_store.record_resume_failure(
                        pending_turn_id,
                        error_message=str(exc),
                    )
                except SessionClosedError as failure_exc:
                    # session 已关闭，lease 回退将由 cleanup_stale_pending_turns 兜底；
                    # 此时不得再抛 SessionClosedError 覆盖原因异常。
                    Log.warning(
                        "pending conversation turn lease 回退被 session 屏障拒绝，"
                        f"依赖 cleanup 兜底: pending_turn_id={pending_turn_id}, "
                        f"session_id={session_id}, error={failure_exc}",
                        module=MODULE,
                    )
            # Host 对外只暴露业务语义异常：把仓储写入屏障抛出的
            # SessionClosedError（session 不存在或已 CLOSED）统一转成 ValueError，
            # 避免将内部 RuntimeError 子类泄漏给 Service / UI，保持 docstring
            # 声明的 "KeyError | ValueError" 契约。
            if isinstance(exc, SessionClosedError):
                raise ValueError(
                    "pending conversation turn 所属 session 已关闭，拒绝恢复: "
                    f"pending_turn_id={pending_turn_id}, session_id={session_id}"
                ) from exc
            raise

    def _require_resume_pending_turn_record(
        self,
        pending_turn_id: str,
        *,
        session_id: str,
    ) -> PendingConversationTurn:
        """校验并返回允许进入恢复流程的 pending turn 内部记录。"""

        pending_turn_record = self._get_pending_turn_record(pending_turn_id)
        if pending_turn_record is None:
            raise KeyError(f"pending conversation turn 不存在: {pending_turn_id}")
        if pending_turn_record.session_id != session_id:
            raise ValueError(
                "pending conversation turn 不属于当前 session，不能恢复: "
                f"pending_turn_id={pending_turn_id}, session_id={session_id}"
            )
        if not pending_turn_record.resumable:
            raise ValueError(f"pending conversation turn 不可恢复: {pending_turn_id}")
        return pending_turn_record

    def _validate_source_run_for_resume(
        self,
        pending_turn_record: PendingConversationTurn,
    ) -> None:
        """校验 pending turn 对应 source run 是否允许恢复。"""

        source_run = self._run_registry.get_run(pending_turn_record.source_run_id)
        if source_run is None:
            raise _PermanentPendingTurnResumeError(
                "pending conversation turn 对应的 source run 不存在，"
                f"pending_turn_id={pending_turn_record.pending_turn_id}, source_run_id={pending_turn_record.source_run_id}"
            )
        if source_run.state in {RunState.CREATED, RunState.QUEUED, RunState.RUNNING}:
            raise ValueError(
                "pending conversation turn 对应的 source run 仍处于活跃状态，不能恢复: "
                f"pending_turn_id={pending_turn_record.pending_turn_id}, source_run_id={pending_turn_record.source_run_id}"
            )
        if source_run.state == RunState.SUCCEEDED:
            raise _PermanentPendingTurnResumeError(
                "pending conversation turn 对应的 source run 已成功完成，V1 不支持补投递恢复: "
                f"pending_turn_id={pending_turn_record.pending_turn_id}, source_run_id={pending_turn_record.source_run_id}"
            )
        if source_run.state == RunState.CANCELLED and source_run.cancel_reason != RunCancelReason.TIMEOUT:
            raise _PermanentPendingTurnResumeError(
                "pending conversation turn 对应的 source run 不是 timeout 取消，不能恢复: "
                f"pending_turn_id={pending_turn_record.pending_turn_id}, source_run_id={pending_turn_record.source_run_id}"
            )

    def _prepare_resume_target_for_acquired_turn(
        self,
        pending_turn_record: PendingConversationTurn,
    ) -> tuple[str, ExecutionContract | PreparedAgentTurnSnapshot]:
        """为已 acquire 的 pending turn 反序列化恢复目标。

        Args:
            pending_turn_record: 已被 ``record_resume_attempt`` 翻到 RESUMING 态的
                pending turn 记录；``pre_resume_state`` 字段必须非空。

        Returns:
            二元组 ``(state_value, 恢复目标)``；来源态为 ACCEPTED_BY_HOST 时
            返回 ``ExecutionContract``，否则返回 ``PreparedAgentTurnSnapshot``。
            返回的 state_value 使用来源态的枚举值，供调用方路由执行路径。

        Raises:
            _PermanentPendingTurnResumeError: resume_source_json 损坏、缺失或
                与 pre_resume_state 不匹配时抛出。
        """

        if pending_turn_record.state is not PendingConversationTurnState.RESUMING:
            raise _PermanentPendingTurnResumeError(
                "pending turn 未持有 RESUMING lease，不能进入真源反序列化: "
                f"pending_turn_id={pending_turn_record.pending_turn_id}, "
                f"state={pending_turn_record.state.value}"
            )
        source_state = pending_turn_record.pre_resume_state
        if source_state is None:
            raise _PermanentPendingTurnResumeError(
                "pending turn 已持有 RESUMING lease 但 pre_resume_state 缺失，"
                f"pending_turn_id={pending_turn_record.pending_turn_id}"
            )
        resume_source_json = str(pending_turn_record.resume_source_json or "").strip()
        if not resume_source_json:
            raise _PermanentPendingTurnResumeError(
                "pending conversation turn 缺少 resume_source_json，"
                f"不支持恢复 legacy 记录: {pending_turn_record.pending_turn_id}"
            )
        try:
            payload = json.loads(resume_source_json)
        except ValueError as exc:
            raise _PermanentPendingTurnResumeError(
                "pending turn resume_source_json 不是合法 JSON object: "
                f"pending_turn_id={pending_turn_record.pending_turn_id}"
            ) from exc
        if not isinstance(payload, dict):
            raise _PermanentPendingTurnResumeError("pending turn resume_source_json 必须是 JSON object")
        if source_state is PendingConversationTurnState.ACCEPTED_BY_HOST:
            try:
                execution_contract = deserialize_execution_contract_snapshot(cast(dict[str, Any], payload))
            except ValueError as exc:
                raise _PermanentPendingTurnResumeError(str(exc)) from exc
            return source_state.value, execution_contract
        try:
            prepared_turn = deserialize_prepared_agent_turn_snapshot(payload)
        except ValueError as exc:
            raise _PermanentPendingTurnResumeError(str(exc)) from exc
        return source_state.value, prepared_turn

    def list_pending_turns(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        resumable_only: bool = False,
    ) -> list[PendingTurnSummary]:
        """列出 Host 侧 pending turn。

        Args:
            session_id: 可选 session 过滤。
            scene_name: 可选 scene 过滤。
            resumable_only: 是否仅返回可恢复记录。

        Returns:
            匹配的 pending turn 摘要列表。

        Raises:
            无。
        """

        records = self._pending_turn_store.list_pending_turns(
            session_id=session_id,
            scene_name=scene_name,
            resumable_only=resumable_only,
        )
        return [_to_pending_turn_summary(record) for record in records]

    def submit_reply_for_delivery(self, request: ReplyOutboxSubmitRequest) -> ReplyOutboxRecord:
        """显式提交待交付回复。

        Args:
            request: reply outbox 提交请求。

        Returns:
            持久化后的交付记录。

        Raises:
            ValueError: 提交参数非法或 delivery_key 负载冲突时抛出。
        """

        return self._reply_outbox_store.submit_reply(request)

    def get_reply_outbox(self, delivery_id: str) -> ReplyOutboxRecord | None:
        """按 ID 查询 reply outbox 记录。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            匹配记录；不存在时返回 ``None``。

        Raises:
            ValueError: delivery_id 为空时抛出。
        """

        return self._reply_outbox_store.get_reply(delivery_id)

    def list_reply_outbox(
        self,
        *,
        session_id: str | None = None,
        scene_name: str | None = None,
        state: ReplyOutboxState | None = None,
    ) -> list[ReplyOutboxRecord]:
        """列出 reply outbox 记录。

        Args:
            session_id: 可选 session 过滤。
            scene_name: 可选 scene 过滤。
            state: 可选状态过滤。

        Returns:
            匹配记录列表。

        Raises:
            ValueError: 过滤字段为空字符串时抛出。
        """

        return self._reply_outbox_store.list_replies(
            session_id=session_id,
            scene_name=scene_name,
            state=state,
        )

    def claim_reply_delivery(self, delivery_id: str) -> ReplyOutboxRecord:
        """把 reply outbox 记录推进到发送中状态。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 当前状态不允许 claim 时抛出。
        """

        return self._reply_outbox_store.claim_reply(delivery_id)

    def mark_reply_delivered(self, delivery_id: str) -> ReplyOutboxRecord:
        """标记 reply outbox 记录已交付完成。

        Args:
            delivery_id: 交付记录 ID。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
        """

        return self._reply_outbox_store.mark_delivered(delivery_id)

    def mark_reply_delivery_failed(
        self,
        delivery_id: str,
        *,
        retryable: bool,
        error_message: str,
    ) -> ReplyOutboxRecord:
        """标记 reply outbox 记录交付失败。

        Args:
            delivery_id: 交付记录 ID。
            retryable: 是否允许后续再次 claim。
            error_message: 失败消息。

        Returns:
            更新后的交付记录。

        Raises:
            KeyError: 记录不存在时抛出。
            ValueError: 已完成交付的记录重复标记失败时抛出。
        """

        return self._reply_outbox_store.mark_failed(
            delivery_id,
            retryable=retryable,
            error_message=error_message,
        )

class _DefaultHostComponents:
    """Host 默认子组件集合。"""

    def __init__(
        self,
        *,
        executor: DefaultHostExecutor,
        session_registry: SQLiteSessionRegistry,
        run_registry: SQLiteRunRegistry,
        concurrency_governor: SQLiteConcurrencyGovernor,
        pending_turn_store: SQLitePendingConversationTurnStore,
        reply_outbox_store: SQLiteReplyOutboxStore,
    ) -> None:
        """初始化默认子组件集合。

        Args:
            executor: 默认宿主执行器。
            session_registry: 默认 Session 注册表。
            run_registry: 默认 Run 注册表。
            concurrency_governor: 默认并发治理器。
            pending_turn_store: 默认 pending turn 仓储。
            reply_outbox_store: 默认 reply outbox 仓储。

        Returns:
            无。

        Raises:
            无。
        """

        self._executor = executor
        self._session_registry = session_registry
        self._run_registry = run_registry
        self._concurrency_governor = concurrency_governor
        self._pending_turn_store = pending_turn_store
        self._reply_outbox_store = reply_outbox_store


def _build_default_host_components(
    *,
    workspace: WorkspaceResourcesProtocol | None,
    model_catalog: ModelCatalogProtocol | None,
    default_execution_options: ResolvedExecutionOptions | None,
    host_store_path: Path,
    lane_config: dict[str, int] | None,
    event_bus: RunEventBusProtocol | None,
    conversation_store: ConversationStore | None,
) -> _DefaultHostComponents:
    """构造 Host 默认内部子组件。

    Args:
        workspace: 工作区稳定资源。
        model_catalog: 模型目录。
        default_execution_options: 默认执行基线。
        host_store_path: Host SQLite 路径。
        lane_config: 并发 lane 配置。
        event_bus: 事件总线。
        conversation_store: Host 外层构造并共享的 conversation 存储；
            传入后将由内部 ScenePreparer 与 Host 公共字段复用同一实例。

    Returns:
        默认子组件集合。

    Raises:
        ValueError: 关键稳定输入缺失时抛出。
    """

    host_store = HostStore(host_store_path)
    host_store.initialize_schema()
    session_registry = SQLiteSessionRegistry(host_store)
    run_registry = SQLiteRunRegistry(host_store)
    concurrency_governor = SQLiteConcurrencyGovernor(host_store, lane_config=lane_config)
    # 注入 session_registry 作为 session activity 查询源，让仓储层可以在 session
    # 已关闭时以 SessionClosedError 屏障 executor 的迟到写入。
    pending_turn_store = SQLitePendingConversationTurnStore(
        host_store, session_activity=session_registry,
    )
    reply_outbox_store = SQLiteReplyOutboxStore(
        host_store, session_activity=session_registry,
    )
    scene_preparation = _build_default_scene_preparation(
        workspace=workspace,
        model_catalog=model_catalog,
        default_execution_options=default_execution_options,
        conversation_store=conversation_store,
    )
    executor = DefaultHostExecutor(
        run_registry=run_registry,
        concurrency_governor=concurrency_governor,
        event_bus=event_bus,
        scene_preparation=scene_preparation,
        pending_turn_store=pending_turn_store,
    )
    return _DefaultHostComponents(
        executor=executor,
        session_registry=session_registry,
        run_registry=run_registry,
        concurrency_governor=concurrency_governor,
        pending_turn_store=pending_turn_store,
        reply_outbox_store=reply_outbox_store,
    )


def _build_default_scene_preparation(
    *,
    workspace: WorkspaceResourcesProtocol | None,
    model_catalog: ModelCatalogProtocol | None,
    default_execution_options: ResolvedExecutionOptions | None,
    conversation_store: ConversationStore | None,
) -> DefaultScenePreparer | None:
    """构造 Host 默认 scene preparation。

    Args:
        workspace: 工作区稳定资源。
        model_catalog: 模型目录。
        default_execution_options: 默认执行基线。
        conversation_store: 由 Host 构造并共享的 conversation 存储；
            传入 ``None`` 时 ScenePreparer 会回退为按 workspace 自建实例，
            正式默认装配路径不应使用该回退。
    Returns:
        默认 scene preparation；若未提供执行路径所需稳定输入则返回 ``None``。

    Raises:
        ValueError: 仅提供部分 scene preparation 输入时抛出。
    """

    provided_inputs = (
        workspace is not None,
        model_catalog is not None,
        default_execution_options is not None,
    )
    if not any(provided_inputs):
        return None
    if not all(provided_inputs):
        raise ValueError(
            "默认 Host scene preparation 需要同时提供 workspace、model_catalog、default_execution_options"
        )
    assert workspace is not None
    assert model_catalog is not None
    assert default_execution_options is not None
    return DefaultScenePreparer(
        workspace=workspace,
        model_catalog=model_catalog,
        default_execution_options=default_execution_options,
        tool_registry_factory=lambda: ToolRegistry(),
        conversation_store=conversation_store,
    )


__all__ = ["Host"]
