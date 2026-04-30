"""Session 数据模型。

定义会话（Session）的来源、状态和记录结构。
Session 是宿主层对一个交互会话的元数据索引，存储在 SQLite 中。
对话内容（ConversationSessionArchive）由 ConversationSessionArchiveStore 独立管理。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from dayu.contracts.execution_metadata import ExecutionDeliveryContext, empty_execution_delivery_context


class SessionSource(str, Enum):
    """Session 来源枚举。"""

    CLI = "cli"
    WEB = "web"
    WECHAT = "wechat"
    GUI = "gui"
    API = "api"
    INTERNAL = "internal"  # write pipeline 等内部创建


class SessionState(str, Enum):
    """Session 状态枚举。"""

    ACTIVE = "active"
    CLOSED = "closed"


@dataclass
class SessionRecord:
    """一个会话的元数据记录。

    与 ConversationTranscript 的关系：
    - SessionRecord 是宿主级元数据索引（来源、状态、时间戳），存 SQLite
    - ConversationTranscript 是对话内容存储（turns、episodes），存文件
    - 创建 session 不创建 transcript；prompt 单轮不需要 transcript
    - ConversationAgent 首次使用时才创建 transcript

    Attributes:
        session_id: 会话唯一标识。
        source: 会话来源（CLI / Web / WeChat 等）。
        state: 当前会话状态。
        scene_name: 首次使用的场景名称。
        created_at: 创建时间。
        last_activity_at: 最后活跃时间（由 service 层 touch 更新）。
        metadata: 会话级交付上下文元数据。
    """

    session_id: str
    source: SessionSource
    state: SessionState
    scene_name: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity_at: datetime = field(default_factory=datetime.now)
    metadata: ExecutionDeliveryContext = field(default_factory=empty_execution_delivery_context)

    def is_active(self) -> bool:
        """判断 session 是否处于活跃状态。

        Returns:
            True 表示 session 仍处于活跃状态。
        """
        return self.state == SessionState.ACTIVE
