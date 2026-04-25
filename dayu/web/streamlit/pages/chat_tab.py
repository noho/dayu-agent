from dayu.web.streamlit.components.watchlist import WatchlistItem


import streamlit as st

from dataclasses import dataclass
from dayu.services.protocols import ChatServiceProtocol, HostAdminServiceProtocol, ReplyDeliveryServiceProtocol

@dataclass(frozen=True)
class ChatServiceClient:
    """聊天服务客户端。

    参数:
        chat_service: 聊天服务协议实例。
        host_admin_service: 宿主管理服务协议实例。
        reply_delivery_service: 回复投递服务协议实例。

    返回值:
        无。

    异常:
        无。
    """

    chat_service: ChatServiceProtocol
    host_admin_service: HostAdminServiceProtocol
    reply_delivery_service: ReplyDeliveryServiceProtocol


def create_chat_service_client(
    *,
    chat_service: ChatServiceProtocol,
    host_admin_service: HostAdminServiceProtocol,
    reply_delivery_service: ReplyDeliveryServiceProtocol,
) -> ChatServiceClient:
    """创建聊天服务客户端。

    参数:
        chat_service: 聊天服务协议实例。
        host_admin_service: 宿主管理服务协议实例。
        reply_delivery_service: 回复投递服务协议实例。

    返回值:
        组装完成的聊天服务客户端。

    异常:
        无。
    """

    return ChatServiceClient(
        chat_service=chat_service,
        host_admin_service=host_admin_service,
        reply_delivery_service=reply_delivery_service,
    )


def render_chat_tab(
    *,
    selected_stock: WatchlistItem,
    chat_service_client: ChatServiceClient | None,
) -> None:
    """渲染交互式分析 Tab。"""

    st.write(f"交互式分析: {selected_stock.ticker}")

