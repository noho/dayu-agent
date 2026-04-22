"""Streamlit 交互式分析页的 Service 客户端封装。"""

from __future__ import annotations

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

    def reserved_admin_service(self) -> HostAdminServiceProtocol:
        """返回预留宿主管理服务实例。

        参数:
            无。

        返回值:
            宿主管理服务协议实例。

        异常:
            无。
        """

        return self.host_admin_service

    def reserved_reply_delivery_service(self) -> ReplyDeliveryServiceProtocol:
        """返回预留回复投递服务实例。

        参数:
            无。

        返回值:
            回复投递服务协议实例。

        异常:
            无。
        """

        return self.reply_delivery_service


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


__all__ = ["ChatServiceClient", "create_chat_service_client"]
