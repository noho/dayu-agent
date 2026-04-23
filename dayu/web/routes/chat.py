"""Chat 操作端点。"""

from __future__ import annotations

import asyncio
from typing import Any

from dayu.contracts.events import AppEventType
from dayu.log import Log
from dayu.services.contracts import ChatTurnRequest, ChatTurnSubmission, ReplyDeliverySubmitRequest
from dayu.services.protocols import ChatServiceProtocol, ReplyDeliveryServiceProtocol

MODULE = "WEB.CHAT"


class _InvalidChatSubmissionError(Exception):
    """ChatService 返回非法提交句柄时抛出的内部异常。"""


def _require_chat_submission(submission: object) -> ChatTurnSubmission:
    """验证 Web chat 路由收到的 Service 提交句柄。

    Args:
        submission: ChatService 返回对象。

    Returns:
        通过验证的 `ChatTurnSubmission`。

    Raises:
        _InvalidChatSubmissionError: 返回值不是稳定 DTO，或缺少可消费事件流时抛出。
    """

    if not isinstance(submission, ChatTurnSubmission):
        raise _InvalidChatSubmissionError("chat service returned invalid submission")
    if not submission.session_id.strip():
        raise _InvalidChatSubmissionError("chat service returned empty session_id")
    event_stream: Any = submission.event_stream
    if not hasattr(event_stream, "__aiter__"):
        raise _InvalidChatSubmissionError("chat service returned non-stream event_stream")
    return submission


def create_chat_router(
    chat_service: ChatServiceProtocol,
    reply_delivery_service: ReplyDeliveryServiceProtocol,
):
    """创建 chat 路由。"""

    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    router = APIRouter(prefix="/api", tags=["chat"])

    class ChatRequest(BaseModel):
        """Chat turn 请求体。"""

        user_text: str
        ticker: str | None = None
        scene_name: str | None = None
        session_id: str | None = None

    class ChatResponse(BaseModel):
        """Chat turn 响应（异步模式，返回 session 句柄）。"""

        session_id: str
        accepted: bool = True

    @router.post("/chat", response_model=ChatResponse, status_code=202)
    async def submit_chat_turn(body: ChatRequest) -> ChatResponse:
        """提交 chat turn，结果通过 SSE 推送。"""

        request = ChatTurnRequest(
            session_id=body.session_id,
            user_text=body.user_text,
            ticker=body.ticker,
            scene_name=body.scene_name,
        )
        try:
            submission = await chat_service.submit_turn(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            validated_submission = _require_chat_submission(submission)
        except _InvalidChatSubmissionError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        asyncio.create_task(
            _consume_stream(
                validated_submission.event_stream,
                reply_delivery_service=reply_delivery_service,
                session_id=validated_submission.session_id,
                scene_name=body.scene_name,
            )
        )
        return ChatResponse(session_id=validated_submission.session_id)

    return router


async def _consume_stream(
    stream,
    *,
    reply_delivery_service: ReplyDeliveryServiceProtocol,
    session_id: str,
    scene_name: str | None,
) -> None:
    """后台消费流式事件，并在终态后显式提交 reply outbox。

    Args:
        stream: ChatService 提交后返回的事件流。
        reply_delivery_service: 投递 reply outbox 的 Service。
        session_id: 当前 chat 的 session ID。
        scene_name: 当前 chat 的 scene 名（缺省视为 ``"interactive"``）。

    Returns:
        无。

    Raises:
        无：所有内部异常被记录后吞掉，避免 asyncio Task 静默存储异常导致
        线上问题难以定位。
    """

    try:
        await _consume_stream_inner(
            stream,
            reply_delivery_service=reply_delivery_service,
            session_id=session_id,
            scene_name=scene_name,
        )
    except Exception as exc:
        Log.error(
            f"chat 后台流消费异常: session={session_id}, error={exc}",
            module=MODULE,
        )


async def _consume_stream_inner(
    stream,
    *,
    reply_delivery_service: ReplyDeliveryServiceProtocol,
    session_id: str,
    scene_name: str | None,
) -> None:
    """实际的事件消费与 reply 提交逻辑，便于统一异常包裹。"""

    content_chunks: list[str] = []
    final_reply_content = ""
    final_reply_filtered = False
    cancelled = False
    source_run_id = ""
    actual_scene_name = str(scene_name or "").strip() or "interactive"

    async for event in stream:
        if event.type == AppEventType.CONTENT_DELTA:
            text = str(event.payload or "")
            if text:
                content_chunks.append(text)
        if event.type == AppEventType.FINAL_ANSWER:
            payload = event.payload if isinstance(event.payload, dict) else {"content": str(event.payload)}
            final_reply_content = str(payload.get("content") or "").strip()
            final_reply_filtered = bool(payload.get("filtered", False))
        if event.type == AppEventType.CANCELLED:
            cancelled = True
        event_run_id = str(event.meta.get("run_id") or "").strip() if isinstance(event.meta, dict) else ""
        if event_run_id:
            source_run_id = event_run_id

    if cancelled:
        return

    reply_content = final_reply_content or "".join(content_chunks).strip()
    if not reply_content or not source_run_id:
        return

    reply_delivery_service.submit_reply_for_delivery(
        ReplyDeliverySubmitRequest(
            delivery_key=f"web:{source_run_id}",
            session_id=session_id,
            scene_name=actual_scene_name,
            source_run_id=source_run_id,
            reply_content=reply_content,
            metadata={
                "delivery_channel": "web",
                "delivery_target": session_id,
                "delivery_thread_id": session_id,
                "filtered": final_reply_filtered,
            },
        )
    )


__all__ = ["create_chat_router"]
