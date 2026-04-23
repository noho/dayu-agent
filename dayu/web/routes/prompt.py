"""Prompt 操作端点。"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from dayu.contracts.events import AppEvent, AppEventType
from dayu.log import Log
from dayu.services.protocols import PromptServiceProtocol

MODULE = "WEB.PROMPT"


def create_prompt_router(prompt_service: PromptServiceProtocol):
    """创建 prompt 路由。

    Args:
        无。

    Returns:
        FastAPI 路由对象。

    Raises:
        无。
    """

    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel

    from dayu.services.contracts import PromptRequest

    router = APIRouter(prefix="/api", tags=["prompt"])

    class PromptRequestBody(BaseModel):
        """Prompt 请求体。"""

        user_text: str
        ticker: str | None = None

    class PromptResponse(BaseModel):
        """Prompt 响应（异步模式)。"""

        session_id: str
        accepted: bool = True

    @router.post("/prompt", response_model=PromptResponse, status_code=202)
    async def submit_prompt(body: PromptRequestBody) -> PromptResponse:
        """提交 prompt，结果通过 SSE 推送。

        Args:
            body: 请求体。

        Returns:
            可订阅的 session 句柄。

        Raises:
            无。
        """

        try:
            submission = await prompt_service.submit(
                PromptRequest(
                    user_text=body.user_text,
                    ticker=body.ticker,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        asyncio.create_task(
            _consume_stream(submission.event_stream, session_id=submission.session_id)
        )
        return PromptResponse(session_id=submission.session_id)

    return router


async def _consume_stream(stream: AsyncIterator[AppEvent], *, session_id: str) -> None:
    """后台消费流式事件。

    Args:
        stream: 事件流句柄。
        session_id: 当前 prompt 的 session ID，用于关联日志。

    Returns:
        无。

    Raises:
        无：所有异常被记录后吞掉，避免污染 asyncio 任务退出原因。
    """

    try:
        async for event in stream:
            # 关键终态事件统一落日志，便于线上排查 prompt 执行结果；
            # 中间增量事件（CONTENT_DELTA 等）继续静默丢弃。
            if event.type in (
                AppEventType.FINAL_ANSWER,
                AppEventType.ERROR,
                AppEventType.CANCELLED,
            ):
                Log.info(
                    f"prompt 后台流事件: session={session_id}, type={event.type.value}",
                    module=MODULE,
                )
    except Exception as exc:
        Log.error(
            f"prompt 后台流消费异常: session={session_id}, error={exc}",
            module=MODULE,
        )


__all__ = ["create_prompt_router"]
