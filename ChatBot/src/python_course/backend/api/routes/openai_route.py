from collections.abc import AsyncGenerator
from enum import StrEnum

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from python_course.backend.service.chat_service import (
    ChatCompletionResponse,
    ChatService,
    OpenAIModel,
)


class MessageRole(StrEnum):
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"


class ChatMessage(BaseModel):
    role: MessageRole = Field(description="The role of the message sender")
    content: str = Field(
        description="The content of the message",
        min_length=1,
        max_length=4096,
    )


class ChatCompletionRequest(BaseModel):
    model: OpenAIModel = Field(
        description="The OpenAI model to use for completion",
        default=OpenAIModel.GPT_4O,
    )
    messages: list[ChatMessage] = Field(
        description="List of messages in the conversation",
        min_length=1,
    )
    stream: bool = Field(description="Whether to stream the response", default=False)


# Create router instance
router = APIRouter(prefix="/openai", tags=["openai"])


@router.post(
    "/chat",
    response_model=ChatCompletionResponse,
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "example": "data: AI response chunk\n\n",
                }
            },
        }
    },
)
async def create_chat_completion(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    try:
        chat_service = ChatService(model=request.model)

        # Convert Pydantic models to dictionaries
        messages_dict = [msg.model_dump() for msg in request.messages]

        if request.stream:
            return StreamingResponse(
                format_sse_data(chat_service.stream_chat(messages_dict)),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        return await chat_service.non_stream_chat(messages_dict)

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def format_sse_data(stream: AsyncGenerator[str]) -> AsyncGenerator[str]:
    """Format streaming data as SSE."""
    try:
        async for chunk in stream:
            if chunk:
                yield f"data: {chunk}\n\n"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
