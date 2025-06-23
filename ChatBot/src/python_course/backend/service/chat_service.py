from collections.abc import AsyncGenerator
from enum import StrEnum
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIError, RateLimitError
from pydantic import BaseModel
from requests.exceptions import RequestException


# Define available OpenAI models
class OpenAIModel(StrEnum):
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1 = "gpt-4.1"


# Response model
class ChatCompletionResponse(BaseModel):
    response: str
    model: str
    total_tokens: int | None = None


class ChatService:
    def __init__(self, model: OpenAIModel, temperature: float = 1) -> None:
        self.model = model
        self.temperature = temperature

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, str]],
    ) -> list[SystemMessage | HumanMessage | AIMessage]:
        """Convert API message format to LangChain message format."""
        message_map: dict[str, type[SystemMessage | HumanMessage | AIMessage]] = {
            "system": SystemMessage,
            "human": HumanMessage,
            "ai": AIMessage,
        }
        return [message_map[msg["role"]](content=msg["content"]) for msg in messages]

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncGenerator[str]:
        """Stream chat response."""
        chat = ChatOpenAI(
            model=self.model.value,
            temperature=self.temperature,
        )

        try:
            langchain_messages = self._convert_messages(messages)
            async for chunk in chat.astream(langchain_messages):
                if isinstance(chunk.content, str):
                    yield chunk.content
        except (APIError, RateLimitError, RequestException) as e:
            raise ValueError(f"Chat service error: {e!s}") from e

    async def non_stream_chat(
        self, messages: list[dict[str, str]]
    ) -> ChatCompletionResponse:
        """Get non-streaming chat response."""
        chat = ChatOpenAI(
            model=self.model.value,
            temperature=self.temperature,
        )

        try:
            langchain_messages = self._convert_messages(messages)
            response = chat.invoke(langchain_messages)

            return ChatCompletionResponse(
                response=cast("str", response.content),
                model=self.model.value,
                total_tokens=None,
            )
        except (APIError, RateLimitError, RequestException) as e:
            raise ValueError(f"Chat service error: {e!s}") from e
