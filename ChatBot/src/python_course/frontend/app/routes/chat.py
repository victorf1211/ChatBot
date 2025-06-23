from collections.abc import Iterator
from dataclasses import asdict, dataclass

import requests
import streamlit as st

from python_course.backend.service.chat_service import OpenAIModel
from python_course.core import settings

# Constants
API_BASE_URL = f"http://localhost:{settings.PORT}"
CHAT_ENDPOINT = f"{API_BASE_URL}/openai/chat"


@dataclass
class Message:
    role: str
    content: str


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    if "model" not in st.session_state:
        st.session_state.model = OpenAIModel.GPT_4O.value


def display_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message.role):
            st.write(message.content)


def handle_streaming_response(response: requests.Response) -> Iterator[str]:
    for line in response.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            yield line[6:]  # Remove 'data: ' prefix


def send_message(message: str) -> None:
    # Add user message to chat history
    user_message = Message(role="human", content=message)
    st.session_state.messages.append(user_message)

    # Prepare the chat messages for API
    api_messages = [asdict(msg) for msg in st.session_state.messages]

    # Prepare the request payload
    payload = {
        "messages": api_messages,
        "stream": st.session_state.streaming,
        "model": st.session_state.model,
    }

    # Send request to API
    with st.chat_message("ai"):
        if st.session_state.streaming:
            message_placeholder = st.empty()
            full_response = ""

            # Make streaming request
            with requests.post(
                CHAT_ENDPOINT, json=payload, stream=True, timeout=30
            ) as response:
                response.raise_for_status()
                for chunk in handle_streaming_response(response):
                    full_response += chunk
                    message_placeholder.write(full_response)

            # Add AI response to chat history
            st.session_state.messages.append(Message(role="ai", content=full_response))
        else:
            # Make non-streaming request
            response = requests.post(CHAT_ENDPOINT, json=payload, timeout=30)
            response.raise_for_status()
            ai_response = response.json()["response"]
            st.write(ai_response)

            # Add AI response to chat history
            st.session_state.messages.append(Message(role="ai", content=ai_response))


def main() -> None:
    st.title("聊天機器人")

    # Initialize session state
    initialize_session_state()

    # Create a container for settings
    settings_col1, settings_col2 = st.columns(2)

    # Add model selection dropdown in the first column
    with settings_col1:
        st.radio(
            "選擇模型",
            options=list(OpenAIModel),
            key="model",
            horizontal=True,
        )

    with settings_col2:
        st.toggle("串流回應", key="streaming")

    # Display chat history
    display_chat_history()

    # Chat input
    message = st.chat_input("輸入訊息...")
    if message:
        st.chat_message("human").write(message)
        send_message(message)


if __name__ == "__main__":
    main()
