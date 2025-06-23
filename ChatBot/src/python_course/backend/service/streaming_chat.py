"""Simple terminal chat using LangChain and OpenAI."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from openai import APIError, RateLimitError
from requests.exceptions import RequestException
from rich.markdown import Markdown

from python_course.core import console


def main() -> None:
    # Initialize the chat model with streaming enabled
    chat = ChatOpenAI(
        model="o4-mini",
        temperature=1,
        streaming=True,
    )

    console.print(Markdown("# Welcome to Terminal Chat!\n"))
    console.print("Type 'exit' to end the conversation.\n")

    # Store conversation history
    chat_history: list[AIMessage | HumanMessage] = []

    while True:
        # Get user input
        user_input = console.input("[bold green]You:[/bold green] ")

        if user_input.lower() == "exit":
            console.print("\nGoodbye! 👋")
            break

        # Add user message to history
        chat_history.append(HumanMessage(content=user_input))

        try:
            console.print("\n[bold blue]Assistant:[/bold blue]", style="bold blue")
            full_response = ""

            # Stream the response
            for chunk in chat.stream(chat_history):
                if isinstance(chunk.content, str):  # Add type check
                    console.print(chunk.content, end="")
                    full_response += chunk.content

            # Add complete response to history
            chat_history.append(AIMessage(content=full_response))
            console.print("\n")  # Empty line for better readability

        except (APIError, RateLimitError) as e:
            console.print(f"\n[bold red]OpenAI API Error:[/bold red] {e!s}\n")
            continue
        except RequestException as e:
            console.print(f"\n[bold red]Network Error:[/bold red] {e!s}\n")
            continue
        except ValueError as e:
            console.print(f"\n[bold red]Value Error:[/bold red] {e!s}\n")
            continue


if __name__ == "__main__":
    main()
