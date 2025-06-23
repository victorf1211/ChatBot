import streamlit as st

from python_course import __version__
from python_course.frontend.app.routes import chat, page

VERSION = __version__


def main() -> None:
    st.set_page_config(
        "My App",
        "🪟",
        "wide",
        "expanded",
    )
    st.sidebar.caption(f"Version: <b><u>{VERSION}</u></b>", True)
    my_page = st.Page(page.__file__, title="My Page", icon="🖼️", default=True)
    chat_page = st.Page(chat.__file__, title="Chat", icon="💬")

    st.navigation([my_page, chat_page]).run()


if __name__ == "__main__":
    main()
