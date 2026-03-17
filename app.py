from datetime import datetime
import json
from pathlib import Path
import time
from uuid import uuid4

import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


API_URL = "https://router.huggingface.co/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path("chats")


def load_hf_token():
    """Return the Hugging Face token or None if it is missing."""
    try:
        token = st.secrets.get("HF_TOKEN", "")
    except StreamlitSecretNotFoundError:
        return None

    token = token.strip() if token else ""
    return token or None


def make_chat_title(messages: list[dict]) -> str:
    """Generate a simple title from the first user message."""
    for message in messages:
        if message["role"] == "user":
            content = message["content"].strip()
            return content[:30] + ("..." if len(content) > 30 else "")
    return "New Chat"


def make_new_chat():
    """Create a new empty chat record."""
    now = datetime.now()
    return {
        "id": str(uuid4()),
        "title": "New Chat",
        "created_at": now.isoformat(timespec="seconds"),
        "timestamp_label": now.strftime("%b %d, %Y %I:%M %p"),
        "messages": [],
    }


def get_chat_file_path(chat_id: str) -> Path:
    """Return the JSON file path for a chat."""
    return CHATS_DIR / f"{chat_id}.json"


def save_chat(chat: dict):
    """Persist one chat to its JSON file."""
    CHATS_DIR.mkdir(exist_ok=True)
    chat_payload = {
        "id": chat["id"],
        "title": chat["title"],
        "created_at": chat["created_at"],
        "timestamp_label": chat["timestamp_label"],
        "messages": chat["messages"],
    }
    get_chat_file_path(chat["id"]).write_text(json.dumps(chat_payload, indent=2), encoding="utf-8")


def load_chats_from_disk() -> list[dict]:
    """Load all saved chats from the chats directory."""
    CHATS_DIR.mkdir(exist_ok=True)
    chats = []

    for chat_file in sorted(CHATS_DIR.glob("*.json")):
        try:
            chat = json.loads(chat_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not {"id", "title", "created_at", "messages"}.issubset(chat):
            continue

        created_at = chat["created_at"]
        timestamp_label = chat.get("timestamp_label")
        if not timestamp_label:
            try:
                parsed_time = datetime.fromisoformat(created_at)
                timestamp_label = parsed_time.strftime("%b %d, %Y %I:%M %p")
            except ValueError:
                timestamp_label = created_at

        chat["timestamp_label"] = timestamp_label
        chats.append(chat)

    chats.sort(key=lambda chat: chat.get("created_at", ""), reverse=True)
    return chats


def initialize_session_state():
    """Set up chat state once per browser session."""
    if "chats" not in st.session_state:
        saved_chats = load_chats_from_disk()
        if saved_chats:
            st.session_state.chats = saved_chats
            st.session_state.active_chat_id = saved_chats[0]["id"]
        else:
            first_chat = make_new_chat()
            st.session_state.chats = [first_chat]
            st.session_state.active_chat_id = first_chat["id"]
            save_chat(first_chat)


def get_active_chat():
    """Return the active chat record."""
    active_chat_id = st.session_state.get("active_chat_id")
    for chat in st.session_state.chats:
        if chat["id"] == active_chat_id:
            return chat

    if st.session_state.chats:
        st.session_state.active_chat_id = st.session_state.chats[0]["id"]
        return st.session_state.chats[0]

    return None


def create_new_chat():
    """Add a fresh conversation and make it active."""
    new_chat = make_new_chat()
    st.session_state.chats.insert(0, new_chat)
    st.session_state.active_chat_id = new_chat["id"]
    save_chat(new_chat)


def delete_chat(chat_id: str):
    """Delete a chat and choose a fallback active chat if needed."""
    remaining_chats = [chat for chat in st.session_state.chats if chat["id"] != chat_id]
    st.session_state.chats = remaining_chats
    chat_file = get_chat_file_path(chat_id)
    if chat_file.exists():
        chat_file.unlink()

    if not remaining_chats:
        create_new_chat()
        return

    if st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = remaining_chats[0]["id"]


def stream_chat_response(hf_token: str, messages: list[dict]):
    """Yield streamed response chunks from the Hugging Face router."""
    headers = {"Authorization": f"Bearer {hf_token}"}
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 512,
        "stream": True,
    }

    with requests.post(
        API_URL,
        headers=headers,
        json=payload,
        timeout=30,
        stream=True,
    ) as response:
        response.raise_for_status()

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            line = raw_line.strip()
            if not line.startswith("data:"):
                continue

            data_str = line.removeprefix("data:").strip()
            if data_str == "[DONE]":
                break

            chunk_data = json.loads(data_str)
            choices = chunk_data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                time.sleep(0.02)
                yield content


def show_api_error(error: Exception):
    """Convert API failures into user-visible Streamlit messages."""
    if isinstance(error, requests.exceptions.HTTPError):
        status_code = error.response.status_code if error.response is not None else "unknown"
        if status_code == 401:
            st.error("The Hugging Face token appears to be invalid or unauthorized.")
        elif status_code == 429:
            st.error("The Hugging Face API rate limit was reached. Please try again shortly.")
        else:
            st.error(f"Hugging Face API error: HTTP {status_code}.")
    elif isinstance(error, requests.exceptions.Timeout):
        st.error("The Hugging Face API timed out. Please try again.")
    elif isinstance(error, requests.exceptions.RequestException):
        st.error("A network error occurred while contacting the Hugging Face API.")
    else:
        st.error("The API returned an unexpected response format.")


def render_sidebar():
    """Render chat creation, navigation, and deletion controls."""
    with st.sidebar:
        st.header("Chats")
        if st.button("New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()

        chat_list = st.container(height=500)
        with chat_list:
            for chat in st.session_state.chats:
                is_active = chat["id"] == st.session_state.active_chat_id
                entry_container = st.container(border=True)
                with entry_container:
                    nav_col, delete_col = st.columns([5, 1], vertical_alignment="top")

                    with nav_col:
                        if st.button(
                            chat["title"],
                            key=f"open_{chat['id']}",
                            use_container_width=True,
                            type="primary" if is_active else "secondary",
                        ):
                            st.session_state.active_chat_id = chat["id"]
                            st.rerun()
                        st.caption(chat["timestamp_label"])

                    with delete_col:
                        if st.button("✕", key=f"delete_{chat['id']}"):
                            delete_chat(chat["id"])
                            st.rerun()


def render_chat_history(active_chat):
    """Display the active conversation above the fixed input."""
    history_container = st.container(height=500)
    with history_container:
        if not active_chat["messages"]:
            st.info("This chat is empty. Send a message to get started.")

        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])


def main():
    st.set_page_config(page_title="My AI Chat", layout="wide")
    st.title("My AI Chat")
    st.caption(f"Chatting with `{MODEL_NAME}`")

    initialize_session_state()
    render_sidebar()

    active_chat = get_active_chat()
    render_chat_history(active_chat)

    hf_token = load_hf_token()
    if not hf_token:
        st.error(
            "Missing Hugging Face token. Add HF_TOKEN to .streamlit/secrets.toml "
            "before running the app."
        )
        st.chat_input("Type your message here...", disabled=True)
        return

    prompt = st.chat_input("Type your message here...")
    if not prompt:
        return

    user_message = {"role": "user", "content": prompt}
    active_chat["messages"].append(user_message)
    active_chat["title"] = make_chat_title(active_chat["messages"])
    save_chat(active_chat)

    with st.chat_message("user"):
        st.write(prompt)

    try:
        with st.chat_message("assistant"):
            assistant_reply = st.write_stream(
                stream_chat_response(hf_token, active_chat["messages"])
            )
    except (requests.exceptions.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        show_api_error(exc)
        return

    assistant_message = {"role": "assistant", "content": assistant_reply}
    active_chat["messages"].append(assistant_message)
    save_chat(active_chat)
    st.rerun()


if __name__ == "__main__":
    main()
