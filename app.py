from backend import (
    extract_text,
    get_all_threads,
    ingest_rag_doc,
    get_state,
    get_pending_interrupt,
    stream_chat,
    stream_resume,
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import streamlit as st
import uuid
import os
import tempfile


def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["pending_interrupt"] = None
    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):
    state = get_state(thread_id)
    messages = state.values.get("messages", [])

    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": extract_text(msg.content)})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": extract_text(msg.content)})
    return history


def get_thread_title(thread_id):
    title = st.session_state["thread_titles"].get(thread_id)
    if title:
        return title
    return thread_id[:8]


def run_stream(get_stream_iterable, current_thread):
    """Renders tool activity + streams the assistant's text for one graph
    turn, whether that turn is a fresh message (stream_chat) or a resume
    after an interrupt (stream_resume). Then checks whether the graph is
    now paused again (e.g. a second approval-gated tool call)."""

    with st.chat_message("assistant"):
        tool_status = st.status("Thinking...", expanded=True)
        seen_tool_calls = set()

        def stream_response():
            for message_chunk, metadata in get_stream_iterable():
                if isinstance(message_chunk, AIMessage) and message_chunk.tool_calls:
                    for call in message_chunk.tool_calls:
                        call_id = call.get("id")
                        if call_id and call_id not in seen_tool_calls:
                            seen_tool_calls.add(call_id)
                            tool_status.update(label=f"🔧 Using tool: {call['name']}", state="running")
                            tool_status.write(f"Calling **{call['name']}** with `{call['args']}`")

                elif isinstance(message_chunk, ToolMessage):
                    result_preview = extract_text(message_chunk.content)[:300]
                    tool_status.write(f"✅ **{message_chunk.name}** returned: {result_preview}")

                elif isinstance(message_chunk, AIMessage):
                    text_piece = extract_text(message_chunk.content)
                    if text_piece:
                        yield text_piece

        ai_message = st.write_stream(stream_response)

        if seen_tool_calls:
            tool_status.update(label="Done", state="complete", expanded=False)
        else:
            tool_status.update(label="Answered directly", state="complete", expanded=False)

    if ai_message:
        st.session_state["message_history"].append({
            "role": "assistant",
            "content": ai_message
        })

    # The graph may pause again immediately (e.g. another approval-gated
    # tool call), so check right after every stream.
    interrupt_payload = get_pending_interrupt(current_thread)
    if interrupt_payload:
        st.session_state["pending_interrupt"] = {
            "thread_id": current_thread,
            "data": interrupt_payload,
        }
    else:
        st.session_state["pending_interrupt"] = None


def render_pending_interrupt():
    """Render an approve/reject card for the interrupt paused on the
    current thread, if any."""
    pending = st.session_state.get("pending_interrupt")
    if not pending or pending["thread_id"] != st.session_state["thread_id"]:
        return

    data = pending["data"]

    with st.chat_message("assistant"):
        st.warning(data.get("message", "Approval required before continuing."))

        col1, col2 = st.columns(2)
        approve_clicked = col1.button("✅ Approve", key=f"approve_{pending['thread_id']}")
        reject_clicked = col2.button("❌ Reject", key=f"reject_{pending['thread_id']}")

        if approve_clicked or reject_clicked:
            current_thread = pending["thread_id"]
            resume_payload = {"approved": approve_clicked}
            run_stream(lambda: stream_resume(current_thread, resume_payload), current_thread)
            st.rerun()


st.title("Kaleem's agentic ai")

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = []

if "pending_interrupt" not in st.session_state:
    st.session_state["pending_interrupt"] = None

add_thread(st.session_state["thread_id"])

st.sidebar.title("All the conversations")

if st.sidebar.button("new chat"):
    reset_chat()
    st.rerun()

st.sidebar.markdown("---")

for thread_id in st.session_state["chat_threads"][::-1]:
    label = get_thread_title(thread_id)
    if st.sidebar.button(label, key=thread_id):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)
        # Restore any interrupt paused on this thread from a previous session.
        interrupt_payload = get_pending_interrupt(thread_id)
        st.session_state["pending_interrupt"] = (
            {"thread_id": thread_id, "data": interrupt_payload} if interrupt_payload else None
        )
        st.rerun()

if st.session_state["ingested_docs"]:
    st.sidebar.markdown("---")
    with st.sidebar.expander(f"📄 Ingested documents ({len(st.session_state['ingested_docs'])})"):
        for name in st.session_state["ingested_docs"]:
            st.write(f"• {name}")

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# If the graph is paused on interrupt(), show the approval card instead of
# (or above) the normal chat input.
render_pending_interrupt()

has_pending = st.session_state["pending_interrupt"] is not None

prompt = st.chat_input(
    "Waiting for your approval above..." if has_pending else "Type here",
    accept_file=True,
    file_type=["pdf"],
    disabled=has_pending,
)

if prompt and not has_pending:
    user_input = prompt.text
    uploaded_files = prompt.files  # list of UploadedFile objects, empty if none attached

    current_thread = st.session_state["thread_id"]

    # Ingest any attached PDFs first, before sending the message to the agent
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state["ingested_docs"]:
                with st.status(f"Ingesting {uploaded_file.name}...", expanded=False) as status:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name

                    try:
                        result = ingest_rag_doc(tmp_path)
                        st.session_state["ingested_docs"].append(uploaded_file.name)
                        status.update(label=f"✅ {uploaded_file.name} ingested", state="complete")
                    except Exception as e:
                        status.update(label=f"❌ Failed to ingest {uploaded_file.name}", state="error")
                        st.error(str(e))
                    finally:
                        os.remove(tmp_path)

    # Only proceed to the chat turn if there's actual text
    if user_input:
        if current_thread not in st.session_state["thread_titles"]:
            title = user_input[:40] + ("..." if len(user_input) > 40 else "")
            st.session_state["thread_titles"][current_thread] = title

        st.session_state['message_history'].append({'role': 'user', 'content': user_input})

        with st.chat_message("user"):
            st.text(user_input)

        run_stream(lambda: stream_chat(current_thread, user_input), current_thread)

        if st.session_state["pending_interrupt"] is not None:
            st.rerun()