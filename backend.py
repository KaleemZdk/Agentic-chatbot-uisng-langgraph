from langgraph.graph import StateGraph, START, END
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from typing import Literal, TypedDict, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage,AIMessage
from langgraph.graph.message import add_messages
import os
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import requests
import math 
from langchain_core.tools import tool 
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_tavily import TavilySearch
from tavily import TavilyClient
load_dotenv()
import ast
import operator
import smtplib
from email.mime.text import MIMEText
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langgraph.types import interrupt, Command
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.5-flash-lite",
#     google_api_key=os.getenv("GEMINI_API_KEY")
# )


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# Tool names that pause the graph via interrupt() and need a human decision
# before they can finish. The frontend can use this to special-case UI, and
# it's a single place to register future approval-gated tools.
APPROVAL_REQUIRED_TOOLS = {"purchase_stock"}


@tool
def purchase_stock(symbol: str, quantity: int, price: float) -> str:
    """Purchase a given quantity of stock at a specified price. Requires human approval before executing."""

    decision = interrupt({
        "action": "purchase_stock",
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "total_cost": quantity * price,
        "message": f"Approve purchase of {quantity} shares of {symbol} at ${price} each (total: ${quantity * price:.2f})?"
    })

    if decision.get("approved"):
        # Prototype logic — replace with real brokerage API call later
        return f"✅ Purchased {quantity} shares of {symbol} at ${price} each. Total: ${quantity * price:.2f}"
    else:
        return f"❌ Purchase of {symbol} was rejected by the user."


def ingest_rag_doc(file_path, save_path="faiss_index"):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    return f"Ingested {len(chunks)} chunks from {file_path} into '{save_path}'."


def load_vector_store(path="faiss_index"):
    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store


def get_retriever(path="faiss_index", k=4):
    vector_store = load_vector_store(path)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever


@tool
def rag_search(query: str) -> str:
    """Search the ingested documents (PDFs you've added to the knowledge base)
    for information relevant to the query. Use this when the user asks about
    content from their uploaded documents rather than general knowledge."""

    try:
        retriever = get_retriever()
    except Exception as e:
        return f"Could not load the document index — has a document been ingested yet? ({e})"

    try:
        results = retriever.invoke(query)
    except Exception as e:
        return f"Search failed: {e}"

    if not results:
        return "No relevant information found in the ingested documents."

    out = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        page_info = f" (page {page})" if page != "" else ""
        out.append(f"[{i}] Source: {source}{page_info}\n{doc.page_content.strip()}\n")

    return "\n----\n".join(out)
# resp = requests.get(
#     "https://api.groq.com/openai/v1/models",
#     headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"}
# )
# for m in resp.json()["data"]:
#     print(m["id"])

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
 
 
#email tool 
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient. Use this when the user explicitly asks
    to send, email, or notify someone. Requires 'to' (recipient address),
    'subject', and 'body' (the message content)."""
 
    sender_email = os.getenv("SMTP_EMAIL")
    sender_password = os.getenv("SMTP_PASSWORD")  # use an app password, not your real password
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
 
    if not sender_email or not sender_password:
        return "Email not sent: SMTP_EMAIL or SMTP_PASSWORD is missing from your .env file."
 
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to
 
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [to], msg.as_string())
        return f"Email successfully sent to {to}."
    except Exception as e:
        return f"Failed to send email: {e}"
 
 
#Web search 
@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or facts you don't
    already know. Returns the top results with titles, URLs, and snippets."""
 
    try:
        results = tavily.search(query=query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"
 
    out = []
    for r in results.get("results", []):
        out.append(f"Title:{r['title']}\nURL:{r['url']}\nSnippet:{r['content'][:300]}\n")
 
    return "\n----\n".join(out) if out else "No results found."
 
 
#Calculator 
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
 
 
def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported or unsafe expression.")
 
 
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '12 * (3 + 4) / 2'. Supports +, -,
    *, /, **, and %. Does not support raw eval — unsafe expressions are
    rejected."""
 
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Could not evaluate '{expression}': {e}"
 
 
#weather tool 
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Uses Open-Meteo, a free weather
    API that needs no API key."""
 
    try:
        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        )
        geo_resp.raise_for_status()
        geo_data = geo_resp.json()
 
        if not geo_data.get("results"):
            return f"Could not find a location named '{city}'."
 
        location = geo_data["results"][0]
        lat, lon = location["latitude"], location["longitude"]
        resolved_name = f"{location['name']}, {location.get('country', '')}"
 
        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=10,
        )
        weather_resp.raise_for_status()
        current = weather_resp.json()["current_weather"]
 
        return (
            f"Weather in {resolved_name}: {current['temperature']}°C, "
            f"wind {current['windspeed']} km/h."
        )
    except Exception as e:
        return f"Could not fetch weather for {city}: {e}"


tools = [send_email,web_search,calculator,get_weather,rag_search,purchase_stock]
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def extract_text(response_content):
    if isinstance(response_content, str):
        return response_content
    elif isinstance(response_content, list):
        return "".join(
            block.get("text", "") for block in response_content if isinstance(block, dict)
        )
    return str(response_content)


def chat_node(state: ChatState):
    messages = state['messages']

    system_prompt = SystemMessage(content=(
        "You are Kaleem's personal AI assistant. You have access to tools for "
        "web search, sending emails, calculating math, checking weather, "
        "purchasing stock, and searching ingested documents (rag_search). Use "
        "rag_search whenever the user asks about content from documents "
        "they've uploaded or ingested, or references 'the document', 'the "
        "file', 'the PDF', or similar — prefer it over web_search for "
        "anything that sounds like it's about their own material rather "
        "than general/public information. Use a tool whenever the request "
        "requires current information, an action, or document lookup — "
        "otherwise answer directly. "
        "\n\n"
        "purchase_stock requires exactly three arguments: symbol (string), "
        "quantity (a whole number of shares — never null, zero, or a guess), "
        "and price (a number, price per share). Call this tool ONLY when the "
        "user has explicitly given you all three values somewhere in the "
        "conversation. If even one is missing, ambiguous, or ONLY implied, do "
        "NOT call the tool — instead ask the user a plain-text question for "
        "exactly the missing value(s) and wait for their reply. Never invent "
        "or default a quantity or price. Once all three are known, call the "
        "tool — it pauses on its own for the user's approval, so you don't "
        "need to ask for confirmation yourself first. "
        "Be concise and clear."
    ))

    try:
        response = llm_with_tools.invoke([system_prompt] + messages)
    except Exception as e:
        # A malformed/incomplete tool call (e.g. the model tried to call
        # purchase_stock with a missing argument and the provider's schema
        # validation rejected it before we ever saw a normal response) or
        # any other API-level failure. Surface this as a normal assistant
        # message instead of crashing the whole graph run / Streamlit app.
        print(f"[chat_node] llm_with_tools.invoke failed: {e}")
        response = AIMessage(content=(
            "Sorry — I tried to take an action but was missing some required "
            "details (or hit an API error) and couldn't complete it. Could "
            "you give me the missing information and try again? "
            f"(details: {e})"
        ))

    return {'messages': [response]}
 
tool_node = ToolNode(tools)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KlaeemAi.db")
conn = sqlite3.connect(database=DB_PATH, check_same_thread=False)
checkpoint = SqliteSaver(conn)
checkpoint.setup()  # idempotent — creates the checkpoint tables if they don't exist yet

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools',tool_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools','chat_node')

chatbot = graph.compile(checkpointer=checkpoint)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def get_state(thread_id: str):
    """Return the graph's current StateSnapshot for this thread."""
    return chatbot.get_state(_config(thread_id))


def get_pending_interrupt(thread_id: str):
    """If the graph is currently paused on an interrupt() call for this
    thread (e.g. purchase_stock waiting for approval), return its payload
    dict. Returns None if nothing is pending."""
    state = get_state(thread_id)
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def stream_chat(thread_id: str, user_input: str):
    """Stream a fresh user message through the graph. Yields (message_chunk,
    metadata) pairs, same as chatbot.stream(..., stream_mode='messages').
    The stream ends early (mid-turn) if a tool call hits interrupt() —
    call get_pending_interrupt() afterwards to check."""
    return chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=_config(thread_id),
        stream_mode="messages",
    )


def stream_resume(thread_id: str, resume_payload: dict):
    """Resume a paused interrupt() with the human's decision (e.g.
    {'approved': True}) and stream the continuation. Same chunk shape and
    early-stop behavior as stream_chat, in case another approval-gated tool
    is called right after."""
    return chatbot.stream(
        Command(resume=resume_payload),
        config=_config(thread_id),
        stream_mode="messages",
    )


def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])
    return list(all_threads)
