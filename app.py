import uuid

import httpx
import streamlit as st

from src.utils.config import settings

API_BASE_URL = settings.api_base_url

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background-color: #0f1117; color: #e2e8f0; }

    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #2d3748;
    }
    [data-testid="stChatMessage"] {
        background-color: #1a2035;
        border-radius: 8px;
        border: 1px solid #2d3748;
        margin-bottom: 8px;
    }
    .source-badge {
        display: inline-block;
        background: #1e3a5f;
        border: 1px solid #2563eb;
        color: #93c5fd;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 2px 4px 2px 0;
    }
    .source-section {
        margin-top: 8px;
        padding-top: 8px;
        border-top: 1px solid #2d3748;
    }
    .status-ok    { color: #34d399; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; }
    .status-empty { color: #f87171; font-size: 0.8rem; font-family: 'IBM Plex Mono', monospace; }
    .thread-id    { color: #4a5568; font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace; word-break: break-all; }
</style>
""",
    unsafe_allow_html=True,
)


if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks_in_store" not in st.session_state:
    st.session_state.chunks_in_store = 0


def check_status() -> dict | None:
    try:
        r = httpx.get(f"{API_BASE_URL}/status", timeout=5)
        data = r.json()
        st.session_state.chunks_in_store = data.get("total_chunks", 0)
        return data
    except Exception:
        return None


def api_ingest_file(file) -> dict:
    try:
        r = httpx.post(
            f"{API_BASE_URL}/ingest/file",
            files={"file": (file.name, file.getvalue(), file.type)},
            timeout=60,
        )
        return r.json()
    except Exception as e:
        return {"detail": str(e)}


def api_ingest_url(url: str) -> dict:
    try:
        r = httpx.post(f"{API_BASE_URL}/ingest/url", json={"url": url}, timeout=60)
        return r.json()
    except Exception as e:
        return {"detail": str(e)}


def api_chat(query: str) -> dict:
    try:
        r = httpx.post(
            f"{API_BASE_URL}/chat",
            json={
                "query": query,
                "thread_id": st.session_state.thread_id,
            },
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        return {"answer": f"Error: {r.json().get('detail', 'unknown')}", "sources": []}
    except Exception as e:
        return {"answer": f"Connection error: {e}", "sources": []}


def api_reset():
    try:
        httpx.delete(f"{API_BASE_URL}/reset", timeout=10)
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.chunks_in_store = 0
    except Exception as e:
        st.error(str(e))


def render_sources(sources: list):
    if not sources:
        return
    st.markdown('<div class="source-section">', unsafe_allow_html=True)
    st.markdown('<small style="color:#4a5568">Sources:</small>', unsafe_allow_html=True)
    for s in sources:
        score = f" · {s['rerank_score']:.3f}" if s.get("rerank_score") else ""
        st.markdown(
            f'<span class="source-badge">{s["source"]}{score}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 🔍 RAG Chatbot")
    st.markdown("---")

    status = check_status()
    chunks = st.session_state.chunks_in_store
    if status:
        if chunks > 0:
            st.markdown(
                f'<p class="status-ok">● {chunks} chunks indexed</p>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<p class="status-empty">● No documents ingested yet</p>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p class="status-empty">● API offline — is uvicorn running?</p>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    st.markdown("### 📄 Upload Document")
    uploaded = st.file_uploader(
        "PDF or text file", type=["pdf", "txt", "md"], label_visibility="collapsed"
    )
    if uploaded and st.button("Ingest File", use_container_width=True):
        with st.spinner("Ingesting..."):
            result = api_ingest_file(uploaded)
        if "chunks_added" in result:
            st.success(f"✓ {result['chunks_added']} chunks added")
            check_status()
        else:
            st.error(result.get("detail", "Ingestion failed"))

    st.markdown("---")

    st.markdown("### 🌐 Ingest URL")
    url_input = st.text_input("Paste a URL", placeholder="https://example.com/docs")
    if st.button("Ingest URL", use_container_width=True):
        if url_input.strip():
            with st.spinner("Scraping and ingesting..."):
                result = api_ingest_url(url_input.strip())
            if "chunks_added" in result:
                st.success(f"✓ {result['chunks_added']} chunks added")
                check_status()
            else:
                st.error(result.get("detail", "Ingestion failed"))
        else:
            st.warning("Please enter a URL")

    st.markdown("---")

    if st.button("🗑 Reset & Clear All", use_container_width=True):
        api_reset()
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#4a5568'>FastAPI · LangGraph · ChromaDB<br>"
        "OpenAI · Cross-Encoder Reranking</small>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="thread-id">thread: {st.session_state.thread_id[:16]}…</p>',
        unsafe_allow_html=True,
    )


st.markdown("### Ask anything about your documents")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))

if prompt := st.chat_input("Ask a question about your documents..."):
    if st.session_state.chunks_in_store == 0:
        st.warning("⚠️ Ingest a document or URL first using the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = api_chat(prompt)
        answer = result.get("answer", "Sorry, something went wrong.")
        sources = result.get("sources", [])
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
