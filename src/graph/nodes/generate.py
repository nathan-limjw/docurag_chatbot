from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import AgentState
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are a helpful assistant that answers questions based STRICTLY on the provided context documents

Rules:

1. Answer ONLY using information from the provided context. DO NOT USE outside knowledge
2. If the context does not contain enough information to answer, say: "I don't have enough information in the provided documents to answer that."
3. Be concise but complete. Use bullet points or numbered lists where it helps with clarity
4. NEVER fabricate facts, statistics or sources

"""


def _build_context(state: AgentState) -> str:
    docs = state.get("reranked_docs", [])
    if not docs:
        return "No relevant documents found."

    parts = []
    total_chars = 0
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        entry = f"[Document {i} | Source: {source}]\n{doc.page_content.strip()}"
        if total_chars + len(entry) > settings.max_context_chars:
            break
        parts.append(entry)
        total_chars += len(entry)

    return "\n\n---\n\n".join(parts)


def generate_node(state: AgentState) -> AgentState:
    logger.info(
        f"[GENERATION NODE] Attempt {state.get('retry_count', 0) + 1}: Generating answer..."
    )

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )

    context = _build_context(state)

    logger.debug(f"""
    - docs = {len(state.get("reranked_docs", []))}
    - context_chas = {len(context)}
    """)

    history = state.get("messages", [])[-settings.max_history_turns :]

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(history)

    messages.append(
        HumanMessage(
            content=(
                f"Context documents:\n{context}\n\n"
                f"---\n\n"
                f"Answer the question above based strictly on the context. "
                f"Question: {state['query']}"
            )
        )
    )

    response = llm.invoke(messages)
    logger.debug(f"LLM Response: {response.content.strip()}")

    ai_message = AIMessage(content=response.content.strip())

    return {
        "answer": ai_message.content,
        "messages": [ai_message],
    }
