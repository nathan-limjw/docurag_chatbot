from src.config import settings
from src.database.reranker import rerank
from src.database.store import retrieve
from src.graph.state import AgentState


def retrieve_node(state: AgentState) -> AgentState:
    query = state["query"]

    candidates = retrieve(
        query, k=settings.retrieval_k
    )  # ChromaDB vector similarity search (overfetches, broad, fast)

    top_docs = rerank(
        query, candidates, top_n=settings.rerank_top_n
    )  # Cross-encoder reranking (precise and runs on small candidate doc set)

    return {
        "retrieved_docs": candidates,
        "reranked_docs": top_docs,
    }
