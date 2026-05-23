from src.database.reranker import rerank
from src.database.store import retrieve
from src.graph.state import AgentState
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_node(state: AgentState) -> AgentState:
    logger.info("[RETRIEVAL NODE] Retrieving documents for query...")
    query = state["query"]

    candidates = retrieve(
        query, k=settings.retrieval_k
    )  # ChromaDB vector similarity search (overfetches, broad, fast)
    logger.info(f"Retrieved {len(candidates)} candidate documents...")

    top_docs = rerank(
        query, candidates, top_n=settings.rerank_top_n
    )  # Cross-encoder reranking (precise and runs on small candidate doc set)

    for i, doc in enumerate(top_docs):
        logger.debug(f"""
        Top doc {i + 1}:
        - score = {doc.metadata.get("rerank_score")}
        - source = {doc.metadata.get("source")}
        """)
    logger.info(f"Reranked to find top {len(top_docs)} documents...")

    return {
        "retrieved_docs": candidates,
        "reranked_docs": top_docs,
    }
