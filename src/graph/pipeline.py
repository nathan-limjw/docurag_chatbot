from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.graph.nodes.generate import generate_node
from src.graph.nodes.increment import increment_retry
from src.graph.nodes.retrieve import retrieve_node
from src.graph.nodes.validate import validate_node
from src.graph.routing import route_after_validate
from src.graph.state import AgentState


def build_pipeline():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("validate", validate_node)
    graph.add_node("increment_retry", increment_retry)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")

    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"end": END, "retry": "increment_retry"},
    )

    graph.add_edge("increment_retry", "generate")

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def run_pipeline(query: str, thread_id: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}

    initial_input = {
        "query": query,
        "messages": [HumanMessage(content=query)],
        "retrieved_docs": [],
        "reranked_docs": [],
        "answer": "",
        "is_grounded": False,
        "retry_count": 0,
        "error": None,
    }

    final_state = get_pipeline().invoke(initial_input, config=config)

    seen = set()
    sources = []
    for doc in final_state.get("reranked_docs", []):
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(
                {
                    "source": src,
                    "page": doc.metadata.get("page"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                }
            )

    return {
        "answer": final_state.get("answer") or "Sorry, I could not generate an answer.",
        "sources": sources,
        "retry_count": final_state.get("retry_count", 0),
        "is_grounded": final_state.get("is_grounded", False),
    }
