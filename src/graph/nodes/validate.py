from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.graph.state import AgentState
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a strict grounding checker. 
Your only job is to decide whether an answer is supported by the provided context.

Rules:
- Reply YES if every claim in the answer can be traced back to the context.
- Reply YES if the answer admits it does not have enough information (that is honest and grounded).
- Reply NO if the answer contains any fact, claim, or detail not found in the context.
- Reply with a single word only: YES or NO. No explanation.
"""


def _build_judge_prompt(query: str, context: str, answer: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"Is the answer fully supported by the context? Reply YES or NO:"
    )


def _build_context(docs: list) -> str:
    return (
        "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)
        or "No context available."
    )


def validate_node(state: AgentState) -> AgentState:
    logger.info("[VALIDATION NODE] Validating answer...")

    answer = state.get("answer", "")
    docs = state.get("reranked_docs", [])

    if not docs:
        return {"is_grounded": True}

    context = _build_context(docs)

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        max_tokens=5,
        api_key=settings.openai_api_key,
    )

    response = llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=_build_judge_prompt(state["query"], context, answer)),
        ]
    )

    verdict = response.content.strip().upper()
    is_grounded = verdict.startswith("YES")

    logger.info(f"Validation verdict = {verdict}")

    return {"is_grounded": is_grounded}
