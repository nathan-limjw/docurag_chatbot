from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.graph.state import AgentState
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnswerVerdict(BaseModel):
    is_grounded: bool = Field(
        description="True if every claim in the answer can be traced to the provided context. False if the answer contains anything not found in the context."
    )
    is_useful: bool = Field(
        description="True if the answer substantively addresses the question. False if the answer is a fallback such as 'I don't have enough information'."
    )


JUDGE_SYSTEM_PROMPT = """You are a strict answer evaluator. 
Given a context, a question, and an answer, evaluate two things and return them as structured output:

1. is_grounded: Is every claim in the answer supported by the provided context?
2. is_useful: Does the answer substantively address the question, or is it a fallback like "I don't have enough information"?
"""


def _build_context(docs: list) -> str:
    return (
        "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)
        or "No context available."
    )


def _build_judge_prompt(query: str, context: str, answer: str) -> str:
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer: {answer}"


def validate_node(state: AgentState) -> AgentState:
    logger.info("[VALIDATION NODE] Validating answer...")

    docs = state.get("reranked_docs", [])

    if not docs:
        return {"is_grounded": False, "is_useful": False}

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.openai_api_key,
    ).with_structured_output(AnswerVerdict)

    verdict: AnswerVerdict = llm.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                content=_build_judge_prompt(
                    state["query"],
                    _build_context(docs),
                    state.get("answer", ""),
                )
            ),
        ]
    )
    logger.debug(f"Verdict: {verdict}")

    return {
        "is_grounded": verdict.is_grounded,
        "is_useful": verdict.is_useful,
    }
