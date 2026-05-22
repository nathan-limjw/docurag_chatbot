from src.config import settings
from src.graph.state import AgentState


def route_after_validate(state: AgentState) -> str:
    if state.get("is_grounded", False):
        return "end"
    if state.get("retry_count", 0) < settings.max_retries:
        return "retry"
    return "end"


def increment_retry(state: AgentState) -> AgentState:
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "answer": "",
    }
