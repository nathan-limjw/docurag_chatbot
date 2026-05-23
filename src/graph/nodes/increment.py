from src.graph.state import AgentState
from utils.logger import get_logger

logger = get_logger(__name__)


def increment_retry(state: AgentState) -> AgentState:
    logger.info("[INCREMENTING RETRY NODE] Trying again...")
    return {
        "retry_count": state.get("retry_count", 0) + 1,
        "answer": "",
    }
