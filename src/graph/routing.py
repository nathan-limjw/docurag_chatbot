from src.graph.state import AgentState
from utils.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def route_after_validate(state: AgentState) -> str:
    logger.info("Routing Decision...")
    logger.debug(f"grounded = {state.get('is_grounded')}")

    if state.get("is_grounded", False):
        return "end"
    if state.get("retry_count", 0) < settings.max_retries:
        return "retry"
    return "end"
