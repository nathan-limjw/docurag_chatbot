from src.graph.state import AgentState
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def route_after_validate(state: AgentState) -> str:
    logger.info("Routing Decision...")

    is_grounded = state.get("is_grounded", False)
    is_useful = state.get("is_useful", False)
    retry_count = state.get("retry_count", 0)

    logger.debug(f"""
                grounded = {is_grounded}
                useful = {is_useful}
                retry_count = {retry_count}
    """)

    if is_grounded and is_useful:
        return "end"

    if retry_count < settings.max_retries:
        return "retry"

    return "end"
