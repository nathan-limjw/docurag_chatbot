from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    retrieved_docs: List[Document]
    reranked_docs: List[Document]
    answer: str
    is_grounded: bool
    retry_count: int
    error: Optional[str]
