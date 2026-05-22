import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# Request Format


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: MessageRole
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


class IngestURLRequest(BaseModel):
    url: str = Field(..., description="A publicly accessible URL to scrape and ingest")


# Response Format


class SourceReference(BaseModel):
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    rerank_score: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceReference] = Field(default_factory=list)
    retry_count: int = 0
    is_grounded: bool = True
    thread_id: str


class IngestResponse(BaseModel):
    message: str
    chunks_added: int
    source: str


class StatusResponse(BaseModel):
    status: str
    total_chunks: int
    collection: str
