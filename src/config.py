from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str

    api_base_url: str = "http://localhost:8000"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "documents"

    chunk_size: int = 800
    chunk_overlap: int = 100

    retrieval_k: int = 10
    rerank_top_n: int = 4

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_temperature: float = 0.1
    max_context_chars: int = 6000
    max_history_turns: int = 6

    max_retries: int = 2
    grounding_overlap_threshold: float = 0.15

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


@lru_cache
def get_settings():
    return Settings()


settings = get_settings()
