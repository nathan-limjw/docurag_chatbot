from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import settings

_encoder: CrossEncoder | None = None


def _get_encoder() -> CrossEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CrossEncoder(settings.reranker_model)
    return _encoder


def rerank(
    query: str, documents: List[Document], top_n: int | None = None
) -> List[Document]:
    if not documents:
        return []

    n = top_n or settings.rerank_top_n
    encoder = _get_encoder()
    scores = encoder.predict([(query, doc.page_content) for doc in documents])

    ranked = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)

    results = []
    for score, doc in ranked[:n]:
        doc.metadata["rerank_score"] = round(float(score), 4)
        results.append(doc)

    return results
