import uuid
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


# Embeddings and Vector Store functions
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


# Text Splitter function


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# Document Loader Functions
def load_pdf(file_path: str) -> List[Document]:
    return PyPDFLoader(file_path).load()


def load_text(file_path: str) -> List[Document]:
    return TextLoader(file_path, encoding="utf-8").load()


def load_url(url: str) -> List[Document]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    lines = [
        l for l in soup.get_text(separator="\n", strip=True).splitlines() if l.strip()
    ]
    return [Document(page_content="\n".join(lines), metadata={"source": url})]


# Ingestion functions


def ingest_documents(docs: List[Document], source_label: str) -> int:
    chunks = get_splitter().split_documents(docs)

    for chunk in chunks:
        chunk.metadata.setdefault("source", source_label)
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    get_vector_store().add_documents(chunks)
    return len(chunks)


def ingest_file(file_path: str) -> int:
    path = Path(file_path)
    docs = (
        load_pdf(file_path) if path.suffix.lower() == ".pdf" else load_text(file_path)
    )
    return ingest_documents(docs, source_label=path.name)


def ingest_url(url: str) -> int:
    return ingest_documents(load_url(url), source_label=url)


# Retrieval functions
def retrieve(query: str, k: int | None = None) -> List[Document]:
    return get_vector_store().similarity_search(query, k=k or settings.retrieval_k)


def get_collection_stats() -> dict:
    store = get_vector_store()
    return {
        "total_chunks": store._collection.count(),
        "collection": settings.chroma_collection_name,
    }
