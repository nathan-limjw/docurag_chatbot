import ipaddress
import socket
import uuid
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import settings


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


def reset_vector_store() -> None:
    store = get_vector_store()
    store.delete_collection()


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


def validate_public_url(url: str) -> None:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http and https URLs are supported.")

    if not parsed.hostname:
        raise ValueError("URL must include a valid hostname.")

    hostname = parsed.hostname.lower()

    if hostname in {"localhost", "127.0.0.1", "0.0.0.0"}:
        raise ValueError("Localhost URLs are not allowed.")

    try:
        addresses = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise ValueError("Could not resolve URL hostname.")

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError("Private or local network URLs are not allowed.")


def load_url(url: str) -> List[Document]:
    validate_public_url(url)

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
        chunk.metadata["source"] = source_label
        chunk.metadata["chunk_id"] = str(uuid.uuid4())

    get_vector_store().add_documents(chunks)
    return len(chunks)


def ingest_file(file_path: str, source_label: str | None = None) -> int:
    path = Path(file_path)
    docs = (
        load_pdf(file_path) if path.suffix.lower() == ".pdf" else load_text(file_path)
    )
    return ingest_documents(docs, source_label=source_label or path.name)


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
