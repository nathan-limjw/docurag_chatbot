import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.database.store import (
    get_collection_stats,
    ingest_file,
    ingest_url,
    reset_vector_store,
)
from src.graph.pipeline import run_pipeline
from src.schemas.llm_output import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    IngestURLRequest,
    SourceReference,
    StatusResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Document / Website Chatbot API",
    description="RAG Chatbot",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/status", response_model=StatusResponse)
def get_status():
    stats = get_collection_stats()

    logger.info(f"""
    Status Check:
    - total_chunks = {stats["total_chunks"]}
    - collection = {stats["collection"]}
    """)
    return StatusResponse(
        status="ok",
        total_chunks=stats["total_chunks"],
        collection=stats["collection"],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    logger.info(f"""
    Chat request received
    - thread_id = {request.thread_id}
    - query = '{request.query[:100]}'
    """)

    if get_collection_stats()["total_chunks"] == 0:
        logger.warning("Chat attempted with no documents ingested yet")
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. Upload a file or ingest a URL first.",
        )

    try:
        result = run_pipeline(
            query=request.query,
            thread_id=request.thread_id,
        )

        logger.debug(f"""
        Chat Results:
        - grounded = {result["is_grounded"]}
        - retries = {result["retry_count"]}
        - sources = {len(result["sources"])}
        """)
    except Exception as e:
        logger.exception("Chat generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Chat generation failed: {e}",
        )

    return ChatResponse(
        answer=result["answer"],
        sources=[SourceReference(**s) for s in result["sources"]],
        retry_count=result["retry_count"],
        is_grounded=result["is_grounded"],
        thread_id=request.thread_id,
    )


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file_endpoint(file: UploadFile = File(...)):
    logger.info(f"File upload received: {file.filename}")

    allowed = {".pdf", ".txt", ".md"}

    original_filename = os.path.basename(file.filename)
    ext = os.path.splitext(original_filename)[1].lower()

    if ext not in allowed:
        logger.warning("Rejected unsupported file type")
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunks = ingest_file(tmp_path, source_label=original_filename)

        logger.info("File ingested successfully!")
        logger.debug(f"""
        File stats:
        - filename = {original_filename}
        - chunks = {chunks}
        """)
    except Exception as e:
        logger.exception(f"File {original_filename} ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    finally:
        os.unlink(tmp_path)

    return IngestResponse(
        message=f"Successfully ingested '{original_filename}'",
        chunks_added=chunks,
        source=original_filename,
    )


@app.post("/ingest/url", response_model=IngestResponse)
def ingest_url_endpoint(request: IngestURLRequest):
    logger.info(f"URL upload received: {request.url}")
    try:
        chunks = ingest_url(request.url)

        logger.info("URL ingested successfully!")
        logger.debug(f"""
        URL stats:
        - url = {request.url}
        - chunks = {chunks}
        """)

    except Exception as e:
        logger.exception(f"URL {request.url} ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"URL ingestion failed: {e}")

    return IngestResponse(
        message="Successfully ingested URL",
        chunks_added=chunks,
        source=request.url,
    )


@app.delete("/reset")
def reset_store():
    try:
        reset_vector_store()
        logger.info("Vector store reset successful")
    except Exception as e:
        logger.exception("Vector store reset failed")
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")

    return {"message": "Vector store reset successfully."}
