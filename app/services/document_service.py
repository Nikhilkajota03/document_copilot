import os
import shutil
from app.ingestion.pipeline import run_ingestion_pipeline
from app.retrieval.vector_store import get_vector_store

UPLOAD_DIR = "data/uploads"
CHROMA_PATH = "data/chroma_db"
DOC_STORE_PATH = "data/doc_store"
BM25_PATH = "data/bm25_index.pkl"


def save_upload(file_bytes: bytes, filename: str) -> str:
    """Saves uploaded file bytes to the uploads directory and returns the path."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, filename)
    with open(dest, "wb") as f:
        f.write(file_bytes)
    return dest


def ingest_document(file_path: str) -> dict:
    """
    Runs the full ingestion pipeline on a given PDF path.
    Returns a summary dict with chunk count.
    """
    result = run_ingestion_pipeline(file_path)
    return result


def get_store_status() -> dict:
    """
    Returns whether the vector store and BM25 index exist and are ready.
    """
    chroma_ready = os.path.isdir(CHROMA_PATH) and bool(os.listdir(CHROMA_PATH))
    bm25_ready = os.path.isfile(BM25_PATH)
    doc_store_ready = os.path.isdir(DOC_STORE_PATH) and bool(os.listdir(DOC_STORE_PATH))

    return {
        "ready": chroma_ready and bm25_ready and doc_store_ready,
        "chroma_db": chroma_ready,
        "bm25_index": bm25_ready,
        "doc_store": doc_store_ready,
    }
