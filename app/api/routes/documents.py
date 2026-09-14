from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from app.services.document_service import save_upload, ingest_document, get_store_status

router = APIRouter()

# Track ongoing ingestion status in-memory (good enough for a single-user dev server)
_ingestion_status: dict = {"running": False, "last_result": None, "last_error": None}


def _run_ingestion_bg(file_path: str):
    global _ingestion_status
    try:
        result = ingest_document(file_path)
        _ingestion_status["last_result"] = result
        _ingestion_status["last_error"] = None
    except Exception as e:
        _ingestion_status["last_error"] = str(e)
        _ingestion_status["last_result"] = None
    finally:
        _ingestion_status["running"] = False


@router.post("/ingest")
async def ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF and trigger the ingestion pipeline in the background.
    Poll /api/documents/status to check when it's ready.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    if _ingestion_status["running"]:
        raise HTTPException(status_code=409, detail="An ingestion is already in progress.")

    file_bytes = await file.read()
    file_path = save_upload(file_bytes, file.filename)

    _ingestion_status["running"] = True
    _ingestion_status["last_result"] = None
    _ingestion_status["last_error"] = None

    background_tasks.add_task(_run_ingestion_bg, file_path)

    return {
        "message": f"Ingestion started for '{file.filename}'.",
        "file_path": file_path,
    }


@router.get("/status")
def status():
    """
    Returns the readiness of the vector store + ingestion job status.
    """
    store = get_store_status()
    return {
        "store": store,
        "ingestion": {
            "running": _ingestion_status["running"],
            "last_result": _ingestion_status["last_result"],
            "last_error": _ingestion_status["last_error"],
        },
    }
