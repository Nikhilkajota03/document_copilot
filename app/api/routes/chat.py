import json
import asyncio
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import AskRequest, AskResponse
from app.services.chat_service import get_rag_chain_with_sources, stream_rag_answer

router = APIRouter()

# Lazy-init the chain once at first request
_chain_with_sources = None


def _get_chain():
    global _chain_with_sources
    if _chain_with_sources is None:
        _chain_with_sources = get_rag_chain_with_sources()
    return _chain_with_sources


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Ask a question about the ingested document.
    Returns the answer and the source chunks used for generation.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        chain = _get_chain()
        result = chain(request.question)
        return AskResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream(request: AskRequest):
    """
    Stream the answer token-by-token using Server-Sent Events (SSE).
    Clients should read the response as an event-stream.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def event_generator():
        try:
            async for token in stream_rag_answer(request.question):
                # SSE format: data: <payload>\n\n
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
